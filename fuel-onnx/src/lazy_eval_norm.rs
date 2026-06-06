//! Lazy-graph ONNX evaluator — sub-port 3 of `port-onnx-eval.md`.
//!
//! Adds norm + activation + softmax op coverage on top of the
//! sub-port-1 dispatcher in [`lazy_eval`](super::lazy_eval).
//!
//! Op set covered by this sub-port:
//!   - `BatchNormalization` (inference / eval mode only).
//!   - `LayerNormalization` (axis + epsilon, no scale-only).
//!   - `Softmax` (axis-aware, ONNX v13+ semantics).
//!   - `LogSoftmax` (axis-aware).
//!   - `Relu`, `Gelu`, `Sigmoid`, `Tanh`, `LeakyRelu`, `Elu`,
//!     `HardSigmoid`.
//!
//! The entrypoint is [`try_dispatch`]: called by `lazy_eval`'s fallthrough
//! arm, returns `Ok(true)` if it handled the op, `Ok(false)` otherwise.

use crate::lazy_eval::{
    get_attr_float_opt, get_attr_int_opt, normalize_axis, set_output,
};
use crate::onnx;
use fuel::lazy::LazyTensor;
use fuel::{Device, Error, Result, Shape};
use std::collections::HashMap;

/// Try to dispatch `node` to a sub-port-3 handler. Returns `Ok(true)`
/// iff the op was recognized and handled; `Ok(false)` lets the caller
/// continue its fallthrough to the unsupported-op error.
pub fn try_dispatch(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
    device: &Device,
    anchor: &mut Option<LazyTensor>,
) -> Result<bool> {
    match node.op_type.as_str() {
        "BatchNormalization" => batch_normalization(node, values)?,
        "LayerNormalization" => layer_normalization(node, values, device, anchor)?,
        "Softmax" => softmax_op(node, values, /*log*/ false)?,
        "LogSoftmax" => softmax_op(node, values, /*log*/ true)?,
        "Relu" => unary_handler(node, values, |x| Ok(x.relu()))?,
        "Gelu" => unary_handler(node, values, |x| Ok(x.gelu()))?,
        "Sigmoid" => unary_handler(node, values, |x| Ok(x.sigmoid()))?,
        "Tanh" => unary_handler(node, values, |x| Ok(x.tanh()))?,
        "LeakyRelu" => leaky_relu(node, values)?,
        "Elu" => elu(node, values)?,
        "HardSigmoid" => hard_sigmoid(node, values)?,
        _ => return Ok(false),
    }
    Ok(true)
}

fn unary_handler(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
    f: impl FnOnce(&LazyTensor) -> Result<LazyTensor>,
) -> Result<()> {
    let x = get(values, &node.input[0], node)?;
    let y = f(&x)?;
    set_output(node, 0, y, values)
}

fn get(
    values: &HashMap<String, LazyTensor>,
    name: &str,
    node: &onnx::NodeProto,
) -> Result<LazyTensor> {
    values
        .get(name)
        .cloned()
        .ok_or_else(|| {
            Error::Msg(format!(
                "missing input '{}' for node '{}' ({})",
                name, node.name, node.op_type
            ))
            .bt()
        })
}

// ---------------------------------------------------------------------------
// BatchNormalization (eval / inference mode)
// ---------------------------------------------------------------------------

fn batch_normalization(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
) -> Result<()> {
    let training_mode = get_attr_int_opt(node, "training_mode").unwrap_or(0);
    if training_mode != 0 {
        return Err(Error::Msg(format!(
            "BatchNormalization node '{}': training_mode=1 not supported (eval mode only)",
            node.name
        ))
        .bt());
    }
    if node.input.len() < 5 {
        return Err(Error::Msg(format!(
            "BatchNormalization node '{}': expected 5 inputs (X, scale, B, mean, var), got {}",
            node.name,
            node.input.len(),
        ))
        .bt());
    }
    let eps = get_attr_float_opt(node, "epsilon").unwrap_or(1e-5) as f64;

    let x = get(values, &node.input[0], node)?;
    let scale = get(values, &node.input[1], node)?;
    let bias = get(values, &node.input[2], node)?;
    let mean = get(values, &node.input[3], node)?;
    let var = get(values, &node.input[4], node)?;

    // Channel axis is dim 1; broadcast shape is [1, C, 1, ..., 1].
    let dims = x.shape().dims().to_vec();
    if dims.len() < 2 {
        return Err(Error::Msg(format!(
            "BatchNormalization node '{}': input rank {} < 2",
            node.name,
            dims.len()
        ))
        .bt());
    }
    let mut bc: Vec<usize> = vec![1; dims.len()];
    bc[1] = dims[1];
    let bc_shape = Shape::from_dims(&bc);

    let mean_b = mean.reshape(bc_shape.clone())?;
    let var_b = var.reshape(bc_shape.clone())?;
    let scale_b = scale.reshape(bc_shape.clone())?;
    let bias_b = bias.reshape(bc_shape)?;

    // y = (x - mean) / sqrt(var + eps) * scale + bias
    let centered = x.broadcast_sub(&mean_b)?;
    let denom = var_b.add_scalar(eps).sqrt();
    let normed = centered.broadcast_div(&denom)?;
    let scaled = normed.broadcast_mul(&scale_b)?;
    let y = scaled.broadcast_add(&bias_b)?;
    set_output(node, 0, y, values)
}

// ---------------------------------------------------------------------------
// LayerNormalization
// ---------------------------------------------------------------------------

fn layer_normalization(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
    _device: &Device,
    _anchor: &mut Option<LazyTensor>,
) -> Result<()> {
    if node.input.len() < 2 {
        return Err(Error::Msg(format!(
            "LayerNormalization node '{}': expected at least 2 inputs (X, scale), got {}",
            node.name,
            node.input.len(),
        ))
        .bt());
    }
    let eps = get_attr_float_opt(node, "epsilon").unwrap_or(1e-5) as f64;
    let axis_raw = get_attr_int_opt(node, "axis").unwrap_or(-1);

    let x = get(values, &node.input[0], node)?;
    let scale = get(values, &node.input[1], node)?;
    let bias_opt = if node.input.len() > 2 && !node.input[2].is_empty() {
        Some(get(values, &node.input[2], node)?)
    } else {
        None
    };

    let rank = x.rank();
    let axis = normalize_axis(axis_raw, rank)?;

    // Eager LayerNorm in fuel normalizes over the *last* dim only, so
    // we run mean/var reductions ourselves to honor an arbitrary axis.
    // keepdim keeps the broadcast shape stable across the loop.
    let dims = x.shape().dims().to_vec();
    let mut acc_mean = x.clone();
    for d in axis..rank {
        acc_mean = acc_mean.mean_keepdim(d)?;
    }
    let mean_bc = acc_mean.broadcast_to(Shape::from_dims(&dims))?;

    let centered = x.broadcast_sub(&mean_bc)?;
    let sq = centered.mul(&centered)?;
    let mut acc_var = sq;
    for d in axis..rank {
        acc_var = acc_var.mean_keepdim(d)?;
    }
    let var_bc = acc_var.broadcast_to(Shape::from_dims(&dims))?;
    let denom = var_bc.add_scalar(eps).sqrt();
    let normed = centered.broadcast_div(&denom)?;

    // Affine: scale and bias have shape == suffix of x starting at axis.
    let mut affine_shape: Vec<usize> = vec![1; rank];
    for (i, d) in dims.iter().enumerate().skip(axis) {
        affine_shape[i] = *d;
    }
    let scale_bc = scale
        .reshape(Shape::from_dims(&affine_shape))?
        .broadcast_to(Shape::from_dims(&dims))?;
    let scaled = normed.broadcast_mul(&scale_bc)?;
    let y = match bias_opt {
        None => scaled,
        Some(bias) => {
            let bias_bc = bias
                .reshape(Shape::from_dims(&affine_shape))?
                .broadcast_to(Shape::from_dims(&dims))?;
            scaled.broadcast_add(&bias_bc)?
        }
    };
    set_output(node, 0, y, values)
}

// ---------------------------------------------------------------------------
// Softmax / LogSoftmax (axis-aware)
// ---------------------------------------------------------------------------

fn softmax_op(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
    log: bool,
) -> Result<()> {
    let x = get(values, &node.input[0], node)?;
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::Msg(format!(
            "Softmax node '{}': scalar input not supported",
            node.name
        ))
        .bt());
    }
    // ONNX v13+: default axis is -1 (last dim). The eager handler in
    // `eval.rs` uses `softmax_last_dim` when the attribute is absent,
    // so we mirror that.
    let axis_raw = get_attr_int_opt(node, "axis").unwrap_or(-1);
    let axis = normalize_axis(axis_raw, rank)?;
    let last = rank - 1;

    let needs_transpose = axis != last;
    let mut perm: Vec<usize> = (0..rank).collect();
    if needs_transpose {
        perm.swap(axis, last);
    }
    let permuted = if needs_transpose {
        x.permute(perm.as_slice())?
    } else {
        x
    };
    let softed = if log {
        permuted.log_softmax_last_dim()?
    } else {
        permuted.softmax_last_dim()?
    };
    let out = if needs_transpose {
        // The swap permutation is self-inverse.
        softed.permute(perm.as_slice())?
    } else {
        softed
    };
    set_output(node, 0, out, values)
}

// ---------------------------------------------------------------------------
// Parametric activations: LeakyRelu, Elu, HardSigmoid
// ---------------------------------------------------------------------------

fn leaky_relu(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
) -> Result<()> {
    // y = relu(x) + alpha * (x - relu(x))
    //   x >= 0 → relu(x) = x, x - relu(x) = 0 → y = x
    //   x <  0 → relu(x) = 0, x - relu(x) = x → y = alpha*x
    let x = get(values, &node.input[0], node)?;
    let alpha = get_attr_float_opt(node, "alpha").unwrap_or(0.01) as f64;
    let pos = x.relu();
    let neg_part = x.sub(&pos)?.mul_scalar(alpha);
    let y = pos.add(&neg_part)?;
    set_output(node, 0, y, values)
}

fn elu(node: &onnx::NodeProto, values: &mut HashMap<String, LazyTensor>) -> Result<()> {
    // y = relu(x) + alpha * (exp(min(x, 0)) - 1)
    //   x >= 0 → relu(x) = x, min(x,0) = 0, exp(0)-1 = 0 → y = x
    //   x <  0 → relu(x) = 0, min(x,0) = x, exp(x)-1   → y = alpha*(exp(x)-1)
    let x = get(values, &node.input[0], node)?;
    let alpha = get_attr_float_opt(node, "alpha").unwrap_or(1.0) as f64;
    let pos = x.relu();
    // min(x, 0) = -relu(-x)
    let neg_clamped = x.neg().relu().neg();
    let expm1 = neg_clamped.exp().add_scalar(-1.0);
    let scaled = expm1.mul_scalar(alpha);
    let y = pos.add(&scaled)?;
    set_output(node, 0, y, values)
}

fn hard_sigmoid(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
) -> Result<()> {
    // y = clamp(alpha*x + beta, 0, 1)
    let x = get(values, &node.input[0], node)?;
    let alpha = get_attr_float_opt(node, "alpha").unwrap_or(0.2) as f64;
    let beta = get_attr_float_opt(node, "beta").unwrap_or(0.5) as f64;
    let y = x.mul_scalar(alpha).add_scalar(beta).clamp(0.0, 1.0);
    set_output(node, 0, y, values)
}

// ===========================================================================
// tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy_eval::LazyOnnxEval;
    use crate::onnx::{attribute_proto::AttributeType, tensor_proto::DataType};
    use prost::Message;
    use std::sync::Arc;

    // ---- proto helpers (mirror lazy_eval tests) ----

    fn tp_float(name: &str, dims: &[i64], data: Vec<f32>) -> onnx::TensorProto {
        onnx::TensorProto {
            dims: dims.to_vec(),
            data_type: DataType::Float as i32,
            float_data: data,
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn attr_int(name: &str, v: i64) -> onnx::AttributeProto {
        onnx::AttributeProto {
            name: name.to_string(),
            r#type: AttributeType::Int as i32,
            i: v,
            ..Default::default()
        }
    }

    fn attr_float(name: &str, v: f32) -> onnx::AttributeProto {
        onnx::AttributeProto {
            name: name.to_string(),
            r#type: AttributeType::Float as i32,
            f: v,
            ..Default::default()
        }
    }

    fn value_info(name: &str) -> onnx::ValueInfoProto {
        onnx::ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn node_(op_type: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> onnx::NodeProto {
        onnx::NodeProto {
            op_type: op_type.to_string(),
            name: name.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn model_from_graph(graph: onnx::GraphProto) -> onnx::ModelProto {
        onnx::ModelProto {
            ir_version: 7,
            graph: Some(graph),
            ..Default::default()
        }
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn run_unary(op_type: &str, attrs: Vec<onnx::AttributeProto>, x_data: Vec<f32>, x_dims: &[usize]) -> Vec<f32> {
        let mut n = node_(op_type, "n", &["X"], &["Y"]);
        n.attribute = attrs;
        let graph = onnx::GraphProto {
            node: vec![n],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let mut buf = Vec::new();
        model_from_graph(graph).encode(&mut buf).unwrap();
        let evaluator = LazyOnnxEval::from_bytes(&buf).unwrap();

        let device = Device::cpu();
        let dims_i64: Vec<usize> = x_dims.to_vec();
        let x = LazyTensor::from_f32(
            Arc::<[f32]>::from(x_data),
            Shape::from_dims(&dims_i64),
            &device,
        );
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let outputs = evaluator.run(&inputs).unwrap();
        outputs.get("Y").unwrap().realize_f32().to_vec()
    }

    // ---- BatchNormalization ----

    #[test]
    fn batchnorm_eval_mode_matches_hand_computed() {
        // X: [N=1, C=2, H=2, W=1] = [[ [[1],[2]], [[3],[4]] ]]
        //   channel 0 values: 1, 2  → values laid out (N,C,H,W)
        //   channel 1 values: 3, 4
        // scale = [2.0, 0.5], bias = [1.0, -1.0]
        // running_mean = [1.5, 3.5], running_var = [0.25, 0.25]
        // eps = 1e-5 (default)
        //
        // For each (n, c, h, w):
        //   y = (x - mean[c]) / sqrt(var[c] + eps) * scale[c] + bias[c]
        // channel 0 denom ≈ sqrt(0.25001) ≈ 0.5
        //   (1 - 1.5)/0.5 * 2 + 1 = -2 + 1 = -1
        //   (2 - 1.5)/0.5 * 2 + 1 =  2 + 1 =  3
        // channel 1 denom ≈ 0.5
        //   (3 - 3.5)/0.5 * 0.5 + (-1) = -0.5 -1 = -1.5
        //   (4 - 3.5)/0.5 * 0.5 + (-1) =  0.5 -1 = -0.5
        let bn = node_("BatchNormalization", "bn", &["X", "S", "B", "M", "V"], &["Y"]);
        let graph = onnx::GraphProto {
            node: vec![bn],
            initializer: vec![
                tp_float("S", &[2], vec![2.0, 0.5]),
                tp_float("B", &[2], vec![1.0, -1.0]),
                tp_float("M", &[2], vec![1.5, 3.5]),
                tp_float("V", &[2], vec![0.25, 0.25]),
            ],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let mut buf = Vec::new();
        model_from_graph(graph).encode(&mut buf).unwrap();
        let evaluator = LazyOnnxEval::from_bytes(&buf).unwrap();

        let device = Device::cpu();
        let x = LazyTensor::from_f32(
            Arc::<[f32]>::from(vec![1.0_f32, 2.0, 3.0, 4.0]),
            Shape::from_dims(&[1, 2, 2, 1]),
            &device,
        );
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let out = evaluator.run(&inputs).unwrap();
        let y = out.get("Y").unwrap();
        assert_eq!(y.shape().dims(), &[1, 2, 2, 1]);
        let got = y.realize_f32();
        let expected = [-1.0_f32, 3.0, -1.5, -0.5];
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(*g, *e, 1e-3),
                "BN mismatch at {i}: got {g} expected {e} (full got={got:?})"
            );
        }
        assert!(got.iter().all(|v| v.is_finite()));
    }

    // ---- LayerNormalization ----

    #[test]
    fn layernorm_normalizes_along_axis() {
        // X: [2, 3] — normalize along the last dim (axis=-1).
        // scale = [1, 1, 1], bias = [0, 0, 0] → pure normalization.
        // row0 = [1, 2, 3]: mean=2, var = ((1-2)^2+(2-2)^2+(3-2)^2)/3 = 2/3
        //   normed = (x - 2) / sqrt(2/3 + 1e-5)
        // row1 = [4, 5, 6]: same shape after centering.
        let mut ln = node_("LayerNormalization", "ln", &["X", "S", "B"], &["Y"]);
        ln.attribute.push(attr_int("axis", -1));
        ln.attribute.push(attr_float("epsilon", 1e-5));
        let graph = onnx::GraphProto {
            node: vec![ln],
            initializer: vec![
                tp_float("S", &[3], vec![1.0, 1.0, 1.0]),
                tp_float("B", &[3], vec![0.0, 0.0, 0.0]),
            ],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let mut buf = Vec::new();
        model_from_graph(graph).encode(&mut buf).unwrap();
        let evaluator = LazyOnnxEval::from_bytes(&buf).unwrap();

        let device = Device::cpu();
        let x = LazyTensor::from_f32(
            Arc::<[f32]>::from(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Shape::from_dims(&[2, 3]),
            &device,
        );
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let out = evaluator.run(&inputs).unwrap();
        let y = out.get("Y").unwrap();
        assert_eq!(y.shape().dims(), &[2, 3]);
        let got = y.realize_f32();

        // Each row's mean should be ~0 and variance ~1 (population variance,
        // which is what BN/LN use).
        for row in 0..2 {
            let r = &got[row * 3..(row + 1) * 3];
            let mean: f32 = r.iter().sum::<f32>() / 3.0;
            assert!(mean.abs() < 1e-4, "row{row} mean = {mean} (want ~0)");
            let var: f32 = r.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 3.0;
            assert!(
                (var - 1.0).abs() < 5e-3,
                "row{row} variance = {var} (want ~1)"
            );
        }
        assert!(got.iter().all(|v| v.is_finite()));
    }

    // ---- Softmax ----

    #[test]
    fn softmax_axis_minus_one_sums_to_one_per_row() {
        // X: [2, 3], axis = -1.
        let mut sm = node_("Softmax", "sm", &["X"], &["Y"]);
        sm.attribute.push(attr_int("axis", -1));
        let graph = onnx::GraphProto {
            node: vec![sm],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let mut buf = Vec::new();
        model_from_graph(graph).encode(&mut buf).unwrap();
        let evaluator = LazyOnnxEval::from_bytes(&buf).unwrap();

        let device = Device::cpu();
        let x = LazyTensor::from_f32(
            Arc::<[f32]>::from(vec![1.0_f32, 2.0, 3.0, 1.0, 1.0, 1.0]),
            Shape::from_dims(&[2, 3]),
            &device,
        );
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let out = evaluator.run(&inputs).unwrap();
        let y = out.get("Y").unwrap();
        assert_eq!(y.shape().dims(), &[2, 3]);
        let got = y.realize_f32();

        for row in 0..2 {
            let s: f32 = got[row * 3..(row + 1) * 3].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row{row} sum = {s}");
        }
        // Uniform row should produce 1/3 everywhere.
        for j in 0..3 {
            assert!(
                approx_eq(got[3 + j], 1.0 / 3.0, 1e-6),
                "uniform row got[{}]={}",
                3 + j,
                got[3 + j]
            );
        }
        // Monotonic row: softmax preserves order.
        assert!(got[0] < got[1] && got[1] < got[2]);
        assert!(got.iter().all(|v| v.is_finite()));
    }

    // ---- Plain activations ----

    #[test]
    fn relu_matches_lazy_tensor_relu_directly() {
        let xs = vec![-2.0_f32, -0.5, 0.0, 0.25, 3.0];
        let got = run_unary("Relu", vec![], xs.clone(), &[xs.len()]);

        let device = Device::cpu();
        let xt = LazyTensor::from_f32(
            Arc::<[f32]>::from(xs.clone()),
            Shape::from_dims(&[xs.len()]),
            &device,
        );
        let expected = xt.relu().realize_f32();
        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(*g, *e, 1e-7),
                "Relu mismatch at {i}: got {g} expected {e}"
            );
        }
    }

    #[test]
    fn gelu_matches_lazy_tensor_gelu_directly() {
        let xs = vec![-2.0_f32, -0.5, 0.0, 0.25, 3.0];
        let got = run_unary("Gelu", vec![], xs.clone(), &[xs.len()]);

        let device = Device::cpu();
        let xt = LazyTensor::from_f32(
            Arc::<[f32]>::from(xs.clone()),
            Shape::from_dims(&[xs.len()]),
            &device,
        );
        let expected = xt.gelu().realize_f32();
        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(*g, *e, 1e-5),
                "Gelu mismatch at {i}: got {g} expected {e}"
            );
        }
    }

    #[test]
    fn sigmoid_at_zero_is_half() {
        let xs = vec![0.0_f32, 1.0, -1.0];
        let got = run_unary("Sigmoid", vec![], xs.clone(), &[xs.len()]);
        assert!(approx_eq(got[0], 0.5, 1e-6), "sigmoid(0) = {}", got[0]);
        // Sanity: sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269; together sum to 1.
        assert!((got[1] + got[2] - 1.0).abs() < 1e-5);
        assert!(got.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tanh_at_zero_is_zero() {
        let xs = vec![0.0_f32, 1.0, -1.0];
        let got = run_unary("Tanh", vec![], xs.clone(), &[xs.len()]);
        assert!(approx_eq(got[0], 0.0, 1e-6), "tanh(0) = {}", got[0]);
        // tanh is odd: tanh(-x) = -tanh(x).
        assert!(approx_eq(got[1], -got[2], 1e-5));
        assert!(got.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn leaky_relu_with_slope_0_1_at_minus_2_is_minus_0_2() {
        let xs = vec![-2.0_f32, -0.5, 0.0, 0.25, 3.0];
        let attrs = vec![attr_float("alpha", 0.1)];
        let got = run_unary("LeakyRelu", attrs, xs.clone(), &[xs.len()]);
        let expected = [-0.2_f32, -0.05, 0.0, 0.25, 3.0];
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(*g, *e, 1e-6),
                "LeakyRelu mismatch at {i}: got {g} expected {e}"
            );
        }
        assert!(got.iter().all(|v| v.is_finite()));
    }
}

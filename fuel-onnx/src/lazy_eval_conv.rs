//! Lazy-graph ONNX evaluator — sub-port 2 of `port-onnx-eval.md`.
//!
//! Extends [`crate::lazy_eval::LazyOnnxEval`] with convolution-family
//! ops:
//!   - `Conv` (1D / 2D — `auto_pad` SAME_UPPER / SAME_LOWER / VALID /
//!     NOTSET, explicit `pads`, `strides`, `group`; dilations rejected
//!     unless == 1).
//!   - `ConvTranspose` (1D / 2D — same `auto_pad` logic).
//!   - `Pad` (constant mode only; reflect / edge / wrap surface as
//!     typed errors pointing at a future sub-port).
//!   - `MaxPool` (2D) / `AveragePool` (2D) /
//!     `GlobalAveragePool` / `GlobalMaxPool`.
//!
//! Hooked into the existing [`crate::lazy_eval`] dispatch as a
//! fallthrough — see [`try_dispatch_node`] which returns `Ok(true)`
//! when this module claimed the op and `Ok(false)` to let the caller
//! continue its own classification (e.g., emit the sub-port-3 error).

use crate::lazy_eval::{
    get_attr_int_opt, get_attr_ints_opt, get_attr_string_opt, get_i64_vec, set_output,
};
use crate::onnx;
use fuel::lazy::LazyTensor;
use fuel::{Device, Error, Result, Shape};
use std::collections::HashMap;

/// Sub-port-2 dispatch entry-point. Returns `Ok(true)` if `node.op_type`
/// is handled here (and the result has been stored into `values`),
/// `Ok(false)` if the caller should keep searching. Errors propagate as
/// typed [`Error::Msg`].
pub fn try_dispatch_node(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
    device: &Device,
    anchor: &mut Option<LazyTensor>,
    i64_cache: &mut HashMap<String, Vec<i64>>,
) -> Result<bool> {
    match node.op_type.as_str() {
        "Conv" => {
            conv_op(node, values)?;
            Ok(true)
        }
        "ConvTranspose" => {
            conv_transpose_op(node, values)?;
            Ok(true)
        }
        "Pad" => {
            pad_op(node, values, device, anchor, i64_cache)?;
            Ok(true)
        }
        "MaxPool" => {
            max_pool_op(node, values)?;
            Ok(true)
        }
        "AveragePool" => {
            avg_pool_op(node, values)?;
            Ok(true)
        }
        "GlobalAveragePool" => {
            global_avg_pool_op(node, values)?;
            Ok(true)
        }
        "GlobalMaxPool" => {
            global_max_pool_op(node, values)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[derive(Clone, Copy)]
enum AutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

fn parse_auto_pad(node: &onnx::NodeProto) -> Result<AutoPad> {
    let s = get_attr_string_opt(node, "auto_pad")?;
    match s.as_deref() {
        None | Some("") | Some("NOTSET") => Ok(AutoPad::NotSet),
        Some("SAME_UPPER") => Ok(AutoPad::SameUpper),
        Some("SAME_LOWER") => Ok(AutoPad::SameLower),
        Some("VALID") => Ok(AutoPad::Valid),
        Some(other) => Err(Error::Msg(format!(
            "node '{}': unsupported auto_pad '{}'",
            node.name, other
        ))
        .bt()),
    }
}

fn get(values: &HashMap<String, LazyTensor>, node: &onnx::NodeProto, name: &str) -> Result<LazyTensor> {
    values
        .get(name)
        .cloned()
        .ok_or_else(|| Error::Msg(format!("missing input '{}' for node '{}'", name, node.name)).bt())
}

fn check_dilations(node: &onnx::NodeProto) -> Result<()> {
    if let Some(d) = get_attr_ints_opt(node, "dilations") {
        if d.iter().any(|&v| v != 1) {
            return Err(Error::Msg(format!(
                "node '{}' ({}): dilations {:?} != 1 are not supported in sub-port 2",
                node.name, node.op_type, d
            ))
            .bt());
        }
    }
    Ok(())
}

/// Compute symmetric/explicit padding (pre, post) per spatial axis from
/// ONNX semantics. Returns one `(pre, post)` per spatial axis (so
/// length-2 for 2D, length-1 for 1D).
fn resolve_pads(
    node: &onnx::NodeProto,
    auto_pad: AutoPad,
    input_spatial: &[usize],
    kernel_spatial: &[usize],
    strides: &[usize],
) -> Result<Vec<(usize, usize)>> {
    let n_spatial = input_spatial.len();
    debug_assert_eq!(kernel_spatial.len(), n_spatial);
    debug_assert_eq!(strides.len(), n_spatial);

    match auto_pad {
        AutoPad::NotSet => {
            let pads = get_attr_ints_opt(node, "pads");
            match pads {
                None => Ok(vec![(0, 0); n_spatial]),
                Some(p) => {
                    if p.len() != 2 * n_spatial {
                        return Err(Error::Msg(format!(
                            "node '{}': pads len {} != 2 * spatial-rank {}",
                            node.name,
                            p.len(),
                            n_spatial
                        ))
                        .bt());
                    }
                    if p.iter().any(|&v| v < 0) {
                        return Err(Error::Msg(format!(
                            "node '{}': negative pads {:?} not supported in sub-port 2",
                            node.name, p
                        ))
                        .bt());
                    }
                    let mut out = Vec::with_capacity(n_spatial);
                    for i in 0..n_spatial {
                        out.push((p[i] as usize, p[i + n_spatial] as usize));
                    }
                    Ok(out)
                }
            }
        }
        AutoPad::Valid => Ok(vec![(0, 0); n_spatial]),
        AutoPad::SameUpper | AutoPad::SameLower => {
            let mut out = Vec::with_capacity(n_spatial);
            for i in 0..n_spatial {
                let in_size = input_spatial[i];
                let k = kernel_spatial[i];
                let s = strides[i];
                if s == 0 {
                    return Err(Error::Msg(format!(
                        "node '{}': stride must be >= 1, got 0",
                        node.name
                    ))
                    .bt());
                }
                let out_size = in_size.div_ceil(s);
                let total: usize = ((out_size - 1) * s + k).saturating_sub(in_size);
                let (pre, post) = match auto_pad {
                    AutoPad::SameUpper => (total / 2, total - total / 2),
                    AutoPad::SameLower => (total - total / 2, total / 2),
                    _ => unreachable!(),
                };
                out.push((pre, post));
            }
            Ok(out)
        }
    }
}

/// Apply asymmetric per-spatial-axis padding via `pad_with_zeros`, then
/// return the symmetric residual `(pre, post)` to feed into Conv as its
/// native `padding` argument. If `(pre, post)` is symmetric already
/// (`pre == post`) we skip the explicit pad and let Conv handle it.
fn apply_asymmetric_pads(
    mut x: LazyTensor,
    per_axis: &[(usize, usize)],
    spatial_start_dim: usize,
) -> Result<(LazyTensor, Vec<usize>)> {
    let mut symmetric = Vec::with_capacity(per_axis.len());
    for (i, &(pre, post)) in per_axis.iter().enumerate() {
        if pre == post {
            symmetric.push(pre);
        } else {
            x = x.pad_with_zeros(spatial_start_dim + i, pre, post)?;
            symmetric.push(0);
        }
    }
    Ok((x, symmetric))
}

fn conv_op(node: &onnx::NodeProto, values: &mut HashMap<String, LazyTensor>) -> Result<()> {
    check_dilations(node)?;
    let auto_pad = parse_auto_pad(node)?;
    let groups = get_attr_int_opt(node, "group").unwrap_or(1) as usize;
    let x = get(values, node, &node.input[0])?;
    let w = get(values, node, &node.input[1])?;
    let bias = if node.input.len() > 2 && !node.input[2].is_empty() {
        Some(get(values, node, &node.input[2])?)
    } else {
        None
    };
    let w_rank = w.rank();
    match w_rank {
        3 => {
            // Conv1D: x [N, Cin, T], w [Cout, Cin/g, K]
            let x_dims = x.shape().dims().to_vec();
            if x_dims.len() != 3 {
                return Err(Error::Msg(format!(
                    "Conv (1D) '{}': input must be rank 3, got {x_dims:?}",
                    node.name
                ))
                .bt());
            }
            let w_dims = w.shape().dims().to_vec();
            let stride = match get_attr_ints_opt(node, "strides") {
                None => 1,
                Some([s]) => *s as usize,
                Some(s) => {
                    return Err(Error::Msg(format!(
                        "Conv1D '{}': expected 1 stride, got {s:?}",
                        node.name
                    ))
                    .bt());
                }
            };
            let pads = resolve_pads(
                node,
                auto_pad,
                &[x_dims[2]],
                &[w_dims[2]],
                &[stride],
            )?;
            let (x_padded, sym) = apply_asymmetric_pads(x, &pads, 2)?;
            let y = x_padded.conv1d(&w, bias.as_ref(), stride, sym[0], groups)?;
            set_output(node, 0, y, values)?;
            Ok(())
        }
        4 => {
            // Conv2D: x [N, Cin, H, W], w [Cout, Cin/g, Kh, Kw]
            let x_dims = x.shape().dims().to_vec();
            if x_dims.len() != 4 {
                return Err(Error::Msg(format!(
                    "Conv (2D) '{}': input must be rank 4, got {x_dims:?}",
                    node.name
                ))
                .bt());
            }
            let w_dims = w.shape().dims().to_vec();
            let strides = match get_attr_ints_opt(node, "strides") {
                None => (1usize, 1usize),
                Some([sh, sw]) => (*sh as usize, *sw as usize),
                Some(s) => {
                    return Err(Error::Msg(format!(
                        "Conv2D '{}': expected 2 strides, got {s:?}",
                        node.name
                    ))
                    .bt());
                }
            };
            let pads = resolve_pads(
                node,
                auto_pad,
                &[x_dims[2], x_dims[3]],
                &[w_dims[2], w_dims[3]],
                &[strides.0, strides.1],
            )?;
            let (x_padded, sym) = apply_asymmetric_pads(x, &pads, 2)?;
            let y = x_padded.conv2d(
                &w,
                bias.as_ref(),
                strides,
                (sym[0], sym[1]),
                groups,
            )?;
            set_output(node, 0, y, values)?;
            Ok(())
        }
        r => Err(Error::Msg(format!(
            "Conv '{}': unsupported weight rank {r} (only 3 / 4 supported)",
            node.name
        ))
        .bt()),
    }
}

fn conv_transpose_op(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
) -> Result<()> {
    check_dilations(node)?;
    let auto_pad = parse_auto_pad(node)?;
    let groups = get_attr_int_opt(node, "group").unwrap_or(1) as usize;
    let x = get(values, node, &node.input[0])?;
    let w = get(values, node, &node.input[1])?;
    let bias = if node.input.len() > 2 && !node.input[2].is_empty() {
        Some(get(values, node, &node.input[2])?)
    } else {
        None
    };
    let w_rank = w.rank();
    match w_rank {
        3 => {
            let x_dims = x.shape().dims().to_vec();
            if x_dims.len() != 3 {
                return Err(Error::Msg(format!(
                    "ConvTranspose (1D) '{}': input must be rank 3, got {x_dims:?}",
                    node.name
                ))
                .bt());
            }
            let w_dims = w.shape().dims().to_vec();
            let stride = match get_attr_ints_opt(node, "strides") {
                None => 1usize,
                Some([s]) => *s as usize,
                Some(s) => {
                    return Err(Error::Msg(format!(
                        "ConvTranspose1D '{}': expected 1 stride, got {s:?}",
                        node.name
                    ))
                    .bt());
                }
            };
            let output_padding = get_attr_ints_opt(node, "output_padding")
                .and_then(|v| v.first().copied())
                .unwrap_or(0) as usize;
            // ConvTranspose auto_pad: SAME_* sizes the *output* to
            // `input * stride`. For NOTSET / VALID, pads come from the
            // explicit attribute (default 0).
            let pads = resolve_conv_transpose_pads(
                node,
                auto_pad,
                &[x_dims[2]],
                &[w_dims[2]],
                &[stride],
            )?;
            let p = symmetric_or_err(node, &pads)?;
            let mut y = x.conv_transpose1d(&w, stride, p[0], output_padding, 1, groups)?;
            if let Some(b) = bias {
                y = add_channel_bias(&y, &b, node)?;
            }
            set_output(node, 0, y, values)?;
            Ok(())
        }
        4 => {
            let x_dims = x.shape().dims().to_vec();
            if x_dims.len() != 4 {
                return Err(Error::Msg(format!(
                    "ConvTranspose (2D) '{}': input must be rank 4, got {x_dims:?}",
                    node.name
                ))
                .bt());
            }
            let w_dims = w.shape().dims().to_vec();
            let strides = match get_attr_ints_opt(node, "strides") {
                None => (1usize, 1usize),
                Some([sh, sw]) => (*sh as usize, *sw as usize),
                Some(s) => {
                    return Err(Error::Msg(format!(
                        "ConvTranspose2D '{}': expected 2 strides, got {s:?}",
                        node.name
                    ))
                    .bt());
                }
            };
            let output_padding = match get_attr_ints_opt(node, "output_padding") {
                None => (0usize, 0usize),
                Some([oh, ow]) => (*oh as usize, *ow as usize),
                Some(s) => {
                    return Err(Error::Msg(format!(
                        "ConvTranspose2D '{}': expected 2 output_padding, got {s:?}",
                        node.name
                    ))
                    .bt());
                }
            };
            let pads = resolve_conv_transpose_pads(
                node,
                auto_pad,
                &[x_dims[2], x_dims[3]],
                &[w_dims[2], w_dims[3]],
                &[strides.0, strides.1],
            )?;
            let p = symmetric_or_err(node, &pads)?;
            let mut y = x.conv_transpose2d(
                &w,
                strides,
                (p[0], p[1]),
                output_padding,
                (1, 1),
                groups,
            )?;
            if let Some(b) = bias {
                y = add_channel_bias(&y, &b, node)?;
            }
            set_output(node, 0, y, values)?;
            Ok(())
        }
        r => Err(Error::Msg(format!(
            "ConvTranspose '{}': unsupported weight rank {r} (only 3 / 4 supported)",
            node.name
        ))
        .bt()),
    }
}

fn resolve_conv_transpose_pads(
    node: &onnx::NodeProto,
    auto_pad: AutoPad,
    input_spatial: &[usize],
    kernel_spatial: &[usize],
    strides: &[usize],
) -> Result<Vec<(usize, usize)>> {
    let n_spatial = input_spatial.len();
    match auto_pad {
        AutoPad::NotSet | AutoPad::Valid => {
            let pads = get_attr_ints_opt(node, "pads");
            match pads {
                None => Ok(vec![(0, 0); n_spatial]),
                Some(p) => {
                    if p.len() != 2 * n_spatial {
                        return Err(Error::Msg(format!(
                            "ConvTranspose '{}': pads len {} != 2 * spatial-rank {}",
                            node.name,
                            p.len(),
                            n_spatial
                        ))
                        .bt());
                    }
                    if p.iter().any(|&v| v < 0) {
                        return Err(Error::Msg(format!(
                            "ConvTranspose '{}': negative pads {:?} not supported",
                            node.name, p
                        ))
                        .bt());
                    }
                    let mut out = Vec::with_capacity(n_spatial);
                    for i in 0..n_spatial {
                        out.push((p[i] as usize, p[i + n_spatial] as usize));
                    }
                    Ok(out)
                }
            }
        }
        AutoPad::SameUpper | AutoPad::SameLower => {
            // ConvTranspose SAME_*: target output = input * stride.
            // total_pad = stride * (in - 1) + kernel - in * stride
            //           = kernel - stride
            // Split SAME_UPPER (more on end) / SAME_LOWER (more on start).
            let mut out = Vec::with_capacity(n_spatial);
            for i in 0..n_spatial {
                let k = kernel_spatial[i] as isize;
                let s = strides[i] as isize;
                let total = (k - s).max(0) as usize;
                let (pre, post) = match auto_pad {
                    AutoPad::SameUpper => (total / 2, total - total / 2),
                    AutoPad::SameLower => (total - total / 2, total / 2),
                    _ => unreachable!(),
                };
                out.push((pre, post));
            }
            let _ = input_spatial;
            Ok(out)
        }
    }
}

fn symmetric_or_err(node: &onnx::NodeProto, per_axis: &[(usize, usize)]) -> Result<Vec<usize>> {
    let mut out = Vec::with_capacity(per_axis.len());
    for (i, &(pre, post)) in per_axis.iter().enumerate() {
        if pre != post {
            return Err(Error::Msg(format!(
                "ConvTranspose '{}': asymmetric pads ({pre}, {post}) on axis {i} \
                 are not supported (sub-port 2 only handles symmetric)",
                node.name
            ))
            .bt());
        }
        out.push(pre);
    }
    Ok(out)
}

fn add_channel_bias(
    y: &LazyTensor,
    bias: &LazyTensor,
    node: &onnx::NodeProto,
) -> Result<LazyTensor> {
    let y_rank = y.rank();
    if y_rank < 2 {
        return Err(Error::Msg(format!(
            "node '{}': bias add requires y rank >= 2, got {y_rank}",
            node.name
        ))
        .bt());
    }
    let cout = y.shape().dims()[1];
    let b_shape = bias.shape().dims().to_vec();
    if b_shape != [cout] {
        return Err(Error::Msg(format!(
            "node '{}': bias shape {b_shape:?} must equal [Cout={cout}]",
            node.name
        ))
        .bt());
    }
    let mut new_shape: Vec<usize> = vec![1; y_rank];
    new_shape[1] = cout;
    let b_reshaped = bias.reshape(Shape::from_dims(&new_shape))?;
    y.broadcast_add(&b_reshaped)
}

fn pad_op(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
    device: &Device,
    anchor: &mut Option<LazyTensor>,
    i64_cache: &mut HashMap<String, Vec<i64>>,
) -> Result<()> {
    // ONNX Pad: mode is an attribute; reflect / edge / wrap are
    // explicitly out of scope.
    let mode = get_attr_string_opt(node, "mode")?.unwrap_or_else(|| "constant".to_string());
    match mode.as_str() {
        "constant" => {}
        "reflect" | "edge" | "wrap" => {
            return Err(Error::Msg(format!(
                "Pad '{}': mode '{}' is not supported in sub-port 2; \
                 it is scheduled for a future sub-port",
                node.name, mode,
            ))
            .bt());
        }
        other => {
            return Err(Error::Msg(format!(
                "Pad '{}': unknown mode '{}'",
                node.name, other
            ))
            .bt());
        }
    }
    let data = get(values, node, &node.input[0])?;
    // Opset 11+: pads is the second input (i64). Opset <11 had it as
    // an attribute named "pads"; we handle that fallback too.
    let pads_raw: Vec<i64> = if node.input.len() > 1 && !node.input[1].is_empty() {
        get_i64_vec(values, i64_cache, &node.input[1])?
    } else if let Some(p) = get_attr_ints_opt(node, "pads") {
        p.to_vec()
    } else {
        return Err(Error::Msg(format!(
            "Pad '{}': missing 'pads' input/attribute",
            node.name
        ))
        .bt());
    };
    let rank = data.rank();
    if pads_raw.len() != 2 * rank {
        return Err(Error::Msg(format!(
            "Pad '{}': pads len {} != 2 * rank {}",
            node.name,
            pads_raw.len(),
            rank
        ))
        .bt());
    }
    if pads_raw.iter().any(|&v| v < 0) {
        return Err(Error::Msg(format!(
            "Pad '{}': negative pads {:?} (trim) not supported in sub-port 2",
            node.name, pads_raw
        ))
        .bt());
    }
    let (pads_pre, pads_post) = pads_raw.split_at(rank);
    // Optional constant_value: ONNX 11+ exposes it as input[2]; older
    // graphs used the "value" attribute (float). Sub-port 2 routes
    // everything through `pad_with_zeros`, so non-zero constants are
    // explicitly rejected (re-routes to a future sub-port).
    let value: f64 = if node.input.len() > 2 && !node.input[2].is_empty() {
        let t = values.get(&node.input[2]).cloned().ok_or_else(|| {
            Error::Msg(format!(
                "Pad '{}': constant_value input '{}' missing",
                node.name, node.input[2]
            ))
            .bt()
        })?;
        scalar_to_f64(&t)?
    } else {
        match node
            .attribute
            .iter()
            .find(|a| a.name == "value")
            .map(|a| a.f)
        {
            Some(v) => v as f64,
            None => 0.0,
        }
    };
    if value != 0.0 {
        return Err(Error::Msg(format!(
            "Pad '{}': non-zero constant_value {value} is not supported in \
             sub-port 2 (only zero-fill 'constant' mode is implemented)",
            node.name
        ))
        .bt());
    }
    let _ = (device, anchor);
    let mut y = data;
    for (axis, (&pre, &post)) in pads_pre.iter().zip(pads_post.iter()).enumerate() {
        if pre == 0 && post == 0 {
            continue;
        }
        y = y.pad_with_zeros(axis, pre as usize, post as usize)?;
    }
    set_output(node, 0, y, values)?;
    Ok(())
}

fn scalar_to_f64(t: &LazyTensor) -> Result<f64> {
    if t.elem_count() != 1 {
        return Err(Error::Msg(format!(
            "Pad: constant_value must be a scalar, got shape {:?}",
            t.shape().dims()
        ))
        .bt());
    }
    match t.dtype() {
        fuel::DType::F32 => Ok(t.realize_f32()[0] as f64),
        fuel::DType::F64 => Ok(t.realize_f64()[0]),
        other => Err(Error::Msg(format!(
            "Pad: constant_value dtype {other:?} not supported in sub-port 2"
        ))
        .bt()),
    }
}

fn max_pool_op(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
) -> Result<()> {
    let (kernel, strides, padding) = parse_pool_attrs(node)?;
    let x = get(values, node, &node.input[0])?;
    if x.rank() != 4 {
        return Err(Error::Msg(format!(
            "MaxPool '{}': input must be rank 4 [N, C, H, W], got {:?}",
            node.name,
            x.shape().dims()
        ))
        .bt());
    }
    if node.output.len() > 1 && !node.output[1].is_empty() {
        return Err(Error::Msg(format!(
            "MaxPool '{}': indices output is not supported in sub-port 2",
            node.name
        ))
        .bt());
    }
    let y = x.max_pool2d(kernel, strides, padding)?;
    set_output(node, 0, y, values)?;
    Ok(())
}

fn avg_pool_op(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
) -> Result<()> {
    let (kernel, strides, padding) = parse_pool_attrs(node)?;
    if get_attr_int_opt(node, "count_include_pad").unwrap_or(0) != 0 && padding != (0, 0) {
        return Err(Error::Msg(format!(
            "AveragePool '{}': count_include_pad=1 with non-zero pads is not supported in sub-port 2",
            node.name
        ))
        .bt());
    }
    let x = get(values, node, &node.input[0])?;
    if x.rank() != 4 {
        return Err(Error::Msg(format!(
            "AveragePool '{}': input must be rank 4 [N, C, H, W], got {:?}",
            node.name,
            x.shape().dims()
        ))
        .bt());
    }
    let y = x.avg_pool2d(kernel, strides, padding)?;
    set_output(node, 0, y, values)?;
    Ok(())
}

fn parse_pool_attrs(
    node: &onnx::NodeProto,
) -> Result<((usize, usize), (usize, usize), (usize, usize))> {
    let auto_pad = parse_auto_pad(node)?;
    check_dilations(node)?;
    if get_attr_int_opt(node, "ceil_mode").unwrap_or(0) != 0 {
        return Err(Error::Msg(format!(
            "node '{}' ({}): ceil_mode != 0 is not supported in sub-port 2",
            node.name, node.op_type
        ))
        .bt());
    }
    let kernel_shape = get_attr_ints_opt(node, "kernel_shape").ok_or_else(|| {
        Error::Msg(format!(
            "node '{}' ({}): missing required attribute 'kernel_shape'",
            node.name, node.op_type
        ))
        .bt()
    })?;
    let kernel: (usize, usize) = match kernel_shape {
        [kh, kw] => (*kh as usize, *kw as usize),
        s => {
            return Err(Error::Msg(format!(
                "node '{}' ({}): only 2D kernels supported in sub-port 2, got {s:?}",
                node.name, node.op_type
            ))
            .bt());
        }
    };
    let strides: (usize, usize) = match get_attr_ints_opt(node, "strides") {
        None => (1, 1),
        Some([sh, sw]) => (*sh as usize, *sw as usize),
        Some(s) => {
            return Err(Error::Msg(format!(
                "node '{}' ({}): expected 2 strides, got {s:?}",
                node.name, node.op_type
            ))
            .bt());
        }
    };
    // For pooling we use the resolve_pads helper but we don't know the
    // input spatial size at this point — auto_pad SAME_* requires it.
    // The pooling primitives below only accept *symmetric* padding,
    // so we lock auto_pad SAME_* on rank > 0 to NOTSET-with-pads
    // resolution and require equal pre/post pads downstream.
    let padding = match auto_pad {
        AutoPad::NotSet | AutoPad::Valid => {
            let pads_attr = get_attr_ints_opt(node, "pads");
            match (auto_pad, pads_attr) {
                (AutoPad::Valid, _) | (_, None) => (0usize, 0usize),
                (_, Some(p)) => {
                    if p.len() != 4 {
                        return Err(Error::Msg(format!(
                            "node '{}' ({}): pads len {} != 4 for 2D pool",
                            node.name,
                            node.op_type,
                            p.len()
                        ))
                        .bt());
                    }
                    if p.iter().any(|&v| v < 0) {
                        return Err(Error::Msg(format!(
                            "node '{}' ({}): negative pads {p:?} not supported",
                            node.name, node.op_type
                        ))
                        .bt());
                    }
                    if p[0] != p[2] || p[1] != p[3] {
                        return Err(Error::Msg(format!(
                            "node '{}' ({}): asymmetric pads {p:?} not supported \
                             in sub-port 2 pooling",
                            node.name, node.op_type
                        ))
                        .bt());
                    }
                    (p[0] as usize, p[1] as usize)
                }
            }
        }
        AutoPad::SameUpper | AutoPad::SameLower => {
            return Err(Error::Msg(format!(
                "node '{}' ({}): auto_pad SAME_* on pooling is not supported \
                 in sub-port 2; convert to explicit symmetric 'pads'",
                node.name, node.op_type
            ))
            .bt());
        }
    };
    Ok((kernel, strides, padding))
}

fn global_avg_pool_op(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
) -> Result<()> {
    let x = get(values, node, &node.input[0])?;
    if x.rank() != 4 {
        return Err(Error::Msg(format!(
            "GlobalAveragePool '{}': input must be rank 4, got {:?}",
            node.name,
            x.shape().dims()
        ))
        .bt());
    }
    // ONNX GlobalAveragePool keeps the spatial dims at size 1.
    let y = x.mean_keepdim(3_usize)?.mean_keepdim(2_usize)?;
    set_output(node, 0, y, values)?;
    Ok(())
}

fn global_max_pool_op(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
) -> Result<()> {
    let x = get(values, node, &node.input[0])?;
    if x.rank() != 4 {
        return Err(Error::Msg(format!(
            "GlobalMaxPool '{}': input must be rank 4, got {:?}",
            node.name,
            x.shape().dims()
        ))
        .bt());
    }
    let y = x.max_keepdim(3_usize)?.max_keepdim(2_usize)?;
    set_output(node, 0, y, values)?;
    Ok(())
}

// ===========================================================================
// tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy_eval::LazyOnnxEval;
    use crate::onnx::attribute_proto::AttributeType;
    use crate::onnx::tensor_proto::DataType;
    use fuel::Device;
    use prost::Message;

    fn tp_float(name: &str, dims: &[i64], data: Vec<f32>) -> onnx::TensorProto {
        onnx::TensorProto {
            dims: dims.to_vec(),
            data_type: DataType::Float as i32,
            float_data: data,
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn tp_int64(name: &str, dims: &[i64], data: Vec<i64>) -> onnx::TensorProto {
        onnx::TensorProto {
            dims: dims.to_vec(),
            data_type: DataType::Int64 as i32,
            int64_data: data,
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn attr_ints(name: &str, v: Vec<i64>) -> onnx::AttributeProto {
        onnx::AttributeProto {
            name: name.to_string(),
            r#type: AttributeType::Ints as i32,
            ints: v,
            ..Default::default()
        }
    }

    fn attr_string(name: &str, s: &str) -> onnx::AttributeProto {
        onnx::AttributeProto {
            name: name.to_string(),
            r#type: AttributeType::String as i32,
            s: s.as_bytes().to_vec(),
            ..Default::default()
        }
    }

    fn value_info(name: &str) -> onnx::ValueInfoProto {
        onnx::ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn node(op_type: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> onnx::NodeProto {
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

    fn run_graph(
        graph: onnx::GraphProto,
        inputs: HashMap<String, LazyTensor>,
    ) -> Result<HashMap<String, LazyTensor>> {
        let mut buf = Vec::new();
        model_from_graph(graph).encode(&mut buf).unwrap();
        let evaluator = LazyOnnxEval::from_bytes(&buf)?;
        evaluator.run(&inputs)
    }

    #[test]
    fn conv2d_forward_known_kernel_shape_match() {
        // x [1,1,3,3] = identity sum-kernel test:
        //   kernel = 3x3 ones, stride 1, no padding → out [1,1,1,1] = sum(x)
        let x_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let w_data: Vec<f32> = vec![1.0; 9];
        let mut conv = node("Conv", "cv", &["X", "W"], &["Y"]);
        conv.attribute.push(attr_ints("strides", vec![1, 1]));
        let graph = onnx::GraphProto {
            node: vec![conv],
            initializer: vec![tp_float("W", &[1, 1, 3, 3], w_data)],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let device = Device::cpu();
        let x = LazyTensor::from_f32(x_data.clone(), Shape::from_dims(&[1, 1, 3, 3]), &device);
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let outputs = run_graph(graph, inputs).unwrap();
        let y = outputs.get("Y").unwrap();
        assert_eq!(y.shape().dims(), &[1, 1, 1, 1]);
        let got = y.realize_f32();
        assert!(approx_eq(got[0], 45.0, 1e-4), "got={got:?}");
    }

    #[test]
    fn conv_transpose2d_forward_shape() {
        // x [1,1,2,2], w [1,1,2,2] stride 2 → out spatial = (in-1)*stride + k
        // = 1*2 + 2 = 4. Shape [1,1,4,4].
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let w_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let mut ct = node("ConvTranspose", "ct", &["X", "W"], &["Y"]);
        ct.attribute.push(attr_ints("strides", vec![2, 2]));
        let graph = onnx::GraphProto {
            node: vec![ct],
            initializer: vec![tp_float("W", &[1, 1, 2, 2], w_data)],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let device = Device::cpu();
        let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[1, 1, 2, 2]), &device);
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let outputs = run_graph(graph, inputs).unwrap();
        let y = outputs.get("Y").unwrap();
        assert_eq!(y.shape().dims(), &[1, 1, 4, 4]);
        let got = y.realize_f32();
        assert!(got.iter().all(|v| v.is_finite()), "got={got:?}");
    }

    #[test]
    fn pad_constant_adds_zeros() {
        // x [1, 2] -> pad pre=[0,1] post=[0,1] -> [1, 4]
        let pads = tp_int64("pads", &[4], vec![0, 1, 0, 1]);
        let mut pad = node("Pad", "p", &["X", "pads"], &["Y"]);
        pad.attribute.push(attr_string("mode", "constant"));
        let graph = onnx::GraphProto {
            node: vec![pad],
            initializer: vec![pads],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let device = Device::cpu();
        let x = LazyTensor::from_f32(vec![5.0_f32, 7.0], Shape::from_dims(&[1, 2]), &device);
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let outputs = run_graph(graph, inputs).unwrap();
        let y = outputs.get("Y").unwrap();
        // pads = [0, 1, 0, 1] → axis 0: (pre=0, post=0); axis 1: (pre=1, post=1)
        // → x [5,7] padded to [0, 5, 7, 0]
        assert_eq!(y.shape().dims(), &[1, 4]);
        let got = y.realize_f32();
        assert!(approx_eq(got[0], 0.0, 1e-7), "got={got:?}");
        assert!(approx_eq(got[1], 5.0, 1e-7), "got={got:?}");
        assert!(approx_eq(got[2], 7.0, 1e-7), "got={got:?}");
        assert!(approx_eq(got[3], 0.0, 1e-7), "got={got:?}");
    }

    #[test]
    fn pad_reflect_errors_with_sub_port_message() {
        let pads = tp_int64("pads", &[4], vec![0, 1, 0, 1]);
        let mut pad = node("Pad", "p", &["X", "pads"], &["Y"]);
        pad.attribute.push(attr_string("mode", "reflect"));
        let graph = onnx::GraphProto {
            node: vec![pad],
            initializer: vec![pads],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let device = Device::cpu();
        let x = LazyTensor::from_f32(vec![1.0_f32, 2.0], Shape::from_dims(&[1, 2]), &device);
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let err = run_graph(graph, inputs).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("reflect"), "msg should mention mode: {msg}");
        assert!(msg.contains("sub-port"), "msg should point at sub-port: {msg}");
    }

    #[test]
    fn maxpool2d_kernel_2_stride_2_halves_spatial_dims() {
        // [1,1,4,4] → pool 2x2 stride 2 → [1,1,2,2]
        let x_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let mut mp = node("MaxPool", "mp", &["X"], &["Y"]);
        mp.attribute.push(attr_ints("kernel_shape", vec![2, 2]));
        mp.attribute.push(attr_ints("strides", vec![2, 2]));
        let graph = onnx::GraphProto {
            node: vec![mp],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let device = Device::cpu();
        let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[1, 1, 4, 4]), &device);
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let outputs = run_graph(graph, inputs).unwrap();
        let y = outputs.get("Y").unwrap();
        assert_eq!(y.shape().dims(), &[1, 1, 2, 2]);
        let got = y.realize_f32();
        // Top-left window [0,1,4,5] → max 5
        // Top-right window [2,3,6,7] → max 7
        // Bottom-left [8,9,12,13] → max 13
        // Bottom-right [10,11,14,15] → max 15
        let expected = [5.0_f32, 7.0, 13.0, 15.0];
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(approx_eq(*g, *e, 1e-5), "mismatch at {i}: got {g} expected {e} (got={got:?})");
        }
    }

    #[test]
    fn avgpool2d_returns_mean_within_kernel_window() {
        // [1,1,2,2] → pool 2x2 stride 1 → [1,1,1,1] = mean of all four
        let x_data: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];
        let mut ap = node("AveragePool", "ap", &["X"], &["Y"]);
        ap.attribute.push(attr_ints("kernel_shape", vec![2, 2]));
        ap.attribute.push(attr_ints("strides", vec![1, 1]));
        let graph = onnx::GraphProto {
            node: vec![ap],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let device = Device::cpu();
        let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[1, 1, 2, 2]), &device);
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let outputs = run_graph(graph, inputs).unwrap();
        let y = outputs.get("Y").unwrap();
        assert_eq!(y.shape().dims(), &[1, 1, 1, 1]);
        let got = y.realize_f32();
        assert!(approx_eq(got[0], 5.0, 1e-5), "got={got:?}"); // (2+4+6+8)/4 = 5
    }

    #[test]
    fn globalavgpool_collapses_to_channel_means() {
        // [1, 2, 2, 2] with C0 spatial = [1,2,3,4] mean 2.5; C1 = [5,6,7,8] mean 6.5
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gap = node("GlobalAveragePool", "g", &["X"], &["Y"]);
        let graph = onnx::GraphProto {
            node: vec![gap],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let device = Device::cpu();
        let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[1, 2, 2, 2]), &device);
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let outputs = run_graph(graph, inputs).unwrap();
        let y = outputs.get("Y").unwrap();
        assert_eq!(y.shape().dims(), &[1, 2, 1, 1]);
        let got = y.realize_f32();
        assert!(approx_eq(got[0], 2.5, 1e-5), "got={got:?}");
        assert!(approx_eq(got[1], 6.5, 1e-5), "got={got:?}");
    }

    #[test]
    fn globalmaxpool_collapses_to_channel_maxes() {
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gmp = node("GlobalMaxPool", "g", &["X"], &["Y"]);
        let graph = onnx::GraphProto {
            node: vec![gmp],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let device = Device::cpu();
        let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[1, 2, 2, 2]), &device);
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let outputs = run_graph(graph, inputs).unwrap();
        let y = outputs.get("Y").unwrap();
        assert_eq!(y.shape().dims(), &[1, 2, 1, 1]);
        let got = y.realize_f32();
        assert!(approx_eq(got[0], 4.0, 1e-5), "got={got:?}");
        assert!(approx_eq(got[1], 8.0, 1e-5), "got={got:?}");
    }
}

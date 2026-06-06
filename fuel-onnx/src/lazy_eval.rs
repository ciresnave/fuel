//! Lazy-graph ONNX evaluator — sub-port 1 of `port-onnx-eval.md`.
//!
//! Walks an ONNX graph and dispatches each node to the matching
//! [`fuel::lazy::LazyTensor`] primitive, building a fuel_graph
//! computation tree rather than running ops eagerly. Realization
//! happens lazily when the caller realizes any output tensor.
//!
//! # Scope (sub-port 1)
//!
//! Op set covered by this sub-port:
//!   - Binary arithmetic with broadcasting: `MatMul`, `Add`, `Sub`,
//!     `Mul`, `Div`.
//!   - Shape ops: `Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`,
//!     `Flatten`, `Identity`.
//!   - Indexing: `Gather` (→ `index_select`).
//!   - Reductions: `ReduceMean`, `ReduceSum`, `ReduceMax`, `ReduceMin`
//!     (with `axes` + `keepdims` support).
//!   - Constants: `Constant`, `ConstantOfShape`.
//!   - Dtype: `Cast` (via [`LazyTensor::to_dtype`]).
//!   - Multi-input shape glue: `Concat`, `Split`.
//!
//! Conv / ConvTranspose / Pooling / Pad land in sub-port 2 — see the
//! sibling [`crate::lazy_eval_conv`] module which hooks into the same
//! [`LazyOnnxEval`] dispatch path via [`crate::lazy_eval_conv::try_dispatch_node`].
//! BatchNorm / LayerNorm / activations / Softmax land in sub-port 3.
//! Quantized ops (`QLinearMatMul`, `Quantize/DequantizeLinear`) land
//! in sub-port 4.

use crate::onnx::{self, attribute_proto::AttributeType, tensor_proto::DataType};
use fuel::lazy::LazyTensor;
use fuel::{DType, Device, Error, Result, Shape};
use prost::Message;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Lazy ONNX evaluator: owns a deserialized model proto and dispatches
/// nodes to [`LazyTensor`] primitives on [`run`](Self::run).
#[derive(Clone)]
pub struct LazyOnnxEval {
    model: onnx::ModelProto,
}

impl LazyOnnxEval {
    /// Build from a protobuf-encoded `ModelProto`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let model = onnx::ModelProto::decode(bytes).map_err(Error::wrap)?;
        Ok(Self { model })
    }

    /// Read and decode an `.onnx` file from disk.
    pub fn from_path<P: AsRef<Path>>(p: P) -> Result<Self> {
        let bytes = std::fs::read(p)?;
        Self::from_bytes(&bytes)
    }

    /// Borrow the underlying ONNX model proto.
    pub fn model(&self) -> &onnx::ModelProto {
        &self.model
    }

    /// Evaluate the graph. Maps each declared output name to the
    /// `LazyTensor` produced by the corresponding node. The returned
    /// tensors have not been realized yet — call e.g. `realize_f32` on
    /// any of them to materialize the result.
    pub fn run(
        &self,
        inputs: &HashMap<String, LazyTensor>,
    ) -> Result<HashMap<String, LazyTensor>> {
        let graph = self
            .model
            .graph
            .as_ref()
            .ok_or_else(|| Error::Msg("ONNX model has no graph".into()).bt())?;

        let mut values: HashMap<String, LazyTensor> = HashMap::new();
        let mut anchor: Option<LazyTensor> = None;
        let mut i64_cache: HashMap<String, Vec<i64>> = HashMap::new();

        // Seed inputs first so they serve as the const_*_like anchor —
        // all initializers / Constant outputs must land on the input's
        // graph, otherwise downstream binary ops panic with
        // "tensors must live on the same graph".
        for (k, v) in inputs.iter() {
            if anchor.is_none() {
                anchor = Some(v.clone());
            }
            values.insert(k.clone(), v.clone());
        }

        let device = Device::cpu();
        for init in graph.initializer.iter() {
            let t = load_initializer(init, &device, &mut anchor)?;
            if init.data_type == DataType::Int64 as i32 {
                i64_cache.insert(init.name.clone(), int64_data_vec(init)?);
            }
            values.insert(init.name.clone(), t);
        }

        for node in graph.node.iter() {
            dispatch_node(node, &mut values, &device, &mut anchor, &mut i64_cache)?;
        }

        let mut outputs = HashMap::new();
        for out in graph.output.iter() {
            let v = values.remove(&out.name).ok_or_else(|| {
                Error::Msg(format!("graph output '{}' not produced", out.name)).bt()
            })?;
            outputs.insert(out.name.clone(), v);
        }
        Ok(outputs)
    }
}

fn dispatch_node(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
    device: &Device,
    anchor: &mut Option<LazyTensor>,
    i64_cache: &mut HashMap<String, Vec<i64>>,
) -> Result<()> {
    let get = |values: &HashMap<String, LazyTensor>, name: &str| -> Result<LazyTensor> {
        values
            .get(name)
            .cloned()
            .ok_or_else(|| Error::Msg(format!("missing input '{}' for node '{}'", name, node.name)).bt())
    };
    let get_i64_vec = |values: &HashMap<String, LazyTensor>,
                       i64_cache: &HashMap<String, Vec<i64>>,
                       name: &str|
     -> Result<Vec<i64>> {
        if let Some(cached) = i64_cache.get(name) {
            return Ok(cached.clone());
        }
        let t = values.get(name).ok_or_else(|| {
            Error::Msg(format!("missing tensor '{}' (expected i64 vector)", name)).bt()
        })?;
        realize_i64_vec(t)
    };

    match node.op_type.as_str() {
        // ---- binary arithmetic with broadcasting ----
        "Add" => {
            let a = get(values, &node.input[0])?;
            let b = get(values, &node.input[1])?;
            let y = a.broadcast_add(&b)?;
            set_output(node, 0, y, values)?;
        }
        "Sub" => {
            let a = get(values, &node.input[0])?;
            let b = get(values, &node.input[1])?;
            let y = a.broadcast_sub(&b)?;
            set_output(node, 0, y, values)?;
        }
        "Mul" => {
            let a = get(values, &node.input[0])?;
            let b = get(values, &node.input[1])?;
            let y = a.broadcast_mul(&b)?;
            set_output(node, 0, y, values)?;
        }
        "Div" => {
            let a = get(values, &node.input[0])?;
            let b = get(values, &node.input[1])?;
            let y = a.broadcast_div(&b)?;
            set_output(node, 0, y, values)?;
        }
        "MatMul" => {
            let a = get(values, &node.input[0])?;
            let b = get(values, &node.input[1])?;
            let y = a.broadcast_matmul(&b)?;
            set_output(node, 0, y, values)?;
        }

        // ---- shape ops ----
        "Reshape" => {
            let x = get(values, &node.input[0])?;
            let raw = get_i64_vec(values, i64_cache, &node.input[1])?;
            let mut other: usize = 1;
            for &v in raw.iter() {
                if v != -1 && v != 0 {
                    other *= v as usize;
                }
            }
            let dims: Vec<usize> = raw
                .iter()
                .enumerate()
                .map(|(i, &v)| match v {
                    -1 => Ok::<usize, Error>(x.elem_count() / other),
                    0 => x.dim(i),
                    n if n < 0 => Err(Error::Msg(format!(
                        "Reshape: negative dim '{n}' (only -1 sentinel is allowed)"
                    ))
                    .bt()),
                    n => Ok(n as usize),
                })
                .collect::<Result<Vec<_>>>()?;
            let y = x.reshape(Shape::from_dims(&dims))?;
            set_output(node, 0, y, values)?;
        }
        "Transpose" => {
            let x = get(values, &node.input[0])?;
            let perm = get_attr_ints_opt(node, "perm");
            let y = match perm {
                None => {
                    let rank = x.rank();
                    if rank < 2 {
                        x.clone()
                    } else {
                        let mut axes: Vec<usize> = (0..rank).collect();
                        axes.swap(rank - 2, rank - 1);
                        x.permute(axes.as_slice())?
                    }
                }
                Some(perm) => {
                    let axes: Vec<usize> = perm.iter().map(|&v| v as usize).collect();
                    x.permute(axes.as_slice())?
                }
            };
            set_output(node, 0, y, values)?;
        }
        "Squeeze" => {
            let x = get(values, &node.input[0])?;
            let mut axes: Vec<usize> = if node.input.len() > 1 && !node.input[1].is_empty() {
                let raw = get_i64_vec(values, i64_cache, &node.input[1])?;
                raw.iter()
                    .map(|&i| normalize_axis(i, x.rank()))
                    .collect::<Result<Vec<_>>>()?
            } else if let Some(a) = get_attr_ints_opt(node, "axes") {
                a.iter().map(|&i| normalize_axis(i, x.rank())).collect::<Result<Vec<_>>>()?
            } else {
                x.shape()
                    .dims()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &d)| if d == 1 { Some(i) } else { None })
                    .collect()
            };
            axes.sort();
            axes.dedup();
            let mut y = x;
            for &axis in axes.iter().rev() {
                y = y.squeeze(axis)?;
            }
            set_output(node, 0, y, values)?;
        }
        "Unsqueeze" => {
            let x = get(values, &node.input[0])?;
            let raw_axes: Vec<i64> = if node.input.len() > 1 && !node.input[1].is_empty() {
                get_i64_vec(values, i64_cache, &node.input[1])?
            } else {
                get_attr_ints_opt(node, "axes")
                    .ok_or_else(|| {
                        Error::Msg(format!(
                            "Unsqueeze node '{}' missing 'axes' input/attribute",
                            node.name
                        ))
                        .bt()
                    })?
                    .to_vec()
            };
            let final_rank = x.rank() + raw_axes.len();
            let mut axes: Vec<usize> = raw_axes
                .iter()
                .map(|&i| normalize_axis_plus_one(i, final_rank))
                .collect::<Result<Vec<_>>>()?;
            axes.sort();
            axes.dedup();
            let mut y = x;
            for &axis in axes.iter() {
                y = y.unsqueeze(axis)?;
            }
            set_output(node, 0, y, values)?;
        }
        "Flatten" => {
            let x = get(values, &node.input[0])?;
            let axis = get_attr_int_opt(node, "axis").unwrap_or(1) as usize;
            let dims = x.shape().dims().to_vec();
            let first: usize = dims.iter().take(axis).product();
            let total: usize = dims.iter().product();
            let y = x.reshape(Shape::from_dims(&[first, total / first.max(1)]))?;
            set_output(node, 0, y, values)?;
        }
        "Identity" => {
            let x = get(values, &node.input[0])?;
            set_output(node, 0, x, values)?;
        }

        // ---- indexing ----
        "Gather" => {
            // Only the (1-D indices, axis selectable) ONNX subset for v1.
            let x = get(values, &node.input[0])?;
            let indices = get(values, &node.input[1])?;
            let axis = normalize_axis(get_attr_int_opt(node, "axis").unwrap_or(0), x.rank())?;
            let dim_size = x.dim(axis)?;
            let raw = get_i64_vec(values, i64_cache, &node.input[1])?;
            let mut normalized = Vec::with_capacity(raw.len());
            for v in raw.iter() {
                let n = if *v < 0 { *v + dim_size as i64 } else { *v };
                if n < 0 || n >= dim_size as i64 {
                    return Err(Error::Msg(format!(
                        "Gather: index {v} out of range for dim size {dim_size}"
                    ))
                    .bt());
                }
                normalized.push(n as u32);
            }
            match indices.rank() {
                0 => {
                    let y = x.narrow(axis, normalized[0] as usize, 1)?.squeeze(axis)?;
                    set_output(node, 0, y, values)?;
                }
                1 => {
                    let a = ensure_anchor(anchor, device);
                    let idx = a.const_u32_like(normalized, Shape::from_dims(&[indices.elem_count()]));
                    let y = x.index_select(axis, &idx)?;
                    set_output(node, 0, y, values)?;
                }
                r => {
                    return Err(Error::Msg(format!(
                        "Gather: only rank-0 and rank-1 indices are supported in sub-port 1, \
                         got rank-{r} indices for node '{}'",
                        node.name,
                    ))
                    .bt());
                }
            }
        }

        // ---- reductions ----
        "ReduceMean" => reduce_op(node, values, i64_cache, ReduceKind::Mean)?,
        "ReduceSum" => reduce_op(node, values, i64_cache, ReduceKind::Sum)?,
        "ReduceMax" => reduce_op(node, values, i64_cache, ReduceKind::Max)?,
        "ReduceMin" => reduce_op(node, values, i64_cache, ReduceKind::Min)?,

        // ---- constants ----
        "Constant" => {
            let attr = node
                .attribute
                .iter()
                .find(|a| a.name == "value")
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "Constant node '{}' missing 'value' attribute (sparse/value_* variants \
                         are unsupported in sub-port 1)",
                        node.name
                    ))
                    .bt()
                })?;
            if attr.r#type() != AttributeType::Tensor {
                return Err(Error::Msg(format!(
                    "Constant node '{}': 'value' attribute must be a tensor in sub-port 1, got {:?}",
                    node.name,
                    attr.r#type()
                ))
                .bt());
            }
            let t = attr
                .t
                .as_ref()
                .ok_or_else(|| Error::Msg("Constant: 'value' has no tensor payload".into()).bt())?;
            if t.data_type == DataType::Int64 as i32 {
                let out_name = node
                    .output
                    .first()
                    .ok_or_else(|| {
                        Error::Msg(format!("Constant '{}' has no output", node.name)).bt()
                    })?
                    .clone();
                i64_cache.insert(out_name, int64_data_vec(t)?);
            }
            let y = load_initializer(t, device, anchor)?;
            set_output(node, 0, y, values)?;
        }
        "ConstantOfShape" => {
            let shape_dims: Vec<usize> = get_i64_vec(values, i64_cache, &node.input[0])?
                .into_iter()
                .map(|v| v as usize)
                .collect();
            let attr = node.attribute.iter().find(|a| a.name == "value");
            let y = match attr {
                None => {
                    let a = ensure_anchor(anchor, device);
                    let n: usize = shape_dims.iter().product();
                    a.const_f32_like(vec![0.0_f32; n], Shape::from_dims(&shape_dims))
                }
                Some(a) => {
                    let t = a.t.as_ref().ok_or_else(|| {
                        Error::Msg("ConstantOfShape: 'value' has no tensor payload".into()).bt()
                    })?;
                    let dt = onnx_dtype_to_fuel(t.data_type)?;
                    fill_from_value_proto(t, dt, &shape_dims, anchor, device)?
                }
            };
            set_output(node, 0, y, values)?;
        }

        // ---- dtype ----
        "Cast" => {
            let x = get(values, &node.input[0])?;
            let to = get_attr_int(node, "to")?;
            let onnx_dt = DataType::try_from(to as i32).map_err(|_| {
                Error::Msg(format!(
                    "Cast node '{}': invalid 'to' value {to}",
                    node.name
                ))
                .bt()
            })?;
            let target = onnx_dtype_to_fuel(onnx_dt as i32)?;
            let y = x.to_dtype(target)?;
            set_output(node, 0, y, values)?;
        }

        // ---- multi-input shape glue ----
        "Concat" => {
            if node.input.is_empty() {
                return Err(Error::Msg(format!(
                    "Concat node '{}': empty input list",
                    node.name
                ))
                .bt());
            }
            let axis_raw = get_attr_int(node, "axis")?;
            let first = get(values, &node.input[0])?;
            let axis = normalize_axis(axis_raw, first.rank())?;
            let mut acc = first;
            for name in &node.input[1..] {
                let next = get(values, name)?;
                acc = acc.concat(&next, axis)?;
            }
            set_output(node, 0, acc, values)?;
        }
        "Split" => {
            let x = get(values, &node.input[0])?;
            let axis_raw = get_attr_int_opt(node, "axis").unwrap_or(0);
            let axis = normalize_axis(axis_raw, x.rank())?;
            let dim_size = x.dim(axis)?;
            let splits: Vec<usize> = if node.input.len() > 1 && !node.input[1].is_empty() {
                get_i64_vec(values, i64_cache, &node.input[1])?
                    .into_iter()
                    .map(|v| v as usize)
                    .collect()
            } else if let Some(s) = get_attr_ints_opt(node, "split") {
                s.iter().map(|&v| v as usize).collect()
            } else {
                let num_outputs = get_attr_int_opt(node, "num_outputs")
                    .map(|v| v as usize)
                    .unwrap_or(node.output.len());
                if num_outputs == 0 {
                    return Err(Error::Msg(format!(
                        "Split node '{}': cannot derive split sizes (num_outputs=0)",
                        node.name
                    ))
                    .bt());
                }
                let base = dim_size / num_outputs;
                let rem = dim_size % num_outputs;
                let mut s = vec![base; num_outputs];
                if rem > 0 {
                    s[num_outputs - 1] += rem;
                }
                s
            };
            let mut start = 0;
            for (i, &len) in splits.iter().enumerate() {
                let out_name = node.output.get(i).ok_or_else(|| {
                    Error::Msg(format!(
                        "Split node '{}': more splits than output slots",
                        node.name
                    ))
                    .bt()
                })?;
                let chunk = x.narrow(axis, start, len)?;
                values.insert(out_name.clone(), chunk);
                start += len;
            }
        }

        // ---- sub-port 2: conv / conv-transpose / pad / pooling ----
        // ---- sub-port 3: norm / activations / softmax            ----
        op_type => {
            if crate::lazy_eval_conv::try_dispatch_node(node, values, device, anchor, i64_cache)? {
                return Ok(());
            }
            if crate::lazy_eval_norm::try_dispatch(node, values, device, anchor)? {
                return Ok(());
            }
            let sub_port = classify_op_sub_port(op_type);
            return Err(Error::Msg(format!(
                "ONNX op '{}' (node '{}') is not supported in sub-port 1; \
                 it is scheduled for {}",
                op_type, node.name, sub_port,
            ))
            .bt());
        }
    }
    Ok(())
}

pub(crate) fn set_output(
    node: &onnx::NodeProto,
    i: usize,
    v: LazyTensor,
    values: &mut HashMap<String, LazyTensor>,
) -> Result<()> {
    let name = node.output.get(i).ok_or_else(|| {
        Error::Msg(format!(
            "node '{}' has no output[{i}] slot to receive result",
            node.name
        ))
        .bt()
    })?;
    values.insert(name.clone(), v);
    Ok(())
}

#[derive(Clone, Copy)]
enum ReduceKind {
    Mean,
    Sum,
    Max,
    Min,
}

fn reduce_op(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, LazyTensor>,
    i64_cache: &HashMap<String, Vec<i64>>,
    kind: ReduceKind,
) -> Result<()> {
    let x = values
        .get(&node.input[0])
        .cloned()
        .ok_or_else(|| {
            Error::Msg(format!("missing input for reduce node '{}'", node.name)).bt()
        })?;
    let keepdims = get_attr_int_opt(node, "keepdims").unwrap_or(1) == 1;
    let noop_with_empty_axes =
        get_attr_int_opt(node, "noop_with_empty_axes").unwrap_or(0) == 1;

    // ONNX 13+: axes is an attribute. ONNX 18+: axes is an input.
    let raw_axes: Option<Vec<i64>> = if node.input.len() > 1 && !node.input[1].is_empty() {
        let name = &node.input[1];
        if let Some(cached) = i64_cache.get(name) {
            Some(cached.clone())
        } else {
            let t = values.get(name).cloned().ok_or_else(|| {
                Error::Msg(format!("missing axes input for reduce node '{}'", node.name))
                    .bt()
            })?;
            Some(realize_i64_vec(&t)?)
        }
    } else {
        get_attr_ints_opt(node, "axes").map(|s| s.to_vec())
    };

    let rank = x.rank();
    let axes: Vec<usize> = match raw_axes {
        Some(a) if a.is_empty() && noop_with_empty_axes => {
            set_output(node, 0, x, values)?;
            return Ok(());
        }
        Some(a) if a.is_empty() => (0..rank).collect(),
        Some(a) => {
            let mut out = a
                .iter()
                .map(|&i| normalize_axis(i, rank))
                .collect::<Result<Vec<_>>>()?;
            out.sort();
            out.dedup();
            out
        }
        None => (0..rank).collect(),
    };

    if x.elem_count() == 0 {
        return Err(Error::Msg(format!(
            "reduce node '{}': reduction over zero-element tensor unsupported",
            node.name
        ))
        .bt());
    }

    // Reduce highest dim first so lower indices remain valid; the
    // non-keepdim path removes the axis on each step.
    let mut acc = x;
    for &axis in axes.iter().rev() {
        acc = match (kind, keepdims) {
            (ReduceKind::Mean, true) => acc.mean_keepdim(axis)?,
            (ReduceKind::Mean, false) => acc.mean_dim(axis)?,
            (ReduceKind::Sum, true) => acc.sum_keepdim(axis)?,
            (ReduceKind::Sum, false) => acc.sum_dim(axis)?,
            (ReduceKind::Max, true) => acc.max_keepdim(axis)?,
            (ReduceKind::Max, false) => acc.max_dim(axis)?,
            (ReduceKind::Min, true) => acc.min_keepdim(axis)?,
            (ReduceKind::Min, false) => acc.min_dim(axis)?,
        };
    }
    set_output(node, 0, acc, values)?;
    Ok(())
}

fn load_initializer(
    t: &onnx::TensorProto,
    device: &Device,
    anchor: &mut Option<LazyTensor>,
) -> Result<LazyTensor> {
    let dims: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
    let shape = Shape::from_dims(&dims);
    let dt = DataType::try_from(t.data_type).map_err(|_| {
        Error::Msg(format!(
            "initializer '{}': invalid data_type {}",
            t.name, t.data_type
        ))
        .bt()
    })?;
    // Float / double tensors can serve as the anchor themselves; the
    // anchor only needs to exist before any binary op runs, and
    // const_*_like requires an existing same-graph tensor — so we
    // bootstrap the first float initializer via `from_f32` (new
    // graph), then thread subsequent ones onto it.
    let built = match dt {
        DataType::Float => {
            let data = float_data(t)?;
            match anchor.as_ref() {
                None => LazyTensor::from_f32(data, shape, device),
                Some(a) => a.const_f32_like(data, shape),
            }
        }
        DataType::Double => {
            let data = double_data(t)?;
            match anchor.as_ref() {
                None => LazyTensor::from_f64(data, shape, device),
                Some(a) => a.const_f64_like(data, shape),
            }
        }
        DataType::Int64 => {
            let data = int64_data(t)?;
            let a = ensure_anchor(anchor, device);
            a.const_i64_like(data, shape)
        }
        DataType::Uint8 | DataType::Bool => {
            let data = u32_from_uint8_bool(t)?;
            let a = ensure_anchor(anchor, device);
            a.const_u32_like(data, shape)
        }
        DataType::Uint32 => {
            let data = u32_data(t)?;
            let a = ensure_anchor(anchor, device);
            a.const_u32_like(data, shape)
        }
        other => {
            return Err(Error::Msg(format!(
                "initializer '{}': unsupported dtype {:?} in sub-port 1",
                t.name, other,
            ))
            .bt())
        }
    };
    if anchor.is_none() {
        *anchor = Some(built.clone());
    }
    Ok(built)
}

pub(crate) fn ensure_anchor(anchor: &mut Option<LazyTensor>, device: &Device) -> LazyTensor {
    if anchor.is_none() {
        *anchor = Some(LazyTensor::from_f32(
            Arc::<[f32]>::from(vec![0.0f32]),
            Shape::from_dims(&[1]),
            device,
        ));
    }
    anchor.clone().unwrap()
}

fn int64_data_vec(t: &onnx::TensorProto) -> Result<Vec<i64>> {
    if !t.int64_data.is_empty() {
        Ok(t.int64_data.clone())
    } else if !t.raw_data.is_empty() {
        let n = t.raw_data.len() / 8;
        let mut out = Vec::with_capacity(n);
        for chunk in t.raw_data.chunks_exact(8) {
            out.push(i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]));
        }
        Ok(out)
    } else {
        Ok(Vec::new())
    }
}

fn float_data(t: &onnx::TensorProto) -> Result<Arc<[f32]>> {
    if !t.float_data.is_empty() {
        Ok(Arc::<[f32]>::from(t.float_data.clone()))
    } else if !t.raw_data.is_empty() {
        let n = t.raw_data.len() / 4;
        let mut out = Vec::with_capacity(n);
        for chunk in t.raw_data.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(Arc::<[f32]>::from(out))
    } else {
        Ok(Arc::<[f32]>::from(Vec::<f32>::new()))
    }
}

fn double_data(t: &onnx::TensorProto) -> Result<Arc<[f64]>> {
    if !t.double_data.is_empty() {
        Ok(Arc::<[f64]>::from(t.double_data.clone()))
    } else if !t.raw_data.is_empty() {
        let n = t.raw_data.len() / 8;
        let mut out = Vec::with_capacity(n);
        for chunk in t.raw_data.chunks_exact(8) {
            out.push(f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]));
        }
        Ok(Arc::<[f64]>::from(out))
    } else {
        Ok(Arc::<[f64]>::from(Vec::<f64>::new()))
    }
}

fn int64_data(t: &onnx::TensorProto) -> Result<Arc<[i64]>> {
    Ok(Arc::<[i64]>::from(int64_data_vec(t)?))
}

fn u32_data(t: &onnx::TensorProto) -> Result<Arc<[u32]>> {
    if !t.uint64_data.is_empty() {
        Ok(Arc::<[u32]>::from(
            t.uint64_data.iter().map(|&v| v as u32).collect::<Vec<_>>(),
        ))
    } else if !t.raw_data.is_empty() {
        let n = t.raw_data.len() / 4;
        let mut out = Vec::with_capacity(n);
        for chunk in t.raw_data.chunks_exact(4) {
            out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(Arc::<[u32]>::from(out))
    } else {
        Ok(Arc::<[u32]>::from(Vec::<u32>::new()))
    }
}

fn u32_from_uint8_bool(t: &onnx::TensorProto) -> Result<Arc<[u32]>> {
    if !t.int32_data.is_empty() {
        Ok(Arc::<[u32]>::from(
            t.int32_data.iter().map(|&v| v as u32).collect::<Vec<_>>(),
        ))
    } else if !t.raw_data.is_empty() {
        Ok(Arc::<[u32]>::from(
            t.raw_data.iter().map(|&b| b as u32).collect::<Vec<_>>(),
        ))
    } else {
        Ok(Arc::<[u32]>::from(Vec::<u32>::new()))
    }
}

fn fill_from_value_proto(
    t: &onnx::TensorProto,
    dt: DType,
    shape_dims: &[usize],
    anchor: &mut Option<LazyTensor>,
    device: &Device,
) -> Result<LazyTensor> {
    let n: usize = shape_dims.iter().product();
    match dt {
        DType::F32 => {
            let v: Vec<f32> = if !t.float_data.is_empty() {
                vec![t.float_data[0]; n]
            } else if t.raw_data.len() >= 4 {
                let v0 = f32::from_le_bytes([
                    t.raw_data[0],
                    t.raw_data[1],
                    t.raw_data[2],
                    t.raw_data[3],
                ]);
                vec![v0; n]
            } else {
                vec![0.0f32; n]
            };
            let a = ensure_anchor(anchor, device);
            Ok(a.const_f32_like(v, Shape::from_dims(shape_dims)))
        }
        DType::F64 => {
            let v: Vec<f64> = if !t.double_data.is_empty() {
                vec![t.double_data[0]; n]
            } else if t.raw_data.len() >= 8 {
                let v0 = f64::from_le_bytes([
                    t.raw_data[0],
                    t.raw_data[1],
                    t.raw_data[2],
                    t.raw_data[3],
                    t.raw_data[4],
                    t.raw_data[5],
                    t.raw_data[6],
                    t.raw_data[7],
                ]);
                vec![v0; n]
            } else {
                vec![0.0f64; n]
            };
            let a = ensure_anchor(anchor, device);
            Ok(a.const_f64_like(v, Shape::from_dims(shape_dims)))
        }
        DType::I64 => {
            let v: Vec<i64> = if !t.int64_data.is_empty() {
                vec![t.int64_data[0]; n]
            } else if t.raw_data.len() >= 8 {
                let v0 = i64::from_le_bytes([
                    t.raw_data[0],
                    t.raw_data[1],
                    t.raw_data[2],
                    t.raw_data[3],
                    t.raw_data[4],
                    t.raw_data[5],
                    t.raw_data[6],
                    t.raw_data[7],
                ]);
                vec![v0; n]
            } else {
                vec![0i64; n]
            };
            let a = ensure_anchor(anchor, device);
            Ok(a.const_i64_like(v, Shape::from_dims(shape_dims)))
        }
        other => Err(Error::Msg(format!(
            "ConstantOfShape: dtype {other:?} not supported in sub-port 1"
        ))
        .bt()),
    }
}

fn onnx_dtype_to_fuel(dt: i32) -> Result<DType> {
    let d = DataType::try_from(dt)
        .map_err(|_| Error::Msg(format!("invalid ONNX data_type {dt}")).bt())?;
    match d {
        DataType::Float => Ok(DType::F32),
        DataType::Double => Ok(DType::F64),
        DataType::Float16 => Ok(DType::F16),
        DataType::Bfloat16 => Ok(DType::BF16),
        DataType::Int64 => Ok(DType::I64),
        DataType::Uint32 => Ok(DType::U32),
        DataType::Uint8 | DataType::Bool => Ok(DType::U8),
        other => Err(Error::Msg(format!(
            "ONNX dtype {other:?} not supported in sub-port 1"
        ))
        .bt()),
    }
}

pub(crate) fn get_i64_vec(
    values: &HashMap<String, LazyTensor>,
    i64_cache: &HashMap<String, Vec<i64>>,
    name: &str,
) -> Result<Vec<i64>> {
    if let Some(cached) = i64_cache.get(name) {
        return Ok(cached.clone());
    }
    let t = values.get(name).ok_or_else(|| {
        Error::Msg(format!("missing tensor '{}' (expected i64 vector)", name)).bt()
    })?;
    realize_i64_vec(t)
}

pub(crate) fn realize_i64_vec(t: &LazyTensor) -> Result<Vec<i64>> {
    // Float-only path keeps fuel-graph-cpu adopt off the I64 slot.
    // Known-integer initializers / Constant outputs are read host-side
    // via the i64_cache; this fallback runs only for runtime-computed
    // float tensors that happen to encode integer values.
    match t.dtype() {
        DType::F32 => Ok(t.realize_f32().into_iter().map(|v| v as i64).collect()),
        DType::F64 => Ok(t.realize_f64().into_iter().map(|v| v as i64).collect()),
        other => Err(Error::Msg(format!(
            "expected integer-like tensor for shape/axes/indices; got {other:?} \
             (sub-port 1 routes known integer initializers through a host-side \
             cache; runtime-computed integer tensors are not supported yet)"
        ))
        .bt()),
    }
}

pub(crate) fn get_attr_int(node: &onnx::NodeProto, name: &str) -> Result<i64> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.i)
        .ok_or_else(|| {
            Error::Msg(format!(
                "node '{}' ({}): missing required attribute '{}'",
                node.name, node.op_type, name
            ))
            .bt()
        })
}

pub(crate) fn get_attr_int_opt(node: &onnx::NodeProto, name: &str) -> Option<i64> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.i)
}

pub(crate) fn get_attr_float_opt(node: &onnx::NodeProto, name: &str) -> Option<f32> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.f)
}

pub(crate) fn get_attr_ints_opt<'a>(node: &'a onnx::NodeProto, name: &str) -> Option<&'a [i64]> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.as_slice())
}

/// Sub-port-2 helper: look up a string attribute (returns `None` if
/// the attribute is absent). UTF-8 decoding errors surface as
/// [`Error::Msg`] at build time.
pub(crate) fn get_attr_string_opt(node: &onnx::NodeProto, name: &str) -> Result<Option<String>> {
    match node.attribute.iter().find(|a| a.name == name) {
        None => Ok(None),
        Some(a) => {
            let s = std::str::from_utf8(&a.s)
                .map_err(|e| Error::Msg(format!("attribute '{}': invalid UTF-8 ({e})", name)).bt())?;
            Ok(Some(s.to_string()))
        }
    }
}

pub(crate) fn normalize_axis(axis: i64, rank: usize) -> Result<usize> {
    let n = if axis < 0 { axis + rank as i64 } else { axis };
    if n < 0 || n >= rank as i64 {
        return Err(Error::Msg(format!(
            "axis {axis} out of range for rank {rank}"
        ))
        .bt());
    }
    Ok(n as usize)
}

fn normalize_axis_plus_one(axis: i64, rank: usize) -> Result<usize> {
    // Unsqueeze: axis may be `== rank` (append a trailing dim).
    let n = if axis < 0 { axis + rank as i64 } else { axis };
    if n < 0 || n > rank as i64 {
        return Err(Error::Msg(format!(
            "axis {axis} out of range [-{rank}, {rank}] for unsqueeze"
        ))
        .bt());
    }
    Ok(n as usize)
}

fn classify_op_sub_port(op_type: &str) -> &'static str {
    match op_type {
        // Sub-port 2: convolutions, pooling, padding.
        "Conv" | "ConvTranspose" | "MaxPool" | "AveragePool" | "Pad" | "GlobalAveragePool"
        | "Resize" => "sub-port 2 (conv / pooling / pad)",
        // Sub-port 3: norms + activations + softmax.
        "BatchNormalization"
        | "LayerNormalization"
        | "InstanceNormalization"
        | "Softmax"
        | "LogSoftmax"
        | "Relu"
        | "Gelu"
        | "Sigmoid"
        | "Tanh"
        | "LeakyRelu"
        | "PRelu"
        | "Selu"
        | "HardSwish"
        | "Erf" => "sub-port 3 (norms / activations / softmax)",
        // Sub-port 4: quantization.
        "QLinearMatMul" | "QuantizeLinear" | "DequantizeLinear" | "QLinearConv" => {
            "sub-port 4 (quantized ops)"
        }
        _ => "a future sub-port (unscoped)",
    }
}

// ===========================================================================
// tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

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

    fn attr_int(name: &str, v: i64) -> onnx::AttributeProto {
        onnx::AttributeProto {
            name: name.to_string(),
            r#type: AttributeType::Int as i32,
            i: v,
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

    fn attr_tensor(name: &str, t: onnx::TensorProto) -> onnx::AttributeProto {
        onnx::AttributeProto {
            name: name.to_string(),
            r#type: AttributeType::Tensor as i32,
            t: Some(t),
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

    #[test]
    fn in_memory_graph_matmul_add() {
        // X: [2, 3]; W initializer: [3, 2]; B initializer: [2]; out = X @ W + B
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let b_data: Vec<f32> = vec![1.0, -1.0];

        let graph = onnx::GraphProto {
            node: vec![
                node("MatMul", "mm", &["X", "W"], &["XW"]),
                node("Add", "add", &["XW", "B"], &["Y"]),
            ],
            initializer: vec![
                tp_float("W", &[3, 2], w_data.clone()),
                tp_float("B", &[2], b_data.clone()),
            ],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };

        let mut buf = Vec::new();
        model_from_graph(graph).encode(&mut buf).unwrap();
        let evaluator = LazyOnnxEval::from_bytes(&buf).unwrap();

        let device = Device::cpu();
        let x = LazyTensor::from_f32(x_data.clone(), Shape::from_dims(&[2, 3]), &device);

        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let outputs = evaluator.run(&inputs).unwrap();
        let y = outputs.get("Y").unwrap();
        assert_eq!(y.shape().dims(), &[2, 2]);
        let got = y.realize_f32();

        // Hand-computed expected (row-major):
        //   row0 = [1*0.1+2*0.3+3*0.5, 1*0.2+2*0.4+3*0.6] + B
        //        = [2.2, 2.8] + [1, -1] = [3.2, 1.8]
        //   row1 = [4*0.1+5*0.3+6*0.5, 4*0.2+5*0.4+6*0.6] + B
        //        = [4.9, 6.4] + [1, -1] = [5.9, 5.4]
        let expected = [3.2_f32, 1.8, 5.9, 5.4];
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(*g, *e, 1e-5),
                "mismatch at {i}: got {g} expected {e} (full got={got:?})"
            );
        }
        assert!(got.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn in_memory_graph_reshape_then_reduce() {
        // X (rank-1, len 4) -> Reshape [2,2] -> ReduceMean axis=1, keepdims=0
        let shape_t = tp_int64("shape", &[2], vec![2, 2]);
        let mut reduce_node = node("ReduceMean", "mean", &["R"], &["Y"]);
        reduce_node.attribute.push(attr_ints("axes", vec![1]));
        reduce_node.attribute.push(attr_int("keepdims", 0));

        let graph = onnx::GraphProto {
            node: vec![
                node("Reshape", "rs", &["X", "shape"], &["R"]),
                reduce_node,
            ],
            initializer: vec![shape_t],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };

        let mut buf = Vec::new();
        model_from_graph(graph).encode(&mut buf).unwrap();
        let evaluator = LazyOnnxEval::from_bytes(&buf).unwrap();

        let device = Device::cpu();
        let x = LazyTensor::from_f32(
            vec![1.0f32, 2.0, 3.0, 4.0],
            Shape::from_dims(&[4]),
            &device,
        );
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);
        let outputs = evaluator.run(&inputs).unwrap();
        let y = outputs.get("Y").unwrap();
        // Reshape gives [[1,2],[3,4]]; mean across axis 1 → [1.5, 3.5].
        assert_eq!(y.shape().dims(), &[2]);
        let got = y.realize_f32();
        assert!(approx_eq(got[0], 1.5, 1e-6), "got[0]={}", got[0]);
        assert!(approx_eq(got[1], 3.5, 1e-6), "got[1]={}", got[1]);
    }

    #[test]
    fn unsupported_op_errors_cleanly() {
        // Sub-port 3 territory: a softmax-class op still routes through the
        // sub-port pointer pattern.
        let graph = onnx::GraphProto {
            node: vec![node("Softmax", "sm", &["X"], &["Y"])],
            input: vec![value_info("X")],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let mut buf = Vec::new();
        model_from_graph(graph).encode(&mut buf).unwrap();
        let evaluator = LazyOnnxEval::from_bytes(&buf).unwrap();

        let device = Device::cpu();
        let x = LazyTensor::from_f32(
            vec![1.0f32; 4],
            Shape::from_dims(&[1, 4]),
            &device,
        );
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), x);

        let err = evaluator.run(&inputs).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Softmax"), "error must mention op name: {msg}");
        assert!(
            msg.contains("sub-port 3"),
            "error must mention target sub-port: {msg}"
        );
    }

    #[test]
    fn initializer_constants_load_correctly() {
        // Graph: Identity passthrough of initializer 'W'. Verifies that
        // (a) initializers land in the symbol table, (b) their values
        // arrive intact through realize.
        let w = tp_float("W", &[3], vec![7.0, -1.5, 0.25]);
        let graph = onnx::GraphProto {
            node: vec![node("Identity", "id", &["W"], &["Y"])],
            initializer: vec![w],
            input: vec![],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let mut buf = Vec::new();
        model_from_graph(graph).encode(&mut buf).unwrap();
        let evaluator = LazyOnnxEval::from_bytes(&buf).unwrap();
        let outputs = evaluator.run(&HashMap::new()).unwrap();
        let y = outputs.get("Y").unwrap();
        assert_eq!(y.shape().dims(), &[3]);
        let got = y.realize_f32();
        assert!(approx_eq(got[0], 7.0, 1e-7));
        assert!(approx_eq(got[1], -1.5, 1e-7));
        assert!(approx_eq(got[2], 0.25, 1e-7));
    }

    #[test]
    fn constant_node_loads_value_attribute() {
        let const_tensor = tp_float("c", &[2], vec![3.5, -2.0]);
        let mut const_node = node("Constant", "k", &[], &["C"]);
        const_node.attribute.push(attr_tensor("value", const_tensor));

        let graph = onnx::GraphProto {
            node: vec![const_node, node("Identity", "id", &["C"], &["Y"])],
            input: vec![],
            output: vec![value_info("Y")],
            ..Default::default()
        };
        let mut buf = Vec::new();
        model_from_graph(graph).encode(&mut buf).unwrap();
        let evaluator = LazyOnnxEval::from_bytes(&buf).unwrap();
        let outputs = evaluator.run(&HashMap::new()).unwrap();
        let got = outputs.get("Y").unwrap().realize_f32();
        assert_eq!(got.len(), 2);
        assert!(approx_eq(got[0], 3.5, 1e-7));
        assert!(approx_eq(got[1], -2.0, 1e-7));
    }
}

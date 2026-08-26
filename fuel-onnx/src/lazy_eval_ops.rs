// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lazy-graph ONNX evaluator — sub-port 5: elementwise, comparison,
//! logical, and shape-manipulation ops.
//!
//! Hooks into [`crate::lazy_eval::OnnxEval`]'s dispatch chain the same way
//! [`crate::lazy_eval_conv`] and [`crate::lazy_eval_norm`] do: [`try_dispatch`]
//! returns `Ok(true)` when it handled the node, `Ok(false)` to fall through.
//!
//! These ops were previously reachable only through the EAGER evaluator
//! (`eval.rs`), which B6 retired along with the eager `Tensor`. Everything here
//! is a thin adapter onto an existing [`Tensor`] primitive — the ONNX
//! semantics (broadcasting rules, attribute defaults, opset-version input-vs-
//! attribute migrations) are the actual content.

use crate::lazy_eval::{
    ensure_anchor, get_attr_float_opt, get_attr_int, get_attr_ints_opt, normalize_axis,
    realize_i64_vec, set_output,
};
use crate::onnx;
use fuel::lazy::Tensor;
use fuel::{DType, Device, Error, Result, Shape};
use std::collections::HashMap;

/// Broadcast two operands to their common shape, ONNX/NumPy style.
///
/// The comparison and logical primitives on [`Tensor`] are *same-shape*
/// (`eq`, `lt`, `maximum`, …) while ONNX specifies numpy broadcasting for all
/// of them, so every binary op here goes through this first. Right-aligns the
/// two shapes, takes the max extent per axis, and rejects a genuine mismatch
/// rather than silently picking one side.
fn broadcast_pair(a: &Tensor, b: &Tensor, op: &str) -> Result<(Tensor, Tensor)> {
    let (da, db) = (a.shape().dims().to_vec(), b.shape().dims().to_vec());
    if da == db {
        return Ok((a.clone(), b.clone()));
    }
    let rank = da.len().max(db.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        // right-align: axis i of the result maps to the tail of each input
        let ea = if i + da.len() >= rank {
            da[i + da.len() - rank]
        } else {
            1
        };
        let eb = if i + db.len() >= rank {
            db[i + db.len() - rank]
        } else {
            1
        };
        let e = match (ea, eb) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            (x, y) => {
                return Err(Error::Msg(format!(
                    "{op}: operands are not broadcast-compatible on axis {i}: {x} vs {y} \
                     (shapes {da:?} and {db:?})"
                ))
                .bt());
            }
        };
        out.push(e);
    }
    let shape = Shape::from_dims(&out);
    Ok((a.broadcast_to(shape.clone())?, b.broadcast_to(shape)?))
}

/// Normalize an arbitrary numeric input to a **0.0/1.0 F32** mask.
///
/// F32 rather than a bool/U8 dtype for a concrete backend reason: the CPU
/// backend registers the comparison and min/max kernels for FLOAT dtypes only
/// (`MinimumElementwise` exists for F16/F32/F64/BF16 and *not* U8), while the
/// comparison ops themselves *produce* U8. Chaining logical ops on their raw
/// output therefore dies at realize with "no backend supports minimum on
/// [U8, U8, U8]". Doing the algebra in F32 and casting the final result back to
/// U8 keeps every intermediate on a kernel that exists.
fn nonzero_f32(x: &Tensor) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let zero = xf.zeros_like()?;
    // `ne` yields U8; widen straight back to F32 so callers can compose.
    xf.ne(&zero)?.to_dtype(DType::F32)
}

/// Fetch a required positional input, erroring with the node name.
fn input(node: &onnx::NodeProto, values: &HashMap<String, Tensor>, idx: usize) -> Result<Tensor> {
    let name = node.input.get(idx).ok_or_else(|| {
        Error::Msg(format!(
            "ONNX op '{}' (node '{}'): missing required input #{idx}",
            node.op_type, node.name
        ))
        .bt()
    })?;
    values
        .get(name)
        .cloned()
        .ok_or_else(|| Error::Msg(format!("missing input '{name}' for node '{}'", node.name)).bt())
}

/// An optional positional input. ONNX encodes "absent" both as a short input
/// list AND as an empty-string name in the middle of the list (Clip, Pad,
/// Resize all do this), so both must be treated as absent.
fn opt_input(
    node: &onnx::NodeProto,
    values: &HashMap<String, Tensor>,
    idx: usize,
) -> Option<Tensor> {
    let name = node.input.get(idx)?;
    if name.is_empty() {
        return None;
    }
    values.get(name).cloned()
}

/// Realize an optional scalar input to f64 — Clip's min/max and Range's
/// start/limit/delta are all rank-0 tensors rather than attributes in modern
/// opsets.
fn scalar_f64(t: &Tensor, what: &str) -> Result<f64> {
    let v = t.realize_f32();
    match v.len() {
        1 => Ok(v[0] as f64),
        n => Err(Error::Msg(format!("{what}: expected a scalar, got {n} elements")).bt()),
    }
}

pub(crate) fn try_dispatch(
    node: &onnx::NodeProto,
    values: &mut HashMap<String, Tensor>,
    device: &Device,
    anchor: &mut Option<Tensor>,
) -> Result<bool> {
    match node.op_type.as_str() {
        // ---- unary elementwise: direct Tensor primitives ----
        "Abs" => set_output(node, 0, input(node, values, 0)?.abs(), values)?,
        "Neg" => set_output(node, 0, input(node, values, 0)?.neg(), values)?,
        "Sign" => set_output(node, 0, input(node, values, 0)?.sign(), values)?,
        "Ceil" => set_output(node, 0, input(node, values, 0)?.ceil(), values)?,
        "Floor" => set_output(node, 0, input(node, values, 0)?.floor(), values)?,
        "Exp" => set_output(node, 0, input(node, values, 0)?.exp(), values)?,
        "Log" => set_output(node, 0, input(node, values, 0)?.log(), values)?,
        "Sqrt" => set_output(node, 0, input(node, values, 0)?.sqrt(), values)?,
        "Sin" => set_output(node, 0, input(node, values, 0)?.sin(), values)?,
        "Cos" => set_output(node, 0, input(node, values, 0)?.cos(), values)?,

        // ---- comparisons: broadcast, then same-shape primitive ----
        "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual" => {
            let (a, b) = broadcast_pair(
                &input(node, values, 0)?,
                &input(node, values, 1)?,
                &node.op_type,
            )?;
            let y = match node.op_type.as_str() {
                "Equal" => a.eq(&b)?,
                "Greater" => a.gt(&b)?,
                "GreaterOrEqual" => a.ge(&b)?,
                "Less" => a.lt(&b)?,
                _ => a.le(&b)?,
            };
            set_output(node, 0, y, values)?
        }

        // ---- logical: normalize to a 0/1 mask, then min/max/ne ----
        "Not" => {
            let x = nonzero_f32(&input(node, values, 0)?)?;
            let zero = x.zeros_like()?;
            // `eq` already returns U8, which is the ONNX bool representation.
            set_output(node, 0, x.eq(&zero)?, values)?
        }
        "And" | "Or" | "Xor" => {
            let a = nonzero_f32(&input(node, values, 0)?)?;
            let b = nonzero_f32(&input(node, values, 1)?)?;
            let (a, b) = broadcast_pair(&a, &b, &node.op_type)?;
            // On a 0.0/1.0 F32 mask: AND is min, OR is max, XOR is inequality.
            // min/max stay in F32 (no U8 kernel exists) and are cast to U8 at
            // the end; `ne` already produces U8.
            let y = match node.op_type.as_str() {
                "And" => a.minimum(&b)?.to_dtype(DType::U8)?,
                "Or" => a.maximum(&b)?.to_dtype(DType::U8)?,
                _ => a.ne(&b)?,
            };
            set_output(node, 0, y, values)?
        }

        // ---- variadic min/max ----
        "Min" | "Max" => {
            if node.input.is_empty() {
                return Err(Error::Msg(format!(
                    "{}: node '{}' has no inputs",
                    node.op_type, node.name
                ))
                .bt());
            }
            let mut acc = input(node, values, 0)?;
            for i in 1..node.input.len() {
                let (a, b) = broadcast_pair(&acc, &input(node, values, i)?, &node.op_type)?;
                acc = if node.op_type == "Min" {
                    a.minimum(&b)?
                } else {
                    a.maximum(&b)?
                };
            }
            set_output(node, 0, acc, values)?
        }

        "Pow" => {
            let (a, b) = broadcast_pair(&input(node, values, 0)?, &input(node, values, 1)?, "Pow")?;
            set_output(node, 0, a.pow(&b)?, values)?
        }

        // ---- Where: cond ? x : y ----
        "Where" => {
            // `where_cond` wants a Bool selector (GAP-168(c)); the F32 nonzero
            // mask (0.0/1.0) casts to Bool.
            let cond = nonzero_f32(&input(node, values, 0)?)?.to_dtype(DType::Bool)?;
            let x = input(node, values, 1)?;
            let y = input(node, values, 2)?;
            // All three broadcast together; do it pairwise against the result
            // extent so a rank-0 cond against rank-2 branches works.
            let (x, y) = broadcast_pair(&x, &y, "Where")?;
            let (cond, _) = broadcast_pair(&cond, &x, "Where")?;
            set_output(node, 0, cond.where_cond(&x, &y)?, values)?
        }

        // ---- Clip: min/max as inputs (opset 11+) or attributes (opset <11) ----
        "Clip" => {
            let x = input(node, values, 0)?;
            let lo = match opt_input(node, values, 1) {
                Some(t) => Some(scalar_f64(&t, "Clip: min")?),
                None => get_attr_float_opt(node, "min").map(|v| v as f64),
            };
            let hi = match opt_input(node, values, 2) {
                Some(t) => Some(scalar_f64(&t, "Clip: max")?),
                None => get_attr_float_opt(node, "max").map(|v| v as f64),
            };
            let lo = lo.unwrap_or(f64::NEG_INFINITY);
            let hi = hi.unwrap_or(f64::INFINITY);
            set_output(node, 0, x.clamp(lo, hi), values)?
        }

        // ---- shape metadata: produce i64 tensors, not shapes ----
        "Shape" => {
            let x = input(node, values, 0)?;
            let dims: Vec<i64> = x.shape().dims().iter().map(|&d| d as i64).collect();
            let rank = dims.len() as i64;
            // opset 15 added start/end slicing of the shape vector
            let norm = |v: i64| -> usize {
                let v = if v < 0 { v + rank } else { v };
                v.clamp(0, rank) as usize
            };
            let start = match node.attribute.iter().find(|a| a.name == "start") {
                Some(a) => norm(a.i),
                None => 0,
            };
            let end = match node.attribute.iter().find(|a| a.name == "end") {
                Some(a) => norm(a.i),
                None => rank as usize,
            };
            let slice: Vec<i64> = if start <= end {
                dims[start..end].to_vec()
            } else {
                vec![]
            };
            let a = ensure_anchor(anchor, device);
            let n = slice.len();
            set_output(
                node,
                0,
                a.const_i64_like(slice, Shape::from_dims(&[n])),
                values,
            )?
        }
        "Size" => {
            let x = input(node, values, 0)?;
            let n = x.shape().elem_count() as i64;
            let a = ensure_anchor(anchor, device);
            set_output(
                node,
                0,
                a.const_i64_like(vec![n], Shape::from_dims(&[] as &[usize])),
                values,
            )?
        }

        // ---- Expand: broadcast to a runtime shape ----
        "Expand" => {
            let x = input(node, values, 0)?;
            let want = realize_i64_vec(&input(node, values, 1)?)?;
            // ONNX Expand allows the target to be *smaller rank* than x, and a
            // target extent of 1 means "keep x's extent" — so the result shape
            // is the broadcast of the two, not simply `want`.
            let xd = x.shape().dims().to_vec();
            let rank = xd.len().max(want.len());
            let mut out = Vec::with_capacity(rank);
            for i in 0..rank {
                let ex = if i + xd.len() >= rank {
                    xd[i + xd.len() - rank]
                } else {
                    1
                };
                let ew = if i + want.len() >= rank {
                    let w = want[i + want.len() - rank];
                    if w < 0 {
                        return Err(Error::Msg(format!(
                            "Expand: negative extent {w} in target shape {want:?}"
                        ))
                        .bt());
                    }
                    w as usize
                } else {
                    1
                };
                out.push(match (ex, ew) {
                    (a, b) if a == b => a,
                    (1, b) => b,
                    (a, 1) => a,
                    (a, b) => {
                        return Err(Error::Msg(format!(
                            "Expand: cannot broadcast extent {a} to {b} on axis {i}"
                        ))
                        .bt());
                    }
                });
            }
            set_output(node, 0, x.broadcast_to(Shape::from_dims(&out))?, values)?
        }

        // ---- Tile: repeat along each axis ----
        "Tile" => {
            let x = input(node, values, 0)?;
            let reps = realize_i64_vec(&input(node, values, 1)?)?;
            if reps.len() != x.rank() {
                return Err(Error::Msg(format!(
                    "Tile: repeats has {} entries but input is rank {}",
                    reps.len(),
                    x.rank()
                ))
                .bt());
            }
            if reps.iter().any(|&r| r < 0) {
                return Err(Error::Msg(format!("Tile: negative repeat in {reps:?}")).bt());
            }
            let reps: Vec<usize> = reps.iter().map(|&r| r as usize).collect();
            set_output(node, 0, x.repeat(Shape::from_dims(&reps))?, values)?
        }

        // ---- Trilu: upper/lower triangular ----
        "Trilu" => {
            let x = input(node, values, 0)?;
            let k = match opt_input(node, values, 1) {
                Some(t) => realize_i64_vec(&t)?.first().copied().unwrap_or(0),
                None => 0,
            };
            // `upper` defaults to 1 per the ONNX spec
            let upper = node
                .attribute
                .iter()
                .find(|a| a.name == "upper")
                .map(|a| a.i)
                .unwrap_or(1);
            let y = if upper != 0 { x.triu(k)? } else { x.tril(k)? };
            set_output(node, 0, y, values)?
        }

        // ---- CumSum ----
        "CumSum" => {
            let x = input(node, values, 0)?;
            let axis_v = realize_i64_vec(&input(node, values, 1)?)?;
            let axis_raw = axis_v.first().copied().unwrap_or(0);
            let axis = normalize_axis(axis_raw, x.rank())?;
            let exclusive = node
                .attribute
                .iter()
                .find(|a| a.name == "exclusive")
                .map(|a| a.i)
                .unwrap_or(0);
            let reverse = node
                .attribute
                .iter()
                .find(|a| a.name == "reverse")
                .map(|a| a.i)
                .unwrap_or(0);
            if exclusive != 0 || reverse != 0 {
                return Err(Error::Msg(format!(
                    "CumSum: exclusive={exclusive} reverse={reverse} not supported \
                     (only the inclusive forward scan is wired to Tensor::cumsum)"
                ))
                .bt());
            }
            set_output(node, 0, x.cumsum(axis)?, values)?
        }

        // ---- ArgMax / ArgMin ----
        "ArgMax" | "ArgMin" => {
            let x = input(node, values, 0)?;
            let axis_raw = node
                .attribute
                .iter()
                .find(|a| a.name == "axis")
                .map(|a| a.i)
                .unwrap_or(0);
            let axis = normalize_axis(axis_raw, x.rank())?;
            let keepdims = node
                .attribute
                .iter()
                .find(|a| a.name == "keepdims")
                .map(|a| a.i)
                .unwrap_or(1);
            let select_last = node
                .attribute
                .iter()
                .find(|a| a.name == "select_last_index")
                .map(|a| a.i)
                .unwrap_or(0);
            if select_last != 0 {
                return Err(Error::Msg(format!(
                    "{}: select_last_index=1 not supported",
                    node.op_type
                ))
                .bt());
            }
            // `argmax_dim`/`argmin_dim` REMOVE the reduced dim and emit U32;
            // ONNX specifies I64 indices and keepdims=1 by default.
            let idx = if node.op_type == "ArgMax" {
                x.argmax_dim(axis)?
            } else {
                x.argmin_dim(axis)?
            };
            let idx = idx.to_dtype(DType::I64)?;
            let y = if keepdims != 0 {
                idx.unsqueeze(axis)?
            } else {
                idx
            };
            set_output(node, 0, y, values)?
        }

        // ---- ReduceL2: sqrt(sum(x^2)) ----
        "ReduceL2" => {
            let x = input(node, values, 0)?;
            let keepdims = node
                .attribute
                .iter()
                .find(|a| a.name == "keepdims")
                .map(|a| a.i)
                .unwrap_or(1);
            let axes: Vec<usize> = match get_attr_ints_opt(node, "axes") {
                Some(a) => a
                    .iter()
                    .map(|&v| normalize_axis(v, x.rank()))
                    .collect::<Result<Vec<_>>>()?,
                None => match opt_input(node, values, 1) {
                    Some(t) => realize_i64_vec(&t)?
                        .iter()
                        .map(|&v| normalize_axis(v, x.rank()))
                        .collect::<Result<Vec<_>>>()?,
                    None => (0..x.rank()).collect(),
                },
            };
            let mut acc = x.sqr();
            // reduce high-to-low so earlier axis indices stay valid
            let mut sorted = axes.clone();
            sorted.sort_unstable();
            sorted.dedup();
            for &ax in sorted.iter().rev() {
                acc = acc.sum_keepdim(ax)?;
            }
            let mut y = acc.sqrt();
            if keepdims == 0 {
                for &ax in sorted.iter().rev() {
                    y = y.squeeze(ax)?;
                }
            }
            set_output(node, 0, y, values)?
        }

        // ---- Dropout: inference-mode identity ----
        "Dropout" => {
            // At inference `training_mode` is false (or absent) and Dropout is
            // the identity. A true training-mode Dropout needs the RNG seam
            // (Op::RandomBits), which is a separate program.
            let training = opt_input(node, values, 2)
                .map(|t| scalar_f64(&t, "Dropout: training_mode").map(|v| v != 0.0))
                .transpose()?
                .unwrap_or(false);
            if training {
                return Err(Error::Msg(format!(
                    "Dropout: node '{}' requests training_mode=true, which needs the \
                     RNG generator seam (Op::RandomBits); inference-mode only for now",
                    node.name
                ))
                .bt());
            }
            let x = input(node, values, 0)?;
            set_output(node, 0, x.clone(), values)?;
            // optional `mask` output: all-ones, same shape
            if node.output.len() > 1 && !node.output[1].is_empty() {
                set_output(node, 1, x.ones_like()?, values)?;
            }
        }

        // ---- activations that are plain arithmetic ----
        "Selu" => {
            let x = input(node, values, 0)?;
            let alpha = get_attr_float_opt(node, "alpha").unwrap_or(1.673_263_2);
            let gamma = get_attr_float_opt(node, "gamma").unwrap_or(1.050_701);
            // gamma * (x > 0 ? x : alpha*(exp(x)-1))
            let zero = x.zeros_like()?;
            let pos = x.gt(&zero)?;
            let neg_branch = x.exp().affine(alpha as f64, -(alpha as f64));
            let y = pos.where_cond(&x, &neg_branch)?;
            set_output(node, 0, y.affine(gamma as f64, 0.0), values)?
        }
        "HardSwish" => {
            // x * clamp(x/6 + 0.5, 0, 1)
            let x = input(node, values, 0)?;
            let inner = x.affine(1.0 / 6.0, 0.5).clamp(0.0, 1.0);
            set_output(node, 0, x.mul(&inner)?, values)?
        }
        "PRelu" => {
            // x > 0 ? x : slope * x   (slope broadcasts against x)
            let x = input(node, values, 0)?;
            let slope = input(node, values, 1)?;
            let (xb, slope) = broadcast_pair(&x, &slope, "PRelu")?;
            let zero = xb.zeros_like()?;
            let pos = xb.gt(&zero)?;
            let neg_branch = xb.mul(&slope)?;
            set_output(node, 0, pos.where_cond(&xb, &neg_branch)?, values)?
        }

        // ---- Gemm: Y = alpha * A' @ B' + beta * C ----
        "Gemm" => {
            let a = input(node, values, 0)?;
            let b = input(node, values, 1)?;
            let alpha = get_attr_float_opt(node, "alpha").unwrap_or(1.0) as f64;
            let beta = get_attr_float_opt(node, "beta").unwrap_or(1.0) as f64;
            let ta = node
                .attribute
                .iter()
                .find(|x| x.name == "transA")
                .map(|x| x.i)
                .unwrap_or(0);
            let tb = node
                .attribute
                .iter()
                .find(|x| x.name == "transB")
                .map(|x| x.i)
                .unwrap_or(0);
            let a = if ta != 0 { a.t()? } else { a };
            let b = if tb != 0 { b.t()? } else { b };
            let mut y = a.matmul(&b)?;
            if alpha != 1.0 {
                y = y.affine(alpha, 0.0);
            }
            // C is optional; when present it broadcasts against the product.
            if let Some(c) = opt_input(node, values, 2) {
                let c = if beta != 1.0 { c.affine(beta, 0.0) } else { c };
                let (y2, c2) = broadcast_pair(&y, &c, "Gemm")?;
                y = y2.add(&c2)?;
            }
            set_output(node, 0, y, values)?
        }

        // ---- Range: start, limit, delta (all rank-0 inputs) ----
        "Range" => {
            let start = scalar_f64(&input(node, values, 0)?, "Range: start")?;
            let limit = scalar_f64(&input(node, values, 1)?, "Range: limit")?;
            let delta = scalar_f64(&input(node, values, 2)?, "Range: delta")?;
            if delta == 0.0 {
                return Err(Error::Msg("Range: delta must be non-zero".into()).bt());
            }
            // ONNX: n = max(ceil((limit - start) / delta), 0)
            let n = (((limit - start) / delta).ceil()).max(0.0) as usize;
            let dtype = input(node, values, 0)?.dtype();
            // NOT `Tensor::arange` — that mints a NEW graph and the result
            // would not combine with anything else in this evaluation. Hang the
            // sequence off the anchor instead.
            let a = ensure_anchor(anchor, device);
            let seq: Vec<f32> = (0..n).map(|i| (start + delta * i as f64) as f32).collect();
            let y = a
                .const_f32_like(seq, Shape::from_dims(&[n]))
                .to_dtype(dtype)?;
            set_output(node, 0, y, values)?
        }

        // ---- GatherElements: per-element gather along one axis ----
        "GatherElements" => {
            let x = input(node, values, 0)?;
            let idx = input(node, values, 1)?;
            let axis_raw = node
                .attribute
                .iter()
                .find(|a| a.name == "axis")
                .map(|a| a.i)
                .unwrap_or(0);
            let axis = normalize_axis(axis_raw, x.rank())?;
            // `gather` is exactly ONNX GatherElements (index tensor has the
            // output's shape); `index_select` is plain ONNX Gather. Do not
            // confuse the two — they differ in index rank.
            let idx = idx.to_dtype(DType::U32)?;
            set_output(node, 0, x.gather(axis, &idx)?, values)?
        }

        // ---- OneHot ----
        "OneHot" => {
            let indices = input(node, values, 0)?;
            let depth = scalar_f64(&input(node, values, 1)?, "OneHot: depth")? as i64;
            if depth <= 0 {
                return Err(
                    Error::Msg(format!("OneHot: depth must be positive, got {depth}")).bt(),
                );
            }
            let vals = input(node, values, 2)?.realize_f32();
            if vals.len() != 2 {
                return Err(Error::Msg(format!(
                    "OneHot: values must have exactly 2 entries [off, on], got {}",
                    vals.len()
                ))
                .bt());
            }
            let (off, on) = (vals[0] as f64, vals[1] as f64);
            let axis_raw = node
                .attribute
                .iter()
                .find(|a| a.name == "axis")
                .map(|a| a.i)
                .unwrap_or(-1);
            // The new axis may be appended, so normalize against rank+1.
            let rank1 = indices.rank() + 1;
            let axis = if axis_raw < 0 {
                (axis_raw + rank1 as i64) as usize
            } else {
                axis_raw as usize
            };
            if axis >= rank1 {
                return Err(Error::Msg(format!(
                    "OneHot: axis {axis_raw} out of range for output rank {rank1}"
                ))
                .bt());
            }
            // one_hot[..., d, ...] = (indices == d) ? on : off
            let idx_f = indices.to_dtype(DType::F32)?.unsqueeze(axis)?;
            // NOT `Tensor::arange` — it mints a NEW graph, and the `eq`
            // below would then fail with a cross-graph error. Build the ramp as
            // a constant on the indices' own graph.
            let mut ramp_dims = vec![1usize; rank1];
            ramp_dims[axis] = depth as usize;
            let ramp_data: Vec<f32> = (0..depth).map(|d| d as f32).collect();
            let ramp = idx_f.const_f32_like(ramp_data, Shape::from_dims(&ramp_dims));
            let (a, b) = broadcast_pair(&idx_f, &ramp, "OneHot")?;
            let mask = a.eq(&b)?.to_dtype(DType::F32)?;
            // mask*(on-off) + off
            set_output(node, 0, mask.affine(on - off, off), values)?
        }

        // ---- Slice ----
        "Slice" => {
            let x = input(node, values, 0)?;
            // opset 10+ passes starts/ends/axes/steps as inputs; opset 1 used
            // attributes. Support both.
            let (starts, ends, axes, steps) = if node.input.len() >= 3 {
                let starts = realize_i64_vec(&input(node, values, 1)?)?;
                let ends = realize_i64_vec(&input(node, values, 2)?)?;
                let axes = match opt_input(node, values, 3) {
                    Some(t) => realize_i64_vec(&t)?,
                    None => (0..starts.len() as i64).collect(),
                };
                let steps = match opt_input(node, values, 4) {
                    Some(t) => realize_i64_vec(&t)?,
                    None => vec![1; starts.len()],
                };
                (starts, ends, axes, steps)
            } else {
                let starts = get_attr_ints_opt(node, "starts")
                    .ok_or_else(|| Error::Msg("Slice: missing 'starts'".into()).bt())?
                    .to_vec();
                let ends = get_attr_ints_opt(node, "ends")
                    .ok_or_else(|| Error::Msg("Slice: missing 'ends'".into()).bt())?
                    .to_vec();
                let axes = get_attr_ints_opt(node, "axes")
                    .map(|a| a.to_vec())
                    .unwrap_or_else(|| (0..starts.len() as i64).collect());
                let steps = vec![1i64; starts.len()];
                (starts, ends, axes, steps)
            };
            if steps.iter().any(|&s| s != 1) {
                return Err(Error::Msg(format!(
                    "Slice: step != 1 is not supported (got {steps:?}); \
                     Tensor::narrow is contiguous-only"
                ))
                .bt());
            }
            let mut y = x;
            for (i, &ax_raw) in axes.iter().enumerate() {
                let ax = normalize_axis(ax_raw, y.rank())?;
                let extent = y.dim(ax)? as i64;
                // ONNX clamps out-of-range and folds negatives; INT_MAX is the
                // idiomatic "to the end" sentinel and must not overflow.
                let clamp = |v: i64| -> i64 {
                    let v = if v < 0 { v + extent } else { v };
                    v.clamp(0, extent)
                };
                let s = clamp(starts[i]);
                let e = clamp(ends[i]);
                let len = (e - s).max(0);
                y = y.narrow(ax, s as usize, len as usize)?;
            }
            set_output(node, 0, y, values)?
        }

        _ => return Ok(false),
    }
    Ok(true)
}

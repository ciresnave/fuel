//! `primitive_shape` — the single source of truth for primitive-Op shape+dtype
//! inference (Convergence Increment A). Called by BOTH the `Tensor` builders
//! (lib.rs) and the runtime `emit` re-emitter (runtime_fused.rs) so there is
//! exactly one place that answers "what does this primitive Op produce". Reads
//! params off the `Op` variant; never panics — a malformed op/shape is an `Err`.
use crate::Op;
use fuel_ir::{DType, Error, Shape};

fn need<'a>(shapes: &'a [Shape], n: usize, op: &Op) -> Result<&'a [Shape], Error> {
    if shapes.len() < n {
        return Err(Error::Msg(format!(
            "primitive_shape: {op:?} needs {n} input shape(s), got {}",
            shapes.len()
        ))
        .bt());
    }
    Ok(shapes)
}

/// The single source of truth for primitive-Op shape+dtype inference. Given a
/// primitive [`Op`] (its params live on the variant) plus its input shapes and
/// dtypes, returns the `(output_shape, output_dtype)` the op produces. Reads
/// params off the `Op`, not off `OpAttrs`. Never panics: a malformed op/shape,
/// a leaf/bookkeeping op with no pure inference, or a higher-order/basis-gap op
/// is an honest `Err`.
pub fn primitive_shape(
    op: &Op,
    input_shapes: &[Shape],
    input_dtypes: &[DType],
) -> Result<(Shape, DType), Error> {
    use Op::*;
    // Helper closure reused below for the elementwise-preserving group.
    let elem0 = || -> Result<(Shape, DType), Error> {
        need(input_shapes, 1, op)?;
        Ok((input_shapes[0].clone(), input_dtypes[0]))
    };
    Ok(match op {
        // --- elementwise unary + binary + scalar-param (shape=in[0], dtype=in[0]) ---
        Add | Sub | Mul | Div | Maximum | Minimum | Pow | Rem
        | Neg | Abs | Sqr | Sqrt | Rsqrt | Recip | Exp | Log | Sin | Cos
        | Tanh | Sigmoid | Silu | Gelu | GeluErf | Relu | Erf | Step
        | Floor | Ceil | Round | Sign | Contiguize
        | AddScalar(_) | MulScalar(_) | PowI(_) | Clamp { .. }
        | CumSum { .. } | Flip { .. } | Roll { .. } | Triu { .. } | Tril { .. }
        | LogSoftmaxLastDim | LogSoftmaxLastDimBackward | MaskedFill { .. } => elem0()?,

        // --- comparison → U8 ---
        Equal | Ne | Lt | Le | Gt | Ge => {
            need(input_shapes, 1, op)?;
            (input_shapes[0].clone(), DType::U8)
        }

        // --- Where: shape = cond (in[0]), dtype = a (in[1]) ---
        Where => {
            need(input_shapes, 2, op)?;
            (input_shapes[0].clone(), input_dtypes[1])
        }

        // --- dtype-changing ---
        Cast(dt) => {
            need(input_shapes, 1, op)?;
            (input_shapes[0].clone(), *dt)
        }

        // --- shape/layout carrying an explicit target shape on the variant ---
        Reshape(sh) | BroadcastTo(sh) | ReduceSumTo(sh) | ReduceMaxTo(sh) => {
            need(input_shapes, 1, op)?;
            (sh.clone(), input_dtypes[0])
        }

        Transpose => {
            need(input_shapes, 1, op)?;
            let d = input_shapes[0].dims();
            if d.len() < 2 {
                return Err(Error::Msg(format!(
                    "primitive_shape: Transpose needs rank>=2, got {d:?}"
                ))
                .bt());
            }
            let mut out = d.to_vec();
            let r = out.len();
            out.swap(r - 2, r - 1);
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Permute(axes) => {
            need(input_shapes, 1, op)?;
            let d = input_shapes[0].dims();
            if axes.len() != d.len() || axes.iter().any(|&a| a >= d.len()) {
                return Err(Error::Msg(format!(
                    "primitive_shape: Permute axes {axes:?} invalid for {d:?}"
                ))
                .bt());
            }
            let out: Vec<usize> = axes.iter().map(|&a| d[a]).collect();
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Unsqueeze { dim } => {
            need(input_shapes, 1, op)?;
            let d = input_shapes[0].dims();
            if *dim > d.len() {
                return Err(Error::Msg(format!(
                    "primitive_shape: Unsqueeze dim {dim} > rank {}",
                    d.len()
                ))
                .bt());
            }
            let mut out = d.to_vec();
            out.insert(*dim, 1);
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Squeeze { dim } => {
            need(input_shapes, 1, op)?;
            let d = input_shapes[0].dims();
            if *dim >= d.len() {
                return Err(Error::Msg(format!(
                    "primitive_shape: Squeeze dim {dim} >= rank {}",
                    d.len()
                ))
                .bt());
            }
            let out: Vec<usize> = d.iter().enumerate().filter(|(i, _)| *i != *dim).map(|(_, &v)| v).collect();
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Slice { dim, start, len } => {
            need(input_shapes, 1, op)?;
            let d = input_shapes[0].dims();
            if *dim >= d.len() || start + len > d[*dim] {
                return Err(Error::Msg(format!(
                    "primitive_shape: Slice{{dim:{dim},start:{start},len:{len}}} invalid for {d:?}"
                ))
                .bt());
            }
            let mut out = d.to_vec();
            out[*dim] = *len;
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Concat { dim } => {
            need(input_shapes, 2, op)?;
            let a = input_shapes[0].dims();
            let b = input_shapes[1].dims();
            if *dim >= a.len() || a.len() != b.len() {
                return Err(Error::Msg(format!(
                    "primitive_shape: Concat dim {dim} invalid for {a:?} / {b:?}"
                ))
                .bt());
            }
            let mut out = a.to_vec();
            out[*dim] = a[*dim] + b[*dim];
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Pad { padding, .. } => {
            need(input_shapes, 1, op)?;
            let d = input_shapes[0].dims();
            if padding.len() != d.len() {
                return Err(Error::Msg(format!(
                    "primitive_shape: Pad padding {padding:?} rank != {d:?}"
                ))
                .bt());
            }
            let out: Vec<usize> = d.iter().zip(padding).map(|(&x, (b, a))| x + b + a).collect();
            (Shape::from_dims(&out), input_dtypes[0])
        }

        // --- rank-reducing reductions ---
        SumDim(dim) | MaxDim(dim) | MinDim(dim) | MeanDim(dim) => {
            need(input_shapes, 1, op)?;
            (reduce_remove(input_shapes[0].dims(), *dim, op)?, input_dtypes[0])
        }
        ArgMaxDim(dim) | ArgMinDim(dim) => {
            need(input_shapes, 1, op)?;
            (reduce_remove(input_shapes[0].dims(), *dim, op)?, DType::U32)
        }
        SumAll | MaxAll | MinAll | MeanAll => {
            need(input_shapes, 1, op)?;
            (Shape::from_dims(&[]), input_dtypes[0])
        }

        // --- matmul (same-rank operands; the builder pre-broadcasts) ---
        MatMul => {
            need(input_shapes, 2, op)?;
            let l = input_shapes[0].dims();
            let r = input_shapes[1].dims();
            if l.len() < 2 || r.len() != l.len() {
                return Err(Error::Msg(format!(
                    "primitive_shape: MatMul needs same-rank>=2 operands, got {l:?} / {r:?}"
                ))
                .bt());
            }
            let rank = l.len();
            let (m, k) = (l[rank - 2], l[rank - 1]);
            let (k2, n) = (r[rank - 2], r[rank - 1]);
            if k != k2 {
                return Err(Error::Msg(format!(
                    "primitive_shape: MatMul inner-dim mismatch {k} vs {k2}"
                ))
                .bt());
            }
            let mut out = l[..rank - 2].to_vec();
            out.push(m);
            out.push(n);
            (Shape::from_dims(&out), input_dtypes[0])
        }

        // --- indexing (data=in[0], index=in[1]) ---
        IndexSelect { dim } => {
            need(input_shapes, 2, op)?;
            let d = input_shapes[0].dims();
            if *dim >= d.len() {
                return Err(Error::Msg(format!(
                    "primitive_shape: IndexSelect dim {dim} invalid for {d:?}"
                ))
                .bt());
            }
            let mut out = d.to_vec();
            out[*dim] = input_shapes[1].elem_count();
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Gather { .. } => {
            need(input_shapes, 2, op)?;
            (input_shapes[1].clone(), input_dtypes[0])
        }
        IndexAdd { .. } | ScatterAdd { .. } => {
            need(input_shapes, 1, op)?;
            (input_shapes[0].clone(), input_dtypes[0])
        }

        // --- value source leaf ---
        Iota { len } => (Shape::from_dims(&[*len]), DType::F32),

        // --- everything else: honest miss, never a crash ---
        _ => {
            return Err(Error::Msg(format!(
                "primitive_shape: {op:?} is not a first-order shape-inferable primitive"
            ))
            .bt());
        }
    })
}

fn reduce_remove(dims: &[usize], dim: usize, op: &Op) -> Result<Shape, Error> {
    if dim >= dims.len() {
        return Err(Error::Msg(format!(
            "primitive_shape: {op:?} dim {dim} out of range for {dims:?}"
        ))
        .bt());
    }
    let out: Vec<usize> = dims.iter().enumerate().filter(|(i, _)| *i != dim).map(|(_, &v)| v).collect();
    Ok(Shape::from_dims(&out))
}

#[cfg(test)]
mod tests {
    use super::primitive_shape;
    use crate::{Op, PadMode};
    use fuel_ir::{DType, Shape};

    fn s(d: &[usize]) -> Shape { Shape::from_dims(d) }

    #[test]
    fn elementwise_preserves_shape_and_dtype() {
        let (sh, dt) = primitive_shape(&Op::Add, &[s(&[2, 3]), s(&[2, 3])], &[DType::F32, DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[2, 3]), DType::F32));
        let (sh, dt) = primitive_shape(&Op::Neg, &[s(&[4])], &[DType::BF16]).unwrap();
        assert_eq!((sh, dt), (s(&[4]), DType::BF16));
    }

    #[test]
    fn comparison_forces_u8() {
        let (sh, dt) = primitive_shape(&Op::Lt, &[s(&[5]), s(&[5])], &[DType::F32, DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[5]), DType::U8));
    }

    #[test]
    fn cast_takes_target_dtype() {
        let (sh, dt) = primitive_shape(&Op::Cast(DType::F16), &[s(&[2, 2])], &[DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[2, 2]), DType::F16));
    }

    #[test]
    fn slice_shrinks_named_dim() {
        let (sh, dt) = primitive_shape(&Op::Slice { dim: 1, start: 2, len: 3 }, &[s(&[4, 8])], &[DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[4, 3]), DType::F32));
    }

    #[test]
    fn concat_sums_the_join_dim() {
        let (sh, _) = primitive_shape(&Op::Concat { dim: 0 }, &[s(&[2, 4]), s(&[5, 4])], &[DType::F32, DType::F32]).unwrap();
        assert_eq!(sh, s(&[7, 4]));
    }

    #[test]
    fn permute_reorders_axes() {
        let (sh, _) = primitive_shape(&Op::Permute(vec![2, 0, 1]), &[s(&[2, 3, 4])], &[DType::F32]).unwrap();
        assert_eq!(sh, s(&[4, 2, 3]));
    }

    #[test]
    fn reshape_and_broadcast_take_target() {
        let (sh, _) = primitive_shape(&Op::Reshape(s(&[6])), &[s(&[2, 3])], &[DType::F32]).unwrap();
        assert_eq!(sh, s(&[6]));
        let (sh, _) = primitive_shape(&Op::BroadcastTo(s(&[2, 3])), &[s(&[1, 3])], &[DType::F32]).unwrap();
        assert_eq!(sh, s(&[2, 3]));
    }

    #[test]
    fn mean_dim_removes_the_dim() {
        let (sh, dt) = primitive_shape(&Op::MeanDim(1), &[s(&[2, 3, 4])], &[DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[2, 4]), DType::F32));
    }

    #[test]
    fn reduce_to_and_scalar_reductions() {
        let (sh, _) = primitive_shape(&Op::ReduceMaxTo(s(&[2, 1])), &[s(&[2, 5])], &[DType::F32]).unwrap();
        assert_eq!(sh, s(&[2, 1]));
        let (sh, _) = primitive_shape(&Op::SumAll, &[s(&[3, 3])], &[DType::F32]).unwrap();
        assert_eq!(sh, s(&[]));
    }

    #[test]
    fn argmax_produces_u32() {
        let (sh, dt) = primitive_shape(&Op::ArgMaxDim(0), &[s(&[4, 6])], &[DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[6]), DType::U32));
    }

    #[test]
    fn matmul_contracts_inner_dim() {
        let (sh, _) = primitive_shape(&Op::MatMul, &[s(&[2, 3]), s(&[3, 5])], &[DType::F32, DType::F32]).unwrap();
        assert_eq!(sh, s(&[2, 5]));
        // batched, same-rank
        let (sh, _) = primitive_shape(&Op::MatMul, &[s(&[7, 2, 3]), s(&[7, 3, 5])], &[DType::F32, DType::F32]).unwrap();
        assert_eq!(sh, s(&[7, 2, 5]));
    }

    #[test]
    fn where_is_cond_shape_a_dtype() {
        let (sh, dt) = primitive_shape(&Op::Where, &[s(&[4]), s(&[4]), s(&[4])], &[DType::U8, DType::F32, DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[4]), DType::F32));
    }

    #[test]
    fn index_ops() {
        // IndexSelect: dim replaced by index length (index is in[1], 1-D)
        let (sh, _) = primitive_shape(&Op::IndexSelect { dim: 0 }, &[s(&[8, 4]), s(&[3])], &[DType::F32, DType::U32]).unwrap();
        assert_eq!(sh, s(&[3, 4]));
        // Gather: out == index shape
        let (sh, _) = primitive_shape(&Op::Gather { dim: 1 }, &[s(&[2, 8]), s(&[2, 3])], &[DType::F32, DType::U32]).unwrap();
        assert_eq!(sh, s(&[2, 3]));
    }

    #[test]
    fn pad_extends_each_axis() {
        let (sh, _) = primitive_shape(
            &Op::Pad { padding: vec![(1, 1), (0, 2)], mode: PadMode::Constant, value: 0.0 },
            &[s(&[3, 4])], &[DType::F32]).unwrap();
        assert_eq!(sh, s(&[5, 6]));
    }

    #[test]
    fn iota_is_a_len_vector() {
        let (sh, dt) = primitive_shape(&Op::Iota { len: 9 }, &[], &[]).unwrap();
        assert_eq!((sh, dt), (s(&[9]), DType::F32));
    }

    #[test]
    fn out_of_scope_ops_error_not_panic() {
        assert!(primitive_shape(&Op::Const, &[], &[]).is_err());
        assert!(primitive_shape(&Op::Release, &[s(&[1])], &[DType::F32]).is_err());
    }

    #[test]
    fn malformed_input_is_err_not_panic() {
        // Elementwise with no inputs.
        assert!(primitive_shape(&Op::Add, &[], &[]).is_err());
        // Slice dim out of range.
        assert!(primitive_shape(&Op::Slice { dim: 5, start: 0, len: 1 }, &[s(&[2, 2])], &[DType::F32]).is_err());
        // MatMul inner-dim mismatch.
        assert!(primitive_shape(&Op::MatMul, &[s(&[2, 3]), s(&[4, 5])], &[DType::F32, DType::F32]).is_err());
    }
}

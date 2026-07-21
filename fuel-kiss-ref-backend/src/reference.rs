//! kiss-ref reference / differential over the mapped floor.
//!
//! kiss-ref's `reference_*`/`diff_*` are ROW-oriented — one row per element,
//! each row the op's argument tuple (`[a]` unary, `[a, b]` binary). Fuel passes
//! column-major operands (one slice per operand), so these functions **transpose**
//! columns → rows first, checking all columns share a length (never panicking on
//! a ragged input). Every failure is a [`KissRefError`].

use crate::mapping::op_to_kiss;
use crate::KissRefError;
use fuel_kernel_seam_types::OpTag;

pub use kiss_ref_core::{DiffReport, Tolerance};

/// Transpose column-major `operands` into kiss-ref's per-element rows. Errors
/// (never panics) on zero operands or a ragged column set.
fn to_rows<T: Copy>(op: OpTag, operands: &[&[T]]) -> Result<Vec<Vec<T>>, KissRefError> {
    if operands.is_empty() {
        return Err(KissRefError::Arity { op, expected: 1, got: 0 });
    }
    let n = operands[0].len();
    for col in operands {
        if col.len() != n {
            return Err(KissRefError::LengthMismatch { expected: n, got: col.len() });
        }
    }
    // i < n == col.len() for every column, so col[i] never panics.
    Ok((0..n)
        .map(|i| operands.iter().map(|col| col[i]).collect())
        .collect())
}

macro_rules! adapter_float {
    ($refr:ident, $diff:ident, $kref:path, $kdiff:path, $t:ty) => {
        /// kiss-ref's reference output for `op` over column-major `operands`.
        pub fn $refr(op: OpTag, operands: &[&[$t]]) -> Result<Vec<$t>, KissRefError> {
            let kiss_op = op_to_kiss(op).ok_or(KissRefError::UnsupportedOp(op))?;
            let rows = to_rows(op, operands)?;
            let row_refs: Vec<&[$t]> = rows.iter().map(|r| r.as_slice()).collect();
            $kref(kiss_op, &row_refs).map_err(KissRefError::Eval)
        }

        /// Differential of `candidate` vs kiss-ref's reference for `op` over
        /// `operands`, under `tol`. `candidate` must hold one value per element.
        pub fn $diff(
            op: OpTag,
            candidate: &[$t],
            operands: &[&[$t]],
            tol: Tolerance,
        ) -> Result<DiffReport, KissRefError> {
            let kiss_op = op_to_kiss(op).ok_or(KissRefError::UnsupportedOp(op))?;
            let rows = to_rows(op, operands)?;
            if candidate.len() != rows.len() {
                return Err(KissRefError::LengthMismatch {
                    expected: rows.len(),
                    got: candidate.len(),
                });
            }
            let row_refs: Vec<&[$t]> = rows.iter().map(|r| r.as_slice()).collect();
            $kdiff(kiss_op, &row_refs, candidate, tol).map_err(KissRefError::Eval)
        }
    };
}

adapter_float!(reference_f32, diff_f32, kiss_ref_core::reference_f32, kiss_ref_core::diff_f32, f32);
adapter_float!(reference_f64, diff_f64, kiss_ref_core::reference_f64, kiss_ref_core::diff_f64, f64);
adapter_float!(reference_f16, diff_f16, kiss_ref_core::reference_f16, kiss_ref_core::diff_f16, half::f16);
adapter_float!(reference_bf16, diff_bf16, kiss_ref_core::reference_bf16, kiss_ref_core::diff_bf16, half::bf16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_add_is_exact() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [10.0f32, 20.0, 30.0];
        let out = reference_f32(OpTag::Add, &[&a, &b]).unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn diff_matching_candidate_conforms() {
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let cand = [4.0f32, 6.0];
        let rep = diff_f32(OpTag::Add, &cand, &[&a, &b], Tolerance::Exact).unwrap();
        assert!(rep.conforms(), "identical output must satisfy Exact");
    }

    #[test]
    fn diff_wrong_candidate_length_errs() {
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let cand = [4.0f32]; // one short
        assert!(matches!(
            diff_f32(OpTag::Add, &cand, &[&a, &b], Tolerance::Exact),
            Err(KissRefError::LengthMismatch { expected: 2, got: 1 })
        ));
    }

    #[test]
    fn ragged_operands_err_not_panic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [3.0f32, 4.0]; // shorter column
        assert!(matches!(
            reference_f32(OpTag::Add, &[&a, &b]),
            Err(KissRefError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn unsupported_op_declines() {
        let a = [1.0f32];
        assert!(matches!(
            reference_f32(OpTag::MatMul, &[&a]),
            Err(KissRefError::UnsupportedOp(_))
        ));
    }

    #[test]
    fn reference_exp_matches_libm() {
        let x = [0.0f32, 1.0];
        let out = reference_f32(OpTag::Exp, &[&x]).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - std::f32::consts::E).abs() < 1e-5);
    }
}

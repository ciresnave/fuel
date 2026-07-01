//! Status-code → `CudaError` mapping for baracuda-kernels-sys.
//!
//! Per `baracuda-kernels-sys/README.md`:
//!
//! | Code | Meaning |
//! | ---: | --- |
//! | `0`  | success |
//! | `1`  | misaligned operand |
//! | `2`  | invalid problem (M/N/K non-positive or shape inconsistency) |
//! | `3`  | not supported (kernel doesn't implement this shape) |
//! | `4`  | workspace too small or null when required |
//! | `5`  | internal kernel error (typically a launch failure) |

use fuel_ir::Error;

use crate::error::CudaError;

/// Convert a baracuda `_run` / `_can_implement` status code to a
/// `fuel_ir::Result`. `op_label` is folded into the error
/// message so the caller doesn't have to repeat it (kernel sites pass
/// the family + dtype, e.g. `"unary_neg_f32"`).
#[inline]
pub fn check(status: i32, op_label: &'static str) -> fuel_ir::Result<()> {
    if status == 0 {
        return Ok(());
    }
    Err(Error::cuda(CudaError::BaracudaKernel {
        op: op_label,
        code: status,
        reason: status_reason(status),
    }))
}

/// Human-readable description for a baracuda status code. Surfaces in
/// error messages; kept inline so a future maintainer can find the
/// contract without leaving this file.
#[inline]
pub fn status_reason(status: i32) -> &'static str {
    match status {
        0 => "success",
        1 => "misaligned operand (128-bit alignment required for sm_80 SIMD/MMA paths)",
        2 => "invalid problem (M/N/K non-positive or shape inconsistency)",
        3 => "not supported (kernel doesn't implement this shape)",
        4 => "workspace too small or null when required",
        5 => "internal kernel error (typically a CUDA launch failure)",
        _ => "unknown baracuda status code",
    }
}

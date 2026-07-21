//! # fuel-kiss-ref-backend
//!
//! CPU, never-panic adapter exposing [`kiss_ref_core`] as Fuel's primitive-floor
//! **reference** / **differential target**. Correctness only — no fusion, no
//! optimizer, no GPU. It maps Fuel's op tag ([`fuel_kernel_seam_types::OpTag`])
//! and dtype ([`fuel_ir::DType`]) onto kiss-ref's vocabulary, then delegates to
//! kiss-ref's spec-exact reference kernels.
//!
//! Per KISS-CONFORM §6.6-0007, kiss-ref is a live **diff target**, never a
//! verdict source. This crate therefore only *computes* references / diffs; the
//! flag-not-verdict decision lives in `fuel-dispatch`'s ingestion path.
//!
//! Every failure is a [`KissRefError`] (kiss-ref-core is itself never-panic), so
//! the adapter is safe on Fuel's never-panic execution path.

use fuel_ir::DType;
use fuel_kernel_seam_types::OpTag;

/// Never-panic failure surface. Every adapter call returns `Result<_, KissRefError>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KissRefError {
    /// The op tag has no mapping into kiss-ref's `Op` vocabulary (off the floor).
    UnsupportedOp(OpTag),
    /// The dtype has no mapping into kiss-ref's `Dtype` vocabulary.
    UnsupportedDtype(DType),
    /// Wrong number of operand columns for the op.
    Arity { op: OpTag, expected: usize, got: usize },
    /// Operand columns / candidate slice lengths disagree.
    LengthMismatch { expected: usize, got: usize },
    /// A kiss-ref reference evaluation failed (wraps its typed error).
    Eval(kiss_ref_core::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_type_constructs() {
        let e = KissRefError::LengthMismatch { expected: 4, got: 3 };
        assert!(matches!(e, KissRefError::LengthMismatch { expected: 4, got: 3 }));
    }
}

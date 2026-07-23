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
    /// A region op node carries non-default attrs (scalar params, axes, …) —
    /// the kiss §6.13 `Expr` grammar has no attribute channel, so the region
    /// declines (a typed coverage gap, not a failure).
    UnsupportedAttrs(OpTag),
    /// A region contains a matcher-only node (`SeeThrough`/`Any`), or is not
    /// rooted at an op node — concrete recipe regions never carry these.
    UnsupportedNode,
}

pub mod mapping;
pub mod reference;
pub mod region;

pub use mapping::{dtype_to_kiss, op_to_kiss, supports};
pub use reference::{
    diff_bf16, diff_f16, diff_f32, diff_f64, reference_bf16, reference_f16, reference_f32,
    reference_f64, DiffReport, Tolerance,
};
pub use region::{
    diff_region_bf16, diff_region_f16, diff_region_f32, diff_region_f64, op_ulp_ceiling,
    reference_region_bf16, reference_region_f16, reference_region_f32, reference_region_f64,
    region_advisory_tolerance, region_op_count, region_supported, region_ulp_ceilings,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_type_constructs() {
        let e = KissRefError::LengthMismatch { expected: 4, got: 3 };
        assert!(matches!(e, KissRefError::LengthMismatch { expected: 4, got: 3 }));
    }
}

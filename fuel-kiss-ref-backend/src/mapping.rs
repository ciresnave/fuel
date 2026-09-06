// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuel ↔ kiss-ref vocabulary mapping for the scalar/elementwise floor.
//!
//! Maps Fuel's [`OpTag`] and [`DType`] onto kiss-ref's `Op` / `Dtype`. Only the
//! floor subset kiss-ref covers is mapped; everything else declines (`None`), so
//! [`supports`] gates every adapter call. NaN-propagating `Maximum`/`Minimum`
//! (Fuel's pinned convention) map to kiss `MaxProp`/`MinProp`; Fuel's `Gelu`
//! (tanh-approx) maps to kiss `GeluTanh` and `GeluErf` (exact) to kiss `Gelu`.

use fuel_ir::DType;
use fuel_kernel_seam_types::OpTag;
use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;

/// Map a Fuel op tag to its kiss-ref `Op`, or `None` if off the mapped floor.
pub fn op_to_kiss(op: OpTag) -> Option<Op> {
    use OpTag as T;
    Some(match op {
        // binary arithmetic / extremum
        T::Add => Op::Add,
        T::Sub => Op::Sub,
        T::Mul => Op::Mul,
        T::Div => Op::Div,
        T::Maximum => Op::MaxProp, // NaN-propagating (Fuel convention)
        T::Minimum => Op::MinProp,
        T::Pow => Op::Pow,
        // unary math
        T::Neg => Op::Neg,
        T::Abs => Op::Abs,
        T::Sqr => Op::Sqr,
        T::Sqrt => Op::Sqrt,
        T::Rsqrt => Op::Rsqrt,
        T::Recip => Op::Recip,
        T::Sign => Op::Sign,
        T::Exp => Op::Exp,
        T::Log => Op::Log,
        T::Sin => Op::Sin,
        T::Cos => Op::Cos,
        T::Tanh => Op::Tanh,
        T::Erf => Op::Erf,
        // activations
        T::Relu => Op::Relu,
        T::Sigmoid => Op::Sigmoid,
        T::Silu => Op::Silu,
        T::Gelu => Op::GeluTanh, // Fuel Gelu = tanh-approx
        T::GeluErf => Op::Gelu,  // Fuel GeluErf = exact erf
        T::Step => Op::Step,
        // rounding
        T::Floor => Op::Floor,
        T::Ceil => Op::Ceil,
        T::Round => Op::RoundEven,
        // everything else (Rem, MatMul, reductions, shape/index ops, …) declines
        _ => return None,
    })
}

/// Map a Fuel dtype to its kiss-ref `Dtype`, or `None` when there is no
/// mappable equivalent. Two DISTINCT reasons, kept apart because one of them
/// stopped being true at kiss-classify-vocab 0.3:
///
/// - `F6E2M3` / `F6E3M2` / `F4` — kiss-classify has no such token at all.
/// - `F8E8M0` / `F8E6M2` — kiss-classify 0.3 DOES define `F8e8m0` / `F8e6m2`,
///   so "no equivalent" is no longer why these decline. They are MX shared-
///   exponent *scale* types, never element-value dtypes: kiss-classify states
///   they decline compute in a value position, and this adapter feeds a
///   compute oracle. Mapping them would hand the differential target a scale
///   operand as if it were an element value.
pub fn dtype_to_kiss(d: DType) -> Option<Dtype> {
    use DType as D;
    Some(match d {
        D::F16 => Dtype::F16,
        D::BF16 => Dtype::Bf16,
        D::F32 => Dtype::F32,
        D::F64 => Dtype::F64,
        D::U8 => Dtype::U8,
        D::I8 => Dtype::I8,
        D::U32 => Dtype::U32,
        D::I16 => Dtype::I16,
        D::I32 => Dtype::I32,
        D::I64 => Dtype::I64,
        D::F8E4M3 => Dtype::F8e4m3fn,
        _ => return None,
    })
}

/// Whether `(op, dtype)` is a live kiss-ref diff target — mapped both ways AND
/// `Support::Done` (an unmapped op/dtype, or a `Pending`/`NotApplicable` cell,
/// declines). Uses `matches!` so the growing `Support` enum needs no arm here.
pub fn supports(op: OpTag, dtype: DType) -> bool {
    let (Some(o), Some(d)) = (op_to_kiss(op), dtype_to_kiss(dtype)) else {
        return false;
    };
    matches!(kiss_ref_core::support(o, d), kiss_ref_core::Support::Done)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_the_floor() {
        assert!(matches!(op_to_kiss(OpTag::Add), Some(Op::Add)));
        assert!(matches!(op_to_kiss(OpTag::Maximum), Some(Op::MaxProp)));
        assert!(matches!(op_to_kiss(OpTag::Gelu), Some(Op::GeluTanh)));
        assert!(matches!(op_to_kiss(OpTag::GeluErf), Some(Op::Gelu)));
        assert!(matches!(dtype_to_kiss(DType::F32), Some(Dtype::F32)));
        assert!(matches!(dtype_to_kiss(DType::I8), Some(Dtype::I8)));
        assert!(matches!(dtype_to_kiss(DType::BF16), Some(Dtype::Bf16)));
    }

    #[test]
    fn declines_off_floor_op() {
        assert!(op_to_kiss(OpTag::MatMul).is_none());
    }

    /// GAP-048 §6.15 no-substitution guard. `op_to_kiss` must never advertise a
    /// native Fuel op as a DIFFERENT-semantics KISS op. Fuel's `Maximum`/`Minimum`
    /// are NaN-PROPAGATING (`MaxProp`/`MinProp`), NOT the IEEE NaN-SUPPRESSING
    /// `FmaxIeee`/`FminIeee`; Fuel's `Rem` is FLOORED, not `RemTrunc`. Those
    /// IEEE/truncated ops are RESOLVED by decomposition (fuel-graph
    /// `fmax_ieee`/`fmin_ieee`/`rem_trunc`), never mapped from a native op —
    /// mapping `Maximum -> FmaxIeee` is the "one careless mapping away" §6.15
    /// substitution, and this test is what keeps it from happening. `kiss-ops-vocab`
    /// DOES define `FmaxIeee`/`FminIeee`/`RemTrunc`, so the wrong arm would
    /// type-check — the guard is not vacuous.
    #[test]
    fn never_substitutes_a_native_op_for_an_ieee_or_truncated_kiss_op() {
        // Truthful mappings — Fuel's native ops advertised as their REAL semantics:
        assert!(matches!(op_to_kiss(OpTag::Maximum), Some(Op::MaxProp)));
        assert!(matches!(op_to_kiss(OpTag::Minimum), Some(Op::MinProp)));
        // The substitutions §6.15 forbids — must NEVER happen:
        assert!(!matches!(op_to_kiss(OpTag::Maximum), Some(Op::FmaxIeee)));
        assert!(!matches!(op_to_kiss(OpTag::Minimum), Some(Op::FminIeee)));
        assert!(!matches!(op_to_kiss(OpTag::Rem), Some(Op::RemTrunc)));
    }

    #[test]
    fn declines_dtype_without_kiss_equivalent() {
        // An MX format Fuel has but kiss-classify lacks.
        assert!(dtype_to_kiss(DType::F6E2M3).is_none());
        assert!(!supports(OpTag::Add, DType::F6E2M3));
    }

    #[test]
    fn supports_floor_cell_and_declines_off_floor() {
        assert!(supports(OpTag::Add, DType::F32));
        assert!(!supports(OpTag::MatMul, DType::F32)); // op declines
    }

    /// `DType::F8E4M3` IS OCP *finite* E4M3FN -- bias 7, max-finite +/-448, no
    /// infinities, single NaN -- an identity `fuel-ir` commits in the variant's
    /// own doc and tests at three sites, none of them this adapter.
    ///
    /// This guard is NEW WORK CREATED BY THE 0.3 BUMP, not a pre-existing hole.
    /// kiss-classify 0.2 had exactly one `E4m3`, so the arm could not be got
    /// wrong. 0.3 splits the family into `F8e4m3fn` and `F8e4m3fnuz` (bias 8,
    /// no -0, no infinities), which kiss-classify itself calls BYTE-INCOMPATIBLE
    /// -- and `fnuz` carries no compute semantics at sk4 at all. Both spellings
    /// type-check here, so the wrong one is a silent wrong-oracle defect rather
    /// than a build failure. Pin the arm so the choice stays deliberate.
    #[test]
    fn f8e4m3_maps_to_the_ocp_finite_variant_not_fnuz() {
        assert!(matches!(
            dtype_to_kiss(DType::F8E4M3),
            Some(Dtype::F8e4m3fn)
        ));
    }

    /// The MX shared-exponent SCALE types decline -- and after the 0.3 bump they
    /// decline for a DIFFERENT REASON than they used to, which is why this is
    /// asserted rather than left to follow from absence.
    ///
    /// Under kiss-classify 0.2 these declined because the vocabulary had no such
    /// token: the outcome was forced. 0.3 DOES define `F8e8m0` and `F8e6m2`, so
    /// declining is now a CHOICE -- they are per-block scale operands, never
    /// element values, and this adapter feeds a compute oracle. Mapping one
    /// would hand the differential target a scale as if it were an element.
    #[test]
    fn mx_scale_dtypes_decline_even_though_kiss_now_defines_them() {
        assert!(dtype_to_kiss(DType::F8E8M0).is_none());
        assert!(dtype_to_kiss(DType::F8E6M2).is_none());
        assert!(!supports(OpTag::Add, DType::F8E8M0));
        assert!(!supports(OpTag::Add, DType::F8E6M2));
    }
}

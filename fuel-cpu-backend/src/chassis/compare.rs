//! Binary-compare chassis — elementwise comparison kernels with
//! typed input and U8 output. Covers Eq / Ne / Lt / Le / Gt / Ge
//! across f32 / f64 / bf16 / f16.
//!
//! Differs from the [`binary`](super::binary) chassis in one
//! structural way: the output dtype is **always** U8 (1 byte per
//! element holding `1` where the predicate holds, `0` otherwise)
//! regardless of input dtype. This means the chassis function
//! validates element counts rather than byte counts (output bytes
//! ≠ input bytes when T ≠ u8) and the op-core trait returns `bool`
//! rather than `T`.
//!
//! ## Layers
//!
//! 1. [`CompareOp<T>`] — what the chassis function consumes.
//!    `apply(T, T) -> u8`.
//! 2. [`CompareOpCore`] — what op authors implement. Two methods
//!    (`f32` + `f64`) returning `bool`.
//! 3. Four blanket impls — every `O: CompareOpCore` automatically
//!    gets `CompareOp<{f32, f64, bf16, f16}>` (half-floats compare
//!    via f32 round-trip). The bool → u8 conversion (1 / 0) lives
//!    in the blanket impls.
//!
//! Half-float comparisons via f32 round-trip are bit-identical to
//! the pre-refactor `binary_compare_kernel!` form for finite values
//! (bf16/f16 → f32 is lossless and order-preserving) and identical
//! for NaN (any comparison involving NaN is unordered → false →
//! `0`, except `!=` which yields `true` → `1`).

use bytemuck::Pod;

use crate::byte_storage::CpuStorageBytes;
use fuel_core_types::{Error, Result};

// =============================================================================
// Traits
// =============================================================================

/// Per-(op, dtype) comparison operation. The chassis function
/// [`compare`] consumes one of these implementations to walk a
/// pair of typed input tensors and produce a `&mut [u8]` mask.
///
/// Implementations are auto-derived from [`CompareOpCore`] via
/// four blanket impls — don't implement this directly.
pub trait CompareOp<T: Copy> {
    fn apply(a: T, b: T) -> u8;
}

/// What op authors actually implement. Two methods carry the f32
/// and f64 comparison predicates respectively; the blanket
/// [`CompareOp`] impls in this module derive the four dtype-
/// specific implementations and convert `bool` → `u8` (1 / 0).
pub trait CompareOpCore {
    fn f32(a: f32, b: f32) -> bool;
    fn f64(a: f64, b: f64) -> bool;
}

// Blanket impls — derive all four `CompareOp<T>` instances from
// one `CompareOpCore` and convert bool → u8 once here.

impl<O: CompareOpCore> CompareOp<f32> for O {
    fn apply(a: f32, b: f32) -> u8 {
        if <O as CompareOpCore>::f32(a, b) { 1 } else { 0 }
    }
}

impl<O: CompareOpCore> CompareOp<f64> for O {
    fn apply(a: f64, b: f64) -> u8 {
        if <O as CompareOpCore>::f64(a, b) { 1 } else { 0 }
    }
}

impl<O: CompareOpCore> CompareOp<half::bf16> for O {
    fn apply(a: half::bf16, b: half::bf16) -> u8 {
        if <O as CompareOpCore>::f32(a.to_f32(), b.to_f32()) { 1 } else { 0 }
    }
}

impl<O: CompareOpCore> CompareOp<half::f16> for O {
    fn apply(a: half::f16, b: half::f16) -> u8 {
        if <O as CompareOpCore>::f32(a.to_f32(), b.to_f32()) { 1 } else { 0 }
    }
}

// =============================================================================
// Chassis function
// =============================================================================

/// Elementwise `out[i] = U::apply(lhs[i], rhs[i])`. Validates the
/// three typed views have equal element counts (not byte counts —
/// output is U8 while inputs may be wider).
///
/// `name` appears in element-count-mismatch error messages.
pub fn compare<T, U>(
    name: &str,
    lhs: &CpuStorageBytes,
    rhs: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
) -> Result<()>
where
    T: Copy + Pod,
    U: CompareOp<T>,
{
    let lhs_view: &[T] = lhs.as_slice()?;
    let rhs_view: &[T] = rhs.as_slice()?;
    let out_view: &mut [u8] = output.as_slice_mut()?;
    if lhs_view.len() != rhs_view.len() || lhs_view.len() != out_view.len() {
        return Err(Error::Msg(format!(
            "{name}: element count mismatch (lhs={}, rhs={}, out={})",
            lhs_view.len(),
            rhs_view.len(),
            out_view.len(),
        ))
        .bt());
    }
    for (i, slot) in out_view.iter_mut().enumerate() {
        *slot = U::apply(lhs_view[i], rhs_view[i]);
    }
    Ok(())
}

// =============================================================================
// Op markers
// =============================================================================
//
// Each op is a zero-sized struct implementing `CompareOpCore`. The
// four `CompareOp<T>` impls fall out of the blanket impls above.

/// Elementwise equality. NaN follows IEEE-754: `NaN != NaN`.
pub struct Eq;
impl CompareOpCore for Eq {
    fn f32(a: f32, b: f32) -> bool { a == b }
    fn f64(a: f64, b: f64) -> bool { a == b }
}

/// Elementwise inequality. NaN follows IEEE-754: `NaN != NaN`
/// yields `true` (→ `1`).
pub struct Ne;
impl CompareOpCore for Ne {
    fn f32(a: f32, b: f32) -> bool { a != b }
    fn f64(a: f64, b: f64) -> bool { a != b }
}

/// Elementwise `<`. NaN-unordered: any comparison involving NaN
/// is `false`.
pub struct Lt;
impl CompareOpCore for Lt {
    fn f32(a: f32, b: f32) -> bool { a < b }
    fn f64(a: f64, b: f64) -> bool { a < b }
}

/// Elementwise `<=`. NaN-unordered.
pub struct Le;
impl CompareOpCore for Le {
    fn f32(a: f32, b: f32) -> bool { a <= b }
    fn f64(a: f64, b: f64) -> bool { a <= b }
}

/// Elementwise `>`. NaN-unordered.
pub struct Gt;
impl CompareOpCore for Gt {
    fn f32(a: f32, b: f32) -> bool { a > b }
    fn f64(a: f64, b: f64) -> bool { a > b }
}

/// Elementwise `>=`. NaN-unordered.
pub struct Ge;
impl CompareOpCore for Ge {
    fn f32(a: f32, b: f32) -> bool { a >= b }
    fn f64(a: f64, b: f64) -> bool { a >= b }
}

// =============================================================================
// Structural tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_op_eq_f32_returns_u8_mask() {
        assert_eq!(<Eq as CompareOp<f32>>::apply(2.5, 2.5), 1);
        assert_eq!(<Eq as CompareOp<f32>>::apply(2.5, 1.5), 0);
    }

    #[test]
    fn compare_op_lt_f32_nan_yields_zero() {
        // NaN < x is always false → 0.
        assert_eq!(<Lt as CompareOp<f32>>::apply(f32::NAN, 1.0), 0);
        assert_eq!(<Lt as CompareOp<f32>>::apply(1.0, f32::NAN), 0);
        assert_eq!(<Lt as CompareOp<f32>>::apply(f32::NAN, f32::NAN), 0);
    }

    #[test]
    fn compare_op_ne_f32_nan_yields_one() {
        // NaN != NaN is true per IEEE-754 → 1.
        assert_eq!(<Ne as CompareOp<f32>>::apply(f32::NAN, f32::NAN), 1);
        assert_eq!(<Ne as CompareOp<f32>>::apply(1.0, 1.0), 0);
        assert_eq!(<Ne as CompareOp<f32>>::apply(1.0, 2.0), 1);
    }

    #[test]
    fn compare_op_ge_le_at_equality() {
        assert_eq!(<Ge as CompareOp<f32>>::apply(2.0, 2.0), 1);
        assert_eq!(<Le as CompareOp<f32>>::apply(2.0, 2.0), 1);
        assert_eq!(<Gt as CompareOp<f32>>::apply(2.0, 2.0), 0);
        assert_eq!(<Lt as CompareOp<f32>>::apply(2.0, 2.0), 0);
    }

    #[test]
    fn compare_op_bf16_blanket_routes_through_f32() {
        let a = half::bf16::from_f32(2.5);
        let b = half::bf16::from_f32(2.5);
        assert_eq!(<Eq as CompareOp<half::bf16>>::apply(a, b), 1);
        let c = half::bf16::from_f32(1.5);
        assert_eq!(<Lt as CompareOp<half::bf16>>::apply(c, a), 1);
        assert_eq!(<Gt as CompareOp<half::bf16>>::apply(c, a), 0);
    }

    #[test]
    fn compare_chassis_element_count_mismatch_errors() {
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let rhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]); // 3 elems
        let mut out = CpuStorageBytes::from_zero_bytes(2); // 2 u8s
        let r = compare::<f32, Eq>("test", &lhs, &rhs, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn compare_chassis_walks_all_elements() {
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let rhs = CpuStorageBytes::from_slice(&[1.0_f32, 1.0, 3.0, 5.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4); // 4 u8s
        compare::<f32, Eq>("test", &lhs, &rhs, &mut out).expect("compare eq_f32");
        let r: &[u8] = out.as_slice().unwrap();
        assert_eq!(r, &[1, 0, 1, 0]);
    }
}

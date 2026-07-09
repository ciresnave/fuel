//! Binary-elementwise chassis — one shape/loop pass shared by every
//! per-(op, dtype) binary kernel (Add / Sub / Mul / Div / Maximum /
//! Minimum / Pow / Rem, across f32 / f64 / bf16 / f16).
//!
//! Mirrors the [`unary`](super::unary) module's three-layer design:
//!
//! 1. [`BinaryOp<T>`] — what the chassis function consumes. One
//!    `apply(T, T) -> T` method.
//! 2. [`BinaryOpCore`] — what op authors implement. Two methods
//!    (`f32` + `f64`) carrying the per-precision math.
//! 3. Four blanket impls — every `O: BinaryOpCore` automatically
//!    gets `BinaryOp<{f32, f64, bf16, f16}>` (half-floats via f32
//!    round-trip, bit-identical to pre-refactor `binary_kernel!`
//!    behavior).
//!
//! See the unary chassis for the rationale behind the `f32`/`f64`
//! split (rather than `T: Float`).

use bytemuck::Pod;

use crate::byte_storage::CpuStorageBytes;
use fuel_ir::{Error, Result};

// =============================================================================
// Traits
// =============================================================================

/// Per-(op, dtype) binary operation. The chassis function
/// [`binary`] consumes one of these implementations to walk a
/// pair of byte-shaped tensors elementwise.
///
/// Implementations are auto-derived from [`BinaryOpCore`] via four
/// blanket impls — don't implement this directly.
pub trait BinaryOp<T: Copy> {
    fn apply(a: T, b: T) -> T;
}

/// What op authors actually implement. Two methods carry the f32
/// and f64 math respectively; the blanket [`BinaryOp`] impls in
/// this module derive the four dtype-specific implementations
/// (f32 / f64 direct, bf16 / f16 via f32 round-trip).
pub trait BinaryOpCore {
    fn f32(a: f32, b: f32) -> f32;
    fn f64(a: f64, b: f64) -> f64;
}

// Blanket impls.

impl<O: BinaryOpCore> BinaryOp<f32> for O {
    fn apply(a: f32, b: f32) -> f32 { <O as BinaryOpCore>::f32(a, b) }
}

impl<O: BinaryOpCore> BinaryOp<f64> for O {
    fn apply(a: f64, b: f64) -> f64 { <O as BinaryOpCore>::f64(a, b) }
}

impl<O: BinaryOpCore> BinaryOp<half::bf16> for O {
    fn apply(a: half::bf16, b: half::bf16) -> half::bf16 {
        half::bf16::from_f32(<O as BinaryOpCore>::f32(a.to_f32(), b.to_f32()))
    }
}

impl<O: BinaryOpCore> BinaryOp<half::f16> for O {
    fn apply(a: half::f16, b: half::f16) -> half::f16 {
        half::f16::from_f32(<O as BinaryOpCore>::f32(a.to_f32(), b.to_f32()))
    }
}

// =============================================================================
// Chassis function
// =============================================================================

/// Elementwise `out[i] = U::apply(lhs[i], rhs[i])`. Validates all
/// three byte lengths match, then walks the typed views.
///
/// `name` appears in size-mismatch error messages so the
/// diagnostic points at the entry the caller invoked.
pub fn binary<T, U>(
    name: &str,
    lhs: &CpuStorageBytes,
    rhs: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
) -> Result<()>
where
    T: Copy + Pod,
    U: BinaryOp<T>,
{
    let lhs_bytes = lhs.len_bytes();
    let rhs_bytes = rhs.len_bytes();
    let out_bytes = output.len_bytes();
    if lhs_bytes != rhs_bytes || lhs_bytes != out_bytes {
        return Err(Error::Msg(format!(
            "{name}: byte length mismatch (lhs={lhs_bytes}, rhs={rhs_bytes}, out={out_bytes})",
        ))
        .bt());
    }
    let lhs_view: &[T] = lhs.as_slice()?;
    let rhs_view: &[T] = rhs.as_slice()?;
    let out_view: &mut [T] = output.as_slice_mut()?;
    for (i, slot) in out_view.iter_mut().enumerate() {
        *slot = U::apply(lhs_view[i], rhs_view[i]);
    }
    Ok(())
}

// =============================================================================
// Op markers
// =============================================================================
//
// Each op is a zero-sized struct implementing `BinaryOpCore`. The
// four `BinaryOp<T>` impls fall out of the blanket impls above.

/// Elementwise addition.
pub struct Add;
impl BinaryOpCore for Add {
    fn f32(a: f32, b: f32) -> f32 { a + b }
    fn f64(a: f64, b: f64) -> f64 { a + b }
}

/// Elementwise subtraction.
pub struct Sub;
impl BinaryOpCore for Sub {
    fn f32(a: f32, b: f32) -> f32 { a - b }
    fn f64(a: f64, b: f64) -> f64 { a - b }
}

/// Elementwise multiplication.
pub struct Mul;
impl BinaryOpCore for Mul {
    fn f32(a: f32, b: f32) -> f32 { a * b }
    fn f64(a: f64, b: f64) -> f64 { a * b }
}

/// Elementwise division. Division by zero yields IEEE-754 inf/NaN.
pub struct Div;
impl BinaryOpCore for Div {
    fn f32(a: f32, b: f32) -> f32 { a / b }
    fn f64(a: f64, b: f64) -> f64 { a / b }
}

/// Elementwise maximum. NaN-propagating (torch parity —
/// `torch.maximum` returns NaN if *either* operand is NaN), pinned
/// 2026-07-08 (`docs/architecture/10-decisions-log.md`). Deliberately
/// does *not* use `f32::max`/`f64::max` (those are NaN-as-missing —
/// they return the non-NaN operand instead). Payload-preserving: the
/// NaN operand is returned as-is (`a` checked before `b`, matching
/// `torch.maximum`'s lhs-first tie-break).
pub struct Maximum;
impl BinaryOpCore for Maximum {
    fn f32(a: f32, b: f32) -> f32 {
        if a.is_nan() { a } else if b.is_nan() { b } else { a.max(b) }
    }
    fn f64(a: f64, b: f64) -> f64 {
        if a.is_nan() { a } else if b.is_nan() { b } else { a.max(b) }
    }
}

/// Elementwise minimum. NaN handling mirrors [`Maximum`] (NaN-propagating,
/// torch parity).
pub struct Minimum;
impl BinaryOpCore for Minimum {
    fn f32(a: f32, b: f32) -> f32 {
        if a.is_nan() { a } else if b.is_nan() { b } else { a.min(b) }
    }
    fn f64(a: f64, b: f64) -> f64 {
        if a.is_nan() { a } else if b.is_nan() { b } else { a.min(b) }
    }
}

/// Elementwise binary power: `a ^ b` via `f32::powf` / `f64::powf`.
/// NaN follows IEEE-754 (e.g. `pow(-2, 0.5) = NaN`).
pub struct Pow;
impl BinaryOpCore for Pow {
    fn f32(a: f32, b: f32) -> f32 { a.powf(b) }
    fn f64(a: f64, b: f64) -> f64 { a.powf(b) }
}

/// Elementwise remainder, PyTorch convention: `a - floor(a/b) * b`.
/// Sign follows the divisor (not the dividend, as `%` would).
pub struct Rem;
impl BinaryOpCore for Rem {
    fn f32(a: f32, b: f32) -> f32 { a - (a / b).floor() * b }
    fn f64(a: f64, b: f64) -> f64 { a - (a / b).floor() * b }
}

// =============================================================================
// Structural tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_op_add_f32_sums() {
        assert_eq!(<Add as BinaryOp<f32>>::apply(2.5, 1.5), 4.0);
        assert_eq!(<Add as BinaryOp<f32>>::apply(-3.0, 3.0), 0.0);
    }

    #[test]
    fn binary_op_maximum_f32_picks_larger() {
        assert_eq!(<Maximum as BinaryOp<f32>>::apply(2.5, 1.5), 2.5);
        assert_eq!(<Maximum as BinaryOp<f32>>::apply(-3.0, 3.0), 3.0);
    }

    #[test]
    fn binary_op_rem_f32_pytorch_sign_follows_divisor() {
        // PyTorch: rem(7, -3) = -2 (sign follows -3, not 7).
        let got = <Rem as BinaryOp<f32>>::apply(7.0, -3.0);
        assert!((got - (-2.0)).abs() < 1e-6, "got {got}");
        // rem(-7, 3) = 2 (sign follows 3).
        let got = <Rem as BinaryOp<f32>>::apply(-7.0, 3.0);
        assert!((got - 2.0).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn binary_op_pow_f32() {
        assert_eq!(<Pow as BinaryOp<f32>>::apply(2.0, 3.0), 8.0);
        assert_eq!(<Pow as BinaryOp<f32>>::apply(4.0, 0.5), 2.0);
    }

    #[test]
    fn binary_op_bf16_blanket_routes_through_f32() {
        // Mul of two bf16 values too narrow to multiply natively
        // without precision loss — the f32 round-trip preserves
        // precision pre/post-narrow.
        let a = half::bf16::from_f32(1.5);
        let b = half::bf16::from_f32(2.5);
        let got = <Mul as BinaryOp<half::bf16>>::apply(a, b).to_f32();
        let expect = half::bf16::from_f32(1.5 * 2.5).to_f32();
        assert_eq!(got, expect);
    }

    #[test]
    fn binary_chassis_length_mismatch_errors() {
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let rhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]); // mismatch
        let mut out = CpuStorageBytes::from_zero_bytes(8);
        let r = binary::<f32, Add>("test", &lhs, &rhs, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn binary_chassis_walks_all_elements() {
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let rhs = CpuStorageBytes::from_slice(&[10.0_f32, 20.0, 30.0, 40.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(lhs.len_bytes());
        binary::<f32, Add>("test", &lhs, &rhs, &mut out).expect("binary add_f32");
        let r: &[f32] = out.as_slice().unwrap();
        assert_eq!(r, &[11.0, 22.0, 33.0, 44.0]);
    }
}

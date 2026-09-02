// SPDX-License-Identifier: MIT OR Apache-2.0
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

    /// Narrow-float lowering. THE DEFAULT PROMOTES, COMPUTES AND ROUNDS BACK,
    /// which is correct for every op that COMPUTES its result.
    ///
    /// ⚠️ OPS THAT **MOVE** AN OPERAND MUST OVERRIDE THIS. `half` quiets a
    /// signalling NaN on BOTH conversion legs independently (`bf16_to_f32` and
    /// `f32_to_bf16` each `| 0x0040`), so a round trip quiets twice --
    /// measured, `0x7F81` returns `0xFFC1`. KISS-OPS-6.16-0009: an op whose
    /// decomposition contains no arithmetic returns the MOVED OPERAND with its
    /// bits preserved exactly, and *"a promote-to-`f32`-and-round-back
    /// implementation of such an op is non-conforming for a narrow float"*.
    fn bf16(a: half::bf16, b: half::bf16) -> half::bf16 {
        half::bf16::from_f32(Self::f32(a.to_f32(), b.to_f32()))
    }

    /// See [`BinaryOpCore::bf16`].
    fn f16(a: half::f16, b: half::f16) -> half::f16 {
        half::f16::from_f32(Self::f32(a.to_f32(), b.to_f32()))
    }
}

// Blanket impls.

impl<O: BinaryOpCore> BinaryOp<f32> for O {
    fn apply(a: f32, b: f32) -> f32 {
        <O as BinaryOpCore>::f32(a, b)
    }
}

impl<O: BinaryOpCore> BinaryOp<f64> for O {
    fn apply(a: f64, b: f64) -> f64 {
        <O as BinaryOpCore>::f64(a, b)
    }
}

impl<O: BinaryOpCore> BinaryOp<half::bf16> for O {
    fn apply(a: half::bf16, b: half::bf16) -> half::bf16 {
        <O as BinaryOpCore>::bf16(a, b)
    }
}

impl<O: BinaryOpCore> BinaryOp<half::f16> for O {
    fn apply(a: half::f16, b: half::f16) -> half::f16 {
        <O as BinaryOpCore>::f16(a, b)
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
    fn f32(a: f32, b: f32) -> f32 {
        a + b
    }
    fn f64(a: f64, b: f64) -> f64 {
        a + b
    }
}

/// Elementwise subtraction.
pub struct Sub;
impl BinaryOpCore for Sub {
    fn f32(a: f32, b: f32) -> f32 {
        a - b
    }
    fn f64(a: f64, b: f64) -> f64 {
        a - b
    }
}

/// Elementwise multiplication.
pub struct Mul;
impl BinaryOpCore for Mul {
    fn f32(a: f32, b: f32) -> f32 {
        a * b
    }
    fn f64(a: f64, b: f64) -> f64 {
        a * b
    }
}

/// Elementwise division. Division by zero yields IEEE-754 inf/NaN.
pub struct Div;
impl BinaryOpCore for Div {
    fn f32(a: f32, b: f32) -> f32 {
        a / b
    }
    fn f64(a: f64, b: f64) -> f64 {
        a / b
    }
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
        if a.is_nan() {
            a
        } else if b.is_nan() {
            b
        } else {
            a.max(b)
        }
    }

    // ⚠️ NARROW OVERRIDE, NaN BRANCH ONLY -- and the split is deliberate.
    //
    // A NaN operand is MOVED, not computed: `if a.is_nan() { a }` returns that
    // operand. Routing it through f32 quiets it on BOTH conversion legs
    // (measured, 0x7F81 -> 0xFFC1), which KISS-OPS-6.16-0009 forbids for an op
    // whose decomposition contains no arithmetic.
    //
    // ⚠️ THE NON-NaN BRANCH DELIBERATELY KEEPS THE PROMOTING PATH, so this
    // change does NOT touch the +-0 tie. That tie is PR #67's subject and
    // deciding it here would pre-empt a ruling that is not mine. The delegation
    // is exact rather than approximate: for finite bf16 the widening is exact,
    // min/max returns one of those exact values, and narrowing an
    // exactly-representable bf16 is exact -- so the non-NaN result is
    // bit-identical to today's, tie included, whatever the tie turns out to be.
    fn bf16(a: half::bf16, b: half::bf16) -> half::bf16 {
        if a.is_nan() {
            a
        } else if b.is_nan() {
            b
        } else {
            half::bf16::from_f32(Self::f32(a.to_f32(), b.to_f32()))
        }
    }

    /// See [`Maximum::bf16`].
    fn f16(a: half::f16, b: half::f16) -> half::f16 {
        if a.is_nan() {
            a
        } else if b.is_nan() {
            b
        } else {
            half::f16::from_f32(Self::f32(a.to_f32(), b.to_f32()))
        }
    }
    fn f64(a: f64, b: f64) -> f64 {
        if a.is_nan() {
            a
        } else if b.is_nan() {
            b
        } else {
            a.max(b)
        }
    }
}

/// Elementwise minimum. NaN handling mirrors [`Maximum`] (NaN-propagating,
/// torch parity).
pub struct Minimum;
impl BinaryOpCore for Minimum {
    fn f32(a: f32, b: f32) -> f32 {
        if a.is_nan() {
            a
        } else if b.is_nan() {
            b
        } else {
            a.min(b)
        }
    }

    // ⚠️ NARROW OVERRIDE, NaN BRANCH ONLY -- and the split is deliberate.
    //
    // A NaN operand is MOVED, not computed: `if a.is_nan() { a }` returns that
    // operand. Routing it through f32 quiets it on BOTH conversion legs
    // (measured, 0x7F81 -> 0xFFC1), which KISS-OPS-6.16-0009 forbids for an op
    // whose decomposition contains no arithmetic.
    //
    // ⚠️ THE NON-NaN BRANCH DELIBERATELY KEEPS THE PROMOTING PATH, so this
    // change does NOT touch the +-0 tie. That tie is PR #67's subject and
    // deciding it here would pre-empt a ruling that is not mine. The delegation
    // is exact rather than approximate: for finite bf16 the widening is exact,
    // min/max returns one of those exact values, and narrowing an
    // exactly-representable bf16 is exact -- so the non-NaN result is
    // bit-identical to today's, tie included, whatever the tie turns out to be.
    fn bf16(a: half::bf16, b: half::bf16) -> half::bf16 {
        if a.is_nan() {
            a
        } else if b.is_nan() {
            b
        } else {
            half::bf16::from_f32(Self::f32(a.to_f32(), b.to_f32()))
        }
    }

    /// See [`Minimum::bf16`].
    fn f16(a: half::f16, b: half::f16) -> half::f16 {
        if a.is_nan() {
            a
        } else if b.is_nan() {
            b
        } else {
            half::f16::from_f32(Self::f32(a.to_f32(), b.to_f32()))
        }
    }
    fn f64(a: f64, b: f64) -> f64 {
        if a.is_nan() {
            a
        } else if b.is_nan() {
            b
        } else {
            a.min(b)
        }
    }
}

/// Elementwise binary power: `a ^ b` via `f32::powf` / `f64::powf`.
/// NaN follows IEEE-754 (e.g. `pow(-2, 0.5) = NaN`).
pub struct Pow;
impl BinaryOpCore for Pow {
    fn f32(a: f32, b: f32) -> f32 {
        a.powf(b)
    }
    fn f64(a: f64, b: f64) -> f64 {
        a.powf(b)
    }
}

/// Elementwise remainder, PyTorch convention: `a - floor(a/b) * b`.
/// Sign follows the divisor (not the dividend, as `%` would).
pub struct Rem;
impl BinaryOpCore for Rem {
    fn f32(a: f32, b: f32) -> f32 {
        a - (a / b).floor() * b
    }
    fn f64(a: f64, b: f64) -> f64 {
        a - (a / b).floor() * b
    }
}

// =============================================================================
// Structural tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// KISS-OPS-6.16-0009: a MOVED NaN operand keeps its bits exactly.
    ///
    /// `Maximum`/`Minimum` return an operand when either is NaN -- that is a
    /// move, not a computation -- so routing it through f32 is non-conforming.
    /// `half` quiets on BOTH conversion legs, so the round trip quiets twice.
    ///
    /// ⚠️ THE POSITIVE CONTROL IS REQUIRED: "still signalling" is only evidence
    /// if the promoting path demonstrably quiets in this same build. `Add`
    /// computes, so it legitimately keeps the default and serves as that control.
    #[test]
    fn minmax_move_a_nan_operand_without_quieting_it() {
        const QUIET: u16 = 0x0040;
        let s = half::bf16::from_bits(0x7F81);
        let one = half::bf16::from_f32(1.0);
        assert_eq!(s.to_bits() & QUIET, 0, "fixture must be SIGNALLING");

        let control = <Add as BinaryOpCore>::bf16(s, one);
        assert_ne!(
            control.to_bits() & QUIET,
            0,
            "control failed: the promoting path did not quiet, so the assertions              below prove nothing"
        );

        for (name, got) in [
            ("Maximum lhs", <Maximum as BinaryOpCore>::bf16(s, one)),
            ("Maximum rhs", <Maximum as BinaryOpCore>::bf16(one, s)),
            ("Minimum lhs", <Minimum as BinaryOpCore>::bf16(s, one)),
            ("Minimum rhs", <Minimum as BinaryOpCore>::bf16(one, s)),
        ] {
            assert_eq!(
                got.to_bits(),
                s.to_bits(),
                "{name} did not MOVE the NaN operand's bits (0x{:04X} -> 0x{:04X});                  KISS-OPS-6.16-0009 requires payload AND signalling bit preserved",
                s.to_bits(),
                got.to_bits()
            );
        }
    }

    /// ⚠️ THE NON-NaN PATH MUST BE BYTE-IDENTICAL TO THE PROMOTING PATH.
    ///
    /// This is the guard that keeps the NaN fix OUT of the ±0 tie, which is
    /// PR #67's subject. If a later edit "simplifies" the override into a
    /// narrow-native compare, the tie gets decided here by accident and THIS
    /// test fails -- which is the point. It includes ±0 in both orders
    /// deliberately: whatever the tie currently does, the override must do the
    /// same thing.
    #[test]
    fn minmax_non_nan_path_is_unchanged_by_the_nan_override() {
        let cases = [0x3F80u16, 0xC020, 0x0000, 0x8000, 0x7F7F, 0xFF7F];
        for &ab in &cases {
            for &bb in &cases {
                let (a, b) = (half::bf16::from_bits(ab), half::bf16::from_bits(bb));
                let want_max =
                    half::bf16::from_f32(<Maximum as BinaryOpCore>::f32(a.to_f32(), b.to_f32()));
                let want_min =
                    half::bf16::from_f32(<Minimum as BinaryOpCore>::f32(a.to_f32(), b.to_f32()));
                assert_eq!(
                    <Maximum as BinaryOpCore>::bf16(a, b).to_bits(),
                    want_max.to_bits(),
                    "Maximum diverged from the promoting path on (0x{ab:04X}, 0x{bb:04X})                      -- the NaN override must not touch the non-NaN result, tie included"
                );
                assert_eq!(
                    <Minimum as BinaryOpCore>::bf16(a, b).to_bits(),
                    want_min.to_bits(),
                    "Minimum diverged from the promoting path on (0x{ab:04X}, 0x{bb:04X})"
                );
            }
        }
    }

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

// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tolerance comparison for float results checked against a reference.
//!
//! Rounding both sides to N decimals and comparing for equality is not a
//! tolerance: two values 1e-7 apart straddle a rounding boundary about
//! `1e-7 / 10^-N` of the time (0.73164999 and 0.73165001 round to 0.7316 and
//! 0.7317), so such a test fails at random on random inputs. [`all_close`]
//! bounds the *distance* instead: `|got - want| <= abs + rel * |want|`.
//!
//! The bounds are the caller's, with a reason: pick `abs`/`rel` from the
//! kernel's documented error, with margin, and keep them far below the size
//! of a real mistake (the unit tests here pin that a 1e-3 error is rejected
//! at the 1e-5 bound the metal `cos` test uses).

/// `Ok(())` when every `got[i]` is within `abs + rel * |want[i]|` of `want[i]`.
/// Two NaNs match; a NaN against a number, or a length mismatch, is an error.
/// The error names the first offending index and both values.
pub fn all_close(got: &[f32], want: &[f32], abs: f32, rel: f32) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "length mismatch: got {} vs want {}",
            got.len(),
            want.len()
        ));
    }
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g.is_nan() && w.is_nan() {
            continue;
        }
        let diff = (g - w).abs();
        // Test `<=` and negate the bool, so a NaN on one side fails rather
        // than passing (`diff > bound` is false for NaN).
        let within = diff <= abs + rel * w.abs();
        if !within {
            return Err(format!(
                "index {i}: got {g} vs want {w}, |diff| = {diff:e} > {abs:e} + {rel:e}*|want|"
            ));
        }
    }
    Ok(())
}

/// [`all_close`], panicking with its message.
#[track_caller]
pub fn assert_all_close(got: &[f32], want: &[f32], abs: f32, rel: f32) {
    if let Err(e) = all_close(got, want, abs, rel) {
        panic!("{e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_pair_that_flipped_the_rounded_comparison() {
        // The macOS flake: rounded to 4 decimals these are 0.7316 vs 0.7317.
        let (g, w) = (0.731_649_9_f32, 0.731_650_1_f32);
        assert_ne!(
            (g * 1e4).round() / 1e4,
            (w * 1e4).round() / 1e4,
            "control: the old comparison flips"
        );
        assert!(all_close(&[g], &[w], 1e-5, 0.0).is_ok());
        // And the wider pair named in review: 0.73165 vs 0.73175 (1e-4 apart).
        assert!(all_close(&[0.73165], &[0.73175], 1.5e-4, 0.0).is_ok());
    }

    #[test]
    fn rejects_a_real_error() {
        // Positive control: an off-by-1e-3 value must fail at the 1e-5 bound.
        let e = all_close(&[0.7317 + 1e-3], &[0.7317], 1e-5, 0.0).unwrap_err();
        assert!(e.contains("index 0"), "{e}");
        // Sabotage sibling: the same pair passes once the bound is loosened
        // past the error, so the rejection above comes from the bound.
        assert!(all_close(&[0.7317 + 1e-3], &[0.7317], 2e-3, 0.0).is_ok());
    }

    #[test]
    fn nan_and_length_are_not_silently_equal() {
        assert!(all_close(&[f32::NAN], &[1.0], 1.0, 1.0).is_err());
        assert!(all_close(&[1.0], &[f32::NAN], 1.0, 1.0).is_err());
        assert!(all_close(&[f32::NAN], &[f32::NAN], 0.0, 0.0).is_ok());
        assert!(all_close(&[1.0], &[1.0, 2.0], 1.0, 1.0).is_err());
    }

    #[test]
    fn relative_bound_scales_with_the_reference() {
        assert!(all_close(&[1000.0 + 0.5], &[1000.0], 0.0, 1e-3).is_ok());
        assert!(all_close(&[1000.0 + 2.0], &[1000.0], 0.0, 1e-3).is_err());
    }
}

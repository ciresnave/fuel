//! Typed byte-shaped kernels — Phase 7.5 B5.
//!
//! These kernels operate on [`CpuStorageBytes`] (bytes-based CPU
//! storage). They take typed slices via `bytemuck::cast_slice` /
//! `as_slice<T>` / `as_slice_mut<T>`, do the per-element work, and
//! return.
//!
//! These are the per-T monomorphic units that the dispatch wrapper
//! in `fuel_storage::dispatch::cpu_wrappers` calls after extracting
//! the `CpuStorageBytes` from a `BackendStorage::Cpu(...)` variant.
//!
//! ## Status
//!
//! B5 shipped the proof-of-concept (`add_f32`); Phase C grows the
//! coverage matrix family-by-family. Today's additions:
//!
//! - elementwise binary `f32`: `add`, `sub`, `mul`, `div`
//! - elementwise unary `f32`: `relu`, `neg`, `sqr`, `sqrt`, `recip`,
//!   `abs`, `tanh`

use crate::byte_storage::CpuStorageBytes;
use fuel_core_types::{Error, Result};

/// Verify three byte buffers have matching lengths.
fn check_lens_3(name: &str, a: usize, b: usize, c: usize) -> Result<()> {
    if a != b || a != c {
        return Err(Error::Msg(format!(
            "{name}: byte length mismatch (lhs={a}, rhs={b}, out={c})",
        ))
        .bt());
    }
    Ok(())
}

/// Verify two byte buffers have matching lengths (unary kernels).
fn check_lens_2(name: &str, a: usize, b: usize) -> Result<()> {
    if a != b {
        return Err(Error::Msg(format!(
            "{name}: byte length mismatch (input={a}, output={b})",
        ))
        .bt());
    }
    Ok(())
}

// =============================================================================
// Elementwise binary kernels (f32)
// =============================================================================

/// Generate a binary f32 kernel of the form
/// `out[i] = op(lhs[i], rhs[i])`.
///
/// Output is pre-allocated by the caller (the dispatch wrapper) and
/// must match the input byte length. The kernel writes into the
/// pre-allocated bytes; it never allocates.
macro_rules! binary_f32_kernel {
    ($name:ident, $op:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(
            lhs: &CpuStorageBytes,
            rhs: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
        ) -> Result<()> {
            check_lens_3(stringify!($name), lhs.len_bytes(), rhs.len_bytes(), out.len_bytes())?;
            let lhs_view: &[f32] = lhs.as_slice()?;
            let rhs_view: &[f32] = rhs.as_slice()?;
            let out_view: &mut [f32] = out.as_slice_mut()?;
            let op: fn(f32, f32) -> f32 = $op;
            for (i, slot) in out_view.iter_mut().enumerate() {
                *slot = op(lhs_view[i], rhs_view[i]);
            }
            Ok(())
        }
    };
}

binary_f32_kernel!(add_f32, |a, b| a + b, "Elementwise `f32` addition: `out[i] = lhs[i] + rhs[i]`.");
binary_f32_kernel!(sub_f32, |a, b| a - b, "Elementwise `f32` subtraction: `out[i] = lhs[i] - rhs[i]`.");
binary_f32_kernel!(mul_f32, |a, b| a * b, "Elementwise `f32` multiplication: `out[i] = lhs[i] * rhs[i]`.");
binary_f32_kernel!(div_f32, |a, b| a / b, "Elementwise `f32` division: `out[i] = lhs[i] / rhs[i]`. Division by zero yields IEEE-754 inf/NaN per platform.");

// =============================================================================
// Elementwise unary kernels (f32)
// =============================================================================

/// Generate a unary f32 kernel of the form `out[i] = op(input[i])`.
///
/// Output is pre-allocated by the caller and must match the input
/// byte length.
macro_rules! unary_f32_kernel {
    ($name:ident, $op:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(input: &CpuStorageBytes, out: &mut CpuStorageBytes) -> Result<()> {
            check_lens_2(stringify!($name), input.len_bytes(), out.len_bytes())?;
            let in_view: &[f32] = input.as_slice()?;
            let out_view: &mut [f32] = out.as_slice_mut()?;
            let op: fn(f32) -> f32 = $op;
            for (i, slot) in out_view.iter_mut().enumerate() {
                *slot = op(in_view[i]);
            }
            Ok(())
        }
    };
}

unary_f32_kernel!(relu_f32, |x| x.max(0.0), "Elementwise `f32` ReLU: `out[i] = max(0, input[i])`.");
unary_f32_kernel!(neg_f32, |x| -x, "Elementwise `f32` negation: `out[i] = -input[i]`.");
unary_f32_kernel!(sqr_f32, |x| x * x, "Elementwise `f32` square: `out[i] = input[i] * input[i]`.");
unary_f32_kernel!(sqrt_f32, |x| x.sqrt(), "Elementwise `f32` square root: `out[i] = sqrt(input[i])`. Negative inputs yield NaN per IEEE-754.");
unary_f32_kernel!(recip_f32, |x| 1.0 / x, "Elementwise `f32` reciprocal: `out[i] = 1 / input[i]`. Zero input yields IEEE-754 inf/NaN.");
unary_f32_kernel!(abs_f32, |x: f32| x.abs(), "Elementwise `f32` absolute value: `out[i] = |input[i]|`.");
unary_f32_kernel!(tanh_f32, |x: f32| x.tanh(), "Elementwise `f32` hyperbolic tangent: `out[i] = tanh(input[i])`.");

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: add two f32 storages elementwise.
    #[test]
    fn add_f32_round_trip() {
        let a = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let b = CpuStorageBytes::from_slice(&[10.0_f32, 20.0, 30.0, 40.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(16);

        add_f32(&a, &b, &mut out).expect("add");

        let result: &[f32] = out.as_slice().unwrap();
        assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    }

    /// Mismatched byte counts produce an error, not a panic.
    #[test]
    fn add_f32_errors_on_size_mismatch() {
        let a = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let b = CpuStorageBytes::from_slice(&[10.0_f32, 20.0, 30.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(8);

        let result = add_f32(&a, &b, &mut out);
        assert!(result.is_err(), "size mismatch must error");
    }

    #[test]
    fn sub_mul_div_f32_round_trip() {
        let a = CpuStorageBytes::from_slice(&[10.0_f32, 20.0, 30.0]);
        let b = CpuStorageBytes::from_slice(&[1.0_f32, 4.0, 5.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(12);

        sub_f32(&a, &b, &mut out).expect("sub");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[9.0, 16.0, 25.0]);

        mul_f32(&a, &b, &mut out).expect("mul");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[10.0, 80.0, 150.0]);

        div_f32(&a, &b, &mut out).expect("div");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[10.0, 5.0, 6.0]);
    }

    #[test]
    fn binary_f32_size_mismatch_errors() {
        let a = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let b = CpuStorageBytes::from_slice(&[1.0_f32]);
        let mut out = CpuStorageBytes::from_zero_bytes(8);
        for f in [sub_f32, mul_f32, div_f32] {
            assert!(f(&a, &b, &mut out).is_err(), "size mismatch must error");
        }
    }

    #[test]
    fn relu_f32_clips_negatives() {
        let input = CpuStorageBytes::from_slice(&[-1.0_f32, 0.0, 0.5, -3.5, 7.25]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        relu_f32(&input, &mut out).expect("relu");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[0.0, 0.0, 0.5, 0.0, 7.25]);
    }

    #[test]
    fn unary_f32_round_trip() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, -2.0, 4.0, 0.5]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());

        neg_f32(&input, &mut out).expect("neg");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[-1.0, 2.0, -4.0, -0.5]);

        sqr_f32(&input, &mut out).expect("sqr");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 4.0, 16.0, 0.25]);

        let pos = CpuStorageBytes::from_slice(&[1.0_f32, 4.0, 9.0, 16.0]);
        let mut sqrt_out = CpuStorageBytes::from_zero_bytes(pos.len_bytes());
        sqrt_f32(&pos, &mut sqrt_out).expect("sqrt");
        assert_eq!(sqrt_out.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

        recip_f32(&pos, &mut sqrt_out).expect("recip");
        assert_eq!(sqrt_out.as_slice::<f32>().unwrap(), &[1.0, 0.25, 1.0 / 9.0, 1.0 / 16.0]);

        abs_f32(&input, &mut out).expect("abs");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 2.0, 4.0, 0.5]);
    }

    #[test]
    fn tanh_f32_at_known_points() {
        let input = CpuStorageBytes::from_slice(&[0.0_f32, 1.0, -1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        tanh_f32(&input, &mut out).expect("tanh");
        let result: &[f32] = out.as_slice().unwrap();
        assert!((result[0] - 0.0).abs() < 1e-7);
        assert!((result[1] - 1.0_f32.tanh()).abs() < 1e-7);
        assert!((result[2] - (-1.0_f32).tanh()).abs() < 1e-7);
    }

    #[test]
    fn unary_f32_size_mismatch_errors() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4); // wrong size
        for f in [relu_f32, neg_f32, sqr_f32, sqrt_f32, recip_f32, abs_f32, tanh_f32] {
            assert!(f(&input, &mut out).is_err(), "size mismatch must error");
        }
    }
}

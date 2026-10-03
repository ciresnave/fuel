// SPDX-License-Identifier: MIT OR Apache-2.0
//! Useful functions for checking features.
//!
//! `get_num_threads`/`with_avx`/`with_neon`/`with_f16c`/`cuda_is_available`
//! moved to [`fuel_hardware::utils`] (fuel-core dissolution step 2, Part 5
//! item 2) and are re-exported below so `fuel_core::utils::*` / `fuel::utils::*`
//! call sites are unchanged. `has_accelerate`/`has_mkl`/`metal_is_available`
//! stay here deliberately: fuel-hardware has no matching
//! `accelerate`/`mkl`/`metal` feature, so moving them would make their
//! `cfg!(feature = ...)` check silently and permanently evaluate to
//! `false` regardless of how `fuel`/`fuel-core` was actually built. Final
//! home TBD — most likely the backend crate that owns each feature
//! (fuel-cpu-backend for accelerate/mkl, fuel-metal-backend for metal) once
//! that census is done; tracked, not resolved, here.
pub use fuel_hardware::utils::*;

/// Returns `true` if the crate was compiled with Apple Accelerate support.
///
/// # Example
///
/// ```rust
/// use fuel_core::utils::has_accelerate;
/// // Returns true only when built with the `accelerate` feature on macOS.
/// let _ = has_accelerate();
/// ```
pub fn has_accelerate() -> bool {
    cfg!(feature = "accelerate")
}

/// Returns `true` if the crate was compiled with Intel MKL support.
///
/// # Example
///
/// ```rust
/// use fuel_core::utils::has_mkl;
/// let _ = has_mkl();
/// ```
pub fn has_mkl() -> bool {
    cfg!(feature = "mkl")
}

/// Returns `true` if the crate was compiled with Apple Metal support.
///
/// # Example
///
/// ```rust
/// use fuel_core::utils::metal_is_available;
/// let _ = metal_is_available();
/// ```
pub fn metal_is_available() -> bool {
    cfg!(feature = "metal")
}

#[cfg(test)]
mod tests {
    // Only brought in by a test below; under default features (none of
    // accelerate/mkl/metal on) every test in this module is cfg'd out, and
    // an unconditional `use super::*;` would then be unused — the clippy
    // failure this comment exists to explain if it recurs.
    #[cfg(any(feature = "accelerate", feature = "mkl", feature = "metal"))]
    use super::*;

    // Positive-control tripwires for the feature landmine this move surfaced
    // (fuel-core dissolution step 2): each of these three functions must read
    // `true` when its OWN feature is the one enabling the build, not some
    // unrelated or absent feature on whatever crate hosts the check. Each
    // test only compiles under its matching feature, so it is silent (not
    // false) when that feature is off — same shape as the rest of this
    // crate's feature-gated test modules.
    #[cfg(feature = "accelerate")]
    #[test]
    fn has_accelerate_is_true_under_its_own_feature() {
        assert!(has_accelerate());
    }

    #[cfg(feature = "mkl")]
    #[test]
    fn has_mkl_is_true_under_its_own_feature() {
        assert!(has_mkl());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_is_available_is_true_under_its_own_feature() {
        assert!(metal_is_available());
    }
}

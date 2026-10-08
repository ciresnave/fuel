// SPDX-License-Identifier: MIT OR Apache-2.0
//! Useful functions for checking features.
//!
//! `get_num_threads`/`with_avx`/`with_neon`/`with_f16c`/`cuda_is_available`
//! moved to [`fuel_hardware::utils`] (fuel-core dissolution step 2, Part 5
//! item 2) and are re-exported below so `fuel_core::utils::*` / `fuel::utils::*`
//! call sites are unchanged.
//!
//! `has_accelerate`/`has_mkl` moved to `fuel-cpu-backend` (fuel-core
//! dissolution, GAP-347 PR 7): that crate is a REQUIRED (non-optional)
//! dependency of this one, and its own `accelerate`/`mkl` features are
//! exactly what this crate's `accelerate`/`mkl` features forward to
//! (`fuel-core/Cargo.toml`), so re-exporting unconditionally preserves
//! `cfg!(feature = ...)` parity with the pre-move behavior.
//!
//! `metal_is_available` moved to `fuel-metal-backend`, which — unlike
//! `fuel-cpu-backend` — is an OPTIONAL dependency gated behind this
//! crate's own `metal` feature (`dep:fuel-metal-backend`). A plain
//! unconditional `pub use` would fail to resolve whenever `metal` is off,
//! so the re-export is itself feature-gated, with a `false`-returning stub
//! for the off case — together they reproduce the exact
//! `cfg!(feature = "metal")` truth table the original local function had.
pub use fuel_hardware::utils::*;

pub use fuel_cpu_backend::{has_accelerate, has_mkl};

#[cfg(feature = "metal")]
pub use fuel_metal_backend::metal_is_available;

/// `fuel-metal-backend` isn't in the dependency graph at all without the
/// `metal` feature (it's an `optional = true`, `dep:`-gated dependency),
/// so there is no real implementation to call here — `false` is correct
/// by construction, not a guess.
#[cfg(not(feature = "metal"))]
pub fn metal_is_available() -> bool {
    false
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
    // (fuel-core dissolution step 2, carried through GAP-347 PR 7): each of
    // these three functions must read `true` when its OWN feature is the
    // one enabling the build, not some unrelated or absent feature on
    // whatever crate hosts the check. Each test only compiles under its
    // matching feature, so it is silent (not false) when that feature is
    // off — same shape as the rest of this crate's feature-gated test
    // modules.
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

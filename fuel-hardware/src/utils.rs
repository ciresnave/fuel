// SPDX-License-Identifier: MIT OR Apache-2.0
//! Useful functions for checking features.
//!
//! Moved from `fuel-core` (fuel-core dissolution step 2) — these five are the
//! subset of the original file whose `cfg!` checks mean the same thing here
//! as they did in `fuel-core`: target-arch flags are crate-independent, and
//! `cuda` is a feature fuel-core's own `cuda` feature already forwards to
//! `fuel-hardware/cuda`. `has_accelerate`/`has_mkl`/`metal_is_available`
//! stayed behind in `fuel-core::utils` — fuel-hardware has no matching
//! `accelerate`/`mkl`/`metal` features, so moving them here would silently
//! make them return `false` forever regardless of how the build was
//! actually configured.
use std::str::FromStr;

/// Returns the number of threads to use for parallel CPU operations.
///
/// Reads the `RAYON_NUM_THREADS` environment variable; falls back to the number of logical CPUs.
///
/// # Example
///
/// ```rust
/// use fuel_hardware::utils::get_num_threads;
/// let n = get_num_threads();
/// assert!(n >= 1);
/// ```
pub fn get_num_threads() -> usize {
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}

/// Returns `true` if the crate was compiled with CUDA support.
///
/// # Example
///
/// ```rust
/// use fuel_hardware::utils::cuda_is_available;
/// // Only true when built with `--features cuda`.
/// let _ = cuda_is_available();
/// ```
pub fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}

/// Returns `true` if the binary was compiled targeting the `avx2` CPU feature.
///
/// # Example
///
/// ```rust
/// use fuel_hardware::utils::with_avx;
/// let _ = with_avx();
/// ```
pub fn with_avx() -> bool {
    cfg!(target_feature = "avx2")
}

/// Returns `true` if the binary was compiled targeting the ARM `neon` CPU feature.
///
/// # Example
///
/// ```rust
/// use fuel_hardware::utils::with_neon;
/// let _ = with_neon();
/// ```
pub fn with_neon() -> bool {
    cfg!(target_feature = "neon")
}

/// Returns `true` if the binary was compiled targeting the x86 `f16c` CPU feature.
///
/// # Example
///
/// ```rust
/// use fuel_hardware::utils::with_f16c;
/// let _ = with_f16c();
/// ```
pub fn with_f16c() -> bool {
    cfg!(target_feature = "f16c")
}

//! Useful functions for checking features.
use std::str::FromStr;

/// Returns the number of threads to use for parallel CPU operations.
///
/// Reads the `RAYON_NUM_THREADS` environment variable; falls back to the number of logical CPUs.
///
/// # Example
///
/// ```rust
/// use candle_core::utils::get_num_threads;
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

/// Returns `true` if the crate was compiled with Apple Accelerate support.
///
/// # Example
///
/// ```rust
/// use candle_core::utils::has_accelerate;
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
/// use candle_core::utils::has_mkl;
/// let _ = has_mkl();
/// ```
pub fn has_mkl() -> bool {
    cfg!(feature = "mkl")
}

/// Returns `true` if the crate was compiled with CUDA support.
///
/// # Example
///
/// ```rust
/// use candle_core::utils::cuda_is_available;
/// // Only true when built with `--features cuda`.
/// let _ = cuda_is_available();
/// ```
pub fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}

/// Returns `true` if the crate was compiled with Apple Metal support.
///
/// # Example
///
/// ```rust
/// use candle_core::utils::metal_is_available;
/// let _ = metal_is_available();
/// ```
pub fn metal_is_available() -> bool {
    cfg!(feature = "metal")
}

/// Returns `true` if the binary was compiled targeting the `avx2` CPU feature.
///
/// # Example
///
/// ```rust
/// use candle_core::utils::with_avx;
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
/// use candle_core::utils::with_neon;
/// let _ = with_neon();
/// ```
pub fn with_neon() -> bool {
    cfg!(target_feature = "neon")
}

/// Returns `true` if the binary was compiled targeting the WebAssembly `simd128` feature.
///
/// # Example
///
/// ```rust
/// use candle_core::utils::with_simd128;
/// let _ = with_simd128();
/// ```
pub fn with_simd128() -> bool {
    cfg!(target_feature = "simd128")
}

/// Returns `true` if the binary was compiled targeting the x86 `f16c` CPU feature.
///
/// # Example
///
/// ```rust
/// use candle_core::utils::with_f16c;
/// let _ = with_f16c();
/// ```
pub fn with_f16c() -> bool {
    cfg!(target_feature = "f16c")
}

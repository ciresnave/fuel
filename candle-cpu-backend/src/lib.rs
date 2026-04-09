//! CPU backend computation kernels for candle.
//!
//! This crate contains the core CPU computation helpers (MatMul, pooling,
//! convolution, reductions, etc.) extracted from `candle-core` so they can be
//! reused and tested independently. It also re-exports the MKL and Accelerate
//! FFI bindings when the corresponding features are enabled.

#[cfg(feature = "accelerate")]
pub mod accelerate;
#[cfg(feature = "mkl")]
pub mod mkl;

#[allow(dead_code)] // Not yet wired to candle-core delegation; kept for future use
pub mod conv2d;
pub mod ops;
pub mod utils;

pub use ops::*;
pub use utils::*;

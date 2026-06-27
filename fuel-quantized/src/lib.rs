//! Backend-agnostic quantized (ggml/gguf block-format) numerics.
//!
//! This crate owns the format-side of fuel's quantization stack:
//!
//! - [`k_quants`] — `BlockQX` block types, the [`GgmlType`](k_quants::GgmlType)
//!   trait, and scalar reference impls of every fmt fuel supports.
//! - [`avx`] / [`neon`] / [`simd128`] — cfg-gated CPU SIMD `vec_dot` helpers
//!   used by `GgmlType::vec_dot` impls in `k_quants`. They live here (rather
//!   than in `fuel-cpu-backend`) because the orphan rule pins them to the
//!   crate that defines `BlockQX`.
//! - [`utils`] — quantization-time helpers shared by k_quants impls.
//! - [`cpu`] — CPU-side adapters that bridge `Vec<BlockQX>` to the
//!   backend-agnostic [`fuel_backend_contract::quantized::DynQuantizedStorage`]
//!   trait, plus the `QuantizedDeviceKernels` impl on
//!   `fuel_cpu_backend::CpuBackendDevice`.
//!
//! Per-backend (CUDA, Metal, future Vulkan) quantized fast paths live in
//! their own crates and implement the same trait pair from
//! `fuel-core-types`.

#[cfg(target_feature = "avx2")]
pub mod avx;
pub mod cpu;
pub mod k_quants;
#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(target_feature = "simd128")]
pub mod simd128;
pub mod utils;

pub use cpu::{QuantizedType, as_t_slice, cpu_from_data, cpu_zeros};
pub use k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1, BlockQ6K,
    BlockQ8K, BlockQ8_0, BlockQ8_1, GgmlType, K_SCALE_SIZE, QK4_0, QK4_1, QK5_0, QK5_1, QK8_0,
    QK8_1, QK_K, matmul, matmul_f16,
};

// Re-export GgmlDType so downstream callers can write
// `fuel_quantized::GgmlDType` without naming fuel-core-types.
pub use fuel_backend_contract::quantized::{DynQuantizedStorage, QuantizedDeviceKernels};
pub use fuel_ir::quantized::GgmlDType;

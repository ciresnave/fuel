//! WGSL compute shader source code, shared across all GPU backends.
//!
//! These shaders are written in WGSL (WebGPU Shading Language) and
//! can be compiled to:
//! - **SPIR-V** for Vulkan (via naga)
//! - **MSL** for Metal (via naga)
//! - **HLSL** for DirectX (via naga)
//!
//! Each backend compiles the same source to its native shader format
//! at init time. One set of shaders, all GPU backends.

/// Element-wise unary ops (13 ops via push-constant selector).
pub const UNARY: &str = include_str!("shaders/unary.wgsl");

/// Element-wise binary ops (6 ops via push-constant selector).
pub const BINARY: &str = include_str!("shaders/binary.wgsl");

/// Affine transform: y = x * mul + add.
pub const AFFINE: &str = include_str!("shaders/affine.wgsl");

/// Tiled matrix multiply with 4x4 register tiling.
pub const MATMUL: &str = include_str!("shaders/matmul.wgsl");

/// Fused softmax along the last dimension (per-row).
pub const SOFTMAX: &str = include_str!("shaders/softmax.wgsl");

/// Parallel reduction (sum/max/min of all elements).
pub const REDUCE: &str = include_str!("shaders/reduce.wgsl");

//! CPU backend computation kernels for fuel.
//!
//! This crate contains the core CPU computation helpers (MatMul, pooling,
//! convolution, reductions, etc.) extracted from `fuel-core` so they can be
//! reused and tested independently. It also re-exports the MKL and Accelerate
//! FFI bindings when the corresponding features are enabled.

#[cfg(feature = "accelerate")]
pub mod accelerate;
#[cfg(feature = "mkl")]
pub mod mkl;

#[allow(dead_code)] // Not yet wired to fuel-core delegation; kept for future use
pub mod conv2d;
pub mod dyn_impl;
pub mod host_storage;
pub mod ops;
pub mod probe;
pub mod quantized;
pub mod system_memory;
pub mod utils;

/// Phase 7.5 storage-unification target: byte-shaped CPU storage
/// with `Arc<AlignedBytes>` backing, 64-byte alignment, and CoW
/// mutation. Coexists with the legacy `CpuStorage` (HostBuffer-based)
/// during op-kernel migration; legacy retires when the last kernel
/// migrates.
pub mod byte_storage;

/// Typed byte-shaped kernels — Phase 7.5 B5. These operate on
/// `CpuStorageBytes` directly via `bytemuck` typed views. The
/// dispatch wrapper in `fuel_dispatch::dispatch::cpu_wrappers`
/// extracts `CpuStorageBytes` from `BackendStorage::Cpu(...)` and
/// calls these kernels.
pub mod byte_kernels;

/// Trait-chassis surface shared by elementwise + reduction CPU
/// kernels. One shape/loop pass per kernel family; op-specific math
/// lives in tiny trait impls. See [`chassis::reduction`].
pub mod chassis;

pub use byte_storage::{CpuStorageBytes, CPU_ALIGN_BYTES};
pub use dyn_impl::CpuStorage;
pub use ops::*;
pub use quantized::CpuQStorage;
pub use utils::*;

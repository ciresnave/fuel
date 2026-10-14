// SPDX-License-Identifier: MIT OR Apache-2.0
//! # Fuel
//!
//! Fuel is a DAG-first, lazy-only ML framework for Rust. This crate is the
//! **public facade** for the framework: it re-exports the public surface that
//! `fuel-core` used to carry, from the crates that own those items now
//! (`fuel-tensor`, `fuel-ir`, `fuel-hardware`, `fuel-dispatch`,
//! `fuel-backend-contract`, `fuel-cpu-backend`, `fuel-loaders`, `fuel-kv-pool`,
//! `fuel-transformers`), so `use fuel::...` resolves exactly as it did when
//! `fuel` re-exported `fuel-core` (GAP-348, the fuel-core dissolution; `fuel-core`
//! itself no longer exists). Feature flags are forwarded to the owning crates
//! (see this crate's manifest).
//!
//! See `docs/restructure-migration-design.md` §Stage 1.

// The surface `fuel-core` used to carry, written out against the crates that
// own each item (GAP-348). Feature-gated items are gated on this crate's
// forwarded features exactly as they were on `fuel-core`'s.
//
// The `lazy_<model>` modules come from `fuel_transformers::models::*` below;
// `tests/facade_disjoint.rs` asserts that set is disjoint from the names here.
pub use fuel_dispatch::topology;
pub use fuel_hardware::probe;
pub use fuel_hardware::transfer_cost;
pub use fuel_tensor::metal_backend;
#[cfg(feature = "vulkan")]
pub use fuel_tensor::vulkan_backend;
pub use fuel_tensor::{
    cuda_backend, decode_shape, factories, inference_context, judge, kv_block_pool_device, lazy,
    lazy_latent_cache, nf4, persistent_decode, pipelined_bridge, planner, scheduling, test_utils,
};

/// Error types, re-exported from `fuel-ir`.
pub mod error {
    pub use fuel_ir::bail;
    pub use fuel_ir::error::{Context, Error, MatMulUnexpectedStriding, Result, zip};
}
pub use error::{Context, Error, Result};

/// Hardware / build-feature probes.
pub mod utils {
    pub use fuel_cpu_backend::{has_accelerate, has_mkl};
    pub use fuel_hardware::utils::*;

    /// `true` iff this build has the `metal` feature (the Metal backend is
    /// compiled in). Like the function it replaces, it is deliberately not
    /// platform-gated: forcing `--features metal` elsewhere returns `true`.
    pub fn metal_is_available() -> bool {
        cfg!(feature = "metal")
    }
}

pub use fuel_backend_contract::Storage;
pub use fuel_ir::layout::Layout;
pub use fuel_ir::shape::{D, Shape};
pub use fuel_ir::strided_index::{StridedBlocks, StridedIndex};
pub use fuel_ir::{CpuStorage, CpuStorageRef, HostBuffer, HostBufferRef};
pub use fuel_tensor::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use fuel_tensor::{Device, DeviceLocation, NdArray};

#[cfg(feature = "cuda")]
pub use fuel_tensor::builder_arg;
#[doc(hidden)]
#[cfg(feature = "cuda")]
pub use fuel_tensor::cuda_backend as cuda;
#[cfg(feature = "cuda")]
pub use fuel_tensor::cuda_backend::{CudaDevice, CudaStorage};
#[cfg(feature = "metal")]
pub use fuel_tensor::metal_backend::{MetalDevice, MetalError, MetalStorage};

// Modules that left `fuel-core` earlier in the dissolution (GAP-347) and are
// re-exported here, from the crates that own them, so `fuel::<name>` callers are
// unaffected. All are unconditional except `telemetry`, which keeps the feature
// gate it always had.
#[cfg(feature = "telemetry")]
pub use fuel_dispatch::telemetry::judge_sink as telemetry;

pub use fuel_kv_pool as kv_block_pool;

pub use fuel_loaders::hf_config;

pub use fuel_loaders::safetensors;

pub use fuel_loaders::quantized;

// `#[macro_export]` macros live at the CRATE ROOT of their defining crate, in
// the MACRO namespace, which a glob re-export does not carry. `bail!` is the
// framework's one exported macro (used in ~60 consumer files); it is defined in
// `fuel-ir` and re-exported explicitly here.
pub use fuel_ir::bail;

// The model zoo lives in fuel-transformers (Models tier). Re-exported so
// `fuel::lazy_bert` etc. resolve. `fuel_transformers::models` is exactly the
// `lazy_<model>` modules, disjoint from the names above (`lazy` and
// `lazy_latent_cache` are not `lazy_<model>` modules) -- asserted in
// tests/facade_disjoint.rs.
pub use fuel_transformers::models::*;

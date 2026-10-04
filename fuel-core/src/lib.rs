// SPDX-License-Identifier: MIT OR Apache-2.0
//! ML framework for Rust
//!
//! ```rust
//! use fuel_core::lazy::{Tensor, realize_many_f32};
//! use fuel_core::Device;
//! # use fuel_core::Error;
//! # fn main() -> Result<(), Error> {
//! let dev = Device::cpu();
//!
//! // Every tensor is a node in a lazy graph. The first `from_*` call mints the
//! // graph; a second operand joins it with `from_*_on(a.graph(), ..)` — ops
//! // require both operands to share one graph.
//! let a = Tensor::from_f32((0..6).map(|x| x as f32).collect::<Vec<_>>(), (2, 3), &dev)?;
//! let b = Tensor::from_f32_on(a.graph(), (0..12).map(|x| x as f32).collect::<Vec<_>>(), (3, 4), &dev)?;
//! let c = a.matmul(&b)?;
//! assert_eq!(c.shape().dims(), &[2, 4]);
//!
//! // Nothing has executed yet — `realize_*` is what runs the graph.
//! let out = realize_many_f32(&[&c]);
//! assert_eq!(out[0].len(), 8);
//! # Ok(())}
//! ```
//!
//! ## Features
//!
//! - Simple syntax (looks and feels like PyTorch)
//! - CPU and Cuda backends (and M1 support)
//! - Enable serverless (CPU) small and fast deployments
//! - Model training
//! - Distributed computing (NCCL).
//! - Models out of the box (Llama, Whisper, Falcon, ...)
//!
//! ## FAQ
//!
//! - Why Fuel?
//!
//! Fuel stems from the need to reduce binary size in order to *enable serverless*
//! possible by making the whole engine smaller than PyTorch very large library volume
//!
//! And simply *removing Python* from production workloads.
//! Python can really add overhead in more complex workflows and the [GIL](https://www.backblaze.com/blog/the-python-gil-past-present-and-future/) is a notorious source of headaches.
//!
//! Rust is cool, and a lot of the HF ecosystem already has Rust crates [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers)
//!
//! ## Other Crates
//!
//! Fuel consists of a number of crates. This crate holds core the common data structures but you may wish
//! to look at the docs for the other crates which can be found here:
//!
//! - [fuel-core](https://github.com/ciresnave/fuel/tree/main/fuel-core). Core Datastructures and DataTypes.
//! - [fuel-nn](https://github.com/ciresnave/fuel/tree/main/fuel-nn). Building blocks for Neural Nets.
//! - [fuel-datasets](https://github.com/ciresnave/fuel/tree/main/fuel-datasets). Rust access to commonly used Datasets like MNIST.
//! - [fuel-examples](https://github.com/ciresnave/fuel/tree/main/fuel-examples). Examples of Fuel in Use.
//! - [fuel-onnx](https://github.com/ciresnave/fuel/tree/main/fuel-onnx). Loading and using ONNX models.
//! - `fuel-pyo3`. Access to Fuel from Python.
//! - [fuel-transformers](https://github.com/ciresnave/fuel/tree/main/fuel-transformers). Fuel implementation of many published transformer models.
//!

// GAP-229: `clippy::identity_op` fires 128x across fuel-core+fuel-dispatch and is a
// defect in 0 of 128 — it measures a house idiom, not debt, so it is allowed at the
// crate root (a MEASURED claim: "identity ops in THIS crate are intentional"; a firing
// in a third crate is a deliberate TRIPWIRE — it reds the gate so someone looks and rules).
// Two intentional classes:
//   * DOC-INDEX: an explicit `0 *`/`1 *` NAMES an index (`idx[i*nb + 0]`, `got[1*head_dim + j]`);
//     the `0 *` partner is separately load-bearing (CLAUDE.md: never delete it), so its `1 *`
//     sibling must survive too — auto-fixing it would strand a bare unexplained `0 *`.
//   * DOC-SHAPE: an explicit unit/batch dim in a shape product (`1 * C * H * W`) mirrors
//     `Shape::from_dims(&[1, C, H, W])`.
// RESIDUAL, named not gated: this ALSO hides FLOAT identity ops, where `x + 0.0` normalizes
// -0.0 -> +0.0 — a real hazard whose population is ZERO today (measured). A float identity op
// landing later is silently admitted; re-measure at the next rust-toolchain.toml pin bump
// (owner-tracked, docs/gaps.md GAP-229) and drop this allow if precision is no longer 0/N.
#![allow(clippy::identity_op)]

// `backend.rs` deleted (PR-D, board #109 follow-up): it was a pure
// `pub use fuel_backend_contract::backend::HostStorage;` compat re-export
// with zero consumers via `fuel_core::backend::`/`fuel::backend::` module
// path anywhere in this repo (every other `*::backend::` hit found was a
// DIFFERENT crate's own local `backend` module — fuel-ir's, fuel-backend-
// contract's — not this one). `HostStorage` itself was never re-exported
// at this crate's root, so there is nothing to replace this module path
// with; a caller wanting it uses `fuel_backend_contract::backend::HostStorage`
// directly, as every real consumer already does.
pub mod error;
pub mod hf_config;
// `cuda_backend`/`device`/`dtype`/`lazy`/`lazy_latent_cache`/`metal_backend`/
// `vulkan_backend`/`decode_shape`/`factories`/`inference_context`/
// `kv_block_pool_device`/`persistent_decode`/`judge`/`pipelined_bridge`/
// `planner`/`scheduling`/`nf4`/`test_utils` all moved to `fuel-tensor`
// (board #109, one PR: they are a single mutually-referencing graph,
// measured via a full crate::-reference census before any file moved).
// Re-exported below so every `fuel_core::<module>::*` / `fuel::<module>::*`
// call site is unchanged. See docs/release-0.13.0-wave.md.
pub use fuel_tensor::metal_backend;
#[cfg(feature = "vulkan")]
pub use fuel_tensor::vulkan_backend;
pub use fuel_tensor::{
    cuda_backend, decode_shape, factories, inference_context, judge, kv_block_pool_device, lazy,
    lazy_latent_cache, nf4, persistent_decode, pipelined_bridge, planner, scheduling, test_utils,
};
pub mod kv_block_pool;
// `telemetry` deleted (fuel-core dissolution, GAP-347 PR 1): it was a pure
// `pub use fuel_dispatch::telemetry::judge_sink::*;` shim with zero real
// external consumers. `fuel`'s facade now re-exports the real module
// directly (`fuel/src/lib.rs`); the `telemetry` Cargo feature here is kept
// as a pass-through (`fuel-dispatch/telemetry` + `fuel-tensor/telemetry`)
// in case anything still enables it transitively through this crate.
/// `SystemTopology` moved to `fuel-dispatch::topology` (retirement B0.2c — it fuses
/// the dispatch overlay with fuel-hardware discovery); re-exported so
/// `crate::topology` / `fuel_core::topology` callers are unchanged.
pub use fuel_dispatch::topology;
/// Hardware discovery moved to the `fuel-hardware` crate (retirement B0.2);
/// re-exported here so `fuel_core::probe` / `crate::probe` callers are unchanged.
pub use fuel_hardware::probe;
/// Transfer (bandwidth) calibration moved to `fuel-hardware` (retirement B0.2b);
/// re-exported so `crate::transfer_cost` / `fuel_core::transfer_cost` is unchanged.
pub use fuel_hardware::transfer_cost;
// `train.rs` moved to `fuel-training/src/train.rs` (board #109 follow-up,
// PM-approved after Row 5's original "no below-consumer; PASSES" was found
// stale: `fuel::train`'s real consumers are at the FACADE level, outside
// fuel-core's own graph). NO compat re-export is possible here -- a
// fuel-core re-export of fuel-training creates fuel-core -> fuel-training,
// and fuel-training already depends on `fuel` -> `fuel-core`: a real cycle.
// `fuel::train` is therefore deliberately retired; callers use
// `fuel_training::train` directly. Both former consumers
// (fuel-examples/src/mnist_train.rs, fuel-lazy-examples's
// llama-finetune-vulkan.rs) updated in the same change.
pub mod quantized;
pub mod safetensors;
pub mod utils;

#[cfg(feature = "cudnn")]
pub use fuel_tensor::cuda_backend::cudnn;

// `cpu_backend/mod.rs` deleted (fuel-core dissolution, Part 1 shims): zero
// consumers via fuel_core::cpu_backend::/fuel::cpu_backend:: (module path)
// or the root re-export below, anywhere in this repo or across 5 sibling
// repos -- the module's own doc comment claimed fuel-nn called `unary_map`
// through it; that claim was stale, not current. `Map1`/`Map1Any`/`Map2`/
// `Map2InPlace`/`Map2U8`/`binary_map`/`binary_map_vec`/`unary_map`/
// `unary_map_vec` were never re-exported at crate root and had zero
// consumers at any path, so they are not replaced by anything. The 4 root
// types ARE re-exported directly from fuel_ir below (also zero measured
// consumers, kept anyway for API-surface stability).
pub use error::{Context, Error, Result};
pub use fuel_tensor::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use fuel_tensor::{Device, DeviceLocation, NdArray};
// `layout.rs`/`storage.rs`/`strided_index.rs`/`dyn_backend.rs`/`shape.rs`/
// `cpu_backend/mod.rs` all deleted (fuel-core dissolution, Part 1 shims):
// zero consumers via fuel_core::<module>::*, fuel::<module>::*, or the root
// re-export below, anywhere in this repo or across 5 sibling repos
// (fresh-fetched, checked module-path + root-qualified + grouped-import
// forms, positive-controlled). shape.rs's 2 tests (stride/test_from_tuple)
// are byte-identical duplicates of tests already in fuel_ir/src/shape.rs;
// cpu_backend/mod.rs's own doc comment claiming a fuel-nn consumer was
// stale. Root re-exports inlined directly from their real homes so
// crate::Layout/fuel_core::Shape/fuel::StridedIndex/fuel::CpuStorage (etc.)
// are unchanged for external callers; device.rs (the one internal consumer
// that used a module path) moved to fuel-tensor along with Device itself.
pub use fuel_backend_contract::Storage;
pub use fuel_ir::layout::Layout;
pub use fuel_ir::shape::{D, Shape};
pub use fuel_ir::strided_index::{StridedBlocks, StridedIndex};
pub use fuel_ir::{CpuStorage, CpuStorageRef, HostBuffer, HostBufferRef};

// Eager `Tensor` is the runtime data type the executor materializes into.
// New user code should use [`lazy::Tensor`] — the graph builder — and
// realize it via `realize_f32` etc. The eager `Tensor` re-export below is
// kept for backend-adjacent crates (fuel-onnx, fuel-pyo3, fuel-parallel,
// fuel-datasets, fuel-examples helpers) that still shuttle
// device-resident buffers around. Marked `#[doc(hidden)]` so it
// does not appear in generated rustdoc; the canonical path
// `fuel_core::tensor::Tensor` remains accessible for the same callers.
#[doc(hidden)]
#[cfg(feature = "cuda")]
pub use fuel_tensor::cuda_backend as cuda;

#[cfg(feature = "cuda")]
pub use fuel_tensor::cuda_backend::{CudaDevice, CudaStorage};

#[cfg(feature = "cuda")]
pub use fuel_cuda_backend::builder_arg;

#[cfg(feature = "metal")]
pub use fuel_tensor::metal_backend::{MetalDevice, MetalError, MetalStorage};

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub trait ToUsize2 {
    fn to_usize2(self) -> (usize, usize);
}

impl ToUsize2 for usize {
    fn to_usize2(self) -> (usize, usize) {
        (self, self)
    }
}

impl ToUsize2 for (usize, usize) {
    fn to_usize2(self) -> (usize, usize) {
        self
    }
}

// `Module` / `ModuleT` were REMOVED in B6. Both were defined over the eager
// `crate::tensor::Tensor` (`forward(&self, xs: &Tensor) -> Result<Tensor>`), so
// they could not survive its deletion. The lazy stack never adopted them — lazy
// models are plain inherent methods on their weight structs
// (e.g. `LlamaModel::forward`), not trait impls, so there is nothing to port.

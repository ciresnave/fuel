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
//! let a = Tensor::from_f32((0..6).map(|x| x as f32).collect::<Vec<_>>(), (2, 3), &dev);
//! let b = Tensor::from_f32_on(a.graph(), (0..12).map(|x| x as f32).collect::<Vec<_>>(), (3, 4), &dev);
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
//! - [fuel-core](https://docs.rs/fuel-core/). Core Datastructures and DataTypes.
//! - [fuel-nn](https://docs.rs/fuel-nn/). Building blocks for Neural Nets.
//! - [fuel-datasets](https://docs.rs/fuel-datasets/). Rust access to commonly used Datasets like MNIST.
//! - [fuel-examples](https://docs.rs/fuel-examples/). Examples of Fuel in Use.
//! - [fuel-onnx](https://docs.rs/fuel-onnx/). Loading and using ONNX models.
//! - [fuel-pyo3](https://docs.rs/fuel-pyo3/). Access to Fuel from Python.
//! - [fuel-transformers](https://docs.rs/fuel-transformers/). Fuel implementation of many published transformer models.
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

#[cfg(feature = "accelerate")]
mod accelerate;
pub mod backend;
pub mod cpu_backend;
pub mod cuda_backend;
mod device;
mod dtype;
pub mod dyn_backend;
pub mod error;
pub mod hf_config;
pub mod layout;
pub mod lazy;
pub mod lazy_latent_cache;
// `seq_bucketing` removed in Phase 6d: paged attention via
// `Op::PagedAttn` (and `Tensor::paged_attn`) supersedes the
// bucket-and-pad approach. Variable-length decode is now expressed
// directly via per-sequence `context_lens`.
pub mod metal_backend;
#[cfg(feature = "mkl")]
mod mkl;
pub mod model_progress;
#[cfg(feature = "vulkan")]
pub mod vulkan_backend;
// dispatch.rs (Judge cache) moved into judge::cache 2026-05-31 — the
// `fuel_core::dispatch` name was a misnomer for what was just the
// cached output of the Judge. Callers now reach the cache via
// `fuel_core::judge::cached()` / `populate_dispatch_table()` /
// `invalidate()` (re-exported at the judge module's top level).
/// The identity a held decode plan is baked against — what makes reusing a
/// [`inference_context::DecodeSession`] safe across models. Read its module
/// docs before adding anything to the key: over-keying is a silent performance
/// regression, under-keying is a silent wrong answer.
pub mod decode_shape;
/// Per-layer decode-state description (GAP-029 / GAP-166) — the vocabulary that
/// DESCRIBES what state a layer requires rather than ASSERTING that every layer
/// holds per-head K/V. Read its module docs before collapsing a spec to a
/// `(n_kv_heads, head_dim)` pair; the collapse is deliberately fallible.
pub mod decode_state_spec;
pub mod factories;
pub mod inference_context;
pub mod kv_block_pool;
pub mod kv_block_pool_device;
/// The shared persistent-decode rebind driver (GAP-029 2b). Collapses what were
/// two hand-copied 48-line per-model bodies. Read its module docs before adding
/// a model: the Llama/Phi decode-path divergence is preserved deliberately, not
/// an accident to be tidied.
pub mod persistent_decode;
// `multi_session` (the K-way decode scheduler) moved to `fuel-inference` (Q2,
// 2026-07-29): it is consumer-side orchestration, not a Foundation primitive.
// It reaches the model through the `DecodeModel` trait, so it no longer belongs
// in `fuel-core`. See `fuel-inference/src/multi_session.rs`.
pub mod judge;
pub mod pipelined_bridge;
pub mod planner;
/// Baracuda dispatch-telemetry / miss-reporting production consumer — the
/// process-wide opt-in switch, sink, hardware stamp, and explicit-flush API
/// that installs the plan-time [`fuel_dispatch::telemetry`] hooks on the
/// realize path. Behind the `telemetry` cargo feature; off by default.
#[cfg(feature = "telemetry")]
pub mod telemetry;
/// Hardware discovery moved to the `fuel-hardware` crate (retirement B0.2);
/// re-exported here so `fuel_core::probe` / `crate::probe` callers are unchanged.
pub use fuel_hardware::probe;
pub mod scheduling;
/// `SystemTopology` moved to `fuel-dispatch::topology` (retirement B0.2c — it fuses
/// the dispatch overlay with fuel-hardware discovery); re-exported so
/// `crate::topology` / `fuel_core::topology` callers are unchanged.
pub use fuel_dispatch::topology;
/// Transfer (bandwidth) calibration moved to `fuel-hardware` (retirement B0.2b);
/// re-exported so `crate::transfer_cost` / `fuel_core::transfer_cost` is unchanged.
pub use fuel_hardware::transfer_cost;
pub mod nf4;
pub mod quantized;
pub mod safetensors;
pub mod shape;
mod storage;
mod strided_index;
pub mod test_utils;
pub mod train;
pub mod utils;

#[cfg(feature = "cudnn")]
pub use cuda_backend::cudnn;

pub use cpu_backend::{CpuStorage, CpuStorageRef, HostBuffer, HostBufferRef};
pub use device::{Device, DeviceLocation, NdArray};
pub use dtype::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use error::{Context, Error, Result};
pub use layout::Layout;
pub use shape::{D, Shape};
pub use storage::Storage;
pub use strided_index::{StridedBlocks, StridedIndex};

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
pub use cuda_backend as cuda;

#[cfg(feature = "cuda")]
pub use cuda_backend::{CudaDevice, CudaStorage};

#[cfg(feature = "cuda")]
pub use fuel_cuda_backend::builder_arg;

#[cfg(feature = "metal")]
pub use metal_backend::{MetalDevice, MetalError, MetalStorage};

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

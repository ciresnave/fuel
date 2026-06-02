//! ML framework for Rust
//!
//! ```rust
//! use fuel_core::{Tensor, DType, Device};
//! # use fuel_core::Error;
//! # fn main() -> Result<(), Error>{
//!
//! let a = Tensor::arange(0f32, 6f32, &Device::cpu())?.reshape((2, 3))?;
//! let b = Tensor::arange(0f32, 12f32, &Device::cpu())?.reshape((3, 4))?;
//! let c = a.matmul(&b)?;
//!
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

/// A small-vector type for dimension/stride storage.
/// Avoids heap allocation for tensors with up to 6 dimensions.
pub(crate) type DimVec = smallvec::SmallVec<[usize; 6]>;

#[cfg(feature = "accelerate")]
mod accelerate;
pub mod backend;
pub mod backprop;
pub mod conv;
mod convert;
pub mod cpu_backend;
pub mod cuda_backend;
mod custom_op;
mod device;
pub mod display;
pub mod dyn_backend;
mod dtype;
pub mod dummy_dtype;
pub mod error;
mod indexer;
pub mod kv_cache;
pub mod lazy;
pub mod lazy_based;
pub mod lazy_beit;
pub mod lazy_bert;
pub mod lazy_bigcode;
pub mod lazy_chatglm;
pub mod lazy_clip;
pub mod lazy_kv_cache;
pub mod lazy_llama2c;
pub mod lazy_llava;
pub mod lazy_convnext;
pub mod lazy_deepseek2;
pub mod lazy_dinov2;
pub mod lazy_distilbert;
pub mod lazy_falcon;
pub mod lazy_gemma;
pub mod lazy_gemma3;
pub mod lazy_gemma4_text;
pub mod lazy_gemma4_vision;
pub mod lazy_glm4;
pub mod lazy_granite;
pub mod lazy_granitemoehybrid;
pub mod lazy_helium;
pub mod lazy_mamba;
pub mod lazy_mamba2;
pub mod lazy_marian;
pub mod lazy_mistral;
pub mod lazy_mixformer;
pub mod lazy_mixtral;
pub mod lazy_moondream;
pub mod lazy_mpt;
pub mod lazy_olmo;
pub mod lazy_olmo2;
pub mod lazy_paligemma;
pub mod lazy_persimmon;
pub mod lazy_phi;
pub mod lazy_phi3;
pub mod lazy_pixtral;
pub mod lazy_qwen2;
pub mod lazy_qwen2_moe;
pub mod lazy_qwen3;
pub mod lazy_qwen3_moe;
pub mod lazy_recurrent_gemma;
pub mod lazy_rwkv5;
pub mod lazy_rwkv6;
pub mod lazy_rwkv7;
pub mod lazy_siglip;
pub mod lazy_smollm3;
pub mod lazy_stablelm;
pub mod lazy_starcoder2;
pub mod lazy_t5;
pub mod lazy_vit;
pub mod lazy_yi;
pub mod lazy_sd_text_encoder;
pub mod lazy_sd_unet;
pub mod lazy_sd_vae;
pub mod lazy_whisper;
pub mod lazy_yolov8;
pub mod layout;
// `seq_bucketing` removed in Phase 6d: paged attention via
// `Op::PagedAttn` (and `LazyTensor::paged_attn`) supersedes the
// bucket-and-pad approach. Variable-length decode is now expressed
// directly via per-sequence `context_lens`.
pub mod metal_backend;
pub mod model_progress;
#[cfg(feature = "vulkan")]
pub mod vulkan_backend;
#[cfg(feature = "mkl")]
mod mkl;
// dispatch.rs (Judge cache) moved into judge::cache 2026-05-31 — the
// `fuel_core::dispatch` name was a misnomer for what was just the
// cached output of the Judge. Callers now reach the cache via
// `fuel_core::judge::cached()` / `populate_dispatch_table()` /
// `invalidate()` (re-exported at the judge module's top level).
pub mod factories;
pub mod inference_context;
pub mod pipelined_bridge;
pub mod judge;
pub mod npy;
pub mod probe;
pub mod scheduling;
pub mod topology;
pub mod transfer_cost;
pub mod op;
pub mod pickle;
pub mod quantized;
pub mod nf4;
pub mod safetensors;
pub mod sampling;
pub mod train;
pub mod scalar;
pub mod shape;
mod sort;
mod storage;
pub mod streaming;
mod strided_index;
mod tensor;
mod tensor_cat;
pub mod test_utils;
pub mod utils;
mod variable;

#[cfg(feature = "cudnn")]
pub use cuda_backend::cudnn;

pub use cpu_backend::{CpuStorage, CpuStorageRef, HostBuffer, HostBufferRef};
pub use custom_op::{CustomOp1, CustomOp2, CustomOp3, InplaceOp1, InplaceOp2, InplaceOp3};
pub use device::{Device, DeviceLocation, NdArray};
pub use dtype::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use dummy_dtype::{F4, F6E2M3, F6E3M2, F8E8M0};
pub use error::{Context, Error, Result};
pub use indexer::{IndexOp, TensorIndexer};
pub use layout::Layout;
pub use shape::{Shape, D};
pub use storage::Storage;
pub use streaming::{StreamMask, StreamTensor, StreamingBinOp, StreamingModule, apply_state_mask};
pub use strided_index::{StridedBlocks, StridedIndex};
pub use tensor::{Tensor, TensorId};
pub use variable::Var;

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

/// Defining a module with forward method using a single argument.
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

impl<T: Fn(&Tensor) -> Result<Tensor>> Module for T {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self(xs)
    }
}

impl<M: Module> Module for Option<&M> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}

/// A single forward method using a single single tensor argument and a flag to
/// separate the training and evaluation behaviors.
pub trait ModuleT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor>;
}

impl<M: Module> ModuleT for M {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        self.forward(xs)
    }
}

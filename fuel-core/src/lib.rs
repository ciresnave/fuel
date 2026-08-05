//! ML framework for Rust
//!
//! ```rust
//! use fuel_core::tensor::Tensor;
//! use fuel_core::{DType, Device};
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

#[cfg(feature = "accelerate")]
mod accelerate;
pub mod backend;
pub mod cpu_backend;
pub mod cuda_backend;
mod device;
pub mod dyn_backend;
mod dtype;
pub mod dummy_dtype;
pub mod error;
pub mod lazy;
pub mod lazy_based;
pub mod lazy_beit;
pub mod lazy_bert;
pub mod lazy_bigcode;
pub mod lazy_blip;
pub mod lazy_blip_text;
pub mod lazy_blip_vision;
pub mod lazy_chatglm;
pub mod lazy_chinese_clip;
pub mod lazy_clip;
pub mod lazy_conv3d;
pub mod lazy_colpali;
pub mod lazy_kv_cache;
pub mod lazy_latent_cache;
pub mod lazy_lfm2;
pub mod lazy_llama2c;
pub mod lazy_llama_full;
pub mod lazy_lstm;
pub mod lazy_llava;
pub mod lazy_convmixer;
pub mod lazy_convnext;
pub mod lazy_csm;
pub mod lazy_dac;
pub mod lazy_debertav2;
pub mod lazy_encodec;
pub mod lazy_eva2;
pub mod lazy_deepseek2;
pub mod lazy_depth_anything_v2;
pub mod lazy_dinov2;
pub mod lazy_dinov2reg4;
pub mod lazy_distilbert;
pub mod lazy_efficientnet;
pub mod lazy_efficientvit;
pub mod lazy_falcon;
pub mod lazy_fastvit;
pub mod lazy_flux;
pub mod lazy_gemma;
pub mod lazy_gemma2;
pub mod lazy_gemma3;
pub mod lazy_gemma4_audio;
pub mod lazy_gemma4_mm_embed;
pub mod lazy_gemma4_text;
pub mod lazy_gemma4_vision;
pub mod lazy_glm4;
pub mod lazy_glm4_new;
pub mod lazy_granite;
pub mod lazy_granitemoehybrid;
pub mod lazy_helium;
pub mod lazy_hiera;
pub mod lazy_jina_bert;
pub mod lazy_mamba;
pub mod lazy_mamba2;
pub mod lazy_metavoice;
pub mod lazy_metavoice_speaker_encoder;
pub mod lazy_mimi;
pub mod lazy_mimi_conv;
pub mod lazy_mimi_conv_transpose;
pub mod lazy_mimi_conv_wrappers;
pub mod lazy_mimi_encodec;
pub mod lazy_mimi_quantization;
pub mod lazy_mimi_resampler;
pub mod lazy_mimi_seanet;
pub mod lazy_mimi_transformer;
pub mod lazy_marian;
pub mod lazy_mistral;
pub mod lazy_mixformer;
pub mod lazy_mmdit;
pub mod lazy_modernbert;
pub mod lazy_mixtral;
pub mod lazy_mobileclip;
pub mod lazy_mobilenetv4;
pub mod lazy_mobileone;
pub mod lazy_moondream;
pub mod lazy_mpt;
pub mod lazy_musicgen;
pub mod lazy_nn;
pub mod lazy_nn_conv_transpose;
pub mod lazy_nn_dropout;
pub mod lazy_nn_gru;
pub mod lazy_nn_loss;
pub mod lazy_nn_one_hot;
pub mod lazy_nn_optim;
pub mod lazy_nn_prelu;
pub mod lazy_nn_varbuilder;
pub mod lazy_nn_varmap;
pub mod lazy_nomic_bert;
pub mod lazy_nvembed_v2;
pub mod lazy_olmo;
pub mod lazy_openclip_text;
pub mod lazy_olmo2;
pub mod lazy_paddleocr_vl;
pub mod lazy_paddleocr_vl_text;
pub mod lazy_paddleocr_vl_vision;
pub mod lazy_parler_tts;
pub mod lazy_paligemma;
pub mod lazy_persimmon;
pub mod lazy_phi;
pub mod lazy_phi3;
pub mod lazy_pixtral;
pub mod lazy_quantized_gemma3;
pub mod lazy_quantized_glm4;
pub mod lazy_quantized_lfm2;
pub mod lazy_quantized_llama;
pub mod lazy_quantized_phi3;
pub mod lazy_quantized_qwen2;
pub mod lazy_quantized_qwen3;
pub mod lazy_quantized_qwen3_moe;
pub mod lazy_quantized_smollm3;
pub mod lazy_quantized_t5;
pub mod lazy_quantized_whisper;
pub mod lazy_qwen2;
pub mod lazy_qwen2_moe;
pub mod lazy_qwen3;
pub mod lazy_qwen3_moe;
pub mod lazy_qwen3_vl;
pub mod lazy_qwen3_vl_text;
pub mod lazy_qwen3_vl_vision;
pub mod lazy_recurrent_gemma;
pub mod lazy_repvgg;
pub mod lazy_resnet;
pub mod lazy_segformer;
pub mod lazy_vgg;
pub mod lazy_sam;
pub mod lazy_tiny_vit;
pub mod lazy_rwkv5;
pub mod lazy_rwkv6;
pub mod lazy_rwkv7;
pub mod lazy_rwkv_tokenizer;
pub mod lazy_siglip;
pub mod lazy_smollm3;
pub mod lazy_snac;
pub mod lazy_stablelm;
pub mod lazy_stella_v5;
pub mod lazy_starcoder2;
pub mod lazy_t5;
pub mod lazy_training_augmentations;
pub mod lazy_training_augmentations_extras;
pub mod lazy_trocr;
pub mod lazy_vit;
pub mod lazy_voxtral;
pub mod lazy_xlm_roberta;
pub mod lazy_yi;
pub mod lazy_z_image;
pub mod lazy_sd3_text_encoder;
pub mod lazy_sd3_vae;
pub mod lazy_sd_samplers;
pub mod lazy_sd_samplers_euler;
pub mod lazy_sd_samplers_sd3;
pub mod lazy_sd_samplers_unipc;
pub mod lazy_sd_text_encoder;
pub mod lazy_sd_unet;
pub mod lazy_sd_vae;
pub mod lazy_whisper;
pub mod lazy_whisper_audio;
pub mod lazy_wuerstchen;
pub mod lazy_yolov3;
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
/// The identity a held decode plan is baked against — what makes reusing a
/// [`inference_context::DecodeSession`] safe across models. Read its module
/// docs before adding anything to the key: over-keying is a silent performance
/// regression, under-keying is a silent wrong answer.
pub mod decode_shape;
pub mod inference_context;
pub mod kv_block_pool;
pub mod kv_block_pool_device;
// `multi_session` (the K-way decode scheduler) moved to `fuel-inference` (Q2,
// 2026-07-29): it is consumer-side orchestration, not a Foundation primitive.
// It reaches the model through the `DecodeModel` trait, so it no longer belongs
// in `fuel-core`. See `fuel-inference/src/multi_session.rs`.
pub mod pipelined_bridge;
pub mod planner;
pub mod judge;
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
pub mod quantized;
pub mod nf4;
pub mod safetensors;
pub mod train;
pub mod shape;
mod storage;
mod strided_index;
pub mod test_utils;
pub mod utils;

#[cfg(feature = "cudnn")]
pub use cuda_backend::cudnn;

pub use cpu_backend::{CpuStorage, CpuStorageRef, HostBuffer, HostBufferRef};
pub use device::{Device, DeviceLocation, NdArray};
pub use dtype::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use dummy_dtype::{F4, F6E2M3, F6E3M2, F8E8M0};
pub use error::{Context, Error, Result};
pub use layout::Layout;
pub use shape::{Shape, D};
pub use storage::Storage;
pub use strided_index::{StridedBlocks, StridedIndex};

// Eager `Tensor` is the runtime data type the executor materializes into.
// New user code should use [`lazy::LazyTensor`] — the graph builder — and
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

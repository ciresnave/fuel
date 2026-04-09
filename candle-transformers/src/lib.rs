//! # candle-transformers
//!
//! **Layer**: Models — sits above `candle-nn` and `candle-core`. Provides published
//! model architectures. The dependency arrow goes downward only: nothing in
//! `candle-core` or `candle-nn` depends on this crate.
//!
//! **Stability**: `evolving` — new models are regularly added; existing model public
//! APIs may change as common patterns are extracted to `candle-nn`.
//!
//! ## What this crate is for
//!
//! `candle-transformers` is large collection of production-ready model implementations
//! built from `candle-nn` primitives:
//!
//! - **LLMs** (LLaMA, Mistral, Mixtral, Falcon, Phi, Gemma, Qwen, DeepSeek, …)
//! - **Vision** (ViT, DINOv2, EfficientNet, ResNet, CLIP, SigLIP, …)
//! - **Audio** (Whisper, EnCodec, Mimi, DAC, Parler TTS, …)
//! - **Diffusion** (Stable Diffusion, Flux, Wuerstchen, …)
//! - **Multimodal** (LLaVA, Moondream, PaliGemma, Pixtral, …)
//! - **Encoders** (BERT, T5, Nomic BERT, …)
//!
//! Each model exposes:
//! - A `Config` struct loaded from the model's `config.json`.
//! - A forward-pass struct constructed from a `VarBuilder`.
//! - Quantized variants (`quantized_*.rs`) for GGUF-format weights.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use candle::{Device, DType};
//! use candle_nn::VarBuilder;
//! // Load weights from a safetensors file, then run a forward pass.
//! // (See candle-examples/ for complete runnable examples per model.)
//! ```
//!
//! ## What is explicitly NOT here
//!
//! - **No serving infrastructure.** Batching schedulers, request queues, and stream
//!   management belong in `candle-inference`.
//! - **No decode loops or sampling.** Token generation, beam search, and
//!   `LogitsProcessor` use belong in `candle-inference`.
//! - **No training policy.** LR scheduling, gradient clipping, and checkpoint
//!   management belong in `candle-training`.
//! - **No dataset utilities.** Use `candle-datasets`.
//!
//! Model files contain architecture definitions and forward passes only.
//! Any runtime glue that is inference-specific will migrate to `candle-inference`
//! as that crate matures (see ROADMAP Phase 2 and Phase 3).
//!
//! ## Ecosystem crates
//!
//! - [`candle-core`](https://docs.rs/candle-core): tensor primitives.
//! - [`candle-nn`](https://docs.rs/candle-nn): layers, optimizers, VarBuilder.
//! - [`candle-datasets`](https://docs.rs/candle-datasets): training datasets.
//! - [`candle-onnx`](https://docs.rs/candle-onnx): ONNX import.

pub mod fused_moe;
pub mod generation;
pub mod models;
pub mod object_detection;
pub mod pipelines;
pub mod quantized_nn;
pub mod quantized_var_builder;
pub mod utils;

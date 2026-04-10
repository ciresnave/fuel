//! # fuel-transformers
//!
//! **Layer**: Models — sits above `fuel-nn` and `fuel-core`. Provides published
//! model architectures. The dependency arrow goes downward only: nothing in
//! `fuel-core` or `fuel-nn` depends on this crate.
//!
//! **Stability**: `evolving` — new models are regularly added; existing model public
//! APIs may change as common patterns are extracted to `fuel-nn`.
//!
//! ## What this crate is for
//!
//! `fuel-transformers` is large collection of production-ready model implementations
//! built from `fuel-nn` primitives:
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
//! use fuel::{Device, DType};
//! use fuel_nn::VarBuilder;
//! // Load weights from a safetensors file, then run a forward pass.
//! // (See fuel-examples/ for complete runnable examples per model.)
//! ```
//!
//! ## What is explicitly NOT here
//!
//! - **No serving infrastructure.** Batching schedulers, request queues, and stream
//!   management belong in `fuel-inference`.
//! - **No decode loops or sampling.** Token generation, beam search, and
//!   `LogitsProcessor` use belong in `fuel-inference`.
//! - **No training policy.** LR scheduling, gradient clipping, and checkpoint
//!   management belong in `fuel-training`.
//! - **No dataset utilities.** Use `fuel-datasets`.
//!
//! Model files contain architecture definitions and forward passes only.
//! Any runtime glue that is inference-specific will migrate to `fuel-inference`
//! as that crate matures (see ROADMAP Phase 2 and Phase 3).
//!
//! ## Ecosystem crates
//!
//! - [`fuel-core`](https://docs.rs/fuel-core): tensor primitives.
//! - [`fuel-nn`](https://docs.rs/fuel-nn): layers, optimizers, VarBuilder.
//! - [`fuel-datasets`](https://docs.rs/fuel-datasets): training datasets.
//! - [`fuel-onnx`](https://docs.rs/fuel-onnx): ONNX import.

pub mod fused_moe;
pub mod generation;
pub mod models;
pub mod object_detection;
pub mod pipelines;
pub mod quantized_nn;
pub mod quantized_var_builder;
pub mod utils;

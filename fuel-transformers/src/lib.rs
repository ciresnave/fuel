// SPDX-License-Identifier: MIT OR Apache-2.0
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
//! use fuel_nn::varbuilder::VarBuilder;
//! // Build a `VarBuilder` over safetensors weights, then run a model's
//! // forward pass. (See fuel-examples/ for complete runnable examples per model.)
//! # let _entry = std::marker::PhantomData::<VarBuilder>;
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

// GAP-229, extended to this crate by Stage 2's 146-file move (see
// docs/restructure-migration-design.md): moving the model zoo out from under
// fuel-core's crate-root `#![allow(clippy::identity_op)]` reds this lint on
// byte-identical code — the deliberate TRIPWIRE firing in a third crate, so it is
// RE-MEASURED here, not transcribed. 15 sites, 15 intentional (12 DOC-SHAPE + 3
// DOC-INDEX), 0 defects:
//   * DOC-SHAPE (12): an explicit unit/batch dim in a shape product (`c * 1 * k * k`,
//     `1 * 77 * dim`) mirroring `Shape::from_dims(&[..])`.
//   * DOC-INDEX (3): an explicit `1 *`/`+ 0` NAMES an index (`reg[1*n + i]`,
//     `1*table_total + ..`); two are the load-bearing `1 *` sibling of a `0 *` partner
//     that carries its own `#[allow(clippy::erasing_op)]`, so the sibling must survive.
// RESIDUAL, re-measured for THIS crate: the allow ALSO hides FLOAT identity ops
// (`x + 0.0` normalizes -0.0 -> +0.0). Float population among the 15 is ZERO — all are
// usize shape/index arithmetic. A float identity op landing later is silently admitted;
// re-measure at the next rust-toolchain pin bump (docs/gaps.md GAP-229).
#![allow(clippy::identity_op)]

pub mod generation;
pub mod models;
pub mod object_detection;
pub mod pipelines;
pub mod utils;

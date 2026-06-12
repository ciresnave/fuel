//! # fuel-inference
//!
//! **Layer**: Inference  |  **Stability**: experimental
//!
//! Inference orchestration for the Fuel ML framework. This crate provides
//! the building blocks for running autoregressive text generation and
//! transformer inference pipelines on top of `fuel-core`, `fuel-nn`, and
//! `fuel-transformers`.
//!
//! ## What is here
//!
//! - **KV cache** — [`kv_cache`] re-exports and unifies all cache variants
//!   (`Cache`, `KvCache`, `RotatingKvCache`, `ConcatKvCache`, `ScatteredKvCache`)
//!   currently living in `fuel-core`.
//! - **Sampling** — [`sampling`] exposes `gumbel_softmax` and related utilities
//!   currently living in `fuel-core`.
//! - **Logits processing** — [`generation`] holds `LogitsProcessor`, `Sampling`,
//!   and all decode-time logit strategies currently living in `fuel-transformers`.
//! - **Eviction policies** — [`eviction`] provides composable KV cache eviction
//!   strategies (LRU, H2O Heavy-Hitter Oracle, weighted voting aggregation).
//! - **Prefix caching** — [`prefix_cache`] provides hash-based KV state reuse
//!   for shared prompt prefixes (system prompts, few-shot examples).
//! - **StreamingLLM** — [`streaming`] implements the sink-token + recent-window
//!   attention strategy for stable generation beyond the training context window.
//! - **Speculative decoding** — [`speculative`] implements the draft-then-verify
//!   parallel token generation algorithm for reduced decode latency.
//! - **Chunked prefill** — [`chunked_prefill`] splits long prompts into bounded
//!   chunks to reduce TTFT and allow decode interleaving.
//! - **Segmented eviction** — [`segmented_eviction`] provides span-level KV cache
//!   management where logical segments are evicted as complete units.
//! - **KV compression** — [`kv_compress`] provides three orthogonal compression
//!   strategies (KIVI quantization, R-KV importance pruning, low-rank approximation).
//! - **Scheduler** — [`scheduler`] is a memory-aware inference scheduler with
//!   priority queuing and eviction-pressure admission control.
//! - **MoE routing** — [`moe_routing`] provides capacity-aware top-K token routing
//!   for Mixture-of-Experts models.
//! - **Tiered storage** — [`tiered_storage`] tracks KV cache segments across
//!   GPU → CPU → Disk tiers with budget-aware demotion/promotion and position
//!   ID preservation for RoPE re-injection.
//! - **Context compression** — [`context_compress`] provides token-budget-aware
//!   turn selection and compression for conversations exceeding the context window.
//! - **Tool calls** — [`tool_call`] provides structured tool call parsing,
//!   validation, dispatch, and result injection for function-calling models.
//! - **Pipelines** — [`pipelines`] is placeholder for future session and
//!   batching abstractions.
//!
//! ## What is NOT here
//!
//! - Model definitions (stay in `fuel-transformers`)
//! - Training loop or gradient utilities (use `fuel-training`)
//! - Tokenisation (use `tokenizers` crate directly)
//! - Data loading (use `fuel-datasets`)
//!
//! ## Quick start
//!
//! ```no_run
//! use fuel_inference::generation::{LogitsProcessor, Sampling};
//! use fuel::{Device, Tensor};
//!
//! # fn main() -> fuel::Result<()> {
//! let device = Device::Cpu;
//! let mut lp = LogitsProcessor::new(42, Some(0.7), None);
//!
//! // logits: shape [vocab_size]
//! let logits = Tensor::zeros(32_000usize, fuel::DType::F32, &device)?;
//! // let next_token = lp.sample(&logits)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Layer placement
//!
//! ```text
//! fuel-inference   ← you are here (inference orchestration)
//! fuel-transformers (model definitions)
//! fuel-nn          (layers, optimisers, VarBuilder)
//! fuel-core        (tensors, devices, autograd)
//! ```
//!
//! Nothing in `fuel-core`, `fuel-nn`, or `fuel-transformers` depends on
//! this crate. It is a leaf that aggregates; it does not define.

// ── Re-export inference building blocks ───────────────────────────────────
//
// `kv_cache` and `sampling` now live in `fuel-core` (no fuel-nn dep
// required here). `generation` lives in `fuel-transformers` (moving it here
// would require fuel-transformers to depend on fuel-inference, violating
// the leaf-crate principle). `fuel-inference` aggregates; it does not define.

/// Token-sampling and logit-processing strategies for autoregressive decode.
///
/// Re-exported from `fuel_transformers::generation`.
pub mod generation {
    pub use fuel_transformers::generation::*;
}

/// KV cache implementations for efficient transformer inference.
///
/// Re-exported from `fuel-core`. The eager `fuel::kv_cache` module
/// retired with the eager-Tensor program (Phase β, `1ab1d0c9`); the
/// pipelined-era equivalents are `fuel::lazy_kv_cache::LazyKvCache`
/// (graph-level KV state) and `fuel::inference_context::KvCache`
/// (persistent per-context storage on the production executor).
pub mod kv_cache {
    pub use fuel::inference_context::KvCache;
    pub use fuel::lazy_kv_cache::*;
}

/// Gumbel-softmax and other sampling primitives.
///
/// Re-exported from `fuel_core::sampling`.
pub mod sampling {
    pub use fuel::sampling::*;
}

// ── Native inference modules ──────────────────────────────────────────────

/// Composable KV cache eviction policies (LRU, H2O, weighted voting).
pub mod eviction;

/// Hash-based prefix caching for KV state reuse across requests.
pub mod prefix_cache;

/// StreamingLLM: sink-token + recent-window KV cache management for
/// stable generation beyond the training context window.
pub mod streaming;

/// Speculative decoding: draft-then-verify parallel token generation.
pub mod speculative;

/// Chunked prefill: split long prompts into bounded-size chunks to
/// reduce time-to-first-token and allow decode interleaving.
pub mod chunked_prefill;

/// Segmented eviction: span-level KV cache management where logical
/// segments (conversation turns, document chunks) are tracked and
/// evicted as complete units.
pub mod segmented_eviction;

/// KV cache compression: KIVI quantization, R-KV importance pruning,
/// and low-rank approximation.
pub mod kv_compress;

/// Memory-aware inference scheduler with priority queuing and
/// eviction-pressure admission control.
pub mod scheduler;

/// Mixture-of-Experts capacity-aware top-K token routing.
pub mod moe_routing;

/// Tiered KV cache storage: GPU (VRAM) → CPU (RAM) → Disk.
/// Segments retain position IDs for correct RoPE re-injection on promotion.
pub mod tiered_storage;

/// Context compression: token-budget-aware turn selection and compression
/// for conversations exceeding the model's context window.
pub mod context_compress;

/// Tool call infrastructure: structured parsing, validation, dispatch,
/// and result injection for function-calling models.
pub mod tool_call;

/// Placeholder for future batching, streaming-decode, and session abstractions.
pub mod pipelines {}

/// Phase 6d Track 4: bridge from fuel-inference's runtime
/// orchestration state into the lazy-graph planner. Provides
/// `SchedulerRule` impls that consult inference-side state
/// (memory pressure, MoE routing decisions, etc.) to bias the
/// planner's placement decisions.
pub mod scheduler_bridge;

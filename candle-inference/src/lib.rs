//! # candle-inference
//!
//! **Layer**: Inference  |  **Stability**: experimental
//!
//! Inference orchestration for the Candle ML framework. This crate provides
//! the building blocks for running autoregressive text generation and
//! transformer inference pipelines on top of `candle-core`, `candle-nn`, and
//! `candle-transformers`.
//!
//! ## What is here
//!
//! - **KV cache** — [`kv_cache`] re-exports and unifies all cache variants
//!   (`Cache`, `KvCache`, `RotatingKvCache`, `ConcatKvCache`, `ScatteredKvCache`)
//!   currently living in `candle-core`.
//! - **Sampling** — [`sampling`] exposes `gumbel_softmax` and related utilities
//!   currently living in `candle-core`.
//! - **Logits processing** — [`generation`] holds `LogitsProcessor`, `Sampling`,
//!   and all decode-time logit strategies currently living in `candle-transformers`.
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
//! - Model definitions (stay in `candle-transformers`)
//! - Training loop or gradient utilities (use `candle-training`)
//! - Tokenisation (use `tokenizers` crate directly)
//! - Data loading (use `candle-datasets`)
//!
//! ## Quick start
//!
//! ```no_run
//! use candle_inference::generation::{LogitsProcessor, Sampling};
//! use candle::{Device, Tensor};
//!
//! # fn main() -> candle::Result<()> {
//! let device = Device::Cpu;
//! let mut lp = LogitsProcessor::new(42, Some(0.7), None);
//!
//! // logits: shape [vocab_size]
//! let logits = Tensor::zeros(32_000usize, candle::DType::F32, &device)?;
//! // let next_token = lp.sample(&logits)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Layer placement
//!
//! ```text
//! candle-inference   ← you are here (inference orchestration)
//! candle-transformers (model definitions)
//! candle-nn          (layers, optimisers, VarBuilder)
//! candle-core        (tensors, devices, autograd)
//! ```
//!
//! Nothing in `candle-core`, `candle-nn`, or `candle-transformers` depends on
//! this crate. It is a leaf that aggregates; it does not define.

// ── Re-export inference building blocks ───────────────────────────────────
//
// `kv_cache` and `sampling` now live in `candle-core` (no candle-nn dep
// required here). `generation` lives in `candle-transformers` (moving it here
// would require candle-transformers to depend on candle-inference, violating
// the leaf-crate principle). `candle-inference` aggregates; it does not define.

/// Token-sampling and logit-processing strategies for autoregressive decode.
///
/// Re-exported from `candle_transformers::generation`.
pub mod generation {
    pub use candle_transformers::generation::*;
}

/// KV cache implementations for efficient transformer inference.
///
/// Re-exported from `candle_core::kv_cache`. The canonical implementation
/// lives in `candle-core` so both `candle-nn` and `candle-inference` can
/// expose it without circular dependencies.
pub mod kv_cache {
    pub use candle::kv_cache::*;
}

/// Gumbel-softmax and other sampling primitives.
///
/// Re-exported from `candle_core::sampling`.
pub mod sampling {
    pub use candle::sampling::*;
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

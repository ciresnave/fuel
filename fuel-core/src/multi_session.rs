//! Multi-session serving — Increment 1: the host-side multi-session decode
//! substrate.
//!
//! Runs **K independent decode sessions concurrently on one `LlamaModel`,
//! correctly** — each session generating its own token stream from its own
//! prompt, reusing the existing single-session persistent decode machinery
//! ([`crate::inference_context`] + [`crate::lazy::LlamaModel`]). It adds **no
//! IR op** and **no kernel** — this is pure host orchestration.
//!
//! ## Components
//!
//! - [`SessionState`] (C1) — a faithful bundle of the four per-generation
//!   loop locals that already exist in
//!   [`crate::lazy::LlamaModel::generate_streaming_with_kv_context`]: one
//!   [`crate::inference_context::KvCache`], one
//!   [`crate::inference_context::InferenceContext`], the plan-once
//!   [`crate::inference_context::DecodeSession`] (lazily built on the first
//!   decode token), and the sampler/RNG/token state. Owns **nothing shared**
//!   — independent `KvCache` allocations + an independent `rng_state` are what
//!   make cross-session contamination structurally impossible.
//! - [`SessionScheduler`] (C2) — the K-way serial driver. Advances K sessions
//!   through prefill → decode, samples each with its own RNG, retires the
//!   finished ones. The serial arm is the byte-exact correctness oracle.
//! - [`BatchedDecode`] (C3) — the live batched-decode arm: a Fuel-internal
//!   shared `[K, n_kv_heads, capacity, head_dim]` batch-slot KV buffer +
//!   `flash_decoding` batch wiring, lockstep-only (a single shared `k_len`, so
//!   sessions batch only at equal `cached_len`), byte-equal to the serial arm.
//!
//! Llama-first but trait-shaped ([`ModelDims`]): `PhiModel`'s identical
//! four-local quartet is a later drop-in.

use fuel_ir::{DType, Error};

use crate::Device;
use crate::inference_context::{DecodeSession, InferenceContext, KvCache};
use crate::lazy::SamplingStrategy;

/// Stable identity for one session within a [`SessionScheduler`]. Minted
/// monotonically by `add_session`; used to correlate a session's output with
/// its input across scheduling.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SessionId(pub u64);

/// Lifecycle phase of a [`SessionState`].
///
/// - `Prefill`: the prompt has not yet been run through the model. The next
///   `step` runs one full-prompt forward and samples the first token.
/// - `Decode`: prefill is done; each `step` advances one decode token.
/// - `Finished`: eos was sampled, the budget is exhausted, or the session
///   errored. It is never advanced again.
#[derive(Clone, PartialEq, Debug)]
pub enum SessionPhase {
    Prefill,
    Decode,
    Finished,
}

/// Model geometry a session needs to size its [`KvCache`] (Llama-first;
/// filled from `LlamaConfig` by the scheduler). The trait-shaped seam that
/// lets a later `DecodeModel` (Phi/…) drop in — the scheduler only needs
/// these three numbers to allocate a session's KV.
#[derive(Clone, Copy, Debug)]
pub struct ModelDims {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

/// One decode session's mutable state — a faithful bundle of the four
/// per-generation loop locals from
/// [`crate::lazy::LlamaModel::generate_streaming_with_kv_context`]
/// (`KvCache` + `InferenceContext` + `Option<DecodeSession>` +
/// sampler/RNG/token state) plus scheduling bookkeeping. Owns **nothing
/// shared**: the independent `KvCache` allocations and the independent
/// `rng_state` are what make cross-session contamination structurally
/// impossible (T1).
pub struct SessionState {
    pub(crate) cache: KvCache,
    pub(crate) ctx: InferenceContext,
    pub(crate) session: Option<DecodeSession>,
    pub(crate) tokens: Vec<u32>,
    pub(crate) rng_state: u64,
    pub(crate) strategy: SamplingStrategy,
    pub(crate) eos_id: Option<u32>,
    /// `max_new_tokens` budget left — decremented once per sampled token.
    pub(crate) remaining: usize,
    pub(crate) phase: SessionPhase,
    /// Logits produced by the last forward, consumed by `sample_and_append`.
    pub(crate) last_logits: Option<Vec<f32>>,
    pub(crate) id: SessionId,
    /// Just the GENERATED tail (excludes the prompt) — for reporting.
    pub(crate) new_tokens: Vec<u32>,
}

impl SessionState {
    /// Construct a session seeded in the `Prefill` phase. Mirrors the
    /// loop-local setup at the top of `generate_streaming_with_kv_context`:
    /// validates a non-empty prompt and a positive budget, seeds the RNG from
    /// a `Temperature` seed (else `0`), allocates the pre-sized `KvCache`
    /// (propagating an OOM `Err`), and creates the per-session
    /// `InferenceContext`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: SessionId,
        dims: ModelDims,
        prompt: &[u32],
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        max_new: usize,
        device: &Device,
        dtype: DType,
    ) -> crate::Result<Self> {
        if prompt.is_empty() {
            return Err(Error::Msg("SessionState::new: prompt is empty".to_string()).bt());
        }
        if max_new == 0 {
            return Err(
                Error::Msg("SessionState::new: max_new must be > 0".to_string()).bt(),
            );
        }
        // Per-session RNG seed — the contamination firewall. A Temperature
        // strategy seeds from its `seed`; Greedy is deterministic (0).
        let rng_state: u64 = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };
        let max_seq_len = prompt.len() + max_new;
        // Propagates OOM (or unwired-device) `Err` in isolation — spec #6
        // fail-on-OOM.
        let cache = KvCache::with_capacity(
            dims.n_layers,
            dims.n_kv_heads,
            dims.head_dim,
            max_seq_len,
            dtype,
            device,
        )?;
        let ctx = InferenceContext::new(device.clone());
        Ok(Self {
            cache,
            ctx,
            session: None,
            tokens: prompt.to_vec(),
            rng_state,
            strategy,
            eos_id,
            remaining: max_new,
            phase: SessionPhase::Prefill,
            last_logits: None,
            id,
            new_tokens: Vec::new(),
        })
    }

    /// Whether this session can still advance (not `Finished`).
    pub fn is_ready(&self) -> bool {
        self.phase != SessionPhase::Finished
    }

    /// This session's stable id.
    pub fn id(&self) -> SessionId {
        self.id
    }

    /// The full running token sequence (prompt + generated).
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::{LlamaConfig, LlamaModel, LlamaWeights, LayerWeights, WeightStorage, SamplingStrategy};
    // NOTE: `fuel_ir::Device` does not exist — the device type is `crate::Device`
    // (fuel_core::Device), which is what `KvCache::with_capacity` takes. `DType`
    // is `fuel_ir::DType`. This mirrors the `use` lines at the top of
    // inference_context.rs (`use fuel_ir::{DType, ..}; use crate::Device;`).
    use crate::Device;
    use fuel_ir::DType;
    use std::sync::Arc;

    fn tiny_cfg() -> LlamaConfig {
        LlamaConfig { vocab_size: 16, dim: 8, n_layers: 2, n_heads: 2,
            n_kv_heads: 2, head_dim: 4, ffn_dim: 16, norm_eps: 1e-5, rope_base: 10000.0 }
    }
    // Mirror lazy.rs generate_tests::make_tiny_weights_seeded.
    fn tiny_weights(cfg: &LlamaConfig, seed: u32) -> LlamaWeights {
        let mut s = seed;
        let mut next = || { s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1 };
        let mut vec_of = |n: usize| -> Arc<[f32]> { Arc::from((0..n).map(|_| next()).collect::<Vec<_>>()) };
        let kv = cfg.n_kv_heads * cfg.head_dim;
        LlamaWeights {
            token_embedding: vec_of(cfg.vocab_size * cfg.dim),
            layers: (0..cfg.n_layers).map(|_| LayerWeights {
                attn_q: vec_of(cfg.dim*cfg.dim).into(), attn_q_bias: None,
                attn_k: vec_of(cfg.dim*kv).into(), attn_k_bias: None,
                attn_v: vec_of(cfg.dim*kv).into(), attn_v_bias: None,
                attn_o: vec_of(cfg.dim*cfg.dim).into(),
                ffn_gate: vec_of(cfg.dim*cfg.ffn_dim).into(),
                ffn_up: vec_of(cfg.dim*cfg.ffn_dim).into(),
                ffn_down: vec_of(cfg.ffn_dim*cfg.dim).into(),
                attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                ffn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            }).collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output: vec_of(cfg.dim*cfg.vocab_size).into(),
        }
    }
    fn tiny_model(seed: u32) -> LlamaModel {
        let cfg = tiny_cfg();
        LlamaModel { config: cfg.clone(), weights: tiny_weights(&cfg, seed) }
    }
    fn dims(cfg: &LlamaConfig) -> ModelDims {
        ModelDims { n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads, head_dim: cfg.head_dim }
    }

    #[test]
    fn session_new_seeds_prefill_state() {
        let cfg = tiny_cfg();
        let s = SessionState::new(SessionId(0), dims(&cfg), &[1,2,3],
            SamplingStrategy::Greedy, None, 5, &Device::cpu(), DType::F32).unwrap();
        assert_eq!(s.tokens(), &[1,2,3]);
        assert_eq!(s.phase, SessionPhase::Prefill);
        assert!(s.is_ready());
    }

    #[test]
    fn session_new_rejects_empty_prompt_and_zero_budget() {
        let cfg = tiny_cfg();
        assert!(SessionState::new(SessionId(0), dims(&cfg), &[],
            SamplingStrategy::Greedy, None, 5, &Device::cpu(), DType::F32).is_err());
        assert!(SessionState::new(SessionId(0), dims(&cfg), &[1,2],
            SamplingStrategy::Greedy, None, 0, &Device::cpu(), DType::F32).is_err());
    }
}

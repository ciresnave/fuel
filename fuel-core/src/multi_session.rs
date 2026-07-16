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
use crate::lazy::{sample_logits, LlamaModel, SamplingStrategy};

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

    /// Consume `last_logits` with THIS session's own `rng_state`, append the
    /// sampled token, decrement the budget, and transition to `Finished` on
    /// eos / budget exhaustion. Returns the sampled token, or `None` if there
    /// was nothing to sample (no `last_logits`, or already `Finished`).
    ///
    /// **Per-session RNG is the contamination firewall** — this reads/advances
    /// ONLY `self.rng_state`, never a shared or global state, so two sessions
    /// sampling the same logits with different seeds diverge and one with the
    /// same seed as a standalone run matches it (T3).
    pub fn sample_and_append(&mut self) -> crate::Result<Option<u32>> {
        if self.phase == SessionPhase::Finished {
            return Ok(None);
        }
        let logits = match self.last_logits.take() {
            Some(l) => l,
            None => return Ok(None),
        };
        let next = sample_logits(&logits, self.strategy, &mut self.rng_state);
        self.tokens.push(next);
        self.new_tokens.push(next);
        self.remaining = self.remaining.saturating_sub(1);
        if self.eos_id == Some(next) {
            self.phase = SessionPhase::Finished;
        } else if self.remaining == 0 {
            self.phase = SessionPhase::Finished;
        } else {
            self.phase = SessionPhase::Decode;
        }
        Ok(Some(next))
    }
}

// ===========================================================================
// C2 — SessionScheduler: the K-way decode driver
// ===========================================================================

/// How the scheduler advances the decode-ready set each `step`.
///
/// - `RoundRobin`: advance every ready session serially (the correctness
///   oracle — always available, always byte-exact).
/// - `Batched { max_batch }`: try the live batched arm ([`BatchedDecode`]) on
///   up to `max_batch` uniform sessions, falling back to serial for any
///   session the uniformity gate rejects. Opt-in fast path; provably equal to
///   `RoundRobin`.
#[derive(Clone, Copy, Debug)]
pub enum SchedulePolicy {
    RoundRobin,
    Batched { max_batch: usize },
}

/// What one `step` did — which sessions produced a token, which finished, and
/// which finished-with-error (isolated, never propagated out of `step`).
#[derive(Clone, Debug, Default)]
pub struct StepReport {
    /// Sessions that produced a token this step.
    pub advanced: Vec<SessionId>,
    /// Sessions that transitioned to `Finished` this step (eos, budget, or
    /// error).
    pub finished: Vec<SessionId>,
    /// Sessions that finished with an error this step (also present in
    /// `finished`).
    pub errored: Vec<(SessionId, String)>,
    /// Set true only when the live C3 batched arm actually advanced sessions.
    pub used_batched_arm: bool,
}

/// The K-way decode driver. Owns a `Vec<SessionState>` + a read-only
/// `&LlamaModel` (shared weights) + the device/dtype. Decides which sessions
/// advance together and how (serial in Increment 1's C2; batched once C3 is
/// wired). Owns no tensor state of its own.
pub struct SessionScheduler<'m> {
    model: &'m LlamaModel,
    device: Device,
    dtype: DType,
    sessions: Vec<SessionState>,
    policy: SchedulePolicy,
    next_id: u64,
}

impl<'m> SessionScheduler<'m> {
    /// Create an empty scheduler over a shared read-only model.
    pub fn new(
        model: &'m LlamaModel,
        device: Device,
        dtype: DType,
        policy: SchedulePolicy,
    ) -> Self {
        Self {
            model,
            device,
            dtype,
            sessions: Vec::new(),
            policy,
            next_id: 0,
        }
    }

    /// Add a session from a prompt. Mints a fresh [`SessionId`], builds the
    /// session's private `KvCache` + `InferenceContext` (all geometry is
    /// uniform by construction — every session shares the one `&LlamaModel`),
    /// and returns the id. Geometry/budget is validated at construction time
    /// by [`SessionState::new`], never deferred to `step`. On a construction
    /// error (empty prompt, zero budget, OOM) no id is consumed.
    pub fn add_session(
        &mut self,
        prompt: &[u32],
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        max_new: usize,
    ) -> crate::Result<SessionId> {
        let id = SessionId(self.next_id);
        let dims = ModelDims {
            n_layers: self.model.config.n_layers,
            n_kv_heads: self.model.config.n_kv_heads,
            head_dim: self.model.config.head_dim,
        };
        let state = SessionState::new(
            id,
            dims,
            prompt,
            strategy,
            eos_id,
            max_new,
            &self.device,
            self.dtype,
        )?;
        self.next_id += 1;
        self.sessions.push(state);
        Ok(id)
    }

    /// Advance one scheduling quantum: (1) run any `Prefill` sessions serially
    /// and sample their first token; (2) collect the `Decode`-ready set;
    /// (3) advance it (serial in C2; batched wiring lands in C3); (4) sample
    /// each. A per-session `Err` is isolated into that session finishing with
    /// a recorded error — never propagated out to kill the batch.
    pub fn step(&mut self) -> crate::Result<StepReport> {
        // Copy the shared model reference out of `self` so the per-session
        // `&mut self.sessions[idx]` borrows below don't conflict with reading
        // `self.model`.
        let model = self.model;
        let mut report = StepReport::default();

        // (1) Prefill pass (serial): forward the FULL prompt, transition to
        // Decode, and sample the first token immediately (mirrors the
        // streaming loop where prefill logits yield the first token).
        for idx in 0..self.sessions.len() {
            if self.sessions[idx].phase != SessionPhase::Prefill {
                continue;
            }
            let prompt = self.sessions[idx].tokens.clone();
            let advance = Self::forward_and_store(model, &mut self.sessions[idx], &prompt);
            if advance.is_ok() {
                self.sessions[idx].phase = SessionPhase::Decode;
            }
            self.finalize_advance(idx, advance, &mut report);
        }

        // (2) Decode ready set (includes sessions just prefilled above).
        let ready: Vec<usize> = (0..self.sessions.len())
            .filter(|&i| self.sessions[i].phase == SessionPhase::Decode)
            .collect();

        // (3) Advance the ready set. C2 ships the serial arm only; C3 (Task 7)
        // wires the Batched policy here in front of the serial fallback.
        match self.policy {
            SchedulePolicy::RoundRobin => {
                for idx in ready {
                    self.serial_advance_one(model, idx, &mut report);
                }
            }
            SchedulePolicy::Batched { .. } => {
                // Task 7 replaces this with try_batched_step + serial
                // fallthrough. Until then a Batched policy is pure serial —
                // provably identical to RoundRobin (T6).
                for idx in ready {
                    self.serial_advance_one(model, idx, &mut report);
                }
            }
        }

        Ok(report)
    }

    /// Advance one decode-ready session serially by exactly one forward on its
    /// last token, then sample. Errors isolate into a recorded per-session
    /// failure.
    fn serial_advance_one(&mut self, model: &LlamaModel, idx: usize, report: &mut StepReport) {
        // Never-panic: an empty token history after a validated prefill is
        // impossible, but the production path must not `unwrap`.
        let last = self.sessions[idx].tokens.last().copied();
        let advance = match last {
            Some(t) => Self::forward_and_store(model, &mut self.sessions[idx], &[t]),
            None => Err(Error::Msg(
                "SessionScheduler: decode advance on empty token history".to_string(),
            )
            .bt()),
        };
        self.finalize_advance(idx, advance, report);
    }

    /// Run one forward and store its logits on the session. Shared by the
    /// prefill (full prompt) and decode (last token) advances.
    fn forward_and_store(
        model: &LlamaModel,
        s: &mut SessionState,
        input: &[u32],
    ) -> crate::Result<()> {
        let logits = model.forward_with_kv_context_persistent(
            input,
            &mut s.cache,
            &mut s.ctx,
            &mut s.session,
        )?;
        s.last_logits = Some(logits);
        Ok(())
    }

    /// Given the result of a single advance, sample the token (per-session
    /// RNG) and record it in the report; on advance error, finish the session
    /// with a recorded error. Never panics, never propagates.
    fn finalize_advance(
        &mut self,
        idx: usize,
        advance: crate::Result<()>,
        report: &mut StepReport,
    ) {
        match advance {
            Ok(()) => match self.sessions[idx].sample_and_append() {
                Ok(Some(_)) => {
                    let id = self.sessions[idx].id;
                    report.advanced.push(id);
                    if self.sessions[idx].phase == SessionPhase::Finished {
                        report.finished.push(id);
                    }
                }
                Ok(None) => {}
                Err(e) => self.record_error(idx, e, report),
            },
            Err(e) => self.record_error(idx, e, report),
        }
    }

    /// Force one session to `Finished`-with-error and record it. Isolation: the
    /// other sessions are untouched.
    fn record_error(&mut self, idx: usize, e: crate::Error, report: &mut StepReport) {
        self.sessions[idx].phase = SessionPhase::Finished;
        let id = self.sessions[idx].id;
        report.errored.push((id, e.to_string()));
        report.finished.push(id);
    }

    /// Loop `step` until every session is `Finished`; return each session's
    /// full token sequence (prompt + generated) in insertion order.
    pub fn run_to_completion(&mut self) -> crate::Result<Vec<(SessionId, Vec<u32>)>> {
        while !self.is_all_finished() {
            self.step()?;
        }
        Ok(self
            .sessions
            .iter()
            .map(|s| (s.id, s.tokens.clone()))
            .collect())
    }

    /// Whether every session has finished (vacuously true when empty).
    pub fn is_all_finished(&self) -> bool {
        self.sessions.iter().all(|s| s.phase == SessionPhase::Finished)
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

    #[test]
    fn sample_and_append_greedy_appends_argmax_and_counts_budget() {
        let cfg = tiny_cfg();
        let mut s = SessionState::new(SessionId(0), dims(&cfg), &[1,2],
            SamplingStrategy::Greedy, None, 2, &Device::cpu(), DType::F32).unwrap();
        // argmax at index 3
        s.last_logits = Some(vec![0.0, 0.1, 0.2, 0.9, 0.3]);
        let t = s.sample_and_append().unwrap();
        assert_eq!(t, Some(3));
        assert_eq!(s.tokens(), &[1,2,3]);
        assert_eq!(s.remaining, 1);
        assert_eq!(s.phase, SessionPhase::Decode);
        // exhaust the budget → Finished
        s.last_logits = Some(vec![0.9, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(s.sample_and_append().unwrap(), Some(0));
        assert_eq!(s.phase, SessionPhase::Finished);
        assert!(!s.is_ready());
    }

    #[test]
    fn sample_and_append_stops_on_eos() {
        let cfg = tiny_cfg();
        let mut s = SessionState::new(SessionId(0), dims(&cfg), &[1],
            SamplingStrategy::Greedy, Some(3), 10, &Device::cpu(), DType::F32).unwrap();
        s.last_logits = Some(vec![0.0,0.0,0.0,0.9,0.0]); // argmax 3 == eos
        assert_eq!(s.sample_and_append().unwrap(), Some(3));
        assert_eq!(s.phase, SessionPhase::Finished);
    }

    #[test]
    fn sample_and_append_noop_without_logits() {
        let cfg = tiny_cfg();
        let mut s = SessionState::new(SessionId(0), dims(&cfg), &[1],
            SamplingStrategy::Greedy, None, 3, &Device::cpu(), DType::F32).unwrap();
        assert_eq!(s.sample_and_append().unwrap(), None);
    }

    #[test]
    fn scheduler_single_session_matches_standalone_generate() {
        let model = tiny_model(9999);
        let prompt = [1u32, 2, 3];
        let max_new = 5;
        let standalone = model.generate_with_kv_context(
            &prompt, max_new, SamplingStrategy::Greedy, None, &Device::cpu(), DType::F32).unwrap();

        let mut sched = SessionScheduler::new(
            &model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin);
        let id = sched.add_session(&prompt, SamplingStrategy::Greedy, None, max_new).unwrap();
        let out = sched.run_to_completion().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, id);
        assert_eq!(out[0].1, standalone);   // byte-identical token stream
    }

    #[test]
    fn t1_no_cross_session_contamination() {
        let model = tiny_model(9999);
        let prompt_a = [1u32, 2, 3];
        let prompt_b = [7u32, 4, 9, 2];
        let max_new = 6;

        // Standalone oracles.
        let solo_a = model.generate_with_kv_context(
            &prompt_a, max_new, SamplingStrategy::Greedy, None, &Device::cpu(), DType::F32).unwrap();
        let solo_b = model.generate_with_kv_context(
            &prompt_b, max_new, SamplingStrategy::Greedy, None, &Device::cpu(), DType::F32).unwrap();

        // K=2 scheduled together.
        let mut sched = SessionScheduler::new(
            &model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin);
        let ida = sched.add_session(&prompt_a, SamplingStrategy::Greedy, None, max_new).unwrap();
        let idb = sched.add_session(&prompt_b, SamplingStrategy::Greedy, None, max_new).unwrap();
        let out = sched.run_to_completion().unwrap();

        let get = |id: SessionId| out.iter().find(|(i,_)| *i==id).map(|(_,t)| t.clone()).unwrap();
        assert_eq!(get(ida), solo_a, "session A contaminated by B");
        assert_eq!(get(idb), solo_b, "session B contaminated by A");
    }

    #[test]
    fn t2_interleave_order_invariance() {
        let model = tiny_model(9999);
        let (pa, pb, max_new) = ([1u32,2,3], [5u32,6], 6);

        // Round-robin (both added, then run together).
        let mut rr = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin);
        let a1 = rr.add_session(&pa, SamplingStrategy::Greedy, None, max_new).unwrap();
        let b1 = rr.add_session(&pb, SamplingStrategy::Greedy, None, max_new).unwrap();
        let out_rr = rr.run_to_completion().unwrap();

        // One-then-the-other: A alone to completion, then B alone.
        let mut s_a = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin);
        s_a.add_session(&pa, SamplingStrategy::Greedy, None, max_new).unwrap();
        let just_a = s_a.run_to_completion().unwrap();
        let mut s_b = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin);
        s_b.add_session(&pb, SamplingStrategy::Greedy, None, max_new).unwrap();
        let just_b = s_b.run_to_completion().unwrap();

        let get = |o: &Vec<(SessionId,Vec<u32>)>, id: SessionId| o.iter().find(|(i,_)| *i==id).unwrap().1.clone();
        assert_eq!(get(&out_rr, a1), just_a[0].1);
        assert_eq!(get(&out_rr, b1), just_b[0].1);
    }

    #[test]
    fn t3_per_session_rng_independence() {
        let model = tiny_model(9999);
        let prompt = [1u32, 2, 3];
        let max_new = 8;

        // Same prompt, DIFFERENT seeds → different streams.
        let mut sched = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin);
        let id1 = sched.add_session(&prompt, SamplingStrategy::Temperature{temp:1.0, seed:1}, None, max_new).unwrap();
        let id2 = sched.add_session(&prompt, SamplingStrategy::Temperature{temp:1.0, seed:2}, None, max_new).unwrap();
        let out = sched.run_to_completion().unwrap();
        let g = |id: SessionId| out.iter().find(|(i,_)| *i==id).unwrap().1.clone();
        assert_ne!(g(id1), g(id2), "different seeds must diverge");

        // Same seed as a standalone Temperature run → identical.
        let solo = model.generate_with_kv_context(
            &prompt, max_new, SamplingStrategy::Temperature{temp:1.0, seed:1}, None,
            &Device::cpu(), DType::F32).unwrap();
        assert_eq!(g(id1), solo, "seed 1 must match its standalone run");
    }
}

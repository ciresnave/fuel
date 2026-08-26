// SPDX-License-Identifier: MIT OR Apache-2.0
//! Multi-session serving — Increment 1: the host-side multi-session decode
//! substrate.
//!
//! Runs **K independent decode sessions concurrently on one `LlamaModel`,
//! correctly** — each session generating its own token stream from its own
//! prompt, reusing the existing single-session persistent decode machinery
//! ([`fuel::inference_context`] + [`fuel::lazy::LlamaModel`]). It adds **no
//! IR op** and **no kernel** — this is pure host orchestration.
//!
//! ## Components
//!
//! - [`SessionState`] (C1) — a faithful bundle of the four per-generation
//!   loop locals that already exist in
//!   [`fuel::lazy::LlamaModel::generate_streaming_with_kv_context`]: one
//!   [`fuel::inference_context::KvCache`], one
//!   [`fuel::inference_context::InferenceContext`], the plan-once
//!   [`fuel::inference_context::DecodeSession`] (lazily built on the first
//!   decode token), and the sampler/RNG/token state. Owns **nothing shared**
//!   — independent `KvCache` allocations + an independent `rng_state` are what
//!   make cross-session contamination structurally impossible.
//! - [`SessionScheduler`] (C2) — the K-way serial driver. Advances K sessions
//!   through prefill → decode, samples each with its own RNG, retires the
//!   finished ones. The serial arm is the byte-exact correctness oracle.
//! - [`BatchedDecode`] (C3) — the live batched-decode arm: a Fuel-internal
//!   shared `[K, n_kv_heads, capacity, head_dim]` batch-slot KV buffer +
//!   `flash_decoding` batch wiring, lockstep-only (a single shared `k_len`, so
//!   sessions batch only at equal `cached_len`). It is a SEPARATE batch=K
//!   plan-once graph (different reduction order than the batch=1 serial arm),
//!   so it is **ε-close** (logits within 1e-4) and **token-identical** to the
//!   serial arm on tested shapes — not bit-exact.
//!
//! Llama-first but trait-shaped ([`ModelDims`]): `PhiModel`'s identical
//! four-local quartet is a later drop-in.

use std::collections::HashMap;

use fuel_ir::{DType, Error};

use fuel::Device;
use fuel::decode_state_spec::LayerStateSpec;
use fuel::inference_context::{
    DecodeSession, InferenceContext, KvCache, PagedDecodePlan, PagedDecodeSession,
};
use fuel::kv_block_pool::{KvBlockPool, KvGeometry, PoolCapacity, PrefixId, SessionHandle};
use fuel::kv_block_pool_device::{DeviceEvicted, DeviceKvPool};
use fuel::lazy::{LlamaModel, SamplingStrategy, sample_logits};

/// The KV memory budget a [`SessionScheduler`] admits sessions against — the
/// C-1 capacity mechanism (from [15-consumer-contract]). `num_blocks` physical
/// blocks of `block_size` tokens each is the pool's ceiling; a session reserves
/// `⌈(prompt + max_new) / block_size⌉` blocks at admission, so the scheduler can
/// answer "will this session fit?" *before* building its KV cache instead of
/// discovering it via a late OOM. The per-token head geometry comes from the
/// model, so only these two knobs are the budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvBudget {
    /// Tokens per block (the `Op::PagedAttn` block size, and the admission
    /// reservation quantum).
    pub block_size: usize,
    /// Total physical blocks — the KV ceiling shared across all sessions.
    pub num_blocks: usize,
}

/// The model capability the [`SessionScheduler`] depends on — the model-agnostic
/// seam. The scheduler is orchestration (mechanism); it needs only the KV-context
/// decode surface, never a concrete model family. `LlamaModel` is the first
/// implementor, but any model providing a persistent-KV decode + the batched-
/// decode logits arm can be served. This is the abstraction that lets the whole
/// module live in a consumer crate (`fuel-inference`) without depending on a
/// specific model definition — sampling (`SamplingStrategy`) is deliberately NOT
/// here: it is consumer policy the scheduler applies per-session, not a model
/// capability.
///
/// Every method mirrors the concrete `LlamaModel` inherent method it fronts, so
/// the impl is a thin forward. The KV/decode types (`KvCache`,
/// `InferenceContext`, `DecodeSession`) are Fuel-core primitives the seam speaks;
/// they are the interchange, not a model dependency.
pub trait DecodeModel {
    /// Transformer layer count — the KV cache geometry the scheduler builds a
    /// session's private cache against (all sessions share one model, so this is
    /// uniform).
    fn n_layers(&self) -> usize;
    /// The decode state each layer keeps, **indexed by layer**. The scheduler
    /// collapses this to a single `(n_kv_heads, head_dim)` cache geometry via
    /// [`LayerStateSpec::collapse_uniform`] at construction — and that collapse
    /// FAILS LOUDLY for a model whose layers are not uniform per-head KV, rather
    /// than fabricating a pair the model never keeps.
    ///
    /// This replaces the former scalar `n_kv_heads()` / `head_dim()`, which
    /// asserted every model's state is uniform per-head KV — a claim MLA
    /// (`DeepSeek2Model`, whose decode state is a `LatentCache` of
    /// `[latent, k_pe]` slots) cannot honor, yet was syntactically able to
    /// return, mis-allocating silently. A uniform per-head-KV model returns the
    /// same [`LayerStateSpec::KeyValue`] for every index and is unaffected. See
    /// GAP-166 and [`fuel::decode_state_spec`].
    fn layer_state_specs(&self) -> Vec<LayerStateSpec>;

    /// One persistent-KV forward: run `tokens` (full prompt on prefill, the last
    /// token on decode), mutating the session's `cache`/`ctx`/`session` in place,
    /// and return the step's logits. The scheduler's serial arm + prefill pass.
    fn forward_with_kv_context_persistent(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<DecodeSession>,
    ) -> fuel::Result<Vec<f32>>;

    /// Does this model implement the batched arm
    /// ([`Self::build_batched_decode_logits`])? Defaults to `false`.
    ///
    /// This is a *predicate*, not an error path, on purpose. The scheduler asks
    /// before it assembles a batch, so a core-only model takes the serial arm
    /// directly instead of building a batch, failing, and falling back — which
    /// would waste the work every step AND make `StepReport::used_batched_arm`
    /// the only way to discover the model can't batch. Override it to `true` in
    /// the same impl that overrides `build_batched_decode_logits`; the two are
    /// consistency-checked by `decode_model_batched_capability_is_self_consistent`.
    fn supports_batched_decode(&self) -> bool {
        false
    }

    /// One batch=K decode step over K sessions' caches — the live batched arm.
    /// Returns one logits row per cache. The scheduler only calls this on a
    /// uniformity-gated ready set; a model may return an error (never a panic)
    /// before mutating any cache (the all-or-nothing contract the batched arm
    /// relies on).
    ///
    /// **Defaulted to a typed decline**, unlike the paged surface below, and the
    /// asymmetry is deliberate rather than an oversight. Batching is selected by
    /// [`SchedulePolicy`], a *runtime* enum handed to `SessionScheduler::new` —
    /// so gating it in the type system would mean making the policy a type
    /// parameter, which churns every construction site to buy a guarantee the
    /// scheduler already provides at runtime: `advance_batched`'s
    /// `NotBatchable`/error path routes unbatched sessions to the serial arm in
    /// isolation, with their KV untouched. Paged decode has no such fallback —
    /// a paged scheduler cannot run a model that has no paged surface at all —
    /// so it is a supertrait ([`PagedDecodeModel`]) and the mismatch is a
    /// compile error.
    fn build_batched_decode_logits(
        &self,
        _caches: &mut [&mut KvCache],
        _last_tokens: &[u32],
        _device: &Device,
        _dtype: DType,
    ) -> fuel::Result<Vec<Vec<f32>>> {
        Err(fuel::Error::Msg(
            "this model implements only the core DecodeModel surface — no \
             batched decode arm (see DecodeModel::supports_batched_decode)"
                .to_string(),
        ))
    }
}

/// Paged-storage decode: the surface [`PagedSessionScheduler`] needs.
///
/// A separate trait rather than more methods on [`DecodeModel`] because paged
/// serving has no serial fallback — a model with no paged surface simply cannot
/// be driven by the paged scheduler, and that is better said at compile time
/// than discovered as a per-session error at run time. `PagedSessionScheduler`
/// is bounded on this trait, so handing it a core-only model does not build.
///
/// Ten of Fuel's twelve model families implement neither this trait nor, today,
/// the core one; the split exists so they can arrive incrementally — contiguous
/// persistent decode first (which is what makes a model *servable*), paged
/// later — instead of needing all eight methods before any of them works.
pub trait PagedDecodeModel: DecodeModel {
    /// One single-token **paged** forward — the paged-storage decode surface
    /// ([`PagedSessionScheduler`]). Feeds `token` (a prompt token during prefill,
    /// a sampled token during decode) into the session's blocks of the shared
    /// `pool`, attends via `Op::PagedAttn`, and returns last-position logits.
    /// Grows the session's block allocation incrementally (one slot per token);
    /// an exhausted pool surfaces as a typed `Err` (the scheduler isolates it
    /// into a per-session finish), never a panic.
    fn forward_paged_step(
        &self,
        token: u32,
        pool: &mut DeviceKvPool,
        session: SessionHandle,
    ) -> fuel::Result<Vec<f32>>;

    /// Batched (`B = K`) sibling of [`forward_paged_step`](Self::forward_paged_step)
    /// — one decode step over K same-position sessions in a single model pass
    /// (`Op::PagedAttn` at B=K), returning one logits row per session in
    /// `sessions` order. The scheduler's batched arm ([`PagedSessionScheduler::
    /// step_batched`]) uses it on uniform-position groups. All sessions must be at
    /// the same position (typed `Err` otherwise).
    fn forward_paged_step_batched(
        &self,
        tokens: &[u32],
        pool: &mut DeviceKvPool,
        sessions: &[SessionHandle],
    ) -> fuel::Result<Vec<Vec<f32>>>;

    /// Plan-once sibling of [`forward_paged_step`](Self::forward_paged_step): the
    /// paged driver holds one [`PagedDecodeSession`] per session and, under
    /// `plan == PagedDecodePlan::PlanOnce`, builds + optimizes the decode graph
    /// ONCE then rebinds it per token — paying the optimizer (Lightbulb-measured at
    /// ~90% of per-token paged cost) once instead of every token — and `PlanOnce`
    /// is the driver default. `Replan` (the explicit opt-out) drops any held
    /// session and re-plans via `forward_paged_step`
    /// — behaviorally identical to the pre-plan-once path, so a session may flip
    /// the flag without leaving stale state. `max_blocks_cap` is the session's
    /// fixed block-table capacity (`⌈(prompt + max_new) / block_size⌉`), which
    /// pins the held graph's `block_table` shape across tokens. Single session
    /// (B = 1); the batched arm stays on `forward_paged_step_batched`.
    #[allow(clippy::too_many_arguments)]
    fn forward_paged_step_persistent(
        &self,
        token: u32,
        pool: &mut DeviceKvPool,
        session: SessionHandle,
        max_blocks_cap: usize,
        plan: PagedDecodePlan,
        decode_session: &mut Option<PagedDecodeSession>,
    ) -> fuel::Result<Vec<f32>>;
}

impl DecodeModel for LlamaModel {
    fn n_layers(&self) -> usize {
        self.config.n_layers
    }
    fn layer_state_specs(&self) -> Vec<LayerStateSpec> {
        vec![
            LayerStateSpec::KeyValue {
                n_kv_heads: self.config.n_kv_heads,
                head_dim: self.config.head_dim,
            };
            self.config.n_layers
        ]
    }
    fn forward_with_kv_context_persistent(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<DecodeSession>,
    ) -> fuel::Result<Vec<f32>> {
        // Explicit inherent-path call: `LlamaModel::` resolves to the inherent
        // method (inherent wins over the same-named trait method), so this is a
        // forward, not a recursion into the trait.
        LlamaModel::forward_with_kv_context_persistent(self, tokens, cache, ctx, session)
    }
    fn supports_batched_decode(&self) -> bool {
        true
    }
    fn build_batched_decode_logits(
        &self,
        caches: &mut [&mut KvCache],
        last_tokens: &[u32],
        device: &Device,
        dtype: DType,
    ) -> fuel::Result<Vec<Vec<f32>>> {
        LlamaModel::build_batched_decode_logits(self, caches, last_tokens, device, dtype)
    }
}

/// `Llama3Model` is a thin wrapper over [`LlamaModel`] that differs only in its
/// RoPE frequencies, so it reaches the same decode path with its own
/// frequencies threaded down — see `Llama3Model::forward_with_kv_context_
/// persistent`. It implements the CORE trait only: the paged surface would need
/// the same override threaded through `forward_paged_step*`, which is a separate
/// tier of work, and [`PagedDecodeModel`] existing as a supertrait is what makes
/// that a compile error here rather than a runtime surprise in the paged
/// scheduler.
impl DecodeModel for fuel::lazy_llama_full::Llama3Model {
    fn n_layers(&self) -> usize {
        self.inner.config.n_layers
    }
    fn layer_state_specs(&self) -> Vec<LayerStateSpec> {
        vec![
            LayerStateSpec::KeyValue {
                n_kv_heads: self.inner.config.n_kv_heads,
                head_dim: self.inner.config.head_dim,
            };
            self.inner.config.n_layers
        ]
    }
    fn forward_with_kv_context_persistent(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<DecodeSession>,
    ) -> fuel::Result<Vec<f32>> {
        fuel::lazy_llama_full::Llama3Model::forward_with_kv_context_persistent(
            self, tokens, cache, ctx, session,
        )
    }
}

/// GGUF/Q4_0-quantized LLaMA — the decode surface GAP-029 was filed for.
///
/// **Every method delegates to [`Self::inner`], the wrapped
/// [`fuel::lazy_llama_full::Llama3Model`], and that choice is the entire
/// content of this impl.** `QuantizedLlama3Model` nests two levels
/// (`QuantizedLlama3Model → Llama3Model → LlamaModel`) and only the *middle*
/// layer knows about LLaMA-3.1 RoPE scaling: `Llama3Model` threads
/// `rope_inv_freq()` into `LlamaModel::forward_with_kv_context_persistent_inv_freq`,
/// while `LlamaModel`'s own entry point uses the **unscaled** RoPE base.
///
/// So delegating one level too deep — to `self.inner().inner` — compiles, runs,
/// and is **silently wrong** on any scaled (LLaMA-3.1) checkpoint: slightly
/// wrong long-context attention, no error. That is not hypothetical here;
/// `QuantizedLlama3Model::forward_hidden` already reaches through to
/// `self.inner.inner` and its own doc comment admits it therefore loses the
/// scaling. This impl must not repeat it, and
/// `quantized_llama_decode_delegates_to_scaling_aware_inner` is the test that
/// fails if it ever does.
///
/// The batched-arm pair (`supports_batched_decode` +
/// `build_batched_decode_logits`) is delegated rather than left to default so
/// the two cannot desync from the inner model: if `Llama3Model` ever gains a
/// batched arm, the quantized wrapper inherits it and its capability predicate
/// in the same change.
impl DecodeModel for fuel::lazy_quantized_llama::QuantizedLlama3Model {
    fn n_layers(&self) -> usize {
        DecodeModel::n_layers(self.inner())
    }
    fn layer_state_specs(&self) -> Vec<LayerStateSpec> {
        DecodeModel::layer_state_specs(self.inner())
    }
    fn forward_with_kv_context_persistent(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<DecodeSession>,
    ) -> fuel::Result<Vec<f32>> {
        DecodeModel::forward_with_kv_context_persistent(self.inner(), tokens, cache, ctx, session)
    }
    fn supports_batched_decode(&self) -> bool {
        DecodeModel::supports_batched_decode(self.inner())
    }
    fn build_batched_decode_logits(
        &self,
        caches: &mut [&mut KvCache],
        last_tokens: &[u32],
        device: &Device,
        dtype: DType,
    ) -> fuel::Result<Vec<Vec<f32>>> {
        DecodeModel::build_batched_decode_logits(self.inner(), caches, last_tokens, device, dtype)
    }
}

impl PagedDecodeModel for LlamaModel {
    fn forward_paged_step(
        &self,
        token: u32,
        pool: &mut DeviceKvPool,
        session: SessionHandle,
    ) -> fuel::Result<Vec<f32>> {
        LlamaModel::forward_paged_step(self, token, pool, session)
    }
    fn forward_paged_step_batched(
        &self,
        tokens: &[u32],
        pool: &mut DeviceKvPool,
        sessions: &[SessionHandle],
    ) -> fuel::Result<Vec<Vec<f32>>> {
        LlamaModel::forward_paged_step_batched(self, tokens, pool, sessions)
    }
    fn forward_paged_step_persistent(
        &self,
        token: u32,
        pool: &mut DeviceKvPool,
        session: SessionHandle,
        max_blocks_cap: usize,
        plan: PagedDecodePlan,
        decode_session: &mut Option<PagedDecodeSession>,
    ) -> fuel::Result<Vec<f32>> {
        LlamaModel::forward_paged_step_persistent(
            self,
            token,
            pool,
            session,
            max_blocks_cap,
            plan,
            decode_session,
        )
    }
}

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

impl ModelDims {
    /// The single cache geometry the scheduler allocates a session's KV against,
    /// extracted from a model's per-layer decode-state description
    /// ([`DecodeModel::layer_state_specs`]) — **failing** when the layers are not
    /// uniform per-head KV.
    ///
    /// This is the one place the old `(n_kv_heads, head_dim)` scalar assumption
    /// survives, and it is a `Result` rather than a silent pair precisely so a
    /// non-uniform / non-KV model (MLA today, a future LFM2) is declined *here*,
    /// at scheduler construction, instead of mis-allocated. `ModelDims` and
    /// [`KvGeometry`] keep their scalar fields untouched — their vLLM
    /// shared-block-table commitment is not widened; they simply receive the
    /// collapsed pair through a fallible boundary. See GAP-166 / GAP-170 and
    /// [`LayerStateSpec::collapse_uniform`].
    pub fn from_model<M: DecodeModel + ?Sized>(model: &M) -> fuel::Result<Self> {
        let (n_kv_heads, head_dim) = LayerStateSpec::collapse_uniform(&model.layer_state_specs())?;
        Ok(Self {
            n_layers: model.n_layers(),
            n_kv_heads,
            head_dim,
        })
    }
}

/// One decode session's mutable state — a faithful bundle of the four
/// per-generation loop locals from
/// [`fuel::lazy::LlamaModel::generate_streaming_with_kv_context`]
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
    ) -> fuel::Result<Self> {
        if prompt.is_empty() {
            return Err(Error::Msg("SessionState::new: prompt is empty".to_string()).bt());
        }
        if max_new == 0 {
            return Err(Error::Msg("SessionState::new: max_new must be > 0".to_string()).bt());
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
    pub fn sample_and_append(&mut self) -> fuel::Result<Option<u32>> {
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
        // Two distinct reasons to stop -- hit EOS, or ran out of budget --
        // but `Finished` is not reason-parameterised, so the distinction was
        // never carried by the value. Merging loses nothing the type holds.
        if self.eos_id == Some(next) || self.remaining == 0 {
            self.phase = SessionPhase::Finished;
        } else {
            self.phase = SessionPhase::Decode;
        }
        Ok(Some(next))
    }
}

// ===========================================================================
// C3 — BatchedDecode: the live batched-decode arm (seam + uniformity gate)
// ===========================================================================

/// Result of a batched decode attempt.
pub enum BatchOutcome {
    /// N logits vectors, one per input session (same order as the input
    /// slice).
    Advanced(Vec<Vec<f32>>),
    /// The ready set was not uniform enough to batch — a NORMAL control value,
    /// not an `Err`. The scheduler serial-steps instead.
    NotBatchable,
}

/// The per-session geometry the uniformity gate compares. All fields must be
/// EQUAL across the batch to share one `flash_decoding` call — crucially
/// `cached_len`, since the kernel takes a single shared `k_len`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BatchDescriptor {
    pub cached_len: usize,
    pub max_seq_len: usize,
    pub n_layers: usize,
    pub cache_dtype: DType,
}

/// Pure gate: are these sessions batchable together THIS step? `false` for
/// fewer than 2 sessions, or if any descriptor differs from the first (in
/// particular a ragged `cached_len` — the single shared `flash_decoding`
/// `k_len` would be wrong for the odd rows). Unit-testable without CUDA (T6).
pub(crate) fn batch_uniform(descs: &[BatchDescriptor]) -> bool {
    if descs.len() < 2 {
        return false;
    }
    descs.iter().all(|d| *d == descs[0])
}

/// The live batched-decode arm. Owns no persistent state — a unit type whose
/// associated `try_batched_step` produces N logits vectors in one batch=K
/// model pass over a shared `[K, n_kv_heads, capacity, head_dim]` KV buffer,
/// or signals `NotBatchable` so the scheduler falls back to serial.
pub(crate) struct BatchedDecode;

impl BatchedDecode {
    /// Attempt one live batched decode step over the given `Decode`-phase
    /// sessions.
    ///
    /// Task 7 ships the SIGNATURE + the uniformity gate + the `NotBatchable`
    /// path only — even a uniform ready set returns `NotBatchable` here so the
    /// scheduler's serial fallback stays the executed path. Task 8 wires the
    /// live `flash_decoding` batch arm into the `Advanced` branch.
    pub(crate) fn try_batched_step<M: DecodeModel>(
        model: &M,
        device: &Device,
        dtype: DType,
        sessions: &mut [&mut SessionState],
    ) -> fuel::Result<BatchOutcome> {
        let descs: Vec<BatchDescriptor> = sessions
            .iter()
            .map(|s| BatchDescriptor {
                cached_len: s.cache.cached_len,
                max_seq_len: s.cache.max_seq_len.unwrap_or(0),
                n_layers: s.cache.n_layers(),
                cache_dtype: s.cache.dtype.unwrap_or(dtype),
            })
            .collect();
        if !batch_uniform(&descs) {
            return Ok(BatchOutcome::NotBatchable);
        }

        // Live batched arm: one batch=K decode step over a shared [K,..] KV
        // buffer — ε-close (logits within 1e-4) and token-identical to K serial
        // steps on tested shapes (a SEPARATE batch=K plan-once graph, so not
        // bit-exact; see build_batched_decode_logits).
        //
        // Each session's last token is the decode input. An empty token
        // history is impossible after a validated prefill, but the production
        // path must not `unwrap`.
        let last_tokens: Vec<u32> = match sessions
            .iter()
            .map(|s| s.tokens.last().copied())
            .collect::<Option<Vec<u32>>>()
        {
            Some(v) => v,
            None => {
                return Err(Error::Msg(
                    "BatchedDecode::try_batched_step: a session has no last token".to_string(),
                )
                .bt());
            }
        };
        // Borrow each session's own KvCache. `build_batched_decode_logits`
        // copies-in, decodes, then copies-out + bumps these caches only after
        // the decode realize succeeds. Session KV is private (T1), so a copy-out
        // fault can only leave the FAULTING batch's own caches partially
        // rewritten — and on any `Err` the scheduler's `advance_batched` Err arm
        // finishes every batch member, so no partial cache is ever decoded again
        // (all-or-nothing; see build_batched_decode_logits step 4's WARNING).
        let mut caches: Vec<&mut fuel::inference_context::KvCache> =
            sessions.iter_mut().map(|s| &mut s.cache).collect();
        let rows = model.build_batched_decode_logits(&mut caches, &last_tokens, device, dtype)?;
        Ok(BatchOutcome::Advanced(rows))
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
pub struct SessionScheduler<'m, M: DecodeModel> {
    model: &'m M,
    device: Device,
    dtype: DType,
    sessions: Vec<SessionState>,
    policy: SchedulePolicy,
    next_id: u64,
    /// C-1 capacity accountant: the KV block pool the scheduler admits sessions
    /// against. Each admitted session reserves the blocks its `max_seq_len`
    /// (prompt + budget) needs; the reservation is a faithful proxy for its
    /// fixed-capacity `KvCache`, so `free_blocks` mirrors real KV headroom.
    /// (Mechanism only — the pool decides *fit*, never *whom* to admit; that is
    /// the caller's policy.) The reservation is released together with the
    /// session's `KvCache` by [`reap_finished`](Self::reap_finished).
    kv_pool: KvBlockPool,
    /// Live sessions' pool reservations, so `reap_finished` frees the exact
    /// blocks a session reserved.
    kv_handles: HashMap<SessionId, SessionHandle>,
}

impl<'m, M: DecodeModel> SessionScheduler<'m, M> {
    /// Create an empty scheduler over a shared read-only model, admitting
    /// sessions against a KV block-pool budget (C-1). The pool's per-token head
    /// geometry is taken from the model (all sessions share it); `budget` sets
    /// the block size + total blocks.
    ///
    /// **Returns `Err` when the model's layers are not uniform per-head KV** —
    /// the [`ModelDims::from_model`] collapse boundary. A model whose decode
    /// state is not a single `(n_kv_heads, head_dim)` (MLA, a future LFM2) is
    /// declined here rather than silently mis-allocated (GAP-166).
    pub fn new(
        model: &'m M,
        device: Device,
        dtype: DType,
        policy: SchedulePolicy,
        budget: KvBudget,
    ) -> fuel::Result<Self> {
        let dims = ModelDims::from_model(model)?;
        let kv_pool = KvBlockPool::new(KvGeometry {
            n_layers: dims.n_layers,
            n_kv_heads: dims.n_kv_heads,
            head_dim: dims.head_dim,
            num_blocks: budget.num_blocks,
            block_size: budget.block_size,
            elem_size: dtype.size_in_bytes(),
        });
        Ok(Self {
            model,
            device,
            dtype,
            sessions: Vec::new(),
            policy,
            next_id: 0,
            kv_pool,
            kv_handles: HashMap::new(),
        })
    }

    // --- C-1 capacity advertisement (the admission primitives) -----------

    /// Physical KV blocks currently free (C-1). A caller sheds/queues load by
    /// comparing this to [`kv_blocks_required`](Self::kv_blocks_required) BEFORE
    /// calling [`add_session`](Self::add_session), rather than discovering the
    /// ceiling via a rejected admit.
    pub fn kv_free_blocks(&self) -> usize {
        self.kv_pool.free_blocks()
    }

    /// The pool's full capacity descriptor (C-1) — free/total blocks + geometry.
    pub fn kv_capacity(&self) -> PoolCapacity {
        self.kv_pool.capacity()
    }

    /// Blocks a fresh session of `prompt_len + max_new` tokens would reserve —
    /// the exact quantity [`add_session`](Self::add_session) checks against
    /// [`kv_free_blocks`](Self::kv_free_blocks). The admission math lives in one
    /// place so a caller's pre-check can never disagree with the reservation.
    pub fn kv_blocks_required(&self, prompt_len: usize, max_new: usize) -> usize {
        self.kv_pool.blocks_required(0, prompt_len + max_new)
    }

    // --- C-4 measured cost -----------------------------------------------

    /// Resident KV bytes across all admitted sessions (C-4) — the scheduler's
    /// budget signal. Reservation-based (each session's full `max_seq_len`),
    /// matching the fixed-capacity caches it stands in for.
    pub fn kv_bytes_resident(&self) -> u64 {
        self.kv_pool.kv_bytes_resident()
    }

    /// Add a session from a prompt. **Capacity-gated (C-1):** reserves the
    /// blocks its `max_seq_len` (prompt + `max_new`) needs from the KV pool
    /// FIRST — if they don't fit, returns `Err` *before* building any KV cache,
    /// so a full scheduler sheds an admission instead of OOMing mid-build. On
    /// success, mints a fresh [`SessionId`], builds the private `KvCache` +
    /// `InferenceContext`, and records the reservation. On any error (capacity,
    /// empty prompt, zero budget, OOM) no id and no blocks are consumed.
    pub fn add_session(
        &mut self,
        prompt: &[u32],
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        max_new: usize,
    ) -> fuel::Result<SessionId> {
        // C-1 gate: does this session's KV reservation fit? Check before any
        // allocation so a rejection is cheap and total (nothing half-built).
        let needed = self.kv_blocks_required(prompt.len(), max_new);
        let free = self.kv_pool.free_blocks();
        if needed > free {
            return Err(Error::Msg(format!(
                "SessionScheduler::add_session: KV capacity — session needs {needed} blocks, \
                 {free} free (shed load, or reap_finished to reclaim). Pre-check with \
                 kv_blocks_required + kv_free_blocks.",
            ))
            .bt());
        }

        let id = SessionId(self.next_id);
        let dims = ModelDims::from_model(self.model)?;
        // Build the cache first (an empty-prompt / zero-budget error must not
        // consume a reservation), THEN reserve — both are pre-gated above so
        // neither the build nor the reserve can fail on capacity here.
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
        let handle = self.kv_pool.open();
        self.kv_pool
            .append(handle, prompt.len() + max_new)
            .map_err(|e| {
                Error::Msg(format!(
                    "SessionScheduler::add_session: reserve failed: {e:?}"
                ))
                .bt()
            })?;
        self.kv_handles.insert(id, handle);
        self.next_id += 1;
        self.sessions.push(state);
        Ok(id)
    }

    /// Reap every `Finished` session: drop it (freeing its `KvCache`) and
    /// release its KV pool reservation, returning each reaped session's
    /// `(id, tokens)`. This is the operation that turns a completed session's KV
    /// back into admission headroom — cache-free and pool-free happen together,
    /// so [`kv_free_blocks`](Self::kv_free_blocks) never drifts from real memory.
    /// A serving loop calls this between `step`s to admit new work as old work
    /// completes. (Non-`Finished` sessions are untouched.)
    pub fn reap_finished(&mut self) -> Vec<(SessionId, Vec<u32>)> {
        let mut reaped = Vec::new();
        let mut kept = Vec::with_capacity(self.sessions.len());
        for s in std::mem::take(&mut self.sessions) {
            if s.phase == SessionPhase::Finished {
                if let Some(h) = self.kv_handles.remove(&s.id) {
                    self.kv_pool.discard(h);
                }
                reaped.push((s.id, s.tokens));
            } else {
                kept.push(s);
            }
        }
        self.sessions = kept;
        reaped
    }

    /// Advance one scheduling quantum: (1) run any `Prefill` sessions serially
    /// and sample their first token; (2) collect the `Decode`-ready set;
    /// (3) advance it (serial in C2; batched wiring lands in C3); (4) sample
    /// each. A per-session `Err` is isolated into that session finishing with
    /// a recorded error — never propagated out to kill the batch.
    pub fn step(&mut self) -> fuel::Result<StepReport> {
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

        // (3) Advance the ready set. The serial arm is always reachable and is
        // the byte-exact oracle; the Batched policy tries the live C3 arm on a
        // uniform prefix of the ready set and falls through to serial for the
        // rest (or on NotBatchable).
        match self.policy {
            SchedulePolicy::RoundRobin => {
                for idx in ready {
                    self.serial_advance_one(model, idx, &mut report);
                }
            }
            SchedulePolicy::Batched { max_batch } => {
                self.advance_batched(model, &ready, max_batch, &mut report);
            }
        }

        Ok(report)
    }

    /// The `Batched`-policy advance: try [`BatchedDecode::try_batched_step`] on
    /// up to `max_batch` ready sessions, then serial-advance everything the
    /// batched arm did not consume (an overflow beyond `max_batch`, a
    /// `NotBatchable` gate, or a batched-arm error — which finishes those
    /// sessions in isolation, their private KV untouched). The serial arm
    /// remains reachable for every session.
    fn advance_batched(
        &mut self,
        model: &M,
        ready: &[usize],
        max_batch: usize,
        report: &mut StepReport,
    ) {
        // `ready` is ascending (built from a 0..len filter). The batched prefix
        // and the serial remainder both preserve that order, so the batched
        // slot ordering matches `batch_idxs`.
        let batch_idxs: Vec<usize> = ready.iter().copied().take(max_batch).collect();
        let serial_idxs: Vec<usize> = ready.iter().copied().skip(max_batch).collect();

        // Ask before assembling. A model that implements only the core
        // `DecodeModel` surface has no batched arm, and discovering that by
        // building a batch and catching the decline would repeat the wasted
        // work every step. The serial remainder below still advances every
        // session, so a `Batched` policy on a core-only model degrades to
        // round-robin rather than failing.
        let mut consumed_by_batch = false;
        if batch_idxs.len() >= 2 && model.supports_batched_decode() {
            let dev = self.device.clone();
            let dt = self.dtype;
            let outcome = {
                let mut refs = collect_disjoint_mut(&mut self.sessions, &batch_idxs);
                BatchedDecode::try_batched_step(model, &dev, dt, &mut refs)
            };
            match outcome {
                Ok(BatchOutcome::Advanced(rows)) if rows.len() == batch_idxs.len() => {
                    for (slot, &idx) in batch_idxs.iter().enumerate() {
                        self.sessions[idx].last_logits = Some(rows[slot].clone());
                    }
                    report.used_batched_arm = true;
                    consumed_by_batch = true;
                    for &idx in &batch_idxs {
                        // Logits already stored; sample + record per session.
                        self.finalize_advance(idx, Ok(()), report);
                    }
                }
                Ok(BatchOutcome::Advanced(rows)) => {
                    // Malformed batched result (row count mismatch): treat as a
                    // batched-arm error — finish those sessions in isolation
                    // (their KV was not mutated). Never panic on a bad slice.
                    let msg = format!(
                        "BatchedDecode: Advanced returned {} rows for {} sessions",
                        rows.len(),
                        batch_idxs.len()
                    );
                    for &idx in &batch_idxs {
                        self.record_error_msg(idx, msg.clone(), report);
                    }
                    consumed_by_batch = true;
                }
                Ok(BatchOutcome::NotBatchable) => {
                    // Fall through to serial for the whole batch prefix.
                }
                Err(e) => {
                    // All-or-nothing (spec risk #7): the batched arm returns
                    // Err only BEFORE any session's KV is mutated, so no
                    // session is half-written. Force the affected sessions to
                    // Finished-with-error in isolation.
                    let msg = e.to_string();
                    for &idx in &batch_idxs {
                        self.record_error_msg(idx, msg.clone(), report);
                    }
                    consumed_by_batch = true;
                }
            }
        }

        if !consumed_by_batch {
            for idx in batch_idxs {
                self.serial_advance_one(model, idx, report);
            }
        }
        for idx in serial_idxs {
            self.serial_advance_one(model, idx, report);
        }
    }

    /// Advance one decode-ready session serially by exactly one forward on its
    /// last token, then sample. Errors isolate into a recorded per-session
    /// failure.
    fn serial_advance_one(&mut self, model: &M, idx: usize, report: &mut StepReport) {
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
    fn forward_and_store(model: &M, s: &mut SessionState, input: &[u32]) -> fuel::Result<()> {
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
    fn finalize_advance(&mut self, idx: usize, advance: fuel::Result<()>, report: &mut StepReport) {
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
    fn record_error(&mut self, idx: usize, e: fuel::Error, report: &mut StepReport) {
        self.record_error_msg(idx, e.to_string(), report);
    }

    /// String-message variant of [`Self::record_error`] — used where the same
    /// error message finishes several batched sessions (`fuel_ir::Error` is
    /// not `Clone`, so the message is formatted once and cloned per session).
    fn record_error_msg(&mut self, idx: usize, msg: String, report: &mut StepReport) {
        self.sessions[idx].phase = SessionPhase::Finished;
        let id = self.sessions[idx].id;
        report.errored.push((id, msg));
        report.finished.push(id);
    }

    /// Loop `step` until every session is `Finished`; return each session's
    /// full token sequence (prompt + generated) in insertion order.
    pub fn run_to_completion(&mut self) -> fuel::Result<Vec<(SessionId, Vec<u32>)>> {
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
        self.sessions
            .iter()
            .all(|s| s.phase == SessionPhase::Finished)
    }

    /// (test-support) Add a session whose FIRST advance is forced to error, for
    /// the isolation gate (T4). After building a normal `SessionState`, this truncates its
    /// `KvCache::layers` to empty so `cache.n_layers() == 0 != model.n_layers`,
    /// which `forward_with_kv_context*` rejects with a typed `Error::Msg`
    /// through the true advance path — no panic, no test-only error branch. The
    /// scheduler catches that `Err` into `StepReport::errored` and continues.
    #[doc(hidden)]
    pub fn add_poisoned_session_for_test(
        &mut self,
        prompt: &[u32],
        max_new: usize,
    ) -> fuel::Result<SessionId> {
        let id = self.add_session(prompt, SamplingStrategy::Greedy, None, max_new)?;
        if let Some(s) = self.sessions.last_mut() {
            // Corrupt the real cache so the real forward path errors.
            s.cache.layers.clear();
        }
        Ok(id)
    }
}

/// Collect disjoint `&mut` references to the elements of `sessions` at the
/// given `idxs`. Returned in ASCENDING index order (a single `iter_mut` pass),
/// which matches the ascending `batch_idxs` the scheduler builds — so the
/// returned slot ordering is the caller's slot ordering. `idxs` must be
/// distinct; duplicates are silently taken once.
fn collect_disjoint_mut<'a>(
    sessions: &'a mut [SessionState],
    idxs: &[usize],
) -> Vec<&'a mut SessionState> {
    let mut want: std::collections::HashSet<usize> = idxs.iter().copied().collect();
    let mut out: Vec<&'a mut SessionState> = Vec::with_capacity(idxs.len());
    for (i, s) in sessions.iter_mut().enumerate() {
        if want.remove(&i) {
            out.push(s);
        }
    }
    out
}

// ===========================================================================
// PagedSessionScheduler — the paged-storage decode driver (PS3)
// ===========================================================================

/// Paged multi-session decode driver — the paged-storage counterpart of
/// [`SessionScheduler`]. Every session's KV physically lives in ONE shared
/// [`DeviceKvPool`], with blocks allocated **incrementally** per token via the
/// model's [`DecodeModel::forward_paged_step`] (`Op::PagedAttn`), not reserved up
/// front against a fixed-capacity per-session cache. That incremental growth is
/// the paging memory win: a session that stops early never held blocks it didn't
/// use.
///
/// Serial arm only (byte-exact per session); paged **batched** decode and C-3
/// (evict/restore/splice) on the live path are PS4. Admission is **optimistic** —
/// `add_session` opens a pool session without reserving its full length; if the
/// shared pool exhausts mid-decode, the growing session's `forward_paged_step`
/// returns a typed error that the scheduler **isolates** into that session
/// finishing (never a panic), and the others keep decoding. A consumer that wants
/// to admit conservatively pre-checks [`kv_free_blocks`](Self::kv_free_blocks)
/// against its expected length first (C-1).
pub struct PagedSessionScheduler<'m, M: PagedDecodeModel> {
    model: &'m M,
    /// The shared device KV pool — all sessions' blocks live here.
    pool: DeviceKvPool,
    sessions: Vec<PagedSession>,
    next_id: u64,
    /// Decode planning mode (default [`PagedDecodePlan::PlanOnce`] — build + reuse
    /// each session's optimized decode plan across tokens). Flip to `Replan` via
    /// [`set_plan`](Self::set_plan) for the pre-plan-once behavior (re-plan and
    /// re-realize every token). Correctness is identical either way; the flag only
    /// trades a one-time build for per-token planner savings.
    plan: PagedDecodePlan,
}

/// One session of a [`PagedSessionScheduler`]. Its KV is the pool blocks reached
/// through `handle`; it holds no tensor state of its own (contrast
/// [`SessionState`], which owns a contiguous `KvCache`). Per-session RNG is the
/// contamination firewall (T1), exactly as in the contiguous scheduler.
struct PagedSession {
    handle: SessionHandle,
    tokens: Vec<u32>,
    new_tokens: Vec<u32>,
    rng_state: u64,
    strategy: SamplingStrategy,
    eos_id: Option<u32>,
    remaining: usize,
    phase: SessionPhase,
    last_logits: Option<Vec<f32>>,
    id: SessionId,
    /// `Some` while the session is **suspended** (C-3): its pool blocks have been
    /// evicted (freed to host in this handle), so it holds no VRAM and does not
    /// decode until [`restore_session`](PagedSessionScheduler::restore_session)
    /// writes the bytes back into fresh blocks.
    suspended: Option<DeviceEvicted>,
    /// Held plan-once decode plan (the paged twin of [`SessionState::session`]).
    /// Built on this session's FIRST persistent decode token when the scheduler's
    /// `plan == PlanOnce`; stays `None` under `Replan`. Rebound per subsequent
    /// token, so the optimizer runs once per session rather than per token. Dropped
    /// on eviction ([`evict_session`](PagedSessionScheduler::evict_session)) so a
    /// restored session rebuilds against its fresh blocks.
    decode_session: Option<PagedDecodeSession>,
    /// Fixed block-table capacity for the held plan
    /// (`⌈(prompt + max_new) / block_size⌉`), computed at admission so the
    /// plan-once graph's `block_table` shape is stable across decode tokens.
    max_blocks_cap: usize,
    /// How many LEADING prompt tokens are already resident in this session's
    /// blocks and must be SKIPPED by prefill. `0` for an ordinary session; for a
    /// prefix-shared session ([`add_session_sharing_prefix`](PagedSessionScheduler::add_session_sharing_prefix))
    /// it is the spliced shared-prefix token count, so prefill feeds only
    /// `tokens[prefill_start..]` — the donor already computed the prefix KV and it
    /// sits at the right absolute positions (`filled_tokens == prefill_start` after
    /// the splice), so re-feeding it would double-write and mis-position.
    prefill_start: usize,
    /// Inspection hook (sibling of
    /// [`session_realize_count`](PagedSessionScheduler::session_realize_count)):
    /// when `Some`, each pre-sample logits vector is cloned here in decode order
    /// (the prefill's first-token logits, then one per decode step) before
    /// `sample` consumes it. `None` (default) captures nothing and costs nothing.
    /// Exists because sampled-token equality is too coarse an oracle for
    /// KV-state-dependent behavior — a sub-1e-3 logit perturbation (e.g. a
    /// mis-positioned prefix) hides under both greedy argmax and seeded
    /// multinomial, so a correctness test must compare logits, not tokens.
    captured_logits: Option<Vec<Vec<f32>>>,
}

impl<'m, M: PagedDecodeModel> PagedSessionScheduler<'m, M> {
    /// Build an empty paged scheduler over a shared model + a KV block-pool
    /// `budget`. The pool's head geometry is taken from the model; `budget` sets
    /// block size + total blocks (the shared VRAM ceiling).
    pub fn new(
        model: &'m M,
        budget: KvBudget,
        dtype: DType,
        device: &Device,
    ) -> fuel::Result<Self> {
        let dims = ModelDims::from_model(model)?;
        let pool = DeviceKvPool::new(
            KvGeometry {
                n_layers: dims.n_layers,
                n_kv_heads: dims.n_kv_heads,
                head_dim: dims.head_dim,
                num_blocks: budget.num_blocks,
                block_size: budget.block_size,
                elem_size: dtype.size_in_bytes(),
            },
            dtype,
            device,
        )?;
        Ok(Self {
            model,
            pool,
            sessions: Vec::new(),
            next_id: 0,
            plan: PagedDecodePlan::PlanOnce,
        })
    }

    /// Change the decode planning mode. **The driver default is
    /// [`PagedDecodePlan::PlanOnce`]** — each session builds its decode graph +
    /// optimized plan once and rebinds it per token. `Replan` re-plans and
    /// re-realizes every token; it is retained as the parity reference arm and an
    /// escape hatch, not as a recommended configuration. Output is identical either
    /// way (the plan-once↔replan parity gate) — the flag trades a one-time build
    /// for per-token planner savings. Applies to sessions' subsequent decode tokens.
    ///
    /// Measured cost of the old `Replan` default on this route: **29.7×**
    /// (8,192.0 → 275.8 ms/token, nsys, 2026-08-01).
    pub fn set_plan(&mut self, plan: PagedDecodePlan) {
        self.plan = plan;
    }

    /// Current decode planning mode.
    pub fn plan(&self) -> PagedDecodePlan {
        self.plan
    }

    /// How many times session `id`'s held plan-once decode graph has been REBOUND
    /// (`None` if the session is unknown or holds no plan — e.g. under `Replan`, or
    /// before its first decode token). A positive count is direct evidence the
    /// persistent decode path ran for this session. Observability/test hook.
    pub fn session_realize_count(&self, id: SessionId) -> Option<usize> {
        self.sessions
            .iter()
            .find(|s| s.id == id)
            .and_then(|s| s.decode_session.as_ref())
            .map(|ds| ds.realize_count())
    }

    /// Begin capturing session `id`'s per-step decode logits (see
    /// [`PagedSession::captured_logits`]) — each pre-sample logits vector is
    /// cloned in decode order until [`take_captured_logits`](Self::take_captured_logits).
    /// Inspection/test observability; the discriminating oracle for
    /// KV-state-dependent behavior (token equality is too coarse). No-op if `id`
    /// is already capturing; `Err` for an unknown session.
    pub fn capture_logits(&mut self, id: SessionId) -> fuel::Result<()> {
        let s = self
            .sessions
            .iter_mut()
            .find(|s| s.id == id)
            .ok_or_else(|| Error::Msg(format!("capture_logits: unknown session {id:?}")).bt())?;
        if s.captured_logits.is_none() {
            s.captured_logits = Some(Vec::new());
        }
        Ok(())
    }

    /// Take (and stop) the logits captured for session `id` since
    /// [`capture_logits`](Self::capture_logits) — the prefill's first-token logits
    /// followed by one vector per decode step. `None` if `id` is unknown or was
    /// never capturing.
    pub fn take_captured_logits(&mut self, id: SessionId) -> Option<Vec<Vec<f32>>> {
        self.sessions
            .iter_mut()
            .find(|s| s.id == id)
            .and_then(|s| s.captured_logits.take())
    }

    /// Free pool blocks (C-1) — the consumer's optional conservative-admission
    /// pre-check against a session's expected length.
    pub fn kv_free_blocks(&self) -> usize {
        self.pool.core().free_blocks()
    }

    /// Number of live (not-yet-reaped) sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Whether session `id` is currently suspended (evicted).
    pub fn is_suspended(&self, id: SessionId) -> bool {
        self.sessions
            .iter()
            .any(|s| s.id == id && s.suspended.is_some())
    }

    // --- C-3 on the live path: evict / restore a decoding session --------

    /// **Suspend** (evict) a live session — the pressure valve for the optimistic
    /// admission of [`add_session`](Self::add_session). Its KV bytes are captured
    /// device→host and its pool blocks are freed for other sessions, so
    /// [`kv_free_blocks`](Self::kv_free_blocks) rises; the session stops decoding
    /// until [`restore_session`](Self::restore_session). *Which* session to evict
    /// is the consumer's policy call — this is the mechanism. No-op (Ok) if the
    /// session is already suspended; `Err` for an unknown or finished session.
    pub fn evict_session(&mut self, id: SessionId) -> fuel::Result<()> {
        let idx = self
            .sessions
            .iter()
            .position(|s| s.id == id)
            .ok_or_else(|| Error::Msg(format!("evict_session: unknown session {id:?}")).bt())?;
        if self.sessions[idx].suspended.is_some() {
            return Ok(()); // already suspended
        }
        if self.sessions[idx].phase == SessionPhase::Finished {
            return Err(Error::Msg(format!("evict_session: session {id:?} is finished")).bt());
        }
        let handle = self.sessions[idx].handle;
        let evicted = self.pool.evict(handle)?;
        self.sessions[idx].suspended = Some(evicted);
        // Drop any held plan-once plan: restore re-allocates fresh physical blocks,
        // so the plan rebuilds on the first post-restore decode token (eviction is
        // already a full host round-trip — one rebuild is negligible).
        self.sessions[idx].decode_session = None;
        Ok(())
    }

    /// **Resume** (restore) a suspended session: re-allocate its blocks and write
    /// the captured bytes back (host→device, byte-exact — see
    /// [`DeviceKvPool::restore`]), so it decodes from exactly where it left off.
    /// `Err` if the session is unknown or not suspended, or if the pool can't fit
    /// its blocks again — in which case the session **stays suspended and
    /// restorable** (the capacity is pre-checked, so a failure never consumes the
    /// captured bytes; the consumer frees room and retries).
    pub fn restore_session(&mut self, id: SessionId) -> fuel::Result<()> {
        let idx = self
            .sessions
            .iter()
            .position(|s| s.id == id)
            .ok_or_else(|| Error::Msg(format!("restore_session: unknown session {id:?}")).bt())?;
        let need = match &self.sessions[idx].suspended {
            Some(ev) => ev.saved_block_count(),
            None => {
                return Err(Error::Msg(format!(
                    "restore_session: session {id:?} is not suspended"
                ))
                .bt());
            }
        };
        let free = self.pool.core().free_blocks();
        if need > free {
            return Err(Error::Msg(format!(
                "restore_session: needs {need} blocks, {free} free — session {id:?} stays \
                 suspended (free room and retry)",
            ))
            .bt());
        }
        // Pre-checked, so `restore` cannot OutOfBlocks — only now consume the handle.
        let evicted = self.sessions[idx]
            .suspended
            .take()
            .expect("checked Some above");
        let handle = self.sessions[idx].handle;
        self.pool.restore(handle, evicted)
    }

    /// Admit a session from a prompt (optimistic — no up-front block reservation).
    /// Opens a pool session; the prompt's blocks are allocated on the first
    /// `step`'s prefill. Rejects an empty prompt / zero budget before opening.
    pub fn add_session(
        &mut self,
        prompt: &[u32],
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        max_new: usize,
    ) -> fuel::Result<SessionId> {
        if prompt.is_empty() {
            return Err(
                Error::Msg("PagedSessionScheduler::add_session: prompt is empty".into()).bt(),
            );
        }
        if max_new == 0 {
            return Err(Error::Msg(
                "PagedSessionScheduler::add_session: max_new must be > 0".into(),
            )
            .bt());
        }
        let id = SessionId(self.next_id);
        let handle = self.pool.core_mut().open();
        let rng_state = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };
        // Fixed block-table capacity for a plan-once hold: the session's whole
        // decode length rounded up to blocks. Known now (prompt + max_new); pins
        // the held graph's block_table shape. `.max(1)` guards a degenerate 0.
        let block_size = self.pool.geometry().block_size;
        let max_blocks_cap = (prompt.len() + max_new).div_ceil(block_size).max(1);
        self.sessions.push(PagedSession {
            handle,
            tokens: prompt.to_vec(),
            new_tokens: Vec::new(),
            rng_state,
            strategy,
            eos_id,
            remaining: max_new,
            phase: SessionPhase::Prefill,
            last_logits: None,
            id,
            suspended: None,
            decode_session: None,
            max_blocks_cap,
            prefill_start: 0,
            captured_logits: None,
        });
        self.next_id += 1;
        Ok(id)
    }

    /// Register the first `prefix_blocks` FULLY-FILLED blocks of a live session as
    /// a shared [`PrefixId`] whose lifetime the pool's registry controls — the
    /// reference wrapper over [`KvBlockPool::register_prefix`]. The donor may then
    /// be reaped/discarded; the owner keeps those blocks resident so a later
    /// [`add_session_sharing_prefix`](Self::add_session_sharing_prefix) can splice
    /// them without racing the donor's teardown. `Err` for an unknown session, or
    /// if the pool refuses (the donor lacks `prefix_blocks` fully-filled blocks) —
    /// never a panic, and a refusal leaves the pool untouched.
    pub fn register_prefix(
        &mut self,
        donor: SessionId,
        prefix_blocks: usize,
    ) -> fuel::Result<PrefixId> {
        let handle = self
            .sessions
            .iter()
            .find(|s| s.id == donor)
            .map(|s| s.handle)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "PagedSessionScheduler::register_prefix: unknown session {donor:?}"
                ))
                .bt()
            })?;
        self.pool
            .core_mut()
            .register_prefix(handle, prefix_blocks)
            .map_err(|e| Error::Msg(format!("PagedSessionScheduler::register_prefix: {e:?}")).bt())
    }

    /// Release a registered prefix ([`KvBlockPool::release_prefix`]): drop the
    /// owner handle so each shared block's refcount falls by one, freeing only the
    /// blocks no live sharer still references. Live sharers keep the prefix alive
    /// independently. `Err` for an unregistered id.
    pub fn release_prefix(&mut self, prefix: PrefixId) -> fuel::Result<()> {
        self.pool
            .core_mut()
            .release_prefix(prefix)
            .map_err(|e| Error::Msg(format!("PagedSessionScheduler::release_prefix: {e:?}")).bt())
    }

    /// Admit a session that REUSES a registered KV prefix (rung-1 prefix sharing) —
    /// the reference caller of the transactional splice. Opens a fresh pool handle,
    /// splices the shared prefix into it ([`KvBlockPool::splice_prefix_from`] —
    /// refcount bump, zero recompute), and admits the session so its prefill feeds
    /// ONLY the unique suffix `prompt[shared_tokens..]`; the donor-computed prefix
    /// KV already occupies positions `0..shared_tokens`, and the first suffix token
    /// derives its position from `filled_tokens == shared_tokens`. CoW is a
    /// non-issue in rung-1: every shared block is fully filled, so the sharer's
    /// first write lands on a FRESH block (guarded by `forward_paged_step`'s
    /// `ensure_writable_block`).
    ///
    /// Rejects — leaving the pool untouched (the just-opened handle is rolled
    /// back) — an empty prompt, a zero budget, a splice the pool refuses (e.g. an
    /// unknown prefix), or a prompt that does not extend past the shared prefix
    /// (`prompt.len() <= shared_tokens`): the suffix must be non-empty to produce
    /// the logits that sample the first generated token.
    pub fn add_session_sharing_prefix(
        &mut self,
        prefix: PrefixId,
        prompt: &[u32],
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        max_new: usize,
    ) -> fuel::Result<SessionId> {
        if prompt.is_empty() {
            return Err(Error::Msg(
                "PagedSessionScheduler::add_session_sharing_prefix: prompt is empty".into(),
            )
            .bt());
        }
        if max_new == 0 {
            return Err(Error::Msg(
                "PagedSessionScheduler::add_session_sharing_prefix: max_new must be > 0".into(),
            )
            .bt());
        }
        let handle = self.pool.core_mut().open();
        // Splice BEFORE pushing the session; a refusal must leave no trace, so the
        // just-opened (empty, block-less) handle is rolled back on any error.
        let spliced = self.pool.core_mut().splice_prefix_from(prefix, handle);
        let shared_tokens = match spliced {
            Ok(n) => n,
            Err(e) => {
                self.pool.core_mut().discard(handle);
                return Err(Error::Msg(format!(
                    "PagedSessionScheduler::add_session_sharing_prefix: \
                     splice_prefix_from failed: {e:?}"
                ))
                .bt());
            }
        };
        if prompt.len() <= shared_tokens {
            self.pool.core_mut().discard(handle);
            return Err(Error::Msg(format!(
                "PagedSessionScheduler::add_session_sharing_prefix: prompt length {} must \
                 exceed the shared prefix ({shared_tokens} tokens) — the suffix must be non-empty",
                prompt.len()
            ))
            .bt());
        }
        let id = SessionId(self.next_id);
        let rng_state = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };
        let block_size = self.pool.geometry().block_size;
        let max_blocks_cap = (prompt.len() + max_new).div_ceil(block_size).max(1);
        self.sessions.push(PagedSession {
            handle,
            tokens: prompt.to_vec(),
            new_tokens: Vec::new(),
            rng_state,
            strategy,
            eos_id,
            remaining: max_new,
            phase: SessionPhase::Prefill,
            last_logits: None,
            id,
            suspended: None,
            decode_session: None,
            max_blocks_cap,
            prefill_start: shared_tokens,
            captured_logits: None,
        });
        self.next_id += 1;
        Ok(id)
    }

    /// Advance one quantum: prefill any `Prefill` sessions (feed the whole prompt
    /// one token at a time — `Op::PagedAttn` is decode-only, and one-at-a-time is
    /// causally equivalent to a batched prefill), then decode-advance every
    /// `Decode`-ready session by one token. Each `forward_paged_step` grows the
    /// session's blocks; a pool exhaustion (or any per-session error) is isolated
    /// into that session finishing-with-error, never propagated.
    pub fn step(&mut self) -> StepReport {
        let mut report = StepReport::default();
        self.prefill_pass(&mut report);
        for idx in self.collect_decode_ready() {
            self.decode_one(idx, &mut report);
        }
        report
    }

    /// Batched decode variant of [`step`](Self::step) (PS4a throughput arm): the
    /// Decode-ready set is partitioned by position and each same-position group
    /// of ≥2 is advanced in ONE `Op::PagedAttn` pass at B=K (up to `max_batch`
    /// per pass); singletons and a non-uniform remainder fall back to serial.
    /// Prefill is always serial (ragged prompts). Provably equal to `step` per
    /// session — the batched forward's row i equals the serial decode (see the
    /// batched↔serial parity gate). `StepReport::used_batched_arm` is set when a
    /// batch actually ran.
    pub fn step_batched(&mut self, max_batch: usize) -> StepReport {
        let mut report = StepReport::default();
        self.prefill_pass(&mut report);

        // Partition ready sessions by position (BTreeMap → deterministic order),
        // then batch each same-position group in chunks of ≤ max_batch.
        let mut by_pos: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for idx in self.collect_decode_ready() {
            let pos = self
                .pool
                .core()
                .filled_tokens(self.sessions[idx].handle)
                .unwrap_or(usize::MAX);
            by_pos.entry(pos).or_default().push(idx);
        }
        for (_pos, group) in by_pos {
            for chunk in group.chunks(max_batch.max(1)) {
                if chunk.len() < 2 {
                    for &idx in chunk {
                        self.decode_one(idx, &mut report);
                    }
                } else {
                    self.decode_batch(chunk, &mut report);
                }
            }
        }
        report
    }

    /// Serial prefill pass: feed each `Prefill` (non-suspended) session's whole
    /// prompt token-by-token, then sample its first token. Errors isolate.
    fn prefill_pass(&mut self, report: &mut StepReport) {
        let model = self.model;
        for idx in 0..self.sessions.len() {
            if self.sessions[idx].phase != SessionPhase::Prefill
                || self.sessions[idx].suspended.is_some()
            {
                continue;
            }
            let prompt = self.sessions[idx].tokens.clone();
            let handle = self.sessions[idx].handle;
            // Skip any leading tokens already resident from a spliced prefix
            // (`prefill_start == 0` for an ordinary session). The donor computed
            // that prefix KV and it sits at the right absolute positions, so we
            // feed ONLY `prompt[prefill_start..]`; the first suffix token derives
            // its position from `filled_tokens == prefill_start`.
            let start = self.sessions[idx].prefill_start.min(prompt.len());
            let mut last_logits: Option<Vec<f32>> = None;
            let mut failure: Option<String> = None;
            for &tok in &prompt[start..] {
                match model.forward_paged_step(tok, &mut self.pool, handle) {
                    Ok(l) => last_logits = Some(l),
                    Err(e) => {
                        failure = Some(e.to_string());
                        break;
                    }
                }
            }
            match failure {
                Some(msg) => self.finish_error(idx, msg, report),
                None => {
                    self.sessions[idx].phase = SessionPhase::Decode;
                    self.sessions[idx].last_logits = last_logits;
                    self.sample(idx, report);
                }
            }
        }
    }

    /// Indices of Decode-ready, non-suspended sessions (ascending).
    fn collect_decode_ready(&self) -> Vec<usize> {
        (0..self.sessions.len())
            .filter(|&i| {
                self.sessions[i].phase == SessionPhase::Decode
                    && self.sessions[i].suspended.is_none()
            })
            .collect()
    }

    /// One serial decode-advance of session `idx` (last token → forward → sample).
    /// Routes through [`DecodeModel::forward_paged_step_persistent`] behind the
    /// scheduler's `plan` flag: `Replan` (default) re-plans this token — identical
    /// to the pre-plan-once `forward_paged_step`; `PlanOnce` builds the held plan on
    /// the first decode token and rebinds it thereafter. `pool` and the session's
    /// `decode_session` are disjoint fields, so both borrow mutably in one call.
    fn decode_one(&mut self, idx: usize, report: &mut StepReport) {
        let model = self.model;
        let handle = self.sessions[idx].handle;
        let cap = self.sessions[idx].max_blocks_cap;
        let plan = self.plan;
        let last = self.sessions[idx].tokens.last().copied();
        match last {
            Some(tok) => {
                let res = model.forward_paged_step_persistent(
                    tok,
                    &mut self.pool,
                    handle,
                    cap,
                    plan,
                    &mut self.sessions[idx].decode_session,
                );
                match res {
                    Ok(l) => {
                        self.sessions[idx].last_logits = Some(l);
                        self.sample(idx, report);
                    }
                    Err(e) => self.finish_error(idx, e.to_string(), report),
                }
            }
            None => self.finish_error(
                idx,
                "PagedSessionScheduler: decode on empty token history".into(),
                report,
            ),
        }
    }

    /// Advance a batch of same-position sessions in one `forward_paged_step_batched`
    /// pass, sampling each. On any error or malformed row count the whole batch
    /// finishes-with-error in isolation (the model layer's KV mutation is
    /// all-or-nothing). `idxs` must be ≥2 same-position, non-suspended Decode
    /// sessions.
    fn decode_batch(&mut self, idxs: &[usize], report: &mut StepReport) {
        let model = self.model;
        let tokens: Vec<u32> = match idxs
            .iter()
            .map(|&i| self.sessions[i].tokens.last().copied())
            .collect::<Option<Vec<u32>>>()
        {
            Some(t) => t,
            None => {
                for &idx in idxs {
                    self.finish_error(idx, "decode_batch: empty token history".into(), report);
                }
                return;
            }
        };
        let handles: Vec<SessionHandle> = idxs.iter().map(|&i| self.sessions[i].handle).collect();
        match model.forward_paged_step_batched(&tokens, &mut self.pool, &handles) {
            Ok(rows) if rows.len() == idxs.len() => {
                for (slot, &idx) in idxs.iter().enumerate() {
                    self.sessions[idx].last_logits = Some(rows[slot].clone());
                    self.sample(idx, report);
                }
                report.used_batched_arm = true;
            }
            Ok(rows) => {
                let msg = format!(
                    "decode_batch: {} rows for {} sessions",
                    rows.len(),
                    idxs.len()
                );
                for &idx in idxs {
                    self.finish_error(idx, msg.clone(), report);
                }
            }
            Err(e) => {
                let msg = e.to_string();
                for &idx in idxs {
                    self.finish_error(idx, msg.clone(), report);
                }
            }
        }
    }

    /// Sample the pending logits with THIS session's own RNG (the T1 firewall),
    /// append, decrement the budget, and transition to `Finished` on eos/budget.
    fn sample(&mut self, idx: usize, report: &mut StepReport) {
        let s = &mut self.sessions[idx];
        if s.phase == SessionPhase::Finished {
            return;
        }
        let logits = match s.last_logits.take() {
            Some(l) => l,
            None => return,
        };
        // Inspection hook: record the exact logits this step sampled from, before
        // they are consumed (opt-in via `capture_logits`; `None` = no-op).
        if let Some(buf) = s.captured_logits.as_mut() {
            buf.push(logits.clone());
        }
        let next = sample_logits(&logits, s.strategy, &mut s.rng_state);
        s.tokens.push(next);
        s.new_tokens.push(next);
        s.remaining = s.remaining.saturating_sub(1);
        let id = s.id;
        report.advanced.push(id);
        if s.eos_id == Some(next) || s.remaining == 0 {
            s.phase = SessionPhase::Finished;
            report.finished.push(id);
        } else {
            s.phase = SessionPhase::Decode;
        }
    }

    /// Force one session to `Finished`-with-error (isolated — the others are
    /// untouched). The pool blocks it holds are freed by [`reap_finished`].
    fn finish_error(&mut self, idx: usize, msg: String, report: &mut StepReport) {
        let s = &mut self.sessions[idx];
        s.phase = SessionPhase::Finished;
        let id = s.id;
        report.errored.push((id, msg));
        report.finished.push(id);
    }

    /// Loop `step` while any session can still make progress — i.e. is neither
    /// `Finished` nor **suspended** — then return each session's full token
    /// sequence in insertion order. A suspended (evicted) session can't advance
    /// until [`restore_session`](Self::restore_session), so this returns with it
    /// still live rather than spinning forever; the consumer restores it and
    /// calls again to finish it.
    pub fn run_to_completion(&mut self) -> Vec<(SessionId, Vec<u32>)> {
        while self
            .sessions
            .iter()
            .any(|s| s.phase != SessionPhase::Finished && s.suspended.is_none())
        {
            self.step();
        }
        self.sessions
            .iter()
            .map(|s| (s.id, s.tokens.clone()))
            .collect()
    }

    /// Reap every `Finished` session: discard its pool session (freeing its
    /// blocks) and drop it, returning each reaped `(id, tokens)`. This is how a
    /// completed session's blocks become admission headroom again.
    pub fn reap_finished(&mut self) -> Vec<(SessionId, Vec<u32>)> {
        let mut reaped = Vec::new();
        let mut kept = Vec::with_capacity(self.sessions.len());
        for s in std::mem::take(&mut self.sessions) {
            if s.phase == SessionPhase::Finished {
                self.pool.core_mut().discard(s.handle);
                reaped.push((s.id, s.tokens));
            } else {
                kept.push(s);
            }
        }
        self.sessions = kept;
        reaped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel::lazy::{LayerWeights, LlamaConfig, LlamaModel, LlamaWeights, SamplingStrategy};
    // NOTE: `fuel_ir::Device` does not exist — the device type is `fuel::Device`
    // (fuel_core::Device), which is what `KvCache::with_capacity` takes. `DType`
    // is `fuel_ir::DType`. This mirrors the `use` lines at the top of
    // inference_context.rs (`use fuel_ir::{DType, ..}; use fuel::Device;`).
    use fuel::Device;
    use fuel_ir::DType;
    use std::sync::Arc;

    fn tiny_cfg() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }
    // Mirror lazy.rs generate_tests::make_tiny_weights_seeded.
    fn tiny_weights(cfg: &LlamaConfig, seed: u32) -> LlamaWeights {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of =
            |n: usize| -> Arc<[f32]> { Arc::from((0..n).map(|_| next()).collect::<Vec<_>>()) };
        let kv = cfg.n_kv_heads * cfg.head_dim;
        LlamaWeights {
            instance: fuel::decode_shape::ModelInstanceId::next(),
            token_embedding: vec_of(cfg.vocab_size * cfg.dim),
            layers: (0..cfg.n_layers)
                .map(|_| LayerWeights {
                    attn_q: vec_of(cfg.dim * cfg.dim).into(),
                    attn_q_bias: None,
                    attn_k: vec_of(cfg.dim * kv).into(),
                    attn_k_bias: None,
                    attn_v: vec_of(cfg.dim * kv).into(),
                    attn_v_bias: None,
                    attn_o: vec_of(cfg.dim * cfg.dim).into(),
                    ffn_gate: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_up: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_down: vec_of(cfg.ffn_dim * cfg.dim).into(),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output: vec_of(cfg.dim * cfg.vocab_size).into(),
        }
    }
    // CAUTION: this tiny random model's GREEDY output is a fixed point — its
    // argmax collapses to one token and stays there regardless of KV context, so
    // greedy-token equality asserts NOTHING about cache-dependent behavior (a
    // model that ignored its KV entirely would pass identically). Seeded
    // temperature sampling is barely better: a sub-1e-3 logit perturbation (e.g. a
    // mis-positioned prefix) does not cross a multinomial boundary either. For any
    // oracle that must distinguish correct KV state from corrupted, compare
    // LOGITS, not sampled tokens (see `paged_scheduler_prefix_shared_matches_from_scratch`
    // and the `capture_logits` hook). Token-equality tests here that DO have teeth
    // get them from a structural guard (e.g. `session_realize_count`), not the tokens.
    fn tiny_model(seed: u32) -> LlamaModel {
        let cfg = tiny_cfg();
        LlamaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg, seed),
        }
    }
    /// A generous KV budget for tests that don't exercise the capacity gate —
    /// large enough that no admission is ever rejected. Capacity-gate tests
    /// build their own tight budget.
    fn test_budget() -> KvBudget {
        KvBudget {
            block_size: 16,
            num_blocks: 4096,
        }
    }
    fn dims(cfg: &LlamaConfig) -> ModelDims {
        ModelDims {
            n_layers: cfg.n_layers,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
        }
    }

    // ---------------- GAP-029: quantized LLaMA decode surface ----------------

    /// A Q4_0-bakeable LLaMA-3.1 config **whose scaling is not a no-op**, which
    /// is a real constraint rather than an arbitrary choice — see the arithmetic
    /// in `quantized_llama_decode_delegates_to_scaling_aware_inner`.
    /// `hidden_size` and `intermediate_size` must both be multiples of 32 (Q4_0
    /// blocks run along in-features).
    fn scaled_q4_0_cfg() -> fuel::lazy_llama_full::LlamaFullConfig {
        use fuel::lazy_llama_full::{Llama3RopeConfig, Llama3RopeType, LlamaFullConfig};
        LlamaFullConfig {
            hidden_size: 32,
            intermediate_size: 64,
            vocab_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 128,
            bos_token_id: None,
            eos_token_id: None,
            // The real LLaMA-3.1 constants.
            rope_scaling: Some(Llama3RopeConfig {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192,
                rope_type: Llama3RopeType::Llama3,
            }),
            tie_word_embeddings: false,
        }
    }

    /// Prefill `prompt`, then decode `n_decode` further tokens, returning every
    /// decode step's logits concatenated. Takes `&dyn DecodeModel` so the three
    /// arms under test run byte-identical driver code and differ only in which
    /// model they are.
    ///
    /// **`n_decode` must be > 1, and that is a correctness requirement, not
    /// thoroughness.** The persistent path has two distinct halves — the first
    /// decode token BUILDS and optimizes the held graph, every later token
    /// REBINDS it — and `LlamaModel::forward_with_kv_context_persistent_inv_freq`
    /// documents that threading the RoPE override into only one of them yields a
    /// model whose first decode token is scaled and whose rest are not: a
    /// silent, position-dependent wrong answer. A single-token probe exercises
    /// only the build half and would miss exactly that.
    fn prefill_then_decode(
        m: &dyn DecodeModel,
        prompt: &[u32],
        first_next: u32,
        n_decode: usize,
        dev: &Device,
    ) -> Vec<f32> {
        assert!(
            n_decode > 1,
            "must cover the rebind half, not just the build half"
        );
        let dims = ModelDims::from_model(m).expect("uniform per-head KV geometry");
        let mut cache = KvCache::with_capacity(
            dims.n_layers,
            dims.n_kv_heads,
            dims.head_dim,
            128,
            DType::F32,
            dev,
        )
        .expect("kv cache");
        let mut ctx = InferenceContext::new(dev.clone());
        let mut sess: Option<DecodeSession> = None;
        m.forward_with_kv_context_persistent(prompt, &mut cache, &mut ctx, &mut sess)
            .expect("prefill");

        let mut all = Vec::new();
        let mut tok = first_next;
        for step in 0..n_decode {
            let logits = m
                .forward_with_kv_context_persistent(&[tok], &mut cache, &mut ctx, &mut sess)
                .unwrap_or_else(|e| panic!("decode step {step}: {e:?}"));
            // Feed a deterministic, model-independent next token so all three
            // arms walk identical token paths — sampling here would let the
            // arms diverge for a reason unrelated to RoPE.
            tok = (tok + 1) % 32;
            all.extend_from_slice(&logits);
        }
        all
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "logit rows must be the same width");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// GAP-029 — the quantized LLaMA decode surface must delegate to the
    /// **scaling-aware** `Llama3Model`, not to the `LlamaModel` two levels down.
    ///
    /// ## Why the negative control *is* the test
    ///
    /// LLaMA-3.1 RoPE scaling is a per-dimension transform of `inv_freq`, and
    /// the angle it feeds is `pos * inv_freq`. So scaled and unscaled agree
    /// **exactly at position 0** and diverge **linearly with position**. At low
    /// positions the real LLaMA-3.1 constants separate the two arms by ~1e-7 —
    /// below f32 realize noise — so a parity check near the start of a sequence
    /// passes no matter which arm the impl delegates to. That has already
    /// produced one green-but-vacuous decode test in this project, so the
    /// load-bearing assertion here is not `A == B`; it is the **negative
    /// control** that the correct arm (scaled) and the wrong arm (unscaled, two
    /// levels down) are genuinely far apart at the position under test. If that
    /// separation ever collapses, this test announces its own vacuity by
    /// failing instead of passing quietly.
    ///
    /// ## Why this shape, specifically
    ///
    /// With `head_dim = 8` and `rope_theta = 1e4`, `inv_freq` is
    /// `[1e0, 1e-1, 1e-2, 1e-3]` → wavelengths `[6.3, 63, 628, 6283]`. The
    /// LLaMA-3.1 cuts are `orig/high = 8192/4 = 2048` and
    /// `orig/low = 8192/1 = 8192`. Only the last wavelength sits between them,
    /// so exactly one of four dimensions lands in the smooth interpolation band
    /// and moves `1e-3 → ~2.14e-4`. At position 40 that is an angle separation
    /// of ~0.031 rad — five orders of magnitude above f32 noise.
    ///
    /// Note `tiny_cfg`'s `head_dim = 4` would make the scaling a **no-op**: its
    /// wavelengths (6.3, 628) both sit below the 2048 cut, nothing is rescaled,
    /// and the negative control would correctly refuse to certify the test.
    ///
    /// Asserts on LOGITS, not sampled tokens — this module's tiny random model
    /// has a greedy argmax that is a fixed point (see the `tiny_model` caution
    /// above), so token equality would be vacuous here as well.
    ///
    /// ## Measured, so the next reader knows the actual headroom
    ///
    /// `|A-B| = 0.000e0` (bit-identical — both arms call the same function, as
    /// a pure delegation should) and `|B-C| = 2.118e-3` against a `1e-3` floor.
    /// So discrimination is total — exact-zero versus 2.1e-3 — but the margin
    /// above the floor is only **~2.1×**, not orders of magnitude. Lowering the
    /// decode position or shrinking `head_dim` will drop it under, at which
    /// point the negative control fires and the test declares its own vacuity.
    /// That is the intended behaviour; do not "fix" it by lowering the floor.
    ///
    /// Note also what `2.118e-3` corroborates: it sits right at the ~1e-3 scale
    /// that seeded temperature sampling is documented to swallow. A
    /// token-equality oracle would have seen **nothing** here — which is the
    /// scar this module's `tiny_model` comment already records, now with a
    /// number attached.
    #[test]
    fn quantized_llama_decode_delegates_to_scaling_aware_inner() {
        use fuel::lazy_quantized_llama::QuantizedLlama3Model;

        let dev = Device::cpu();
        let cfg = scaled_q4_0_cfg();
        let lazy_cfg = cfg.to_lazy_config();
        let src = tiny_weights(&lazy_cfg, 909);
        let qmodel = QuantizedLlama3Model::from_f32_bake(cfg.clone(), src).expect("Q4_0 bake");

        // Position 40 for the decode step — high enough that scaled/unscaled
        // separate (see the arithmetic above), well inside max_position 128.
        let prompt: Vec<u32> = (0..40u32).map(|i| i % lazy_cfg.vocab_size as u32).collect();
        let next = 7u32;

        // Four decode steps: the first BUILDS the held graph, the rest REBIND
        // it. Both halves must carry the scaling (see `prefill_then_decode`).
        const N_DECODE: usize = 4;

        // A — the surface under test: `DecodeModel` on the quantized wrapper.
        let a = prefill_then_decode(&qmodel, &prompt, next, N_DECODE, &dev);
        // B — the CORRECT oracle: the scaling-aware `Llama3Model` it wraps.
        let b = prefill_then_decode(qmodel.inner(), &prompt, next, N_DECODE, &dev);
        // C — the WRONG delegation, simulated: the `LlamaModel` two levels
        //     down, whose own entry point uses the UNSCALED RoPE base. This is
        //     what `forward_hidden` already does by reaching `self.inner.inner`.
        let c = prefill_then_decode(&qmodel.inner().inner, &prompt, next, N_DECODE, &dev);

        let ab = max_abs_diff(&a, &b);
        let bc = max_abs_diff(&b, &c);
        eprintln!(
            "[GAP-029] delegation parity: |A-B| = {ab:.3e} (want ~0), \
             separation |B-C| = {bc:.3e} (want >> 0)"
        );

        // NEGATIVE CONTROL FIRST: if the two arms are not distinguishable at
        // this position, nothing below this line means anything.
        assert!(
            bc > 1e-3,
            "VACUOUS TEST: scaled and unscaled RoPE differ by only {bc:.3e} at \
             position {}, so `A == B` cannot distinguish correct delegation \
             from delegating one level too deep. Raise the decode position or \
             fix the config so the scaling is not a no-op — do NOT lower this \
             threshold.",
            prompt.len(),
        );

        // The actual claim: the wrapper's decode surface IS the scaled path.
        assert!(
            ab < 1e-6,
            "QuantizedLlama3Model's DecodeModel impl does not match the \
             scaling-aware Llama3Model (|A-B| = {ab:.3e}). It is almost \
             certainly delegating to `self.inner().inner` (the unscaled \
             LlamaModel); |B-C| here is {bc:.3e}.",
        );
    }

    /// The dims the scheduler builds a session's KV cache from must come from
    /// the quantized wrapper unchanged — a wrong `n_kv_heads`/`head_dim` would
    /// mis-shape every cache allocated for a GGUF model.
    #[test]
    fn quantized_llama_reports_inner_cache_geometry() {
        use fuel::lazy_quantized_llama::QuantizedLlama3Model;

        let cfg = scaled_q4_0_cfg();
        let src = tiny_weights(&cfg.to_lazy_config(), 4242);
        let m = QuantizedLlama3Model::from_f32_bake(cfg.clone(), src).expect("Q4_0 bake");

        // The wrapper's per-layer specs must collapse to the inner geometry —
        // the path the scheduler actually takes at construction.
        let dims = ModelDims::from_model(&m).expect("uniform per-head KV geometry");
        assert_eq!(dims.n_layers, cfg.num_hidden_layers);
        assert_eq!(dims.n_kv_heads, cfg.num_key_value_heads);
        assert_eq!(dims.head_dim, cfg.head_dim);
        // Guards the consistency rule the trait documents: the batched-arm
        // predicate and the batched-arm method must agree. Delegated, so this
        // stays true if `Llama3Model` ever gains the arm.
        assert_eq!(
            DecodeModel::supports_batched_decode(&m),
            DecodeModel::supports_batched_decode(m.inner()),
        );
    }

    #[test]
    fn session_new_seeds_prefill_state() {
        let cfg = tiny_cfg();
        let s = SessionState::new(
            SessionId(0),
            dims(&cfg),
            &[1, 2, 3],
            SamplingStrategy::Greedy,
            None,
            5,
            &Device::cpu(),
            DType::F32,
        )
        .unwrap();
        assert_eq!(s.tokens(), &[1, 2, 3]);
        assert_eq!(s.phase, SessionPhase::Prefill);
        assert!(s.is_ready());
    }

    #[test]
    fn session_new_rejects_empty_prompt_and_zero_budget() {
        let cfg = tiny_cfg();
        assert!(
            SessionState::new(
                SessionId(0),
                dims(&cfg),
                &[],
                SamplingStrategy::Greedy,
                None,
                5,
                &Device::cpu(),
                DType::F32
            )
            .is_err()
        );
        assert!(
            SessionState::new(
                SessionId(0),
                dims(&cfg),
                &[1, 2],
                SamplingStrategy::Greedy,
                None,
                0,
                &Device::cpu(),
                DType::F32
            )
            .is_err()
        );
    }

    #[test]
    fn sample_and_append_greedy_appends_argmax_and_counts_budget() {
        let cfg = tiny_cfg();
        let mut s = SessionState::new(
            SessionId(0),
            dims(&cfg),
            &[1, 2],
            SamplingStrategy::Greedy,
            None,
            2,
            &Device::cpu(),
            DType::F32,
        )
        .unwrap();
        // argmax at index 3
        s.last_logits = Some(vec![0.0, 0.1, 0.2, 0.9, 0.3]);
        let t = s.sample_and_append().unwrap();
        assert_eq!(t, Some(3));
        assert_eq!(s.tokens(), &[1, 2, 3]);
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
        let mut s = SessionState::new(
            SessionId(0),
            dims(&cfg),
            &[1],
            SamplingStrategy::Greedy,
            Some(3),
            10,
            &Device::cpu(),
            DType::F32,
        )
        .unwrap();
        s.last_logits = Some(vec![0.0, 0.0, 0.0, 0.9, 0.0]); // argmax 3 == eos
        assert_eq!(s.sample_and_append().unwrap(), Some(3));
        assert_eq!(s.phase, SessionPhase::Finished);
    }

    #[test]
    fn sample_and_append_noop_without_logits() {
        let cfg = tiny_cfg();
        let mut s = SessionState::new(
            SessionId(0),
            dims(&cfg),
            &[1],
            SamplingStrategy::Greedy,
            None,
            3,
            &Device::cpu(),
            DType::F32,
        )
        .unwrap();
        assert_eq!(s.sample_and_append().unwrap(), None);
    }

    #[test]
    fn scheduler_single_session_matches_standalone_generate() {
        let model = tiny_model(9999);
        let prompt = [1u32, 2, 3];
        let max_new = 5;
        let standalone = model
            .generate_with_kv_context(
                &prompt,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &Device::cpu(),
                DType::F32,
            )
            .unwrap();

        let mut sched = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::RoundRobin,
            test_budget(),
        )
        .unwrap();
        let id = sched
            .add_session(&prompt, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let out = sched.run_to_completion().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, id);
        assert_eq!(out[0].1, standalone); // byte-identical token stream
    }

    /// C-1: admission is gated on the KV block pool. A tight budget admits until
    /// blocks run out, rejects the over-committing session BEFORE building its
    /// cache (rejection consumes nothing), and `reap_finished` turns completed
    /// sessions' reservations back into admission headroom. C-4 resident bytes
    /// track the reservations.
    #[test]
    fn admission_is_capacity_gated_c1_and_reap_reclaims() {
        let model = tiny_model(31);
        // block_size 8; a (prompt 3 + max_new 5 = 8)-token session = exactly 1 block.
        let budget = KvBudget {
            block_size: 8,
            num_blocks: 2,
        };
        let mut sched = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::RoundRobin,
            budget,
        )
        .unwrap();
        let prompt = [1u32, 2, 3];

        assert_eq!(
            sched.kv_blocks_required(prompt.len(), 5),
            1,
            "8 tokens / block_size 8 = 1 block"
        );
        assert_eq!(sched.kv_free_blocks(), 2);
        assert_eq!(sched.kv_bytes_resident(), 0, "C-4: nothing reserved yet");

        sched
            .add_session(&prompt, SamplingStrategy::Greedy, None, 5)
            .unwrap();
        sched
            .add_session(&prompt, SamplingStrategy::Greedy, None, 5)
            .unwrap();
        assert_eq!(sched.kv_free_blocks(), 0, "both blocks reserved");
        assert!(
            sched.kv_bytes_resident() > 0,
            "C-4: resident bytes reflect the 2 reserved blocks"
        );

        // Third admission is rejected on capacity — total, nothing half-built.
        let rejected = sched.add_session(&prompt, SamplingStrategy::Greedy, None, 5);
        assert!(
            rejected.is_err(),
            "no room → typed capacity rejection (C-1)"
        );
        assert_eq!(
            sched.kv_free_blocks(),
            0,
            "a rejected admit consumes no blocks"
        );

        // Complete the two sessions, reap them, and the reservations return.
        let _ = sched.run_to_completion().unwrap();
        let reaped = sched.reap_finished();
        assert_eq!(reaped.len(), 2, "both finished sessions reaped");
        assert_eq!(sched.kv_free_blocks(), 2, "reap reclaimed both blocks");
        assert_eq!(sched.kv_bytes_resident(), 0, "C-4: back to zero after reap");

        // A fresh session admits again into the reclaimed headroom.
        sched
            .add_session(&prompt, SamplingStrategy::Greedy, None, 5)
            .unwrap();
        assert_eq!(sched.kv_free_blocks(), 1);
    }

    /// A multi-block session reserves ⌈max_seq_len / block_size⌉ blocks, and the
    /// pre-check query agrees with what admission actually reserves.
    #[test]
    fn kv_blocks_required_matches_multi_block_reservation() {
        let model = tiny_model(7);
        let budget = KvBudget {
            block_size: 4,
            num_blocks: 16,
        };
        let mut sched = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::RoundRobin,
            budget,
        )
        .unwrap();
        let prompt = [1u32, 2, 3, 4, 5]; // 5 + max_new 6 = 11 tokens → ceil(11/4) = 3 blocks
        assert_eq!(sched.kv_blocks_required(prompt.len(), 6), 3);
        let free_before = sched.kv_free_blocks();
        sched
            .add_session(&prompt, SamplingStrategy::Greedy, None, 6)
            .unwrap();
        assert_eq!(
            sched.kv_free_blocks(),
            free_before - 3,
            "admission reserved exactly the queried 3 blocks"
        );
    }

    // --- PagedSessionScheduler (PS3) -------------------------------------

    fn paged_budget() -> KvBudget {
        KvBudget {
            block_size: 4,
            num_blocks: 64,
        }
    }

    /// A single paged session runs to its `max_new` budget: prefill (feed the
    /// prompt token-by-token) + decode, ending with prompt + max_new tokens.
    #[test]
    fn paged_scheduler_single_session_runs_to_budget() {
        let model = tiny_model(42);
        let mut s =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        let prompt = [1u32, 2, 3];
        let id = s
            .add_session(&prompt, SamplingStrategy::Greedy, None, 5)
            .unwrap();
        let out = s.run_to_completion();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, id);
        assert_eq!(
            out[0].1.len(),
            prompt.len() + 5,
            "prompt + 5 generated tokens"
        );
    }

    /// eos stops a paged session right after it is emitted (budget not exhausted).
    #[test]
    fn paged_scheduler_stops_on_eos() {
        let model = tiny_model(42);
        let prompt = [1u32, 2, 3];
        // Learn the first greedy token, then use it as eos.
        let mut s0 =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        s0.add_session(&prompt, SamplingStrategy::Greedy, None, 5)
            .unwrap();
        let full = s0.run_to_completion()[0].1.clone();
        let first_gen = full[prompt.len()];

        let mut s =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        s.add_session(&prompt, SamplingStrategy::Greedy, Some(first_gen), 5)
            .unwrap();
        let out = s.run_to_completion()[0].1.clone();
        assert_eq!(
            out.len(),
            prompt.len() + 1,
            "stops right after emitting eos"
        );
        assert_eq!(*out.last().unwrap(), first_gen);
    }

    /// THE PS3 GATE — shared-pool isolation (T1). Two sessions decoding over ONE
    /// shared DeviceKvPool produce token-for-token the SAME output as each run
    /// alone in its own pool. Proves per-session block handles never cross-
    /// contaminate through the shared physical pool.
    #[test]
    fn paged_scheduler_two_sessions_isolated_over_shared_pool() {
        let model = tiny_model(9999);
        let a = [1u32, 2, 3];
        let b = [7u32, 4, 9, 2];
        let max_new = 6;

        // Each ALONE (own scheduler, own pool).
        let mut sa =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        sa.add_session(&a, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let a_alone = sa.run_to_completion()[0].1.clone();
        let mut sb =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        sb.add_session(&b, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let b_alone = sb.run_to_completion()[0].1.clone();

        // TOGETHER (one scheduler, SHARED pool).
        let mut both =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        let ida = both
            .add_session(&a, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let idb = both
            .add_session(&b, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let out = both.run_to_completion();
        let a_shared = out.iter().find(|(id, _)| *id == ida).unwrap().1.clone();
        let b_shared = out.iter().find(|(id, _)| *id == idb).unwrap().1.clone();

        assert_eq!(a_shared, a_alone, "A unaffected by sharing the pool with B");
        assert_eq!(b_shared, b_alone, "B unaffected by sharing the pool with A");
    }

    /// Reaping a finished paged session returns all its blocks to the shared pool.
    #[test]
    fn paged_scheduler_reap_frees_pool_blocks() {
        let model = tiny_model(42);
        let mut s =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        let free0 = s.kv_free_blocks();
        s.add_session(&[1u32, 2, 3], SamplingStrategy::Greedy, None, 5)
            .unwrap();
        s.run_to_completion();
        assert!(
            s.kv_free_blocks() < free0,
            "a decoded session holds pool blocks"
        );
        let reaped = s.reap_finished();
        assert_eq!(reaped.len(), 1);
        assert_eq!(s.kv_free_blocks(), free0, "reap returns every block");
        assert_eq!(s.session_count(), 0);
    }

    /// End-to-end tie: the paged scheduler's greedy output matches the contiguous
    /// `generate_with_kv_context` oracle (PS2 showed the paged forward is ε-close
    /// per step; greedy argmax is stable at that closeness on this tiny model).
    #[test]
    fn paged_scheduler_greedy_matches_contiguous_generate() {
        let model = tiny_model(9999);
        let prompt = [1u32, 2, 3];
        let max_new = 5;
        let contig = model
            .generate_with_kv_context(
                &prompt,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &Device::cpu(),
                DType::F32,
            )
            .unwrap();
        let mut s =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        s.add_session(&prompt, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let paged = s.run_to_completion()[0].1.clone();
        assert_eq!(
            paged, contig,
            "paged scheduler greedy == contiguous generate oracle"
        );
    }

    /// **Plan reuse is the DEFAULT.** A consumer that never calls
    /// [`set_plan`](PagedSessionScheduler::set_plan) gets the plan-once path, and
    /// gets it *routed*, not merely configured. Two teeth, because "the field says
    /// `PlanOnce`" and "the persistent path actually ran" are different claims and
    /// only the second one is worth anything:
    ///
    /// 1. the freshly-built scheduler reports [`PagedDecodePlan::PlanOnce`];
    /// 2. a default-constructed scheduler that decodes to completion holds a
    ///    REBOUND decode plan (`session_realize_count >= 1`).
    ///
    /// Tooth 2 is the one that catches the regression that matters. Before this
    /// default flipped, the correct, test-gated persistent seam shipped OFF and
    /// cost measured consumers **29.7×** on the paged route (8,192.0 → 275.8
    /// ms/token, nsys, 2026-08-01) — a seam nobody was routed to. If a future
    /// refactor leaves `plan()` reading `PlanOnce` while `decode_one` quietly takes
    /// the replan branch, tooth 1 still passes and tooth 2 fails.
    #[test]
    fn paged_scheduler_defaults_to_plan_once() {
        let model = tiny_model(42);
        let prompt = [1u32, 2, 3];
        let max_new = 5;

        let mut s =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        assert_eq!(
            s.plan(),
            PagedDecodePlan::PlanOnce,
            "driver default is plan reuse — a consumer that never opts in gets the fast path"
        );

        // ...and the default is ROUTED: decode through it and the held plan rebinds.
        let id = s
            .add_session(&prompt, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let _ = s.run_to_completion();
        let rc = s
            .session_realize_count(id)
            .expect("default-configured session holds a decode plan after decoding");
        assert!(
            rc >= 1,
            "default path rebound the held decode plan per token (got {rc})"
        );
    }

    /// Max absolute per-element difference between two captured logit streams
    /// (from [`PagedSessionScheduler::take_captured_logits`]) — the discriminating
    /// metric for KV-state-dependent behavior (see `tiny_model`'s note: token
    /// equality is vacuous here). Asserts the streams are the same shape.
    fn logits_maxdiff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
        assert_eq!(a.len(), b.len(), "same number of sampled steps");
        let mut m = 0.0f32;
        for (x, y) in a.iter().zip(b) {
            assert_eq!(x.len(), y.len(), "logit vectors are the same width");
            for (p, q) in x.iter().zip(y) {
                m = m.max((p - q).abs());
            }
        }
        m
    }

    /// Run a prompt from scratch through a private scheduler, returning its
    /// per-step captured logits — the from-scratch reference for a prefix-sharing
    /// parity check.
    fn from_scratch_logits(
        model: &LlamaModel,
        budget: KvBudget,
        prompt: &[u32],
        max_new: usize,
    ) -> Vec<Vec<f32>> {
        let mut s = PagedSessionScheduler::new(model, budget, DType::F32, &Device::cpu()).unwrap();
        let id = s
            .add_session(prompt, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        s.capture_logits(id).unwrap();
        s.run_to_completion();
        s.take_captured_logits(id).unwrap()
    }

    /// TASK 4 — the SCHEDULER-level prefix-sharing correctness anchor (the twin
    /// of Task 1's model-level `prefix_shared_session_decodes_like_from_scratch`).
    /// A session admitted through the reference caller
    /// [`PagedSessionScheduler::add_session_sharing_prefix`] — which splices a
    /// registered [`PrefixId`] then prefills ONLY its unique suffix — computes,
    /// step for step, the SAME logits as a from-scratch session that prefilled the
    /// WHOLE prompt. This proves the suffix-only prefill lands the donor-computed
    /// prefix KV at the correct absolute positions (rung-1: same positions), so
    /// the sharer never recomputes the prefix yet decodes identically.
    ///
    /// The oracle is LOGITS, not sampled tokens, and that choice has teeth: a
    /// mis-positioned prefill (re-feeding the whole prompt onto the spliced
    /// prefix) perturbs the logits by only ~1.7e-3, which is BELOW the token
    /// threshold — the sampled stream is byte-identical under both greedy argmax
    /// (a sticky constant on `tiny_model`, see its note) and seeded temperature.
    /// Only a per-step logit comparison catches it. (Mutation-verified: setting
    /// `prefill_start = 0` makes `maxdiff` jump from 0 to ~1.7e-3 and this fails.)
    #[test]
    fn paged_scheduler_prefix_shared_matches_from_scratch() {
        let model = tiny_model(9999);
        // block_size 4: a 2-whole-block shared prefix, then a unique suffix.
        let budget = KvBudget {
            block_size: 4,
            num_blocks: 64,
        };
        let prefix = [1u32, 2, 3, 4, 5, 6, 7, 8]; // 2 full blocks at bs=4
        let suffix = [9u32, 10, 11];
        let full: Vec<u32> = prefix.iter().chain(suffix.iter()).copied().collect();
        let max_new = 6;

        // From scratch: the whole prompt through the normal admission path,
        // capturing every step's pre-sample logits.
        let mut scratch =
            PagedSessionScheduler::new(&model, budget, DType::F32, &Device::cpu()).unwrap();
        let scratch_id = scratch
            .add_session(&full, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        scratch.capture_logits(scratch_id).unwrap();
        let scratch_tokens = scratch
            .run_to_completion()
            .into_iter()
            .find(|(id, _)| *id == scratch_id)
            .unwrap()
            .1;
        let scratch_logits = scratch.take_captured_logits(scratch_id).unwrap();

        // Shared: a throwaway donor computes the prefix ONCE, we register its two
        // full blocks, reap the donor (the prefix owner keeps them alive), then a
        // sharer splices the prefix and prefills only its unique suffix.
        let mut sched =
            PagedSessionScheduler::new(&model, budget, DType::F32, &Device::cpu()).unwrap();
        // Donor prompt = the prefix, max_new = 1: one `step()` prefills the two
        // full blocks (filled == 8, block-aligned) and finishes — it never feeds
        // the sampled token back, so no partial third block is written.
        let donor = sched
            .add_session(&prefix, SamplingStrategy::Greedy, None, 1)
            .unwrap();
        sched.step();
        let pid = sched.register_prefix(donor, 2).unwrap();
        sched.reap_finished(); // donor gone; owner still pins the 2 blocks

        let shared_id = sched
            .add_session_sharing_prefix(pid, &full, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        // The splice made the whole prefix resident before any prefill runs.
        let sharer_handle = sched.sessions.last().unwrap().handle;
        assert_eq!(
            sched.pool.core().filled_tokens(sharer_handle),
            Some(prefix.len()),
            "splice_prefix_from made the shared prefix resident in the sharer",
        );
        sched.capture_logits(shared_id).unwrap();
        let shared_tokens = sched
            .run_to_completion()
            .into_iter()
            .find(|(id, _)| *id == shared_id)
            .unwrap()
            .1;
        let shared_logits = sched.take_captured_logits(shared_id).unwrap();

        // Product-level sanity: same token stream (both include the full prompt).
        assert_eq!(
            shared_tokens, scratch_tokens,
            "prefix-shared token stream == from-scratch token stream",
        );
        // The discriminating oracle: byte-identical per-step logits. A wiring bug
        // that re-fed or mis-positioned the prefix would diverge here (~1.7e-3)
        // while leaving the token stream above untouched.
        let maxdiff = logits_maxdiff(&shared_logits, &scratch_logits);
        assert_eq!(
            maxdiff, 0.0,
            "prefix-shared per-step logits are byte-identical to from-scratch (maxdiff {maxdiff})",
        );
    }

    /// TASK 5 (adversarial) — the actual serving value: TWO sessions splice the
    /// SAME registered prefix and decode DIFFERENT suffixes concurrently over one
    /// pool, and each is byte-identical (per-step logits) to its own from-scratch
    /// run. This is the multi-tenant isolation claim under sharing: the shared
    /// prefix blocks are read by both, and each sharer's first suffix write lands
    /// on a FRESH block (rung-1: prefix fully filled), so neither corrupts the
    /// shared prefix nor the other's suffix.
    #[test]
    fn two_sharers_of_one_prefix_decode_independently() {
        let model = tiny_model(9999);
        let budget = KvBudget {
            block_size: 4,
            num_blocks: 64,
        };
        let prefix = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let full_a: Vec<u32> = prefix
            .iter()
            .chain([9u32, 10, 11].iter())
            .copied()
            .collect();
        let full_b: Vec<u32> = prefix
            .iter()
            .chain([12u32, 13, 14, 15].iter())
            .copied()
            .collect();
        let max_new = 5;
        let ref_a = from_scratch_logits(&model, budget, &full_a, max_new);
        let ref_b = from_scratch_logits(&model, budget, &full_b, max_new);

        let mut sched =
            PagedSessionScheduler::new(&model, budget, DType::F32, &Device::cpu()).unwrap();
        let donor = sched
            .add_session(&prefix, SamplingStrategy::Greedy, None, 1)
            .unwrap();
        sched.step();
        let pid = sched.register_prefix(donor, 2).unwrap();
        sched.reap_finished();

        let a = sched
            .add_session_sharing_prefix(pid, &full_a, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let b = sched
            .add_session_sharing_prefix(pid, &full_b, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        sched.capture_logits(a).unwrap();
        sched.capture_logits(b).unwrap();
        sched.run_to_completion();
        let la = sched.take_captured_logits(a).unwrap();
        let lb = sched.take_captured_logits(b).unwrap();

        assert_eq!(
            logits_maxdiff(&la, &ref_a),
            0.0,
            "sharer A decodes byte-identically to its own from-scratch run",
        );
        assert_eq!(
            logits_maxdiff(&lb, &ref_b),
            0.0,
            "sharer B decodes byte-identically to its own from-scratch run (not contaminated by A)",
        );
    }

    /// TASK 5 (adversarial) — releasing the prefix OWNER while a sharer is still
    /// live (and about to decode over the shared blocks) must not pull the rug:
    /// the sharer's own splice refcount keeps the blocks alive, so it decodes
    /// byte-identically to from-scratch. A refcount bug that freed the blocks on
    /// owner-release would make the sharer read reclaimed memory and diverge.
    #[test]
    fn prefix_owner_release_while_sharer_live_keeps_it_correct() {
        let model = tiny_model(9999);
        let budget = KvBudget {
            block_size: 4,
            num_blocks: 64,
        };
        let prefix = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let full: Vec<u32> = prefix
            .iter()
            .chain([9u32, 10, 11].iter())
            .copied()
            .collect();
        let max_new = 5;
        let reference = from_scratch_logits(&model, budget, &full, max_new);

        let mut sched =
            PagedSessionScheduler::new(&model, budget, DType::F32, &Device::cpu()).unwrap();
        let donor = sched
            .add_session(&prefix, SamplingStrategy::Greedy, None, 1)
            .unwrap();
        sched.step();
        let pid = sched.register_prefix(donor, 2).unwrap();
        sched.reap_finished();
        let sharer = sched
            .add_session_sharing_prefix(pid, &full, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        // Drop the owner NOW, before the sharer has decoded a single token.
        sched.release_prefix(pid).unwrap();
        sched.capture_logits(sharer).unwrap();
        sched.run_to_completion();
        let got = sched.take_captured_logits(sharer).unwrap();

        assert_eq!(
            logits_maxdiff(&got, &reference),
            0.0,
            "sharer decodes byte-identically after its prefix owner is released mid-flight",
        );
    }

    /// TASK 5 (adversarial) — a refused `add_session_sharing_prefix` leaks
    /// nothing: no session is admitted, no blocks are consumed, AND the whole run
    /// later unwinds to a fully-free pool. That last check is the teeth — a
    /// refusal that failed to roll back its just-opened handle would pin the
    /// shared blocks at a positive refcount forever, so reap + release could never
    /// return the pool to `num_blocks`.
    #[test]
    fn add_session_sharing_prefix_refusal_leaks_nothing() {
        let model = tiny_model(9999);
        let budget = KvBudget {
            block_size: 4,
            num_blocks: 64,
        };
        let prefix = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let full: Vec<u32> = prefix
            .iter()
            .chain([9u32, 10, 11].iter())
            .copied()
            .collect();

        let mut sched =
            PagedSessionScheduler::new(&model, budget, DType::F32, &Device::cpu()).unwrap();
        let donor = sched
            .add_session(&prefix, SamplingStrategy::Greedy, None, 1)
            .unwrap();
        sched.step();
        let pid = sched.register_prefix(donor, 2).unwrap();
        sched.reap_finished();

        let free_with_prefix = sched.kv_free_blocks();
        let sessions_before = sched.session_count();

        // Refusal: the prompt does not extend past the shared prefix (this path
        // splices SUCCESSFULLY, then must roll the handle back). Err, untouched.
        assert!(
            sched
                .add_session_sharing_prefix(pid, &prefix, SamplingStrategy::Greedy, None, 3)
                .is_err(),
            "a prompt that does not extend past the prefix is refused",
        );
        assert_eq!(
            sched.session_count(),
            sessions_before,
            "refusal admits no session"
        );
        assert_eq!(
            sched.kv_free_blocks(),
            free_with_prefix,
            "refusal consumes no blocks"
        );

        // A legitimate sharer still works after the refusal, and the whole thing
        // unwinds to an entirely free pool — impossible if the refusal leaked its
        // rolled-back handle (which would pin the shared blocks).
        let ok = sched
            .add_session_sharing_prefix(pid, &full, SamplingStrategy::Greedy, None, 3)
            .unwrap();
        sched.run_to_completion();
        let reaped = sched.reap_finished();
        assert!(
            reaped.iter().any(|(id, _)| *id == ok),
            "the legit sharer ran and reaped"
        );
        sched.release_prefix(pid).unwrap();
        assert_eq!(
            sched.kv_free_blocks(),
            budget.num_blocks,
            "after reap + release every block is free — the refusal leaked no handle",
        );
    }

    /// Plan-once wiring (Task 5): opting a paged session into
    /// [`PagedDecodePlan::PlanOnce`] via [`PagedSessionScheduler::set_plan`]
    /// produces output BYTE-IDENTICAL to the `Replan` driver — the flag
    /// changes only WHEN the decode graph is optimized (once vs per token), never
    /// the numbers — AND the session's held decode plan is actually REBOUND per
    /// decode token (`session_realize_count` > 0), direct evidence the persistent
    /// path ran through the driver rather than being silently bypassed. Single
    /// session, serial arm (the B=1 measured case). The two guards are
    /// complementary teeth: byte-identity catches a mis-wire that changes results;
    /// the rebind-count catches a mis-wire where the flag is set but the plan-once
    /// path never actually runs (e.g. the persistent call handed a throwaway
    /// `&mut None`, or `decode_one` left on the plain path) — which would read as
    /// "no speedup" and wrongly retire the feature.
    ///
    /// **Scope honesty on the word "byte-exact": this arm compares TOKEN
    /// STREAMS, and token equality is a coarse oracle for anything
    /// KV-state-dependent.** A peer measured a 1.67e-3 max logit perturbation
    /// surviving *both* greedy argmax and seeded multinomial sampling — i.e. a
    /// real numerical divergence that a token-stream comparison cannot see, at
    /// either sampler. So this test's genuine teeth are `session_realize_count`
    /// (wiring) rather than the token comparison. The **logit-level** plan-once
    /// parity — the claim that reusing the plan does not change the numbers —
    /// is gated a layer down, at the model, by
    /// `lazy.rs::plan_once_second_token_reuses_graph`, which compares the
    /// persistent tokens' actual logits against the re-planning
    /// `forward_paged_step` reference. Read the two together; neither is
    /// sufficient alone.
    #[test]
    fn paged_scheduler_plan_once_matches_replan_byte_exact() {
        let model = tiny_model(42);
        let prompt = [1u32, 2, 3];
        let max_new = 5;

        // Reference arm: `Replan` — re-plans every token. Set EXPLICITLY, never
        // inherited from the driver default: the default is now `PlanOnce`, and an
        // arm that leans on the default would silently become a PlanOnce-vs-PlanOnce
        // comparison the day the default moved — a parity test that passes by
        // comparing a thing to itself.
        let mut s_replan =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        s_replan.set_plan(PagedDecodePlan::Replan);
        assert_eq!(
            s_replan.plan(),
            PagedDecodePlan::Replan,
            "reference arm is Replan"
        );
        s_replan
            .add_session(&prompt, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let replan_out = s_replan.run_to_completion()[0].1.clone();

        // Plan-once arm: build + optimize the decode graph once, rebind per token.
        let mut s_plan =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        s_plan.set_plan(PagedDecodePlan::PlanOnce);
        let id = s_plan
            .add_session(&prompt, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let plan_out = s_plan.run_to_completion()[0].1.clone();

        assert_eq!(
            plan_out, replan_out,
            "plan-once output == replan output (byte-exact)"
        );

        // The held plan was REUSED: the first decode token builds the plan
        // (realize 0), each subsequent decode token rebinds it. With prompt 3 +
        // max_new 5, the first new token comes from the prefill sample and the
        // remaining 4 from decode forwards — 1 build + 3 rebinds ⇒ rebind count 3.
        // Assert ≥ 1 (robust to sampling/budget arithmetic; still fails hard if
        // the persistent path never ran — `session_realize_count` is `None`
        // without a held plan, so `.expect` panics).
        let rc = s_plan
            .session_realize_count(id)
            .expect("plan-once session holds a decode plan after decoding");
        assert!(
            rc >= 1,
            "plan-once path rebound the held decode plan per token (got {rc})"
        );
    }

    /// C-3 ON THE LIVE PATH (PS4). Evicting a decoding paged session frees its
    /// blocks (the pressure valve for PS3's optimistic admission), a step while
    /// suspended advances nothing, and restoring it resumes BYTE-EXACT — the same
    /// final tokens as an uninterrupted run (DeviceKvPool.evict/restore round-
    /// trips the block bytes exactly, and the session's rng/tokens/budget are
    /// preserved across the suspension).
    #[test]
    fn paged_scheduler_evict_restore_resumes_byte_exact() {
        let model = tiny_model(9999);
        let prompt = [1u32, 2, 3];
        let max_new = 6;

        // Reference: uninterrupted run.
        let mut s0 =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        s0.add_session(&prompt, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let reference = s0.run_to_completion()[0].1.clone();

        // Evict mid-decode, restore, finish — final tokens must match.
        let mut s =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        let id = s
            .add_session(&prompt, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        s.step(); // prefill + 1st decode token
        s.step(); // 2nd decode token
        let free_decoding = s.kv_free_blocks();

        s.evict_session(id).unwrap();
        assert!(s.is_suspended(id), "suspended after evict");
        assert!(
            s.kv_free_blocks() > free_decoding,
            "evict returned the session's blocks to the pool",
        );

        s.step(); // a step while suspended advances nothing
        assert!(s.is_suspended(id), "suspended session is skipped by step");

        s.restore_session(id).unwrap();
        assert!(!s.is_suspended(id), "resumed after restore");
        assert_eq!(
            s.kv_free_blocks(),
            free_decoding,
            "restore re-took exactly the freed blocks",
        );

        let out = s.run_to_completion()[0].1.clone();
        assert_eq!(
            out, reference,
            "evict→restore resumes byte-exact (same tokens as uninterrupted)"
        );
    }

    /// The pressure-valve flow end to end: evict a session to free the pool, let
    /// another session run in the freed room, then restore the first and finish
    /// it. `run_to_completion` returns with a suspended session still live rather
    /// than spinning.
    #[test]
    fn paged_scheduler_evict_lets_another_run_then_restore_finishes() {
        let model = tiny_model(9999);
        let mut s = PagedSessionScheduler::new(
            &model,
            KvBudget {
                block_size: 4,
                num_blocks: 16,
            },
            DType::F32,
            &Device::cpu(),
        )
        .unwrap();
        let a = s
            .add_session(&[1u32, 2, 3], SamplingStrategy::Greedy, None, 4)
            .unwrap();
        s.step();
        let free_with_a = s.kv_free_blocks();
        s.evict_session(a).unwrap();
        assert!(s.kv_free_blocks() > free_with_a, "A's blocks freed");

        // B runs to completion in the pool while A is suspended; run_to_completion
        // returns (does not spin) with A still suspended.
        let b = s
            .add_session(&[5u32, 6], SamplingStrategy::Greedy, None, 3)
            .unwrap();
        let out1 = s.run_to_completion();
        let b_len = out1.iter().find(|(id, _)| *id == b).map(|(_, t)| t.len());
        assert_eq!(
            b_len,
            Some(2 + 3),
            "B (prompt 2 + 3 generated) finished in the freed pool"
        );
        assert!(s.is_suspended(a), "A stayed suspended while B ran");

        // Restore A and finish it.
        s.restore_session(a).unwrap();
        let out2 = s.run_to_completion();
        let a_len = out2.iter().find(|(id, _)| *id == a).map(|(_, t)| t.len());
        assert_eq!(
            a_len,
            Some(3 + 4),
            "A resumed and finished (prompt 3 + 4 generated)"
        );
    }

    /// PS4a scheduler arm: `step_batched` (batches same-position ready sessions
    /// via Op::PagedAttn at B=K) produces token-identical output to serial `step`,
    /// and the batched arm actually fires (`used_batched_arm`).
    #[test]
    fn paged_scheduler_batched_arm_matches_serial() {
        let model = tiny_model(9999);
        let a = [1u32, 2, 3];
        let b = [4u32, 5, 6]; // equal length → the two stay position-uniform in lockstep
        let max_new = 5;

        // Serial reference.
        let mut ss =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        let ida = ss
            .add_session(&a, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let idb = ss
            .add_session(&b, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let serial = ss.run_to_completion();
        let sa = serial.iter().find(|(id, _)| *id == ida).unwrap().1.clone();
        let sb = serial.iter().find(|(id, _)| *id == idb).unwrap().1.clone();

        // Batched: loop step_batched to completion.
        let mut bs =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        let ida2 = bs
            .add_session(&a, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let idb2 = bs
            .add_session(&b, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let mut fired = false;
        while bs
            .sessions
            .iter()
            .any(|s| s.phase != SessionPhase::Finished)
        {
            let r = bs.step_batched(8);
            fired |= r.used_batched_arm;
        }
        assert!(
            fired,
            "the batched arm actually ran (two same-position sessions)"
        );
        let out: Vec<_> = bs
            .sessions
            .iter()
            .map(|s| (s.id, s.tokens.clone()))
            .collect();
        let ba = out.iter().find(|(id, _)| *id == ida2).unwrap().1.clone();
        let bb = out.iter().find(|(id, _)| *id == idb2).unwrap().1.clone();
        assert_eq!(ba, sa, "batched arm A == serial A (token-identical)");
        assert_eq!(bb, sb, "batched arm B == serial B (token-identical)");
    }

    #[test]
    fn t1_no_cross_session_contamination() {
        let model = tiny_model(9999);
        let prompt_a = [1u32, 2, 3];
        let prompt_b = [7u32, 4, 9, 2];
        let max_new = 6;

        // Standalone oracles.
        let solo_a = model
            .generate_with_kv_context(
                &prompt_a,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &Device::cpu(),
                DType::F32,
            )
            .unwrap();
        let solo_b = model
            .generate_with_kv_context(
                &prompt_b,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &Device::cpu(),
                DType::F32,
            )
            .unwrap();

        // K=2 scheduled together.
        let mut sched = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::RoundRobin,
            test_budget(),
        )
        .unwrap();
        let ida = sched
            .add_session(&prompt_a, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let idb = sched
            .add_session(&prompt_b, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let out = sched.run_to_completion().unwrap();

        let get = |id: SessionId| {
            out.iter()
                .find(|(i, _)| *i == id)
                .map(|(_, t)| t.clone())
                .unwrap()
        };
        assert_eq!(get(ida), solo_a, "session A contaminated by B");
        assert_eq!(get(idb), solo_b, "session B contaminated by A");
    }

    #[test]
    fn t2_interleave_order_invariance() {
        let model = tiny_model(9999);
        let (pa, pb, max_new) = ([1u32, 2, 3], [5u32, 6], 6);

        // Round-robin (both added, then run together).
        let mut rr = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::RoundRobin,
            test_budget(),
        )
        .unwrap();
        let a1 = rr
            .add_session(&pa, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let b1 = rr
            .add_session(&pb, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let out_rr = rr.run_to_completion().unwrap();

        // One-then-the-other: A alone to completion, then B alone.
        let mut s_a = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::RoundRobin,
            test_budget(),
        )
        .unwrap();
        s_a.add_session(&pa, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let just_a = s_a.run_to_completion().unwrap();
        let mut s_b = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::RoundRobin,
            test_budget(),
        )
        .unwrap();
        s_b.add_session(&pb, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let just_b = s_b.run_to_completion().unwrap();

        let get = |o: &Vec<(SessionId, Vec<u32>)>, id: SessionId| {
            o.iter().find(|(i, _)| *i == id).unwrap().1.clone()
        };
        assert_eq!(get(&out_rr, a1), just_a[0].1);
        assert_eq!(get(&out_rr, b1), just_b[0].1);
    }

    #[test]
    fn t3_per_session_rng_independence() {
        let model = tiny_model(9999);
        let prompt = [1u32, 2, 3];
        let max_new = 8;

        // Same prompt, DIFFERENT seeds → different streams.
        let mut sched = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::RoundRobin,
            test_budget(),
        )
        .unwrap();
        let id1 = sched
            .add_session(
                &prompt,
                SamplingStrategy::Temperature { temp: 1.0, seed: 1 },
                None,
                max_new,
            )
            .unwrap();
        let id2 = sched
            .add_session(
                &prompt,
                SamplingStrategy::Temperature { temp: 1.0, seed: 2 },
                None,
                max_new,
            )
            .unwrap();
        let out = sched.run_to_completion().unwrap();
        let g = |id: SessionId| out.iter().find(|(i, _)| *i == id).unwrap().1.clone();
        assert_ne!(g(id1), g(id2), "different seeds must diverge");

        // Same seed as a standalone Temperature run → identical.
        let solo = model
            .generate_with_kv_context(
                &prompt,
                max_new,
                SamplingStrategy::Temperature { temp: 1.0, seed: 1 },
                None,
                &Device::cpu(),
                DType::F32,
            )
            .unwrap();
        assert_eq!(g(id1), solo, "seed 1 must match its standalone run");
    }

    #[test]
    fn t4_session_isolation_on_error() {
        let model = tiny_model(9999);
        let good = [1u32, 2, 3];
        let mut sched = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::RoundRobin,
            test_budget(),
        )
        .unwrap();
        let bad_id = sched.add_poisoned_session_for_test(&[4u32, 5], 5).unwrap();
        let good_id = sched
            .add_session(&good, SamplingStrategy::Greedy, None, 5)
            .unwrap();

        // First step: the poisoned session errors, the good one advances. No panic.
        let r0 = sched.step().unwrap();
        assert!(
            r0.errored.iter().any(|(id, _)| *id == bad_id),
            "poisoned session must be reported errored"
        );

        let out = sched.run_to_completion().unwrap();
        let solo_good = model
            .generate_with_kv_context(
                &good,
                5,
                SamplingStrategy::Greedy,
                None,
                &Device::cpu(),
                DType::F32,
            )
            .unwrap();
        let g = out.iter().find(|(i, _)| *i == good_id).unwrap().1.clone();
        assert_eq!(g, solo_good, "the healthy session must complete unaffected");
    }

    #[test]
    fn t7_mid_run_add_prefills_and_joins() {
        let model = tiny_model(9999);
        let pa = [1u32, 2, 3];
        let pb = [8u32, 1];
        let max_new = 6;

        let mut sched = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::RoundRobin,
            test_budget(),
        )
        .unwrap();
        let ida = sched
            .add_session(&pa, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        // Advance A alone for two steps.
        sched.step().unwrap();
        sched.step().unwrap();
        // Now add B mid-run.
        let idb = sched
            .add_session(&pb, SamplingStrategy::Greedy, None, max_new)
            .unwrap();
        let out = sched.run_to_completion().unwrap();

        let solo_a = model
            .generate_with_kv_context(
                &pa,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &Device::cpu(),
                DType::F32,
            )
            .unwrap();
        let solo_b = model
            .generate_with_kv_context(
                &pb,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &Device::cpu(),
                DType::F32,
            )
            .unwrap();
        let g = |id: SessionId| out.iter().find(|(i, _)| *i == id).unwrap().1.clone();
        assert_eq!(g(ida), solo_a, "A unaffected by mid-run B");
        assert_eq!(g(idb), solo_b, "B prefills correctly mid-run");
    }

    #[test]
    fn t6_uniformity_gate_rejects_ragged_cached_len() {
        let d = |cl: usize| BatchDescriptor {
            cached_len: cl,
            max_seq_len: 64,
            n_layers: 2,
            cache_dtype: DType::F32,
        };
        assert!(batch_uniform(&[d(3), d(3)])); // equal → batchable
        assert!(!batch_uniform(&[d(3), d(4)])); // ragged → not
        assert!(!batch_uniform(&[d(3)])); // <2 sessions → not
    }

    #[test]
    fn t6_batched_policy_falls_back_to_serial_equals_roundrobin() {
        // With Task 7's stub always returning NotBatchable, a Batched policy
        // must produce byte-identical output to RoundRobin (pure serial).
        let model = tiny_model(9999);
        let (pa, pb, max_new) = ([1u32, 2, 3], [5u32, 6, 7], 6);
        let run = |policy| {
            let mut s =
                SessionScheduler::new(&model, Device::cpu(), DType::F32, policy, test_budget())
                    .unwrap();
            let a = s
                .add_session(&pa, SamplingStrategy::Greedy, None, max_new)
                .unwrap();
            let b = s
                .add_session(&pb, SamplingStrategy::Greedy, None, max_new)
                .unwrap();
            (a, b, s.run_to_completion().unwrap())
        };
        let (a1, b1, rr) = run(SchedulePolicy::RoundRobin);
        let (a2, b2, ba) = run(SchedulePolicy::Batched { max_batch: 4 });
        let g = |o: &Vec<(SessionId, Vec<u32>)>, id: SessionId| {
            o.iter().find(|(i, _)| *i == id).unwrap().1.clone()
        };
        assert_eq!(g(&rr, a1), g(&ba, a2));
        assert_eq!(g(&rr, b1), g(&ba, b2));
    }

    #[test]
    fn t5_cpu_batched_step_equals_serial_step() {
        // Two sessions, SAME prompt length, prefilled to equal cached_len,
        // then ONE batched decode step must equal one serial step each.
        let model = tiny_model(9999);
        let pa = [1u32, 2, 3];
        let pb = [4u32, 5, 6];

        // Serial oracle: prefill each, take one decode step, record logits.
        let serial_logits = |prompt: &[u32]| -> Vec<f32> {
            use fuel::inference_context::{InferenceContext, KvCache};
            let cfg = &model.config;
            let msl = prompt.len() + 2;
            let mut cache = KvCache::with_capacity(
                cfg.n_layers,
                cfg.n_kv_heads,
                cfg.head_dim,
                msl,
                DType::F32,
                &Device::cpu(),
            )
            .unwrap();
            let mut ctx = InferenceContext::new(Device::cpu());
            let mut sess = None;
            let pre = model
                .forward_with_kv_context_persistent(prompt, &mut cache, &mut ctx, &mut sess)
                .unwrap();
            let next = fuel::lazy::sample_logits(&pre, SamplingStrategy::Greedy, &mut 0u64);
            model
                .forward_with_kv_context_persistent(&[next], &mut cache, &mut ctx, &mut sess)
                .unwrap()
        };
        let sa = serial_logits(&pa);
        let sb = serial_logits(&pb);

        // Batched: build two SessionStates prefilled + first token sampled,
        // at equal cached_len, then one try_batched_step.
        let mut states: Vec<SessionState> = Vec::new();
        for (id, p) in [(0u64, &pa[..]), (1u64, &pb[..])] {
            let mut s = SessionState::new(
                SessionId(id),
                ModelDims {
                    n_layers: model.config.n_layers,
                    n_kv_heads: model.config.n_kv_heads,
                    head_dim: model.config.head_dim,
                },
                p,
                SamplingStrategy::Greedy,
                None,
                2,
                &Device::cpu(),
                DType::F32,
            )
            .unwrap();
            // Prefill + sample first token so cached_len == prompt.len() and
            // last token is set (mirrors scheduler.step prefill pass).
            s.last_logits = Some(
                model
                    .forward_with_kv_context_persistent(
                        &s.tokens.clone(),
                        &mut s.cache,
                        &mut s.ctx,
                        &mut s.session,
                    )
                    .unwrap(),
            );
            s.sample_and_append().unwrap();
            states.push(s);
        }
        assert_eq!(
            states[0].cache.cached_len, states[1].cache.cached_len,
            "equal cached_len (uniform)"
        );
        let mut refs: Vec<&mut SessionState> = states.iter_mut().collect();
        let outcome =
            BatchedDecode::try_batched_step(&model, &Device::cpu(), DType::F32, &mut refs).unwrap();
        match outcome {
            BatchOutcome::Advanced(rows) => {
                assert_eq!(rows.len(), 2);
                // Batched decode step logits == serial decode step logits (f32, ε-tol).
                let close = |a: &[f32], b: &[f32]| {
                    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
                };
                assert!(close(&rows[0], &sa), "batched row 0 != serial A");
                assert!(close(&rows[1], &sb), "batched row 1 != serial B");
            }
            BatchOutcome::NotBatchable => panic!("uniform sessions must batch"),
        }
    }

    #[test]
    #[ignore = "live-GPU: RTX 4070, run locally after CUDA build; one live suite at a time"]
    fn t5_gpu_batched_flash_equals_serial_bf16() {
        // Same structure as t5_cpu, but Device::cuda(0) + DType::BF16 so the
        // optimizer offers the flash_decoding batch arm. bf16 weights required
        // (mirror lazy.rs generate_tests::make_tiny_weights_bf16).
        //
        // Assert batched rows == serial rows within a sabotage-calibrated ε
        // (batched GEMM reduction order may differ from serial — see
        // [[sabotage-test-calibration]]). Start ε = 5e-3, tighten to the
        // measured serial-vs-serial bf16 drift floor. Confirm the flash arm is
        // actually PICKED (temporary eprintln of the chosen arm) and that a
        // KV-perturbation sabotage makes the test FAIL (a passing sabotage run
        // is invalid without confirmed recompilation).
        use fuel::lazy::{LayerWeights, LlamaConfig, LlamaModel, LlamaWeights, WeightStorage};

        fn bf16_weights(cfg: &LlamaConfig) -> LlamaWeights {
            // f32 tiny weights → BF16 for every WeightStorage matrix (embedding
            // + norm gains stay f32, per make_tiny_weights_bf16's frozen seams).
            let mut s: u32 = 9999;
            let mut next = || {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
            };
            let mut vec_of = |n: usize| -> std::sync::Arc<[f32]> {
                std::sync::Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
            };
            let to_bf16 = |a: std::sync::Arc<[f32]>| -> WeightStorage {
                WeightStorage::BF16(std::sync::Arc::from(
                    a.iter()
                        .map(|&v| half::bf16::from_f32(v))
                        .collect::<Vec<_>>(),
                ))
            };
            let kv = cfg.n_kv_heads * cfg.head_dim;
            LlamaWeights {
                instance: fuel::decode_shape::ModelInstanceId::next(),
                token_embedding: vec_of(cfg.vocab_size * cfg.dim),
                layers: (0..cfg.n_layers)
                    .map(|_| LayerWeights {
                        attn_q: to_bf16(vec_of(cfg.dim * cfg.dim)),
                        attn_q_bias: None,
                        attn_k: to_bf16(vec_of(cfg.dim * kv)),
                        attn_k_bias: None,
                        attn_v: to_bf16(vec_of(cfg.dim * kv)),
                        attn_v_bias: None,
                        attn_o: to_bf16(vec_of(cfg.dim * cfg.dim)),
                        ffn_gate: to_bf16(vec_of(cfg.dim * cfg.ffn_dim)),
                        ffn_up: to_bf16(vec_of(cfg.dim * cfg.ffn_dim)),
                        ffn_down: to_bf16(vec_of(cfg.ffn_dim * cfg.dim)),
                        attn_norm_gain: std::sync::Arc::from(vec![1.0f32; cfg.dim]),
                        ffn_norm_gain: std::sync::Arc::from(vec![1.0f32; cfg.dim]),
                    })
                    .collect(),
                final_norm_gain: std::sync::Arc::from(vec![1.0f32; cfg.dim]),
                output: to_bf16(vec_of(cfg.dim * cfg.vocab_size)),
            }
        }

        let cfg = tiny_cfg();
        let model = LlamaModel {
            config: cfg.clone(),
            weights: bf16_weights(&cfg),
        };
        let dev = fuel::cuda_backend::new_device(0).expect("cuda device 0");
        let dt = DType::BF16;
        let pa = [1u32, 2, 3];
        let pb = [4u32, 5, 6];

        let serial_logits = |prompt: &[u32]| -> Vec<f32> {
            use fuel::inference_context::{InferenceContext, KvCache};
            let msl = prompt.len() + 2;
            let mut cache =
                KvCache::with_capacity(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, msl, dt, &dev)
                    .unwrap();
            let mut ctx = InferenceContext::new(dev.clone());
            let mut sess = None;
            let pre = model
                .forward_with_kv_context_persistent(prompt, &mut cache, &mut ctx, &mut sess)
                .unwrap();
            let next = fuel::lazy::sample_logits(&pre, SamplingStrategy::Greedy, &mut 0u64);
            model
                .forward_with_kv_context_persistent(&[next], &mut cache, &mut ctx, &mut sess)
                .unwrap()
        };
        let sa = serial_logits(&pa);
        let sb = serial_logits(&pb);

        let mut states: Vec<SessionState> = Vec::new();
        for (id, p) in [(0u64, &pa[..]), (1u64, &pb[..])] {
            let mut s = SessionState::new(
                SessionId(id),
                ModelDims {
                    n_layers: cfg.n_layers,
                    n_kv_heads: cfg.n_kv_heads,
                    head_dim: cfg.head_dim,
                },
                p,
                SamplingStrategy::Greedy,
                None,
                2,
                &dev,
                dt,
            )
            .unwrap();
            s.last_logits = Some(
                model
                    .forward_with_kv_context_persistent(
                        &s.tokens.clone(),
                        &mut s.cache,
                        &mut s.ctx,
                        &mut s.session,
                    )
                    .unwrap(),
            );
            s.sample_and_append().unwrap();
            states.push(s);
        }
        assert_eq!(
            states[0].cache.cached_len, states[1].cache.cached_len,
            "equal cached_len (uniform)"
        );
        let mut refs: Vec<&mut SessionState> = states.iter_mut().collect();
        let outcome = BatchedDecode::try_batched_step(&model, &dev, dt, &mut refs).unwrap();
        match outcome {
            BatchOutcome::Advanced(rows) => {
                assert_eq!(rows.len(), 2);
                let eps = 5e-3_f32;
                let close = |a: &[f32], b: &[f32]| {
                    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < eps)
                };
                assert!(
                    close(&rows[0], &sa),
                    "batched row 0 != serial A (bf16 ε={eps})"
                );
                assert!(
                    close(&rows[1], &sb),
                    "batched row 1 != serial B (bf16 ε={eps})"
                );
            }
            BatchOutcome::NotBatchable => panic!("uniform bf16 sessions must batch"),
        }
    }

    // Fix C — build_batched_decode_logits must reject a cache whose dtype
    // disagrees with the requested (scheduler) dtype, rather than silently
    // binding its bytes to a different-width placeholder (reinterpretation).
    #[test]
    fn build_batched_decode_logits_rejects_dtype_mismatch() {
        use fuel::inference_context::KvCache;
        let model = tiny_model(9999);
        let cfg = &model.config;
        let msl = 4;
        // Both caches BF16 (uniform among themselves in cached_len/msl/n_layers)
        // but the requested dtype is F32 → a byte-reinterpretation hazard.
        let mut c0 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            msl,
            DType::BF16,
            &Device::cpu(),
        )
        .unwrap();
        let mut c1 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            msl,
            DType::BF16,
            &Device::cpu(),
        )
        .unwrap();
        let mut caches: Vec<&mut KvCache> = vec![&mut c0, &mut c1];
        let last_tokens = [1u32, 2];
        let err = model
            .build_batched_decode_logits(&mut caches, &last_tokens, &Device::cpu(), DType::F32)
            .expect_err("dtype mismatch (cache BF16 vs requested F32) must Err");
        let msg = err.to_string();
        // Require the EARLY validation error (fail-fast at call time), not a
        // deep byte-width error surfaced later inside realize.
        assert!(
            msg.contains("requested dtype"),
            "expected the early dtype-validation error, got: {msg}"
        );
    }

    // Coverage E — serial↔batched transition: K=3 under Batched{max_batch:2}
    // with STAGGERED eos. A session that is overflow-serial in early steps
    // shifts INTO the batch after an earlier session finishes. Assert parity
    // (token-identical / ε-close on tokens) vs 3 standalone generate runs.
    #[test]
    fn batched_serial_transition_k3_max_batch_2_staggered_eos() {
        let model = tiny_model(9999);
        // Distinct prompts; distinct eos ids so the three sessions finish at
        // different steps (staggered), forcing the overflow-serial session to
        // migrate into the batch as earlier ones retire.
        let pa = [1u32, 2, 3];
        let pb = [4u32, 5];
        let pc = [6u32, 7, 8, 9];
        let cases: [(&[u32], Option<u32>, usize); 3] =
            [(&pa, Some(5), 8), (&pb, Some(7), 8), (&pc, None, 8)];

        let solo: Vec<Vec<u32>> = cases
            .iter()
            .map(|(p, eos, mn)| {
                model
                    .generate_with_kv_context(
                        p,
                        *mn,
                        SamplingStrategy::Greedy,
                        *eos,
                        &Device::cpu(),
                        DType::F32,
                    )
                    .unwrap()
            })
            .collect();

        let mut sched = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::Batched { max_batch: 2 },
            test_budget(),
        )
        .unwrap();
        let ids: Vec<SessionId> = cases
            .iter()
            .map(|(p, eos, mn)| {
                sched
                    .add_session(p, SamplingStrategy::Greedy, *eos, *mn)
                    .unwrap()
            })
            .collect();
        let out = sched.run_to_completion().unwrap();

        for (i, id) in ids.iter().enumerate() {
            let got = out.iter().find(|(x, _)| x == id).unwrap().1.clone();
            assert_eq!(
                got, solo[i],
                "session {i} (K=3, max_batch=2 transition) != standalone"
            );
        }
    }

    // Coverage F — K>2 AND a GQA config (n_kv_heads < n_heads). Exercises slot
    // indexing beyond K=2 and the GQA-batched attention head layout.
    #[test]
    fn batched_parity_k3_gqa() {
        // GQA: 4 query heads, 2 kv heads (head_dim 4 → dim 16).
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        assert!(cfg.n_kv_heads < cfg.n_heads, "must be a GQA config");
        let model = LlamaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg, 4242),
        };
        // Same prompt LENGTH across the 3 sessions so they stay lockstep
        // (equal cached_len) and the batched arm is used every step.
        let prompts: [[u32; 3]; 3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let max_new = 6;

        let solo: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| {
                model
                    .generate_with_kv_context(
                        p,
                        max_new,
                        SamplingStrategy::Greedy,
                        None,
                        &Device::cpu(),
                        DType::F32,
                    )
                    .unwrap()
            })
            .collect();

        let mut sched = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::Batched { max_batch: 4 },
            test_budget(),
        )
        .unwrap();
        let ids: Vec<SessionId> = prompts
            .iter()
            .map(|p| {
                sched
                    .add_session(p, SamplingStrategy::Greedy, None, max_new)
                    .unwrap()
            })
            .collect();
        let out = sched.run_to_completion().unwrap();

        for (i, id) in ids.iter().enumerate() {
            let got = out.iter().find(|(x, _)| x == id).unwrap().1.clone();
            assert_eq!(got, solo[i], "GQA K=3 session {i} != standalone");
        }
    }

    // ===== The core/paged split: what a model must provide to be served =====

    /// A model that does ONLY contiguous persistent decode — no paged storage,
    /// no batched arm. Ten of Fuel's twelve model families are shaped exactly
    /// like this today (Gemma3, Glm4, LFM2, Llama3, Phi3, Qwen2, Qwen3,
    /// Qwen3Moe, SmolLm3, T5 have prefix-recompute forwards and nothing else),
    /// so this stub is not a hypothetical.
    ///
    /// Before the trait split this type could not exist: [`DecodeModel`]
    /// required all eight methods with no defaults, including three paged ones,
    /// so a model with no paged surface could not be handed to
    /// [`SessionScheduler`] *even to run the serial arm this test uses*. That is
    /// the whole cost of a fat trait — it wasn't that paged decode was slow for
    /// these models, it was that they couldn't be named to the scheduler at all.
    struct CoreOnlyModel {
        vocab: usize,
    }

    impl DecodeModel for CoreOnlyModel {
        fn n_layers(&self) -> usize {
            2
        }
        fn layer_state_specs(&self) -> Vec<LayerStateSpec> {
            vec![
                LayerStateSpec::KeyValue {
                    n_kv_heads: 2,
                    head_dim: 4
                };
                2
            ]
        }

        /// Deterministic and **cache-dependent**: the argmax walks with
        /// `cached_len`, so a session that fails to advance its cache produces a
        /// visibly different token stream. Per this module's own warning about
        /// the tiny random model, a stub whose output ignored KV state would
        /// make the assertions below vacuous.
        fn forward_with_kv_context_persistent(
            &self,
            tokens: &[u32],
            cache: &mut KvCache,
            _ctx: &mut InferenceContext,
            _session: &mut Option<DecodeSession>,
        ) -> fuel::Result<Vec<f32>> {
            cache.cached_len += tokens.len();
            let mut logits = vec![0.0f32; self.vocab];
            logits[cache.cached_len % self.vocab] = 1.0;
            Ok(logits)
        }
    }

    #[test]
    fn core_only_model_is_servable_by_the_serial_scheduler() {
        let model = CoreOnlyModel { vocab: 16 };
        let mut sched = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::RoundRobin,
            test_budget(),
        )
        .unwrap();
        let id = sched
            .add_session(&[1u32, 2, 3], SamplingStrategy::Greedy, None, 4)
            .expect("a core-only model must be admissible");
        let out = sched.run_to_completion().expect("serial arm must run");

        // `run_to_completion` returns prompt ++ generated.
        let toks = out.iter().find(|(i, _)| *i == id).unwrap().1.clone();
        assert_eq!(
            toks.len(),
            3 + 4,
            "expected prompt + 4 sampled, got {toks:?}"
        );

        // The stub's argmax is `cached_len % vocab`. Prefill consumes the 3
        // prompt tokens (cached_len 3 → token 3), then each decode step consumes
        // 1 more — so the generated tail is strictly increasing. A scheduler that
        // failed to thread the cache through would produce a constant tail.
        assert_eq!(&toks[3..], &[3, 4, 5, 6], "cache did not advance per step");
    }

    /// A model whose decode state is NOT per-head KV — two layers, each an MLA
    /// slot pair (`[kv_lora_rank]` latent + `[qk_rope_head_dim]` `k_pe`), exactly
    /// the shape `DeepSeek2Model` decodes through a `LatentCache`. It has no
    /// honest `(n_kv_heads, head_dim)`; the scheduler must decline it at
    /// construction rather than allocate a KV cache it never reads (GAP-166).
    struct MlaShapedModel;

    impl DecodeModel for MlaShapedModel {
        fn n_layers(&self) -> usize {
            2
        }
        fn layer_state_specs(&self) -> Vec<LayerStateSpec> {
            vec![
                LayerStateSpec::Slots(vec![
                    fuel::decode_state_spec::StateSlot::new(vec![512usize]),
                    fuel::decode_state_spec::StateSlot::new(vec![64usize]),
                ]);
                2
            ]
        }
        fn forward_with_kv_context_persistent(
            &self,
            _tokens: &[u32],
            _cache: &mut KvCache,
            _ctx: &mut InferenceContext,
            _session: &mut Option<DecodeSession>,
        ) -> fuel::Result<Vec<f32>> {
            // Unreachable in this test: the model is declined at construction,
            // before any forward runs. A non-`KvCache` model cannot honestly
            // serve this `&mut KvCache` surface — that is Unit B (`type State`,
            // GAP-166); this body exists only to satisfy the trait.
            Err(Error::Msg(
                "MlaShapedModel: decode unreachable — declined at construction".into(),
            ))
        }
    }

    /// GAP-166: the scheduler seam must DECLINE a model whose layers are not
    /// uniform per-head KV — at construction — instead of fabricating a
    /// `(n_kv_heads, head_dim)` pair and allocating the wrong state. This is the
    /// test that proves [`LayerStateSpec::collapse_uniform`] is a LIVE guard on
    /// the seam rather than a helper exercised only by its own unit tests. The
    /// accept side is covered by every uniform-KV `SessionScheduler::new(..)
    /// .unwrap()` above (e.g. `core_only_model_is_servable_by_the_serial_scheduler`).
    #[test]
    fn mla_shaped_model_is_declined_at_scheduler_construction() {
        let model = MlaShapedModel;
        // `SessionScheduler` is not `Debug`, so `expect_err` doesn't apply — match
        // explicitly. The accept branch panics: reaching it means the seam
        // fabricated a geometry for a non-KV model, the exact GAP-166 defect.
        let err = match SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::RoundRobin,
            test_budget(),
        ) {
            Ok(_) => panic!(
                "a model whose layers are not per-head KV must be declined at \
                 construction, not allocated a KV cache it never reads (GAP-166)"
            ),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("not per-head KV"),
            "decline must come from the collapse boundary \
             (LayerStateSpec::collapse_uniform), got: {err}",
        );
    }

    /// The predicate and the method must agree, or `supports_batched_decode`
    /// becomes exactly the kind of correctly-named-but-lying accessor this
    /// codebase keeps finding: a model could claim the arm and hand back the
    /// default decline, and the scheduler would route every batch into an
    /// error-isolation path while `used_batched_arm` stayed false.
    #[test]
    fn decode_model_batched_capability_is_self_consistent() {
        // Declines: predicate false, default body returns a typed Err (not a
        // panic, and not a silently-empty Ok).
        let core = CoreOnlyModel { vocab: 16 };
        assert!(!core.supports_batched_decode());
        let mut cache = KvCache::with_capacity(2, 2, 4, 8, DType::F32, &Device::cpu()).unwrap();
        let mut refs = [&mut cache];
        let err = DecodeModel::build_batched_decode_logits(
            &core,
            &mut refs,
            &[1u32],
            &Device::cpu(),
            DType::F32,
        )
        .expect_err("the core-only default must decline, not fabricate logits");
        assert!(
            err.to_string().contains("core DecodeModel surface"),
            "the decline must say WHY, got: {err}"
        );

        // Claims and delivers: LlamaModel says true and really overrides.
        let llama = tiny_model(7);
        assert!(llama.supports_batched_decode());
        let mut c0 = KvCache::with_capacity(
            llama.config.n_layers,
            llama.config.n_kv_heads,
            llama.config.head_dim,
            8,
            DType::F32,
            &Device::cpu(),
        )
        .unwrap();
        let mut c1 = KvCache::with_capacity(
            llama.config.n_layers,
            llama.config.n_kv_heads,
            llama.config.head_dim,
            8,
            DType::F32,
            &Device::cpu(),
        )
        .unwrap();
        let mut ctx0 = InferenceContext::new(Device::cpu());
        let mut ctx1 = InferenceContext::new(Device::cpu());
        let mut s0 = None;
        let mut s1 = None;
        llama
            .forward_with_kv_context_persistent(&[1, 2], &mut c0, &mut ctx0, &mut s0)
            .unwrap();
        llama
            .forward_with_kv_context_persistent(&[1, 2], &mut c1, &mut ctx1, &mut s1)
            .unwrap();
        let mut both = [&mut c0, &mut c1];
        let rows = DecodeModel::build_batched_decode_logits(
            &llama,
            &mut both,
            &[3u32, 4u32],
            &Device::cpu(),
            DType::F32,
        )
        .expect("a model claiming the batched arm must actually implement it");
        assert_eq!(rows.len(), 2, "one logits row per cache");
    }

    /// A `Batched` policy on a core-only model must degrade to round-robin,
    /// not fail. Every session still advances; the report says the batched arm
    /// did not run, which is the honest answer rather than a silent claim.
    #[test]
    fn batched_policy_on_a_core_only_model_degrades_to_serial() {
        let model = CoreOnlyModel { vocab: 16 };
        let mut sched = SessionScheduler::new(
            &model,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::Batched { max_batch: 4 },
            test_budget(),
        )
        .unwrap();
        let ids: Vec<SessionId> = [&[1u32, 2, 3][..], &[4u32, 5, 6][..]]
            .iter()
            .map(|p| {
                sched
                    .add_session(p, SamplingStrategy::Greedy, None, 3)
                    .unwrap()
            })
            .collect();

        let report = sched
            .step()
            .expect("step must not fail on a core-only model");
        assert!(
            !report.used_batched_arm,
            "a core-only model has no batched arm — the report must not claim one"
        );

        let out = sched.run_to_completion().unwrap();
        for id in &ids {
            let toks = out.iter().find(|(i, _)| i == id).unwrap().1.clone();
            assert_eq!(
                toks.len(),
                3 + 3,
                "session {id:?} did not advance under Batched"
            );
        }
    }

    /// The point of the whole exercise: `Llama3Model` — the type
    /// `QuantizedLlama3Model::from_gguf` wraps — is now servable. Before this
    /// change its entire method set was `new`/`forward`/`forward_embeds`/
    /// `forward_hidden_embeds`, so a GGUF checkpoint re-ran its whole prefix
    /// for every token; it could not be handed to this scheduler at all.
    #[test]
    fn llama3_model_is_servable_by_the_scheduler() {
        let m = fuel::lazy_llama_full::Llama3Model::new(tiny_model(11), None, None);
        let mut sched = SessionScheduler::new(
            &m,
            Device::cpu(),
            DType::F32,
            SchedulePolicy::RoundRobin,
            test_budget(),
        )
        .unwrap();
        let id = sched
            .add_session(&[1u32, 2, 3], SamplingStrategy::Greedy, None, 4)
            .expect("Llama3Model must be admissible");
        let out = sched.run_to_completion().expect("Llama3Model must decode");
        let toks = out.iter().find(|(i, _)| *i == id).unwrap().1.clone();
        assert_eq!(toks.len(), 3 + 4, "prompt + 4 generated, got {toks:?}");

        // Unscaled Llama3Model is documented as bit-identical to its inner
        // LlamaModel, so the same prompt through the inner model must agree —
        // this is what catches the wrapper accidentally decoding a different
        // graph rather than merely decoding *something*.
        let inner_out = tiny_model(11)
            .generate_with_kv_context(
                &[1u32, 2, 3],
                4,
                SamplingStrategy::Greedy,
                None,
                &Device::cpu(),
                DType::F32,
            )
            .unwrap();
        assert_eq!(
            toks, inner_out,
            "unscaled wrapper diverged from its inner model"
        );
    }
}

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
use fuel::inference_context::{
    DecodeSession, InferenceContext, KvCache, PagedDecodePlan, PagedDecodeSession,
};
use fuel::kv_block_pool::{KvBlockPool, KvGeometry, PoolCapacity, SessionHandle};
use fuel::kv_block_pool_device::{DeviceEvicted, DeviceKvPool};
use fuel::lazy::{sample_logits, LlamaModel, SamplingStrategy};

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
    /// KV head count (cache geometry).
    fn n_kv_heads(&self) -> usize;
    /// Head dimension (cache geometry).
    fn head_dim(&self) -> usize;

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

    /// One batch=K decode step over K sessions' caches — the live batched arm.
    /// Returns one logits row per cache. The scheduler only calls this on a
    /// uniformity-gated ready set; a model may return an error (never a panic)
    /// before mutating any cache (the all-or-nothing contract the batched arm
    /// relies on).
    fn build_batched_decode_logits(
        &self,
        caches: &mut [&mut KvCache],
        last_tokens: &[u32],
        device: &Device,
        dtype: DType,
    ) -> fuel::Result<Vec<Vec<f32>>>;

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
    fn n_kv_heads(&self) -> usize {
        self.config.n_kv_heads
    }
    fn head_dim(&self) -> usize {
        self.config.head_dim
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
    fn build_batched_decode_logits(
        &self,
        caches: &mut [&mut KvCache],
        last_tokens: &[u32],
        device: &Device,
        dtype: DType,
    ) -> fuel::Result<Vec<Vec<f32>>> {
        LlamaModel::build_batched_decode_logits(self, caches, last_tokens, device, dtype)
    }
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
            self, token, pool, session, max_blocks_cap, plan, decode_session,
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
                .bt())
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
    pub fn new(
        model: &'m M,
        device: Device,
        dtype: DType,
        policy: SchedulePolicy,
        budget: KvBudget,
    ) -> Self {
        let kv_pool = KvBlockPool::new(KvGeometry {
            n_layers: model.n_layers(),
            n_kv_heads: model.n_kv_heads(),
            head_dim: model.head_dim(),
            num_blocks: budget.num_blocks,
            block_size: budget.block_size,
            elem_size: dtype.size_in_bytes(),
        });
        Self {
            model,
            device,
            dtype,
            sessions: Vec::new(),
            policy,
            next_id: 0,
            kv_pool,
            kv_handles: HashMap::new(),
        }
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
            )).bt());
        }

        let id = SessionId(self.next_id);
        let dims = ModelDims {
            n_layers: self.model.n_layers(),
            n_kv_heads: self.model.n_kv_heads(),
            head_dim: self.model.head_dim(),
        };
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
            .map_err(|e| Error::Msg(format!("SessionScheduler::add_session: reserve failed: {e:?}")).bt())?;
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

        let mut consumed_by_batch = false;
        if batch_idxs.len() >= 2 {
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
    fn forward_and_store(
        model: &M,
        s: &mut SessionState,
        input: &[u32],
    ) -> fuel::Result<()> {
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
        advance: fuel::Result<()>,
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
        self.sessions.iter().all(|s| s.phase == SessionPhase::Finished)
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
            s.cache.layers.truncate(0);
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
pub struct PagedSessionScheduler<'m, M: DecodeModel> {
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
}

impl<'m, M: DecodeModel> PagedSessionScheduler<'m, M> {
    /// Build an empty paged scheduler over a shared model + a KV block-pool
    /// `budget`. The pool's head geometry is taken from the model; `budget` sets
    /// block size + total blocks (the shared VRAM ceiling).
    pub fn new(
        model: &'m M,
        budget: KvBudget,
        dtype: DType,
        device: &Device,
    ) -> fuel::Result<Self> {
        let pool = DeviceKvPool::new(
            KvGeometry {
                n_layers: model.n_layers(),
                n_kv_heads: model.n_kv_heads(),
                head_dim: model.head_dim(),
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
        self.sessions.iter().any(|s| s.id == id && s.suspended.is_some())
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
                return Err(Error::Msg(format!("restore_session: session {id:?} is not suspended")).bt())
            }
        };
        let free = self.pool.core().free_blocks();
        if need > free {
            return Err(Error::Msg(format!(
                "restore_session: needs {need} blocks, {free} free — session {id:?} stays \
                 suspended (free room and retry)",
            )).bt());
        }
        // Pre-checked, so `restore` cannot OutOfBlocks — only now consume the handle.
        let evicted = self.sessions[idx].suspended.take().expect("checked Some above");
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
            return Err(Error::Msg("PagedSessionScheduler::add_session: prompt is empty".into()).bt());
        }
        if max_new == 0 {
            return Err(Error::Msg("PagedSessionScheduler::add_session: max_new must be > 0".into()).bt());
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
            let mut last_logits: Option<Vec<f32>> = None;
            let mut failure: Option<String> = None;
            for &tok in &prompt {
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
                self.sessions[i].phase == SessionPhase::Decode && self.sessions[i].suspended.is_none()
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
                let msg = format!("decode_batch: {} rows for {} sessions", rows.len(), idxs.len());
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
        self.sessions.iter().map(|s| (s.id, s.tokens.clone())).collect()
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
    use fuel::lazy::{LlamaConfig, LlamaModel, LlamaWeights, LayerWeights, WeightStorage, SamplingStrategy};
    // NOTE: `fuel_ir::Device` does not exist — the device type is `fuel::Device`
    // (fuel_core::Device), which is what `KvCache::with_capacity` takes. `DType`
    // is `fuel_ir::DType`. This mirrors the `use` lines at the top of
    // inference_context.rs (`use fuel_ir::{DType, ..}; use fuel::Device;`).
    use fuel::Device;
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
    /// A generous KV budget for tests that don't exercise the capacity gate —
    /// large enough that no admission is ever rejected. Capacity-gate tests
    /// build their own tight budget.
    fn test_budget() -> KvBudget {
        KvBudget { block_size: 16, num_blocks: 4096 }
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
            &model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin, test_budget());
        let id = sched.add_session(&prompt, SamplingStrategy::Greedy, None, max_new).unwrap();
        let out = sched.run_to_completion().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, id);
        assert_eq!(out[0].1, standalone);   // byte-identical token stream
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
        let budget = KvBudget { block_size: 8, num_blocks: 2 };
        let mut sched =
            SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin, budget);
        let prompt = [1u32, 2, 3];

        assert_eq!(sched.kv_blocks_required(prompt.len(), 5), 1, "8 tokens / block_size 8 = 1 block");
        assert_eq!(sched.kv_free_blocks(), 2);
        assert_eq!(sched.kv_bytes_resident(), 0, "C-4: nothing reserved yet");

        sched.add_session(&prompt, SamplingStrategy::Greedy, None, 5).unwrap();
        sched.add_session(&prompt, SamplingStrategy::Greedy, None, 5).unwrap();
        assert_eq!(sched.kv_free_blocks(), 0, "both blocks reserved");
        assert!(sched.kv_bytes_resident() > 0, "C-4: resident bytes reflect the 2 reserved blocks");

        // Third admission is rejected on capacity — total, nothing half-built.
        let rejected = sched.add_session(&prompt, SamplingStrategy::Greedy, None, 5);
        assert!(rejected.is_err(), "no room → typed capacity rejection (C-1)");
        assert_eq!(sched.kv_free_blocks(), 0, "a rejected admit consumes no blocks");

        // Complete the two sessions, reap them, and the reservations return.
        let _ = sched.run_to_completion().unwrap();
        let reaped = sched.reap_finished();
        assert_eq!(reaped.len(), 2, "both finished sessions reaped");
        assert_eq!(sched.kv_free_blocks(), 2, "reap reclaimed both blocks");
        assert_eq!(sched.kv_bytes_resident(), 0, "C-4: back to zero after reap");

        // A fresh session admits again into the reclaimed headroom.
        sched.add_session(&prompt, SamplingStrategy::Greedy, None, 5).unwrap();
        assert_eq!(sched.kv_free_blocks(), 1);
    }

    /// A multi-block session reserves ⌈max_seq_len / block_size⌉ blocks, and the
    /// pre-check query agrees with what admission actually reserves.
    #[test]
    fn kv_blocks_required_matches_multi_block_reservation() {
        let model = tiny_model(7);
        let budget = KvBudget { block_size: 4, num_blocks: 16 };
        let mut sched =
            SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin, budget);
        let prompt = [1u32, 2, 3, 4, 5]; // 5 + max_new 6 = 11 tokens → ceil(11/4) = 3 blocks
        assert_eq!(sched.kv_blocks_required(prompt.len(), 6), 3);
        let free_before = sched.kv_free_blocks();
        sched.add_session(&prompt, SamplingStrategy::Greedy, None, 6).unwrap();
        assert_eq!(sched.kv_free_blocks(), free_before - 3, "admission reserved exactly the queried 3 blocks");
    }

    // --- PagedSessionScheduler (PS3) -------------------------------------

    fn paged_budget() -> KvBudget {
        KvBudget { block_size: 4, num_blocks: 64 }
    }

    /// A single paged session runs to its `max_new` budget: prefill (feed the
    /// prompt token-by-token) + decode, ending with prompt + max_new tokens.
    #[test]
    fn paged_scheduler_single_session_runs_to_budget() {
        let model = tiny_model(42);
        let mut s =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        let prompt = [1u32, 2, 3];
        let id = s.add_session(&prompt, SamplingStrategy::Greedy, None, 5).unwrap();
        let out = s.run_to_completion();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, id);
        assert_eq!(out[0].1.len(), prompt.len() + 5, "prompt + 5 generated tokens");
    }

    /// eos stops a paged session right after it is emitted (budget not exhausted).
    #[test]
    fn paged_scheduler_stops_on_eos() {
        let model = tiny_model(42);
        let prompt = [1u32, 2, 3];
        // Learn the first greedy token, then use it as eos.
        let mut s0 =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        s0.add_session(&prompt, SamplingStrategy::Greedy, None, 5).unwrap();
        let full = s0.run_to_completion()[0].1.clone();
        let first_gen = full[prompt.len()];

        let mut s =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        s.add_session(&prompt, SamplingStrategy::Greedy, Some(first_gen), 5).unwrap();
        let out = s.run_to_completion()[0].1.clone();
        assert_eq!(out.len(), prompt.len() + 1, "stops right after emitting eos");
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
        sa.add_session(&a, SamplingStrategy::Greedy, None, max_new).unwrap();
        let a_alone = sa.run_to_completion()[0].1.clone();
        let mut sb =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        sb.add_session(&b, SamplingStrategy::Greedy, None, max_new).unwrap();
        let b_alone = sb.run_to_completion()[0].1.clone();

        // TOGETHER (one scheduler, SHARED pool).
        let mut both =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        let ida = both.add_session(&a, SamplingStrategy::Greedy, None, max_new).unwrap();
        let idb = both.add_session(&b, SamplingStrategy::Greedy, None, max_new).unwrap();
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
        s.add_session(&[1u32, 2, 3], SamplingStrategy::Greedy, None, 5).unwrap();
        s.run_to_completion();
        assert!(s.kv_free_blocks() < free0, "a decoded session holds pool blocks");
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
                &prompt, max_new, SamplingStrategy::Greedy, None, &Device::cpu(), DType::F32,
            )
            .unwrap();
        let mut s =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        s.add_session(&prompt, SamplingStrategy::Greedy, None, max_new).unwrap();
        let paged = s.run_to_completion()[0].1.clone();
        assert_eq!(paged, contig, "paged scheduler greedy == contiguous generate oracle");
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
        let id = s.add_session(&prompt, SamplingStrategy::Greedy, None, max_new).unwrap();
        let _ = s.run_to_completion();
        let rc = s
            .session_realize_count(id)
            .expect("default-configured session holds a decode plan after decoding");
        assert!(rc >= 1, "default path rebound the held decode plan per token (got {rc})");
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
        assert_eq!(s_replan.plan(), PagedDecodePlan::Replan, "reference arm is Replan");
        s_replan.add_session(&prompt, SamplingStrategy::Greedy, None, max_new).unwrap();
        let replan_out = s_replan.run_to_completion()[0].1.clone();

        // Plan-once arm: build + optimize the decode graph once, rebind per token.
        let mut s_plan =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        s_plan.set_plan(PagedDecodePlan::PlanOnce);
        let id = s_plan.add_session(&prompt, SamplingStrategy::Greedy, None, max_new).unwrap();
        let plan_out = s_plan.run_to_completion()[0].1.clone();

        assert_eq!(plan_out, replan_out, "plan-once output == replan output (byte-exact)");

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
        assert!(rc >= 1, "plan-once path rebound the held decode plan per token (got {rc})");
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
        s0.add_session(&prompt, SamplingStrategy::Greedy, None, max_new).unwrap();
        let reference = s0.run_to_completion()[0].1.clone();

        // Evict mid-decode, restore, finish — final tokens must match.
        let mut s =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        let id = s.add_session(&prompt, SamplingStrategy::Greedy, None, max_new).unwrap();
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
            s.kv_free_blocks(), free_decoding,
            "restore re-took exactly the freed blocks",
        );

        let out = s.run_to_completion()[0].1.clone();
        assert_eq!(out, reference, "evict→restore resumes byte-exact (same tokens as uninterrupted)");
    }

    /// The pressure-valve flow end to end: evict a session to free the pool, let
    /// another session run in the freed room, then restore the first and finish
    /// it. `run_to_completion` returns with a suspended session still live rather
    /// than spinning.
    #[test]
    fn paged_scheduler_evict_lets_another_run_then_restore_finishes() {
        let model = tiny_model(9999);
        let mut s = PagedSessionScheduler::new(
            &model, KvBudget { block_size: 4, num_blocks: 16 }, DType::F32, &Device::cpu(),
        ).unwrap();
        let a = s.add_session(&[1u32, 2, 3], SamplingStrategy::Greedy, None, 4).unwrap();
        s.step();
        let free_with_a = s.kv_free_blocks();
        s.evict_session(a).unwrap();
        assert!(s.kv_free_blocks() > free_with_a, "A's blocks freed");

        // B runs to completion in the pool while A is suspended; run_to_completion
        // returns (does not spin) with A still suspended.
        let b = s.add_session(&[5u32, 6], SamplingStrategy::Greedy, None, 3).unwrap();
        let out1 = s.run_to_completion();
        let b_len = out1.iter().find(|(id, _)| *id == b).map(|(_, t)| t.len());
        assert_eq!(b_len, Some(2 + 3), "B (prompt 2 + 3 generated) finished in the freed pool");
        assert!(s.is_suspended(a), "A stayed suspended while B ran");

        // Restore A and finish it.
        s.restore_session(a).unwrap();
        let out2 = s.run_to_completion();
        let a_len = out2.iter().find(|(id, _)| *id == a).map(|(_, t)| t.len());
        assert_eq!(a_len, Some(3 + 4), "A resumed and finished (prompt 3 + 4 generated)");
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
        let ida = ss.add_session(&a, SamplingStrategy::Greedy, None, max_new).unwrap();
        let idb = ss.add_session(&b, SamplingStrategy::Greedy, None, max_new).unwrap();
        let serial = ss.run_to_completion();
        let sa = serial.iter().find(|(id, _)| *id == ida).unwrap().1.clone();
        let sb = serial.iter().find(|(id, _)| *id == idb).unwrap().1.clone();

        // Batched: loop step_batched to completion.
        let mut bs =
            PagedSessionScheduler::new(&model, paged_budget(), DType::F32, &Device::cpu()).unwrap();
        let ida2 = bs.add_session(&a, SamplingStrategy::Greedy, None, max_new).unwrap();
        let idb2 = bs.add_session(&b, SamplingStrategy::Greedy, None, max_new).unwrap();
        let mut fired = false;
        while bs.sessions.iter().any(|s| s.phase != SessionPhase::Finished) {
            let r = bs.step_batched(8);
            fired |= r.used_batched_arm;
        }
        assert!(fired, "the batched arm actually ran (two same-position sessions)");
        let out: Vec<_> = bs.sessions.iter().map(|s| (s.id, s.tokens.clone())).collect();
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
        let solo_a = model.generate_with_kv_context(
            &prompt_a, max_new, SamplingStrategy::Greedy, None, &Device::cpu(), DType::F32).unwrap();
        let solo_b = model.generate_with_kv_context(
            &prompt_b, max_new, SamplingStrategy::Greedy, None, &Device::cpu(), DType::F32).unwrap();

        // K=2 scheduled together.
        let mut sched = SessionScheduler::new(
            &model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin, test_budget());
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
        let mut rr = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin, test_budget());
        let a1 = rr.add_session(&pa, SamplingStrategy::Greedy, None, max_new).unwrap();
        let b1 = rr.add_session(&pb, SamplingStrategy::Greedy, None, max_new).unwrap();
        let out_rr = rr.run_to_completion().unwrap();

        // One-then-the-other: A alone to completion, then B alone.
        let mut s_a = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin, test_budget());
        s_a.add_session(&pa, SamplingStrategy::Greedy, None, max_new).unwrap();
        let just_a = s_a.run_to_completion().unwrap();
        let mut s_b = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin, test_budget());
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
        let mut sched = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin, test_budget());
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

    #[test]
    fn t4_session_isolation_on_error() {
        let model = tiny_model(9999);
        let good = [1u32,2,3];
        let mut sched = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin, test_budget());
        let bad_id  = sched.add_poisoned_session_for_test(&[4u32,5], 5).unwrap();
        let good_id = sched.add_session(&good, SamplingStrategy::Greedy, None, 5).unwrap();

        // First step: the poisoned session errors, the good one advances. No panic.
        let r0 = sched.step().unwrap();
        assert!(r0.errored.iter().any(|(id,_)| *id==bad_id), "poisoned session must be reported errored");

        let out = sched.run_to_completion().unwrap();
        let solo_good = model.generate_with_kv_context(
            &good, 5, SamplingStrategy::Greedy, None, &Device::cpu(), DType::F32).unwrap();
        let g = out.iter().find(|(i,_)| *i==good_id).unwrap().1.clone();
        assert_eq!(g, solo_good, "the healthy session must complete unaffected");
    }

    #[test]
    fn t7_mid_run_add_prefills_and_joins() {
        let model = tiny_model(9999);
        let pa = [1u32,2,3];
        let pb = [8u32,1];
        let max_new = 6;

        let mut sched = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin, test_budget());
        let ida = sched.add_session(&pa, SamplingStrategy::Greedy, None, max_new).unwrap();
        // Advance A alone for two steps.
        sched.step().unwrap();
        sched.step().unwrap();
        // Now add B mid-run.
        let idb = sched.add_session(&pb, SamplingStrategy::Greedy, None, max_new).unwrap();
        let out = sched.run_to_completion().unwrap();

        let solo_a = model.generate_with_kv_context(&pa, max_new, SamplingStrategy::Greedy, None, &Device::cpu(), DType::F32).unwrap();
        let solo_b = model.generate_with_kv_context(&pb, max_new, SamplingStrategy::Greedy, None, &Device::cpu(), DType::F32).unwrap();
        let g = |id: SessionId| out.iter().find(|(i,_)| *i==id).unwrap().1.clone();
        assert_eq!(g(ida), solo_a, "A unaffected by mid-run B");
        assert_eq!(g(idb), solo_b, "B prefills correctly mid-run");
    }

    #[test]
    fn t6_uniformity_gate_rejects_ragged_cached_len() {
        let d = |cl: usize| BatchDescriptor { cached_len: cl, max_seq_len: 64, n_layers: 2, cache_dtype: DType::F32 };
        assert!(batch_uniform(&[d(3), d(3)]));           // equal → batchable
        assert!(!batch_uniform(&[d(3), d(4)]));          // ragged → not
        assert!(!batch_uniform(&[d(3)]));                // <2 sessions → not
    }

    #[test]
    fn t6_batched_policy_falls_back_to_serial_equals_roundrobin() {
        // With Task 7's stub always returning NotBatchable, a Batched policy
        // must produce byte-identical output to RoundRobin (pure serial).
        let model = tiny_model(9999);
        let (pa, pb, max_new) = ([1u32,2,3], [5u32,6,7], 6);
        let run = |policy| {
            let mut s = SessionScheduler::new(&model, Device::cpu(), DType::F32, policy, test_budget());
            let a = s.add_session(&pa, SamplingStrategy::Greedy, None, max_new).unwrap();
            let b = s.add_session(&pb, SamplingStrategy::Greedy, None, max_new).unwrap();
            (a, b, s.run_to_completion().unwrap())
        };
        let (a1,b1,rr) = run(SchedulePolicy::RoundRobin);
        let (a2,b2,ba) = run(SchedulePolicy::Batched { max_batch: 4 });
        let g = |o: &Vec<(SessionId,Vec<u32>)>, id: SessionId| o.iter().find(|(i,_)| *i==id).unwrap().1.clone();
        assert_eq!(g(&rr,a1), g(&ba,a2));
        assert_eq!(g(&rr,b1), g(&ba,b2));
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
            use fuel::inference_context::{KvCache, InferenceContext};
            let cfg = &model.config;
            let msl = prompt.len() + 2;
            let mut cache = KvCache::with_capacity(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, msl, DType::F32, &Device::cpu()).unwrap();
            let mut ctx = InferenceContext::new(Device::cpu());
            let mut sess = None;
            let pre = model.forward_with_kv_context_persistent(prompt, &mut cache, &mut ctx, &mut sess).unwrap();
            let next = fuel::lazy::sample_logits(&pre, SamplingStrategy::Greedy, &mut 0u64);
            model.forward_with_kv_context_persistent(&[next], &mut cache, &mut ctx, &mut sess).unwrap()
        };
        let sa = serial_logits(&pa);
        let sb = serial_logits(&pb);

        // Batched: build two SessionStates prefilled + first token sampled,
        // at equal cached_len, then one try_batched_step.
        let mut states: Vec<SessionState> = Vec::new();
        for (id, p) in [(0u64, &pa[..]), (1u64, &pb[..])] {
            let mut s = SessionState::new(SessionId(id),
                ModelDims { n_layers: model.config.n_layers, n_kv_heads: model.config.n_kv_heads, head_dim: model.config.head_dim },
                p, SamplingStrategy::Greedy, None, 2, &Device::cpu(), DType::F32).unwrap();
            // Prefill + sample first token so cached_len == prompt.len() and
            // last token is set (mirrors scheduler.step prefill pass).
            s.last_logits = Some(model.forward_with_kv_context_persistent(&s.tokens.clone(), &mut s.cache, &mut s.ctx, &mut s.session).unwrap());
            s.sample_and_append().unwrap();
            states.push(s);
        }
        assert_eq!(states[0].cache.cached_len, states[1].cache.cached_len, "equal cached_len (uniform)");
        let mut refs: Vec<&mut SessionState> = states.iter_mut().collect();
        let outcome = BatchedDecode::try_batched_step(&model, &Device::cpu(), DType::F32, &mut refs).unwrap();
        match outcome {
            BatchOutcome::Advanced(rows) => {
                assert_eq!(rows.len(), 2);
                // Batched decode step logits == serial decode step logits (f32, ε-tol).
                let close = |a: &[f32], b: &[f32]| a.len()==b.len() && a.iter().zip(b).all(|(x,y)| (x-y).abs() < 1e-4);
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
        use fuel::lazy::{LlamaConfig, LlamaModel, LayerWeights, LlamaWeights, WeightStorage};

        fn bf16_weights(cfg: &LlamaConfig) -> LlamaWeights {
            // f32 tiny weights → BF16 for every WeightStorage matrix (embedding
            // + norm gains stay f32, per make_tiny_weights_bf16's frozen seams).
            let mut s: u32 = 9999;
            let mut next = || { s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1 };
            let mut vec_of = |n: usize| -> std::sync::Arc<[f32]> {
                std::sync::Arc::from((0..n).map(|_| next()).collect::<Vec<_>>()) };
            let to_bf16 = |a: std::sync::Arc<[f32]>| -> WeightStorage {
                WeightStorage::BF16(std::sync::Arc::from(
                    a.iter().map(|&v| half::bf16::from_f32(v)).collect::<Vec<_>>())) };
            let kv = cfg.n_kv_heads * cfg.head_dim;
            LlamaWeights {
                token_embedding: vec_of(cfg.vocab_size * cfg.dim),
                layers: (0..cfg.n_layers).map(|_| LayerWeights {
                    attn_q: to_bf16(vec_of(cfg.dim*cfg.dim)), attn_q_bias: None,
                    attn_k: to_bf16(vec_of(cfg.dim*kv)), attn_k_bias: None,
                    attn_v: to_bf16(vec_of(cfg.dim*kv)), attn_v_bias: None,
                    attn_o: to_bf16(vec_of(cfg.dim*cfg.dim)),
                    ffn_gate: to_bf16(vec_of(cfg.dim*cfg.ffn_dim)),
                    ffn_up: to_bf16(vec_of(cfg.dim*cfg.ffn_dim)),
                    ffn_down: to_bf16(vec_of(cfg.ffn_dim*cfg.dim)),
                    attn_norm_gain: std::sync::Arc::from(vec![1.0f32; cfg.dim]),
                    ffn_norm_gain: std::sync::Arc::from(vec![1.0f32; cfg.dim]),
                }).collect(),
                final_norm_gain: std::sync::Arc::from(vec![1.0f32; cfg.dim]),
                output: to_bf16(vec_of(cfg.dim*cfg.vocab_size)),
            }
        }

        let cfg = tiny_cfg();
        let model = LlamaModel { config: cfg.clone(), weights: bf16_weights(&cfg) };
        let dev = fuel::cuda_backend::new_device(0).expect("cuda device 0");
        let dt = DType::BF16;
        let pa = [1u32, 2, 3];
        let pb = [4u32, 5, 6];

        let serial_logits = |prompt: &[u32]| -> Vec<f32> {
            use fuel::inference_context::{KvCache, InferenceContext};
            let msl = prompt.len() + 2;
            let mut cache = KvCache::with_capacity(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, msl, dt, &dev).unwrap();
            let mut ctx = InferenceContext::new(dev.clone());
            let mut sess = None;
            let pre = model.forward_with_kv_context_persistent(prompt, &mut cache, &mut ctx, &mut sess).unwrap();
            let next = fuel::lazy::sample_logits(&pre, SamplingStrategy::Greedy, &mut 0u64);
            model.forward_with_kv_context_persistent(&[next], &mut cache, &mut ctx, &mut sess).unwrap()
        };
        let sa = serial_logits(&pa);
        let sb = serial_logits(&pb);

        let mut states: Vec<SessionState> = Vec::new();
        for (id, p) in [(0u64, &pa[..]), (1u64, &pb[..])] {
            let mut s = SessionState::new(SessionId(id),
                ModelDims { n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads, head_dim: cfg.head_dim },
                p, SamplingStrategy::Greedy, None, 2, &dev, dt).unwrap();
            s.last_logits = Some(model.forward_with_kv_context_persistent(&s.tokens.clone(), &mut s.cache, &mut s.ctx, &mut s.session).unwrap());
            s.sample_and_append().unwrap();
            states.push(s);
        }
        assert_eq!(states[0].cache.cached_len, states[1].cache.cached_len, "equal cached_len (uniform)");
        let mut refs: Vec<&mut SessionState> = states.iter_mut().collect();
        let outcome = BatchedDecode::try_batched_step(&model, &dev, dt, &mut refs).unwrap();
        match outcome {
            BatchOutcome::Advanced(rows) => {
                assert_eq!(rows.len(), 2);
                let eps = 5e-3_f32;
                let close = |a: &[f32], b: &[f32]| a.len()==b.len() && a.iter().zip(b).all(|(x,y)| (x-y).abs() < eps);
                assert!(close(&rows[0], &sa), "batched row 0 != serial A (bf16 ε={eps})");
                assert!(close(&rows[1], &sb), "batched row 1 != serial B (bf16 ε={eps})");
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
        let mut c0 = KvCache::with_capacity(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, msl, DType::BF16, &Device::cpu()).unwrap();
        let mut c1 = KvCache::with_capacity(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, msl, DType::BF16, &Device::cpu()).unwrap();
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
                    .generate_with_kv_context(p, *mn, SamplingStrategy::Greedy, *eos, &Device::cpu(), DType::F32)
                    .unwrap()
            })
            .collect();

        let mut sched = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::Batched { max_batch: 2 }, test_budget());
        let ids: Vec<SessionId> = cases
            .iter()
            .map(|(p, eos, mn)| sched.add_session(p, SamplingStrategy::Greedy, *eos, *mn).unwrap())
            .collect();
        let out = sched.run_to_completion().unwrap();

        for (i, id) in ids.iter().enumerate() {
            let got = out.iter().find(|(x, _)| x == id).unwrap().1.clone();
            assert_eq!(got, solo[i], "session {i} (K=3, max_batch=2 transition) != standalone");
        }
    }

    // Coverage F — K>2 AND a GQA config (n_kv_heads < n_heads). Exercises slot
    // indexing beyond K=2 and the GQA-batched attention head layout.
    #[test]
    fn batched_parity_k3_gqa() {
        // GQA: 4 query heads, 2 kv heads (head_dim 4 → dim 16).
        let cfg = LlamaConfig {
            vocab_size: 16, dim: 16, n_layers: 2, n_heads: 4,
            n_kv_heads: 2, head_dim: 4, ffn_dim: 32, norm_eps: 1e-5, rope_base: 10000.0,
        };
        assert!(cfg.n_kv_heads < cfg.n_heads, "must be a GQA config");
        let model = LlamaModel { config: cfg.clone(), weights: tiny_weights(&cfg, 4242) };
        // Same prompt LENGTH across the 3 sessions so they stay lockstep
        // (equal cached_len) and the batched arm is used every step.
        let prompts: [[u32; 3]; 3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let max_new = 6;

        let solo: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| model.generate_with_kv_context(p, max_new, SamplingStrategy::Greedy, None, &Device::cpu(), DType::F32).unwrap())
            .collect();

        let mut sched = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::Batched { max_batch: 4 }, test_budget());
        let ids: Vec<SessionId> = prompts
            .iter()
            .map(|p| sched.add_session(p, SamplingStrategy::Greedy, None, max_new).unwrap())
            .collect();
        let out = sched.run_to_completion().unwrap();

        for (i, id) in ids.iter().enumerate() {
            let got = out.iter().find(|(x, _)| x == id).unwrap().1.clone();
            assert_eq!(got, solo[i], "GQA K=3 session {i} != standalone");
        }
    }
}

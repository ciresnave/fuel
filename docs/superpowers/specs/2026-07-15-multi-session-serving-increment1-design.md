# Multi-session serving — Increment 1: the multi-session decode substrate — design

**Date:** 2026-07-15 · **Status:** design, pre-plan · **Part of:** the multi-agent / multi-session serving program (CireSnave's stated near-term roadmap priority — "basics first, multi-agent/multi-session serving very soon after"; see memory [[multi-agent-serving-goal]], [[parallel-branch-kv-sharing-audit]]). **Greenfield.** This spec scopes only the FIRST increment — K independent decode sessions on one model — and defers KV-content sharing, a paged block-pool, and admission/preemption policy to named later increments.

> **Grounding:** every file:line below was read against the working tree on branch `op-scan-phase1` (2026-07-15). Verify against current code before treating a citation as load-bearing; a few thin-evidence flags are called out inline.

## Goal

Run **K independent decode sessions concurrently on one model, correctly** — each session generating its own token stream from its own prompt — reusing the existing single-session persistent/captured decode machinery and, where beneficial, the existing batched-decode kernels. The bar is **correctness first**: K sessions advanced together must produce **the exact same tokens** each would produce run alone (no cross-session contamination), and a batched decode step must match K serial single-session steps. Throughput from batching is the *reason* to build it, but the increment is complete when it is correct and reuses the substrate; a measured batched speedup is a bench, not a gate.

This increment is a **host-side orchestration layer** over parts that already exist. It adds: (a) a **session-state** abstraction that bundles one session's `KvCache` + position + sampler/RNG state; (b) a **scheduler** that advances K sessions' decode steps; (c) an optional **batched decode path** that runs N ready sessions' attention through the existing `flash_decoding` batch dimension in one call. It adds **no new IR op** and **no new kernel**.

## Scope decision (2026-07-15, CireSnave) — the LIVE batched arm is IN SCOPE

Increment 1 ships the **live** batched-decode arm, not just a gated seam. That fixes these previously-open decisions:

- **#1 scope → build the live batched arm.** C3 is a real, executed batched-decode path (not offered-but-stubbed). The serial arm remains the correctness oracle; the batched arm must be provably byte-equal to it (T5 parity) and is the fast path when the ready set is uniform.
- **#2 KV presentation → a Fuel-internal shared `[K,…]` batch-slot buffer.** Do NOT open a baracuda pointer-array ask (that would block the autonomous build on a cross-project change). The batched arm gathers the K ready sessions' per-layer K/V into a shared contiguous `[K, n_kv_heads, capacity, head_dim]` buffer (host-side, Fuel-allocated) that `flash_decoding`'s `batch` + per-tensor batch strides consume; scatter results back per session. The exact copy-in/copy-out vs. sessions-allocated-into-slots-of-one-buffer tradeoff is a plan-level design point (grounded in `KvCache`/`WriteSlice`).
- **#3 lockstep-only.** The batched arm batches ONLY sessions at **equal `cached_len`** (`flash_decoding` takes one shared `k_len`, [attention.rs:1006](fuel-cuda-backend/src/baracuda/attention.rs#L1006)). Sessions at different positions fall to the serial arm. Ragged-length batching (padding-to-max + masking, or a per-seq k_len variant) is deferred.
- **#4 captures.** The serial arm keeps **per-session** `DecodeSession`/`CapturedRun` (each baked to its own buffers). The batched arm needs a **batched decode pass over the shared `[K,…]` buffer** — a *separate* batched plan/capture (or a non-captured plan-once batched realize); whether it is CUDA-graph-captured is a plan-level call (start non-captured; capturing the batched pass is an optimization).
- **#5 genericity → Llama-first, trait-shaped.** Build against `LlamaModel` concretely, but shape `SessionState` + the scheduler so `PhiModel`'s identical quartet is a later drop-in (a `DecodeModel` trait can be extracted then; don't build Phi now).
- **#6 fail-on-OOM.** No admission/KV-pressure controller in inc-1; a session that can't allocate its KV fails in isolation (never-panic → typed error), the others continue.
- **#7 mid-batch fault → all-or-nothing KV commit.** On a fault mid-batch, no session's KV is left half-written (commit the batched KV-write atomically per step, or roll the batch back to the serial arm) — a plan-level never-panic detail.

The live batched arm is the hardest part of this increment (a Fuel-side shared batch-slot KV buffer + `flash_decoding` batch wiring + byte-exact parity); the plan must sequence it *behind* a working serial substrate so the parity oracle exists before the batched arm is written.

## Background (grounded)

### Where one decode session's state lives today

A single generation is driven by `LlamaModel::generate_streaming_with_kv_context` ([lazy.rs:8333](fuel-core/src/lazy.rs#L8333)) — the loop that owns everything that is per-generation state, threaded as **four separate locals**:

- `cache: KvCache` — allocated once via `KvCache::with_capacity(n_layers, n_kv_heads, head_dim, max_seq_len, dtype, device)` ([inference_context.rs:181](fuel-core/src/inference_context.rs#L181)). Backend-erased; each layer holds `Arc<RwLock<Storage>>` for K and V pre-allocated as `[1, n_kv_heads, max_seq_len, head_dim]` zero buffers, written in place per step by `Op::WriteSlice`/`Op::WriteSliceDoff` ([inference_context.rs:128](fuel-core/src/inference_context.rs#L128), [:96](fuel-core/src/inference_context.rs#L96)). `cached_len` is the live position; `truncate_to`/`clear` reset it ([:376](fuel-core/src/inference_context.rs#L376), [:358](fuel-core/src/inference_context.rs#L358)).
- `ctx: InferenceContext` — the per-session persistent storage map (`HashMap<NodeId, Arc<RwLock<Storage>>>`) that survives across realize calls so weights + KV are not re-uploaded ([inference_context.rs:965](fuel-core/src/inference_context.rs#L965)).
- `session: Option<DecodeSession>` — the plan-once decode state: the HELD, already-optimized decode-step graph + cached `OptimizedGraph` view + the stable re-bindable data-Const NodeIds (`token_ids`, `rope_cos/sin`, `mask`, per-layer `kv_nodes`, `offset_node`) + the full realized `base_cache` ([inference_context.rs:688](fuel-core/src/inference_context.rs#L688)). Built on the first `seq==1` token, reused for every later token; `realize_token` clones `base_cache`, overwrites only the per-token data Consts, and re-realizes via the plan-once seam ([:810](fuel-core/src/inference_context.rs#L810)). Validity keyed by `(seq, max_seq_len, n_layers, cache_dtype)` ([:844](fuel-core/src/inference_context.rs#L844)).
- `rng_state: u64` + the running `tokens: Vec<u32>` — the sampler state, held as loop locals ([lazy.rs:8349-8353](fuel-core/src/lazy.rs#L8349)).

The forward step itself is `forward_with_kv_context_persistent(tokens, &mut cache, &mut ctx, &mut session)` ([lazy.rs:7094](fuel-core/src/lazy.rs#L7094)); prefill (`seq>1`) routes through the same entry and internally falls back to the D1 rebuild path without building a session. **`PhiModel` has the identical quartet** ([lazy.rs:9350](fuel-core/src/lazy.rs#L9350), [:9692](fuel-core/src/lazy.rs#L9692)) — so a session-state abstraction that generalizes over the four locals is not Llama-specific.

**Observation that drives C1:** these four objects ARE a session; today they are unbundled loop locals. Increment 1's session-state type is a faithful bundle of exactly this quartet — no new state, just a name and an owner.

### The captured-replay (CapturedRun) path

`CapturedDecodeSession` ([pipelined.rs:357](fuel-dispatch/src/pipelined.rs#L357)) is the CUDA-graph capture/replay wrapper: `capture(graph, target, inputs, per_token_input_ids, sym_env)` captures the decode graph once against fixed device input addresses ([:383](fuel-dispatch/src/pipelined.rs#L383)); `replay_token(updates)` H2D-overwrites the per-token input buffers in place, replays one `cuGraphLaunch`, and returns the output Arc ([:436](fuel-dispatch/src/pipelined.rs#L436)). Driven from the model layer by `forward_with_kv_context_captured` ([lazy.rs:7582](fuel-core/src/lazy.rs#L7582)). Memory [[capturedrun-executor-buildout]] records this as byte-exact + ~10× on TinyLlama-1.1B. **Load-bearing property for this increment:** a `CapturedDecodeSession` bakes a *specific* graph against *specific* fixed input addresses (its own KV Arcs, its own per-token buffers). Two sessions cannot share one capture unless they share those buffers — so **each session owns its own capture** (see Open questions).

### The batched-decode kernel substrate

Two batched kernels exist and are the reuse target:

1. **`flash_decoding` (CUDA, baracuda alpha.72)** — split-K decode attention over a fixed-capacity KV cache. FFI at [attention.rs:996](fuel-cuda-backend/src/baracuda/attention.rs#L996): it takes a `batch: i32` ([:1003](fuel-cuda-backend/src/baracuda/attention.rs#L1003)) with explicit per-tensor `q_b_stride`/`k_b_stride`/`v_b_stride`/`y_b_stride`, GQA-native (`num_kv_heads` separate), `seq_q == 1`, `head_dim ∈ [1,128]`, f16/bf16. It is reachable in decode via the optimizer-emitted `Op::Branch` flash arm (`decode_flash::DecodeFlashSpec`, [decode_flash.rs:112](fuel-dispatch/src/decode_flash.rs#L112); admissibility `flash_decode_admissible`, [:161](fuel-dispatch/src/decode_flash.rs#L161)). **The pivotal constraint for batching (see Risks): the FFI takes a single `k_len: i32` ([attention.rs:1006](fuel-cuda-backend/src/baracuda/attention.rs#L1006)) — ONE iteration bound shared across the whole batch.** N sessions can batch through one call only if they attend the *same* prefix length; sessions at different decode positions cannot (without padding-to-max, which changes results unless masked, or a per-seq variant).
2. **`gemm_dense` (CUDA, baracuda alpha.67)** — the cuBLAS-backed batched dense GEMM that backs `Op::Matmul` on CUDA, with a per-slot batch loop and `stride_b=0` broadcast ([gemm_dense.rs:1](fuel-cuda-backend/src/baracuda/gemm_dense.rs#L1), [:38](fuel-cuda-backend/src/baracuda/gemm_dense.rs#L38)). **Correction to memory [[parallel-branch-kv-sharing-audit]]'s "gemm_dense exists but unused":** as read, `gemm_dense` is the *live* matmul kernel (it "retires the last hand-written matmul path"), so the QKV/MLP projection GEMMs of a batched decode step reach it automatically via a batch axis — no new wiring for the projection half.

### What is ABSENT (confirmed — do not assume cheap)

- **No paged block-pool allocator.** `Op::PagedAttn` exists as a fused-op *registry entry* ([paged_attn.rs:40](fuel-graph/src/registry/paged_attn.rs#L40)) with a **panicking `decompose`** and a stubbed pattern; its inputs name a `block_table`/`context_lens`/paged `k_cache` layout ([:8-15](fuel-graph/src/registry/paged_attn.rs#L8)) but **nothing allocates or manages blocks** — there is no block-pool, no block table producer, no live kernel path. Building on PagedAttn is NOT cheap; it is a later increment.
- **No scheduler / no multi-session driver.** Every `generate*` entry ([lazy.rs:8280](fuel-core/src/lazy.rs#L8280), [:8333](fuel-core/src/lazy.rs#L8333), [:8398](fuel-core/src/lazy.rs#L8398)) drives exactly one generation to completion in a straight loop. There is no type that holds >1 session or interleaves their steps.
- **No shared/contiguous cross-session KV buffer.** Each `KvCache` is its own set of `Arc<RwLock<Storage>>` allocations ([inference_context.rs:181](fuel-core/src/inference_context.rs#L181)); there is no `[K_sessions, ...]` batched KV tensor. This is the crux of the batched-attention integration cost (Risks).
- **No cross-session KV splicing / donation.** Deferred to Increment 2 per [[parallel-branch-kv-sharing-audit]] (splice via a host-level `KvCache` method, not graph ops).

## Architecture

Three components, host-side, layered over the existing single-session parts. Nothing below the model API changes; the IR, the executor, and every kernel are reused as-is.

```
   ┌─────────────────────────────────────────────────────────────┐
   │ SessionScheduler  (C2)  — owns K SessionState, advances them │
   │   step():  pick ready sessions → decode (batched|serial) →   │
   │            per-session sample → append → retire finished      │
   └───────────────┬───────────────────────────┬─────────────────┘
                   │                            │
        ┌──────────▼──────────┐      ┌──────────▼───────────────┐
        │ SessionState[0..K]  │ ...  │  BatchedDecode (C3)       │
        │  (C1)               │      │  N ready seq==1 sessions  │
        │  KvCache            │      │  → one decode pass reusing│
        │  InferenceContext   │      │    flash_decoding batch / │
        │  DecodeSession?     │      │    gemm_dense batch, OR    │
        │  rng + tokens + pos │      │  → serial fallback (K×1)  │
        └─────────────────────┘      └───────────────────────────┘
                   │
        one shared read-only  LlamaModel  (weights)
```

- **The model is shared read-only.** `LlamaModel` is immutable weights; a session is per-generation state (the `DecodeSession` doc already states this, [inference_context.rs:670-677](fuel-core/src/inference_context.rs#L670)). K sessions borrow one `&LlamaModel`.
- **The scheduler owns the sessions and the loop.** It is the new top-level driver, replacing the single straight `for` loop with a K-way interleave.
- **Batched decode is an optional fast arm, not the correctness path.** The serial arm (advance each ready session with the existing `forward_with_kv_context_persistent`) is arm 0 / the oracle; the batched arm is offered only when the ready set is uniform enough to satisfy the kernel constraints, and must be provably equal to the serial arm.

## Components

### C1 — `SessionState`: the per-session bundle (fuel-core)

**Boundary.** Owns one session's mutable decode state and NOTHING shared. A faithful bundle of today's four loop locals, no new state semantics.

```
struct SessionState {
    cache:   KvCache,                 // inference_context.rs:128
    ctx:     InferenceContext,        // inference_context.rs:965
    session: Option<DecodeSession>,   // inference_context.rs:688  (plan-once; None until 1st decode)
    // sampler/output state (today's loop locals, lazy.rs:8349)
    tokens:    Vec<u32>,
    rng_state: u64,
    strategy:  SamplingStrategy,
    eos_id:    Option<u32>,
    // scheduling bookkeeping
    remaining: usize,                 // max_new_tokens budget left
    phase:     SessionPhase,          // Prefill | Decode | Finished
    last_logits: Option<Vec<f32>>,    // produced by the last step, consumed by sample
    id:        SessionId,
}
enum SessionPhase { Prefill, Decode, Finished }
```

- **`cached_len` / position** lives inside `cache` already (`KvCache::cached_len`) — not duplicated.
- **Interface (all `Result`, never panic):**
  - `SessionState::new(model_dims, prompt, strategy, eos, max_new, device, dtype) -> Result<Self>` — allocates the `KvCache::with_capacity` (can `Err` on OOM, propagated) + `InferenceContext`, seeds `tokens = prompt`, `phase = Prefill`.
  - `fn is_ready(&self) -> bool` — `phase != Finished`.
  - `fn sample_and_append(&mut self) -> Result<Option<u32>>` — consumes `last_logits` with the existing `sample_logits` ([lazy.rs:8380](fuel-core/src/lazy.rs#L8380)) + this session's own `rng_state`, appends, checks `eos`/`remaining`, transitions to `Finished` when done. **Per-session RNG is the contamination firewall** (see Testing).
- **What C1 does NOT do:** it does not advance the graph. Stepping the model is the scheduler's job (C2), because a *batched* step spans multiple `SessionState`s and cannot be a method on one.

### C2 — `SessionScheduler`: the K-way decode driver (fuel-core)

**Boundary.** Owns `Vec<SessionState>` + `&LlamaModel` + the device. Decides *which* sessions advance together and *how* (batched vs serial). Owns no tensor state of its own beyond a scratch batch descriptor.

```
struct SessionScheduler<'m> {
    model:    &'m LlamaModel,
    device:   Device,
    dtype:    DType,
    sessions: Vec<SessionState>,
    policy:   SchedulePolicy,   // RoundRobin | Batched { max_batch }
}
```

- **Interface:**
  - `fn add_session(&mut self, prompt, strategy, eos, max_new) -> Result<SessionId>`.
  - `fn step(&mut self) -> Result<StepReport>` — advance one scheduling quantum: (1) run any `Prefill`-phase sessions (serial — prefill graphs are per-prompt-length, not batchable in Increment 1); (2) collect the `Decode`-phase ready set; (3) advance them via C3 (batched if policy + uniformity allow, else serial round-robin); (4) `sample_and_append` each; (5) retire `Finished`. Returns which sessions produced a token and which finished.
  - `fn run_to_completion(&mut self) -> Result<Vec<(SessionId, Vec<u32>)>>` — loop `step` until all `Finished`. The convenience wrapper the tests + first users call.
- **Isolation contract:** a `Result::Err` from advancing session *i* is captured into `session[i].phase = Finished` with a recorded error in the `StepReport`, and **does not abort the other sessions** (Error handling below). One session's failure is not a scheduler failure.
- **Policy default:** `RoundRobin` for correctness-first landing; `Batched { max_batch }` is the opt-in fast path gated on C3's uniformity check. **This default is an Open question** (below) — recommended default `RoundRobin`, user to confirm whether Increment 1 should land batched-by-default.

### C3 — `BatchedDecode`: the optional batched attention step (fuel-core + reuse)

**Boundary.** Given N `Decode`-phase `SessionState`s, produce N logits vectors in one model pass **iff** the ready set satisfies the batched-kernel constraints; otherwise signal "not batchable" so C2 falls back to serial. C3 owns the batch-axis assembly and the equality-to-serial contract; it owns no persistent state.

- **The projection half (QKV, MLP GEMMs) batches for free:** stacking N sessions on a leading batch axis makes those `Op::Matmul` nodes carry a batch dim already served by `gemm_dense`'s per-slot loop ([gemm_dense.rs:38](fuel-cuda-backend/src/baracuda/gemm_dense.rs#L38)). No new op.
- **The attention half is the hard part** because `flash_decoding` takes a single shared `k_len` ([attention.rs:1006](fuel-cuda-backend/src/baracuda/attention.rs#L1006)) and per-tensor batch strides that assume one contiguous `[batch, Hkv, capacity, D]` KV buffer ([:1008-1015](fuel-cuda-backend/src/baracuda/attention.rs#L1008)), whereas each session's KV is a *separate* allocation. Increment 1 therefore restricts the batched arm to a **uniformity-gated subset**: sessions batchable together only when they share `(model, dtype, head geometry, max_seq_len)` AND `cached_len` (so one `k_len` is correct for all) AND their KV can be presented to the kernel as a batch (via the per-tensor batch-stride ABI — see Risks for the two candidate mechanisms). When the ready set is not uniform, C3 returns `NotBatchable` and C2 serial-steps.
- **Interface:** `fn try_batched_step(model, device, sessions: &mut [&mut SessionState]) -> Result<BatchOutcome>` where `BatchOutcome ∈ { Advanced(Vec<Vec<f32>>), NotBatchable }`. `NotBatchable` is a normal control-flow value, not an error.
- **Increment-1 conservative scope (recommended):** land C3 with the **serial fallback as the only wired arm**, plus the uniformity gate + the `BatchOutcome::NotBatchable` seam and its parity test, and treat the *live* batched `flash_decoding` wiring as the increment's single measured-benefit milestone behind the gate. This keeps the increment correct-and-shippable even if the batched-KV presentation (Risks) needs a follow-up, while proving the seam. **Whether to require the live batched arm in Increment 1 or defer it one step is an Open question.**

## Data flow

K prompts → `scheduler.add_session` ×K (each allocates its own `KvCache` + `InferenceContext`, seeds `SessionState`) → `run_to_completion` loops:

1. **Prefill:** each `Prefill` session runs `forward_with_kv_context_persistent(prompt, …)` serially → `last_logits` set, `phase = Decode`.
2. **Ready set:** collect `Decode` sessions with budget left.
3. **Advance:**
   - **Batched (policy + uniform):** `BatchedDecode::try_batched_step` → if `Advanced`, each session's `last_logits` filled from its batch slot; if `NotBatchable`, fall through.
   - **Serial (default / fallback):** for each ready session, `forward_with_kv_context_persistent([last_token], &mut s.cache, &mut s.ctx, &mut s.session)` — the *existing* plan-once path, each session reusing its own held `DecodeSession`.
4. **Sample:** each advanced session `sample_and_append` with ITS OWN `rng_state` → next token appended / `eos` / budget → maybe `Finished`.
5. Repeat until all `Finished`; return `(SessionId, tokens)` per session.

Per-session `KvCache.cached_len` advances independently; per-session `Op::WriteSlice` writes into that session's own KV Arcs; per-session `DecodeSession` re-binds only that session's per-token Consts. **No storage Arc is shared between two sessions** in Increment 1 — that is what makes cross-session contamination structurally impossible on the serial path and is the property the batched path must preserve.

## Error handling / never-panic

- Every new surface returns `Result` (`SessionState::new`, `sample_and_append`, `scheduler.step`, `try_batched_step`). No new `.unwrap()`/`.expect()` on production paths (CLAUDE.md never-panic).
- **Session-level isolation:** `scheduler.step` advances sessions in a way that converts a per-session `Err` into that session finishing with a recorded error (surfaced in `StepReport`), never propagating out and killing the batch. A poisoned lock, a CUDA OOM on one session's realize, or a `TopologyChanged` on one held `DecodeSession` isolates to that session. (`TopologyChanged` on the serial path already invalidates + rebuilds the single session per [inference_context.rs:809](fuel-core/src/inference_context.rs#L809)'s contract — the scheduler wraps that same recovery per session.)
- **Batched arm safety:** if `try_batched_step` hits any error mid-batch it returns `Err` for the whole batch *only* before any session's KV was mutated; once KV writes begin the batched step must complete or leave a detectable per-session inconsistency that forces those sessions to `Finished`-with-error rather than silently corrupting. **Recommended:** compute the full batched step functionally and commit KV writes last (mirrors the existing `Op::WriteSlice` in-place discipline). This ordering is a plan-level detail flagged in Open questions.
- Validation at construction time: `SessionState::new` validates `max_new > 0`, non-empty prompt, and that all sessions added to one `Batched` scheduler share the model geometry the batched arm assumes (reject at `add_session`, not at `step`).

## Testing (TDD, born-red)

Core scheduler logic is pure host orchestration — most gates run **CPU-only** (serial path). The batched-kernel parity gates are CUDA `#[ignore]` live tests (one live-GPU suite at a time per CLAUDE.md).

- **T1 — no cross-session contamination (the headline gate, CPU).** Run K=2 sessions with *different* prompts through `SessionScheduler::run_to_completion`, and independently run each prompt alone through the existing `generate_with_kv_context` ([lazy.rs:8398](fuel-core/src/lazy.rs#L8398)). Assert each scheduled session's token stream is **identical** to its standalone run. This proves independent `KvCache` + independent `rng_state` fully isolate sessions. Born-red: fails today because no scheduler exists.
- **T2 — interleave-order invariance (CPU).** K=2 sessions, greedy strategy: assert `run_to_completion` yields the same per-session tokens regardless of scheduling order (round-robin vs one-then-the-other), i.e. `step` is order-independent for independent sessions.
- **T3 — per-session RNG independence (CPU).** Two sessions, same prompt, *different* `Temperature` seeds → different streams; same seed → identical to the standalone seeded run ([lazy.rs:8350](fuel-core/src/lazy.rs#L8350)'s seed semantics). Guards against a shared/global RNG sneaking in.
- **T4 — session isolation on error (CPU).** Inject a forced error in one session's advance (e.g. a deliberately invalid budget/geometry) → assert the *other* session still completes and the failing one is reported `Finished`-with-error in the `StepReport`, no panic.
- **T5 — batched == serial parity (CUDA, `#[ignore]`).** With the batched arm wired: N=2 uniform sessions (same `cached_len`, f16/bf16) advanced via `try_batched_step` → `Advanced`, vs the same two advanced serially → assert **bit-identical** logits (or sabotage-calibrated ε if the batched GEMM reduction order differs, per [[sabotage-test-calibration]]). This is the batched-substrate correctness gate.
- **T6 — uniformity gate (CPU-testable).** Sessions at *different* `cached_len` → `try_batched_step` returns `NotBatchable` (not a wrong shared-`k_len` result); the gate logic is unit-testable without CUDA by feeding the descriptor.
- **T7 — prefill-then-decode lifecycle (CPU).** A session added mid-run (after others have decoded several tokens) prefills correctly and joins the decode ready set without disturbing the others.

## Boundaries — explicitly deferred (named later increments)

- **Increment 2 — KV-content SHARING / splicing across sessions.** Prompt-prefix sharing, residual donation entry point. Per [[parallel-branch-kv-sharing-audit]]: splice via a host-level `KvCache` method (a new `KvCache` API that Arc-shares or copies K/V slabs), **not** via `Op::Branch`/`WriteSlice`/graph ops. Increment 1 keeps every session's KV strictly private.
- **Later — a PagedAttn block-pool allocator.** The `Op::PagedAttn` registry entry exists ([paged_attn.rs:40](fuel-graph/src/registry/paged_attn.rs#L40)) but the allocator/block-table machinery is entirely absent; a real paged cache (non-contiguous per-session KV, dynamic block reuse) is its own increment and is what ultimately relaxes the batched-attention contiguity constraint.
- **Later — residual-stream donation / partial re-evaluation** (the layer-L dial from [[parallel-branch-kv-sharing-audit]]).
- **Later — admission / preemption / fair-share policy** beyond simple round-robin + a size-capped batch. Priorities, deadlines, KV-pressure eviction, and continuous-batching admission (adding sessions to an in-flight batch every step) are out of scope; Increment 1's `add_session`-then-`run` is batch-static per scheduler instance (T7's mid-run add is allowed but not a scheduling *policy*).
- **Unchanged:** the IR, the executor, every kernel, the FKC/Spec-B verification stack. Increment 1 is host orchestration only.

## Open questions / risks (flagged for the user — each carries a recommended default)

1. **Batched vs round-robin as the Increment-1 default.** *Recommend:* land `RoundRobin` (serial) as the correctness path + ship the batched arm behind a uniformity gate as an opt-in `SchedulePolicy::Batched`. **Confirm:** should Increment 1 *require* a live, measured batched `flash_decoding` arm, or is proving the seam (gate + `NotBatchable` + parity harness) with the serial arm sufficient, deferring the live batched wiring to a fast follow-up?
2. **Batched-attention KV presentation — the integration risk.** `flash_decoding` needs one `batch`-strided view over N sessions' KV ([attention.rs:1008](fuel-cuda-backend/src/baracuda/attention.rs#L1008)), but each session's KV is a *separate* allocation ([inference_context.rs:181](fuel-core/src/inference_context.rs#L181)). Two candidate mechanisms, both plan-level: **(a)** allocate one shared `[K, Hkv, capacity, D]` KV buffer up front and give each session a batch-slot view (contiguous, kernel-friendly, but caps K at allocation time and wastes memory for short sessions); **(b)** pass per-session base pointers via the batch-stride ABI if baracuda's `flash_decoding` accepts a gather/pointer-array batch (grep-confirmed the FFI takes uniform strides, NOT a pointer array — so (b) likely needs a baracuda ask, which is a cross-project proposal per CLAUDE.md, not a Fuel-internal change). *Recommend:* **(a)** for Increment 1 (self-contained, no external dependency), documented as the reason batched K is fixed-at-construction. **Confirm** the memory-vs-flexibility tradeoff, and whether to open a baracuda conversation about a pointer-array batch for the paged future.
3. **The single shared `k_len` constraint.** Even with a shared KV buffer, one `flash_decoding` call attends one `k_len` for all batch rows ([attention.rs:1006](fuel-cuda-backend/src/baracuda/attention.rs#L1006)). *Recommend:* Increment 1 only batches sessions at **equal `cached_len`** (the uniformity gate, T6) — realistic when K sessions start together / step in lockstep. Ragged-length batching (pad-to-max + mask, or a per-seq-`k_len` kernel) is deferred with PagedAttn. **Confirm** this lockstep restriction is acceptable for the first target workload.
4. **One CapturedRun plan per session, or shared?** A `CapturedDecodeSession` bakes fixed input addresses ([pipelined.rs:383](fuel-dispatch/src/pipelined.rs#L383)) — a session's own KV + per-token Arcs. *Recommend:* **each session owns its own `DecodeSession`/capture** in Increment 1 (K plan-once graphs; the plan-once cost is amortized per session, and captures cannot alias another session's buffers). A *batched* capture (one captured graph over the shared batch buffer) is the natural Increment-1.5 optimization once (2a) lands. **Confirm** K independent captures is acceptable memory/warmup overhead for the first K.
5. **Session-state ownership boundary.** *Recommend:* `SessionState` bundles the four existing locals verbatim and lives in `fuel-core` alongside `InferenceContext`/`DecodeSession` ([inference_context.rs](fuel-core/src/inference_context.rs)); the `SessionScheduler` is the new public driver, model-agnostic where possible (Llama + Phi share the quartet, [lazy.rs:9350](fuel-core/src/lazy.rs#L9350)). **Confirm** whether the scheduler should be generic over a small `DecodeModel` trait (Llama/Phi/…) now, or Llama-only for Increment 1 with the trait extracted in a later increment.
6. **Memory budgeting across K sessions.** K × `KvCache::with_capacity` can OOM the device (each cache is `n_layers·2·n_kv_heads·max_seq_len·head_dim·dtype_size`, [inference_context.rs:173](fuel-core/src/inference_context.rs#L173)). *Recommend:* Increment 1 does **no** budgeting — `add_session` returns the propagated allocation `Err` on OOM and the caller sizes K; a KV-pressure admission controller is deferred (Boundaries). **Confirm** a fail-on-OOM (no eviction, no admission control) posture is acceptable for the first increment.
7. **Batched KV-write commit ordering (never-panic detail).** The batched step must not leave some sessions' KV written and others not on a mid-batch error. *Recommend:* compute functionally, commit `Op::WriteSlice`s last; on error force the affected sessions to `Finished`-with-error. Plan-level; flagged so the plan writes T4/T5 to cover a mid-batch fault.

---

### Summary

Increment 1 is a **host-side multi-session decode substrate**: a `SessionState` bundle of the four per-generation locals that already exist (`KvCache`, `InferenceContext`, the plan-once `DecodeSession`, and sampler/RNG state — all read at [inference_context.rs](fuel-core/src/inference_context.rs) and driven today by [lazy.rs:8333](fuel-core/src/lazy.rs#L8333)); a `SessionScheduler` that advances K independent sessions with per-session isolation; and an optional `BatchedDecode` arm that reuses the existing `flash_decoding` batch dimension ([attention.rs:996](fuel-cuda-backend/src/baracuda/attention.rs#L996)) and the live `gemm_dense` batched matmul ([gemm_dense.rs:1](fuel-cuda-backend/src/baracuda/gemm_dense.rs#L1)) behind a uniformity gate. It adds no IR op and no kernel. Correctness is the bar: K scheduled sessions must produce byte-identical token streams to K standalone runs (T1), and a batched step must equal serial (T5). The genuinely hard part — and the increment's main risk — is presenting N separate per-session KV allocations to a kernel that wants one contiguous `batch`-strided buffer with a single shared `k_len`; the recommended answer is a shared batch-slot KV buffer at equal `cached_len`, with paged/ragged batching explicitly deferred (PagedAttn's allocator is absent, [paged_attn.rs:40](fuel-graph/src/registry/paged_attn.rs#L40)). KV-content sharing, the block-pool allocator, residual donation, and admission/preemption policy are named later increments.

### Open design decisions flagged for the user

1. **Batched-by-default vs serial-default** (and whether Increment 1 must ship the *live* batched arm or just the seam + parity harness).
2. **Batched-attention KV presentation:** shared `[K, …]` KV buffer (self-contained, caps K early) vs a baracuda pointer-array batch ask (cross-project). Recommended: shared buffer.
3. **Lockstep-only batching** (equal `cached_len`) for Increment 1 vs ragged-length batching — confirm the lockstep restriction fits the first workload.
4. **One CapturedRun/`DecodeSession` per session** vs a shared batched capture. Recommended: per-session for Increment 1.
5. **Session-state ownership + scheduler genericity:** Llama-only now vs a `DecodeModel` trait (Llama/Phi share the state quartet) up front.
6. **Cross-session memory budgeting:** fail-on-OOM with no admission control for Increment 1 vs building a KV-pressure controller now. Recommended: fail-on-OOM.
7. **Batched KV-write commit ordering** on a mid-batch fault (plan-level never-panic detail).

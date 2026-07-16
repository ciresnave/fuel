# Multi-session serving — Increment 1 Implementation Plan

> **For agentic workers:** execute task-by-task, TDD. Each task is a checkbox sequence: write the failing test → run it RED → implement → run GREEN → commit. Do not skip the RED step. Do not batch tasks. One `cargo` invocation at a time, FOREGROUND.

## Goal

Run **K independent decode sessions concurrently on one `LlamaModel`, correctly** — each session generating its own token stream from its own prompt, reusing the existing single-session persistent decode machinery. Correctness is the bar:

- **T1:** K scheduled sessions produce **byte-identical** token streams to K standalone `generate_with_kv_context` runs (no cross-session contamination).
- **T5:** a **live batched** decode step (shared `[K,…]` KV buffer through `flash_decoding`'s batch dim) is **byte-equal** to K serial single-session steps.

This is a **host-side orchestration layer**. It adds **no IR op** and **no kernel**. Three components: `SessionState` (C1 — bundle the four per-generation loop locals), `SessionScheduler` (C2 — the K-way serial driver + the T1 parity oracle), and the **live** `BatchedDecode` arm (C3 — a Fuel-internal shared `[K, n_kv_heads, capacity, head_dim]` batch-slot KV buffer + `flash_decoding` batch wiring, lockstep-only, byte-equal to the serial arm).

Full design: [docs/superpowers/specs/2026-07-15-multi-session-serving-increment1-design.md](../specs/2026-07-15-multi-session-serving-increment1-design.md). Read the **Scope decision** block: the live batched arm is IN SCOPE; the serial arm is the byte-exact oracle; lockstep-only (equal `cached_len`); shared Fuel-internal KV buffer (no baracuda pointer-array ask).

## Architecture

```
   SessionScheduler (C2)  — owns Vec<SessionState> + &LlamaModel + device/dtype
     step(): run Prefill sessions serially → collect Decode ready set →
             advance (batched if uniform+policy, else serial) →
             per-session sample_and_append → retire Finished
                   │                                    │
        ┌──────────▼──────────┐            ┌────────────▼─────────────┐
        │ SessionState[0..K]  │            │ BatchedDecode (C3)        │
        │  (C1)               │            │  N uniform Decode sessions│
        │  KvCache            │            │  → shared [K,Hkv,cap,D] KV│
        │  InferenceContext   │            │    buffer + batch=K graph │
        │  Option<DecodeSess> │            │    → flash_decoding batch  │
        │  tokens+rng+phase   │            │  → NotBatchable → serial   │
        └─────────────────────┘            └───────────────────────────┘
                   │
        one shared read-only &LlamaModel (weights)
```

The model is shared read-only. Every `SessionState` owns its own `KvCache` (own `Arc<RwLock<Storage>>` allocations) and its own `rng_state` — that is what makes cross-session contamination structurally impossible on the serial path. The batched arm must preserve that property.

## Tech Stack

- New module `fuel-core/src/multi_session.rs` (declared `pub mod multi_session;` in `fuel-core/src/lib.rs`). **Do NOT reuse the name `scheduling`** — `fuel-core/src/scheduling.rs` already exists (probe/judge dispatch table) and would collide.
- Reuses, unchanged: `KvCache` / `InferenceContext` / `DecodeSession` ([fuel-core/src/inference_context.rs](../../../fuel-core/src/inference_context.rs)), `LlamaModel::forward_with_kv_context_persistent` / `forward_with_kv_context` / `sample_logits` / `SamplingStrategy` ([fuel-core/src/lazy.rs](../../../fuel-core/src/lazy.rs)).
- C3 batched realize goes through `InferenceContext::realize_one_as_with_env::<f32>` (non-captured plan-once-per-step, per spec #4) over a shared batched KV buffer built with the same `Op::Alloc`→`Op::ZeroFill` emission pattern as `KvCache::with_capacity` ([inference_context.rs:181](../../../fuel-core/src/inference_context.rs#L181)).
- Errors: `fuel_ir::Error` / `crate::Result` (the alias used throughout `lazy.rs` / `inference_context.rs`), `.bt()` on every constructed error.

## Global Constraints (binding — copy these into every working session)

- **`-p <crate>` builds — NEVER workspace-wide.** `tensor-tools` has a standing `Device::Cpu` break and is a default-member, so bare `cargo check`/`cargo test` at the root fails. Always `cargo test -p fuel-core …`.
- **ONE `cargo` invocation at a time.** The build-dir lock serializes; parallel invocations thrash. Long builds: background + wait.
- **Run `cargo` in the FOREGROUND.** A subagent deadlocks waiting on its own backgrounded `cargo` job (bg notifications reach the main loop, not the sub-subagent). Run cargo foreground; the controller recovers a stall by running verify+commit itself.
- **One live-GPU test suite at a time.** Two concurrent live suites OOM the dev GPU (RTX 4070, 12 GB). C3's GPU parity test is `#[ignore]` + local-only.
- **TDD, born-red.** Write the failing test first, run it and SEE it fail for the expected reason, then make it green. A change that touches behavior ships with the test that exercises it, observed to run.
- **Never panic on production paths.** `Result` from day one; no new `.unwrap()`/`.expect()` on production paths (tests may `.unwrap()`). Every new surface returns `Result`.
- **Validate at graph-build / construction time.** Every check that *can* run early *must* — reject bad geometry at `SessionState::new` / `add_session`, not at `step`.
- **Docs are part of the change.** This is a greenfield host layer; on completion, add a one-line pointer to the new module from `ROADMAP.md`'s frontier and a `10-decisions-log.md` note only if a core claim changes (it does not — no IR/kernel change).

## CUDA build recipe (Task 8 GPU test ONLY — optional/local, RTX 4070)

Build `--features cuda` from a **VS Developer shell** (or set `NVCC_CCBIN=<path-to-cl.exe>`) or `nvcc` fails `Cannot find compiler 'cl.exe'`. To *launch* the fuel-core cuda test exe, prepend cuDNN's CUDA-13.3 bin to PATH: `C:\Program Files\NVIDIA\CUDNN\v9.23\bin\13.3\x64` (else `0xc0000135 STATUS_DLL_NOT_FOUND`). Cold cuda build ~30+ min — build FOREGROUND and wait.

---

## Task 1 — `SessionState` bundle + construction (C1)

**Files:**
- Create `fuel-core/src/multi_session.rs`.
- Modify `fuel-core/src/lib.rs` — add `pub mod multi_session;` next to `pub mod inference_context;` ([lib.rs:246](../../../fuel-core/src/lib.rs#L246)).

**Interfaces — Produces:**
```rust
pub struct SessionId(pub u64);                        // Copy, Eq, Hash, Debug
pub enum SessionPhase { Prefill, Decode, Finished }   // Clone, PartialEq, Debug

/// Model geometry a session needs to size its KvCache (Llama-first;
/// filled from LlamaConfig by the scheduler).
#[derive(Clone, Copy, Debug)]
pub struct ModelDims { pub n_layers: usize, pub n_kv_heads: usize, pub head_dim: usize }

pub struct SessionState {
    pub(crate) cache:       crate::inference_context::KvCache,
    pub(crate) ctx:         crate::inference_context::InferenceContext,
    pub(crate) session:     Option<crate::inference_context::DecodeSession>,
    pub(crate) tokens:      Vec<u32>,
    pub(crate) rng_state:   u64,
    pub(crate) strategy:    crate::lazy::SamplingStrategy,
    pub(crate) eos_id:      Option<u32>,
    pub(crate) remaining:   usize,                     // max_new_tokens budget left
    pub(crate) phase:       SessionPhase,
    pub(crate) last_logits: Option<Vec<f32>>,
    pub(crate) id:          SessionId,
    pub(crate) new_tokens:  Vec<u32>,                  // just the GENERATED tail (for reporting)
}

impl SessionState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: SessionId,
        dims: ModelDims,
        prompt: &[u32],
        strategy: crate::lazy::SamplingStrategy,
        eos_id: Option<u32>,
        max_new: usize,
        device: &fuel_ir::Device,
        dtype: fuel_ir::DType,
    ) -> crate::Result<Self>;
    pub fn is_ready(&self) -> bool;   // phase != Finished
    pub fn id(&self) -> SessionId;
    pub fn tokens(&self) -> &[u32];
}
```
- `new` mirrors the loop-local setup in `generate_streaming_with_kv_context` ([lazy.rs:8349-8371](../../../fuel-core/src/lazy.rs#L8349)): validate `max_new > 0` and non-empty `prompt` (else `Err`); `rng_state` = `seed` for `Temperature { seed, .. }`, else `0`; `max_seq_len = prompt.len() + max_new`; `cache = KvCache::with_capacity(dims.n_layers, dims.n_kv_heads, dims.head_dim, max_seq_len, dtype, device)?` (propagates OOM `Err`); `ctx = InferenceContext::new(device.clone())`; `session = None`; `tokens = prompt.to_vec()`; `new_tokens = vec![]`; `phase = Prefill`.
- Exact `Device`/`DType` import: they resolve via `fuel_ir` (re-exported and used across `lazy.rs`/`inference_context.rs`). Match the `use` lines at the top of `inference_context.rs`.

**Steps:**
- [ ] Write the failing test (append to a `#[cfg(test)] mod tests` in `multi_session.rs`):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::{LlamaConfig, LlamaModel, LlamaWeights, LayerWeights, WeightStorage, SamplingStrategy};
    use fuel_ir::{Device, DType};
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
```
- [ ] Run RED: `cargo test -p fuel-core --lib multi_session::tests::session_new` (FOREGROUND). Expect a compile error (module/type missing) — that is the RED.
- [ ] Implement `SessionId`/`SessionPhase`/`ModelDims`/`SessionState` + `new`/`is_ready`/`id`/`tokens` per the interface. Keep the test helpers (`tiny_cfg`/`tiny_weights`/`tiny_model`/`dims`) in the test module — later tasks reuse them.
- [ ] Run GREEN: same command; both tests pass.
- [ ] Commit: `feat(multi-session): SessionState bundle + construction (C1, T-none)`.

**Deliverable:** a constructible `SessionState` bundling the four per-generation locals, validated at construction.

---

## Task 2 — `SessionState::sample_and_append` (C1)

**Files:** Modify `fuel-core/src/multi_session.rs`.

**Interfaces — Consumes:** `crate::lazy::sample_logits(&[f32], SamplingStrategy, &mut u64) -> u32` ([lazy.rs:8691](../../../fuel-core/src/lazy.rs#L8691)). **Produces:**
```rust
impl SessionState {
    /// Consume `last_logits` with THIS session's own rng_state, append the
    /// token, decrement budget, transition to Finished on eos / budget
    /// exhaustion. Returns the sampled token (None if there was nothing to
    /// sample — no last_logits, or already Finished).
    pub fn sample_and_append(&mut self) -> crate::Result<Option<u32>>;
}
```
- Logic mirrors the decode-loop body ([lazy.rs:8379-8391](../../../fuel-core/src/lazy.rs#L8379)): take `last_logits` (`self.last_logits.take()`); if `None` or `phase == Finished` return `Ok(None)`. `let next = sample_logits(&logits, self.strategy, &mut self.rng_state);` push to `tokens` + `new_tokens`; `self.remaining = self.remaining.saturating_sub(1);` if `Some(eos) = self.eos_id` and `next == eos` → `phase = Finished`; else if `self.remaining == 0` → `phase = Finished`; else `phase = Decode`. Return `Ok(Some(next))`.
- **Per-session RNG is the contamination firewall** — `sample_and_append` must read/advance ONLY `self.rng_state`, never a shared/global state.

**Steps:**
- [ ] Write the failing test (in the same `mod tests`):
```rust
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
```
- [ ] Run RED: `cargo test -p fuel-core --lib multi_session::tests::sample_and_append` (FOREGROUND). Expect method-missing compile error.
- [ ] Implement `sample_and_append`.
- [ ] Run GREEN.
- [ ] Commit: `feat(multi-session): SessionState::sample_and_append with per-session rng (C1)`.

**Deliverable:** a session can sample its next token from held logits using only its own RNG and correctly transition to `Finished`.

---

## Task 3 — `SessionScheduler` serial core + `step` / `run_to_completion` (C2)

**Files:** Modify `fuel-core/src/multi_session.rs`.

**Interfaces — Consumes:** `LlamaModel::forward_with_kv_context_persistent(&self, tokens, &mut KvCache, &mut InferenceContext, &mut Option<DecodeSession>) -> Result<Vec<f32>>` ([lazy.rs:7094](../../../fuel-core/src/lazy.rs#L7094)); `LlamaConfig` fields ([lazy.rs:5826](../../../fuel-core/src/lazy.rs#L5826)). **Produces:**
```rust
pub enum SchedulePolicy { RoundRobin, Batched { max_batch: usize } }

pub struct StepReport {
    pub advanced:  Vec<SessionId>,              // produced a token this step
    pub finished:  Vec<SessionId>,              // transitioned to Finished this step
    pub errored:   Vec<(SessionId, String)>,    // finished-with-error this step
    pub used_batched_arm: bool,                 // set true only when C3 Advanced
}

pub struct SessionScheduler<'m> {
    model:    &'m LlamaModel,
    device:   fuel_ir::Device,
    dtype:    fuel_ir::DType,
    sessions: Vec<SessionState>,
    policy:   SchedulePolicy,
    next_id:  u64,
}

impl<'m> SessionScheduler<'m> {
    pub fn new(model: &'m LlamaModel, device: fuel_ir::Device, dtype: fuel_ir::DType,
               policy: SchedulePolicy) -> Self;
    pub fn add_session(&mut self, prompt: &[u32], strategy: crate::lazy::SamplingStrategy,
                       eos_id: Option<u32>, max_new: usize) -> crate::Result<SessionId>;
    pub fn step(&mut self) -> crate::Result<StepReport>;
    pub fn run_to_completion(&mut self) -> crate::Result<Vec<(SessionId, Vec<u32>)>>;
    pub fn is_all_finished(&self) -> bool;
}
```

**Behaviour (this task ships the SERIAL arm only — C3 lands in Tasks 7–8):**
- `add_session`: mint `SessionId(self.next_id)` (post-increment); build `ModelDims` from `self.model.config`; `SessionState::new(…, &self.device, self.dtype)?`; push; return the id. Reject at add time (never at step) any geometry a `Batched` policy would require to be uniform — for Increment 1 all sessions share the one `&LlamaModel`, so geometry is uniform by construction; validate only `max_new>0`/non-empty (delegated to `SessionState::new`).
- `step`:
  1. **Prefill pass (serial):** for each session with `phase == Prefill`, advance it once with its FULL `tokens` (the prompt): `model.forward_with_kv_context_persistent(&s.tokens, &mut s.cache, &mut s.ctx, &mut s.session)` → set `s.last_logits`, `s.phase = Decode`. (`seq>1` internally routes to the D1 prefill path per [lazy.rs:7110](../../../fuel-core/src/lazy.rs#L7110) — identical to `generate_streaming_with_kv_context`'s prefill.) **Sample immediately after prefill** (mirrors the streaming loop: prefill logits produce the first token) via `s.sample_and_append()`.
  2. **Decode ready set:** collect indices with `phase == Decode`.
  3. **Advance (serial for this task):** for each ready session, advance with its LAST token only: `model.forward_with_kv_context_persistent(&[*s.tokens.last().unwrap()], …)` → set `s.last_logits`; then `s.sample_and_append()`.
  4. Record `advanced` / `finished` in the `StepReport`.
- **Per-session isolation:** wrap each session's advance in a closure returning `crate::Result`; on `Err(e)`, set `s.phase = Finished`, push `(s.id, e.to_string())` into `report.errored`, and CONTINUE the other sessions. A per-session error is never propagated out of `step`. (Task 6 gates this.)
- `run_to_completion`: loop `step` while `!is_all_finished()`; return `(id, s.tokens.clone())` per session in insertion order.
- **Never-panic:** replace the `*s.tokens.last().unwrap()` sketch with a `.last().copied().ok_or_else(|| Error::Msg(...).bt())?` inside the per-session closure (an empty `tokens` is impossible after a validated prefill, but the production path must not `unwrap`).

**Steps:**
- [ ] Write the failing test — a single-session scheduler must equal the standalone `generate_with_kv_context`:
```rust
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
```
- [ ] Run RED: `cargo test -p fuel-core --lib multi_session::tests::scheduler_single_session` (FOREGROUND). Expect type-missing then a real assertion once it compiles — confirm the RED reason is "scheduler does not exist", not a helper typo.
- [ ] Implement `SchedulePolicy` / `StepReport` / `SessionScheduler` + `new` / `add_session` / `step` (serial) / `run_to_completion` / `is_all_finished`.
- [ ] Run GREEN.
- [ ] Commit: `feat(multi-session): SessionScheduler serial step + run_to_completion (C2)`.

**Deliverable:** a K=1 scheduler reproduces the standalone generate byte-for-byte — the serial substrate + the oracle harness the parity tests build on.

---

## Task 4 — T1: no cross-session contamination (C2, the headline gate)

**Files:** Modify `fuel-core/src/multi_session.rs` (test only). No production change expected — this task is a pure gate on Task 3. If it fails, the bug is in Task 3's isolation and is fixed here.

**Steps:**
- [ ] Write the failing test:
```rust
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
```
- [ ] Run RED: `cargo test -p fuel-core --lib multi_session::tests::t1_no_cross_session` (FOREGROUND). It should FAIL only if Task 3 leaks state across sessions (shared cache/ctx/rng); otherwise it passes immediately, which is the acceptance signal — note in the commit that it was born-green against a correct Task 3 and document why (independent `KvCache` + independent `rng_state`).
- [ ] If RED for a real reason: fix the isolation in `step`/`SessionState` (each session must own its `cache`/`ctx`/`session`/`rng_state`; nothing shared).
- [ ] Run GREEN.
- [ ] Commit: `test(multi-session): T1 no cross-session contamination (K=2 == 2 standalone)`.

**Deliverable:** proof that K scheduled sessions equal K standalone runs — the increment's correctness headline.

---

## Task 5 — T2 interleave-order invariance + T3 per-session RNG independence (C2)

**Files:** Modify `fuel-core/src/multi_session.rs` (tests; production change only if a gate fails).

**Interfaces — Consumes:** `SamplingStrategy::Temperature { temp, seed }` ([lazy.rs:8256](../../../fuel-core/src/lazy.rs#L8256)) — the seed is the per-session RNG seed set by `SessionState::new`.

**Steps:**
- [ ] Write the failing tests:
```rust
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
```
- [ ] Run RED: `cargo test -p fuel-core --lib multi_session::tests::t2_ multi_session::tests::t3_` (FOREGROUND). (If both pass immediately against a correct Task 3, that is the acceptance signal — note it.)
- [ ] Fix only if a gate fails (a shared/global RNG or shared prefill state would surface here).
- [ ] Run GREEN.
- [ ] Commit: `test(multi-session): T2 order-invariance + T3 per-session rng independence`.

**Deliverable:** independence of scheduling order and of RNG streams, locked.

---

## Task 6 — T4 session isolation on error + T7 mid-run add lifecycle (C2)

**Files:** Modify `fuel-core/src/multi_session.rs`.

**Interfaces — Produces (test-support surface on the scheduler; keep minimal):**
```rust
impl<'m> SessionScheduler<'m> {
    /// Add a session whose FIRST advance is forced to error, for the
    /// isolation gate. Injects an impossible KV geometry so the model's
    /// forward returns Err on this session only. (Test-support; #[doc(hidden)].)
    #[doc(hidden)]
    pub fn add_poisoned_session_for_test(&mut self, prompt: &[u32], max_new: usize)
        -> crate::Result<SessionId>;
}
```
- The poison mechanism must produce a real per-session `Err` from the advance path WITHOUT a panic. Simplest grounded lever: after building the `SessionState`, corrupt its `cache` so `forward_with_kv_context_persistent` returns the typed `Err` at [lazy.rs:7178](../../../fuel-core/src/lazy.rs#L7178) (`cache n_layers != model`) — e.g. `s.cache.layers.truncate(0)` makes `cache.n_layers()==0 != cfg.n_layers`, which the forward rejects with `Error::Msg`. Do NOT panic; the scheduler catches the `Err` into `report.errored`.
- **T7 mid-run add:** `add_session` while others are mid-decode must start `Prefill` and join the ready set on the next `step` without disturbing the others. This already holds if `step`'s Prefill pass keys purely on `phase == Prefill` — the test proves it.

**Steps:**
- [ ] Write the failing tests:
```rust
    #[test]
    fn t4_session_isolation_on_error() {
        let model = tiny_model(9999);
        let good = [1u32,2,3];
        let mut sched = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin);
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

        let mut sched = SessionScheduler::new(&model, Device::cpu(), DType::F32, SchedulePolicy::RoundRobin);
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
```
- [ ] Run RED: `cargo test -p fuel-core --lib multi_session::tests::t4_ multi_session::tests::t7_` (FOREGROUND). Expect `add_poisoned_session_for_test` missing (compile RED), then real assertions.
- [ ] Implement `add_poisoned_session_for_test`; confirm `step`'s per-session catch records `errored` and continues; confirm the Prefill pass keys on `phase`.
- [ ] Run GREEN.
- [ ] Commit: `test(multi-session): T4 error isolation + T7 mid-run add lifecycle (C2)`.

**Deliverable:** one session's failure cannot kill the batch; a session added mid-run prefills and joins cleanly. C2 (serial substrate + full parity oracle) is complete.

---

## Task 7 — C3 seam: `BatchOutcome` + uniformity gate + `Batched` policy fallthrough (T6, CPU)

**Files:** Modify `fuel-core/src/multi_session.rs`.

**Interfaces — Produces:**
```rust
pub enum BatchOutcome {
    /// N logits vectors, one per input session (same order as the slice).
    Advanced(Vec<Vec<f32>>),
    /// Ready set not uniform enough to batch — a NORMAL control value, not Err.
    NotBatchable,
}

pub(crate) struct BatchDescriptor {
    pub cached_len: usize,     // must be EQUAL across the batch (one shared k_len)
    pub max_seq_len: usize,
    pub n_layers: usize,
    pub cache_dtype: fuel_ir::DType,
}

/// Pure gate: are these sessions batchable together THIS step? (T6 unit-tests this
/// without CUDA by feeding descriptors.)
pub(crate) fn batch_uniform(descs: &[BatchDescriptor]) -> bool;

impl BatchedDecode {
    /// Live batched step. Task 7 ships the SIGNATURE + gate + the
    /// NotBatchable path only; Task 8 wires the live flash_decoding arm
    /// inside the `Advanced` branch.
    pub(crate) fn try_batched_step(
        model: &crate::lazy::LlamaModel,
        device: &fuel_ir::Device,
        dtype: fuel_ir::DType,
        sessions: &mut [&mut SessionState],
    ) -> crate::Result<BatchOutcome>;
}
pub(crate) struct BatchedDecode;
```
- `batch_uniform`: `false` if fewer than 2 descs; else all fields equal to `descs[0]` (crucially `cached_len` — the single shared `flash_decoding` `k_len` at [attention.rs:1006](../../../fuel-cuda-backend/src/baracuda/attention.rs#L1006)).
- `try_batched_step` (Task 7 body): build a `BatchDescriptor` per session (from `s.cache.cached_len`, `s.cache.max_seq_len`, `s.cache.n_layers()`, `s.cache.dtype`); if `!batch_uniform(&descs)` return `Ok(BatchOutcome::NotBatchable)`. Otherwise (Task 7) ALSO return `NotBatchable` with a `// Task 8: live batched arm goes here` marker — the seam exists; the live arm is Task 8. This keeps Task 7 CPU-shippable and correct (the scheduler always has a working serial fallback).
- **Wire `SchedulePolicy::Batched { max_batch }` into `step`:** when policy is `Batched`, after the Prefill pass, take up to `max_batch` `Decode`-ready sessions, borrow them `&mut`, call `BatchedDecode::try_batched_step`; on `Advanced(logits)` assign each `s.last_logits` from its slot and set `report.used_batched_arm = true`; on `NotBatchable` fall through to the existing serial advance. Either way, `sample_and_append` each afterward. **The serial arm remains reachable for every session** (non-uniform ready set, or `max_batch` overflow).

**Steps:**
- [ ] Write the failing tests (CPU — T6 gate + fallthrough equivalence):
```rust
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
            let mut s = SessionScheduler::new(&model, Device::cpu(), DType::F32, policy);
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
```
- [ ] Run RED: `cargo test -p fuel-core --lib multi_session::tests::t6_` (FOREGROUND). Expect missing types (compile RED).
- [ ] Implement `BatchOutcome` / `BatchDescriptor` / `batch_uniform` / `BatchedDecode::try_batched_step` (stub `Advanced` path → `NotBatchable` for now) + the `Batched` policy wiring in `step`.
- [ ] Run GREEN.
- [ ] Commit: `feat(multi-session): C3 seam — BatchOutcome + uniformity gate + Batched fallthrough (T6)`.

**Deliverable:** the batched seam + uniformity gate exist and are unit-tested; a `Batched` scheduler is provably equal to `RoundRobin` while the live arm is unwired — the safe base the live arm slots into.

---

## Task 8 — C3 live batched arm: shared `[K,…]` KV buffer + `flash_decoding` batch wiring (T5)

**Files:** Modify `fuel-core/src/multi_session.rs`. Reference mirror: `LlamaModel::build_and_realize_first_decode_token` ([lazy.rs:7159-7375](../../../fuel-core/src/lazy.rs#L7159)) and `apply_layer_with_kv_writes` ([lazy.rs:6635](../../../fuel-core/src/lazy.rs#L6635)) — both already thread a leading `batch` dim (`batch = dims[0]`, [lazy.rs:6651](../../../fuel-core/src/lazy.rs#L6651)).

**Interfaces — Produces:** the live `Advanced` branch inside `BatchedDecode::try_batched_step` (signature unchanged from Task 7). No public API change.

**Design (the hard part — spec §C3 + risk #2 mechanism (a), copy-in/copy-out, all-or-nothing commit):**

The projection GEMMs batch for free — stacking N sessions on a leading batch axis makes those `Op::Matmul` nodes carry a batch dim already served by `gemm_dense`'s per-slot loop ([gemm_dense.rs:38](../../../fuel-cuda-backend/src/baracuda/gemm_dense.rs#L38)). The attention half reaches `flash_decoding` (batch dim + per-tensor batch strides, [attention.rs:996-1020](../../../fuel-cuda-backend/src/baracuda/attention.rs#L996)) automatically via the optimizer-emitted flash arm ([decode_flash.rs:112](../../../fuel-dispatch/src/decode_flash.rs#L112)) — `DecodeFlashSpec` already models `q:[B,Hq,Sq,D]`, `k/v:[B,Hkv,capacity,D]`; a batch=K graph offers a batch-K flash node with ONE shared `k_len` (correct because the uniformity gate guarantees equal `cached_len`).

So the live arm builds a **batch=K analogue of `build_and_realize_first_decode_token`** over a shared KV buffer:

1. **Shared batched KV buffer.** Allocate, once per `try_batched_step` call (Increment 1; slot-ownership persistence is deferred), a `Vec<(Arc<RwLock<Storage>> /*K*/, Arc /*V*/)>` per layer of shape `[K, n_kv_heads, max_seq_len, head_dim]` using the SAME `Op::Alloc`→`Op::ZeroFill` emission + `PipelinedExecutor::realize_many` pattern as `KvCache::with_capacity` ([inference_context.rs:195-291](../../../fuel-core/src/inference_context.rs#L195)) — just leading dim `K` instead of `1`. Factor a `fn alloc_batched_kv(k: usize, dims: ModelDims, max_seq_len, dtype, device) -> Result<Vec<(Arc,Arc)>>` (copy the constructor body, change the shape's dim 0).
2. **Copy-in.** For each session `i`, copy its per-layer `KvCache` K/V history (`[1, Hkv, cap, D]`) into batch slot `i` of the shared buffer (`[i:i+1, …]`). A `Op::WriteSlice` at batch-offset `i` (start on axis 0), width 1, mirroring the `write_ranges` construction in `apply_layer_with_kv_writes` ([lazy.rs:6705](../../../fuel-core/src/lazy.rs#L6705)) but on axis 0. Since all sessions share `cached_len`, only `[.., 0:cached_len, ..]` carries live data; the zero tail is masked.
3. **Batch=K decode graph.** Build the decode graph exactly as `build_and_realize_first_decode_token` but: `batch = K`; `token_ids` placeholder `[K]` seeded with each session's last token; `h` reshaped `[K, seq, dim]`; RoPE cos/sin tables `[seq, head_dim]` (shared — same position for all K under lockstep, so ONE table broadcasts across batch); mask `[1,1,seq,max_seq_len]` (broadcasts across batch); per-layer KV placeholder Consts bound to the SHARED buffer's slot Arcs; `cached_len` offset = the shared `cached_len`. Slice logits at the last position → `[K, vocab]`.
4. **Realize** via `ctx.realize_one_as_with_env::<f32>(&graph, logits_node, &env)` where `env` binds `cached_len_sym = cached_len` and `attended_len_sym = cached_len + 1` (the flash `k_len`) — non-captured plan-once-per-step per spec #4. Use a fresh `InferenceContext` seeded with the shared KV Arcs.
5. **Scatter + all-or-nothing commit.** ONLY after realize succeeds: reshape the `[K, vocab]` logits into K `Vec<f32>` rows → `BatchOutcome::Advanced(rows)`; and copy slot `i`'s freshly-written K/V row (`[.., cached_len:cached_len+1, ..]`) back into session `i`'s own `KvCache` (copy-out) + bump each session's `cache.cached_len` / versions. If ANY step (2)–(4) returns `Err`, return that `Err` BEFORE any session's `KvCache` is mutated (no copy-out happened yet) — the scheduler forces the affected sessions to `Finished`-with-error; no session is left half-written (spec risk #7).

**On CPU** the flash arm is not offered (f32; `flash_decode_admissible` requires f16/bf16, [decode_flash.rs:175](../../../fuel-dispatch/src/decode_flash.rs#L175)) — the batch=K graph runs the DECOMPOSED batched attention. That still exercises the entire batched-graph + shared-buffer + scatter path and is the CPU parity gate below. The GPU test confirms the flash-arm batch specifically.

**Steps:**
- [ ] Write the failing CPU parity test (born-red; the primary Task-8 gate — batched f32 == serial f32):
```rust
    #[test]
    fn t5_cpu_batched_step_equals_serial_step() {
        // Two sessions, SAME prompt length, prefilled to equal cached_len,
        // then ONE batched decode step must equal one serial step each.
        let model = tiny_model(9999);
        let pa = [1u32, 2, 3];
        let pb = [4u32, 5, 6];

        // Serial oracle: prefill each, take one decode step, record logits.
        let serial_logits = |prompt: &[u32]| -> Vec<f32> {
            use crate::inference_context::{KvCache, InferenceContext};
            let cfg = &model.config;
            let msl = prompt.len() + 2;
            let mut cache = KvCache::with_capacity(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, msl, DType::F32, &Device::cpu()).unwrap();
            let mut ctx = InferenceContext::new(Device::cpu());
            let mut sess = None;
            let pre = model.forward_with_kv_context_persistent(prompt, &mut cache, &mut ctx, &mut sess).unwrap();
            let next = crate::lazy::sample_logits(&pre, SamplingStrategy::Greedy, &mut 0u64);
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
```
- [ ] Run RED: `cargo test -p fuel-core --lib multi_session::tests::t5_cpu_batched` (FOREGROUND). Expect `NotBatchable` panic (Task 7 stub) or logits mismatch until the live arm is built.
- [ ] Implement `alloc_batched_kv` + the live `Advanced` branch (steps 1–5). Reuse `LlamaModel`'s existing builders where possible — the cleanest path is to add a `pub(crate) fn build_batched_decode_logits(&self, states: &mut [&mut SessionState], device, dtype) -> Result<Vec<Vec<f32>>>` on `LlamaModel` in `lazy.rs` that mirrors `build_and_realize_first_decode_token` with `batch=K` over the shared buffer, and call it from `try_batched_step`. Keep the copy-out (step 5) in `try_batched_step` so the commit-ordering / all-or-nothing discipline lives at the batch boundary.
- [ ] Run GREEN (CPU): `cargo test -p fuel-core --lib multi_session::tests::t5_cpu_batched` (FOREGROUND).
- [ ] Write the GPU parity test (local-only, `#[ignore]`):
```rust
    #[test]
    #[ignore = "live-GPU: RTX 4070, run locally after CUDA build; one live suite at a time"]
    fn t5_gpu_batched_flash_equals_serial_bf16() {
        // Same structure as t5_cpu, but Device::cuda(0) + DType::BF16 so the
        // optimizer offers the flash_decoding batch arm. bf16 weights required
        // (mirror lazy.rs generate_tests::make_tiny_weights_bf16).
        // Assert batched rows == serial rows within a sabotage-calibrated ε
        // (batched GEMM reduction order may differ from serial — see
        // [[sabotage-test-calibration]]). Skeleton mirrors t5_cpu; swap
        // Device/DType, build a bf16 model, and set ε from a measured
        // serial-vs-serial bf16 drift baseline (start 5e-3, tighten to the
        // measured floor).
        // (Full body: adapt t5_cpu with cuda device + bf16 model.)
    }
```
- [ ] Run the GPU test locally (optional/local): build per the CUDA recipe above, then `cargo test -p fuel-core --features cuda --lib multi_session::tests::t5_gpu_batched -- --ignored --nocapture` (FOREGROUND, VS dev shell + cuDNN on PATH). Confirm the flash arm is picked (add a temporary eprintln of the chosen arm if needed) and rows match within the calibrated ε. **Calibrate ε against a passing sabotage run** ([[sabotage-test-calibration]]): confirm the test FAILS if you perturb one session's KV, and PASSES on the true path — a passing sabotage run is invalid without confirmed recompilation.
- [ ] Commit: `feat(multi-session): C3 live batched arm — shared [K,..] KV buffer + flash_decoding batch (T5)`.

**Deliverable:** a live batched decode step, byte-equal to serial on CPU (f32) and equal within a calibrated ε on CUDA (bf16, through the `flash_decoding` batch dim). Increment 1 complete.

---

## Self-review — spec coverage

| Spec component | Task(s) |
| --- | --- |
| C1 `SessionState` bundle (KvCache+InferenceContext+DecodeSession?+rng/tokens/pos), Llama-first trait-shaped (`ModelDims`) | 1, 2 |
| C1 `new` (validation, OOM-propagating `KvCache::with_capacity`, seed Prefill), `is_ready` | 1 |
| C1 `sample_and_append` (per-session RNG firewall, eos/budget → Finished) | 2 |
| C2 `SessionScheduler` (add_session, step, run_to_completion), serial arm = oracle | 3 |
| C2 `StepReport` + session-level error isolation (never-panic) | 3, 6 |
| C2 `SchedulePolicy` (RoundRobin default; Batched opt-in) | 3, 7 |
| T1 no cross-session contamination (K sched == K standalone) | 4 |
| T2 interleave-order invariance; T3 per-session RNG independence | 5 |
| T4 session isolation on error; T7 prefill-then-decode mid-run lifecycle | 6 |
| C3 uniformity gate + `BatchOutcome::NotBatchable` seam (T6, CPU) | 7 |
| C3 live batched arm: shared `[K,Hkv,cap,D]` buffer + `flash_decoding` batch, lockstep-only, all-or-nothing commit, fail-on-OOM | 8 |
| T5 batched == serial parity (CPU f32 gate + CUDA bf16 local-only) | 8 |
| Boundaries: no KV sharing/splice, no PagedAttn block-pool, no residual donation, no admission/preemption | out of scope by construction (each session owns private KV; no shared-content path built) |

**Sequencing check:** the serial substrate + full parity oracle (Tasks 1–6) land before the batched arm (Tasks 7–8), so the oracle exists first; Task 7 ships a safe `NotBatchable`-only seam proven equal to serial before Task 8 wires the live flash arm. Every earlier deliverable is the oracle for the next (Task 3's standalone-parity → T1's contamination gate → T5's batched-vs-serial).

**Type-name consistency:** `SessionState`, `SessionId`, `SessionPhase`, `ModelDims`, `SessionScheduler`, `SchedulePolicy`, `StepReport`, `BatchOutcome`, `BatchDescriptor`, `BatchedDecode`, `batch_uniform` — used identically across tasks. Module `fuel-core/src/multi_session.rs` (NOT `scheduling`, which exists). All new surfaces return `crate::Result`; no `.unwrap()`/`.expect()` on production paths (tests excepted).

**Judgment calls made (flagged):**
1. **Module name `multi_session`** — the spec says "lives in fuel-core"; `scheduling` was taken, so I chose `multi_session` to avoid the collision.
2. **C3 KV presentation = copy-in/copy-out into a per-call shared buffer** (spec risk #2 mechanism (a)), NOT sessions-allocated-into-slots. This keeps each session's own `KvCache` the source of truth so the serial fallback stays byte-consistent, and makes the all-or-nothing commit trivial (mutate session caches only after realize succeeds). The slot-ownership / persistent-shared-buffer optimization (avoiding per-step copies) is a throughput follow-up, explicitly deferred — correctness is the Increment-1 gate, throughput is a bench.
3. **Added a CPU f32 batched-vs-serial parity test (`t5_cpu`) as Task 8's primary gate**, with the CUDA bf16 flash-arm test kept `#[ignore]`/local-only. The spec's T5 is CUDA-only; a CPU gate makes the batched-graph builder + shared-buffer + scatter machinery TDD-verifiable without a GPU (the flash arm is the only CPU-absent piece), which the "serial is the oracle" principle supports.
4. **Prefill produces the first token in the same `step`** (sample immediately after the prefill forward), mirroring `generate_streaming_with_kv_context`'s loop where prefill logits yield the first sampled token. This keeps scheduled output byte-identical to the standalone generate the oracle uses.
5. **`add_poisoned_session_for_test` corrupts `cache.layers` to force the model's existing `n_layers` mismatch `Err`** ([lazy.rs:7178](../../../fuel-core/src/lazy.rs#L7178)) — a real typed error through the true advance path, no panic, no new production error branch just for testing.

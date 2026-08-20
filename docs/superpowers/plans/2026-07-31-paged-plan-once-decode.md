# Paged plan-once decode — Implementation Plan

> Executed INLINE (superpowers:executing-plans) by the author, who has full context. Steps are TDD; each task ends independently testable. Line numbers are `fuel` @ the branch tip — GREP the symbol, don't trust the line.

**Goal:** Make paged decode build its graph + optimized plan ONCE and reuse it across tokens (a `PagedDecodeSession`, the paged twin of the contiguous `DecodeSession`), behind a runtime flag — targeting the ~90% of per-token paged cost that is the planner (Lightbulb: plan≈4s vs execute≈0.95s; paged pays plan+execute every token).

**Architecture:** Mirror `DecodeSession` (inference_context.rs) + its `prebuild_optimized_capturing…`/`realize_one_prebuilt_env` machinery (pipelined_bridge.rs) for the paged forward. The map found ONLY TWO structural variabilities that force a re-plan: (1) the fresh `Tensor::from_f32` graph root per call, and (2) the L-varying `block_table` shape (`max_blocks = max(row.len())`). Everything else is shape-stable data (rebind via the existing `const_placeholder_like` + `StorageCache::insert` precedent already in `forward_paged_step`) or a concrete write offset (symbolize via the existing `write_slice_dyn`/`write_slice_doff` precedent). `Op::PagedAttn` already tolerates a fixed-capacity padded block_table + the `context_len` mask (confirmed in the decompose recipe: score width `max_blk*block_size`, mask `Ge(pos, context_len)→−inf`).

**Tech stack:** Rust, fuel-core lazy DAG. Reuses: `DecodeSession`, `prebuild_optimized_capturing_as_with_env`, `realize_one_prebuilt_env`, `const_placeholder_like`, `StorageCache`, `write_slice_dyn`/`write_slice_doff`, `SymId`/`SymEnv`.

## Global constraints
- Single cargo invocation at a time; `-p fuel-core` only; never workspace-wide.
- TDD: born-red observed before green. Never-panic (Result). f32/CPU-verifiable throughout (no GPU needed — the win is CPU-side planning).
- Commit-producing work in the `fuel-kv-alloc` worktree; push to origin/main after re-fetch.
- **Plan-once must land as ONE commit with a clean parent** (Lightbulb re-pins fuel-lightbulb-port to parent=before, commit=after; a fresh 4c2e9407-free base avoids the ragged confound).

## Test design (Lightbulb-hardened — the gate must have teeth)
The correctness gate is flag-toggled and runs BOTH arms in one process:
- **plan-once arm:** assert plan **HIT** + byte-identical to the control (catches flag-off).
- **control (re-planning) arm:** assert plan **MISS** (catches flag stuck ON — the inversion where a control secretly running plan-once passes identity self-vs-self, passes HIT, and reads ~1.0× = "doesn't help", a false negative that would retire the feature).
- **mutation check** on each guard: force the opposite (stub the cache to always-miss / always-hit) → confirm the test goes RED. Proves the assertions have teeth before relying on them.
- **ragged coverage:** staggered positions (the 4c2e9407 batched work) — where a cached plan is *wrong* not merely rebuilt; the lockstep-uniform sweep won't exercise it.
- Perf (Lightbulb runs, single harness, parent-vs-commit): warm-up/steady ratio must move above 1.0 (instrumentation-independent). Never compute plan/execute as a residual (tautology) — use a difference of two direct measurements.

---

### Task 1 — Fixed-capacity (padded) block_table

**Files:** Modify `fuel-core/src/kv_block_pool_device.rs` (`materialize_block_table` / `PageTableHost`). Test: same file's `tests`.

**Why:** removes structural variability #2. Today `materialize_block_table` sets `max_blocks = rows.iter().map(|r| r.len()).max().max(1)` = f(L). Add a capacity-padded variant so the `[B, max_blocks]` shape is constant across steps. Padded entries are 0 (in-bounds); the `context_len` mask neutralizes them.

**Interface produced:** `DeviceKvPool::materialize_block_table_padded(&self, sessions: &[SessionHandle], max_blocks_cap: usize) -> Result<PageTableHost, KvAllocError>` — same as `materialize_block_table` but every row is right-padded with 0 to width `max_blocks_cap` (errors if any session already occupies `> max_blocks_cap` blocks). `PageTableHost.max_blocks == max_blocks_cap`.

- [ ] **Step 1 (RED):** test `padded_block_table_paged_attn_matches_unpadded` — build a pool, one session at ~sk tokens (occupies N<CAP blocks), compute `paged_attn` two ways: (a) `materialize_block_table` (`[B,N]`), (b) `materialize_block_table_padded(.., CAP)` (`[B,CAP]`, 0-padded) with the SAME `context_lens`; assert the two realized outputs are byte-identical (the mask neutralizes the pad). Also assert `pt.block_table_shape().dims() == [B, CAP]`.
- [ ] **Step 2:** run → RED (method missing).
- [ ] **Step 3 (GREEN):** implement `materialize_block_table_padded` (reuse `materialize_block_table`'s row projection; pad each row to `max_blocks_cap`; set `max_blocks = max_blocks_cap`; error on overflow).
- [ ] **Step 4:** run → GREEN. Regress the existing `materialize_block_table` tests.
- [ ] **Step 5:** commit `feat(kv-pool): capacity-padded block_table (paged plan-once prereq)`.

---

### Task 2 — Symbolic K/V write offset (graph-stable pool write)

**Files:** Modify `fuel-core/src/kv_block_pool_device.rs` (`build_decode_attn` / `build_decode_attn_batched` write path). Test: same file.

**Why:** removes the per-step write-range variability. Today `build_decode_attn` writes the new K/V at CONCRETE `slot_ranges = [(p,p+1),(slot,slot+1),(0,Hkv),(0,D)]` via plain `write_slice`, so the write op changes every step. Reshape the pool buffer `[num_blocks, block_size, Hkv, D] → [num_blocks*block_size, Hkv, D]`, write `[1,Hkv,D]` at flattened offset `linear = phys*block_size + slot` via `write_slice_doff` (device offset scalar) / `write_slice_dyn` (SymEnv), then the `[num_blocks,block_size,…]` view feeds `paged_attn`. Offset is a single DynScalar bound per step.

**Interface produced:** a `build_decode_attn` variant taking the write offset as a bound scalar (device `Tensor` offset or `SymId`), mirroring `apply_layer_with_kv_writes`'s `write_slice_doff`/`write_slice_dyn` split (lazy.rs ~:7404/:7409).

- [ ] **Step 1 (RED):** test `symbolic_offset_write_matches_concrete_write` — write the same K/V into the same (phys,slot) two ways (concrete `write_slice` vs flattened symbolic offset), realize the post-write pool block via `read_block_bytes`, assert byte-identical. Then the full `build_decode_attn` output byte-identical.
- [ ] **Step 2:** run → RED.
- [ ] **Step 3 (GREEN):** implement the flattened symbolic-offset write; verify the reshape composes with the pool-buffer placeholder binding (the buffer is mutated in place).
- [ ] **Step 4:** run → GREEN + regress the existing decode-attn parity tests.
- [ ] **Step 5:** commit `feat(kv-pool): symbolic K/V write offset (paged plan-once prereq)`.

---

### Task 3 — `PagedDecodeSession` + `forward_paged_step_persistent` (build-once / rebind)

**Files:** Modify `fuel-core/src/lazy.rs` (new `forward_paged_step_persistent`, build-once + rebind helpers) + `fuel-core/src/inference_context.rs` (new `PagedDecodeSession`, mirroring `DecodeSession` :778). Test: `lazy.rs` tests.

**Why:** removes structural variability #1 (fresh root) — the ~90% lever. Mirror `DecodeSession`: build the paged decode graph ONCE with `const_placeholder_like` data nodes (token_ids, rope tables, block_table[B,CAP], context_lens) + the symbolic write offset, call `prebuild_optimized_capturing_as_with_env` to cache the `OptimizedGraph` + `base_cache`, store a `PagedDecodeSession`; per subsequent token recompute the per-token Arcs + bind the write-offset scalar + `realize_one_prebuilt_env`.

**Interface produced:** `PagedDecodeSession { graph, optimized: OptimizedGraph, effective_target, logits_node, token_ids_node, rope_*_node, block_table_node, context_lens_node, kv_nodes, write_offset_node/sym, base_cache, validity keys }` + `LlamaModel::forward_paged_step_persistent(&self, token, pool, session_handle, decode_session: &mut Option<PagedDecodeSession>) -> Result<Vec<f32>>` (seq==1; invalid/absent → build-once; valid → rebind; `TopologyChanged` → drop + fall back to `forward_paged_step`).

- [ ] Steps: RED (byte-identity test deferred to Task 4's gate — Task 3's own test asserts a second token reuses the plan: `plan_once_second_token_reuses_graph` via a graph-node-count / HIT counter on the session). Implement build-once then rebind. GREEN. Commit `feat(paged-decode): PagedDecodeSession — build graph + plan once, reuse per token`.
- Note: batched (`forward_paged_step_batched`, ragged) persistent is a follow-on; Task 3 does single-session (`forward_paged_step`) first — it's the B=1 5.58× case and the simplest correct target.

---

### Task 4 — Runtime flag + teeth-bearing correctness gate — **COMPLETE**

**Files:** Modify `lazy.rs`/`inference_context.rs` (flag on the persistent path + a plan-HIT counter on the session). Test: `lazy.rs` tests.

**Why:** the gate is the deliverable that makes the win trustworthy. Flag toggles plan-once vs re-planning in ONE process.

**Shipped design:**
- **Flag** = `PagedDecodePlan { Replan, PlanOnce }` (inference_context.rs), a new param to `forward_paged_step_persistent`. `Replan` (the driver default, off) drops any held session and delegates to `forward_paged_step` — the exact production toggle Task 5 wires, and safe across a flip (no stale session lingers). `PlanOnce` builds-once / rebinds.
- **Plan-HIT counter** = `PagedDecodeSession.realize_count: AtomicUsize`, bumped in `realize_token` (the rebind seam; the build realizes via prebuild, NOT `realize_token`, so it stays 0 after build and +1 per reuse). DEVICE-INDEPENDENT — unlike the `optimize_calls_thread_local` delta, which is CPU-scoped (GPU per-token uploads bump it). The gate asserts BOTH per step.
- **Teeth via a non-panicking checker:** the invariants live in `check_paged_gate(&report) -> Result<(),String>`; the primary test `unwrap`s it (relies on those exact checks), the mutation test asserts the SAME checker goes `Err` under a flipped flag — no `catch_unwind`/global-panic-hook fiddling under the concurrent suite.

- [x] `plan_once_paged_matches_replanning_with_hit_asserted` — decodes 4 tokens crossing a block boundary, plan-once (A) vs `Replan` control (B) in one process off identical primed pools; asserts byte-identity + HIT (`opt_delta_a==0` **exactly** AND one session rebind, every step after the first) + control MISS (`opt_delta_b>=1` every step). Per-step exact-zero HIT is strictly stronger than the contiguous `==2` aggregate.
- [x] Mutation check `plan_once_gate_has_teeth` — flips arm A to `Replan` (force always-miss → HIT invariant breaks) and arm B to `PlanOnce` (force always-hit → MISS invariant breaks); asserts the shared `check_paged_gate` returns `Err` for each. **Sabotage-calibrated:** confirmed with recompilation that a toothless `check_paged_gate` (forced `Ok`) makes THIS test fail (the primary passes vacuously — so the teeth test is what actually guards the checker).
- [x] `plan_once_paged_ragged_matches_replanning` — single-session (B=1) staggered histories {2,5,3} → different absolute positions / block-slot phases (mid-block, boundary-adjacent) the lockstep primary gate can't reach; each satisfies the full gate. (Batched-ragged persistent is a Task-3-batched follow-on.)
- [x] Audit — **FINDING: no hole.** The contiguous `generate_loop_persistent_byte_exact_and_plans_once` (lazy.rs) already carries the plan-HIT assertion in the **exact-delta** form (`opt_after - opt_before == 2`, i.e. 1 prefill-fallback + 1 decode-build, NOT `<= N`); the Phi/DeepSeek twins mirror it. So there is no latent `<=N` weakness to fix; the paged gate matches that exact-delta rigor (and tightens it to per-step).
- [x] Commit `test(paged-decode): flag-gated plan-once correctness gate w/ HIT+MISS guards + mutation check`.

**Verification:** the 3 new tests + Task 3's `plan_once_second_token_reuses_graph` all green; full `paged/persistent/kv_block_pool` regression **48 passed / 0 failed** (2 live-GPU ignored). CPU f32.

---

### Task 5 — Wire + measure — **COMPLETE (wiring); measurement handed to Lightbulb**

**Files:** Modified `fuel-inference/src/multi_session.rs` (the `PagedSessionScheduler` paged driver).

**Shipped wiring:**
- **`DecodeModel::forward_paged_step_persistent`** — new trait method (thin `LlamaModel` forward), the plan-once sibling of `forward_paged_step`; carries `max_blocks_cap` + `plan` + `&mut Option<PagedDecodeSession>`.
- **`PagedSession.decode_session: Option<PagedDecodeSession>`** + **`max_blocks_cap`** — per-session held plan (the paged twin of `SessionState.session: Option<DecodeSession>`). `max_blocks_cap = ⌈(prompt+max_new)/block_size⌉` computed at `add_session` from the live pool geometry.
- **`PagedSessionScheduler.plan: PagedDecodePlan`** — default `Replan` (off). `set_plan(PlanOnce)` opts in; `plan()` reads it back.
- **`decode_one`** routes through `forward_paged_step_persistent` (flag-gated; `pool` and the session's `decode_session` are disjoint fields → both borrow mutably in one call). Prefill stays on the plain `forward_paged_step` (the plan builds on the FIRST decode token, exactly the Task-3/4 gate flow); the batched arm (`decode_batch`, B=K) stays on `forward_paged_step_batched` (persistent is B=1). Interleaving serial↔batched is safe: the rebind recomputes block_table/context_lens/offset from live pool state each token; only the stable big-buffer K/V Arc is captured.
- **`evict_session`** defensively drops the held plan (restore re-allocates fresh blocks; one rebuild on resume is negligible against the host round-trip).
- **`session_realize_count(id) -> Option<usize>`** — observability/test hook: how many times a session's held plan was rebound (`None` if no plan held).

**Test (TDD, RED→GREEN→sabotage-calibrated):** `paged_scheduler_plan_once_matches_replan_byte_exact` — decodes one session both ways in one process (default `Replan` reference vs `set_plan(PlanOnce)`); asserts byte-identical token stream + the held plan was rebound (`session_realize_count ≥ 1`, via `.expect` that panics if no plan persisted). Two complementary teeth: byte-identity catches a result-changing mis-wire; the rebind count catches the "flag set but persistent path silently bypassed" inversion (reads as no-speedup → would wrongly retire the feature). **Sabotage-calibrated** (confirmed recompile): forcing `decode_one` to pass `Replan` regardless of the flag makes the test FAIL at the rebind-count `.expect`.

- [x] Wire `forward_paged_step_persistent` into the paged decode driver behind the flag; default off, opt-in on.
- [x] Commit + push (see below).
- [ ] **Lightbulb measurement (handed off):** with the flag now reachable, the cleanest A/B is flag-off vs flag-on **at the same commit** (`set_plan(Replan)` vs `set_plan(PlanOnce)`), which removes any parent-vs-commit confound from intervening commits. Harness knobs unchanged (single session, prompt8/steps4/BLOCK16/max_blocks==1). Success = paged warm-up/steady ratio moves above 1.0 and per-token cost drops toward the ~950ms execution floor.

**Verification:** the new test + full `multi_session` regression **28 passed / 0 failed** (1 live-GPU ignored). CPU f32.

## Self-review notes
- Spec coverage: two structural variabilities (Task 1 block_table shape, Task 3 fresh root) + the write-offset (Task 2) + rebind (Task 3) + flag/gate (Task 4) + wire/measure (Task 5). Ragged batched persistent is explicitly a follow-on after single-session (Task 3 note).
- Open design detail: `max_blocks_cap` source (Task 1) = the session's decode capacity `ceil(max_seq_len/block_size)`; the persistent forward passes it (the driver knows prompt+max_new). Confirm during Task 1/3.
- Risk: the flattened symbolic write offset (Task 2) composing with in-place pool-buffer mutation is the subtlest piece — a byte-identity read-back test gates it directly.

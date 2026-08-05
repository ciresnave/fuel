# Prefix-sharing serving API (rung-1) — Implementation Plan

> Executed INLINE (superpowers:executing-plans) by the author, who has full context. Steps are TDD; each task ends independently testable. Line numbers are `fuel` @ the branch tip — GREP the symbol, don't trust the line.

> **STATUS (2026-08-05) — ALL TASKS IMPLEMENTED + VERIFIED; push pending a peer.** Tasks 1–3 landed on `origin/main` earlier (`496d581d`, `084e2bfd`). Tasks 4 (`add_session_sharing_prefix` + `register_prefix`/`release_prefix` reference callers) and 5 (adversarial verification) are committed on branch `feat/prefix-parity` in worktree `../fuel-crash-vmm`; `fuel-inference --lib` 185/0. The push is HELD until peer `kblt7uwd` lands their `PagedSessionScheduler::new` default flip (`Replan` → `PlanOnce`, disjoint region), then rebase + re-verify + push.
>
> **KEY DEVIATION — the planned oracle was vacuous; the test asserts LOGITS, not tokens.** Task 1/4's spec said "byte-identical tokens." That proves nothing: `tiny_model`'s greedy argmax is a sticky constant and even seeded temperature swallows the signal — a direct `forward_paged_step` experiment measured suffix-only-prefill vs from-scratch = maxdiff **0.0 exactly**, but a real mis-positioning bug (re-feed the whole prompt) = only **~1.7e-3**, below the token-sampling threshold on both greedy and temperature. The tests capture per-step logits (new opt-in `capture_logits` hook, sibling of `session_realize_count`) and assert **maxdiff == 0**. Mutation-verified with teeth (`prefill_start = 0` → logit maxdiff ~1e-2, tokens unchanged). Also verified byte-exact under `PlanOnce`. Recorded in `tiny_model`'s doc note + memory `sabotage-test-calibration`.
>
> **Task 5 adversarial coverage (scheduler-level; pool-level CoW/refcount/refusal already in Tasks 2/3):** `two_sharers_of_one_prefix_decode_independently` (multi-tenant isolation — the serving value), `prefix_owner_release_while_sharer_live_keeps_it_correct` (owner release mid-flight, refcount keeps blocks alive), `add_session_sharing_prefix_refusal_leaks_nothing` (transactional rollback — teeth: reap+release must return the pool to `num_blocks`; mutation-verified by removing the rollback `discard`).

**Goal:** Let a new paged decode session reuse an already-computed KV prefix (e.g. a shared system prompt) via refcounted zero-copy splice — computing the prefix ONCE and reading it from many concurrent sessions — behind a reusable, **transactional** helper (the product) with a `PagedSessionScheduler` reference caller, plus a named refcounted `PrefixId` owner whose lifetime the registry controls.

**Architecture:** Build on the complete `KvBlockPool` splice / `cow_break` / refcount substrate (already tested at the pool + model layer; only the consumer surface is missing). Three layers, bottom-up:
1. **`PrefixId` registry (primary primitive, pure core).** A prefix *owner* — a `SessionHandle` that holds shared blocks and never decodes — minted from a filled session's block range. Its own refcount keeps the blocks alive even if the original donor is discarded, so a consumer's prefix index references something whose lifetime *it* controls (deletes the donor-teardown race).
2. **Transactional `splice_prefix` helper (pure core).** Validates block-alignment + projects the post-splice fill and refuses **before** any mutation; returns the shared token count so the "prefill only the suffix" contract is a **Fuel-enforced invariant**, not a per-consumer convention; reports allocator **facts** (`freed`/`still_shared`/shared-tokens), never conclusions.
3. **`PagedSessionScheduler::add_session_sharing_prefix` (fuel-inference reference).** A thin caller of the helper that splices then prefills only the suffix through the existing paged forward.

**Tech stack:** Rust, fuel-core lazy DAG + fuel-inference. Reuses `KvBlockPool::{splice, cow_break, open, append, discard, evict_blocks, session_block_table, block_refcount, filled_tokens}`, `DeviceKvPool::{ensure_writable_block, materialize_block_table_padded, geometry}`, `PagedSessionScheduler`, `LlamaModel::forward_paged_step`.

## Global constraints
- Single cargo invocation at a time; `-p fuel-core` / `-p fuel-inference` only; never workspace-wide.
- TDD: born-red observed before green. Never-panic (Result). f32/CPU-verifiable throughout.
- Commit-producing work in the `fuel-kv-alloc` worktree; branch from PUSHED origin/main; re-fetch right before push.
- rung-1 only (same absolute positions — the shared-system-prompt case, prefix at position 0). rung-2 (RoPE delta-rotation for position-shifted / mid-prompt donation) is a **separate** op, scoped in the background, sequenced "next" (confirmed with Lightbulb — rung-1 covers every case they serve today).

## Design invariants (from the Lightbulb co-design — these have teeth, honor them)
- **Transactional guard:** validate alignment + project post-splice fill `(blocks*block_size).min(donor_filled)` **before** `splice`. A refused splice mutates NOTHING — no half-spliced `dst` with bumped refcounts and no unsplice. **Test the POOL STATE after refusal, not the returned `Err`** — the Err-asserting test passes on the broken (validate-after-splice) implementation because the guard fires, just too late; only the post-refusal state assertion catches the damage. Mutation-check it (move validation after splice → the state test must go red, with confirmed recompilation).
- **`still_shared` exactness (silent-corruption trap):** a prefix-owned block reports `still_shared` while ANY session references it, AND when the ONLY remaining reference is the prefix owner itself. An owner-only block reporting `freed` would let a consumer's report-reconciliation treat those tokens as reclaimed and re-prefill over a live shared prefix.
- **Facts, not conclusions:** the helper reports `freed`/`still_shared`/shared-tokens; it never concludes "span alive." That stays consumer policy (§15).
- **Seam:** the pure-core helper + registry is the product (Lightbulb drives the pool directly and won't adopt the scheduler); the scheduler method is the reference caller. Adoption by Lightbulb is the user's call, not assumed.

---

### Task 1 — Seam-invariant correctness foundation (born-red, through the new API)

**Files:** Test in `fuel-core/tests/paged_decode_parity.rs` (mirrors `paged_decode_into_spliced_prefix_does_not_corrupt_donor`). Depends on Task 2/3's helper API existing (so it is red until they land — write it first as the spec).

**Why:** The existing splice test only proves the *donor* isn't corrupted. The actual serving-value claim — a prefix-shared session decodes IDENTICALLY to computing the whole prompt from scratch — is untested. This is the correctness anchor, and it goes THROUGH the new helper (not hand-rolled pool calls) so it is a real born-red test of the product.

- [ ] **Step 1 (RED):** `prefix_shared_session_decodes_like_from_scratch` — build a `DeviceKvPool` + `LlamaModel`; session A prefills a system-prompt prefix (fills N whole blocks); mint a `PrefixId` from A's first N blocks (Task 2); a sharer session splices the prefix (Task 3 helper) + prefills ONLY `prompt[N*block_size..]` + decodes K tokens; assert its logits/tokens ε-equal a from-scratch session that prefilled the FULL prompt and decoded K tokens. Also assert the prefix owner's KV is byte-unchanged after the sharer's first CoW write.
- [ ] **Step 2:** run → RED (helper/`PrefixId` API missing).
- [ ] Steps 3-4: GREEN after Tasks 2-4.

---

### Task 2 — `PrefixId` registry + owner (primary primitive)

**Files:** Modify `fuel-core/src/kv_block_pool.rs` (+ device passthrough in `kv_block_pool_device.rs` if needed). Test: same file's `tests`.

**Interface produced:**
- `pub struct PrefixId(u64);` — registry-minted handle for a shared prefix owner.
- `KvBlockPool::register_prefix(&mut self, donor: SessionHandle, prefix_blocks: usize) -> Result<PrefixId, KvAllocError>` — opens an owner `SessionHandle`, transactionally splices donor's first `prefix_blocks` blocks into it (refcount bump), records it; the owner independently keeps those blocks alive.
- `KvBlockPool::release_prefix(&mut self, id: PrefixId) -> Result<(), KvAllocError>` — discards the owner handle (decrement refcounts; frees only blocks at refcount 0).
- `KvBlockPool::prefix_blocks(&self, id: PrefixId) -> Result<usize, KvAllocError>` (introspection).

- [ ] **Step 1 (RED):** `prefix_owner_keeps_blocks_alive_after_donor_discarded` — fill donor A (N blocks); `register_prefix(A, N)`; `discard(A)`; assert the N blocks are still resident (refcount 1, held by the owner) and `still_shared` reports them while owner-referenced. **The owner-only `still_shared` pin:** evict-query the owner's blocks and assert they report `still_shared` (NOT `freed`) with the owner as sole reference.
- [ ] **Step 2:** run → RED.
- [ ] **Step 3 (GREEN):** implement the registry (owner handle map) + `register_prefix`/`release_prefix` reusing `open`/`splice`/`discard`.
- [ ] **Step 4:** run → GREEN; regress existing `kv_block_pool` splice/refcount tests.
- [ ] **Step 5:** commit `feat(kv-pool): named refcounted PrefixId prefix-owner (registry-controlled lifetime)`.

---

### Task 3 — Transactional `splice_prefix` helper

**Files:** Modify `fuel-core/src/kv_block_pool.rs`. Test: same file.

**Interface produced:**
- `KvBlockPool::splice_prefix(&mut self, src: SessionHandle, dst: SessionHandle, prefix_blocks: usize) -> Result<usize, KvAllocError>` — validates ALL preconditions **before** mutating; on success splices + returns the shared token count `= prefix_blocks * block_size`. Same-signature `splice_prefix_from(&mut self, prefix: PrefixId, dst)` variant.
  - **`dst` STRICTLY empty** (`Err(PrefixTargetNotEmpty)`). A shared prefix must be `dst`'s FIRST blocks, but `splice` APPENDS — so a non-empty `dst` puts the prefix at positions `M..M+N` while its RoPE'd keys were computed for `0..N`. That is position-shifted and numerically wrong regardless of slot-alignment; fixing it is rung-2's job, not a precondition rung-1 can relax. (Corrected from an earlier draft that permitted "alignment-legal" non-empty targets — a Lightbulb-flagged error.)
  - **every shared block FULLY filled** (`prefix_blocks * block_size <= src_filled`, else `Err(PrefixNotFullyFilled)`). Block-granular *count* ≠ block-aligned *fill*: a donor with 6 tokens at bs=4 has 2 blocks but `filled==6`, so sharing 2 blocks would give a misaligned `filled==6` sharer. Enforcing fully-filled blocks keeps `filled % block_size == 0` AND means the sharer's first suffix write lands on a FRESH block — so rung-1 never writes into a shared block and CoW does not enter the rung-1 path at all (it re-enters only with partial sharing or rung-2). Matches Lightbulb's floor-aligned-whole-blocks discipline (their `tokens_lost_to_alignment`).
  - NOTE (doc line in the impl): distinct from `lightbulb::model_fuel::policies::splice_prefix` — that takes a `PrefixMatch` with stricter policy; this is the pool mechanism it can build on. Different signature + semantics; confusable in review.

- [ ] **Step 1 (RED):** `refused_splice_prefix_leaves_the_pool_completely_untouched` — snapshot donor block count, fill, EVERY donor refcount, and `free_blocks()` before a `splice_prefix` into a NON-EMPTY target; after the refusal assert **on the pool state**: `dst` block count + fill unchanged, `dst`'s own block untouched, no second block gained, every donor refcount == snapshot, free list unmoved, and a subsequent LEGITIMATE `splice_prefix` into a FRESH empty session still succeeds (returns block-aligned `shared_tokens`, blocks now refcount 2). Assert the `Err` too, but the pool snapshot is the guard (the Err-only assertion passes on the broken validate-after-splice code — do not rely on it).
- [ ] **Step 1b (RED):** `splice_prefix_refuses_a_partial_last_block` — donor filled 6 (bs 4 → block 1 half-full); `splice_prefix(a, c, 2)` → `Err(PrefixNotFullyFilled)` (would share a partial block); `splice_prefix(a, c, 1)` → `Ok(4)` (one full block, alignment preserved).
- [ ] **Step 2:** run → RED.
- [ ] **Step 3 (GREEN):** implement validate-before-mutate: project the post-splice fill, check alignment/bounds, return `Err` before any `splice`.
- [ ] **Step 4:** run → GREEN. **MUTATION-CHECK (confirmed recompile):** move the validation to AFTER `splice`, rebuild, confirm the state-asserting test goes RED while the Err assertion stays green (proving only the pool assertion has teeth); revert.
- [ ] **Step 5:** commit `feat(kv-pool): transactional splice_prefix (validate-before-mutate, prefill-suffix count)`.

---

### Task 4 — `PagedSessionScheduler::add_session_sharing_prefix` (reference caller)

**Files:** Modify `fuel-inference/src/multi_session.rs`. Test: same file + Task 1's parity test goes green.

**Interface produced:**
- `add_session_sharing_prefix(&mut self, prefix: PrefixId, prompt: &[u32], strategy, eos_id, max_new) -> fuel::Result<SessionId>` — opens a session, `splice_prefix_from(prefix, handle)`, sets its phase so prefill feeds ONLY `prompt[shared_tokens..]`, then normal decode. CoW-guarded by the existing `forward_paged_step` (`ensure_writable_block`).

- [ ] **Step 1 (RED):** `paged_scheduler_prefix_shared_matches_from_scratch` (scheduler-level twin of Task 1) — a prefix-shared session run to completion == a from-scratch session, byte-identical tokens.
- [ ] Steps 2-4: RED → implement suffix-only prefill → GREEN; Task 1 parity green.
- [ ] **Step 5:** commit `feat(paged-decode): add_session_sharing_prefix reference caller (rung-1 prefix reuse)`.

---

### Task 5 — Adversarial verification (ultracode) + push

**Why:** The splice×decode×evict intersection is exactly where an ultracode adversarial-verify pass previously found two real corruption bugs happy-path tests missed (missing CoW, non-atomic batched append). Verify before shipping.

- [ ] Run an adversarial-verify workflow over: donor/owner CoW on the sharer's first write; owner eviction/release while sharers are live; refcount lifecycle on sharer finish (owner-only → still_shared); the transactional-refusal state. Confirm/repair findings.
- [ ] Land the increment (Tasks 1-5) as clean commits; re-fetch; push to origin/main. Ping Lightbulb with the merged commit for their consumer-side read.

## Self-review notes
- Spec coverage: PrefixId owner (Task 2) + transactional helper (Task 3) + reference caller (Task 4) + the two seam-invariant correctness anchors (Task 1 pool+model, Task 4 scheduler) + adversarial verify (Task 5). Every co-design invariant maps to a born-red test with an explicit teeth check (pool-state assertion + mutation check for the transactional guard; owner-only case for still_shared).
- Open detail: whether `register_prefix` requires whole-block alignment (`prefix_blocks` is already block-granular, so yes by construction) — a partial last block would be shared-then-CoW'd by the first sharer write, which the correctness test exercises.
- Deferred: rung-2 (position-shifted donation) is a separate op; the alignment-tail `block_size-1` token loss Lightbulb accepts today is recovered only by rung-2.

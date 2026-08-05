# rung-2 — position-shifted prefix donation (RoPE delta-rotation) — Design

**Status:** primitive SHIPPED (C1–C3, exact); mid-prompt CONSUMER PARKED (C4-exact/C5). Successor to rung-1 prefix sharing ([[prefix-sharing-serving-shipped]], on main @ `1c640648`).

> ## ⚠ CORRECTION (2026-08-05, discovered during implementation) — "byte-exact" was WRONG for the mid-prompt case
>
> The delta-rotation fixes the RoPE **position** exactly, but exact mid-prompt KV
> reuse is **impossible for multi-layer transformers**, and this design's
> "byte-exact" premise (below) only holds at `n_layers == 1` (or `M == 0`, i.e.
> rung-1). Reason: a prefix computed in isolation (the donor, with no preamble)
> has layer-0 K/V that depend only on token + position — context-free — but its
> layer-1+ K/V depend on the token's hidden state, which *should* have attended to
> the preamble it now sits behind (positions `0..M`) and never did. So the reused
> prefix's deep K/V are wrong by exactly the preamble's contribution. Measured on
> the tiny 2-layer fixture: `n_layers=1` maxdiff **0.0** (byte-exact),
> `n_layers=2` maxdiff **~2.1e-3** (nonzero, unbounded in principle). Recorded in
> `fuel-core/tests/paged_decode_parity.rs::shifted_prefix_reuse_is_exact_at_depth_1_and_lossy_deeper`.
>
> **Outcome (user decision):** ship C1–C3 as the exact, position-correct
> primitive (they're sound and committed); **park** the mid-prompt consumer —
> C4's `maxdiff==0` anchor is unachievable and C5 would be an approximate feature
> with no consumer today (Lightbulb serves the start-prefix case, which is rung-1).
> The delta-rotation stays available for any future exact-context shift and is the
> exact answer at `n_layers=1`. Everything below is the as-designed spec; read it
> through this correction.

## Goal

Let a paged decode session reuse an already-computed KV prefix at a **non-zero position offset** — a shared block placed *after* `M` unique tokens (mid-prompt donation) rather than only at position 0. rung-1 shares a prefix at the same absolute positions it was computed for; rung-2 lifts that positional constraint via a **RoPE delta-rotation** of the cached keys, so the same donor KV can be dropped in mid-prompt.

## Why this is a delta-rotation (the grounding fact)

Fuel stores **post-RoPE keys**: `project_qkv_roped` (`fuel-core/src/lazy.rs`) rotates K to its absolute position `tok_pos` *before* the pool write; V is stored raw (RoPE never touches V). So a prefix whose keys were rotated for positions `0..N` is numerically wrong at positions `M..M+N` — its keys carry the wrong rotation. rung-2 corrects them.

The correction is a **uniform** θ·M rotation: every cached token `i` moves from rotation `p=i` to `p=M+i`, a delta of `M` for **all** tokens. RoPE rotate-half composes additively per dimension-pair (`R(M)·R(p) = R(p+M)`), so applying θ·M to the cached (already-rotated) keys yields exactly the keys as if computed at the shifted positions. **Byte-exact**, not approximate.

It reuses the **existing** RoPE op (`rope_with_tables_decomposed`) fed **constant-position tables** — every table row is position `M` (not `M, M+1, …`) — so there is **no new numeric `Op`** at the primitive floor. Reusing the model's own RoPE path (rather than hand-rolling a rotation) is deliberate: it inherits any inv-freq / rope-scaling the model applies and avoids the interleaved-vs-rotate-half convention trap ([[rope-convention-mismatch-baracuda-fuel]]).

## Decisions (from the brainstorm)

1. **Materialize at splice** (not on-the-fly in attention, not pre-RoPE storage). Allocate fresh blocks for the shifted prefix, write rotated K + copied V. Consequence: **no memory sharing** for a shifted prefix (one copy per distinct offset); the win is **compute** — skip re-running the prefix through the model. No attention-kernel changes on any backend.
2. **Block-aligned offset only.** `M = filled_tokens(dst)` must satisfy `M % block_size == 0`, so the shifted prefix occupies whole fresh blocks appended cleanly after the sharer's (full) blocks. Non-block-aligned offsets (partial-block merge + CoW) are a separate follow-on.

## Components

### C1 — `build_rope_delta_tables` (constant-position RoPE tables)
`fuel-core` (beside `rope_tables_const`). Produces cos/sin of shape `[rows, head_dim]` where **every** row is the *same* θ·`delta` rotation. Construction (the delta is uniform, so all rows are identical): take the single position-`delta` row from the canonical builder — `fuel_graph::build_rope_tables(rope_base, delta, /*seq*/1, head_dim)` → `[1, head_dim]` — and repeat it `rows` times. This is NOT `build_rope_tables(rope_base, delta, rows, head_dim)`, which would give the *incrementing* positions `[delta, delta+1, …]` (the standard RoPE progression) — wrong here; we need the constant `[delta, delta, …]`. Routing through the canonical builder inherits the model's inv-freq/scaling so the rotation matches the model's RoPE exactly — the logit anchor (C4) is the check that it does. `rows` is `block_size` (C2 applies the rotation per block).

### C2 — `DeviceKvPool::splice_prefix_shifted` (the product)
```rust
pub fn splice_prefix_shifted(
    &mut self,
    prefix: PrefixId,
    dst: SessionHandle,
    rope_base: f64,
) -> Result<usize, KvAllocError>  // returns shared_tokens = N
```
On `DeviceKvPool` (not the pure `KvBlockPool`) because rotation is a numeric op needing the device + K/V buffers. `rope_base` is a plain parameter, so a consumer that drives the pool directly (Lightbulb, per §15) supplies its own — the seam is preserved. Flow:
1. Validate (before any mutation): `dst` open; `M = filled_tokens(dst)` block-aligned; prefix registered; every shared block fully filled; pool has capacity for `N/block_size` fresh blocks.
2. For each layer, for each shared block: read cached K → build the RoPE-delta lazy graph — reshape the block to `[1, n_kv_heads, block_size, head_dim]` (the `project_qkv_roped` layout: RoPE broadcasts over heads, tables index `(position, head_dim)`) → `rope_with_tables_decomposed` with C1's `[block_size, head_dim]` constant-`M` tables → **realize on the pool's device** → write to a freshly-allocated block. A single uniform θ·M rotation is correct for the whole block even though it holds `block_size` distinct original positions — the delta is the same `M` for every position. Copy the V block verbatim (`read_block_bytes` → `write_block_bytes`).
3. Extend `dst`'s block table with the fresh blocks; bump `filled_tokens(dst)` by `N`.
4. On any failure, roll back fresh blocks taken (transactional).

Realizing through `rope_with_tables_decomposed` (not a host loop) is what guarantees byte-exactness with the model and device-correctness on CUDA/Vulkan pools.

### C3 — preconditions / errors (new `KvAllocError` variants as needed)
- `OffsetNotBlockAligned { filled, block_size }`
- reuse `PrefixNotFullyFilled`, `UnknownPrefix`, capacity errors
- f32 pool first; bf16 is the known cast-around-RoPE follow-on (mirror `project_qkv_roped`'s `to_dtype(F32)` … `to_dtype(act)`).

### C4 — correctness anchor (model level, LOGIT oracle)
`shifted_prefix_session_decodes_like_from_scratch`: build a donor prefix (positions `0..N`, registered); a sharer prefills a **block-aligned** preamble of `M` tokens, calls `splice_prefix_shifted` (prefix now at `M..M+N`), prefills its suffix, decodes. Assert **per-step logits `maxdiff == 0`** vs a from-scratch session on the identical full prompt `[preamble][shared][suffix]`. Reuses rung-1's `capture_logits` at the model driver level (or the scheduler once C5 lands). Token equality is a **vacuous** oracle here ([[sabotage-test-calibration]] pt 4) — logits only. Mutation-verify teeth: a wrong delta (e.g. `M±1`, or delta 0) drives `maxdiff` off zero.

### C5 — scheduler reference caller (increment 2)
`PagedSessionScheduler::add_session_with_shifted_prefix(prefix, preamble, suffix, strategy, eos, max_new)`: open → prefill the block-aligned `preamble` → `splice_prefix_shifted(prefix, handle, rope_base)` → set `prefill_start = M + N` so prefill feeds only `suffix` → decode. Requires the admission to prefill the preamble eagerly (unlike rung-1's splice-before-prefill). Its own scheduler-level logit-parity twin (`paged_scheduler_shifted_prefix_matches_from_scratch`) + a refusal test (non-aligned preamble → Err, pool untouched).

## Sequencing

- **Increment 1** (this spec's core): C1 + C2 + C3 + C4 — the primitive, the pool product, and the model-level logit anchor. Independently shippable and testable.
- **Increment 2**: C5 — the scheduler reference caller + its parity/refusal tests.
- **Out of scope** (separate future work): arbitrary (non-aligned) offsets via partial-block CoW; the rung-1 alignment-tail recovery; bf16; a zero-copy on-the-fly-rotation variant (attention applies the delta on read).

## Invariants honored

- **Never-panic** (`Result` throughout), **validate-before-mutate** (transactional splice), **f32/CPU-verifiable** first, **lazy-only** (rotation is a realized graph, not a host loop), **exactness via reuse** (the model's own RoPE path + table builder), **seam** (product on the device pool taking `rope_base`; scheduler caller is the reference).

## Risks

- **inv-freq / rope-scaling drift.** If C1's tables don't match the model's forward RoPE scaling (e.g. LLaMA-3.1 scaled rope), the rotation is wrong. Mitigation: route C1 through the canonical builder; the C4 logit anchor (`maxdiff==0`) is the catch. Note the sibling [[decode-model-tiering]] work just added a RoPE inv-freq seam — confirm C1 threads the same seam.
- **Copy cost.** Materializing duplicates K+V per offset. Acceptable: the value is skipping prefill compute, not memory. Documented, not hidden.

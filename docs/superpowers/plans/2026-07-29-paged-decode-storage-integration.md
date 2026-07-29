# Paged-decode storage integration (multi-session serving)

**Status:** in progress (2026-07-29). Builds on the KV block-pool allocator
(`fuel-core/src/kv_block_pool{,_device}.rs`) + its C-1 wiring into
`fuel-inference`'s `SessionScheduler`.

## Goal

Make sessions' KV **physically live in `DeviceKvPool` blocks**, decoded via
`Op::PagedAttn`, replacing the per-session contiguous `KvCache`. This is what
turns the allocator from a capacity *accountant* (the C-1 wiring) into the actual
KV *store* — unlocking:

- **No `max_seq_len` reservation.** Blocks are allocated incrementally as a
  session generates tokens (crossing a block boundary allocates one block),
  instead of reserving the full `prompt + max_new` up front. The paging win.
- **C-3 into the live path.** `evict`/`restore`/`splice` operate on the blocks a
  session is *actually* decoding against, not a proxy — so suspend/resume and
  prefix-sharing become real on the decode path, not just allocator mechanism.

## Increments

- **PS1 — the paged decode-step building block + parity — DONE 2026-07-29.**
  `DeviceKvPool::build_decode_attn` builds (does not realize) one decode step's
  paged attention for one layer: write the new token's `[Hkv, D]` K/V into its
  physical pool slot (`Op::WriteSlice` at `[(phys,phys+1),(slot,slot+1),…]`),
  then `Op::PagedAttn` over the post-write pool via the session's block table.
  Gate: `paged_decode_multistep_matches_dense_reference_across_block_growth` —
  decode 10 tokens one at a time (block_size 4 → 3 blocks) and the attention
  output equals a dense `softmax(scale·q·kᵀ)·v` over the tokens so far, at EVERY
  step. Proves the token-by-token write/accumulate path: block-boundary growth,
  the running `context_len` mask, and in-place pool writes persisting across
  steps. (2c had already proved single-shot paged_attn over a pool.)

- **PS2 — `LlamaModel::forward_with_paged_kv` (single session) + parity.** Compose
  `build_decode_attn` across all layers with the existing projection/RoPE, into a
  full decode forward that produces logits **ε-close** to
  `forward_with_kv_context` for the same tokens. Binds all `n_layers` pool buffers
  (K+V) into one realize (mirroring the contiguous forward's per-layer
  `const_placeholder_like` + `ctx.insert` loop). Prefill writes the prompt's K/V
  into blocks; decode appends one token/step. Rebuild-per-step first (the D1-style
  path); the plan-once/persistent optimization is a later rider. Duplicate the
  projection/RoPE rather than refactor the tested contiguous forward — factor the
  shared half only once both are green.

- **PS3 — wire it into `SessionScheduler`.** A session's KV lives in a shared
  `DeviceKvPool` (one pool, `open()` per session) instead of a private `KvCache`.
  Admission grows from the C-1 *reservation* model to *incremental* allocation
  (append blocks per step; the reservation becomes an upper bound / optional
  pre-reserve). The `DecodeModel` trait gains a paged-decode method; `LlamaModel`
  implements it. `reap_finished` already frees pool blocks — it stays. Parity: a
  scheduled single session matches the contiguous scheduler's tokens.

- **PS4 (later) — batched paged decode + C-3 on the live path.** Batch=K paged
  decode over the shared pool (block_table `[K, max_blocks]`); evict/restore/splice
  driven from the scheduler; GQA + bf16 + live-GPU.

## Layering / correctness notes

- `build_decode_attn` is a **graph-builder** (no realize) because the pool buffers
  bind once at the forward's single realize (all layers write different
  `k_pool(layer)` buffers), exactly like the contiguous forward binds all layer
  caches then realizes once.
- RoPE parity: paged applies RoPE to q and to the new token's K **at write time**,
  identical to the contiguous path; `Op::PagedAttn` does no RoPE internally
  (gather + dense SDPA + `context_len` mask). So paged ≈ contiguous ε-close (the
  gather-then-dense reduction order differs slightly from sliced SDPA — the
  ε-close bar the batched arm already uses, not bit-exact).
- Padding in the block table is masked by `context_len` (a padded key position is
  always `≥ context_len`), so it need only be an in-bounds index (0).

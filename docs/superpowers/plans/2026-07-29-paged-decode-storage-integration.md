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

- **PS2 — `LlamaModel::forward_paged_step` + parity — DONE 2026-07-29.** A
  single-token forward that composes `build_decode_attn` across all layers with
  the projection/RoPE (duplicated from `apply_layer_with_kv_writes` into
  `apply_layer_paged`, so the tested contiguous forward is untouched), binding all
  `n_layers` pool buffers (K+V) into one realize. `Op::PagedAttn` is decode-only
  (Sq=1), so prefill feeds the prompt one token at a time — position-for-position
  equivalent to a batched causal prefill (each token's K/V depends only on
  `0..=i`, all resident when token `i` is fed). Gate:
  `fuel-core/tests/paged_decode_parity.rs` — logits ε-close (rel 1e-4) to
  `forward_with_kv_context` across prefill + decode, for **both no-GQA and GQA**
  (n_rep 2) configs. Rebuild-per-step (no plan-once yet); f32-only. The shared
  projection/RoPE half can be factored now that both are green (a cleanup, not a
  blocker).

- **PS3 — `PagedSessionScheduler` — DONE 2026-07-29.** A session's KV lives in a
  shared `DeviceKvPool` (one pool, `open()` per session) instead of a private
  `KvCache`; blocks grow **incrementally** per token via `forward_paged_step`, not
  reserved up front. The `DecodeModel` trait gained `forward_paged_step`
  (`LlamaModel` impls it). Serial arm only; admission is optimistic (pool
  exhaustion mid-decode isolates into a per-session finish, never a panic);
  `reap_finished` frees blocks. Rather than entangle the tested contiguous
  `SessionScheduler` (whose `SessionState` owns a `KvCache`), PS3 is a distinct
  driver sharing the small types (`SessionId`/`SessionPhase`/`SamplingStrategy`/
  `StepReport`). Gates (fuel-inference multi_session, 5 tests): single-session
  budget, eos-stop, **shared-pool isolation** (2 sessions over one pool == each
  alone, token-identical), reap-frees-blocks, and an end-to-end tie — paged greedy
  output matches the contiguous `generate_with_kv_context` oracle **exactly**
  (PS2's ε-closeness is tight enough that greedy argmax is stable).

- **PS4b — C-3 (evict/restore) on the live path — DONE 2026-07-29.** The pressure
  valve for PS3's optimistic admission: `PagedSessionScheduler::evict_session(id)`
  suspends a live session (captures its KV bytes device→host via the tested
  byte-exact `DeviceKvPool::evict`, frees its pool blocks for others);
  `restore_session(id)` re-allocates + writes the bytes back (capacity pre-checked,
  so a failed restore leaves the session suspended + retryable, never losing
  bytes). `step()` skips suspended sessions; `run_to_completion` stops rather than
  spins when the only remaining work is suspended. *Which* session to evict is
  consumer policy; this is the mechanism. Gates: evict→restore resumes **byte-exact**
  (same final tokens as an uninterrupted run — bytes round-trip + rng/tokens/budget
  preserved), and an end-to-end pressure-valve flow (evict A → run B in the freed
  room → restore A → both finish). fuel-inference multi_session: 26 tests.

- **PS4a — paged BATCHED decode — DONE 2026-07-29.** `DeviceKvPool::
  build_decode_attn_batched` (K per-session slot writes + one Op::PagedAttn at
  B=K) + `LlamaModel::forward_paged_step_batched` (K uniform-position sessions,
  reusing the shared `project_qkv_roped`/`ffn_block`) + `PagedSessionScheduler::
  step_batched` (partition ready by position, batch same-position groups ≤
  `max_batch`, serial fallback). Gates: a batched step's row `i` == that session's
  standalone serial decode (no-GQA + GQA), and the scheduler arm is token-
  identical to serial + actually fires. Uniform-position is the batching
  precondition (matching the contiguous arm's gate). f32-only.

- **PS4c-splice — SPLICE on the live path — DONE 2026-07-29.** The core's
  refcounted `splice` composes with the paged read path with zero new code: a
  session reading another's spliced (shared) blocks via `Op::PagedAttn` equals a
  dense reference over the donor's K/V (test
  `spliced_shared_blocks_are_read_by_paged_attention`). This is the prefix-
  sharing / residual-stream-donation substrate; a *scheduler* prefix-share
  admission API is consumer policy (fuel-inference's prefix_cache) — built when a
  consumer needs it, not speculatively.

- **PS4c-CUDA (remaining — GPU-gated).** Byte/dtype-generic `write_block`/
  `read_block` + `build_decode_attn` for a bf16 pool, and a live-GPU (`#[ignore]`)
  parity run of the paged path on CUDA/RTX 4070. Deliberately NOT shipped from
  this CPU flow: the bf16 kernels can only be exercised on the GPU, and shipping
  unverified numeric code violates the test-gated norm. Needs a GPU session (cold
  CUDA build ~36 min + cuDNN on PATH per environment discipline); the CPU-side
  seam is ready to generalize.

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

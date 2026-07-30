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

- **PS4c-CUDA (remaining — GPU + baracuda-gated). Execution plan (2026-07-30,
  CireSnave chose this as the next track).** Serving is f32/CPU-only; this makes
  the paged path real on the RTX 4070 in bf16. It is GPU + baracuda-gated —
  bf16 attention compute can only be exercised on CUDA (the existing bf16 decode
  test `bf16_cuda_decode_graph_offers_flash_arm` is `#[cfg(cuda)]` + `#[ignore]`),
  so shipping bf16 numeric code from a CPU flow would violate test-gated. Three
  increments:

  - **PC-1 — dtype-agnostic block MOVEMENT (CPU-verifiable, buildable now).**
    Today `write_block`/`read_block` (and the CoW `ensure_writable_block` +
    evict/restore that build on them) hard-code f32 (`from_f32`/`const_f32_like`/
    `realize_one_as::<f32>`). A bf16 pool stores bf16 bytes, so movement must be
    dtype-agnostic or it reinterprets bf16 as f32. Refactor the movement core to
    carry BYTES keyed by the pool's dtype (a const-from-bytes-with-dtype source +
    a byte-width read), keeping f32 typed wrappers for existing callers. **Verify
    with f32** (round-trip through bytes is byte-identical) — no bf16 kernel
    needed. This is the real CPU-verifiable prerequisite; movement side only.
  - **PC-2 — bf16 paged decode on CUDA (GPU-gated).** Build a bf16 `DeviceKvPool`;
    `forward_paged_step`/`_batched` in bf16 (the graph already threads bf16 —
    "BF16-throughout decode", Phase D increment A). Depends on **PC-3**. Gate: an
    `#[ignore]`'d live-GPU parity test (paged bf16 == the contiguous bf16
    reference / an f32 reference within ε) on the 4070.
  - **PC-3 — the paged/flash CUDA kernel (GPU + baracuda ask).** Whether
    `Op::PagedAttn` dispatches to a fast baracuda kernel (`flash_decoding` /
    paged-attention) on CUDA vs decomposing to the gather+SDPA primitives (whose
    bf16 CUDA kernels must then all exist). This is the "flash arm" for the paged
    path — distinct from the contiguous path's `offer_flash_decode_arm_for_region`
    (→ `Op::Branch` to a CUDA-pinned `FlashAttn`). A cross-project ask: does
    baracuda expose a paged/flash-decoding kernel Fuel can register for
    `Op::PagedAttn`? (cf. the memory `bf16-cuda-decode-part2`: "CUDA BF16 kernels
    mostly EXIST; blockers are graph-level".)

  Sequencing: PC-1 now (CPU); PC-2/PC-3 in a GPU session — cold CUDA build ~36 min
  + cuDNN on PATH (environment discipline), one live-GPU suite at a time
  (coordinate the 4070 with the peer's tri-backend `DeviceGroup` test over the
  claude-peers channel), and a baracuda kernel ask for PC-3.

## PS4 hardening — adversarial verification (ultracode, 2026-07-29)

A multi-agent adversarial-verification pass over the batched B>1 path (5 attack
angles → refute-verify) found two REAL defects the happy-path parity tests missed
(they used non-spliced, over-provisioned sessions) — both now FIXED (`7676d9f5`):

- **Missing copy-on-write** (the root of 5 confirmed findings): the decode write
  path (single + batched) wrote a new token into the resident block without
  breaking a share, so decoding a *spliced* session corrupted its co-sharers.
  Fix: `DeviceKvPool::ensure_writable_block` (cow_break + byte copy if shared),
  called in both forwards after append. Regressions: device `ensure_writable_
  block_copy_on_writes_a_shared_block`; end-to-end `paged_decode_into_spliced_
  prefix_does_not_corrupt_donor`.
- **Non-atomic batched append**: a mid-loop OutOfBlocks left earlier sessions
  advanced, wedging the batch non-uniform. Fix: pre-check total block need
  (boundary appends + CoW splits) vs free before mutating; error atomically.
  Regression: `batched_step_out_of_blocks_is_atomic`.

The pass also flagged the **highest-risk coverage gap** — batched decode at
`max_blocks > 1` was untested (1-block sessions make the per-row gather a no-op).
Closed with `paged_batched_multiblock_matches_serial` (2-block sessions, batched
row == serial). Remaining lower-risk follow-ups (not live bugs): a build-time
`Sq == 1` assert on the `Op::PagedAttn` builder (decode-only; deferred to avoid
re-entering `fuel-graph/lib.rs` while the seam owner works there); `B >= 3`;
genuine variable-length (non-uniform) batching + its block-table padding path.

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

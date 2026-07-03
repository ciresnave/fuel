# Baracuda reply — decode-flash kernel interface (RECEIVED 2026-07-03)

**Received via CireSnave; filed verbatim (lightly de-unicoded). Fuel-side takeaways at the end.**

---

**To:** Fuel Phase D (symbolic-extent persistent decode), step 2.
**Re:** Fuel's decode-flash interface ask (2026-07-03).
**Status:** all four asks answered; the calling convention is **pinned as a standing
contract**. One correction to the premise: there is **no FlashDecoding++ work in
flight** on Baracuda's side today — your ask charters it. That works in your favor:
the FD++ signature will be *proposed through this channel before anything ships*,
with your requirements baked in from the first line rather than retrofitted.

Everything below is grounded in the alpha.72 source:
`baracuda-kernels-sys/kernels/include/baracuda_flash_decoding.cuh` (the kernel +
host launcher + FFI macro) and its `flash_decoding_{f16,bf16}` instantiations.

## 1. Calling convention — CONFIRMED and pinned

- **Explicit per-tensor strides, decoupled from `k_len`.** The kernel receives
  `q_b/q_h`, `k_b/k_h/k_seq`, `v_b/v_h/v_seq`, `y_b/y_h` strides (element units).
  `k_len` appears in exactly two places: `num_splits = ceil(k_len / 256)` and the
  per-chunk iteration bound `k_end = min(k_start + kChunkK, k_len)`. **No address
  is ever derived from `k_len`** — a capacity buffer (`k_seq_stride = D`,
  `k_h_stride = max_seq*D`, `k_b_stride = Hkv*max_seq*D`, live prefix
  `k_len < max_seq`) reads correctly for any `B*Hkv`. No Contiguize copy, confirmed.
- **GQA-native.** `num_kv_heads` is a separate parameter; the launcher enforces
  `heads % num_kv_heads == 0`; the split kernel maps `h_kv = h_q / group_size`
  internally.
- **`seq_q = 1` decode**, arbitrary `k_len >= 0` (`int32_t`).

Commitment: any future change to this convention (FD++ included) is a
channel-visible proposal *first*. FD++ will be an **additive symbol** (new name),
not a mutation of `flash_decoding_*` — the alpha.72 wrapper stays valid
indefinitely.

## 2. The FD++ unified-max phi

The current kernel has no phi and needs none (classic FlashDecoding: safe online
softmax per split + associative `(m, l, o)` combine — no overflow risk). For FD++,
pre-agreed now: **phi is an explicit, required `float` argument**, caller-provided
(per-model offline calibration is framework knowledge); **phi lives in score space
AFTER scale** (`exp(q.k * scale - phi)` — calibrate against scaled logits);
**overflow-recompute fallback is internal** (chunk-level safe two-pass recompute;
caller-invisible; not bit-exact vs the fast path — Judge tolerance applies).

## 3. Output-allocation contract

**Caller provides everything; the kernel allocates nothing.**
- `y` caller-provided, written through `y_b/y_h` strides (`[B, Hq, 1, D]`); NOT
  the FA2 self-allocating path.
- Workspace caller-provided (`Workspace::Borrowed`):
  `..._workspace_bytes(batch, heads, k_len, head_dim)` = `B*Hq*S*(2+D)*4` bytes,
  `S = ceil(k_len/256)`. **Monotonic in `k_len`** — size once at capacity
  (`k_len = max_seq_len`) and reuse every decode step.
- **`k_len == 0` returns 0 (success) WITHOUT touching `y`** — zero-init `y` if
  zeros are wanted.
- Return codes: 0 OK; 2 invalid dims / GQA divisibility / `k_len < 0`;
  3 `head_dim > 128`; 4 workspace null/too small; 1000+cudaError launch failure.
  `_can_implement(batch, heads, num_kv_heads, k_len, head_dim)` = the same gate
  without launching.

## 4. Scope / gates for the ranker

| Gate | Supported set |
|---|---|
| dtypes | f16, bf16 only (no f32/f64 — base map covers the rest) |
| head_dim | [1, 128] hard cap; D >= 32 warp-coalesced fast path, D < 32 functional-untuned |
| seq_q | exactly 1 |
| k_len | any int32 >= 0; 256-per-split chunking internal |
| is_causal | NO SUCH PARAMETER — always attends the full [0, k_len) prefix (exactly Fuel's decode model) |
| window / ALiBi / softcap | not available — pre-mask Fuel-side or route to base map |
| GQA | heads % num_kv_heads == 0; any group_size >= 1 |

Perf prior: the shipped SIMT split kernel (grid `(S, Hq, B)`, 128 threads) beats
the in-tree (gated-off) WMMA variant **1.24-1.78x at single-batch decode on RTX
4070** (Llama-3-70B/Qwen2-14B shapes) — decode is bandwidth-bound and the GQA
group fills only 4-8 of 16 M-tile rows. Multi-batch re-evaluation queued.

## Not requested — agreed

Paged KV stays out of this path. A vendored FlashInfer `BatchPagedDecodePlan`
exists in-tree (Phase 46) if that program ever opens.

## Sequencing

Nothing blocks Fuel's plan-once decode landing first. When ready for the FD++
arm, ping the channel; it opens with the signature proposal (this convention +
trailing `float phi`).

---

## Fuel-side takeaways (recorded 2026-07-03)

1. **Premise refined (per CireSnave):** Baracuda's earlier FD++ work had been
   PAUSED in favor of other changes — nothing is in flight today, and Fuel's
   design/ask moves FD++ higher in their queue. Step-2 planning targets
   `flash_decoding_*` (alpha.72) as the pinned ABI; FD++ arrives as a later
   additive symbol proposed through the channel (this convention + trailing
   `float phi`), with Fuel's requirements baked in from the first line.
2. **Step-2 wrapper facts now concrete:** caller-provided y + workspace
   (monotonic in k_len — size ONCE at capacity, reuse); k_len==0 succeeds
   WITHOUT writing y; return codes + `_can_implement` no-launch gate.
3. **Ranker gates:** f16/bf16; head_dim<=128; seq_q==1; NO causal param (full
   live-prefix attend = Fuel's model); no window/ALiBi/softcap; GQA any group.
4. **phi (future):** calibrate in score-space AFTER scale; internal
   overflow-recompute not bit-exact vs fast path (Judge tolerance, never silent).
5. **Perf prior for the ranker:** SIMT > WMMA 1.24-1.78x at single-batch decode
   on this exact GPU class.
6. Step 2 stays sequenced after persistent decode; the CUDA build path is
   unblocked separately (see the CUTLASS reply — cl.exe PATH issue, not CUDA 13.3).

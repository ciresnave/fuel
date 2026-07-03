# Baracuda ask — decode-flash kernel interface for Fuel (FlashDecoding / FlashDecoding++)

**From:** Fuel Phase D (symbolic-extent persistent decode), step 2 — the CUDA decode-flash arm.
**Status:** proposal / interface confirmation. `flash_decoding_{f16,bf16}` (alpha.72) already meets
most of this; the ask is to **keep that calling convention in the in-flight FlashDecoding++ work** and
add the one new FD++ input (the unified-max φ). Not blocking — Fuel's decode win lands first via
persistent plan-once decode (the decomposed base-map arm); the flash arm is an additive per-token
compute/memory win sequenced after it.
**Nature:** kernel ABI / argument surface for autoregressive decode over a fixed-capacity KV cache.

## What Fuel is doing (the why)

Phase D decode keeps the KV cache as a **fixed-capacity buffer** `[B, Hkv, max_seq_len, D]` and tracks
the live length as a **runtime value** `k_len = cached_len + seq` (a symbolic extent), so the decode
graph is structurally identical every token (the prerequisite for plan-once decode). Attention is a
DAG **base map** (`matmul → mask → softmax → matmul`) that is correct on every backend; the flash
kernel is an **additive optimizer-chosen arm** (`Op::Branch`) that the ranker prefers on CUDA and
**validates against the base map via the Judge** — so Fuel never trusts the kernel for correctness,
only speed. Consequence: a φ-overflow recompute fallback or any non-bit-exact path inside the kernel is
fine; the base map is the oracle.

What the kernel must therefore accept: **a capacity buffer whose memory layout is `max_seq_len` but
whose attended prefix is a runtime `k_len < max_seq_len`**, for **arbitrary `B·Hkv`** (real decode is
GQA + batched).

## What we already see (alpha.72 `flash_decoding` — this is great)

`flash_decoding_{f16,bf16}` (`baracuda-kernels-sys` lib.rs:44509; `baracuda_flash_decoding.cuh`)
already has exactly the right shape, and it removed the old FA2 `_sdpa_*_run_v2` constraint (which
derived every K stride from `seq_k`, valid only for `B·Hkv == 1`):

- **Explicit `k_b_stride / k_h_stride / k_seq_stride` (and V), decoupled from `k_len`** — so a capacity
  buffer with `k_seq_stride = D`, `k_h_stride = max_seq·D`, `k_b_stride = Hkv·max_seq·D` and
  `k_len < max_seq` reads correctly for any head/batch.
- **`k_len` used only as the iteration bound** (`min(k_start+chunk, k_len)`), not for addressing.
- **GQA-native** (separate `num_kv_heads`; broadcast stride).

This is precisely Fuel's capacity-K + symbolic-`k_len` decode model. **No Contiguize copy needed.**

## The ask (for the FlashDecoding++ work)

1. **Keep `flash_decoding`'s calling convention in FD++**: explicit per-tensor K/V strides decoupled
   from `k_len`, `k_len` as the runtime iteration bound, GQA-native (`num_kv_heads` separate from
   `num_qo_heads`). This is the load-bearing property for capacity-K decode; please don't regress it
   to an FA2-style `seq_k`-derived layout.
2. **Expose the FD++ unified-max φ as an argument** (with overflow bounds), or document how it's
   derived. Per the FD++ paper (asynchronized softmax) φ is a per-model offline-calibrated constant;
   Fuel is happy to compute and pass it (see below). If the kernel auto-derives or defaults φ, say so;
   if it's a required arg, confirm the units/semantics (the scaling applied as `exp(x_i − φ)`), and
   whether the overflow-recompute fallback is internal (preferred) or caller-visible.
3. **Output-allocation contract**: does the kernel allocate its own `[B, Hq, Sq, D]` output (like the
   FA2 `launch()` path), or write into a caller-provided buffer? Fuel's dispatch wrapper adapts either
   way but needs to know which.
4. **Scope / gates**: confirm the supported set so Fuel's ranker can gate the arm and fall back to the
   decomposed base map otherwise — dtypes (f16/bf16; f32?), `head_dim` cap (≤128?), `is_causal`
   handling (Fuel bounds the causal history via `k_len`, so decode is effectively non-causal over the
   live prefix — confirm the kernel's expectation), and whether sliding-window / ALiBi / softcap are
   available at decode (most Llama-family decode needs none).

## What Fuel does on its side (so this stays a thin ask)

- **φ calibration** is Fuel's job: a one-time offline pass (or a sane per-architecture default, e.g.
  the paper's Llama2 range) computed and stored per model, passed to the kernel each decode step.
- **Dispatch wrapper**: an FFI wrapper in `fuel-cuda-backend/src/baracuda/attention.rs` (alongside the
  existing `flash_sdpa_*` wrappers), passing capacity-derived strides + runtime `k_len` (+ φ for FD++),
  wired as the CUDA `Op::FlashAttn` arm (`OpParams::FlashAttn { k_len, … }` already carries it).
- **Ranker gates + Judge validation**: emit the flash arm only inside the supported set; validate its
  numerics against the decomposed base map. The non-bit-exact φ fallback is acceptable under the Judge.

## Not requested

A paged KV cache (FlashInfer-style `paged_decode`) — that's a separate, larger architectural direction
for Fuel (a dedicated paged-attention program), not this flat-capacity-buffer decode path.

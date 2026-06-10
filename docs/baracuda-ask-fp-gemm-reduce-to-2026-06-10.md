# Baracuda ask — dense FP GEMM + broadcast-reverse reductions (2026-06-10)

Coordinated ask following Fuel's alpha.66 audit (2026-06-10). The
audit walked every outstanding Fuel-side ask against the alpha.66
source; most of the backlog turned out to be already shipped (see
"Resolved" below). Two structural gaps remain — both block the same
architectural goal: **retiring the last non-baracuda CUDA code in
Fuel** so baracuda is the single CUDA kernel home
(`register_cuda_kernels` shrinks to `Op::Copy` D2H only).

Fuel repo root: `c:/Users/cires/OneDrive/Documents/projects/fuel/`.
Relevant Fuel commits: `9b53da38` (gelu flavor remap + live-sweep
repair), `99165f0a` (native `unary_step` adoption).

---

## Resolved in alpha.66 — no action needed, just confirming receipt

| Item | Status |
|---|---|
| Native step/heaviside kernel | ✅ `unary_step_*` (Phase 31) — Fuel adopted it in `99165f0a`, full 4-dtype + strided. Note: the baracuda-kernels facade never wrapped it (sys-only); Fuel binds sys symbols directly so this is cosmetic on your side. |
| Device memory introspection | ✅ `Device::vram_info` / `mem_get_info` — integrated in Fuel `5f16ff9d`; powers `BackendRuntime` + the VramPressure runtime selector. |
| `DeviceSlice::from_raw_parts` (+ `Mut`) | ✅ `baracuda-driver/src/memory.rs:976,1049` — this was the blocker for CUTLASS "Track B" in Fuel's earlier critique (`docs/baracuda-comprehensive-ask-2026-05-25.md` lineage). We'll revive that integration on our side. |

---

## Ask 1 — dense FP GEMM family in baracuda-kernels

**Priority: high.** This is the largest remaining non-baracuda CUDA
surface in Fuel.

### Current state

`baracuda-kernels/src/gemm/` covers: `bin_gemm`, `fp8_gemm`,
`gptq_to_marlin`, `int4_awq`, `int4_gemm`, `int4_marlin`,
`int_gemm` (S8/U8 RRR), `sparse24` — but no plain dense
**f32/f64/f16/bf16** GEMM. Fuel therefore keeps a cuBLAS-backed
MatMul path (`matmul_via_cublas` in fuel-cuda-backend, registered by
`fuel-dispatch::dispatch::register_cuda_kernels`) that predates the
fuel-cuda-kernels retirement and can't retire until baracuda offers
an equivalent.

### What Fuel needs

- `gemm_{f32,f64,f16,bf16}_run` (+ `_can_implement`) with the usual
  binding-table-shaped FFI contract: `(m, n, k, batch)` dims,
  lhs/rhs/out pointers, scratch + stream.
- **Leading-dimension flexibility**: `lda/ldb` may exceed the row
  size (row-slice views of larger tensors). This bit us before —
  Fuel's cuBLAS path had to relax exactly this
  (`project_cuda_matmul_noncontig_gap`: BERT / SD CLIP / Qwen2-MoE
  all produce non-contiguous matmul operands). A `gemm_config`-style
  descriptor with explicit strides is ideal; transpose flags (RRR /
  RCR / CRR) acceptable as a v1 narrowing.
- **Batched variant** (strided-batch is fine; Fuel's IR carries a
  uniform batch stride).
- f16/bf16 accumulating in f32, matching your reduce family's
  convention.

**Implementation latitude**: a cuBLAS-backed wrapper *inside*
baracuda is completely fine — Fuel doesn't need hand-tuned kernels
here, it needs the dependency direction fixed (Fuel → baracuda only).
If baracuda-cutlass eventually supersedes it per-arch, that's
invisible to us behind the same FFI symbols. We're aware
baracuda-cutlass has f16/bf16 RCR sm80 today; the ask is the
general dense family in baracuda-kernels proper, with the
fallback-to-cuBLAS story handled on your side of the boundary.

---

## Ask 2 — broadcast-reverse reductions (reduce-to-shape)

**Priority: medium-high.** Caps Fuel's CUDA autograd dtype coverage
today.

### Current state

`baracuda-kernels/src/reduce/axis.rs` is single-axis keepdim
(`{Sum, Mean, Max, Min, Prod, Norm2, LogSumExp, Var, Std} × 4
dtypes`). Fuel's autograd needs the *gradient-of-broadcast*
operation: reduce an input to a **target shape** where each target
dim is either equal to the input dim or 1 (or missing). Fuel calls
these `ReduceSumTo` / `ReduceMaxTo`.

Fuel currently ships its own f32-only kernels for these
(`fuel-cuda-backend::byte_kernels::{reduce_sum_to_f32,
reduce_max_to_f32}`) — the last hand-written compute kernels left in
fuel-cuda-backend. Consequences: CUDA `ReduceSumTo`/`ReduceMaxTo`
are f32-only while CPU covers f32/f64/bf16/f16, so any
f16/bf16/f64 training graph with a broadcast falls back to CPU at
every gradient edge.

### What Fuel needs

Either shape works for us, in preference order:

1. **`reduce_to_shape`**: `(input_shape, target_shape, op ∈ {Sum,
   Max}) × {f32, f64, f16, bf16}` — semantics: `out[j] = op over all
   input indices i that map to j under broadcasting`. This is the
   exact consumer contract and lets us delete our kernels 1:1.
2. **Multi-axis single-pass reduce** (axes bitmask on the existing
   axis-reduce plan): Fuel can derive the axes-to-reduce from the
   shape pair and reshape afterward. Slightly more Fuel-side glue,
   same coverage.

A sequential loop of single-axis passes is what we'd build as a
stopgap composition, but it allocates an intermediate per collapsed
axis — fine for occasional use, wrong as the permanent story for
something on the backward path of every broadcast.

Same precision conventions as your axis family: deterministic,
sequential per-cell accumulation, f16/bf16 accumulate in f32.

---

## Info items (no action required, flagging for your docs/roadmap)

1. **Gelu flavor naming**: plain `unary_gelu_*` computes **erf**-exact
   gelu, while `unary_gelu_erf_*` also exists (functionally a twin)
   and `unary_gelu_tanh_*` is the tanh approximation. Fuel
   mis-registered plain `unary_gelu` as its tanh-flavored
   `GeluElementwise` for ~2 weeks; the ~1e-4 divergence hid inside
   our cross-backend consensus epsilon (1e-3) and only a live
   value-level test caught it (fixed in Fuel `9b53da38`). A one-line
   flavor note on the sys symbols' doc comments would prevent the
   next consumer from making the same assumption; deprecating one of
   the erf twins in a future alpha would be even better.
2. **`unary_step` facade gap**: the kernel + sys symbols shipped in
   alpha.66 but `baracuda-kernels`' elementwise facade doesn't expose
   it. Harmless for Fuel (we bind sys directly) — flagging in case
   the facade is meant to be complete.
3. **Memory-pressure notifications**: Fuel's backend contract
   wishlisted these, but CUDA has no native pressure-event API, so we
   consider this *not actionable upstream* — Fuel polls
   `mem_get_info` via `would_fit`. Dropping it from our ask backlog.

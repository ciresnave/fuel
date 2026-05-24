# fuel-cuda-kernels retirement audit (2026-05-25)

Companion to the architectural-cleanness conversation about why
`fuel-cuda-kernels` exists despite baracuda being the strategic CUDA
kernel home. This document captures the audit findings, what was
shipped in the initial cleanup, and what remains.

## Initial framing (revised by the audit)

The original 4-step plan was:

1. Strip duplicate PTX registrations from
   `dispatch::register_cuda_kernels`.
2. Audit three "possibly unique" candidates (`conv.cu`,
   `indexed_moe_forward_*`, Q8_1).
3. Ask baracuda to take genuinely-used Fuel-unique kernels.
4. Retire the crate.

Step 1 shipped cleanly (commit `d9898fec`). The audit for steps 2–3
turned up a much broader picture than "three unique candidates" — the
crate is load-bearing for substantial portions of
`fuel-cuda-backend`'s legacy `CudaStorageSlice` API, not just the
three flagged op families. Step 4 (full retirement) is multi-commit
work, not a single follow-up.

## Step 1 outcome — shipped (commit `d9898fec`)

Stripped from `dispatch::register_cuda_kernels`:

- Binary (Add/Sub/Mul/Div/Maximum/Minimum)
- 15 unary ops (Relu/Neg/Sqr/Sqrt/Recip/Abs/Tanh/Exp/Log/Sin/Cos/
  Sigmoid/Silu/Gelu/Step)
- Reduce (Sum/Max/Min/Mean)
- Affine / Clamp / PowI / Concat
- IndexSelect / Gather (f32+u32)
- ArgMaxDim / ArgMinDim (f32+u32)
- ~45 Cast (src, dst) pairs

Kept (no baracuda equivalent today):

- MatMul (f32/bf16/f16) — pure-FP cuBLAS matmul; baracuda only has
  int-GEMM (S8/U8 RRR).
- ReduceSumTo / ReduceMaxTo — broadcast-reverse reductions used by
  autograd.
- `Op::Copy` (D2H byte-buffer transfer; Fuel-specific cross-device
  path).

50/50 live RTX 4070 tests green after the strip. 35 dead PTX wrapper
functions remain in `dispatch.rs` (dead_code warnings) — queued for
cleanup as part of the broader retirement.

## Step 2 findings — wider than expected

`fuel-cuda-kernels` is used by **34 call sites** across four files
in `fuel-cuda-backend`:

| File | Kernels used | Notes |
|---|---|---|
| `storage.rs` (~25 sites) | UNARY, BINARY, TERNARY, AFFINE, CAST, REDUCE, INDEXING, CONV, QUANTIZED, FILL | Legacy `CudaStorageSlice` API (pre-binding-table); supports the eager-tensor dispatch path. |
| `byte_kernels.rs` (~8 sites) | AFFINE, UNARY, BINARY, CAST, INDEXING, REDUCE | Newer byte-storage path; some of these had PTX-duplicate registrations stripped in step 1 but the wrapper functions still exist and call PTX. |
| `quantized.rs` (~6 sites) | QUANTIZED (Q8_1 staging + mul_mat_vec_q*_q8_1) | The architecturally-divergent GGUF path. |
| `device.rs` | (entry-point loading) | Boots the PTX modules into the CUDA context. |

### Categorization

**A. Architecturally divergent from baracuda (load-bearing; needs design discussion):**

- **MoE GEMM** (`moe_gemm_wmma`, `moe_gemm_gguf`, `moe_gemm_gguf_prefill`)
  — used by `fuel-nn::moe::moe_gemm` + `moe_gemm_gguf`, called from
  `fuel-transformers::fused_moe`. Baracuda explicitly dropped
  `indexed_moe_forward_*` in favor of MMVQ × N-experts dispatch.
  Fuel uses a different architectural path (fused MoE GEMM with
  internal expert routing). Migration ⇒ either baracuda accepts the
  fused approach or Fuel rewrites to MMVQ × N.
- **Q8_1 staging + `mul_mat_vec_q*_q8_1_cuda`** — used by
  `fuel-cuda-backend::quantized::quantize_q8_1`. Baracuda's MMVQ
  takes FP activations (no Q8_1 staging); Vulkan QMatMul already
  migrated to that path in alpha.31. Migration ⇒ port CUDA QMatMul
  to baracuda's MMVQ approach (same path the Vulkan QMatMul
  followed in commit `b7360fbc`).

**B. Could migrate to baracuda's existing primitives (mechanical):**

- **Conv family** (conv1d, conv2d, conv_transpose1d, conv_transpose2d,
  im2col, im2col1d, col2im1d) — baracuda handles conv via
  GEMM/CUTLASS/cuDNN wrappers (im2col + GEMM is what cuDNN does
  internally). Fuel could route through `baracuda-cublas` or
  `baracuda-cudnn`.
- **Pool family** (avg_pool, max_pool) — baracuda has Adaptive +
  Lp + FractionalMax pools as of alpha.33; non-adaptive AvgPool /
  MaxPool aren't direct baracuda exposures today. Could either
  expose them upstream or rewrite via the existing primitives.
- **Upsample** (upsample_nearest2d, upsample_bilinear2d) — no direct
  baracuda equivalent; trivially expressible via gather + bilinear
  arithmetic.

**C. Legacy CudaStorageSlice duplicates of stripped binding-table ops:**

The PTX duplicates were stripped from the binding table in step 1,
but `storage.rs` still has methods like `fn affine(...)`, `fn binary(...)`,
`fn cast(...)` that route through PTX. These exist to support the
non-binding-table eager dispatch path. **They're dead at the
binding-table layer but live in the legacy direct-API layer.**

Migration ⇒ either:
- (a) Retire the legacy `storage.rs` CudaStorageSlice API entirely
  once all consumers (fuel-core's eager path? fuel-graph-cuda
  remnants?) route through the binding table; OR
- (b) Port each storage.rs method to call baracuda's per-op
  wrapper instead of `kernels::*`.

**D. fill / sort / indexing utilities** — small utility set; depending
on whether baracuda exposes equivalents (sort is mentioned as
"sourced from Fuel" in the user's LICENSE-thirdparty.md note, so it
exists upstream).

## Step 3 — comprehensive baracuda ask

Drafted as a single coordinated ask covering all four open items:

- `docs/baracuda-comprehensive-ask-2026-05-25.md` — the document to
  send upstream.
- `docs/moe-design-analysis.md` — the MoE deep-dive that backs the
  Item 4 recommendation (hybrid batched MMVQ × N-experts as the
  preferred option).

Four items, ordered easy → hard:

1. **Item 1 — Non-adaptive AvgPool/MaxPool exposure.** Low severity;
   almost certainly already implemented internally for the adaptive
   variants' fallback path, just not surfaced in the FFI.
2. **Item 2 — Conv direction.** Ask whether the recommended path is
   `baracuda-cudnn`'s convolution API or im2col + `baracuda-cublas`
   GEMM. Then Fuel mirrors the Vulkan im2col + matmul pattern (per
   `fuel-conv`) on CUDA.
3. **Item 3 — Q8_1 staging deprecation.** Confirm MMVQ-everywhere is
   the recommendation; Fuel mirrors Vulkan QMatMul commit `b7360fbc`
   on the CUDA side.
4. **Item 4 — MoE direction.** Three options ordered by Fuel's
   preference: (a) batched MMVQ × N-experts (hybrid; preferred), (b)
   accept Fuel's fused MoE kernels upstream, (c) Fuel rewrites to
   MMVQ × N + routing on its side. The design memo quantifies the
   tradeoffs (3N vs. 3 launches per layer, tensor-core utilization,
   in-kernel routing).

## Step 4 — retirement plan

Full crate retirement is **multi-commit, multi-session work**. Phased
plan:

1. **Phase 1 (this session, shipped):** strip PTX duplicates from
   binding-table dispatch. ✅ Commit `d9898fec`.
2. **Phase 2:** delete the 35 now-dead PTX wrapper functions in
   `fuel-storage::dispatch` (clears the dead_code warnings).
   Single-commit cleanup; ~500 LOC delete.
3. **Phase 3:** port CUDA QMatMul to baracuda's MMVQ path
   (mirror commit `b7360fbc`'s Vulkan QMatMul work). Retires
   `quantize_q8_1` + `mul_mat_vec_q*_q8_1_cuda` + the `QUANTIZED`
   PTX module.
4. **Phase 4:** decide MoE direction with baracuda team. Either
   accept upstream OR rewrite to MMVQ × N. Retires the `moe/`
   subdir.
5. **Phase 5:** port Conv2D / ConvTranspose2D / Pool2D / Upsample
   to baracuda primitives (GEMM-based conv + adaptive pool + bilinear
   arithmetic). Retires `CONV` PTX module and storage.rs's conv
   methods.
6. **Phase 6:** audit + delete legacy `CudaStorageSlice` API from
   storage.rs (the methods that still call `kernels::*` for ops
   the binding table now handles via baracuda). Determine consumers
   first — if `fuel-graph-cuda` or `fuel-core::eager` still depends
   on them, port those to the binding table first.
7. **Phase 7:** retire `fuel-cuda-kernels` crate. Drop workspace
   member, drop `cudaforge` build-time CUDA compilation.

Phases 2-7 are individually well-scoped commits. Phase 2 is trivial
(delete dead code); phases 3+5 mirror previous Vulkan/baracuda work
so the pattern is established; phases 4+6 need design discussion.

## Why "stay handwritten in Fuel" doesn't apply anymore

The April 19 `project_cuda_kernels_stay_handwritten` memory was about
*Slang→CUDA codegen* (don't generate `.cu` from Slang sources). That
decision is unaffected — baracuda's `.cu` files are also hand-written.
The question this audit answers is different: *given baracuda is
hand-writing the kernels anyway, why is Fuel hand-writing duplicates?*
Answer: it isn't, by design — it's mid-migration with no documented
endpoint. This document is now the documented endpoint.

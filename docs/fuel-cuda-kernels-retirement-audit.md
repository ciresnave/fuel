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

1. **Phase 1 — SHIPPED:** strip PTX duplicates from
   binding-table dispatch. Commit `d9898fec`.
2. **Phase 2 — SHIPPED:** delete the 35 now-dead PTX wrapper functions in
   `fuel-storage::dispatch` (clears the dead_code warnings).
   Commit `5c79ae69`.
3. **Phase 1b (Pool 2D) — SHIPPED (2026-05-25):** `CudaStorage::{avg,max}_pool2d`
   now calls `baracuda_kernels_{max,avg}_pool_2d_fw_<dtype>_run` directly
   (alpha.36's cuDNN-feature-gated symbols). Cargo.toml `baracuda-kernels-sys`
   gains `features = ["cudnn"]`. The PTX `Pool2D::Map1` retired. Pool tests
   relaxed from 4 → 3 decimal precision to absorb cuDNN's ±1 ULP accumulation
   drift (both rounds are IEEE-754-valid). Commit `7a3cd5d1`.
4. **Phase 5a (UpsampleNearest2D + UpsampleBilinear2D) — SHIPPED (2026-05-25):**
   `CudaStorage::upsample_nearest2d` calls `baracuda_kernels_upsample_nearest_2d_fw_<dtype>_run`.
   Baracuda alpha.38 added `align_corners: i32` + `scale_h_factor`/`scale_w_factor: f64`
   parameters to `baracuda_kernels_interpolate_bilinear_2d_<dtype>_run`
   (plus f16/bf16 fanout), closing the parity gap with PyTorch's
   `nn.functional.interpolate(mode='bilinear', align_corners=...)`.
   `CudaStorage::upsample_bilinear2d` migrated to call the alpha.38 FFI
   directly; `Option<f64>` scale factors map to `0.0 = derive` per baracuda's
   convention. 24/24 fuel-core `bilinear_tests --features cuda` green
   on RTX 4070, including `bilinear_pytorch_align_corners_true_gpu`,
   `bilinear_align_corners_difference_gpu`, and `bilinear_pytorch_scale_factor_gpu`.
5. **Phase 3 (Q8_1 → MMVQ) — SHIPPED (2026-05-25):**
   Baracuda alpha.37 dropped batched MMVQ FFI symbols
   `baracuda_kernels_mmvq_<fmt>_batched_run` (36 quant variants +
   pure-FP siblings) that take fp32 activations directly (no Q8_1
   staging) and degrade cleanly to no-routing batched MMVQ when
   `n_experts=1 + top_k=1 + identity-permutation sorted_token_ids`.
   Cargo.toml bumped alpha.36→.37 (cuDNN feature kept). Fuel's
   `mul_mat_vec_via_q8_1` (b_size=1..8) + `mul_mat_via_q8_1`
   (matrix-matrix) now collapse to one helper `baracuda_batched_mmvq`.
   Tests: 4/4 fuel-cuda-backend `quantized::test` + 62/62
   `fuel-core::quantized_tests --features cuda` green on RTX 4070.
   Test fixture for `cuda_mm_q8_1_pad` bumped ncols 16→64 to satisfy
   the type-0/1 batched-MMVQ `ncols ≥ 64` invariant noted by the
   baracuda team (silent garbage at ncols<64 in batched mode).
   Bit-stable diff4 across b_size buckets in `qmm_b_cuda` (was
   `0 < diff4 < 1e-4` historically; now exactly 0 because baracuda
   uses the same kernel for all m sizes).
   Still using PTX: `quantize_q8_1` + `indexed_moe_forward_fused_q8_1_input`
   (slated for Phase 4 MoE migration to baracuda's `moe_*` symbols);
   `dequantize_f32`/`dequantize_f16`/`dequantize_mul_mat_vec` (the
   FORCE_DMMV dequant fallback path). The QUANTIZED PTX module
   doesn't fully retire until MoE Phase 4 lands.
6. **Phase 5b (Conv family) — QUEUED, larger scope than initially
   assessed.** `CudaStorage::conv2d` (the `feature = "cudnn"` path) already
   goes through `crate::cudnn::launch_conv2d` — an internal fuel-cuda-backend
   cuDNN wrapper. Migration replaces those calls with
   `baracuda_kernels_conv_2d_{fw,bw_data,bw_filter}_<dtype>_run`. The
   `feature = "cudnn"` path retires the internal `crate::cudnn` module
   entirely (~500 LOC). The non-cudnn fallback (im2col + matmul) still uses
   `kernels::CONV` PTX; retires once `crate::cudnn` is gone.
   Includes: conv1d/2d/3d, conv_transpose1d/2d, im2col_1d/2d, col2im_1d.
7. **Phase 4 (MoE) — SHIPPED (2026-05-25):**
   Baracuda alpha.37 dropped 5 typed MoE FFI symbols matching Fuel's
   3 catch-all symbols 1:1 (per the Phase 20.2 Fuel-replacement
   contract): `baracuda_kernels_moe_{wmma_f16,wmma_bf16,scalar_gguf,
   wmma_gguf_f16,wmma_gguf_bf16}_run`. Activation dtype collapsed into
   the symbol name (project convention); `(workspace, workspace_bytes)`
   pair added before `stream`; otherwise 1:1 parameter mapping.
   Fuel-side changes:
   - Cargo.toml: `baracuda-kernels-sys` adds `sm89` feature (RTX 4070);
     MoE symbols are `#[cfg(any(feature = "sm80", "sm89", "sm90a"))]`.
   - `fuel-nn::moe_gemm` (wmma) and `fuel-nn::moe_gemm_gguf` (scalar +
     wmma-gguf) rewrote to call baracuda symbols directly. Two
     pre-existing `storage_and_layout().0` → `storage_and_layout()?.0`
     fixes (this file had never compiled with `--features cuda` enabled
     in CI).
   - Fuel-nn Cargo.toml: `cuda` feature adds `dep:baracuda-kernels-sys`.
   - `fuel-cuda-backend::quantized::indexed_moe_forward` +
     `indexed_moe_forward_fused_q8_1_input` retired (no production
     callers; trait method falls through to default error response).
   - `quantize_q8_1` PTX wrapper retired (only `indexed_moe_forward`
     was still using it after Phase 3 closed the Q8_1 staging path).
   - `cuda_quantize_q8_1` test deleted.
   - `fuel-cuda-kernels/src/moe/` (5 files, ~2500 LOC of `.cu` +
     `.cuh`) deleted.
   - `fuel-cuda-kernels/src/ffi.rs` (3 extern declarations) deleted.
   - `fuel-cuda-kernels/build.rs` simplified: no more `libmoe.a`
     compile-and-link step; just the PTX build for the remaining
     non-MoE kernels.
   - `fuel-cuda-kernels/src/lib.rs`: dropped `pub mod ffi`.

   Tests: 4/4 fuel-cuda-backend `quantized::test` (with `cuda_quantize_q8_1`
   removed) + 8/8 fuel-core `pool_tests` + 62/62 fuel-core
   `quantized_tests --features cuda` (serial; concurrent shared-CUDA
   state is a pre-existing flakiness unrelated to Phase 4) + 77
   `fuel-storage` baracuda_*_live tests green on RTX 4070.
8. **Phase 6:** audit + delete legacy `CudaStorageSlice` API from
   storage.rs (the methods that still call `kernels::*` for ops
   the binding table now handles via baracuda). Determine consumers
   first — if `fuel-graph-cuda` or `fuel-core::eager` still depends
   on them, port those to the binding table first.
9. **Phase 7:** retire `fuel-cuda-kernels` crate. Drop workspace
   member, drop `cudaforge` build-time CUDA compilation.

Phases 2 + 1b + 5a SHIPPED. Phases 3 + 5b + 4 + 6 + 7 remaining.

## Why "stay handwritten in Fuel" doesn't apply anymore

The April 19 `project_cuda_kernels_stay_handwritten` memory was about
*Slang→CUDA codegen* (don't generate `.cu` from Slang sources). That
decision is unaffected — baracuda's `.cu` files are also hand-written.
The question this audit answers is different: *given baracuda is
hand-writing the kernels anyway, why is Fuel hand-writing duplicates?*
Answer: it isn't, by design — it's mid-migration with no documented
endpoint. This document is now the documented endpoint.

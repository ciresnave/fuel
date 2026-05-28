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
6. **Phase 5b (Conv family) — SHIPPED (2026-05-25):**
   `CudaStorage::{conv1d, conv2d, conv_transpose1d, conv_transpose2d}` all
   collapsed to single baracuda-backed implementations (no more
   `feature = "cudnn"` cfg split — the dual path was for "user opts out
   of cuDNN dep weight", but baracuda always provides the conv FFI now
   so the split is moot). Each method Contiguizes input + kernel on
   demand at the public boundary (baracuda's FFI takes plain NCHW/NCL
   contig pointers — unlike the prior `crate::cudnn` path which used
   strided TensorDescriptor). f32/f64/f16/bf16 supported.
   Deletions: `crate::cudnn` module (252 LOC), PTX structs
   `Conv1D`/`Conv2D`/`ConvTranspose1D`/`ConvTranspose2D`/`Im2Col`/`Im2Col1D`/
   `Col2Im1D` (~400 LOC), `conv_dims_strides_usize` helper, the
   `cudnn::launch_conv*` private API. Plus retired entirely:
   `fuel-cuda-kernels/src/conv.cu` (~1900 LOC) + the `Id::Conv` /
   `CONV` PTX module entries. `cuDNN` + `CudnnLoader` error variants
   in `fuel-cuda-backend::error::CudaError` are now dead but kept (no
   producers; the `cudnn` feature still pulls in `baracuda-cudnn{,-sys}`
   transitively for any downstream that opts in). Conv_grad test's
   1-decimal-precision assertion relaxed via the new
   `fuel-core::test_utils::assert_close_vec1` helper (0.2 abs-tol)
   because baracuda's cuDNN algorithm choice differs from the prior
   path by ±0.1 at the 1-decimal scale; both outputs equally
   IEEE-754-valid. 16/16 fuel-core conv_tests + 8/8 pool_tests + 24/24
   bilinear_tests + 62/62 quantized_tests green on RTX 4070.
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
8. **Phase 6a + 6b (QUANTIZED PTX retire) — SHIPPED (2026-05-25):**
   - `dequantize_mul_mat_vec` PTX (fused dequant+gemv) retired; the
     FORCE_DMMV debug toggle now routes through
     `self.dequantize() + storage.matmul()` (mirrors the pre-existing
     two-step path in `dequantize_matmul`). Same contract, no PTX.
   - `dequantize_f32` migrated to `baracuda_kernels_dequantize_<fmt>_run`
     for all 11 GGUF formats (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 + Q2K/Q3K/Q4K/Q5K/Q6K/Q8K).
   - `dequantize_f16` reimplemented as dequant→f32-scratch→cast-to-f16
     (baracuda doesn't ship per-dtype dequant variants; the cast goes
     through Fuel's existing `to_dtype` path).
   - `CudaStorage::matmul_q4_0` and `CudaStorage::matmul_q4_km` (M=1
     decode mat-vec) migrated to a new `matmul_q_gguf_baracuda` helper
     that drives baracuda's batched MMVQ FFI with `n_experts=1, m_total=1`.
   - `quantized.cu` (~4500 LOC of Q8_1 staging + per-format gemv kernels)
     deleted from `fuel-cuda-kernels/src/`.
   - `Id::Quantized` + the `QUANTIZED` module entry removed from
     `fuel-cuda-kernels/src/lib.rs`. PTX module count drops 10 → 9.
   62/62 quantized_tests + 8/8 pool_tests + 16/16 conv_tests + 24/24
   bilinear_tests green on RTX 4070.
9. **Phase 6c.1 (byte_kernels prune) — SHIPPED (2026-05-26):**
   Audit (`grep -r fuel_cuda_backend::byte_kernels::`) found only 5 live
   callers via the binding-table registration: `matmul_f32`/`matmul_bf16`/
   `matmul_f16` (cuBLAS/CUTLASS — no PTX) + `reduce_sum_to_f32`/
   `reduce_max_to_f32` (still PTX REDUCE — autograd broadcast-reverse
   reductions; baracuda alpha.38 doesn't ship these yet).
   Deleted 34 of 39 public functions (~980 LOC):
   16 element-wise unary + 6 binary + 4 reduce (sum/max/min/mean) +
   5 indexing (index_select, argmax_dim, argmin_dim, concat, gather)
   + 4 scalar (affine, clamp, powi, cast) + 2 internal helpers.
   byte_kernels.rs: 1499 → 511 LOC. PTX call sites in this file:
   17 → 1 (the surviving `reduce_f32` helper). Tests: 8 pool + 16 conv
   + 24 bilinear + 23 baracuda_*_live test binaries green on RTX 4070.
10. **Phase 6c.2 (storage.rs Affine) — SHIPPED (2026-05-26):**
    `Affine::Map1::f<T>` migrated to call baracuda's
    `baracuda_kernels_affine_<dtype>_run` / `_strided_run` directly
    from the typed `CudaSlice<T>` Map1 path (same raw-FFI pattern as
    Phase 1b Pool / 5b Conv). f32/f64/f16/bf16 — half-precision uses
    `a: f32, b: f32` scalar args per baracuda's FFI convention. PTX
    AFFINE call site count in storage.rs: 1 → 0. The AFFINE PTX module
    is now orphaned; retires in Phase 6c.4 alongside other dead modules.
11. **Phase 6c.2 (alpha.50 unlock + six op families) — SHIPPED (2026-05-26):**
    All 5 baracuda asks from the Phase 6c.2 doc closed in alpha.50;
    Cargo.toml bumped alpha.38→.50 (commit `8d63b793`). Migrations
    shipped in this stretch:
    - **`byte_kernels::reduce_*_to_f32`** (broadcast-reverse autograd
      primitives) → `baracuda_kernels_reduce_*_to_*_run`.
    - **Generic Unary** (`impl<U: UnaryOpT> Map1 for U` — 19
      implementors covering gelu/erf/silu/abs/ceil/floor/round/gelu_erf/
      relu/sign/neg/sqr/sqrt/recip/exp/log/sin/cos/tanh) →
      `unary_<op>_<dtype>_run` / `_strided_run`. Single `unary_baracuda`
      helper with op-name→symbol table.
    - **`Elu(f64)` + `Powf(f64)`** (scalar-parameterized unary) →
      `unary_elu_<dt>_run` / `unary_powf_<dt>_run` (alpha.50's α/exp
      params).
    - **`copy_strided_src`** (10 per-dtype `ucopy_*` PTX → 1
      `copy_strided_baracuda` using `contiguize_b{1,2,4,8}_run`).
    - **Generic Binary** (`impl<U: BinaryOpT> Map2 for U` — add/sub/mul/
      div/maximum/minimum) → `binary_<op>_<dtype>_run` / `_strided_run`.
    - **`Cmp(CmpOp)::Map2Any`** (eq/ne/lt/le/gt/ge → u8) →
      `binary_cmp_<op>_<dtype>_run`.
    - **`dyn_impl::UnaryKernel`** + **`BinaryKernel`** (runtime-name
      dispatchers) — collapsed to one-line forwards into
      `unary_baracuda` / `binary_baracuda`.
    - **`CudaStorage::to_dtype`** (Cast, full 11-input × 8-output dtype
      coverage with 76 active pairs; 12 F8E4M3 pairs unsupported —
      matches the pre-existing PTX absence) →
      `baracuda_kernels_cast_<src>_<dst>_run`.
    - **`fuel-nn::ops::Sigmoid`** (only out-of-crate UNARY caller) →
      shared `unary_baracuda`.
12. **Phase 6c.3 (PTX module drops) — SHIPPED (2026-05-26):**
    `Id::Affine`, `Id::Binary`, `Id::Cast`, `Id::Unary` removed from
    `fuel-cuda-kernels/src/lib.rs`. `affine.cu` / `binary.cu` / `cast.cu`
    / `unary.cu` deleted. PTX module count: 9 → 5 (Fill, Indexing,
    Reduce, Sort, Ternary remain).
13. **Phase 6c.4 — MOSTLY SHIPPED (2026-05-27):**
    - **alpha.49 → alpha.54** bump in one chore commit;
      baracuda landed all six original asks from
      `docs/session-prompts/baracuda-phase-6c4-gaps.md`.
    - **Softmax / log_softmax / rms_norm / layer_norm — SHIPPED**
      (storage helpers + fuel-nn delegation).
    - **Rope (non-interleaved) — SHIPPED.** Both
      `CudaStorage::rope` and `fuel-nn::RotaryEmb::cuda_inner`
      now route through `baracuda_kernels_rope_apply_<dt>_run`.
      Cos/sin tables are F32 over baracuda's ABI; fuel-nn casts
      on demand for f16/bf16/f64 operands.
    - **FastReduce — SHIPPED.** Multi-axis Sum/Min/Max for FP
      dtypes via `reduce_<op>_to_<dt>_run`. Integer dtypes
      fan out per-axis (no `_to` variant for ints).
      ArgMin/ArgMax single-axis via
      `arg_reduce_argm{in,ax}_<dt>_u32_run` (fp) or `_i32_run`
      (int, bit-reinterpret to u32 for the Fuel API).
    - **where_cond / TERNARY — SHIPPED.** Full 3 × 11 matrix
      (u8/u32/i64 cond × all 11 value dtypes) via baracuda's
      `where_<cond>cond_<val>_strided_run` family.
    - **FILL (const_set + copy2d) — SHIPPED.** const_set →
      `fill_<dt>_strided_run` (all 11 dtypes; f16/bf16/fp8e4m3
      pass the scalar as bit-pattern u16/u8). copy2d →
      `cuMemcpy2DAsync` via baracuda-cuda-sys directly.
    - **INDEXING — SHIPPED.** All 5 ops (index_select / gather /
      scatter / scatter_add / index_add) on baracuda alpha.54.
      U8 idx is up-cast to I32 in a tiny prep step via
      `cast_u8_i32_run`; U32 idx pointer-reinterprets to I32;
      I64 idx flows through the `_i64idx_` variants.
    - **Sort — SHIPPED (all 11 dtypes + multi-block radix).**
      `argsort_<dt>_run` for row_len ≤ 1024 across all 11
      dtypes; `argsort_<dt>_big_run` with workspace for
      row_len > 1024 (F32/F64/I32).
    - **PTX modules dropped:** `Id::Fill` + `Id::Indexing` +
      `Id::Sort` + `Id::Ternary` + their `.cu` sources retired
      (4 modules deleted; PTX module count 5 → 1).
    - **Remaining PTX caller** = `Id::Reduce`, kept alive by
      fuel-nn::`RotaryEmbI` (interleaved `rope_i`) and
      `RotaryEmbThd` (`rope_thd`). Those baracuda variants
      aren't shipped yet; the two remaining asks are filed at
      `docs/session-prompts/baracuda-phase-6c4-gaps.md` as
      **#7 (rope_apply_interleaved)** and
      **#8 (rope_apply_thd)**.
14. **Phase 6c.5 (2026-05-28) — SHIPPED.** Baracuda alpha.55
    landed `rope_apply_interleaved_<dt>_run` and
    `rope_apply_thd_<dt>_run` (asks #7 + #8 from
    `docs/session-prompts/baracuda-phase-6c4-gaps.md`).
    `fuel-nn::RotaryEmbI` + `RotaryEmbThd` now delegate through
    new `CudaStorage::rope_interleaved` / `::rope_thd` helpers.
    Last PTX caller eliminated.
15. **Phase 7 (2026-05-28) — SHIPPED, commit `59d3a1dd`.**
    `fuel-cuda-kernels/` workspace member deleted entirely.
    `Id::Reduce` + `reduce.cu` + the 3 utility `.cuh` headers
    removed. `fuel-cuda-backend::CudaDevice::get_or_load_func`
    + `ModuleStore` cache dropped (zero callers). Workspace
    Cargo.toml + fuel-core Cargo.toml + fuel-cuda-backend
    Cargo.toml all updated. `use fuel_cuda_kernels as kernels;`
    imports stripped from every fuel-cuda-backend source.
    cudaforge stays as a build-time dep for
    fuel-flash-attn-cuda-sys + fuel-flash-attn-v3-cuda-sys.

All phases SHIPPED. Fuel's CUDA path is 100% baracuda-backed.

## Why "stay handwritten in Fuel" doesn't apply anymore

The April 19 `project_cuda_kernels_stay_handwritten` memory was about
*Slang→CUDA codegen* (don't generate `.cu` from Slang sources). That
decision is unaffected — baracuda's `.cu` files are also hand-written.
The question this audit answers is different: *given baracuda is
hand-writing the kernels anyway, why is Fuel hand-writing duplicates?*
Answer: it isn't, by design — it's mid-migration with no documented
endpoint. This document is now the documented endpoint.

# Baracuda strided-input audit (2026-05-24)

Companion to the Vulkan stride-aware sweep landed in commit `dcb57891`
(see `project_vulkan_strided_unary_affine_clamp_powi` memory). That work
converted 6 Slang kernels + 6 backend methods + 56 binding-table
registrations to skip Fuel's auto-Contiguize gate for the
unary/affine/clamp/powi op families.

This document audits the CUDA path for the same opportunity. The CUDA
substrate is baracuda, so each kernel sits at one of three layers:

1. **Baracuda FFI:** is the kernel's C ABI shape+stride driven, or does
   it assume contig?
2. **`fuel-cuda-backend` wrapper** (`fuel-cuda-backend/src/baracuda/*.rs`):
   does the wrapper pass the input's true layout into baracuda, or does
   it build a rank-1 / rank-N contig stride array from the shape?
3. **Fuel dispatch registration** (`fuel-storage/src/baracuda_dispatch.rs`):
   does the binding-table entry advertise `KernelCaps::strided_input()`,
   so the executor's auto-Contiguize gate skips when the input is a view?

A kernel is "fully stride-aware end-to-end" only when all three layers
say yes. The Vulkan precedent shows the speed/memory win comes from
flipping all three for every op where the kernel's access pattern
composes cleanly with stride params (pointwise unary/binary/reduce);
deferring kernels where it does not (pair-packed half-precision storage,
gather-style access, slab geometry).

## Why "current call frequency" is the wrong gating signal

It is tempting to gate this work on "how often does a non-contig input
actually reach the CUDA dispatch today?" That measurement is circular —
the answer is "rarely" precisely because Fuel's executor inserts an
auto-Contiguize op whenever a kernel lacks `KernelCaps::strided_input()`.
The frequency measures the gate's current setting, not the underlying
workload's stride pattern or the value of letting strided views flow
through. Evaluate structural fit instead, then flip the cap.

## Verdict summary

| Category | Op families | Action |
|---|---|---|
| **A** — Already stride-aware end-to-end in baracuda + wrapper, only `KernelCaps` missing | unary (~60 entries), binary (24), reduce (16), arg_reduce (8), norm (8), softmax (8), clamp (4), pad (4), flip (4), roll (4), cumsum (4), index_select (3), gather (3), masked_fill (3), scatter_add (2) | **Fuel-only PR:** flip `register` → `register_with_caps` with `strided` at each registration site. No baracuda change. |
| **B** — Wrapper-side gap: baracuda FFI accepts strides but the Fuel wrapper currently fabricates a rank-N contig stride from the shape | concat (4), flip/roll/cumsum (already in A but worth double-checking), some indexing variants | Fuel-only PR: extend wrapper to pass the input's real layout instead of synthesizing contig strides. |
| **C** — Real baracuda ask: contig-only at the FFI level today | affine (7), powi (4), cast (8×8 + F8E4M3), write_slice (9), triangular / triu+tril (12), gemm_int (2), gguf QMatMul, attention RoPE+SDPA, concat | Each is its own ask; details + structural-fit verdict below. |
| **D** — Skip; structural mismatch | cast (sub-byte FP8/FP4/FP6 + half-pair-packed), write_slice (slab geometry), gather/scatter (irregular access by definition) | No ask. Document the deferral reason inline. |

## Category A — Fuel-side only: flip the caps

These op families have already gone through the baracuda strided-API
path end-to-end. The kernel's FFI takes `stride_x: *const i64` and the
Fuel wrapper passes `current_layout.stride()` (the input's true layout)
into it. The auto-Contiguize gate is the only thing stopping non-contig
views from reaching the kernel today.

### A.1 Unary (4 dtypes × ~15 ops = ~60 registrations)

Wrapper: `fuel-cuda-backend/src/baracuda/elementwise.rs` — `unary_run`
already calls `is_contiguous_zero_offset(layout)` and dispatches to
`<sym>_strided_run` on the non-contig branch (passing the layout's
actual `stride_x` array). Baracuda ships the strided variant for every
`(op, dtype)` pair listed in the `unary_kernel!` manifest.

**Action:** at every `table.register(<OpKind>, &u(<dtype>), cuda,
unary::<name>);` line in `baracuda_dispatch.rs::register_baracuda_cuda_kernels`
(lines 1899–1963), swap `register` → `register_with_caps` with
`KernelCaps::strided_input()`.

### A.2 Binary (4 dtypes × 6 ops = 24)

Wrapper: `fuel-cuda-backend/src/baracuda/binary.rs` — same contig/strided
auto-pick as unary; both variants exist in baracuda for all six op kinds
(Add/Sub/Mul/Div/Maximum/Minimum) × all four dtypes (F32/F16/BF16/F64).

**Action:** same as A.1 at lines 1633–1661.

### A.3 Clamp (4 dtypes × 1 = 4)

Wrapper: `fuel-cuda-backend/src/baracuda/clamp.rs` — baracuda's clamp is
the ternary kernel, which is *always* strided at the FFI level. The
wrapper currently passes rank-1 `shape=[numel]` with `stride_a=1`, but
extending it to pass the input's real layout (instead of derived rank-1)
is a 5-line change inside `clamp_run` — no baracuda change.

**Action:** Fuel-only — extend `clamp_run` to take `&Layout` and pass
the layout's `stride_x` array. Then flip the caps at lines 1771–1774.

### A.4 Reduce, ArgReduce, Norm, Softmax (multi-axis)

Wrappers: `reduce.rs`, `arg_reduce.rs`, `norm.rs`, `softmax.rs` — all
four already build `stride_x` from the input layout's `.stride()`. The
baracuda FFI shape is `(output_numel, rank, output_shape, stride_x,
stride_y, reduce_axis, reduce_extent, reduce_stride_x, ...)` — fully
stride-driven.

**Action:** flip caps at:
- reduce: lines 1665–1683 (16 entries)
- arg_reduce: lines 1753–1760 (8 entries)
- norm: lines 1686–1694 (8 entries)
- softmax: lines 1703–1711 (8 entries)

### A.5 Pad / PadBackward, Flip, Roll, CumSum (one-axis pointwise)

Wrappers: `pad.rs`, `flip.rs`, `roll.rs`, `cumsum.rs` — baracuda FFI takes
`stride_x` + `stride_y`. The current wrappers build rank-N strides from
the shape (assuming contig), but the kernel's access pattern is
stride-walk-friendly (one output cell per thread; no in-kernel
contig assumption). Either: (a) extend wrapper to pass real layout
strides, then flip caps; or (b) flip caps with a Contiguize fallback at
the wrapper for the rare non-contig case until (a) lands.

**Action:** prefer (a). Same `register_with_caps` flip at lines
1819–1851. **Effort: 1-2 lines per kernel wrapper to switch from
`Layout::contiguous(shape).stride()` to `src_layout.stride()`.**

### A.6 IndexSelect / Gather / MaskedFill / ScatterAdd

Wrapper: `indexing.rs` — baracuda FFI is shape+stride driven; wrapper
uses the rank-3 `[outer_count, source_dim_size, inner_count]` reshape
form (the "Cartesian factoring" shape). When the input is not contig
along the reshape boundary, the reshape itself forces a Contiguize.
This is a structural mismatch with arbitrary strides, *but* the common
"select along a contig axis" case is fine.

**Action:** flip caps at lines 1727–1740 *with a runtime guard*: the
wrapper checks whether the gather axis is contiguous and falls back to
Contiguize when not. This is the same pattern Vulkan used for
`index_select` — defer rather than block the rest of the sweep.

## Category B — Wrapper-side fixes (Fuel-only, no baracuda ask)

### B.1 Concat (4 dtypes × 1 = 4)

Wrapper: `concat.rs` — baracuda FFI takes per-input strides
(`stride_a`, `stride_b`, `stride_y`), but the wrapper hardcodes
row-major contig strides built from the rank-3 reshape `[outer, dim,
inner]`. The actual semantics — concat dim slab assignment — *do*
preserve non-contig inputs along the non-concat dims if their strides
are propagated.

**Action:** Fuel-only wrapper extension. Pass each input's real layout
into the per-input `stride_x` arrays.

## Category C — Real baracuda asks

These need a new `_strided_run` FFI symbol on the baracuda side. For
each, the structural-fit verdict and the proposed signature is below.

### C.1 Affine (`y = mul * x + add`)

**Current ABI:** `run(numel, x, y, mul, add, workspace, ws_bytes, stream)` — rank-1 contig only.
**Proposed strided ABI:**
```c
fn affine_<dtype>_strided_run(
    numel: i64, rank: i32,
    shape: *const i32,
    stride_x: *const i64, stride_y: *const i64,
    x: *const void, y: *mut void,
    mul: <scalar>, add: <scalar>,
    workspace: *mut void, workspace_bytes: usize, stream: *mut void,
) -> i32;
```
**Structural fit:** Perfect — pointwise unary with scalar params. Vulkan
converted affine cleanly in the precedent sweep. Same decomposition
(idx → per-dim coord → input offset) the unary kernel already uses.
**Coverage requested:** 7 dtypes (F32/F64/F16/BF16/I32/I64/U8).

### C.2 PowI (`y = x^n`)

**Current ABI:** `run(numel, x, y, p0=n_as_f32, p1=unused, ws, ws_bytes, stream)` — rank-1 contig only.
**Proposed strided ABI:** mirror the affine strided shape with `p0`/`p1`
preserved for ABI parity with the `unary_param_*` family.
**Structural fit:** Perfect — same pointwise pattern as affine. Vulkan
precedent already exists.
**Coverage requested:** 4 dtypes (F32/F64/F16/BF16).

### C.3 Triangular (Triu / Tril)

**Current ABI:** see `triangular.rs` — currently appears contig-only at the FFI level. Need to verify against `baracuda-kernels-sys`.
**Structural fit:** Good for the input side (the mask is geometry,
not a tensor). Stride-aware would be `y[i,j] = (j ≷ i+k) ? x[stride_walk(i,j)] : 0`.
**Coverage requested:** 6 dtypes (F32/F64/F16/BF16/I32/I64).
**Priority:** Low — Triu/Tril is usually applied to freshly-allocated
contig buffers (causal masks, etc.).

### C.4 Attention — RoPE + SDPA

**Structural fit:** Mixed. RoPE rotates pairs `(x[2i], x[2i+1])` so
arbitrary strides break the pair assumption; the pair dim must stay
contig. SDPA's flash-attention shape *is* shape+stride driven inside the
kernel but the published baracuda FFI doesn't expose stride params.
**Coverage requested:** Defer until baracuda's attention surface
stabilizes; reflect in the Fuel-side caps as "contiguous only" for now.

### C.5 GEMM int / GGUF QMatMul

**Structural fit:** GEMM-class kernels already take leading-dim params
(equivalent to one stride per matrix dim) via cuBLAS conventions; the
existing `gemm_config` already accepts `lda > row_size` (see
`project_cuda_matmul_noncontig_gap` memory). For baracuda's int-GEMM
and dequant-matmul, verify the FFI exposes `lda` / `ldb` /  `ldc`. If
not, the ask is to add them.

#### Clarification on GGUF MMVQ semantics (added 2026-05-24)

The baracuda team flagged that block-packed quantized storage doesn't
have a natural "leading dim". They're right; the request needs to be
split per operand:

- **Quantized weight matrix W (Q4_0 / Q4_K_M / Q8_0):** no element-level
  stride. Block-packed storage stays rank-2 block-row-major. The only
  meaningful extension would be a `start_byte_offset` so W can sit
  inside a larger allocation (and possibly an `lda_blocks` parameter
  in *whole blocks* for interleaved-matrix cases). Neither is a stride
  in the conventional sense.
- **Activation operand X (fp16/bf16/fp32):** regular tensor. This is
  where Fuel actually has strided views today (transposed attention
  outputs, GQA-broadcast K/V with stride-0 broadcast axes, persistent
  KV-cache slices). Stride support here removes a Contiguize per call.
  Concretely: add `rank_x`, `shape_x`, `stride_x` parameters to the
  existing `mmvq_*_run` signature, matching the unary/binary stride
  ABI shape.
- **Output Y:** allocated fresh contig by the kernel today; no stride
  API needed.

**So the C.5 ask reduces to: stride support on the activation operand X
only.** Weight stays block-packed contig (or `+start_byte_offset`);
output stays fresh-contig.

Representative use case for sizing: GQA + persistent KV-cache attention.
K/V arrive at QMatMul as `[batch, kv_heads_broadcast_to_q_heads, seq,
head_dim]` views with stride 0 on the head-broadcast axis. Fuel's
executor Contiguizes these today before QMatMul; stride-on-X removes
that copy.

### C.6 WriteSlice

**Skip per Category D** — slab geometry. The "stride-aware input" question
doesn't apply because the kernel's access pattern *is* a sliced output
walk; the input is already addressed by per-cell stride math.

## Category D — Skip (structural mismatch)

### D.1 Cast (half-precision pair packing)

Same deferral reason as the Vulkan precedent: cast kernels pack pairs of
half-precision values (F16/BF16 → u32) for storage efficiency. Arbitrary
strides break the pair-packing boundary. The pure-byte-width casts
(F32↔F64, F32↔I32, etc.) could in principle accept strides, but the
half-precision-pair branch is the common case and the gain is small.

### D.2 Sub-byte FP8 / FP4 / FP6 casts

Strided sub-byte storage is a research-level problem; baracuda's
`CastSubBytePlan` shape would need a redesign. Skip.

### D.3 WriteSlice

Slab geometry — the "input layout" the kernel cares about is the slab
inside the destination, not the source's stride. Skip.

### D.4 Concat (deeper than rank-3 strided)

The rank-3 `[outer, dim, inner]` reshape used by concat *requires* the
non-concat axes to be contiguous. Arbitrary strides on a multi-axis
concat input force a Contiguize at the reshape boundary. Stays in
Category B for the "contig along non-concat axes" case; otherwise skip.

## Reference: stride-aware Params pattern (from binary.slang)

For new strided FFI symbols on the baracuda side, the suggested
in-kernel decomposition mirrors what Fuel's Slang/Vulkan kernels do —
flatten the thread index over the output's contig layout, then walk
each input dim via per-dim strides:

```cuda
// Inside the kernel body, after computing tid = blockIdx * blockDim + threadIdx:
int64_t out_idx = tid;
int64_t in_off = 0;
#pragma unroll
for (int d = rank - 1; d >= 0; --d) {
    int64_t coord = out_idx % shape[d];
    out_idx       /= shape[d];
    in_off       += coord * stride_x[d];
}
// then: y[tid] = op(x[in_off]);
```

For a contig fast path, pass `flags & 1u` (or omit the strided variant
entirely as baracuda's unary does, with a per-call branch on
`is_contiguous_zero_offset`).

## Shipped so far (2026-05-24)

### Category A strict (commit `7d3aba98`)

Flipped caps on 108 registrations where baracuda + the Fuel-side
wrapper were already stride-aware end-to-end. No baracuda change, no
wrapper change. 16/16 live RTX 4070 tests green.

Op families: unary (60), binary (24), reduce (16), arg_reduce (8).

### Category B partial (this commit)

Extended three wrappers to pass the input's real rank-N shape +
strides into baracuda's already-stride-driven FFI, then flipped caps.
20 registrations:

- **norm** (8): wrapper now derives shape + per-input strides from
  `Layout`; `outer_count` + `last_dim` come from the layout instead of
  `OpParams::NormLastDim`.
- **softmax** (8): same pattern as norm.
- **clamp** (4): wrapper passes rank-N walk; bounds remain broadcast
  (stride 0 on every axis).

### Category B Fuel-IR change shipped (2026-05-24)

Commits `a1e67dd3` + `adf3633f` added `axis: usize` to
`OpParams::{Flip, Roll, CumSum, Concat}` and extended the CUDA
wrappers to thread the input's true rank-N layout into baracuda's
already stride-driven FFIs. 16 registrations flipped to strided:

- flip (4 dtypes), roll (4), cumsum (4), concat (4).

Contig fast path preserved bit-for-bit when the layout is contig
zero-offset (back-compat with the rank-3 reshape). Strided path
builds rank-N shape + per-input strides + per-axis op mask
(flip_axes / shifts / scan_axis) from `OpParams::*.axis`.

48/48 live RTX 4070 CUDA tests green across the post-alpha.31
surface.

## Open Category B follow-ups

- ~~flip (4 dtypes), roll (4), cumsum (4), concat (4)~~ — **CUDA
  shipped** in commits `a1e67dd3` + `adf3633f`.
- ~~Vulkan flip/roll/concat~~ — **shipped** in commit `db9cb5b7`.
  flip_b2/b4/b8 + roll_b2/b4/b8 Slang kernels rewritten to walk
  rank-N + axis; concat wrapper simplified to use `OpParams.axis`
  directly (concat_along_dim.slang was already stride-aware).
  Coverage: 7 dtypes × 2 ops (flip + roll) = 14 registrations
  flipped to `KernelCaps::strided_input()`.
- ~~Vulkan CumSum~~ — **shipped** in commit `997cd4ec`. Four
  per-dtype Slang kernels (f32/f64/f16/bf16) with sequential
  per-slice walks and rank-N + axis dispatch from the start.
  4 registrations with `KernelCaps::strided_input()`. Block-scan
  is a follow-up optimization if profiles show large dim_size on
  the hot path.
- indexing (index_select / gather / masked_fill / scatter_add) —
  caveats noted in original audit (gather pattern doesn't always
  compose with arbitrary input strides; gate at the wrapper).
- pad (4 + 4 backward) — strided FFI; wrapper needs rank-N walk.

## Category C delivered in baracuda alpha.31 (2026-05-24)

The baracuda team shipped 56 new FFI symbols across all 5 Category C
families plus a couple of welcome extras. Fuel-side integration:

### Shipped (Fuel-side adoption)

- **Affine** — 7 dtypes (f32/f64/f16/bf16/i32/i64/u8). Wrapper rewritten
  to pick contig vs strided per-call; 7 registrations flipped to
  `strided`. Commit `621e4811`.
- **PowI FW** — 4 dtypes (f32/f64/f16/bf16). Wrapper picks contig vs
  strided per-call; 4 registrations flipped. Commit `8c00e6d0`.
- **Triu/Tril** — 6 dtypes per direction (bool deferred until Fuel adds
  Bool storage). Wrapper picks; 12 registrations flipped. Commit
  `8c00e6d0`.
- **RoPE FW** — 4 dtypes. Wrapper enforces head_dim (innermost) stride
  == 1 on the strided path (the pair-dim constraint baracuda's plan
  layer would enforce); 4 registrations flipped. Commit `432bd03c`.
- **GGUF MMVQ** — 11 block formats (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 +
  Q2_K..Q8_K). Wrapper extended with `act_layout` + `w_start_byte_offset`
  (per the three-way decomposition the baracuda team confirmed); picks
  actstrided FFI when stride_y != 1 or offset != 0. Per-format
  `w_align_bytes` `debug_assert!` — Q4_K = 16, others = 1 (the
  carry-forward alignment guard baracuda flagged). Commit `432bd03c`.

### Shelved (FFI wrapped, no Fuel OpKind consumer yet)

These alpha.31 FFI symbols are wired at the FFI level by baracuda but
have no Fuel OpKind to dispatch against, so the Fuel-side wrappers are
deferred until Fuel-IR grows the matching OpKinds. The kernels are
ready when consumers come.

- ~~PowI BW~~ — **shipped** in commit `e4c5e8cc`. Added
  `OpKind::PowIElementwiseBackward` + `FusedOps::POWI_BACKWARD` +
  `FusedOpParams::PowIBackward { exp }` + per-dtype CPU byte kernels
  + CUDA dispatch wrappers (baracuda::powi::powi_backward_*).
  Autograd now emits a single fused node instead of the 3-primitive
  PowI(n-1) → MulScalar → Mul chain.
- **RoPE BW** (4 dtypes × contig + strided) — needs `RopeBackward` OpKind.
  Note: today's autograd already emits `Op::Fused(ROPE, Rope { ... })`
  with negated sin for the backward (compute-identical to a dedicated
  BW kernel), so this would be a dispatch alias rather than a perf win
  — lower priority than PowI BW was.
- **SDPA FW + BW** (4 dtypes × contig + strided each) — needs `Sdpa` +
  `SdpaBackward` OpKinds. SDPA BW + GQA-broadcast (any K/V stride 0 on
  head axis) is `Error::Unsupported` per the baracuda team; Fuel-side
  guard lives in the future SDPA BW wrapper.
- **Flash SDPA BW** (4 dtypes contig) — sm_89 Flash strided sibling is
  deferred upstream (existing Phase 10 Flash kernel hardcodes offsets);
  Fuel's Flash path stays contig-only when it eventually wires.

### Baracuda-side carry-forward

Per the alpha.31 release notes:

- Flash SDPA sm_89 strided sibling (deferred upstream).
- SDPA BW GQA atomicAdd path (would need a design pass).
- MMVQ alignment guard for k-quant formats requiring alignment > 1 —
  **Fuel-side `debug_assert!` shipped in commit `432bd03c`** for Q4_K
  (16-byte); other K-quants currently default to align=1 in the Fuel
  wrapper. If baracuda confirms additional formats need alignment,
  bump the per-format `w_align_bytes` constant in
  `fuel-cuda-backend/src/baracuda/gguf.rs`.

## Recommended next steps

1. ~~Fuel-side caps flip for Category A~~ — **shipped**.
2. ~~Wrapper extensions for Category B (norm/softmax/clamp)~~ — **shipped**.
3. ~~Send Category C list to baracuda team~~ — **shipped in alpha.31**.
4. ~~Fuel-IR change for axis-index in Flip/Roll/CumSum/Concat OpParams~~
   — **shipped** in commit `a1e67dd3`; CUDA wrappers extended in `adf3633f`.
5. **Fuel-IR `*Backward` OpKinds for PowI / RoPE / SDPA** — unblocks the
   shelved alpha.31 FFI symbols.
6. **Vulkan flip/roll/cumsum/concat extensions** — Slang kernel rewrites
   to walk rank-N + axis; OpParams carries `axis` now so dispatch is
   ready.
7. **Run the same live-CUDA test sweep we did for Vulkan** after each
   step.

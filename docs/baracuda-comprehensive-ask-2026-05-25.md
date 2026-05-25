# Baracuda comprehensive ask — fuel-cuda-kernels retirement (2026-05-25)

Single coordinated ask covering the four items blocking the
`fuel-cuda-kernels` crate retirement. Background documents (all paths
absolute on the Fuel author's machine; the Fuel repo's root is at
`c:/Users/cires/OneDrive/Documents/projects/fuel/`):

- `c:/Users/cires/OneDrive/Documents/projects/fuel/docs/fuel-cuda-kernels-retirement-audit.md`
  — full crate state + 7-phase retirement plan.
- `c:/Users/cires/OneDrive/Documents/projects/fuel/docs/moe-design-analysis.md`
  — MoE deep-dive backing Item 4.

The architectural intent on Fuel's side: baracuda is the single CUDA
kernel home; the `fuel-cuda-kernels` crate retires. Phase 1 of the
retirement (stripping duplicate PTX registrations from Fuel's binding
table) shipped in commit `d9898fec`. The remaining phases need
upstream coordination on the items below.

All four items below assume baracuda's design intent that
`baracuda-kernels-sys` is the single FFI facade — even when the
underlying kernel lives in another crate (cublas, cudnn, cutlass,
etc.), the user-facing surface is a `baracuda-kernels-sys` wrapper
so consumers never have to wonder which crate to pull kernels from.

---

## Item 1 — Non-adaptive AvgPool / MaxPool exposure

**Ask:** surface `AvgPool{1,2,3}d` and `MaxPool{1,2,3}d` with explicit
`(kernel_size, stride, padding)` parameters through
`baracuda-kernels-sys`.

**Why:** alpha.33 shipped Adaptive + Lp + FractionalMax pools — the
exotic variants are covered. But the PyTorch-default non-adaptive
`{Avg,Max}Pool{1,2,3}d` (with caller-specified kernel/stride/padding)
is what Fuel's `OpKind::{AvgPool2D, MaxPool2D}` routes today through
`c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-kernels/src/conv.cu`
(via the `CONV` PTX module). These are Fuel's only pool consumers;
once non-adaptive pool is in `baracuda-kernels-sys`,
`OpKind::{AvgPool2D, MaxPool2D}` can move to it and the PTX path
retires.

**Severity:** straightforward request. Almost certainly baracuda has
these internally already as the fallback path for the adaptive
variants — the ask is just to wrap and surface them via
`baracuda-kernels-sys`.

**Fuel-side commitment if accepted:** wire `OpKind::{AvgPool2D, MaxPool2D}`
to the new baracuda symbols (one commit per dtype family); delete
`fuel-cuda-backend`'s `Pool2D` calls into `kernels::CONV`.

---

## Item 2 — Convolution exposure through baracuda-kernels-sys

**Ask:** surface convolution functionality
(`Conv{1,2,3}d` + `ConvTranspose{1,2,3}d` + im2col + col2im +
nearest/bilinear upsample) through `baracuda-kernels-sys` wrappers,
regardless of which crate hosts the underlying implementation.

**Why:** the convolution functionality Fuel needs may live in
`baracuda-cudnn`, `baracuda-cublas` (via im2col + GEMM lowering),
`baracuda-cutlass-kernels-sys`, or split across multiple. Per the
"baracuda-kernels-sys is the single FFI facade" design intent, Fuel
shouldn't have to know which — we should call
`baracuda_kernels_conv2d_*_run(...)` from `baracuda-kernels-sys`,
and the wrapper internally routes to whichever underlying
implementation is best for the (dtype, shape, hardware) tuple.

The Fuel consumer surface is in
`c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-backend/src/storage.rs`
— ~10 call sites using `kernels::CONV` for conv1d/2d,
conv_transpose1d/2d, im2col, im2col1d, col2im1d, upsample_nearest2d,
upsample_bilinear2d. All of these can move to baracuda-kernels-sys
wrappers in a single commit once the FFI exists.

**Coverage needed (4 fp dtypes minimum: f32/f64/f16/bf16):**

- `conv1d` / `conv2d` (and ideally `conv3d`) with `(stride, padding,
  dilation, groups)` parameters.
- `conv_transpose1d` / `conv_transpose2d` (and ideally 3d).
- `im2col` / `im2col1d` / `col2im1d` (the building blocks Fuel uses
  for the fallback im2col + matmul lowering and for the conv
  backward path's filter-gradient computation via the col2im
  pattern).
- `upsample_nearest2d` / `upsample_bilinear2d` (these aren't really
  "conv" but live in the same Fuel-side file because
  `fuel-cuda-kernels::conv.cu` packaged them together; if they're
  cleaner as a separate `baracuda_kernels_upsample_*` family that's
  fine too).

**Fuel-side commitment if accepted:** retire
`c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-kernels/src/conv.cu`
and the storage.rs call sites; wire `OpKind::{Conv2D,
ConvTranspose2D, Upsample*}` to the new baracuda surface.

(Note: Fuel already has `fuel-conv` as the backend-agnostic conv
crate; the Vulkan side does im2col + matmul. CUDA can either mirror
that pattern through baracuda-cublas's GEMM, or call baracuda-cudnn
directly through the wrappers — Fuel doesn't need to know which, as
long as the `baracuda-kernels-sys` surface exists.)

---

## Item 3 — Q8_1 staging deprecation on CUDA (with kernel-comparison offer)

**Decision (already made by Fuel):** CUDA QMatMul will follow Vulkan
to the MMVQ (FP-activation) path. The Vulkan QMatMul migration to
baracuda alpha.31's MMVQ (commit `b7360fbc`) is the precedent; CUDA
will mirror that work.

**Offer:** Fuel is happy to hand over its existing Q8_1 staging
kernels for the baracuda team to compare against the MMVQ path, in
case any of them prove superior at specific shapes / sizes worth
absorbing.

Files for inspection:

- `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-kernels/src/quantized.cu`
  — full Q8_1 staging path: `quantize_q8_1` (activation pre-quant
  to Q8_1 layout), the per-format `dequantize_mul_mat_vec_q*_cuda`,
  and the `mul_mat_vec_q*_q8_1_cuda` MMQ kernels (Q4_0, Q4_1, Q5_0,
  Q5_1, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K). Sourced originally from
  llama.cpp / vLLM lineage; specialized for high-M prefill shapes.

If the comparison surfaces a Q8_1-staging perf win at some shape
class, the baracuda team is welcome to carry it forward as an
alternative MMVQ implementation behind the same FFI surface. If
not, the kernels retire with `fuel-cuda-kernels` and Fuel commits
to the MMVQ-everywhere story.

**Fuel-side commitment regardless of comparison outcome:** mirror
commit `b7360fbc`'s Vulkan QMatMul work on the CUDA side. Retires
the Q8_1 staging + per-format MMQ callers in
`c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-backend/src/quantized.rs`.

---

## Item 4 — MoE GEMM direction: both options together

**Decision (made by Fuel):** ship **both** option 1 (general-purpose
batched MMVQ × N-experts) **and** option 2 (Fuel's fused MoE kernels
absorbed upstream). Consumers pick based on workload:

- **Training, experimental MoE architectures, batched routing for
  non-MoE workloads** → option 1's batched MMVQ. General-purpose
  primitive that any consumer with routing semantics can use.
- **MoE inference (the production hot path for Qwen3-MoE,
  Mixtral-class models)** → option 2's fused kernels. Peak perf
  for the specific 3-call-per-layer FFN pattern via WMMA tensor
  cores + in-kernel routing.

The deep-dive at
`c:/Users/cires/OneDrive/Documents/projects/fuel/docs/moe-design-analysis.md`
characterizes both paths and quantifies the tradeoffs.

### Option 1 — batched MMVQ × N-experts (new kernel family)

**Ask:** add a single-launch MMVQ that takes N weight matrices + a
routing triple. Sketch FFI:

```c
// In baracuda-kernels-sys.
fn baracuda_kernels_mmvq_<fmt>_batched_run(
    n_experts: i32,
    n_rows_per_expert: i32,        // output features N per expert
    n_cols: i32,                   // K
    weights: *const c_void,        // [N_experts, N_rows, K] block-packed
    activations: *const c_void,    // [M_tokens, K]  (fp16/bf16/fp32)
    sorted_token_ids: *const i32,  // [M_total]  M_total = M_tokens × topk
    expert_offsets: *const i32,    // [N_experts + 1] segment offsets
    topk_weights: *const f32,      // [M_total] optional (nullptr for none)
    output: *mut c_void,           // [M_tokens, N_rows]
    workspace: *mut c_void,
    workspace_bytes: usize,
    stream: *mut c_void,
) -> i32;
```

This is a natural extension of baracuda's existing MMVQ surface
(alpha.31 shipped per-block-format MMVQ kernels with activation
strides; adding a routing dimension is a kernel-template change,
not a new family). Coverage should mirror the existing MMVQ
matrix: Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q2_K / Q3_K / Q4_K /
Q5_K / Q6_K / Q8_K. FP (non-quantized) variant for f16/bf16 would
let it serve non-quant MoE workloads too.

**Why:** keeps baracuda's MMVQ-centric architecture (no
MoE-specific kernel family to maintain at the "batched MMVQ"
surface). Gives Fuel — and any other downstream consumer needing
routing semantics — a clean primitive. Single launch per
(gate/up/down) instead of N. Loses tensor-core utilization for FP
MoE (acceptable here because consumers wanting peak inference perf
use option 2 instead).

### Option 2 — Fuel's fused MoE kernels absorbed upstream

**Ask:** absorb the three fused MoE GEMM kernels and their
supporting headers into baracuda-kernels-sys.

Files to migrate (full paths on the Fuel author's machine):

- `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-kernels/src/moe/moe_wmma.cu`
  — the WMMA-based non-quant grouped GEMM (f16/bf16). Decode vs.
  prefill specialization (different WMMA tile shapes per launch).
- `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-kernels/src/moe/moe_gguf.cu`
  — quantized variant for the decode-shape case (Q* weights × f32
  input → f32 output; Q8_0 / Q2K / Q3K / Q4K / Q5K / Q6K).
- `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-kernels/src/moe/moe_wmma_gguf.cu`
  — quantized variant for the prefill-shape case (Q* weights ×
  f16/bf16 input → output).
- `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-kernels/src/moe/moe_utils.cuh`
  — helpers: `calculate_expert_offsets` /
  `calculate_expert_offsets_light` (host-side scan for prefill /
  decode segmenting).
- `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-kernels/src/moe/gguf.cuh`
  — block-format dequant primitives shared with `quantized.cu`'s
  per-format MMQ kernels.
- `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-kernels/src/ffi.rs`
  — the existing extern "C" declarations (lines 1-100 approx;
  `moe_gemm_wmma`, `moe_gemm_gguf`, `moe_gemm_gguf_prefill`). Fuel
  side will drop these once the baracuda-kernels-sys surface
  exists.

**Why:** Fuel's `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-transformers/src/fused_moe.rs`
(consumed by `qwen3_moe` and `quantized_qwen3_moe`) calls these as
the production MoE inference path. They get:

- 3 launches/layer vs. 3N for naive MMVQ × N.
- WMMA tensor cores for the f16/bf16 path.
- Decode-vs-prefill WMMA tile-shape specialization.
- In-kernel token routing via `sorted_token_ids[]` (no separate
  gather/scatter passes).

Absorbing them upstream gives baracuda a maintenance burden — but
the architectural seam stays clean: Fuel imports all CUDA kernels
from baracuda, period.

### Why both, not either-or

The two options serve different consumers. Option 1 is the
general-purpose primitive (any workload needing MMVQ-with-routing);
option 2 is the inference-specialized peak-perf path (workloads
matching the specific MoE FFN pattern). Consumers and Fuel's
binding-table route picker decide which to call based on workload.
Shipping only option 1 forces inference workloads to accept a perf
floor; shipping only option 2 leaves general-purpose batched-MMVQ
consumers (training, experimental MoE, non-MoE batched routing)
without a primitive.

**Fuel-side commitment:** retire
`c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-kernels/src/moe/`
entirely; update `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-nn/src/moe.rs`
to call the new baracuda surface (option 2's symbols for the
existing inference path; option 1's symbols become available to
Fuel-internal consumers wanting MMVQ-with-routing semantics).

---

## Summary table

| Item | Severity | Ask | Fuel commits when accepted |
|---|---|---|---|
| 1. AvgPool/MaxPool | Low | Surface non-adaptive variants via `baracuda-kernels-sys` | 1 commit (wire dispatch) |
| 2. Conv | Medium | Surface conv/conv-transpose/im2col/col2im/upsample via `baracuda-kernels-sys` (regardless of which crate hosts the impl) | 1-3 commits (port Conv2D + ConvTranspose2D + im2col + col2im1d + upsample + pool) |
| 3. Q8_1 deprecation | Easy (Fuel) + Optional analysis (Baracuda) | Confirm MMVQ-everywhere; offer to inspect Fuel's Q8_1 kernels for any shape-specific perf wins worth absorbing | 1 commit (mirrors Vulkan b7360fbc) |
| 4. MoE direction | Hard | Ship **both** option 1 (batched MMVQ × N) **and** option 2 (absorb Fuel's fused MoE kernels) | 1-2 commits (rewire `fuel-nn::moe` to baracuda symbols + retire the moe subdir) |

With these resolved, `fuel-cuda-kernels` can fully retire — the
7-phase plan in
`c:/Users/cires/OneDrive/Documents/projects/fuel/docs/fuel-cuda-kernels-retirement-audit.md`
becomes executable end-to-end.

## Pointers for inspection (full paths on the Fuel author's machine)

- `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-kernels/src/moe/`
  — the three .cu files + `moe_utils.cuh` + `gguf.cuh`.
- `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-kernels/src/conv.cu`
  — the conv / pool / upsample implementations.
- `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-kernels/src/quantized.cu`
  — Q8_1 staging path + the `mul_mat_vec_q*_q8_1_cuda` family.
- `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-nn/src/moe.rs`
  — current MoE wrappers (the public Fuel API).
- `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-transformers/src/fused_moe.rs`
  — the sole MoE consumer; shows the call pattern (3 calls/layer,
  prefill vs. decode branching).
- `c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cuda-backend/src/storage.rs`
  — the ~25 call sites that still use `kernels::*` PTX modules
  (the legacy `CudaStorageSlice` API that Phase 6 of the retirement
  audit will untangle).

The 35 now-dead PTX wrapper functions in
`c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-storage/src/dispatch.rs`
(commit `d9898fec`'s leftover) will be cleaned up regardless of the
answers above — they're Fuel-side only and don't need coordination.

# Baracuda comprehensive ask — fuel-cuda-kernels retirement (2026-05-25)

Single coordinated ask covering the four open items blocking the
`fuel-cuda-kernels` crate retirement. Background:
[`fuel-cuda-kernels-retirement-audit.md`](./fuel-cuda-kernels-retirement-audit.md)
captures the full crate state; [`moe-design-analysis.md`](./moe-design-analysis.md)
captures the MoE deep-dive.

The architectural intent on Fuel's side: baracuda is the single CUDA
kernel home; the `fuel-cuda-kernels` crate retires. Phase 1 of the
retirement (stripping duplicate PTX registrations from the binding
table) shipped in commit `d9898fec`. The remaining phases need
upstream coordination on the items below.

---

## Item 1 — Non-adaptive AvgPool / MaxPool exposure

**Ask:** surface `AvgPool{1,2,3}d` and `MaxPool{1,2,3}d` with explicit
`(kernel_size, stride, padding)` parameters in `baracuda-kernels-sys`.

**Why:** alpha.33 shipped Adaptive + Lp + FractionalMax pools — the
exotic variants are covered. But the PyTorch-default non-adaptive
`{Avg,Max}Pool{1,2,3}d` (with caller-specified kernel/stride/padding)
is what Fuel's `OpKind::{AvgPool2D, MaxPool2D}` routes today through
`fuel-cuda-kernels::CONV` (PTX module compiled from `conv.cu`'s pool
implementations). These are Fuel's only pool consumers; once
non-adaptive pool is in baracuda, `OpKind::{AvgPool2D, MaxPool2D}`
can move to it and the PTX path retires.

**Severity:** straightforward request. Almost certainly baracuda has
these internally for the adaptive variants' fallback path — the ask
is just to surface them in the FFI.

**Fuel-side commitment if accepted:** wire `OpKind::{AvgPool2D, MaxPool2D}`
to the new baracuda symbols (one commit per dtype family); delete
fuel-cuda-backend's `Pool2D` calls into `kernels::CONV`.

---

## Item 2 — Convolution direction (Conv2D / ConvTranspose2D / im2col / col2im)

**Ask:** confirm the recommended migration path for Fuel's CUDA conv.

The user's earlier overview noted: *"convolution functionality in
Baracuda is handled via other primitives / GEMM/CUDA libraries
(CUTLASS/CuDNN wrappers) and specialized GEMM kernels instead."* That
implies the right path for Fuel is to:

- Use `baracuda-cublas`'s GEMM for the im2col + GEMM lowering, OR
- Use `baracuda-cudnn`'s convolution API directly.

**Question:** which does baracuda recommend? Specifically:

- (a) Does `baracuda-cudnn` expose `cudnnConvolutionForward` /
  `cudnnConvolutionBackwardData` / `cudnnConvolutionBackwardFilter`
  in a stable shape Fuel should call?
- (b) Or do you recommend Fuel write its own im2col + cublas GEMM
  lowering, with baracuda only providing the GEMM?

If (a), Fuel's `OpKind::{Conv2D, ConvTranspose2D}` becomes a cudnn
dispatch; if (b), Fuel writes a small `fuel-conv-cuda` shim that
mirrors what `fuel-cuda-kernels::conv.cu` does today, but built on
baracuda-cublas's GEMM primitives.

(Note: `fuel-conv` already exists as the backend-agnostic conv crate
per `project_fuel_conv_and_vulkan_conv2d.md` — the Vulkan side does
im2col + matmul. CUDA could mirror that pattern, picking baracuda's
GEMM at the matmul step.)

**Fuel-side commitment:** retire `fuel-cuda-kernels::conv.cu` and the
~10 storage.rs call sites that use `kernels::CONV` (conv1d/2d,
conv_transpose1d/2d, im2col, im2col1d, col2im1d, upsample_nearest2d,
upsample_bilinear2d, plus the pool path from Item 1).

---

## Item 3 — Q8_1 staging deprecation timeline (CUDA QMatMul)

**Ask:** confirm that we should migrate Fuel's CUDA QMatMul to
baracuda's MMVQ (FP-activation) path, retiring `quantize_q8_1` and
the `mul_mat_vec_q*_q8_1_cuda` kernels.

The Vulkan QMatMul migration to baracuda alpha.31's MMVQ (commit
`b7360fbc`) is the precedent. CUDA QMatMul still uses the older
Q8_1-staging GGUF approach via fuel-cuda-kernels. The user's overview
confirmed: *"Baracuda intentionally excluded the q8_1 staging MMQ
family ... preferring the FP-activation MMVQ path for GGUF."*

**Question:** is there any reason Fuel's CUDA QMatMul *shouldn't*
follow Vulkan to the MMVQ path? Specifically:

- (a) Are there perf reasons the Q8_1 staging approach is still
  preferable for CUDA at certain shapes (e.g. very large M)?
- (b) Or is MMVQ the all-batch-sizes recommendation on CUDA too?

If (b), Fuel will mirror commit `b7360fbc`'s Vulkan work on the CUDA
side (one focused commit; the wrapper signature is already exercised
on Vulkan).

**Fuel-side commitment:** retire `fuel-cuda-kernels::quantized.cu`'s
Q8_1 path; rewrite `fuel-cuda-backend/src/quantized.rs::quantize_q8_1`
+ the per-format `mul_mat_vec_q*_q8_1_cuda` callers to dispatch
through baracuda's MMVQ.

---

## Item 4 — MoE GEMM direction (the hard one)

**Ask:** decide one of three options for `fuel-nn::moe::moe_gemm` +
`moe_gemm_gguf`'s implementation backend.

Full deep-dive in [`moe-design-analysis.md`](./moe-design-analysis.md).
Short version: Fuel's `FusedMoe` (used by `qwen3_moe` and
`quantized_qwen3_moe`) calls three fused grouped-GEMM kernels per
layer (`moe_gemm_wmma`, `moe_gemm_gguf`, `moe_gemm_gguf_prefill`).
The fused approach gets:

- 3 launches/layer vs. `3 × N_experts` for naive MMVQ × N
  (Mixtral-8x: 8× launch amortization).
- WMMA tensor-core utilization for the non-quant FP path.
- In-kernel token routing via `sorted_token_ids[]` (no separate
  gather/scatter).

**Three options, ordered by our preference:**

1. **(Preferred) Hybrid: batched MMVQ × N-experts** — baracuda adds
   a single-launch MMVQ that takes N weight matrices + a
   `(sorted_token_ids, expert_offsets, topk_weights)` routing
   triple. Sketch FFI in the design memo.
   - **Why preferred:** keeps baracuda's MMVQ-centric architecture
     (no fused-MoE-specific kernel family to maintain); adds the
     routing dimension Fuel needs; gives us launch amortization.
     Loses tensor-core utilization for FP MoE training (acceptable
     — quantized MoE is the dominant inference workload, and FP
     MoE training isn't a baracuda perf focus today).
2. **(Acceptable) Accept Fuel's fused MoE kernels upstream** —
   `moe_gemm_wmma` + `moe_gemm_gguf` + `moe_gemm_gguf_prefill` (+ the
   `moe_utils.cuh` helpers) move to baracuda. Fuel's `moe/` subdir
   deletes; `fuel-nn::moe` rewires to baracuda symbols.
   - **Why acceptable:** clean architectural seam ("baracuda owns
     all CUDA kernels"); zero perf regression. Cost: baracuda
     absorbs a Fuel-specific MoE kernel family — non-trivial
     maintenance surface.
3. **(Fallback) Decline both; Fuel rewrites to MMVQ × N + routing**
   on its side. Multi-week rewrite of `fuel-transformers::fused_moe`
   to gather → N × MMVQ → scatter. Loses ~8× launch amortization
   and tensor cores for FP MoE.
   - **Why fallback:** painful for inference-heavy MoE workloads.
     Defensible if baracuda is firm on not absorbing MoE-specific
     kernels and not extending MMVQ with routing.

**What we'd find most helpful:** a yes/no on option 1 (the batched
MMVQ extension). If yes, we can defer option 3 forever. If no, your
preference between options 2 and 3 settles the matter.

**Fuel-side commitment if option 1 or 2:** retire
`fuel-cuda-kernels/src/moe/` and update `fuel-nn::moe` to call the
new baracuda surface. If option 3, queue the MMVQ × N rewrite as
Fuel-side work with no further baracuda dependency.

---

## Summary table

| Item | Severity | Ask | Fuel commits when accepted |
|---|---|---|---|
| 1. AvgPool/MaxPool | Low | Surface in FFI | 1 commit (wire dispatch) |
| 2. Conv direction | Medium | Confirm cudnn vs. cublas+im2col | 1-3 commits (port Conv2D + ConvTranspose2D + im2col + col2im1d + upsample + pool) |
| 3. Q8_1 deprecation | Medium | Confirm MMVQ-everywhere | 1 commit (mirrors Vulkan b7360fbc) |
| 4. MoE direction | Hard | Pick option 1, 2, or 3 | 1-3 commits depending on choice |

With these resolved, `fuel-cuda-kernels` can fully retire — the
remaining 7-phase plan in the retirement audit becomes executable
end-to-end.

## Pointers for inspection on your side

- `fuel-cuda-kernels/src/moe/` — the three .cu files + moe_utils.cuh
  + gguf.cuh.
- `fuel-cuda-kernels/src/conv.cu` — the conv/pool/upsample
  implementations.
- `fuel-cuda-kernels/src/quantized.cu` — Q8_1 staging path + the
  `mul_mat_vec_q*_q8_1_cuda` family.
- `fuel-nn/src/moe.rs` — current MoE wrappers (the public Fuel API).
- `fuel-transformers/src/fused_moe.rs` — the sole consumer; shows
  the call pattern (3 calls/layer, prefill vs. decode branching).

The 35 now-dead PTX wrapper functions in `fuel-storage/src/dispatch.rs`
(commit `d9898fec`'s leftover) will be cleaned up regardless of your
answers above — they're Fuel-side only and don't need coordination.

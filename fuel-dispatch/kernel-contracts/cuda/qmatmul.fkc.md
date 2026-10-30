---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend — QMatMul (GGML block-quant matmul) kernel contract

The production CUDA **QMatMul** binding (ROADMAP row 105): `out[batch*m, n] = A[batch*m, k] @ dequant(W)[n, k]`,
F32 activations against a GGML block-quantized weight stream, F32 output. This is the PRIMITIVE
`OpKind::QMatMul` (`fuel-ir/src/dispatch.rs`), registered at the single key `(QMatMul, [F32, U32, F32], Cuda)`
— ONE wrapper `KernelRef` (`baracuda_dispatch::qmatmul::qmatmul_f32`) that picks the per-format baracuda
kernel by the `OpParams::QMatMul.quant_type` discriminant and by the row count `batch * m`:

- `batch * m == 1` → baracuda `mmvq_<fmt>` (the decode matrix-vector kernel, fused dequant + dot).
- `batch * m > 1` → baracuda `mmvq_rows_<fmt>` (`fuel-cuda-backend/src/baracuda/gguf.rs`): the batched MMVQ
  launcher driven with one expert and the identity routing, which is exactly `rows x k` times `n x k`.
- Formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (column read chunk 64) and Q2_K, Q3_K, Q4_K (`Q4_K_M`), Q5_K, Q6_K
  (chunk 256). `Q8_1`, or a `k` that is not a multiple of the format's read chunk, has no baracuda kernel:
  the wrapper then computes on the host through the CPU kernel the CPU binding uses (copy down, compute,
  copy up), so this binding is never an error where the CPU path worked.

**As-built binding model — one wrapper `KernelRef` at one key (route-picking is *internal*)**, the same
shape as `vulkan/qmatmul.fkc.md`. The per-format kernels and the host fallback are alternatives *inside*
`qmatmul_f32`, not distinct bindings (a section per internal alternative would register duplicate
`KernelRef`s at one key, which `register_into`'s `finalize` rejects).

**Weight operand dtype — LOGICAL `U32`** (DType-logical / SType-physical split): the key slot is `U32`; the
physical meaning (an opaque packed GGML block byte stream) rides the `fdx.quant` GGML_BLOCK block below.

**Scale single-place rule (§3.9.3).** Every format here is a GGML block quant whose scales are INLINE in the
block stream, so there is no separate FKC scale operand: `fdx.quant.scale_operand` stays `~`.

**Layout model — contiguous-only.** The GGML weight stream has a fixed per-block layout, so strides on the
weight buffer would break the block walk; activations are read contiguous. The planner auto-Contiguizes a
strided operand first. The output is a freshly allocated contiguous row-major `C[batch*m, n]`.

**Cost provenance.** `judge_measured`: the Judge bootstraps it. The derivable GEMM FLOP hint is recorded; no
other coefficient is fabricated.

**Determinism.** The kernels accumulate in f32 over a scheduler-dependent reduction, and the activations are
consumed as f32 by the baracuda kernels (measured on the RTX 4070 against an f64 reference: max error about 1e-7
of the largest output on the device route, 6e-4 to 8e-4 on the host fallback, whose CPU kernel quantizes the
activations to 8 bits), so this is `nondeterministic`, `bit_stable_on_same_hardware: false`, an audited
`none(reason)` precision. The host fallback is the CPU kernel's result. The pre-baked weight
quantization is a model property, not this kernel's contribution. **Wired, and checked on hardware only by the
live tests under `scripts/gpu-run.ps1`** (`fuel-dispatch/tests/baracuda_qmatmul_live.rs`).

---

## qmatmul_f32  (GGML block-quant matmul wrapper; MMVQ on device, host fallback)

```fkc
kernel: qmatmul_f32
op_kind: QMatMul
blurb: "GGML block-quant matmul wrapper: F32 activations @ dequant(block-quant W); baracuda MMVQ per quant_type (gemv for one row, batched MMVQ for many), host fallback for Q8_1 or unaligned k; inline block scales; f32 accumulate; F32 out; contiguous-only"
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::qmatmul_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=3"
      shape_constraint: "dim[-1]=k"
    - name: weight
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "divisible(k, block)"
      fdx:
        requires_ext: true
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q4_0
          role: weight
          scale_operand: ~
  op_params:
    variant: QMatMul
    fields:
      quant_type:   { kind: QuantType, note: "selects the per-format baracuda MMVQ kernel; Q8_1 and a k off the format's read chunk take the host fallback" }
      batch_count:  { kind: usize }
      m:            { kind: usize }
      n:            { kind: usize, note: "output cols; == weight.dim[0]" }
      k:            { kind: usize, constraint: "== activations.dim[-1]; k % block(quant_type) == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(batch_count, m, n)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "batch * m == 1", class: gemm_like, note: "the mmvq decode gemv" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: "batch * m * n * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate over dequantized GGML blocks with f32 activations in baracuda's MMVQ (the host fallback's CPU kernel quantizes them to 8 bits); scheduler-dependent reduction order, NOT bit-stable. The host fallback is the CPU kernel."

determinism: nondeterministic
```

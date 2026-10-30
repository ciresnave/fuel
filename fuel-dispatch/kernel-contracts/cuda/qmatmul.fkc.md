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
  (chunk 256). A `k` that is not a multiple of the format's read chunk (in practice Q4_0, Q4_1, Q5_0, Q5_1
  and Q8_0 with `k % 64 == 32`; the K-quants need `k % 256 == 0` anyway) has no baracuda kernel: the wrapper
  then dequantizes the weights on the host and multiplies with the f32 activations (copy down, compute in
  f64, copy up), which is as accurate as the device route. `Q8_1` also has no baracuda kernel, and no
  dequantization either (GAP-125); it keeps the CPU kernel's result, which is outside the bound below.
  This binding is never an error where the CPU path worked.

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

**Determinism and accuracy.** The kernels accumulate in f32 over a scheduler-dependent reduction and consume
the activations as f32, so this is `nondeterministic`, `bit_stable_on_same_hardware: false`. **Declared
`max_relative: 0.0004` and `max_absolute: 0.0004`**, under the metric every claim in the verification ledger uses
(`abs = |out - ref|`, `rel = |out - ref| / max(|ref|, 1.0)`, per element, inputs uniform in `[-1, 1)`), against the
f64 product of the f32 activations and the dequantized weights. Earned in the ledger by `fuel-dev-tools
seed-cuda-accuracy` (`fkc/verify/accuracy_qmatmul.rs`; without a ledger record the importer downgrades the claim
to UNAUDITED, which is what kept the planner from choosing this kernel): Q4_0, Q8_0 and Q4_K, 11 shapes (rows 1
to 140 including ragged 3, 7, 17, 5; `k` 96 to 8192; `n` 17 to 2048), seeds 1 to 4, both routes of the wrapper, on
the RTX 4070 Laptop. **Measured worst: abs 3.8e-5 (Q8_0 rows=5 n=200 k=8192 seed=2), rel 3.2e-5 (Q4_0 same shape,
seed 3); the error grows with `k`, small-`k` cases are around 1e-7.** The bound has about 10x margin over that
and sits under the planner's default accuracy bound (1e-3). Not audited: `k` above 8192, formats other than the
three, any other GPU or input distribution. `fuel-dispatch/tests/baracuda_qmatmul_audit.rs` is the independent
scale-relative audit (`max|err| / max|ref|`, worst 4.3e-7 on the device route) with the CPU kernel alongside, and
its injected-error control (1% on every block scale) must exceed this bound. The CPU `QMatMul` kernel is
bit-stable but about 3e-3 to 8e-3 off the exact product, because it quantizes the activations to 8 bits as ggml
does; this kernel differs from the CPU kernel by about that much, and the CPU kernel is the less exact of the two.
The pre-baked weight quantization is a model property, not this kernel's contribution.
**Wired, and checked on hardware only by the
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
  max_relative: 0.0004                  # MEASURED (fuel-dev-tools seed-cuda-accuracy): worst rel 3.2e-5, abs 3.8e-5 at k=8192, Q4_0/Q8_0/Q4_K, host-fallback shapes included
  max_absolute: 0.0004
  audited: true
  notes: "rel = |out-ref|/max(|ref|,1), abs = |out-ref|, per element, inputs uniform [-1,1), against the f64 product of f32 activations and dequantized weights; earned by seed-cuda-accuracy over Q4_0/Q8_0/Q4_K, k to 8192, both routes; Q8_1 is excluded and keeps the CPU kernel (about 7e-3). f32 accumulate over dequantized GGML blocks with f32 activations in baracuda's MMVQ and the host fallback; scheduler-dependent reduction order, NOT bit-stable. The host fallback dequantizes on the host and multiplies in f64."

determinism: nondeterministic
```

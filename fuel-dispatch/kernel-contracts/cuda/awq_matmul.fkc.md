---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - AWQ W4A16 matmul (AwqMatmul; 4-input key) kernel contract

CUDA (baracuda `baracuda_kernels_int4_awq_gemm_f16_run`, the vendored mit-han-lab llm-awq kernel)
weight-only int4 GEMM. One section, one key: `[F16, U32, U32, F16, F16]` (inputs then output, sec 3.2) =
`activations`, `weight`, `zeros`, `scales`, `out`. Per-dtype (single-dtype `entry_point` resolved AS-IS).
Activations and output are F16 only: the kernel is fp16-only, so BF16/F32 activations are served by the
op's `decompose` (dequantize-then-matmul), not by this binding.

---

## awq_matmul_f16  (AwqMatmul - F16 activations, U32 packed int4 weight/zeros, F16 scales; contiguous-only)

AWQ asymmetric int4 weight-only GEMM, `dequant(q) = (q - zero) * scale` per output row and group, f32
accumulation, split-K partial sums reduced in a fixed order. Backs `OpKind::AwqMatmul`
(`FusedOps::AWQ_MATMUL`, `OpParams::AwqMatmul`). Output: fresh, contiguous, no aliasing.

The three quantized operands are in **baracuda's kernel layout** (OC-major, `N = OC`, `K = IC`), which is
what `fuel_quantized::awq_repack::repack_hf_awq_for_kernel` produces from a HuggingFace `*-AWQ` checkpoint:
`weight [N, K/8]` U32 (nibble `k % 8` of word `k / 8`), `zeros [N, zeros_w]` U32 (group `g` is nibble
`g % 8` of word `g / 8`), `scales [N, 8 * zeros_w]` F16 (entry `g`), `zeros_w = ceil(ceil(K / group_size / 8)
/ mult) * mult`, `mult = 128 / group_size`. The U32 operands are packed int4 codes, not integers: this
contract does not declare an FDX quant descriptor (the packing is the op's own layout, not a registered FDX
family), so nothing in the importer can tell a mis-packed buffer from a packed one. The only defence is the
build-time shape check in `NodeHandle::awq_matmul` and the repack module.

Kernel constraints (baracuda `can_implement`; a violation is a hard `Err`, never a launch): `group_size` in
{64, 128}; `N % 64 == 0`; `K % group_size == 0`; `K % (32 * split_k_iters) == 0`; sm_80 or newer.

```fkc
kernel: awq_matmul_f16
op_kind: AwqMatmul
blurb: "AWQ W4A16 asymmetric int4 weight-only GEMM (CUDA/baracuda) F16 activations; kernel-layout operands; contiguous-only; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::awq_matmul_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: [..., m, k]; leading dims flatten into batch"
    - name: weight
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "notes: [n, k/8]; eight int4 codes per word, sequential nibble order"
    - name: zeros
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "notes: [n, zeros_w]; zeros_w padded per awq_zeros_w(k, group_size)"
    - name: scales
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "notes: [n, 8 * zeros_w]; entry g is group g's scale"
  op_params:
    variant: AwqMatmul            # OpParams::AwqMatmul (primitive namespace; sec 3.7)
    fields:
      batch:         { kind: usize, note: "product of the activations' leading dims (>= 1)" }
      m:             { kind: usize, note: "activation rows per batch" }
      n:             { kind: usize, constraint: "% 64 == 0", note: "output features (OC)" }
      k:             { kind: usize, constraint: "% group_size == 0 and % (32 * split_k_iters) == 0", note: "input features (IC)" }
      group_size:    { kind: usize, constraint: "in {64, 128}" }
      split_k_iters: { kind: i32, constraint: "> 0", note: "launch tuning, typically 8; sizes the workspace" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(activations)
      shape_rule: from_params(batch, m, n)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "batch * m * k * 2 + n * k / 2 + batch * m * n * 2"
  memory: { device_bytes: "batch * m * n * 2", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "author-declared UNAUDITED seed: f32 accumulation inside the kernel, split-K partials written to a staging buffer and reduced in a fixed order (no atomics), so bit-stable on the same hardware; no ledger entry yet. Reduced precision: partials are staged as F16 before the reduce."

determinism: same_hardware_bitwise
```

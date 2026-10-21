---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - last-dim normalizations (RmsNorm / LayerNorm) kernel contracts

CUDA (baracuda) Stride-driven norm FFI; wrapper passes the input's true rank-N shape + strides. eps rides OpParams::NormLastDim.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is the author-declared `audited: false` -> UNAUDITED seed.


---

## rms  (RmsNormLastDim - {F32, F16, BF16, F64}, strided+broadcast)

RMSNorm over the last dim Backs `OpKind::RmsNormLastDim`. Stride-driven norm FFI; wrapper passes the input's true rank-N shape + strides. eps rides OpParams::NormLastDim. Output: fresh, contiguous, no aliasing.

```fkc
kernel: rms
op_kind: RmsNormLastDim
blurb: "RMSNorm over the last dim (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::rms"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: NormLastDim }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "2 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "baracuda_norm.cuh launch_rms_norm_fp dispatches between rms_norm_fp_kernel (legacy: one thread per output cell, serial per-thread sum-of-squares over the row in fixed j=0..norm_total_extent order, no cross-thread state) and rms_norm_smem_kernel (SMEM fast path: block_reduce_sum_f32/f64, a fixed warp-shuffle butterfly + cross-warp SMEM tree reduction, no atomics). Eligibility (contig last axis, contig outer stride, SMEM budget) is a pure function of rank/shape/stride/dtype computed host-side before launch, not a runtime occupancy heuristic — a fixed input shape always selects the same path and launch config (kBlock=256 is a compile-time constant in both). Bit-identical for bit-identical inputs on the same hardware. (LayerNormLastDim, this file's other section, is a separate kernel/audit — not covered by this claim.)"

determinism: same_hardware_bitwise
```

---

## layer  (LayerNormLastDim - {F32, F16, BF16, F64}, strided+broadcast)

LayerNorm over the last dim Backs `OpKind::LayerNormLastDim`. Stride-driven norm FFI; wrapper passes the input's true rank-N shape + strides. eps rides OpParams::NormLastDim. Output: fresh, contiguous, no aliasing.

```fkc
kernel: layer
op_kind: LayerNormLastDim
blurb: "LayerNorm over the last dim (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::layer"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: NormLastDim }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "2 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "LayerNorm over the last dim; author-declared UNAUDITED seed (byte-for-byte the deleted plain register default); pointwise, bit-stable same hardware."

determinism: same_hardware_bitwise
```

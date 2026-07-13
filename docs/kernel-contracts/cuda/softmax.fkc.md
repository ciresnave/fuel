---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - last-dim softmax / log-softmax kernel contracts

CUDA (baracuda) Stride-driven softmax FFI; wrapper passes the input's true rank-N shape + strides.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is the author-declared `audited: false` -> UNAUDITED seed.


---

## softmax  (SoftmaxLastDim - {F32, F16, BF16, F64}, strided+broadcast)

softmax over the last dim Backs `OpKind::SoftmaxLastDim`. Stride-driven softmax FFI; wrapper passes the input's true rank-N shape + strides. Output: fresh, contiguous, no aliasing.

```fkc
kernel: softmax
op_kind: SoftmaxLastDim
blurb: "softmax over the last dim (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::softmax"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: SoftmaxLastDim }

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
  notes: "baracuda_softmax.cuh launch_softmax_fp dispatches between softmax_fp_kernel (legacy: one thread per output cell, two sequential fixed-order passes — row max then sum-of-exp, no cross-thread state) and softmax_smem_kernel (SMEM fast path: block_reduce_max_f32/block_reduce_sum_f32, a fixed warp-shuffle butterfly + cross-warp SMEM tree reduction, no atomics). Eligibility between the two paths (contig last axis, contig outer stride, SMEM budget) is a pure function of rank/shape/stride/dtype, not a runtime occupancy heuristic, so a fixed input shape always selects the same path. Neither path contains atomicAdd/atomicMax/atomicCAS (grepped the whole file: zero hits). Bit-identical for bit-identical inputs on the same hardware."

determinism: same_hardware_bitwise
```

---

## log_softmax  (LogSoftmaxLastDim - {F32, F16, BF16, F64}, strided+broadcast)

log-softmax over the last dim Backs `OpKind::LogSoftmaxLastDim`. Stride-driven softmax FFI; wrapper passes the input's true rank-N shape + strides. Output: fresh, contiguous, no aliasing.

```fkc
kernel: log_softmax
op_kind: LogSoftmaxLastDim
blurb: "log-softmax over the last dim (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::log_softmax"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: LogSoftmaxLastDim }

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
  notes: "Same structure and eligibility-dispatch pattern as softmax (baracuda_softmax.cuh launch_log_softmax_fp): log_softmax_fp_kernel (legacy, per-thread two-pass max+sum, no cross-thread state) vs log_softmax_smem_kernel (block_reduce_max_f32/block_reduce_sum_f32, fixed warp-shuffle + cross-warp SMEM reduction, no atomics), path choice a pure function of shape/stride/dtype. Zero atomics anywhere in the file. Bit-identical for bit-identical inputs on the same hardware."

determinism: same_hardware_bitwise
```

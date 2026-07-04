---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - broadcast-reverse reductions (sum_to / max_to) kernel contracts

CUDA (baracuda) Gradient-shaped reductions to a target broadcastable shape; stride-driven on the input side.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is the author-declared `audited: false` -> UNAUDITED seed.


---

## sum_to  (ReduceSumTo - {F32, F16, BF16, F64}, strided+broadcast)

sum-reduce to the target shape Backs `OpKind::ReduceSumTo`. Gradient-shaped reductions to a target broadcastable shape; stride-driven on the input side. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sum_to
op_kind: ReduceSumTo
blurb: "sum-reduce to the target shape (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sum_to"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduces_to=out"
  op_params: { variant: ReduceSumTo }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: reduced(in)
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
  notes: "sum-reduce to the target shape; author-declared UNAUDITED seed (byte-for-byte the deleted plain register default); pointwise, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## max_to  (ReduceMaxTo - {F32, F16, BF16, F64}, strided+broadcast)

max-reduce to the target shape Backs `OpKind::ReduceMaxTo`. Gradient-shaped reductions to a target broadcastable shape; stride-driven on the input side. Output: fresh, contiguous, no aliasing.

```fkc
kernel: max_to
op_kind: ReduceMaxTo
blurb: "max-reduce to the target shape (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::max_to"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduces_to=out"
  op_params: { variant: ReduceMaxTo }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: reduced(in)
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
  notes: "max-reduce to the target shape; author-declared UNAUDITED seed (byte-for-byte the deleted plain register default); pointwise, bit-stable same hardware."

determinism: same_hardware_bitwise
```

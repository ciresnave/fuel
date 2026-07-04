---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - rotary position embedding (RoPE) kernel contracts

CUDA (baracuda) head_dim (innermost) stride==1 on the strided path; (outer_count, seq, head_dim) from OpParams::Rope.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is the author-declared `audited: false` -> UNAUDITED seed.


---

## rope  (Rope - {F32, F16, BF16, F64}, strided+broadcast)

apply RoPE rotation Backs `OpKind::Rope`. head_dim (innermost) stride==1 on the strided path; (outer_count, seq, head_dim) from OpParams::Rope. Output: fresh, contiguous, no aliasing.

```fkc
kernel: rope
op_kind: Rope
blurb: "apply RoPE rotation (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::rope"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: Rope }

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
  notes: "apply RoPE rotation; author-declared UNAUDITED seed (byte-for-byte the deleted plain register default); pointwise, bit-stable same hardware."

determinism: same_hardware_bitwise
```

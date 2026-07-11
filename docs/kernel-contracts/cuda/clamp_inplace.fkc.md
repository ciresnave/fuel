---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - in-place scalar-bounds clamp kernel contracts

CUDA (baracuda) same-pointer ternary clamp (a==y) with (min,max) from OpParams::Clamp; contiguous target.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is the author-declared `audited: false` -> UNAUDITED seed.


---

## clamp_inplace  (ClampInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

out[i]=clamp(x[i],min,max) in place Backs `OpKind::ClampInplace`. same-pointer ternary clamp (a==y) with (min,max) from OpParams::Clamp; contiguous target. Output: fresh, contiguous, no aliasing.

```fkc
kernel: clamp_inplace
op_kind: ClampInplace
blurb: "out[i]=clamp(x[i],min,max) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::clamp_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: Clamp }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: in_place(x)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: true
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
  notes: "fuel-cuda-backend/src/baracuda/clamp.rs clamp_inplace_run reuses the same ternary_pointwise_strided_kernel (baracuda_elementwise.cuh) with rank-1 contig stride and the same pointer for input `a` and output `y`; each thread reads its own a[i] then writes y[i] at the identical address, no cross-thread aliasing, no atomics. Deterministic same-hardware repeat calls."

determinism: same_hardware_bitwise
```

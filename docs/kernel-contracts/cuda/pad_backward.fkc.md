---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - padding backward (PadBackward; Constant) kernel contracts

CUDA (baracuda) Constant-mode backward (slice-out of the padded region); OpParams::PadBackward; contiguous.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is the author-declared `audited: false` -> UNAUDITED seed.


---

## pad_backward  (PadBackward - {F32, F16, BF16, F64}, contiguous-only)

grad_in=slice(grad_out) of the pad region Backs `OpKind::PadBackward`. Constant-mode backward (slice-out of the padded region); OpParams::PadBackward; contiguous. Output: fresh, contiguous, no aliasing.

```fkc
kernel: pad_backward
op_kind: PadBackward
blurb: "grad_in=slice(grad_out) of the pad region (CUDA/baracuda) {F32, F16, BF16, F64}; contiguous-only; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::pad_backward"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: PadBackward }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
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
  bytes_moved: "2 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "baracuda-kernels-sys/kernels/include/baracuda_elementwise.cuh pad_constant_backward_kernel: one grid-stride thread per dx cell, single index computation + single read from dy + single write to dx (pure slice, no arithmetic, no atomics/shared memory); deterministic given identical inputs and launch config. Only Constant-mode backward is wired (Reflect/Replicate/Circular backward is unimplemented, not covered by this contract)."

determinism: same_hardware_bitwise
```

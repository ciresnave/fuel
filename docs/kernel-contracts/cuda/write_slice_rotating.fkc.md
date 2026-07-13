---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - sliding-window in-place slab assign (WriteSliceRotating; byte-width umbrella) kernel contracts

CUDA (baracuda) same byte-width surface as WriteSlice; wrapper handles position D2H + ring-boundary split; OpParams::WriteSliceRotating { dest_shape, axis, modulus, ranges }.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is the author-declared `audited: false` -> UNAUDITED seed.


---

## write_slice_rotating  (WriteSliceRotating - {F32, F64, F16, BF16, I32, I64, U32, U8, I8}, contiguous-only, in-place)

out=dest with the src slab written at ranges mod modulus (in place) Backs `OpKind::WriteSliceRotating`. same byte-width surface as WriteSlice; wrapper handles position D2H + ring-boundary split; OpParams::WriteSliceRotating { dest_shape, axis, modulus, ranges }. Output: fresh, contiguous, no aliasing.

```fkc
kernel: write_slice_rotating
op_kind: WriteSliceRotating
blurb: "out=dest with the src slab written at ranges mod modulus (in place) (CUDA/baracuda) {F32, F64, F16, BF16, I32, I64, U32, U8, I8}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::write_slice_rotating"
kernel_revision_hash: auto

accept:
  inputs:
    - name: dest
      dtypes: [F32, F64, F16, BF16, I32, I64, U32, U8, I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: WriteSliceRotating }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(dest)
      shape_rule: same_as(dest)
      layout_guarantee: contiguous
      aliasing: in_place(dest)

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
  notes: "No dedicated .cu kernel: fuel-dispatch's wrapper (baracuda_dispatch.rs cuda_write_slice_rotating_baracuda_wrapper) D2H-reads the position scalar, computes wrapped_start/first_len/second_len with deterministic host integer math, extracts each chunk via CudaStorageBytes::extract_strided_to_new (sequential cudaMemcpyDtoD per tile on one stream + explicit synchronize, no atomics), then reuses write_slice's own audited write_slice_byte_kernel for each of the (at most two) disjoint dest ranges. Every stage is a pure function of its inputs with no cross-thread reduction; the two chunk writes target non-overlapping dest regions by construction (first_len = min(slab_axis_len, modulus - wrapped_start)) and run in-order on the same stream, so there is no race even though they don't overlap."

determinism: same_hardware_bitwise
```

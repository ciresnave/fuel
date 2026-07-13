---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - in-place rectangular slab assign (WriteSlice; byte-width umbrella) kernel contracts

CUDA (baracuda) 9 dtypes fan to b1/b2/b4/b8 by element byte-width; OpParams::WriteSlice { dest_shape, ranges }; contiguous dest.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is the author-declared `audited: false` -> UNAUDITED seed.


---

## write_slice  (WriteSlice - {F32, F64, F16, BF16, I32, I64, U32, U8, I8}, contiguous-only, in-place)

out=dest with the src slab written at ranges (in place) Backs `OpKind::WriteSlice`. 9 dtypes fan to b1/b2/b4/b8 by element byte-width; OpParams::WriteSlice { dest_shape, ranges }; contiguous dest. Output: fresh, contiguous, no aliasing.

```fkc
kernel: write_slice
op_kind: WriteSlice
blurb: "out=dest with the src slab written at ranges (in place) (CUDA/baracuda) {F32, F64, F16, BF16, I32, I64, U32, U8, I8}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::write_slice"
kernel_revision_hash: auto

accept:
  inputs:
    - name: dest
      dtypes: [F32, F64, F16, BF16, I32, I64, U32, U8, I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: WriteSlice }

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
  notes: "Reasoned from source (baracuda_write_slice.cuh write_slice_byte_kernel, baracuda-kernels-sys/kernels/shape_layout/write_slice.cu): one thread per source element, no atomics, no shared memory, no cross-thread reduction; each thread computes its own dest_off via a pure coordinate-shift (bijective source->dest mapping, no write collisions) and does a single memcpy-style store. Pure byte copy, no floating-point arithmetic at all, so bit-identical output for bit-identical inputs on repeat calls follows directly from the launch config being a deterministic function of source_numel."

determinism: same_hardware_bitwise
```

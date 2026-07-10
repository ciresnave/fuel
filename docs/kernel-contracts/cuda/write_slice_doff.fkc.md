---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:doff-alpha77"
---

# fuel-cuda-backend - device-resident-offset in-place slab assign (WriteSliceDoff; byte-width umbrella) kernel contracts

CUDA (baracuda) form-B WriteSlice: the start of ONE axis is read from a DEVICE pointer (`dyn_start_dev`,
a single `i64`) at kernel entry instead of being host-baked into `range_start`. The wrapper threads the
`offset` operand's device pointer straight through (NO D2H — that would break CUDA-graph capture);
`OpParams::WriteSliceDoff { dest_shape, axis, ranges }`. b1/b2/b4/b8 only (no b16 — the KV-decode dtype
set). Each section binds one concrete `OpKind` and fans its operand(s) over the accepted dtypes (sec 3.4
dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through [`crate::fkc::CudaLinkRegistry`]). Caps
ride through the import truthfully (sec 6 / caps_map). Cost is `judge_measured`; precision is the
author-declared `audited: false` -> UNAUDITED seed.


---

## write_slice_doff  (WriteSliceDoff - {F32, F64, F16, BF16, I32, I64, U32, U8, I8}, contiguous-only, in-place)

out=dest with the src slab written at ranges, the `axis` start read device-side from an i64 offset (in place). Backs `OpKind::WriteSliceDoff`. Wrapper threads the offset device pointer to baracuda's `_doff` launcher (no D2H, no wrap); OpParams::WriteSliceDoff { dest_shape, axis, ranges }. Output: in-place on dest, contiguous.

```fkc
kernel: write_slice_doff
op_kind: WriteSliceDoff
blurb: "out=dest with the src slab written at ranges, axis start read device-side from an i64 offset (in place) (CUDA/baracuda) {F32, F64, F16, BF16, I32, I64, U32, U8, I8}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::write_slice_doff"
kernel_revision_hash: auto

accept:
  inputs:
    - name: dest
      dtypes: [F32, F64, F16, BF16, I32, I64, U32, U8, I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: WriteSliceDoff }

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
  audited: false
  notes: "out=dest with the src slab written at ranges, axis start read device-side from an i64 offset (in place); author-declared UNAUDITED seed; pointwise byte copy, bit-stable same hardware. Bounds on the device-resident start are the caller's contract (the kernel does not clamp)."

determinism: same_hardware_bitwise
```

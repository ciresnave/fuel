---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - int8 matmul facade (gemm_int s8/u8 RRR at OpKind::MatMul) kernel contracts

CUDA (baracuda) S8/U8 RRR identity (W8A8 phase 1); i8->gemm_s8_rrr, u8->gemm_u8_rrr; contiguous; OpParams::Matmul.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is the author-declared `audited: false` -> UNAUDITED seed.


---

## gemm  (MatMul - {I8, U8}, contiguous-only)

out=lhs@rhs (int8 RRR identity) Backs `OpKind::MatMul`. S8/U8 RRR identity (W8A8 phase 1); i8->gemm_s8_rrr, u8->gemm_u8_rrr; contiguous; OpParams::Matmul. Output: fresh, contiguous, no aliasing.

```fkc
kernel: gemm
op_kind: MatMul
blurb: "out=lhs@rhs (int8 RRR identity) (CUDA/baracuda) {I8, U8}; contiguous-only; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::gemm"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [I8, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
    - name: rhs
      dtypes: [I8, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: Matmul }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: same_as(lhs)
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
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "out=lhs@rhs (int8 RRR identity); author-declared UNAUDITED seed (byte-for-byte the deleted plain register default); pointwise, bit-stable same hardware."

determinism: same_hardware_bitwise
```

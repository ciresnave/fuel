---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - causal depthwise conv1d (CausalConv1d; 4-input key) kernel contracts

CUDA (baracuda) Fuel-prepad bridge (strips kernel-1 leading pad); (x, weight, bias) -> out; OpParams::CausalConv1d.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is the author-declared `audited: false` -> UNAUDITED seed.


---

## causal_conv1d  (CausalConv1d - {F32, F64, BF16, F16}, contiguous-only)

causal depthwise conv1d (+ optional silu) Backs `OpKind::CausalConv1d`. Fuel-prepad bridge (strips kernel-1 leading pad); (x, weight, bias) -> out; OpParams::CausalConv1d. Output: fresh, contiguous, no aliasing.

```fkc
kernel: causal_conv1d
op_kind: CausalConv1d
blurb: "causal depthwise conv1d (+ optional silu) (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::causal_conv1d"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
    - name: weight
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
    - name: bias
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: CausalConv1d }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(out_seq)   # C-3 hygiene: NOT same_as(x). out_seq = x.dim[-1] - (kernel-1) shrinks the seq axis for kernel>1 (Mamba K=4 ⇒ -3); same_as(x) only coincides at kernel==1. Non-evaluable ⇒ the shape-oracle skips it (no false-green if CausalConv1d ever joins synth).
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
  bytes_moved: "4 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "causal depthwise conv1d (+ optional silu); author-declared UNAUDITED seed (byte-for-byte the deleted plain register default); pointwise, bit-stable same hardware."

determinism: same_hardware_bitwise
```

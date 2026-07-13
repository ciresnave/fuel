---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - elementwise binary (add / sub / mul / div / maximum / minimum / pow / rem) kernel contracts

CUDA (baracuda) Per-op elementwise binary `out=f(lhs,rhs)`; baracuda ships contig+strided FFI, the wrapper picks per-call.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is `audited: true` for all 8 kernels (2026-07-11 precision
audit — each op is a pure single-thread-per-element functor over the shared
`binary_pointwise_{contig,strided}_kernel` in `include/baracuda_elementwise.cuh`, no atomics or
cross-thread reduction; see each section's `precision.notes` for the source citation).


---

## add  (AddElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=lhs[i]+rhs[i] Backs `OpKind::AddElementwise`. Per-op elementwise binary `out=f(lhs,rhs)`; baracuda ships contig+strided FFI, the wrapper picks per-call. Output: fresh, contiguous, no aliasing.

```fkc
kernel: add
op_kind: AddElementwise
blurb: "out[i]=lhs[i]+rhs[i] (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::add"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: same_as(lhs)
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
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Audited against baracuda-kernels-sys/kernels/elementwise/binary_add_fp.cu (AddFunctor = `a + b`) over the shared binary_pointwise_{contig,strided}_kernel in include/baracuda_elementwise.cuh: one thread per output element, disjoint writes, no atomics or cross-thread reduction — deterministic given fixed inputs + grid config."

determinism: same_hardware_bitwise
```

---

## sub  (SubElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=lhs[i]-rhs[i] Backs `OpKind::SubElementwise`. Per-op elementwise binary `out=f(lhs,rhs)`; baracuda ships contig+strided FFI, the wrapper picks per-call. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sub
op_kind: SubElementwise
blurb: "out[i]=lhs[i]-rhs[i] (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sub"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: same_as(lhs)
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
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Audited against baracuda-kernels-sys/kernels/elementwise/binary_sub_fp.cu (SubFunctor = `a - b`) over the shared binary_pointwise_{contig,strided}_kernel in include/baracuda_elementwise.cuh: one thread per output element, disjoint writes, no atomics or cross-thread reduction — deterministic given fixed inputs + grid config."

determinism: same_hardware_bitwise
```

---

## mul  (MulElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=lhs[i]*rhs[i] Backs `OpKind::MulElementwise`. Per-op elementwise binary `out=f(lhs,rhs)`; baracuda ships contig+strided FFI, the wrapper picks per-call. Output: fresh, contiguous, no aliasing.

```fkc
kernel: mul
op_kind: MulElementwise
blurb: "out[i]=lhs[i]*rhs[i] (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::mul"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: same_as(lhs)
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
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Audited against baracuda-kernels-sys/kernels/elementwise/binary_mul_fp.cu (MulFunctor = `a * b`) over the shared binary_pointwise_{contig,strided}_kernel in include/baracuda_elementwise.cuh: one thread per output element, disjoint writes, no atomics or cross-thread reduction — deterministic given fixed inputs + grid config. Same seam that previously blocked task 4b-delta (fuel-dispatch placement refusing this kernel unaudited)."

determinism: same_hardware_bitwise
```

---

## div  (DivElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=lhs[i]/rhs[i] Backs `OpKind::DivElementwise`. Per-op elementwise binary `out=f(lhs,rhs)`; baracuda ships contig+strided FFI, the wrapper picks per-call. Output: fresh, contiguous, no aliasing.

```fkc
kernel: div
op_kind: DivElementwise
blurb: "out[i]=lhs[i]/rhs[i] (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::div"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: same_as(lhs)
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
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Audited against baracuda-kernels-sys/kernels/elementwise/binary_div_fp.cu (DivFunctor = `a / b`) over the shared binary_pointwise_{contig,strided}_kernel in include/baracuda_elementwise.cuh: one thread per output element, disjoint writes, no atomics or cross-thread reduction — deterministic given fixed inputs + grid config."

determinism: same_hardware_bitwise
```

---

## maximum  (MaximumElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=max(lhs[i],rhs[i]) Backs `OpKind::MaximumElementwise`. Per-op elementwise binary `out=f(lhs,rhs)`; baracuda ships contig+strided FFI, the wrapper picks per-call. Output: fresh, contiguous, no aliasing.

```fkc
kernel: maximum
op_kind: MaximumElementwise
blurb: "out[i]=max(lhs[i],rhs[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::maximum"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: same_as(lhs)
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
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Audited against baracuda-kernels-sys/kernels/elementwise/binary_maximum_fp.cu (explicit NaN-guard compare-and-select, `a!=a`/`b!=b` checks before `a>b`) over the shared binary_pointwise_{contig,strided}_kernel: one thread per output element, disjoint writes, no atomics or cross-thread reduction — deterministic given fixed inputs + grid config."

determinism: same_hardware_bitwise
```

---

## minimum  (MinimumElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=min(lhs[i],rhs[i]) Backs `OpKind::MinimumElementwise`. Per-op elementwise binary `out=f(lhs,rhs)`; baracuda ships contig+strided FFI, the wrapper picks per-call. Output: fresh, contiguous, no aliasing.

```fkc
kernel: minimum
op_kind: MinimumElementwise
blurb: "out[i]=min(lhs[i],rhs[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::minimum"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: same_as(lhs)
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
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Audited against baracuda-kernels-sys/kernels/elementwise/binary_minimum_fp.cu (explicit NaN-guard compare-and-select, `a!=a`/`b!=b` checks before `a<b`) over the shared binary_pointwise_{contig,strided}_kernel: one thread per output element, disjoint writes, no atomics or cross-thread reduction — deterministic given fixed inputs + grid config."

determinism: same_hardware_bitwise
```

---

## pow  (PowElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=lhs[i]^rhs[i] (tensor^tensor) Backs `OpKind::PowElementwise`. Per-op elementwise binary `out=f(lhs,rhs)`; baracuda ships contig+strided FFI, the wrapper picks per-call. Output: fresh, contiguous, no aliasing.

```fkc
kernel: pow
op_kind: PowElementwise
blurb: "out[i]=lhs[i]^rhs[i] (tensor^tensor) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::pow"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: same_as(lhs)
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
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Audited against baracuda-kernels-sys/kernels/elementwise/binary_pow_fp.cu (`powf`/`pow` libdevice call per element, f16/bf16 via f32-detour) over the shared binary_pointwise_{contig,strided}_kernel: one thread per output element, disjoint writes, no atomics or cross-thread reduction; CUDA libdevice math is deterministic for fixed inputs on fixed hardware/toolchain."

determinism: same_hardware_bitwise
```

---

## rem  (RemElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=rem(lhs[i],rhs[i]) (PyTorch sign-of-divisor; baracuda binary_mod) Backs `OpKind::RemElementwise`. Per-op elementwise binary `out=f(lhs,rhs)`; baracuda ships contig+strided FFI, the wrapper picks per-call. Output: fresh, contiguous, no aliasing.

```fkc
kernel: rem
op_kind: RemElementwise
blurb: "out[i]=rem(lhs[i],rhs[i]) (PyTorch sign-of-divisor; baracuda binary_mod) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::rem"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: same_as(lhs)
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
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Audited against baracuda-kernels-sys/kernels/elementwise/binary_mod_fp.cu (fmodf/fmod + sign-fix add for Python-style modulo) over the shared binary_pointwise_{contig,strided}_kernel: one thread per output element, disjoint writes, no atomics or cross-thread reduction; deterministic given fixed inputs + grid config."

determinism: same_hardware_bitwise
```

---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - in-place unary activations + the 16-op unary expansion kernel contracts

CUDA (baracuda) baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is audited (2026-07-11 precision-audit program, inplace-unary
family): all 21 kernels reasoned from their baracuda `.cu` source (the shared
`unary_pointwise_contig_kernel` grid-stride template — one thread per output element, pure
per-element math, no atomics/shared-memory/cross-thread state) — `bit_stable_on_same_hardware: true`,
`audited: true`. `max_ulp`/`max_relative`/`max_absolute` remain unclaimed (`~`) pending numerical-
accuracy evidence, which is a separate question from bit-stability.


---

## relu_inplace  (ReluInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=relu(x[i]) in place Backs `OpKind::ReluInplace`. baracuda `unary_relu_propagating_*_run` same-pointer dispatch (a==y); contiguous target, no params. **NaN-PROPAGATING** (torch parity, rebound 2026-07-08 to match the forward `ReluElementwise` kernels + CPU; pinned by `cuda_relu_inplace_propagates_nan_f32`). Output: fresh, contiguous, no aliasing.

```fkc
kernel: relu_inplace
op_kind: ReluInplace
blurb: "x[i]=relu(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::relu_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_relu_propagating_fp.cu ReluPropagatingFunctor via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone (branch-on-value, no atomics/shared-mem/cross-thread state); same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## silu_inplace  (SiluInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=silu(x[i]) in place Backs `OpKind::SiluInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: silu_inplace
op_kind: SiluInplace
blurb: "x[i]=silu(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::silu_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_silu_fp.cu SiluFunctor via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone (branch-on-value only, no atomics/shared-mem/cross-thread state); same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## gelu_inplace  (GeluInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=gelu(x[i]) in place Backs `OpKind::GeluInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: gelu_inplace
op_kind: GeluInplace
blurb: "x[i]=gelu(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::gelu_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_gelu_tanh_fp.cu GeluTanhFunctor (Fuel's GeluInplace binds the tanh-approx stem, not erf) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## tanh_inplace  (TanhInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=tanh(x[i]) in place Backs `OpKind::TanhInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: tanh_inplace
op_kind: TanhInplace
blurb: "x[i]=tanh(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::tanh_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_tanh_fp.cu TanhFunctor (plain tanhf/tanh intrinsic call) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## sigmoid_inplace  (SigmoidInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=sigmoid(x[i]) in place Backs `OpKind::SigmoidInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sigmoid_inplace
op_kind: SigmoidInplace
blurb: "x[i]=sigmoid(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sigmoid_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_sigmoid_fp.cu SigmoidFunctor via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone (branch-on-value only, no atomics/shared-mem/cross-thread state); same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## neg_inplace  (NegInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=neg(x[i]) in place Backs `OpKind::NegInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: neg_inplace
op_kind: NegInplace
blurb: "x[i]=neg(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::neg_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_neg_fp.cu NegFunctor (`return -x`, generic template covers all 4 dtypes) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## abs_inplace  (AbsInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=abs(x[i]) in place Backs `OpKind::AbsInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: abs_inplace
op_kind: AbsInplace
blurb: "x[i]=abs(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::abs_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_abs_fp.cu AbsFunctor (fabsf/fabs/__habs intrinsics) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## sqr_inplace  (SqrInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=sqr(x[i]) in place Backs `OpKind::SqrInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sqr_inplace
op_kind: SqrInplace
blurb: "x[i]=sqr(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sqr_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_square_fp.cu SquareFunctor (`return x * x`, generic template covers all 4 dtypes) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run, bound via unary_square_* stems). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## sqrt_inplace  (SqrtInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=sqrt(x[i]) in place Backs `OpKind::SqrtInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sqrt_inplace
op_kind: SqrtInplace
blurb: "x[i]=sqrt(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sqrt_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_sqrt_fp.cu SqrtFunctor (sqrtf/sqrt intrinsic call) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## rsqrt_inplace  (RsqrtInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=rsqrt(x[i]) in place Backs `OpKind::RsqrtInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: rsqrt_inplace
op_kind: RsqrtInplace
blurb: "x[i]=rsqrt(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::rsqrt_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_rsqrt_fp.cu RsqrtFunctor (rsqrtf intrinsic / 1.0/sqrt(x) for f64) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## recip_inplace  (RecipInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=recip(x[i]) in place Backs `OpKind::RecipInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: recip_inplace
op_kind: RecipInplace
blurb: "x[i]=recip(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::recip_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_reciprocal_fp.cu ReciprocalFunctor (`1/x` per-dtype) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run, bound via unary_reciprocal_* stems). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## exp_inplace  (ExpInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=exp(x[i]) in place Backs `OpKind::ExpInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: exp_inplace
op_kind: ExpInplace
blurb: "x[i]=exp(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::exp_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_exp_fp.cu ExpFunctor (expf/exp intrinsic call) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## log_inplace  (LogInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=log(x[i]) in place Backs `OpKind::LogInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: log_inplace
op_kind: LogInplace
blurb: "x[i]=log(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::log_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_log_fp.cu LogFunctor (logf/log intrinsic call) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## sin_inplace  (SinInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=sin(x[i]) in place Backs `OpKind::SinInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sin_inplace
op_kind: SinInplace
blurb: "x[i]=sin(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sin_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_sin_fp.cu SinFunctor (sinf/sin intrinsic call) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## cos_inplace  (CosInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=cos(x[i]) in place Backs `OpKind::CosInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: cos_inplace
op_kind: CosInplace
blurb: "x[i]=cos(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::cos_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_cos_fp.cu CosFunctor (cosf/cos intrinsic call) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## sign_inplace  (SignInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=sign(x[i]) in place Backs `OpKind::SignInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sign_inplace
op_kind: SignInplace
blurb: "x[i]=sign(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sign_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_sign_fp.cu SignFunctor (`(x>0)?1:(x<0)?-1:0` per-dtype, __hgt/__hlt for f16/bf16) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone (branch-on-value only), no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## floor_inplace  (FloorInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=floor(x[i]) in place Backs `OpKind::FloorInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: floor_inplace
op_kind: FloorInplace
blurb: "x[i]=floor(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::floor_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_floor_fp.cu FloorFunctor (floorf/floor intrinsic call) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## ceil_inplace  (CeilInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=ceil(x[i]) in place Backs `OpKind::CeilInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: ceil_inplace
op_kind: CeilInplace
blurb: "x[i]=ceil(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::ceil_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_ceil_fp.cu CeilFunctor (ceilf/ceil intrinsic call) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## round_inplace  (RoundInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=round(x[i]) in place Backs `OpKind::RoundInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: round_inplace
op_kind: RoundInplace
blurb: "x[i]=round(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::round_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_round_fp.cu RoundFunctor (rintf/rint — round-half-to-even, matching Fuel's RoundElementwise contract) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## erf_inplace  (ErfInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=erf(x[i]) in place Backs `OpKind::ErfInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: erf_inplace
op_kind: ErfInplace
blurb: "x[i]=erf(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::erf_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_erf_fp.cu ErfFunctor (erff/erf intrinsic call, plain Gauss error function not a gelu flavor) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## gelu_erf_inplace  (GeluErfInplace - {F32, F64, BF16, F16}, contiguous-only, in-place)

x[i]=gelu_erf(x[i]) in place Backs `OpKind::GeluErfInplace`. baracuda unary_*_run same-pointer dispatch (a==y); contiguous target, no params. Output: fresh, contiguous, no aliasing.

```fkc
kernel: gelu_erf_inplace
op_kind: GeluErfInplace
blurb: "x[i]=gelu_erf(x[i]) in place (CUDA/baracuda) {F32, F64, BF16, F16}; contiguous-only, in-place; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::gelu_erf_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  notes: "Reasoned from source: baracuda unary_gelu_erf_fp.cu GeluErfFunctor (`0.5*x*(1+erf(x/sqrt2))`) via the shared unary_pointwise_contig_kernel grid-stride template — each thread computes y[i]=op(x[i]) independently from x[i] alone, no atomics/shared-mem/cross-thread state; same-pointer in-place dispatch is safe since each thread reads x[i] before writing y[i] (fuel-cuda-backend unary_inplace_run). Bit-stable same hardware."

determinism: same_hardware_bitwise
```

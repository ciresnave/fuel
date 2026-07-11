---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - forward elementwise unary (21 ops x 4 dtypes + f32-only Sign) kernel contracts

CUDA (baracuda) baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is now `audited: true` for all 22 sections (2026-07-11
precision audit) — every kernel here is a baracuda whitebox elementwise functor instantiated
through the shared `unary_pointwise_contig_kernel`/`unary_pointwise_strided_kernel` templates
(one thread per output index, no atomics, no `__shared__`, no cross-thread reduction), so
`bit_stable_on_same_hardware: true` is reasoned from source, not an unaudited seed. See each
section's `precision.notes` for the specific source file/functor cited.


---

## neg  (NegElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=neg(in[i]) Backs `OpKind::NegElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: neg
op_kind: NegElementwise
blurb: "out[i]=neg(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::neg"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_neg_fp.cu: NegFunctor is a single PTX neg instruction per element, instantiated through the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__, no cross-thread reduction) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## abs  (AbsElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=abs(in[i]) Backs `OpKind::AbsElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: abs
op_kind: AbsElementwise
blurb: "out[i]=abs(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::abs"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_abs_fp.cu: AbsFunctor is a single fabsf/fabs (or f16/bf16 f32-detour) per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## sqr  (SqrElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=sqr(in[i]) Backs `OpKind::SqrElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sqr
op_kind: SqrElementwise
blurb: "out[i]=sqr(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sqr"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_square_fp.cu: SquareFunctor is a single `x*x` FMUL/HMUL per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## sqrt  (SqrtElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=sqrt(in[i]) Backs `OpKind::SqrtElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sqrt
op_kind: SqrtElementwise
blurb: "out[i]=sqrt(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sqrt"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_sqrt_fp.cu: SqrtFunctor calls sqrtf/sqrt (f16/bf16 via f32-detour) per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## recip  (RecipElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=recip(in[i]) Backs `OpKind::RecipElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: recip
op_kind: RecipElementwise
blurb: "out[i]=recip(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::recip"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_reciprocal_fp.cu: ReciprocalFunctor is a single `1/x` division per element (f16/bf16 via f32-detour) via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## exp  (ExpElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=exp(in[i]) Backs `OpKind::ExpElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: exp
op_kind: ExpElementwise
blurb: "out[i]=exp(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::exp"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_exp_fp.cu: ExpFunctor calls expf/exp (f16/bf16 via f32-detour) per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## log  (LogElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=log(in[i]) Backs `OpKind::LogElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: log
op_kind: LogElementwise
blurb: "out[i]=log(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::log"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_log_fp.cu: LogFunctor calls logf/log (f16/bf16 via f32-detour) per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## sin  (SinElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=sin(in[i]) Backs `OpKind::SinElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sin
op_kind: SinElementwise
blurb: "out[i]=sin(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sin"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_sin_fp.cu: SinFunctor calls sinf/sin (f16/bf16 via f32-detour) per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## cos  (CosElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=cos(in[i]) Backs `OpKind::CosElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: cos
op_kind: CosElementwise
blurb: "out[i]=cos(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::cos"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_cos_fp.cu: CosFunctor calls cosf/cos (f16/bf16 via f32-detour) per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## tanh  (TanhElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=tanh(in[i]) Backs `OpKind::TanhElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: tanh
op_kind: TanhElementwise
blurb: "out[i]=tanh(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::tanh"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_tanh_fp.cu: TanhFunctor calls tanhf/tanh (f16/bf16 via f32-detour) per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## relu  (ReluElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=relu(in[i]), NaN-propagating (torch.relu convention: NaN input passes through; `-0.0`
stays `-0.0`). Backs `OpKind::ReluElementwise`. Binds baracuda's bespoke
`unary_relu_propagating_*` family (alpha.76+, `baracuda_kernels_unary_relu_propagating_{f32,f16,bf16,f64}[_strided]_run`)
— contig + strided_run per dtype; wrapper picks per-call. Sibling of the NaN-SCRUBBING
`unary_relu_*` (Fmax/`fmaxf`) family, which stays registered for other consumers (and still
backs `OpKind::ReluInplace` — a residual, not-yet-rebound gap; see
`fuel-cuda-backend/src/baracuda/elementwise.rs` next to `unary_inplace_relu_f32`). CPU/CUDA now
agree on NaN handling (pinned 2026-07-08, `docs/architecture/10-decisions-log.md`; live pin:
`fuel-dispatch/tests/cuda_dispatch_live.rs::cuda_relu_propagates_nan_f32` + bf16 sibling —
direct binding-table invocation, born-red-verified). Gelu is the TANH approx
(unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no
aliasing.

```fkc
kernel: relu
op_kind: ReluElementwise
blurb: "out[i]=relu(in[i]), NaN-propagating (torch parity) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::relu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_relu_propagating_fp.cu: ReluPropagatingFunctor is a single `x < 0 ? 0 : x` compare-select per element (f16/bf16 via f32-detour) via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## gelu_tanh  (GeluElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=gelu_tanh(in[i]) Backs `OpKind::GeluElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: gelu_tanh
op_kind: GeluElementwise
blurb: "out[i]=gelu_tanh(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::gelu_tanh"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_gelu_tanh_fp.cu: GeluTanhFunctor computes the tanh-approx GELU (`0.5*x*(1+tanh(...))`, f16/bf16 via f32-detour) per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## gelu  (GeluErfElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=gelu(in[i]) Backs `OpKind::GeluErfElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: gelu
op_kind: GeluErfElementwise
blurb: "out[i]=gelu(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::gelu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_gelu_fp.cu (the erf-exact GELU, bound to GeluErfElementwise): GeluFunctor computes `0.5*x*(1+erf(x/sqrt(2)))` via erff/erf (f16/bf16 via f32-detour) per element through the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## step  (StepElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=step(in[i]) Backs `OpKind::StepElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: step
op_kind: StepElementwise
blurb: "out[i]=step(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::step"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_step_fp.cu: StepFunctor is a single `x > 0 ? 1 : 0` compare-select per element (f16/bf16 route through f32 compares) via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## silu  (SiluElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=silu(in[i]) Backs `OpKind::SiluElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: silu
op_kind: SiluElementwise
blurb: "out[i]=silu(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::silu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_silu_fp.cu: SiluFunctor computes `x*sigmoid(x)` with a numerically-stable two-branch form (branch is purely a function of x, no cross-thread state; f16/bf16 via f32-detour) via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## sigmoid  (SigmoidElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=sigmoid(in[i]) Backs `OpKind::SigmoidElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sigmoid
op_kind: SigmoidElementwise
blurb: "out[i]=sigmoid(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sigmoid"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_sigmoid_fp.cu: SigmoidFunctor computes `1/(1+exp(-x))` with a numerically-stable two-branch form (branch is purely a function of x, no cross-thread state; f16/bf16 via f32-detour) via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## rsqrt  (RsqrtElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=rsqrt(in[i]) Backs `OpKind::RsqrtElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: rsqrt
op_kind: RsqrtElementwise
blurb: "out[i]=rsqrt(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::rsqrt"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_rsqrt_fp.cu: f32 uses the single-instruction `rsqrtf` intrinsic, f64 composes `1/sqrt(x)`, f16/bf16 detour through f32 — all per-element, no atomics, no __shared__, via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates; the approx PTX instruction is still deterministic same-hardware/same-driver (accuracy, not determinism, is what 'approx' affects) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## floor  (FloorElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=floor(in[i]) Backs `OpKind::FloorElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: floor
op_kind: FloorElementwise
blurb: "out[i]=floor(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::floor"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_floor_fp.cu: FloorFunctor calls floorf/floor (f16/bf16 via f32-detour) per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## ceil  (CeilElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=ceil(in[i]) Backs `OpKind::CeilElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: ceil
op_kind: CeilElementwise
blurb: "out[i]=ceil(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::ceil"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_ceil_fp.cu: CeilFunctor calls ceilf/ceil (f16/bf16 via f32-detour) per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## round  (RoundElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=round(in[i]) Backs `OpKind::RoundElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: round
op_kind: RoundElementwise
blurb: "out[i]=round(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::round"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_round_fp.cu: RoundFunctor calls rintf/rint (round-half-to-even, f16/bf16 via f32-detour) per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## erf  (ErfElementwise - {F32, F16, BF16, F64}, strided+broadcast)

out[i]=erf(in[i]) Backs `OpKind::ErfElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: erf
op_kind: ErfElementwise
blurb: "out[i]=erf(in[i]) (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::erf"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_erf_fp.cu: ErfFunctor calls erff/erf (f16/bf16 via f32-detour) per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## sign  (SignElementwise - {F32}, strided+broadcast)

out[i]=sign(in[i]) in {-1,0,1} (f32-only) Backs `OpKind::SignElementwise`. baracuda unary_* contig + <sym>_strided_run; wrapper picks per-call. Gelu is the TANH approx (unary::gelu_tanh); GeluErf is erf (unary::gelu); Sign is f32-only. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sign
op_kind: SignElementwise
blurb: "out[i]=sign(in[i]) in {-1,0,1} (f32-only) (CUDA/baracuda) {F32}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sign"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params: { variant: None }

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
  audited: true
  notes: "baracuda-kernels-sys/kernels/elementwise/unary_sign_fp.cu: SignFunctor<float> is a single piecewise compare-select (`x>0 ? 1 : x<0 ? -1 : 0`) per element via the shared unary_pointwise_contig_kernel/unary_pointwise_strided_kernel templates (one thread per output index, no atomics, no __shared__) — bit-stable same hardware."

determinism: same_hardware_bitwise
```

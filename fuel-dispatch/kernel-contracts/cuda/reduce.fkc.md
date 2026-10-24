---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - axis reductions (sum / max / min / mean) kernel contracts

CUDA (baracuda) Single dispatch covers all axis configs; the wrapper destructures OpParams::Reduce { dims, keepdim }. Stride-driven FFI.. Each section binds one concrete `OpKind` and fans its operand(s)
over the accepted dtypes (sec 3.4 dtype-fan; base `entry_point` -> `<op>_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]). Caps ride through the import truthfully (sec 6 / caps_map):
each per-operand five-flag layout projects onto
`KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)` (AND-ed across
operands) - byte-for-byte the deleted hand-written `register_with_caps(..., strided)` regs. Cost is
`judge_measured` (the fill_unset pass upgrades the imported unknown_cost sentinel to the shared
per-OpKind CUDA cost fn); precision is the author-declared `audited: false` -> UNAUDITED seed, except the F32 sections, which carry measured accuracy claims (row 39).


---

## sum_f32  (SumReduce - F32, strided+broadcast; carries the numeric accuracy claim)

out=sum over dims Backs `OpKind::SumReduce`. Single dispatch covers all axis configs; the wrapper destructures OpParams::Reduce { dims, keepdim }. Stride-driven FFI. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sum_f32
op_kind: SumReduce
blurb: "out=sum over dims (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sum_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduces_to=out"
  op_params: { variant: Reduce }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: reduced(in)
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
  max_relative: 0.0001   # MEASURED (row 39): worst 3.2e-6 rel / 4.3e-5 abs over 9 reduce cases up to 65536 elements, RTX 4070 Laptop (fuel-dev-tools seed-cuda-accuracy); not evidence about other inputs, GPUs, or the other dtype sections
  max_absolute: 0.0001
  audited: true
  notes: "out=sum over dims. Accuracy claim earned by measurement against an f64 host reduction (row 39); accumulation order inside the baracuda reduce kernel was not source-audited here, and bit-stability is the ledger pass on the same hardware."

determinism: same_hardware_bitwise
```

---

## sum  (SumReduce - {F16, BF16, F64}, strided+broadcast)

out=sum over dims Backs `OpKind::SumReduce`. Single dispatch covers all axis configs; the wrapper destructures OpParams::Reduce { dims, keepdim }. Stride-driven FFI. Output: fresh, contiguous, no aliasing.

```fkc
kernel: sum
op_kind: SumReduce
blurb: "out=sum over dims (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::sum"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduces_to=out"
  op_params: { variant: Reduce }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: reduced(in)
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
  notes: "out=sum over dims. No numeric accuracy claim for these dtypes (row 39 earns the F32 section only); bit-stability is the ledger pass on the same hardware, earned by repeat-call byte identity, not by reading the accumulation order."

determinism: same_hardware_bitwise
```

---

## max_f32  (MaxReduce - F32, strided+broadcast; carries the numeric accuracy claim)

out=max over dims Backs `OpKind::MaxReduce`. Single dispatch covers all axis configs; the wrapper destructures OpParams::Reduce { dims, keepdim }. Stride-driven FFI. Output: fresh, contiguous, no aliasing.

```fkc
kernel: max_f32
op_kind: MaxReduce
blurb: "out=max over dims (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::max_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduces_to=out"
  op_params: { variant: Reduce }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: reduced(in)
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
  max_relative: 0   # MEASURED (row 39): worst 0 rel / 0 abs, RTX 4070 Laptop (fuel-dev-tools seed-cuda-accuracy); not evidence about other inputs, GPUs, or the other dtype sections
  max_absolute: 0
  audited: true
  notes: "out=max over dims. Accuracy claim earned by measurement against an f64 host reduction (row 39); accumulation order inside the baracuda reduce kernel was not source-audited here, and bit-stability is the ledger pass on the same hardware."

determinism: same_hardware_bitwise
```

---

## max  (MaxReduce - {F16, BF16, F64}, strided+broadcast)

out=max over dims Backs `OpKind::MaxReduce`. Single dispatch covers all axis configs; the wrapper destructures OpParams::Reduce { dims, keepdim }. Stride-driven FFI. Output: fresh, contiguous, no aliasing.

```fkc
kernel: max
op_kind: MaxReduce
blurb: "out=max over dims (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::max"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduces_to=out"
  op_params: { variant: Reduce }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: reduced(in)
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
  notes: "out=max over dims. No numeric accuracy claim for these dtypes (row 39 earns the F32 section only); bit-stability is the ledger pass on the same hardware, earned by repeat-call byte identity, not by reading the accumulation order."

determinism: same_hardware_bitwise
```

---

## min_f32  (MinReduce - F32, strided+broadcast; carries the numeric accuracy claim)

out=min over dims Backs `OpKind::MinReduce`. Single dispatch covers all axis configs; the wrapper destructures OpParams::Reduce { dims, keepdim }. Stride-driven FFI. Output: fresh, contiguous, no aliasing.

```fkc
kernel: min_f32
op_kind: MinReduce
blurb: "out=min over dims (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::min_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduces_to=out"
  op_params: { variant: Reduce }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: reduced(in)
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
  max_relative: 0   # MEASURED (row 39): worst 0 rel / 0 abs, RTX 4070 Laptop (fuel-dev-tools seed-cuda-accuracy); not evidence about other inputs, GPUs, or the other dtype sections
  max_absolute: 0
  audited: true
  notes: "out=min over dims. Accuracy claim earned by measurement against an f64 host reduction (row 39); accumulation order inside the baracuda reduce kernel was not source-audited here, and bit-stability is the ledger pass on the same hardware."

determinism: same_hardware_bitwise
```

---

## min  (MinReduce - {F16, BF16, F64}, strided+broadcast)

out=min over dims Backs `OpKind::MinReduce`. Single dispatch covers all axis configs; the wrapper destructures OpParams::Reduce { dims, keepdim }. Stride-driven FFI. Output: fresh, contiguous, no aliasing.

```fkc
kernel: min
op_kind: MinReduce
blurb: "out=min over dims (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::min"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduces_to=out"
  op_params: { variant: Reduce }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: reduced(in)
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
  notes: "out=min over dims. No numeric accuracy claim for these dtypes (row 39 earns the F32 section only); bit-stability is the ledger pass on the same hardware, earned by repeat-call byte identity, not by reading the accumulation order."

determinism: same_hardware_bitwise
```

---

## mean_f32  (MeanReduce - F32, strided+broadcast; carries the numeric accuracy claim)

out=mean over dims Backs `OpKind::MeanReduce`. Single dispatch covers all axis configs; the wrapper destructures OpParams::Reduce { dims, keepdim }. Stride-driven FFI. Output: fresh, contiguous, no aliasing.

```fkc
kernel: mean_f32
op_kind: MeanReduce
blurb: "out=mean over dims (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::mean_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduces_to=out"
  op_params: { variant: Reduce }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: reduced(in)
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
  max_relative: 0.00001   # MEASURED (row 39): worst 7.5e-8 rel / 7.5e-8 abs, RTX 4070 Laptop (fuel-dev-tools seed-cuda-accuracy); not evidence about other inputs, GPUs, or the other dtype sections
  max_absolute: 0.00001
  audited: true
  notes: "out=mean over dims. Accuracy claim earned by measurement against an f64 host reduction (row 39); accumulation order inside the baracuda reduce kernel was not source-audited here, and bit-stability is the ledger pass on the same hardware."

determinism: same_hardware_bitwise
```

---

## mean  (MeanReduce - {F16, BF16, F64}, strided+broadcast)

out=mean over dims Backs `OpKind::MeanReduce`. Single dispatch covers all axis configs; the wrapper destructures OpParams::Reduce { dims, keepdim }. Stride-driven FFI. Output: fresh, contiguous, no aliasing.

```fkc
kernel: mean
op_kind: MeanReduce
blurb: "out=mean over dims (CUDA/baracuda) {F32, F16, BF16, F64}; strided+broadcast; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::mean"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduces_to=out"
  op_params: { variant: Reduce }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: reduced(in)
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
  notes: "out=mean over dims. No numeric accuracy claim for these dtypes (row 39 earns the F32 section only); bit-stability is the ledger pass on the same hardware, earned by repeat-call byte identity, not by reading the accumulation order."

determinism: same_hardware_bitwise
```

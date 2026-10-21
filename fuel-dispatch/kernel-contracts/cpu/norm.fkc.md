---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                 # maps to BackendId::Cpu
  kernel_source: "portable-cpu"   # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::fkc::ENTRY_POINTS  # symbol → KernelRef map (§12.6)
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — norm-family kernel contracts

Portable byte-shaped last-dim normalization and softmax kernels from
`fuel-cpu-backend/src/byte_kernels.rs`. Every kernel here is a **row-wise reduction along the last
dimension**: the flat buffer is viewed as `outer_count` rows of `last_dim` contiguous elements, and
each row is reduced (max / sum / mean / variance) and rewritten independently. All kernels operate
on contiguous, zero-offset, row-major `CpuStorageBytes` slices (they validate byte/element length
against `outer_count × last_dim`; they never consult a `Layout`, strides, or an offset — the
pipelined executor's auto-Contiguize pass realizes any strided/broadcast/offset input into a dense
buffer first). Output is caller-pre-allocated, fully overwritten, same dtype and shape as the input,
contiguous row-major; no aliasing.

The dtype-monomorphization rule: f32 and f64 evaluate natively in their own type; bf16 and f16 widen
to **f32** for all reduction arithmetic (row-max, sum, sum-of-squares, exp, ln, reciprocal-sqrt) and
narrow back to the half type only on the final store. The f32 (or f64) accumulator for half I/O is a
load-bearing precision invariant, not an accident. The norm kernels (RMS / LayerNorm) carry **no
affine (gamma/beta) parameters** — they are the bare normalization; an affine scale/shift is a
separate downstream op. `eps` is an `f64` op-param (graph-API consistency), narrowed to `f32` inside
the half/f32 kernels and used natively in the f64 kernel.

`last_dim == 0` is an early `Ok(())` no-op (nothing to reduce). The cost of every kernel in this file
is marked **`declared`** — each block carries an authored absolute `overhead_ns` launch-cost prior
(a legitimate author prior the Judge later refines, §4.4); the genuinely derivable bandwidth/FLOP-shape
hints are recorded in the cost-expression strings alongside it.

---

## softmax_last_dim_f32  (numerically-stable softmax along the last dim — f32)

Row-wise softmax with the standard max-subtract stabilization: per row, find `row_max`, write
`exp(x - row_max)` into the output, accumulate the sum, then multiply the row by `1/sum`. All
arithmetic is native f32 (`byte_kernels.rs:2259`). No affine, no temperature; pure softmax. Two
passes over each row plus the reciprocal-scale pass (read+write the row twice). Validates
`input.len_bytes == out.len_bytes == outer_count × last_dim × 4`. Output is the same dtype/shape,
contiguous, overwritten. Numerics: standard stable softmax; `sum` is a flat f32 accumulator (no
pairwise/Kahan), so it is deterministic and bit-stable on the same hardware but order-dependent.
Limitation: contiguous-only — any strided/broadcast/offset input is contiguized by the planner first.

```fkc
kernel: softmax_last_dim_f32
op_kind: SoftmaxLastDim
blurb: "Numerically-stable softmax along the last dim (f32); row max-subtract, exp, normalize."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::softmax_last_dim_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: SoftmaxLastDim          # OpParams::SoftmaxLastDim (primitive namespace; §3.7)
    fields:
      outer_count: { kind: usize, note: "product of all dims before the last = number of rows" }
      last_dim:    { kind: usize, note: "reduced last-dim length = row width" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines the hints below (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 4"   # ~max + exp + sum + scale per element (exp dominates)
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"   # read input once, write out once (bandwidth-bound)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic sequential reduction; native f32
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32; stable softmax (row max-subtract). Flat f32 sum accumulator: deterministic, order-dependent, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## softmax_last_dim_f64  (numerically-stable softmax along the last dim — f64)

Same stable algorithm as `softmax_last_dim_f32`, native f64 arithmetic throughout
(`byte_kernels.rs:2430`). Validates against `outer_count × last_dim × 8` bytes. f64 gives the widest
range/precision of the family; the flat f64 `sum` accumulator is deterministic and bit-stable on the
same hardware. Contiguous-only.

```fkc
kernel: softmax_last_dim_f64
op_kind: SoftmaxLastDim
blurb: "Numerically-stable softmax along the last dim (f64); native f64 arithmetic."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::softmax_last_dim_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: SoftmaxLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 4"
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic sequential reduction; native f64
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64; stable softmax (row max-subtract). Flat f64 sum accumulator: deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## softmax_last_dim_bf16  (numerically-stable softmax along the last dim — bf16, f32 accumulator)

bf16 I/O softmax; all arithmetic (row-max, exp, sum, reciprocal-scale) is performed in **f32**, with
narrowing to bf16 only on store (`softmax_last_dim_half!`, `byte_kernels.rs:2425`). The f32 work
accumulator is the precision invariant for the half path. Validates against `outer_count × last_dim ×
2` bytes. Output bf16, same shape, contiguous, overwritten. Deterministic and bit-stable on the same
hardware (the half→f32 widening, the f32 reduction, and the f32→bf16 narrowing are all fixed-order).
Contiguous-only.

```fkc
kernel: softmax_last_dim_bf16
op_kind: SoftmaxLastDim
blurb: "Numerically-stable softmax along the last dim (bf16; f32 accumulator)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::softmax_last_dim_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: SoftmaxLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 4"
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32-accumulated, fixed-order; narrow only on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 I/O, all math in f32 (widen on load, narrow bf16 on store). Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## softmax_last_dim_f16  (numerically-stable softmax along the last dim — f16, f32 accumulator)

f16 I/O softmax; identical structure to the bf16 variant — all arithmetic in **f32**, narrow to f16
only on store (`softmax_last_dim_half!`, `byte_kernels.rs:2426`). Validates against `outer_count ×
last_dim × 2` bytes. Output f16, same shape, contiguous, overwritten. Deterministic, bit-stable on
the same hardware. Contiguous-only.

```fkc
kernel: softmax_last_dim_f16
op_kind: SoftmaxLastDim
blurb: "Numerically-stable softmax along the last dim (f16; f32 accumulator)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::softmax_last_dim_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: SoftmaxLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 4"
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32-accumulated, fixed-order; narrow only on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O, all math in f32 (widen on load, narrow f16 on store). Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## log_softmax_last_dim_f32  (numerically-stable log-softmax along the last dim — f32)

Row-wise log-softmax via row-max + log-sum-exp: per row, find `row_max` (seeded
`f32::NEG_INFINITY`), accumulate `sum += exp(x - row_max)`, take `log_sum = ln(sum)`, then write
`x - row_max - log_sum`. Native f32 (`log_softmax_last_dim_kernel!` instantiation,
`byte_kernels.rs:1002`). Validates **element counts** (`in.len() == out.len() == outer × last_dim`).
Output f32, same shape, contiguous, overwritten. The flat f32 `sum` accumulator is deterministic and
bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: log_softmax_last_dim_f32
op_kind: LogSoftmaxLastDim
blurb: "Numerically-stable log-softmax along the last dim (f32); row max + log-sum-exp."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::log_softmax_last_dim_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: LogSoftmaxLastDim          # OpParams::LogSoftmaxLastDim (primitive namespace; §3.7)
    fields:
      outer_count: { kind: usize, note: "number of rows" }
      last_dim:    { kind: usize, note: "row width" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 4"   # max + exp/sum + ln + subtract per element
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"   # read input once, write out once
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f32, deterministic sequential reduction
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32; row max + log-sum-exp. Flat f32 sum accumulator: deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## log_softmax_last_dim_f64  (numerically-stable log-softmax along the last dim — f64)

Hand-written f64 log-softmax with a native f64 accumulator path for maximum range/precision
(`byte_kernels.rs:1008`). Same row-max + log-sum-exp algorithm as the f32 variant, all arithmetic in
f64. Validates element counts. Output f64, same shape, contiguous, overwritten. Deterministic,
bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: log_softmax_last_dim_f64
op_kind: LogSoftmaxLastDim
blurb: "Numerically-stable log-softmax along the last dim (f64); native f64 accumulator."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::log_softmax_last_dim_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: LogSoftmaxLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 4"
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f64, deterministic
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64 accumulator for max range/precision; row max + log-sum-exp. Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## log_softmax_last_dim_bf16  (numerically-stable log-softmax along the last dim — bf16, f32 accumulator)

bf16 I/O log-softmax; all arithmetic (row-max, exp, sum, ln, subtract) in **f32**, narrow to bf16
only on the final store (`log_softmax_last_dim_kernel!` over bf16, `byte_kernels.rs:1041`). The f32
accumulator is the precision invariant. Validates element counts. Output bf16, same shape,
contiguous, overwritten. Deterministic, bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: log_softmax_last_dim_bf16
op_kind: LogSoftmaxLastDim
blurb: "Numerically-stable log-softmax along the last dim (bf16; f32 accumulator)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::log_softmax_last_dim_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: LogSoftmaxLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 4"
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32-accumulated, fixed-order; narrow only on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 I/O, all math in f32 (widen on load, narrow bf16 on store). Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## log_softmax_last_dim_f16  (numerically-stable log-softmax along the last dim — f16, f32 accumulator)

f16 I/O log-softmax; identical structure to the bf16 variant — all math in **f32**, narrow to f16
only on store (`log_softmax_last_dim_kernel!` over f16, `byte_kernels.rs:1045`). Validates element
counts. Output f16, same shape, contiguous, overwritten. Deterministic, bit-stable on the same
hardware. Contiguous-only.

```fkc
kernel: log_softmax_last_dim_f16
op_kind: LogSoftmaxLastDim
blurb: "Numerically-stable log-softmax along the last dim (f16; f32 accumulator)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::log_softmax_last_dim_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: LogSoftmaxLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 4"
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32-accumulated, fixed-order; narrow only on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O, all math in f32 (widen on load, narrow f16 on store). Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_f32  (RMS normalization along the last dim, no affine — f32)

Row-wise RMS normalization: `out[i] = x[i] / sqrt(mean(x²) + eps)` per row, where `mean(x²) =
(Σ x²) / last_dim`. Native f32 (`byte_kernels.rs:2339`). **No affine (gamma) parameter** — bare RMS
norm. `eps` arrives as an `f64` op-param and is narrowed to `f32` (`eps32 = eps as f32`) before use.
Validates `input.len_bytes == out.len_bytes == outer_count × last_dim × 4` (`check_norm_lens`).
Output f32, same shape, contiguous, overwritten. One reduction pass (sum-of-squares) + one write
pass. Deterministic, bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: rms_norm_last_dim_f32
op_kind: RmsNormLastDim
blurb: "RMS norm along the last dim, no affine (f32): x / sqrt(mean(x^2) + eps)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rms_norm_last_dim_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim                 # OpParams::NormLastDim (shared by RmsNorm + LayerNorm; OpKind selects); §3.7
    fields:
      outer_count: { kind: usize, note: "number of rows" }
      last_dim:    { kind: usize, note: "row width; divisor of the mean" }
      eps:         { kind: f64, note: "f64 op-param, narrowed to f32 in-kernel" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 3"   # x^2 + accumulate (reduce pass) + scale (write pass)
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"   # read input once, write out once
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f32, deterministic sequential reduction
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32; no affine; eps f64 narrowed to f32. Flat f32 sum-of-squares: deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_f64  (RMS normalization along the last dim, no affine — f64)

Native f64 RMS norm; same algorithm as the f32 variant, all arithmetic in f64, `eps` used natively
without narrowing (`byte_kernels.rs:2559`). No affine. Validates against `outer_count × last_dim × 8`
bytes. Output f64, same shape, contiguous, overwritten. Deterministic, bit-stable on the same
hardware. Contiguous-only.

```fkc
kernel: rms_norm_last_dim_f64
op_kind: RmsNormLastDim
blurb: "RMS norm along the last dim, no affine (f64): x / sqrt(mean(x^2) + eps)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rms_norm_last_dim_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }
      eps:         { kind: f64, note: "used natively in f64 (no narrowing)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 3"
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f64, deterministic
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64; no affine; eps used natively. Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_bf16  (RMS normalization along the last dim, no affine — bf16, f32 accumulator)

bf16 I/O RMS norm; sum-of-squares and reciprocal-sqrt computed in **f32**, narrow to bf16 only on
store (`rms_norm_last_dim_half!`, `byte_kernels.rs:2554`). `eps` narrowed to `f32`. No affine. The
f32 accumulator is the precision invariant. Validates against `outer_count × last_dim × 2` bytes.
Output bf16, same shape, contiguous, overwritten. Deterministic, bit-stable on the same hardware.
Contiguous-only.

```fkc
kernel: rms_norm_last_dim_bf16
op_kind: RmsNormLastDim
blurb: "RMS norm along the last dim, no affine (bf16; f32 accumulator)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rms_norm_last_dim_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }
      eps:         { kind: f64, note: "narrowed to f32 in-kernel" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 3"
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32-accumulated, fixed-order; narrow only on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 I/O, sum-of-squares + rsqrt in f32 (widen on load, narrow bf16 on store); no affine; eps f32. Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_f16  (RMS normalization along the last dim, no affine — f16, f32 accumulator)

f16 I/O RMS norm; identical structure to the bf16 variant — all reduction math in **f32**, narrow to
f16 only on store (`rms_norm_last_dim_half!`, `byte_kernels.rs:2555`). `eps` narrowed to `f32`. No
affine. Validates against `outer_count × last_dim × 2` bytes. Output f16, same shape, contiguous,
overwritten. Deterministic, bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: rms_norm_last_dim_f16
op_kind: RmsNormLastDim
blurb: "RMS norm along the last dim, no affine (f16; f32 accumulator)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rms_norm_last_dim_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }
      eps:         { kind: f64, note: "narrowed to f32 in-kernel" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 3"
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32-accumulated, fixed-order; narrow only on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O, sum-of-squares + rsqrt in f32 (widen on load, narrow f16 on store); no affine; eps f32. Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_f32  (layer normalization along the last dim, no affine — f32)

Row-wise layer normalization: `out[i] = (x[i] - mean(x)) / sqrt(var(x) + eps)` per row, with
`mean = (Σ x) / last_dim` and `var = (Σ (x - mean)²) / last_dim`. Native f32 (`byte_kernels.rs:2477`).
**No affine (gamma/beta) parameters** — bare LayerNorm. `eps` arrives as an `f64` op-param and is
narrowed to `f32` before use. Two reduction passes per row (mean, then variance) + one write pass.
Validates `input.len_bytes == out.len_bytes == outer_count × last_dim × 4` (`check_norm_lens`).
Output f32, same shape, contiguous, overwritten. Deterministic, bit-stable on the same hardware.
Contiguous-only.

```fkc
kernel: layer_norm_last_dim_f32
op_kind: LayerNormLastDim
blurb: "LayerNorm along the last dim, no affine (f32): (x - mean) / sqrt(var + eps)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::layer_norm_last_dim_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim                 # OpParams::NormLastDim (shared by RmsNorm + LayerNorm; OpKind selects); §3.7
    fields:
      outer_count: { kind: usize, note: "number of rows" }
      last_dim:    { kind: usize, note: "row width; divisor of mean and variance" }
      eps:         { kind: f64, note: "f64 op-param, narrowed to f32 in-kernel" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 5"   # mean pass + variance pass + normalize pass
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"   # read input once, write out once
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f32, deterministic sequential reductions
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32; no affine; two-pass mean/variance; eps f64 narrowed to f32. Flat f32 accumulators: deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_f64  (layer normalization along the last dim, no affine — f64)

Native f64 LayerNorm; same two-pass mean/variance algorithm as the f32 variant, all arithmetic in
f64, `eps` used natively (`byte_kernels.rs:2660`). No affine. Validates against `outer_count ×
last_dim × 8` bytes. Output f64, same shape, contiguous, overwritten. Deterministic, bit-stable on
the same hardware. Contiguous-only.

```fkc
kernel: layer_norm_last_dim_f64
op_kind: LayerNormLastDim
blurb: "LayerNorm along the last dim, no affine (f64): (x - mean) / sqrt(var + eps)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::layer_norm_last_dim_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }
      eps:         { kind: f64, note: "used natively in f64 (no narrowing)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 5"
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f64, deterministic
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64; no affine; two-pass mean/variance; eps used natively. Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_bf16  (layer normalization along the last dim, no affine — bf16, f32 accumulator)

bf16 I/O LayerNorm; mean, variance, and reciprocal-sqrt computed in **f32**, narrow to bf16 only on
store (`layer_norm_last_dim_half!`, `byte_kernels.rs:2655`). `eps` narrowed to `f32`. No affine. The
f32 accumulator is the precision invariant. Validates against `outer_count × last_dim × 2` bytes.
Output bf16, same shape, contiguous, overwritten. Deterministic, bit-stable on the same hardware.
Contiguous-only.

```fkc
kernel: layer_norm_last_dim_bf16
op_kind: LayerNormLastDim
blurb: "LayerNorm along the last dim, no affine (bf16; f32 accumulator)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::layer_norm_last_dim_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }
      eps:         { kind: f64, note: "narrowed to f32 in-kernel" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 5"
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32-accumulated, fixed-order; narrow only on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 I/O, mean/variance/rsqrt in f32 (widen on load, narrow bf16 on store); no affine; eps f32. Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_f16  (layer normalization along the last dim, no affine — f16, f32 accumulator)

f16 I/O LayerNorm; identical structure to the bf16 variant — all reduction math in **f32**, narrow to
f16 only on store (`layer_norm_last_dim_half!`, `byte_kernels.rs:2656`). `eps` narrowed to `f32`. No
affine. Validates against `outer_count × last_dim × 2` bytes. Output f16, same shape, contiguous,
overwritten. Deterministic, bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: layer_norm_last_dim_f16
op_kind: LayerNormLastDim
blurb: "LayerNorm along the last dim, no affine (f16; f32 accumulator)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::layer_norm_last_dim_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }
      eps:         { kind: f64, note: "narrowed to f32 in-kernel" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared              # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 5"
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32-accumulated, fixed-order; narrow only on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O, mean/variance/rsqrt in f32 (widen on load, narrow f16 on store); no affine; eps f32. Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

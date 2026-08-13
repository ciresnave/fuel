---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend — elementwise comparison (eq / ne / lt / le / gt / ge) kernel contracts

CUDA elementwise comparison kernels: typed `T x T` inputs, **`Bool` output mask** (1 where
the predicate holds else 0) — GAP-168(c); these returned a `U8` mask before that cut.

> **The `_u8` in every kernel NAME below is historical and deliberately NOT renamed.**
> The FKC verifier enforces the dtype TOKEN and provably ignores the kernel name, so
> `eq_f32_u8` declaring `fixed(BOOL)` is accepted because `BOOL` is legal — not because
> nothing is checking. Renaming would churn 24 entry_points and their Rust symbols for
> no enforcement gain; the load-bearing declaration is the `dtype_rule`. 6 ops x 4 float dtypes = 24 cells, each backed by a bespoke baracuda
kernel pair (`baracuda_kernels_binary_cmp_<op>_<dtype>_{run,strided_run}`) reached through
`fuel-cuda-backend/src/baracuda/binary.rs`'s `compare_kernel!` and the dispatch wrappers in
`fuel-dispatch/src/baracuda_dispatch.rs`.

**Why this bundle exists.** `Op::PagedAttn`'s primitive recipe has one node that could not resolve
on CUDA — `GreaterEqualElementwise[F32,F32,U8]` — so the paged decode graph was host-placed, which
forced cross-device copies, which `capture_decode` rejects. The whole family is authored rather
than `ge` alone because the next consumer needing `lt` would hit an identical wall and the
incremental cost is one section each.

**Layout caps are INHERITED, not re-derived.** These kernels share `binary_run` / `binary_run_into`
and `BinaryStrides` **verbatim** with the arithmetic family in `binary.fkc.md`, so their layout
admissibility is identical by construction: strided accepted (baracuda ships a real `_strided_run`,
unlike the contiguous-only CPU chassis), broadcast stride-0 accepted, and **`start_offset`
rejected** — `BinaryStrides::from` consults only shapes and strides, and the pointer handed to the
kernel is the buffer base, so a non-zero offset would silently read from element 0. That rejection
is what keeps the executor's auto-Contiguize pass in front of these kernels; declaring it
`accepted` would produce wrong values with no error.

**What differs from the arithmetic family** is only what genuinely differs: `dtype_rule: fixed(BOOL)`
rather than `passthrough(lhs)` (one byte per element, `0`/`1`), an 8-bit output granularity, and a `bytes_moved` term whose write
side is `n` bytes rather than `n * sizeof(T)`.

**NaN.** baracuda states the family rule on the FFI surface: *"NaN handling follows IEEE 754: `Eq` /
ordered comparisons return 0 when either operand is NaN; `Ne` returns 1 (since `NaN != anything`)."*
That is exactly the CPU family's behaviour (`compare.rs:129-171`) and PyTorch's, so the two backends
agree with no Fuel-side adjustment. The kernels write only `0u8` and `1u8`.

Every cost block is `provenance: declared` — an author prior the Judge refines (§4.4), not a
fabricated measurement. All are bandwidth-bound elementwise kernels: one predicate evaluation per
output element (`n`), and `bytes_moved` is the literal read+write traffic.

---

## eq_f32_u8  (Equal, f32 -> Bool)

`a == b` over f32, Bool mask out. `compare_kernel!(binary_cmp_eq_f32, cmp_eq_f32, 4, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 4 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: `NaN == NaN` is false -> 0.

```fkc
kernel: eq_f32_u8
op_kind: EqualElementwise
blurb: "Elementwise a == b on f32 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::eq_f32_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 4 + n"          # read 2x f32, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f32; `NaN == NaN` is false -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## eq_f64_u8  (Equal, f64 -> Bool)

`a == b` over f64, Bool mask out. `compare_kernel!(binary_cmp_eq_f64, cmp_eq_f64, 8, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 8 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: `NaN == NaN` is false -> 0.

```fkc
kernel: eq_f64_u8
op_kind: EqualElementwise
blurb: "Elementwise a == b on f64 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::eq_f64_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 8 + n"          # read 2x f64, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f64; `NaN == NaN` is false -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## eq_f16_u8  (Equal, f16 -> Bool)

`a == b` over f16, Bool mask out. `compare_kernel!(binary_cmp_eq_f16, cmp_eq_f16, 2, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 2 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: `NaN == NaN` is false -> 0.

```fkc
kernel: eq_f16_u8
op_kind: EqualElementwise
blurb: "Elementwise a == b on f16 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::eq_f16_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 2 + n"          # read 2x f16, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f16; `NaN == NaN` is false -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## eq_bf16_u8  (Equal, bf16 -> Bool)

`a == b` over bf16, Bool mask out. `compare_kernel!(binary_cmp_eq_bf16, cmp_eq_bf16, 2, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 2 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: `NaN == NaN` is false -> 0.

```fkc
kernel: eq_bf16_u8
op_kind: EqualElementwise
blurb: "Elementwise a == b on bf16 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::eq_bf16_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 2 + n"          # read 2x bf16, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on bf16; `NaN == NaN` is false -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## ne_f32_u8  (NotEqual, f32 -> Bool)

`a != b` over f32, Bool mask out. `compare_kernel!(binary_cmp_ne_f32, cmp_ne_f32, 4, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 4 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: `NaN != anything` is true -> 1 (the ONE op that returns 1 on NaN).

```fkc
kernel: ne_f32_u8
op_kind: NotEqualElementwise
blurb: "Elementwise a != b on f32 (CUDA/baracuda); Bool mask out; NaN -> 1."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::ne_f32_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 4 + n"          # read 2x f32, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f32; `NaN != anything` is true -> 1 (the ONE op that returns 1 on NaN); kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## ne_f64_u8  (NotEqual, f64 -> Bool)

`a != b` over f64, Bool mask out. `compare_kernel!(binary_cmp_ne_f64, cmp_ne_f64, 8, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 8 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: `NaN != anything` is true -> 1 (the ONE op that returns 1 on NaN).

```fkc
kernel: ne_f64_u8
op_kind: NotEqualElementwise
blurb: "Elementwise a != b on f64 (CUDA/baracuda); Bool mask out; NaN -> 1."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::ne_f64_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 8 + n"          # read 2x f64, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f64; `NaN != anything` is true -> 1 (the ONE op that returns 1 on NaN); kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## ne_f16_u8  (NotEqual, f16 -> Bool)

`a != b` over f16, Bool mask out. `compare_kernel!(binary_cmp_ne_f16, cmp_ne_f16, 2, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 2 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: `NaN != anything` is true -> 1 (the ONE op that returns 1 on NaN).

```fkc
kernel: ne_f16_u8
op_kind: NotEqualElementwise
blurb: "Elementwise a != b on f16 (CUDA/baracuda); Bool mask out; NaN -> 1."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::ne_f16_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 2 + n"          # read 2x f16, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f16; `NaN != anything` is true -> 1 (the ONE op that returns 1 on NaN); kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## ne_bf16_u8  (NotEqual, bf16 -> Bool)

`a != b` over bf16, Bool mask out. `compare_kernel!(binary_cmp_ne_bf16, cmp_ne_bf16, 2, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 2 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: `NaN != anything` is true -> 1 (the ONE op that returns 1 on NaN).

```fkc
kernel: ne_bf16_u8
op_kind: NotEqualElementwise
blurb: "Elementwise a != b on bf16 (CUDA/baracuda); Bool mask out; NaN -> 1."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::ne_bf16_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 2 + n"          # read 2x bf16, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on bf16; `NaN != anything` is true -> 1 (the ONE op that returns 1 on NaN); kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## lt_f32_u8  (Less, f32 -> Bool)

`a < b` over f32, Bool mask out. `compare_kernel!(binary_cmp_lt_f32, cmp_lt_f32, 4, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 4 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: lt_f32_u8
op_kind: LessElementwise
blurb: "Elementwise a < b on f32 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::lt_f32_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 4 + n"          # read 2x f32, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f32; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## lt_f64_u8  (Less, f64 -> Bool)

`a < b` over f64, Bool mask out. `compare_kernel!(binary_cmp_lt_f64, cmp_lt_f64, 8, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 8 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: lt_f64_u8
op_kind: LessElementwise
blurb: "Elementwise a < b on f64 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::lt_f64_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 8 + n"          # read 2x f64, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f64; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## lt_f16_u8  (Less, f16 -> Bool)

`a < b` over f16, Bool mask out. `compare_kernel!(binary_cmp_lt_f16, cmp_lt_f16, 2, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 2 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: lt_f16_u8
op_kind: LessElementwise
blurb: "Elementwise a < b on f16 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::lt_f16_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 2 + n"          # read 2x f16, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f16; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## lt_bf16_u8  (Less, bf16 -> Bool)

`a < b` over bf16, Bool mask out. `compare_kernel!(binary_cmp_lt_bf16, cmp_lt_bf16, 2, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 2 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: lt_bf16_u8
op_kind: LessElementwise
blurb: "Elementwise a < b on bf16 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::lt_bf16_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 2 + n"          # read 2x bf16, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on bf16; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## le_f32_u8  (LessEqual, f32 -> Bool)

`a <= b` over f32, Bool mask out. `compare_kernel!(binary_cmp_le_f32, cmp_le_f32, 4, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 4 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: le_f32_u8
op_kind: LessEqualElementwise
blurb: "Elementwise a <= b on f32 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::le_f32_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 4 + n"          # read 2x f32, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f32; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## le_f64_u8  (LessEqual, f64 -> Bool)

`a <= b` over f64, Bool mask out. `compare_kernel!(binary_cmp_le_f64, cmp_le_f64, 8, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 8 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: le_f64_u8
op_kind: LessEqualElementwise
blurb: "Elementwise a <= b on f64 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::le_f64_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 8 + n"          # read 2x f64, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f64; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## le_f16_u8  (LessEqual, f16 -> Bool)

`a <= b` over f16, Bool mask out. `compare_kernel!(binary_cmp_le_f16, cmp_le_f16, 2, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 2 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: le_f16_u8
op_kind: LessEqualElementwise
blurb: "Elementwise a <= b on f16 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::le_f16_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 2 + n"          # read 2x f16, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f16; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## le_bf16_u8  (LessEqual, bf16 -> Bool)

`a <= b` over bf16, Bool mask out. `compare_kernel!(binary_cmp_le_bf16, cmp_le_bf16, 2, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 2 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: le_bf16_u8
op_kind: LessEqualElementwise
blurb: "Elementwise a <= b on bf16 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::le_bf16_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 2 + n"          # read 2x bf16, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on bf16; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## gt_f32_u8  (Greater, f32 -> Bool)

`a > b` over f32, Bool mask out. `compare_kernel!(binary_cmp_gt_f32, cmp_gt_f32, 4, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 4 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: gt_f32_u8
op_kind: GreaterElementwise
blurb: "Elementwise a > b on f32 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::gt_f32_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 4 + n"          # read 2x f32, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f32; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## gt_f64_u8  (Greater, f64 -> Bool)

`a > b` over f64, Bool mask out. `compare_kernel!(binary_cmp_gt_f64, cmp_gt_f64, 8, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 8 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: gt_f64_u8
op_kind: GreaterElementwise
blurb: "Elementwise a > b on f64 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::gt_f64_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 8 + n"          # read 2x f64, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f64; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## gt_f16_u8  (Greater, f16 -> Bool)

`a > b` over f16, Bool mask out. `compare_kernel!(binary_cmp_gt_f16, cmp_gt_f16, 2, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 2 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: gt_f16_u8
op_kind: GreaterElementwise
blurb: "Elementwise a > b on f16 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::gt_f16_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 2 + n"          # read 2x f16, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f16; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## gt_bf16_u8  (Greater, bf16 -> Bool)

`a > b` over bf16, Bool mask out. `compare_kernel!(binary_cmp_gt_bf16, cmp_gt_bf16, 2, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 2 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: gt_bf16_u8
op_kind: GreaterElementwise
blurb: "Elementwise a > b on bf16 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::gt_bf16_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 2 + n"          # read 2x bf16, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on bf16; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## ge_f32_u8  (GreaterEqual, f32 -> Bool)

`a >= b` over f32, Bool mask out. `compare_kernel!(binary_cmp_ge_f32, cmp_ge_f32, 4, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 4 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: ge_f32_u8
op_kind: GreaterEqualElementwise
blurb: "Elementwise a >= b on f32 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::ge_f32_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 4 + n"          # read 2x f32, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f32; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## ge_f64_u8  (GreaterEqual, f64 -> Bool)

`a >= b` over f64, Bool mask out. `compare_kernel!(binary_cmp_ge_f64, cmp_ge_f64, 8, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 8 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: ge_f64_u8
op_kind: GreaterEqualElementwise
blurb: "Elementwise a >= b on f64 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::ge_f64_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 8 + n"          # read 2x f64, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f64; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## ge_f16_u8  (GreaterEqual, f16 -> Bool)

`a >= b` over f16, Bool mask out. `compare_kernel!(binary_cmp_ge_f16, cmp_ge_f16, 2, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 2 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: ge_f16_u8
op_kind: GreaterEqualElementwise
blurb: "Elementwise a >= b on f16 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::ge_f16_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 2 + n"          # read 2x f16, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on f16; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

## ge_bf16_u8  (GreaterEqual, bf16 -> Bool)

`a >= b` over bf16, Bool mask out. `compare_kernel!(binary_cmp_ge_bf16, cmp_ge_bf16, 2, ...)`
-> `binary_run{,_into}` with an output element width of **1** (inputs are 2 bytes; conflating the
two widths is what would size the output buffer wrongly). NaN: ordered comparison with NaN -> 0.

```fkc
kernel: ge_bf16_u8
op_kind: GreaterEqualElementwise
blurb: "Elementwise a >= b on bf16 (CUDA/baracuda); Bool mask out; NaN -> 0."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::ge_bf16_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BOOL)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * 2 + n"          # read 2x bf16, write n Bool
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact IEEE-754 ordered comparison on bf16; ordered comparison with NaN -> 0; kernel writes only 0u8/1u8."

determinism: bitwise
```

---

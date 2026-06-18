---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                                   # maps to BackendId::Cpu
  kernel_source: "portable-cpu"                  # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"                  # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — affine / clamp / powi kernel contracts

Out-of-place scalar-param elementwise kernels from the portable byte-kernel surface
(`fuel-cpu-backend/src/byte_kernels.rs`): the affine transform `mul·x + add`, the bounded
clamp `clamp(x, min, max)`, the integer power `x.powi(exp)`, and the powi backward
`exp·x^(exp-1)·upstream`. Each is a contiguous-only, zero-offset, row-major positional walk
over flat `CpuStorageBytes` slices that validates *byte length* against the declared shape
(`check_lens_2`) and **fully overwrites** a pre-allocated output buffer. Half floats (`BF16`,
`F16`) widen to `f32`, compute, and narrow on store; `f32`/`f64` compute natively. None of these
kernels consults a `Layout`/strides/offset — the pipelined executor's auto-Contiguize pass
realizes any strided/broadcast/offset input into a dense buffer first, so every operand's layout
contract is `contiguous, offset 0`. Validation returns `Result` and never panics on the
production path.

This bundle does not include the **in-place** affine/clamp/powi family
(`affine_inplace_*` / `clamp_inplace_*` / `powi_inplace_*`, `OpKind::InplaceAffine` /
`ClampInplace` / `PowIInplace`); those are a separate surface (`caps.in_place: true`,
`aliasing: in_place`) and are contracted elsewhere.

## affine_f32  (y = mul·x + add, f32)

Element-wise affine transformation `out[i] = mul * input[i] + add`, native f32 arithmetic. One
kernel covers both `Op::AddScalar(c)` (lowered as `mul=1, add=c`) and `Op::MulScalar(c)` (lowered
as `mul=c, add=0`), so the planner sees a single contract for the whole scalar-affine family. The
scalar params `(mul, add)` arrive on `OpParams::Affine { mul: f64, add: f64 }` and are consumed at
`f32` for this dtype. Positional walk over a contiguous, zero-offset, row-major buffer; validates
`input.len_bytes() == out.len_bytes()` (`check_lens_2`). Pre-allocated output, full overwrite, no
aliasing. Deterministic and bit-stable on the same hardware (a fused-multiply-add is not used; the
loop is a literal `mul * x + add`). Known limitation: contiguous-only — any strided/broadcast/
offset operand must be contiguized by the planner first.

```fkc
kernel: affine_f32
op_kind: Affine
blurb: "Elementwise affine y = mul*x + add (f32, native); covers AddScalar/MulScalar; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::affine_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Affine                 # OpParams::Affine { mul: f64, add: f64 }
    fields:
      mul: { kind: f64, note: "consumed at f32 for this dtype" }
      add: { kind: f64, note: "consumed at f32 for this dtype" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous    # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured        # Judge bootstraps; bandwidth-bound elementwise, hint below
  class: cheap_elementwise
  flops: "2 * n"                    # one multiply + one add per element
  bytes_moved: "2 * n * dtype_bytes"   # read input, write out
  overhead_ns: ~                    # judge_measured
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                    # CPU primitive: importer applies PRIMITIVE_DETERMINISTIC_CPU family default (§12.4)
  notes: "native f32 mul-then-add; deterministic positional loop; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

## affine_f64  (y = mul·x + add, f64)

Element-wise affine `out[i] = mul * input[i] + add` in native f64. Identical structure and
semantics to `affine_f32` but the params and arithmetic are full `f64` (`OpParams::Affine` carries
`mul: f64, add: f64` directly, no narrowing). Covers `AddScalar`/`MulScalar`. Contiguous,
zero-offset, byte-length-validated, full overwrite, no aliasing. Bit-stable on the same hardware.

```fkc
kernel: affine_f64
op_kind: Affine
blurb: "Elementwise affine y = mul*x + add (f64, native); covers AddScalar/MulScalar; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::affine_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Affine                 # OpParams::Affine { mul: f64, add: f64 }
    fields:
      mul: { kind: f64 }
      add: { kind: f64 }

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
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"                    # one multiply + one add per element
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64 mul-then-add; deterministic positional loop; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

## affine_bf16  (y = mul·x + add, bf16 via f32)

Element-wise affine for `BF16`. The scalar params arrive on `OpParams::Affine { mul: f64, add: f64 }`
but are taken at **f32** at this kernel's ABI (`affine_half_kernel!` signature is `mul: f32, add: f32`);
each element widens to f32, computes `mul·x + add`, and narrows back to bf16 on store. This is the
load-bearing precision invariant of the half family. Contiguous, zero-offset, byte-length-validated,
full overwrite, no aliasing. Bit-stable on the same hardware (the f32 round-trip is deterministic).

```fkc
kernel: affine_bf16
op_kind: Affine
blurb: "Elementwise affine y = mul*x + add (bf16; widen to f32, narrow on store); contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::affine_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Affine                 # OpParams::Affine { mul: f64, add: f64 }; consumed at f32
    fields:
      mul: { kind: f64, note: "narrowed to f32 at the kernel ABI" }
      add: { kind: f64, note: "narrowed to f32 at the kernel ABI" }

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
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"                    # multiply + add per element (computed in f32)
  bytes_moved: "2 * n * dtype_bytes"   # dtype_bytes = 2 (bf16)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen to f32, mul-then-add in f32, narrow to bf16 on store; deterministic; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## affine_f16  (y = mul·x + add, f16 via f32)

Element-wise affine for `F16`. Identical to `affine_bf16` but the storage dtype is `F16`: each
element widens to f32, computes `mul·x + add` (params taken at f32), narrows back to f16 on store.
Contiguous, zero-offset, byte-length-validated, full overwrite, no aliasing. Bit-stable on the same
hardware.

```fkc
kernel: affine_f16
op_kind: Affine
blurb: "Elementwise affine y = mul*x + add (f16; widen to f32, narrow on store); contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::affine_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Affine                 # OpParams::Affine { mul: f64, add: f64 }; consumed at f32
    fields:
      mul: { kind: f64, note: "narrowed to f32 at the kernel ABI" }
      add: { kind: f64, note: "narrowed to f32 at the kernel ABI" }

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
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"
  bytes_moved: "2 * n * dtype_bytes"   # dtype_bytes = 2 (f16)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen to f32, mul-then-add in f32, narrow to f16 on store; deterministic; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## clamp_f32  (y = clamp(x, min, max), f32)

Element-wise bounded clamp `out[i] = input[i].clamp(min, max)` in native f32. The scalar bounds
arrive on `OpParams::Clamp { min: f64, max: f64 }` (taken at f32 for this dtype). The kernel
**rejects `min > max`** with a typed `Error` (build/runtime validation, never a panic) before
touching the buffer. Uses Rust `f32::clamp`, which propagates NaN per IEEE (a NaN input clamps to
NaN). Positional walk over a contiguous, zero-offset, row-major buffer; validates
`input.len_bytes() == out.len_bytes()`. Pre-allocated output, full overwrite, no aliasing. Bit-stable
on the same hardware. Contiguous-only.

```fkc
kernel: clamp_f32
op_kind: ClampElementwise
blurb: "Elementwise clamp(x, min, max) (f32); rejects min>max; IEEE NaN; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::clamp_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Clamp                  # OpParams::Clamp { min: f64, max: f64 }
    fields:
      min: { kind: f64, constraint: "min <= max", note: "consumed at f32; min>max rejected" }
      max: { kind: f64 }

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
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                        # two compares (min/max) per element; ~1 op/element
  bytes_moved: "2 * n * dtype_bytes"   # read input, write out
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32::clamp; exact (no rounding); IEEE NaN propagates; min>max is a typed Error, not a panic."

determinism: same_hardware_bitwise
```

## clamp_f64  (y = clamp(x, min, max), f64)

Element-wise clamp in native f64. Identical structure and semantics to `clamp_f32` but params and
arithmetic are full f64 (`OpParams::Clamp` carries `min: f64, max: f64` directly). Rejects
`min > max` with a typed `Error`. `f64::clamp`, IEEE NaN. Contiguous, byte-length-validated, full
overwrite, no aliasing. Bit-stable on the same hardware.

```fkc
kernel: clamp_f64
op_kind: ClampElementwise
blurb: "Elementwise clamp(x, min, max) (f64); rejects min>max; IEEE NaN; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::clamp_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Clamp                  # OpParams::Clamp { min: f64, max: f64 }
    fields:
      min: { kind: f64, constraint: "min <= max" }
      max: { kind: f64 }

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
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f64::clamp; exact; IEEE NaN propagates; min>max is a typed Error, not a panic."

determinism: same_hardware_bitwise
```

## clamp_bf16  (y = clamp(x, min, max), bf16 via f32)

Element-wise clamp for `BF16`. The bounds arrive on `OpParams::Clamp { min: f64, max: f64 }` but are
taken at **f32** at this kernel's ABI (`clamp_half_kernel!` signature is `min: f32, max: f32`); each
element widens to f32, clamps to `[min, max]`, and narrows back to bf16 on store. Rejects
`min > max` (in f32 space) with a typed `Error`. IEEE NaN. Contiguous, byte-length-validated, full
overwrite, no aliasing. Bit-stable on the same hardware.

```fkc
kernel: clamp_bf16
op_kind: ClampElementwise
blurb: "Elementwise clamp(x, min, max) (bf16; widen to f32, narrow on store); rejects min>max; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::clamp_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Clamp                  # OpParams::Clamp { min: f64, max: f64 }; consumed at f32
    fields:
      min: { kind: f64, constraint: "min <= max", note: "narrowed to f32 at the kernel ABI" }
      max: { kind: f64, note: "narrowed to f32 at the kernel ABI" }

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
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # dtype_bytes = 2 (bf16)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen to f32, clamp in f32, narrow to bf16 on store; IEEE NaN; min>max is a typed Error; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## clamp_f16  (y = clamp(x, min, max), f16 via f32)

Element-wise clamp for `F16`. Identical to `clamp_bf16` but storage dtype is `F16`: widen to f32,
clamp, narrow to f16 on store; bounds taken at f32. Rejects `min > max` with a typed `Error`. IEEE
NaN. Contiguous, byte-length-validated, full overwrite, no aliasing. Bit-stable on the same hardware.

```fkc
kernel: clamp_f16
op_kind: ClampElementwise
blurb: "Elementwise clamp(x, min, max) (f16; widen to f32, narrow on store); rejects min>max; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::clamp_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Clamp                  # OpParams::Clamp { min: f64, max: f64 }; consumed at f32
    fields:
      min: { kind: f64, constraint: "min <= max", note: "narrowed to f32 at the kernel ABI" }
      max: { kind: f64, note: "narrowed to f32 at the kernel ABI" }

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
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # dtype_bytes = 2 (f16)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen to f32, clamp in f32, narrow to f16 on store; IEEE NaN; min>max is a typed Error; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## powi_f32  (y = x.powi(exp), f32)

Element-wise integer power `out[i] = input[i].powi(exp)` in native f32. The integer exponent
arrives on `OpParams::PowI { exp: i32 }`. Uses Rust `f32::powi`, which computes by repeated
multiplication / reciprocal for negative exponents (NOT `powf`), so the result is an exact-as-IEEE
product chain; `exp == 0` yields `1.0` for every input including `0.0`/NaN per `powi` semantics.
Positional walk over a contiguous, zero-offset, row-major buffer; validates
`input.len_bytes() == out.len_bytes()`. Pre-allocated output, full overwrite, no aliasing.
Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: powi_f32
op_kind: PowIElementwise
blurb: "Elementwise integer power x.powi(exp) (f32, native); contiguous; full overwrite."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::powi_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: PowI                   # OpParams::PowI { exp: i32 }
    fields:
      exp: { kind: i32 }

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
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured        # per-element cost grows with |exp| (powi mul chain) — Judge measures
  class: cheap_elementwise
  flops: ~                          # ~ O(log2(|exp|)) or O(|exp|) muls per element; judge_measured
  bytes_moved: "2 * n * dtype_bytes"   # read input, write out
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f32::powi (integer-exponent mul chain, not powf); exp==0 -> 1.0; deterministic; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## powi_f64  (y = x.powi(exp), f64)

Element-wise integer power in native f64. Identical structure and semantics to `powi_f32` but the
arithmetic is full f64 (`f64::powi`). Exponent on `OpParams::PowI { exp: i32 }`. Contiguous,
byte-length-validated, full overwrite, no aliasing. Bit-stable on the same hardware.

```fkc
kernel: powi_f64
op_kind: PowIElementwise
blurb: "Elementwise integer power x.powi(exp) (f64, native); contiguous; full overwrite."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::powi_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: PowI                   # OpParams::PowI { exp: i32 }
    fields:
      exp: { kind: i32 }

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
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                          # mul chain over |exp|; judge_measured
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64::powi (integer-exponent mul chain, not powf); exp==0 -> 1.0; deterministic; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## powi_bf16  (y = x.powi(exp), bf16 via f32)

Element-wise integer power for `BF16`. Each element widens to f32, computes `f32::powi(exp)`, and
narrows back to bf16 on store. Exponent on `OpParams::PowI { exp: i32 }`. Contiguous,
byte-length-validated, full overwrite, no aliasing. Bit-stable on the same hardware.

```fkc
kernel: powi_bf16
op_kind: PowIElementwise
blurb: "Elementwise integer power x.powi(exp) (bf16; widen to f32, narrow on store); contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::powi_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: PowI                   # OpParams::PowI { exp: i32 }
    fields:
      exp: { kind: i32 }

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
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                          # f32 mul chain over |exp|; judge_measured
  bytes_moved: "2 * n * dtype_bytes"   # dtype_bytes = 2 (bf16)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen to f32, f32::powi, narrow to bf16 on store; exp==0 -> 1.0; deterministic; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## powi_f16  (y = x.powi(exp), f16 via f32)

Element-wise integer power for `F16`. Identical to `powi_bf16` but storage dtype is `F16`: widen to
f32, `f32::powi(exp)`, narrow to f16 on store. Exponent on `OpParams::PowI { exp: i32 }`. Contiguous,
byte-length-validated, full overwrite, no aliasing. Bit-stable on the same hardware.

```fkc
kernel: powi_f16
op_kind: PowIElementwise
blurb: "Elementwise integer power x.powi(exp) (f16; widen to f32, narrow on store); contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::powi_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: PowI                   # OpParams::PowI { exp: i32 }
    fields:
      exp: { kind: i32 }

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
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                          # f32 mul chain over |exp|; judge_measured
  bytes_moved: "2 * n * dtype_bytes"   # dtype_bytes = 2 (f16)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen to f32, f32::powi, narrow to f16 on store; exp==0 -> 1.0; deterministic; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## powi_backward_f32  (grad_x = exp·x^(exp-1)·upstream, f32)

Backward of `powi`: `out[i] = exp * x[i]^(exp-1) * upstream[i]` in native f32. **Two inputs**
`(x, upstream)` — the forward input and the upstream gradient — and **one output**, the gradient with
respect to `x`. This is the single-launch alternative to the pre-`PowIElementwiseBackward` autograd
decomposition (`PowI(exp-1) → MulScalar(exp) → Mul`). The integer exponent arrives on the **same**
`OpParams::PowI { exp: i32 }` carrier as the forward (per `OpKind::PowIElementwiseBackward`,
`fuel-core-types/src/dispatch.rs`). Both inputs and the output are validated to the same byte length
(`check_lens_2` on `x` vs `out` and on `upstream` vs `out`). Inner term uses `f32::powi(exp-1)`.
Contiguous, zero-offset, row-major; pre-allocated output, full overwrite, no aliasing
(distinct buffer from both inputs). Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: powi_backward_f32
op_kind: PowIElementwiseBackward
blurb: "Backward of powi: grad_x = exp*x^(exp-1)*upstream (f32); two inputs (x, upstream); contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::powi_backward_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
    - name: upstream
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: PowI                   # OpParams::PowI { exp: i32 } — same carrier as the forward
    fields:
      exp: { kind: i32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured        # powi(exp-1) mul chain + 2 muls per element — Judge measures
  class: cheap_elementwise
  flops: ~                          # ~ powi(exp-1) chain + 2 muls per element; judge_measured
  bytes_moved: "3 * n * dtype_bytes"   # read x + upstream, write out
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f32: coeff(exp) * x.powi(exp-1) * upstream; deterministic; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## powi_backward_f64  (grad_x = exp·x^(exp-1)·upstream, f64)

Backward of `powi` in native f64. Identical structure to `powi_backward_f32` — two inputs
`(x, upstream)`, one output `grad_x`, exponent on `OpParams::PowI { exp: i32 }`, inner term
`f64::powi(exp-1)` — but all arithmetic is f64. Both inputs validated to the output byte length.
Contiguous, full overwrite, no aliasing. Bit-stable on the same hardware.

```fkc
kernel: powi_backward_f64
op_kind: PowIElementwiseBackward
blurb: "Backward of powi: grad_x = exp*x^(exp-1)*upstream (f64); two inputs (x, upstream); contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::powi_backward_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
    - name: upstream
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: PowI                   # OpParams::PowI { exp: i32 }
    fields:
      exp: { kind: i32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                          # powi(exp-1) chain + 2 muls per element; judge_measured
  bytes_moved: "3 * n * dtype_bytes"   # read x + upstream, write out
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64: coeff(exp) * x.powi(exp-1) * upstream; deterministic; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## powi_backward_bf16  (grad_x = exp·x^(exp-1)·upstream, bf16 via f32)

Backward of `powi` for `BF16`. Two inputs `(x, upstream)`, one output `grad_x`. Both inputs widen to
f32, the kernel computes `coeff(exp) * x.powi(exp-1) * upstream` entirely in f32, then narrows the
result to bf16 on store (the half precision invariant). Exponent on `OpParams::PowI { exp: i32 }`.
Both inputs validated to the output byte length. Contiguous, full overwrite, no aliasing. Bit-stable
on the same hardware.

```fkc
kernel: powi_backward_bf16
op_kind: PowIElementwiseBackward
blurb: "Backward of powi: grad_x = exp*x^(exp-1)*upstream (bf16; compute in f32, narrow on store); contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::powi_backward_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
    - name: upstream
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: PowI                   # OpParams::PowI { exp: i32 }
    fields:
      exp: { kind: i32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                          # f32 powi(exp-1) chain + 2 muls per element; judge_measured
  bytes_moved: "3 * n * dtype_bytes"   # read x + upstream, write out; dtype_bytes = 2 (bf16)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen x+upstream to f32, coeff*x.powi(exp-1)*upstream in f32, narrow to bf16 on store; deterministic; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## powi_backward_f16  (grad_x = exp·x^(exp-1)·upstream, f16 via f32)

Backward of `powi` for `F16`. Identical to `powi_backward_bf16` but storage dtype is `F16`: both
inputs widen to f32, compute `coeff(exp) * x.powi(exp-1) * upstream` in f32, narrow to f16 on store.
Two inputs `(x, upstream)`, one output `grad_x`, exponent on `OpParams::PowI { exp: i32 }`. Both
inputs validated to the output byte length. Contiguous, full overwrite, no aliasing. Bit-stable on
the same hardware.

```fkc
kernel: powi_backward_f16
op_kind: PowIElementwiseBackward
blurb: "Backward of powi: grad_x = exp*x^(exp-1)*upstream (f16; compute in f32, narrow on store); contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::powi_backward_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
    - name: upstream
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: PowI                   # OpParams::PowI { exp: i32 }
    fields:
      exp: { kind: i32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                          # f32 powi(exp-1) chain + 2 muls per element; judge_measured
  bytes_moved: "3 * n * dtype_bytes"   # read x + upstream, write out; dtype_bytes = 2 (f16)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen x+upstream to f32, coeff*x.powi(exp-1)*upstream in f32, narrow to f16 on store; deterministic; bit-stable same hardware."

determinism: same_hardware_bitwise
```

---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                       # maps to BackendId::Cpu
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — in-place unary / affine / clamp / powi kernel contracts

Portable CPU byte-kernels from the `CpuStorageBytes` surface
(`fuel-cpu-backend/src/byte_kernels.rs`) for the **in-place scalar-param** op family:
the in-place unary family (`unary_inplace_thunk!`), and the in-place affine / clamp / powi
kernels. Every kernel in this bundle mutates a **single** buffer in place — the output IS the
input Storage (`&mut CpuStorageBytes`, `out[i] = op(out[i])`) — over contiguous, zero-offset,
row-major bytes. None consults a `Layout`/strides/offset; the executor's auto-Contiguize pass
realizes any strided/broadcast/offset input into a contiguous buffer before these kernels run.
Half floats (`bf16`/`f16`) widen to **f32**, do the math, narrow on store, matching the
non-in-place cousins exactly (the in-place unary thunks literally reuse the chassis
`UnaryOp<T>::apply`). Cost is marked `judge_measured` for every kernel — the Judge bootstraps
the coefficients; the bandwidth-bound elementwise FLOPs/bytes formula hints are recorded as the
only genuinely op-derivable structure.

## unary_inplace  (Relu / Silu / Gelu / Tanh / Sigmoid / Neg / Abs / Sqr / Sqrt / Rsqrt / Recip / Exp / Log / Sin / Cos / Sign / Floor / Ceil / Round / Erf / GeluErf)

In-place element-wise unary: `out[i] = op(out[i])` over a single contiguous, zero-offset,
row-major buffer (`&mut CpuStorageBytes`, `unary_inplace_thunk!`, `byte_kernels.rs:2822`). One
logical kernel selected per `OpKind`; 21 ops share this contract (Relu, Silu, Gelu(=GeluTanh,
the in-place `gelu_*` thunk maps to `GeluTanh`), Tanh, Sigmoid, Neg, Abs, Sqr, Sqrt, Rsqrt,
Recip, Exp, Log, Sin, Cos, Sign, Floor, Ceil, Round, Erf, GeluErf). Each thunk delegates to
`chassis::unary::UnaryOp<T>::apply`, so the numerics are **bit-identical to the non-in-place
cousins** (Erf/GeluErf via `libm::erf{,f}`; GeluTanh uses a 7-digit √(2/π) const for f32 and
16-digit for f64; Sign(0)=0; Round = `round_ties_even` banker's rounding; half via f32). f32/f64
evaluate natively; bf16/f16 widen to f32, apply, narrow on store. The buffer is fully overwritten
positionally — no broadcasting, no read of an out-of-line input. Known limitation: contiguous-only
— any strided/broadcast/offset operand must be contiguized by the planner first; the kernel reads
no `Layout`. Element count is carried implicitly by the buffer byte length (validated against
`dtype` width by `as_slice_mut`).

```fkc
kernel: unary_inplace
op_kind: ReluInplace          # representative; one contract per in-place unary OpKind
                              # (Silu/GeluTanh/Tanh/Sigmoid/Neg/Abs/Sqr/Sqrt/Rsqrt/Recip/Exp/
                              #  Log/Sin/Cos/Sign/Floor/Ceil/Round/Erf/GeluErf — same shape)
blurb: "In-place elementwise unary out[i]=op(out[i]); single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::relu_inplace_f32"   # one per (op,dtype); §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: out                 # the SINGLE in-place buffer: read-modify-write
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }  # OpParams::None — no auxiliary scalar params

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)     # in-place: output dtype == input dtype
      shape_rule: same_as(out)         # shape preserved; symbolic extents carry through
      layout_guarantee: contiguous
      aliasing: in_place(out)          # output IS the input buffer (caps.in_place: true)

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true                # mutates its single buffer (§4.6)
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured    # the Judge bootstraps/calibrates the coefficients (§4.4)
  class: cheap_elementwise
  # FLOPs/bandwidth hint (op-derivable): elementwise unary is bandwidth-bound —
  # ~n element ops, read+write the single buffer (1 read + 1 write).
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read out, write out (in-place)
  overhead_ns: ~                # judge_measured
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # in-place: no new alloc

precision:
  bit_stable_on_same_hardware: true    # deterministic positional loop; reuses non-in-place numerics
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                # CPU primitive: importer applies PRIMITIVE_DETERMINISTIC_CPU (§12.4)
  notes: "Reuses chassis UnaryOp<T>::apply — numerics identical to the non-in-place unary kernel. f32/f64 native; bf16/f16 widen to f32 then narrow. Erf/GeluErf via libm; Round ties-to-even; Sign(0)=0."

determinism: same_hardware_bitwise
```

## affine_inplace_f32  (out = mul*out + add)

In-place affine over a single contiguous f32 buffer: `out[i] = mul * out[i] + add`
(`affine_inplace_f32`, `byte_kernels.rs:2946`). `mul`/`add` are native f32 scalar params from
`OpParams::Affine`. Covers in-place AddScalar (`mul == 1`) and MulScalar (`add == 0`). Full
positional overwrite of the single buffer; no separate input. Native f32 arithmetic (one FMA-style
multiply-add per element), IEEE inf/NaN propagation. Contiguous-only; the executor contiguizes
strided/offset inputs first.

```fkc
kernel: affine_inplace_f32
op_kind: Affine               # OpParams::Affine signals the op family; in-place dispatch arm
blurb: "In-place affine out[i]=mul*out[i]+add over a single contiguous f32 buffer; native f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::affine_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Affine           # OpParams::Affine { mul: f64, add: f64 }
    fields:
      mul: { kind: f64, note: "f32 kernel consumes the f64 param directly as f32" }
      add: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"                # one multiply + one add per element
  bytes_moved: "2 * n * dtype_bytes"   # read out, write out (in-place)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                # CPU primitive: PRIMITIVE_DETERMINISTIC_CPU default (§12.4)
  notes: "Native f32 mul-add; IEEE inf/NaN. mul==1 ⇒ AddScalar; add==0 ⇒ MulScalar."

determinism: same_hardware_bitwise
```

## affine_inplace_f64  (out = mul*out + add)

In-place affine over a single contiguous f64 buffer: `out[i] = mul * out[i] + add`
(`affine_inplace_f64`, `byte_kernels.rs:2955`). Same shape as `affine_inplace_f32` with native f64
`mul`/`add` from `OpParams::Affine`. Full positional overwrite; IEEE inf/NaN. Contiguous-only.

```fkc
kernel: affine_inplace_f64
op_kind: Affine
blurb: "In-place affine out[i]=mul*out[i]+add over a single contiguous f64 buffer; native f64."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::affine_inplace_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Affine
    fields:
      mul: { kind: f64 }
      add: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Native f64 mul-add; IEEE inf/NaN. mul==1 ⇒ AddScalar; add==0 ⇒ MulScalar."

determinism: same_hardware_bitwise
```

## affine_inplace_bf16  (out = mul*out + add, half via f32)

In-place affine over a single contiguous bf16 buffer: `out[i] = bf16(mul_f32 * out[i].to_f32() +
add_f32)` (`affine_inplace_bf16`, `byte_kernels.rs:2965`). The `mul`/`add` params arrive as **f64**
(`OpParams::Affine`) and are narrowed to f32; each element widens to f32, the mul-add runs in f32,
the result narrows back to bf16 on store — matching the non-in-place `affine_*` half-precision
convention. Contiguous-only.

```fkc
kernel: affine_inplace_bf16
op_kind: Affine
blurb: "In-place affine out=mul*out+add over a single contiguous bf16 buffer; params f64→f32, math in f32, narrow on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::affine_inplace_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Affine
    fields:
      mul: { kind: f64, note: "param arrives f64, narrowed to f32 for the half compute" }
      add: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16 widens to f32, mul-add in f32 (params f64→f32), narrows to bf16 on store. Matches the non-in-place affine_* half convention."

determinism: same_hardware_bitwise
```

## affine_inplace_f16  (out = mul*out + add, half via f32)

In-place affine over a single contiguous f16 buffer: `out[i] = f16(mul_f32 * out[i].to_f32() +
add_f32)` (`affine_inplace_f16`, `byte_kernels.rs:2977`). Identical structure to
`affine_inplace_bf16` with the f16 narrow on store; `mul`/`add` arrive f64 and narrow to f32.
Contiguous-only.

```fkc
kernel: affine_inplace_f16
op_kind: Affine
blurb: "In-place affine out=mul*out+add over a single contiguous f16 buffer; params f64→f32, math in f32, narrow on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::affine_inplace_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Affine
    fields:
      mul: { kind: f64, note: "param arrives f64, narrowed to f32 for the half compute" }
      add: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f16 widens to f32, mul-add in f32 (params f64→f32), narrows to f16 on store. Matches the non-in-place affine_* half convention."

determinism: same_hardware_bitwise
```

## clamp_inplace_f32  (out = clamp(out, min, max))

In-place clamp over a single contiguous f32 buffer: `out[i] = out[i].clamp(min, max)`
(`clamp_inplace_f32`, `byte_kernels.rs:2992`). `min`/`max` are native f32 from `OpParams::Clamp`.
**Rejects `min > max`** at build/runtime with a `Result` error (never panics) — a hard precondition
on the params, not a clamp-flip. `f32::clamp` semantics for IEEE values. Full positional overwrite;
contiguous-only.

```fkc
kernel: clamp_inplace_f32
op_kind: ClampInplace         # OpParams::Clamp signals the op family; in-place dispatch arm
blurb: "In-place clamp out[i]=clamp(out[i],min,max) over a single contiguous f32 buffer; rejects min>max."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::clamp_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Clamp            # OpParams::Clamp { min: f64, max: f64 }
    fields:
      min: { kind: f64, constraint: "min <= max", note: "min > max returns a typed Error (no panic)" }
      max: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                   # one clamp (two compares) per element
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Native f32 clamp. min>max ⇒ typed Error (no panic). Exact: clamp returns one of the operands or a bound."

determinism: same_hardware_bitwise
```

## clamp_inplace_f64  (out = clamp(out, min, max))

In-place clamp over a single contiguous f64 buffer: `out[i] = out[i].clamp(min, max)`
(`clamp_inplace_f64`, `byte_kernels.rs:3001`). Native f64 `min`/`max` from `OpParams::Clamp`;
**rejects `min > max`** with a `Result` error. Full positional overwrite; contiguous-only.

```fkc
kernel: clamp_inplace_f64
op_kind: ClampInplace
blurb: "In-place clamp out[i]=clamp(out[i],min,max) over a single contiguous f64 buffer; rejects min>max."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::clamp_inplace_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Clamp
    fields:
      min: { kind: f64, constraint: "min <= max", note: "min > max returns a typed Error (no panic)" }
      max: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Native f64 clamp. min>max ⇒ typed Error (no panic)."

determinism: same_hardware_bitwise
```

## clamp_inplace_bf16  (out = clamp(out, min, max), half via f32)

In-place clamp over a single contiguous bf16 buffer: `out[i] = bf16(out[i].to_f32().clamp(min_f32,
max_f32))` (`clamp_inplace_bf16`, `byte_kernels.rs:3012`). `min`/`max` arrive as **f64**
(`OpParams::Clamp`), narrow to f32; each element widens to f32, clamps in f32, narrows to bf16 on
store. **Rejects `min > max`** with a `Result` error. Contiguous-only.

```fkc
kernel: clamp_inplace_bf16
op_kind: ClampInplace
blurb: "In-place clamp over a single contiguous bf16 buffer; params f64→f32, clamp in f32, narrow on store; rejects min>max."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::clamp_inplace_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Clamp
    fields:
      min: { kind: f64, constraint: "min <= max", note: "min > max returns a typed Error; param narrowed to f32 for the half compute" }
      max: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16 widens to f32, clamps against f32-narrowed bounds, narrows on store. min>max ⇒ typed Error (no panic)."

determinism: same_hardware_bitwise
```

## clamp_inplace_f16  (out = clamp(out, min, max), half via f32)

In-place clamp over a single contiguous f16 buffer: `out[i] = f16(out[i].to_f32().clamp(min_f32,
max_f32))` (`clamp_inplace_f16`, `byte_kernels.rs:3024`). Identical structure to
`clamp_inplace_bf16` with the f16 narrow on store; `min`/`max` arrive f64 and narrow to f32.
**Rejects `min > max`** with a `Result` error. Contiguous-only.

```fkc
kernel: clamp_inplace_f16
op_kind: ClampInplace
blurb: "In-place clamp over a single contiguous f16 buffer; params f64→f32, clamp in f32, narrow on store; rejects min>max."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::clamp_inplace_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Clamp
    fields:
      min: { kind: f64, constraint: "min <= max", note: "min > max returns a typed Error; param narrowed to f32 for the half compute" }
      max: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f16 widens to f32, clamps against f32-narrowed bounds, narrows on store. min>max ⇒ typed Error (no panic)."

determinism: same_hardware_bitwise
```

## powi_inplace_f32  (out = out.powi(exp))

In-place integer power over a single contiguous f32 buffer: `out[i] = out[i].powi(exp)`
(`powi_inplace_f32`, `byte_kernels.rs:3036`). `exp` is an `i32` from `OpParams::PowI`; native f32
`powi` (repeated multiplication semantics, including negative exponents → reciprocal, exp==0 → 1.0).
Full positional overwrite; contiguous-only.

```fkc
kernel: powi_inplace_f32
op_kind: PowIInplace          # OpParams::PowI signals the op family; in-place dispatch arm
blurb: "In-place integer power out[i]=out[i].powi(exp) over a single contiguous f32 buffer; native f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::powi_inplace_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: PowI             # OpParams::PowI { exp: i32 }
    fields:
      exp: { kind: i32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  # powi cost scales with the exponent's bit-length (square-and-multiply); the
  # per-element op count is not a fixed constant, so only n (element count) is given.
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Native f32 powi (i32 exponent). exp==0 ⇒ 1.0; negative exp ⇒ reciprocal; IEEE inf/NaN."

determinism: same_hardware_bitwise
```

## powi_inplace_f64  (out = out.powi(exp))

In-place integer power over a single contiguous f64 buffer: `out[i] = out[i].powi(exp)`
(`powi_inplace_f64`, `byte_kernels.rs:3042`). Native f64 `powi`; `exp` is an `i32` from
`OpParams::PowI`. Full positional overwrite; contiguous-only.

```fkc
kernel: powi_inplace_f64
op_kind: PowIInplace
blurb: "In-place integer power out[i]=out[i].powi(exp) over a single contiguous f64 buffer; native f64."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::powi_inplace_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: PowI
    fields:
      exp: { kind: i32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Native f64 powi (i32 exponent). exp==0 ⇒ 1.0; negative exp ⇒ reciprocal; IEEE inf/NaN."

determinism: same_hardware_bitwise
```

## powi_inplace_bf16  (out = out.powi(exp), half via f32)

In-place integer power over a single contiguous bf16 buffer: `out[i] =
bf16(out[i].to_f32().powi(exp))` (`powi_inplace_bf16`, `byte_kernels.rs:3048`). Each element widens
to f32, raises to the `i32` power in f32, narrows to bf16 on store — matching the half-precision
convention. `exp` from `OpParams::PowI`. Contiguous-only.

```fkc
kernel: powi_inplace_bf16
op_kind: PowIInplace
blurb: "In-place integer power over a single contiguous bf16 buffer; widen to f32, powi, narrow on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::powi_inplace_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: PowI
    fields:
      exp: { kind: i32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16 widens to f32, powi in f32 (i32 exponent), narrows to bf16 on store. exp==0 ⇒ 1.0; negative exp ⇒ reciprocal."

determinism: same_hardware_bitwise
```

## powi_inplace_f16  (out = out.powi(exp), half via f32)

In-place integer power over a single contiguous f16 buffer: `out[i] =
f16(out[i].to_f32().powi(exp))` (`powi_inplace_f16`, `byte_kernels.rs:3056`). Identical structure to
`powi_inplace_bf16` with the f16 narrow on store; `exp` from `OpParams::PowI`. Contiguous-only.

```fkc
kernel: powi_inplace_f16
op_kind: PowIInplace
blurb: "In-place integer power over a single contiguous f16 buffer; widen to f32, powi, narrow on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::powi_inplace_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: PowI
    fields:
      exp: { kind: i32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(out)
      shape_rule: same_as(out)
      layout_guarantee: contiguous
      aliasing: in_place(out)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f16 widens to f32, powi in f32 (i32 exponent), narrows to f16 on store. exp==0 ⇒ 1.0; negative exp ⇒ reciprocal."

determinism: same_hardware_bitwise
```

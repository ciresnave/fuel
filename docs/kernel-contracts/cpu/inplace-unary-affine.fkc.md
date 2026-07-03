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

**Re-author note (2026-07-03).** The in-place unary family was originally authored as a SINGLE
`unary_inplace` **representative** section (`op_kind: ReluInplace`) standing in for 21 distinct
`<Op>Inplace` OpKinds — one section could not register 21 different OpKinds. Now that the importer's
§3.4 per-dtype fan-out is shipped, the representative is SUPERSEDED by **21 per-op sections** (one
per OpKind: `ReluInplace`, `SiluInplace`, `GeluInplace`, `TanhInplace`, `SigmoidInplace`,
`NegInplace`, `AbsInplace`, `SqrInplace`, `SqrtInplace`, `RsqrtInplace`, `RecipInplace`,
`ExpInplace`, `LogInplace`, `SinInplace`, `CosInplace`, `SignInplace`, `FloorInplace`,
`CeilInplace`, `RoundInplace`, `ErfInplace`, `GeluErfInplace`), each fanning `[F32, F64, BF16, F16]`.
The old `unary_inplace` section is retained below, converted to a **`registrable: false`**
describe-only umbrella (§3.10) documenting the shared chassis. The four `affine_inplace_<dt>`
sections' `op_kind` was corrected from `Affine` (the OUT-of-place op — importing it there would
silently pile the in-place wrappers onto the out-of-place key and leave `InplaceAffine` unbound) to
the correct **`InplaceAffine`**. The eight clamp/powi sections were already correctly specified
per-dtype (`ClampInplace` / `PowIInplace`) and are unchanged.

**Entry-point fan-out (§3.4).** Each per-op unary section declares a **base** `entry_point`
(`fuel_cpu_backend::byte_kernels::<op>_inplace`, e.g. `…::relu_inplace`) and enumerates
`dtypes: [F32, F64, BF16, F16]`. The FKC importer fans the section into one binding per enumerated
dtype, resolving `<base>_<dtype>` (`relu_inplace_f32` … `relu_inplace_f16`) through the CPU link
registry — so a single section registers all four dtype thunks. The affine / clamp / powi sections
are instead **per-dtype single sections** (one enumerated dtype each), so they do NOT fan — their
specific `<op>_inplace_<dt>` `entry_point` resolves AS-IS. Every section binds the key **`[T, T]`**
(the single `out` operand + its `passthrough(out)` mirror; the executor's `WorkItemKind::InplaceKernel`
arm passes the target as `outputs[0]`, so the production wrapper takes 0 inputs + 1 output, but the
binding-table KEY is what lookup matches). Scalar params (affine `mul`/`add`, clamp `min`/`max`, powi
`exp`) ride in `OpParams::{Affine, Clamp, PowI}`, NOT the dtype-list. The `aliasing: in_place(out)`
return-contract is retained metadata (§4.6), NOT a key slot — the hand-written path keyed `[T, T]`
with empty caps and this reproduces it byte-for-byte.

## unary_inplace  (shared in-place elementwise-unary chassis — describe-only umbrella)

Shared in-place element-wise unary chassis: `out[i] = op(out[i])` over a single contiguous,
zero-offset, row-major buffer (`&mut CpuStorageBytes`, `unary_inplace_thunk!`,
`byte_kernels.rs:2822`). This umbrella documents the shape/loop/precision contract that every
concrete in-place unary op below specializes; it binds **no** `OpKind` of its own (each named op
pins one distinct `<Op>Inplace` OpKind), so it is **`registrable: false`** (§3.10 describe-only) and
registers no binding. 21 ops specialize it (Relu, Silu, Gelu(=GeluTanh, the in-place `gelu_*` thunk
maps to `GeluTanh`, `OpKind::GeluInplace`), Tanh, Sigmoid, Neg, Abs, Sqr, Sqrt, Rsqrt, Recip, Exp,
Log, Sin, Cos, Sign, Floor, Ceil, Round, Erf, GeluErf). Each thunk delegates to
`chassis::unary::UnaryOp<T>::apply`, so the numerics are **bit-identical to the non-in-place cousins**
(Erf/GeluErf via `libm::erf{,f}`; GeluTanh uses a 7-digit √(2/π) const for f32 and 16-digit for f64;
Sign(0)=0; Round = `round_ties_even` banker's rounding; half via f32). f32/f64 evaluate natively;
bf16/f16 widen to f32, apply, narrow on store. The buffer is fully overwritten positionally — no
broadcasting, no read of an out-of-line input. Known limitation: contiguous-only — any
strided/broadcast/offset operand must be contiguized by the planner first; the kernel reads no
`Layout`. Element count is carried implicitly by the buffer byte length (validated against `dtype`
width by `as_slice_mut`).

```fkc
kernel: unary_inplace
registrable: false            # §3.10 describe-only: shared chassis umbrella, NOT a dispatch target
op_kind: ~                    # the chassis itself binds no OpKind; each named op below pins one
fused_op: ~
blurb: "Shared in-place elementwise-unary chassis out[i]=op(out[i]); single contiguous buffer; half via f32; not separately dispatchable."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::unary_inplace_thunk"   # the generic in-place thunk generator; §12.6
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
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read out, write out (in-place)
  overhead_ns: ~                # judge_measured
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # in-place: no new alloc

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                # CPU primitive: importer applies PRIMITIVE_DETERMINISTIC_CPU (§12.4)
  notes: "Generic chassis; per-op numerics in the specialized sections. Reuses chassis UnaryOp<T>::apply. f32/f64 native; bf16/f16 widen to f32 then narrow."

determinism: same_hardware_bitwise
```

## relu_inplace  (out = max(0, out))

In-place ReLU clamp: `out[i] = max(0, out[i])` over a single contiguous buffer
(`relu_inplace_<dt>`). f32/f64 native; bf16/f16 widen to f32, clamp, narrow. NaN-as-missing
(`f32::max`). Numerics identical to the non-in-place `relu` cousin. Contiguous-only.

```fkc
kernel: relu_inplace
op_kind: ReluInplace
blurb: "In-place ReLU out[i]=max(0,out[i]); single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::relu_inplace"   # base; §3.4 fans relu_inplace_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "max(0, x); exact f32/f64. bf16/f16 widen to f32 then narrow. NaN-as-missing (f32::max). Identical to the non-in-place relu."

determinism: same_hardware_bitwise
```

## silu_inplace  (out = out * sigmoid(out))

In-place SiLU / swish: `out[i] = out[i] * sigmoid(out[i])` over a single contiguous buffer
(`silu_inplace_<dt>`). Transcendental. f32/f64 native; bf16/f16 via f32. Numerics identical to the
non-in-place `silu` cousin. Contiguous-only.

```fkc
kernel: silu_inplace
op_kind: SiluInplace
blurb: "In-place SiLU out[i]=out[i]*sigmoid(out[i]); single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::silu_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "x*sigmoid(x); transcendental. f32/f64 native; bf16/f16 via f32. Identical to the non-in-place silu."

determinism: same_hardware_bitwise
```

## gelu_inplace  (out = GeluTanh(out))

In-place GELU (tanh approximation, the canonical `OpKind::GeluInplace`):
`out[i] = 0.5·out[i]·(1 + tanh(√(2/π)·(out[i] + 0.044715·out[i]³)))` over a single contiguous buffer
(`gelu_inplace_<dt>`). 7-digit √(2/π) const for f32, 16-digit for f64; bf16/f16 via f32. Numerics
identical to the non-in-place `gelu` (GeluTanh) cousin. Distinct from `gelu_erf_inplace`. Contiguous-only.

```fkc
kernel: gelu_inplace
op_kind: GeluInplace
blurb: "In-place GELU (tanh approx) out[i]=0.5x(1+tanh(√(2/π)(x+0.044715x³))); single contiguous buffer; half via f32; identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::gelu_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "GeluTanh: 0.5x(1+tanh(√(2/π)(x+0.044715x³))); 7-digit √(2/π) for f32, 16-digit for f64; bf16/f16 via f32. OpKind GeluInplace (canonical tanh GELU), distinct from GeluErfInplace. Identical to the non-in-place gelu."

determinism: same_hardware_bitwise
```

## tanh_inplace  (out = tanh(out))

In-place hyperbolic tangent: `out[i] = tanh(out[i])` over a single contiguous buffer
(`tanh_inplace_<dt>`). Transcendental via libm/std. f32/f64 native; bf16/f16 via f32. Numerics
identical to the non-in-place `tanh` cousin. Contiguous-only.

```fkc
kernel: tanh_inplace
op_kind: TanhInplace
blurb: "In-place tanh out[i]=tanh(out[i]); single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::tanh_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "tanh(x); transcendental via libm/std. f32/f64 native; bf16/f16 via f32. Identical to the non-in-place tanh."

determinism: same_hardware_bitwise
```

## sigmoid_inplace  (out = 1/(1+exp(-out)))

In-place logistic sigmoid: `out[i] = 1/(1 + exp(-out[i]))` over a single contiguous buffer
(`sigmoid_inplace_<dt>`). Transcendental. f32/f64 native; bf16/f16 via f32. Numerics identical to the
non-in-place `sigmoid` cousin. Contiguous-only.

```fkc
kernel: sigmoid_inplace
op_kind: SigmoidInplace
blurb: "In-place sigmoid out[i]=1/(1+exp(-out[i])); single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sigmoid_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "1/(1+exp(-x)); transcendental. f32/f64 native; bf16/f16 via f32. Identical to the non-in-place sigmoid."

determinism: same_hardware_bitwise
```

## neg_inplace  (out = -out)

In-place negation: `out[i] = -out[i]` over a single contiguous buffer (`neg_inplace_<dt>`). Exact.
bf16/f16 via f32. Numerics identical to the non-in-place `neg` cousin. Contiguous-only.

```fkc
kernel: neg_inplace
op_kind: NegInplace
blurb: "In-place negate out[i]=-out[i]; single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::neg_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "-x; exact. bf16/f16 via f32. Identical to the non-in-place neg."

determinism: same_hardware_bitwise
```

## abs_inplace  (out = |out|)

In-place absolute value: `out[i] = |out[i]|` over a single contiguous buffer (`abs_inplace_<dt>`).
Exact. bf16/f16 via f32. Numerics identical to the non-in-place `abs` cousin. Contiguous-only.

```fkc
kernel: abs_inplace
op_kind: AbsInplace
blurb: "In-place abs out[i]=|out[i]|; single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::abs_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "|x|; exact. bf16/f16 via f32. Identical to the non-in-place abs."

determinism: same_hardware_bitwise
```

## sqr_inplace  (out = out * out)

In-place square: `out[i] = out[i] * out[i]` over a single contiguous buffer (`sqr_inplace_<dt>`).
Exact. bf16/f16 via f32. Numerics identical to the non-in-place `sqr` cousin. Contiguous-only.

```fkc
kernel: sqr_inplace
op_kind: SqrInplace
blurb: "In-place square out[i]=out[i]*out[i]; single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sqr_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "x*x; exact. bf16/f16 via f32. Identical to the non-in-place sqr."

determinism: same_hardware_bitwise
```

## sqrt_inplace  (out = √out)

In-place square root: `out[i] = √out[i]` over a single contiguous buffer (`sqrt_inplace_<dt>`).
f32/f64 native; bf16/f16 via f32. NaN for x<0. Numerics identical to the non-in-place `sqrt` cousin.
Contiguous-only.

```fkc
kernel: sqrt_inplace
op_kind: SqrtInplace
blurb: "In-place sqrt out[i]=√out[i]; single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sqrt_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "√x; f32/f64 native; bf16/f16 via f32. NaN for x<0. Identical to the non-in-place sqrt."

determinism: same_hardware_bitwise
```

## rsqrt_inplace  (out = 1/√out)

In-place reciprocal square root: `out[i] = 1/√out[i]` over a single contiguous buffer
(`rsqrt_inplace_<dt>`). f32/f64 native; bf16/f16 via f32. Numerics identical to the non-in-place
`rsqrt` cousin. Contiguous-only.

```fkc
kernel: rsqrt_inplace
op_kind: RsqrtInplace
blurb: "In-place rsqrt out[i]=1/√out[i]; single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rsqrt_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "1/√x; f32/f64 native; bf16/f16 via f32. Identical to the non-in-place rsqrt."

determinism: same_hardware_bitwise
```

## recip_inplace  (out = 1/out)

In-place reciprocal: `out[i] = 1/out[i]` over a single contiguous buffer (`recip_inplace_<dt>`).
Exact IEEE division; ±inf at 0. f32/f64 native; bf16/f16 via f32. Numerics identical to the
non-in-place `recip` cousin. Contiguous-only.

```fkc
kernel: recip_inplace
op_kind: RecipInplace
blurb: "In-place reciprocal out[i]=1/out[i]; single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::recip_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "1/x; exact IEEE division; ±inf at 0. bf16/f16 via f32. Identical to the non-in-place recip."

determinism: same_hardware_bitwise
```

## exp_inplace  (out = e^out)

In-place exponential: `out[i] = e^out[i]` over a single contiguous buffer (`exp_inplace_<dt>`).
Transcendental. f32/f64 native; bf16/f16 via f32. Numerics identical to the non-in-place `exp`
cousin. Contiguous-only.

```fkc
kernel: exp_inplace
op_kind: ExpInplace
blurb: "In-place exp out[i]=e^out[i]; single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::exp_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "e^x; transcendental. f32/f64 native; bf16/f16 via f32. Identical to the non-in-place exp."

determinism: same_hardware_bitwise
```

## log_inplace  (out = ln(out))

In-place natural log: `out[i] = ln(out[i])` over a single contiguous buffer (`log_inplace_<dt>`).
Transcendental. NaN for x<0, -inf at 0. f32/f64 native; bf16/f16 via f32. Numerics identical to the
non-in-place `log` cousin. Contiguous-only.

```fkc
kernel: log_inplace
op_kind: LogInplace
blurb: "In-place ln out[i]=ln(out[i]); single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::log_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "ln(x); transcendental. NaN for x<0, -inf at 0. f32/f64 native; bf16/f16 via f32. Identical to the non-in-place log."

determinism: same_hardware_bitwise
```

## sin_inplace  (out = sin(out))

In-place sine: `out[i] = sin(out[i])` over a single contiguous buffer (`sin_inplace_<dt>`).
Transcendental. f32/f64 native; bf16/f16 via f32. Numerics identical to the non-in-place `sin`
cousin. Contiguous-only.

```fkc
kernel: sin_inplace
op_kind: SinInplace
blurb: "In-place sin out[i]=sin(out[i]); single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sin_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "sin(x); transcendental. f32/f64 native; bf16/f16 via f32. Identical to the non-in-place sin."

determinism: same_hardware_bitwise
```

## cos_inplace  (out = cos(out))

In-place cosine: `out[i] = cos(out[i])` over a single contiguous buffer (`cos_inplace_<dt>`).
Transcendental. f32/f64 native; bf16/f16 via f32. Numerics identical to the non-in-place `cos`
cousin. Contiguous-only.

```fkc
kernel: cos_inplace
op_kind: CosInplace
blurb: "In-place cos out[i]=cos(out[i]); single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cos_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "cos(x); transcendental. f32/f64 native; bf16/f16 via f32. Identical to the non-in-place cos."

determinism: same_hardware_bitwise
```

## sign_inplace  (out = sign(out))

In-place sign: `out[i] = sign(out[i]) ∈ {-1, 0, 1}` over a single contiguous buffer
(`sign_inplace_<dt>`). Sign(0)=0. bf16/f16 via f32. Numerics identical to the non-in-place `sign`
cousin. Contiguous-only.

```fkc
kernel: sign_inplace
op_kind: SignInplace
blurb: "In-place sign out[i]=sign(out[i]) in {-1,0,1}; single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sign_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "sign(x) ∈ {-1,0,1}; Sign(0)=0. bf16/f16 via f32. Identical to the non-in-place sign."

determinism: same_hardware_bitwise
```

## floor_inplace  (out = ⌊out⌋)

In-place floor: `out[i] = ⌊out[i]⌋` over a single contiguous buffer (`floor_inplace_<dt>`). Exact.
bf16/f16 via f32. Numerics identical to the non-in-place `floor` cousin. Contiguous-only.

```fkc
kernel: floor_inplace
op_kind: FloorInplace
blurb: "In-place floor out[i]=⌊out[i]⌋; single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::floor_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "⌊x⌋; exact. bf16/f16 via f32. Identical to the non-in-place floor."

determinism: same_hardware_bitwise
```

## ceil_inplace  (out = ⌈out⌉)

In-place ceiling: `out[i] = ⌈out[i]⌉` over a single contiguous buffer (`ceil_inplace_<dt>`). Exact.
bf16/f16 via f32. Numerics identical to the non-in-place `ceil` cousin. Contiguous-only.

```fkc
kernel: ceil_inplace
op_kind: CeilInplace
blurb: "In-place ceil out[i]=⌈out[i]⌉; single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::ceil_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "⌈x⌉; exact. bf16/f16 via f32. Identical to the non-in-place ceil."

determinism: same_hardware_bitwise
```

## round_inplace  (out = round_ties_even(out))

In-place round (banker's rounding): `out[i] = round_ties_even(out[i])` over a single contiguous
buffer (`round_inplace_<dt>`). Ties to even. bf16/f16 via f32. Numerics identical to the
non-in-place `round` cousin. Contiguous-only.

```fkc
kernel: round_inplace
op_kind: RoundInplace
blurb: "In-place round (ties-to-even) out[i]=round_ties_even(out[i]); single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::round_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "round_ties_even (banker's rounding). bf16/f16 via f32. Identical to the non-in-place round."

determinism: same_hardware_bitwise
```

## erf_inplace  (out = erf(out))

In-place error function: `out[i] = erf(out[i])` over a single contiguous buffer (`erf_inplace_<dt>`).
Via `libm::erf{,f}`. f32/f64 native; bf16/f16 via f32. Numerics identical to the non-in-place `erf`
cousin. Contiguous-only.

```fkc
kernel: erf_inplace
op_kind: ErfInplace
blurb: "In-place erf out[i]=erf(out[i]) via libm; single contiguous buffer; half via f32; numerics identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::erf_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "erf(x) via libm::erf{,f}; f32/f64 native; bf16/f16 via f32. Identical to the non-in-place erf."

determinism: same_hardware_bitwise
```

## gelu_erf_inplace  (out = GeluErf(out))

In-place exact-erf GELU (`OpKind::GeluErfInplace`, DISTINCT from the tanh `GeluInplace`):
`out[i] = 0.5·out[i]·(1 + erf(out[i]/√2))` over a single contiguous buffer (`gelu_erf_inplace_<dt>`).
Uses `libm::erf{,f}`. f32/f64 native; bf16/f16 via f32. Numerics identical to the non-in-place
`gelu_erf` cousin. Contiguous-only.

```fkc
kernel: gelu_erf_inplace
op_kind: GeluErfInplace
blurb: "In-place exact-erf GELU out[i]=0.5x(1+erf(x/√2)) via libm; single contiguous buffer; half via f32; identical to the non-in-place cousin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::gelu_erf_inplace"
kernel_revision_hash: auto

accept:
  inputs:
    - name: out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  notes: "exact-erf GELU: 0.5x(1+erf(x/√2)) via libm; OpKind GeluErfInplace, DISTINCT from the tanh GeluInplace. bf16/f16 via f32. Identical to the non-in-place gelu_erf."

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
op_kind: InplaceAffine         # OpParams::Affine signals the op family; in-place dispatch arm
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
op_kind: InplaceAffine
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
op_kind: InplaceAffine
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
op_kind: InplaceAffine
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

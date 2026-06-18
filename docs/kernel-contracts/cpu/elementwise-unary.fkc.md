---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                     # maps to BackendId::Cpu
  kernel_source: "portable-cpu"    # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::byte_kernels::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"    # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — elementwise-unary kernel contracts

Portable, byte-shaped CPU elementwise-unary kernels. Every kernel here is one logical pass over a
flat `CpuStorageBytes` buffer, `out[i] = op(in[i])`, generated from the shared `chassis::unary`
walker (`fuel-cpu-backend/src/chassis/unary.rs:94`) with per-(op, dtype) public thunks emitted by
`unary_thunk!` (`fuel-cpu-backend/src/byte_kernels.rs:93`).

Cross-cutting facts for this whole family (verified against the inventory and source):

- **Layout: contiguous-only, offset 0, row-major.** Every kernel reads the input via `as_slice()`
  and validates `in.len_bytes() == out.len_bytes()`; none consults a `Layout`/strides/offset. The
  pipelined executor's auto-Contiguize pass realizes any strided / broadcast / non-zero-offset /
  reversed input into a dense zero-offset buffer **before** these kernels run. Each operand
  therefore declares `awkward_layout_strategy: requires_contiguous` (§4.3) — the planner inserts
  `Op::Contiguize` (itself a CPU FKC kernel) for a non-contiguous producer and sums its cost
  (§4.4). `reverse_strides: rejected` (no signed-stride walk; a flipped view is normalized first).
- **dtypes `{F32, F64, BF16, F16}`.** f32/f64 compute natively; bf16/f16 widen each element to f32,
  do the math, narrow on store (`chassis/unary.rs:72-82`, the blanket half impls). This f32
  round-trip is byte-identical to the pre-chassis per-dtype kernels.
- **Output: same dtype, same element count, contiguous row-major, full overwrite.** No read of
  prior output content; no aliasing with the input (the in-place unary family is a separate
  surface — `unary_inplace_thunk!`, `byte_kernels.rs:2822` — not contracted here).
- **op_params: `OpParams::None`.** No scalar/shape params ride a unary op; shape is positional
  (`fuel-dispatch/src/kernel.rs:167`).
- **Cost: bandwidth-bound elementwise.** Every op touches `n` input + `n` output elements, so a
  genuinely derivable hint is `bytes_moved = 2 * n * dtype_bytes` (read in, write out); `flops`
  is op-dependent (one cheap arithmetic op vs a transcendental). **Cost is marked
  `judge_measured` (the Judge bootstraps it)** — no per-op timing numbers are fabricated. The
  bandwidth formula is supplied as a real, derivable hint; `overhead_ns` and the precise
  per-op `flops` are left to the Judge. Provenance is therefore `judge_measured` (§4.4): a
  first-class, visible marker, not a hidden gap.
- **Precision:** these are CPU primitive kernels; per §4.8 / §12.4 a CPU primitive kernel may leave
  its precision block `audited: false` and let the importer's `fill_unset_cpu_precision` pass apply
  `PRIMITIVE_DETERMINISTIC_CPU`. Each kernel below states `bit_stable_on_same_hardware: true`
  (deterministic single-threaded nested loop; no atomic/reduction reordering) and carries the
  numeric notes from source; bounds are left null as Judge-audited seeds. This satisfies the
  always-built bit-stable coverage commitment (§4.8, §10.9).
- **Determinism: `same_hardware_bitwise`.** A fixed positional loop with no FP reordering is
  bit-identical on re-run on the same hardware; cross-hardware bit-stability is not claimed
  (libm/std transcendental implementations differ).

---

## unary  (the shared elementwise-unary chassis `out[i] = op(in[i])`)

Generic single-pass elementwise-unary walker shared by every per-op kernel in this family.

The `unary<T, U>` chassis (`chassis/unary.rs:94`) is the one shape/loop pass behind all the named
unary kernels below: it validates `input.len_bytes() == output.len_bytes()`, reinterprets both
buffers as typed `&[T]` / `&mut [T]` slices, and walks `out[i] = U::apply(in[i])` positionally. The
per-op math is supplied by a zero-sized `UnaryOpCore` marker (`U`); the four dtype impls
(f32/f64 direct, bf16/f16 via f32 round-trip) fall out of blanket impls, so a kernel author writes
the math once and gets the whole `{F32, F64, BF16, F16}` set. This section documents the chassis
contract that every named op specializes; it is **not separately dispatchable** (it carries no
`OpKind` of its own — each concrete op below pins one `OpKind`), so it is described here for
completeness and registers no binding. Layout/dtype/output/precision facts are exactly the
cross-cutting facts above. Known limitation: contiguous-only (no internal layout handling); the
executor contiguizes first.

```fkc
kernel: unary
registrable: false           # §3.10 describe-only: shared chassis umbrella, NOT a dispatch target
op_kind: ~                   # the chassis itself binds no OpKind; each named op below pins one
fused_op: ~
blurb: "Shared elementwise-unary chassis walker out[i]=op(in[i]); contiguous; half via f32; not separately dispatchable."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::chassis::unary::unary"   # the generic walker; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured              # Judge bootstraps; bandwidth hint derivable, per-op flops not
  class: cheap_elementwise
  flops: "n"                              # one op per element (op-dependent magnitude; Judge refines)
  bytes_moved: "2 * n * dtype_bytes"      # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                          # CPU primitive; fill_unset_cpu_precision applies PRIMITIVE_DETERMINISTIC_CPU (§12.4)
  notes: "Generic chassis; per-op numerics in the specialized sections. f32/f64 native; bf16/f16 widen to f32 then narrow."

determinism: same_hardware_bitwise
```

---

## relu  (rectified linear unit, `max(0, x)`)

Elementwise ReLU clamp.

`out[i] = max(0, in[i])` (`chassis/unary.rs:130`, `f32`/`f64` via `x.max(0.0)`). bf16/f16 widen to
f32, clamp, narrow. NaN propagation follows `f32::max` (NaN-as-missing: `max(NaN, 0)` returns 0,
the non-NaN operand). One cheap branchless op per element; bandwidth-bound. Contiguous-only;
executor contiguizes any strided/broadcast/offset input first.

```fkc
kernel: relu
op_kind: ReluElementwise
blurb: "Elementwise ReLU max(0, x); contiguous same-shape; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::relu_f32"   # one per (op, dtype): relu_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "max(0, x); exact for f32/f64. bf16/f16 widen to f32 then narrow. NaN-as-missing (f32::max)."

determinism: same_hardware_bitwise
```

---

## neg  (negation, `-x`)

Elementwise negation.

`out[i] = -in[i]` (`chassis/unary.rs:137`). Exact for all dtypes (a sign flip; bf16/f16 round-trip
through f32 is bit-exact for negation). Bandwidth-bound.

```fkc
kernel: neg
op_kind: NegElementwise
blurb: "Elementwise negation -x; contiguous same-shape; exact all dtypes."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::neg_f32"   # neg_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "-x; exact (sign flip) for every dtype, including bf16/f16."

determinism: same_hardware_bitwise
```

---

## sqr  (square, `x * x`)

Elementwise square.

`out[i] = in[i] * in[i]` (`chassis/unary.rs:144`). f32/f64 native multiply; bf16/f16 widen to f32,
multiply, narrow on store (so the product is rounded once at f32 then again at half — the
documented round-trip behavior). Bandwidth-bound.

```fkc
kernel: sqr
op_kind: SqrElementwise
blurb: "Elementwise square x*x; contiguous same-shape; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sqr_f32"   # sqr_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "x*x; f32/f64 native. bf16/f16 widen to f32 then narrow (double rounding)."

determinism: same_hardware_bitwise
```

---

## sqrt  (square root)

Elementwise square root.

`out[i] = sqrt(in[i])` (`chassis/unary.rs:151`, `x.sqrt()`). Negative inputs yield NaN per
IEEE-754; `sqrt` is correctly rounded for f32/f64 (IEEE-754 mandated). bf16/f16 widen to f32,
sqrt, narrow. Bandwidth-bound.

```fkc
kernel: sqrt
op_kind: SqrtElementwise
blurb: "Elementwise square root; contiguous same-shape; negatives -> NaN; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sqrt_f32"   # sqrt_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "sqrt(x), IEEE-754 correctly rounded for f32/f64; negatives -> NaN. bf16/f16 via f32."

determinism: same_hardware_bitwise
```

---

## recip  (reciprocal, `1 / x`)

Elementwise reciprocal.

`out[i] = 1 / in[i]` (`chassis/unary.rs:158`). Zero input yields IEEE-754 inf (±) / NaN(0/0 does
not arise — `1/0` is inf). f32/f64 native divide; bf16/f16 via f32. Bandwidth-bound.

```fkc
kernel: recip
op_kind: RecipElementwise
blurb: "Elementwise reciprocal 1/x; contiguous same-shape; IEEE inf/NaN; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::recip_f32"   # recip_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "1/x; f32/f64 IEEE divide (1/0 -> inf). bf16/f16 via f32."

determinism: same_hardware_bitwise
```

---

## abs  (absolute value, `|x|`)

Elementwise absolute value.

`out[i] = |in[i]|` (`chassis/unary.rs:165`, `x.abs()`). Exact for all dtypes (sign clear).
Bandwidth-bound.

```fkc
kernel: abs
op_kind: AbsElementwise
blurb: "Elementwise absolute value |x|; contiguous same-shape; exact all dtypes."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::abs_f32"   # abs_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "|x|; exact (sign clear) for every dtype, including bf16/f16."

determinism: same_hardware_bitwise
```

---

## tanh  (hyperbolic tangent)

Elementwise hyperbolic tangent.

`out[i] = tanh(in[i])` (`chassis/unary.rs:172`, `x.tanh()` — std/libm transcendental). f32/f64
native; bf16/f16 widen to f32. Not correctly-rounded (transcendental); accurate to the std library
guarantee, not bit-stable across different libm implementations. Bandwidth-bound (a transcendental
is more FLOPs than an arithmetic op but still memory-bound at scale).

```fkc
kernel: tanh
op_kind: TanhElementwise
blurb: "Elementwise tanh; contiguous same-shape; half via f32; not bit-stable cross-hardware."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::tanh_f32"   # tanh_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "tanh via std/libm; not correctly-rounded; bf16/f16 via f32. Same-hardware bitwise but not cross-hardware (libm differs)."

determinism: same_hardware_bitwise
```

---

## exp  (exponential, `e^x`)

Elementwise exponential.

`out[i] = exp(in[i])` (`chassis/unary.rs:179`, `x.exp()`). Transcendental, std/libm-accurate, not
correctly-rounded; overflow -> +inf, large negative -> 0. f32/f64 native; bf16/f16 via f32.

```fkc
kernel: exp
op_kind: ExpElementwise
blurb: "Elementwise exp e^x; contiguous same-shape; half via f32; not bit-stable cross-hardware."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::exp_f32"   # exp_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "e^x via std/libm; not correctly-rounded; bf16/f16 via f32. Same-hardware bitwise, not cross-hardware."

determinism: same_hardware_bitwise
```

---

## log  (natural logarithm, `ln(x)`)

Elementwise natural logarithm.

`out[i] = ln(in[i])` (`chassis/unary.rs:186`, `x.ln()`). Negative inputs yield NaN, `ln(0) = -inf`
per IEEE-754. Transcendental, std/libm-accurate, not correctly-rounded. f32/f64 native; bf16/f16
via f32.

```fkc
kernel: log
op_kind: LogElementwise
blurb: "Elementwise natural log ln(x); contiguous same-shape; negatives -> NaN; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::log_f32"   # log_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "ln(x) via std/libm; negatives -> NaN, ln(0) -> -inf; not correctly-rounded; bf16/f16 via f32."

determinism: same_hardware_bitwise
```

---

## sin  (sine)

Elementwise sine.

`out[i] = sin(in[i])` (`chassis/unary.rs:193`, `x.sin()`). Transcendental, std/libm-accurate, not
correctly-rounded; large-argument range reduction follows the std library. f32/f64 native; bf16/f16
via f32.

```fkc
kernel: sin
op_kind: SinElementwise
blurb: "Elementwise sine; contiguous same-shape; half via f32; not bit-stable cross-hardware."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sin_f32"   # sin_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "sin(x) via std/libm; not correctly-rounded; bf16/f16 via f32. Same-hardware bitwise, not cross-hardware."

determinism: same_hardware_bitwise
```

---

## cos  (cosine)

Elementwise cosine.

`out[i] = cos(in[i])` (`chassis/unary.rs:200`, `x.cos()`). Transcendental, std/libm-accurate, not
correctly-rounded. f32/f64 native; bf16/f16 via f32.

```fkc
kernel: cos
op_kind: CosElementwise
blurb: "Elementwise cosine; contiguous same-shape; half via f32; not bit-stable cross-hardware."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cos_f32"   # cos_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "cos(x) via std/libm; not correctly-rounded; bf16/f16 via f32. Same-hardware bitwise, not cross-hardware."

determinism: same_hardware_bitwise
```

---

## sigmoid  (logistic sigmoid, `1 / (1 + e^-x)`)

Elementwise logistic sigmoid.

`out[i] = 1 / (1 + exp(-in[i]))` (`chassis/unary.rs:207`). Composed of `exp` + add + divide;
transcendental-class accuracy (std/libm `exp`), not correctly-rounded. f32/f64 native; bf16/f16
widen to f32. Bandwidth-bound at scale.

```fkc
kernel: sigmoid
op_kind: SigmoidElementwise
blurb: "Elementwise logistic sigmoid 1/(1+e^-x); contiguous same-shape; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sigmoid_f32"   # sigmoid_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "1/(1+exp(-x)) via std/libm exp; not correctly-rounded; bf16/f16 via f32."

determinism: same_hardware_bitwise
```

---

## silu  (SiLU / Swish, `x * sigmoid(x)`)

Elementwise SiLU / Swish activation.

`out[i] = in[i] / (1 + exp(-in[i]))` (`chassis/unary.rs:214` — algebraically `x * sigmoid(x)`,
computed as `x / (1 + exp(-x))`). Transcendental-class accuracy (std/libm `exp`), not
correctly-rounded. f32/f64 native; bf16/f16 via f32.

```fkc
kernel: silu
op_kind: SiluElementwise
blurb: "Elementwise SiLU/Swish x*sigmoid(x); contiguous same-shape; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::silu_f32"   # silu_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "x/(1+exp(-x)) via std/libm exp; not correctly-rounded; bf16/f16 via f32."

determinism: same_hardware_bitwise
```

---

## step  (Heaviside step, `1 where x > 0 else 0`)

Elementwise Heaviside step (the derivative of ReLU).

`out[i] = if in[i] > 0 { 1 } else { 0 }` (`chassis/unary.rs:221`). Exact (a comparison + select);
NaN compares false so `step(NaN) = 0`. f32/f64 native; bf16/f16 compare in f32. Bandwidth-bound.

```fkc
kernel: step
op_kind: StepElementwise
blurb: "Elementwise Heaviside step 1 where x>0 else 0; contiguous same-shape; exact."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::step_f32"   # step_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "x>0 ? 1 : 0; exact compare+select; step(NaN)=0 (NaN compares false). bf16/f16 compared in f32."

determinism: same_hardware_bitwise
```

---

## floor  (floor, `⌊x⌋`)

Elementwise floor.

`out[i] = floor(in[i])` (`chassis/unary.rs:228`, `x.floor()`). Exact (IEEE-754 roundTowardNegative
on a value already representable in the dtype). f32/f64 native; bf16/f16 via f32. Bandwidth-bound.

```fkc
kernel: floor
op_kind: FloorElementwise
blurb: "Elementwise floor ⌊x⌋; contiguous same-shape; exact."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::floor_f32"   # floor_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "floor(x); exact (roundTowardNegative). bf16/f16 via f32 (the f32 floor is exactly representable back in half)."

determinism: same_hardware_bitwise
```

---

## ceil  (ceiling, `⌈x⌉`)

Elementwise ceiling.

`out[i] = ceil(in[i])` (`chassis/unary.rs:235`, `x.ceil()`). Exact (IEEE-754
roundTowardPositive). f32/f64 native; bf16/f16 via f32. Bandwidth-bound.

```fkc
kernel: ceil
op_kind: CeilElementwise
blurb: "Elementwise ceiling ⌈x⌉; contiguous same-shape; exact."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::ceil_f32"   # ceil_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "ceil(x); exact (roundTowardPositive). bf16/f16 via f32 (exactly representable back in half)."

determinism: same_hardware_bitwise
```

---

## round  (round-half-to-even, banker's rounding)

Elementwise round-half-to-even.

`out[i] = round_ties_even(in[i])` (`chassis/unary.rs:243`, `x.round_ties_even()` — IEEE-754
roundTiesToEven / banker's rounding, NOT round-half-away-from-zero). Exact. f32/f64 native;
bf16/f16 via f32. Bandwidth-bound.

```fkc
kernel: round
op_kind: RoundElementwise
blurb: "Elementwise round-half-to-even (banker's); contiguous same-shape; exact."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::round_f32"   # round_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "round_ties_even (banker's, roundTiesToEven, NOT half-away-from-zero); exact. bf16/f16 via f32."

determinism: same_hardware_bitwise
```

---

## sign  (sign, `-1 / 0 / 1`)

Elementwise sign with `sign(0) = 0`.

`out[i] = if in[i] > 0 { 1 } else if in[i] < 0 { -1 } else { 0 }` (`chassis/unary.rs:249`).
`sign(0) = 0` by subgradient convention (matches PyTorch `torch.sign`); `sign(NaN) = 0` (NaN is
neither `> 0` nor `< 0`, falls to the else branch). Exact compare + select. f32/f64 native;
bf16/f16 compare in f32. Bandwidth-bound.

```fkc
kernel: sign
op_kind: SignElementwise
blurb: "Elementwise sign -1/0/1 with sign(0)=0; contiguous same-shape; exact."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sign_f32"   # sign_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "x>0 ? 1 : x<0 ? -1 : 0; exact; sign(0)=0, sign(NaN)=0. bf16/f16 compared in f32."

determinism: same_hardware_bitwise
```

---

## erf  (Gauss error function, `erf(x)`)

Elementwise Gauss error function.

`out[i] = erf(in[i])` via `libm::erff` (f32) / `libm::erf` (f64) (`chassis/unary.rs:261`).
libm-accurate, not correctly-rounded. f32/f64 native; bf16/f16 widen to f32 and use `erff`.
Bandwidth-bound.

```fkc
kernel: erf
op_kind: ErfElementwise
blurb: "Elementwise Gauss error function erf(x) via libm; contiguous same-shape; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::erf_f32"   # erf_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "erf via libm::erff (f32) / libm::erf (f64); not correctly-rounded; bf16/f16 via f32 (erff). Same-hardware bitwise (libm pinned), not cross-hardware."

determinism: same_hardware_bitwise
```

---

## gelu_erf  (GELU, exact erf form `0.5*x*(1+erf(x/√2))`)

Elementwise GELU, exact error-function formulation.

`out[i] = 0.5 * in[i] * (1 + erf(in[i] * FRAC_1_SQRT_2))` via libm `erff`/`erf`
(`chassis/unary.rs:267`). Distinct from `gelu_tanh` (the tanh approximation) — this is the exact
erf-based GELU and must NOT be confused with it under a Judge epsilon. f32 uses `FRAC_1_SQRT_2`
(f32 const) + `erff`; f64 uses the f64 const + `erf`. bf16/f16 widen to f32 and use the f32 path.
libm-accurate, not correctly-rounded. Bandwidth-bound.

```fkc
kernel: gelu_erf
op_kind: GeluErfElementwise
blurb: "Elementwise exact-erf GELU 0.5*x*(1+erf(x/sqrt2)); contiguous same-shape; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::gelu_erf_f32"   # gelu_erf_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "EXACT erf GELU 0.5*x*(1+erf(x/sqrt2)) via libm; DISTINCT from gelu_tanh. bf16/f16 via f32. Same-hardware bitwise, not cross-hardware."

determinism: same_hardware_bitwise
```

---

## gelu_tanh  (GELU, tanh approximation — the canonical `Gelu` op)

Elementwise GELU, tanh approximation (`OpKind::GeluElementwise`).

`out[i] = 0.5 * in[i] * (1 + tanh(√(2/π) * (in[i] + 0.044715 * in[i]³)))` (`chassis/unary.rs:290`,
the `GeluTanh` marker). This is the canonical `Op::Gelu` / `OpKind::GeluElementwise` (the tanh
approximation IS Fuel's default GELU; the exact-erf form is the separate `gelu_erf` above). The
√(2/π) constant is **7-digit for f32** (`0.797_884_56`) and **16-digit for f64**
(`0.797_884_560_802_865_4`) — both match the pre-chassis `gelu_*` functions bit-for-bit. bf16/f16
route through the f32 path. The public thunks for half are named `gelu_bf16` / `gelu_f16`
(`byte_kernels.rs:181,207`), f32/f64 via the chassis. libm/std `tanh` accuracy, not
correctly-rounded. Bandwidth-bound.

```fkc
kernel: gelu_tanh
op_kind: GeluElementwise
blurb: "Elementwise tanh-approx GELU (the canonical Gelu); contiguous same-shape; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::gelu_f32"   # gelu_{f32,f64,bf16,f16} (half thunks gelu_bf16/gelu_f16); §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "TANH-approx GELU 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))); sqrt(2/pi) 7-digit f32 / 16-digit f64; DISTINCT from gelu_erf. bf16/f16 via f32. Same-hardware bitwise, not cross-hardware."

determinism: same_hardware_bitwise
```

---

## rsqrt  (reciprocal square root, `1 / sqrt(x)`)

Elementwise reciprocal square root.

`out[i] = 1 / sqrt(in[i])` (`chassis/unary.rs:278`). A single op (not `Sqrt` then `Recip` —
combining them loses precision and doubles launches); critical for RMSNorm
(`x * rsqrt(mean(x²)+eps)`). Negative inputs yield NaN, `rsqrt(0) = +inf` per IEEE-754. f32/f64
native (`sqrt` correctly rounded then one divide); bf16/f16 via f32. Bandwidth-bound.

```fkc
kernel: rsqrt
op_kind: RsqrtElementwise
blurb: "Elementwise reciprocal sqrt 1/sqrt(x); single op; contiguous same-shape; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rsqrt_f32"   # rsqrt_{f32,f64,bf16,f16}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

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
  notes: "1/sqrt(x) as a single op (not Sqrt+Recip); negatives -> NaN, rsqrt(0) -> +inf. f32/f64 native; bf16/f16 via f32."

determinism: same_hardware_bitwise
```

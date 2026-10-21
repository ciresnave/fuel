---
fkc_version: 1
provider:
  name: fuel-reference-backend
  backend: Cpu                  # maps to BackendId::Cpu
  kernel_source: "reference-oracle"   # the BindingEntry.kernel_source tag
  link_registry: fuel_reference_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-reference-backend — elementwise kernel contracts

The pure-Rust, correctness-first **oracle** backend. Every kernel here is the reference
implementation Fuel's Judge measures the GPU/SIMD backends against, so the precision claims are
intentionally tight and every kernel is `same_hardware_bitwise` deterministic.

**Crate-wide layout invariant (from the inventory).** `RefTensor<T>` is *always* a contiguous,
row-major `Vec`/`Arc<[T]>` plus a `Shape`, carrying **no strides and no offset**. There is no
`is_contiguous()` branch and no strided/broadcast/offset input path anywhere — callers must
materialize any non-contiguous view into a fresh contiguous `RefTensor` before calling. Every
kernel below is therefore, by construction, **contiguous-only with zero offset**
(`awkward_layout_strategy: requires_contiguous`), and every output is a **fresh contiguous
buffer** (`pure iter().map()` / `zip`, no in-place, no aliasing). The numeric kernels are generic
over `T: num_traits::Float`, monomorphized by the executor to `{F32, F64, BF16, F16}`; bf16/f16
are computed at the kernel's `T` precision unless the per-kernel description states an explicit
widen-to-f64.

All cost blocks are marked `provenance: judge_measured` — this is the oracle crate the Judge
bootstraps cost from, so the coefficients are populated by measurement rather than authored
priors. A real bandwidth formula hint (`bytes_moved`) is given where it is genuinely derivable
from the op (elementwise = N elements, bandwidth-bound: read inputs + write one output); `flops`
is left as the per-element count `n`, and `overhead_ns` is `judge_measured` (no fabricated number).

---

## neg  (y = -x)

Element-wise negation `out[i] = -x[i]` over a contiguous, zero-offset, row-major buffer
(`ops.rs:30`). Pure `iter().map()` over the flat slice; output is the same shape and dtype in a
fresh contiguous buffer. f32/f64 evaluate natively; bf16/f16 negate at `T`. No broadcasting, no
strides, no offset, no in-place. Bandwidth-bound: one element read, one written.

```fkc
kernel: neg
op_kind: NegElementwise
blurb: "Element-wise negation y = -x; contiguous same-shape; fresh contiguous output."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::neg"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read x, write out
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact sign flip; bit-exact for all finite inputs; IEEE-754 negation (handles -0.0, NaN)."

determinism: same_hardware_bitwise
```

---

## relu  (y = max(0, x))

Element-wise rectified linear unit `out[i] = x[i] > 0 ? x[i] : 0` (`ops.rs:36`). Exactly `0`
returns `0` (the branch is `x > 0`, so `+0.0`/`-0.0`/negatives all map to `0`). Contiguous,
zero-offset; fresh contiguous output, same shape/dtype. Bandwidth-bound.

```fkc
kernel: relu
op_kind: ReluElementwise
blurb: "Element-wise ReLU y = max(0, x) (0 at exactly 0); contiguous same-shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::relu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact select (no arithmetic): passes x or +0.0; bit-exact. NaN passes through (NaN > 0 is false ⇒ returns 0). 0 at exactly 0."

determinism: same_hardware_bitwise
```

---

## sqr  (y = x * x)

Element-wise square `out[i] = x[i] * x[i]` (`ops.rs:47`). Single multiply per element; computed
at `T`. Contiguous, zero-offset; fresh contiguous output, same shape/dtype. Bandwidth-bound.

```fkc
kernel: sqr
op_kind: SqrElementwise
blurb: "Element-wise square y = x*x; contiguous same-shape; fresh contiguous output."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::sqr"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Single IEEE multiply x*x at T; correctly-rounded, ≤0.5 ULP for f32/f64; bf16/f16 round at T (no widening). May overflow to +inf for large |x|."

determinism: same_hardware_bitwise
```

---

## sqrt  (y = √x)

Element-wise square root `out[i] = x[i].sqrt()` (`ops.rs:53`). IEEE correctly-rounded sqrt;
negative input → NaN. Contiguous, zero-offset; fresh contiguous output, same shape/dtype.
Bandwidth-bound.

```fkc
kernel: sqrt
op_kind: SqrtElementwise
blurb: "Element-wise square root y = sqrt(x); contiguous same-shape; NaN on negative input."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::sqrt"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "IEEE-754 correctly-rounded sqrt at T (≤0.5 ULP for f32/f64). Negative → NaN, -0.0 → -0.0 (IEEE). bf16/f16 computed at T."

determinism: same_hardware_bitwise
```

---

## exp  (y = e^x)

Element-wise natural exponential `out[i] = e^x[i]` (`ops.rs:59`). Transcendental via the `T: Float`
`exp()` (libm/std math). Contiguous, zero-offset; fresh contiguous output, same shape/dtype.
Bandwidth-bound; the ULP bound is the math-library bound, not bit-exact.

```fkc
kernel: exp
op_kind: ExpElementwise
blurb: "Element-wise exponential y = e^x; contiguous same-shape; math-library accuracy."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::exp"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 1
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Transcendental exp at T via the platform math library (typically ≤1 ULP for f32/f64). Same-hardware bit-stable (same libm). Overflow → +inf; large negative → +0.0."

determinism: same_hardware_bitwise
```

---

## sign  (y = sgn(x))

Element-wise sign: `-1` if negative, `0` if zero (including `-0.0`), `+1` if positive
(`ops.rs:65`). Returns `0` at exactly `0` and at `-0.0`. Contiguous, zero-offset; fresh contiguous
output, same shape/dtype. Bandwidth-bound.

```fkc
kernel: sign
op_kind: SignElementwise
blurb: "Element-wise sign -1/0/+1 (0 at ±0); contiguous same-shape; fresh contiguous output."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::sign"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact comparison-select producing -1.0/0.0/+1.0; bit-exact. 0 at exactly 0 and at -0.0."

determinism: same_hardware_bitwise
```

---

## log  (y = ln(x))

Element-wise natural logarithm `out[i] = x[i].ln()` (`ops.rs:87`). Non-positive inputs follow IEEE
passthrough: `ln(0) → -inf`, `ln(negative) → NaN`. Contiguous, zero-offset; fresh contiguous
output, same shape/dtype. Bandwidth-bound; ULP bound is the math-library bound.

```fkc
kernel: log
op_kind: LogElementwise
blurb: "Element-wise natural log y = ln(x); non-positive → IEEE NaN/-inf; contiguous same-shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::log"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 1
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Transcendental ln at T via platform math library (typically ≤1 ULP for f32/f64). ln(0) → -inf, ln(<0) → NaN (IEEE passthrough). Same-hardware bit-stable."

determinism: same_hardware_bitwise
```

---

## sin  (y = sin(x))

Element-wise sine `out[i] = x[i].sin()` (`ops.rs:93`). Transcendental via `T: Float` `sin()`.
Contiguous, zero-offset; fresh contiguous output, same shape/dtype. Bandwidth-bound; ULP bound is
the math-library bound (degrades for large arguments via range reduction).

```fkc
kernel: sin
op_kind: SinElementwise
blurb: "Element-wise sine y = sin(x); contiguous same-shape; math-library accuracy."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::sin"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 1
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Transcendental sin at T via platform math library (typically ≤1 ULP for f32/f64 in the principal range; accuracy degrades with |x| due to argument reduction). Same-hardware bit-stable."

determinism: same_hardware_bitwise
```

---

## cos  (y = cos(x))

Element-wise cosine `out[i] = x[i].cos()` (`ops.rs:99`). Transcendental via `T: Float` `cos()`.
Contiguous, zero-offset; fresh contiguous output, same shape/dtype. Bandwidth-bound; ULP bound is
the math-library bound.

```fkc
kernel: cos
op_kind: CosElementwise
blurb: "Element-wise cosine y = cos(x); contiguous same-shape; math-library accuracy."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cos"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 1
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Transcendental cos at T via platform math library (typically ≤1 ULP for f32/f64 in the principal range; degrades with |x| due to argument reduction). Same-hardware bit-stable."

determinism: same_hardware_bitwise
```

---

## abs  (y = |x|)

Element-wise absolute value `out[i] = |x[i]|` (`ops.rs:105`). Contiguous, zero-offset; fresh
contiguous output, same shape/dtype. Bandwidth-bound.

```fkc
kernel: abs
op_kind: AbsElementwise
blurb: "Element-wise absolute value y = |x|; contiguous same-shape; fresh contiguous output."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::abs"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact sign-bit clear; bit-exact for all finite inputs. |NaN| = NaN, |-0.0| = +0.0 (IEEE)."

determinism: same_hardware_bitwise
```

---

## recip  (y = 1/x)

Element-wise reciprocal `out[i] = 1 / x[i]` (`ops.rs:112`). `1/0 → inf` (IEEE). Contiguous,
zero-offset; fresh contiguous output, same shape/dtype. Bandwidth-bound.

```fkc
kernel: recip
op_kind: RecipElementwise
blurb: "Element-wise reciprocal y = 1/x; 1/0 → inf (IEEE); contiguous same-shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::recip"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Single IEEE divide 1.0/x at T; correctly-rounded ≤0.5 ULP for f32/f64. 1/0 → ±inf, 1/inf → ±0.0 (IEEE). bf16/f16 round at T."

determinism: same_hardware_bitwise
```

---

## tanh  (y = tanh(x))

Element-wise hyperbolic tangent `out[i] = x[i].tanh()` (`ops.rs:119`). Transcendental via
`T: Float` `tanh()`. Contiguous, zero-offset; fresh contiguous output, same shape/dtype.
Bandwidth-bound; ULP bound is the math-library bound.

```fkc
kernel: tanh
op_kind: TanhElementwise
blurb: "Element-wise hyperbolic tangent y = tanh(x); contiguous same-shape; math-library accuracy."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::tanh"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 1
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Transcendental tanh at T via platform math library (typically ≤1 ULP for f32/f64). Saturates to ±1 for large |x|. Same-hardware bit-stable."

determinism: same_hardware_bitwise
```

---

## floor  (y = ⌊x⌋)

Element-wise floor `out[i] = x[i].floor()` (`ops.rs:125`). Exact integer-valued result; backward
drops the gradient (zero almost everywhere). Contiguous, zero-offset; fresh contiguous output,
same shape/dtype. Bandwidth-bound.

```fkc
kernel: floor
op_kind: FloorElementwise
blurb: "Element-wise floor y = ⌊x⌋; exact; contiguous same-shape; fresh contiguous output."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::floor"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact round-toward-negative-infinity to an integer value; bit-exact (IEEE roundToIntegral). NaN/inf pass through."

determinism: same_hardware_bitwise
```

---

## ceil  (y = ⌈x⌉)

Element-wise ceiling `out[i] = x[i].ceil()` (`ops.rs:131`). Exact integer-valued result; backward
drops the gradient (mirrors floor). Contiguous, zero-offset; fresh contiguous output, same
shape/dtype. Bandwidth-bound.

```fkc
kernel: ceil
op_kind: CeilElementwise
blurb: "Element-wise ceiling y = ⌈x⌉; exact; contiguous same-shape; fresh contiguous output."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::ceil"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact round-toward-positive-infinity to an integer value; bit-exact (IEEE roundToIntegral). NaN/inf pass through."

determinism: same_hardware_bitwise
```

---

## erf  (y = erf(x))

Element-wise Gauss error function (`ops.rs:141`). Computed by **widening every element to f64,
calling `libm::erf`, then narrowing back to `T`** — so the result is ≤1 ULP relative to the true
`erf` at f64 and is **precision-sensitive for bf16/f16** (the narrowing dominates the error). This
is the oracle path the GPU `erf` is judged against. Contiguous, zero-offset; fresh contiguous
output, same shape/dtype. Bandwidth-bound.

```fkc
kernel: erf
op_kind: ErfElementwise
blurb: "Element-wise erf(x) via widen-to-f64 → libm::erf → narrow; ≤1 ULP; contiguous same-shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::erf"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 1
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Widen to f64 → libm::erf → narrow to T; ≤1 ULP vs true erf at f64. Precision-sensitive for bf16/f16 (narrowing dominates). Same-hardware bit-stable (same libm)."

determinism: same_hardware_bitwise
```

---

## gelu_erf  (y = 0.5·x·(1 + erf(x/√2)))

Element-wise GELU, **exact erf formulation** (PyTorch `approximate='none'`): `0.5*x*(1 +
erf(x/√2))` computed entirely in **f64** then narrowed to `T` (`ops.rs:160`). Distinct from `gelu`
(the tanh approximation). The f64 intermediate makes this the high-accuracy oracle the GPU
gelu-erf is judged against; the error budget for bf16/f16 comes from the final narrowing.
Contiguous, zero-offset; fresh contiguous output, same shape/dtype. Bandwidth-bound.

```fkc
kernel: gelu_erf
op_kind: GeluErfElementwise
blurb: "Element-wise exact-erf GELU 0.5·x·(1+erf(x/√2)) computed in f64; contiguous same-shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::gelu_erf"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 2
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact erf form 0.5·x·(1+erf(x/√2)) evaluated in f64 (libm::erf) then narrowed to T; high-accuracy oracle (PyTorch approximate='none'). bf16/f16 error dominated by final narrowing. Same-hardware bit-stable."

determinism: same_hardware_bitwise
```

---

## round  (y = round-half-to-even(x))

Element-wise round-to-nearest with **banker's rounding** (round-half-to-even / IEEE roundeven)
(`ops.rs:180`). Overrides `Float::round` (half-away-from-zero) **only** at exact `.5` ties, picking
the even neighbour; non-tie cases delegate to `Float::round` (which agrees with roundeven
everywhere except the ties). Backward drops gradient. Contiguous, zero-offset; fresh contiguous
output, same shape/dtype. Bandwidth-bound.

```fkc
kernel: round
op_kind: RoundElementwise
blurb: "Element-wise round-half-to-even (banker's rounding); contiguous same-shape; fresh output."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::round"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact round-half-to-even (IEEE roundeven): ties at exact .5 go to the even integer; non-ties via Float::round. Bit-exact; NaN/inf pass through."

determinism: same_hardware_bitwise
```

---

## sigmoid  (y = 1/(1 + e^-x))

Element-wise logistic sigmoid in a numerically-stable split form, branching on `x >= 0`
(`ops.rs:214`). The branch avoids overflow of `e^-x` for large-magnitude inputs. Contiguous,
zero-offset; fresh contiguous output, same shape/dtype. Bandwidth-bound; one `exp` per element.

```fkc
kernel: sigmoid
op_kind: SigmoidElementwise
blurb: "Element-wise stable logistic sigmoid (branch on x≥0); contiguous same-shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::sigmoid"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 2
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Numerically-stable split form (branch on x≥0) avoiding exp overflow; one transcendental exp at T. Error dominated by the exp; ≤2 ULP for f32/f64. Same-hardware bit-stable."

determinism: same_hardware_bitwise
```

---

## silu  (y = x·sigmoid(x))

Element-wise SiLU / Swish `out[i] = x[i] * sigmoid(x[i])` (`ops.rs:234`). The reference computes
`sigmoid` first then does an elementwise multiply — **two passes** over the buffer (it reuses the
stable `sigmoid` kernel). Contiguous, zero-offset; fresh contiguous output, same shape/dtype.
Bandwidth-bound.

```fkc
kernel: silu
op_kind: SiluElementwise
blurb: "Element-wise SiLU/Swish y = x·sigmoid(x) (two passes); contiguous same-shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::silu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 2
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "x·sigmoid(x); computed as stable sigmoid then a multiply (two passes). Error inherited from sigmoid's exp; ≤2 ULP for f32/f64. Same-hardware bit-stable. NOTE: reference makes a temporary sigmoid buffer internally (two passes), so peak host memory is 2× the output during the call."

determinism: same_hardware_bitwise
```

---

## gelu  (y = 0.5·x·(1 + tanh(√(2/π)(x + 0.044715x³))))

Element-wise GELU, **tanh approximation** (`ops.rs:244`). Constants (`√(2/π)`, `0.044715`) are
materialized in dtype `T` via `cst`, so for bf16/f16 the polynomial is evaluated at `T` precision
(no f64 widening — distinct from `gelu_erf`). Contiguous, zero-offset; fresh contiguous output,
same shape/dtype. Bandwidth-bound.

```fkc
kernel: gelu
op_kind: GeluElementwise
blurb: "Element-wise tanh-approx GELU 0.5·x·(1+tanh(√(2/π)(x+0.044715x³))); contiguous same-shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::gelu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_relative: 0.001
  max_absolute: ~
  audited: true
  notes: "Tanh approximation (PyTorch approximate='tanh'); polynomial + tanh constants materialized in T (no f64 widening, unlike gelu_erf). This is an APPROXIMATION of true GELU — rel err vs exact erf-GELU ~1e-3 by design; ≤1 ULP vs its own tanh formula. Same-hardware bit-stable."

determinism: same_hardware_bitwise
```

---

## step  (y = Heaviside(x))

Element-wise Heaviside step `out[i] = x[i] > 0 ? 1 : 0` (`ops.rs:264`); `0` at exactly `0`. This is
the subgradient of `relu`. Contiguous, zero-offset; fresh contiguous output, same shape/dtype.
Bandwidth-bound.

```fkc
kernel: step
op_kind: StepElementwise
blurb: "Element-wise Heaviside step y = (x>0 ? 1 : 0) (0 at 0); contiguous same-shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::step"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact comparison-select producing 0.0/1.0; bit-exact. 0 at exactly 0 (branch is x>0). NaN > 0 is false ⇒ returns 0."

determinism: same_hardware_bitwise
```

---

## rsqrt  (y = 1/√x)

Element-wise reciprocal square root `out[i] = 1 / sqrt(x[i])` as a **single op** (`ops.rs:1520`),
not Sqrt+Recip — combining loses precision and doubles launches. Critical for the RMSNorm pattern
`x * rsqrt(mean(x²)+eps)`. Contiguous, zero-offset; fresh contiguous output, same shape/dtype.
Bandwidth-bound.

```fkc
kernel: rsqrt
op_kind: RsqrtElementwise
blurb: "Element-wise reciprocal sqrt y = 1/sqrt(x) (single op); contiguous same-shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::rsqrt"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }

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
  max_ulp: 1
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Single 1/sqrt(x) at T (correctly-rounded sqrt then divide); ≤1 ULP for f32/f64. Negative → NaN, 0 → +inf (IEEE). Same-hardware bit-stable."

determinism: same_hardware_bitwise
```

---

## powi  (y = x^n, integer exponent)

Element-wise integer power `out[i] = x[i].powi(exp)` with a single `i32` exponent shared across all
elements (`ops.rs:1511`). `Float::powi` is repeated multiplication, handling negative and zero base
correctly. The exponent rides `OpParams::PowI { exp }`. Contiguous, zero-offset; fresh contiguous
output, same shape/dtype. Bandwidth-bound; per-element cost grows with `|exp|` but is `O(log|exp|)`.

```fkc
kernel: powi
op_kind: PowIElementwise
blurb: "Element-wise integer power y = x.powi(exp) (uniform i32 exponent); contiguous same-shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::powi"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: PowI
    fields:
      exp: { kind: i32, note: "uniform integer exponent applied to every element" }

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
  max_ulp: 2
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Float::powi (repeated multiply, O(log|exp|) products) at T; rounding error accumulates with |exp| (a few ULP). Handles negative/zero base exactly per IEEE. Same-hardware bit-stable."

determinism: same_hardware_bitwise
```

---

## add  (out = a + b, same-shape)

Element-wise addition `out[i] = a[i] + b[i]` (`ops.rs:291`). `assert_same_shape` (`ops.rs:277`):
**exact dims equality, NO broadcasting** (broadcasting `add` is the separate `broadcast_add`
kernel, not in this family). Two contiguous, zero-offset inputs walked with `zip`; fresh contiguous
output = `a`'s shape and dtype. Both inputs share dtype. Bandwidth-bound: two reads + one write.

```fkc
kernel: add
op_kind: AddElementwise
blurb: "Element-wise addition out = a + b; same-shape (no broadcast); contiguous; fresh output."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::add"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=b
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=a
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: same_as(a)
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
  bytes_moved: "3 * n * dtype_bytes"   # read a + b, write out
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Single IEEE add a+b at T; correctly-rounded ≤0.5 ULP for f32/f64. bf16/f16 round at T (no f32 widening in the reference). NaN/inf per IEEE. Bit-exact same-hardware."

determinism: same_hardware_bitwise
```

---

## sub  (out = a - b, same-shape)

Element-wise subtraction `out[i] = a[i] - b[i]` (`ops.rs:303`). Same-shape only (no broadcast).
Two contiguous, zero-offset inputs; fresh contiguous output = `a`'s shape/dtype. Bandwidth-bound.

```fkc
kernel: sub
op_kind: SubElementwise
blurb: "Element-wise subtraction out = a - b; same-shape (no broadcast); contiguous; fresh output."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::sub"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=b
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=a
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: same_as(a)
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
  bytes_moved: "3 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Single IEEE subtract a-b at T; correctly-rounded ≤0.5 ULP for f32/f64. bf16/f16 round at T. Catastrophic cancellation possible for near-equal operands (inherent, not a kernel defect). Bit-exact same-hardware."

determinism: same_hardware_bitwise
```

---

## mul  (out = a * b, same-shape)

Element-wise multiplication `out[i] = a[i] * b[i]` (`ops.rs:315`). Same-shape only (no broadcast).
Two contiguous, zero-offset inputs; fresh contiguous output = `a`'s shape/dtype. Bandwidth-bound.

```fkc
kernel: mul
op_kind: MulElementwise
blurb: "Element-wise multiplication out = a * b; same-shape (no broadcast); contiguous; fresh output."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::mul"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=b
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=a
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: same_as(a)
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
  bytes_moved: "3 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Single IEEE multiply a*b at T; correctly-rounded ≤0.5 ULP for f32/f64. bf16/f16 round at T. NaN/inf per IEEE (0*inf = NaN). Bit-exact same-hardware."

determinism: same_hardware_bitwise
```

---

## div  (out = a / b, same-shape)

Element-wise division `out[i] = a[i] / b[i]` (`ops.rs:327`). Same-shape only (no broadcast).
Follows IEEE inf/NaN rules (`x/0 → ±inf`, `0/0 → NaN`). Two contiguous, zero-offset inputs; fresh
contiguous output = `a`'s shape/dtype. Bandwidth-bound.

```fkc
kernel: div
op_kind: DivElementwise
blurb: "Element-wise division out = a / b; same-shape (no broadcast); IEEE inf/NaN; contiguous."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::div"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=b
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=a
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: same_as(a)
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
  bytes_moved: "3 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Single IEEE divide a/b at T; correctly-rounded ≤0.5 ULP for f32/f64. bf16/f16 round at T. x/0 → ±inf, 0/0 → NaN (IEEE). Bit-exact same-hardware."

determinism: same_hardware_bitwise
```

---

## maximum  (out = max(a, b), same-shape)

Element-wise tensor maximum `out[i] = max(a[i], b[i])` via `f32::max`/`min`-style NaN-as-missing
(`ops.rs:1822`). Same-shape only (no broadcast). Two contiguous, zero-offset inputs; fresh
contiguous output = `a`'s shape/dtype. Bandwidth-bound.

```fkc
kernel: maximum
op_kind: MaximumElementwise
blurb: "Element-wise tensor max out = max(a, b); NaN-as-missing; same-shape; contiguous."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::maximum"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=b
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=a
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: same_as(a)
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
  bytes_moved: "3 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact select of the larger operand (no arithmetic); bit-exact. NaN-as-missing (f32::max semantics): max(x, NaN) = x. Bit-exact same-hardware."

determinism: same_hardware_bitwise
```

---

## minimum  (out = min(a, b), same-shape)

Element-wise tensor minimum `out[i] = min(a[i], b[i])` with NaN-as-missing (`ops.rs:1834`).
Same-shape only (no broadcast). Two contiguous, zero-offset inputs; fresh contiguous output =
`a`'s shape/dtype. Bandwidth-bound.

```fkc
kernel: minimum
op_kind: MinimumElementwise
blurb: "Element-wise tensor min out = min(a, b); NaN-as-missing; same-shape; contiguous."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::minimum"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=b
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=a
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: same_as(a)
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
  bytes_moved: "3 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact select of the smaller operand (no arithmetic); bit-exact. NaN-as-missing (f32::min semantics): min(x, NaN) = x. Bit-exact same-hardware."

determinism: same_hardware_bitwise
```

---

## rem  (out = a − ⌊a/b⌋·b, PyTorch remainder, same-shape)

Element-wise remainder, **PyTorch convention** `out[i] = a[i] - floor(a[i] / b[i]) * b[i]` — the
sign of the result matches the **divisor** (`ops.rs:1771`). Differs from C99 `fmod` (sign of
dividend; `f32::%` is also fmod-style). Same-shape only (a local `assert_eq` on dims, identical
semantics to `assert_same_shape`). Two contiguous, zero-offset inputs; fresh contiguous output =
`a`'s shape/dtype. Bandwidth-bound.

```fkc
kernel: rem
op_kind: RemElementwise
blurb: "Element-wise PyTorch remainder out = a - floor(a/b)·b (sign of divisor); same-shape; contiguous."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::rem"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=b
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=a
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: same_as(a)
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
  bytes_moved: "3 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 1
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "PyTorch remainder a - floor(a/b)·b at T (divide + floor + fma); sign matches divisor (NOT C99 fmod). A few ULP from the divide/floor/multiply chain. b=0 → NaN/inf per IEEE. Bit-exact same-hardware."

determinism: same_hardware_bitwise
```

---

## pow  (out = a^b, real exponent, same-shape)

Element-wise binary power `out[i] = a[i].powf(b[i])` — real (per-element) exponent (`ops.rs:1788`).
Distinct from `powi` (uniform `i32` exponent). IEEE NaN rules (e.g. `pow(-2, 0.5) = NaN`).
Same-shape only (no broadcast). Two contiguous, zero-offset inputs; fresh contiguous output =
`a`'s shape/dtype. Bandwidth-bound; transcendental per element.

```fkc
kernel: pow
op_kind: PowElementwise
blurb: "Element-wise binary power out = a.powf(b) (real exponent); same-shape; IEEE NaN; contiguous."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::pow"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=b
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=a
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: same_as(a)
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
  bytes_moved: "3 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 2
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Transcendental powf(a, b) at T via platform math library (typically ≤2 ULP for f32/f64). pow(<0, non-integer) → NaN, IEEE special cases per powf. Same-hardware bit-stable (same libm)."

determinism: same_hardware_bitwise
```

---

## add_scalar  (y = x + c)

Element-wise scalar add `out[i] = x[i] + c` with `c: f64` coerced to `T` via `cst` (`ops.rs:1495`).
Maps onto `OpKind::Affine` (`mul = 1, add = c`), so the param carrier is `OpParams::Affine`.
Single input; the scalar rides the params, not a second operand. Contiguous, zero-offset; fresh
contiguous output, same shape/dtype. Bandwidth-bound: one read, one write.

```fkc
kernel: add_scalar
op_kind: Affine
blurb: "Element-wise scalar add y = x + c (Affine mul=1, add=c); single input; contiguous same-shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::add_scalar"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Affine
    fields:
      mul: { kind: f64, constraint: "== 1.0", note: "add_scalar fixes mul = 1" }
      add: { kind: f64, note: "the scalar c, coerced to T via cst" }

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
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read x, write out (scalar is a param, not a buffer)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "x + cst(c) at T; single IEEE add, ≤0.5 ULP for f32/f64 (plus the f64→T coercion of c). bf16/f16 round at T. Bit-exact same-hardware."

determinism: same_hardware_bitwise
```

---

## mul_scalar  (y = x · c)

Element-wise scalar multiply `out[i] = x[i] * c` with `c: f64` coerced to `T` via `cst`
(`ops.rs:1502`). Maps onto `OpKind::Affine` (`mul = c, add = 0`), param carrier `OpParams::Affine`.
Single input; scalar rides the params. Contiguous, zero-offset; fresh contiguous output, same
shape/dtype. Bandwidth-bound.

```fkc
kernel: mul_scalar
op_kind: Affine
blurb: "Element-wise scalar multiply y = x·c (Affine mul=c, add=0); single input; contiguous same-shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::mul_scalar"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Affine
    fields:
      mul: { kind: f64, note: "the scalar c, coerced to T via cst" }
      add: { kind: f64, constraint: "== 0.0", note: "mul_scalar fixes add = 0" }

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
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "x * cst(c) at T; single IEEE multiply, ≤0.5 ULP for f32/f64 (plus the f64→T coercion of c). bf16/f16 round at T. Bit-exact same-hardware."

determinism: same_hardware_bitwise
```

---

## clamp  (y = clamp(x, min, max))

Element-wise clamp `out[i] = clamp(x[i], min, max)` with `min, max: f64` coerced to `T`
(`ops.rs:1802`). Maps onto `OpKind::ClampElementwise`, param carrier `OpParams::Clamp { min, max }`.
Single input; bounds ride the params. Contiguous, zero-offset; fresh contiguous output, same
shape/dtype. Bandwidth-bound.

```fkc
kernel: clamp
op_kind: ClampElementwise
blurb: "Element-wise clamp y = clamp(x, min, max); single input; contiguous same-shape; fresh output."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::clamp"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Clamp
    fields:
      min: { kind: f64, note: "lower bound, coerced to T via cst" }
      max: { kind: f64, constraint: ">= min", note: "upper bound, coerced to T via cst" }

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
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Two exact comparison-selects against cst(min)/cst(max) (no arithmetic on x); bit-exact for in-range values, exact substitution at the bounds. NaN handling follows the clamp comparison order. Bit-exact same-hardware."

determinism: same_hardware_bitwise
```

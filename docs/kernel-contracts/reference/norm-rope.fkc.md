---
fkc_version: 1
provider:
  name: fuel-reference-backend
  backend: Cpu                       # the pure-Rust oracle runs on the CPU substrate
  kernel_source: "reference-oracle"  # the BindingEntry.kernel_source tag
  link_registry: fuel_reference_backend::fkc::ENTRY_POINTS  # §12.6 symbol → KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-reference-backend — normalization, softmax & RoPE kernel contracts

Reference (oracle) contracts for the last-dim **softmax / log-softmax** family, the affine-free
**LayerNorm / RMSNorm** family (forward + fused backward), and **rotary position embedding (RoPE)**.

> **Crate-wide invariants** (`fuel-reference-backend/src/lib.rs:68`, inventory "How to read"):
> `RefTensor<T>` is *always* a contiguous, row-major buffer + a `Shape`, carrying **no strides and
> no offset**. Every kernel here is therefore **contiguous-only, zero-offset** by construction —
> there is no strided/broadcast/offset/reverse-stride input path anywhere, so every operand
> declares `contiguous: required` with the other four layout flags `rejected`, and
> `awkward_layout_strategy: requires_contiguous`. The crate is a **correctness-first oracle, not a
> production path**: it is the bit-stable CPU reference the Judge audits other backends against, so
> every contract declares `bit_stable_on_same_hardware: true` / `determinism: same_hardware_bitwise`
> (deterministic single-threaded per-row loops, no atomics, fixed reduction order). All math is in
> the kernel's element type `T` (the inventory notes no f32-accum widening for these row loops);
> `eps` is taken as `f64` and coerced to `T` via `cst`. Every kernel is generic over
> `T: num_traits::Float`, monomorphized to **f32 / f64 / bf16 / f16**, and writes a **fresh
> contiguous output** of the input element-count (`RefTensor::from_vec`). Output dtype equals the
> input dtype (passthrough) for all nine kernels — none of them changes dtype.

> **Cost provenance.** Every cost block here is marked `provenance: judge_measured` (§4.4): these
> are oracle kernels whose absolute cost the Judge bootstraps by measurement. Where a FLOPs /
> bandwidth shape is *genuinely derivable from the op itself* a formula hint is given as the
> measurement's shape prior (a last-dim reduction is `O(n)` arithmetic over `n` elements, with the
> output a fresh `n`-element contiguous buffer); no launch-overhead or per-tier constants are
> fabricated — the Judge populates those. `n` denotes the product of all output elements;
> `last` denotes the last-dim extent; `dtype_bytes` the element width.

---

## softmax_last_dim  (numerically-stable softmax over the last dimension)

Numerically-stable softmax over the last dim; per-row max-subtract, exp, normalize.

Stable softmax along the last axis: per row (the product of all leading dims), subtract the row
max, exponentiate, accumulate the denominator, then divide. `out[r,i] = exp(x[r,i] - max_j x[r,j])
/ sum_j exp(x[r,j] - max_j x[r,j])` (`ops.rs:2484`). Three passes per row (max, exp+accumulate,
normalize). All math in `T`; the row-max subtraction is the standard overflow guard, so a row of
finite inputs cannot overflow `exp`. Input rank must be ≥ 1; a rank-1 input is treated as a single
row. Migrated to the fused registry as `SOFTMAX_LAST_DIM` (`FusedOpId(1)`); the last-dim axis is
implicit in the shape, so the op is parameterless. Contiguous, zero-offset oracle: any
non-contiguous producer must be contiguized by the planner first.

```fkc
kernel: softmax_last_dim
fused_op: SOFTMAX_LAST_DIM
blurb: "Numerically-stable softmax over the last dim; per-row max-subtract, exp, normalize."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::softmax_last_dim"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."          # rank >= 1; last dim is the softmax axis
  op_params:
    variant: SoftmaxLastDim   # FusedOpParams::SoftmaxLastDim (fused namespace; §3.7) — parameterless

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)        # element-shape preserved; symbolic extents carried through (§5.2)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured        # oracle; Judge bootstraps absolute cost (§4.4)
  class: normalization
  flops: "n"                        # ~O(n) elementwise+reduction over n elements (3 passes/row)
  bytes_moved: "2 * n * dtype_bytes"   # read x, write out
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }   # fresh n-element output

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "oracle reference; stable max-subtract softmax; all math in T (no f32-accum widening); fixed per-row reduction order."

determinism: same_hardware_bitwise
```

---

## softmax_last_dim_backward  (gradient of last-dim softmax)

Gradient of last-dim softmax: dx = y * (g - sum(y*g, last)).

Backward of `softmax_last_dim`. Inputs are the forward output `y` and the upstream gradient `g`
(same shape, exact-equality checked — no broadcasting); produces `dx = y * (g - sum_j(y_j * g_j))`
per row (`ops.rs:2529`). Two passes per row (dot product `sum(y·g)`, then the fused
subtract-and-scale). Closed-form fused gradient — does not re-decompose into separate mul/sub/sum
ops. Migrated as `SOFTMAX_LAST_DIM_BACKWARD` (`FusedOpId(7)`), parameterless, wired from the
forward entry via `BackwardKind::Fused`.

```fkc
kernel: softmax_last_dim_backward
fused_op: SOFTMAX_LAST_DIM_BACKWARD
blurb: "Gradient of last-dim softmax: dx = y * (g - sum(y*g, last))."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::softmax_last_dim_backward"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y                       # forward softmax output
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."
    - name: g                       # upstream gradient
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."
      shape_constraint: same_as=y   # exact dims equality, no broadcast
  op_params:
    variant: SoftmaxLastDimBackward   # FusedOpParams::SoftmaxLastDimBackward (parameterless)

return:
  outputs:
    - name: dx
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n): per-row dot + fused subtract-scale (2 passes)
  bytes_moved: "3 * n * dtype_bytes"   # read y + g, write dx
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "oracle reference; fused closed-form dx = y*(g - sum(y*g)); all math in T; fixed per-row reduction order."

determinism: same_hardware_bitwise
```

---

## log_softmax_last_dim  (numerically-stable log-softmax over the last dimension)

Numerically-stable log-softmax over the last dim; (x - max) - log(sum exp(x - max)).

Stable log-softmax along the last axis: `out[r,i] = (x[r,i] - max_j x[r,j]) - log(sum_j exp(x[r,j]
- max_j x[r,j]))` (`ops.rs:2895`). Three passes per row (row max, accumulate `sum(exp(x-max))`,
then subtract `max` and `log_sum`). The max-subtraction is the overflow guard for the `exp`.
**Primitive op** — `OpKind::LogSoftmaxLastDim` (`dispatch.rs:283`); **not** migrated to the fused
registry (per the inventory it is still a primitive `Op::LogSoftmaxLastDim`), so its cost compiles
to the **primitive** `CostFn` and it carries no op-params variant. Input rank must be ≥ 1.
Contiguous, zero-offset oracle.

```fkc
kernel: log_softmax_last_dim
op_kind: LogSoftmaxLastDim
blurb: "Numerically-stable log-softmax over the last dim; (x - max) - log(sum exp(x - max))."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::log_softmax_last_dim"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."
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
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n) elementwise+reduction over n elements (3 passes/row)
  bytes_moved: "2 * n * dtype_bytes"   # read x, write out
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "oracle reference; stable (x-max) - log(sum exp(x-max)); all math in T; fixed per-row reduction order."

determinism: same_hardware_bitwise
```

---

## log_softmax_last_dim_backward  (gradient of last-dim log-softmax)

Gradient of last-dim log-softmax: dx = g - exp(y) * sum(g, last).

Backward of `log_softmax_last_dim`. Inputs are the forward output `y = log_softmax(x)` and the
upstream gradient `g` (same shape, exact-equality checked); produces `dx_i = g_i - exp(y_i) *
sum_j g_j` per row (`ops.rs:2926`). Two passes per row (accumulate `sum(g)`, then the fused
correction). Note `exp(y_i)` recovers the softmax probability from the log-softmax output.
**Primitive op** — `OpKind::LogSoftmaxLastDimBackward` (`dispatch.rs:286`); **not** in the fused
registry, so it compiles to the primitive `CostFn` with no op-params variant.

```fkc
kernel: log_softmax_last_dim_backward
op_kind: LogSoftmaxLastDimBackward
blurb: "Gradient of last-dim log-softmax: dx = g - exp(y) * sum(g, last)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::log_softmax_last_dim_backward"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y                       # forward log-softmax output
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."
    - name: g                       # upstream gradient
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."
      shape_constraint: same_as=y
  op_params: { variant: None }

return:
  outputs:
    - name: dx
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n): per-row sum(g) + fused correction (2 passes)
  bytes_moved: "3 * n * dtype_bytes"   # read y + g, write dx
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "oracle reference; fused dx = g - exp(y)*sum(g); all math in T; fixed per-row reduction order."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim  (affine-free LayerNorm over the last dimension)

Affine-free LayerNorm over the last dim; (x - mean)/sqrt(var + eps), biased variance (/n).

LayerNorm without affine parameters: per row, `out = (x - mean) / sqrt(var + eps)` where `mean`
and `var` are taken along the last dim. **Variance is the biased estimator (divide by `n`, not
`n-1`)**, matching PyTorch's `LayerNorm` (`ops.rs:2646`). Three passes per row (mean, variance,
normalize). Affine (gamma/beta) is *deliberately out of this op* — the caller applies it as a
separate mul+add, keeping the primitive validatable in isolation. `eps: f64` coerced to `T`.
Asserts rank ≥ 1 and `last > 0`. Migrated as `LAYER_NORM_LAST_DIM` (`FusedOpId(4)`), carrying
`eps` in `FusedOpParams::LayerNormLastDim { eps }` (CSE keys on the eps bit pattern).

```fkc
kernel: layer_norm_last_dim
fused_op: LAYER_NORM_LAST_DIM
blurb: "Affine-free LayerNorm over the last dim; (x - mean)/sqrt(var + eps), biased variance (/n)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::layer_norm_last_dim"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."          # rank >= 1; last dim must be > 0 (checked)
  op_params:
    variant: LayerNormLastDim   # FusedOpParams::LayerNormLastDim { eps }
    fields:
      eps: { kind: f64, note: "added under the sqrt; coerced to T internally" }

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
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n): mean + variance + normalize (3 passes/row)
  bytes_moved: "2 * n * dtype_bytes"   # read x, write out
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "oracle reference; biased variance (/n) PyTorch-matching; affine applied separately; all math in T; fixed per-row reduction order."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_backward  (gradient of affine-free LayerNorm)

Gradient of affine-free LayerNorm: dx = rstd*(g - mean(g) - y*mean(g*y)), stats recomputed from x.

Fused backward of `layer_norm_last_dim`. Inputs are the original input `x` and the upstream
gradient `g` (same shape, exact-equality checked) plus `eps`; recomputes per-row `mean`, biased
`var`, `rstd = 1/sqrt(var+eps)`, then `dx_i = rstd * (g_i - mean(g) - y_i * mean(g*y))` where
`y_i = (x_i - mean) * rstd` (`ops.rs:2577`). Several passes per row (recompute mean/var, then
`mean(g)` and `mean(g*y)`, then the fused gradient). Closed-form fused gradient; affine is the
caller's separate concern. Migrated as `LAYER_NORM_LAST_DIM_BACKWARD` (`FusedOpId(8)`), carrying
`eps` in `FusedOpParams::LayerNormLastDimBackward { eps }`.

> Note the as-built backward takes the **original `x`** (not the forward output `y`) as its data
> input — it recomputes the normalization statistics internally, so it does not need the forward
> output buffer.

```fkc
kernel: layer_norm_last_dim_backward
fused_op: LAYER_NORM_LAST_DIM_BACKWARD
blurb: "Gradient of affine-free LayerNorm: dx = rstd*(g - mean(g) - y*mean(g*y)), stats recomputed from x."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::layer_norm_last_dim_backward"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x                       # original forward input (stats recomputed from it)
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."
    - name: g                       # upstream gradient
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."
      shape_constraint: same_as=x
  op_params:
    variant: LayerNormLastDimBackward   # FusedOpParams::LayerNormLastDimBackward { eps }
    fields:
      eps: { kind: f64, note: "same eps as the forward; coerced to T internally" }

return:
  outputs:
    - name: dx
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n): recompute mean/var + mean(g)/mean(g*y) + fused dx
  bytes_moved: "3 * n * dtype_bytes"   # read x + g, write dx
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "oracle reference; fused closed-form dx; biased variance (/n) recomputed from x; all math in T; fixed per-row reduction order."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim  (affine-free RMSNorm over the last dimension)

Affine-free RMSNorm over the last dim; x / sqrt(mean(x^2) + eps).

RMSNorm without affine parameters: per row, `out = x / sqrt(mean(x²) + eps)` (`ops.rs:2699`). Two
passes per row (accumulate `mean(x²)`, then scale by the reciprocal RMS `rrms = 1/sqrt(mean_sq +
eps)`). Unlike LayerNorm there is no mean-centering. `eps: f64` coerced to `T`. Asserts rank ≥ 1
and `last > 0`. Migrated as `RMS_NORM_LAST_DIM` (`FusedOpId(3)`), carrying `eps` in
`FusedOpParams::RmsNormLastDim { eps }`.

```fkc
kernel: rms_norm_last_dim
fused_op: RMS_NORM_LAST_DIM
blurb: "Affine-free RMSNorm over the last dim; x / sqrt(mean(x^2) + eps)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::rms_norm_last_dim"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."          # rank >= 1; last dim must be > 0 (checked)
  op_params:
    variant: RmsNormLastDim   # FusedOpParams::RmsNormLastDim { eps }
    fields:
      eps: { kind: f64, note: "added under the sqrt; coerced to T internally" }

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
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n): mean(x^2) + scale (2 passes/row)
  bytes_moved: "2 * n * dtype_bytes"   # read x, write out
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "oracle reference; x / sqrt(mean(x^2) + eps), no mean-centering; all math in T; fixed per-row reduction order."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_backward  (gradient of affine-free RMSNorm)

Gradient of affine-free RMSNorm: dx = r_rms*(g - x*sum(g*x)/(n*(mean_sq+eps))).

Fused backward of `rms_norm_last_dim`. Inputs are the original input `x` and the upstream gradient
`g_y` (same shape, exact-equality checked) plus `eps`; per row computes `s = sum_i(g_y_i * x_i)`,
`mean_sq = mean(x²)`, `r_rms = 1/sqrt(mean_sq + eps)`, then `dx_j = r_rms * (g_y_j - x_j * s /
(n*(mean_sq + eps)))` (`ops.rs:2749`). Two passes per row (the two reductions `sum(x²)` and
`sum(g·x)`, then the fused gradient). Closed-form fused gradient. Migrated as
`RMS_NORM_LAST_DIM_BACKWARD` (`FusedOpId(9)`), carrying `eps` in
`FusedOpParams::RmsNormLastDimBackward { eps }`.

> As with the LayerNorm backward, the as-built kernel takes the **original `x`** as its data input
> (it recomputes `mean(x²)` internally), not the forward output.

```fkc
kernel: rms_norm_last_dim_backward
fused_op: RMS_NORM_LAST_DIM_BACKWARD
blurb: "Gradient of affine-free RMSNorm: dx = r_rms*(g - x*sum(g*x)/(n*(mean_sq+eps)))."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::rms_norm_last_dim_backward"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x                       # original forward input (mean(x^2) recomputed from it)
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."
    - name: g_y                     # upstream gradient
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."
      shape_constraint: same_as=x
  op_params:
    variant: RmsNormLastDimBackward   # FusedOpParams::RmsNormLastDimBackward { eps }
    fields:
      eps: { kind: f64, note: "same eps as the forward; coerced to T internally" }

return:
  outputs:
    - name: dx
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n): sum(x^2) + sum(g*x) + fused dx (2 passes/row)
  bytes_moved: "3 * n * dtype_bytes"   # read x + g_y, write dx
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "oracle reference; fused closed-form dx; mean(x^2) recomputed from x; all math in T; fixed per-row reduction order."

determinism: same_hardware_bitwise
```

---

## rope  (rotary position embedding with caller-supplied cos/sin tables)

Rotary position embedding; rotate-halves of x using caller-supplied [seq, head_dim] cos/sin tables.

Fused rotary position embedding. `x` has shape `[..., seq, head_dim]` (rank ≥ 2, **head_dim must
be even**); `cos` and `sin` are `[seq, head_dim]` tables that **broadcast across the leading dims**
(the leading dims are iterated as `outer`, the table indexed only by `[seq, head_dim]`). The
broadcast is exact: `cos`/`sin` dims are asserted to equal `[seq, head_dim]` (not general NumPy
broadcasting). With `half = head_dim/2`, the rotate-halves form (`ops.rs:2801`):
`out[..,i]      = x0*cos[i]      - x1*sin[i]`,
`out[..,i+half] = x1*cos[i+half] + x0*sin[i+half]`  for `i in 0..half`, where `x0 = x[..,i]`,
`x1 = x[..,i+half]`. All math in `T`; pure elementwise rotation (no reduction, no overflow guard
needed). Migrated as `ROPE` (`FusedOpId(5)`); parameterless (`seq`/`head_dim` are recovered from
the input shapes), so `FusedOpParams::Rope` carries no fields.

> The cos/sin tables are honest separate inputs, **not** a sidecar / gather descriptor — they are
> ordinary contiguous operands the kernel indexes by `[seq, head_dim]` while iterating the leading
> dims. No FDX extension is required.

```fkc
kernel: rope
fused_op: ROPE
blurb: "Rotary position embedding; rotate-halves of x using caller-supplied [seq, head_dim] cos/sin tables."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::rope"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2.."          # [..., seq, head_dim]; head_dim must be even (checked)
      shape_constraint: "divisible(x.dim[-1], 2)"   # head_dim even
    - name: cos
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2              # exactly [seq, head_dim]; broadcast over x's leading dims
      shape_constraint: "dim[0]=x.dim[-2]; dim[1]=x.dim[-1]"
    - name: sin
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2              # exactly [seq, head_dim]; broadcast over x's leading dims
      shape_constraint: same_as=cos
  op_params:
    variant: Rope          # FusedOpParams::Rope (parameterless; seq/head_dim from input shapes)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(x)    # output is x's shape [..., seq, head_dim]; symbolic seq preserved
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n): pure elementwise rotate-halves (no reduction)
  bytes_moved: "2 * n * dtype_bytes"   # read x, write out (cos/sin tables re-read per leading row)
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "oracle reference; pure elementwise rotate-halves; cos/sin exact [seq, head_dim] broadcast; all math in T."

determinism: same_hardware_bitwise
```

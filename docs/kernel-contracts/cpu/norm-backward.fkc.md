---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                       # maps to BackendId::Cpu
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag
  link_registry: fuel_dispatch::fkc::CPU_ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — softmax / norm backward kernel contracts (family: norm-backward)

Portable CPU backward kernels for the last-dim softmax / log-softmax / layer-norm / rms-norm
family. All live in `fuel-cpu-backend/src/byte_kernels.rs` and are dispatched through the
production `fuel_dispatch::dispatch` CPU wrappers. Each kernel is a per-dtype monomorphization:
`f32`/`f64` compute natively; `bf16`/`f16` widen each element to **f32**, accumulate the
per-row reduction in **f32**, and narrow back on store (the load-bearing precision invariant the
inventory calls out).

Cross-cutting facts for every kernel in this file (from the inventory + as-built source):

- **Two inputs, one output**, all `[outer_count × last_dim]` contiguous row-major slices.
  Softmax / log-softmax backward take `(y, g)` where `y` is the forward output and `g` the
  upstream gradient; layer-norm / rms-norm backward take `(x, g)` where `x` is the forward
  *input* (stats are recomputed) plus an `eps`.
- **Layout: contiguous-only, offset 0, row-major.** None of these kernels consult a
  `Layout`/strides/offset — they call `as_slice()` and validate **element counts**
  (`y.len() == g.len() == out.len() == outer_count * last_dim`), erroring with the kernel name on
  mismatch. So every input is `requires_contiguous`; the pipelined executor's auto-Contiguize
  pass realizes any strided/broadcast/offset operand into a dense buffer *before* these kernels
  run (§4.3). Negative strides are likewise not walked — a reversed view is normalized upstream.
- **Output: pre-allocated, fully overwritten.** `out` is caller-allocated to the exact byte size
  and written `out[off+i] = …` for every element (no read of prior `out` contents, no aliasing
  with the inputs). Output dtype == input dtype; output shape == the gradient's shape.
- **Validation is a runtime element-count check** returning `Result` (never panics on the
  production path).
- **Cost provenance is `judge_measured`** for every kernel here (the Judge bootstraps it). Where a
  genuine bandwidth/FLOP shape is derivable from the op it is recorded as a formula *hint* in the
  cost-expression fields and `notes`, but the coefficients are owned by measurement, not authored:
  these are streaming row-reductions (two passes over `last_dim` per row), so they are
  **bandwidth-bound** at `≈ 3 · n · dtype_bytes` (read two inputs, write one output) with
  `O(n)` arithmetic — a hint the Judge refines, not a declared number.
- **dispatch key** for each is `(OpKind, [in_dtype, in_dtype, out_dtype], BackendId::Cpu)` +
  `kernel_source: "portable-cpu"`; the three dtype slots are equal per kernel (passthrough), so
  the per-dtype sections below differ only in their dtype list and `entry_point`.

---

## softmax_last_dim_backward_f32  (softmax last-dim backward, f32)

Softmax-last-dim backward — F32. Per row, computes `out_i = y_i · (g_i - Σ_j y_j·g_j)` where the
dot product `Σ_j y_j·g_j` is accumulated once per row in f32 and reused for every element. Two
contiguous `[outer_count × last_dim]` inputs `(y, g)` (forward softmax output and upstream
gradient), one contiguous output of identical dtype/shape, fully overwritten. Two passes over each
row's `last_dim` elements (dot accumulate, then write) — bandwidth-bound streaming reduction.
Known limitation: contiguous-only, no broadcasting; any strided/offset operand is contiguized by
the executor first.

```fkc
kernel: softmax_last_dim_backward_f32
op_kind: SoftmaxLastDimBackward
blurb: "Softmax last-dim backward (f32): out = y*(g - sum(y*g)) per row; contiguous; native f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::softmax_last_dim_backward_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: SoftmaxLastDim          # OpParams::SoftmaxLastDim (primitive namespace; §3.7)
    fields:
      outer_count: { kind: usize, note: "rows; outer_count*last_dim == y elem count" }
      last_dim:    { kind: usize, note: "reduction width" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous     # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured        # Judge bootstraps; hints below are derivable shape, not authored numbers
  class: normalization
  flops: "3 * n"                    # HINT: per element ~1 mul (dot) + 1 sub + 1 mul; O(n) two-pass
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read y + g, write out; bandwidth-bound
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic single-threaded row loop; fixed reduction order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                      # CPU primitive: importer applies PRIMITIVE_DETERMINISTIC_CPU family default (§12.4)
  notes: "native f32 dot + write; deterministic fixed-order per-row reduction."

determinism: same_hardware_bitwise
```

---

## softmax_last_dim_backward_f64  (softmax last-dim backward, f64)

Softmax-last-dim backward — F64. Identical algorithm to the f32 variant (`out_i = y_i · (g_i -
Σ_j y_j·g_j)` per row) computed natively in f64; the per-row dot accumulator is f64. Two
contiguous `[outer_count × last_dim]` inputs `(y, g)`, one contiguous f64 output, overwrite.
Bandwidth-bound two-pass row reduction. Contiguous-only.

```fkc
kernel: softmax_last_dim_backward_f64
op_kind: SoftmaxLastDimBackward
blurb: "Softmax last-dim backward (f64): out = y*(g - sum(y*g)) per row; contiguous; native f64."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::softmax_last_dim_backward_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: SoftmaxLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "3 * n"                    # HINT: O(n) two-pass; ~3 fp ops/elem
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read y + g, write out
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64 dot + write; deterministic fixed-order per-row reduction."

determinism: same_hardware_bitwise
```

---

## softmax_last_dim_backward_bf16  (softmax last-dim backward, bf16, f32 accumulator)

Softmax-last-dim backward — BF16. Same `out_i = y_i · (g_i - Σ_j y_j·g_j)` formula, but every
bf16 element is widened to **f32** before arithmetic, the per-row dot product is accumulated in
**f32**, and the result is narrowed back to bf16 on store (round-trip). Two contiguous
`[outer_count × last_dim]` bf16 inputs `(y, g)`, one contiguous bf16 output, overwrite.
Bandwidth-bound; the f32 accumulator is the precision invariant. Contiguous-only.

```fkc
kernel: softmax_last_dim_backward_bf16
op_kind: SoftmaxLastDimBackward
blurb: "Softmax last-dim backward (bf16): out = y*(g - sum(y*g)) per row; f32 accumulator; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::softmax_last_dim_backward_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: SoftmaxLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "3 * n"                    # HINT: O(n) two-pass; widen→f32 math→narrow per elem
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read y + g, write out (dtype_bytes=2)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic loop; same hardware → same result
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16 widened to f32, dot accumulated in f32, narrowed to bf16 on store; deterministic per-row reduction."

determinism: same_hardware_bitwise
```

---

## softmax_last_dim_backward_f16  (softmax last-dim backward, f16, f32 accumulator)

Softmax-last-dim backward — F16. Same algorithm as the bf16 variant with `half::f16` as the
storage type: widen each element to f32, accumulate the per-row dot in **f32**, narrow back to f16
on store. Two contiguous `[outer_count × last_dim]` f16 inputs `(y, g)`, one contiguous f16
output, overwrite. Bandwidth-bound. Contiguous-only.

```fkc
kernel: softmax_last_dim_backward_f16
op_kind: SoftmaxLastDimBackward
blurb: "Softmax last-dim backward (f16): out = y*(g - sum(y*g)) per row; f32 accumulator; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::softmax_last_dim_backward_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: SoftmaxLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "3 * n"                    # HINT: O(n) two-pass; widen→f32 math→narrow per elem
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read y + g, write out (dtype_bytes=2)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f16 widened to f32, dot accumulated in f32, narrowed to f16 on store; deterministic per-row reduction."

determinism: same_hardware_bitwise
```

---

## log_softmax_last_dim_backward_f32  (log-softmax last-dim backward, f32)

LogSoftmax-last-dim backward — F32. Per row, computes `out_i = g_i - exp(y_i)·Σ_j g_j`, where `y`
is the forward log-softmax output and `Σ_j g_j` is the per-row gradient sum accumulated once.
Note the f32 path accumulates `g_sum` in an **f32** accumulator (`let mut g_sum = 0.0f32`) even
for native f32 I/O. Two contiguous `[outer_count × last_dim]` inputs `(y, g)`, one contiguous f32
output, overwrite. Two passes per row (sum, then write with `exp`). Bandwidth-bound with a
per-element `exp`. Contiguous-only.

```fkc
kernel: log_softmax_last_dim_backward_f32
op_kind: LogSoftmaxLastDimBackward
blurb: "LogSoftmax last-dim backward (f32): out = g - exp(y)*sum(g) per row; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::log_softmax_last_dim_backward_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: LogSoftmaxLastDim       # OpParams::LogSoftmaxLastDim (primitive namespace; §3.7)
    fields:
      outer_count: { kind: usize, note: "rows" }
      last_dim:    { kind: usize, note: "reduction width" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "4 * n"                    # HINT: per elem ~ exp + mul + sub (+ sum pass); transcendental exp dominates
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read y + g, write out; bandwidth-bound
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic loop; note: f32 g_sum accumulator even for f32 I/O
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "out = g - exp(y)*sum(g); per-row g_sum in f32 accumulator; deterministic fixed-order reduction."

determinism: same_hardware_bitwise
```

---

## log_softmax_last_dim_backward_f64  (log-softmax last-dim backward, f64)

LogSoftmax-last-dim backward — F64. Same `out_i = g_i - exp(y_i)·Σ_j g_j` formula computed
natively in f64; the per-row gradient sum is accumulated in **f64** (unlike the f32 variant, which
uses an f32 accumulator). Two contiguous `[outer_count × last_dim]` inputs `(y, g)`, one
contiguous f64 output, overwrite. Bandwidth-bound with per-element `exp`. Contiguous-only.

```fkc
kernel: log_softmax_last_dim_backward_f64
op_kind: LogSoftmaxLastDimBackward
blurb: "LogSoftmax last-dim backward (f64): out = g - exp(y)*sum(g) per row; contiguous; native f64."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::log_softmax_last_dim_backward_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: LogSoftmaxLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "4 * n"                    # HINT: exp-dominated O(n) two-pass
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read y + g, write out
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "out = g - exp(y)*sum(g); native f64, f64 g_sum accumulator; deterministic per-row reduction."

determinism: same_hardware_bitwise
```

---

## log_softmax_last_dim_backward_bf16  (log-softmax last-dim backward, bf16, f32 accumulator)

LogSoftmax-last-dim backward — BF16 (f32 accumulator). Same `out_i = g_i - exp(y_i)·Σ_j g_j`
formula; each bf16 element is widened to f32, the per-row gradient sum and the `exp` are computed
in **f32**, and the result is narrowed to bf16 on store. Two contiguous `[outer_count × last_dim]`
bf16 inputs `(y, g)`, one contiguous bf16 output, overwrite. Bandwidth-bound. Contiguous-only.

```fkc
kernel: log_softmax_last_dim_backward_bf16
op_kind: LogSoftmaxLastDimBackward
blurb: "LogSoftmax last-dim backward (bf16): out = g - exp(y)*sum(g) per row; f32 accumulator; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::log_softmax_last_dim_backward_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: LogSoftmaxLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "4 * n"                    # HINT: exp-dominated; widen→f32 math→narrow per elem
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read y + g, write out (dtype_bytes=2)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16 widened to f32, g_sum + exp in f32, narrowed to bf16 on store; deterministic per-row reduction."

determinism: same_hardware_bitwise
```

---

## log_softmax_last_dim_backward_f16  (log-softmax last-dim backward, f16, f32 accumulator)

LogSoftmax-last-dim backward — F16 (f32 accumulator). Same algorithm as the bf16 variant with
`half::f16` storage: widen to f32, accumulate the per-row gradient sum and compute `exp` in
**f32**, narrow to f16 on store. Two contiguous `[outer_count × last_dim]` f16 inputs `(y, g)`,
one contiguous f16 output, overwrite. Bandwidth-bound. Contiguous-only.

```fkc
kernel: log_softmax_last_dim_backward_f16
op_kind: LogSoftmaxLastDimBackward
blurb: "LogSoftmax last-dim backward (f16): out = g - exp(y)*sum(g) per row; f32 accumulator; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::log_softmax_last_dim_backward_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: LogSoftmaxLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "4 * n"                    # HINT: exp-dominated; widen→f32 math→narrow per elem
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read y + g, write out (dtype_bytes=2)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f16 widened to f32, g_sum + exp in f32, narrowed to f16 on store; deterministic per-row reduction."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_backward_f32  (layer-norm last-dim backward, f32)

LayerNorm-last-dim backward — F32 (no affine params). Inputs `(x, g)` where `x` is the forward
*input* (the kernel recomputes the same statistics the forward used) and `g` is the upstream
gradient; carries `eps` so the recomputed stats match. Per row it computes `mean`, `var`, `rstd =
1/sqrt(var + eps)`, then `mean_g = mean(g)`, `mean_g_y = mean(g·y)` with `y_i = (x_i - μ)·rstd`,
and finally `out_i = rstd · (g_i - mean_g - y_i · mean_g_y)`. Native f32 throughout. Two
contiguous `[outer_count × last_dim]` inputs, one contiguous f32 output, overwrite. Multiple
streaming passes per row (mean, var, the two gradient means, the write). Bandwidth-bound with a
per-element `sqrt` per row. Contiguous-only.

```fkc
kernel: layer_norm_last_dim_backward_f32
op_kind: LayerNormLastDimBackward
blurb: "LayerNorm last-dim backward (f32): grad_x = rstd*(g - mean(g) - y*mean(g*y)); recomputes stats; eps; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::layer_norm_last_dim_backward_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim             # OpParams::NormLastDim (primitive namespace; shared by LayerNorm + RmsNorm; §3.7)
    fields:
      outer_count: { kind: usize, note: "rows" }
      last_dim:    { kind: usize, note: "normalized width; mean/var over this axis" }
      eps:         { kind: f64, note: "variance floor; matches forward; narrowed to f32 here" }

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
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "8 * n"                    # HINT: ~4 passes/row (mean, var, grad-means, write) + 1 sqrt/row
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read x + g, write out; bandwidth-bound
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic fixed-order multi-pass reduction
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f32; recomputes mean/var/rstd; eps narrowed to f32; deterministic per-row passes."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_backward_f64  (layer-norm last-dim backward, f64)

LayerNorm-last-dim backward — F64 (no affine). Same algorithm as the f32 variant computed
natively in f64 throughout (eps stays f64, no narrowing). Inputs `(x, g)` + `eps`; recomputes
`mean`, `var`, `rstd` and emits `out_i = rstd · (g_i - mean_g - y_i · mean_g_y)`. Two contiguous
`[outer_count × last_dim]` inputs, one contiguous f64 output, overwrite. Bandwidth-bound,
multi-pass. Contiguous-only.

```fkc
kernel: layer_norm_last_dim_backward_f64
op_kind: LayerNormLastDimBackward
blurb: "LayerNorm last-dim backward (f64): grad_x = rstd*(g - mean(g) - y*mean(g*y)); recomputes stats; eps; native f64."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::layer_norm_last_dim_backward_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }
      eps:         { kind: f64, note: "variance floor; used natively in f64" }

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
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "8 * n"                    # HINT: ~4 passes/row + 1 sqrt/row
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read x + g, write out
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64 throughout; eps used in f64; deterministic per-row passes."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_backward_bf16  (layer-norm last-dim backward, bf16, f32 accumulator)

LayerNorm-last-dim backward — BF16 (f32 accumulator). Same algorithm; each bf16 element is widened
to f32, all statistics (`mean`, `var`, `rstd`, `mean_g`, `mean_g_y`) and the final combine are
computed in **f32**, and `eps` is narrowed to f32; the result is narrowed back to bf16 on store.
Inputs `(x, g)` + `eps`; two contiguous `[outer_count × last_dim]` bf16 inputs, one contiguous
bf16 output, overwrite. Bandwidth-bound, multi-pass. Contiguous-only.

```fkc
kernel: layer_norm_last_dim_backward_bf16
op_kind: LayerNormLastDimBackward
blurb: "LayerNorm last-dim backward (bf16): grad_x = rstd*(g - mean(g) - y*mean(g*y)); f32 accumulator; eps; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::layer_norm_last_dim_backward_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }
      eps:         { kind: f64, note: "variance floor; narrowed to f32 here" }

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
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "8 * n"                    # HINT: ~4 passes/row + 1 sqrt/row; widen→f32 math→narrow
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read x + g, write out (dtype_bytes=2)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16 widened to f32, stats + combine in f32, eps narrowed to f32, narrowed to bf16 on store; deterministic per-row passes."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_backward_f16  (layer-norm last-dim backward, f16, f32 accumulator)

LayerNorm-last-dim backward — F16 (f32 accumulator). Same algorithm as the bf16 variant with
`half::f16` storage: widen to f32, compute all statistics and the combine in **f32**, narrow `eps`
to f32, narrow the result to f16 on store. Inputs `(x, g)` + `eps`; two contiguous `[outer_count ×
last_dim]` f16 inputs, one contiguous f16 output, overwrite. Bandwidth-bound, multi-pass.
Contiguous-only.

```fkc
kernel: layer_norm_last_dim_backward_f16
op_kind: LayerNormLastDimBackward
blurb: "LayerNorm last-dim backward (f16): grad_x = rstd*(g - mean(g) - y*mean(g*y)); f32 accumulator; eps; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::layer_norm_last_dim_backward_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }
      eps:         { kind: f64, note: "variance floor; narrowed to f32 here" }

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
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "8 * n"                    # HINT: ~4 passes/row + 1 sqrt/row; widen→f32 math→narrow
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read x + g, write out (dtype_bytes=2)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f16 widened to f32, stats + combine in f32, eps narrowed to f32, narrowed to f16 on store; deterministic per-row passes."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_backward_f32  (rms-norm last-dim backward, f32)

RmsNorm-last-dim backward — F32 (no affine params). Inputs `(x, g_y)` where `x` is the forward
input and `g_y` is the upstream gradient; carries `eps`. Per row it accumulates `sum_sq = Σ x²`
and `sum_gx = Σ g_y·x` in one pass, then `mean_sq = sum_sq/n`, `denom_sq = mean_sq + eps`, `r_rms
= 1/sqrt(denom_sq)`, `coeff = sum_gx / (n·denom_sq)`, and writes the closed form `out_j = r_rms ·
(g_y_j - x_j · coeff)`. Native f32. Two contiguous `[outer_count × last_dim]` inputs, one
contiguous f32 output, overwrite. Two passes per row (the fused reduction, then the write).
Bandwidth-bound with one `sqrt` per row. Contiguous-only.

```fkc
kernel: rms_norm_last_dim_backward_f32
op_kind: RmsNormLastDimBackward
blurb: "RmsNorm last-dim backward (f32): grad_x = r_rms*(g_y - x*sum(g_y*x)/(n*(mean_sq+eps))); eps; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rms_norm_last_dim_backward_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g_y
    - name: g_y
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim             # OpParams::NormLastDim (primitive namespace; shared with LayerNorm; §3.7)
    fields:
      outer_count: { kind: usize, note: "rows" }
      last_dim:    { kind: usize, note: "normalized width n; mean_sq over this axis" }
      eps:         { kind: f64, note: "mean-square floor; matches forward; narrowed to f32 here" }

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
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "5 * n"                    # HINT: 1 fused reduction pass (2 madd/elem) + 1 write pass + 1 sqrt/row
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read x + g_y, write out; bandwidth-bound
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic fixed-order reduction
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f32; one fused sum_sq/sum_gx pass; eps narrowed to f32; deterministic per-row reduction."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_backward_f64  (rms-norm last-dim backward, f64)

RmsNorm-last-dim backward — F64 (no affine). Same closed form as the f32 variant computed natively
in f64 throughout (eps stays f64). Inputs `(x, g_y)` + `eps`; accumulates `sum_sq`, `sum_gx` in
f64, then `out_j = r_rms · (g_y_j - x_j · coeff)`. Two contiguous `[outer_count × last_dim]`
inputs, one contiguous f64 output, overwrite. Bandwidth-bound, two-pass. Contiguous-only.

```fkc
kernel: rms_norm_last_dim_backward_f64
op_kind: RmsNormLastDimBackward
blurb: "RmsNorm last-dim backward (f64): grad_x = r_rms*(g_y - x*sum(g_y*x)/(n*(mean_sq+eps))); eps; native f64."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rms_norm_last_dim_backward_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g_y
    - name: g_y
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }
      eps:         { kind: f64, note: "mean-square floor; used natively in f64" }

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
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "5 * n"                    # HINT: 1 fused reduction pass + 1 write pass + 1 sqrt/row
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read x + g_y, write out
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64 throughout; eps used in f64; deterministic per-row reduction."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_backward_bf16  (rms-norm last-dim backward, bf16, f32 accumulator)

RmsNorm-last-dim backward — BF16 (f32 accumulator). Same closed form; each bf16 element is widened
to f32, `sum_sq`/`sum_gx` and all derived quantities are computed in **f32**, `eps` is narrowed to
f32, and the result is narrowed to bf16 on store. Inputs `(x, g_y)` + `eps`; two contiguous
`[outer_count × last_dim]` bf16 inputs, one contiguous bf16 output, overwrite. Bandwidth-bound,
two-pass. Contiguous-only.

```fkc
kernel: rms_norm_last_dim_backward_bf16
op_kind: RmsNormLastDimBackward
blurb: "RmsNorm last-dim backward (bf16): grad_x = r_rms*(g_y - x*sum(g_y*x)/(n*(mean_sq+eps))); f32 accumulator; eps."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rms_norm_last_dim_backward_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g_y
    - name: g_y
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }
      eps:         { kind: f64, note: "mean-square floor; narrowed to f32 here" }

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
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "5 * n"                    # HINT: 1 fused reduction pass + 1 write pass + 1 sqrt/row; widen→f32→narrow
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read x + g_y, write out (dtype_bytes=2)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16 widened to f32, sum_sq/sum_gx + derived in f32, eps narrowed to f32, narrowed to bf16 on store; deterministic per-row reduction."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_backward_f16  (rms-norm last-dim backward, f16, f32 accumulator)

RmsNorm-last-dim backward — F16 (f32 accumulator). Same algorithm as the bf16 variant with
`half::f16` storage: widen to f32, compute `sum_sq`/`sum_gx` and all derived values in **f32**,
narrow `eps` to f32, narrow the result to f16 on store. Inputs `(x, g_y)` + `eps`; two contiguous
`[outer_count × last_dim]` f16 inputs, one contiguous f16 output, overwrite. Bandwidth-bound,
two-pass. Contiguous-only.

```fkc
kernel: rms_norm_last_dim_backward_f16
op_kind: RmsNormLastDimBackward
blurb: "RmsNorm last-dim backward (f16): grad_x = r_rms*(g_y - x*sum(g_y*x)/(n*(mean_sq+eps))); f32 accumulator; eps."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rms_norm_last_dim_backward_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g_y
    - name: g_y
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim
    fields:
      outer_count: { kind: usize }
      last_dim:    { kind: usize }
      eps:         { kind: f64, note: "mean-square floor; narrowed to f32 here" }

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
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: normalization
  flops: "5 * n"                    # HINT: 1 fused reduction pass + 1 write pass + 1 sqrt/row; widen→f32→narrow
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read x + g_y, write out (dtype_bytes=2)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f16 widened to f32, sum_sq/sum_gx + derived in f32, eps narrowed to f32, narrowed to f16 on store; deterministic per-row reduction."

determinism: same_hardware_bitwise
```

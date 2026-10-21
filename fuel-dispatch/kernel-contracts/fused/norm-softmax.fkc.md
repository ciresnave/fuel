---
fkc_version: 1
provider:
  name: fuel-fused-registry
  backend: Cpu                       # maps to BackendId::Cpu (the always-built CPU fused kernels)
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag
  link_registry: fuel_dispatch::dispatch::FUSED_ENTRY_POINTS   # §12.6 symbol→KernelRef map (the single join target: register_default_fused_kernels)
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-fused-registry — norm / softmax fused-op contracts (crate: fused, family: norm)

Fused-op contracts for the last-dim softmax / RMS-norm / LayerNorm family and the backward helpers
that pair with it. Every kernel here is a registry-level **fused op** (`FusedOpId` in
`fuel-graph/src/registry.rs`) whose graph-side metadata (`shape_rule` / `dtype_rule` / `decompose` /
`backward` / `pattern`) lives in `fuel-graph/src/registry/*.rs` and whose always-built CPU payload
is dispatched through `fuel-dispatch::dispatch` / `register_default_fused_kernels`. Because these are
**fused** contracts (not primitive `op_kind` contracts), each one:

- declares `fused_op:` (a `FusedOps::*` id), never `op_kind:` — its cost compiles to the **fused**
  cost-fn shape `fn(&[Shape], &FusedOpParams, &BackendCapabilities) -> CostEstimate`
  (`fuel-dispatch/src/fused.rs:63`, **no `&[DType]` argument**), and its `return.*` rules compile
  to the graph-side `FusedOp.shape_rule` / `dtype_rule` (`registry.rs:104,108`), not a key check;
- names a `FusedOpParams::*` variant (the fused-param namespace, §3.7), never an `OpParams` variant.

Cross-cutting facts (apply to every kernel in this file unless a section says otherwise; from the
inventory `docs/kernel-contracts/_inventory/fused.md` + the as-built registry sources):

- **Layout: contiguous-only, offset-0, row-major.** The graph-side registry does not encode layout,
  and the kernel-side CPU wrappers in `fuel-dispatch/src/dispatch.rs` take a `_layouts: &[Layout]`
  and **ignore it** — they read the raw byte buffer via `cpu_input()` with no stride application.
  No fused kernel advertises `strided_input` (every `register_fused!` omits `caps`, so caps default
  to `KernelCaps::empty()`), so a non-contiguous / broadcast / non-zero-offset / reversed input is
  realized into a dense buffer by the executor's auto-Contiguize pass (`StridedInputPreferenceFilter`)
  **before** the kernel runs. Hence every input is `requires_contiguous`, `reverse_strides: rejected`
  (a reversed view is normalized upstream), and the planner inserts `Op::Contiguize` (itself an FKC
  kernel) and sums its cost (§4.3, §4.4).
- **dtype monomorphization.** CPU coverage is registered per-dtype `{F32, F64, BF16, F16}`; the
  task lists these as one dtype list on a single contract per fused op. `f32`/`f64` evaluate
  natively (`f64` for f64 input); `bf16`/`f16` widen each element to **f32**, accumulate the per-row
  reduction in **f32**, and narrow back on store — the load-bearing precision invariant the
  inventory calls out.
- **Output: single, caller-pre-allocated, fully overwritten, contiguous row-major; no aliasing.**
  Output dtype and shape follow the per-kernel `dtype_rule` / `shape_rule` (passthrough of the
  primary input for every kernel in this file). No fused kernel here is in-place or accumulating.
- **Precision (CPU).** All CPU fused kernels claim `bit_stable_on_same_hardware: true` with no
  static ULP / relative / absolute bound. The norm/softmax family shares
  `NORM_FAMILY_CPU_PRECISION`; `ReduceMaxToBackward` and `PowIBackward` carry their own
  (`REDUCE_MAX_TO_BACKWARD_CPU_PRECISION` / `POWI_BACKWARD_CPU_PRECISION`). Per the **2026-07-03
  maintainer decision (CireSnave)**, every section here declares `audited: true`: the FKC import is
  now the production registration path (`register_cpu_norm_softmax_fused_from_contract`), so each
  kernel's bit-stable claim **relocates** from its `*_CPU_PRECISION` const onto the contract — same
  author, same guarantee, so the flip moves the evidentiary bar, it does not lower it. Without the
  flip the import would lower to `UNAUDITED` and DOWNGRADE production metadata. The Judge still
  audits/refines these bit-stable seeds (§4.8).
- **Cost provenance is `judge_measured`** for every kernel here — the Judge bootstraps it from
  measurement. Only genuinely derivable bandwidth/FLOP *shape* hints are recorded in the
  cost-expression strings as priors. These are streaming row reductions: forwards read one input
  and write one output (**bandwidth-bound** ≈ `2 · n · dtype_bytes`); the two-input backwards read
  two inputs and write one output (≈ `3 · n · dtype_bytes`), arithmetic `O(n)`. The coefficients are
  owned by measurement, not authored. `n` denotes the product of output elements (= `outer_count ·
  last_dim`).
- **`decompose` semantics differ forward vs backward.** The forward entries (`SoftmaxLastDim`,
  `RmsNormLastDim`, `LayerNormLastDim`) lower to a primitive subgraph and (softmax/rms) carry a live
  pattern matcher. The five backward helpers have **no primitive decomposition** — their registry
  `decompose` panics and the executor falls through to `cpu_fallback` to the always-built CPU
  kernel; their pattern matcher is a stub (autograd emits `Op::Fused(<id>, _)` directly), and they
  are `NotDifferentiable` (higher-order grads are out of scope for the MVP). This is registry/graph
  behavior, not a kernel-contract field, and is noted in prose per kernel.

---

## softmax_last_dim  (fused numerically-stable softmax along the last dim)

Fused last-dim softmax `softmax(x)_i = exp(x_i - row_max) / Σ_j exp(x_j - row_max)` over a
contiguous `[outer_count, last_dim]` view (one row = the reduced last dim). Standard max-subtract
stabilization: per row find `row_max`, write `exp(x - row_max)`, accumulate the sum, then scale the
row by `1/sum`. `FusedOps::SOFTMAX_LAST_DIM` (id 1),
`fuel-graph/src/registry/softmax_last_dim.rs:25`. Single input `x`, single output of identical
dtype/shape, fresh contiguous buffer. `f32`/`f64` native; `bf16`/`f16` widen to f32 for the
reduction and narrow on store.

Algorithm/numerics: shape rule and dtype rule are both passthrough of input 0. The flat per-row sum
accumulator is order-dependent but deterministic, so the kernel is bit-stable on the same hardware.
This forward op is differentiable: its registry `backward` is `BackwardKind::Fused(SOFTMAX_LAST_DIM_BACKWARD)`
(autograd emits the paired backward helper below). It also lowers (`decompose`) to a 7-node
primitive subgraph `ReduceMaxTo → BroadcastTo → Sub → Exp → ReduceSumTo → BroadcastTo → Div`, and a
live pattern matcher recognizes that subgraph (single-consumer guards) so a user-written softmax is
fused — these are graph-rewrite facts, not kernel-contract fields. Limitation: contiguous-only — any
strided/broadcast/offset input is contiguized by the planner first.

```fkc
kernel: softmax_last_dim
fused_op: SOFTMAX_LAST_DIM
blurb: "Fused numerically-stable softmax along the last dim; row max-subtract, exp, normalize; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::softmax_last_dim_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: SoftmaxLastDim          # FusedOpParams::SoftmaxLastDim (fused namespace; parameterless; §3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)     # FusedOp.dtype_rule (registry.rs:108): = input 0
      shape_rule: same_as(x)         # FusedOp.shape_rule (registry.rs:104): = input 0
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured        # Judge bootstraps; hints below are derivable shape, not authored numbers
  class: normalization
  flops: "4 * n"                    # HINT: ~max + exp + sum + scale per element (transcendental exp dominates)
  bytes_moved: "2 * n * dtype_bytes"   # HINT: read x once, write out once (bandwidth-bound)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # NORM_FAMILY_CPU_PRECISION; F32 accumulator for half I/O
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                       # 2026-07-03 maintainer flip (CireSnave): relocates the NORM_FAMILY_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§12.4).
  notes: "stable softmax (row max-subtract); f32/f64 native, bf16/f16 widen to f32 + narrow on store. Flat f32 sum accumulator: deterministic, order-dependent, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim  (fused RMS normalization along the last dim, no affine)

Fused RMS norm `out_i = x_i / sqrt(mean(x²) + eps)` over a contiguous `[outer_count, last_dim]`
view, where `mean(x²) = (Σ x²) / last_dim` per row. `FusedOps::RMS_NORM_LAST_DIM` (id 3),
`fuel-graph/src/registry/rms_norm_last_dim.rs:29`. **No affine (gamma) parameter** — bare RMS norm;
an affine scale is a separate downstream op. Single input `x`, single output of identical
dtype/shape, fresh contiguous buffer. `eps` is an `f64` `FusedOpParams::RmsNormLastDim` field
(narrowed to `f32` inside the half/f32 kernels, used natively in the f64 kernel).

Algorithm/numerics: one reduction pass (sum-of-squares) + one write pass per row; shape and dtype
rules are passthrough of input 0. `f32`/`f64` native; `bf16`/`f16` compute the sum-of-squares and
reciprocal-sqrt in **f32** and narrow on store (the f32 accumulator is the precision invariant).
Deterministic, bit-stable on the same hardware. Differentiable: registry `backward` is
`BackwardKind::Fused(RMS_NORM_LAST_DIM_BACKWARD)`. It lowers (`decompose`) to a 7-node primitive
subgraph `Sqr → MeanDim → Reshape → AddScalar(eps) → Sqrt → BroadcastTo → Div`, with a live pattern
matcher (extracts `eps` from the `AddScalar`, single-consumer guards) — graph-rewrite facts, not
kernel-contract fields. Limitation: contiguous-only.

```fkc
kernel: rms_norm_last_dim
fused_op: RMS_NORM_LAST_DIM
blurb: "Fused RMS norm along the last dim, no affine: x / sqrt(mean(x^2) + eps); contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::rms_norm_last_dim_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: RmsNormLastDim          # FusedOpParams::RmsNormLastDim (fused namespace; §3.7)
    fields:
      eps: { kind: f64, note: "f64 fused-param; narrowed to f32 in the half/f32 kernels, native in f64" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)     # FusedOp.dtype_rule: = input 0
      shape_rule: same_as(x)         # FusedOp.shape_rule: = input 0
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
  flops: "3 * n"                    # HINT: x^2 + accumulate (reduce pass) + scale (write pass); 1 sqrt/row
  bytes_moved: "2 * n * dtype_bytes"   # HINT: read x once, write out once (bandwidth-bound)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # NORM_FAMILY_CPU_PRECISION; F32 accumulator for half I/O
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                       # 2026-07-03 maintainer flip (CireSnave): relocates the NORM-family *_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§12.4).
  notes: "no affine; eps f64 narrowed to f32 (native in f64). f32/f64 native, bf16/f16 sum-of-squares + rsqrt in f32, narrow on store. Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim  (fused layer normalization along the last dim, no affine)

Fused LayerNorm `out_i = (x_i - mean(x)) / sqrt(var(x) + eps)` over a contiguous
`[outer_count, last_dim]` view, with `mean = (Σ x)/last_dim` and `var = (Σ (x-mean)²)/last_dim` per
row. `FusedOps::LAYER_NORM_LAST_DIM` (id 4), `fuel-graph/src/registry/layer_norm_last_dim.rs:39`.
**No affine (gamma/beta) parameters** — bare LayerNorm. Single input `x`, single output of identical
dtype/shape, fresh contiguous buffer. `eps` is an `f64` `FusedOpParams::LayerNormLastDim` field
(narrowed to `f32` in the half/f32 kernels, native in f64).

Algorithm/numerics: two reduction passes per row (mean, then variance) + one write pass; shape and
dtype rules are passthrough of input 0. `f32`/`f64` native; `bf16`/`f16` compute mean, variance, and
reciprocal-sqrt in **f32** and narrow on store. Deterministic, bit-stable on the same hardware.
Differentiable: registry `backward` is `BackwardKind::Fused(LAYER_NORM_LAST_DIM_BACKWARD)`. It lowers
(`decompose`) to an 11-node mean/var/normalize primitive chain
(`MeanDim → Reshape → BroadcastTo → Sub → Sqr → MeanDim → Reshape → AddScalar(eps) → Sqrt →
BroadcastTo → Div`); its pattern matcher is a **stub (`None`)** — fusion fires only through the
builder, never through subgraph recognition (a one-way builder→fused migration). These are
graph-rewrite facts, not kernel-contract fields. Limitation: contiguous-only.

```fkc
kernel: layer_norm_last_dim
fused_op: LAYER_NORM_LAST_DIM
blurb: "Fused LayerNorm along the last dim, no affine: (x - mean) / sqrt(var + eps); contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::layer_norm_last_dim_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: LayerNormLastDim        # FusedOpParams::LayerNormLastDim (fused namespace; §3.7)
    fields:
      eps: { kind: f64, note: "f64 fused-param; narrowed to f32 in the half/f32 kernels, native in f64" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)     # FusedOp.dtype_rule: = input 0
      shape_rule: same_as(x)         # FusedOp.shape_rule: = input 0
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
  flops: "5 * n"                    # HINT: mean pass + variance pass + normalize pass; 1 sqrt/row
  bytes_moved: "2 * n * dtype_bytes"   # HINT: read x once, write out once (bandwidth-bound)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # NORM_FAMILY_CPU_PRECISION; F32 accumulator for half I/O
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                       # 2026-07-03 maintainer flip (CireSnave): relocates the NORM-family *_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§12.4).
  notes: "no affine; two-pass mean/variance; eps f64 narrowed to f32 (native in f64). bf16/f16 stats + rsqrt in f32, narrow on store. Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## softmax_last_dim_backward  (fused softmax last-dim backward)

Backward helper for `SoftmaxLastDim`. Per row computes `out_i = y_i · (g_i - Σ_j y_j·g_j)`, where
`y` is the forward softmax output and `g` the upstream gradient; the per-row dot `Σ_j y_j·g_j` is
accumulated once and reused for every element. `FusedOps::SOFTMAX_LAST_DIM_BACKWARD` (id 7),
`fuel-graph/src/registry/softmax_last_dim_backward.rs:45`. Two inputs `(y, upstream)`, single output
of `y`'s dtype/shape, fresh contiguous buffer. Two passes over each row's `last_dim`
(dot-accumulate, then write) — bandwidth-bound. `f32`/`f64` native; `bf16`/`f16` widen to f32, dot
in f32, narrow on store.

Algorithm/numerics: parameterless (`FusedOpParams::SoftmaxLastDimBackward`); shape and dtype rules
are passthrough of input 0 (`y`). Deterministic fixed-order per-row reduction, bit-stable on the
same hardware. As a backward helper it is `NotDifferentiable` (higher-order grads panic — MVP), its
registry `decompose` **panics** (no primitive form worth materializing — backends without a native
kernel use the always-built CPU kernel via `cpu_fallback`), and its pattern matcher is a **stub
(`None`)** (autograd emits this op directly). Those are graph-rewrite facts, not kernel-contract
fields. Limitation: contiguous-only.

```fkc
kernel: softmax_last_dim_backward
fused_op: SOFTMAX_LAST_DIM_BACKWARD
blurb: "Fused softmax last-dim backward: out = y*(g - sum(y*g)) per row; contiguous; f32 accumulator for half."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::softmax_last_dim_backward_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=upstream
    - name: upstream
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: SoftmaxLastDimBackward  # FusedOpParams::SoftmaxLastDimBackward (fused namespace; parameterless; §3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(y)     # FusedOp.dtype_rule: = input 0
      shape_rule: same_as(y)         # FusedOp.shape_rule: = input 0
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
  flops: "3 * n"                    # HINT: per elem ~1 mul (dot) + 1 sub + 1 mul; O(n) two-pass
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read y + upstream, write out; bandwidth-bound
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # NORM_FAMILY_CPU_PRECISION; deterministic per-row reduction
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                       # 2026-07-03 maintainer flip (CireSnave): relocates the NORM-family *_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§12.4).
  notes: "out = y*(g - sum(y*g)) per row; f32/f64 native, bf16/f16 widen to f32 (dot in f32), narrow on store. Deterministic fixed-order reduction, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_backward  (fused layer-norm last-dim backward)

Backward helper for `LayerNormLastDim` (no affine). Inputs `(x, upstream)` where `x` is the forward
**input** (the kernel recomputes the mean/variance statistics the forward used) and `upstream` is
the gradient; carries `eps` so the recomputed stats match. Per row it computes `mean`, `var`,
`rstd = 1/sqrt(var + eps)`, then `mean_g = mean(g)`, `mean_g_y = mean(g·y)` with `y_i = (x_i-μ)·rstd`,
and writes `out_i = rstd · (g_i - mean_g - y_i · mean_g_y)`. `FusedOps::LAYER_NORM_LAST_DIM_BACKWARD`
(id 8), `fuel-graph/src/registry/layer_norm_last_dim_backward.rs:21`. Single output of `x`'s
dtype/shape, fresh contiguous buffer. Multiple streaming passes per row; bandwidth-bound with one
`sqrt` per row.

Algorithm/numerics: `eps` is an `f64` `FusedOpParams::LayerNormLastDimBackward` field (narrowed to
f32 in the half/f32 kernels, native in f64); shape and dtype rules are passthrough of input 0 (`x`).
`f32`/`f64` native; `bf16`/`f16` compute all statistics and the combine in **f32** and narrow on
store. Deterministic fixed-order multi-pass reduction, bit-stable on the same hardware. As a backward
helper it is `NotDifferentiable`, its registry `decompose` **panics** (no primitive form;
`cpu_fallback`), and its pattern matcher is a **stub (`None`)** — graph-rewrite facts, not
kernel-contract fields. Limitation: contiguous-only.

```fkc
kernel: layer_norm_last_dim_backward
fused_op: LAYER_NORM_LAST_DIM_BACKWARD
blurb: "Fused LayerNorm last-dim backward: grad_x = rstd*(g - mean(g) - y*mean(g*y)); recomputes stats; eps; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::layer_norm_last_dim_backward_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=upstream
    - name: upstream
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: LayerNormLastDimBackward   # FusedOpParams::LayerNormLastDimBackward (fused namespace; §3.7)
    fields:
      eps: { kind: f64, note: "variance floor; matches forward; narrowed to f32 in half/f32, native in f64" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)     # FusedOp.dtype_rule: = input 0
      shape_rule: same_as(x)         # FusedOp.shape_rule: = input 0
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
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read x + upstream, write out; bandwidth-bound
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # NORM_FAMILY_CPU_PRECISION; deterministic multi-pass reduction
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                       # 2026-07-03 maintainer flip (CireSnave): relocates the NORM-family *_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§12.4).
  notes: "grad_x = rstd*(g - mean(g) - y*mean(g*y)); recomputes mean/var/rstd; eps f64 narrowed to f32 (native in f64). bf16/f16 stats + combine in f32, narrow on store. Deterministic per-row passes, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_backward  (fused rms-norm last-dim backward)

Backward helper for `RmsNormLastDim` (no affine), closed form. Inputs `(x, upstream)` where `x` is
the forward input and `upstream` is the gradient; carries `eps`. Per row it accumulates
`sum_sq = Σ x²` and `sum_gx = Σ g·x` in one pass, then `mean_sq = sum_sq/n`,
`denom_sq = mean_sq + eps`, `r_rms = 1/sqrt(denom_sq)`, `coeff = sum_gx / (n·denom_sq)`, and writes
`out_j = r_rms · (g_j - x_j · coeff)` — i.e. `r_rms · (g - x·sum(g·x)/(n·(mean_sq+eps)))`.
`FusedOps::RMS_NORM_LAST_DIM_BACKWARD` (id 9), `fuel-graph/src/registry/rms_norm_last_dim_backward.rs:19`.
Single output of `x`'s dtype/shape, fresh contiguous buffer. Two passes per row (fused reduction,
then write); bandwidth-bound with one `sqrt` per row.

Algorithm/numerics: `eps` is an `f64` `FusedOpParams::RmsNormLastDimBackward` field (narrowed to f32
in the half/f32 kernels, native in f64); shape and dtype rules are passthrough of input 0 (`x`).
`f32`/`f64` native; `bf16`/`f16` compute `sum_sq`/`sum_gx` and all derived values in **f32** and
narrow on store. Deterministic fixed-order reduction, bit-stable on the same hardware. As a backward
helper it is `NotDifferentiable`, its registry `decompose` **panics** (no primitive form;
`cpu_fallback`), and its pattern matcher is a **stub (`None`)** — graph-rewrite facts, not
kernel-contract fields. Limitation: contiguous-only.

```fkc
kernel: rms_norm_last_dim_backward
fused_op: RMS_NORM_LAST_DIM_BACKWARD
blurb: "Fused RmsNorm last-dim backward: grad_x = r_rms*(g - x*sum(g*x)/(n*(mean_sq+eps))); eps; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::rms_norm_last_dim_backward_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=upstream
    - name: upstream
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: RmsNormLastDimBackward  # FusedOpParams::RmsNormLastDimBackward (fused namespace; §3.7)
    fields:
      eps: { kind: f64, note: "mean-square floor; matches forward; narrowed to f32 in half/f32, native in f64" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)     # FusedOp.dtype_rule: = input 0
      shape_rule: same_as(x)         # FusedOp.shape_rule: = input 0
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
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read x + upstream, write out; bandwidth-bound
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # NORM_FAMILY_CPU_PRECISION; deterministic fused reduction
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                       # 2026-07-03 maintainer flip (CireSnave): relocates the NORM-family *_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§12.4).
  notes: "grad_x = r_rms*(g_y - x*sum(g_y*x)/(n*(mean_sq+eps))); one fused sum_sq/sum_gx pass; eps f64 narrowed to f32 (native in f64). bf16/f16 derived in f32, narrow on store. Deterministic per-row reduction, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## reduce_max_to_backward  (fused backward of Op::ReduceMaxTo)

Backward of the **primitive** `Op::ReduceMaxTo`: route the upstream gradient to the position(s)
where `x` equals its per-window max, with tied maxes sharing the gradient equally (fair-share
subgradient). `FusedOps::REDUCE_MAX_TO_BACKWARD` (id 10),
`fuel-graph/src/registry/reduce_max_to_backward.rs:28`. Inputs `(x, upstream)` — `x` is the original
reduce input (its shape is the output shape) and `upstream` carries the forward target (reduced)
shape. Single output `grad_x` of `x`'s dtype/shape, fresh contiguous buffer. Unique among the
backward helpers: it is the backward of a primitive, not of a fused forward, so there is no
`BackwardKind::Fused` edge from a forward entry — autograd reaches it directly from `Op::ReduceMaxTo`.

Algorithm/numerics: parameterless (`FusedOpParams::ReduceMaxToBackward`); shape and dtype rules are
passthrough of input 0 (`x`). The fair-share tie split is exact for `f32`/`f64`; `bf16`/`f16` widen
to f32 and narrow on store. Deterministic (fixed window/tie traversal order), bit-stable on the same
hardware. Carries its own precision constant `REDUCE_MAX_TO_BACKWARD_CPU_PRECISION` (and cost
`cost_reduce_max_to_backward_cpu`). As a backward helper it is `NotDifferentiable`, its registry
`decompose` **panics** (a primitive decomposition would need an equality primitive that does not
exist today; `cpu_fallback`), and its pattern matcher is a **stub (`None`)** — graph-rewrite facts,
not kernel-contract fields. Limitation: contiguous-only.

```fkc
kernel: reduce_max_to_backward
fused_op: REDUCE_MAX_TO_BACKWARD
blurb: "Fused backward of Op::ReduceMaxTo: route upstream to argmax positions, fair-share ties; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::reduce_max_to_backward_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
    - name: upstream
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(x) shape == upstream shape (forward target_shape)"
  op_params:
    variant: ReduceMaxToBackward     # FusedOpParams::ReduceMaxToBackward (fused namespace; parameterless; §3.7)

return:
  outputs:
    - name: grad_x
      dtype_rule: passthrough(x)     # FusedOp.dtype_rule: = input 0
      shape_rule: same_as(x)         # FusedOp.shape_rule: grad_x = x's shape (input 0)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured        # cost_reduce_max_to_backward_cpu; Judge bootstraps
  class: reduction
  flops: "2 * n"                    # HINT: per elem ~1 compare (is-max) + 1 routed write/share; O(n)
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read x + upstream, write grad_x; bandwidth-bound
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # REDUCE_MAX_TO_BACKWARD_CPU_PRECISION; deterministic fixed-order traversal
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                       # 2026-07-03 maintainer flip (CireSnave): relocates the NORM-family *_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§12.4).
  notes: "route upstream to per-window argmax, fair-share on ties; f32/f64 exact, bf16/f16 widen to f32 + narrow on store. Deterministic, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## powi_backward  (fused backward of Op::PowI)

Backward of the **primitive** `Op::PowI(exp)`: closed form `grad_x = exp · x^(exp-1) · upstream`,
elementwise. `FusedOps::POWI_BACKWARD` (id 15), `fuel-graph/src/registry/powi_backward.rs:26`.
Inputs `(x, upstream)` + an `i32` `exp` field. Single output `grad_x` of `x`'s dtype/shape, fresh
contiguous buffer. Like `ReduceMaxToBackward`, it is the backward of a primitive — autograd reaches
it directly from `Op::PowI` rather than through a `BackwardKind::Fused` edge. It replaces the prior
three-node autograd decomposition `PowI(n-1) → MulScalar → Mul` (which still lives in
`Tensor::backward` as the fallback when this fused kernel is unregistered for a backend).

Algorithm/numerics: elementwise `grad_x[i] = exp · x[i]^(exp-1) · upstream[i]` — a fully parallel
map with no cross-element reduction, so it is bandwidth-bound (read two inputs, write one output).
`exp` is the `i32` `FusedOpParams::PowIBackward` field; shape and dtype rules are passthrough of
input 0 (`x`). `f32`/`f64` native; `bf16`/`f16` widen to f32 for the power-and-multiply and narrow on
store. Deterministic (no reduction order dependence), bit-stable on the same hardware. Carries its
own precision constant `POWI_BACKWARD_CPU_PRECISION` (and cost `cost_powi_backward_cpu`). As a
backward helper it is `NotDifferentiable`, its registry `decompose` **panics** (the primitive
fallback lives in `Tensor::backward`, not the registry; `cpu_fallback`), and its pattern matcher is a
**stub (`None`)** — graph-rewrite facts, not kernel-contract fields. Limitation: contiguous-only.

```fkc
kernel: powi_backward
fused_op: POWI_BACKWARD
blurb: "Fused backward of Op::PowI: grad_x = exp * x^(exp-1) * upstream; elementwise; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::powi_backward_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=upstream
    - name: upstream
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: PowIBackward            # FusedOpParams::PowIBackward (fused namespace; §3.7)
    fields:
      exp: { kind: i32, note: "the integer exponent of the forward Op::PowI; grad uses x^(exp-1)" }

return:
  outputs:
    - name: grad_x
      dtype_rule: passthrough(x)     # FusedOp.dtype_rule: = input 0
      shape_rule: same_as(x)         # FusedOp.shape_rule: = input 0
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
  provenance: judge_measured        # cost_powi_backward_cpu; Judge bootstraps
  class: cheap_elementwise
  flops: "2 * n"                    # HINT: per elem ~ x^(exp-1) (integer power) + 2 muls; elementwise O(n)
  bytes_moved: "3 * n * dtype_bytes"  # HINT: read x + upstream, write grad_x; bandwidth-bound
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # POWI_BACKWARD_CPU_PRECISION; deterministic elementwise map
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                       # 2026-07-03 maintainer flip (CireSnave): relocates the NORM-family *_CPU_PRECISION bit-stable claim onto the contract (same author, same guarantee); FKC import is now production — false would DOWNGRADE to UNAUDITED (§12.4).
  notes: "grad_x = exp * x^(exp-1) * upstream elementwise; f32/f64 native, bf16/f16 widen to f32 + narrow on store. Deterministic (no reduction), bit-stable same hardware."

determinism: same_hardware_bitwise
```

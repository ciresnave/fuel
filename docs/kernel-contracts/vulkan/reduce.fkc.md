---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan
  kernel_source: "vulkan-slang"
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS   # §12.6 symbol → KernelRef map
  revision_base: "git:f41137b4"                           # folded into kernel_revision_hash (§4.7)
---

# fuel-vulkan-kernels — reduce family kernel contracts

The Vulkan **reduction / arg-reduction** kernels Fuel ships in its own Slang/SPIR-V stack: the
full-tensor `reduce{,_f16,_bf16,_f64}` scalar reduction, the per-row `reduce_last_dim{,_f16,_bf16,_f64}`
reduction, and the two arg-reduction families — `arg_reduce_last_dim_{f32,f16,bf16,f64}` (fast
last-dim argmax/argmin) and `arg_reduce_any_dim_{f32,f64,bf16,f16}` (the slow arbitrary-dim path).
Kernel sources live under `fuel-kernels-source/kernels/*.slang`; the Rust dispatch wrappers
(param packing, layout gating, validation) live in `fuel-vulkan-backend/src/lib.rs`. Inventory
source: `docs/kernel-contracts/_inventory/vulkan.md` (Reductions / arg-reductions / norms section,
crate `vulkan`, family `reduce`).

Cross-cutting facts for this family (from the inventory):

- **Layout: contiguous, zero-offset, row-major** on every input. Every kernel in this family walks
  its input via the linear dispatch index (full-reduce: a flat `0..n` strided walk; per-row /
  last-dim: a row-major `[n_rows, n_cols]` walk with a subgroup tree reduction). None consult a
  `Layout`/strides/offset internally — the **one exception** is `arg_reduce_any_dim_*`, which
  walks the *reduction dim* with a fixed `n_inner` stride over a logical `[n_outer, d_dim,
  n_inner]` view, but is still **not non-zero-offset capable** (the linear `[n_outer, n_inner]`
  output index plus the `n_inner` reduction stride; offset is handled by an upstream Contiguize).
  Every reduce/arg-reduce kernel therefore declares `awkward_layout_strategy:
  requires_contiguous` and the planner prices an inserted `Op::Contiguize` (itself an FKC kernel,
  §4.3) for any strided/broadcast/offset operand.
- **Output: pre-allocated, contiguous, fully overwritten** — with two atomic-write caveats made
  explicit below. The full `reduce` writes a single-element scalar; `reduce_last_dim` and both
  arg-reduce families write a `[n_rows]` / drop-the-dim vector. No aliasing with the input.
  - **`reduce_bf16` packed-output caveat:** the single bf16 result is written into the **low 16
    bits of `output[0]`** (one u32 slot); the wrapper sizes/zeros the slot.
  - **`reduce_last_dim_bf16` zero-init caveat:** the output buffer **must be zero-initialized by
    the wrapper before dispatch** — the kernel writes one bf16 half-word per row with
    `InterlockedOr` to avoid racing the adjacent half-word in the shared u32. This is a
    wrapper-side precondition, not an aliasing read of prior output content.
- **Accumulator / precision invariant:** f32 reduces/accumulates natively; f64 reduces in native
  `double`; **bf16/f16 accumulate in f32** and narrow on store (the load-bearing
  accumulator-promotion invariant). The value reductions (`reduce*`, `reduce_last_dim*`) are
  **not bit-stable cross-hardware and run-to-run order is fixed but device-defined** — a tree /
  subgroup reduction sums in a hardware-/subgroup-width-dependent order (FADD is non-associative).
  They are deterministic on the same hardware (fixed dispatch, no atomic FP accumulation here) but
  the static numeric bound is device-dependent, so the value reduces declare
  `bit_stable_on_same_hardware: true` with `audited: false` (UNAUDITED — the importer applies the
  family default; the cross-hardware caveat is in `notes`). The **arg-reductions select an index**
  with no FP accumulation: exact, ties resolve to the **lowest index** (numpy/PyTorch), NaN never
  wins, and the result is bitwise-identical on any hardware — so they declare `determinism:
  bitwise` and `audited: true`.
- **op_id selector:** the value reduces share one kernel per dtype, selecting Sum/Max/Min/Mean by
  `op_id` (0=sum 1=max 2=min 3=mean); the arg-reduces select argmax/argmin by `op_id` (0=argmax
  1=argmin). Each registrable contract is keyed by `OpKind` (SumReduce/MaxReduce/MinReduce/
  MeanReduce, or ArgMaxDim/ArgMinDim) — the four op-id'd OpKinds register the **same** Vulkan
  `KernelRef` at distinct keys (distinct keys ⇒ legal sibling registrations, not a
  `DuplicateKernelRef`, §10.10). The contract below is written per **Vulkan kernel entry point**
  (one `## ` section per inventory kernel name); `op_kind` names the representative OpKind and the
  prose records the op-id'd siblings.
- **Cost provenance:** every cost block is marked `judge_measured` (the Judge bootstraps the
  coefficients, §4.4). The `flops` / `bytes_moved` **hints** are retained because a single-pass
  reduction is genuinely reduction/bandwidth-bound and the structure is derivable from the op (one
  fold/compare per input element; read all input + write all output). The Judge refines the
  absolute numbers, the launch overhead (Vulkan command-buffer submit), and the per-tier memory.
  No overhead constant or cost number is fabricated.

---

## reduce  (full-tensor Sum / Max / Min / Mean reduction → scalar, f32)

Full-tensor reduction of an f32 tensor to a **single scalar**
(`fuel-kernels-source/kernels/reduce.slang:44`; wrapper `reduce_f32_bytes`
`fuel-vulkan-backend/src/lib.rs:7613`). The full-reduce fast path
(`dims.is_empty() || dims.len() == rank`) flattens the input to `n = product(shape)` and runs a
shared-memory **tree reduction** (256 shared slots) over the flat `0..n` walk; `op_id` selects the
fold: 0=Sum 1=Max 2=Min 3=Mean (Mean = Sum / n). The same `reduce_f32_bytes` wrapper also routes a
last-dim reduction to `reduce_last_dim` (its own contract); **this section is the full-tensor
scalar path**. Output is one f32 element. Known limitation: contiguous-only — any
strided/broadcast/offset operand must be contiguized by the planner first. The tree-reduction sum
order is device-defined, so the result is not bit-stable cross-hardware.

```fkc
kernel: reduce
op_kind: SumReduce          # op_id selector: also MaxReduce/MinReduce/MeanReduce at distinct keys
blurb: "Full-tensor Sum/Max/Min/Mean reduction to a scalar (f32); contiguous; shared-mem tree reduction."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::reduce"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; full reduce ⇒ dims empty or dims.len()==rank (n = product(shape))"
  op_params:
    variant: Reduce          # OpParams::Reduce (primitive namespace; §3.7); n,op_id packed by the wrapper
    fields:
      dims:    { kind: "Vec<usize>", constraint: "empty or all axes (full reduce); sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output is a scalar (reduced dims removed)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)         # same dtype as input (f32)
      shape_rule: reduce(input, dims, keepdim)   # all dims removed ⇒ scalar (1 element)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured     # Judge bootstraps the coefficients (§4.4)
  class: reduction
  flops: "n"                     # one fold per input element (single pass); derivable
  bytes_moved: "(n + 1) * 4"     # read all input (f32), write one f32 scalar; reduction/bandwidth-bound
  # overhead_ns (Vulkan submit) + per-tier memory: bootstrapped by the Judge (not fabricated)

precision:
  bit_stable_on_same_hardware: true   # fixed dispatch, no atomic FP accumulation
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                # UNAUDITED: importer applies the family default; cross-hardware caveat in notes
  notes: "f32 tree reduction (256 shared slots); Mean = Sum/n; Max/Min via comparison. NOT bit-stable cross-hardware (device-defined sum order)."

determinism: same_hardware_bitwise
```

---

## reduce_f16  (full-tensor Sum / Max / Min / Mean reduction → scalar, f16)

Full-tensor reduction of an f16 tensor to a single f16 scalar (`reduce.slang:44`; wrapper in the
`reduce_f32_bytes`…`:7926` family). Each element widens to f32, folds in an **f32 accumulator**
(the tree reduction runs in f32 space), then narrows back to f16 on store. `op_id` selects
Sum/Max/Min/Mean. Same scalar-output and contiguous-only contract as `reduce`. Not bit-stable
cross-hardware.

```fkc
kernel: reduce_f16
op_kind: SumReduce          # op_id selector: also MaxReduce/MinReduce/MeanReduce at distinct keys
blurb: "Full-tensor Sum/Max/Min/Mean reduction to a scalar (f16); f32 accumulator narrowed on store; contiguous."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::reduce_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; full reduce ⇒ dims empty or dims.len()==rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "empty or all axes (full reduce); sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output is a scalar" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)   # scalar (1 element)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n"
  bytes_moved: "n * 2 + 2"       # read all input (f16, 2B), write one f16 scalar; bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen f16→f32, accumulate in f32, narrow on store; Mean = Sum/n. NOT bit-stable cross-hardware (device-defined sum order)."

determinism: same_hardware_bitwise
```

---

## reduce_bf16  (full-tensor Sum / Max / Min / Mean reduction → scalar, bf16)

Full-tensor reduction of a bf16 tensor to a single bf16 scalar (`reduce.slang:44`). bf16 is stored
**packed (u16 pairs in u32 lanes)**, so the input element count **`n` must be even** (lane-pair
processing); the wrapper pads an odd count. Each bf16 widens to f32 (`bits << 16`, exact), folds in
an **f32 accumulator**, then narrows back to bf16 on store (RNE upper-16). The single bf16 result
is written into the **low 16 bits of `output[0]`** (one u32 slot the wrapper sizes/zeros). `op_id`
selects Sum/Max/Min/Mean. Contiguous-only; not bit-stable cross-hardware.

```fkc
kernel: reduce_bf16
op_kind: SumReduce          # op_id selector: also MaxReduce/MinReduce/MeanReduce at distinct keys
blurb: "Full-tensor Sum/Max/Min/Mean reduction to a bf16 scalar (low 16b of output[0]); f32 accumulator; n even; contiguous."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::reduce_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; full reduce ⇒ dims empty or dims.len()==rank; n (= product(shape)) even (lane-pair; wrapper pads odd)"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "empty or all axes (full reduce); sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output is a scalar" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)   # scalar; bf16 result in low 16 bits of output[0]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n"
  bytes_moved: "n * 2 + 4"       # read all input (bf16, 2B), write one u32 output slot; bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16→f32 exact widen (bits<<16), accumulate in f32, narrow on store RNE; result in low 16b of output[0]; n must be even. NOT bit-stable cross-hardware (device-defined sum order)."

determinism: same_hardware_bitwise
```

---

## reduce_f64  (full-tensor Sum / Max / Min / Mean reduction → scalar, f64)

Full-tensor reduction of an f64 tensor to a single f64 scalar (`reduce.slang:44`; wrapper
`…:7926`). Reduces in **native `double`** (no widen/narrow); `op_id` selects Sum/Max/Min/Mean
(Mean = Sum / n). Same scalar-output and contiguous-only contract as `reduce`. The tree-reduction
sum order is device-defined, so not bit-stable cross-hardware.

```fkc
kernel: reduce_f64
op_kind: SumReduce          # op_id selector: also MaxReduce/MinReduce/MeanReduce at distinct keys
blurb: "Full-tensor Sum/Max/Min/Mean reduction to a scalar (f64, native double); contiguous; tree reduction."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::reduce_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; full reduce ⇒ dims empty or dims.len()==rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "empty or all axes (full reduce); sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output is a scalar" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)   # scalar (1 element)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n"
  bytes_moved: "n * 8 + 8"       # read all input (f64, 8B), write one f64 scalar; bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native double accumulate; Mean = Sum/n; Max/Min via comparison. NOT bit-stable cross-hardware (device-defined sum order)."

determinism: same_hardware_bitwise
```

---

## reduce_last_dim  (per-row Sum / Max / Min / Mean along the last dim, f32)

Per-row reduction of an f32 `[n_rows, n_cols]` tensor along the **last dim**
(`fuel-kernels-source/kernels/reduce_last_dim.slang:56`). One workgroup per row runs a **subgroup
tree reduction** over the `n_cols` contiguous lane; `op_id` selects 0=Sum 1=Max 2=Min 3=Mean
(Mean = row-sum / n_cols). Output is a contiguous `[n_rows]` vector. Contiguous row-major input
only. The subgroup reduction order is device-defined, so not bit-stable cross-hardware.

```fkc
kernel: reduce_last_dim
op_kind: SumReduce          # op_id selector: also MaxReduce/MinReduce/MeanReduce at distinct keys
blurb: "Per-row Sum/Max/Min/Mean along the last dim (f32); contiguous [n_rows,n_cols]; subgroup reduction."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::reduce_last_dim"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; flattened to [n_rows, n_cols] (n_cols = last dim); reduce dim = last (dims == [rank-1])"
  op_params:
    variant: Reduce          # OpParams::Reduce; wrapper packs n_rows,n_cols,op_id
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly the last dim ([rank-1]); n_cols = input_shape[-1]" }
      keepdim: { kind: bool, note: "always false today; output drops the last dim ⇒ [n_rows]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)         # f32
      shape_rule: reduce(input, dims, keepdim)   # last dim removed ⇒ [n_rows]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_rows * n_cols"           # one fold per input element (single pass); derivable
  bytes_moved: "(n_rows * n_cols + n_rows) * 4"   # read all input (f32), write [n_rows]; bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32 subgroup reduction per row; Mean = row-sum/n_cols. NOT bit-stable cross-hardware (device-defined subgroup order)."

determinism: same_hardware_bitwise
```

---

## reduce_last_dim_f16  (per-row Sum / Max / Min / Mean along the last dim, f16)

Per-row last-dim reduction of an f16 `[n_rows, n_cols]` tensor (`reduce_last_dim.slang:56`). Each
element widens to f32, the subgroup reduction runs in an **f32 accumulator**, then narrows to f16
on store. `op_id` selects Sum/Max/Min/Mean. Output is a contiguous `[n_rows]` vector. Same
contiguous row-major contract; not bit-stable cross-hardware.

```fkc
kernel: reduce_last_dim_f16
op_kind: SumReduce          # op_id selector: also MaxReduce/MinReduce/MeanReduce at distinct keys
blurb: "Per-row Sum/Max/Min/Mean along the last dim (f16); f32 accumulator narrowed on store; contiguous."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::reduce_last_dim_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; flattened to [n_rows, n_cols]; reduce dim = last ([rank-1])"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly the last dim ([rank-1]); n_cols = input_shape[-1]" }
      keepdim: { kind: bool, note: "always false today; output ⇒ [n_rows]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)   # [n_rows]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_rows * n_cols"
  bytes_moved: "(n_rows * n_cols + n_rows) * 2"   # read all input (f16, 2B), write [n_rows]; bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen f16→f32, subgroup-reduce in f32, narrow on store; Mean = row-sum/n_cols. NOT bit-stable cross-hardware (device-defined subgroup order)."

determinism: same_hardware_bitwise
```

---

## reduce_last_dim_bf16  (per-row Sum / Max / Min / Mean along the last dim, bf16)

Per-row last-dim reduction of a bf16 `[n_rows, n_cols]` tensor (`reduce_last_dim.slang:56`). The
input is **packed (u16 lane-pairs in u32)**; each bf16 widens to f32 (`bits << 16`, exact),
the subgroup reduction runs in an **f32 accumulator**, then narrows to bf16 on store.
**Precondition: the output buffer MUST be zero-initialized by the wrapper before dispatch** — the
kernel writes one bf16 half-word per row with `InterlockedOr` so it does not race the adjacent
half-word in the shared u32. `op_id` selects Sum/Max/Min/Mean. Output is a contiguous `[n_rows]`
vector; not bit-stable cross-hardware.

```fkc
kernel: reduce_last_dim_bf16
op_kind: SumReduce          # op_id selector: also MaxReduce/MinReduce/MeanReduce at distinct keys
blurb: "Per-row Sum/Max/Min/Mean along the last dim (bf16); f32 accumulator; output pre-zeroed (InterlockedOr halves); contiguous."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::reduce_last_dim_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; flattened to [n_rows, n_cols] (packed lane-pairs); reduce dim = last ([rank-1])"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly the last dim ([rank-1]); n_cols = input_shape[-1]" }
      keepdim: { kind: bool, note: "always false today; output ⇒ [n_rows]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)   # [n_rows]
      layout_guarantee: contiguous
      aliasing: none                        # not in-place; wrapper PRE-ZEROES out (kernel InterlockedOr-writes halves)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_rows * n_cols"
  bytes_moved: "(n_rows * n_cols + n_rows) * 2"   # read all input (bf16, 2B), write [n_rows]; bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16→f32 exact widen, subgroup-reduce in f32, narrow on store; output must be pre-zeroed (InterlockedOr half-word writes). NOT bit-stable cross-hardware (device-defined subgroup order)."

determinism: same_hardware_bitwise
```

---

## reduce_last_dim_f64  (per-row Sum / Max / Min / Mean along the last dim, f64)

Per-row last-dim reduction of an f64 `[n_rows, n_cols]` tensor (`reduce_last_dim.slang:56`).
Reduces in **native `double`** via the per-row subgroup tree reduction; `op_id` selects
Sum/Max/Min/Mean (Mean = row-sum / n_cols). Output is a contiguous `[n_rows]` vector. Same
contiguous row-major contract; not bit-stable cross-hardware.

```fkc
kernel: reduce_last_dim_f64
op_kind: SumReduce          # op_id selector: also MaxReduce/MinReduce/MeanReduce at distinct keys
blurb: "Per-row Sum/Max/Min/Mean along the last dim (f64, native double); contiguous; subgroup reduction."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::reduce_last_dim_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; flattened to [n_rows, n_cols]; reduce dim = last ([rank-1])"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly the last dim ([rank-1]); n_cols = input_shape[-1]" }
      keepdim: { kind: bool, note: "always false today; output ⇒ [n_rows]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)   # [n_rows]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_rows * n_cols"
  bytes_moved: "(n_rows * n_cols + n_rows) * 8"   # read all input (f64, 8B), write [n_rows]; bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native double subgroup reduction per row; Mean = row-sum/n_cols. NOT bit-stable cross-hardware (device-defined subgroup order)."

determinism: same_hardware_bitwise
```

---

## arg_reduce_last_dim_f32  (argmax / argmin along the last dim, f32 → U32 index)

Index reduction along the **last dim** of an f32 `[n_rows, n_cols]` tensor
(`fuel-kernels-source/kernels/arg_reduce_last_dim_f32.slang:27`; wrapper
`arg_reduce_last_dim_bytes` `fuel-vulkan-backend/src/lib.rs:6096`). One workgroup per row runs a
tree reduction over `(val, idx)` pairs; `op_id` selects 0=argmax 1=argmin. **Lower index wins on
ties** (numpy/PyTorch). Input dtype f32; **output dtype U32** (one index per row, `[n_rows]`).
Index selection is exact (no FP accumulation), so the result is bitwise-identical on any hardware.
Contiguous row-major input only.

```fkc
kernel: arg_reduce_last_dim_f32
op_kind: ArgMaxDim          # op_id selector: also ArgMinDim at a distinct key (op_id 0=argmax 1=argmin)
blurb: "Argmax/argmin along the last dim (f32 in, U32 index out); lowest index wins ties; contiguous; tree reduction."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::arg_reduce_last_dim_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; flattened to [n_rows, n_cols] (outer_count rows, last_dim cols); reduce dim = last ([rank-1])"
  op_params:
    variant: Reduce          # OpParams::Reduce reused for ArgMaxDim/ArgMinDim; dim = last; wrapper packs outer_count,last_dim,op_id
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly the last dim ([rank-1]); last_dim = input_shape[-1]" }
      keepdim: { kind: bool, note: "always false; output drops the last dim ⇒ [n_rows]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)                 # output is always U32 indices
      shape_rule: reduce(input, dims, keepdim)   # last dim removed ⇒ [n_rows]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_rows * n_cols"           # one compare per input element (single pass); derivable
  bytes_moved: "n_rows * n_cols * 4 + n_rows * 4"   # read f32 input (4B), write U32 indices (4B); bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true   # integer index selection; exact, no rounding
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact index selection (no FP accumulation); ties resolve to the lowest index; bitwise-identical on any hardware for given input values."

determinism: bitwise
```

---

## arg_reduce_last_dim_f16  (argmax / argmin along the last dim, f16 → U32 index)

Last-dim index reduction of an f16 `[n_rows, n_cols]` tensor (`arg_reduce_last_dim_f32.slang:27`
family). Values are **lane-selected from the packed half input** and compared per row; `op_id`
selects argmax/argmin; **lower index wins on ties**. Input f16; **output U32** (`[n_rows]`). Exact
index selection ⇒ bitwise-identical on any hardware. Contiguous row-major input only.

```fkc
kernel: arg_reduce_last_dim_f16
op_kind: ArgMaxDim          # op_id selector: also ArgMinDim at a distinct key
blurb: "Argmax/argmin along the last dim (f16 in, U32 index out); lowest index wins ties; lane-select from packed; contiguous."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::arg_reduce_last_dim_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; flattened to [n_rows, n_cols]; reduce dim = last ([rank-1])"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly the last dim ([rank-1]); last_dim = input_shape[-1]" }
      keepdim: { kind: bool, note: "always false; output ⇒ [n_rows]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: reduce(input, dims, keepdim)   # [n_rows]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_rows * n_cols"
  bytes_moved: "n_rows * n_cols * 2 + n_rows * 4"   # read f16 input (2B), write U32 indices (4B); bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact index selection (lane-select from packed f16, no FP accumulation); ties resolve to the lowest index; bitwise-identical on any hardware."

determinism: bitwise
```

---

## arg_reduce_last_dim_bf16  (argmax / argmin along the last dim, bf16 → U32 index)

Last-dim index reduction of a bf16 `[n_rows, n_cols]` tensor (`arg_reduce_last_dim_f32.slang:27`
family; wrapper `arg_reduce_last_dim_bytes` `:6096`). Values are **lane-selected from the packed
bf16 input**; **`last_dim` must be even** (lane-pair; the wrapper bails with a typed error
otherwise, `lib.rs:6106`). `op_id` selects argmax/argmin; **lower index wins on ties**. Input
bf16; **output U32** (`[n_rows]`). Exact index selection ⇒ bitwise-identical on any hardware.
Contiguous row-major input only.

```fkc
kernel: arg_reduce_last_dim_bf16
op_kind: ArgMaxDim          # op_id selector: also ArgMinDim at a distinct key
blurb: "Argmax/argmin along the last dim (bf16 in, U32 index out); last_dim even; lowest index wins ties; contiguous."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::arg_reduce_last_dim_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; flattened to [n_rows, n_cols] (packed lane-pairs); reduce dim = last ([rank-1]); last_dim even"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly the last dim ([rank-1]); last_dim = input_shape[-1]; last_dim % 2 == 0" }
      keepdim: { kind: bool, note: "always false; output ⇒ [n_rows]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: reduce(input, dims, keepdim)   # [n_rows]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_rows * n_cols"
  bytes_moved: "n_rows * n_cols * 2 + n_rows * 4"   # read bf16 input (2B), write U32 indices (4B); bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact index selection (lane-select from packed bf16, no FP accumulation); last_dim must be even; ties resolve to the lowest index; bitwise-identical on any hardware."

determinism: bitwise
```

---

## arg_reduce_last_dim_f64  (argmax / argmin along the last dim, f64 → U32 index)

Last-dim index reduction of an f64 `[n_rows, n_cols]` tensor (`arg_reduce_last_dim_f32.slang:27`
family). Compares native `double` values per row; `op_id` selects argmax/argmin; **lower index
wins on ties**. Input f64; **output U32** (`[n_rows]`). Exact index selection ⇒ bitwise-identical
on any hardware. Contiguous row-major input only.

```fkc
kernel: arg_reduce_last_dim_f64
op_kind: ArgMaxDim          # op_id selector: also ArgMinDim at a distinct key
blurb: "Argmax/argmin along the last dim (f64 in, U32 index out); lowest index wins ties; contiguous."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::arg_reduce_last_dim_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; flattened to [n_rows, n_cols]; reduce dim = last ([rank-1])"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly the last dim ([rank-1]); last_dim = input_shape[-1]" }
      keepdim: { kind: bool, note: "always false; output ⇒ [n_rows]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: reduce(input, dims, keepdim)   # [n_rows]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_rows * n_cols"
  bytes_moved: "n_rows * n_cols * 8 + n_rows * 4"   # read f64 input (8B), write U32 indices (4B); bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact index selection (native double compare, no FP accumulation); ties resolve to the lowest index; bitwise-identical on any hardware."

determinism: bitwise
```

---

## arg_reduce_any_dim_f32  (argmax / argmin along an arbitrary dim, f32 → U32 index)

Index reduction along an **arbitrary (non-last) dim** — the slow path
(`fuel-kernels-source/kernels/arg_reduce_any_dim_f32.slang:31`; wrapper `arg_reduce_any_dim_bytes`
`fuel-vulkan-backend/src/lib.rs:6372`). The input is viewed logically as `[n_outer, d_dim,
n_inner]`; **one thread per output element** serially scans the reduction dim with **stride
`n_inner`** (so this kernel walks a non-unit stride over the reduction axis — the only
strided-walk kernel in this family, though it remains non-offset-capable). `op_id` selects
0=argmax 1=argmin; **lower index wins on ties**. Input f32; **output U32**, dropping the reduced
dim (shape `[n_outer, n_inner]`, contiguous). Exact index selection ⇒ bitwise-identical on any
hardware.

```fkc
kernel: arg_reduce_any_dim_f32
op_kind: ArgMaxDim          # op_id selector: also ArgMinDim at a distinct key
blurb: "Argmax/argmin along an arbitrary dim (f32 in, U32 index out); serial scan stride n_inner; lowest index wins ties."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::arg_reduce_any_dim_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      # Walks the reduction dim with stride n_inner over the logical [n_outer, d_dim, n_inner] view,
      # but the physical buffer is contiguous row-major and non-offset-capable (offset → upstream Contiguize).
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; logical [n_outer, d_dim, n_inner]; dim = dims[0] (single reduce dim) in 0..rank; n_inner = product(dims after `dim`)"
  op_params:
    variant: Reduce          # OpParams::Reduce reused for ArgMaxDim/ArgMinDim; dim = dims[0]; wrapper packs n_outer,n_inner,d_dim,op_id
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly one dim; dim in 0..input_rank" }
      keepdim: { kind: bool, note: "always false; output drops dim ⇒ [n_outer, n_inner]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)                 # output is always U32 indices
      shape_rule: reduce(input, dims, keepdim)   # input with `dim` removed
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_outer * d_dim * n_inner"          # one compare per input element (serial scan); derivable
  bytes_moved: "n_outer * d_dim * n_inner * 4 + n_outer * n_inner * 4"   # read f32 input (4B), write U32 indices (4B); bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact index selection (serial scan stride n_inner, no FP accumulation); ties resolve to the lowest index; bitwise-identical on any hardware."

determinism: bitwise
```

---

## arg_reduce_any_dim_f64  (argmax / argmin along an arbitrary dim, f64 → U32 index)

Arbitrary-dim index reduction of an f64 tensor — the slow path
(`arg_reduce_any_dim_f32.slang:31` family). Logical `[n_outer, d_dim, n_inner]`; one thread per
output element serially scans the reduction dim with stride `n_inner`, comparing native `double`
values; `op_id` selects argmax/argmin; **lower index wins on ties**. Input f64; **output U32**,
dropping the reduced dim. Exact index selection ⇒ bitwise-identical on any hardware.

```fkc
kernel: arg_reduce_any_dim_f64
op_kind: ArgMaxDim          # op_id selector: also ArgMinDim at a distinct key
blurb: "Argmax/argmin along an arbitrary dim (f64 in, U32 index out); serial scan stride n_inner; lowest index wins ties."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::arg_reduce_any_dim_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; logical [n_outer, d_dim, n_inner]; dim = dims[0] in 0..rank; n_inner = product(dims after `dim`)"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly one dim; dim in 0..input_rank" }
      keepdim: { kind: bool, note: "always false; output drops dim ⇒ [n_outer, n_inner]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: reduce(input, dims, keepdim)   # input with `dim` removed
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_outer * d_dim * n_inner"
  bytes_moved: "n_outer * d_dim * n_inner * 8 + n_outer * n_inner * 4"   # read f64 input (8B), write U32 indices (4B); bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact index selection (serial scan stride n_inner, native double compare, no FP accumulation); ties resolve to the lowest index; bitwise-identical on any hardware."

determinism: bitwise
```

---

## arg_reduce_any_dim_bf16  (argmax / argmin along an arbitrary dim, bf16 → U32 index)

Arbitrary-dim index reduction of a bf16 tensor — the slow path
(`arg_reduce_any_dim_f32.slang:31` family). Logical `[n_outer, d_dim, n_inner]`; one thread per
output element serially scans the reduction dim with stride `n_inner`, **lane-selecting bf16
values from the packed input** (widen `bits << 16` to f32 for the compare); `op_id` selects
argmax/argmin; **lower index wins on ties**. Input bf16; **output U32**, dropping the reduced dim.
Exact index selection ⇒ bitwise-identical on any hardware.

```fkc
kernel: arg_reduce_any_dim_bf16
op_kind: ArgMaxDim          # op_id selector: also ArgMinDim at a distinct key
blurb: "Argmax/argmin along an arbitrary dim (bf16 in, U32 index out); serial scan stride n_inner; lowest index wins ties."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::arg_reduce_any_dim_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; logical [n_outer, d_dim, n_inner]; dim = dims[0] in 0..rank; n_inner = product(dims after `dim`)"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly one dim; dim in 0..input_rank" }
      keepdim: { kind: bool, note: "always false; output drops dim ⇒ [n_outer, n_inner]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: reduce(input, dims, keepdim)   # input with `dim` removed
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_outer * d_dim * n_inner"
  bytes_moved: "n_outer * d_dim * n_inner * 2 + n_outer * n_inner * 4"   # read bf16 input (2B), write U32 indices (4B); bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact index selection (lane-select from packed bf16, widen to f32 for compare, no FP accumulation); ties resolve to the lowest index; bitwise-identical on any hardware."

determinism: bitwise
```

---

## arg_reduce_any_dim_f16  (argmax / argmin along an arbitrary dim, f16 → U32 index)

Arbitrary-dim index reduction of an f16 tensor — the slow path
(`arg_reduce_any_dim_f32.slang:31` family). Logical `[n_outer, d_dim, n_inner]`; one thread per
output element serially scans the reduction dim with stride `n_inner`, **lane-selecting f16 values
from the packed input** (widen to f32 for the compare); `op_id` selects argmax/argmin; **lower
index wins on ties**. Input f16; **output U32**, dropping the reduced dim. Exact index selection ⇒
bitwise-identical on any hardware.

```fkc
kernel: arg_reduce_any_dim_f16
op_kind: ArgMaxDim          # op_id selector: also ArgMinDim at a distinct key
blurb: "Argmax/argmin along an arbitrary dim (f16 in, U32 index out); serial scan stride n_inner; lowest index wins ties."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::arg_reduce_any_dim_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; logical [n_outer, d_dim, n_inner]; dim = dims[0] in 0..rank; n_inner = product(dims after `dim`)"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly one dim; dim in 0..input_rank" }
      keepdim: { kind: bool, note: "always false; output drops dim ⇒ [n_outer, n_inner]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: reduce(input, dims, keepdim)   # input with `dim` removed
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_outer * d_dim * n_inner"
  bytes_moved: "n_outer * d_dim * n_inner * 2 + n_outer * n_inner * 4"   # read f16 input (2B), write U32 indices (4B); bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact index selection (lane-select from packed f16, widen to f32 for compare, no FP accumulation); ties resolve to the lowest index; bitwise-identical on any hardware."

determinism: bitwise
```

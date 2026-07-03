---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — reduction primitives (production) kernel contracts

The Vulkan backend's **reduction** primitives as PRODUCTION registers them (crate `vulkan`, family
`reduce`): the value reduces `OpKind::{SumReduce, MaxReduce, MinReduce, MeanReduce}` and the
index reduces `OpKind::{ArgMaxDim, ArgMinDim}`.

**As-built binding model — production truth (DISTINCT per-(op, dtype) wrappers).** This production
contract supersedes the aspirational `vulkan/reduce.fkc.md` corpus, which describes each Slang kernel
by entry point (`reduce`, `reduce_last_dim`, `arg_reduce_last_dim_*`, …) with an internal `op_id`
selector picking Sum/Max/Min/Mean — a binding model FKC's one-`op_kind`-per-section importer does not
express. Production instead registers a DISTINCT `KernelRef` for each `(OpKind, dtype)` combination
(`reduce::{sum,max,min,mean}_{f32,f16,bf16,f64}`, `arg_reduce::{argmax,argmin}_{f32,f16,bf16,f64}` —
each a thin wrapper that packs its op-id + routes full-vs-last-dim internally). So this file authors
ONE `op_kind` section per OpKind, fanned over its dtype list to the distinct per-dtype wrapper
(mirroring the matmul/conv per-combo precedent + the elementwise per-op precedent).

- **Value reduces** key `(OpKind, [in_dtype, out_dtype], Vulkan)` = 2-slot `[T, T]` (`passthrough`
  output), over `[F32, F16, BF16, F64]`.
- **Index reduces** key `(OpKind, [in_dtype, U32], Vulkan)` = 2-slot `[T, U32]` (`fixed(U32)`
  index output), over `[F32, F16, BF16, F64]`.

Each section fans the BASE `entry_point` over its dtype list; the link registry resolves
`<base>_<suffix>` to the per-dtype wrapper, byte-for-byte the deleted hand-written
`register_with_precision(OpKind::{SumReduce,…,ArgMinDim}, …)` regs.

**Layout model — contiguous-only (matches the as-built reg).** Every reduction kernel reads its input
flat (full reduce: a `0..n` walk; last-dim: a `[n_rows, n_cols]` walk with a subgroup tree reduction);
none consults a `Layout`/strides/offset, so the production registrations are plain
`register_with_precision` (no strided caps): `awkward_layout_strategy: requires_contiguous`
(`strided_input == false`); the planner auto-Contiguizes a strided operand first (§4.3). Output is a
fresh contiguous buffer (scalar / dropped-dim vector), no aliasing.

**Cost provenance.** Every cost block is `judge_measured` (§4.4). The `flops` / `bytes_moved` hints
are the derivable single-pass reduction structure; no overhead constant is fabricated. The imported
`unknown_cost` sentinel is upgraded to the shared OpKind cost fn by `fill_unset_cost_for_backend`.

**Determinism.** The value reduces accumulate in a subgroup / shared-memory tree whose FADD order is
scheduler-dependent (non-associative f32), so — matching the matmul / conv sibling reductions and §10
rule 9 — they are `determinism: nondeterministic` with `bit_stable_on_same_hardware: false` and an
audited `none(reason)` precision (byte-for-byte the deleted regs' `PrecisionGuarantee::none(reason)`).
The index reduces select an INTEGER index with no FP accumulation (ties → lowest index, numpy/PyTorch),
so they are exact and `determinism: bitwise` (`max_ulp: 0`).

---

## reduce_sum  (Sum reduction; f32/f16/bf16/f64; contiguous)

Sum reduction (full-tensor scalar or per-row last-dim; the wrapper routes internally). Subgroup tree
reduction; f32/f64 native, f16/bf16 accumulate in f32 and narrow on store. FOUR distinct per-dtype
wrappers (`reduce::sum_{f32,f16,bf16,f64}`); this section fans the BASE `entry_point` over
`[F32, F16, BF16, F64]`. Contiguous-only binding. Dispatch key `(SumReduce, [T, T], Vulkan)`.

```fkc
kernel: reduce_sum
op_kind: SumReduce
blurb: "Sum reduction (full or last-dim); f32/f16/bf16/f64; subgroup tree reduction; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::reduce_sum"   # BASE symbol; fans reduce_sum_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "full reduce (dims empty / all axes) or last-dim reduce"
  op_params:
    variant: Reduce               # OpParams::Reduce (primitive namespace; §3.7)
    fields:
      dims:    { kind: "Vec<usize>", note: "empty/all axes (full) or the last dim" }
      keepdim: { kind: bool, note: "always false today (reduced dims removed)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)       # key [T, T]
      shape_rule: reduce(input, dims, keepdim)
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
  bytes_moved: "2 * n * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false   # subgroup tree reduction; scheduler-dependent FADD order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                        # audited none(reason): no static bound (non-associative subgroup reduction)
  notes: "subgroup tree reduction (f32/f64 native, f16/bf16 accumulate in f32, narrow on store); FADD order scheduler-dependent; NOT bit-stable."

determinism: nondeterministic
```

---

## reduce_max  (Max reduction; f32/f16/bf16/f64; contiguous)

Max reduction (full or last-dim). Subgroup tree reduction over the comparator. FOUR distinct per-dtype
wrappers (`reduce::max_{f32,f16,bf16,f64}`); fans `[F32, F16, BF16, F64]`. Dispatch key
`(MaxReduce, [T, T], Vulkan)`.

```fkc
kernel: reduce_max
op_kind: MaxReduce
blurb: "Max reduction (full or last-dim); f32/f16/bf16/f64; subgroup tree reduction; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::reduce_max"   # BASE symbol; fans reduce_max_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "full reduce or last-dim reduce"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", note: "empty/all axes (full) or the last dim" }
      keepdim: { kind: bool, note: "always false today" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
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
  bytes_moved: "2 * n * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "subgroup tree reduction (element selection deterministic, comparison order across subgroups scheduler-dependent); NOT bit-stable cross-hardware."

determinism: nondeterministic
```

---

## reduce_min  (Min reduction; f32/f16/bf16/f64; contiguous)

Min reduction (full or last-dim), sibling of `reduce_max` with the min comparator. FOUR distinct
per-dtype wrappers (`reduce::min_{f32,f16,bf16,f64}`); fans `[F32, F16, BF16, F64]`. Dispatch key
`(MinReduce, [T, T], Vulkan)`.

```fkc
kernel: reduce_min
op_kind: MinReduce
blurb: "Min reduction (full or last-dim); f32/f16/bf16/f64; subgroup tree reduction; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::reduce_min"   # BASE symbol; fans reduce_min_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "full reduce or last-dim reduce"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", note: "empty/all axes (full) or the last dim" }
      keepdim: { kind: bool, note: "always false today" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
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
  bytes_moved: "2 * n * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "subgroup tree reduction with min comparator; same scheduler-dependence as MaxReduce; NOT bit-stable cross-hardware."

determinism: nondeterministic
```

---

## reduce_mean  (Mean reduction; f32/f16/bf16/f64; contiguous)

Mean reduction (full or last-dim) = sum then divide by the element count. Subgroup tree reduction +
scalar division. FOUR distinct per-dtype wrappers (`reduce::mean_{f32,f16,bf16,f64}`); fans
`[F32, F16, BF16, F64]`. Dispatch key `(MeanReduce, [T, T], Vulkan)`.

```fkc
kernel: reduce_mean
op_kind: MeanReduce
blurb: "Mean reduction (full or last-dim) = sum/count; f32/f16/bf16/f64; subgroup tree reduction; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::reduce_mean"   # BASE symbol; fans reduce_mean_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "full reduce or last-dim reduce"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", note: "empty/all axes (full) or the last dim" }
      keepdim: { kind: bool, note: "always false today" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
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
  bytes_moved: "2 * n * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "subgroup tree reduction + scalar division (Mean = Sum/n); accumulation order scheduler-dependent; NOT bit-stable."

determinism: nondeterministic
```

---

## arg_max  (ArgMax along a dim; f32/f16/bf16/f64 → U32 index; contiguous)

Index of the maximum along the reduction dim (last-dim fast path or arbitrary-dim scan). Tree
reduction over `(val, idx)` pairs; **lowest index wins on ties** (numpy/PyTorch); NaN never wins.
Output is `U32` indices (exact index selection — no FP accumulation ⇒ bitwise-identical on any
hardware). FOUR distinct per-dtype wrappers (`arg_reduce::argmax_{f32,f16,bf16,f64}`); this section
fans the BASE `entry_point` over `[F32, F16, BF16, F64]` with a `fixed(U32)` output. Dispatch key
`(ArgMaxDim, [T, U32], Vulkan)`.

```fkc
kernel: arg_max
op_kind: ArgMaxDim
blurb: "ArgMax along a dim (f32/f16/bf16/f64 in, U32 index out); lowest index wins ties; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::arg_max"   # BASE symbol; fans arg_max_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce along a dim; last-dim fast path or arbitrary-dim scan"
  op_params:
    variant: Reduce               # OpParams::Reduce reused for ArgMaxDim/ArgMinDim
    fields:
      dims:    { kind: "Vec<usize>", note: "the reduced dim" }
      keepdim: { kind: bool, note: "always false (dim dropped)" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)               # index output; key [T, U32]
      shape_rule: reduce(input, dims, keepdim)
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
  bytes_moved: "n * dtype_bytes + n_out * 4"   # read input + write U32 indices

precision:
  bit_stable_on_same_hardware: true   # exact integer index selection — no FP accumulation
  max_ulp: 0
  max_relative: 0.0
  max_absolute: 0.0
  audited: true
  notes: "exact index selection; ties resolve to the lowest index; NaN never wins; bitwise-identical on any hardware for given input values."

determinism: bitwise
```

---

## arg_min  (ArgMin along a dim; f32/f16/bf16/f64 → U32 index; contiguous)

Sibling of `arg_max` with the min comparator: index of the minimum along the reduction dim; **lowest
index wins on ties**. Output is `U32` indices (exact ⇒ bitwise-identical on any hardware). FOUR
distinct per-dtype wrappers (`arg_reduce::argmin_{f32,f16,bf16,f64}`); fans `[F32, F16, BF16, F64]`
with a `fixed(U32)` output. Dispatch key `(ArgMinDim, [T, U32], Vulkan)`.

```fkc
kernel: arg_min
op_kind: ArgMinDim
blurb: "ArgMin along a dim (f32/f16/bf16/f64 in, U32 index out); lowest index wins ties; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::arg_min"   # BASE symbol; fans arg_min_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce along a dim; last-dim fast path or arbitrary-dim scan"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", note: "the reduced dim" }
      keepdim: { kind: bool, note: "always false (dim dropped)" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: reduce(input, dims, keepdim)
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
  bytes_moved: "n * dtype_bytes + n_out * 4"

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: 0.0
  max_absolute: 0.0
  audited: true
  notes: "exact index selection (min comparator); ties resolve to the lowest index; bitwise-identical on any hardware for given input values."

determinism: bitwise
```

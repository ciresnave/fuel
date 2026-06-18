---
fkc_version: 1
provider:
  name: fuel-reference-backend
  backend: Cpu                 # the pure-Rust correctness oracle runs host-side (BackendId::Cpu)
  kernel_source: "reference-oracle"   # the BindingEntry.kernel_source tag
  link_registry: fuel_reference_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"       # provider build id, folded into kernel_revision_hash
---

# fuel-reference-backend — indexing / gather / scatter kernel contracts

Reference (oracle) contracts for the indexing family from `src/ops.rs`. Every kernel here is, by
the crate-wide invariant (`RefTensor<T>` is always a contiguous, row-major, zero-offset buffer with
no strides and no offset), **contiguous-only zero-offset** at the data layer: the kernel never walks
strides, never tolerates a non-zero view base, and never walks negative strides. The internal
`row_major_strides` + unflatten machinery is index arithmetic over a *contiguous* buffer, not a
strided input path. Callers materialize any non-contiguous view into a fresh contiguous `RefTensor`
before calling; the planner therefore inserts (and prices, from the `Op::Contiguize` contract) any
required contiguize. The index operand is itself a contiguous integer tensor. Cost is left to the
Judge to bootstrap (`provenance: judge_measured`) except where a FLOPs/bandwidth shape is genuinely
derivable from the op, in which case an author-declared bandwidth hint accompanies it.

## index_select_tensor  (index-select along dim with a U32 index tensor)

Select slices from `data` along `dim` using a **rank-1** integer index tensor. For a rank-N input
the output has the same rank; only the size of `dim` changes from `data.shape[dim]` to
`indices.len()`. The output is filled element-by-element: each output flat position is unflattened
into a multi-index, the `dim` coordinate is translated through `indices`, and the source element is
copied — `out[..., k, ...] = data[..., indices[k], ...]`. Indices are bounds-checked against the
input dim size (`indices[k] < data.shape[dim]`); the index dtype is converted to `usize` (the
conversion is the only fallible step). This is the exec-wired index-select arm (`Op::IndexSelect`),
generic over the data dtype and monomorphized to f32/f64/bf16/f16/**u32**; the index tensor is
**always U32** in the executor (`AnyRefTensor::as_u32`). Algorithm is a single pass over the output
elements, pure copy — no arithmetic, so it is bit-exact for every dtype and dtype-agnostic in
numerics. Known limitations: the index tensor MUST be rank-1; data and index must both be
contiguous zero-offset; no broadcasting; output is a fresh contiguous buffer.

```fkc
kernel: index_select_tensor
op_kind: IndexSelect
blurb: "Index-select along dim with a rank-1 U32 index tensor; copy out[...,k,...]=data[...,idx[k],...]."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::index_select_tensor"
kernel_revision_hash: auto

accept:
  inputs:
    - name: data
      dtypes: [F32, F64, BF16, F16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "rank=1"
  op_params:
    variant: IndexSelect       # OpParams::IndexSelect { dim }
    fields:
      dim: { kind: usize, constraint: "< data.rank" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(data)
      shape_rule: same_as(data)        # identical except dim → indices.len()
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured     # Judge bootstraps; the bandwidth hint below is the author prior it refines
  class: cheap_elementwise
  flops: "0"                     # pure copy, no arithmetic
  bytes_moved: "2 * n * dtype_bytes"   # n = output elements; read one source elem + write one out elem per output
  overhead_ns: ~                 # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true    # pure element copy; no arithmetic
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure copy (no arithmetic); bit-exact for every dtype. Index out-of-range / non-U32 index is a hard error, not a clamp."

determinism: bitwise
```

## index_select  (index-select along dim via a host `&[usize]` slice)

The `&[usize]`-driven index-select: the underlying implementation re-used by the tensor variant,
but taking the indices as a plain host slice rather than as a graph-input tensor. Same algorithm and
semantics as `index_select_tensor` (output rank = input rank, `dim` size → `indices.len()`,
per-output-element unflatten + dim-coordinate translation + copy), but the index dtype bound is
relaxed to a `usize` slice held outside the tensor system, and the data bound is `T: Float`
(f32/f64/bf16/f16) — it does **not** carry the `u32` data arm. **Not directly exec-wired**: the
executor's `Op::IndexSelect` arm routes to `index_select_tensor`, so this entry exists as a
kernel-level fact (the shared implementation surface) but has no production dispatch path of its
own. Bounds-checked (`indices[k] < data.shape[dim]`). Output is a fresh contiguous buffer.

```fkc
kernel: index_select
op_kind: IndexSelect
blurb: "Index-select along dim via a host &[usize] slice; T: Float; shared impl, not exec-wired."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::index_select"
kernel_revision_hash: auto

accept:
  inputs:
    - name: data
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: IndexSelect       # OpParams::IndexSelect { dim }; the index slice is a host param, not a tensor operand
    fields:
      dim: { kind: usize, constraint: "< data.rank" }
      indices: { kind: "Vec<usize>", note: "host index slice; each < data.dim[dim]; not a graph-input tensor" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(data)
      shape_rule: same_as(data)        # identical except dim → indices.len()
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
  flops: "0"                     # pure copy, no arithmetic
  bytes_moved: "2 * n * dtype_bytes"   # n = output elements
  overhead_ns: ~                 # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true    # pure element copy
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure copy; bit-exact. Shared impl behind index_select_tensor; not exec-wired (executor routes IndexSelect to the tensor variant)."

determinism: bitwise
```

## gather  (N-D PyTorch gather along dim)

PyTorch-semantics N-D gather: the `indices` tensor has the **same rank as `data`**, and the output
has the **same shape as `indices`**. For each output/index position `p`, the `dim` coordinate is
replaced by the index value and the source element is copied —
`out[p] = data[p with p[dim] ← indices[p]]`. Indices are bounds-checked against the data dim size
(`indices[p] < data.shape[dim]`) and converted to `usize` (the only fallible step). Single pass over
the index elements via internal `row_major_strides` + unflatten over contiguous buffers; pure copy,
no arithmetic, so bit-exact and dtype-agnostic. Exec-wired (`Op::Gather`), generic over the data
dtype and monomorphized to f32/f64/bf16/f16/**u32**; the index tensor is **always U32** in the
executor. Known limitations: index rank must equal data rank; both operands contiguous zero-offset;
output is a fresh contiguous buffer shaped like `indices`.

```fkc
kernel: gather
op_kind: Gather
blurb: "N-D PyTorch gather along dim; index same rank as data, output shape == index shape; U32 index."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::gather"
kernel_revision_hash: auto

accept:
  inputs:
    - name: data
      dtypes: [F32, F64, BF16, F16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=data"
  op_params:
    variant: Gather            # OpParams::Gather { dim }
    fields:
      dim: { kind: usize, constraint: "< data.rank" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(data)
      shape_rule: same_as(indices)     # output shape == index shape
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
  flops: "0"                     # pure copy, no arithmetic
  bytes_moved: "2 * n * dtype_bytes"   # n = output (== index) elements; read one source + write one out per element
  overhead_ns: ~                 # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true    # pure element copy
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure copy; bit-exact for every dtype. Bounds-checked index; out-of-range / non-U32 index is a hard error."

determinism: bitwise
```

## index_add  (functional index-add along dim)

Functional (out-of-place) index-add: returns a copy of `base` with values from `src` accumulated at
positions translated through a **rank-1** index tensor. `base` and `src` have the same rank and
equal sizes in every non-`dim` dimension; `src.shape[dim]` must equal the index length. The output
is initialized as a full copy of `base`, then for every `src` element the `dim` coordinate is mapped
through `indices` and added: `out[..., indices[i], ...] += src[..., i, ...]`. Indices are
bounds-checked (`indices[i] < base.shape[dim]`) and converted to `usize`. The accumulation `+` is in
the element dtype `T` (no f32-accumulator widening); for bf16/f16 the add is performed in the half
type, so repeated adds into the same slot are **not** bit-stable across hardware that reorders or
widens half arithmetic (the contract is `T`-precision, summation-order-dependent only if the same
output slot is hit by multiple `src` rows — which the rank-1 index permits). Exec-wired
(`Op::IndexAdd`), restricted by the executor to **f32/f64/bf16/f16** (the U32 data arm of the
generic impl is not exec-reachable). The output aliases nothing but is read-then-accumulated from
`base` (a copy of `base`, not a view). Known limitations: rank-1 index only; `src[dim] == len(index)`;
all non-`dim` dims of `base` and `src` must match; contiguous zero-offset operands.

```fkc
kernel: index_add
op_kind: IndexAdd
blurb: "Functional index-add: out = copy(base); out[...,idx[i],...] += src[...,i,...]; rank-1 U32 index."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::index_add"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "rank=1"
    - name: src
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=base"   # equal in every non-dim dim; src.dim[dim] == indices.len()
  op_params:
    variant: IndexAdd          # OpParams::IndexAdd { dim }
    fields:
      dim: { kind: usize, constraint: "< base.rank; src.dim[dim] == indices.len()" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: none           # functional: fresh buffer initialized from a COPY of base, then accumulated

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false             # out-of-place: copies base first; does NOT write into base's buffer
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  # base_n = product(base.shape) (the initializing copy); src_n = product(src.shape) (the accumulation pass)
  flops: "src_n"                # one add per src element
  bytes_moved: "(2 * base_n + 2 * src_n) * dtype_bytes"   # copy base in+out, then read src + RMW out per src elem
  overhead_ns: ~                 # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "base_n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true    # deterministic single-thread sequential accumulation in dtype T
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Accumulation in element dtype T (no f32-accum widening). Deterministic sequential order on same hardware; bf16/f16 add in the half type. Bounds-checked, non-U32 index is a hard error."

determinism: same_hardware_bitwise
```

## scatter_add  (functional scatter-add along dim)

Functional (out-of-place) scatter-add: returns a copy of `base` with values from `src` accumulated
at positions given by an N-D `indices` tensor that has the **same shape as `src`**. The output is
initialized as a full copy of `base`, then for every `src` position `p` the `dim` coordinate is
replaced by `indices[p]` and added: `out[p with dim ← indices[p]] += src[p]`. Indices are
bounds-checked (`indices[p] < base.shape[dim]`) and converted to `usize`. Accumulation `+` is in the
element dtype `T` (no widening). Single-thread sequential walk over `src` elements, so it is
deterministic and bit-stable on the same hardware even when multiple `src` positions collide on one
output slot (the collision order is the `src` flat order). Exec-wired (`Op::ScatterAdd`), restricted
by the executor to **f32/f64/bf16/f16**. Output aliases nothing (a copy of `base`, read-then-
accumulated). Known limitations: `indices` must have exactly `src`'s shape; `base` and `src` share
rank; contiguous zero-offset operands.

```fkc
kernel: scatter_add
op_kind: ScatterAdd
blurb: "Functional scatter-add: out = copy(base); out[p with dim←idx[p]] += src[p]; index shape == src shape."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::scatter_add"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=src"     # index shape == src shape (exact)
    - name: src
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=base"
  op_params:
    variant: ScatterAdd        # OpParams::ScatterAdd { dim }
    fields:
      dim: { kind: usize, constraint: "< base.rank" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: none           # functional: fresh buffer initialized from a COPY of base, then accumulated

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
  # base_n = product(base.shape) (the initializing copy); src_n = product(src.shape == index.shape) (scatter pass)
  flops: "src_n"                # one add per src element
  bytes_moved: "(2 * base_n + 2 * src_n) * dtype_bytes"   # copy base in+out, then read src + RMW out per src elem
  overhead_ns: ~                 # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "base_n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true    # deterministic single-thread sequential accumulation in dtype T
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Accumulation in element dtype T (no f32-accum widening). Collision order is src flat order; deterministic on same hardware. Bounds-checked, non-U32 index is a hard error."

determinism: same_hardware_bitwise
```

## embedding  (rank-2 embedding-table lookup)

Embedding lookup: for a rank-2 `table` of shape `[V, D]` (vocab size × hidden dim) and a flat host
slice of token `ids`, produce a rank-2 tensor `[ids.len(), D]` whose row `i` is `table[ids[i]]`,
copied whole-row via `copy_from_slice`. Ids are bounds-checked (`ids[i] < V`). This is a pure
row-copy — no arithmetic — so it is bit-exact and dtype-agnostic. The data bound is `T: Float`
(f32/f64/bf16/f16). **Not directly exec-wired**: there is no `Op::Embedding` executor arm; embedding
is modeled in the graph as an `index_select` / `gather` over the table (so this entry is the oracle
kernel surface, not a standalone dispatch cell). Higher-rank id tensors (e.g. `[batch, seq]`) are
handled by the caller reshaping the output afterward — this kernel is strictly rank-2-in / rank-2-out.
Known limitations: `table` must be rank-2; `ids` is a host `&[usize]` slice, not a graph-input
tensor; contiguous zero-offset table; output is a fresh contiguous `[ids.len(), D]` buffer.

```fkc
kernel: embedding
op_kind: IndexSelect          # embedding is modeled as index_select(table, dim=0, ids); no dedicated Op::Embedding
blurb: "Rank-2 embedding lookup: out[i] = table[ids[i]] for table [V,D]; T: Float; modeled via index_select, not exec-wired."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::embedding"
kernel_revision_hash: auto

accept:
  inputs:
    - name: table
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                  # [V, D]
      shape_constraint: "rank=2"
  op_params:
    variant: IndexSelect       # OpParams::IndexSelect { dim }; dim is implicitly 0; ids is a host param slice
    fields:
      dim: { kind: usize, constraint: "== 0", note: "row lookup along the vocab axis" }
      indices: { kind: "Vec<usize>", note: "host token-id slice; each < table.dim[0] (V); not a graph-input tensor" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(table)
      shape_rule: from_params(ids)     # [ids.len(), D]
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
  flops: "0"                     # pure row copy, no arithmetic
  bytes_moved: "2 * ids_len * d * dtype_bytes"   # read + write one [D] row per id; n = ids_len * d
  overhead_ns: ~                 # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "ids_len * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true    # pure row copy (copy_from_slice); no arithmetic
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure row copy; bit-exact for every dtype. Bounds-checked ids. Rank-2-in / rank-2-out; not exec-wired (modeled as index_select/gather)."

determinism: bitwise
```

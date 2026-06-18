---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                  # maps to BackendId::Cpu
  kernel_source: "portable-cpu" # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"  # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — indexing / gather / scatter kernel contracts

Portable byte-shaped CPU indexing kernels from `fuel-cpu-backend/src/byte_kernels.rs`. Every
kernel in this family operates on flat, contiguous, zero-offset `CpuStorageBytes` slices and
validates *byte length* against explicit `usize` shape parameters — none of them consult a
`Layout`/strides/offset (the pipelined executor's auto-Contiguize pass realizes any
strided/broadcast/offset input into a contiguous buffer first). The **index tensor is always
`U32`, contiguous**; out-of-bounds index values are a hard `Result::Err` (never a panic, never a
silent clamp). The `index_select` / `gather` pair is **dtype-agnostic** (one kernel that copies
`dtype_size` bytes per element regardless of dtype); the `index_add` / `scatter_add` pair is
**per-dtype** (it does arithmetic accumulation, half floats via an f32 accumulator) and **seeds
the output from `base` then accumulates** (not a pure overwrite). Sources cited per section.

## index_select  (pick slices along `dim` by a rank-1 U32 index tensor)

Pick `n_indices` slices from `source` along the selected axis using a rank-1 `U32` `indices`
tensor; the output's selected-dim size equals the index count. Implemented as a **dtype-agnostic
byte copy**: the kernel is parameterized by an explicit `dtype_size` and copies
`inner_count × dtype_size` contiguous bytes per `(outer, index)` pair — so a single kernel serves
every element width (the `index_select_f32` shim is just `index_select_cpu` with `dtype_size = 4`).
The flat layout is `[outer_count, source_dim_size, inner_count]` for the source and
`[outer_count, n_indices, inner_count]` for the output, with `outer_count` = product of the dims
before the selected axis and `inner_count` = product of the dims after it. The index tensor is
`U32`, contiguous, length `n_indices`. Byte lengths of source / indices / output are validated up
front; an index `≥ source_dim_size` returns `Err` (out-of-bounds, never a panic). Output is the
**same dtype as the source**, contiguous row-major, fully overwritten. No broadcasting, no strided
input — contiguous, offset-0 only. Source `byte_kernels.rs:1788` (`index_select_f32` shim 1850).

```fkc
kernel: index_select
op_kind: IndexSelect
blurb: "Pick slices along dim by a rank-1 U32 index tensor; dtype-agnostic byte copy; OOB index errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::index_select_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [U8, I8, U32, I16, I32, I64, BF16, F16, F32, F64]   # dtype-agnostic: copied by byte width
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, source_dim_size, inner_count]; bytes == outer*source_dim*inner*dtype_size"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "notes: length n_indices; each value < source_dim_size or the kernel returns Err"
  op_params:
    variant: IndexSelect          # OpParams::IndexSelect (primitive namespace; §3.7)
    fields:
      outer_count:     { kind: usize, note: "product of dims before the selected axis" }
      source_dim_size: { kind: usize, note: "size of the selected source axis; index bound" }
      n_indices:       { kind: usize, note: "number of indices == output selected-dim size" }
      inner_count:     { kind: usize, note: "product of dims after the selected axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)          # dtype-agnostic byte copy preserves source dtype
      shape_rule: from_params(outer_count, n_indices, inner_count)   # selected axis size := n_indices
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
  provenance: declared           # author prior (overhead_ns launch cost); Judge refines the bandwidth hints below (§4.4)
  class: cheap_elementwise
  # bandwidth-bound gather copy: read+write n_selected elements (index reads negligible). Hint only.
  flops: "0"
  bytes_moved: "2 * outer_count * n_indices * inner_count * dtype_bytes"   # read selected + write out
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * n_indices * inner_count * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # pure byte copy — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact byte-for-byte copy of selected slices; no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

## gather  (N-D gather along `dim` by a same-rank U32 index tensor)

N-dimensional gather: `source` and `indices`/`output` have the **same rank** and agree on every
axis except `dim`; the output shape equals the index tensor's shape. For each output position the
source is read at the same multi-index except the `dim` coordinate is taken from the `U32` index
value. Implemented as a **dtype-agnostic byte copy** (parameterized `dtype_size`, copies
`dtype_size` bytes per output element; the `gather_f32` shim is `gather_cpu` with `dtype_size = 4`).
Row-major source strides are computed from `source_shape`; the output is walked in flat row-major
order and unraveled against `output_shape`. Rank equality and the per-axis agreement
(`source_shape[d] == output_shape[d]` for `d != dim`) are validated, as are all three byte lengths;
an index `≥ source_shape[dim]` returns `Err` (out-of-bounds). Output is the **same dtype as the
source**, contiguous row-major (shape == `output_shape`), fully overwritten. Contiguous, offset-0
only. Source `byte_kernels.rs:2132` (`gather_f32` shim 2234).

```fkc
kernel: gather
op_kind: Gather
blurb: "N-D gather along dim by a same-rank U32 index tensor; dtype-agnostic byte copy; OOB index errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::gather_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [U8, I8, U32, I16, I32, I64, BF16, F16, F32, F64]   # dtype-agnostic: copied by byte width
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: source_shape; bytes == prod(source_shape)*dtype_size"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=source"   # source/output/indices share rank; differ only on dim; each value < source_shape[dim]
  op_params:
    variant: Gather               # OpParams::Gather (primitive namespace; §3.7)
    fields:
      source_shape: { kind: "Vec<usize>", note: "row-major source extents" }
      output_shape: { kind: "Vec<usize>", constraint: "== indices.shape; agrees with source_shape on every axis != dim", note: "output == index tensor shape" }
      dim:          { kind: usize, constraint: "< source_shape.len()", note: "gathered axis; index bound is source_shape[dim]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)          # dtype-agnostic byte copy preserves source dtype
      shape_rule: from_params(output_shape)     # output == index tensor shape
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
  provenance: declared           # author prior (overhead_ns launch cost); Judge refines the bandwidth hints below (§4.4)
  class: cheap_elementwise
  # bandwidth-bound: one element read + written per output position (n = product of output_shape). Hint only.
  flops: "0"
  bytes_moved: "2 * n * dtype_bytes"   # n = prod(output_shape); read gathered element + write out
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # pure byte copy — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact byte-for-byte gather copy; no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

## index_add_f32  (accumulate src into a copy of base at rank-1 U32 indices, f32)

Index-add along a single axis: seed the output from `base`, then for each `i ∈ 0..n_indices`
accumulate `src[..., i, ...]` into `out[..., indices[i], ...]`. Output shape equals `base` shape;
the flat layout is `[outer_count, base_dim_size, inner_count]` for base/out and
`[outer_count, n_indices, inner_count]` for src, with `outer_count`/`inner_count` the products of
the dims before/after the indexed axis. The index tensor is rank-1 `U32`, length `n_indices`. This
is **f32** arithmetic: native `+=` accumulation. The kernel **first copies `base` into `out`**
(read-modify-write of a base copy, not a fresh overwrite), then walks indices; a duplicate index
accumulates multiple `src` rows into the same destination. All four byte lengths are validated; an
index `≥ base_dim_size` returns `Err`. The empty case (`n_indices == 0`) is the base copy alone.
Output is **F32**, contiguous row-major, shape == base. Contiguous, offset-0 only. Source
`byte_kernels.rs:1255`.

```fkc
kernel: index_add_f32
op_kind: IndexAdd
blurb: "Accumulate src into a base copy at rank-1 U32 indices along one axis; f32; OOB index errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::index_add_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, base_dim_size, inner_count]; out shape == base shape"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "notes: length n_indices; each value < base_dim_size or the kernel returns Err"
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, n_indices, inner_count]"
  op_params:
    variant: IndexAdd             # OpParams::IndexAdd (primitive namespace; §3.7)
    fields:
      outer_count:   { kind: usize, note: "product of dims before the indexed axis" }
      base_dim_size: { kind: usize, note: "size of the indexed base axis; index bound" }
      n_indices:     { kind: usize, note: "number of indices == src indexed-dim size" }
      inner_count:   { kind: usize, note: "product of dims after the indexed axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # out is base read-modify-written: seeded from base then += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared           # author prior (overhead_ns launch cost); Judge refines the bandwidth hints below (§4.4)
  class: cheap_elementwise
  # base copy (base_dim_size) + accumulate (n_indices) rows, each outer*inner f32 elements. Hint only.
  flops: "outer_count * n_indices * inner_count"
  bytes_moved: "(outer_count * base_dim_size * inner_count * 2 + outer_count * n_indices * inner_count) * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * base_dim_size * inner_count * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic in-order f32 accumulation
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32 +=, deterministic index order; bit-stable on same hardware. Duplicate indices accumulate in index order."

determinism: same_hardware_bitwise
```

## index_add_f64  (accumulate src into a copy of base at rank-1 U32 indices, f64)

Same algorithm and layout as `index_add_f32` (seed out from `base`, then `out[..., indices[i],
...] += src[..., i, ...]` for each rank-1 `U32` index), in **f64** native arithmetic. Generated by
`index_add_native_kernel!` over `f64`. Byte lengths of base/out, indices (`n_indices × 4`), and src
are validated; an index `≥ base_dim_size` returns `Err`; `n_indices == 0` yields the base copy.
Output is **F64**, contiguous, shape == base, base-seeded then accumulated. Contiguous, offset-0
only. Source `byte_kernels.rs:1397` (`index_add_native_kernel!`).

```fkc
kernel: index_add_f64
op_kind: IndexAdd
blurb: "Accumulate src into a base copy at rank-1 U32 indices along one axis; f64; OOB index errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::index_add_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, base_dim_size, inner_count]; out shape == base shape"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "notes: length n_indices; each value < base_dim_size or the kernel returns Err"
    - name: src
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, n_indices, inner_count]"
  op_params:
    variant: IndexAdd             # OpParams::IndexAdd (primitive namespace; §3.7)
    fields:
      outer_count:   { kind: usize, note: "product of dims before the indexed axis" }
      base_dim_size: { kind: usize, note: "size of the indexed base axis; index bound" }
      n_indices:     { kind: usize, note: "number of indices == src indexed-dim size" }
      inner_count:   { kind: usize, note: "product of dims after the indexed axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # out is base read-modify-written: seeded from base then += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared           # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "outer_count * n_indices * inner_count"
  bytes_moved: "(outer_count * base_dim_size * inner_count * 2 + outer_count * n_indices * inner_count) * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * base_dim_size * inner_count * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic in-order f64 accumulation
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64 +=, deterministic index order; bit-stable on same hardware. Duplicate indices accumulate in index order."

determinism: same_hardware_bitwise
```

## index_add_bf16  (accumulate src into a copy of base at rank-1 U32 indices, bf16 via f32 acc)

Same algorithm and layout as `index_add_f32`, for **bf16** I/O computed with an **f32
accumulator**: each destination element is widened to f32, the `src` row (widened to f32) is added,
and the result is narrowed back to bf16 on store — the load-bearing half-float precision invariant.
Generated by `index_add_half_kernel!` over `bf16`. Out is seeded from `base`, then accumulated; the
same byte-length validation and `≥ base_dim_size → Err` bound apply; `n_indices == 0` yields the
base copy. Output is **BF16**, contiguous, shape == base, base-seeded then accumulated. Contiguous,
offset-0 only. Source `byte_kernels.rs:1473` (`index_add_half_kernel!`).

```fkc
kernel: index_add_bf16
op_kind: IndexAdd
blurb: "Accumulate src into a base copy at rank-1 U32 indices along one axis; bf16 I/O, f32 accumulator; OOB index errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::index_add_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, base_dim_size, inner_count]; out shape == base shape"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "notes: length n_indices; each value < base_dim_size or the kernel returns Err"
    - name: src
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, n_indices, inner_count]"
  op_params:
    variant: IndexAdd             # OpParams::IndexAdd (primitive namespace; §3.7)
    fields:
      outer_count:   { kind: usize, note: "product of dims before the indexed axis" }
      base_dim_size: { kind: usize, note: "size of the indexed base axis; index bound" }
      n_indices:     { kind: usize, note: "number of indices == src indexed-dim size" }
      inner_count:   { kind: usize, note: "product of dims after the indexed axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # out is base read-modify-written: seeded from base then += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared           # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "outer_count * n_indices * inner_count"
  bytes_moved: "(outer_count * base_dim_size * inner_count * 2 + outer_count * n_indices * inner_count) * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * base_dim_size * inner_count * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic in-order accumulation in f32, narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 I/O widened to f32 for accumulation, narrowed on store; deterministic index order; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

## index_add_f16  (accumulate src into a copy of base at rank-1 U32 indices, f16 via f32 acc)

Same algorithm and layout as `index_add_f32`, for **f16** I/O computed with an **f32 accumulator**
(widen → `+=` → narrow on store). Generated by `index_add_half_kernel!` over `f16`. Out is seeded
from `base`, then accumulated; the same byte-length validation and `≥ base_dim_size → Err` bound
apply; `n_indices == 0` yields the base copy. Output is **F16**, contiguous, shape == base,
base-seeded then accumulated. Contiguous, offset-0 only. Source `byte_kernels.rs:1474`
(`index_add_half_kernel!`).

```fkc
kernel: index_add_f16
op_kind: IndexAdd
blurb: "Accumulate src into a base copy at rank-1 U32 indices along one axis; f16 I/O, f32 accumulator; OOB index errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::index_add_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, base_dim_size, inner_count]; out shape == base shape"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "notes: length n_indices; each value < base_dim_size or the kernel returns Err"
    - name: src
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, n_indices, inner_count]"
  op_params:
    variant: IndexAdd             # OpParams::IndexAdd (primitive namespace; §3.7)
    fields:
      outer_count:   { kind: usize, note: "product of dims before the indexed axis" }
      base_dim_size: { kind: usize, note: "size of the indexed base axis; index bound" }
      n_indices:     { kind: usize, note: "number of indices == src indexed-dim size" }
      inner_count:   { kind: usize, note: "product of dims after the indexed axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # out is base read-modify-written: seeded from base then += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared           # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "outer_count * n_indices * inner_count"
  bytes_moved: "(outer_count * base_dim_size * inner_count * 2 + outer_count * n_indices * inner_count) * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * base_dim_size * inner_count * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic in-order accumulation in f32, narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O widened to f32 for accumulation, narrowed on store; deterministic index order; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

## scatter_add_f32  (N-D scatter-add — functional inverse of gather, f32)

N-dimensional scatter-add, the functional inverse of `gather`: seed the output from `base`, then
for every flat position `p` in the `indices`/`src` shape, read `indices[p]` (`U32`) as the
destination's `dim` coordinate and accumulate `src[p]` into `base` at that multi-index. `base` and
`src` share rank and agree on every axis except `dim`; `indices` and `src` share the same shape.
Implemented over **f32** native arithmetic via `scatter_add_native_kernel!`. Row-major `base`
strides are computed from `base_shape`; `src`/`indices` are walked flat and unraveled against
`src_shape`. Rank equality, the per-axis agreement (`base_shape[d] == src_shape[d]` for `d != dim`),
`dim < rank`, and all three byte lengths are validated; an index `≥ base_shape[dim]` returns `Err`;
`src_total == 0` yields the base copy. Duplicate index values accumulate into the same destination
in flat-`p` order. Output is **F32**, contiguous, shape == base, base-seeded then accumulated.
Contiguous, offset-0 only. Source `byte_kernels.rs:1682`.

```fkc
kernel: scatter_add_f32
op_kind: ScatterAdd
blurb: "N-D scatter-add (inverse of gather) into a base copy via a same-shape U32 index tensor; f32; OOB index errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::scatter_add_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: base_shape; out shape == base_shape; agrees with src_shape on every axis != dim"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=src"   # indices.shape == src_shape; each value < base_shape[dim] or the kernel returns Err
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=base"   # src_shape; differs from base only along dim
  op_params:
    variant: ScatterAdd           # OpParams::ScatterAdd (primitive namespace; §3.7)
    fields:
      base_shape: { kind: "Vec<usize>", note: "row-major base extents; out == base_shape" }
      src_shape:  { kind: "Vec<usize>", constraint: "== indices.shape; agrees with base_shape on every axis != dim" }
      dim:        { kind: usize, constraint: "< base_shape.len()", note: "scattered axis; index bound is base_shape[dim]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # out is base read-modify-written: seeded from base then += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared           # author prior (overhead_ns launch cost); Judge refines the bandwidth hints below (§4.4)
  class: cheap_elementwise
  # base copy (prod(base_shape)) + one += per src element (n = prod(src_shape)). Hint only.
  flops: "n"
  bytes_moved: "(2 * base_total + 3 * n) * dtype_bytes"   # n = prod(src_shape); base_total = prod(base_shape)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "base_total * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic flat-p-order f32 accumulation
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32 +=, deterministic flat-position order; bit-stable on same hardware. Duplicate indices accumulate in flat-p order."

determinism: same_hardware_bitwise
```

## scatter_add_f64  (N-D scatter-add — functional inverse of gather, f64)

Same algorithm and layout as `scatter_add_f32` (seed out from `base`, then `out[dest(p)] += src[p]`
with the `dim` coordinate of `dest` taken from `indices[p]`), in **f64** native arithmetic.
Generated by `scatter_add_native_kernel!` over `f64`. Rank equality, per-axis agreement,
`dim < rank`, and all byte lengths are validated; an index `≥ base_shape[dim]` returns `Err`;
`src_total == 0` yields the base copy. Output is **F64**, contiguous, shape == base, base-seeded
then accumulated. Contiguous, offset-0 only. Source `byte_kernels.rs:1575`
(`scatter_add_native_kernel!`).

```fkc
kernel: scatter_add_f64
op_kind: ScatterAdd
blurb: "N-D scatter-add (inverse of gather) into a base copy via a same-shape U32 index tensor; f64; OOB index errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::scatter_add_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: base_shape; out shape == base_shape; agrees with src_shape on every axis != dim"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=src"   # indices.shape == src_shape; each value < base_shape[dim] or the kernel returns Err
    - name: src
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=base"   # src_shape; differs from base only along dim
  op_params:
    variant: ScatterAdd           # OpParams::ScatterAdd (primitive namespace; §3.7)
    fields:
      base_shape: { kind: "Vec<usize>", note: "row-major base extents; out == base_shape" }
      src_shape:  { kind: "Vec<usize>", constraint: "== indices.shape; agrees with base_shape on every axis != dim" }
      dim:        { kind: usize, constraint: "< base_shape.len()", note: "scattered axis; index bound is base_shape[dim]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # out is base read-modify-written: seeded from base then += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared           # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "(2 * base_total + 3 * n) * dtype_bytes"   # n = prod(src_shape); base_total = prod(base_shape)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "base_total * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic flat-p-order f64 accumulation
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64 +=, deterministic flat-position order; bit-stable on same hardware. Duplicate indices accumulate in flat-p order."

determinism: same_hardware_bitwise
```

## scatter_add_bf16  (N-D scatter-add — inverse of gather, bf16 via f32 acc)

Same algorithm and layout as `scatter_add_f32`, for **bf16** I/O computed with an **f32
accumulator** (widen the destination + `src[p]` to f32, add, narrow on store). Generated by
`scatter_add_half_kernel!` over `bf16`. Rank equality, per-axis agreement, `dim < rank`, and all
byte lengths are validated; an index `≥ base_shape[dim]` returns `Err`; `src_total == 0` yields the
base copy. Output is **BF16**, contiguous, shape == base, base-seeded then accumulated. Contiguous,
offset-0 only. Source `byte_kernels.rs:1673` (`scatter_add_half_kernel!`).

```fkc
kernel: scatter_add_bf16
op_kind: ScatterAdd
blurb: "N-D scatter-add (inverse of gather) into a base copy via a same-shape U32 index tensor; bf16 I/O, f32 accumulator; OOB index errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::scatter_add_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: base_shape; out shape == base_shape; agrees with src_shape on every axis != dim"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=src"   # indices.shape == src_shape; each value < base_shape[dim] or the kernel returns Err
    - name: src
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=base"   # src_shape; differs from base only along dim
  op_params:
    variant: ScatterAdd           # OpParams::ScatterAdd (primitive namespace; §3.7)
    fields:
      base_shape: { kind: "Vec<usize>", note: "row-major base extents; out == base_shape" }
      src_shape:  { kind: "Vec<usize>", constraint: "== indices.shape; agrees with base_shape on every axis != dim" }
      dim:        { kind: usize, constraint: "< base_shape.len()", note: "scattered axis; index bound is base_shape[dim]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # out is base read-modify-written: seeded from base then += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared           # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "(2 * base_total + 3 * n) * dtype_bytes"   # n = prod(src_shape); base_total = prod(base_shape)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "base_total * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic flat-p-order accumulation in f32, narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 I/O widened to f32 for accumulation, narrowed on store; deterministic flat-position order; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

## scatter_add_f16  (N-D scatter-add — inverse of gather, f16 via f32 acc)

Same algorithm and layout as `scatter_add_f32`, for **f16** I/O computed with an **f32
accumulator** (widen → `+=` → narrow on store). Generated by `scatter_add_half_kernel!` over
`f16`. Rank equality, per-axis agreement, `dim < rank`, and all byte lengths are validated; an
index `≥ base_shape[dim]` returns `Err`; `src_total == 0` yields the base copy. Output is **F16**,
contiguous, shape == base, base-seeded then accumulated. Contiguous, offset-0 only. Source
`byte_kernels.rs:1674` (`scatter_add_half_kernel!`).

```fkc
kernel: scatter_add_f16
op_kind: ScatterAdd
blurb: "N-D scatter-add (inverse of gather) into a base copy via a same-shape U32 index tensor; f16 I/O, f32 accumulator; OOB index errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::scatter_add_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: base_shape; out shape == base_shape; agrees with src_shape on every axis != dim"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=src"   # indices.shape == src_shape; each value < base_shape[dim] or the kernel returns Err
    - name: src
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=base"   # src_shape; differs from base only along dim
  op_params:
    variant: ScatterAdd           # OpParams::ScatterAdd (primitive namespace; §3.7)
    fields:
      base_shape: { kind: "Vec<usize>", note: "row-major base extents; out == base_shape" }
      src_shape:  { kind: "Vec<usize>", constraint: "== indices.shape; agrees with base_shape on every axis != dim" }
      dim:        { kind: usize, constraint: "< base_shape.len()", note: "scattered axis; index bound is base_shape[dim]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # out is base read-modify-written: seeded from base then += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared           # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "(2 * base_total + 3 * n) * dtype_bytes"   # n = prod(src_shape); base_total = prod(base_shape)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "base_total * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic flat-p-order accumulation in f32, narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O widened to f32 for accumulation, narrowed on store; deterministic flat-position order; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

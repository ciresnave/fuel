---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan               # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang" # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"  # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — indexing / gather / scatter kernel contracts

Vulkan compute-shader indexing kernels from `fuel-kernels-source/kernels/*.slang` (AOT-compiled to
SPIR-V in `fuel-vulkan-kernels/spv/`), with Rust dispatch wrappers in
`fuel-vulkan-backend/src/lib.rs`. Every kernel in this family consumes **flat, contiguous,
zero-offset** buffers — the rank-4/rank-N shape data rides in `op_params` (or a `shape_buf` storage
buffer), and none of them walk a `Layout`/strides/offset (the executor's auto-Contiguize pass
realizes any strided/broadcast/offset input into a contiguous buffer first). The **index tensor is
always `U32`, contiguous**. Two families split by behavior:

- **Pure data movers** (`index_select*`, `gather_b*`) copy `dtype_size` / byte-width bytes per
  element with **no arithmetic** — bit-exact, deterministic. `index_select` is **dtype-monomorphized**
  (one kernel per element type) and **clamps** an out-of-range index to `axis_in-1` (does not error);
  `gather_b*` is **byte-width-keyed** (one kernel per element size) and applies **no bounds clamp**
  (the caller must pre-validate indices).
- **Atomic accumulators** (`scatter_add_*`, `index_add_*`) seed the output from `base` (the wrapper
  pre-initializes / copies base → out), then accumulate `src` via a **bounded compare-and-swap (CAS)
  atomic add** — f32 via `uint` CAS, f64 via `u64` CAS (needs `shaderInt64` + f64 atomics), bf16/f16
  via sub-word CAS (math in f32, narrow on store). The CAS loop is bounded to 1000 iterations: under
  extreme contention a value can be dropped, and FP atomic accumulation order is scheduler-dependent,
  so these kernels are **nondeterministic** (run-to-run variation possible) — declared honestly per
  §4.9.

Half-precision (`*_bf16`/`*_f16`) follow the family invariant: packed-u16-in-u32 storage, f32 math,
narrow on store; some require even inner counts (`index_select_bf16` needs `inner % 2 == 0`). Sources
cited per section.

## index_select  (row-wise lookup along a dim by a rank-1 U32 index tensor, f32)

Pick `axis_out` (= length of `ids`) slices from `source` along the selected axis. The tensor is
flattened to `[outer, axis_in, inner]` for the source and `[outer, axis_out, inner]` for the output,
with `outer` = product of the dims before the selected axis and `inner` = product of the dims after
it. For each `(outer, out_index, inner)` the source row is read at the `ids[out_index]`-th position
along the selected axis and copied through (f32 data move, no arithmetic). The index tensor is `U32`,
contiguous, length `axis_out`. An **out-of-range index is clamped to `axis_in-1`** (it does NOT
error) — the GPU shader reads a valid in-bounds row rather than faulting. Output is **F32**,
contiguous row-major, fully overwritten; the selected dim is resized to `len(ids)`. Contiguous,
offset-0 only; no broadcasting, no strided input. Source `index_select.slang:30`; wrapper
`index_select_f32_bytes` `fuel-vulkan-backend/src/lib.rs:8025`.

```fkc
kernel: index_select
op_kind: IndexSelect
blurb: "Row-wise lookup along a dim by a rank-1 U32 index tensor; f32 byte copy; OOB index clamped to axis_in-1."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::index_select_f32_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer, axis_in, inner]; bytes == outer*axis_in*inner*4"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "notes: length axis_out; an index >= axis_in is CLAMPED to axis_in-1 (no error)"
  op_params:
    variant: IndexSelect          # OpParams::IndexSelect (primitive namespace; §3.7)
    fields:
      outer_count:     { kind: usize, note: "outer = product of dims before the selected axis" }
      source_dim_size: { kind: usize, note: "axis_in = size of the selected source axis; clamp bound axis_in-1" }
      n_indices:       { kind: usize, note: "axis_out = number of indices == output selected-dim size" }
      inner_count:     { kind: usize, note: "inner = product of dims after the selected axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)          # data move preserves source dtype
      shape_rule: from_params(outer_count, n_indices, inner_count)   # selected axis size := n_indices
      layout_guarantee: contiguous
      aliasing: none
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured     # Judge bootstraps; author declares only the derivable bandwidth hint below
  class: cheap_elementwise
  # bandwidth-bound gather copy: read+write n_selected elements (index reads negligible). Hint only.
  flops: "0"
  bytes_moved: "2 * outer_count * n_indices * inner_count * dtype_bytes"   # read selected + write out
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "outer_count * n_indices * inner_count * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # pure byte copy — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact byte-for-byte copy of selected slices; no arithmetic, bit-identical across any hardware. OOB index clamped to axis_in-1 (no fault)."

determinism: bitwise
```

## index_select_f16  (row-wise lookup along a dim by a rank-1 U32 index tensor, f16)

Same algorithm and `[outer, axis_in, inner]` flattening as `index_select` (one thread per output
element, read source row at `ids[out_index]`, OOB index **clamped to `axis_in-1`**), for **F16** I/O.
A pure data move — the f16 lanes are copied through without unpacking to f32 (no arithmetic). Index
tensor is `U32`, contiguous, length `axis_out`. Output is **F16**, contiguous, selected dim resized
to `len(ids)`, fully overwritten. Contiguous, offset-0 only. Source `index_select.slang:30`;
typed wrapper `index_select_typed_bytes` `fuel-vulkan-backend/src/lib.rs:8172`.

```fkc
kernel: index_select_f16
op_kind: IndexSelect
blurb: "Row-wise lookup along a dim by a rank-1 U32 index tensor; f16 data move; OOB index clamped to axis_in-1."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::index_select_typed_bytes::f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer, axis_in, inner]; bytes == outer*axis_in*inner*2"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "notes: length axis_out; an index >= axis_in is CLAMPED to axis_in-1 (no error)"
  op_params:
    variant: IndexSelect          # OpParams::IndexSelect (primitive namespace; §3.7)
    fields:
      outer_count:     { kind: usize, note: "outer = product of dims before the selected axis" }
      source_dim_size: { kind: usize, note: "axis_in = size of the selected source axis; clamp bound axis_in-1" }
      n_indices:       { kind: usize, note: "axis_out = number of indices == output selected-dim size" }
      inner_count:     { kind: usize, note: "inner = product of dims after the selected axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)
      shape_rule: from_params(outer_count, n_indices, inner_count)
      layout_guarantee: contiguous
      aliasing: none
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * outer_count * n_indices * inner_count * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "outer_count * n_indices * inner_count * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # pure data move — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact f16-lane copy of selected slices; no arithmetic, bit-identical across any hardware. OOB index clamped to axis_in-1."

determinism: bitwise
```

## index_select_bf16  (row-wise lookup along a dim by a rank-1 U32 index tensor, bf16, inner even)

Same `[outer, axis_in, inner]` lookup as `index_select`, for **BF16** I/O. The bf16 lanes are stored
packed two-per-u32, so the kernel is **pair-threaded**: it moves whole u32 words (two bf16 elements)
and therefore **requires `inner % 2 == 0`** (the wrapper validates this). It is a pure data move (no
unpack to f32, no arithmetic). An out-of-range index is **clamped to `axis_in-1`** (no error). Index
tensor is `U32`, contiguous, length `axis_out`. Output is **BF16**, contiguous, selected dim resized
to `len(ids)`, fully overwritten. Contiguous, offset-0 only. Source `index_select_bf16.slang:28`;
typed wrapper `index_select_typed_bytes` `fuel-vulkan-backend/src/lib.rs:8172`.

```fkc
kernel: index_select_bf16
op_kind: IndexSelect
blurb: "Row-wise lookup along a dim by a rank-1 U32 index tensor; bf16 packed-u32 pair-thread (inner even); OOB index clamped."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::index_select_typed_bytes::bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "divisible(inner_count, 2)"   # packed-u32 pair-thread requires inner % 2 == 0; flat [outer, axis_in, inner]
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "notes: length axis_out; an index >= axis_in is CLAMPED to axis_in-1 (no error)"
  op_params:
    variant: IndexSelect          # OpParams::IndexSelect (primitive namespace; §3.7)
    fields:
      outer_count:     { kind: usize, note: "outer = product of dims before the selected axis" }
      source_dim_size: { kind: usize, note: "axis_in = size of the selected source axis; clamp bound axis_in-1" }
      n_indices:       { kind: usize, note: "axis_out = number of indices == output selected-dim size" }
      inner_count:     { kind: usize, constraint: "inner_count % 2 == 0", note: "inner = product of dims after the selected axis; must be even (pair-thread)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)
      shape_rule: from_params(outer_count, n_indices, inner_count)
      layout_guarantee: contiguous
      aliasing: none
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # whole-u32 (2 bf16 lanes) word moves

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * outer_count * n_indices * inner_count * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "outer_count * n_indices * inner_count * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # pure word copy — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact bf16-lane (packed-u32 word) copy of selected slices; no arithmetic, bit-identical across any hardware. Requires inner % 2 == 0. OOB index clamped to axis_in-1."

determinism: bitwise
```

## index_select_f64  (row-wise lookup along a dim by a rank-1 U32 index tensor, f64)

Same `[outer, axis_in, inner]` lookup as `index_select`, for **F64** (native double) I/O. A pure
data move — the f64 element is copied through without arithmetic. An out-of-range index is **clamped
to `axis_in-1`** (no error). Index tensor is `U32`, contiguous, length `axis_out`. Output is **F64**,
contiguous, selected dim resized to `len(ids)`, fully overwritten. Contiguous, offset-0 only. Source
`index_select.slang:30`; typed wrapper `index_select_typed_bytes` `fuel-vulkan-backend/src/lib.rs:8172`.

```fkc
kernel: index_select_f64
op_kind: IndexSelect
blurb: "Row-wise lookup along a dim by a rank-1 U32 index tensor; f64 data move; OOB index clamped to axis_in-1."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::index_select_typed_bytes::f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer, axis_in, inner]; bytes == outer*axis_in*inner*8"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "notes: length axis_out; an index >= axis_in is CLAMPED to axis_in-1 (no error)"
  op_params:
    variant: IndexSelect          # OpParams::IndexSelect (primitive namespace; §3.7)
    fields:
      outer_count:     { kind: usize, note: "outer = product of dims before the selected axis" }
      source_dim_size: { kind: usize, note: "axis_in = size of the selected source axis; clamp bound axis_in-1" }
      n_indices:       { kind: usize, note: "axis_out = number of indices == output selected-dim size" }
      inner_count:     { kind: usize, note: "inner = product of dims after the selected axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)
      shape_rule: from_params(outer_count, n_indices, inner_count)
      layout_guarantee: contiguous
      aliasing: none
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * outer_count * n_indices * inner_count * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "outer_count * n_indices * inner_count * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # pure byte copy — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact f64 copy of selected slices; no arithmetic, bit-identical across any hardware. OOB index clamped to axis_in-1."

determinism: bitwise
```

## gather_b1  (N-D gather along dim by a same-shape U32 index tensor; 1-byte elements)

N-dimensional gather: `src` and `output`/`indices` agree on every axis except `dim`, and the output
shape equals the index tensor's shape. For each output position the source is read at the same
multi-index except the `dim` coordinate, which is taken from the `U32` index value. **Byte-width-keyed
(`b1` = 1-byte elements: U8/I8)** — a dtype-agnostic word-mover that copies 1 byte per output
element; row-major src/out strides are computed from `shape_buf = [src_shape, out_shape]`. Rank ≤ 8.
**No bounds clamp** is applied (unlike `index_select`, the caller must pre-validate that every index
< `src_shape[dim]`). Output is the **same element width as the source**, contiguous row-major
(shape == out_shape), fully overwritten. Contiguous, offset-0 only. Source `gather_b4.slang:29`
(shared byte-width kernel); wrapper `gather_bytes` `fuel-vulkan-backend/src/lib.rs:5546`.

```fkc
kernel: gather_b1
op_kind: Gather
blurb: "N-D gather along dim by a same-shape U32 index tensor; 1-byte (u8/i8) data move; no bounds clamp."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::gather_bytes::b1"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [U8, I8]              # byte-width-keyed b1: 1-byte elements, copied by width
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "notes: src_shape; bytes == prod(src_shape)*1"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "same_as=out"   # indices.shape == out_shape; agrees with src on every axis != dim; NO bounds clamp (caller pre-validates < src_shape[dim])
  op_params:
    variant: Gather               # OpParams::Gather (primitive namespace; §3.7)
    fields:
      source_shape: { kind: "Vec<usize>", note: "row-major source extents; rank <= 8" }
      output_shape: { kind: "Vec<usize>", constraint: "== indices.shape; agrees with source_shape on every axis != dim", note: "output == index tensor shape" }
      dim:          { kind: usize, constraint: "< source_shape.len()", note: "gathered axis; NO bounds clamp on index values" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)          # byte-width data move preserves source dtype
      shape_rule: from_params(output_shape)     # output == index tensor shape
      layout_guarantee: contiguous
      aliasing: none
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured     # Judge bootstraps; author declares only the derivable bandwidth hint below
  class: cheap_elementwise
  # bandwidth-bound: one element read + written per output position (n = product of output_shape). Hint only.
  flops: "0"
  bytes_moved: "2 * n * dtype_bytes"   # n = prod(output_shape); read gathered element + write out
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # pure byte copy — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact byte-for-byte gather copy; no arithmetic, bit-identical across any hardware. No bounds clamp on indices."

determinism: bitwise
```

## gather_b2  (N-D gather along dim by a same-shape U32 index tensor; 2-byte elements)

Same N-D gather algorithm as `gather_b1` (output position read at the source multi-index with the
`dim` coordinate taken from the `U32` index; `shape_buf = [src_shape, out_shape]`; rank ≤ 8; **no
bounds clamp**), for **2-byte elements** (F16/BF16/I16/U16) read as half-words of u32 words. A
dtype-agnostic data move. Output is the **same 2-byte width as the source**, contiguous row-major
(shape == out_shape), fully overwritten. Contiguous, offset-0 only. Source `gather_b4.slang:29`
(shared byte-width kernel); wrapper `gather_bytes` `fuel-vulkan-backend/src/lib.rs:5546`.

```fkc
kernel: gather_b2
op_kind: Gather
blurb: "N-D gather along dim by a same-shape U32 index tensor; 2-byte (f16/bf16/i16/u16) data move; no bounds clamp."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::gather_bytes::b2"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [F16, BF16, I16]      # byte-width-keyed b2: 2-byte elements, copied by width (u16 carried as I16 slot)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "notes: src_shape; bytes == prod(src_shape)*2"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "same_as=out"   # indices.shape == out_shape; agrees with src on every axis != dim; NO bounds clamp
  op_params:
    variant: Gather               # OpParams::Gather (primitive namespace; §3.7)
    fields:
      source_shape: { kind: "Vec<usize>", note: "row-major source extents; rank <= 8" }
      output_shape: { kind: "Vec<usize>", constraint: "== indices.shape; agrees with source_shape on every axis != dim", note: "output == index tensor shape" }
      dim:          { kind: usize, constraint: "< source_shape.len()", note: "gathered axis; NO bounds clamp on index values" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)
      shape_rule: from_params(output_shape)
      layout_guarantee: contiguous
      aliasing: none
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n * dtype_bytes"   # n = prod(output_shape)
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # pure byte copy — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact 2-byte gather copy; no arithmetic, bit-identical across any hardware. No bounds clamp on indices."

determinism: bitwise
```

## gather_b4  (N-D gather along dim by a same-shape U32 index tensor; 4-byte elements)

Same N-D gather algorithm as `gather_b1` (output read at the source multi-index with the `dim`
coordinate from the `U32` index; `shape_buf = [src_shape, out_shape]`; rank ≤ 8; **no bounds clamp**),
for **4-byte elements** (F32/I32/U32) read as whole u32 words — the canonical width and the home of
the shared Slang source. A dtype-agnostic word move. Output is the **same 4-byte width as the
source**, contiguous row-major (shape == out_shape), fully overwritten. Contiguous, offset-0 only.
Source `gather_b4.slang:29`; wrapper `gather_bytes` `fuel-vulkan-backend/src/lib.rs:5546`.

```fkc
kernel: gather_b4
op_kind: Gather
blurb: "N-D gather along dim by a same-shape U32 index tensor; 4-byte (f32/i32/u32) word move; no bounds clamp."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::gather_bytes::b4"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [F32, I32, U32]       # byte-width-keyed b4: 4-byte elements, copied by width
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "notes: src_shape; bytes == prod(src_shape)*4"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "same_as=out"   # indices.shape == out_shape; agrees with src on every axis != dim; NO bounds clamp
  op_params:
    variant: Gather               # OpParams::Gather (primitive namespace; §3.7)
    fields:
      source_shape: { kind: "Vec<usize>", note: "row-major source extents; rank <= 8" }
      output_shape: { kind: "Vec<usize>", constraint: "== indices.shape; agrees with source_shape on every axis != dim", note: "output == index tensor shape" }
      dim:          { kind: usize, constraint: "< source_shape.len()", note: "gathered axis; NO bounds clamp on index values" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)
      shape_rule: from_params(output_shape)
      layout_guarantee: contiguous
      aliasing: none
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n * dtype_bytes"   # n = prod(output_shape)
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # pure word copy — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact 4-byte gather copy; no arithmetic, bit-identical across any hardware. No bounds clamp on indices."

determinism: bitwise
```

## gather_b8  (N-D gather along dim by a same-shape U32 index tensor; 8-byte elements)

Same N-D gather algorithm as `gather_b1` (output read at the source multi-index with the `dim`
coordinate from the `U32` index; `shape_buf = [src_shape, out_shape]`; rank ≤ 8; **no bounds clamp**),
for **8-byte elements** (F64/I64) moved as two u32 words per element. A dtype-agnostic data move.
Output is the **same 8-byte width as the source**, contiguous row-major (shape == out_shape), fully
overwritten. Contiguous, offset-0 only. Source `gather_b4.slang:29` (shared byte-width kernel);
wrapper `gather_bytes` `fuel-vulkan-backend/src/lib.rs:5546`.

```fkc
kernel: gather_b8
op_kind: Gather
blurb: "N-D gather along dim by a same-shape U32 index tensor; 8-byte (f64/i64) data move; no bounds clamp."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::gather_bytes::b8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [F64, I64]            # byte-width-keyed b8: 8-byte elements, copied by width
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "notes: src_shape; bytes == prod(src_shape)*8"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "same_as=out"   # indices.shape == out_shape; agrees with src on every axis != dim; NO bounds clamp
  op_params:
    variant: Gather               # OpParams::Gather (primitive namespace; §3.7)
    fields:
      source_shape: { kind: "Vec<usize>", note: "row-major source extents; rank <= 8" }
      output_shape: { kind: "Vec<usize>", constraint: "== indices.shape; agrees with source_shape on every axis != dim", note: "output == index tensor shape" }
      dim:          { kind: usize, constraint: "< source_shape.len()", note: "gathered axis; NO bounds clamp on index values" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)
      shape_rule: from_params(output_shape)
      layout_guarantee: contiguous
      aliasing: none
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n * dtype_bytes"   # n = prod(output_shape)
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # pure byte copy — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact 8-byte gather copy; no arithmetic, bit-identical across any hardware. No bounds clamp on indices."

determinism: bitwise
```

## scatter_add_f32  (N-D scatter-add into a base copy via atomic CAS, f32)

N-dimensional scatter-add, the functional inverse of `gather`: the **wrapper pre-initializes the
output to `base`**, then for every flat position `p` in the `indices`/`src` shape the kernel reads
`indices[p]` (`U32`) as the destination's `dim` coordinate and **atomically accumulates** `src[p]`
into the output at that multi-index. `base` and `src` share rank-N and agree on every axis except
`dim`; `indices` and `src` share the same shape; `shape_buf = [src_shape, base_shape]`. The atomic add
is implemented as a **bounded compare-and-swap on `uint`** (the f32 bits) — under extreme contention
the 1000-iteration CAS loop may **drop a value**, and concurrent FP accumulation order is
scheduler-dependent, so this kernel is **nondeterministic** (run-to-run variation possible). `dim <
rank`, per-axis agreement, and byte lengths are validated by the wrapper; the kernel applies no
bounds clamp on index values (caller pre-validates). Duplicate index values accumulate into the same
destination. Output is **F32**, contiguous, shape == base, base-seeded then atomically accumulated.
Contiguous, offset-0 only. Source `scatter_add_f32.slang:46`; wrapper
`fuel-vulkan-backend/src/lib.rs:5965`.

```fkc
kernel: scatter_add_f32
op_kind: ScatterAdd
blurb: "N-D scatter-add (inverse of gather) into a base copy via uint-CAS atomic accumulate; f32; nondeterministic under contention."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::scatter_add_f32_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "notes: base_shape; out shape == base_shape; agrees with src_shape on every axis != dim"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "same_as=src"   # indices.shape == src_shape; index value is the dim coordinate; no bounds clamp
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "same_rank=base"   # src_shape; differs from base only along dim
  op_params:
    variant: ScatterAdd           # OpParams::ScatterAdd (primitive namespace; §3.7)
    fields:
      base_shape: { kind: "Vec<usize>", note: "row-major base extents; out == base_shape; rank <= 8" }
      src_shape:  { kind: "Vec<usize>", constraint: "== indices.shape; agrees with base_shape on every axis != dim" }
      dim:        { kind: usize, constraint: "< base_shape.len()", note: "scattered axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # wrapper pre-inits out from base, then kernel atomically += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured     # Judge bootstraps; author declares only the derivable bandwidth hint below
  class: cheap_elementwise
  # base pre-init (prod(base_shape)) + one atomic += per src element (n = prod(src_shape)). Hint only.
  flops: "n"
  bytes_moved: "(2 * base_total + 3 * n) * dtype_bytes"   # n = prod(src_shape); base_total = prod(base_shape)
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "base_total * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # bounded CAS may drop a value under contention; scheduler-dependent atomic order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 atomic accumulate via bounded uint-CAS (1000 iters); under extreme contention a value may be dropped and accumulation order is scheduler-dependent. Audited, no static bound applies."

determinism: nondeterministic
```

## scatter_add_f64  (N-D scatter-add into a base copy via u64 atomic CAS, f64)

Same algorithm and rank-N layout as `scatter_add_f32` (wrapper pre-inits out from `base`, kernel
atomically `out[dest(p)] += src[p]` with the `dim` coordinate of `dest` taken from `indices[p]`), for
**F64** computed with a **`u64` compare-and-swap**. Requires the Vulkan device to advertise
`shaderInt64` plus 64-bit atomics and f64 support. The bounded 1000-iteration CAS loop may **drop a
value** under extreme contention and the atomic order is scheduler-dependent, so the kernel is
**nondeterministic**. `dim < rank`, per-axis agreement, and byte lengths are validated; no index
bounds clamp. Output is **F64**, contiguous, shape == base, base-seeded then atomically accumulated.
Contiguous, offset-0 only. Source `scatter_add_f32.slang:46` (f64 variant); wrapper
`scatter_add_f64_bytes` `fuel-vulkan-backend/src/lib.rs:5676`.

```fkc
kernel: scatter_add_f64
op_kind: ScatterAdd
blurb: "N-D scatter-add into a base copy via u64-CAS atomic accumulate; f64 (needs shaderInt64+atomics+f64); nondeterministic."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::scatter_add_f64_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "notes: base_shape; out shape == base_shape; agrees with src_shape on every axis != dim"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "same_as=src"   # indices.shape == src_shape; index value is the dim coordinate; no bounds clamp
    - name: src
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "same_rank=base"   # src_shape; differs from base only along dim
  op_params:
    variant: ScatterAdd           # OpParams::ScatterAdd (primitive namespace; §3.7)
    fields:
      base_shape: { kind: "Vec<usize>", note: "row-major base extents; out == base_shape; rank <= 8" }
      src_shape:  { kind: "Vec<usize>", constraint: "== indices.shape; agrees with base_shape on every axis != dim" }
      dim:        { kind: usize, constraint: "< base_shape.len()", note: "scattered axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # wrapper pre-inits out from base, then kernel atomically += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "(2 * base_total + 3 * n) * dtype_bytes"   # n = prod(src_shape); base_total = prod(base_shape)
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "base_total * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # bounded CAS may drop a value under contention; scheduler-dependent atomic order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f64 atomic accumulate via bounded u64-CAS (1000 iters); requires shaderInt64 + 64-bit atomics + f64. May drop a value under extreme contention; order scheduler-dependent. Audited, no static bound applies."

determinism: nondeterministic
```

## scatter_add_bf16  (N-D scatter-add into a base copy via sub-word atomic CAS, bf16, f32 acc)

Same algorithm and rank-N layout as `scatter_add_f32`, for **BF16** I/O computed with an **f32
accumulator**: the destination half-word and `src[p]` are widened to f32, added, and narrowed back to
bf16 on the atomic store via a **sub-word compare-and-swap** (the kernel atomically RMWs the 16-bit
lane inside its enclosing u32 word). Wrapper pre-inits out from `base`. The bounded 1000-iteration CAS
loop may **drop a value** under extreme contention and the atomic order is scheduler-dependent, so the
kernel is **nondeterministic**. `dim < rank`, per-axis agreement, and byte lengths are validated; no
index bounds clamp. Output is **BF16**, contiguous, shape == base, base-seeded then atomically
accumulated. Contiguous, offset-0 only. Source `scatter_add_f32.slang:46` (sub-word variant); wrapper
`scatter_add_subword_bytes` `fuel-vulkan-backend/src/lib.rs:5843`.

```fkc
kernel: scatter_add_bf16
op_kind: ScatterAdd
blurb: "N-D scatter-add into a base copy via sub-word CAS atomic accumulate; bf16 I/O, f32 acc; nondeterministic."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::scatter_add_subword_bytes::bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "notes: base_shape; out shape == base_shape; agrees with src_shape on every axis != dim"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "same_as=src"   # indices.shape == src_shape; index value is the dim coordinate; no bounds clamp
    - name: src
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "same_rank=base"   # src_shape; differs from base only along dim
  op_params:
    variant: ScatterAdd           # OpParams::ScatterAdd (primitive namespace; §3.7)
    fields:
      base_shape: { kind: "Vec<usize>", note: "row-major base extents; out == base_shape; rank <= 8" }
      src_shape:  { kind: "Vec<usize>", constraint: "== indices.shape; agrees with base_shape on every axis != dim" }
      dim:        { kind: usize, constraint: "< base_shape.len()", note: "scattered axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # wrapper pre-inits out from base, then kernel atomically += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "(2 * base_total + 3 * n) * dtype_bytes"   # n = prod(src_shape); base_total = prod(base_shape)
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "base_total * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # bounded sub-word CAS may drop a value under contention; scheduler-dependent atomic order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 I/O accumulated in f32 (widen, add, narrow on store) via bounded sub-word CAS (1000 iters); may drop a value under extreme contention; order scheduler-dependent. Audited, no static bound applies."

determinism: nondeterministic
```

## scatter_add_f16  (N-D scatter-add into a base copy via sub-word atomic CAS, f16, f32 acc)

Same algorithm and rank-N layout as `scatter_add_f32`, for **F16** I/O computed with an **f32
accumulator** (widen the destination half-word + `src[p]` to f32, add, narrow on store) via a
**sub-word compare-and-swap** RMW of the 16-bit lane. Wrapper pre-inits out from `base`. The bounded
1000-iteration CAS loop may **drop a value** under extreme contention and the atomic order is
scheduler-dependent, so the kernel is **nondeterministic**. `dim < rank`, per-axis agreement, and byte
lengths are validated; no index bounds clamp. Output is **F16**, contiguous, shape == base, base-seeded
then atomically accumulated. Contiguous, offset-0 only. Source `scatter_add_f32.slang:46` (sub-word
variant); wrapper `scatter_add_subword_bytes` `fuel-vulkan-backend/src/lib.rs:5843`.

```fkc
kernel: scatter_add_f16
op_kind: ScatterAdd
blurb: "N-D scatter-add into a base copy via sub-word CAS atomic accumulate; f16 I/O, f32 acc; nondeterministic."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::scatter_add_subword_bytes::f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "notes: base_shape; out shape == base_shape; agrees with src_shape on every axis != dim"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "same_as=src"   # indices.shape == src_shape; index value is the dim coordinate; no bounds clamp
    - name: src
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1..=8
      shape_constraint: "same_rank=base"   # src_shape; differs from base only along dim
  op_params:
    variant: ScatterAdd           # OpParams::ScatterAdd (primitive namespace; §3.7)
    fields:
      base_shape: { kind: "Vec<usize>", note: "row-major base extents; out == base_shape; rank <= 8" }
      src_shape:  { kind: "Vec<usize>", constraint: "== indices.shape; agrees with base_shape on every axis != dim" }
      dim:        { kind: usize, constraint: "< base_shape.len()", note: "scattered axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # wrapper pre-inits out from base, then kernel atomically += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "(2 * base_total + 3 * n) * dtype_bytes"   # n = prod(src_shape); base_total = prod(base_shape)
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "base_total * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # bounded sub-word CAS may drop a value under contention; scheduler-dependent atomic order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O accumulated in f32 (widen, add, narrow on store) via bounded sub-word CAS (1000 iters); may drop a value under extreme contention; order scheduler-dependent. Audited, no static bound applies."

determinism: nondeterministic
```

## index_add_f32  (index-add into a base copy along one axis via atomic CAS, f32)

Index-add along a single axis: the **wrapper first copies `base` into `out`**, then for each
`i ∈ 0..n_indices` the kernel **atomically accumulates** `src[..., i, ...]` into
`out[..., indices[i], ...]`. The tensor is flattened to `[outer_count, base_dim_size, inner_count]`
for base/out and `[outer_count, n_indices, inner_count]` for src, with `outer_count`/`inner_count`
the products of the dims before/after the indexed axis. The index tensor is rank-1 `U32`, length
`n_indices`. The atomic add uses the same **bounded compare-and-swap on `uint`** (f32 bits) as
`scatter_add_f32` — under extreme contention the 1000-iteration CAS loop may **drop a value**, and FP
atomic accumulation order is scheduler-dependent, so the kernel is **nondeterministic**. A duplicate
index accumulates multiple `src` rows into the same destination. Output is **F32**, contiguous, shape
== base, base-seeded then atomically accumulated. Contiguous, offset-0 only. Source
`index_add_f32.slang:33`; wrapper `fuel-vulkan-backend/src/lib.rs:6176`
(`index_add_bytes_impl` `:6263`).

```fkc
kernel: index_add_f32
op_kind: IndexAdd
blurb: "Index-add into a base copy at rank-1 U32 indices along one axis via uint-CAS atomic accumulate; f32; nondeterministic."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::index_add_f32_bytes"
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
      shape_constraint: "notes: length n_indices; index value is the base axis coordinate; no bounds clamp"
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, n_indices, inner_count]"
  op_params:
    variant: IndexAdd             # OpParams::IndexAdd (primitive namespace; §3.7)
    fields:
      outer_count:   { kind: usize, note: "outer_count = product of dims before the indexed axis" }
      base_dim_size: { kind: usize, note: "base_dim_size = size of the indexed base axis" }
      n_indices:     { kind: usize, note: "n_indices = number of indices == src indexed-dim size" }
      inner_count:   { kind: usize, note: "inner_count = product of dims after the indexed axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # wrapper copies base -> out, then kernel atomically += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured     # Judge bootstraps; author declares only the derivable bandwidth hint below
  class: cheap_elementwise
  # base copy (base_dim_size) + atomic accumulate (n_indices) rows, each outer*inner f32 elements. Hint only.
  flops: "outer_count * n_indices * inner_count"
  bytes_moved: "(outer_count * base_dim_size * inner_count * 2 + outer_count * n_indices * inner_count * 2) * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "outer_count * base_dim_size * inner_count * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # bounded CAS may drop a value under contention; scheduler-dependent atomic order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 atomic accumulate via bounded uint-CAS (1000 iters); under extreme contention a value may be dropped and accumulation order is scheduler-dependent. Audited, no static bound applies."

determinism: nondeterministic
```

## index_add_f64  (index-add into a base copy along one axis via u64 atomic CAS, f64)

Same algorithm and `[outer_count, base_dim_size, inner_count]` flattening as `index_add_f32` (wrapper
copies `base` → `out`, then atomically `out[..., indices[i], ...] += src[..., i, ...]`), for **F64**
computed with a **`u64` compare-and-swap** (needs `shaderInt64` + 64-bit atomics + f64). The bounded
1000-iteration CAS loop may **drop a value** under extreme contention and the order is
scheduler-dependent, so the kernel is **nondeterministic**. Index tensor is rank-1 `U32`, length
`n_indices`; no index bounds clamp. Output is **F64**, contiguous, shape == base, base-seeded then
atomically accumulated. Contiguous, offset-0 only. Source `index_add_f32.slang:33` (f64 variant);
wrapper `index_add_bytes_impl` `fuel-vulkan-backend/src/lib.rs:6263`.

```fkc
kernel: index_add_f64
op_kind: IndexAdd
blurb: "Index-add into a base copy at rank-1 U32 indices via u64-CAS atomic accumulate; f64 (needs shaderInt64+atomics+f64); nondeterministic."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::index_add_f64_bytes"
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
      shape_constraint: "notes: length n_indices; index value is the base axis coordinate; no bounds clamp"
    - name: src
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, n_indices, inner_count]"
  op_params:
    variant: IndexAdd             # OpParams::IndexAdd (primitive namespace; §3.7)
    fields:
      outer_count:   { kind: usize, note: "outer_count = product of dims before the indexed axis" }
      base_dim_size: { kind: usize, note: "base_dim_size = size of the indexed base axis" }
      n_indices:     { kind: usize, note: "n_indices = number of indices == src indexed-dim size" }
      inner_count:   { kind: usize, note: "inner_count = product of dims after the indexed axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # wrapper copies base -> out, then kernel atomically += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "outer_count * n_indices * inner_count"
  bytes_moved: "(outer_count * base_dim_size * inner_count * 2 + outer_count * n_indices * inner_count * 2) * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "outer_count * base_dim_size * inner_count * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # bounded CAS may drop a value under contention; scheduler-dependent atomic order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f64 atomic accumulate via bounded u64-CAS (1000 iters); requires shaderInt64 + 64-bit atomics + f64. May drop a value under extreme contention; order scheduler-dependent. Audited, no static bound applies."

determinism: nondeterministic
```

## index_add_bf16  (index-add into a base copy along one axis via sub-word atomic CAS, bf16, f32 acc)

Same algorithm and `[outer_count, base_dim_size, inner_count]` flattening as `index_add_f32` (wrapper
copies `base` → `out`, then atomically accumulates `src` rows at the indexed positions), for **BF16**
I/O computed with an **f32 accumulator**: each destination half-word and the `src` element are widened
to f32, added, and narrowed back to bf16 on the atomic store via a **sub-word compare-and-swap** RMW
of the 16-bit lane. The bounded 1000-iteration CAS loop may **drop a value** under extreme contention
and the order is scheduler-dependent, so the kernel is **nondeterministic**. Index tensor is rank-1
`U32`, length `n_indices`; no index bounds clamp. Output is **BF16**, contiguous, shape == base,
base-seeded then atomically accumulated. Contiguous, offset-0 only. Source `index_add_f32.slang:33`
(sub-word variant); wrapper `index_add_bytes_impl` `fuel-vulkan-backend/src/lib.rs:6263`.

```fkc
kernel: index_add_bf16
op_kind: IndexAdd
blurb: "Index-add into a base copy at rank-1 U32 indices via sub-word CAS atomic accumulate; bf16 I/O, f32 acc; nondeterministic."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::index_add_bf16_bytes"
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
      shape_constraint: "notes: length n_indices; index value is the base axis coordinate; no bounds clamp"
    - name: src
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, n_indices, inner_count]"
  op_params:
    variant: IndexAdd             # OpParams::IndexAdd (primitive namespace; §3.7)
    fields:
      outer_count:   { kind: usize, note: "outer_count = product of dims before the indexed axis" }
      base_dim_size: { kind: usize, note: "base_dim_size = size of the indexed base axis" }
      n_indices:     { kind: usize, note: "n_indices = number of indices == src indexed-dim size" }
      inner_count:   { kind: usize, note: "inner_count = product of dims after the indexed axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # wrapper copies base -> out, then kernel atomically += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "outer_count * n_indices * inner_count"
  bytes_moved: "(outer_count * base_dim_size * inner_count * 2 + outer_count * n_indices * inner_count * 2) * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "outer_count * base_dim_size * inner_count * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # bounded sub-word CAS may drop a value under contention; scheduler-dependent atomic order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 I/O accumulated in f32 (widen, add, narrow on store) via bounded sub-word CAS (1000 iters); may drop a value under extreme contention; order scheduler-dependent. Audited, no static bound applies."

determinism: nondeterministic
```

## index_add_f16  (index-add into a base copy along one axis via sub-word atomic CAS, f16, f32 acc)

Same algorithm and `[outer_count, base_dim_size, inner_count]` flattening as `index_add_f32` (wrapper
copies `base` → `out`, then atomically accumulates `src` rows at the indexed positions), for **F16**
I/O computed with an **f32 accumulator** (widen the destination half-word + `src` to f32, add, narrow
on store) via a **sub-word compare-and-swap** RMW of the 16-bit lane. The bounded 1000-iteration CAS
loop may **drop a value** under extreme contention and the order is scheduler-dependent, so the kernel
is **nondeterministic**. Index tensor is rank-1 `U32`, length `n_indices`; no index bounds clamp.
Output is **F16**, contiguous, shape == base, base-seeded then atomically accumulated. Contiguous,
offset-0 only. Source `index_add_f32.slang:33` (sub-word variant); wrapper `index_add_bytes_impl`
`fuel-vulkan-backend/src/lib.rs:6263`.

```fkc
kernel: index_add_f16
op_kind: IndexAdd
blurb: "Index-add into a base copy at rank-1 U32 indices via sub-word CAS atomic accumulate; f16 I/O, f32 acc; nondeterministic."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::index_add_f16_bytes"
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
      shape_constraint: "notes: length n_indices; index value is the base axis coordinate; no bounds clamp"
    - name: src
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, n_indices, inner_count]"
  op_params:
    variant: IndexAdd             # OpParams::IndexAdd (primitive namespace; §3.7)
    fields:
      outer_count:   { kind: usize, note: "outer_count = product of dims before the indexed axis" }
      base_dim_size: { kind: usize, note: "base_dim_size = size of the indexed base axis" }
      n_indices:     { kind: usize, note: "n_indices = number of indices == src indexed-dim size" }
      inner_count:   { kind: usize, note: "inner_count = product of dims after the indexed axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)     # wrapper copies base -> out, then kernel atomically += src
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "outer_count * n_indices * inner_count"
  bytes_moved: "(outer_count * base_dim_size * inner_count * 2 + outer_count * n_indices * inner_count * 2) * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "outer_count * base_dim_size * inner_count * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # bounded sub-word CAS may drop a value under contention; scheduler-dependent atomic order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O accumulated in f32 (widen, add, narrow on store) via bounded sub-word CAS (1000 iters); may drop a value under extreme contention; order scheduler-dependent. Audited, no static bound applies."

determinism: nondeterministic
```

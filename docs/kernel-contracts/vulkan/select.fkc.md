---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — select / gather / masked-fill family kernel contracts

The Vulkan backend's byte-level **data-selection** primitives (crate `vulkan`, family `select`):
`OpKind::IndexSelect` (row-wise lookup along a dim by a rank-1 U32 index tensor),
`OpKind::Gather` (N-D gather along a dim by a same-shape U32 index tensor), and
`OpKind::MaskedFill` (per-element select between the data value and a constant fill by a U8 mask).
Every kernel here is a **pure byte / word data mover** — it copies the selected element's bytes
through with no arithmetic, so the result is **bit-identical on any hardware** (byte-exact,
`determinism: bitwise`).

**As-built binding model — production truth.** The three OpKinds register as PRIMITIVE bindings keyed
`(OpKind, [data_dtype, index_dtype, out_dtype], Vulkan) + kernel_source`; the index slot is `U32`
(IndexSelect / Gather) or `U8` (MaskedFill mask). Two wrapper shapes:

- **IndexSelect** has FOUR distinct per-dtype wrappers (`indexing::index_select_{f32,f16,bf16,f64}` →
  `VulkanBackend::index_select_*_bytes`); the `index_select` section below fans `[F32, F16, BF16, F64]`
  on the BASE `entry_point` (§3.4) so the importer resolves `index_select_<suffix>` to each per-dtype
  wrapper, byte-for-byte the deleted hand-written `register_with_precision(OpKind::IndexSelect, …)`
  regs.
- **Gather** and **MaskedFill** each dispatch through ONE dtype-agnostic wrapper (`gather::gather`,
  `masked_fill::masked_fill`) that picks its element byte-width from the OUTPUT dtype at the shim —
  so each fanned dtype key resolves to the SAME `KernelRef` (a **synthetic-base umbrella**, the CPU
  `pad_cpu` precedent). The `gather` / `masked_fill` sections fan `[F32, F16, BF16, F64, U8, U32]`;
  the link registry maps every `<base>_<suffix>` symbol to the one wrapper. Distinct dtype keys ⇒
  legal sibling registrations of one `KernelRef`, byte-for-byte the deleted per-dtype
  `register_with_precision` regs.

**Layout model — contiguous-only at the binding boundary (matches the as-built reg).** Every wrapper
reads its `source`/`data` + `indices`/`mask` as flat, contiguous, zero-offset buffers and writes a
fresh contiguous output — none walks a `Layout`/strides/offset, so the production registrations are
plain `register_with_precision` (no strided caps): `awkward_layout_strategy: requires_contiguous`
(`strided_input == false`), and the planner auto-Contiguizes a transposed / sliced / non-zero-offset
operand *first* and sums the `Op::Contiguize` cost (§4.3). Output is always freshly-allocated
**contiguous** row-major, no aliasing, not in-place (the universal output-contiguity rule).

**Cost provenance.** Every cost block is `judge_measured`: the Judge bootstraps it (§4.4). The
bandwidth `bytes_moved` hint is retained (a gather/select copy is genuinely bandwidth-bound; read the
selected element + write it). No overhead constant is fabricated; the imported `unknown_cost` sentinel
is upgraded to the shared OpKind cost fn by the `fill_unset_cost_for_backend` pass at registration.

**Determinism.** Pure byte/word copy — no FP arithmetic, no atomics, no reduction — so every kernel is
`determinism: bitwise` with an audited byte-exact precision (`max_ulp: 0`), byte-for-byte the deleted
regs' `VULKAN_BYTE_LEVEL_PRECISION`.

---

## index_select  (row-wise lookup along a dim by a rank-1 U32 index tensor; f32/f16/bf16/f64)

Pick `n_indices` slices from `source` along the selected axis: the tensor is flattened to
`[outer_count, source_dim_size, inner_count]` and for each `(outer, out_index, inner)` the source row
at the `indices[out_index]`-th position along the selected axis is copied through (pure data move, no
arithmetic). The index tensor is `U32`, contiguous, length `n_indices`. Output is the SAME dtype as
`source`, contiguous row-major, the selected dim resized to `n_indices`. FOUR distinct per-dtype
wrappers (`indexing::index_select_{f32,f16,bf16,f64}` → `VulkanBackend::index_select_*_bytes`); this
section fans the BASE `entry_point` over `[F32, F16, BF16, F64]`. Contiguous-only binding — a strided /
offset operand is auto-Contiguized first. Dispatch key `(IndexSelect, [T, U32, T], Vulkan)`.

```fkc
kernel: index_select
op_kind: IndexSelect
blurb: "Row-wise lookup along a dim by a rank-1 U32 index tensor; f32/f16/bf16/f64 byte copy; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::index_select"   # BASE symbol; fans index_select_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [F32, F16, BF16, F64]     # fans the per-dtype wrapper (§3.4); indices fixed U32
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "flat [outer_count, source_dim_size, inner_count]"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "length n_indices; index value is the selected-axis coordinate"
  op_params:
    variant: IndexSelect          # OpParams::IndexSelect (primitive namespace; §3.7)
    fields:
      outer_count:     { kind: usize, note: "product of dims before the selected axis" }
      source_dim_size: { kind: usize, note: "size of the selected source axis" }
      n_indices:       { kind: usize, note: "number of indices == output selected-dim size" }
      inner_count:     { kind: usize, note: "product of dims after the selected axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)      # data move preserves source dtype; key [T, U32, T]
      shape_rule: from_params(outer_count, n_indices, inner_count)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize + sums its cost (§4.3)
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * outer_count * n_indices * inner_count * dtype_bytes"   # read selected + write out

precision:
  bit_stable_on_same_hardware: true   # pure byte copy — exact, no arithmetic
  max_ulp: 0
  max_relative: 0.0
  max_absolute: 0.0
  audited: true
  notes: "exact byte-for-byte copy of selected slices; no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

---

## gather  (N-D gather along a dim by a same-shape U32 index tensor; f32/f16/bf16/f64/u8/u32)

N-dimensional gather: `source` and `indices`/`out` agree on every axis except `dim`, and the output
shape equals the index tensor's shape. For each output position the source is read at the same
multi-index except the `dim` coordinate, which is taken from the `U32` index value (pure data move,
no arithmetic). ONE dtype-agnostic wrapper (`gather::gather` → `VulkanBackend::gather_bytes`) picks
its element byte-width from the OUTPUT dtype at the shim, so every fanned dtype key resolves to the
SAME `KernelRef` (synthetic-base umbrella). This section fans `[F32, F16, BF16, F64, U8, U32]`; the
link registry maps every `gather_<suffix>` symbol to the one wrapper. Contiguous-only binding.
Dispatch key `(Gather, [T, U32, T], Vulkan)`.

```fkc
kernel: gather
op_kind: Gather
blurb: "N-D gather along a dim by a same-shape U32 index tensor; dtype-agnostic byte-width word move; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::gather"   # BASE symbol; fans gather_<suffix>, all → gather::gather
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [F32, F16, BF16, F64, U8, U32]   # fans the dtype key; the ONE wrapper is dtype-agnostic (byte-width from out)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "row-major source extents; rank <= 8"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "indices.shape == out_shape; agrees with source on every axis != dim"
  op_params:
    variant: Gather               # OpParams::Gather (primitive namespace; §3.7)
    fields:
      source_shape: { kind: "Vec<usize>", note: "row-major source extents; rank <= 8" }
      output_shape: { kind: "Vec<usize>", note: "== indices.shape; agrees with source on every axis != dim" }
      dim:          { kind: usize, constraint: "< source_shape.len()", note: "gathered axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)      # byte-width data move preserves source dtype; key [T, U32, T]
      shape_rule: from_params(output_shape)
      layout_guarantee: contiguous
      aliasing: none

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
  bytes_moved: "2 * n * dtype_bytes"   # n = prod(output_shape); read gathered element + write out

precision:
  bit_stable_on_same_hardware: true   # pure byte/word copy — exact, no arithmetic
  max_ulp: 0
  max_relative: 0.0
  max_absolute: 0.0
  audited: true
  notes: "exact byte-width gather copy; no arithmetic, bit-identical across any hardware. No bounds clamp on indices (caller pre-validates)."

determinism: bitwise
```

---

## masked_fill  (per-element select data-vs-constant by a U8 mask; f32/f16/bf16/f64/u8/u32)

Two inputs (`data`, `mask`) → one output: where `mask != 0` the output takes the constant fill bytes,
else it copies `data` through (a per-element byte select, no arithmetic). The mask is `U8`, contiguous;
the fill constant rides in `OpParams::MaskedFill.fill_bytes`. ONE dtype-agnostic wrapper
(`masked_fill::masked_fill` → `VulkanBackend::masked_fill_bytes`) picks its element byte-width from
the OUTPUT dtype at the shim, so every fanned dtype key resolves to the SAME `KernelRef` (synthetic-
base umbrella). This section fans `[F32, F16, BF16, F64, U8, U32]`; the link registry maps every
`masked_fill_<suffix>` symbol to the one wrapper. Contiguous-only binding. Dispatch key
`(MaskedFill, [T, U8, T], Vulkan)`.

```fkc
kernel: masked_fill
op_kind: MaskedFill
blurb: "Per-element select data-vs-constant by a U8 mask; dtype-agnostic byte-width copy; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::masked_fill"   # BASE symbol; fans masked_fill_<suffix>, all → masked_fill::masked_fill
kernel_revision_hash: auto

accept:
  inputs:
    - name: data
      dtypes: [F32, F16, BF16, F64, U8, U32]   # fans the dtype key; the ONE wrapper is dtype-agnostic (byte-width from out)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same shape as out"
    - name: mask
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same element count as data; mask != 0 selects the constant fill"
  op_params:
    variant: MaskedFill           # OpParams::MaskedFill (primitive namespace; §3.7)
    fields:
      fill_bytes: { kind: "Vec<u8>", note: "the constant fill value, dtype-width bytes" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(data)        # byte-width select preserves data dtype; key [T, U8, T]
      shape_rule: same_as(data)
      layout_guarantee: contiguous
      aliasing: none

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
  bytes_moved: "3 * n * dtype_bytes"   # n = element count; read data + read mask + write out

precision:
  bit_stable_on_same_hardware: true   # pure byte select — exact, no arithmetic
  max_ulp: 0
  max_relative: 0.0
  max_absolute: 0.0
  audited: true
  notes: "exact per-element byte select (data vs constant fill by U8 mask); no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

---
fkc_version: 1
provider:
  name: fuel-metal-kernels
  backend: Metal               # maps to BackendId::Metal
  kernel_source: "metal-msl"   # the BindingEntry.kernel_source tag
  link_registry: fuel_metal_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-metal-kernels — indexing contracts

Index / gather / scatter family from `indexing.metal` + `kernels/indexing.rs`, wired by
`fuel-metal-backend/src/storage.rs`. Six kernels: `index_select`, `gather`, `scatter`,
`scatter_add`, `index_add`, `where_cond`. Every kernel here uses an integer index/condition
operand and a value operand; the value-output dtype is a passthrough of the value/source dtype.
All math is exact element movement (copy / `+=`); there is no floating-point reduction, so the
precision story is byte-exact dtype-passthrough rather than a numeric bound. Rendered by mdBook;
the importer reads only the ` ```fkc ` blocks.

> **As-built dispatch-key note (faithfulness).** `index_select`, `gather`, `index_add`, and
> `scatter_add` each have a real `OpKind` (`fuel-core-types/src/dispatch.rs:326,331,339,345`) and
> an `OpParams` carrier (`fuel-dispatch/src/kernel.rs:413,424,455,467`). `where_cond` is
> `OpKind::Where` (`dispatch.rs:215`) and rides `OpParams::None` (it is an elementwise ternary —
> shape/stride data is derived from the operand layouts at dispatch; `kernel.rs:167`). **`scatter`
> (the SET form, `s_<idx>_<t>`) has no `OpKind` and no `OpParams` variant** in the as-built
> dispatch surface — only `OpKind::ScatterAdd` exists. It is a reachable *backend method*
> (`storage.rs:1489` `scatter_set`, exposed via `dyn_backend.rs:205` `scatter_set_dyn`) but is not
> registrable on today's binding table. Its section below is contracted faithfully against the
> kernel/backend facts and flagged `[consumer-ahead]`: it cannot be imported until an
> `OpKind::Scatter` + `OpParams::Scatter` carrier lands (the same posture as the spec's
> `MxNotYetRegistrable` / `GatherNotYetSupported` gaps, §6 / §3.9.1).

## index_select  (slice gather along dim with a rank-1 index tensor)

Pick slices from a source tensor along `dim` using an index tensor, one output slice per index:
`output[tid] = input[left*dimsz*right + id*right + right_off]` with `id` clamped to `[0, dim-1]`
and the sentinel max-value id writing 0. The index tensor must be contiguous (the backend
enforces this, `storage.rs:1589-1654`). The source supports a `contiguous` flag: when false, the
source element is addressed through `get_strided_index(src_i, src_dims, src_strides)`, so the
source may carry arbitrary / non-contiguous strides; when true the source is walked as dense.
Both source and ids are offset-capable via `BufferOffset`. Output is a freshly allocated
contiguous buffer whose shape is the source shape with `dim` replaced by the index count, and
whose dtype is the value dtype. Idx dtypes ∈ {I64, U32, U8}; value dtypes ∈ {U8, U32, I64, F32,
F16, BF16} (the per-idx subset the backend match wires, `storage.rs:1606-1633`). Pure data
movement — no arithmetic, so the result is byte-exact for the value dtype. Known limitation: the
index tensor is contiguous-only.

```fkc
kernel: index_select
op_kind: IndexSelect
blurb: "Slice gather along dim with a rank-1 index tensor; strided-capable source, contiguous ids; byte-exact passthrough."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::indexing::is_u32_f32"   # one per (idx,val) monomorph; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [U8, U32, I64, F32, F16, BF16]
      # source supports a `contiguous` flag: strided via get_strided_index, or dense.
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any
    - name: ids
      dtypes: [I64, U32, U8]
      # backend enforces contiguous ids; offset-capable via BufferOffset. Contiguous-only aux operand
      # under the kernel-wide handles_strided default ⇒ per-operand requires_contiguous override (§4.3.1/§10.5).
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected, awkward_layout_strategy: requires_contiguous }
      rank: 1
  op_params:
    variant: IndexSelect          # OpParams::IndexSelect (kernel.rs:413)
    fields:
      outer_count:     { kind: usize, note: "left_size; dims before `dim`" }
      source_dim_size: { kind: usize, note: "src_dim_size; index clamp bound" }
      n_indices:       { kind: usize, note: "ids_size; == output `dim` length" }
      inner_count:     { kind: usize, note: "right_size; dims after `dim`" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)
      shape_rule: from_params(source)   # source shape with `dim` replaced by n_indices
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # source walks arbitrary strides; ids must be contiguous (backend-enforced before dispatch)
  fast_paths:
    - { when: "all_inputs_contiguous", note: "source contiguous flag set; dense linear addressing" }
    - { when: "any_input_strided", note: "source via get_strided_index (mixed-radix de-linearize)" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured      # Judge bootstraps; author adds a real bandwidth hint only
  class: strided_elementwise
  # gather of n_indices slices each of inner_count elements; bandwidth-bound.
  flops: "0"                      # pure data movement, no arithmetic
  bytes_moved: "2 * n_indices * inner_count * dtype_bytes"   # read selected slice + write out
  overhead_ns: ~                  # per-device launch cost — Judge-measured (not a true zero; not a fabricated sentinel)
  memory: { device_bytes: "n_indices * inner_count * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # byte-exact element copy; no numeric op
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "pure index-select copy; output bytes are an exact copy of selected source elements; bit-identical."

determinism: bitwise   # exact copy; each tid owns a disjoint output element
```

## gather  (N-D gather along dim from a full index tensor)

Gather along `dim` where the index tensor has the output shape and supplies the source `dim`
coordinate at every output position: `output[tid] = input[(left*dimsz + ids[tid])*right +
right_off]`; a sentinel max-value id writes 0. The index tensor must be contiguous (backend,
`storage.rs:1439-1487`). Unlike `index_select`, the gather kernel addresses the source with an
**implied contiguous layout** — there is no strided remap inside the kernel — so the source is
effectively contiguous-assumed; both source and ids are offset-capable via `BufferOffset`. Output
is a fresh contiguous buffer with the index tensor's shape and the value dtype. Idx dtypes ∈ {U8,
U32, I64}; value dtypes ∈ {F32, F16, BF16, U8, U32, I64} (`indexing.metal:287-304`). Byte-exact
data movement.

```fkc
kernel: gather
op_kind: Gather
blurb: "N-D gather along dim from a full index tensor; contiguous-assumed source + contiguous ids; byte-exact passthrough."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::indexing::gather_u32_f32"   # one per (idx,val) monomorph; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [F32, F16, BF16, U8, U32, I64]
      # NO strided remap inside gather: source is addressed with an implied contiguous layout.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
    - name: ids
      dtypes: [U8, U32, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: same_rank=source   # ids has output shape; agrees with source on every dim but `dim`
  op_params:
    variant: Gather               # OpParams::Gather (kernel.rs:424)
    fields:
      source_shape: { kind: "Vec<usize>" }
      output_shape: { kind: "Vec<usize>", note: "== ids shape == out shape" }
      dim:          { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)
      shape_rule: same_as(ids)      # output shape == indices shape
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # source addressed contiguous-assumed; planner inserts Op::Contiguize (its own FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", note: "dense linear addressing; one thread per output element" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: strided_elementwise
  # n = product(output_shape) output elements; bandwidth-bound element gather.
  flops: "0"
  bytes_moved: "2 * n * dtype_bytes"   # read one source element + write one out element per output position
  overhead_ns: ~                       # per-device launch cost — Judge-measured (not a true zero; not a fabricated sentinel)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact element gather; output bytes copied verbatim from source; bit-identical."

determinism: bitwise   # exact copy; each tid owns a disjoint output element
```

## scatter  (N-D scatter SET along dim; in-place write into dst)

Scatter-write source values into a destination along `dim`: `out[(left*dst_dimsz + idx)*right +
r] = in[src_i]`, looping over `src_dim_size`; ids ≥ the max-value sentinel are skipped. Writes
into the caller's destination storage (the backend `scatter_set` mutates the dst buffer in place,
`storage.rs:1489-1537`). All three operands — destination, ids, and source — must be **contiguous**
(the backend enforces `RequiresContiguous`); byte offsets via `BufferOffset` are still applied,
and addressing inside the kernel is implicit row-major. Each `tid` owns a disjoint `(left,right)`
column, so the SET writes do not race. Idx dtypes ∈ {U8, U32, I64}; value dtypes ∈ {F32, F16,
BF16} (+ a U32/U32 monomorph; backend match `storage.rs:1501-1517`). Byte-exact element movement.

> **[consumer-ahead] — not registrable today.** There is no `OpKind::Scatter` (set) and no
> `OpParams::Scatter` variant in the as-built dispatch surface (`dispatch.rs` has only
> `ScatterAdd`; `kernel.rs` has only `OpParams::ScatterAdd`). The kernel and the backend method
> exist and are reachable (`scatter_set` / `scatter_set_dyn`), but this contract cannot be bound
> to the dispatch table until an `OpKind::Scatter` + `OpParams::Scatter` carrier lands. An importer
> that reaches this section before those types exist returns the same class of typed
> not-yet-supported error as the spec's `MxNotYetRegistrable` / `GatherNotYetSupported` gaps
> (§6 / §3.9.1). The `op_kind`/`op_params` slots below name the *intended* (not-yet-existing)
> carrier so the contract is complete and ready when the dispatch types arrive.

```fkc
kernel: scatter
registrable: false                # §3.10 describe-only: NO OpKind::Scatter / OpParams::Scatter (dispatch.rs has only ScatterAdd; kernel.rs only OpParams::ScatterAdd)
op_kind: Scatter                  # [consumer-ahead] intended carrier; no as-built OpKind::Scatter yet (see note)
blurb: "N-D scatter SET along dim; all-contiguous operands; in-place write into dst; byte-exact."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::indexing::s_u32_f32"   # one per (idx,val) monomorph; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: dst
      dtypes: [F32, F16, BF16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
    - name: ids
      dtypes: [U8, U32, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
    - name: src
      dtypes: [F32, F16, BF16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=ids
  op_params:
    variant: Scatter              # [consumer-ahead] intended OpParams::Scatter (no as-built variant)
    fields:
      outer_count:  { kind: usize, note: "left_size; dims before `dim`" }
      src_dim_size: { kind: usize, note: "scatter loop count along `dim`" }
      inner_count:  { kind: usize, note: "right_size; dims after `dim`" }
      dst_dim_size: { kind: usize, note: "destination `dim` extent" }
      dim:          { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(dst)
      shape_rule: same_as(dst)        # output IS the dst buffer; shape unchanged
      layout_guarantee: same_as(dst)  # written in place into dst's storage
      aliasing: in_place(dst)         # mutates caller's dst storage (requires caps.in_place)

caps:
  awkward_layout_strategy: requires_contiguous   # backend enforces RequiresContiguous on dst/ids/src
  fast_paths:
    - { when: "all_inputs_contiguous", note: "the only path; implicit row-major addressing, disjoint columns" }
  in_place: true                  # scatter_set mutates the dst buffer (§4.6)
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: strided_elementwise
  # writes src_dim_size * outer_count * inner_count elements into dst; bandwidth-bound.
  flops: "0"
  bytes_moved: "2 * outer_count * src_dim_size * inner_count * dtype_bytes"  # read src + write dst
  overhead_ns: ~                  # per-device launch cost — Judge-measured (not a true zero; not a fabricated sentinel)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # in-place: no fresh output alloc

precision:
  bit_stable_on_same_hardware: true   # exact element write; no numeric op
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact element SET into dst; each tid owns a disjoint (left,right) column so writes do not race; bit-identical."

determinism: bitwise   # disjoint-column exact writes; no atomic accumulation
```

## scatter_add  (N-D scatter-add along dim; in-place accumulate into dst)

The accumulate form of `scatter`: `out[(left*dst_dimsz + idx)*right + r] += in[src_i]`, looping
over `src_dim_size`; sentinel ids are skipped. The functional inverse of `gather`. Writes into the
caller's destination storage (`scatter_add_set`, `storage.rs:1539-1587`); the `+=` is **non-atomic**,
which is safe here because each `tid` owns a disjoint `(left,right)` column, so concurrent threads
never touch the same destination element. All three operands — dst, ids, src — must be
**contiguous** (backend `RequiresContiguous`); offsets via `BufferOffset` still apply. Idx dtypes ∈
{U8, U32, I64}; value dtypes ∈ {F32, F16, BF16} (+ U32/U32; backend match `storage.rs:1551-1567`).
Accumulation is in the value dtype `T` (no f32 promotion), so f16/bf16 sums round at the element
dtype. Implementation note from the inventory: the backend `scatter_add_set` dispatches the `sa_*`
add kernel through the generic `call_scatter` wrapper (the kernel-name string selects the add
variant).

```fkc
kernel: scatter_add
op_kind: ScatterAdd
blurb: "N-D scatter-add along dim; all-contiguous operands; in-place non-atomic += into dst (disjoint columns)."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::indexing::sa_u32_f32"   # one per (idx,val) monomorph; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: base                  # the accumulator destination (dst)
      dtypes: [F32, F16, BF16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
    - name: ids
      dtypes: [U8, U32, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
    - name: src
      dtypes: [F32, F16, BF16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=ids
  op_params:
    variant: ScatterAdd           # OpParams::ScatterAdd (kernel.rs:467)
    fields:
      base_shape: { kind: "Vec<usize>" }
      src_shape:  { kind: "Vec<usize>", note: "== ids shape" }
      dim:        { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)         # output IS base; shape unchanged
      layout_guarantee: same_as(base)   # accumulated in place into base's storage
      aliasing: accumulate(base)        # reads prior base content, adds src, writes back (§5.4)

caps:
  awkward_layout_strategy: requires_contiguous   # backend enforces RequiresContiguous on base/ids/src
  fast_paths:
    - { when: "all_inputs_contiguous", note: "the only path; disjoint-column non-atomic accumulate" }
  in_place: true                  # scatter_add_set mutates base in place (§4.6)
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: strided_elementwise
  # accumulates product(src_shape) source elements into base; bandwidth-bound (read base + read src + write base).
  flops: "n"                      # one add per scattered source element (n = product(src_shape))
  bytes_moved: "3 * n * dtype_bytes"   # read base elem + read src + write base
  overhead_ns: ~                  # per-device launch cost — Judge-measured (not a true zero; not a fabricated sentinel)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # in-place accumulate: no fresh output alloc

precision:
  bit_stable_on_same_hardware: true   # disjoint columns ⇒ deterministic order; accumulate in value dtype T
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "non-atomic += but each tid owns a disjoint (left,right) column so no race; accumulation in value dtype T (no f32 promotion); bit-identical on same hardware."

determinism: same_hardware_bitwise   # FP accumulate order is fixed (disjoint columns); not cross-hardware exact for half
```

## index_add  (accumulate src slices into a copied base along dim, rank-1 indices)

Accumulate source slices into a copy of `base` at index-selected positions along `dim`:
`out[(left*dst_dimsz + ids[j])*right + r] += in[(left*src_dimsz + j)*right + r]` over
`ids_dim_size`. Unlike `scatter_add`, the destination is a **fresh accumulator buffer**: the
backend copies `base` into a new buffer (`copy_strided_src` of the base into a fresh allocation,
`storage.rs:1656-1717`) and accumulates into that, so `index_add` is NOT in-place on the caller's
base. The ids and src operands must be **contiguous** (backend); all operands are offset-capable.
The `+=` is non-atomic and safe for the same disjoint-column reason as `scatter_add`. Idx dtypes ∈
{I64, U32, U8}; value dtypes ∈ {F16, F32, I64, U32, U8, BF16} (backend `storage.rs:1670-1697`).
Output dtype is the base/value dtype; accumulation in the value dtype `T`.

```fkc
kernel: index_add
op_kind: IndexAdd
blurb: "Accumulate src slices into a copied base along dim with rank-1 indices; fresh accumulator output; non-atomic disjoint-column +=."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::indexing::ia_u32_f32"   # one per (idx,val) monomorph; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: base                  # accumulator seed; copied into a fresh output buffer first
      dtypes: [F16, F32, I64, U32, U8, BF16]
      # base is copied via copy_strided_src into a fresh accumulator internally, so the kernel
      # absorbs the contiguize of a strided base — per-operand contiguize_internally (§4.3.1/§10.5).
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected, awkward_layout_strategy: contiguize_internally }
      rank: any
    - name: ids
      dtypes: [I64, U32, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 1
    - name: src
      dtypes: [F16, F32, I64, U32, U8, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
  op_params:
    variant: IndexAdd             # OpParams::IndexAdd (kernel.rs:455)
    fields:
      outer_count:  { kind: usize, note: "left_size; dims before `dim`" }
      base_dim_size: { kind: usize, note: "destination `dim` extent" }
      n_indices:    { kind: usize, note: "ids_dim_size; src `dim` extent" }
      inner_count:  { kind: usize, note: "right_size; dims after `dim`" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)
      shape_rule: same_as(base)         # output shape == base shape
      layout_guarantee: contiguous      # fresh contiguous accumulator buffer
      aliasing: none                    # NOT in-place: base is copied into a new buffer (§5.4)

caps:
  awkward_layout_strategy: requires_contiguous   # ids/src enforced contiguous by backend; base seed is copied via copy_strided_src
  fast_paths:
    - { when: "all_inputs_contiguous", note: "dense accumulate; disjoint (left,right) columns" }
  in_place: false                 # fresh accumulator buffer (base copied in)
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: strided_elementwise
  # copy base (outer_count*base_dim_size*inner_count) + accumulate n_indices slices of inner_count each.
  flops: "n_indices * inner_count * outer_count"   # one add per scattered source element
  bytes_moved: "(2 * outer_count * base_dim_size * inner_count + 3 * outer_count * n_indices * inner_count) * dtype_bytes"  # copy base (read+write) + accumulate (read base elem + read src + write base elem)
  overhead_ns: ~                  # per-device launch cost — Judge-measured (not a true zero; not a fabricated sentinel)
  memory: { device_bytes: "outer_count * base_dim_size * inner_count * dtype_bytes", host_bytes: 0, disk_bytes: 0 }   # fresh accumulator alloc

precision:
  bit_stable_on_same_hardware: true   # disjoint columns ⇒ deterministic accumulate order; accumulate in value dtype T
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "base copied then non-atomic += over disjoint (left,right) columns (no race); accumulation in value dtype T (no f32 promotion); bit-identical on same hardware."

determinism: same_hardware_bitwise   # fixed accumulate order (disjoint columns); not cross-hardware exact for half
```

## where_cond  (elementwise ternary select)

Elementwise ternary select: `out[i] = select(f[f_idx], t[t_idx], cond[idx])` — pick the
`t` (true) value where `cond != 0`, else the `f` (false) value. Each of the three operands
(`cond`, `t`, `f`) is independently contiguous OR strided, driven by three boolean **function
constants** (`IDS_CONTIGUOUS` / `T_CONTIGUOUS` / `F_CONTIGUOUS`) set at pipeline-compile time; the
strided path uses `get_strided_index`, so any operand may carry arbitrary / broadcast strides.
All three are offset-capable. Output is a fresh contiguous buffer whose shape follows the `t`
layout and whose dtype is the value dtype `T`. In the kernels crate cond ∈ {U8, U32, I64} and value
T ∈ {F16, F32, U8, U32, I64, BF16}, but the **Metal backend wires only** cond ∈ {U8, U32} with T ∈
{F32, F16, BF16, I64, U32, U8} (`storage.rs:868-879`); the graph builder additionally pins cond to
U8 (`fuel-graph/src/lib.rs:4887`). Pure selection — byte-exact passthrough of the chosen value
element. `OpKind::Where` rides `OpParams::None` (shape/stride data comes from the operand layouts).

```fkc
kernel: where_cond
op_kind: Where
blurb: "Elementwise ternary select out = cond!=0 ? t : f; per-operand contiguous-or-strided; byte-exact value passthrough."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::ternary::where_u8_f32"   # one per (cond,val) monomorph; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: cond
      dtypes: [U8, U32]            # backend-wired set; graph builder pins U8 (lib.rs:4887)
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any
    - name: t                      # value chosen where cond != 0
      dtypes: [F32, F16, BF16, I64, U32, U8]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any
    - name: f                      # value chosen where cond == 0
      dtypes: [F32, F16, BF16, I64, U32, U8]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=t
  op_params: { variant: None }     # OpParams::None — elementwise ternary; shape/strides from layouts (kernel.rs:167)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(t)       # value dtype T (== f dtype)
      shape_rule: same_as(t)           # shape from the t layout
      layout_guarantee: contiguous     # fresh contiguous buffer
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # each operand independently strided via per-operand function constants + get_strided_index
  fast_paths:
    - { when: "all_inputs_contiguous", note: "all three contiguity constants true; dense linear addressing" }
    - { when: "any_input_strided", note: "per-operand get_strided_index path (broadcast-capable)" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: strided_elementwise
  # n = product(out elements); read cond + read selected value + write out; bandwidth-bound.
  flops: "n"                       # one select branch per element
  bytes_moved: "3 * n * dtype_bytes"   # read cond + read chosen value element + write out (cond dtype ≤ value dtype; pessimistic)
  overhead_ns: ~                   # per-device launch cost — Judge-measured (not a true zero; not a fabricated sentinel)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # exact selection; the chosen value element is copied verbatim
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "pure ternary select; output is a byte-exact copy of the chosen t/f element; no arithmetic; bit-identical."

determinism: bitwise   # exact copy of the selected element; each tid owns a disjoint output element
```

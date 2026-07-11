---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda                    # maps to BackendId::Cuda
  kernel_source: "baracuda"        # the BindingEntry.kernel_source tag
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"    # provider build id, folded into kernel_revision_hash
---

# fuel-cuda-backend — indexing / gather / scatter kernel contracts

CUDA (baracuda) indexing kernels: `IndexSelect` / `Gather` (pick/gather by a `U32` index tensor),
`MaskedFill` (fill where a `U8` mask is nonzero), and `ScatterAdd` (accumulate `src` into a `base`
copy at `U32` indices — the functional inverse of gather). Bound in `baracuda_dispatch.rs` (`mod
indexing`) to the baracuda FFI (`fuel_cuda_backend::baracuda::indexing`). Every kernel here is a
**per-dtype typed wrapper** — unlike the CPU indexing family's dtype-agnostic byte-copy umbrella, the
CUDA `index_select_f32` / `index_select_f64` / `index_select_i32` (and likewise gather / masked_fill /
scatter_add) each resolve to a DISTINCT baracuda FFI symbol. So `IndexSelect` / `Gather` /
`MaskedFill` are authored as a **data-dtype fan** (§3.4: BASE `entry_point` → `<op>_<dtype>` resolved
through [`crate::fkc::CudaLinkRegistry`], each fanned symbol → its own wrapper — the binary/unary
precedent, not the cast/shape synthetic-base umbrella); `ScatterAdd` is authored **per-dtype**
(single-dtype `entry_point` resolved AS-IS, mirroring the CPU indexing contract's IndexAdd/ScatterAdd
split).

**The index operand is a FIXED single-dtype slot.** `IndexSelect` / `Gather` / `ScatterAdd` take a
`U32` index tensor; `MaskedFill` takes a `U8` mask. That slot carries exactly ONE dtype in every
section, so it is NOT an independent fan axis (the compare `U8`-mask / paged `U32` block-table
precedent) — the fan varies only the DATA dtype and the fixed index/mask rides through unfanned.
There is therefore no multi-axis `FanoutDtypeMismatch`; the whole family migrates. Binding keys
(inputs then output, §3.2):
- **IndexSelect** `[T, U32, T]` — `source` (data, fanned), `indices` (U32, fixed), `out:
  passthrough(source)`.
- **Gather** `[T, U32, T]` — `source` (data, fanned), `indices` (U32, fixed), `out:
  passthrough(source)`.
- **MaskedFill** `[T, U8, T]` — `source` (data, fanned), `mask` (U8, fixed), `out:
  passthrough(source)`.
- **ScatterAdd** `[T, U32, T, T]` — `base` (data), `indices` (U32, fixed), `src` (data), `out:
  passthrough(base)` — accumulated (out seeded from `base`, then `+= src`).

**Coverage (exactly production's 11 keys).** `IndexSelect` / `Gather` / `MaskedFill` over
`{F32, F64, I32}` (3 keys each = 9); `ScatterAdd` over `{F32, F64}` (2 keys) — narrower than the CPU
family (no BF16/F16 accumulator variants, no I64/U8/… byte-copy widths), trimmed to the baracuda FFI
symbols actually wired in `mod indexing` — byte-for-byte the deleted hand-written
`table.register(IndexSelect/Gather/MaskedFill/ScatterAdd, …)` regs.

**Universal facts for every kernel in this file.** Inputs are **contiguous-only** — the deleted
hand-written path used plain `table.register(...)` (default, all-false `KernelCaps`), so
`strided_input == false` and the planner inserts an `Op::Contiguize` (itself an FKC kernel, §4.3) for
any strided / broadcast / offset operand; the contract carries `awkward_layout_strategy:
requires_contiguous` with every operand layout `contiguous: required`. Out-of-bounds index values are
a hard `Result::Err` (never a panic, never a silent clamp). Output is freshly-allocated **contiguous**
(ScatterAdd is base-seeded then accumulated); no in-place. Cost is `judge_measured` — the
`fill_unset_cost_for_backend(Cuda, …)` pass (`dispatch.rs`) upgrades the imported `unknown_cost`
sentinel to the shared per-OpKind CUDA cost fn every other CUDA primitive gets (so cost is preserved
vs. the deleted plain-register path). Precision is the author-declared `audited: false` →
`PrecisionGuarantee::UNAUDITED` seed — byte-for-byte the deleted regs' default (they set no explicit
precision). Pointwise gather/scatter, no cross-thread reduction on the same destination for the
copy kernels, so bit-stable on the same hardware.

---

## index_select  (IndexSelect — pick slices along dim by a rank-1 U32 index tensor; data-dtype fan)

Pick `n_indices` slices from `source` along the selected axis using a rank-1 `U32` `indices` tensor;
the output's selected-dim size equals the index count. The flat layout is `[outer_count,
source_dim_size, inner_count]` for the source and `[outer_count, n_indices, inner_count]` for the
output, with `outer_count` = product of the dims before the selected axis and `inner_count` = product
of the dims after it. The index tensor is `U32`, contiguous, length `n_indices`; an index
`≥ source_dim_size` returns `Err`. Output is the **same dtype as the source**, contiguous row-major,
fully overwritten. The `source` operand fans over `{F32, F64, I32}` (§3.4), so the importer resolves
`index_select_<dtype>` and keys `[T, U32, T]`. Backs `OpKind::IndexSelect`.

```fkc
kernel: index_select
op_kind: IndexSelect
blurb: "Pick slices along dim by a rank-1 U32 index tensor (CUDA/baracuda) {F32, F64, I32}; contiguous; OOB index errors; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::index_select"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [F32, F64, I32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, source_dim_size, inner_count]"
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
      dtype_rule: passthrough(source)          # per-dtype typed copy preserves source dtype
      shape_rule: from_params(outer_count, n_indices, inner_count)   # selected axis size := n_indices
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured     # fill_unset upgrades the imported unknown_cost sentinel to the shared CUDA OpKind cost fn
  class: cheap_elementwise
  bytes_moved: "2 * outer_count * n_indices * inner_count * dtype_bytes"   # read selected + write out
  memory: { device_bytes: "outer_count * n_indices * inner_count * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # per-dtype copy — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "reasoned from source: index_select_kernel (baracuda/crates/baracuda-kernels-sys/kernels/include/baracuda_indexing.cuh) is one-thread-per-output-element with a single plain store `out[out_off] = src[src_off]` — no atomics, no shared-memory reduction, no launch-config-dependent ordering. Matches the Rust plan's own PrecisionGuarantee { bit_stable_on_same_hardware: true } (baracuda-kernels/src/indexing/index_select.rs)."

determinism: bitwise
```

---

## gather  (Gather — N-D gather along dim by a same-rank U32 index tensor; data-dtype fan)

N-dimensional gather: `source` and `indices`/`output` have the **same rank** and agree on every axis
except `dim`; the output shape equals the index tensor's shape. For each output position the source is
read at the same multi-index except the `dim` coordinate is taken from the `U32` index value. Rank
equality and the per-axis agreement (`source_shape[d] == output_shape[d]` for `d != dim`) are
validated; an index `≥ source_shape[dim]` returns `Err`. Output is the **same dtype as the source**,
contiguous row-major (shape == `output_shape`), fully overwritten. The `source` operand fans over
`{F32, F64, I32}` (§3.4), so the importer resolves `gather_<dtype>` and keys `[T, U32, T]`. Backs
`OpKind::Gather`.

```fkc
kernel: gather
op_kind: Gather
blurb: "N-D gather along dim by a same-rank U32 index tensor (CUDA/baracuda) {F32, F64, I32}; contiguous; OOB index errors; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::gather"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [F32, F64, I32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: source_shape; agrees with output_shape on every axis != dim"
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
      dtype_rule: passthrough(source)          # per-dtype typed copy preserves source dtype
      shape_rule: from_params(output_shape)     # output == index tensor shape
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "2 * n * dtype_bytes"   # n = prod(output_shape); read gathered element + write out
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # per-dtype copy — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "reasoned from source: gather_kernel (baracuda/crates/baracuda-kernels-sys/kernels/include/baracuda_indexing.cuh) is one-thread-per-output-element with a single plain store `out[out_off] = src[src_off]` — no atomics, no cross-thread accumulation (that's gather_backward_kernel, a distinct kernel not bound here). Matches the Rust plan's own PrecisionGuarantee { bit_stable_on_same_hardware: true } (baracuda-kernels/src/indexing/gather.rs)."

determinism: bitwise
```

---

## masked_fill  (MaskedFill — fill source where a U8 mask is nonzero; data-dtype fan)

Elementwise masked fill: for every position where the `U8` `mask` is nonzero, write the pre-encoded
`fill_bytes` value; elsewhere copy `source`. `mask` shares the element count with `source` (the kernel
reads the count from the layout); `fill_bytes` is one element's worth of the fill value, pre-encoded
in the output dtype and carried in `OpParams::MaskedFill`. Output is the **same dtype as the source**,
contiguous row-major, fully written. The `source` operand fans over `{F32, F64, I32}` (§3.4), so the
importer resolves `masked_fill_<dtype>` and keys `[T, U8, T]` (the `U8` mask is the fixed index-slot
analogue — the compare-mask precedent). Backs `OpKind::MaskedFill`.

```fkc
kernel: masked_fill
op_kind: MaskedFill
blurb: "Fill source where a U8 mask is nonzero (CUDA/baracuda) {F32, F64, I32}; contiguous; per-(op,dtype)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::masked_fill"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [F32, F64, I32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
    - name: mask
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=source"   # element-count match; nonzero → fill
  op_params:
    variant: MaskedFill           # OpParams::MaskedFill (primitive namespace; §3.7)
    fields:
      fill_bytes: { kind: "Vec<u8>", note: "one element's worth of the fill value, pre-encoded in the output dtype" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)
      shape_rule: same_as(source)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "2 * n * dtype_bytes"   # n = element count; read source + write out
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # copy / constant fill — no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "reasoned from source: masked_fill_kernel (baracuda/crates/baracuda-kernels-sys/kernels/include/baracuda_indexing.cuh) is one-thread-per-element with a single plain store `out[i] = mask[i] ? fill_value : src[i]` — no atomics, no shared state, no launch-config-dependent ordering. Matches the Rust plan's own PrecisionGuarantee { bit_stable_on_same_hardware: true } (baracuda-kernels/src/indexing/masked_fill.rs)."

determinism: bitwise
```

---

## scatter_add_f32  (ScatterAdd — N-D scatter-add into a base copy via a U32 index tensor, f32)

N-dimensional scatter-add, the functional inverse of `gather`: seed the output from `base`, then for
every flat position `p` in the `indices`/`src` shape, read `indices[p]` (`U32`) as the destination's
`dim` coordinate and accumulate `src[p]` into `base` at that multi-index. `base` and `src` share rank
and agree on every axis except `dim`; `indices` and `src` share shape. Rank equality, the per-axis
agreement, `dim < rank`, and byte lengths are validated; an index `≥ base_shape[dim]` returns `Err`.
Duplicate index values accumulate into the same destination. Output is **F32**, contiguous, shape ==
base, base-seeded then accumulated. This is a **per-dtype** section: the single-dtype `entry_point` is
resolved AS-IS (no fan), key `[F32, U32, F32, F32]`. Backs `OpKind::ScatterAdd`.

```fkc
kernel: scatter_add_f32
op_kind: ScatterAdd
blurb: "N-D scatter-add (inverse of gather) into a base copy via a U32 index tensor (CUDA/baracuda) f32; contiguous; OOB index errors."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::scatter_add_f32"
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

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "(2 * base_total + 3 * n) * dtype_bytes"   # n = prod(src_shape); base_total = prod(base_shape)
  memory: { device_bytes: "base_total * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "genuinely nondeterministic, NOT a static-bound gap: the kernel accumulates duplicate-index writes via atomicAdd (confirmed baracuda/crates/baracuda-kernels/src/indexing/scatter_add.rs, whose own doc comment states 'Precision guarantee: non-deterministic — atomicAdd ordering varies between launches'). Corrects a prior seed that wrongly carried bit_stable_on_same_hardware: true while audited: false — the false claim predates this audit; per-thread float addition is not associative, so accumulation order varies run-to-run for any index with >1 contributor."

determinism: nondeterministic
```

---

## scatter_add_f64  (ScatterAdd — N-D scatter-add into a base copy via a U32 index tensor, f64)

Same algorithm and layout as `scatter_add_f32` (seed out from `base`, then `out[dest(p)] += src[p]`
with the `dim` coordinate of `dest` taken from `indices[p]`), in **f64** native arithmetic. Rank
equality, per-axis agreement, `dim < rank`, and byte lengths are validated; an index
`≥ base_shape[dim]` returns `Err`. Output is **F64**, contiguous, shape == base, base-seeded then
accumulated. **Per-dtype** section: single-dtype `entry_point` resolved AS-IS, key
`[F64, U32, F64, F64]`. Backs `OpKind::ScatterAdd`.

```fkc
kernel: scatter_add_f64
op_kind: ScatterAdd
blurb: "N-D scatter-add (inverse of gather) into a base copy via a U32 index tensor (CUDA/baracuda) f64; contiguous; OOB index errors."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::scatter_add_f64"
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

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "(2 * base_total + 3 * n) * dtype_bytes"   # n = prod(src_shape); base_total = prod(base_shape)
  memory: { device_bytes: "base_total * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "genuinely nondeterministic, NOT a static-bound gap: the kernel accumulates duplicate-index writes via atomicAdd (confirmed baracuda/crates/baracuda-kernels/src/indexing/scatter_add.rs, whose own doc comment states 'Precision guarantee: non-deterministic — atomicAdd ordering varies between launches'). Corrects a prior seed that wrongly carried bit_stable_on_same_hardware: true while audited: false — the false claim predates this audit; per-thread float addition is not associative, so accumulation order varies run-to-run for any index with >1 contributor."

determinism: nondeterministic
```

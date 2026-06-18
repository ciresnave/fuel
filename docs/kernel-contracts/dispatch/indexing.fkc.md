---
fkc_version: 1
provider:
  name: fuel-dispatch
  backend: Cpu                  # the always-built universal fallback this bundle's blocks describe
  kernel_source: "portable-cpu" # the BindingEntry.kernel_source tag (CPU canonical binding)
  link_registry: fuel_dispatch::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"  # provider build id, folded into kernel_revision_hash
---

# fuel-dispatch — concat / indexing / gather / scatter / masked-fill kernel contracts

Dispatch-layer indexing-family kernels that `fuel-dispatch` registers across **three backends**:
the always-built **CPU** fallback (`register_cpu_kernels`, `dispatch.rs`), the single post-alpha.67
**baracuda CUDA** home (`register_baracuda_cuda_kernels`, `baracuda_dispatch.rs`, tag `CU`), and the
**Vulkan** compute-shader backend (`register_vulkan_kernels`, `vulkan_dispatch.rs`, tag `VK`). One
`(op, dtypes, backend)` key holds the binding for each backend; this bundle declares **one ` ```fkc `
block per `OpKind`** carrying the **canonical CPU binding** (the universal, always-built, bit-stable
fallback — front-matter `backend: Cpu`, `kernel_source: "portable-cpu"`), and the **prose for each
section faithfully records the full per-backend dtype coverage, layout caps, and precision/determinism
divergences** from the dispatch inventory. The backend-specific blocks (distinct `entry_point` /
`backend` / dtypes / caps that the importer registers as sibling alternatives at the same key, §12.5)
live in the per-backend bundles (`docs/kernel-contracts/cpu/indexing.fkc.md`,
`docs/kernel-contracts/vulkan/indexing.fkc.md`); the CPU block here is the dispatch-layer canonical
record.

Universal layout facts for this family (from the inventory): every CPU wrapper takes
`_layouts: &[Layout]` **unused** and operates on raw byte buffers (`CpuStorageBytes`); geometry comes
entirely from `OpParams`, so the CPU bindings are **contiguous-only, offset-0, not strided**. CUDA
and Vulkan vary per op (Concat is stride-aware on both; the index/gather/scatter/masked-fill family is
contiguous-only on every backend). The **index tensor is always `U32`**. No kernel in this family is
offset-capable (non-zero `start_offset` inputs always auto-Contiguize first). Output Storage is
**always pre-allocated by the executor** — no kernel allocates. Cost provenance is **`declared`** in
each block: every cost block carries an authored absolute launch prior (`overhead_ns: 40`), a
legitimate author prior the Judge later refines (§4.4) — so the block is `declared`, not
`judge_measured` (no authored absolute constant may sit under `judge_measured`). The `flops` /
`bytes_moved` strings are genuinely derivable bandwidth/FLOPs *hints* recorded as priors; the Judge
refines the absolute coefficients from measurement.

## concat  (variadic concatenation along an axis)

Concatenate `N` input tensors along `axis` into one output. The flat layout collapses to
`[outer_count, Σ input_dim_sizes[i], inner_count]`: `outer_count` is the product of the dims before
the concatenated axis (shared by every input), `inner_count` the product of the dims after it (shared),
and each input contributes `input_dim_sizes[i]` along the axis. The CPU wrapper is a
**dtype-agnostic byte-slab copy** (it reads the element size from the output Storage's dtype tag and
copies `input_dim_sizes[i] × inner_count × dtype_size` contiguous bytes per `(outer, input)` block);
it validates `input_dim_sizes.len() == inputs.len()`. Output dtype is the uniform input dtype `T`;
output shape equals the inputs concatenated along `axis`. **Per-backend coverage:** CPU dtypes
`f32, f64, bf16, f16, u32, u8, i16, i32, i64` (one byte-agnostic wrapper), **contiguous-only**
(`dispatch.rs:4525`); CU `f32, f64, f16, bf16`, **strided** (stride-aware when layouts are supplied,
`baracuda_dispatch.rs:2558`); VK `f32` (+ `f16/bf16/f64` feature-gated), **strided**
(`vulkan_dispatch.rs:4542`). The canonical short key is `[T, T]` (uniform-dtype N inputs + output).
The block below is the CPU binding; the CU/VK strided siblings are registered from the per-backend
bundles. Not in-place; contiguous output, offset-0.

```fkc
kernel: concat
op_kind: Concat
blurb: "Variadic concatenation of N same-dtype tensors along an axis; CPU dtype-agnostic byte-slab copy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::concat_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: inputs
      dtypes: [U8, I16, I32, I64, U32, BF16, F16, F32, F64]   # CPU byte-agnostic; uniform dtype across the N inputs
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: variadic N inputs, all same dtype; flat [outer_count, input_dim_sizes[i], inner_count]; agree on every axis != axis; input_dim_sizes.len() == inputs.len()"
  op_params:
    variant: Concat               # OpParams::Concat (primitive namespace; §3.7)
    fields:
      outer_count:     { kind: usize, note: "product of dims before the concatenated axis (shared by all inputs)" }
      input_dim_sizes: { kind: "Vec<usize>", constraint: "len() == inputs.len()", note: "per-input extent along the concatenated axis" }
      inner_count:     { kind: usize, note: "product of dims after the concatenated axis (shared by all inputs)" }
      axis:            { kind: usize, note: "concatenated axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(inputs)          # uniform input dtype; byte-slab copy preserves it
      shape_rule: from_params(outer_count, sum(input_dim_sizes), inner_count)   # axis size := Σ input_dim_sizes
      layout_guarantee: contiguous
      aliasing: none
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous   # CPU contiguous-only; planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost. (CU/VK siblings declare handles_strided.)
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared           # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4); bandwidth hint below is a derivable prior
  class: cheap_elementwise
  # bandwidth-bound concat copy: read+write every output element once (n = product of output shape). Hint only.
  flops: "0"
  bytes_moved: "2 * n * dtype_bytes"   # n = outer_count * sum(input_dim_sizes) * inner_count; read each input element + write out
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # pure byte-slab copy — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact byte-for-byte slab copy of each input along the axis; no arithmetic, bit-identical across any hardware. (CU/VK siblings are equally exact data moves.)"

determinism: bitwise
```

## index_select  (pick slices along a dim by a rank-1 U32 index tensor)

Pick `n_indices` slices from `source` along the selected axis using a **rank-1 `U32`** `indices`
tensor; the output's selected-dim size equals the index count. The flat layout is
`[outer_count, source_dim_size, inner_count]` for the source and `[outer_count, n_indices, inner_count]`
for the output, with `outer_count` the product of the dims before the selected axis and `inner_count`
the product of the dims after it. Implemented on CPU as a **dtype-agnostic byte copy** (parameterized
by element size; copies `inner_count × dtype_size` contiguous bytes per `(outer, index)` pair). The
key is `[data, U32, data]`. **Per-backend coverage:** CPU dtypes
`f32, f64, bf16, f16, u32, u8, i16, i32, i64`, **contiguous-only** (`dispatch.rs:4550`); CU
`f32, f64, i32`, **contiguous-only** (default caps, `baracuda_dispatch.rs:2488`); VK `f32`
(+ `f16/bf16/f64` gated), **contiguous-only** (byte-level, `vulkan_dispatch.rs:4435`). All three
backends are contiguous-only for this op. Out-of-bounds index behavior is backend-specific (CPU errors;
the Vulkan shader clamps to `axis_in-1` — see the per-backend bundle); the dispatch-layer CPU contract
errors on OOB (never panic, never silent clamp). Output is the **same dtype as the source**, contiguous
row-major, fully overwritten. Offset-0 only.

```fkc
kernel: index_select
op_kind: IndexSelect
blurb: "Pick slices along a dim by a rank-1 U32 index tensor; CPU dtype-agnostic byte copy; OOB index errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::index_select_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [U8, I16, I32, I64, U32, BF16, F16, F32, F64]   # CPU dtype-agnostic: copied by byte width
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, source_dim_size, inner_count]"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "notes: length n_indices; each value < source_dim_size or the CPU kernel returns Err"
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
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous   # all backends contiguous-only; planner inserts Op::Contiguize + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared           # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4); bandwidth hint below is a derivable prior
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
  notes: "exact byte-for-byte copy of selected slices; no arithmetic, bit-identical across any hardware. CPU errors on OOB index (VK sibling clamps to axis_in-1)."

determinism: bitwise
```

## gather  (N-D gather along a dim by a same-shape U32 index tensor)

N-dimensional gather: `source` and `indices`/`output` agree on every axis except `dim`, and the output
shape equals the `U32` index tensor's shape. For each output position the source is read at the same
multi-index except the `dim` coordinate, which is taken from the index value. The CPU kernel is a
**dtype-agnostic byte copy** (row-major source strides from `source_shape`, output walked flat and
unraveled against `output_shape`); rank equality and the per-axis agreement
(`source_shape[d] == output_shape[d]` for `d != dim`) are validated. The key is `[data, U32, data]`.
**Per-backend coverage:** CPU dtypes `f32, f64, bf16, f16, u32, u8, i16, i32, i64`, **contiguous-only**
(`dispatch.rs:4551`); CU `f32, f64, i32`, **contiguous-only** (`baracuda_dispatch.rs:2492`); VK `f32`
(+ `f16/bf16/f64/u8/u32` gated), **contiguous-only** (byte-level, `vulkan_dispatch.rs:4584`). OOB index
behavior is backend-specific (CPU errors; the Vulkan byte-width gather applies **no bounds clamp** and
relies on the caller to pre-validate — see the per-backend bundle); the dispatch-layer CPU contract
errors on an index `≥ source_shape[dim]`. Output is the **same dtype as the source**, contiguous
row-major (shape == `output_shape`), fully overwritten. Offset-0 only.

```fkc
kernel: gather
op_kind: Gather
blurb: "N-D gather along a dim by a same-rank U32 index tensor; CPU dtype-agnostic byte copy; OOB index errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::gather_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [U8, I16, I32, I64, U32, BF16, F16, F32, F64]   # CPU dtype-agnostic: copied by byte width
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: source_shape; bytes == prod(source_shape)*dtype_size"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=source"   # source/output/indices share rank; differ only on dim; each value < source_shape[dim] (CPU errors otherwise)
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
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous   # all backends contiguous-only; planner inserts Op::Contiguize + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared           # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4); bandwidth hint below is a derivable prior
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
  notes: "exact byte-for-byte gather copy; no arithmetic, bit-identical across any hardware. CPU errors on OOB index (VK sibling applies no bounds clamp)."

determinism: bitwise
```

## index_add  (accumulate src into a copy of base at rank-1 U32 indices along one axis)

Index-add along a single axis: **seed the output from `base`**, then for each `i ∈ 0..n_indices`
accumulate `src[..., i, ...]` into `out[..., indices[i], ...]`. Output shape equals `base`; the flat
layout is `[outer_count, base_dim_size, inner_count]` for base/out and
`[outer_count, n_indices, inner_count]` for src. The index tensor is rank-1 `U32`, length `n_indices`.
This is **arithmetic accumulation** (read-modify-write of a base copy, **not** a pure overwrite): a
duplicate index accumulates multiple `src` rows into the same destination. f32/f64 accumulate
natively; bf16/f16 widen to an **f32 accumulator** and narrow on store (the load-bearing half-float
invariant). The key is `[base, U32, src, out]`. **Per-backend coverage:** CPU dtypes
`f32, f64, bf16, f16`, **contiguous-only**, and the CPU accumulation is **deterministic in index
order** → bit-stable on the same hardware (`dispatch.rs:4564`); VK `f32, f64, bf16, f16`,
**contiguous-only**, but the Vulkan path accumulates via a **bounded CAS atomic** that is
`PrecisionGuarantee::none` / **nondeterministic** (scheduler-dependent order; may drop a value under
extreme contention — see the per-backend bundle). **There is no baracuda CUDA binding for IndexAdd in
this crate.** OOB index `≥ base_dim_size` errors on CPU. Output dtype = base, contiguous, shape ==
base, base-seeded then accumulated. Offset-0 only. The block below is the deterministic CPU binding.

```fkc
kernel: index_add
op_kind: IndexAdd
blurb: "Accumulate src into a base copy at rank-1 U32 indices along one axis; f32/f64 native, bf16/f16 via f32 acc; CPU deterministic, OOB errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::index_add_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F32, F64, BF16, F16]   # half I/O accumulates in f32
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, base_dim_size, inner_count]; out shape == base shape"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "notes: length n_indices; each value < base_dim_size or the CPU kernel returns Err"
    - name: src
      dtypes: [F32, F64, BF16, F16]   # same dtype as base
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: flat [outer_count, n_indices, inner_count]; same dtype as base"
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
  awkward_layout_strategy: requires_contiguous   # all backends contiguous-only; planner inserts Op::Contiguize + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared           # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4); formula hints below are derivable priors
  class: cheap_elementwise
  # base copy (base_dim_size) + accumulate (n_indices) rows, each outer*inner elements. Hint only.
  flops: "outer_count * n_indices * inner_count"   # one += per accumulated element
  bytes_moved: "(outer_count * base_dim_size * inner_count * 2 + outer_count * n_indices * inner_count) * dtype_bytes"   # base read+write + src read
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * base_dim_size * inner_count * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # CPU: deterministic in-order accumulation (f32/f64 native; half via f32 acc, narrow on store)
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU native f32/f64 += (half widened to f32, narrowed on store), deterministic index order; bit-stable on same hardware. Duplicate indices accumulate in index order. VK sibling is nondeterministic (bounded CAS atomic)."

determinism: same_hardware_bitwise
```

## scatter_add  (N-D scatter-add — functional inverse of gather — into a base copy)

N-dimensional scatter-add, the functional inverse of `gather`: **seed the output from `base`**, then
for every flat position `p` in the `indices`/`src` shape read `indices[p]` (`U32`) as the destination's
`dim` coordinate and accumulate `src[p]` into `base` at that multi-index. `base` and `src` share rank
and agree on every axis except `dim`; `indices` and `src` share the same shape. This is **arithmetic
accumulation** (read-modify-write of a base copy): duplicate index values accumulate into the same
destination. f32/f64 accumulate natively; bf16/f16 use an **f32 accumulator** (widen, add, narrow on
store). The key is `[base, U32, src, out]`. **Per-backend coverage:** CPU dtypes `f32, f64, bf16, f16`,
**contiguous-only**, **deterministic in flat-`p` order** → bit-stable on same hardware
(`dispatch.rs:4568`); CU `f32, f64`, **contiguous-only** (`baracuda_dispatch.rs:2500`); VK
`f32, f64, bf16, f16`, **contiguous-only**, but `PrecisionGuarantee::none` / **nondeterministic** (the
Vulkan path uses a bounded CAS atomic — scheduler-dependent order, may drop a value under contention,
see the per-backend bundle). OOB index `≥ base_shape[dim]` errors on CPU. Output dtype = base,
contiguous, shape == base, base-seeded then accumulated. Offset-0 only. The block below is the
deterministic CPU binding.

```fkc
kernel: scatter_add
op_kind: ScatterAdd
blurb: "N-D scatter-add (inverse of gather) into a base copy via a same-shape U32 index tensor; f32/f64 native, bf16/f16 via f32 acc; CPU deterministic, OOB errors."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::scatter_add_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F32, F64, BF16, F16]   # half I/O accumulates in f32
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "notes: base_shape; out shape == base_shape; agrees with src_shape on every axis != dim"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=src"   # indices.shape == src_shape; each value < base_shape[dim] or the CPU kernel returns Err
    - name: src
      dtypes: [F32, F64, BF16, F16]   # same dtype as base
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=base"   # src_shape; differs from base only along dim; same dtype as base
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
  awkward_layout_strategy: requires_contiguous   # all backends contiguous-only; planner inserts Op::Contiguize + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared           # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4); formula hints below are derivable priors
  class: cheap_elementwise
  # base copy (prod(base_shape)) + one += per src element (n = prod(src_shape)). Hint only.
  flops: "n"   # n = prod(src_shape); one += per scattered element
  bytes_moved: "(2 * base_total + 3 * n) * dtype_bytes"   # base read+write + src read + dest RMW; base_total = prod(base_shape)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "base_total * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # CPU: deterministic flat-p-order accumulation (f32/f64 native; half via f32 acc, narrow on store)
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU native f32/f64 += (half widened to f32, narrowed on store), deterministic flat-position order; bit-stable on same hardware. Duplicate indices accumulate in flat-p order. VK sibling is nondeterministic (bounded CAS atomic)."

determinism: same_hardware_bitwise
```

## masked_fill  (write a scalar fill where a U8 mask is nonzero)

Element-wise masked fill: `out[i] = mask[i] != 0 ? fill : x[i]`. The fill value arrives **pre-encoded**
as `fill_bytes` — one element already serialized in the output dtype, so the kernel writes raw bytes
without re-encoding (a **dtype-agnostic byte op** on CPU). The `mask` is `U8` (any nonzero byte selects
the fill); `x` and `mask` share the broadcast/element shape and the output shape equals `x`. The key is
`[T, U8, T]` (`x`, `mask`, `out`). **Per-backend coverage:** CPU dtypes `f32, f64, bf16, f16, u32, u8`
(one dtype-agnostic byte kernel), **contiguous-only** (`dispatch.rs:4453`); CU `f32, f64, i32`,
**contiguous-only** (`baracuda_dispatch.rs:2496`); VK `f32, f16, bf16, f64, u8, u32`, **contiguous-only**
(byte-level, `vulkan_dispatch.rs:4573`). Output dtype = `x`, contiguous row-major, fully overwritten
(it is a fresh buffer, not an in-place RMW of `x` — the kernel writes either the passthrough or the
fill at every position). Offset-0 only. The block below is the CPU binding.

```fkc
kernel: masked_fill
op_kind: MaskedFill
blurb: "Write a pre-encoded scalar fill where a U8 mask is nonzero, else passthrough x; CPU dtype-agnostic byte op."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::masked_fill_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [U8, U32, BF16, F16, F32, F64]   # CPU byte-agnostic; output dtype follows x
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=mask"   # x and mask share the element shape; out shape == x
    - name: mask
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=x"   # any nonzero byte selects the fill
  op_params:
    variant: MaskedFill           # OpParams::MaskedFill (primitive namespace; §3.7)
    fields:
      fill_bytes: { kind: "Vec<u8>", note: "ONE element pre-encoded in the OUTPUT dtype; written raw where mask != 0 (no re-encoding)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)               # output dtype follows x (fill_bytes is pre-encoded in this dtype)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none                           # fresh buffer: passthrough-or-fill written at every position
  bundle: ~

caps:
  awkward_layout_strategy: requires_contiguous   # all backends contiguous-only; planner inserts Op::Contiguize + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared           # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4); bandwidth hint below is a derivable prior
  class: cheap_elementwise
  # bandwidth-bound elementwise: read x + mask, write out, one element per position. Hint only.
  flops: "n"   # n = product of x shape; one compare + select per element
  bytes_moved: "2 * n * dtype_bytes + n"   # read x + write out (dtype_bytes each) + read 1-byte mask
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # byte select/copy — exact, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact byte select: passthrough x or raw pre-encoded fill_bytes where mask != 0; no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

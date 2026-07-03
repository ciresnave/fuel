---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — scatter-add / index-add family kernel contracts

The Vulkan backend's **atomic accumulate** primitives (crate `vulkan`, family `scatter`):
`OpKind::IndexAdd` (index-add into a base copy at rank-1 U32 indices along one axis) and
`OpKind::ScatterAdd` (N-D scatter-add, the functional inverse of gather). Both **seed the output from
`base`** (the wrapper copies `base → out`), then **atomically accumulate** `src` into the output at
the index-derived positions via a **bounded compare-and-swap (CAS) atomic add** — f32 via `uint` CAS,
f64 via `u64` CAS, bf16/f16 via sub-word CAS (math in f32, narrow on store). The CAS loop is bounded
(1000 iterations): under extreme contention a value can be dropped, and FP atomic accumulation order
is scheduler-dependent, so these kernels are **nondeterministic** (run-to-run variation possible),
declared honestly per §4.9.

**As-built binding model — production truth.** Each OpKind registers as a PRIMITIVE binding keyed
`(OpKind, [base_dtype, U32, src_dtype, out_dtype], Vulkan) + kernel_source` — a **4-slot** key
`[base, indices, src, out]`. Each op has FOUR distinct per-dtype wrappers
(`index_add::index_add_{f32,f64,bf16,f16}`, `scatter_add::scatter_add_{f32,f64,bf16,f16}` →
`VulkanBackend::index_add_*` / `scatter_add_*` paths). The sections below fan the BASE `entry_point`
over `[F32, F64, BF16, F16]`: `base` and `src` BOTH enumerate that shared list, so the importer fans
them TOGETHER (one fan over the shared list, §3.4 — not a `FanoutDtypeMismatch`, which only fires on
DIFFERENT lists), keying `[T, U32, T, T]` byte-for-byte the deleted hand-written
`register_with_precision(OpKind::{IndexAdd,ScatterAdd}, &[T, U32, T, T], …)` regs.

**Layout model — contiguous-only at the binding boundary (matches the as-built reg).** Every wrapper
reads `base`/`indices`/`src` as flat, contiguous, zero-offset buffers and writes a fresh
base-seeded output — none walks a `Layout`/strides/offset, so the production registrations are plain
`register_with_precision` (no strided caps): `awkward_layout_strategy: requires_contiguous`
(`strided_input == false`), and the planner auto-Contiguizes a strided / offset operand *first* and
sums the `Op::Contiguize` cost (§4.3). Output is always the freshly-allocated **contiguous** copy of
`base` (`aliasing: accumulate(base)` — the wrapper pre-inits out from base, then the kernel atomically
`+= src`), not in-place on an input.

**Cost provenance.** Every cost block is `judge_measured`: the Judge bootstraps it (§4.4). The FLOPs
hint (one add per accumulated `src` element) and the base-copy + accumulate `bytes_moved` are the
derivable structure; no overhead constant is fabricated. The imported `unknown_cost` sentinel is
upgraded to the shared OpKind cost fn by `fill_unset_cost_for_backend` at registration.

**Determinism (honest).** Bounded CAS may drop a value under extreme contention and the atomic order
is scheduler-dependent, so each kernel is `determinism: nondeterministic` with
`bit_stable_on_same_hardware: false` and an audited `none(reason)` precision (no silent unaudited
nondeterminism, §10 rule 9) — byte-for-byte the deleted regs' `PrecisionGuarantee::none(reason)`.

---

## index_add  (index-add into a base copy at rank-1 U32 indices along one axis; f32/f64/bf16/f16)

Index-add along a single axis: the wrapper first copies `base` into `out`, then for each
`i ∈ 0..n_indices` the kernel **atomically accumulates** `src[..., i, ...]` into
`out[..., indices[i], ...]`. The tensors are flattened to `[outer_count, base_dim_size, inner_count]`
(base/out) and `[outer_count, n_indices, inner_count]` (src); the index tensor is rank-1 `U32`,
length `n_indices`. FOUR distinct per-dtype wrappers (`index_add::index_add_{f32,f64,bf16,f16}` →
`VulkanBackend::index_add_bytes` paths, uint/u64/sub-word CAS); this section fans the BASE
`entry_point` over `[F32, F64, BF16, F16]` (base + src share the list). Contiguous-only binding;
nondeterministic (bounded CAS). Dispatch key `(IndexAdd, [T, U32, T, T], Vulkan)`.

```fkc
kernel: index_add
op_kind: IndexAdd
blurb: "Index-add into a base copy at rank-1 U32 indices along one axis via bounded-CAS atomic accumulate; f32/f64/bf16/f16; nondeterministic; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::index_add"   # BASE symbol; fans index_add_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F32, F64, BF16, F16]     # fans the per-dtype wrapper (§3.4); src shares the list
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "flat [outer_count, base_dim_size, inner_count]; out shape == base shape"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "length n_indices; index value is the base axis coordinate; no bounds clamp"
    - name: src
      dtypes: [F32, F64, BF16, F16]     # shares base's list ⇒ fans together (one fan, not a mismatch)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "flat [outer_count, n_indices, inner_count]"
  op_params:
    variant: IndexAdd             # OpParams::IndexAdd (primitive namespace; §3.7)
    fields:
      outer_count:   { kind: usize, note: "product of dims before the indexed axis" }
      base_dim_size: { kind: usize, note: "size of the indexed base axis" }
      n_indices:     { kind: usize, note: "number of indices == src indexed-dim size" }
      inner_count:   { kind: usize, note: "product of dims after the indexed axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)        # key [T, U32, T, T]
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)           # wrapper copies base -> out, then kernel atomically += src

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
  flops: "outer_count * n_indices * inner_count"
  bytes_moved: "(outer_count * base_dim_size * inner_count + 2 * outer_count * n_indices * inner_count) * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false   # bounded CAS may drop a value under contention; scheduler-dependent atomic order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                        # audited none(reason): no static bound (nondeterministic FP atomic accumulate)
  notes: "atomic float add via uint/u64/sub-word bounded CAS (1000 iters); f16/bf16 accumulate in f32 (widen/narrow); may drop a value under extreme contention; order scheduler-dependent; NOT bit-stable."

determinism: nondeterministic
```

---

## scatter_add  (N-D scatter-add into a base copy via atomic CAS; f32/f64/bf16/f16)

N-dimensional scatter-add, the functional inverse of gather: the wrapper pre-initializes `out` to
`base`, then for every flat position `p` in the `indices`/`src` shape the kernel reads `indices[p]`
(`U32`) as the destination's `dim` coordinate and **atomically accumulates** `src[p]` into the output
at that multi-index. `base` and `src` agree on every axis except `dim`; `indices` and `src` share the
same shape. FOUR distinct per-dtype wrappers (`scatter_add::scatter_add_{f32,f64,bf16,f16}` →
`VulkanBackend::scatter_add_*` paths, uint/u64/sub-word CAS); this section fans the BASE `entry_point`
over `[F32, F64, BF16, F16]` (base + src share the list). Contiguous-only binding; nondeterministic
(bounded CAS). Dispatch key `(ScatterAdd, [T, U32, T, T], Vulkan)`.

```fkc
kernel: scatter_add
op_kind: ScatterAdd
blurb: "N-D scatter-add (inverse of gather) into a base copy via bounded-CAS atomic accumulate; f32/f64/bf16/f16; nondeterministic; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::scatter_add"   # BASE symbol; fans scatter_add_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: base
      dtypes: [F32, F64, BF16, F16]     # fans the per-dtype wrapper (§3.4); src shares the list
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "row-major base extents; out == base_shape; agrees with src on every axis != dim; rank <= 8"
    - name: indices
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "indices.shape == src.shape; index value is the dim coordinate; no bounds clamp"
    - name: src
      dtypes: [F32, F64, BF16, F16]     # shares base's list ⇒ fans together (one fan, not a mismatch)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "differs from base only along dim"
  op_params:
    variant: ScatterAdd           # OpParams::ScatterAdd (primitive namespace; §3.7)
    fields:
      base_shape: { kind: "Vec<usize>", note: "row-major base extents; out == base_shape; rank <= 8" }
      src_shape:  { kind: "Vec<usize>", note: "== indices.shape; agrees with base on every axis != dim" }
      dim:        { kind: usize, constraint: "< base_shape.len()", note: "scattered axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(base)        # key [T, U32, T, T]
      shape_rule: same_as(base)
      layout_guarantee: contiguous
      aliasing: accumulate(base)           # wrapper pre-inits out from base, then kernel atomically += src

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
  flops: "n"
  bytes_moved: "(2 * base_total + 3 * n) * dtype_bytes"   # n = prod(src_shape); base_total = prod(base_shape)

precision:
  bit_stable_on_same_hardware: false   # bounded CAS may drop a value under contention; scheduler-dependent atomic order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                        # audited none(reason): no static bound (nondeterministic FP atomic accumulate)
  notes: "atomic float add via uint/u64/sub-word bounded CAS (1000 iters); f16/bf16 accumulate in f32 (widen/narrow); may drop a value under extreme contention; order scheduler-dependent; NOT bit-stable."

determinism: nondeterministic
```

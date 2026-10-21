---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — write-slice family kernel contracts

The Vulkan backend's **destination-slab write** primitives (crate `vulkan`, family `write-slice`):
`OpKind::WriteSlice` (write a source tensor into a slab of a larger destination) and
`OpKind::WriteSliceRotating` (the sliding-window KV variant — the wrapper handles the position D2H +
ring-boundary split before delegating to the same slab-write path). Both are **pure byte / word data
movers** — they copy `dtype_size` bytes per element with no arithmetic, so the result is
**bit-identical on any hardware** (`determinism: bitwise`, `max_ulp: 0`).

**As-built binding model — BYTE-WIDTH-keyed (production truth).** Each OpKind registers as a PRIMITIVE
binding keyed `(OpKind, [in_dtype, out_dtype], Vulkan) + kernel_source` — a 2-slot `[T, T]` key — over
NINE element dtypes `[F32, I32, U32, F16, BF16, F64, I64, U8, I8]`. But the underlying Slang kernels
are **byte-WIDTH-dispatched** (`write_slice_b1/b2/b4/b8`), so the 9 dtype keys collapse to FOUR
wrappers by element size — the cast family's "several sections share one wrapper" precedent:

- `b4` (4-byte): F32 / I32 / U32
- `b2` (2-byte): F16 / BF16
- `b8` (8-byte): F64 / I64
- `b1` (1-byte): U8 / I8

Each section fans the BASE `entry_point` over the 9-dtype list; the link registry maps every
`<base>_<suffix>` symbol to the byte-width wrapper for that dtype's size. Distinct dtype keys ⇒ legal
sibling registrations of the shared byte-width `KernelRef`, byte-for-byte the deleted hand-written
`register_with_precision(OpKind::{WriteSlice,WriteSliceRotating}, &[T, T], …)` regs.

**Layout model — contiguous-only at the binding boundary (matches the as-built reg).** The
destination slab's geometry IS the layout — `write_slice_b*` reads `src` contiguously in its own
rank-N shape and writes the matching slab inside `dst` (also contiguous in its larger rank-N shape).
Strided inputs wouldn't compose with the slab-walk's own-shape strides, so the production
registrations are plain `register_with_precision` (no strided caps):
`awkward_layout_strategy: requires_contiguous` (`strided_input == false`); the planner
auto-Contiguizes any non-contiguous source first (§4.3). Output is a fresh contiguous buffer (the
destination with the slab overwritten), no aliasing at the binding boundary.

**Cost provenance.** Every cost block is `judge_measured` (§4.4). The bandwidth `bytes_moved` hint is
retained (bandwidth-bound slab write); no overhead constant is fabricated. The imported `unknown_cost`
sentinel is upgraded to the shared OpKind cost fn by `fill_unset_cost_for_backend`.

**Determinism.** Pure byte/word copy — no FP arithmetic, no atomics — so every kernel is
`determinism: bitwise` with an audited byte-exact precision (`max_ulp: 0`), byte-for-byte the deleted
regs' `VULKAN_BYTE_LEVEL_PRECISION`.

---

## write_slice  (write a source into a slab of a larger destination; 9 dtypes; byte-width-keyed)

Write the `src` tensor into a slab (offset sub-region) of a larger destination `dst`, both contiguous
row-major (a pure byte/word data move, no arithmetic). BYTE-WIDTH-keyed: the 9 element dtypes map to 4
wrappers (`write_slice::write_slice_{b1,b2,b4,b8}` → `VulkanBackend::write_slice_bytes` by element
size). This section fans the BASE `entry_point` over `[F32, I32, U32, F16, BF16, F64, I64, U8, I8]`;
the link registry maps each `<suffix>` to its byte-width wrapper. Contiguous-only binding. Dispatch
key `(WriteSlice, [T, T], Vulkan)`.

```fkc
kernel: write_slice
op_kind: WriteSlice
blurb: "Write a source tensor into a slab of a larger destination; byte-width-keyed (b1/b2/b4/b8) word move; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::write_slice"   # BASE symbol; fans write_slice_<suffix> → byte-width wrapper
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, I32, U32, F16, BF16, F64, I64, U8, I8]   # 9 dtypes → 4 byte-width wrappers (b1/b2/b4/b8)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "src fits inside the destination slab at the write offset"
  op_params:
    variant: WriteSlice           # OpParams::WriteSlice (primitive namespace; §3.7)
    fields:
      dst_shape: { kind: "Vec<usize>", note: "the larger destination extents" }
      offsets:   { kind: "Vec<usize>", note: "per-axis slab start offset in dst" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(src)         # byte-width copy preserves dtype; key [T, T]
      shape_rule: from_params(dst_shape)
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
  bytes_moved: "2 * n * dtype_bytes"   # n = src element count; read src + write slab

precision:
  bit_stable_on_same_hardware: true   # pure byte/word copy — exact, no arithmetic
  max_ulp: 0
  max_relative: 0.0
  max_absolute: 0.0
  audited: true
  notes: "exact byte-width slab write; no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

---

## write_slice_rotating  (sliding-window KV slab write; 9 dtypes; byte-width-keyed)

The sliding-window (rotating KV-cache) variant of `write_slice`: the wrapper handles the position D2H
readback + ring-boundary split, then delegates to the byte-width slab-write path (a pure byte/word
data move). BYTE-WIDTH-keyed: the 9 element dtypes map to 4 wrappers
(`write_slice_rotating::write_slice_rotating_{b1,b2,b4,b8}`). This section fans the BASE `entry_point`
over `[F32, I32, U32, F16, BF16, F64, I64, U8, I8]`. Contiguous-only binding. Dispatch key
`(WriteSliceRotating, [T, T], Vulkan)`.

```fkc
kernel: write_slice_rotating
op_kind: WriteSliceRotating
blurb: "Sliding-window KV slab write (position D2H + ring-boundary split, then slab write); byte-width-keyed; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::write_slice_rotating"   # BASE symbol; fans write_slice_rotating_<suffix> → byte-width wrapper
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, I32, U32, F16, BF16, F64, I64, U8, I8]   # 9 dtypes → 4 byte-width wrappers (b1/b2/b4/b8)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "src fits inside the rotating destination window at the (ring-wrapped) write offset"
  op_params:
    variant: WriteSliceRotating   # OpParams::WriteSliceRotating (primitive namespace; §3.7)
    fields:
      dst_shape: { kind: "Vec<usize>", note: "the ring-buffer destination extents" }
      capacity:  { kind: usize, note: "ring window capacity along the rotating axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(src)
      shape_rule: from_params(dst_shape)
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
  bytes_moved: "2 * n * dtype_bytes"

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: 0.0
  max_absolute: 0.0
  audited: true
  notes: "exact byte-width slab write (rotating KV window); no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

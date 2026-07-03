---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — triangular / flip / roll (shape) family kernel contracts

The Vulkan backend's byte-level **shape** primitives (crate `vulkan`, family `shape`):
`OpKind::Triu` / `OpKind::Tril` (upper / lower triangular mask) and `OpKind::Flip` / `OpKind::Roll`
(axis reversal / cyclic shift). Every kernel here is a **pure byte / word data mover** — it copies (or
zeros, for the triangular mask's off-triangle cells) `dtype_size` bytes per element with no arithmetic,
so the result is **bit-identical on any hardware** (`determinism: bitwise`, `max_ulp: 0`).

**As-built binding model — production truth.** Each OpKind registers as a PRIMITIVE binding keyed
`(OpKind, [in_dtype, out_dtype], Vulkan) + kernel_source` — a 2-slot `[T, T]` key — over the SEVEN
element dtypes `[F32, F16, BF16, F64, I32, U32, I64]`. Each op dispatches through ONE dtype-agnostic
wrapper (`triangular::triu` / `triangular::tril` / `flip::flip` / `roll::roll`) that picks its element
byte-width from the dtype at the shim, so every fanned dtype key resolves to the SAME `KernelRef` (a
**synthetic-base umbrella**, the CPU `pad_cpu` precedent). Each section fans the BASE `entry_point`
over the 7-dtype list; the link registry maps every `<base>_<suffix>` symbol to the one wrapper.
Distinct dtype keys ⇒ legal sibling registrations of one `KernelRef`, byte-for-byte the deleted
hand-written `register_with_precision(OpKind::{Triu,Tril}, …)` /
`register_with_caps_and_precision(OpKind::{Flip,Roll}, …, strided, …)` regs.

**Layout model — split by op (matches the as-built regs).**

- **Triu / Tril** view the input as a flat `(outer, dim_size, inner)` 3-tuple over a canonical
  row-major buffer — CONTIGUOUS-ONLY (`awkward_layout_strategy: requires_contiguous`,
  `strided_input == false`); a strided operand is auto-Contiguized first.
- **Flip / Roll** are STRIDE-AWARE (`flip_b*` / `roll_b*` walk rank-N + per-input strides with the
  axis from `OpParams::{Flip,Roll}`), so they carry `KernelCaps::strided_input()`
  (`strided: accepted` ⇒ `strided_input == true`) — the caps-through-import proof for this family.

Output is always freshly-allocated **contiguous** row-major over the input's shape, no aliasing, not
in-place.

**Cost provenance.** Every cost block is `judge_measured` (§4.4). The bandwidth `bytes_moved` hint is
retained (all are bandwidth-bound single-pass movers); no overhead constant is fabricated. The
imported `unknown_cost` sentinel is upgraded to the shared OpKind cost fn by
`fill_unset_cost_for_backend`.

**Determinism.** Pure byte/word copy — no FP arithmetic, no atomics — so every kernel is
`determinism: bitwise` with an audited byte-exact precision (`max_ulp: 0`), byte-for-byte the deleted
regs' `VULKAN_BYTE_LEVEL_PRECISION`.

---

## triu  (upper-triangular mask; 7 dtypes; contiguous)

Keep the elements on/above the k-th diagonal, zero the rest (a per-element byte select: copy the input
value or write zero, no arithmetic). Views the input as a flat `(outer, dim_size, inner)` 3-tuple.
ONE dtype-agnostic wrapper (`triangular::triu` → `VulkanBackend::triu_bytes`) picks its byte-width from
the dtype; this section fans the BASE `entry_point` over the 7 element dtypes. Contiguous-only binding.
Dispatch key `(Triu, [T, T], Vulkan)`.

```fkc
kernel: triu
op_kind: Triu
blurb: "Upper-triangular mask (keep on/above the k-th diagonal, zero the rest); dtype-agnostic byte select; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::triu"   # BASE symbol; fans triu_<suffix>, all → triangular::triu
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64, I32, U32, I64]   # fans the dtype key; the ONE wrapper is dtype-agnostic (byte-width)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "flat (outer, dim_size, inner); last two dims are the matrix"
  op_params:
    variant: Triangular           # OpParams::Triangular (primitive namespace; §3.7); upper=true selects Triu
    fields:
      diagonal: { kind: i64, note: "k-th diagonal offset (0 = main)" }
      upper:    { kind: bool, note: "true ⇒ upper-triangular (Triu)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)       # byte select preserves dtype; key [T, T]
      shape_rule: same_as(input)
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
  bytes_moved: "2 * n * dtype_bytes"   # read input + write out

precision:
  bit_stable_on_same_hardware: true   # pure byte select — exact, no arithmetic
  max_ulp: 0
  max_relative: 0.0
  max_absolute: 0.0
  audited: true
  notes: "exact per-element byte select (input value or zero, by triangle position); no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

---

## tril  (lower-triangular mask; 7 dtypes; contiguous)

Sibling of `triu`: keep the elements on/below the k-th diagonal, zero the rest (per-element byte
select). ONE dtype-agnostic wrapper (`triangular::tril` → `VulkanBackend::tril_bytes`); this section
fans the BASE `entry_point` over the 7 element dtypes. Contiguous-only binding. Dispatch key
`(Tril, [T, T], Vulkan)`.

```fkc
kernel: tril
op_kind: Tril
blurb: "Lower-triangular mask (keep on/below the k-th diagonal, zero the rest); dtype-agnostic byte select; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::tril"   # BASE symbol; fans tril_<suffix>, all → triangular::tril
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64, I32, U32, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "flat (outer, dim_size, inner); last two dims are the matrix"
  op_params:
    variant: Triangular           # OpParams::Triangular (primitive namespace; §3.7); upper=false selects Tril
    fields:
      diagonal: { kind: i64, note: "k-th diagonal offset (0 = main)" }
      upper:    { kind: bool, note: "false ⇒ lower-triangular (Tril)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
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
  notes: "exact per-element byte select (input value or zero, by triangle position); no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

---

## flip  (axis reversal; 7 dtypes; stride-aware)

Reverse the input along the axis from `OpParams::Flip` (a pure permutation byte move). STRIDE-AWARE:
`flip_b*` walks rank-N + per-input strides. ONE dtype-agnostic wrapper (`flip::flip` →
`VulkanBackend::flip_bytes`) picks its byte-width from the dtype; this section fans the BASE
`entry_point` over the 7 element dtypes. Dispatch key `(Flip, [T, T], Vulkan)`.

```fkc
kernel: flip
op_kind: Flip
blurb: "Axis reversal (permutation byte move); dtype-agnostic byte-width; stride-aware."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flip"   # BASE symbol; fans flip_<suffix>, all → flip::flip
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64, I32, U32, I64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: accepted }
      rank: any
      shape_constraint: "reverse along OpParams::Flip.axis"
  op_params:
    variant: Flip                 # OpParams::Flip (primitive namespace; §3.7)
    fields:
      axis: { kind: usize, note: "the reversed axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: strided         # stride-aware: lazy views reach the kernel unmaterialized
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
  notes: "exact permutation byte move (axis reversal); no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

---

## roll  (cyclic shift; 7 dtypes; stride-aware)

Cyclically shift the input along the axis from `OpParams::Roll` by `shift` positions (a pure
permutation byte move). STRIDE-AWARE: `roll_b*` walks rank-N + per-input strides. ONE dtype-agnostic
wrapper (`roll::roll` → `VulkanBackend::roll_bytes`); this section fans the BASE `entry_point` over the
7 element dtypes. Dispatch key `(Roll, [T, T], Vulkan)`.

```fkc
kernel: roll
op_kind: Roll
blurb: "Cyclic shift along an axis (permutation byte move); dtype-agnostic byte-width; stride-aware."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::roll"   # BASE symbol; fans roll_<suffix>, all → roll::roll
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64, I32, U32, I64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: accepted }
      rank: any
      shape_constraint: "cyclic shift along OpParams::Roll.axis by OpParams::Roll.shift"
  op_params:
    variant: Roll                 # OpParams::Roll (primitive namespace; §3.7)
    fields:
      axis:  { kind: usize, note: "the shifted axis" }
      shift: { kind: usize, note: "cyclic shift amount" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: strided
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
  notes: "exact permutation byte move (cyclic shift); no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

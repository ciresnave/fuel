---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — pad / copy family kernel contracts

The Vulkan backend's byte-level **pad + device→host copy** primitives (crate `vulkan`, family
`pad`): `OpKind::Pad` (constant-fill pad), `OpKind::PadBackward` (its adjoint — slice the valid
region back out, const mode only), and `OpKind::Copy` (the D2H bridge download). Every kernel here is
a **pure byte / word data mover** — it copies (or constant-fills, for Pad's border) `dtype_size` bytes
per element with no arithmetic, so the result is **bit-identical on any hardware**
(`determinism: bitwise`, `max_ulp: 0`).

**As-built binding model — production truth.** Each OpKind registers as a PRIMITIVE binding keyed
`(OpKind, [in_dtype, out_dtype], Vulkan) + kernel_source` — a 2-slot `[T, T]` key. Each op dispatches
through ONE dtype-agnostic wrapper that picks its element byte-width from the dtype at the shim:
`pad::pad_const` (Pad, byte-width from the OUTPUT dtype), `pad::pad_backward` (PadBackward),
`copy_to_cpu_vulkan` (Copy, a Vulkan staging-buffer D2H download that ignores `OpParams`). Every
fanned dtype key resolves to the SAME `KernelRef` (a **synthetic-base umbrella**, the CPU `pad_cpu`
precedent). Each section fans the BASE `entry_point` over its dtype list; the link registry maps every
`<base>_<suffix>` symbol to the one wrapper. Distinct dtype keys ⇒ legal sibling registrations of one
`KernelRef`, byte-for-byte the deleted hand-written
`register_with_precision(OpKind::{Pad,PadBackward,Copy}, …)` regs. (Pad / PadBackward fan the 6
dtypes `[F32, F16, BF16, F64, U8, U32]`; Copy fans 9 `[F32, F16, BF16, F64, U32, U8, I16, I32, I64]`.)

**Layout model — contiguous-only at the binding boundary (matches the as-built reg).** The pad
byte-width kernels write the output slab from the contiguous input; `copy_to_cpu_vulkan` sizes its
staging buffer to the source's flat byte count — none walks a `Layout`/strides/offset, so the
production registrations are plain `register_with_precision` (no strided caps):
`awkward_layout_strategy: requires_contiguous` (`strided_input == false`), and the planner
auto-Contiguizes a strided / offset operand *first* (§4.3). Output is always freshly-allocated
**contiguous** row-major, no aliasing, not in-place. (Reflect / replicate pad modes fall through to
CPU — the Vulkan wrappers implement const mode only.)

**Cost provenance.** Every cost block is `judge_measured` (§4.4). The bandwidth `bytes_moved` hint is
retained (all are bandwidth-bound single-pass movers); no overhead constant is fabricated. The
imported `unknown_cost` sentinel is upgraded to the shared OpKind cost fn by
`fill_unset_cost_for_backend`.

**Determinism.** Pure byte/word copy — no FP arithmetic, no atomics — so every kernel is
`determinism: bitwise` with an audited byte-exact precision (`max_ulp: 0`), byte-for-byte the deleted
regs' `VULKAN_BYTE_LEVEL_PRECISION`.

---

## pad  (constant-fill pad; 6 dtypes; contiguous)

Pad the input on each side with a constant fill (`OpParams::Pad` carries the per-axis before/after
widths + the fill bytes). ONE dtype-agnostic wrapper (`pad::pad_const` →
`VulkanBackend::pad_bytes` byte-width family b1/b2/b4/b8, keyed by the output dtype's size); this
section fans the BASE `entry_point` over `[F32, F16, BF16, F64, U8, U32]`. Const mode only
(reflect / replicate fall through to CPU). Contiguous-only binding. Dispatch key `(Pad, [T, T], Vulkan)`.

```fkc
kernel: pad
op_kind: Pad
blurb: "Constant-fill pad; dtype-agnostic byte-width copy + border fill; const mode only; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad"   # BASE symbol; fans pad_<suffix>, all → pad::pad_const
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64, U8, U32]   # fans the dtype key; the ONE wrapper is dtype-agnostic (byte-width from out)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "output extends input by the per-axis before/after pad widths"
  op_params:
    variant: Pad                  # OpParams::Pad (primitive namespace; §3.7)
    fields:
      pads:       { kind: "Vec<(usize, usize)>", note: "per-axis (before, after) widths" }
      fill_bytes: { kind: "Vec<u8>", note: "the constant fill value, dtype-width bytes" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)       # byte-width copy preserves dtype; key [T, T]
      shape_rule: pad(input, pads)
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
  bytes_moved: "2 * n * dtype_bytes"   # n = output element count; read input + write out (incl. fill)

precision:
  bit_stable_on_same_hardware: true   # pure byte copy + constant fill — exact, no arithmetic
  max_ulp: 0
  max_relative: 0.0
  max_absolute: 0.0
  audited: true
  notes: "exact byte-width copy of the input + constant border fill; no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

---

## pad_backward  (adjoint of constant pad — slice the valid region back out; 6 dtypes; contiguous)

The adjoint of the constant pad: slice the un-padded (valid) region back out of the gradient (a pure
byte copy, const mode only). ONE dtype-agnostic wrapper (`pad::pad_backward` →
`VulkanBackend::pad_backward_bytes`); this section fans the BASE `entry_point` over
`[F32, F16, BF16, F64, U8, U32]`. Contiguous-only binding. Dispatch key `(PadBackward, [T, T], Vulkan)`.

```fkc
kernel: pad_backward
op_kind: PadBackward
blurb: "Adjoint of constant pad — slice the valid region out; dtype-agnostic byte-width copy; const mode only; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_backward"   # BASE symbol; fans pad_backward_<suffix>, all → pad::pad_backward
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64, U8, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "output is the un-padded (valid) sub-region of the input gradient"
  op_params:
    variant: PadBackward          # OpParams::PadBackward (primitive namespace; §3.7)
    fields:
      pads: { kind: "Vec<(usize, usize)>", note: "per-axis (before, after) widths to strip" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: pad_backward(input, pads)
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
  bytes_moved: "2 * n * dtype_bytes"   # n = output element count; read valid region + write out

precision:
  bit_stable_on_same_hardware: true   # pure byte copy — exact, no arithmetic
  max_ulp: 0
  max_relative: 0.0
  max_absolute: 0.0
  audited: true
  notes: "exact byte-width copy of the valid sub-region (const mode); no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

---

## copy  (D2H bridge download; 9 dtypes; contiguous)

The device→host `Op::Copy` bridge: download the source's bytes via a Vulkan staging buffer
(`vkCmdCopyBuffer`). ONE dtype-agnostic wrapper (`copy_to_cpu_vulkan` — ignores `OpParams`, sizes the
staging buffer to the source's flat byte count); this section fans the BASE `entry_point` over
`[F32, F16, BF16, F64, U32, U8, I16, I32, I64]`. Contiguous-only binding (stride-aware D2H would need
per-row staging). Dispatch key `(Copy, [T, T], Vulkan)`.

```fkc
kernel: copy
op_kind: Copy
blurb: "Device->host bridge download via a Vulkan staging buffer; dtype-agnostic byte copy; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::copy"   # BASE symbol; fans copy_<suffix>, all → copy_to_cpu_vulkan
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64, U32, U8, I16, I32, I64]   # fans the dtype key; the ONE wrapper is dtype-agnostic
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "download the flat byte extent of the source"
  op_params:
    variant: None                 # copy_to_cpu_vulkan ignores OpParams (staging-buffer download)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)       # byte copy preserves dtype; key [T, T]
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
  bytes_moved: "2 * n * dtype_bytes"   # read device source + write host output

precision:
  bit_stable_on_same_hardware: true   # pure byte download — exact, no arithmetic
  max_ulp: 0
  max_relative: 0.0
  max_absolute: 0.0
  audited: true
  notes: "exact byte-for-byte device->host download via staging buffer; no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

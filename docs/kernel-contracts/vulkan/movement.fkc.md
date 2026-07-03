---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — concat / cumsum (stride-aware movement) family kernel contracts

The Vulkan backend's **stride-aware data-movement** primitives (crate `vulkan`, family `movement`):
`OpKind::Concat` (join two tensors along a dim) and `OpKind::CumSum` (inclusive prefix sum along one
axis). Both are **STRIDE-AWARE** — the underlying Slang kernels walk per-input rank-N strides (Concat
supports per-operand permute/broadcast views; CumSum walks the rank-N layout with the axis from
`OpParams::CumSum`), so the production registrations carry `KernelCaps::strided_input()` (the deleted
hand-written `register_with_caps_and_precision(…, strided, …)` regs). The planner therefore passes
lazy views through UNMATERIALIZED (no auto-Contiguize) — the caps-through-import proof: each section's
`strided: accepted, broadcast_stride0: accepted` layout projects `strided_input == true` (§4.1,
`caps_map`).

**As-built binding model — production truth.** Each OpKind registers as a PRIMITIVE binding keyed
`(OpKind, [in_dtype, out_dtype], Vulkan) + kernel_source` — a 2-slot `[T, T]` key. Each op has FOUR
distinct per-dtype wrappers (`concat::concat_{f32,f16,bf16,f64}`,
`cumsum::cumsum_{f32,f64,f16,bf16}` → `VulkanBackend::concat_*` / `cumsum_*_bytes`). The sections
below fan the BASE `entry_point` over the dtype list, resolving `<base>_<suffix>` to each per-dtype
wrapper and keying `[T, T]` (`passthrough(input)` output), byte-for-byte the deleted regs.

**Output is contiguous, freshly-allocated.** Every wrapper writes a fresh contiguous row-major output
over the input's (concatenated / prefix-summed) shape — no aliasing, not in-place.

**Cost provenance.** Every cost block is `judge_measured` (§4.4). The bandwidth `bytes_moved` hint is
retained (both are bandwidth-bound single-pass movers); no overhead constant is fabricated. The
imported `unknown_cost` sentinel is upgraded to the shared OpKind cost fn by
`fill_unset_cost_for_backend`.

**Determinism.** Concat is a pure byte/word copy — byte-exact (`determinism: bitwise`, `max_ulp: 0`),
byte-for-byte the deleted regs' `VULKAN_BYTE_LEVEL_PRECISION`. CumSum is a sequential per-slice FP
prefix sum (deterministic, no cross-thread reduction) but carries the conservative author-seed
posture the elementwise migration set for Vulkan pointwise arithmetic (`audited: false` ⇒
`PrecisionGuarantee::UNAUDITED`; the Judge audits the ULP bound later) rather than re-asserting the
retired hand-written `VULKAN_{FLOAT,HALF}_POINTWISE_PRECISION` consts.

---

## concat  (join two tensors along a dim; f32/f16/bf16/f64; stride-aware)

Two inputs joined along the concat axis into one contiguous output (N==2; N>2 falls through to the
next alternative). STRIDE-AWARE: `concat_along_dim.slang` walks per-operand strides, so either input
may be a lazy permute / broadcast view. FOUR distinct per-dtype wrappers
(`concat::concat_{f32,f16,bf16,f64}` → `VulkanBackend::concat_*_bytes`); this section fans the BASE
`entry_point` over `[F32, F16, BF16, F64]`. Pure byte/word data move (no arithmetic). Dispatch key
`(Concat, [T, T], Vulkan)`.

```fkc
kernel: concat
op_kind: Concat
blurb: "Join two tensors along a dim; f32/f16/bf16/f64 stride-aware byte copy; N==2."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::concat"   # BASE symbol; fans concat_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64]     # fans the per-dtype wrapper (§3.4)
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: accepted }
      rank: any
      shape_constraint: "the two operands agree on every axis except the concat dim; N==2"
  op_params:
    variant: Concat               # OpParams::Concat (primitive namespace; §3.7)
    fields:
      dim: { kind: usize, note: "concat axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)       # data move preserves dtype; key [T, T]
      shape_rule: concat(inputs, dim)
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
  bytes_moved: "2 * n * dtype_bytes"   # n = output element count; read both operands + write out

precision:
  bit_stable_on_same_hardware: true   # pure byte/word copy — exact, no arithmetic
  max_ulp: 0
  max_relative: 0.0
  max_absolute: 0.0
  audited: true
  notes: "exact byte/word copy of the two operands into the joined output; no arithmetic, bit-identical across any hardware."

determinism: bitwise
```

---

## cumsum  (inclusive prefix sum along one axis; f32/f64/f16/bf16; stride-aware)

Inclusive prefix (running) sum along one axis. STRIDE-AWARE: the kernel walks the rank-N layout with
the axis from `OpParams::CumSum`; the accumulation is a sequential per-slice walk (deterministic, no
cross-thread reduction). F32/F64 accumulate in their native types; F16 accumulates in f16; BF16
accumulates in f32 with bit-level bf16↔f32 conversion at the edges. FOUR distinct per-dtype wrappers
(`cumsum::cumsum_{f32,f64,f16,bf16}` → `VulkanBackend::cumsum_*_bytes`); this section fans the BASE
`entry_point` over `[F32, F64, F16, BF16]`. Dispatch key `(CumSum, [T, T], Vulkan)`.

```fkc
kernel: cumsum
op_kind: CumSum
blurb: "Inclusive prefix sum along one axis; f32/f64/f16/bf16 stride-aware sequential scan."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cumsum"   # BASE symbol; fans cumsum_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, F16, BF16]     # fans the per-dtype wrapper (§3.4)
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: accepted }
      rank: any
      shape_constraint: "prefix sum along OpParams::CumSum.axis"
  op_params:
    variant: CumSum               # OpParams::CumSum (primitive namespace; §3.7)
    fields:
      axis: { kind: usize, note: "the prefix-sum axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)       # same dtype; key [T, T]
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
  flops: "n"                           # one add per element (single sequential pass); derivable
  bytes_moved: "2 * n * dtype_bytes"   # read input + write out

precision:
  bit_stable_on_same_hardware: false   # author seed (UNAUDITED); sequential FP add, Judge audits the ULP bound later
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "inclusive prefix sum; f32/f64 native, f16 in-type, bf16 accumulate in f32 (bit-level edges). Sequential per-slice scan (deterministic, no cross-thread reduction); ULP bound not yet Judge-audited."

determinism: same_hardware_bitwise
```

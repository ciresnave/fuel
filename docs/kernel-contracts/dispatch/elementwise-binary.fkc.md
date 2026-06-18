---
fkc_version: 1
provider:
  name: fuel-dispatch
  backend: Cpu                       # default; per-kernel blocks override to Cuda / Vulkan
  kernel_source: "dispatch-cpu"      # default BindingEntry.kernel_source tag; overridden per backend
  link_registry: fuel_dispatch::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-dispatch — elementwise-binary kernel contracts

The `fuel-dispatch` binding-table registrations for the elementwise-binary family: the four
arithmetic ops (Add / Sub / Mul / Div), the two extrema (Maximum / Minimum), the two
extra-numeric ops (Pow / Rem), the six comparison predicates (Equal / NotEqual / Less /
LessEqual / Greater / GreaterEqual), and the ternary select (Where). These are the rows of the
`fuel-dispatch` inventory's "Elementwise binary", "Comparison family", and "Where" entries
(`docs/kernel-contracts/_inventory/dispatch.md`), keyed against the as-built
`KernelBindingTable` registrations in `fuel-dispatch/src/{dispatch.rs,baracuda_dispatch.rs,
vulkan_dispatch.rs}`.

`fuel-dispatch` is a **multi-backend provider**: a single op (e.g. `AddElementwise`) is registered
on as many of CPU / CUDA(baracuda) / Vulkan as ship a wrapper, and each backend is a **distinct
dispatch key** `(OpKind, KernelDTypes, BackendId)`. Each named kernel below is therefore **one
`##` section** that carries **one ` ```fkc ` block per backend that registers it** — the section
boundary is the op (the named kernel), the per-backend blocks are its sibling alternatives at the
op's keys. dtype-monomorphized wrappers (f32/f64/bf16/f16) are collapsed into one block's
`dtypes:` list, per the inventory's "one entry per kernel" rule; the entry-point symbol names the
per-`(op, dtype)` wrapper for the representative dtype and the link registry resolves the rest.

**Family-wide facts (every kernel below shares these unless its section/block says otherwise):**

- **CPU backend (`register`, default all-false caps).** Every CPU wrapper takes `_layouts:
  &[Layout]` **unused** and operates on raw contiguous byte buffers (`CpuStorageBytes`); the
  pipelined executor's auto-Contiguize pass realizes any strided / broadcast / non-zero-offset
  input into a dense buffer *before* the wrapper runs. Layout capability is therefore
  `{contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected,
  reverse_strides: rejected}` on every operand, and `awkward_layout_strategy: requires_contiguous`.
  CPU is the always-built universal fallback; precision is bulk-upgraded to
  `PRIMITIVE_DETERMINISTIC_CPU` (bit-stable per hardware) and `f64`/`f32` evaluate natively while
  `bf16`/`f16` accumulate/round-trip through f32.
- **CUDA (baracuda) backend (`register_with_caps(..., KernelCaps::strided_input())`).** The
  baracuda FFI is stride-driven: the wrapper consumes `layouts[..]` and walks arbitrary
  non-negative input strides, including stride-0 broadcast axes — no auto-Contiguize for those.
  Layout capability is `{contiguous: accepted, strided: accepted, broadcast_stride0: accepted,
  start_offset: rejected, reverse_strides: rejected}` (NOT offset-capable: a non-zero
  `start_offset` input STILL routes through auto-Contiguize even on strided-capable kernels —
  `compiled.rs:58`, `KernelCaps` doc). `awkward_layout_strategy: handles_strided`.
- **Vulkan backend (`register_with_caps_and_precision(..., strided, VULKAN_FLOAT_POINTWISE_PRECISION)`).**
  `binary.slang` is stride-aware (per-dim decomposition + a contiguous fast path; the wrapper packs
  each input's strides into the Params struct), so the same strided/broadcast (not offset) layout
  capability as CUDA applies. Precision is `VULKAN_FLOAT_POINTWISE_PRECISION` (pointwise float,
  not bit-stable cross-hardware). Vulkan ships only the four arithmetic ops + the two extrema; it
  has **no** Pow / Rem / comparison / Where wrapper in this crate.
- **Output: pre-allocated, fully overwritten, not in-place.** The executor pre-allocates the
  output Storage; no kernel allocates. Output dtype follows the per-op `dtype_rule` (arithmetic /
  extrema / Pow / Rem → same dtype as inputs; comparisons → U8; Where → the a/b dtype T). Output
  shape is the broadcast of the inputs (realized by auto-Contiguize on CPU, by stride-0 on the
  strided GPU paths). No operand aliases the output.

**Cost provenance.** Every block in this family marks `cost.provenance: judge_measured` — the
Judge bootstraps and refines the measured cost for these surfaces. Where genuinely derivable from
the op, the cost block carries the **formula hints** an elementwise kernel admits: it is
bandwidth-bound, touching `n` output elements while reading the inputs and writing the output, with
one scalar op per element (`flops ≈ n`). For a two-input arithmetic/extremum/Pow/Rem op
`bytes_moved ≈ 3 · n · dtype_bytes` (read lhs + rhs, write out); for a comparison
`bytes_moved ≈ 2 · n · dtype_bytes + n · 1` (read two T inputs, write a U8 mask); for Where
`bytes_moved ≈ n · 1 + 3 · n · dtype_bytes` (read the U8 cond + a + b, write out). These hints are
structural facts of the loop, not fabricated coefficients; launch overhead and absolute timing are
left for the Judge to populate (no placeholder numbers). `dtype_bytes` is the byte width of the
arithmetic dtype T (F32→4, F64→8, BF16→2, F16→2).

**Precision.** CPU `f32`/`f64` are the native IEEE-754 operation, bit-stable on the same hardware;
`bf16`/`f16` are bit-stable on the same hardware via the deterministic f32 round-trip
(widen → op → narrow). Vulkan arithmetic/extrema carry `VULKAN_FLOAT_POINTWISE_PRECISION` (audited,
no static ULP bound, not bit-stable cross-hardware). All bounds are author-declared seeds the Judge
audits.

---

## add_elementwise  (AddElementwise — out = lhs + rhs, broadcast)

Elementwise addition `out = lhs + rhs` with broadcasting; one block per backend (CPU contiguous,
CUDA/Vulkan strided).

`AddElementwise` is registered on CPU (`add_elementwise_f32_cpu_wrapper` and the f64/bf16/f16
siblings, contiguous-only, `dispatch.rs:3913`), CUDA via baracuda (`binary::add_f32` …, strided,
`baracuda_dispatch.rs:2372`), and Vulkan (`binary::add_f32` …, strided,
`vulkan_dispatch.rs:4293`). Output dtype = input dtype T; output shape = broadcast of the two
inputs; not in-place. f32/f64 native; bf16/f16 round-trip through f32. Known limitation: the CPU
path is contiguous-only — any strided / broadcast / non-zero-offset operand is contiguized by the
executor first; the GPU paths walk strides directly (broadcast via stride-0) but are still NOT
offset-capable (non-zero `start_offset` auto-contiguizes everywhere).

```fkc
kernel: add_elementwise_cpu
op_kind: AddElementwise
blurb: "Elementwise addition out=lhs+rhs (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::add_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous     # CPU wrapper ignores _layouts; executor auto-Contiguizes
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                            # one add per output element
  bytes_moved: "3 * n * dtype_bytes"    # read lhs + rhs, write out — bandwidth-bound
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32/f64 native IEEE addition; bf16/f16 widen to f32 then narrow on store. Deterministic; bit-stable on same hardware (PRIMITIVE_DETERMINISTIC_CPU)."

determinism: same_hardware_bitwise
```

```fkc
kernel: add_elementwise_cuda
op_kind: AddElementwise
blurb: "Elementwise addition out=lhs+rhs (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_dispatch::baracuda_dispatch::binary::add_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided          # baracuda FFI walks strides incl. stride-0 broadcast
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "baracuda elementwise add; f32/f64 native, bf16/f16 widen to f32. Bit-stable on same hardware (pointwise, no reduction)."

determinism: same_hardware_bitwise
```

```fkc
kernel: add_elementwise_vulkan
op_kind: AddElementwise
blurb: "Elementwise addition out=lhs+rhs (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_dispatch::vulkan_dispatch::binary::add_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided          # binary.slang packs per-input strides into Params
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "VULKAN_FLOAT_POINTWISE_PRECISION — pointwise float op; audited, no static ULP bound; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## sub_elementwise  (SubElementwise — out = lhs - rhs, broadcast)

Elementwise subtraction `out = lhs - rhs` with broadcasting; one block per backend.

`SubElementwise` is registered on CPU (`sub_elementwise_f32_cpu_wrapper` …, `dispatch.rs:3914`),
CUDA via baracuda (`binary::sub_f32` …, strided, `baracuda_dispatch.rs:2373`), and Vulkan
(`binary::sub_f32` …, strided, `vulkan_dispatch.rs:4294`). Output dtype = T; broadcast shape; not
in-place. Identical layout/cost/precision profile to `add_elementwise`; only the scalar op differs.

```fkc
kernel: sub_elementwise_cpu
op_kind: SubElementwise
blurb: "Elementwise subtraction out=lhs-rhs (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::sub_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32/f64 native IEEE subtraction; bf16/f16 widen to f32 then narrow. Bit-stable on same hardware."

determinism: same_hardware_bitwise
```

```fkc
kernel: sub_elementwise_cuda
op_kind: SubElementwise
blurb: "Elementwise subtraction out=lhs-rhs (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_dispatch::baracuda_dispatch::binary::sub_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "baracuda elementwise sub; f32/f64 native, bf16/f16 widen to f32. Bit-stable on same hardware."

determinism: same_hardware_bitwise
```

```fkc
kernel: sub_elementwise_vulkan
op_kind: SubElementwise
blurb: "Elementwise subtraction out=lhs-rhs (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_dispatch::vulkan_dispatch::binary::sub_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "VULKAN_FLOAT_POINTWISE_PRECISION — pointwise float op; audited, no static ULP bound; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## mul_elementwise  (MulElementwise — out = lhs * rhs, broadcast)

Elementwise multiplication `out = lhs * rhs` with broadcasting; one block per backend.

`MulElementwise` is registered on CPU (`mul_elementwise_f32_cpu_wrapper` …, `dispatch.rs:3915`),
CUDA via baracuda (`binary::mul_f32` …, strided, `baracuda_dispatch.rs:2374`), and Vulkan
(`binary::mul_f32` …, strided, `vulkan_dispatch.rs:4295`). Output dtype = T; broadcast shape; not
in-place. On CPU the bf16/f16 paths multiply on widened f32 then narrow (avoids the intermediate
truncation a native half multiply would incur).

```fkc
kernel: mul_elementwise_cpu
op_kind: MulElementwise
blurb: "Elementwise multiplication out=lhs*rhs (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::mul_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32/f64 native IEEE multiplication; bf16/f16 widen to f32 then narrow (avoids intermediate truncation). Bit-stable on same hardware."

determinism: same_hardware_bitwise
```

```fkc
kernel: mul_elementwise_cuda
op_kind: MulElementwise
blurb: "Elementwise multiplication out=lhs*rhs (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_dispatch::baracuda_dispatch::binary::mul_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "baracuda elementwise mul; f32/f64 native, bf16/f16 widen to f32. Bit-stable on same hardware."

determinism: same_hardware_bitwise
```

```fkc
kernel: mul_elementwise_vulkan
op_kind: MulElementwise
blurb: "Elementwise multiplication out=lhs*rhs (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_dispatch::vulkan_dispatch::binary::mul_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "VULKAN_FLOAT_POINTWISE_PRECISION — pointwise float op; audited, no static ULP bound; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## div_elementwise  (DivElementwise — out = lhs / rhs, broadcast, IEEE inf/NaN)

Elementwise division `out = lhs / rhs` with broadcasting and IEEE inf/NaN semantics; one block per
backend.

`DivElementwise` is registered on CPU (`div_elementwise_f32_cpu_wrapper` …, `dispatch.rs:3916`),
CUDA via baracuda (`binary::div_f32` …, strided, `baracuda_dispatch.rs:2375`), and Vulkan
(`binary::div_f32` …, strided, `vulkan_dispatch.rs:4296`). Division by zero yields ±inf; `0/0` and
`inf/inf` yield NaN. Output dtype = T; broadcast shape; not in-place.

```fkc
kernel: div_elementwise_cpu
op_kind: DivElementwise
blurb: "Elementwise division out=lhs/rhs (broadcast, IEEE inf/NaN); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::div_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32/f64 native IEEE division; bf16/f16 widen to f32 then narrow. Div-by-zero -> +-inf; 0/0, inf/inf -> NaN. Bit-stable on same hardware."

determinism: same_hardware_bitwise
```

```fkc
kernel: div_elementwise_cuda
op_kind: DivElementwise
blurb: "Elementwise division out=lhs/rhs (broadcast, IEEE inf/NaN); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_dispatch::baracuda_dispatch::binary::div_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "baracuda elementwise div; f32/f64 native, bf16/f16 widen to f32. Div-by-zero -> +-inf; 0/0, inf/inf -> NaN. Bit-stable on same hardware."

determinism: same_hardware_bitwise
```

```fkc
kernel: div_elementwise_vulkan
op_kind: DivElementwise
blurb: "Elementwise division out=lhs/rhs (broadcast, IEEE inf/NaN); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_dispatch::vulkan_dispatch::binary::div_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "VULKAN_FLOAT_POINTWISE_PRECISION — pointwise float op; div-by-zero -> +-inf, 0/0/inf-over-inf -> NaN; audited, no static ULP bound; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## maximum_elementwise  (MaximumElementwise — out = max(lhs, rhs), NaN-as-missing, broadcast)

Elementwise maximum `out = max(lhs, rhs)` with NaN-as-missing semantics and broadcasting; one
block per backend.

`MaximumElementwise` is registered on CPU (`maximum_elementwise_f32_cpu_wrapper` and f64/bf16/f16
siblings, `dispatch.rs:4264/4481/4503`), CUDA via baracuda (`binary::maximum_f32` …, strided,
`baracuda_dispatch.rs:2376`), and Vulkan (`binary::maximum_f32` …, strided,
`vulkan_dispatch.rs:4297`). NaN-as-missing (`f32::max`/`f64::max`): returns the non-NaN operand
when exactly one is NaN, NaN when both are. Output dtype = T; broadcast shape; not in-place.

```fkc
kernel: maximum_elementwise_cpu
op_kind: MaximumElementwise
blurb: "Elementwise maximum out=max(lhs,rhs) NaN-as-missing (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::maximum_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact: returns one of the two operands. f32/f64::max NaN-as-missing (non-NaN wins; NaN only if both NaN). bf16/f16 compute max on widened f32 then narrow. Bit-stable on same hardware."

determinism: same_hardware_bitwise
```

```fkc
kernel: maximum_elementwise_cuda
op_kind: MaximumElementwise
blurb: "Elementwise maximum out=max(lhs,rhs) NaN-as-missing (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_dispatch::baracuda_dispatch::binary::maximum_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "baracuda elementwise maximum; returns one of the two operands (NaN-as-missing). Bit-stable on same hardware."

determinism: same_hardware_bitwise
```

```fkc
kernel: maximum_elementwise_vulkan
op_kind: MaximumElementwise
blurb: "Elementwise maximum out=max(lhs,rhs) NaN-as-missing (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_dispatch::vulkan_dispatch::binary::maximum_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "VULKAN_FLOAT_POINTWISE_PRECISION — returns one of the two operands (NaN-as-missing); audited, no static ULP bound; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## minimum_elementwise  (MinimumElementwise — out = min(lhs, rhs), NaN-as-missing, broadcast)

Elementwise minimum `out = min(lhs, rhs)` with NaN-as-missing semantics and broadcasting; one
block per backend.

`MinimumElementwise` is registered on CPU (`minimum_elementwise_f32_cpu_wrapper` and f64/bf16/f16
siblings, `dispatch.rs:4265/4481/4503`), CUDA via baracuda (`binary::minimum_f32` …, strided,
`baracuda_dispatch.rs:2377`), and Vulkan (`binary::minimum_f32` …, strided,
`vulkan_dispatch.rs:4298`). NaN-as-missing, mirroring maximum. Output dtype = T; broadcast shape;
not in-place.

```fkc
kernel: minimum_elementwise_cpu
op_kind: MinimumElementwise
blurb: "Elementwise minimum out=min(lhs,rhs) NaN-as-missing (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::minimum_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact: returns one of the two operands. f32/f64::min NaN-as-missing (non-NaN wins; NaN only if both NaN). bf16/f16 compute min on widened f32 then narrow. Bit-stable on same hardware."

determinism: same_hardware_bitwise
```

```fkc
kernel: minimum_elementwise_cuda
op_kind: MinimumElementwise
blurb: "Elementwise minimum out=min(lhs,rhs) NaN-as-missing (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_dispatch::baracuda_dispatch::binary::minimum_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "baracuda elementwise minimum; returns one of the two operands (NaN-as-missing). Bit-stable on same hardware."

determinism: same_hardware_bitwise
```

```fkc
kernel: minimum_elementwise_vulkan
op_kind: MinimumElementwise
blurb: "Elementwise minimum out=min(lhs,rhs) NaN-as-missing (broadcast); one block per backend (CPU contiguous, CUDA/Vulkan strided)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_dispatch::vulkan_dispatch::binary::minimum_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "VULKAN_FLOAT_POINTWISE_PRECISION — returns one of the two operands (NaN-as-missing); audited, no static ULP bound; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## pow_elementwise  (PowElementwise — out = lhs ^ rhs, tensor^tensor, broadcast)

Elementwise power `out = lhs ^ rhs` (tensor base, tensor exponent) with broadcasting; CPU + CUDA
only (no Vulkan wrapper in this crate).

`PowElementwise` is registered on CPU (`pow_elementwise_f32_cpu_wrapper` and f64/bf16/f16 siblings,
`dispatch.rs:4340`) and CUDA via baracuda (`binary::pow_f32` …, strided,
`baracuda_dispatch.rs:2381`). This is the tensor^tensor power (distinct from the scalar-exponent
`PowIElementwise`); the CPU path uses `powf`. There is **no** Vulkan registration for this op.
Output dtype = T; broadcast shape; not in-place. Numerics follow the platform `powf` (IEEE
domain/edge cases — negative base with non-integer exponent → NaN, `pow(x,0) == 1`, etc.).

```fkc
kernel: pow_elementwise_cpu
op_kind: PowElementwise
blurb: "Elementwise tensor^tensor power out=lhs^rhs (broadcast, powf); CPU + CUDA only."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::pow_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise        # bandwidth-bound elementwise, but powf is heavier per element than +-*/
  flops: "n"                      # one powf per output element (transcendental; the Judge measures the real cost)
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32/f64 use platform powf; bf16/f16 widen to f32 powf then narrow. powf domain: negative base with non-integer exponent -> NaN; pow(x,0)=1. Deterministic; bit-stable on same hardware (PRIMITIVE_DETERMINISTIC_CPU); transcendental, not exact-rounded."

determinism: same_hardware_bitwise
```

```fkc
kernel: pow_elementwise_cuda
op_kind: PowElementwise
blurb: "Elementwise tensor^tensor power out=lhs^rhs (broadcast, powf); CPU + CUDA only."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_dispatch::baracuda_dispatch::binary::pow_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "baracuda elementwise pow (tensor^tensor, transcendental). powf domain semantics. Bit-stable on same hardware; transcendental, not exact-rounded."

determinism: same_hardware_bitwise
```

---

## rem_elementwise  (RemElementwise — out = lhs mod rhs, PyTorch/Python sign-of-divisor, broadcast)

Elementwise remainder `out = lhs mod rhs` with the PyTorch/Python convention (result takes the sign
of the divisor); CPU + CUDA only (no Vulkan wrapper in this crate).

`RemElementwise` is registered on CPU (`rem_elementwise_f32_cpu_wrapper` and f64/bf16/f16 siblings,
`dispatch.rs:4350`) and CUDA via baracuda (`binary::rem_f32` …, strided,
`baracuda_dispatch.rs:2382`). The CUDA path binds baracuda's `binary_mod_*` (Python-style floored
remainder, sign of the divisor), **not** `binary_remainder_*`/C99 `fmod` (sign of the dividend) —
this is the load-bearing convention note from the inventory. There is **no** Vulkan registration.
Output dtype = T; broadcast shape; not in-place.

```fkc
kernel: rem_elementwise_cpu
op_kind: RemElementwise
blurb: "Elementwise remainder out=lhs mod rhs (PyTorch/Python sign-of-divisor, broadcast); CPU + CUDA only."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::rem_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                      # one modulo per output element
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "PyTorch/Python-convention remainder: result takes the sign of the divisor (floored mod), NOT C99 fmod. f32/f64 native; bf16/f16 widen to f32 then narrow. Deterministic; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

```fkc
kernel: rem_elementwise_cuda
op_kind: RemElementwise
blurb: "Elementwise remainder out=lhs mod rhs (PyTorch/Python sign-of-divisor, broadcast); CPU + CUDA only."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_dispatch::baracuda_dispatch::binary::rem_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis, no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Bound to baracuda binary_mod_* (Python-style floored remainder, sign of divisor), NOT binary_remainder_*/C99 fmod. f32/f64 native; bf16/f16 widen to f32. Bit-stable on same hardware."

determinism: same_hardware_bitwise
```

---

## equal_elementwise  (EqualElementwise — out = (lhs == rhs) ? 1 : 0, U8 mask, broadcast)

Elementwise equality predicate producing a U8 mask; CPU only.

`EqualElementwise` is registered on CPU only (`eq_elementwise_f32_cpu_wrapper` and f64/bf16/f16
siblings, `dispatch.rs:4271`); the binding-table dtype key is `[T, T, U8]` (two T inputs, a U8
output). Output dtype is **always U8** (1 byte/element) regardless of input T: `1` where
`lhs == rhs`, `0` otherwise. Output shape = broadcast of the inputs; not in-place. No CUDA or Vulkan
registration for the comparison family in this crate. Comparison uses the IEEE `==` (NaN compares
unequal to everything, including itself).

```fkc
kernel: equal_elementwise_cpu
op_kind: EqualElementwise
blurb: "Elementwise equality out=(lhs==rhs)?1:0 producing a U8 mask (broadcast); CPU only."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::eq_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U8)              # mask: 1 where predicate holds, 0 otherwise
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                            # one compare per output element
  bytes_moved: "2 * n * dtype_bytes + n"  # read two T inputs, write n bytes of U8 mask
  memory: { device_bytes: 0, host_bytes: "n", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact boolean predicate; output is an exact 0/1 U8 mask. IEEE == (NaN != NaN). bf16/f16 compared via their f32 value. Deterministic; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

---

## not_equal_elementwise  (NotEqualElementwise — out = (lhs != rhs) ? 1 : 0, U8 mask, broadcast)

Elementwise inequality predicate producing a U8 mask; CPU only.

`NotEqualElementwise` is registered on CPU only (`ne_elementwise_f32_cpu_wrapper` and f64/bf16/f16
siblings, `dispatch.rs:4276`); key `[T, T, U8]`. Output dtype always U8: `1` where `lhs != rhs`,
`0` otherwise. Broadcast shape; not in-place. IEEE `!=` (NaN compares unequal to everything, so
`NaN != NaN` is true). No CUDA / Vulkan registration.

```fkc
kernel: not_equal_elementwise_cpu
op_kind: NotEqualElementwise
blurb: "Elementwise inequality out=(lhs!=rhs)?1:0 producing a U8 mask (broadcast); CPU only."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::ne_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U8)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes + n"
  memory: { device_bytes: 0, host_bytes: "n", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact boolean predicate; 0/1 U8 mask. IEEE != (NaN != NaN is true). bf16/f16 compared via their f32 value. Deterministic; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

---

## less_elementwise  (LessElementwise — out = (lhs < rhs) ? 1 : 0, U8 mask, broadcast)

Elementwise less-than predicate producing a U8 mask; CPU only.

`LessElementwise` is registered on CPU only (`lt_elementwise_f32_cpu_wrapper` and f64/bf16/f16
siblings, `dispatch.rs:4281`); key `[T, T, U8]`. Output dtype always U8: `1` where `lhs < rhs`, `0`
otherwise. Broadcast shape; not in-place. IEEE ordering (any comparison involving NaN is false). No
CUDA / Vulkan registration.

```fkc
kernel: less_elementwise_cpu
op_kind: LessElementwise
blurb: "Elementwise less-than out=(lhs<rhs)?1:0 producing a U8 mask (broadcast); CPU only."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::lt_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U8)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes + n"
  memory: { device_bytes: 0, host_bytes: "n", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact boolean predicate; 0/1 U8 mask. IEEE < (any NaN comparison is false). bf16/f16 compared via their f32 value. Deterministic; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

---

## less_equal_elementwise  (LessEqualElementwise — out = (lhs <= rhs) ? 1 : 0, U8 mask, broadcast)

Elementwise less-than-or-equal predicate producing a U8 mask; CPU only.

`LessEqualElementwise` is registered on CPU only (`le_elementwise_f32_cpu_wrapper` and f64/bf16/f16
siblings, `dispatch.rs:4286`); key `[T, T, U8]`. Output dtype always U8: `1` where `lhs <= rhs`,
`0` otherwise. Broadcast shape; not in-place. IEEE ordering (NaN comparisons false). No CUDA /
Vulkan registration.

```fkc
kernel: less_equal_elementwise_cpu
op_kind: LessEqualElementwise
blurb: "Elementwise less-or-equal out=(lhs<=rhs)?1:0 producing a U8 mask (broadcast); CPU only."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::le_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U8)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes + n"
  memory: { device_bytes: 0, host_bytes: "n", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact boolean predicate; 0/1 U8 mask. IEEE <= (any NaN comparison is false). bf16/f16 compared via their f32 value. Deterministic; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

---

## greater_elementwise  (GreaterElementwise — out = (lhs > rhs) ? 1 : 0, U8 mask, broadcast)

Elementwise greater-than predicate producing a U8 mask; CPU only.

`GreaterElementwise` is registered on CPU only (`gt_elementwise_f32_cpu_wrapper` and f64/bf16/f16
siblings, `dispatch.rs:4291`); key `[T, T, U8]`. Output dtype always U8: `1` where `lhs > rhs`, `0`
otherwise. Broadcast shape; not in-place. IEEE ordering (NaN comparisons false). No CUDA / Vulkan
registration.

```fkc
kernel: greater_elementwise_cpu
op_kind: GreaterElementwise
blurb: "Elementwise greater-than out=(lhs>rhs)?1:0 producing a U8 mask (broadcast); CPU only."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::gt_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U8)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes + n"
  memory: { device_bytes: 0, host_bytes: "n", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact boolean predicate; 0/1 U8 mask. IEEE > (any NaN comparison is false). bf16/f16 compared via their f32 value. Deterministic; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

---

## greater_equal_elementwise  (GreaterEqualElementwise — out = (lhs >= rhs) ? 1 : 0, U8 mask, broadcast)

Elementwise greater-than-or-equal predicate producing a U8 mask; CPU only.

`GreaterEqualElementwise` is registered on CPU only (`ge_elementwise_f32_cpu_wrapper` and
f64/bf16/f16 siblings, `dispatch.rs:4296`); key `[T, T, U8]`. Output dtype always U8: `1` where
`lhs >= rhs`, `0` otherwise. Broadcast shape; not in-place. IEEE ordering (NaN comparisons false).
No CUDA / Vulkan registration.

```fkc
kernel: greater_equal_elementwise_cpu
op_kind: GreaterEqualElementwise
blurb: "Elementwise greater-or-equal out=(lhs>=rhs)?1:0 producing a U8 mask (broadcast); CPU only."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::ge_elementwise_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U8)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes + n"
  memory: { device_bytes: 0, host_bytes: "n", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact boolean predicate; 0/1 U8 mask. IEEE >= (any NaN comparison is false). bf16/f16 compared via their f32 value. Deterministic; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

---

## where_select  (Where — out = cond ? a : b, ternary select, broadcast)

Ternary select `out = cond ? a : b`: a U8 condition mask selects between two same-dtype tensors;
CPU only.

`Where` is registered on CPU only (`where_f32_cpu_wrapper` and f64/bf16/f16 siblings,
`dispatch.rs:4304`). The binding-table dtype key is `[U8, T, T, T]` = `(cond, a, b, out)` — three
input operands plus the output. The wrapper validates `cond.dtype == U8` (`dispatch.rs:483`); where
the condition byte is non-zero the corresponding `a` element is selected, else `b`. Output dtype =
the a/b dtype T (NOT U8); output shape = broadcast of the three inputs; not in-place. No CUDA /
Vulkan registration for Where in this crate.

```fkc
kernel: where_select_cpu
op_kind: Where
blurb: "Ternary select out=cond?a:b with a U8 condition mask (broadcast); CPU only."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::where_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: cond
      dtypes: [U8]                        # validated == U8 in the wrapper (dispatch.rs:483)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=a
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=b
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=a
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)          # output dtype is the a/b dtype T, NOT the U8 cond
      shape_rule: broadcast(cond, a, b)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                            # one branchless select per output element
  bytes_moved: "n + 3 * n * dtype_bytes"  # read n cond bytes + a + b (T), write out (T)
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact value-select: output is bitwise one of the two input operands (no arithmetic). cond nonzero -> a, else b. Deterministic; bit-stable on same hardware."

determinism: same_hardware_bitwise
```

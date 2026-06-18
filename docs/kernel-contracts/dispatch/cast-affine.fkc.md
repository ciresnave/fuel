---
fkc_version: 1
provider:
  name: fuel-dispatch
  backend: Cpu                                   # default; per-kernel `backend:` overrides for CU/VK
  kernel_source: "dispatch-cpu"                  # default BindingEntry.kernel_source tag; overridden per kernel
  link_registry: fuel_dispatch::fkc::ENTRY_POINTS  # §12.6 symbol -> KernelRef map
  revision_base: "git:f41137b4"                  # provider build id, folded into kernel_revision_hash
---

# fuel-dispatch — cast / affine kernel contracts

Dtype-conversion (`OpKind::Cast`) and scalar-affine (`OpKind::Affine`, `y = mul*x + add`) kernels as
registered by the `fuel-dispatch` crate across its three backend registration paths:
`register_cpu_kernels` (`dispatch.rs`), `register_baracuda_cuda_kernels` (`baracuda_dispatch.rs`),
and `register_vulkan_kernels` (`vulkan_dispatch.rs`). These two ops share this file (crate
`dispatch`, family `cast`) per the inventory grouping (`docs/kernel-contracts/_inventory/dispatch.md`,
"Affine" lines 368-376 and "Cast" lines 378-393).

This contract is authored at the **dispatch-wrapper granularity**, faithful to the inventory's
"one entry per kernel" rule (`_inventory/dispatch.md` lines 22-26): dtype-monomorphized variants are
collapsed into the dtype list of a single registered wrapper, because the dispatch layer registers
ONE dtype-dispatching wrapper per `(op, backend)` — e.g. the CUDA cast is a single
`cast_baracuda_wrapper` that dispatches on the in/out dtype pair, and the CPU cast is the per-target
`cpu_cast_wrapper` family matching the source dtype internally. A section is split only where the
backend genuinely registers a distinct kernel with different caps (the Vulkan bf16 affine, which is
contiguous-only while its f32/f64/f16 siblings are strided — `_inventory/dispatch.md` lines 371-376).
Each section's `dtypes` list and per-backend `kernel_source` / `entry_point` make the distinct
`(OpKind, [DType...], BackendId) + kernel_source` dispatch keys (§3.2, §12.1).

**Universal facts in this file.**
- **Output behavior** is always executor-preallocated, fully overwritten, **fresh contiguous** buffer,
  no aliasing, not in-place (`_inventory/dispatch.md` lines 45-47). Affine output dtype/shape equal
  the input (`dtype_rule: passthrough(input)`); Cast output dtype is the destination dtype carried on
  the output Storage (`OpParams::Cast` is a unit marker, target on the output `dtype` field), so the
  output dtype rule is `cast(output)` (§5.1).
- **Layout caps mirror `KernelCaps` exactly, not a guess** (`_inventory/dispatch.md` lines 27-38):
  CPU wrappers take `_layouts` UNUSED and operate on raw `CpuStorageBytes`, so CPU is
  `contiguous: required` (auto-Contiguize realizes any strided/broadcast/offset input first). CUDA
  baracuda Affine registers `strided_input` (stride/broadcast capable, **not** offset-capable — even
  strided CU kernels send non-zero-`start_offset` inputs through auto-Contiguize, `compiled.rs:58`).
  Vulkan Affine is strided for f32/f64/f16 but contiguous-only for bf16. **All Cast kernels are
  contiguous-only on every backend** (`_inventory/dispatch.md` line 388). NO kernel in this crate is
  offset-capable (`_inventory/dispatch.md` lines 700-703), and none walks negative strides, so
  `reverse_strides: rejected` everywhere.
- **Cost is `judge_measured`** for every kernel: the Judge bootstraps it (§4.4). Where genuinely
  derivable from the op an FDX/FLOPs hint is given — both Affine and Cast are bandwidth-bound
  elementwise (`bytes_moved` derivable from element count and dtype widths); no `flops`/`overhead_ns`/
  precise-frontier number is fabricated.
- **Precision** seeds are author-declared and Judge-audited (§4.8). CPU primitives leave precision
  `audited: false` so the importer applies the `PRIMITIVE_DETERMINISTIC_CPU` family default
  (`fill_unset_cpu_precision`; `_inventory/dispatch.md` lines 53-56, §12.4); CPU half (bf16/f16)
  computes/accumulates in f32. Vulkan/CUDA elementwise pointwise casts/affines are bit-stable on the
  same hardware (no cross-thread reduction).

---

## affine_cpu  (y = mul*x + add, CPU, contiguous)

Element-wise affine transform `out[i] = mul * input[i] + add` over a contiguous, zero-offset,
row-major `CpuStorageBytes` buffer. One registered surface (`register_cpu_kernels`, dispatch.rs:4099 /
4583) covers `f32, f64, bf16, f16`; the byte wrapper ignores `_layouts` and relies on the executor's
auto-Contiguize pass, so every operand is `contiguous, offset 0`. Scalar params `(mul, add)` arrive on
`OpParams::Affine { mul: f64, add: f64 }`; `f32`/`f64` compute natively, `bf16`/`f16` widen each
element to f32, compute `mul*x + add`, and narrow on store (the half-family precision invariant). One
kernel covers `Op::AddScalar(c)` (lowered `mul=1, add=c`) and `Op::MulScalar(c)` (lowered `mul=c,
add=0`). Pre-allocated output, full overwrite, no aliasing. Bit-stable on the same hardware (literal
`mul*x + add`, no FMA). Known limitation: contiguous-only — any strided/broadcast/offset operand must
be contiguized by the planner first (the inserted `Op::Contiguize` is itself an FKC kernel, §4.3).

```fkc
kernel: affine_cpu
op_kind: Affine
blurb: "Elementwise affine y = mul*x + add (CPU f32/f64/bf16/f16; half via f32); contiguous; covers AddScalar/MulScalar."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::affine_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params:
    variant: Affine                 # OpParams::Affine { mul: f64, add: f64 }
    fields:
      mul: { kind: f64, note: "consumed at f32 for bf16/f16; native for f32/f64" }
      add: { kind: f64, note: "consumed at f32 for bf16/f16; native for f32/f64" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous    # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8           # dtype-agnostic byte wrapper; element width from output Storage dtype tag

cost:
  provenance: judge_measured           # Judge bootstraps; bandwidth-bound elementwise hint below (§4.4)
  class: cheap_elementwise
  flops: "2 * n"                       # one multiply + one add per element
  bytes_moved: "2 * n * dtype_bytes"   # read input, write out
  overhead_ns: ~                       # judge_measured
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                       # CPU primitive: importer applies PRIMITIVE_DETERMINISTIC_CPU family default (§12.4)
  notes: "f32/f64 native mul-then-add; bf16/f16 widen to f32, compute, narrow on store; deterministic; bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## affine_cuda  (y = mul*x + add, CUDA/baracuda, strided)

Element-wise affine for the CUDA backend, bound to baracuda (`register_baracuda_cuda_kernels`,
baracuda_dispatch.rs:2656). Registered `strided_input`: the baracuda FFI is stride-driven, so the
wrapper passes the input `Layout` strides (including stride-0 broadcast axes for the scalar-broadcast
pattern) and walks them directly — no auto-Contiguize for a plain strided/broadcast view. Per the
inventory this is the widest dtype surface of the affine family: `f32, f64, f16, bf16, i32, i64, u8`
(integer affine is supported here, unlike CPU/VK which are float-only). Scalar params on
`OpParams::Affine { mul: f64, add: f64 }`. **Not offset-capable**: a non-zero-`start_offset` input
still routes through auto-Contiguize (`compiled.rs:58`), so `start_offset: rejected`. Pre-allocated
output, full overwrite, no aliasing.

```fkc
kernel: affine_cuda
op_kind: Affine
blurb: "Elementwise affine y = mul*x + add (CUDA/baracuda f32/f64/f16/bf16/i32/i64/u8); strided/broadcast; not offset-capable."
backend: Cuda
kernel_source: "baracuda"
entry_point: "baracuda::affine"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, F16, BF16, I32, I64, U8]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params:
    variant: Affine                 # OpParams::Affine { mul: f64, add: f64 }
    fields:
      mul: { kind: f64 }
      add: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided        # walks input strides directly via baracuda FFI; no contiguize cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured           # Judge bootstraps; bandwidth-bound elementwise hint below (§4.4)
  class: cheap_elementwise
  flops: "2 * n"                       # one multiply + one add per element
  bytes_moved: "2 * n * dtype_bytes"   # read input, write out
  overhead_ns: ~                       # judge_measured (CUDA launch)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                       # author-declared seed; Judge audits (§4.8)
  notes: "pointwise mul-then-add (no cross-thread reduction); deterministic; bit-stable same hardware; integer dtypes exact."

determinism: same_hardware_bitwise
```

---

## affine_vulkan  (y = mul*x + add, Vulkan f32/f64/f16, strided)

Element-wise affine for the Vulkan backend (`register_vulkan_kernels`, vulkan_dispatch.rs:4648),
strided variant covering `f32, f64, f16`. The Slang kernel walks input strides (stride/broadcast
capable). Scalar params on `OpParams::Affine { mul: f64, add: f64 }`. **Not offset-capable**
(non-zero `start_offset` inputs auto-Contiguize). Pre-allocated output, full overwrite, no aliasing.
Bit-stable on the same hardware (pointwise, no reduction). The `bf16` affine is a **separate kernel**
with different caps — see `affine_vulkan_bf16` (contiguous-only).

```fkc
kernel: affine_vulkan
op_kind: Affine
blurb: "Elementwise affine y = mul*x + add (Vulkan f32/f64/f16); strided/broadcast; not offset-capable."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::affine"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
  op_params:
    variant: Affine                 # OpParams::Affine { mul: f64, add: f64 }
    fields:
      mul: { kind: f64 }
      add: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided        # Slang kernel walks strides; no contiguize cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured           # Judge bootstraps; bandwidth-bound elementwise hint below (§4.4)
  class: cheap_elementwise
  flops: "2 * n"                       # one multiply + one add per element
  bytes_moved: "2 * n * dtype_bytes"   # read input, write out
  overhead_ns: ~                       # judge_measured (Vulkan command-buffer submit)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                       # author-declared seed; Judge audits (§4.8)
  notes: "pointwise mul-then-add (no cross-thread reduction); deterministic; bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## affine_vulkan_bf16  (y = mul*x + add, Vulkan bf16, CONTIGUOUS-ONLY)

Element-wise affine for `BF16` on Vulkan (`register_vulkan_kernels`, vulkan_dispatch.rs:4658). This
is a **distinct kernel** from `affine_vulkan`: it uses a pair-thread packed-`u32` kernel that has **no
strided cap**, a deliberate divergence from its f32/f64/f16 siblings (`_inventory/dispatch.md` lines
371-376). It is therefore `contiguous: required` — the planner must contiguize any strided/broadcast/
offset input first (the inserted `Op::Contiguize` is its own FKC kernel, §4.3). Scalar params on
`OpParams::Affine { mul: f64, add: f64 }`, consumed at f32 (widen to f32, compute, narrow to bf16 on
store). Pre-allocated output, full overwrite, no aliasing. Bit-stable on the same hardware.

```fkc
kernel: affine_vulkan_bf16
op_kind: Affine
blurb: "Elementwise affine y = mul*x + add (Vulkan bf16; pair-thread packed u32; CONTIGUOUS-ONLY; half via f32)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::affine_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "pair-thread packed-u32 kernel: NO strided cap (divergence from f32/f64/f16 siblings)."
  op_params:
    variant: Affine                 # OpParams::Affine { mul: f64, add: f64 }; consumed at f32
    fields:
      mul: { kind: f64, note: "narrowed to f32 at the kernel ABI" }
      add: { kind: f64, note: "narrowed to f32 at the kernel ABI" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous    # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32           # packed two bf16 per u32 word

cost:
  provenance: judge_measured           # Judge bootstraps; bandwidth-bound elementwise hint below (§4.4)
  class: cheap_elementwise
  flops: "2 * n"                       # multiply + add per element (computed in f32)
  bytes_moved: "2 * n * dtype_bytes"   # read input, write out; dtype_bytes = 2 (bf16)
  overhead_ns: ~                       # judge_measured (Vulkan command-buffer submit)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                       # author-declared seed; Judge audits (§4.8)
  notes: "widen to f32, mul-then-add in f32, narrow to bf16 on store; deterministic; bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## cast_cpu  (dtype conversion, CPU, contiguous)

Dtype-conversion for the CPU backend (`register_cpu_kernels`, dispatch.rs:3993). Each per-target
`cpu_cast_wrapper` matches the source dtype internally; identity casts are elided by the optimizer
(the identity arm is omitted). The registered src->dst surface is: `F64->F32, BF16->F32, F16->F32,
F32->F64, F32->BF16, F32->F16`, plus `F8E4M3 <-> {F32, BF16, F16}`. `half<->half` (bf16<->f16) pivots
through f32 on CPU. The wrappers ignore `_layouts` and operate on contiguous, zero-offset byte
buffers (auto-Contiguize realizes any awkward layout first), so `contiguous: required`. `OpParams::Cast`
is a unit marker — the **destination dtype lives on the output Storage's `dtype` field** (the last
entry of the dispatch key), so the output dtype rule is `cast(output)` (§5.1) and the same
`OpKind::Cast` is shared across all pairs, distinguished by the operand dtype slots
(`(OpKind::Cast, [SRC, DST], Cpu)`, §3.2). Pre-allocated output, full overwrite, no aliasing.
Bandwidth-bound elementwise.

```fkc
kernel: cast_cpu
op_kind: Cast
blurb: "Dtype cast (CPU): F64/BF16/F16<->F32, F32<->F64/BF16/F16, F8E4M3<->{F32,BF16,F16}; half<->half via f32; contiguous."
backend: Cpu
kernel_source: "dispatch-cpu"
entry_point: "fuel_dispatch::dispatch::cpu_cast_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F64, BF16, F16, F8E4M3]   # accepted SOURCE dtypes; valid (src,dst) pairs per the registered surface (see blurb)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "per-target wrapper matches src internally; identity (src==dst) casts elided by the optimizer; bf16<->f16 pivots through f32."
  op_params: { variant: Cast }     # OpParams::Cast is a unit marker; target dtype on the output Storage

return:
  outputs:
    - name: out
      dtype_rule: cast(output)          # destination dtype = output Storage dtype (last key entry) (§5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous    # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8            # dtype-agnostic byte handling; element widths from src/dst dtype tags

cost:
  provenance: judge_measured           # Judge bootstraps; bandwidth-bound elementwise hint below (§4.4)
  class: cheap_elementwise
  bytes_moved: "n * (src_bytes + dst_bytes)"   # read n src elems + write n dst elems
  memory: { device_bytes: 0, host_bytes: "n * dst_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                       # CPU primitive: importer applies PRIMITIVE_DETERMINISTIC_CPU family default (§12.4)
  notes: "widening (e.g. F16/BF16/F32->wider) exact; narrowing per IEEE rounding; bf16<->f16 pivots through f32; F8E4M3 RNE/saturate on narrow, exact decode on widen."

determinism: same_hardware_bitwise
```

---

## cast_cuda  (dtype conversion, CUDA/baracuda, contiguous, full cross-product)

Dtype-conversion for the CUDA backend, bound to baracuda (`register_baracuda_cuda_kernels`,
baracuda_dispatch.rs:2747). A **single** `cast_baracuda_wrapper` dispatches on the in/out dtype pair,
covering the **full 8x8 cross product** over `{F32, F64, F16, BF16, I32, U32, I64, U8}` plus
`F8E4M3 <-> {F32, F16, BF16}`. Registered with default (all-false) caps -> `contiguous: required`
(the inventory marks CU Cast as **C**, distinct from baracuda's strided elementwise kernels;
`_inventory/dispatch.md` lines 61-62, 388). `OpParams::Cast` is a unit marker — destination dtype on
the output Storage. Pre-allocated output, full overwrite, no aliasing. Bandwidth-bound elementwise;
int<->float and narrowing casts follow CUDA conversion semantics. Bit-stable on the same hardware
(pointwise, no reduction).

```fkc
kernel: cast_cuda
op_kind: Cast
blurb: "Dtype cast (CUDA/baracuda): full 8x8 over {F32,F64,F16,BF16,I32,U32,I64,U8} + F8E4M3<->{F32,F16,BF16}; contiguous."
backend: Cuda
kernel_source: "baracuda"
entry_point: "baracuda::cast"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F64, F16, BF16, I32, U32, I64, U8, F8E4M3]   # accepted SOURCE dtypes; full cross-product + F8E4M3 legs (see blurb)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "one cast_baracuda_wrapper dispatching on (src,dst); F8E4M3 only pairs with {F32,F16,BF16}; identity casts elided by the optimizer."
  op_params: { variant: Cast }     # OpParams::Cast unit marker; target dtype on the output Storage

return:
  outputs:
    - name: out
      dtype_rule: cast(output)          # destination dtype = output Storage dtype (last key entry) (§5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous    # default caps; planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured           # Judge bootstraps; bandwidth-bound elementwise hint below (§4.4)
  class: cheap_elementwise
  bytes_moved: "n * (src_bytes + dst_bytes)"   # read n src elems + write n dst elems
  memory: { device_bytes: "n * dst_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                       # author-declared seed; Judge audits (§4.8)
  notes: "widening exact; narrowing per CUDA IEEE rounding; int<->float per CUDA convert semantics; F8E4M3 RNE/saturate on narrow, exact decode on widen; pointwise, bit-stable same hardware."

determinism: same_hardware_bitwise
```

---

## cast_vulkan  (dtype conversion, Vulkan, contiguous)

Dtype-conversion for the Vulkan backend (`register_vulkan_kernels`, vulkan_dispatch.rs:4680 / 5014).
The registered surface: `f32<->f16`, `f32<->bf16` (the `cast_f32_half` pair-packed kernels);
`f32<->f64` (feature-gated, requires device `shaderFloat64`); and `F8E4M3 <-> {F32, F16, BF16}` (the
`cast_f8e4m3` byte-packed kernels). Contiguous-only (byte-level; `_inventory/dispatch.md` line 388).
`OpParams::Cast` is a unit marker — destination dtype on the output Storage. Pre-allocated output,
full overwrite, no aliasing. Bandwidth-bound elementwise; bit-stable on the same hardware. (The
per-direction numeric detail — pair-packing, truncate-vs-RNE for bf16, F8E4M3 saturation to +/-448 —
is contracted at per-pair granularity in `docs/kernel-contracts/vulkan/cast.fkc.md`; this section is
the dispatch-crate registration view, faithful to the dispatch inventory's collapsed Vulkan Cast row.)

```fkc
kernel: cast_vulkan
op_kind: Cast
blurb: "Dtype cast (Vulkan): f32<->f16, f32<->bf16, f32<->f64 (gated), F8E4M3<->{F32,F16,BF16}; contiguous."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::cast"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16, F64, F8E4M3]   # accepted SOURCE dtypes; valid pairs per the registered surface (see blurb)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "f32<->{f16,bf16} pair-packed; f32<->f64 needs shaderFloat64; F8E4M3 routes non-F8 side via f32; identity casts elided by the optimizer."
  op_params: { variant: Cast }     # OpParams::Cast unit marker; target dtype on the output Storage

return:
  outputs:
    - name: out
      dtype_rule: cast(output)          # destination dtype = output Storage dtype (last key entry) (§5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous    # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured           # Judge bootstraps; bandwidth-bound elementwise hint below (§4.4)
  class: cheap_elementwise
  bytes_moved: "n * (src_bytes + dst_bytes)"   # read n src elems + write n dst elems
  memory: { device_bytes: "n * dst_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                       # author-declared seed; Judge audits (§4.8)
  notes: "f16/bf16/f8<->f32 per per-pair semantics (f32->bf16 truncate, f16/f8 RNE, widening exact); F8E4M3 saturate +/-448 on narrow; pointwise, bit-stable same hardware. Per-direction detail in vulkan/cast.fkc.md."

determinism: same_hardware_bitwise
```

---
fkc_version: 1
provider:
  name: fuel-dispatch
  backend: Cpu                       # the always-built registration site this bundle's `fkc` blocks describe (BackendId::Cpu)
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag for the CPU block; CU/VK siblings override per section
  link_registry: fuel_dispatch::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-dispatch — shape-ops kernel contracts

The dispatch-layer (`KernelBindingTable`) registrations for Fuel's shape / movement, triangular,
prefix-sum, padding, in-place KV-cache scatter, and cross-device copy ops. Family: **shape-ops**.
Source of truth: `fuel-dispatch/src/{kernel.rs, dispatch.rs, baracuda_dispatch.rs,
vulkan_dispatch.rs, compiled.rs}` and the inventory `docs/kernel-contracts/_inventory/dispatch.md`.

**This is a multi-backend provider.** Each op below is registered across some subset of
{CPU (`register_cpu_kernels`), baracuda-CUDA (`register_baracuda_cuda_kernels`),
Vulkan (`register_vulkan_kernels`)} — distinct `KernelRef`s at distinct `(op, dtypes, backend)`
keys, i.e. **sibling alternatives** the route picker ranks (§12.5). The `fkc` block in each section
describes the **CPU registration** (the universal always-built fallback the inventory's `source`
column points at first); the **per-backend layout-capability and dtype differences are documented
in each section's prose and in the layout-block comments**, because they are the load-bearing
planner facts that differ. Where a backend genuinely diverges in the five-flag layout set
(CPU contiguous-only vs baracuda/Vulkan strided), that fact is surfaced explicitly — it is never
hidden behind backend code (the 01 visibility gate, §4.3).

**Cross-cutting layout facts (from the inventory).**

- **Every CPU wrapper is `contiguous-only`, zero-offset, row-major.** CPU wrappers take
  `_layouts: &[Layout]` *unused* and operate on raw `CpuStorageBytes`; geometry comes from
  `OpParams`, element size from the output Storage's `dtype` tag. They rely entirely on the
  executor's auto-Contiguize pass, so each CPU contract declares
  `awkward_layout_strategy: requires_contiguous` and the planner prices the inserted
  `Op::Contiguize` from the `contiguize` contract (§4.3, §4.4).
- **baracuda-CUDA and Vulkan are stride-driven for Flip/Roll/CumSum** (`strided` = stride +
  stride-0-broadcast capable, **NOT offset-capable**): they consume `layouts[0]` and walk
  rank-N strides. For Pad/PadBackward/WriteSlice/WriteSliceRotating/Copy the CU/VK kernels are
  also `contiguous-only` (byte-width-keyed slab/transfer kernels). Triu/Tril: CU is `strided`,
  VK is `contiguous-only` (byte-level).
- **No kernel in this crate is offset-capable.** Even a strided-capable kernel sends any input
  with a non-zero `start_offset` through auto-Contiguize (`compiled.rs:58`, `KernelCaps` doc
  `kernel.rs:66-74`). So `start_offset: rejected` everywhere; `strided: accepted` (where declared)
  means stride + broadcast only.
- **Negative strides are not walked by any dispatch kernel.** Flip *materializes* the reversal
  (it does not produce or consume a negative-stride view), so `reverse_strides: rejected` on every
  operand here.

**Output behavior.** Output Storage is **always pre-allocated by the executor**; no kernel
allocates (the `KernelRef` ABI hard rule). Wrappers fill the pre-allocated bytes. The movement /
triangular / pad / write-slice kernels are **dtype-agnostic byte copies** (element size from the
output dtype tag); CumSum is **per-dtype** (typed add, f32-accumulator for half). WriteSlice /
WriteSliceRotating **partially overwrite and alias** the destination buffer in place.

**Cost provenance.** Every kernel below is marked **`provenance: judge_measured`** — the Judge
bootstraps and refines the coefficients (FKC stays agnostic to *how*, §4.4). These are
bandwidth-bound byte-movement / linear-scan kernels, so a genuinely derivable bandwidth/FLOPs
shape is recorded in the expression strings as the honest cost shape (the Judge refines the
constants); no kernel ships a placeholder or unmarked cost (§10.8a).

---

## pad  (multi-dim constant/reflect/replicate padding, forward)

Pad each axis of an input tensor with `(before, after)` elements, filling the new border per a
`mode_tag` (0=Constant, 1=Reflect, 2=Replicate).

`OpKind::Pad` (registration `dispatch.rs:4463` CPU, `baracuda_dispatch.rs:2642` CU) copies a dense
contiguous `in_shape` tensor into a larger `out_shape` tensor where
`out_shape[i] == in_shape[i] + padding[i].before + padding[i].after`. The CPU wrapper is a
**dtype-agnostic byte kernel** (element size from the output Storage's dtype tag); `fill_bytes`
is one element's worth of bytes pre-encoded in the output dtype for the Constant border. **Mode
support differs by backend (an as-built fact):** CPU has **Constant wired; Reflect and Replicate
error in the wrapper today**; the baracuda-CUDA kernel implements **Constant / Reflect /
Replicate**. Both backends are `contiguous-only` (`input_layouts` **C** for CPU and CU). Numerics:
pure byte copy of interior elements + a deterministic border fill — bit-exact, hardware-independent
for every dtype. Perf: bandwidth-bound; the output is written once (interior copy + border fill)
≈ `out_elems * dtype_bytes`.

```fkc
kernel: pad
op_kind: Pad
blurb: "Multi-dim pad (Constant/Reflect/Replicate via mode_tag); dtype-agnostic byte copy + border fill; contiguous-only."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::pad_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      # CPU dtype-agnostic byte wrapper: f32, f64, bf16, f16, u32, u8. (CU: f32, f64, f16, bf16.)
      dtypes: [F32, F64, BF16, F16, U32, U8]
      # CPU C and CU C: contiguous-only, zero-offset. Geometry from OpParams::Pad, not a Layout.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any                       # rank == padding.len() == in_shape.len() == out_shape.len()
      shape_constraint: "dim[i] == in_shape[i]"
  op_params:
    variant: Pad                      # OpParams::Pad (kernel.rs:554)
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", constraint: "out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "len == rank; per-axis (before, after)" }
      mode_tag:  { kind: u8, constraint: "0=Constant (CPU+CU), 1=Reflect (CU only), 2=Replicate (CU only)" }
      fill_bytes:{ kind: "Vec<u8>", note: "one output-dtype element's worth; Constant border fill" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)        # output dtype == input dtype
      shape_rule: "from_params(out_shape)"  # out_shape[i] = in_shape[i] + before + after
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (priced from `contiguize`) for non-contig input
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound; output written once ≈ in_bytes read + out_bytes written
  class: strided_elementwise
  flops: "0"                          # pure byte copy + border fill; no arithmetic
  bytes_moved: "(prod(in_shape) + prod(out_shape)) * dtype_bytes"   # read interior + write padded output
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # output is caller-preallocated

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy + deterministic border fill; bit-exact, hardware-independent. CPU wires Constant only (Reflect/Replicate error); CU wires all three modes."

determinism: bitwise
```

---

## pad_backward  (gradient of Pad — sum-accumulate the unpadded interior)

Backward of `Pad`: extract (Constant) / sum-accumulate (Reflect/Replicate) the gradient of the
padded region back into the input-shaped gradient.

`OpKind::PadBackward` (registration `dispatch.rs:4471` CPU, `baracuda_dispatch.rs:2647` CU) takes an
`out_shape`-sized upstream gradient and produces an `in_shape`-sized gradient. For **Constant** mode
this is a plain interior slice (the border gradient is discarded); for **Reflect / Replicate** the
border gradient is **scatter-added** back to the interior positions the forward op read from — a
typed accumulation, which is why this is a per-dtype (not byte-agnostic) operation. **As-built scope
(inventory):** the CPU wrapper handles the sum-accumulating backward; the baracuda-CUDA kernel is
**Constant only** (the sum-accumulating backward is CPU-only). Both are `contiguous-only`
(`input_layouts` **C**). Numerics: typed accumulation in the operand dtype (half accumulates per the
op's f32-widen convention); deterministic gather/scatter order ⇒ bit-stable on the same hardware.
Perf: bandwidth-bound in the output (interior) volume; Reflect/Replicate add a small number of
extra accumulating writes per border element.

```fkc
kernel: pad_backward
op_kind: PadBackward
blurb: "Gradient of Pad: interior slice (Constant) or sum-accumulate border grads (Reflect/Replicate); per-dtype typed accumulation."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::pad_backward_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: upstream
      dtypes: [F32, F64, BF16, F16]   # per-dtype typed accumulation (CPU + CU)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any                       # rank == out_shape.len() == in_shape.len()
      shape_constraint: "dim[i] == out_shape[i]"
  op_params:
    variant: PadBackward              # OpParams::PadBackward (kernel.rs:536)
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", constraint: "out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "len == rank" }
      mode_tag:  { kind: u8, constraint: "0=Constant (CPU+CU), 1=Reflect / 2=Replicate (CPU only — sum-accumulating backward)" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(upstream)     # gradient dtype == upstream dtype
      shape_rule: "from_params(in_shape)"
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "mode_tag == 0", note: "Constant: interior slice, no accumulation" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound; read out_shape grad, write in_shape grad; border accumulation adds extra writes
  class: strided_elementwise
  flops: "prod(out_shape)"            # ~1 add/element for the accumulating (Reflect/Replicate) path; 0 for Constant
  bytes_moved: "(prod(out_shape) + prod(in_shape)) * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # typed accumulation in a fixed, deterministic gather/scatter order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Typed accumulation in the operand dtype (half widens to f32); fixed deterministic order. CPU handles all modes; CU is Constant only."

determinism: same_hardware_bitwise
```

---

## flip  (reverse element order along one dim)

Reverse the order of elements along one dimension. **Materializing** byte reorder — not a
negative-stride view.

`OpKind::Flip` (registration `dispatch.rs:4359` CPU, `baracuda_dispatch.rs:2618` CU,
`vulkan_dispatch.rs:4839` VK) computes `out[outer, j, inner] = in[outer, dim_size-1-j, inner]`,
copying `dtype_size` bytes per element into a reversed position. The CPU wrapper is a single
**dtype-agnostic byte kernel** (`flip_cpu_wrapper`) over a contiguous, zero-offset buffer factored
into `(outer_count, dim_size, inner_count)`; element size from the output dtype tag. **Layout caps
differ by backend (inventory):** CPU **C** (contiguous-only); baracuda-CUDA **S** and Vulkan **S**
walk rank-N strides via the `axis` field (the FFI/Slang kernels are stride-driven). No backend walks
*negative* strides — the reversal is always materialized — so `reverse_strides: rejected` on accept
and return. Numerics: pure byte permutation — bit-exact, deterministic across any hardware. Perf:
bandwidth-bound, `outer*dim_size` row copies of `inner*dtype_size` bytes ≈ `2*N*dtype_bytes`.

```fkc
kernel: flip
op_kind: Flip
blurb: "Reverse element order along one dim (out[..,j,..]=in[..,dim-1-j,..]); materializing dtype-agnostic byte reorder."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flip_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      # CPU dtype-agnostic byte wrapper: f32, f64, bf16, f16, u32, u8. (CU: f32, f64, f16, bf16; VK: per dtype.)
      dtypes: [F32, F64, BF16, F16, U32, U8]
      # CPU C (contiguous-only). CU/VK are STRIDED (S) — they walk rank-N strides via OpParams::Flip.axis.
      # No backend walks negative strides: the reversal is MATERIALIZED, never a flip-view ⇒ reverse_strides: rejected.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any                       # scanned axis folded into outer/dim_size/inner; `axis` is the original rank-N dim
  op_params:
    variant: Flip                     # OpParams::Flip (kernel.rs:478)
    fields:
      outer_count: { kind: usize }
      dim_size:    { kind: usize }
      inner_count: { kind: usize }
      axis:        { kind: usize, note: "original rank-N dim index; used by the stride-aware CU/VK kernels" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)            # same shape, reversed along the flipped dim
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # CPU contract: planner inserts Op::Contiguize (CU/VK siblings declare strided)
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound, 2*N*dtype_bytes moved (read each row + write reversed)
  class: strided_elementwise
  flops: "0"                          # pure reorder, no arithmetic
  bytes_moved: "2 * outer_count * dim_size * inner_count * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # output caller-preallocated (internal scratch on CPU)

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte permutation; bit-exact, hardware-independent."

determinism: bitwise
```

---

## roll  (cyclic shift along one dim)

Cyclically shift elements along one dimension by a signed `shift` (always wraps).

`OpKind::Roll` (registration `dispatch.rs:4366` CPU, `baracuda_dispatch.rs:2624` CU,
`vulkan_dispatch.rs:4840` VK) computes `out[outer, j, inner] = in[outer, (j-shift) mod dim_size,
inner]` with Python-style modulo (positive `shift` moves elements to higher indices, negative the
opposite); the shift is normalized into `[0, dim_size)` once. CPU is a **dtype-agnostic byte kernel**
(`roll_cpu_wrapper`) over a contiguous, zero-offset `(outer_count, dim_size, inner_count)` factoring;
element size from the output dtype tag. **Layout caps (inventory):** CPU **C**; baracuda-CUDA **S**
and Vulkan **S** (stride-driven via `axis`). Numerics: pure byte permutation — bit-exact,
deterministic across any hardware. `dim_size == 0` is a no-op. Perf: bandwidth-bound,
`outer*dim_size` row copies ≈ `2*N*dtype_bytes`.

```fkc
kernel: roll
op_kind: Roll
blurb: "Cyclic shift along one dim (out[..,j,..]=in[..,(j-shift) mod dim,..]); always wraps; dtype-agnostic byte reorder."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::roll_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16, U32, U8]   # CPU byte-agnostic; CU: f32,f64,f16,bf16; VK: per dtype
      # CPU C (contiguous-only). CU/VK are STRIDED (S) — walk rank-N strides via OpParams::Roll.axis.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Roll                     # OpParams::Roll (kernel.rs:489)
    fields:
      outer_count: { kind: usize }
      dim_size:    { kind: usize }
      inner_count: { kind: usize }
      shift:       { kind: i64, note: "signed; normalized into [0,dim_size) by Python-style modulo" }
      axis:        { kind: usize, note: "original rank-N dim index; used by stride-aware CU/VK kernels" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)            # same shape, cyclically shifted along the dim
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dim_size == 0", class: free, note: "empty axis: no-op early return" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound, 2*N*dtype_bytes moved
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * outer_count * dim_size * inner_count * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte permutation; bit-exact, hardware-independent."

determinism: bitwise
```

---

## cumsum  (running inclusive prefix sum along one dim)

Running inclusive prefix sum along one dimension. **Per-dtype typed add** (not a byte copy);
half-precision accumulates in f32.

`OpKind::CumSum` (registration `dispatch.rs:4374` CPU, `baracuda_dispatch.rs:2632` CU,
`vulkan_dispatch.rs:4850` VK) computes, for each `(outer, inner)` lane,
`out[..,j,..] = Σ_{t<=j} in[..,t,..]` along `dim_size`, factored as
`(outer_count, dim_size, inner_count)`. This is **per-dtype** (f32/f64 native accumulator; bf16/f16
widen to f32 and narrow on store — the load-bearing half-precision precision invariant), unlike the
byte-agnostic Flip/Roll. **Layout caps (inventory):** CPU **C** (contiguous-only); baracuda-CUDA
**S** and Vulkan **S** (stride-driven). Numerics: standard IEEE-754 accumulation; the sequential
add order along the scanned axis is fixed, so the **CPU** path is bit-stable on the same hardware
(rounding error accumulates with `dim_size`, the inherent prefix-sum behavior). Reverse-cumsum is
expressed upstream as Flip→CumSum→Flip. Perf: bandwidth-bound, `n` reads + `n` writes, one add per
element.

```fkc
kernel: cumsum
op_kind: CumSum
blurb: "Inclusive prefix sum along one dim; per-dtype typed add (f32 accumulator for half); fixed sequential order."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::cumsum_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]   # per-dtype typed add (CPU + CU + VK). NO byte-agnostic int path.
      # CPU C (contiguous-only). CU/VK are STRIDED (S).
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any                       # scanned axis folded into outer/dim_size/inner
  op_params:
    variant: CumSum                   # OpParams::CumSum (kernel.rs:500)
    fields:
      outer_count: { kind: usize }
      dim_size:    { kind: usize }
      inner_count: { kind: usize }
      axis:        { kind: usize, note: "original rank-N dim index; used by stride-aware CU/VK kernels" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)        # output dtype == input dtype (half stays half on store)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32         # f32 default; bf16/f16 access 16-bit, f64 64-bit (per-dtype sibling kernels)

cost:
  provenance: judge_measured          # hint: bandwidth-bound; n = outer*dim_size*inner; ~1 add/elem
  class: reduction
  flops: "outer_count * dim_size * inner_count"          # one add per element along the scan
  bytes_moved: "2 * outer_count * dim_size * inner_count * dtype_bytes"   # read input + write output
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # CPU: fixed sequential add order; half widens to f32 and narrows on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32/f64 native accumulator; bf16/f16 widen to f32, add, narrow on store. Fixed sequential summation order on CPU; IEEE-754 rounding accumulates with dim_size. (CU/VK accumulation order is the device kernel's; CPU is the bit-stable path.)"

determinism: same_hardware_bitwise
```

---

## triu  (upper-triangular mask along the last two dims)

Keep `x[..., i, j]` on the upper side of a signed `diagonal` offset (`j >= i + diagonal`), zero
elsewhere; dtype-agnostic.

`OpKind::Triu` (registration `dispatch.rs:4405` CPU, `baracuda_dispatch.rs:2599` CU,
`vulkan_dispatch.rs:4837` VK) shares `OpParams::Triangular` with `Tril`; the OpKind selects
keep-upper. The CPU kernel zeros the output (`fill(0)`) then overlays the kept positions, copying
`dtype_size` bytes per kept element from the input — **dtype-agnostic**, since all-zero bytes are the
correct zero for every IEEE-754 / integer dtype Fuel supports. A batch of `batch_count` `rows × cols`
matrices is processed (leading dims fold into `batch_count`). **Layout caps (inventory):** CPU **C**;
baracuda-CUDA **S** (stride-driven); Vulkan **C** (byte-level). Numerics: pure mask-and-copy —
bit-exact, deterministic across any hardware. Limitation: the last two dims define the triangle, so
rank ≥ 2. Perf: bandwidth-bound, output fully written once ≈ `batch*rows*cols*dtype_bytes`.

```fkc
kernel: triu
op_kind: Triu
blurb: "Upper-triangular mask (keep j>=i+diagonal, zero else) over the last two dims; dtype-agnostic zero-fill + copy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::triu_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      # CPU/VK byte-agnostic: f32, f64, bf16, f16, u32, u8. (CU adds i32, i64.)
      dtypes: [F32, F64, BF16, F16, U32, U8]
      # CPU C and VK C (byte-level). CU is STRIDED (S).
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=8"                   # last two dims = rows×cols; leading dims fold into batch_count
  op_params:
    variant: Triangular               # OpParams::Triangular (kernel.rs:512) — shared with Tril; OpKind picks keep-upper
    fields:
      batch_count: { kind: usize }
      rows:        { kind: usize }
      cols:        { kind: usize }
      diagonal:    { kind: i64, note: "signed; keep boundary is j vs i+diagonal (0=main diagonal)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # CPU/VK contract; CU sibling declares strided
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound, output fully written once (~N*dtype_bytes) + kept-element reads
  class: strided_elementwise
  flops: "0"                          # comparison + copy; no FP arithmetic
  bytes_moved: "2 * batch_count * rows * cols * dtype_bytes"   # zero-fill output + copy kept inputs
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Mask-and-copy with all-zero fill (valid zero for every Fuel dtype); bit-exact, hardware-independent."

determinism: bitwise
```

---

## tril  (lower-triangular mask along the last two dims)

Keep `x[..., i, j]` on the lower side of a signed `diagonal` offset (`j <= i + diagonal`), zero
elsewhere; dtype-agnostic. The mirror of `triu`.

`OpKind::Tril` (registration `dispatch.rs:4411` CPU, `baracuda_dispatch.rs:2599` CU,
`vulkan_dispatch.rs:4837` VK) shares the same `OpParams::Triangular` and wrapper family as `Triu`,
with the OpKind selecting keep-lower (`j <= i + diagonal`). Identical mechanics otherwise: zero the
output, overlay kept positions, dtype-agnostic byte copy, batched `rows × cols`. **Layout caps
(inventory):** CPU **C**; baracuda-CUDA **S**; Vulkan **C** (byte-level). Numerics: pure
mask-and-copy — bit-exact, deterministic. Rank ≥ 2. Perf: bandwidth-bound, output written once.

```fkc
kernel: tril
op_kind: Tril
blurb: "Lower-triangular mask (keep j<=i+diagonal, zero else) over the last two dims; dtype-agnostic zero-fill + copy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::tril_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16, U32, U8]   # CPU/VK byte-agnostic; CU adds i32, i64
      # CPU C and VK C (byte-level). CU is STRIDED (S).
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=8"                   # last two dims = rows×cols; leading dims fold into batch_count
  op_params:
    variant: Triangular               # OpParams::Triangular (kernel.rs:512) — shared with Triu; OpKind picks keep-lower
    fields:
      batch_count: { kind: usize }
      rows:        { kind: usize }
      cols:        { kind: usize }
      diagonal:    { kind: i64, note: "signed; keep boundary is j vs i+diagonal (0=main diagonal)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound, output fully written once (~N*dtype_bytes) + kept-element reads
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * batch_count * rows * cols * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Mask-and-copy with all-zero fill (valid zero for every Fuel dtype); bit-exact, hardware-independent."

determinism: bitwise
```

---

## write_slice  (in-place rectangular slab assign — persistent KV-cache write)

In-place rectangular scatter: write `source` into a per-axis half-open slab of `dest`; only the slab
bytes are touched. **Aliases `dest` in place.**

`OpKind::WriteSlice` (registration `dispatch.rs:4385` CPU, `baracuda_dispatch.rs:2567` CU,
`vulkan_dispatch.rs:4771` VK) copies `source`'s bytes into the rectangular slab of `dest` defined by
`ranges[i] = (start, end)` (half-open) per axis; the slab size on axis `i` is `end - start`, and
`source`'s element count equals the slab's. **`outputs[0]` IS the destination buffer, mutated in
place** — a partial overwrite (bytes outside the slab are preserved) — so the return-contract
aliasing is `in_place(dest)` and `caps.in_place: true`. This backs persistent KV-cache writes
(Phase E.3.2). The CPU wrapper is **dtype-agnostic** (element size from the output dtype tag); CU and
VK are **byte-width-keyed** (b1/b2/b4/b8 kernels) covering f32, f64, f16, bf16, i32, i64, u32, u8, i8.
**Layout caps (inventory):** CPU **C**, CU **C**, VK **C** (contiguous-only, byte-level — one source
input + one dest output). Numerics: pure byte copy — bit-exact, deterministic. Empty slab is a no-op.
Perf: bandwidth-bound in the slab volume; no read of prior `dest` content (only the slab is written).

```fkc
kernel: write_slice
op_kind: WriteSlice
blurb: "In-place rectangular scatter of source into a per-axis half-open slab of dest; dtype-agnostic; aliases dest."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::write_slice_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      # CPU dtype-agnostic. CU/VK byte-width-keyed: f32,f64,f16,bf16,i32,i64,u32,u8,i8.
      dtypes: [F32, F64, BF16, F16, U32, U8, I8, I16, I32, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"                   # source rank == dest rank; slab dim i = ranges[i].end - ranges[i].start
      shape_constraint: "dim[i] == ranges[i].end - ranges[i].start"
    - name: dest
      dtypes: [F32, F64, BF16, F16, U32, U8, I8, I16, I32, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"                   # rank >= 1 (scalar dest rejected — no slab)
      shape_constraint: "same_as=out; read-modify-written in place (this operand IS the output)"
  op_params:
    variant: WriteSlice               # OpParams::WriteSlice (kernel.rs:570)
    fields:
      dest_shape: { kind: "Vec<usize>", constraint: "rank >= 1" }
      ranges:     { kind: "Vec<(usize,usize)>", constraint: "len == rank; 0 <= start <= end <= dest_shape[i]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(dest)
      shape_rule: same_as(dest)
      layout_guarantee: same_as(dest)        # dest's contiguous layout preserved; only slab bytes change
      aliasing: in_place(dest)               # output IS dest's buffer; partial (slab-only) overwrite

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dim[0] == 0", class: free, note: "empty slab: no-op early return" }
    - { note: "rank==1: single contiguous span copy" }
  in_place: true                      # writes into dest's buffer (§4.6)
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound in slab volume; ~2*slab_bytes moved (read source + write slab)
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * source.n * dtype_bytes"   # source.n = slab element count; dest bytes outside the slab untouched
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # in-place; no allocation

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy into a dest slab; bit-exact, hardware-independent. Bytes outside the slab are preserved."

determinism: bitwise
```

---

## write_slice_rotating  (in-place ring-buffer slab assign — sliding-window KV cache)

In-place ring-buffer scatter: write `source` into a slab of `dest` whose `axis` wraps modulo
`modulus`, with the dynamic write start read from a `position` operand. **Aliases `dest` in place.**

`OpKind::WriteSliceRotating` (registration `dispatch.rs:4397` CPU, `baracuda_dispatch.rs:2581` CU,
`vulkan_dispatch.rs:4785` VK) is the sliding-window-KV-cache write (Phase C). The dynamic write start
on `axis` is `position % modulus`, where `position` is read from an **extra rank-0/rank-1 U32 input**
(data-determined per token, **not** a compile-time param). On the rotating axis `ranges[axis].0` is
ignored (dynamic) and the slab width `ranges[axis].1 - ranges[axis].0` must equal the source dim on
that axis and not exceed `modulus`; when the write crosses the ring boundary it splits into two slab
copies (prefix + suffix). **`outputs[0]` IS the destination buffer, mutated in place** (partial
overwrite) — `in_place(dest)`, `caps.in_place: true`. CPU is **dtype-agnostic**; CU/VK are
**byte-width-keyed** (same surface as WriteSlice). **Layout caps (inventory):** CPU **C**, CU **C**,
VK **C** (contiguous-only). Numerics: pure byte copy — bit-exact, deterministic. Perf:
bandwidth-bound in the slab volume (one or two slab copies).

```fkc
kernel: write_slice_rotating
op_kind: WriteSliceRotating
blurb: "In-place ring-buffer scatter: write source into a dest slab whose axis wraps mod modulus; dynamic start from a position operand."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::write_slice_rotating_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      dtypes: [F32, F64, BF16, F16, U32, U8, I8, I16, I32, I64]   # CPU byte-agnostic; CU/VK byte-width-keyed
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "dim[i] == ranges[i].end - ranges[i].start; dim[axis] <= modulus"
    - name: position
      dtypes: [U32]                   # rank-0/rank-1 dynamic write position (data-determined per token)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "0..=1"
      shape_constraint: "byte length >= 4 (one u32)"
    - name: dest
      dtypes: [F32, F64, BF16, F16, U32, U8, I8, I16, I32, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_as=out; read-modify-written in place (this operand IS the output)"
  op_params:
    variant: WriteSliceRotating       # OpParams::WriteSliceRotating (kernel.rs:585)
    fields:
      dest_shape: { kind: "Vec<usize>", constraint: "rank >= 1" }
      axis:       { kind: usize, constraint: "< rank; the rotating dim" }
      modulus:    { kind: usize, constraint: "0 < modulus <= dest_shape[axis]" }
      ranges:     { kind: "Vec<(usize,usize)>", constraint: "len == rank; on `axis` start ignored (dynamic), width <= modulus; off-axis end <= dest_shape[i]" }
      position:   { kind: DynScalar, note: "read from the `position` operand's first u32; data-determined write offset (rides the runtime, not a compile-time param)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(dest)
      shape_rule: same_as(dest)
      layout_guarantee: same_as(dest)
      aliasing: in_place(dest)               # output IS dest's buffer; partial (ring-slab) overwrite

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dim[0] == 0", class: free, note: "empty slab: no-op early return" }
    - { note: "position % modulus + slab_axis_width <= modulus: single (non-split) slab write" }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound in slab volume; up to 2 slab copies on a ring split
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * source.n * dtype_bytes"   # source.n = slab element count; ring split splits the same volume across two copies
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # in-place; no allocation

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy into a ring slab; bit-exact, hardware-independent. Bytes outside the written rows are preserved."

determinism: bitwise
```

---

## copy  (cross-device / same-device byte transfer)

Full-buffer byte transfer keyed on the **source** backend, routing on the output's substrate
(D2H / D2D / host memcpy).

`OpKind::Copy` (registration `dispatch.rs:4602` CPU memcpy, `dispatch.rs:4790` CUDA PTX path,
`vulkan_dispatch.rs:5033` VK) is the one op whose dispatch key is keyed on the **SOURCE** backend
and routes on the output's substrate variant: the baracuda/PTX wrapper does CPU output → **D2H**
(`to_cpu_bytes`) and CUDA output → **D2D** (`slot_copy_to_new`); the Vulkan wrapper does **D2H** to
a CPU output; the CPU wrapper is a CPU→CPU **memcpy** (effectively a no-op clone). It is the only
remaining op on the PTX CUDA path (every compute kernel migrated to baracuda). Full-buffer
**contiguous, byte-level** copy (`input_layouts` **C** on every backend); the output is pre-allocated
by the executor (`WorkItemKind::Copy`). Dtype-agnostic (whole-buffer byte transfer). Numerics: pure
byte transfer — bit-exact, deterministic across any hardware. Perf: bandwidth-bound (or
transfer-bus-bound for D2H/D2D); `N*dtype_bytes` read + written once.

```fkc
kernel: copy
op_kind: Copy
blurb: "Full-buffer byte transfer keyed on the source backend; D2H/D2D/host memcpy; dtype-agnostic; contiguous-only."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::copy_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      # CPU CPU->CPU: f32,f64,bf16,f16,u32,u8,i16,i32,i64. (CU source: same + ; VK source: every byte-substrate dtype.)
      dtypes: [F32, F64, BF16, F16, U32, U8, I16, I32, I64]
      # Full-buffer byte-level copy; contiguous-only on every backend. Keyed on the SOURCE backend;
      # routes on the output substrate (CPU output => D2H; CUDA output => D2D; CPU source => host memcpy).
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any                       # whole-buffer transfer; shape preserved verbatim
  op_params: { variant: None }        # no geometry params; the whole buffer is transferred

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)        # dtype unchanged; only the residency/substrate changes
      shape_rule: same_as(input)            # shape preserved verbatim
      layout_guarantee: contiguous          # dense row-major output (matches the source buffer)
      aliasing: none                        # output is a distinct (possibly cross-device) buffer

caps:
  awkward_layout_strategy: requires_contiguous   # whole-buffer copy needs a dense source; non-contig producer is contiguized first
  fast_paths:
    - { note: "CPU->CPU same-substrate: memcpy (clone); D2H/D2D dominated by bus bandwidth" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth/transfer-bus-bound; 2*N*dtype_bytes (read src + write dst); D2H/D2D pay bus latency+bandwidth
  class: strided_elementwise
  flops: "0"                          # pure transfer, no arithmetic
  bytes_moved: "2 * n * dtype_bytes"  # read source buffer + write destination buffer
  overhead_ns: ~                      # D2H/D2D launch + bus latency is judge-measured (varies by transfer path)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # destination is caller-preallocated

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte transfer (D2H / D2D / host memcpy); bit-exact, hardware-independent for every dtype."

determinism: bitwise
```

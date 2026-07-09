---
fkc_version: 1
provider:
  name: fuel-dispatch
  backend: Cpu                       # default backend; per-kernel siblings override to Cuda / Vulkan
  kernel_source: "portable-cpu"      # default BindingEntry.kernel_source; overridden per sibling
  link_registry: fuel_dispatch::ENTRY_POINTS    # §12.6 symbol→KernelRef map (the dispatch wrappers)
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-dispatch — elementwise-unary kernel contracts

The `fuel-dispatch` crate is the **binding/registration layer**, not a kernel-source layer: it wires
each elementwise-unary `OpKind` onto Fuel's `KernelBindingTable` for **every backend that ships an
implementation**, attaching the per-backend `KernelCaps`/`PrecisionGuarantee` and a thin `KernelRef`
wrapper that adapts the backend's kernel ABI. The kernel *bodies* live in the backend crates
(`fuel-cpu-backend` byte kernels, `baracuda` CUDA kernels via the `fuel-cuda-backend` wrappers,
`fuel-vulkan-backend` Slang). Each section below is **one OpKind**; within a section the contract
records every registered backend as a sibling alternative at the dispatch key (`§12.5`). Sources:
`fuel-dispatch/src/dispatch.rs` (CPU `register_cpu_kernels`), `fuel-dispatch/src/baracuda_dispatch.rs`
(`register_baracuda_cuda_kernels`), `fuel-dispatch/src/vulkan_dispatch.rs` (`register_vulkan_kernels`).

Cross-cutting facts for this whole family (verified against the inventory and the registration sites):

- **One input, one output; `out[i] = op(in[i])`; same dtype, same shape.** Output dtype = input dtype
  (`passthrough(in)`), output shape = input shape (`same_as(in)`), output is a fresh contiguous
  row-major buffer the executor pre-allocates (no kernel allocates; `kernel.rs:121-149`). No aliasing —
  the in-place unary family (`ReluInplace`, `NegInplace`, …) is a **separate dispatch surface**, not
  contracted here.
- **`op_params: OpParams::None`** for every op except the two that carry a scalar: `ClampElementwise`
  (`OpParams::Clamp { min: f64, max: f64 }`) and `PowIElementwise` / `PowIElementwiseBackward`
  (`OpParams::PowI { exp: i32 }`). Shape is positional.
- **Layout is per-backend, not per-op** (the core reason this is a `fuel-dispatch` bundle and not a
  single-backend one):
  - **CPU (`portable-cpu`): contiguous-only.** Every CPU wrapper takes `_layouts: &[Layout]` UNUSED and
    operates on flat `CpuStorageBytes`; geometry is positional. Default (all-false) `KernelCaps` ⇒
    `awkward_layout_strategy: requires_contiguous`. The executor's auto-Contiguize pass realizes any
    strided / broadcast / non-zero-offset / reversed input dense before the wrapper runs; the planner
    inserts `Op::Contiguize` (itself an FKC kernel) for a non-contiguous producer and sums its cost
    (§4.4). `reverse_strides: rejected`.
  - **CUDA (`baracuda`): strided + broadcast capable, NOT offset-capable.** Registered
    `register_with_caps(..., KernelCaps::strided_input())`; the baracuda FFI is stride-driven and the
    wrapper picks the contig vs `<sym>_strided_run` variant per call via `is_contiguous_zero_offset`.
    `awkward_layout_strategy: handles_strided`. Non-zero `start_offset` inputs STILL auto-Contiguize
    (offset-slicing the device buffer is a separate concern, `compiled.rs:58`), so `start_offset:
    rejected`. `reverse_strides: rejected` (no signed-stride walk advertised).
  - **Vulkan (`vulkan-slang`): strided + broadcast capable** for the ops it ships; `unary.slang`
    mirrors `binary.slang`'s per-dim decomposition + contig fast path and packs strides into Params.
    Same offset / reverse caveats as CUDA.
- **dtypes are per-backend** (faithfully transcribed per op):
  - CPU covers the full `{F32, F64, BF16, F16}` set for every op here.
  - CUDA covers `{F32, F64, BF16, F16}` for every op here.
  - Vulkan covers a **subset of ops** and, per op, `F32` (with `F16`/`F64` feature-gated where the
    inventory notes it) — Vulkan ships the pointwise + transcendental + Clamp + PowI ops but **not**
    the rounding/special family (Floor/Ceil/Round/Erf/GeluErf/Rsqrt) nor PowI-backward.
- **half precision (bf16/f16) widens to f32, computes, narrows on store** on CPU and matches the
  pre-chassis per-dtype kernels bit-for-bit; CUDA half is native-half where the baracuda kernel is.
- **Cost: bandwidth-bound elementwise.** Each op touches `n` input + `n` output elements, so the
  genuinely-derivable hint is `bytes_moved = 2 * n * dtype_bytes` (read in, write out). `flops` is
  op-dependent (one cheap arithmetic op vs a transcendental) and `overhead_ns` is launch-dependent;
  both are left to the Judge. **Cost is marked `judge_measured` (the Judge bootstraps it)** — no
  per-op timing numbers are fabricated; the bandwidth formula is the only declared coefficient and it
  is a derivable hint, not a measured value. Provenance `judge_measured` is a first-class, visible
  marker (§4.4), not a hidden gap. `memory.host_bytes = n * dtype_bytes` for the CPU output alloc;
  `device_bytes = n * dtype_bytes` for the GPU output alloc.
- **Precision is per-backend:**
  - CPU primitive kernels leave their precision block `audited: false` so the importer's
    `fill_unset_cpu_precision` pass applies `PRIMITIVE_DETERMINISTIC_CPU`; each states
    `bit_stable_on_same_hardware: true` (deterministic single-threaded positional loop, no atomic /
    reduction reordering) with op-specific numeric notes. This satisfies the always-built bit-stable
    coverage commitment (§4.8, §10.9).
  - CUDA pointwise/arithmetic ops are same-hardware bit-stable; transcendentals are same-hardware
    bit-stable but not cross-hardware (libm/PTX intrinsics differ).
  - Vulkan **pointwise** ops carry `VULKAN_FLOAT_POINTWISE_PRECISION`; **transcendental** ops carry
    `VULKAN_TRANSCENDENTAL_PRECISION` (GLSL.std.450, 3-4 ULP per the Vulkan spec). These map to a
    bounded `PrecisionGuarantee`; the exact `max_ulp` is the backend constant, left as a Judge-audited
    seed in the contract notes.
- **Determinism: `same_hardware_bitwise`** for CPU/CUDA pointwise; transcendentals are
  same-hardware-bitwise on a pinned library but not cross-hardware.

Because every entry is a sibling at one `(OpKind, [T, T], BackendId)` key, the route picker ranks the
backends by the cost vector + precision pre-filter at plan time; the contiguous-only CPU sibling pays
an inserted `Op::Contiguize` for a non-contiguous producer while the strided CUDA/Vulkan siblings do
not (§4.3, §4.4).

---

## relu  (rectified linear unit, `max(0, x)`)

Elementwise ReLU clamp `out[i] = max(0, in[i])`.

CPU (`relu_elementwise_{f32,f64,bf16,f16}_cpu_wrapper` → `fuel_cpu_backend::byte_kernels::relu_*`
→ `chassis::unary::Relu`): NaN-propagating (torch parity — `torch.relu(nan) == nan`), pinned
2026-07-08 (`docs/architecture/10-decisions-log.md`); contiguous-only. CUDA
(`baracuda::unary::relu_*`, strided) f32/f64/bf16/f16 — also NaN-propagating as of the baracuda
alpha.76 rebind (bound to the bespoke `unary_relu_propagating_*` family; the transitional
NaN-as-missing/`fmaxf` divergence noted through alpha.75 is closed — pinned by the direct
binding-table live tests
`fuel-dispatch/tests/cuda_dispatch_live.rs::cuda_relu_propagates_nan_{f32,bf16}`, which
supersede the `relu_cuda_still_scrubs_nan_pending_alpha76_rebind` lazy-realize pin, de-scoped to
`fuel-core/src/lazy.rs::relu_nan_convention_lazy_realize_smoke`). Vulkan (`unary::relu_f32`, strided,
`VULKAN_FLOAT_POINTWISE_PRECISION`) f32 (+f16/f64 gated) — NaN handling unaudited by this change,
left as previously documented. `ReluInplace` (CUDA) is NOT yet rebound — see the residual-gap
note in `fuel-cuda-backend/src/baracuda/elementwise.rs` next to `unary_inplace_relu_f32`. One
cheap branchless op per element; bandwidth-bound.

```fkc
kernel: relu
op_kind: ReluElementwise
blurb: "Elementwise ReLU max(0, x), NaN-propagating on CPU and CUDA (torch parity, alpha.76+); CPU contiguous, CUDA/Vulkan strided; half via f32; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::relu_elementwise_f32_cpu_wrapper"   # CPU sibling; CUDA baracuda::unary::relu_f32, VK unary::relu_f32 register at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CPU sibling; CUDA/Vulkan siblings declare strided: accepted, broadcast_stride0: accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CPU; CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "max(0, x); exact for f32/f64. bf16/f16 widen to f32 then narrow. CPU + CUDA: NaN-propagating (torch parity, pinned 2026-07-08 CPU / alpha.76 CUDA rebind); CUDA's ReluInplace still uses the NaN-scrubbing symbol (residual gap, not yet rebound). CPU bit-stable; Vulkan VULKAN_FLOAT_POINTWISE_PRECISION (NaN handling unaudited by this change)."

determinism: same_hardware_bitwise
```

---

## neg  (negation, `-x`)

Elementwise negation `out[i] = -in[i]`.

CPU (`neg_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`) exact for all dtypes (sign flip). CUDA
(`baracuda::unary::neg_*`, strided) f32/f64/bf16/f16. Vulkan (`unary::neg_f32`, strided,
`VULKAN_FLOAT_POINTWISE_PRECISION`) f32 (+f16/f64 gated). Bandwidth-bound.

```fkc
kernel: neg
op_kind: NegElementwise
blurb: "Elementwise negation -x; CPU contiguous, CUDA/Vulkan strided; exact all dtypes; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::neg_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::neg_f32, VK unary::neg_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "-x; exact (sign flip) for every dtype incl. bf16/f16. CPU bit-stable; Vulkan VULKAN_FLOAT_POINTWISE_PRECISION."

determinism: same_hardware_bitwise
```

---

## sqr  (square, `x * x`)

Elementwise square `out[i] = in[i] * in[i]`.

CPU (`sqr_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`): native multiply; bf16/f16 widen→multiply→narrow
(double rounding). CUDA (`baracuda::unary::sqr_*`, strided) f32/f64/bf16/f16. Vulkan (`unary::sqr_f32`,
strided, `VULKAN_FLOAT_POINTWISE_PRECISION`) f32 (+f16/f64 gated). Bandwidth-bound.

```fkc
kernel: sqr
op_kind: SqrElementwise
blurb: "Elementwise square x*x; CPU contiguous, CUDA/Vulkan strided; half via f32; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::sqr_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::sqr_f32, VK unary::sqr_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x*x; f32/f64 native. bf16/f16 widen to f32 then narrow (double rounding). CPU bit-stable; Vulkan VULKAN_FLOAT_POINTWISE_PRECISION."

determinism: same_hardware_bitwise
```

---

## sqrt  (square root)

Elementwise square root `out[i] = sqrt(in[i])`.

CPU (`sqrt_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`): IEEE-754 correctly-rounded f32/f64; negatives→NaN.
CUDA (`baracuda::unary::sqrt_*`, strided) f32/f64/bf16/f16. Vulkan (`unary::sqrt_f32`, strided,
`VULKAN_FLOAT_POINTWISE_PRECISION`) f32 (+f16/f64 gated). Bandwidth-bound.

```fkc
kernel: sqrt
op_kind: SqrtElementwise
blurb: "Elementwise square root; CPU contiguous, CUDA/Vulkan strided; negatives -> NaN; half via f32; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::sqrt_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::sqrt_f32, VK unary::sqrt_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "sqrt(x), IEEE-754 correctly rounded for f32/f64; negatives -> NaN. bf16/f16 via f32. CPU bit-stable; Vulkan VULKAN_FLOAT_POINTWISE_PRECISION."

determinism: same_hardware_bitwise
```

---

## recip  (reciprocal, `1 / x`)

Elementwise reciprocal `out[i] = 1 / in[i]` (`1/0 -> inf` per IEEE-754).

CPU (`recip_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`) native divide. CUDA
(`baracuda::unary::recip_*`, strided) f32/f64/bf16/f16. Vulkan (`unary::recip_f32`, strided,
`VULKAN_FLOAT_POINTWISE_PRECISION`) f32 (+f16/f64 gated). Bandwidth-bound.

```fkc
kernel: recip
op_kind: RecipElementwise
blurb: "Elementwise reciprocal 1/x; CPU contiguous, CUDA/Vulkan strided; IEEE inf/NaN; half via f32; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::recip_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::recip_f32, VK unary::recip_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "1/x; f32/f64 IEEE divide (1/0 -> inf). bf16/f16 via f32. CPU bit-stable; Vulkan VULKAN_FLOAT_POINTWISE_PRECISION."

determinism: same_hardware_bitwise
```

---

## abs  (absolute value, `|x|`)

Elementwise absolute value `out[i] = |in[i]|`.

CPU (`abs_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`) exact for all dtypes (sign clear). CUDA
(`baracuda::unary::abs_*`, strided) f32/f64/bf16/f16. Vulkan (`unary::abs_f32`, strided,
`VULKAN_FLOAT_POINTWISE_PRECISION`) f32 (+f16/f64 gated). Bandwidth-bound.

```fkc
kernel: abs
op_kind: AbsElementwise
blurb: "Elementwise absolute value |x|; CPU contiguous, CUDA/Vulkan strided; exact all dtypes; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::abs_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::abs_f32, VK unary::abs_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "|x|; exact (sign clear) for every dtype incl. bf16/f16. CPU bit-stable; Vulkan VULKAN_FLOAT_POINTWISE_PRECISION."

determinism: same_hardware_bitwise
```

---

## step  (Heaviside step, `1 where x > 0 else 0`)

Elementwise Heaviside step (the derivative of ReLU): `out[i] = if in[i] > 0 { 1 } else { 0 }`.

CPU (`step_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`) exact compare+select; `step(NaN)=0` (NaN
compares false). CUDA (`baracuda::unary::step_*`, strided, baracuda's native `unary_step_*`)
f32/f64/bf16/f16. Vulkan (`unary::step_f32`, strided, `VULKAN_FLOAT_POINTWISE_PRECISION`) f32 (+f16/f64
gated). Bandwidth-bound.

```fkc
kernel: step
op_kind: StepElementwise
blurb: "Elementwise Heaviside step 1 where x>0 else 0; CPU contiguous, CUDA/Vulkan strided; exact; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::step_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::step_f32, VK unary::step_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x>0 ? 1 : 0; exact compare+select; step(NaN)=0 (NaN compares false). bf16/f16 compared in f32. CPU bit-stable; Vulkan VULKAN_FLOAT_POINTWISE_PRECISION."

determinism: same_hardware_bitwise
```

---

## sign  (sign, `-1 / 0 / 1`, with `sign(0)=0`)

Elementwise sign: `out[i] = if in[i] > 0 { 1 } else if in[i] < 0 { -1 } else { 0 }`. `sign(0)=0`
(matches `torch.sign`); `sign(NaN)=0`.

CPU (`sign_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`) exact compare+select. CUDA
(`baracuda::unary::sign_*`, strided) — **f32 only in the as-built CUDA registration** (the bf16/f16
sign CUDA bindings are not present alongside the other unary half kernels). Vulkan (`unary::sign_f32`,
strided, `VULKAN_FLOAT_POINTWISE_PRECISION`) f32 (+f16/f64 gated). Bandwidth-bound.

```fkc
kernel: sign
op_kind: SignElementwise
blurb: "Elementwise sign -1/0/1 with sign(0)=0; CPU contiguous, CUDA/Vulkan strided; exact; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::sign_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::sign_f32 (f32), VK unary::sign_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x>0 ? 1 : x<0 ? -1 : 0; exact; sign(0)=0, sign(NaN)=0. bf16/f16 compared in f32. CUDA registers f32 only. CPU bit-stable; Vulkan VULKAN_FLOAT_POINTWISE_PRECISION."

determinism: same_hardware_bitwise
```

---

## tanh  (hyperbolic tangent)

Elementwise `out[i] = tanh(in[i])` — transcendental.

CPU (`tanh_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`) std/libm, not correctly-rounded. CUDA
(`baracuda::unary::tanh_*`, strided) f32/f64/bf16/f16. Vulkan (`unary::tanh_f32`, strided,
`VULKAN_TRANSCENDENTAL_PRECISION`, 3-4 ULP) f32 (+f16/f64 gated). Bandwidth-bound at scale.

```fkc
kernel: tanh
op_kind: TanhElementwise
blurb: "Elementwise tanh; CPU contiguous, CUDA/Vulkan strided; half via f32; not bit-stable cross-hardware; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::tanh_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::tanh_f32, VK unary::tanh_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "tanh via std/libm; not correctly-rounded; bf16/f16 via f32. Same-hardware bitwise but not cross-hardware (libm differs). Vulkan VULKAN_TRANSCENDENTAL_PRECISION (3-4 ULP)."

determinism: same_hardware_bitwise
```

---

## exp  (exponential, `e^x`)

Elementwise `out[i] = exp(in[i])`; overflow→+inf, large-negative→0.

CPU (`exp_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`) std/libm. CUDA (`baracuda::unary::exp_*`,
strided) f32/f64/bf16/f16. Vulkan (`unary::exp_f32`, strided, `VULKAN_TRANSCENDENTAL_PRECISION`) f32
(+f16/f64 gated). Transcendental, not correctly-rounded.

```fkc
kernel: exp
op_kind: ExpElementwise
blurb: "Elementwise exp e^x; CPU contiguous, CUDA/Vulkan strided; half via f32; not bit-stable cross-hardware; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::exp_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::exp_f32, VK unary::exp_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "e^x via std/libm; not correctly-rounded; bf16/f16 via f32. Same-hardware bitwise, not cross-hardware. Vulkan VULKAN_TRANSCENDENTAL_PRECISION (3-4 ULP)."

determinism: same_hardware_bitwise
```

---

## log  (natural logarithm, `ln(x)`)

Elementwise `out[i] = ln(in[i])`; negatives→NaN, `ln(0)=-inf`.

CPU (`log_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`) std/libm. CUDA (`baracuda::unary::log_*`,
strided) f32/f64/bf16/f16. Vulkan (`unary::log_f32`, strided, `VULKAN_TRANSCENDENTAL_PRECISION`) f32
(+f16/f64 gated). Transcendental, not correctly-rounded.

```fkc
kernel: log
op_kind: LogElementwise
blurb: "Elementwise natural log ln(x); CPU contiguous, CUDA/Vulkan strided; negatives -> NaN; half via f32; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::log_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::log_f32, VK unary::log_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "ln(x) via std/libm; negatives -> NaN, ln(0) -> -inf; not correctly-rounded; bf16/f16 via f32. Vulkan VULKAN_TRANSCENDENTAL_PRECISION (3-4 ULP)."

determinism: same_hardware_bitwise
```

---

## sin  (sine)

Elementwise `out[i] = sin(in[i])` — transcendental.

CPU (`sin_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`) std/libm. CUDA (`baracuda::unary::sin_*`,
strided) f32/f64/bf16/f16. Vulkan (`unary::sin_f32`, strided, `VULKAN_TRANSCENDENTAL_PRECISION`) f32
(+f16/f64 gated). Not correctly-rounded; range reduction per the library.

```fkc
kernel: sin
op_kind: SinElementwise
blurb: "Elementwise sine; CPU contiguous, CUDA/Vulkan strided; half via f32; not bit-stable cross-hardware; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::sin_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::sin_f32, VK unary::sin_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "sin(x) via std/libm; not correctly-rounded; bf16/f16 via f32. Same-hardware bitwise, not cross-hardware. Vulkan VULKAN_TRANSCENDENTAL_PRECISION (3-4 ULP)."

determinism: same_hardware_bitwise
```

---

## cos  (cosine)

Elementwise `out[i] = cos(in[i])` — transcendental.

CPU (`cos_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`) std/libm. CUDA (`baracuda::unary::cos_*`,
strided) f32/f64/bf16/f16. Vulkan (`unary::cos_f32`, strided, `VULKAN_TRANSCENDENTAL_PRECISION`) f32
(+f16/f64 gated). Not correctly-rounded.

```fkc
kernel: cos
op_kind: CosElementwise
blurb: "Elementwise cosine; CPU contiguous, CUDA/Vulkan strided; half via f32; not bit-stable cross-hardware; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::cos_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::cos_f32, VK unary::cos_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "cos(x) via std/libm; not correctly-rounded; bf16/f16 via f32. Same-hardware bitwise, not cross-hardware. Vulkan VULKAN_TRANSCENDENTAL_PRECISION (3-4 ULP)."

determinism: same_hardware_bitwise
```

---

## sigmoid  (logistic sigmoid, `1 / (1 + e^-x)`)

Elementwise `out[i] = 1 / (1 + exp(-in[i]))`.

CPU (`sigmoid_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`) std/libm exp. CUDA
(`baracuda::unary::sigmoid_*`, strided) f32/f64/bf16/f16. Vulkan (`unary::sigmoid_f32`, strided,
`VULKAN_TRANSCENDENTAL_PRECISION`) f32 (+f16/f64 gated). Transcendental-class, not correctly-rounded.

```fkc
kernel: sigmoid
op_kind: SigmoidElementwise
blurb: "Elementwise logistic sigmoid 1/(1+e^-x); CPU contiguous, CUDA/Vulkan strided; half via f32; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::sigmoid_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::sigmoid_f32, VK unary::sigmoid_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "1/(1+exp(-x)) via std/libm exp; not correctly-rounded; bf16/f16 via f32. Vulkan VULKAN_TRANSCENDENTAL_PRECISION (3-4 ULP)."

determinism: same_hardware_bitwise
```

---

## silu  (SiLU / Swish, `x * sigmoid(x)`)

Elementwise `out[i] = in[i] / (1 + exp(-in[i]))` (algebraically `x * sigmoid(x)`).

CPU (`silu_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`) std/libm exp. CUDA (`baracuda::unary::silu_*`,
strided) f32/f64/bf16/f16. Vulkan (`unary::silu_f32`, strided, `VULKAN_TRANSCENDENTAL_PRECISION`) f32
(+f16/f64 gated). Transcendental-class, not correctly-rounded.

```fkc
kernel: silu
op_kind: SiluElementwise
blurb: "Elementwise SiLU/Swish x*sigmoid(x); CPU contiguous, CUDA/Vulkan strided; half via f32; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::silu_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::silu_f32, VK unary::silu_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x/(1+exp(-x)) via std/libm exp; not correctly-rounded; bf16/f16 via f32. Vulkan VULKAN_TRANSCENDENTAL_PRECISION (3-4 ULP)."

determinism: same_hardware_bitwise
```

---

## gelu  (GELU, tanh approximation — the canonical `Gelu` op)

Elementwise GELU, **tanh approximation** (`OpKind::GeluElementwise` IS Fuel's default GELU; the exact
erf form is the separate `gelu_erf` below — the two must NOT be conflated under a Judge epsilon).
`out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.

CPU (`gelu_elementwise_{f32,f64,bf16,f16}_cpu_wrapper` → `fuel_cpu_backend::byte_kernels::gelu_*`):
sqrt(2/pi) 7-digit f32 / 16-digit f64; std/libm tanh. CUDA (`baracuda::unary::gelu_tanh_*`, strided)
f32/f64/bf16/f16 — note CUDA's plain `unary_gelu_*` is erf-flavored and registers under
`GeluErfElementwise` instead (conflation fixed in the 2026-06-10 sweep). Vulkan (`unary::gelu_f32`,
strided, `VULKAN_TRANSCENDENTAL_PRECISION`) f32 (+f16/f64 gated).

```fkc
kernel: gelu
op_kind: GeluElementwise
blurb: "Elementwise tanh-approx GELU (the canonical Gelu); CPU contiguous, CUDA/Vulkan strided; half via f32; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::gelu_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::gelu_tanh_f32, VK unary::gelu_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "TANH-approx GELU 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))); sqrt(2/pi) 7-digit f32 / 16-digit f64; DISTINCT from gelu_erf. CUDA binds gelu_tanh_*. bf16/f16 via f32. Vulkan VULKAN_TRANSCENDENTAL_PRECISION (3-4 ULP)."

determinism: same_hardware_bitwise
```

---

## floor  (floor, `⌊x⌋`)

Elementwise `out[i] = floor(in[i])` — exact (roundTowardNegative).

CPU (`floor_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`). CUDA (`baracuda::unary::floor_*`, strided)
f32/f64/bf16/f16. **No Vulkan binding** for this op (the rounding/special family is CPU+CUDA only).
Bandwidth-bound.

```fkc
kernel: floor
op_kind: FloorElementwise
blurb: "Elementwise floor ⌊x⌋; CPU contiguous, CUDA strided; exact; CPU+CUDA only (no Vulkan)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::floor_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::floor_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA sibling: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA sibling: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "floor(x); exact (roundTowardNegative). bf16/f16 via f32 (the f32 floor is exactly representable back in half). CPU + CUDA only."

determinism: same_hardware_bitwise
```

---

## ceil  (ceiling, `⌈x⌉`)

Elementwise `out[i] = ceil(in[i])` — exact (roundTowardPositive).

CPU (`ceil_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`). CUDA (`baracuda::unary::ceil_*`, strided)
f32/f64/bf16/f16. **No Vulkan binding.** Bandwidth-bound.

```fkc
kernel: ceil
op_kind: CeilElementwise
blurb: "Elementwise ceiling ⌈x⌉; CPU contiguous, CUDA strided; exact; CPU+CUDA only (no Vulkan)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::ceil_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::ceil_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA sibling: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA sibling: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "ceil(x); exact (roundTowardPositive). bf16/f16 via f32 (exactly representable back in half). CPU + CUDA only."

determinism: same_hardware_bitwise
```

---

## round  (round-half-to-even, banker's rounding)

Elementwise `out[i] = round_ties_even(in[i])` — IEEE-754 roundTiesToEven (NOT half-away-from-zero).

CPU (`round_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`, `round_ties_even`). CUDA
(`baracuda::unary::round_*`, strided, `rint`) f32/f64/bf16/f16 — both sides are banker's rounding.
**No Vulkan binding.** Bandwidth-bound.

```fkc
kernel: round
op_kind: RoundElementwise
blurb: "Elementwise round-half-to-even (banker's); CPU contiguous, CUDA strided; exact; CPU+CUDA only (no Vulkan)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::round_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::round_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA sibling: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA sibling: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "round_ties_even (banker's, roundTiesToEven, NOT half-away-from-zero); exact. CPU round_ties_even, CUDA rint (both banker's). bf16/f16 via f32. CPU + CUDA only."

determinism: same_hardware_bitwise
```

---

## erf  (Gauss error function, `erf(x)`)

Elementwise `out[i] = erf(in[i])` via libm `erff`(f32) / `erf`(f64).

CPU (`erf_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`). CUDA (`baracuda::unary::erf_*`, strided,
plain error function) f32/f64/bf16/f16. **No Vulkan binding.** libm-accurate, not correctly-rounded.

```fkc
kernel: erf
op_kind: ErfElementwise
blurb: "Elementwise Gauss error function erf(x); CPU contiguous, CUDA strided; half via f32; CPU+CUDA only (no Vulkan)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::erf_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::erf_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA sibling: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA sibling: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "erf via libm erff(f32)/erf(f64); not correctly-rounded; bf16/f16 via f32 (erff). Same-hardware bitwise (libm pinned), not cross-hardware. CPU + CUDA only."

determinism: same_hardware_bitwise
```

---

## gelu_erf  (GELU, exact erf form `0.5*x*(1+erf(x/√2))`)

Elementwise GELU, **exact error-function** formulation — DISTINCT from the tanh-approx `gelu` above.
`out[i] = 0.5 * x * (1 + erf(x * FRAC_1_SQRT_2))`.

CPU (`gelu_erf_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`) via libm erff/erf. CUDA
(`baracuda::unary::gelu_*`, strided — baracuda's plain `unary_gelu_*` is the erf flavor, bound here)
f32/f64/bf16/f16. **No Vulkan binding.** libm-accurate, not correctly-rounded.

```fkc
kernel: gelu_erf
op_kind: GeluErfElementwise
blurb: "Elementwise exact-erf GELU 0.5*x*(1+erf(x/sqrt2)); CPU contiguous, CUDA strided; half via f32; CPU+CUDA only (no Vulkan)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::gelu_erf_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::gelu_f32 (erf flavor) at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA sibling: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA sibling: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "EXACT erf GELU 0.5*x*(1+erf(x/sqrt2)) via libm; DISTINCT from gelu (tanh approx). CUDA binds plain unary_gelu_* (erf). bf16/f16 via f32. Same-hardware bitwise, not cross-hardware. CPU + CUDA only."

determinism: same_hardware_bitwise
```

---

## rsqrt  (reciprocal square root, `1 / sqrt(x)`)

Elementwise `out[i] = 1 / sqrt(in[i])` as a **single op** (not Sqrt then Recip — fusing loses
precision and a launch); critical for RMSNorm. negatives→NaN, `rsqrt(0)=+inf`.

CPU (`rsqrt_elementwise_{f32,f64,bf16,f16}_cpu_wrapper`): sqrt correctly-rounded then one divide. CUDA
(`baracuda::unary::rsqrt_*`, strided) f32/f64/bf16/f16. **No Vulkan binding.** Bandwidth-bound.

```fkc
kernel: rsqrt
op_kind: RsqrtElementwise
blurb: "Elementwise reciprocal sqrt 1/sqrt(x); single op; CPU contiguous, CUDA strided; half via f32; CPU+CUDA only (no Vulkan)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::rsqrt_elementwise_f32_cpu_wrapper"   # CUDA baracuda::unary::rsqrt_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA sibling: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA sibling: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "1/sqrt(x) as a single op (not Sqrt+Recip); negatives -> NaN, rsqrt(0) -> +inf. f32/f64 native; bf16/f16 via f32. CPU + CUDA only."

determinism: same_hardware_bitwise
```

---

## clamp  (elementwise clamp to scalar bounds)

Elementwise `out[i] = min(max(in[i], min_bound), max_bound)` with scalar bounds from
`OpParams::Clamp { min: f64, max: f64 }`. This is a unary op (one tensor input) plus two scalar params;
the bounds are NOT tensor operands. NaN handling follows the backend min/max convention.

CPU (`clamp_elementwise_f32_cpu_wrapper` (f32) + `clamp_{f64,bf16,f16}_cpu_wrapper`) f32/f64/bf16/f16.
CUDA (`baracuda::clamp` family, strided; bounds broadcast via stride-0) f32/f64/bf16/f16. Vulkan
(`clamp::clamp_f32`, strided, `VULKAN_FLOAT_POINTWISE_PRECISION`) f32 only. Bandwidth-bound.

```fkc
kernel: clamp
op_kind: ClampElementwise
blurb: "Elementwise clamp to scalar [min,max]; CPU contiguous, CUDA/Vulkan strided; half via f32; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::clamp_elementwise_f32_cpu_wrapper"   # CUDA baracuda::clamp_f32, VK clamp::clamp_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]                  # Vulkan sibling: F32 only
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Clamp                                   # OpParams::Clamp
    fields:
      min: { kind: f64 }
      max: { kind: f64, constraint: "max >= min" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "min(max(x, min), max); exact for f32/f64 (compare+select). bf16/f16 via f32. CUDA broadcasts scalar bounds via stride-0. Vulkan f32 only, VULKAN_FLOAT_POINTWISE_PRECISION."

determinism: same_hardware_bitwise
```

---

## powi  (elementwise integer power, `x^exp`)

Elementwise `out[i] = in[i] ^ exp` with integer exponent from `OpParams::PowI { exp: i32 }` (repeated
multiply / FMA). One tensor input + one scalar param.

CPU (`powi_elementwise_f32_cpu_wrapper` (f32) + `powi_{f64,bf16,f16}_cpu_wrapper`) f32/f64/bf16/f16.
CUDA (`baracuda::powi::powi_*`, strided) f32/f64/bf16/f16. Vulkan (`powi::powi_f32`, strided,
`VULKAN_FLOAT_POINTWISE_PRECISION`, repeated FMA bit-stable on same hardware) f32 only. Bandwidth-bound.

```fkc
kernel: powi
op_kind: PowIElementwise
blurb: "Elementwise integer power x^exp; CPU contiguous, CUDA/Vulkan strided; half via f32; multi-backend siblings."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::powi_elementwise_f32_cpu_wrapper"   # CUDA baracuda::powi::powi_f32, VK powi::powi_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]                  # Vulkan sibling: F32 only
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA/Vulkan siblings: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: PowI                                    # OpParams::PowI
    fields:
      exp: { kind: i32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA/Vulkan siblings: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                                          # ~|exp| multiplies per element (op-param-dependent magnitude; Judge refines)
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x^exp via repeated multiply/FMA; magnitude scales with |exp|. f32/f64 native; bf16/f16 via f32. Vulkan f32 only (repeated FMA, VULKAN_FLOAT_POINTWISE_PRECISION)."

determinism: same_hardware_bitwise
```

---

## powi_backward  (elementwise integer-power gradient)

Single-launch backward for `PowI`: given `(x, upstream)` produce `grad_x = upstream * exp * x^(exp-1)`
with the integer exponent from `OpParams::PowI { exp: i32 }`. This is a **two-input** op (`x`,
`upstream`) → one output (`grad_x`), all the same dtype — a single-launch alternative to autograd's
3-node decomposition (`PowI(n-1) → MulScalar → Mul`).

CPU (`powi_backward_{f32,f64,bf16,f16}_cpu_wrapper`) f32/f64/bf16/f16. CUDA
(`baracuda::powi_backward::powi_backward_*`, strided) f32/f64/bf16/f16. **No Vulkan binding.**
Bandwidth-bound (reads two inputs, writes one output).

```fkc
kernel: powi_backward
op_kind: PowIElementwiseBackward
blurb: "Single-launch PowI gradient grad_x = upstream*exp*x^(exp-1); CPU contiguous, CUDA strided; CPU+CUDA only (no Vulkan)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::powi_backward_f32_cpu_wrapper"   # CUDA baracuda::powi_backward::powi_backward_f32 at same key; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA sibling: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=upstream
    - name: upstream
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }   # CUDA sibling: strided + broadcast accepted
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: PowI                                    # OpParams::PowI { exp }
    fields:
      exp: { kind: i32 }

return:
  outputs:
    - name: grad_x
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # CUDA sibling: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                                          # ~|exp| multiplies + 1 mul per element (op-param-dependent; Judge refines)
  bytes_moved: "3 * n * dtype_bytes"                  # read x + upstream, write grad_x
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "grad_x = upstream * exp * x^(exp-1) in one launch (vs the 3-node decomposition). f32/f64 native; bf16/f16 via f32. CPU + CUDA only."

determinism: same_hardware_bitwise
```

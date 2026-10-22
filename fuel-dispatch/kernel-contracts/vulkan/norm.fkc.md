---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — normalization / softmax (production primitives) kernel contracts

The Vulkan backend's **last-dim normalization** primitives as PRODUCTION registers them (crate
`vulkan`, family `norm`): `OpKind::SoftmaxLastDim` + `OpKind::SoftmaxLastDimBackward`,
`OpKind::LayerNormLastDim` + `OpKind::LayerNormLastDimBackward`, and `OpKind::RmsNormLastDim`. Each is
a per-row reduction (max/sum for softmax; mean/variance for layer norm; sum-of-squares for RMS norm)
with a **subgroup-tree internal accumulation**.

**As-built binding model — production truth (DISTINCT per-(op, dtype) wrappers).** This production
contract authors ONE `op_kind` section per OpKind, fanned over `[F32, F16, BF16, F64]` to its DISTINCT
per-dtype wrapper (`softmax::softmax_*`, `softmax::softmax_last_dim_backward_*`, `norm::layer_norm_*`,
`norm::layer_norm_backward_*`, `norm::rms_*`), keying byte-for-byte the deleted hand-written
`register_with_precision(OpKind::{SoftmaxLastDim,…,RmsNormLastDim}, …)` regs:

- **Forward** (SoftmaxLastDim / LayerNormLastDim / RmsNormLastDim): 1 input ⇒ 2-slot `[T, T]`
  (`passthrough`).
- **Backward** (SoftmaxLastDimBackward / LayerNormLastDimBackward): 2 inputs ⇒ 3-slot `[T, T, T]`
  (`[y, g, dx]` / `[x, g, dx]`; both inputs share the fanned dtype, so they fan together, §3.4;
  `passthrough`).

Each section fans the BASE `entry_point`; the link registry resolves `<base>_<suffix>` to the
per-dtype wrapper.

**Layout model — contiguous-only (matches the as-built reg).** Every kernel issues one workgroup per
row and reads the row's elements flat; an arbitrary stride on the last-dim axis would break the
workgroup-shared-memory reduction. So the production registrations are plain `register_with_precision`
(no strided caps): `awkward_layout_strategy: requires_contiguous` (`strided_input == false`); the
planner auto-Contiguizes a broadcast / transpose / offset operand first (§4.3). Output is a fresh
contiguous buffer, no aliasing.

**Cost provenance.** Every cost block is `judge_measured` (§4.4). No overhead constant is fabricated;
the imported `unknown_cost` sentinel is upgraded to the shared OpKind cost fn by
`fill_unset_cost_for_backend`.

**Determinism.** Each kernel accumulates its per-row reduction in a subgroup / shared-memory tree
whose FADD order is scheduler-dependent (non-associative f32), so — matching the matmul / conv / value-
reduce siblings and §10 rule 9 — they are `determinism: nondeterministic` with
`bit_stable_on_same_hardware: false` and an audited `none(reason)` precision (byte-for-byte the deleted
regs' `PrecisionGuarantee::none(reason)`). f16/bf16 accumulate in f32 and narrow on store; f64 is
native.

---

## softmax_last_dim  (softmax along the last dim; f16/bf16/f64; contiguous)

Numerically-stable softmax along the last dim: per-row max, then exp, then sum, then divide (subgroup-
tree reduction internally). FOUR distinct per-dtype wrappers (`softmax::softmax_{f32,f16,bf16,f64}`);
this section fans the BASE `entry_point` over `[F16, BF16, F64]` (f32 is `softmax_last_dim_f32` below,
which carries a measured accuracy claim). Dispatch key `(SoftmaxLastDim, [T, T], Vulkan)`.

```fkc
kernel: softmax_last_dim
op_kind: SoftmaxLastDim
blurb: "Numerically-stable softmax along the last dim; f32/f16/bf16/f64; subgroup-tree reduction; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::softmax_last_dim"   # BASE symbol; fans softmax_last_dim_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "softmax over the last dim (per-row)"
  op_params:
    variant: SoftmaxLastDim       # OpParams::SoftmaxLastDim (primitive namespace; §3.7)
    fields: {}

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)       # key [T, T]
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "5 * n"                       # per element: sub-max, exp, add, div (+ the row reductions)
  bytes_moved: "2 * n * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false   # per-row subgroup-tree reduction; scheduler-dependent FADD order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                        # audited none(reason): no static bound (non-associative subgroup reduction)
  notes: "per-row max + exp + sum + divide (subgroup-tree reduction); f16/bf16 accumulate in f32; FADD order scheduler-dependent; NOT bit-stable."

determinism: nondeterministic
```

---

## softmax_last_dim_f32  (softmax along the last dim; f32 only; measured accuracy claim)

The f32 binding of the section above, split out so it can carry its own precision block: a fanned
section shares ONE precision block across its dtypes, and a claim measured for f32 is not true for
f16/bf16 (their output narrowing alone exceeds it). Same kernel, same dispatch key
`(SoftmaxLastDim, [F32, F32], Vulkan)`; the single-dtype section resolves its `entry_point` as-is.

```fkc
kernel: softmax_last_dim_f32
op_kind: SoftmaxLastDim
blurb: "Numerically-stable softmax along the last dim; f32/f16/bf16/f64; subgroup-tree reduction; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::softmax_last_dim_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "softmax over the last dim (per-row)"
  op_params:
    variant: SoftmaxLastDim       # OpParams::SoftmaxLastDim (primitive namespace; §3.7)
    fields: {}

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)       # key [T, T]
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "5 * n"                       # per element: sub-max, exp, add, div (+ the row reductions)
  bytes_moved: "2 * n * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false   # per-row subgroup-tree reduction; scheduler-dependent FADD order; repeatability is NOT claimed
  max_ulp: ~
  max_relative: 0.00001                 # MEASURED (board 240, option D); see notes
  max_absolute: 0.00001                 # MEASURED; see notes
  audited: true                         # accuracy claim backed by ledger records (seed_vulkan_matmul_accuracy)
  notes: "per-row max + exp + sum + divide (subgroup-tree reduction); FADD order scheduler-dependent; NOT bit-stable. [evidence: accuracy MEASURED, not derived. The registered kernel was driven through VulkanInvoker over 7 shapes (rows 1-28, cols 22-8192; attention-row widths of a Qwen3-0.6B decode plus wide rows), x uniform [-4,4), against an f64 host exp(x-max)/sum. Worst |out-ref| = 5.83e-8 (also the worst relative error, since softmax outputs are below the 1.0 floor); claim 1e-5 is ~170x headroom (ledger evidence, fuel-dev-tools seed-vulkan-accuracy, RTX 4070 Laptop). Not evidence about other inputs, other kernels in the family (f16/bf16/f64), or shapes beyond the listed ones. The same cases on the integrated AMD Radeon 610M adapter also stayed inside the claim (fuel-dev-tools check-vulkan-accuracy with FUEL_VULKAN_TEST_ADAPTER=AMD; not recorded in the ledger, which has no adapter field). Repeatability (bit_stable) is separately NOT claimed. The claim is what lets the default placement rule consider this kernel against the CPU kernel on cost.]"

determinism: nondeterministic
```

---

## softmax_last_dim_backward  (softmax backward along the last dim; f32/f16/bf16/f64; contiguous)

Backward of the last-dim softmax: 2 inputs `[y, g]` (forward output + upstream grad) → 1 output `dx`;
per-row `dot(y, g)` reduction + per-element `y * (g - dot)` (reuses `OpParams::SoftmaxLastDim`). FOUR
distinct per-dtype wrappers (`softmax::softmax_last_dim_backward_{f32,f16,bf16,f64}`); this section
fans the BASE `entry_point` over `[F32, F16, BF16, F64]` (both inputs share the list ⇒ fan together).
Dispatch key `(SoftmaxLastDimBackward, [T, T, T], Vulkan)` (`[y, g, dx]`).

```fkc
kernel: softmax_last_dim_backward
op_kind: SoftmaxLastDimBackward
blurb: "Softmax backward along the last dim (y, g -> dx); f32/f16/bf16/f64; subgroup-tree reduction; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::softmax_last_dim_backward"   # BASE symbol; fans <base>_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [F32, F16, BF16, F64]     # forward output; fans the per-dtype wrapper (§3.4)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "softmax forward output"
    - name: g
      dtypes: [F32, F16, BF16, F64]     # shares y's list ⇒ fans together
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "upstream gradient; same shape as y"
  op_params:
    variant: SoftmaxLastDim       # OpParams::SoftmaxLastDim reused for the backward
    fields: {}

return:
  outputs:
    - name: dx
      dtype_rule: passthrough(y)           # key [T, T, T]
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "4 * n"
  bytes_moved: "3 * n * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "per-row dot(y, g) reduction + per-element y*(g - dot) (subgroup-tree reduction); f16/bf16 accumulate in f32; FADD order scheduler-dependent; NOT bit-stable."

determinism: nondeterministic
```

---

## layer_norm_last_dim  (layer norm along the last dim; f32/f16/bf16/f64; contiguous)

Layer normalization along the last dim: per-row mean + variance (two reductions), then per-element
normalize. FOUR distinct per-dtype wrappers (`norm::layer_norm_{f32,f16,bf16,f64}`); this section fans
the BASE `entry_point` over `[F32, F16, BF16, F64]`. Dispatch key `(LayerNormLastDim, [T, T], Vulkan)`.

```fkc
kernel: layer_norm_last_dim
op_kind: LayerNormLastDim
blurb: "Layer norm along the last dim (per-row mean + variance, then normalize); f32/f16/bf16/f64; subgroup-tree reduction; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::layer_norm_last_dim"   # BASE symbol; fans <base>_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "layer norm over the last dim (per-row)"
  op_params:
    variant: NormLastDim          # OpParams::NormLastDim (primitive namespace; §3.7)
    fields:
      eps: { kind: f64, note: "numerical-stability epsilon" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)       # key [T, T]
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "6 * n"
  bytes_moved: "2 * n * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "per-row mean + variance reductions (subgroup-tree) then normalize; f16/bf16 accumulate in f32; FADD order scheduler-dependent; NOT bit-stable."

determinism: nondeterministic
```

---

## layer_norm_last_dim_backward  (layer norm backward along the last dim; f32/f16/bf16/f64; contiguous)

Backward of the last-dim layer norm: 2 inputs `[x, g]` → 1 output `dx`; per-row reductions of the
gradient contributions. FOUR distinct per-dtype wrappers
(`norm::layer_norm_backward_{f32,f16,bf16,f64}`); this section fans the BASE `entry_point` over
`[F32, F16, BF16, F64]` (both inputs share the list ⇒ fan together). Dispatch key
`(LayerNormLastDimBackward, [T, T, T], Vulkan)` (`[x, g, dx]`).

```fkc
kernel: layer_norm_last_dim_backward
op_kind: LayerNormLastDimBackward
blurb: "Layer norm backward along the last dim (x, g -> dx); f32/f16/bf16/f64; subgroup-tree reduction; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::layer_norm_last_dim_backward"   # BASE symbol; fans <base>_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F16, BF16, F64]     # forward input; fans the per-dtype wrapper (§3.4)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "layer norm forward input"
    - name: g
      dtypes: [F32, F16, BF16, F64]     # shares x's list ⇒ fans together
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "upstream gradient; same shape as x"
  op_params:
    variant: NormLastDim          # OpParams::NormLastDim reused for the backward
    fields:
      eps: { kind: f64, note: "numerical-stability epsilon" }

return:
  outputs:
    - name: dx
      dtype_rule: passthrough(x)           # key [T, T, T]
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "8 * n"
  bytes_moved: "3 * n * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "per-row gradient-contribution reductions (subgroup-tree) then per-element dx; f16/bf16 accumulate in f32; FADD order scheduler-dependent; NOT bit-stable."

determinism: nondeterministic
```

---

## rms_norm_last_dim  (RMS norm along the last dim; f16/bf16/f64; contiguous)

Root-mean-square normalization along the last dim: per-row `x²` sum (subgroup-tree reduction), then
`rsqrt` and divide. FOUR distinct per-dtype wrappers (`norm::rms_{f32,f16,bf16,f64}`); this section
fans the BASE `entry_point` over `[F16, BF16, F64]` (f32 is `rms_norm_last_dim_f32` below, which carries
a measured accuracy claim). Dispatch key `(RmsNormLastDim, [T, T], Vulkan)`.

```fkc
kernel: rms_norm_last_dim
op_kind: RmsNormLastDim
blurb: "RMS norm along the last dim (per-row x^2 sum, rsqrt, divide); f32/f16/bf16/f64; subgroup-tree reduction; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::rms_norm_last_dim"   # BASE symbol; fans <base>_<suffix>
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "RMS norm over the last dim (per-row)"
  op_params:
    variant: NormLastDim          # OpParams::NormLastDim (primitive namespace; §3.7)
    fields:
      eps: { kind: f64, note: "numerical-stability epsilon" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)       # key [T, T]
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "4 * n"
  bytes_moved: "2 * n * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "per-row x^2 sum (subgroup-tree reduction) + rsqrt + divide; f16/bf16 accumulate in f32, f64 native; FADD order scheduler-dependent; NOT bit-stable."

determinism: nondeterministic
```

---

## rms_norm_last_dim_f32  (RMS norm along the last dim; f32 only; measured accuracy claim)

The f32 binding of the section above, split out so it can carry its own precision block (see
`softmax_last_dim_f32` for why). Dispatch key `(RmsNormLastDim, [F32, F32], Vulkan)`.

```fkc
kernel: rms_norm_last_dim_f32
op_kind: RmsNormLastDim
blurb: "RMS norm along the last dim (per-row x^2 sum, rsqrt, divide); f32/f16/bf16/f64; subgroup-tree reduction; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::rms_norm_last_dim_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "RMS norm over the last dim (per-row)"
  op_params:
    variant: NormLastDim          # OpParams::NormLastDim (primitive namespace; §3.7)
    fields:
      eps: { kind: f64, note: "numerical-stability epsilon" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)       # key [T, T]
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "4 * n"
  bytes_moved: "2 * n * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false   # per-row subgroup-tree sum of squares; scheduler-dependent FADD order; repeatability is NOT claimed
  max_ulp: ~
  max_relative: 0.00001                 # MEASURED (board 240, option D); see notes
  max_absolute: 0.00001                 # MEASURED; see notes
  audited: true                         # accuracy claim backed by ledger records (seed_vulkan_matmul_accuracy)
  notes: "native f32; no affine; subgroup sum-of-squares + rsqrt; FADD order scheduler-dependent; NOT bit-stable. [evidence: accuracy MEASURED, not derived. The registered kernel was driven through VulkanInvoker over 9 shapes (rows 1-448, cols 64-4096; Qwen3-0.6B decode and prefill widths), x uniform [-2,2), eps 1e-5, against an f64 host x/sqrt(mean(x^2)+eps). Worst |out-ref| = 2.29e-7, worst |out-ref|/max(|ref|,1.0) = 1.52e-7; claim 1e-5 is ~44x headroom (ledger evidence, fuel-dev-tools seed-vulkan-accuracy, RTX 4070 Laptop). Not evidence about other inputs, other kernels in the family (f16/bf16/f64), or shapes beyond the listed ones. The same cases on the integrated AMD Radeon 610M adapter also stayed inside the claim (fuel-dev-tools check-vulkan-accuracy with FUEL_VULKAN_TEST_ADAPTER=AMD; not recorded in the ledger, which has no adapter field). Repeatability (bit_stable) is separately NOT claimed. The claim is what lets the default placement rule consider this kernel against the CPU kernel on cost.]"

determinism: nondeterministic
```

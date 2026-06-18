---
fkc_version: 1
provider:
  name: fuel-dispatch
  backend: Cpu                                   # default; per-kernel/per-block overrides to Cuda / Vulkan
  kernel_source: "portable-cpu"                  # default; overridden to "baracuda" / "vulkan-slang" per backend block
  link_registry: fuel_dispatch::fkc::ENTRY_POINTS   # §12.6 symbol → KernelRef map
  revision_base: "git:f41137b4"                  # provider build id, folded into kernel_revision_hash (§4.7)
---

# fuel-dispatch — reduce family kernel contracts

The **dispatch-layer** reduction kernels Fuel registers across its built-in backends: the per-axis
`{Sum,Max,Min,Mean}Reduce` set, the broadcast-target `ReduceSumTo` / `ReduceMaxTo` reductions and
the `ReduceMaxToBackward` gradient, and the `ArgMaxDim` / `ArgMinDim` index reductions. Inventory
source: `docs/kernel-contracts/_inventory/dispatch.md` (crate `dispatch`, family `reduce`) — the
`KernelBindingTable` registrations in `fuel-dispatch/src/{dispatch.rs, baracuda_dispatch.rs,
vulkan_dispatch.rs}`.

This is the **dispatch crate's** view, so it is multi-backend: a single `OpKind` is registered by
up to three different backends, each at its own `(OpKind, KernelDTypes, BackendId)` key and each a
distinct `BindingEntry` sibling that the route picker ranks (§12.1, §12.5). Each `## ` section
below covers exactly **one named kernel (OpKind)** and carries one ` ```fkc ` block **per backend
that materially differs** in admissibility-affecting facts (layout capability + precision). Where
backends agree on those facts, one block stands for the family and the prose lists the dtype/source
siblings; per-(op, dtype) entry-point fan-out is documented in prose rather than duplicating an
otherwise-identical block per dtype.

Cross-cutting facts for this family (from the inventory's legend + "Cross-cutting contract facts"):

- **Layout caps are the as-built `KernelCaps`, not a guess.** `C` (contiguous-only) =
  `register*` with default all-false caps → `awkward_layout_strategy: requires_contiguous`; `S`
  (strided) = `register_with_caps(..., KernelCaps::strided_input())` →
  `awkward_layout_strategy: handles_strided` with `strided: accepted` + `broadcast_stride0:
  accepted` on the input. **No reduce kernel in this crate is offset-capable** — even the
  strided-capable baracuda kernels send a non-zero-`start_offset` input through auto-Contiguize
  (`compiled.rs` caps gate; `KernelCaps` doc, `kernel.rs:66-74`), so every operand declares
  `start_offset: rejected`. **No reduce kernel walks negative strides**, so `reverse_strides:
  rejected` everywhere.
- **Output: pre-allocated, fully overwritten, contiguous.** Output Storage is always
  caller/executor-allocated; no kernel allocates. The wrapper overwrites the pre-allocated bytes
  (no read of prior output content); no input/output aliasing. Output dtype is the last entry of
  the binding key.
- **Shape arrives via the side-channel or `OpParams`, never invented.** For the per-axis reductions
  the input shape flows through the `KernelRef` `layouts[0]` side-channel and `OpParams::Reduce {
  dims, keepdim }` carries the reduce axes; for the broadcast-target reductions the
  `input_shape` / `output_shape` ride `OpParams::ReduceSumTo` / `ReduceMaxTo` /
  `ReduceMaxToBackward`. `keepdim` is `false` for every call fuel-graph emits today (the field is
  reserved); the output therefore has the reduced dims **removed**.
- **Half-float accumulator-promotion invariant.** On every backend `f32`/`f64` accumulate in their
  own dtype; `bf16`/`f16` accumulate in **f32** and narrow on store. (CPU encodes this in
  `ReduceOp::Acc`; baracuda promotes in the FFI kernel; Vulkan widens in the shader.)
- **Precision differs by backend (this is admissibility-affecting, so it is per-block).** CPU
  reductions are deterministic on the same hardware (fixed fold order) → the importer applies
  `PRIMITIVE_DETERMINISTIC_CPU` (§12.4). **Vulkan reductions/argreduce carry
  `PrecisionGuarantee::none`** (subgroup/tree reduction; FADD/accumulation order is
  scheduler-determined per dispatch — `vulkan_dispatch.rs:4455-4513`), so their blocks declare
  `audited: true` + `determinism: nondeterministic`. The baracuda CUDA reductions are registered
  without an explicit precision (family bulk-fill applies); the integer index reductions
  (Arg*Dim) are exact on every backend (no FP accumulation in the result).
- **Cost provenance: every cost block is `judge_measured`** (the Judge bootstraps the coefficients,
  §4.4). The `flops` / `bytes_moved` formula **hints** are retained because a single-pass reduction
  is genuinely reduction/bandwidth-bound and the structure is derivable from the op (one fold per
  input element; read all input + write all output). `overhead_ns` and per-tier `memory` are
  Judge-bootstrapped, **not fabricated** (no constant invented).

---

## sum_reduce_cpu  (SumReduce — per-axis sum reduction, CPU)

Sum a tensor over `OpParams::Reduce { dims }`, removing the reduced dims. One fold per input
element; `bf16`/`f16` accumulate in f32 and narrow on store. One section per backend; registered by
three backends:

- **CPU** (`dispatch.rs:3956`): `sum_reduce_{f32,f64,bf16,f16}_cpu_wrapper`, default caps →
  **contiguous-only**, bit-stable (fixed left-to-right fold). `kernel_source: "portable-cpu"`.
- **CU / baracuda** (`baracuda_dispatch.rs:2416`): `reduce::sum_{f32,f16,bf16}` (plus the f64
  registration at `dispatch.rs`-equivalent), `KernelCaps::strided_input()` → **strided**
  (the FFI passes `current_layout.stride()` each iteration). `kernel_source: "baracuda"`.
- **VK** (`vulkan_dispatch.rs:4457`): `reduce::sum_f32` (+ f16/bf16/f64 feature-gated),
  default caps → **contiguous-only**, `PrecisionGuarantee::none` (subgroup tree reduction,
  scheduler-dependent order). `kernel_source: "vulkan-slang"`.

The CPU and Vulkan blocks differ only in layout caps (both `requires_contiguous`) versus precision;
the baracuda block differs in layout caps (`handles_strided`). All three are sibling registrations
at distinct `(SumReduce, [T,T], backend)` keys.

```fkc
kernel: sum_reduce_cpu
op_kind: SumReduce
blurb: "Per-axis sum reduction (CPU); contiguous row-major; half via f32 accumulator; bit-stable."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sum_reduce_f32"   # representative; f64/bf16/f16 siblings per dtype
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce          # OpParams::Reduce (primitive namespace; §3.7)
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output drops reduced dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)   # input with reduce_dims removed (keepdim=false)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured     # Judge bootstraps the coefficients (§4.4)
  class: reduction
  flops: "n_in"                  # one fold per input element (single pass); derivable
  bytes_moved: "(n_in + n_out) * dtype_bytes"   # read all input, write all output; reduction/bandwidth-bound
  # overhead_ns + per-tier memory: Judge-bootstrapped (not fabricated)

precision:
  bit_stable_on_same_hardware: true   # fixed left-to-right fold order; deterministic
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                # CPU primitive: importer applies PRIMITIVE_DETERMINISTIC_CPU (§12.4)
  notes: "f32/f64 accumulate native; bf16/f16 accumulate in f32 then narrow on store; fixed fold order."

determinism: same_hardware_bitwise
```

---

## sum_reduce_cuda  (SumReduce — per-axis sum reduction, CUDA/baracuda)

CUDA (baracuda) registration of `SumReduce` — stride-driven; `bf16`/`f16` accumulate in f32.

```fkc
kernel: sum_reduce_cuda
op_kind: SumReduce
blurb: "Per-axis sum reduction (CUDA/baracuda); stride-driven; half via f32 accumulator."
backend: Cuda
kernel_source: "baracuda"
entry_point: "baracuda::reduce::sum_f32"   # representative; f64/bf16/f16 siblings per dtype
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output drops reduced dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided       # baracuda FFI walks current_layout.stride(); no contiguize for strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "any_input_strided", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_in"
  bytes_moved: "(n_in + n_out) * dtype_bytes"
  # overhead_ns (launch) + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: false   # baracuda reduction accumulation order not guaranteed bit-stable
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                 # PrecisionGuarantee::none(reason) — nondeterministic FP fold; no static bound applies (§4.9 / §10.9)
  notes: "f32/f64 native; bf16/f16 accumulate in f32 (FFI promotion) then narrow on store; stride-driven. baracuda reduction accumulation order is not bit-stable; advertised none(reason) per the family precision bulk-fill (§12.4)."

determinism: nondeterministic
```

---

## sum_reduce_vulkan  (SumReduce — per-axis sum reduction, Vulkan)

Vulkan registration of `SumReduce` — contiguous; subgroup tree reduction, `PrecisionGuarantee::none`.

```fkc
kernel: sum_reduce_vulkan
op_kind: SumReduce
blurb: "Per-axis sum reduction (Vulkan); contiguous row-major; subgroup tree reduction; not bit-stable."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::reduce::sum_f32"   # f16/bf16/f64 feature-gated siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]            # f16/bf16/f64 feature-gated
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output drops reduced dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
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
  flops: "n_in"
  bytes_moved: "(n_in + n_out) * dtype_bytes"
  # overhead_ns (command-buffer submit) + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: false   # subgroup tree reduction: accumulation order scheduler-determined
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                 # PrecisionGuarantee::none(reason) — audited, no static bound applies
  notes: "subgroup tree reduction; FADD accumulation order is scheduler-determined per dispatch; not bit-stable (vulkan_dispatch.rs:4457 SUM_REASON)."

determinism: nondeterministic
```

---

## max_reduce_cpu  (MaxReduce — per-axis max reduction, CPU)

Max a tensor over `OpParams::Reduce { dims }`, removing the reduced dims. One section per backend.
Extremum fold inits to
`-inf`; `f32::max` semantics (**NaN-as-missing**); `bf16`/`f16` run the extremum in f32 space and
narrow on store (the kept value is a representable half, so the round-trip is exact). Same backend
fan-out as `SumReduce`: CPU `max_reduce_{f32,f64,bf16,f16}_cpu_wrapper` (`dispatch.rs:3957`,
contiguous, bit-stable); baracuda `reduce::max_{f32,f16,bf16}` (+f64) (`baracuda_dispatch.rs:2417`,
strided); Vulkan `reduce::max_f32` (+gated) (`vulkan_dispatch.rs:4458`, contiguous,
`PrecisionGuarantee::none`).

```fkc
kernel: max_reduce_cpu
op_kind: MaxReduce
blurb: "Per-axis max reduction (CPU); f32::max NaN-as-missing; -inf init; contiguous; bit-stable."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::max_reduce_f32"   # f64/bf16/f16 siblings per dtype
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output drops reduced dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_in"                  # one compare per input element (single pass)
  bytes_moved: "(n_in + n_out) * dtype_bytes"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true   # extremum exact (no rounding); fixed fold order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "extremum via f32::max (NaN-as-missing); -inf init; bf16/f16 extremum in f32 space, narrow on store (kept value is representable). Exact; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## max_reduce_cuda  (MaxReduce — per-axis max reduction, CUDA/baracuda)

CUDA (baracuda) registration of `MaxReduce` — stride-driven; NaN-as-missing extremum.

```fkc
kernel: max_reduce_cuda
op_kind: MaxReduce
blurb: "Per-axis max reduction (CUDA/baracuda); stride-driven; NaN-as-missing extremum."
backend: Cuda
kernel_source: "baracuda"
entry_point: "baracuda::reduce::max_f32"   # f64/bf16/f16 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output drops reduced dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "any_input_strided", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_in"
  bytes_moved: "(n_in + n_out) * dtype_bytes"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true   # extremum (max) is order-independent / exact
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "extremum exact (no FP accumulation); NaN-as-missing; stride-driven; bf16/f16 extremum in f32 space."

determinism: same_hardware_bitwise
```

---

## max_reduce_vulkan  (MaxReduce — per-axis max reduction, Vulkan)

Vulkan registration of `MaxReduce` — contiguous; subgroup tree reduction, `PrecisionGuarantee::none`.

```fkc
kernel: max_reduce_vulkan
op_kind: MaxReduce
blurb: "Per-axis max reduction (Vulkan); contiguous; subgroup tree reduction; not bit-stable."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::reduce::max_f32"   # f16/bf16/f64 feature-gated
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output drops reduced dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
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
  flops: "n_in"
  bytes_moved: "(n_in + n_out) * dtype_bytes"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: false   # subgroup tree reduction; same scheduler-dependence as SumReduce
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                 # PrecisionGuarantee::none(reason)
  notes: "subgroup tree reduction (val, idx) over scheduler-determined order; not bit-stable (vulkan_dispatch.rs:4458 MAX_REASON). Result max is value-deterministic but advertised none per the registration."

determinism: nondeterministic
```

---

## min_reduce_cpu  (MinReduce — per-axis min reduction, CPU)

Mirror of `MaxReduce`: extremum inits to `+inf`, `f32::min` (NaN-as-missing). One section per
backend. Same backend
fan-out and per-block facts as `MaxReduce`: CPU `min_reduce_{f32,f64,bf16,f16}_cpu_wrapper`
(`dispatch.rs:3958`, contiguous, bit-stable); baracuda `reduce::min_{f32,f16,bf16}` (+f64)
(`baracuda_dispatch.rs:2418`, strided); Vulkan `reduce::min_f32` (+gated)
(`vulkan_dispatch.rs:4459`, contiguous, `PrecisionGuarantee::none`).

```fkc
kernel: min_reduce_cpu
op_kind: MinReduce
blurb: "Per-axis min reduction (CPU); f32::min NaN-as-missing; +inf init; contiguous; bit-stable."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::min_reduce_f32"   # f64/bf16/f16 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output drops reduced dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_in"
  bytes_moved: "(n_in + n_out) * dtype_bytes"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "extremum via f32::min (NaN-as-missing); +inf init; bf16/f16 extremum in f32 space, narrow on store. Exact; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## min_reduce_cuda  (MinReduce — per-axis min reduction, CUDA/baracuda)

CUDA (baracuda) registration of `MinReduce` — stride-driven; NaN-as-missing extremum.

```fkc
kernel: min_reduce_cuda
op_kind: MinReduce
blurb: "Per-axis min reduction (CUDA/baracuda); stride-driven; NaN-as-missing extremum."
backend: Cuda
kernel_source: "baracuda"
entry_point: "baracuda::reduce::min_f32"   # f64/bf16/f16 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output drops reduced dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "any_input_strided", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_in"
  bytes_moved: "(n_in + n_out) * dtype_bytes"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "extremum exact (no FP accumulation); NaN-as-missing; stride-driven; bf16/f16 extremum in f32 space."

determinism: same_hardware_bitwise
```

---

## min_reduce_vulkan  (MinReduce — per-axis min reduction, Vulkan)

Vulkan registration of `MinReduce` — contiguous; subgroup tree reduction, `PrecisionGuarantee::none`.

```fkc
kernel: min_reduce_vulkan
op_kind: MinReduce
blurb: "Per-axis min reduction (Vulkan); contiguous; subgroup tree reduction; not bit-stable."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::reduce::min_f32"   # f16/bf16/f64 feature-gated
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output drops reduced dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
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
  flops: "n_in"
  bytes_moved: "(n_in + n_out) * dtype_bytes"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                 # PrecisionGuarantee::none(reason)
  notes: "subgroup tree reduction; same scheduler-dependence as MaxReduce; not bit-stable (vulkan_dispatch.rs:4459 MIN_REASON)."

determinism: nondeterministic
```

---

## mean_reduce_cpu  (MeanReduce — per-axis mean reduction, CPU)

One section per backend. Sum a tensor over `OpParams::Reduce { dims }`, then divide each output slot by `count` (= product
of reduced-dim sizes). Sum uses the same accumulator as `SumReduce` (f32 for half); finalize is a
single divide per output slot, so `flops = n_in + n_out`. CPU rejects `count == 0` (divisor zero →
typed `Error`, not silent NaN). Same backend fan-out as `SumReduce`: CPU
`mean_reduce_{f32,f64,bf16,f16}_cpu_wrapper` (`dispatch.rs:3959`, contiguous, bit-stable);
baracuda `reduce::mean_{f32,f16,bf16}` (+f64) (`baracuda_dispatch.rs:2419`, strided); Vulkan
`reduce::mean_f32` (+gated) (`vulkan_dispatch.rs:4460`, contiguous, `PrecisionGuarantee::none`).

```fkc
kernel: mean_reduce_cpu
op_kind: MeanReduce
blurb: "Per-axis mean reduction (CPU); f32 accumulator; divide by count; rejects count==0; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::mean_reduce_f32"   # f64/bf16/f16 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank; product(reduced dims) > 0 (count != 0)"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output drops reduced dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_in + n_out"          # n_in folds + one divide per output slot
  bytes_moved: "(n_in + n_out) * dtype_bytes"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32/f64 accumulate native; bf16/f16 accumulate in f32 then divide by count and narrow on store; rejects count==0; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## mean_reduce_cuda  (MeanReduce — per-axis mean reduction, CUDA/baracuda)

CUDA (baracuda) registration of `MeanReduce` — stride-driven; divide by count; half via f32 accumulator.

```fkc
kernel: mean_reduce_cuda
op_kind: MeanReduce
blurb: "Per-axis mean reduction (CUDA/baracuda); stride-driven; divide by count; half via f32 accumulator."
backend: Cuda
kernel_source: "baracuda"
entry_point: "baracuda::reduce::mean_f32"   # f64/bf16/f16 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank; product(reduced dims) > 0"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output drops reduced dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "any_input_strided", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_in + n_out"
  bytes_moved: "(n_in + n_out) * dtype_bytes"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                 # PrecisionGuarantee::none(reason) — nondeterministic FP fold; no static bound applies (§4.9 / §10.9)
  notes: "f32/f64 native; bf16/f16 accumulate in f32 then divide by count, narrow on store; stride-driven. baracuda reduction accumulation order is not bit-stable; advertised none(reason) per the family precision bulk-fill (§12.4)."

determinism: nondeterministic
```

---

## mean_reduce_vulkan  (MeanReduce — per-axis mean reduction, Vulkan)

Vulkan registration of `MeanReduce` — contiguous; subgroup tree reduction + scalar divide, `PrecisionGuarantee::none`.

```fkc
kernel: mean_reduce_vulkan
op_kind: MeanReduce
blurb: "Per-axis mean reduction (Vulkan); contiguous; subgroup tree reduction + scalar divide; not bit-stable."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::reduce::mean_f32"   # f16/bf16/f64 feature-gated
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank; product(reduced dims) > 0"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output drops reduced dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, dims, keepdim)
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
  flops: "n_in + n_out"
  bytes_moved: "(n_in + n_out) * dtype_bytes"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                 # PrecisionGuarantee::none(reason)
  notes: "subgroup tree reduction + scalar division; accumulation order is scheduler-determined; not bit-stable (vulkan_dispatch.rs:4460 MEAN_REASON)."

determinism: nondeterministic
```

---

## reduce_sum_to_cpu  (ReduceSumTo — broadcast-target sum reduction, CPU)

One section per backend. The backward of a forward broadcast: fold a tensor down to a smaller, broadcast-compatible
`output_shape` (the `grad` of a broadcast in autograd). `output_shape` is left-padded with 1s to
`input_shape`'s rank; per padded axis it must equal the input size (axis carries through) or `1`
(axis summed away). One fold per input element + one finalize (identity) per output slot.
Registered by two backends:

- **CPU** (`dispatch.rs:4032`): `reduce_sum_to_{f32,f64,bf16,f16}_cpu_wrapper`, default caps →
  **contiguous-only**, bit-stable. `kernel_source: "portable-cpu"`.
- **CU / baracuda** (`baracuda_dispatch.rs:2900`): stride-driven on the input
  (`KernelCaps::strided_input()`) → **strided** (transposed-view grads skip Contiguize).
  `kernel_source: "baracuda"`.

No Vulkan registration exists for this op. The shape/param carrier is `OpParams::ReduceSumTo {
input_shape, output_shape }`.

```fkc
kernel: reduce_sum_to_cpu
op_kind: ReduceSumTo
blurb: "Sum-reduce to a broadcast-compatible target shape (CPU); half via f32 accumulator; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_sum_to_f32"   # f64/bf16/f16 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(input, output_shape)"   # each padded output axis == input size OR 1
  op_params:
    variant: ReduceSumTo          # OpParams::ReduceSumTo (primitive namespace; §3.7)
    fields:
      input_shape:  { kind: "Vec<usize>", constraint: "product == input.elem_count" }
      output_shape: { kind: "Vec<usize>", constraint: "left-pad-1 to input rank; each axis == input size OR 1" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, output_shape, keepdim)   # = output_shape; symbolic axes preserved where shape-preserving
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "in_elems"              # in_elems = product(input_shape); one fold per input element
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # read every input + write every output slot
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true   # deterministic single-pass fold in a fixed order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32/f64 native accumulator; bf16/f16 accumulate in f32 then narrow on store; IEEE add; fixed fold order."

determinism: same_hardware_bitwise
```

---

## reduce_sum_to_cuda  (ReduceSumTo — broadcast-target sum reduction, CUDA/baracuda)

CUDA (baracuda) registration of `ReduceSumTo` — stride-driven on input; transposed-view grads skip Contiguize.

```fkc
kernel: reduce_sum_to_cuda
op_kind: ReduceSumTo
blurb: "Sum-reduce to a broadcast-compatible target shape (CUDA/baracuda); stride-driven on input."
backend: Cuda
kernel_source: "baracuda"
entry_point: "baracuda::reduce::reduce_sum_to_f32"   # f64/f16/bf16 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(input, output_shape)"
  op_params:
    variant: ReduceSumTo
    fields:
      input_shape:  { kind: "Vec<usize>", constraint: "product == input.elem_count" }
      output_shape: { kind: "Vec<usize>", constraint: "left-pad-1 to input rank; each axis == input size OR 1" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, output_shape, keepdim)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided       # stride-driven on input; transposed-view grads skip Contiguize
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "any_input_strided", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "in_elems"
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                 # PrecisionGuarantee::none(reason) — nondeterministic FP fold; no static bound applies (§4.9 / §10.9)
  notes: "f32/f64 native; bf16/f16 accumulate in f32 then narrow on store; stride-driven on input. baracuda reduction accumulation order is not bit-stable; advertised none(reason) per the family precision bulk-fill (§12.4)."

determinism: nondeterministic
```

---

## reduce_max_to_cpu  (ReduceMaxTo — broadcast-target max reduction, CPU)

One section per backend. Like `ReduceSumTo` but folds the **maximum** of every input element mapping to an output slot
(`f32::max`, NaN-as-missing, `-inf` init). Registered by two backends: CPU
`reduce_max_to_{f32,f64,bf16,f16}_cpu_wrapper` (`dispatch.rs:4037`, contiguous, bit-stable);
baracuda (`baracuda_dispatch.rs:2900`, strided on input). No Vulkan registration. Shape/param
carrier: `OpParams::ReduceMaxTo { input_shape, output_shape }`.

```fkc
kernel: reduce_max_to_cpu
op_kind: ReduceMaxTo
blurb: "Max-reduce to a broadcast-compatible target shape (CPU); f32::max NaN-as-missing; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_max_to_f32"   # f64/bf16/f16 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(input, output_shape)"
  op_params:
    variant: ReduceMaxTo          # OpParams::ReduceMaxTo (primitive namespace)
    fields:
      input_shape:  { kind: "Vec<usize>", constraint: "product == input.elem_count" }
      output_shape: { kind: "Vec<usize>", constraint: "left-pad-1 to input rank; each axis == input size OR 1" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, output_shape, keepdim)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "in_elems"             # one compare per folded element
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true   # extremum is exact (no rounding); deterministic order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "exact extremum via f32::max (NaN-as-missing); -inf init; bf16/f16 extremum in f32 space, narrow on store (kept value representable)."

determinism: same_hardware_bitwise
```

---

## reduce_max_to_cuda  (ReduceMaxTo — broadcast-target max reduction, CUDA/baracuda)

CUDA (baracuda) registration of `ReduceMaxTo` — stride-driven on input; NaN-as-missing extremum.

```fkc
kernel: reduce_max_to_cuda
op_kind: ReduceMaxTo
blurb: "Max-reduce to a broadcast-compatible target shape (CUDA/baracuda); stride-driven on input."
backend: Cuda
kernel_source: "baracuda"
entry_point: "baracuda::reduce::reduce_max_to_f32"   # f64/f16/bf16 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(input, output_shape)"
  op_params:
    variant: ReduceMaxTo
    fields:
      input_shape:  { kind: "Vec<usize>", constraint: "product == input.elem_count" }
      output_shape: { kind: "Vec<usize>", constraint: "left-pad-1 to input rank; each axis == input size OR 1" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, output_shape, keepdim)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "any_input_strided", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "in_elems"
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true   # extremum (max) is exact / order-independent
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "exact extremum (no FP accumulation); NaN-as-missing; stride-driven on input; bf16/f16 extremum in f32 space."

determinism: same_hardware_bitwise
```

---

## ReduceMaxToBackward  (gradient of ReduceMaxTo)

Backward of `Op::ReduceMaxTo`: two inputs `(x, upstream)` where `x.shape == input_shape` and
`upstream.shape == output_shape`; output `grad_x` of `input_shape`. Recomputes the forward max,
builds the argmax mask `x == broadcast(max)`, sum-reduces the mask to per-slot tie counts (clamped
≥ 1), then routes `upstream / count` back to the argmax positions gated by the mask (**fair-share
on ties**). NaN follows the forward `ReduceMaxTo` semantics. `bf16`/`f16` promote the **whole**
backward to f32 then narrow the grad on store. **CPU-only** in this crate
(`dispatch.rs:4445`); no CU/VK registration. Shape/param carrier:
`OpParams::ReduceMaxToBackward { input_shape, output_shape }`. Multi-pass (recompute max + mask +
sum + scale + gate), so it allocates f32 scratch — the memory term is left to the Judge.

```fkc
kernel: reduce_max_to_backward_cpu
op_kind: ReduceMaxToBackward
blurb: "ReduceMaxTo backward (CPU): route upstream to argmax positions, fair-share on ties; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_max_to_backward_f32"   # f64/bf16/f16 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(x, output_shape)"   # x == input_shape; out reduces x along collapsed axes
    - name: upstream
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out_shape"          # upstream.elem_count == product(output_shape)
  op_params:
    variant: ReduceMaxToBackward   # OpParams::ReduceMaxToBackward (primitive namespace)
    fields:
      input_shape:  { kind: "Vec<usize>", constraint: "product == x.elem_count == out.elem_count" }
      output_shape: { kind: "Vec<usize>", constraint: "product == upstream.elem_count; left-pad-1 to input rank; each axis == input size OR 1" }

return:
  outputs:
    - name: grad_x
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)                 # grad of x = input_shape
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  # multi-pass: recompute max (read in_elems), build mask (in_elems), sum mask (in_elems),
  # scale (out_elems), broadcast+gate (in_elems). O(in_elems) compute, O(in_elems) write.
  flops: "in_elems"
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"
  # overhead_ns + per-tier memory (f32 scratch for max/mask/scaled + half promotion): Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f32/f64; bf16/f16 promote whole backward to f32 then narrow grad on store; upstream / tie_count fair-share; counts clamped >= 1; deterministic order."

determinism: same_hardware_bitwise
```

---

## argmax_dim_cpu  (ArgMaxDim — argmax along one dim → U32 index, CPU)

One section per backend. Index reduction along a single `dim` (`dim = dims[0]` of `OpParams::Reduce`). For each
`(outer, inner)` lane, scans the `dim_size` slice and writes the **U32 index** of the maximum.
The first slice element seeds `best_idx = 0`; subsequent elements replace the best only on a
**strict** `new > best`, so **ties keep the first (lowest) index**; NaN candidates never displace
the running best (IEEE `new > best` is false for NaN). **Output dtype is always U32**, regardless
of input dtype (binding key `[input_dt, U32]`). `dim` size 0 is rejected (argmax undefined). The
output drops `dim`. Registered by three backends:

- **CPU** (`dispatch.rs:4579`): `argmax_dim_u32_cpu_dispatch` matches input dtype internally
  (the as-built CPU argextremum kernel is f32; the f64/bf16/f16 key entries are dispatch-side
  adapters), default caps → **contiguous-only**, exact/bitwise. `kernel_source: "portable-cpu"`.
- **CU / baracuda** (`baracuda_dispatch.rs:2515`): `KernelCaps::strided_input()` → **strided**.
  `kernel_source: "baracuda"`.
- **VK** (`vulkan_dispatch.rs:4507`): `arg_reduce::argmax_{f32,f16,bf16,f64}`, default caps →
  **contiguous-only**, `PrecisionGuarantee::none` (tree reduction over (val, idx) pairs; lower
  index wins ties — value-deterministic given inputs, but advertised none per the registration).
  `kernel_source: "vulkan-slang"`.

The U32-index output and the exact index-selection precision are shared by all three; CPU/VK are
contiguous, baracuda is strided.

```fkc
kernel: argmax_dim_cpu
op_kind: ArgMaxDim
blurb: "Argmax along one dim (CPU; U32 index out); first/lowest index wins ties; NaN never wins; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::argmax_dim_f32"   # f64/bf16/f16 via dispatch adapter
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dim = dims[0] (single reduce dim), in 0..rank; input_shape[dim] > 0"
  op_params:
    variant: Reduce          # OpParams::Reduce reused for ArgMaxDim; dim = dims[0] (single-dim constraint)
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly one dim; dim in 0..input_rank; input_shape[dim] != 0" }
      keepdim: { kind: bool, note: "always false; output drops dim" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)             # output is always U32 indices, regardless of input dtype
      shape_rule: reduce(input, dims, keepdim)   # input with dim removed
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_in"                  # one compare per input element (single pass)
  bytes_moved: "n_in * dtype_bytes + n_out * 4"   # read input, write U32 indices (4B); bandwidth-bound
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true   # integer index selection; exact, no rounding
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact index selection (no FP accumulation); ties resolve to the first/lowest index (strict > compare); NaN candidates never replace the running best (IEEE); same result on any hardware for given input values."

determinism: bitwise
```

---

## argmax_dim_cuda  (ArgMaxDim — argmax along one dim → U32 index, CUDA/baracuda)

CUDA (baracuda) registration of `ArgMaxDim` — stride-driven; first/lowest index wins ties.

```fkc
kernel: argmax_dim_cuda
op_kind: ArgMaxDim
blurb: "Argmax along one dim (CUDA/baracuda; U32 index out); stride-driven; first/lowest index wins ties."
backend: Cuda
kernel_source: "baracuda"
entry_point: "baracuda::reduce::argmax_dim_f32"   # f64/f16/bf16 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dim = dims[0], in 0..rank; input_shape[dim] > 0"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly one dim; dim in 0..input_rank; input_shape[dim] != 0" }
      keepdim: { kind: bool, note: "always false; output drops dim" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: reduce(input, dims, keepdim)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "any_input_strided", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_in"
  bytes_moved: "n_in * dtype_bytes + n_out * 4"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true   # exact index selection
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact index selection (no FP accumulation); ties resolve to first/lowest index; NaN never replaces the running best; stride-driven."

determinism: bitwise
```

---

## argmax_dim_vulkan  (ArgMaxDim — argmax along one dim → U32 index, Vulkan)

Vulkan registration of `ArgMaxDim` — contiguous; tree reduction over (val, idx) pairs, `PrecisionGuarantee::none`.

```fkc
kernel: argmax_dim_vulkan
op_kind: ArgMaxDim
blurb: "Argmax along one dim (Vulkan; U32 index out); contiguous; tree reduction over (val, idx) pairs."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::arg_reduce::argmax_f32"   # f16/bf16/f64 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dim = dims[0], in 0..rank; input_shape[dim] > 0"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly one dim; dim in 0..input_rank; input_shape[dim] != 0" }
      keepdim: { kind: bool, note: "always false; output drops dim" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: reduce(input, dims, keepdim)
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
  flops: "n_in"
  bytes_moved: "n_in * dtype_bytes + n_out * 4"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: false   # advertised none at registration (tree reduction)
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                 # PrecisionGuarantee::none(reason)
  notes: "tree reduction over (val, idx) pairs; lower index wins on ties — value-deterministic given input values, advertised none per vulkan_dispatch.rs:4505 ARG_MAX_REASON."

determinism: nondeterministic
```

---

## argmin_dim_cpu  (ArgMinDim — argmin along one dim → U32 index, CPU)

One section per backend. Mirror of `ArgMaxDim`: identical scan with a strict `new < best` comparator and `+inf` init,
writing the **U32 index** of the minimum along `dim`; ties keep the first (lowest) index; NaN never
displaces the running best. **Output always U32.** `dim` size 0 rejected; output drops `dim`. Same
backend fan-out and per-block facts as `ArgMaxDim`: CPU `argmin_dim_f32` (+adapters)
(`dispatch.rs:4579`, contiguous, exact/bitwise); baracuda (`baracuda_dispatch.rs:2515`, strided);
Vulkan `arg_reduce::argmin_{f32,f16,bf16,f64}` (`vulkan_dispatch.rs:4511`, contiguous,
`PrecisionGuarantee::none`).

```fkc
kernel: argmin_dim_cpu
op_kind: ArgMinDim
blurb: "Argmin along one dim (CPU; U32 index out); first/lowest index wins ties; NaN never wins; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::argmin_dim_f32"   # f64/bf16/f16 via dispatch adapter
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dim = dims[0] (single reduce dim), in 0..rank; input_shape[dim] > 0"
  op_params:
    variant: Reduce          # OpParams::Reduce reused for ArgMinDim; dim = dims[0]
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly one dim; dim in 0..input_rank; input_shape[dim] != 0" }
      keepdim: { kind: bool, note: "always false; output drops dim" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: reduce(input, dims, keepdim)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_in"
  bytes_moved: "n_in * dtype_bytes + n_out * 4"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact index selection (no FP accumulation); ties resolve to the first/lowest index (strict < compare); NaN candidates never replace the running best (IEEE); same result on any hardware for given input values."

determinism: bitwise
```

---

## argmin_dim_cuda  (ArgMinDim — argmin along one dim → U32 index, CUDA/baracuda)

CUDA (baracuda) registration of `ArgMinDim` — stride-driven; first/lowest index wins ties.

```fkc
kernel: argmin_dim_cuda
op_kind: ArgMinDim
blurb: "Argmin along one dim (CUDA/baracuda; U32 index out); stride-driven; first/lowest index wins ties."
backend: Cuda
kernel_source: "baracuda"
entry_point: "baracuda::reduce::argmin_dim_f32"   # f64/f16/bf16 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dim = dims[0], in 0..rank; input_shape[dim] > 0"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly one dim; dim in 0..input_rank; input_shape[dim] != 0" }
      keepdim: { kind: bool, note: "always false; output drops dim" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: reduce(input, dims, keepdim)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "any_input_strided", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n_in"
  bytes_moved: "n_in * dtype_bytes + n_out * 4"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact index selection (no FP accumulation); ties resolve to first/lowest index; NaN never replaces the running best; stride-driven."

determinism: bitwise
```

---

## argmin_dim_vulkan  (ArgMinDim — argmin along one dim → U32 index, Vulkan)

Vulkan registration of `ArgMinDim` — contiguous; tree reduction over (val, idx) pairs, `PrecisionGuarantee::none`.

```fkc
kernel: argmin_dim_vulkan
op_kind: ArgMinDim
blurb: "Argmin along one dim (Vulkan; U32 index out); contiguous; tree reduction over (val, idx) pairs."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::arg_reduce::argmin_f32"   # f16/bf16/f64 siblings
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dim = dims[0], in 0..rank; input_shape[dim] > 0"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "exactly one dim; dim in 0..input_rank; input_shape[dim] != 0" }
      keepdim: { kind: bool, note: "always false; output drops dim" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: reduce(input, dims, keepdim)
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
  flops: "n_in"
  bytes_moved: "n_in * dtype_bytes + n_out * 4"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                 # PrecisionGuarantee::none(reason)
  notes: "tree reduction over (val, idx) pairs with min comparator; lower index wins on ties — value-deterministic given inputs, advertised none per vulkan_dispatch.rs:4506 ARG_MIN_REASON."

determinism: nondeterministic
```

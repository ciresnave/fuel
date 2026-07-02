---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu
  kernel_source: "portable-cpu"
  link_registry: fuel_cpu_backend::fkc::ENTRY_POINTS   # §12.6 symbol → KernelRef map
  revision_base: "git:f41137b4"                        # folded into kernel_revision_hash (§4.7)
---

# fuel-cpu-backend — reduce family kernel contracts

The portable byte-shaped **reduction** kernels Fuel itself provides: the per-axis
`{sum,mean,max,min}_reduce_{f32,f64,bf16,f16}` set backed by the reduction chassis
(`fuel-cpu-backend/src/chassis/reduction.rs:282`, thunks `byte_kernels.rs:7306-7468`), plus the
`argmax_dim_f32` / `argmin_dim_f32` index reductions (`byte_kernels.rs:7486-7587`). Inventory
source: `docs/kernel-contracts/_inventory/cpu.md` (Reductions section, crate `cpu`, family
`reduce`).

Cross-cutting facts for this family (from the inventory's "Cross-cutting facts" and the per-axis
reduce section):

- **Layout: contiguous, zero-offset, row-major** on every input. None of these kernels consult a
  `Layout`/strides/offset internally — they walk a flat `CpuStorageBytes` slice and validate byte
  length against the declared shape. The pipelined executor's auto-Contiguize pass realizes any
  strided/broadcast/offset input into a contiguous buffer **before** the kernel runs, so every
  reduce kernel declares `awkward_layout_strategy: requires_contiguous` and the planner prices an
  inserted `Op::Contiguize` (itself an FKC kernel, §4.3) for a non-contiguous operand.
- **Output: pre-allocated, fully overwritten, contiguous.** The output buffer is caller-allocated
  to the exact byte size and overwritten (no read of prior contents); no aliasing with the input.
- **Shape:** `input_shape` flows through the `KernelRef` `layouts` side-channel (`layouts[0]`),
  **not** through `OpParams`; `OpParams::Reduce { dims, keepdim }` carries only the reduce axes and
  the keepdim flag. `keepdim` is `false` for every call fuel-graph emits today (the field is
  reserved); the output therefore has the reduced dims **removed**.
- **Accumulator / precision invariant:** f32/f64 accumulate in their own dtype (`Acc = T`);
  bf16/f16 accumulate in **f32** and narrow on store (`Acc = f32`, the load-bearing
  accumulator-promotion invariant encoded in `ReduceOp::Acc`,
  `fuel-cpu-backend/src/chassis/reduction.rs:47-76`). All four reduce kernels are deterministic
  on the same hardware (a fixed left-to-right fold order over the contiguous input).
- **Cost provenance:** every cost block in this family is marked `judge_measured` (the Judge
  bootstraps the coefficients, §4.4). The `flops` / `bytes_moved` formula **hints** are retained
  because a single-pass reduction is genuinely reduction/bandwidth-bound and the structure is
  derivable from the op (one fold per input element; read all input + write all output); the
  Judge refines the absolute numbers and the launch overhead. No overhead constant is fabricated.

---

## reduce  (per-axis Sum / Mean / Max / Min reduction chassis)

The shared reduction chassis (`fuel-cpu-backend/src/chassis/reduction.rs:282`) behind every
`{sum,mean,max,min}_reduce_*` thunk. This is the **family overview** section; the registrable
per-(op, dtype) contracts follow below (`sum_reduce_f32`, `mean_reduce_f64`, …). One pass over
the contiguous input: for each element, decode its multi-index, project it to its destination
output slot, fold it into that slot's accumulator; after the pass, finalize each slot once. Sum
finalizes as identity; Mean divides the accumulator by `count` (= product of reduced-dim sizes)
and rejects `count == 0`; Max/Min finalize as identity over an extremum fold (`-inf`/`+inf`
init). bf16/f16 widen each element to f32, fold in f32, narrow on store. Ops Sum/Mean/Max/Min ×
dtypes f32/f64/bf16/f16 give 16 thunks. Known limitation: contiguous-only — any
strided/broadcast/offset operand must be contiguized by the planner first.

```fkc
kernel: reduce
op_kind: SumReduce          # family chassis; one registrable contract per (op,dtype) below
registrable: false          # §3.10 describe-only: shared reduction chassis, NOT a dispatch target — the per-(op,dtype) thunks below are the registrable contracts (WITHOUT this the chassis double-registers SumReduce/[F32] → DuplicateKernelRef at init)
blurb: "Per-axis Sum/Mean/Max/Min reduction chassis; contiguous row-major; half via f32 accumulator."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sum_reduce_f32"   # representative; per-(op,dtype) below
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape flows via KernelRef.layouts[0]; reduce_dims sorted-ascending+unique, in range 0..rank"
  op_params:
    variant: Reduce          # OpParams::Reduce (primitive namespace; §3.7)
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today; output drops reduced dims" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)         # same dtype as input
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
  # overhead_ns + per-tier memory: bootstrapped by the Judge (not fabricated)

precision:
  bit_stable_on_same_hardware: true   # fixed left-to-right fold order; deterministic
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                # CPU primitive: importer applies PRIMITIVE_DETERMINISTIC_CPU (§12.4)
  notes: "f32/f64 accumulate native; bf16/f16 accumulate in f32 then narrow on store. Mean rejects count==0; Max/Min use f32::max/min (NaN-as-missing)."

determinism: same_hardware_bitwise
```

---

## sum_reduce_f32  (sum-reduce, f32)

Sum-reduce an f32 tensor over `dims`. Accumulator is f32 (`Acc = T`); finalize is identity. One
pass over the contiguous row-major input; output is the input with the reduced dims removed,
overwritten. Contiguous-only.

```fkc
kernel: sum_reduce_f32
op_kind: SumReduce
blurb: "Sum-reduce f32 over dims; f32 accumulator; contiguous row-major; reduced dims removed."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sum_reduce_f32"
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
      keepdim: { kind: bool, note: "always false today" }

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
  bytes_moved: "(n_in + n_out) * 4"     # f32 = 4 bytes/elem
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32 accumulator; fixed fold order; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## sum_reduce_f64  (sum-reduce, f64)

Sum-reduce an f64 tensor over `dims`. Accumulator is f64; finalize is identity. Same shape/layout
contract as `sum_reduce_f32`.

```fkc
kernel: sum_reduce_f64
op_kind: SumReduce
blurb: "Sum-reduce f64 over dims; f64 accumulator; contiguous row-major; reduced dims removed."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sum_reduce_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today" }

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
  bytes_moved: "(n_in + n_out) * 8"     # f64 = 8 bytes/elem
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f64 accumulator (native); fixed fold order; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## sum_reduce_bf16  (sum-reduce, bf16)

Sum-reduce a bf16 tensor over `dims`. Each element widens to f32, folds in an **f32 accumulator**
(the precision invariant — a streaming f32 sum avoids per-add bf16 rounding), then narrows back to
bf16 on store. Same shape/layout contract as the f32 variant.

```fkc
kernel: sum_reduce_bf16
op_kind: SumReduce
blurb: "Sum-reduce bf16 over dims; f32 accumulator narrowed on store; contiguous row-major."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sum_reduce_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today" }

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
  bytes_moved: "(n_in + n_out) * 2"     # bf16 = 2 bytes/elem
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen bf16→f32, accumulate in f32, narrow on store; fixed fold order; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## sum_reduce_f16  (sum-reduce, f16)

Sum-reduce an f16 tensor over `dims`. f32 accumulator (widen on load, narrow on store), as for
bf16. Same shape/layout contract.

```fkc
kernel: sum_reduce_f16
op_kind: SumReduce
blurb: "Sum-reduce f16 over dims; f32 accumulator narrowed on store; contiguous row-major."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::sum_reduce_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today" }

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
  bytes_moved: "(n_in + n_out) * 2"     # f16 = 2 bytes/elem
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen f16→f32, accumulate in f32, narrow on store; fixed fold order; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## mean_reduce_f32  (mean-reduce, f32)

Mean-reduce an f32 tensor over `dims`: sum via the same f32 accumulator as `sum_reduce_f32`, then
divide each slot by `count` (= product of reduced-dim sizes) in finalize. **Rejects `count == 0`**
(reduced dim has size 0 → divisor zero → typed `Error`, not a silent NaN). Contiguous-only.

```fkc
kernel: mean_reduce_f32
op_kind: MeanReduce
blurb: "Mean-reduce f32 over dims; f32 accumulator; divide by count; rejects count==0; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::mean_reduce_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank; product(reduced dims) > 0 (count != 0)"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today" }

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
  bytes_moved: "(n_in + n_out) * 4"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32 accumulator then divide by count; rejects count==0; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## mean_reduce_f64  (mean-reduce, f64)

Mean-reduce an f64 tensor: f64 accumulator, divide by `count`, reject `count == 0`. Same
shape/layout contract as `mean_reduce_f32`.

```fkc
kernel: mean_reduce_f64
op_kind: MeanReduce
blurb: "Mean-reduce f64 over dims; f64 accumulator; divide by count; rejects count==0; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::mean_reduce_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank; product(reduced dims) > 0"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today" }

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
  flops: "n_in + n_out"
  bytes_moved: "(n_in + n_out) * 8"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f64 accumulator then divide by count; rejects count==0; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## mean_reduce_bf16  (mean-reduce, bf16)

Mean-reduce a bf16 tensor: widen to f32, accumulate in f32, divide by `count`, narrow to bf16 on
store. Rejects `count == 0`. Same shape/layout contract.

```fkc
kernel: mean_reduce_bf16
op_kind: MeanReduce
blurb: "Mean-reduce bf16 over dims; f32 accumulator divided by count, narrowed on store; rejects count==0."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::mean_reduce_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank; product(reduced dims) > 0"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today" }

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
  flops: "n_in + n_out"
  bytes_moved: "(n_in + n_out) * 2"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen bf16→f32, accumulate in f32, divide by count, narrow on store; rejects count==0; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## mean_reduce_f16  (mean-reduce, f16)

Mean-reduce an f16 tensor: f32 accumulator, divide by `count`, narrow to f16 on store. Rejects
`count == 0`. Same shape/layout contract.

```fkc
kernel: mean_reduce_f16
op_kind: MeanReduce
blurb: "Mean-reduce f16 over dims; f32 accumulator divided by count, narrowed on store; rejects count==0."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::mean_reduce_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank; product(reduced dims) > 0"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today" }

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
  flops: "n_in + n_out"
  bytes_moved: "(n_in + n_out) * 2"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "widen f16→f32, accumulate in f32, divide by count, narrow on store; rejects count==0; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## max_reduce_f32  (max-reduce, f32)

Max-reduce an f32 tensor over `dims`. Accumulator inits to `-inf`; fold keeps the larger of the
two via `f32::max` (**NaN-as-missing** — `f32::max(a, NaN) == a`); finalize is identity.
Contiguous-only.

```fkc
kernel: max_reduce_f32
op_kind: MaxReduce
blurb: "Max-reduce f32 over dims; f32::max (NaN-as-missing); -inf init; contiguous row-major."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::max_reduce_f32"
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
      keepdim: { kind: bool, note: "always false today" }

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
  bytes_moved: "(n_in + n_out) * 4"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "extremum via f32::max (NaN-as-missing); exact (no rounding); deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## max_reduce_f64  (max-reduce, f64)

Max-reduce an f64 tensor: `f64::max`, `-inf` init, NaN-as-missing. Same shape/layout contract.

```fkc
kernel: max_reduce_f64
op_kind: MaxReduce
blurb: "Max-reduce f64 over dims; f64::max (NaN-as-missing); -inf init; contiguous row-major."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::max_reduce_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today" }

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
  bytes_moved: "(n_in + n_out) * 8"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "extremum via f64::max (NaN-as-missing); exact; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## max_reduce_bf16  (max-reduce, bf16)

Max-reduce a bf16 tensor: each element widens to f32, the extremum runs in f32 space (uniform NaN
handling via `f32::max`), then narrows to bf16 on store (lossless — the kept element is already a
representable bf16 value). `-inf` init. Same shape/layout contract.

```fkc
kernel: max_reduce_bf16
op_kind: MaxReduce
blurb: "Max-reduce bf16 over dims; extremum in f32 space (NaN-as-missing); narrowed on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::max_reduce_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today" }

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
  bytes_moved: "(n_in + n_out) * 2"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "extremum in f32 space (NaN-as-missing), narrow on store (kept value is a representable bf16); deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## max_reduce_f16  (max-reduce, f16)

Max-reduce an f16 tensor: extremum in f32 space, `-inf` init, narrow to f16 on store. Same
shape/layout contract.

```fkc
kernel: max_reduce_f16
op_kind: MaxReduce
blurb: "Max-reduce f16 over dims; extremum in f32 space (NaN-as-missing); narrowed on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::max_reduce_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today" }

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
  bytes_moved: "(n_in + n_out) * 2"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "extremum in f32 space (NaN-as-missing), narrow on store (kept value is a representable f16); deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## min_reduce_f32  (min-reduce, f32)

Min-reduce an f32 tensor over `dims`. Mirror of `max_reduce_f32`: accumulator inits to `+inf`;
fold keeps the smaller via `f32::min` (NaN-as-missing); finalize is identity. Contiguous-only.

```fkc
kernel: min_reduce_f32
op_kind: MinReduce
blurb: "Min-reduce f32 over dims; f32::min (NaN-as-missing); +inf init; contiguous row-major."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::min_reduce_f32"
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
      keepdim: { kind: bool, note: "always false today" }

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
  bytes_moved: "(n_in + n_out) * 4"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "extremum via f32::min (NaN-as-missing); exact; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## min_reduce_f64  (min-reduce, f64)

Min-reduce an f64 tensor: `f64::min`, `+inf` init, NaN-as-missing. Same shape/layout contract.

```fkc
kernel: min_reduce_f64
op_kind: MinReduce
blurb: "Min-reduce f64 over dims; f64::min (NaN-as-missing); +inf init; contiguous row-major."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::min_reduce_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today" }

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
  bytes_moved: "(n_in + n_out) * 8"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "extremum via f64::min (NaN-as-missing); exact; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## min_reduce_bf16  (min-reduce, bf16)

Min-reduce a bf16 tensor: extremum in f32 space (uniform NaN handling via `f32::min`), `+inf`
init, narrow to bf16 on store. Same shape/layout contract.

```fkc
kernel: min_reduce_bf16
op_kind: MinReduce
blurb: "Min-reduce bf16 over dims; extremum in f32 space (NaN-as-missing); narrowed on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::min_reduce_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today" }

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
  bytes_moved: "(n_in + n_out) * 2"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "extremum in f32 space (NaN-as-missing), narrow on store (kept value is a representable bf16); deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## min_reduce_f16  (min-reduce, f16)

Min-reduce an f16 tensor: extremum in f32 space, `+inf` init, narrow to f16 on store. Same
shape/layout contract.

```fkc
kernel: min_reduce_f16
op_kind: MinReduce
blurb: "Min-reduce f16 over dims; extremum in f32 space (NaN-as-missing); narrowed on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::min_reduce_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dims sorted-ascending+unique in 0..rank"
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "sorted ascending, unique, each in 0..input_rank" }
      keepdim: { kind: bool, note: "always false today" }

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
  bytes_moved: "(n_in + n_out) * 2"
  # overhead_ns + memory: Judge-bootstrapped

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "extremum in f32 space (NaN-as-missing), narrow on store (kept value is a representable f16); deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## argmax_dim_f32  (argmax along one dim, f32 → U32 index)

Index reduction along a single `dim` (`fuel-cpu-backend/src/byte_kernels.rs:7554`, via
`argextremum_dim_f32:7486`). For each `(outer, inner)` lane, scans the `dim_size` slice and writes
the **U32 index** of the maximum. The first element of the slice seeds `best_val`/`best_idx=0`;
each subsequent element replaces the best only on a **strict** `new > best`, so **ties keep the
first (lowest) index**. NaN follows IEEE: `new > best` is false for any NaN candidate, so a NaN
never displaces the running best, and a leading NaN persists in `best_val` only until a strictly
greater non-NaN appears. **Input dtype F32 only** (the as-built CPU argextremum kernel is f32;
the f64/bf16/f16 entries in the inventory are dispatch-side adapters, not separate CPU kernels);
**output dtype U32**. The reduce axis arrives via `OpParams::Reduce { dims }` with the single-dim
constraint (`dim = dims[0]`); `dim` size 0 is rejected (argmax undefined). The output drops
`dim`. Contiguous-only.

```fkc
kernel: argmax_dim_f32
op_kind: ArgMaxDim
registrable: false          # DEFERRED: this contract section is f32-only, but production registers ArgMaxDim for ALL input dtypes {F32,F64,BF16,F16} via the single argmax_dim_u32_cpu_dispatch (dispatch.rs, in a dtype loop). Reconcile the contract (add the other-dtype sections / match the dispatch granularity) before importing; the hand-written arg regs stay authoritative until then.
blurb: "Argmax along one dim (f32 in, U32 index out); first/lowest index wins ties; NaN never wins; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::argmax_dim_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
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
      dtype_rule: fixed(U32)             # output is always U32 indices
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
  flops: "n_in"                  # one compare per input element (single pass); derivable
  bytes_moved: "n_in * 4 + n_out * 4"   # read f32 input (4B), write U32 indices (4B); bandwidth-bound
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

## argmin_dim_f32  (argmin along one dim, f32 → U32 index)

Mirror of `argmax_dim_f32` (`fuel-cpu-backend/src/byte_kernels.rs:7572`): identical scan with a
strict `new < best` comparator and `+inf` init. For each `(outer, inner)` lane, writes the **U32
index** of the minimum along `dim`; the first slice element seeds `best_idx=0` and subsequent
elements replace only on strict `<`, so **ties keep the first (lowest) index**. NaN never
displaces the running best (`new < best` is false for NaN). **Input F32 only; output U32.** `dim`
arrives via `OpParams::Reduce { dims }` (single-dim, `dim = dims[0]`); `dim` size 0 rejected. The
output drops `dim`. Contiguous-only.

```fkc
kernel: argmin_dim_f32
op_kind: ArgMinDim
registrable: false          # DEFERRED: this contract section is f32-only, but production registers ArgMinDim for ALL input dtypes {F32,F64,BF16,F16} via the single argmin_dim_u32_cpu_dispatch (dispatch.rs, in a dtype loop). Reconcile the contract (add the other-dtype sections / match the dispatch granularity) before importing; the hand-written arg regs stay authoritative until then.
blurb: "Argmin along one dim (f32 in, U32 index out); first/lowest index wins ties; NaN never wins; contiguous."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::argmin_dim_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "input_shape via KernelRef.layouts[0]; dim = dims[0] (single reduce dim), in 0..rank; input_shape[dim] > 0"
  op_params:
    variant: Reduce          # OpParams::Reduce reused for ArgMinDim; dim = dims[0] (single-dim constraint)
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
  bytes_moved: "n_in * 4 + n_out * 4"
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

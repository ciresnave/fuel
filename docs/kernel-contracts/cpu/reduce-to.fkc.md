---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                                   # maps to BackendId::Cpu
  kernel_source: "portable-cpu"                  # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"                  # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — reduce-to family kernel contracts

Broadcast-target reductions: the backward of a forward broadcast. `Op::ReduceSumTo` and
`Op::ReduceMaxTo` fold a tensor down to a smaller, broadcast-compatible `output_shape`
(`grad` of a broadcast in autograd), and `Op::ReduceMaxToBackward` routes the upstream gradient
of a `ReduceMaxTo` back to the argmax positions. All entries share the `chassis::reduction`
single-pass loop (`fuel-cpu-backend/src/chassis/reduction.rs`), with the per-(op, dtype)
public thunks in `byte_kernels.rs`.

Shared facts (the cross-cutting CPU byte-kernel contract — inventory §"Cross-cutting facts"):

- **Layout: contiguous, zero-offset, row-major.** None of these kernels consult a
  `Layout`/strides/offset; they take flat `CpuStorageBytes` slices and explicit `usize`
  `input_shape` / `output_shape` parameters and validate *byte length* against them. The
  pipelined executor's auto-Contiguize pass realizes strided/broadcast/offset inputs into dense
  buffers *before* these kernels run, so `awkward_layout_strategy = requires_contiguous` for
  every entry. They are not the strided materializer (`contiguize_cpu` is); they never walk
  negative strides, so `reverse_strides: rejected` on every operand.
- **Output: pre-allocated, fully overwritten.** Output buffers are caller-allocated to the exact
  byte size; the chassis allocates one accumulator slot per output element, folds the input once,
  then finalizes each slot — overwriting (no read of prior output content). No input/output
  aliasing.
- **Half-float precision invariant.** `bf16`/`f16` accumulate in an **f32** accumulator
  (`ReduceOp::Acc = f32`), narrowing back on store — the load-bearing precision invariant encoded
  in the chassis associated type. `f32`/`f64` accumulate natively in their own dtype.
- **Shape rule.** `output_shape` is left-padded with 1s to `input_shape`'s rank; per padded axis
  it must equal the input size (axis carries through) or `1` (axis reduced away); any other value
  is a build/runtime contract violation (`align_reduce_to`, `reduction.rs:395`).

---

## reduce_to  (broadcast-target reduce chassis — Sum / Max)

The single chassis function `reduce_to<T, R>` (`fuel-cpu-backend/src/chassis/reduction.rs:323`)
that backs every `reduce_sum_to_*` / `reduce_max_to_*` thunk. It is the shared algorithm, not a
distinct dispatch entry-point: `R: ReduceOp<T>` selects Sum vs Max, `T` selects the dtype, and the
public per-(op, dtype) `byte_kernels.rs` functions are 1-line thunks over it. It is documented here
once so the per-dtype sections below need not restate the loop, validation, and precision invariant.

Algorithm: left-pad `output_shape` with 1s to the input rank and validate every axis equals the
input size or 1 (`align_reduce_to`); validate `input.len_bytes() == in_elems*elem` and
`output.len_bytes() == out_elems*elem`; allocate one accumulator slot per output element
(`R::init()`); walk the input once, projecting each input multi-index to its output slot by
clamping coords to 0 on collapsed (padded==1) axes, folding via `R::fold`; finalize each slot once
via `R::finalize` (Sum/Max ignore `count`; only Mean — not in this family — reads it). Numerics:
`f32`/`f64` native accumulator; `bf16`/`f16` accumulate in **f32** and narrow on store (the
`ReduceOp::Acc = f32` half-float invariant). Single input pass, single finalize pass; memory-bandwidth
bound. Limitations: contiguous-only, offset-0, no negative/strided walk; the executor contiguizes
awkward layouts first. Dispatched via `Op::ReduceSumTo` / `Op::ReduceMaxTo` and their
`OpParams::ReduceSumTo` / `OpParams::ReduceMaxTo` carriers (`input_shape`, `output_shape`).

```fkc
kernel: reduce_to
op_kind: ReduceSumTo               # shared chassis; concrete entries are ReduceSumTo / ReduceMaxTo (per-dtype sections below)
registrable: false                # §3.10 describe-only: shared reduction chassis, NOT a dispatch target — the per-(op,dtype) thunks below are the registrable contracts (WITHOUT this the chassis double-registers ReduceSumTo/[F32] → DuplicateKernelRef at init)
blurb: "Broadcast-target reduce chassis (Sum/Max): fold to output_shape; one pass; half via f32 accumulator."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::chassis::reduction::reduce_to"   # generic chassis fn; monomorphized per dtype/op (§12.6)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(input, output_shape)"   # each padded output axis == input size OR 1
  op_params:
    variant: ReduceSumTo           # OpParams::ReduceSumTo (Sum) / OpParams::ReduceMaxTo (Max) — primitive namespace
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
  awkward_layout_strategy: requires_contiguous     # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured        # Judge bootstraps; FLOPs/bandwidth hint below is the structural prior it refines
  class: reduction
  # one fold per input element (read), one finalize per output element (write).
  flops: "in_elems"                 # in_elems = product(input_shape)
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # read every input + write every output slot
  memory: { device_bytes: 0, host_bytes: "out_elems * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic single-pass fold in a fixed order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32/f64 native accumulator; bf16/f16 accumulate in f32 then narrow on store. Sum=IEEE add; Max=f32::max (NaN-as-missing)."

determinism: same_hardware_bitwise
```

---

## reduce_sum_to_f32  (sum-reduce F32 to a broadcast-compatible target)

`reduce_sum_to_f32` (`byte_kernels.rs:4742`) — thunk over `reduce_to::<f32, Sum>`. Sums every
input element that maps to a given output slot (collapsed axes summed), output dtype/shape =
`output_shape`. Native f32 accumulator (`Sum::Acc = f32`); IEEE addition. Used by the broadcast
backward of `Op::ReduceSumTo`. Contiguous, offset-0; overwrite.

```fkc
kernel: reduce_sum_to_f32
op_kind: ReduceSumTo
blurb: "Sum-reduce F32 to a broadcast-compatible target shape; native f32 accumulator."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_sum_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
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
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "in_elems"
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # dtype_bytes = 4
  memory: { device_bytes: 0, host_bytes: "out_elems * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32 accumulator; IEEE add; fixed fold order."

determinism: same_hardware_bitwise
```

---

## reduce_sum_to_f64  (sum-reduce F64 to a broadcast-compatible target)

`reduce_sum_to_f64` (`byte_kernels.rs:4754`) — thunk over `reduce_to::<f64, Sum>`. Native f64
accumulator (`Sum::Acc = f64`); IEEE addition. Otherwise identical to the f32 entry.

```fkc
kernel: reduce_sum_to_f64
op_kind: ReduceSumTo
blurb: "Sum-reduce F64 to a broadcast-compatible target shape; native f64 accumulator."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_sum_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
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
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "in_elems"
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # dtype_bytes = 8
  memory: { device_bytes: 0, host_bytes: "out_elems * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64 accumulator; IEEE add; fixed fold order."

determinism: same_hardware_bitwise
```

---

## reduce_sum_to_bf16  (sum-reduce BF16 to a broadcast-compatible target)

`reduce_sum_to_bf16` (`byte_kernels.rs:4767`) — thunk over `reduce_to::<bf16, Sum>`. I/O dtype is
bf16 but the **accumulator runs in f32** (`Sum::Acc = f32`), narrowing on store — the half-float
precision invariant (a per-element bf16 add would round to ~3 decimal digits; the f32 accumulator
gives full f32 precision up to ~16M elements).

```fkc
kernel: reduce_sum_to_bf16
op_kind: ReduceSumTo
blurb: "Sum-reduce BF16 to a broadcast-compatible target shape; f32 accumulator, narrow on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_sum_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
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
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "in_elems"
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # dtype_bytes = 2 (bf16 I/O)
  memory: { device_bytes: 0, host_bytes: "out_elems * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 I/O; f32 accumulator then narrow on store (Acc=f32 half-float invariant); fixed fold order."

determinism: same_hardware_bitwise
```

---

## reduce_sum_to_f16  (sum-reduce F16 to a broadcast-compatible target)

`reduce_sum_to_f16` (`byte_kernels.rs:4780`) — thunk over `reduce_to::<f16, Sum>`. f16 I/O,
**f32 accumulator** narrowing on store (same half-float invariant as the bf16 entry).

```fkc
kernel: reduce_sum_to_f16
op_kind: ReduceSumTo
blurb: "Sum-reduce F16 to a broadcast-compatible target shape; f32 accumulator, narrow on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_sum_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
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
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "in_elems"
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # dtype_bytes = 2 (f16 I/O)
  memory: { device_bytes: 0, host_bytes: "out_elems * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O; f32 accumulator then narrow on store (Acc=f32 half-float invariant); fixed fold order."

determinism: same_hardware_bitwise
```

---

## reduce_max_to_f32  (max-reduce F32 to a broadcast-compatible target)

`reduce_max_to_f32` (`byte_kernels.rs:4792`) — thunk over `reduce_to::<f32, Max>`. Folds the
maximum of every input element mapping to an output slot; native f32 extremum
(`Max::init = -INFINITY`, `Max::fold = f32::max`, **NaN-as-missing**). Used by the broadcast
backward of `Op::ReduceMaxTo`.

```fkc
kernel: reduce_max_to_f32
op_kind: ReduceMaxTo
blurb: "Max-reduce F32 to a broadcast-compatible target shape; f32::max (NaN-as-missing)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_max_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
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
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "in_elems"                  # one compare per folded element
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # dtype_bytes = 4
  memory: { device_bytes: 0, host_bytes: "out_elems * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true  # extremum is exact (no rounding); deterministic order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact f32 extremum; init -INFINITY; f32::max NaN-as-missing."

determinism: same_hardware_bitwise
```

---

## reduce_max_to_f64  (max-reduce F64 to a broadcast-compatible target)

`reduce_max_to_f64` (`byte_kernels.rs:4804`) — thunk over `reduce_to::<f64, Max>`. Native f64
extremum; same NaN-as-missing semantics as the f32 entry.

```fkc
kernel: reduce_max_to_f64
op_kind: ReduceMaxTo
blurb: "Max-reduce F64 to a broadcast-compatible target shape; f64 max (NaN-as-missing)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_max_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
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
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "in_elems"
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # dtype_bytes = 8
  memory: { device_bytes: 0, host_bytes: "out_elems * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact f64 extremum; init -INFINITY; NaN-as-missing."

determinism: same_hardware_bitwise
```

---

## reduce_max_to_bf16  (max-reduce BF16 to a broadcast-compatible target)

`reduce_max_to_bf16` (`byte_kernels.rs:4817`) — thunk over `reduce_to::<bf16, Max>`. bf16 I/O;
the extremum runs in **f32** accumulator space (`Max::Acc = f32`) for uniform NaN handling, then
narrows back to bf16. The max value itself is one of the inputs, so the f32 round-trip is exact
for any representable bf16.

```fkc
kernel: reduce_max_to_bf16
op_kind: ReduceMaxTo
blurb: "Max-reduce BF16 to a broadcast-compatible target shape; f32 extremum space, narrow on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_max_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
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
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "in_elems"
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # dtype_bytes = 2 (bf16 I/O)
  memory: { device_bytes: 0, host_bytes: "out_elems * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 I/O; extremum in f32 then narrow on store; max is a representable input so the round-trip is exact; NaN-as-missing."

determinism: same_hardware_bitwise
```

---

## reduce_max_to_f16  (max-reduce F16 to a broadcast-compatible target)

`reduce_max_to_f16` (`byte_kernels.rs:4829`) — thunk over `reduce_to::<f16, Max>`. f16 I/O;
extremum in **f32** space then narrow on store (same as the bf16 entry).

```fkc
kernel: reduce_max_to_f16
op_kind: ReduceMaxTo
blurb: "Max-reduce F16 to a broadcast-compatible target shape; f32 extremum space, narrow on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_max_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
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
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: reduction
  flops: "in_elems"
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # dtype_bytes = 2 (f16 I/O)
  memory: { device_bytes: 0, host_bytes: "out_elems * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O; extremum in f32 then narrow on store; max is a representable input so the round-trip is exact; NaN-as-missing."

determinism: same_hardware_bitwise
```

---

## reduce_max_to_backward_f32  (ReduceMaxTo backward — F32)

`reduce_max_to_backward_f32` (`byte_kernels.rs:8026`, shared `reduce_max_to_backward_impl`:8113).
Backward of `Op::ReduceMaxTo`: two inputs `(x, upstream)` where `x.shape == input_shape` and
`upstream.shape == output_shape`; output `grad_x` of `input_shape`. Routes the upstream gradient
back to the argmax positions, fair-sharing on ties. Algorithm (`byte_kernels.rs:8009-8021`):
(1) recompute the forward max via `reduce_max_to` over `input_shape→output_shape`; (2) broadcast
the max to input shape and build a mask `x == broadcast(max)`; (3) sum-reduce the mask to
`output_shape` to get per-slot tie counts; (4) clamp counts to ≥ 1 (defensive guard); (5)
`scaled_upstream = upstream / count`; (6) broadcast `scaled_upstream` to `input_shape` and gate by
the mask. Native f32 throughout. Contiguous, offset-0; overwrite (no input aliasing). Limitation:
two same-rank-aligned shapes (`input_shape`, `output_shape`) per the reduce-to axis rule; awkward
layouts contiguized by the executor first.

```fkc
kernel: reduce_max_to_backward_f32
op_kind: ReduceMaxToBackward
blurb: "ReduceMaxTo backward (F32): route upstream to argmax positions, fair-share on ties."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_max_to_backward_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(x, output_shape)"   # x == input_shape; out reduces x along collapsed axes
    - name: upstream
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out_shape"          # upstream.elem_count == product(output_shape)
  op_params:
    variant: ReduceMaxToBackward
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
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # dtype_bytes = 4
  memory: { device_bytes: 0, host_bytes: "(3 * in_elems + out_elems) * 4", disk_bytes: 0 }   # max_b, mask, scaled_b scratch + grad

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32; upstream / tie_count fair-share; counts clamped >= 1; deterministic order."

determinism: same_hardware_bitwise
```

---

## reduce_max_to_backward_f64  (ReduceMaxTo backward — F64)

`reduce_max_to_backward_f64` (`byte_kernels.rs:8050`) — same algorithm as the f32 entry over a
native f64 path.

```fkc
kernel: reduce_max_to_backward_f64
op_kind: ReduceMaxToBackward
blurb: "ReduceMaxTo backward (F64): route upstream to argmax positions, fair-share on ties."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_max_to_backward_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(x, output_shape)"
    - name: upstream
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out_shape"
  op_params:
    variant: ReduceMaxToBackward
    fields:
      input_shape:  { kind: "Vec<usize>", constraint: "product == x.elem_count == out.elem_count" }
      output_shape: { kind: "Vec<usize>", constraint: "product == upstream.elem_count; left-pad-1 to input rank; each axis == input size OR 1" }

return:
  outputs:
    - name: grad_x
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
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
  flops: "in_elems"
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # dtype_bytes = 8
  memory: { device_bytes: 0, host_bytes: "(3 * in_elems + out_elems) * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64; upstream / tie_count fair-share; counts clamped >= 1; deterministic order."

determinism: same_hardware_bitwise
```

---

## reduce_max_to_backward_bf16  (ReduceMaxTo backward — BF16)

`reduce_max_to_backward_bf16` (`byte_kernels.rs:8106`, macro `reduce_max_to_backward_half!`:8073).
bf16 I/O; the entire backward (`reduce_max_to_backward_impl`) is **promoted to f32** — x and
upstream are widened to f32, the f32 kernel runs, then the grad is narrowed back to bf16 on output.
Same argmax-routing / tie-fair-share algorithm as the f32 entry.

```fkc
kernel: reduce_max_to_backward_bf16
op_kind: ReduceMaxToBackward
blurb: "ReduceMaxTo backward (BF16): promote to f32, route upstream to argmax (fair-share ties), narrow grad."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_max_to_backward_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(x, output_shape)"
    - name: upstream
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out_shape"
  op_params:
    variant: ReduceMaxToBackward
    fields:
      input_shape:  { kind: "Vec<usize>", constraint: "product == x.elem_count == out.elem_count" }
      output_shape: { kind: "Vec<usize>", constraint: "product == upstream.elem_count; left-pad-1 to input rank; each axis == input size OR 1" }

return:
  outputs:
    - name: grad_x
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
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
  flops: "in_elems"
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # dtype_bytes = 2 (bf16 I/O)
  memory: { device_bytes: 0, host_bytes: "(5 * in_elems + 2 * out_elems) * 4", disk_bytes: 0 }   # f32 promotion scratch (xv32,uv32,outv32) + impl scratch

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 I/O; whole backward promoted to f32 then narrowed on store; upstream / tie_count fair-share; counts clamped >= 1."

determinism: same_hardware_bitwise
```

---

## reduce_max_to_backward_f16  (ReduceMaxTo backward — F16)

`reduce_max_to_backward_f16` (`byte_kernels.rs:8107`, macro `reduce_max_to_backward_half!`:8073).
f16 I/O; whole backward promoted to f32 then narrowed on store (same as the bf16 entry).

```fkc
kernel: reduce_max_to_backward_f16
op_kind: ReduceMaxToBackward
blurb: "ReduceMaxTo backward (F16): promote to f32, route upstream to argmax (fair-share ties), narrow grad."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::reduce_max_to_backward_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(x, output_shape)"
    - name: upstream
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out_shape"
  op_params:
    variant: ReduceMaxToBackward
    fields:
      input_shape:  { kind: "Vec<usize>", constraint: "product == x.elem_count == out.elem_count" }
      output_shape: { kind: "Vec<usize>", constraint: "product == upstream.elem_count; left-pad-1 to input rank; each axis == input size OR 1" }

return:
  outputs:
    - name: grad_x
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
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
  flops: "in_elems"
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # dtype_bytes = 2 (f16 I/O)
  memory: { device_bytes: 0, host_bytes: "(5 * in_elems + 2 * out_elems) * 4", disk_bytes: 0 }   # f32 promotion scratch + impl scratch

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O; whole backward promoted to f32 then narrowed on store; upstream / tie_count fair-share; counts clamped >= 1."

determinism: same_hardware_bitwise
```

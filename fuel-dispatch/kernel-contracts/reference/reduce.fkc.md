---
fkc_version: 1
provider:
  name: fuel-reference-backend
  backend: Cpu                                       # the oracle is a pure-Rust CPU implementation → BackendId::Cpu
  kernel_source: "reference-oracle"                  # the BindingEntry.kernel_source tag (FuelNative-class, §4.11)
  link_registry: fuel_reference_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"                      # provider build id, folded into kernel_revision_hash
---

# fuel-reference-backend — reduce family kernel contracts

Correctness-oracle reductions from `fuel-reference-backend/src/ops.rs`. The crate is the
pure-Rust **oracle** (`fuel-reference-backend/Cargo.toml`: "used only for validation, never for
production") against which every accelerated backend is judged; these contracts describe the
oracle's reduction kernels so the boundary advertises them like any other provider.

Three sub-families live here:

- **Reductions to scalar** (`sum_all` / `max_all` / `min_all` / `mean_all`, `ops.rs:341-387`) —
  fold every element to a **rank-0** tensor.
- **Reductions along one dim** (`sum_dim` / `max_dim` / `min_dim` / `mean_dim`, via `reduce_dim`
  `ops.rs:409`; arg-index `argmax_dim` / `argmin_dim` / `argindex_dim`, `ops.rs:495-563`) — the
  reduced dim is **removed** (no keepdim); arg-index ops emit a **U32** index tensor.
- **Reduce-to-shape** (`reduce_sum_to` / `reduce_max_to`, `ops.rs:854-961`; and
  `reduce_max_to_backward`, `ops.rs:979`) — the broadcast inverses (gradient of `broadcast_to`)
  that fold to a smaller broadcast-compatible target shape.

Shared facts (the cross-cutting oracle contract — inventory §"How to read this inventory"):

- **Layout: contiguous, zero-offset, row-major — always.** `RefTensor<T>` (`src/lib.rs:68`) is an
  `Arc<[T]>` + `Shape` carrying **no strides and no offset**. There is no `is_contiguous()` branch
  and no `StridedIndex` anywhere; callers must materialize any non-contiguous view into a fresh
  contiguous `RefTensor` before calling. Where an axis reduction needs stride math it computes it
  **internally** over the contiguous flat buffer via `row_major_strides` (`ops.rs:392`). Hence every
  operand is `{contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset:
  rejected, reverse_strides: rejected}`, and `awkward_layout_strategy = requires_contiguous` for
  every entry — the planner contiguizes awkward layouts (an `Op::Contiguize` FKC kernel, §4.3)
  first. None of these walk negative strides, so `reverse_strides: rejected` everywhere.
- **Output: fresh contiguous, fully overwritten.** Every kernel builds a fresh `Vec` and exits via
  `RefTensor::from_vec` — no input/output aliasing, no read of prior output content.
- **Accumulation is in the element type `T`, NOT widened to f32 for half.** Unlike the production
  CPU byte-kernels, the oracle folds in `T` directly (`fold(T::zero(), …)`, `T::neg_infinity()`,
  `T::infinity()`; `ops.rs:341-387`, `ops.rs:467-481`). For `bf16`/`f16` this means a per-element
  half-precision add — the oracle's job is to define the *expected* numerics of the reference
  algorithm, and the algorithm accumulates in the I/O dtype. This is the load-bearing precision
  fact for these contracts and is stated explicitly in each `precision.notes`.
- **Generic over `T: Float` → monomorphized to f32/f64/bf16/f16.** One source kernel, four dtype
  monomorphizations; each contract lists the full float dtype set on its operand (arg-index ops
  output U32). There are no separate per-dtype thunks for this family — the dtype list on the
  single contract section drives the per-dtype keys.

Cost provenance: every cost block is marked `judge_measured` — the Judge bootstraps it (§4.4).
A genuinely derivable FLOPs/bandwidth hint is given where the op admits one (a reduction reads
every input element once → `flops = n`, bandwidth-bound on `(n_in + n_out) * dtype_bytes`); no
other coefficients are fabricated.

---

## sum_all  (sum-reduce all elements to a scalar)

`sum_all<T: Float>` (`ops.rs:341`). Folds every element with IEEE addition from the identity
`T::zero()`, producing a **rank-0** tensor (`Shape::from_dims(&[])`, one element) of the same
dtype. An empty tensor sums to `0` (the additive identity). Accumulator is the element type `T`
(no f32 widening for half — the oracle defines the reference numerics of an in-dtype fold).
Single linear pass; memory-bandwidth bound. Contiguous, zero-offset; fresh-buffer overwrite.
Dispatched via `OpKind::SumReduce` over `OpParams::Reduce` with **all** dims reduced
(`dims = vec![]` / every axis; `keepdim: false`).

```fkc
kernel: sum_all
op_kind: SumReduce
blurb: "Sum-reduce all elements to a rank-0 scalar; in-dtype IEEE-add fold from zero."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::sum_all"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Reduce                  # OpParams::Reduce (primitive namespace); all dims reduced for the *_all variants
    fields:
      dims:    { kind: "Vec<usize>", constraint: "empty / all axes (full reduction)" }
      keepdim: { kind: bool, constraint: "== false (reduced dim removed; scalar output is rank-0)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: reduce(x, all_dims, keepdim=false)   # rank-0 ([], 1 element)
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
  provenance: judge_measured        # Judge bootstraps; the FLOPs/bandwidth hint below is the structural prior it refines
  class: reduction
  flops: "n"                        # one add per input element (n = x.elem_count)
  bytes_moved: "(n + 1) * dtype_bytes"   # read every input element, write one scalar
  memory: { device_bytes: 0, host_bytes: "dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic single-pass fold in a fixed flat order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "in-dtype accumulator (NOT widened to f32 for half — oracle reference numerics); IEEE add from T::zero(); empty -> 0; fixed fold order."

determinism: same_hardware_bitwise
```

---

## max_all  (max-reduce all elements to a scalar)

`max_all<T: Float>` (`ops.rs:352`). Walks every element keeping the running maximum (a `v > best`
compare seeded with `T::neg_infinity()`), producing a **rank-0** tensor of the same dtype. An
empty tensor returns `-inf` (the max identity). The compare is `>` (strict), so the first element
of a tie wins; `NaN` never satisfies `v > best`, so NaNs are skipped (NaN-as-missing) unless the
tensor is all-NaN (then `-inf` is returned). Single linear pass; bandwidth bound. Contiguous,
zero-offset; fresh overwrite. Dispatched via `OpKind::MaxReduce` over `OpParams::Reduce`, all dims.

```fkc
kernel: max_all
op_kind: MaxReduce
blurb: "Max-reduce all elements to a rank-0 scalar; strict > compare from -inf; empty -> -inf."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::max_all"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "empty / all axes (full reduction)" }
      keepdim: { kind: bool, constraint: "== false (scalar output is rank-0)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: reduce(x, all_dims, keepdim=false)   # rank-0
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
  flops: "n"                        # one compare per input element
  bytes_moved: "(n + 1) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # extremum is exact (no rounding); deterministic flat order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact extremum; seed -inf; strict > (first of a tie wins); NaN skipped (NaN-as-missing) unless all-NaN -> -inf; empty -> -inf."

determinism: same_hardware_bitwise
```

---

## min_all  (min-reduce all elements to a scalar)

`min_all<T: Float>` (`ops.rs:364`). The min-symmetric counterpart of `max_all`: keeps the running
minimum (a `v < best` compare seeded with `T::infinity()`), producing a **rank-0** tensor of the
same dtype. An empty tensor returns `+inf` (the min identity). NaN skipped (NaN never satisfies
`v < best`); all-NaN returns `+inf`. Single linear pass; bandwidth bound. Contiguous, zero-offset;
fresh overwrite. Dispatched via `OpKind::MinReduce` over `OpParams::Reduce`, all dims.

```fkc
kernel: min_all
op_kind: MinReduce
blurb: "Min-reduce all elements to a rank-0 scalar; strict < compare from +inf; empty -> +inf."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::min_all"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "empty / all axes (full reduction)" }
      keepdim: { kind: bool, constraint: "== false (scalar output is rank-0)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: reduce(x, all_dims, keepdim=false)   # rank-0
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
  flops: "n"                        # one compare per input element
  bytes_moved: "(n + 1) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact extremum; seed +inf; strict < (first of a tie wins); NaN skipped unless all-NaN -> +inf; empty -> +inf."

determinism: same_hardware_bitwise
```

---

## mean_all  (mean of all elements to a scalar)

`mean_all<T: Float>` (`ops.rs:376`). Sums every element (in-dtype IEEE add from `T::zero()`) then
divides by the element count `n`, producing a **rank-0** tensor of the same dtype. An empty tensor
returns `NaN` (the mean of zero samples is undefined; the kernel short-circuits to `T::nan()`).
Two logical operations over one input pass (fold then a single scalar divide). Bandwidth bound.
Contiguous, zero-offset; fresh overwrite. Dispatched via `OpKind::MeanReduce` over
`OpParams::Reduce`, all dims; the count divisor is the product of reduced dims.

```fkc
kernel: mean_all
op_kind: MeanReduce
blurb: "Mean of all elements to a rank-0 scalar; in-dtype sum/n; empty -> NaN."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::mean_all"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "empty / all axes (full reduction); divisor = product of reduced dims = n" }
      keepdim: { kind: bool, constraint: "== false (scalar output is rank-0)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: reduce(x, all_dims, keepdim=false)   # rank-0
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
  flops: "n + 1"                    # n adds + 1 divide
  bytes_moved: "(n + 1) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic fold then a single divide, fixed order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "in-dtype accumulator (NOT widened to f32 for half); sum from T::zero() then / n; empty -> NaN; fixed fold order."

determinism: same_hardware_bitwise
```

---

## sum_dim  (sum-reduce along one dim, dim removed)

`sum_dim<T: Float>` (`ops.rs:467`) — `reduce_dim` (`ops.rs:409`) with identity `T::zero()` and
`acc + v`. Reduces a single axis `dim`; the reduced dim is **removed** from the output (no
keepdim), so input `[a, b, c]` with `dim=1` yields `[a, c]`. The kernel allocates one accumulator
slot per output element, walks the input once projecting each input flat index to its output slot
via `row_major_strides` + unflatten (dropping the reduced coord), and folds with IEEE add. In-dtype
accumulator. Single input pass; bandwidth bound. `assert dim < rank`. Contiguous, zero-offset; fresh
overwrite. Dispatched via `OpKind::SumReduce` over `OpParams::Reduce { dims: vec![dim], keepdim:
false }`.

```fkc
kernel: sum_dim
op_kind: SumReduce
blurb: "Sum-reduce along one dim (dim removed, no keepdim); in-dtype IEEE-add fold from zero."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::sum_dim"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "single dim [dim]; each dim < rank" }
      keepdim: { kind: bool, constraint: "== false (reduced dim removed from output shape)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: reduce(x, dims, keepdim=false)   # input shape with `dim` dropped; symbolic non-reduced axes preserved
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
  flops: "n"                        # one add per input element (n = x.elem_count)
  bytes_moved: "(n + out_elems) * dtype_bytes"   # read every input, write every output slot
  memory: { device_bytes: 0, host_bytes: "out_elems * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "in-dtype accumulator (NOT widened to f32 for half); IEEE add from T::zero(); fixed projection/fold order."

determinism: same_hardware_bitwise
```

---

## max_dim  (max-reduce along one dim, dim removed)

`max_dim<T: Float>` (`ops.rs:473`) — `reduce_dim` with identity `T::neg_infinity()` and the fold
`if v > acc { v } else { acc }`. Reduces a single axis; reduced dim removed (no keepdim). Strict `>`
compare (first of a tie wins); NaN never beats the accumulator (NaN-as-missing) unless a window is
all-NaN (then the `-inf` seed survives). One accumulator slot per output element, one input pass via
projected stride math. In-dtype extremum (exact for any representable value). `assert dim < rank`.
Contiguous, zero-offset; fresh overwrite. Dispatched via `OpKind::MaxReduce` over `OpParams::Reduce
{ dims: vec![dim], keepdim: false }`.

```fkc
kernel: max_dim
op_kind: MaxReduce
blurb: "Max-reduce along one dim (dim removed); strict > compare from -inf; NaN-as-missing."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::max_dim"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "single dim [dim]; each dim < rank" }
      keepdim: { kind: bool, constraint: "== false (reduced dim removed)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: reduce(x, dims, keepdim=false)   # input shape with `dim` dropped
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
  flops: "n"                        # one compare per input element
  bytes_moved: "(n + out_elems) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "out_elems * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # extremum is exact; deterministic projection/fold order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact extremum; seed -inf; strict > (first of a tie wins); NaN-as-missing; all-NaN window -> -inf."

determinism: same_hardware_bitwise
```

---

## min_dim  (min-reduce along one dim, dim removed)

`min_dim<T: Float>` (`ops.rs:479`) — `reduce_dim` with identity `T::infinity()` and the fold
`if v < acc { v } else { acc }`. The min-symmetric counterpart of `max_dim`: reduced dim removed,
strict `<` compare (first of a tie wins), NaN-as-missing (all-NaN window keeps the `+inf` seed). One
accumulator slot per output element, one input pass via projected stride math. In-dtype extremum.
`assert dim < rank`. Contiguous, zero-offset; fresh overwrite. Dispatched via `OpKind::MinReduce`
over `OpParams::Reduce { dims: vec![dim], keepdim: false }`.

```fkc
kernel: min_dim
op_kind: MinReduce
blurb: "Min-reduce along one dim (dim removed); strict < compare from +inf; NaN-as-missing."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::min_dim"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "single dim [dim]; each dim < rank" }
      keepdim: { kind: bool, constraint: "== false (reduced dim removed)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: reduce(x, dims, keepdim=false)   # input shape with `dim` dropped
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
  flops: "n"                        # one compare per input element
  bytes_moved: "(n + out_elems) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "out_elems * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact extremum; seed +inf; strict < (first of a tie wins); NaN-as-missing; all-NaN window -> +inf."

determinism: same_hardware_bitwise
```

---

## mean_dim  (mean along one dim, dim removed)

`mean_dim<T: Float>` (`ops.rs:485`) — **two passes**: `sum_dim(x, dim)` then an elementwise divide
of every output slot by the reduced extent `dims[dim]` (coerced to `T` via `cst`). Reduced dim
removed (no keepdim). In-dtype accumulator (the divisor is the in-dtype cast of the reduced-axis
length). One full input pass (the sum) plus one output pass (the divide); bandwidth bound.
Contiguous, zero-offset; fresh overwrite. Dispatched via `OpKind::MeanReduce` over `OpParams::Reduce
{ dims: vec![dim], keepdim: false }`; the divisor is the product of reduced dims.

```fkc
kernel: mean_dim
op_kind: MeanReduce
blurb: "Mean along one dim (dim removed); two-pass sum_dim then /dims[dim]; in-dtype."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::mean_dim"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "single dim [dim]; each dim < rank; divisor = product of reduced dims = dims[dim]" }
      keepdim: { kind: bool, constraint: "== false (reduced dim removed)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: reduce(x, dims, keepdim=false)   # input shape with `dim` dropped
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
  flops: "n + out_elems"            # n adds (sum_dim pass) + one divide per output slot
  bytes_moved: "(n + 2 * out_elems) * dtype_bytes"   # read input + (write+read) the sum slots for the divide pass
  memory: { device_bytes: 0, host_bytes: "out_elems * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic two-pass (sum_dim then divide), fixed order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "in-dtype accumulator (NOT widened to f32 for half); sum_dim then / dims[dim] (cst<T>); two passes; fixed order."

determinism: same_hardware_bitwise
```

---

## argmax_dim  (index of the max along one dim → U32)

`argmax_dim<T: Float>` (`ops.rs:495`) — thin wrapper over `argindex_dim(x, dim, is_max=true)`
(`ops.rs:506`). Returns the **U32** index of the maximum element along `dim` (output dtype differs
from input); the reduced dim is removed. Algorithm: for each output slot, seed `best_idx=0` /
`best_val = x[base]`, then scan `k in 1..reduced_size`, updating on a strict `v > best_val`
compare — so **ties resolve to the smallest index** (PyTorch convention; a later equal value never
displaces an earlier winner). One pass per output slot over the reduced axis via projected stride
math. `assert dim < rank`. Contiguous, zero-offset; fresh overwrite. Dispatched via
`OpKind::ArgMaxDim`, key dtypes `[in: T, out: U32]`, over `OpParams::Reduce { dims: vec![dim],
keepdim: false }`.

```fkc
kernel: argmax_dim
op_kind: ArgMaxDim
blurb: "Index (U32) of the max along one dim; dim removed; ties -> smallest index (PyTorch)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::argmax_dim"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "single dim [dim]; each dim < rank" }
      keepdim: { kind: bool, constraint: "== false (reduced dim removed)" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)                        # index output, NOT passthrough (differs from input dtype)
      shape_rule: reduce(x, dims, keepdim=false)    # input shape with `dim` dropped
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32                        # U32 index store

cost:
  provenance: judge_measured
  class: reduction
  flops: "n"                        # one compare per scanned element (n = x.elem_count)
  bytes_moved: "n * dtype_bytes + out_elems * 4"     # read every input element; write U32 index per output slot
  memory: { device_bytes: 0, host_bytes: "out_elems * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # integer index; deterministic scan, exact ties-to-smallest rule
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact U32 index; strict > scan from index 0 -> ties resolve to smallest index (PyTorch); deterministic scan order."

determinism: bitwise
```

---

## argmin_dim  (index of the min along one dim → U32)

`argmin_dim<T: Float>` (`ops.rs:501`) — thin wrapper over `argindex_dim(x, dim, is_max=false)`.
The min-symmetric counterpart of `argmax_dim`: returns the **U32** index of the minimum along
`dim`, reduced dim removed. Strict `v < best_val` compare, so **ties resolve to the smallest
index**. One pass per output slot over the reduced axis. `assert dim < rank`. Contiguous,
zero-offset; fresh overwrite. Dispatched via `OpKind::ArgMinDim`, key dtypes `[in: T, out: U32]`,
over `OpParams::Reduce { dims: vec![dim], keepdim: false }`.

```fkc
kernel: argmin_dim
op_kind: ArgMinDim
blurb: "Index (U32) of the min along one dim; dim removed; ties -> smallest index (PyTorch)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::argmin_dim"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "single dim [dim]; each dim < rank" }
      keepdim: { kind: bool, constraint: "== false (reduced dim removed)" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)                        # index output, NOT passthrough
      shape_rule: reduce(x, dims, keepdim=false)    # input shape with `dim` dropped
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n"                        # one compare per scanned element
  bytes_moved: "n * dtype_bytes + out_elems * 4"
  memory: { device_bytes: 0, host_bytes: "out_elems * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact U32 index; strict < scan from index 0 -> ties resolve to smallest index (PyTorch); deterministic scan order."

determinism: bitwise
```

---

## argindex_dim  (shared arg-index reduction core → U32)

`argindex_dim<T: Float>` (`ops.rs:506`) — the shared implementation backing both `argmax_dim`
(`is_max=true`) and `argmin_dim` (`is_max=false`). It is the algorithm, not a distinct dispatch
entry-point: a private `is_max: bool` selects the compare direction (`v > best_val` for max,
`v < best_val` for min); both directions use a strict compare so ties resolve to the **smallest
index** (PyTorch). Output dtype is **U32**, reduced dim removed. Per output slot it builds the seed
input multi-index (reduced coord = 0), reads `best_val = x[base]`, then scans `k in 1..reduced_size`
via `base_flat + k * in_strides[dim]`, updating `best_idx` on a strict win. `assert dim < rank`.
Contiguous, zero-offset; fresh overwrite. Documented here once so the public `argmax_dim` /
`argmin_dim` sections need not restate the scan; it is keyed by the same `OpParams::Reduce { dims:
vec![dim], keepdim: false }` carrier as those two, with the `is_max` direction selected by the
dispatching `OpKind` (`ArgMaxDim` / `ArgMinDim`).

```fkc
kernel: argindex_dim
op_kind: ArgMaxDim                # shared core; the concrete dispatch OpKinds are ArgMaxDim (is_max=true) / ArgMinDim (is_max=false)
blurb: "Shared arg-index core (U32) along one dim; strict compare -> ties resolve to smallest index."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::argindex_dim"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Reduce
    fields:
      dims:    { kind: "Vec<usize>", constraint: "single dim [dim]; each dim < rank" }
      keepdim: { kind: bool, constraint: "== false (reduced dim removed)" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)                        # index output, NOT passthrough
      shape_rule: reduce(x, dims, keepdim=false)    # input shape with `dim` dropped
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n"                        # one compare per scanned element
  bytes_moved: "n * dtype_bytes + out_elems * 4"
  memory: { device_bytes: 0, host_bytes: "out_elems * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact U32 index; is_max selects > vs < (strict either way) -> ties resolve to smallest index (PyTorch); deterministic scan order."

determinism: bitwise
```

---

## reduce_sum_to  (sum-reduce to a broadcast-compatible target shape)

`reduce_sum_to<T: Float>` (`ops.rs:854`). The backward of a forward `broadcast_to`: sum-reduces the
input down to a smaller, broadcast-compatible `target` shape. Alignment: left-pad `target` with 1s
to the source rank, and assert every padded axis equals the source size (axis carries through) or
`1` (axis collapsed/summed); any other value is a contract violation (`ops.rs:866`). Output is the
target shape, init `T::zero()`, accumulated by IEEE add. One accumulator slot per output element,
one input pass projecting each source flat index to its output slot (collapsed axes clamp to coord
0) via `row_major_strides`. In-dtype accumulator. Bandwidth bound. Contiguous, zero-offset; fresh
overwrite. Dispatched via `OpKind::ReduceSumTo` over `OpParams::ReduceSumTo { input_shape,
output_shape }`.

```fkc
kernel: reduce_sum_to
op_kind: ReduceSumTo
blurb: "Sum-reduce to a broadcast-compatible target shape (broadcast backward); in-dtype add from zero."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::reduce_sum_to"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(input, output_shape)"   # each left-padded output axis == input size OR 1
  op_params:
    variant: ReduceSumTo            # OpParams::ReduceSumTo (primitive namespace)
    fields:
      input_shape:  { kind: "Vec<usize>", constraint: "product == input.elem_count" }
      output_shape: { kind: "Vec<usize>", constraint: "rank <= input rank; left-pad-1 to input rank; each axis == input size OR 1" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, output_shape, keepdim)   # = output_shape; symbolic carried-through axes preserved
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
  flops: "in_elems"                 # one add per input element (in_elems = product(input_shape))
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"   # read every input + write every output slot
  memory: { device_bytes: 0, host_bytes: "out_elems * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "in-dtype accumulator (NOT widened to f32 for half); IEEE add from T::zero(); fixed projection/fold order."

determinism: same_hardware_bitwise
```

---

## reduce_max_to  (max-reduce to a broadcast-compatible target shape)

`reduce_max_to<T: Float>` (`ops.rs:911`). The max-symmetric counterpart of `reduce_sum_to`: same
left-pad / `== size OR 1` axis-alignment rule, but folds with `max` instead of `+`. Output is the
target shape, init `T::neg_infinity()`, updated on a strict `src > out` compare (NaN-as-missing; a
fully-collapsed window of all-NaN keeps the `-inf` seed). One accumulator slot per output element,
one input pass with collapsed axes clamping to coord 0. In-dtype extremum (exact). Bandwidth bound.
Contiguous, zero-offset; fresh overwrite. Dispatched via `OpKind::ReduceMaxTo` over
`OpParams::ReduceMaxTo { input_shape, output_shape }`.

```fkc
kernel: reduce_max_to
op_kind: ReduceMaxTo
blurb: "Max-reduce to a broadcast-compatible target shape; strict > from -inf; NaN-as-missing."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::reduce_max_to"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(input, output_shape)"   # each left-padded output axis == input size OR 1
  op_params:
    variant: ReduceMaxTo
    fields:
      input_shape:  { kind: "Vec<usize>", constraint: "product == input.elem_count" }
      output_shape: { kind: "Vec<usize>", constraint: "rank <= input rank; left-pad-1 to input rank; each axis == input size OR 1" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: reduce(input, output_shape, keepdim)   # = output_shape
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
  flops: "in_elems"                 # one compare per input element
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "out_elems * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # extremum is exact; deterministic projection/fold order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact extremum; seed -inf; strict > compare; NaN-as-missing; all-NaN collapsed window -> -inf."

determinism: same_hardware_bitwise
```

---

## reduce_max_to_backward  (ReduceMaxTo backward — route upstream to argmax, fair-share ties)

`reduce_max_to_backward<T: Float>` (`ops.rs:979`). Backward of `Op::ReduceMaxTo`: takes **three**
inputs — `x` (the original forward input, shape `S_in`), `upstream` (the upstream gradient, shape
`S_target` == the forward target shape, broadcast-compatible into `S_in`), and the `target` shape
(carried in `op_params`) — and returns `grad_x` of shape `S_in`. Routes the upstream gradient back
to the position(s) where `x` equals its per-window max, splitting equally across ties (fair-share
subgradient). Algorithm (`ops.rs:987-1032`): (1) recompute `y = reduce_max_to(x, target)`; (2)
broadcast `y` to `S_in` and build a mask `x == broadcast(y)`; (3) `count = reduce_sum_to(mask,
target)` (ties per output cell); (4) `count_safe = max(count, 1)` (defensive 0/0 guard, e.g. empty
window / NaN); (5) `scaled = upstream / count_safe`; (6) `grad_x = broadcast(scaled) * mask`.
In-dtype throughout (no f32 promotion in the oracle). Multi-pass (recompute max + mask build + tie
sum + scale + broadcast-gate), all O(in_elems) compute over O(in_elems) writes; allocates `max_b` /
`mask` / `scaled_b` scratch of input size. Contiguous, zero-offset; fresh overwrite (no input
aliasing). Dispatched via `OpKind::ReduceMaxToBackward` over `OpParams::ReduceMaxToBackward {
input_shape, output_shape }`.

```fkc
kernel: reduce_max_to_backward
op_kind: ReduceMaxToBackward
blurb: "ReduceMaxTo backward: route upstream to argmax positions, fair-share split on ties; in-dtype."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::reduce_max_to_backward"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "reduce(x, output_shape)"   # x == input_shape; output_shape reduces x along collapsed axes
    - name: upstream
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out_shape"          # upstream.shape == S_target == output_shape (broadcast-compatible into S_in)
  op_params:
    variant: ReduceMaxToBackward
    fields:
      input_shape:  { kind: "Vec<usize>", constraint: "product == x.elem_count == grad_x.elem_count (S_in)" }
      output_shape: { kind: "Vec<usize>", constraint: "product == upstream.elem_count; left-pad-1 to input rank; each axis == input size OR 1 (S_target)" }

return:
  outputs:
    - name: grad_x
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)                 # grad of x = input_shape (S_in)
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
  # multi-pass: recompute max (read in_elems), build mask (in_elems), sum-to ties (in_elems),
  # scale upstream (out_elems), broadcast + gate (in_elems). O(in_elems) compute, O(in_elems) write.
  flops: "in_elems"                 # in_elems = product(input_shape) = product(S_in)
  bytes_moved: "(in_elems + out_elems) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "(3 * in_elems + out_elems) * dtype_bytes", disk_bytes: 0 }   # max_b + mask + scaled_b scratch + grad

precision:
  bit_stable_on_same_hardware: true   # deterministic multi-pass in a fixed order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "in-dtype throughout (NOT widened to f32 for half — oracle reference numerics); upstream / tie_count fair-share; counts clamped >= 1; equality mask via ==; deterministic order."

determinism: same_hardware_bitwise
```

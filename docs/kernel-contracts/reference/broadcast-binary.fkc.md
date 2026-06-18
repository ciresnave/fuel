---
fkc_version: 1
provider:
  name: fuel-reference-backend
  backend: Cpu                                       # the oracle realizes on the host (RefTensor = host Vec/Arc); maps to BackendId::Cpu
  kernel_source: "reference-oracle"                  # the BindingEntry.kernel_source tag (the pure-Rust correctness oracle)
  link_registry: fuel_reference_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"                      # provider build id, folded into kernel_revision_hash
---

# fuel-reference-backend — broadcasting elementwise-binary kernel contracts

Broadcast-aware element-wise binary arithmetic from the reference oracle's op surface
(`fuel-reference-backend/src/ops.rs`): `broadcast_add` / `broadcast_sub` / `broadcast_mul` /
`broadcast_div` (`ops.rs:2179-2196`), all driven by the shared `broadcast_binary`
(`ops.rs:2142`) using NumPy/PyTorch right-aligned broadcast rules (`broadcast_shape`
`ops.rs:2094`, `broadcast_src_flat` `ops.rs:2121`). Each is generic over `T: num_traits::Float`
and is monomorphized to `{F32, F64, BF16, F16}` (one kernel, four dtypes) — the index dtype
`U32` is **not** admitted (the `Float` bound excludes it).

**Crate-wide layout invariant (the load-bearing fact for this family).** `RefTensor<T>`
(`lib.rs:68`) is *always* a contiguous, row-major buffer with **no strides and no offset**. The
two input *buffers* a broadcast-binary kernel reads are therefore physically contiguous and
zero-offset; the broadcast is a **logical** relation between two differently-shaped contiguous
tensors, and the kernel computes the right-align / size-1-collapse / per-output-element unflatten
*internally* over those contiguous buffers (`broadcast_src_flat`, via `row_major_strides`). It
allocates a fresh contiguous output of the broadcast shape (`RefTensor::from_vec`) and fills it
element by element. This is why every operand below declares `contiguous: required` /
`broadcast_stride0: rejected` at the data layer and the kernel's `awkward_layout_strategy` is
`requires_contiguous`: the kernel never reads a stride-0 broadcast axis at the data layer — both
input buffers are genuinely contiguous and zero-offset, and the broadcast is purely **internal
index math** (`broadcast_src_flat` unflattens each output position back to a source flat index), not
a strided walk. There is therefore no operand to contiguize and no stride-0 read to declare; the
expansion arithmetic is folded into the kernel's own declared `bytes_moved` (§4.3). It is **not** a
silent materialization, and the planner MUST NOT insert a separate `Op::BroadcastTo`/`Op::Contiguize`
in front of it.

**Half-precision numerics differ from the CPU byte-kernel family — be precise here.** The
reference `broadcast_binary` applies the arithmetic closure directly in the element type `T`
(`|x, y| x + y` etc. over `T: Float`); for `BF16`/`F16` the add/sub/mul/div is computed **in the
half type itself**, *not* widened to f32 and narrowed on store. This is a deliberate property of
the oracle (it is the correctness reference for the exact-as-written semantics), and it is the
distinguishing precision fact versus `fuel-cpu-backend`'s widen-to-f32 half path.

**Output dtype = input dtype.** No comparison/U8 form lives here (the reference oracle's
`AnyRefTensor` has no U8 variant; comparisons have no reference kernel — inventory bottom). Output
shape = the NumPy broadcast of the two inputs; output is a fresh contiguous buffer with no
aliasing of either input. `broadcast_shape` **panics** on incompatible shapes — acceptable because
the crate is a pure oracle with **no production path** (inventory "How to read", crate-wide note);
on Fuel's production surface the equivalent shape mismatch is rejected at graph-build time.

> **Dispatch-key note (faithful mapping).** Fuel's dispatch surface has **no
> `OpKind::BroadcastAdd`/`Sub`/`Mul`/`Div`** — the broadcasting binary semantic dispatches under
> the *same* `OpKind` as the same-shape binary (`AddElementwise` / `SubElementwise` /
> `MulElementwise` / `DivElementwise`, `fuel-core-types/src/dispatch.rs:56-62`). What distinguishes
> a broadcast-binary contract from the same-shape `add`/`sub`/`mul`/`div` contract at that key is
> the **accept shape predicate**: a broadcast contract admits a `broadcast_to`-related operand pair
> (`shape_constraint: broadcast_to`), whereas the same-shape contract requires exact-equal dims.
> Both declare `awkward_layout_strategy: requires_contiguous` with `broadcast_stride0: rejected` —
> the broadcast is internal index math over contiguous buffers, not a stride-0 data read. Both are
> legal sibling alternatives at the key (§12.5); the planner picks by admissibility against the
> concrete operand shapes.

## broadcast_add  (out = a + b, NumPy broadcast)

Broadcast-aware element-wise addition `out[k] = a[idx_a(k)] + b[idx_b(k)]`, where the output index
`k` ranges over the NumPy-broadcast shape of `a` and `b`, and `idx_a`/`idx_b` are the right-aligned,
size-1-collapsed source flat indices (`broadcast_src_flat`, `ops.rs:2121`). Both input buffers are
contiguous zero-offset row-major `RefTensor`s; the broadcast/stride math is computed internally over
those contiguous buffers via `row_major_strides`, and a fresh contiguous output of the broadcast
shape is allocated and filled per element. Generic over `T: Float` → `{F32, F64, BF16, F16}`. For
`BF16`/`F16` the addition is performed **in the half type itself** (no widen-to-f32), the oracle's
exact-as-written semantics. IEEE inf/NaN propagate. Fully overwrites the fresh output; no aliasing.
Known limitation: oracle-only — `broadcast_shape` panics on shapes that do not broadcast (no
production path; the production graph rejects incompatible shapes at build time).

```fkc
kernel: broadcast_add
op_kind: AddElementwise          # NO OpKind::BroadcastAdd exists; broadcasting binary dispatches under AddElementwise (dispatch.rs:56)
blurb: "Broadcast-aware elementwise a + b (NumPy rules); contiguous buffers, broadcast computed internally; half in-type."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::broadcast_add"   # one generic fn over T: Float; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=b      # NumPy-broadcast-compatible with b; output = broadcast(a, b)
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: broadcast_to=a
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)            # a.dtype == b.dtype (monomorphized over one T); output is that dtype
      shape_rule: broadcast(a, b)           # NumPy broadcast shape (§5.2)
      layout_guarantee: contiguous          # fresh from_vec buffer
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # input buffers are ALWAYS physically contiguous zero-offset; the broadcast is internal index math, not a strided walk — no operand to contiguize (§10.4 coherence)
  fast_paths:
    - { when: "all_inputs_contiguous", note: "input buffers are always contiguous; the broadcast expansion still runs" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured        # Judge bootstraps; bandwidth-bound elementwise over the OUTPUT element count (hint below)
  class: cheap_elementwise
  flops: "n"                        # one add per OUTPUT element; n = product of broadcast (output) dims
  bytes_moved: "n * dtype_bytes + a_count * dtype_bytes + b_count * dtype_bytes"   # write n out; read each source once per its own size (a_count/b_count = product of a/b dims)
  overhead_ns: ~                    # judge_measured
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                    # CPU/host primitive: importer applies PRIMITIVE_DETERMINISTIC_CPU family default (§12.4)
  notes: "f32/f64 native; bf16/f16 add IN the half type (NOT widened to f32 — oracle exact semantics); IEEE inf/NaN; deterministic positional fill; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## broadcast_sub  (out = a - b, NumPy broadcast)

Broadcast-aware element-wise subtraction `out[k] = a[idx_a(k)] - b[idx_b(k)]` over the NumPy-broadcast
shape of `a` and `b`. Identical structure to `broadcast_add` (same `broadcast_binary` driver,
`ops.rs:2142`) with the `|x, y| x - y` closure (`ops.rs:2184`). Contiguous zero-offset input buffers;
broadcast math internal; fresh contiguous output; `{F32, F64, BF16, F16}`. Half subtraction is
computed **in the half type** (no widen-to-f32). IEEE inf/NaN. Full overwrite, no aliasing.
Oracle-only: `broadcast_shape` panics on non-broadcastable shapes (no production path).

```fkc
kernel: broadcast_sub
op_kind: SubElementwise           # NO OpKind::BroadcastSub; dispatches under SubElementwise (dispatch.rs:58)
blurb: "Broadcast-aware elementwise a - b (NumPy rules); contiguous buffers, broadcast internal; half in-type."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::broadcast_sub"
kernel_revision_hash: auto

accept:
  inputs:
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
      dtype_rule: passthrough(a)
      shape_rule: broadcast(a, b)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # contiguous buffers; broadcast is internal index math, not a strided walk (§10.4 coherence)
  fast_paths:
    - { when: "all_inputs_contiguous", note: "input buffers always contiguous; broadcast expansion still runs" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                        # one subtract per OUTPUT element
  bytes_moved: "n * dtype_bytes + a_count * dtype_bytes + b_count * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32/f64 native; bf16/f16 subtract IN the half type (NOT widened to f32); IEEE inf/NaN; deterministic positional fill; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## broadcast_mul  (out = a * b, NumPy broadcast)

Broadcast-aware element-wise multiplication `out[k] = a[idx_a(k)] * b[idx_b(k)]` over the
NumPy-broadcast shape of `a` and `b`. Same `broadcast_binary` driver with `|x, y| x * y`
(`ops.rs:2189`). The col-against-row broadcast of two vectors produces an outer product (inventory
test `broadcast_mul_col_against_row_makes_outer_product`). Contiguous zero-offset input buffers;
broadcast math internal; fresh contiguous output; `{F32, F64, BF16, F16}`. Half multiplication is
computed **in the half type** (no widen-to-f32). IEEE inf/NaN. Full overwrite, no aliasing.
Oracle-only: panics on non-broadcastable shapes (no production path).

```fkc
kernel: broadcast_mul
op_kind: MulElementwise           # NO OpKind::BroadcastMul; dispatches under MulElementwise (dispatch.rs:60)
blurb: "Broadcast-aware elementwise a * b (NumPy rules); contiguous buffers, broadcast internal; half in-type."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::broadcast_mul"
kernel_revision_hash: auto

accept:
  inputs:
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
      dtype_rule: passthrough(a)
      shape_rule: broadcast(a, b)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # contiguous buffers; broadcast is internal index math, not a strided walk (§10.4 coherence)
  fast_paths:
    - { when: "all_inputs_contiguous", note: "input buffers always contiguous; broadcast expansion still runs" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                        # one multiply per OUTPUT element
  bytes_moved: "n * dtype_bytes + a_count * dtype_bytes + b_count * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32/f64 native; bf16/f16 multiply IN the half type (NOT widened to f32); IEEE inf/NaN; deterministic positional fill; bit-stable same hardware."

determinism: same_hardware_bitwise
```

## broadcast_div  (out = a / b, NumPy broadcast)

Broadcast-aware element-wise division `out[k] = a[idx_a(k)] / b[idx_b(k)]` over the NumPy-broadcast
shape of `a` and `b`. Same `broadcast_binary` driver with `|x, y| x / y` (`ops.rs:2194`). Contiguous
zero-offset input buffers; broadcast math internal; fresh contiguous output; `{F32, F64, BF16, F16}`.
Half division is computed **in the half type** (no widen-to-f32). Division by zero follows IEEE
(`x/0 → ±inf`, `0/0 → NaN`); inf/NaN propagate. Full overwrite, no aliasing. Oracle-only: panics on
non-broadcastable shapes (no production path).

```fkc
kernel: broadcast_div
op_kind: DivElementwise           # NO OpKind::BroadcastDiv; dispatches under DivElementwise (dispatch.rs:62)
blurb: "Broadcast-aware elementwise a / b (NumPy rules); contiguous buffers, broadcast internal; half in-type; IEEE div."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::broadcast_div"
kernel_revision_hash: auto

accept:
  inputs:
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
      dtype_rule: passthrough(a)
      shape_rule: broadcast(a, b)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # contiguous buffers; broadcast is internal index math, not a strided walk (§10.4 coherence)
  fast_paths:
    - { when: "all_inputs_contiguous", note: "input buffers always contiguous; broadcast expansion still runs" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                        # one divide per OUTPUT element
  bytes_moved: "n * dtype_bytes + a_count * dtype_bytes + b_count * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32/f64 native; bf16/f16 divide IN the half type (NOT widened to f32); IEEE x/0 -> +/-inf, 0/0 -> NaN; deterministic positional fill; bit-stable same hardware."

determinism: same_hardware_bitwise
```

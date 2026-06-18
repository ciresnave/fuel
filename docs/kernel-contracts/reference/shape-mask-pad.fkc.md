---
fkc_version: 1
provider:
  name: fuel-reference-backend
  backend: Cpu                       # the reference oracle runs host-side (BackendId::Cpu)
  kernel_source: "reference-oracle"  # the BindingEntry.kernel_source tag
  link_registry: fuel_reference_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-reference-backend — shape / mask / pad kernel contracts

Pure-Rust, correctness-first oracle kernels for triangular masking, masked-fill, the three
padding modes + pad-backward, reshape, cumulative-sum, flip, roll, concat, slice, and the forward
broadcast (`broadcast_to`).

**Crate-wide layout invariant (the single most load-bearing fact in this file).** `RefTensor<T>`
(`fuel-reference-backend/src/lib.rs:68`) is *always* a contiguous, row-major `Vec`/`Arc<[T]>` plus a
`Shape`. It carries **no strides and no offset**. Every kernel below is therefore
**contiguous-only, zero-offset** at the data layer — there is no `StridedIndex`, no broadcast
walk, no negative-stride (`reverse_strides`) path anywhere. Each kernel declares
`awkward_layout_strategy: requires_contiguous`, so the planner must insert an `Op::Contiguize`
(itself an FKC kernel, §4.3) for any non-contiguous producer and sum its cost (§4.4). Where an op
computes outer/dim/inner stride *math internally over a contiguous buffer* (cumsum/flip/roll/triu/
tril/pad/concat/slice all do this via a flat-3-axis or row-major walk) the input buffer is still a
dense contiguous `RefTensor`; that internal math is not a strided-input capability.

`Op::Flip` is explicitly a **materializing** op in this crate, not a negative-stride view:
`fuel-core-types/src/dispatch.rs:257-260` records "Layout strides are unsigned; the negative-stride
view path requires a wider stride representation that's a separate scope." So `flip` here copies to
a fresh contiguous buffer and declares `reverse_strides: rejected` on both accept and return —
the zero-copy negative-stride flip of §4.1.1 is a *different* (GPU/strided) backend's capability,
not this oracle's.

All costs in this file are marked **`provenance: declared`** (author priors the Judge later refines,
§4.4): these are bandwidth-bound byte-movement kernels, so each block carries an honest derivable
byte-count `bytes_moved` / `flops` formula together with an authored absolute `overhead_ns` launch
prior — an authored absolute constant belongs under `declared`, not `judge_measured` (the
match-content rule). The metadata-only `reshape` is the one genuinely-free op: its all-zero
coefficients (including `overhead_ns: 0`) are a true-zero declaration, not a placeholder (§10.8a).

---

## triu  (upper-triangular mask along the last two dims)

Zero out the strictly-lower-triangular region of every `[rows, cols]` matrix in a batched tensor,
keeping element `[i, j]` iff `j >= i + diagonal`. `diagonal` is a signed `i64` (0 = main diagonal,
positive shifts the kept band up/right, negative shifts it down/left). The leading dims are batched
(`batch_count = product of all-but-last-two`). Dtype-agnostic at the byte level — the kernel selects
each output element from the source byte or from a zero-init slot, so no arithmetic and no dtype
widening occurs; this is why `u32` index tensors are admissible alongside the float dtypes. Input is
a dense contiguous `[..batch.., rows, cols]` buffer; the per-position keep/zero decision is computed
from the unraveled `(i, j)` coordinates over `row_major` strides internally. Output is a fresh
contiguous buffer of the same shape and dtype. Numerically exact (bit-identical to the input for
kept positions; exact zero for masked positions). Limitation: contiguous-only — any strided/offset
producer is contiguized by the planner first; the last two dims define the triangle, so rank must be
≥ 2.

Source: `fuel-reference-backend/src/ops.rs:2847`; exec arm `src/exec.rs:416`.

```fkc
kernel: triu
op_kind: Triu
blurb: "Upper-triangular mask over the last two dims; keep j>=i+diagonal else 0; byte-select, no math."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::triu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=8                     # last two dims are the triangle; leading dims batched
      shape_constraint: same_as=out
  op_params:
    variant: Triangular              # OpParams::Triangular (shared with tril)
    fields:
      batch_count: { kind: usize, note: "product of leading dims" }
      rows: { kind: usize }
      cols: { kind: usize }
      diagonal: { kind: i64, note: "0=main; +up/right, -down/left" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8         # byte-level select; dtype-agnostic

cost:
  provenance: declared               # author prior; Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                         # pure byte select; no arithmetic
  bytes_moved: "2 * n * dtype_bytes" # read x, write out (n = output elements)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true  # byte-exact select; kept bytes copied verbatim, masked = 0
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact: kept positions bit-identical to input, masked positions exact zero; no arithmetic, all dtypes incl. u32."

determinism: bitwise                 # pure byte shuffle/select; hardware-independent
```

---

## tril  (lower-triangular mask along the last two dims)

Mirror of `triu`: keep element `[i, j]` iff `j <= i + diagonal`, else zero. Same batched
`[..batch.., rows, cols]` contract, same signed `i64` `diagonal`, same byte-level select (dtype-
agnostic, `u32` admissible), same fresh-contiguous output. Shares `OpParams::Triangular` with
`triu`; the `OpKind` selects the direction. Numerically exact; contiguous-only.

Source: `fuel-reference-backend/src/ops.rs:2871`; exec arm `src/exec.rs:426`.

```fkc
kernel: tril
op_kind: Tril
blurb: "Lower-triangular mask over the last two dims; keep j<=i+diagonal else 0; byte-select, no math."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::tril"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=8
      shape_constraint: same_as=out
  op_params:
    variant: Triangular              # OpParams::Triangular (shared with triu)
    fields:
      batch_count: { kind: usize, note: "product of leading dims" }
      rows: { kind: usize }
      cols: { kind: usize }
      diagonal: { kind: i64, note: "0=main; +up/right, -down/left" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n * dtype_bytes" # read x, write out
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact: kept positions bit-identical to input, masked positions exact zero; no arithmetic, all dtypes incl. u32."

determinism: bitwise
```

---

## masked_fill  (fill positions where a U8 mask is nonzero with a scalar)

`out[i] = if mask[i] != 0 { value } else { x[i] }`, elementwise over equal-shaped `x` and `mask`.
Two inputs: a data tensor `x` (any `T: Copy`) and a `U8` mask of the **same shape** (exact dims
equality, no broadcasting). The fill scalar `value` is one element's worth of pre-encoded output-
dtype bytes (`OpParams::MaskedFill.fill_bytes`). The kernel selects per position — no arithmetic,
no dtype widening — so it is exact. Output is a fresh contiguous buffer of `x`'s shape and dtype.

> **Not reachable through the legacy reference executor.** The dtype-erased `AnyRefTensor` carries
> no `U8` variant, so `eval_node` *panics* on `Op::MaskedFill` (`src/exec.rs:457`: "legacy
> fuel-reference-backend executor doesn't support U8-mask ops; use the storage-path
> PipelinedExecutor instead"). The kernel `ops::masked_fill` itself **exists and is correct**
> (`src/ops.rs:2953`) — it is exercised directly as a functional oracle and through the storage-path
> pipelined executor, not the legacy `AnyRefTensor` exec arm. This contract describes the kernel as
> built; the planner routes `MaskedFill` to the pipelined path. The `dtypes` list below is the data
> operand's; the mask operand is always `U8`.

Source: `fuel-reference-backend/src/ops.rs:2953`; exec arm panics at `src/exec.rs:457` (oracle/
pipelined-only).

```fkc
kernel: masked_fill
op_kind: MaskedFill
blurb: "out[i] = mask[i]!=0 ? value : x[i]; same-shape U8 mask; byte-select fill, no arithmetic."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::masked_fill"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=mask
    - name: mask
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: MaskedFill              # OpParams::MaskedFill
    fields:
      fill_bytes: { kind: "Vec<u8>", note: "one element's worth, pre-encoded in the output dtype" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "0"                         # select, not arithmetic
  bytes_moved: "n * dtype_bytes + n + n * dtype_bytes"  # read x, read mask (1 byte/elt), write out
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact: unfilled positions bit-identical to x, filled positions = value bytes verbatim; no arithmetic."

determinism: bitwise
```

---

## pad_const  (constant-value pad, per-axis before/after)

Expand every axis by `(before, after)` extra slots filled with a constant `value` (`f64` coerced to
`T`). `padding.len() == rank`; output axis `i` is `in_shape[i] + before[i] + after[i]`. The kernel
pre-fills the whole output with the constant then copies the interior block from the input. Backs
`Op::Pad { mode: Constant }`. Output is a fresh contiguous buffer; same dtype as input. The fill is
the only "arithmetic" (a `cst`-narrowing of the `f64` constant into `T`), so the interior is bit-
identical to the input and the border is the narrowed constant — exact, deterministic.

> **Exec dtypes F32/F64/BF16/F16 only.** The legacy executor's `Op::Pad` arm panics on `U32`
> (`src/exec.rs:477`: "pad: not supported on U32 tensors"). The three pad modes share `OpKind::Pad`
> / `OpParams::Pad`, distinguished by `mode_tag` (0=Constant) and the `PadMode::Constant` graph tag.

Source: `fuel-reference-backend/src/ops.rs:1532`; exec arm `src/exec.rs:462` (PadMode::Constant).

```fkc
kernel: pad_const
op_kind: Pad
blurb: "Per-axis (before,after) constant pad; pre-fill with value then copy interior; F32/F64/BF16/F16."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::pad_const"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "rank == padding.len()"
  op_params:
    variant: Pad                     # OpParams::Pad
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", constraint: "out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "padding.len() == in_shape.len()" }
      mode_tag:  { kind: u8, constraint: "== 0", note: "0 = Constant" }
      fill_bytes:{ kind: "Vec<u8>", note: "one element's worth, pre-encoded in the output dtype" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(out_shape)   # per-axis in + before + after
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "0"                         # value pre-coerced to fill_bytes; copy + fill only
  bytes_moved: "n_in * dtype_bytes + n_out * dtype_bytes"  # read interior, write padded output
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n_out * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "interior bit-identical to input; border = constant narrowed to T via cst; deterministic."

determinism: same_hardware_bitwise   # f64->T narrowing of the fill constant is dtype-rounding
```

---

## pad_reflect  (reflect/mirror-edge pad, per-axis before/after)

Expand every axis by `(before, after)`, filling the border by mirroring the interior across each
edge (no edge repetition; index `n-1` is the mirror pivot). Per-axis constraint `before <= n-1` and
`after <= n-1` (you cannot reflect more than the axis length minus one). Backs
`Op::Pad { mode: Reflect }`; takes **no fill value**. Output is a fresh contiguous buffer, same
dtype. The border is exact copies of interior elements — no arithmetic — so the result is bit-
identical to a manual mirror. Exec dtypes F32/F64/BF16/F16 (U32 panics, `src/exec.rs:477`).

Source: `fuel-reference-backend/src/ops.rs:1570`; exec arm `src/exec.rs:466` (PadMode::Reflect).

```fkc
kernel: pad_reflect
op_kind: Pad
blurb: "Per-axis (before,after) reflect pad; mirror interior across each edge; before/after<=n-1; F32/F64/BF16/F16."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::pad_reflect"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "rank == padding.len()"
  op_params:
    variant: Pad                     # OpParams::Pad
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", constraint: "out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "padding[i].0 <= in_shape[i]-1 && padding[i].1 <= in_shape[i]-1" }
      mode_tag:  { kind: u8, constraint: "== 1", note: "1 = Reflect" }
      fill_bytes:{ kind: "Vec<u8>", note: "unused for Reflect" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(out_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "0"                         # pure copy/mirror; no arithmetic
  bytes_moved: "n_in * dtype_bytes + n_out * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n_out * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact: every output element is a verbatim copy of an interior element (mirror); no arithmetic."

determinism: bitwise                 # pure copy/mirror; hardware-independent
```

---

## pad_replicate  (edge-clamp/replicate pad, per-axis before/after)

Expand every axis by `(before, after)`, filling the border by repeating the nearest edge element
(clamp-to-edge). Backs `Op::Pad { mode: Replicate }`; takes **no fill value**. Output is a fresh
contiguous buffer, same dtype. The border is exact copies of the edge elements — no arithmetic —
so the result is bit-identical to a manual edge clamp. Exec dtypes F32/F64/BF16/F16 (U32 panics,
`src/exec.rs:477`).

Source: `fuel-reference-backend/src/ops.rs:1582`; exec arm `src/exec.rs:467` (PadMode::Replicate).

```fkc
kernel: pad_replicate
op_kind: Pad
blurb: "Per-axis (before,after) replicate pad; clamp-repeat the nearest edge element; F32/F64/BF16/F16."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::pad_replicate"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "rank == padding.len()"
  op_params:
    variant: Pad                     # OpParams::Pad
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", constraint: "out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "padding.len() == in_shape.len()" }
      mode_tag:  { kind: u8, constraint: "== 2", note: "2 = Replicate" }
      fill_bytes:{ kind: "Vec<u8>", note: "unused for Replicate" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(out_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "0"                         # pure copy/clamp; no arithmetic
  bytes_moved: "n_in * dtype_bytes + n_out * dtype_bytes"
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n_out * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact: every output element is a verbatim copy of an edge or interior element; no arithmetic."

determinism: bitwise
```

---

## pad_backward  (gradient of all three pad modes)

Reverse any of the three padding modes, routing each output-gradient element back to the input
position(s) it came from and **accumulating** there: an interior crop for Constant, a scatter-add
back across the mirror pivots for Reflect, a scatter-add back onto the clamped edges for Replicate.
Inputs `(grad_out, in_shape, padding, mode_tag: 0/1/2)`; output has shape `in_shape`. The `mode_tag`
field selects the routing. Because gradient elements that mapped to the same input position must
sum, the kernel accumulates in an **f64 accumulator** and narrows to `T` on store — so this is the
one kernel in this file with a real (small, dtype-rounding) numeric character rather than a pure
copy. Output is a fresh contiguous buffer of the input shape, same dtype. Per-dtype (typed addition).
Exec dtypes F32/F64/BF16/F16.

Source: `fuel-reference-backend/src/ops.rs:1634`; exec arm `src/exec.rs:480`.

```fkc
kernel: pad_backward
op_kind: PadBackward
blurb: "Backward of Pad (all 3 modes); routes/scatter-adds grad_out back to in_shape; f64 accumulator."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::pad_backward"
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "rank == padding.len()"
  op_params:
    variant: PadBackward             # OpParams::PadBackward
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", note: "the padded (forward output) shape == grad_out shape" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "padding.len() == in_shape.len()" }
      mode_tag:  { kind: u8, constraint: "in 0..=2", note: "0=Constant,1=Reflect,2=Replicate" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "mode_tag == 0", note: "Constant: pure interior crop, no scatter-add" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  # one accumulate per grad_out element (Reflect/Replicate scatter-add); Constant is a crop.
  flops: "n_out"                     # n_out = product(out_shape) = grad_out elements
  bytes_moved: "n_out * dtype_bytes + n_in * dtype_bytes"  # read grad_out, write grad_in
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n_in * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true  # fixed deterministic accumulation order in f64
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f64 accumulator then narrow to T; deterministic accumulation order; Constant mode is an exact crop."

determinism: same_hardware_bitwise
```

---

## reshape  (metadata-only shape change, zero-copy)

Re-label a contiguous tensor with a new shape of the same element count. **Pure metadata: the output
shares the input `Arc` (zero-copy) and its storage aliases the input's** — no buffer is allocated and
no byte moves. `T: Clone` (every float dtype plus `u32` index tensors). Backs `Op::Reshape(Shape)`;
the executor routes it (and `Op::Contiguize`, `Op::Unsqueeze`, `Op::Squeeze`) through `eval_reshape`
(`src/exec.rs:620-621`, `1086`). The only validation is that the new element count equals the old.

> **No dispatch-`OpKind`.** `Reshape` is a graph-level `Op` (`fuel-graph/src/lib.rs:549`,
> `Op::Reshape(Shape)`), not a `fuel-core-types` `OpKind` variant — it never reaches binding-table
> dispatch because it is satisfied by a metadata rewrite (`Op::is_view`-adjacent; the executor's
> `eval_reshape` is the materialized fallback that still only clones the `Arc`). This contract is the
> oracle's *metadata-rewrite* description so the planner can cost it as `free`. The `op_kind:` below
> names the graph `Op` by its registry string `"Reshape"`; an importer with no matching dispatch
> `OpKind` treats it as a metadata/view op (cost `free`), consistent with §5.3's `free` view ops.

Source: `fuel-reference-backend/src/ops.rs:829`; exec `eval_reshape` `src/exec.rs:1086`.

```fkc
kernel: reshape
op_kind: Reshape                     # graph-level Op::Reshape(Shape); metadata-only, no dispatch OpKind
blurb: "Metadata-only shape change; same element count; shares input Arc (zero-copy, output aliases input)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::reshape"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "product(x.shape) == product(target.shape)"
  op_params:
    variant: None                    # the target Shape lives on the output Storage / Op payload, not OpParams

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(target_shape)   # the requested new shape; element count preserved
      layout_guarantee: contiguous            # row-major reinterpretation of the same dense bytes
      aliasing: in_place(x)                    # shares the input Arc — output storage aliases input (immutable share)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "always", class: free, note: "metadata-only; no byte movement" }
  in_place: true                     # output aliases input storage (zero-copy Arc share)
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared
  class: free                        # honest free metadata-only op (§4.4): zero coefficients are a declaration, not a placeholder
  flops: "0"
  bytes_moved: "0"
  overhead_ns: 0
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true  # no arithmetic; bytes unchanged
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "no arithmetic, no byte movement; output is the identical bytes under a new shape (Arc share)."

determinism: bitwise
```

---

## cumsum  (running cumulative sum along one dim)

`out[..., i, ...] = sum(in[..., 0..=i, ...])` along `dim`, output same shape as input. The kernel
views the tensor as a flat `outer × dim_size × inner` walk (the same flat-3-axis factoring as `flip`
/`roll`), running a typed accumulator forward along the `dim` axis for each `(outer, inner)` lane.
Per-dtype (it needs typed addition), so **no `u32`** — exec panics on U32 (`src/exec.rs:413`: "cumsum:
not supported on U32 tensors"). Output is a fresh contiguous buffer, same dtype. The accumulation is
in the tensor's own dtype `T` (no f64 widening), so bf16/f16 carry the usual half-precision round-off
along the running sum; the order is fixed (low index → high index), so it is deterministic on the same
hardware. The arithmetic is one add per element.

Source: `fuel-reference-backend/src/ops.rs:1696`; exec arm `src/exec.rs:406`.

```fkc
kernel: cumsum
op_kind: CumSum
blurb: "Running cumulative sum along one dim; flat outer×dim×inner walk; typed accumulator (no u32)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cumsum"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: CumSum                  # OpParams::CumSum
    fields:
      outer_count: { kind: usize, note: "product of dims before dim" }
      dim_size:    { kind: usize }
      inner_count: { kind: usize, note: "product of dims after dim" }
      axis:        { kind: usize, note: "original dim index in the rank-N shape" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16        # smallest supported element is bf16/f16 (no u32)

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"                         # one add per element (running prefix sum)
  bytes_moved: "2 * n * dtype_bytes" # read x, write out
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true  # fixed low->high accumulation order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "accumulates in T (no f64 widening); bf16/f16 carry half-precision round-off along the running sum; deterministic order."

determinism: same_hardware_bitwise
```

---

## flip  (reverse element order along one dim)

Reverse the order of elements along `dim` (`out[..., i, ...] = in[..., dim_size-1-i, ...]`), output
same shape. The kernel views the tensor as a flat `outer × dim_size × inner` walk and `copy_from_slice`s
each row in reversed order — **a materializing copy, not a negative-stride view**. Dtype-agnostic at
the byte level (`T: Copy + Default`), so `u32` is admissible. Output is a fresh contiguous buffer.

> **Materializing, NOT `reverse_strides`.** Per `fuel-core-types/src/dispatch.rs:257-260`, this crate's
> unsigned `Layout` strides cannot represent a backward walk, so `flip` copies. Both accept and return
> declare `reverse_strides: rejected`. The zero-copy negative-stride flip of FKC §4.1.1 is a different
> backend's capability; the oracle does not advertise it.

Source: `fuel-reference-backend/src/ops.rs:1720`; exec arm `src/exec.rs:386`.

```fkc
kernel: flip
op_kind: Flip
blurb: "Reverse element order along one dim; flat outer×dim×inner copy (materializing, not a stride view)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::flip"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Flip                    # OpParams::Flip
    fields:
      outer_count: { kind: usize }
      dim_size:    { kind: usize }
      inner_count: { kind: usize }
      axis:        { kind: usize, note: "original dim index in the rank-N shape" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous   # materialized fresh buffer; reverse_strides NOT produced
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "0"                         # pure byte copy in reversed order; no arithmetic
  bytes_moved: "2 * n * dtype_bytes" # read x, write out
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true  # exact byte shuffle
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact: every output element is a verbatim copy of an input element in reversed order; no arithmetic, all dtypes incl. u32."

determinism: bitwise
```

---

## roll  (cyclic shift along one dim)

Cyclically shift elements along `dim` by a signed `shift` (positive moves elements toward higher
indices, wrapping; negative toward lower). `out[..., i, ...] = in[..., (i - shift).rem_euclid(dim_size), ...]`.
Same flat `outer × dim_size × inner` view as `flip`; a materializing copy. Dtype-agnostic byte shuffle
(`T: Copy + Default`), `u32` admissible. Output is a fresh contiguous buffer, same shape and dtype.

Source: `fuel-reference-backend/src/ops.rs:1742`; exec arm `src/exec.rs:396`.

```fkc
kernel: roll
op_kind: Roll
blurb: "Cyclic shift along one dim by signed shift (rem_euclid wrap); flat outer×dim×inner copy; all dtypes incl. u32."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::roll"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Roll                    # OpParams::Roll
    fields:
      outer_count: { kind: usize }
      dim_size:    { kind: usize }
      inner_count: { kind: usize }
      shift:       { kind: i64, note: "signed; wraps via rem_euclid" }
      axis:        { kind: usize, note: "original dim index in the rank-N shape" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "0"                         # pure byte copy with wrap; no arithmetic
  bytes_moved: "2 * n * dtype_bytes" # read x, write out
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact: every output element is a verbatim copy of an input element at a wrapped index; no arithmetic, all dtypes incl. u32."

determinism: bitwise
```

---

## concat  (concatenate two tensors along one dim)

Concatenate **two** tensors along `dim`. Both inputs must be the same rank, contiguous, and equal in
every non-`dim` dim; the output's `dim` size is the sum of the two inputs' `dim` sizes. The kernel
walks `outer_count` slabs and copies each input's slab bytes into the right offset of the output —
dtype-agnostic byte copy (`T: Clone + Default`), `u32` admissible. Output is a fresh contiguous
buffer, same dtype.

> **2-input only in this oracle.** `ops::concat` (`src/ops.rs:1398`) takes exactly two tensors,
> even though `OpKind::Concat` / `Op::Concat { dim }` and `OpParams::Concat` are N-ary in general
> (`input_dim_sizes: Vec<usize>` carries N sizes). This contract describes the as-built reference
> kernel's two-input arity; a backend that concatenates N inputs would declare a variadic
> accept-list at the same key. The `OpParams::Concat` `axis` is the concat dim in the output's
> rank-N shape; `outer_count`/`inner_count` are the products before/after it.

Source: `fuel-reference-backend/src/ops.rs:1398`; exec arm `src/exec.rs:739`.

```fkc
kernel: concat
op_kind: Concat
blurb: "Concatenate two equal-rank tensors along one dim (equal in non-dim dims); slab byte-copy; all dtypes incl. u32."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::concat"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32, F64, BF16, F16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=b && equal in all dims except axis"
    - name: b
      dtypes: [F32, F64, BF16, F16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=a && equal in all dims except axis"
  op_params:
    variant: Concat                  # OpParams::Concat
    fields:
      outer_count:     { kind: usize, note: "product of output dims before axis" }
      input_dim_sizes: { kind: "Vec<usize>", note: "per-input size along axis; len == 2 for this oracle" }
      inner_count:     { kind: usize, note: "product of output dims after axis" }
      axis:            { kind: usize, note: "concat dim in the output rank-N shape" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: from_params(axis)  # out[axis] = a[axis] + b[axis]; other dims unchanged
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise
  flops: "0"                         # pure slab byte copy; no arithmetic
  bytes_moved: "2 * n_out * dtype_bytes"  # read both inputs (= n_out elements total), write out
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n_out * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact: output is the two inputs' bytes copied verbatim into adjacent slabs; no arithmetic, all dtypes incl. u32."

determinism: bitwise
```

---

## slice  (narrow along one dim)

Narrow along `dim`: take elements `[start, start+len)`, requiring `start + len <= dim_size`. Output
keeps every other dim and shrinks `dim` to `len`. The reference kernel **materializes** a fresh
contiguous buffer (it copies the selected slab — it is *not* the zero-copy strided view that the
graph-level `Op::Slice` is, `fuel-graph/src/lib.rs:1090` lists Slice among view ops). Dtype-agnostic
byte copy (`T: Clone + Default`), `u32` admissible.

> **Graph-view vs oracle-materialize.** `Op::Slice { dim, start, len }` (`fuel-graph/src/lib.rs:677`)
> is a metadata-only *view* in the graph (`input_layout.narrow(dim, start, len)`), and there is no
> dispatch `OpKind::Slice` — the dispatch-side carrier is `OpParams::Slice { dim, start, end, step }`
> (`fuel-dispatch/src/kernel.rs:336`, the strided/step-capable shape). This **reference** kernel is
> the *materialized* oracle (`ops::slice`, `start`+`len`, step-1, fresh buffer) used to validate the
> view. The `op_kind:` names the graph `Op` by its registry string `"Slice"`; a backend that returns
> the zero-copy view declares `layout_guarantee: same_as(x)` + `cost.class: free` instead.

Source: `fuel-reference-backend/src/ops.rs:1453`; exec arm `src/exec.rs:740` (`eval_slice`).

```fkc
kernel: slice
op_kind: Slice                       # graph-level Op::Slice{dim,start,len}; oracle materializes (no dispatch OpKind)
blurb: "Narrow along one dim to [start, start+len); materializing slab copy (oracle) vs zero-copy view (graph)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::slice"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "start + len <= x.dim[dim]"
  op_params:
    variant: Slice                   # OpParams::Slice (dispatch carrier; oracle uses start+len, step=1)
    fields:
      dim:   { kind: usize }
      start: { kind: usize }
      end:   { kind: usize, note: "oracle uses len = end - start; step fixed at 1" }
      step:  { kind: usize, constraint: "== 1", note: "reference oracle is step-1 only" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(dim)   # out[dim] = len; other dims unchanged
      layout_guarantee: contiguous   # materialized fresh buffer
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared
  class: cheap_elementwise          # materializing oracle; a true view backend declares class: free
  flops: "0"                         # pure slab byte copy; no arithmetic
  bytes_moved: "2 * n_out * dtype_bytes"  # read selected slab, write out
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n_out * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact: output is the selected sub-slab copied verbatim; no arithmetic, all dtypes incl. u32."

determinism: bitwise
```

---

## broadcast_to  (NumPy broadcast to a target shape; zero-copy pure-pad fast path)

Expand a tensor to a larger, broadcast-compatible `target` shape using NumPy rules (right-align,
pad with leading 1s, a size-1 axis maps every output coord to source coord 0). `T: Float`
(`{F32, F64, BF16, F16}`). The reference has two paths (`fuel-reference-backend/src/ops.rs:1045`):

- **Zero-copy pure-pad fast path** (`ops.rs:1076`): when the source already matches its aligned
  target dims and only **leading size-1 padding** is added (no interior size-1 → N expansion), it
  returns `RefTensor::from_arc(x.as_arc().clone(), target)` — **no buffer is allocated, the output
  shares the input `Arc`, and its storage aliases the input's** (an immutable share). This is the
  metadata-only case the production graph's view op realizes for *every* broadcast.
- **Materializing path** (otherwise): allocates a fresh contiguous buffer and fills it via
  per-output-element unflatten → source-coord lookup over the contiguous source. This is the oracle
  expanding an interior broadcast axis into real repeated bytes.

Source must be contiguous; output contiguous; same dtype. **`u32` is rejected** — the executor arm
`eval_broadcast_to` (`src/exec.rs:1071-1084`) `panic!`s on `AnyRefTensor::U32` ("broadcast_to: not
supported on U32 (index) tensors"), so index tensors are not admissible here.

> **Graph-view vs oracle, and the no-dispatch-`OpKind` honesty (inv11 coverage; mirrors `reshape` /
> `slice`).** `Op::BroadcastTo(Shape)` (`fuel-graph/src/lib.rs:543`) is a **metadata-only view op**
> in the production graph — `Op::is_view_op()` returns `true` for it (`lib.rs:1089`) and its output
> Layout is derived by `derive_view_output_layout` → `input_layout.broadcast_as(target)`
> (`lib.rs:1124`) with **no Storage allocation** (a stride-0 broadcast view; the layout side-table
> note at `lib.rs:1385` lists `Op::BroadcastTo` among the metadata-only view ops). **There is NO
> `OpKind::BroadcastTo`** in the dispatch enum (`fuel-core-types/src/dispatch.rs`) and **no
> `BroadcastTo` `OpParams`/`FusedOpParams` carrier** — the forward broadcast never reaches the
> binding table or the fused registry; it is satisfied entirely by the layout rewrite (the executor's
> `eval_broadcast_to` is the materialized oracle fallback that still hits the zero-copy `Arc`-share
> for the pure-pad case). Per the never-invent / no-fabricated-dispatch-surface discipline (§0,
> inv10) this contract names the graph `Op` by its registry string **`"BroadcastTo"`** (`lib.rs:1210`)
> in the `op_kind:` slot, exactly as `reshape`/`slice` name their graph Ops; an importer with no
> matching dispatch `OpKind` treats it as a **metadata/view op** (cost `free` on the zero-copy path),
> consistent with §5.3's `free` view ops. The `op_params: None` reflects that the `target` Shape
> rides the `Op::BroadcastTo` payload / output Storage, not an `OpParams` variant.

Source: `fuel-reference-backend/src/ops.rs:1045`; exec arm `src/exec.rs:1071` (`eval_broadcast_to`).

```fkc
kernel: broadcast_to
op_kind: BroadcastTo                  # graph-level Op::BroadcastTo(Shape); metadata-only view, NO dispatch OpKind/OpParams (note above)
blurb: "NumPy broadcast to target shape; zero-copy pure-pad fast path shares input Arc, else fresh buffer; U32 rejected."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::broadcast_to"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]    # T: Float — U32 REJECTED (eval_broadcast_to panics on AnyRefTensor::U32)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "broadcast_to=target (NumPy: right-align, pad leading 1s, size-1 axis → coord 0)"
  op_params:
    variant: None                      # target Shape rides the Op::BroadcastTo payload / output Storage, not an OpParams variant

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(target_shape)   # the requested broadcast target shape
      layout_guarantee: contiguous            # fresh contiguous on the materializing path; pure-pad path is a zero-copy Arc share
      aliasing: none                          # materializing default; the pure-pad fast path aliases x (immutable Arc share) — a true view backend declares aliasing: in_place(x)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "leading size-1 padding only (no interior 1→N expansion)", class: free, note: "zero-copy: shares input Arc, no byte movement (ops.rs:1076)" }
  in_place: false                      # materializing path allocates fresh; pure-pad fast path is a zero-copy Arc share (see fast_paths)
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Op::BroadcastTo is a metadata-only view op (is_view_op, lib.rs:1089); NO OpKind::BroadcastTo. U32 rejected (exec.rs:1082 panic). Pure-pad fast path shares the input Arc (zero-copy)."

cost:
  provenance: declared
  class: cheap_elementwise            # materializing path is a bandwidth-bound expand; the pure-pad fast path is class: free
  flops: "0"                           # pure data movement, no arithmetic
  bytes_moved: "2 * n_out * dtype_bytes"  # materializing path: read source coords, write expanded out (n_out = product of target dims)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n_out * dtype_bytes", disk_bytes: 0 }   # fresh buffer on the materializing path; 0 on the pure-pad Arc-share path

precision:
  bit_stable_on_same_hardware: true   # no arithmetic; output elements are exact copies of source elements
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "no arithmetic; each output element is an exact copy of the source element at its broadcast coord. Pure-pad path is a verbatim Arc share."

determinism: bitwise
```

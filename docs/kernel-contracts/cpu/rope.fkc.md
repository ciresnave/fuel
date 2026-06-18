---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                       # maps to BackendId::Cpu
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::byte_kernels::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — RoPE kernel contracts

Rotary position embedding (RoPE) for the portable `CpuStorageBytes` surface. One logical op
(`OpKind::Rope`) monomorphized over the four float dtypes `{F32, F64, BF16, F16}`. Each dtype is a
distinct registered kernel (distinct `entry_point` → `KernelRef`); they share one accept/return
shape but differ in element width and the accumulation/narrowing rule. Source:
`fuel-cpu-backend/src/byte_kernels.rs:1885` (`rope_f32`), `:2058` (`rope_f64`), and the
`rope_half!` macro `:1968` instantiating `rope_bf16` / `rope_f16` (`:2053-2054`). All four are
primitive `op_kind` contracts (RoPE is **not** a fused op — `OpParams::Rope`,
`fuel-dispatch/src/kernel.rs:434`; `OpKind::Rope`, `fuel-core-types/src/dispatch.rs:335`).

These kernels are the production `CpuStorageBytes` path the dispatch wrapper
(`fuel_dispatch::dispatch::cpu_wrappers`, `fuel-dispatch/src/dispatch.rs:1558-1600`) extracts and
calls; they consume flat contiguous slices and the explicit `(outer_count, seq, head_dim)`
geometry, never a `Layout`/strides/offset.

## rope_f32  (rotary position embedding, f32 native, rotate_half convention)

Applies rotary position embedding to `x [outer_count, seq, head_dim]` using precomputed
`cos`/`sin` tables of shape `[seq, head_dim]` that broadcast over the `outer_count` (batch×heads)
axis. Uses the **rotate_half** convention: the head dimension is split in two halves of size
`h = head_dim/2`, and for each pair `(lo = i, hi = i+h)` in a row it computes
`out[lo] = x[lo]·cos[lo] − x[hi]·sin[lo]` and `out[hi] = x[hi]·cos[hi] + x[lo]·sin[hi]`
(`byte_kernels.rs:1957-1958`). `head_dim` must be even (odd `head_dim` is a build/run error,
`:1894`). Arithmetic is native f32 throughout. The kernel is a pure positional triple-nested walk
(`outer × seq × h`) over contiguous, zero-offset, row-major buffers; it reads `x`/`cos`/`sin` and
fully overwrites a caller-preallocated `out` of identical shape to `x`. Validation is byte-length
checks (`x.len_bytes == out.len_bytes == outer·seq·head_dim·4`, `cos.len_bytes == sin.len_bytes ==
seq·head_dim·4`) returning `Result`, never a panic on the production path (`:1906-1933`). Empty
work (`seq==0 || head_dim==0`) returns `Ok(())` after validation. Known limitations: contiguous
zero-offset only (any strided/broadcast/offset `x` must be contiguized by the planner first — the
`cos`/`sin` "broadcast over outer" is realized by the kernel re-indexing the `[seq, head_dim]`
tables per outer, NOT by a stride-0 view); no in-place; head_dim must be even.

```fkc
kernel: rope_f32
op_kind: Rope
blurb: "Rotary position embedding (rotate_half), f32 native; x[outer,seq,head_dim] with cos/sin[seq,head_dim] broadcast over outer."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rope_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [outer_count, seq, head_dim]
      shape_constraint: "divisible(x.dim[2], 2)"   # head_dim even (h = head_dim/2)
    - name: cos
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [seq, head_dim]; re-indexed per outer (NOT a stride-0 view)
      shape_constraint: "last_dim_eq=x"    # head_dim matches x; seq matches x.dim[1]
    - name: sin
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [seq, head_dim]
      shape_constraint: "same_as=cos"
  op_params:
    variant: Rope                          # OpParams::Rope (primitive namespace; §3.7)
    fields:
      outer_count: { kind: usize, note: "batch×heads flattened (cos/sin broadcast axis)" }
      seq:         { kind: usize, constraint: "== x.dim[1] == cos.dim[0]" }
      head_dim:    { kind: usize, constraint: "== x.dim[2] == cos.dim[1]; head_dim % 2 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)               # [outer_count, seq, head_dim]; symbolic seq preserved
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "seq == 0", note: "empty-work early return after validation" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: normalization
  # n = outer_count * seq * head_dim (output element count). Two FMA pairs across the two rotation
  # planes per element => derivable FLOPs; bandwidth is read x + write out + the [seq,head_dim] table reads.
  flops: "4 * outer_count * seq * head_dim"
  bytes_moved: "(2 * outer_count * seq * head_dim + 2 * seq * head_dim) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "outer_count * seq * head_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic positional nested loop; native f32 arithmetic
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                         # CPU primitive: family default (PRIMITIVE_DETERMINISTIC_CPU) applies (§4.8/§12.4)
  notes: "native f32 rotate_half; deterministic; not bit-stable cross-hardware (FMA contraction may differ)."

determinism: same_hardware_bitwise
```

## rope_f64  (rotary position embedding, f64 native, rotate_half convention)

Identical algorithm and rotate_half formula to `rope_f32`, evaluated in native f64 throughout
(`byte_kernels.rs:2058`). Same `[outer_count, seq, head_dim]` `x` with `[seq, head_dim]`
`cos`/`sin` broadcast over the outer axis; same even-`head_dim` requirement; same contiguous
zero-offset row-major byte-length validation (now against an 8-byte element); same full overwrite
of a fresh preallocated `out`. f64 gives the widest precision of the family (no widen/narrow
round-trip). Limitations match `rope_f32`: contiguous zero-offset only, no in-place, head_dim even.

```fkc
kernel: rope_f64
op_kind: Rope
blurb: "Rotary position embedding (rotate_half), f64 native; x[outer,seq,head_dim] with cos/sin[seq,head_dim] broadcast over outer."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rope_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "divisible(x.dim[2], 2)"
    - name: cos
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "last_dim_eq=x"
    - name: sin
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "same_as=cos"
  op_params:
    variant: Rope
    fields:
      outer_count: { kind: usize, note: "batch×heads flattened (cos/sin broadcast axis)" }
      seq:         { kind: usize, constraint: "== x.dim[1] == cos.dim[0]" }
      head_dim:    { kind: usize, constraint: "== x.dim[2] == cos.dim[1]; head_dim % 2 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "seq == 0", note: "empty-work early return after validation" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: normalization
  flops: "4 * outer_count * seq * head_dim"
  bytes_moved: "(2 * outer_count * seq * head_dim + 2 * seq * head_dim) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "outer_count * seq * head_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64 rotate_half; deterministic; widest precision of the family (no widen/narrow round-trip)."

determinism: same_hardware_bitwise
```

## rope_bf16  (rotary position embedding, bf16 I/O with f32 compute, rotate_half convention)

The `rope_half!`-instantiated bf16 kernel (`byte_kernels.rs:2053`, macro at `:1968`). Same
rotate_half algorithm, geometry, validation, overwrite semantics, and even-`head_dim` requirement
as `rope_f32`, but **bf16 in/out with an f32 compute round-trip**: each `x`/`cos`/`sin` element is
widened to f32 via `.to_f32()`, the rotation `out[lo] = x[lo]·cos[lo] − x[hi]·sin[lo]` /
`out[hi] = x[hi]·cos[hi] + x[lo]·sin[hi]` is done in f32, then `<bf16>::from_f32(...)` narrows on
store (`:2037-2044`). This is the family's load-bearing precision invariant: compute is f32, only
I/O is bf16. Element width is 2 bytes (validation `:1992-2019`). Limitations match the family:
contiguous zero-offset only, no in-place, head_dim even.

```fkc
kernel: rope_bf16
op_kind: Rope
blurb: "Rotary position embedding (rotate_half), bf16 I/O with f32 compute; x[outer,seq,head_dim], cos/sin[seq,head_dim] broadcast over outer."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rope_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "divisible(x.dim[2], 2)"
    - name: cos
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "last_dim_eq=x"
    - name: sin
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "same_as=cos"
  op_params:
    variant: Rope
    fields:
      outer_count: { kind: usize, note: "batch×heads flattened (cos/sin broadcast axis)" }
      seq:         { kind: usize, constraint: "== x.dim[1] == cos.dim[0]" }
      head_dim:    { kind: usize, constraint: "== x.dim[2] == cos.dim[1]; head_dim % 2 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "seq == 0", note: "empty-work early return after validation" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: normalization
  flops: "4 * outer_count * seq * head_dim"
  bytes_moved: "(2 * outer_count * seq * head_dim + 2 * seq * head_dim) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "outer_count * seq * head_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic loop; f32 compute, bf16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); bf16 I/O. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## rope_f16  (rotary position embedding, f16 I/O with f32 compute, rotate_half convention)

The `rope_half!`-instantiated f16 kernel (`byte_kernels.rs:2054`, macro at `:1968`). Byte-for-byte
the same code path as `rope_bf16` with `half::f16` substituted for `half::bf16`: rotate_half,
f32-compute round-trip (`.to_f32()` widen, `<f16>::from_f32(...)` narrow on store, `:2037-2044`),
same geometry/validation/overwrite/even-`head_dim` requirement, 2-byte element width. Differs from
bf16 only in the IEEE half-precision storage format (10-bit mantissa vs bf16's 7-bit, narrower
exponent range). Limitations match the family: contiguous zero-offset only, no in-place, head_dim
even.

```fkc
kernel: rope_f16
op_kind: Rope
blurb: "Rotary position embedding (rotate_half), f16 I/O with f32 compute; x[outer,seq,head_dim], cos/sin[seq,head_dim] broadcast over outer."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::rope_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "divisible(x.dim[2], 2)"
    - name: cos
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "last_dim_eq=x"
    - name: sin
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "same_as=cos"
  op_params:
    variant: Rope
    fields:
      outer_count: { kind: usize, note: "batch×heads flattened (cos/sin broadcast axis)" }
      seq:         { kind: usize, constraint: "== x.dim[1] == cos.dim[0]" }
      head_dim:    { kind: usize, constraint: "== x.dim[2] == cos.dim[1]; head_dim % 2 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "seq == 0", note: "empty-work early return after validation" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: normalization
  flops: "4 * outer_count * seq * head_dim"
  bytes_moved: "(2 * outer_count * seq * head_dim + 2 * seq * head_dim) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "outer_count * seq * head_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic loop; f32 compute, f16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); f16 I/O (IEEE half). Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

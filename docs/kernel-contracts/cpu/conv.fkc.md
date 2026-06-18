---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                       # maps to BackendId::Cpu
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::byte_kernels::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — 2D convolution kernel contracts

Direct (non-im2col) 2D convolution and 2D transposed convolution for the portable `CpuStorageBytes`
surface. Two logical ops — `OpKind::Conv2D` and `OpKind::ConvTranspose2D` — each monomorphized over
the four float dtypes `{F32, F64, BF16, F16}`, for eight distinct registered kernels (distinct
`entry_point` → `KernelRef`). Each op shares one accept/return shape across its dtypes but differs in
element width and the accumulation/narrowing rule: f32/f64 accumulate natively, bf16/f16 accumulate
in **f32** and narrow on store (the family precision invariant). All eight are primitive `op_kind`
contracts (convolution is **not** a fused op — `OpParams::Conv2D` / `OpParams::ConvTranspose2D`,
`fuel-dispatch/src/kernel.rs:222` / `:244`; `OpKind::Conv2D` / `OpKind::ConvTranspose2D`,
`fuel-core-types/src/dispatch.rs:122` / `:127`).

Sources: `conv2d_f32` is hand-written (`fuel-cpu-backend/src/byte_kernels.rs:7162`); `conv2d_f64` is
the native macro `conv2d_native_kernel!` (`:4380`); `conv2d_bf16` / `conv2d_f16` are the
`conv2d_half_kernel!` macro (`:4384`, instantiated `:4476-4477`). `conv_transpose2d_f32` /
`conv_transpose2d_f64` are `conv_transpose2d_native_kernel!` (`:4499`, instantiated `:4606-4607`);
`conv_transpose2d_bf16` / `conv_transpose2d_f16` are `conv_transpose2d_half_kernel!` (`:4612`,
instantiated `:4724-4725`).

These kernels are the production `CpuStorageBytes` path the dispatch wrapper
(`fuel_dispatch::dispatch::cpu_wrappers`) extracts and calls; they consume flat contiguous,
zero-offset, row-major NCHW slices and the explicit `(x_shape, w_shape, out_shape, stride, padding,
[output_padding,] dilation, groups)` geometry, never a `Layout`/strides/offset. The shared accept
surface is three inputs — `x [N, Cin, H_in, W_in]`, `weight`, and an **optional** `bias [Cout]` — and
one output `out [N, Cout, H_out, W_out]`; the spatial geometry (H_out/W_out, stride/padding/dilation,
groups, and for the transpose `output_padding`) flows entirely through the op-params, not through any
tensor layout. Group/channel divisibility (`groups != 0`, `Cin % groups == 0`, `Cout % groups == 0`,
and the weight's per-group channel slot) is build/run-validated, returning `Result` (never a panic on
the production path). The output buffer is caller-preallocated to the exact byte size and is seeded
with the broadcast bias (or zero) before accumulation.

## conv2d_f32  (2D convolution, f32 native, direct NCHW)

Direct 2D cross-correlation. For each output position `out[b, co, oh, ow]` the kernel sums
`weight[co, ci, kh_i, kw_i] · x[b, ci_offset + ci, in_h, in_w]` over the input channels of the
output's group and the `Kh × Kw` kernel window, where `in_h = oh·sh + kh_i·dh − ph`,
`in_w = ow·sw + kw_i·dw − pw`; receptive-field positions that fall outside `[0, H_in) × [0, W_in)`
(the zero-padding region) are skipped, so padding is implicit (no materialized padded input). The
accumulator is seeded with `bias[co]` (or `0` when bias is absent) and the result written once
(`byte_kernels.rs:7256-7288`). Weight layout is `[Cout, Cin/groups, Kh, Kw]`; grouped convolution is
handled by mapping each output channel to its group (`group = co / cout_per_group`) and offsetting the
input-channel base (`ci_offset = group · cin_per_group`); depthwise is the `groups == Cin`,
`Cout = Cin` special case. Arithmetic is native f32 throughout. The kernel is a pure positional nested
walk over contiguous, zero-offset, row-major NCHW buffers; it reads `x`/`weight`/optional `bias` and
fully overwrites the caller-preallocated `out`. Validation is byte-length checks against the declared
shapes (saturating, to avoid overflow on adversarial dims) plus the group/channel divisibility
contract (`:7178-7241`), returning `Result`, never a panic on the production path. Known limitations:
contiguous zero-offset NCHW only (any strided/broadcast/offset `x` or `weight` must be contiguized by
the planner first); no in-place; H_out/W_out are not recomputed — the kernel trusts the
`out_shape` it is handed (the graph builder derives the conv geometry); no im2col/Winograd fast path
(naive direct loop).

```fkc
kernel: conv2d_f32
op_kind: Conv2D
blurb: "Direct 2D convolution (cross-correlation), f32 native; x[N,Cin,H,W], weight[Cout,Cin/groups,Kh,Kw], optional bias[Cout], grouped/depthwise."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::conv2d_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [N, Cin, H_in, W_in] NCHW
    - name: weight
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [Cout, Cin/groups, Kh, Kw]
      shape_constraint: "divisible(x.dim[1], weight.dim[1]); weight.dim[1] == x.dim[1] / groups"
    - name: bias
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Cout]; broadcast over N×H_out×W_out by the kernel (NOT a stride-0 view)
      shape_constraint: "weight.dim[0] == bias.dim[0]"   # Cout
      optional: true                       # absent ⇒ accumulator seeded with 0
  op_params:
    variant: Conv2D                        # OpParams::Conv2D (primitive namespace; §3.7)
    fields:
      x_shape:   { kind: "[usize; 4]", note: "[N, Cin, H_in, W_in]" }
      w_shape:   { kind: "[usize; 4]", note: "[Cout, Cin/groups, Kh, Kw]" }
      out_shape: { kind: "[usize; 4]", note: "[N, Cout, H_out, W_out]; kernel trusts this geometry" }
      stride:    { kind: "(usize, usize)" }
      padding:   { kind: "(usize, usize)" }
      dilation:  { kind: "(usize, usize)" }
      groups:    { kind: usize, constraint: "groups != 0; Cin % groups == 0; Cout % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv2d(params)           # [N, Cout, H_out, W_out] from OpParams::Conv2D geometry (§5.2)
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer; bias/zero-seeded then accumulated

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "groups == 1", class: conv }         # dense (non-grouped) convolution
    - { when: "depthwise", note: "groups == Cin, Cout == Cin special case" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); the formula hint below is the derivable prior
  class: conv
  # FLOPs derivable: 2 MACs-as-2-flops per (output element × in-channels-per-group × Kh × Kw); this is
  # the dense upper bound (the padding skip only reduces it at the borders). out_n = N·Cout·H_out·W_out.
  flops: "2 * out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * (x_shape[1] / groups) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic positional nested loop; native f32 arithmetic, fixed accumulation order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                         # CPU primitive: family default (PRIMITIVE_DETERMINISTIC_CPU) applies (§4.8/§12.4)
  notes: "native f32 direct conv; fixed reduction order over (Cin/group, Kh, Kw); deterministic; not bit-stable cross-hardware (FMA contraction may differ)."

determinism: same_hardware_bitwise
```

## conv2d_f64  (2D convolution, f64 native, direct NCHW)

Identical algorithm, weight layout, grouping, padding-skip, and bias-seed semantics to `conv2d_f32`,
evaluated in native f64 throughout (`conv2d_native_kernel!`, `byte_kernels.rs:4380`). Same NCHW
contiguous zero-offset row-major byte-length validation (now against an 8-byte element) and the same
group/channel divisibility contract; same single full overwrite of a fresh preallocated `out` seeded
with `bias[co]` (or `0`). f64 gives the widest precision of the family (no widen/narrow round-trip).
Limitations match `conv2d_f32`: contiguous zero-offset NCHW only, no in-place, naive direct loop.

```fkc
kernel: conv2d_f64
op_kind: Conv2D
blurb: "Direct 2D convolution (cross-correlation), f64 native; x[N,Cin,H,W], weight[Cout,Cin/groups,Kh,Kw], optional bias[Cout], grouped/depthwise."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::conv2d_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: weight
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "divisible(x.dim[1], weight.dim[1]); weight.dim[1] == x.dim[1] / groups"
    - name: bias
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "weight.dim[0] == bias.dim[0]"
      optional: true
  op_params:
    variant: Conv2D
    fields:
      x_shape:   { kind: "[usize; 4]", note: "[N, Cin, H_in, W_in]" }
      w_shape:   { kind: "[usize; 4]", note: "[Cout, Cin/groups, Kh, Kw]" }
      out_shape: { kind: "[usize; 4]", note: "[N, Cout, H_out, W_out]" }
      stride:    { kind: "(usize, usize)" }
      padding:   { kind: "(usize, usize)" }
      dilation:  { kind: "(usize, usize)" }
      groups:    { kind: usize, constraint: "groups != 0; Cin % groups == 0; Cout % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv2d(params)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "groups == 1", class: conv }
    - { when: "depthwise", note: "groups == Cin, Cout == Cin special case" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: conv
  flops: "2 * out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * (x_shape[1] / groups) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64 direct conv; fixed reduction order; deterministic; widest precision of the family (no widen/narrow round-trip)."

determinism: same_hardware_bitwise
```

## conv2d_bf16  (2D convolution, bf16 I/O with f32 accumulator, direct NCHW)

The `conv2d_half_kernel!`-instantiated bf16 kernel (`byte_kernels.rs:4476`, macro at `:4384`). Same
direct algorithm, NCHW weight layout, grouping, padding-skip, and overwrite semantics as
`conv2d_f32`, but **bf16 in/out with an f32 accumulator**: the per-output accumulator is seeded with
`bias[co].to_f32()` (or `0`), each `weight·x` product widens both operands via `.to_f32()` and
accumulates in f32 over the `(Cin/group, Kh, Kw)` receptive field, then `<bf16>::from_f32(acc)`
narrows once on store (`:4446-4466`). This is the family's load-bearing precision invariant: each
output position's full reduction is computed in f32, only the final stored value is bf16. Element
width is 2 bytes (`:4414-4423`). Limitations match the family: contiguous zero-offset NCHW only, no
in-place, naive direct loop.

```fkc
kernel: conv2d_bf16
op_kind: Conv2D
blurb: "Direct 2D convolution (cross-correlation), bf16 I/O with f32 accumulator; x[N,Cin,H,W], weight[Cout,Cin/groups,Kh,Kw], optional bias[Cout], grouped/depthwise."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::conv2d_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: weight
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "divisible(x.dim[1], weight.dim[1]); weight.dim[1] == x.dim[1] / groups"
    - name: bias
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "weight.dim[0] == bias.dim[0]"
      optional: true
  op_params:
    variant: Conv2D
    fields:
      x_shape:   { kind: "[usize; 4]", note: "[N, Cin, H_in, W_in]" }
      w_shape:   { kind: "[usize; 4]", note: "[Cout, Cin/groups, Kh, Kw]" }
      out_shape: { kind: "[usize; 4]", note: "[N, Cout, H_out, W_out]" }
      stride:    { kind: "(usize, usize)" }
      padding:   { kind: "(usize, usize)" }
      dilation:  { kind: "(usize, usize)" }
      groups:    { kind: usize, constraint: "groups != 0; Cin % groups == 0; Cout % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv2d(params)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "groups == 1", class: conv }
    - { when: "depthwise", note: "groups == Cin, Cout == Cin special case" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: conv
  flops: "2 * out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * (x_shape[1] / groups) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic loop; f32 accumulator per output position, bf16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "per-output-position reduction accumulated in f32 (widen on load, narrow on store); bf16 I/O. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## conv2d_f16  (2D convolution, f16 I/O with f32 accumulator, direct NCHW)

The `conv2d_half_kernel!`-instantiated f16 kernel (`byte_kernels.rs:4477`, macro at `:4384`).
Byte-for-byte the same code path as `conv2d_bf16` with `half::f16` substituted for `half::bf16`:
direct conv, per-output f32 accumulator round-trip (`.to_f32()` widen on each product,
`<f16>::from_f32(acc)` narrow once on store, `:4446-4466`), same NCHW weight layout, grouping,
padding-skip, geometry/validation/overwrite, 2-byte element width. Differs from bf16 only in the IEEE
half-precision storage format (10-bit mantissa vs bf16's 7-bit, narrower exponent range).
Limitations match the family: contiguous zero-offset NCHW only, no in-place, naive direct loop.

```fkc
kernel: conv2d_f16
op_kind: Conv2D
blurb: "Direct 2D convolution (cross-correlation), f16 I/O with f32 accumulator; x[N,Cin,H,W], weight[Cout,Cin/groups,Kh,Kw], optional bias[Cout], grouped/depthwise."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::conv2d_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: weight
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "divisible(x.dim[1], weight.dim[1]); weight.dim[1] == x.dim[1] / groups"
    - name: bias
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "weight.dim[0] == bias.dim[0]"
      optional: true
  op_params:
    variant: Conv2D
    fields:
      x_shape:   { kind: "[usize; 4]", note: "[N, Cin, H_in, W_in]" }
      w_shape:   { kind: "[usize; 4]", note: "[Cout, Cin/groups, Kh, Kw]" }
      out_shape: { kind: "[usize; 4]", note: "[N, Cout, H_out, W_out]" }
      stride:    { kind: "(usize, usize)" }
      padding:   { kind: "(usize, usize)" }
      dilation:  { kind: "(usize, usize)" }
      groups:    { kind: usize, constraint: "groups != 0; Cin % groups == 0; Cout % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv2d(params)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "groups == 1", class: conv }
    - { when: "depthwise", note: "groups == Cin, Cout == Cin special case" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: conv
  flops: "2 * out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * (x_shape[1] / groups) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic loop; f32 accumulator per output position, f16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "per-output-position reduction accumulated in f32 (widen on load, narrow on store); f16 I/O (IEEE half). Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## conv_transpose2d_f32  (2D transposed convolution, f32 native, scatter-accumulate NCHW)

Direct 2D transposed convolution (the gradient-of-convolution / fractionally-strided conv). Unlike
forward conv's per-output gather, this kernel **scatters**: the output is first seeded with the
broadcast bias `bias[co]` (or `0`), then for every input element `x[n, ci, hi, wi]` its kernel-shaped
contribution `x · weight[ci, co_local, kh_i, kw_i]` is accumulated into the output positions
`oh = hi·sh + kh_i·dh − ph`, `ow = wi·sw + kw_i·dw − pw` that land inside `[0, H_out) × [0, W_out)`
(`byte_kernels.rs:4556-4600`). The weight layout is **transposed channel order vs forward conv** —
`[Cin, Cout/groups, Kh, Kw]` (`:4516`) — and grouping splits both Cin and Cout into `groups` blocks
(`cin_per_group = Cin/groups`, `cout_per_group = Cout/groups`). A micro-optimization skips the inner
scatter when the input value is exactly zero (`val == 0`, `:4579`); this is value-data-dependent but
result-identical. Arithmetic is native f32 throughout, accumulating directly into the output buffer.
The `output_padding` op-param disambiguates the output spatial size
(`H_out = (H_in−1)·sh − 2·ph + dh·(Kh−1) + out_pad.0 + 1`) but the kernel trusts the `out_shape` it is
handed. Validation is byte-length checks against the declared shapes plus the transpose-specific
divisibility contract (`Cin == weight.dim[0]`, `Cout/groups == cout_per_group`, `groups != 0`,
`Cin % groups == 0`, `Cout % groups == 0`; `:4518-4548`), returning `Result`, never a panic. Known
limitations: contiguous zero-offset NCHW only; no in-place; H_out/W_out not recomputed (the kernel
trusts `out_shape`); naive direct scatter loop (no im2col/col2im).

```fkc
kernel: conv_transpose2d_f32
op_kind: ConvTranspose2D
blurb: "Direct 2D transposed convolution (scatter-accumulate), f32 native; x[N,Cin,H,W], weight[Cin,Cout/groups,Kh,Kw], optional bias[Cout], grouped."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::conv_transpose2d_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [N, Cin, H_in, W_in] NCHW
    - name: weight
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [Cin, Cout/groups, Kh, Kw] (transposed channel order vs Conv2D)
      shape_constraint: "weight.dim[0] == x.dim[1]"   # weight Cin axis == input Cin
    - name: bias
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Cout]; broadcast over N×H_out×W_out by the kernel (NOT a stride-0 view)
      shape_constraint: "out.dim[1] == bias.dim[0]"   # Cout
      optional: true                       # absent ⇒ output seeded with 0
  op_params:
    variant: ConvTranspose2D               # OpParams::ConvTranspose2D (primitive namespace; §3.7)
    fields:
      x_shape:        { kind: "[usize; 4]", note: "[N, Cin, H_in, W_in]" }
      w_shape:        { kind: "[usize; 4]", note: "[Cin, Cout/groups, Kh, Kw]" }
      out_shape:      { kind: "[usize; 4]", note: "[N, Cout, H_out, W_out]; kernel trusts this geometry" }
      stride:         { kind: "(usize, usize)" }
      padding:        { kind: "(usize, usize)" }
      output_padding: { kind: "(usize, usize)", note: "disambiguates H_out/W_out; transpose-only param" }
      dilation:       { kind: "(usize, usize)" }
      groups:         { kind: usize, constraint: "groups != 0; Cin % groups == 0; Cout % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv_transpose2d(params) # [N, Cout, H_out, W_out] from OpParams::ConvTranspose2D geometry (§5.2)
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer; bias/zero-seeded then scatter-accumulated

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "groups == 1", class: conv }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); the formula hint below is the derivable prior
  class: conv
  # FLOPs derivable: transposed conv scatters from EVERY input element across the kernel window into
  # Cout/group output channels => 2 flops per (input element × Cout/group × Kh × Kw); dense upper bound
  # (border scatters that fall outside the output, and the val==0 skip, only reduce it). in_n = N·Cin·H_in·W_in.
  flops: "2 * x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3] * (out_shape[1] / groups) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic scatter order; native f32 arithmetic accumulated into the output buffer
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                         # CPU primitive: family default (PRIMITIVE_DETERMINISTIC_CPU) applies (§4.8/§12.4)
  notes: "native f32 scatter-accumulate; fixed scatter order; val==0 skip is value-dependent but result-identical; deterministic; not bit-stable cross-hardware (FMA contraction may differ)."

determinism: same_hardware_bitwise
```

## conv_transpose2d_f64  (2D transposed convolution, f64 native, scatter-accumulate NCHW)

Identical algorithm, transposed `[Cin, Cout/groups, Kh, Kw]` weight layout, grouping, scatter-mapping,
`val == 0` skip, and bias-seed semantics to `conv_transpose2d_f32`, evaluated in native f64 throughout
(`conv_transpose2d_native_kernel!`, `byte_kernels.rs:4607`). Same NCHW contiguous zero-offset row-major
byte-length validation (now against an 8-byte element) and the same transpose divisibility contract;
same accumulation directly into the output buffer. f64 gives the widest precision of the family (no
widen/narrow round-trip). Limitations match `conv_transpose2d_f32`: contiguous zero-offset NCHW only,
no in-place, naive direct scatter loop.

```fkc
kernel: conv_transpose2d_f64
op_kind: ConvTranspose2D
blurb: "Direct 2D transposed convolution (scatter-accumulate), f64 native; x[N,Cin,H,W], weight[Cin,Cout/groups,Kh,Kw], optional bias[Cout], grouped."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::conv_transpose2d_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: weight
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "weight.dim[0] == x.dim[1]"
    - name: bias
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "out.dim[1] == bias.dim[0]"
      optional: true
  op_params:
    variant: ConvTranspose2D
    fields:
      x_shape:        { kind: "[usize; 4]", note: "[N, Cin, H_in, W_in]" }
      w_shape:        { kind: "[usize; 4]", note: "[Cin, Cout/groups, Kh, Kw]" }
      out_shape:      { kind: "[usize; 4]", note: "[N, Cout, H_out, W_out]" }
      stride:         { kind: "(usize, usize)" }
      padding:        { kind: "(usize, usize)" }
      output_padding: { kind: "(usize, usize)", note: "disambiguates H_out/W_out; transpose-only param" }
      dilation:       { kind: "(usize, usize)" }
      groups:         { kind: usize, constraint: "groups != 0; Cin % groups == 0; Cout % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv_transpose2d(params)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "groups == 1", class: conv }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: conv
  flops: "2 * x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3] * (out_shape[1] / groups) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64 scatter-accumulate; fixed scatter order; val==0 skip value-dependent but result-identical; deterministic; widest precision of the family (no widen/narrow round-trip)."

determinism: same_hardware_bitwise
```

## conv_transpose2d_bf16  (2D transposed convolution, bf16 I/O with f32 accumulator, scatter-accumulate NCHW)

The `conv_transpose2d_half_kernel!`-instantiated bf16 kernel (`byte_kernels.rs:4724`, macro at
`:4612`). Same scatter algorithm, transposed `[Cin, Cout/groups, Kh, Kw]` weight layout, grouping,
scatter-mapping, and `val == 0` skip as `conv_transpose2d_f32`, but **bf16 in/out with an f32
accumulator buffer**: a parallel `Vec<f32>` of the full output size is allocated, seeded with
`bias[co].to_f32()` (or `0`), every input element's contribution is widened via `.to_f32()` and
scatter-accumulated into that f32 buffer, and finally `<bf16>::from_f32(...)` narrows the whole buffer
into `out` (`:4669-4718`). This is the family precision invariant: all scatter accumulation happens in
f32, only the final stored values are bf16. Note the half path therefore allocates a transient
host-side f32 scratch buffer of `N·Cout·H_out·W_out` f32 elements (the native f32/f64 paths accumulate
in place and allocate none). Element width is 2 bytes. Limitations match the family: contiguous
zero-offset NCHW only, no in-place, naive direct scatter loop.

```fkc
kernel: conv_transpose2d_bf16
op_kind: ConvTranspose2D
blurb: "Direct 2D transposed convolution (scatter-accumulate), bf16 I/O with f32 accumulator; x[N,Cin,H,W], weight[Cin,Cout/groups,Kh,Kw], optional bias[Cout], grouped."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::conv_transpose2d_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: weight
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "weight.dim[0] == x.dim[1]"
    - name: bias
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "out.dim[1] == bias.dim[0]"
      optional: true
  op_params:
    variant: ConvTranspose2D
    fields:
      x_shape:        { kind: "[usize; 4]", note: "[N, Cin, H_in, W_in]" }
      w_shape:        { kind: "[usize; 4]", note: "[Cin, Cout/groups, Kh, Kw]" }
      out_shape:      { kind: "[usize; 4]", note: "[N, Cout, H_out, W_out]" }
      stride:         { kind: "(usize, usize)" }
      padding:        { kind: "(usize, usize)" }
      output_padding: { kind: "(usize, usize)", note: "disambiguates H_out/W_out; transpose-only param" }
      dilation:       { kind: "(usize, usize)" }
      groups:         { kind: usize, constraint: "groups != 0; Cin % groups == 0; Cout % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv_transpose2d(params)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "groups == 1", class: conv }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured             # Judge bootstraps/refines (§4.4); formula hints below are the derivable prior
  class: conv
  flops: "2 * x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3] * (out_shape[1] / groups) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  # half path allocates a transient f32 accumulator of the full output size (host scratch); native paths allocate none.
  memory: { device_bytes: 0, host_bytes: "out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * (dtype_bytes + 4)", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic scatter order; f32 accumulator buffer, bf16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "all scatter accumulation in an f32 buffer (widen on load, narrow on store); bf16 I/O. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## conv_transpose2d_f16  (2D transposed convolution, f16 I/O with f32 accumulator, scatter-accumulate NCHW)

The `conv_transpose2d_half_kernel!`-instantiated f16 kernel (`byte_kernels.rs:4725`, macro at
`:4612`). Byte-for-byte the same code path as `conv_transpose2d_bf16` with `half::f16` substituted for
`half::bf16`: scatter-accumulate into a parallel `Vec<f32>` buffer seeded with `bias.to_f32()` (or
`0`), `.to_f32()` widen on each product, `<f16>::from_f32(...)` narrow the whole buffer once on store
(`:4669-4718`); same transposed `[Cin, Cout/groups, Kh, Kw]` weight layout, grouping, scatter-mapping,
`val == 0` skip, geometry/validation/overwrite, 2-byte element width, and transient f32 scratch of the
full output size. Differs from bf16 only in the IEEE half-precision storage format (10-bit mantissa vs
bf16's 7-bit, narrower exponent range). Limitations match the family: contiguous zero-offset NCHW only,
no in-place, naive direct scatter loop.

```fkc
kernel: conv_transpose2d_f16
op_kind: ConvTranspose2D
blurb: "Direct 2D transposed convolution (scatter-accumulate), f16 I/O with f32 accumulator; x[N,Cin,H,W], weight[Cin,Cout/groups,Kh,Kw], optional bias[Cout], grouped."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::conv_transpose2d_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: weight
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "weight.dim[0] == x.dim[1]"
    - name: bias
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "out.dim[1] == bias.dim[0]"
      optional: true
  op_params:
    variant: ConvTranspose2D
    fields:
      x_shape:        { kind: "[usize; 4]", note: "[N, Cin, H_in, W_in]" }
      w_shape:        { kind: "[usize; 4]", note: "[Cin, Cout/groups, Kh, Kw]" }
      out_shape:      { kind: "[usize; 4]", note: "[N, Cout, H_out, W_out]" }
      stride:         { kind: "(usize, usize)" }
      padding:        { kind: "(usize, usize)" }
      output_padding: { kind: "(usize, usize)", note: "disambiguates H_out/W_out; transpose-only param" }
      dilation:       { kind: "(usize, usize)" }
      groups:         { kind: usize, constraint: "groups != 0; Cin % groups == 0; Cout % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv_transpose2d(params)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "groups == 1", class: conv }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: conv
  flops: "2 * x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3] * (out_shape[1] / groups) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  memory: { device_bytes: 0, host_bytes: "out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * (dtype_bytes + 4)", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic scatter order; f32 accumulator buffer, f16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "all scatter accumulation in an f32 buffer (widen on load, narrow on store); f16 I/O (IEEE half). Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

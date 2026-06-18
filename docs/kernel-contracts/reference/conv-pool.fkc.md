---
fkc_version: 1
provider:
  name: fuel-reference-backend
  backend: Cpu                       # the reference oracle runs on the host
  kernel_source: "reference-oracle"  # the BindingEntry.kernel_source tag
  link_registry: fuel_reference_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-reference-backend — convolution & pooling contracts

Reference (correctness-first) contracts for the conv/pool family of the pure-Rust oracle crate
(`fuel-reference-backend`, family `conv-attn` in `_inventory/reference.md`). Every kernel in this
crate is, by the crate-wide invariant, **contiguous-only, zero-offset, row-major** (`RefTensor<T>`
carries no strides and no offset — `lib.rs:68`), so every operand below declares
`contiguous: required` and `awkward_layout_strategy: requires_contiguous`: the planner contiguizes
any non-contiguous producer (pricing the inserted `Op::Contiguize` from its own FKC contract, §4.3)
before these kernels see the buffer. All math is in the generic `T: num_traits::Float` accumulator
(NO f32-accum widening for half), monomorphized to the dtype list the executor wires. Output is
always a fresh contiguous tensor (`RefTensor::from_vec`), aliasing none.

This file is rendered by mdBook and parsed by the FKC importer; the prose is documentation and the
` ```fkc ` block is authoritative (§3.1). Costs are marked `judge_measured` — the Judge bootstraps
them; FLOPs/bandwidth hints are given where genuinely derivable from the op geometry (§4.4) and are
the author's structural prior, not a fabricated calibrated number.

---

## conv2d  (2-D direct convolution; production, registry CONV2D)

Direct NCHW 2-D convolution with groups + optional bias; f32/f64; contiguous; T-precision accumulator.

Direct (im2col-free) 2-D convolution over a packed NCHW input. `x [N, Cin, H, W]`,
`weight [Cout, Cin/groups, Kh, Kw]`, optional `bias [Cout]`. Asymmetric `stride (sh, sw)` and
symmetric zero-`padding (ph, pw)`, grouped via `groups` (`Cin` and `Cout` both divisible by
`groups`; depthwise is the `groups == Cin` case). The kernel validates the geometry (`ConvShape`)
then delegates the inner loops to `fuel_conv::conv2d_direct` (`ops.rs:704`). Output
`[N, Cout, Hout, Wout]` with `Hout = (H + 2·ph − Kh)/sh + 1`, `Wout = (W + 2·pw − Kw)/sw + 1`.

Numerics: the accumulator is the operand dtype `T` (no f32 widening), so bf16/f16 accumulation
drifts versus an f32-accum implementation — this is a correctness oracle that matches the naive
nested-sum, not a numerically-hardened production GEMM. Zero padding contributes literal `T::zero()`
(skipped, not summed). Executor wiring (`exec.rs:1291`) admits **f32/f64 only**; bf16/f16 are
expressible by the generic kernel but the exec arm does not route them, so this contract declares
the exec-admitted dtype set. Perf: dense NCHW walk, output-stationary; cost scales with
`2·N·Cout·(Cin/groups)·Kh·Kw·Hout·Wout` MACs. Known limitation: contiguous-only — any strided or
offset producer is contiguized by the planner first; no internal contiguize, no in-place.

```fkc
kernel: conv2d
op_kind: Conv2D                # OpParams::Conv2D carrier; OpKind::Conv2D (dispatch.rs:122)
blurb: "Direct NCHW 2-D convolution with groups + optional bias; f32/f64; contiguous; T-precision accumulator."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::conv2d_f32"   # one per dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [N, Cin, H, W]
    - name: weight
      dtypes: [F32, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [Cout, Cin/groups, Kh, Kw]
      shape_constraint: "divisible(x.dim[1], weight.dim[1])"   # Cin divisible by Cin/groups
    - name: bias
      dtypes: [F32, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # [Cout]
      shape_constraint: "same_as=weight.dim[0]"
      optional: true
  op_params:
    variant: Conv2D               # OpParams::Conv2D (primitive namespace; §3.7)
    fields:
      x_shape:   { kind: "[usize; 4]" }
      w_shape:   { kind: "[usize; 4]" }
      out_shape: { kind: "[usize; 4]" }
      stride:    { kind: "(usize, usize)" }
      padding:   { kind: "(usize, usize)" }
      dilation:  { kind: "(usize, usize)", note: "reference kernel walks dilation=(1,1) only; carried for ABI parity" }
      groups:    { kind: usize, constraint: "x.dim[1] % groups == 0 && weight.dim[0] % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)        # output dtype = input dtype (T)
      shape_rule: conv2d(params)        # [N, Cout, Hout, Wout] from OpParams::Conv2D geometry
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "groups == 1", note: "dense full-conv path; no per-group channel slicing" }
    - { when: "depthwise", note: "groups == Cin: one filter per channel" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # Judge bootstraps; the flops/bytes below are the structural prior
  class: conv
  # MACs = N * Cout * (Cin/groups) * Kh * Kw * Hout * Wout; 2 flops per MAC (mul + add)
  flops: "2 * out_shape[0] * out_shape[1] * (x_shape[1] / groups) * w_shape[2] * w_shape[3] * out_shape[2] * out_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  overhead_ns: ~                      # launch cost (host call + ConvShape validation) not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3] * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic nested loop, fixed accumulation order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "T-precision accumulator (NO f32 widening for half); deterministic fixed summation order; zero padding adds T::zero(). Matches naive direct conv, not an f32-accum production GEMM."

determinism: same_hardware_bitwise
```

---

## conv_transpose2d  (2-D transposed convolution; registry CONV_TRANSPOSE2D)

Scatter-form NCHW 2-D transposed convolution with groups; f32/f64; contiguous; T-precision accumulator.

Textbook 2-D transposed convolution ("deconv") in scatter form: each input pixel is scattered into
a `Kh × Kw` output region through every kernel position (`ops.rs:751`). `x [N, Cin, H, W]`,
`weight [Cin, Cout/groups, Kh, Kw]` (note the **transposed channel order** versus `conv2d` — `Cin`
leads). Asymmetric `stride`, `padding`, `output_padding`, and `dilation` (all `(h, w)`), grouped via
`groups` (`Cin` divisible by `groups`). Output `[N, Cout, Hout, Wout]` with
`Hout = (H − 1)·sh − 2·ph + dh·(Kh − 1) + oph + 1` and the symmetric `Wout`.

Numerics: scatter-accumulate into a zero-initialized output buffer in operand dtype `T` (no f32
widening). Because the accumulation order over scattered contributions is fixed by the nested loop,
the result is deterministic and same-hardware bit-stable, but half-precision accumulation drifts
versus an f32-accum form. The kernel asserts the padding does not exceed the produced unpadded
dims. Executor wiring (`exec.rs:1322`) admits **f32/f64 only**; this contract declares that set.
Perf: scatter form is slower than a gather/im2col deconv (`O(N·Cin·H·W·Cout/groups·Kh·Kw)` writes
with output-overlap accumulation) — a correctness oracle, not a tuned kernel. Known limitation:
contiguous-only; no in-place; the output region overlap means writes are read-modify-write into the
fresh output buffer (still `aliasing: none` — no input is aliased).

```fkc
kernel: conv_transpose2d
op_kind: ConvTranspose2D       # OpParams::ConvTranspose2D carrier; OpKind::ConvTranspose2D (dispatch.rs:127)
blurb: "Scatter-form NCHW 2-D transposed convolution with groups; f32/f64; contiguous; T-precision accumulator."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::conv_transpose2d_f32"   # one per dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [N, Cin, H, W]
    - name: weight
      dtypes: [F32, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [Cin, Cout/groups, Kh, Kw] — transposed channel order
      shape_constraint: "same_as=x.dim[1]"   # weight.dim[0] (Cin) == x.dim[1] (Cin)
  op_params:
    variant: ConvTranspose2D      # OpParams::ConvTranspose2D (primitive namespace; §3.7)
    fields:
      x_shape:        { kind: "[usize; 4]" }
      w_shape:        { kind: "[usize; 4]" }
      out_shape:      { kind: "[usize; 4]" }
      stride:         { kind: "(usize, usize)" }
      padding:        { kind: "(usize, usize)" }
      output_padding: { kind: "(usize, usize)" }
      dilation:       { kind: "(usize, usize)" }
      groups:         { kind: usize, constraint: "x.dim[1] % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)            # output dtype = input dtype (T)
      shape_rule: conv_transpose2d(params)  # [N, Cout, Hout, Wout] from OpParams::ConvTranspose2D geometry
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "groups == 1", note: "full deconv; no per-group channel slicing" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # Judge bootstraps; the flops/bytes below are the structural prior
  class: conv
  # scatter MACs = N * Cin * H * W * (Cout/groups) * Kh * Kw; 2 flops per MAC (mul + accumulate)
  flops: "2 * x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3] * (w_shape[1]) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3] * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic scatter order into zero-init buffer
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "T-precision accumulator (NO f32 widening for half); deterministic fixed scatter-accumulation order; output region overlap accumulated into a fresh zero-init buffer."

determinism: same_hardware_bitwise
```

---

## conv2d_simple  (2-D convolution; legacy, no bias/groups — test-only oracle)

Legacy gather-form NCHW 2-D convolution, no bias/groups, scalar stride/padding; f32/f64; contiguous.

Legacy textbook 2-D convolution: gather form, output-stationary, **no bias and no groups**, with a
**scalar** `stride` and **scalar** `padding` (`ops.rs:2209`). `x [N, Cin, H, W]`,
`kernel [Cout, Cin, Kh, Kw]` (full `Cin`, since there are no groups). Symmetric zero padding;
output `[N, Cout, Hout, Wout]` with `Hout = (H + 2·pad − Kh)/stride + 1`, `Wout` symmetric.
Out-of-range (padded) coordinates contribute nothing (skipped). This is the predecessor of the
production `conv2d` above and is retained as a simpler cross-check oracle.

**Status — no dispatch carrier (test-only).** Per the inventory (`reference.md:211`) this kernel is
**not wired in the executor** and has **no `OpKind` and no `OpParams` variant** in `fuel-dispatch`
(only the production `conv2d`/`OpKind::Conv2D` is wired; grep confirms no `Conv2dSimple` op-kind).
The contract below is therefore authored faithfully but, like an MX contract (§6), it
**parse-validates yet is NOT registrable as-is**: §10.7 (op-param variant must be a real
`OpParams` variant) and the required `op_kind` ∈ real `OpKind` both fail because no such variant
exists. The `op_kind` field names the intended `Conv2dSimple` carrier as a forward marker; an
importer returns a `BadOpParamsVariant`/unknown-`OpKind` error (the `MxNotYetRegistrable`-class
discipline) until a dispatch carrier lands or the kernel is folded into `OpKind::Conv2D` with
`groups == 1`. Numerics match `conv2d`: T-precision accumulator, deterministic, contiguous-only.

```fkc
kernel: conv2d_simple
op_kind: Conv2dSimple          # forward marker only; NO such OpKind/OpParams variant exists in fuel-dispatch
                               # (test-only oracle, no dispatch carrier) — describe-only (§3.10)
registrable: false             # §3.10 — documentation-only; not registered, op_kind/op_params not required to resolve
blurb: "Legacy gather-form NCHW 2-D convolution, no bias/groups, scalar stride/padding; f32/f64; contiguous."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::conv2d_simple_f32"   # one per dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [N, Cin, H, W]
    - name: kernel
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [Cout, Cin, Kh, Kw]
      shape_constraint: "same_as=x.dim[1]"   # kernel.dim[1] (Cin) == x.dim[1] (Cin)
  op_params:
    variant: Conv2dSimple         # [consumer-ahead] NO such OpParams variant; see Status
    fields:
      stride:  { kind: usize, constraint: "stride > 0" }
      padding: { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)        # output dtype = input dtype (T)
      shape_rule: "from_params(N=x.dim[0], Cout=kernel.dim[0], Hout=(x.dim[2]+2*padding-kernel.dim[2])/stride+1, Wout=(x.dim[3]+2*padding-kernel.dim[3])/stride+1)"
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # Judge bootstraps; the flops/bytes below are the structural prior
  class: conv
  # MACs = N * Cout * Cin * Kh * Kw * Hout * Wout; 2 flops per MAC
  flops: "2 * x.dim[0] * kernel.dim[0] * x.dim[1] * kernel.dim[2] * kernel.dim[3] * out.dim[2] * out.dim[3]"
  bytes_moved: "(x.dim[0]*x.dim[1]*x.dim[2]*x.dim[3] + kernel.dim[0]*kernel.dim[1]*kernel.dim[2]*kernel.dim[3] + out.dim[0]*out.dim[1]*out.dim[2]*out.dim[3]) * dtype_bytes"
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "out.dim[0]*out.dim[1]*out.dim[2]*out.dim[3] * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "T-precision accumulator (NO f32 widening for half); deterministic gather summation order; zero padding skipped (contributes 0)."

determinism: same_hardware_bitwise
```

---

## max_pool2d  (2-D max pooling, no padding — test-only oracle)

Per-channel 2-D max pool, square window, no padding, scalar stride; f32/f64/bf16/f16; contiguous; exact (no accumulation).

2-D max pooling on a rank-4 NCHW input with **no padding** (`ops.rs:2301`). For each
`kernel_size × kernel_size` window stepped by **scalar** `stride`, emit the maximum value in the
window. `x [N, C, H, W]`; output `[N, C, Hout, Wout]` with `Hout = (H − kernel_size)/stride + 1`,
`Wout` symmetric. Channels pass through unchanged (pool is per-channel). NaN handling follows the
`T::max` reduction used in the window scan; window max is a pure max-reduction, so the result is
deterministic and same-hardware bit-stable (a max is exact — no rounding, no accumulation).

**Status — no dispatch carrier (test-only).** Per the inventory (`reference.md:212`) this kernel is
**not wired in the executor**. A graph-side `Op::MaxPool2D` exists in `fuel-core`
(`op.rs:93`, fields `kernel_size: (usize, usize)`, `stride: (usize, usize)`), but there is **no
`OpKind::MaxPool2D` and no `OpParams::MaxPool2D` in `fuel-dispatch`** (grep over
`fuel-core-types/src` and `fuel-dispatch/src` returns nothing). Note also a shape mismatch the
contract surfaces honestly: the reference oracle takes **scalar** `kernel_size`/`stride` (a square
window, no padding), whereas the graph `Op::MaxPool2D` carries **tuple** `(usize, usize)` for each
and the lazy API (`lazy.rs:3785`) additionally supports padding + pad-value — so this oracle covers
only the square-window, no-padding subset. Like `conv2d_simple`, this contract parse-validates but
is **NOT registrable as-is** (§10.7 fails: no real `OpKind`/`OpParams` variant); an importer returns
the unknown-`OpKind` / `BadOpParamsVariant` error (the `MxNotYetRegistrable`-class discipline) until
a dispatch carrier lands. Contiguous-only; no in-place; fresh output buffer.

```fkc
kernel: max_pool2d
op_kind: MaxPool2D             # forward marker only; NO such OpKind/OpParams in fuel-dispatch (graph Op::MaxPool2D
                               # exists, op.rs:93, but no dispatch carrier; test-only oracle) — describe-only (§3.10)
registrable: false             # §3.10 — documentation-only; not registered, op_kind/op_params not required to resolve
blurb: "Per-channel 2-D max pool, square window, no padding, scalar stride; f32/f64/bf16/f16; contiguous; exact (no accumulation)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::max_pool2d_f32"   # one per dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [N, C, H, W]
  op_params:
    variant: MaxPool2D            # [consumer-ahead] NO such OpParams variant; see Status
    fields:
      kernel_size: { kind: usize, constraint: "kernel_size > 0 && kernel_size <= x.dim[2] && kernel_size <= x.dim[3]" }
      stride:      { kind: usize, constraint: "stride > 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)        # output dtype = input dtype (T)
      shape_rule: "from_params(N=x.dim[0], C=x.dim[1], Hout=(x.dim[2]-kernel_size)/stride+1, Wout=(x.dim[3]-kernel_size)/stride+1)"
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # Judge bootstraps; the bytes/flops below are the structural prior
  class: reduction
  # comparisons = N * C * Hout * Wout * kernel_size^2 (one max-compare per window element)
  flops: "x.dim[0] * x.dim[1] * out.dim[2] * out.dim[3] * kernel_size * kernel_size"
  bytes_moved: "(x.dim[0]*x.dim[1]*x.dim[2]*x.dim[3] + out.dim[0]*out.dim[1]*out.dim[2]*out.dim[3]) * dtype_bytes"
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "out.dim[0]*out.dim[1]*out.dim[2]*out.dim[3] * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # max is exact (selection, not accumulation): bit-identical everywhere
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure max-reduction over the window: no rounding, no accumulation. Window max is exact; result is bitwise-stable. NaN follows T::max semantics."

determinism: same_hardware_bitwise
```

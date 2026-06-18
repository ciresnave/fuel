---
fkc_version: 1
provider:
  name: fuel-conv
  backend: Cpu                       # maps to BackendId::Cpu
  kernel_source: "fuel-conv-reference"   # the BindingEntry.kernel_source tag
  link_registry: fuel_conv::fkc::ENTRY_POINTS   # §12.6 symbol → KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-conv — kernel contracts (conv-attn family)

Reference 2D-convolution primitives Fuel itself provides in `fuel-conv`. All three operate on
**raw `&[T]` host slices** (`T: num_traits::Float`, f32/f64 in practice), NOT `Tensor`/`Layout`:
there is no Layout type in scope, so layout is an *unenforced convention* (NCHW `x`, OIHW `weight`,
`[Cout]` bias, **row-major contiguous, zero offset assumed**). A non-contiguous / offset / strided
caller silently produces wrong results — there is no `is_contiguous()` / `StridedIndex` guard, only
`debug_assert_eq!` length checks in debug builds. None of the three support **dilation** (documented
unsupported, `lib.rs:40`), nor half / integer dtypes (the `Float` bound excludes them). All three
call `ConvShape::validate().expect(...)` and therefore **panic on a malformed `ConvShape`** — a
known violation of the never-panic-on-production-paths rule, surfaced here rather than hidden.

These are the parity oracle the GPU/vendor backends verify against, not a placement target the
planner would normally pick. Each cost block carries a derivable FLOP/byte formula hint plus a
genuine true-zero `overhead_ns: 0` (a CPU host call has no launch cost), so `provenance: declared`
is the honest marker — an author prior the Judge later refines (§4.4) — not a fabricated constant.

## conv2d_direct  (textbook nested-loop conv2d forward; the parity oracle)

One-line: Reference conv2d forward by direct nested loops over `(N, groups, Cout/g, Hout, Wout, Cin/g, Kh, Kw)`; f32/f64 host slices, contiguous NCHW/OIHW only.

Direct textbook 2D convolution forward pass (`fuel-conv/src/lib.rs:137`,
`conv2d_direct<T: Float>(x, weight, bias: Option<&[T]>, s: &ConvShape, out: &mut [T])`). The
seven-deep nested loop walks `batch × groups × c_out_per_g × h_out × w_out × c_in_per_g × k_h × k_w`
and is the parity oracle every other backend's conv is checked against. It reads x as NCHW
`[batch, c_in, h, w]`, weight as OIHW `[c_out, c_in/groups, k_h, k_w]`, optional bias as `[c_out]`,
all row-major contiguous with zero offset *by convention* — flat offsets are computed manually
(`x_off`, `w_off`, `lib.rs:174-175`); no stride / broadcast / offset / negative-stride handling
exists (there is no `Layout`). The reduction accumulates straight in `T` (`acc = acc + x*w`, no
higher-precision accumulator, no Kahan) in a fixed `channel → ky → kx` order, then adds bias after
the reduction. **Numerics:** deterministic same-hardware accumulation order; no widening accumulator,
so accuracy is exactly native-`T` summation. **Output:** `[batch, c_out, h_out, w_out]` row-major
contiguous, `h_out = (h + 2*pad_h - k_h)/stride_h + 1` (ditto width, `lib.rs:75-80`); every output
element is written exactly once (no accumulation), so pre-zeroing is **not** required (`lib.rs:136`).
**Limitations:** no dilation; no half/int dtypes; contiguous-zero-offset-only and unenforced;
`s.validate().expect()` **panics** on a malformed `ConvShape` (`lib.rs:144`); length mismatches are
`debug_assert` only (no release guard).

```fkc
kernel: conv2d_direct
op_kind: Conv2D                # OpKind::Conv2D (fuel-core-types/src/dispatch.rs:122)
blurb: "Reference conv2d forward by direct nested loops over (N, groups, Cout/g, Hout, Wout, Cin/g, Kh, Kw); f32/f64 host slices, contiguous NCHW/OIHW only."
backend: Cpu
kernel_source: "fuel-conv-reference"
entry_point: "fuel_conv::conv2d_direct"   # generic over T; resolved per dtype, §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64]          # num_traits::Float; no half/int (Float bound)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                      # NCHW [batch, c_in, h, w]; row-major, zero-offset by convention
    - name: weight
      dtypes: [F32, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                      # OIHW [c_out, c_in/groups, k_h, k_w]
      shape_constraint: "divisible(x.dim[1], op_params.groups)"   # c_in % groups == 0
    - name: bias
      dtypes: [F32, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                      # [c_out]
      optional: true
  op_params:
    variant: Conv2D               # OpParams::Conv2D (fuel-dispatch/src/kernel.rs:222)
    fields:
      x_shape:   { kind: "[usize; 4]" }
      w_shape:   { kind: "[usize; 4]" }
      out_shape: { kind: "[usize; 4]" }
      stride:    { kind: "(usize, usize)", note: "asymmetric (h, w) supported" }
      padding:   { kind: "(usize, usize)", note: "asymmetric (h, w); applied to both sides" }
      dilation:  { kind: "(usize, usize)", constraint: "== (1, 1)", note: "fuel-conv has NO dilation support (lib.rs:40); only (1,1) is honest for this kernel" }
      groups:    { kind: usize, constraint: ">= 1; c_in % groups == 0; c_out % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)            # out: &mut [T], same T as x
      shape_rule: conv2d(op_params)         # [batch, c_out, h_out, w_out]
      layout_guarantee: contiguous          # caller-owned buffer sized s.output_len(); written once, no pre-zero
      aliasing: none                        # no aliasing with inputs

caps:
  awkward_layout_strategy: requires_contiguous   # rejects non-contiguous; planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost (§4.3)
  fast_paths:
    - { when: "op_params.groups == 1", note: "dense conv; no per-group slicing" }
    - { when: "depthwise", note: "groups == c_in == c_out; per-channel loop" }
  in_place: false
  alignment_bytes: 8                # CPU host-slice; native T alignment
  access_granularity_bits: 8

cost:
  provenance: declared              # carries the true-zero overhead_ns: 0 (a CPU host call, no launch) — an author prior the Judge later refines (§4.4)
  class: conv
  # conv2d MACs = N * Cout * Hout * Wout * (Cin/groups) * Kh * Kw, ×2 for FLOPs (mul + add)
  flops: "2 * out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * (x_shape[1] / groups) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  overhead_ns: 0                    # genuine true-zero: plain host function call, no launch overhead (legit declared zero, not a fabricated constant)
  memory: { device_bytes: 0, host_bytes: "out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3] * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic channel→ky→kx accumulation order, no atomics
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "straight native-T accumulation (no widening accumulator, no Kahan); deterministic reduction order; bias added after reduction; f32/f64 only."

determinism: same_hardware_bitwise
```

## im2col  (conv patch extraction into the im2col matrix; data movement only)

One-line: Extracts conv patches into the im2col matrix (channel, ky, kx)-ordered with zero-fill padding; pure data movement, lossless, f32/f64 host slices.

Input-rearrangement primitive (`fuel-conv/src/lib.rs:221`, `im2col<T: Float>(x, s: &ConvShape, out: &mut [T])`).
It extracts the conv patches into the im2col matrix so a vendor BLAS gemm can carry the conv
arithmetic — **the matmul step is explicitly not in this crate.** x is NCHW `[batch, c_in, h, w]`,
row-major contiguous, zero-offset by convention; flat indices are computed manually
(`x_channel_offset`, `lib.rs:249, 265`) with no Layout / stride / broadcast / offset inspection.
Out-of-bounds (padding) positions are zero-filled (`lib.rs:258-268`). **Numerics:** none — pure data
movement plus `T::zero()` fill for padding; lossless. **Output:** flattened length
`s.im2col_len() = batch * groups * (c_in_per_g * k_h * k_w) * (h_out * w_out)` (`lib.rs:117-122`),
logical layout `[batch*groups, c_in_per_g*k_h*k_w, h_out*w_out]` with inner patch-axis ordering
`(channel, ky, kx)` to line up with the weight reshape's K-dimension (`lib.rs:217-220`); row-major
contiguous, every element written exactly once (padding → `T::zero()`), no pre-zeroing required.
**Limitations:** no dilation; no half/int dtypes; contiguous-zero-offset-only and unenforced;
`s.validate().expect()` **panics** on a malformed `ConvShape` (`lib.rs:226`); length mismatch is
`debug_assert` only.

NOTE (faithful to as-built): there is **no** as-built `OpKind::Im2Col` / `OpParams::Im2Col`
variant — im2col is a fuel-conv free function parameterized by the crate-local `ConvShape`
(`fuel-conv/src/lib.rs:48`), not a Fuel dispatch op. Per the never-invent / never-re-number
discipline (§0, invariant 10), the `op_kind:` slot below names the **closest honest dispatch tag**,
`Conv2D` (`OpKind::Conv2D`, `fuel-core-types/src/dispatch.rs:122`) — the only `OpKind` im2col
participates in (it is the patch-extraction sub-step `conv2d_via_gemm` lowers through) — and is
flagged **[consumer-ahead]** rather than fabricating an `Im2Col` variant. This contract records that
the param carrier is `ConvShape` (not a `fuel-dispatch::OpParams` variant); until a dedicated
`Im2Col` op-kind lands, the §10.7 op-param namespace check has no matching variant and this section
is a documentation / authoring record of the primitive rather than a registrable dispatch entry.

```fkc
kernel: im2col
registrable: false            # §3.10 describe-only: no as-built OpKind::Im2Col and op_params.variant ConvShape is NOT a real OpParams variant (rule-7); im2col is OpKind::Conv2D's patch-extraction lowering sub-step, not a dispatched op. Skips rule-2 (op_kind/fused_op) + rule-7 (op-param namespace); all descriptive facts still validated.
op_kind: Conv2D               # [consumer-ahead] forward marker — the carrier is OpKind::Conv2D's im2col lowering; no as-built OpKind::Im2Col (see prose note above); param carrier is fuel-conv ConvShape
blurb: "Extracts conv patches into the im2col matrix (channel, ky, kx)-ordered with zero-fill padding; pure data movement, lossless, f32/f64 host slices."
backend: Cpu
kernel_source: "fuel-conv-reference"
entry_point: "fuel_conv::im2col"   # generic over T; resolved per dtype, §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64]          # num_traits::Float; no half/int
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                      # NCHW [batch, c_in, h, w]; row-major, zero-offset by convention
  op_params:
    variant: ConvShape            # fuel-conv ConvShape (lib.rs:48); NOT a fuel-dispatch OpParams variant (see prose note)
    fields:
      batch:   { kind: usize }
      c_in:    { kind: usize }
      c_out:   { kind: usize }
      h:       { kind: usize }
      w:       { kind: usize }
      k_h:     { kind: usize }
      k_w:     { kind: usize }
      stride:  { kind: "(usize, usize)", note: "asymmetric (h, w)" }
      padding: { kind: "(usize, usize)", note: "asymmetric (h, w), both sides" }
      groups:  { kind: usize, constraint: ">= 1; c_in % groups == 0; c_out % groups == 0" }
      # NOTE: ConvShape carries NO dilation field (lib.rs:40, unsupported).

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)            # out: &mut [T], same T as x
      shape_rule: from_params("[batch*groups, c_in_per_g*k_h*k_w, h_out*w_out]; flat len s.im2col_len()")
      layout_guarantee: contiguous          # caller-owned, written once (padding → zero), no pre-zero
      aliasing: none                        # no aliasing with x

caps:
  awkward_layout_strategy: requires_contiguous   # rejects non-contiguous x; planner inserts Op::Contiguize + sums its cost (§4.3)
  fast_paths:
    - { when: "op_params.padding == (0, 0)", note: "no out-of-bounds branch; no zero-fill" }
  in_place: false
  alignment_bytes: 8
  access_granularity_bits: 8

cost:
  provenance: declared              # carries the true-zero overhead_ns: 0 (a CPU host call, no launch) — an author prior the Judge later refines (§4.4)
  class: strided_elementwise        # pure data movement / gather with zero-fill; bandwidth-bound, no arithmetic
  flops: "0"                        # no arithmetic; lossless rearrangement
  # writes s.im2col_len() elements, reads each into-bounds source element once
  bytes_moved: "2 * (batch * groups * (c_in / groups) * k_h * k_w * h_out * w_out) * dtype_bytes"
  overhead_ns: 0                    # genuine true-zero: plain host function call, no launch overhead (legit declared zero)
  memory: { device_bytes: 0, host_bytes: "(batch * groups * (c_in / groups) * k_h * k_w * h_out * w_out) * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # pure copy + T::zero() fill; no arithmetic, lossless
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "no arithmetic — exact data movement plus T::zero() padding fill; lossless and bitwise across compatible hardware."

determinism: bitwise               # exact copy/zero-fill; bit-identical on any compatible hardware
```

## conv2d_via_gemm  (conv2d forward = im2col + caller-provided gemm + optional bias)

One-line: Full conv2d forward as im2col + a caller-supplied gemm (c = a @ b, no accumulate) per (batch, group) + optional bias add; reduction precision is the caller's gemm.

Full conv2d forward (`fuel-conv/src/lib.rs:300`,
`conv2d_via_gemm<T: Float, F: FnMut(usize,usize,usize,&[T],&[T],&mut [T])>(x, weight, bias, s, out, patches_scratch, gemm)`)
= an `im2col` patch extraction followed by a **caller-provided** gemm invoked once per
`(batch, group)` pair, then an optional bias add. This lets AOCL / oneMKL / the reference backend
plug their own `c = a @ b` without re-writing the im2col loop. x is NCHW, weight OIHW, bias `[c_out]`,
all row-major contiguous zero-offset *by convention* (no Layout / stride / offset / broadcast
handling); the caller also supplies a `patches_scratch` buffer of `s.im2col_len()` and the per-group
slices into weight / patches / out are computed by flat offsets (`lib.rs:331-338`). **Gemm
contract:** `m = cout_per_group`, `n = h_out*w_out`, `k = cin_per_group*k_h*k_w`; weight slice
`[m,k]` row-major, patches slice `[k,n]` row-major, out slice `[m,n]` row-major; the gemm MUST do
`c = a @ b` with **no accumulate** (overwrites — the caller need not pre-zero), and the bias add
happens here afterward, in place (`lib.rs:293-294, 343-355`). **Numerics:** im2col is lossless; the
reduction precision is **whatever the caller's gemm does** — it is *not* controlled by this kernel —
and the bias add is in `T`. **Output:** `[batch, c_out, h_out, w_out]` row-major contiguous, sized
`s.output_len()`; each `(batch, group)` block `[m,n]` is written by the gemm callback. **Aliasing:**
`out` and `patches_scratch` are caller-owned and must be distinct; overwrite-vs-accumulate
correctness is delegated to the gemm callback. **Limitations:** no dilation; no half/int dtypes;
contiguous-zero-offset-only and unenforced; `s.validate().expect()` **panics** on a malformed
`ConvShape` (`lib.rs:312`); bias length is `debug_assert` only.

NOTE (faithful to as-built): this maps onto `OpKind::Conv2D` for the conv geometry, but it carries
two facts no `OpParams::Conv2D` variant models — (1) a **caller gemm callback** (an `FnMut`, not
serializable data — it is the per-`(batch,group)` `c = a @ b` reduction backend, and is the reason
the reduction precision is the caller's, not this kernel's) and (2) a caller-owned
`patches_scratch` buffer of `s.im2col_len()`. Neither is a tensor operand nor an op-param; both are
recorded in prose and in `caps`/`notes` here, since FKC has no slot for a function-pointer param
(P9 forbids pointers in a contract). The contract therefore advertises the conv geometry and the
im2col+bias steps honestly; the gemm step's precision is explicitly **delegated** (see `precision`).

```fkc
kernel: conv2d_via_gemm
op_kind: Conv2D               # OpKind::Conv2D (conv geometry); gemm reduction is a caller callback (see prose note)
blurb: "Full conv2d forward as im2col + a caller-supplied gemm (c = a @ b, no accumulate) per (batch, group) + optional bias add; reduction precision is the caller's gemm."
backend: Cpu
kernel_source: "fuel-conv-reference"
entry_point: "fuel_conv::conv2d_via_gemm"   # generic over T and the gemm FnMut; resolved per dtype, §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64]          # num_traits::Float; no half/int
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                      # NCHW [batch, c_in, h, w]; row-major, zero-offset by convention
    - name: weight
      dtypes: [F32, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                      # OIHW [c_out, c_in/groups, k_h, k_w]
      shape_constraint: "divisible(x.dim[1], op_params.groups)"   # c_in % groups == 0
    - name: bias
      dtypes: [F32, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                      # [c_out]; added in place after gemm
      optional: true
  op_params:
    variant: Conv2D               # OpParams::Conv2D (fuel-dispatch/src/kernel.rs:222)
    fields:
      x_shape:   { kind: "[usize; 4]" }
      w_shape:   { kind: "[usize; 4]" }
      out_shape: { kind: "[usize; 4]" }
      stride:    { kind: "(usize, usize)", note: "asymmetric (h, w)" }
      padding:   { kind: "(usize, usize)", note: "asymmetric (h, w), both sides" }
      dilation:  { kind: "(usize, usize)", constraint: "== (1, 1)", note: "fuel-conv has NO dilation support (lib.rs:40)" }
      groups:    { kind: usize, constraint: ">= 1; c_in % groups == 0; c_out % groups == 0" }
      # NON-OPPARAM (as-built, not modeled by OpParams::Conv2D; see prose note):
      #   gemm:            caller FnMut(m, n, k, &[T] a, &[T] b, &mut [T] c) — c = a @ b, NO accumulate; reduction precision is the caller's
      #   patches_scratch: caller-owned &mut [T] of s.im2col_len(); distinct from out

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)            # out: &mut [T], same T as x
      shape_rule: conv2d(op_params)         # [batch, c_out, h_out, w_out]; sized s.output_len()
      layout_guarantee: contiguous          # caller-owned; each (batch,group) block written by gemm, bias added in place
      aliasing: none                        # out and patches_scratch caller-owned and must be distinct; no input aliasing

caps:
  awkward_layout_strategy: requires_contiguous   # rejects non-contiguous; planner inserts Op::Contiguize + sums its cost (§4.3)
  fast_paths:
    - { when: "op_params.groups == 1", note: "single gemm per batch; no per-group slicing" }
    - { when: "op_params.padding == (0, 0)", note: "im2col has no zero-fill branch" }
  in_place: false
  alignment_bytes: 8
  access_granularity_bits: 8

cost:
  provenance: declared              # carries the true-zero overhead_ns: 0 (a CPU host call, no launch) — an author prior the Judge later refines (§4.4); FLOP hint derivable, but the GEMM term's real time is the caller's gemm
  class: gemm_like
  # conv MACs via gemm: per (batch, group) a [m,k]·[k,n] matmul, summed over batch*groups; 2*M*N*K shape
  flops: "2 * out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * (x_shape[1] / groups) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  overhead_ns: 0                    # genuine true-zero: plain host calls, no launch overhead; the gemm callback's own overhead is the caller's (legit declared zero)
  memory:
    device_bytes: 0
    # output buffer + the caller-owned patches_scratch of s.im2col_len()
    host_bytes: "(out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3] + (x_shape[0] * groups * (x_shape[1] / groups) * w_shape[2] * w_shape[3] * out_shape[2] * out_shape[3])) * dtype_bytes"
    disk_bytes: 0

precision:
  bit_stable_on_same_hardware: false   # the gemm reduction is the caller's; this kernel does NOT control its precision/determinism
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "im2col is lossless and bias add is in T; the reduction precision/determinism is DELEGATED to the caller-provided gemm (c = a @ b, no accumulate) and is not controlled by this kernel — hence no static bound and not bit-stable here. This is an honest audited none(reason), per §4.8."

determinism: nondeterministic       # reduction order/precision is the caller's gemm; this kernel makes no determinism guarantee for it
```

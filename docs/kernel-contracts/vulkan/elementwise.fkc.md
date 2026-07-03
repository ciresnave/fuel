---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                                  # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"                    # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"                    # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — elementwise (unary / binary / affine / clamp / powi / add_assign_scaled) kernel contracts

The Vulkan elementwise compute family, **re-authored per-op** so each concrete Fuel `OpKind` binds
its own section (the importer registers exactly ONE `op_kind` per registrable section). Slang/GLSL
sources live in `fuel-kernels-source/kernels/*.slang`, AOT-compiled to SPIR-V in
`fuel-vulkan-kernels/spv/*.spv`; the Rust dispatch wrappers live in `fuel-vulkan-backend/src/lib.rs`
and the `fuel-dispatch` `vulkan_dispatch` adapters.

**Re-author note (2026-07-03).** This file previously used a REPRESENTATIVE-CHASSIS pattern: one
`unary` section (`op_kind: ReluElementwise # representative`) standing in for the 16-op `op_id`
selector, likewise `unary_f16/f64/bf16` and `binary*`. A representative section can register only ONE
of its 16 (or 6) OpKinds, so it was NOT migratable. Following the **CPU inplace-unary-affine
precedent** (`docs/kernel-contracts/cpu/inplace-unary-affine.fkc.md`, which went 1→21 per-op
unary sections), the representatives are SUPERSEDED by per-op sections:

- **Unary** (16 ops: Neg/Sqr/Sqrt/Exp/Log/Sin/Cos/Tanh/Sigmoid/Silu/Gelu(tanh)/Relu/Step/Abs/Sign/
  Recip): **caps split by dtype**. The f32/f16/f64 variants are strided+broadcast capable, so each op
  is ONE section fanning `[F32, F16, F64]` (§3.4 dtype-fan, base `entry_point` →
  `<op>_{f32,f16,f64}`). The **bf16** variant is a **contiguous-only** pair-thread kernel — different
  caps — so it is a SEPARATE single-dtype `<op>_bf16` section (16 + 16 sections).
- **Binary** (6 ops: Add/Sub/Mul/Div/Maximum/Minimum): ALL four dtypes (incl. the lane-masked
  `binary_bf16`) are strided, so each op is ONE section fanning `[F32, F16, F64, BF16]` (6 sections).
- **Affine** (`y = mul·x + add`): f32/f16/f64 strided (one fanning section) + a contiguous-only
  `affine_bf16` section.
- **Clamp** (f32) / **PowI** (f32): single-dtype strided sections.
- The old `unary` / `binary` representatives are retained below as **`registrable: false`** (§3.10)
  describe-only chassis umbrellas, and **`add_assign_scaled`** stays describe-only (no real
  `OpKind`).

**Caps ride through the import truthfully (§6 / caps_map).** Each section's per-operand five-flag
layout set projects onto `KernelCaps.strided_input = (strided==accepted) && (broadcast_stride0==accepted)`
(AND-ed across operands), stamped onto the binding by the importer. So the strided sections yield
`strided_input=true` (byte-for-byte the deleted `register_with_caps_and_precision(strided)` regs) and
the bf16-unary / affine_bf16 sections yield `strided_input=false` (the deleted plain
`register_with_precision` regs). **`reverse_strides: rejected` everywhere** (no kernel walks a signed
stride; a flipped view is normalized upstream). Output contiguity is universal.

**Cost is `judge_measured` for every kernel** — GPU dispatch overhead/occupancy are device-specific;
the bandwidth-bound elementwise formula hints (`flops ≈ n`, `bytes_moved ≈ k·n·dtype_bytes`) are the
only author-derivable structure. **Precision** is an author-declared seed (`audited: false` ⇒ lowers
to `PrecisionGuarantee::UNAUDITED`; the Judge audits later) — the same posture the Vulkan cast
migration took (the hand-written `VULKAN_{FLOAT,HALF,TRANSCENDENTAL}_POINTWISE_PRECISION` consts are
retired from this seam). `Gelu` in the selector is the **tanh** approximation, NOT erf.

---

## unary  (shared 16-op elementwise-unary chassis — describe-only umbrella)

The shared in-Slang 16-op `op_id` selector chassis (`unary.slang`): `out[i] = op(in[i])` where a
single dtype-monomorphized kernel backs the whole selector keyed by `op_id` (0..15 →
Neg/Sqr/Sqrt/Exp/Log/Sin/Cos/Tanh/Sigmoid/Silu/Gelu(tanh)/Relu/Step/Abs/Sign/Recip). This umbrella
documents the shared shape/loop/precision contract that every concrete per-op section above
specializes; it binds **no** `OpKind` of its own (each named op pins one distinct
`<Op>Elementwise` OpKind), so it is **`registrable: false`** (§3.10 describe-only) and registers no
binding. The f32/f16/f64 variants are strided+broadcast (rank-4 stride Params + `flags` bit0); the
bf16 variant is the contiguous-only packed-u32 pair-thread kernel (documented per-op in the
`*_bf16` sections). All math is f32 (f64 native double). `Gelu` = tanh approximation.

```fkc
kernel: unary
registrable: false            # §3.10 describe-only: shared 16-op selector chassis, NOT a dispatch target
op_kind: ~                    # the chassis binds no OpKind; each named op above pins one
blurb: "Shared 16-op elementwise-unary op_id selector chassis out[i]=op(in[i]); f32/f16/f64 strided, bf16 contiguous; not separately dispatchable."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::unary"   # the generic selector; never resolved (describe-only); §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Generic 16-op selector chassis; per-op numerics in the specialized sections. f32/f64 native, f16 float16_t, bf16 packed-u32; all f32 math. Gelu = tanh approximation."

determinism: same_hardware_bitwise
```

## neg  (NegElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = neg(in[i])` over an f32 / f16 / f64 buffer (-x; exact). Backs the
single Fuel `OpKind::NegElementwise` (`op_id=0` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: neg
op_kind: NegElementwise
blurb: "Elementwise unary out[i]=neg(in[i]) (f32/f16/f64); strided+broadcast; op_id=0."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::neg"   # base; §3.4 fans neg_{f32,f16,f64} → unary::/unary_f16::/unary_f64::neg_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "-x; exact. f32 native; f16 native float16_t (f32 intermediate); f64 native double. Exact/pointwise arithmetic."

determinism: same_hardware_bitwise
```

## sqr  (SqrElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = sqr(in[i])` over an f32 / f16 / f64 buffer (x*x; exact). Backs the
single Fuel `OpKind::SqrElementwise` (`op_id=1` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: sqr
op_kind: SqrElementwise
blurb: "Elementwise unary out[i]=sqr(in[i]) (f32/f16/f64); strided+broadcast; op_id=1."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::sqr"   # base; §3.4 fans sqr_{f32,f16,f64} → unary::/unary_f16::/unary_f64::sqr_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x*x; exact. f32 native; f16 native float16_t (f32 intermediate); f64 native double. Exact/pointwise arithmetic."

determinism: same_hardware_bitwise
```

## sqrt  (SqrtElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = sqrt(in[i])` over an f32 / f16 / f64 buffer (√x; NaN for x<0). Backs the
single Fuel `OpKind::SqrtElementwise` (`op_id=2` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: sqrt
op_kind: SqrtElementwise
blurb: "Elementwise unary out[i]=sqrt(in[i]) (f32/f16/f64); strided+broadcast; op_id=2."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::sqrt"   # base; §3.4 fans sqrt_{f32,f16,f64} → unary::/unary_f16::/unary_f64::sqrt_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "√x; NaN for x<0. f32 native; f16 native float16_t (f32 intermediate); f64 native double. Exact/pointwise arithmetic."

determinism: same_hardware_bitwise
```

## exp  (ExpElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = exp(in[i])` over an f32 / f16 / f64 buffer (e^x). Backs the
single Fuel `OpKind::ExpElementwise` (`op_id=3` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: exp
op_kind: ExpElementwise
blurb: "Elementwise unary out[i]=exp(in[i]) (f32/f16/f64); strided+broadcast; op_id=3."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::exp"   # base; §3.4 fans exp_{f32,f16,f64} → unary::/unary_f16::/unary_f64::exp_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "e^x. f32 native; f16 native float16_t (f32 intermediate); f64 native double. Transcendental (Exp/Log/Sin/Cos/Tanh/Sigmoid/Silu/Gelu) not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## log  (LogElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = log(in[i])` over an f32 / f16 / f64 buffer (ln(x); NaN for x<0, -inf at 0). Backs the
single Fuel `OpKind::LogElementwise` (`op_id=4` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: log
op_kind: LogElementwise
blurb: "Elementwise unary out[i]=log(in[i]) (f32/f16/f64); strided+broadcast; op_id=4."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::log"   # base; §3.4 fans log_{f32,f16,f64} → unary::/unary_f16::/unary_f64::log_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "ln(x); NaN for x<0, -inf at 0. f32 native; f16 native float16_t (f32 intermediate); f64 native double. Transcendental (Exp/Log/Sin/Cos/Tanh/Sigmoid/Silu/Gelu) not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## sin  (SinElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = sin(in[i])` over an f32 / f16 / f64 buffer (sin(x)). Backs the
single Fuel `OpKind::SinElementwise` (`op_id=5` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: sin
op_kind: SinElementwise
blurb: "Elementwise unary out[i]=sin(in[i]) (f32/f16/f64); strided+broadcast; op_id=5."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::sin"   # base; §3.4 fans sin_{f32,f16,f64} → unary::/unary_f16::/unary_f64::sin_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "sin(x). f32 native; f16 native float16_t (f32 intermediate); f64 native double. Transcendental (Exp/Log/Sin/Cos/Tanh/Sigmoid/Silu/Gelu) not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## cos  (CosElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = cos(in[i])` over an f32 / f16 / f64 buffer (cos(x)). Backs the
single Fuel `OpKind::CosElementwise` (`op_id=6` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: cos
op_kind: CosElementwise
blurb: "Elementwise unary out[i]=cos(in[i]) (f32/f16/f64); strided+broadcast; op_id=6."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cos"   # base; §3.4 fans cos_{f32,f16,f64} → unary::/unary_f16::/unary_f64::cos_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "cos(x). f32 native; f16 native float16_t (f32 intermediate); f64 native double. Transcendental (Exp/Log/Sin/Cos/Tanh/Sigmoid/Silu/Gelu) not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## tanh  (TanhElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = tanh(in[i])` over an f32 / f16 / f64 buffer (tanh(x)). Backs the
single Fuel `OpKind::TanhElementwise` (`op_id=7` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: tanh
op_kind: TanhElementwise
blurb: "Elementwise unary out[i]=tanh(in[i]) (f32/f16/f64); strided+broadcast; op_id=7."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::tanh"   # base; §3.4 fans tanh_{f32,f16,f64} → unary::/unary_f16::/unary_f64::tanh_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "tanh(x). f32 native; f16 native float16_t (f32 intermediate); f64 native double. Transcendental (Exp/Log/Sin/Cos/Tanh/Sigmoid/Silu/Gelu) not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## sigmoid  (SigmoidElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = sigmoid(in[i])` over an f32 / f16 / f64 buffer (1/(1+exp(-x))). Backs the
single Fuel `OpKind::SigmoidElementwise` (`op_id=8` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: sigmoid
op_kind: SigmoidElementwise
blurb: "Elementwise unary out[i]=sigmoid(in[i]) (f32/f16/f64); strided+broadcast; op_id=8."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::sigmoid"   # base; §3.4 fans sigmoid_{f32,f16,f64} → unary::/unary_f16::/unary_f64::sigmoid_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "1/(1+exp(-x)). f32 native; f16 native float16_t (f32 intermediate); f64 native double. Transcendental (Exp/Log/Sin/Cos/Tanh/Sigmoid/Silu/Gelu) not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## silu  (SiluElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = silu(in[i])` over an f32 / f16 / f64 buffer (x*sigmoid(x)). Backs the
single Fuel `OpKind::SiluElementwise` (`op_id=9` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: silu
op_kind: SiluElementwise
blurb: "Elementwise unary out[i]=silu(in[i]) (f32/f16/f64); strided+broadcast; op_id=9."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::silu"   # base; §3.4 fans silu_{f32,f16,f64} → unary::/unary_f16::/unary_f64::silu_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x*sigmoid(x). f32 native; f16 native float16_t (f32 intermediate); f64 native double. Transcendental (Exp/Log/Sin/Cos/Tanh/Sigmoid/Silu/Gelu) not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## gelu  (GeluElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = gelu(in[i])` over an f32 / f16 / f64 buffer (GELU tanh approximation (NOT erf)). Backs the
single Fuel `OpKind::GeluElementwise` (`op_id=10` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: gelu
op_kind: GeluElementwise
blurb: "Elementwise unary out[i]=gelu(in[i]) (f32/f16/f64); strided+broadcast; op_id=10."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::gelu"   # base; §3.4 fans gelu_{f32,f16,f64} → unary::/unary_f16::/unary_f64::gelu_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "GELU tanh approximation (NOT erf). f32 native; f16 native float16_t (f32 intermediate); f64 native double. Transcendental (Exp/Log/Sin/Cos/Tanh/Sigmoid/Silu/Gelu) not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## relu  (ReluElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = relu(in[i])` over an f32 / f16 / f64 buffer (max(0,x)). Backs the
single Fuel `OpKind::ReluElementwise` (`op_id=11` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: relu
op_kind: ReluElementwise
blurb: "Elementwise unary out[i]=relu(in[i]) (f32/f16/f64); strided+broadcast; op_id=11."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::relu"   # base; §3.4 fans relu_{f32,f16,f64} → unary::/unary_f16::/unary_f64::relu_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "max(0,x). f32 native; f16 native float16_t (f32 intermediate); f64 native double. Exact/pointwise arithmetic."

determinism: same_hardware_bitwise
```

## step  (StepElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = step(in[i])` over an f32 / f16 / f64 buffer (x>0 ? 1 : 0). Backs the
single Fuel `OpKind::StepElementwise` (`op_id=12` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: step
op_kind: StepElementwise
blurb: "Elementwise unary out[i]=step(in[i]) (f32/f16/f64); strided+broadcast; op_id=12."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::step"   # base; §3.4 fans step_{f32,f16,f64} → unary::/unary_f16::/unary_f64::step_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x>0 ? 1 : 0. f32 native; f16 native float16_t (f32 intermediate); f64 native double. Exact/pointwise arithmetic."

determinism: same_hardware_bitwise
```

## abs  (AbsElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = abs(in[i])` over an f32 / f16 / f64 buffer (|x|). Backs the
single Fuel `OpKind::AbsElementwise` (`op_id=13` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: abs
op_kind: AbsElementwise
blurb: "Elementwise unary out[i]=abs(in[i]) (f32/f16/f64); strided+broadcast; op_id=13."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::abs"   # base; §3.4 fans abs_{f32,f16,f64} → unary::/unary_f16::/unary_f64::abs_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "|x|. f32 native; f16 native float16_t (f32 intermediate); f64 native double. Exact/pointwise arithmetic."

determinism: same_hardware_bitwise
```

## sign  (SignElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = sign(in[i])` over an f32 / f16 / f64 buffer (sign(x) in {-1,0,1}). Backs the
single Fuel `OpKind::SignElementwise` (`op_id=14` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: sign
op_kind: SignElementwise
blurb: "Elementwise unary out[i]=sign(in[i]) (f32/f16/f64); strided+broadcast; op_id=14."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::sign"   # base; §3.4 fans sign_{f32,f16,f64} → unary::/unary_f16::/unary_f64::sign_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "sign(x) in {-1,0,1}. f32 native; f16 native float16_t (f32 intermediate); f64 native double. Exact/pointwise arithmetic."

determinism: same_hardware_bitwise
```

## recip  (RecipElementwise — f32 / f16 / f64, strided)

Element-wise unary `out[i] = recip(in[i])` over an f32 / f16 / f64 buffer (1/x; ±inf at 0). Backs the
single Fuel `OpKind::RecipElementwise` (`op_id=15` in the shared 16-op selector kernel). The Slang kernel
carries a rank-4 `(shape0..3, in_s0..3)` Params block + a `flags` bit0 contiguity flag, so it is
**strided + broadcast capable, offset-incapable** (a non-zero `byte_offset` is realized by an upstream
`Op::Contiguize`). Output: same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: recip
op_kind: RecipElementwise
blurb: "Elementwise unary out[i]=recip(in[i]) (f32/f16/f64); strided+broadcast; op_id=15."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::recip"   # base; §3.4 fans recip_{f32,f16,f64} → unary::/unary_f16::/unary_f64::recip_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params: { variant: None }   # OpParams::None — per-element params ride the wrapper (§3.7)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "1/x; ±inf at 0. f32 native; f16 native float16_t (f32 intermediate); f64 native double. Exact/pointwise arithmetic."

determinism: same_hardware_bitwise
```

## neg_bf16  (NegElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = neg(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: neg_bf16
op_kind: NegElementwise
blurb: "Elementwise unary out[i]=neg(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=0."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::neg_bf16"   # single-dtype, resolved AS-IS → unary_bf16::neg_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "-x; exact. bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Exact/pointwise arithmetic. Requires even n."

determinism: same_hardware_bitwise
```

## sqr_bf16  (SqrElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = sqr(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: sqr_bf16
op_kind: SqrElementwise
blurb: "Elementwise unary out[i]=sqr(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=1."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::sqr_bf16"   # single-dtype, resolved AS-IS → unary_bf16::sqr_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x*x; exact. bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Exact/pointwise arithmetic. Requires even n."

determinism: same_hardware_bitwise
```

## sqrt_bf16  (SqrtElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = sqrt(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: sqrt_bf16
op_kind: SqrtElementwise
blurb: "Elementwise unary out[i]=sqrt(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=2."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::sqrt_bf16"   # single-dtype, resolved AS-IS → unary_bf16::sqrt_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "√x; NaN for x<0. bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Exact/pointwise arithmetic. Requires even n."

determinism: same_hardware_bitwise
```

## exp_bf16  (ExpElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = exp(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: exp_bf16
op_kind: ExpElementwise
blurb: "Elementwise unary out[i]=exp(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=3."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::exp_bf16"   # single-dtype, resolved AS-IS → unary_bf16::exp_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "e^x. bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Transcendentals not bit-stable cross-hardware. Requires even n."

determinism: same_hardware_bitwise
```

## log_bf16  (LogElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = log(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: log_bf16
op_kind: LogElementwise
blurb: "Elementwise unary out[i]=log(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=4."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::log_bf16"   # single-dtype, resolved AS-IS → unary_bf16::log_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "ln(x); NaN for x<0, -inf at 0. bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Transcendentals not bit-stable cross-hardware. Requires even n."

determinism: same_hardware_bitwise
```

## sin_bf16  (SinElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = sin(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: sin_bf16
op_kind: SinElementwise
blurb: "Elementwise unary out[i]=sin(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=5."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::sin_bf16"   # single-dtype, resolved AS-IS → unary_bf16::sin_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "sin(x). bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Transcendentals not bit-stable cross-hardware. Requires even n."

determinism: same_hardware_bitwise
```

## cos_bf16  (CosElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = cos(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: cos_bf16
op_kind: CosElementwise
blurb: "Elementwise unary out[i]=cos(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=6."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cos_bf16"   # single-dtype, resolved AS-IS → unary_bf16::cos_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "cos(x). bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Transcendentals not bit-stable cross-hardware. Requires even n."

determinism: same_hardware_bitwise
```

## tanh_bf16  (TanhElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = tanh(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: tanh_bf16
op_kind: TanhElementwise
blurb: "Elementwise unary out[i]=tanh(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=7."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::tanh_bf16"   # single-dtype, resolved AS-IS → unary_bf16::tanh_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "tanh(x). bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Transcendentals not bit-stable cross-hardware. Requires even n."

determinism: same_hardware_bitwise
```

## sigmoid_bf16  (SigmoidElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = sigmoid(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: sigmoid_bf16
op_kind: SigmoidElementwise
blurb: "Elementwise unary out[i]=sigmoid(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=8."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::sigmoid_bf16"   # single-dtype, resolved AS-IS → unary_bf16::sigmoid_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "1/(1+exp(-x)). bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Transcendentals not bit-stable cross-hardware. Requires even n."

determinism: same_hardware_bitwise
```

## silu_bf16  (SiluElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = silu(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: silu_bf16
op_kind: SiluElementwise
blurb: "Elementwise unary out[i]=silu(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=9."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::silu_bf16"   # single-dtype, resolved AS-IS → unary_bf16::silu_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x*sigmoid(x). bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Transcendentals not bit-stable cross-hardware. Requires even n."

determinism: same_hardware_bitwise
```

## gelu_bf16  (GeluElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = gelu(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: gelu_bf16
op_kind: GeluElementwise
blurb: "Elementwise unary out[i]=gelu(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=10."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::gelu_bf16"   # single-dtype, resolved AS-IS → unary_bf16::gelu_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "GELU tanh approximation (NOT erf). bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Transcendentals not bit-stable cross-hardware. Requires even n."

determinism: same_hardware_bitwise
```

## relu_bf16  (ReluElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = relu(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: relu_bf16
op_kind: ReluElementwise
blurb: "Elementwise unary out[i]=relu(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=11."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::relu_bf16"   # single-dtype, resolved AS-IS → unary_bf16::relu_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "max(0,x). bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Exact/pointwise arithmetic. Requires even n."

determinism: same_hardware_bitwise
```

## step_bf16  (StepElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = step(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: step_bf16
op_kind: StepElementwise
blurb: "Elementwise unary out[i]=step(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=12."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::step_bf16"   # single-dtype, resolved AS-IS → unary_bf16::step_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x>0 ? 1 : 0. bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Exact/pointwise arithmetic. Requires even n."

determinism: same_hardware_bitwise
```

## abs_bf16  (AbsElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = abs(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: abs_bf16
op_kind: AbsElementwise
blurb: "Elementwise unary out[i]=abs(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=13."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::abs_bf16"   # single-dtype, resolved AS-IS → unary_bf16::abs_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "|x|. bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Exact/pointwise arithmetic. Requires even n."

determinism: same_hardware_bitwise
```

## sign_bf16  (SignElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = sign(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: sign_bf16
op_kind: SignElementwise
blurb: "Elementwise unary out[i]=sign(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=14."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::sign_bf16"   # single-dtype, resolved AS-IS → unary_bf16::sign_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "sign(x) in {-1,0,1}. bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Exact/pointwise arithmetic. Requires even n."

determinism: same_hardware_bitwise
```

## recip_bf16  (RecipElementwise — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary `out[i] = recip(in[i])` for `BF16`. Unlike the wide variants this is the
**contiguous-only** pair-thread kernel (`unary_bf16.slang`): bf16 is packed-u16-in-u32, one thread
per u32 (two lanes), so `n` must be even. It carries no rank/stride Params — any strided / broadcast /
offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math at f32. Output: bf16, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: recip_bf16
op_kind: RecipElementwise
blurb: "Elementwise unary out[i]=recip(in[i]) (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY; n even; op_id=15."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::recip_bf16"   # single-dtype, resolved AS-IS → unary_bf16::recip_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "1/x; ±inf at 0. bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Exact/pointwise arithmetic. Requires even n."

determinism: same_hardware_bitwise
```

## binary  (shared 6-op elementwise-binary chassis — describe-only umbrella)

The shared in-Slang 6-op `op_id` selector chassis (`binary.slang`): `out[i] = op(lhs[i], rhs[i])`
where a single dtype-monomorphized kernel backs the selector keyed by `op_id` (0..5 →
Add/Sub/Mul/Div/Max/Min). Documents the shared shape/loop/precision contract that every concrete
per-op section above specializes; binds **no** `OpKind` of its own, so it is **`registrable: false`**
(§3.10). All four dtypes are per-operand strided+broadcast capable (`binary_bf16` masks single lanes
out of the packed u32). Output is the broadcasted shape, contiguous.

```fkc
kernel: binary
registrable: false            # §3.10 describe-only: shared 6-op selector chassis, NOT a dispatch target
op_kind: ~
blurb: "Shared 6-op elementwise-binary op_id selector chassis out[i]=op(lhs[i],rhs[i]); f32/f16/f64/bf16 per-operand strided; not separately dispatchable."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::binary"   # the generic selector; never resolved (describe-only); §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
    - name: rhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Generic 6-op selector chassis; per-op numerics in the specialized sections. Div IEEE inf/NaN; Max/Min NaN-as-missing."

determinism: same_hardware_bitwise
```

## add  (AddElementwise — f32 / f16 / f64 / bf16, per-operand strided)

Element-wise binary `out[i] = lhs+rhs` over f32 / f16 / f64 / bf16 buffers. Backs the single Fuel
`OpKind::AddElementwise` (`op_id=0` in the shared 6-op selector). Per-operand rank-4 strides
(`a_s0..3` / `b_s0..3`) + a `flags` field (bit0 = a contiguous, bit1 = b contiguous), so it is
**per-operand strided + broadcast capable, offset-incapable**; output shape is the broadcasted shape.
`binary_bf16` remains strided (its strided path masks single lanes out of the packed u32); `out_size`
even for bf16. Output: same dtype, broadcasted shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: add
op_kind: AddElementwise
blurb: "Elementwise binary out[i]=add(lhs[i],rhs[i]) (f32/f16/f64/bf16); per-operand strided+broadcast; op_id=0."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::add"   # base; §3.4 fans add_{f32,f16,f64,bf16} → binary::/binary_f16::/binary_f64::/binary_bf16::add_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
    - name: rhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"   # read lhs + rhs, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "lhs+rhs. f32/f64 native; f16 native float16_t; bf16 packed-u32 (lane-masked strided), all f32 math. Deterministic per-element dispatch."

determinism: same_hardware_bitwise
```

## sub  (SubElementwise — f32 / f16 / f64 / bf16, per-operand strided)

Element-wise binary `out[i] = lhs-rhs` over f32 / f16 / f64 / bf16 buffers. Backs the single Fuel
`OpKind::SubElementwise` (`op_id=1` in the shared 6-op selector). Per-operand rank-4 strides
(`a_s0..3` / `b_s0..3`) + a `flags` field (bit0 = a contiguous, bit1 = b contiguous), so it is
**per-operand strided + broadcast capable, offset-incapable**; output shape is the broadcasted shape.
`binary_bf16` remains strided (its strided path masks single lanes out of the packed u32); `out_size`
even for bf16. Output: same dtype, broadcasted shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: sub
op_kind: SubElementwise
blurb: "Elementwise binary out[i]=sub(lhs[i],rhs[i]) (f32/f16/f64/bf16); per-operand strided+broadcast; op_id=1."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::sub"   # base; §3.4 fans sub_{f32,f16,f64,bf16} → binary::/binary_f16::/binary_f64::/binary_bf16::sub_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
    - name: rhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"   # read lhs + rhs, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "lhs-rhs. f32/f64 native; f16 native float16_t; bf16 packed-u32 (lane-masked strided), all f32 math. Deterministic per-element dispatch."

determinism: same_hardware_bitwise
```

## mul  (MulElementwise — f32 / f16 / f64 / bf16, per-operand strided)

Element-wise binary `out[i] = lhs*rhs` over f32 / f16 / f64 / bf16 buffers. Backs the single Fuel
`OpKind::MulElementwise` (`op_id=2` in the shared 6-op selector). Per-operand rank-4 strides
(`a_s0..3` / `b_s0..3`) + a `flags` field (bit0 = a contiguous, bit1 = b contiguous), so it is
**per-operand strided + broadcast capable, offset-incapable**; output shape is the broadcasted shape.
`binary_bf16` remains strided (its strided path masks single lanes out of the packed u32); `out_size`
even for bf16. Output: same dtype, broadcasted shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: mul
op_kind: MulElementwise
blurb: "Elementwise binary out[i]=mul(lhs[i],rhs[i]) (f32/f16/f64/bf16); per-operand strided+broadcast; op_id=2."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::mul"   # base; §3.4 fans mul_{f32,f16,f64,bf16} → binary::/binary_f16::/binary_f64::/binary_bf16::mul_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
    - name: rhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"   # read lhs + rhs, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "lhs*rhs. f32/f64 native; f16 native float16_t; bf16 packed-u32 (lane-masked strided), all f32 math. Deterministic per-element dispatch."

determinism: same_hardware_bitwise
```

## div  (DivElementwise — f32 / f16 / f64 / bf16, per-operand strided)

Element-wise binary `out[i] = lhs/rhs; IEEE inf/NaN` over f32 / f16 / f64 / bf16 buffers. Backs the single Fuel
`OpKind::DivElementwise` (`op_id=3` in the shared 6-op selector). Per-operand rank-4 strides
(`a_s0..3` / `b_s0..3`) + a `flags` field (bit0 = a contiguous, bit1 = b contiguous), so it is
**per-operand strided + broadcast capable, offset-incapable**; output shape is the broadcasted shape.
`binary_bf16` remains strided (its strided path masks single lanes out of the packed u32); `out_size`
even for bf16. Output: same dtype, broadcasted shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: div
op_kind: DivElementwise
blurb: "Elementwise binary out[i]=div(lhs[i],rhs[i]) (f32/f16/f64/bf16); per-operand strided+broadcast; op_id=3."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::div"   # base; §3.4 fans div_{f32,f16,f64,bf16} → binary::/binary_f16::/binary_f64::/binary_bf16::div_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
    - name: rhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"   # read lhs + rhs, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "lhs/rhs; IEEE inf/NaN. f32/f64 native; f16 native float16_t; bf16 packed-u32 (lane-masked strided), all f32 math. Deterministic per-element dispatch."

determinism: same_hardware_bitwise
```

## maximum  (MaximumElementwise — f32 / f16 / f64 / bf16, per-operand strided)

Element-wise binary `out[i] = max(lhs,rhs); NaN-as-missing` over f32 / f16 / f64 / bf16 buffers. Backs the single Fuel
`OpKind::MaximumElementwise` (`op_id=4` in the shared 6-op selector). Per-operand rank-4 strides
(`a_s0..3` / `b_s0..3`) + a `flags` field (bit0 = a contiguous, bit1 = b contiguous), so it is
**per-operand strided + broadcast capable, offset-incapable**; output shape is the broadcasted shape.
`binary_bf16` remains strided (its strided path masks single lanes out of the packed u32); `out_size`
even for bf16. Output: same dtype, broadcasted shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: maximum
op_kind: MaximumElementwise
blurb: "Elementwise binary out[i]=maximum(lhs[i],rhs[i]) (f32/f16/f64/bf16); per-operand strided+broadcast; op_id=4."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::maximum"   # base; §3.4 fans maximum_{f32,f16,f64,bf16} → binary::/binary_f16::/binary_f64::/binary_bf16::maximum_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
    - name: rhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"   # read lhs + rhs, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "max(lhs,rhs); NaN-as-missing. f32/f64 native; f16 native float16_t; bf16 packed-u32 (lane-masked strided), all f32 math. Deterministic per-element dispatch."

determinism: same_hardware_bitwise
```

## minimum  (MinimumElementwise — f32 / f16 / f64 / bf16, per-operand strided)

Element-wise binary `out[i] = min(lhs,rhs); NaN-as-missing` over f32 / f16 / f64 / bf16 buffers. Backs the single Fuel
`OpKind::MinimumElementwise` (`op_id=5` in the shared 6-op selector). Per-operand rank-4 strides
(`a_s0..3` / `b_s0..3`) + a `flags` field (bit0 = a contiguous, bit1 = b contiguous), so it is
**per-operand strided + broadcast capable, offset-incapable**; output shape is the broadcasted shape.
`binary_bf16` remains strided (its strided path masks single lanes out of the packed u32); `out_size`
even for bf16. Output: same dtype, broadcasted shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: minimum
op_kind: MinimumElementwise
blurb: "Elementwise binary out[i]=minimum(lhs[i],rhs[i]) (f32/f16/f64/bf16); per-operand strided+broadcast; op_id=5."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::minimum"   # base; §3.4 fans minimum_{f32,f16,f64,bf16} → binary::/binary_f16::/binary_f64::/binary_bf16::minimum_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
    - name: rhs
      dtypes: [F32, F16, F64, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # per-dtype: 32 (f32) / 16 (f16) / 64 (f64); shared metadata across the fan

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"   # read lhs + rhs, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "min(lhs,rhs); NaN-as-missing. f32/f64 native; f16 native float16_t; bf16 packed-u32 (lane-masked strided), all f32 math. Deterministic per-element dispatch."

determinism: same_hardware_bitwise
```

## affine  (Affine y = x·mul + add — f32 / f16 / f64, strided)

Element-wise affine `out[i] = mul · in[i] + add` over an f32 / f16 / f64 buffer; backs `AddScalar`
(`mul=1`), `MulScalar` (`add=0`), and the general `Affine`. Same rank-4 stride Params + `flags` bit0
contiguity model as `unary`, so **strided + broadcast capable, offset-incapable**. `(mul, add)` arrive
on `OpParams::Affine { mul: f64, add: f64 }`, consumed at f32 (native f64 for the f64 variant). Output:
same dtype, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: affine
op_kind: Affine
blurb: "Elementwise affine y = mul*x + add (f32/f16/f64); covers AddScalar/MulScalar; strided+broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::affine"   # base; §3.4 fans affine_{f32,f16,f64} → affine::affine_<dt>; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F16, F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params:
    variant: Affine              # OpParams::Affine { mul: f64, add: f64 }
    fields:
      out_size: { kind: usize, note: "= n, the output element count" }
      flags:    { kind: u32, note: "bit0 = input contiguous fast path" }
      mul:      { kind: f64, note: "consumed at f32 (native f64 for the f64 variant)" }
      add:      { kind: f64, note: "consumed at f32 (native f64 for the f64 variant)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32/f16 widen to f32 mul-add then narrow; f64 native double. mul=1 ⇒ AddScalar; add=0 ⇒ MulScalar. Deterministic per-element dispatch."

determinism: same_hardware_bitwise
```

## affine_bf16  (Affine y = x·mul + add — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise affine for `BF16` — the **contiguous-only** pair-thread kernel (packed-u32, one thread
per u32 = two bf16 lanes). No rank/stride Params, so any strided / broadcast / offset input is realized
by an upstream `Op::Contiguize` first (`awkward_layout_strategy: requires_contiguous`). `mul`/`add`
arrive f64 (`OpParams::Affine`), narrow to f32. Output: bf16, input shape, contiguous, fresh buffer,
no aliasing.

```fkc
kernel: affine_bf16
op_kind: Affine
blurb: "Elementwise affine y = mul*x + add (bf16 packed-u32 pair-thread; f32 math); CONTIGUOUS-ONLY."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::affine_bf16"   # single-dtype, resolved AS-IS → affine::affine_bf16; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Affine
    fields:
      out_size: { kind: usize, note: "pair-thread; one u32 per thread = 2 bf16 lanes" }
      mul:      { kind: f64, note: "narrowed to f32 for the half compute" }
      add:      { kind: f64, note: "narrowed to f32 for the half compute" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16 packed-u32; widen to f32, mul-add in f32 (params f64→f32), narrow to bf16 on store. mul=1 ⇒ AddScalar; add=0 ⇒ MulScalar."

determinism: same_hardware_bitwise
```

## clamp  (ClampElementwise y = clamp(x, lo, hi) — f32)

Element-wise bounded clamp `out[i] = clamp(in[i], lo, hi)` over an f32 buffer (`clamp.slang`).
**f32 only.** Rank-4 stride Params + `flags` bit0, so **strided + broadcast capable, offset-incapable**.
`(lo, hi)` arrive on `OpParams::Clamp { min: f64, max: f64 }`, consumed at f32. Output: f32, input
shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: clamp
op_kind: ClampElementwise
blurb: "Elementwise clamp(x, lo, hi) (f32 only); strided+broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::clamp_f32"   # single-dtype, resolved AS-IS → clamp::clamp_f32; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params:
    variant: Clamp               # OpParams::Clamp { min: f64, max: f64 }
    fields:
      out_size: { kind: usize }
      flags:    { kind: u32, note: "bit0 = input contiguous fast path" }
      lo:       { kind: f64, note: "min bound; consumed at f32" }
      hi:       { kind: f64, note: "max bound; consumed at f32" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32 clamp; exact (no rounding). Deterministic per-element dispatch."

determinism: same_hardware_bitwise
```

## powi  (PowIElementwise y = x^exp — f32)

Element-wise integer power `out[i] = in[i]^exp` over an f32 buffer (`powi.slang`). **f32 only.**
Special-cases `e == 0/1/2/3` (direct multiplies) else `pow`; `pow(0,-k) → +inf` matches the CPU
reference. Rank-4 stride Params + `flags` bit0, so **strided + broadcast capable, offset-incapable**.
`exp` arrives on `OpParams::PowI { exp: i32 }`. Output: f32, input shape, contiguous, fresh buffer,
no aliasing.

```fkc
kernel: powi
op_kind: PowIElementwise
blurb: "Elementwise integer power y = x^exp (f32 only); e=0/1/2/3 special-cased else pow; strided+broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::powi_f32"   # single-dtype, resolved AS-IS → powi::powi_f32; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params:
    variant: PowI                # OpParams::PowI { exp: i32 }
    fields:
      out_size: { kind: usize }
      flags:    { kind: u32, note: "bit0 = input contiguous fast path" }
      exp:      { kind: i32, note: "e=0/1/2/3 special-cased; else pow" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32. e=0/1/2/3 by direct multiply; else GLSL pow (not bit-stable cross-hardware). pow(0,-k) -> +inf matches CPU."

determinism: same_hardware_bitwise
```

## add_assign_scaled  (in-place dst[i] += src[i]·scale — f32, describe-only)

In-place scaled accumulate `dst[i] += src[i] · scale` over two equal-length f32 buffers
(`add_assign_scaled.slang`). **f32 only.** Binding 0 is the read-write `dst` (the in-place output);
binding 1 is `src`. Element-aligned 1:1, **contiguous-only**. `scale` is an `f` param. No atomics
(one thread per element, distinct outputs), so bit-stable on same hardware. Kept **`registrable:
false`** (§3.10): `AddAssignScaled` is a graph-rewrite / in-place accumulate with no real `OpKind` and
`OpParams::AddAssignScaled` is not a real variant — documented, not registered.

```fkc
kernel: add_assign_scaled
registrable: false            # §3.10 describe-only: no real OpKind (graph rewrite / in-place accumulate); op_kind/op_params below are forward-looking markers.
op_kind: AddAssignScaled
blurb: "In-place scaled accumulate dst[i] += src[i]*scale (f32 only); contiguous; dst is RW (output aliases dst)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::add_assign_scaled"   # source add_assign_scaled.slang; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: dst                  # binding 0: READ-WRITE accumulator; the in-place output buffer
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=src
    - name: src                  # binding 1: read-only addend source
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=dst
  op_params:
    variant: AddAssignScaled     # n, scale (f)
    fields:
      n:     { kind: usize, note: "element count; dst and src element-aligned 1:1" }
      scale: { kind: f32 }

return:
  outputs:
    - name: dst
      dtype_rule: passthrough(dst)
      shape_rule: same_as(dst)
      layout_guarantee: contiguous
      aliasing: in_place(dst)    # output IS dst's buffer (caps.in_place: true, §4.6)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true                 # binding 0 (dst) is RW; output aliases dst (§4.6)
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"
  bytes_moved: "3 * n * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32 dst + src*scale; plain per-element accumulate (no atomics); IEEE inf/NaN. Bit-stable on same hardware."

determinism: same_hardware_bitwise
```

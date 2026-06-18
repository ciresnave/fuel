---
fkc_version: 1
provider:
  name: fuel-reference-backend
  backend: Cpu                        # the reference oracle runs on the host (BackendId::Cpu)
  kernel_source: "reference-oracle"   # the BindingEntry.kernel_source tag
  link_registry: fuel_reference_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"       # provider build id, folded into kernel_revision_hash
---

# fuel-reference-backend — cast kernel contracts

Directional dtype-conversion kernels from `fuel-reference-backend/src/ops.rs` (`// ---------- dtype
casts ----------`, `ops.rs:1132-1241`). One logical `Cast` op (`OpKind::Cast`; `OpParams::Cast`
unit variant) is realized as **distinct concrete-typed directional kernels** — each (src dtype →
dst dtype) pair is its own function, **not** a generic `T`, because the input and output element
widths differ (so the output byte size differs from the input). The executor routes by the target
dtype via `eval_cast` (`exec.rs:1000`, dispatched from `Op::Cast(target)` at `exec.rs:618`); the
per-operand ordered (src, dst) dtype slots make each direction a distinct dispatch key (§12.1).

Implementation shape (all sixteen): each kernel takes a `&RefTensor<Src>`, walks its flat
contiguous slice with `x.as_slice().iter().map(|&v| <convert>(v)).collect()`, and returns a fresh
`RefTensor::from_vec(data, x.shape().clone())` of the target dtype. There is no `Layout`, no
strides, no offset, no in-place path — `RefTensor<T>` (`lib.rs:68`) is *always* a contiguous,
row-major buffer with zero offset (inventory §"Crate-wide layout invariant"). The input-layout
contract is therefore **contiguous, offset 0, row-major** for every kernel; any strided / broadcast
/ offset view must be materialized into a fresh contiguous `RefTensor` by the planner *before* the
kernel runs (the reference crate is an oracle with no internal contiguize). The output is always a
**fresh contiguous buffer**, fully written, with no input/output aliasing. Element count is
preserved across the cast; only the per-element byte width changes.

Three precision regimes appear below:

- **Lossless widening** (`max_ulp: 0`) — the source is a strict subset of the target
  (`f32→f64`, `bf16→f32`, `bf16→f64`, `f16→f32`, `f16→f64`, `u32→f64`).
- **Lossy narrowing** — a single IEEE / `half`-crate round per element
  (`f64→f32`, `f32→bf16`, `f32→f16`, `f64→bf16`, `f64→f16`, and the via-f32 `bf16→f16` / `f16→bf16`,
  whose f32 pivot leg is lossless and whose loss is on the narrowing leg only).
- **Float→uint truncation** (`f32→u32`, `f64→u32`) — `as`-cast truncation toward zero; values
  outside `[0, u32::MAX]` are out-of-range / implementation-defined and must not occur in
  well-formed graphs. `u32→f32` is exact below 2^24 and rounds above.

> **COST PROVENANCE.** Every cost block below is marked `provenance: judge_measured` — the Judge
> bootstraps these by measurement (§4.4). No author cost numbers are fabricated; the only cost
> hints carried are the derivable elementwise-bandwidth `bytes_moved` (read N×src_bytes + write
> N×dst_bytes) and `flops: "0"` (a cast does no arithmetic, only a per-element format conversion).

## cast_f32_to_f64  (f32 → f64)

Convert `f32` → `f64`. **Lossless widening** — every f32 value is exactly representable in f64.

Walks a contiguous, zero-offset, row-major `f32` slice and collects `out[i] = in[i] as f64` into a
fresh contiguous `f64` `RefTensor` (`ops.rs:1135`). Output is 2× the input byte size (8 bytes/elem
vs 4) with the same element count. The Rust `as` widening is the identity embedding f32 ⊂ f64, so it
is exact and bit-stable on the same hardware. Bandwidth-bound elementwise op: read N×4 bytes, write
N×8 bytes. Known limitation: contiguous-only — any strided/broadcast/offset operand must be
materialized first.

```fkc
kernel: cast_f32_to_f64
op_kind: Cast
blurb: "Cast f32 -> f64; contiguous; lossless widening."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_f32_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F64)          # target dtype lives on the output Storage; key-pinned (§5.1)
      shape_rule: same_as(src)        # element count preserved
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # Judge bootstraps; coefficients below are bandwidth hints
  class: cheap_elementwise
  flops: "0"                          # pure widen; no arithmetic
  bytes_moved: "n * (4 + 8)"          # read N*4 (f32) + write N*8 (f64); elementwise = bandwidth-bound
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact: f32 is a strict subset of f64
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossless widening; f32 strictly subset of f64; exact, deterministic widen via Rust `as`."

determinism: bitwise
```

## cast_f32_to_bf16  (f32 → bf16)

Convert `f32` → `bf16`. **Lossy narrowing** — keeps the f32 exponent and the top mantissa bits.

Walks a contiguous `f32` slice and collects `out[i] = half::bf16::from_f32(in[i])` into a fresh
contiguous `bf16` `RefTensor` (`ops.rs:1141`). Output is half the input byte size (2 bytes/elem vs
4). bf16 shares f32's 8-bit exponent, so the conversion is exponent-preserving; the 23-bit f32
mantissa is rounded to bf16's 7 mantissa bits (rounding per the `half` crate's `bf16::from_f32`).
Bandwidth-bound elementwise op. Deterministic on the same hardware (a single round per element).
Contiguous-only.

```fkc
kernel: cast_f32_to_bf16
op_kind: Cast
blurb: "Cast f32 -> bf16; contiguous; lossy narrowing (keeps exponent, top mantissa)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_f32_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BF16)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (4 + 2)"          # read N*4 (f32) + write N*2 (bf16)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy narrowing; preserves f32 exponent, rounds 23-bit mantissa to bf16's 7 bits via half::bf16::from_f32; deterministic per element."

determinism: same_hardware_bitwise
```

## cast_f32_to_f16  (f32 → f16)

Convert `f32` → `f16`. **Lossy narrowing** — clips to f16 range with NaN/inf preserved.

Walks a contiguous `f32` slice and collects `out[i] = half::f16::from_f32(in[i])` into a fresh
contiguous `f16` `RefTensor` (`ops.rs:1147`). Output is half the input byte size (2 bytes/elem vs
4). f16 has a 5-bit exponent and 10-bit mantissa: values exceeding f16's finite range saturate to
±inf, subnormals/zero round per the `half` crate's `f16::from_f32`, NaN is preserved. Bandwidth-bound
elementwise op. Deterministic on the same hardware. Contiguous-only.

```fkc
kernel: cast_f32_to_f16
op_kind: Cast
blurb: "Cast f32 -> f16; contiguous; lossy narrowing (clip to f16 range, NaN/inf preserved)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_f32_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F16)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (4 + 2)"          # read N*4 (f32) + write N*2 (f16)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy narrowing; out-of-range saturates to +/-inf, NaN preserved, rounds via half::f16::from_f32; deterministic per element."

determinism: same_hardware_bitwise
```

## cast_f64_to_f32  (f64 → f32)

Convert `f64` → `f32`. **Lossy narrowing** per IEEE-754 round-to-nearest-even.

Walks a contiguous `f64` slice and collects `out[i] = in[i] as f32` into a fresh contiguous `f32`
`RefTensor` (`ops.rs:1153`). Output is half the input byte size (4 bytes/elem vs 8). Rounding follows
the platform IEEE-754 `f64`→`f32` `as` conversion; values outside f32's range saturate to ±inf, NaN
is preserved. Bandwidth-bound elementwise op. Deterministic on the same hardware (a single round per
element). Contiguous-only.

```fkc
kernel: cast_f64_to_f32
op_kind: Cast
blurb: "Cast f64 -> f32; contiguous; lossy IEEE narrowing."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_f64_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (8 + 4)"          # read N*8 (f64) + write N*4 (f32)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy narrowing; IEEE-754 round-to-nearest-even; overflow saturates to +/-inf, NaN preserved; deterministic single round per element."

determinism: same_hardware_bitwise
```

## cast_f64_to_bf16  (f64 → bf16)

Convert `f64` → `bf16`. **Lossy narrowing** — double-precision down to bf16's 8-bit exp / 7-bit mantissa.

Walks a contiguous `f64` slice and collects `out[i] = half::bf16::from_f64(in[i])` into a fresh
contiguous `bf16` `RefTensor` (`ops.rs:1159`). Output is one quarter the input byte size (2 bytes/elem
vs 8). The conversion is a single `half::bf16::from_f64` round per element: bf16's exponent range
covers f64's normal range so large magnitudes saturate to ±inf, and the f64 mantissa is rounded to
bf16's 7 bits, NaN preserved. Bandwidth-bound elementwise op. Deterministic on the same hardware.
Contiguous-only.

```fkc
kernel: cast_f64_to_bf16
op_kind: Cast
blurb: "Cast f64 -> bf16; contiguous; lossy narrowing (single round)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_f64_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BF16)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (8 + 2)"          # read N*8 (f64) + write N*2 (bf16)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy narrowing; single half::bf16::from_f64 round per element; overflow saturates to +/-inf, NaN preserved; deterministic."

determinism: same_hardware_bitwise
```

## cast_f64_to_f16  (f64 → f16)

Convert `f64` → `f16`. **Lossy narrowing** — double-precision down to f16's 5-bit exp / 10-bit mantissa.

Walks a contiguous `f64` slice and collects `out[i] = half::f16::from_f64(in[i])` into a fresh
contiguous `f16` `RefTensor` (`ops.rs:1165`). Output is one quarter the input byte size (2 bytes/elem
vs 8). A single `half::f16::from_f64` round per element: f16's narrow finite range means most
large-magnitude f64 values saturate to ±inf, the f64 mantissa rounds to f16's 10 bits, NaN
preserved. Bandwidth-bound elementwise op. Deterministic on the same hardware. Contiguous-only.

```fkc
kernel: cast_f64_to_f16
op_kind: Cast
blurb: "Cast f64 -> f16; contiguous; lossy narrowing (single round)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_f64_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F16)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (8 + 2)"          # read N*8 (f64) + write N*2 (f16)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy narrowing; single half::f16::from_f64 round per element; out-of-range saturates to +/-inf, NaN preserved; deterministic."

determinism: same_hardware_bitwise
```

## cast_bf16_to_f32  (bf16 → f32)

Convert `bf16` → `f32`. **Lossless widening** — bf16 is a strict subset of f32.

Walks a contiguous `bf16` slice and collects `out[i] = in[i].to_f32()` into a fresh contiguous `f32`
`RefTensor` (`ops.rs:1171`). Output is 2× the input byte size (4 bytes/elem vs 2). bf16 shares f32's
exponent and is a 16-bit truncation of f32, so every bf16 value is exactly representable in f32 — the
conversion is exact. Bandwidth-bound elementwise op. Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: cast_bf16_to_f32
op_kind: Cast
blurb: "Cast bf16 -> f32; contiguous; lossless widening."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_bf16_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (2 + 4)"          # read N*2 (bf16) + write N*4 (f32)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact: bf16 strict subset of f32
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossless widening; bf16 strict subset of f32 (16-bit truncation widened back); exact, deterministic."

determinism: bitwise
```

## cast_bf16_to_f64  (bf16 → f64)

Convert `bf16` → `f64`. **Lossless widening** — bf16 is a strict subset of f64.

Walks a contiguous `bf16` slice and collects `out[i] = in[i].to_f64()` into a fresh contiguous `f64`
`RefTensor` (`ops.rs:1177`). Output is 4× the input byte size (8 bytes/elem vs 2). bf16's exponent
and mantissa both fit within f64, so every bf16 value is exactly representable in f64 — the
conversion is exact. Bandwidth-bound elementwise op. Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: cast_bf16_to_f64
op_kind: Cast
blurb: "Cast bf16 -> f64; contiguous; lossless widening."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_bf16_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F64)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (2 + 8)"          # read N*2 (bf16) + write N*8 (f64)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact: bf16 strict subset of f64
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossless widening; bf16 exponent+mantissa fit f64; exact, deterministic via half::bf16::to_f64."

determinism: bitwise
```

## cast_bf16_to_f16  (bf16 → f16, via f32)

Convert `bf16` → `f16`. **Lossy** (different mantissa/exponent layouts); routes through f32.

Walks a contiguous `bf16` slice and collects `out[i] = half::f16::from_f32(in[i].to_f32())` into a
fresh contiguous `f16` `RefTensor` (`ops.rs:1184`). Same byte size in and out (2 bytes/elem). The
two-leg pivot first widens bf16→f32 (exact, bf16 ⊂ f32) then narrows f32→f16 (lossy: bf16's wider
8-bit exponent does not fit f16's 5-bit exponent, so large magnitudes saturate to ±inf, and the
mantissa is re-rounded). Only the second leg loses precision; NaN preserved. Bandwidth-bound
elementwise op. Deterministic on the same hardware. Contiguous-only.

```fkc
kernel: cast_bf16_to_f16
op_kind: Cast
blurb: "Cast bf16 -> f16 via f32; contiguous; lossy on the f16 leg (exponent does not fit)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_bf16_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F16)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (2 + 2)"          # read N*2 (bf16) + write N*2 (f16)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Two-leg pivot bf16->f32 (lossless, bf16 subset of f32) then f32->f16 (lossy: bf16's 8-bit exponent does not fit f16's 5-bit; saturates to +/-inf, re-rounds mantissa); NaN preserved; deterministic per element."

determinism: same_hardware_bitwise
```

## cast_f16_to_f32  (f16 → f32)

Convert `f16` → `f32`. **Lossless widening** — f16 is a strict subset of f32.

Walks a contiguous `f16` slice and collects `out[i] = in[i].to_f32()` into a fresh contiguous `f32`
`RefTensor` (`ops.rs:1194`). Output is 2× the input byte size (4 bytes/elem vs 2). Every f16 value
(finite, subnormal, inf, NaN) is exactly representable in f32, so the conversion is exact.
Bandwidth-bound elementwise op. Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: cast_f16_to_f32
op_kind: Cast
blurb: "Cast f16 -> f32; contiguous; lossless widening."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_f16_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (2 + 4)"          # read N*2 (f16) + write N*4 (f32)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact: f16 strict subset of f32
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossless widening; every f16 value exactly representable in f32; exact, deterministic."

determinism: bitwise
```

## cast_f16_to_f64  (f16 → f64)

Convert `f16` → `f64`. **Lossless widening** — f16 is a strict subset of f64.

Walks a contiguous `f16` slice and collects `out[i] = in[i].to_f64()` into a fresh contiguous `f64`
`RefTensor` (`ops.rs:1200`). Output is 4× the input byte size (8 bytes/elem vs 2). f16's exponent and
mantissa both fit within f64, so every f16 value is exactly representable in f64 — the conversion is
exact. Bandwidth-bound elementwise op. Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: cast_f16_to_f64
op_kind: Cast
blurb: "Cast f16 -> f64; contiguous; lossless widening."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_f16_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F64)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (2 + 8)"          # read N*2 (f16) + write N*8 (f64)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact: f16 strict subset of f64
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossless widening; f16 exponent+mantissa fit f64; exact, deterministic via half::f16::to_f64."

determinism: bitwise
```

## cast_f16_to_bf16  (f16 → bf16, via f32)

Convert `f16` → `bf16`. **Lossy** (different mantissa/exponent layouts); routes through f32.

Walks a contiguous `f16` slice and collects `out[i] = half::bf16::from_f32(in[i].to_f32())` into a
fresh contiguous `bf16` `RefTensor` (`ops.rs:1206`). Same byte size in and out (2 bytes/elem). The
two-leg pivot first widens f16→f32 (exact, f16 ⊂ f32) then narrows f32→bf16 (lossy: bf16's 7-bit
mantissa is narrower than f16's 10-bit, so the mantissa is re-rounded; bf16's wider exponent fully
covers f16's range). Only the second leg loses precision; NaN preserved. Bandwidth-bound elementwise
op. Deterministic on the same hardware. Contiguous-only.

```fkc
kernel: cast_f16_to_bf16
op_kind: Cast
blurb: "Cast f16 -> bf16 via f32; contiguous; lossy on the bf16 leg (mantissa re-rounded)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_f16_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BF16)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (2 + 2)"          # read N*2 (f16) + write N*2 (bf16)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Two-leg pivot f16->f32 (lossless, f16 subset of f32) then f32->bf16 (lossy: bf16's 7-bit mantissa narrower than f16's 10-bit, re-rounds; bf16 exponent covers f16 range); NaN preserved; deterministic per element."

determinism: same_hardware_bitwise
```

## cast_u32_to_f32  (u32 → f32)

Convert `u32` → `f32`. **Exact below 2^24**, rounds above (f32 has a 24-bit mantissa).

Walks a contiguous `u32` slice and collects `out[i] = in[i] as f32` into a fresh contiguous `f32`
`RefTensor` (`ops.rs:1218`). Same byte size in and out (4 bytes/elem). Integers below 2^24 round-trip
exactly (small label indices and counts are always lossless); values at or above 2^24 are rounded to
the nearest representable f32 per IEEE round-to-nearest-even. Bandwidth-bound elementwise op.
Deterministic on the same hardware. Contiguous-only.

```fkc
kernel: cast_u32_to_f32
op_kind: Cast
blurb: "Cast u32 -> f32; contiguous; exact below 2^24, rounds above."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_u32_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (4 + 4)"          # read N*4 (u32) + write N*4 (f32)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~                          # exact below 2^24; >=2^24 rounds (not a strict 0-ULP claim)
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Integer->float; exact for u32 values < 2^24 (f32 24-bit mantissa); values >= 2^24 round to nearest representable f32 (IEEE round-to-nearest-even); deterministic."

determinism: same_hardware_bitwise
```

## cast_u32_to_f64  (u32 → f64)

Convert `u32` → `f64`. **Lossless** — f64's 53-bit mantissa covers the full u32 range exactly.

Walks a contiguous `u32` slice and collects `out[i] = in[i] as f64` into a fresh contiguous `f64`
`RefTensor` (`ops.rs:1224`). Output is 2× the input byte size (8 bytes/elem vs 4). Every u32 value
fits within f64's 53-bit mantissa, so the conversion is exact across the entire u32 range.
Bandwidth-bound elementwise op. Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: cast_u32_to_f64
op_kind: Cast
blurb: "Cast u32 -> f64; contiguous; lossless (full u32 range exact)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_u32_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F64)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (4 + 8)"          # read N*4 (u32) + write N*8 (f64)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact: full u32 range fits f64's 53-bit mantissa
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Integer->float; lossless across the full u32 range (f64 53-bit mantissa covers u32); exact, deterministic."

determinism: bitwise
```

## cast_f32_to_u32  (f32 → u32)

Convert `f32` → `u32` via **truncation toward zero**. Out-of-range values are implementation-defined.

Walks a contiguous `f32` slice and collects `out[i] = in[i] as u32` into a fresh contiguous `u32`
`RefTensor` (`ops.rs:1232`). Same byte size in and out (4 bytes/elem). The Rust `as` float→int cast
truncates toward zero (drops the fractional part). Values outside `[0, u32::MAX]` produce
implementation-defined results and must not occur in well-formed graphs (Rust `as` saturates NaN→0
and clamps out-of-range, but the contract treats out-of-range as a caller error per the inventory's
"out-of-range UB" note). Bandwidth-bound elementwise op. Deterministic on the same hardware.
Contiguous-only.

```fkc
kernel: cast_f32_to_u32
op_kind: Cast
blurb: "Cast f32 -> u32; contiguous; trunc-toward-zero (out-of-range is caller error)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_f32_to_u32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (4 + 4)"          # read N*4 (f32) + write N*4 (u32)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~                          # integer result; truncation toward zero (not a ULP claim)
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Float->uint truncation toward zero (drops fractional part); values outside [0, u32::MAX] are out-of-range/caller-error (must not occur in well-formed graphs); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_f64_to_u32  (f64 → u32)

Convert `f64` → `u32` via **truncation toward zero**. Out-of-range values are implementation-defined.

Walks a contiguous `f64` slice and collects `out[i] = in[i] as u32` into a fresh contiguous `u32`
`RefTensor` (`ops.rs:1238`). Output is half the input byte size (4 bytes/elem vs 8). The Rust `as`
float→int cast truncates toward zero. Values outside `[0, u32::MAX]` produce implementation-defined
results and must not occur in well-formed graphs (per the inventory's "trunc-toward-zero" note).
Bandwidth-bound elementwise op. Deterministic on the same hardware. Contiguous-only.

```fkc
kernel: cast_f64_to_u32
op_kind: Cast
blurb: "Cast f64 -> u32; contiguous; trunc-toward-zero (out-of-range is caller error)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::cast_f64_to_u32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (8 + 4)"          # read N*8 (f64) + write N*4 (u32)
  overhead_ns: ~                      # launch cost not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~                          # integer result; truncation toward zero (not a ULP claim)
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Float->uint truncation toward zero (drops fractional part); values outside [0, u32::MAX] are out-of-range/caller-error (must not occur in well-formed graphs); deterministic per element."

determinism: same_hardware_bitwise
```

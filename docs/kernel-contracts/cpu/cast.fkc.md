---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                        # maps to BackendId::Cpu
  kernel_source: "portable-cpu"       # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"        # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — cast kernel contracts

Directional dtype-conversion kernels from `fuel-cpu-backend/src/byte_kernels.rs`. One logical
`Cast` op (`OpKind::Cast`, `fuel-core-types/src/dispatch.rs:117`; `OpParams::Cast` unit variant,
`fuel-dispatch/src/kernel.rs:352`) is realized as **distinct directional kernels** — each
(src dtype → dst dtype) pair is a separate kernel because the input and output element widths
differ, so the output byte size differs from the input. The binding-table lookup is keyed on the
**target** dtype (= the Node's dtype, `fuel-dispatch/src/kernel.rs:115-116`); the per-operand
ordered (src, dst) dtype slots make each direction a distinct dispatch key (§12.1).

This contract covers the **COMPLETE directed-pair matrix**: every ordered pair of the 11 real
numeric dtypes {F32, F64, F16, BF16, F8E4M3, U8, I8, U32, I16, I32, I64}, identity pairs excluded
(the optimizer elides `Cast` where src == dst), = **11 × 10 = 110 kernels**. The MX dummy dtypes
(F6E2M3, F6E3M2, F4, F8E8M0) have no Rust scalar type and are out of scope. The first twelve sections
below (the float/fp8 f32-hub + f8-spoke pairs) were authored first; the remaining ninety-eight (all
integer casts + the non-f32-hub float pairs) complete the matrix and retire the last hand-written
`table.register(Cast, …)` regs — the family is now **fully contract-sourced**.

Three implementation families back these 110 kernels:

- **`cast_kernel!`** (`byte_kernels.rs:3392-3469`, +3578-…) — `bytemuck::Pod`-to-`Pod` element
  conversion (all float↔float and all int↔int / int↔float / float↔int pairs whose element types are
  `Pod`). Validates `input.len_bytes() % in_elem_size == 0` and
  `out.len_bytes() == elem_count * out_elem_size`, then walks `out[i] = convert(in[i])`. Integer
  conversions use Rust's `as` (float→int truncates toward zero and **saturates** out-of-range
  magnitudes; int→int narrowing wraps two's-complement; widening is exact). Anything touching a
  half or F8E4M3 format pivots through f32 (`half::{f16,bf16}::from_f32` / `.to_f32()`).
- **`cast_kernel_to_fp8!` / `cast_kernel_from_fp8!`** (`byte_kernels.rs:3481-3570`) — `float8::F8E4M3`
  is 1 byte and does **not** implement `bytemuck::Pod`, so these handle F8E4M3 as raw `u8` via
  `from_bits` / `to_bits`. `to_fp8` validates `out.len_bytes() == elem_count` (1 byte/elem);
  `from_fp8` validates `out.len_bytes() == elem_count * out_elem_size`. Every direction touching
  F8E4M3 **pivots through f32** (`float8::F8E4M3` only exposes `from_f32`/`to_f32`); the f32 pivot
  leg is lossless for both f16 and bf16 (each is a strict subset of f32).
- **Per-TARGET dispatch wrappers** (`fuel-dispatch/src/dispatch.rs`, the `cpu_cast_wrapper!` macro —
  `cast_to_{f32,f64,f16,bf16,f8e4m3,u8,i8,u32,i16,i32,i64}_cpu_wrapper`). The binding-table lookup is
  keyed on the **target** dtype, so all 10 of a target's source pairs resolve to the SAME per-target
  wrapper, which `match`es on the source dtype to pick the right byte kernel. Each contract section's
  specific `cast_<src>_to_<dst>` `entry_point` maps to its target's wrapper in the `link_registry`
  (`CPU_CAST_ENTRY_POINTS`) — the synthetic-umbrella precedent: 10 distinct entry_points → 1 wrapper.

**Cross-cutting facts (inventory §"Cross-cutting facts" / §"Cast"):** every cast operates on a flat
`CpuStorageBytes` slice via `as_slice()` / `bytes()`, validates **byte length only**, and never
consults a `Layout`/strides/offset. The input-layout contract is therefore **contiguous, offset 0,
row-major** for all of them — the pipelined executor's auto-Contiguize pass realizes any
strided/broadcast/offset input into a contiguous buffer *before* the kernel runs. The output buffer
is caller-pre-allocated to the exact byte size and **fully overwritten** (no read of prior contents,
no input/output aliasing). Element count is preserved across the cast; only the byte size changes.

## cast_f32_to_f64  (f32 → f64)

Convert `f32` → `f64`. **Lossless widening** — every f32 value is exactly representable in f64.

Walks a contiguous, zero-offset, row-major `f32` buffer and writes `out[i] = in[i] as f64` into a
contiguous `f64` output (`cast_kernel!`, `byte_kernels.rs:3434-3439`). Output is 2× the input byte
size (8 bytes/elem vs 4) with the same element count. Validates `input.len_bytes() % 4 == 0` and
`out.len_bytes() == elem_count * 8`, returning a typed `Result` (never panics) on a size mismatch.
Bandwidth-bound elementwise op: read N×4 bytes, write N×8 bytes. Algorithm-exact (the Rust `as`
widening is the identity embedding f32 ⊂ f64), so it is bit-stable on the same hardware. Known
limitation: contiguous-only — any strided/broadcast/offset operand must be contiguized first.

```fkc
kernel: cast_f32_to_f64
op_kind: Cast
blurb: "Cast f32 -> f64; contiguous; lossless widening."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f32_to_f64"
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/widen; no arithmetic
  bytes_moved: "n * (4 + 8)"          # read N*4 (f32) + write N*8 (f64); elementwise = bandwidth-bound
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact: f32 is a strict subset of f64
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossless widening; f32 strictly subset of f64; exact, deterministic copy/widen."

determinism: same_hardware_bitwise
```

## cast_f64_to_f32  (f64 → f32)

Convert `f64` → `f32`. **Lossy narrowing** per IEEE-754 round-to-nearest-even.

Walks a contiguous, zero-offset `f64` buffer and writes `out[i] = in[i] as f32` into a contiguous
`f32` output (`cast_kernel!`, `byte_kernels.rs:3440-3445`). Output is half the input byte size
(4 bytes/elem vs 8). Rounding follows the platform IEEE-754 `f64`→`f32` conversion; values outside
f32's range saturate to ±inf, NaN is preserved. Validates `input.len_bytes() % 8 == 0` and
`out.len_bytes() == elem_count * 4`. Bandwidth-bound elementwise op. Deterministic on the same
hardware (a single round per element). Contiguous-only.

```fkc
kernel: cast_f64_to_f32
op_kind: Cast
blurb: "Cast f64 -> f32; contiguous; lossy IEEE narrowing."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f64_to_f32"
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (8 + 4)"          # read N*8 (f64) + write N*4 (f32)
  overhead_ns: 40
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

## cast_f32_to_bf16  (f32 → bf16)

Convert `f32` → `bf16`. **Lossy narrowing** — keeps the f32 exponent and the top mantissa bits.

Walks a contiguous `f32` buffer and writes `out[i] = bf16::from_f32(in[i])` into a contiguous
`bf16` output (`cast_kernel!`, `byte_kernels.rs:3446-3451`). Output is half the input byte size
(2 bytes/elem vs 4). bf16 shares f32's 8-bit exponent, so the conversion is exponent-preserving;
the 23-bit f32 mantissa is rounded to bf16's 7 mantissa bits (rounding per the `half` crate's
`bf16::from_f32`). Validates `input.len_bytes() % 4 == 0` and `out.len_bytes() == elem_count * 2`.
Bandwidth-bound elementwise op. Deterministic on the same hardware. Contiguous-only.

```fkc
kernel: cast_f32_to_bf16
op_kind: Cast
blurb: "Cast f32 -> bf16; contiguous; lossy narrowing (keeps exponent, top mantissa)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f32_to_bf16"
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (4 + 2)"          # read N*4 (f32) + write N*2 (bf16)
  overhead_ns: 40
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

## cast_bf16_to_f32  (bf16 → f32)

Convert `bf16` → `f32`. **Lossless widening** — bf16 is a strict subset of f32.

Walks a contiguous `bf16` buffer and writes `out[i] = in[i].to_f32()` into a contiguous `f32`
output (`cast_kernel!`, `byte_kernels.rs:3452-3457`). Output is 2× the input byte size (4 bytes/elem
vs 2). Because bf16 shares f32's exponent and is a 16-bit truncation of f32, every bf16 value is
exactly representable in f32, so the conversion is exact. Validates `input.len_bytes() % 2 == 0` and
`out.len_bytes() == elem_count * 4`. Bandwidth-bound elementwise op. Bit-stable on the same hardware.
Contiguous-only.

```fkc
kernel: cast_bf16_to_f32
op_kind: Cast
blurb: "Cast bf16 -> f32; contiguous; lossless widening."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_bf16_to_f32"
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (2 + 4)"          # read N*2 (bf16) + write N*4 (f32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact: bf16 strict subset of f32
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossless widening; bf16 strict subset of f32 (16-bit truncation widened back); exact, deterministic."

determinism: same_hardware_bitwise
```

## cast_f32_to_f16  (f32 → f16)

Convert `f32` → `f16`. **Lossy narrowing** — clips to f16 range with NaN/inf preserved.

Walks a contiguous `f32` buffer and writes `out[i] = f16::from_f32(in[i])` into a contiguous `f16`
output (`cast_kernel!`, `byte_kernels.rs:3458-3463`). Output is half the input byte size
(2 bytes/elem vs 4). f16 has a 5-bit exponent and 10-bit mantissa: values exceeding f16's finite
range saturate to ±inf, subnormals/zero round per the `half` crate's `f16::from_f32`, NaN is
preserved. Validates `input.len_bytes() % 4 == 0` and `out.len_bytes() == elem_count * 2`.
Bandwidth-bound elementwise op. Deterministic on the same hardware. Contiguous-only.

```fkc
kernel: cast_f32_to_f16
op_kind: Cast
blurb: "Cast f32 -> f16; contiguous; lossy narrowing (clip to f16 range, NaN/inf preserved)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f32_to_f16"
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (4 + 2)"          # read N*4 (f32) + write N*2 (f16)
  overhead_ns: 40
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

## cast_f16_to_f32  (f16 → f32)

Convert `f16` → `f32`. **Lossless widening** within f16's representable range.

Walks a contiguous `f16` buffer and writes `out[i] = in[i].to_f32()` into a contiguous `f32`
output (`cast_kernel!`, `byte_kernels.rs:3464-3469`). Output is 2× the input byte size (4 bytes/elem
vs 2). Every f16 value (finite, subnormal, inf, NaN) is exactly representable in f32, so the
conversion is exact. Validates `input.len_bytes() % 2 == 0` and `out.len_bytes() == elem_count * 4`.
Bandwidth-bound elementwise op. Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: cast_f16_to_f32
op_kind: Cast
blurb: "Cast f16 -> f32; contiguous; lossless widening."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f16_to_f32"
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (2 + 4)"          # read N*2 (f16) + write N*4 (f32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact: f16 strict subset of f32
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossless widening; every f16 value exactly representable in f32; exact, deterministic."

determinism: same_hardware_bitwise
```

## cast_f32_to_f8e4m3  (f32 → F8E4M3)

Convert `f32` → `F8E4M3`. **Lossy narrowing** per NV/OCP FP8 E4M3.

Walks a contiguous `f32` buffer and writes the 1-byte `F8E4M3` bit pattern
`out[i] = F8E4M3::from_f32(in[i]).to_bits()` into a contiguous `u8`-width output
(`cast_kernel_to_fp8!`, `byte_kernels.rs:3535-3540`). `float8::F8E4M3` does not implement
`bytemuck::Pod`, so it is handled as raw bytes via `to_bits`. F8E4M3 is 1 byte/elem, so the output
byte size equals the element count; the kernel validates `input.len_bytes() % 4 == 0` and
`out.len_bytes() == elem_count`. Rounding follows the `float8` crate's NV/OCP E4M3
(4-bit exponent, 3-bit mantissa, no inf, saturating max-finite) conversion. Bandwidth-bound
elementwise op. Deterministic on the same hardware. Contiguous-only.

```fkc
kernel: cast_f32_to_f8e4m3
op_kind: Cast
blurb: "Cast f32 -> F8E4M3 (NV/OCP E4M3); contiguous; lossy narrowing."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f32_to_f8e4m3"
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
      dtype_rule: fixed(F8E4M3)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (4 + 1)"          # read N*4 (f32) + write N*1 (f8e4m3)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy narrowing to NV/OCP FP8 E4M3 (4-bit exp, 3-bit mantissa) via float8::F8E4M3::from_f32; deterministic per element."

determinism: same_hardware_bitwise
```

## cast_f8e4m3_to_f32  (F8E4M3 → f32)

Convert `F8E4M3` → `f32`. **Lossless widening** — FP8 is a strict subset of f32.

Reads a contiguous `u8`-width `F8E4M3` buffer (1 byte/elem) and writes
`out[i] = F8E4M3::from_bits(in[i]).to_f32()` into a contiguous `f32` output
(`cast_kernel_from_fp8!`, `byte_kernels.rs:3541-3546`). Output is 4× the input byte size. Because
F8E4M3 has fewer exponent and mantissa bits than f32, every F8E4M3 value is exactly representable
in f32, so the conversion is exact. The kernel takes element count = `input.len_bytes()` and
validates `out.len_bytes() == elem_count * 4`. Bandwidth-bound elementwise op. Bit-stable on the
same hardware. Contiguous-only.

```fkc
kernel: cast_f8e4m3_to_f32
op_kind: Cast
blurb: "Cast F8E4M3 -> f32; contiguous; lossless widening."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      fdx:
        sub_byte: F8E4M3              # 1-byte opaque fp8; bit-width/packing owned by FDX (§3.4)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (1 + 4)"          # read N*1 (f8e4m3) + write N*4 (f32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact: F8E4M3 strict subset of f32
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossless widening; every F8E4M3 value exactly representable in f32; exact, deterministic."

determinism: same_hardware_bitwise
```

## cast_bf16_to_f8e4m3  (bf16 → F8E4M3, via f32)

Convert `bf16` → `F8E4M3`. **Lossy narrowing on the f8 leg only** — the bf16→f32 pivot is lossless.

Reads a contiguous `bf16` buffer and writes the 1-byte `F8E4M3` pattern
`out[i] = F8E4M3::from_f32(in[i].to_f32()).to_bits()` (`cast_kernel_to_fp8!`,
`byte_kernels.rs:3547-3552`). The two-leg pivot first widens bf16→f32 (exact, bf16 ⊂ f32) then
narrows f32→F8E4M3 (lossy per NV/OCP E4M3); only the second leg loses precision. F8E4M3 is 1
byte/elem, so output byte size = element count; the kernel validates `input.len_bytes() % 2 == 0`
and `out.len_bytes() == elem_count`. Bandwidth-bound elementwise op. Deterministic on the same
hardware. Contiguous-only.

```fkc
kernel: cast_bf16_to_f8e4m3
op_kind: Cast
blurb: "Cast bf16 -> F8E4M3 via f32; contiguous; lossy on the f8 leg only."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_bf16_to_f8e4m3"
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
      dtype_rule: fixed(F8E4M3)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (2 + 1)"          # read N*2 (bf16) + write N*1 (f8e4m3)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Two-leg pivot bf16->f32 (lossless, bf16 subset of f32) then f32->F8E4M3 (lossy NV/OCP E4M3); precision lost only on the f8 leg; deterministic per element."

determinism: same_hardware_bitwise
```

## cast_f8e4m3_to_bf16  (F8E4M3 → bf16, via f32)

Convert `F8E4M3` → `bf16`. **Lossless within F8E4M3's range** — both legs are exact for this direction.

Reads a contiguous `u8`-width `F8E4M3` buffer (1 byte/elem) and writes
`out[i] = bf16::from_f32(F8E4M3::from_bits(in[i]).to_f32())` (`cast_kernel_from_fp8!`,
`byte_kernels.rs:3553-3558`). The pivot widens F8E4M3→f32 (exact) then narrows f32→bf16; because
F8E4M3's mantissa (3 bits) is narrower than bf16's (7 bits) and its exponent fits bf16's, every
F8E4M3 value lands exactly on a representable bf16 value — so the round-trip through f32 is lossless
for this direction. The kernel takes element count = `input.len_bytes()` and validates
`out.len_bytes() == elem_count * 2`. Bandwidth-bound elementwise op. Bit-stable on the same
hardware. Contiguous-only.

```fkc
kernel: cast_f8e4m3_to_bf16
op_kind: Cast
blurb: "Cast F8E4M3 -> bf16 via f32; contiguous; lossless within F8E4M3's range."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      fdx:
        sub_byte: F8E4M3
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (1 + 2)"          # read N*1 (f8e4m3) + write N*2 (bf16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact within F8E4M3's range (F8E4M3 lands on representable bf16)
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Two-leg pivot F8E4M3->f32->bf16; both legs exact (F8E4M3 mantissa 3 bits < bf16 7 bits, exponent fits); lossless within F8E4M3's range; deterministic."

determinism: same_hardware_bitwise
```

## cast_f16_to_f8e4m3  (f16 → F8E4M3, via f32)

Convert `f16` → `F8E4M3`. **Lossy narrowing on the f8 leg only** — the f16→f32 pivot is lossless.

Reads a contiguous `f16` buffer and writes the 1-byte `F8E4M3` pattern
`out[i] = F8E4M3::from_f32(in[i].to_f32()).to_bits()` (`cast_kernel_to_fp8!`,
`byte_kernels.rs:3559-3564`). The pivot widens f16→f32 (exact, f16 ⊂ f32) then narrows
f32→F8E4M3 (lossy per NV/OCP E4M3); only the second leg loses precision. F8E4M3 is 1 byte/elem, so
output byte size = element count; the kernel validates `input.len_bytes() % 2 == 0` and
`out.len_bytes() == elem_count`. Bandwidth-bound elementwise op. Deterministic on the same hardware.
Contiguous-only.

```fkc
kernel: cast_f16_to_f8e4m3
op_kind: Cast
blurb: "Cast f16 -> F8E4M3 via f32; contiguous; lossy on the f8 leg only."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f16_to_f8e4m3"
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
      dtype_rule: fixed(F8E4M3)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (2 + 1)"          # read N*2 (f16) + write N*1 (f8e4m3)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Two-leg pivot f16->f32 (lossless, f16 subset of f32) then f32->F8E4M3 (lossy NV/OCP E4M3); precision lost only on the f8 leg; deterministic per element."

determinism: same_hardware_bitwise
```

## cast_f8e4m3_to_f16  (F8E4M3 → f16, via f32)

Convert `F8E4M3` → `f16`. **Lossless within F8E4M3's range** — both legs are exact for this direction.

Reads a contiguous `u8`-width `F8E4M3` buffer (1 byte/elem) and writes
`out[i] = f16::from_f32(F8E4M3::from_bits(in[i]).to_f32())` (`cast_kernel_from_fp8!`,
`byte_kernels.rs:3565-3570`). The pivot widens F8E4M3→f32 (exact) then narrows f32→f16; because
F8E4M3's mantissa (3 bits) is narrower than f16's (10 bits) and its exponent range fits within
f16's, every F8E4M3 value lands exactly on a representable f16 value — so the round-trip through f32
is lossless for this direction. The kernel takes element count = `input.len_bytes()` and validates
`out.len_bytes() == elem_count * 2`. Bandwidth-bound elementwise op. Bit-stable on the same
hardware. Contiguous-only.

```fkc
kernel: cast_f8e4m3_to_f16
op_kind: Cast
blurb: "Cast F8E4M3 -> f16 via f32; contiguous; lossless within F8E4M3's range."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      fdx:
        sub_byte: F8E4M3
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "n * (1 + 2)"          # read N*1 (f8e4m3) + write N*2 (f16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact within F8E4M3's range (F8E4M3 lands on representable f16)
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Two-leg pivot F8E4M3->f32->f16; both legs exact (F8E4M3 mantissa 3 bits < f16 10 bits, exponent fits); lossless within F8E4M3's range; deterministic."

determinism: same_hardware_bitwise
```

## cast_u8_to_f32  (u8 → f32)

Convert `u8` → `f32`. **Exact** — every U8 value lands on a representable F32 value.

Walks a contiguous, zero-offset, row-major `u8` buffer and writes `in[i] as f32` into a contiguous `f32` output, element count preserved (output widens the input: 4 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u8_to_f32
op_kind: Cast
blurb: "Cast u8 -> f32; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u8_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 4)"          # read N*1 (u8) + write N*4 (f32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every U8 value is representable in F32; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i8_to_f32  (i8 → f32)

Convert `i8` → `f32`. **Exact** — every I8 value lands on a representable F32 value.

Walks a contiguous, zero-offset, row-major `i8` buffer and writes `in[i] as f32` into a contiguous `f32` output, element count preserved (output widens the input: 4 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i8_to_f32
op_kind: Cast
blurb: "Cast i8 -> f32; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i8_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 4)"          # read N*1 (i8) + write N*4 (f32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every I8 value is representable in F32; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_u32_to_f32  (u32 → f32)

Convert `u32` → `f32`. **Lossy** — large-magnitude U32 values round to the nearest F32 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `u32` buffer and writes `in[i] as f32` into a contiguous `f32` output, element count preserved (output same-width the input: 4 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u32_to_f32
op_kind: Cast
blurb: "Cast u32 -> f32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u32_to_f32"
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
      dtype_rule: fixed(F32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 4)"          # read N*4 (u32) + write N*4 (f32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large U32 magnitudes round to nearest F32; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_i16_to_f32  (i16 → f32)

Convert `i16` → `f32`. **Exact** — every I16 value lands on a representable F32 value.

Walks a contiguous, zero-offset, row-major `i16` buffer and writes `in[i] as f32` into a contiguous `f32` output, element count preserved (output widens the input: 4 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i16_to_f32
op_kind: Cast
blurb: "Cast i16 -> f32; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i16_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 4)"          # read N*2 (i16) + write N*4 (f32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every I16 value is representable in F32; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i32_to_f32  (i32 → f32)

Convert `i32` → `f32`. **Lossy** — large-magnitude I32 values round to the nearest F32 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `i32` buffer and writes `in[i] as f32` into a contiguous `f32` output, element count preserved (output same-width the input: 4 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i32_to_f32
op_kind: Cast
blurb: "Cast i32 -> f32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i32_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 4)"          # read N*4 (i32) + write N*4 (f32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large I32 magnitudes round to nearest F32; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_i64_to_f32  (i64 → f32)

Convert `i64` → `f32`. **Lossy** — large-magnitude I64 values round to the nearest F32 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `i64` buffer and writes `in[i] as f32` into a contiguous `f32` output, element count preserved (output narrows the input: 4 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i64_to_f32
op_kind: Cast
blurb: "Cast i64 -> f32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i64_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 4)"          # read N*8 (i64) + write N*4 (f32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large I64 magnitudes round to nearest F32; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_f16_to_f64  (f16 → f64)

Convert `f16` → `f64`. **Lossless widening** — F16 is a strict value-subset of F64.

Walks a contiguous, zero-offset, row-major `f16` buffer and writes `in[i] as f64` into a contiguous `f64` output, element count preserved (output widens the input: 8 vs 2 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f16_to_f64
op_kind: Cast
blurb: "Cast f16 -> f64; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f16_to_f64"
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 8)"          # read N*2 (f16) + write N*8 (f64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every F16 value is representable in F64; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_bf16_to_f64  (bf16 → f64)

Convert `bf16` → `f64`. **Lossless widening** — BF16 is a strict value-subset of F64.

Walks a contiguous, zero-offset, row-major `bf16` buffer and writes `in[i] as f64` into a contiguous `f64` output, element count preserved (output widens the input: 8 vs 2 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_bf16_to_f64
op_kind: Cast
blurb: "Cast bf16 -> f64; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_bf16_to_f64"
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 8)"          # read N*2 (bf16) + write N*8 (f64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every BF16 value is representable in F64; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_f8e4m3_to_f64  (f8e4m3 → f64)

Convert `f8e4m3` → `f64`. **Lossless widening** — F8E4M3 is a strict value-subset of F64.

Walks a contiguous, zero-offset, row-major `f8e4m3` buffer and writes `F8E4M3::from_bits(in[i]).to_f32() as f64` into a contiguous `f64` output, element count preserved (output widens the input: 8 vs 1 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f8e4m3_to_f64
op_kind: Cast
blurb: "Cast f8e4m3 -> f64; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      fdx:
        sub_byte: F8E4M3              # 1-byte opaque fp8; bit-width/packing owned by FDX (§3.4)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 8)"          # read N*1 (f8e4m3) + write N*8 (f64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every F8E4M3 value is representable in F64; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_u8_to_f64  (u8 → f64)

Convert `u8` → `f64`. **Exact** — every U8 value lands on a representable F64 value.

Walks a contiguous, zero-offset, row-major `u8` buffer and writes `in[i] as f64` into a contiguous `f64` output, element count preserved (output widens the input: 8 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u8_to_f64
op_kind: Cast
blurb: "Cast u8 -> f64; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u8_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 8)"          # read N*1 (u8) + write N*8 (f64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every U8 value is representable in F64; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i8_to_f64  (i8 → f64)

Convert `i8` → `f64`. **Exact** — every I8 value lands on a representable F64 value.

Walks a contiguous, zero-offset, row-major `i8` buffer and writes `in[i] as f64` into a contiguous `f64` output, element count preserved (output widens the input: 8 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i8_to_f64
op_kind: Cast
blurb: "Cast i8 -> f64; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i8_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I8]
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 8)"          # read N*1 (i8) + write N*8 (f64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every I8 value is representable in F64; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_u32_to_f64  (u32 → f64)

Convert `u32` → `f64`. **Exact** — every U32 value lands on a representable F64 value.

Walks a contiguous, zero-offset, row-major `u32` buffer and writes `in[i] as f64` into a contiguous `f64` output, element count preserved (output widens the input: 8 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u32_to_f64
op_kind: Cast
blurb: "Cast u32 -> f64; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u32_to_f64"
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 8)"          # read N*4 (u32) + write N*8 (f64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every U32 value is representable in F64; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i16_to_f64  (i16 → f64)

Convert `i16` → `f64`. **Exact** — every I16 value lands on a representable F64 value.

Walks a contiguous, zero-offset, row-major `i16` buffer and writes `in[i] as f64` into a contiguous `f64` output, element count preserved (output widens the input: 8 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i16_to_f64
op_kind: Cast
blurb: "Cast i16 -> f64; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i16_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I16]
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 8)"          # read N*2 (i16) + write N*8 (f64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every I16 value is representable in F64; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i32_to_f64  (i32 → f64)

Convert `i32` → `f64`. **Exact** — every I32 value lands on a representable F64 value.

Walks a contiguous, zero-offset, row-major `i32` buffer and writes `in[i] as f64` into a contiguous `f64` output, element count preserved (output widens the input: 8 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i32_to_f64
op_kind: Cast
blurb: "Cast i32 -> f64; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i32_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I32]
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 8)"          # read N*4 (i32) + write N*8 (f64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every I32 value is representable in F64; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i64_to_f64  (i64 → f64)

Convert `i64` → `f64`. **Lossy** — large-magnitude I64 values round to the nearest F64 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `i64` buffer and writes `in[i] as f64` into a contiguous `f64` output, element count preserved (output same-width the input: 8 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i64_to_f64
op_kind: Cast
blurb: "Cast i64 -> f64; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i64_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I64]
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 8)"          # read N*8 (i64) + write N*8 (f64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large I64 magnitudes round to nearest F64; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_f64_to_f16  (f64 → f16)

Convert `f64` → `f16`. **Lossy narrowing** per IEEE-754 round-to-nearest-even (out-of-range saturates, NaN preserved).

Walks a contiguous, zero-offset, row-major `f64` buffer and writes `in[i] as f16` into a contiguous `f16` output, element count preserved (output narrows the input: 2 vs 8 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f64_to_f16
op_kind: Cast
blurb: "Cast f64 -> f16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f64_to_f16"
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
      dtype_rule: fixed(F16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 2)"          # read N*8 (f64) + write N*2 (f16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy narrowing (IEEE round-to-nearest-even, out-of-range saturates, NaN preserved); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_bf16_to_f16  (bf16 → f16)

Convert `bf16` → `f16`. **Lossy narrowing** per IEEE-754 round-to-nearest-even (out-of-range saturates, NaN preserved).

Walks a contiguous, zero-offset, row-major `bf16` buffer and writes `in[i] as f16` into a contiguous `f16` output, element count preserved (output same-width the input: 2 vs 2 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_bf16_to_f16
op_kind: Cast
blurb: "Cast bf16 -> f16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_bf16_to_f16"
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
      dtype_rule: fixed(F16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 2)"          # read N*2 (bf16) + write N*2 (f16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy narrowing (IEEE round-to-nearest-even, out-of-range saturates, NaN preserved); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_u8_to_f16  (u8 → f16)

Convert `u8` → `f16`. **Exact** — every U8 value lands on a representable F16 value.

Walks a contiguous, zero-offset, row-major `u8` buffer and writes `in[i] as f16` into a contiguous `f16` output, element count preserved (output widens the input: 2 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u8_to_f16
op_kind: Cast
blurb: "Cast u8 -> f16; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u8_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 2)"          # read N*1 (u8) + write N*2 (f16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every U8 value is representable in F16; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i8_to_f16  (i8 → f16)

Convert `i8` → `f16`. **Exact** — every I8 value lands on a representable F16 value.

Walks a contiguous, zero-offset, row-major `i8` buffer and writes `in[i] as f16` into a contiguous `f16` output, element count preserved (output widens the input: 2 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i8_to_f16
op_kind: Cast
blurb: "Cast i8 -> f16; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i8_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 2)"          # read N*1 (i8) + write N*2 (f16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every I8 value is representable in F16; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_u32_to_f16  (u32 → f16)

Convert `u32` → `f16`. **Lossy** — large-magnitude U32 values round to the nearest F16 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `u32` buffer and writes `in[i] as f16` into a contiguous `f16` output, element count preserved (output narrows the input: 2 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u32_to_f16
op_kind: Cast
blurb: "Cast u32 -> f16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u32_to_f16"
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
      dtype_rule: fixed(F16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 2)"          # read N*4 (u32) + write N*2 (f16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large U32 magnitudes round to nearest F16; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_i16_to_f16  (i16 → f16)

Convert `i16` → `f16`. **Lossy** — large-magnitude I16 values round to the nearest F16 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `i16` buffer and writes `in[i] as f16` into a contiguous `f16` output, element count preserved (output same-width the input: 2 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i16_to_f16
op_kind: Cast
blurb: "Cast i16 -> f16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i16_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 2)"          # read N*2 (i16) + write N*2 (f16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large I16 magnitudes round to nearest F16; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_i32_to_f16  (i32 → f16)

Convert `i32` → `f16`. **Lossy** — large-magnitude I32 values round to the nearest F16 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `i32` buffer and writes `in[i] as f16` into a contiguous `f16` output, element count preserved (output narrows the input: 2 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i32_to_f16
op_kind: Cast
blurb: "Cast i32 -> f16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i32_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 2)"          # read N*4 (i32) + write N*2 (f16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large I32 magnitudes round to nearest F16; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_i64_to_f16  (i64 → f16)

Convert `i64` → `f16`. **Lossy** — large-magnitude I64 values round to the nearest F16 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `i64` buffer and writes `in[i] as f16` into a contiguous `f16` output, element count preserved (output narrows the input: 2 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i64_to_f16
op_kind: Cast
blurb: "Cast i64 -> f16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i64_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 2)"          # read N*8 (i64) + write N*2 (f16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large I64 magnitudes round to nearest F16; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_f64_to_bf16  (f64 → bf16)

Convert `f64` → `bf16`. **Lossy narrowing** per IEEE-754 round-to-nearest-even (out-of-range saturates, NaN preserved).

Walks a contiguous, zero-offset, row-major `f64` buffer and writes `in[i] as bf16` into a contiguous `bf16` output, element count preserved (output narrows the input: 2 vs 8 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f64_to_bf16
op_kind: Cast
blurb: "Cast f64 -> bf16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f64_to_bf16"
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
      dtype_rule: fixed(BF16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 2)"          # read N*8 (f64) + write N*2 (bf16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy narrowing (IEEE round-to-nearest-even, out-of-range saturates, NaN preserved); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_f16_to_bf16  (f16 → bf16)

Convert `f16` → `bf16`. **Lossy narrowing** per IEEE-754 round-to-nearest-even (out-of-range saturates, NaN preserved).

Walks a contiguous, zero-offset, row-major `f16` buffer and writes `in[i] as bf16` into a contiguous `bf16` output, element count preserved (output same-width the input: 2 vs 2 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f16_to_bf16
op_kind: Cast
blurb: "Cast f16 -> bf16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f16_to_bf16"
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
      dtype_rule: fixed(BF16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 2)"          # read N*2 (f16) + write N*2 (bf16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy narrowing (IEEE round-to-nearest-even, out-of-range saturates, NaN preserved); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_u8_to_bf16  (u8 → bf16)

Convert `u8` → `bf16`. **Exact** — every U8 value lands on a representable BF16 value.

Walks a contiguous, zero-offset, row-major `u8` buffer and writes `in[i] as bf16` into a contiguous `bf16` output, element count preserved (output widens the input: 2 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u8_to_bf16
op_kind: Cast
blurb: "Cast u8 -> bf16; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u8_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BF16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 2)"          # read N*1 (u8) + write N*2 (bf16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every U8 value is representable in BF16; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i8_to_bf16  (i8 → bf16)

Convert `i8` → `bf16`. **Exact** — every I8 value lands on a representable BF16 value.

Walks a contiguous, zero-offset, row-major `i8` buffer and writes `in[i] as bf16` into a contiguous `bf16` output, element count preserved (output widens the input: 2 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i8_to_bf16
op_kind: Cast
blurb: "Cast i8 -> bf16; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i8_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BF16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 2)"          # read N*1 (i8) + write N*2 (bf16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every I8 value is representable in BF16; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_u32_to_bf16  (u32 → bf16)

Convert `u32` → `bf16`. **Lossy** — large-magnitude U32 values round to the nearest BF16 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `u32` buffer and writes `in[i] as bf16` into a contiguous `bf16` output, element count preserved (output narrows the input: 2 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u32_to_bf16
op_kind: Cast
blurb: "Cast u32 -> bf16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u32_to_bf16"
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
      dtype_rule: fixed(BF16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 2)"          # read N*4 (u32) + write N*2 (bf16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large U32 magnitudes round to nearest BF16; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_i16_to_bf16  (i16 → bf16)

Convert `i16` → `bf16`. **Lossy** — large-magnitude I16 values round to the nearest BF16 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `i16` buffer and writes `in[i] as bf16` into a contiguous `bf16` output, element count preserved (output same-width the input: 2 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i16_to_bf16
op_kind: Cast
blurb: "Cast i16 -> bf16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i16_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BF16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 2)"          # read N*2 (i16) + write N*2 (bf16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large I16 magnitudes round to nearest BF16; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_i32_to_bf16  (i32 → bf16)

Convert `i32` → `bf16`. **Lossy** — large-magnitude I32 values round to the nearest BF16 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `i32` buffer and writes `in[i] as bf16` into a contiguous `bf16` output, element count preserved (output narrows the input: 2 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i32_to_bf16
op_kind: Cast
blurb: "Cast i32 -> bf16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i32_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BF16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 2)"          # read N*4 (i32) + write N*2 (bf16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large I32 magnitudes round to nearest BF16; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_i64_to_bf16  (i64 → bf16)

Convert `i64` → `bf16`. **Lossy** — large-magnitude I64 values round to the nearest BF16 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `i64` buffer and writes `in[i] as bf16` into a contiguous `bf16` output, element count preserved (output narrows the input: 2 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i64_to_bf16
op_kind: Cast
blurb: "Cast i64 -> bf16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i64_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BF16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 2)"          # read N*8 (i64) + write N*2 (bf16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large I64 magnitudes round to nearest BF16; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_f64_to_f8e4m3  (f64 → f8e4m3)

Convert `f64` → `f8e4m3`. **Lossy narrowing** per IEEE-754 round-to-nearest-even (out-of-range saturates, NaN preserved).

Walks a contiguous, zero-offset, row-major `f64` buffer and writes `in[i] as f8e4m3` into a contiguous `f8e4m3` output, element count preserved (output narrows the input: 1 vs 8 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f64_to_f8e4m3
op_kind: Cast
blurb: "Cast f64 -> f8e4m3; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f64_to_f8e4m3"
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
      dtype_rule: fixed(F8E4M3)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 1)"          # read N*8 (f64) + write N*1 (f8e4m3)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy narrowing (IEEE round-to-nearest-even, out-of-range saturates, NaN preserved); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_u8_to_f8e4m3  (u8 → f8e4m3)

Convert `u8` → `f8e4m3`. **Lossy** — large-magnitude U8 values round to the nearest F8E4M3 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `u8` buffer and writes `in[i] as f8e4m3` into a contiguous `f8e4m3` output, element count preserved (output same-width the input: 1 vs 1 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u8_to_f8e4m3
op_kind: Cast
blurb: "Cast u8 -> f8e4m3; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u8_to_f8e4m3"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F8E4M3)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 1)"          # read N*1 (u8) + write N*1 (f8e4m3)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large U8 magnitudes round to nearest F8E4M3; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_i8_to_f8e4m3  (i8 → f8e4m3)

Convert `i8` → `f8e4m3`. **Lossy** — large-magnitude I8 values round to the nearest F8E4M3 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `i8` buffer and writes `in[i] as f8e4m3` into a contiguous `f8e4m3` output, element count preserved (output same-width the input: 1 vs 1 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i8_to_f8e4m3
op_kind: Cast
blurb: "Cast i8 -> f8e4m3; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i8_to_f8e4m3"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F8E4M3)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 1)"          # read N*1 (i8) + write N*1 (f8e4m3)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large I8 magnitudes round to nearest F8E4M3; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_u32_to_f8e4m3  (u32 → f8e4m3)

Convert `u32` → `f8e4m3`. **Lossy** — large-magnitude U32 values round to the nearest F8E4M3 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `u32` buffer and writes `in[i] as f8e4m3` into a contiguous `f8e4m3` output, element count preserved (output narrows the input: 1 vs 4 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u32_to_f8e4m3
op_kind: Cast
blurb: "Cast u32 -> f8e4m3; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u32_to_f8e4m3"
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
      dtype_rule: fixed(F8E4M3)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 1)"          # read N*4 (u32) + write N*1 (f8e4m3)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large U32 magnitudes round to nearest F8E4M3; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_i16_to_f8e4m3  (i16 → f8e4m3)

Convert `i16` → `f8e4m3`. **Lossy** — large-magnitude I16 values round to the nearest F8E4M3 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `i16` buffer and writes `in[i] as f8e4m3` into a contiguous `f8e4m3` output, element count preserved (output narrows the input: 1 vs 2 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i16_to_f8e4m3
op_kind: Cast
blurb: "Cast i16 -> f8e4m3; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i16_to_f8e4m3"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F8E4M3)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 1)"          # read N*2 (i16) + write N*1 (f8e4m3)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large I16 magnitudes round to nearest F8E4M3; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_i32_to_f8e4m3  (i32 → f8e4m3)

Convert `i32` → `f8e4m3`. **Lossy** — large-magnitude I32 values round to the nearest F8E4M3 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `i32` buffer and writes `in[i] as f8e4m3` into a contiguous `f8e4m3` output, element count preserved (output narrows the input: 1 vs 4 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i32_to_f8e4m3
op_kind: Cast
blurb: "Cast i32 -> f8e4m3; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i32_to_f8e4m3"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F8E4M3)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 1)"          # read N*4 (i32) + write N*1 (f8e4m3)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large I32 magnitudes round to nearest F8E4M3; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_i64_to_f8e4m3  (i64 → f8e4m3)

Convert `i64` → `f8e4m3`. **Lossy** — large-magnitude I64 values round to the nearest F8E4M3 (mantissa too narrow for the full integer range).

Walks a contiguous, zero-offset, row-major `i64` buffer and writes `in[i] as f8e4m3` into a contiguous `f8e4m3` output, element count preserved (output narrows the input: 1 vs 8 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i64_to_f8e4m3
op_kind: Cast
blurb: "Cast i64 -> f8e4m3; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i64_to_f8e4m3"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F8E4M3)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 1)"          # read N*8 (i64) + write N*1 (f8e4m3)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy: large I64 magnitudes round to nearest F8E4M3; deterministic round-to-nearest-even per element."

determinism: same_hardware_bitwise
```

## cast_f32_to_u8  (f32 → u8)

Convert `f32` → `u8`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the U8 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f32` buffer and writes `in[i] as u8` into a contiguous `u8` output, element count preserved (output narrows the input: 1 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f32_to_u8
op_kind: Cast
blurb: "Cast f32 -> u8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f32_to_u8"
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
      dtype_rule: fixed(U8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 1)"          # read N*4 (f32) + write N*1 (u8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f64_to_u8  (f64 → u8)

Convert `f64` → `u8`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the U8 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f64` buffer and writes `in[i] as u8` into a contiguous `u8` output, element count preserved (output narrows the input: 1 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f64_to_u8
op_kind: Cast
blurb: "Cast f64 -> u8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f64_to_u8"
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
      dtype_rule: fixed(U8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 1)"          # read N*8 (f64) + write N*1 (u8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f16_to_u8  (f16 → u8)

Convert `f16` → `u8`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the U8 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f16` buffer and writes `in[i].to_f32() as u8` into a contiguous `u8` output, element count preserved (output narrows the input: 1 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f16_to_u8
op_kind: Cast
blurb: "Cast f16 -> u8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f16_to_u8"
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
      dtype_rule: fixed(U8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 1)"          # read N*2 (f16) + write N*1 (u8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_bf16_to_u8  (bf16 → u8)

Convert `bf16` → `u8`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the U8 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `bf16` buffer and writes `in[i].to_f32() as u8` into a contiguous `u8` output, element count preserved (output narrows the input: 1 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_bf16_to_u8
op_kind: Cast
blurb: "Cast bf16 -> u8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_bf16_to_u8"
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
      dtype_rule: fixed(U8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 1)"          # read N*2 (bf16) + write N*1 (u8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f8e4m3_to_u8  (f8e4m3 → u8)

Convert `f8e4m3` → `u8`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the U8 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f8e4m3` buffer and writes `F8E4M3::from_bits(in[i]).to_f32() as u8` into a contiguous `u8` output, element count preserved (output same-width the input: 1 vs 1 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f8e4m3_to_u8
op_kind: Cast
blurb: "Cast f8e4m3 -> u8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      fdx:
        sub_byte: F8E4M3              # 1-byte opaque fp8; bit-width/packing owned by FDX (§3.4)
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 1)"          # read N*1 (f8e4m3) + write N*1 (u8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_i8_to_u8  (i8 → u8)

Convert `i8` → `u8`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i8` buffer and writes `in[i] as u8` into a contiguous `u8` output, element count preserved (output same-width the input: 1 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i8_to_u8
op_kind: Cast
blurb: "Cast i8 -> u8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i8_to_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 1)"          # read N*1 (i8) + write N*1 (u8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_u32_to_u8  (u32 → u8)

Convert `u32` → `u8`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `u32` buffer and writes `in[i] as u8` into a contiguous `u8` output, element count preserved (output narrows the input: 1 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u32_to_u8
op_kind: Cast
blurb: "Cast u32 -> u8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u32_to_u8"
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
      dtype_rule: fixed(U8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 1)"          # read N*4 (u32) + write N*1 (u8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_i16_to_u8  (i16 → u8)

Convert `i16` → `u8`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i16` buffer and writes `in[i] as u8` into a contiguous `u8` output, element count preserved (output narrows the input: 1 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i16_to_u8
op_kind: Cast
blurb: "Cast i16 -> u8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i16_to_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 1)"          # read N*2 (i16) + write N*1 (u8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_i32_to_u8  (i32 → u8)

Convert `i32` → `u8`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i32` buffer and writes `in[i] as u8` into a contiguous `u8` output, element count preserved (output narrows the input: 1 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i32_to_u8
op_kind: Cast
blurb: "Cast i32 -> u8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i32_to_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 1)"          # read N*4 (i32) + write N*1 (u8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_i64_to_u8  (i64 → u8)

Convert `i64` → `u8`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i64` buffer and writes `in[i] as u8` into a contiguous `u8` output, element count preserved (output narrows the input: 1 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i64_to_u8
op_kind: Cast
blurb: "Cast i64 -> u8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i64_to_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 1)"          # read N*8 (i64) + write N*1 (u8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_f32_to_i8  (f32 → i8)

Convert `f32` → `i8`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I8 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f32` buffer and writes `in[i] as i8` into a contiguous `i8` output, element count preserved (output narrows the input: 1 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f32_to_i8
op_kind: Cast
blurb: "Cast f32 -> i8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f32_to_i8"
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
      dtype_rule: fixed(I8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 1)"          # read N*4 (f32) + write N*1 (i8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f64_to_i8  (f64 → i8)

Convert `f64` → `i8`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I8 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f64` buffer and writes `in[i] as i8` into a contiguous `i8` output, element count preserved (output narrows the input: 1 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f64_to_i8
op_kind: Cast
blurb: "Cast f64 -> i8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f64_to_i8"
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
      dtype_rule: fixed(I8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 1)"          # read N*8 (f64) + write N*1 (i8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f16_to_i8  (f16 → i8)

Convert `f16` → `i8`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I8 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f16` buffer and writes `in[i].to_f32() as i8` into a contiguous `i8` output, element count preserved (output narrows the input: 1 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f16_to_i8
op_kind: Cast
blurb: "Cast f16 -> i8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f16_to_i8"
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
      dtype_rule: fixed(I8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 1)"          # read N*2 (f16) + write N*1 (i8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_bf16_to_i8  (bf16 → i8)

Convert `bf16` → `i8`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I8 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `bf16` buffer and writes `in[i].to_f32() as i8` into a contiguous `i8` output, element count preserved (output narrows the input: 1 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_bf16_to_i8
op_kind: Cast
blurb: "Cast bf16 -> i8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_bf16_to_i8"
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
      dtype_rule: fixed(I8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 1)"          # read N*2 (bf16) + write N*1 (i8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f8e4m3_to_i8  (f8e4m3 → i8)

Convert `f8e4m3` → `i8`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I8 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f8e4m3` buffer and writes `F8E4M3::from_bits(in[i]).to_f32() as i8` into a contiguous `i8` output, element count preserved (output same-width the input: 1 vs 1 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f8e4m3_to_i8
op_kind: Cast
blurb: "Cast f8e4m3 -> i8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_i8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      fdx:
        sub_byte: F8E4M3              # 1-byte opaque fp8; bit-width/packing owned by FDX (§3.4)
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 1)"          # read N*1 (f8e4m3) + write N*1 (i8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_u8_to_i8  (u8 → i8)

Convert `u8` → `i8`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `u8` buffer and writes `in[i] as i8` into a contiguous `i8` output, element count preserved (output same-width the input: 1 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u8_to_i8
op_kind: Cast
blurb: "Cast u8 -> i8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u8_to_i8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 1)"          # read N*1 (u8) + write N*1 (i8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_u32_to_i8  (u32 → i8)

Convert `u32` → `i8`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `u32` buffer and writes `in[i] as i8` into a contiguous `i8` output, element count preserved (output narrows the input: 1 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u32_to_i8
op_kind: Cast
blurb: "Cast u32 -> i8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u32_to_i8"
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
      dtype_rule: fixed(I8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 1)"          # read N*4 (u32) + write N*1 (i8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_i16_to_i8  (i16 → i8)

Convert `i16` → `i8`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i16` buffer and writes `in[i] as i8` into a contiguous `i8` output, element count preserved (output narrows the input: 1 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i16_to_i8
op_kind: Cast
blurb: "Cast i16 -> i8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i16_to_i8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 1)"          # read N*2 (i16) + write N*1 (i8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_i32_to_i8  (i32 → i8)

Convert `i32` → `i8`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i32` buffer and writes `in[i] as i8` into a contiguous `i8` output, element count preserved (output narrows the input: 1 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i32_to_i8
op_kind: Cast
blurb: "Cast i32 -> i8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i32_to_i8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 1)"          # read N*4 (i32) + write N*1 (i8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_i64_to_i8  (i64 → i8)

Convert `i64` → `i8`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i64` buffer and writes `in[i] as i8` into a contiguous `i8` output, element count preserved (output narrows the input: 1 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i64_to_i8
op_kind: Cast
blurb: "Cast i64 -> i8; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i64_to_i8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I8)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 1)"          # read N*8 (i64) + write N*1 (i8)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 1", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_f32_to_u32  (f32 → u32)

Convert `f32` → `u32`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the U32 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f32` buffer and writes `in[i] as u32` into a contiguous `u32` output, element count preserved (output same-width the input: 4 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f32_to_u32
op_kind: Cast
blurb: "Cast f32 -> u32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f32_to_u32"
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
      dtype_rule: fixed(U32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 4)"          # read N*4 (f32) + write N*4 (u32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f64_to_u32  (f64 → u32)

Convert `f64` → `u32`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the U32 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f64` buffer and writes `in[i] as u32` into a contiguous `u32` output, element count preserved (output narrows the input: 4 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f64_to_u32
op_kind: Cast
blurb: "Cast f64 -> u32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f64_to_u32"
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
      dtype_rule: fixed(U32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 4)"          # read N*8 (f64) + write N*4 (u32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f16_to_u32  (f16 → u32)

Convert `f16` → `u32`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the U32 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f16` buffer and writes `in[i].to_f32() as u32` into a contiguous `u32` output, element count preserved (output widens the input: 4 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f16_to_u32
op_kind: Cast
blurb: "Cast f16 -> u32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f16_to_u32"
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
      dtype_rule: fixed(U32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 4)"          # read N*2 (f16) + write N*4 (u32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_bf16_to_u32  (bf16 → u32)

Convert `bf16` → `u32`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the U32 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `bf16` buffer and writes `in[i].to_f32() as u32` into a contiguous `u32` output, element count preserved (output widens the input: 4 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_bf16_to_u32
op_kind: Cast
blurb: "Cast bf16 -> u32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_bf16_to_u32"
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
      dtype_rule: fixed(U32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 4)"          # read N*2 (bf16) + write N*4 (u32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f8e4m3_to_u32  (f8e4m3 → u32)

Convert `f8e4m3` → `u32`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the U32 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f8e4m3` buffer and writes `F8E4M3::from_bits(in[i]).to_f32() as u32` into a contiguous `u32` output, element count preserved (output widens the input: 4 vs 1 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f8e4m3_to_u32
op_kind: Cast
blurb: "Cast f8e4m3 -> u32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_u32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      fdx:
        sub_byte: F8E4M3              # 1-byte opaque fp8; bit-width/packing owned by FDX (§3.4)
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 4)"          # read N*1 (f8e4m3) + write N*4 (u32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_u8_to_u32  (u8 → u32)

Convert `u8` → `u32`. **Exact** — U8 is a value-subset of U32.

Walks a contiguous, zero-offset, row-major `u8` buffer and writes `in[i] as u32` into a contiguous `u32` output, element count preserved (output widens the input: 4 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u8_to_u32
op_kind: Cast
blurb: "Cast u8 -> u32; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u8_to_u32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 4)"          # read N*1 (u8) + write N*4 (u32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every U8 value is representable in U32; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i8_to_u32  (i8 → u32)

Convert `i8` → `u32`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i8` buffer and writes `in[i] as u32` into a contiguous `u32` output, element count preserved (output widens the input: 4 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i8_to_u32
op_kind: Cast
blurb: "Cast i8 -> u32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i8_to_u32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 4)"          # read N*1 (i8) + write N*4 (u32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_i16_to_u32  (i16 → u32)

Convert `i16` → `u32`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i16` buffer and writes `in[i] as u32` into a contiguous `u32` output, element count preserved (output widens the input: 4 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i16_to_u32
op_kind: Cast
blurb: "Cast i16 -> u32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i16_to_u32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 4)"          # read N*2 (i16) + write N*4 (u32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_i32_to_u32  (i32 → u32)

Convert `i32` → `u32`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i32` buffer and writes `in[i] as u32` into a contiguous `u32` output, element count preserved (output same-width the input: 4 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i32_to_u32
op_kind: Cast
blurb: "Cast i32 -> u32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i32_to_u32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 4)"          # read N*4 (i32) + write N*4 (u32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_i64_to_u32  (i64 → u32)

Convert `i64` → `u32`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i64` buffer and writes `in[i] as u32` into a contiguous `u32` output, element count preserved (output narrows the input: 4 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i64_to_u32
op_kind: Cast
blurb: "Cast i64 -> u32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i64_to_u32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 4)"          # read N*8 (i64) + write N*4 (u32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_f32_to_i16  (f32 → i16)

Convert `f32` → `i16`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I16 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f32` buffer and writes `in[i] as i16` into a contiguous `i16` output, element count preserved (output narrows the input: 2 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f32_to_i16
op_kind: Cast
blurb: "Cast f32 -> i16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f32_to_i16"
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
      dtype_rule: fixed(I16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 2)"          # read N*4 (f32) + write N*2 (i16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f64_to_i16  (f64 → i16)

Convert `f64` → `i16`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I16 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f64` buffer and writes `in[i] as i16` into a contiguous `i16` output, element count preserved (output narrows the input: 2 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f64_to_i16
op_kind: Cast
blurb: "Cast f64 -> i16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f64_to_i16"
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
      dtype_rule: fixed(I16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 2)"          # read N*8 (f64) + write N*2 (i16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f16_to_i16  (f16 → i16)

Convert `f16` → `i16`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I16 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f16` buffer and writes `in[i].to_f32() as i16` into a contiguous `i16` output, element count preserved (output same-width the input: 2 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f16_to_i16
op_kind: Cast
blurb: "Cast f16 -> i16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f16_to_i16"
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
      dtype_rule: fixed(I16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 2)"          # read N*2 (f16) + write N*2 (i16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_bf16_to_i16  (bf16 → i16)

Convert `bf16` → `i16`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I16 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `bf16` buffer and writes `in[i].to_f32() as i16` into a contiguous `i16` output, element count preserved (output same-width the input: 2 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_bf16_to_i16
op_kind: Cast
blurb: "Cast bf16 -> i16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_bf16_to_i16"
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
      dtype_rule: fixed(I16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 2)"          # read N*2 (bf16) + write N*2 (i16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f8e4m3_to_i16  (f8e4m3 → i16)

Convert `f8e4m3` → `i16`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I16 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f8e4m3` buffer and writes `F8E4M3::from_bits(in[i]).to_f32() as i16` into a contiguous `i16` output, element count preserved (output widens the input: 2 vs 1 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f8e4m3_to_i16
op_kind: Cast
blurb: "Cast f8e4m3 -> i16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_i16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      fdx:
        sub_byte: F8E4M3              # 1-byte opaque fp8; bit-width/packing owned by FDX (§3.4)
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 2)"          # read N*1 (f8e4m3) + write N*2 (i16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_u8_to_i16  (u8 → i16)

Convert `u8` → `i16`. **Exact** — U8 is a value-subset of I16.

Walks a contiguous, zero-offset, row-major `u8` buffer and writes `in[i] as i16` into a contiguous `i16` output, element count preserved (output widens the input: 2 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u8_to_i16
op_kind: Cast
blurb: "Cast u8 -> i16; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u8_to_i16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 2)"          # read N*1 (u8) + write N*2 (i16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every U8 value is representable in I16; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i8_to_i16  (i8 → i16)

Convert `i8` → `i16`. **Exact** — I8 is a value-subset of I16.

Walks a contiguous, zero-offset, row-major `i8` buffer and writes `in[i] as i16` into a contiguous `i16` output, element count preserved (output widens the input: 2 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i8_to_i16
op_kind: Cast
blurb: "Cast i8 -> i16; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i8_to_i16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 2)"          # read N*1 (i8) + write N*2 (i16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every I8 value is representable in I16; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_u32_to_i16  (u32 → i16)

Convert `u32` → `i16`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `u32` buffer and writes `in[i] as i16` into a contiguous `i16` output, element count preserved (output narrows the input: 2 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u32_to_i16
op_kind: Cast
blurb: "Cast u32 -> i16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u32_to_i16"
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
      dtype_rule: fixed(I16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 2)"          # read N*4 (u32) + write N*2 (i16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_i32_to_i16  (i32 → i16)

Convert `i32` → `i16`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i32` buffer and writes `in[i] as i16` into a contiguous `i16` output, element count preserved (output narrows the input: 2 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i32_to_i16
op_kind: Cast
blurb: "Cast i32 -> i16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i32_to_i16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 2)"          # read N*4 (i32) + write N*2 (i16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_i64_to_i16  (i64 → i16)

Convert `i64` → `i16`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i64` buffer and writes `in[i] as i16` into a contiguous `i16` output, element count preserved (output narrows the input: 2 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i64_to_i16
op_kind: Cast
blurb: "Cast i64 -> i16; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i64_to_i16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I16)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 2)"          # read N*8 (i64) + write N*2 (i16)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 2", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_f32_to_i32  (f32 → i32)

Convert `f32` → `i32`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I32 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f32` buffer and writes `in[i] as i32` into a contiguous `i32` output, element count preserved (output same-width the input: 4 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f32_to_i32
op_kind: Cast
blurb: "Cast f32 -> i32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f32_to_i32"
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
      dtype_rule: fixed(I32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 4)"          # read N*4 (f32) + write N*4 (i32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f64_to_i32  (f64 → i32)

Convert `f64` → `i32`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I32 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f64` buffer and writes `in[i] as i32` into a contiguous `i32` output, element count preserved (output narrows the input: 4 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f64_to_i32
op_kind: Cast
blurb: "Cast f64 -> i32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f64_to_i32"
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
      dtype_rule: fixed(I32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 4)"          # read N*8 (f64) + write N*4 (i32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f16_to_i32  (f16 → i32)

Convert `f16` → `i32`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I32 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f16` buffer and writes `in[i].to_f32() as i32` into a contiguous `i32` output, element count preserved (output widens the input: 4 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f16_to_i32
op_kind: Cast
blurb: "Cast f16 -> i32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f16_to_i32"
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
      dtype_rule: fixed(I32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 4)"          # read N*2 (f16) + write N*4 (i32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_bf16_to_i32  (bf16 → i32)

Convert `bf16` → `i32`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I32 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `bf16` buffer and writes `in[i].to_f32() as i32` into a contiguous `i32` output, element count preserved (output widens the input: 4 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_bf16_to_i32
op_kind: Cast
blurb: "Cast bf16 -> i32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_bf16_to_i32"
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
      dtype_rule: fixed(I32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 4)"          # read N*2 (bf16) + write N*4 (i32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f8e4m3_to_i32  (f8e4m3 → i32)

Convert `f8e4m3` → `i32`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I32 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f8e4m3` buffer and writes `F8E4M3::from_bits(in[i]).to_f32() as i32` into a contiguous `i32` output, element count preserved (output widens the input: 4 vs 1 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f8e4m3_to_i32
op_kind: Cast
blurb: "Cast f8e4m3 -> i32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_i32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      fdx:
        sub_byte: F8E4M3              # 1-byte opaque fp8; bit-width/packing owned by FDX (§3.4)
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 4)"          # read N*1 (f8e4m3) + write N*4 (i32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_u8_to_i32  (u8 → i32)

Convert `u8` → `i32`. **Exact** — U8 is a value-subset of I32.

Walks a contiguous, zero-offset, row-major `u8` buffer and writes `in[i] as i32` into a contiguous `i32` output, element count preserved (output widens the input: 4 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u8_to_i32
op_kind: Cast
blurb: "Cast u8 -> i32; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u8_to_i32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 4)"          # read N*1 (u8) + write N*4 (i32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every U8 value is representable in I32; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i8_to_i32  (i8 → i32)

Convert `i8` → `i32`. **Exact** — I8 is a value-subset of I32.

Walks a contiguous, zero-offset, row-major `i8` buffer and writes `in[i] as i32` into a contiguous `i32` output, element count preserved (output widens the input: 4 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i8_to_i32
op_kind: Cast
blurb: "Cast i8 -> i32; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i8_to_i32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 4)"          # read N*1 (i8) + write N*4 (i32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every I8 value is representable in I32; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_u32_to_i32  (u32 → i32)

Convert `u32` → `i32`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `u32` buffer and writes `in[i] as i32` into a contiguous `i32` output, element count preserved (output same-width the input: 4 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u32_to_i32
op_kind: Cast
blurb: "Cast u32 -> i32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u32_to_i32"
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
      dtype_rule: fixed(I32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 4)"          # read N*4 (u32) + write N*4 (i32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_i16_to_i32  (i16 → i32)

Convert `i16` → `i32`. **Exact** — I16 is a value-subset of I32.

Walks a contiguous, zero-offset, row-major `i16` buffer and writes `in[i] as i32` into a contiguous `i32` output, element count preserved (output widens the input: 4 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i16_to_i32
op_kind: Cast
blurb: "Cast i16 -> i32; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i16_to_i32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 4)"          # read N*2 (i16) + write N*4 (i32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every I16 value is representable in I32; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i64_to_i32  (i64 → i32)

Convert `i64` → `i32`. **Lossy** integer narrowing — out-of-range values wrap (two's-complement truncation, Rust `as`).

Walks a contiguous, zero-offset, row-major `i64` buffer and writes `in[i] as i32` into a contiguous `i32` output, element count preserved (output narrows the input: 4 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i64_to_i32
op_kind: Cast
blurb: "Cast i64 -> i32; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i64_to_i32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I32)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 4)"          # read N*8 (i64) + write N*4 (i32)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy integer narrowing (two's-complement wrap on overflow); deterministic per element."

determinism: same_hardware_bitwise
```

## cast_f32_to_i64  (f32 → i64)

Convert `f32` → `i64`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I64 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f32` buffer and writes `in[i] as i64` into a contiguous `i64` output, element count preserved (output widens the input: 8 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f32_to_i64
op_kind: Cast
blurb: "Cast f32 -> i64; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f32_to_i64"
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
      dtype_rule: fixed(I64)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 8)"          # read N*4 (f32) + write N*8 (i64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f64_to_i64  (f64 → i64)

Convert `f64` → `i64`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I64 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f64` buffer and writes `in[i] as i64` into a contiguous `i64` output, element count preserved (output same-width the input: 8 vs 8 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f64_to_i64
op_kind: Cast
blurb: "Cast f64 -> i64; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f64_to_i64"
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
      dtype_rule: fixed(I64)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (8 + 8)"          # read N*8 (f64) + write N*8 (i64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f16_to_i64  (f16 → i64)

Convert `f16` → `i64`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I64 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f16` buffer and writes `in[i].to_f32() as i64` into a contiguous `i64` output, element count preserved (output widens the input: 8 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f16_to_i64
op_kind: Cast
blurb: "Cast f16 -> i64; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f16_to_i64"
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
      dtype_rule: fixed(I64)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 8)"          # read N*2 (f16) + write N*8 (i64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_bf16_to_i64  (bf16 → i64)

Convert `bf16` → `i64`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I64 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `bf16` buffer and writes `in[i].to_f32() as i64` into a contiguous `i64` output, element count preserved (output widens the input: 8 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_bf16_to_i64
op_kind: Cast
blurb: "Cast bf16 -> i64; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_bf16_to_i64"
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
      dtype_rule: fixed(I64)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 8)"          # read N*2 (bf16) + write N*8 (i64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_f8e4m3_to_i64  (f8e4m3 → i64)

Convert `f8e4m3` → `i64`. **Lossy** float→integer conversion: truncates toward zero and saturates out-of-range magnitudes to the I64 bounds (Rust `as` saturating cast).

Walks a contiguous, zero-offset, row-major `f8e4m3` buffer and writes `F8E4M3::from_bits(in[i]).to_f32() as i64` into a contiguous `i64` output, element count preserved (output widens the input: 8 vs 1 bytes/elem). The conversion pivots through f32 (the only widening/narrowing bridge these formats expose). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_f8e4m3_to_i64
op_kind: Cast
blurb: "Cast f8e4m3 -> i64; contiguous; lossy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_f8e4m3_to_i64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      fdx:
        sub_byte: F8E4M3              # 1-byte opaque fp8; bit-width/packing owned by FDX (§3.4)
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I64)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 8)"          # read N*1 (f8e4m3) + write N*8 (i64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Lossy float->integer (truncate toward zero, saturating); deterministic single conversion per element."

determinism: same_hardware_bitwise
```

## cast_u8_to_i64  (u8 → i64)

Convert `u8` → `i64`. **Exact** — U8 is a value-subset of I64.

Walks a contiguous, zero-offset, row-major `u8` buffer and writes `in[i] as i64` into a contiguous `i64` output, element count preserved (output widens the input: 8 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u8_to_i64
op_kind: Cast
blurb: "Cast u8 -> i64; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u8_to_i64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I64)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 8)"          # read N*1 (u8) + write N*8 (i64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every U8 value is representable in I64; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i8_to_i64  (i8 → i64)

Convert `i8` → `i64`. **Exact** — I8 is a value-subset of I64.

Walks a contiguous, zero-offset, row-major `i8` buffer and writes `in[i] as i64` into a contiguous `i64` output, element count preserved (output widens the input: 8 vs 1 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i8_to_i64
op_kind: Cast
blurb: "Cast i8 -> i64; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i8_to_i64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I64)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (1 + 8)"          # read N*1 (i8) + write N*8 (i64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every I8 value is representable in I64; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_u32_to_i64  (u32 → i64)

Convert `u32` → `i64`. **Exact** — U32 is a value-subset of I64.

Walks a contiguous, zero-offset, row-major `u32` buffer and writes `in[i] as i64` into a contiguous `i64` output, element count preserved (output widens the input: 8 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_u32_to_i64
op_kind: Cast
blurb: "Cast u32 -> i64; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_u32_to_i64"
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
      dtype_rule: fixed(I64)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 8)"          # read N*4 (u32) + write N*8 (i64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every U32 value is representable in I64; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i16_to_i64  (i16 → i64)

Convert `i16` → `i64`. **Exact** — I16 is a value-subset of I64.

Walks a contiguous, zero-offset, row-major `i16` buffer and writes `in[i] as i64` into a contiguous `i64` output, element count preserved (output widens the input: 8 vs 2 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i16_to_i64
op_kind: Cast
blurb: "Cast i16 -> i64; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i16_to_i64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I64)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (2 + 8)"          # read N*2 (i16) + write N*8 (i64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every I16 value is representable in I64; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

## cast_i32_to_i64  (i32 → i64)

Convert `i32` → `i64`. **Exact** — I32 is a value-subset of I64.

Walks a contiguous, zero-offset, row-major `i32` buffer and writes `in[i] as i64` into a contiguous `i64` output, element count preserved (output widens the input: 8 vs 4 bytes/elem). Validates byte lengths only, returns a typed `Result` on a size mismatch (never panics). Bandwidth-bound elementwise op; deterministic on the same hardware; contiguous-only (any strided/broadcast/offset operand is contiguized first).

```fkc
kernel: cast_i32_to_i64
op_kind: Cast
blurb: "Cast i32 -> i64; contiguous; lossless."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cast_i32_to_i64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [I32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I64)          # target dtype lives on the output Storage; key-pinned (§5.1)
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
  provenance: declared                # author prior (overhead_ns launch cost); Judge refines (§4.4)
  class: cheap_elementwise
  flops: "0"                          # pure copy/convert; no arithmetic
  bytes_moved: "n * (4 + 8)"          # read N*4 (i32) + write N*8 (i64)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "n * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact/deterministic conversion; every I32 value is representable in I64; bit-stable on the same hardware."

determinism: same_hardware_bitwise
```


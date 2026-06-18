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

Two implementation families back these twelve kernels:

- **`cast_kernel!`** (`byte_kernels.rs:3393-3469`) — `bytemuck::Pod`-to-`Pod` element conversion
  (`f32↔f64`, `f32↔bf16`, `f32↔f16`). Validates `input.len_bytes() % in_elem_size == 0` and
  `out.len_bytes() == elem_count * out_elem_size`, then walks `out[i] = convert(in[i])`.
- **`cast_kernel_to_fp8!` / `cast_kernel_from_fp8!`** (`byte_kernels.rs:3481-3570`) — `float8::F8E4M3`
  is 1 byte and does **not** implement `bytemuck::Pod`, so these handle F8E4M3 as raw `u8` via
  `from_bits` / `to_bits`. `to_fp8` validates `out.len_bytes() == elem_count` (1 byte/elem);
  `from_fp8` validates `out.len_bytes() == elem_count * out_elem_size`. The `f16↔F8E4M3` and
  `bf16↔F8E4M3` directions **pivot through f32** (`float8::F8E4M3` only exposes `from_f32`/`to_f32`);
  the f32 pivot leg is lossless for both f16 and bf16 (each is a strict subset of f32).

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

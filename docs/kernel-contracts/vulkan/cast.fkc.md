---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — cast (dtype-conversion) kernel contracts

Dtype-conversion kernels for the Vulkan backend (crate `vulkan`, family `cast`). Every kernel here
implements `OpKind::Cast` (`fuel-core-types/src/dispatch.rs:117`) for one fixed source→destination
dtype pair. The pair is the dispatch key: `(OpKind::Cast, [SRC, DST], Vulkan) + kernel_source`
(§3.2, §12.1). The `OpParams::Cast` variant is a unit marker (`fuel-dispatch/src/kernel.rs:352`) —
the target dtype lives on the output Storage's `dtype` field, so the output dtype rule is
`cast(output)` (§5.1).

Three structural classes appear, faithful to the inventory
(`docs/kernel-contracts/_inventory/vulkan.md` "Casts", lines 105-131):

- **Pair-packed half casts** (`f32↔f16`, `f32↔bf16`): one thread per output u32 word = 2 elements.
  The `f32→half` direction requires an even element count (`n` even; the wrapper pads odd at
  `cast_f32_bytes`). f32↔f16 use `f32tof16`/`f16tof32` (RNE); f32↔bf16 use bit shifts (f32→bf16
  truncate `bits>>16`; bf16→f32 exact `bits<<16`).
- **One-per-element wide casts** (`f32↔f64`): one thread per element, NOT packed, 1:1; f64→f32 RNE.
- **Byte-packed F8E4M3 casts** (`F8E4M3 ↔ {f32, f16, bf16}`): F8 is 1 byte, 4 packed per u32. All
  non-f8 sides are routed via f32 internally, so `f16↔f8e4m3` / `bf16↔f8e4m3` narrow through f32.
  f32→F8E4M3 is RNE with saturation to ±448. SPIR-V only (`cast_*_f8e4m3.spv`) — these contracts
  are read from the Rust wrapper (`cast_f8e4m3_bytes`, `fuel-vulkan-backend/src/lib.rs:9451`) and
  the `EMBEDDED` doc comments.

**Universal facts for every cast in this file.** Input is **contiguous-only** (no stride/broadcast/
offset path — the planner must insert an `Op::Contiguize`, itself an FKC kernel, for any
non-contiguous operand; §4.3). Output is always freshly-allocated **contiguous**, same logical
shape as the input, no aliasing, not in-place. Every kernel is bandwidth-bound elementwise: it
reads N source elements and writes N destination elements, so `bytes_moved` is derivable
(`n*(src_bytes+dst_bytes)`) while `flops`, `overhead_ns`, and the precise frontier number are
`judge_measured` — the Judge bootstraps cost (§4.4); no cost number is fabricated here.

---

## cast_f32_to_f16  (F32 → F16, pair-packed, RNE)

One-line: Cast F32 → F16, two elements per thread (packed u32), round-to-nearest-even.

Narrowing cast from F32 to F16. Each thread handles one output u32 word holding two packed
`float16_t` values, so the element count `n` must be even (the wrapper pads an odd count at
`cast_f32_bytes`, `fuel-vulkan-backend/src/lib.rs:2460`). The conversion uses the hardware
`f32tof16` path (round-to-nearest-even), so values outside the F16 normal range overflow to ±inf
and subnormals round per IEEE-754 half semantics. Input is contiguous-only, packed; output is a
fresh contiguous F16 buffer of the same logical shape. Bandwidth-bound: reads `n` f32 (4 B) and
writes `n` f16 (2 B); no compute beyond the per-element narrowing. The dispatch key is
`(Cast, [F32, F16], Vulkan)`; the same `OpKind::Cast` is shared with every other pair in this file,
distinguished by the operand dtype slots.

```fkc
kernel: cast_f32_to_f16
op_kind: Cast
blurb: "Cast F32 -> F16, two elements per thread (packed u32), round-to-nearest-even."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cast_f32_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "pair-packed (one thread per output u32 = 2 elems); n must be even (wrapper pads odd)."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F16)            # target dtype = F16 (was cast(output); FKC-recognized §5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (its own FKC kernel) + sums its cost
  fast_paths:
    - { when: "dim[-1] % 2 == 0", note: "no odd-tail pad; one thread per packed u32" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; no fabricated number (§4.4)
  class: cheap_elementwise
  bytes_moved: "n * (4 + 2)"            # read n f32 (4 B) + write n f16 (2 B) — bandwidth-bound elementwise
  memory: { device_bytes: "n * 2", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                        # author-declared seed; Judge audits (§4.8)
  notes: "f32tof16 round-to-nearest-even; values beyond F16 range overflow to +/-inf per IEEE-754 half."

determinism: same_hardware_bitwise
```

---

## cast_f16_to_f32  (F16 → F32, pair-packed, exact)

One-line: Cast F16 → F32, two elements per thread (packed u32), exact widening.

Widening cast from F16 to F32. Each thread reads one source u32 word holding two packed
`float16_t` values and writes two F32 outputs via the hardware `f16tof32` path. Widening F16→F32 is
**exact** (every representable F16 value, including inf/NaN/subnormals, maps to its exact F32
equivalent). Input is contiguous-only, packed; output is a fresh contiguous F32 buffer of the same
logical shape. Bandwidth-bound: reads `n` f16 (2 B), writes `n` f32 (4 B). Dispatch key
`(Cast, [F16, F32], Vulkan)`.

```fkc
kernel: cast_f16_to_f32
op_kind: Cast
blurb: "Cast F16 -> F32, two elements per thread (packed u32), exact widening."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cast_f16_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "pair-packed source (two f16 per u32 word)."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)            # target dtype = F32 (was cast(output); FKC-recognized §5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "n * (2 + 4)"            # read n f16 (2 B) + write n f32 (4 B)
  memory: { device_bytes: "n * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16->f32 widening is exact (every F16 value, incl. inf/NaN/subnormal, maps exactly)."

determinism: same_hardware_bitwise
```

---

## cast_f32_to_bf16  (F32 → BF16, pair-packed, truncate)

One-line: Cast F32 → BF16, two elements per thread (packed u32), upper-16-bit truncation.

Narrowing cast from F32 to BF16. Each thread produces one output u32 word holding two packed BF16
values, so `n` must be even (wrapper pads odd, `cast_f32_bytes`,
`fuel-vulkan-backend/src/lib.rs:2460`). BF16 shares F32's exponent, so the conversion is a bit
operation: the inventory wrapper note records this as **truncation** (`bits >> 16`,
truncate-toward-zero of the F32 mantissa), NOT round-to-nearest. This is the load-bearing numeric
fact — it differs from the elementwise `*_bf16` store paths elsewhere in the Vulkan stack, which
use RNE upper-16. Input is contiguous-only, packed; output is a fresh contiguous BF16 buffer of the
same logical shape. Bandwidth-bound: reads `n` f32 (4 B), writes `n` bf16 (2 B). Dispatch key
`(Cast, [F32, BF16], Vulkan)`.

```fkc
kernel: cast_f32_to_bf16
op_kind: Cast
blurb: "Cast F32 -> BF16, two elements per thread (packed u32), upper-16-bit truncation."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cast_f32_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "pair-packed (one thread per output u32 = 2 elems); n must be even (wrapper pads odd)."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BF16)           # target dtype = BF16 (was cast(output); FKC-recognized §5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dim[-1] % 2 == 0", note: "no odd-tail pad; one thread per packed u32" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "n * (4 + 2)"            # read n f32 (4 B) + write n bf16 (2 B)
  memory: { device_bytes: "n * 2", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32->bf16 by TRUNCATION (bits>>16, truncate-toward-zero of F32 mantissa) per wrapper note; NOT RNE. BF16 shares F32 exponent so no overflow."

determinism: same_hardware_bitwise
```

---

## cast_bf16_to_f32  (BF16 → F32, pair-packed, exact)

One-line: Cast BF16 → F32, two elements per thread (packed u32), exact widening.

Widening cast from BF16 to F32. Each thread reads one source u32 word holding two packed BF16
values and writes two F32 outputs. Because BF16 is literally the upper 16 bits of F32, the widening
is **exact** — the kernel left-shifts the bits (`bits << 16`), reproducing the F32 value with zero
mantissa loss (inf/NaN preserved). Input is contiguous-only, packed; output is a fresh contiguous
F32 buffer of the same logical shape. Bandwidth-bound: reads `n` bf16 (2 B), writes `n` f32 (4 B).
Dispatch key `(Cast, [BF16, F32], Vulkan)`.

```fkc
kernel: cast_bf16_to_f32
op_kind: Cast
blurb: "Cast BF16 -> F32, two elements per thread (packed u32), exact widening."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cast_bf16_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "pair-packed source (two bf16 per u32 word)."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)            # target dtype = F32 (was cast(output); FKC-recognized §5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "n * (2 + 4)"            # read n bf16 (2 B) + write n f32 (4 B)
  memory: { device_bytes: "n * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16->f32 widening is EXACT (bits<<16); BF16 is the upper 16 bits of F32, no mantissa loss, inf/NaN preserved."

determinism: same_hardware_bitwise
```

---

## cast_f32_to_f64  (F32 → F64, one thread per element, exact)

One-line: Cast F32 → F64, one thread per element (not packed), exact widening.

Widening cast from F32 to F64. One thread per element (NOT packed), strict 1:1
(`cast_f32_f64_bytes`, `fuel-vulkan-backend/src/lib.rs:2558`). F32→F64 widening is **exact**: every
F32 value, including inf/NaN/subnormals, has an exact F64 representation. Requires the device to
support native `double` (shaderFloat64). Input is contiguous-only, 1:1; output is a fresh
contiguous F64 buffer of the same logical shape. Bandwidth-bound: reads `n` f32 (4 B), writes `n`
f64 (8 B). Dispatch key `(Cast, [F32, F64], Vulkan)`.

```fkc
kernel: cast_f32_to_f64
op_kind: Cast
blurb: "Cast F32 -> F64, one thread per element (not packed), exact widening."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cast_f32_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "one thread per element, 1:1 (NOT packed)."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F64)            # target dtype = F64 (was cast(output); FKC-recognized §5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "n * (4 + 8)"            # read n f32 (4 B) + write n f64 (8 B)
  memory: { device_bytes: "n * 8", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32->f64 widening is EXACT; requires device shaderFloat64 (native double)."

determinism: same_hardware_bitwise
```

---

## cast_f64_to_f32  (F64 → F32, one thread per element, RNE)

One-line: Cast F64 → F32, one thread per element (not packed), round-to-nearest-even.

Narrowing cast from F64 to F32. One thread per element (NOT packed), strict 1:1
(`cast_f32_f64_bytes`, `fuel-vulkan-backend/src/lib.rs:2558`). F64→F32 narrowing uses
round-to-nearest-even; values outside the F32 normal range overflow to ±inf and subnormals round
per IEEE-754 single semantics. Requires native `double` support to load the F64 input. Input is
contiguous-only, 1:1; output is a fresh contiguous F32 buffer of the same logical shape.
Bandwidth-bound: reads `n` f64 (8 B), writes `n` f32 (4 B). Dispatch key `(Cast, [F64, F32], Vulkan)`.

```fkc
kernel: cast_f64_to_f32
op_kind: Cast
blurb: "Cast F64 -> F32, one thread per element (not packed), round-to-nearest-even."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cast_f64_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "one thread per element, 1:1 (NOT packed)."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)            # target dtype = F32 (was cast(output); FKC-recognized §5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "n * (8 + 4)"            # read n f64 (8 B) + write n f32 (4 B)
  memory: { device_bytes: "n * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f64->f32 narrowing is round-to-nearest-even; overflow to +/-inf per IEEE-754 single; requires device shaderFloat64."

determinism: same_hardware_bitwise
```

---

## cast_f32_to_f8e4m3  (F32 → F8E4M3, byte-packed, RNE saturate ±448)

One-line: Cast F32 → F8E4M3 (1-byte float), 4 per u32, RNE with saturation to ±448.

Narrowing cast from F32 to F8E4M3 (1-byte E4M3 float; `DType::F8E4M3`,
`fuel-core-types/src/dtype.rs:38`). F8 elements are 1 byte each, 4 packed per u32 word. The
conversion is round-to-nearest-even with **saturation to the E4M3 finite range ±448** (E4M3 has no
inf encoding; out-of-range values clamp rather than overflow). SPIR-V only (`cast_f32_to_f8e4m3.spv`);
the contract is read from the Rust wrapper `cast_f8e4m3_bytes`
(`fuel-vulkan-backend/src/lib.rs:9451`) and the `EMBEDDED` doc comments. Input is contiguous-only,
byte-packed; output is a fresh contiguous F8E4M3 buffer of the same logical shape. Bandwidth-bound:
reads `n` f32 (4 B), writes `n` f8 (1 B). This is a sub-byte-adjacent narrowing but F8E4M3 is a full
1-byte `DType` (`size_in_bytes() == 1`, `dtype.rs:122`), so no FDX sub-byte/quant descriptor is
required — it is an ordinary dense dtype cast. Dispatch key `(Cast, [F32, F8E4M3], Vulkan)`.

```fkc
kernel: cast_f32_to_f8e4m3
op_kind: Cast
blurb: "Cast F32 -> F8E4M3 (1-byte float), 4 per u32, RNE with saturation to +/-448."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cast_f32_to_f8e4m3"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "byte-packed output (F8 is 1 byte, 4 per u32)."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F8E4M3)         # target dtype = F8E4M3 (was cast(output); FKC-recognized §5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32           # F8 sub-word writes packed 4-per-u32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "n * (4 + 1)"            # read n f32 (4 B) + write n f8 (1 B)
  memory: { device_bytes: "n * 1", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32->F8E4M3 round-to-nearest-even, SATURATE to E4M3 finite range +/-448 (E4M3 has no inf encoding)."

determinism: same_hardware_bitwise
```

---

## cast_f8e4m3_to_f32  (F8E4M3 → F32, byte-packed, exact decode)

One-line: Cast F8E4M3 → F32 (1-byte float), 4 per u32, exact E4M3 decode.

Widening cast from F8E4M3 to F32. F8 elements are 1 byte, read 4-per-u32; each is decoded to its
exact F32 value. The decode is **exact** — every finite E4M3 code (including NaN) maps to its exact
F32 equivalent (E4M3 has no inf encoding). SPIR-V only (`cast_f8e4m3_to_f32.spv`); contract read
from the Rust wrapper `cast_f8e4m3_bytes` (`fuel-vulkan-backend/src/lib.rs:9451`). Input is
contiguous-only, byte-packed; output is a fresh contiguous F32 buffer of the same logical shape.
Bandwidth-bound: reads `n` f8 (1 B), writes `n` f32 (4 B). F8E4M3 is a full 1-byte `DType`, so no
FDX sub-byte/quant descriptor is needed. Dispatch key `(Cast, [F8E4M3, F32], Vulkan)`.

```fkc
kernel: cast_f8e4m3_to_f32
op_kind: Cast
blurb: "Cast F8E4M3 -> F32 (1-byte float), 4 per u32, exact E4M3 decode."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cast_f8e4m3_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "byte-packed input (F8 is 1 byte, 4 per u32)."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)            # target dtype = F32 (was cast(output); FKC-recognized §5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "n * (1 + 4)"            # read n f8 (1 B) + write n f32 (4 B)
  memory: { device_bytes: "n * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "F8E4M3->f32 decode is EXACT (every finite E4M3 code, incl. NaN, maps exactly; E4M3 has no inf)."

determinism: same_hardware_bitwise
```

---

## cast_f16_to_f8e4m3  (F16 → F8E4M3, via F32, RNE saturate ±448)

One-line: Cast F16 → F8E4M3, routed through F32, RNE with saturation to ±448.

Narrowing cast from F16 to F8E4M3. Per the inventory, all non-F8 sides are **routed via F32**
internally: the F16 source is widened to F32 (exact), then narrowed to F8E4M3 with
round-to-nearest-even and saturation to ±448. The F16→F32 leg is exact, so the only rounding/clamp
is the F32→F8E4M3 leg — net behavior is equivalent to a direct RNE-saturate narrow of the F16
value. SPIR-V only (`cast_f16_to_f8e4m3.spv`); contract read from `cast_f8e4m3_bytes`
(`fuel-vulkan-backend/src/lib.rs:9451`). Input is contiguous-only; output is a fresh contiguous
F8E4M3 buffer of the same logical shape. Bandwidth-bound: reads `n` f16 (2 B), writes `n` f8 (1 B).
Dispatch key `(Cast, [F16, F8E4M3], Vulkan)`.

```fkc
kernel: cast_f16_to_f8e4m3
op_kind: Cast
blurb: "Cast F16 -> F8E4M3, routed through F32, RNE with saturation to +/-448."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cast_f16_to_f8e4m3"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "non-F8 side routed via F32 internally; byte-packed F8 output."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F8E4M3)         # target dtype = F8E4M3 (was cast(output); FKC-recognized §5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "n * (2 + 1)"            # read n f16 (2 B) + write n f8 (1 B)
  memory: { device_bytes: "n * 1", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "routed via F32 (F16->F32 exact); F32->F8E4M3 leg is RNE + SATURATE to +/-448. Net == direct RNE-saturate narrow of the F16 value."

determinism: same_hardware_bitwise
```

---

## cast_f8e4m3_to_f16  (F8E4M3 → F16, via F32, RNE)

One-line: Cast F8E4M3 → F16, routed through F32, round-to-nearest-even.

Widening cast from F8E4M3 to F16. Routed via F32: the F8E4M3 source is decoded to F32 (exact), then
narrowed to F16 with round-to-nearest-even. Since F8E4M3's finite range (±448) lies within the F16
normal range, no overflow occurs; the F32→F16 leg may round mantissa bits but never saturates for
in-range E4M3 inputs (the only rounding is mapping the wider E4M3 mantissa landing onto F16 — in
practice E4M3 has fewer mantissa bits than F16, so finite E4M3 values are represented exactly in
F16). SPIR-V only (`cast_f8e4m3_to_f16.spv`); contract read from `cast_f8e4m3_bytes`
(`fuel-vulkan-backend/src/lib.rs:9451`). Input is contiguous-only; output is a fresh contiguous F16
buffer of the same logical shape. Bandwidth-bound: reads `n` f8 (1 B), writes `n` f16 (2 B).
Dispatch key `(Cast, [F8E4M3, F16], Vulkan)`.

```fkc
kernel: cast_f8e4m3_to_f16
op_kind: Cast
blurb: "Cast F8E4M3 -> F16, routed through F32, round-to-nearest-even."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cast_f8e4m3_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "byte-packed F8 input; non-F8 side routed via F32 internally."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F16)            # target dtype = F16 (was cast(output); FKC-recognized §5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "n * (1 + 2)"            # read n f8 (1 B) + write n f16 (2 B)
  memory: { device_bytes: "n * 2", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "routed via F32 (F8E4M3->F32 exact); F32->F16 leg is RNE. E4M3 finite range +/-448 is within F16 normals (no overflow)."

determinism: same_hardware_bitwise
```

---

## cast_bf16_to_f8e4m3  (BF16 → F8E4M3, via F32, RNE saturate ±448)

One-line: Cast BF16 → F8E4M3, routed through F32, RNE with saturation to ±448.

Narrowing cast from BF16 to F8E4M3. Routed via F32: the BF16 source is widened to F32 (exact —
BF16 is the upper 16 bits of F32), then narrowed to F8E4M3 with round-to-nearest-even and
saturation to ±448. The BF16→F32 leg is exact, so the only rounding/clamp is the F32→F8E4M3 leg;
because BF16 spans a much wider exponent range than E4M3, large BF16 magnitudes saturate to ±448.
SPIR-V only (`cast_bf16_to_f8e4m3.spv`); contract read from `cast_f8e4m3_bytes`
(`fuel-vulkan-backend/src/lib.rs:9451`). Input is contiguous-only; output is a fresh contiguous
F8E4M3 buffer of the same logical shape. Bandwidth-bound: reads `n` bf16 (2 B), writes `n` f8 (1 B).
Dispatch key `(Cast, [BF16, F8E4M3], Vulkan)`.

```fkc
kernel: cast_bf16_to_f8e4m3
op_kind: Cast
blurb: "Cast BF16 -> F8E4M3, routed through F32, RNE with saturation to +/-448."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cast_bf16_to_f8e4m3"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "non-F8 side routed via F32 internally; byte-packed F8 output."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F8E4M3)         # target dtype = F8E4M3 (was cast(output); FKC-recognized §5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "n * (2 + 1)"            # read n bf16 (2 B) + write n f8 (1 B)
  memory: { device_bytes: "n * 1", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "routed via F32 (BF16->F32 exact); F32->F8E4M3 leg is RNE + SATURATE to +/-448. Wide BF16 magnitudes clamp to +/-448."

determinism: same_hardware_bitwise
```

---

## cast_f8e4m3_to_bf16  (F8E4M3 → BF16, via F32, truncate)

One-line: Cast F8E4M3 → BF16, routed through F32, upper-16-bit truncation on the BF16 store.

Widening-then-restore cast from F8E4M3 to BF16. Routed via F32: the F8E4M3 source is decoded to F32
(exact), then narrowed to BF16. Following the Vulkan cast convention for the F32→BF16 leg (see
`cast_f32_to_bf16`), the inventory records the BF16 store as **truncation** of the F32 upper 16
bits (`bits >> 16`), NOT round-to-nearest. Since every finite E4M3 value is exactly representable in
both F32 and BF16 (E4M3 has 3 mantissa bits, fewer than BF16's 7, and its exponent range fits BF16),
the truncation leg is lossless for in-range E4M3 inputs in practice; the truncation note is retained
as the load-bearing numeric fact for the general path. SPIR-V only (`cast_f8e4m3_to_bf16.spv`);
contract read from `cast_f8e4m3_bytes` (`fuel-vulkan-backend/src/lib.rs:9451`). Input is
contiguous-only; output is a fresh contiguous BF16 buffer of the same logical shape. Bandwidth-bound:
reads `n` f8 (1 B), writes `n` bf16 (2 B). Dispatch key `(Cast, [F8E4M3, BF16], Vulkan)`.

```fkc
kernel: cast_f8e4m3_to_bf16
op_kind: Cast
blurb: "Cast F8E4M3 -> BF16, routed through F32, upper-16-bit truncation on the BF16 store."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cast_f8e4m3_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "byte-packed F8 input; non-F8 side routed via F32 internally."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BF16)           # target dtype = BF16 (was cast(output); FKC-recognized §5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  bytes_moved: "n * (1 + 2)"            # read n f8 (1 B) + write n bf16 (2 B)
  memory: { device_bytes: "n * 2", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "routed via F32 (F8E4M3->F32 exact); F32->BF16 leg is TRUNCATION (bits>>16) per Vulkan cast convention. In-range E4M3 (3 mantissa bits) fits BF16 exactly."

determinism: same_hardware_bitwise
```

---
fkc_version: 1
provider:
  name: fuel-quantized
  backend: Cpu                       # backend-agnostic numerics, reached through the CPU dyn adapter (BackendId::Cpu)
  kernel_source: "fuel-quantized"    # the BindingEntry.kernel_source tag
  link_registry: fuel_quantized::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-quantized — dequantize (`GgmlType::to_float`) kernel contracts

The backend-agnostic ggml/gguf **dequantize** family: one `GgmlType::to_float` impl per block
format plus the three float "block" formats (`f32`/`f16`/`bf16`, `BLCK_SIZE == 1`,
`DIRECT_COPY`). Each reconstructs a dense **F32** buffer from a packed block stream. These are the
numeric kernels behind the dequantize direction; the CPU dyn adapter
(`fuel_quantized::cpu::QuantizedType::dequantize`, `cpu.rs`) forwards a dequantize dispatch to the
matching `to_float`, writing into a `HostBuffer::F32`. The same impls also feed the quantized
`matmul` driver (the on-the-fly per-block dequant inside `vec_dot`), but the contracts here
describe the **standalone dequant** op only.

> **AS-BUILT DISPATCH NOTE — read before trusting `op_kind` (mirrors the sibling quantize bundle).**
> There is **no `Dequantize` `OpKind`** in the as-built dispatch enum
> (`fuel-core-types/src/dispatch.rs`); the only quant-direction `OpKind` these block formats
> participate in is `QMatMul`. These dequantizers are therefore **not** registered as standalone
> primitive kernels on the `KernelBindingTable` at a `(Dequantize, dtypes, backend)` key — they reach
> the dispatch surface through the **`DynQuantizedStorage` / `QuantizedType::dequantize` trait method**
> (the CPU adapter `fuel-quantized/src/cpu.rs` forwards to the matching `GgmlType::to_float`). Per the
> never-invent / never-re-number discipline (§0), the `op_kind:` slot below names the **closest honest
> dispatch tag**, `QMatMul`, and each kernel records — in prose, in `entry_point`, and in `caps.notes`
> — that the real dequant path is the `QuantizedType::dequantize` trait method, not a dedicated key.
> A future `OpKind::Dequantize` (or a `DynQuantizedStorage`-trait FKC import surface) would let these
> register as their own keyed kernels; until it lands the `op_kind` slot is the closest faithful tag
> and the trait-method path is authoritative.
>
> **Only three dequantize `Capability` tokens exist as-built** (`fuel-core-types/src/capability.rs`):
> `DequantizeQ8_0`, `DequantizeQ4_0`, and `DequantizeQ4KM`. There is **no** `Capability::Dequantize`
> umbrella token and **no** per-format token for Q4_1/Q5_0/Q5_1/Q2K/Q3K/Q5K/Q6K/Q8K or the float
> formats — those op-level capability codes **do not yet exist** and are documented as gaps below
> (each such kernel records `op-level capability: NONE as-built — gap`). A kernel does not invent a
> capability token it lacks.

Cross-cutting facts for this family (from the quantized inventory, "Crate-wide layout reality" /
"Dequantize — `GgmlType::to_float`" / "Float block formats"):

- **Flat slices only — NO `Layout`, NO `Shape`, NO `StridedIndex`, NO offset, NO broadcast.** Every
  kernel takes `xs: &[BlockX]` (or `&[f32]`/`&[f16]`/`&[bf16]` for the float formats) and writes
  `ys: &mut [f32]`. All stride/offset/broadcast handling lives in the backend adapters
  (`fuel-cpu-backend`, `fuel-core/src/quantized/`) which contiguify **before** calling in. The
  contract is therefore **contiguous, zero-offset, no-broadcast, dense row-major** for every
  operand: `requires_contiguous` throughout, all five layout flags rejected except `contiguous:
  required`, and `reverse_strides: rejected` everywhere (none of these kernels walk negative
  strides). The planner inserts (and costs, from the relevant `Op::Contiguize` contract, §4.3/§4.4)
  a contiguize for any non-contiguous producer.
- **Output dtype is always F32; output shape = `xs.len() * BLCK_SIZE` elements, dense, block order.**
  The output buffer is caller-pre-allocated to the exact element count and **fully overwritten** (no
  read of prior content, no input/output aliasing): `aliasing: none`, `layout_guarantee:
  preallocated` + `contiguous`, `dtype_rule: fixed(F32)`. Block-format kernels require the output
  length `k` to be a multiple of `BLCK_SIZE` and equal to `xs.len() * BLCK_SIZE` (32 for the legacy
  `Q4_0..Q8_1`, 256 = `QK_K` for the K-quants `Q2K..Q8K`). K-quants validate via
  `group_for_dequantization`; legacy formats validate via `debug_assert!` only (in release a size
  mismatch is UB/panic-on-OOB, **not** a typed `Result::Err`).
- **Scale single-place rule (§3.9.3).** GGML block scales are **INLINE** in the packed block — the
  GGML `#[repr(C)]` block carries its own scale bytes (f16 for the legacy formats and `Q2K..Q6K`,
  f32 for `Q8K`). There is **no** separate FKC scale operand: `fdx.quant.scale_operand` stays `~`
  and the scale rides the FDX tensor's `scale_buffer` (placement INLINE). The dequant *applies* the
  inline scale in this single place; declaring it both inline and as an operand would be
  `ScaleDoubleDeclared` (§10.6).
- **Cost is `judge_measured` (formula hints only; no fabricated constants).** No FLOPs/bandwidth
  coefficient is fabricated. Dequant is a pure unpack-and-scale stream with **no arithmetic FLOPs in
  the matmul sense** (`flops` is effectively 0 — it is bit-unpacking plus one f32 multiply-add per
  element), and it is **bandwidth-bound**: it reads the packed block stream (`xs.len()` blocks × the
  per-format block byte size) and writes `xs.len() * BLCK_SIZE * 4` output bytes. The cost block is
  marked `provenance: judge_measured` because the per-format unpack cost varies widely (2-bit
  super-block scale reconstruction vs an 8-bit i8 copy vs a plain float convert) and is exactly the
  kind of per-format constant best measured, not guessed. Under the match-content rule (§4.4): a
  `judge_measured` block carries only **derivable formula hints** (`bytes_moved` written as a real
  read+write formula) and `~` for any non-derivable absolute constant (`overhead_ns`, the i8/unpack
  `flops`), **never** a fabricated number and **never** the `judge_measured` token sitting inside a
  numeric field. FKC stays agnostic to *how* the Judge measures; it records only that the provenance
  is measurement.
- **Precision is author-declared, Judge-audited.** Dequant *reconstructs* f32 from the already-
  quantized block; the lossy step (the original quantization) is fixed at quantize time and is
  **not introduced by `to_float`**. The reconstruction itself — `(quant value) × scale (+ min)`,
  with the scale read from f16 via `f16::to_f32` (legacy + `Q2K..Q6K`) or f32 (`Q8K`) — is exact
  for the dequantized operands and **deterministic** (fixed per-element order). Each is therefore
  `bit_stable_on_same_hardware: true` with no cross-quant ULP bound (the error is dominated by the
  pre-baked weight quantization, audited at the model level). The float formats `f16`/`bf16` →
  `to_float` are exact widenings (`max_ulp: 0`); `f32` → `to_float` is an exact copy.
- **`to_float_q8_1` is `unimplemented!()` and PANICS** (`k_quants.rs:759`). Q8_1 exists only as a
  `VecDotType` (the activation-side block for the Q4_1/Q5_1/Q8_1 dot product); it has no dequant.
  Its section below is written as a **non-registrable, panicking** kernel and is reported as such —
  it cannot be faithfully contracted as a working dequant.

---

## to_float_q4_0  (Q4_0 4-bit block dequantize → F32)

One-line: Dequantize a Q4_0 4-bit block stream to dense F32; (nibble - 8) * d(f16); contiguous, block order.

Reconstructs `ys[blk*32 + i] = (nibble_i - 8) * d` over a flat `&[BlockQ4_0]` input
(`k_quants.rs:177`). Each 32-element `BlockQ4_0` is a 2-byte f16 scale `d` plus 16 bytes of packed
u4 quants (18 bytes/block); the 4-bit quant is centered by subtracting 8 and scaled by `d` (read
via `f16::to_f32`). Output is dense F32 of length `xs.len() * 32`, written in block order. The
output length `k` MUST be a multiple of 32 and equal `xs.len() * 32` (legacy `debug_assert!` only —
in release a mismatch is UB/OOB, not a typed `Err`). Numerics: exact reconstruction of the
dequantized operands; deterministic per-element. Perf: bandwidth-bound — reads ~`xs.len() * 18`
packed bytes, writes `xs.len() * 32 * 4` F32 bytes; no matmul-style FLOPs (one unpack + one f32
multiply per element). Limitation: contiguous, zero-offset, dense-row-major only.

Dispatch: dequantize has NO dedicated `OpKind`; the closest honest tag is `QMatMul` and the real
path is the `QuantizedType::dequantize` trait method (bundle note). The source's quant facts
(`family=GGML_BLOCK, ggml_dtype=Q4_0`) enrich its operand slot. Op-level capability:
`Capability::DequantizeQ4_0` — one of the **three** dequantize tokens that actually exist
(`capability.rs`).

FLOPs/bandwidth hint: `flops ≈ 0` (unpack + scale, not MACs); read ~`xs.len() * 18` bytes, write
`xs.len() * 32 * 4` bytes (= `n * 4`, `n` = output elements). Marked `judge_measured` (per-format
unpack cost measured, not fabricated).

```fkc
kernel: to_float_q4_0
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "Dequantize a Q4_0 4-bit block stream to dense F32; (nibble - 8) * d(f16); contiguous, block order."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_q4_0"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]                    # opaque block byte stream (reinterpreted as &[BlockQ4_0]); kDLUInt bits:8 packed-quant stand-in (FDX §3)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                         # flat [xs.len()] blocks; no Layout in this crate
      shape_constraint: "divisible(out.dim[-1], 32)"   # out length k must be a multiple of BLCK_SIZE=32
      fdx:
        requires_ext: true            # the U8 base is meaning-bearing: it IS Q4_0 blocks
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q4_0            # GgmlDType variant name (code 2); §3.4 — ggml_dtype IS the format
          granularity: ~             # GGML_BLOCK carries ggml_dtype ONLY; NO PerBlock (PerBlock is MX-only, FDX §6.2/V5)
          role: weight
          scale_operand: ~            # INLINE block scale — single-place rule: NOT a separate operand
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # dequant output is always F32
      shape_rule: from_params(src)    # xs.len() * 32 elements, dense block order
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. Op-level Capability::DequantizeQ4_0 exists. out length multiple of 32 debug-asserted only."

cost:
  provenance: judge_measured          # Judge bootstraps; per-format unpack cost measured, not fabricated
  class: cheap_elementwise            # bandwidth-bound unpack-and-scale stream
  flops: ~                            # unpack + one f32 scale per element; not MACs — Judge-measured
  bytes_moved: "(n / 32) * 18 + n * 4"   # read (n/32) blocks * 18 B + write n*4 F32 bytes (n = output elements)
  overhead_ns: ~                      # launch/call overhead Judge-measured
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }   # F32 output alloc

precision:
  bit_stable_on_same_hardware: true   # deterministic per-element unpack/scale, fixed order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact reconstruction (nibble-8)*d of the dequantized operands; f16 scale via to_f32. Only lossy step is the pre-baked Q4_0 quantization, audited at model level. Deterministic."

determinism: same_hardware_bitwise
```

---

## to_float_q4_1  (Q4_1 4-bit affine block dequantize → F32)

One-line: Dequantize a Q4_1 4-bit affine block stream to dense F32; nibble * d + m (f16 scale/min); contiguous, block order.

Reconstructs `ys[blk*32 + i] = nibble_i * d + m` over a flat `&[BlockQ4_1]` input
(`k_quants.rs:354`). Each 32-element `BlockQ4_1` is a 2-byte f16 scale `d`, a 2-byte f16 min `m`,
and 16 bytes of packed u4 quants (20 bytes/block) — Q4_0 plus a per-block minimum (affine block
quant). The uncentered 4-bit quant is scaled by `d` and offset by `m` (both read via `f16::to_f32`).
Output dense F32 length `xs.len() * 32`, block order, `k % 32 == 0` (legacy `debug_assert!`). Same
structure/numerics as Q4_0 with the per-block min reconstructed during dequant. Bandwidth-bound:
read ~`xs.len() * 20` bytes, write `xs.len() * 32 * 4`. Contiguous-only.

Dispatch: closest honest tag `QMatMul`; real path `QuantizedType::dequantize` (bundle note). Source
`ggml_dtype=Q4_1` (code 3). Op-level capability: **NONE as-built — gap** (no `Capability::DequantizeQ4_1`
exists; only Q8_0/Q4_0/Q4KM do).

FLOPs/bandwidth hint: `flops ≈ 0`; read ~`xs.len() * 20` bytes, write `n * 4`. Marked
`judge_measured`.

```fkc
kernel: to_float_q4_1
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "Dequantize a Q4_1 4-bit affine block stream to dense F32; nibble * d + m (f16 scale/min); contiguous, block order."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_q4_1"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(out.dim[-1], 32)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q4_1, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. NO op-level Capability::DequantizeQ4_1 exists (gap). out length multiple of 32 debug-asserted only."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                            # unpack + scale per element; not MACs — Judge-measured
  bytes_moved: "(n / 32) * 20 + n * 4"   # read (n/32) blocks * 20 B + write n*4 F32 bytes
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact reconstruction nibble*d + m of the dequantized operands; f16 scale/min via to_f32. Only lossy step is the pre-baked Q4_1 (affine, per-block d+m) quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## to_float_q5_0  (Q5_0 5-bit block dequantize → F32)

One-line: Dequantize a Q5_0 5-bit block stream to dense F32; 5th bit from qh, (val - 16) * d(f16); contiguous, block order.

Reconstructs `ys[blk*32 + i] = (val_i - 16) * d` over a flat `&[BlockQ5_0]` input
(`k_quants.rs:463`). Each 32-element `BlockQ5_0` is a 2-byte f16 scale `d`, a 4-byte high-bit field
`qh`, and 16 bytes of packed u4 quants (22 bytes/block); the 5th bit per quant is reassembled from
`qh` (giving a 5-bit value `val`), centered by subtracting 16, and scaled by `d` (f16 via
`to_f32`). Output dense F32 length `xs.len() * 32`, block order, `k % 32 == 0` (legacy
`debug_assert!`). Same structure as Q4_0 with the 5th bit reassembly. Bandwidth-bound: read
~`xs.len() * 22` bytes, write `xs.len() * 32 * 4`. Contiguous-only.

Dispatch: closest honest tag `QMatMul`; real path `QuantizedType::dequantize` (bundle note). Source
`ggml_dtype=Q5_0` (code 6). Op-level capability: **NONE as-built — gap** (no `Capability::DequantizeQ5_0`).

FLOPs/bandwidth hint: `flops ≈ 0`; read ~`xs.len() * 22` bytes, write `n * 4`. Marked
`judge_measured`.

```fkc
kernel: to_float_q5_0
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "Dequantize a Q5_0 5-bit block stream to dense F32; 5th bit from qh, (val - 16) * d(f16); contiguous, block order."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_q5_0"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(out.dim[-1], 32)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q5_0, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. NO op-level Capability::DequantizeQ5_0 exists (gap). out length multiple of 32 debug-asserted only."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                            # unpack + scale per element; not MACs — Judge-measured
  bytes_moved: "(n / 32) * 22 + n * 4"   # read (n/32) blocks * 22 B + write n*4 F32 bytes
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact reconstruction (val-16)*d with the 5th bit reassembled from qh; f16 scale via to_f32. Only lossy step is the pre-baked Q5_0 (5-bit) quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## to_float_q5_1  (Q5_1 5-bit affine block dequantize → F32)

One-line: Dequantize a Q5_1 5-bit affine block stream to dense F32; 5th bit from qh, val * d + m (f16 scale/min); contiguous, block order.

Reconstructs `ys[blk*32 + i] = val_i * d + m` over a flat `&[BlockQ5_1]` input
(`k_quants.rs:578`). Each 32-element `BlockQ5_1` is a 2-byte f16 scale `d`, a 2-byte f16 min `m`, a
4-byte high-bit field `qh`, and 16 bytes of packed u4 quants (24 bytes/block) — Q5_0 plus a
per-block min (affine). The 5th bit per quant is reassembled from `qh`; the uncentered 5-bit value
is scaled by `d` and offset by `m` (both f16 via `to_f32`). Output dense F32 length `xs.len() * 32`,
block order, `k % 32 == 0` (legacy `debug_assert!`). Bandwidth-bound: read ~`xs.len() * 24` bytes,
write `xs.len() * 32 * 4`. Contiguous-only.

Dispatch: closest honest tag `QMatMul`; real path `QuantizedType::dequantize` (bundle note). Source
`ggml_dtype=Q5_1` (code 7). Op-level capability: **NONE as-built — gap** (no `Capability::DequantizeQ5_1`).

FLOPs/bandwidth hint: `flops ≈ 0`; read ~`xs.len() * 24` bytes, write `n * 4`. Marked
`judge_measured`.

```fkc
kernel: to_float_q5_1
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "Dequantize a Q5_1 5-bit affine block stream to dense F32; 5th bit from qh, val * d + m (f16 scale/min); contiguous, block order."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_q5_1"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(out.dim[-1], 32)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q5_1, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. NO op-level Capability::DequantizeQ5_1 exists (gap). out length multiple of 32 debug-asserted only."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                            # unpack + scale per element; not MACs — Judge-measured
  bytes_moved: "(n / 32) * 24 + n * 4"   # read (n/32) blocks * 24 B + write n*4 F32 bytes
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact reconstruction val*d + m with the 5th bit reassembled from qh; f16 scale/min via to_f32. Only lossy step is the pre-baked Q5_1 (5-bit affine, per-block d+m) quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## to_float_q8_0  (Q8_0 8-bit block dequantize → F32)

One-line: Dequantize a Q8_0 8-bit block stream to dense F32; i8 * d(f16); contiguous, block order.

Reconstructs `ys[blk*32 + i] = q_i * d` over a flat `&[BlockQ8_0]` input (`k_quants.rs:611`). Each
32-element `BlockQ8_0` is a 2-byte f16 scale `d` plus 32 bytes of i8 quants (34 bytes/block); each
signed i8 quant is scaled by `d` (f16 via `to_f32`). Output dense F32 length `xs.len() * 32`, block
order, `k % 32 == 0` (legacy `debug_assert!`). Numerically the least-lossy legacy format (8-bit
quants); exact deterministic reconstruction. Bandwidth-bound: read ~`xs.len() * 34` bytes, write
`xs.len() * 32 * 4`. Contiguous-only.

Dispatch: closest honest tag `QMatMul`; real path `QuantizedType::dequantize` (bundle note). Source
`ggml_dtype=Q8_0` (code 8). Op-level capability: `Capability::DequantizeQ8_0` — one of the **three**
dequantize tokens that actually exist (`capability.rs`).

FLOPs/bandwidth hint: `flops ≈ 0`; read ~`xs.len() * 34` bytes, write `n * 4`. Marked
`judge_measured`.

```fkc
kernel: to_float_q8_0
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "Dequantize a Q8_0 8-bit block stream to dense F32; i8 * d(f16); contiguous, block order."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_q8_0"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(out.dim[-1], 32)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8_0, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. Op-level Capability::DequantizeQ8_0 exists. out length multiple of 32 debug-asserted only."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                            # unpack + scale per element; not MACs — Judge-measured
  bytes_moved: "(n / 32) * 34 + n * 4"   # read (n/32) blocks * 34 B + write n*4 F32 bytes
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact reconstruction i8*d of the dequantized operands; f16 scale via to_f32. Only lossy step is the pre-baked Q8_0 (8-bit, least lossy legacy) quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## to_float_q8_1  (Q8_1 — `unimplemented!()`; NOT REGISTRABLE / panics)

One-line: Q8_1 has NO dequantize — `to_float` is `unimplemented!()` and panics; Q8_1 exists only as a VecDotType.

**This kernel cannot be faithfully contracted as a working dequantize.** `GgmlType::to_float` for
`BlockQ8_1` is `unimplemented!()` (`k_quants.rs:759`) and **panics** if called. Q8_1 exists only as
the `VecDotType` activation block for the Q4_1/Q5_1/Q8_1 quantized dot product (the matmul builds
it via `from_float`, never dequantizes it). There is therefore no `to_float_q8_1` numeric kernel to
advertise: a dequant contract claiming an output buffer + precision guarantee would be a fiction,
and registering it would put a **panicking** primitive on a production path — a never-panic /
constitutional violation (CLAUDE.md; FKC G9 §1). It is documented here for completeness and is
**reported as un-contractable** (see the return summary). When/if a real `BlockQ8_1::to_float`
lands, it would follow the `to_float_q8_0` shape with `ggml_dtype: Q8_1` (code 9); until then there
is no admissible contract.

> No ` ```fkc ` block is emitted for `to_float_q8_1`: a registrable contract requires a working
> kernel with a real return/precision/cost surface, and this entry point only panics. Emitting a
> block would either lie about the behavior or register a panic on the dispatch surface.

---

## to_float_q2k  (Q2_K 2-bit K-quant super-block dequantize → F32)

One-line: Dequantize a Q2_K 2-bit K-quant super-block stream to dense F32; per-16 sub-scales d/dmin (f16); contiguous, block order.

Reconstructs a dense F32 buffer from a flat `&[BlockQ2K]` input (`k_quants.rs:958`). Each
**256-element super-block** carries 16 packed sub-block scales + 64 bytes of 2-bit-packed quants +
a 2-byte f16 super-scale `d` and a 2-byte f16 super-min `dmin` (84 bytes/block); each of the 16
sub-blocks reconstructs its own scale/min from the packed Q4SCALE field times `d`/`dmin`, then
`ys = sub_scale * q - sub_min`. Output dense F32 length `xs.len() * 256`, block order; the output
length is validated by `group_for_dequantization` (`k % 256 == 0`, `k == xs.len() * 256`). Smallest
footprint K-quant (2-bit), highest quantization error of the family; per-super-block scale
reconstruction makes the unpack cost non-trivial — a clear `judge_measured` case. Bandwidth-bound:
read ~`xs.len() * 84` bytes, write `xs.len() * 256 * 4`. Contiguous-only.

Dispatch: closest honest tag `QMatMul`; real path `QuantizedType::dequantize` (bundle note). Source
`family=GGML_BLOCK, ggml_dtype=Q2K` (code 10). Op-level capability: **NONE as-built — gap** (no
`Capability::DequantizeQ2K`).

FLOPs/bandwidth hint: `flops ≈ 0` (unpack + per-sub-block scale, not MACs); read ~`xs.len() * 84`
bytes, write `n * 4`. Marked `judge_measured`.

```fkc
kernel: to_float_q2k
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "Dequantize a Q2_K 2-bit K-quant super-block stream to dense F32; per-16 sub-scales d/dmin (f16); contiguous, block order."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_q2k"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]                    # opaque super-block byte stream (reinterpreted as &[BlockQ2K]); kDLUInt bits:8 packed-quant stand-in (FDX §3)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                         # flat [xs.len()] super-blocks
      shape_constraint: "divisible(out.dim[-1], 256)"   # QK_K = 256; validated by group_for_dequantization
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q2K, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(src)    # xs.len() * 256 elements, dense block order
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. NO op-level Capability::DequantizeQ2K exists (gap). out length multiple of 256 validated via group_for_dequantization (debug)."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                            # unpack + per-sub-block scale; not MACs — Judge-measured
  bytes_moved: "(n / 256) * 84 + n * 4"   # read (n/256) super-blocks * 84 B + write n*4 F32 bytes
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact reconstruction over per-16 sub-block scales/mins from f16 d/dmin via to_f32. Only lossy step is the pre-baked 2-bit Q2_K quantization (highest error of the family). Deterministic."

determinism: same_hardware_bitwise
```

---

## to_float_q3k  (Q3_K 3-bit K-quant super-block dequantize → F32)

One-line: Dequantize a Q3_K 3-bit K-quant super-block stream to dense F32; 2-bit + hmask high bit, 6-bit packed scales; contiguous, block order.

Reconstructs a dense F32 buffer from a flat `&[BlockQ3K]` input (`k_quants.rs:1314`). Each
256-element super-block carries a 32-byte hmask (the per-quant high/3rd bit) + 64 bytes of
2-bit-packed quants + 12 bytes of 6-bit-packed sub-block scales + a 2-byte f16 super-scale `d`
(110 bytes/block). The 3rd bit per quant comes from the hmask; the per-16 sub-block scales are
unpacked from the 6-bit field and multiplied by `d` (f16 via `to_f32`), then `ys = scale * (q -
4)`-style centered reconstruction. Output dense F32 length `xs.len() * 256`, block order, validated
by `group_for_dequantization` (`k % 256 == 0`). Same super-block structure as Q2_K with the extra
hmask bit reassembled. Bandwidth-bound: read ~`xs.len() * 110` bytes, write `xs.len() * 256 * 4`.
Contiguous-only.

Dispatch: closest honest tag `QMatMul`; real path `QuantizedType::dequantize` (bundle note). Source
`ggml_dtype=Q3K` (code 11). Op-level capability: **NONE as-built — gap** (no `Capability::DequantizeQ3K`).

FLOPs/bandwidth hint: `flops ≈ 0`; read ~`xs.len() * 110` bytes, write `n * 4`. Marked
`judge_measured`.

```fkc
kernel: to_float_q3k
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "Dequantize a Q3_K 3-bit K-quant super-block stream to dense F32; 2-bit + hmask high bit, 6-bit packed scales; contiguous, block order."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_q3k"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(out.dim[-1], 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q3K, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. NO op-level Capability::DequantizeQ3K exists (gap). out length multiple of 256 validated via group_for_dequantization (debug)."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                            # unpack + per-sub-block scale; not MACs — Judge-measured
  bytes_moved: "(n / 256) * 110 + n * 4"   # read (n/256) super-blocks * 110 B + write n*4 F32 bytes
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact reconstruction over reconstructed Q3_K sub-block scales + hmask high bit; f16 d via to_f32. Only lossy step is the pre-baked 3-bit quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## to_float_q4k  (Q4_K 4-bit K-quant super-block dequantize → F32)

One-line: Dequantize a Q4_K 4-bit K-quant super-block stream to dense F32; 6-bit packed scale/min (get_scale_min_k4); contiguous, block order.

Reconstructs a dense F32 buffer from a flat `&[BlockQ4K]` input (`k_quants.rs:1595`). Each
256-element super-block is a 2-byte f16 super-scale `d`, a 2-byte f16 super-min `dmin`, 12 bytes of
6-bit-packed sub-block scales/mins, and 128 bytes of 4-bit-packed quants (144 bytes/block). This is
the storage dtype behind the GGUF `Q4_K_M` ("medium" mixed-precision K-quant) — its `GgmlDType` is
`Q4K` (code 12); there is **no** `Q4_K_M` `GgmlDType` variant (§3.4). Each sub-block reconstructs
its scale/min from the 6-bit packing via `get_scale_min_k4` (scale = `d * packed_scale`, min =
`dmin * packed_min`), then `ys = scale * nibble - min`. Output dense F32 length `xs.len() * 256`,
block order, validated by `group_for_dequantization` (`k % 256 == 0`). Per-super-block scale/min
reconstruction makes the unpack cost materially higher than the 32-element formats — a clear
`judge_measured` case. Bandwidth-bound: read ~`xs.len() * 144` bytes, write `xs.len() * 256 * 4`.
Contiguous-only.

Dispatch: closest honest tag `QMatMul`; real path `QuantizedType::dequantize` (bundle note). Source
`family=GGML_BLOCK, ggml_dtype=Q4K` (code 12; GGUF `Q4_K_M` → `Q4K`, §3.4). Op-level capability:
`Capability::DequantizeQ4KM` — one of the **three** dequantize tokens that actually exist
(`capability.rs`).

FLOPs/bandwidth hint: `flops ≈ 0`; read ~`xs.len() * 144` bytes, write `n * 4`. Marked
`judge_measured`.

```fkc
kernel: to_float_q4k
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "Dequantize a Q4_K 4-bit K-quant super-block stream to dense F32; 6-bit packed scale/min (get_scale_min_k4); contiguous, block order."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_q4k"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(out.dim[-1], 256)"
      fdx:
        requires_ext: true
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q4K             # GgmlDType variant (code 12); GGUF "Q4_K_M" → Q4K (§3.4)
          granularity: ~             # GGML_BLOCK carries ggml_dtype ONLY; NO PerBlock (PerBlock is MX-only, FDX §6.2/V5)
          role: weight
          scale_operand: ~
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. Op-level Capability::DequantizeQ4KM exists (GGUF Q4_K_M). out length multiple of 256 validated via group_for_dequantization (debug)."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                            # unpack + per-sub-block scale; not MACs — Judge-measured
  bytes_moved: "(n / 256) * 144 + n * 4"   # read (n/256) super-blocks * 144 B + write n*4 F32 bytes
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact reconstruction over per-sub-block scales/mins unpacked via get_scale_min_k4 from f16 d/dmin. Only lossy step is the pre-baked Q4_K_M (4-bit K-quant) quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## to_float_q5k  (Q5_K 5-bit K-quant super-block dequantize → F32)

One-line: Dequantize a Q5_K 5-bit K-quant super-block stream to dense F32; 4-bit + qh high bit, 6-bit packed scale/min; contiguous, block order.

Reconstructs a dense F32 buffer from a flat `&[BlockQ5K]` input (`k_quants.rs:1890`). Each
256-element super-block is a 2-byte f16 `d`, a 2-byte f16 `dmin`, 12 bytes of 6-bit-packed sub-block
scales/mins, a 32-byte hmask (the per-quant 5th bit), and 128 bytes of 4-bit-packed quants
(176 bytes/block). The 5th bit per quant comes from the hmask (`qh`); per-sub-block scale/min are
unpacked from the 6-bit packing (like Q4_K) times `d`/`dmin` (f16 via `to_f32`), then `ys = scale *
val - min`. Output dense F32 length `xs.len() * 256`, block order, validated by
`group_for_dequantization` (`k % 256 == 0`). K-quant analogue of Q4_K with a 5th bit (hmask).
Bandwidth-bound: read ~`xs.len() * 176` bytes, write `xs.len() * 256 * 4`. Contiguous-only.

Dispatch: closest honest tag `QMatMul`; real path `QuantizedType::dequantize` (bundle note). Source
`ggml_dtype=Q5K` (code 13). Op-level capability: **NONE as-built — gap** (no `Capability::DequantizeQ5K`).

FLOPs/bandwidth hint: `flops ≈ 0`; read ~`xs.len() * 176` bytes, write `n * 4`. Marked
`judge_measured`.

```fkc
kernel: to_float_q5k
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "Dequantize a Q5_K 5-bit K-quant super-block stream to dense F32; 4-bit + qh high bit, 6-bit packed scale/min; contiguous, block order."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_q5k"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(out.dim[-1], 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q5K, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. NO op-level Capability::DequantizeQ5K exists (gap). out length multiple of 256 validated via group_for_dequantization (debug)."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                            # unpack + per-sub-block scale; not MACs — Judge-measured
  bytes_moved: "(n / 256) * 176 + n * 4"   # read (n/256) super-blocks * 176 B + write n*4 F32 bytes
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact reconstruction over per-sub-block scales/mins + hmask 5th bit; f16 d/dmin via to_f32. Only lossy step is the pre-baked 5-bit Q5_K quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## to_float_q6k  (Q6_K 6-bit K-quant super-block dequantize → F32)

One-line: Dequantize a Q6_K 6-bit K-quant super-block stream to dense F32; 4-bit ql + 2-bit qh, i8 per-16 scales; contiguous, block order.

Reconstructs a dense F32 buffer from a flat `&[BlockQ6K]` input (`k_quants.rs:2158`). Each
256-element super-block is 128 bytes of 4-bit-packed low quants `ql`, 64 bytes of 2-bit-packed high
bits `qh`, 16 bytes of i8 per-16 sub-block scales, and a 2-byte f16 super-scale `d`
(210 bytes/block). The 6-bit quant per element is assembled from `ql` (low 4 bits) + `qh` (high 2
bits) and centered (`val - 32`); the per-16 i8 sub-block scale times `d` (f16 via `to_f32`) gives
`ys = d * sub_scale * (val - 32)`. Output dense F32 length `xs.len() * 256`, block order, validated
by `group_for_dequantization` (`k % 256 == 0`). Highest-fidelity K-quant of the family (6-bit);
heaviest K-quant traffic. Bandwidth-bound: read ~`xs.len() * 210` bytes, write `xs.len() * 256 * 4`.
Contiguous-only.

Dispatch: closest honest tag `QMatMul`; real path `QuantizedType::dequantize` (bundle note). Source
`ggml_dtype=Q6K` (code 14). Op-level capability: **NONE as-built — gap** (no `Capability::DequantizeQ6K`).

FLOPs/bandwidth hint: `flops ≈ 0`; read ~`xs.len() * 210` bytes, write `n * 4`. Marked
`judge_measured`.

```fkc
kernel: to_float_q6k
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "Dequantize a Q6_K 6-bit K-quant super-block stream to dense F32; 4-bit ql + 2-bit qh, i8 per-16 scales; contiguous, block order."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_q6k"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(out.dim[-1], 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q6K, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. NO op-level Capability::DequantizeQ6K exists (gap). out length multiple of 256 validated via group_for_dequantization (debug)."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                            # unpack + per-sub-block scale; not MACs — Judge-measured
  bytes_moved: "(n / 256) * 210 + n * 4"   # read (n/256) super-blocks * 210 B + write n*4 F32 bytes
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact reconstruction d * i8_sub_scale * (val-32) with val from ql low + qh high; f16 d via to_f32. Only lossy step is the pre-baked 6-bit Q6_K quantization (highest-fidelity K-quant). Deterministic."

determinism: same_hardware_bitwise
```

---

## to_float_q8k  (Q8_K 8-bit K-quant super-block dequantize → F32)

One-line: Dequantize a Q8_K 8-bit K-quant super-block stream to dense F32; i8 * d(f32); contiguous, block order.

Reconstructs `ys[blk*256 + i] = q_i * d` over a flat `&[BlockQ8K]` input (`k_quants.rs:2269`). Each
256-element super-block is a **f32** super-scale `d` (note: f32, not f16 — unique in this family),
256 bytes of i8 quants, and a 16-entry i16 `bsums` field (block sums, used by the dot product, not
by dequant); each signed i8 quant is scaled by `d`. Output dense F32 length `xs.len() * 256`, block
order, validated by `group_for_dequantization` (`k % 256 == 0`). All accumulation/reconstruction in
f32; the f32 scale means no `f16::to_f32` widening on the scale path. Bandwidth-bound: read
~`xs.len() * (4 + 256 + 32)` bytes, write `xs.len() * 256 * 4`. Contiguous-only.

Dispatch: closest honest tag `QMatMul`; real path `QuantizedType::dequantize` (bundle note). Source
`ggml_dtype=Q8K` (code 15). Op-level capability: **NONE as-built — gap** (no `Capability::DequantizeQ8K`).

FLOPs/bandwidth hint: `flops ≈ 0`; read ~`xs.len() * 292` bytes, write `n * 4`. Marked
`judge_measured`.

```fkc
kernel: to_float_q8k
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "Dequantize a Q8_K 8-bit K-quant super-block stream to dense F32; i8 * d(f32); contiguous, block order."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_q8k"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(out.dim[-1], 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8K, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. NO op-level Capability::DequantizeQ8K exists (gap). out length multiple of 256 validated via group_for_dequantization (debug)."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~                            # unpack + scale; not MACs — Judge-measured
  bytes_moved: "(n / 256) * 292 + n * 4"   # read (n/256) super-blocks * 292 B (f32 d + 256 i8 + 32 bsums) + write n*4 F32
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact reconstruction i8*d; super-scale d is f32 (no f16 widen). Only lossy step is the pre-baked 8-bit Q8_K quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## to_float_f32  (F32 "block" identity copy → F32)

One-line: F32 to_float; BLCK_SIZE=1 direct elementwise copy; contiguous, exact identity.

The `f32` `GgmlType` impl (`BLCK_SIZE == 1`, `DIRECT_COPY == true`; `k_quants.rs:2376`) flows
through the same dequant/matmul machinery as the block formats but `to_float` is a plain
**elementwise copy** `ys[i] = xs[i]` over flat `&[f32]` → `&mut [f32]`. Input and output lengths
MUST be equal (debug-asserted). Output dense F32, same element count. Exact identity — no
quantization, no precision loss. Bandwidth-bound: read N×4, write N×4. Contiguous-only.

Dispatch: closest honest tag `QMatMul`; real path `QuantizedType::dequantize` (bundle note). Source
`family=none, dtype=F32` (no block-quant facts; the dispatch-key quant slot is empty). Op-level
capability: **NONE as-built — gap** (no `Capability::DequantizeF32` token; the float-format dequant
is identity/widen with no dedicated capability code).

FLOPs/bandwidth hint: `flops = 0` (pure copy); `bytes_moved = n * (4 + 4)` (read + write F32);
elementwise = bandwidth-bound. Marked `judge_measured` for consistency with the family; the copy
bandwidth is the derivable hint.

```fkc
kernel: to_float_f32
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "F32 to_float; BLCK_SIZE=1 direct elementwise copy; contiguous, exact identity."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_f32"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]                   # true F32, not a packed block; BLCK_SIZE=1, DIRECT_COPY
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      fdx:
        requires_ext: false           # plain F32 — no block-quant meaning
        quant: { family: none, ggml_dtype: ~, granularity: ~, role: ~, scale_operand: ~ }
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # identity: F32 -> F32
      shape_rule: same_as(src)        # element count preserved (BLCK_SIZE=1)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. NO op-level Capability::DequantizeF32 exists (gap); BLCK_SIZE=1 DIRECT_COPY identity."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"                          # pure copy; no arithmetic
  bytes_moved: "n * (4 + 4)"          # read N*4 + write N*4 F32; bandwidth-bound
  overhead_ns: ~                      # launch/call overhead Judge-measured
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact identity copy
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "BLCK_SIZE=1 direct copy ys[i]=xs[i]; exact identity, no quantization; deterministic."

determinism: same_hardware_bitwise
```

---

## to_float_f16  (F16 "block" widen → F32)

One-line: F16 to_float; BLCK_SIZE=1 elementwise widen to F32 (HalfFloatSliceExt); contiguous, lossless.

The `f16` `GgmlType` impl (`BLCK_SIZE == 1`, `DIRECT_COPY == true`; `k_quants.rs:2421`) `to_float`
is an elementwise **widen** `ys[i] = xs[i].to_f32()` over flat `&[f16]` → `&mut [f32]` (via the
`HalfFloatSliceExt` slice convert). Input and output lengths MUST be equal (debug-asserted). Output
dense F32, 2× the input byte size, same element count. Every f16 value (finite, subnormal, inf, NaN)
is exactly representable in f32, so the widen is **lossless / exact** (`max_ulp: 0`). Bandwidth-bound:
read N×2, write N×4. Contiguous-only.

Dispatch: closest honest tag `QMatMul`; real path `QuantizedType::dequantize` (bundle note). Source
`family=none, dtype=F16`. Op-level capability: **NONE as-built — gap** (no `Capability::DequantizeF16`
token; identity/widen has no dedicated capability code).

FLOPs/bandwidth hint: `flops = 0` (convert, no arithmetic); `bytes_moved = n * (2 + 4)`;
bandwidth-bound. Marked `judge_measured`.

```fkc
kernel: to_float_f16
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "F16 to_float; BLCK_SIZE=1 elementwise widen to F32 (HalfFloatSliceExt); contiguous, lossless."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_f16"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F16]                   # true F16, not a packed block; BLCK_SIZE=1, DIRECT_COPY
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      fdx:
        requires_ext: false
        quant: { family: none, ggml_dtype: ~, granularity: ~, role: ~, scale_operand: ~ }
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # widen F16 -> F32
      shape_rule: same_as(src)        # element count preserved (BLCK_SIZE=1)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. NO op-level Capability::DequantizeF16 exists (gap); BLCK_SIZE=1 widen."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"                          # widen-convert; no arithmetic
  bytes_moved: "n * (2 + 4)"          # read N*2 (f16) + write N*4 (f32); bandwidth-bound
  overhead_ns: ~                      # launch/call overhead Judge-measured
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact: f16 strict subset of f32
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "BLCK_SIZE=1 widen ys[i]=xs[i].to_f32(); every f16 value exactly representable in f32; exact, deterministic."

determinism: same_hardware_bitwise
```

---

## to_float_bf16  (BF16 "block" widen → F32)

One-line: BF16 to_float; BLCK_SIZE=1 elementwise widen to F32 (HalfFloatSliceExt); contiguous, lossless.

The `bf16` `GgmlType` impl (`BLCK_SIZE == 1`, `DIRECT_COPY == true`; `k_quants.rs:2466`) `to_float`
is an elementwise **widen** `ys[i] = xs[i].to_f32()` over flat `&[bf16]` → `&mut [f32]` (via the
`HalfFloatSliceExt` slice convert). Input and output lengths MUST be equal (debug-asserted). Output
dense F32, 2× the input byte size, same element count. bf16 shares f32's 8-bit exponent and is a
16-bit truncation of f32, so every bf16 value is exactly representable in f32 — the widen is
**lossless / exact** (`max_ulp: 0`). Bandwidth-bound: read N×2, write N×4. Contiguous-only.

Dispatch: closest honest tag `QMatMul`; real path `QuantizedType::dequantize` (bundle note). Source
`family=none, dtype=BF16`. Op-level capability: **NONE as-built — gap** (no `Capability::DequantizeBF16`
token; identity/widen has no dedicated capability code).

FLOPs/bandwidth hint: `flops = 0`; `bytes_moved = n * (2 + 4)`; bandwidth-bound. Marked
`judge_measured`.

```fkc
kernel: to_float_bf16
op_kind: QMatMul            # closest honest tag; dequant direction has NO dedicated OpKind (bundle note)
blurb: "BF16 to_float; BLCK_SIZE=1 elementwise widen to F32 (HalfFloatSliceExt); contiguous, lossless."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::to_float_bf16"   # QuantizedType::dequantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [BF16]                  # true BF16, not a packed block; BLCK_SIZE=1, DIRECT_COPY
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      fdx:
        requires_ext: false
        quant: { family: none, ggml_dtype: ~, granularity: ~, role: ~, scale_operand: ~ }
  op_params: { variant: QMatMul }     # no Dequantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # widen BF16 -> F32
      shape_rule: same_as(src)        # element count preserved (BLCK_SIZE=1)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8
  notes: "Dispatched via QuantizedType::dequantize, not a keyed OpKind. NO op-level Capability::DequantizeBF16 exists (gap); BLCK_SIZE=1 widen."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"                          # widen-convert; no arithmetic
  bytes_moved: "n * (2 + 4)"          # read N*2 (bf16) + write N*4 (f32); bandwidth-bound
  overhead_ns: ~                      # launch/call overhead Judge-measured
  memory: { device_bytes: 0, host_bytes: "n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0                          # exact: bf16 strict subset of f32
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "BLCK_SIZE=1 widen ys[i]=xs[i].to_f32(); bf16 strict subset of f32 (16-bit truncation widened back); exact, deterministic."

determinism: same_hardware_bitwise
```

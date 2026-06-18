---
fkc_version: 1
provider:
  name: fuel-quantized
  backend: Cpu                       # the numeric impls in this crate are the portable CPU reference
  kernel_source: "fuel-quantized"    # the BindingEntry.kernel_source tag
  link_registry: fuel_quantized::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map (to be defined)
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-quantized — quantize (`from_float` / `from_float_imatrix`) kernel contracts

This bundle contracts the **quantize-direction** numeric kernels of `fuel-quantized`: the
`GgmlType::from_float` (round-trip quantize) and `GgmlType::from_float_imatrix`
(importance-matrix-weighted quantize) implementations in `fuel-quantized/src/k_quants.rs`. They are
the backend-agnostic ggml/gguf block-format quantizers — the numeric body behind
`DynQuantizedStorage::quantize` / `quantize_imatrix` (`fuel-core-types/src/quantized.rs:130,133`).

> **AS-BUILT DISPATCH NOTE — read before trusting `op_kind`.** There is **no `Quantize` /
> `FromFloat` `OpKind`** in the as-built dispatch enum (`fuel-core-types/src/dispatch.rs:52`); the
> only quant-direction `OpKind` is `QMatMul` (line 356). These quantizers are **not** registered as
> standalone primitive kernels on the `KernelBindingTable` at a `(OpKind, dtypes, backend)` key the
> way `binary` is; they reach the dispatch surface through the **`DynQuantizedStorage::quantize` /
> `quantize_imatrix` trait methods** (`fuel-core-types/src/quantized.rs:124-149`), which the CPU
> adapter (`fuel-quantized/src/cpu.rs` `QuantizedType`) forwards to these `from_float*` impls.
> Per the never-invent / never-re-number discipline (§0), the `op_kind:` slot below names the
> **closest honest dispatch tag**, `QMatMul` (the only `OpKind` these block formats participate in),
> and each kernel records — in prose, in `entry_point`, and in `caps.notes` — that the quantize
> direction is dispatched through the `DynQuantizedStorage` trait, not a dedicated key. A future
> `OpKind::Quantize` (or a `DynQuantizedStorage`-trait FKC import surface) would let these register
> as their own keyed kernels; until it lands the `op_kind` slot is the closest faithful tag and the
> trait-method path is authoritative. The `Capability` tokens `QuantizeQ8_0`
> (`fuel-core-types/src/capability.rs:73`) are the only quantize-direction capability codes that
> exist as-built; no broader per-format quantize capability set exists yet.

**Crate-wide layout reality (applies to EVERY kernel here, from the inventory).** Every kernel
operates on flat `&[f32]` input / `&mut [BlockX]` output slices. There is **no `Layout`, no
`Shape`, no `StridedIndex`, no offset, no broadcast** anywhere in this crate; all stride/offset/
broadcast handling lives in the backend adapters, which contiguify before calling in. So the
universal accept precondition is **contiguous, zero-offset, no-broadcast, dense row-major**, and
the universal `awkward_layout_strategy` is `requires_contiguous`. Input element count MUST be an
exact multiple of `BLCK_SIZE` (32 for the legacy `Q4_0..Q8_1`, 256 = `QK_K` for the K-quants);
size checks are `debug_assert!` only (K-quants additionally validate via `group_for_quantization`).
Output is `ys.len() == xs.len() / BLCK_SIZE` blocks, written densely in block order; output dtype is
the block type. **No in-place / no aliasing** (separate input/output buffers). Per-block scales are
stored as `f16` (legacy + Q2K..Q6K) or `f32` (Q8K). The scale is the per-block `d` (and per-block
`m`/sub-scales where the format has them) baked **INLINE** into the output block — the GGML INLINE
single-place rule (§3.9.3, §6): no separate scale operand, no FDX `scale_buffer`.

**GGML quant block carries `ggml_dtype` ONLY (no `scale_granularity`, no `PerBlock`).** Per the
2026-06-18 FDX regime fix (FDX §6.2 family table / V5 `QuantRegimeViolation`), a `family=GGML_BLOCK`
descriptor carries **only** `ggml_dtype` — the baked-scale layout *is* the format. It MUST NOT set
`granularity: PerBlock` (`PerBlock` is now pinned **MX-only**); every GGML `quant` block below leaves
`granularity: ~`. The per-block scale being INLINE is expressed by the INLINE scale placement
(single-place rule), not by a granularity code.

**Cost is `judge_measured` across this bundle.** Quantization is an amax/min scan plus a scale
search (some formats run an iterative RMSE/MSE refine — `make_qx_quants`, `make_qkx*_quants`,
`make_qp_quants`) whose per-element cost is not cleanly derivable from a closed-form FLOPs count
(the refine iteration count and the branchy bit-packing dominate). Per the prompt's COST discipline,
no cost numbers are fabricated: every kernel marks `cost.provenance: judge_measured` (the Judge
bootstraps it). Where a genuinely derivable bandwidth hint exists it is given as a `bytes_moved`
formula only (the quantizer is fundamentally a streaming read of `n` f32 + a dense write of
`n/BLCK_SIZE * type_size` bytes — bandwidth-bound on the read); `flops` is left to the Judge because
the scale-search arithmetic is data- and iteration-dependent.

---

## from_float_q4_0  (quantize f32 → GGML Q4_0 blocks)

One-line blurb: see structured `blurb:`.

Per-block (32 elements) symmetric 4-bit quantizer. For each block: scan for the element of greatest
magnitude (`amax`/`max`), set the block scale `d = max / -8.0` (so the most-negative-or-largest
magnitude maps to the int extreme), `id = 1/d` (0 when `d==0`). Each weight is `q = min(15, (x*id +
8.5) as u8)` — the `+8.5` cast is the round-half-away-from-zero trick; two nibbles pack into one
`u8` (`qs[j] = lo | (hi << 4)`, low half = elements `0..16`, high half = `16..32`). The block stores
`d` as `f16` and 16 packed `u8` nibbles (`type_size` 18 bytes). Numerics: scale math in f32, `d`
narrowed to f16 on store; lossy 4-bit quantization. No imatrix weighting. Source:
`fuel-quantized/src/k_quants.rs:199`. Limitations: contiguous/dense/zero-offset only; `n % 32 == 0`
(debug-asserted, UB/OOB on violation in release); dispatched via `DynQuantizedStorage::quantize`,
not a dedicated `OpKind` (see bundle note).

```fkc
kernel: from_float_q4_0
op_kind: QMatMul            # closest honest tag; quantize direction has NO dedicated OpKind (bundle note)
blurb: "Quantize f32 -> GGML Q4_0 (per-32 symmetric 4-bit, d=max/-8, f16 scale, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ4_0::from_float"   # DynQuantizedStorage::quantize body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # flat &[f32]; no Layout/Shape in this crate
      shape_constraint: "divisible(dim[0], 32)"   # n % BLCK_SIZE == 0 (debug-asserted)
  op_params: { variant: QMatMul }   # no Quantize OpParams variant exists; trait-method dispatch

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)         # opaque block bytes; logical dtype is GgmlDType::Q4_0 (FDX sub_byte)
      shape_rule: from_params(n / 32)   # ys.len() == xs.len() / BLCK_SIZE blocks
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q4_0, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
        # scale d rides INLINE in the block (scale_buffer sidecar = INLINE); NO separate scale_operand (§3.9.3)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Dispatched via DynQuantizedStorage::quantize, not a keyed OpKind. n % 32 == 0 debug-asserted only."

cost:
  provenance: judge_measured        # amax scan + nibble pack; Judge bootstraps (no fabricated numbers)
  class: cheap_elementwise
  flops: ~                          # scale search + pack is data/branch-dependent; Judge-measured
  bytes_moved: "n * 4 + (n / 32) * 18"   # read n f32 (4B) + write n/32 blocks (18B each); bandwidth-bound
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 32) * 18", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic per-block scalar loop
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                      # lossy 4-bit quantize; bound not yet audited (Judge seeds)
  notes: "Lossy 4-bit symmetric quant; scale math f32, d narrowed to f16. Deterministic per-block loop."

determinism: same_hardware_bitwise
```

---

## from_float_q4_1  (quantize f32 → GGML Q4_1 blocks)

Per-block (32) asymmetric 4-bit quantizer with a min offset. For each block: find `min`/`max`,
`d = (max - min) / 15`, `id = 1/d` (0 when `d==0`); each weight `q = min(15, (((x - min)*id) +
0.5) as u8)`, packed two-nibbles-per-byte. The block stores both `d` and `m` (= `min`) as `f16`
plus 16 packed `u8` (`type_size` 20 bytes). Numerics: f32 scale math, `d`/`m` narrowed to f16; lossy
4-bit. No imatrix. Source: `fuel-quantized/src/k_quants.rs:315`. Limitations as the bundle preamble;
`n % 32 == 0`.

```fkc
kernel: from_float_q4_1
op_kind: QMatMul
blurb: "Quantize f32 -> GGML Q4_1 (per-32 asymmetric 4-bit, d=(max-min)/15, f16 d+m, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ4_1::from_float"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 32)"
  op_params: { variant: QMatMul }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)
      shape_rule: from_params(n / 32)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q4_1, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Dispatched via DynQuantizedStorage::quantize, not a keyed OpKind. n % 32 == 0 debug-asserted only."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 32) * 20"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 32) * 20", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Lossy 4-bit asymmetric quant; f32 scale math, d and m narrowed to f16. Deterministic per-block loop."

determinism: same_hardware_bitwise
```

---

## from_float_q5_0  (quantize f32 → GGML Q5_0 blocks)

Per-block (32) symmetric 5-bit quantizer. `amax`/`max` scan, `d = max / -16.0`, `id = 1/d`. Each
weight `q = min(31, (x*id + 16.5) as u8)`; the low 4 bits pack into `qs` nibbles, the 5th (high) bit
of each weight packs into a 4-byte `qh` bitfield. The block stores `d` (f16) + 4-byte `qh` + 16
packed `u8` (`type_size` 22 bytes). Numerics: f32 scale math, `d` to f16; lossy 5-bit. No imatrix.
Source: `fuel-quantized/src/k_quants.rs:424`. `n % 32 == 0`.

```fkc
kernel: from_float_q5_0
op_kind: QMatMul
blurb: "Quantize f32 -> GGML Q5_0 (per-32 symmetric 5-bit, d=max/-16, 5th bit in qh, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ5_0::from_float"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 32)"
  op_params: { variant: QMatMul }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)
      shape_rule: from_params(n / 32)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q5_0, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Dispatched via DynQuantizedStorage::quantize, not a keyed OpKind. n % 32 == 0 debug-asserted only."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 32) * 22"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 32) * 22", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Lossy 5-bit symmetric quant; 5th bit packed into qh; f32 scale math, d to f16. Deterministic per-block."

determinism: same_hardware_bitwise
```

---

## from_float_q5_1  (quantize f32 → GGML Q5_1 blocks)

Per-block (32) asymmetric 5-bit quantizer with a min offset. `min`/`max` scan, `d = (max - min) /
31`, `id = 1/d`; each weight `q = (((x - min)*id) + 0.5) as u8` (low 4 bits to `qs`, 5th bit to a
4-byte `qh`). The block stores `d` and `m` (f16) + 4-byte `qh` + 16 packed `u8` (`type_size` 24
bytes). Numerics: f32 scale math, `d`/`m` to f16; lossy 5-bit. No imatrix. Source:
`fuel-quantized/src/k_quants.rs:535`. `n % 32 == 0`.

```fkc
kernel: from_float_q5_1
op_kind: QMatMul
blurb: "Quantize f32 -> GGML Q5_1 (per-32 asymmetric 5-bit, d=(max-min)/31, 5th bit in qh, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ5_1::from_float"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 32)"
  op_params: { variant: QMatMul }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)
      shape_rule: from_params(n / 32)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q5_1, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Dispatched via DynQuantizedStorage::quantize, not a keyed OpKind. n % 32 == 0 debug-asserted only."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 32) * 24"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 32) * 24", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Lossy 5-bit asymmetric quant; 5th bit in qh; f32 scale math, d and m to f16. Deterministic per-block."

determinism: same_hardware_bitwise
```

---

## from_float_q8_0  (quantize f32 → GGML Q8_0 blocks)

Per-block (32) symmetric 8-bit quantizer. `amax`/`max` scan, `d = amax / 127`, `id = 1/d`; each
weight `q = round(x*id) as i8`. The block stores `d` (f16) + 32 `i8` (`type_size` 34 bytes).
Numerics: f32 scale math, `d` to f16; 8-bit (the least-lossy legacy format). No imatrix. Source:
`fuel-quantized/src/k_quants.rs:629`. `n % 32 == 0`.

```fkc
kernel: from_float_q8_0
op_kind: QMatMul
blurb: "Quantize f32 -> GGML Q8_0 (per-32 symmetric 8-bit, d=amax/127, f16 scale, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ8_0::from_float"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 32)"
  op_params: { variant: QMatMul }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)         # opaque packed GGML block bytes (f16 d + 32 i8 qs interleaved); kDLUInt bits:8 packed-quant stand-in (FDX §3/§13.2). Logical dtype GgmlDType::Q8_0 rides fdx.quant
      shape_rule: from_params(n / 32)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8                # packed block byte stream — honesty base is U8, NOT the i8 quant access width
        quant: { family: GGML_BLOCK, ggml_dtype: Q8_0, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Dispatched via DynQuantizedStorage::quantize; Capability::QuantizeQ8_0 is the as-built quant cap token. n % 32 == 0 debug-asserted only."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 32) * 34"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 32) * 34", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "8-bit symmetric quant (least-lossy legacy); round-to-nearest, f32 scale math, d to f16. Deterministic per-block."

determinism: same_hardware_bitwise
```

---

## from_float_q8_1  (quantize f32 → GGML Q8_1 blocks, stores block sum)

Per-block (32) symmetric 8-bit quantizer that additionally stores the block sum. `amax` scan,
`d = amax / 127`, `id = 1/d`; each weight `q = round(x*id) as i8`, accumulating `sum += q`; the
block stores `d` (f16), `s = f16(sum) * d` (the sum-times-scale used by the Q4_1/Q5_1 dot product),
and 32 `i8` (`type_size` 36 bytes). Q8_1 is primarily the activation-side `VecDotType` for the
asymmetric legacy formats; **it has no `to_float` (dequant) — that path `unimplemented!()`s**.
Numerics: f32 scale math, `d`/`s` to f16. No imatrix. Source: `fuel-quantized/src/k_quants.rs:728`.
`n % 32 == 0`.

```fkc
kernel: from_float_q8_1
op_kind: QMatMul
blurb: "Quantize f32 -> GGML Q8_1 (per-32 8-bit + stored block sum s=sum*d; activation VecDotType, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ8_1::from_float"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 32)"
  op_params: { variant: QMatMul }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)         # opaque packed GGML block bytes (f16 d + f16 s + 32 i8 qs interleaved); kDLUInt bits:8 packed-quant stand-in (FDX §3/§13.2). Logical dtype GgmlDType::Q8_1 rides fdx.quant
      shape_rule: from_params(n / 32)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8                # packed block byte stream — honesty base is U8, NOT the i8 quant access width
        quant: { family: GGML_BLOCK, ggml_dtype: Q8_1, granularity: ~, role: activation }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Activation-side VecDotType for Q4_1/Q5_1; no dequant (to_float unimplemented). Trait-method dispatch. n % 32 == 0 debug-asserted only."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 32) * 36"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 32) * 36", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "8-bit symmetric quant + block-sum s=sum*d; round-to-nearest, f32 scale math, d and s to f16. Deterministic per-block."

determinism: same_hardware_bitwise
```

---

## from_float_q2k  (quantize f32 → GGML Q2K blocks)

Per-super-block (256 = QK_K) 2-bit K-quant. Splits the super-block into 16-element sub-groups; runs
`make_qkx1_quants(3, 5)` per sub-group to find a 2-bit quant plus a per-16 sub-scale `d` and
sub-min `dmin`, then a `Q4SCALE=15` 4-bit re-quantization of the sub-scales/sub-mins. The block
stores the super-block `d`/`dmin` (f16), 16 packed scale bytes, and `QK_K/4` packed 2-bit weights
(`type_size` 84 bytes). Size validated via `group_for_quantization` (debug). Numerics: f32 scale
search, scales narrowed to f16; heavily lossy 2-bit. No imatrix in this entry (see
`from_float_imatrix_q2k`). Source: `fuel-quantized/src/k_quants.rs:836`. `n % 256 == 0`.

```fkc
kernel: from_float_q2k
op_kind: QMatMul
blurb: "Quantize f32 -> GGML Q2K (per-256 2-bit K-quant, make_qkx1_quants(3,5) + Q4SCALE=15, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ2K::from_float"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 256)"   # n % QK_K == 0 (group_for_quantization, debug)
  op_params: { variant: QMatMul }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)
      shape_rule: from_params(n / 256)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q2K, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "K-quant super-block of 256; trait-method dispatch. n % 256 == 0 validated via group_for_quantization (debug)."

cost:
  provenance: judge_measured        # make_qkx1_quants scale search dominates; Judge bootstraps
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 256) * 84"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 256) * 84", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Lossy 2-bit K-quant via make_qkx1_quants(3,5) + Q4SCALE=15; f32 scale search, scales to f16. Deterministic."

determinism: same_hardware_bitwise
```

---

## from_float_q3k  (quantize f32 → GGML Q3K blocks)

Per-super-block (256) 3-bit K-quant. Runs `make_q3_quants(., 4, rmse)` per sub-group (RMSE-refined
scale search) to pick the 3-bit quant; the high bit of each 3-bit weight packs into an `hmask`
bitfield, the low 2 bits into the 2-bit packed `qs`; per-16 6-bit scales pack into 12 scale bytes.
The block stores `d` (f16), 12 scale bytes, `QK_K/8` hmask bits, `QK_K/4` 2-bit weights (`type_size`
110 bytes). Numerics: f32 RMSE scale search, `d` to f16; lossy 3-bit. No imatrix in this entry (see
`from_float_imatrix_q3k`). Source: `fuel-quantized/src/k_quants.rs:1135`. `n % 256 == 0`.

```fkc
kernel: from_float_q3k
op_kind: QMatMul
blurb: "Quantize f32 -> GGML Q3K (per-256 3-bit K-quant, make_q3_quants RMSE, hmask high bit, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ3K::from_float"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 256)"
  op_params: { variant: QMatMul }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)
      shape_rule: from_params(n / 256)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q3K, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "K-quant super-block of 256; trait-method dispatch. n % 256 == 0 validated via group_for_quantization (debug)."

cost:
  provenance: judge_measured        # make_q3_quants RMSE iteration dominates; Judge bootstraps
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 256) * 110"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 256) * 110", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Lossy 3-bit K-quant via make_q3_quants RMSE; 6-bit packed scales, hmask high bit; f32 search, d to f16. Deterministic."

determinism: same_hardware_bitwise
```

---

## from_float_q4k  (quantize f32 → GGML Q4K blocks)

Per-super-block (256) 4-bit K-quant. Runs `make_qkx1_quants(15, 5)` per 32-element sub-group to find
the 4-bit quant plus per-sub-group 6-bit scale and 6-bit min, packed via the `get_scale_min_k4`
6-bit layout into 12 scale bytes. The block stores super-block `d`/`dmin` (f16), 12 scale/min bytes,
`QK_K/2` packed 4-bit weights (`type_size` 144 bytes). This is the storage dtype for the GGUF
`Q4_K_M` mixed-precision weight (`GgmlDType::Q4K`, code 12; op-level `Capability::MatMulQ4KM`/
`DequantizeQ4KM`). Numerics: f32 scale search, scales to f16; lossy 4-bit. No imatrix in this entry
(see `from_float_imatrix_q4k`). Source: `fuel-quantized/src/k_quants.rs:1470`. `n % 256 == 0`.

```fkc
kernel: from_float_q4k
op_kind: QMatMul
blurb: "Quantize f32 -> GGML Q4K (per-256 4-bit K-quant, make_qkx1_quants(15,5), 6-bit scale/min, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ4K::from_float"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 256)"
  op_params: { variant: QMatMul }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)
      shape_rule: from_params(n / 256)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q4K, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Storage dtype for GGUF Q4_K_M (GgmlDType::Q4K code 12); trait-method dispatch. n % 256 == 0 validated (debug)."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 256) * 144"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 256) * 144", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Lossy 4-bit K-quant via make_qkx1_quants(15,5); 6-bit packed scale/min; f32 search, scales to f16. Deterministic."

determinism: same_hardware_bitwise
```

---

## from_float_q5k  (quantize f32 → GGML Q5K blocks)

Per-super-block (256) 5-bit K-quant. Runs `make_qkx1_quants(31, 5)` per 32-element sub-group; the
low 4 bits pack into `qs`, the 5th (high) bit of each weight packs into a `qh` bitfield; per-sub
6-bit scale/min pack into 12 scale bytes. The block stores `d`/`dmin` (f16), 12 scale/min bytes,
`QK_K/8` qh bits, `QK_K/2` 4-bit low weights (`type_size` 176 bytes). Numerics: f32 scale search,
scales to f16; lossy 5-bit. No imatrix in this entry (see `from_float_imatrix_q5k`). Source:
`fuel-quantized/src/k_quants.rs:1732`. `n % 256 == 0`.

```fkc
kernel: from_float_q5k
op_kind: QMatMul
blurb: "Quantize f32 -> GGML Q5K (per-256 5-bit K-quant, make_qkx1_quants(31,5), 5th bit in qh, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ5K::from_float"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 256)"
  op_params: { variant: QMatMul }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)
      shape_rule: from_params(n / 256)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q5K, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "K-quant super-block of 256; trait-method dispatch. n % 256 == 0 validated via group_for_quantization (debug)."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 256) * 176"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 256) * 176", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Lossy 5-bit K-quant via make_qkx1_quants(31,5); 5th bit in qh, 6-bit scale/min; f32 search, scales to f16. Deterministic."

determinism: same_hardware_bitwise
```

---

## from_float_q6k  (quantize f32 → GGML Q6K blocks)

Per-super-block (256) 6-bit K-quant. Runs `make_qx_quants(16, 32, rmse=1)` per sub-group (RMSE
search); the low 4 bits pack into `ql`, the high 2 bits into `qh`, with per-16 `i8` scales. The
block stores `QK_K/2` ql + `QK_K/4` qh + `QK_K/16` i8 scales + super-block `d` (f16) (`type_size`
210 bytes). **Implementation uses raw pointers (`unsafe`) directly.** Numerics: f32 RMSE scale
search, `d` to f16; lossy 6-bit (the least-lossy K-quant). No imatrix in this entry (see
`from_float_imatrix_q6k`). Source: `fuel-quantized/src/k_quants.rs:2005`. `n % 256 == 0`.

```fkc
kernel: from_float_q6k
op_kind: QMatMul
blurb: "Quantize f32 -> GGML Q6K (per-256 6-bit K-quant, make_qx_quants(16,32,rmse=1), i8 per-16 scales, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ6K::from_float"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 256)"
  op_params: { variant: QMatMul }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)
      shape_rule: from_params(n / 256)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q6K, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Impl uses unsafe raw pointers; least-lossy K-quant; trait-method dispatch. n % 256 == 0 validated (debug)."

cost:
  provenance: judge_measured        # make_qx_quants RMSE iteration dominates; Judge bootstraps
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 256) * 210"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 256) * 210", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Lossy 6-bit K-quant via make_qx_quants(16,32,rmse=1); ql/qh split, i8 per-16 scales; f32 search, d to f16. Deterministic."

determinism: same_hardware_bitwise
```

---

## from_float_q8k  (quantize f32 → GGML Q8K blocks)

Per-super-block (256) 8-bit K-quant — the activation-side `VecDotType` for every K-quant dot. Per
super-block: `amax`/`max` scan; if `amax==0` the block is zeroed, else `iscale = -128 / max`, each
weight `q = min(127, round(iscale*x)) as i8`, the block scale `d = 1 / iscale` is stored as **`f32`**
(unlike the f16-scale formats), and per-16 sums fill the `bsums` (`i16`) field. The block stores `d`
(f32), `QK_K` i8 weights, and `QK_K/16` i16 bsums (`type_size` 292 bytes). Numerics: f32 scale math,
**f32 scale stored** (most accurate K-quant scale); 8-bit. No imatrix (the K-quant imatrix path
quantizes weights, not the Q8K activation). Source: `fuel-quantized/src/k_quants.rs:2230`.
`n % 256 == 0`.

```fkc
kernel: from_float_q8k
op_kind: QMatMul
blurb: "Quantize f32 -> GGML Q8K (per-256 8-bit K-quant activation VecDotType, f32 scale + i16 bsums, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ8K::from_float"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 256)"
  op_params: { variant: QMatMul }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)         # opaque packed GGML block bytes (f32 d + 256 i8 qs + 16 i16 bsums interleaved); kDLUInt bits:8 packed-quant stand-in (FDX §3/§13.2). Logical dtype GgmlDType::Q8K rides fdx.quant
      shape_rule: from_params(n / 256)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8                # packed block byte stream — honesty base is U8, NOT the i8 quant access width
        quant: { family: GGML_BLOCK, ggml_dtype: Q8K, granularity: ~, role: activation }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Activation VecDotType for all K-quant dots; f32 (not f16) block scale; trait-method dispatch. n % 256 == 0 validated (debug)."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 256) * 292"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 256) * 292", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "8-bit K-quant activation; round-to-nearest, f32 scale math AND f32 scale stored (most accurate); per-16 i16 bsums. Deterministic."

determinism: same_hardware_bitwise
```

---

## from_float_imatrix_q2k  (importance-matrix quantize f32 → GGML Q2K)

Per-super-block (256) 2-bit K-quant with importance-matrix weighting. Same Q2K storage layout as
`from_float_q2k`, but the scale search uses `make_qkx3_quants` + `make_qp_quants` weighted by
`imatrix_weights` (with `sigma2 = Σx²/QK_K` as the diagonal-fallback weight); the imatrix row is
selected by `sblk_idx % (n_per_row / QK_K)`, indexed `imatrix_weights[imatrix_row*QK_K + 16*j + l]`.
Extra params beyond `from_float`: `imatrix_weights: &[f32]`, `n_per_row: usize`. Numerics: f32
weighted scale search, scales to f16; lossy 2-bit, lower error than unweighted for important
columns. Source: `fuel-quantized/src/k_quants.rs:900`. `n % 256 == 0`; dispatched via
`DynQuantizedStorage::quantize_imatrix`.

```fkc
kernel: from_float_imatrix_q2k
op_kind: QMatMul
blurb: "Imatrix quantize f32 -> GGML Q2K (per-256 2-bit, make_qkx3_quants + make_qp_quants weighted, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ2K::from_float_imatrix"   # DynQuantizedStorage::quantize_imatrix body
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 256)"
    - name: imatrix_weights
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # row-major importance matrix; row = sblk_idx % (n_per_row / QK_K)
  op_params:
    variant: QMatMul              # no Quantize OpParams variant; n_per_row rides the trait call
    fields:
      n_per_row: { kind: usize, note: "imatrix row pitch; row = sblk_idx % (n_per_row / 256)" }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)
      shape_rule: from_params(n / 256)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q2K, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Imatrix-weighted; dispatched via DynQuantizedStorage::quantize_imatrix. n % 256 == 0 validated (debug)."

cost:
  provenance: judge_measured        # weighted make_qkx3/make_qp scale search; Judge bootstraps
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 256) * 84"   # imatrix re-read not counted (reused row); Judge refines
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 256) * 84", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Lossy 2-bit imatrix K-quant via make_qkx3_quants + make_qp_quants (sigma2=Σx²/QK_K); f32 weighted search, scales to f16. Deterministic."

determinism: same_hardware_bitwise
```

---

## from_float_imatrix_q3k  (importance-matrix quantize f32 → GGML Q3K)

Per-super-block (256) 3-bit K-quant with importance-matrix weighting. Same Q3K storage layout as
`from_float_q3k`, but the scale search uses `make_qx_quants` (the **unsafe** raw-ptr RMSE search)
weighted by `imatrix_weights` with `sigma2 = 2·Σx²/QK_K`; imatrix row = `sblk_idx % (n_per_row /
QK_K)`, indexed `imatrix_weights[imatrix_row*QK_K + 16*j + l]`. Extra params: `imatrix_weights:
&[f32]`, `n_per_row: usize`. Numerics: f32 weighted RMSE search, `d` to f16; lossy 3-bit. Source:
`fuel-quantized/src/k_quants.rs:1216`. `n % 256 == 0`; via `DynQuantizedStorage::quantize_imatrix`.

```fkc
kernel: from_float_imatrix_q3k
op_kind: QMatMul
blurb: "Imatrix quantize f32 -> GGML Q3K (per-256 3-bit, make_qx_quants weighted RMSE, sigma2=2*sumx2/QK_K, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ3K::from_float_imatrix"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 256)"
    - name: imatrix_weights
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
  op_params:
    variant: QMatMul
    fields:
      n_per_row: { kind: usize, note: "imatrix row pitch; row = sblk_idx % (n_per_row / 256)" }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)
      shape_rule: from_params(n / 256)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q3K, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Imatrix-weighted; make_qx_quants uses unsafe raw ptrs; via DynQuantizedStorage::quantize_imatrix. n % 256 == 0 validated (debug)."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 256) * 110"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 256) * 110", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Lossy 3-bit imatrix K-quant via make_qx_quants weighted RMSE (sigma2=2*Σx²/QK_K); f32 search, d to f16. Deterministic."

determinism: same_hardware_bitwise
```

---

## from_float_imatrix_q4k  (importance-matrix quantize f32 → GGML Q4K)

Per-super-block (256) 4-bit K-quant with importance-matrix weighting. Same Q4K storage layout as
`from_float_q4k`, but the scale search uses `make_qkx3_quants` + `make_qp_quants` weighted by
`imatrix_weights`; imatrix row = `sblk_idx % (n_per_row / QK_K)`, indexed `imatrix_weights[
imatrix_row*QK_K + 32*j + l]` (32-element sub-groups). Extra params: `imatrix_weights: &[f32]`,
`n_per_row: usize`. Produces the imatrix-weighted GGUF `Q4_K_M` weight. Numerics: f32 weighted scale
search, scales to f16; lossy 4-bit. Source: `fuel-quantized/src/k_quants.rs:1530`. `n % 256 == 0`;
via `DynQuantizedStorage::quantize_imatrix`.

```fkc
kernel: from_float_imatrix_q4k
op_kind: QMatMul
blurb: "Imatrix quantize f32 -> GGML Q4K (per-256 4-bit, make_qkx3_quants + make_qp_quants weighted, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ4K::from_float_imatrix"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 256)"
    - name: imatrix_weights
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
  op_params:
    variant: QMatMul
    fields:
      n_per_row: { kind: usize, note: "imatrix row pitch; row = sblk_idx % (n_per_row / 256)" }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)
      shape_rule: from_params(n / 256)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q4K, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Imatrix-weighted GGUF Q4_K_M weight; via DynQuantizedStorage::quantize_imatrix. n % 256 == 0 validated (debug)."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 256) * 144"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 256) * 144", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Lossy 4-bit imatrix K-quant via make_qkx3_quants + make_qp_quants; f32 weighted search, scales to f16. Deterministic."

determinism: same_hardware_bitwise
```

---

## from_float_imatrix_q5k  (importance-matrix quantize f32 → GGML Q5K)

Per-super-block (256) 5-bit K-quant with importance-matrix weighting. Same Q5K storage layout as
`from_float_q5k`, but the scale search uses `make_qkx3_quants` + `make_qp_quants` weighted by
`imatrix_weights`; imatrix row = `sblk_idx % (n_per_row / QK_K)`, indexed `imatrix_weights[
imatrix_row*QK_K + 32*j + l]`. Extra params: `imatrix_weights: &[f32]`, `n_per_row: usize`.
Numerics: f32 weighted scale search, scales to f16; lossy 5-bit. Source:
`fuel-quantized/src/k_quants.rs:1807`. `n % 256 == 0`; via `DynQuantizedStorage::quantize_imatrix`.

```fkc
kernel: from_float_imatrix_q5k
op_kind: QMatMul
blurb: "Imatrix quantize f32 -> GGML Q5K (per-256 5-bit, make_qkx3_quants + make_qp_quants weighted, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ5K::from_float_imatrix"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 256)"
    - name: imatrix_weights
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
  op_params:
    variant: QMatMul
    fields:
      n_per_row: { kind: usize, note: "imatrix row pitch; row = sblk_idx % (n_per_row / 256)" }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)
      shape_rule: from_params(n / 256)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q5K, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Imatrix-weighted; via DynQuantizedStorage::quantize_imatrix. n % 256 == 0 validated (debug)."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 256) * 176"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 256) * 176", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Lossy 5-bit imatrix K-quant via make_qkx3_quants + make_qp_quants; f32 weighted search, scales to f16. Deterministic."

determinism: same_hardware_bitwise
```

---

## from_float_imatrix_q6k  (importance-matrix quantize f32 → GGML Q6K)

Per-super-block (256) 6-bit K-quant with importance-matrix weighting. Same Q6K storage layout as
`from_float_q6k`, but the scale search uses `make_qx_quants` (the **unsafe** raw-ptr RMSE search)
with a per-16 imatrix row; the impl operates through raw pointers
(`imatrix_weights.add(QK_K*imatrix_row + 16*ib)`), imatrix row = `sblk_idx % (n_per_row / QK_K)`.
Extra params: `imatrix_weights: &[f32]`, `n_per_row: usize`. Numerics: f32 weighted RMSE search, `d`
to f16; lossy 6-bit. Source: `fuel-quantized/src/k_quants.rs:2077`. `n % 256 == 0`; via
`DynQuantizedStorage::quantize_imatrix`.

```fkc
kernel: from_float_imatrix_q6k
op_kind: QMatMul
blurb: "Imatrix quantize f32 -> GGML Q6K (per-256 6-bit, make_qx_quants weighted RMSE per-16 imatrix row, INLINE)."
backend: Cpu
kernel_source: "fuel-quantized"
entry_point: "fuel_quantized::k_quants::BlockQ6K::from_float_imatrix"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "divisible(dim[0], 256)"
    - name: imatrix_weights
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
  op_params:
    variant: QMatMul
    fields:
      n_per_row: { kind: usize, note: "imatrix row pitch; row = sblk_idx % (n_per_row / 256)" }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(U8)
      shape_rule: from_params(n / 256)
      layout_guarantee: contiguous
      aliasing: none
      fdx:
        sub_byte: U8
        quant: { family: GGML_BLOCK, ggml_dtype: Q6K, granularity: ~, role: weight }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32
  notes: "Imatrix-weighted; make_qx_quants + the impl body use unsafe raw ptrs; via DynQuantizedStorage::quantize_imatrix. n % 256 == 0 validated (debug)."

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: ~
  bytes_moved: "n * 4 + (n / 256) * 210"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n / 256) * 210", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Lossy 6-bit imatrix K-quant via make_qx_quants weighted RMSE (per-16 imatrix row); f32 search, d to f16. Deterministic."

determinism: same_hardware_bitwise
```

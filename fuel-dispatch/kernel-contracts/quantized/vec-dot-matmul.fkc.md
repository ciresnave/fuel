---
fkc_version: 1
provider:
  name: fuel-quantized
  backend: Cpu                       # maps to BackendId::Cpu
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag (scalar reference path)
  link_registry: fuel_quantized::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-quantized — vec_dot / matmul kernel contracts

Backend-agnostic ggml/gguf block-format GEMM numerics for the CPU backend. These contracts cover
the `GgmlType::vec_dot` building blocks (one quantized dot of one weight row against one activation
row) and the two `matmul` / `matmul_f16` drivers that turn `vec_dot` into a quantized GEMM.

**Crate-wide layout reality (applies to EVERY kernel in this file).** Every kernel operates on flat
`&[T]` / `&[f32]` / `&[f16]` slices. There is NO `Layout`, NO `Shape`, NO `StridedIndex`, NO offset,
NO broadcast anywhere in `fuel-quantized`. All stride/offset/broadcast handling lives in the backend
adapters (`fuel-cpu-backend`, `fuel-core/src/quantized/`) which contiguify before calling in. The
universal precondition is therefore **contiguous, zero-offset, no-broadcast, dense row-major** — so
every contract below declares `layout: { contiguous: required, strided: rejected, broadcast_stride0:
rejected, start_offset: rejected, reverse_strides: rejected }` and `awkward_layout_strategy:
requires_contiguous`. Size validation is `debug_assert!` only — in release a wrong length is UB /
OOB, not a returned error.

**Scale single-place (§3.9.3).** GGML block scales are sidecar-bundled (INLINE) inside the block
struct (f16 `d`/`m`/`dmin` for legacy + Q2K..Q6K, f32 `d` for Q8K). They ride the FDX tensor's
`FDXQuant.scale_buffer` (placement INLINE) and have **no** separate FKC scale operand —
`fdx.quant.scale_operand` stays `~` everywhere here, satisfying the single-place rule.

**Op-level dispatch.** The `vec_dot_*` and `matmul*` numeric kernels compose into the op-level
quantized GEMM (`OpKind::QMatMul`, `OpParams::QMatMul { quant_type, batch_count, m, n, k }`) for the
quantized formats, and `OpKind::MatMul` / `OpParams::Matmul` for the dense float "block" formats
(`f32`/`f16`/`bf16`, `BLCK_SIZE = 1`, `DIRECT_COPY = true`). The per-format `QuantType` token
distinguishes the quant key (`QuantType::Q4_0 … Q6K`, with the GGUF `Q4_K_M` → `GgmlDType::Q4K`
storage mapping, §3.4; note `QuantType` exposes `Q4_K_M` as its variant name for the Q4K medium
format and carries **no** `Q8K` variant — Q8K exists only as a `VecDotType`). All accumulation is in
f32 regardless of stored scale precision.

---

## vec_dot_q4_0  (Q4×Q8_0 row dot)

One quantized dot product of a Q4_0 weight row against a Q8_0-quantized activation row, returning an
f32 scalar. `fn vec_dot(n, xs: &[BlockQ4_0], ys: &[BlockQ8_0]) -> f32` (`k_quants.rs:252`, scalar
reference; avx2 SIMD at `avx.rs:51`, also neon/simd128). `n` is the element count; both slices hold
`n / 32` blocks (`BLCK_SIZE = QK4_0 = 32`). The inner loop unpacks each 4-bit nibble to a signed
`(nibble − 8)` integer, multiplies it against the i8 Q8_0 lane, accumulates the per-block integer dot
in i32, then scales by `d_x · d_y` in f32 where both `d` are f16 read via `f16::to_f32`. The
activation right operand `ys` is pre-quantized to `Q8_0` (`VecDotType = Q8_0`). `n` must be a
multiple of 32 (debug-asserted). Building block of the `matmul` driver, never dispatched standalone;
contiguous flat slices only.

```fkc
kernel: vec_dot_q4_0
op_kind: QMatMul
blurb: "Q4×Q8_0 row dot: (nibble-8)·i8 i32-accumulate, ×(d_x·d_y) f32; right operand pre-quantized to Q8_0."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::BlockQ4_0::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[BlockQ4_0]  (n/32 blocks)
      dtypes: [U8]                  # opaque block bytes; logical dtype carried in fdx.quant
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "divisible(dim[-1], 32)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q4_0, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
    - name: activation              # ys: &[BlockQ8_0]  pre-quantized to VecDotType
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8_0, granularity: ~, role: activation, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params:
    variant: QMatMul                # OpParams::QMatMul { quant_type, batch_count, m, n, k }
    fields:
      quant_type:  { kind: QuantType, constraint: "== Q4_0" }
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "k % 32 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)        # vec_dot returns f32 regardless of block scale precision
      shape_rule: from_params(out)  # scalar per (row,col); GEMM shape from QMatMul m,n
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dtype == U8", note: "avx2/neon/simd128 SIMD path when target_feature enabled; scalar otherwise" }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 8

cost:
  provenance: judge_measured        # Judge bootstraps; flops/bandwidth hint below is the only derivable formula
  class: gemm_like
  flops: "2 * k"                    # per row-dot: one i32 madd + one f32 fmadd per element
  bytes_moved: "(18 + 34) * (k / 32)"   # 18 B/Q4_0 block read + 34 B/Q8_0 block read
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # integer inner dot + deterministic f32 fmadd order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "i8/i32 integer inner dot, then ×(d_x·d_y) f32; legacy d is f16 via f16::to_f32. SIMD and scalar are the same logical kernel."

determinism: same_hardware_bitwise
```

---

## vec_dot_q4_1  (Q4×Q8_1 row dot)

One quantized dot of a Q4_1 weight row against a Q8_1-quantized activation row. `fn vec_dot(n, xs:
&[BlockQ4_1], ys: &[BlockQ8_1]) -> f32` (`k_quants.rs:285`; avx2 `avx.rs:148`). Q4_1 stores a per-
block scale `d` and minimum `m` (both f16); the nibble is used directly (no −8 bias) so the
dequantized contribution is `nibble · d + m`. The integer dot accumulates the raw nibble × i8 lane,
and the f16 `d`/`m` plus the Q8_1 block sum `s` combine in f32. `VecDotType = Q8_1`. `n` must be a
multiple of 32, **and a multiple of 2 blocks** (debug-asserted). Contiguous flat slices only.

```fkc
kernel: vec_dot_q4_1
op_kind: QMatMul
blurb: "Q4×Q8_1 row dot: nibble·d + m via f16 d/m and Q8_1 block sum; right operand pre-quantized to Q8_1."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::BlockQ4_1::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[BlockQ4_1]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "divisible(dim[-1], 64)"   # 2-block (64-elem) multiple required
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q4_1, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
    - name: activation              # ys: &[BlockQ8_1]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8_1, granularity: ~, role: activation, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params:
    variant: QMatMul
    fields:
      quant_type:  { kind: QuantType, constraint: "== Q4_1" }
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "k % 64 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dtype == U8", note: "avx2 SIMD path when target_feature enabled; scalar otherwise" }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"
  bytes_moved: "(20 + 36) * (k / 32)"   # 20 B/Q4_1 block + 36 B/Q8_1 block
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "integer inner dot then ×d + ×m in f32 with Q8_1 block sum s; f16 d/m. Requires n a multiple of 2 blocks."

determinism: same_hardware_bitwise
```

---

## vec_dot_q5_0  (Q5×Q8_0 row dot)

One quantized dot of a Q5_0 weight row against a Q8_0-quantized activation row. `fn vec_dot(n, xs:
&[BlockQ5_0], ys: &[BlockQ8_0]) -> f32` (`k_quants.rs:401`; avx2 `avx.rs:176`). Q5_0 reconstructs a
5-bit value by combining the 4-bit nibble with a 5th high bit unpacked from `qh`, then biases by −16,
so the contribution is `(val − 16) · d`. The avx path uses the local `bytes_from_bits_32_fifth`
helper (`avx.rs:94`) to expand the high-bit plane. Integer i32 inner dot, f16 `d` scale, f32
accumulate. `VecDotType = Q8_0`. `n` must be a multiple of 32 **and a multiple of 2 blocks**
(debug-asserted). Contiguous flat slices only.

```fkc
kernel: vec_dot_q5_0
op_kind: QMatMul
blurb: "Q5×Q8_0 row dot: 5th bit from qh, (val-16)·d; right operand pre-quantized to Q8_0."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::BlockQ5_0::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[BlockQ5_0]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "divisible(dim[-1], 64)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q5_0, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
    - name: activation              # ys: &[BlockQ8_0]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8_0, granularity: ~, role: activation, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params:
    variant: QMatMul
    fields:
      quant_type:  { kind: QuantType, constraint: "== Q5_0" }
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "k % 64 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dtype == U8", note: "avx2/neon/simd128 SIMD path when target_feature enabled; scalar otherwise" }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"
  bytes_moved: "(22 + 34) * (k / 32)"   # 22 B/Q5_0 block (2 d + 4 qh + 16 qs) + 34 B/Q8_0 block
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "5th bit unpacked from qh, (val-16)·d; i32 integer inner dot, f16 d, f32 accumulate. n a multiple of 2 blocks."

determinism: same_hardware_bitwise
```

---

## vec_dot_q5_1  (Q5×Q8_1 row dot)

One quantized dot of a Q5_1 weight row against a Q8_1-quantized activation row. `fn vec_dot(n, xs:
&[BlockQ5_1], ys: &[BlockQ8_1]) -> f32` (`k_quants.rs:501`; avx2 `avx.rs:212`). Q5_1 combines the
4-bit nibble with the 5th bit from `qh` to form a 5-bit value used directly (no −16 bias), with a
per-block scale `d` and minimum `m` (both f16): contribution `val · d + m`. Uses the same
`bytes_from_bits_32_fifth` avx helper as Q5_0. `VecDotType = Q8_1`. `n` must be a multiple of 32
**and a multiple of 2 blocks** (debug-asserted). Contiguous flat slices only.

```fkc
kernel: vec_dot_q5_1
op_kind: QMatMul
blurb: "Q5×Q8_1 row dot: 5th bit from qh, val·d + m; right operand pre-quantized to Q8_1."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::BlockQ5_1::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[BlockQ5_1]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "divisible(dim[-1], 64)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q5_1, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
    - name: activation              # ys: &[BlockQ8_1]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8_1, granularity: ~, role: activation, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params:
    variant: QMatMul
    fields:
      quant_type:  { kind: QuantType, constraint: "== Q5_1" }
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "k % 64 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dtype == U8", note: "avx2 SIMD path when target_feature enabled; scalar otherwise" }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"
  bytes_moved: "(24 + 36) * (k / 32)"   # 24 B/Q5_1 block (2 d + 2 m + 4 qh + 16 qs) + 36 B/Q8_1 block
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "5th bit from qh, val·d + m; i32 integer inner dot, f16 d/m, f32 accumulate with Q8_1 block sum. n a multiple of 2 blocks."

determinism: same_hardware_bitwise
```

---

## vec_dot_q8_0  (Q8×Q8_0 row dot)

One quantized dot of a Q8_0 weight row against a Q8_0-quantized activation row. `fn vec_dot(n, xs:
&[BlockQ8_0], ys: &[BlockQ8_0]) -> f32` (`k_quants.rs:674`; avx2 `avx.rs:72`, also neon/simd128).
Both operands are i8 quants with an f16 per-block scale `d`: the i8×i8 lane products accumulate in
i32 per block, then scale by `d_x · d_y` in f32. `VecDotType = Q8_0`. `n` must be a multiple of 32
(debug-asserted). Contiguous flat slices only.

```fkc
kernel: vec_dot_q8_0
op_kind: QMatMul
blurb: "Q8×Q8_0 row dot: i8·i8 i32-accumulate, ×(d_x·d_y) f32; right operand pre-quantized to Q8_0."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::BlockQ8_0::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[BlockQ8_0]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "divisible(dim[-1], 32)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8_0, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
    - name: activation              # ys: &[BlockQ8_0]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8_0, granularity: ~, role: activation, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params:
    variant: QMatMul
    fields:
      quant_type:  { kind: QuantType, constraint: "== Q8_0" }
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "k % 32 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dtype == U8", note: "avx2/neon/simd128 SIMD path when target_feature enabled; scalar otherwise" }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"
  bytes_moved: "(34 + 34) * (k / 32)"   # 34 B/Q8_0 block (2 d + 32 qs) on each side
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "i8·i8 i32-accumulate per block then ×(d_x·d_y) f32; f16 d. SIMD and scalar are the same logical kernel."

determinism: same_hardware_bitwise
```

---

## vec_dot_q8_1  (Q8×Q8_1 row dot)

One quantized dot of a Q8_1 weight row against a Q8_1-quantized activation row. `fn vec_dot(n, xs:
&[BlockQ8_1], ys: &[BlockQ8_1]) -> f32` (`k_quants.rs:708`; avx2 `avx.rs:127`). Q8_1 carries an i8
quant array plus a per-block scale `d` and the precomputed sum `s = Σq · d` (both f16). The dot
combines the i8×i8 integer inner product scaled by `d_x · d_y` with the `s` cross terms in f32.
`VecDotType = Q8_1`. `n` must be a multiple of 32 (debug-asserted). Note: Q8_1 has **no** `to_float`
(dequantize) impl — its `to_float` is `unimplemented!()` (`k_quants.rs:759`); Q8_1 exists only as a
`VecDotType` / matmul building block. Contiguous flat slices only.

```fkc
kernel: vec_dot_q8_1
op_kind: QMatMul
blurb: "Q8×Q8_1 row dot: i8·i8 i32-accumulate ×(d_x·d_y) plus block-sum s in f32; right operand Q8_1."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::BlockQ8_1::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[BlockQ8_1]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "divisible(dim[-1], 32)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8_1, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
    - name: activation              # ys: &[BlockQ8_1]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8_1, granularity: ~, role: activation, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params:
    variant: QMatMul
    fields:
      quant_type:  { kind: QuantType, constraint: "== Q8_1" }
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "k % 32 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dtype == U8", note: "avx2 SIMD path when target_feature enabled; scalar otherwise" }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"
  bytes_moved: "(36 + 36) * (k / 32)"   # 36 B/Q8_1 block (2 d + 2 s + 32 qs) on each side
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "i8·i8 i32-accumulate ×(d_x·d_y) plus block-sum s cross terms in f32; f16 d/s. No dequantize path exists (to_float unimplemented)."

determinism: same_hardware_bitwise
```

---

## vec_dot_q2k  (Q2_K×Q8_K row dot)

One quantized dot of a Q2_K weight row against a Q8_K-quantized activation row. `fn vec_dot(n, xs:
&[BlockQ2K], ys: &[BlockQ8K]) -> f32` (`k_quants.rs:783`; avx2 `avx.rs:380`, also neon/simd128). Q2_K
is a 256-element super-block (`QK_K = 256 = BLCK_SIZE`) of 2-bit quants with per-16 sub-scales packed
against a block-level `d`/`dmin` (both f16). The dot unpacks the 2-bit lanes, applies the per-16
sub-scale and min, multiplies against the i8 Q8_K activation, and accumulates in f32 using the Q8_K
block scale `d` (f32) and per-16 `bsums`. `VecDotType = Q8K`. `n` must be a multiple of 256
(debug-asserted). Contiguous flat slices only.

```fkc
kernel: vec_dot_q2k
op_kind: QMatMul
blurb: "Q2_K×Q8_K super-block row dot: 2-bit quants with per-16 sub-scales, f32 accumulate via Q8_K bsums."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::BlockQ2K::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[BlockQ2K]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "divisible(dim[-1], 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q2K, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
    - name: activation              # ys: &[BlockQ8K]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8K, granularity: ~, role: activation, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params:
    variant: QMatMul
    fields:
      quant_type:  { kind: QuantType, constraint: "== Q2K" }
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "k % 256 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dtype == U8", note: "avx2/neon/simd128 SIMD path when target_feature enabled; scalar otherwise" }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"
  bytes_moved: "(84 + 292) * (k / 256)"   # 84 B/Q2_K super-block + 292 B/Q8_K super-block
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "2-bit quants, per-16 d/dmin sub-scales (f16), Q8_K activation d in f32; integer inner dot then f32 accumulate."

determinism: same_hardware_bitwise
```

---

## vec_dot_q3k  (Q3_K×Q8_K row dot)

One quantized dot of a Q3_K weight row against a Q8_K-quantized activation row. `fn vec_dot(n, xs:
&[BlockQ3K], ys: &[BlockQ8K]) -> f32` (`k_quants.rs:1013`; avx2 `avx.rs:463`, also neon). Q3_K is a
256-element super-block of 2-bit quants whose high (3rd) bit comes from an `hmask` plane, with 6-bit
packed sub-block scales and a block-level f16 `d`. The dot reconstructs the signed 3-bit value,
applies the sub-block scale, multiplies against the i8 Q8_K activation, and accumulates in f32 via
the Q8_K f32 `d` and `bsums`. `VecDotType = Q8K`. `n` must be a multiple of 256 (debug-asserted).
Contiguous flat slices only.

```fkc
kernel: vec_dot_q3k
op_kind: QMatMul
blurb: "Q3_K×Q8_K super-block row dot: 2-bit + hmask high bit, 6-bit packed scales, f32 accumulate."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::BlockQ3K::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[BlockQ3K]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "divisible(dim[-1], 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q3K, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
    - name: activation              # ys: &[BlockQ8K]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8K, granularity: ~, role: activation, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params:
    variant: QMatMul
    fields:
      quant_type:  { kind: QuantType, constraint: "== Q3K" }
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "k % 256 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dtype == U8", note: "avx2/neon SIMD path when target_feature enabled; scalar otherwise" }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"
  bytes_moved: "(110 + 292) * (k / 256)"   # 110 B/Q3_K super-block + 292 B/Q8_K super-block
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "2-bit + hmask high bit, 6-bit packed sub-block scales, f16 block d; Q8_K f32 d; integer inner dot, f32 accumulate."

determinism: same_hardware_bitwise
```

---

## vec_dot_q4k  (Q4_K×Q8_K row dot)

One quantized dot of a Q4_K weight row against a Q8_K-quantized activation row. `fn vec_dot(n, xs:
&[BlockQ4K], ys: &[BlockQ8K]) -> f32` (`k_quants.rs:1388`; avx2 `avx.rs:599`, also neon/simd128).
Q4_K is a 256-element super-block of 4-bit quants with 6-bit-packed per-sub-block scale/min
(`get_scale_min_k4`) and a block-level f16 `d`/`dmin`. The dot applies per-sub-block scale and min,
multiplies the 4-bit quant against the i8 Q8_K activation, and accumulates in f32 via the Q8_K f32
`d` and `bsums`. This is the storage dtype for the GGUF `Q4_K_M` weight (`GgmlDType::Q4K`, code 12;
§3.4); the medium-mix op distinction is the `QuantType::Q4_K_M` key token, not a separate storage
dtype. `VecDotType = Q8K`. `n` must be a multiple of 256 (debug-asserted). Contiguous flat slices
only.

```fkc
kernel: vec_dot_q4k
op_kind: QMatMul
blurb: "Q4_K×Q8_K super-block row dot: 4-bit quants, 6-bit packed scale/min, f32 accumulate (GGUF Q4_K_M weight)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::BlockQ4K::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[BlockQ4K]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "divisible(dim[-1], 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q4K, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
    - name: activation              # ys: &[BlockQ8K]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8K, granularity: ~, role: activation, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params:
    variant: QMatMul
    fields:
      quant_type:  { kind: QuantType, constraint: "== Q4_K_M" }   # QuantType variant name for the Q4K medium format (§3.4)
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "k % 256 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dtype == U8", note: "avx2/neon/simd128 SIMD path when target_feature enabled; scalar otherwise" }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"
  bytes_moved: "(144 + 292) * (k / 256)"   # 144 B/Q4_K super-block + 292 B/Q8_K super-block
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "4-bit quants, 6-bit packed scale/min (get_scale_min_k4), f16 d/dmin; Q8_K f32 d; integer inner dot, f32 accumulate."

determinism: same_hardware_bitwise
```

---

## vec_dot_q5k  (Q5_K×Q8_K row dot)

One quantized dot of a Q5_K weight row against a Q8_K-quantized activation row. `fn vec_dot(n, xs:
&[BlockQ5K], ys: &[BlockQ8K]) -> f32` (`k_quants.rs:1642`; avx2 `avx.rs:684`, also neon). Q5_K is a
256-element super-block of 4-bit quants whose 5th high bit comes from a `qh` plane, with 6-bit-packed
per-sub-block scale/min and a block-level f16 `d`/`dmin`. The dot reconstructs the 5-bit value,
applies per-sub-block scale and min, multiplies against the i8 Q8_K activation, and accumulates in
f32 via the Q8_K f32 `d` and `bsums`. `VecDotType = Q8K`. `n` must be a multiple of 256
(debug-asserted). Contiguous flat slices only.

```fkc
kernel: vec_dot_q5k
op_kind: QMatMul
blurb: "Q5_K×Q8_K super-block row dot: 4-bit + qh high bit, 6-bit packed scale/min, f32 accumulate."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::BlockQ5K::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[BlockQ5K]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "divisible(dim[-1], 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q5K, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
    - name: activation              # ys: &[BlockQ8K]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8K, granularity: ~, role: activation, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params:
    variant: QMatMul
    fields:
      quant_type:  { kind: QuantType, constraint: "== Q5K" }
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "k % 256 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dtype == U8", note: "avx2/neon SIMD path when target_feature enabled; scalar otherwise" }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"
  bytes_moved: "(176 + 292) * (k / 256)"   # 176 B/Q5_K super-block + 292 B/Q8_K super-block
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "4-bit + qh high bit, 6-bit packed scale/min, f16 d/dmin; Q8_K f32 d; integer inner dot, f32 accumulate."

determinism: same_hardware_bitwise
```

---

## vec_dot_q6k  (Q6_K×Q8_K row dot)

One quantized dot of a Q6_K weight row against a Q8_K-quantized activation row. `fn vec_dot(n, xs:
&[BlockQ6K], ys: &[BlockQ8K]) -> f32` (`k_quants.rs:1946`; avx2 `avx.rs:287`, also neon/simd128).
Q6_K is a 256-element super-block whose 6-bit quant is assembled from a 4-bit `ql` plane plus a 2-bit
`qh` plane, with per-16 i8 sub-scales and a block-level f16 `d`. The dot reconstructs the signed
6-bit value, applies the per-16 scale, multiplies against the i8 Q8_K activation, and accumulates in
f32 via the Q8_K f32 `d` and `bsums`. `VecDotType = Q8K`. `n` must be a multiple of 256
(debug-asserted). Contiguous flat slices only.

```fkc
kernel: vec_dot_q6k
op_kind: QMatMul
blurb: "Q6_K×Q8_K super-block row dot: 4-bit ql + 2-bit qh, per-16 i8 scales, f32 accumulate."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::BlockQ6K::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[BlockQ6K]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "divisible(dim[-1], 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q6K, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
    - name: activation              # ys: &[BlockQ8K]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8K, granularity: ~, role: activation, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params:
    variant: QMatMul
    fields:
      quant_type:  { kind: QuantType, constraint: "== Q6K" }
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "k % 256 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dtype == U8", note: "avx2/neon/simd128 SIMD path when target_feature enabled; scalar otherwise" }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"
  bytes_moved: "(210 + 292) * (k / 256)"   # 210 B/Q6_K super-block + 292 B/Q8_K super-block
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "6-bit from 4-bit ql + 2-bit qh, per-16 i8 scales, f16 block d; Q8_K f32 d; integer inner dot, f32 accumulate."

determinism: same_hardware_bitwise
```

---

## vec_dot_q8k  (Q8_K×Q8_K row dot)

One quantized dot of a Q8_K weight row against a Q8_K activation row. `fn vec_dot(n, xs: &[BlockQ8K],
ys: &[BlockQ8K]) -> f32` (`k_quants.rs:2211`; avx2 `avx.rs:797`, also neon/simd128). Q8_K is a
256-element super-block of i8 quants with an **f32** per-block scale `d` (unlike the f16 scales of the
legacy / other K-quant formats) plus per-16 `bsums` (i16). The dot accumulates the i8×i8 integer
inner product and scales by `d_x · d_y` in f32. `VecDotType = Q8K` (self). `n` must be a multiple of
256 (debug-asserted). Note: Q8_K is also the activation `VecDotType` for all the K-quant weight dots
above. **Documented gap (not a dispatch-key claim): there is no `QuantType::Q8K` variant** — Q8K is
only a `VecDotType`, never a weight `QuantType`, so this self×self dot has no standalone `QMatMul`
dispatch key and is not separately registrable; it is reached only as the activation side of the
K-quant weight dots. Contiguous flat slices only.

```fkc
kernel: vec_dot_q8k
op_kind: QMatMul
blurb: "Q8_K×Q8_K super-block row dot: i8·i8 i32-accumulate ×(d_x·d_y) f32; f32 block scale."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::BlockQ8K::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[BlockQ8K]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "divisible(dim[-1], 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8K, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
    - name: activation              # ys: &[BlockQ8K]
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8K, granularity: ~, role: activation, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
  op_params:
    variant: QMatMul
    fields:
      # AS-BUILT NOTE (documented, NOT a dispatch-key claim): there is NO `QuantType::Q8K` variant —
      # Q8K is only a `VecDotType` (the activation block for every K-quant weight dot), never a
      # weight QuantType. So this self×self Q8K dot has NO `quant_type` dispatch key to bind: it is
      # NOT separately registrable as a QMatMul cell keyed on Q8K, and is reached only as the
      # activation side of the K-quant dots above. `quant_type` is therefore left unconstrained here.
      quant_type:  { kind: QuantType, constraint: ~, note: "NO QuantType::Q8K variant exists (Q8K is a VecDotType only); not a registrable key" }
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "k % 256 == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(out)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dtype == U8", note: "avx2/neon/simd128 SIMD path when target_feature enabled; scalar otherwise" }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 8
  notes: "NO QuantType::Q8K variant exists (capability/QuantType): Q8K is a VecDotType only, so this dot has no standalone QMatMul dispatch key — it serves as the activation side of every K-quant weight dot."

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"
  bytes_moved: "(292 + 292) * (k / 256)"   # 292 B/Q8_K super-block (4 d + 256 qs + 32 bsums) on each side
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "i8·i8 i32-accumulate ×(d_x·d_y) f32; Q8_K block scale d is f32 (not f16). Primarily the K-quant activation VecDotType; no QuantType::Q8K dispatch key exists."

determinism: same_hardware_bitwise
```

---

## vec_dot_f32  (dense f32 row dot)

One dense f32 dot product of a weight row against an activation row. `fn vec_dot(n, xs: &[f32], ys:
&[f32]) -> f32` (`k_quants.rs:2382`, forwards to `vec_dot_unopt` at `:2386`). f32 implements
`GgmlType` as a degenerate "block" format (`BLCK_SIZE = 1`, `DIRECT_COPY = true`, `VecDotType = f32`)
so it flows through the same `matmul` driver. The dot delegates to the unsafe FFI-style helper
`fuel_core_types::cpu::vec_dot_f32(xs_ptr, ys_ptr, &mut res, n)` and accumulates in f32. Length must
satisfy `xs.len() >= n` and `ys.len() >= n` (debug-asserted). Contiguous flat slices only.

```fkc
kernel: vec_dot_f32
op_kind: MatMul
blurb: "Dense f32 row dot via fuel_core_types::cpu::vec_dot_f32; f32 accumulate; contiguous slices."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::f32::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[f32]
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=activation"
    - name: activation              # ys: &[f32]
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
  op_params:
    variant: Matmul                 # OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k }
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>" }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "== weight.dim[-1] == activation.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(weight)   # f32 in, f32 out
      shape_rule: matmul(weight, activation)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"                    # per row-dot: one fmadd per element
  bytes_moved: "2 * k * 4"          # read both f32 rows
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "dense f32 dot via fuel_core_types::cpu::vec_dot_f32 (unsafe FFI-style call); deterministic f32 accumulation order."

determinism: same_hardware_bitwise
```

---

## vec_dot_f16  (dense f16 row dot)

One dense f16 dot product of a weight row against an activation row, accumulated in f32. `fn
vec_dot(n, xs: &[f16], ys: &[f16]) -> f32` (`k_quants.rs:2421`, forwards to `vec_dot_unopt`). f16
implements `GgmlType` as a degenerate block format (`BLCK_SIZE = 1`, `DIRECT_COPY = true`,
`VecDotType = f16`). The dot delegates to `fuel_core_types::cpu::vec_dot_f16(xs_ptr, ys_ptr, &mut
res, n)`, widening f16 lanes and accumulating in f32 (the return type). Length must satisfy `xs.len()
>= n` and `ys.len() >= n` (debug-asserted). Contiguous flat slices only.

```fkc
kernel: vec_dot_f16
op_kind: MatMul
blurb: "Dense f16 row dot via fuel_core_types::cpu::vec_dot_f16; widen to f32, f32 accumulate."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::f16::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[f16]
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=activation"
    - name: activation              # ys: &[f16]
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
  op_params:
    variant: Matmul
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>" }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "== weight.dim[-1] == activation.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(weight)   # f16 in → f16 out (vec_dot returns f32, narrowed by the matmul_f16 driver)
      shape_rule: matmul(weight, activation)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 2
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"
  bytes_moved: "2 * k * 2"          # read both f16 rows
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "dense f16 dot via fuel_core_types::cpu::vec_dot_f16; f16 lanes widened to f32, accumulated in f32."

determinism: same_hardware_bitwise
```

---

## vec_dot_bf16  (dense bf16 row dot)

One dense bf16 dot product of a weight row against an activation row, accumulated in f32. `fn
vec_dot(n, xs: &[bf16], ys: &[bf16]) -> f32` (`k_quants.rs:2472`, forwards to `vec_dot_unopt` at
`:2476`). bf16 implements `GgmlType` as a degenerate block format (`BLCK_SIZE = 1`, `DIRECT_COPY =
true`, `VecDotType = bf16`). The dot delegates to `fuel_core_types::cpu::vec_dot_bf16(xs_ptr, ys_ptr,
&mut res, n)`, widening bf16 lanes and accumulating in f32. Length must satisfy `xs.len() >= n` and
`ys.len() >= n` (debug-asserted). Contiguous flat slices only.

```fkc
kernel: vec_dot_bf16
op_kind: MatMul
blurb: "Dense bf16 row dot via fuel_core_types::cpu::vec_dot_bf16; widen to f32, f32 accumulate."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::bf16::vec_dot"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # xs: &[bf16]
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=activation"
    - name: activation              # ys: &[bf16]
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=weight"
  op_params:
    variant: Matmul
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>" }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "== weight.dim[-1] == activation.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(weight)   # bf16 in → bf16 out
      shape_rule: matmul(weight, activation)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 2
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * k"
  bytes_moved: "2 * k * 2"          # read both bf16 rows
  overhead_ns: ~                    # inlined vec_dot building block, NO launch; not free/metadata-only (gemm_like), so the inline-call overhead is Judge-measured, not an authored 0 (§4.4)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "dense bf16 dot via fuel_core_types::cpu::vec_dot_bf16; bf16 lanes widened to f32, accumulated in f32."

determinism: same_hardware_bitwise
```

---

## matmul  (f32-activation quantized GEMM driver)

The quantized matmul driver. `fn matmul<T: GgmlType>((m, k, n), lhs: &[f32], rhs_t: &[T], dst:
&mut [f32]) -> Result<()>` (`k_quants.rs:2284`). `lhs` is the f32 activation `(m × k)`; `rhs_t` is
the **transposed / row-major** quantized weight `(n × k)` held as `T` blocks; `dst` is the `(m × n)`
f32 result. Internally it (1) allocates a scratch `lhs_b` and quantizes each `lhs` row to
`T::VecDotType` via `from_float` (or `direct_copy` when `T::DIRECT_COPY`), then (2) for each output
cell runs `T::vec_dot(k, rhs_col, lhs_row)` over rayon (`with_min_len(128).max_len(512)`). `lhs` is
indexed `row_idx * k`, `rhs_t` indexed `col_idx * k_in_blocks`, `dst` indexed `row_idx * n`; `k` is
rounded up to the block boundary via `div_ceil`. **Contiguous, row-major, zero-offset only.** Output
dtype f32, shape `m * n`, dense, no aliasing. Returns `Result` but only `Ok` in practice — size
mismatches are `debug_assert` (the f16 driver uses a real `bail!`; this one does not). Allocates the
`lhs_b` scratch every call (TODO: pre-allocate). This driver is the GEMM entry point that turns the
per-format `vec_dot` building blocks above into a quantized matmul; `T` is monomorphized to a block
format (`QMatMul`) or to a dense float format (`MatMul`).

```fkc
kernel: matmul
op_kind: QMatMul
blurb: "Quantized GEMM driver: f32 activation × transposed block-quant weight → f32, per-cell vec_dot over rayon."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::matmul"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs                     # f32 activation (m × k), row-major
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "dim[1]=k"
    - name: rhs_t                   # transposed quantized weight (n × k) as T blocks
      dtypes: [U8]                  # opaque block bytes; logical dtype carried in fdx.quant
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "last_dim_eq=lhs"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q4_0, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
        # ggml_dtype monomorphized per T; Q4_0 shown as the canonical instance. The QMatMul key's
        # QuantType token (Q4_0..Q6K / Q4_K_M) distinguishes the registered cell per format.
  op_params:
    variant: QMatMul                # OpParams::QMatMul { quant_type, batch_count, m, n, k }
    fields:
      quant_type:  { kind: QuantType, note: "the T block format: Q4_0..Q6K / Q4_K_M" }
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "rounded up to T::BLCK_SIZE via div_ceil" }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(F32)        # output is always &mut [f32]
      shape_rule: matmul(rhs_t, lhs)   # (m × n) result
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
    - { when: "dtype == U8", note: "vec_dot inner uses avx2/neon/simd128 when target_feature enabled" }
  in_place: false
  alignment_bytes: 4
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * m * n * k"            # standard GEMM: 2·M·N·K
  bytes_moved: "m * k * 4 + n * k * w_block_bytes + m * n * 4"   # f32 lhs + block-quant weight + f32 dst
  overhead_ns: ~                    # per-call lhs_b scratch alloc + rayon spawn is real (NOT zero); Judge-measured
  memory: { device_bytes: 0, host_bytes: "m * (k / w_block_size) * vecdot_block_bytes", disk_bytes: 0 }   # lhs_b scratch

precision:
  bit_stable_on_same_hardware: false   # rayon parallel per-cell scheduling; per-cell dot itself is deterministic
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "each output cell is an independent vec_dot (f32 accumulate); cells are computed over rayon, so cross-run cell ordering is scheduler-dependent though each cell is bitwise-stable. Activation re-quantized to T::VecDotType per row."

determinism: nondeterministic
```

---

## matmul_f16  (f16-activation quantized GEMM driver)

The f16-activation variant of the quantized matmul driver. `fn matmul_f16<T: GgmlType>((m, k, n),
lhs: &[f16], rhs_t: &[T], dst: &mut [f16]) -> Result<()>` (`k_quants.rs:2333`). Same structure as
`matmul` but `lhs` / `dst` are f16. Each `lhs` row is widened through an intermediate f32 buffer
(`lhs.to_f32()`) and then quantized to `T::VecDotType` via `from_float`; the per-cell `vec_dot`
computes in f32 and the result is narrowed with `f16::from_f32(value)` on store. Uses
`fuel_core_types::bail!` for the lhs length mismatch — a **real `Err`**, unlike the f32 `matmul`
which only `debug_assert`s. Output dtype f16, shape `m * n`, dense, no aliasing. **Contiguous,
row-major, zero-offset only.** Allocates an `lhs_b` scratch and an `lhs_f32` per-row buffer.

```fkc
kernel: matmul_f16
op_kind: QMatMul
blurb: "f16-activation quantized GEMM driver: widen f16→f32 → quantize → vec_dot f32 → narrow f16 store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_quantized::k_quants::matmul_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs                     # f16 activation (m × k), row-major
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "dim[1]=k"
    - name: rhs_t                   # transposed quantized weight (n × k) as T blocks
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "last_dim_eq=lhs"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q4_0, granularity: ~, role: weight, scale_operand: ~ }   # GGML carries ggml_dtype ONLY; no PerBlock (FDX §6.2/V5)
        # ggml_dtype monomorphized per T; Q4_0 shown as the canonical instance.
  op_params:
    variant: QMatMul
    fields:
      quant_type:  { kind: QuantType, note: "the T block format: Q4_0..Q6K / Q4_K_M" }
      batch_count: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "rounded up to T::BLCK_SIZE via div_ceil" }

return:
  outputs:
    - name: dst
      dtype_rule: fixed(F16)        # output is &mut [f16]; vec_dot computes f32, narrowed on store
      shape_rule: matmul(rhs_t, lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
    - { when: "dtype == U8", note: "vec_dot inner uses avx2/neon/simd128 when target_feature enabled" }
  in_place: false
  alignment_bytes: 2
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * m * n * k"            # standard GEMM: 2·M·N·K
  bytes_moved: "m * k * 2 + n * k * w_block_bytes + m * n * 2"   # f16 lhs + block-quant weight + f16 dst
  overhead_ns: ~                    # per-call lhs_b + lhs_f32 scratch alloc + rayon spawn is real (NOT zero); Judge-measured
  memory: { device_bytes: 0, host_bytes: "m * (k / w_block_size) * vecdot_block_bytes + k * 4", disk_bytes: 0 }   # lhs_b + lhs_f32 scratch

precision:
  bit_stable_on_same_hardware: false   # rayon per-cell scheduling; plus f32 intermediate widen + f16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "each cell is an independent f32 vec_dot narrowed to f16 on store; lhs widened f16→f32 before re-quantize. Cross-run cell order scheduler-dependent; per-cell bitwise stable. lhs length mismatch returns a real Err (bail!)."

determinism: nondeterministic
```

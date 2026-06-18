# Kernel inventory — `fuel-quantized`

Backend-agnostic ggml/gguf block-format numerics. Owns the `BlockQX` block
structs, the `GgmlType` trait + its scalar reference impls, cfg-gated CPU SIMD
`vec_dot` helpers (avx/neon/simd128), and the `matmul`/`matmul_f16` drivers that
turn `vec_dot` into a quantized GEMM. Also provides the `cpu.rs` dyn adapter
(`QuantizedType` for `Vec<BlockQX>`) plus `cpu_zeros`/`cpu_from_data`.

## Crate-wide layout reality (applies to EVERY kernel here)

- **Every kernel operates on flat `&[T]` / `&[f32]` / `&[f16]` slices. There is
  NO `Layout`, NO `Shape`, NO `StridedIndex`, NO offset, NO broadcast anywhere
  in this crate.** All stride/offset/broadcast handling lives in the backend
  adapters (`fuel-cpu-backend`, `fuel-core/src/quantized/`) which contiguify
  before calling in. So for the contract: **contiguous-only, zero-offset,
  no-broadcast, dense row-major** is the universal precondition.
- Block-format kernels (`Q*`) require element counts that are exact multiples of
  `BLCK_SIZE` (32 for the legacy `Q4_0..Q8_1`, 256 = `QK_K` for the K-quants).
  Size checks are `debug_assert!` only — in release they are UB/panic-on-OOB,
  not validated.
- The opt `vec_dot` is selected at compile time via `#[cfg(target_feature=...)]`
  (avx2 → neon → simd128 → scalar `vec_dot_unopt`). On the dev box (x86_64,
  no `avx2` target-feature by default) the scalar path is what actually runs
  unless `RUSTFLAGS` enables avx2. The SIMD and scalar forms are the SAME
  logical kernel with two implementations; they are listed once with the SIMD
  source noted.
- All accumulation is in f32 (the `vec_dot` return type) regardless of the
  block's stored scale precision (f16 for most, f32 for Q8K).

---

## Per-format kernels (the `GgmlType` impls in `k_quants.rs`)

For each ggml block dtype, fuel provides up to four ops:
`to_float` (dequantize), `from_float` (quantize), `from_float_imatrix`
(importance-matrix quantize), `vec_dot` (the matmul building block, paired with
a fixed `VecDotType`).

### Dequantize — `GgmlType::to_float` (one per format)

| dtype | source (scalar) | notes |
|---|---|---|
| Q4_0 | k_quants.rs:177 | nibble−8 × d(f16) |
| Q4_1 | k_quants.rs:354 | nibble × d + m |
| Q5_0 | k_quants.rs:463 | 5th bit from qh; (val−16) × d |
| Q5_1 | k_quants.rs:578 | 5th bit from qh; val × d + m |
| Q8_0 | k_quants.rs:611 | i8 × d |
| Q8_1 | k_quants.rs:759 | **`unimplemented!()`** — panics |
| Q2K  | k_quants.rs:958 | 2-bit, per-16 sub-scales d/dmin |
| Q3K  | k_quants.rs:1314 | 2-bit + hmask high bit, 6-bit packed scales |
| Q4K  | k_quants.rs:1595 | 4-bit, 6-bit packed scale/min (`get_scale_min_k4`) |
| Q5K  | k_quants.rs:1890 | 4-bit + qh high bit, 6-bit packed scale/min |
| Q6K  | k_quants.rs:2158 | 4-bit ql + 2-bit qh, i8 per-16 scales |
| Q8K  | k_quants.rs:2269 | i8 × d(f32) |

Common contract: input `xs: &[BlockX]`, output `ys: &mut [f32]` whose length
`k` MUST be a multiple of `BLCK_SIZE` and equal to `xs.len() * BLCK_SIZE`
(K-quants validate via `group_for_dequantization`, legacy via `debug_assert`).
Output dtype is always **f32**; output shape = `xs.len() * BLCK_SIZE` elements,
written densely in block order. No aliasing (separate buffers). Q8_1 has no
dequant (it only exists as a `VecDotType`).

### Quantize — `GgmlType::from_float` (one per format)

| dtype | source | scale search |
|---|---|---|
| Q4_0 | k_quants.rs:199 | amax/-8, round-half via +8.5 cast |
| Q4_1 | k_quants.rs:315 | (max−min)/15 |
| Q5_0 | k_quants.rs:424 | amax/-16 |
| Q5_1 | k_quants.rs:535 | (max−min)/31 |
| Q8_0 | k_quants.rs:629 | amax/127 |
| Q8_1 | k_quants.rs:728 | amax/127, also stores `s`=sum×d |
| Q2K  | k_quants.rs:836 | `make_qkx1_quants(3,5)` + Q4SCALE=15 |
| Q3K  | k_quants.rs:1135 | `make_q3_quants(.,4,rmse)` |
| Q4K  | k_quants.rs:1470 | `make_qkx1_quants(15,5)`, 6-bit scale/min |
| Q5K  | k_quants.rs:1732 | `make_qkx1_quants(31,5)` |
| Q6K  | k_quants.rs:2005 | `make_qx_quants(16,32,rmse=1)`; **unsafe raw-ptr** |
| Q8K  | k_quants.rs:2230 | iscale=-128/max; also fills `bsums` (i16) |

Common contract: input `xs: &[f32]` whose length MUST be a multiple of
`BLCK_SIZE`, output `ys: &mut [BlockX]` with `ys.len() == xs.len()/BLCK_SIZE`.
K-quants validate via `group_for_quantization` (debug). Output dtype is the
block type; per-block scales stored as f16 (legacy + Q2K..Q6K) or f32 (Q8K).
Q6K's impl uses raw pointers (`unsafe`) directly. No in-place/aliasing.

### Imatrix quantize — `GgmlType::from_float_imatrix` (subset only)

Default trait impl **`panic!`s** (`k_quants.rs:34`); only the K-quants override.

| dtype | source | weighting |
|---|---|---|
| Q2K | k_quants.rs:900 | `make_qkx3_quants` + `make_qp_quants`, sigma2=Σx²/QK_K |
| Q3K | k_quants.rs:1216 | `make_qx_quants` (unsafe), sigma2=2Σx²/QK_K |
| Q4K | k_quants.rs:1530 | `make_qkx3_quants` + `make_qp_quants` |
| Q5K | k_quants.rs:1807 | `make_qkx3_quants` + `make_qp_quants` |
| Q6K | k_quants.rs:2077 | `make_qx_quants` (unsafe) per-16 imatrix row |

Extra params beyond `from_float`: `imatrix_weights: &[f32]`, `n_per_row: usize`.
The block index modulo `n_per_row/QK_K` picks the imatrix row. NOT implemented
for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q8K/f32/f16/bf16 → those panic if called.

### Dot product — `GgmlType::vec_dot` / `vec_dot_unopt` (one per format)

`fn vec_dot(n, xs: &[Self], ys: &[Self::VecDotType]) -> f32`. `n` = number of
elements; both slices hold `n/BLCK_SIZE` blocks. Returns an f32 scalar (the
quantized dot of one weight row against one activation row). The right operand
(`ys`) is the activation pre-quantized to `VecDotType`.

| dtype | VecDotType | scalar src | SIMD (avx2) src |
|---|---|---|---|
| Q4_0 | Q8_0 | k_quants.rs:252 | avx.rs:51 (also neon/simd128) |
| Q4_1 | Q8_1 | k_quants.rs:285 | avx.rs:148 |
| Q5_0 | Q8_0 | k_quants.rs:401 | avx.rs:176 |
| Q5_1 | Q8_1 | k_quants.rs:501 | avx.rs:212 |
| Q8_0 | Q8_0 | k_quants.rs:674 | avx.rs:72 (also neon/simd128) |
| Q8_1 | Q8_1 | k_quants.rs:708 | avx.rs:127 |
| Q2K  | Q8K  | k_quants.rs:783 | avx.rs:380 (also neon/simd128) |
| Q3K  | Q8K  | k_quants.rs:1013 | avx.rs:463 (also neon) |
| Q4K  | Q8K  | k_quants.rs:1388 | avx.rs:599 (also neon/simd128) |
| Q5K  | Q8K  | k_quants.rs:1642 | avx.rs:684 (also neon) |
| Q6K  | Q8K  | k_quants.rs:1946 | avx.rs:287 (also neon/simd128) |
| Q8K  | Q8K  | k_quants.rs:2211 | avx.rs:797 (also neon/simd128) |

Precision: i8/i16/i32 integer dot inner-accumulate, then × (d_x·d_y) in f32.
Legacy d is f16 (read via `f16::to_f32`); Q8K d is f32. SIMD uses `_mm256_madd*`
i16/i32 lanes + `_mm256_fmadd_ps` f32 accumulate; the avx `q5_0/q5_1` path uses
the local `bytes_from_bits_32_fifth` helper (avx.rs:94). `n` must be a multiple
of `BLCK_SIZE` (and a multiple of 2 blocks for Q4_1/Q5_0/Q5_1) — debug-asserted.

---

## Float "block" formats (BLCK_SIZE = 1, DIRECT_COPY = true)

`f32` (k_quants.rs:2376), `f16` (k_quants.rs:2421), `bf16` (k_quants.rs:2466)
implement `GgmlType` so they flow through the same `matmul` driver.

- `to_float`/`from_float`: element-wise copy / convert (`HalfFloatSliceExt`
  for f16/bf16). Lengths must be equal (debug).
- `vec_dot`: dense f32/f16/bf16 dot via `fuel_core_types::cpu::vec_dot_{f32,f16,
  bf16}` (unsafe FFI-style call). Accumulates in f32, returns f32.
- `direct_copy`: lets `matmul` skip the per-row `from_float` (used when
  `DIRECT_COPY` is set).
- `from_float_imatrix`: NOT overridden → panics.

---

## Matmul drivers (the GEMM entry points)

### `matmul` — k_quants.rs:2284

`fn matmul<T: GgmlType>((m,k,n), lhs: &[f32], rhs_t: &[T], dst: &mut [f32])`.
The quantized matmul: `lhs` is the f32 activation `(m×k)`, `rhs_t` is the
**transposed/row-major** quantized weight `(n×k)` as blocks, `dst` is `(m×n)`
f32. Internally:
1. allocates `lhs_b` and quantizes each `lhs` row to `T::VecDotType` via
   `from_float` (or `direct_copy` when `T::DIRECT_COPY`),
2. for each output cell runs `T::vec_dot(k, rhs_col, lhs_row)` over rayon
   (`with_min_len(128).max_len(512)`).

Layout: **contiguous, row-major, zero-offset only.** `lhs` indexed
`row_idx*k`, `rhs_t` indexed `col_idx*k_in_blocks`, `dst` indexed `row_idx*n`.
Output dtype f32, shape `m*n`, dense. `k` rounded up to block boundary via
`div_ceil`. Allocates a scratch `lhs_b` each call (TODO: pre-allocate).
Returns `Result` but only `Ok` in practice (size mismatch is `debug_assert`).

### `matmul_f16` — k_quants.rs:2333

Same as `matmul` but `lhs`/`dst` are `&[f16]`/`&mut [f16]`. Quantizes each lhs
row through an intermediate f32 buffer (`lhs.to_f32()` → `from_float`), computes
`vec_dot` in f32, writes `f16::from_f32(value)`. Uses
`fuel_core_types::bail!` for the lhs length mismatch (real `Err`, unlike the
f32 path which debug-asserts). Output dtype f16, shape `m*n`, dense, no aliasing.

---

## CPU dyn adapter (`cpu.rs`) — not numeric kernels, but the dispatch surface

- `QuantizedType` trait (cpu.rs:17) impl'd for `Vec<T: GgmlType>` (cpu.rs:32):
  thin forwarders — `matmul_t` → `k_quants::matmul`, `matmul_t_f16` →
  `k_quants::matmul_f16`, `dequantize` → `to_float` into a `HostBuffer::F32`,
  `from_float`/`from_float_imatrix` → trait, plus size/ptr accessors.
- `cpu_zeros(dtype, elem_count)` (cpu.rs:83): allocate zeroed `Vec<BlockX>`
  sized `elem_count / BLCK_SIZE` (or `elem_count` for f32/f16/bf16).
- `cpu_from_data(dtype, data)` (cpu.rs:103): reinterpret raw `Cow<[u8]>` as
  `&[BlockX]` (`as_t_slice`, asserts size-multiple + alignment) and `.to_vec()`.

These are storage/dispatch glue, not distinct numeric kernels; the actual math
is the `GgmlType` impls above. The `QuantizedDeviceKernels` /
`DynQuantizedStorage` traits themselves live in `fuel-core-types`; the CPU impl
of them lives in `fuel-cpu-backend`, NOT in this crate.

---

## Quantization-search helpers (`utils.rs`) — internal, called by `from_float*`

Not standalone kernels (private `pub(super)`), but they carry the numeric
contracts of the quantizers: `nearest_int` (round), `make_qkx1_quants`,
`make_qkx3_quants`, `make_q3_quants`, `make_qx_quants` (unsafe raw-ptr, RMSE
iteration), `make_qp_quants` (positive-only MSE refine), `get_scale_min_k4`
(6-bit scale/min unpack), `group_for_quantization`/`group_for_dequantization`
(size validation + block zip). Listed here for completeness; the per-format
quantize entries above are the contract surface.

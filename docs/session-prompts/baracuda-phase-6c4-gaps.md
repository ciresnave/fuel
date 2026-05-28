# Baracuda asks — Fuel-cuda-kernels retirement, Phase 6c.4 blockers

**Status as of 2026-05-27 (post-alpha.54).** Baracuda alpha.54
closed all six original Phase 6c.4 asks. Fuel-cuda-kernels PTX
module count: **9 → 1**. The remaining `Id::Reduce` is kept alive
by two fuel-nn callers (RoPE variants baracuda alpha.54 doesn't
ship yet):

| Caller                      | Op label  | Baracuda equivalent     |
|-----------------------------|-----------|-------------------------|
| `fuel-nn::RotaryEmbI`       | `rope_i`  | none (interleaved pair) |
| `fuel-nn::RotaryEmbThd`     | `rope_thd`| none (THD layout)       |

(`fuel-nn::RotaryEmb` — the non-interleaved batched rope —
shipped on `baracuda_kernels_rope_apply_<dt>_run` in alpha.54.)

---

## Remaining asks

### 7. Rope — interleaved-pair variant

**Fuel's `RotaryEmbI`** rotates **adjacent pairs `(2i, 2i+1)`**
within the last dim instead of `(i, i + d/2)`. Same FW shape as
`rope_apply` otherwise (precomputed cos/sin tables + `[bh, td]`
layout). Used by Phi-3, Llama-style models with interleaved
rotation convention.

**Reference signature** (mirror `rope_apply` exactly, swap pair
indexing):

```c
int baracuda_kernels_rope_apply_interleaved_f32_run(
    i32 bh,         // batch * heads
    i32 td,         // seq * head_dim
    i32 d,          // head_dim
    i32 stride_b,   // 0 if shared cos/sin; td/2 if per-batch
    const float* x,
    const float* cos_tab,
    const float* sin_tab,
    float* y,
    void* ws, usize ws_b, cudaStream_t stream);
```

Same shape for f32 / f64 / f16 / bf16 + matching `_backward`.

### 8. Rope — THD layout variant

**Fuel's `RotaryEmbThd`** handles **`[T, H, D]` layout** rather
than the typical `[B, H, T, D]`. The flatten/launcher math is
different from both `rope_apply` and the interleaved variant.

**Reference signature** mirrors `rope_apply` but uses `(t, h, d)`
shape input rather than `(bh, td, d)`:

```c
int baracuda_kernels_rope_apply_thd_f32_run(
    i32 t, i32 h, i32 d, i32 stride_b,
    const float* x,
    const float* cos_tab,
    const float* sin_tab,
    float* y,
    void* ws, usize ws_b, cudaStream_t stream);
```

---

## Closed asks (alpha.54)

All six original asks from this doc landed in alpha.54:

| #  | Family       | Status                                                                                |
|----|--------------|---------------------------------------------------------------------------------------|
| 1  | Reduce       | ✓ `reduce_{min,prod}_to_<fp>` + integer reduce sum/min/max/prod + integer `arg_reduce` |
| 2  | Rope         | ✓ `rope_apply_<dt>_run` with precomputed cos/sin                                       |
| 3  | Ternary      | ✓ `where_<cond>cond_<val>_run` full 3 × 11 matrix (contig + strided)                   |
| 4  | Fill         | ✓ `fill_u32/i16/fp8e4m3_run` + every `fill_<dt>_strided_run`                           |
| 5  | Indexing     | ✓ scatter (no `_add`) + index_add + integer values + i64idx variants                   |
| 6  | Sort         | ✓ `argsort_<dt>_run` for u8/i8/u32/i16/bf16/f16/fp8e4m3 + `argsort_<dt>_big_run`       |

---

## Phase 6c.5 — what unblocks crate retirement

Once gaps **7, 8** close (interleaved + THD rope), the last PTX
module (`Id::Reduce` + `reduce.cu`) drops and the workspace
member + `cudaforge` build dep go away in Phase 7.

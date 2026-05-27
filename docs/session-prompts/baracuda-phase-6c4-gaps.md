# Baracuda asks — Fuel-cuda-kernels retirement, Phase 6c.4 blockers

**Status as of 2026-05-27.** Phase 6c.4 has migrated softmax /
log_softmax / rms_norm / layer_norm / argsort from the PTX
`fuel-cuda-kernels` modules onto baracuda alpha.50. The five PTX
modules still on the workspace member (`Fill`, `Indexing`, `Reduce`,
`Sort`, `Ternary`) cannot be dropped without the following baracuda
gaps closed:

---

## 1. Reduce — multi-axis `_to` variants + integer dtypes

**Fuel's typed `CudaStorage::reduce_op` (`FastReduce`)** dispatches
through `Map1Any` over **all 11 dtypes** (U8/I8/U32/I16/I32/I64/
BF16/F16/F32/F64/F8E4M3) and **multi-axis `sum_dims: &[usize]`**.
Today on baracuda alpha.50:

| Op       | dtypes covered           | single-axis | multi-axis      |
|----------|--------------------------|-------------|-----------------|
| Sum      | f32/f64/f16/bf16         | ✓ `reduce_sum_<dt>_run` | ✓ `reduce_sum_to_<dt>_run` |
| Max      | f32/f64/f16/bf16         | ✓ `reduce_max_<dt>_run` | ✓ `reduce_max_to_<dt>_run` |
| Min      | f32/f64/f16/bf16         | ✓ `reduce_min_<dt>_run` | ✗ no `reduce_min_to` |
| Prod     | f32/f64/f16/bf16         | ✓ `reduce_prod_<dt>_run` | ✗ no `reduce_prod_to` |
| ArgMin   | f32/f64/f16/bf16         | ✓ `arg_reduce_argmin_<dt>_<idx>_run` | n/a (semantics) |
| ArgMax   | f32/f64/f16/bf16         | ✓ `arg_reduce_argmax_<dt>_<idx>_run` | n/a (semantics) |

**Asks:**

- **`reduce_min_to_<dt>_run`** / **`reduce_prod_to_<dt>_run`** — match
  the broadcast-reverse semantics of `reduce_sum_to` /
  `reduce_max_to` so multi-axis Min/Prod are one kernel call.
- **Integer-dtype reduce surface** — `reduce_{sum,min,max,prod,
  argmin,argmax}_<int_dt>_run` for U8/I8/U32/I16/I32/I64. (Today
  only `reduce_any` / `reduce_all` / `reduce_count_nonzero` /
  `bincount` cover integer inputs.) Fuel's typed Map1Any path
  cannot bail on the integer arms while the legacy
  `realize_f32_cuda` path is still in use.

---

## 2. Reduce — RoPE with precomputed cos/sin tables

**Fuel's `CudaStorage::rope(x, cos, sin, x_layout)`** rotates the
input by **caller-supplied** `cos` / `sin` tables of shape
`[seq, head_dim/2]`. This is the natural API for LLaMA-style
extended-context scaling (YaRN, NTK-aware, etc. — the caller
applies whichever angle schedule the model demands).

**Baracuda alpha.50's `rope_<dt>_run`** takes `(batch, heads, seq,
head_dim, base, pos_default_flag, x, positions, y, ...)` — it
generates θ internally from `pos · base^(-2i/D)`. Fuel cannot
route to this without losing the caller's angle-schedule control.

**Ask:** **`rope_apply_<dt>_run`** that takes precomputed `cos` +
`sin` tables in addition to (or in place of) the base/positions
parameterization. Reference signature (mirrors Fuel's Vulkan rope):

```c
int baracuda_kernels_rope_apply_f32_run(
    i32 bh,           // outer = batch * heads
    i32 td,           // seq * head_dim per (batch,head)
    i32 d,            // head_dim
    i32 stride_b,     // 0 if cos/sin shared; td/2 if per-batch
    const float* x,
    const float* cos,
    const float* sin,
    float* y,
    void* ws, usize ws_b, cudaStream_t stream);
```

Same FW + BW shape for f32 / f64 / f16 / bf16. The current
`rope_<dt>_run` covers the simpler base+positions case where Fuel
doesn't need it; both can coexist.

---

## 3. Ternary — `where_cond` integer-cond + integer-value coverage

**Fuel's typed `CudaStorage` `WhereCond`** Map2 dispatches over:
- **cond dtypes:** U8, U32, I64
- **value dtypes:** all 11 (including integer + F8E4M3)

**Baracuda alpha.50's `where_<dt>_{run,strided_run}`:**
- **cond dtype:** U8 only
- **value dtypes:** f32, f64, f16, bf16

**Asks:**

- **U32-cond + I64-cond variants** for the four fp value dtypes —
  e.g. `where_u32cond_f32_run`, `where_i64cond_f32_run`. Or a single
  `cond_kind` enum parameter that selects U8 vs U32 vs I64 at the
  FFI boundary (whichever is cleaner upstream).
- **Integer value-dtype coverage** — `where_<value_dt>_run` for
  U8/I8/U32/I16/I32/I64 values. Both contig + strided variants
  (Fuel's CPU path already handles all 11 combos).

---

## 4. Fill — missing dtypes + strided support

**Fuel's typed `CudaStorage::const_set(scalar, layout)`** writes a
scalar to a (possibly strided) destination across 11 dtypes:
U8/I8/U32/I16/I32/I64/BF16/F16/F32/F64/F8E4M3.

**Baracuda alpha.50's `fill_<dt>_run`:**
- **dtypes covered:** F32/F64/I32/I64/U8/I8/F16/BF16 (8)
- **missing:** U32, I16, F8E4M3
- **layout:** contig only (`numel + scalar + output`, no stride
  descriptor)

**Asks:**

- **`fill_u32_run`** / **`fill_i16_run`** / **`fill_fp8e4m3_run`**
  to close the dtype gap.
- **Strided fill** — `fill_<dt>_strided_run(numel, rank, shape,
  stride_y, scalar, y, …)` so Fuel can `const_set` directly into a
  view (today the PTX `const_set_<dt>` kernel walks the descriptor
  per-element).

---

## 5. Indexing — scatter / index_add + non-f32-f64 indices

**Fuel's typed `CudaStorage` indexing op surface:**

| Fuel op       | PTX kernel name        | value dtypes (11) | index dtypes      |
|---------------|------------------------|-------------------|-------------------|
| index_select  | `is_<dt>_<idx>`        | all 11            | U8/U32/I64        |
| gather        | `gather_<dt>_<idx>`    | all 11            | U8/U32/I64        |
| scatter_add   | `scadd_<dt>_<idx>`     | all 11            | U8/U32/I64        |
| index_add     | `iadd_<dt>_<idx>`      | all 11            | U8/U32/I64        |
| where_cond    | `where_<idx>`          | all 11 (value)    | U8/U32/I64 (cond) |

**Baracuda alpha.50 coverage:**

| Baracuda op     | dtypes covered           | strided? |
|-----------------|--------------------------|----------|
| index_select    | f32/f64                  | ?        |
| gather          | f32/f64 (U32 idx only)   | ?        |
| scatter         | — (no scatter without add) | —      |
| index_add       | — (missing)              | —        |

**Asks:**

- **`scatter_<dt>_<idx>_run`** without the `_add` reduction —
  Fuel's `scatter` (default) needs pure assign, not Σ-accumulate.
- **`index_add_<dt>_<idx>_run`** — Fuel's gradient path for
  `index_select` decomposes to `index_add` on the scatter-symmetric
  axis.
- **Integer value dtype coverage** for index_select / gather /
  scatter — U8/I8/U32/I16/I32/I64/BF16/F16/F8E4M3.
- **U8-idx + I64-idx variants** for the indexing ops (Fuel's
  legacy `WithDType` constraints permit all three idx widths).

---

## 6. Sort — non-f32-f64-i32-i64 dtypes + row_len > 1024

**Fuel's `arg_sort_last_dim`** dispatches via `Map1Any` over **all
11 dtypes**. Baracuda alpha.50's `argsort_<dt>_run`:
- **dtypes covered:** F32/F64/I32/I64 (4)
- **row_len cap:** 1024 (block-bitonic)

The Phase 6c.4 commit (2026-05-27) lands the 4-dtype migration with
a hard `row_len ≤ 1024` check. Fuel's `asort_big` test (SIZE=2000)
hits the cap; today's PTX path silently truncates via
`block_dim.min(1024)`.

**Asks:**

- **`argsort_<dt>_run` for U8/I8/U32/I16/BF16/F16/F8E4M3** — close
  the dtype gap so the fallback "PTX or bail" decision goes away.
- **Multi-block radix-sort variant** for `row_len > 1024` — common
  case is top-k sampling over vocab-size logits (32k–256k for
  LLMs). Could be a separate `argsort_<dt>_big_run` to keep the
  block-bitonic fast path simple.

---

## Phase 6c.5 — what unblocks crate retirement

Once gaps **3, 4, 5** close, fuel-cuda-kernels can drop:
- `Id::Ternary` + `ternary.cu` (gap #3)
- `Id::Fill` + `fill.cu` (gap #4)
- `Id::Indexing` + `indexing.cu` (gap #5)

Gaps **1, 2** are needed for `Id::Reduce` retirement (the
`FastReduce` typed-storage path + `rope_f32`).

Gap **6** is needed for `Id::Sort` retirement; alternatively, the
4-dtype baracuda + the `row_len ≤ 1024` bail-out is "good enough"
to drop the SORT module if Fuel's `asort_big` test gets a
`#[cfg(not(feature = "cuda"))]` gate (or migrates to CPU-only).

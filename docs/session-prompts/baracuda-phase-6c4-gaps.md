# Baracuda asks — Fuel-cuda-kernels retirement, all gaps closed

**Status as of 2026-05-28 (post-alpha.55).** Baracuda alpha.51 →
alpha.55 landed every Fuel-side ask in the original Phase 6c.4
sweep plus the two follow-up asks for interleaved + THD rope.
`fuel-cuda-kernels` is retired in commit `59d3a1dd`. The
workspace member, the `Id::*` PTX module set, and the
`reduce.cu`/`ternary.cu`/etc `.cu` sources are all gone. Fuel's
CUDA path is **100% baracuda-backed** end-to-end.

## Closed asks (alpha.51 – alpha.55)

| #  | Alpha | Family   | Status                                                                                |
|----|-------|----------|---------------------------------------------------------------------------------------|
| 1  | .52   | Reduce   | ✓ `reduce_{min,prod}_to_<fp>` + integer reduce + integer `arg_reduce`                  |
| 2  | .51   | Rope     | ✓ `rope_apply_<dt>_run` with precomputed cos/sin                                       |
| 3  | .53   | Ternary  | ✓ `where_<cond>cond_<val>_run` full 3 × 11 matrix (contig + strided)                   |
| 4  | .51   | Fill     | ✓ `fill_u32/i16/fp8e4m3_run` + every `fill_<dt>_strided_run`                           |
| 5  | .53/.54 | Indexing | ✓ scatter (no `_add`) + index_add + integer values + i64idx variants                  |
| 6  | .51/.54 | Sort   | ✓ `argsort_<dt>_run` full 11-dtype + `argsort_<dt>_big_run` (multi-block radix)        |
| 7  | .55   | Rope     | ✓ `rope_apply_interleaved_<dt>_run` (pair `(2k, 2k+1)`)                                |
| 8  | .55   | Rope     | ✓ `rope_apply_thd_<dt>_run` (`[T, H, D]` layout)                                       |

## Phase 7 — crate retirement (DONE, commit `59d3a1dd`)

- `fuel-cuda-kernels/` workspace member deleted.
- Last PTX module (`Id::Reduce` + `reduce.cu`) removed.
- `fuel-cuda-backend::CudaDevice::get_or_load_func` +
  `ModuleStore` cache deleted (zero callers post-Phase-6c.5).
- `cudaforge` retained as a build-time dep only for
  fuel-flash-attn-cuda-sys / fuel-flash-attn-v3-cuda-sys (they
  build their own CUDA sources, unrelated to fuel-cuda-kernels).

## Where Fuel's CUDA kernels live now

Single source of truth: **baracuda** (across the
`baracuda-kernels-sys` FFI surface + the optional `cudnn`/`cublas`/
`curand`/`nccl`/`cutlass` library-backed plans). Fuel
contributes call-site dispatch (`fuel-cuda-backend/src/
storage.rs`, `fuel-cuda-backend/src/baracuda/*.rs`,
`fuel-cuda-backend/src/byte_kernels.rs`,
`fuel-core/src/sort.rs`, `fuel-nn/src/{ops,rotary_emb,moe}.rs`)
and shape/stride descriptors. Kernel bodies live upstream.

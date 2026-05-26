# Baracuda ask — Phase 6c.2 storage.rs legacy CudaStorage migration

Context: Fuel's `fuel-cuda-backend/src/storage.rs` still has ~32 PTX
call sites backing the legacy `CudaStorage` eager API. Phase 6c.1
(byte_kernels prune) shipped; Phase 6c.2 (storage.rs Affine) shipped.
Continuing the storage.rs migration uncovers several baracuda-side
gaps. Audit fully runnable on baracuda alpha.38 — these are upstream
ASKS, not blocking complaints.

## Gap 1 — `unary_elu_<dtype>_run` α parameter

**Today:** baracuda hardcodes α = 1.0 in
`baracuda_kernels_unary_elu_<dtype>_run` (per the doc comment).

**Ask:** Add an `alpha: f32` (or per-dtype scalar) parameter to both
contig and strided variants:
```rust
pub fn baracuda_kernels_unary_elu_<dtype>_run(
    numel: i64, x: *const c_void, y: *mut c_void,
    alpha: f32,  // ← new
    workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
) -> i32;
```

**Why:** Fuel's public `Tensor::elu(α)` exposes an arbitrary `α: f64`
(`Elu(f64)` Map1 in storage.rs). The PTX kernel reads it; baracuda
hardcodes 1.0. Most models use α=1.0 in practice but the API allows
otherwise.

**Fuel-side commitment:** delete the PTX `uelu` kernel from
`fuel-cuda-kernels/src/unary.cu` once the parameter ships.

## Gap 2 — `unary_powf_<dtype>_run` (float-exponent power)

**Today:** baracuda ships `unary_powi_<dtype>_run` (integer exponent)
but no float-exponent variant.

**Ask:** Add `unary_powf_<dtype>_run` and `_strided_run` mirroring
the powi shape:
```rust
pub fn baracuda_kernels_unary_powf_<dtype>_run(
    numel: i64, x: *const c_void, y: *mut c_void,
    exponent: f32,  // (or wider for f64)
    workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
) -> i32;
```

**Why:** Fuel's `Tensor::powf(f64)` is a distinct op from powi (the
PTX kernel uses `powf` not `pow_int`). Fuel-side struct: `Powf(f64)`
in storage.rs.

**Fuel-side commitment:** retire the PTX `upowf` kernel.

## Gap 3 — `unary_step_<dtype>_run` and `unary_gelu_erf_<dtype>_run`

**Today:** baracuda has `unary_gelu_<dtype>_run` (the tanh
approximation) and `unary_erf_<dtype>_run` but no `unary_step_*`
(Heaviside step function) and no `unary_gelu_erf_*` (the exact erf-
based GELU).

**Ask:** Add `unary_step_<dtype>_run` (returns 1 where x > 0 else 0)
and `unary_gelu_erf_<dtype>_run` (the exact GELU = 0.5x(1+erf(x/√2))).
Both standard contig + strided pair, all 4 fp dtypes.

**Why:** Fuel exposes both via `UnaryOpT` (`Step`, `GeluErf` constants
in fuel-core/src/op.rs). Models that hard-pin to the exact GELU need
the erf-based variant; the existing baracuda `unary_gelu` uses the
tanh approximation which has slightly different numerics.

**Fuel-side commitment:** retire the PTX `ustep` and `ugelu_erf`
kernels.

## Gap 4 — Cast missing `u32` and `i16` dtypes

**Today:** baracuda has 98 cast symbols covering `bool/bf16/f16/f32/f64/
fp8e4m3/fp8e5m2/i8/i32/i64/u8` (11 dtypes → 88 cross-pairs + identity).
**Missing entirely:** `u32` and `i16`.

**Ask:** Add `cast_<src>_u32_run` and `cast_u32_<dst>_run` for the
other 11 dtypes (22 new symbols), plus `cast_<src>_i16_run` and
`cast_i16_<dst>_run` similarly (22 new symbols). Total: 44 symbols.
Same `(numel, x, y, ws, ws_b, stream)` shape as the existing cast FFI.

**Why:** Fuel's `CudaStorage::to_dtype` supports `U32` and `I16` as
source dtypes via the PTX `cast_*_*` family in fuel-cuda-kernels/cast.cu.
Without `u32` and `i16` in baracuda, the migration of `to_dtype` is
partial — those input dtypes would need to stay on PTX.

**Fuel-side commitment:** delete `fuel-cuda-kernels/src/cast.cu`
(~50 dtype-pair PTX kernels). The remaining match-arm dispatch in
`CudaStorage::to_dtype` collapses to a single `baracuda_kernels_cast_*_run`
call by `(src_dtype, dst_dtype)` lookup.

## Gap 5 — `reduce_sum_to` / `reduce_max_to` (autograd primitives)

**Today:** baracuda has rich reduce coverage (sum/max/min/mean across
dim, dim-set, full) but no broadcast-reverse reduction. Fuel's
`reduce_sum_to_f32` / `reduce_max_to_f32` (called from autograd's
`Op::ReduceSumTo` / `Op::ReduceMaxTo`) consume the `REDUCE` PTX
module's `fast_sum_f32` / `fast_max_f32` symbols.

**Ask:** Add `baracuda_kernels_reduce_sum_to_<dtype>_run` and
`baracuda_kernels_reduce_max_to_<dtype>_run`:
```rust
pub fn baracuda_kernels_reduce_sum_to_<dtype>_run(
    src: *const c_void,
    dst: *mut c_void,
    input_shape: *const i32,   // [rank_in]
    input_stride: *const i64,  // [rank_in]
    rank_in: i32,
    output_shape: *const i32,  // [rank_out]; left-padded with 1s to rank_in
    workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
) -> i32;
```
Semantics: for each output element, sum (or max) all input elements
that broadcast to that position — i.e., the reverse of
`Op::BroadcastTo`. Coverage: f32/f64/f16/bf16.

**Why:** these are the only remaining PTX `REDUCE` callers after
Phase 6c.1; without them, the REDUCE module can't retire. Fuel-side
struct: the byte-storage wrappers
`fuel-cuda-backend::byte_kernels::reduce_sum_to_f32` /
`reduce_max_to_f32` (in the binding-table registration path).

**Fuel-side commitment:** delete the `REDUCE` PTX module entirely
from `fuel-cuda-kernels/src/reduce.cu` + `lib.rs::Id::Reduce`.

## Gap 6 — Composed ops in storage.rs (Softmax / LogSoftmax / RmsNorm / Rope)

**Today:** baracuda ships `softmax_*`, `log_softmax_*`, `rms_norm_*`,
`rope_*` symbol families (all four fp dtypes, contig + strided). Fuel
storage.rs uses the legacy `REDUCE` PTX kernels for these via op-
specific argument packing (`rmsnorm_f32_noalpha`, `rope_f32`, etc.).

**No ask needed.** These migrations are pure Fuel-side work — rewire
the storage.rs calls to baracuda's typed FFI (same pattern as Phase 1b
Pool / 5b Conv / 6c.2 Affine). Just noting them here for completeness.

## Summary

5 baracuda asks (Gaps 1–5), 1 pure Fuel-side migration block (Gap 6).
Closing all five unblocks the rest of Phase 6c.2 (storage.rs eager
API migration) + Phase 6c.3 (drop the AFFINE / UNARY / BINARY /
CAST / REDUCE / INDEXING / TERNARY / FILL / SORT PTX modules
entirely) + Phase 7 (retire `fuel-cuda-kernels` crate from the
workspace; drop the `cudaforge` build-time CUDA compilation
dependency).

Phase 6c.1 + 6c.2 (Affine) already shipped on the Fuel side; the
migration pattern is established (call baracuda raw FFI directly
from the typed `CudaSlice<T>` Map1/Map2 paths, with contig vs
strided dispatch and dtype-name dispatch via match). New symbols
land → Fuel mirrors the pattern in subsequent commits.

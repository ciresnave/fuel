# Vulkan kernel inventory — `fuel-vulkan-kernels` + `fuel-vulkan-backend`

Scope: every distinct compute kernel Fuel ships in its own Vulkan stack.
Kernel **sources** live in `fuel-kernels-source/kernels/*.{slang,glsl}`; they are
AOT-compiled to SPIR-V committed in `fuel-vulkan-kernels/spv/*.spv` and registered
in the `EMBEDDED` table at `fuel-vulkan-kernels/src/lib.rs:39`. The Rust dispatch
wrappers (param packing, layout gating, route picking, validation) live in
`fuel-vulkan-backend/src/lib.rs`; pipeline objects in
`fuel-vulkan-backend/src/pipelines.rs`.

Conventions used below:
- **dtype-monomorphized families** (f32 / f16 / bf16 / f64) are listed as ONE entry
  with a dtype list, since they share an algorithm. Genuinely different kernels
  (e.g. coop-matrix vs scalar matmul) are separate entries.
- **byte-width-keyed families** (`*_b1/b2/b4/b8`) are dtype-agnostic data movers
  keyed by element size (b1=1B u8/i8; b2=2B f16/bf16/i16/u16; b4=4B f32/i32/u32;
  b8=8B f64/i64). One entry, byte-widths noted.
- "Output always contiguous" is the near-universal rule: nearly every kernel writes
  its output via the linear dispatch index. Inputs vary (contiguous-only vs
  per-dim-stride vs offset-capable) — captured per entry.

A recurring layout idiom: elementwise/movement kernels carry a rank-4 (shape0..3,
strides0..3) Params block + a `flags` contiguity bit. When the contiguous flag is
set they index linearly; otherwise they decompose the linear out-index into rank-4
coords and apply per-input strides (stride=0 ⇒ broadcast). These are **strided +
broadcast capable but NOT non-zero-offset capable** (offset handled by an upstream
Contiguize, except `strided_copy*` which take an explicit `src_offset`).

---

## Element-wise unary / binary / affine

### `unary` / `unary_f16` / `unary_f64` / `unary_bf16`
- **op_kind:** elementwise unary, 16-op uniform selector (Neg, Sqr, Sqrt, Exp, Log,
  Sin, Cos, Tanh, Sigmoid, Silu, Gelu(tanh approx), Relu, Step, Abs, Sign, Recip).
- **dtypes:** f32 (`unary`), f16 (native `float16_t`), f64 (native double), bf16
  (packed u16 pairs, math at f32). bf16 op list has the same surface.
- **input layout:** `unary`/`f16`/`f64` are **strided + broadcast** capable (rank-4
  shape/strides + `flags` bit0=contiguous fast path). `unary_bf16` is
  **contiguous-only**, pair-thread (`n_pairs = n/2`, `n` must be even).
- **op_params:** `out_size`, `op_id`, `rank`, `flags`, shape0..3, in_s0..3.
  (bf16: `n_pairs`, `op_id` only.)
- **output:** dtype = input dtype; shape = input shape; always contiguous; not in-place.
- **precision:** f32 math everywhere; bf16/f16 narrow on store (bf16 RNE upper-16,
  canonical qNaN). Gelu = tanh approximation (NOT erf).
- **source:** `fuel-kernels-source/kernels/unary.slang:63`; `unary_bf16.slang:71`;
  wrappers `fuel-vulkan-backend/src/lib.rs:9527` (f32), `:8664` (f16), `:8680` (f64),
  `:8777` (bf16).

### `binary` / `binary_f16` / `binary_f64` / `binary_bf16`
- **op_kind:** elementwise binary, 6-op selector (Add, Sub, Mul, Div, Max, Min).
- **dtypes:** f32, f16 (native), f64 (native), bf16 (packed u16 pairs, f32 math).
- **input layout:** per-operand **strided + broadcast** (rank-4 a_s0..3 / b_s0..3,
  `flags` bit0=a_contig bit1=b_contig; both-contig fast path). bf16 strided path
  reads single lanes by masking the packed u32; contiguous path reads u32 pairs;
  `out_size` must be even (wrapper pads odd).
- **op_params:** `out_size`, `op_id`, `rank`, `flags`, shape0..3, a_s0..3, b_s0..3.
- **output:** dtype = operand dtype; shape = broadcasted output shape; contiguous.
- **precision:** f32 math; bf16/f16 narrow on store.
- **source:** `binary.slang:44`; `binary_bf16.slang:70`; wrappers
  `fuel-vulkan-backend/src/lib.rs:1722` (f32), `:1591` (f16), `:1609` (f64),
  `:8836` (bf16); shared `binary_typed_bytes` `:1628`.

### `affine` / `affine_f16` / `affine_f64` / `affine_bf16`
- **op_kind:** affine `y = x*mul + add` (backs AddScalar / MulScalar / Affine).
- **dtypes:** f32, f16 (native, f32 math), f64 (native double), bf16 (packed u32
  pair-thread).
- **input layout:** f32/f16/f64 **strided + broadcast** (affine Params shape +
  `flags` bit0). `affine_bf16` is **contiguous-only** (pair-thread, packed-u32).
- **op_params:** `out_size`, `flags`, `mul` (f), `add` (f), shape0..3, in_s0..3.
- **output:** dtype=input; shape=input; contiguous.
- **precision:** f32 math; narrow on store for half types.
- **source:** `affine.slang:22`; wrappers `:3711`/`build_affine_f32_dispatch:3648`
  (f32), `:3501` (f16), `:3421` (f64), `:3583` (bf16).

### `clamp`
- **op_kind:** elementwise `clamp(x, lo, hi)`.
- **dtypes:** f32 only.
- **input layout:** **strided + broadcast** (affine Params shape, `flags` bit0).
- **op_params:** `out_size`, `flags`, `lo`, `hi`, shape0..3, in_s0..3.
- **output:** f32; input shape; contiguous.
- **source:** `clamp.slang:22`.

### `powi`
- **op_kind:** elementwise integer power `y = x^exp` (special-cased e=0/1/2/3, else `pow`).
- **dtypes:** f32 only.
- **input layout:** **strided + broadcast** (affine Params shape, `flags` bit0).
- **op_params:** `out_size`, `flags`, `exp` (i32), shape0..3, in_s0..3.
- **output:** f32; input shape; contiguous. `pow(0,-k)→+inf` matches CPU.
- **source:** `powi.slang:27`.

### `add_assign_scaled`
- **op_kind:** in-place scaled accumulate `dst[i] += src[i]*scale`.
- **dtypes:** f32 only.
- **input layout:** **contiguous-only**, element-aligned 1:1 (no stride/shape).
- **op_params:** `n`, `scale` (f).
- **output:** **in-place on `dst`** (binding 0 is RW dst, binding 1 src); aliases dst;
  contiguous.
- **source:** `add_assign_scaled.slang:15`.

---

## Casts

### `cast_f32_to_f16` / `cast_f16_to_f32` / `cast_f32_to_bf16` / `cast_bf16_to_f32`
- **op_kind:** dtype cast (pack/unpack to/from packed-u32 half types).
- **dtypes:** the named src→dst pair.
- **input layout:** **contiguous-only**, pair-packed. `f32→f16`/`→bf16` need `n` even
  (one thread per output u32 = 2 elems). f32↔f16 via `f32tof16`/`f16tof32` (RNE);
  f32↔bf16 by bit shift (f32→bf16 truncate `bits>>16`... wrapper note says
  truncate-toward-zero; bf16→f32 exact `bits<<16`).
- **op_params:** `n`, `_pad`.
- **output:** dst dtype; same logical shape; contiguous.
- **source:** `cast_f32_to_f16.slang:16`; wrapper `cast_f32_bytes` `:2460`.

### `cast_f32_to_f64` / `cast_f64_to_f32`
- **op_kind:** widening/narrowing cast, one thread per element (NOT packed).
- **dtypes:** f32↔f64.
- **input layout:** **contiguous-only**, 1:1.
- **op_params:** `n`.
- **output:** dst dtype; same shape; contiguous. f64→f32 RNE.
- **source:** wrappers `cast_f32_f64_bytes` `fuel-vulkan-backend/src/lib.rs:2558`.

### `cast_f32_to_f8e4m3` / `cast_f8e4m3_to_f32` / `cast_f16_to_f8e4m3` / `cast_f8e4m3_to_f16` / `cast_bf16_to_f8e4m3` / `cast_f8e4m3_to_bf16`
- **op_kind:** F8E4M3 (1-byte float) casts; all non-f8 sides routed via f32.
- **dtypes:** F8E4M3 ↔ {f32, f16, bf16}.
- **input layout:** **contiguous-only**, byte-packed (F8 is 1 byte, 4-per-u32).
- **op_params:** count/pad (per cast variant).
- **output:** dst dtype; same shape; contiguous. f32→F8E4M3 RNE, saturate to ±448.
- **source:** SPIR-V only (`cast_*_f8e4m3.spv`); wrapper `cast_f8e4m3_bytes`
  `fuel-vulkan-backend/src/lib.rs:9451`.

---

## Layout / data-movement (byte-width-keyed, dtype-agnostic)

### `strided_copy`
- **op_kind:** Contiguize / gather-copy (permute, broadcast, slice, concat-via-offset).
- **dtypes:** f32 buffer typed but byte-pattern-agnostic for same-width (4B).
- **input layout:** **strided + broadcast + non-zero src_offset capable** (unsigned
  strides). Reads `out_size` elems via `shape_strides` (shape[0..rank]+stride[0..rank])
  starting at `src_offset`; writes contiguously from `dst_offset`.
- **op_params:** `out_size`, `rank`, `src_offset` (u32), `dst_offset` (u32);
  + `shape_strides` storage buffer.
- **output:** same dtype; contiguous from dst_offset.
- **source:** `strided_copy.slang:26`.

### `strided_copy_signed_b2` / `_b4` / `_b8`
- **op_kind:** Contiguize for **negative-stride** views (Flip/Roll/layout-on-Node).
- **dtypes:** byte-width-keyed (b2/b4/b8); reads as `uint`.
- **input layout:** **signed strides + signed src_offset capable** (strides read as
  i32 via `asint`; `src_offset` is i32 — base may be the LAST element).
- **op_params:** `out_size`, `rank`, `src_offset` (i32), `dst_offset` (u32);
  + `shape_strides` buffer.
- **output:** same width; contiguous from dst_offset.
- **source:** `strided_copy_signed_b4.slang:26`; wrapper `strided_copy_signed_bytes`
  `fuel-vulkan-backend/src/lib.rs:9366`. (No b1 variant.)

### `flip_b2` / `_b4` / `_b8`
- **op_kind:** reverse along one axis (flat outer×dim×inner view).
- **dtypes:** byte-width-keyed (reads as u32 words; no math).
- **input layout:** **strided** (rank-4 shape + per-input strides; output contiguous
  over input shape). One axis reversed via coord arithmetic.
- **op_params:** `out_size`, `axis` (0..3), pad, shape0..3, in_s0..3.
- **output:** same width; input shape; contiguous.
- **source:** `flip_b4.slang:20`; wrapper `flip_bytes` `:9020`. (No b1.)

### `roll_b2` / `_b4` / `_b8`
- **op_kind:** cyclic shift along one axis.
- **dtypes:** byte-width-keyed (u32 word move).
- **input layout:** **strided** (rank-4 shape + strides). `offset` pre-normalized
  to `[0, shape[axis])`; `src_coord=(out_coord+offset) % shape[axis]`.
- **op_params:** `out_size`, `axis`, `offset`, pad, shape0..3, in_s0..3.
- **output:** same width; input shape; contiguous.
- **source:** `roll_b4.slang:25`; wrapper `roll_bytes` `:9111`. (No b1.)

### `triu_b2` / `_b4` / `_b8` and `tril_b2` / `_b4` / `_b8`
- **op_kind:** upper/lower triangular masking on the last 2 dims (keep input or
  emit zero). Two entry points (`triu_b4`, `tril_b4`) share one Slang file
  `triangular_b4.slang`.
- **dtypes:** byte-width-keyed (reads as u32; zero write is dtype-agnostic).
- **input layout:** **contiguous-only**, 1:1 (`output[tid] = keep ? input[tid] : 0`).
- **op_params:** `batch_count`, `rows`, `cols`, `diagonal` (i32). Predicate triu:
  `j >= i+diag`; tril: `j <= i+diag`.
- **output:** same width; input shape; contiguous.
- **source:** `triangular_b4.slang:44/48`. (No b1.)

### `write_slice_b1` / `_b2` / `_b4` / `_b8`
- **op_kind:** in-place rectangular slab write into a larger dst (Op::WriteSlice;
  KV-cache writes).
- **dtypes:** byte-width-keyed.
- **input layout:** src **contiguous** in its own rank-N shape; dst **contiguous**
  larger shape; writes at per-axis `range_start`. Rank ≤ 8. b2 requires last-dim
  range_start & src_shape even; b1 requires multiples of 4 (sub-u32 alignment) —
  wrapper bails to CPU otherwise (`fuel-vulkan-backend/src/lib.rs:2359`).
- **op_params:** `n_src`, `rank`; + `shape_buf` = src_shape+dst_shape+range_start.
- **output:** **in-place on dst** (aliases dst); dst contiguous.
- **source:** `write_slice_b4.slang:29`; wrapper `write_slice_bytes` `:2305`.

### `concat_along_dim` / `_f16` / `_bf16` / `_f64`
- **op_kind:** single-dispatch 2-input concat along an arbitrary dim.
- **dtypes:** f32, f16 (native), bf16 (packed-u32, InterlockedOr half-word writes;
  wrapper zero-fills output), f64 (native double).
- **input layout:** per-operand **strided + broadcast** (rank-4 a_s0..3/b_s0..3; either
  side may be a lazy view). bf16 single-thread-per-bf16 to handle (a,b) boundary.
- **op_params:** out_d0..3, `concat_dim` (0..3), `a_dim`, `b_dim`, `total`,
  a_s0..3, b_s0..3.
- **output:** dtype=operand; out_d[concat_dim]=a_dim+b_dim; contiguous.
- **source:** `concat_along_dim.slang:40`; wrapper `concat_along_dim_f32_bytes`
  `:5159`, typed `concat_along_dim_typed_bytes_with_bind` `:7489`.

---

## Indexing / scatter / gather

### `index_select` / `_f16` / `_bf16` / `_f64`
- **op_kind:** row-wise lookup along a dim (embedding lookup). Flattened to
  [outer, axis, inner].
- **dtypes:** f32, f16 (data move), bf16 (packed-u32 pair-thread, requires
  `inner % 2 == 0`), f64.
- **input layout:** input **contiguous** (flattened outer/axis/inner); `ids` U32.
  Out-of-range index clamped to `axis_in-1`.
- **op_params:** `out_size`, `outer`, `axis_out`(=len ids), `inner`, `axis_in`, pad×3.
- **output:** dtype=input; `dim` resized to len(ids); contiguous.
- **source:** `index_select.slang:30`; `index_select_bf16.slang:28`; wrappers
  `index_select_f32_bytes` `:8025`, typed `index_select_typed_bytes` `:8172`.

### `gather_b1` / `_b2` / `_b4` / `_b8`
- **op_kind:** gather along `dim` with a per-output index tensor.
- **dtypes:** byte-width-keyed (reads as u32 words).
- **input layout:** src + output agree on all dims except `dim`; index U32 (shape =
  output). src indexed via `shape_buf` = [src_shape, out_shape]. Rank ≤ 8.
  **contiguous** src/out (computed strides). No bounds clamp.
- **op_params:** `n_out`, `rank`, `dim`, pad; + `shape_buf` storage.
- **output:** same width; output shape; contiguous.
- **source:** `gather_b4.slang:29`; wrapper `gather_bytes` `:5546`.

### `scatter_add_f32` / `_f64` / `_bf16` / `_f16`
- **op_kind:** scatter-add along `dim` (`out[dst] += src[p]`).
- **dtypes:** f32 (uint CAS), f64 (u64 CAS, needs shaderInt64+atomics+f64),
  bf16 / f16 (sub-word CAS, math f32).
- **input layout:** src **contiguous** (rank-N via `shape_buf`=[src_shape, base_shape]);
  index U32 (src_shape). Rank-N. **Output must be pre-initialized to base by the
  wrapper** (kernel only accumulates).
- **op_params:** `n_src`, `rank`, `dim`, pad; + `shape_buf`.
- **output:** dtype=src; base shape; contiguous; **read-modify-write / atomic
  accumulate** (aliases the base copy). Bounded CAS (1000 iters) under extreme
  contention may drop a value.
- **source:** `scatter_add_f32.slang:46`; wrappers `:5965` (f32), `:5676` (f64),
  `scatter_add_subword_bytes` `:5843` (bf16/f16).

### `index_add_f32` / `_f64` / `_bf16` / `_f16`
- **op_kind:** index-add along an axis (`base[..,idx[k],..] += src[..,k,..]`).
- **dtypes:** f32 (uint CAS), f64 (u64 CAS), bf16/f16 (sub-word CAS, f32 math).
- **input layout:** flattened [outer, axis, inner]; src **contiguous**; rank-1 indices
  (U32). Same CAS atomic-add primitive as scatter_add. **Wrapper copies base→out
  first.**
- **op_params:** `outer_count`, `base_dim_size`, `n_indices`, `inner_count`.
- **output:** dtype=src; base shape; contiguous; atomic accumulate.
- **source:** `index_add_f32.slang:33`; wrappers `:6176` (f32) … `index_add_bytes_impl`
  `:6263`.

---

## Reductions / arg-reductions / norms

### `reduce` / `reduce_f16` / `reduce_bf16` / `reduce_f64`
- **op_kind:** full-tensor reduction to a scalar; op_id 0=sum 1=max 2=min 3=mean.
- **dtypes:** f32, f16 (f32 accum), bf16 (packed-u32 lane-pair, `n` must be even,
  single bf16 output in low 16 bits of output[0]), f64.
- **input layout:** **contiguous-only** (linear strided walk over `n`).
- **op_params:** `n`, `op_id`.
- **output:** dtype=input; shape=scalar (1 elem); contiguous.
- **precision:** f32 accumulator for half types; tree reduction (256 shared slots).
- **source:** `reduce.slang:44`; wrappers `reduce_f32_bytes` `:7613` … `:7926` (f64).

### `reduce_last_dim` / `_f16` / `_bf16` / `_f64`
- **op_kind:** per-row reduction along the last dim; op_id 0=sum 1=max 2=min 3=mean.
- **dtypes:** f32, f16 (f32 accum), bf16 (packed-u32 lane-pair input; **output buffer
  must be zero-initialized** — kernel uses InterlockedOr to write one bf16 half-word
  per row without racing), f64.
- **input layout:** **contiguous** [n_rows, n_cols], row-major. Subgroup reduction.
- **op_params:** `n_rows`, `n_cols`, `op_id`, pad.
- **output:** dtype=input; shape [n_rows]; contiguous.
- **source:** `reduce_last_dim.slang:56`.

### `arg_reduce_last_dim_f32` / `_f16` / `_bf16` / `_f64`
- **op_kind:** argmax/argmin along the last dim; op_id 0=argmax 1=argmin.
- **dtypes:** value f32/f16/bf16/f64; **lower index wins on ties** (numpy/PyTorch).
- **input layout:** **contiguous** [n_rows, n_cols]; per-row workgroup tree reduction
  on (val,idx) pairs. bf16/f16 lane-select from packed input.
- **op_params:** `n_rows`, `n_cols`, `op_id`, pad.
- **output:** **U32 indices**, shape [n_rows]; contiguous.
- **source:** `arg_reduce_last_dim_f32.slang:27`; wrapper `arg_reduce_last_dim_bytes`
  `:6096`.

### `arg_reduce_any_dim_f32` / `_f64` / `_bf16` / `_f16`
- **op_kind:** argmax/argmin along an arbitrary (non-last) dim (slow path).
- **dtypes:** value f32/f64/bf16/f16; lower-index tie-break.
- **input layout:** **strided over the reduction dim** — one thread per output elem,
  serial scan with stride `n_inner`. Logical [n_outer, d_dim, n_inner].
- **op_params:** `n_outer`, `n_inner`, `d_dim`, `op_id`.
- **output:** **U32 indices**; output drops `dim`; contiguous.
- **source:** `arg_reduce_any_dim_f32.slang:31`; wrapper `arg_reduce_any_dim_bytes`
  `:6372`.

### `softmax` / `_f16` / `_bf16` / `_f64`
- **op_kind:** fused last-dim softmax `exp(x-max)/Σexp`.
- **dtypes:** f32, f16 (f32 intermediate), bf16 (packed-u32 lane-pair, `n_cols` even),
  f64 (native, GLSL.std.450 Exp).
- **input layout:** **contiguous** [n_rows, n_cols], one workgroup/row, subgroup
  reductions.
- **op_params:** `n_rows`, `n_cols`.
- **output:** dtype=input; same shape; contiguous.
- **source:** `softmax.slang:32`; wrappers `softmax_last_dim_f32_bytes` `:1845` …
  `:2006` (f64).

### `softmax_last_dim_backward` / `_f16` / `_bf16` / `_f64`
- **op_kind:** fused softmax backward `dx = y*(g - dot(y,g))`.
- **dtypes:** f32, f16 (f32 dot), bf16 (packed-u32 pair-thread, `n_cols` even, no
  race), f64.
- **input layout:** two **contiguous** inputs (y, g) [n_rows, n_cols].
- **op_params:** `n_rows`, `n_cols`.
- **output:** dx; same shape; contiguous.
- **source:** `softmax_last_dim_backward.slang:35`; wrapper `:7265`, typed `:7347`.

### `rms_norm_last_dim` / `_f16` / `_bf16` / `_f64`
- **op_kind:** fused RMSNorm `x / sqrt(mean(x²)+eps)`.
- **dtypes:** f32, f16 (f32 accum), bf16 (packed-u32 lane-pair, `n_cols` even),
  f64 (native, GLSL.std.450 Sqrt).
- **input layout:** **contiguous** [n_rows, n_cols]; one workgroup/row; subgroup
  reduction. No weight (pure normalization).
- **op_params:** `n_rows`, `n_cols`, `eps` (f), pad.
- **output:** dtype=input; same shape; contiguous.
- **source:** `rms_norm_last_dim.slang:41`; wrappers `:2058` … `:2239` (f64).

### `rms_norm_last_dim_backward`
- **op_kind:** fused RMSNorm backward (closed form; 2 reductions sum(x²), sum(g·x)).
- **dtypes:** f32 only.
- **input layout:** two **contiguous** inputs (x, upstream g_y) [n_rows, n_cols].
- **op_params:** `n_rows`, `n_cols`, `eps`, pad.
- **output:** grad_x; same shape; contiguous.
- **source:** `rms_norm_last_dim_backward.slang:68`.

### `layer_norm_last_dim` / `_f16` / `_bf16` / `_f64`
- **op_kind:** fused LayerNorm `(x-mean)/sqrt(var+eps)` (no affine weight/bias).
- **dtypes:** f32, f16 (f32 reductions), bf16 (packed-u32), f64.
- **input layout:** **contiguous** [n_rows, n_cols]; one workgroup/row; two subgroup
  reductions (mean, var).
- **op_params:** `n_rows`, `n_cols`, `eps`, pad.
- **output:** dtype=input; same shape; contiguous.
- **source:** `layer_norm_last_dim.slang:27`; wrappers `:5408` … typed `:5480`.

### `layer_norm_last_dim_backward` / `_f16` / `_bf16` / `_f64`
- **op_kind:** fused LayerNorm backward (4 reductions: sum_x, sum_x², sum_g, sum_gx).
- **dtypes:** f32, f16 (f32 reductions), bf16, f64.
- **input layout:** two **contiguous** inputs (x, upstream g) [n_rows, n_cols].
- **op_params:** `n_rows`, `n_cols`, `eps`, pad.
- **output:** dx; same shape; contiguous.
- **source:** `layer_norm_last_dim_backward.slang:48`; wrappers `:5266` … typed `:5339`.

### `cumsum_f32` / `_f64` / `_f16` / `_bf16`
- **op_kind:** inclusive prefix sum (cumulative sum) along one axis.
- **dtypes:** f32, f64, f16, bf16 (per-dtype because the accumulator needs typed add).
- **input layout:** **strided** input (rank-4 shape + per-input strides); one thread
  per slice, serial walk along axis. Output contiguous over input shape.
- **op_params:** `slice_count`, `axis`, `dim_size`, pad, shape0..3, in_s0..3.
- **output:** dtype=input; input shape; contiguous. f32 accumulator.
- **source:** `cumsum_f32.slang:27`; wrappers `:9213` … typed `cumsum_typed_bytes`
  `:9277`.

---

## Padding (forward + backward)

### `pad_const_b1` / `_b2` / `_b4` / `_b8`
- **op_kind:** constant-fill pad.
- **dtypes:** byte-width-keyed (fill passed as u32 bit pattern).
- **input layout:** src **contiguous** in in_shape; out **contiguous** in out_shape.
  Rank ≤ 8. `shape_buf`=[in_shape, out_shape, left_pad].
- **op_params:** `n_out`, `rank`, `fill_value` (u32), pad.
- **output:** same width; out_shape; contiguous.
- **source:** `pad_const_b4.slang:25`; wrapper `pad_const_bytes` `:7116`.

### `pad_reflect_b1` / `_b2` / `_b4` / `_b8`
- **op_kind:** reflect-pad (no-repeat, PyTorch "reflect").
- **dtypes:** byte-width-keyed.
- **input layout:** src/out **contiguous**; `shape_buf`=[in_shape, out_shape, left_pad].
  **PRECONDITION:** per-axis left_pad ≤ in_dim-1 AND right_pad ≤ in_dim-1.
- **op_params:** `n_out`, `rank`, pad×2.
- **output:** same width; out_shape; contiguous.
- **source:** `pad_reflect_b4.slang:42`; wrapper `pad_reflect_bytes` `:6835`.

### `pad_replicate_b1` / `_b2` / `_b4` / `_b8`
- **op_kind:** replicate (edge-repeat) pad; out-of-range coord clamps to [0,in_dim-1].
  No precondition on pad sizes.
- **dtypes:** byte-width-keyed.
- **input layout:** src/out **contiguous**; same `shape_buf` layout.
- **op_params:** `n_out`, `rank`, pad×2.
- **output:** same width; out_shape; contiguous.
- **source:** SPIR-V `pad_replicate_b4.spv`; wrapper `pad_replicate_bytes` `:6710`.

### `pad_backward_const_b1` / `_b2` / `_b4` / `_b8`
- **op_kind:** Pad-const backward — one thread per input elem reads grad_out at
  `in_coord+left_pad` (no accumulation).
- **dtypes:** byte-width-keyed.
- **input layout:** grad_out/grad_in **contiguous**; `shape_buf`=[in_shape, out_shape,
  left_pad].
- **op_params:** `n_in`, `rank`, pad×2.
- **output:** grad_in; in_shape; contiguous.
- **source:** `pad_backward_const_b4.slang:28`; wrapper `pad_backward_const_bytes`
  `:6594`.

### `pad_backward_reflect_f32/f64/f16/bf16` and `pad_backward_replicate_f32/f64/f16/bf16`
- **op_kind:** Pad reflect/replicate backward — one thread per OUTPUT elem,
  atomic-accumulate grad_out into grad_in (forward maps multiple out→same in).
- **dtypes:** **dtype-specific** (NOT byte-keyed): f32 (uint CAS), f64 (u64 CAS),
  f16/bf16 (sub-word CAS). Math at f32 for half types.
- **input layout:** grad_out **contiguous**; grad_in **contiguous** and **must be
  zero-filled by the wrapper before dispatch**. `shape_buf`=[in_shape,out_shape,
  left_pad].
- **op_params:** `n_out`, `rank`, pad×2.
- **output:** grad_in; in_shape; contiguous; **atomic accumulate** (CAS).
- **source:** `pad_backward_reflect_f32.slang:51`; wrapper `pad_backward_atomic_bytes`
  `:6462`.

### `masked_fill_b1` / `_b2` / `_b4` / `_b8`
- **op_kind:** masked fill — `out = mask!=0 ? fill : input`.
- **dtypes:** byte-width-keyed; **mask always U8** (4 bytes packed per u32 word).
- **input layout:** **contiguous** 1:1 input + mask (b4 mask word indexed `i>>2`;
  b1/b2 pack element + mask).
- **op_params:** `n`, `fill_value` (bit pattern of the element width).
- **output:** same width; input shape; contiguous.
- **source:** `masked_fill_b4.slang:16`; wrapper `masked_fill_bytes` `:6982`.

---

## Matmul / matvec family

### `matmul` (WGSL-origin Slang, 4×4 register tile)
- **op_kind:** batched GEMM `C = A@B`, register-tiled (no shared mem). Picker uses it
  for 1 < m < 32.
- **dtypes:** f32 × f32 → f32.
- **input layout:** **stride-capable, offset-incapable** — A/B addressed via
  sa_batch/sa_row/sa_col, sb_* (any row/col strides ⇒ transpose-friendly); batch
  base only, no element offset. GQA via `n_rep` (`b_off=(batch/n_rep)*sb_batch`).
  Out-of-range guarded.
- **op_params:** M,N,K, sa_batch/row/col, sb_batch/row/col, sc_batch, n_rep, pad.
- **output:** f32; C[batch, M, N] row-major contiguous (`r*N+c`).
- **source:** `matmul.slang:26`; wrapper `matmul_f32_bytes` `:3759` (route picker
  `:3830`).

### `matmul_tiled` (GLSL, 64×64 shared-mem tile, BK=16)
- **op_kind:** batched GEMM, shared-memory blocked. Picker uses it for m ≥ 32.
- **dtypes:** f32 × f32 → f32.
- **input layout:** same stride model as `matmul` (sa_*/sb_* strides; batch base).
- **op_params:** identical MatmulParams.
- **output:** f32; contiguous row-major.
- **source:** `matmul_tiled.glsl:39`.

### `matvec` (GLSL gemv, M==1)
- **op_kind:** gemv specialization (M==1); subgroup-reduced dot, one workgroup/col.
- **dtypes:** f32 × f32 → f32.
- **input layout:** **stride-aware** (A via sa_col; B via sb_row/sb_col), batch base;
  permute/transpose-friendly.
- **op_params:** MatmulParams (M,N,K + strides + n_rep).
- **output:** f32; C[batch, N]; contiguous.
- **source:** `matvec.glsl:26`; selected by `matmul_f32_bytes` when m==1.

### `matvec_bf16_b` (GLSL, mixed precision gemv)
- **op_kind:** gemv (M==1), f32 A × **bf16 B** → f32 C (decode hot path for bf16 weights).
- **dtypes:** A f32, B bf16 (packed 2-per-u32), C f32.
- **input layout:** stride-aware; B unpacked by bit-shift (`bits<<16`, exact f32
  extension); `sb_batch` is in bf16 elements, u32 base = half.
- **op_params:** MatmulParams.
- **output:** f32; C[N]; contiguous.
- **source:** `matvec_bf16_b.glsl:53`; wrapper `matmul_f32_bf16_b_bytes` `:2644`.

### `matmul_tiled_bf16_b` (GLSL)
- **op_kind:** tiled GEMM (m>1), f32 A × bf16 B → f32 C; bf16 unpacked on B load.
- **dtypes:** A f32, B bf16, C f32.
- **input layout:** same tiling/stride model as `matmul_tiled`.
- **output:** f32; contiguous.
- **source:** SPIR-V `matmul_tiled_bf16_b.spv`; wrapper `matmul_f32_bf16_b_bytes` `:2644`.

### `matmul_coop` (GLSL cooperative-matrix / tensor-core)
- **op_kind:** coop-matrix GEMM, f32 A × bf16 B → f32 C; A,B downcast to f16 on
  shared load, f32 accumulator (coop shape M=N=K=16). Dispatched only when
  VK_KHR_cooperative_matrix present (pipeline is `Option`).
- **dtypes:** A f32, B bf16, C f32 (internal f16 inputs).
- **input layout:** stride-aware via Params; needs the coop-tile shape constraints
  (picker `matmul_coop_ok` `:2824`).
- **op_params:** MatmulParams.
- **output:** f32; contiguous; **precision:** inputs lose to f16, accumulation f32.
- **source:** `matmul_coop.glsl:54`.

### `matmul_coop_bf16_bf16` / `matmul_coop_f16_f16`
- **op_kind:** coop-matrix GEMM → **f32** output. bf16×bf16 (both downcast bf16→f16
  on load) or f16×f16 (native). f32 accumulator. `Option` pipelines.
- **dtypes:** {bf16, f16} × same → f32.
- **input layout:** stride-aware; coop-tile constraints apply.
- **output:** f32; contiguous.
- **source:** `matmul_coop_bf16_bf16.glsl`, `matmul_coop_f16_f16.glsl`; wrappers
  `matmul_bf16_bf16_f32_bytes` `:2790`, `matmul_f16_f16_f32_bytes` `:3118`
  (`matmul_half_half_f32_coop_bytes` `:3154`).

### `matmul_coop_bf16_bf16_bf16` / `matmul_coop_f16_f16_f16`
- **op_kind:** coop-matrix GEMM with **downcast store** → bf16 / f16 output (closes the
  half-precision inference chain). f32 accumulator staged to shared mem, then packed.
  `Option` pipelines.
- **dtypes:** {bf16,f16} × same → same.
- **input layout:** stride-aware; coop constraints.
- **output:** bf16 (packed-u32) / f16; contiguous; narrow on store.
- **source:** `matmul_coop_bf16_bf16_bf16.glsl`, `matmul_coop_f16_f16_f16.glsl`;
  wrappers `matmul_bf16_bf16_bf16_bytes` `:2936`, `matmul_f16_f16_f16_bytes` `:3085`
  (`matmul_half_half_half_coop_bytes` `:2972`).

### `matmul_small_bf16_bf16_f32` / `matmul_small_bf16_bf16_bf16` / `matmul_small_f16_f16_f32` / `matmul_small_f16_f16_f16`
- **op_kind:** scalar-accumulator GEMM fallback (one thread per output elem); handles
  ANY shape when coop-tile constraints fail (m<16, m%16≠0, n%16≠0, matvec).
- **dtypes:** {bf16,f16} × same → {f32 | same(downcast store)}. f32 accumulator.
- **input layout:** **stride-aware** (sa_*/sb_*); 16×16 workgroup, grid ceil(N/16)×
  ceil(M/16)×batch. bf16/f16 unpacked per-load (`uint16_t` typed buffers).
- **op_params:** MatmulParams.
- **output:** f32 or {bf16,f16}; contiguous row-major.
- **source:** `matmul_small_bf16_bf16_f32.glsl:32`; shared `matmul_small_half_inner`
  `:2834`.

---

## Quantized (GGML) family

### `dequant_q4_0`
- **op_kind:** GGML Q4_0 block dequant → f32 (18-byte/32-elem blocks; `(nibble-8)*d`).
- **dtypes:** input raw bytes (ByteAddressBuffer); output f32.
- **input layout:** byte stream, **contiguous** blocks; one thread per (k,k+16) pair;
  unaligned reads via `load_u8` word-extract.
- **op_params:** `n_blocks`, `out_elements`, pad×2.
- **output:** f32; `n_blocks*32` elems; contiguous.
- **source:** `dequant_q4_0.slang:38`; wrapper `dequantize_q4_0` `:9616`.

### `dequant_q8_0`
- **op_kind:** GGML Q8_0 block dequant → f32 (34-byte/32-elem blocks; `qs*d`).
- **dtypes:** raw bytes → f32.
- **input layout:** byte stream, contiguous; one thread per output elem.
- **op_params:** `n_blocks`, `out_elements`, pad×2.
- **output:** f32; contiguous.
- **source:** `dequant_q8_0.slang:34`; wrapper `dequantize_q8_0` `:9656`,
  `_from_storage` `:10025`.

### `dequant_q4_km`
- **op_kind:** GGML Q4_K_M super-block dequant → f32 (144-byte/256-elem; 6-bit packed
  scales+mins, llama.cpp `get_scale_min_k4`).
- **dtypes:** raw bytes → f32.
- **input layout:** byte stream, contiguous; one workgroup/super-block, 32 threads,
  8 elems/thread.
- **op_params:** `n_blocks`, `out_elements`, pad×2.
- **output:** f32; `n_blocks*256`; contiguous.
- **source:** `dequant_q4_km.slang:75`; wrapper `dequantize_q4_km` `:10160`.

### `qmatvec_q4_0`
- **op_kind:** fused Q4_0×F32 gemv (decode hot path, M==1). `out[n]=Σ A[k]·dequant(W)`.
- **dtypes:** A f32, W Q4_0 bytes ([N, K/32] blocks), C f32.
- **input layout:** A **contiguous** [K]; W byte stream [N,K/32]; one workgroup/col,
  128 threads, subgroup reduction. K must be multiple of 32.
- **op_params:** `n`, `k`, `blocks_per_row`(=K/32), pad.
- **output:** f32; [N]; contiguous.
- **source:** `qmatvec_q4_0.slang:79`; wrapper `qmatvec_q4_0` `:10064`,
  `_slice` `:10105`; routed by `matmul_q4_0_bytes` `:4001` for M==1.

### `matmul_q4_0_tiled`
- **op_kind:** fused Q4_0×F32 tiled matmul (prefill, M>1); TM=8 m-rows/tile, WG=128.
- **dtypes:** A f32, W Q4_0 bytes [N,K/32], C f32.
- **input layout:** A **contiguous** [M,K]; W byte stream; one workgroup per
  (m_tile, n_col).
- **op_params:** `m`, `n`, `k`, `blocks_per_row`.
- **output:** f32; [M,N] row-major; contiguous.
- **source:** `matmul_q4_0_tiled.slang:68`; wrapper `matmul_q4_0_tiled` `:10199`,
  `matmul_q4_0_bytes` `:4001` for M>1.

### `quantize_q8_0`
- **op_kind:** F32 → GGML Q8_0 quantize (KV-cache compression). `d=max|x|/127`.
- **dtypes:** f32 → Q8_0 bytes (34-byte/32-elem).
- **input layout:** src f32 **contiguous**, `n_elements` multiple of 32; one thread
  per block (serial 32-elem), byte writes via InterlockedXor RMW (boundary-word safe).
- **op_params:** `n_elements`, `n_blocks`, pad×2.
- **output:** byte stream Q8_0; contiguous.
- **source:** `quantize_q8_0.slang:76`; wrapper `quantize_q8_0` `:10243`.

---

## Conv

### `conv2d_im2col` / `conv2d_im2col_bf16`
- **op_kind:** im2col patch rearrangement (NCHW → patches matrix that feeds matmul).
  Supports groups + asymmetric stride/padding; zero-fill out-of-bounds.
- **dtypes:** f32 (`conv2d_im2col`), bf16 (`_bf16`, pairs with coop bf16 matmul).
- **input layout:** x **contiguous** NCHW [batch,c_in,h,w]; one thread per patches elem.
- **op_params:** batch, c_in, h, w, h_out, w_out, k_h, k_w, stride_h/w, pad_h/w,
  groups, cin_per_g, total_elements, pad.
- **output:** patches [batch*groups, cin_per_g*k_h*k_w, h_out*w_out]; contiguous.
  (f16 path `conv2d_f16_bytes` `:4992` builds on these.)
- **source:** `conv2d_im2col.slang:57`; wrappers `conv2d_f32_bytes` `:4226`,
  `conv2d_bf16_bytes` `:4819`.

---

## Attention

### `flash_attn_f32` / `flash_attn_bf16` / `flash_attn_f16`
- **op_kind:** **naive single-pass** multi-head attention forward (NOT tiled online
  softmax) — materializes one [Sk] score row in shared mem per (b,h,q_i) workgroup.
  Supports GQA, causal, softmax_scale, alibi. Sk ≤ 4096, D ≤ 256. Window + softcap
  bail to other backends.
- **dtypes:** f32; bf16 (in/out, f32 accum); f16 (native in/out, f32 accum).
- **input layout:** Q[B,Hq,Sq,D], K/V[B,Hkv,Sk,D], O[B,Hq,Sq,D], all **contiguous
  NCHW-like**; optional alibi[Hq] (dummy buffer when absent). GQA `kv_h=hi/(Hq/Hkv)`.
- **op_params:** B,Hq,Hkv,Sq,Sk,D, softmax_scale, causal, use_alibi, pad×3.
- **output:** dtype=input; O same shape as Q; contiguous; all-masked rows → 0.
- **source:** `flash_attn_f32.glsl:59`; wrappers `flash_attn_f32_bytes` `:4601`,
  `flash_attn_bf16_bytes` `:4627`, `flash_attn_f16_bytes` `:4653`
  (`flash_attn_bytes_impl` `:4682`).

### `flash_attention` (Slang FlashAttention-2)
- **op_kind:** **tiled** scaled-dot-product attention forward with **online softmax**
  (BR=BC=16). Supports GQA, causal, sliding window (L/R), ALiBi, softcap. head_dim
  ≤ 128 (D_MAX).
- **dtypes:** f32.
- **input layout:** Q/K/V/O **contiguous** [B,H,S,D]; alibi[Hq] optional (5 storage +
  uniform, `layout_5s1u`). Grid (B, Hq, ceil(Sq/16)).
- **op_params:** b,hq,hkv,sq,sk,d,groups,causal, window_left/right + has_* flags,
  has_alibi, has_softcap, softmax_scale, softcap.
- **output:** f32; O same shape as Q; contiguous.
- **source:** `flash_attention.slang:80`.

### `flash_attn_backward_q_f32` / `_k_f32` / `_v_f32`
- **op_kind:** FlashAttention backward (dQ / dK / dV), f32. dQ: one workgroup per
  (b,h_q,q_i), recompute softmax+dP+dS in shared mem. dK/dV: one workgroup per
  (b,h_kv,k_j) looping (h_q in group, q_i). Supports GQA+causal+scale+alibi; bails
  on window/softcap/Sk>4096/D>256.
- **dtypes:** f32.
- **input layout:** Q,K,V,dO **contiguous** [B,H,S,D]; alibi[Hq] (6 storage + uniform,
  `layout_6s1u`). dQ all-masked rows zeroed.
- **op_params:** B,Hq,Hkv,Sq,Sk,D, softmax_scale, causal, use_alibi, pad×3.
- **output:** dQ / dK / dV; same shape as Q / K / V; contiguous.
- **source:** `flash_attn_backward_q_f32.glsl:51`; wrappers `:4384` (q), `:4412` (k),
  `:4440` (v) (`flash_attn_backward_bytes_impl` `:4470`).

---

## RoPE

### `rope` / `rope_f16` / `rope_bf16` / `rope_f64`
- **op_kind:** fused rotary position embedding (rotate_half convention). One thread
  per (o,s,i), writes positions i and i+h.
- **dtypes:** f32; f16 (f32 rotation math); bf16 (packed-u32 pair-thread, 2 u32 words /
  4 bf16 positions, requires `head_dim % 4 == 0`); f64 (native).
- **input layout:** **x strided-capable** (contiguous fast path via `x_contiguous`,
  else per-dim strides x_s0/x_s1/x_s_seq/x_s_hd; supports a lazy [0,2,1,3] permute).
  cos/sin tables **always contiguous** [seq, head_dim]. Output always contiguous.
- **op_params:** outer, seq, head_dim, total, x_s0, x_s1, x_s_seq, x_s_hd, x_outer1,
  x_contiguous, pad×2.
- **output:** dtype=input; x shape; contiguous.
- **source:** `rope.slang:44`; wrappers `rope_f32_bytes` `:8258` … typed
  `rope_typed_bytes` `:8543`.

---

## Notes / cross-cutting contracts

- **Output contiguity is universal.** Every kernel writes its output linearly; none
  emit a strided/offset output. Any input offset is handled by an upstream Contiguize
  except `strided_copy`/`strided_copy_signed` which take an explicit `src_offset`.
- **Strided-input families** (no Contiguize needed): `unary`/`binary`/`affine`/`clamp`/
  `powi` (f32/f16/f64), `cumsum`, `flip`, `roll`, `concat_along_dim`, `rope` (on x),
  `matmul`/`matvec` family (row/col strides), `arg_reduce_any_dim`, `strided_copy*`.
  `*_bf16` elementwise/affine variants are mostly **contiguous-only** (packed-u32).
- **Contiguous-only families:** all reductions/norms (`reduce*`, `*_last_dim*`,
  softmax, rms/layer norm + backwards), `index_select`, `gather`, `scatter_add`,
  `index_add`, `pad*`, `masked_fill`, `triu/tril`, `write_slice`, casts, conv2d im2col,
  flash attention (all), all quant kernels.
- **Atomic-accumulate kernels** (output must be pre-initialized/zeroed by the wrapper):
  `scatter_add_*`, `index_add_*`, `pad_backward_reflect/replicate_*`. f32 = uint CAS,
  f64 = u64 CAS, bf16/f16 = sub-word CAS; bounded 1000-iter CAS loop.
- **Half-precision packing:** bf16 stored as packed-u16-in-u32; many bf16 kernels
  require even counts (`n`/`n_cols`/`inner % 2`) and some require `% 4` (rope_bf16).
  bf16↔f32 is exact extension `bits<<16`; f32→bf16 is RNE upper-16 with canonical
  qNaN. f16 uses `f32tof16`/`f16tof32` (RNE) or native `float16_t`.
- **`Option` (capability-gated) pipelines:** the 6 `matmul_coop*` variants are only
  built/dispatched when `VK_KHR_cooperative_matrix` is present (`has_coop_matrix`).
- **Gelu** is the **tanh approximation** in `unary`/`unary_bf16`, not erf.
- **Embedded but no Slang source in tree** (SPIR-V committed only):
  `pad_replicate_b*`, `matmul_tiled_bf16_b`, `cast_*_f8e4m3` — contracts read from
  the Rust wrappers + `EMBEDDED` doc comments.

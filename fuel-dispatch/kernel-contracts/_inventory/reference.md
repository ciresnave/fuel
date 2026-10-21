# Kernel inventory — `fuel-reference-backend`

> Source-of-truth crate: **`fuel-reference-backend`** (the pure-Rust, correctness-first
> oracle). This document enumerates every distinct kernel/op the crate provides in
> `src/ops.rs`, `src/attention.rs`, and the executor bridge `src/exec.rs`.
>
> **Generated 2026-06-17.** Drives a per-kernel contract for every entry.

## How to read this inventory

- **Crate-wide layout invariant.** `RefTensor<T>` (`src/lib.rs:68`) is *always* a
  contiguous, row-major `Vec`/`Arc<[T]>` plus a `Shape`. It carries **no strides**
  and **no offset**. Therefore every kernel below is, by construction,
  **contiguous-only with zero offset** at the data layer. There is no
  `is_contiguous()` branch, no `StridedIndex`, no strided/broadcast/offset
  input path anywhere — callers must materialize any non-contiguous view into a
  fresh contiguous `RefTensor` *before* calling. The "input_layouts" field in each
  entry reflects this; where an op computes broadcast/stride *math internally over a
  contiguous buffer* (e.g. `broadcast_to`, `reduce_sum_to`, `broadcast_add`) that is
  noted explicitly, but the input buffer itself is still contiguous.
- **Dtype generalization.** Most numeric kernels are generic over
  `T: num_traits::Float` and are monomorphized by the executor to `{f32, f64, bf16, f16}`
  (one kernel, dtype list). Shape-only / copy-only kernels relax the bound to
  `T: Clone`/`T: Copy + Default` and also accept `u32` (index tensors). These are
  listed as a single kernel with the full dtype list.
- **Output is always a fresh contiguous tensor** unless noted. `RefTensor::from_vec`
  / `from_arc` is the universal exit; reshape/broadcast-pure-pad share the input
  `Arc` (zero-copy metadata-only) but still present as contiguous.
- **No production path.** The crate is an oracle. `exec.rs` panics/`unreachable!`s on
  in-place ops, U8-output comparisons, `Where`, `MaskedFill`, `WriteSlice*`, `Alloc`,
  `ZeroFill`, `View*`, `ScatterIntoSlot` — these are pipelined-executor-only and have
  **no reference kernel** (documented at the bottom).

---

## Unary elementwise (generic `T: Float`, dtypes f32/f64/bf16/f16)

All of these: input contiguous zero-offset; output same shape, same dtype, fresh
contiguous buffer; pure `iter().map()` over the flat slice. No broadcasting, no
strides, no in-place.

| Kernel | Formula / notes | Source |
|---|---|---|
| `neg` | `-x` | `ops.rs:30` |
| `relu` | `max(0,x)`; `x>0 ? x : 0` (note: exactly 0 returns 0) | `ops.rs:36` |
| `sqr` | `x*x` | `ops.rs:47` |
| `sqrt` | `x.sqrt()` | `ops.rs:53` |
| `exp` | `e^x` | `ops.rs:59` |
| `sign` | `-1/0/+1`; `0` at exactly 0 (and at `-0.0`) | `ops.rs:65` |
| `log` | `ln(x)`; non-positive → IEEE NaN/-inf (passthrough) | `ops.rs:87` |
| `sin` | `sin(x)` | `ops.rs:93` |
| `cos` | `cos(x)` | `ops.rs:99` |
| `abs` | `|x|` | `ops.rs:105` |
| `recip` | `1/x`; `1/0 → inf` (IEEE) | `ops.rs:112` |
| `tanh` | `tanh(x)` | `ops.rs:119` |
| `floor` | `floor(x)` | `ops.rs:125` |
| `ceil` | `ceil(x)` | `ops.rs:131` |
| `erf` | `erf(x)` via **widen-to-f64 → libm::erf → narrow**; ≤1 ULP. Precision-sensitive for bf16/f16. | `ops.rs:141` |
| `gelu_erf` | `0.5*x*(1+erf(x/√2))` exact; computed in **f64** then narrowed (PyTorch `approximate='none'`) | `ops.rs:160` |
| `round` | round-half-to-**even** (banker's); overrides `Float::round` only at exact .5 ties | `ops.rs:180` |
| `sigmoid` | logistic, numerically-stable split form (branch on `x>=0`) | `ops.rs:214` |
| `silu` | `x*sigmoid(x)`; computes `sigmoid` then elementwise-mul (two passes) | `ops.rs:234` |
| `gelu` | tanh approximation `0.5*x*(1+tanh(√(2/π)(x+0.044715x³)))`; constants via `cst` in dtype `T` | `ops.rs:244` |
| `step` | Heaviside `x>0 ? 1 : 0` (0 at exactly 0); subgradient of relu | `ops.rs:264` |
| `rsqrt` | `1/sqrt(x)` single op (RMSNorm pattern) | `ops.rs:1520` |
| `powi` | `x^n`, scalar `i32` exponent uniform; `Float::powi` (repeated mul, handles neg/zero base) | `ops.rs:1511` |

---

## Binary elementwise — same-shape (generic `T: Float`)

`assert_same_shape` (`ops.rs:277`) — **exact dims equality, NO broadcasting**.
Input contiguous zero-offset; output = lhs shape, same dtype, fresh buffer.
`zip` over the two flat slices.

| Kernel | Formula | Source |
|---|---|---|
| `add` | `a+b` | `ops.rs:291` |
| `sub` | `a-b` | `ops.rs:303` |
| `mul` | `a*b` | `ops.rs:315` |
| `div` | `a/b` | `ops.rs:327` |
| `maximum` | `max(a,b)` | `ops.rs:1822` |
| `minimum` | `min(a,b)` | `ops.rs:1834` |
| `rem` | PyTorch remainder `a - floor(a/b)*b` (sign of divisor) | `ops.rs:1771` |
| `pow` | `a.powf(b)` (real exponent, elementwise); IEEE NaN rules | `ops.rs:1788` |

> `rem`/`pow` use a local `assert_eq` on dims rather than the shared `assert_same_shape`,
> same semantics (exact equality, no broadcast).

---

## Binary elementwise — broadcasting (generic `T: Float`)

`broadcast_binary` (`ops.rs:2142`) drives these via NumPy broadcast rules
(`broadcast_shape` `ops.rs:2094`, `broadcast_src_flat` `ops.rs:2121`). **The input
buffers are still contiguous zero-offset**; the broadcast/stride math is computed
*internally* (right-align, pad with 1s, size-1 dim → coord 0) over those contiguous
buffers via `row_major_strides`. Output shape = broadcast shape; fresh buffer;
per-output-element unflatten → two source flats. Panics on incompatible shapes at
build of `broadcast_shape`.

| Kernel | Formula | Source |
|---|---|---|
| `broadcast_add` | `a+b` (NumPy broadcast) | `ops.rs:2179` |
| `broadcast_sub` | `a-b` (NumPy broadcast) | `ops.rs:2184` |
| `broadcast_mul` | `a*b` (NumPy broadcast) | `ops.rs:2189` |
| `broadcast_div` | `a/b` (NumPy broadcast) | `ops.rs:2194` |

---

## Scalar-by-tensor / clamp (generic `T: Float`)

Scalar passed as `f64`, coerced to `T` via `cst`. Input contiguous; output same
shape/dtype; fresh buffer.

| Kernel | Formula / params | Source |
|---|---|---|
| `add_scalar` | `x + c` (`c: f64`) | `ops.rs:1495` |
| `mul_scalar` | `x * c` (`c: f64`) | `ops.rs:1502` |
| `clamp` | clamp to `[min,max]` (`min,max: f64`) | `ops.rs:1802` |

---

## Reductions to scalar (generic `T: Float`)

Input contiguous; output is a **rank-0** tensor (`Shape::from_dims(&[])`, 1 element),
same dtype. Fresh buffer.

| Kernel | Identity / empty behavior | Source |
|---|---|---|
| `sum_all` | fold-add from `T::zero()` | `ops.rs:341` |
| `max_all` | empty → `-inf` | `ops.rs:352` |
| `min_all` | empty → `+inf` | `ops.rs:364` |
| `mean_all` | empty → `NaN`; else `sum/n` | `ops.rs:376` |

---

## Reductions along one dim (generic `T: Float`)

`reduce_dim` (`ops.rs:409`) — reduced dim **removed** from output (no keepdim).
Input contiguous; uses `row_major_strides` + per-element unflatten internally; output
contiguous, same dtype, shape = input with `dim` dropped. `assert dim < rank`.

| Kernel | Reduction / identity | Source |
|---|---|---|
| `sum_dim` | `+`, init `0` | `ops.rs:467` |
| `max_dim` | `max`, init `-inf` | `ops.rs:473` |
| `min_dim` | `min`, init `+inf` | `ops.rs:479` |
| `mean_dim` | `sum_dim / dims[dim]` (two passes) | `ops.rs:485` |

### Integer-producing reductions (generic input `T: Float` → output `u32`)

`argindex_dim` (`ops.rs:506`). **Output dtype = `u32`** (differs from input). Reduced
dim removed. Ties → **smallest index** (PyTorch). Input contiguous; output contiguous.

| Kernel | Source |
|---|---|
| `argmax_dim` | `ops.rs:495` |
| `argmin_dim` | `ops.rs:501` |

---

## Reduce-to-shape (broadcast inverses; generic `T: Float`)

Input contiguous; broadcast-alignment math internal (right-align, size-1 dim collapse).
Target shape must be broadcast-compatible into source. Output = target shape, fresh
contiguous buffer.

| Kernel | Behavior | Source |
|---|---|---|
| `reduce_sum_to` | sum-reduce to target (backward of `broadcast_to`); init `0` | `ops.rs:854` |
| `reduce_max_to` | max-reduce to target; init `-inf` | `ops.rs:911` |
| `reduce_max_to_backward` | routes upstream to argmax positions, **fair-share split on ties**; recomputes forward max, builds mask, counts ties, `count_safe=max(count,1)`. Output = input shape `S_in`. Takes `(x, upstream, target)`. | `ops.rs:979` |

---

## Broadcast forward (generic `T: Float`)

`broadcast_to` (`ops.rs:1045`) — NumPy broadcast to a target shape. **Pure-padding
fast path** (`ops.rs:1076`): when the source matches its aligned target dims with
only size-1 leading padding, returns `from_arc(x.as_arc().clone(), target)` —
**zero-copy, shares the input `Arc`, output aliases input storage** (immutable share).
Otherwise allocates a fresh buffer and fills via per-element unflatten. Source must be
contiguous; output contiguous; same dtype. Note: executor `eval_broadcast_to` panics
on U32 input.

---

## Matmul / linear algebra (generic `T: Float`)

| Kernel | Op kind | Layout / params | Output | Source |
|---|---|---|---|---|
| `matmul` | N-D batched matmul | `a=[...batch,m,k]`, `b=[...batch,k,n]`; **batch prefix must match exactly (no batch broadcast)**; rank≥2, equal rank. Contiguous. Rank-2 defers to `matmul_2d`. | `[...batch,m,n]`, same dtype, fresh contiguous. Naive triple loop, `T`-precision accumulator (no f32-accum widening). | `ops.rs:578` |
| `matmul_2d` | rank-2 matmul | `[m,k]·[k,n]`, contiguous | `[m,n]` | `ops.rs:651` |
| `transpose_2d` | rank-2 transpose | `[m,n]→[n,m]`, contiguous; physically reorders | fresh contiguous | `ops.rs:2457` |
| `transpose_last_two` | swap last two dims, rank≥2 | leading dims batched; rank-2 defers to `transpose_2d`; contiguous | `[...,n,m]` fresh contiguous | `ops.rs:2420` |
| `permute` | N-D axis permutation (`T: Clone+Default`, incl. u32) | `axes` must be a permutation of `0..rank`; contiguous; physically reorders to row-major | fresh contiguous, dtype unchanged | `ops.rs:2363` |

> Executor extra: `eval_matmul` (`exec.rs:1264`) adds a **mixed-precision arm**:
> `f32 activations × bf16 weights → f32` by upcasting B to f32 (exact) then f32 matmul.
> No standalone kernel — it reuses `ops::matmul`.

---

## Convolution / pooling (generic `T: Float`)

| Kernel | Op kind | Layout / params | Output | Source |
|---|---|---|---|---|
| `conv2d` | 2-D conv (production, registry `CONV2D`) | `x=[N,Cin,H,W]`, `weight=[Cout,Cin/groups,Kh,Kw]`, optional `bias=[Cout]`; `stride/padding=(h,w)`, `groups`. Symmetric zero-pad. Delegates to `fuel_conv::conv2d_direct`. Contiguous. | `[N,Cout,Hout,Wout]`, same dtype, fresh. Exec dtypes: **f32/f64 only** (`exec.rs:1291`). | `ops.rs:704` |
| `conv_transpose2d` | 2-D transposed conv (registry `CONV_TRANSPOSE2D`) | `x=[N,Cin,H,W]`, `weight=[Cin,Cout/groups,Kh,Kw]`; `stride/padding/output_padding/dilation=(h,w)`, `groups`. Scatter form. Contiguous. | `[N,Cout,Hout,Wout]`, same dtype, fresh. Exec dtypes: **f32/f64 only** (`exec.rs:1322`). | `ops.rs:751` |
| `conv2d_simple` | 2-D conv (legacy, no bias/groups) | `x=[N,Cin,H,W]`, `kernel=[Cout,Cin,kH,kW]`; scalar `stride`, `padding`. Contiguous. **Not wired in exec** (test-only). | `[N,Cout,Hout,Wout]` | `ops.rs:2209` |
| `max_pool2d` | 2-D max pool (no padding) | `x=[N,C,H,W]`; scalar `kernel_size`, `stride`. Contiguous. **Not wired in exec** (test-only). | `[N,C,Hout,Wout]` | `ops.rs:2301` |

---

## Normalization & attention compositions (generic `T: Float`)

Forward and backward. Input contiguous; per-row loop along last dim (row_count =
product of leading dims). Output same shape/dtype, fresh contiguous.

| Kernel | Op kind | Params / notes | Source |
|---|---|---|---|
| `softmax_last_dim` | softmax over last dim | stable (subtract row max). Registry `SOFTMAX_LAST_DIM`. | `ops.rs:2484` |
| `softmax_last_dim_backward` | softmax bwd | inputs `(y=forward out, g=upstream)`; `dx = y*(g - sum(y*g))`. Registry `SOFTMAX_LAST_DIM_BACKWARD`. | `ops.rs:2529` |
| `log_softmax_last_dim` | log-softmax | stable form. `Op::LogSoftmaxLastDim`. | `ops.rs:2895` |
| `log_softmax_last_dim_backward` | log-softmax bwd | `(y,g)`; `dx = g - exp(y)*sum(g)`. `Op::LogSoftmaxLastDimBackward`. | `ops.rs:2926` |
| `layer_norm_last_dim` | LayerNorm (no affine) | `eps: f64`; **biased variance (/n)**, PyTorch-matching. Registry `LAYER_NORM_LAST_DIM`. | `ops.rs:2646` |
| `layer_norm_last_dim_backward` | LayerNorm bwd | `(x,g,eps)`; `dx = rstd*(g - mean(g) - y*mean(g*y))`. Registry `LAYER_NORM_LAST_DIM_BACKWARD`. | `ops.rs:2577` |
| `rms_norm_last_dim` | RMSNorm (no affine) | `eps: f64`; `y = x/sqrt(mean(x²)+eps)`. Registry `RMS_NORM_LAST_DIM`. | `ops.rs:2699` |
| `rms_norm_last_dim_backward` | RMSNorm bwd | `(x,g,eps)`; closed-form fused gradient. Registry `RMS_NORM_LAST_DIM_BACKWARD`. | `ops.rs:2749` |
| `rope` | rotary position embedding | `x=[...,seq,head_dim]` (head_dim even), `cos/sin=[seq,head_dim]` broadcast over leading dims (asserted exact `[seq,head_dim]`). Registry `ROPE`. | `ops.rs:2801` |

---

## Masking / triangular (mixed bounds)

| Kernel | Op kind | Bound / dtypes | Layout / params | Output | Source |
|---|---|---|---|---|---|
| `triu` | upper-tri mask | `T: Copy + Default` → f32/f64/bf16/f16/**u32** | last-two-dims, leading batched; `diagonal: i64`; keep `j >= i+diag` else 0. Contiguous. | same shape/dtype, fresh | `ops.rs:2847` |
| `tril` | lower-tri mask | `T: Copy + Default` (incl. u32) | keep `j <= i+diag` else 0 | same shape/dtype | `ops.rs:2871` |
| `masked_fill` | masked fill | `T: Copy` data + `mask: RefTensor<u8>` + `value: T` | shapes must match exactly; `out=mask!=0?value:x`. **NOT wired in exec** (executor panics: U8 mask unsupported, `exec.rs:457`). | same shape/dtype | `ops.rs:2953` |

---

## Padding (generic `T: Float`)

`padding: &[(before,after)]` per axis, `padding.len()==rank`. Input contiguous;
output = per-axis expanded dims; fresh contiguous.

| Kernel | Op kind | Params / notes | Source |
|---|---|---|---|
| `pad_const` | constant pad | `value: f64` → `T`; pre-fill then copy interior. `Op::Pad{Constant}`. | `ops.rs:1532` |
| `pad_reflect` | reflect pad | per-axis `before/after <= n-1`. `Op::Pad{Reflect}`. | `ops.rs:1570` |
| `pad_replicate` | replicate pad | edge clamp. `Op::Pad{Replicate}`. | `ops.rs:1582` |
| `pad_backward` | pad bwd (all 3 modes) | `(grad_out, in_shape, padding, mode_tag: 0/1/2)`; **f64 accumulator** then narrow to `T`. Output = `in_shape`. `Op::PadBackward`. | `ops.rs:1634` |

---

## Sequence / shape movement (mixed bounds; include u32)

| Kernel | Op kind | Bound / dtypes | Layout / params | Output | Source |
|---|---|---|---|---|---|
| `reshape` | reshape | `T: Clone` (incl. u32) | contiguous; **metadata-only, shares input `Arc` (zero-copy, output aliases input storage)**; elem-count must match | target shape, same dtype | `ops.rs:829` |
| `cumsum` | cumulative sum along dim | `T: Float` (no u32) | contiguous outer×d×inner walk; `Op::CumSum{dim}` | same shape/dtype, fresh | `ops.rs:1696` |
| `flip` | reverse along dim | `T: Copy + Default` (incl. u32) | contiguous; `copy_from_slice` rows; `Op::Flip{dim}` | same shape/dtype, fresh | `ops.rs:1720` |
| `roll` | cyclic shift along dim | `T: Copy + Default` (incl. u32) | `shift: i64` (wraps, rem_euclid); `Op::Roll{dim,shift}` | same shape/dtype, fresh | `ops.rs:1742` |
| `concat` | concat two tensors along dim | `T: Clone + Default` (incl. u32) | both contiguous, same rank, equal in all non-`dim` dims; **2-input only**; `Op::Concat{dim}` | `dim` summed, fresh | `ops.rs:1398` |
| `slice` | narrow along dim | `T: Clone + Default` (incl. u32) | `start,len`; `start+len <= dim size`; contiguous; `Op::Slice{dim,start,len}` | `dim`→`len`, fresh | `ops.rs:1453` |

> Executor `Op::Contiguize` → `eval_reshape` (no-op metadata copy);
> `Op::Unsqueeze{dim}` / `Op::Squeeze{dim}` → reshape with an inserted/removed size-1
> axis (`exec.rs:1103`/`1140`). No dedicated kernels — they reuse `ops::reshape`.

---

## Indexing / gather / scatter (mixed bounds; index operand integer)

Input data contiguous; index operand is a contiguous integer tensor.
`row_major_strides` + per-element unflatten internally. Output contiguous.

| Kernel | Op kind | Data dtypes / index | Layout / params | Output | Source |
|---|---|---|---|---|---|
| `index_select_tensor` | index-select along dim | data `T: Clone+Default` (f32/f64/bf16/f16/u32); index `I: PrimInt` (u32) | **index tensor must be rank-1**; bounds-checked. `Op::IndexSelect{dim}`. | shape = data with `dim`→`indices.len()`, same dtype | `ops.rs:1254` |
| `index_select` | index-select via `&[usize]` | `T: Float` | `&[usize]` indices (not a tensor); bounds-checked. Underlying impl re-used by tensor variant. Not directly exec-wired (exec uses `index_select_tensor`). | shape = data with `dim`→`indices.len()` | `ops.rs:2004` |
| `gather` | N-D gather (PyTorch) | data `T: Clone+Default`; index `I: PrimInt` (u32) | **index same rank as data**; output shape == index shape; per-element `dim`-coord replaced by index value; bounds-checked. `Op::Gather{dim}`. | shape = index shape, data dtype | `ops.rs:1339` |
| `index_add` | functional index-add | `T: Clone+Add`; index `I: PrimInt` (u32) | base & src same rank, non-`dim` dims match; **rank-1 index**, `src[dim]==len(index)`. Copies base then `out[...,idx[i],...] += src[...,i,...]`. `Op::IndexAdd{dim}`. | base shape, data dtype | `ops.rs:1855` |
| `scatter_add` | functional scatter-add | `T: Clone+Add`; index `I: PrimInt` (u32) | **index same shape as src**; copies base then `out[p with dim←idx[p]] += src[p]`. `Op::ScatterAdd{dim}`. | base shape, data dtype | `ops.rs:1934` |
| `embedding` | embedding lookup | `T: Float` | `table=[V,D]` rank-2, `ids: &[usize]`; bounds-checked; `copy_from_slice` rows. Not directly exec-wired (no `Op::Embedding` arm; modeled as index_select/gather). | `[ids.len(), D]` | `ops.rs:2065` |

---

## Dtype casts (concrete typed kernels — NOT generic)

Each is a distinct concrete-typed function (one src→dst pair). Input contiguous;
output = same shape, **target dtype**, fresh buffer. Routed by `eval_cast`
(`exec.rs:1000`). Identity casts are clones (zero arithmetic). bf16↔f16 route via f32.

| Kernel | Cast | Precision | Source |
|---|---|---|---|
| `cast_f32_to_f64` | f32→f64 | lossless | `ops.rs:1135` |
| `cast_f32_to_bf16` | f32→bf16 | round-to-nearest | `ops.rs:1141` |
| `cast_f32_to_f16` | f32→f16 | round-to-nearest | `ops.rs:1147` |
| `cast_f64_to_f32` | f64→f32 | lossy out-of-range | `ops.rs:1153` |
| `cast_f64_to_bf16` | f64→bf16 | lossy | `ops.rs:1159` |
| `cast_f64_to_f16` | f64→f16 | lossy | `ops.rs:1165` |
| `cast_bf16_to_f32` | bf16→f32 | lossless | `ops.rs:1171` |
| `cast_bf16_to_f64` | bf16→f64 | lossless | `ops.rs:1177` |
| `cast_bf16_to_f16` | bf16→f16 | lossy (via f32) | `ops.rs:1184` |
| `cast_f16_to_f32` | f16→f32 | lossless | `ops.rs:1194` |
| `cast_f16_to_f64` | f16→f64 | lossless | `ops.rs:1200` |
| `cast_f16_to_bf16` | f16→bf16 | lossy (via f32) | `ops.rs:1206` |
| `cast_u32_to_f32` | u32→f32 | exact below 2^24 | `ops.rs:1218` |
| `cast_u32_to_f64` | u32→f64 | lossless | `ops.rs:1224` |
| `cast_f32_to_u32` | f32→u32 | trunc-toward-zero; out-of-range UB | `ops.rs:1232` |
| `cast_f64_to_u32` | f64→u32 | trunc-toward-zero | `ops.rs:1238` |

---

## Attention (`src/attention.rs`, generic `T: Float`)

All shapes `[B,H,S,D]`, batch-first heads-second, **contiguous zero-offset**.
GQA via `Hq` multiple of `Hkv` (broadcast each KV head over the Q group).
`AttentionParams` (`attention.rs:36`): `softmax_scale: f32`, `causal: bool`,
`window_size_left/right: Option<usize>`, `softcap: Option<f32>`. Optional ALiBi
slopes `[Hq]`. Mask admissibility via `position_admissible` (`attention.rs:76`).

| Kernel | Op kind | Inputs / params | Output | Precision / notes | Source |
|---|---|---|---|---|---|
| `attention_naive` | MHSDPA (materializes full `[B,H,Sq,Sk]` matrix) | `q,k,v=[B,H,S,D]`, opt `alibi=[Hq]`, `p`. Registry `FLASH_ATTN` exec arm uses this as the oracle. Exec dtypes **f32/f64**. | `[Bq,Hq,Sq,Dq]`, same dtype, fresh contiguous | stable softmax; fully-masked row → all-zero output | `attention.rs:99` |
| `attention_flash` | FlashAttention-v2 forward (tiled, online softmax, `BR=BC=16`) | same as naive | `[Bq,Hq,Sq,Dq]` | bit-equal to naive up to f32-associativity drift; never materializes attn matrix. Not the exec arm (exec FLASH_ATTN routes to `attention_naive`). | `attention.rs:229` |
| `attention_paged_naive` | paged-cache attention | `q=[B,Hq,Sq,D]`, `k_cache/v_cache=[num_blocks,block_size,Hkv,D]`, `block_table=[B,max_blocks] u32`, `context_lens=[B] u32`, opt `alibi`, `softmax_scale`, `block_size`, `softcap`. Registry `PAGED_ATTN`. Exec dtypes **f32/f64**. | `[B,Hq,Sq,D]` | implicit causal mask tied to `context_lens[b]-Sq+q_pos`; per-seq variable context | `attention.rs:420` |
| `attention_flash_backward` | attention backward via recompute | `q,k,v,do_grad=[B,H,S,D]`, opt `alibi`, `p` | `(dQ=[Bq,Hq,Sq,Dq], dK=[Bq,Hk,Sk,Dq], dV=[Bq,Hk,Sk,Dv])` — **3 outputs**, fresh contiguous, dK/dV summed over GQA groups | recomputes softmax; includes softcap derivative `1-tanh²`. **Not wired in exec** (multi-output; functional-oracle only). | `attention.rs:567` |

---

## Quantized matmul (exec-only, `src/exec.rs`)

| Kernel | Op kind | Inputs / params | Output | Notes | Source |
|---|---|---|---|---|---|
| `eval_qmatmul` + `dequantize_blocks` + `dequantize_q4_km_block` | dequant-then-matmul (registry `QMATMUL`) | activations **F32** `[...,M,K]`; weight bytes **U32-reinterpreted-as-bytes**; `quant_type ∈ {Q4_0, Q8_0, Q4_K_M}`, `k`, `n`. HF weight convention `[N,K]`, transposed to `[K,N]`. | **F32** `[...,M,N]`, fresh contiguous | Dequant must bit-match GPU `dequant_q4_0`/`q8_0`/`q4_km`. Q4_K_M = 144-byte super-block → 256 f32 (llama.cpp `get_scale_min_k4`). Other quant types `unimplemented!`. | `exec.rs:1494` / `1530` / `1583` |

> `eval_fused_linear` (registry `FUSED_LINEAR`, `exec.rs:1394`): reference = `matmul`
> then broadcast-add a rank-1 bias along the last axis. Exec dtypes **f32/f64**.
> No standalone kernel; composes `ops::matmul` + `ops::broadcast_to` + `ops::add`.

---

## Pass-through / inert executor arms (no kernel)

`Op::Copy` / `Op::Move` → clone the cached input (host-only; `exec.rs:785`).
`Op::Release` and `Op::Branch` → return a zero-element F32 marker (`exec.rs:794`,
`942`).

## Ops with NO reference kernel (executor panics / unreachable)

These are **pipelined-executor-only** or unsupported in the dtype-erased
`AnyRefTensor` (which has no U8 variant). They are intentionally *not* kernels of this
crate; listed so the per-kernel contract effort knows to skip them here:

- **U8-output comparisons**: `Equal`, `Ne`, `Lt`, `Le`, `Gt`, `Ge` (`exec.rs:504-533`).
- **Ternary**: `Where` (`exec.rs:534`).
- **Masked fill via U8**: `Op::MaskedFill` (`exec.rs:457`) — note the *kernel*
  `ops::masked_fill` exists but is unreachable through the executor.
- **In-place ops** (`ReluInplace` … `PowIInplace`) — `exec.rs:869`.
- **KV-cache / alloc**: `WriteSlice`, `WriteSliceRotating`, `Alloc`, `ZeroFill`
  (`exec.rs:816-868`).
- **Multi-output projection**: `View`, `ViewOwned`, `ScatterIntoSlot`
  (`exec.rs:907-941`).
- **`Op::Const`** — handled by slot-first dispatch (`try_adopt_slot_ref`,
  `exec.rs:111`), never reaches `eval_node`.

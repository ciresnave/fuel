# Kernel inventory — `fuel-conv` + `fuel-flash-attn-cuda`

Scope: kernels/ops Fuel **itself** provides in the two Fuel-side crates
`fuel-conv` (reference conv primitives) and `fuel-flash-attn-cuda` (the
Fuel-side flash-attention surface). Drives a per-kernel contract for each.

Generated from a full read of:
- `fuel-conv/src/lib.rs`
- `fuel-flash-attn-cuda/src/lib.rs`
- supporting: `fuel-flash-attn-cuda-sys/src/lib.rs` (FFI `run_mha`),
  `fuel-flash-attn-cuda/tests/flash_attn_tests.rs`, both `Cargo.toml`s.

---

## Summary

| # | Kernel / op | Crate | Op kind | Dtypes | Device |
|---|-------------|-------|---------|--------|--------|
| 1 | `conv2d_direct` | fuel-conv | conv2d forward (direct nested loop) | any `num_traits::Float` (f32/f64 in practice) | CPU host slices |
| 2 | `im2col` | fuel-conv | input rearrangement (im2col patch extraction) | any `num_traits::Float` | CPU host slices |
| 3 | `conv2d_via_gemm` | fuel-conv | conv2d forward via im2col + caller gemm | any `num_traits::Float` | CPU host slices |
| 4 | `flash-attn` (`FlashAttn` CustomOp3) | fuel-flash-attn-cuda | scaled dot-product attention, fixed-length (rank-4) | f16, bf16 | CUDA |
| 5 | `flash-attn-varlen` (`FlashAttnVarLen` CustomOp3) | fuel-flash-attn-cuda | scaled dot-product attention, variable-length (rank-3 + cu_seqlens) | f16, bf16 | CUDA |

Notes on counting:
- The free functions `flash_attn`, `flash_attn_windowed`, `flash_attn_alibi`,
  `flash_attn_alibi_windowed`, `flash_attn_alibi_windowed_softcap` are **thin
  wrappers** that only set fields of the same `FlashAttn` op — one kernel.
- Likewise the five `flash_attn_varlen*` free functions all build the same
  `FlashAttnVarLen` op — one kernel.
- Both CUDA ops dispatch into the **same** `run_mha` FFI entry point
  (`fuel-flash-attn-cuda-sys`); they are distinguished here because they are
  genuinely different ops at the Fuel surface (different rank, different
  varlen/cu_seqlens semantics, different `unpadded_lse` flag, different
  softmax_lse allocation) and carry different contracts.
- The f16/bf16 split is dtype monomorphization of one kernel each
  (`cuda_fwd_t::<T>` selected by `q.dtype()`), not separate kernels.

---

## 1. `conv2d_direct`  — fuel-conv/src/lib.rs:137

- **Op kind:** 2D convolution forward, direct textbook nested loop
  (batch × groups × c_out_per_g × h_out × w_out × c_in_per_g × k_h × k_w).
  The parity oracle other backends verify against.
- **Signature:** `conv2d_direct<T: Float>(x, weight, bias: Option, s: &ConvShape, out: &mut [T])`.
- **Dtypes:** generic over `num_traits::Float`. Used with f32 (and works for
  f64); no half/integer support (Float bound excludes them). One generic
  kernel, not monomorphized variants.
- **Input layout handling:** operates on **raw `&[T]` host slices**, NOT
  `Tensor`/`Layout`. Layout is *fixed by convention*, not inspected:
  - x: NCHW `[batch, c_in, h, w]`, **row-major contiguous, zero offset assumed**.
  - weight: `[c_out, c_in/groups, k_h, k_w]`, row-major contiguous.
  - bias (optional): `[c_out]`.
  - No `is_contiguous()` / `StridedIndex` / strided / broadcast / offset
    handling exists — there is no Layout type here. Offsets are computed
    manually as flat row-major indices (`x_off`, `w_off`, lib.rs:174-175).
    A non-contiguous or offset caller would silently produce wrong results;
    only `debug_assert_eq!` length checks guard it (lib.rs:152-155).
- **Op params (`ConvShape`, lib.rs:48):** batch, c_in, c_out, h, w, k_h, k_w,
  stride `(h,w)` (asymmetric OK), padding `(h,w)` (asymmetric, applied to
  both sides), groups. **No dilation** (documented unsupported, lib.rs:40).
- **Output behavior:**
  - dtype rule: same `T` as input (out: `&mut [T]`).
  - shape rule: `[batch, c_out, h_out, w_out]`, row-major contiguous;
    `h_out = (h + 2*pad_h - k_h)/stride_h + 1`, ditto width (lib.rs:75-80).
  - layout guarantee: writes every output element exactly once (no
    accumulation), so **pre-zeroing not required** (lib.rs:136).
  - in-place/aliasing: caller-owned `out` buffer; must be sized
    `s.output_len()`; no aliasing with inputs.
- **Precision:** straight accumulation in `T` (`acc = acc + x*w`), no
  higher-precision accumulator, no Kahan. Deterministic accumulation order
  (channel→ky→kx). Bias added after the reduction.
- **Validation / panic surface:** `s.validate()` is called and `.expect()`ed
  (lib.rs:144) — **panics on a malformed ConvShape** (violates the
  never-panic-on-production-paths rule). Length mismatches are `debug_assert`
  only (no release-mode guard).

## 2. `im2col`  — fuel-conv/src/lib.rs:221

- **Op kind:** input rearrangement — extracts conv patches into the im2col
  matrix so a vendor BLAS gemm can carry the conv arithmetic. The matmul step
  is explicitly *not* in this crate.
- **Signature:** `im2col<T: Float>(x, s: &ConvShape, out: &mut [T])`.
- **Dtypes:** generic over `num_traits::Float` (f32/f64). One generic kernel.
- **Input layout handling:** raw `&[T]` host slice, layout fixed by
  convention: x is NCHW `[batch, c_in, h, w]` **row-major contiguous, zero
  offset assumed**. No Layout/stride/offset/broadcast inspection; flat
  indices computed manually (`x_channel_offset`, lib.rs:249, 265). Padding
  out-of-bounds positions are zero-filled (lib.rs:258-268). Only
  `debug_assert_eq!` length checks (lib.rs:233-234).
- **Op params:** `ConvShape` (same as above). No dilation.
- **Output behavior:**
  - dtype rule: same `T` as input.
  - shape rule: flattened length `s.im2col_len()` =
    `batch * groups * (c_in_per_g * k_h * k_w) * (h_out * w_out)`
    (lib.rs:117-122). Logical layout
    `[batch*groups, c_in_per_g*k_h*k_w, h_out*w_out]`; inner patch-axis
    ordering is `(channel, ky, kx)` to line up with the weight reshape's
    K-dimension (lib.rs:217-220).
  - layout guarantee: row-major contiguous; every element written exactly
    once (padding -> `T::zero()`), no pre-zeroing required.
  - in-place/aliasing: caller-owned `out`; no aliasing with x.
- **Precision:** no arithmetic — pure data movement + zero-fill for padding;
  lossless.
- **Validation / panic surface:** `s.validate().expect(...)` — **panics on
  malformed ConvShape** (lib.rs:226). Length mismatch is `debug_assert` only.

## 3. `conv2d_via_gemm`  — fuel-conv/src/lib.rs:300

- **Op kind:** full conv2d forward = `im2col` + a **caller-provided** gemm
  invoked once per `(batch, group)` pair, then optional bias add. Lets AOCL /
  oneMKL / reference backend plug their own `c = a @ b` without re-writing the
  im2col loop.
- **Signature:** `conv2d_via_gemm<T: Float, F: FnMut(usize,usize,usize,&[T],&[T],&mut [T])>(x, weight, bias: Option, s, out, patches_scratch, gemm)`.
- **Dtypes:** generic over `num_traits::Float`. One generic kernel; gemm
  callback is caller's responsibility (e.g. f32 BLAS).
- **Input layout handling:** raw `&[T]` host slices, fixed-convention layout
  (NCHW x, OIHW weight, `[c_out]` bias) — **contiguous, zero offset assumed**.
  Requires a `patches_scratch` buffer of `s.im2col_len()`. No
  Layout/stride/offset/broadcast handling. Slices into weight/patches/out are
  computed by flat offsets per group (lib.rs:331-338).
- **Op params:** `ConvShape`; gemm contract: `m = cout_per_group`,
  `n = h_out*w_out`, `k = cin_per_group*k_h*k_w`; weight slice `[m,k]`
  row-major, patches slice `[k,n]` row-major, out slice `[m,n]` row-major.
  Gemm must do **`c = a @ b` with no accumulate** (overwrites; caller need not
  pre-zero) — bias add happens here after gemm (lib.rs:293-294, 343-355).
- **Output behavior:**
  - dtype rule: same `T`.
  - shape rule: `[batch, c_out, h_out, w_out]` row-major contiguous;
    out sized `s.output_len()`.
  - layout guarantee: each `(batch,group)` output block `[m,n]` written by the
    gemm callback; bias added in place afterward.
  - in-place/aliasing: `out` and `patches_scratch` caller-owned & distinct;
    correctness of overwrite-vs-accumulate delegated to the gemm callback.
- **Precision:** im2col is lossless; the reduction precision is **whatever the
  caller's gemm does** (not controlled here). Bias add in `T`.
- **Validation / panic surface:** `s.validate().expect(...)` — **panics on
  malformed ConvShape** (lib.rs:312). Bias length is `debug_assert` only.

### Shared `ConvShape` helper — fuel-conv/src/lib.rs:48
Not a kernel; the descriptor all three consume. `validate()` (lib.rs:87)
rejects: groups==0, c_in/c_out not divisible by groups, zero stride, zero
kernel dims, kernel larger than padded input. Returns `Result<(),&'static
str>`. (The three kernels `.expect()` this rather than propagating, hence the
panic surface above.)

---

## 4. `flash-attn` (`FlashAttn`, CustomOp3)  — fuel-flash-attn-cuda/src/lib.rs:8 / op impl :213 / kernel :21

- **Op kind:** FlashAttention-v2 scaled dot-product attention,
  `softmax(Q @ K^T * softmax_scale) @ V`, fixed (padded) batch layout. Supports
  MHA / MQA / GQA (num_heads_k divides num_heads), causal & sliding-window
  masking, ALiBi slopes, Gemma-style softcap. CUDA sm80 (Ampere) only.
  Dispatched via `CustomOp3::fwd` -> `cuda_fwd_t::<T>` -> `ffi::run_mha`.
- **Dtypes:** **f16, bf16 only** (dispatch at lib.rs:242-246; anything else
  bails). Selected by `q.dtype()`; `is_bf16` flag forwarded to the kernel.
  k and v assumed same dtype as q (sliced as `T`). ALiBi slopes, if present,
  must be **F32** (lib.rs:92). softmax_lse scratch is F32.
- **Input layout handling (precise):** inputs are `Tensor`/`Layout` (CUDA
  storage).
  - **Rank must be exactly 4** for q/k/v (`q_rank != 4 ...` bail, lib.rs:55).
  - **Last dim must be contiguous** (`stride[rank-1] == 1`) for q, k, v
    (lib.rs:60-68) — otherwise bail. Other dims may be **strided**: batch /
    row(seq) / head strides are read straight from the Layout and passed to
    the kernel (`q_batch_stride`, `q_row_stride`=stride[rank-3],
    `q_head_stride`=stride[rank-2], lib.rs:176-188). So the op is
    **strided-capable on the outer 3 axes, contiguous-only on the last axis**.
  - **Non-zero offset capable:** each input is sliced from
    `start_offset()..len` (lib.rs:41-43); ALiBi slopes also sliced from its
    start_offset (lib.rs:113). The op does **not** require zero offset.
  - Output layout is forced **contiguous** (`Layout::contiguous(&out_shape)`,
    lib.rs:36) and its strides are passed as the o_* strides.
- **Op params:** softmax_scale (f32); alibi_slopes (Option<Tensor>, shape
  `(num_heads_q)`, F32); window_size_left/right (Option<usize>);
  softcap (Option<f32>, 0.0 disables). Masking semantics: causal =
  (window_right==0 && window_left<0); a one-sided window is expanded to
  seqlen_k on the open side (lib.rs:149-159). Window values > seqlen_k are
  treated as None/-1 (lib.rs:123-134).
  - Shape constraints enforced: q/k/v as `(b, seqlen, num_heads, head_size)`;
    k and v must match `(b, seqlen_k, num_heads_k, head_size)` (lib.rs:70-78);
    `head_size <= 512` (lib.rs:79); `head_size % 8 == 0` (lib.rs:82);
    `num_heads % num_heads_k == 0` (lib.rs:86).
  - Internal rounding: head_size→mult of 8 then 32 (d_rounded), seqlen_q/k→
    mult of 128 (lib.rs:136-139).
- **Output behavior:**
  - dtype rule: **same as q** (T = f16/bf16); `dev.alloc::<T>(elem_count)`
    (lib.rs:142).
  - shape rule: **identical to q's shape** `(b, seqlen_q, num_heads_q,
    head_size)` (out_shape = q_l.shape().clone(), lib.rs:35).
  - layout guarantee: **freshly allocated, contiguous** output buffer
    (lib.rs:36, 208). Not in-place; no aliasing with inputs.
  - side output: `softmax_lse` F32 scratch of size
    `b * 128 * num_heads * seqlen_q` is allocated and written by the kernel but
    **not returned** (lib.rs:143) — fwd returns only `(dst, out_shape)`.
- **Precision:** kernel internals are FlashAttention-v2 (f16/bf16 inputs,
  fp32 accumulation inside the kernel; LSE in fp32). softcap applies a
  tanh-based logit cap before softmax. The Fuel-side test (`flash_attn_tests`)
  validates against an fp32 reference to ~1e-5 (acausal) / 1e-3 (softcap).
- **CPU path:** explicitly unsupported — `fwd` bails if storage is CPU
  (lib.rs:227, "no cpu support for flash-attn"). Wrong dtype bails.
- **Panic surface:** validation uses `fuel::bail!` (Result) — **no panics on
  bad shape/dtype**. (One latent `.unwrap()` on the alibi RwLock read guard,
  lib.rs:101.)

## 5. `flash-attn-varlen` (`FlashAttnVarLen`, CustomOp3)  — fuel-flash-attn-cuda/src/lib.rs:442 / op impl :682 / kernel :455

- **Op kind:** FlashAttention-v2 with **variable-length batching** — packed
  (ragged) sequences indexed by cumulative `seqlens_q`/`seqlens_k`. Same
  attention math, masking, ALiBi, softcap as #4, but no padded batch axis.
  CUDA sm80 only. `CustomOp3::fwd` -> `cuda_fwd_t::<T>` -> `ffi::run_mha` with
  `cu_seqlens_*` pointers and `unpadded_lse = 1`.
- **Dtypes:** **f16, bf16 only** (dispatch lib.rs:711-715). seqlens_q/seqlens_k
  must be **u32** (`as_cuda_slice::<u32>()`, lib.rs:477, 488; passed to FFI as
  `*const i32`). ALiBi slopes (optional) F32. softmax_lse scratch F32.
- **Input layout handling (precise):**
  - q/k/v **rank must be exactly 3** `(total_tokens, num_heads, head_size)`
    (`q_rank != 3` bail, lib.rs:511).
  - **Last dim must be contiguous** (`stride[rank-1]==1`) for q/k/v
    (lib.rs:516-524); outer axes **strided** (row=stride[rank-3],
    head=stride[rank-2] forwarded; batch strides forced to 0 since varlen,
    lib.rs:645-657).
  - **Non-zero offset capable** for q/k/v (sliced start_offset..len,
    lib.rs:497-499) and alibi (lib.rs:580).
  - **seqlens_q / seqlens_k must be contiguous** — uses
    `contiguous_offsets()` and bails if None (lib.rs:478-481, 489-492).
  - Output forced contiguous (`Layout::contiguous`, lib.rs:470).
- **Op params:** softmax_scale; max_seqlen_q, max_seqlen_k (usize, used for
  rounding + window clamping, lib.rs:605-606, 622-625); seqlens_q, seqlens_k
  (Tensor, u32, len = batch_size+1, cumulative offsets); alibi_slopes
  (Option, F32, `(num_heads_q)`); window_size_left/right; softcap.
  - batch_size = `nseqlens_q - 1` (lib.rs:555). Constraints: nseqlens_q >= 2
    (lib.rs:547), nseqlens_q == nseqlens_k (lib.rs:551); head_size <= 512,
    %8==0; num_heads % num_heads_k == 0 (lib.rs:535-544).
  - Masking/window semantics identical to #4 but clamped against
    `max_seqlen_k` (lib.rs:590-626).
- **Output behavior:**
  - dtype rule: same as q (T); `dev.alloc::<T>(elem_count)` (lib.rs:609).
  - shape rule: **identical to q** `(total_q, num_heads_q, head_size)`
    (out_shape = q_l.shape().clone(), lib.rs:469).
  - layout guarantee: freshly allocated, **contiguous**; not in-place, no
    aliasing.
  - side output: `softmax_lse` F32 of size `num_heads * total_q`
    (lib.rs:610) — written but not returned (`unpadded_lse=1`).
- **Precision:** same as #4 (FlashAttention-v2, fp32 internal accumulation /
  LSE).
- **CPU path:** unsupported — bails on CPU storage (lib.rs:696).
- **Panic surface:** uses `fuel::bail!` for validation (no shape/dtype
  panics); latent `.unwrap()` on seqlens/alibi RwLock read guards
  (lib.rs:473, 484, 568).

---

## Cross-cutting notes for the contracts

- **fuel-conv kernels (1-3)** are pure-host reference primitives over `&[T]`
  with **no Tensor/Layout awareness at all** — their "layout contract" is an
  unenforced *assumption* of NCHW/OIHW row-major contiguous, zero-offset
  data. Any contract should state contiguous-only + zero-offset explicitly and
  flag the `validate().expect()` panic as a known violation of the
  never-panic rule. No half/int dtypes (Float bound). No dilation.
- **flash-attn kernels (4-5)** are CUDA-only (`f16`/`bf16`), reject CPU,
  require last-dim-contiguous but accept strided outer axes and non-zero
  offsets, always emit a fresh contiguous output equal in shape to q, and
  internally route to one shared `run_mha` FFI (sys crate). They differ in
  rank (4 vs 3), varlen/cu_seqlens, `unpadded_lse`, and softmax_lse sizing.
- ALiBi slopes and seqlens are auxiliary inputs (F32 and u32 respectively),
  not the primary q/k/v dtype.

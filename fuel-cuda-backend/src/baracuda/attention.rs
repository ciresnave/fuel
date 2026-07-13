//! Attention kernels from `baracuda-kernels-sys`: RoPE, SDPA, and
//! FlashSDPA. Per OP-MATRIX in baracuda alpha.27.
//!
//! ## Coverage today
//!
//! - **RoPE** — wired end-to-end to Fuel's `OpKind::Rope` (single
//!   public function per dtype, dispatched from
//!   `baracuda_dispatch::attention`).
//! - **SDPA** and **FlashSDPA** — kernel wrappers shipped as
//!   utility functions. Fuel's `OpKind::FlashAttn` has a richer
//!   shape (GQA via `hq != hkv`, sliding window, softcap) than
//!   baracuda's FlashSDPA exposes today, so dispatch wiring waits
//!   for either (a) Fuel adding an equal-heads FlashAttn variant
//!   or (b) baracuda adding GQA + window + softcap support.
//!   Either way, the kernel wrappers below are the seam.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_ir::{Error, Layout, Result};

use crate::byte_storage::CudaStorageBytes;

use super::scratch::{RopeTableSlot, Workspace};
use super::status::check;

type RopeRun = unsafe extern "C" fn(
    batch: i32,
    heads: i32,
    seq: i32,
    head_dim: i32,
    base: f32,
    pos_default_flag: i32,
    x: *const std::ffi::c_void,
    positions: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Strided RoPE FFI (alpha.31): adds (b, h, s) input + output strides.
/// Head-dim (innermost) stride is implicit = 1 — enforced at the
/// wrapper layer per the baracuda team's guidance ("RoPE pair dim
/// (head_dim) must stay stride=1, enforced at plan layer").
type RopeStridedRun = unsafe extern "C" fn(
    batch: i32,
    heads: i32,
    seq: i32,
    head_dim: i32,
    stride_x_b: i64, stride_x_h: i64, stride_x_s: i64,
    stride_y_b: i64, stride_y_h: i64, stride_y_s: i64,
    base: f32,
    pos_default_flag: i32,
    x: *const std::ffi::c_void,
    positions: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

fn is_contiguous_zero_offset(layout: &Layout) -> bool {
    layout.start_offset() == 0 && layout.is_contiguous()
}

type FlashSdpaRun = unsafe extern "C" fn(
    batch: i32,
    heads: i32,
    q_len: i32,
    k_len: i32,
    d_k: i32,
    d_v: i32,
    scale: f32,
    is_causal: i32,
    q: *const std::ffi::c_void,
    k: *const std::ffi::c_void,
    v: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    lse: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// SDPA with arbitrary additive mask (Phase 51). Same online-softmax
/// algorithm as `flash_sdpa` with an additional `mask: f32[B, H, Q, K]`
/// applied to `S = Q·K^T·scale` before row max/softmax. Mask is **always
/// f32** regardless of operand dtype (`mask: -INFINITY` cells suppress).
type SdpaArbmaskRun = unsafe extern "C" fn(
    batch: i32,
    heads: i32,
    q_len: i32,
    k_len: i32,
    d_k: i32,
    d_v: i32,
    scale: f32,
    is_causal: i32,
    q: *const std::ffi::c_void,
    k: *const std::ffi::c_void,
    v: *const std::ffi::c_void,
    mask: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    lse: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// RoPE driver mapping Fuel's `(outer_count, seq, head_dim)` to
/// baracuda's `(batch, heads, seq, head_dim)`. `outer_count`
/// = `batch * heads`; we collapse all outer dims into `heads` and
/// pass `batch = 1` — the layout `[batch, heads, seq, head_dim]`
/// is byte-identical to `[1, outer_count, seq, head_dim]` so the
/// kernel does the same work either way.
///
/// `base` defaults to 10000.0 (matches Llama / Mistral / Gemma).
/// `positions = null` selects baracuda's default `[0..seq)`
/// sequence.
fn rope_run(
    src: &CudaStorageBytes,
    src_layout: Option<&Layout>,
    outer_count: usize,
    seq: usize,
    head_dim: usize,
    contig: RopeRun,
    strided: RopeStridedRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    if head_dim % 2 != 0 {
        return Err(Error::Msg(format!(
            "{op_label}: head_dim must be even (got {head_dim})",
        )).bt());
    }
    let device = src.device().clone();
    let numel = outer_count * seq * head_dim;
    let out_bytes = numel * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let out = CudaStorageBytes::from_parts(Arc::new(out_buf), device, out_bytes);
    rope_run_into(
        src,
        src_layout,
        outer_count,
        seq,
        head_dim,
        &out,
        contig,
        strided,
        op_label,
        dtype_size_bytes,
    )?;
    Ok(out)
}

/// Write-into-output RoPE driver (CapturedRun executor build-out).
///
/// Identical rotary-embedding math (contig + strided dispatch) to
/// [`rope_run`], but writes into the caller-provided `out` buffer instead
/// of allocating one — the enabler for the pipelined executor's
/// persistent-output (capture) mode where a fixed-address output is written
/// in place so **no device allocation happens** (mandatory inside a
/// CUDA-graph capture scope). Byte-identical result to the alloc-and-return
/// path for a same-sized `out`.
///
/// `out` must already hold at least
/// `outer_count * seq * head_dim * dtype_size_bytes` bytes; a smaller
/// buffer is a surfaced error, never an out-of-bounds device write.
#[allow(clippy::too_many_arguments)]
fn rope_run_into(
    src: &CudaStorageBytes,
    src_layout: Option<&Layout>,
    outer_count: usize,
    seq: usize,
    head_dim: usize,
    out: &CudaStorageBytes,
    contig: RopeRun,
    strided: RopeStridedRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<()> {
    if head_dim % 2 != 0 {
        return Err(Error::Msg(format!(
            "{op_label}: head_dim must be even (got {head_dim})",
        )).bt());
    }
    let device = src.device().clone();
    let numel = outer_count * seq * head_dim;
    let out_bytes = numel * dtype_size_bytes;
    if out_bytes == 0 {
        return Ok(());
    }
    if out.len_bytes() < out_bytes {
        return Err(Error::Msg(format!(
            "{op_label}: write-into output buffer too small ({} < {} bytes)",
            out.len_bytes(),
            out_bytes,
        )).bt());
    }
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out.buffer().as_raw().0 as *mut std::ffi::c_void;

    let heads_i32 = i32::try_from(outer_count).map_err(|_| {
        Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label, dim_index: 0, dim_value: outer_count,
        })
    })?;
    let seq_i32 = i32::try_from(seq).map_err(|_| {
        Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label, dim_index: 1, dim_value: seq,
        })
    })?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(|_| {
        Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label, dim_index: 2, dim_value: head_dim,
        })
    })?;

    let take_strided = src_layout
        .map(|l| !is_contiguous_zero_offset(l))
        .unwrap_or(false);

    let status = if take_strided {
        // Strided path. Enforce head_dim (innermost) stride == 1, per
        // baracuda's RoPE pair-dim constraint. Then extract per-dim
        // strides for (batch=1, heads=outer_count, seq).
        let layout = src_layout.expect("guarded by take_strided");
        let strides = layout.stride();
        let last_stride = *strides.last().ok_or_else(|| {
            Error::Msg(format!("{op_label}: rank-0 input not supported")).bt()
        })?;
        if last_stride != 1 {
            return Err(Error::Msg(format!(
                "{op_label}: RoPE requires head_dim stride == 1 (got {last_stride}); \
                 Contiguize the input before dispatching"
            )).bt());
        }
        // Derive (stride_b, stride_h, stride_s) from the input's
        // rank-N layout. For rank-3 [outer, seq, head_dim] we treat
        // batch=1, stride_b=0. For rank-4 [batch, heads, seq, head_dim]
        // we collapse batch+heads into heads=outer_count with the
        // stride pattern of contig over batch*heads (matches what the
        // contig path produces for `heads=outer_count`).
        let rank = strides.len();
        let (stride_b, stride_h, stride_s) = match rank {
            3 => (0_i64, strides[0] as i64, strides[1] as i64),
            4 => {
                // batch * heads collapsed into heads. The collapsed
                // stride is the smaller of strides[0] / strides[1] —
                // for a row-major-ish layout, strides[1] (heads) is
                // contig within batch, so it's the right per-head
                // stride after collapsing.
                (0_i64, strides[1] as i64, strides[2] as i64)
            }
            other => {
                return Err(Error::Msg(format!(
                    "{op_label}: RoPE expects rank 3 or 4 input (got {other})",
                )).bt());
            }
        };
        let stride_y_h = (seq * head_dim) as i64;
        let stride_y_s = head_dim as i64;
        // SAFETY: pointers + dims validated.
        unsafe {
            strided(
                1, heads_i32, seq_i32, head_dim_i32,
                stride_b, stride_h, stride_s,
                0, stride_y_h, stride_y_s,
                10000.0, 1,
                x_ptr, std::ptr::null(), y_ptr,
                scratch.as_raw(), scratch.bytes(), stream,
            )
        }
    } else {
        // SAFETY: pointers + dims validated; default positions selected
        // via pos_default_flag=1 + null positions pointer.
        unsafe {
            contig(
                1, heads_i32, seq_i32, head_dim_i32,
                10000.0, 1,
                x_ptr, std::ptr::null(), y_ptr,
                scratch.as_raw(), scratch.bytes(), stream,
            )
        }
    };
    check(status, op_label)?;
    Ok(())
}

/// FlashSDPA driver (utility — no Fuel OpKind dispatch yet because
/// of the GQA / window / softcap shape gap). Returns the attention
/// output `[B, H, q_len, d_v]`. LSE is allocated as scratch and
/// dropped — when Fuel grows a backward op that needs it, it
/// becomes an output.
fn flash_sdpa_run(
    q: &CudaStorageBytes,
    k: &CudaStorageBytes,
    v: &CudaStorageBytes,
    batch: usize,
    heads: usize,
    q_len: usize,
    k_len: usize,
    d_k: usize,
    d_v: usize,
    scale: f32,
    is_causal: bool,
    kernel: FlashSdpaRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let device = q.device().clone();
    let numel = batch * heads * q_len * d_v;
    let out_bytes = numel * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    // LSE buffer: [B, H, q_len] f32 — scratch for the kernel's
    // online softmax reduction tracking; not consumed by callers.
    let lse_bytes = batch * heads * q_len * std::mem::size_of::<f32>();
    let lse_buf = device.alloc_zeros::<u8>(lse_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let q_ptr = q.buffer().as_raw().0 as *const std::ffi::c_void;
    let k_ptr = k.buffer().as_raw().0 as *const std::ffi::c_void;
    let v_ptr = v.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;
    let lse_ptr = lse_buf.as_raw().0 as *mut std::ffi::c_void;

    let i32_or = |dim_index: usize, dim_value: usize| -> Result<i32> {
        i32::try_from(dim_value).map_err(|_| {
            fuel_ir::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                op: op_label,
                dim_index,
                dim_value,
            })
        })
    };

    let status = unsafe {
        kernel(
            i32_or(0, batch)?,
            i32_or(1, heads)?,
            i32_or(2, q_len)?,
            i32_or(3, k_len)?,
            i32_or(4, d_k)?,
            i32_or(5, d_v)?,
            scale,
            if is_causal { 1 } else { 0 },
            q_ptr,
            k_ptr,
            v_ptr,
            y_ptr,
            lse_ptr,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    drop(lse_buf);
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out_buf),
        device,
        out_bytes,
    ))
}

macro_rules! rope_kernel {
    ($name:ident, $dtype_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `rope_", stringify!($dtype_stem), "` kernel (contig + strided dispatch).")]
            pub fn $name(
                src: &CudaStorageBytes,
                src_layout: Option<&Layout>,
                outer_count: usize,
                seq: usize,
                head_dim: usize,
            ) -> Result<CudaStorageBytes> {
                rope_run(
                    src,
                    src_layout,
                    outer_count,
                    seq,
                    head_dim,
                    sys::[<baracuda_kernels_rope_ $dtype_stem _run>],
                    sys::[<baracuda_kernels_rope_ $dtype_stem _strided_run>],
                    $op_label,
                    $dtype_size,
                )
            }

            #[doc = concat!(
                "Write-into-output variant of baracuda `rope_", stringify!($dtype_stem),
                "` — writes into `out` (no alloc; CapturedRun capture mode)."
            )]
            #[allow(clippy::too_many_arguments)]
            pub fn [<$name _into>](
                src: &CudaStorageBytes,
                src_layout: Option<&Layout>,
                outer_count: usize,
                seq: usize,
                head_dim: usize,
                out: &CudaStorageBytes,
            ) -> Result<()> {
                rope_run_into(
                    src,
                    src_layout,
                    outer_count,
                    seq,
                    head_dim,
                    out,
                    sys::[<baracuda_kernels_rope_ $dtype_stem _run>],
                    sys::[<baracuda_kernels_rope_ $dtype_stem _strided_run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

macro_rules! flash_sdpa_kernel {
    ($name:ident, $dtype_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `flash_sdpa_", stringify!($dtype_stem), "` kernel.")]
            #[allow(clippy::too_many_arguments)]
            pub fn $name(
                q: &CudaStorageBytes,
                k: &CudaStorageBytes,
                v: &CudaStorageBytes,
                batch: usize,
                heads: usize,
                q_len: usize,
                k_len: usize,
                d_k: usize,
                d_v: usize,
                scale: f32,
                is_causal: bool,
            ) -> Result<CudaStorageBytes> {
                flash_sdpa_run(
                    q, k, v, batch, heads, q_len, k_len, d_k, d_v, scale, is_causal,
                    sys::[<baracuda_kernels_flash_sdpa_ $dtype_stem _run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

rope_kernel!(rope_f32, f32, 4, "rope_f32");
rope_kernel!(rope_f16, f16, 2, "rope_f16");
rope_kernel!(rope_bf16, bf16, 2, "rope_bf16");
rope_kernel!(rope_f64, f64, 8, "rope_f64");

// ---------------------------------------------------------------------------
// Task 4.6 (FKC gap-closure): `rope_apply_<dt>_run` — baracuda's STANDALONE
// caller-supplied-cos/sin RoPE kernel. A separate FFI family from
// `rope_<dt>_run` above (which computes trig device-side from `base` +
// `positions`); NOT the same op-wrapper as `rope_f32`/etc. This FFI family
// had exactly ONE prior caller anywhere in the repo:
// `storage::CudaStorage::rope` (`fuel-cuda-backend/src/storage.rs:3883`), a
// method on the LEGACY dtype-tagged `CudaStorage`/`CudaStorageSlice`
// storage representation that predates `CudaStorageBytes` and has NO call
// sites anywhere — "shipped, never wired in" (the designated acceptance
// kernel for the FKC verification harness, see
// `docs/session-prompts/capturedrun-4b-paused-pending-fkc-verification.md`).
// This driver is the `CudaStorageBytes`-based, write-into-output wiring the
// CURRENT dispatch layer actually needs; it does not reuse the legacy
// method or its storage type.
//
// baracuda ABI (verified against `baracuda-kernels-sys` 0.0.1-alpha.77,
// `kernels/include/baracuda_attention.cuh:1703`'s `_INSTANTIATE` macro; FFI
// decl `src/lib.rs:49243`): `cos`/`sin` are ALWAYS F32 regardless of the
// operand dtype, shaped `[seq, head_dim/2]` — HALF the width of Fuel's
// `OpParams::Rope`-wide `[seq, head_dim]` convention that the CPU
// `rope_<dt>` family and the `rope_<dt>` driver above both use (those
// re-index a FULL-width table per pair; this kernel wants only the
// `head_dim/2` distinct trig values per position — see
// `docs/kernel-contracts/cuda/rope-apply.fkc.md` for the full note).
// `stride_b` is always 0 here (a single table shared across every one of
// the `outer_count` = batch*heads rows) — Fuel's cos/sin are always
// model-wide, never per-batch, so that is the only value ever needed.
// Contiguous input ONLY — no strided path (unlike `rope_run_into` above);
// a non-contiguous `x` must be Contiguized first (mirrors the CPU
// `rope_<dt>` contract's posture).
type RopeApplyRun = unsafe extern "C" fn(
    bh: i32,
    td: i32,
    d: i32,
    stride_b: i32,
    x: *const std::ffi::c_void,
    cos_tab: *const std::ffi::c_void,
    sin_tab: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Write-into-output driver for baracuda's `rope_apply_<dt>_run` (Task 4.6
/// FKC gap-closure acceptance kernel). See the module note above for the
/// ABI (`cos`/`sin` always F32, half-width `[seq, head_dim/2]`, `stride_b
/// == 0`). `out` must already hold at least `outer_count * seq * head_dim *
/// dtype_size_bytes` bytes (same never-alloc contract as `rope_run_into`).
#[allow(clippy::too_many_arguments)]
fn rope_apply_run_into(
    x: &CudaStorageBytes,
    cos: &CudaStorageBytes,
    sin: &CudaStorageBytes,
    outer_count: usize,
    seq: usize,
    head_dim: usize,
    out: &CudaStorageBytes,
    run: RopeApplyRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<()> {
    if head_dim % 2 != 0 {
        return Err(Error::Msg(format!(
            "{op_label}: head_dim must be even (got {head_dim})",
        )).bt());
    }
    let device = x.device().clone();
    let numel = outer_count * seq * head_dim;
    let out_bytes = numel * dtype_size_bytes;
    if out_bytes == 0 {
        return Ok(());
    }
    if out.len_bytes() < out_bytes {
        return Err(Error::Msg(format!(
            "{op_label}: write-into output buffer too small ({} < {} bytes)",
            out.len_bytes(), out_bytes,
        )).bt());
    }
    // cos/sin are ALWAYS F32, half-width [seq, head_dim/2] — validate the
    // byte length up front so a shape mismatch is a typed error, never an
    // out-of-bounds device read.
    let expected_trig_elems = seq * (head_dim / 2);
    let expected_trig_bytes = expected_trig_elems * 4;
    if cos.len_bytes() < expected_trig_bytes || sin.len_bytes() < expected_trig_bytes {
        return Err(Error::Msg(format!(
            "{op_label}: cos/sin table too small for [seq={seq}, head_dim/2={}] F32 \
             (need >= {expected_trig_bytes} bytes each, got cos={} sin={})",
            head_dim / 2, cos.len_bytes(), sin.len_bytes(),
        )).bt());
    }

    let bh = i32::try_from(outer_count).map_err(|_| {
        Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label, dim_index: 0, dim_value: outer_count,
        })
    })?;
    let td = i32::try_from(seq * head_dim).map_err(|_| {
        Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label, dim_index: 1, dim_value: seq * head_dim,
        })
    })?;
    let d = i32::try_from(head_dim).map_err(|_| {
        Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label, dim_index: 2, dim_value: head_dim,
        })
    })?;
    let stride_b: i32 = 0; // single shared cos/sin table across every outer row

    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = x.buffer().as_raw().0 as *const std::ffi::c_void;
    let cos_ptr = cos.buffer().as_raw().0 as *const std::ffi::c_void;
    let sin_ptr = sin.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out.buffer().as_raw().0 as *mut std::ffi::c_void;

    let status = unsafe {
        run(
            bh, td, d, stride_b,
            x_ptr, cos_ptr, sin_ptr, y_ptr,
            scratch.as_raw(), scratch.bytes(), stream,
        )
    };
    check(status, op_label)
}

macro_rules! rope_apply_kernel {
    ($name:ident, $dtype_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!(
                "Write-into-output driver for baracuda `rope_apply_",
                stringify!($dtype_stem), "` (Task 4.6 FKC acceptance kernel)."
            )]
            #[allow(clippy::too_many_arguments)]
            pub fn [<$name _into>](
                x: &CudaStorageBytes,
                cos: &CudaStorageBytes,
                sin: &CudaStorageBytes,
                outer_count: usize,
                seq: usize,
                head_dim: usize,
                out: &CudaStorageBytes,
            ) -> Result<()> {
                rope_apply_run_into(
                    x, cos, sin, outer_count, seq, head_dim, out,
                    sys::[<baracuda_kernels_rope_apply_ $dtype_stem _run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

rope_apply_kernel!(rope_apply_f32, f32, 4, "rope_apply_f32");
rope_apply_kernel!(rope_apply_f16, f16, 2, "rope_apply_f16");
rope_apply_kernel!(rope_apply_bf16, bf16, 2, "rope_apply_bf16");
rope_apply_kernel!(rope_apply_f64, f64, 8, "rope_apply_f64");

// ---------------------------------------------------------------------------
// STAGED, NOT WIRED — FUSED `FusedOps::ROPE` CUDA driver (CapturedRun 4b-resume).
//
// ⚠ These `rope_apply_fused_<dt>_into` drivers are NOT registered as the CUDA
// impl of `FusedOps::ROPE`. baracuda's `rope_apply` uses the INTERLEAVED pairing
// `(2k, 2k+1)` (GPT-J; confirmed by baracuda's `rope_fp.cu` "rotates pairs
// (2i, 2i+1)" AND a GPU numerical test — see
// `fused_rope_tests::fused_rope_driver_is_interleaved_not_fuel_rotate_half`
// below). Fuel's `FusedOps::ROPE` (`Tensor::rope_with_tables`) is ROTATE-HALF
// `(j, j+head_dim/2)` (Llama/HF; `rope_with_tables_decomposed`,
// `fuel-graph/src/lib.rs:6502-6509`). They compute DIFFERENT functions, so
// binding this kernel to the rotate-half op is a correctness bug — the Step-2
// registration (commit 3be28ab1) was reverted on 2026-07-13. CUDA rope runs via
// the correct rotate-half primitive decompose until baracuda ships a rotate-half
// table-driven `rope_apply` variant.
//
// PRIOR (flawed) rationale, corrected: the Step-2 note claimed the narrow's
// correctness was "DERIVED not assumed" from `rope_with_tables_decomposed`. That
// derivation covered only the full-width→half-width narrowing and silently
// ASSUMED baracuda's pairing matched Fuel's ("the standard shared-angle
// rotation"). It does not. The narrow itself is still correct (see below); the
// hidden assumption was the pairing convention.
//
// WHY THE MACHINERY IS KEPT (ready to wire for a rotate-half baracuda kernel):
// baracuda wants HALF-WIDTH cos/sin `[seq, head_dim/2]` while
// `Tensor::rope_with_tables` hard-asserts FULL-WIDTH `[seq, head_dim]`. The
// narrow extracts the first `head_dim/2` columns of each row; because Fuel's
// full-width table is BY CONSTRUCTION the half-width table duplicated across
// both halves (`cos[j] == cos[j+half]`), that extraction is byte-exact, not an
// approximation — convention-INDEPENDENT, so it is reusable as-is once a
// rotate-half `rope_apply` exists. The narrow-copy is one `cuMemcpy2DAsync` D2D
// copy (same pattern as `super::mamba::strip_prepad_d2d`).
//
// CAPTURE-SAFETY: `narrow_rope_table_f32` narrows into the device's grow-only
// two-slot `RopeTableCache` (`device.rope_tables()`, one slot cos, one sin), so
// a stable-capacity decode loop reuses a fixed device address after step 1 —
// zero `cuMemAlloc` during a capture scope. The retained cache `Arc` keeps the
// buffer live across the async copy + kernel. See `super::scratch::RopeTableCache`.

/// Narrow a FULL-WIDTH `[seq, head_dim]` F32 table (Fuel's `rope_with_tables`
/// convention) to a HALF-WIDTH `[seq, head_dim/2]` table via one
/// `cuMemcpy2DAsync` D2D copy of each row's first `head_dim/2` F32 elements.
/// See the module note above for why this is exact, not approximate (and why
/// it is convention-independent). Narrows into the device's grow-only
/// [`RopeTableCache`](super::scratch::RopeTableCache) `slot` (capture-safe:
/// reuses a fixed device buffer across a stable-capacity decode loop).
fn narrow_rope_table_f32(
    full: &CudaStorageBytes,
    seq: usize,
    head_dim: usize,
    slot: RopeTableSlot,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    use baracuda_cuda_sys::driver;
    use baracuda_cuda_sys::types::{CUmemorytype, CUDA_MEMCPY2D};

    let half = head_dim / 2;
    let device = full.device().clone();
    let elem_bytes = 4usize; // cos/sin are always F32 (baracuda ABI)
    let dst_bytes = seq.saturating_mul(half).saturating_mul(elem_bytes);
    if dst_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let expected_full_bytes = seq * head_dim * elem_bytes;
    if full.len_bytes() < expected_full_bytes {
        return Err(Error::Msg(format!(
            "{op_label}: full-width cos/sin table too small for [seq={seq}, head_dim={head_dim}] \
             F32 (need >= {expected_full_bytes} bytes, got {})",
            full.len_bytes(),
        )).bt());
    }
    // Grow-only capture-safe scratch: reuses a fixed device buffer per slot
    // across a stable-capacity decode loop (no per-call `cuMemAlloc`). The
    // returned `Arc` + the cache's retained `Arc` keep it live across the
    // async copy and the downstream async `rope_apply` kernel.
    let dst_arc = device.rope_tables().ensure(&device, slot, dst_bytes)?;
    let src_pitch = head_dim * elem_bytes;
    let dst_pitch = half * elem_bytes;
    let p = CUDA_MEMCPY2D {
        src_memory_type: CUmemorytype::DEVICE,
        src_device: full.buffer().as_raw(),
        src_pitch,
        dst_memory_type: CUmemorytype::DEVICE,
        dst_device: dst_arc.as_raw(),
        dst_pitch,
        width_in_bytes: dst_pitch,
        height: seq,
        ..Default::default()
    };
    let d = driver().map_err(|e| Error::Msg(format!("{op_label}: driver(): {e:?}")).bt())?;
    let cu = d.cu_memcpy_2d_async().map_err(|e| {
        Error::Msg(format!("{op_label}: cu_memcpy_2d_async: {e:?}")).bt()
    })?;
    let stream = device.stream().as_raw();
    // SAFETY: src/dst are live device buffers of the checked byte sizes (the
    // cached slot buffer is >= dst_bytes, grow-only); pitches + width/height
    // describe an in-bounds 2D rectangle for both (width_in_bytes == dst_pitch
    // <= src_pitch, height == seq rows); stream is this device's stream. The
    // narrowed region [0, dst_bytes) is fully overwritten here, so a reused
    // slot's stale tail is never read downstream.
    let status = unsafe { cu(&p, stream) };
    if status.0 != 0 {
        return Err(Error::Msg(format!(
            "{op_label}: cuMemcpy2DAsync failed: status={status:?}",
        )).bt());
    }
    Ok(CudaStorageBytes::from_parts(dst_arc, device, dst_bytes))
}

/// Write-into-output driver for the STAGED (unwired, interleaved) rope_apply,
/// F32 operand dtype. Narrows Fuel's full-width cos/sin (see the module
/// note above) then forwards to [`rope_apply_f32_into`].
pub fn rope_apply_fused_f32_into(
    x: &CudaStorageBytes,
    cos_full: &CudaStorageBytes,
    sin_full: &CudaStorageBytes,
    outer_count: usize,
    seq: usize,
    head_dim: usize,
    out: &CudaStorageBytes,
) -> Result<()> {
    let cos_half = narrow_rope_table_f32(cos_full, seq, head_dim, RopeTableSlot::Cos, "rope_apply_fused_f32:cos")?;
    let sin_half = narrow_rope_table_f32(sin_full, seq, head_dim, RopeTableSlot::Sin, "rope_apply_fused_f32:sin")?;
    rope_apply_f32_into(x, &cos_half, &sin_half, outer_count, seq, head_dim, out)
}

/// Write-into-output driver for the STAGED (unwired, interleaved) rope_apply,
/// F16 operand dtype (cos/sin are always F32 regardless — see the module
/// note above). Narrows Fuel's full-width cos/sin then forwards to
/// [`rope_apply_f16_into`].
pub fn rope_apply_fused_f16_into(
    x: &CudaStorageBytes,
    cos_full: &CudaStorageBytes,
    sin_full: &CudaStorageBytes,
    outer_count: usize,
    seq: usize,
    head_dim: usize,
    out: &CudaStorageBytes,
) -> Result<()> {
    let cos_half = narrow_rope_table_f32(cos_full, seq, head_dim, RopeTableSlot::Cos, "rope_apply_fused_f16:cos")?;
    let sin_half = narrow_rope_table_f32(sin_full, seq, head_dim, RopeTableSlot::Sin, "rope_apply_fused_f16:sin")?;
    rope_apply_f16_into(x, &cos_half, &sin_half, outer_count, seq, head_dim, out)
}

/// Write-into-output driver for the STAGED (unwired, interleaved) rope_apply,
/// BF16 operand dtype (cos/sin are always F32 regardless — see the module
/// note above). Narrows Fuel's full-width cos/sin then forwards to
/// [`rope_apply_bf16_into`].
pub fn rope_apply_fused_bf16_into(
    x: &CudaStorageBytes,
    cos_full: &CudaStorageBytes,
    sin_full: &CudaStorageBytes,
    outer_count: usize,
    seq: usize,
    head_dim: usize,
    out: &CudaStorageBytes,
) -> Result<()> {
    let cos_half = narrow_rope_table_f32(cos_full, seq, head_dim, RopeTableSlot::Cos, "rope_apply_fused_bf16:cos")?;
    let sin_half = narrow_rope_table_f32(sin_full, seq, head_dim, RopeTableSlot::Sin, "rope_apply_fused_bf16:sin")?;
    rope_apply_bf16_into(x, &cos_half, &sin_half, outer_count, seq, head_dim, out)
}

/// Write-into-output driver for the STAGED (unwired, interleaved) rope_apply,
/// F64 operand dtype (cos/sin are always F32 regardless — see the module
/// note above). Narrows Fuel's full-width cos/sin then forwards to
/// [`rope_apply_f64_into`].
pub fn rope_apply_fused_f64_into(
    x: &CudaStorageBytes,
    cos_full: &CudaStorageBytes,
    sin_full: &CudaStorageBytes,
    outer_count: usize,
    seq: usize,
    head_dim: usize,
    out: &CudaStorageBytes,
) -> Result<()> {
    let cos_half = narrow_rope_table_f32(cos_full, seq, head_dim, RopeTableSlot::Cos, "rope_apply_fused_f64:cos")?;
    let sin_half = narrow_rope_table_f32(sin_full, seq, head_dim, RopeTableSlot::Sin, "rope_apply_fused_f64:sin")?;
    rope_apply_f64_into(x, &cos_half, &sin_half, outer_count, seq, head_dim, out)
}

flash_sdpa_kernel!(flash_sdpa_f32, f32, 4, "flash_sdpa_f32");
flash_sdpa_kernel!(flash_sdpa_f16, f16, 2, "flash_sdpa_f16");
flash_sdpa_kernel!(flash_sdpa_bf16, bf16, 2, "flash_sdpa_bf16");
flash_sdpa_kernel!(flash_sdpa_f64, f64, 8, "flash_sdpa_f64");

/// Driver for `sdpa_<dt>_arbmask_run` (Phase 51 additive-mask SDPA).
/// Same shape as `flash_sdpa_run` plus a `mask` byte view shaped
/// `[B, H, q_len, k_len]` in F32 regardless of operand dtype.
#[allow(clippy::too_many_arguments)]
fn sdpa_arbmask_run(
    q: &CudaStorageBytes,
    k: &CudaStorageBytes,
    v: &CudaStorageBytes,
    mask: &CudaStorageBytes,
    batch: usize,
    heads: usize,
    q_len: usize,
    k_len: usize,
    d_k: usize,
    d_v: usize,
    scale: f32,
    is_causal: bool,
    kernel: SdpaArbmaskRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let device = q.device().clone();
    let numel = batch * heads * q_len * d_v;
    let out_bytes = numel * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let lse_bytes = batch * heads * q_len * std::mem::size_of::<f32>();
    let lse_buf = device.alloc_zeros::<u8>(lse_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let q_ptr = q.buffer().as_raw().0 as *const std::ffi::c_void;
    let k_ptr = k.buffer().as_raw().0 as *const std::ffi::c_void;
    let v_ptr = v.buffer().as_raw().0 as *const std::ffi::c_void;
    let mask_ptr = mask.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;
    let lse_ptr = lse_buf.as_raw().0 as *mut std::ffi::c_void;

    let i32_or = |dim_index: usize, dim_value: usize| -> Result<i32> {
        i32::try_from(dim_value).map_err(|_| {
            fuel_ir::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                op: op_label,
                dim_index,
                dim_value,
            })
        })
    };

    let status = unsafe {
        kernel(
            i32_or(0, batch)?,
            i32_or(1, heads)?,
            i32_or(2, q_len)?,
            i32_or(3, k_len)?,
            i32_or(4, d_k)?,
            i32_or(5, d_v)?,
            scale,
            if is_causal { 1 } else { 0 },
            q_ptr, k_ptr, v_ptr, mask_ptr, y_ptr, lse_ptr,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    drop(lse_buf);
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out_buf),
        device,
        out_bytes,
    ))
}

macro_rules! sdpa_arbmask_kernel {
    ($name:ident, $dtype_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `sdpa_", stringify!($dtype_stem), "_arbmask` kernel.")]
            #[allow(clippy::too_many_arguments)]
            pub fn $name(
                q: &CudaStorageBytes,
                k: &CudaStorageBytes,
                v: &CudaStorageBytes,
                mask: &CudaStorageBytes,
                batch: usize,
                heads: usize,
                q_len: usize,
                k_len: usize,
                d_k: usize,
                d_v: usize,
                scale: f32,
                is_causal: bool,
            ) -> Result<CudaStorageBytes> {
                sdpa_arbmask_run(
                    q, k, v, mask, batch, heads, q_len, k_len, d_k, d_v, scale, is_causal,
                    sys::[<baracuda_kernels_sdpa_ $dtype_stem _arbmask_run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

sdpa_arbmask_kernel!(sdpa_arbmask_f32, f32, 4, "sdpa_arbmask_f32");
sdpa_arbmask_kernel!(sdpa_arbmask_f16, f16, 2, "sdpa_arbmask_f16");
sdpa_arbmask_kernel!(sdpa_arbmask_bf16, bf16, 2, "sdpa_arbmask_bf16");
sdpa_arbmask_kernel!(sdpa_arbmask_f64, f64, 8, "sdpa_arbmask_f64");

// ─────────────────────── can_implement ───────────────────────
//
// Host-side validators for the sdpa_arbmask kernel set. Mirror the
// Phase 51 contract: 0 = ok, non-zero = caller's shape/causal/dtype
// combo is rejected. Fuel dispatch code should call these before
// allocating output buffers and launching.

macro_rules! sdpa_arbmask_can_impl {
    ($name:ident, $sys:ident, $label:expr $(,)?) => {
        #[doc = concat!("Pre-launch validation for `", $label, "_run`.")]
        pub fn $name(
            batch: i32, heads: i32,
            q_len: i32, k_len: i32,
            d_k: i32, d_v: i32,
            is_causal: bool,
        ) -> Result<()> {
            let status = unsafe {
                sys::$sys(
                    batch, heads, q_len, k_len, d_k, d_v,
                    if is_causal { 1 } else { 0 },
                )
            };
            check(status, concat!($label, "_can_implement"))
        }
    };
}

sdpa_arbmask_can_impl!(sdpa_arbmask_f32_can_implement,  baracuda_kernels_sdpa_f32_arbmask_can_implement,  "sdpa_arbmask_f32");
sdpa_arbmask_can_impl!(sdpa_arbmask_f64_can_implement,  baracuda_kernels_sdpa_f64_arbmask_can_implement,  "sdpa_arbmask_f64");
sdpa_arbmask_can_impl!(sdpa_arbmask_f16_can_implement,  baracuda_kernels_sdpa_f16_arbmask_can_implement,  "sdpa_arbmask_f16");
sdpa_arbmask_can_impl!(sdpa_arbmask_bf16_can_implement, baracuda_kernels_sdpa_bf16_arbmask_can_implement, "sdpa_arbmask_bf16");

// ─────────────────────── FlashDecoding (decode-flash) ───────────────────────
//
// Baracuda `flash_decoding_{f16,bf16}` (alpha.72) — split-K parallel
// attention for autoregressive **decode** (`seq_q == 1`) over a
// fixed-capacity KV cache with a runtime live prefix `k_len <= max_seq`.
// This is the FIRST capacity-K flash binding in Fuel; it maps
// `OpKind::FlashAttn`'s Phase-D decode shape (`OpParams::FlashAttn`'s
// `sk` = physical capacity, `k_len` = logical attended length) onto the
// kernel's per-tensor strides + runtime iteration bound.
//
// Load-bearing ABI facts (see docs/outreach/baracuda-flashdecoding-decode-
// interface-reply.md — a PINNED standing contract):
//   * Explicit per-tensor strides (element units) DECOUPLED from `k_len`;
//     `k_len` is only the iteration bound + `num_splits = ceil(k_len/256)`.
//     A capacity buffer (`k_seq_stride = D`, `k_h_stride = max_seq*D`,
//     `k_b_stride = Hkv*max_seq*D`, live prefix `k_len < max_seq`) reads
//     correctly for any `B*Hkv` — NO Contiguize copy. The innermost
//     (head_dim) axis is assumed contiguous (no head_dim stride arg).
//   * GQA-native: `num_kv_heads` is a separate parameter; the launcher
//     enforces `heads % num_kv_heads == 0`.
//   * `seq_q == 1`; `head_dim` in [1, 128] (`kMaxD`); f16/bf16 only.
//   * Caller provides `y` (allocated + zero-init here) AND workspace; the
//     kernel allocates nothing. Workspace bytes are MONOTONIC in `k_len`,
//     so we size ONCE at capacity (`sk`) — every decode step allocates the
//     same size, so a per-device grow-only cache is a drop-in follow-up
//     (the plan-once decode arc; see scratch.rs's deferred-pooling note).
//   * `k_len == 0` returns 0 (success) WITHOUT touching `y` → the
//     zero-init output stays zeros.
//   * Return codes 0 ok / 2 dims|GQA|k_len<0 / 3 head_dim>128 / 4
//     workspace / 1000+cudaError, mapped to a `fuel_ir::Result` via
//     `status::check` (never panics).
//
// ## Decline-to-decomposed is a STATIC (ranker) decision, not a runtime
// ## bail
//
// Fuel dispatch is FAIL-FAST: a registered kernel that returns `Err`
// fails `realize` outright — there is NO runtime fallback to the
// decomposed base map (pipelined.rs "Fail-fast dispatch"). Therefore the
// unsupported-shape gates (`seq_q != 1`, `head_dim > 128`, window /
// softcap) are enforced at REGISTRATION/RANKER level (the dtype key gates
// f16/bf16; `cost::cost_flash_decoding_cuda` returns an infeasible cost
// outside the supported set so the ranker prefers the decomposed arm).
// The hard errors below are DEFENSE-IN-DEPTH for states the ranker should
// have excluded — they signal a routing bug, never a shape the caller is
// entitled to fall back from.

type FlashDecodingRun = unsafe extern "C" fn(
    q: *const std::ffi::c_void,
    k: *const std::ffi::c_void,
    v: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    batch: i32,
    heads: i32,
    num_kv_heads: i32,
    k_len: i32,
    head_dim: i32,
    q_b_stride: i64,
    q_h_stride: i64,
    k_b_stride: i64,
    k_h_stride: i64,
    k_seq_stride: i64,
    v_b_stride: i64,
    v_h_stride: i64,
    v_seq_stride: i64,
    y_b_stride: i64,
    y_h_stride: i64,
    scale: f32,
    stream: *mut std::ffi::c_void,
) -> i32;

type FlashDecodingCanImpl = unsafe extern "C" fn(
    batch: i32,
    heads: i32,
    num_kv_heads: i32,
    k_len: i32,
    head_dim: i32,
) -> i32;

type FlashDecodingWorkspaceBytes = unsafe extern "C" fn(
    batch: i32,
    heads: i32,
    k_len: i32,
    head_dim: i32,
) -> usize;

/// Derive `(b_stride, h_stride, seq_stride)` in ELEMENT units for a rank-4
/// `[B, H, S, D]` tensor from its `Layout` (decoupled from `k_len` — the
/// capacity buffer's strides read a live prefix correctly). Requires the
/// innermost (head_dim) axis to be contiguous (`stride[3] == 1`), matching
/// the kernel's implicit unit head_dim stride. When no layout is supplied
/// (executor passed none), falls back to the contiguous strides of `dims`.
fn flash_decoding_rank4_strides(
    layout: Option<&Layout>,
    dims: [usize; 4],
    op_label: &'static str,
    tensor: &'static str,
) -> Result<(i64, i64, i64)> {
    match layout {
        Some(l) => {
            let s = l.stride();
            if s.len() != 4 {
                return Err(Error::Msg(format!(
                    "{op_label}: {tensor} must be rank 4 [B,H,S,D], got stride rank {}",
                    s.len(),
                ))
                .bt());
            }
            if s[3] != 1 {
                return Err(Error::Msg(format!(
                    "{op_label}: {tensor} head_dim (innermost) axis must be contiguous \
                     (stride[3] == 1), got {}",
                    s[3],
                ))
                .bt());
            }
            Ok((s[0] as i64, s[1] as i64, s[2] as i64))
        }
        None => {
            // Contiguous fallback for [d0, d1, d2, d3].
            let seq_stride = dims[3] as i64;
            let h_stride = (dims[2] * dims[3]) as i64;
            let b_stride = (dims[1] * dims[2] * dims[3]) as i64;
            Ok((b_stride, h_stride, seq_stride))
        }
    }
}

/// FlashDecoding driver. Allocates a zero-initialized output
/// `[B, Hq, 1, D]`, sizes workspace at capacity (`sk`), and launches the
/// baracuda decode kernel. Returns the attention output.
#[allow(clippy::too_many_arguments)]
fn flash_decoding_run(
    q: &CudaStorageBytes,
    k: &CudaStorageBytes,
    v: &CudaStorageBytes,
    q_layout: Option<&Layout>,
    k_layout: Option<&Layout>,
    v_layout: Option<&Layout>,
    b: usize,
    hq: usize,
    hkv: usize,
    sq: usize,
    sk: usize,
    d: usize,
    k_len: usize,
    scale: f32,
    run: FlashDecodingRun,
    can_impl: FlashDecodingCanImpl,
    ws_bytes: FlashDecodingWorkspaceBytes,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    // ---- Static-shape gates (DEFENSE-IN-DEPTH; the ranker excludes these) ----
    if sq != 1 {
        return Err(Error::Msg(format!(
            "{op_label}: flash_decoding is a decode kernel (seq_q must be 1, got {sq}); \
             the ranker must route seq_q>1 to the decomposed base map",
        ))
        .bt());
    }
    if hkv == 0 || hq % hkv != 0 {
        return Err(Error::Msg(format!(
            "{op_label}: Hq={hq} must be a positive multiple of Hkv={hkv} (GQA)",
        ))
        .bt());
    }
    if k_len > sk {
        return Err(Error::Msg(format!(
            "{op_label}: logical k_len ({k_len}) exceeds physical K capacity sk ({sk})",
        ))
        .bt());
    }

    let batch_i = i32::try_from(b).map_err(|_| shape_overflow(op_label, 0, b))?;
    let heads_i = i32::try_from(hq).map_err(|_| shape_overflow(op_label, 1, hq))?;
    let kv_heads_i = i32::try_from(hkv).map_err(|_| shape_overflow(op_label, 2, hkv))?;
    let d_i = i32::try_from(d).map_err(|_| shape_overflow(op_label, 3, d))?;
    let k_len_i = i32::try_from(k_len).map_err(|_| shape_overflow(op_label, 4, k_len))?;
    let sk_i = i32::try_from(sk).map_err(|_| shape_overflow(op_label, 5, sk))?;

    // No-launch admissibility gate (head_dim<=128, GQA divisibility, k_len>=0).
    // SAFETY: pure host-side integer check, no pointers.
    let can = unsafe { can_impl(batch_i, heads_i, kv_heads_i, k_len_i, d_i) };
    check(can, op_label)?;

    let device = q.device().clone();
    let out_bytes = b * hq * sq * d * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    // Zero-init: covers the `k_len == 0` edge (kernel writes nothing → the
    // output must already be zeros).
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;

    // Per-tensor strides (element units), decoupled from k_len.
    let (q_b, q_h, _q_s) =
        flash_decoding_rank4_strides(q_layout, [b, hq, sq, d], op_label, "q")?;
    let (k_b, k_h, k_s) =
        flash_decoding_rank4_strides(k_layout, [b, hkv, sk, d], op_label, "k")?;
    let (v_b, v_h, v_s) =
        flash_decoding_rank4_strides(v_layout, [b, hkv, sk, d], op_label, "v")?;
    // Output is freshly allocated contiguous [B, Hq, 1, D].
    let y_b = (hq * sq * d) as i64;
    let y_h = (sq * d) as i64;

    // Workspace sized at CAPACITY (monotonic in k_len ⇒ covers every step's
    // live prefix). Served from the device's grow-only per-device cache: a
    // decode session's capacity is fixed, so after the first step this reuses
    // one allocation instead of allocating every step (the plan-once decode
    // arc). The cache holds its lock across the launch (single-stream dispatch
    // ⇒ no contention), keeping the scratch live for the kernel.
    // SAFETY: pure host-side size query.
    let ws_need = unsafe { ws_bytes(batch_i, heads_i, sk_i, d_i) };

    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let q_ptr = q.buffer().as_raw().0 as *const std::ffi::c_void;
    let k_ptr = k.buffer().as_raw().0 as *const std::ffi::c_void;
    let v_ptr = v.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    // SAFETY: pointers are live device buffers of the checked byte sizes;
    // strides are element units matching the ABI; workspace >= the kernel's
    // capacity requirement; stream is this device's stream.
    let status = device.flash_workspace().with(&device, ws_need, |ws_ptr, ws_len| unsafe {
        run(
            q_ptr, k_ptr, v_ptr, y_ptr,
            ws_ptr, ws_len,
            batch_i, heads_i, kv_heads_i, k_len_i, d_i,
            q_b, q_h,
            k_b, k_h, k_s,
            v_b, v_h, v_s,
            y_b, y_h,
            scale,
            stream,
        )
    })?;
    check(status, op_label)?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out_buf),
        device,
        out_bytes,
    ))
}

fn shape_overflow(op: &'static str, dim_index: usize, dim_value: usize) -> Error {
    Error::cuda(crate::error::CudaError::BaracudaShapeOverflow { op, dim_index, dim_value })
}

macro_rules! flash_decoding_kernel {
    ($name:ident, $dtype_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `flash_decoding_", stringify!($dtype_stem), "` decode-flash kernel (seq_q==1, capacity-K).")]
            #[allow(clippy::too_many_arguments)]
            pub fn $name(
                q: &CudaStorageBytes,
                k: &CudaStorageBytes,
                v: &CudaStorageBytes,
                q_layout: Option<&Layout>,
                k_layout: Option<&Layout>,
                v_layout: Option<&Layout>,
                b: usize,
                hq: usize,
                hkv: usize,
                sq: usize,
                sk: usize,
                d: usize,
                k_len: usize,
                scale: f32,
            ) -> Result<CudaStorageBytes> {
                flash_decoding_run(
                    q, k, v, q_layout, k_layout, v_layout,
                    b, hq, hkv, sq, sk, d, k_len, scale,
                    sys::[<baracuda_kernels_flash_decoding_ $dtype_stem _run>],
                    sys::[<baracuda_kernels_flash_decoding_ $dtype_stem _can_implement>],
                    sys::[<baracuda_kernels_flash_decoding_ $dtype_stem _workspace_bytes>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

flash_decoding_kernel!(flash_decoding_f16,  f16,  2, "flash_decoding_f16");
flash_decoding_kernel!(flash_decoding_bf16, bf16, 2, "flash_decoding_bf16");

// ---------------------------------------------------------------------------
// CapturedRun 4b-resume Step 2b — GPU acceptance for the FUSED `FusedOps::ROPE`
// CUDA candidate: (b) numerical correctness of the full-width→half-width narrow
// + rope math against Fuel's AUTHORITATIVE `rope_with_tables_decomposed`
// semantics, and (a-verify) the `RopeTableCache` zero-alloc-during-capture
// property end-to-end through the fused driver. Both need a live device →
// `#[ignore]`'d (this crate only compiles under a CUDA toolchain anyway).
#[cfg(test)]
mod fused_rope_tests {
    use super::*;
    use crate::CudaDevice;

    fn dev_or_skip() -> Option<CudaDevice> {
        match CudaDevice::new(0) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("no CUDA device; skipping: {e:?}");
                None
            }
        }
    }

    fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// GUARD (CapturedRun 4b-resume, 2026-07-13): pins the discovered fact
    /// that baracuda's `rope_apply` is **INTERLEAVED** `(2k, 2k+1)` (GPT-J),
    /// which is a DIFFERENT function than Fuel's `FusedOps::ROPE`
    /// (`Tensor::rope_with_tables`), which is **ROTATE-HALF** `(j, j+half)`
    /// (Llama/HF, `rope_with_tables_decomposed` in `fuel-graph`). This is
    /// exactly why `rope_apply` is NOT wired as the CUDA impl of `FusedOps::ROPE`
    /// (the registration was reverted — see the module note on
    /// `rope_apply_fused_f32_into` and `dispatch.rs`).
    ///
    /// Computes BOTH references on the SAME full-width (shared-angle /
    /// duplicated-half) tables and asserts the GPU output equals the
    /// **interleaved** one and DIFFERS materially from the **rotate-half** one.
    /// Regression tripwire: if baracuda ever switches `rope_apply` to
    /// rotate-half, `got == interleaved` fails and `got != rotate_half` fails —
    /// signalling the kernel is now wireable as `FusedOps::ROPE`.
    #[test]
    #[ignore = "requires a live CUDA device"]
    fn fused_rope_driver_is_interleaved_not_fuel_rotate_half() {
        let Some(dev) = dev_or_skip() else { return };

        const OUTER: usize = 2;
        const SEQ: usize = 2;
        const HD: usize = 4; // head_dim (even)
        const HALF: usize = HD / 2;

        // Full-width cos/sin [SEQ, HD] with the shared-angle duplication
        // property Fuel's real `rope_with_tables` always produces:
        // cos[s, j] == cos[s, j+half], sin[s, j] == sin[s, j+half].
        let mut cos_full = vec![0f32; SEQ * HD];
        let mut sin_full = vec![0f32; SEQ * HD];
        for s in 0..SEQ {
            for j in 0..HALF {
                let theta = 0.3f64 * (s as f64) + 0.7f64 * (j as f64) + 0.1f64;
                let (c, sn) = (theta.cos() as f32, theta.sin() as f32);
                cos_full[s * HD + j] = c;
                cos_full[s * HD + j + HALF] = c; // duplicated half
                sin_full[s * HD + j] = sn;
                sin_full[s * HD + j + HALF] = sn;
            }
        }

        // Deterministic x [OUTER, SEQ, HD].
        let n = OUTER * SEQ * HD;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.13 - 0.9).collect();

        // Reference A — Fuel's ROTATE-HALF (`rope_with_tables_decomposed`):
        // pairs (j, j+half). This is what a correct `FusedOps::ROPE` impl must
        // produce; baracuda's kernel does NOT.
        let mut rotate_half = vec![0f32; n];
        for b in 0..OUTER {
            for s in 0..SEQ {
                for j in 0..HALF {
                    let e0 = b * SEQ * HD + s * HD + j;
                    let e1 = b * SEQ * HD + s * HD + j + HALF;
                    let c = cos_full[s * HD + j] as f64;
                    let sn = sin_full[s * HD + j] as f64;
                    let (x0, x1) = (x[e0] as f64, x[e1] as f64);
                    rotate_half[e0] = (x0 * c - x1 * sn) as f32;
                    rotate_half[e1] = (x1 * c + x0 * sn) as f32;
                }
            }
        }

        // Reference B — baracuda's INTERLEAVED (`rope_apply`): pairs (2k, 2k+1),
        // cos/sin indexed by pair k from the narrowed half-width table.
        let mut interleaved = vec![0f32; n];
        for b in 0..OUTER {
            for s in 0..SEQ {
                for k in 0..HALF {
                    let e0 = b * SEQ * HD + s * HD + 2 * k;
                    let e1 = e0 + 1;
                    let c = cos_full[s * HD + k] as f64; // narrowed: first HALF cols
                    let sn = sin_full[s * HD + k] as f64;
                    let (x0, x1) = (x[e0] as f64, x[e1] as f64);
                    interleaved[e0] = (x0 * c - x1 * sn) as f32;
                    interleaved[e1] = (x0 * sn + x1 * c) as f32;
                }
            }
        }

        // Device round-trip through the (staged, unwired) fused driver.
        let x_d = CudaStorageBytes::from_cpu_bytes(&dev, &f32s_to_bytes(&x)).expect("x upload");
        let cos_d = CudaStorageBytes::from_cpu_bytes(&dev, &f32s_to_bytes(&cos_full)).expect("cos upload");
        let sin_d = CudaStorageBytes::from_cpu_bytes(&dev, &f32s_to_bytes(&sin_full)).expect("sin upload");
        let out_d = CudaStorageBytes::alloc(&dev, n * 4).expect("out alloc");

        rope_apply_fused_f32_into(&x_d, &cos_d, &sin_d, OUTER, SEQ, HD, &out_d)
            .expect("fused rope launch");
        let got = bytes_to_f32s(&out_d.to_cpu_bytes().expect("D2H"));
        assert_eq!(got.len(), n);

        // GPU output MATCHES interleaved (proves narrow + cache + baracuda ABI).
        for (i, (g, r)) in got.iter().zip(interleaved.iter()).enumerate() {
            assert!(
                (g - r).abs() <= 1e-5,
                "baracuda rope_apply element {i}: got {g}, interleaved-ref {r} (|Δ|={})",
                (g - r).abs(),
            );
        }
        // ...and DIFFERS materially from Fuel's rotate-half — the incompatibility
        // that keeps this kernel unwired as `FusedOps::ROPE`.
        let max_delta_vs_rothalf = got
            .iter()
            .zip(rotate_half.iter())
            .map(|(g, r)| (g - r).abs())
            .fold(0f32, f32::max);
        assert!(
            max_delta_vs_rothalf > 1e-3,
            "baracuda rope_apply must DIFFER from Fuel rotate-half (max|Δ|={max_delta_vs_rothalf}); \
             if this fails, baracuda may have switched to rotate-half — re-evaluate wiring it as ROPE",
        );
        eprintln!(
            "[fused_rope] confirmed INTERLEAVED (== interleaved-ref, max|Δ| vs rotate-half = {max_delta_vs_rothalf:.4})",
        );
    }

    /// Step 2b(a) verify — end-to-end capture-safety: two same-shape fused
    /// launches must allocate the cos/sin scratch EXACTLY ONCE each (count 2
    /// after both calls), proving the second launch reused the cache (zero
    /// `cuMemAlloc` — the CapturedRun invariant). A fresh `CudaDevice::new(0)`
    /// starts with an empty per-device cache (allocation_count 0).
    #[test]
    #[ignore = "requires a live CUDA device"]
    fn fused_rope_is_capture_safe_zero_alloc_on_reuse() {
        let Some(dev) = dev_or_skip() else { return };

        const OUTER: usize = 2;
        const SEQ: usize = 2;
        const HD: usize = 4;
        let n = OUTER * SEQ * HD;

        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.07).collect();
        let trig: Vec<f32> = (0..SEQ * HD).map(|i| (i as f32) * 0.01).collect();
        let x_d = CudaStorageBytes::from_cpu_bytes(&dev, &f32s_to_bytes(&x)).expect("x upload");
        let cos_d = CudaStorageBytes::from_cpu_bytes(&dev, &f32s_to_bytes(&trig)).expect("cos upload");
        let sin_d = CudaStorageBytes::from_cpu_bytes(&dev, &f32s_to_bytes(&trig)).expect("sin upload");
        let out_d = CudaStorageBytes::alloc(&dev, n * 4).expect("out alloc");

        assert_eq!(dev.rope_tables().allocation_count(), 0, "fresh device cache starts empty");

        rope_apply_fused_f32_into(&x_d, &cos_d, &sin_d, OUTER, SEQ, HD, &out_d).expect("launch 1");
        assert_eq!(
            dev.rope_tables().allocation_count(), 2,
            "first fused launch allocates cos + sin scratch once each",
        );

        rope_apply_fused_f32_into(&x_d, &cos_d, &sin_d, OUTER, SEQ, HD, &out_d).expect("launch 2");
        assert_eq!(
            dev.rope_tables().allocation_count(), 2,
            "second same-shape fused launch must REUSE the scratch (zero alloc during capture)",
        );
        eprintln!("[fused_rope] capture-safe: 2 launches, 2 allocations total (reuse confirmed)");
    }
}

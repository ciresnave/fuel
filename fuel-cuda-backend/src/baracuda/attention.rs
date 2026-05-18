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
use fuel_core_types::Result;

use crate::byte_storage::CudaStorageBytes;

use super::scratch::Workspace;
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
    outer_count: usize,
    seq: usize,
    head_dim: usize,
    kernel: RopeRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    if head_dim % 2 != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "{op_label}: head_dim must be even (got {head_dim})",
        ))
        .bt());
    }
    let device = src.device().clone();
    let numel = outer_count * seq * head_dim;
    let out_bytes = numel * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    let heads_i32 = i32::try_from(outer_count).map_err(|_| {
        fuel_core_types::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label,
            dim_index: 0,
            dim_value: outer_count,
        })
    })?;
    let seq_i32 = i32::try_from(seq).map_err(|_| {
        fuel_core_types::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label,
            dim_index: 1,
            dim_value: seq,
        })
    })?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(|_| {
        fuel_core_types::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label,
            dim_index: 2,
            dim_value: head_dim,
        })
    })?;

    // SAFETY: pointers + dims validated; default positions selected
    // via pos_default_flag=1 + null positions pointer.
    let status = unsafe {
        kernel(
            1,
            heads_i32,
            seq_i32,
            head_dim_i32,
            10000.0,
            1,
            x_ptr,
            std::ptr::null(),
            y_ptr,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out_buf),
        device,
        out_bytes,
    ))
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
            fuel_core_types::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
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
    device.synchronize()?;
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
            #[doc = concat!("Baracuda `rope_", stringify!($dtype_stem), "` kernel.")]
            pub fn $name(
                src: &CudaStorageBytes,
                outer_count: usize,
                seq: usize,
                head_dim: usize,
            ) -> Result<CudaStorageBytes> {
                rope_run(
                    src,
                    outer_count,
                    seq,
                    head_dim,
                    sys::[<baracuda_kernels_rope_ $dtype_stem _run>],
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

flash_sdpa_kernel!(flash_sdpa_f32, f32, 4, "flash_sdpa_f32");
flash_sdpa_kernel!(flash_sdpa_f16, f16, 2, "flash_sdpa_f16");
flash_sdpa_kernel!(flash_sdpa_bf16, bf16, 2, "flash_sdpa_bf16");
flash_sdpa_kernel!(flash_sdpa_f64, f64, 8, "flash_sdpa_f64");

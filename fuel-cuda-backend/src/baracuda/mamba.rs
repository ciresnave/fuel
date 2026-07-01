//! Mamba / Mamba-2 state-space model (SSM) primitives from baracuda
//! alpha.58 (Phase 50 + Phase 50b):
//!
//! - **causal_conv1d** (Phase 50, BSD-3-Clause Dao-AILab port) —
//!   depthwise causal 1D convolution with optional fused SiLU. FW + BW
//!   across F32 / F16 / BF16 / F64. Used as Mamba's input-projection
//!   short conv.
//! - **ssd_chunk_scan** (Phase 50, Apache-2.0 state-spaces/mamba port)
//!   — Mamba-2's SSD (state space dual) chunk-scan block. Chunked
//!   matmul-form SSM with per-head state and per-chunk recurrence.
//!   FW + BW across F32 / F16 / BF16.
//! - **selective_scan** (Phase 50b) — Mamba-1's selective scan
//!   (per-element delta + A/B/C with optional D-skip and Z gate).
//!   FW + BW across F32 / F16 / BF16.
//!
//! All three primitives are gated behind baracuda's `mamba` cargo
//! feature (enabled at the workspace level). No Fuel `OpKind` dispatch
//! yet — adding Mamba model support requires Op kinds for these three
//! primitives plus autograd integration, which is a dedicated session.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_ir::Result;

use crate::byte_storage::CudaStorageBytes;
use crate::error::CudaError;

use super::status::check;

fn shape_i32(op: &'static str, dim_index: usize, dim_value: usize) -> Result<i32> {
    i32::try_from(dim_value).map_err(|_| {
        CudaError::BaracudaShapeOverflow { op, dim_index, dim_value }.into()
    })
}

// ───────────────────────── causal_conv1d ─────────────────────────

/// Causal 1-D depthwise convolution forward. Optionally fuses a SiLU
/// activation on the output (`use_silu`). `x` / `y` shape
/// `[batch, channels, seqlen]`; `weight` shape `[channels, width]`;
/// `bias` may be null (passed as zero-len `CudaStorageBytes`).
///
/// The kernel is causal: output `y[..., t]` depends only on `x[..., t-w+1 .. t]`.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_f32(
    x: &CudaStorageBytes,
    weight: &CudaStorageBytes,
    bias: Option<&CudaStorageBytes>,
    batch: usize, channels: usize, seqlen: usize, width: usize,
    use_silu: bool,
) -> Result<CudaStorageBytes> {
    causal_conv1d_inner(
        x, weight, bias, batch, channels, seqlen, width, use_silu,
        std::mem::size_of::<f32>(),
        sys::baracuda_kernels_causal_conv1d_f32_run,
        "causal_conv1d_f32",
    )
}
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_f16(
    x: &CudaStorageBytes, weight: &CudaStorageBytes, bias: Option<&CudaStorageBytes>,
    batch: usize, channels: usize, seqlen: usize, width: usize, use_silu: bool,
) -> Result<CudaStorageBytes> {
    causal_conv1d_inner(
        x, weight, bias, batch, channels, seqlen, width, use_silu,
        std::mem::size_of::<half::f16>(),
        sys::baracuda_kernels_causal_conv1d_f16_run,
        "causal_conv1d_f16",
    )
}
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_bf16(
    x: &CudaStorageBytes, weight: &CudaStorageBytes, bias: Option<&CudaStorageBytes>,
    batch: usize, channels: usize, seqlen: usize, width: usize, use_silu: bool,
) -> Result<CudaStorageBytes> {
    causal_conv1d_inner(
        x, weight, bias, batch, channels, seqlen, width, use_silu,
        std::mem::size_of::<half::bf16>(),
        sys::baracuda_kernels_causal_conv1d_bf16_run,
        "causal_conv1d_bf16",
    )
}
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_f64(
    x: &CudaStorageBytes, weight: &CudaStorageBytes, bias: Option<&CudaStorageBytes>,
    batch: usize, channels: usize, seqlen: usize, width: usize, use_silu: bool,
) -> Result<CudaStorageBytes> {
    causal_conv1d_inner(
        x, weight, bias, batch, channels, seqlen, width, use_silu,
        std::mem::size_of::<f64>(),
        sys::baracuda_kernels_causal_conv1d_f64_run,
        "causal_conv1d_f64",
    )
}

type CausalConv1dRun = unsafe extern "C" fn(
    i32, i32, i32, i32, i32,
    *const std::ffi::c_void, *const std::ffi::c_void, *const std::ffi::c_void,
    *mut std::ffi::c_void,
    *mut std::ffi::c_void, usize,
    *mut std::ffi::c_void,
) -> i32;

#[allow(clippy::too_many_arguments)]
fn causal_conv1d_inner(
    x: &CudaStorageBytes,
    weight: &CudaStorageBytes,
    bias: Option<&CudaStorageBytes>,
    batch: usize, channels: usize, seqlen: usize, width: usize,
    use_silu: bool,
    dtype_size_bytes: usize,
    kernel: CausalConv1dRun,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let device = x.device().clone();
    let numel = batch * channels * seqlen;
    let out_bytes = numel * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let bias_ptr = bias
        .map(|b| b.buffer().as_raw().0 as *const std::ffi::c_void)
        .unwrap_or(std::ptr::null());
    let status = unsafe {
        kernel(
            shape_i32(op_label, 0, batch)?,
            shape_i32(op_label, 1, channels)?,
            shape_i32(op_label, 2, seqlen)?,
            shape_i32(op_label, 3, width)?,
            if use_silu { 1 } else { 0 },
            x.buffer().as_raw().0 as *const std::ffi::c_void,
            weight.buffer().as_raw().0 as *const std::ffi::c_void,
            bias_ptr,
            out_buf.as_raw().0 as *mut std::ffi::c_void,
            std::ptr::null_mut(), 0,
            stream,
        )
    };
    check(status, op_label)?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out_buf), device, out_bytes))
}

// ──────── causal_conv1d FW with Fuel-IR pre-pad convention ────────
//
// Fuel's `Op::CausalConv1d` requires the caller to left-pad `x` with
// `kernel - 1` zeros (matches the mamba2.rs prefill convention):
//   x_fuel : [batch, channels, seq_in]   where seq_in = seq_out + kernel - 1
//   y_fuel : [batch, channels, seq_out]
//
// baracuda's `causal_conv1d_*` takes a single `seqlen` for both input
// and output, with internal zero-padding for the causal window's
// out-of-bounds reads. Equivalently: input length == output length.
//
// To bridge the convention gap, we pass baracuda the pre-padded `x`
// with `seqlen = seq_in` (its full length), receive a `seq_in`-long
// output buffer, and discard the first `kernel - 1` timesteps per
// (batch, channel) row via one `cuMemcpy2D` D→D — exactly the rows
// whose convolution windows are dominated by the leading zero-pad.
// The discarded fraction is `(kernel - 1) / seq_in`; for Mamba kernel
// = 4 it's a fraction of a percent of the convolution work, so the
// wasted compute is negligible relative to the cuBLAS-class kernel
// dispatch overhead we'd add by hand-rolling a strip kernel.

/// Strip the first `prepad_elems` elements from each of `rows` rows of
/// a contiguous `[rows, src_cols]` byte buffer; return a fresh
/// `[rows, src_cols - prepad_elems]` buffer. Single `cuMemcpy2D` D→D.
fn strip_prepad_d2d(
    src: &CudaStorageBytes,
    rows: usize,
    src_cols: usize,
    prepad_elems: usize,
    elem_bytes: usize,
) -> Result<CudaStorageBytes> {
    use baracuda_cuda_sys::driver;
    use baracuda_cuda_sys::types::{CUmemorytype, CUDA_MEMCPY2D};
    use baracuda_cuda_sys::CUdeviceptr;

    let device = src.device().clone();
    let dst_cols = src_cols - prepad_elems;
    let dst_bytes = rows
        .saturating_mul(dst_cols)
        .saturating_mul(elem_bytes);
    if dst_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let dst_buf = device.alloc_zeros::<u8>(dst_bytes)?;
    let src_pitch = src_cols * elem_bytes;
    let dst_pitch = dst_cols * elem_bytes;
    // Shift the source pointer by `prepad_elems * elem_bytes` so the
    // first byte of each src row is the first non-padding element.
    let src_dev = CUdeviceptr(src.buffer().as_raw().0 + (prepad_elems * elem_bytes) as u64);
    let p = CUDA_MEMCPY2D {
        src_memory_type: CUmemorytype::DEVICE,
        src_device:      src_dev,
        src_pitch,
        dst_memory_type: CUmemorytype::DEVICE,
        dst_device:      dst_buf.as_raw(),
        dst_pitch,
        width_in_bytes:  dst_pitch,
        height:          rows,
        ..Default::default()
    };
    let d = driver().map_err(|e| {
        fuel_ir::Error::Msg(format!("strip_prepad_d2d: driver(): {e:?}")).bt()
    })?;
    let cu = d.cu_memcpy_2d_async().map_err(|e| {
        fuel_ir::Error::Msg(format!("strip_prepad_d2d: cu_memcpy_2d_async: {e:?}")).bt()
    })?;
    let stream = device.stream().as_raw();
    let status = unsafe { cu(&p, stream) };
    if status.0 != 0 {
        return Err(fuel_ir::Error::Msg(format!(
            "strip_prepad_d2d: cuMemcpy2DAsync failed: status={status:?}",
        ))
        .bt());
    }
    Ok(CudaStorageBytes::from_parts(Arc::new(dst_buf), device, dst_bytes))
}

/// Causal conv1d forward in Fuel-IR convention: pre-padded `x` shape
/// `[batch, channels, seq_in]`, contiguous output `[batch, channels,
/// seq_out]` where `seq_in == seq_out + kernel - 1`. Bias is required
/// (matches the Fuel-IR builder gate).
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_fuel_prepad_f32(
    x: &CudaStorageBytes, weight: &CudaStorageBytes, bias: &CudaStorageBytes,
    batch: usize, channels: usize,
    seq_in: usize, seq_out: usize, kernel: usize,
    use_silu: bool,
) -> Result<CudaStorageBytes> {
    causal_conv1d_fuel_prepad_inner(
        x, weight, bias, batch, channels, seq_in, seq_out, kernel, use_silu,
        std::mem::size_of::<f32>(),
        causal_conv1d_f32, "causal_conv1d_fuel_prepad_f32",
    )
}
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_fuel_prepad_f64(
    x: &CudaStorageBytes, weight: &CudaStorageBytes, bias: &CudaStorageBytes,
    batch: usize, channels: usize,
    seq_in: usize, seq_out: usize, kernel: usize,
    use_silu: bool,
) -> Result<CudaStorageBytes> {
    causal_conv1d_fuel_prepad_inner(
        x, weight, bias, batch, channels, seq_in, seq_out, kernel, use_silu,
        std::mem::size_of::<f64>(),
        causal_conv1d_f64, "causal_conv1d_fuel_prepad_f64",
    )
}
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_fuel_prepad_bf16(
    x: &CudaStorageBytes, weight: &CudaStorageBytes, bias: &CudaStorageBytes,
    batch: usize, channels: usize,
    seq_in: usize, seq_out: usize, kernel: usize,
    use_silu: bool,
) -> Result<CudaStorageBytes> {
    causal_conv1d_fuel_prepad_inner(
        x, weight, bias, batch, channels, seq_in, seq_out, kernel, use_silu,
        std::mem::size_of::<half::bf16>(),
        causal_conv1d_bf16, "causal_conv1d_fuel_prepad_bf16",
    )
}
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_fuel_prepad_f16(
    x: &CudaStorageBytes, weight: &CudaStorageBytes, bias: &CudaStorageBytes,
    batch: usize, channels: usize,
    seq_in: usize, seq_out: usize, kernel: usize,
    use_silu: bool,
) -> Result<CudaStorageBytes> {
    causal_conv1d_fuel_prepad_inner(
        x, weight, bias, batch, channels, seq_in, seq_out, kernel, use_silu,
        std::mem::size_of::<half::f16>(),
        causal_conv1d_f16, "causal_conv1d_fuel_prepad_f16",
    )
}

type CausalConv1dRawFw = fn(
    &CudaStorageBytes, &CudaStorageBytes, Option<&CudaStorageBytes>,
    usize, usize, usize, usize, bool,
) -> Result<CudaStorageBytes>;

#[allow(clippy::too_many_arguments)]
fn causal_conv1d_fuel_prepad_inner(
    x: &CudaStorageBytes, weight: &CudaStorageBytes, bias: &CudaStorageBytes,
    batch: usize, channels: usize,
    seq_in: usize, seq_out: usize, kernel: usize,
    use_silu: bool,
    elem_bytes: usize,
    raw_fw: CausalConv1dRawFw,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    if seq_in != seq_out + kernel - 1 {
        return Err(fuel_ir::Error::Msg(format!(
            "{op_label}: seq_in={seq_in} must equal seq_out={seq_out} + kernel-1={}",
            kernel - 1,
        ))
        .bt());
    }
    // Call baracuda with seqlen == seq_in (pre-padded length). The
    // first `kernel - 1` output timesteps per (batch, channel) row are
    // convolutions of the leading zero-pad; the remaining `seq_out`
    // timesteps match Fuel's CPU kernel exactly.
    let raw_out = raw_fw(
        x, weight, Some(bias),
        batch, channels, seq_in, kernel, use_silu,
    )?;
    if kernel == 1 {
        // No prepad — raw_out is already Fuel-shaped.
        return Ok(raw_out);
    }
    strip_prepad_d2d(
        &raw_out,
        batch * channels,
        seq_in,
        kernel - 1,
        elem_bytes,
    )
}

/// Outputs from causal_conv1d backward: gradients matching the FW
/// inputs `x`, `weight`, `bias`. All are owned `CudaStorageBytes`
/// allocated by the wrapper.
pub struct CausalConv1dBackward {
    pub dx: CudaStorageBytes,
    pub dw: CudaStorageBytes,
    pub db: CudaStorageBytes,
}

/// Causal conv1d backward, F32. Allocates dx / dw / db internally;
/// bias gradient is always written (caller can drop it if bias was
/// null on the FW pass).
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_backward_f32(
    x: &CudaStorageBytes, weight: &CudaStorageBytes,
    bias: Option<&CudaStorageBytes>, dy: &CudaStorageBytes,
    batch: usize, channels: usize, seqlen: usize, width: usize,
    use_silu: bool,
) -> Result<CausalConv1dBackward> {
    causal_conv1d_backward_inner(
        x, weight, bias, dy, batch, channels, seqlen, width, use_silu,
        std::mem::size_of::<f32>(),
        sys::baracuda_kernels_causal_conv1d_f32_backward_run,
        "causal_conv1d_f32_backward",
    )
}
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_backward_f16(
    x: &CudaStorageBytes, weight: &CudaStorageBytes,
    bias: Option<&CudaStorageBytes>, dy: &CudaStorageBytes,
    batch: usize, channels: usize, seqlen: usize, width: usize, use_silu: bool,
) -> Result<CausalConv1dBackward> {
    causal_conv1d_backward_inner(
        x, weight, bias, dy, batch, channels, seqlen, width, use_silu,
        std::mem::size_of::<half::f16>(),
        sys::baracuda_kernels_causal_conv1d_f16_backward_run,
        "causal_conv1d_f16_backward",
    )
}
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_backward_bf16(
    x: &CudaStorageBytes, weight: &CudaStorageBytes,
    bias: Option<&CudaStorageBytes>, dy: &CudaStorageBytes,
    batch: usize, channels: usize, seqlen: usize, width: usize, use_silu: bool,
) -> Result<CausalConv1dBackward> {
    causal_conv1d_backward_inner(
        x, weight, bias, dy, batch, channels, seqlen, width, use_silu,
        std::mem::size_of::<half::bf16>(),
        sys::baracuda_kernels_causal_conv1d_bf16_backward_run,
        "causal_conv1d_bf16_backward",
    )
}
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_backward_f64(
    x: &CudaStorageBytes, weight: &CudaStorageBytes,
    bias: Option<&CudaStorageBytes>, dy: &CudaStorageBytes,
    batch: usize, channels: usize, seqlen: usize, width: usize, use_silu: bool,
) -> Result<CausalConv1dBackward> {
    causal_conv1d_backward_inner(
        x, weight, bias, dy, batch, channels, seqlen, width, use_silu,
        std::mem::size_of::<f64>(),
        sys::baracuda_kernels_causal_conv1d_f64_backward_run,
        "causal_conv1d_f64_backward",
    )
}

type CausalConv1dBackwardRun = unsafe extern "C" fn(
    i32, i32, i32, i32, i32,
    *const std::ffi::c_void, *const std::ffi::c_void, *const std::ffi::c_void, *const std::ffi::c_void,
    *mut std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void,
    *mut std::ffi::c_void, usize,
    *mut std::ffi::c_void,
) -> i32;

#[allow(clippy::too_many_arguments)]
fn causal_conv1d_backward_inner(
    x: &CudaStorageBytes, weight: &CudaStorageBytes,
    bias: Option<&CudaStorageBytes>, dy: &CudaStorageBytes,
    batch: usize, channels: usize, seqlen: usize, width: usize,
    use_silu: bool,
    dtype_size_bytes: usize,
    kernel: CausalConv1dBackwardRun,
    op_label: &'static str,
) -> Result<CausalConv1dBackward> {
    let device = x.device().clone();
    let dx_bytes = batch * channels * seqlen * dtype_size_bytes;
    let dw_bytes = channels * width * dtype_size_bytes;
    let db_bytes = channels * dtype_size_bytes;
    let dx_buf = device.alloc_zeros::<u8>(dx_bytes)?;
    let dw_buf = device.alloc_zeros::<u8>(dw_bytes)?;
    let db_buf = device.alloc_zeros::<u8>(db_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let bias_ptr = bias
        .map(|b| b.buffer().as_raw().0 as *const std::ffi::c_void)
        .unwrap_or(std::ptr::null());
    let status = unsafe {
        kernel(
            shape_i32(op_label, 0, batch)?,
            shape_i32(op_label, 1, channels)?,
            shape_i32(op_label, 2, seqlen)?,
            shape_i32(op_label, 3, width)?,
            if use_silu { 1 } else { 0 },
            x.buffer().as_raw().0      as *const std::ffi::c_void,
            weight.buffer().as_raw().0 as *const std::ffi::c_void,
            bias_ptr,
            dy.buffer().as_raw().0     as *const std::ffi::c_void,
            dx_buf.as_raw().0 as *mut std::ffi::c_void,
            dw_buf.as_raw().0 as *mut std::ffi::c_void,
            db_buf.as_raw().0 as *mut std::ffi::c_void,
            std::ptr::null_mut(), 0,
            stream,
        )
    };
    check(status, op_label)?;
    let device2 = device.clone();
    let device3 = device.clone();
    Ok(CausalConv1dBackward {
        dx: CudaStorageBytes::from_parts(Arc::new(dx_buf), device, dx_bytes),
        dw: CudaStorageBytes::from_parts(Arc::new(dw_buf), device2, dw_bytes),
        db: CudaStorageBytes::from_parts(Arc::new(db_buf), device3, db_bytes),
    })
}

// ───────────────────────── ssd_chunk_scan ─────────────────────────

/// Mamba-2 SSD chunk-scan forward. Inputs:
/// - `x`:   `[batch, seqlen, heads, head_dim]`
/// - `dt`:  `[batch, seqlen, heads]`
/// - `a`:   `[heads]`
/// - `b`/`c`: `[batch, seqlen, heads, state_dim]`
///
/// Output `y: [batch, seqlen, heads, head_dim]` allocated fresh.
/// `chunk_size` is the SSD chunk-scan block size; typically 256.
#[allow(clippy::too_many_arguments)]
pub fn ssd_chunk_scan_f32(
    x: &CudaStorageBytes, dt: &CudaStorageBytes, a: &CudaStorageBytes,
    b: &CudaStorageBytes, c: &CudaStorageBytes,
    batch: usize, seqlen: usize, heads: usize,
    head_dim: usize, state_dim: usize, chunk_size: usize,
) -> Result<CudaStorageBytes> {
    ssd_chunk_scan_inner(
        x, dt, a, b, c, batch, seqlen, heads, head_dim, state_dim, chunk_size,
        std::mem::size_of::<f32>(), 0,
        sys::baracuda_kernels_ssd_chunk_scan_f32_run,
        "ssd_chunk_scan_f32",
    )
}
#[allow(clippy::too_many_arguments)]
pub fn ssd_chunk_scan_f16(
    x: &CudaStorageBytes, dt: &CudaStorageBytes, a: &CudaStorageBytes,
    b: &CudaStorageBytes, c: &CudaStorageBytes,
    batch: usize, seqlen: usize, heads: usize,
    head_dim: usize, state_dim: usize, chunk_size: usize,
) -> Result<CudaStorageBytes> {
    ssd_chunk_scan_inner(
        x, dt, a, b, c, batch, seqlen, heads, head_dim, state_dim, chunk_size,
        std::mem::size_of::<half::f16>(), 1,
        sys::baracuda_kernels_ssd_chunk_scan_f16_run,
        "ssd_chunk_scan_f16",
    )
}
#[allow(clippy::too_many_arguments)]
pub fn ssd_chunk_scan_bf16(
    x: &CudaStorageBytes, dt: &CudaStorageBytes, a: &CudaStorageBytes,
    b: &CudaStorageBytes, c: &CudaStorageBytes,
    batch: usize, seqlen: usize, heads: usize,
    head_dim: usize, state_dim: usize, chunk_size: usize,
) -> Result<CudaStorageBytes> {
    ssd_chunk_scan_inner(
        x, dt, a, b, c, batch, seqlen, heads, head_dim, state_dim, chunk_size,
        std::mem::size_of::<half::bf16>(), 2,
        sys::baracuda_kernels_ssd_chunk_scan_bf16_run,
        "ssd_chunk_scan_bf16",
    )
}

type SsdChunkScanRun = unsafe extern "C" fn(
    i32, i32, i32, i32, i32, i32,
    *const std::ffi::c_void, *const std::ffi::c_void, *const std::ffi::c_void,
    *const std::ffi::c_void, *const std::ffi::c_void,
    *mut std::ffi::c_void,
    *mut std::ffi::c_void, usize,
    *mut std::ffi::c_void,
) -> i32;

#[allow(clippy::too_many_arguments)]
fn ssd_chunk_scan_inner(
    x: &CudaStorageBytes, dt: &CudaStorageBytes, a: &CudaStorageBytes,
    b: &CudaStorageBytes, c: &CudaStorageBytes,
    batch: usize, seqlen: usize, heads: usize,
    head_dim: usize, state_dim: usize, chunk_size: usize,
    dtype_size_bytes: usize, dtype_id: i32,
    kernel: SsdChunkScanRun,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let device = x.device().clone();
    let numel = batch * seqlen * heads * head_dim;
    let out_bytes = numel * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let ws_bytes = unsafe {
        sys::baracuda_kernels_ssd_chunk_scan_workspace_bytes(
            shape_i32(op_label, 0, batch)?,
            shape_i32(op_label, 1, seqlen)?,
            shape_i32(op_label, 2, heads)?,
            shape_i32(op_label, 3, head_dim)?,
            shape_i32(op_label, 4, state_dim)?,
            shape_i32(op_label, 5, chunk_size)?,
            dtype_id,
        )
    };
    let ws_buf = if ws_bytes > 0 { Some(device.alloc_zeros::<u8>(ws_bytes)?) } else { None };
    let ws_ptr = ws_buf
        .as_ref()
        .map(|b| b.as_raw().0 as *mut std::ffi::c_void)
        .unwrap_or(std::ptr::null_mut());
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let status = unsafe {
        kernel(
            shape_i32(op_label, 0, batch)?,
            shape_i32(op_label, 1, seqlen)?,
            shape_i32(op_label, 2, heads)?,
            shape_i32(op_label, 3, head_dim)?,
            shape_i32(op_label, 4, state_dim)?,
            shape_i32(op_label, 5, chunk_size)?,
            x.buffer().as_raw().0  as *const std::ffi::c_void,
            dt.buffer().as_raw().0 as *const std::ffi::c_void,
            a.buffer().as_raw().0  as *const std::ffi::c_void,
            b.buffer().as_raw().0  as *const std::ffi::c_void,
            c.buffer().as_raw().0  as *const std::ffi::c_void,
            out_buf.as_raw().0 as *mut std::ffi::c_void,
            ws_ptr, ws_bytes,
            stream,
        )
    };
    check(status, op_label)?;
    drop(ws_buf);
    Ok(CudaStorageBytes::from_parts(Arc::new(out_buf), device, out_bytes))
}

// ───────────────────────── selective_scan ─────────────────────────

/// Outputs from selective_scan forward: the per-element output `y`
/// and the final state `last_state` (used by autoregressive decode
/// to chain forward across calls).
pub struct SelectiveScanForward {
    pub y: CudaStorageBytes,
    pub last_state: CudaStorageBytes,
}

/// Mamba-1 selective scan forward. Inputs:
/// - `u`: `[batch, seqlen, dim]` element input
/// - `delta`: `[batch, seqlen, dim]` per-element delta
/// - `a`: `[dim, dstate]` state matrix
/// - `b`/`c`: `[batch, seqlen, dstate]`
/// - `d_skip` (optional, may be empty): `[dim]` D-skip
/// - `z` (optional): `[batch, seqlen, dim]` gating
/// - `delta_bias` (optional): `[dim]`
/// - `delta_softplus`: when true, apply softplus to delta + delta_bias
///
/// Outputs `y: [batch, seqlen, dim]`, `last_state: [batch, dim, dstate]`.
#[allow(clippy::too_many_arguments)]
pub fn selective_scan_f32(
    u: &CudaStorageBytes, delta: &CudaStorageBytes,
    a: &CudaStorageBytes, b: &CudaStorageBytes, c: &CudaStorageBytes,
    d_skip: Option<&CudaStorageBytes>, z: Option<&CudaStorageBytes>,
    delta_bias: Option<&CudaStorageBytes>,
    batch: usize, seqlen: usize, dim: usize, dstate: usize,
    delta_softplus: bool,
) -> Result<SelectiveScanForward> {
    selective_scan_inner(
        u, delta, a, b, c, d_skip, z, delta_bias,
        batch, seqlen, dim, dstate, delta_softplus,
        std::mem::size_of::<f32>(), 0,
        sys::baracuda_kernels_selective_scan_f32_run,
        "selective_scan_f32",
    )
}
#[allow(clippy::too_many_arguments)]
pub fn selective_scan_f16(
    u: &CudaStorageBytes, delta: &CudaStorageBytes,
    a: &CudaStorageBytes, b: &CudaStorageBytes, c: &CudaStorageBytes,
    d_skip: Option<&CudaStorageBytes>, z: Option<&CudaStorageBytes>,
    delta_bias: Option<&CudaStorageBytes>,
    batch: usize, seqlen: usize, dim: usize, dstate: usize,
    delta_softplus: bool,
) -> Result<SelectiveScanForward> {
    selective_scan_inner(
        u, delta, a, b, c, d_skip, z, delta_bias,
        batch, seqlen, dim, dstate, delta_softplus,
        std::mem::size_of::<half::f16>(), 1,
        sys::baracuda_kernels_selective_scan_f16_run,
        "selective_scan_f16",
    )
}
#[allow(clippy::too_many_arguments)]
pub fn selective_scan_bf16(
    u: &CudaStorageBytes, delta: &CudaStorageBytes,
    a: &CudaStorageBytes, b: &CudaStorageBytes, c: &CudaStorageBytes,
    d_skip: Option<&CudaStorageBytes>, z: Option<&CudaStorageBytes>,
    delta_bias: Option<&CudaStorageBytes>,
    batch: usize, seqlen: usize, dim: usize, dstate: usize,
    delta_softplus: bool,
) -> Result<SelectiveScanForward> {
    selective_scan_inner(
        u, delta, a, b, c, d_skip, z, delta_bias,
        batch, seqlen, dim, dstate, delta_softplus,
        std::mem::size_of::<half::bf16>(), 2,
        sys::baracuda_kernels_selective_scan_bf16_run,
        "selective_scan_bf16",
    )
}

type SelectiveScanRun = unsafe extern "C" fn(
    i32, i32, i32, i32, i32,
    *const std::ffi::c_void, *const std::ffi::c_void, *const std::ffi::c_void,
    *const std::ffi::c_void, *const std::ffi::c_void,
    *const std::ffi::c_void, *const std::ffi::c_void, *const std::ffi::c_void,
    *mut std::ffi::c_void, *mut std::ffi::c_void,
    *mut std::ffi::c_void, usize,
    *mut std::ffi::c_void,
) -> i32;

#[allow(clippy::too_many_arguments)]
fn selective_scan_inner(
    u: &CudaStorageBytes, delta: &CudaStorageBytes,
    a: &CudaStorageBytes, b: &CudaStorageBytes, c: &CudaStorageBytes,
    d_skip: Option<&CudaStorageBytes>, z: Option<&CudaStorageBytes>,
    delta_bias: Option<&CudaStorageBytes>,
    batch: usize, seqlen: usize, dim: usize, dstate: usize,
    delta_softplus: bool,
    dtype_size_bytes: usize, dtype_id: i32,
    kernel: SelectiveScanRun,
    op_label: &'static str,
) -> Result<SelectiveScanForward> {
    let device = u.device().clone();
    let y_bytes = batch * seqlen * dim * dtype_size_bytes;
    let last_state_bytes = batch * dim * dstate * dtype_size_bytes;
    let y_buf = device.alloc_zeros::<u8>(y_bytes)?;
    let last_state_buf = device.alloc_zeros::<u8>(last_state_bytes)?;
    let ws_bytes = unsafe {
        sys::baracuda_kernels_selective_scan_workspace_bytes(
            shape_i32(op_label, 0, batch)?,
            shape_i32(op_label, 1, seqlen)?,
            shape_i32(op_label, 2, dim)?,
            shape_i32(op_label, 3, dstate)?,
            dtype_id,
        )
    };
    let ws_buf = if ws_bytes > 0 { Some(device.alloc_zeros::<u8>(ws_bytes)?) } else { None };
    let ws_ptr = ws_buf
        .as_ref()
        .map(|b| b.as_raw().0 as *mut std::ffi::c_void)
        .unwrap_or(std::ptr::null_mut());
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let d_skip_ptr = d_skip
        .map(|p| p.buffer().as_raw().0 as *const std::ffi::c_void)
        .unwrap_or(std::ptr::null());
    let z_ptr = z
        .map(|p| p.buffer().as_raw().0 as *const std::ffi::c_void)
        .unwrap_or(std::ptr::null());
    let delta_bias_ptr = delta_bias
        .map(|p| p.buffer().as_raw().0 as *const std::ffi::c_void)
        .unwrap_or(std::ptr::null());
    let status = unsafe {
        kernel(
            shape_i32(op_label, 0, batch)?,
            shape_i32(op_label, 1, seqlen)?,
            shape_i32(op_label, 2, dim)?,
            shape_i32(op_label, 3, dstate)?,
            if delta_softplus { 1 } else { 0 },
            u.buffer().as_raw().0     as *const std::ffi::c_void,
            delta.buffer().as_raw().0 as *const std::ffi::c_void,
            a.buffer().as_raw().0     as *const std::ffi::c_void,
            b.buffer().as_raw().0     as *const std::ffi::c_void,
            c.buffer().as_raw().0     as *const std::ffi::c_void,
            d_skip_ptr, z_ptr, delta_bias_ptr,
            y_buf.as_raw().0          as *mut std::ffi::c_void,
            last_state_buf.as_raw().0 as *mut std::ffi::c_void,
            ws_ptr, ws_bytes,
            stream,
        )
    };
    check(status, op_label)?;
    drop(ws_buf);
    let device2 = device.clone();
    Ok(SelectiveScanForward {
        y: CudaStorageBytes::from_parts(Arc::new(y_buf), device, y_bytes),
        last_state: CudaStorageBytes::from_parts(Arc::new(last_state_buf), device2, last_state_bytes),
    })
}

// Backward primitives for ssd_chunk_scan + selective_scan are large
// multi-output kernels (5 and 8 gradient tensors respectively). The
// Op-surface session adding `Op::SelectiveScan` / `Op::SsdChunkScan`
// will land the BW wrappers alongside the autograd nodes — keeps
// signature design close to the Op-level integration. baracuda's BW
// `_run` symbols are already linkable; nothing in baracuda changes.

// ──────────────────────── can_implement ────────────────────────
//
// Pre-launch host-side validators. Call before allocating output
// buffers / sizing workspaces; return Ok(()) iff baracuda accepts
// the shape combination.

macro_rules! mamba_can_impl {
    ($name:ident, $sys:ident, ($($pname:ident: $pty:ty),*) $(,)?) => {
        pub fn $name($($pname: $pty),*) -> Result<()> {
            let status = unsafe { sys::$sys($($pname),*) };
            check(status, stringify!($name))
        }
    };
}

mamba_can_impl!(causal_conv1d_f32_can_implement,  baracuda_kernels_causal_conv1d_f32_can_implement,  (batch: i32, channels: i32, seqlen: i32, width: i32));
mamba_can_impl!(causal_conv1d_f64_can_implement,  baracuda_kernels_causal_conv1d_f64_can_implement,  (batch: i32, channels: i32, seqlen: i32, width: i32));
mamba_can_impl!(causal_conv1d_f16_can_implement,  baracuda_kernels_causal_conv1d_f16_can_implement,  (batch: i32, channels: i32, seqlen: i32, width: i32));
mamba_can_impl!(causal_conv1d_bf16_can_implement, baracuda_kernels_causal_conv1d_bf16_can_implement, (batch: i32, channels: i32, seqlen: i32, width: i32));

mamba_can_impl!(ssd_chunk_scan_f32_can_implement,  baracuda_kernels_ssd_chunk_scan_f32_can_implement,  (batch: i32, seqlen: i32, heads: i32, head_dim: i32, state_dim: i32, chunk_size: i32));
mamba_can_impl!(ssd_chunk_scan_f16_can_implement,  baracuda_kernels_ssd_chunk_scan_f16_can_implement,  (batch: i32, seqlen: i32, heads: i32, head_dim: i32, state_dim: i32, chunk_size: i32));
mamba_can_impl!(ssd_chunk_scan_bf16_can_implement, baracuda_kernels_ssd_chunk_scan_bf16_can_implement, (batch: i32, seqlen: i32, heads: i32, head_dim: i32, state_dim: i32, chunk_size: i32));

mamba_can_impl!(selective_scan_f32_can_implement,  baracuda_kernels_selective_scan_f32_can_implement,  (batch: i32, seqlen: i32, dim: i32, dstate: i32));
mamba_can_impl!(selective_scan_f16_can_implement,  baracuda_kernels_selective_scan_f16_can_implement,  (batch: i32, seqlen: i32, dim: i32, dstate: i32));
mamba_can_impl!(selective_scan_bf16_can_implement, baracuda_kernels_selective_scan_bf16_can_implement, (batch: i32, seqlen: i32, dim: i32, dstate: i32));

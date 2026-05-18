//! Normalization kernels from `baracuda-kernels-sys`.
//!
//! Coverage today: `rms_norm` + `layer_norm` for the LastDim case
//! Fuel's `OpKind::RmsNormLastDim` / `LayerNormLastDim` expects.
//! Fuel's params (`OpParams::NormLastDim { outer_count, last_dim,
//! eps }`) decompose to baracuda's
//! `(norm_axes_mask = 1 << (rank-1), norm_total_extent = last_dim)`
//! on a flattened `[outer_count, last_dim]` shape.
//!
//! Baracuda's surface also includes BatchNorm / GroupNorm /
//! InstanceNorm — those don't have matching Fuel `OpKind`s today;
//! they ship when Fuel grows those primitive ops.
//!
//! ## Aux outputs
//!
//! - RmsNorm writes `rms_out` (per-instance rms value) used by the
//!   backward kernel. Fuel's forward-only path doesn't need it; the
//!   wrapper allocates a scratch buffer to satisfy the kernel's
//!   non-null contract.
//! - LayerNorm writes `mean_out` + `inv_std_out` (per-instance
//!   summaries). Same scratch-buffer strategy.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::{DType, Result};

use crate::byte_storage::CudaStorageBytes;
use crate::error::CudaError;

use super::scratch::Workspace;
use super::status::check;

type RmsNormRun = unsafe extern "C" fn(
    eps: f32,
    numel: i64,
    rank: i32,
    shape: *const i32,
    stride_x: *const i64,
    stride_y: *const i64,
    stride_rms: *const i64,
    norm_axes_mask: i32,
    norm_total_extent: i32,
    x: *const std::ffi::c_void,
    gamma: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    rms_out: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

type LayerNormRun = unsafe extern "C" fn(
    eps: f32,
    numel: i64,
    rank: i32,
    shape: *const i32,
    stride_x: *const i64,
    stride_y: *const i64,
    stride_save: *const i64,
    norm_axes_mask: i32,
    norm_total_extent: i32,
    x: *const std::ffi::c_void,
    gamma: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    mean_out: *mut std::ffi::c_void,
    inv_std_out: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Driver for the LastDim RMS-norm shape Fuel exposes today.
/// Treats input as `[outer_count, last_dim]` (rank 2). Allocates a
/// fresh contig output of the same byte count. Aux `rms_out` is
/// a scratch buffer (size = outer_count × sizeof(T)) discarded
/// after the call.
fn rms_norm_last_dim_run(
    src: &CudaStorageBytes,
    outer_count: usize,
    last_dim: usize,
    eps: f64,
    kernel: RmsNormRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let device = src.device().clone();
    let numel: i64 = (outer_count * last_dim) as i64;
    let out_bytes = (numel as usize) * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let rms_buf = device.alloc_zeros::<u8>(outer_count * dtype_size_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    // Flattened [outer_count, last_dim] rank-2 representation; the
    // axis mask is the bit for the last axis.
    let oc = i32::try_from(outer_count).map_err(|_| {
        fuel_core_types::Error::cuda(CudaError::BaracudaShapeOverflow {
            op: op_label,
            dim_index: 0,
            dim_value: outer_count,
        })
    })?;
    let ld = i32::try_from(last_dim).map_err(|_| {
        fuel_core_types::Error::cuda(CudaError::BaracudaShapeOverflow {
            op: op_label,
            dim_index: 1,
            dim_value: last_dim,
        })
    })?;
    let shape: [i32; 2] = [oc, ld];
    let stride_x: [i64; 2] = [last_dim as i64, 1];
    let stride_y: [i64; 2] = [last_dim as i64, 1];
    // rms_out has shape [outer_count]; its stride is rank-1 → [1].
    // The norm kernel reads `stride_rms` as rank-2 too, with the
    // norm-axis stride being 0 (no extent in rms_out for the
    // normalized dim).
    let stride_rms: [i64; 2] = [1, 0];

    let x_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;
    let rms_ptr = rms_buf.as_raw().0 as *mut std::ffi::c_void;

    // SAFETY: pointers + lengths validated; shape/stride buffers
    // owned through the call.
    let status = unsafe {
        kernel(
            eps as f32,
            numel,
            2,
            shape.as_ptr(),
            stride_x.as_ptr(),
            stride_y.as_ptr(),
            stride_rms.as_ptr(),
            1 << 1, // bit for last (= rank-1 = 1) axis
            ld,
            x_ptr,
            std::ptr::null(), // no affine gamma
            y_ptr,
            rms_ptr,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    drop(rms_buf);
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out_buf),
        device,
        out_bytes,
    ))
}

/// Driver for the LastDim LayerNorm shape. Same flattening +
/// axis-mask logic as RMS-norm. Aux `mean_out` and `inv_std_out`
/// are scratch (size = outer_count × sizeof(T) each).
fn layer_norm_last_dim_run(
    src: &CudaStorageBytes,
    outer_count: usize,
    last_dim: usize,
    eps: f64,
    kernel: LayerNormRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let device = src.device().clone();
    let numel: i64 = (outer_count * last_dim) as i64;
    let out_bytes = (numel as usize) * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let mean_buf = device.alloc_zeros::<u8>(outer_count * dtype_size_bytes)?;
    let inv_std_buf = device.alloc_zeros::<u8>(outer_count * dtype_size_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let oc = i32::try_from(outer_count).map_err(|_| {
        fuel_core_types::Error::cuda(CudaError::BaracudaShapeOverflow {
            op: op_label,
            dim_index: 0,
            dim_value: outer_count,
        })
    })?;
    let ld = i32::try_from(last_dim).map_err(|_| {
        fuel_core_types::Error::cuda(CudaError::BaracudaShapeOverflow {
            op: op_label,
            dim_index: 1,
            dim_value: last_dim,
        })
    })?;
    let shape: [i32; 2] = [oc, ld];
    let stride_x: [i64; 2] = [last_dim as i64, 1];
    let stride_y: [i64; 2] = [last_dim as i64, 1];
    let stride_save: [i64; 2] = [1, 0];

    let x_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;
    let mean_ptr = mean_buf.as_raw().0 as *mut std::ffi::c_void;
    let inv_std_ptr = inv_std_buf.as_raw().0 as *mut std::ffi::c_void;

    let status = unsafe {
        kernel(
            eps as f32,
            numel,
            2,
            shape.as_ptr(),
            stride_x.as_ptr(),
            stride_y.as_ptr(),
            stride_save.as_ptr(),
            1 << 1,
            ld,
            x_ptr,
            std::ptr::null(),
            std::ptr::null(),
            y_ptr,
            mean_ptr,
            inv_std_ptr,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    drop(mean_buf);
    drop(inv_std_buf);
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out_buf),
        device,
        out_bytes,
    ))
}

macro_rules! rms_norm_kernel {
    ($name:ident, $dtype_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `rms_norm_", stringify!($dtype_stem), "` LastDim kernel.")]
            pub fn $name(
                src: &CudaStorageBytes,
                outer_count: usize,
                last_dim: usize,
                eps: f64,
            ) -> Result<CudaStorageBytes> {
                rms_norm_last_dim_run(
                    src,
                    outer_count,
                    last_dim,
                    eps,
                    sys::[<baracuda_kernels_rms_norm_ $dtype_stem _run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

macro_rules! layer_norm_kernel {
    ($name:ident, $dtype_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `layer_norm_", stringify!($dtype_stem), "` LastDim kernel.")]
            pub fn $name(
                src: &CudaStorageBytes,
                outer_count: usize,
                last_dim: usize,
                eps: f64,
            ) -> Result<CudaStorageBytes> {
                layer_norm_last_dim_run(
                    src,
                    outer_count,
                    last_dim,
                    eps,
                    sys::[<baracuda_kernels_layer_norm_ $dtype_stem _run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

rms_norm_kernel!(rms_norm_last_dim_f32, f32, 4, "rms_norm_last_dim_f32");
rms_norm_kernel!(rms_norm_last_dim_f16, f16, 2, "rms_norm_last_dim_f16");
rms_norm_kernel!(rms_norm_last_dim_bf16, bf16, 2, "rms_norm_last_dim_bf16");
rms_norm_kernel!(rms_norm_last_dim_f64, f64, 8, "rms_norm_last_dim_f64");

layer_norm_kernel!(layer_norm_last_dim_f32, f32, 4, "layer_norm_last_dim_f32");
layer_norm_kernel!(layer_norm_last_dim_f16, f16, 2, "layer_norm_last_dim_f16");
layer_norm_kernel!(layer_norm_last_dim_bf16, bf16, 2, "layer_norm_last_dim_bf16");
layer_norm_kernel!(layer_norm_last_dim_f64, f64, 8, "layer_norm_last_dim_f64");

/// Byte-size lookup for norm dtypes.
pub fn dtype_byte_size(dt: DType) -> usize {
    match dt {
        DType::F32 => 4,
        DType::F64 => 8,
        DType::F16 | DType::BF16 => 2,
        _ => 0,
    }
}

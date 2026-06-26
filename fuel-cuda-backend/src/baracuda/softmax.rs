//! Softmax / LogSoftmax / GumbelSoftmax / Sparsemax kernels from
//! `baracuda-kernels-sys`.
//!
//! Coverage today: `softmax` + `log_softmax` for the LastDim case
//! Fuel's `OpKind::SoftmaxLastDim` / `LogSoftmaxLastDim` expects.
//!
//! Baracuda's signature is shape + stride driven (`softmax_axis`,
//! `softmax_extent`); we flatten Fuel's
//! `OpParams::SoftmaxLastDim { outer_count, last_dim }` into a
//! rank-2 `[outer_count, last_dim]` shape with the softmax over
//! the last axis.
//!
//! GumbelSoftmax / Sparsemax ship in baracuda alpha.27 but don't
//! have matching Fuel `OpKind`s today; they wire up when Fuel
//! grows those primitives.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_ir::{DType, Layout, Result};

use crate::byte_storage::CudaStorageBytes;

use super::scratch::Workspace;
use super::status::check;

type SoftmaxRun = unsafe extern "C" fn(
    numel: i64,
    rank: i32,
    shape: *const i32,
    stride_x: *const i64,
    stride_y: *const i64,
    softmax_axis: i32,
    softmax_extent: i32,
    softmax_stride_x: i64,
    softmax_stride_y: i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Driver for the LastDim softmax / log-softmax shape. Flattens
/// to rank-2 `[outer_count, last_dim]`, dispatches across the last
/// axis. Output is a fresh contig buffer with `numel × sizeof(T)`
/// bytes.
fn softmax_last_dim_run(
    src: &CudaStorageBytes,
    src_layout: &Layout,
    kernel: SoftmaxRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let device = src.device().clone();
    let dims = src_layout.shape().dims();
    let rank = dims.len();
    let last_dim = *dims.last().ok_or_else(|| fuel_ir::Error::Msg(
        format!("{op_label}: rank-0 input not supported"),
    ).bt())?;
    let numel: i64 = src_layout.shape().elem_count() as i64;
    let out_bytes = (numel as usize) * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    // Build rank-N shape + per-input strides from the layout.
    let mut shape_i32: Vec<i32> = Vec::with_capacity(rank);
    for (i, &d) in dims.iter().enumerate() {
        shape_i32.push(i32::try_from(d).map_err(|_| {
            fuel_ir::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                op: op_label, dim_index: i, dim_value: d,
            })
        })?);
    }
    let stride_x: Vec<i64> = src_layout.stride().iter().map(|&s| s as i64).collect();
    // Output is freshly allocated contig over the input's shape.
    let stride_y: Vec<i64> = {
        let mut s = vec![1_i64; rank];
        for d in (0..rank.saturating_sub(1)).rev() {
            s[d] = s[d + 1] * dims[d + 1] as i64;
        }
        s
    };
    let ld_i32 = i32::try_from(last_dim).map_err(|_| {
        fuel_ir::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label, dim_index: rank - 1, dim_value: last_dim,
        })
    })?;
    let softmax_axis = (rank - 1) as i32;
    let softmax_stride_x: i64 = stride_x[rank - 1];
    let softmax_stride_y: i64 = stride_y[rank - 1];

    let x_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    let status = unsafe {
        kernel(
            numel,
            rank as i32,
            shape_i32.as_ptr(),
            stride_x.as_ptr(),
            stride_y.as_ptr(),
            softmax_axis,
            ld_i32,
            softmax_stride_x,
            softmax_stride_y,
            x_ptr,
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

macro_rules! softmax_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `", $op_label, "` LastDim kernel.")]
            pub fn $name(
                src: &CudaStorageBytes,
                src_layout: &Layout,
            ) -> Result<CudaStorageBytes> {
                softmax_last_dim_run(
                    src,
                    src_layout,
                    sys::[<baracuda_kernels_ $sys_stem _run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

softmax_kernel!(softmax_last_dim_f32, softmax_f32, 4, "softmax_last_dim_f32");
softmax_kernel!(softmax_last_dim_f16, softmax_f16, 2, "softmax_last_dim_f16");
softmax_kernel!(softmax_last_dim_bf16, softmax_bf16, 2, "softmax_last_dim_bf16");
softmax_kernel!(softmax_last_dim_f64, softmax_f64, 8, "softmax_last_dim_f64");

softmax_kernel!(log_softmax_last_dim_f32, log_softmax_f32, 4, "log_softmax_last_dim_f32");
softmax_kernel!(log_softmax_last_dim_f16, log_softmax_f16, 2, "log_softmax_last_dim_f16");
softmax_kernel!(log_softmax_last_dim_bf16, log_softmax_bf16, 2, "log_softmax_last_dim_bf16");
softmax_kernel!(log_softmax_last_dim_f64, log_softmax_f64, 8, "log_softmax_last_dim_f64");

/// Byte-size lookup for softmax dtypes.
pub fn dtype_byte_size(dt: DType) -> usize {
    match dt {
        DType::F32 => 4,
        DType::F64 => 8,
        DType::F16 | DType::BF16 => 2,
        _ => 0,
    }
}

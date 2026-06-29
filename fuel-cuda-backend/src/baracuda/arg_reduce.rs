//! ArgMaxDim / ArgMinDim kernels from `baracuda-kernels-sys`.
//! Alpha.28 added U32 + I32 output variants alongside the existing
//! I64 default — the wrapper picks the output-dtype variant by trait
//! tag. Fuel's binding-table key is `[input_dt, output_dt]` and the
//! existing dispatch path hard-codes U32 output (sibling to the PTX
//! `fast_argmax/fast_argmin` kernels). I64 + I32 variants are
//! exposed too for future use.
//!
//! ## ABI
//!
//! ```text
//! fn run(
//!     output_numel: i64,
//!     rank: i32,
//!     output_shape: *const i32,    // keepdim-style: reduce axis = 1
//!     stride_x: *const i64,        // strides over input shape (rank N)
//!     stride_y: *const i64,        // strides over output shape (rank N, reduce axis = 1)
//!     reduce_axis: i32,
//!     reduce_extent: i32,
//!     reduce_stride_x: i64,
//!     x: *const c_void,
//!     y: *mut c_void,
//!     workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
//! ) -> i32
//! ```
//!
//! Ties are broken by first-occurrence (smallest index wins) — same as
//! Fuel's CPU/PTX paths.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_ir::{Error, Layout, Result, Shape};

use crate::byte_storage::CudaStorageBytes;
use crate::error::CudaError;

use super::scratch::Workspace;
use super::status::check;

type ArgReduceRun = unsafe extern "C" fn(
    output_numel: i64,
    rank: i32,
    output_shape: *const i32,
    stride_x: *const i64,
    stride_y: *const i64,
    reduce_axis: i32,
    reduce_extent: i32,
    reduce_stride_x: i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Single-axis argmax/argmin driver. Internally uses a keepdim-shape
/// output layout (reduce axis kept as size 1) — that matches baracuda's
/// expected `output_shape` + stride_y inputs. The returned byte buffer
/// holds `out_numel * dst_size_bytes` bytes; Fuel's graph node carries
/// the squeezed shape so downstream consumers see rank N-1.
fn arg_reduce_run(
    src: &CudaStorageBytes,
    src_layout: &Layout,
    dim: usize,
    src_size_bytes: usize,
    dst_size_bytes: usize,
    kernel: ArgReduceRun,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let device = src.device().clone();
    let src_dims = src_layout.shape().dims();
    if dim >= src_dims.len() {
        return Err(Error::Msg(format!(
            "{op_label}: reduce axis {dim} out of bounds for rank {}",
            src_dims.len(),
        ))
        .bt());
    }
    if src_dims[dim] == 0 {
        return Err(Error::Msg(format!(
            "{op_label}: reduce dim {dim} has size 0 — argmax/argmin undefined",
        ))
        .bt());
    }
    let reduce_extent = src_dims[dim];
    let reduce_stride = src_layout.stride()[dim];

    // Output shape: same as input with `dim` set to 1 (keepdim-style).
    let mut out_shape: Vec<usize> = src_dims.to_vec();
    out_shape[dim] = 1;
    let out_layout = Layout::contiguous(Shape::from_dims(&out_shape));
    let out_numel: i64 = out_shape.iter().product::<usize>() as i64;
    let out_bytes = (out_numel as usize) * dst_size_bytes;

    if src.len_bytes() != (src_dims.iter().product::<usize>()) * src_size_bytes {
        return Err(Error::Msg(format!(
            "{op_label}: src.len_bytes={} disagrees with layout shape {:?} × {} bytes/elem",
            src.len_bytes(),
            src_dims,
            src_size_bytes,
        ))
        .bt());
    }
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }

    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    // baracuda's i32 shape buffer.
    let mut shape_i32: Vec<i32> = Vec::with_capacity(out_shape.len());
    for (i, &d) in out_shape.iter().enumerate() {
        shape_i32.push(i32::try_from(d).map_err(|_| {
            Error::cuda(CudaError::BaracudaShapeOverflow {
                op: op_label,
                dim_index: i,
                dim_value: d,
            })
        })?);
    }
    let stride_x: Vec<i64> = src_layout.stride().iter().map(|&s| s as i64).collect();
    let stride_y: Vec<i64> = out_layout.stride().iter().map(|&s| s as i64).collect();
    let rank = shape_i32.len() as i32;

    let x_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    // SAFETY: shape/stride buffers stay alive for the duration of the
    // call; pointers + extents validated above; workspace null/0 (no
    // scratch needed for argmax/argmin).
    let status = unsafe {
        kernel(
            out_numel,
            rank,
            shape_i32.as_ptr(),
            stride_x.as_ptr(),
            stride_y.as_ptr(),
            dim as i32,
            i32::try_from(reduce_extent).map_err(|_| {
                Error::cuda(CudaError::BaracudaShapeOverflow {
                    op: op_label,
                    dim_index: dim,
                    dim_value: reduce_extent,
                })
            })?,
            reduce_stride as i64,
            x_ptr,
            y_ptr,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out_buf),
        device,
        out_bytes,
    ))
}

macro_rules! arg_reduce_kernel {
    ($name:ident, $sys_stem:ident, $src_size:expr, $dst_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `", $op_label, "` kernel.")]
            pub fn $name(
                src: &CudaStorageBytes,
                src_layout: &Layout,
                dim: usize,
            ) -> Result<CudaStorageBytes> {
                arg_reduce_run(
                    src,
                    src_layout,
                    dim,
                    $src_size,
                    $dst_size,
                    sys::[<baracuda_kernels_arg_reduce_ $sys_stem _run>],
                    $op_label,
                )
            }
        }
    };
}

// ---------------------------------------------------------------------------
// U32 output variants (alpha.28). Match Fuel's existing `[F32, U32]` /
// `[F64, U32]` / `[F16, U32]` / `[BF16, U32]` binding-table keys.
// ---------------------------------------------------------------------------

arg_reduce_kernel!(argmax_dim_u32_f32, argmax_f32_u32, 4, 4, "argmax_dim_u32_f32");
arg_reduce_kernel!(argmin_dim_u32_f32, argmin_f32_u32, 4, 4, "argmin_dim_u32_f32");
arg_reduce_kernel!(argmax_dim_u32_f64, argmax_f64_u32, 8, 4, "argmax_dim_u32_f64");
arg_reduce_kernel!(argmin_dim_u32_f64, argmin_f64_u32, 8, 4, "argmin_dim_u32_f64");
arg_reduce_kernel!(argmax_dim_u32_f16, argmax_f16_u32, 2, 4, "argmax_dim_u32_f16");
arg_reduce_kernel!(argmin_dim_u32_f16, argmin_f16_u32, 2, 4, "argmin_dim_u32_f16");
arg_reduce_kernel!(argmax_dim_u32_bf16, argmax_bf16_u32, 2, 4, "argmax_dim_u32_bf16");
arg_reduce_kernel!(argmin_dim_u32_bf16, argmin_bf16_u32, 2, 4, "argmin_dim_u32_bf16");

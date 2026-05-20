//! Flip (reverse along selected axes) kernels from
//! `baracuda-kernels-sys`. Per-dtype symbols cover f32/f16/bf16/f64.
//!
//! baracuda's Flip operates on arbitrary-rank tensors with a
//! per-axis `flip_axes[d]` boolean (1 = reverse this axis, 0 = no-
//! op) plus separate input/output strides. Fuel's
//! `OpParams::Flip { outer_count, dim_size, inner_count }` only
//! supports a single-axis flip via the flat-3-axis view; this
//! integration presents the input to baracuda as a rank-3
//! `[outer, dim, inner]` shape with `flip_axes = [0, 1, 0]`.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::{Error, Result};

use crate::byte_storage::CudaStorageBytes;

use super::status::check;

/// FFI signature shared by every Flip dtype symbol.
type FlipRun = unsafe extern "C" fn(
    numel: i64,
    rank: i32,
    shape: *const i32,
    flip_axes: *const i32,
    stride_x: *const i64,
    stride_y: *const i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Run a single-axis flip via baracuda's rank-3 view. Fuel's IR
/// supports one flip dim at a time; multi-axis flip composes
/// upstream by chaining Flip nodes.
fn flip_run(
    input: &CudaStorageBytes,
    outer_count: usize,
    dim_size: usize,
    inner_count: usize,
    kernel: FlipRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let device = input.device().clone();
    let numel = outer_count * dim_size * inner_count;
    let out_bytes = numel * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let i32_or = |dim_index: usize, dim_value: usize| -> Result<i32> {
        i32::try_from(dim_value).map_err(|_| {
            Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                op: op_label,
                dim_index,
                dim_value,
            })
        })
    };
    let shape_i32: [i32; 3] = [
        i32_or(0, outer_count)?,
        i32_or(1, dim_size)?,
        i32_or(2, inner_count)?,
    ];
    // Flip the middle axis only — Fuel's OpParams::Flip is one-axis.
    let flip_axes_i32: [i32; 3] = [0, 1, 0];
    // Row-major contiguous strides for both input and output.
    let stride_x_i64: [i64; 3] = [
        (dim_size * inner_count) as i64,
        inner_count as i64,
        1,
    ];
    let stride_y_i64 = stride_x_i64;
    let out = device.alloc_zeros::<u8>(out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let in_ptr = input.buffer().as_raw().0 as *const std::ffi::c_void;
    let out_ptr = out.as_raw().0 as *mut std::ffi::c_void;
    // SAFETY: pointers valid; stack-arrays live for the FFI call.
    let status = unsafe {
        kernel(
            numel as i64,
            3,
            shape_i32.as_ptr(),
            flip_axes_i32.as_ptr(),
            stride_x_i64.as_ptr(),
            stride_y_i64.as_ptr(),
            in_ptr,
            out_ptr,
            std::ptr::null_mut(),
            0,
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out), device, out_bytes))
}

macro_rules! flip_kernel {
    ($name:ident, $sys_fn:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        #[doc = concat!("Baracuda `", $op_label, "` — single-axis flip.")]
        pub fn $name(
            input: &CudaStorageBytes,
            outer_count: usize,
            dim_size: usize,
            inner_count: usize,
        ) -> Result<CudaStorageBytes> {
            flip_run(
                input, outer_count, dim_size, inner_count,
                sys::$sys_fn,
                $op_label,
                $dtype_size,
            )
        }
    };
}

flip_kernel!(flip_f32,  baracuda_kernels_flip_f32_run,  4, "flip_f32");
flip_kernel!(flip_f64,  baracuda_kernels_flip_f64_run,  8, "flip_f64");
flip_kernel!(flip_f16,  baracuda_kernels_flip_f16_run,  2, "flip_f16");
flip_kernel!(flip_bf16, baracuda_kernels_flip_bf16_run, 2, "flip_bf16");

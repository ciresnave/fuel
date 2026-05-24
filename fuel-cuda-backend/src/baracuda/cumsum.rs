//! CumSum (inclusive prefix sum along one axis) kernels from
//! `baracuda-kernels-sys`. Per-dtype symbols cover f32/f16/bf16/f64
//! under the `scan_cumsum_*` family.
//!
//! Fuel's `OpParams::CumSum { outer_count, dim_size, inner_count }`
//! carries a single-axis scan via the flat-3-axis view; this
//! integration presents the rank-3 `[outer, dim, inner]` shape with
//! `scan_axis = 1` (middle axis) and `reverse = 0`. Fuel's backward
//! is expressed as `Flip → CumSum → Flip` upstream, so the kernel
//! itself never needs reverse-mode.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::{Error, Layout, Result};

use crate::byte_storage::CudaStorageBytes;

use super::status::check;

fn is_contiguous_zero_offset(layout: &Layout) -> bool {
    layout.start_offset() == 0 && layout.is_contiguous()
}

/// FFI signature shared by every CumSum dtype symbol.
type CumSumRun = unsafe extern "C" fn(
    numel: i64,
    rank: i32,
    shape: *const i32,
    stride_x: *const i64,
    stride_y: *const i64,
    scan_axis: i32,
    scan_extent: i32,
    scan_stride_x: i64,
    reverse: i32,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

#[allow(clippy::too_many_arguments)]
fn cumsum_run(
    input: &CudaStorageBytes,
    src_layout: Option<&Layout>,
    axis: usize,
    outer_count: usize,
    dim_size: usize,
    inner_count: usize,
    kernel: CumSumRun,
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
    let out = device.alloc_zeros::<u8>(out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let in_ptr = input.buffer().as_raw().0 as *const std::ffi::c_void;
    let out_ptr = out.as_raw().0 as *mut std::ffi::c_void;

    let take_strided = src_layout
        .map(|l| !is_contiguous_zero_offset(l))
        .unwrap_or(false);

    // SAFETY: pointers valid; arrays live for the FFI call.
    // workspace null+0: scan_cumsum's workspace is 0 for the shapes
    // fuel produces (the kernel's internal block-scan suffices for
    // common axis sizes; very large axes that would need workspace
    // are reduced upstream — Phase 7.6's reduction chassis splits
    // them before they reach this op).
    let status = if take_strided {
        let layout = src_layout.expect("guarded by take_strided");
        let dims = layout.shape().dims();
        let rank = dims.len();
        if axis >= rank {
            return Err(Error::Msg(format!(
                "{op_label}: axis {axis} out of range for rank {rank}",
            )).bt());
        }
        let mut shape_i32: Vec<i32> = Vec::with_capacity(rank);
        for (i, &d) in dims.iter().enumerate() {
            shape_i32.push(i32_or(i, d)?);
        }
        let stride_x_i64: Vec<i64> = layout.stride().iter().map(|&s| s as i64).collect();
        let stride_y_i64: Vec<i64> = {
            let mut s = vec![1_i64; rank];
            for d in (0..rank.saturating_sub(1)).rev() {
                s[d] = s[d + 1] * dims[d + 1] as i64;
            }
            s
        };
        let scan_axis_i32 = axis as i32;
        let scan_extent = shape_i32[axis];
        let scan_stride_x = stride_x_i64[axis];
        unsafe {
            kernel(
                numel as i64,
                rank as i32,
                shape_i32.as_ptr(),
                stride_x_i64.as_ptr(),
                stride_y_i64.as_ptr(),
                scan_axis_i32,
                scan_extent,
                scan_stride_x,
                0,
                in_ptr,
                out_ptr,
                std::ptr::null_mut(),
                0,
                stream,
            )
        }
    } else {
        let shape_i32: [i32; 3] = [
            i32_or(0, outer_count)?,
            i32_or(1, dim_size)?,
            i32_or(2, inner_count)?,
        ];
        let stride_x_i64: [i64; 3] = [
            (dim_size * inner_count) as i64,
            inner_count as i64,
            1,
        ];
        let stride_y_i64 = stride_x_i64;
        let scan_axis: i32 = 1;
        let scan_extent: i32 = shape_i32[1];
        let scan_stride_x: i64 = stride_x_i64[1];
        unsafe {
            kernel(
                numel as i64,
                3,
                shape_i32.as_ptr(),
                stride_x_i64.as_ptr(),
                stride_y_i64.as_ptr(),
                scan_axis,
                scan_extent,
                scan_stride_x,
                0,
                in_ptr,
                out_ptr,
                std::ptr::null_mut(),
                0,
                stream,
            )
        }
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out), device, out_bytes))
}

macro_rules! cumsum_kernel {
    ($name:ident, $sys_fn:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        #[doc = concat!("Baracuda `", $op_label, "` — single-axis cumulative sum (contig + strided dispatch).")]
        pub fn $name(
            input: &CudaStorageBytes,
            src_layout: Option<&Layout>,
            axis: usize,
            outer_count: usize,
            dim_size: usize,
            inner_count: usize,
        ) -> Result<CudaStorageBytes> {
            cumsum_run(
                input, src_layout, axis,
                outer_count, dim_size, inner_count,
                sys::$sys_fn,
                $op_label,
                $dtype_size,
            )
        }
    };
}

cumsum_kernel!(cumsum_f32,  baracuda_kernels_scan_cumsum_f32_run,  4, "cumsum_f32");
cumsum_kernel!(cumsum_f64,  baracuda_kernels_scan_cumsum_f64_run,  8, "cumsum_f64");
cumsum_kernel!(cumsum_f16,  baracuda_kernels_scan_cumsum_f16_run,  2, "cumsum_f16");
cumsum_kernel!(cumsum_bf16, baracuda_kernels_scan_cumsum_bf16_run, 2, "cumsum_bf16");

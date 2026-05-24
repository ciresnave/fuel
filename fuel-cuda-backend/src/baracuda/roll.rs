//! Roll (cyclic shift along selected axes) kernels from
//! `baracuda-kernels-sys`. Per-dtype symbols cover f32/f16/bf16/f64.
//!
//! Fuel's `OpParams::Roll { outer_count, dim_size, inner_count,
//! shift }` carries a single-axis shift via the flat-3-axis view;
//! this integration presents the rank-3 `[outer, dim, inner]`
//! shape with `shifts = [0, shift_i32, 0]`.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::{Error, Layout, Result};

use crate::byte_storage::CudaStorageBytes;

use super::status::check;

fn is_contiguous_zero_offset(layout: &Layout) -> bool {
    layout.start_offset() == 0 && layout.is_contiguous()
}

/// FFI signature shared by every Roll dtype symbol.
type RollRun = unsafe extern "C" fn(
    numel: i64,
    rank: i32,
    shape: *const i32,
    shifts: *const i32,
    stride_x: *const i64,
    stride_y: *const i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

#[allow(clippy::too_many_arguments)]
fn roll_run(
    input: &CudaStorageBytes,
    src_layout: Option<&Layout>,
    axis: usize,
    outer_count: usize,
    dim_size: usize,
    inner_count: usize,
    shift: i64,
    kernel: RollRun,
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
    // baracuda handles modular wrap internally; we just hand the raw
    // shift value through. Saturate-on-overflow via `as i32` is the
    // correct semantic here since shifts beyond i32::MAX would already
    // be modular-equivalent to shifts in range.
    let shift_i32 = i32::try_from(shift).unwrap_or_else(|_| {
        let m = i64::from(i32_or(1, dim_size).unwrap_or(1));
        (shift.rem_euclid(m.max(1))) as i32
    });
    let out = device.alloc_zeros::<u8>(out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let in_ptr = input.buffer().as_raw().0 as *const std::ffi::c_void;
    let out_ptr = out.as_raw().0 as *mut std::ffi::c_void;

    let take_strided = src_layout
        .map(|l| !is_contiguous_zero_offset(l))
        .unwrap_or(false);

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
        let mut shifts_i32: Vec<i32> = vec![0; rank];
        shifts_i32[axis] = shift_i32;
        // SAFETY: buffers owned through the call.
        unsafe {
            kernel(
                numel as i64,
                rank as i32,
                shape_i32.as_ptr(),
                shifts_i32.as_ptr(),
                stride_x_i64.as_ptr(),
                stride_y_i64.as_ptr(),
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
        let shifts_i32: [i32; 3] = [0, shift_i32, 0];
        let stride_x_i64: [i64; 3] = [
            (dim_size * inner_count) as i64,
            inner_count as i64,
            1,
        ];
        let stride_y_i64 = stride_x_i64;
        // SAFETY: pointers valid; stack-arrays live for the FFI call.
        unsafe {
            kernel(
                numel as i64,
                3,
                shape_i32.as_ptr(),
                shifts_i32.as_ptr(),
                stride_x_i64.as_ptr(),
                stride_y_i64.as_ptr(),
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

macro_rules! roll_kernel {
    ($name:ident, $sys_fn:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        #[doc = concat!("Baracuda `", $op_label, "` — single-axis cyclic shift (contig + strided dispatch).")]
        pub fn $name(
            input: &CudaStorageBytes,
            src_layout: Option<&Layout>,
            axis: usize,
            outer_count: usize,
            dim_size: usize,
            inner_count: usize,
            shift: i64,
        ) -> Result<CudaStorageBytes> {
            roll_run(
                input, src_layout, axis,
                outer_count, dim_size, inner_count, shift,
                sys::$sys_fn,
                $op_label,
                $dtype_size,
            )
        }
    };
}

roll_kernel!(roll_f32,  baracuda_kernels_roll_f32_run,  4, "roll_f32");
roll_kernel!(roll_f64,  baracuda_kernels_roll_f64_run,  8, "roll_f64");
roll_kernel!(roll_f16,  baracuda_kernels_roll_f16_run,  2, "roll_f16");
roll_kernel!(roll_bf16, baracuda_kernels_roll_bf16_run, 2, "roll_bf16");

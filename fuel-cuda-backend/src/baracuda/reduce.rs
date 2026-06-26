//! Single-axis reduction kernels from `baracuda-kernels-sys`.
//!
//! Each baracuda reduce kernel signature is:
//! ```text
//! fn run(
//!     output_numel: i64,
//!     rank: i32,
//!     output_shape: *const i32,
//!     stride_x: *const i64,
//!     stride_y: *const i64,
//!     reduce_axis: i32,
//!     reduce_extent: i32,
//!     reduce_stride_x: i64,
//!     x: *const c_void,
//!     y: *mut c_void,
//!     workspace, workspace_bytes, stream,
//! ) -> i32
//! ```
//!
//! For Fuel's `OpParams::Reduce { dims, keepdim }` which reduces
//! multiple axes at once, this module's higher-level wrapper loops:
//! one baracuda call per axis (in descending order so axis indices
//! stay stable). Single-axis reductions hit the kernel directly.
//!
//! ## Coverage today
//!
//! Kinds: Sum, Max, Min, Mean. (Prod / Std / Var / LogSumExp / Any
//! / All / CountNonzero ship in baracuda but don't have matching
//! Fuel `OpKind`s yet — they land when Fuel's primitive ops grow.)

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_ir::{DType, Layout, Result, Shape};

use crate::byte_storage::CudaStorageBytes;
use crate::error::CudaError;

use super::scratch::Workspace;
use super::status::check;

type ReduceRun = unsafe extern "C" fn(
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

/// Multi-axis reduce by repeated single-axis reduction.
///
/// Reduces `dims` in descending order so each call's axis index
/// remains valid as the rank shrinks. When `dims` is empty this is a
/// straight copy of `src` into a fresh contiguous buffer (matches
/// PyTorch / NumPy's "reduce over no axes is identity" semantics).
///
/// `keepdim`: when true, each reduced axis stays in the output as
/// size 1; when false, reduced axes are squeezed out.
fn reduce_multi_axis(
    src: &CudaStorageBytes,
    src_layout: &Layout,
    dims: &[usize],
    keepdim: bool,
    kernel: ReduceRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    if dims.is_empty() {
        // Identity copy. Allocate output, dtod-copy src into it.
        let device = src.device().clone();
        let bytes = src.len_bytes();
        if bytes == 0 {
            return CudaStorageBytes::alloc(&device, 0);
        }
        let out_buf = device.alloc_zeros::<u8>(bytes)?;
        // Use the device's existing dtod helper via CudaStorageBytes
        // round-trip: copy bytes by reading then writing. Cheaper
        // path lands when we add a memcpy_dtod direct helper.
        let host = src.to_cpu_bytes()?;
        let out = CudaStorageBytes::from_cpu_bytes(&device, &host)?;
        let _ = out_buf;
        return Ok(out);
    }

    let mut dims_sorted: Vec<usize> = dims.to_vec();
    dims_sorted.sort_unstable();
    dims_sorted.dedup();

    let mut current_shape: Vec<usize> = src_layout.shape().dims().to_vec();
    let mut current_storage = src.clone();
    let mut current_layout = src_layout.clone();

    // Reduce one axis at a time, from highest index down so the
    // remaining axis indices stay stable.
    for &axis in dims_sorted.iter().rev() {
        if axis >= current_shape.len() {
            return Err(fuel_ir::Error::Msg(format!(
                "{op_label}: reduce axis {axis} out of bounds for shape {current_shape:?}",
            ))
            .bt());
        }
        let reduce_extent = current_shape[axis];
        let reduce_stride = current_layout.stride()[axis];

        // Output shape: same as current with `axis` set to 1.
        let mut out_shape = current_shape.clone();
        out_shape[axis] = 1;
        let out_layout = Layout::contiguous(Shape::from_dims(&out_shape));
        let out_numel: i64 = out_shape.iter().product::<usize>() as i64;
        let out_bytes = (out_numel as usize) * dtype_size_bytes;

        let device = current_storage.device().clone();
        let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
        let scratch = Workspace::alloc(&device, 0)?;
        let stream = device.stream().as_raw() as *mut std::ffi::c_void;

        // Build the shape + stride buffers in baracuda's i32/i64 types.
        let mut shape_i32: Vec<i32> = Vec::with_capacity(out_shape.len());
        for (i, &d) in out_shape.iter().enumerate() {
            shape_i32.push(i32::try_from(d).map_err(|_| {
                fuel_ir::Error::cuda(CudaError::BaracudaShapeOverflow {
                    op: op_label,
                    dim_index: i,
                    dim_value: d,
                })
            })?);
        }
        let stride_x: Vec<i64> = current_layout.stride().iter().map(|&s| s as i64).collect();
        let stride_y: Vec<i64> = out_layout.stride().iter().map(|&s| s as i64).collect();
        let rank = shape_i32.len() as i32;

        let x_ptr = current_storage.buffer().as_raw().0 as *const std::ffi::c_void;
        let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

        // SAFETY: shape/stride buffers owned for the duration of the
        // call; pointers valid for the launch; stream borrows from
        // device which outlives the call.
        let status = unsafe {
            kernel(
                out_numel,
                rank,
                shape_i32.as_ptr(),
                stride_x.as_ptr(),
                stride_y.as_ptr(),
                axis as i32,
                reduce_extent as i32,
                reduce_stride as i64,
                x_ptr,
                y_ptr,
                scratch.as_raw(),
                scratch.bytes(),
                stream,
            )
        };
        check(status, op_label)?;
        device.synchronize()?;

        // Output of this round becomes input to the next.
        let bytes = out_bytes;
        current_storage = CudaStorageBytes::from_parts(Arc::new(out_buf), device, bytes);
        current_shape = out_shape;
        current_layout = out_layout;
    }

    // If !keepdim, squeeze out the size-1 reduced axes from the
    // final shape. The byte storage itself doesn't carry shape — the
    // executor reads the shape from the graph node — so this is
    // informational only at this layer.
    let _ = keepdim;

    Ok(current_storage)
}

/// Manifest macro for one (kind, dtype) reduce entry.
macro_rules! reduce_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda reduce `", $op_label, "` kernel.")]
            pub fn $name(
                src: &CudaStorageBytes,
                src_layout: &Layout,
                dims: &[usize],
                keepdim: bool,
            ) -> Result<CudaStorageBytes> {
                reduce_multi_axis(
                    src,
                    src_layout,
                    dims,
                    keepdim,
                    sys::[<baracuda_kernels_reduce_ $sys_stem _run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

// F32
reduce_kernel!(reduce_sum_f32, sum_f32, 4, "reduce_sum_f32");
reduce_kernel!(reduce_max_f32, max_f32, 4, "reduce_max_f32");
reduce_kernel!(reduce_min_f32, min_f32, 4, "reduce_min_f32");
reduce_kernel!(reduce_mean_f32, mean_f32, 4, "reduce_mean_f32");

// F16 / BF16 / F64
reduce_kernel!(reduce_sum_f16, sum_f16, 2, "reduce_sum_f16");
reduce_kernel!(reduce_max_f16, max_f16, 2, "reduce_max_f16");
reduce_kernel!(reduce_min_f16, min_f16, 2, "reduce_min_f16");
reduce_kernel!(reduce_mean_f16, mean_f16, 2, "reduce_mean_f16");

reduce_kernel!(reduce_sum_bf16, sum_bf16, 2, "reduce_sum_bf16");
reduce_kernel!(reduce_max_bf16, max_bf16, 2, "reduce_max_bf16");
reduce_kernel!(reduce_min_bf16, min_bf16, 2, "reduce_min_bf16");
reduce_kernel!(reduce_mean_bf16, mean_bf16, 2, "reduce_mean_bf16");

reduce_kernel!(reduce_sum_f64, sum_f64, 8, "reduce_sum_f64");
reduce_kernel!(reduce_max_f64, max_f64, 8, "reduce_max_f64");
reduce_kernel!(reduce_min_f64, min_f64, 8, "reduce_min_f64");
reduce_kernel!(reduce_mean_f64, mean_f64, 8, "reduce_mean_f64");

/// Byte-size lookup for reduce dtypes.
pub fn dtype_byte_size(dt: DType) -> usize {
    match dt {
        DType::F32 => 4,
        DType::F64 => 8,
        DType::F16 | DType::BF16 => 2,
        _ => 0,
    }
}

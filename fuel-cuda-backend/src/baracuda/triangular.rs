//! Triu / Tril kernels from `baracuda-kernels-sys` — upper- and
//! lower-triangular masking with per-dtype symbols. baracuda's
//! Triu/Tril operate on tensors of rank >= 2 (`[..., rows, cols]`);
//! Fuel's `OpParams::Triangular` carries `batch_count, rows, cols,
//! diagonal` with leading dims folded. We present the input to
//! baracuda as a rank-3 `[batch_count, rows, cols]` tensor — the
//! kernel walks any rank >= 2 the same way.
//!
//! ## Coverage (alpha.29)
//!
//! 7 dtypes per direction: f16, bf16, f32, f64, i32, i64, bool.
//! Fuel registrations route bool through the i32-or-u8 path when
//! Fuel adds Bool storage; today we cover the float + integer set.
//!
//! ## Backward
//!
//! Triu's backward is Triu with the same diagonal (mask is its own
//! adjoint). The same kernel handles both forward and backward —
//! no separate backward symbols exist or are needed.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::{Error, Result};

use crate::byte_storage::CudaStorageBytes;

use super::status::check;

/// FFI signature shared by every Triu/Tril dtype symbol.
type TriangularRun = unsafe extern "C" fn(
    input: *const std::ffi::c_void,
    output: *mut std::ffi::c_void,
    shape: *const i32,
    rank: i32,
    diagonal: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Run a triangular mask kernel. Validates batch/rows/cols fit in
/// i32 and dispatches to the chosen kernel pointer.
fn triangular_run(
    input: &CudaStorageBytes,
    batch_count: usize,
    rows: usize,
    cols: usize,
    diagonal: i64,
    kernel: TriangularRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let device = input.device().clone();
    let elem_count = batch_count * rows * cols;
    let out_bytes = elem_count * dtype_size_bytes;
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
        i32_or(0, batch_count)?,
        i32_or(1, rows)?,
        i32_or(2, cols)?,
    ];
    let diag_i32 = i32::try_from(diagonal).map_err(|_| {
        Error::Msg(format!(
            "{op_label}: diagonal {diagonal} does not fit in i32",
        ))
        .bt()
    })?;
    let out = device.alloc_zeros::<u8>(out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let in_ptr = input.buffer().as_raw().0 as *const std::ffi::c_void;
    let out_ptr = out.as_raw().0 as *mut std::ffi::c_void;
    // SAFETY: pointers valid for the bytes claimed; shape array lives
    // for the FFI call; stream borrowed from device.
    let status = unsafe {
        kernel(
            in_ptr,
            out_ptr,
            shape_i32.as_ptr(),
            3,
            diag_i32,
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out), device, out_bytes))
}

macro_rules! triangular_kernel {
    ($name:ident, $sys_fn:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        #[doc = concat!("Baracuda `", $op_label, "` — per-dtype triangular mask.")]
        pub fn $name(
            input: &CudaStorageBytes,
            batch_count: usize,
            rows: usize,
            cols: usize,
            diagonal: i64,
        ) -> Result<CudaStorageBytes> {
            triangular_run(
                input, batch_count, rows, cols, diagonal,
                sys::$sys_fn,
                $op_label,
                $dtype_size,
            )
        }
    };
}

// ----- Triu --------------------------------------------------------------

triangular_kernel!(triu_f32,  baracuda_kernels_triu_f32_run,  4, "triu_f32");
triangular_kernel!(triu_f64,  baracuda_kernels_triu_f64_run,  8, "triu_f64");
triangular_kernel!(triu_f16,  baracuda_kernels_triu_f16_run,  2, "triu_f16");
triangular_kernel!(triu_bf16, baracuda_kernels_triu_bf16_run, 2, "triu_bf16");
triangular_kernel!(triu_i32,  baracuda_kernels_triu_i32_run,  4, "triu_i32");
triangular_kernel!(triu_i64,  baracuda_kernels_triu_i64_run,  8, "triu_i64");
triangular_kernel!(triu_bool, baracuda_kernels_triu_bool_run, 1, "triu_bool");

// ----- Tril --------------------------------------------------------------

triangular_kernel!(tril_f32,  baracuda_kernels_tril_f32_run,  4, "tril_f32");
triangular_kernel!(tril_f64,  baracuda_kernels_tril_f64_run,  8, "tril_f64");
triangular_kernel!(tril_f16,  baracuda_kernels_tril_f16_run,  2, "tril_f16");
triangular_kernel!(tril_bf16, baracuda_kernels_tril_bf16_run, 2, "tril_bf16");
triangular_kernel!(tril_i32,  baracuda_kernels_tril_i32_run,  4, "tril_i32");
triangular_kernel!(tril_i64,  baracuda_kernels_tril_i64_run,  8, "tril_i64");
triangular_kernel!(tril_bool, baracuda_kernels_tril_bool_run, 1, "tril_bool");

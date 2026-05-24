//! Triu / Tril kernels from `baracuda-kernels-sys` — upper- and
//! lower-triangular masking with per-dtype symbols. baracuda's
//! Triu/Tril operate on tensors of rank >= 2 (`[..., rows, cols]`);
//! Fuel's `OpParams::Triangular` carries `batch_count, rows, cols,
//! diagonal` with leading dims folded.
//!
//! ## Contig vs strided dispatch
//!
//! Baracuda alpha.31 ships `<sym>_strided_run` siblings; the wrapper
//! picks per-call via `is_contiguous_zero_offset(layout)`. The contig
//! fast path uses the rank-3 `[batch, rows, cols]` reshape; the
//! strided path passes the input's true rank-N shape + per-input
//! strides + contig output strides over the same shape.
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
//! adjoint, per Phase 13.4). The same kernel handles both forward
//! and backward — no separate backward symbols exist or are needed.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::{Error, Layout, Result, Shape};

use crate::byte_storage::CudaStorageBytes;

use super::status::check;

/// FFI signature shared by every Triu/Tril dtype contig symbol.
type TriangularRun = unsafe extern "C" fn(
    input: *const std::ffi::c_void,
    output: *mut std::ffi::c_void,
    shape: *const i32,
    rank: i32,
    diagonal: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Strided FFI variant (alpha.31): adds stride_x + stride_y.
type TriangularStridedRun = unsafe extern "C" fn(
    input: *const std::ffi::c_void,
    output: *mut std::ffi::c_void,
    shape: *const i32,
    rank: i32,
    stride_x: *const i64,
    stride_y: *const i64,
    diagonal: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

fn is_contiguous_zero_offset(layout: &Layout) -> bool {
    layout.start_offset() == 0 && layout.is_contiguous()
}

/// Build rank-N shape + stride_x + contig stride_y for the strided FFI.
fn build_strided_args(
    layout: &Layout,
    op_label: &'static str,
) -> Result<(Vec<i32>, Vec<i64>, Vec<i64>)> {
    let dims = layout.shape().dims();
    let rank = dims.len();
    if rank < 2 {
        return Err(Error::Msg(format!(
            "{op_label}: rank {rank} < 2 (triangular requires at least [rows, cols])",
        )).bt());
    }
    let mut shape_i32: Vec<i32> = Vec::with_capacity(rank);
    for (i, &d) in dims.iter().enumerate() {
        shape_i32.push(i32::try_from(d).map_err(|_| {
            Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                op: op_label, dim_index: i, dim_value: d,
            })
        })?);
    }
    let stride_x: Vec<i64> = layout.stride().iter().map(|&s| s as i64).collect();
    let stride_y: Vec<i64> = {
        let mut s = vec![1_i64; rank];
        for d in (0..rank.saturating_sub(1)).rev() {
            s[d] = s[d + 1] * dims[d + 1] as i64;
        }
        s
    };
    Ok((shape_i32, stride_x, stride_y))
}

/// Run a triangular mask kernel with contig vs strided dispatch.
/// `(batch_count, rows, cols)` come from `OpParams::Triangular` and
/// define the contig fast-path shape; the strided path uses the
/// input's actual rank-N shape from `layout`.
fn triangular_run(
    input: &CudaStorageBytes,
    src_layout: Option<&Layout>,
    batch_count: usize,
    rows: usize,
    cols: usize,
    diagonal: i64,
    contig: TriangularRun,
    strided: TriangularStridedRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let device = input.device().clone();
    let elem_count = batch_count * rows * cols;
    let out_bytes = elem_count * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let diag_i32 = i32::try_from(diagonal).map_err(|_| {
        Error::Msg(format!("{op_label}: diagonal {diagonal} does not fit in i32")).bt()
    })?;
    let out = device.alloc_zeros::<u8>(out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let in_ptr = input.buffer().as_raw().0 as *const std::ffi::c_void;
    let out_ptr = out.as_raw().0 as *mut std::ffi::c_void;

    // Pick contig vs strided. With no layout we fall back to the
    // rank-3 contig path (matches the pre-alpha.31 behavior).
    let take_strided = src_layout
        .map(|l| !is_contiguous_zero_offset(l))
        .unwrap_or(false);

    let status = if take_strided {
        let layout = src_layout.expect("guarded by take_strided");
        let (shape_i32, stride_x, stride_y) = build_strided_args(layout, op_label)?;
        let rank = shape_i32.len() as i32;
        // SAFETY: shape/stride buffers owned through the call.
        unsafe {
            strided(
                in_ptr, out_ptr,
                shape_i32.as_ptr(), rank,
                stride_x.as_ptr(), stride_y.as_ptr(),
                diag_i32, stream,
            )
        }
    } else {
        let i32_or = |dim_index: usize, dim_value: usize| -> Result<i32> {
            i32::try_from(dim_value).map_err(|_| {
                Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                    op: op_label, dim_index, dim_value,
                })
            })
        };
        let shape_i32: [i32; 3] = [
            i32_or(0, batch_count)?,
            i32_or(1, rows)?,
            i32_or(2, cols)?,
        ];
        // SAFETY: shape array lives for the FFI call.
        unsafe {
            contig(
                in_ptr, out_ptr,
                shape_i32.as_ptr(), 3, diag_i32, stream,
            )
        }
    };
    let _ = Shape::from_dims(&[batch_count, rows, cols]); // suppress unused warnings if any
    check(status, op_label)?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out), device, out_bytes))
}

macro_rules! triangular_kernel {
    ($name:ident, $contig_sym:ident, $strided_sym:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        #[doc = concat!("Baracuda `", $op_label, "` — per-dtype triangular mask.")]
        pub fn $name(
            input: &CudaStorageBytes,
            src_layout: Option<&Layout>,
            batch_count: usize,
            rows: usize,
            cols: usize,
            diagonal: i64,
        ) -> Result<CudaStorageBytes> {
            triangular_run(
                input, src_layout, batch_count, rows, cols, diagonal,
                sys::$contig_sym,
                sys::$strided_sym,
                $op_label,
                $dtype_size,
            )
        }
    };
}

// ----- Triu --------------------------------------------------------------

triangular_kernel!(
    triu_f32,
    baracuda_kernels_triu_f32_run,
    baracuda_kernels_triu_f32_strided_run,
    4, "triu_f32"
);
triangular_kernel!(
    triu_f64,
    baracuda_kernels_triu_f64_run,
    baracuda_kernels_triu_f64_strided_run,
    8, "triu_f64"
);
triangular_kernel!(
    triu_f16,
    baracuda_kernels_triu_f16_run,
    baracuda_kernels_triu_f16_strided_run,
    2, "triu_f16"
);
triangular_kernel!(
    triu_bf16,
    baracuda_kernels_triu_bf16_run,
    baracuda_kernels_triu_bf16_strided_run,
    2, "triu_bf16"
);
triangular_kernel!(
    triu_i32,
    baracuda_kernels_triu_i32_run,
    baracuda_kernels_triu_i32_strided_run,
    4, "triu_i32"
);
triangular_kernel!(
    triu_i64,
    baracuda_kernels_triu_i64_run,
    baracuda_kernels_triu_i64_strided_run,
    8, "triu_i64"
);
triangular_kernel!(
    triu_bool,
    baracuda_kernels_triu_bool_run,
    baracuda_kernels_triu_bool_strided_run,
    1, "triu_bool"
);

// ----- Tril --------------------------------------------------------------

triangular_kernel!(
    tril_f32,
    baracuda_kernels_tril_f32_run,
    baracuda_kernels_tril_f32_strided_run,
    4, "tril_f32"
);
triangular_kernel!(
    tril_f64,
    baracuda_kernels_tril_f64_run,
    baracuda_kernels_tril_f64_strided_run,
    8, "tril_f64"
);
triangular_kernel!(
    tril_f16,
    baracuda_kernels_tril_f16_run,
    baracuda_kernels_tril_f16_strided_run,
    2, "tril_f16"
);
triangular_kernel!(
    tril_bf16,
    baracuda_kernels_tril_bf16_run,
    baracuda_kernels_tril_bf16_strided_run,
    2, "tril_bf16"
);
triangular_kernel!(
    tril_i32,
    baracuda_kernels_tril_i32_run,
    baracuda_kernels_tril_i32_strided_run,
    4, "tril_i32"
);
triangular_kernel!(
    tril_i64,
    baracuda_kernels_tril_i64_run,
    baracuda_kernels_tril_i64_strided_run,
    8, "tril_i64"
);
triangular_kernel!(
    tril_bool,
    baracuda_kernels_tril_bool_run,
    baracuda_kernels_tril_bool_strided_run,
    1, "tril_bool"
);

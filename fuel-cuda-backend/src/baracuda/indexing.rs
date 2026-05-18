//! Indexing kernels from `baracuda-kernels-sys` — `index_select` so
//! far. `gather` / `scatter_add` / `masked_fill` / `one_hot` /
//! `nonzero` follow this pattern and wire up incrementally.
//!
//! ## Index dtype
//!
//! Baracuda alpha.27 ships i32 (default) and i64 (`_i64idx_`)
//! index variants. Fuel currently passes U32 indices through the
//! binding table. Since U32 and i32 are bit-identical for the
//! non-negative index range Fuel constructs them with (and
//! baracuda's contract per OP-MATRIX is "out-of-bounds + negative
//! indices are silently skipped"), we reinterpret U32 → i32 at the
//! byte level — no value conversion needed.
//!
//! Fuel's I64 index path (when it grows one) will route to
//! baracuda's `_i64idx_` variants which now exist alpha.27 (the
//! Tier-2 #7 finding's resolution).

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::Result;

use crate::byte_storage::CudaStorageBytes;

use super::scratch::Workspace;
use super::status::check;

type IndexSelectRun = unsafe extern "C" fn(
    out_numel: i64,
    rank: i32,
    select_dim: i32,
    src_dim_size: i32,
    out_shape: *const i32,
    stride_src: *const i64,
    stride_out: *const i64,
    src: *const std::ffi::c_void,
    idx: *const std::ffi::c_void,
    out: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// IndexSelect driver. Flattens Fuel's
/// `(outer_count, source_dim_size, n_indices, inner_count)` shape
/// into a rank-3 `[outer_count, n_indices, inner_count]` output
/// shape with `select_dim = 1`.
///
/// `src_layout` is `[outer_count, source_dim_size, inner_count]`
/// in elements; the source dim size is the middle one.
fn index_select_run(
    src: &CudaStorageBytes,
    idx: &CudaStorageBytes,
    outer_count: usize,
    source_dim_size: usize,
    n_indices: usize,
    inner_count: usize,
    kernel: IndexSelectRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let device = src.device().clone();
    let out_numel = outer_count * n_indices * inner_count;
    let out_bytes = out_numel * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let i32_or = |dim_index: usize, dim_value: usize| -> Result<i32> {
        i32::try_from(dim_value).map_err(|_| {
            fuel_core_types::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                op: op_label,
                dim_index,
                dim_value,
            })
        })
    };

    let oc = i32_or(0, outer_count)?;
    let ni = i32_or(1, n_indices)?;
    let ic = i32_or(2, inner_count)?;
    let src_dim = i32_or(3, source_dim_size)?;

    let out_shape: [i32; 3] = [oc, ni, ic];
    let stride_src: [i64; 3] = [
        (source_dim_size * inner_count) as i64,
        inner_count as i64,
        1,
    ];
    let stride_out: [i64; 3] = [(n_indices * inner_count) as i64, inner_count as i64, 1];

    let src_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let idx_ptr = idx.buffer().as_raw().0 as *const std::ffi::c_void;
    let out_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    let status = unsafe {
        kernel(
            out_numel as i64,
            3,
            1,
            src_dim,
            out_shape.as_ptr(),
            stride_src.as_ptr(),
            stride_out.as_ptr(),
            src_ptr,
            idx_ptr,
            out_ptr,
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

macro_rules! index_select_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `", $op_label, "` kernel (U32 indices).")]
            #[allow(clippy::too_many_arguments)]
            pub fn $name(
                src: &CudaStorageBytes,
                idx: &CudaStorageBytes,
                outer_count: usize,
                source_dim_size: usize,
                n_indices: usize,
                inner_count: usize,
            ) -> Result<CudaStorageBytes> {
                index_select_run(
                    src,
                    idx,
                    outer_count,
                    source_dim_size,
                    n_indices,
                    inner_count,
                    sys::[<baracuda_kernels_index_select_ $sys_stem _run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

// I32-index (interpreted as U32 by Fuel's binding-table callers —
// bit-identical for the non-negative range Fuel constructs).
index_select_kernel!(index_select_f32, f32, 4, "index_select_f32");
index_select_kernel!(index_select_f64, f64, 8, "index_select_f64");
index_select_kernel!(index_select_i32, i32, 4, "index_select_i32");

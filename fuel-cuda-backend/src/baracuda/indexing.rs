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
use fuel_ir::Result;

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
#[allow(clippy::too_many_arguments)]
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
    let out = CudaStorageBytes::from_parts(Arc::new(out_buf), device, out_bytes);
    index_select_run_into(
        src,
        idx,
        outer_count,
        source_dim_size,
        n_indices,
        inner_count,
        &out,
        kernel,
        op_label,
        dtype_size_bytes,
    )?;
    Ok(out)
}

/// Write-into-output IndexSelect driver (CapturedRun executor build-out).
///
/// Identical gather-by-index math to [`index_select_run`], but writes into
/// the caller-provided `out` buffer instead of allocating one — the enabler
/// for the pipelined executor's persistent-output (capture) mode where a
/// fixed-address output is written in place so **no device allocation
/// happens** (mandatory inside a CUDA-graph capture scope). Byte-identical
/// result to the alloc-and-return path for a same-sized `out`.
///
/// `out` must already hold at least
/// `outer_count * n_indices * inner_count * dtype_size_bytes` bytes; a
/// smaller buffer is a surfaced error, never an out-of-bounds device write.
#[allow(clippy::too_many_arguments)]
fn index_select_run_into(
    src: &CudaStorageBytes,
    idx: &CudaStorageBytes,
    outer_count: usize,
    source_dim_size: usize,
    n_indices: usize,
    inner_count: usize,
    out: &CudaStorageBytes,
    kernel: IndexSelectRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<()> {
    let device = src.device().clone();
    let out_numel = outer_count * n_indices * inner_count;
    let out_bytes = out_numel * dtype_size_bytes;
    if out_bytes == 0 {
        return Ok(());
    }
    if out.len_bytes() < out_bytes {
        return Err(fuel_ir::Error::Msg(format!(
            "{op_label}: write-into output buffer too small ({} < {} bytes)",
            out.len_bytes(),
            out_bytes,
        ))
        .bt());
    }
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let i32_or = |dim_index: usize, dim_value: usize| -> Result<i32> {
        i32::try_from(dim_value).map_err(|_| {
            fuel_ir::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
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
    let out_ptr = out.buffer().as_raw().0 as *mut std::ffi::c_void;

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
    Ok(())
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

            #[doc = concat!(
                "Write-into-output variant of baracuda `", $op_label,
                "` — writes into `out` (no alloc; CapturedRun capture mode)."
            )]
            #[allow(clippy::too_many_arguments)]
            pub fn [<$name _into>](
                src: &CudaStorageBytes,
                idx: &CudaStorageBytes,
                outer_count: usize,
                source_dim_size: usize,
                n_indices: usize,
                inner_count: usize,
                out: &CudaStorageBytes,
            ) -> Result<()> {
                index_select_run_into(
                    src,
                    idx,
                    outer_count,
                    source_dim_size,
                    n_indices,
                    inner_count,
                    out,
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

// ===========================================================================
// Gather (N-dim, single dim, U32 indices)
// ===========================================================================

type GatherRun = unsafe extern "C" fn(
    out_numel: i64,
    rank: i32,
    gather_dim: i32,
    src_dim_size: i32,
    out_shape: *const i32,
    stride_src: *const i64,
    stride_index: *const i64,
    stride_out: *const i64,
    src: *const std::ffi::c_void,
    index: *const std::ffi::c_void,
    out: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Build i32 shape + i64 stride buffers from a `Vec<usize>` source
/// shape. Stride convention: row-major contiguous.
fn shape_strides_for(
    shape: &[usize],
    op_label: &'static str,
) -> Result<(Vec<i32>, Vec<i64>)> {
    let rank = shape.len();
    let mut shape_i32 = Vec::with_capacity(rank);
    for (i, &d) in shape.iter().enumerate() {
        shape_i32.push(i32::try_from(d).map_err(|_| {
            fuel_ir::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                op: op_label,
                dim_index: i,
                dim_value: d,
            })
        })?);
    }
    let mut stride = vec![0_i64; rank];
    if rank > 0 {
        stride[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            stride[i] = stride[i + 1] * shape[i + 1] as i64;
        }
    }
    Ok((shape_i32, stride))
}

fn gather_run(
    src: &CudaStorageBytes,
    index: &CudaStorageBytes,
    source_shape: &[usize],
    output_shape: &[usize],
    dim: usize,
    kernel: GatherRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    if source_shape.len() != output_shape.len() {
        return Err(fuel_ir::Error::Msg(format!(
            "{op_label}: source rank {} != output rank {}",
            source_shape.len(),
            output_shape.len(),
        ))
        .bt());
    }
    if dim >= source_shape.len() {
        return Err(fuel_ir::Error::Msg(format!(
            "{op_label}: dim {dim} out of bounds for rank {}",
            source_shape.len(),
        ))
        .bt());
    }
    let device = src.device().clone();
    let out_numel: i64 = output_shape.iter().product::<usize>() as i64;
    let out_bytes = (out_numel as usize) * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let (out_shape_i32, stride_out) = shape_strides_for(output_shape, op_label)?;
    let (_src_shape_i32, stride_src) = shape_strides_for(source_shape, op_label)?;
    // Index shape matches output shape for gather.
    let (_idx_shape_i32, stride_index) = shape_strides_for(output_shape, op_label)?;
    let rank = source_shape.len() as i32;
    let src_dim = i32::try_from(source_shape[dim]).map_err(|_| {
        fuel_ir::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label,
            dim_index: dim,
            dim_value: source_shape[dim],
        })
    })?;

    let src_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let idx_ptr = index.buffer().as_raw().0 as *const std::ffi::c_void;
    let out_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    let status = unsafe {
        kernel(
            out_numel,
            rank,
            dim as i32,
            src_dim,
            out_shape_i32.as_ptr(),
            stride_src.as_ptr(),
            stride_index.as_ptr(),
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
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out_buf),
        device,
        out_bytes,
    ))
}

macro_rules! gather_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `", $op_label, "` kernel (U32 indices).")]
            pub fn $name(
                src: &CudaStorageBytes,
                index: &CudaStorageBytes,
                source_shape: &[usize],
                output_shape: &[usize],
                dim: usize,
            ) -> Result<CudaStorageBytes> {
                gather_run(
                    src,
                    index,
                    source_shape,
                    output_shape,
                    dim,
                    sys::[<baracuda_kernels_gather_ $sys_stem _run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

gather_kernel!(gather_f32, f32, 4, "gather_f32");
gather_kernel!(gather_f64, f64, 8, "gather_f64");
gather_kernel!(gather_i32, i32, 4, "gather_i32");

// ===========================================================================
// MaskedFill
// ===========================================================================

type MaskedFillRun = unsafe extern "C" fn(
    numel: i64,
    src: *const std::ffi::c_void,
    mask: *const std::ffi::c_void,
    out: *mut std::ffi::c_void,
    fill_bits: i64,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Pack `fill_bytes` (Fuel's pre-encoded scalar in output dtype's
/// representation) into baracuda's `i64 fill_bits`. The kernel
/// reads only the low `sizeof(T)` bytes per element; higher bytes
/// are ignored.
fn pack_fill_bits(fill_bytes: &[u8], op_label: &'static str) -> Result<i64> {
    if fill_bytes.len() > 8 {
        return Err(fuel_ir::Error::Msg(format!(
            "{op_label}: fill_bytes length {} exceeds i64 (8 bytes)",
            fill_bytes.len(),
        ))
        .bt());
    }
    let mut buf = [0_u8; 8];
    buf[..fill_bytes.len()].copy_from_slice(fill_bytes);
    Ok(i64::from_le_bytes(buf))
}

fn masked_fill_run(
    src: &CudaStorageBytes,
    mask: &CudaStorageBytes,
    fill_bytes: &[u8],
    kernel: MaskedFillRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let device = src.device().clone();
    let numel = src.len_bytes() / dtype_size_bytes.max(1);
    let out_bytes = numel * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let fill_bits = pack_fill_bits(fill_bytes, op_label)?;
    let src_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let mask_ptr = mask.buffer().as_raw().0 as *const std::ffi::c_void;
    let out_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    let status = unsafe {
        kernel(
            numel as i64,
            src_ptr,
            mask_ptr,
            out_ptr,
            fill_bits,
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

macro_rules! masked_fill_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `", $op_label, "` kernel.")]
            pub fn $name(
                src: &CudaStorageBytes,
                mask: &CudaStorageBytes,
                fill_bytes: &[u8],
            ) -> Result<CudaStorageBytes> {
                masked_fill_run(
                    src,
                    mask,
                    fill_bytes,
                    sys::[<baracuda_kernels_masked_fill_ $sys_stem _run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

masked_fill_kernel!(masked_fill_f32, f32, 4, "masked_fill_f32");
masked_fill_kernel!(masked_fill_f64, f64, 8, "masked_fill_f64");
masked_fill_kernel!(masked_fill_i32, i32, 4, "masked_fill_i32");

// ===========================================================================
// ScatterAdd
// ===========================================================================

type ScatterAddRun = unsafe extern "C" fn(
    upd_numel: i64,
    rank: i32,
    scatter_dim: i32,
    out_dim_size: i32,
    upd_shape: *const i32,
    stride_upd: *const i64,
    stride_index: *const i64,
    stride_out: *const i64,
    updates: *const std::ffi::c_void,
    index: *const std::ffi::c_void,
    out: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// ScatterAdd: `out[..., index[i, ...], ...] += src[i, ...]`. The
/// caller pre-allocates `out` as a copy of `base` (Fuel's
/// `OpKind::ScatterAdd` semantics — accumulate into base).
fn scatter_add_run(
    base: &CudaStorageBytes,
    index: &CudaStorageBytes,
    updates: &CudaStorageBytes,
    base_shape: &[usize],
    src_shape: &[usize],
    dim: usize,
    kernel: ScatterAddRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    if base_shape.len() != src_shape.len() {
        return Err(fuel_ir::Error::Msg(format!(
            "{op_label}: base rank {} != src rank {}",
            base_shape.len(),
            src_shape.len(),
        ))
        .bt());
    }
    if dim >= base_shape.len() {
        return Err(fuel_ir::Error::Msg(format!(
            "{op_label}: dim {dim} out of bounds for rank {}",
            base_shape.len(),
        ))
        .bt());
    }
    let device = base.device().clone();
    // Output = base copy; ScatterAdd accumulates into the copy.
    let base_bytes = base.len_bytes();
    let host_base = base.to_cpu_bytes()?;
    let out = CudaStorageBytes::from_cpu_bytes(&device, &host_base)?;
    let _ = base_bytes;

    let upd_numel: i64 = src_shape.iter().product::<usize>() as i64;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let (upd_shape_i32, stride_upd) = shape_strides_for(src_shape, op_label)?;
    let (_idx_shape_i32, stride_index) = shape_strides_for(src_shape, op_label)?;
    let (_out_shape_i32, stride_out) = shape_strides_for(base_shape, op_label)?;
    let out_dim = i32::try_from(base_shape[dim]).map_err(|_| {
        fuel_ir::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label,
            dim_index: dim,
            dim_value: base_shape[dim],
        })
    })?;
    let rank = base_shape.len() as i32;

    let upd_ptr = updates.buffer().as_raw().0 as *const std::ffi::c_void;
    let idx_ptr = index.buffer().as_raw().0 as *const std::ffi::c_void;
    let out_ptr = out.buffer().as_raw().0 as *mut std::ffi::c_void;
    let _ = dtype_size_bytes;

    let status = unsafe {
        kernel(
            upd_numel,
            rank,
            dim as i32,
            out_dim,
            upd_shape_i32.as_ptr(),
            stride_upd.as_ptr(),
            stride_index.as_ptr(),
            stride_out.as_ptr(),
            upd_ptr,
            idx_ptr,
            out_ptr,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    Ok(out)
}

macro_rules! scatter_add_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `", $op_label, "` kernel (U32 indices).")]
            #[allow(clippy::too_many_arguments)]
            pub fn $name(
                base: &CudaStorageBytes,
                index: &CudaStorageBytes,
                updates: &CudaStorageBytes,
                base_shape: &[usize],
                src_shape: &[usize],
                dim: usize,
            ) -> Result<CudaStorageBytes> {
                scatter_add_run(
                    base,
                    index,
                    updates,
                    base_shape,
                    src_shape,
                    dim,
                    sys::[<baracuda_kernels_scatter_add_ $sys_stem _run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

scatter_add_kernel!(scatter_add_f32, f32, 4, "scatter_add_f32");
scatter_add_kernel!(scatter_add_f64, f64, 8, "scatter_add_f64");

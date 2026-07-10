//! WriteSlice kernels from `baracuda-kernels-sys` — in-place
//! rectangular slab assignment. Byte-width-dispatched (b1/b2/b4/b8/
//! b16) covering all aligned-element dtypes; the nibble variant
//! (S4/U4) is parked until Fuel grows sub-byte dtype support.
//!
//! ## Op shape
//!
//! `dest[start_0..end_0, ..., start_{R-1}..end_{R-1}] = source`
//!
//! Assigns (not accumulates). Both tensors are contiguous row-major,
//! zero-offset. Per-axis the source's shape equals the slab width
//! `end - start`; per-axis range `[start, end)` must be within the
//! destination's extent on that axis.
//!
//! ## Wiring to Fuel's WorkItemKind::WriteSlice
//!
//! The pipelined executor wires `inputs=[source]` + `outputs=[dest]`
//! where `outputs[0]`'s Arc IS the destination's Storage Arc (zero-
//! copy adoption from the pre-allocated KV-cache buffer). The kernel
//! mutates dest's bytes in place through the write lock; no fresh
//! buffer is allocated.
//!
//! ## Fast paths
//!
//! Baracuda's underlying kernel already detects two fast paths
//! internally at the Plan layer: whole-dest replacement and the
//! "contiguous chunk" case (all axes except the outermost are full-
//! width — the KV-cache append shape). Fuel calls the generic FFI
//! and lets the kernel pick.

use baracuda_kernels_sys as sys;
use fuel_ir::{Error, Result};

use crate::byte_storage::CudaStorageBytes;

use super::status::check;

/// FFI signature shared by every byte-width WriteSlice symbol.
type WriteSliceRun = unsafe extern "C" fn(
    dest: *mut std::ffi::c_void,
    source: *const std::ffi::c_void,
    source_numel: i64,
    rank: i32,
    dest_shape: *const i32,
    source_shape: *const i32,
    range_start: *const i32,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Run a single in-place WriteSlice. `dest` is mutated in place;
/// the function returns `Ok(())` on success. No new buffer is
/// allocated — the caller's `dest` is the same buffer the
/// destination NodeId's Storage Arc points at.
///
/// `dest_shape`, `source_shape`, and `range_start` are host-side
/// arrays; this function converts them to the `i32` arrays the FFI
/// expects and validates each fits.
pub fn write_slice_run(
    dest: &mut CudaStorageBytes,
    source: &CudaStorageBytes,
    dest_shape: &[usize],
    source_shape: &[usize],
    range_start: &[usize],
    kernel: WriteSliceRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<()> {
    let rank = dest_shape.len();
    if source_shape.len() != rank || range_start.len() != rank {
        return Err(Error::Msg(format!(
            "{op_label}: dest_shape rank {} != source_shape rank {} or range_start rank {}",
            rank, source_shape.len(), range_start.len(),
        ))
        .bt());
    }
    if rank == 0 || rank > 8 {
        return Err(Error::Msg(format!(
            "{op_label}: rank {rank} out of range (baracuda WriteSlice supports 1..=8)",
        ))
        .bt());
    }
    let source_numel: usize = source_shape.iter().copied().product();
    if source_numel == 0 {
        return Ok(()); // empty slab — nothing to do.
    }
    // Validate dest bytes >= dest shape * dtype_size_bytes; same for
    // source.
    let dest_numel: usize = dest_shape.iter().copied().product();
    let dest_bytes_needed = dest_numel * dtype_size_bytes;
    if dest.len_bytes() < dest_bytes_needed {
        return Err(Error::Msg(format!(
            "{op_label}: dest buffer {} bytes < required {dest_bytes_needed} bytes \
             (dest_shape {:?} * dtype_size {dtype_size_bytes})",
            dest.len_bytes(), dest_shape,
        ))
        .bt());
    }
    let source_bytes_needed = source_numel * dtype_size_bytes;
    if source.len_bytes() < source_bytes_needed {
        return Err(Error::Msg(format!(
            "{op_label}: source buffer {} bytes < required {source_bytes_needed} bytes \
             (source_shape {:?} * dtype_size {dtype_size_bytes})",
            source.len_bytes(), source_shape,
        ))
        .bt());
    }
    // Per-axis range bounds.
    for (i, (&dim, &start)) in dest_shape.iter().zip(range_start.iter()).enumerate() {
        let end = start + source_shape[i];
        if end > dim {
            return Err(Error::Msg(format!(
                "{op_label}: ranges[{i}] = ({start}, {end}) past dest dim {i} = {dim}",
            ))
            .bt());
        }
    }

    // Convert to the i32 / i64 arrays the FFI expects. Rank is
    // bounded to 8 above, so stack-fixed arrays of length 8 are
    // safe to use.
    let mut dest_shape_i32 = [0_i32; 8];
    let mut source_shape_i32 = [0_i32; 8];
    let mut range_start_i32 = [0_i32; 8];
    let i32_or = |dim_index: usize, dim_value: usize| -> Result<i32> {
        i32::try_from(dim_value).map_err(|_| {
            Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                op: op_label,
                dim_index,
                dim_value,
            })
        })
    };
    for i in 0..rank {
        dest_shape_i32[i] = i32_or(i, dest_shape[i])?;
        source_shape_i32[i] = i32_or(i, source_shape[i])?;
        range_start_i32[i] = i32_or(i, range_start[i])?;
    }

    let device = dest.device().clone();
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let dest_ptr = dest.buffer().as_raw().0 as *mut std::ffi::c_void;
    let source_ptr = source.buffer().as_raw().0 as *const std::ffi::c_void;

    // Workspace is always 0 bytes for WriteSlice (per the safe Plan
    // wrapper's workspace_size() == 0). Pass null + 0.
    //
    // SAFETY: dest + source point at device buffers validated for
    // byte size above; the shape arrays live in `*_i32` stack vars
    // for the duration of the FFI call; `stream` is borrowed from
    // the device. The kernel only reads the rank-prefix of each
    // array (the trailing zeros are unused).
    let status = unsafe {
        kernel(
            dest_ptr,
            source_ptr,
            source_numel as i64,
            rank as i32,
            dest_shape_i32.as_ptr(),
            source_shape_i32.as_ptr(),
            range_start_i32.as_ptr(),
            std::ptr::null_mut(),
            0,
            stream,
        )
    };
    check(status, op_label)?;
    Ok(())
}

/// Generate one byte-width-dispatched WriteSlice entry point.
macro_rules! write_slice_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!(
                "Baracuda `", $op_label, "` — in-place rectangular slab assign for ",
                stringify!($dtype_size), "-byte elements.",
            )]
            pub fn $name(
                dest: &mut CudaStorageBytes,
                source: &CudaStorageBytes,
                dest_shape: &[usize],
                source_shape: &[usize],
                range_start: &[usize],
            ) -> Result<()> {
                write_slice_run(
                    dest, source,
                    dest_shape, source_shape, range_start,
                    sys::[<baracuda_kernels_write_slice_ $sys_stem _run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

write_slice_kernel!(write_slice_b1,  b1,  1,  "write_slice_b1");
write_slice_kernel!(write_slice_b2,  b2,  2,  "write_slice_b2");
write_slice_kernel!(write_slice_b4,  b4,  4,  "write_slice_b4");
write_slice_kernel!(write_slice_b8,  b8,  8,  "write_slice_b8");
write_slice_kernel!(write_slice_b16, b16, 16, "write_slice_b16");

// ===========================================================================
// WriteSlice `_doff` — device-resident dynamic-axis start (form-B)
// ===========================================================================
//
// Like `write_slice_run` except the start on ONE axis (`dyn_axis`) is read
// from a DEVICE pointer (`dyn_start_dev`, a single `i64`) at kernel entry
// instead of being host-baked into `range_start`. This is what makes a
// captured CUDA-graph node replay at the host-updated sequence position
// (KV-cache decode): the host bumps `*dyn_start_dev` per token via a
// fixed-address H2D memcpy, capture-tolerant, where the host-baked `_run`
// would freeze at the captured position.
//
// NO D2H of the offset here — that would defeat the whole point (a host
// round-trip is capture-illegal + bakes a fixed value). The offset buffer's
// device pointer is threaded straight to the kernel.
//
// Bound caveat (per baracuda): the true start (`cached_len`) is device-only,
// so the kernel deliberately does NOT clamp — guaranteeing the in-bounds
// write is the CALLER's contract (DecodeSession sizes `max_seq`). b1/b2/b4/b8
// only (KV decode dtype set); no b16.

/// FFI signature shared by every byte-width WriteSlice `_doff` symbol.
type WriteSliceDoffRun = unsafe extern "C" fn(
    dest: *mut std::ffi::c_void,
    source: *const std::ffi::c_void,
    source_numel: i64,
    rank: i32,
    dest_shape: *const i32,
    source_shape: *const i32,
    range_start: *const i32,
    dyn_axis: i32,
    dyn_start_dev: *const i64,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Run a single in-place WriteSlice with a device-resident start on
/// `dyn_axis`. `dest` is mutated in place. `dyn_start_dev` is the raw
/// device pointer of a live `i64` (the offset buffer's device address).
/// `range_start[dyn_axis]` is a placeholder (the kernel reads the true
/// start from the device); the other axes' starts are host-baked as in
/// `write_slice_run`.
///
/// Validates rank / shapes / byte-sizes host-side; the `dyn_axis` start
/// is NOT bounds-checked here (it is device-only — the caller's
/// contract, matching the kernel's no-clamp behavior).
#[allow(clippy::too_many_arguments)]
pub fn write_slice_doff_run(
    dest: &mut CudaStorageBytes,
    source: &CudaStorageBytes,
    dest_shape: &[usize],
    source_shape: &[usize],
    range_start: &[usize],
    dyn_axis: usize,
    dyn_start_dev: *const i64,
    kernel: WriteSliceDoffRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<()> {
    let rank = dest_shape.len();
    if source_shape.len() != rank || range_start.len() != rank {
        return Err(Error::Msg(format!(
            "{op_label}: dest_shape rank {} != source_shape rank {} or range_start rank {}",
            rank, source_shape.len(), range_start.len(),
        ))
        .bt());
    }
    if rank == 0 || rank > 8 {
        return Err(Error::Msg(format!(
            "{op_label}: rank {rank} out of range (baracuda WriteSlice supports 1..=8)",
        ))
        .bt());
    }
    if dyn_axis >= rank {
        return Err(Error::Msg(format!(
            "{op_label}: dyn_axis {dyn_axis} out of bounds for rank {rank}",
        ))
        .bt());
    }
    if dyn_start_dev.is_null() {
        return Err(Error::Msg(format!(
            "{op_label}: dyn_start_dev is null (need a live device i64)",
        ))
        .bt());
    }
    let source_numel: usize = source_shape.iter().copied().product();
    if source_numel == 0 {
        return Ok(()); // empty slab — nothing to do.
    }
    let dest_numel: usize = dest_shape.iter().copied().product();
    let dest_bytes_needed = dest_numel * dtype_size_bytes;
    if dest.len_bytes() < dest_bytes_needed {
        return Err(Error::Msg(format!(
            "{op_label}: dest buffer {} bytes < required {dest_bytes_needed} bytes \
             (dest_shape {:?} * dtype_size {dtype_size_bytes})",
            dest.len_bytes(), dest_shape,
        ))
        .bt());
    }
    let source_bytes_needed = source_numel * dtype_size_bytes;
    if source.len_bytes() < source_bytes_needed {
        return Err(Error::Msg(format!(
            "{op_label}: source buffer {} bytes < required {source_bytes_needed} bytes \
             (source_shape {:?} * dtype_size {dtype_size_bytes})",
            source.len_bytes(), source_shape,
        ))
        .bt());
    }
    // Per-axis range bounds — SKIP the dynamic axis (its start is
    // device-only). Off-axis: end = start + width must fit.
    for (i, (&dim, &start)) in dest_shape.iter().zip(range_start.iter()).enumerate() {
        if i == dyn_axis {
            continue;
        }
        let end = start + source_shape[i];
        if end > dim {
            return Err(Error::Msg(format!(
                "{op_label}: ranges[{i}] = ({start}, {end}) past dest dim {i} = {dim}",
            ))
            .bt());
        }
    }

    let mut dest_shape_i32 = [0_i32; 8];
    let mut source_shape_i32 = [0_i32; 8];
    let mut range_start_i32 = [0_i32; 8];
    let i32_or = |dim_index: usize, dim_value: usize| -> Result<i32> {
        i32::try_from(dim_value).map_err(|_| {
            Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                op: op_label,
                dim_index,
                dim_value,
            })
        })
    };
    for i in 0..rank {
        dest_shape_i32[i] = i32_or(i, dest_shape[i])?;
        source_shape_i32[i] = i32_or(i, source_shape[i])?;
        // The dyn_axis slot is a placeholder (kernel ignores it); host
        // it as 0 to keep the value in-range.
        range_start_i32[i] = if i == dyn_axis { 0 } else { i32_or(i, range_start[i])? };
    }

    let device = dest.device().clone();
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let dest_ptr = dest.buffer().as_raw().0 as *mut std::ffi::c_void;
    let source_ptr = source.buffer().as_raw().0 as *const std::ffi::c_void;

    // SAFETY: dest + source point at device buffers validated for byte
    // size above; `dyn_start_dev` is a live device i64 (non-null checked);
    // the shape arrays live in `*_i32` stack vars for the FFI call;
    // `stream` is borrowed from the device. The kernel reads the
    // rank-prefix of each array + the device start for `dyn_axis`.
    let status = unsafe {
        kernel(
            dest_ptr,
            source_ptr,
            source_numel as i64,
            rank as i32,
            dest_shape_i32.as_ptr(),
            source_shape_i32.as_ptr(),
            range_start_i32.as_ptr(),
            dyn_axis as i32,
            dyn_start_dev,
            std::ptr::null_mut(),
            0,
            stream,
        )
    };
    check(status, op_label)?;
    Ok(())
}

/// Generate one byte-width-dispatched WriteSlice `_doff` entry point.
macro_rules! write_slice_doff_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!(
                "Baracuda `", $op_label, "` — in-place slab assign with a device-resident ",
                "axis start for ", stringify!($dtype_size), "-byte elements.",
            )]
            #[allow(clippy::too_many_arguments)]
            pub fn $name(
                dest: &mut CudaStorageBytes,
                source: &CudaStorageBytes,
                dest_shape: &[usize],
                source_shape: &[usize],
                range_start: &[usize],
                dyn_axis: usize,
                dyn_start_dev: *const i64,
            ) -> Result<()> {
                write_slice_doff_run(
                    dest, source,
                    dest_shape, source_shape, range_start,
                    dyn_axis, dyn_start_dev,
                    sys::[<baracuda_kernels_write_slice_ $sys_stem _doff_run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

write_slice_doff_kernel!(write_slice_doff_b1, b1, 1, "write_slice_b1_doff");
write_slice_doff_kernel!(write_slice_doff_b2, b2, 2, "write_slice_b2_doff");
write_slice_doff_kernel!(write_slice_doff_b4, b4, 4, "write_slice_b4_doff");
write_slice_doff_kernel!(write_slice_doff_b8, b8, 8, "write_slice_b8_doff");

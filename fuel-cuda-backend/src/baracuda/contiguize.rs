//! Contiguize kernels from `baracuda-kernels-sys` — strided-source
//! to contiguous-dest copy. Byte-width-dispatched (b1/b2/b4/b8/b16)
//! covering all aligned-element dtypes; the nibble variant (S4/U4)
//! is parked until Fuel grows sub-byte dtype support.
//!
//! ## Op shape
//!
//! `dest[output_linear_index]` ← `source[start_offset +
//!   sum_d(unravel(linear_index, shape)[d] * source_strides[d])]`
//!
//! Source may have signed strides (Flip), zero strides (BroadcastTo),
//! and a non-zero element-offset. Dest is freshly allocated, row-
//! major contiguous, zero-offset.
//!
//! ## Wiring
//!
//! Contiguize is an executor-internal pass — invoked by
//! [`pipelined::auto_contiguize`] when the executor needs a
//! contiguous version of a strided input. Unlike WriteSlice it has
//! no IR-level op (`Op::Reshape`'s strided input handler is the only
//! consumer); no binding-table registration needed.
//!
//! Replaces the prior D2H → CPU contiguize_cpu → H2D fallback in
//! `pipelined.rs::auto_contiguize` (two device round-trips per non-
//! contig CUDA input).
//!
//! ## Fast paths
//!
//! Baracuda's underlying launchers detect three fast paths at host
//! side: already-contiguous-zero-offset → single
//! `cuMemcpyDtoDAsync`; innermost-stride-1 → per-outer-coord
//! contiguous run copy; generic → one thread per output element.
//! Fuel calls the byte-width dispatch wrapper and lets baracuda pick.

use std::sync::Arc;

use baracuda_driver::DeviceBuffer;
use baracuda_kernels_sys as sys;
use fuel_ir::{Error, Layout, Result};

use crate::byte_storage::CudaStorageBytes;

use super::status::check;

/// FFI signature shared by every byte-width Contiguize symbol.
type ContiguizeRun = unsafe extern "C" fn(
    dest: *mut std::ffi::c_void,
    source: *const std::ffi::c_void,
    shape: *const i32,
    source_strides: *const i64,
    source_offset: i64,
    rank: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Materialize a contiguous `CudaStorageBytes` from a strided source
/// + layout. Allocates a fresh device buffer of `elem_count *
/// dtype_size_bytes` and dispatches the appropriate byte-width
/// baracuda kernel.
///
/// Selects the byte-width kernel from `dtype_size_bytes`:
/// 1 → b1, 2 → b2, 4 → b4, 8 → b8, 16 → b16. Unsupported widths
/// surface a typed error.
pub fn contiguize_to_fresh(
    source: &CudaStorageBytes,
    layout: &Layout,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let kernel = match dtype_size_bytes {
        1 => sys::baracuda_kernels_contiguize_b1_run as ContiguizeRun,
        2 => sys::baracuda_kernels_contiguize_b2_run as ContiguizeRun,
        4 => sys::baracuda_kernels_contiguize_b4_run as ContiguizeRun,
        8 => sys::baracuda_kernels_contiguize_b8_run as ContiguizeRun,
        16 => sys::baracuda_kernels_contiguize_b16_run as ContiguizeRun,
        other => {
            return Err(Error::Msg(format!(
                "baracuda contiguize: unsupported byte width {other} (supported: \
                 1/2/4/8/16). Sub-byte dtypes (S4/U4) need a separate \
                 nibble-aware integration.",
            ))
            .bt());
        }
    };
    contiguize_with(source, layout, kernel, dtype_size_bytes)
}

/// Run a contiguize with a specific byte-width kernel pointer.
/// Allocates the dest buffer; returns a fresh `CudaStorageBytes`.
fn contiguize_with(
    source: &CudaStorageBytes,
    layout: &Layout,
    kernel: ContiguizeRun,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let device = source.device().clone();
    let shape_dims = layout.shape().dims();
    let rank = shape_dims.len();
    if rank == 0 {
        // Scalar contiguize = single element copy. Handle as a 1-D
        // contiguize of length 1; the kernel walks 1 output element.
        let dest_bytes = dtype_size_bytes;
        let dest = device.alloc_zeros::<u8>(dest_bytes)?;
        // Treat as rank-1 shape [1] with stride [0] (the lone element
        // is at start_offset). source_offset honored by the FFI.
        let stream = device.stream().as_raw() as *mut std::ffi::c_void;
        let dest_ptr = dest.as_raw().0 as *mut std::ffi::c_void;
        let source_ptr = source.buffer().as_raw().0 as *const std::ffi::c_void;
        let shape_i32: [i32; 1] = [1];
        let strides_i64: [i64; 1] = [0];
        // SAFETY: pointers + extents valid; offset is in elements.
        let status = unsafe {
            kernel(
                dest_ptr,
                source_ptr,
                shape_i32.as_ptr(),
                strides_i64.as_ptr(),
                layout.start_offset() as i64,
                1,
                stream,
            )
        };
        check(status, "contiguize_b*_scalar")?;
        device.synchronize()?;
        return Ok(CudaStorageBytes::from_parts(Arc::new(dest), device, dest_bytes));
    }
    if rank > 8 {
        return Err(Error::Msg(format!(
            "baracuda contiguize: rank {rank} > 8 not supported",
        ))
        .bt());
    }
    let elem_count: usize = shape_dims.iter().copied().product();
    let dest_bytes = elem_count * dtype_size_bytes;
    if dest_bytes == 0 {
        // Empty tensor — alloc a zero-byte buffer and return it.
        return CudaStorageBytes::alloc(&device, 0);
    }
    let dest = device.alloc_zeros::<u8>(dest_bytes)?;

    // Convert shape (usize → i32) + strides (isize → i64). Pre-size
    // arrays at rank 8 so the stack layout doesn't escape.
    let mut shape_i32 = [0_i32; 8];
    let mut strides_i64 = [0_i64; 8];
    let strides = layout.stride();
    if strides.len() != rank {
        return Err(Error::Msg(format!(
            "baracuda contiguize: layout stride rank {} != shape rank {rank}",
            strides.len(),
        ))
        .bt());
    }
    for i in 0..rank {
        shape_i32[i] = i32::try_from(shape_dims[i]).map_err(|_| {
            Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                op: "contiguize_b*",
                dim_index: i,
                dim_value: shape_dims[i],
            })
        })?;
        strides_i64[i] = strides[i] as i64;
    }

    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let dest_ptr = dest.as_raw().0 as *mut std::ffi::c_void;
    let source_ptr = source.buffer().as_raw().0 as *const std::ffi::c_void;

    // SAFETY: pointers + extents validated; shape/strides arrays
    // live for the FFI call's duration; stream is borrowed from
    // device. offset is in elements per the FFI contract.
    let status = unsafe {
        kernel(
            dest_ptr,
            source_ptr,
            shape_i32.as_ptr(),
            strides_i64.as_ptr(),
            layout.start_offset() as i64,
            rank as i32,
            stream,
        )
    };
    check(status, "contiguize_b*")?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(dest), device, dest_bytes))
}

// Re-export the unused DeviceBuffer ref so `cargo build` doesn't
// flag it. (Some sm-feature combinations strip the kernel-import use.)
#[allow(dead_code)]
fn _keep_imports_live(_: &DeviceBuffer<u8>) {}

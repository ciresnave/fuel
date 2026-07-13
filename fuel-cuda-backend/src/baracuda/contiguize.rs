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
    Ok(CudaStorageBytes::from_parts(Arc::new(dest), device, dest_bytes))
}

/// Write-into sibling of [`contiguize_to_fresh`] (CapturedRun executor
/// build-out, 4b-γ): materializes a contiguous copy of `source` at `layout`
/// INTO `dest`'s EXISTING device buffer instead of allocating a fresh one —
/// device allocation is illegal inside a CUDA-graph capture scope. Same
/// byte-width dispatch as `contiguize_to_fresh`; `dest`'s byte length must
/// already equal the expected `elem_count * dtype_size_bytes` (validated,
/// typed `Error::Msg` on mismatch — this should never trigger in practice,
/// since the persistent map only ever holds a buffer THIS SAME code path
/// sized on the warm pass, but per this project's "never panic on
/// production paths" rule we check rather than assume).
pub fn contiguize_into(
    source: &CudaStorageBytes,
    layout: &Layout,
    dest: &CudaStorageBytes,
    dtype_size_bytes: usize,
) -> Result<()> {
    let kernel = match dtype_size_bytes {
        1 => sys::baracuda_kernels_contiguize_b1_run as ContiguizeRun,
        2 => sys::baracuda_kernels_contiguize_b2_run as ContiguizeRun,
        4 => sys::baracuda_kernels_contiguize_b4_run as ContiguizeRun,
        8 => sys::baracuda_kernels_contiguize_b8_run as ContiguizeRun,
        16 => sys::baracuda_kernels_contiguize_b16_run as ContiguizeRun,
        other => {
            return Err(Error::Msg(format!(
                "baracuda contiguize_into: unsupported byte width {other} (supported: \
                 1/2/4/8/16). Sub-byte dtypes (S4/U4) need a separate \
                 nibble-aware integration.",
            ))
            .bt());
        }
    };
    contiguize_into_with(source, layout, dest, kernel, dtype_size_bytes)
}

/// Run a write-into contiguize with a specific byte-width kernel pointer.
/// Shares `contiguize_with`'s shape/stride marshaling but writes into a
/// pre-allocated `dest` instead of allocating one.
fn contiguize_into_with(
    source: &CudaStorageBytes,
    layout: &Layout,
    dest: &CudaStorageBytes,
    kernel: ContiguizeRun,
    dtype_size_bytes: usize,
) -> Result<()> {
    let device = source.device().clone();
    let shape_dims = layout.shape().dims();
    let rank = shape_dims.len();
    if rank == 0 {
        // Scalar contiguize = single element copy. Handle as a 1-D
        // contiguize of length 1; the kernel walks 1 output element.
        let dest_bytes = dtype_size_bytes;
        if dest.len_bytes() != dest_bytes {
            return Err(Error::Msg(format!(
                "contiguize_into: dest buffer is {} bytes, expected {dest_bytes} for a \
                 rank-0 scalar",
                dest.len_bytes(),
            ))
            .bt());
        }
        let stream = device.stream().as_raw() as *mut std::ffi::c_void;
        let dest_ptr = dest.buffer().as_raw().0 as *mut std::ffi::c_void;
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
        check(status, "contiguize_b*_scalar_into")?;
        return Ok(());
    }
    if rank > 8 {
        return Err(Error::Msg(format!(
            "baracuda contiguize_into: rank {rank} > 8 not supported",
        ))
        .bt());
    }
    let elem_count: usize = shape_dims.iter().copied().product();
    let dest_bytes = elem_count * dtype_size_bytes;
    if dest.len_bytes() != dest_bytes {
        return Err(Error::Msg(format!(
            "contiguize_into: dest buffer is {} bytes, expected {dest_bytes} for \
             {elem_count} elements",
            dest.len_bytes(),
        ))
        .bt());
    }
    if dest_bytes == 0 {
        // Empty tensor — no-op write, don't call the kernel.
        return Ok(());
    }

    // Convert shape (usize → i32) + strides (isize → i64). Pre-size
    // arrays at rank 8 so the stack layout doesn't escape.
    let mut shape_i32 = [0_i32; 8];
    let mut strides_i64 = [0_i64; 8];
    let strides = layout.stride();
    if strides.len() != rank {
        return Err(Error::Msg(format!(
            "baracuda contiguize_into: layout stride rank {} != shape rank {rank}",
            strides.len(),
        ))
        .bt());
    }
    for i in 0..rank {
        shape_i32[i] = i32::try_from(shape_dims[i]).map_err(|_| {
            Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                op: "contiguize_b*_into",
                dim_index: i,
                dim_value: shape_dims[i],
            })
        })?;
        strides_i64[i] = strides[i] as i64;
    }

    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let dest_ptr = dest.buffer().as_raw().0 as *mut std::ffi::c_void;
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
    check(status, "contiguize_b*_into")?;
    Ok(())
}

// Re-export the unused DeviceBuffer ref so `cargo build` doesn't
// flag it. (Some sm-feature combinations strip the kernel-import use.)
#[allow(dead_code)]
fn _keep_imports_live(_: &DeviceBuffer<u8>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaDevice;

    fn dev_or_skip() -> Option<CudaDevice> {
        match CudaDevice::new(0) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("no CUDA device; skipping: {e:?}");
                None
            }
        }
    }

    /// `contiguize_into` (write-into) must byte-match `contiguize_to_fresh`
    /// (allocating) for the same non-contiguous (transposed) input —
    /// CapturedRun executor build-out, 4b-γ.
    #[test]
    #[ignore = "requires a live CUDA device"]
    fn contiguize_into_matches_contiguize_to_fresh_strided() {
        let Some(dev) = dev_or_skip() else { return };
        use fuel_ir::Shape;

        // shape [3, 4] row-major -> transpose(0,1) is a non-contiguous
        // [4, 3] view over the same 12-element buffer.
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let source = CudaStorageBytes::from_cpu_bytes(&dev, &bytes).unwrap();
        let layout = Layout::contiguous(Shape::from_dims(&[3, 4]))
            .transpose(0, 1)
            .unwrap();
        assert!(!layout.is_contiguous(), "transpose must be non-contiguous");

        let expect = contiguize_to_fresh(&source, &layout, 4).unwrap();
        let expect_bytes = expect.to_cpu_bytes().unwrap();

        let dest = CudaStorageBytes::alloc(&dev, expect.len_bytes()).unwrap();
        contiguize_into(&source, &layout, &dest, 4).unwrap();
        let got_bytes = dest.to_cpu_bytes().unwrap();

        assert_eq!(
            got_bytes, expect_bytes,
            "contiguize_into must byte-match contiguize_to_fresh"
        );
    }

    /// `contiguize_into` refuses a dest buffer of the wrong size (typed
    /// error, never a panic) — the "never panic on production paths" rule.
    #[test]
    #[ignore = "requires a live CUDA device"]
    fn contiguize_into_rejects_mismatched_dest_size() {
        let Some(dev) = dev_or_skip() else { return };
        use fuel_ir::Shape;

        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let source = CudaStorageBytes::from_cpu_bytes(&dev, &bytes).unwrap();
        let layout = Layout::contiguous(Shape::from_dims(&[3, 4]))
            .transpose(0, 1)
            .unwrap();

        // Correct size is 12 * 4 = 48 bytes; give it a too-small dest.
        let dest = CudaStorageBytes::alloc(&dev, 16).unwrap();
        let err = contiguize_into(&source, &layout, &dest, 4);
        assert!(err.is_err(), "mismatched dest size must be a typed error, not a panic");
    }
}

//! Affine kernels from `baracuda-kernels-sys` — `y = a * x + b` with
//! scalar `(a, b)`. Contig-only on the baracuda surface; rank-1 in
//! effect (the kernel treats x as a flat numel-long buffer).
//!
//! ## Per-dtype scalar typing
//!
//! Baracuda's affine signature carries the scalar params in the storage
//! dtype's natural arithmetic type:
//!
//! | Storage dtype | `a, b` C type | Note                            |
//! |---------------|---------------|---------------------------------|
//! | `f32`         | `f32`         |                                 |
//! | `f64`         | `f64`         |                                 |
//! | `f16`         | `f32`         | f32 compute, f16 storage        |
//! | `bf16`        | `f32`         | f32 compute, bf16 storage       |
//! | `i32`         | `i32`         |                                 |
//! | `i64`         | `i64`         |                                 |
//! | `u8`          | `u8`          |                                 |
//! | `i8`          | `i8`          |                                 |
//!
//! Fuel's `OpParams::Affine { mul: f64, add: f64 }` always holds f64
//! params. The wrappers cast to the per-dtype scalar type at the FFI
//! boundary. For float storage this is lossless to f32 (the host f64
//! is downcast); for integers it's lossy if the host scalars exceed
//! the target range (a debug-level concern — callers building integer
//! Affine specify in-range constants).

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::Result;

use crate::byte_storage::CudaStorageBytes;

use super::scratch::Workspace;
use super::status::check;

/// Common per-launch state for every (dtype) variant. Returns the
/// allocated output `CudaStorageBytes` to the caller after the kernel
/// returns + the device synchronizes.
fn affine_alloc_and_sync(
    src: &CudaStorageBytes,
    dtype_size_bytes: usize,
    op_label: &'static str,
) -> Result<(CudaStorageBytes, i64, *const std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void)> {
    let device = src.device().clone();
    if src.len_bytes() % dtype_size_bytes != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "{op_label}: src.len_bytes={} not a multiple of dtype size {}",
            src.len_bytes(),
            dtype_size_bytes,
        ))
        .bt());
    }
    let numel = (src.len_bytes() / dtype_size_bytes) as i64;
    if numel == 0 {
        let zero = CudaStorageBytes::alloc(&device, 0)?;
        // Return null pointers; caller short-circuits via numel == 0.
        return Ok((
            zero,
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            device.stream().as_raw() as *mut std::ffi::c_void,
        ));
    }
    let out_bytes = numel as usize * dtype_size_bytes;
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;
    let out = CudaStorageBytes::from_parts(Arc::new(out_buf), device, out_bytes);
    Ok((out, numel, x_ptr, y_ptr, stream))
}

/// Affine f32: `y = a*x + b`.
pub fn affine_f32(src: &CudaStorageBytes, mul: f32, add: f32) -> Result<CudaStorageBytes> {
    let op_label = "affine_f32";
    let (out, numel, x, y, stream) = affine_alloc_and_sync(src, 4, op_label)?;
    if numel == 0 {
        return Ok(out);
    }
    let device = src.device().clone();
    let scratch = Workspace::alloc(&device, 0)?;
    // SAFETY: pointers + numel validated above; workspace null/0
    // (no scratch needed for affine).
    let status = unsafe {
        sys::baracuda_kernels_affine_f32_run(
            numel,
            x,
            y,
            mul,
            add,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(out)
}

/// Affine f64: `y = a*x + b`.
pub fn affine_f64(src: &CudaStorageBytes, mul: f64, add: f64) -> Result<CudaStorageBytes> {
    let op_label = "affine_f64";
    let (out, numel, x, y, stream) = affine_alloc_and_sync(src, 8, op_label)?;
    if numel == 0 {
        return Ok(out);
    }
    let device = src.device().clone();
    let scratch = Workspace::alloc(&device, 0)?;
    let status = unsafe {
        sys::baracuda_kernels_affine_f64_run(
            numel,
            x,
            y,
            mul,
            add,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(out)
}

/// Affine f16: `y = a*x + b`. `a` / `b` arrive as `f32` (storage is
/// f16, compute is f32).
pub fn affine_f16(src: &CudaStorageBytes, mul: f32, add: f32) -> Result<CudaStorageBytes> {
    let op_label = "affine_f16";
    let (out, numel, x, y, stream) = affine_alloc_and_sync(src, 2, op_label)?;
    if numel == 0 {
        return Ok(out);
    }
    let device = src.device().clone();
    let scratch = Workspace::alloc(&device, 0)?;
    let status = unsafe {
        sys::baracuda_kernels_affine_f16_run(
            numel,
            x,
            y,
            mul,
            add,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(out)
}

/// Affine bf16: `y = a*x + b`. `a` / `b` arrive as `f32`.
pub fn affine_bf16(src: &CudaStorageBytes, mul: f32, add: f32) -> Result<CudaStorageBytes> {
    let op_label = "affine_bf16";
    let (out, numel, x, y, stream) = affine_alloc_and_sync(src, 2, op_label)?;
    if numel == 0 {
        return Ok(out);
    }
    let device = src.device().clone();
    let scratch = Workspace::alloc(&device, 0)?;
    let status = unsafe {
        sys::baracuda_kernels_affine_bf16_run(
            numel,
            x,
            y,
            mul,
            add,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(out)
}

/// Affine i32: `y = a*x + b`. Truncating arithmetic (CUDA `__add_sat`
/// semantics? — baracuda's docs are silent; treat as wrap-on-overflow).
pub fn affine_i32(src: &CudaStorageBytes, mul: i32, add: i32) -> Result<CudaStorageBytes> {
    let op_label = "affine_i32";
    let (out, numel, x, y, stream) = affine_alloc_and_sync(src, 4, op_label)?;
    if numel == 0 {
        return Ok(out);
    }
    let device = src.device().clone();
    let scratch = Workspace::alloc(&device, 0)?;
    let status = unsafe {
        sys::baracuda_kernels_affine_i32_run(
            numel,
            x,
            y,
            mul,
            add,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(out)
}

/// Affine i64: `y = a*x + b`.
pub fn affine_i64(src: &CudaStorageBytes, mul: i64, add: i64) -> Result<CudaStorageBytes> {
    let op_label = "affine_i64";
    let (out, numel, x, y, stream) = affine_alloc_and_sync(src, 8, op_label)?;
    if numel == 0 {
        return Ok(out);
    }
    let device = src.device().clone();
    let scratch = Workspace::alloc(&device, 0)?;
    let status = unsafe {
        sys::baracuda_kernels_affine_i64_run(
            numel,
            x,
            y,
            mul,
            add,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(out)
}

/// Affine u8: `y = a*x + b`.
pub fn affine_u8(src: &CudaStorageBytes, mul: u8, add: u8) -> Result<CudaStorageBytes> {
    let op_label = "affine_u8";
    let (out, numel, x, y, stream) = affine_alloc_and_sync(src, 1, op_label)?;
    if numel == 0 {
        return Ok(out);
    }
    let device = src.device().clone();
    let scratch = Workspace::alloc(&device, 0)?;
    let status = unsafe {
        sys::baracuda_kernels_affine_u8_run(
            numel,
            x,
            y,
            mul,
            add,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(out)
}

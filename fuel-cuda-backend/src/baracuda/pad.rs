//! Pad / PadBackward kernels from `baracuda-kernels-sys`.
//!
//! Forward: 4 modes (Constant/Reflect/Replicate/Circular) × 4 dtypes
//! (f32/f16/bf16/f64). Backward: only Constant mode (the other modes'
//! backward gradients are reflective/replicating sums that baracuda
//! doesn't ship — Fuel's CPU path is the only differentiable target
//! for non-Constant modes today).
//!
//! Fuel's `OpParams::Pad { in_shape, out_shape, padding, mode_tag,
//! fill_bytes }` carries per-axis `(before, after)` and a mode tag
//! (0=Constant, 1=Reflect, 2=Replicate, 3=Circular). baracuda's
//! interface takes per-axis `pad_low` (= the `before` side); the
//! `after` side is implied by `output_shape[i] - input_shape[i] -
//! pad_low[i]`. We pre-encode `fill_bytes` to the typed `value`
//! argument at dispatch time.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::{Error, Result};

use crate::byte_storage::CudaStorageBytes;

use super::status::check;

/// Mode tags shared with Fuel's IR.
pub const PAD_MODE_CONSTANT:  u8 = 0;
pub const PAD_MODE_REFLECT:   u8 = 1;
pub const PAD_MODE_REPLICATE: u8 = 2;
pub const PAD_MODE_CIRCULAR:  u8 = 3;

/// Helper: validate ranks, convert shape/stride arrays, produce
/// `(input_shape_i32, output_shape_i32, pad_low_i32, stride_x_i64,
/// stride_y_i64)`. Returns `Err` on overflow or rank mismatch.
struct PadShapes {
    rank: i32,
    output_numel: i64,
    input_shape: [i32; 8],
    output_shape: [i32; 8],
    pad_low: [i32; 8],
    stride_x: [i64; 8],
    stride_y: [i64; 8],
}

fn build_pad_shapes(
    in_shape: &[usize],
    out_shape: &[usize],
    padding: &[(usize, usize)],
    op_label: &'static str,
) -> Result<PadShapes> {
    let rank = in_shape.len();
    if out_shape.len() != rank || padding.len() != rank {
        return Err(Error::Msg(format!(
            "{op_label}: rank mismatch in_shape={} out_shape={} padding={}",
            rank, out_shape.len(), padding.len(),
        ))
        .bt());
    }
    if rank == 0 || rank > 8 {
        return Err(Error::Msg(format!(
            "{op_label}: rank {rank} out of range (1..=8)",
        ))
        .bt());
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

    let mut input_shape = [0_i32; 8];
    let mut output_shape = [0_i32; 8];
    let mut pad_low = [0_i32; 8];
    for i in 0..rank {
        input_shape[i] = i32_or(i, in_shape[i])?;
        output_shape[i] = i32_or(i, out_shape[i])?;
        pad_low[i] = i32_or(i, padding[i].0)?;
    }

    // Row-major contiguous strides for both input and output.
    let mut stride_x = [0_i64; 8];
    let mut stride_y = [0_i64; 8];
    if rank > 0 {
        stride_x[rank - 1] = 1;
        stride_y[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            stride_x[i] = stride_x[i + 1] * in_shape[i + 1] as i64;
            stride_y[i] = stride_y[i + 1] * out_shape[i + 1] as i64;
        }
    }

    let output_numel: usize = out_shape.iter().copied().product();
    Ok(PadShapes {
        rank: rank as i32,
        output_numel: output_numel as i64,
        input_shape,
        output_shape,
        pad_low,
        stride_x,
        stride_y,
    })
}

// ===========================================================================
// Forward — per-mode FFI signatures
// ===========================================================================

type PadConstantRunF32 = unsafe extern "C" fn(
    output_numel: i64,
    rank: i32,
    input_shape: *const i32,
    output_shape: *const i32,
    pad_low: *const i32,
    stride_x: *const i64,
    stride_y: *const i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    value: f32,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

type PadConstantRunF64 = unsafe extern "C" fn(
    output_numel: i64,
    rank: i32,
    input_shape: *const i32,
    output_shape: *const i32,
    pad_low: *const i32,
    stride_x: *const i64,
    stride_y: *const i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    value: f64,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

type PadConstantRunU16 = unsafe extern "C" fn(
    output_numel: i64,
    rank: i32,
    input_shape: *const i32,
    output_shape: *const i32,
    pad_low: *const i32,
    stride_x: *const i64,
    stride_y: *const i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    value: u16,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// FFI signature for Reflect/Replicate/Circular forward (no value param).
type PadModeRun = unsafe extern "C" fn(
    output_numel: i64,
    rank: i32,
    input_shape: *const i32,
    output_shape: *const i32,
    pad_low: *const i32,
    stride_x: *const i64,
    stride_y: *const i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

fn alloc_dest(input: &CudaStorageBytes, out_bytes: usize) -> Result<(crate::CudaDevice, baracuda_driver::DeviceBuffer<u8>)> {
    let device = input.device().clone();
    let dest = device.alloc_zeros::<u8>(out_bytes)?;
    Ok((device, dest))
}

/// Run a non-constant-mode forward (reflect/replicate/circular).
fn run_pad_mode(
    input: &CudaStorageBytes,
    shapes: &PadShapes,
    kernel: PadModeRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let out_bytes = shapes.output_numel as usize * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(input.device(), 0);
    }
    let (device, dest) = alloc_dest(input, out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let in_ptr = input.buffer().as_raw().0 as *const std::ffi::c_void;
    let out_ptr = dest.as_raw().0 as *mut std::ffi::c_void;
    // SAFETY: pointers valid; stack arrays live for the FFI call.
    let status = unsafe {
        kernel(
            shapes.output_numel,
            shapes.rank,
            shapes.input_shape.as_ptr(),
            shapes.output_shape.as_ptr(),
            shapes.pad_low.as_ptr(),
            shapes.stride_x.as_ptr(),
            shapes.stride_y.as_ptr(),
            in_ptr,
            out_ptr,
            std::ptr::null_mut(),
            0,
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(dest), device, out_bytes))
}

// ----- Per-dtype constant runners ----------------------------------------

fn run_pad_constant_f32(
    input: &CudaStorageBytes,
    shapes: &PadShapes,
    kernel: PadConstantRunF32,
    value: f32,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let out_bytes = shapes.output_numel as usize * 4;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(input.device(), 0);
    }
    let (device, dest) = alloc_dest(input, out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let in_ptr = input.buffer().as_raw().0 as *const std::ffi::c_void;
    let out_ptr = dest.as_raw().0 as *mut std::ffi::c_void;
    let status = unsafe {
        kernel(
            shapes.output_numel, shapes.rank,
            shapes.input_shape.as_ptr(), shapes.output_shape.as_ptr(),
            shapes.pad_low.as_ptr(),
            shapes.stride_x.as_ptr(), shapes.stride_y.as_ptr(),
            in_ptr, out_ptr, value,
            std::ptr::null_mut(), 0, stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(dest), device, out_bytes))
}

fn run_pad_constant_f64(
    input: &CudaStorageBytes,
    shapes: &PadShapes,
    kernel: PadConstantRunF64,
    value: f64,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let out_bytes = shapes.output_numel as usize * 8;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(input.device(), 0);
    }
    let (device, dest) = alloc_dest(input, out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let in_ptr = input.buffer().as_raw().0 as *const std::ffi::c_void;
    let out_ptr = dest.as_raw().0 as *mut std::ffi::c_void;
    let status = unsafe {
        kernel(
            shapes.output_numel, shapes.rank,
            shapes.input_shape.as_ptr(), shapes.output_shape.as_ptr(),
            shapes.pad_low.as_ptr(),
            shapes.stride_x.as_ptr(), shapes.stride_y.as_ptr(),
            in_ptr, out_ptr, value,
            std::ptr::null_mut(), 0, stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(dest), device, out_bytes))
}

fn run_pad_constant_u16(
    input: &CudaStorageBytes,
    shapes: &PadShapes,
    kernel: PadConstantRunU16,
    value: u16,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let out_bytes = shapes.output_numel as usize * 2;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(input.device(), 0);
    }
    let (device, dest) = alloc_dest(input, out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let in_ptr = input.buffer().as_raw().0 as *const std::ffi::c_void;
    let out_ptr = dest.as_raw().0 as *mut std::ffi::c_void;
    let status = unsafe {
        kernel(
            shapes.output_numel, shapes.rank,
            shapes.input_shape.as_ptr(), shapes.output_shape.as_ptr(),
            shapes.pad_low.as_ptr(),
            shapes.stride_x.as_ptr(), shapes.stride_y.as_ptr(),
            in_ptr, out_ptr, value,
            std::ptr::null_mut(), 0, stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(dest), device, out_bytes))
}

// ===========================================================================
// Public per-dtype entry points (forward, all 3 modes Fuel emits)
// ===========================================================================
//
// One entry point per (dtype, mode_tag) dispatch — the dispatch
// wrapper in fuel-storage's baracuda_dispatch reads mode_tag from
// OpParams::Pad and picks the right entry. fill_bytes is decoded to
// the typed value at the dispatch layer (we receive a typed f32/f64/
// u16/u16 value here).

pub fn pad_constant_f32(
    input: &CudaStorageBytes,
    in_shape: &[usize],
    out_shape: &[usize],
    padding: &[(usize, usize)],
    value: f32,
) -> Result<CudaStorageBytes> {
    let shapes = build_pad_shapes(in_shape, out_shape, padding, "pad_constant_f32")?;
    run_pad_constant_f32(
        input, &shapes,
        sys::baracuda_kernels_pad_constant_f32_run,
        value, "pad_constant_f32",
    )
}

pub fn pad_constant_f64(
    input: &CudaStorageBytes,
    in_shape: &[usize],
    out_shape: &[usize],
    padding: &[(usize, usize)],
    value: f64,
) -> Result<CudaStorageBytes> {
    let shapes = build_pad_shapes(in_shape, out_shape, padding, "pad_constant_f64")?;
    run_pad_constant_f64(
        input, &shapes,
        sys::baracuda_kernels_pad_constant_f64_run,
        value, "pad_constant_f64",
    )
}

pub fn pad_constant_f16(
    input: &CudaStorageBytes,
    in_shape: &[usize],
    out_shape: &[usize],
    padding: &[(usize, usize)],
    value: half::f16,
) -> Result<CudaStorageBytes> {
    let shapes = build_pad_shapes(in_shape, out_shape, padding, "pad_constant_f16")?;
    run_pad_constant_u16(
        input, &shapes,
        sys::baracuda_kernels_pad_constant_f16_run,
        value.to_bits(), "pad_constant_f16",
    )
}

pub fn pad_constant_bf16(
    input: &CudaStorageBytes,
    in_shape: &[usize],
    out_shape: &[usize],
    padding: &[(usize, usize)],
    value: half::bf16,
) -> Result<CudaStorageBytes> {
    let shapes = build_pad_shapes(in_shape, out_shape, padding, "pad_constant_bf16")?;
    run_pad_constant_u16(
        input, &shapes,
        sys::baracuda_kernels_pad_constant_bf16_run,
        value.to_bits(), "pad_constant_bf16",
    )
}

macro_rules! pad_mode_kernel {
    ($name:ident, $sys_fn:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        pub fn $name(
            input: &CudaStorageBytes,
            in_shape: &[usize],
            out_shape: &[usize],
            padding: &[(usize, usize)],
        ) -> Result<CudaStorageBytes> {
            let shapes = build_pad_shapes(in_shape, out_shape, padding, $op_label)?;
            run_pad_mode(input, &shapes, sys::$sys_fn, $op_label, $dtype_size)
        }
    };
}

pad_mode_kernel!(pad_reflect_f32,  baracuda_kernels_pad_reflect_f32_run,  4, "pad_reflect_f32");
pad_mode_kernel!(pad_reflect_f64,  baracuda_kernels_pad_reflect_f64_run,  8, "pad_reflect_f64");
pad_mode_kernel!(pad_reflect_f16,  baracuda_kernels_pad_reflect_f16_run,  2, "pad_reflect_f16");
pad_mode_kernel!(pad_reflect_bf16, baracuda_kernels_pad_reflect_bf16_run, 2, "pad_reflect_bf16");

pad_mode_kernel!(pad_replicate_f32,  baracuda_kernels_pad_replicate_f32_run,  4, "pad_replicate_f32");
pad_mode_kernel!(pad_replicate_f64,  baracuda_kernels_pad_replicate_f64_run,  8, "pad_replicate_f64");
pad_mode_kernel!(pad_replicate_f16,  baracuda_kernels_pad_replicate_f16_run,  2, "pad_replicate_f16");
pad_mode_kernel!(pad_replicate_bf16, baracuda_kernels_pad_replicate_bf16_run, 2, "pad_replicate_bf16");

// ===========================================================================
// Backward — pad-constant slice (only Constant mode is differentiable
// through this path; other modes' backwards are sum-accumulating and
// baracuda doesn't ship them)
// ===========================================================================

type PadConstantBackwardRun = unsafe extern "C" fn(
    input_numel: i64,
    rank: i32,
    input_shape: *const i32,
    pad_low: *const i32,
    stride_dy: *const i64,
    stride_dx: *const i64,
    dy: *const std::ffi::c_void,
    dx: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

fn run_pad_backward(
    dy: &CudaStorageBytes,
    dx_shape: &[usize],
    dy_shape: &[usize],
    padding: &[(usize, usize)],
    kernel: PadConstantBackwardRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let shapes = build_pad_shapes(dx_shape, dy_shape, padding, op_label)?;
    let dx_bytes = (dx_shape.iter().product::<usize>()) * dtype_size_bytes;
    if dx_bytes == 0 {
        return CudaStorageBytes::alloc(dy.device(), 0);
    }
    let (device, dx) = alloc_dest(dy, dx_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let dy_ptr = dy.buffer().as_raw().0 as *const std::ffi::c_void;
    let dx_ptr = dx.as_raw().0 as *mut std::ffi::c_void;
    let input_numel = dx_shape.iter().product::<usize>() as i64;
    // SAFETY: pointers valid; stack arrays live for the FFI call.
    let status = unsafe {
        kernel(
            input_numel,
            shapes.rank,
            shapes.input_shape.as_ptr(),
            shapes.pad_low.as_ptr(),
            // stride_dy is the FORWARD output's stride pattern
            // (= y_shape contig strides). stride_dx is the FORWARD
            // input's stride pattern (= x_shape contig strides).
            // We map: forward y → backward dy; forward x → backward dx.
            shapes.stride_y.as_ptr(),
            shapes.stride_x.as_ptr(),
            dy_ptr,
            dx_ptr,
            std::ptr::null_mut(),
            0,
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(dx), device, dx_bytes))
}

macro_rules! pad_backward_kernel {
    ($name:ident, $sys_fn:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        pub fn $name(
            dy: &CudaStorageBytes,
            dx_shape: &[usize],
            dy_shape: &[usize],
            padding: &[(usize, usize)],
        ) -> Result<CudaStorageBytes> {
            run_pad_backward(
                dy, dx_shape, dy_shape, padding,
                sys::$sys_fn, $op_label, $dtype_size,
            )
        }
    };
}

pad_backward_kernel!(pad_backward_f32,  baracuda_kernels_pad_constant_backward_f32_run,  4, "pad_backward_f32");
pad_backward_kernel!(pad_backward_f64,  baracuda_kernels_pad_constant_backward_f64_run,  8, "pad_backward_f64");
pad_backward_kernel!(pad_backward_f16,  baracuda_kernels_pad_constant_backward_f16_run,  2, "pad_backward_f16");
pad_backward_kernel!(pad_backward_bf16, baracuda_kernels_pad_constant_backward_bf16_run, 2, "pad_backward_bf16");

//! Broadcast-reverse reductions over baracuda's `reduce_{sum,max}_to_*`
//! family — the autograd gradient-of-broadcast primitive Fuel calls
//! `ReduceSumTo` / `ReduceMaxTo`.
//!
//! The symbols have shipped sys-only since alpha.46 (Phase 31) at full
//! `{f32, f64, f16, bf16}` coverage; the 2026-06-10 ask-reply surfaced
//! them (they were invisible to a facade-level audit — same gap class
//! as `unary_step`). This module generalizes the f32-only wrappers
//! that previously lived in `byte_kernels` and retires that module's
//! reduce arm.
//!
//! Contract (per the alpha.67 reply): per-dim `out[d] ∈ {1, in[d]}`,
//! output left-padded with 1s to input rank, arbitrary input strides,
//! contiguous output, deterministic sequential per-cell accumulation,
//! f16/bf16 accumulating in f32, rank ≤ 8.
//!
//! Empty-reduce-set caveat: cells whose reduce set is empty (a 0-dim
//! input collapsed away) receive the identity — `0` for sum,
//! most-extreme finite value for max on f32/f64, but **`-inf` on the
//! f16/bf16 symbols** (the f32 identity overflows the storage dtype
//! on the final narrowing store).

use std::sync::Arc;

use fuel_ir::{Layout, Result};

use crate::byte_storage::CudaStorageBytes;

use baracuda_kernels_sys as sys;

/// Shared FFI shape of every `reduce_{sum,max,min,prod}_to_<dt>_run`
/// symbol (host-pointer shape/stride arrays read before launch).
type ReduceToRun = unsafe extern "C" fn(
    src: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    input_shape: *const i32,
    input_stride: *const i64,
    rank_in: i32,
    output_shape: *const i32,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Common launcher: validates ranks/byte-multiples, left-pads the
/// output shape to the input rank, and issues one FFI call.
fn reduce_to(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    input_shape: &[usize],
    output_shape: &[usize],
    run: ReduceToRun,
    elem: usize,
    label: &'static str,
) -> Result<CudaStorageBytes> {
    if src.len_bytes() % elem != 0 {
        return Err(fuel_ir::Error::Msg(format!(
            "{label}: src.len_bytes={} not a multiple of element size {elem}",
            src.len_bytes(),
        ))
        .bt());
    }
    if output_shape.len() > input_shape.len() {
        return Err(fuel_ir::Error::Msg(format!(
            "{label}: output rank {} exceeds input rank {}",
            output_shape.len(),
            input_shape.len(),
        ))
        .bt());
    }
    let rank_in = input_shape.len();
    let in_shape_i32: Vec<i32> = input_shape.iter().map(|&d| d as i32).collect();
    let in_stride_i64: Vec<i64> = input_layout.stride().iter().map(|&s| s as i64).collect();
    // Left-pad output_shape with 1s to match rank_in (baracuda's contract).
    let mut out_shape_padded: Vec<i32> = vec![1_i32; rank_in - output_shape.len()];
    out_shape_padded.extend(output_shape.iter().map(|&d| d as i32));

    let dst_el: usize = output_shape.iter().product();
    let dst_bytes = dst_el * elem;
    let device = src.device().clone();
    if dst_el == 0 {
        return CudaStorageBytes::alloc(&device, dst_bytes);
    }

    let out_buf = device.alloc_zeros::<u8>(dst_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    // SAFETY: `input_shape`, `input_stride`, `output_shape` are HOST
    // pointers per baracuda's documented ABI (the launcher reads them
    // before issuing the kernel). The Vecs live through the call, so
    // their `.as_ptr()` is valid for the duration. `src`/`out_buf` are
    // device-resident, `stream` is valid; workspace null/0.
    let status = unsafe {
        run(
            src.buffer().as_raw().0 as *const std::ffi::c_void,
            out_buf.as_raw().0 as *mut std::ffi::c_void,
            in_shape_i32.as_ptr(),
            in_stride_i64.as_ptr(),
            rank_in as i32,
            out_shape_padded.as_ptr(),
            std::ptr::null_mut(),
            0,
            stream,
        )
    };
    crate::baracuda::status::check(status, label)?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out_buf),
        device,
        dst_bytes,
    ))
}

/// One `(op, dtype)` manifest line, mirroring the
/// `unary_kernel!`-style pattern used across `baracuda::*`.
macro_rules! reduce_to_kernel {
    ($name:ident, $run:path, $elem:expr, $label:expr $(,)?) => {
        #[doc = concat!(
            "Broadcast-reverse `", $label, "` via baracuda's ",
            "`reduce_*_to` family. Strided input, contiguous output.",
        )]
        pub fn $name(
            src: &CudaStorageBytes,
            input_layout: &Layout,
            input_shape: &[usize],
            output_shape: &[usize],
        ) -> Result<CudaStorageBytes> {
            reduce_to(
                src, input_layout, input_shape, output_shape,
                $run, $elem, $label,
            )
        }
    };
}

reduce_to_kernel!(reduce_sum_to_f32,  sys::baracuda_kernels_reduce_sum_to_f32_run,  4, "reduce_sum_to_f32");
reduce_to_kernel!(reduce_sum_to_f64,  sys::baracuda_kernels_reduce_sum_to_f64_run,  8, "reduce_sum_to_f64");
reduce_to_kernel!(reduce_sum_to_f16,  sys::baracuda_kernels_reduce_sum_to_f16_run,  2, "reduce_sum_to_f16");
reduce_to_kernel!(reduce_sum_to_bf16, sys::baracuda_kernels_reduce_sum_to_bf16_run, 2, "reduce_sum_to_bf16");

reduce_to_kernel!(reduce_max_to_f32,  sys::baracuda_kernels_reduce_max_to_f32_run,  4, "reduce_max_to_f32");
reduce_to_kernel!(reduce_max_to_f64,  sys::baracuda_kernels_reduce_max_to_f64_run,  8, "reduce_max_to_f64");
reduce_to_kernel!(reduce_max_to_f16,  sys::baracuda_kernels_reduce_max_to_f16_run,  2, "reduce_max_to_f16");
reduce_to_kernel!(reduce_max_to_bf16, sys::baracuda_kernels_reduce_max_to_bf16_run, 2, "reduce_max_to_bf16");

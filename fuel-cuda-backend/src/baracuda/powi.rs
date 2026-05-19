//! PowI (integer-exponent power) kernels from `baracuda-kernels-sys`.
//! Alpha.28's `unary_param`-shaped op: `y = x^n` via power-by-squaring,
//! correct on negative bases (no `pow(x, n as f32)` shim).
//!
//! ## ABI
//!
//! ```text
//! fn run(numel, x, y, p0, p1, workspace, workspace_bytes, stream) -> i32
//! ```
//!
//! The exponent `n: i32` is reinterpret-cast to `f32` and shipped via
//! `p0`. baracuda's kernel casts back to `int` on the device side.
//! Per the baracuda docs, reasonable `|n| ≤ 2^24` round-trip through
//! f32 exactly; out-of-range exponents land in the "lossy" zone but
//! aren't valid PowI inputs anyway (the operation has integer
//! semantics).
//!
//! `p1` is ignored — kept by baracuda for ABI parity with the rest of
//! the `unary_param_*` family.
//!
//! Contig-only on the baracuda surface for now; Fuel's executor
//! Contiguizes non-contig consumers before the dispatch.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::{Error, Result};

use crate::byte_storage::CudaStorageBytes;

use super::scratch::Workspace;
use super::status::check;

type PowiRun = unsafe extern "C" fn(
    numel: i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    p0: f32,
    p1: f32,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

fn powi_run(
    src: &CudaStorageBytes,
    exp: i32,
    dtype_size_bytes: usize,
    kernel: PowiRun,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let device = src.device().clone();
    if src.len_bytes() % dtype_size_bytes != 0 {
        return Err(Error::Msg(format!(
            "{op_label}: src.len_bytes={} not a multiple of dtype size {}",
            src.len_bytes(),
            dtype_size_bytes,
        ))
        .bt());
    }
    let numel = (src.len_bytes() / dtype_size_bytes) as i64;
    if numel == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_bytes = src.len_bytes();
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    // Per baracuda ABI: exponent is shipped as f32-cast-from-i32. The
    // round-trip is exact for |n| ≤ 2^24 — covering every realistic
    // PowI argument.
    let p0 = exp as f32;

    // SAFETY: pointers + numel validated above; workspace null/0 (no
    // scratch needed for powi); stream lives on the device for the
    // call's duration.
    let status = unsafe {
        kernel(
            numel,
            x_ptr,
            y_ptr,
            p0,
            0.0_f32, // p1 ignored
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

macro_rules! powi_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `", $op_label, "` kernel.")]
            pub fn $name(src: &CudaStorageBytes, exp: i32) -> Result<CudaStorageBytes> {
                powi_run(
                    src,
                    exp,
                    $dtype_size,
                    sys::[<baracuda_kernels_unary_powi_ $sys_stem _run>],
                    $op_label,
                )
            }
        }
    };
}

powi_kernel!(powi_f32, f32, 4, "powi_f32");
powi_kernel!(powi_f64, f64, 8, "powi_f64");
powi_kernel!(powi_f16, f16, 2, "powi_f16");
powi_kernel!(powi_bf16, bf16, 2, "powi_bf16");

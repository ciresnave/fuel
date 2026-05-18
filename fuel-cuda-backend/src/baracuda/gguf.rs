//! GGUF dequant + MMVQ kernels from `baracuda-kernels-sys`.
//!
//! ## Coverage today (alpha.27)
//!
//! Dequant: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K,
//! Q6_K, Q8_K. Block→f32. Per-format dispatch via concrete public
//! functions; the dispatch wrapper picks by `QuantType` at call
//! time.
//!
//! MMVQ: same 11 formats (alpha.27 added Q8_K MMVQ — closes the
//! Tier-2 gap from the alpha.26 audit). MMVQ is strictly
//! matrix-vector (one activation row × quantized weight matrix);
//! Fuel's `QMatMul` with `m > 1` errors out today — looping over
//! `m` rows for the matrix-matrix case lands in a follow-up.
//!
//! ## Dequant signature
//!
//! ```text
//! fn run(numel, x, y, workspace, workspace_bytes, stream) -> i32
//! ```
//!
//! - `x` — quantized block bytes (size = `numel * block_size / type_size`
//!   bytes per format; caller responsible for getting this right).
//! - `y` — fresh `f32` output buffer of `numel × sizeof(f32)` bytes.
//! - `numel` — block count × block_size (i.e., total dequantized
//!   element count).
//!
//! ## MMVQ signature
//!
//! ```text
//! fn run(ncols, nrows, x, y, dst, workspace, workspace_bytes, stream) -> i32
//! ```
//!
//! - `x` — quantized weight matrix (`[nrows, ncols]` packed in the
//!   block format).
//! - `y` — `f32` activation vector (`[ncols]`).
//! - `dst` — `f32` output vector (`[nrows]`).

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::Result;

use crate::byte_storage::CudaStorageBytes;

use super::scratch::Workspace;
use super::status::check;

type DequantRun = unsafe extern "C" fn(
    numel: i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

type MmvqRun = unsafe extern "C" fn(
    ncols: i32,
    nrows: i32,
    x: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Dequant one block-format-encoded buffer into a fresh `f32`
/// output. Caller passes the dequantized element count (`numel`).
/// Output size = `numel * sizeof(f32)` bytes.
fn dequant_run(
    src: &CudaStorageBytes,
    numel: usize,
    kernel: DequantRun,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let device = src.device().clone();
    let out_bytes = numel * std::mem::size_of::<f32>();
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    // SAFETY: x bytes are validated by the caller's QuantType +
    // block size contract; output buffer is contig + correctly
    // sized for the f32 element count.
    let status = unsafe {
        kernel(
            numel as i64,
            x_ptr,
            y_ptr,
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

/// MMVQ — fused dequant + matrix-vector multiply.
/// `weights` is `[nrows, ncols]` packed in the block format;
/// `activations` is `[ncols]` f32; output is `[nrows]` f32.
fn mmvq_run(
    weights: &CudaStorageBytes,
    activations: &CudaStorageBytes,
    ncols: usize,
    nrows: usize,
    kernel: MmvqRun,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let device = weights.device().clone();
    if activations.device().id() != device.id() {
        return Err(fuel_core_types::Error::Msg(format!(
            "{op_label}: weights and activations on different CUDA devices",
        ))
        .bt());
    }
    let ncols_i32 = i32::try_from(ncols).map_err(|_| {
        fuel_core_types::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label,
            dim_index: 0,
            dim_value: ncols,
        })
    })?;
    let nrows_i32 = i32::try_from(nrows).map_err(|_| {
        fuel_core_types::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label,
            dim_index: 1,
            dim_value: nrows,
        })
    })?;
    let out_bytes = nrows * std::mem::size_of::<f32>();
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = weights.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = activations.buffer().as_raw().0 as *const std::ffi::c_void;
    let dst_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    let status = unsafe {
        kernel(
            ncols_i32,
            nrows_i32,
            x_ptr,
            y_ptr,
            dst_ptr,
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

macro_rules! gguf_dequant {
    ($name:ident, $sys_stem:ident, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda GGUF `", $op_label, "` block-format dequantize → f32.")]
            pub fn $name(src: &CudaStorageBytes, numel: usize) -> Result<CudaStorageBytes> {
                dequant_run(
                    src,
                    numel,
                    sys::[<baracuda_kernels_dequantize_ $sys_stem _run>],
                    $op_label,
                )
            }
        }
    };
}

macro_rules! gguf_mmvq {
    ($name:ident, $sys_stem:ident, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda GGUF `", $op_label, "` MMVQ — fused dequant + matrix-vector multiply.")]
            pub fn $name(
                weights: &CudaStorageBytes,
                activations: &CudaStorageBytes,
                ncols: usize,
                nrows: usize,
            ) -> Result<CudaStorageBytes> {
                mmvq_run(
                    weights,
                    activations,
                    ncols,
                    nrows,
                    sys::[<baracuda_kernels_mmvq_ $sys_stem _run>],
                    $op_label,
                )
            }
        }
    };
}

// Dequant — type-0/1 formats (Q4_0..Q8_0, lowercase block tag)
gguf_dequant!(dequant_q4_0, q4_0, "dequant_q4_0");
gguf_dequant!(dequant_q4_1, q4_1, "dequant_q4_1");
gguf_dequant!(dequant_q5_0, q5_0, "dequant_q5_0");
gguf_dequant!(dequant_q5_1, q5_1, "dequant_q5_1");
gguf_dequant!(dequant_q8_0, q8_0, "dequant_q8_0");

// Dequant — k-quants (uppercase K block tag per the FFI naming)
gguf_dequant!(dequant_q2_k, q2_K, "dequant_q2_K");
gguf_dequant!(dequant_q3_k, q3_K, "dequant_q3_K");
gguf_dequant!(dequant_q4_k, q4_K, "dequant_q4_K");
gguf_dequant!(dequant_q5_k, q5_K, "dequant_q5_K");
gguf_dequant!(dequant_q6_k, q6_K, "dequant_q6_K");
gguf_dequant!(dequant_q8_k, q8_K, "dequant_q8_K");

// MMVQ — all 11 formats (Q8_K MMVQ added in alpha.27)
gguf_mmvq!(mmvq_q4_0, q4_0, "mmvq_q4_0");
gguf_mmvq!(mmvq_q4_1, q4_1, "mmvq_q4_1");
gguf_mmvq!(mmvq_q5_0, q5_0, "mmvq_q5_0");
gguf_mmvq!(mmvq_q5_1, q5_1, "mmvq_q5_1");
gguf_mmvq!(mmvq_q8_0, q8_0, "mmvq_q8_0");
gguf_mmvq!(mmvq_q2_k, q2_K, "mmvq_q2_K");
gguf_mmvq!(mmvq_q3_k, q3_K, "mmvq_q3_K");
gguf_mmvq!(mmvq_q4_k, q4_K, "mmvq_q4_K");
gguf_mmvq!(mmvq_q5_k, q5_K, "mmvq_q5_K");
gguf_mmvq!(mmvq_q6_k, q6_K, "mmvq_q6_K");
gguf_mmvq!(mmvq_q8_k, q8_K, "mmvq_q8_K");

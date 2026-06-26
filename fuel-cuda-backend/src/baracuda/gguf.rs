//! GGUF dequant + MMVQ kernels from `baracuda-kernels-sys`.
//!
//! ## Coverage today (alpha.27 + alpha.31 MMVQ actstrided)
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
//! ## MMVQ contig vs activation-strided
//!
//! Baracuda alpha.31 ships `mmvq_<fmt>_actstrided_run` siblings that
//! add a `stride_y: i64` for the activation operand and a
//! `w_start_byte_offset: i64` for sub-allocation of W within a larger
//! buffer. The W operand stays block-packed contig (no element-level
//! stride is meaningful for block-packed quantized storage; see the
//! baracuda team's C.5 GGUF MMVQ clarification in
//! `docs/baracuda-strided-input-audit.md`).
//!
//! The wrapper picks the actstrided FFI when:
//! - The activation layout is supplied AND non-contig; OR
//! - A non-zero `w_start_byte_offset` is supplied.
//!
//! ## K-quant alignment
//!
//! Per the baracuda team's alpha.31 carry-forward, Q4_K's actstrided
//! kernel requires the W pointer to be 16-byte aligned at the slab
//! offset. We `debug_assert!` the offset is aligned per-format; the
//! debug-build trip catches misalignment before it becomes silent
//! corruption in release.
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
//! ## MMVQ signatures
//!
//! Contig: `fn run(ncols, nrows, x, y, dst, ws, ws_b, stream) -> i32`
//! Actstrided: `fn run(ncols, nrows, x, w_off, stride_y, y, dst, ws, ws_b, stream) -> i32`

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_ir::{Error, Layout, Result};

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

/// Activation-strided MMVQ sibling (alpha.31). Adds
/// `w_start_byte_offset: i64` (W slab offset within a larger
/// allocation; 0 ⇒ no offset) and `stride_y: i64` (activation
/// element stride; 1 ⇒ contig).
type MmvqActStridedRun = unsafe extern "C" fn(
    ncols: i32,
    nrows: i32,
    x: *const std::ffi::c_void,
    w_start_byte_offset: i64,
    stride_y: i64,
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

/// Activation-element stride for `activations`. Returns `Some(1)` if
/// the layout is rank-1 contig (or absent), `Some(stride)` for a
/// rank-1 strided view, or `Err` for ranks outside `[1]`. MMVQ is
/// matrix-vector; the activation is a single vector.
fn activation_stride(
    act_layout: Option<&Layout>,
    op_label: &'static str,
) -> Result<i64> {
    let Some(layout) = act_layout else { return Ok(1); };
    let strides = layout.stride();
    match strides.len() {
        0 => Err(Error::Msg(
            format!("{op_label}: rank-0 activation not supported"),
        ).bt()),
        1 => Ok(strides[0] as i64),
        n => Err(Error::Msg(format!(
            "{op_label}: MMVQ activation must be rank-1 (got rank {n})",
        )).bt()),
    }
}

/// MMVQ — fused dequant + matrix-vector multiply.
/// `weights` is `[nrows, ncols]` packed in the block format;
/// `activations` is `[ncols]` f32; output is `[nrows]` f32.
///
/// Picks contig vs actstrided per-call:
/// - Both `stride_y == 1` AND `w_start_byte_offset == 0` ⇒ contig FFI.
/// - Otherwise ⇒ actstrided FFI (alpha.31).
///
/// `w_align_bytes` is the per-format alignment requirement (16 for
/// Q4_K, 1 for the type-0/1 formats). When non-trivial and
/// `w_start_byte_offset != 0`, a `debug_assert!` enforces alignment.
#[allow(clippy::too_many_arguments)]
fn mmvq_run(
    weights: &CudaStorageBytes,
    activations: &CudaStorageBytes,
    act_layout: Option<&Layout>,
    w_start_byte_offset: i64,
    w_align_bytes: i64,
    ncols: usize,
    nrows: usize,
    contig: MmvqRun,
    strided: MmvqActStridedRun,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let device = weights.device().clone();
    if activations.device().id() != device.id() {
        return Err(Error::Msg(format!(
            "{op_label}: weights and activations on different CUDA devices",
        )).bt());
    }
    if w_start_byte_offset < 0 {
        return Err(Error::Msg(format!(
            "{op_label}: negative w_start_byte_offset {w_start_byte_offset}",
        )).bt());
    }
    // Alignment guard (debug-only — release builds trust the caller).
    debug_assert!(
        w_align_bytes <= 1 || w_start_byte_offset % w_align_bytes == 0,
        "{}: w_start_byte_offset ({}) must be a multiple of {} bytes for this block format",
        op_label, w_start_byte_offset, w_align_bytes,
    );

    let ncols_i32 = i32::try_from(ncols).map_err(|_| {
        Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label, dim_index: 0, dim_value: ncols,
        })
    })?;
    let nrows_i32 = i32::try_from(nrows).map_err(|_| {
        Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label, dim_index: 1, dim_value: nrows,
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

    let stride_y = activation_stride(act_layout, op_label)?;
    let take_strided = stride_y != 1 || w_start_byte_offset != 0;

    let status = if take_strided {
        // SAFETY: pointers validated; FFI accepts byte offset + stride.
        unsafe {
            strided(
                ncols_i32, nrows_i32,
                x_ptr, w_start_byte_offset, stride_y,
                y_ptr, dst_ptr,
                scratch.as_raw(), scratch.bytes(), stream,
            )
        }
    } else {
        // SAFETY: pointers validated; contig fast path.
        unsafe {
            contig(
                ncols_i32, nrows_i32,
                x_ptr, y_ptr, dst_ptr,
                scratch.as_raw(), scratch.bytes(), stream,
            )
        }
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

/// Per-format MMVQ wrapper macro. `$w_align` is the alignment
/// requirement for the W byte offset (16 for Q4_K, 1 otherwise).
macro_rules! gguf_mmvq {
    ($name:ident, $sys_stem:ident, $w_align:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda GGUF `", $op_label, "` MMVQ — fused dequant + matrix-vector multiply.")]
            pub fn $name(
                weights: &CudaStorageBytes,
                activations: &CudaStorageBytes,
                act_layout: Option<&Layout>,
                w_start_byte_offset: i64,
                ncols: usize,
                nrows: usize,
            ) -> Result<CudaStorageBytes> {
                mmvq_run(
                    weights,
                    activations,
                    act_layout,
                    w_start_byte_offset,
                    $w_align,
                    ncols,
                    nrows,
                    sys::[<baracuda_kernels_mmvq_ $sys_stem _run>],
                    sys::[<baracuda_kernels_mmvq_ $sys_stem _actstrided_run>],
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

// MMVQ — all 11 formats. Alignment defaults to 1 except Q4_K which
// the baracuda alpha.31 actstrided kernel requires to be 16-byte
// aligned at the W slab offset.
gguf_mmvq!(mmvq_q4_0, q4_0, 1,  "mmvq_q4_0");
gguf_mmvq!(mmvq_q4_1, q4_1, 1,  "mmvq_q4_1");
gguf_mmvq!(mmvq_q5_0, q5_0, 1,  "mmvq_q5_0");
gguf_mmvq!(mmvq_q5_1, q5_1, 1,  "mmvq_q5_1");
gguf_mmvq!(mmvq_q8_0, q8_0, 1,  "mmvq_q8_0");
gguf_mmvq!(mmvq_q2_k, q2_K, 1,  "mmvq_q2_K");
gguf_mmvq!(mmvq_q3_k, q3_K, 1,  "mmvq_q3_K");
gguf_mmvq!(mmvq_q4_k, q4_K, 16, "mmvq_q4_K");
gguf_mmvq!(mmvq_q5_k, q5_K, 1,  "mmvq_q5_K");
gguf_mmvq!(mmvq_q6_k, q6_K, 1,  "mmvq_q6_K");
gguf_mmvq!(mmvq_q8_k, q8_K, 1,  "mmvq_q8_K");

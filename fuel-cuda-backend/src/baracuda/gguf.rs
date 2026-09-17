// SPDX-License-Identifier: MIT OR Apache-2.0
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
//! ## MMVQ argument validation (GAP-331)
//!
//! `baracuda-kernels-sys` checks neither the W offset's alignment nor
//! either operand's extent: its FFI takes no buffer length, and the
//! launcher adds `w_start_byte_offset` to the W pointer and launches.
//! So `validate_mmvq_extent` is the only check between a caller's
//! arguments and a device read. It runs first in `mmvq_run`, before
//! any allocation or launch, and every failure is a typed `Err`:
//!
//! - **Offset alignment.** The offset must be a multiple of the block
//!   struct's natural alignment: 2 for Q4_0/Q5_0/Q8_0/Q3_K/Q6_K, and
//!   4 for Q4_1/Q5_1/Q2_K/Q4_K/Q5_K/Q8_K. The table is copied from
//!   baracuda's safe layer (`baracuda-kernels 0.0.1-alpha.79`,
//!   `src/quantize/gguf/mmvq.rs`, `required_alignment`), which is
//!   private and so cannot be called. baracuda enforces it in debug
//!   builds only, and says a misaligned offset reads misaligned
//!   blocks. **Re-read that table at every baracuda bump.** This
//!   replaces an older "Q4_K needs 16" rule cited to alpha.31, which
//!   was stricter than the current table.
//! - **Column read chunk.** The type-0/1 kernels (Q4_0..Q8_0) read
//!   columns in chunks of 64 (`2 * GGML_CUDA_DMMV_X` with
//!   `WARP_SIZE = 32`, in `kernels/include/baracuda_gguf.cuh`), while
//!   their launchers only require `ncols % 32 == 0`. When
//!   `ncols % 64 == 32`, the last chunk reads 32 columns past `ncols`:
//!   the next row's blocks, past the end of W on the last row, and past
//!   the activation. ggml avoids this with zero padding, and nothing
//!   here guarantees any. So type-0/1 require `ncols % 64 == 0`, and
//!   the k-quants require `ncols % 256 == 0`. This is read from the
//!   alpha.79 kernel source and not probed, because probing it is an
//!   out-of-bounds device read (baracuda#127).
//! - **W extent.** `offset + nrows * (ncols / block) * block_bytes`
//!   must fit in the weights' `len_bytes`, using checked arithmetic.
//! - **Activation extent.** The kernel reads `y[i * stride_y]` for
//!   `i < ncols`, starting from the buffer base. So a view with a
//!   non-zero `start_offset` is declined (it would read the wrong
//!   elements), a negative stride over more than one column is
//!   declined (it would read before the base), and
//!   `(ncols - 1) * stride_y + 1` f32s must fit in `len_bytes`.
//!
//! `len_bytes` itself is trusted, per the contract of
//! [`CudaStorageBytes::from_parts`].
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
use fuel_ir::quantized::GgmlDType;
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
    let stream = device.stream().as_raw();
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
fn activation_stride(act_layout: Option<&Layout>, op_label: &'static str) -> Result<i64> {
    let Some(layout) = act_layout else {
        return Ok(1);
    };
    let strides = layout.stride();
    match strides.len() {
        0 => Err(Error::Msg(format!("{op_label}: rank-0 activation not supported")).bt()),
        1 => Ok(strides[0] as i64),
        n => Err(Error::Msg(format!(
            "{op_label}: MMVQ activation must be rank-1 (got rank {n})",
        ))
        .bt()),
    }
}

/// What `validate_mmvq_extent` needs to know about one MMVQ block format.
#[derive(Clone, Copy, Debug)]
struct MmvqFormat {
    /// The W offset must be a multiple of this (the block struct's
    /// natural alignment, per baracuda's `required_alignment`).
    w_align_bytes: usize,
    /// Elements per block (32 for type-0/1, 256 for k-quants).
    block_elems: usize,
    /// Bytes per block.
    block_bytes: usize,
    /// The kernel reads columns in chunks of this many, so `ncols` must
    /// be a multiple of it (64 for type-0/1, 256 for k-quants).
    read_chunk: usize,
}

/// The MMVQ facts for `dtype`, or `Err` for a dtype baracuda has no
/// MMVQ kernel for. Alignment is baracuda's `required_alignment` table
/// (alpha.79); the read chunk is 64 for type-0/1 and 256 for k-quants.
/// No wildcard arm, so a new `GgmlDType` has to be placed here on purpose.
fn mmvq_format(dtype: GgmlDType) -> Result<MmvqFormat> {
    let (w_align_bytes, read_chunk) = match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q5_0 | GgmlDType::Q8_0 => (2, 64),
        GgmlDType::Q4_1 | GgmlDType::Q5_1 => (4, 64),
        GgmlDType::Q3K | GgmlDType::Q6K => (2, 256),
        GgmlDType::Q2K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q8K => (4, 256),
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8_1 => {
            return Err(Error::Msg(format!("{dtype:?} has no baracuda MMVQ kernel")).bt());
        }
    };
    Ok(MmvqFormat {
        w_align_bytes,
        block_elems: dtype.block_size(),
        block_bytes: dtype.type_size(),
        read_chunk,
    })
}

/// The caller-supplied arguments that `validate_mmvq_extent` checks.
#[derive(Clone, Copy, Debug)]
struct MmvqCall {
    w_len_bytes: usize,
    w_start_byte_offset: i64,
    act_len_bytes: usize,
    act_start_offset: usize,
    stride_y: i64,
    ncols: usize,
    nrows: usize,
}

fn mmvq_err(op_label: &str, msg: String) -> Error {
    Error::Msg(format!("{op_label}: {msg}")).bt()
}

fn mmvq_overflow(op_label: &str) -> Error {
    mmvq_err(
        op_label,
        "W or activation extent overflows usize".to_string(),
    )
}

/// GAP-331: check every argument the MMVQ kernel trusts, before any
/// allocation or launch. The module docs list the predicates and where
/// each one comes from. Host-only, so each predicate is unit-tested
/// without a device.
fn validate_mmvq_extent(op_label: &'static str, fmt: MmvqFormat, call: MmvqCall) -> Result<()> {
    let offset = checked_w_offset(op_label, fmt, call.w_start_byte_offset)?;
    check_mmvq_shape(op_label, fmt, &call)?;
    check_act_extent(op_label, &call)?;
    check_w_extent(op_label, fmt, &call, offset)
}

/// The W offset: not negative, and a multiple of the format's alignment.
fn checked_w_offset(op_label: &str, fmt: MmvqFormat, w_start_byte_offset: i64) -> Result<usize> {
    let offset = usize::try_from(w_start_byte_offset).map_err(|_| {
        mmvq_err(
            op_label,
            format!("negative w_start_byte_offset {w_start_byte_offset}"),
        )
    })?;
    if !offset.is_multiple_of(fmt.w_align_bytes) {
        return Err(mmvq_err(
            op_label,
            format!(
                "w_start_byte_offset ({offset}) is not a multiple of {} bytes, this block \
                 format's alignment; the kernel would read misaligned blocks",
                fmt.w_align_bytes,
            ),
        ));
    }
    Ok(offset)
}

/// Shape: the kernel's column read chunk, a start offset the kernel would
/// ignore, and a stride that would read before the buffer.
fn check_mmvq_shape(op_label: &str, fmt: MmvqFormat, call: &MmvqCall) -> Result<()> {
    if !call.ncols.is_multiple_of(fmt.read_chunk) {
        return Err(mmvq_err(
            op_label,
            format!(
                "ncols ({}) is not a multiple of {}, the kernel's column read chunk; \
                 the kernel would read past ncols",
                call.ncols, fmt.read_chunk,
            ),
        ));
    }
    if call.act_start_offset != 0 {
        return Err(mmvq_err(
            op_label,
            format!(
                "activation view has start_offset {}; the kernel reads from the buffer \
                 base, so it would read the wrong elements",
                call.act_start_offset,
            ),
        ));
    }
    if call.stride_y < 0 && call.ncols > 1 {
        return Err(mmvq_err(
            op_label,
            format!(
                "negative activation stride {}; the kernel would read before the buffer base",
                call.stride_y,
            ),
        ));
    }
    Ok(())
}

/// The activation read: `y[i * stride_y]` for `i < ncols`, from the buffer base.
fn check_act_extent(op_label: &str, call: &MmvqCall) -> Result<()> {
    let elems = match (call.ncols, call.stride_y) {
        (0, _) => Some(0),
        (_, s) if s <= 0 => Some(1),
        (n, s) => usize::try_from(s)
            .ok()
            .and_then(|s| (n - 1).checked_mul(s))
            .and_then(|last| last.checked_add(1)),
    };
    let need = elems
        .and_then(|e| e.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| mmvq_overflow(op_label))?;
    if need > call.act_len_bytes {
        return Err(mmvq_err(
            op_label,
            format!(
                "activation read needs {need} bytes (ncols {}, stride {}) but the buffer \
                 holds {}",
                call.ncols, call.stride_y, call.act_len_bytes,
            ),
        ));
    }
    Ok(())
}

/// The W read: `nrows` rows of `ncols / block_elems` blocks, from `offset`.
fn check_w_extent(op_label: &str, fmt: MmvqFormat, call: &MmvqCall, offset: usize) -> Result<()> {
    let need = call
        .ncols
        .div_ceil(fmt.block_elems)
        .checked_mul(call.nrows)
        .and_then(|blocks| blocks.checked_mul(fmt.block_bytes))
        .and_then(|bytes| bytes.checked_add(offset))
        .ok_or_else(|| mmvq_overflow(op_label))?;
    if need > call.w_len_bytes {
        return Err(mmvq_err(
            op_label,
            format!(
                "W read needs {need} bytes (offset {offset}, {} rows x {} cols) but the \
                 buffer holds {}",
                call.nrows, call.ncols, call.w_len_bytes,
            ),
        ));
    }
    Ok(())
}

/// MMVQ — fused dequant + matrix-vector multiply.
/// `weights` is `[nrows, ncols]` packed in the block format;
/// `activations` is `[ncols]` f32; output is `[nrows]` f32.
///
/// Picks contig vs actstrided per-call:
/// - Both `stride_y == 1` AND `w_start_byte_offset == 0` ⇒ contig FFI.
/// - Otherwise ⇒ actstrided FFI (alpha.31).
///
/// Every argument the kernel trusts is checked by
/// `validate_mmvq_extent` before anything is allocated or launched.
#[allow(clippy::too_many_arguments)]
fn mmvq_run(
    weights: &CudaStorageBytes,
    activations: &CudaStorageBytes,
    act_layout: Option<&Layout>,
    w_start_byte_offset: i64,
    dtype: GgmlDType,
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
        ))
        .bt());
    }
    let stride_y = activation_stride(act_layout, op_label)?;
    validate_mmvq_extent(
        op_label,
        mmvq_format(dtype)?,
        MmvqCall {
            w_len_bytes: weights.len_bytes(),
            w_start_byte_offset,
            act_len_bytes: activations.len_bytes(),
            act_start_offset: act_layout.map_or(0, Layout::start_offset),
            stride_y,
            ncols,
            nrows,
        },
    )?;

    let ncols_i32 = i32::try_from(ncols).map_err(|_| {
        Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label,
            dim_index: 0,
            dim_value: ncols,
        })
    })?;
    let nrows_i32 = i32::try_from(nrows).map_err(|_| {
        Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
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
    let stream = device.stream().as_raw();
    let x_ptr = weights.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = activations.buffer().as_raw().0 as *const std::ffi::c_void;
    let dst_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    let take_strided = stride_y != 1 || w_start_byte_offset != 0;

    // SAFETY (both arms): the two operands are on one device (checked
    // above). `validate_mmvq_extent` has checked, against each buffer's
    // `len_bytes`, that W's read extent and the activation's read extent
    // (from the buffer base) are in bounds, that the offset meets the
    // format's alignment, and that `ncols` is a multiple of the kernel's
    // column read chunk. `dst` is a fresh `nrows`-f32 buffer. Nothing
    // checks `len_bytes` against the real allocation; that is
    // `CudaStorageBytes::from_parts`'s contract.
    let status = if take_strided {
        // SAFETY: see above; the FFI adds the offset and applies the stride.
        unsafe {
            strided(
                ncols_i32,
                nrows_i32,
                x_ptr,
                w_start_byte_offset,
                stride_y,
                y_ptr,
                dst_ptr,
                scratch.as_raw(),
                scratch.bytes(),
                stream,
            )
        }
    } else {
        // SAFETY: see above; offset 0 and stride 1, so the contig kernel reads
        // exactly the validated extents.
        unsafe {
            contig(
                ncols_i32,
                nrows_i32,
                x_ptr,
                y_ptr,
                dst_ptr,
                scratch.as_raw(),
                scratch.bytes(),
                stream,
            )
        }
    };
    check(status, op_label)?;
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

/// Per-format MMVQ wrapper macro. `$dtype` names the block format;
/// `mmvq_format` turns it into the facts the argument checks need.
macro_rules! gguf_mmvq {
    ($name:ident, $sys_stem:ident, $dtype:ident, $op_label:expr $(,)?) => {
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
                    GgmlDType::$dtype,
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

// MMVQ — all 11 formats. Per-format checks come from `mmvq_format`.
gguf_mmvq!(mmvq_q4_0, q4_0, Q4_0, "mmvq_q4_0");
gguf_mmvq!(mmvq_q4_1, q4_1, Q4_1, "mmvq_q4_1");
gguf_mmvq!(mmvq_q5_0, q5_0, Q5_0, "mmvq_q5_0");
gguf_mmvq!(mmvq_q5_1, q5_1, Q5_1, "mmvq_q5_1");
gguf_mmvq!(mmvq_q8_0, q8_0, Q8_0, "mmvq_q8_0");
gguf_mmvq!(mmvq_q2_k, q2_K, Q2K, "mmvq_q2_K");
gguf_mmvq!(mmvq_q3_k, q3_K, Q3K, "mmvq_q3_K");
gguf_mmvq!(mmvq_q4_k, q4_K, Q4K, "mmvq_q4_K");
gguf_mmvq!(mmvq_q5_k, q5_K, Q5K, "mmvq_q5_K");
gguf_mmvq!(mmvq_q6_k, q6_K, Q6K, "mmvq_q6_K");
gguf_mmvq!(mmvq_q8_k, q8_K, Q8K, "mmvq_q8_K");

/// GAP-331: the MMVQ argument checks, without a device. Every decline test
/// breaks exactly one predicate and pins that predicate's message, so a
/// different check cannot satisfy it; each has an accepted neighbour, so an
/// over-rejecting check goes red too.
#[cfg(test)]
mod tests {
    use super::*;

    const F32: usize = std::mem::size_of::<f32>();

    #[derive(Clone, Copy)]
    struct Call {
        fmt: MmvqFormat,
        w_len: usize,
        offset: i64,
        a_len: usize,
        a_start: usize,
        stride: i64,
        ncols: usize,
        nrows: usize,
    }

    impl Call {
        fn run(self) -> Result<()> {
            validate_mmvq_extent(
                "mmvq_test",
                self.fmt,
                MmvqCall {
                    w_len_bytes: self.w_len,
                    w_start_byte_offset: self.offset,
                    act_len_bytes: self.a_len,
                    act_start_offset: self.a_start,
                    stride_y: self.stride,
                    ncols: self.ncols,
                    nrows: self.nrows,
                },
            )
        }

        #[track_caller]
        fn accepted(self) {
            if let Err(e) = self.run() {
                panic!("expected Ok, got: {e}");
            }
        }

        #[track_caller]
        fn declined_with(self, needle: &str) {
            let msg = self.run().expect_err("expected a decline").to_string();
            assert!(msg.contains(needle), "wrong decline: {msg}");
        }
    }

    fn fmt(dtype: GgmlDType) -> MmvqFormat {
        mmvq_format(dtype).expect("an MMVQ format")
    }

    /// Q4_0, 3 rows of 64 cols (2 blocks of 18 bytes each), buffers exactly full.
    fn q4_0_call() -> Call {
        Call {
            fmt: fmt(GgmlDType::Q4_0),
            w_len: 3 * 2 * 18,
            offset: 0,
            a_len: 64 * F32,
            a_start: 0,
            stride: 1,
            ncols: 64,
            nrows: 3,
        }
    }

    /// Q4_K, 1 row of 256 cols (1 block of 144 bytes), buffers exactly full.
    fn q4_k_call() -> Call {
        Call {
            fmt: fmt(GgmlDType::Q4K),
            w_len: 144,
            offset: 0,
            a_len: 256 * F32,
            a_start: 0,
            stride: 1,
            ncols: 256,
            nrows: 1,
        }
    }

    #[test]
    fn exactly_full_buffers_are_accepted() {
        q4_0_call().accepted();
        q4_k_call().accepted();
    }

    /// Pins the table against baracuda-kernels 0.0.1-alpha.79's
    /// `required_alignment` and the kernels' read chunks. Re-check it at every
    /// baracuda bump.
    #[test]
    fn mmvq_format_table_matches_baracuda() {
        // (dtype, alignment, read chunk, block elems, block bytes)
        let want = [
            (GgmlDType::Q4_0, 2, 64, 32, 18),
            (GgmlDType::Q4_1, 4, 64, 32, 20),
            (GgmlDType::Q5_0, 2, 64, 32, 22),
            (GgmlDType::Q5_1, 4, 64, 32, 24),
            (GgmlDType::Q8_0, 2, 64, 32, 34),
            (GgmlDType::Q2K, 4, 256, 256, 84),
            (GgmlDType::Q3K, 2, 256, 256, 110),
            (GgmlDType::Q4K, 4, 256, 256, 144),
            (GgmlDType::Q5K, 4, 256, 256, 176),
            (GgmlDType::Q6K, 2, 256, 256, 210),
            (GgmlDType::Q8K, 4, 256, 256, 292),
        ];
        for (dtype, align, chunk, elems, bytes) in want {
            let f = fmt(dtype);
            assert_eq!(
                (f.w_align_bytes, f.read_chunk, f.block_elems, f.block_bytes),
                (align, chunk, elems, bytes),
                "{dtype:?}",
            );
        }
    }

    #[test]
    fn dtypes_without_an_mmvq_kernel_are_declined() {
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::Q8_1,
        ] {
            let msg = mmvq_format(dtype).expect_err("no MMVQ kernel").to_string();
            assert!(
                msg.contains("has no baracuda MMVQ kernel"),
                "{dtype:?}: {msg}"
            );
        }
    }

    #[test]
    fn negative_offset_is_declined() {
        Call {
            offset: -2,
            ..q4_0_call()
        }
        .declined_with("negative w_start_byte_offset -2");
    }

    #[test]
    fn misaligned_offset_is_declined() {
        let base = q4_0_call();
        Call {
            offset: 1,
            w_len: base.w_len + 1,
            ..base
        }
        .declined_with("w_start_byte_offset (1) is not a multiple of 2 bytes");
        Call {
            offset: 2,
            w_len: base.w_len + 2,
            ..base
        }
        .accepted();
    }

    /// The old rule required 16 for Q4_K; baracuda's table says 4.
    #[test]
    fn q4_k_offset_needs_alignment_4_not_16() {
        let base = q4_k_call();
        Call {
            offset: 4,
            w_len: base.w_len + 4,
            ..base
        }
        .accepted();
        Call {
            offset: 2,
            w_len: base.w_len + 2,
            ..base
        }
        .declined_with("w_start_byte_offset (2) is not a multiple of 4 bytes");
    }

    #[test]
    fn type01_ncols_must_be_a_multiple_of_64() {
        let base = q4_0_call();
        // Buffers sized for the shape, so only the read-chunk check can fire.
        for ncols in [32, 96] {
            Call {
                ncols,
                w_len: base.nrows * (ncols / 32) * 18,
                a_len: ncols * F32,
                ..base
            }
            .declined_with(&format!(
                "ncols ({ncols}) is not a multiple of 64, the kernel's column read chunk"
            ));
        }
        Call {
            ncols: 128,
            w_len: base.nrows * 4 * 18,
            a_len: 128 * F32,
            ..base
        }
        .accepted();
    }

    #[test]
    fn k_quant_ncols_must_be_a_multiple_of_256() {
        let base = q4_k_call();
        Call {
            ncols: 128,
            w_len: 144,
            a_len: 128 * F32,
            ..base
        }
        .declined_with("ncols (128) is not a multiple of 256");
    }

    #[test]
    fn activation_view_with_a_start_offset_is_declined() {
        let base = q4_0_call();
        Call {
            a_start: 4,
            a_len: base.a_len + 4 * F32,
            ..base
        }
        .declined_with("activation view has start_offset 4");
    }

    #[test]
    fn negative_stride_over_several_columns_is_declined() {
        Call {
            stride: -1,
            ..q4_0_call()
        }
        .declined_with("negative activation stride -1");
    }

    #[test]
    fn activation_one_float_short_is_declined() {
        let base = q4_0_call();
        Call {
            a_len: base.a_len - F32,
            ..base
        }
        .declined_with(
            "activation read needs 256 bytes (ncols 64, stride 1) but the buffer holds 252",
        );
    }

    /// A strided read needs `(ncols - 1) * stride + 1` elements, not `ncols * stride`.
    #[test]
    fn strided_activation_needs_last_index_plus_one() {
        let base = q4_0_call();
        Call {
            stride: 2,
            a_len: 127 * F32,
            ..base
        }
        .accepted();
        Call {
            stride: 2,
            a_len: 126 * F32,
            ..base
        }
        .declined_with("activation read needs 508 bytes (ncols 64, stride 2)");
    }

    #[test]
    fn stride_zero_reads_one_element() {
        let base = q4_0_call();
        Call {
            stride: 0,
            a_len: F32,
            ..base
        }
        .accepted();
        Call {
            stride: 0,
            a_len: 0,
            ..base
        }
        .declined_with("activation read needs 4 bytes");
    }

    #[test]
    fn weights_one_byte_short_is_declined() {
        let base = q4_0_call();
        Call {
            w_len: base.w_len - 1,
            ..base
        }
        .declined_with(
            "W read needs 108 bytes (offset 0, 3 rows x 64 cols) but the buffer holds 107",
        );
    }

    #[test]
    fn offset_counts_toward_the_weights_extent() {
        let base = q4_0_call();
        Call { offset: 2, ..base }.declined_with("W read needs 110 bytes (offset 2,");
        Call {
            offset: 2,
            w_len: base.w_len + 2,
            ..base
        }
        .accepted();
    }

    #[test]
    fn extent_overflow_is_a_typed_error() {
        let base = q4_0_call();
        Call {
            nrows: usize::MAX,
            ..base
        }
        .declined_with("extent overflows usize");
        Call {
            stride: i64::MAX,
            ..base
        }
        .declined_with("extent overflows usize");
    }
}

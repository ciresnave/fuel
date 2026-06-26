//! Fused Linear Cross-Entropy (FLCE) primitives from baracuda
//! alpha.58 (Phase 47, Liger-Kernel algorithm port — BSD-2-Clause
//! credit, clean-room CUDA reimplementation).
//!
//! Fuses the `Linear → LogSoftmax → NLL` chain into one in-place
//! pass over `logits: [n_rows, V]`:
//!
//!   1. **per-row** computes `grad_logits = (softmax - one_hot) ·
//!      scale_per_row` in place AND writes per-row
//!      `-log_softmax[target]` into `loss_1d: f32[n_rows]`. Targets
//!      equal to `ignore_index` skip both writes.
//!   2. **cast** converts the `loss_1d` f32 accumulator to the
//!      caller's output dtype unchanged (None reduction mode).
//!   3. **scalar_finalize** sums `loss_1d` with `denom_inv` scale
//!      into a scalar (Mean / Sum reduction modes — caller picks
//!      `denom_inv = 1/count_non_ignore` for Mean or `1.0` for Sum).
//!   4. **inplace_scale** multiplies an arbitrary buffer (intended
//!      for `grad_logits`) in place by a scalar — used by the
//!      gradient renormalization step when the loss reduction
//!      scales the upstream `dy`.
//!   5. **count_non_ignore** single-block tree reduction that
//!      counts `target[i] != ignore_index` into a scalar i64 —
//!      feeds `denom_inv` for Mean mode.
//!
//! Algorithm vs. the unfused path:
//!   logits is mutated in place to grad_logits (saves a buffer
//!   pass + materialization), and the softmax is computed implicitly
//!   inside per_row (no LogSoftmax tensor materialization).
//!
//! All FFI take `n_rows` and `numel` as `i64`; row stride feeds
//! per_row to support strided `logits`. `target` is `i64` indices.

use baracuda_kernels_sys as sys;
use fuel_ir::Result;

use crate::byte_storage::CudaStorageBytes;

use super::status::check;

// ─────────────────────────── per-row ───────────────────────────

/// FLCE per-row fused step, F32 logits.
///
/// **Mutates `logits` in place** to `grad_logits = (softmax -
/// one_hot) · scale_per_row`. Writes per-row `-log_softmax[target]`
/// into `loss_1d: f32[n_rows]`. `target` is `i64[n_rows]`; rows
/// where `target[i] == target_ignore` skip both writes.
///
/// `row_stride` is in elements (the i64 stride from row `i` to
/// `i+1` in `logits`); for a contig `[n_rows, V]` it's just `V`.
/// `scale_per_row` matches PyTorch's gradient scaling — typically
/// `1.0` (None reduction) or `1.0 / count_non_ignore` (Mean).
#[allow(clippy::too_many_arguments)]
pub fn per_row_f32(
    logits: &CudaStorageBytes,        // mutated in place
    target: &CudaStorageBytes,        // i64
    loss_1d: &CudaStorageBytes,       // f32, n_rows
    n_rows: i32,
    v: i32,
    row_stride: i64,
    target_ignore: i64,
    scale_per_row: f32,
) -> Result<()> {
    per_row_inner(
        logits, target, loss_1d, n_rows, v, row_stride, target_ignore, scale_per_row,
        sys::baracuda_kernels_loss_flce_per_row_f32_run,
        "loss_flce_per_row_f32",
    )
}

/// FLCE per-row, F16 logits.
#[allow(clippy::too_many_arguments)]
pub fn per_row_f16(
    logits: &CudaStorageBytes, target: &CudaStorageBytes, loss_1d: &CudaStorageBytes,
    n_rows: i32, v: i32, row_stride: i64, target_ignore: i64, scale_per_row: f32,
) -> Result<()> {
    per_row_inner(
        logits, target, loss_1d, n_rows, v, row_stride, target_ignore, scale_per_row,
        sys::baracuda_kernels_loss_flce_per_row_f16_run,
        "loss_flce_per_row_f16",
    )
}

/// FLCE per-row, BF16 logits.
#[allow(clippy::too_many_arguments)]
pub fn per_row_bf16(
    logits: &CudaStorageBytes, target: &CudaStorageBytes, loss_1d: &CudaStorageBytes,
    n_rows: i32, v: i32, row_stride: i64, target_ignore: i64, scale_per_row: f32,
) -> Result<()> {
    per_row_inner(
        logits, target, loss_1d, n_rows, v, row_stride, target_ignore, scale_per_row,
        sys::baracuda_kernels_loss_flce_per_row_bf16_run,
        "loss_flce_per_row_bf16",
    )
}

/// FLCE per-row, F64 logits.
#[allow(clippy::too_many_arguments)]
pub fn per_row_f64(
    logits: &CudaStorageBytes, target: &CudaStorageBytes, loss_1d: &CudaStorageBytes,
    n_rows: i32, v: i32, row_stride: i64, target_ignore: i64, scale_per_row: f32,
) -> Result<()> {
    per_row_inner(
        logits, target, loss_1d, n_rows, v, row_stride, target_ignore, scale_per_row,
        sys::baracuda_kernels_loss_flce_per_row_f64_run,
        "loss_flce_per_row_f64",
    )
}

type PerRowRun = unsafe extern "C" fn(
    n_rows: i32, v: i32, row_stride: i64, target_ignore: i64,
    scale_per_row: f32,
    logits: *mut std::ffi::c_void,
    target: *const std::ffi::c_void,
    loss_1d: *mut std::ffi::c_void,
    stream: *mut std::ffi::c_void,
) -> i32;

#[allow(clippy::too_many_arguments)]
fn per_row_inner(
    logits: &CudaStorageBytes,
    target: &CudaStorageBytes,
    loss_1d: &CudaStorageBytes,
    n_rows: i32,
    v: i32,
    row_stride: i64,
    target_ignore: i64,
    scale_per_row: f32,
    kernel: PerRowRun,
    op_label: &'static str,
) -> Result<()> {
    let device = logits.device().clone();
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let status = unsafe {
        kernel(
            n_rows, v, row_stride, target_ignore,
            scale_per_row,
            logits.buffer().as_raw().0  as *mut std::ffi::c_void,
            target.buffer().as_raw().0  as *const std::ffi::c_void,
            loss_1d.buffer().as_raw().0 as *mut std::ffi::c_void,
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(())
}

// ─────────────────────────── per-row cast ───────────────────────────
//
// f32 → T finalizer for `reduction = None`. Copies `loss_1d` to
// `out` casting to T (T ∈ {f32, f16, bf16, f64}).

macro_rules! per_row_cast {
    ($name:ident, $sys:ident, $label:expr) => {
        #[doc = concat!("FLCE per-row cast (None reduction): f32 → ", $label, ".")]
        pub fn $name(
            loss_1d: &CudaStorageBytes,
            out: &CudaStorageBytes,
            n_rows: i64,
        ) -> Result<()> {
            let device = loss_1d.device().clone();
            let stream = device.stream().as_raw() as *mut std::ffi::c_void;
            let status = unsafe {
                sys::$sys(
                    n_rows,
                    loss_1d.buffer().as_raw().0 as *const std::ffi::c_void,
                    out.buffer().as_raw().0     as *mut std::ffi::c_void,
                    stream,
                )
            };
            check(status, stringify!($name))?;
            device.synchronize()?;
            Ok(())
        }
    };
}

per_row_cast!(per_row_cast_f32,  baracuda_kernels_loss_flce_per_row_cast_f32_run,  "f32");
per_row_cast!(per_row_cast_f16,  baracuda_kernels_loss_flce_per_row_cast_f16_run,  "f16");
per_row_cast!(per_row_cast_bf16, baracuda_kernels_loss_flce_per_row_cast_bf16_run, "bf16");
per_row_cast!(per_row_cast_f64,  baracuda_kernels_loss_flce_per_row_cast_f64_run,  "f64");

// ─────────────────────────── scalar finalize ───────────────────────────
//
// Mean / Sum reduction. `denom_inv = 1.0 / count_non_ignore` for
// Mean; `denom_inv = 1.0` for Sum. Writes the scalar into out[0]
// (caller chooses out's dtype).

macro_rules! scalar_finalize {
    ($name:ident, $sys:ident, $label:expr) => {
        #[doc = concat!("FLCE scalar finalize (Mean/Sum): f32 → ", $label, ".")]
        pub fn $name(
            loss_1d: &CudaStorageBytes,
            out: &CudaStorageBytes,
            n_rows: i64,
            denom_inv: f32,
        ) -> Result<()> {
            let device = loss_1d.device().clone();
            let stream = device.stream().as_raw() as *mut std::ffi::c_void;
            let status = unsafe {
                sys::$sys(
                    n_rows, denom_inv,
                    loss_1d.buffer().as_raw().0 as *const std::ffi::c_void,
                    out.buffer().as_raw().0     as *mut std::ffi::c_void,
                    stream,
                )
            };
            check(status, stringify!($name))?;
            device.synchronize()?;
            Ok(())
        }
    };
}

scalar_finalize!(scalar_finalize_f32,  baracuda_kernels_loss_flce_scalar_finalize_f32_run,  "f32");
scalar_finalize!(scalar_finalize_f16,  baracuda_kernels_loss_flce_scalar_finalize_f16_run,  "f16");
scalar_finalize!(scalar_finalize_bf16, baracuda_kernels_loss_flce_scalar_finalize_bf16_run, "bf16");
scalar_finalize!(scalar_finalize_f64,  baracuda_kernels_loss_flce_scalar_finalize_f64_run,  "f64");

// ─────────────────────────── inplace_scale ───────────────────────────
//
// `buf[i] *= scalar`. Used to renormalize `grad_logits` when the
// loss reduction scales the upstream `dy`. T ∈ {f32, f16, bf16, f64}.

macro_rules! inplace_scale {
    ($name:ident, $sys:ident, $label:expr) => {
        #[doc = concat!("FLCE in-place scale ", $label, ": `buf[i] *= scalar` over `numel` elements.")]
        pub fn $name(
            buf: &CudaStorageBytes,
            numel: i64,
            scalar: f32,
        ) -> Result<()> {
            let device = buf.device().clone();
            let stream = device.stream().as_raw() as *mut std::ffi::c_void;
            let status = unsafe {
                sys::$sys(
                    numel, scalar,
                    buf.buffer().as_raw().0 as *mut std::ffi::c_void,
                    stream,
                )
            };
            check(status, stringify!($name))?;
            device.synchronize()?;
            Ok(())
        }
    };
}

inplace_scale!(inplace_scale_f32,  baracuda_kernels_loss_flce_inplace_scale_f32_run,  "f32");
inplace_scale!(inplace_scale_f16,  baracuda_kernels_loss_flce_inplace_scale_f16_run,  "f16");
inplace_scale!(inplace_scale_bf16, baracuda_kernels_loss_flce_inplace_scale_bf16_run, "bf16");
inplace_scale!(inplace_scale_f64,  baracuda_kernels_loss_flce_inplace_scale_f64_run,  "f64");

// ─────────────────────────── count_non_ignore ───────────────────────────

/// FLCE count-non-ignore. Single-block tree reduction. Writes the
/// count of `target[i] != ignore_index` into `count_out[0]` as an
/// `i64`. `target` is `i64[bt]`. Feeds `denom_inv = 1 / count` for
/// Mean reduction.
pub fn count_non_ignore(
    target: &CudaStorageBytes,
    count_out: &CudaStorageBytes,
    bt: i32,
    ignore_index: i64,
) -> Result<()> {
    let device = target.device().clone();
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let status = unsafe {
        sys::baracuda_kernels_loss_flce_count_non_ignore_run(
            bt, ignore_index,
            target.buffer().as_raw().0    as *const std::ffi::c_void,
            count_out.buffer().as_raw().0 as *mut std::ffi::c_void,
            stream,
        )
    };
    check(status, "loss_flce_count_non_ignore")?;
    device.synchronize()?;
    Ok(())
}

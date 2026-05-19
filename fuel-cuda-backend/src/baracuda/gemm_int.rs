//! Integer GEMM kernels from `baracuda-kernels-sys` — Phase 1 of the
//! W8A8 quantization path. Identity epilogue only this session;
//! Bias / BiasRelu / BiasGelu / BiasSilu variants are deferred until
//! Fuel grows a `FusedLinearInt` op (or extends FusedLinear to carry
//! integer dtypes).
//!
//! ## SKUs
//!
//! - `gemm_s8_rrr_sm80_run` — `i8 @ i8 → i8`, RRR layout, sm_80
//!   tensor cores. Accumulator: `int32` via
//!   `mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32`;
//!   epilogue is `α · accum + β · C` in f32 with saturating-cast back
//!   to s8 on store.
//! - `gemm_u8_rrr_sm80_run` — `u8 @ u8 → u8`, same shape, MMA encoding
//!   `.u8.u8`.
//!
//! ## Layout convention (`RRR`)
//!
//! - A: row-major `[M, K]`, `lda = K`
//! - B: row-major `[K, N]`, `ldb = N`
//! - C: row-major `[M, N]`, `ldc = N` (optional — pass null + `β = 0`
//!   to skip; this wrapper always does)
//! - D: row-major `[M, N]`, `ldd = N` (always written)
//!
//! Matches `Op::MatMul`'s natural shape exactly; no transposition pass
//! needed (unlike the f32 cuBLAS path which views row-major as
//! col-major and swaps operands).
//!
//! ## Batching
//!
//! Today: non-batched only. `lhs_batch_dims` / `rhs_batch_dims` must
//! both be empty (or product to 1). Batched int-GEMM is a loop over
//! `gemm_s8_rrr_sm80_run` calls; that lands when a W8A8 transformer
//! workload exercises it. The common quantized-Linear shape is rank-2
//! after the standard `[batch * seq, hidden]` flatten, so v1 is enough
//! for inference-bench-style usage.
//!
//! ## Defaults
//!
//! `alpha = 1.0`, `beta = 0.0` — plain GEMM, no `C` accumulation.
//! The baracuda Identity SKU treats β as f32; the int32 accumulator is
//! cast through the (α, β) operations to f32 before being saturating-
//! cast back to the storage int type.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::{Error, Result};

use crate::byte_storage::CudaStorageBytes;

use super::scratch::Workspace;
use super::status::check;

type GemmInt8Run = unsafe extern "C" fn(
    m: i32,
    n: i32,
    k: i32,
    a: *const std::ffi::c_void,
    lda: i64,
    b: *const std::ffi::c_void,
    ldb: i64,
    c: *const std::ffi::c_void,
    ldc: i64,
    d: *mut std::ffi::c_void,
    ldd: i64,
    alpha: f32,
    beta: f32,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

type GemmInt8WorkspaceSize = unsafe extern "C" fn(m: i32, n: i32, k: i32) -> usize;

/// Common driver for s8/u8 RRR Identity. Validates byte counts,
/// rejects batched inputs (v1 limitation), allocates the output, calls
/// the FFI, syncs.
#[allow(clippy::too_many_arguments)]
fn gemm_int8_run(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
    kernel: GemmInt8Run,
    workspace_size: GemmInt8WorkspaceSize,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let lhs_batch_count: usize = lhs_batch_dims.iter().product::<usize>().max(1);
    let rhs_batch_count: usize = rhs_batch_dims.iter().product::<usize>().max(1);
    if lhs_batch_count != 1 || rhs_batch_count != 1 {
        return Err(Error::Msg(format!(
            "{op_label}: batched int8 GEMM not supported yet \
             (lhs_batch={:?}, rhs_batch={:?}); flatten to rank-2 before \
             dispatching, or extend the wrapper to loop over batches",
            lhs_batch_dims, rhs_batch_dims,
        ))
        .bt());
    }

    // i8 / u8 storage is 1 byte/element; numel == byte count.
    let need_lhs = m.saturating_mul(k);
    let need_rhs = k.saturating_mul(n);
    let need_out = m.saturating_mul(n);
    if lhs.len_bytes() != need_lhs {
        return Err(Error::Msg(format!(
            "{op_label}: lhs bytes={} doesn't match [{m}, {k}] (1 byte/elem)",
            lhs.len_bytes(),
        ))
        .bt());
    }
    if rhs.len_bytes() != need_rhs {
        return Err(Error::Msg(format!(
            "{op_label}: rhs bytes={} doesn't match [{k}, {n}] (1 byte/elem)",
            rhs.len_bytes(),
        ))
        .bt());
    }

    let device = lhs.device().clone();
    if rhs.device().id() != device.id() {
        return Err(Error::Msg(format!(
            "{op_label}: lhs and rhs are on different CUDA devices; insert Op::Move first",
        ))
        .bt());
    }
    if need_out == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(need_out)?;

    let m_i32 = i32::try_from(m).map_err(|_| {
        Error::Msg(format!("{op_label}: m={m} exceeds i32 (baracuda's shape dtype)")).bt()
    })?;
    let n_i32 = i32::try_from(n).map_err(|_| {
        Error::Msg(format!("{op_label}: n={n} exceeds i32 (baracuda's shape dtype)")).bt()
    })?;
    let k_i32 = i32::try_from(k).map_err(|_| {
        Error::Msg(format!("{op_label}: k={k} exceeds i32 (baracuda's shape dtype)")).bt()
    })?;

    // Workspace query first so the alloc is right-sized for the
    // (m, n, k) the kernel will see. Today this returns zero across
    // every shape (Identity SKU has no scratch); the alloc still goes
    // through Workspace::alloc which short-circuits on zero bytes.
    let workspace_bytes = unsafe { workspace_size(m_i32, n_i32, k_i32) };
    let scratch = Workspace::alloc(&device, workspace_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let a_ptr = lhs.buffer().as_raw().0 as *const std::ffi::c_void;
    let b_ptr = rhs.buffer().as_raw().0 as *const std::ffi::c_void;
    let d_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;
    // C is null; β = 0 means no accumulation read.
    let c_ptr: *const std::ffi::c_void = std::ptr::null();

    // Leading dimensions are row-major: lda = K, ldb = N, ldc/ldd = N.
    let lda = k as i64;
    let ldb = n as i64;
    let ldc = n as i64;
    let ldd = n as i64;

    // SAFETY: pointers + shape validated; lhs/rhs/out outlive the
    // launch (device.synchronize below); workspace is null/0 (Identity
    // SKU has no scratch in alpha.28); stream borrows device.
    let status = unsafe {
        kernel(
            m_i32,
            n_i32,
            k_i32,
            a_ptr,
            lda,
            b_ptr,
            ldb,
            c_ptr,
            ldc,
            d_ptr,
            ldd,
            1.0_f32, // alpha
            0.0_f32, // beta — skip C accumulation
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
        need_out,
    ))
}

/// Signed-int8 GEMM `D = A @ B` with `A: [M, K] s8 row-major`,
/// `B: [K, N] s8 row-major`, `D: [M, N] s8 row-major`. Tensor cores
/// on sm_80+. Saturating cast on store — values outside `[-128, 127]`
/// clamp instead of wrapping.
pub fn gemm_s8_rrr(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaStorageBytes> {
    gemm_int8_run(
        lhs,
        rhs,
        lhs_batch_dims,
        rhs_batch_dims,
        m,
        n,
        k,
        sys::baracuda_kernels_gemm_s8_rrr_sm80_run,
        sys::baracuda_kernels_gemm_s8_rrr_sm80_workspace_size,
        "gemm_s8_rrr",
    )
}

/// Unsigned-int8 GEMM `D = A @ B` with `A: [M, K] u8 row-major`,
/// `B: [K, N] u8 row-major`, `D: [M, N] u8 row-major`. Tensor cores
/// on sm_80+. Saturating cast on store — values outside `[0, 255]`
/// clamp.
pub fn gemm_u8_rrr(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaStorageBytes> {
    gemm_int8_run(
        lhs,
        rhs,
        lhs_batch_dims,
        rhs_batch_dims,
        m,
        n,
        k,
        sys::baracuda_kernels_gemm_u8_rrr_sm80_run,
        sys::baracuda_kernels_gemm_u8_rrr_sm80_workspace_size,
        "gemm_u8_rrr",
    )
}

//! 4-bit weight-only GEMM families from baracuda alpha.58:
//!
//! - **Marlin** (Phase 48, `marlin` feature) — symmetric int4 W4A16 GEMM.
//!   GPTQ checkpoints reshuffled via `gptq_to_marlin` (host-side, lives
//!   in `baracuda-kernels`). FP16 activation + output. groupsize ∈
//!   {-1, 128}. ~3.87× over FP16 GEMM at batch 1-32 on Ampere/Ada;
//!   NOT sm_90.
//! - **AWQ** (Phase 48, `awq` feature) — asymmetric int4 W4A16 GEMM
//!   with explicit per-group zero-points. Loads HuggingFace `*-AWQ`
//!   checkpoints with no repack. FP16 activation + output, F32 acc.
//!   group_size ∈ {64, 128}, split_k_iters caller-chosen (typ. 8).
//! - **NF4** (Phase 53, `bnb_nf4` feature) — bitsandbytes NormalFloat
//!   4-bit. Block-quantized with per-block FP32 absmax. Dequant +
//!   GEMV at M ∈ {1, 2, 4, 8}; F16/BF16 activations. block_size
//!   typically 64.
//!
//! Each helper is a thin transparent wrapper over the matching
//! `baracuda_kernels_int4_<family>_<dt>_<shape>_run` FFI. Workspace
//! allocation, scratch tracking, and status-code translation match
//! the rest of `baracuda/*` (status `0` ok, `2` invalid, `3`
//! unsupported, `4` workspace too small, `5` launch failure).
//!
//! No Fuel `OpKind` dispatches here yet — these are primitives. Each
//! checkpoint format ships its own loader; once Fuel-side
//! `QuantFormat::{Awq, Marlin, NF4}` storage variants land, the
//! dispatchers call into here.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::Result;

use crate::byte_storage::CudaStorageBytes;

use super::scratch::Workspace;
use super::status::check;

// ───────────────────────────── Marlin ─────────────────────────────

/// Marlin W4A16 symmetric GEMM (FP16 activations, FP16 output).
///
/// Inputs in baracuda's packed layout:
/// - `a` : `[M, K]` `__half` row-major contiguous activations.
/// - `b_packed` : `[K/16, N*16/8]` `int32` Marlin-shuffled int4
///   weights (use `gptq_to_marlin` host-side to repack from GPTQ).
/// - `scales` : `[K/groupsize, N]` `__half` per-group scales (or
///   `[1, N]` for `groupsize == -1`), pre-permuted by the packer.
///
/// Output `[M, N]` `__half` allocated fresh.
///
/// `groupsize ∈ {-1, 128}`. `max_par` is the parallel-tile upper
/// bound — typical 16 (matches upstream IST-DASLab default).
#[allow(clippy::too_many_arguments)]
pub fn marlin_gemm_f16(
    a: &CudaStorageBytes,
    b_packed: &CudaStorageBytes,
    scales: &CudaStorageBytes,
    m: usize,
    n: usize,
    k: usize,
    groupsize: i32,
    max_par: i32,
) -> Result<CudaStorageBytes> {
    let device = a.device().clone();
    let out_bytes = m * n * std::mem::size_of::<half::f16>();
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    // Marlin requires a zero-initialised int32 workspace with
    // `>= (N / 128) * max_par` entries.
    let ws_entries = ((n + 127) / 128) * (max_par as usize);
    let ws_buf = device.alloc_zeros::<u8>(ws_entries * std::mem::size_of::<i32>())?;
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let status = unsafe {
        sys::baracuda_kernels_int4_marlin_gemm_f16_run(
            i32::try_from(m).map_err(|_| crate::error::CudaError::BaracudaShapeOverflow {
                op: "int4_marlin_gemm_f16", dim_index: 0, dim_value: m,
            })?,
            i32::try_from(n).map_err(|_| crate::error::CudaError::BaracudaShapeOverflow {
                op: "int4_marlin_gemm_f16", dim_index: 1, dim_value: n,
            })?,
            i32::try_from(k).map_err(|_| crate::error::CudaError::BaracudaShapeOverflow {
                op: "int4_marlin_gemm_f16", dim_index: 2, dim_value: k,
            })?,
            a.buffer().as_raw().0 as *const std::ffi::c_void,
            b_packed.buffer().as_raw().0 as *const std::ffi::c_void,
            out_buf.as_raw().0 as *mut std::ffi::c_void,
            scales.buffer().as_raw().0 as *const std::ffi::c_void,
            ws_buf.as_raw().0 as *mut std::ffi::c_void,
            groupsize,
            max_par,
            stream,
        )
    };
    check(status, "int4_marlin_gemm_f16")?;
    device.synchronize()?;
    drop(ws_buf);
    Ok(CudaStorageBytes::from_parts(Arc::new(out_buf), device, out_bytes))
}

/// Marlin shape/alignment validator (no kernel launch). Returns
/// `Ok(())` iff baracuda will accept this (M, N, K, groupsize).
pub fn marlin_can_implement_f16(m: i32, n: i32, k: i32, groupsize: i32) -> Result<()> {
    let status = unsafe {
        sys::baracuda_kernels_int4_marlin_gemm_f16_can_implement(m, n, k, groupsize)
    };
    check(status, "int4_marlin_gemm_f16_can_implement")
}

// ────────────────────────────── AWQ ──────────────────────────────

/// AWQ W4A16 asymmetric GEMM (FP16 activations + output, F32 acc).
///
/// Inputs in baracuda's packed layout (matches HuggingFace `*-AWQ`):
/// - `in_feats` : `[M, IC]` `__half` row-major activations.
/// - `kernel_weights` : `[OC, IC/8]` `int32` packed int4
///   (OC-major, IC-minor — transpose of naive `[K, N]`).
/// - `scaling_factors` : `[IC/group_size, OC]` `__half`.
/// - `zeros` : `[IC/group_size, OC/8]` `int32` packed int4 zero-points.
///
/// Output `[M, OC]` `__half` allocated fresh.
///
/// `group_size ∈ {64, 128}`. `split_k_iters` is caller-chosen — typical
/// 8. Internal workspace sized via baracuda's
/// `int4_awq_gemm_f16_workspace_bytes(M, OC, split_k_iters)`.
#[allow(clippy::too_many_arguments)]
pub fn awq_gemm_f16(
    in_feats: &CudaStorageBytes,
    kernel_weights: &CudaStorageBytes,
    scaling_factors: &CudaStorageBytes,
    zeros: &CudaStorageBytes,
    m: usize,
    ic: usize,
    oc: usize,
    group_size: i32,
    split_k_iters: i32,
) -> Result<CudaStorageBytes> {
    let device = in_feats.device().clone();
    let out_bytes = m * oc * std::mem::size_of::<half::f16>();
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let m_i32 = i32::try_from(m).map_err(|_| crate::error::CudaError::BaracudaShapeOverflow {
        op: "int4_awq_gemm_f16", dim_index: 0, dim_value: m,
    })?;
    let ic_i32 = i32::try_from(ic).map_err(|_| crate::error::CudaError::BaracudaShapeOverflow {
        op: "int4_awq_gemm_f16", dim_index: 1, dim_value: ic,
    })?;
    let oc_i32 = i32::try_from(oc).map_err(|_| crate::error::CudaError::BaracudaShapeOverflow {
        op: "int4_awq_gemm_f16", dim_index: 2, dim_value: oc,
    })?;
    let ws_bytes = unsafe {
        sys::baracuda_kernels_int4_awq_gemm_f16_workspace_bytes(m_i32, oc_i32, split_k_iters)
    };
    let ws_buf = if ws_bytes > 0 {
        Some(device.alloc_zeros::<u8>(ws_bytes)?)
    } else {
        None
    };
    let ws_ptr = ws_buf
        .as_ref()
        .map(|b| b.as_raw().0 as *mut std::ffi::c_void)
        .unwrap_or(std::ptr::null_mut());
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let status = unsafe {
        sys::baracuda_kernels_int4_awq_gemm_f16_run(
            m_i32, ic_i32, oc_i32,
            group_size, split_k_iters,
            in_feats.buffer().as_raw().0 as *const std::ffi::c_void,
            kernel_weights.buffer().as_raw().0 as *const std::ffi::c_void,
            scaling_factors.buffer().as_raw().0 as *const std::ffi::c_void,
            zeros.buffer().as_raw().0 as *const std::ffi::c_void,
            out_buf.as_raw().0 as *mut std::ffi::c_void,
            ws_ptr, ws_bytes, stream,
        )
    };
    check(status, "int4_awq_gemm_f16")?;
    device.synchronize()?;
    drop(ws_buf);
    Ok(CudaStorageBytes::from_parts(Arc::new(out_buf), device, out_bytes))
}

/// AWQ shape/alignment validator (no kernel launch).
pub fn awq_can_implement_f16(
    m: i32, ic: i32, oc: i32, group_size: i32, split_k_iters: i32,
) -> Result<()> {
    let status = unsafe {
        sys::baracuda_kernels_int4_awq_gemm_f16_can_implement(m, ic, oc, group_size, split_k_iters)
    };
    check(status, "int4_awq_gemm_f16_can_implement")
}

// ────────────────────────────── NF4 ──────────────────────────────
//
// bitsandbytes NormalFloat-4 dequant + GEMV. Packing matches
// bitsandbytes `Linear4bit`: `weight[N/2, K]` u8 (two 4-bit codes /
// byte), `absmax[N * K / block_size]` f32 (per-output-row,
// per-K-block scale). `block_size` typically 64.

/// NF4 dequantize to FP16 (smoke / debug path; production uses the
/// fused GEMV variants below).
pub fn nf4_dequantize_f16(
    w_packed: &CudaStorageBytes,
    absmax: &CudaStorageBytes,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<CudaStorageBytes> {
    nf4_dequantize_inner(
        w_packed, absmax, n, k, block_size,
        std::mem::size_of::<half::f16>(),
        sys::baracuda_kernels_nf4_dequantize_f16_run,
        "nf4_dequantize_f16",
    )
}

/// NF4 dequantize to BF16.
pub fn nf4_dequantize_bf16(
    w_packed: &CudaStorageBytes,
    absmax: &CudaStorageBytes,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<CudaStorageBytes> {
    nf4_dequantize_inner(
        w_packed, absmax, n, k, block_size,
        std::mem::size_of::<half::bf16>(),
        sys::baracuda_kernels_nf4_dequantize_bf16_run,
        "nf4_dequantize_bf16",
    )
}

/// NF4 dequantize to FP32 (roundtrip smoke-test path only).
pub fn nf4_dequantize_f32(
    w_packed: &CudaStorageBytes,
    absmax: &CudaStorageBytes,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<CudaStorageBytes> {
    nf4_dequantize_inner(
        w_packed, absmax, n, k, block_size,
        std::mem::size_of::<f32>(),
        sys::baracuda_kernels_nf4_dequantize_f32_run,
        "nf4_dequantize_f32",
    )
}

type Nf4DequantizeRun = unsafe extern "C" fn(
    n: i32, k: i32, block_size: i32,
    w_packed: *const std::ffi::c_void,
    absmax: *const std::ffi::c_void,
    out: *mut std::ffi::c_void,
    stream: *mut std::ffi::c_void,
) -> i32;

fn nf4_dequantize_inner(
    w_packed: &CudaStorageBytes,
    absmax: &CudaStorageBytes,
    n: usize,
    k: usize,
    block_size: usize,
    dtype_size_bytes: usize,
    kernel: Nf4DequantizeRun,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let device = w_packed.device().clone();
    let out_bytes = n * k * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let n_i32 = i32::try_from(n).map_err(|_| crate::error::CudaError::BaracudaShapeOverflow {
        op: op_label, dim_index: 0, dim_value: n,
    })?;
    let k_i32 = i32::try_from(k).map_err(|_| crate::error::CudaError::BaracudaShapeOverflow {
        op: op_label, dim_index: 1, dim_value: k,
    })?;
    let bs_i32 = i32::try_from(block_size).map_err(|_| {
        crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label, dim_index: 2, dim_value: block_size,
        }
    })?;
    let status = unsafe {
        kernel(
            n_i32, k_i32, bs_i32,
            w_packed.buffer().as_raw().0 as *const std::ffi::c_void,
            absmax.buffer().as_raw().0 as *const std::ffi::c_void,
            out_buf.as_raw().0 as *mut std::ffi::c_void,
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out_buf), device, out_bytes))
}

type Nf4GemvRun = unsafe extern "C" fn(
    n: i32, k: i32, block_size: i32,
    w_packed: *const std::ffi::c_void,
    absmax: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    out: *mut std::ffi::c_void,
    stream: *mut std::ffi::c_void,
) -> i32;

fn nf4_gemv_inner(
    w_packed: &CudaStorageBytes,
    absmax: &CudaStorageBytes,
    activations: &CudaStorageBytes,
    n: usize,
    k: usize,
    block_size: usize,
    m: usize,
    dtype_size_bytes: usize,
    kernel: Nf4GemvRun,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let device = w_packed.device().clone();
    let out_bytes = m * n * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let n_i32 = i32::try_from(n).map_err(|_| crate::error::CudaError::BaracudaShapeOverflow {
        op: op_label, dim_index: 0, dim_value: n,
    })?;
    let k_i32 = i32::try_from(k).map_err(|_| crate::error::CudaError::BaracudaShapeOverflow {
        op: op_label, dim_index: 1, dim_value: k,
    })?;
    let bs_i32 = i32::try_from(block_size).map_err(|_| {
        crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label, dim_index: 2, dim_value: block_size,
        }
    })?;
    let status = unsafe {
        kernel(
            n_i32, k_i32, bs_i32,
            w_packed.buffer().as_raw().0 as *const std::ffi::c_void,
            absmax.buffer().as_raw().0 as *const std::ffi::c_void,
            activations.buffer().as_raw().0 as *const std::ffi::c_void,
            out_buf.as_raw().0 as *mut std::ffi::c_void,
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    drop(m); // silence unused (kept for documentation purposes)
    Ok(CudaStorageBytes::from_parts(Arc::new(out_buf), device, out_bytes))
}

macro_rules! nf4_gemv {
    ($name:ident, $sys:ident, $dtype_size:expr, $m:expr, $label:expr) => {
        #[doc = concat!("NF4 W4A16 GEMV (M=", stringify!($m), "), ", $label, ".")]
        pub fn $name(
            w_packed: &CudaStorageBytes,
            absmax: &CudaStorageBytes,
            activations: &CudaStorageBytes,
            n: usize,
            k: usize,
            block_size: usize,
        ) -> Result<CudaStorageBytes> {
            nf4_gemv_inner(
                w_packed, absmax, activations, n, k, block_size, $m, $dtype_size,
                sys::$sys,
                stringify!($name),
            )
        }
    };
}

nf4_gemv!(nf4_gemv_m1_f16,  baracuda_kernels_nf4_gemv_m1_f16_run,  2, 1, "f16 act");
nf4_gemv!(nf4_gemv_m2_f16,  baracuda_kernels_nf4_gemv_m2_f16_run,  2, 2, "f16 act");
nf4_gemv!(nf4_gemv_m4_f16,  baracuda_kernels_nf4_gemv_m4_f16_run,  2, 4, "f16 act");
nf4_gemv!(nf4_gemv_m8_f16,  baracuda_kernels_nf4_gemv_m8_f16_run,  2, 8, "f16 act");
nf4_gemv!(nf4_gemv_m1_bf16, baracuda_kernels_nf4_gemv_m1_bf16_run, 2, 1, "bf16 act");
nf4_gemv!(nf4_gemv_m2_bf16, baracuda_kernels_nf4_gemv_m2_bf16_run, 2, 2, "bf16 act");
nf4_gemv!(nf4_gemv_m4_bf16, baracuda_kernels_nf4_gemv_m4_bf16_run, 2, 4, "bf16 act");
nf4_gemv!(nf4_gemv_m8_bf16, baracuda_kernels_nf4_gemv_m8_bf16_run, 2, 8, "bf16 act");

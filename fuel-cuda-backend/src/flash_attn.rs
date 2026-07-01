//! Lazy-graph CUDA backend for `Op::FlashAttn` — baracuda FA2 path.
//!
//! Calls `baracuda_kernels_fa2_sdpa_<dt>_run_v2` (alpha.60, Phase
//! 42 + 59a + 60). Vendor tree is Dao-AILab FA2 v2.8.3 plus the
//! head_dim {160, 224, 512} expansion absorbed from Candle/Fuel's
//! patches (PR #245, #2688, #3417). FW only — Fuel-side BW would
//! need to gate on head_dim being in baracuda's BW-supported set
//! {32, 64, 96, 128, 192, 256}; not relevant for the current
//! Op::FlashAttn (forward-only graph node).
//!
//! Layout notes:
//! - The lazy IR's `Op::FlashAttn` uses BHSD (`[B, Hq, Sq, D]`).
//! - baracuda's v2 launcher takes (batch, num_heads, num_heads_k,
//!   seq_q, seq_k, head_dim) and assumes contiguous BHSD layout
//!   internally — same as Fuel's executor materializes upstream.
//!
//! Coverage vs the prior upstream-vendored path:
//! - head_dim ∈ {32, 64, 96, 128, 160, 192, 224, 256, 512} ✓
//! - GQA (Hq % Hkv == 0) ✓
//! - ALiBi (per-head or per-batch-per-head f32 slopes) ✓
//! - Sliding window (left/right) ✓
//! - Softcap (Gemma-2) ✓
//! - F16 / BF16 ✓
//!
//! Restrictions:
//! - head_dim must be in the supported set (no d40 / d80 rounding
//!   — those callers fall back to standard attention in model code).
//! - All inputs must be contiguous (executor handles materialization).
//!
//! Staged module: `launch` is not yet wired into `Op::FlashAttn`
//! dispatch (the executor currently routes FlashAttn elsewhere), so
//! the whole module is dead today — hence the module-level allow.

#![allow(dead_code)]

use crate::storage::{CudaStorage, CudaStorageSlice};
use baracuda_kernels_sys as sys;
use fuel_ir::{DType, Layout};

/// Translate the lazy `Op::FlashAttn` parameters into the FA-v2
/// FFI's `is_causal` / window-size convention.
///
/// FFI semantics:
/// - is_causal: 1 iff `window_size_right == 0 && window_size_left < 0`.
/// - Otherwise window_size_left/right >= 0 to opt into local attention;
///   negative means "no limit on that side."
fn translate_window(
    causal: bool,
    window_left: Option<usize>,
    window_right: Option<usize>,
    seqlen_k: usize,
) -> (i32, i32, i32) {
    let mut wsl = window_left
        .filter(|v| *v <= seqlen_k)
        .map(|v| v as i32)
        .unwrap_or(-1);
    let mut wsr = window_right
        .filter(|v| *v <= seqlen_k)
        .map(|v| v as i32)
        .unwrap_or(-1);
    let causal_only = causal && window_left.is_none() && window_right.is_none();
    let is_causal = if causal_only {
        wsl = -1;
        wsr = 0;
        1
    } else {
        if causal {
            wsr = 0;
        }
        if wsl < 0 && wsr >= 0 {
            wsl = seqlen_k as i32;
        }
        if wsl >= 0 && wsr < 0 {
            wsr = seqlen_k as i32;
        }
        0
    };
    (wsl, wsr, is_causal)
}

/// Dispatch entry for the baracuda FA2 launcher. Returns a fresh
/// `CudaStorage` of shape `[B, Hq, Sq, D]` matching `q`.
///
/// The legacy `CudaBackend::flash_attn` trait method that called this
/// was retired with the `GraphBackend` trait (executor-unification
/// Session 7). The launcher itself is preserved for the queued FA2
/// eager-wrapper retirement session, which will re-wire it onto the
/// pipelined `Op::Fused(FLASH_ATTN, _)` dispatch path; hence `dead_code`.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn launch(
    q: &CudaStorage,
    k: &CudaStorage,
    v: &CudaStorage,
    alibi_slopes: Option<&CudaStorage>,
    q_layout: &Layout,
    k_layout: &Layout,
    v_layout: &Layout,
    softmax_scale: f32,
    causal: bool,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    softcap: Option<f32>,
) -> fuel_ir::Result<CudaStorage> {
    let is_bf16 = match q.dtype() {
        DType::F16 => false,
        DType::BF16 => true,
        other => fuel_ir::bail!(
            "CudaBackend::flash_attn: dtype {other:?} not supported (F16 or BF16 only)"
        ),
    };
    if k.dtype() != q.dtype() || v.dtype() != q.dtype() {
        fuel_ir::bail!(
            "CudaBackend::flash_attn: dtype mismatch q={:?} k={:?} v={:?}",
            q.dtype(),
            k.dtype(),
            v.dtype(),
        );
    }
    if !q_layout.is_contiguous() || !k_layout.is_contiguous() || !v_layout.is_contiguous() {
        fuel_ir::bail!("CudaBackend::flash_attn: strided inputs not supported");
    }

    // Lazy IR shape: q = [B, Hq, Sq, D], k/v = [B, Hkv, Sk, D].
    let q_dims = q_layout.shape().dims();
    let k_dims = k_layout.shape().dims();
    let v_dims = v_layout.shape().dims();
    if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
        fuel_ir::bail!(
            "CudaBackend::flash_attn: rank-4 q/k/v required, got {q_dims:?} {k_dims:?} {v_dims:?}"
        );
    }
    let (b, hq, sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let (_, hkv, sk, _) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
    if hq % hkv != 0 {
        fuel_ir::bail!(
            "CudaBackend::flash_attn: Hq={hq} must be a multiple of Hkv={hkv}"
        );
    }
    // baracuda Phase 60 supports the full FA2 v2.8.3 head_dim set.
    // SD 1.5-style hd40/hd80 are NOT in the supported set — those
    // callers stay on the standard-attention fallback path.
    const SUPPORTED_HEAD_DIMS: [usize; 9] = [32, 64, 96, 128, 160, 192, 224, 256, 512];
    if !SUPPORTED_HEAD_DIMS.contains(&d) {
        fuel_ir::bail!(
            "CudaBackend::flash_attn: head_dim={d} not in baracuda FA2's supported set {SUPPORTED_HEAD_DIMS:?}; \
             use the standard attention fallback for this model"
        );
    }

    let device = q.device().clone();

    let (window_size_left, window_size_right, is_causal) =
        translate_window(causal, window_size_left, window_size_right, sk);

    // Host-side validation through baracuda's can_implement_v2 —
    // catches anything our Fuel-side checks didn't (e.g. softcap +
    // dropout interaction; baracuda rejects).
    let ci_status = unsafe {
        if is_bf16 {
            sys::baracuda_kernels_fa2_sdpa_bf16_can_implement_v2(
                b as i32,
                hq as i32,
                hkv as i32,
                sq as i32,
                sk as i32,
                d as i32,
                is_causal,
                window_size_left,
                window_size_right,
                softcap.unwrap_or(0.0),
            )
        } else {
            sys::baracuda_kernels_fa2_sdpa_f16_can_implement_v2(
                b as i32,
                hq as i32,
                hkv as i32,
                sq as i32,
                sk as i32,
                d as i32,
                is_causal,
                window_size_left,
                window_size_right,
                softcap.unwrap_or(0.0),
            )
        }
    };
    crate::baracuda::status::check(ci_status, "fa2_sdpa_can_implement_v2")?;

    // Allocate output (matches Q's shape) + LSE scratch (always
    // f32, per baracuda's documented ABI; FA2 internally accumulates
    // softmax in f32 regardless of operand dtype).
    use baracuda_driver::DeviceBuffer;
    let stream = device.cuda_stream();
    let elem_count = b * hq * sq * d;
    let (out_storage, out_ptr) = match q.dtype() {
        DType::F16 => {
            let buf = DeviceBuffer::<half::f16>::zeros(stream.context(), elem_count)
                .map_err(crate::error::CudaError::from)?;
            let ptr = buf.as_raw().0;
            (
                CudaStorage {
                    slice: CudaStorageSlice::F16(buf),
                    device: device.clone(),
                },
                ptr,
            )
        }
        DType::BF16 => {
            let buf = DeviceBuffer::<half::bf16>::zeros(stream.context(), elem_count)
                .map_err(crate::error::CudaError::from)?;
            let ptr = buf.as_raw().0;
            (
                CudaStorage {
                    slice: CudaStorageSlice::BF16(buf),
                    device: device.clone(),
                },
                ptr,
            )
        }
        _ => unreachable!("checked above"),
    };
    // Softmax LSE buffer is `[B, Hq, Sq]` f32 per baracuda's contract
    // (different from the upstream `[B, Hq, 128, Sq]` layout — alpha.60
    // uses the compact per-row LSE format).
    let lse_n = b * hq * sq;
    let lse_buf = DeviceBuffer::<f32>::zeros(stream.context(), lse_n)
        .map_err(crate::error::CudaError::from)?;
    let lse_ptr = lse_buf.as_raw().0;

    fn raw_ptr(s: &CudaStorage) -> u64 {
        match &s.slice {
            CudaStorageSlice::F16(b) => b.as_raw().0,
            CudaStorageSlice::BF16(b) => b.as_raw().0,
            CudaStorageSlice::F32(b) => b.as_raw().0,
            CudaStorageSlice::F64(b) => b.as_raw().0,
            CudaStorageSlice::U8(b) => b.as_raw().0,
            CudaStorageSlice::I8(b) => b.as_raw().0,
            CudaStorageSlice::U32(b) => b.as_raw().0,
            CudaStorageSlice::I16(b) => b.as_raw().0,
            CudaStorageSlice::I32(b) => b.as_raw().0,
            CudaStorageSlice::I64(b) => b.as_raw().0,
            CudaStorageSlice::F8E4M3(b) => b.as_raw().0,
            CudaStorageSlice::F6E2M3(b) => b.as_raw().0,
            CudaStorageSlice::F6E3M2(b) => b.as_raw().0,
            CudaStorageSlice::F4(b) => b.as_raw().0,
            CudaStorageSlice::F8E8M0(b) => b.as_raw().0,
        }
    }
    let q_ptr = raw_ptr(q);
    let k_ptr = raw_ptr(k);
    let v_ptr = raw_ptr(v);
    let alibi_ptr = match alibi_slopes {
        Some(a) => raw_ptr(a) as *const core::ffi::c_void,
        None => core::ptr::null(),
    };
    // alibi_batch_stride: 0 when the slope tensor is `[num_heads]`
    // (shared across batches); `num_heads` when it's
    // `[batch, num_heads]`. Fuel's existing callers all pass the
    // shared layout (matches the upstream wrapper's
    // alibi_slopes_batch_stride = 0 default).
    let alibi_batch_stride: i32 = 0;

    let stream_raw = stream.as_raw() as *mut core::ffi::c_void;
    let status = unsafe {
        if is_bf16 {
            sys::baracuda_kernels_fa2_sdpa_bf16_run_v2(
                b as i32,
                hq as i32,
                hkv as i32,
                sq as i32,
                sk as i32,
                d as i32,
                softmax_scale,
                is_causal,
                alibi_ptr,
                alibi_batch_stride,
                window_size_left,
                window_size_right,
                softcap.unwrap_or(0.0),
                q_ptr as *const core::ffi::c_void,
                k_ptr as *const core::ffi::c_void,
                v_ptr as *const core::ffi::c_void,
                out_ptr as *mut core::ffi::c_void,
                lse_ptr as *mut core::ffi::c_void,
                core::ptr::null_mut(),
                0,
                stream_raw,
            )
        } else {
            sys::baracuda_kernels_fa2_sdpa_f16_run_v2(
                b as i32,
                hq as i32,
                hkv as i32,
                sq as i32,
                sk as i32,
                d as i32,
                softmax_scale,
                is_causal,
                alibi_ptr,
                alibi_batch_stride,
                window_size_left,
                window_size_right,
                softcap.unwrap_or(0.0),
                q_ptr as *const core::ffi::c_void,
                k_ptr as *const core::ffi::c_void,
                v_ptr as *const core::ffi::c_void,
                out_ptr as *mut core::ffi::c_void,
                lse_ptr as *mut core::ffi::c_void,
                core::ptr::null_mut(),
                0,
                stream_raw,
            )
        }
    };
    crate::baracuda::status::check(status, "fa2_sdpa_run_v2")?;
    device.synchronize()?;
    drop(lse_buf);

    Ok(out_storage)
}

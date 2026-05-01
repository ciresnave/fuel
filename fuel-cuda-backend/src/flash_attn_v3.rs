//! Lazy-graph CUDA backend for `Op::FlashAttn` on sm90 (Phase 8 Tier 3 v3).
//!
//! Calls `fuel-flash-attn-v3-cuda-sys::run_mha_v3` (Dao-AILab FA-v3
//! Hopper kernels) directly with raw `CUdeviceptr`. Mirrors
//! `flash_attn.rs` (the sm80 path) — same input contract, same
//! BHSD layout convention, same dtype + head_dim restrictions.
//!
//! Only the FFI signature differs: v3 drops `softcap` (the kernel
//! doesn't support it), adds `use_gqa_packing` + `total_q`/`total_k`.
//! `softcap=Some(_)` returns an Err at this layer so the executor
//! falls back to attention_naive for soft-cap workloads.
//!
//! No live-GPU validation in-tree — sm90 hardware not available on
//! the dev rig. The compile-time check is the gate. Once a Hopper
//! tester is online, mirror `tests/flash_attn_cuda.rs` to point at
//! the v3 path with the appropriate compute-capability gate.

use crate::storage::{CudaStorage, CudaStorageSlice};
use baracuda_driver::DevicePtr;
use fuel_core_types::{DType, Layout};

fn round_multiple(x: usize, m: usize) -> usize {
    (x + m - 1) / m * m
}

/// Translate the lazy `Op::FlashAttn` mask params into the v3 FFI's
/// `is_causal` / window-size convention. Identical to the sm80 path.
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
        if causal { wsr = 0; }
        if wsl < 0 && wsr >= 0 { wsl = seqlen_k as i32; }
        if wsl >= 0 && wsr < 0 { wsr = seqlen_k as i32; }
        0
    };
    (wsl, wsr, is_causal)
}

/// Dispatch entry called from `CudaBackend::flash_attn` when the v3
/// feature is selected. Returns a fresh `CudaStorage` of shape
/// `[B, Hq, Sq, D]` matching `q`.
#[allow(clippy::too_many_arguments)]
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
) -> fuel_core_types::Result<CudaStorage> {
    // FA-v3's run_mha doesn't carry a softcap parameter. If the caller
    // wants softcap, we bail and let the executor fall back.
    if softcap.is_some() {
        fuel_core_types::bail!(
            "CudaBackend::flash_attn (v3): softcap not supported by FA-v3 kernels; \
             falling back to reference attention_naive"
        );
    }

    let is_bf16 = match q.dtype() {
        DType::F16 => false,
        DType::BF16 => true,
        other => fuel_core_types::bail!(
            "CudaBackend::flash_attn (v3): dtype {other:?} not supported (F16 or BF16 only)"
        ),
    };
    if k.dtype() != q.dtype() || v.dtype() != q.dtype() {
        fuel_core_types::bail!(
            "CudaBackend::flash_attn (v3): dtype mismatch q={:?} k={:?} v={:?}",
            q.dtype(), k.dtype(), v.dtype(),
        );
    }
    if !q_layout.is_contiguous() || !k_layout.is_contiguous() || !v_layout.is_contiguous() {
        fuel_core_types::bail!("CudaBackend::flash_attn (v3): strided inputs not supported");
    }

    let q_dims = q_layout.shape().dims();
    let k_dims = k_layout.shape().dims();
    let v_dims = v_layout.shape().dims();
    if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
        fuel_core_types::bail!(
            "CudaBackend::flash_attn (v3): rank-4 q/k/v required, got {q_dims:?} {k_dims:?} {v_dims:?}"
        );
    }
    let (b, hq, sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let (_, hkv, sk, _) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
    if hq % hkv != 0 {
        fuel_core_types::bail!(
            "CudaBackend::flash_attn (v3): Hq={hq} must be a multiple of Hkv={hkv}"
        );
    }
    // FA-v3 supports head_dim ∈ {64, 128, 256, 512} per the kernel matrix.
    if !matches!(d, 64 | 128 | 256 | 512) {
        fuel_core_types::bail!(
            "CudaBackend::flash_attn (v3): head_dim={d} unsupported (FA-v3 ships kernels for {{64,128,256,512}})"
        );
    }

    let device = q.device().clone();
    let head_size = round_multiple(d, 8);
    let head_size_rounded = round_multiple(head_size, 32);
    let seqlen_q_rounded = round_multiple(sq, 128);
    let seqlen_k_rounded = round_multiple(sk, 128);

    // BHSD strides (same as the v2 sm80 path).
    let q_batch_stride = (hq * sq * d) as u32;
    let q_head_stride  = (sq * d) as u32;
    let q_row_stride   = d as u32;
    let k_batch_stride = (hkv * sk * d) as u32;
    let k_head_stride  = (sk * d) as u32;
    let k_row_stride   = d as u32;
    let v_batch_stride = k_batch_stride;
    let v_head_stride  = k_head_stride;
    let v_row_stride   = k_row_stride;
    let o_batch_stride = q_batch_stride;
    let o_head_stride  = q_head_stride;
    let o_row_stride   = q_row_stride;

    let (wsl, wsr, is_causal) =
        translate_window(causal, window_size_left, window_size_right, sk);

    let stream = device.cuda_stream();
    use baracuda_driver::DeviceBuffer;
    let elem_count = b * hq * sq * d;

    let (out_storage, out_ptr) = match q.dtype() {
        DType::F16 => {
            let buf = DeviceBuffer::<half::f16>::zeros(stream.context(), elem_count)
                .map_err(crate::error::CudaError::from)?;
            let ptr = buf.as_raw().0;
            (CudaStorage { slice: CudaStorageSlice::F16(buf), device: device.clone() }, ptr)
        }
        DType::BF16 => {
            let buf = DeviceBuffer::<half::bf16>::zeros(stream.context(), elem_count)
                .map_err(crate::error::CudaError::from)?;
            let ptr = buf.as_raw().0;
            (CudaStorage { slice: CudaStorageSlice::BF16(buf), device: device.clone() }, ptr)
        }
        _ => unreachable!("checked above"),
    };
    let lse_n = b * hq * 128 * sq;
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

    // FA-v3's `total_q` / `total_k` are for variable-length packing.
    // For fixed-length (which is all we expose at the lazy IR layer),
    // they're b·sq and b·sk.
    let total_q = (b * sq) as u32;
    let total_k = (b * sk) as u32;

    unsafe {
        fuel_flash_attn_v3_cuda_sys::run_mha_v3(
            q_ptr as *const core::ffi::c_void,
            k_ptr as *const core::ffi::c_void,
            v_ptr as *const core::ffi::c_void,
            out_ptr as *const core::ffi::c_void,
            lse_ptr as *const core::ffi::c_void,
            alibi_ptr,
            /* cu_seqlens_q_ptr */ core::ptr::null(),
            /* cu_seqlens_k_ptr */ core::ptr::null(),
            q_batch_stride, k_batch_stride, v_batch_stride, o_batch_stride,
            /* alibi_slopes_batch_stride */ 0,
            q_row_stride, k_row_stride, v_row_stride, o_row_stride,
            q_head_stride, k_head_stride, v_head_stride, o_head_stride,
            b as u32, hq as u32, hkv as u32,
            head_size as u32, head_size_rounded as u32,
            softmax_scale,
            sq as u32, sk as u32,
            seqlen_q_rounded as u32, seqlen_k_rounded as u32,
            if is_bf16 { 1 } else { 0 },
            is_causal,
            /* unpadded_lse */ 0,
            /* use_gqa_packing */ if hq != hkv { 1 } else { 0 },
            wsl, wsr,
            total_q, total_k,
        );
    }
    drop(lse_buf);
    Ok(out_storage)
}

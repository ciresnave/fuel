//! Sort-free sampling kernels from baracuda alpha.58 (Phase 46,
//! FlashInfer cherry-pick, Apache-2.0). Take a per-batch
//! `probs: [batch, vocab]` F32 distribution (already softmax-
//! normalized), emit one token per batch row directly on-device —
//! no D2H per token, which is the inference-loop win over the
//! existing CPU-side `LogitsProcessor::sample` path.
//!
//! All four samplers share the same shape:
//!   - `batch` × `vocab` input
//!   - `output: i32[batch]` sampled token indices
//!   - `valid:  i32[batch]` per-row status (0 = sample succeeded;
//!     non-zero = the row's filter rejected all candidates)
//!   - Philox `(seed, offset)` for deterministic replay
//!   - `deterministic: 0/1` — when 1, the kernel argmaxes instead of
//!     sampling (useful for "greedy with the same plumbing")
//!
//! F32 probs only — flashinfer's upstream surface is F32-only here.
//! Callers in fp16/bf16 should cast to F32 before calling.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::Result;

use crate::byte_storage::CudaStorageBytes;

use super::status::check;

/// Result of a sampling call: a `[batch]` byte buffer of i32 sampled
/// token indices and a `[batch]` byte buffer of i32 per-row status
/// codes (0 = success).
pub struct SamplingOutput {
    pub tokens: CudaStorageBytes,
    pub valid: CudaStorageBytes,
}

fn alloc_pair(device: &crate::CudaDevice, batch: usize) -> Result<(CudaStorageBytes, CudaStorageBytes)> {
    let bytes = batch * std::mem::size_of::<i32>();
    let tokens_buf = device.alloc_zeros::<u8>(bytes)?;
    let valid_buf  = device.alloc_zeros::<u8>(bytes)?;
    Ok((
        CudaStorageBytes::from_parts(Arc::new(tokens_buf), device.clone(), bytes),
        CudaStorageBytes::from_parts(Arc::new(valid_buf),  device.clone(), bytes),
    ))
}

fn batch_vocab_i32(
    op: &'static str,
    batch: usize,
    vocab: usize,
) -> Result<(i32, i32)> {
    let b = i32::try_from(batch).map_err(|_| crate::error::CudaError::BaracudaShapeOverflow {
        op, dim_index: 0, dim_value: batch,
    })?;
    let v = i32::try_from(vocab).map_err(|_| crate::error::CudaError::BaracudaShapeOverflow {
        op, dim_index: 1, dim_value: vocab,
    })?;
    Ok((b, v))
}

/// Top-K sampling. Keeps the `top_k_val` highest-probability tokens
/// per row and samples proportionally to their renormalized mass.
/// `deterministic = 1` selects the argmax instead.
pub fn top_k_sampling_f32(
    probs: &CudaStorageBytes,
    batch: usize,
    vocab: usize,
    top_k_val: i32,
    deterministic: bool,
    seed_val: u64,
    offset_val: u64,
) -> Result<SamplingOutput> {
    let device = probs.device().clone();
    let (b, v) = batch_vocab_i32("flashinfer_top_k_sampling_f32", batch, vocab)?;
    let (tokens, valid) = alloc_pair(&device, batch)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let status = unsafe {
        sys::baracuda_kernels_flashinfer_top_k_sampling_f32_run(
            b, v, top_k_val,
            if deterministic { 1 } else { 0 },
            seed_val, offset_val,
            probs.buffer().as_raw().0 as *const std::ffi::c_void,
            tokens.buffer().as_raw().0 as *mut std::ffi::c_void,
            valid.buffer().as_raw().0  as *mut std::ffi::c_void,
            stream,
        )
    };
    check(status, "flashinfer_top_k_sampling_f32")?;
    device.synchronize()?;
    Ok(SamplingOutput { tokens, valid })
}

/// Top-P (nucleus) sampling. Keeps the smallest set of tokens whose
/// cumulative probability ≥ `top_p_val` per row, renormalizes, and
/// samples.
pub fn top_p_sampling_f32(
    probs: &CudaStorageBytes,
    batch: usize,
    vocab: usize,
    top_p_val: f32,
    deterministic: bool,
    seed_val: u64,
    offset_val: u64,
) -> Result<SamplingOutput> {
    let device = probs.device().clone();
    let (b, v) = batch_vocab_i32("flashinfer_top_p_sampling_f32", batch, vocab)?;
    let (tokens, valid) = alloc_pair(&device, batch)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let status = unsafe {
        sys::baracuda_kernels_flashinfer_top_p_sampling_f32_run(
            b, v, top_p_val,
            if deterministic { 1 } else { 0 },
            seed_val, offset_val,
            probs.buffer().as_raw().0 as *const std::ffi::c_void,
            tokens.buffer().as_raw().0 as *mut std::ffi::c_void,
            valid.buffer().as_raw().0  as *mut std::ffi::c_void,
            stream,
        )
    };
    check(status, "flashinfer_top_p_sampling_f32")?;
    device.synchronize()?;
    Ok(SamplingOutput { tokens, valid })
}

/// Min-P sampling. Filters tokens with `probs[i] < min_p_val * max_prob`
/// per row, then samples from the survivors. Reference: Minh et al.
/// "Min P Sampling" (2024).
pub fn min_p_sampling_f32(
    probs: &CudaStorageBytes,
    batch: usize,
    vocab: usize,
    min_p_val: f32,
    deterministic: bool,
    seed_val: u64,
    offset_val: u64,
) -> Result<SamplingOutput> {
    let device = probs.device().clone();
    let (b, v) = batch_vocab_i32("flashinfer_min_p_sampling_f32", batch, vocab)?;
    let (tokens, valid) = alloc_pair(&device, batch)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let status = unsafe {
        sys::baracuda_kernels_flashinfer_min_p_sampling_f32_run(
            b, v, min_p_val,
            if deterministic { 1 } else { 0 },
            seed_val, offset_val,
            probs.buffer().as_raw().0 as *const std::ffi::c_void,
            tokens.buffer().as_raw().0 as *mut std::ffi::c_void,
            valid.buffer().as_raw().0  as *mut std::ffi::c_void,
            stream,
        )
    };
    check(status, "flashinfer_min_p_sampling_f32")?;
    device.synchronize()?;
    Ok(SamplingOutput { tokens, valid })
}

/// Combined top-K + top-P sampling: top-K filter first, then top-P
/// within the survivors, then sample. Matches the existing Fuel
/// `Sampling::TopKThenTopP` semantics — one kernel instead of two
/// CPU passes + a D2H.
pub fn top_k_top_p_sampling_f32(
    probs: &CudaStorageBytes,
    batch: usize,
    vocab: usize,
    top_k_val: i32,
    top_p_val: f32,
    deterministic: bool,
    seed_val: u64,
    offset_val: u64,
) -> Result<SamplingOutput> {
    let device = probs.device().clone();
    let (b, v) = batch_vocab_i32("flashinfer_top_k_top_p_sampling_f32", batch, vocab)?;
    let (tokens, valid) = alloc_pair(&device, batch)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let status = unsafe {
        sys::baracuda_kernels_flashinfer_top_k_top_p_sampling_f32_run(
            b, v, top_k_val, top_p_val,
            if deterministic { 1 } else { 0 },
            seed_val, offset_val,
            probs.buffer().as_raw().0 as *const std::ffi::c_void,
            tokens.buffer().as_raw().0 as *mut std::ffi::c_void,
            valid.buffer().as_raw().0  as *mut std::ffi::c_void,
            stream,
        )
    };
    check(status, "flashinfer_top_k_top_p_sampling_f32")?;
    device.synchronize()?;
    Ok(SamplingOutput { tokens, valid })
}

// ─────────────────────────── can_implement ───────────────────────────
//
// Host-side validators that return `Ok(())` iff the kernel will accept
// the given problem shape. Pre-launch checks let dispatch code reject
// invalid shapes without paying the kernel-launch round trip on
// failure. Pattern matches what baracuda alpha.59 exposes for its
// FlashInfer sampling family.

/// Pre-launch validation for [`top_k_sampling_f32`].
pub fn top_k_sampling_can_implement(batch: i32, vocab: i32, top_k_val: i32) -> Result<()> {
    let status = unsafe {
        sys::baracuda_kernels_flashinfer_top_k_sampling_f32_can_implement(batch, vocab, top_k_val)
    };
    check(status, "flashinfer_top_k_sampling_f32_can_implement")
}

/// Pre-launch validation for [`top_p_sampling_f32`].
pub fn top_p_sampling_can_implement(batch: i32, vocab: i32, top_p_val: f32) -> Result<()> {
    let status = unsafe {
        sys::baracuda_kernels_flashinfer_top_p_sampling_f32_can_implement(batch, vocab, top_p_val)
    };
    check(status, "flashinfer_top_p_sampling_f32_can_implement")
}

/// Pre-launch validation for [`min_p_sampling_f32`].
pub fn min_p_sampling_can_implement(batch: i32, vocab: i32, min_p_val: f32) -> Result<()> {
    let status = unsafe {
        sys::baracuda_kernels_flashinfer_min_p_sampling_f32_can_implement(batch, vocab, min_p_val)
    };
    check(status, "flashinfer_min_p_sampling_f32_can_implement")
}

/// Pre-launch validation for [`top_k_top_p_sampling_f32`].
pub fn top_k_top_p_sampling_can_implement(
    batch: i32,
    vocab: i32,
    top_k_val: i32,
    top_p_val: f32,
) -> Result<()> {
    let status = unsafe {
        sys::baracuda_kernels_flashinfer_top_k_top_p_sampling_f32_can_implement(
            batch, vocab, top_k_val, top_p_val,
        )
    };
    check(status, "flashinfer_top_k_top_p_sampling_f32_can_implement")
}

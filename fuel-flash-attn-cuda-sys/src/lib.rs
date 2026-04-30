//! `fuel-flash-attn-cuda-sys` — bare `extern "C"` bindings to the
//! Dao-AILab FlashAttention v2 CUDA kernels (sm80 / Ampere).
//!
//! This crate has no Rust API beyond the FFI declaration and the
//! kernels-as-static-lib that ships with it. It exists so that both
//! the eager-mode `fuel-flash-attn-cuda` wrapper AND the lazy-graph
//! `fuel-graph-cuda` backend can call the same kernels without
//! creating a circular dependency between those two crates.
//!
//! Callers must hand in raw `CUdeviceptr` values + dimensions + a
//! stream. Type safety, dtype dispatch, and shape validation belong
//! at the call site — this crate is intentionally as thin as
//! possible.

use core::ffi::{c_int, c_void};

extern "C" {
    pub fn run_mha(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        alibi_slopes_ptr: *const c_void,

        cu_seqlens_q_ptr: *const i32,
        cu_seqlens_k_ptr: *const i32,

        q_batch_stride: u32,
        k_batch_stride: u32,
        v_batch_stride: u32,
        o_batch_stride: u32,
        alibi_slopes_batch_stride: u32,

        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,

        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,

        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        d_rounded: u32,
        softmax_scale: f32,

        seqlen_q: u32,
        seqlen_k: u32,
        seqlen_q_rounded: u32,
        seqlen_k_rounded: u32,

        is_bf16: c_int,
        is_causal: c_int,
        unpadded_lse: c_int,

        window_size_left: c_int,
        window_size_right: c_int,

        softcap: f32,
    );

}

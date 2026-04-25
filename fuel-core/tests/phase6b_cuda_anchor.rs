//! Phase 6b oracle-gate: anchor-class CUDA forward passes match the
//! reference backend within tolerance.
//!
//! The simple-matmul case is the smoke test — it just confirms the
//! `realize_f32_reference()` vs `realize_f32_cuda(&mut exe)` oracle
//! comparison loop is wired end-to-end. The LLaMA case is the real
//! gate: a 2-layer synthetic forward exercises matmul, RoPE,
//! softmax, RMS norm, and SwiGLU on CUDA, and the rms_norm /
//! layernorm shared-memory uninit-read fix in
//! `fuel-cuda-kernels/src/reduce.cu` (committed alongside this test)
//! is what makes it pass.
//!
//! Feature-gated on `cuda` and requires a CUDA device. Skips
//! cleanly when no CUDA visible.

#![cfg(feature = "cuda")]

use fuel_core::lazy::{LayerWeights, LazyTensor, LlamaConfig, LlamaModel, LlamaWeights};
use fuel_core_types::{probe::BackendId, Shape};
use fuel_graph_executor::GraphExecutor;
use std::sync::Arc;

fn cuda_present() -> bool {
    let probe = fuel_core::probe::ProbeReport::probe_all();
    probe.devices.iter().any(|d| d.backend == BackendId::Cuda)
}

#[test]
fn single_matmul_cuda_matches_reference_within_tolerance() {
    if !cuda_present() {
        eprintln!("skipping: no CUDA device visible to Fuel");
        return;
    }

    // 32×48 @ 48×24 — deterministic inputs.
    let (m, k, n) = (32usize, 48, 24);
    let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let a = LazyTensor::from_f32(a_data, Shape::from_dims(&[m, k]));
    let b = a.const_f32_like(b_data, Shape::from_dims(&[k, n]));
    let c = a.matmul(&b);

    let reference = c.realize_f32_reference();

    let cuda_device = fuel_graph_cuda::CudaDevice::new(0)
        .expect("cuda device 0 should be available");
    let mut cuda_exe = GraphExecutor::new(
        fuel_graph_cuda::CudaBackend::new(cuda_device),
    );
    let cuda_out = c.realize_f32_cuda(&mut cuda_exe);

    assert_eq!(reference.len(), cuda_out.len());
    assert_eq!(reference.len(), m * n);

    // Tight tolerance — matmul bit-parity per PR #6's individual
    // kernel suite. Any drift is gemm sum-order accumulation and is
    // well under 1e-4 at these shapes.
    fuel_core::test_utils::assert_allclose_f32(&cuda_out, &reference, 1e-4, 1e-4);
}

/// Deterministic LCG-backed tiny weights for a synthetic LLaMA.
/// Same recipe as `lazy::generate_tests::make_tiny_weights`, copied
/// here so this integration test is self-contained.
fn tiny_llama_weights(cfg: &LlamaConfig) -> LlamaWeights {
    let mut s: u32 = 9999;
    let mut next = || -> f32 {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
    };
    let mut vec_of = |n: usize| -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    };
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    LlamaWeights {
        token_embedding: vec_of(cfg.vocab_size * cfg.dim),
        layers: (0..cfg.n_layers).map(|_| LayerWeights {
            attn_q:         vec_of(cfg.dim * cfg.dim).into(),
            attn_q_bias:    None,
            attn_k:         vec_of(cfg.dim * kv_dim).into(),
            attn_k_bias:    None,
            attn_v:         vec_of(cfg.dim * kv_dim).into(),
            attn_v_bias:    None,
            attn_o:         vec_of(cfg.dim * cfg.dim).into(),
            ffn_gate:       vec_of(cfg.dim * cfg.ffn_dim).into(),
            ffn_up:         vec_of(cfg.dim * cfg.ffn_dim).into(),
            ffn_down:       vec_of(cfg.ffn_dim * cfg.dim).into(),
            attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            ffn_norm_gain:  Arc::from(vec![1.0; cfg.dim]),
        }).collect(),
        final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
        output:          vec_of(cfg.dim * cfg.vocab_size).into(),
    }
}

/// Phase 6b exit-criterion gate: a 2-layer LLaMA forward pass on
/// CUDA matches the reference backend within tolerance. Exercises
/// matmul, RoPE, softmax, RMS norm, and SwiGLU on CUDA all in one
/// graph — a real end-to-end anchor smoke vs a minimal subgraph.
#[test]
fn llama_2layer_cuda_matches_reference() {
    if !cuda_present() {
        eprintln!("skipping: no CUDA device visible to Fuel");
        return;
    }

    let cfg = LlamaConfig {
        vocab_size:     32,
        dim:            16,
        n_layers:       2,
        n_heads:        4,
        n_kv_heads:     2,
        head_dim:       4,
        ffn_dim:        32,
        norm_eps:       1e-5,
        rope_base:      10_000.0,
    };
    let weights = tiny_llama_weights(&cfg);
    let model = LlamaModel { config: cfg.clone(), weights };
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let logits: LazyTensor = model.forward(&tokens, 0);

    let reference = logits.realize_f32_reference();

    let cuda_device = fuel_graph_cuda::CudaDevice::new(0)
        .expect("cuda device 0 should be available");
    let mut cuda_exe = GraphExecutor::new(
        fuel_graph_cuda::CudaBackend::new(cuda_device),
    );
    let cuda_out = logits.realize_f32_cuda(&mut cuda_exe);

    assert_eq!(reference.len(), cuda_out.len());
    assert_eq!(reference.len(), tokens.len() * cfg.vocab_size);

    // 2-layer transformer accumulates gemm sum-order drift through
    // RMS norm, RoPE, attention's softmax + matmul, FFN's SwiGLU.
    // 5e-3 absorbs that drift while staying far below "wrong
    // backend implementation" levels.
    fuel_core::test_utils::assert_allclose_f32(&cuda_out, &reference, 5e-3, 5e-3);
}

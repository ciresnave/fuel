//! Phase 6a oracle gate: anchor-class CPU forward passes match the
//! reference backend within tolerance.
//!
//! Where `phase6b_cuda_anchor.rs` validates **CUDA executor vs reference**,
//! this file validates **CPU executor vs reference** — the same lazy
//! graphs realized via `LazyTensor::realize_f32()` (which goes through
//! the `fuel-graph-cpu` executor with gemm-backed matmul + reference
//! ops for everything else) must match the textbook reference.
//!
//! Why a separate test file: CI without GPU still runs this; it catches
//! CPU-executor regressions (a wrong dispatch arm, a broken Op::Reshape,
//! a missing ConvTranspose2D handler) that wouldn't otherwise surface
//! until somebody plugs in a model. Tolerance is tight (1e-4 rel) since
//! both sides are deterministic CPU code — the only drift is gemm
//! sum-order vs textbook nested-loop matmul.
//!
//! No feature gating, no skip-if-missing: this is a load-bearing CI
//! gate that must run on every PR.

use fuel_core::lazy::{LayerWeights, LazyTensor, LlamaConfig, LlamaModel, LlamaWeights};
use fuel_core::lazy_convnext::ConvNextModel;
use fuel_core_types::Shape;
use std::sync::Arc;

/// Compare CPU-executor forward against textbook reference. Tight
/// tolerance — both are deterministic CPU code; differences are
/// gemm sum-order drift only.
fn assert_cpu_oracle(t: &LazyTensor, atol: f32, rtol: f32) {
    let reference = t.realize_f32_reference();
    let cpu = t.realize_f32();
    assert_eq!(reference.len(), cpu.len(), "length mismatch");
    fuel_core::test_utils::assert_allclose_f32(&cpu, &reference, atol, rtol);
}

#[test]
fn single_matmul_cpu_matches_reference() {
    let (m, k, n) = (32usize, 48, 24);
    let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let a = LazyTensor::from_f32(a_data, Shape::from_dims(&[m, k]), &fuel_core::Device::cpu());
    let b = a.const_f32_like(b_data, Shape::from_dims(&[k, n]));
    let c = a.matmul(&b);
    assert_cpu_oracle(&c, 1e-4, 1e-4);
}

#[test]
fn dense_conv2d_cpu_matches_reference() {
    // Stride 1, padding 1, groups 1 — the SD VAE / standard conv shape.
    let (n, cin, h, w_sz) = (1usize, 3, 8, 8);
    let (cout, k, pad) = (4usize, 3, 1);
    let x_data: Vec<f32> = (0..(n * cin * h * w_sz))
        .map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let w_data: Vec<f32> = (0..(cout * cin * k * k))
        .map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[n, cin, h, w_sz]), &fuel_core::Device::cpu());
    let weight = x.const_f32_like(w_data, Shape::from_dims(&[cout, cin, k, k]));
    let y = x.conv2d(&weight, None, (1, 1), (pad, pad), 1);
    assert_cpu_oracle(&y, 1e-4, 1e-4);
}

#[test]
fn depthwise_conv2d_cpu_matches_reference() {
    // groups = c_in = c_out — ConvNeXt 7x7 depthwise.
    let (n, c, h, w_sz) = (1usize, 4, 6, 6);
    let (k, pad) = (3, 1);
    let x_data: Vec<f32> = (0..(n * c * h * w_sz))
        .map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let w_data: Vec<f32> = (0..(c * 1 * k * k))
        .map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[n, c, h, w_sz]), &fuel_core::Device::cpu());
    let weight = x.const_f32_like(w_data, Shape::from_dims(&[c, 1, k, k]));
    let y = x.conv2d(&weight, None, (1, 1), (pad, pad), c);
    assert_cpu_oracle(&y, 1e-4, 1e-4);
}

#[test]
fn conv_transpose2d_cpu_matches_reference() {
    // Stride-2 upsampler with 3x3 kernel, padding 1, output_padding 1
    // — SD UNet's upsampler shape.
    let (n, cin, h, w_sz) = (1usize, 3, 4, 4);
    let (cout, k) = (2usize, 3);
    let x_data: Vec<f32> = (0..(n * cin * h * w_sz))
        .map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let w_data: Vec<f32> = (0..(cin * cout * k * k))
        .map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[n, cin, h, w_sz]), &fuel_core::Device::cpu());
    let weight = x.const_f32_like(w_data, Shape::from_dims(&[cin, cout, k, k]));
    let y = x.conv_transpose2d(&weight, (2, 2), (1, 1), (1, 1), (1, 1), 1);
    assert_cpu_oracle(&y, 1e-4, 1e-4);
}

/// LCG-backed tiny LLaMA weights. Same recipe as
/// `phase6b_cuda_anchor::tiny_llama_weights`.
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

#[test]
fn llama_2layer_cpu_matches_reference() {
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
    // Tighter tolerance than the CUDA equivalent (5e-3) since both
    // sides are CPU; only gemm-vs-textbook drift to absorb.
    assert_cpu_oracle(&logits, 5e-4, 5e-4);
}

#[test]
fn convnext_cpu_matches_reference() {
    let cfg = fuel_core::lazy_convnext::tiny_cfg();
    let weights = fuel_core::lazy_convnext::zero_weights(&cfg);
    let model = ConvNextModel { weights, config: cfg.clone() };
    let image = vec![0.0_f32; cfg.in_channels * cfg.image_size * cfg.image_size];
    let logits = model.forward(&image);
    assert_cpu_oracle(&logits, 1e-4, 1e-4);
}

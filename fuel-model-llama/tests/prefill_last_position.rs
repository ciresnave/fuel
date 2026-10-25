// SPDX-License-Identifier: MIT OR Apache-2.0
//! Time-to-first-token (lightbulb report, 2026-10-25): a prefill needs the logits of the LAST
//! prompt position only, so the LM head (`[seq, dim] x [dim, vocab]`) must run on one row, not on
//! all `seq` rows. Measured before this fix on SmolLM2-135M, 140 tokens, CPU: that one f32 matmul
//! cost ~6 s of a 10.5 s prefill.
//!
//! Detector: `op_profile`'s launch trace records the byte length of each launch's first input.
//! The LM head is the LAST `MatMul` launch of a prefill, so its first input is `rows * dim * 4`
//! bytes. This file holds ONE test on purpose: the trace is process-global.

use std::sync::Arc;

use fuel_dispatch::compiled::op_profile;
use fuel_ir::DType;
use fuel_model_llama::{LlamaConfig, LlamaModel, LlamaWeights};
use fuel_tensor::Device;
use fuel_tensor::inference_context::{InferenceContext, KvCache};
use fuel_tensor::lazy::LayerWeights;

fn tiny_weights(cfg: &LlamaConfig, seed: u32) -> LlamaWeights {
    let mut s: u32 = seed;
    let mut next = || -> f32 {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
    };
    let mut vec_of = |n: usize| -> Arc<[f32]> { (0..n).map(|_| next()).collect::<Vec<_>>().into() };
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    LlamaWeights {
        instance: fuel_tensor::decode_shape::ModelInstanceId::next(),
        token_embedding: vec_of(cfg.vocab_size * cfg.dim),
        layers: (0..cfg.n_layers)
            .map(|_| LayerWeights {
                attn_q: vec_of(cfg.dim * cfg.dim).into(),
                attn_q_bias: None,
                attn_k: vec_of(cfg.dim * kv_dim).into(),
                attn_k_bias: None,
                attn_v: vec_of(cfg.dim * kv_dim).into(),
                attn_v_bias: None,
                attn_o: vec_of(cfg.dim * cfg.dim).into(),
                ffn_gate: vec_of(cfg.dim * cfg.ffn_dim).into(),
                ffn_up: vec_of(cfg.dim * cfg.ffn_dim).into(),
                ffn_down: vec_of(cfg.ffn_dim * cfg.dim).into(),
                attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                ffn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            })
            .collect(),
        final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
        output: vec_of(cfg.dim * cfg.vocab_size).into(),
    }
}

/// First-input byte length of the last `MatMul` launch recorded while `run` executes.
fn last_matmul_input_bytes(run: impl FnOnce()) -> usize {
    let _ = op_profile::take_trace();
    op_profile::set_trace(true);
    run();
    op_profile::set_trace(false);
    let rows = op_profile::take_trace();
    rows.iter()
        .rev()
        .find(|r| r.op == "MatMul")
        .map(|r| r.bytes)
        .expect("a prefill launches at least one MatMul")
}

const PROMPT_LEN: usize = 7;

fn tiny_model() -> LlamaModel {
    let cfg = LlamaConfig {
        vocab_size: 44,
        dim: 24,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 6,
        ffn_dim: 28,
        norm_eps: 1e-5,
        rope_base: 10000.0,
    };
    LlamaModel {
        weights: tiny_weights(&cfg, 9999),
        config: cfg,
    }
}

/// `forward()` returns logits for every position, so its LM head sees `PROMPT_LEN` rows: the control
/// that proves the detector can tell the two shapes apart. Returns `(LM-head input bytes, logits)`.
fn all_position_run(model: &LlamaModel, prompt: &[u32]) -> (usize, Vec<f32>) {
    let mut logits = Vec::new();
    let bytes = last_matmul_input_bytes(|| {
        logits = model
            .forward(prompt, 0)
            .expect("forward")
            .realize_f32()
            .to_vec();
    });
    (bytes, logits)
}

/// The prefill path (`forward_with_kv_context`). Returns `(LM-head input bytes, last-position logits)`.
fn prefill_run(model: &LlamaModel, prompt: &[u32]) -> (usize, Vec<f32>) {
    let cfg = &model.config;
    let dev = Device::cpu();
    let mut cache = KvCache::with_capacity(
        cfg.n_layers,
        cfg.n_kv_heads,
        cfg.head_dim,
        prompt.len(),
        DType::F32,
        &dev,
    )
    .expect("cache");
    let mut ctx = InferenceContext::new(dev);
    let mut logits = Vec::new();
    let bytes = last_matmul_input_bytes(|| {
        logits = model
            .forward_with_kv_context(prompt, &mut cache, &mut ctx)
            .expect("prefill");
    });
    (bytes, logits)
}

#[test]
fn prefill_runs_the_lm_head_on_the_last_position_only() {
    let model = tiny_model();
    let (dim, vocab) = (model.config.dim, model.config.vocab_size);
    let prompt: Vec<u32> = (1..=PROMPT_LEN as u32).collect();

    let (full_bytes, full_logits) = all_position_run(&model, &prompt);
    assert_eq!(
        full_bytes,
        PROMPT_LEN * dim * 4,
        "control: all-position head"
    );

    let (bytes, last) = prefill_run(&model, &prompt);
    assert_eq!(bytes, dim * 4, "prefill LM head must see exactly one row");

    // Same numbers: the last row of the all-position logits, bit for bit (CPU f32, row-wise
    // identical accumulation order).
    let want = &full_logits[(PROMPT_LEN - 1) * vocab..PROMPT_LEN * vocab];
    assert_eq!(last.len(), want.len());
    for (i, (a, b)) in last.iter().zip(want).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "logit {i}: {a} vs {b}");
    }
}

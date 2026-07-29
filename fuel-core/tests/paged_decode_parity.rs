//! Paged-decode parity (multi-session serving, paged-storage integration PS2):
//! `LlamaModel::forward_paged_step` — sessions' KV physically in `DeviceKvPool`
//! blocks, decoded via `Op::PagedAttn` — produces logits ε-close to the
//! contiguous `forward_with_kv_context` for the same token sequence.
//!
//! `Op::PagedAttn` is decode-only (Sq=1), so the paged path feeds EVERY token
//! (prompt + generated) one at a time; the contiguous path does a batched
//! prefill then decodes. They agree position-for-position because each token's
//! K/V depends only on tokens `0..=i` (causal), which are all resident by the
//! time token `i` is fed — the standard one-at-a-time ≡ batched-prefill identity.

use std::sync::Arc;

use fuel_core::Device;
use fuel_core::inference_context::{InferenceContext, KvCache};
use fuel_core::kv_block_pool::KvGeometry;
use fuel_core::kv_block_pool_device::DeviceKvPool;
use fuel_core::lazy::{LayerWeights, LlamaConfig, LlamaModel, LlamaWeights};
use fuel_ir::DType;

/// Deterministic tiny weights sized to `cfg` (mirrors lazy.rs's test builder).
fn tiny_weights(cfg: &LlamaConfig, seed: u32) -> LlamaWeights {
    let mut s: u32 = seed;
    let mut next = || -> f32 {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
    };
    let mut vec_of = |n: usize| -> Arc<[f32]> {
        let v: Vec<f32> = (0..n).map(|_| next()).collect();
        Arc::from(v)
    };
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    LlamaWeights {
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

fn assert_close(paged: &[f32], contig: &[f32], label: &str) {
    assert_eq!(paged.len(), contig.len(), "{label}: logits length");
    for (i, (&a, &b)) in paged.iter().zip(contig.iter()).enumerate() {
        let diff = (a - b).abs();
        let den = a.abs().max(b.abs()).max(f32::MIN_POSITIVE);
        assert!(
            diff < 1e-4 || diff / den < 1e-4,
            "{label}[{i}]: paged={a} contig={b} (abs={diff})",
        );
    }
}

fn parity_for(cfg: LlamaConfig) {
    let model = LlamaModel { config: cfg.clone(), weights: tiny_weights(&cfg, 9999) };
    let prompt = [1u32, 2, 3];
    let decode = [4u32, 5, 6];
    let all: Vec<u32> = prompt.iter().chain(decode.iter()).copied().collect();
    let max_seq_len = all.len();
    let dev = Device::cpu();

    // --- Contiguous reference: batched prefill, then decode one token/step. ---
    let mut cache = KvCache::with_capacity(
        cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, max_seq_len, DType::F32, &dev,
    ).unwrap();
    let mut ctx = InferenceContext::new(dev.clone());
    let mut contig: Vec<Vec<f32>> = Vec::new();
    contig.push(model.forward_with_kv_context(&prompt, &mut cache, &mut ctx).unwrap());
    for &t in &decode {
        contig.push(model.forward_with_kv_context(&[t], &mut cache, &mut ctx).unwrap());
    }

    // --- Paged: feed EVERY token one at a time into the pool. ---
    let geom = KvGeometry {
        n_layers: cfg.n_layers,
        num_blocks: 32,
        block_size: 4,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        elem_size: 4,
    };
    let mut pool = DeviceKvPool::new(geom, DType::F32, &dev).unwrap();
    let session = pool.core_mut().open();
    let mut paged: Vec<Vec<f32>> = Vec::new();
    for &t in &all {
        paged.push(model.forward_paged_step(t, &mut pool, session).unwrap());
    }

    // Contiguous prefill returns position P-1 (predict token after the prompt),
    // = paged's P-th step; then each decode position lines up.
    let p = prompt.len();
    assert_close(&paged[p - 1], &contig[0], "prefill-last");
    for i in 0..decode.len() {
        assert_close(&paged[p + i], &contig[1 + i], &format!("decode-{i}"));
    }
}

#[test]
fn paged_forward_matches_contiguous_no_gqa() {
    parity_for(LlamaConfig {
        vocab_size: 16,
        dim: 16,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 4, // no GQA
        head_dim: 4,
        ffn_dim: 16,
        norm_eps: 1e-5,
        rope_base: 10000.0,
    });
}

#[test]
fn paged_forward_matches_contiguous_gqa() {
    parity_for(LlamaConfig {
        vocab_size: 16,
        dim: 16,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2, // GQA: n_rep = 2
        head_dim: 4,
        ffn_dim: 16,
        norm_eps: 1e-5,
        rope_base: 10000.0,
    });
}

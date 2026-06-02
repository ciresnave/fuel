//! Llama2-C decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Llama2-C is Andrej Karpathy's stripped-down
//! Llama2 implementation (`llama2.c` repo) targeting tiny models
//! trained from scratch. Architecturally **identical to LLaMA**:
//! bias-free GQA + RmsNorm + SwiGLU FFN + RoPE. Only the field
//! names differ (`dim` ↔ `hidden_size`, `n_layers` ↔ `num_hidden_layers`,
//! etc.).
//!
//! Thin wrapper over [`crate::lazy::LlamaModel`] + adapter from
//! [`Llama2cConfig`] to [`crate::lazy::LlamaConfig`].

use crate::lazy::{LlamaConfig, LlamaModel, LlamaWeights, LazyTensor};
use crate::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct Llama2cConfig {
    /// Transformer dim (== `hidden_size` in HF).
    pub dim: usize,
    /// FFN hidden dim (== `intermediate_size` in HF).
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    /// `dim / n_heads`.
    pub head_dim: usize,
    pub norm_eps: f64,
    pub rope_theta: f64,
}

impl Llama2cConfig {
    pub fn from_dim(dim: usize, hidden_dim: usize, n_layers: usize, n_heads: usize, n_kv_heads: usize, vocab_size: usize) -> Self {
        Self {
            dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size,
            head_dim: dim / n_heads,
            norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    /// Convert to the [`LlamaConfig`] shape so the underlying lazy
    /// LLaMA model accepts it.
    pub fn to_llama_config(&self) -> LlamaConfig {
        LlamaConfig {
            vocab_size: self.vocab_size,
            dim:        self.dim,
            n_layers:   self.n_layers,
            n_heads:    self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim:   self.head_dim,
            ffn_dim:    self.hidden_dim,
            norm_eps:   self.norm_eps,
            rope_base:  self.rope_theta,
        }
    }
}

/// Llama2-C language model. Stores its own config naming for
/// safetensors-loader interop with the `llama2.c` checkpoint format;
/// the forward delegates to [`LlamaModel`].
#[derive(Debug, Clone)]
pub struct Llama2cModel {
    pub config: Llama2cConfig,
    pub weights: LlamaWeights,
}

impl Llama2cModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        llama.forward(tokens, start_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::{LayerWeights, WeightStorage};
    use fuel_core_types::Shape;
    use std::sync::Arc;

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = Llama2cConfig {
            dim: 16, hidden_dim: 32, n_layers: 2,
            n_heads: 4, n_kv_heads: 2, vocab_size: 32,
            head_dim: 4, norm_eps: 1e-5, rope_theta: 10_000.0,
        };
        let mut s: u32 = 99999;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.dim; let i = cfg.hidden_dim;
        let kv = cfg.n_kv_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.n_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)), attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let weights = LlamaWeights {
            token_embedding,
            layers,
            final_norm_gain: Arc::from(vec![1.0_f32; h]),
            output: WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb)),
        };
        let _ = Shape::from_dims(&[1, 3, cfg.vocab_size]); // unused; included for future debug.
        let model = Llama2cModel { config: cfg.clone(), weights };
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn config_field_mapping_matches_llama_config() {
        let cfg = Llama2cConfig::from_dim(64, 128, 4, 8, 2, 256);
        let l = cfg.to_llama_config();
        assert_eq!(l.dim, 64);
        assert_eq!(l.ffn_dim, 128);
        assert_eq!(l.n_layers, 4);
        assert_eq!(l.n_heads, 8);
        assert_eq!(l.n_kv_heads, 2);
        assert_eq!(l.head_dim, 8); // 64 / 8
        assert_eq!(l.vocab_size, 256);
    }
}

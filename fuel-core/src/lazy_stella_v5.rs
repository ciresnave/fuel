//! Stella-en-v5 (Dunzhang 2024) ported to the lazy-graph API.
//!
//! Stella v5 is a text-embedding model built on the Qwen2
//! architecture with the causal LM head replaced by a small
//! `embed_head` Linear projection. The output is mean-pooled
//! across the sequence (weighted by the attention mask) and
//! L2-normalized to unit length. Trained with [Matryoshka
//! Representation Learning](https://arxiv.org/abs/2205.13147)
//! so the same backbone supports multiple output dimensions
//! (256, 768, 1024, 2048, 4096, 6144, 8192) by selecting
//! different `embed_head` weights.
//!
//! v1 of the lazy port targets the 1.5B variant
//! (Qwen2-1.5B backbone, [Model card](https://huggingface.co/dunzhang/stella_en_1.5B_v5)).
//! The 400M variant uses a BERT-RoPE backbone with token-type
//! embeddings and absolute position scaling and is structurally
//! distinct — it would need its own port.
//!
//! # Composition pattern
//!
//! ```text
//!   tokens
//!     → Qwen2Model::forward_hidden  → hidden states (B, seq, hidden)
//!     → mean_pool(hidden, attention_mask)  → (B, hidden)
//!     → embed_head Linear(hidden, out_features)  → (B, out_features)
//!     → L2-normalize on last dim   → (B, out_features)
//! ```
//!
//! Embedding-model convention: the attention mask is `(B, seq)`
//! with `1` for real tokens and `0` for padding. Mean pooling
//! sums hidden states weighted by the mask and divides by the
//! mask sum.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. The lazy port assumes no
//! padding (all tokens are real) and skips the mask
//! construction; callers with padded inputs should pass an
//! explicit mask via `forward_with_mask`. The 1.5B variant
//! ships with seven canonical `embed_head` dimensions
//! enumerated by [`StellaEmbedDim`].

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_qwen2::{Qwen2Config, Qwen2Model, Qwen2Weights};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

/// Canonical output dimensions for Stella-en-v5's Matryoshka
/// head. Selecting a dim picks a specific `embed_head` weight
/// from the safetensors checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StellaEmbedDim {
    Dim256,
    Dim768,
    Dim1024,
    Dim2048,
    Dim4096,
    Dim6144,
    Dim8192,
}

impl StellaEmbedDim {
    pub fn out_features(&self) -> usize {
        match self {
            Self::Dim256 => 256,
            Self::Dim768 => 768,
            Self::Dim1024 => 1024,
            Self::Dim2048 => 2048,
            Self::Dim4096 => 4096,
            Self::Dim6144 => 6144,
            Self::Dim8192 => 8192,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StellaV5Config {
    pub backbone: Qwen2Config,
    pub embed_dim: StellaEmbedDim,
}

impl StellaV5Config {
    /// `dunzhang/stella_en_1.5B_v5` preset (approximate;
    /// actual `config.json` from HuggingFace overrides). The
    /// Qwen2 `head_dim` is derived from `hidden_size /
    /// num_attention_heads` → 1536 / 12 = 128.
    pub fn stella_en_1_5b_v5(embed_dim: StellaEmbedDim) -> Self {
        let backbone = Qwen2Config {
            vocab_size: 151_646,
            hidden_size: 1_536,
            intermediate_size: 8_960,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            max_position_embeddings: 131_072,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            sliding_window: 32_768,
            use_sliding_window: false,
            max_window_layers: 28,
            tie_word_embeddings: false,
        };
        Self { backbone, embed_dim }
    }
}

#[derive(Debug, Clone)]
pub struct StellaV5Weights {
    pub backbone: Qwen2Weights,
    /// `[hidden_size, out_features]`. No bias in the public release.
    pub embed_head: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct StellaV5Model {
    pub config: StellaV5Config,
    pub weights: StellaV5Weights,
}

impl StellaV5Model {
    /// Run a forward pass on un-padded input tokens. All tokens
    /// are weighted equally during mean pooling. Returns
    /// L2-normalized embeddings `(1, out_features)`.
    pub fn forward(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let seq = tokens.len();
        assert!(seq > 0, "StellaV5Model::forward: tokens must be non-empty");

        let backbone = Qwen2Model {
            config: cfg.backbone.clone(),
            weights: self.weights.backbone.clone(),
        };
        let hidden = backbone.forward_hidden(tokens, 0)?; // (1, seq, hidden)

        // Mean pool over the sequence dimension (no mask — all tokens valid).
        let pooled = hidden.mean_dim(1_usize)?; // (1, hidden)

        // Project through embed_head: hidden → out_features.
        let out_features = cfg.embed_dim.out_features();
        let projected = self.weights.embed_head.apply_linear(
            &pooled, cfg.backbone.hidden_size, out_features,
        );

        // L2-normalize on the last dim.
        l2_normalize(&projected)
    }

    /// Run a forward pass with a caller-supplied attention mask
    /// of shape `(seq,)` (1 for keep, 0 for pad). Mean pools the
    /// hidden states weighted by the mask before projection.
    pub fn forward_with_mask(
        &self,
        tokens: &[u32],
        attention_mask: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let seq = tokens.len();
        assert!(seq > 0, "StellaV5Model::forward_with_mask: tokens must be non-empty");
        assert_eq!(
            attention_mask.len(), seq,
            "attention_mask length must equal tokens length",
        );

        let backbone = Qwen2Model {
            config: cfg.backbone.clone(),
            weights: self.weights.backbone.clone(),
        };
        let hidden = backbone.forward_hidden(tokens, 0)?; // (1, seq, hidden)

        // Build the mask const + sum normalizer on the same graph as `hidden`.
        let mask_f32: Vec<f32> = attention_mask.iter().map(|&m| m as f32).collect();
        let sum_mask: f32 = mask_f32.iter().sum();
        assert!(sum_mask > 0.0, "attention_mask sum must be > 0");
        let mask_t = hidden
            .const_f32_like(Arc::<[f32]>::from(mask_f32), Shape::from_dims(&[seq]))
            .reshape(Shape::from_dims(&[1, seq, 1]))?;
        let masked = hidden.broadcast_mul(&mask_t)?; // (1, seq, hidden)
        // Sum over seq, then divide by mask sum.
        let summed = masked.sum_dim(1_usize)?; // (1, hidden)
        let pooled = summed.mul_scalar(1.0_f64 / sum_mask as f64);

        let out_features = cfg.embed_dim.out_features();
        let projected = self.weights.embed_head.apply_linear(
            &pooled, cfg.backbone.hidden_size, out_features,
        );
        l2_normalize(&projected)
    }
}

/// L2-normalize on the last dim of a rank-2 tensor `(B, D)`.
fn l2_normalize(x: &LazyTensor) -> Result<LazyTensor> {
    // ||x||_2 = sqrt(sum(x*x, last_dim, keepdim))
    let sq = x.mul(x)?;
    let summed = sq.sum_keepdim(1_usize)?;
    let norm = summed.sqrt();
    x.broadcast_div(&norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::LayerWeights;
    use crate::Device;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }

    fn tiny_backbone_cfg() -> Qwen2Config {
        // head_dim derived from hidden_size / num_attention_heads = 16/4 = 4.
        Qwen2Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4,
            num_key_value_heads: 2, max_position_embeddings: 32,
            rms_norm_eps: 1e-6, rope_theta: 1_000_000.0,
            sliding_window: 32, use_sliding_window: false,
            max_window_layers: 2,
            tie_word_embeddings: false,
        }
    }

    fn tiny_qwen2_weights(cfg: &Qwen2Config, nb: &mut dyn FnMut() -> f32) -> Qwen2Weights {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h, nb);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| LayerWeights {
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                attn_q: WeightStorage::F32(vec_of(h * q_dim, nb)),
                attn_q_bias: Some(vec_of(q_dim, nb)),
                attn_k: WeightStorage::F32(vec_of(h * kv_dim, nb)),
                attn_k_bias: Some(vec_of(kv_dim, nb)),
                attn_v: WeightStorage::F32(vec_of(h * kv_dim, nb)),
                attn_v_bias: Some(vec_of(kv_dim, nb)),
                attn_o: WeightStorage::F32(vec_of(q_dim * h, nb)),
                ffn_gate: WeightStorage::F32(vec_of(h * i, nb)),
                ffn_up: WeightStorage::F32(vec_of(h * i, nb)),
                ffn_down: WeightStorage::F32(vec_of(i * h, nb)),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
            })
            .collect();
        Qwen2Weights {
            token_embedding,
            layers,
            final_norm_gain: Arc::from(vec![1.0_f32; h]),
            output: WeightStorage::F32(vec_of(h * cfg.vocab_size, nb)),
        }
    }

    fn tiny_stella(out_dim: StellaEmbedDim, seed: u32) -> StellaV5Model {
        let mut nb = rng_seed(seed);
        let backbone_cfg = tiny_backbone_cfg();
        let backbone_weights = tiny_qwen2_weights(&backbone_cfg, &mut nb);
        let embed_head = WeightStorage::F32(vec_of(backbone_cfg.hidden_size * out_dim.out_features(), &mut nb));
        StellaV5Model {
            config: StellaV5Config { backbone: backbone_cfg, embed_dim: out_dim },
            weights: StellaV5Weights { backbone: backbone_weights, embed_head },
        }
    }

    #[test]
    fn forward_shape_and_l2_norm() {
        let model = tiny_stella(StellaEmbedDim::Dim256, 11);
        let tokens = [1_u32, 2, 3, 4, 5];
        let emb = model.forward(&tokens).unwrap();
        assert_eq!(emb.shape().dims(), &[1, 256]);
        let realized = emb.realize_f32();
        // L2 norm should be ~1.
        let norm_sq: f32 = realized.iter().map(|v| v * v).sum();
        assert!((norm_sq - 1.0).abs() < 1e-5,
            "L2 norm² expected ~1.0, got {norm_sq}");
    }

    /// All output dims produce the expected `(1, D)` shape with
    /// finite values.
    #[test]
    fn matryoshka_dims_produce_right_shape() {
        for dim in [
            StellaEmbedDim::Dim256,
            StellaEmbedDim::Dim768,
            StellaEmbedDim::Dim1024,
            StellaEmbedDim::Dim2048,
        ] {
            let model = tiny_stella(dim, 22);
            let tokens = [1_u32, 2, 3];
            let emb = model.forward(&tokens).unwrap();
            assert_eq!(emb.shape().dims(), &[1, dim.out_features()]);
            for &v in &emb.realize_f32() {
                assert!(v.is_finite(), "non-finite embedding at dim {:?}", dim);
            }
        }
    }

    /// `forward_with_mask` and `forward` agree when the mask is
    /// all-ones (no padding).
    #[test]
    fn forward_with_mask_all_ones_matches_forward() {
        let model = tiny_stella(StellaEmbedDim::Dim256, 33);
        let tokens = [1_u32, 2, 3, 4];
        let mask = [1_u32, 1, 1, 1];
        let a = model.forward(&tokens).unwrap().realize_f32();
        let b = model.forward_with_mask(&tokens, &mask).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff < 1e-5,
            "forward and forward_with_mask(all-ones) must agree, max_diff = {max_diff}");
    }

    /// Masking out the last token must change the pooled embedding.
    #[test]
    fn mask_zero_token_alters_embedding() {
        let model = tiny_stella(StellaEmbedDim::Dim256, 44);
        let tokens = [1_u32, 2, 3, 4];
        let mask_all = [1_u32, 1, 1, 1];
        let mask_last = [1_u32, 1, 1, 0];
        let a = model.forward_with_mask(&tokens, &mask_all).unwrap().realize_f32();
        let b = model.forward_with_mask(&tokens, &mask_last).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "masking out the last token must change the embedding, max_diff = {max_diff}");
    }

    /// EmbedDim mappings match the canonical Matryoshka schedule.
    #[test]
    fn embed_dim_mapping() {
        assert_eq!(StellaEmbedDim::Dim256.out_features(), 256);
        assert_eq!(StellaEmbedDim::Dim768.out_features(), 768);
        assert_eq!(StellaEmbedDim::Dim1024.out_features(), 1024);
        assert_eq!(StellaEmbedDim::Dim8192.out_features(), 8192);
    }

    fn _drop_unused_imports() {
        let _ = Device::cpu();
    }
}

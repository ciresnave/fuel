//! DistilBERT (distilled BERT) ported to the lazy-graph API.
//!
//! Sanh et al. 2019. Distilled, smaller, faster, lighter version
//! of BERT. Used for many encoder downstream tasks (sentence
//! embeddings, classification, NER, etc.). v1 ports the
//! `DistilBertModel` (embeddings + transformer encoder)
//! returning per-token hidden states.
//!
//! Architecture:
//!
//!   1. **Embeddings**: word + absolute position lookup, sum,
//!      LayerNorm. NO token type embeddings (vs. BERT).
//!   2. **TransformerBlock (post-LN)**: `out = LN(x +
//!      attn(x)); out = LN(out + ffn(out))`. The LayerNorm is
//!      applied AFTER the residual sum, matching the original
//!      BERT convention.
//!   3. **MultiHeadSelfAttention**: separate Q/K/V/O linears
//!      (NO fused QKV); Q scaled by `1/sqrt(head_dim)` at the
//!      Q projection (not at the score matmul); optional
//!      additive attention mask broadcast-added to scores.
//!   4. **FFN**: `Linear → activation (GELU/ReLU) → Linear`.
//!      No SwiGLU; no gating.
//!   5. **No CLS pooling head in the base model** — the per-
//!      token hidden states are returned. Downstream task
//!      heads (NSP, MLM, etc.) sit on top of the base model.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. The optional attention
//! mask is an additive `(1, 1, seq, seq)` tensor with `0` for
//! keep and `-inf` for mask. The caller is responsible for
//! constructing it (e.g., from a pad-token mask).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistilBertActivation {
    Gelu,
    Relu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DistilBertConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub hidden_dim: usize,
    pub activation: DistilBertActivation,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
}

impl DistilBertConfig {
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// `distilbert-base-uncased`.
    pub fn distilbert_base() -> Self {
        Self {
            vocab_size: 30522,
            dim: 768,
            n_layers: 6,
            n_heads: 12,
            hidden_dim: 3072,
            activation: DistilBertActivation::Gelu,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DistilBertLayerWeights {
    pub q_lin: WeightStorage,
    pub q_lin_bias: Arc<[f32]>,
    pub k_lin: WeightStorage,
    pub k_lin_bias: Arc<[f32]>,
    pub v_lin: WeightStorage,
    pub v_lin_bias: Arc<[f32]>,
    pub out_lin: WeightStorage,
    pub out_lin_bias: Arc<[f32]>,
    pub sa_ln_gain: Arc<[f32]>,
    pub sa_ln_bias: Arc<[f32]>,
    pub lin1: WeightStorage,
    pub lin1_bias: Arc<[f32]>,
    pub lin2: WeightStorage,
    pub lin2_bias: Arc<[f32]>,
    pub output_ln_gain: Arc<[f32]>,
    pub output_ln_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct DistilBertWeights {
    pub word_embedding: Arc<[f32]>,
    pub position_embedding: Arc<[f32]>,
    pub embed_ln_gain: Arc<[f32]>,
    pub embed_ln_bias: Arc<[f32]>,
    pub layers: Vec<DistilBertLayerWeights>,
}

#[derive(Debug, Clone)]
pub struct DistilBertModel {
    pub config: DistilBertConfig,
    pub weights: DistilBertWeights,
}

impl DistilBertModel {
    /// Run a forward pass with an optional additive attention
    /// mask of shape `(1, 1, seq, seq)`. Returns the per-token
    /// hidden states of shape `(1, seq, dim)`.
    pub fn forward(
        &self,
        tokens: &[u32],
        attention_mask: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert!(seq <= cfg.max_position_embeddings);

        // Embeddings: word + position, sum, LayerNorm.
        let word_emb_t = LazyTensor::from_f32(
            weights.word_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &Device::cpu(),
        );
        let token_ids = word_emb_t.const_u32_like(
            tokens.to_vec(),
            Shape::from_dims(&[seq]),
        );
        let word_embeds = word_emb_t
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))?;
        let pos_full = word_emb_t.const_f32_like(
            Arc::clone(&weights.position_embedding),
            Shape::from_dims(&[cfg.max_position_embeddings, cfg.dim]),
        );
        let pos_slice = pos_full
            .slice(0_usize, 0, seq)?
            .reshape(Shape::from_dims(&[1, seq, cfg.dim]))?;
        let pos_bc = pos_slice.broadcast_to(Shape::from_dims(&[batch, seq, cfg.dim]))?;
        let sum = word_embeds.add(&pos_bc)?;
        let mut h = sum.layer_norm_affine(std::sync::Arc::clone(&weights.embed_ln_gain), std::sync::Arc::clone(&weights.embed_ln_bias), cfg.layer_norm_eps)?;

        // Encoder blocks.
        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, attention_mask)?;
        }

        Ok(h)
    }

    /// Extract per-token features at the requested layer
    /// indices. Output shape per layer: `(1, seq, dim)`.
    /// Each captured tensor is the post-block hidden state
    /// (DistilBERT is post-LN throughout, so it's already
    /// fully normalized).
    ///
    /// Layer-id contract: 0-based, strictly increasing, all in
    /// `[0, n_layers)`. Mirrors the
    /// [`crate::lazy_bert::BertModel::forward_intermediate_layers`]
    /// hook contract.
    pub fn forward_intermediate_layers(
        &self,
        tokens: &[u32],
        layer_ids: &[usize],
        attention_mask: Option<&LazyTensor>,
    ) -> Result<Vec<LazyTensor>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert!(seq <= cfg.max_position_embeddings);
        assert!(!layer_ids.is_empty(), "layer_ids must not be empty");
        for w in layer_ids.windows(2) {
            assert!(w[0] < w[1], "layer_ids must be strictly increasing");
        }
        let depth = weights.layers.len();
        assert!(
            *layer_ids.last().unwrap() < depth,
            "layer_ids must all be in [0, n_layers = {depth})",
        );

        // Same embedding setup as `forward`.
        let word_emb_t = LazyTensor::from_f32(
            weights.word_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &Device::cpu(),
        );
        let token_ids = word_emb_t.const_u32_like(
            tokens.to_vec(),
            Shape::from_dims(&[seq]),
        );
        let word_embeds = word_emb_t
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))?;
        let pos_full = word_emb_t.const_f32_like(
            Arc::clone(&weights.position_embedding),
            Shape::from_dims(&[cfg.max_position_embeddings, cfg.dim]),
        );
        let pos_slice = pos_full
            .slice(0_usize, 0, seq)?
            .reshape(Shape::from_dims(&[1, seq, cfg.dim]))?;
        let pos_bc = pos_slice.broadcast_to(Shape::from_dims(&[batch, seq, cfg.dim]))?;
        let sum = word_embeds.add(&pos_bc)?;
        let mut h = sum.layer_norm_affine(std::sync::Arc::clone(&weights.embed_ln_gain), std::sync::Arc::clone(&weights.embed_ln_bias), cfg.layer_norm_eps)?;

        let mut out = Vec::with_capacity(layer_ids.len());
        let mut next_capture = 0;
        for (idx, layer) in weights.layers.iter().enumerate() {
            h = self.apply_layer(&h, layer, attention_mask)?;
            if next_capture < layer_ids.len() && layer_ids[next_capture] == idx {
                out.push(h.clone());
                next_capture += 1;
            }
        }
        Ok(out)
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &DistilBertLayerWeights,
        attention_mask: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = x.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let seq = dims[1];
        let d = cfg.dim;
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim();

        // ---- Self-attention -----------------------------------------------
        let q = layer.q_lin.apply_linear(x, d, d);
        let q = q.add_trailing_bias(std::sync::Arc::clone(&layer.q_lin_bias))?;
        let k = layer.k_lin.apply_linear(x, d, d);
        let k = k.add_trailing_bias(std::sync::Arc::clone(&layer.k_lin_bias))?;
        let v = layer.v_lin.apply_linear(x, d, d);
        let v = v.add_trailing_bias(std::sync::Arc::clone(&layer.v_lin_bias))?;

        let _ = (batch, seq);
        let q = q.split_heads(n_heads, head_dim)?;
        let k = k.split_heads(n_heads, head_dim)?;
        let v = v.split_heads(n_heads, head_dim)?;

        // Q-scale (matches eager: scaling applied to Q, not the matmul).
        let q_scaled = q.mul_scalar(1.0 / (head_dim as f64).sqrt());
        let k_t = k.transpose()?;
        let scores = q_scaled.matmul(&k_t)?;
        let scores = match attention_mask {
            None => scores,
            Some(mask) => scores.broadcast_add(mask)?,
        };
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx.merge_heads()?;
        let attn_out = layer.out_lin.apply_linear(&merged, d, d);
        let attn_out = attn_out.add_trailing_bias(std::sync::Arc::clone(&layer.out_lin_bias))?;

        // Post-LN: LN(attn + x).
        let h1 = x.add(&attn_out)?.layer_norm_affine(std::sync::Arc::clone(&layer.sa_ln_gain), std::sync::Arc::clone(&layer.sa_ln_bias), cfg.layer_norm_eps)?;

        // ---- FFN ----------------------------------------------------------
        let fc1 = layer.lin1.apply_linear(&h1, d, cfg.hidden_dim);
        let fc1 = fc1.add_trailing_bias(std::sync::Arc::clone(&layer.lin1_bias))?;
        let act = match cfg.activation {
            DistilBertActivation::Gelu => fc1.gelu_erf(),
            DistilBertActivation::Relu => fc1.relu(),
        };
        let fc2 = layer.lin2.apply_linear(&act, cfg.hidden_dim, d);
        let fc2 = fc2.add_trailing_bias(std::sync::Arc::clone(&layer.lin2_bias))?;

        // Post-LN: LN(ffn + h1).
        let h2 = h1.add(&fc2)?.layer_norm_affine(std::sync::Arc::clone(&layer.output_ln_gain), std::sync::Arc::clone(&layer.output_ln_bias), cfg.layer_norm_eps)?;
        Ok(h2)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_cfg() -> DistilBertConfig {
        DistilBertConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            hidden_dim: 32,
            activation: DistilBertActivation::Gelu,
            max_position_embeddings: 8,
            layer_norm_eps: 1e-12,
        }
    }

    fn tiny_weights(cfg: &DistilBertConfig) -> DistilBertWeights {
        let mut s: u32 = 27272;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let d = cfg.dim;
        let word_embedding = vec_of(cfg.vocab_size * d, &mut *nb);
        let position_embedding = vec_of(cfg.max_position_embeddings * d, &mut *nb);
        let embed_ln_gain = Arc::from(vec![1.0_f32; d]);
        let embed_ln_bias = Arc::from(vec![0.0_f32; d]);

        let layers: Vec<DistilBertLayerWeights> = (0..cfg.n_layers).map(|_| DistilBertLayerWeights {
            q_lin: WeightStorage::F32(vec_of(d * d, &mut *nb)),
            q_lin_bias: vec_of(d, &mut *nb),
            k_lin: WeightStorage::F32(vec_of(d * d, &mut *nb)),
            k_lin_bias: vec_of(d, &mut *nb),
            v_lin: WeightStorage::F32(vec_of(d * d, &mut *nb)),
            v_lin_bias: vec_of(d, &mut *nb),
            out_lin: WeightStorage::F32(vec_of(d * d, &mut *nb)),
            out_lin_bias: vec_of(d, &mut *nb),
            sa_ln_gain: Arc::from(vec![1.0_f32; d]),
            sa_ln_bias: Arc::from(vec![0.0_f32; d]),
            lin1: WeightStorage::F32(vec_of(d * cfg.hidden_dim, &mut *nb)),
            lin1_bias: vec_of(cfg.hidden_dim, &mut *nb),
            lin2: WeightStorage::F32(vec_of(cfg.hidden_dim * d, &mut *nb)),
            lin2_bias: vec_of(d, &mut *nb),
            output_ln_gain: Arc::from(vec![1.0_f32; d]),
            output_ln_bias: Arc::from(vec![0.0_f32; d]),
        }).collect();

        DistilBertWeights {
            word_embedding, position_embedding,
            embed_ln_gain, embed_ln_bias,
            layers,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_cfg();
        let model = DistilBertModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens = [1_u32, 2, 3, 4];
        let out = model.forward(&tokens, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.dim]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// Bidirectional attention: changing any token affects ALL
    /// positions' hidden states. Verify by changing the last
    /// token and checking that position 0's output differs.
    #[test]
    fn bidirectional_attention() {
        let cfg = tiny_cfg();
        let model = DistilBertModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let toks_a = [1_u32, 2, 3, 4];
        let toks_b = [1_u32, 2, 3, 15];
        let a = model.forward(&toks_a, None).unwrap().realize_f32();
        let b = model.forward(&toks_b, None).unwrap().realize_f32();
        let d = cfg.dim;
        // Compare position 0 (which should be affected by the last-token change).
        let mut max_diff = 0.0_f32;
        for i in 0..d {
            max_diff = max_diff.max((a[i] - b[i]).abs());
        }
        assert!(max_diff > 1e-6,
            "last-token change must affect position 0 (bidirectional), max_diff = {max_diff}");
    }

    /// Position-embedding lookup is wired: changing only the
    /// position embedding row corresponding to position 0
    /// alters the output at position 0.
    #[test]
    fn position_embedding_is_wired() {
        let cfg = tiny_cfg();
        let mut base = tiny_weights(&cfg);
        let original_pos_embed = (*base.position_embedding).to_vec();
        // Modify the first position's embedding.
        let mut modified = original_pos_embed.clone();
        for i in 0..cfg.dim {
            modified[i] = 1.0;
        }
        let modified_pos = Arc::from(modified);
        let mut model_zero = base.clone();
        model_zero.position_embedding = modified_pos;
        // base keeps the original pos embed.
        base.position_embedding = Arc::from(original_pos_embed);

        let m_a = DistilBertModel { config: cfg.clone(), weights: base };
        let m_b = DistilBertModel { config: cfg, weights: model_zero };
        let toks = [1_u32, 2, 3, 4];
        let a = m_a.forward(&toks, None).unwrap().realize_f32();
        let b = m_b.forward(&toks, None).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "position-embed change must alter output, max_diff = {max_diff}");
    }

    /// `forward_intermediate_layers` returns one tensor per
    /// requested layer index, each shaped `(1, seq, dim)`.
    #[test]
    fn forward_intermediate_layers_shape() {
        let cfg = tiny_cfg();
        let model = DistilBertModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens = [1_u32, 2, 3, 4];
        let outs = model.forward_intermediate_layers(&tokens, &[0_usize, 1], None).unwrap();
        assert_eq!(outs.len(), 2);
        for out in &outs {
            assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.dim]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite intermediate: {v}");
            }
        }
        // Layer 0 and layer 1 outputs must differ.
        let a = outs[0].realize_f32();
        let b = outs[1].realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "layer 0 and layer 1 intermediates must differ, max_diff = {max_diff}");
    }
}

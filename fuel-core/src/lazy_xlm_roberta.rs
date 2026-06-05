//! XLM-RoBERTa (multilingual BERT variant) ported to the
//! lazy-graph API.
//!
//! Conneau et al. 2019. RoBERTa pre-training procedure applied
//! to multilingual data. Used widely for multilingual encoder
//! tasks (classification, NER, retrieval). v1 ports the
//! `XLMRobertaModel` returning per-token hidden states.
//!
//! Differences from DistilBERT:
//!
//!   1. **Token type embeddings** (eager `type_vocab_size = 1`
//!      for XLM-R, but the embedding is still indexed — by 0
//!      for all positions). Carried as a learned
//!      `[type_vocab_size, hidden]` table.
//!   2. **RoBERTa position-id convention**:
//!      `position_ids = padding_idx + 1 + [0, 1, 2, ...]`
//!      (cumulative-count of non-pad tokens, then offset by
//!      `padding_idx + 1`). v1 assumes no padding inside the
//!      sequence — all input tokens are non-pad — so
//!      `position_ids` is a simple arithmetic progression
//!      starting at `padding_idx + 1`.
//!   3. **Post-LN sublayer structure (BERT-shape)**: same as
//!      DistilBERT. `out = LN(x + attn(x)); out = LN(out +
//!      ffn(out))`.
//!   4. **Separate Q/K/V/O linears** (no fused QKV).
//!
//! Otherwise identical to DistilBERT structurally.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. Cross-attention path
//! (`encoder_hidden_states`) and KV cache (`past_key_value`)
//! both deferred. Optional additive attention mask of shape
//! `(1, 1, seq, seq)` for padding masking.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XlmrActivation {
    Gelu,
    GeluPytorchTanh,
    Relu,
    Silu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct XlmrConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    /// XLM-R uses `type_vocab_size = 1` (always pass token type 0).
    pub type_vocab_size: usize,
    pub hidden_activation: XlmrActivation,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
}

impl XlmrConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    /// `xlm-roberta-base` preset.
    pub fn xlm_roberta_base() -> Self {
        Self {
            vocab_size: 250002,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 514,
            type_vocab_size: 1,
            hidden_activation: XlmrActivation::Gelu,
            layer_norm_eps: 1e-5,
            pad_token_id: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct XlmrLayerWeights {
    pub q_proj: WeightStorage,
    pub q_proj_bias: Arc<[f32]>,
    pub k_proj: WeightStorage,
    pub k_proj_bias: Arc<[f32]>,
    pub v_proj: WeightStorage,
    pub v_proj_bias: Arc<[f32]>,
    pub out_proj: WeightStorage,
    pub out_proj_bias: Arc<[f32]>,
    pub attn_ln_gain: Arc<[f32]>,
    pub attn_ln_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
    pub ffn_ln_gain: Arc<[f32]>,
    pub ffn_ln_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct XlmrWeights {
    pub word_embedding: Arc<[f32]>,
    pub position_embedding: Arc<[f32]>,
    pub token_type_embedding: Arc<[f32]>,
    pub embed_ln_gain: Arc<[f32]>,
    pub embed_ln_bias: Arc<[f32]>,
    pub layers: Vec<XlmrLayerWeights>,
}

#[derive(Debug, Clone)]
pub struct XlmrModel {
    pub config: XlmrConfig,
    pub weights: XlmrWeights,
}

impl XlmrModel {
    /// Run a forward pass and return per-token hidden states
    /// of shape `(1, seq, hidden_size)`.
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
        // RoBERTa position-id offset by `padding_idx + 1`.
        assert!(
            seq + cfg.pad_token_id as usize + 1 <= cfg.max_position_embeddings,
            "seq + padding_idx + 1 ({}) exceeds max_position_embeddings ({})",
            seq + cfg.pad_token_id as usize + 1, cfg.max_position_embeddings,
        );

        // ---- Embeddings: word + position + token_type, sum, LayerNorm ---
        let word_emb_t = LazyTensor::from_f32(
            weights.word_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = word_emb_t.const_u32_like(
            tokens.to_vec(),
            Shape::from_dims(&[seq]),
        );
        let word_embeds = word_emb_t
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;

        // RoBERTa positions: padding_idx + 1, padding_idx + 2, ..., padding_idx + seq.
        let pad_off = cfg.pad_token_id as usize + 1;
        let position_ids_vec: Vec<u32> = (0..seq).map(|i| (pad_off + i) as u32).collect();
        let pos_full = word_emb_t.const_f32_like(
            Arc::clone(&weights.position_embedding),
            Shape::from_dims(&[cfg.max_position_embeddings, cfg.hidden_size]),
        );
        let pos_ids = word_emb_t.const_u32_like(
            position_ids_vec,
            Shape::from_dims(&[seq]),
        );
        let pos_embeds = pos_full
            .index_select(0_usize, &pos_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;

        // Token type embeddings (XLM-R always uses type 0).
        let tt_full = word_emb_t.const_f32_like(
            Arc::clone(&weights.token_type_embedding),
            Shape::from_dims(&[cfg.type_vocab_size, cfg.hidden_size]),
        );
        let tt_ids = word_emb_t.const_u32_like(
            vec![0_u32; seq],
            Shape::from_dims(&[seq]),
        );
        let tt_embeds = tt_full
            .index_select(0_usize, &tt_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;

        let sum = word_embeds.add(&pos_embeds)?.add(&tt_embeds)?;
        let mut h = sum.layer_norm_affine(std::sync::Arc::clone(&weights.embed_ln_gain), std::sync::Arc::clone(&weights.embed_ln_bias), cfg.layer_norm_eps)?;

        // ---- Encoder layers --------------------------------------------
        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, attention_mask)?;
        }
        Ok(h)
    }

    /// Extract per-token features at the requested layer
    /// indices. Output shape per layer:
    /// `(1, seq, hidden_size)`. Each captured tensor is the
    /// post-block hidden state (XLM-R is post-LN throughout
    /// like BERT, so the captures are already fully normalized).
    ///
    /// Mirrors
    /// [`crate::lazy_bert::BertModel::forward_intermediate_layers`]
    /// and the DistilBERT hook. Useful for multilingual
    /// layer-wise probing, multi-layer feature fusion on
    /// `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`-shape
    /// checkpoints, and cross-lingual distillation.
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
        assert!(
            seq + cfg.pad_token_id as usize + 1 <= cfg.max_position_embeddings,
            "seq + padding_idx + 1 ({}) exceeds max_position_embeddings ({})",
            seq + cfg.pad_token_id as usize + 1, cfg.max_position_embeddings,
        );
        assert!(!layer_ids.is_empty(), "layer_ids must not be empty");
        for w in layer_ids.windows(2) {
            assert!(w[0] < w[1], "layer_ids must be strictly increasing");
        }
        let depth = weights.layers.len();
        assert!(
            *layer_ids.last().unwrap() < depth,
            "layer_ids must all be in [0, num_hidden_layers = {depth})",
        );

        // Same embedding setup as `forward`.
        let word_emb_t = LazyTensor::from_f32(
            weights.word_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = word_emb_t.const_u32_like(
            tokens.to_vec(),
            Shape::from_dims(&[seq]),
        );
        let word_embeds = word_emb_t
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;
        let pad_off = cfg.pad_token_id as usize + 1;
        let position_ids_vec: Vec<u32> = (0..seq).map(|i| (pad_off + i) as u32).collect();
        let pos_full = word_emb_t.const_f32_like(
            Arc::clone(&weights.position_embedding),
            Shape::from_dims(&[cfg.max_position_embeddings, cfg.hidden_size]),
        );
        let pos_ids = word_emb_t.const_u32_like(
            position_ids_vec, Shape::from_dims(&[seq]),
        );
        let pos_embeds = pos_full
            .index_select(0_usize, &pos_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;
        let tt_full = word_emb_t.const_f32_like(
            Arc::clone(&weights.token_type_embedding),
            Shape::from_dims(&[cfg.type_vocab_size, cfg.hidden_size]),
        );
        let tt_ids = word_emb_t.const_u32_like(
            vec![0_u32; seq], Shape::from_dims(&[seq]),
        );
        let tt_embeds = tt_full
            .index_select(0_usize, &tt_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;
        let sum = word_embeds.add(&pos_embeds)?.add(&tt_embeds)?;
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
        layer: &XlmrLayerWeights,
        attention_mask: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = x.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let seq = dims[1];
        let d = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();

        let q = layer.q_proj.apply_linear(x, d, d);
        let q = q.add_trailing_bias(std::sync::Arc::clone(&layer.q_proj_bias))?;
        let k = layer.k_proj.apply_linear(x, d, d);
        let k = k.add_trailing_bias(std::sync::Arc::clone(&layer.k_proj_bias))?;
        let v = layer.v_proj.apply_linear(x, d, d);
        let v = v.add_trailing_bias(std::sync::Arc::clone(&layer.v_proj_bias))?;

        let _ = (batch, seq);
        let q = q.split_heads(n_heads, head_dim)?;
        let k = k.split_heads(n_heads, head_dim)?;
        let v = v.split_heads(n_heads, head_dim)?;

        let k_t = k.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k_t)?.mul_scalar(scale);
        let scores = match attention_mask {
            None => scores,
            Some(mask) => scores.broadcast_add(mask)?,
        };
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx.merge_heads()?;
        let attn_out = layer.out_proj.apply_linear(&merged, d, d);
        let attn_out = attn_out.add_trailing_bias(std::sync::Arc::clone(&layer.out_proj_bias))?;

        // Post-LN: LN(attn + x).
        let h1 = x.add(&attn_out)?.layer_norm_affine(std::sync::Arc::clone(&layer.attn_ln_gain), std::sync::Arc::clone(&layer.attn_ln_bias), cfg.layer_norm_eps)?;

        // FFN.
        let fc1 = layer.fc1.apply_linear(&h1, d, cfg.intermediate_size);
        let fc1 = fc1.add_trailing_bias(std::sync::Arc::clone(&layer.fc1_bias))?;
        let act = match cfg.hidden_activation {
            XlmrActivation::Gelu => fc1.gelu_erf(),
            XlmrActivation::GeluPytorchTanh => fc1.gelu(),
            XlmrActivation::Relu => fc1.relu(),
            XlmrActivation::Silu => fc1.silu(),
        };
        let fc2 = layer.fc2.apply_linear(&act, cfg.intermediate_size, d);
        let fc2 = fc2.add_trailing_bias(std::sync::Arc::clone(&layer.fc2_bias))?;

        // Post-LN: LN(ffn + h1).
        Ok(h1.add(&fc2)?.layer_norm_affine(std::sync::Arc::clone(&layer.ffn_ln_gain), std::sync::Arc::clone(&layer.ffn_ln_bias), cfg.layer_norm_eps)?)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_cfg() -> XlmrConfig {
        XlmrConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 32,
            max_position_embeddings: 16,
            type_vocab_size: 1,
            hidden_activation: XlmrActivation::Gelu,
            layer_norm_eps: 1e-12,
            pad_token_id: 1,
        }
    }

    fn tiny_weights(cfg: &XlmrConfig) -> XlmrWeights {
        let mut s: u32 = 31313;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let d = cfg.hidden_size;
        let word_embedding = vec_of(cfg.vocab_size * d, &mut *nb);
        let position_embedding = vec_of(cfg.max_position_embeddings * d, &mut *nb);
        let token_type_embedding = vec_of(cfg.type_vocab_size * d, &mut *nb);
        let embed_ln_gain = Arc::from(vec![1.0_f32; d]);
        let embed_ln_bias = Arc::from(vec![0.0_f32; d]);

        let layers: Vec<XlmrLayerWeights> = (0..cfg.num_hidden_layers).map(|_| XlmrLayerWeights {
            q_proj: WeightStorage::F32(vec_of(d * d, &mut *nb)),
            q_proj_bias: vec_of(d, &mut *nb),
            k_proj: WeightStorage::F32(vec_of(d * d, &mut *nb)),
            k_proj_bias: vec_of(d, &mut *nb),
            v_proj: WeightStorage::F32(vec_of(d * d, &mut *nb)),
            v_proj_bias: vec_of(d, &mut *nb),
            out_proj: WeightStorage::F32(vec_of(d * d, &mut *nb)),
            out_proj_bias: vec_of(d, &mut *nb),
            attn_ln_gain: Arc::from(vec![1.0_f32; d]),
            attn_ln_bias: Arc::from(vec![0.0_f32; d]),
            fc1: WeightStorage::F32(vec_of(d * cfg.intermediate_size, &mut *nb)),
            fc1_bias: vec_of(cfg.intermediate_size, &mut *nb),
            fc2: WeightStorage::F32(vec_of(cfg.intermediate_size * d, &mut *nb)),
            fc2_bias: vec_of(d, &mut *nb),
            ffn_ln_gain: Arc::from(vec![1.0_f32; d]),
            ffn_ln_bias: Arc::from(vec![0.0_f32; d]),
        }).collect();

        XlmrWeights {
            word_embedding, position_embedding,
            token_type_embedding,
            embed_ln_gain, embed_ln_bias,
            layers,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_cfg();
        let model = XlmrModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens = [1_u32, 2, 3, 4];
        let out = model.forward(&tokens, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// Token type embedding is wired: changing it must alter
    /// the output.
    #[test]
    fn token_type_embedding_is_wired() {
        let cfg = tiny_cfg();
        let base = tiny_weights(&cfg);
        let mut modified = base.clone();
        let new_tt: Vec<f32> = (0..cfg.type_vocab_size * cfg.hidden_size).map(|_| 0.5).collect();
        modified.token_type_embedding = Arc::from(new_tt);
        let m_base = XlmrModel { config: cfg.clone(), weights: base };
        let m_mod = XlmrModel { config: cfg.clone(), weights: modified };
        let toks = [1_u32, 2, 3];
        let a = m_base.forward(&toks, None).unwrap().realize_f32();
        let b = m_mod.forward(&toks, None).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "token type embedding must affect output, max_diff = {max_diff}");
    }

    /// RoBERTa position-id convention: position_ids start at
    /// `padding_idx + 1`. Verify by checking that modifying the
    /// position embedding row at `padding_idx + 1` alters the
    /// output (proves we're reading the right row).
    #[test]
    fn roberta_position_offset() {
        let cfg = tiny_cfg();
        let base = tiny_weights(&cfg);
        let mut modified = base.clone();
        // Modify the position embedding at row padding_idx + 1
        // (= 2 for pad_token_id=1). This is the row index that
        // our forward should read for token at position 0.
        let target_row = cfg.pad_token_id as usize + 1;
        let mut pe_vec = (*base.position_embedding).to_vec();
        for i in 0..cfg.hidden_size {
            pe_vec[target_row * cfg.hidden_size + i] = 1.0;
        }
        modified.position_embedding = Arc::from(pe_vec);
        let m_base = XlmrModel { config: cfg.clone(), weights: base };
        let m_mod = XlmrModel { config: cfg.clone(), weights: modified };
        let toks = [1_u32, 2, 3];
        let a = m_base.forward(&toks, None).unwrap().realize_f32();
        let b = m_mod.forward(&toks, None).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "RoBERTa position offset (modifying row pad_idx+1) must affect output, max_diff = {max_diff}");
    }

    /// `forward_intermediate_layers` returns one tensor per
    /// requested layer index, each shaped `(1, seq, hidden_size)`.
    #[test]
    fn forward_intermediate_layers_shape() {
        let cfg = tiny_cfg();
        let model = XlmrModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let toks = [1_u32, 2, 3];
        let outs = model.forward_intermediate_layers(&toks, &[0_usize, 1], None).unwrap();
        assert_eq!(outs.len(), 2);
        for out in &outs {
            assert_eq!(out.shape().dims(), &[1, toks.len(), cfg.hidden_size]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite intermediate: {v}");
            }
        }
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

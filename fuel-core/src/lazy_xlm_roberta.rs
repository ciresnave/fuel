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
use fuel_ir::Shape;
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

// ---- HuggingFace safetensors loader ----------------------------------------

impl XlmrWeights {
    /// Load XLM-RoBERTa (xlm-roberta-{base,large}) weights from a HuggingFace
    /// safetensors file. Naming follows the upstream RoBERTa layout at
    /// `roberta.embeddings.*` / `roberta.encoder.layer.{i}.*`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &XlmrConfig,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype as ltm};
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;

        // Some XLM-R checkpoints prefix with "roberta." and some don't;
        // probe both, prefer the prefixed form (standard HF).
        let prefix = if load_tensor_as_f32(st, "roberta.embeddings.word_embeddings.weight").is_ok() {
            "roberta."
        } else {
            ""
        };

        let word_embedding = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}embeddings.word_embeddings.weight"),
        )?);
        let position_embedding = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}embeddings.position_embeddings.weight"),
        )?);
        let token_type_embedding = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}embeddings.token_type_embeddings.weight"),
        )?);
        let embed_ln_gain = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}embeddings.LayerNorm.weight"),
        )?);
        let embed_ln_bias = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}embeddings.LayerNorm.bias"),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("{prefix}encoder.layer.{i}");
            let q_proj = ltm(st, &format!("{p}.attention.self.query.weight"), h, h)?;
            let q_proj_bias = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.attention.self.query.bias"),
            )?);
            let k_proj = ltm(st, &format!("{p}.attention.self.key.weight"), h, h)?;
            let k_proj_bias = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.attention.self.key.bias"),
            )?);
            let v_proj = ltm(st, &format!("{p}.attention.self.value.weight"), h, h)?;
            let v_proj_bias = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.attention.self.value.bias"),
            )?);
            let out_proj = ltm(st, &format!("{p}.attention.output.dense.weight"), h, h)?;
            let out_proj_bias = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.attention.output.dense.bias"),
            )?);
            let attn_ln_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.attention.output.LayerNorm.weight"),
            )?);
            let attn_ln_bias = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.attention.output.LayerNorm.bias"),
            )?);
            let fc1 = ltm(st, &format!("{p}.intermediate.dense.weight"), inter, h)?;
            let fc1_bias = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.intermediate.dense.bias"),
            )?);
            let fc2 = ltm(st, &format!("{p}.output.dense.weight"), h, inter)?;
            let fc2_bias = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.output.dense.bias"),
            )?);
            let ffn_ln_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.output.LayerNorm.weight"),
            )?);
            let ffn_ln_bias = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.output.LayerNorm.bias"),
            )?);
            layers.push(XlmrLayerWeights {
                q_proj, q_proj_bias, k_proj, k_proj_bias,
                v_proj, v_proj_bias, out_proj, out_proj_bias,
                attn_ln_gain, attn_ln_bias,
                fc1, fc1_bias, fc2, fc2_bias, ffn_ln_gain, ffn_ln_bias,
            });
        }

        Ok(Self {
            word_embedding, position_embedding, token_type_embedding,
            embed_ln_gain, embed_ln_bias, layers,
        })
    }
}

// ---- Masked LM head --------------------------------------------------------

/// Weights for the XLM-RoBERTa masked-language-model head.
///
/// HuggingFace layout (`xlm-roberta-base` / `xlm-roberta-large`):
///
///   - `lm_head.dense.weight`        `[hidden, hidden]`
///   - `lm_head.dense.bias`          `[hidden]`
///   - `lm_head.layer_norm.weight`   `[hidden]`
///   - `lm_head.layer_norm.bias`     `[hidden]`
///   - `lm_head.decoder.weight`      `[vocab, hidden]`
///   - `lm_head.decoder.bias`        `[vocab]`
///
/// The eager port tied `lm_head.decoder.weight` to
/// `roberta.embeddings.word_embeddings.weight` transposed. v1 of the
/// lazy port carries the decoder explicitly (HF checkpoints store the
/// tensor, so loading it is the simpler path), at the cost of a small
/// memory duplication. A `tied`-mode follow-up is welcome.
#[derive(Debug, Clone)]
pub struct ForMaskedLMWeights {
    pub lm_head_dense_weight:   WeightStorage,
    pub lm_head_dense_bias:     Arc<[f32]>,
    pub lm_head_ln_gain:        Arc<[f32]>,
    pub lm_head_ln_bias:        Arc<[f32]>,
    pub lm_head_decoder_weight: WeightStorage,
    pub lm_head_decoder_bias:   Arc<[f32]>,
}

/// XLM-RoBERTa with a masked-language-model head on top of the base
/// encoder. Output shape: `(1, seq, vocab_size)`.
#[derive(Debug, Clone)]
pub struct XlmrForMaskedLM {
    pub base:                   XlmrModel,
    pub lm_head_dense_weight:   WeightStorage,
    pub lm_head_dense_bias:     Arc<[f32]>,
    pub lm_head_ln_gain:        Arc<[f32]>,
    pub lm_head_ln_bias:        Arc<[f32]>,
    pub lm_head_decoder_weight: WeightStorage,
    pub lm_head_decoder_bias:   Arc<[f32]>,
}

impl XlmrForMaskedLM {
    /// Run a forward pass and return masked-LM logits over the vocab,
    /// shape `(1, seq, vocab_size)`.
    pub fn forward(
        &self,
        tokens: &[u32],
        attention_mask: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        let cfg = &self.base.config;
        let h = self.base.forward(tokens, attention_mask)?;

        // lm_head: dense -> gelu -> layer_norm -> decoder.
        // `apply_linear` matmuls a `(1, seq, hidden)` activation by a
        // `(hidden, hidden)` weight to give `(1, seq, hidden)`.
        let dense = self.lm_head_dense_weight
            .apply_linear(&h, cfg.hidden_size, cfg.hidden_size);
        let dense = dense.add_trailing_bias(Arc::clone(&self.lm_head_dense_bias))?;
        let act = dense.gelu_erf();
        let normed = act.layer_norm_affine(
            Arc::clone(&self.lm_head_ln_gain),
            Arc::clone(&self.lm_head_ln_bias),
            cfg.layer_norm_eps,
        )?;
        let logits = self.lm_head_decoder_weight
            .apply_linear(&normed, cfg.hidden_size, cfg.vocab_size);
        let logits = logits.add_trailing_bias(Arc::clone(&self.lm_head_decoder_bias))?;
        Ok(logits)
    }
}

impl ForMaskedLMWeights {
    /// Load the MLM-head tensors from a HuggingFace safetensors file
    /// using the standard `lm_head.*` naming. Tensor dtypes follow the
    /// existing XlmrWeights loader (F32 vectors via `load_tensor_as_f32`,
    /// transposed matrices via `load_transposed_matrix_preserve_dtype`).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &XlmrConfig,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype as ltm};
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;
        let lm_head_dense_weight = ltm(st, "lm_head.dense.weight", h, h)?;
        let lm_head_dense_bias = Arc::from(load_tensor_as_f32(
            st, "lm_head.dense.bias",
        )?);
        let lm_head_ln_gain = Arc::from(load_tensor_as_f32(
            st, "lm_head.layer_norm.weight",
        )?);
        let lm_head_ln_bias = Arc::from(load_tensor_as_f32(
            st, "lm_head.layer_norm.bias",
        )?);
        let lm_head_decoder_weight = ltm(st, "lm_head.decoder.weight", v, h)?;
        let lm_head_decoder_bias = Arc::from(load_tensor_as_f32(
            st, "lm_head.decoder.bias",
        )?);
        Ok(Self {
            lm_head_dense_weight,
            lm_head_dense_bias,
            lm_head_ln_gain,
            lm_head_ln_bias,
            lm_head_decoder_weight,
            lm_head_decoder_bias,
        })
    }
}

impl XlmrForMaskedLM {
    /// Load the full MaskedLM model (base encoder + lm_head) from a
    /// HuggingFace safetensors file in one call.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: XlmrConfig,
    ) -> Result<Self> {
        let base_weights = XlmrWeights::load_from_mmapped(st, &cfg)?;
        let head = ForMaskedLMWeights::load_from_mmapped(st, &cfg)?;
        Ok(Self {
            base: XlmrModel { config: cfg, weights: base_weights },
            lm_head_dense_weight:   head.lm_head_dense_weight,
            lm_head_dense_bias:     head.lm_head_dense_bias,
            lm_head_ln_gain:        head.lm_head_ln_gain,
            lm_head_ln_bias:        head.lm_head_ln_bias,
            lm_head_decoder_weight: head.lm_head_decoder_weight,
            lm_head_decoder_bias:   head.lm_head_decoder_bias,
        })
    }
}

// ---- Sequence classification head -----------------------------------------

/// Weights for the XLM-RoBERTa sequence-classification head.
///
/// HuggingFace layout (`xlm-roberta-{base,large}` fine-tunes):
///
///   - `classifier.dense.weight`     `[hidden, hidden]`
///   - `classifier.dense.bias`       `[hidden]`
///   - `classifier.out_proj.weight`  `[num_labels, hidden]`
///   - `classifier.out_proj.bias`    `[num_labels]`
#[derive(Debug, Clone)]
pub struct ForSequenceClassificationWeights {
    pub classifier_dense_weight:    WeightStorage,
    pub classifier_dense_bias:      Arc<[f32]>,
    pub classifier_out_proj_weight: WeightStorage,
    pub classifier_out_proj_bias:   Arc<[f32]>,
}

/// XLM-RoBERTa with a sequence-classification head on top of the base
/// encoder. The eager port (and HF) take the first-token hidden state
/// (`<s>` / CLS), run it through a tanh-activated dense projection, and
/// then through an `out_proj` linear. Output shape: `(1, num_labels)`.
#[derive(Debug, Clone)]
pub struct XlmrForSequenceClassification {
    pub base:                       XlmrModel,
    pub num_labels:                 usize,
    pub classifier_dense_weight:    WeightStorage,
    pub classifier_dense_bias:      Arc<[f32]>,
    pub classifier_out_proj_weight: WeightStorage,
    pub classifier_out_proj_bias:   Arc<[f32]>,
}

impl XlmrForSequenceClassification {
    /// Run a forward pass and return per-label logits of shape
    /// `(1, num_labels)`.
    pub fn forward(
        &self,
        tokens: &[u32],
        attention_mask: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        let cfg = &self.base.config;
        let h = self.base.forward(tokens, attention_mask)?;
        // First-token hidden state: (1, seq, hidden) -> slice along dim
        // 1 starting at 0, length 1 -> (1, 1, hidden) -> reshape to
        // (1, hidden).
        let cls = h
            .slice(1_usize, 0, 1)?
            .reshape(Shape::from_dims(&[1, cfg.hidden_size]))?;
        // dense (hidden -> hidden) + bias + tanh.
        let dense = self.classifier_dense_weight
            .apply_linear(&cls, cfg.hidden_size, cfg.hidden_size);
        let dense = dense.add_trailing_bias(Arc::clone(&self.classifier_dense_bias))?;
        let act = dense.tanh();
        // out_proj (hidden -> num_labels) + bias.
        let logits = self.classifier_out_proj_weight
            .apply_linear(&act, cfg.hidden_size, self.num_labels);
        let logits = logits.add_trailing_bias(Arc::clone(&self.classifier_out_proj_bias))?;
        Ok(logits)
    }
}

impl ForSequenceClassificationWeights {
    /// Load the classifier head tensors using the HF
    /// `classifier.{dense,out_proj}.{weight,bias}` naming.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &XlmrConfig,
        num_labels: usize,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype as ltm};
        let h = cfg.hidden_size;
        let classifier_dense_weight = ltm(st, "classifier.dense.weight", h, h)?;
        let classifier_dense_bias = Arc::from(load_tensor_as_f32(
            st, "classifier.dense.bias",
        )?);
        let classifier_out_proj_weight = ltm(
            st, "classifier.out_proj.weight", num_labels, h,
        )?;
        let classifier_out_proj_bias = Arc::from(load_tensor_as_f32(
            st, "classifier.out_proj.bias",
        )?);
        Ok(Self {
            classifier_dense_weight,
            classifier_dense_bias,
            classifier_out_proj_weight,
            classifier_out_proj_bias,
        })
    }
}

impl XlmrForSequenceClassification {
    /// Load the full sequence-classification model (base encoder +
    /// classifier head) from a HuggingFace safetensors file in one call.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: XlmrConfig,
        num_labels: usize,
    ) -> Result<Self> {
        let base_weights = XlmrWeights::load_from_mmapped(st, &cfg)?;
        let head = ForSequenceClassificationWeights::load_from_mmapped(st, &cfg, num_labels)?;
        Ok(Self {
            base: XlmrModel { config: cfg, weights: base_weights },
            num_labels,
            classifier_dense_weight:    head.classifier_dense_weight,
            classifier_dense_bias:      head.classifier_dense_bias,
            classifier_out_proj_weight: head.classifier_out_proj_weight,
            classifier_out_proj_bias:   head.classifier_out_proj_bias,
        })
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

    // ---- Head fixtures ---------------------------------------------------

    fn tiny_for_masked_lm(cfg: &XlmrConfig) -> XlmrForMaskedLM {
        let mut s: u32 = 71717;
        let next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;
        XlmrForMaskedLM {
            base: XlmrModel { config: cfg.clone(), weights: tiny_weights(cfg) },
            lm_head_dense_weight:   WeightStorage::F32(vec_of(h * h, &mut *nb)),
            lm_head_dense_bias:     vec_of(h, &mut *nb),
            lm_head_ln_gain:        Arc::from(vec![1.0_f32; h]),
            lm_head_ln_bias:        Arc::from(vec![0.0_f32; h]),
            lm_head_decoder_weight: WeightStorage::F32(vec_of(h * v, &mut *nb)),
            lm_head_decoder_bias:   vec_of(v, &mut *nb),
        }
    }

    fn tiny_for_sequence_classification(
        cfg: &XlmrConfig, num_labels: usize,
    ) -> XlmrForSequenceClassification {
        let mut s: u32 = 90909;
        let next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let h = cfg.hidden_size;
        XlmrForSequenceClassification {
            base: XlmrModel { config: cfg.clone(), weights: tiny_weights(cfg) },
            num_labels,
            classifier_dense_weight:    WeightStorage::F32(vec_of(h * h, &mut *nb)),
            classifier_dense_bias:      vec_of(h, &mut *nb),
            classifier_out_proj_weight: WeightStorage::F32(vec_of(h * num_labels, &mut *nb)),
            classifier_out_proj_bias:   vec_of(num_labels, &mut *nb),
        }
    }

    /// `XlmrForMaskedLM::forward` returns logits of shape
    /// `(1, seq, vocab_size)` and all outputs are finite.
    #[test]
    fn for_masked_lm_forward_shape() {
        let cfg = tiny_cfg();
        let model = tiny_for_masked_lm(&cfg);
        let toks = [1_u32, 2, 3, 4, 5];
        let out = model.forward(&toks, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, toks.len(), cfg.vocab_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite MLM logit: {v}");
        }
    }

    /// `XlmrForSequenceClassification::forward` returns logits of
    /// shape `(1, num_labels)` and all outputs are finite.
    #[test]
    fn for_sequence_classification_forward_shape() {
        let cfg = tiny_cfg();
        let num_labels = 3;
        let model = tiny_for_sequence_classification(&cfg, num_labels);
        let toks = [1_u32, 2, 3, 4];
        let out = model.forward(&toks, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, num_labels]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite classifier logit: {v}");
        }
    }

    // ---- Safetensors round-trip fixtures --------------------------------

    /// Append `n` f32 values to `owned` under `name` as a 1-D shape.
    fn push_f32_1d(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        name: &str,
        values: &[f32],
    ) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values { bytes.extend_from_slice(&v.to_le_bytes()); }
        owned.push((name.to_string(), vec![values.len()], bytes));
    }

    /// Append a multi-dim f32 tensor of given shape, filled by `nb()`.
    fn push_f32(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        name: &str,
        shape: Vec<usize>,
        nb: &mut dyn FnMut() -> f32,
    ) {
        let n: usize = shape.iter().product();
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n { bytes.extend_from_slice(&nb().to_le_bytes()); }
        owned.push((name.to_string(), shape, bytes));
    }

    /// Push a HuggingFace LayerNorm prefix (`<prefix>.{weight,bias}`).
    fn push_ln(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        c: usize,
        nb: &mut dyn FnMut() -> f32,
    ) {
        push_f32(owned, &format!("{prefix}.weight"), vec![c], nb);
        push_f32(owned, &format!("{prefix}.bias"),   vec![c], nb);
    }

    /// Push an HF Linear (`<prefix>.{weight,bias}`). HF stores weight
    /// as `[out, in]`.
    fn push_linear(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        nb: &mut dyn FnMut() -> f32,
    ) {
        push_f32(
            owned, &format!("{prefix}.weight"),
            vec![out_features, in_features], nb,
        );
        push_f32(owned, &format!("{prefix}.bias"), vec![out_features], nb);
    }

    fn push_base_encoder(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        cfg: &XlmrConfig,
        nb: &mut dyn FnMut() -> f32,
    ) {
        let prefix = "roberta.";
        // Embeddings.
        push_f32(
            owned, &format!("{prefix}embeddings.word_embeddings.weight"),
            vec![cfg.vocab_size, cfg.hidden_size], nb,
        );
        push_f32(
            owned, &format!("{prefix}embeddings.position_embeddings.weight"),
            vec![cfg.max_position_embeddings, cfg.hidden_size], nb,
        );
        push_f32(
            owned, &format!("{prefix}embeddings.token_type_embeddings.weight"),
            vec![cfg.type_vocab_size, cfg.hidden_size], nb,
        );
        push_ln(owned, &format!("{prefix}embeddings.LayerNorm"), cfg.hidden_size, nb);
        // Encoder layers.
        for i in 0..cfg.num_hidden_layers {
            let p = format!("{prefix}encoder.layer.{i}");
            push_linear(owned, &format!("{p}.attention.self.query"),
                cfg.hidden_size, cfg.hidden_size, nb);
            push_linear(owned, &format!("{p}.attention.self.key"),
                cfg.hidden_size, cfg.hidden_size, nb);
            push_linear(owned, &format!("{p}.attention.self.value"),
                cfg.hidden_size, cfg.hidden_size, nb);
            push_linear(owned, &format!("{p}.attention.output.dense"),
                cfg.hidden_size, cfg.hidden_size, nb);
            push_ln(owned, &format!("{p}.attention.output.LayerNorm"),
                cfg.hidden_size, nb);
            push_linear(owned, &format!("{p}.intermediate.dense"),
                cfg.hidden_size, cfg.intermediate_size, nb);
            push_linear(owned, &format!("{p}.output.dense"),
                cfg.intermediate_size, cfg.hidden_size, nb);
            push_ln(owned, &format!("{p}.output.LayerNorm"), cfg.hidden_size, nb);
        }
    }

    fn build_safetensors_file(
        owned: Vec<(String, Vec<usize>, Vec<u8>)>,
        tag: &str,
    ) -> std::path::PathBuf {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        for (name, shape, bytes) in &owned {
            let view = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                .expect("TensorView::new");
            tensors.insert(name.clone(), view);
        }
        let serialized = safetensors::serialize(&tensors, None)
            .expect("safetensors::serialize");
        let tmp = std::env::temp_dir().join(format!(
            "fuel_xlmr_heads_{}_{tag}_{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
        ));
        std::fs::write(&tmp, &serialized).expect("write tmp");
        tmp
    }

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }

    /// Round-trip the MaskedLM head through safetensors using the
    /// standard HF naming and check the forward shape.
    #[test]
    fn for_masked_lm_load_from_mmapped_round_trip() {
        let cfg = tiny_cfg();
        let mut nb = rng_seed(12345);

        let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        push_base_encoder(&mut owned, &cfg, &mut nb);
        // lm_head.
        push_linear(&mut owned, "lm_head.dense",
            cfg.hidden_size, cfg.hidden_size, &mut nb);
        push_ln(&mut owned, "lm_head.layer_norm", cfg.hidden_size, &mut nb);
        push_linear(&mut owned, "lm_head.decoder",
            cfg.hidden_size, cfg.vocab_size, &mut nb);

        let tmp = build_safetensors_file(owned, "mlm");
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&tmp) }
            .expect("MmapedSafetensors::new");

        let model = XlmrForMaskedLM::load_from_mmapped(&st, cfg.clone())
            .expect("XlmrForMaskedLM::load_from_mmapped");

        let toks = [1_u32, 2, 3, 4];
        let out = model.forward(&toks, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, toks.len(), cfg.vocab_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite MLM logit from loaded model: {v}");
        }

        let _ = std::fs::remove_file(&tmp);
    }

    /// Round-trip the SequenceClassification head through safetensors
    /// using the standard HF naming and check the forward shape.
    #[test]
    fn for_sequence_classification_load_from_mmapped_round_trip() {
        let cfg = tiny_cfg();
        let num_labels = 4;
        let mut nb = rng_seed(54321);

        let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        push_base_encoder(&mut owned, &cfg, &mut nb);
        push_linear(&mut owned, "classifier.dense",
            cfg.hidden_size, cfg.hidden_size, &mut nb);
        push_linear(&mut owned, "classifier.out_proj",
            cfg.hidden_size, num_labels, &mut nb);

        let tmp = build_safetensors_file(owned, "seq");
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&tmp) }
            .expect("MmapedSafetensors::new");

        let model = XlmrForSequenceClassification::load_from_mmapped(
            &st, cfg.clone(), num_labels,
        ).expect("XlmrForSequenceClassification::load_from_mmapped");

        let toks = [1_u32, 2, 3, 4, 5];
        let out = model.forward(&toks, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, num_labels]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite classifier logit from loaded model: {v}");
        }

        let _ = std::fs::remove_file(&tmp);
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

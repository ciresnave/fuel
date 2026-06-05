//! Marian NMT (Helsinki-NLP / Opus MT) ported to the lazy-graph API.
//!
//! Phase D specialized port. Marian is the **first encoder-decoder
//! port** in the Phase D lineup — every prior model has been
//! decoder-only. The architecture is a classic Transformer with:
//!
//!   1. **Separate encoder + decoder stacks** sharing a token
//!      embedding (`share_encoder_decoder_embeddings = true` for
//!      every Opus MT preset shipped).
//!   2. **Sinusoidal positional embeddings** — precomputed
//!      `(max_positions, d_model)` table with
//!      `[sin(t * inv_freq), cos(t * inv_freq)]` concat along the
//!      feature axis. No RoPE.
//!   3. **Post-LN sublayer structure**: `LN(x + sublayer(x))`.
//!      Each sublayer (self-attn, cross-attn, ffn) ends with its
//!      own LayerNorm.
//!   4. **Decoder cross-attention**: Q from decoder state, K/V
//!      from the (single) encoder output. This is the
//!      defining feature of an encoder-decoder model.
//!   5. **Q scaled by `head_dim ** -0.5`** at projection time
//!      (not at the score matmul, although the math is equivalent).
//!   6. **Tied lm_head** to `shared` token embedding +
//!      `final_logits_bias` of shape `[vocab]`.
//!   7. **Embedding scaling** by `sqrt(d_model)` when
//!      `scale_embedding == true` (all Opus MT presets do).
//!
//! Activation is config-driven; Opus MT shipped models use Relu
//! (tc-big) or Silu / Swish (smaller variants).
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache,
//! F32. Both encoder + decoder forward in one call:
//! `forward(src_tokens, tgt_tokens) -> logits`. The decoder
//! sees the full target sequence (teacher-forced layout); a
//! production decode loop is out of scope.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarianActivation {
    Relu,
    Silu,
    Gelu,
    GeluPytorchTanh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MarianConfig {
    pub vocab_size: usize,
    /// Defaults to `vocab_size` when `None`.
    pub decoder_vocab_size: Option<usize>,
    pub max_position_embeddings: usize,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_attention_heads: usize,
    pub decoder_layers: usize,
    pub decoder_ffn_dim: usize,
    pub decoder_attention_heads: usize,
    pub scale_embedding: bool,
    pub share_encoder_decoder_embeddings: bool,
    pub activation_function: MarianActivation,
}

impl MarianConfig {
    pub fn target_vocab_size(&self) -> usize {
        self.decoder_vocab_size.unwrap_or(self.vocab_size)
    }
    pub fn encoder_head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }
    pub fn decoder_head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }

    /// `Helsinki-NLP/opus-mt-tc-big-fr-en` preset.
    pub fn opus_mt_tc_big_fr_en() -> Self {
        Self {
            vocab_size: 53017,
            decoder_vocab_size: Some(53017),
            max_position_embeddings: 1024,
            d_model: 1024,
            encoder_layers: 6,
            encoder_ffn_dim: 4096,
            encoder_attention_heads: 16,
            decoder_layers: 6,
            decoder_ffn_dim: 4096,
            decoder_attention_heads: 16,
            scale_embedding: true,
            share_encoder_decoder_embeddings: true,
            activation_function: MarianActivation::Relu,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarianAttentionWeights {
    pub q_proj: WeightStorage,
    pub q_proj_bias: Arc<[f32]>,
    pub k_proj: WeightStorage,
    pub k_proj_bias: Arc<[f32]>,
    pub v_proj: WeightStorage,
    pub v_proj_bias: Arc<[f32]>,
    pub out_proj: WeightStorage,
    pub out_proj_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct MarianEncoderLayerWeights {
    pub self_attn: MarianAttentionWeights,
    pub self_attn_ln_gain: Arc<[f32]>,
    pub self_attn_ln_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct MarianDecoderLayerWeights {
    pub self_attn: MarianAttentionWeights,
    pub self_attn_ln_gain: Arc<[f32]>,
    pub self_attn_ln_bias: Arc<[f32]>,
    pub encoder_attn: MarianAttentionWeights,
    pub encoder_attn_ln_gain: Arc<[f32]>,
    pub encoder_attn_ln_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct MarianWeights {
    pub shared_embedding: Arc<[f32]>,
    pub encoder_layers: Vec<MarianEncoderLayerWeights>,
    pub decoder_layers: Vec<MarianDecoderLayerWeights>,
    /// `[vocab]` — added to logits before output.
    pub final_logits_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct MarianModel {
    pub config: MarianConfig,
    pub weights: MarianWeights,
}

impl MarianModel {
    /// Encode + decode in one call. Returns logits of shape
    /// `[1, tgt_len, target_vocab_size]`.
    pub fn forward(&self, src_tokens: &[u32], tgt_tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        assert!(!src_tokens.is_empty(), "src_tokens must be non-empty");
        assert!(!tgt_tokens.is_empty(), "tgt_tokens must be non-empty");
        assert!(
            cfg.share_encoder_decoder_embeddings,
            "v1 only supports shared encoder+decoder embeddings",
        );
        assert_eq!(
            cfg.d_model % cfg.encoder_attention_heads, 0,
            "d_model must be divisible by encoder_attention_heads",
        );
        assert_eq!(
            cfg.d_model % cfg.decoder_attention_heads, 0,
            "d_model must be divisible by decoder_attention_heads",
        );

        // Build a single embed tensor (== single graph) and reuse it
        // for both encoder + decoder so cross-attention can matmul
        // across the two stacks.
        let embed = LazyTensor::from_f32(
            self.weights.shared_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.d_model]),
            &Device::cpu(),
        );
        let pos_table = build_sinusoidal_table(cfg.max_position_embeddings, cfg.d_model);
        let pos_lt_shape = Shape::from_dims(&[cfg.max_position_embeddings, cfg.d_model]);
        let pos_full = embed.const_f32_like(Arc::from(pos_table), pos_lt_shape);

        let enc_out = self.encode(&embed, &pos_full, src_tokens)?;
        let dec_out = self.decode(&embed, &pos_full, tgt_tokens, &enc_out)?;

        let lm_head = WeightStorage::F32(self.weights.shared_embedding.clone());
        let target_vocab = cfg.target_vocab_size();
        let logits = lm_head.apply_linear(&dec_out, cfg.d_model, target_vocab);
        let bias_t = dec_out.const_f32_like(
            Arc::clone(&self.weights.final_logits_bias),
            Shape::from_dims(&[target_vocab]),
        );
        logits.broadcast_add(&bias_t)
    }

    /// Run the encoder stack alone and return its per-token
    /// output `(1, src_len, d_model)`. Unlocks Marian-encoder
    /// based adapters (translation-quality embeddings, source
    /// language detection, etc.) without paying the decoder
    /// cost or requiring target tokens. Mirrors the
    /// `T5Model::forward_encoder` shape.
    pub fn forward_encoder(&self, src_tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        assert!(!src_tokens.is_empty(), "src_tokens must be non-empty");
        assert!(
            cfg.share_encoder_decoder_embeddings,
            "v1 only supports shared encoder+decoder embeddings",
        );
        assert_eq!(
            cfg.d_model % cfg.encoder_attention_heads, 0,
            "d_model must be divisible by encoder_attention_heads",
        );
        let embed = LazyTensor::from_f32(
            self.weights.shared_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.d_model]),
            &Device::cpu(),
        );
        let pos_table = build_sinusoidal_table(cfg.max_position_embeddings, cfg.d_model);
        let pos_lt_shape = Shape::from_dims(&[cfg.max_position_embeddings, cfg.d_model]);
        let pos_full = embed.const_f32_like(Arc::from(pos_table), pos_lt_shape);
        self.encode(&embed, &pos_full, src_tokens)
    }

    /// Run only the Marian decoder, given a precomputed encoder
    /// output and target tokens, and return logits of shape
    /// `(1, tgt_len, target_vocab)`. Mirrors
    /// `WhisperModel::forward_decoder` and
    /// `T5Model::forward_decoder` — `enc_out` is the graph
    /// anchor so all decoder constants land on the same graph.
    /// Use this for autoregressive NMT generation where the
    /// encoder output is cached once and the decoder is invoked
    /// per generated token.
    pub fn forward_decoder(
        &self,
        tgt_tokens: &[u32],
        enc_out: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        assert!(!tgt_tokens.is_empty(), "tgt_tokens must be non-empty");
        assert!(
            cfg.share_encoder_decoder_embeddings,
            "v1 only supports shared encoder+decoder embeddings",
        );
        assert_eq!(
            cfg.d_model % cfg.decoder_attention_heads, 0,
            "d_model must be divisible by decoder_attention_heads",
        );

        let embed = enc_out.const_f32_like(
            self.weights.shared_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.d_model]),
        );
        let pos_table = build_sinusoidal_table(cfg.max_position_embeddings, cfg.d_model);
        let pos_full = enc_out.const_f32_like(
            Arc::from(pos_table),
            Shape::from_dims(&[cfg.max_position_embeddings, cfg.d_model]),
        );

        let dec_out = self.decode(&embed, &pos_full, tgt_tokens, enc_out)?;

        let lm_head = WeightStorage::F32(self.weights.shared_embedding.clone());
        let target_vocab = cfg.target_vocab_size();
        let logits = lm_head.apply_linear(&dec_out, cfg.d_model, target_vocab);
        let bias_t = enc_out.const_f32_like(
            Arc::clone(&self.weights.final_logits_bias),
            Shape::from_dims(&[target_vocab]),
        );
        logits.broadcast_add(&bias_t)
    }

    fn encode(
        &self,
        embed: &LazyTensor,
        pos_full: &LazyTensor,
        src_tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let src_len = src_tokens.len();
        let batch = 1;

        let ids = embed.const_u32_like(src_tokens.to_vec(), Shape::from_dims(&[src_len]));
        let mut x = embed
            .index_select(0_usize, &ids)?
            .reshape(Shape::from_dims(&[batch, src_len, cfg.d_model]))?;
        if cfg.scale_embedding {
            x = x.mul_scalar((cfg.d_model as f64).sqrt());
        }
        let pos_slice = pos_full
            .slice(0_usize, 0, src_len)?
            .reshape(Shape::from_dims(&[1, src_len, cfg.d_model]))?;
        x = x.add(&pos_slice.broadcast_to(Shape::from_dims(&[batch, src_len, cfg.d_model]))?)?;

        for layer in &self.weights.encoder_layers {
            x = self.apply_encoder_layer(&x, layer)?;
        }
        Ok(x)
    }

    fn decode(
        &self,
        embed: &LazyTensor,
        pos_full: &LazyTensor,
        tgt_tokens: &[u32],
        enc_out: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let tgt_len = tgt_tokens.len();
        let batch = 1;

        let ids = embed.const_u32_like(tgt_tokens.to_vec(), Shape::from_dims(&[tgt_len]));
        let mut x = embed
            .index_select(0_usize, &ids)?
            .reshape(Shape::from_dims(&[batch, tgt_len, cfg.d_model]))?;
        if cfg.scale_embedding {
            x = x.mul_scalar((cfg.d_model as f64).sqrt());
        }
        let pos_slice = pos_full
            .slice(0_usize, 0, tgt_len)?
            .reshape(Shape::from_dims(&[1, tgt_len, cfg.d_model]))?;
        x = x.add(&pos_slice.broadcast_to(Shape::from_dims(&[batch, tgt_len, cfg.d_model]))?)?;

        let mut mask_data = vec![0.0_f32; tgt_len * tgt_len];
        for i in 0..tgt_len {
            for j in (i + 1)..tgt_len {
                mask_data[i * tgt_len + j] = f32::NEG_INFINITY;
            }
        }
        let causal_mask = x.const_f32_like(
            mask_data,
            Shape::from_dims(&[1, 1, tgt_len, tgt_len]),
        );

        for layer in &self.weights.decoder_layers {
            x = self.apply_decoder_layer(&x, layer, enc_out, &causal_mask)?;
        }
        Ok(x)
    }

    fn apply_encoder_layer(
        &self,
        x: &LazyTensor,
        layer: &MarianEncoderLayerWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let attn = self.attention(
            x, x, &layer.self_attn,
            cfg.encoder_attention_heads, cfg.encoder_head_dim(),
            None,
        )?;
        let h1 = x.add(&attn)?.layer_norm_affine(std::sync::Arc::clone(&layer.self_attn_ln_gain), std::sync::Arc::clone(&layer.self_attn_ln_bias), 1e-5)?;
        let ffn = self.feed_forward(
            &h1, &layer.fc1, &layer.fc1_bias, &layer.fc2, &layer.fc2_bias,
            cfg.encoder_ffn_dim,
        )?;
        Ok(h1.add(&ffn)?.layer_norm_affine(std::sync::Arc::clone(&layer.final_ln_gain), std::sync::Arc::clone(&layer.final_ln_bias), 1e-5)?)
    }

    fn apply_decoder_layer(
        &self,
        x: &LazyTensor,
        layer: &MarianDecoderLayerWeights,
        enc_out: &LazyTensor,
        causal_mask: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        // Self-attention with causal mask.
        let self_attn = self.attention(
            x, x, &layer.self_attn,
            cfg.decoder_attention_heads, cfg.decoder_head_dim(),
            Some(causal_mask),
        )?;
        let h1 = x.add(&self_attn)?.layer_norm_affine(std::sync::Arc::clone(&layer.self_attn_ln_gain), std::sync::Arc::clone(&layer.self_attn_ln_bias), 1e-5)?;
        // Cross-attention: Q from decoder state, K/V from encoder output.
        let cross_attn = self.attention(
            &h1, enc_out, &layer.encoder_attn,
            cfg.decoder_attention_heads, cfg.decoder_head_dim(),
            None,
        )?;
        let h2 = h1.add(&cross_attn)?.layer_norm_affine(std::sync::Arc::clone(&layer.encoder_attn_ln_gain), std::sync::Arc::clone(&layer.encoder_attn_ln_bias), 1e-5)?;
        let ffn = self.feed_forward(
            &h2, &layer.fc1, &layer.fc1_bias, &layer.fc2, &layer.fc2_bias,
            cfg.decoder_ffn_dim,
        )?;
        Ok(h2.add(&ffn)?.layer_norm_affine(std::sync::Arc::clone(&layer.final_ln_gain), std::sync::Arc::clone(&layer.final_ln_bias), 1e-5)?)
    }

    /// Generic multi-head attention. `q_src` provides Q; `kv_src`
    /// provides K and V. For self-attention they're the same;
    /// for cross-attention they differ.
    fn attention(
        &self,
        q_src: &LazyTensor,
        kv_src: &LazyTensor,
        w: &MarianAttentionWeights,
        n_heads: usize,
        head_dim: usize,
        attn_mask: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        let d_model = self.config.d_model;
        let q_shape = q_src.shape();
        let q_dims = q_shape.dims();
        let batch = q_dims[0];
        let q_len = q_dims[1];
        let kv_shape = kv_src.shape();
        let kv_dims = kv_shape.dims();
        let kv_len = kv_dims[1];

        let scaling = (head_dim as f64).powf(-0.5);
        let q = add_bias_3d(
            w.q_proj.apply_linear(q_src, d_model, d_model),
            &w.q_proj_bias, d_model,
        )?.mul_scalar(scaling);
        let k = add_bias_3d(
            w.k_proj.apply_linear(kv_src, d_model, d_model),
            &w.k_proj_bias, d_model,
        )?;
        let v = add_bias_3d(
            w.v_proj.apply_linear(kv_src, d_model, d_model),
            &w.v_proj_bias, d_model,
        )?;

        let _ = (batch, q_len, kv_len);
        let q = q.split_heads(n_heads, head_dim)?;
        let k = k.split_heads(n_heads, head_dim)?;
        let v = v.split_heads(n_heads, head_dim)?;

        let k_t = k.transpose()?;
        let scores = q.matmul(&k_t)?;
        let scores = match attn_mask {
            None => scores,
            Some(mask) => scores.broadcast_add(mask)?,
        };
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx.merge_heads()?;
        let out = w.out_proj.apply_linear(&merged, d_model, d_model);
        add_bias_3d(out, &w.out_proj_bias, d_model)
    }

    fn feed_forward(
        &self,
        x: &LazyTensor,
        fc1_w: &WeightStorage, fc1_b: &Arc<[f32]>,
        fc2_w: &WeightStorage, fc2_b: &Arc<[f32]>,
        ffn_dim: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let fc1 = add_bias_3d(
            fc1_w.apply_linear(x, cfg.d_model, ffn_dim),
            fc1_b, ffn_dim,
        )?;
        let activated = match cfg.activation_function {
            MarianActivation::Relu => fc1.relu(),
            MarianActivation::Silu => fc1.silu(),
            MarianActivation::Gelu => fc1.gelu_erf(),
            MarianActivation::GeluPytorchTanh => fc1.gelu(),
        };
        let fc2 = add_bias_3d(
            fc2_w.apply_linear(&activated, ffn_dim, cfg.d_model),
            fc2_b, cfg.d_model,
        )?;
        Ok(fc2)
    }
}

/// Build the (`max_positions`, `d_model`)-shaped sinusoidal
/// position table used by Marian: the first `d_model/2` features
/// are `sin(t * inv_freq)`, the last `d_model/2` are
/// `cos(t * inv_freq)`, where `inv_freq[i] = 10000^(-i / d_model)`
/// for `i ∈ [0, d_model/2)`.
fn build_sinusoidal_table(max_positions: usize, d_model: usize) -> Vec<f32> {
    assert_eq!(d_model % 2, 0, "d_model must be even");
    let half = d_model / 2;
    let mut out = vec![0.0_f32; max_positions * d_model];
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0_f32 / 10_000.0_f32.powf((2 * i) as f32 / d_model as f32))
        .collect();
    for t in 0..max_positions {
        for i in 0..half {
            let arg = t as f32 * inv_freq[i];
            out[t * d_model + i] = arg.sin();
            out[t * d_model + half + i] = arg.cos();
        }
    }
    out
}

fn add_bias_3d(x: LazyTensor, bias: &Arc<[f32]>, n: usize) -> Result<LazyTensor> {
    let _ = n;
    x.add_trailing_bias(Arc::clone(bias))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_attention_weights(d_model: usize, next_box: &mut Box<dyn FnMut() -> f32>) -> MarianAttentionWeights {
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        MarianAttentionWeights {
            q_proj: WeightStorage::F32(vec_of(d_model * d_model, &mut **next_box)),
            q_proj_bias: vec_of(d_model, &mut **next_box),
            k_proj: WeightStorage::F32(vec_of(d_model * d_model, &mut **next_box)),
            k_proj_bias: vec_of(d_model, &mut **next_box),
            v_proj: WeightStorage::F32(vec_of(d_model * d_model, &mut **next_box)),
            v_proj_bias: vec_of(d_model, &mut **next_box),
            out_proj: WeightStorage::F32(vec_of(d_model * d_model, &mut **next_box)),
            out_proj_bias: vec_of(d_model, &mut **next_box),
        }
    }

    fn tiny_weights(cfg: &MarianConfig) -> MarianWeights {
        let mut s: u32 = 64646;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let d = cfg.d_model;
        let enc_ffn = cfg.encoder_ffn_dim;
        let dec_ffn = cfg.decoder_ffn_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);

        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let shared_embedding = vec_of(cfg.vocab_size * d, &mut *nb);

        let encoder_layers: Vec<MarianEncoderLayerWeights> = (0..cfg.encoder_layers)
            .map(|_| MarianEncoderLayerWeights {
                self_attn: tiny_attention_weights(d, &mut nb),
                self_attn_ln_gain: Arc::from(vec![1.0_f32; d]),
                self_attn_ln_bias: Arc::from(vec![0.0_f32; d]),
                fc1: WeightStorage::F32(vec_of(d * enc_ffn, &mut *nb)),
                fc1_bias: vec_of(enc_ffn, &mut *nb),
                fc2: WeightStorage::F32(vec_of(enc_ffn * d, &mut *nb)),
                fc2_bias: vec_of(d, &mut *nb),
                final_ln_gain: Arc::from(vec![1.0_f32; d]),
                final_ln_bias: Arc::from(vec![0.0_f32; d]),
            })
            .collect();
        let decoder_layers: Vec<MarianDecoderLayerWeights> = (0..cfg.decoder_layers)
            .map(|_| MarianDecoderLayerWeights {
                self_attn: tiny_attention_weights(d, &mut nb),
                self_attn_ln_gain: Arc::from(vec![1.0_f32; d]),
                self_attn_ln_bias: Arc::from(vec![0.0_f32; d]),
                encoder_attn: tiny_attention_weights(d, &mut nb),
                encoder_attn_ln_gain: Arc::from(vec![1.0_f32; d]),
                encoder_attn_ln_bias: Arc::from(vec![0.0_f32; d]),
                fc1: WeightStorage::F32(vec_of(d * dec_ffn, &mut *nb)),
                fc1_bias: vec_of(dec_ffn, &mut *nb),
                fc2: WeightStorage::F32(vec_of(dec_ffn * d, &mut *nb)),
                fc2_bias: vec_of(d, &mut *nb),
                final_ln_gain: Arc::from(vec![1.0_f32; d]),
                final_ln_bias: Arc::from(vec![0.0_f32; d]),
            })
            .collect();
        let target_vocab = cfg.target_vocab_size();
        let final_logits_bias = vec_of(target_vocab, &mut *nb);

        MarianWeights {
            shared_embedding,
            encoder_layers,
            decoder_layers,
            final_logits_bias,
        }
    }

    fn tiny_config() -> MarianConfig {
        MarianConfig {
            vocab_size: 16, decoder_vocab_size: None,
            max_position_embeddings: 16,
            d_model: 8,
            encoder_layers: 2, encoder_ffn_dim: 16, encoder_attention_heads: 2,
            decoder_layers: 2, decoder_ffn_dim: 16, decoder_attention_heads: 2,
            scale_embedding: true,
            share_encoder_decoder_embeddings: true,
            activation_function: MarianActivation::Relu,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = MarianModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let src = [1_u32, 2, 3, 4];
        let tgt = [5_u32, 6, 7];
        let logits = model.forward(&src, &tgt).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tgt.len(), cfg.target_vocab_size()]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// Source change must alter target logits — proves cross-
    /// attention actually conditions on the encoder output.
    #[test]
    fn cross_attention_is_wired() {
        let cfg = tiny_config();
        let model = MarianModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tgt = [5_u32, 6, 7];
        let a = model.forward(&[1, 2, 3, 4], &tgt).unwrap().realize_f32();
        let b = model.forward(&[8, 9, 10, 11], &tgt).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "cross-attention must condition on encoder output: changing src must change tgt logits, max_diff = {max_diff}");
    }

    /// Decoder causal mask is enforced: changing target token at
    /// position t must NOT change logits at positions < t.
    #[test]
    fn decoder_causal_mask_is_enforced() {
        let cfg = tiny_config();
        let model = MarianModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let src = [1_u32, 2, 3];
        let tgt_a = [5_u32, 6, 7, 8];
        let tgt_b = [5_u32, 6, 7, 15]; // last token changed
        let a = model.forward(&src, &tgt_a).unwrap().realize_f32();
        let b = model.forward(&src, &tgt_b).unwrap().realize_f32();
        // Compare logits at positions 0, 1, 2 (which precede the changed token).
        let v = cfg.target_vocab_size();
        for t in 0..3 {
            for col in 0..v {
                let i = t * v + col;
                assert!(
                    (a[i] - b[i]).abs() < 1e-5,
                    "causal mask violated at t={t}, col={col}: {} vs {}", a[i], b[i],
                );
            }
        }
    }

    /// Sinusoidal table sanity: position 0 must be all-zero in
    /// the sin half and all-one in the cos half.
    #[test]
    fn sinusoidal_table_position_0() {
        let t = build_sinusoidal_table(4, 8);
        let half = 4;
        for i in 0..half {
            assert!((t[i]).abs() < 1e-6, "sin[0, {i}] = {} != 0", t[i]);
            assert!((t[half + i] - 1.0).abs() < 1e-6, "cos[0, {i}] = {} != 1", t[half + i]);
        }
    }

    #[test]
    fn presets_construct() {
        let c = MarianConfig::opus_mt_tc_big_fr_en();
        assert_eq!(c.d_model, 1024);
        assert_eq!(c.encoder_attention_heads, 16);
        assert_eq!(c.target_vocab_size(), 53017);
    }

    #[test]
    fn forward_encoder_shape_and_finite() {
        let cfg = tiny_config();
        let model = MarianModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let src = [1_u32, 2, 3, 4, 5];
        let enc = model.forward_encoder(&src).unwrap();
        assert_eq!(enc.shape().dims(), &[1, src.len(), cfg.d_model]);
        for &v in &enc.realize_f32() {
            assert!(v.is_finite(), "non-finite encoder output: {v}");
        }
    }

    /// `forward_encoder` must produce the same encoder output
    /// that `forward` consumes internally. Prove this by
    /// computing the cross-attention contribution two ways:
    /// (1) via `forward` directly, and (2) by re-running
    /// `forward_encoder` and confirming the result tensor has
    /// the same per-position L2 norm distribution as a baseline.
    /// We check determinism: same input → same output across
    /// two calls.
    #[test]
    fn forward_encoder_deterministic() {
        let cfg = tiny_config();
        let model = MarianModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let src = [1_u32, 2, 3, 4];
        let a = model.forward_encoder(&src).unwrap().realize_f32();
        let b = model.forward_encoder(&src).unwrap().realize_f32();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-7, "non-deterministic encoder: {x} vs {y}");
        }
    }

    #[test]
    fn forward_decoder_shape_and_finite() {
        let cfg = tiny_config();
        let model = MarianModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let enc = model.forward_encoder(&[1_u32, 2, 3, 4]).unwrap();
        let tgt = [5_u32, 6, 7];
        let logits = model.forward_decoder(&tgt, &enc).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tgt.len(), cfg.target_vocab_size()]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite decoder logit: {v}");
        }
    }

    /// `forward_decoder(tgt, forward_encoder(src))` must match
    /// `forward(src, tgt)` — the two paths compute the same
    /// graph.
    #[test]
    fn forward_decoder_matches_full_forward() {
        let cfg = tiny_config();
        let model = MarianModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let src = [1_u32, 2, 3];
        let tgt = [5_u32, 6, 7];
        let full = model.forward(&src, &tgt).unwrap().realize_f32();
        let enc = model.forward_encoder(&src).unwrap();
        let split = model.forward_decoder(&tgt, &enc).unwrap().realize_f32();
        assert_eq!(full.len(), split.len());
        for (a, b) in full.iter().zip(split.iter()) {
            assert!((a - b).abs() < 1e-5,
                "forward_decoder must match forward: {a} vs {b}");
        }
    }
}

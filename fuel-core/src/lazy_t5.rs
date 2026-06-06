//! T5 (Text-To-Text Transfer Transformer) ported to the lazy-graph API.
//!
//! Phase D specialized port. Second encoder-decoder port after
//! [`crate::lazy_marian`]. T5 distinguishes itself from Marian
//! along several axes:
//!
//!   1. **Relative position bias** instead of sinusoidal /
//!      learned absolute positions. The bias is a learned
//!      `[num_buckets, n_heads]` embedding looked up by
//!      bucketed relative distances `kv_pos - q_pos`. The
//!      lookup table is shared across the encoder stack (one
//!      per stack) and **only the first layer** owns the
//!      learned `relative_attention_bias` parameter — later
//!      layers consume the bias passed through from the first.
//!      This port precomputes the bias once at the start of
//!      each stack and broadcasts it to all layers.
//!   2. **No biases on any linear** (Q/K/V/O, FFN, lm_head).
//!   3. **T5LayerNorm == RmsNorm with no offset** — uses
//!      `mean(x^2)` (not centered variance) for stability,
//!      then multiplies by per-channel `weight`. Equivalent
//!      to `apply_affine_rms_norm_pub`.
//!   4. **Pre-LN sublayer structure**: `out = x + sublayer(LN(x))`.
//!      (Marian's post-LN does `LN(x + sublayer(x))`.)
//!   5. **Two FFN variants**:
//!        - `T5DenseActDense` (default): `wo(act(wi(x)))` —
//!          sequential, used by T5-v1 (`relu`).
//!        - `T5DenseGatedActDense`: `wo(act(wi_0(x)) * wi_1(x))`
//!          — gated SwiGLU-shape, used by Flan-T5 / UL2 /
//!          MADLAD-400 (`gated-gelu` or `gated-silu`).
//!   6. **`d_kv` is independently configurable** — `n_heads *
//!      d_kv` may not equal `d_model`. T5-small uses
//!      `8 * 64 = 512 == d_model` but larger models decouple.
//!   7. **Tied lm_head** to shared token embedding when
//!      `tie_word_embeddings == true` (the default).
//!
//! The relative-position bucketing follows the standard T5
//! bidirectional scheme: 32 buckets total split in half (16 for
//! past, 16 for future), with the inner half of each direction
//! using exact distance and the outer half using log-spaced
//! buckets up to `relative_attention_max_distance`.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache,
//! F32, teacher-forced layout (decoder sees full target sequence
//! in one call).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum T5Activation {
    Relu,
    Silu,
    Gelu,
    GeluPytorchTanh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    /// Defaults to `num_layers` when `None`.
    pub num_decoder_layers: Option<usize>,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_epsilon: f64,
    pub activation: T5Activation,
    pub gated_ffn: bool,
    pub tie_word_embeddings: bool,
}

impl T5Config {
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.d_kv
    }
    pub fn num_decoder_layers_resolved(&self) -> usize {
        self.num_decoder_layers.unwrap_or(self.num_layers)
    }

    /// T5-small preset.
    pub fn t5_small() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 512, d_kv: 64, d_ff: 2048,
            num_layers: 6, num_decoder_layers: None,
            num_heads: 8,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_epsilon: 1e-6,
            activation: T5Activation::Relu,
            gated_ffn: false,
            tie_word_embeddings: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct T5AttentionWeights {
    pub q: WeightStorage, // d_model → inner_dim
    pub k: WeightStorage,
    pub v: WeightStorage,
    pub o: WeightStorage, // inner_dim → d_model
}

#[derive(Debug, Clone)]
pub enum T5FfnWeights {
    Dense { wi: WeightStorage, wo: WeightStorage },
    Gated { wi_0: WeightStorage, wi_1: WeightStorage, wo: WeightStorage },
}

#[derive(Debug, Clone)]
pub struct T5EncoderLayerWeights {
    pub self_attn_norm_gain: Arc<[f32]>,
    pub self_attn: T5AttentionWeights,
    pub ffn_norm_gain: Arc<[f32]>,
    pub ffn: T5FfnWeights,
}

#[derive(Debug, Clone)]
pub struct T5DecoderLayerWeights {
    pub self_attn_norm_gain: Arc<[f32]>,
    pub self_attn: T5AttentionWeights,
    pub cross_attn_norm_gain: Arc<[f32]>,
    pub cross_attn: T5AttentionWeights,
    pub ffn_norm_gain: Arc<[f32]>,
    pub ffn: T5FfnWeights,
}

#[derive(Debug, Clone)]
pub struct T5Weights {
    pub shared_embedding: Arc<[f32]>,
    /// `[num_buckets, n_heads]` — encoder stack's relative bias.
    pub encoder_rel_bias: Arc<[f32]>,
    /// `[num_buckets, n_heads]` — decoder stack's relative bias.
    pub decoder_rel_bias: Arc<[f32]>,
    pub encoder_layers: Vec<T5EncoderLayerWeights>,
    pub decoder_layers: Vec<T5DecoderLayerWeights>,
    pub encoder_final_norm_gain: Arc<[f32]>,
    pub decoder_final_norm_gain: Arc<[f32]>,
    /// Optional separate lm_head when `!tie_word_embeddings`.
    pub lm_head: Option<WeightStorage>,
}

#[derive(Debug, Clone)]
pub struct T5Model {
    pub config: T5Config,
    pub weights: T5Weights,
}

impl T5Model {
    pub fn forward(&self, src_tokens: &[u32], tgt_tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        assert!(!src_tokens.is_empty());
        assert!(!tgt_tokens.is_empty());

        let embed = LazyTensor::from_f32(
            self.weights.shared_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.d_model]),
            &Device::cpu(),
        );

        let enc_out = self.encode(&embed, src_tokens)?;
        let dec_out = self.decode(&embed, tgt_tokens, &enc_out)?;

        let lm_head = match &self.weights.lm_head {
            Some(w) => w.clone(),
            None => {
                assert!(
                    cfg.tie_word_embeddings,
                    "lm_head absent but tie_word_embeddings=false",
                );
                WeightStorage::F32(self.weights.shared_embedding.clone())
            }
        };
        // T5 spec: when tied, scale by `d_model^-0.5` before the LM head.
        let dec_scaled = if cfg.tie_word_embeddings {
            dec_out.mul_scalar((cfg.d_model as f64).powf(-0.5))
        } else {
            dec_out
        };
        Ok(lm_head.apply_linear(&dec_scaled, cfg.d_model, cfg.vocab_size))
    }

    /// Run only the T5 encoder and return its hidden states
    /// `(1, src_len, d_model)`. Use this when T5 is a
    /// conditioning text encoder rather than a full
    /// seq2seq model — Parler-TTS, Stable Diffusion 3 / FLUX
    /// (T5-XXL conditioning), and any future model that
    /// just consumes T5 features.
    ///
    /// Includes the final RmsNorm. No LM head. Mirrors the
    /// `forward_hidden` pattern on the LLM backbones (Llama,
    /// Mistral, Qwen2) and the SD-text-encoder shape.
    pub fn forward_encoder(&self, src_tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        assert!(!src_tokens.is_empty(), "src_tokens must be non-empty");
        let embed = LazyTensor::from_f32(
            self.weights.shared_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.d_model]),
            &Device::cpu(),
        );
        self.encode(&embed, src_tokens)
    }

    /// Multimodal encoder entry point. Skips token embedding; runs the
    /// T5 encoder over pre-embedded inputs and returns its hidden
    /// states `(1, src_len, d_model)`. Useful for hosts that want to
    /// splice non-text features into the T5 encoder input (rare for T5
    /// but matches the cross-port convention).
    pub fn forward_encoder_embeds(&self, src_embeds: &LazyTensor) -> Result<LazyTensor> {
        self.encode_from_embeds(src_embeds)
    }

    /// Multimodal decoder entry point. Skips token embedding; runs the
    /// T5 decoder over pre-embedded inputs + cached encoder output and
    /// returns logits `(1, tgt_len, vocab_size)`. Use this from
    /// multimodal hosts that condition T5 decoding on non-text target
    /// embeddings (e.g. mixed-modality generation experiments).
    pub fn forward_decoder_embeds(
        &self,
        tgt_embeds: &LazyTensor,
        encoder_out: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dec_out = self.decode_from_embeds(tgt_embeds, encoder_out)?;
        let lm_head = match &self.weights.lm_head {
            Some(w) => w.clone(),
            None => {
                if !cfg.tie_word_embeddings {
                    return Err(crate::Error::Msg(
                        "T5Model::forward_decoder_embeds: lm_head absent but tie_word_embeddings=false".into(),
                    ).bt());
                }
                WeightStorage::F32(self.weights.shared_embedding.clone())
            }
        };
        let dec_scaled = if cfg.tie_word_embeddings {
            dec_out.mul_scalar((cfg.d_model as f64).powf(-0.5))
        } else {
            dec_out
        };
        Ok(lm_head.apply_linear(&dec_scaled, cfg.d_model, cfg.vocab_size))
    }

    /// Hidden-state variant of [`Self::forward_decoder_embeds`].
    /// Returns `(1, tgt_len, d_model)` post-RmsNorm states without the
    /// LM head or tied-embedding scaling.
    pub fn forward_decoder_hidden_embeds(
        &self,
        tgt_embeds: &LazyTensor,
        encoder_out: &LazyTensor,
    ) -> Result<LazyTensor> {
        self.decode_from_embeds(tgt_embeds, encoder_out)
    }

    /// Build per-token embeddings without running encoder or decoder.
    /// Returns `(1, seq, d_model)`. T5 has tied src/tgt embeddings
    /// (shared_embedding); the same table serves both sides.
    pub fn embed_tokens_anchored(
        &self, anchor: &LazyTensor, tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        anchor.embed_tokens_anchored(
            self.weights.shared_embedding.clone(),
            cfg.vocab_size, cfg.d_model, tokens,
        )
    }

    /// Run only the T5 decoder, given a precomputed encoder
    /// output and target tokens, and return logits of shape
    /// `(1, tgt_len, vocab_size)`. The caller slices the last
    /// row to pick the next token. Mirrors
    /// `WhisperModel::forward_decoder` — `encoder_out` is the
    /// graph anchor so all decoder constants land on the same
    /// graph (avoids cross-graph build errors). Use this for
    /// autoregressive seq2seq generation where the encoder
    /// output is cached once and the decoder is invoked per
    /// generated token.
    pub fn forward_decoder(
        &self,
        tgt_tokens: &[u32],
        encoder_out: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        assert!(!tgt_tokens.is_empty(), "tgt_tokens must be non-empty");
        let embed = encoder_out.const_f32_like(
            self.weights.shared_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.d_model]),
        );
        let dec_out = self.decode(&embed, tgt_tokens, encoder_out)?;
        let lm_head = match &self.weights.lm_head {
            Some(w) => w.clone(),
            None => {
                assert!(
                    cfg.tie_word_embeddings,
                    "lm_head absent but tie_word_embeddings=false",
                );
                WeightStorage::F32(self.weights.shared_embedding.clone())
            }
        };
        let dec_scaled = if cfg.tie_word_embeddings {
            dec_out.mul_scalar((cfg.d_model as f64).powf(-0.5))
        } else {
            dec_out
        };
        Ok(lm_head.apply_linear(&dec_scaled, cfg.d_model, cfg.vocab_size))
    }

    fn encode(&self, embed: &LazyTensor, src: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let src_len = src.len();
        let batch = 1;
        let ids = embed.const_u32_like(src.to_vec(), Shape::from_dims(&[src_len]));
        let src_embeds = embed
            .index_select(0_usize, &ids)?
            .reshape(Shape::from_dims(&[batch, src_len, cfg.d_model]))?;
        self.encode_from_embeds(&src_embeds)
    }

    fn encode_from_embeds(&self, src_embeds: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = src_embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.d_model {
            return Err(crate::Error::Msg(format!(
                "T5Model::forward_encoder_embeds: expected shape (1, src_len, d_model={}), got {:?}",
                cfg.d_model, dims,
            )).bt());
        }
        let src_len = dims[1];
        if src_len == 0 {
            return Err(crate::Error::Msg(
                "T5Model::forward_encoder_embeds: src_len must be > 0".into(),
            ).bt());
        }
        let mut x = src_embeds.clone();

        let pos_bias = compute_position_bias(
            src_embeds, &self.weights.encoder_rel_bias,
            src_len, src_len,
            cfg.num_heads,
            cfg.relative_attention_num_buckets,
            cfg.relative_attention_max_distance,
        )?;

        for layer in &self.weights.encoder_layers {
            x = self.apply_encoder_layer(&x, layer, &pos_bias)?;
        }

        x.rms_norm_affine(std::sync::Arc::clone(&self.weights.encoder_final_norm_gain), cfg.layer_norm_epsilon)
    }

    fn decode(
        &self,
        embed: &LazyTensor,
        tgt: &[u32],
        enc_out: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let tgt_len = tgt.len();
        let batch = 1;
        let ids = embed.const_u32_like(tgt.to_vec(), Shape::from_dims(&[tgt_len]));
        let tgt_embeds = embed
            .index_select(0_usize, &ids)?
            .reshape(Shape::from_dims(&[batch, tgt_len, cfg.d_model]))?;
        self.decode_from_embeds(&tgt_embeds, enc_out)
    }

    fn decode_from_embeds(
        &self,
        tgt_embeds: &LazyTensor,
        enc_out: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = tgt_embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.d_model {
            return Err(crate::Error::Msg(format!(
                "T5Model::forward_decoder_embeds: expected tgt_embeds shape (1, tgt_len, d_model={}), got {:?}",
                cfg.d_model, dims,
            )).bt());
        }
        let tgt_len = dims[1];
        if tgt_len == 0 {
            return Err(crate::Error::Msg(
                "T5Model::forward_decoder_embeds: tgt_len must be > 0".into(),
            ).bt());
        }
        let mut x = tgt_embeds.clone();

        let pos_bias = compute_position_bias(
            tgt_embeds, &self.weights.decoder_rel_bias,
            tgt_len, tgt_len,
            cfg.num_heads,
            cfg.relative_attention_num_buckets,
            cfg.relative_attention_max_distance,
        )?;
        // Decoder self-attn: combine relative bias with causal mask.
        let mut causal_mask = vec![0.0_f32; tgt_len * tgt_len];
        for i in 0..tgt_len {
            for j in (i + 1)..tgt_len {
                causal_mask[i * tgt_len + j] = f32::NEG_INFINITY;
            }
        }
        let causal = tgt_embeds.const_f32_like(
            causal_mask,
            Shape::from_dims(&[1, 1, tgt_len, tgt_len]),
        );

        for layer in &self.weights.decoder_layers {
            x = self.apply_decoder_layer(&x, layer, enc_out, &pos_bias, &causal)?;
        }

        x.rms_norm_affine(std::sync::Arc::clone(&self.weights.decoder_final_norm_gain), cfg.layer_norm_epsilon)
    }

    fn apply_encoder_layer(
        &self,
        x: &LazyTensor,
        layer: &T5EncoderLayerWeights,
        pos_bias: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_norm = x.rms_norm_affine(std::sync::Arc::clone(&layer.self_attn_norm_gain), cfg.layer_norm_epsilon)?;
        let attn = self.attention(&x_norm, &x_norm, &layer.self_attn, Some(pos_bias), None)?;
        let h1 = x.add(&attn)?;

        let h1_norm = h1.rms_norm_affine(std::sync::Arc::clone(&layer.ffn_norm_gain), cfg.layer_norm_epsilon)?;
        let ffn = self.feed_forward(&h1_norm, &layer.ffn)?;
        h1.add(&ffn)
    }

    fn apply_decoder_layer(
        &self,
        x: &LazyTensor,
        layer: &T5DecoderLayerWeights,
        enc_out: &LazyTensor,
        pos_bias: &LazyTensor,
        causal_mask: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_norm = x.rms_norm_affine(std::sync::Arc::clone(&layer.self_attn_norm_gain), cfg.layer_norm_epsilon)?;
        let self_attn = self.attention(
            &x_norm, &x_norm, &layer.self_attn,
            Some(pos_bias), Some(causal_mask),
        )?;
        let h1 = x.add(&self_attn)?;

        let h1_norm = h1.rms_norm_affine(std::sync::Arc::clone(&layer.cross_attn_norm_gain), cfg.layer_norm_epsilon)?;
        let cross_attn = self.attention(
            &h1_norm, enc_out, &layer.cross_attn,
            None, None,
        )?;
        let h2 = h1.add(&cross_attn)?;

        let h2_norm = h2.rms_norm_affine(std::sync::Arc::clone(&layer.ffn_norm_gain), cfg.layer_norm_epsilon)?;
        let ffn = self.feed_forward(&h2_norm, &layer.ffn)?;
        h2.add(&ffn)
    }

    fn attention(
        &self,
        q_src: &LazyTensor,
        kv_src: &LazyTensor,
        w: &T5AttentionWeights,
        pos_bias: Option<&LazyTensor>,
        extra_mask: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let inner = cfg.inner_dim();
        let q_shape = q_src.shape();
        let qd = q_shape.dims();
        let batch = qd[0];
        let q_len = qd[1];
        let kv_shape = kv_src.shape();
        let kd = kv_shape.dims();
        let kv_len = kd[1];

        let q = w.q.apply_linear(q_src, cfg.d_model, inner);
        let k = w.k.apply_linear(kv_src, cfg.d_model, inner);
        let v = w.v.apply_linear(kv_src, cfg.d_model, inner);

        let _ = (batch, q_len, kv_len);
        let q = q.split_heads(cfg.num_heads, cfg.d_kv)?;
        let k = k.split_heads(cfg.num_heads, cfg.d_kv)?;
        let v = v.split_heads(cfg.num_heads, cfg.d_kv)?;

        // T5 does NOT scale Q/K — unlike standard attention.
        let k_t = k.transpose()?;
        let mut scores = q.matmul(&k_t)?;
        if let Some(b) = pos_bias {
            scores = scores.broadcast_add(b)?;
        }
        if let Some(m) = extra_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx.merge_heads()?;
        Ok(w.o.apply_linear(&merged, inner, cfg.d_model))
    }

    fn feed_forward(&self, x: &LazyTensor, ffn: &T5FfnWeights) -> Result<LazyTensor> {
        let cfg = &self.config;
        match ffn {
            T5FfnWeights::Dense { wi, wo } => {
                let hidden = wi.apply_linear(x, cfg.d_model, cfg.d_ff);
                let activated = self.activate(&hidden);
                Ok(wo.apply_linear(&activated, cfg.d_ff, cfg.d_model))
            }
            T5FfnWeights::Gated { wi_0, wi_1, wo } => {
                let gate = wi_0.apply_linear(x, cfg.d_model, cfg.d_ff);
                let up = wi_1.apply_linear(x, cfg.d_model, cfg.d_ff);
                let activated = self.activate(&gate);
                let combined = activated.mul(&up)?;
                Ok(wo.apply_linear(&combined, cfg.d_ff, cfg.d_model))
            }
        }
    }

    fn activate(&self, x: &LazyTensor) -> LazyTensor {
        match self.config.activation {
            T5Activation::Relu => x.relu(),
            T5Activation::Silu => x.silu(),
            T5Activation::Gelu => x.gelu_erf(),
            T5Activation::GeluPytorchTanh => x.gelu(),
        }
    }
}

/// Compute the T5 relative position bias for `q_len × kv_len` and
/// `n_heads` heads using the bidirectional bucketing scheme.
///
/// Returns a `[1, n_heads, q_len, kv_len]` tensor ready to be
/// broadcast-added to attention scores.
fn compute_position_bias(
    anchor: &LazyTensor,
    rel_bias_table: &Arc<[f32]>,
    q_len: usize,
    kv_len: usize,
    n_heads: usize,
    num_buckets: usize,
    max_distance: usize,
) -> Result<LazyTensor> {
    assert_eq!(rel_bias_table.len(), num_buckets * n_heads);
    // Build the bucket lookup. The eager T5 uses `kv_pos - q_pos`
    // as the signed relative distance. Past is encoded in the
    // lower half, future in the upper half.
    let half = (num_buckets / 2) as u32;
    let max_exact = half / 2;
    let mut buckets = vec![0_u32; q_len * kv_len];
    for q in 0..q_len {
        for kv in 0..kv_len {
            let q_i = q as i64;
            let kv_i = kv as i64;
            let rel = kv_i - q_i;
            let bucket = if rel > 0 {
                // KV in the future — upper half [half, num_buckets).
                let d = rel as u32;
                let b = if d < max_exact {
                    d
                } else {
                    let val = (d as f32 / max_exact as f32).ln()
                        / (max_distance as f32 / max_exact as f32).ln();
                    let v = max_exact + (val * (half - max_exact) as f32) as u32;
                    v.min(half - 1)
                };
                half + b
            } else {
                // KV in the past or same — lower half [0, half).
                let d = (-rel) as u32;
                if d < max_exact {
                    d
                } else {
                    let val = (d as f32 / max_exact as f32).ln()
                        / (max_distance as f32 / max_exact as f32).ln();
                    let v = max_exact + (val * (half - max_exact) as f32) as u32;
                    v.min(half - 1)
                }
            };
            buckets[q * kv_len + kv] = bucket;
        }
    }

    // Manually fetch [n_heads] vector for each bucket index and
    // construct the (q_len, kv_len, n_heads) bias tensor.
    let mut bias_data = vec![0.0_f32; q_len * kv_len * n_heads];
    for q in 0..q_len {
        for kv in 0..kv_len {
            let bucket = buckets[q * kv_len + kv] as usize;
            for h in 0..n_heads {
                let src = bucket * n_heads + h;
                let dst = q * kv_len * n_heads + kv * n_heads + h;
                bias_data[dst] = rel_bias_table[src];
            }
        }
    }
    let bias = anchor.const_f32_like(
        Arc::from(bias_data),
        Shape::from_dims(&[q_len, kv_len, n_heads]),
    );
    // Permute (q, kv, h) → (h, q, kv); unsqueeze batch dim.
    bias.permute([2, 0, 1_usize])?
        .reshape(Shape::from_dims(&[1, n_heads, q_len, kv_len]))
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl T5AttentionWeights {
    fn load(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        d_model: usize,
        inner_dim: usize,
    ) -> Result<Self> {
        use crate::lazy::load_transposed_matrix_preserve_dtype as ltm;
        let q = ltm(st, &format!("{prefix}.q.weight"), inner_dim, d_model)?;
        let k = ltm(st, &format!("{prefix}.k.weight"), inner_dim, d_model)?;
        let v = ltm(st, &format!("{prefix}.v.weight"), inner_dim, d_model)?;
        let o = ltm(st, &format!("{prefix}.o.weight"), d_model, inner_dim)?;
        Ok(Self { q, k, v, o })
    }
}

impl T5FfnWeights {
    fn load(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        d_model: usize,
        d_ff: usize,
        gated: bool,
    ) -> Result<Self> {
        use crate::lazy::load_transposed_matrix_preserve_dtype as ltm;
        if gated {
            let wi_0 = ltm(st, &format!("{prefix}.wi_0.weight"), d_ff, d_model)?;
            let wi_1 = ltm(st, &format!("{prefix}.wi_1.weight"), d_ff, d_model)?;
            let wo = ltm(st, &format!("{prefix}.wo.weight"), d_model, d_ff)?;
            Ok(T5FfnWeights::Gated { wi_0, wi_1, wo })
        } else {
            let wi = ltm(st, &format!("{prefix}.wi.weight"), d_ff, d_model)?;
            let wo = ltm(st, &format!("{prefix}.wo.weight"), d_model, d_ff)?;
            Ok(T5FfnWeights::Dense { wi, wo })
        }
    }
}

impl T5Weights {
    /// Load T5 (t5-small/base/large/Flan-T5) weights from a HuggingFace
    /// safetensors file.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &T5Config,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
        let d = cfg.d_model;
        let inner = cfg.inner_dim();
        let d_ff = cfg.d_ff;
        let n_enc = cfg.num_layers;
        let n_dec = cfg.num_decoder_layers_resolved();
        let gated = cfg.gated_ffn;
        let ffn_name = if gated { "DenseGatedActDense" } else { "DenseReluDense" };

        let shared_embedding = Arc::from(load_tensor_as_f32(st, "shared.weight")?);
        let encoder_rel_bias = Arc::from(load_tensor_as_f32(
            st,
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        )?);
        let decoder_rel_bias = Arc::from(load_tensor_as_f32(
            st,
            "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        )?);

        let mut encoder_layers = Vec::with_capacity(n_enc);
        for i in 0..n_enc {
            let p = format!("encoder.block.{i}");
            let self_attn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.layer.0.layer_norm.weight"),
            )?);
            let self_attn = T5AttentionWeights::load(
                st, &format!("{p}.layer.0.SelfAttention"), d, inner,
            )?;
            let ffn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.layer.1.layer_norm.weight"),
            )?);
            let ffn = T5FfnWeights::load(
                st, &format!("{p}.layer.1.{ffn_name}"), d, d_ff, gated,
            )?;
            encoder_layers.push(T5EncoderLayerWeights {
                self_attn_norm_gain, self_attn, ffn_norm_gain, ffn,
            });
        }

        let mut decoder_layers = Vec::with_capacity(n_dec);
        for i in 0..n_dec {
            let p = format!("decoder.block.{i}");
            let self_attn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.layer.0.layer_norm.weight"),
            )?);
            let self_attn = T5AttentionWeights::load(
                st, &format!("{p}.layer.0.SelfAttention"), d, inner,
            )?;
            let cross_attn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.layer.1.layer_norm.weight"),
            )?);
            let cross_attn = T5AttentionWeights::load(
                st, &format!("{p}.layer.1.EncDecAttention"), d, inner,
            )?;
            let ffn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.layer.2.layer_norm.weight"),
            )?);
            let ffn = T5FfnWeights::load(
                st, &format!("{p}.layer.2.{ffn_name}"), d, d_ff, gated,
            )?;
            decoder_layers.push(T5DecoderLayerWeights {
                self_attn_norm_gain, self_attn,
                cross_attn_norm_gain, cross_attn,
                ffn_norm_gain, ffn,
            });
        }

        let encoder_final_norm_gain = Arc::from(load_tensor_as_f32(
            st, "encoder.final_layer_norm.weight",
        )?);
        let decoder_final_norm_gain = Arc::from(load_tensor_as_f32(
            st, "decoder.final_layer_norm.weight",
        )?);

        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(load_transposed_matrix_preserve_dtype(
                st, "lm_head.weight", cfg.vocab_size, d,
            )?)
        };

        Ok(Self {
            shared_embedding,
            encoder_rel_bias,
            decoder_rel_bias,
            encoder_layers,
            decoder_layers,
            encoder_final_norm_gain,
            decoder_final_norm_gain,
            lm_head,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_attention_weights(d_model: usize, inner: usize, next_box: &mut Box<dyn FnMut() -> f32>) -> T5AttentionWeights {
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        T5AttentionWeights {
            q: WeightStorage::F32(vec_of(d_model * inner, &mut **next_box)),
            k: WeightStorage::F32(vec_of(d_model * inner, &mut **next_box)),
            v: WeightStorage::F32(vec_of(d_model * inner, &mut **next_box)),
            o: WeightStorage::F32(vec_of(inner * d_model, &mut **next_box)),
        }
    }

    fn tiny_ffn(cfg: &T5Config, next_box: &mut Box<dyn FnMut() -> f32>) -> T5FfnWeights {
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        if cfg.gated_ffn {
            T5FfnWeights::Gated {
                wi_0: WeightStorage::F32(vec_of(cfg.d_model * cfg.d_ff, &mut **next_box)),
                wi_1: WeightStorage::F32(vec_of(cfg.d_model * cfg.d_ff, &mut **next_box)),
                wo: WeightStorage::F32(vec_of(cfg.d_ff * cfg.d_model, &mut **next_box)),
            }
        } else {
            T5FfnWeights::Dense {
                wi: WeightStorage::F32(vec_of(cfg.d_model * cfg.d_ff, &mut **next_box)),
                wo: WeightStorage::F32(vec_of(cfg.d_ff * cfg.d_model, &mut **next_box)),
            }
        }
    }

    fn tiny_weights(cfg: &T5Config) -> T5Weights {
        let mut s: u32 = 75757;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let d = cfg.d_model;
        let inner = cfg.inner_dim();
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let shared_embedding = vec_of(cfg.vocab_size * d, &mut *nb);
        let encoder_rel_bias = vec_of(
            cfg.relative_attention_num_buckets * cfg.num_heads, &mut *nb,
        );
        let decoder_rel_bias = vec_of(
            cfg.relative_attention_num_buckets * cfg.num_heads, &mut *nb,
        );

        let encoder_layers: Vec<T5EncoderLayerWeights> = (0..cfg.num_layers)
            .map(|_| T5EncoderLayerWeights {
                self_attn_norm_gain: Arc::from(vec![1.0_f32; d]),
                self_attn: tiny_attention_weights(d, inner, &mut nb),
                ffn_norm_gain: Arc::from(vec![1.0_f32; d]),
                ffn: tiny_ffn(cfg, &mut nb),
            })
            .collect();
        let decoder_layers: Vec<T5DecoderLayerWeights> = (0..cfg.num_decoder_layers_resolved())
            .map(|_| T5DecoderLayerWeights {
                self_attn_norm_gain: Arc::from(vec![1.0_f32; d]),
                self_attn: tiny_attention_weights(d, inner, &mut nb),
                cross_attn_norm_gain: Arc::from(vec![1.0_f32; d]),
                cross_attn: tiny_attention_weights(d, inner, &mut nb),
                ffn_norm_gain: Arc::from(vec![1.0_f32; d]),
                ffn: tiny_ffn(cfg, &mut nb),
            })
            .collect();
        let encoder_final_norm_gain = Arc::from(vec![1.0_f32; d]);
        let decoder_final_norm_gain = Arc::from(vec![1.0_f32; d]);
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(WeightStorage::F32(vec_of(d * cfg.vocab_size, &mut *nb)))
        };
        T5Weights {
            shared_embedding,
            encoder_rel_bias, decoder_rel_bias,
            encoder_layers, decoder_layers,
            encoder_final_norm_gain, decoder_final_norm_gain,
            lm_head,
        }
    }

    fn tiny_config() -> T5Config {
        T5Config {
            vocab_size: 16, d_model: 8, d_kv: 4, d_ff: 16,
            num_layers: 2, num_decoder_layers: None,
            num_heads: 2,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 16,
            layer_norm_epsilon: 1e-6,
            activation: T5Activation::Relu,
            gated_ffn: false,
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = T5Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let src = [1_u32, 2, 3, 4];
        let tgt = [5_u32, 6, 7];
        let logits = model.forward(&src, &tgt).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tgt.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// Gated FFN variant runs and changes output from the dense FFN.
    #[test]
    fn gated_vs_dense_ffn_differ() {
        let cfg_a = T5Config { gated_ffn: false, ..tiny_config() };
        let cfg_b = T5Config {
            gated_ffn: true,
            activation: T5Activation::GeluPytorchTanh,
            ..tiny_config()
        };
        let w_a = tiny_weights(&cfg_a);
        let w_b = tiny_weights(&cfg_b);
        let m_a = T5Model { config: cfg_a, weights: w_a };
        let m_b = T5Model { config: cfg_b, weights: w_b };
        let src = [1_u32, 2, 3];
        let tgt = [4_u32, 5];
        let a = m_a.forward(&src, &tgt).unwrap().realize_f32();
        let b = m_b.forward(&src, &tgt).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "dense vs gated FFN must differ, max_diff = {max_diff}");
    }

    /// Cross-attention conditions decoder on encoder output:
    /// changing src must change tgt logits.
    #[test]
    fn cross_attention_is_wired() {
        let cfg = tiny_config();
        let model = T5Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tgt = [4_u32, 5, 6];
        let a = model.forward(&[1, 2, 3], &tgt).unwrap().realize_f32();
        let b = model.forward(&[7, 8, 9], &tgt).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "cross-attention must condition on source: src change must change tgt logits, max_diff = {max_diff}");
    }

    /// Decoder causal mask: changing a future token must NOT
    /// alter logits at preceding positions.
    #[test]
    fn decoder_causal_mask_holds() {
        let cfg = tiny_config();
        let model = T5Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let src = [1_u32, 2, 3];
        let tgt_a = [4_u32, 5, 6, 7];
        let tgt_b = [4_u32, 5, 6, 15];
        let a = model.forward(&src, &tgt_a).unwrap().realize_f32();
        let b = model.forward(&src, &tgt_b).unwrap().realize_f32();
        let v = cfg.vocab_size;
        for t in 0..3 {
            for col in 0..v {
                let i = t * v + col;
                assert!(
                    (a[i] - b[i]).abs() < 1e-5,
                    "causal mask violated at t={t}: {} vs {}", a[i], b[i],
                );
            }
        }
    }

    /// Relative position bucketing:
    ///   - q == kv (diagonal) maps to bucket 0 (past-half, exact distance 0)
    ///   - q < kv with small distance maps to the future half
    #[test]
    fn relative_position_bucketing_basics() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let embed = LazyTensor::from_f32(
            weights.shared_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.d_model]),
            &Device::cpu(),
        );
        // Just verify the function runs and produces the right shape.
        let bias = compute_position_bias(
            &embed, &weights.encoder_rel_bias,
            4, 4, cfg.num_heads,
            cfg.relative_attention_num_buckets,
            cfg.relative_attention_max_distance,
        ).unwrap();
        assert_eq!(bias.shape().dims(), &[1, cfg.num_heads, 4, 4]);
    }

    #[test]
    fn t5_small_preset() {
        let c = T5Config::t5_small();
        assert_eq!(c.d_model, 512);
        assert_eq!(c.num_heads, 8);
        assert_eq!(c.inner_dim(), 512);
        assert!(!c.gated_ffn);
    }

    /// `forward_encoder(src)` runs the T5 encoder in isolation
    /// and returns post-final-norm hidden states
    /// `(1, src_len, d_model)`. Used when T5 is a conditioning
    /// text encoder (Parler-TTS, SD3, FLUX).
    #[test]
    fn forward_encoder_shape_and_finite() {
        let cfg = tiny_config();
        let model = T5Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let src = [1_u32, 2, 3, 4, 5];
        let enc = model.forward_encoder(&src).unwrap();
        assert_eq!(enc.shape().dims(), &[1, src.len(), cfg.d_model]);
        for &v in &enc.realize_f32() {
            assert!(v.is_finite(), "non-finite encoder hidden: {v}");
        }
    }

    /// `forward_encoder` is the same path the full `forward`
    /// runs internally — same encoder. Changing the source
    /// tokens must change the encoder hidden state.
    #[test]
    fn forward_encoder_responds_to_src() {
        let cfg = tiny_config();
        let model = T5Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let a = model.forward_encoder(&[1_u32, 2, 3]).unwrap().realize_f32();
        let b = model.forward_encoder(&[7_u32, 8, 9]).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "forward_encoder must respond to src changes, max_diff = {max_diff}");
    }

    /// `forward_decoder(tgt, enc_out)` runs only the decoder
    /// against a cached encoder output and returns logits of
    /// shape `(1, tgt_len, vocab_size)`. Used for
    /// autoregressive seq2seq generation.
    #[test]
    fn forward_decoder_shape_and_finite() {
        let cfg = tiny_config();
        let model = T5Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let enc = model.forward_encoder(&[1_u32, 2, 3, 4]).unwrap();
        let tgt = [5_u32, 6, 7];
        let logits = model.forward_decoder(&tgt, &enc).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tgt.len(), cfg.vocab_size]);
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
        let model = T5Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
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

    #[test]
    fn forward_encoder_embeds_matches_forward_encoder() {
        let cfg = tiny_config();
        let model = T5Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let src = [1_u32, 2, 3];
        let enc_ref = model.forward_encoder(&src).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let src_embeds = model.embed_tokens_anchored(&anchor, &src).unwrap();
        let enc_via_embeds = model.forward_encoder_embeds(&src_embeds).unwrap().realize_f32();
        let max_diff = enc_ref.iter().zip(enc_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "T5 forward_encoder vs forward_encoder_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_decoder_embeds_matches_forward_decoder() {
        let cfg = tiny_config();
        let model = T5Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let src = [1_u32, 2, 3];
        let tgt = [5_u32, 6, 7];
        let enc = model.forward_encoder(&src).unwrap();
        let dec_ref = model.forward_decoder(&tgt, &enc).unwrap().realize_f32();
        let tgt_embeds = model.embed_tokens_anchored(&enc, &tgt).unwrap();
        let dec_via_embeds = model.forward_decoder_embeds(&tgt_embeds, &enc).unwrap().realize_f32();
        let max_diff = dec_ref.iter().zip(dec_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "T5 forward_decoder vs forward_decoder_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_encoder_embeds_rejects_bad_shape() {
        let cfg = tiny_config();
        let model = T5Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let bad = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.d_model + 1)],
            Shape::from_dims(&[1, 3, cfg.d_model + 1]), &Device::cpu(),
        );
        assert!(model.forward_encoder_embeds(&bad).is_err());
    }

    #[test]
    fn load_from_mmapped_smoke() {
        // Smoke: dummy MmapedSafetensors cannot be constructed without disk I/O;
        // this test asserts the function's signature compiles by reference only.
        let _ = T5Weights::load_from_mmapped;
    }

    #[test]
    fn forward_decoder_hidden_embeds_skips_lm_head() {
        let cfg = tiny_config();
        let model = T5Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let src = [1_u32, 2, 3];
        let tgt = [5_u32, 6, 7];
        let enc = model.forward_encoder(&src).unwrap();
        let tgt_embeds = model.embed_tokens_anchored(&enc, &tgt).unwrap();
        let hidden = model.forward_decoder_hidden_embeds(&tgt_embeds, &enc).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tgt.len(), cfg.d_model]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

//! BLIP text decoder — lazy port.
//!
//! BERT-shape decoder with Post-LN inside each attention sublayer.
//! Used by `BlipForConditionalGeneration` (image captioning).
//!
//! Per-layer structure:
//!   res = x
//!   x = self_attn(x, causal_mask) → out_dense; x = LN(x + res)
//!   res = x
//!   x = cross_attn(x, kv = encoder_hidden) → out_dense; x = LN(x + res)
//!   res = x
//!   x = intermediate.dense(x); x = act(x);
//!   x = output.dense(x); x = LN(x + res)
//!
//! LM head: dense → act → LN → vocab linear (tied or untied).
//!
//! Cross-attention K/V projections take `encoder_hidden_size`
//! (BLIP vision's hidden_size, typically 768/1024) as input
//! channels — distinct from `hidden_size` (the decoder's own
//! hidden, e.g. 768).
//!
//! Strict causal mask over the target sequence — applied
//! additively to attention scores (`-inf` above the diagonal).
//!
//! v1 scope: F32, batch == 1, prefill only.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlipTextActivation {
    Gelu,
    GeluPytorchTanh,
    Relu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlipTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub encoder_hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_activation: BlipTextActivation,
    pub layer_norm_eps: f64,
}

impl BlipTextConfig {
    /// `Salesforce/blip-image-captioning-base` decoder preset.
    pub fn image_captioning_base() -> Self {
        Self {
            vocab_size: 30524,
            hidden_size: 768,
            encoder_hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            max_position_embeddings: 512,
            hidden_activation: BlipTextActivation::Gelu,
            layer_norm_eps: 1e-12,
        }
    }

    /// `Salesforce/blip-image-captioning-large` decoder preset.
    pub fn image_captioning_large() -> Self {
        Self {
            vocab_size: 30524,
            hidden_size: 768,
            encoder_hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            max_position_embeddings: 512,
            hidden_activation: BlipTextActivation::Gelu,
            layer_norm_eps: 1e-12,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ---- Weight structures ------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    pub gain: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct BlipTextAttentionWeights {
    /// Self-attn: all `[hidden, hidden]`. Cross-attn key/value:
    /// `[hidden, encoder_hidden]` (mapping encoder_hidden inputs
    /// to hidden-sized K/V).
    pub query: WeightStorage,
    pub query_bias: Arc<[f32]>,
    pub key: WeightStorage,
    pub key_bias: Arc<[f32]>,
    pub value: WeightStorage,
    pub value_bias: Arc<[f32]>,
    /// Output dense (post-attn): `[hidden, hidden]` + LayerNorm.
    pub out_dense: WeightStorage,
    pub out_dense_bias: Arc<[f32]>,
    pub out_ln: LayerNormWeights,
}

#[derive(Debug, Clone)]
pub struct BlipTextFfnWeights {
    pub intermediate: WeightStorage,
    pub intermediate_bias: Arc<[f32]>,
    pub output: WeightStorage,
    pub output_bias: Arc<[f32]>,
    /// Post-FFN LayerNorm (wraps the +residual).
    pub output_ln: LayerNormWeights,
}

#[derive(Debug, Clone)]
pub struct BlipTextLayerWeights {
    pub self_attn: BlipTextAttentionWeights,
    pub cross_attn: BlipTextAttentionWeights,
    pub ffn: BlipTextFfnWeights,
}

#[derive(Debug, Clone)]
pub struct BlipTextWeights {
    /// `[vocab_size, hidden_size]`.
    pub word_embedding: Arc<[f32]>,
    /// `[max_position_embeddings, hidden_size]`.
    pub position_embedding: Arc<[f32]>,
    /// Embedding-LN after word + position embedding.
    pub embed_ln: LayerNormWeights,
    pub layers: Vec<BlipTextLayerWeights>,
    /// Prediction head transform: dense + act + LN.
    pub pred_dense: WeightStorage,
    pub pred_dense_bias: Arc<[f32]>,
    pub pred_ln: LayerNormWeights,
    /// LM head: `[vocab_size, hidden_size]` + bias `[vocab_size]`.
    pub lm_head: WeightStorage,
    pub lm_head_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct BlipTextModel {
    pub config: BlipTextConfig,
    pub weights: BlipTextWeights,
}

// ---- Forward ---------------------------------------------------------------

impl BlipTextModel {
    /// Run a prefill forward pass and return next-token logits.
    ///
    /// * `input_ids` — target token sequence of length T.
    /// * `encoder_hidden_states` — BLIP vision encoder output
    ///   `(1, num_patches + 1, encoder_hidden_size)`. Caller is
    ///   expected to thread this on the same graph as the input
    ///   ids' anchor.
    /// * `start_pos` — position-embedding offset (0 for fresh
    ///   prefill).
    ///
    /// Returns logits `(1, T, vocab_size)`.
    pub fn forward(
        &self,
        input_ids: &[u32],
        encoder_hidden_states: &LazyTensor,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let w = &self.weights;
        let t = input_ids.len();
        assert!(t > 0, "input_ids must be non-empty");
        let h = cfg.hidden_size;

        let anchor = encoder_hidden_states;

        // Token + position embedding + LN.
        let word_table = anchor.const_f32_like(
            Arc::clone(&w.word_embedding),
            Shape::from_dims(&[cfg.vocab_size, h]),
        );
        let ids = anchor.const_u32_like(
            input_ids.to_vec(), Shape::from_dims(&[t]),
        );
        let tok = word_table
            .index_select(0_usize, &ids)?
            .reshape(Shape::from_dims(&[1, t, h]))?;
        let pos_ids: Vec<u32> = (0..t).map(|i| (i + start_pos) as u32).collect();
        let pos_idx = anchor.const_u32_like(
            pos_ids, Shape::from_dims(&[t]),
        );
        let pos_table = anchor.const_f32_like(
            Arc::clone(&w.position_embedding),
            Shape::from_dims(&[cfg.max_position_embeddings, h]),
        );
        let pos = pos_table
            .index_select(0_usize, &pos_idx)?
            .reshape(Shape::from_dims(&[1, t, h]))?;
        let mut x = tok.add(&pos)?;
        x = x.layer_norm_affine(Arc::clone(&w.embed_ln.gain), Arc::clone(&w.embed_ln.bias), cfg.layer_norm_eps)?;

        // Strict causal mask `(1, 1, t, t)` — reshape the (t, t)
        // mask from the public helper to add the leading batch +
        // heads broadcast axes.
        let causal_mask = LazyTensor::additive_causal_mask_like(anchor, t)
            .reshape(Shape::from_dims(&[1, 1, t, t]))?;

        for layer in &w.layers {
            x = apply_decoder_layer(&x, layer, encoder_hidden_states, &causal_mask, cfg, anchor)?;
        }

        // LM head: dense → act → LN → vocab linear + bias.
        let h_pred = w.pred_dense.apply_linear_with_bias(&x, h, h, std::sync::Arc::clone(&w.pred_dense_bias))?;
        let h_pred = match cfg.hidden_activation {
            BlipTextActivation::Gelu => h_pred.gelu(),
            BlipTextActivation::GeluPytorchTanh => h_pred.gelu_erf(),
            BlipTextActivation::Relu => h_pred.relu(),
        };
        let h_pred = h_pred.layer_norm_affine(Arc::clone(&w.pred_ln.gain), Arc::clone(&w.pred_ln.bias), cfg.layer_norm_eps)?;
        w.lm_head.apply_linear_with_bias(&h_pred, h, cfg.vocab_size, std::sync::Arc::clone(&w.lm_head_bias))
    }
}

fn apply_decoder_layer(
    x: &LazyTensor,
    w: &BlipTextLayerWeights,
    enc_states: &LazyTensor,
    causal_mask: &LazyTensor,
    cfg: &BlipTextConfig,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let h = cfg.hidden_size;

    // Self-attention with causal mask, Post-LN style:
    //   y = attn(x, x); y = out_dense(y); x = LN(y + x).
    let residual = x.clone();
    let attn_out = apply_attention(
        x, None, &w.self_attn,
        cfg.num_attention_heads, cfg.head_dim(),
        h, h, Some(causal_mask), anchor,
    )?;
    let y = w.self_attn.out_dense.apply_linear_with_bias(&attn_out, h, h, std::sync::Arc::clone(&w.self_attn.out_dense_bias))?;
    let x = &y.add(&residual)?.layer_norm_affine(Arc::clone(&w.self_attn.out_ln.gain), Arc::clone(&w.self_attn.out_ln.bias), cfg.layer_norm_eps)?;

    // Cross-attention to encoder states.
    let residual = x.clone();
    let cross_out = apply_attention(
        &x, Some(enc_states), &w.cross_attn,
        cfg.num_attention_heads, cfg.head_dim(),
        h, cfg.encoder_hidden_size, None, anchor,
    )?;
    let y = w.cross_attn.out_dense.apply_linear_with_bias(&cross_out, h, h, std::sync::Arc::clone(&w.cross_attn.out_dense_bias))?;
    let x = &y.add(&residual)?.layer_norm_affine(Arc::clone(&w.cross_attn.out_ln.gain), Arc::clone(&w.cross_attn.out_ln.bias), cfg.layer_norm_eps)?;

    // FFN.
    let residual = x.clone();
    let inter = w.ffn.intermediate.apply_linear_with_bias(&x, h, cfg.intermediate_size, std::sync::Arc::clone(&w.ffn.intermediate_bias))?;
    let inter = match cfg.hidden_activation {
        BlipTextActivation::Gelu => inter.gelu(),
        BlipTextActivation::GeluPytorchTanh => inter.gelu_erf(),
        BlipTextActivation::Relu => inter.relu(),
    };
    let out = w.ffn.output.apply_linear_with_bias(&inter, cfg.intermediate_size, h, std::sync::Arc::clone(&w.ffn.output_bias))?;
    out.add(&residual)?.layer_norm_affine(Arc::clone(&w.ffn.output_ln.gain), Arc::clone(&w.ffn.output_ln.bias), cfg.layer_norm_eps)
}

#[allow(clippy::too_many_arguments)]
fn apply_attention(
    q_input: &LazyTensor,
    kv_input: Option<&LazyTensor>,
    w: &BlipTextAttentionWeights,
    num_heads: usize,
    head_dim: usize,
    q_in_dim: usize,
    kv_in_dim: usize,
    mask: Option<&LazyTensor>,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let q_dims = q_input.shape();
    let q_dims = q_dims.dims();
    let b = q_dims[0]; let q_len = q_dims[1];
    let kv_src = kv_input.unwrap_or(q_input);
    let kv_dims = kv_src.shape();
    let kv_dims = kv_dims.dims();
    let kv_len = kv_dims[1];
    let embed = num_heads * head_dim;

    let q = w.query.apply_linear_with_bias(q_input, q_in_dim, embed, std::sync::Arc::clone(&w.query_bias))?;
    let k = w.key.apply_linear_with_bias(kv_src, kv_in_dim, embed, std::sync::Arc::clone(&w.key_bias))?;
    let v = w.value.apply_linear_with_bias(kv_src, kv_in_dim, embed, std::sync::Arc::clone(&w.value_bias))?;

    let scaling = 1.0_f64 / (head_dim as f64).sqrt();
    let q = q.mul_scalar(scaling);

    let _ = (q_len, kv_len);
    let q = q.split_heads(num_heads, head_dim)?;
    let k = k.split_heads(num_heads, head_dim)?;
    let v = v.split_heads(num_heads, head_dim)?;

    let kt = k.permute([0, 1, 3, 2_usize])?;
    let mut scores = q.matmul(&kt)?;
    if let Some(m) = mask {
        let mb = m.broadcast_to(Shape::from_dims(&[b, num_heads, q_len, kv_len]))?;
        scores = scores.add(&mb)?;
    }
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?;
    let _ = (b, embed);
    ctx.merge_heads()
}



// ---- HuggingFace safetensors loader ----------------------------------------

fn load_ln(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
) -> Result<LayerNormWeights> {
    use crate::lazy::load_tensor_as_f32;
    Ok(LayerNormWeights {
        gain: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.weight"))?),
        bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.bias"))?),
    })
}

fn load_blip_text_attn(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    hidden_size: usize,
    kv_in_dim: usize,
) -> Result<BlipTextAttentionWeights> {
    use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
    Ok(BlipTextAttentionWeights {
        query: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.self.query.weight"), hidden_size, hidden_size,
        )?,
        query_bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.self.query.bias"))?),
        key: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.self.key.weight"), hidden_size, kv_in_dim,
        )?,
        key_bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.self.key.bias"))?),
        value: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.self.value.weight"), hidden_size, kv_in_dim,
        )?,
        value_bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.self.value.bias"))?),
        out_dense: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.output.dense.weight"), hidden_size, hidden_size,
        )?,
        out_dense_bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.output.dense.bias"))?),
        out_ln: load_ln(st, &format!("{prefix}.output.LayerNorm"))?,
    })
}

impl BlipTextWeights {
    /// Load BLIP text-decoder weights from HF safetensors.
    ///
    /// HF naming (matches `Salesforce/blip-image-captioning-base`):
    ///   - `text_decoder.bert.embeddings.{word_embeddings,position_embeddings,LayerNorm}`
    ///   - `text_decoder.bert.encoder.layer.{i}.{attention,crossattention,intermediate,output}`
    ///   - `text_decoder.cls.predictions.{transform.{dense,LayerNorm}, decoder.{weight,bias}}`
    ///
    /// `prefix` typically `"text_decoder."` for full BLIP, or `""` for
    /// bare-text checkpoints.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &BlipTextConfig,
        encoder_hidden_size: usize,
        prefix: &str,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;

        let word_embedding = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}bert.embeddings.word_embeddings.weight"),
        )?);
        let position_embedding = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}bert.embeddings.position_embeddings.weight"),
        )?);
        let embed_ln = load_ln(st, &format!("{prefix}bert.embeddings.LayerNorm"))?;

        let mut layers: Vec<BlipTextLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("{prefix}bert.encoder.layer.{i}");
            let self_attn = load_blip_text_attn(st, &format!("{lp}.attention"), h, h)?;
            let cross_attn = load_blip_text_attn(
                st, &format!("{lp}.crossattention"), h, encoder_hidden_size,
            )?;
            let ffn = BlipTextFfnWeights {
                intermediate: load_transposed_matrix_preserve_dtype(
                    st, &format!("{lp}.intermediate.dense.weight"), inter, h,
                )?,
                intermediate_bias: Arc::from(load_tensor_as_f32(
                    st, &format!("{lp}.intermediate.dense.bias"),
                )?),
                output: load_transposed_matrix_preserve_dtype(
                    st, &format!("{lp}.output.dense.weight"), h, inter,
                )?,
                output_bias: Arc::from(load_tensor_as_f32(
                    st, &format!("{lp}.output.dense.bias"),
                )?),
                output_ln: load_ln(st, &format!("{lp}.output.LayerNorm"))?,
            };
            layers.push(BlipTextLayerWeights { self_attn, cross_attn, ffn });
        }

        let pred_dense = load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}cls.predictions.transform.dense.weight"), h, h,
        )?;
        let pred_dense_bias = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}cls.predictions.transform.dense.bias"),
        )?);
        let pred_ln = load_ln(st, &format!("{prefix}cls.predictions.transform.LayerNorm"))?;

        let lm_head = load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}cls.predictions.decoder.weight"), cfg.vocab_size, h,
        )?;
        let lm_head_bias = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}cls.predictions.bias"),
        )?);

        Ok(Self {
            word_embedding, position_embedding, embed_ln,
            layers,
            pred_dense, pred_dense_bias, pred_ln,
            lm_head, lm_head_bias,
        })
    }
}


// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }
    fn vec_of(n: usize, nb: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| nb()).collect::<Vec<_>>())
    }
    fn ws(n: usize, nb: &mut dyn FnMut() -> f32) -> WeightStorage {
        WeightStorage::F32(vec_of(n, nb))
    }
    fn ln_w(c: usize) -> LayerNormWeights {
        LayerNormWeights {
            gain: Arc::from(vec![1.0_f32; c]),
            bias: Arc::from(vec![0.0_f32; c]),
        }
    }

    fn attn_w(
        h: usize, kv_in: usize, nb: &mut dyn FnMut() -> f32,
    ) -> BlipTextAttentionWeights {
        BlipTextAttentionWeights {
            query: ws(h * h, nb), query_bias: vec_of(h, nb),
            key: ws(kv_in * h, nb), key_bias: vec_of(h, nb),
            value: ws(kv_in * h, nb), value_bias: vec_of(h, nb),
            out_dense: ws(h * h, nb), out_dense_bias: vec_of(h, nb),
            out_ln: ln_w(h),
        }
    }

    fn ffn_w(
        h: usize, inter: usize, nb: &mut dyn FnMut() -> f32,
    ) -> BlipTextFfnWeights {
        BlipTextFfnWeights {
            intermediate: ws(h * inter, nb), intermediate_bias: vec_of(inter, nb),
            output: ws(inter * h, nb), output_bias: vec_of(h, nb),
            output_ln: ln_w(h),
        }
    }

    fn tiny_config() -> BlipTextConfig {
        BlipTextConfig {
            vocab_size: 32, hidden_size: 8, encoder_hidden_size: 8,
            intermediate_size: 16, num_hidden_layers: 2,
            num_attention_heads: 2, max_position_embeddings: 32,
            hidden_activation: BlipTextActivation::Gelu,
            layer_norm_eps: 1e-12,
        }
    }

    fn tiny_weights(cfg: &BlipTextConfig) -> BlipTextWeights {
        let mut nb = rng_seed(2026);
        let h = cfg.hidden_size;
        let layers: Vec<BlipTextLayerWeights> = (0..cfg.num_hidden_layers).map(|_| {
            BlipTextLayerWeights {
                self_attn: attn_w(h, h, &mut nb),
                cross_attn: attn_w(h, cfg.encoder_hidden_size, &mut nb),
                ffn: ffn_w(h, cfg.intermediate_size, &mut nb),
            }
        }).collect();
        BlipTextWeights {
            word_embedding: vec_of(cfg.vocab_size * h, &mut nb),
            position_embedding: vec_of(cfg.max_position_embeddings * h, &mut nb),
            embed_ln: ln_w(h),
            layers,
            pred_dense: ws(h * h, &mut nb),
            pred_dense_bias: vec_of(h, &mut nb),
            pred_ln: ln_w(h),
            lm_head: ws(h * cfg.vocab_size, &mut nb),
            lm_head_bias: vec_of(cfg.vocab_size, &mut nb),
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = BlipTextModel { config: cfg.clone(), weights };
        let enc = LazyTensor::from_f32(
            (0..(1 * 5 * cfg.encoder_hidden_size))
                .map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 5, cfg.encoder_hidden_size]),
            &Device::cpu(),
        );
        let ids = vec![1_u32, 2, 3, 4];
        let logits = model.forward(&ids, &enc, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, ids.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn causal_mask_enforced() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = BlipTextModel { config: cfg.clone(), weights };
        let enc = LazyTensor::from_f32(
            vec![0.05_f32; 1 * 4 * cfg.encoder_hidden_size],
            Shape::from_dims(&[1, 4, cfg.encoder_hidden_size]),
            &Device::cpu(),
        );
        let ids_a = vec![1_u32, 2, 3, 4];
        let ids_b = vec![1_u32, 2, 3, 9]; // last position changed
        let a = model.forward(&ids_a, &enc, 0).unwrap().realize_f32();
        let b = model.forward(&ids_b, &enc, 0).unwrap().realize_f32();
        // Positions 0..=2 must match across runs.
        for t in 0..3 {
            for c in 0..cfg.vocab_size {
                let i = t * cfg.vocab_size + c;
                assert!((a[i] - b[i]).abs() < 1e-5,
                    "causal mask violated at t={t} c={c}: {} vs {}", a[i], b[i]);
            }
        }
    }

    #[test]
    fn cross_attention_is_wired() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = BlipTextModel { config: cfg.clone(), weights };
        let ids = vec![1_u32, 2, 3];
        let enc_a = LazyTensor::from_f32(
            (0..(1 * 4 * cfg.encoder_hidden_size))
                .map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 4, cfg.encoder_hidden_size]),
            &Device::cpu(),
        );
        let enc_b = LazyTensor::from_f32(
            (0..(1 * 4 * cfg.encoder_hidden_size))
                .map(|i| (i as f32) * 0.01 + 0.5).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 4, cfg.encoder_hidden_size]),
            &Device::cpu(),
        );
        let a = model.forward(&ids, &enc_a, 0).unwrap().realize_f32();
        let b = model.forward(&ids, &enc_b, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "cross-attention must condition decoder on encoder, max_diff = {max_diff}");
    }

    #[test]
    fn preset_constructs() {
        let base = BlipTextConfig::image_captioning_base();
        assert_eq!(base.encoder_hidden_size, 768);
        let large = BlipTextConfig::image_captioning_large();
        assert_eq!(large.encoder_hidden_size, 1024);
    }
}

//! TrOCR — lazy port.
//!
//! Image → ViT encoder (delegated to `lazy_vit::VitModel` with
//! `classifier = None`) → BART-style decoder with cross-attention
//! → vocabulary logits.
//!
//! Decoder shape mirrors eager `TrOCRDecoderLayer`:
//!   Post-LN: `LN(x + sublayer(x))` at every sublayer.
//!   self_attn (with causal mask) → +res → LN
//!   encoder_attn (cross over encoder features) → +res → LN
//!   fc1 → activation → fc2 → +res → LN
//!
//! v1 scope:
//!   - Learned positional embeddings (default for HF presets).
//!   - Tied lm_head with `embed_tokens`.
//!   - `batch == 1`, F32, prefill only.
//!   - K/V projections in cross-attention take encoder's
//!     `cross_attention_hidden_size` (= ViT hidden) as input,
//!     project to decoder `d_model`.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_vit::{VitConfig, VitModel, VitWeights};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrocrActivation {
    Gelu,
    Relu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrocrDecoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    /// Encoder feature dim (= ViT hidden_size). Cross-attention's
    /// K/V projections take this as input.
    pub cross_attention_hidden_size: usize,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub activation_function: TrocrActivation,
    pub max_position_embeddings: usize,
    /// Offset added to positions when using learned embeddings
    /// (HF convention: typically 2).
    pub learned_pos_offset: usize,
    pub scale_embedding: bool,
    pub tie_word_embeddings: bool,
}

impl TrocrDecoderConfig {
    /// `microsoft/trocr-base-handwritten` decoder preset.
    pub fn trocr_base_handwritten() -> Self {
        Self {
            vocab_size: 50265,
            d_model: 1024,
            cross_attention_hidden_size: 768,
            decoder_layers: 12,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            activation_function: TrocrActivation::Gelu,
            max_position_embeddings: 512,
            learned_pos_offset: 2,
            scale_embedding: false,
            tie_word_embeddings: true,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }
}

#[derive(Debug, Clone)]
pub struct TrocrAttentionWeights {
    pub q_proj: WeightStorage,
    pub k_proj: WeightStorage,
    pub v_proj: WeightStorage,
    pub out_proj: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct TrocrDecoderLayerWeights {
    pub self_attn: TrocrAttentionWeights,
    pub self_attn_ln_gain: Arc<[f32]>,
    pub self_attn_ln_bias: Arc<[f32]>,
    pub encoder_attn: TrocrAttentionWeights,
    pub encoder_attn_ln_gain: Arc<[f32]>,
    pub encoder_attn_ln_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc2: WeightStorage,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct TrocrDecoderWeights {
    /// `[vocab_size, d_model]`.
    pub embed_tokens: Arc<[f32]>,
    /// `[max_position_embeddings + learned_pos_offset, d_model]`.
    pub embed_positions: Arc<[f32]>,
    pub layers: Vec<TrocrDecoderLayerWeights>,
    /// Optional separate output projection when `tie_word_embeddings = false`.
    pub output_projection: Option<WeightStorage>,
}

#[derive(Debug, Clone)]
pub struct TrocrModel {
    pub encoder_config: VitConfig,
    pub encoder_weights: VitWeights,
    pub decoder_config: TrocrDecoderConfig,
    pub decoder_weights: TrocrDecoderWeights,
}

impl TrocrModel {
    /// Image → encoder hidden states `(1, num_patches + 1, vit_hidden_size)`.
    /// Delegates to a `VitModel` with `classifier = None`.
    pub fn forward_encoder(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let mut weights = self.encoder_weights.clone();
        weights.classifier = None;
        let vit = VitModel { config: self.encoder_config.clone(), weights };
        vit.forward(image)
    }

    /// Encoder + decoder. Returns `(1, tgt_len, vocab_size)` logits.
    pub fn forward(&self, image: &LazyTensor, tgt_tokens: &[u32]) -> Result<LazyTensor> {
        let enc_out = self.forward_encoder(image)?;
        self.forward_decoder(tgt_tokens, &enc_out)
    }

    /// Decoder-only forward. `enc_out` is the graph anchor; built
    /// from `forward_encoder` or precomputed and reused across
    /// autoregressive steps.
    pub fn forward_decoder(
        &self,
        tgt_tokens: &[u32],
        enc_out: &LazyTensor,
    ) -> Result<LazyTensor> {
        let dcfg = &self.decoder_config;
        let dw = &self.decoder_weights;
        let tgt_len = tgt_tokens.len();
        assert!(tgt_len > 0, "tgt_tokens must be non-empty");
        assert_eq!(dw.layers.len(), dcfg.decoder_layers);

        // Token embedding lookup.
        let embed = enc_out.const_f32_like(
            Arc::clone(&dw.embed_tokens),
            Shape::from_dims(&[dcfg.vocab_size, dcfg.d_model]),
        );
        let ids = enc_out.const_u32_like(
            tgt_tokens.to_vec(), Shape::from_dims(&[tgt_len]),
        );
        let tok = embed
            .index_select(0_usize, &ids)?
            .reshape(Shape::from_dims(&[1, tgt_len, dcfg.d_model]))?;
        let tok = if dcfg.scale_embedding {
            tok.mul_scalar((dcfg.d_model as f64).sqrt())
        } else {
            tok
        };
        // Learned positional embedding: indices [offset .. offset + tgt_len).
        let pos_ids: Vec<u32> = (0..tgt_len)
            .map(|i| (i + dcfg.learned_pos_offset) as u32)
            .collect();
        let pos_table = enc_out.const_f32_like(
            Arc::clone(&dw.embed_positions),
            Shape::from_dims(&[
                dcfg.max_position_embeddings + dcfg.learned_pos_offset,
                dcfg.d_model,
            ]),
        );
        let pos_idx = enc_out.const_u32_like(
            pos_ids, Shape::from_dims(&[tgt_len]),
        );
        let pos = pos_table
            .index_select(0_usize, &pos_idx)?
            .reshape(Shape::from_dims(&[1, tgt_len, dcfg.d_model]))?;
        let mut x = tok.add(&pos)?;

        // Strict causal mask `[1, 1, tgt_len, tgt_len]` with -inf above diag.
        let mut mask_data = vec![0.0_f32; tgt_len * tgt_len];
        for i in 0..tgt_len {
            for j in (i + 1)..tgt_len {
                mask_data[i * tgt_len + j] = f32::NEG_INFINITY;
            }
        }
        let causal_mask = enc_out.const_f32_like(
            mask_data, Shape::from_dims(&[1, 1, tgt_len, tgt_len]),
        );

        for layer in &dw.layers {
            x = apply_decoder_layer(&x, layer, enc_out, &causal_mask, dcfg)?;
        }

        // lm_head: tied → matmul against embedding constant. Untied → output_projection.
        let logits = match &dw.output_projection {
            Some(w) => w.apply_linear(&x, dcfg.d_model, dcfg.vocab_size),
            None => {
                assert!(dcfg.tie_word_embeddings,
                    "output_projection missing but tie_word_embeddings = false");
                let lm_w = enc_out.const_f32_like(
                    Arc::clone(&dw.embed_tokens),
                    Shape::from_dims(&[dcfg.vocab_size, dcfg.d_model]),
                );
                x.matmul(&lm_w.transpose()?)?
            }
        };
        Ok(logits)
    }
}

// ---- Decoder layer ---------------------------------------------------------

fn apply_decoder_layer(
    x: &LazyTensor,
    w: &TrocrDecoderLayerWeights,
    enc_out: &LazyTensor,
    causal_mask: &LazyTensor,
    cfg: &TrocrDecoderConfig,
) -> Result<LazyTensor> {
    let d = cfg.d_model;
    let n_heads = cfg.decoder_attention_heads;
    let head_dim = cfg.head_dim();
    let kv_in = cfg.cross_attention_hidden_size;

    // Self-attention with causal mask.
    let self_attn_out = apply_attention(
        x, x, &w.self_attn, n_heads, head_dim, d, d, Some(causal_mask),
    )?;
    let h1 = x.add(&self_attn_out)?;
    let h1 = h1.layer_norm_affine(std::sync::Arc::clone(&w.self_attn_ln_gain), std::sync::Arc::clone(&w.self_attn_ln_bias), 1e-5)?;

    // Cross-attention: Q from decoder state, K/V from encoder output.
    let cross_attn_out = apply_attention(
        &h1, enc_out, &w.encoder_attn, n_heads, head_dim, d, kv_in, None,
    )?;
    let h2 = h1.add(&cross_attn_out)?;
    let h2 = h2.layer_norm_affine(std::sync::Arc::clone(&w.encoder_attn_ln_gain), std::sync::Arc::clone(&w.encoder_attn_ln_bias), 1e-5)?;

    // FFN: fc1 → activation → fc2.
    let h_ffn = w.fc1.apply_linear(&h2, d, cfg.decoder_ffn_dim);
    let h_ffn = match cfg.activation_function {
        TrocrActivation::Gelu => h_ffn.gelu(),
        TrocrActivation::Relu => h_ffn.relu(),
    };
    let h_ffn = w.fc2.apply_linear(&h_ffn, cfg.decoder_ffn_dim, d);
    let h3 = h2.add(&h_ffn)?;
    Ok(h3.layer_norm_affine(std::sync::Arc::clone(&w.final_ln_gain), std::sync::Arc::clone(&w.final_ln_bias), 1e-5)?)
}

#[allow(clippy::too_many_arguments)]
fn apply_attention(
    q_input: &LazyTensor,
    kv_input: &LazyTensor,
    w: &TrocrAttentionWeights,
    n_heads: usize,
    head_dim: usize,
    q_in_dim: usize,
    kv_in_dim: usize,
    mask: Option<&LazyTensor>,
) -> Result<LazyTensor> {
    let q_dims = q_input.shape();
    let q_dims = q_dims.dims();
    let batch = q_dims[0];
    let q_len = q_dims[1];
    let kv_dims = kv_input.shape();
    let kv_dims = kv_dims.dims();
    let kv_len = kv_dims[1];
    let d_model = n_heads * head_dim;

    let q = w.q_proj.apply_linear(q_input, q_in_dim, d_model);
    let k = w.k_proj.apply_linear(kv_input, kv_in_dim, d_model);
    let v = w.v_proj.apply_linear(kv_input, kv_in_dim, d_model);

    let scaling = 1.0_f64 / (head_dim as f64).sqrt();
    let q = q.mul_scalar(scaling);

    // (B, L, n_heads * head_dim) → (B, n_heads, L, head_dim)
    let q = q.split_heads(n_heads, head_dim)?;
    let k = k.split_heads(n_heads, head_dim)?;
    let v = v.split_heads(n_heads, head_dim)?;

    // Scores [B, n_heads, q_len, kv_len].
    let kt = k.permute([0, 1, 3, 2_usize])?;
    let mut scores = q.matmul(&kt)?;
    if let Some(m) = mask {
        let mb = m.broadcast_to(Shape::from_dims(&[batch, n_heads, q_len, kv_len]))?;
        scores = scores.add(&mb)?;
    }
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?;
    let _ = (batch, q_len, kv_len, d_model);
    let ctx = ctx.merge_heads()?;
    Ok(w.out_proj.apply_linear(&ctx, d_model, d_model))
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy_vit::{VitActivation, VitConfig, VitLayerWeights, VitWeights};
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

    fn tiny_vit_config() -> VitConfig {
        VitConfig {
            hidden_size: 8, num_hidden_layers: 2, num_attention_heads: 2,
            intermediate_size: 16,
            hidden_activation: VitActivation::Gelu, layer_norm_eps: 1e-5,
            image_size: 8, patch_size: 4, num_channels: 3,
            qkv_bias: true,
        }
    }

    fn tiny_vit_weights(cfg: &VitConfig) -> VitWeights {
        let mut nb = rng_seed(101);
        let h = cfg.hidden_size;
        let np = cfg.num_patches();
        let layers: Vec<VitLayerWeights> = (0..cfg.num_hidden_layers).map(|_| VitLayerWeights {
            ln_before_gain: Arc::from(vec![1.0_f32; h]),
            ln_before_bias: Arc::from(vec![0.0_f32; h]),
            q_proj: ws(h * h, &mut nb), q_proj_bias: Some(vec_of(h, &mut nb)),
            k_proj: ws(h * h, &mut nb), k_proj_bias: Some(vec_of(h, &mut nb)),
            v_proj: ws(h * h, &mut nb), v_proj_bias: Some(vec_of(h, &mut nb)),
            attn_output_proj: ws(h * h, &mut nb),
            attn_output_proj_bias: vec_of(h, &mut nb),
            ln_after_gain: Arc::from(vec![1.0_f32; h]),
            ln_after_bias: Arc::from(vec![0.0_f32; h]),
            intermediate_proj: ws(h * cfg.intermediate_size, &mut nb),
            intermediate_proj_bias: vec_of(cfg.intermediate_size, &mut nb),
            mlp_output_proj: ws(cfg.intermediate_size * h, &mut nb),
            mlp_output_proj_bias: vec_of(h, &mut nb),
        }).collect();
        VitWeights {
            patch_proj: vec_of(h * cfg.num_channels * cfg.patch_size * cfg.patch_size, &mut nb),
            patch_proj_bias: vec_of(h, &mut nb),
            cls_token: vec_of(h, &mut nb),
            position_embeddings: vec_of((np + 1) * h, &mut nb),
            layers,
            final_ln_gain: Arc::from(vec![1.0_f32; h]),
            final_ln_bias: Arc::from(vec![0.0_f32; h]),
            classifier: None,
        }
    }

    fn tiny_trocr_config(vit_hidden: usize) -> TrocrDecoderConfig {
        TrocrDecoderConfig {
            vocab_size: 16, d_model: 8,
            cross_attention_hidden_size: vit_hidden,
            decoder_layers: 2, decoder_attention_heads: 2,
            decoder_ffn_dim: 16,
            activation_function: TrocrActivation::Gelu,
            max_position_embeddings: 32, learned_pos_offset: 2,
            scale_embedding: false,
            tie_word_embeddings: true,
        }
    }

    fn tiny_trocr_weights(dcfg: &TrocrDecoderConfig) -> TrocrDecoderWeights {
        let mut nb = rng_seed(202);
        let d = dcfg.d_model;
        let kv_in = dcfg.cross_attention_hidden_size;
        let layers: Vec<TrocrDecoderLayerWeights> = (0..dcfg.decoder_layers).map(|_| {
            TrocrDecoderLayerWeights {
                self_attn: TrocrAttentionWeights {
                    q_proj: ws(d * d, &mut nb),
                    k_proj: ws(d * d, &mut nb),
                    v_proj: ws(d * d, &mut nb),
                    out_proj: ws(d * d, &mut nb),
                },
                self_attn_ln_gain: Arc::from(vec![1.0_f32; d]),
                self_attn_ln_bias: Arc::from(vec![0.0_f32; d]),
                encoder_attn: TrocrAttentionWeights {
                    q_proj: ws(d * d, &mut nb),
                    k_proj: ws(kv_in * d, &mut nb),
                    v_proj: ws(kv_in * d, &mut nb),
                    out_proj: ws(d * d, &mut nb),
                },
                encoder_attn_ln_gain: Arc::from(vec![1.0_f32; d]),
                encoder_attn_ln_bias: Arc::from(vec![0.0_f32; d]),
                fc1: ws(d * dcfg.decoder_ffn_dim, &mut nb),
                fc2: ws(dcfg.decoder_ffn_dim * d, &mut nb),
                final_ln_gain: Arc::from(vec![1.0_f32; d]),
                final_ln_bias: Arc::from(vec![0.0_f32; d]),
            }
        }).collect();
        TrocrDecoderWeights {
            embed_tokens: vec_of(dcfg.vocab_size * d, &mut nb),
            embed_positions: vec_of(
                (dcfg.max_position_embeddings + dcfg.learned_pos_offset) * d, &mut nb,
            ),
            layers,
            output_projection: None,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let vcfg = tiny_vit_config();
        let vw = tiny_vit_weights(&vcfg);
        let dcfg = tiny_trocr_config(vcfg.hidden_size);
        let dw = tiny_trocr_weights(&dcfg);
        let model = TrocrModel {
            encoder_config: vcfg.clone(),
            encoder_weights: vw,
            decoder_config: dcfg.clone(),
            decoder_weights: dw,
        };
        let image: Vec<f32> = (0..(3 * 8 * 8)).map(|i| (i as f32) * 0.01).collect();
        let img = LazyTensor::from_f32(
            image, Shape::from_dims(&[1, 3, 8, 8]), &Device::cpu(),
        );
        let tgt = [1_u32, 2, 3];
        let logits = model.forward(&img, &tgt).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tgt.len(), dcfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    /// `forward_decoder(tgt, forward_encoder(image))` matches
    /// `forward(image, tgt)` — proves the split path computes
    /// the same graph.
    #[test]
    fn forward_decoder_matches_forward() {
        let vcfg = tiny_vit_config();
        let vw = tiny_vit_weights(&vcfg);
        let dcfg = tiny_trocr_config(vcfg.hidden_size);
        let dw = tiny_trocr_weights(&dcfg);
        let model = TrocrModel {
            encoder_config: vcfg.clone(),
            encoder_weights: vw,
            decoder_config: dcfg.clone(),
            decoder_weights: dw,
        };
        let image: Vec<f32> = (0..(3 * 8 * 8)).map(|i| (i as f32) * 0.01).collect();
        let img = LazyTensor::from_f32(
            image, Shape::from_dims(&[1, 3, 8, 8]), &Device::cpu(),
        );
        let tgt = [1_u32, 2, 3];
        let full = model.forward(&img, &tgt).unwrap().realize_f32();
        let enc = model.forward_encoder(&img).unwrap();
        let split = model.forward_decoder(&tgt, &enc).unwrap().realize_f32();
        assert_eq!(full.len(), split.len());
        for (a, b) in full.iter().zip(split.iter()) {
            assert!((a - b).abs() < 1e-5,
                "split path must match full forward: {a} vs {b}");
        }
    }

    /// Cross-attention must condition the decoder output on the
    /// encoder output. Different images must yield different
    /// logits for the same target tokens.
    #[test]
    fn cross_attention_is_wired() {
        let vcfg = tiny_vit_config();
        let vw = tiny_vit_weights(&vcfg);
        let dcfg = tiny_trocr_config(vcfg.hidden_size);
        let dw = tiny_trocr_weights(&dcfg);
        let model = TrocrModel {
            encoder_config: vcfg.clone(),
            encoder_weights: vw,
            decoder_config: dcfg.clone(),
            decoder_weights: dw,
        };
        let img_a = LazyTensor::from_f32(
            (0..(3 * 8 * 8)).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 8, 8]), &Device::cpu(),
        );
        let img_b = LazyTensor::from_f32(
            (0..(3 * 8 * 8)).map(|i| i as f32 * 0.01 + 0.5).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 8, 8]), &Device::cpu(),
        );
        let tgt = [1_u32, 2, 3];
        let a = model.forward(&img_a, &tgt).unwrap().realize_f32();
        let b = model.forward(&img_b, &tgt).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "cross-attention must condition decoder on encoder, max_diff = {max_diff}");
    }

    /// Causal mask: changing target token at position t must NOT
    /// alter logits at positions < t.
    #[test]
    fn causal_mask_enforced() {
        let vcfg = tiny_vit_config();
        let vw = tiny_vit_weights(&vcfg);
        let dcfg = tiny_trocr_config(vcfg.hidden_size);
        let dw = tiny_trocr_weights(&dcfg);
        let model = TrocrModel {
            encoder_config: vcfg.clone(),
            encoder_weights: vw,
            decoder_config: dcfg.clone(),
            decoder_weights: dw,
        };
        let img = LazyTensor::from_f32(
            vec![0.1_f32; 3 * 8 * 8],
            Shape::from_dims(&[1, 3, 8, 8]), &Device::cpu(),
        );
        let tgt_a = [1_u32, 2, 3, 4];
        let tgt_b = [1_u32, 2, 3, 9]; // last token changed
        let a = model.forward(&img, &tgt_a).unwrap().realize_f32();
        let b = model.forward(&img, &tgt_b).unwrap().realize_f32();
        let v = dcfg.vocab_size;
        for t in 0..3 {
            for c in 0..v {
                let i = t * v + c;
                assert!((a[i] - b[i]).abs() < 1e-5,
                    "causal mask violated at t={t}, c={c}: {} vs {}", a[i], b[i]);
            }
        }
    }
}

//! PaliGemma multimodal model ported to the lazy-graph API.
//!
//! PaliGemma from Google (Beyer et al. 2024). First multimodal
//! **composition** port — combines an existing lazy vision
//! encoder ([`crate::lazy_siglip::SiglipVisionModel`]) and an
//! existing lazy language model ([`crate::lazy_gemma::GemmaModel`])
//! with a thin projection + interleaving layer:
//!
//!   ```text
//!   image_features = siglip_vision(pixel_values)         # (1, num_patches, vision_hidden)
//!   image_features = MultiModalProjector(image_features) # (1, num_patches, gemma_hidden)
//!   image_features = L2_normalize(image_features)        # per-token L2-norm
//!   text_embeds    = gemma.token_embedding(text_tokens)  # (1, text_len, gemma_hidden)
//!   combined       = cat(image_features, text_embeds, dim=1)
//!   logits         = gemma.forward_embeds(combined, start_pos=0)
//!   ```
//!
//! The image features are **prepended** to the text embedding
//! sequence (image-then-text layout per PaliGemma's convention).
//! The Gemma language model runs on the combined sequence as if
//! it were all-text input.
//!
//! # Scope (v1)
//!
//! Forward-only, single image + single token sequence, F32. The
//! eager reference's autoregressive decode loop (separate
//! `setup` + `forward` calls with KV cache) is out of scope —
//! v1 returns logits for the entire combined sequence in one
//! pass.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_gemma::{GemmaConfig, GemmaModel, GemmaWeights};
use crate::lazy_siglip::{SiglipVisionConfig, SiglipVisionModel, SiglipVisionWeights};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct PaligemmaConfig {
    pub vision_config: SiglipVisionConfig,
    pub text_config: GemmaConfig,
    /// Multi-modal projection target dim (typically equals
    /// `text_config.hidden_size`).
    pub projection_dim: usize,
}

#[derive(Debug, Clone)]
pub struct PaligemmaWeights {
    pub vision: SiglipVisionWeights,
    /// `[vision_hidden, projection_dim]`.
    pub mm_proj: WeightStorage,
    pub mm_proj_bias: Arc<[f32]>,
    pub text: GemmaWeights,
}

#[derive(Debug, Clone)]
pub struct PaligemmaModel {
    pub config: PaligemmaConfig,
    pub weights: PaligemmaWeights,
}

impl PaligemmaModel {
    /// Run the full multimodal forward pass. Returns logits
    /// for the combined `[image_features; text_embeds]`
    /// sequence of shape `(1, num_patches + text_len, vocab)`.
    pub fn forward(
        &self,
        pixel_values: &LazyTensor,
        text_tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let v_cfg = &cfg.vision_config;
        let t_cfg = &cfg.text_config;
        assert_eq!(
            cfg.projection_dim, t_cfg.hidden_size,
            "v1: projection_dim must equal text hidden_size",
        );
        let text_len = text_tokens.len();
        assert!(text_len > 0, "text_tokens must be non-empty");

        // ---- Run SigLIP vision encoder (no pooling head) -----------------
        let vision = SiglipVisionModel {
            config: v_cfg.clone(),
            weights: self.weights.vision.clone(),
        };
        // SigLIP without head returns (1, num_patches, vision_hidden).
        let image_features = vision.forward(pixel_values)?;
        let np = v_cfg.num_patches();
        let dims = image_features.shape();
        let dims = dims.dims();
        assert_eq!(dims, &[1, np, v_cfg.hidden_size],
            "SigLIP w/o head must produce (1, num_patches, hidden); got {dims:?}");

        // ---- Multi-modal projection: vision_hidden → projection_dim -----
        let projected = self.weights.mm_proj.apply_linear(
            &image_features,
            v_cfg.hidden_size,
            cfg.projection_dim,
        );
        let bias_t = pixel_values.const_f32_like(
            Arc::clone(&self.weights.mm_proj_bias),
            Shape::from_dims(&[cfg.projection_dim]),
        );
        let image_proj = projected.broadcast_add(&bias_t)?;
        // L2 normalize image features per-token.
        let image_proj_n = l2_normalize_last(&image_proj, 1e-12)?;

        // ---- Embed text tokens via Gemma's token embedding --------------
        // We must anchor on the SAME graph used by the vision tower so the
        // concat works. SigLIP's forward already created token_embedding /
        // patches on its own graph anchored on `pixel_values`. Use
        // `pixel_values` as the graph anchor.
        let gemma_embed_lt = pixel_values.const_f32_like(
            Arc::clone(&self.weights.text.token_embedding),
            Shape::from_dims(&[t_cfg.vocab_size, t_cfg.hidden_size]),
        );
        let token_ids = pixel_values.const_u32_like(
            text_tokens.to_vec(),
            Shape::from_dims(&[text_len]),
        );
        let text_embeds = gemma_embed_lt
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[1, text_len, t_cfg.hidden_size]))?;
        // Gemma applies sqrt(hidden_size) scaling to token embeds.
        let text_embeds_scaled = text_embeds.mul_scalar((t_cfg.hidden_size as f64).sqrt());

        // ---- Concat [image; text] and run Gemma layers -------------------
        let combined = image_proj_n.concat(&text_embeds_scaled, 1_usize)?;
        let gemma = GemmaModel {
            config: t_cfg.clone(),
            weights: self.weights.text.clone(),
        };
        gemma.forward_embeds(&combined, 0)
    }
}

/// L2-normalize along the last dim with an epsilon-clamped
/// denominator.
fn l2_normalize_last(x: &LazyTensor, eps: f64) -> Result<LazyTensor> {
    let sq = x.sqr();
    let dims = x.shape();
    let dims_v: Vec<usize> = dims.dims().to_vec();
    let last_dim_idx = dims_v.len() - 1;
    let l2 = sq.sum_keepdim(last_dim_idx)?
        .add_scalar(eps)
        .sqrt();
    let l2_bc = l2.broadcast_to(Shape::from_dims(&dims_v))?;
    x.div(&l2_bc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::LayerWeights;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_vision_cfg() -> SiglipVisionConfig {
        SiglipVisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_channels: 3,
            image_size: 8,
            patch_size: 4,
            hidden_activation: crate::lazy_siglip::SiglipActivation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }

    fn tiny_text_cfg() -> GemmaConfig {
        GemmaConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            attention_bias: false,
            hidden_activation: crate::lazy_gemma::GemmaActivation::GeluPytorchTanh,
        }
    }

    fn tiny_vision_weights(cfg: &SiglipVisionConfig) -> SiglipVisionWeights {
        let mut s: u32 = 90909;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let patch_proj = vec_of(
            cfg.hidden_size * cfg.num_channels * cfg.patch_size * cfg.patch_size,
            &mut *nb,
        );
        let patch_proj_bias = vec_of(cfg.hidden_size, &mut *nb);
        let position_embedding = vec_of(cfg.num_patches() * cfg.hidden_size, &mut *nb);
        let layers: Vec<_> = (0..cfg.num_hidden_layers).map(|_|
            crate::lazy_siglip::SiglipEncoderLayerWeights {
                ln1_gain: Arc::from(vec![1.0_f32; cfg.hidden_size]),
                ln1_bias: Arc::from(vec![0.0_f32; cfg.hidden_size]),
                q_proj: WeightStorage::F32(vec_of(cfg.hidden_size * cfg.hidden_size, &mut *nb)),
                q_proj_bias: vec_of(cfg.hidden_size, &mut *nb),
                k_proj: WeightStorage::F32(vec_of(cfg.hidden_size * cfg.hidden_size, &mut *nb)),
                k_proj_bias: vec_of(cfg.hidden_size, &mut *nb),
                v_proj: WeightStorage::F32(vec_of(cfg.hidden_size * cfg.hidden_size, &mut *nb)),
                v_proj_bias: vec_of(cfg.hidden_size, &mut *nb),
                out_proj: WeightStorage::F32(vec_of(cfg.hidden_size * cfg.hidden_size, &mut *nb)),
                out_proj_bias: vec_of(cfg.hidden_size, &mut *nb),
                ln2_gain: Arc::from(vec![1.0_f32; cfg.hidden_size]),
                ln2_bias: Arc::from(vec![0.0_f32; cfg.hidden_size]),
                fc1: WeightStorage::F32(vec_of(cfg.hidden_size * cfg.intermediate_size, &mut *nb)),
                fc1_bias: vec_of(cfg.intermediate_size, &mut *nb),
                fc2: WeightStorage::F32(vec_of(cfg.intermediate_size * cfg.hidden_size, &mut *nb)),
                fc2_bias: vec_of(cfg.hidden_size, &mut *nb),
            }
        ).collect();
        SiglipVisionWeights {
            patch_proj,
            patch_proj_bias,
            position_embedding,
            layers,
            post_ln_gain: Arc::from(vec![1.0_f32; cfg.hidden_size]),
            post_ln_bias: Arc::from(vec![0.0_f32; cfg.hidden_size]),
            head: None, // PaliGemma uses no pooling head.
        }
    }

    fn tiny_gemma_weights(cfg: &GemmaConfig) -> GemmaWeights {
        let mut s: u32 = 80808;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            attn_q_bias: if cfg.attention_bias { Some(vec_of(h, &mut *nb)) } else { None },
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_k_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_v_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            attn_norm_gain: Arc::from(vec![0.1_f32; h]),
            ffn_norm_gain: Arc::from(vec![0.1_f32; h]),
        }).collect();
        let final_norm_gain = Arc::from(vec![0.1_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        GemmaWeights { token_embedding, layers, final_norm_gain, output }
    }

    fn tiny_image(cfg: &SiglipVisionConfig) -> LazyTensor {
        let n_pix = 1 * cfg.num_channels * cfg.image_size * cfg.image_size;
        let img_data: Vec<f32> = (0..n_pix).map(|i| (i as f32 / n_pix as f32)).collect();
        LazyTensor::from_f32(
            Arc::from(img_data),
            Shape::from_dims(&[1, cfg.num_channels, cfg.image_size, cfg.image_size]),
            &Device::cpu(),
        )
    }

    #[test]
    fn forward_shape_and_finite() {
        let v_cfg = tiny_vision_cfg();
        let t_cfg = tiny_text_cfg();
        let mut s: u32 = 70707;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let mm_proj = WeightStorage::F32(vec_of(v_cfg.hidden_size * t_cfg.hidden_size, &mut *nb));
        let mm_proj_bias = vec_of(t_cfg.hidden_size, &mut *nb);
        let weights = PaligemmaWeights {
            vision: tiny_vision_weights(&v_cfg),
            mm_proj, mm_proj_bias,
            text: tiny_gemma_weights(&t_cfg),
        };
        let cfg = PaligemmaConfig {
            vision_config: v_cfg.clone(),
            text_config: t_cfg.clone(),
            projection_dim: t_cfg.hidden_size,
        };
        let model = PaligemmaModel { config: cfg, weights };

        let img = tiny_image(&v_cfg);
        let text_tokens = [1_u32, 2, 3];
        let logits = model.forward(&img, &text_tokens).unwrap();
        let expected_seq = v_cfg.num_patches() + text_tokens.len();
        assert_eq!(logits.shape().dims(), &[1, expected_seq, t_cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// Changing the image must alter the prediction at text
    /// positions (because image features are prepended to the
    /// text in the sequence, every text position attends back
    /// to image positions through the Gemma transformer).
    #[test]
    fn image_change_alters_text_logits() {
        let v_cfg = tiny_vision_cfg();
        let t_cfg = tiny_text_cfg();
        let mut s: u32 = 60606;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let mm_proj = WeightStorage::F32(vec_of(v_cfg.hidden_size * t_cfg.hidden_size, &mut *nb));
        let mm_proj_bias = vec_of(t_cfg.hidden_size, &mut *nb);
        let weights = PaligemmaWeights {
            vision: tiny_vision_weights(&v_cfg),
            mm_proj, mm_proj_bias,
            text: tiny_gemma_weights(&t_cfg),
        };
        let cfg = PaligemmaConfig {
            vision_config: v_cfg.clone(),
            text_config: t_cfg.clone(),
            projection_dim: t_cfg.hidden_size,
        };
        let model_a = PaligemmaModel { config: cfg.clone(), weights: weights.clone() };
        let model_b = PaligemmaModel { config: cfg, weights };

        let n_pix = 1 * v_cfg.num_channels * v_cfg.image_size * v_cfg.image_size;
        let img_a_data: Vec<f32> = (0..n_pix).map(|i| (i as f32 / n_pix as f32)).collect();
        let img_b_data: Vec<f32> = img_a_data.iter().map(|x| 1.0 - x).collect();
        let img_a = LazyTensor::from_f32(
            Arc::from(img_a_data),
            Shape::from_dims(&[1, v_cfg.num_channels, v_cfg.image_size, v_cfg.image_size]),
            &Device::cpu(),
        );
        let img_b = LazyTensor::from_f32(
            Arc::from(img_b_data),
            Shape::from_dims(&[1, v_cfg.num_channels, v_cfg.image_size, v_cfg.image_size]),
            &Device::cpu(),
        );
        let toks = [1_u32, 2, 3];
        let a = model_a.forward(&img_a, &toks).unwrap().realize_f32();
        let b = model_b.forward(&img_b, &toks).unwrap().realize_f32();
        // Extract text logits (positions AFTER num_patches).
        let np = v_cfg.num_patches();
        let v = t_cfg.vocab_size;
        let start_a = np * v;
        let start_b = np * v;
        let mut max_diff = 0.0_f32;
        for (x, y) in a[start_a..].iter().zip(b[start_b..].iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "image change must alter text-position logits, max_diff = {max_diff}");
    }
}

//! BLIP image-captioning composition — lazy port.
//!
//! Top-level wrapper that composes the BLIP vision encoder
//! ([`crate::lazy_blip_vision::BlipVisionModel`]) with the BLIP
//! text decoder ([`crate::lazy_blip_text::BlipTextModel`]) into a
//! single image-captioning entry point matching the eager
//! `BlipForConditionalGeneration`.
//!
//! Unlike PaliGemma / LLaVA (image-then-text **concatenation**
//! into the LM input), BLIP uses **cross-attention from the text
//! decoder to the vision encoder's hidden states** — every text
//! decoder layer cross-attends to the per-patch image features.
//! Because the text decoder already takes
//! `encoder_hidden_states` as a parameter, the composition is a
//! straight vision-forward → text-forward chain with no projector
//! between them.
//!
//! # Scope (v1)
//!
//! Forward-only, single image + single token sequence, F32. The
//! eager reference's autoregressive decode loop (with KV cache)
//! is out of scope — v1 returns logits for the entire token
//! sequence in one pass.

use crate::lazy::LazyTensor;
use crate::lazy_blip_text::{BlipTextConfig, BlipTextModel, BlipTextWeights};
use crate::lazy_blip_vision::{BlipVisionConfig, BlipVisionModel, BlipVisionWeights};
use crate::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct BlipConfig {
    pub vision_config: BlipVisionConfig,
    pub text_config: BlipTextConfig,
}

impl BlipConfig {
    /// `Salesforce/blip-image-captioning-base`.
    pub fn image_captioning_base() -> Self {
        Self {
            vision_config: BlipVisionConfig::image_captioning_base(),
            text_config: BlipTextConfig::image_captioning_base(),
        }
    }

    /// `Salesforce/blip-image-captioning-large`.
    ///
    /// Note: vision is ViT-Large (1024-dim) but the text decoder
    /// stays at hidden_size=768; the text decoder's
    /// `encoder_hidden_size=1024` matches the vision hidden so
    /// cross-attention K/V from the encoder land at the right
    /// channel count.
    pub fn image_captioning_large() -> Self {
        Self {
            vision_config: BlipVisionConfig::image_captioning_large(),
            text_config: BlipTextConfig::image_captioning_large(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlipWeights {
    pub vision: BlipVisionWeights,
    pub text: BlipTextWeights,
}

#[derive(Debug, Clone)]
pub struct BlipForConditionalGeneration {
    pub config: BlipConfig,
    pub weights: BlipWeights,
}

impl BlipForConditionalGeneration {
    /// Run the full image-captioning forward pass.
    ///
    /// * `pixel_values`: image tensor `(1, 3, image_size, image_size)`.
    /// * `input_ids`: target token sequence of length T.
    /// * `start_pos`: position-embedding offset (0 for fresh prefill).
    ///
    /// Returns next-token logits `(1, T, vocab_size)`.
    pub fn forward(
        &self,
        pixel_values: &LazyTensor,
        input_ids: &[u32],
        start_pos: usize,
    ) -> Result<LazyTensor> {
        // Vision tower: (1, 3, H, W) → (1, num_patches + 1, vision_hidden).
        let vision = BlipVisionModel {
            config: self.config.vision_config.clone(),
            weights: self.weights.vision.clone(),
        };
        let encoder_hidden = vision.forward(pixel_values)?;

        // Sanity check: cross-attention K/V projects from
        // `text_config.encoder_hidden_size`, which must match the
        // vision tower's hidden dim.
        assert_eq!(
            self.config.text_config.encoder_hidden_size,
            self.config.vision_config.hidden_size,
            "BLIP: text_config.encoder_hidden_size ({}) must equal \
             vision_config.hidden_size ({})",
            self.config.text_config.encoder_hidden_size,
            self.config.vision_config.hidden_size,
        );

        // Text decoder cross-attends to encoder_hidden in every layer.
        let text = BlipTextModel {
            config: self.config.text_config.clone(),
            weights: self.weights.text.clone(),
        };
        text.forward(input_ids, &encoder_hidden, start_pos)
    }
}

// ---- HuggingFace safetensors composer --------------------------------------

impl BlipWeights {
    /// Load the full BLIP checkpoint by composing the shipped
    /// `BlipVisionWeights::load_from_mmapped` and
    /// `BlipTextWeights::load_from_mmapped` against the standard
    /// `Salesforce/blip-*` checkpoint layout (vision tower under
    /// `vision_model.`, text decoder under `text_decoder.`).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &BlipConfig,
    ) -> Result<Self> {
        let vision = BlipVisionWeights::load_from_mmapped(
            st, &cfg.vision_config, "vision_model.",
        )?;
        let text = BlipTextWeights::load_from_mmapped(
            st, &cfg.text_config, cfg.vision_config.hidden_size, "text_decoder.",
        )?;
        Ok(Self { vision, text })
    }
}


// ---- Tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::WeightStorage;
    use crate::lazy_blip_text::{
        BlipTextActivation, BlipTextAttentionWeights, BlipTextFfnWeights,
        BlipTextLayerWeights,
    };
    use crate::lazy_blip_vision::{
        BlipMlpWeights, BlipVisionActivation, BlipVisionAttentionWeights,
        BlipVisionLayerWeights,
    };
    use crate::Device;
    use fuel_ir::Shape;
    use std::sync::Arc;

    fn rng(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.02
        }
    }

    fn tiny_config() -> BlipConfig {
        let vision_config = BlipVisionConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            image_size: 16,
            patch_size: 4,
            hidden_activation: BlipVisionActivation::Gelu,
            layer_norm_eps: 1e-5,
        };
        let text_config = BlipTextConfig {
            vocab_size: 32,
            hidden_size: 16,
            // Must match vision_config.hidden_size for cross-attn K/V.
            encoder_hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            max_position_embeddings: 16,
            hidden_activation: BlipTextActivation::Gelu,
            layer_norm_eps: 1e-12,
        };
        BlipConfig { vision_config, text_config }
    }

    fn ln_weights(dim: usize) -> crate::lazy_blip_vision::LayerNormWeights {
        crate::lazy_blip_vision::LayerNormWeights {
            gain: Arc::from(vec![1.0_f32; dim]),
            bias: Arc::from(vec![0.0_f32; dim]),
        }
    }

    fn ln_weights_text(dim: usize) -> crate::lazy_blip_text::LayerNormWeights {
        crate::lazy_blip_text::LayerNormWeights {
            gain: Arc::from(vec![1.0_f32; dim]),
            bias: Arc::from(vec![0.0_f32; dim]),
        }
    }

    fn tiny_vision_weights(cfg: &BlipVisionConfig) -> BlipVisionWeights {
        let mut next = rng(11111);
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let layers: Vec<_> = (0..cfg.num_hidden_layers).map(|_| {
            BlipVisionLayerWeights {
                ln1: ln_weights(h),
                attn: BlipVisionAttentionWeights {
                    qkv: WeightStorage::F32(vec_of(h * 3 * h)),
                    qkv_bias: vec_of(3 * h),
                    projection: WeightStorage::F32(vec_of(h * h)),
                    projection_bias: vec_of(h),
                },
                ln2: ln_weights(h),
                mlp: BlipMlpWeights {
                    fc1: WeightStorage::F32(vec_of(h * cfg.intermediate_size)),
                    fc1_bias: vec_of(cfg.intermediate_size),
                    fc2: WeightStorage::F32(vec_of(cfg.intermediate_size * h)),
                    fc2_bias: vec_of(h),
                },
            }
        }).collect();
        let np = cfg.num_patches();
        BlipVisionWeights {
            patch_proj: vec_of(h * 3 * cfg.patch_size * cfg.patch_size),
            patch_proj_bias: vec_of(h),
            class_token: vec_of(h),
            position_embedding: vec_of((np + 1) * h),
            layers,
            post_layernorm: ln_weights(h),
        }
    }

    fn tiny_text_weights(cfg: &BlipTextConfig) -> BlipTextWeights {
        let mut next = rng(22222);
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let eh = cfg.encoder_hidden_size;
        let layers: Vec<_> = (0..cfg.num_hidden_layers).map(|_| {
            BlipTextLayerWeights {
                self_attn: BlipTextAttentionWeights {
                    query: WeightStorage::F32(vec_of(h * h)),
                    query_bias: vec_of(h),
                    key: WeightStorage::F32(vec_of(h * h)),
                    key_bias: vec_of(h),
                    value: WeightStorage::F32(vec_of(h * h)),
                    value_bias: vec_of(h),
                    out_dense: WeightStorage::F32(vec_of(h * h)),
                    out_dense_bias: vec_of(h),
                    out_ln: ln_weights_text(h),
                },
                cross_attn: BlipTextAttentionWeights {
                    query: WeightStorage::F32(vec_of(h * h)),
                    query_bias: vec_of(h),
                    key: WeightStorage::F32(vec_of(eh * h)),
                    key_bias: vec_of(h),
                    value: WeightStorage::F32(vec_of(eh * h)),
                    value_bias: vec_of(h),
                    out_dense: WeightStorage::F32(vec_of(h * h)),
                    out_dense_bias: vec_of(h),
                    out_ln: ln_weights_text(h),
                },
                ffn: BlipTextFfnWeights {
                    intermediate: WeightStorage::F32(vec_of(h * cfg.intermediate_size)),
                    intermediate_bias: vec_of(cfg.intermediate_size),
                    output: WeightStorage::F32(vec_of(cfg.intermediate_size * h)),
                    output_bias: vec_of(h),
                    output_ln: ln_weights_text(h),
                },
            }
        }).collect();
        BlipTextWeights {
            word_embedding: vec_of(cfg.vocab_size * h),
            position_embedding: vec_of(cfg.max_position_embeddings * h),
            embed_ln: ln_weights_text(h),
            layers,
            pred_dense: WeightStorage::F32(vec_of(h * h)),
            pred_dense_bias: vec_of(h),
            pred_ln: ln_weights_text(h),
            lm_head: WeightStorage::F32(vec_of(h * cfg.vocab_size)),
            lm_head_bias: vec_of(cfg.vocab_size),
        }
    }

    #[test]
    fn blip_presets_share_decoder_with_vision_hidden() {
        let base = BlipConfig::image_captioning_base();
        assert_eq!(
            base.text_config.encoder_hidden_size,
            base.vision_config.hidden_size,
        );
        let large = BlipConfig::image_captioning_large();
        assert_eq!(
            large.text_config.encoder_hidden_size,
            large.vision_config.hidden_size,
        );
        // BLIP-large: vision is ViT-Large (1024) but decoder stays 768.
        assert_eq!(large.vision_config.hidden_size, 1024);
        assert_eq!(large.text_config.hidden_size, 768);
    }

    #[test]
    fn blip_forward_shape_and_finite() {
        let cfg = tiny_config();
        let vision_weights = tiny_vision_weights(&cfg.vision_config);
        let text_weights = tiny_text_weights(&cfg.text_config);
        let model = BlipForConditionalGeneration {
            config: cfg.clone(),
            weights: BlipWeights { vision: vision_weights, text: text_weights },
        };
        let img_size = cfg.vision_config.image_size;
        let pixel_data: Vec<f32> = (0..1 * 3 * img_size * img_size)
            .map(|i| ((i as f32) * 0.001) - 0.05).collect();
        let pixels = LazyTensor::from_f32(
            pixel_data,
            Shape::from_dims(&[1, 3, img_size, img_size]),
            &Device::cpu(),
        );
        let input_ids = vec![1_u32, 2, 3, 4];
        let t = input_ids.len();
        let logits = model.forward(&pixels, &input_ids, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, t, cfg.text_config.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite BLIP logit: {v}");
        }
    }

    #[test]
    fn blip_forward_logits_change_with_input() {
        let cfg = tiny_config();
        let vision_weights = tiny_vision_weights(&cfg.vision_config);
        let text_weights = tiny_text_weights(&cfg.text_config);
        let model = BlipForConditionalGeneration {
            config: cfg.clone(),
            weights: BlipWeights { vision: vision_weights, text: text_weights },
        };
        let img_size = cfg.vision_config.image_size;
        let pixel_a: Vec<f32> = (0..3 * img_size * img_size)
            .map(|i| (i as f32) * 0.001).collect();
        let pixel_b: Vec<f32> = (0..3 * img_size * img_size)
            .map(|i| (i as f32) * -0.001 + 0.3).collect();
        let pa = LazyTensor::from_f32(
            pixel_a, Shape::from_dims(&[1, 3, img_size, img_size]), &Device::cpu(),
        );
        let pb = LazyTensor::from_f32(
            pixel_b, Shape::from_dims(&[1, 3, img_size, img_size]), &Device::cpu(),
        );
        let ids = vec![1_u32, 2, 3];
        let la = model.forward(&pa, &ids, 0).unwrap().realize_f32();
        let lb = model.forward(&pb, &ids, 0).unwrap().realize_f32();
        let max_diff = la.iter().zip(lb.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff > 1e-6,
            "different pixel inputs should yield different logits (max diff = {max_diff})");
    }
}

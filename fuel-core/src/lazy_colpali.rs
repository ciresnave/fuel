//! ColPali — lazy port.
//!
//! ColPali is a retrieval model that wraps a PaliGemma backbone
//! with a 128-dim projection head + L2 normalization. The same
//! model handles two paths:
//!
//!   - `forward_images(pixel_values, text_tokens)` — encode an
//!     image+text pair through PaliGemma's vision tower +
//!     multi-modal projection + Gemma backbone, then project the
//!     per-token hidden states to 128-d and L2-normalize.
//!
//!   - `forward_text(text_tokens)` — text-only path through Gemma
//!     (no vision), same projection + normalization.
//!
//! Output shape: `(1, seq_len, 128)` — per-token dense embeddings
//! suitable for ColBERT-style late-interaction similarity scoring.
//!
//! Built on top of:
//!   - `lazy_paligemma::PaligemmaModel::forward_hidden` (added in
//!     this commit) — multi-modal hidden states without lm_head.
//!   - `lazy_gemma::GemmaModel::forward_hidden` — text-only
//!     hidden states without lm_head.
//!
//! v1 scope: F32, batch == 1, prefill only.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_gemma::GemmaModel;
use crate::lazy_paligemma::{PaligemmaConfig, PaligemmaModel, PaligemmaWeights};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

/// Output embedding dim for ColPali's late-interaction
/// projection. Matches the eager `colpali` constant 128.
pub const COLPALI_PROJ_DIM: usize = 128;

#[derive(Debug, Clone)]
pub struct ColPaliWeights {
    pub paligemma: PaligemmaWeights,
    /// `[text_hidden_size, 128]`.
    pub custom_text_projection: WeightStorage,
    /// `[128]`.
    pub custom_text_projection_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct ColPaliModel {
    pub config: PaligemmaConfig,
    pub weights: ColPaliWeights,
}

impl ColPaliModel {
    /// Encode an image+text pair into per-token 128-d L2-normalized
    /// embeddings `(1, num_patches + text_len, 128)`.
    pub fn forward_images(
        &self,
        pixel_values: &LazyTensor,
        text_tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let pg = PaligemmaModel {
            config: cfg.clone(),
            weights: self.weights.paligemma.clone(),
        };
        let hidden = pg.forward_hidden(pixel_values, text_tokens)?;
        self.project_and_normalize(&hidden, pixel_values)
    }

    /// Encode a text-only token sequence into per-token 128-d
    /// L2-normalized embeddings `(1, text_len, 128)`.
    pub fn forward_text(
        &self,
        text_tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let gemma = GemmaModel {
            config: cfg.text_config.clone(),
            weights: self.weights.paligemma.text.clone(),
        };
        let hidden = gemma.forward_hidden(text_tokens, 0)?;
        // Build an anchor on the same graph as `hidden` (which is
        // gemma's graph) so the projection constants land on-graph.
        self.project_and_normalize(&hidden, &hidden)
    }

    fn project_and_normalize(
        &self,
        hidden: &LazyTensor,
        anchor: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h_dim = cfg.text_config.hidden_size;
        let projected = self.weights.custom_text_projection.apply_linear(
            hidden, h_dim, COLPALI_PROJ_DIM,
        );
        let bias = anchor.const_f32_like(
            Arc::clone(&self.weights.custom_text_projection_bias),
            Shape::from_dims(&[COLPALI_PROJ_DIM]),
        );
        let biased = projected.broadcast_add(&bias)?;
        l2_normalize_last(&biased, 1e-12)
    }
}

fn l2_normalize_last(x: &LazyTensor, eps: f64) -> Result<LazyTensor> {
    let last = x.shape().dims().len() - 1;
    x.l2_normalize(last, eps)
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy_gemma::{GemmaConfig, GemmaWeights};
    use crate::lazy_paligemma::PaligemmaWeights;
    use crate::lazy_siglip::{SiglipVisionConfig, SiglipVisionWeights, SiglipActivation};
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

    fn ws(n: usize, nb: &mut dyn FnMut() -> f32) -> WeightStorage {
        WeightStorage::F32(vec_of(n, nb))
    }

    fn tiny_vision_cfg() -> SiglipVisionConfig {
        SiglipVisionConfig {
            hidden_size: 8, intermediate_size: 16,
            num_hidden_layers: 1, num_attention_heads: 2,
            num_channels: 3, image_size: 8, patch_size: 4,
            hidden_activation: SiglipActivation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }

    fn tiny_text_cfg() -> GemmaConfig {
        GemmaConfig {
            vocab_size: 16, hidden_size: 8, intermediate_size: 16,
            num_hidden_layers: 2, num_attention_heads: 2,
            num_key_value_heads: 1, head_dim: 4,
            rms_norm_eps: 1e-6, rope_theta: 10_000.0,
            max_position_embeddings: 64,
            attention_bias: false,
            hidden_activation: crate::lazy_gemma::GemmaActivation::GeluPytorchTanh,
        }
    }

    /// Tiny synthetic weights — small enough to exercise the
    /// graph in milliseconds. Mirrors the PaliGemma test scaffolding.
    fn tiny_colpali_weights(
        v_cfg: &SiglipVisionConfig, t_cfg: &GemmaConfig,
    ) -> ColPaliWeights {
        use crate::lazy::LayerWeights;
        let mut nb = rng_seed(2026);
        let v_h = v_cfg.hidden_size;
        let i_size = v_cfg.intermediate_size;
        let np = v_cfg.num_patches();
        let layers: Vec<crate::lazy_siglip::SiglipEncoderLayerWeights> =
            (0..v_cfg.num_hidden_layers).map(|_| {
                crate::lazy_siglip::SiglipEncoderLayerWeights {
                    ln1_gain: Arc::from(vec![1.0_f32; v_h]),
                    ln1_bias: Arc::from(vec![0.0_f32; v_h]),
                    q_proj: ws(v_h * v_h, &mut nb), q_proj_bias: vec_of(v_h, &mut nb),
                    k_proj: ws(v_h * v_h, &mut nb), k_proj_bias: vec_of(v_h, &mut nb),
                    v_proj: ws(v_h * v_h, &mut nb), v_proj_bias: vec_of(v_h, &mut nb),
                    out_proj: ws(v_h * v_h, &mut nb),
                    out_proj_bias: vec_of(v_h, &mut nb),
                    ln2_gain: Arc::from(vec![1.0_f32; v_h]),
                    ln2_bias: Arc::from(vec![0.0_f32; v_h]),
                    fc1: ws(v_h * i_size, &mut nb),
                    fc1_bias: vec_of(i_size, &mut nb),
                    fc2: ws(i_size * v_h, &mut nb),
                    fc2_bias: vec_of(v_h, &mut nb),
                }
            }).collect();
        let vision = SiglipVisionWeights {
            patch_proj: vec_of(v_h * v_cfg.num_channels * v_cfg.patch_size * v_cfg.patch_size, &mut nb),
            patch_proj_bias: vec_of(v_h, &mut nb),
            position_embedding: vec_of(np * v_h, &mut nb),
            layers,
            post_ln_gain: Arc::from(vec![1.0_f32; v_h]),
            post_ln_bias: Arc::from(vec![0.0_f32; v_h]),
            head: None,
        };

        let t_h = t_cfg.hidden_size;
        let i_t = t_cfg.intermediate_size;
        let n_kv = t_cfg.num_key_value_heads;
        let kv_dim = n_kv * t_cfg.head_dim;
        let gemma_layers: Vec<LayerWeights> = (0..t_cfg.num_hidden_layers).map(|_| LayerWeights {
            attn_q: ws(t_h * t_h, &mut nb), attn_q_bias: None,
            attn_k: ws(t_h * kv_dim, &mut nb), attn_k_bias: None,
            attn_v: ws(t_h * kv_dim, &mut nb), attn_v_bias: None,
            attn_o: ws(t_h * t_h, &mut nb),
            ffn_gate: ws(t_h * i_t, &mut nb),
            ffn_up:   ws(t_h * i_t, &mut nb),
            ffn_down: ws(i_t * t_h, &mut nb),
            attn_norm_gain: Arc::from(vec![1.0_f32; t_h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; t_h]),
        }).collect();
        let text = GemmaWeights {
            token_embedding: vec_of(t_cfg.vocab_size * t_h, &mut nb),
            layers: gemma_layers,
            final_norm_gain: Arc::from(vec![1.0_f32; t_h]),
            output: ws(t_h * t_cfg.vocab_size, &mut nb),
        };

        let paligemma = PaligemmaWeights {
            vision,
            mm_proj: ws(v_h * t_h, &mut nb),
            mm_proj_bias: vec_of(t_h, &mut nb),
            text,
        };

        ColPaliWeights {
            paligemma,
            custom_text_projection: ws(t_h * COLPALI_PROJ_DIM, &mut nb),
            custom_text_projection_bias: vec_of(COLPALI_PROJ_DIM, &mut nb),
        }
    }

    fn tiny_cfg() -> PaligemmaConfig {
        PaligemmaConfig {
            vision_config: tiny_vision_cfg(),
            text_config: tiny_text_cfg(),
            projection_dim: 8, // == text hidden_size
        }
    }

    #[test]
    fn forward_text_shape_and_l2_normalized() {
        let cfg = tiny_cfg();
        let weights = tiny_colpali_weights(&cfg.vision_config, &cfg.text_config);
        let model = ColPaliModel { config: cfg.clone(), weights };
        let tokens = vec![1_u32, 2, 3, 4];
        let out = model.forward_text(&tokens).unwrap();
        assert_eq!(out.shape().dims(), &[1, tokens.len(), COLPALI_PROJ_DIM]);
        let data = out.realize_f32();
        // Per-token L2 norm must be 1 (within tolerance).
        for t in 0..tokens.len() {
            let mut sum_sq = 0.0_f32;
            for d in 0..COLPALI_PROJ_DIM {
                let v = data[t * COLPALI_PROJ_DIM + d];
                sum_sq += v * v;
                assert!(v.is_finite(), "non-finite at t={t} d={d}: {v}");
            }
            assert!((sum_sq - 1.0).abs() < 1e-4,
                "L2 norm not unit at t={t}: sum_sq = {sum_sq}");
        }
    }

    #[test]
    fn forward_images_shape_and_l2_normalized() {
        let cfg = tiny_cfg();
        let weights = tiny_colpali_weights(&cfg.vision_config, &cfg.text_config);
        let model = ColPaliModel { config: cfg.clone(), weights };
        let img_size = cfg.vision_config.image_size;
        let pixel_values = LazyTensor::from_f32(
            (0..(3 * img_size * img_size))
                .map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, img_size, img_size]),
            &Device::cpu(),
        );
        let tokens = vec![5_u32, 6, 7];
        let out = model.forward_images(&pixel_values, &tokens).unwrap();
        let np = cfg.vision_config.num_patches();
        assert_eq!(out.shape().dims(), &[1, np + tokens.len(), COLPALI_PROJ_DIM]);
        let data = out.realize_f32();
        for t in 0..(np + tokens.len()) {
            let mut sum_sq = 0.0_f32;
            for d in 0..COLPALI_PROJ_DIM {
                let v = data[t * COLPALI_PROJ_DIM + d];
                sum_sq += v * v;
                assert!(v.is_finite(), "non-finite at t={t} d={d}: {v}");
            }
            assert!((sum_sq - 1.0).abs() < 1e-4,
                "image embeddings L2 norm not unit at t={t}: sum_sq = {sum_sq}");
        }
    }

    /// Different text tokens must produce different embeddings —
    /// proves the text projection is wired through to the output.
    #[test]
    fn forward_text_responds_to_tokens() {
        let cfg = tiny_cfg();
        let weights = tiny_colpali_weights(&cfg.vision_config, &cfg.text_config);
        let model = ColPaliModel { config: cfg, weights };
        let a = model.forward_text(&[1_u32, 2, 3]).unwrap().realize_f32();
        let b = model.forward_text(&[5_u32, 6, 7]).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "text embeddings must respond to token changes, max_diff = {max_diff}");
    }
}

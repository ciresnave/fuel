//! MobileCLIP — lazy port.
//!
//! Composition wrapper: pairs a [`crate::lazy_fastvit::FastVitModel`]
//! image encoder with a [`crate::lazy_openclip_text::OpenClipTextModel`]
//! text encoder, plus a trainable text projection and a
//! `logit_scale` scalar. Outputs L2-normalized image / text features
//! and contrastive logits, matching the eager MobileCLIP API.
//!
//! v1 scope: F32, batch == 1, prefill only.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_fastvit::{FastVitConfig, FastVitModel, FastVitWeights};
use crate::lazy_openclip_text::{OpenClipTextConfig, OpenClipTextModel, OpenClipTextWeights};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MobileClipConfig {
    pub vision: FastVitConfig,
    pub text: OpenClipTextConfig,
    pub image_size: usize,
    pub projection_dim: usize,
}

impl MobileClipConfig {
    /// MobileCLIP-S1: FastViT-MCI1 vision + ViT-B/32 text @ 256×256.
    pub fn s1() -> Self {
        Self {
            vision: FastVitConfig::mci1(),
            text: OpenClipTextConfig::vit_base_patch32(),
            image_size: 256,
            projection_dim: 512,
        }
    }
    /// MobileCLIP-S2: FastViT-MCI2 vision + ViT-B/32 text @ 256×256.
    pub fn s2() -> Self {
        Self {
            vision: FastVitConfig::mci2(),
            text: OpenClipTextConfig::vit_base_patch32(),
            image_size: 256,
            projection_dim: 512,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MobileClipWeights {
    pub vision: FastVitWeights,
    pub text: OpenClipTextWeights,
    /// `(text.embed_dim, projection_dim)`.
    pub text_projection: WeightStorage,
    /// Scalar `logit_scale`, exponentiated at forward time.
    pub logit_scale: f32,
}

#[derive(Debug, Clone)]
pub struct MobileClipModel {
    pub config: MobileClipConfig,
    pub weights: MobileClipWeights,
}

impl MobileClipModel {
    /// Build the contained vision sub-model with `num_classes =
    /// Some(projection_dim)` so its trailing linear head doubles as
    /// the image-feature projection (matches the eager
    /// `fastvit(cfg, 512, ...)` construction).
    pub fn vision_model(&self) -> FastVitModel {
        let mut cfg = self.config.vision.clone();
        cfg.num_classes = Some(self.config.projection_dim);
        FastVitModel { config: cfg, weights: self.weights.vision.clone() }
    }

    pub fn text_model(&self) -> OpenClipTextModel {
        OpenClipTextModel {
            config: self.config.text.clone(),
            weights: self.weights.text.clone(),
        }
    }

    /// Encode an image into a `(1, projection_dim)` feature vector.
    pub fn get_image_features(&self, image: &LazyTensor) -> Result<LazyTensor> {
        self.vision_model().forward(image)
    }

    /// Encode `input_ids` with `eot_pos` (position of the EOT token)
    /// into a `(1, projection_dim)` feature vector.
    pub fn get_text_features(
        &self, input_ids: &[u32], eot_pos: usize, anchor: &LazyTensor,
    ) -> Result<LazyTensor> {
        let pooled = self.text_model().forward_pooled(input_ids, eot_pos)?;
        let cfg = &self.config;
        let proj = self.weights.text_projection.apply_linear(
            &pooled, cfg.text.embed_dim, cfg.projection_dim,
        );
        let _ = anchor;
        Ok(proj)
    }

    /// Compute the contrastive logits given pre-extracted image and text
    /// feature vectors `(N_image, projection_dim)` / `(N_text,
    /// projection_dim)` — both must already live on the same graph.
    /// Returns `(logits_per_text, logits_per_image)` shaped
    /// `(N_text, N_image)` and `(N_image, N_text)`.
    ///
    /// Production CLIP usage typically realizes image and text features
    /// independently (different anchors, pre-cached for many query
    /// texts) and only assembles the contrastive matmul once — that's
    /// what this entry point is for.
    pub fn contrastive_logits(
        &self, image_features: &LazyTensor, text_features: &LazyTensor,
    ) -> Result<(LazyTensor, LazyTensor)> {
        let image_normed = l2_normalize_last(image_features)?;
        let text_normed = l2_normalize_last(text_features)?;
        let logits = text_normed.matmul(&image_normed.permute([1, 0_usize])?)?;
        let logit_scale = self.weights.logit_scale.exp() as f64;
        let logits_per_text = logits.mul_scalar(logit_scale);
        let logits_per_image = logits_per_text.permute([1, 0_usize])?;
        Ok((logits_per_text, logits_per_image))
    }
}

fn l2_normalize_last(x: &LazyTensor) -> Result<LazyTensor> {
    let sq = x.mul(x)?;
    let sum = sq.sum_dim(1_usize)?;
    let norm = sum
        .sqrt()
        .reshape(Shape::from_dims(&[x.shape().dims()[0], 1]))?
        .broadcast_to(Shape::from_dims(x.shape().dims()))?;
    x.div(&norm)
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy_fastvit::{
        AttentionBlockWeights, BnWeights, Conv2dBiasWeights, ConvMlpWeights,
        FastVitAttentionWeights, FastVitHeadWeights, FastVitStageBlocks,
        PatchEmbedWeights, RepMixerBlockWeights, RepMixerWeights,
        ReparamMobileOneWeights, SeWeights, StageWeights,
    };

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

    fn presets_construct_works() {
        let s1 = MobileClipConfig::s1();
        assert_eq!(s1.projection_dim, 512);
        let s2 = MobileClipConfig::s2();
        assert_eq!(s2.projection_dim, 512);
    }

    #[test]
    fn presets_construct() { presets_construct_works(); }

    fn conv_w(
        c_in: usize, c_out: usize, k: usize, stride: usize, pad: usize, groups: usize,
        nb: &mut dyn FnMut() -> f32,
    ) -> Conv2dBiasWeights {
        Conv2dBiasWeights {
            w: vec_of(c_out * (c_in / groups) * k * k, nb),
            b: vec_of(c_out, nb),
            c_in, c_out, k, stride, pad, groups,
        }
    }
    fn se_w(c: usize, nb: &mut dyn FnMut() -> f32) -> SeWeights {
        let sq = (c / 16).max(1);
        SeWeights {
            fc1: conv_w(c, sq, 1, 1, 0, 1, nb),
            fc2: conv_w(sq, c, 1, 1, 0, 1, nb),
        }
    }
    fn reparam_mobileone(
        c_in: usize, c_out: usize, k: usize, stride: usize, groups: usize,
        with_se: bool, use_act: bool, nb: &mut dyn FnMut() -> f32,
    ) -> ReparamMobileOneWeights {
        let pad = k / 2;
        ReparamMobileOneWeights {
            conv: conv_w(c_in, c_out, k, stride, pad, groups, nb),
            se: if with_se { Some(se_w(c_out, nb)) } else { None },
            use_act,
        }
    }
    fn conv_mlp_w(dim: usize, exp: usize, nb: &mut dyn FnMut() -> f32) -> ConvMlpWeights {
        ConvMlpWeights {
            conv_norm: conv_w(dim, dim, 7, 1, 3, dim, nb),
            fc1: conv_w(dim, dim * exp, 1, 1, 0, 1, nb),
            fc2: conv_w(dim * exp, dim, 1, 1, 0, 1, nb),
        }
    }
    fn repmixer_block_w(dim: usize, exp: usize, nb: &mut dyn FnMut() -> f32) -> RepMixerBlockWeights {
        RepMixerBlockWeights {
            gamma_mlp: vec_of(dim, nb),
            token_mixer: RepMixerWeights {
                gamma: vec_of(dim, nb),
                mixer: reparam_mobileone(dim, dim, 3, 1, dim, false, false, nb),
                norm: reparam_mobileone(dim, dim, 3, 1, dim, false, false, nb),
            },
            mlp: conv_mlp_w(dim, exp, nb),
        }
    }
    fn attention_block_w(dim: usize, exp: usize, nb: &mut dyn FnMut() -> f32) -> AttentionBlockWeights {
        AttentionBlockWeights {
            gamma1: vec_of(dim, nb), gamma2: vec_of(dim, nb),
            norm_bn: BnWeights {
                w: Arc::from(vec![1.0_f32; dim]),
                b: Arc::from(vec![0.0_f32; dim]),
            },
            token_mixer: FastVitAttentionWeights {
                qkv: ws(dim * 3 * dim, nb),
                proj: ws(dim * dim, nb),
                proj_bias: vec_of(dim, nb),
            },
            mlp: conv_mlp_w(dim, exp, nb),
        }
    }
    fn patch_embed_w(c_in: usize, c_out: usize, nb: &mut dyn FnMut() -> f32) -> PatchEmbedWeights {
        PatchEmbedWeights {
            large_conv: conv_w(c_in, c_out, 7, 2, 3, 1, nb),
            small_conv: conv_w(c_in, c_out, 3, 2, 1, 1, nb),
            se: Some(se_w(c_out, nb)),
            mobileone_1x1: reparam_mobileone(c_out, c_out, 1, 1, 1, false, true, nb),
        }
    }

    fn build_tiny_fastvit_weights(cfg: &FastVitConfig, nb: &mut dyn FnMut() -> f32) -> FastVitWeights {
        let c0 = cfg.in_channels;
        let stem = [
            reparam_mobileone(3, c0, 3, 2, 1, false, true, nb),
            reparam_mobileone(c0, c0, 3, 2, c0, false, true, nb),
            reparam_mobileone(c0, c0, 1, 1, 1, false, true, nb),
        ];
        let stages: [StageWeights; 4] = [
            StageWeights {
                downsample: None,
                pos_emb: Some(conv_w(c0, c0, 7, 1, 3, c0, nb)),
                blocks: FastVitStageBlocks::RepMixer(
                    (0..cfg.blocks[0]).map(|_| repmixer_block_w(c0, cfg.exp_ratio, nb)).collect()),
            },
            StageWeights {
                downsample: Some(patch_embed_w(c0, c0 * 2, nb)),
                pos_emb: Some(conv_w(c0 * 2, c0 * 2, 7, 1, 3, c0 * 2, nb)),
                blocks: FastVitStageBlocks::RepMixer(
                    (0..cfg.blocks[1]).map(|_| repmixer_block_w(c0 * 2, cfg.exp_ratio, nb)).collect()),
            },
            StageWeights {
                downsample: Some(patch_embed_w(c0 * 2, c0 * 4, nb)),
                pos_emb: Some(conv_w(c0 * 4, c0 * 4, 7, 1, 3, c0 * 4, nb)),
                blocks: FastVitStageBlocks::RepMixer(
                    (0..cfg.blocks[2]).map(|_| repmixer_block_w(c0 * 4, cfg.exp_ratio, nb)).collect()),
            },
            StageWeights {
                downsample: Some(patch_embed_w(c0 * 4, c0 * 8, nb)),
                pos_emb: Some(conv_w(c0 * 8, c0 * 8, 7, 1, 3, c0 * 8, nb)),
                blocks: FastVitStageBlocks::Attention(
                    (0..cfg.blocks[3]).map(|_| attention_block_w(c0 * 8, cfg.exp_ratio, nb)).collect()),
            },
        ];
        let final_c = c0 * 8;
        let head = cfg.num_classes.map(|n| FastVitHeadWeights {
            conv: conv_w(final_c, final_c, 1, 1, 0, 1, nb),
            linear_w: ws(final_c * n, nb),
            linear_b: vec_of(n, nb),
        });
        FastVitWeights { stem, stages, head }
    }

    fn tiny_mobileclip() -> MobileClipModel {
        // Vision tiny config matching the test setup in lazy_fastvit.
        let vision_cfg = FastVitConfig {
            in_channels: 8, blocks: [1, 1, 1, 1],
            exp_ratio: 2, attn: true, lkc_use_act: true,
            head_dim: 4, image_size: 32,
            num_classes: Some(16),
        };
        // Text tiny config (vocab 32, embed 8, 1 layer, max_pos 4).
        let text_cfg = OpenClipTextConfig {
            vocab_size: 32, embed_dim: 8, intermediate_size: 16,
            num_hidden_layers: 1, num_attention_heads: 2,
            max_position_embeddings: 4,
        };
        let cfg = MobileClipConfig {
            vision: vision_cfg.clone(),
            text: text_cfg.clone(),
            image_size: 32,
            projection_dim: 16,
        };
        let mut nb = rng_seed(2026);
        let vision = build_tiny_fastvit_weights(&vision_cfg, &mut nb);
        let text = build_tiny_openclip_text_weights(&text_cfg, &mut nb);
        let weights = MobileClipWeights {
            vision, text,
            text_projection: ws(text_cfg.embed_dim * cfg.projection_dim, &mut nb),
            logit_scale: 0.07_f32.ln(),
        };
        MobileClipModel { config: cfg, weights }
    }

    // Build a tiny OpenClipText weight set mirroring the layout in
    // lazy_openclip_text (kept inline to avoid exposing private weight
    // helpers from that crate).
    fn build_tiny_openclip_text_weights(
        cfg: &OpenClipTextConfig, nb: &mut dyn FnMut() -> f32,
    ) -> OpenClipTextWeights {
        use crate::lazy_openclip_text::{
            LayerNormWeights, MlpWeights, OpenClipAttentionWeights,
            OpenClipEncoderLayerWeights,
        };
        let e = cfg.embed_dim;
        let layers: Vec<OpenClipEncoderLayerWeights> = (0..cfg.num_hidden_layers).map(|_| {
            OpenClipEncoderLayerWeights {
                ln1: LayerNormWeights {
                    gain: Arc::from(vec![1.0_f32; e]),
                    bias: Arc::from(vec![0.0_f32; e]),
                },
                attn: OpenClipAttentionWeights {
                    q_proj: ws(e * e, nb),
                    q_proj_bias: vec_of(e, nb),
                    k_proj: ws(e * e, nb),
                    k_proj_bias: vec_of(e, nb),
                    v_proj: ws(e * e, nb),
                    v_proj_bias: vec_of(e, nb),
                    out_proj: ws(e * e, nb),
                    out_proj_bias: vec_of(e, nb),
                },
                ln2: LayerNormWeights {
                    gain: Arc::from(vec![1.0_f32; e]),
                    bias: Arc::from(vec![0.0_f32; e]),
                },
                mlp: MlpWeights {
                    fc1: ws(e * cfg.intermediate_size, nb),
                    fc1_bias: vec_of(cfg.intermediate_size, nb),
                    fc2: ws(cfg.intermediate_size * e, nb),
                    fc2_bias: vec_of(e, nb),
                },
            }
        }).collect();
        OpenClipTextWeights {
            token_embedding: vec_of(cfg.vocab_size * e, nb),
            position_embedding: vec_of(cfg.max_position_embeddings * e, nb),
            layers,
            final_ln: LayerNormWeights {
                gain: Arc::from(vec![1.0_f32; e]),
                bias: Arc::from(vec![0.0_f32; e]),
            },
        }
    }

    #[test]
    fn image_features_shape_and_finite() {
        use crate::Device;
        let model = tiny_mobileclip();
        let image = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let img_feats = model.get_image_features(&image).unwrap();
        assert_eq!(img_feats.shape().dims(), &[1, 16]);
        for &v in &img_feats.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn text_features_shape_and_finite() {
        let model = tiny_mobileclip();
        let ids = vec![1_u32, 2, 3, 4];
        // Anchor unused in text path; pass a stub.
        use crate::Device;
        let stub = LazyTensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu());
        let txt_feats = model.get_text_features(&ids, 3, &stub).unwrap();
        assert_eq!(txt_feats.shape().dims(), &[1, 16]);
        for &v in &txt_feats.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn l2_normalize_unit_norm() {
        use crate::Device;
        let x = LazyTensor::from_f32(
            vec![3.0_f32, 4.0, 0.0, 0.0, 1.0, 2.0],
            Shape::from_dims(&[2, 3]), &Device::cpu(),
        );
        let n = l2_normalize_last(&x).unwrap();
        let n_data = n.realize_f32();
        let n0 = (n_data[0].powi(2) + n_data[1].powi(2) + n_data[2].powi(2)).sqrt();
        let n1 = (n_data[3].powi(2) + n_data[4].powi(2) + n_data[5].powi(2)).sqrt();
        assert!((n0 - 1.0).abs() < 1e-5, "row 0 norm = {n0}");
        assert!((n1 - 1.0).abs() < 1e-5, "row 1 norm = {n1}");
    }
}

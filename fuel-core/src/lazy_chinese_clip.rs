//! Chinese-CLIP — lazy port.
//!
//! Composition wrapper: pairs a [`crate::lazy_bert::BertModel`]
//! Chinese-BERT text encoder with a
//! [`crate::lazy_clip::ClipVisionModel`] CLIP-style ViT image
//! encoder, plus separate text/visual projections and a
//! learnable `logit_scale` scalar.
//!
//! The architectural differences from MobileCLIP / OpenAI-CLIP
//! are weight-loading only — the vision tower is bit-identical
//! to OpenAI-CLIP and the text tower is bit-identical to BERT.
//! Chinese-CLIP differs only in its training data
//! (Chinese-language) and weight values.
//!
//! v1 scope: F32, batch == 1, prefill only.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_bert::{BertConfig, BertModel, BertWeights};
use crate::lazy_clip::{ClipVisionConfig, ClipVisionModel, ClipVisionWeights};
use crate::Result;
use fuel_ir::Shape;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ChineseClipConfig {
    pub text: BertConfig,
    pub vision: ClipVisionConfig,
    pub projection_dim: usize,
    pub logit_scale_init_value: f32,
    pub image_size: usize,
}

impl ChineseClipConfig {
    /// `OFA-Sys/chinese-clip-vit-base-patch16` preset.
    pub fn clip_vit_base_patch16() -> Self {
        let text = BertConfig {
            vocab_size: 21128,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        };
        let vision = ClipVisionConfig {
            embed_dim: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 512,
            num_channels: 3,
            image_size: 224,
            patch_size: 16,
        };
        Self {
            text, vision,
            projection_dim: 512,
            logit_scale_init_value: 2.6592,
            image_size: 224,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChineseClipWeights {
    pub text: BertWeights,
    pub vision: ClipVisionWeights,
    /// `(vision_embed_dim, projection_dim)`.
    pub visual_projection: WeightStorage,
    /// `(text_hidden_size, projection_dim)`.
    pub text_projection: WeightStorage,
    /// Scalar; exponentiated at logit-scale time.
    pub logit_scale: f32,
}

#[derive(Debug, Clone)]
pub struct ChineseClipModel {
    pub config: ChineseClipConfig,
    pub weights: ChineseClipWeights,
}

impl ChineseClipModel {
    pub fn text_model(&self) -> BertModel {
        BertModel::new(self.config.text.clone(), self.weights.text.clone())
    }

    pub fn vision_model(&self) -> ClipVisionModel {
        ClipVisionModel {
            config: self.config.vision.clone(),
            weights: self.weights.vision.clone(),
        }
    }

    /// Encode an image into a `(1, projection_dim)` feature vector.
    pub fn get_image_features(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let pooled = self.vision_model().forward(image)?;
        // (1, vision_embed_dim) → (1, projection_dim)
        Ok(self.weights.visual_projection.apply_linear(
            &pooled, self.config.vision.embed_dim, self.config.projection_dim,
        ))
    }

    /// Encode `input_ids` (Chinese-BERT-tokenized) into a
    /// `(1, projection_dim)` feature vector. CLS-pooled then
    /// projected through `text_projection` (no bias, matches eager).
    pub fn get_text_features(&self, input_ids: &[u32]) -> Result<LazyTensor> {
        let hidden = self.text_model().forward(input_ids)?;
        let h = self.config.text.hidden_size;
        // (1, T, hidden) → CLS at position 0 → (1, hidden)
        let cls = hidden
            .narrow(1_usize, 0, 1)?
            .reshape(Shape::from_dims(&[1, h]))?;
        Ok(self.weights.text_projection.apply_linear(
            &cls, h, self.config.projection_dim,
        ))
    }

    /// Build contrastive logits from already-extracted features.
    /// Both inputs must live on the same graph; see `lazy_mobileclip`
    /// for the rationale (the BERT text path anchors on its own
    /// U32 ids tensor, distinct from the vision graph).
    pub fn contrastive_logits(
        &self, image_features: &LazyTensor, text_features: &LazyTensor,
    ) -> Result<(LazyTensor, LazyTensor)> {
        let image_normed = l2_normalize_last(image_features)?;
        let text_normed = l2_normalize_last(text_features)?;
        let logits = text_normed.matmul(&image_normed.permute([1, 0_usize])?)?;
        let scale = self.weights.logit_scale.exp() as f64;
        let logits_per_text = logits.mul_scalar(scale);
        let logits_per_image = logits_per_text.permute([1, 0_usize])?;
        Ok((logits_per_text, logits_per_image))
    }
}

fn l2_normalize_last(x: &LazyTensor) -> Result<LazyTensor> {
    x.l2_normalize(1_usize, 0.0)
}

// ---- HuggingFace safetensors composer --------------------------------------

impl ChineseClipWeights {
    /// Load a full Chinese-CLIP checkpoint
    /// (e.g. `OFA-Sys/chinese-clip-vit-base-patch16`). Composes the
    /// shipped BertWeights + ClipVisionWeights loaders and reads the
    /// two projection heads + logit_scale at the top level.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &ChineseClipConfig,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
        let text = BertWeights::load_from_mmapped(st, &cfg.text)?;
        let vision = ClipVisionWeights::load_from_mmapped(
            st, &cfg.vision, "vision_model.",
        )?;
        let text_projection = load_transposed_matrix_preserve_dtype(
            st, "text_projection.weight",
            cfg.projection_dim, cfg.text.hidden_size,
        )?;
        let visual_projection = load_transposed_matrix_preserve_dtype(
            st, "visual_projection.weight",
            cfg.projection_dim, cfg.vision.embed_dim,
        )?;
        let logit_scale = load_tensor_as_f32(st, "logit_scale")?
            .first().copied().unwrap_or(0.0);
        Ok(Self { text, vision, visual_projection, text_projection, logit_scale })
    }
}


// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy_bert::BertLayerWeights;
    use crate::lazy_clip::ClipEncoderLayerWeights;
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

    fn tiny_text_cfg() -> BertConfig {
        BertConfig {
            vocab_size: 32, hidden_size: 8,
            num_hidden_layers: 1, num_attention_heads: 2,
            intermediate_size: 16, max_position_embeddings: 16,
            type_vocab_size: 2, layer_norm_eps: 1e-12,
        }
    }

    fn tiny_vision_cfg() -> ClipVisionConfig {
        ClipVisionConfig {
            embed_dim: 8, intermediate_size: 16,
            num_hidden_layers: 1, num_attention_heads: 2,
            projection_dim: 4, num_channels: 3,
            image_size: 16, patch_size: 8,
        }
    }

    fn build_tiny_bert_layer(h: usize, ff: usize, nb: &mut dyn FnMut() -> f32) -> BertLayerWeights {
        BertLayerWeights {
            attn_q_w: vec_of(h * h, nb), attn_q_b: vec_of(h, nb),
            attn_k_w: vec_of(h * h, nb), attn_k_b: vec_of(h, nb),
            attn_v_w: vec_of(h * h, nb), attn_v_b: vec_of(h, nb),
            attn_out_w: vec_of(h * h, nb), attn_out_b: vec_of(h, nb),
            attn_ln_gamma: Arc::from(vec![1.0_f32; h]),
            attn_ln_beta: Arc::from(vec![0.0_f32; h]),
            ffn_in_w: vec_of(h * ff, nb), ffn_in_b: vec_of(ff, nb),
            ffn_out_w: vec_of(ff * h, nb), ffn_out_b: vec_of(h, nb),
            ffn_ln_gamma: Arc::from(vec![1.0_f32; h]),
            ffn_ln_beta: Arc::from(vec![0.0_f32; h]),
        }
    }

    fn build_tiny_clip_vision_layer(
        e: usize, ff: usize, nb: &mut dyn FnMut() -> f32,
    ) -> ClipEncoderLayerWeights {
        ClipEncoderLayerWeights {
            ln1_gain: Arc::from(vec![1.0_f32; e]),
            ln1_bias: Arc::from(vec![0.0_f32; e]),
            q_proj: ws(e * e, nb), q_proj_bias: vec_of(e, nb),
            k_proj: ws(e * e, nb), k_proj_bias: vec_of(e, nb),
            v_proj: ws(e * e, nb), v_proj_bias: vec_of(e, nb),
            out_proj: ws(e * e, nb), out_proj_bias: vec_of(e, nb),
            ln2_gain: Arc::from(vec![1.0_f32; e]),
            ln2_bias: Arc::from(vec![0.0_f32; e]),
            fc1: ws(e * ff, nb), fc1_bias: vec_of(ff, nb),
            fc2: ws(ff * e, nb), fc2_bias: vec_of(e, nb),
        }
    }

    fn tiny_model() -> ChineseClipModel {
        let text_cfg = tiny_text_cfg();
        let vision_cfg = tiny_vision_cfg();
        let projection_dim = 4;
        let cfg = ChineseClipConfig {
            text: text_cfg.clone(),
            vision: vision_cfg.clone(),
            projection_dim,
            logit_scale_init_value: 2.6592,
            image_size: 16,
        };
        let mut nb = rng_seed(2026);
        let h = text_cfg.hidden_size;
        let ff = text_cfg.intermediate_size;
        let text = BertWeights {
            word_embeddings: vec_of(text_cfg.vocab_size * h, &mut nb),
            position_embeddings: vec_of(text_cfg.max_position_embeddings * h, &mut nb),
            token_type_embeddings: vec_of(text_cfg.type_vocab_size * h, &mut nb),
            emb_ln_gamma: Arc::from(vec![1.0_f32; h]),
            emb_ln_beta: Arc::from(vec![0.0_f32; h]),
            layers: (0..text_cfg.num_hidden_layers)
                .map(|_| build_tiny_bert_layer(h, ff, &mut nb))
                .collect(),
        };
        let e = vision_cfg.embed_dim;
        let vff = vision_cfg.intermediate_size;
        let vision = ClipVisionWeights {
            patch_proj: vec_of(e * vision_cfg.num_channels * vision_cfg.patch_size * vision_cfg.patch_size, &mut nb),
            class_embedding: vec_of(e, &mut nb),
            position_embedding: vec_of((vision_cfg.num_patches() + 1) * e, &mut nb),
            pre_ln_gain: Arc::from(vec![1.0_f32; e]),
            pre_ln_bias: Arc::from(vec![0.0_f32; e]),
            layers: (0..vision_cfg.num_hidden_layers)
                .map(|_| build_tiny_clip_vision_layer(e, vff, &mut nb))
                .collect(),
            post_ln_gain: Arc::from(vec![1.0_f32; e]),
            post_ln_bias: Arc::from(vec![0.0_f32; e]),
        };
        let weights = ChineseClipWeights {
            text, vision,
            visual_projection: ws(e * projection_dim, &mut nb),
            text_projection: ws(h * projection_dim, &mut nb),
            logit_scale: 2.6592_f32.ln().max(0.0_f32),
        };
        ChineseClipModel { config: cfg, weights }
    }

    #[test]
    fn presets_construct() {
        let p = ChineseClipConfig::clip_vit_base_patch16();
        assert_eq!(p.projection_dim, 512);
        assert_eq!(p.text.vocab_size, 21128);
        assert_eq!(p.vision.patch_size, 16);
    }

    #[test]
    fn image_features_shape_and_finite() {
        let model = tiny_model();
        let image = LazyTensor::from_f32(
            (0..(3 * 16 * 16)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 16, 16]), &Device::cpu(),
        );
        let f = model.get_image_features(&image).unwrap();
        assert_eq!(f.shape().dims(), &[1, 4]);
        for &v in &f.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn text_features_shape_and_finite() {
        let model = tiny_model();
        let f = model.get_text_features(&[1_u32, 2, 3, 4]).unwrap();
        assert_eq!(f.shape().dims(), &[1, 4]);
        for &v in &f.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn contrastive_logits_shape_and_finite() {
        let model = tiny_model();
        // Build features on the SAME graph (test-only) by constructing
        // synthetic (1, 4) feature tensors anchored on a shared LazyTensor.
        let anchor = LazyTensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu());
        let img_feats = anchor.const_f32_like(
            Arc::from(vec![0.1_f32, 0.2, 0.3, 0.4]),
            Shape::from_dims(&[1, 4]),
        );
        let txt_feats = anchor.const_f32_like(
            Arc::from(vec![0.5_f32, -0.2, 0.1, 0.3]),
            Shape::from_dims(&[1, 4]),
        );
        let (lpt, lpi) = model.contrastive_logits(&img_feats, &txt_feats).unwrap();
        assert_eq!(lpt.shape().dims(), &[1, 1]);
        assert_eq!(lpi.shape().dims(), &[1, 1]);
        for &v in &lpt.realize_f32() { assert!(v.is_finite()); }
    }
}

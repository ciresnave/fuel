//! LLaVA (Large Language and Vision Assistant) ported to the
//! lazy-graph API.
//!
//! LLaVA (Liu et al. 2023) is a vision-language model that
//! combines a CLIP vision encoder with a LLaMA language model
//! via a Multi-Modal projector. Like PaliGemma, this is a
//! composition port — reuses [`crate::lazy_clip::ClipVisionModel`]
//! and [`crate::lazy::LlamaModel`] with a thin projection +
//! interleaving layer:
//!
//!   ```text
//!   image_hidden   = clip_vision(pixel_values)
//!                       # CLIP returns (1, embed_dim) CLS-pooled
//!                       # but LLaVA uses per-patch features, so
//!                       # we re-run without final pool & take
//!                       # all patches.
//!   image_features = MMProjector(image_hidden)
//!                       # Linear (or MLP) projects to LLaMA dim
//!   text_embeds    = llama.embed_tokens(text_tokens)
//!   combined       = cat(image_features, text_embeds, dim=1)
//!   logits         = llama.forward_embeds(combined, start_pos=0)
//!   ```
//!
//! v1 supports the **"linear"** projector variant
//! (`mm_projector_type = "linear"`). The MLP variants
//! (`mlp2x_gelu`, `mlp3x_gelu`, …) used by newer LLaVA
//! checkpoints are extensions — defer.
//!
//! v1 also uses the **full per-patch CLIP encoder output**
//! (no class token), matching `select_feature_method =
//! "patch"` in eager. The `cls_patch` variant (include CLS
//! token) and `select_layer = -2` (second-to-last hidden) are
//! left to follow-ups.
//!
//! # Scope (v1)
//!
//! Forward-only, single image + single token sequence, F32.
//! Multi-image / anyres / `image_newline` injection deferred.

use crate::lazy::{LayerWeights, LazyTensor, LlamaConfig, LlamaModel, LlamaWeights, WeightStorage};
use crate::lazy_clip::{ClipVisionConfig, ClipVisionWeights};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct LlavaConfig {
    pub vision_config: ClipVisionConfig,
    pub text_config: LlamaConfig,
    /// Projected dim — must equal `text_config.dim`.
    pub projection_dim: usize,
}

#[derive(Debug, Clone)]
pub struct LlavaWeights {
    pub vision: ClipVisionWeights,
    /// `[vision_embed_dim, projection_dim]`.
    pub mm_proj: WeightStorage,
    pub mm_proj_bias: Arc<[f32]>,
    pub text: LlamaWeights,
}

#[derive(Debug, Clone)]
pub struct LlavaModel {
    pub config: LlavaConfig,
    pub weights: LlavaWeights,
}

impl LlavaModel {
    /// Run the full multimodal forward pass. Returns logits for
    /// the combined `[image_features; text_embeds]` sequence
    /// of shape `(1, num_patches + text_len, vocab_size)`.
    pub fn forward(
        &self,
        pixel_values: &LazyTensor,
        text_tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let v_cfg = &cfg.vision_config;
        let t_cfg = &cfg.text_config;
        assert_eq!(
            cfg.projection_dim, t_cfg.dim,
            "v1: projection_dim must equal text_config.dim",
        );
        let text_len = text_tokens.len();
        assert!(text_len > 0, "text_tokens must be non-empty");

        // ---- Run CLIP vision encoder and harvest per-patch features ------
        // The packaged ClipVisionModel::forward returns CLS-pooled output;
        // LLaVA needs per-patch features, so we replicate the encoder body
        // here and DROP the CLS token after the final layer.
        let image_features = self.clip_vision_per_patch(pixel_values)?;
        let np = v_cfg.num_patches();
        let dims = image_features.shape();
        let dims = dims.dims();
        assert_eq!(dims, &[1, np, v_cfg.embed_dim],
            "CLIP per-patch features must be (1, num_patches, embed_dim); got {dims:?}");

        // ---- Multi-modal projection: vision_embed → text_dim ------------
        let projected = self.weights.mm_proj.apply_linear(
            &image_features,
            v_cfg.embed_dim,
            cfg.projection_dim,
        );
        let bias_t = pixel_values.const_f32_like(
            Arc::clone(&self.weights.mm_proj_bias),
            Shape::from_dims(&[cfg.projection_dim]),
        );
        let image_proj = projected.broadcast_add(&bias_t)?;

        // ---- Embed text tokens via LLaMA's token embedding -------------
        // Anchor on `pixel_values`' graph.
        let llama_embed_lt = pixel_values.const_f32_like(
            Arc::clone(&self.weights.text.token_embedding),
            Shape::from_dims(&[t_cfg.vocab_size, t_cfg.dim]),
        );
        let token_ids = pixel_values.const_u32_like(
            text_tokens.to_vec(),
            Shape::from_dims(&[text_len]),
        );
        let text_embeds = llama_embed_lt
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[1, text_len, t_cfg.dim]))?;

        // ---- Concat [image; text] and run LLaMA layers -----------------
        let combined = image_proj.concat(&text_embeds, 1_usize)?;
        let llama = LlamaModel {
            config: t_cfg.clone(),
            weights: self.weights.text.clone(),
        };
        llama.forward_embeds(&combined, 0)
    }

    /// Run the CLIP vision encoder and return the per-patch
    /// hidden states (NO class token in the output) after the
    /// final post-LN. Used by LLaVA as the visual feature
    /// stream feeding the MM projector.
    fn clip_vision_per_patch(&self, pixel_values: &LazyTensor) -> Result<LazyTensor> {
        let v_cfg = &self.config.vision_config;
        let weights = &self.weights.vision;
        let dims = pixel_values.shape();
        let dims = dims.dims();
        let batch = dims[0];
        assert_eq!(batch, 1, "v1 supports batch == 1");
        assert_eq!(dims[1], v_cfg.num_channels);
        assert_eq!(dims[2], v_cfg.image_size);
        assert_eq!(dims[3], v_cfg.image_size);

        // Patch Conv2d (no bias in CLIP).
        let conv_w = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj),
            Shape::from_dims(&[v_cfg.embed_dim, v_cfg.num_channels, v_cfg.patch_size, v_cfg.patch_size]),
        );
        let conv_out = pixel_values.conv2d(
            &conv_w,
            None,
            (v_cfg.patch_size, v_cfg.patch_size),
            (0, 0),
            1,
        )?;
        let np = v_cfg.num_patches();
        let patches = conv_out
            .reshape(Shape::from_dims(&[batch, v_cfg.embed_dim, np]))?
            .permute([0, 2, 1_usize])?;

        // Prepend class_embedding (CLIP does this; LLaVA later drops it).
        let cls = pixel_values.const_f32_like(
            Arc::clone(&weights.class_embedding),
            Shape::from_dims(&[1, 1, v_cfg.embed_dim]),
        );
        let cls_bc = cls.broadcast_to(Shape::from_dims(&[batch, 1, v_cfg.embed_dim]))?;
        let with_cls = cls_bc.concat(&patches, 1_usize)?;

        // Add position embedding.
        let pos = pixel_values.const_f32_like(
            Arc::clone(&weights.position_embedding),
            Shape::from_dims(&[np + 1, v_cfg.embed_dim]),
        );
        let pos_bc = pos
            .reshape(Shape::from_dims(&[1, np + 1, v_cfg.embed_dim]))?
            .broadcast_to(Shape::from_dims(&[batch, np + 1, v_cfg.embed_dim]))?;
        let pre = with_cls.add(&pos_bc)?;

        // Pre-LayerNorm.
        let pre_ln = crate::lazy::apply_affine_layer_norm_pub(
            &pre, &weights.pre_ln_gain, &weights.pre_ln_bias,
            v_cfg.embed_dim, 1e-5,
        );

        // Encoder layers (call the CLIP shared apply via the
        // public ClipVisionModel forward — but we need to override
        // the final pool. Easier: re-implement the encoder pass.
        let mut h = pre_ln;
        let n_heads = v_cfg.num_attention_heads;
        let head_dim = v_cfg.head_dim();
        for layer in &weights.layers {
            h = clip_encoder_layer(&h, layer, n_heads, head_dim, None)?;
        }

        // Drop CLS token (position 0); keep positions 1..num_patches+1.
        let patches_only = h.slice(1_usize, 1, np)?;

        // Post-LayerNorm on the patch tokens.
        Ok(crate::lazy::apply_affine_layer_norm_pub(
            &patches_only, &weights.post_ln_gain, &weights.post_ln_bias,
            v_cfg.embed_dim, 1e-5,
        ))
    }
}

/// One CLIP encoder layer (Pre-LN, MHA, residual, Pre-LN, MLP,
/// residual). Inlined here so we can drop the CLS token before
/// the post-LN.
fn clip_encoder_layer(
    x: &LazyTensor,
    layer: &crate::lazy_clip::ClipEncoderLayerWeights,
    n_heads: usize,
    head_dim: usize,
    causal_mask: Option<&LazyTensor>,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let batch = dims[0];
    let seq = dims[1];
    let h = dims[2];

    let x_norm = crate::lazy::apply_affine_layer_norm_pub(
        x, &layer.ln1_gain, &layer.ln1_bias, h, 1e-5,
    );

    let q = layer.q_proj.apply_linear(&x_norm, h, h);
    let q = bias_add(q, &layer.q_proj_bias, h, x)?;
    let k = layer.k_proj.apply_linear(&x_norm, h, h);
    let k = bias_add(k, &layer.k_proj_bias, h, x)?;
    let v = layer.v_proj.apply_linear(&x_norm, h, h);
    let v = bias_add(v, &layer.v_proj_bias, h, x)?;

    let _ = (batch, seq);
    let q = q.split_heads(n_heads, head_dim)?;
    let k = k.split_heads(n_heads, head_dim)?;
    let v = v.split_heads(n_heads, head_dim)?;

    let k_t = k.transpose()?;
    let scale = 1.0_f64 / (head_dim as f64).sqrt();
    let scores = q.matmul(&k_t)?.mul_scalar(scale);
    let scores = match causal_mask {
        None => scores,
        Some(m) => scores.broadcast_add(m)?,
    };
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?;
    let merged = ctx.merge_heads()?;
    let attn_out = layer.out_proj.apply_linear(&merged, h, h);
    let attn_out = bias_add(attn_out, &layer.out_proj_bias, h, x)?;
    let h1 = x.add(&attn_out)?;

    let h1_norm = crate::lazy::apply_affine_layer_norm_pub(
        &h1, &layer.ln2_gain, &layer.ln2_bias, h, 1e-5,
    );

    let inter_dim = layer.fc1_bias.len();
    let inter = layer.fc1.apply_linear(&h1_norm, h, inter_dim);
    let inter = bias_add(inter, &layer.fc1_bias, inter_dim, x)?;
    // QuickGelu: x * sigmoid(1.702 * x).
    let activated = {
        let scaled = inter.mul_scalar(1.702);
        let sig = scaled.sigmoid();
        inter.mul(&sig)?
    };
    let mlp_out = layer.fc2.apply_linear(&activated, inter_dim, h);
    let mlp_out = bias_add(mlp_out, &layer.fc2_bias, h, x)?;
    h1.add(&mlp_out)
}

fn bias_add(
    x: LazyTensor,
    bias: &Arc<[f32]>,
    n: usize,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let _ = (n, anchor);
    x.add_trailing_bias(Arc::clone(bias))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy_clip::ClipEncoderLayerWeights;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_vision_cfg() -> ClipVisionConfig {
        ClipVisionConfig {
            embed_dim: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            projection_dim: 8,
            num_channels: 3,
            image_size: 8,
            patch_size: 4,
        }
    }

    fn tiny_text_cfg() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-6,
            rope_base: 10_000.0,
        }
    }

    fn tiny_clip_layers(embed: usize, inter: usize, nb: &mut Box<dyn FnMut() -> f32>) -> ClipEncoderLayerWeights {
        ClipEncoderLayerWeights {
            ln1_gain: Arc::from(vec![1.0_f32; embed]),
            ln1_bias: Arc::from(vec![0.0_f32; embed]),
            q_proj: WeightStorage::F32(vec_of(embed * embed, &mut **nb)),
            q_proj_bias: vec_of(embed, &mut **nb),
            k_proj: WeightStorage::F32(vec_of(embed * embed, &mut **nb)),
            k_proj_bias: vec_of(embed, &mut **nb),
            v_proj: WeightStorage::F32(vec_of(embed * embed, &mut **nb)),
            v_proj_bias: vec_of(embed, &mut **nb),
            out_proj: WeightStorage::F32(vec_of(embed * embed, &mut **nb)),
            out_proj_bias: vec_of(embed, &mut **nb),
            ln2_gain: Arc::from(vec![1.0_f32; embed]),
            ln2_bias: Arc::from(vec![0.0_f32; embed]),
            fc1: WeightStorage::F32(vec_of(embed * inter, &mut **nb)),
            fc1_bias: vec_of(inter, &mut **nb),
            fc2: WeightStorage::F32(vec_of(inter * embed, &mut **nb)),
            fc2_bias: vec_of(embed, &mut **nb),
        }
    }

    fn tiny_vision_weights(cfg: &ClipVisionConfig) -> ClipVisionWeights {
        let mut s: u32 = 12121;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let patch_proj = vec_of(
            cfg.embed_dim * cfg.num_channels * cfg.patch_size * cfg.patch_size,
            &mut *nb,
        );
        let class_embedding = vec_of(cfg.embed_dim, &mut *nb);
        let position_embedding = vec_of((cfg.num_patches() + 1) * cfg.embed_dim, &mut *nb);
        let layers: Vec<_> = (0..cfg.num_hidden_layers).map(|_|
            tiny_clip_layers(cfg.embed_dim, cfg.intermediate_size, &mut nb)
        ).collect();
        ClipVisionWeights {
            patch_proj, class_embedding, position_embedding,
            pre_ln_gain: Arc::from(vec![1.0_f32; cfg.embed_dim]),
            pre_ln_bias: Arc::from(vec![0.0_f32; cfg.embed_dim]),
            layers,
            post_ln_gain: Arc::from(vec![1.0_f32; cfg.embed_dim]),
            post_ln_bias: Arc::from(vec![0.0_f32; cfg.embed_dim]),
        }
    }

    fn tiny_llama_weights(cfg: &LlamaConfig) -> LlamaWeights {
        let mut s: u32 = 34343;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let h = cfg.dim;
        let i = cfg.ffn_dim;
        let kv = cfg.n_kv_heads * cfg.head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.n_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        LlamaWeights { token_embedding, layers, final_norm_gain, output }
    }

    fn tiny_image(cfg: &ClipVisionConfig) -> LazyTensor {
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
        let mut s: u32 = 56565;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let mm_proj = WeightStorage::F32(vec_of(v_cfg.embed_dim * t_cfg.dim, &mut *nb));
        let mm_proj_bias = vec_of(t_cfg.dim, &mut *nb);
        let weights = LlavaWeights {
            vision: tiny_vision_weights(&v_cfg),
            mm_proj, mm_proj_bias,
            text: tiny_llama_weights(&t_cfg),
        };
        let cfg = LlavaConfig {
            vision_config: v_cfg.clone(),
            text_config: t_cfg.clone(),
            projection_dim: t_cfg.dim,
        };
        let model = LlavaModel { config: cfg, weights };

        let img = tiny_image(&v_cfg);
        let text_tokens = [1_u32, 2, 3];
        let logits = model.forward(&img, &text_tokens).unwrap();
        let expected_seq = v_cfg.num_patches() + text_tokens.len();
        assert_eq!(logits.shape().dims(), &[1, expected_seq, t_cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }
}

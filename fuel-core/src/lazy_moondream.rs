//! Moondream vision-language model ported to the lazy-graph API.
//!
//! Moondream (Vikhyat Korrapati). Lightweight (1.6B param)
//! multimodal model that answers visual questions about images.
//! Composes:
//!
//!   - A custom **ViT-style vision encoder** with a **linear
//!     patch embedding** (NOT Conv2d like standard ViT — the
//!     image is reshaped into per-patch flat vectors and
//!     projected with a single linear layer).
//!   - A **vision projection** MLP (Linear + GeluPytorchTanh
//!     + Linear) that maps vision embeddings into the
//!     language-model space.
//!   - **MixFormer** (Phi-1.5 family) as the text decoder
//!     ([`crate::lazy_mixformer::MixFormerModel`]).
//!
//! Architecturally identical to LLaVA / PaliGemma at the
//! composition level (`vision → projector → forward_embeds`)
//! but with a custom vision encoder and a different language
//! model. v1 establishes the recipe for any other vision
//! encoder that needs a linear (rather than convolutional)
//! patch embedding.
//!
//! # Scope (v1)
//!
//! Forward-only, single image + single token sequence, F32.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_mixformer::{MixFormerConfig, MixFormerModel, MixFormerWeights};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoondreamActivation {
    GeluPytorchTanh,
    Gelu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoondreamVisionConfig {
    /// Vision tower embedding dim (eager: `embed_dim`).
    pub embed_dim: usize,
    /// Number of vision tower blocks.
    pub num_blocks: usize,
    /// Number of attention heads per vision block.
    pub num_heads: usize,
    /// Vision MLP hidden dim (eager: `hidden_features`).
    pub mlp_hidden: usize,
    /// Number of patches per axis (eager: `embed_len` is total).
    /// For a 378×378 input at patch 14 ⇒ 27×27 = 729 patches.
    pub num_patches: usize,
    /// Patch side length.
    pub patch_size: usize,
    /// Number of input channels.
    pub num_channels: usize,
    /// Image side length (eager: 378 for v2).
    pub image_size: usize,
    pub activation: MoondreamActivation,
    pub layer_norm_eps: f64,
}

impl MoondreamVisionConfig {
    /// Patch-embed input dim = `num_channels * patch * patch`.
    pub fn patch_input_dim(&self) -> usize {
        self.num_channels * self.patch_size * self.patch_size
    }
    /// Moondream v2 vision config (378×378 image, 14-patch ⇒ 729 patches).
    pub fn v2() -> Self {
        Self {
            embed_dim: 1152,
            num_blocks: 27,
            num_heads: 16,
            mlp_hidden: 4304,
            num_patches: 729, // (378 / 14)^2
            patch_size: 14,
            num_channels: 3,
            image_size: 378,
            activation: MoondreamActivation::GeluPytorchTanh,
            layer_norm_eps: 1e-5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoondreamProjectionConfig {
    /// Vision embed_dim that feeds the projector (eager:
    /// `image_embedding_dim`).
    pub in_dim: usize,
    /// Hidden dim of the projector MLP (eager: `hidden_dim`).
    pub hidden_dim: usize,
    /// Output dim — must equal the language model's
    /// hidden_size (eager: `model_dim`).
    pub out_dim: usize,
    pub activation: MoondreamActivation,
}

impl MoondreamProjectionConfig {
    pub fn v2() -> Self {
        Self {
            in_dim: 1152,
            hidden_dim: 2048 * 4,
            out_dim: 2048,
            activation: MoondreamActivation::GeluPytorchTanh,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MoondreamVisionBlockWeights {
    pub norm1_gain: Arc<[f32]>,
    pub norm1_bias: Arc<[f32]>,
    /// Fused `[embed_dim, 3 * embed_dim]`.
    pub qkv: WeightStorage,
    pub qkv_bias: Arc<[f32]>,
    pub proj: WeightStorage,
    pub proj_bias: Arc<[f32]>,
    pub norm2_gain: Arc<[f32]>,
    pub norm2_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct MoondreamVisionWeights {
    /// Linear patch embedding `[c * p * p, embed_dim]`.
    pub patch_embed: WeightStorage,
    pub patch_embed_bias: Arc<[f32]>,
    /// `[num_patches, embed_dim]` learned position embedding.
    pub pos_embed: Arc<[f32]>,
    pub blocks: Vec<MoondreamVisionBlockWeights>,
    pub norm_gain: Arc<[f32]>,
    pub norm_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct MoondreamProjectionWeights {
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct MoondreamWeights {
    pub vision: MoondreamVisionWeights,
    pub projection: MoondreamProjectionWeights,
    pub text: MixFormerWeights,
}

#[derive(Debug, Clone)]
pub struct MoondreamConfig {
    pub vision: MoondreamVisionConfig,
    pub projection: MoondreamProjectionConfig,
    pub text: MixFormerConfig,
}

#[derive(Debug, Clone)]
pub struct MoondreamModel {
    pub config: MoondreamConfig,
    pub weights: MoondreamWeights,
}

impl MoondreamModel {
    /// Run the full multimodal forward pass.
    pub fn forward(
        &self,
        pixel_values: &LazyTensor,
        text_tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        assert_eq!(
            cfg.projection.out_dim, cfg.text.hidden_size,
            "projection out_dim must equal text hidden_size",
        );
        let text_len = text_tokens.len();
        assert!(text_len > 0, "text_tokens must be non-empty");

        // ---- Vision encoder ------------------------------------------------
        let vision_out = self.vision_encode(pixel_values)?;
        // (1, num_patches, embed_dim)

        // ---- Projection MLP ------------------------------------------------
        let projected = self.apply_projection(&vision_out)?;
        // (1, num_patches, projection.out_dim == text.hidden_size)

        // ---- Embed text tokens via MixFormer's token embedding -------------
        let mf_embed_lt = pixel_values.const_f32_like(
            Arc::clone(&self.weights.text.token_embedding),
            Shape::from_dims(&[cfg.text.vocab_size, cfg.text.hidden_size]),
        );
        let token_ids = pixel_values.const_u32_like(
            text_tokens.to_vec(),
            Shape::from_dims(&[text_len]),
        );
        let text_embeds = mf_embed_lt
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[1, text_len, cfg.text.hidden_size]))?;

        // ---- Concat and run MixFormer --------------------------------------
        let combined = projected.concat(&text_embeds, 1_usize)?;
        let mf = MixFormerModel {
            config: cfg.text.clone(),
            weights: self.weights.text.clone(),
        };
        mf.forward_embeds(&combined, 0)
    }

    fn vision_encode(&self, pixel_values: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config.vision;
        let weights = &self.weights.vision;
        let dims = pixel_values.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        let batch = dims[0];
        assert_eq!(batch, 1, "v1 supports batch == 1");
        assert_eq!(dims[1], cfg.num_channels);
        assert_eq!(dims[2], cfg.image_size);
        assert_eq!(dims[3], cfg.image_size);
        let np_side = cfg.image_size / cfg.patch_size;
        assert_eq!(np_side * np_side, cfg.num_patches,
            "num_patches must equal (image_size / patch_size)^2");

        // ---- Patchify into (b, num_patches, c*p*p) -----------------------
        // Eager: reshape (b, c, h/p, p, w/p, p) → permute (0, 2, 4, 1, 3, 5)
        //        → reshape (b, num_patches, c*p*p).
        let patches = pixel_values
            .reshape(Shape::from_dims(&[
                batch, cfg.num_channels,
                np_side, cfg.patch_size,
                np_side, cfg.patch_size,
            ]))?
            .permute([0, 2, 4, 1, 3, 5_usize])?
            .reshape(Shape::from_dims(&[
                batch, cfg.num_patches,
                cfg.patch_input_dim(),
            ]))?;

        // ---- Linear patch embedding -------------------------------------
        let patch_proj = weights.patch_embed.apply_linear(
            &patches, cfg.patch_input_dim(), cfg.embed_dim,
        );
        let patch_bias_t = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_embed_bias),
            Shape::from_dims(&[cfg.embed_dim]),
        );
        let patch_embeds = patch_proj.broadcast_add(&patch_bias_t)?;

        // ---- Add position embedding -------------------------------------
        let pos = pixel_values.const_f32_like(
            Arc::clone(&weights.pos_embed),
            Shape::from_dims(&[cfg.num_patches, cfg.embed_dim]),
        );
        let pos_bc = pos
            .reshape(Shape::from_dims(&[1, cfg.num_patches, cfg.embed_dim]))?
            .broadcast_to(Shape::from_dims(&[batch, cfg.num_patches, cfg.embed_dim]))?;
        let mut h = patch_embeds.add(&pos_bc)?;

        // ---- Encoder blocks ----------------------------------------------
        for block in &weights.blocks {
            h = self.apply_block(&h, block)?;
        }

        // ---- Final LayerNorm ---------------------------------------------
        Ok(crate::lazy::apply_affine_layer_norm_pub(
            &h, &weights.norm_gain, &weights.norm_bias,
            cfg.embed_dim, cfg.layer_norm_eps,
        ))
    }

    fn apply_block(
        &self,
        x: &LazyTensor,
        block: &MoondreamVisionBlockWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config.vision;
        let dims = x.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.embed_dim;
        let n_heads = cfg.num_heads;
        let head_dim = cfg.embed_dim / cfg.num_heads;

        // Pre-LN before attention.
        let x_norm = crate::lazy::apply_affine_layer_norm_pub(
            x, &block.norm1_gain, &block.norm1_bias, h, cfg.layer_norm_eps,
        );

        // Fused Wqkv: hidden → 3 * hidden.
        let qkv_lin = block.qkv.apply_linear(&x_norm, h, 3 * h);
        let qkv_b_t = x.const_f32_like(
            Arc::clone(&block.qkv_bias),
            Shape::from_dims(&[3 * h]),
        );
        let qkv = qkv_lin.broadcast_add(&qkv_b_t)?;
        let q = qkv.slice(2_usize, 0, h)?;
        let k = qkv.slice(2_usize, h, h)?;
        let v = qkv.slice(2_usize, 2 * h, h)?;

        let _ = (batch, seq);
        let q = q.split_heads(n_heads, head_dim)?;
        let k = k.split_heads(n_heads, head_dim)?;
        let v = v.split_heads(n_heads, head_dim)?;

        let k_t = k.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k_t)?.mul_scalar(scale);
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx.merge_heads()?;
        let proj = block.proj.apply_linear(&merged, h, h);
        let proj_b_t = x.const_f32_like(
            Arc::clone(&block.proj_bias),
            Shape::from_dims(&[h]),
        );
        let attn_out = proj.broadcast_add(&proj_b_t)?;
        let h1 = x.add(&attn_out)?;

        // Pre-LN before MLP.
        let h1_norm = crate::lazy::apply_affine_layer_norm_pub(
            &h1, &block.norm2_gain, &block.norm2_bias, h, cfg.layer_norm_eps,
        );
        let mlp_h = cfg.mlp_hidden;
        let fc1 = block.fc1.apply_linear(&h1_norm, h, mlp_h);
        let fc1_b_t = x.const_f32_like(
            Arc::clone(&block.fc1_bias),
            Shape::from_dims(&[mlp_h]),
        );
        let fc1 = fc1.broadcast_add(&fc1_b_t)?;
        let act = activate(&fc1, cfg.activation);
        let fc2 = block.fc2.apply_linear(&act, mlp_h, h);
        let fc2_b_t = x.const_f32_like(
            Arc::clone(&block.fc2_bias),
            Shape::from_dims(&[h]),
        );
        let mlp_out = fc2.broadcast_add(&fc2_b_t)?;
        h1.add(&mlp_out)
    }

    fn apply_projection(&self, vision_out: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config.projection;
        let weights = &self.weights.projection;
        let fc1 = weights.fc1.apply_linear(vision_out, cfg.in_dim, cfg.hidden_dim);
        let fc1_b_t = vision_out.const_f32_like(
            Arc::clone(&weights.fc1_bias),
            Shape::from_dims(&[cfg.hidden_dim]),
        );
        let fc1 = fc1.broadcast_add(&fc1_b_t)?;
        let act = activate(&fc1, cfg.activation);
        let fc2 = weights.fc2.apply_linear(&act, cfg.hidden_dim, cfg.out_dim);
        let fc2_b_t = vision_out.const_f32_like(
            Arc::clone(&weights.fc2_bias),
            Shape::from_dims(&[cfg.out_dim]),
        );
        fc2.broadcast_add(&fc2_b_t)
    }
}

fn activate(x: &LazyTensor, kind: MoondreamActivation) -> LazyTensor {
    match kind {
        MoondreamActivation::GeluPytorchTanh => x.gelu(),
        MoondreamActivation::Gelu => x.gelu_erf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy_mixformer::{MixFormerActivation, MixFormerLayerWeights};

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_vision_cfg() -> MoondreamVisionConfig {
        MoondreamVisionConfig {
            embed_dim: 8,
            num_blocks: 1,
            num_heads: 2,
            mlp_hidden: 16,
            num_patches: 4, // 2×2 = 4 patches
            patch_size: 4,
            num_channels: 3,
            image_size: 8,
            activation: MoondreamActivation::GeluPytorchTanh,
            layer_norm_eps: 1e-5,
        }
    }

    fn tiny_projection_cfg(text_hidden: usize) -> MoondreamProjectionConfig {
        MoondreamProjectionConfig {
            in_dim: 8,
            hidden_dim: 16,
            out_dim: text_hidden,
            activation: MoondreamActivation::GeluPytorchTanh,
        }
    }

    fn tiny_text_cfg() -> MixFormerConfig {
        MixFormerConfig {
            vocab_size: 16,
            hidden_size: 8,
            n_inner: Some(16),
            num_hidden_layers: 1,
            num_attention_heads: 2,
            rotary_dim: 2,
            layer_norm_eps: 1e-5,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            hidden_activation: MixFormerActivation::GeluPytorchTanh,
            tie_word_embeddings: false,
        }
    }

    fn tiny_vision_weights(cfg: &MoondreamVisionConfig) -> MoondreamVisionWeights {
        let mut s: u32 = 23232;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let pid = cfg.patch_input_dim();
        let h = cfg.embed_dim;
        let mlp_h = cfg.mlp_hidden;
        let patch_embed = WeightStorage::F32(vec_of(pid * h, &mut *nb));
        let patch_embed_bias = vec_of(h, &mut *nb);
        let pos_embed = vec_of(cfg.num_patches * h, &mut *nb);
        let blocks: Vec<MoondreamVisionBlockWeights> = (0..cfg.num_blocks).map(|_| MoondreamVisionBlockWeights {
            norm1_gain: Arc::from(vec![1.0_f32; h]),
            norm1_bias: Arc::from(vec![0.0_f32; h]),
            qkv: WeightStorage::F32(vec_of(h * (3 * h), &mut *nb)),
            qkv_bias: vec_of(3 * h, &mut *nb),
            proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            proj_bias: vec_of(h, &mut *nb),
            norm2_gain: Arc::from(vec![1.0_f32; h]),
            norm2_bias: Arc::from(vec![0.0_f32; h]),
            fc1: WeightStorage::F32(vec_of(h * mlp_h, &mut *nb)),
            fc1_bias: vec_of(mlp_h, &mut *nb),
            fc2: WeightStorage::F32(vec_of(mlp_h * h, &mut *nb)),
            fc2_bias: vec_of(h, &mut *nb),
        }).collect();
        MoondreamVisionWeights {
            patch_embed, patch_embed_bias, pos_embed, blocks,
            norm_gain: Arc::from(vec![1.0_f32; h]),
            norm_bias: Arc::from(vec![0.0_f32; h]),
        }
    }

    fn tiny_projection_weights(cfg: &MoondreamProjectionConfig) -> MoondreamProjectionWeights {
        let mut s: u32 = 34234;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        MoondreamProjectionWeights {
            fc1: WeightStorage::F32(vec_of(cfg.in_dim * cfg.hidden_dim, &mut *nb)),
            fc1_bias: vec_of(cfg.hidden_dim, &mut *nb),
            fc2: WeightStorage::F32(vec_of(cfg.hidden_dim * cfg.out_dim, &mut *nb)),
            fc2_bias: vec_of(cfg.out_dim, &mut *nb),
        }
    }

    fn tiny_text_weights(cfg: &MixFormerConfig) -> MixFormerWeights {
        let mut s: u32 = 56756;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let h = cfg.hidden_size;
        let inner = cfg.inner_dim();
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<MixFormerLayerWeights> = (0..cfg.num_hidden_layers).map(|_| MixFormerLayerWeights {
            ln_gain: Arc::from(vec![1.0_f32; h]),
            ln_bias: Arc::from(vec![0.0_f32; h]),
            wqkv: WeightStorage::F32(vec_of(h * (3 * h), &mut *nb)),
            wqkv_bias: vec_of(3 * h, &mut *nb),
            out_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            out_proj_bias: vec_of(h, &mut *nb),
            fc1: WeightStorage::F32(vec_of(h * inner, &mut *nb)),
            fc1_bias: vec_of(inner, &mut *nb),
            fc2: WeightStorage::F32(vec_of(inner * h, &mut *nb)),
            fc2_bias: vec_of(h, &mut *nb),
        }).collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let lm_head = Some(WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb)));
        let lm_head_bias = vec_of(cfg.vocab_size, &mut *nb);
        MixFormerWeights {
            token_embedding, layers,
            final_ln_gain, final_ln_bias,
            lm_head, lm_head_bias,
        }
    }

    fn tiny_image(cfg: &MoondreamVisionConfig) -> LazyTensor {
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
        let p_cfg = tiny_projection_cfg(t_cfg.hidden_size);
        let cfg = MoondreamConfig {
            vision: v_cfg.clone(),
            projection: p_cfg.clone(),
            text: t_cfg.clone(),
        };
        let weights = MoondreamWeights {
            vision: tiny_vision_weights(&v_cfg),
            projection: tiny_projection_weights(&p_cfg),
            text: tiny_text_weights(&t_cfg),
        };
        let model = MoondreamModel { config: cfg, weights };
        let img = tiny_image(&v_cfg);
        let toks = [1_u32, 2, 3];
        let logits = model.forward(&img, &toks).unwrap();
        let expected = v_cfg.num_patches + toks.len();
        assert_eq!(logits.shape().dims(), &[1, expected, t_cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// Linear patch-embedding is wired: zeroing it kills the
    /// image contribution; output at text positions must change.
    #[test]
    fn linear_patch_embed_is_wired() {
        let v_cfg = tiny_vision_cfg();
        let t_cfg = tiny_text_cfg();
        let p_cfg = tiny_projection_cfg(t_cfg.hidden_size);
        let cfg = MoondreamConfig {
            vision: v_cfg.clone(),
            projection: p_cfg.clone(),
            text: t_cfg.clone(),
        };
        let weights_a = MoondreamWeights {
            vision: tiny_vision_weights(&v_cfg),
            projection: tiny_projection_weights(&p_cfg),
            text: tiny_text_weights(&t_cfg),
        };
        let mut weights_b = weights_a.clone();
        let pid = v_cfg.patch_input_dim();
        weights_b.vision.patch_embed = WeightStorage::F32(
            Arc::from(vec![0.0_f32; pid * v_cfg.embed_dim])
        );
        weights_b.vision.patch_embed_bias = Arc::from(vec![0.0_f32; v_cfg.embed_dim]);
        let m_a = MoondreamModel { config: cfg.clone(), weights: weights_a };
        let m_b = MoondreamModel { config: cfg, weights: weights_b };
        let img_a = tiny_image(&v_cfg);
        let img_b = tiny_image(&v_cfg);
        let toks = [1_u32, 2, 3];
        let a = m_a.forward(&img_a, &toks).unwrap().realize_f32();
        let b = m_b.forward(&img_b, &toks).unwrap().realize_f32();
        let np = v_cfg.num_patches;
        let v = t_cfg.vocab_size;
        let start = np * v;
        let mut max_diff = 0.0_f32;
        for (x, y) in a[start..].iter().zip(b[start..].iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "patch-embed change must alter text-position logits, max_diff = {max_diff}");
    }
}

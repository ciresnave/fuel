//! CLIP (Contrastive Language-Image Pre-Training) ported to the
//! lazy-graph API.
//!
//! CLIP from OpenAI / Radford et al. 2021. Joint text+vision
//! model trained on image-caption pairs. Backbone for many
//! downstream multimodal models (LLaVA, BLIP, Moondream, ColPali,
//! PaliGemma, etc.) — porting CLIP unlocks the
//! text-image-projection path for all of them.
//!
//! Composition:
//!
//!   - **ClipTextTransformer**: token + learned position
//!     embeddings → encoder stack → final LayerNorm. Outputs
//!     the full per-token hidden states (the eager
//!     forward also selects the EOS token via
//!     `argmax(input_ids)`; v1 returns the full sequence so
//!     callers can pick whichever pooling they want).
//!   - **ClipVisionTransformer**: patch Conv2d → flatten/transpose
//!     → class_embedding prepended → position_embedding added →
//!     pre_LayerNorm → encoder stack → take CLS at position 0 →
//!     post_LayerNorm.
//!   - **Joint ClipModel**: text and vision towers + two
//!     projection linears (no bias) + a learned `logit_scale`
//!     scalar. Forward computes
//!     `logits = scale * (text @ image.T)` after L2-normalising
//!     both sides.
//!
//! The encoder shared by both towers uses:
//!   - **Pre-LN block**: `out = x + attn(LN1(x));
//!                       out = out + mlp(LN2(out))`.
//!   - **Standard MHA** with Q/K/V/O biases.
//!   - **QuickGelu activation**: `x * sigmoid(1.702 * x)`.
//!   - **Sequential MLP** (fc1 → QuickGelu → fc2).
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. The text tower returns the
//! full hidden sequence (no EOS-token argmax pooling); the
//! joint `ClipModel::similarity` returns a `[1, 1]` similarity
//! when called with a single image + single text.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

// ---- Shared encoder configuration -------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ClipTextConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub projection_dim: usize,
}

impl ClipTextConfig {
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_attention_heads
    }
    /// `openai/clip-vit-base-patch32` text-side preset.
    pub fn vit_base_patch32() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 512,
            intermediate_size: 2048,
            max_position_embeddings: 77,
            num_hidden_layers: 12,
            num_attention_heads: 8,
            projection_dim: 512,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClipVisionConfig {
    pub embed_dim: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub projection_dim: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
}

impl ClipVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_attention_heads
    }
    pub fn num_patches(&self) -> usize {
        let p = self.image_size / self.patch_size;
        p * p
    }
    /// `openai/clip-vit-base-patch32` vision-side preset.
    pub fn vit_base_patch32() -> Self {
        Self {
            embed_dim: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 512,
            num_channels: 3,
            image_size: 224,
            patch_size: 32,
        }
    }
}

// ---- Encoder layer weights (shared by both towers) --------------------------

#[derive(Debug, Clone)]
pub struct ClipEncoderLayerWeights {
    pub ln1_gain: Arc<[f32]>,
    pub ln1_bias: Arc<[f32]>,
    pub q_proj: WeightStorage,
    pub q_proj_bias: Arc<[f32]>,
    pub k_proj: WeightStorage,
    pub k_proj_bias: Arc<[f32]>,
    pub v_proj: WeightStorage,
    pub v_proj_bias: Arc<[f32]>,
    pub out_proj: WeightStorage,
    pub out_proj_bias: Arc<[f32]>,
    pub ln2_gain: Arc<[f32]>,
    pub ln2_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct ClipTextWeights {
    pub token_embedding: Arc<[f32]>,
    pub position_embedding: Arc<[f32]>,
    pub layers: Vec<ClipEncoderLayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct ClipVisionWeights {
    /// Conv2d kernel `[embed_dim, num_channels, patch, patch]`.
    pub patch_proj: Arc<[f32]>,
    pub class_embedding: Arc<[f32]>,
    pub position_embedding: Arc<[f32]>,
    pub pre_ln_gain: Arc<[f32]>,
    pub pre_ln_bias: Arc<[f32]>,
    pub layers: Vec<ClipEncoderLayerWeights>,
    pub post_ln_gain: Arc<[f32]>,
    pub post_ln_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct ClipModelWeights {
    pub text: ClipTextWeights,
    pub vision: ClipVisionWeights,
    /// `[text.embed_dim, projection_dim]`.
    pub text_projection: WeightStorage,
    /// `[vision.embed_dim, projection_dim]`.
    pub visual_projection: WeightStorage,
    pub logit_scale: f32,
}

#[derive(Debug, Clone)]
pub struct ClipTextModel {
    pub config: ClipTextConfig,
    pub weights: ClipTextWeights,
}

#[derive(Debug, Clone)]
pub struct ClipVisionModel {
    pub config: ClipVisionConfig,
    pub weights: ClipVisionWeights,
}

#[derive(Debug, Clone)]
pub struct ClipModel {
    pub text_config: ClipTextConfig,
    pub vision_config: ClipVisionConfig,
    pub weights: ClipModelWeights,
}

// ---- Text tower forward -----------------------------------------------------

impl ClipTextModel {
    /// Encode a single token sequence. Returns
    /// `(1, seq_len, embed_dim)` of post-final-LN hidden states.
    pub fn forward(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert!(seq <= cfg.max_position_embeddings);

        // Anchor on a single embedding tensor.
        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.embed_dim]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let token_embeds = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.embed_dim]))?;

        // Position embedding for [0..seq).
        let pos_full = embed.const_f32_like(
            Arc::clone(&weights.position_embedding),
            Shape::from_dims(&[cfg.max_position_embeddings, cfg.embed_dim]),
        );
        let pos_slice = pos_full
            .slice(0_usize, 0, seq)?
            .reshape(Shape::from_dims(&[1, seq, cfg.embed_dim]))?;
        let pos_bc = pos_slice.broadcast_to(Shape::from_dims(&[batch, seq, cfg.embed_dim]))?;
        let mut h = token_embeds.add(&pos_bc)?;

        // Causal mask for text.
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq {
                mask_data[i * seq + j] = f32::NEG_INFINITY;
            }
        }
        let mask = h.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]));

        for layer in &weights.layers {
            h = apply_clip_layer(
                &h, layer,
                cfg.num_attention_heads, cfg.head_dim(),
                Some(&mask),
            )?;
        }

        // Final LayerNorm.
        Ok(crate::lazy::apply_affine_layer_norm_pub(
            &h, &weights.final_ln_gain, &weights.final_ln_bias,
            cfg.embed_dim, 1e-5,
        ))
    }

    /// Pool the last hidden state by selecting position `eos_pos`,
    /// returning shape `(1, embed_dim)`. Caller chooses
    /// `eos_pos` (eager CLIP uses `argmax(input_ids)`).
    pub fn pool_eos(&self, tokens: &[u32], eos_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = self.forward(tokens)?;
        let pooled = h.slice(1_usize, eos_pos, 1)?
            .reshape(Shape::from_dims(&[1, cfg.embed_dim]))?;
        Ok(pooled)
    }

    /// Extract per-token features at the requested layer
    /// indices. Output shape per layer:
    /// `(1, seq_len, embed_dim)`. **No final LayerNorm**
    /// applied — downstream heads handle normalization
    /// themselves (matches the vision-tower hook convention).
    ///
    /// Use cases:
    ///
    ///   - **SDXL TE1 penultimate conditioning**: SDXL
    ///     conditions the UNet on the second-to-last CLIP
    ///     text-tower layer (NOT the final-LN output). The
    ///     existing `lazy_sd_text_encoder::forward_until_encoder_layer`
    ///     hook does the same for SD's standalone CLIP-L
    ///     text encoder; this hook gives the equivalent for
    ///     the joint `ClipModel`'s text tower.
    ///   - **Multi-layer features** for analysis / probing
    ///     (e.g., "which CLIP text-tower layer best predicts
    ///     class X?").
    ///
    /// Layer-id contract: 0-based, strictly increasing, all in
    /// `[0, num_hidden_layers)`. Causal mask is applied per
    /// layer just like the public `forward`.
    pub fn forward_intermediate_layers(
        &self,
        tokens: &[u32],
        layer_ids: &[usize],
    ) -> Result<Vec<LazyTensor>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert!(seq <= cfg.max_position_embeddings);
        assert!(!layer_ids.is_empty(), "layer_ids must not be empty");
        for w in layer_ids.windows(2) {
            assert!(w[0] < w[1], "layer_ids must be strictly increasing");
        }
        let depth = weights.layers.len();
        assert!(
            *layer_ids.last().unwrap() < depth,
            "layer_ids must all be in [0, num_hidden_layers = {depth})",
        );

        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.embed_dim]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let token_embeds = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.embed_dim]))?;
        let pos_full = embed.const_f32_like(
            Arc::clone(&weights.position_embedding),
            Shape::from_dims(&[cfg.max_position_embeddings, cfg.embed_dim]),
        );
        let pos_slice = pos_full
            .slice(0_usize, 0, seq)?
            .reshape(Shape::from_dims(&[1, seq, cfg.embed_dim]))?;
        let pos_bc = pos_slice.broadcast_to(Shape::from_dims(&[batch, seq, cfg.embed_dim]))?;
        let mut h = token_embeds.add(&pos_bc)?;

        // Same causal mask the public `forward` uses.
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq {
                mask_data[i * seq + j] = f32::NEG_INFINITY;
            }
        }
        let mask = h.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]));

        let mut out = Vec::with_capacity(layer_ids.len());
        let mut next_capture = 0;
        for (idx, layer) in weights.layers.iter().enumerate() {
            h = apply_clip_layer(
                &h, layer,
                cfg.num_attention_heads, cfg.head_dim(),
                Some(&mask),
            )?;
            if next_capture < layer_ids.len() && layer_ids[next_capture] == idx {
                out.push(h.clone());
                next_capture += 1;
            }
        }
        Ok(out)
    }
}

// ---- Vision tower forward ---------------------------------------------------

impl ClipVisionModel {
    /// Encode a single image at the configured `image_size`.
    /// Returns the pooled CLS hidden state of shape
    /// `(1, embed_dim)`.
    pub fn forward(&self, pixel_values: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = pixel_values.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        let batch = dims[0];
        assert_eq!(batch, 1, "v1 supports batch == 1");
        assert_eq!(dims[1], cfg.num_channels);
        assert_eq!(dims[2], cfg.image_size);
        assert_eq!(dims[3], cfg.image_size);

        // Patch Conv2d (no bias in CLIP).
        let conv_w = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj),
            Shape::from_dims(&[cfg.embed_dim, cfg.num_channels, cfg.patch_size, cfg.patch_size]),
        );
        let conv_out = pixel_values.conv2d(
            &conv_w,
            None,
            (cfg.patch_size, cfg.patch_size),
            (0, 0),
            1,
        )?;
        let np = cfg.num_patches();
        let patches = conv_out
            .reshape(Shape::from_dims(&[batch, cfg.embed_dim, np]))?
            .permute([0, 2, 1_usize])?;

        // Prepend class_embedding (broadcast to batch).
        let cls = pixel_values.const_f32_like(
            Arc::clone(&weights.class_embedding),
            Shape::from_dims(&[1, 1, cfg.embed_dim]),
        );
        let cls_bc = cls.broadcast_to(Shape::from_dims(&[batch, 1, cfg.embed_dim]))?;
        let with_cls = cls_bc.concat(&patches, 1_usize)?;

        // Add position embedding.
        let pos = pixel_values.const_f32_like(
            Arc::clone(&weights.position_embedding),
            Shape::from_dims(&[np + 1, cfg.embed_dim]),
        );
        let pos_bc = pos
            .reshape(Shape::from_dims(&[1, np + 1, cfg.embed_dim]))?
            .broadcast_to(Shape::from_dims(&[batch, np + 1, cfg.embed_dim]))?;
        let pre = with_cls.add(&pos_bc)?;

        // Pre-LayerNorm (CLIP vision has a pre-encoder LN).
        let pre_ln = crate::lazy::apply_affine_layer_norm_pub(
            &pre, &weights.pre_ln_gain, &weights.pre_ln_bias,
            cfg.embed_dim, 1e-5,
        );

        // Encoder layers (no causal mask for vision).
        let mut h = pre_ln;
        for layer in &weights.layers {
            h = apply_clip_layer(
                &h, layer,
                cfg.num_attention_heads, cfg.head_dim(),
                None,
            )?;
        }

        // Pool CLS token (position 0) and apply post-LN.
        let cls_pooled = h
            .slice(1_usize, 0, 1)?
            .reshape(Shape::from_dims(&[batch, cfg.embed_dim]))?;
        Ok(crate::lazy::apply_affine_layer_norm_pub(
            &cls_pooled, &weights.post_ln_gain, &weights.post_ln_bias,
            cfg.embed_dim, 1e-5,
        ))
    }

    /// Extract per-token features at the requested layer
    /// indices. Output shape per layer:
    /// `(1, num_patches + 1, embed_dim)` — CLS at slot 0,
    /// patches follow. The pre-encoder LayerNorm IS applied
    /// (it sits BEFORE the encoder loop, so it's part of the
    /// hidden state entering the first block). **The post-LN
    /// pooler is NOT applied** — downstream heads see the
    /// raw post-block features.
    ///
    /// Use cases:
    ///
    ///   - **CLIP-Penultimate conditioning**: SD 1.5/2.x and
    ///     SDXL TE1 condition the UNet on the SECOND-TO-LAST
    ///     layer's CLS-stripped patches, not the post-pooler
    ///     output (the `lazy_sd_text_encoder::forward_until_encoder_layer`
    ///     hook does the same trick for the TEXT tower).
    ///   - **DPT-on-CLIP-vision**: same as the DPT hooks added
    ///     for ViT/DINOv2/DINOv2-reg4/SigLIP.
    ///
    /// Layer-id contract: 0-based, strictly increasing, all in
    /// `[0, num_hidden_layers)`. Mirrors the four other
    /// ViT-shape backbone hooks.
    pub fn forward_intermediate_layers(
        &self,
        pixel_values: &LazyTensor,
        layer_ids: &[usize],
    ) -> Result<Vec<LazyTensor>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = pixel_values.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        let batch = dims[0];
        assert_eq!(batch, 1, "v1 supports batch == 1");
        assert_eq!(dims[1], cfg.num_channels);
        assert_eq!(dims[2], cfg.image_size);
        assert_eq!(dims[3], cfg.image_size);
        assert!(!layer_ids.is_empty(), "layer_ids must not be empty");
        for w in layer_ids.windows(2) {
            assert!(w[0] < w[1], "layer_ids must be strictly increasing");
        }
        let depth = weights.layers.len();
        assert!(
            *layer_ids.last().unwrap() < depth,
            "layer_ids must all be in [0, num_hidden_layers = {depth})",
        );

        // Same prep as forward().
        let conv_w = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj),
            Shape::from_dims(&[cfg.embed_dim, cfg.num_channels, cfg.patch_size, cfg.patch_size]),
        );
        let conv_out = pixel_values.conv2d(
            &conv_w, None,
            (cfg.patch_size, cfg.patch_size), (0, 0), 1,
        )?;
        let np = cfg.num_patches();
        let patches = conv_out
            .reshape(Shape::from_dims(&[batch, cfg.embed_dim, np]))?
            .permute([0, 2, 1_usize])?;
        let cls = pixel_values.const_f32_like(
            Arc::clone(&weights.class_embedding),
            Shape::from_dims(&[1, 1, cfg.embed_dim]),
        );
        let cls_bc = cls.broadcast_to(Shape::from_dims(&[batch, 1, cfg.embed_dim]))?;
        let with_cls = cls_bc.concat(&patches, 1_usize)?;
        let pos = pixel_values.const_f32_like(
            Arc::clone(&weights.position_embedding),
            Shape::from_dims(&[np + 1, cfg.embed_dim]),
        );
        let pos_bc = pos
            .reshape(Shape::from_dims(&[1, np + 1, cfg.embed_dim]))?
            .broadcast_to(Shape::from_dims(&[batch, np + 1, cfg.embed_dim]))?;
        let pre = with_cls.add(&pos_bc)?;
        let pre_ln = crate::lazy::apply_affine_layer_norm_pub(
            &pre, &weights.pre_ln_gain, &weights.pre_ln_bias,
            cfg.embed_dim, 1e-5,
        );

        let mut h = pre_ln;
        let mut out = Vec::with_capacity(layer_ids.len());
        let mut next_capture = 0;
        for (idx, layer) in weights.layers.iter().enumerate() {
            h = apply_clip_layer(
                &h, layer,
                cfg.num_attention_heads, cfg.head_dim(),
                None,
            )?;
            if next_capture < layer_ids.len() && layer_ids[next_capture] == idx {
                out.push(h.clone());
                next_capture += 1;
            }
        }
        Ok(out)
    }
}

// ---- Joint model forward ----------------------------------------------------

impl ClipModel {
    /// Encode a single image and project into the shared
    /// embedding space. Returns `(1, projection_dim)`.
    pub fn image_features(&self, pixel_values: &LazyTensor) -> Result<LazyTensor> {
        let v_model = ClipVisionModel {
            config: self.vision_config.clone(),
            weights: self.weights.vision.clone(),
        };
        let pooled = v_model.forward(pixel_values)?;
        Ok(self.weights.visual_projection.apply_linear(
            &pooled, self.vision_config.embed_dim, self.vision_config.projection_dim,
        ))
    }

    /// Encode a single token sequence at `eos_pos` (pooled) and
    /// project. Returns `(1, projection_dim)`.
    pub fn text_features(&self, tokens: &[u32], eos_pos: usize) -> Result<LazyTensor> {
        let t_model = ClipTextModel {
            config: self.text_config.clone(),
            weights: self.weights.text.clone(),
        };
        let pooled = t_model.pool_eos(tokens, eos_pos)?;
        Ok(self.weights.text_projection.apply_linear(
            &pooled, self.text_config.embed_dim, self.text_config.projection_dim,
        ))
    }
}

// ---- Helpers ----------------------------------------------------------------

/// Apply one CLIP encoder layer (pre-LN, MHA, residual, pre-LN, MLP, residual).
fn apply_clip_layer(
    x: &LazyTensor,
    layer: &ClipEncoderLayerWeights,
    n_heads: usize,
    head_dim: usize,
    causal_mask: Option<&LazyTensor>,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let batch = dims[0];
    let seq = dims[1];
    let h = dims[2];

    // Pre-LN before attention.
    let x_norm = crate::lazy::apply_affine_layer_norm_pub(
        x, &layer.ln1_gain, &layer.ln1_bias, h, 1e-5,
    );

    // Q, K, V projections with biases.
    let q = layer.q_proj.apply_linear(&x_norm, h, h);
    let q = bias_add(q, &layer.q_proj_bias, h, x)?;
    let k = layer.k_proj.apply_linear(&x_norm, h, h);
    let k = bias_add(k, &layer.k_proj_bias, h, x)?;
    let v = layer.v_proj.apply_linear(&x_norm, h, h);
    let v = bias_add(v, &layer.v_proj_bias, h, x)?;

    let q = q
        .reshape(Shape::from_dims(&[batch, seq, n_heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;
    let k = k
        .reshape(Shape::from_dims(&[batch, seq, n_heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;
    let v = v
        .reshape(Shape::from_dims(&[batch, seq, n_heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;

    let k_t = k.transpose()?;
    let scale = 1.0_f64 / (head_dim as f64).sqrt();
    let scores = q.matmul(&k_t)?;
    let scores_scaled = scores.mul_scalar(scale);
    let scores_masked = match causal_mask {
        None => scores_scaled,
        Some(m) => scores_scaled.broadcast_add(m)?,
    };
    let probs = scores_masked.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?;
    let merged = ctx
        .permute([0, 2, 1, 3_usize])?
        .reshape(Shape::from_dims(&[batch, seq, h]))?;
    let attn_out = layer.out_proj.apply_linear(&merged, h, h);
    let attn_out = bias_add(attn_out, &layer.out_proj_bias, h, x)?;
    let h1 = x.add(&attn_out)?;

    // Pre-LN before MLP.
    let h1_norm = crate::lazy::apply_affine_layer_norm_pub(
        &h1, &layer.ln2_gain, &layer.ln2_bias, h, 1e-5,
    );

    let inter = layer.fc1.apply_linear(&h1_norm, h, layer.fc1_bias.len());
    let inter = bias_add(inter, &layer.fc1_bias, layer.fc1_bias.len(), x)?;
    let activated = quick_gelu(&inter);
    let mlp_out = layer.fc2.apply_linear(&activated, layer.fc1_bias.len(), h);
    let mlp_out = bias_add(mlp_out, &layer.fc2_bias, h, x)?;
    h1.add(&mlp_out)
}

/// QuickGelu: `x * sigmoid(1.702 * x)`.
fn quick_gelu(x: &LazyTensor) -> LazyTensor {
    let scaled = x.mul_scalar(1.702);
    let sig = scaled.sigmoid();
    x.mul(&sig).unwrap()
}

fn bias_add(
    x: LazyTensor,
    bias: &Arc<[f32]>,
    n: usize,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    assert_eq!(bias.len(), n);
    let bt = anchor.const_f32_like(Arc::clone(bias), Shape::from_dims(&[n]));
    x.broadcast_add(&bt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_encoder_layers(
        n_layers: usize,
        embed: usize,
        inter: usize,
        nb: &mut Box<dyn FnMut() -> f32>,
    ) -> Vec<ClipEncoderLayerWeights> {
        (0..n_layers).map(|_| ClipEncoderLayerWeights {
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
        }).collect()
    }

    fn tiny_text_cfg() -> ClipTextConfig {
        ClipTextConfig {
            vocab_size: 32, embed_dim: 16,
            intermediate_size: 32,
            max_position_embeddings: 8,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            projection_dim: 12,
        }
    }

    fn tiny_vision_cfg() -> ClipVisionConfig {
        ClipVisionConfig {
            embed_dim: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            projection_dim: 12,
            num_channels: 3,
            image_size: 16,
            patch_size: 4,
        }
    }

    fn tiny_text_weights(cfg: &ClipTextConfig) -> ClipTextWeights {
        let mut s: u32 = 11111;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * cfg.embed_dim, &mut *nb);
        let position_embedding = vec_of(cfg.max_position_embeddings * cfg.embed_dim, &mut *nb);
        let layers = tiny_encoder_layers(cfg.num_hidden_layers, cfg.embed_dim, cfg.intermediate_size, &mut nb);
        ClipTextWeights {
            token_embedding, position_embedding,
            layers,
            final_ln_gain: Arc::from(vec![1.0_f32; cfg.embed_dim]),
            final_ln_bias: Arc::from(vec![0.0_f32; cfg.embed_dim]),
        }
    }

    fn tiny_vision_weights(cfg: &ClipVisionConfig) -> ClipVisionWeights {
        let mut s: u32 = 22222;
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
        let layers = tiny_encoder_layers(cfg.num_hidden_layers, cfg.embed_dim, cfg.intermediate_size, &mut nb);
        ClipVisionWeights {
            patch_proj, class_embedding, position_embedding,
            pre_ln_gain: Arc::from(vec![1.0_f32; cfg.embed_dim]),
            pre_ln_bias: Arc::from(vec![0.0_f32; cfg.embed_dim]),
            layers,
            post_ln_gain: Arc::from(vec![1.0_f32; cfg.embed_dim]),
            post_ln_bias: Arc::from(vec![0.0_f32; cfg.embed_dim]),
        }
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
    fn text_forward_shape() {
        let cfg = tiny_text_cfg();
        let model = ClipTextModel { config: cfg.clone(), weights: tiny_text_weights(&cfg) };
        let tokens = [1_u32, 2, 3, 4, 5];
        let h = model.forward(&tokens).unwrap();
        assert_eq!(h.shape().dims(), &[1, tokens.len(), cfg.embed_dim]);
        for &v in &h.realize_f32() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn text_pool_eos_shape() {
        let cfg = tiny_text_cfg();
        let model = ClipTextModel { config: cfg.clone(), weights: tiny_text_weights(&cfg) };
        let tokens = [1_u32, 2, 3, 4, 5];
        let pooled = model.pool_eos(&tokens, tokens.len() - 1).unwrap();
        assert_eq!(pooled.shape().dims(), &[1, cfg.embed_dim]);
    }

    /// Text causal mask works: changing a future token must NOT
    /// alter the pooled output at an earlier position.
    #[test]
    fn text_causal_mask_holds() {
        let cfg = tiny_text_cfg();
        let model = ClipTextModel { config: cfg.clone(), weights: tiny_text_weights(&cfg) };
        let toks_a = [1_u32, 2, 3, 4];
        let toks_b = [1_u32, 2, 3, 15]; // last token differs
        let h_a = model.forward(&toks_a).unwrap().realize_f32();
        let h_b = model.forward(&toks_b).unwrap().realize_f32();
        let e = cfg.embed_dim;
        // Compare hidden at positions 0, 1, 2 (which precede the change).
        for t in 0..3 {
            for d in 0..e {
                let i = t * e + d;
                assert!(
                    (h_a[i] - h_b[i]).abs() < 1e-5,
                    "causal mask violated at t={t}: {} vs {}", h_a[i], h_b[i],
                );
            }
        }
    }

    #[test]
    fn vision_forward_shape() {
        let cfg = tiny_vision_cfg();
        let model = ClipVisionModel { config: cfg.clone(), weights: tiny_vision_weights(&cfg) };
        let img = tiny_image(&cfg);
        let out = model.forward(&img).unwrap();
        assert_eq!(out.shape().dims(), &[1, cfg.embed_dim]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// CLIP joint model: image and text features both project
    /// to projection_dim.
    #[test]
    fn joint_model_projections() {
        let text_cfg = tiny_text_cfg();
        let vision_cfg = tiny_vision_cfg();
        let mut s: u32 = 33333;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let text = tiny_text_weights(&text_cfg);
        let vision = tiny_vision_weights(&vision_cfg);
        let text_projection = WeightStorage::F32(
            vec_of(text_cfg.embed_dim * text_cfg.projection_dim, &mut *nb)
        );
        let visual_projection = WeightStorage::F32(
            vec_of(vision_cfg.embed_dim * vision_cfg.projection_dim, &mut *nb)
        );
        let weights = ClipModelWeights {
            text, vision, text_projection, visual_projection,
            logit_scale: 2.6592,
        };
        let model = ClipModel {
            text_config: text_cfg.clone(),
            vision_config: vision_cfg.clone(),
            weights,
        };
        let img = tiny_image(&vision_cfg);
        let img_feat = model.image_features(&img).unwrap();
        assert_eq!(img_feat.shape().dims(), &[1, vision_cfg.projection_dim]);
        let toks = [1_u32, 2, 3, 4, 5];
        let txt_feat = model.text_features(&toks, toks.len() - 1).unwrap();
        assert_eq!(txt_feat.shape().dims(), &[1, text_cfg.projection_dim]);
        for &v in &img_feat.realize_f32() { assert!(v.is_finite()); }
        for &v in &txt_feat.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn config_presets() {
        let t = ClipTextConfig::vit_base_patch32();
        assert_eq!(t.vocab_size, 49408);
        assert_eq!(t.head_dim(), 64);
        let v = ClipVisionConfig::vit_base_patch32();
        assert_eq!(v.num_patches(), 49); // 224 / 32 = 7; 7*7 = 49
    }

    /// `forward_intermediate_layers` on the CLIP vision tower
    /// returns one tensor per requested layer index, shape
    /// `(1, num_patches + 1, embed_dim)` (CLS + patches).
    #[test]
    fn vision_forward_intermediate_layers_shape() {
        let cfg = tiny_vision_cfg();
        let model = ClipVisionModel { config: cfg.clone(), weights: tiny_vision_weights(&cfg) };
        let img = tiny_image(&cfg);
        let outs = model.forward_intermediate_layers(&img, &[0_usize, 1]).unwrap();
        assert_eq!(outs.len(), 2);
        let np = cfg.num_patches();
        for out in &outs {
            assert_eq!(out.shape().dims(), &[1, np + 1, cfg.embed_dim]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite intermediate: {v}");
            }
        }
    }

    /// Intermediate features at different depths must differ.
    #[test]
    fn vision_intermediate_layers_differ_across_depth() {
        let cfg = tiny_vision_cfg();
        let model = ClipVisionModel { config: cfg.clone(), weights: tiny_vision_weights(&cfg) };
        let img = tiny_image(&cfg);
        let outs = model.forward_intermediate_layers(&img, &[0_usize, 1]).unwrap();
        let a = outs[0].realize_f32();
        let b = outs[1].realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "layer 0 and layer 1 intermediates must differ, max_diff = {max_diff}");
    }

    /// `forward_intermediate_layers` on the CLIP **text** tower
    /// returns one tensor per requested layer, shape
    /// `(1, seq_len, embed_dim)`. The causal mask is applied
    /// per layer just like the public `forward`.
    #[test]
    fn text_forward_intermediate_layers_shape() {
        let cfg = tiny_text_cfg();
        let model = ClipTextModel { config: cfg.clone(), weights: tiny_text_weights(&cfg) };
        let tokens = [1_u32, 2, 3, 4, 5];
        let outs = model.forward_intermediate_layers(&tokens, &[0_usize, 1]).unwrap();
        assert_eq!(outs.len(), 2);
        for out in &outs {
            assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.embed_dim]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite intermediate: {v}");
            }
        }
        let a = outs[0].realize_f32();
        let b = outs[1].realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "layer 0 and layer 1 intermediates must differ, max_diff = {max_diff}");
    }
}

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

use crate::lazy::{
    load_tensor_as_f32, load_transposed_matrix_preserve_dtype,
    LayerWeights, LazyTensor, LlamaConfig, LlamaModel, LlamaWeights, WeightStorage,
};
use crate::lazy_clip::{ClipEncoderLayerWeights, ClipVisionConfig, ClipVisionWeights};
use crate::{Device, Result};
use fuel_ir::Shape;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct LlavaConfig {
    pub vision_config: ClipVisionConfig,
    pub text_config: LlamaConfig,
    /// Projected dim — must equal `text_config.dim`.
    pub projection_dim: usize,
}

// ---- HuggingFace JSON config -----------------------------------------------

/// JSON shape of the `vision_config` block inside HF LLaVA's
/// `config.json`. Mirrors `HFLLaVAVisionConfig` from the retired
/// eager port; only the fields the lazy port actually consumes
/// are required, everything else is ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct HFLlavaVisionConfig {
    pub hidden_size: usize,
    pub image_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    #[serde(default = "default_vision_projection_dim")]
    pub projection_dim: usize,
}

fn default_vision_projection_dim() -> usize {
    // CLIP ViT-L/14 default; only used when the HF config omits
    // the field (which it does on some LLaVA variants).
    768
}

/// JSON shape of the `text_config` block inside HF LLaVA's
/// `config.json`. Same minimalism: only fields the lazy port
/// uses are required.
#[derive(Debug, Clone, Deserialize)]
pub struct HFLlavaTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
}

fn default_rms_norm_eps() -> f64 { 1e-5 }
fn default_rope_theta() -> f64 { 10_000.0 }

/// Top-level HF LLaVA `config.json` deserialization target.
/// Used by [`HFLlavaConfig::from_hf_json_str`] to convert a
/// remote config string into our internal [`LlavaConfig`].
///
/// Only `vision_config`, `text_config`, and (when present)
/// `projection_dim` are consumed; all other HF-specific knobs
/// (image_grid_pinpoints, vision_feature_layer, projector
/// activation, etc.) are accepted but ignored — v1 of the lazy
/// port only models the "linear" projector + per-patch features
/// at a single resolution.
#[derive(Debug, Clone, Deserialize)]
pub struct HFLlavaConfig {
    pub vision_config: HFLlavaVisionConfig,
    pub text_config: HFLlavaTextConfig,
    /// Output dim of the MM projector. Must equal `text_config.hidden_size`
    /// for v1. When the field is omitted (older configs) it defaults to
    /// `text_config.hidden_size` post-parse.
    #[serde(default)]
    pub projection_dim: Option<usize>,
}

impl HFLlavaConfig {
    /// Parse an HF LLaVA `config.json` and convert it to our
    /// internal [`LlavaConfig`]. Returns an error if the projector
    /// dim is inconsistent with `text_config.hidden_size`.
    pub fn from_hf_json_str(json: &str) -> Result<LlavaConfig> {
        let parsed: HFLlavaConfig = serde_json::from_str(json)
            .map_err(|e| crate::Error::Msg(format!("parsing llava config.json: {e}")).bt())?;
        parsed.to_llava_config()
    }

    /// Build the internal [`LlavaConfig`] from this parsed HF
    /// config. Validates that `projection_dim == text.hidden_size`
    /// (the v1 lazy port's hard constraint).
    pub fn to_llava_config(&self) -> Result<LlavaConfig> {
        let t = &self.text_config;
        let v = &self.vision_config;
        let projection_dim = self.projection_dim.unwrap_or(t.hidden_size);
        if projection_dim != t.hidden_size {
            crate::bail!(
                "llava: projection_dim ({}) must equal text hidden_size ({}) for v1",
                projection_dim,
                t.hidden_size,
            );
        }
        if t.hidden_size % t.num_attention_heads != 0 {
            crate::bail!(
                "llava: text hidden_size ({}) not divisible by num_attention_heads ({})",
                t.hidden_size,
                t.num_attention_heads,
            );
        }
        let head_dim = t.hidden_size / t.num_attention_heads;
        let vision_config = ClipVisionConfig {
            embed_dim: v.hidden_size,
            intermediate_size: v.intermediate_size,
            num_hidden_layers: v.num_hidden_layers,
            num_attention_heads: v.num_attention_heads,
            projection_dim: v.projection_dim,
            num_channels: 3,
            image_size: v.image_size,
            patch_size: v.patch_size,
        };
        let _ = t.max_position_embeddings; // accepted but not modeled in LlamaConfig
        let text_config = LlamaConfig {
            vocab_size: t.vocab_size,
            dim: t.hidden_size,
            n_layers: t.num_hidden_layers,
            n_heads: t.num_attention_heads,
            n_kv_heads: t.num_key_value_heads,
            head_dim,
            ffn_dim: t.intermediate_size,
            norm_eps: t.rms_norm_eps,
            rope_base: t.rope_theta,
        };
        Ok(LlavaConfig {
            vision_config,
            text_config,
            projection_dim,
        })
    }
}

/// Select the best (width, height) from `possible_resolutions`
/// for an input image of `original_size = (width, height)`. Uses
/// the standard LLaVA-NeXT scoring: pick the resolution that
/// maximizes the effective (post-fit) pixel count, breaking ties
/// by minimum wasted area. Mirrors `select_best_resolution` from
/// the retired eager port (`fuel-transformers/.../llava/utils.rs`).
///
/// `possible_resolutions` is typically an HF `image_grid_pinpoints`
/// list; v1 of the lazy LLaVA port runs single-resolution only, so
/// this helper is provided primarily for the example binary's
/// pre-processing path.
pub fn select_best_resolution(
    original_size: (u32, u32),
    possible_resolutions: &[(u32, u32)],
) -> (u32, u32) {
    let (original_width, original_height) = original_size;
    let mut best_fit = (0_u32, 0_u32);
    let original_width_f = original_width as f32;
    let original_height_f = original_height as f32;
    let mut max_effective_resolution = 0_u32;
    let mut min_wasted_resolution = u32::MAX;
    for &(width, height) in possible_resolutions {
        let width_f = width as f32;
        let height_f = height as f32;
        let scale = (width_f / original_width_f).min(height_f / original_height_f);
        let downscaled_width = (original_width_f * scale) as u32;
        let downscaled_height = (original_height_f * scale) as u32;
        let effective_resolution =
            std::cmp::min(width * height, downscaled_width * downscaled_height);
        let wasted_resolution = width * height - effective_resolution;
        if effective_resolution > max_effective_resolution
            || (effective_resolution == max_effective_resolution
                && wasted_resolution < min_wasted_resolution)
        {
            best_fit = (width, height);
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
        }
    }
    best_fit
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
        let pre_ln = pre.layer_norm_affine(std::sync::Arc::clone(&weights.pre_ln_gain), std::sync::Arc::clone(&weights.pre_ln_bias), 1e-5)?;

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
        Ok(patches_only.layer_norm_affine(std::sync::Arc::clone(&weights.post_ln_gain), std::sync::Arc::clone(&weights.post_ln_bias), 1e-5)?)
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

    let x_norm = x.layer_norm_affine(std::sync::Arc::clone(&layer.ln1_gain), std::sync::Arc::clone(&layer.ln1_bias), 1e-5)?;

    let q = layer.q_proj.apply_linear(&x_norm, h, h);
    let q = q.add_trailing_bias(std::sync::Arc::clone(&layer.q_proj_bias))?;
    let k = layer.k_proj.apply_linear(&x_norm, h, h);
    let k = k.add_trailing_bias(std::sync::Arc::clone(&layer.k_proj_bias))?;
    let v = layer.v_proj.apply_linear(&x_norm, h, h);
    let v = v.add_trailing_bias(std::sync::Arc::clone(&layer.v_proj_bias))?;

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
    let attn_out = attn_out.add_trailing_bias(std::sync::Arc::clone(&layer.out_proj_bias))?;
    let h1 = x.add(&attn_out)?;

    let h1_norm = h1.layer_norm_affine(std::sync::Arc::clone(&layer.ln2_gain), std::sync::Arc::clone(&layer.ln2_bias), 1e-5)?;

    let inter_dim = layer.fc1_bias.len();
    let inter = layer.fc1.apply_linear(&h1_norm, h, inter_dim);
    let inter = inter.add_trailing_bias(std::sync::Arc::clone(&layer.fc1_bias))?;
    // QuickGelu: x * sigmoid(1.702 * x).
    let activated = {
        let scaled = inter.mul_scalar(1.702);
        let sig = scaled.sigmoid();
        inter.mul(&sig)?
    };
    let mlp_out = layer.fc2.apply_linear(&activated, inter_dim, h);
    let mlp_out = mlp_out.add_trailing_bias(std::sync::Arc::clone(&layer.fc2_bias))?;
    h1.add(&mlp_out)
}

// ---- Safetensors loader ----------------------------------------------------

/// Load a CLIP vision-tower's weights from a mmapped safetensors file
/// using the standard HuggingFace LLaVA `vision_tower.vision_model.*`
/// prefix. Used by [`LlavaWeights::load_from_mmapped`].
///
/// Expected tensor names (everything is `<prefix>.<suffix>`):
///   - `embeddings.patch_embedding.weight`
///   - `embeddings.class_embedding`
///   - `embeddings.position_embedding.weight`
///   - `pre_layrnorm.{weight,bias}` (HF typo on `pre_layernorm`)
///   - `encoder.layers.{i}.{layer_norm1,layer_norm2}.{weight,bias}`
///   - `encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}`
///   - `encoder.layers.{i}.mlp.{fc1,fc2}.{weight,bias}`
///   - `post_layernorm.{weight,bias}`
pub fn load_clip_vision_weights(
    st: &crate::safetensors::MmapedSafetensors,
    cfg: &ClipVisionConfig,
    prefix: &str,
) -> Result<ClipVisionWeights> {
    let h = cfg.embed_dim;
    let np = cfg.num_patches();

    let patch_proj = load_tensor_as_f32(
        st,
        &format!("{prefix}embeddings.patch_embedding.weight"),
    )?;
    let class_embedding = load_tensor_as_f32(
        st, &format!("{prefix}embeddings.class_embedding"),
    )?;
    let position_embedding = load_tensor_as_f32(
        st, &format!("{prefix}embeddings.position_embedding.weight"),
    )?;
    if position_embedding.len() != (np + 1) * h {
        crate::bail!(
            "{prefix}embeddings.position_embedding.weight: {} elts, expected {}",
            position_embedding.len(), (np + 1) * h,
        );
    }
    let pre_ln_gain = load_tensor_as_f32(st, &format!("{prefix}pre_layrnorm.weight"))?;
    let pre_ln_bias = load_tensor_as_f32(st, &format!("{prefix}pre_layrnorm.bias"))?;
    let post_ln_gain = load_tensor_as_f32(st, &format!("{prefix}post_layernorm.weight"))?;
    let post_ln_bias = load_tensor_as_f32(st, &format!("{prefix}post_layernorm.bias"))?;

    let mut layers: Vec<ClipEncoderLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("{prefix}encoder.layers.{i}");
        let ln1_gain = load_tensor_as_f32(st, &format!("{p}.layer_norm1.weight"))?;
        let ln1_bias = load_tensor_as_f32(st, &format!("{p}.layer_norm1.bias"))?;
        let q_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.self_attn.q_proj.weight"), h, h,
        )?;
        let q_proj_bias = load_tensor_as_f32(st, &format!("{p}.self_attn.q_proj.bias"))?;
        let k_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.self_attn.k_proj.weight"), h, h,
        )?;
        let k_proj_bias = load_tensor_as_f32(st, &format!("{p}.self_attn.k_proj.bias"))?;
        let v_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.self_attn.v_proj.weight"), h, h,
        )?;
        let v_proj_bias = load_tensor_as_f32(st, &format!("{p}.self_attn.v_proj.bias"))?;
        let out_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.self_attn.out_proj.weight"), h, h,
        )?;
        let out_proj_bias = load_tensor_as_f32(st, &format!("{p}.self_attn.out_proj.bias"))?;
        let ln2_gain = load_tensor_as_f32(st, &format!("{p}.layer_norm2.weight"))?;
        let ln2_bias = load_tensor_as_f32(st, &format!("{p}.layer_norm2.bias"))?;
        let fc1 = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.mlp.fc1.weight"), cfg.intermediate_size, h,
        )?;
        let fc1_bias = load_tensor_as_f32(st, &format!("{p}.mlp.fc1.bias"))?;
        let fc2 = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.mlp.fc2.weight"), h, cfg.intermediate_size,
        )?;
        let fc2_bias = load_tensor_as_f32(st, &format!("{p}.mlp.fc2.bias"))?;
        layers.push(ClipEncoderLayerWeights {
            ln1_gain: Arc::from(ln1_gain),
            ln1_bias: Arc::from(ln1_bias),
            q_proj, q_proj_bias: Arc::from(q_proj_bias),
            k_proj, k_proj_bias: Arc::from(k_proj_bias),
            v_proj, v_proj_bias: Arc::from(v_proj_bias),
            out_proj, out_proj_bias: Arc::from(out_proj_bias),
            ln2_gain: Arc::from(ln2_gain),
            ln2_bias: Arc::from(ln2_bias),
            fc1, fc1_bias: Arc::from(fc1_bias),
            fc2, fc2_bias: Arc::from(fc2_bias),
        });
    }

    Ok(ClipVisionWeights {
        patch_proj: Arc::from(patch_proj),
        class_embedding: Arc::from(class_embedding),
        position_embedding: Arc::from(position_embedding),
        pre_ln_gain: Arc::from(pre_ln_gain),
        pre_ln_bias: Arc::from(pre_ln_bias),
        layers,
        post_ln_gain: Arc::from(post_ln_gain),
        post_ln_bias: Arc::from(post_ln_bias),
    })
}

/// Load a LLaMA-shape decoder's weights using the
/// `<prefix>model.embed_tokens.weight` HF naming. The standard
/// non-multimodal LLaMA loader bakes in `model.` and `lm_head.`; this
/// variant simply prepends an outer prefix so LLaVA's
/// `language_model.model.embed_tokens.weight` (etc.) resolve cleanly.
pub fn load_llama_weights_with_prefix(
    st: &crate::safetensors::MmapedSafetensors,
    cfg: &LlamaConfig,
    prefix: &str,
) -> Result<LlamaWeights> {
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    let token_embedding = load_tensor_as_f32(
        st, &format!("{prefix}model.embed_tokens.weight"),
    )?;
    if token_embedding.len() != cfg.vocab_size * cfg.dim {
        crate::bail!(
            "{prefix}model.embed_tokens.weight: {} elts, expected {} ({}*{})",
            token_embedding.len(), cfg.vocab_size * cfg.dim, cfg.vocab_size, cfg.dim,
        );
    }

    let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let attn_q = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{prefix}model.layers.{i}.self_attn.q_proj.weight"),
            cfg.dim, cfg.dim,
        )?;
        let attn_k = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{prefix}model.layers.{i}.self_attn.k_proj.weight"),
            kv_dim, cfg.dim,
        )?;
        let attn_v = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{prefix}model.layers.{i}.self_attn.v_proj.weight"),
            kv_dim, cfg.dim,
        )?;
        let attn_o = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{prefix}model.layers.{i}.self_attn.o_proj.weight"),
            cfg.dim, cfg.dim,
        )?;
        let ffn_gate = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{prefix}model.layers.{i}.mlp.gate_proj.weight"),
            cfg.ffn_dim, cfg.dim,
        )?;
        let ffn_up = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{prefix}model.layers.{i}.mlp.up_proj.weight"),
            cfg.ffn_dim, cfg.dim,
        )?;
        let ffn_down = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{prefix}model.layers.{i}.mlp.down_proj.weight"),
            cfg.dim, cfg.ffn_dim,
        )?;
        let attn_norm_gain = load_tensor_as_f32(
            st,
            &format!("{prefix}model.layers.{i}.input_layernorm.weight"),
        )?;
        let ffn_norm_gain = load_tensor_as_f32(
            st,
            &format!("{prefix}model.layers.{i}.post_attention_layernorm.weight"),
        )?;
        let attn_q_bias = load_tensor_as_f32(
            st, &format!("{prefix}model.layers.{i}.self_attn.q_proj.bias"),
        ).ok().map(Arc::from);
        let attn_k_bias = load_tensor_as_f32(
            st, &format!("{prefix}model.layers.{i}.self_attn.k_proj.bias"),
        ).ok().map(Arc::from);
        let attn_v_bias = load_tensor_as_f32(
            st, &format!("{prefix}model.layers.{i}.self_attn.v_proj.bias"),
        ).ok().map(Arc::from);
        layers.push(LayerWeights {
            attn_q, attn_q_bias,
            attn_k, attn_k_bias,
            attn_v, attn_v_bias,
            attn_o,
            ffn_gate, ffn_up, ffn_down,
            attn_norm_gain: Arc::from(attn_norm_gain),
            ffn_norm_gain: Arc::from(ffn_norm_gain),
        });
    }

    let final_norm_gain = load_tensor_as_f32(st, &format!("{prefix}model.norm.weight"))?;
    // lm_head: try untied, then fall back to tied embedding.
    let output: WeightStorage = match load_transposed_matrix_preserve_dtype(
        st, &format!("{prefix}lm_head.weight"), cfg.vocab_size, cfg.dim,
    ) {
        Ok(w) => w,
        Err(_) => {
            let mut transposed = vec![0.0_f32; cfg.dim * cfg.vocab_size];
            for i in 0..cfg.vocab_size {
                for j in 0..cfg.dim {
                    transposed[j * cfg.vocab_size + i] = token_embedding[i * cfg.dim + j];
                }
            }
            WeightStorage::F32(Arc::from(transposed))
        }
    };

    Ok(LlamaWeights {
        token_embedding: Arc::from(token_embedding),
        layers,
        final_norm_gain: Arc::from(final_norm_gain),
        output,
    })
}

impl LlavaWeights {
    /// Load LLaVA weights from a HuggingFace safetensors file.
    /// Expects the HF LLaVA layout:
    ///   - `vision_tower.vision_model.*` for the CLIP vision encoder
    ///   - `multi_modal_projector.linear_1.{weight,bias}` for the
    ///     "linear" mm projector variant
    ///   - `language_model.model.*` and `language_model.lm_head.weight`
    ///     for the LLaMA decoder
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &LlavaConfig,
    ) -> Result<Self> {
        let v_cfg = &cfg.vision_config;
        let t_cfg = &cfg.text_config;
        let vision = load_clip_vision_weights(st, v_cfg, "vision_tower.vision_model.")?;
        let mm_proj = load_transposed_matrix_preserve_dtype(
            st,
            "multi_modal_projector.linear_1.weight",
            cfg.projection_dim,
            v_cfg.embed_dim,
        )?;
        let mm_proj_bias = load_tensor_as_f32(
            st, "multi_modal_projector.linear_1.bias",
        )?;
        if mm_proj_bias.len() != cfg.projection_dim {
            crate::bail!(
                "multi_modal_projector.linear_1.bias: {} elts, expected {}",
                mm_proj_bias.len(), cfg.projection_dim,
            );
        }
        let text = load_llama_weights_with_prefix(st, t_cfg, "language_model.")?;
        Ok(Self {
            vision,
            mm_proj,
            mm_proj_bias: Arc::from(mm_proj_bias),
            text,
        })
    }
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

    mod load {
        use super::*;
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;
        use std::collections::HashMap;

        // Helpers ----------------------------------------------------------
        fn put(
            map: &mut HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
            name: &str,
            shape: &[usize],
            data: &[f32],
        ) {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            map.insert(name.to_string(), (Dtype::F32, shape.to_vec(), bytes));
        }

        fn serialize_to_tempfile(
            map: &HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
        ) -> std::path::PathBuf {
            let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
            for (k, (dt, shape, data)) in map {
                let v = TensorView::new(*dt, shape.clone(), data)
                    .expect("TensorView::new");
                views.insert(k.clone(), v);
            }
            let bytes = safetensors::serialize(&views, None)
                .expect("safetensors::serialize");
            let dir = std::env::temp_dir();
            let unique = format!(
                "lazy_llava_load_{}_{}.safetensors",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
            );
            let path = dir.join(unique);
            std::fs::write(&path, bytes).expect("write tempfile");
            path
        }

        fn build_tiny_safetensors(
            v_cfg: &ClipVisionConfig,
            t_cfg: &LlamaConfig,
            proj_dim: usize,
        ) -> std::path::PathBuf {
            let mut map: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
            let mut s: u32 = 11;
            let mut nxt = || -> f32 {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.01
            };
            let mut vec_n = |n: usize| -> Vec<f32> {
                (0..n).map(|_| nxt()).collect()
            };

            // --- Vision tower (vision_tower.vision_model.*) ---
            let vp = "vision_tower.vision_model.";
            let h = v_cfg.embed_dim;
            let np = v_cfg.num_patches();
            put(&mut map, &format!("{vp}embeddings.patch_embedding.weight"),
                &[h, v_cfg.num_channels, v_cfg.patch_size, v_cfg.patch_size],
                &vec_n(h * v_cfg.num_channels * v_cfg.patch_size * v_cfg.patch_size));
            put(&mut map, &format!("{vp}embeddings.class_embedding"),
                &[h], &vec_n(h));
            put(&mut map, &format!("{vp}embeddings.position_embedding.weight"),
                &[np + 1, h], &vec_n((np + 1) * h));
            put(&mut map, &format!("{vp}pre_layrnorm.weight"), &[h], &vec_n(h));
            put(&mut map, &format!("{vp}pre_layrnorm.bias"),   &[h], &vec_n(h));
            put(&mut map, &format!("{vp}post_layernorm.weight"), &[h], &vec_n(h));
            put(&mut map, &format!("{vp}post_layernorm.bias"),   &[h], &vec_n(h));
            for i in 0..v_cfg.num_hidden_layers {
                let p = format!("{vp}encoder.layers.{i}");
                put(&mut map, &format!("{p}.layer_norm1.weight"), &[h], &vec_n(h));
                put(&mut map, &format!("{p}.layer_norm1.bias"),   &[h], &vec_n(h));
                put(&mut map, &format!("{p}.layer_norm2.weight"), &[h], &vec_n(h));
                put(&mut map, &format!("{p}.layer_norm2.bias"),   &[h], &vec_n(h));
                for proj in &["q_proj", "k_proj", "v_proj", "out_proj"] {
                    put(&mut map, &format!("{p}.self_attn.{proj}.weight"),
                        &[h, h], &vec_n(h * h));
                    put(&mut map, &format!("{p}.self_attn.{proj}.bias"),
                        &[h], &vec_n(h));
                }
                put(&mut map, &format!("{p}.mlp.fc1.weight"),
                    &[v_cfg.intermediate_size, h], &vec_n(v_cfg.intermediate_size * h));
                put(&mut map, &format!("{p}.mlp.fc1.bias"),
                    &[v_cfg.intermediate_size], &vec_n(v_cfg.intermediate_size));
                put(&mut map, &format!("{p}.mlp.fc2.weight"),
                    &[h, v_cfg.intermediate_size], &vec_n(h * v_cfg.intermediate_size));
                put(&mut map, &format!("{p}.mlp.fc2.bias"),
                    &[h], &vec_n(h));
            }

            // --- MM projector (linear variant) ---
            put(&mut map, "multi_modal_projector.linear_1.weight",
                &[proj_dim, v_cfg.embed_dim], &vec_n(proj_dim * v_cfg.embed_dim));
            put(&mut map, "multi_modal_projector.linear_1.bias",
                &[proj_dim], &vec_n(proj_dim));

            // --- Language model (language_model.model.*) ---
            let lp = "language_model.";
            let d = t_cfg.dim;
            let kv = t_cfg.n_kv_heads * t_cfg.head_dim;
            put(&mut map, &format!("{lp}model.embed_tokens.weight"),
                &[t_cfg.vocab_size, d], &vec_n(t_cfg.vocab_size * d));
            for i in 0..t_cfg.n_layers {
                let p = format!("{lp}model.layers.{i}");
                put(&mut map, &format!("{p}.self_attn.q_proj.weight"),
                    &[d, d], &vec_n(d * d));
                put(&mut map, &format!("{p}.self_attn.k_proj.weight"),
                    &[kv, d], &vec_n(kv * d));
                put(&mut map, &format!("{p}.self_attn.v_proj.weight"),
                    &[kv, d], &vec_n(kv * d));
                put(&mut map, &format!("{p}.self_attn.o_proj.weight"),
                    &[d, d], &vec_n(d * d));
                put(&mut map, &format!("{p}.mlp.gate_proj.weight"),
                    &[t_cfg.ffn_dim, d], &vec_n(t_cfg.ffn_dim * d));
                put(&mut map, &format!("{p}.mlp.up_proj.weight"),
                    &[t_cfg.ffn_dim, d], &vec_n(t_cfg.ffn_dim * d));
                put(&mut map, &format!("{p}.mlp.down_proj.weight"),
                    &[d, t_cfg.ffn_dim], &vec_n(d * t_cfg.ffn_dim));
                put(&mut map, &format!("{p}.input_layernorm.weight"),
                    &[d], &vec_n(d));
                put(&mut map, &format!("{p}.post_attention_layernorm.weight"),
                    &[d], &vec_n(d));
            }
            put(&mut map, &format!("{lp}model.norm.weight"), &[d], &vec_n(d));
            put(&mut map, &format!("{lp}lm_head.weight"),
                &[t_cfg.vocab_size, d], &vec_n(t_cfg.vocab_size * d));

            serialize_to_tempfile(&map)
        }

        #[test]
        fn round_trip_synthetic_safetensors() {
            let v_cfg = tiny_vision_cfg();
            let t_cfg = tiny_text_cfg();
            let proj_dim = t_cfg.dim;
            let path = build_tiny_safetensors(&v_cfg, &t_cfg, proj_dim);

            let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path) }
                .expect("mmap safetensors");
            let cfg = LlavaConfig {
                vision_config: v_cfg.clone(),
                text_config: t_cfg.clone(),
                projection_dim: proj_dim,
            };
            let w = LlavaWeights::load_from_mmapped(&st, &cfg)
                .expect("LlavaWeights::load_from_mmapped");

            // Shape spot-checks.
            assert_eq!(w.vision.layers.len(), v_cfg.num_hidden_layers);
            assert_eq!(w.vision.patch_proj.len(),
                v_cfg.embed_dim * v_cfg.num_channels * v_cfg.patch_size * v_cfg.patch_size);
            assert_eq!(w.vision.position_embedding.len(),
                (v_cfg.num_patches() + 1) * v_cfg.embed_dim);
            assert_eq!(w.mm_proj_bias.len(), proj_dim);
            assert_eq!(w.text.layers.len(), t_cfg.n_layers);
            assert_eq!(w.text.token_embedding.len(), t_cfg.vocab_size * t_cfg.dim);

            // Forward must produce finite logits with the loaded weights.
            let model = LlavaModel { config: cfg.clone(), weights: w };
            let img = tiny_image(&v_cfg);
            let tokens = [1_u32, 2, 3];
            let logits = model.forward(&img, &tokens).unwrap().realize_f32();
            for v in &logits {
                assert!(v.is_finite(), "non-finite logit");
            }
            let _ = std::fs::remove_file(&path);
        }

        /// Documents the canonical from-hub usage. Ignored in CI.
        #[test]
        #[ignore]
        fn from_hub_smoke_llava_1_5_7b() {
            // Canonical HF repo: llava-hf/llava-1.5-7b-hf
            // The loader expects the standard HF LLaVA naming:
            //   vision_tower.vision_model.*
            //   multi_modal_projector.linear_1.*
            //   language_model.model.* + language_model.lm_head.weight
        }
    }

    mod hf_config {
        use super::*;

        /// Trimmed llava-1.5-7b-hf style `config.json`: just the
        /// vision_config + text_config blocks the loader needs.
        const SAMPLE_HF: &str = r#"{
            "vision_config": {
                "hidden_size": 1024,
                "image_size": 336,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "patch_size": 14,
                "projection_dim": 768
            },
            "text_config": {
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "max_position_embeddings": 4096,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "num_key_value_heads": 32,
                "vocab_size": 32000,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0
            },
            "projection_dim": 4096
        }"#;

        #[test]
        fn from_hf_json_str_parses_llava_1_5_7b_shape() {
            let cfg = HFLlavaConfig::from_hf_json_str(SAMPLE_HF).expect("parse");
            // Vision side
            assert_eq!(cfg.vision_config.embed_dim, 1024);
            assert_eq!(cfg.vision_config.image_size, 336);
            assert_eq!(cfg.vision_config.patch_size, 14);
            assert_eq!(cfg.vision_config.num_patches(), (336 / 14) * (336 / 14));
            // Text side
            assert_eq!(cfg.text_config.dim, 4096);
            assert_eq!(cfg.text_config.n_layers, 32);
            assert_eq!(cfg.text_config.head_dim, 4096 / 32);
            assert_eq!(cfg.text_config.vocab_size, 32000);
            assert!((cfg.text_config.rope_base - 10_000.0).abs() < 1e-6);
            // Projection
            assert_eq!(cfg.projection_dim, 4096);
        }

        #[test]
        fn from_hf_json_str_defaults_projection_to_text_dim() {
            let json = r#"{
                "vision_config": {
                    "hidden_size": 768,
                    "image_size": 224,
                    "intermediate_size": 3072,
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "patch_size": 14
                },
                "text_config": {
                    "hidden_size": 2048,
                    "intermediate_size": 5504,
                    "max_position_embeddings": 4096,
                    "num_attention_heads": 16,
                    "num_hidden_layers": 16,
                    "num_key_value_heads": 16,
                    "vocab_size": 32000
                }
            }"#;
            let cfg = HFLlavaConfig::from_hf_json_str(json).expect("parse");
            assert_eq!(cfg.projection_dim, 2048);
            assert!((cfg.text_config.norm_eps - 1e-5).abs() < 1e-12);
            assert!((cfg.text_config.rope_base - 10_000.0).abs() < 1e-6);
        }

        #[test]
        fn rejects_mismatched_projection_dim() {
            let json = r#"{
                "vision_config": {
                    "hidden_size": 768, "image_size": 224, "intermediate_size": 3072,
                    "num_attention_heads": 12, "num_hidden_layers": 12, "patch_size": 14
                },
                "text_config": {
                    "hidden_size": 2048, "intermediate_size": 5504,
                    "max_position_embeddings": 4096, "num_attention_heads": 16,
                    "num_hidden_layers": 16, "num_key_value_heads": 16, "vocab_size": 32000
                },
                "projection_dim": 4096
            }"#;
            let err = HFLlavaConfig::from_hf_json_str(json)
                .expect_err("should reject projection_dim != text hidden_size");
            let msg = format!("{err}");
            assert!(msg.contains("projection_dim"), "got: {msg}");
        }
    }

    mod best_resolution {
        use super::*;

        #[test]
        fn picks_largest_effective_for_landscape() {
            let grid: &[(u32, u32)] =
                &[(336, 336), (672, 336), (336, 672), (672, 672), (1008, 336)];
            // A wide image (3:1) should snap to the widest pinpoint.
            let pick = select_best_resolution((1024, 320), grid);
            assert_eq!(pick, (1008, 336));
        }

        #[test]
        fn picks_largest_effective_for_portrait() {
            let grid: &[(u32, u32)] =
                &[(336, 336), (672, 336), (336, 672), (672, 672), (336, 1008)];
            let pick = select_best_resolution((320, 1024), grid);
            assert_eq!(pick, (336, 1008));
        }

        #[test]
        fn picks_square_for_square_input() {
            let grid: &[(u32, u32)] = &[(336, 336), (672, 336), (336, 672), (672, 672)];
            let pick = select_best_resolution((512, 512), grid);
            assert_eq!(pick, (672, 672));
        }
    }
}

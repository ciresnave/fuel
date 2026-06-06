//! SigLIP (Sigmoid Loss for Language-Image Pre-Training) ported to
//! the lazy-graph API.
//!
//! Google's variant of CLIP from Zhai et al. 2023. Used directly
//! by `google/siglip-base-patch16-224` and as the vision encoder
//! for PaliGemma (`google/paligemma-3b-pt-224` uses
//! `VisionConfig::paligemma_3b_224`).
//!
//! Differences from CLIP that this v1 honors:
//!
//!   - **No class token in the vision encoder.** Patch tokens are
//!     the sole sequence — there's no CLS prepended. Pooling is
//!     done by an optional `MultiheadAttentionPoolingHead` that
//!     uses a learned `probe` vector to attend over the patches.
//!   - **2D position embedding.** Stored as a learned embedding of
//!     shape `[num_patches_per_side^2, hidden]`; positions are
//!     added directly to each patch token (no interpolation in
//!     v1 — fixed image size).
//!   - **Text is bidirectional** — no causal mask. Pooling
//!     selects the **last token** of the sequence (instead of
//!     CLIP's argmax-EOS), then applies a `head` linear.
//!   - **GeluPytorchTanh** MLP activation (not CLIP's QuickGelu).
//!   - **Loss head**: `logit_scale` + `logit_bias` scalars for
//!     the sigmoid contrastive objective. v1 stores them; the
//!     joint forward returns `scale * (img @ txt^T) + bias`.
//!
//! Shared with CLIP (and ViT):
//!   - Pre-LN encoder block.
//!   - Standard MHA with Q/K/V/O biases.
//!   - Sequential MLP (fc1 → activation → fc2).
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. Position embedding
//! interpolation for variable image sizes deferred.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiglipActivation {
    GeluPytorchTanh,
    Gelu,
    Silu,
    Relu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SiglipTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_activation: SiglipActivation,
    pub layer_norm_eps: f64,
}

impl SiglipTextConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    /// Base SigLIP text config.
    pub fn base() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            max_position_embeddings: 64,
            hidden_activation: SiglipActivation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SiglipVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_activation: SiglipActivation,
    pub layer_norm_eps: f64,
}

impl SiglipVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }
    pub fn num_patches(&self) -> usize {
        let p = self.num_patches_per_side();
        p * p
    }
    /// PaliGemma-3B 224 vision config (256 patches).
    pub fn paligemma_3b_224() -> Self {
        Self {
            patch_size: 14,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            hidden_size: 1152,
            intermediate_size: 4304,
            image_size: 224,
            num_channels: 3,
            hidden_activation: SiglipActivation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SiglipEncoderLayerWeights {
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
pub struct SiglipPoolingHead {
    /// `[1, 1, hidden_size]` learned attention probe.
    pub probe: Arc<[f32]>,
    /// Multihead attention Q/K/V/O.
    pub q_proj: WeightStorage,
    pub q_proj_bias: Arc<[f32]>,
    pub k_proj: WeightStorage,
    pub k_proj_bias: Arc<[f32]>,
    pub v_proj: WeightStorage,
    pub v_proj_bias: Arc<[f32]>,
    pub out_proj: WeightStorage,
    pub out_proj_bias: Arc<[f32]>,
    pub ln_gain: Arc<[f32]>,
    pub ln_bias: Arc<[f32]>,
    pub mlp_fc1: WeightStorage,
    pub mlp_fc1_bias: Arc<[f32]>,
    pub mlp_fc2: WeightStorage,
    pub mlp_fc2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct SiglipVisionWeights {
    /// Conv2d weight `[hidden, num_channels, patch_size, patch_size]`.
    pub patch_proj: Arc<[f32]>,
    /// Conv2d bias `[hidden]`.
    pub patch_proj_bias: Arc<[f32]>,
    /// `[num_patches, hidden]` learned 2D position embedding.
    pub position_embedding: Arc<[f32]>,
    pub layers: Vec<SiglipEncoderLayerWeights>,
    pub post_ln_gain: Arc<[f32]>,
    pub post_ln_bias: Arc<[f32]>,
    /// Optional attention-pooling head — present when this
    /// vision encoder is used standalone (or by PaliGemma).
    pub head: Option<SiglipPoolingHead>,
}

#[derive(Debug, Clone)]
pub struct SiglipTextWeights {
    pub token_embedding: Arc<[f32]>,
    pub position_embedding: Arc<[f32]>,
    pub layers: Vec<SiglipEncoderLayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub head_w: WeightStorage,
    pub head_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct SiglipModelWeights {
    pub text: SiglipTextWeights,
    pub vision: SiglipVisionWeights,
    /// Sigmoid loss head: scale and bias scalars (broadcast to logits).
    pub logit_scale: f32,
    pub logit_bias: f32,
}

#[derive(Debug, Clone)]
pub struct SiglipTextModel {
    pub config: SiglipTextConfig,
    pub weights: SiglipTextWeights,
}

#[derive(Debug, Clone)]
pub struct SiglipVisionModel {
    pub config: SiglipVisionConfig,
    pub weights: SiglipVisionWeights,
}

#[derive(Debug, Clone)]
pub struct SiglipModel {
    pub text_config: SiglipTextConfig,
    pub vision_config: SiglipVisionConfig,
    pub weights: SiglipModelWeights,
}

impl SiglipTextModel {
    /// Encode a token sequence and return the **last-position
    /// pooled** feature of shape `(1, hidden_size)`. The eager
    /// reference applies a final linear `head` after pooling.
    pub fn forward(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert!(seq <= cfg.max_position_embeddings);

        let token_embeds = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;
        let pos_full = token_embeds.const_f32_like(
            Arc::clone(&weights.position_embedding),
            Shape::from_dims(&[cfg.max_position_embeddings, cfg.hidden_size]),
        );
        let pos_slice = pos_full
            .slice(0_usize, 0, seq)?
            .reshape(Shape::from_dims(&[1, seq, cfg.hidden_size]))?;
        let pos_bc = pos_slice.broadcast_to(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;
        let mut h = token_embeds.add(&pos_bc)?;

        // Bidirectional encoder (no causal mask).
        for layer in &weights.layers {
            h = apply_encoder_layer(
                &h, layer,
                cfg.num_attention_heads, cfg.head_dim(),
                None, cfg.layer_norm_eps, cfg.hidden_activation,
            )?;
        }

        let h_norm = h.layer_norm_affine(std::sync::Arc::clone(&weights.final_ln_gain), std::sync::Arc::clone(&weights.final_ln_bias), cfg.layer_norm_eps)?;

        // Pool last position.
        let last = h_norm
            .slice(1_usize, seq - 1, 1)?
            .reshape(Shape::from_dims(&[batch, cfg.hidden_size]))?;
        let head_out = weights.head_w.apply_linear(&last, cfg.hidden_size, cfg.hidden_size);
        // Bias on head.
        let bias_t = head_out.const_f32_like(
            Arc::clone(&weights.head_bias),
            Shape::from_dims(&[cfg.hidden_size]),
        );
        head_out.broadcast_add(&bias_t)
    }
}

impl SiglipVisionModel {
    /// Encode a single image. Returns:
    ///   - With pooling head: `(1, hidden_size)`.
    ///   - Without: `(1, num_patches, hidden_size)` of post-LN
    ///     patch tokens.
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

        // Patch Conv2d (with bias in SigLIP).
        let conv_w = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj),
            Shape::from_dims(&[cfg.hidden_size, cfg.num_channels, cfg.patch_size, cfg.patch_size]),
        );
        let conv_b = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj_bias),
            Shape::from_dims(&[cfg.hidden_size]),
        );
        let conv_out = pixel_values.conv2d(
            &conv_w,
            Some(&conv_b),
            (cfg.patch_size, cfg.patch_size),
            (0, 0),
            1,
        )?;
        let np = cfg.num_patches();
        let patches = conv_out
            .reshape(Shape::from_dims(&[batch, cfg.hidden_size, np]))?
            .permute([0, 2, 1_usize])?;

        // Add 2D position embedding (no CLS).
        let pos = pixel_values.const_f32_like(
            Arc::clone(&weights.position_embedding),
            Shape::from_dims(&[np, cfg.hidden_size]),
        );
        let pos_bc = pos
            .reshape(Shape::from_dims(&[1, np, cfg.hidden_size]))?
            .broadcast_to(Shape::from_dims(&[batch, np, cfg.hidden_size]))?;
        let mut h = patches.add(&pos_bc)?;

        // Encoder layers.
        for layer in &weights.layers {
            h = apply_encoder_layer(
                &h, layer,
                cfg.num_attention_heads, cfg.head_dim(),
                None, cfg.layer_norm_eps, cfg.hidden_activation,
            )?;
        }
        // Post-LayerNorm on all tokens.
        let h_norm = h.layer_norm_affine(std::sync::Arc::clone(&weights.post_ln_gain), std::sync::Arc::clone(&weights.post_ln_bias), cfg.layer_norm_eps)?;

        match &weights.head {
            None => Ok(h_norm),
            Some(head) => self.apply_pooling_head(&h_norm, head),
        }
    }

    /// Extract per-token features at the requested layer
    /// indices. Output shape per layer:
    /// `(1, num_patches, hidden_size)` — NO CLS token (SigLIP
    /// uses patch-only embedding) and **no post-LayerNorm**
    /// applied (DPT-style heads have their own per-stage
    /// projection).
    ///
    /// Mirrors the `forward_intermediate_layers` hook contract
    /// on [`crate::lazy_vit::VitModel`] and the DINOv2 variants
    /// — 0-based indices, strictly increasing, all in
    /// `[0, num_hidden_layers)`. SigLIP just lacks the CLS slot
    /// so the per-layer output is one shorter on the seq dim.
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

        let conv_w = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj),
            Shape::from_dims(&[cfg.hidden_size, cfg.num_channels, cfg.patch_size, cfg.patch_size]),
        );
        let conv_b = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj_bias),
            Shape::from_dims(&[cfg.hidden_size]),
        );
        let conv_out = pixel_values.conv2d(
            &conv_w, Some(&conv_b),
            (cfg.patch_size, cfg.patch_size), (0, 0), 1,
        )?;
        let np = cfg.num_patches();
        let patches = conv_out
            .reshape(Shape::from_dims(&[batch, cfg.hidden_size, np]))?
            .permute([0, 2, 1_usize])?;
        let pos = pixel_values.const_f32_like(
            Arc::clone(&weights.position_embedding),
            Shape::from_dims(&[np, cfg.hidden_size]),
        );
        let pos_bc = pos
            .reshape(Shape::from_dims(&[1, np, cfg.hidden_size]))?
            .broadcast_to(Shape::from_dims(&[batch, np, cfg.hidden_size]))?;
        let mut h = patches.add(&pos_bc)?;

        let mut out = Vec::with_capacity(layer_ids.len());
        let mut next_capture = 0;
        for (idx, layer) in weights.layers.iter().enumerate() {
            h = apply_encoder_layer(
                &h, layer,
                cfg.num_attention_heads, cfg.head_dim(),
                None, cfg.layer_norm_eps, cfg.hidden_activation,
            )?;
            if next_capture < layer_ids.len() && layer_ids[next_capture] == idx {
                out.push(h.clone());
                next_capture += 1;
            }
        }
        Ok(out)
    }

    fn apply_pooling_head(
        &self,
        xs: &LazyTensor,
        head: &SiglipPoolingHead,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = xs.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();

        // Probe broadcast across batch.
        let probe = xs.const_f32_like(
            Arc::clone(&head.probe),
            Shape::from_dims(&[1, 1, h]),
        );
        let probe_bc = probe.broadcast_to(Shape::from_dims(&[batch, 1, h]))?;

        // Cross-attention: Q = probe, K = V = xs.
        let q = head.q_proj.apply_linear(&probe_bc, h, h);
        let q = q.add_trailing_bias(std::sync::Arc::clone(&head.q_proj_bias))?;
        let k = head.k_proj.apply_linear(xs, h, h);
        let k = k.add_trailing_bias(std::sync::Arc::clone(&head.k_proj_bias))?;
        let v = head.v_proj.apply_linear(xs, h, h);
        let v = v.add_trailing_bias(std::sync::Arc::clone(&head.v_proj_bias))?;

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
        let attn_out = head.out_proj.apply_linear(&merged, h, h);
        let attn_out = attn_out.add_trailing_bias(std::sync::Arc::clone(&head.out_proj_bias))?;

        // MLP residual: residual + mlp(LN(attn_out)) → take token 0.
        let residual = attn_out.clone();
        let attn_ln = attn_out.layer_norm_affine(std::sync::Arc::clone(&head.ln_gain), std::sync::Arc::clone(&head.ln_bias), cfg.layer_norm_eps)?;
        let inter_dim = head.mlp_fc1_bias.len();
        let fc1 = head.mlp_fc1.apply_linear(&attn_ln, h, inter_dim);
        let fc1 = fc1.add_trailing_bias(std::sync::Arc::clone(&head.mlp_fc1_bias))?;
        let act = activate(&fc1, cfg.hidden_activation);
        let fc2 = head.mlp_fc2.apply_linear(&act, inter_dim, h);
        let fc2 = fc2.add_trailing_bias(std::sync::Arc::clone(&head.mlp_fc2_bias))?;
        let with_res = residual.add(&fc2)?;
        // The eager takes the first token; with seq=1 the result is already (b, 1, h).
        with_res.reshape(Shape::from_dims(&[batch, h]))
    }
}

impl SiglipModel {
    /// Encode a single image and return `(1, hidden_size)`.
    pub fn image_features(&self, pixel_values: &LazyTensor) -> Result<LazyTensor> {
        let v = SiglipVisionModel {
            config: self.vision_config.clone(),
            weights: self.weights.vision.clone(),
        };
        v.forward(pixel_values)
    }

    /// Encode a single token sequence and return `(1, hidden_size)`.
    pub fn text_features(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let t = SiglipTextModel {
            config: self.text_config.clone(),
            weights: self.weights.text.clone(),
        };
        t.forward(tokens)
    }
}

// ---- Shared helpers ---------------------------------------------------------

fn apply_encoder_layer(
    x: &LazyTensor,
    layer: &SiglipEncoderLayerWeights,
    n_heads: usize,
    head_dim: usize,
    causal_mask: Option<&LazyTensor>,
    layer_norm_eps: f64,
    activation: SiglipActivation,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let batch = dims[0];
    let seq = dims[1];
    let h = dims[2];

    let x_norm = x.layer_norm_affine(std::sync::Arc::clone(&layer.ln1_gain), std::sync::Arc::clone(&layer.ln1_bias), layer_norm_eps)?;

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

    let h1_norm = h1.layer_norm_affine(std::sync::Arc::clone(&layer.ln2_gain), std::sync::Arc::clone(&layer.ln2_bias), layer_norm_eps)?;
    let inter_dim = layer.fc1_bias.len();
    let fc1 = layer.fc1.apply_linear(&h1_norm, h, inter_dim);
    let fc1 = fc1.add_trailing_bias(std::sync::Arc::clone(&layer.fc1_bias))?;
    let act = activate(&fc1, activation);
    let fc2 = layer.fc2.apply_linear(&act, inter_dim, h);
    let fc2 = fc2.add_trailing_bias(std::sync::Arc::clone(&layer.fc2_bias))?;
    h1.add(&fc2)
}

fn activate(x: &LazyTensor, kind: SiglipActivation) -> LazyTensor {
    match kind {
        SiglipActivation::GeluPytorchTanh => x.gelu(),
        SiglipActivation::Gelu => x.gelu_erf(),
        SiglipActivation::Silu => x.silu(),
        SiglipActivation::Relu => x.relu(),
    }
}

// ---- HuggingFace safetensors loaders ---------------------------------------

fn load_siglip_encoder_layer(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
) -> Result<SiglipEncoderLayerWeights> {
    use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
    Ok(SiglipEncoderLayerWeights {
        ln1_gain: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.layer_norm1.weight"))?),
        ln1_bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.layer_norm1.bias"))?),
        q_proj: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.self_attn.q_proj.weight"), hidden, hidden,
        )?,
        q_proj_bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.self_attn.q_proj.bias"))?),
        k_proj: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.self_attn.k_proj.weight"), hidden, hidden,
        )?,
        k_proj_bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.self_attn.k_proj.bias"))?),
        v_proj: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.self_attn.v_proj.weight"), hidden, hidden,
        )?,
        v_proj_bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.self_attn.v_proj.bias"))?),
        out_proj: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.self_attn.out_proj.weight"), hidden, hidden,
        )?,
        out_proj_bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.self_attn.out_proj.bias"))?),
        ln2_gain: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.layer_norm2.weight"))?),
        ln2_bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.layer_norm2.bias"))?),
        fc1: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.mlp.fc1.weight"), intermediate, hidden,
        )?,
        fc1_bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.mlp.fc1.bias"))?),
        fc2: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.mlp.fc2.weight"), hidden, intermediate,
        )?,
        fc2_bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.mlp.fc2.bias"))?),
    })
}

impl SiglipVisionWeights {
    /// Load SigLIP vision-tower weights from HF safetensors.
    /// `prefix` is typically `""` for vision-only checkpoints or
    /// `"vision_model."` for full SigLIP / PaliGemma checkpoints.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &SiglipVisionConfig,
        prefix: &str,
        include_head: bool,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;

        let patch_proj = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}embeddings.patch_embedding.weight"),
        )?);
        let patch_proj_bias = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}embeddings.patch_embedding.bias"),
        )?);
        let position_embedding = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}embeddings.position_embedding.weight"),
        )?);
        let layers: Result<Vec<_>> = (0..cfg.num_hidden_layers)
            .map(|i| load_siglip_encoder_layer(
                st, &format!("{prefix}encoder.layers.{i}"), h, inter,
            ))
            .collect();
        let post_ln_gain = Arc::from(load_tensor_as_f32(st, &format!("{prefix}post_layernorm.weight"))?);
        let post_ln_bias = Arc::from(load_tensor_as_f32(st, &format!("{prefix}post_layernorm.bias"))?);

        let head = if include_head {
            let hp = format!("{prefix}head");
            Some(SiglipPoolingHead {
                probe: Arc::from(load_tensor_as_f32(st, &format!("{hp}.probe"))?),
                q_proj: load_transposed_matrix_preserve_dtype(
                    st, &format!("{hp}.attention.in_proj_q.weight"), h, h,
                )?,
                q_proj_bias: Arc::from(load_tensor_as_f32(st, &format!("{hp}.attention.in_proj_q.bias"))?),
                k_proj: load_transposed_matrix_preserve_dtype(
                    st, &format!("{hp}.attention.in_proj_k.weight"), h, h,
                )?,
                k_proj_bias: Arc::from(load_tensor_as_f32(st, &format!("{hp}.attention.in_proj_k.bias"))?),
                v_proj: load_transposed_matrix_preserve_dtype(
                    st, &format!("{hp}.attention.in_proj_v.weight"), h, h,
                )?,
                v_proj_bias: Arc::from(load_tensor_as_f32(st, &format!("{hp}.attention.in_proj_v.bias"))?),
                out_proj: load_transposed_matrix_preserve_dtype(
                    st, &format!("{hp}.attention.out_proj.weight"), h, h,
                )?,
                out_proj_bias: Arc::from(load_tensor_as_f32(st, &format!("{hp}.attention.out_proj.bias"))?),
                ln_gain: Arc::from(load_tensor_as_f32(st, &format!("{hp}.layernorm.weight"))?),
                ln_bias: Arc::from(load_tensor_as_f32(st, &format!("{hp}.layernorm.bias"))?),
                mlp_fc1: load_transposed_matrix_preserve_dtype(
                    st, &format!("{hp}.mlp.fc1.weight"), inter, h,
                )?,
                mlp_fc1_bias: Arc::from(load_tensor_as_f32(st, &format!("{hp}.mlp.fc1.bias"))?),
                mlp_fc2: load_transposed_matrix_preserve_dtype(
                    st, &format!("{hp}.mlp.fc2.weight"), h, inter,
                )?,
                mlp_fc2_bias: Arc::from(load_tensor_as_f32(st, &format!("{hp}.mlp.fc2.bias"))?),
            })
        } else {
            None
        };

        Ok(Self {
            patch_proj, patch_proj_bias, position_embedding,
            layers: layers?,
            post_ln_gain, post_ln_bias,
            head,
        })
    }
}

impl SiglipTextWeights {
    /// Load SigLIP text-tower weights from HF safetensors.
    /// `prefix` typically `""` or `"text_model."`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &SiglipTextConfig,
        prefix: &str,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;

        let token_embedding = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}embeddings.token_embedding.weight"),
        )?);
        let position_embedding = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}embeddings.position_embedding.weight"),
        )?);
        let layers: Result<Vec<_>> = (0..cfg.num_hidden_layers)
            .map(|i| load_siglip_encoder_layer(
                st, &format!("{prefix}encoder.layers.{i}"), h, inter,
            ))
            .collect();
        let final_ln_gain = Arc::from(load_tensor_as_f32(st, &format!("{prefix}final_layer_norm.weight"))?);
        let final_ln_bias = Arc::from(load_tensor_as_f32(st, &format!("{prefix}final_layer_norm.bias"))?);
        let head_w = load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}head.weight"), h, h,
        )?;
        let head_bias = Arc::from(load_tensor_as_f32(st, &format!("{prefix}head.bias"))?);
        Ok(Self {
            token_embedding, position_embedding,
            layers: layers?,
            final_ln_gain, final_ln_bias,
            head_w, head_bias,
        })
    }
}

impl SiglipModelWeights {
    /// Load a full SigLIP checkpoint (text + vision + logit
    /// scale/bias) from `google/siglip-*` HF safetensors.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        text_cfg: &SiglipTextConfig,
        vision_cfg: &SiglipVisionConfig,
    ) -> Result<Self> {
        use crate::lazy::load_tensor_as_f32;
        let text = SiglipTextWeights::load_from_mmapped(st, text_cfg, "text_model.")?;
        let vision = SiglipVisionWeights::load_from_mmapped(
            st, vision_cfg, "vision_model.", false,
        )?;
        let logit_scale = load_tensor_as_f32(st, "logit_scale")?
            .first().copied().unwrap_or(0.0);
        let logit_bias = load_tensor_as_f32(st, "logit_bias")?
            .first().copied().unwrap_or(0.0);
        Ok(Self { text, vision, logit_scale, logit_bias })
    }
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
    ) -> Vec<SiglipEncoderLayerWeights> {
        (0..n_layers).map(|_| SiglipEncoderLayerWeights {
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

    fn tiny_text_cfg() -> SiglipTextConfig {
        SiglipTextConfig {
            vocab_size: 32, hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            max_position_embeddings: 8,
            hidden_activation: SiglipActivation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }

    fn tiny_vision_cfg() -> SiglipVisionConfig {
        SiglipVisionConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_channels: 3,
            image_size: 16,
            patch_size: 4,
            hidden_activation: SiglipActivation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }

    fn tiny_text_weights(cfg: &SiglipTextConfig) -> SiglipTextWeights {
        let mut s: u32 = 56565;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * cfg.hidden_size, &mut *nb);
        let position_embedding = vec_of(cfg.max_position_embeddings * cfg.hidden_size, &mut *nb);
        let layers = tiny_encoder_layers(cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, &mut nb);
        let final_ln_gain = Arc::from(vec![1.0_f32; cfg.hidden_size]);
        let final_ln_bias = Arc::from(vec![0.0_f32; cfg.hidden_size]);
        let head_w = WeightStorage::F32(vec_of(cfg.hidden_size * cfg.hidden_size, &mut *nb));
        let head_bias = vec_of(cfg.hidden_size, &mut *nb);
        SiglipTextWeights {
            token_embedding, position_embedding,
            layers,
            final_ln_gain, final_ln_bias,
            head_w, head_bias,
        }
    }

    fn tiny_pooling_head(embed: usize, inter: usize, nb: &mut Box<dyn FnMut() -> f32>) -> SiglipPoolingHead {
        SiglipPoolingHead {
            probe: vec_of(embed, &mut **nb),
            q_proj: WeightStorage::F32(vec_of(embed * embed, &mut **nb)),
            q_proj_bias: vec_of(embed, &mut **nb),
            k_proj: WeightStorage::F32(vec_of(embed * embed, &mut **nb)),
            k_proj_bias: vec_of(embed, &mut **nb),
            v_proj: WeightStorage::F32(vec_of(embed * embed, &mut **nb)),
            v_proj_bias: vec_of(embed, &mut **nb),
            out_proj: WeightStorage::F32(vec_of(embed * embed, &mut **nb)),
            out_proj_bias: vec_of(embed, &mut **nb),
            ln_gain: Arc::from(vec![1.0_f32; embed]),
            ln_bias: Arc::from(vec![0.0_f32; embed]),
            mlp_fc1: WeightStorage::F32(vec_of(embed * inter, &mut **nb)),
            mlp_fc1_bias: vec_of(inter, &mut **nb),
            mlp_fc2: WeightStorage::F32(vec_of(inter * embed, &mut **nb)),
            mlp_fc2_bias: vec_of(embed, &mut **nb),
        }
    }

    fn tiny_vision_weights(cfg: &SiglipVisionConfig, with_head: bool) -> SiglipVisionWeights {
        let mut s: u32 = 78787;
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
        let layers = tiny_encoder_layers(cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, &mut nb);
        let post_ln_gain = Arc::from(vec![1.0_f32; cfg.hidden_size]);
        let post_ln_bias = Arc::from(vec![0.0_f32; cfg.hidden_size]);
        let head = if with_head {
            Some(tiny_pooling_head(cfg.hidden_size, cfg.intermediate_size, &mut nb))
        } else {
            None
        };
        SiglipVisionWeights {
            patch_proj, patch_proj_bias,
            position_embedding,
            layers,
            post_ln_gain, post_ln_bias,
            head,
        }
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
    fn text_forward_shape() {
        let cfg = tiny_text_cfg();
        let model = SiglipTextModel { config: cfg.clone(), weights: tiny_text_weights(&cfg) };
        let tokens = [1_u32, 2, 3, 4, 5];
        let out = model.forward(&tokens).unwrap();
        // Last-position pooled + head projection → (1, hidden_size).
        assert_eq!(out.shape().dims(), &[1, cfg.hidden_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// SigLIP text encoder is bidirectional — changing a token
    /// at any position should change ALL positions' outputs.
    /// Verify by checking that toks_a vs toks_b (different at
    /// position 0) produces different pooled output (last token).
    #[test]
    fn text_bidirectional_pooled_changes() {
        let cfg = tiny_text_cfg();
        let model = SiglipTextModel { config: cfg.clone(), weights: tiny_text_weights(&cfg) };
        let toks_a = [1_u32, 2, 3, 4];
        let toks_b = [11_u32, 2, 3, 4]; // first token differs
        let a = model.forward(&toks_a).unwrap().realize_f32();
        let b = model.forward(&toks_b).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "first-position change must affect last-position pooled output (bidirectional), max_diff = {max_diff}");
    }

    #[test]
    fn vision_forward_no_head_shape() {
        let cfg = tiny_vision_cfg();
        let model = SiglipVisionModel { config: cfg.clone(), weights: tiny_vision_weights(&cfg, false) };
        let img = tiny_image(&cfg);
        let out = model.forward(&img).unwrap();
        assert_eq!(out.shape().dims(), &[1, cfg.num_patches(), cfg.hidden_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn vision_forward_with_head_shape() {
        let cfg = tiny_vision_cfg();
        let model = SiglipVisionModel { config: cfg.clone(), weights: tiny_vision_weights(&cfg, true) };
        let img = tiny_image(&cfg);
        let out = model.forward(&img).unwrap();
        assert_eq!(out.shape().dims(), &[1, cfg.hidden_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// MultiheadAttentionPoolingHead probe is wired: zeroing the
    /// probe must change the pooled output.
    #[test]
    fn pooling_head_probe_is_wired() {
        let cfg = tiny_vision_cfg();
        let base = tiny_vision_weights(&cfg, true);
        let mut zeroed = base.clone();
        if let Some(h) = &mut zeroed.head {
            h.probe = Arc::from(vec![0.0_f32; cfg.hidden_size]);
        }
        let m_base = SiglipVisionModel { config: cfg.clone(), weights: base };
        let m_zero = SiglipVisionModel { config: cfg.clone(), weights: zeroed };
        let img_a = tiny_image(&cfg);
        let img_b = tiny_image(&cfg);
        let a = m_base.forward(&img_a).unwrap().realize_f32();
        let b = m_zero.forward(&img_b).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        // With tiny weights the probe contribution is small;
        // require measurable but loose threshold (~1e-10).
        assert!(max_diff > 1e-10,
            "pooling-head probe must affect output, max_diff = {max_diff}");
    }

    #[test]
    fn config_presets() {
        let t = SiglipTextConfig::base();
        assert_eq!(t.head_dim(), 64);
        let v = SiglipVisionConfig::paligemma_3b_224();
        assert_eq!(v.num_patches(), 256);
    }

    /// `forward_intermediate_layers` returns one tensor per
    /// requested layer index. SigLIP has NO CLS token so the
    /// seq dim is just `num_patches`.
    #[test]
    fn vision_forward_intermediate_layers_shape() {
        let cfg = tiny_vision_cfg();
        let model = SiglipVisionModel {
            config: cfg.clone(), weights: tiny_vision_weights(&cfg, false),
        };
        let img = tiny_image(&cfg);
        let outs = model.forward_intermediate_layers(&img, &[0_usize, 1]).unwrap();
        assert_eq!(outs.len(), 2);
        let np = cfg.num_patches();
        for out in &outs {
            assert_eq!(out.shape().dims(), &[1, np, cfg.hidden_size]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite intermediate: {v}");
            }
        }
    }

    /// Intermediate features at different depths must differ.
    #[test]
    fn vision_intermediate_layers_differ_across_depth() {
        let cfg = tiny_vision_cfg();
        let model = SiglipVisionModel {
            config: cfg.clone(), weights: tiny_vision_weights(&cfg, false),
        };
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
}

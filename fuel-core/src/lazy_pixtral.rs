//! Pixtral (Mistral AI vision-language model) ported to the
//! lazy-graph API.
//!
//! Pixtral 12B (Mistral 2024). Fourth multimodal composition
//! port after PaliGemma / LLaVA / Moondream. Distinguishing
//! features:
//!
//!   - Vision encoder is a **Mistral-shape ViT** (RmsNorm
//!     pre-LN, SwiGLU MLP, NO biases anywhere) rather than a
//!     CLIP-shape or DINOv2-shape ViT.
//!   - **2D RoPE** on Q/K with separate height/width frequency
//!     interleaving: `inv_freq[::2]` for x-positions and
//!     `inv_freq[1::2]` for y-positions. Each patch's RoPE
//!     entry is built from its `(row, col)` index. (Different
//!     from CLIP's no-RoPE, ViT's none, Gemma 4 vision's
//!     half-head-split 2D RoPE, and GLM-4's interleaved-rope
//!     trick.)
//!   - Pre-encoder **`ln_pre` RmsNorm** applied AFTER patch
//!     embedding but BEFORE the transformer stack.
//!   - **Conv2d patch embedding** (no bias).
//!   - Vision encoder uses `Activation::Silu` for the SwiGLU
//!     gate path (`gate * silu(up) → down`).
//!   - Vision projector is a **2-layer MLP** (Linear →
//!     activation → Linear, with biases) — slightly richer
//!     than PaliGemma's single linear.
//!
//! Text decoder is [`crate::lazy_mistral::MistralModel`]
//! (already in lazy). The composition is:
//!
//!   ```text
//!   image_features = pixtral_vision(pixel_values)
//!                       # (1, num_patches, vision_hidden)
//!   image_features = MMProjector(image_features)
//!                       # 2-layer MLP → text hidden
//!   text_embeds    = mistral.token_embedding(text_tokens)
//!   combined       = cat(image_features, text_embeds, dim=1)
//!   logits         = mistral.forward_embeds(combined, 0)
//!   ```
//!
//! # Scope (v1)
//!
//! Forward-only, single fixed-size image + single token
//! sequence, F32. Subsampled positions / variable image sizes
//! and the attention mask path deferred.

use crate::lazy::{
    load_tensor_as_f32, load_transposed_matrix_preserve_dtype,
    LazyTensor, WeightStorage,
};
use crate::lazy_mistral::{
    load_mistral_weights_with_prefix, MistralConfig, MistralModel, MistralWeights,
};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixtralActivation {
    Silu,
    Gelu,
    GeluPytorchTanh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PixtralVisionConfig {
    pub hidden_size: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub rope_theta: f64,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: Option<usize>,
    pub activation: PixtralActivation,
    pub rms_norm_eps: f64,
}

impl PixtralVisionConfig {
    pub fn head_dim_resolved(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }
    pub fn num_patches(&self) -> usize {
        let p = self.num_patches_per_side();
        p * p
    }
    /// Preset for the Pixtral-12B-2409 vision encoder.
    pub fn pixtral_12b_2409() -> Self {
        Self {
            hidden_size: 1024,
            num_channels: 3,
            image_size: 1024,
            patch_size: 16,
            rope_theta: 10_000.0,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            head_dim: None,
            activation: PixtralActivation::Silu,
            rms_norm_eps: 1e-5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PixtralVisionBlockWeights {
    pub attn_norm_gain: Arc<[f32]>,
    pub q_proj: WeightStorage, // no bias
    pub k_proj: WeightStorage,
    pub v_proj: WeightStorage,
    pub o_proj: WeightStorage,
    pub ffn_norm_gain: Arc<[f32]>,
    pub gate_proj: WeightStorage,
    pub up_proj: WeightStorage,
    pub down_proj: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct PixtralVisionWeights {
    /// Conv2d patch projection `[hidden, num_channels, patch, patch]`.
    pub patch_conv: Arc<[f32]>,
    pub ln_pre_gain: Arc<[f32]>,
    pub blocks: Vec<PixtralVisionBlockWeights>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PixtralProjectorConfig {
    pub in_dim: usize,
    pub out_dim: usize,
    pub activation: PixtralActivation,
}

#[derive(Debug, Clone)]
pub struct PixtralProjectorWeights {
    pub linear_1: WeightStorage,
    pub linear_1_bias: Arc<[f32]>,
    pub linear_2: WeightStorage,
    pub linear_2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct PixtralWeights {
    pub vision: PixtralVisionWeights,
    pub projector: PixtralProjectorWeights,
    pub text: MistralWeights,
}

#[derive(Debug, Clone)]
pub struct PixtralConfig {
    pub vision: PixtralVisionConfig,
    pub projector: PixtralProjectorConfig,
    pub text: MistralConfig,
}

#[derive(Debug, Clone)]
pub struct PixtralModel {
    pub config: PixtralConfig,
    pub weights: PixtralWeights,
}

impl PixtralModel {
    pub fn forward(
        &self,
        pixel_values: &LazyTensor,
        text_tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        assert_eq!(
            cfg.projector.out_dim, cfg.text.hidden_size,
            "projector out_dim must equal text hidden_size",
        );
        let text_len = text_tokens.len();
        assert!(text_len > 0);

        let vision_out = self.encode_vision(pixel_values)?;
        let projected = self.apply_projector(&vision_out)?;

        let mistral_embed_lt = pixel_values.const_f32_like(
            Arc::clone(&self.weights.text.token_embedding),
            Shape::from_dims(&[cfg.text.vocab_size, cfg.text.hidden_size]),
        );
        let token_ids = pixel_values.const_u32_like(
            text_tokens.to_vec(),
            Shape::from_dims(&[text_len]),
        );
        let text_embeds = mistral_embed_lt
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[1, text_len, cfg.text.hidden_size]))?;

        let combined = projected.concat(&text_embeds, 1_usize)?;
        let model = MistralModel {
            config: cfg.text.clone(),
            weights: self.weights.text.clone(),
        };
        model.forward_embeds(&combined, 0)
    }

    fn encode_vision(&self, pixel_values: &LazyTensor) -> Result<LazyTensor> {
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

        let np_side = cfg.num_patches_per_side();
        let np = cfg.num_patches();

        // Patch Conv2d (no bias).
        let conv_w = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_conv),
            Shape::from_dims(&[cfg.hidden_size, cfg.num_channels, cfg.patch_size, cfg.patch_size]),
        );
        let conv_out = pixel_values.conv2d(
            &conv_w,
            None,
            (cfg.patch_size, cfg.patch_size),
            (0, 0),
            1,
        )?;
        // (b, hidden, ph, pw) → (b, hidden, num_patches) → (b, num_patches, hidden)
        let patches = conv_out
            .reshape(Shape::from_dims(&[batch, cfg.hidden_size, np]))?
            .permute([0, 2, 1_usize])?;

        // Pre-encoder RmsNorm (Mistral-shape, no offset).
        let pre = patches.rms_norm_affine(std::sync::Arc::clone(&weights.ln_pre_gain), cfg.rms_norm_eps)?;

        // Precompute 2D RoPE cos/sin tables for all patches.
        let head_dim = cfg.head_dim_resolved();
        assert_eq!(head_dim % 2, 0, "head_dim must be even");
        let (cos_data, sin_data) = build_pixtral_2d_rope_tables(
            cfg.rope_theta, head_dim, np_side,
        );
        let cos = pixel_values.const_f32_like(
            Arc::from(cos_data),
            Shape::from_dims(&[np, head_dim]),
        );
        let sin = pixel_values.const_f32_like(
            Arc::from(sin_data),
            Shape::from_dims(&[np, head_dim]),
        );

        let mut h = pre;
        for block in &weights.blocks {
            h = self.apply_block(&h, block, &cos, &sin)?;
        }
        Ok(h)
    }

    fn apply_block(
        &self,
        x: &LazyTensor,
        block: &PixtralVisionBlockWeights,
        cos: &LazyTensor,
        sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config.vision;
        let dims = x.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim_resolved();

        // Pre-attn RmsNorm.
        let x_norm = x.rms_norm_affine(std::sync::Arc::clone(&block.attn_norm_gain), cfg.rms_norm_eps)?;

        let q = block.q_proj.apply_linear(&x_norm, h, h);
        let k = block.k_proj.apply_linear(&x_norm, h, h);
        let v = block.v_proj.apply_linear(&x_norm, h, h);

        let _ = (batch, seq);
        let q = q.split_heads(n_heads, head_dim)?;
        let k = k.split_heads(n_heads, head_dim)?;
        let v = v.split_heads(n_heads, head_dim)?;

        // Apply 2D RoPE to Q and K.
        let q_r = q.rope_with_tables(cos, sin)?;
        let k_r = k.rope_with_tables(cos, sin)?;

        let k_t = k_r.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?.mul_scalar(scale);
        // No causal mask — bidirectional vision attention.
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx.merge_heads()?;
        let attn_out = block.o_proj.apply_linear(&merged, h, h);
        let h1 = x.add(&attn_out)?;

        // Pre-FFN RmsNorm.
        let h1_norm = h1.rms_norm_affine(std::sync::Arc::clone(&block.ffn_norm_gain), cfg.rms_norm_eps)?;
        let gate = block.gate_proj.apply_linear(&h1_norm, h, cfg.intermediate_size);
        let up = block.up_proj.apply_linear(&h1_norm, h, cfg.intermediate_size);
        let activated = match cfg.activation {
            PixtralActivation::Silu => up.silu(),
            PixtralActivation::Gelu => up.gelu_erf(),
            PixtralActivation::GeluPytorchTanh => up.gelu(),
        };
        let ffn_inner = gate.mul(&activated)?;
        let down = block.down_proj.apply_linear(&ffn_inner, cfg.intermediate_size, h);
        h1.add(&down)
    }

    fn apply_projector(&self, vision_out: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config.projector;
        let weights = &self.weights.projector;
        let l1 = weights.linear_1.apply_linear(vision_out, cfg.in_dim, cfg.out_dim);
        let l1_b_t = vision_out.const_f32_like(
            Arc::clone(&weights.linear_1_bias),
            Shape::from_dims(&[cfg.out_dim]),
        );
        let l1 = l1.broadcast_add(&l1_b_t)?;
        let activated = match cfg.activation {
            PixtralActivation::Silu => l1.silu(),
            PixtralActivation::Gelu => l1.gelu_erf(),
            PixtralActivation::GeluPytorchTanh => l1.gelu(),
        };
        let l2 = weights.linear_2.apply_linear(&activated, cfg.out_dim, cfg.out_dim);
        let l2_b_t = vision_out.const_f32_like(
            Arc::clone(&weights.linear_2_bias),
            Shape::from_dims(&[cfg.out_dim]),
        );
        l2.broadcast_add(&l2_b_t)
    }
}

/// Build the 2D RoPE cos/sin tables for Pixtral.
///
/// `inv_freq[2i]` (even indices) feeds height (row) positions;
/// `inv_freq[2i+1]` (odd indices) feeds width (col) positions.
/// Each patch `(r, c)` for `r, c ∈ [0, num_patches_per_side)`
/// gets a cos/sin entry of length `head_dim`. The standard
/// split-half RoPE convention is used for the per-head layout.
fn build_pixtral_2d_rope_tables(
    theta: f64,
    head_dim: usize,
    num_patches_per_side: usize,
) -> (Vec<f32>, Vec<f32>) {
    let dim = head_dim;
    let half = dim / 2;
    // Per-frequency base: 1 / theta^(2i / dim) for i in [0, dim/2).
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (theta.powf(-2.0 * i as f64 / dim as f64)) as f32)
        .collect();
    // Split into height (even-indexed inv_freq) and width (odd-indexed).
    let freqs_h: Vec<f32> = inv_freq.iter().step_by(2).copied().collect();
    let freqs_w: Vec<f32> = inv_freq.iter().skip(1).step_by(2).copied().collect();
    let qh = freqs_h.len(); // = (dim + 2) / 4 typically
    let qw = freqs_w.len();
    assert_eq!(qh + qw, half, "freq splits must cover the half-dim");

    let np = num_patches_per_side * num_patches_per_side;
    let mut cos = vec![0.0_f32; np * dim];
    let mut sin = vec![0.0_f32; np * dim];

    // For each patch at (r, c):
    //   first qh features ← cos/sin of r * freqs_h[i]
    //   next qw features  ← cos/sin of c * freqs_w[i]
    //   second half mirrors the first (standard split-half).
    for r in 0..num_patches_per_side {
        for c in 0..num_patches_per_side {
            let p = r * num_patches_per_side + c;
            let off = p * dim;
            // First half (indices 0..half): cat(r*freqs_h, c*freqs_w).
            for i in 0..qh {
                let theta_val = r as f32 * freqs_h[i];
                cos[off + i] = theta_val.cos();
                sin[off + i] = theta_val.sin();
            }
            for i in 0..qw {
                let theta_val = c as f32 * freqs_w[i];
                cos[off + qh + i] = theta_val.cos();
                sin[off + qh + i] = theta_val.sin();
            }
            // Second half (indices half..dim) duplicates the first
            // (standard rope_with_tables expects this layout).
            for i in 0..half {
                cos[off + half + i] = cos[off + i];
                sin[off + half + i] = sin[off + i];
            }
        }
    }
    (cos, sin)
}

// ---- Safetensors loader ----------------------------------------------------

/// Load the Pixtral vision tower's weights under the given HF
/// prefix (typically `"vision_tower."` for a full Pixtral
/// checkpoint). Pixtral's vision encoder uses Mistral-shape
/// RmsNorm + SwiGLU + bias-free Q/K/V/O projections. HF tensor
/// names (under `<prefix>`):
///   - `patch_conv.weight` (Conv2d, no bias)
///   - `ln_pre.weight` (RmsNorm gain)
///   - `transformer.layers.{i}.attention_norm.weight`
///   - `transformer.layers.{i}.attention.{q,k,v,o}_proj.weight`
///   - `transformer.layers.{i}.ffn_norm.weight`
///   - `transformer.layers.{i}.feed_forward.{gate,up,down}_proj.weight`
pub fn load_pixtral_vision_weights(
    st: &crate::safetensors::MmapedSafetensors,
    cfg: &PixtralVisionConfig,
    prefix: &str,
) -> Result<PixtralVisionWeights> {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;

    let patch_conv = load_tensor_as_f32(st, &format!("{prefix}patch_conv.weight"))?;
    let expected_patch = h * cfg.num_channels * cfg.patch_size * cfg.patch_size;
    if patch_conv.len() != expected_patch {
        crate::bail!(
            "{prefix}patch_conv.weight: {} elts, expected {}",
            patch_conv.len(), expected_patch,
        );
    }
    let ln_pre_gain = load_tensor_as_f32(st, &format!("{prefix}ln_pre.weight"))?;
    if ln_pre_gain.len() != h {
        crate::bail!(
            "{prefix}ln_pre.weight: {} elts, expected {}",
            ln_pre_gain.len(), h,
        );
    }

    let mut blocks: Vec<PixtralVisionBlockWeights> =
        Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("{prefix}transformer.layers.{i}");
        let attn_norm_gain = load_tensor_as_f32(
            st, &format!("{p}.attention_norm.weight"),
        )?;
        let q_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.attention.q_proj.weight"), h, h,
        )?;
        let k_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.attention.k_proj.weight"), h, h,
        )?;
        let v_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.attention.v_proj.weight"), h, h,
        )?;
        let o_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.attention.o_proj.weight"), h, h,
        )?;
        let ffn_norm_gain = load_tensor_as_f32(
            st, &format!("{p}.ffn_norm.weight"),
        )?;
        let gate_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.feed_forward.gate_proj.weight"), inter, h,
        )?;
        let up_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.feed_forward.up_proj.weight"), inter, h,
        )?;
        let down_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.feed_forward.down_proj.weight"), h, inter,
        )?;
        blocks.push(PixtralVisionBlockWeights {
            attn_norm_gain: Arc::from(attn_norm_gain),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            ffn_norm_gain: Arc::from(ffn_norm_gain),
            gate_proj,
            up_proj,
            down_proj,
        });
    }

    Ok(PixtralVisionWeights {
        patch_conv: Arc::from(patch_conv),
        ln_pre_gain: Arc::from(ln_pre_gain),
        blocks,
    })
}

/// Load the Pixtral multi-modal projector (2-layer MLP with
/// biases) under the given HF prefix (typically
/// `"multi_modal_projector."`).
pub fn load_pixtral_projector_weights(
    st: &crate::safetensors::MmapedSafetensors,
    cfg: &PixtralProjectorConfig,
    prefix: &str,
) -> Result<PixtralProjectorWeights> {
    let linear_1 = load_transposed_matrix_preserve_dtype(
        st, &format!("{prefix}linear_1.weight"), cfg.out_dim, cfg.in_dim,
    )?;
    let linear_1_bias = load_tensor_as_f32(
        st, &format!("{prefix}linear_1.bias"),
    )?;
    if linear_1_bias.len() != cfg.out_dim {
        crate::bail!(
            "{prefix}linear_1.bias: {} elts, expected {}",
            linear_1_bias.len(), cfg.out_dim,
        );
    }
    let linear_2 = load_transposed_matrix_preserve_dtype(
        st, &format!("{prefix}linear_2.weight"), cfg.out_dim, cfg.out_dim,
    )?;
    let linear_2_bias = load_tensor_as_f32(
        st, &format!("{prefix}linear_2.bias"),
    )?;
    if linear_2_bias.len() != cfg.out_dim {
        crate::bail!(
            "{prefix}linear_2.bias: {} elts, expected {}",
            linear_2_bias.len(), cfg.out_dim,
        );
    }
    Ok(PixtralProjectorWeights {
        linear_1,
        linear_1_bias: Arc::from(linear_1_bias),
        linear_2,
        linear_2_bias: Arc::from(linear_2_bias),
    })
}

impl PixtralWeights {
    /// Load full Pixtral weights from a HuggingFace safetensors
    /// file. HF Pixtral naming:
    ///   - `vision_tower.*` — Mistral-shape vision encoder
    ///   - `multi_modal_projector.linear_{1,2}.*` — 2-layer MLP
    ///   - `language_model.*` — Mistral decoder
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &PixtralConfig,
    ) -> Result<Self> {
        let vision = load_pixtral_vision_weights(st, &cfg.vision, "vision_tower.")?;
        let projector = load_pixtral_projector_weights(
            st, &cfg.projector, "multi_modal_projector.",
        )?;
        let text = load_mistral_weights_with_prefix(st, &cfg.text, "language_model.")?;
        Ok(PixtralWeights { vision, projector, text })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::LayerWeights;
    use crate::lazy_mistral::MistralConfig;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_vision_cfg() -> PixtralVisionConfig {
        PixtralVisionConfig {
            hidden_size: 16,
            num_channels: 3,
            image_size: 8,
            patch_size: 4,
            rope_theta: 10_000.0,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            head_dim: None,
            activation: PixtralActivation::Silu,
            rms_norm_eps: 1e-5,
        }
    }

    fn tiny_text_cfg() -> MistralConfig {
        MistralConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: 4,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 64,
            sliding_window: None,
        }
    }

    fn tiny_projector_cfg(text_hidden: usize) -> PixtralProjectorConfig {
        PixtralProjectorConfig {
            in_dim: 16,
            out_dim: text_hidden,
            activation: PixtralActivation::Silu,
        }
    }

    fn tiny_vision_weights(cfg: &PixtralVisionConfig) -> PixtralVisionWeights {
        let mut s: u32 = 89898;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let patch_conv = vec_of(
            h * cfg.num_channels * cfg.patch_size * cfg.patch_size,
            &mut *nb,
        );
        let ln_pre_gain = Arc::from(vec![1.0_f32; h]);
        let blocks: Vec<PixtralVisionBlockWeights> = (0..cfg.num_hidden_layers).map(|_| PixtralVisionBlockWeights {
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            q_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            k_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            v_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            o_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
            gate_proj: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
            up_proj: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
            down_proj: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
        }).collect();
        PixtralVisionWeights { patch_conv, ln_pre_gain, blocks }
    }

    fn tiny_projector_weights(cfg: &PixtralProjectorConfig) -> PixtralProjectorWeights {
        let mut s: u32 = 11212;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        PixtralProjectorWeights {
            linear_1: WeightStorage::F32(vec_of(cfg.in_dim * cfg.out_dim, &mut *nb)),
            linear_1_bias: vec_of(cfg.out_dim, &mut *nb),
            linear_2: WeightStorage::F32(vec_of(cfg.out_dim * cfg.out_dim, &mut *nb)),
            linear_2_bias: vec_of(cfg.out_dim, &mut *nb),
        }
    }

    fn tiny_text_weights(cfg: &MistralConfig) -> MistralWeights {
        let mut s: u32 = 33445;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let h = cfg.hidden_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
            ffn_up: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        MistralWeights { token_embedding, layers, final_norm_gain, output }
    }

    fn tiny_image(cfg: &PixtralVisionConfig) -> LazyTensor {
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
        let p_cfg = tiny_projector_cfg(t_cfg.hidden_size);
        let cfg = PixtralConfig {
            vision: v_cfg.clone(),
            projector: p_cfg.clone(),
            text: t_cfg.clone(),
        };
        let weights = PixtralWeights {
            vision: tiny_vision_weights(&v_cfg),
            projector: tiny_projector_weights(&p_cfg),
            text: tiny_text_weights(&t_cfg),
        };
        let model = PixtralModel { config: cfg, weights };
        let img = tiny_image(&v_cfg);
        let toks = [1_u32, 2, 3];
        let logits = model.forward(&img, &toks).unwrap();
        let expected = v_cfg.num_patches() + toks.len();
        assert_eq!(logits.shape().dims(), &[1, expected, t_cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// 2D RoPE table values: position (0,0) should have cos = 1
    /// and sin = 0 across all features (theta = 0 → cos/sin
    /// reduce to 1/0).
    #[test]
    fn rope_position_zero_is_identity() {
        let (cos, sin) = build_pixtral_2d_rope_tables(10_000.0, 8, 4);
        // Position (0, 0) = first row of the table.
        for i in 0..8 {
            assert!((cos[i] - 1.0).abs() < 1e-6, "cos[0, {i}] = {} != 1", cos[i]);
            assert!((sin[i]).abs() < 1e-6, "sin[0, {i}] = {} != 0", sin[i]);
        }
    }

    mod load {
        use super::*;
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;
        use std::collections::HashMap;

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
                let v = TensorView::new(*dt, shape.clone(), data).expect("TensorView");
                views.insert(k.clone(), v);
            }
            let bytes = safetensors::serialize(&views, None).expect("serialize");
            let path = std::env::temp_dir().join(format!(
                "lazy_pixtral_load_{}_{}.safetensors",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
            ));
            std::fs::write(&path, bytes).expect("write tempfile");
            path
        }

        fn build_tiny_safetensors(
            v_cfg: &PixtralVisionConfig,
            t_cfg: &MistralConfig,
            p_cfg: &PixtralProjectorConfig,
        ) -> std::path::PathBuf {
            let mut map: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
            let mut s: u32 = 9090;
            let mut nxt = || -> f32 {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.01
            };
            let mut vec_n = |n: usize| -> Vec<f32> { (0..n).map(|_| nxt()).collect() };

            // Vision tower under `vision_tower.*`
            let vp = "vision_tower.";
            let h = v_cfg.hidden_size;
            let inter = v_cfg.intermediate_size;
            put(&mut map, &format!("{vp}patch_conv.weight"),
                &[h, v_cfg.num_channels, v_cfg.patch_size, v_cfg.patch_size],
                &vec_n(h * v_cfg.num_channels * v_cfg.patch_size * v_cfg.patch_size));
            put(&mut map, &format!("{vp}ln_pre.weight"), &[h], &vec_n(h));
            for i in 0..v_cfg.num_hidden_layers {
                let p = format!("{vp}transformer.layers.{i}");
                put(&mut map, &format!("{p}.attention_norm.weight"), &[h], &vec_n(h));
                for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                    put(&mut map, &format!("{p}.attention.{proj}.weight"),
                        &[h, h], &vec_n(h * h));
                }
                put(&mut map, &format!("{p}.ffn_norm.weight"), &[h], &vec_n(h));
                put(&mut map, &format!("{p}.feed_forward.gate_proj.weight"),
                    &[inter, h], &vec_n(inter * h));
                put(&mut map, &format!("{p}.feed_forward.up_proj.weight"),
                    &[inter, h], &vec_n(inter * h));
                put(&mut map, &format!("{p}.feed_forward.down_proj.weight"),
                    &[h, inter], &vec_n(h * inter));
            }

            // Projector under `multi_modal_projector.*`
            let pp = "multi_modal_projector.";
            put(&mut map, &format!("{pp}linear_1.weight"),
                &[p_cfg.out_dim, p_cfg.in_dim], &vec_n(p_cfg.out_dim * p_cfg.in_dim));
            put(&mut map, &format!("{pp}linear_1.bias"),
                &[p_cfg.out_dim], &vec_n(p_cfg.out_dim));
            put(&mut map, &format!("{pp}linear_2.weight"),
                &[p_cfg.out_dim, p_cfg.out_dim], &vec_n(p_cfg.out_dim * p_cfg.out_dim));
            put(&mut map, &format!("{pp}linear_2.bias"),
                &[p_cfg.out_dim], &vec_n(p_cfg.out_dim));

            // Mistral text decoder under `language_model.*`
            let lp = "language_model.";
            let d = t_cfg.hidden_size;
            let kv = t_cfg.num_key_value_heads * t_cfg.head_dim;
            put(&mut map, &format!("{lp}model.embed_tokens.weight"),
                &[t_cfg.vocab_size, d], &vec_n(t_cfg.vocab_size * d));
            for i in 0..t_cfg.num_hidden_layers {
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
                    &[t_cfg.intermediate_size, d], &vec_n(t_cfg.intermediate_size * d));
                put(&mut map, &format!("{p}.mlp.up_proj.weight"),
                    &[t_cfg.intermediate_size, d], &vec_n(t_cfg.intermediate_size * d));
                put(&mut map, &format!("{p}.mlp.down_proj.weight"),
                    &[d, t_cfg.intermediate_size], &vec_n(d * t_cfg.intermediate_size));
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
            let p_cfg = tiny_projector_cfg(t_cfg.hidden_size);
            let path = build_tiny_safetensors(&v_cfg, &t_cfg, &p_cfg);
            let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path) }
                .expect("mmap safetensors");
            let cfg = PixtralConfig {
                vision: v_cfg.clone(),
                projector: p_cfg.clone(),
                text: t_cfg.clone(),
            };
            let w = PixtralWeights::load_from_mmapped(&st, &cfg)
                .expect("PixtralWeights::load_from_mmapped");

            // Shape spot-checks.
            assert_eq!(w.vision.blocks.len(), v_cfg.num_hidden_layers);
            assert_eq!(w.text.layers.len(), t_cfg.num_hidden_layers);
            assert_eq!(w.projector.linear_1_bias.len(), p_cfg.out_dim);
            assert_eq!(w.projector.linear_2_bias.len(), p_cfg.out_dim);
            assert_eq!(w.text.token_embedding.len(),
                t_cfg.vocab_size * t_cfg.hidden_size);

            // Forward must produce finite logits with the loaded weights.
            let model = PixtralModel { config: cfg, weights: w };
            let img = tiny_image(&v_cfg);
            let toks = [1_u32, 2, 3];
            let logits = model.forward(&img, &toks).unwrap().realize_f32();
            for v in &logits {
                assert!(v.is_finite(), "non-finite logit");
            }
            let _ = std::fs::remove_file(&path);
        }

        /// Documents the canonical from-hub usage. Ignored in CI.
        #[test]
        #[ignore]
        fn from_hub_smoke_pixtral_12b() {
            // Canonical HF repo: mistralai/Pixtral-12B-2409 (or
            // mistral-community/pixtral-12b for the easier-to-load
            // HF-format mirror). The loader expects the standard HF
            // Pixtral naming:
            //   vision_tower.*  (patch_conv + ln_pre + transformer.layers.{i}.*)
            //   multi_modal_projector.linear_{1,2}.*
            //   language_model.*  (Mistral decoder under model.*)
        }
    }
}

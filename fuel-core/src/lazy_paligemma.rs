//! PaliGemma multimodal model ported to the lazy-graph API.
//!
//! PaliGemma from Google (Beyer et al. 2024). First multimodal
//! **composition** port — combines an existing lazy vision
//! encoder ([`crate::lazy_siglip::SiglipVisionModel`]) and an
//! existing lazy language model ([`crate::lazy_gemma::GemmaModel`])
//! with a thin projection + interleaving layer:
//!
//!   ```text
//!   image_features = siglip_vision(pixel_values)         # (1, num_patches, vision_hidden)
//!   image_features = MultiModalProjector(image_features) # (1, num_patches, gemma_hidden)
//!   image_features = L2_normalize(image_features)        # per-token L2-norm
//!   text_embeds    = gemma.token_embedding(text_tokens)  # (1, text_len, gemma_hidden)
//!   combined       = cat(image_features, text_embeds, dim=1)
//!   logits         = gemma.forward_embeds(combined, start_pos=0)
//!   ```
//!
//! The image features are **prepended** to the text embedding
//! sequence (image-then-text layout per PaliGemma's convention).
//! The Gemma language model runs on the combined sequence as if
//! it were all-text input.
//!
//! # Scope (v1)
//!
//! Forward-only, single image + single token sequence, F32. The
//! eager reference's autoregressive decode loop (separate
//! `setup` + `forward` calls with KV cache) is out of scope —
//! v1 returns logits for the entire combined sequence in one
//! pass.

use crate::lazy::{
    load_tensor_as_f32, load_transposed_matrix_preserve_dtype,
    LayerWeights, LazyTensor, WeightStorage,
};
use crate::lazy_gemma::{GemmaConfig, GemmaModel, GemmaWeights};
use crate::lazy_siglip::{
    SiglipEncoderLayerWeights, SiglipVisionConfig, SiglipVisionModel, SiglipVisionWeights,
};
use crate::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct PaligemmaConfig {
    pub vision_config: SiglipVisionConfig,
    pub text_config: GemmaConfig,
    /// Multi-modal projection target dim (typically equals
    /// `text_config.hidden_size`).
    pub projection_dim: usize,
}

#[derive(Debug, Clone)]
pub struct PaligemmaWeights {
    pub vision: SiglipVisionWeights,
    /// `[vision_hidden, projection_dim]`.
    pub mm_proj: WeightStorage,
    pub mm_proj_bias: Arc<[f32]>,
    pub text: GemmaWeights,
}

#[derive(Debug, Clone)]
pub struct PaligemmaModel {
    pub config: PaligemmaConfig,
    pub weights: PaligemmaWeights,
}

impl PaligemmaModel {
    /// Run the full multimodal forward pass. Returns logits
    /// for the combined `[image_features; text_embeds]`
    /// sequence of shape `(1, num_patches + text_len, vocab)`.
    pub fn forward(
        &self,
        pixel_values: &LazyTensor,
        text_tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let v_cfg = &cfg.vision_config;
        let t_cfg = &cfg.text_config;
        assert_eq!(
            cfg.projection_dim, t_cfg.hidden_size,
            "v1: projection_dim must equal text hidden_size",
        );
        let text_len = text_tokens.len();
        assert!(text_len > 0, "text_tokens must be non-empty");

        // ---- Run SigLIP vision encoder (no pooling head) -----------------
        let vision = SiglipVisionModel {
            config: v_cfg.clone(),
            weights: self.weights.vision.clone(),
        };
        // SigLIP without head returns (1, num_patches, vision_hidden).
        let image_features = vision.forward(pixel_values)?;
        let np = v_cfg.num_patches();
        let dims = image_features.shape();
        let dims = dims.dims();
        assert_eq!(dims, &[1, np, v_cfg.hidden_size],
            "SigLIP w/o head must produce (1, num_patches, hidden); got {dims:?}");

        // ---- Multi-modal projection: vision_hidden → projection_dim -----
        let projected = self.weights.mm_proj.apply_linear(
            &image_features,
            v_cfg.hidden_size,
            cfg.projection_dim,
        );
        let bias_t = pixel_values.const_f32_like(
            Arc::clone(&self.weights.mm_proj_bias),
            Shape::from_dims(&[cfg.projection_dim]),
        );
        let image_proj = projected.broadcast_add(&bias_t)?;
        // L2 normalize image features per-token.
        let image_proj_n = l2_normalize_last(&image_proj, 1e-12)?;

        // ---- Embed text tokens via Gemma's token embedding --------------
        // We must anchor on the SAME graph used by the vision tower so the
        // concat works. SigLIP's forward already created token_embedding /
        // patches on its own graph anchored on `pixel_values`. Use
        // `pixel_values` as the graph anchor.
        let gemma_embed_lt = pixel_values.const_f32_like(
            Arc::clone(&self.weights.text.token_embedding),
            Shape::from_dims(&[t_cfg.vocab_size, t_cfg.hidden_size]),
        );
        let token_ids = pixel_values.const_u32_like(
            text_tokens.to_vec(),
            Shape::from_dims(&[text_len]),
        );
        let text_embeds = gemma_embed_lt
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[1, text_len, t_cfg.hidden_size]))?;
        // Gemma applies sqrt(hidden_size) scaling to token embeds.
        let text_embeds_scaled = text_embeds.mul_scalar((t_cfg.hidden_size as f64).sqrt());

        // ---- Concat [image; text] and run Gemma layers -------------------
        let combined = image_proj_n.concat(&text_embeds_scaled, 1_usize)?;
        let gemma = GemmaModel {
            config: t_cfg.clone(),
            weights: self.weights.text.clone(),
        };
        gemma.forward_embeds(&combined, 0)
    }

    /// Like [`Self::forward`] but skips the LM head and returns
    /// post-final-RmsNorm hidden states
    /// `(1, num_patches + text_len, hidden_size)`.
    ///
    /// Used by retrieval models (ColPali, ColIdefics, etc.) that
    /// project the per-token hidden states into a dense embedding
    /// space rather than predicting tokens.
    pub fn forward_hidden(
        &self,
        pixel_values: &LazyTensor,
        text_tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let v_cfg = &cfg.vision_config;
        let t_cfg = &cfg.text_config;
        assert_eq!(
            cfg.projection_dim, t_cfg.hidden_size,
            "v1: projection_dim must equal text hidden_size",
        );
        let text_len = text_tokens.len();
        assert!(text_len > 0, "text_tokens must be non-empty");

        let vision = SiglipVisionModel {
            config: v_cfg.clone(),
            weights: self.weights.vision.clone(),
        };
        let image_features = vision.forward(pixel_values)?;
        let np = v_cfg.num_patches();
        let dims = image_features.shape();
        let dims = dims.dims();
        assert_eq!(dims, &[1, np, v_cfg.hidden_size]);

        let projected = self.weights.mm_proj.apply_linear(
            &image_features,
            v_cfg.hidden_size,
            cfg.projection_dim,
        );
        let bias_t = pixel_values.const_f32_like(
            Arc::clone(&self.weights.mm_proj_bias),
            Shape::from_dims(&[cfg.projection_dim]),
        );
        let image_proj = projected.broadcast_add(&bias_t)?;
        let image_proj_n = l2_normalize_last(&image_proj, 1e-12)?;

        let gemma_embed_lt = pixel_values.const_f32_like(
            Arc::clone(&self.weights.text.token_embedding),
            Shape::from_dims(&[t_cfg.vocab_size, t_cfg.hidden_size]),
        );
        let token_ids = pixel_values.const_u32_like(
            text_tokens.to_vec(),
            Shape::from_dims(&[text_len]),
        );
        let text_embeds = gemma_embed_lt
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[1, text_len, t_cfg.hidden_size]))?;
        let text_embeds_scaled = text_embeds.mul_scalar((t_cfg.hidden_size as f64).sqrt());

        let combined = image_proj_n.concat(&text_embeds_scaled, 1_usize)?;
        let gemma = GemmaModel {
            config: t_cfg.clone(),
            weights: self.weights.text.clone(),
        };
        gemma.forward_hidden_embeds(&combined, 0)
    }
}

/// L2-normalize along the last dim with an epsilon-clamped
/// denominator.
fn l2_normalize_last(x: &LazyTensor, eps: f64) -> Result<LazyTensor> {
    let last = x.shape().dims().len() - 1;
    x.l2_normalize(last, eps)
}

// ---- Safetensors loader ----------------------------------------------------

/// Load a SigLIP vision tower's weights from a safetensors file using
/// the standard HF prefix-relative naming (under `<prefix>`):
///   - `embeddings.patch_embedding.{weight,bias}` (Conv2d, with bias)
///   - `embeddings.position_embedding.weight` (shape `[num_patches, hidden]`)
///   - `encoder.layers.{i}.{layer_norm1,layer_norm2}.{weight,bias}`
///   - `encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}`
///   - `encoder.layers.{i}.mlp.{fc1,fc2}.{weight,bias}`
///   - `post_layernorm.{weight,bias}`
///
/// The pooling head is **NOT** loaded — PaliGemma uses SigLIP without
/// a head, matching the existing forward path.
pub fn load_siglip_vision_weights(
    st: &crate::safetensors::MmapedSafetensors,
    cfg: &SiglipVisionConfig,
    prefix: &str,
) -> Result<SiglipVisionWeights> {
    let h = cfg.hidden_size;
    let np = cfg.num_patches();
    let inter = cfg.intermediate_size;

    let patch_proj = load_tensor_as_f32(
        st, &format!("{prefix}embeddings.patch_embedding.weight"),
    )?;
    let patch_proj_bias = load_tensor_as_f32(
        st, &format!("{prefix}embeddings.patch_embedding.bias"),
    )?;
    let position_embedding = load_tensor_as_f32(
        st, &format!("{prefix}embeddings.position_embedding.weight"),
    )?;
    if position_embedding.len() != np * h {
        crate::bail!(
            "{prefix}embeddings.position_embedding.weight: {} elts, expected {}",
            position_embedding.len(), np * h,
        );
    }
    let post_ln_gain = load_tensor_as_f32(st, &format!("{prefix}post_layernorm.weight"))?;
    let post_ln_bias = load_tensor_as_f32(st, &format!("{prefix}post_layernorm.bias"))?;

    let mut layers: Vec<SiglipEncoderLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("{prefix}encoder.layers.{i}");
        let ln1_gain = load_tensor_as_f32(st, &format!("{p}.layer_norm1.weight"))?;
        let ln1_bias = load_tensor_as_f32(st, &format!("{p}.layer_norm1.bias"))?;
        let ln2_gain = load_tensor_as_f32(st, &format!("{p}.layer_norm2.weight"))?;
        let ln2_bias = load_tensor_as_f32(st, &format!("{p}.layer_norm2.bias"))?;
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
        let fc1 = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.mlp.fc1.weight"), inter, h,
        )?;
        let fc1_bias = load_tensor_as_f32(st, &format!("{p}.mlp.fc1.bias"))?;
        let fc2 = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.mlp.fc2.weight"), h, inter,
        )?;
        let fc2_bias = load_tensor_as_f32(st, &format!("{p}.mlp.fc2.bias"))?;
        layers.push(SiglipEncoderLayerWeights {
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

    Ok(SiglipVisionWeights {
        patch_proj: Arc::from(patch_proj),
        patch_proj_bias: Arc::from(patch_proj_bias),
        position_embedding: Arc::from(position_embedding),
        layers,
        post_ln_gain: Arc::from(post_ln_gain),
        post_ln_bias: Arc::from(post_ln_bias),
        head: None,
    })
}

/// Load Gemma decoder weights using `<prefix>model.<tensor>` HF naming.
/// Gemma's `attention_bias` flag toggles per-layer Q/K/V biases.
/// Output projection ties to `model.embed_tokens.weight` when
/// `lm_head.weight` is absent (HF Gemma tied default).
pub fn load_gemma_weights_with_prefix(
    st: &crate::safetensors::MmapedSafetensors,
    cfg: &GemmaConfig,
    prefix: &str,
) -> Result<GemmaWeights> {
    let h = cfg.hidden_size;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    let token_embedding = load_tensor_as_f32(
        st, &format!("{prefix}model.embed_tokens.weight"),
    )?;
    if token_embedding.len() != cfg.vocab_size * h {
        crate::bail!(
            "{prefix}model.embed_tokens.weight: {} elts, expected {}",
            token_embedding.len(), cfg.vocab_size * h,
        );
    }

    let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("{prefix}model.layers.{i}");
        let attn_q = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.self_attn.q_proj.weight"), h, h,
        )?;
        let attn_k = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.self_attn.k_proj.weight"), kv, h,
        )?;
        let attn_v = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.self_attn.v_proj.weight"), kv, h,
        )?;
        let attn_o = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.self_attn.o_proj.weight"), h, h,
        )?;
        let ffn_gate = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.mlp.gate_proj.weight"), cfg.intermediate_size, h,
        )?;
        let ffn_up = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.mlp.up_proj.weight"), cfg.intermediate_size, h,
        )?;
        let ffn_down = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.mlp.down_proj.weight"), h, cfg.intermediate_size,
        )?;
        let attn_norm_gain = load_tensor_as_f32(
            st, &format!("{p}.input_layernorm.weight"),
        )?;
        let ffn_norm_gain = load_tensor_as_f32(
            st, &format!("{p}.post_attention_layernorm.weight"),
        )?;
        let (attn_q_bias, attn_k_bias, attn_v_bias) = if cfg.attention_bias {
            (
                load_tensor_as_f32(st, &format!("{p}.self_attn.q_proj.bias")).ok().map(Arc::from),
                load_tensor_as_f32(st, &format!("{p}.self_attn.k_proj.bias")).ok().map(Arc::from),
                load_tensor_as_f32(st, &format!("{p}.self_attn.v_proj.bias")).ok().map(Arc::from),
            )
        } else {
            (None, None, None)
        };
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
    // Gemma typically ties lm_head to embeddings; fall back to tied
    // when the explicit `lm_head.weight` is absent.
    let output: WeightStorage = match load_transposed_matrix_preserve_dtype(
        st, &format!("{prefix}lm_head.weight"), cfg.vocab_size, h,
    ) {
        Ok(w) => w,
        Err(_) => {
            let mut transposed = vec![0.0_f32; h * cfg.vocab_size];
            for i in 0..cfg.vocab_size {
                for j in 0..h {
                    transposed[j * cfg.vocab_size + i] = token_embedding[i * h + j];
                }
            }
            WeightStorage::F32(Arc::from(transposed))
        }
    };

    Ok(GemmaWeights {
        token_embedding: Arc::from(token_embedding),
        layers,
        final_norm_gain: Arc::from(final_norm_gain),
        output,
    })
}

impl PaligemmaWeights {
    /// Load PaliGemma weights from a HuggingFace safetensors file.
    /// HF PaliGemma naming:
    ///   - `vision_tower.vision_model.*` — SigLIP vision tower (no head)
    ///   - `multi_modal_projector.linear.{weight,bias}` — MM projector
    ///   - `language_model.*` — Gemma decoder (model.embed_tokens.weight, etc.)
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &PaligemmaConfig,
    ) -> Result<Self> {
        let v_cfg = &cfg.vision_config;
        let t_cfg = &cfg.text_config;
        let vision = load_siglip_vision_weights(
            st, v_cfg, "vision_tower.vision_model.",
        )?;
        let mm_proj = load_transposed_matrix_preserve_dtype(
            st,
            "multi_modal_projector.linear.weight",
            cfg.projection_dim,
            v_cfg.hidden_size,
        )?;
        let mm_proj_bias = load_tensor_as_f32(
            st, "multi_modal_projector.linear.bias",
        )?;
        if mm_proj_bias.len() != cfg.projection_dim {
            crate::bail!(
                "multi_modal_projector.linear.bias: {} elts, expected {}",
                mm_proj_bias.len(), cfg.projection_dim,
            );
        }
        let text = load_gemma_weights_with_prefix(st, t_cfg, "language_model.")?;
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
    use crate::lazy::LayerWeights;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_vision_cfg() -> SiglipVisionConfig {
        SiglipVisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_channels: 3,
            image_size: 8,
            patch_size: 4,
            hidden_activation: crate::lazy_siglip::SiglipActivation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }

    fn tiny_text_cfg() -> GemmaConfig {
        GemmaConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            attention_bias: false,
            hidden_activation: crate::lazy_gemma::GemmaActivation::GeluPytorchTanh,
        }
    }

    fn tiny_vision_weights(cfg: &SiglipVisionConfig) -> SiglipVisionWeights {
        let mut s: u32 = 90909;
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
        let layers: Vec<_> = (0..cfg.num_hidden_layers).map(|_|
            crate::lazy_siglip::SiglipEncoderLayerWeights {
                ln1_gain: Arc::from(vec![1.0_f32; cfg.hidden_size]),
                ln1_bias: Arc::from(vec![0.0_f32; cfg.hidden_size]),
                q_proj: WeightStorage::F32(vec_of(cfg.hidden_size * cfg.hidden_size, &mut *nb)),
                q_proj_bias: vec_of(cfg.hidden_size, &mut *nb),
                k_proj: WeightStorage::F32(vec_of(cfg.hidden_size * cfg.hidden_size, &mut *nb)),
                k_proj_bias: vec_of(cfg.hidden_size, &mut *nb),
                v_proj: WeightStorage::F32(vec_of(cfg.hidden_size * cfg.hidden_size, &mut *nb)),
                v_proj_bias: vec_of(cfg.hidden_size, &mut *nb),
                out_proj: WeightStorage::F32(vec_of(cfg.hidden_size * cfg.hidden_size, &mut *nb)),
                out_proj_bias: vec_of(cfg.hidden_size, &mut *nb),
                ln2_gain: Arc::from(vec![1.0_f32; cfg.hidden_size]),
                ln2_bias: Arc::from(vec![0.0_f32; cfg.hidden_size]),
                fc1: WeightStorage::F32(vec_of(cfg.hidden_size * cfg.intermediate_size, &mut *nb)),
                fc1_bias: vec_of(cfg.intermediate_size, &mut *nb),
                fc2: WeightStorage::F32(vec_of(cfg.intermediate_size * cfg.hidden_size, &mut *nb)),
                fc2_bias: vec_of(cfg.hidden_size, &mut *nb),
            }
        ).collect();
        SiglipVisionWeights {
            patch_proj,
            patch_proj_bias,
            position_embedding,
            layers,
            post_ln_gain: Arc::from(vec![1.0_f32; cfg.hidden_size]),
            post_ln_bias: Arc::from(vec![0.0_f32; cfg.hidden_size]),
            head: None, // PaliGemma uses no pooling head.
        }
    }

    fn tiny_gemma_weights(cfg: &GemmaConfig) -> GemmaWeights {
        let mut s: u32 = 80808;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            attn_q_bias: if cfg.attention_bias { Some(vec_of(h, &mut *nb)) } else { None },
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_k_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_v_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            attn_norm_gain: Arc::from(vec![0.1_f32; h]),
            ffn_norm_gain: Arc::from(vec![0.1_f32; h]),
        }).collect();
        let final_norm_gain = Arc::from(vec![0.1_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        GemmaWeights { token_embedding, layers, final_norm_gain, output }
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
    fn forward_shape_and_finite() {
        let v_cfg = tiny_vision_cfg();
        let t_cfg = tiny_text_cfg();
        let mut s: u32 = 70707;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let mm_proj = WeightStorage::F32(vec_of(v_cfg.hidden_size * t_cfg.hidden_size, &mut *nb));
        let mm_proj_bias = vec_of(t_cfg.hidden_size, &mut *nb);
        let weights = PaligemmaWeights {
            vision: tiny_vision_weights(&v_cfg),
            mm_proj, mm_proj_bias,
            text: tiny_gemma_weights(&t_cfg),
        };
        let cfg = PaligemmaConfig {
            vision_config: v_cfg.clone(),
            text_config: t_cfg.clone(),
            projection_dim: t_cfg.hidden_size,
        };
        let model = PaligemmaModel { config: cfg, weights };

        let img = tiny_image(&v_cfg);
        let text_tokens = [1_u32, 2, 3];
        let logits = model.forward(&img, &text_tokens).unwrap();
        let expected_seq = v_cfg.num_patches() + text_tokens.len();
        assert_eq!(logits.shape().dims(), &[1, expected_seq, t_cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// Changing the image must alter the prediction at text
    /// positions (because image features are prepended to the
    /// text in the sequence, every text position attends back
    /// to image positions through the Gemma transformer).
    #[test]
    fn image_change_alters_text_logits() {
        let v_cfg = tiny_vision_cfg();
        let t_cfg = tiny_text_cfg();
        let mut s: u32 = 60606;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let mm_proj = WeightStorage::F32(vec_of(v_cfg.hidden_size * t_cfg.hidden_size, &mut *nb));
        let mm_proj_bias = vec_of(t_cfg.hidden_size, &mut *nb);
        let weights = PaligemmaWeights {
            vision: tiny_vision_weights(&v_cfg),
            mm_proj, mm_proj_bias,
            text: tiny_gemma_weights(&t_cfg),
        };
        let cfg = PaligemmaConfig {
            vision_config: v_cfg.clone(),
            text_config: t_cfg.clone(),
            projection_dim: t_cfg.hidden_size,
        };
        let model_a = PaligemmaModel { config: cfg.clone(), weights: weights.clone() };
        let model_b = PaligemmaModel { config: cfg, weights };

        let n_pix = 1 * v_cfg.num_channels * v_cfg.image_size * v_cfg.image_size;
        let img_a_data: Vec<f32> = (0..n_pix).map(|i| (i as f32 / n_pix as f32)).collect();
        let img_b_data: Vec<f32> = img_a_data.iter().map(|x| 1.0 - x).collect();
        let img_a = LazyTensor::from_f32(
            Arc::from(img_a_data),
            Shape::from_dims(&[1, v_cfg.num_channels, v_cfg.image_size, v_cfg.image_size]),
            &Device::cpu(),
        );
        let img_b = LazyTensor::from_f32(
            Arc::from(img_b_data),
            Shape::from_dims(&[1, v_cfg.num_channels, v_cfg.image_size, v_cfg.image_size]),
            &Device::cpu(),
        );
        let toks = [1_u32, 2, 3];
        let a = model_a.forward(&img_a, &toks).unwrap().realize_f32();
        let b = model_b.forward(&img_b, &toks).unwrap().realize_f32();
        // Extract text logits (positions AFTER num_patches).
        let np = v_cfg.num_patches();
        let v = t_cfg.vocab_size;
        let start_a = np * v;
        let start_b = np * v;
        let mut max_diff = 0.0_f32;
        for (x, y) in a[start_a..].iter().zip(b[start_b..].iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "image change must alter text-position logits, max_diff = {max_diff}");
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
                "lazy_paligemma_load_{}_{}.safetensors",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
            ));
            std::fs::write(&path, bytes).expect("write tempfile");
            path
        }

        fn build_tiny_safetensors(
            v_cfg: &SiglipVisionConfig,
            t_cfg: &GemmaConfig,
            proj_dim: usize,
        ) -> std::path::PathBuf {
            let mut map: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
            let mut s: u32 = 7777;
            let mut nxt = || -> f32 {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.01
            };
            let mut vec_n = |n: usize| -> Vec<f32> { (0..n).map(|_| nxt()).collect() };

            // SigLIP vision tower under vision_tower.vision_model.*
            let vp = "vision_tower.vision_model.";
            let h = v_cfg.hidden_size;
            let np = v_cfg.num_patches();
            let inter = v_cfg.intermediate_size;
            put(&mut map, &format!("{vp}embeddings.patch_embedding.weight"),
                &[h, v_cfg.num_channels, v_cfg.patch_size, v_cfg.patch_size],
                &vec_n(h * v_cfg.num_channels * v_cfg.patch_size * v_cfg.patch_size));
            put(&mut map, &format!("{vp}embeddings.patch_embedding.bias"),
                &[h], &vec_n(h));
            put(&mut map, &format!("{vp}embeddings.position_embedding.weight"),
                &[np, h], &vec_n(np * h));
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
                    &[inter, h], &vec_n(inter * h));
                put(&mut map, &format!("{p}.mlp.fc1.bias"),
                    &[inter], &vec_n(inter));
                put(&mut map, &format!("{p}.mlp.fc2.weight"),
                    &[h, inter], &vec_n(h * inter));
                put(&mut map, &format!("{p}.mlp.fc2.bias"),
                    &[h], &vec_n(h));
            }

            // MM projector: multi_modal_projector.linear.{weight,bias}
            put(&mut map, "multi_modal_projector.linear.weight",
                &[proj_dim, h], &vec_n(proj_dim * h));
            put(&mut map, "multi_modal_projector.linear.bias",
                &[proj_dim], &vec_n(proj_dim));

            // Gemma language model under language_model.*
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
            // lm_head omitted to exercise tied-fallback path.

            serialize_to_tempfile(&map)
        }

        #[test]
        fn round_trip_synthetic_safetensors() {
            let v_cfg = tiny_vision_cfg();
            let t_cfg = tiny_text_cfg();
            let proj_dim = t_cfg.hidden_size;
            let path = build_tiny_safetensors(&v_cfg, &t_cfg, proj_dim);
            let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path) }
                .expect("mmap safetensors");
            let cfg = PaligemmaConfig {
                vision_config: v_cfg.clone(),
                text_config: t_cfg.clone(),
                projection_dim: proj_dim,
            };
            let w = PaligemmaWeights::load_from_mmapped(&st, &cfg)
                .expect("PaligemmaWeights::load_from_mmapped");
            assert_eq!(w.vision.layers.len(), v_cfg.num_hidden_layers);
            assert_eq!(w.text.layers.len(), t_cfg.num_hidden_layers);
            assert_eq!(w.mm_proj_bias.len(), proj_dim);
            assert_eq!(w.text.token_embedding.len(), t_cfg.vocab_size * t_cfg.hidden_size);
            assert!(w.vision.head.is_none(), "PaliGemma uses SigLIP without pooling head");

            // Forward should succeed and produce finite logits.
            let model = PaligemmaModel { config: cfg, weights: w };
            let img = tiny_image(&v_cfg);
            let toks = [1_u32, 2, 3];
            let logits = model.forward(&img, &toks).unwrap().realize_f32();
            for v in &logits { assert!(v.is_finite()); }
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        #[ignore]
        fn from_hub_smoke_paligemma_3b_mix_224() {
            // Canonical: google/paligemma-3b-mix-224 — uses the HF layout
            // documented above.
        }
    }
}

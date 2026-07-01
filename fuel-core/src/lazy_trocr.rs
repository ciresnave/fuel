//! TrOCR — lazy port.
//!
//! Image → ViT encoder (delegated to `lazy_vit::VitModel` with
//! `classifier = None`) → BART-style decoder with cross-attention
//! → vocabulary logits.
//!
//! Decoder shape mirrors eager `TrOCRDecoderLayer`:
//!   Post-LN: `LN(x + sublayer(x))` at every sublayer.
//!   self_attn (with causal mask) → +res → LN
//!   encoder_attn (cross over encoder features) → +res → LN
//!   fc1 → activation → fc2 → +res → LN
//!
//! v1 scope:
//!   - Learned positional embeddings (default for HF presets).
//!   - Tied lm_head with `embed_tokens`.
//!   - `batch == 1`, F32, prefill only.
//!   - K/V projections in cross-attention take encoder's
//!     `cross_attention_hidden_size` (= ViT hidden) as input,
//!     project to decoder `d_model`.

use crate::lazy::{
    load_tensor_as_f32, load_transposed_matrix_preserve_dtype, LazyTensor, WeightStorage,
};
use crate::lazy_vit::{VitConfig, VitLayerWeights, VitModel, VitWeights};
use crate::Result;
use fuel_ir::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrocrActivation {
    Gelu,
    Relu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrocrDecoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    /// Encoder feature dim (= ViT hidden_size). Cross-attention's
    /// K/V projections take this as input.
    pub cross_attention_hidden_size: usize,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub activation_function: TrocrActivation,
    pub max_position_embeddings: usize,
    /// Offset added to positions when using learned embeddings
    /// (HF convention: typically 2).
    pub learned_pos_offset: usize,
    pub scale_embedding: bool,
    pub tie_word_embeddings: bool,
}

impl TrocrDecoderConfig {
    /// `microsoft/trocr-base-handwritten` decoder preset.
    pub fn trocr_base_handwritten() -> Self {
        Self {
            vocab_size: 50265,
            d_model: 1024,
            cross_attention_hidden_size: 768,
            decoder_layers: 12,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            activation_function: TrocrActivation::Gelu,
            max_position_embeddings: 512,
            learned_pos_offset: 2,
            scale_embedding: false,
            tie_word_embeddings: true,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }
}

#[derive(Debug, Clone)]
pub struct TrocrAttentionWeights {
    pub q_proj: WeightStorage,
    pub k_proj: WeightStorage,
    pub v_proj: WeightStorage,
    pub out_proj: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct TrocrDecoderLayerWeights {
    pub self_attn: TrocrAttentionWeights,
    pub self_attn_ln_gain: Arc<[f32]>,
    pub self_attn_ln_bias: Arc<[f32]>,
    pub encoder_attn: TrocrAttentionWeights,
    pub encoder_attn_ln_gain: Arc<[f32]>,
    pub encoder_attn_ln_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc2: WeightStorage,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct TrocrDecoderWeights {
    /// `[vocab_size, d_model]`.
    pub embed_tokens: Arc<[f32]>,
    /// `[max_position_embeddings + learned_pos_offset, d_model]`.
    pub embed_positions: Arc<[f32]>,
    pub layers: Vec<TrocrDecoderLayerWeights>,
    /// Optional separate output projection when `tie_word_embeddings = false`.
    pub output_projection: Option<WeightStorage>,
}

#[derive(Debug, Clone)]
pub struct TrocrModel {
    pub encoder_config: VitConfig,
    pub encoder_weights: VitWeights,
    pub decoder_config: TrocrDecoderConfig,
    pub decoder_weights: TrocrDecoderWeights,
}

impl TrocrModel {
    /// Image → encoder hidden states `(1, num_patches + 1, vit_hidden_size)`.
    /// Delegates to a `VitModel` with `classifier = None`.
    pub fn forward_encoder(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let mut weights = self.encoder_weights.clone();
        weights.classifier = None;
        let vit = VitModel { config: self.encoder_config.clone(), weights };
        vit.forward(image)
    }

    /// Encoder + decoder. Returns `(1, tgt_len, vocab_size)` logits.
    pub fn forward(&self, image: &LazyTensor, tgt_tokens: &[u32]) -> Result<LazyTensor> {
        let enc_out = self.forward_encoder(image)?;
        self.forward_decoder(tgt_tokens, &enc_out)
    }

    /// Decoder-only forward. `enc_out` is the graph anchor; built
    /// from `forward_encoder` or precomputed and reused across
    /// autoregressive steps.
    pub fn forward_decoder(
        &self,
        tgt_tokens: &[u32],
        enc_out: &LazyTensor,
    ) -> Result<LazyTensor> {
        let dcfg = &self.decoder_config;
        let dw = &self.decoder_weights;
        let tgt_len = tgt_tokens.len();
        assert!(tgt_len > 0, "tgt_tokens must be non-empty");
        assert_eq!(dw.layers.len(), dcfg.decoder_layers);

        // Token embedding lookup.
        let embed = enc_out.const_f32_like(
            Arc::clone(&dw.embed_tokens),
            Shape::from_dims(&[dcfg.vocab_size, dcfg.d_model]),
        );
        let ids = enc_out.const_u32_like(
            tgt_tokens.to_vec(), Shape::from_dims(&[tgt_len]),
        );
        let tok = embed
            .index_select(0_usize, &ids)?
            .reshape(Shape::from_dims(&[1, tgt_len, dcfg.d_model]))?;
        let tok = if dcfg.scale_embedding {
            tok.mul_scalar((dcfg.d_model as f64).sqrt())
        } else {
            tok
        };
        // Learned positional embedding: indices [offset .. offset + tgt_len).
        let pos_ids: Vec<u32> = (0..tgt_len)
            .map(|i| (i + dcfg.learned_pos_offset) as u32)
            .collect();
        let pos_table = enc_out.const_f32_like(
            Arc::clone(&dw.embed_positions),
            Shape::from_dims(&[
                dcfg.max_position_embeddings + dcfg.learned_pos_offset,
                dcfg.d_model,
            ]),
        );
        let pos_idx = enc_out.const_u32_like(
            pos_ids, Shape::from_dims(&[tgt_len]),
        );
        let pos = pos_table
            .index_select(0_usize, &pos_idx)?
            .reshape(Shape::from_dims(&[1, tgt_len, dcfg.d_model]))?;
        let mut x = tok.add(&pos)?;

        // Strict causal mask `[1, 1, tgt_len, tgt_len]` with -inf above diag.
        let mut mask_data = vec![0.0_f32; tgt_len * tgt_len];
        for i in 0..tgt_len {
            for j in (i + 1)..tgt_len {
                mask_data[i * tgt_len + j] = f32::NEG_INFINITY;
            }
        }
        let causal_mask = enc_out.const_f32_like(
            mask_data, Shape::from_dims(&[1, 1, tgt_len, tgt_len]),
        );

        for layer in &dw.layers {
            x = apply_decoder_layer(&x, layer, enc_out, &causal_mask, dcfg)?;
        }

        // lm_head: tied → matmul against embedding constant. Untied → output_projection.
        let logits = match &dw.output_projection {
            Some(w) => w.apply_linear(&x, dcfg.d_model, dcfg.vocab_size),
            None => {
                assert!(dcfg.tie_word_embeddings,
                    "output_projection missing but tie_word_embeddings = false");
                let lm_w = enc_out.const_f32_like(
                    Arc::clone(&dw.embed_tokens),
                    Shape::from_dims(&[dcfg.vocab_size, dcfg.d_model]),
                );
                x.matmul(&lm_w.transpose()?)?
            }
        };
        Ok(logits)
    }
}

// ---- Decoder layer ---------------------------------------------------------

fn apply_decoder_layer(
    x: &LazyTensor,
    w: &TrocrDecoderLayerWeights,
    enc_out: &LazyTensor,
    causal_mask: &LazyTensor,
    cfg: &TrocrDecoderConfig,
) -> Result<LazyTensor> {
    let d = cfg.d_model;
    let n_heads = cfg.decoder_attention_heads;
    let head_dim = cfg.head_dim();
    let kv_in = cfg.cross_attention_hidden_size;

    // Self-attention with causal mask.
    let self_attn_out = apply_attention(
        x, x, &w.self_attn, n_heads, head_dim, d, d, Some(causal_mask),
    )?;
    let h1 = x.add(&self_attn_out)?;
    let h1 = h1.layer_norm_affine(std::sync::Arc::clone(&w.self_attn_ln_gain), std::sync::Arc::clone(&w.self_attn_ln_bias), 1e-5)?;

    // Cross-attention: Q from decoder state, K/V from encoder output.
    let cross_attn_out = apply_attention(
        &h1, enc_out, &w.encoder_attn, n_heads, head_dim, d, kv_in, None,
    )?;
    let h2 = h1.add(&cross_attn_out)?;
    let h2 = h2.layer_norm_affine(std::sync::Arc::clone(&w.encoder_attn_ln_gain), std::sync::Arc::clone(&w.encoder_attn_ln_bias), 1e-5)?;

    // FFN: fc1 → activation → fc2.
    let h_ffn = w.fc1.apply_linear(&h2, d, cfg.decoder_ffn_dim);
    let h_ffn = match cfg.activation_function {
        TrocrActivation::Gelu => h_ffn.gelu(),
        TrocrActivation::Relu => h_ffn.relu(),
    };
    let h_ffn = w.fc2.apply_linear(&h_ffn, cfg.decoder_ffn_dim, d);
    let h3 = h2.add(&h_ffn)?;
    Ok(h3.layer_norm_affine(std::sync::Arc::clone(&w.final_ln_gain), std::sync::Arc::clone(&w.final_ln_bias), 1e-5)?)
}

#[allow(clippy::too_many_arguments)]
fn apply_attention(
    q_input: &LazyTensor,
    kv_input: &LazyTensor,
    w: &TrocrAttentionWeights,
    n_heads: usize,
    head_dim: usize,
    q_in_dim: usize,
    kv_in_dim: usize,
    mask: Option<&LazyTensor>,
) -> Result<LazyTensor> {
    let q_dims = q_input.shape();
    let q_dims = q_dims.dims();
    let batch = q_dims[0];
    let q_len = q_dims[1];
    let kv_dims = kv_input.shape();
    let kv_dims = kv_dims.dims();
    let kv_len = kv_dims[1];
    let d_model = n_heads * head_dim;

    let q = w.q_proj.apply_linear(q_input, q_in_dim, d_model);
    let k = w.k_proj.apply_linear(kv_input, kv_in_dim, d_model);
    let v = w.v_proj.apply_linear(kv_input, kv_in_dim, d_model);

    let scaling = 1.0_f64 / (head_dim as f64).sqrt();
    let q = q.mul_scalar(scaling);

    // (B, L, n_heads * head_dim) → (B, n_heads, L, head_dim)
    let q = q.split_heads(n_heads, head_dim)?;
    let k = k.split_heads(n_heads, head_dim)?;
    let v = v.split_heads(n_heads, head_dim)?;

    // Scores [B, n_heads, q_len, kv_len].
    let kt = k.permute([0, 1, 3, 2_usize])?;
    let mut scores = q.matmul(&kt)?;
    if let Some(m) = mask {
        let mb = m.broadcast_to(Shape::from_dims(&[batch, n_heads, q_len, kv_len]))?;
        scores = scores.add(&mb)?;
    }
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?;
    let _ = (batch, q_len, kv_len, d_model);
    let ctx = ctx.merge_heads()?;
    Ok(w.out_proj.apply_linear(&ctx, d_model, d_model))
}

// ---- Safetensors loader ----------------------------------------------------

/// Load HF ViT weights under `<prefix>` using the standard HF ViT naming:
///   - `embeddings.patch_embeddings.projection.{weight,bias}`
///   - `embeddings.cls_token`
///   - `embeddings.position_embeddings`
///   - `encoder.layer.{i}.attention.attention.{query,key,value}.{weight,bias}` (bias optional via cfg.qkv_bias)
///   - `encoder.layer.{i}.attention.output.dense.{weight,bias}`
///   - `encoder.layer.{i}.intermediate.dense.{weight,bias}`
///   - `encoder.layer.{i}.output.dense.{weight,bias}`
///   - `encoder.layer.{i}.layernorm_before.{weight,bias}`
///   - `encoder.layer.{i}.layernorm_after.{weight,bias}`
///   - `layernorm.{weight,bias}` (final post-encoder LN)
pub fn load_vit_weights(
    st: &crate::safetensors::MmapedSafetensors,
    cfg: &VitConfig,
    prefix: &str,
) -> Result<VitWeights> {
    let h = cfg.hidden_size;
    let np = cfg.num_patches();
    let inter = cfg.intermediate_size;
    let patch_proj = load_tensor_as_f32(
        st, &format!("{prefix}embeddings.patch_embeddings.projection.weight"),
    )?;
    let patch_proj_bias = load_tensor_as_f32(
        st, &format!("{prefix}embeddings.patch_embeddings.projection.bias"),
    )?;
    let cls_token = load_tensor_as_f32(st, &format!("{prefix}embeddings.cls_token"))?;
    let position_embeddings = load_tensor_as_f32(
        st, &format!("{prefix}embeddings.position_embeddings"),
    )?;
    if position_embeddings.len() != (np + 1) * h {
        crate::bail!(
            "{prefix}embeddings.position_embeddings: {} elts, expected {}",
            position_embeddings.len(), (np + 1) * h,
        );
    }
    let final_ln_gain = load_tensor_as_f32(st, &format!("{prefix}layernorm.weight"))?;
    let final_ln_bias = load_tensor_as_f32(st, &format!("{prefix}layernorm.bias"))?;

    let mut layers: Vec<VitLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("{prefix}encoder.layer.{i}");
        let ln_before_gain = load_tensor_as_f32(st, &format!("{p}.layernorm_before.weight"))?;
        let ln_before_bias = load_tensor_as_f32(st, &format!("{p}.layernorm_before.bias"))?;
        let ln_after_gain = load_tensor_as_f32(st, &format!("{p}.layernorm_after.weight"))?;
        let ln_after_bias = load_tensor_as_f32(st, &format!("{p}.layernorm_after.bias"))?;
        let q_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.attention.attention.query.weight"), h, h,
        )?;
        let k_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.attention.attention.key.weight"), h, h,
        )?;
        let v_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.attention.attention.value.weight"), h, h,
        )?;
        let (q_proj_bias, k_proj_bias, v_proj_bias) = if cfg.qkv_bias {
            (
                load_tensor_as_f32(st, &format!("{p}.attention.attention.query.bias")).ok().map(Arc::from),
                load_tensor_as_f32(st, &format!("{p}.attention.attention.key.bias")).ok().map(Arc::from),
                load_tensor_as_f32(st, &format!("{p}.attention.attention.value.bias")).ok().map(Arc::from),
            )
        } else {
            (None, None, None)
        };
        let attn_output_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.attention.output.dense.weight"), h, h,
        )?;
        let attn_output_proj_bias = load_tensor_as_f32(
            st, &format!("{p}.attention.output.dense.bias"),
        )?;
        let intermediate_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.intermediate.dense.weight"), inter, h,
        )?;
        let intermediate_proj_bias = load_tensor_as_f32(
            st, &format!("{p}.intermediate.dense.bias"),
        )?;
        let mlp_output_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.output.dense.weight"), h, inter,
        )?;
        let mlp_output_proj_bias = load_tensor_as_f32(
            st, &format!("{p}.output.dense.bias"),
        )?;
        layers.push(VitLayerWeights {
            ln_before_gain: Arc::from(ln_before_gain),
            ln_before_bias: Arc::from(ln_before_bias),
            q_proj, q_proj_bias,
            k_proj, k_proj_bias,
            v_proj, v_proj_bias,
            attn_output_proj,
            attn_output_proj_bias: Arc::from(attn_output_proj_bias),
            ln_after_gain: Arc::from(ln_after_gain),
            ln_after_bias: Arc::from(ln_after_bias),
            intermediate_proj,
            intermediate_proj_bias: Arc::from(intermediate_proj_bias),
            mlp_output_proj,
            mlp_output_proj_bias: Arc::from(mlp_output_proj_bias),
        });
    }

    Ok(VitWeights {
        patch_proj: Arc::from(patch_proj),
        patch_proj_bias: Arc::from(patch_proj_bias),
        cls_token: Arc::from(cls_token),
        position_embeddings: Arc::from(position_embeddings),
        layers,
        final_ln_gain: Arc::from(final_ln_gain),
        final_ln_bias: Arc::from(final_ln_bias),
        classifier: None,
    })
}

/// Load TrOCR decoder weights from a safetensors file using the HF
/// `decoder.model.decoder.*` prefix:
///   - `embed_tokens.weight`
///   - `embed_positions.weight`
///   - `layers.{i}.self_attn.{q,k,v,out}_proj.weight`
///   - `layers.{i}.self_attn_layer_norm.{weight,bias}`
///   - `layers.{i}.encoder_attn.{q,k,v,out}_proj.weight`
///   - `layers.{i}.encoder_attn_layer_norm.{weight,bias}`
///   - `layers.{i}.fc1.weight`, `layers.{i}.fc2.weight`
///   - `layers.{i}.final_layer_norm.{weight,bias}`
///   - `decoder.output_projection.weight` (when tie_word_embeddings is false)
pub fn load_trocr_decoder_weights(
    st: &crate::safetensors::MmapedSafetensors,
    cfg: &TrocrDecoderConfig,
) -> Result<TrocrDecoderWeights> {
    let pfx = "decoder.model.decoder.";
    let d = cfg.d_model;
    let kv_in = cfg.cross_attention_hidden_size;
    let embed_tokens = load_tensor_as_f32(
        st, &format!("{pfx}embed_tokens.weight"),
    )?;
    let embed_positions = load_tensor_as_f32(
        st, &format!("{pfx}embed_positions.weight"),
    )?;
    let mut layers: Vec<TrocrDecoderLayerWeights> = Vec::with_capacity(cfg.decoder_layers);
    for i in 0..cfg.decoder_layers {
        let p = format!("{pfx}layers.{i}");
        let self_attn = TrocrAttentionWeights {
            q_proj: load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.q_proj.weight"), d, d,
            )?,
            k_proj: load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.k_proj.weight"), d, d,
            )?,
            v_proj: load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.v_proj.weight"), d, d,
            )?,
            out_proj: load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.out_proj.weight"), d, d,
            )?,
        };
        let self_attn_ln_gain = load_tensor_as_f32(
            st, &format!("{p}.self_attn_layer_norm.weight"),
        )?;
        let self_attn_ln_bias = load_tensor_as_f32(
            st, &format!("{p}.self_attn_layer_norm.bias"),
        )?;
        let encoder_attn = TrocrAttentionWeights {
            q_proj: load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.encoder_attn.q_proj.weight"), d, d,
            )?,
            k_proj: load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.encoder_attn.k_proj.weight"), d, kv_in,
            )?,
            v_proj: load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.encoder_attn.v_proj.weight"), d, kv_in,
            )?,
            out_proj: load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.encoder_attn.out_proj.weight"), d, d,
            )?,
        };
        let encoder_attn_ln_gain = load_tensor_as_f32(
            st, &format!("{p}.encoder_attn_layer_norm.weight"),
        )?;
        let encoder_attn_ln_bias = load_tensor_as_f32(
            st, &format!("{p}.encoder_attn_layer_norm.bias"),
        )?;
        let fc1 = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.fc1.weight"), cfg.decoder_ffn_dim, d,
        )?;
        let fc2 = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.fc2.weight"), d, cfg.decoder_ffn_dim,
        )?;
        let final_ln_gain = load_tensor_as_f32(
            st, &format!("{p}.final_layer_norm.weight"),
        )?;
        let final_ln_bias = load_tensor_as_f32(
            st, &format!("{p}.final_layer_norm.bias"),
        )?;
        layers.push(TrocrDecoderLayerWeights {
            self_attn,
            self_attn_ln_gain: Arc::from(self_attn_ln_gain),
            self_attn_ln_bias: Arc::from(self_attn_ln_bias),
            encoder_attn,
            encoder_attn_ln_gain: Arc::from(encoder_attn_ln_gain),
            encoder_attn_ln_bias: Arc::from(encoder_attn_ln_bias),
            fc1, fc2,
            final_ln_gain: Arc::from(final_ln_gain),
            final_ln_bias: Arc::from(final_ln_bias),
        });
    }
    let output_projection = if cfg.tie_word_embeddings {
        None
    } else {
        Some(load_transposed_matrix_preserve_dtype(
            st, "decoder.output_projection.weight", cfg.vocab_size, d,
        )?)
    };
    Ok(TrocrDecoderWeights {
        embed_tokens: Arc::from(embed_tokens),
        embed_positions: Arc::from(embed_positions),
        layers,
        output_projection,
    })
}

impl TrocrModel {
    /// Load a TrOCR model (ViT encoder + BART-style decoder) from a
    /// HuggingFace safetensors file. Encoder weights live under
    /// `encoder.*` and decoder under `decoder.model.decoder.*` per the
    /// `microsoft/trocr-*` convention.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        encoder_config: VitConfig,
        decoder_config: TrocrDecoderConfig,
    ) -> Result<Self> {
        let encoder_weights = load_vit_weights(st, &encoder_config, "encoder.")?;
        let decoder_weights = load_trocr_decoder_weights(st, &decoder_config)?;
        Ok(Self {
            encoder_config,
            encoder_weights,
            decoder_config,
            decoder_weights,
        })
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy_vit::{VitActivation, VitConfig, VitLayerWeights, VitWeights};
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

    fn tiny_vit_config() -> VitConfig {
        VitConfig {
            hidden_size: 8, num_hidden_layers: 2, num_attention_heads: 2,
            intermediate_size: 16,
            hidden_activation: VitActivation::Gelu, layer_norm_eps: 1e-5,
            image_size: 8, patch_size: 4, num_channels: 3,
            qkv_bias: true,
        }
    }

    fn tiny_vit_weights(cfg: &VitConfig) -> VitWeights {
        let mut nb = rng_seed(101);
        let h = cfg.hidden_size;
        let np = cfg.num_patches();
        let layers: Vec<VitLayerWeights> = (0..cfg.num_hidden_layers).map(|_| VitLayerWeights {
            ln_before_gain: Arc::from(vec![1.0_f32; h]),
            ln_before_bias: Arc::from(vec![0.0_f32; h]),
            q_proj: ws(h * h, &mut nb), q_proj_bias: Some(vec_of(h, &mut nb)),
            k_proj: ws(h * h, &mut nb), k_proj_bias: Some(vec_of(h, &mut nb)),
            v_proj: ws(h * h, &mut nb), v_proj_bias: Some(vec_of(h, &mut nb)),
            attn_output_proj: ws(h * h, &mut nb),
            attn_output_proj_bias: vec_of(h, &mut nb),
            ln_after_gain: Arc::from(vec![1.0_f32; h]),
            ln_after_bias: Arc::from(vec![0.0_f32; h]),
            intermediate_proj: ws(h * cfg.intermediate_size, &mut nb),
            intermediate_proj_bias: vec_of(cfg.intermediate_size, &mut nb),
            mlp_output_proj: ws(cfg.intermediate_size * h, &mut nb),
            mlp_output_proj_bias: vec_of(h, &mut nb),
        }).collect();
        VitWeights {
            patch_proj: vec_of(h * cfg.num_channels * cfg.patch_size * cfg.patch_size, &mut nb),
            patch_proj_bias: vec_of(h, &mut nb),
            cls_token: vec_of(h, &mut nb),
            position_embeddings: vec_of((np + 1) * h, &mut nb),
            layers,
            final_ln_gain: Arc::from(vec![1.0_f32; h]),
            final_ln_bias: Arc::from(vec![0.0_f32; h]),
            classifier: None,
        }
    }

    fn tiny_trocr_config(vit_hidden: usize) -> TrocrDecoderConfig {
        TrocrDecoderConfig {
            vocab_size: 16, d_model: 8,
            cross_attention_hidden_size: vit_hidden,
            decoder_layers: 2, decoder_attention_heads: 2,
            decoder_ffn_dim: 16,
            activation_function: TrocrActivation::Gelu,
            max_position_embeddings: 32, learned_pos_offset: 2,
            scale_embedding: false,
            tie_word_embeddings: true,
        }
    }

    fn tiny_trocr_weights(dcfg: &TrocrDecoderConfig) -> TrocrDecoderWeights {
        let mut nb = rng_seed(202);
        let d = dcfg.d_model;
        let kv_in = dcfg.cross_attention_hidden_size;
        let layers: Vec<TrocrDecoderLayerWeights> = (0..dcfg.decoder_layers).map(|_| {
            TrocrDecoderLayerWeights {
                self_attn: TrocrAttentionWeights {
                    q_proj: ws(d * d, &mut nb),
                    k_proj: ws(d * d, &mut nb),
                    v_proj: ws(d * d, &mut nb),
                    out_proj: ws(d * d, &mut nb),
                },
                self_attn_ln_gain: Arc::from(vec![1.0_f32; d]),
                self_attn_ln_bias: Arc::from(vec![0.0_f32; d]),
                encoder_attn: TrocrAttentionWeights {
                    q_proj: ws(d * d, &mut nb),
                    k_proj: ws(kv_in * d, &mut nb),
                    v_proj: ws(kv_in * d, &mut nb),
                    out_proj: ws(d * d, &mut nb),
                },
                encoder_attn_ln_gain: Arc::from(vec![1.0_f32; d]),
                encoder_attn_ln_bias: Arc::from(vec![0.0_f32; d]),
                fc1: ws(d * dcfg.decoder_ffn_dim, &mut nb),
                fc2: ws(dcfg.decoder_ffn_dim * d, &mut nb),
                final_ln_gain: Arc::from(vec![1.0_f32; d]),
                final_ln_bias: Arc::from(vec![0.0_f32; d]),
            }
        }).collect();
        TrocrDecoderWeights {
            embed_tokens: vec_of(dcfg.vocab_size * d, &mut nb),
            embed_positions: vec_of(
                (dcfg.max_position_embeddings + dcfg.learned_pos_offset) * d, &mut nb,
            ),
            layers,
            output_projection: None,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let vcfg = tiny_vit_config();
        let vw = tiny_vit_weights(&vcfg);
        let dcfg = tiny_trocr_config(vcfg.hidden_size);
        let dw = tiny_trocr_weights(&dcfg);
        let model = TrocrModel {
            encoder_config: vcfg.clone(),
            encoder_weights: vw,
            decoder_config: dcfg.clone(),
            decoder_weights: dw,
        };
        let image: Vec<f32> = (0..(3 * 8 * 8)).map(|i| (i as f32) * 0.01).collect();
        let img = LazyTensor::from_f32(
            image, Shape::from_dims(&[1, 3, 8, 8]), &Device::cpu(),
        );
        let tgt = [1_u32, 2, 3];
        let logits = model.forward(&img, &tgt).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tgt.len(), dcfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    /// `forward_decoder(tgt, forward_encoder(image))` matches
    /// `forward(image, tgt)` — proves the split path computes
    /// the same graph.
    #[test]
    fn forward_decoder_matches_forward() {
        let vcfg = tiny_vit_config();
        let vw = tiny_vit_weights(&vcfg);
        let dcfg = tiny_trocr_config(vcfg.hidden_size);
        let dw = tiny_trocr_weights(&dcfg);
        let model = TrocrModel {
            encoder_config: vcfg.clone(),
            encoder_weights: vw,
            decoder_config: dcfg.clone(),
            decoder_weights: dw,
        };
        let image: Vec<f32> = (0..(3 * 8 * 8)).map(|i| (i as f32) * 0.01).collect();
        let img = LazyTensor::from_f32(
            image, Shape::from_dims(&[1, 3, 8, 8]), &Device::cpu(),
        );
        let tgt = [1_u32, 2, 3];
        let full = model.forward(&img, &tgt).unwrap().realize_f32();
        let enc = model.forward_encoder(&img).unwrap();
        let split = model.forward_decoder(&tgt, &enc).unwrap().realize_f32();
        assert_eq!(full.len(), split.len());
        for (a, b) in full.iter().zip(split.iter()) {
            assert!((a - b).abs() < 1e-5,
                "split path must match full forward: {a} vs {b}");
        }
    }

    /// Cross-attention must condition the decoder output on the
    /// encoder output. Different images must yield different
    /// logits for the same target tokens.
    #[test]
    fn cross_attention_is_wired() {
        let vcfg = tiny_vit_config();
        let vw = tiny_vit_weights(&vcfg);
        let dcfg = tiny_trocr_config(vcfg.hidden_size);
        let dw = tiny_trocr_weights(&dcfg);
        let model = TrocrModel {
            encoder_config: vcfg.clone(),
            encoder_weights: vw,
            decoder_config: dcfg.clone(),
            decoder_weights: dw,
        };
        let img_a = LazyTensor::from_f32(
            (0..(3 * 8 * 8)).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 8, 8]), &Device::cpu(),
        );
        let img_b = LazyTensor::from_f32(
            (0..(3 * 8 * 8)).map(|i| i as f32 * 0.01 + 0.5).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 8, 8]), &Device::cpu(),
        );
        let tgt = [1_u32, 2, 3];
        let a = model.forward(&img_a, &tgt).unwrap().realize_f32();
        let b = model.forward(&img_b, &tgt).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "cross-attention must condition decoder on encoder, max_diff = {max_diff}");
    }

    /// Causal mask: changing target token at position t must NOT
    /// alter logits at positions < t.
    #[test]
    fn causal_mask_enforced() {
        let vcfg = tiny_vit_config();
        let vw = tiny_vit_weights(&vcfg);
        let dcfg = tiny_trocr_config(vcfg.hidden_size);
        let dw = tiny_trocr_weights(&dcfg);
        let model = TrocrModel {
            encoder_config: vcfg.clone(),
            encoder_weights: vw,
            decoder_config: dcfg.clone(),
            decoder_weights: dw,
        };
        let img = LazyTensor::from_f32(
            vec![0.1_f32; 3 * 8 * 8],
            Shape::from_dims(&[1, 3, 8, 8]), &Device::cpu(),
        );
        let tgt_a = [1_u32, 2, 3, 4];
        let tgt_b = [1_u32, 2, 3, 9]; // last token changed
        let a = model.forward(&img, &tgt_a).unwrap().realize_f32();
        let b = model.forward(&img, &tgt_b).unwrap().realize_f32();
        let v = dcfg.vocab_size;
        for t in 0..3 {
            for c in 0..v {
                let i = t * v + c;
                assert!((a[i] - b[i]).abs() < 1e-5,
                    "causal mask violated at t={t}, c={c}: {} vs {}", a[i], b[i]);
            }
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
                "lazy_trocr_load_{}_{}.safetensors",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
            ));
            std::fs::write(&path, bytes).expect("write tempfile");
            path
        }

        fn build_tiny_safetensors(
            v_cfg: &VitConfig,
            d_cfg: &TrocrDecoderConfig,
        ) -> std::path::PathBuf {
            let mut map: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
            let mut s: u32 = 5511;
            let mut nxt = || -> f32 {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.01
            };
            let mut vec_n = |n: usize| -> Vec<f32> { (0..n).map(|_| nxt()).collect() };

            // ViT encoder under encoder.*
            let ep = "encoder.";
            let h = v_cfg.hidden_size;
            let np = v_cfg.num_patches();
            put(&mut map, &format!("{ep}embeddings.patch_embeddings.projection.weight"),
                &[h, v_cfg.num_channels, v_cfg.patch_size, v_cfg.patch_size],
                &vec_n(h * v_cfg.num_channels * v_cfg.patch_size * v_cfg.patch_size));
            put(&mut map, &format!("{ep}embeddings.patch_embeddings.projection.bias"),
                &[h], &vec_n(h));
            put(&mut map, &format!("{ep}embeddings.cls_token"),
                &[1, 1, h], &vec_n(h));
            put(&mut map, &format!("{ep}embeddings.position_embeddings"),
                &[1, np + 1, h], &vec_n((np + 1) * h));
            put(&mut map, &format!("{ep}layernorm.weight"), &[h], &vec_n(h));
            put(&mut map, &format!("{ep}layernorm.bias"),   &[h], &vec_n(h));
            for i in 0..v_cfg.num_hidden_layers {
                let p = format!("{ep}encoder.layer.{i}");
                put(&mut map, &format!("{p}.layernorm_before.weight"), &[h], &vec_n(h));
                put(&mut map, &format!("{p}.layernorm_before.bias"),   &[h], &vec_n(h));
                put(&mut map, &format!("{p}.layernorm_after.weight"),  &[h], &vec_n(h));
                put(&mut map, &format!("{p}.layernorm_after.bias"),    &[h], &vec_n(h));
                for kn in &["query", "key", "value"] {
                    put(&mut map, &format!("{p}.attention.attention.{kn}.weight"),
                        &[h, h], &vec_n(h * h));
                    if v_cfg.qkv_bias {
                        put(&mut map, &format!("{p}.attention.attention.{kn}.bias"),
                            &[h], &vec_n(h));
                    }
                }
                put(&mut map, &format!("{p}.attention.output.dense.weight"),
                    &[h, h], &vec_n(h * h));
                put(&mut map, &format!("{p}.attention.output.dense.bias"),
                    &[h], &vec_n(h));
                put(&mut map, &format!("{p}.intermediate.dense.weight"),
                    &[v_cfg.intermediate_size, h], &vec_n(v_cfg.intermediate_size * h));
                put(&mut map, &format!("{p}.intermediate.dense.bias"),
                    &[v_cfg.intermediate_size], &vec_n(v_cfg.intermediate_size));
                put(&mut map, &format!("{p}.output.dense.weight"),
                    &[h, v_cfg.intermediate_size], &vec_n(h * v_cfg.intermediate_size));
                put(&mut map, &format!("{p}.output.dense.bias"),
                    &[h], &vec_n(h));
            }

            // Decoder under decoder.model.decoder.*
            let pfx = "decoder.model.decoder.";
            let d = d_cfg.d_model;
            let kv_in = d_cfg.cross_attention_hidden_size;
            put(&mut map, &format!("{pfx}embed_tokens.weight"),
                &[d_cfg.vocab_size, d], &vec_n(d_cfg.vocab_size * d));
            put(&mut map, &format!("{pfx}embed_positions.weight"),
                &[d_cfg.max_position_embeddings + d_cfg.learned_pos_offset, d],
                &vec_n((d_cfg.max_position_embeddings + d_cfg.learned_pos_offset) * d));
            for i in 0..d_cfg.decoder_layers {
                let p = format!("{pfx}layers.{i}");
                for proj in &["q_proj", "k_proj", "v_proj", "out_proj"] {
                    put(&mut map, &format!("{p}.self_attn.{proj}.weight"),
                        &[d, d], &vec_n(d * d));
                }
                put(&mut map, &format!("{p}.self_attn_layer_norm.weight"),
                    &[d], &vec_n(d));
                put(&mut map, &format!("{p}.self_attn_layer_norm.bias"),
                    &[d], &vec_n(d));
                // encoder_attn: q/out project from d to d, k/v project from kv_in to d.
                put(&mut map, &format!("{p}.encoder_attn.q_proj.weight"),
                    &[d, d], &vec_n(d * d));
                put(&mut map, &format!("{p}.encoder_attn.k_proj.weight"),
                    &[d, kv_in], &vec_n(d * kv_in));
                put(&mut map, &format!("{p}.encoder_attn.v_proj.weight"),
                    &[d, kv_in], &vec_n(d * kv_in));
                put(&mut map, &format!("{p}.encoder_attn.out_proj.weight"),
                    &[d, d], &vec_n(d * d));
                put(&mut map, &format!("{p}.encoder_attn_layer_norm.weight"),
                    &[d], &vec_n(d));
                put(&mut map, &format!("{p}.encoder_attn_layer_norm.bias"),
                    &[d], &vec_n(d));
                put(&mut map, &format!("{p}.fc1.weight"),
                    &[d_cfg.decoder_ffn_dim, d], &vec_n(d_cfg.decoder_ffn_dim * d));
                put(&mut map, &format!("{p}.fc2.weight"),
                    &[d, d_cfg.decoder_ffn_dim], &vec_n(d * d_cfg.decoder_ffn_dim));
                put(&mut map, &format!("{p}.final_layer_norm.weight"),
                    &[d], &vec_n(d));
                put(&mut map, &format!("{p}.final_layer_norm.bias"),
                    &[d], &vec_n(d));
            }
            serialize_to_tempfile(&map)
        }

        #[test]
        fn round_trip_synthetic_safetensors() {
            let v_cfg = tiny_vit_config();
            let d_cfg = tiny_trocr_config(v_cfg.hidden_size);
            let path = build_tiny_safetensors(&v_cfg, &d_cfg);
            let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path) }
                .expect("mmap safetensors");
            let model = TrocrModel::load_from_mmapped(&st, v_cfg.clone(), d_cfg.clone())
                .expect("TrocrModel::load_from_mmapped");
            assert_eq!(model.encoder_weights.layers.len(), v_cfg.num_hidden_layers);
            assert_eq!(model.decoder_weights.layers.len(), d_cfg.decoder_layers);
            assert!(model.decoder_weights.output_projection.is_none(),
                "tie_word_embeddings=true => output_projection should be None");
            let image: Vec<f32> = (0..(3 * 8 * 8)).map(|i| (i as f32) * 0.01).collect();
            let img = LazyTensor::from_f32(
                image, Shape::from_dims(&[1, 3, 8, 8]), &crate::Device::cpu(),
            );
            let logits = model.forward(&img, &[1_u32, 2, 3]).unwrap().realize_f32();
            for v in &logits { assert!(v.is_finite()); }
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        #[ignore]
        fn from_hub_smoke_trocr_base_handwritten() {
            // Canonical: microsoft/trocr-base-handwritten — encoder.* + decoder.model.decoder.*
        }
    }
}

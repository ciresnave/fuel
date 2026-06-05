//! NV-Embed v2 (NVIDIA 2024) ported to the lazy-graph API.
//!
//! NV-Embed v2 is a text-embedding model built on a Mistral 7B
//! backbone (run in **bidirectional** mode — no causal mask —
//! using only the pad-token attention mask) topped with a
//! Perceiver-style **latent attention pooler**: 512 learned
//! latent tokens that the backbone hidden states attend INTO
//! as keys/values. The output of the latent cross-attention is
//! summed back into the original hidden states (residual), fed
//! through a **GeGLU FFN** with another residual, then mean-
//! pooled over the sequence (mask-weighted) to produce a
//! single embedding per input.
//!
//! Composition:
//!
//! ```text
//!   tokens, attn_mask
//!     → MistralModel::forward_hidden_embeds_with_mask(embeds, bidirectional_mask)
//!     → cross_attn_norm  → cross_attn(Q=hiddens, K/V=norm(latents)) + hiddens
//!     → ff_norm          → GeGLU(↑)                                  + cross_hiddens
//!     → mean_pool(hidden, attn_mask)
//!     → (B, hidden)
//! ```
//!
//! The bidirectional attention mask is built from the padding
//! mask via `(1, 1, tgt, src)` with `0` at "keep" positions
//! and `-inf` at "mask" positions. There is NO causal triangle.
//!
//! # Cross attention layout (Perceiver-style)
//!
//! - **Query** projects from `hiddens` (B, seq, hidden).
//! - **Key, Value** both project from `latents` (B, 512, hidden).
//! - The result has Q-shape `(B, seq, hidden)` — i.e. the
//!   sequence-length output, NOT the latent count. The latents
//!   form a learned KV bank that every token attends over.
//! - The eager port heads-split via `reshape_heads_to_batch_dim`
//!   (B * H, seq, head_dim). The lazy port uses the standard
//!   per-head permute pattern instead.
//!
//! # GeGLU FFN
//!
//! `proj(x).chunk(2)` → `(hidden_chunk * gate_chunk).gelu_erf()`
//! followed by `down(...)`. The eager port reshapes the chunked
//! output as `gate = first half (size dim_out)`,
//! `value = second half`. The lazy port reproduces this with
//! a single `[hidden, 2 * inner]` projection sliced on the
//! last dim.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. The backbone is the Mistral
//! 7B preset (4096 hidden, 32 layers, 32 heads × 8 KV, head_dim
//! 128, theta 10k, NO sliding window — bidirectional encoder
//! semantics). The latent attention head adds 8 heads × 512 dim
//! per head = 4096 dim, projected back to 4096. Returns L2-
//! normalized embeddings `(1, hidden_size)`. The output dim is
//! NOT Matryoshka — NV-Embed v2 ships a fixed 4096-d embedding.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_mistral::{MistralConfig, MistralModel, MistralWeights};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct NvEmbedV2Config {
    pub backbone: MistralConfig,
    /// Number of learned latent tokens (eager: 512).
    pub num_latents: usize,
    /// Latent-attention head count (eager: 8).
    pub latent_heads: usize,
    /// Per-head dimension for the latent attention. The
    /// eager port uses `4096` here (same as hidden) — the
    /// product `latent_heads * latent_head_dim` need not
    /// equal `hidden_size`; the cross-attn output projects
    /// back via `to_out`.
    pub latent_head_dim: usize,
    /// GeGLU inner dim (eager: `mult = 4`, so `4 * hidden`).
    pub ff_mult: usize,
    pub layer_norm_eps: f64,
}

impl NvEmbedV2Config {
    /// `nvidia/NV-Embed-v2` preset (approximate; actual
    /// `config.json` from HuggingFace overrides).
    pub fn nv_embed_v2() -> Self {
        let backbone = MistralConfig {
            vocab_size: 32_000,
            hidden_size: 4_096,
            intermediate_size: 14_336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 32_768,
            // Bidirectional encoder semantics: no causal sliding window.
            // The caller-supplied bidirectional mask is what actually
            // runs (set when calling forward_hidden_embeds_with_mask),
            // but we still need a config value here for the standard
            // forward path that this model never uses.
            sliding_window: None,
        };
        Self {
            backbone,
            num_latents: 512,
            latent_heads: 8,
            latent_head_dim: 4_096,
            ff_mult: 4,
            layer_norm_eps: 1e-5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NvEmbedV2Weights {
    pub backbone: MistralWeights,
    /// `[num_latents, hidden_size]`. The Perceiver-style
    /// learned KV bank.
    pub latents: Arc<[f32]>,
    /// LayerNorm gain/bias on the hidden states BEFORE the
    /// cross-attention Q projection.
    pub cross_attn_norm_gain: Arc<[f32]>,
    pub cross_attn_norm_bias: Arc<[f32]>,
    /// LayerNorm gain/bias on the latents BEFORE the
    /// cross-attention K/V projection.
    pub cross_attn_context_norm_gain: Arc<[f32]>,
    pub cross_attn_context_norm_bias: Arc<[f32]>,
    /// Cross attention projections. All `no_bias`.
    /// `[hidden_size, latent_heads * latent_head_dim]`.
    pub to_q: WeightStorage,
    /// `[hidden_size, 2 * latent_heads * latent_head_dim]`
    /// (fused K/V).
    pub to_kv: WeightStorage,
    /// `[latent_heads * latent_head_dim, hidden_size]`.
    pub to_out: WeightStorage,
    /// LayerNorm before the GeGLU FFN.
    pub ff_norm_gain: Arc<[f32]>,
    pub ff_norm_bias: Arc<[f32]>,
    /// `[hidden_size, 2 * (hidden_size * ff_mult)]` GeGLU up-projection.
    pub ff_proj: WeightStorage,
    /// `[hidden_size * ff_mult, hidden_size]` down-projection.
    pub ff_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct NvEmbedV2Model {
    pub config: NvEmbedV2Config,
    pub weights: NvEmbedV2Weights,
}

impl NvEmbedV2Model {
    /// Run a forward pass with an attention mask `(seq,)` of
    /// `1` for keep and `0` for pad. Returns L2-normalized
    /// embeddings `(1, hidden_size)`.
    pub fn forward(&self, tokens: &[u32], attention_mask: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let bcfg = &cfg.backbone;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "NvEmbedV2Model::forward: tokens must be non-empty");
        assert_eq!(attention_mask.len(), seq,
            "attention_mask length must equal tokens length");

        // ---- Embedding lookup --------------------------------------------
        let embed = LazyTensor::from_f32(
            self.weights.backbone.token_embedding.clone(),
            Shape::from_dims(&[bcfg.vocab_size, bcfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let embeds = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, bcfg.hidden_size]))?;

        // ---- Build bidirectional 4-D pad mask -----------------------------
        // shape (1, 1, seq, seq). `keep & keep` → 0; either-pad → -inf.
        // The eager port builds `(1 - mask) * f32::MIN` from a 2-D mask
        // expanded to (B, 1, tgt, src). Lazy version computes it as a
        // const since the pad layout is known at graph-build time.
        let bidirectional_mask = self.build_bidirectional_pad_mask(&embeds, attention_mask);

        // ---- Run Mistral backbone in bidirectional mode -------------------
        let backbone = MistralModel {
            config: bcfg.clone(), weights: self.weights.backbone.clone(),
        };
        let hidden = backbone.forward_hidden_embeds_with_mask(
            &embeds, &bidirectional_mask, 0,
        )?;

        // ---- Latent attention head (Perceiver-style) ----------------------
        // norm(hidden) → Q; norm(latents) → K, V.
        let hidden_normed = hidden.layer_norm_affine(std::sync::Arc::clone(&self.weights.cross_attn_norm_gain), std::sync::Arc::clone(&self.weights.cross_attn_norm_bias), cfg.layer_norm_eps)?;
        let latents = embeds.const_f32_like(
            Arc::clone(&self.weights.latents),
            Shape::from_dims(&[cfg.num_latents, bcfg.hidden_size]),
        );
        let latents = latents
            .reshape(Shape::from_dims(&[1, cfg.num_latents, bcfg.hidden_size]))?
            .broadcast_to(Shape::from_dims(&[batch, cfg.num_latents, bcfg.hidden_size]))?;
        let latents_normed = latents.layer_norm_affine(std::sync::Arc::clone(&self.weights.cross_attn_context_norm_gain), std::sync::Arc::clone(&self.weights.cross_attn_context_norm_bias), cfg.layer_norm_eps)?;
        let inner = cfg.latent_heads * cfg.latent_head_dim;
        let q = self.weights.to_q.apply_linear(&hidden_normed, bcfg.hidden_size, inner);
        let kv = self.weights.to_kv.apply_linear(&latents_normed, bcfg.hidden_size, 2 * inner);
        let k = kv.slice(2_usize, 0, inner)?;
        let v = kv.slice(2_usize, inner, inner)?;
        // Heads split: (batch, len, heads, head_dim) → permute(0, 2, 1, 3).
        let _ = batch;
        let q = q.split_heads(cfg.latent_heads, cfg.latent_head_dim)?;
        let k = k.split_heads(cfg.latent_heads, cfg.latent_head_dim)?;
        let v = v.split_heads(cfg.latent_heads, cfg.latent_head_dim)?;
        let scale = 1.0_f64 / (cfg.latent_head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose()?)?.mul_scalar(scale);
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?; // (batch, heads, seq, head_dim)
        let merged = ctx.merge_heads()?;
        let cross_out = self.weights.to_out.apply_linear(&merged, inner, bcfg.hidden_size);
        // Residual: hidden + cross_out.
        let cross_hidden = hidden.add(&cross_out)?;

        // ---- GeGLU FFN with residual --------------------------------------
        let ff_in = cross_hidden.layer_norm_affine(std::sync::Arc::clone(&self.weights.ff_norm_gain), std::sync::Arc::clone(&self.weights.ff_norm_bias), cfg.layer_norm_eps)?;
        let ff_hidden = bcfg.hidden_size * cfg.ff_mult;
        let ff_up = self.weights.ff_proj.apply_linear(&ff_in, bcfg.hidden_size, 2 * ff_hidden);
        let ff_value = ff_up.slice(2_usize, 0, ff_hidden)?;
        let ff_gate = ff_up.slice(2_usize, ff_hidden, ff_hidden)?;
        let ff_inner = ff_value.mul(&ff_gate.gelu_erf())?;
        let ff_out = self.weights.ff_down.apply_linear(&ff_inner, ff_hidden, bcfg.hidden_size);
        let pooled_input = cross_hidden.add(&ff_out)?;

        // ---- Mask-weighted mean pool --------------------------------------
        let mask_f32: Vec<f32> = attention_mask.iter().map(|&m| m as f32).collect();
        let sum_mask: f32 = mask_f32.iter().sum();
        assert!(sum_mask > 0.0, "attention_mask sum must be > 0");
        let mask_t = embeds
            .const_f32_like(Arc::<[f32]>::from(mask_f32), Shape::from_dims(&[seq]))
            .reshape(Shape::from_dims(&[1, seq, 1]))?;
        let masked = pooled_input.broadcast_mul(&mask_t)?;
        let summed = masked.sum_dim(1_usize)?;
        let pooled = summed.mul_scalar(1.0_f64 / sum_mask as f64);

        // ---- L2-normalize -------------------------------------------------
        l2_normalize(&pooled)
    }

    /// Build the bidirectional pad-mask matching eager's
    /// `prepare_4d_attention_mask`: broadcast the 1-D pad mask
    /// over the target dim so only the SOURCE (j) is masked.
    /// `mask[b, 0, i, j] = (1 - mask[j]) * f32::MIN`.
    /// This keeps position i's row valid even if `mask[i] = 0`
    /// (the pooling step is what drops i's contribution at
    /// the end). Without this, masking position i would
    /// produce a row of `-inf`s and the softmax would NaN.
    fn build_bidirectional_pad_mask(
        &self,
        anchor: &LazyTensor,
        attention_mask: &[u32],
    ) -> LazyTensor {
        let seq = attention_mask.len();
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if attention_mask[j] == 0 {
                    mask_data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        anchor.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]))
    }
}

fn l2_normalize(x: &LazyTensor) -> Result<LazyTensor> {
    x.l2_normalize(1_usize, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::LayerWeights;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }

    fn tiny_backbone_cfg() -> MistralConfig {
        MistralConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4,
            num_key_value_heads: 2, head_dim: 4,
            rms_norm_eps: 1e-6, rope_theta: 10_000.0,
            max_position_embeddings: 32, sliding_window: None,
        }
    }

    fn tiny_mistral_weights(cfg: &MistralConfig, nb: &mut dyn FnMut() -> f32) -> MistralWeights {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h, nb);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, nb)),
            attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv, nb)),
            attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv, nb)),
            attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(h * h, nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, nb)),
            ffn_up: WeightStorage::F32(vec_of(h * i, nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, nb)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
        }).collect();
        MistralWeights {
            token_embedding,
            layers,
            final_norm_gain: Arc::from(vec![1.0_f32; h]),
            output: WeightStorage::F32(vec_of(h * cfg.vocab_size, nb)),
        }
    }

    fn tiny_nvembed_model(seed: u32) -> NvEmbedV2Model {
        let mut nb = rng_seed(seed);
        let backbone_cfg = tiny_backbone_cfg();
        let backbone = tiny_mistral_weights(&backbone_cfg, &mut nb);
        let cfg = NvEmbedV2Config {
            backbone: backbone_cfg.clone(),
            num_latents: 8,
            latent_heads: 2,
            latent_head_dim: 16,
            ff_mult: 2,
            layer_norm_eps: 1e-6,
        };
        let inner = cfg.latent_heads * cfg.latent_head_dim;
        let h = backbone_cfg.hidden_size;
        let ff_hidden = h * cfg.ff_mult;
        let weights = NvEmbedV2Weights {
            backbone,
            latents: vec_of(cfg.num_latents * h, &mut nb),
            cross_attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            cross_attn_norm_bias: Arc::from(vec![0.0_f32; h]),
            cross_attn_context_norm_gain: Arc::from(vec![1.0_f32; h]),
            cross_attn_context_norm_bias: Arc::from(vec![0.0_f32; h]),
            to_q: WeightStorage::F32(vec_of(h * inner, &mut nb)),
            to_kv: WeightStorage::F32(vec_of(h * 2 * inner, &mut nb)),
            to_out: WeightStorage::F32(vec_of(inner * h, &mut nb)),
            ff_norm_gain: Arc::from(vec![1.0_f32; h]),
            ff_norm_bias: Arc::from(vec![0.0_f32; h]),
            ff_proj: WeightStorage::F32(vec_of(h * 2 * ff_hidden, &mut nb)),
            ff_down: WeightStorage::F32(vec_of(ff_hidden * h, &mut nb)),
        };
        NvEmbedV2Model { config: cfg, weights }
    }

    #[test]
    fn forward_shape_and_l2_norm() {
        let model = tiny_nvembed_model(11);
        let tokens = [1_u32, 2, 3, 4, 5];
        let mask = [1_u32, 1, 1, 1, 1];
        let emb = model.forward(&tokens, &mask).unwrap();
        let h = model.config.backbone.hidden_size;
        assert_eq!(emb.shape().dims(), &[1, h]);
        let realized = emb.realize_f32();
        let norm_sq: f32 = realized.iter().map(|v| v * v).sum();
        assert!((norm_sq - 1.0).abs() < 1e-5,
            "L2 norm² expected ~1.0, got {norm_sq}");
    }

    /// Bidirectional attention: changing the last token must
    /// affect position 0's hidden state, which then propagates
    /// through the pooled mean.
    #[test]
    fn bidirectional_affects_pooling() {
        let model = tiny_nvembed_model(22);
        let toks_a = [1_u32, 2, 3, 4, 5];
        let toks_b = [1_u32, 2, 3, 4, 15];
        let mask = [1_u32; 5];
        let a = model.forward(&toks_a, &mask).unwrap().realize_f32();
        let b = model.forward(&toks_b, &mask).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "last-token change must affect bidirectional pooled embedding, max_diff = {max_diff}");
    }

    /// Masking out the last token alters the pooled embedding
    /// AND, because the bidirectional mask also drops that
    /// token from every other position's attention, the change
    /// is larger than just "average over fewer tokens".
    #[test]
    fn mask_zero_changes_embedding() {
        let model = tiny_nvembed_model(33);
        let tokens = [1_u32, 2, 3, 4, 5];
        let mask_all = [1_u32; 5];
        let mask_last = [1_u32, 1, 1, 1, 0];
        let a = model.forward(&tokens, &mask_all).unwrap().realize_f32();
        let b = model.forward(&tokens, &mask_last).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "padding the last token must change the embedding, max_diff = {max_diff}");
    }

    /// Latent KV bank is wired: zeroing the latents must alter
    /// the embedding (the latent attention output residual no
    /// longer contributes meaningfully).
    #[test]
    fn latents_are_wired() {
        let mut model = tiny_nvembed_model(44);
        let tokens = [1_u32, 2, 3, 4, 5];
        let mask = [1_u32; 5];
        let a = model.forward(&tokens, &mask).unwrap().realize_f32();
        // Replace latents with zeros.
        let h = model.config.backbone.hidden_size;
        model.weights.latents = Arc::from(vec![0.0_f32; model.config.num_latents * h]);
        let b = model.forward(&tokens, &mask).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "zeroing latents must alter embedding, max_diff = {max_diff}");
    }
}

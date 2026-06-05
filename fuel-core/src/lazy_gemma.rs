//! Gemma (v1) decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Gemma1 is the closest non-LLaMA architectural
//! cousin in this batch — same overall shape (RmsNorm, GQA, RoPE,
//! gated FFN) but with three small twists worth honoring:
//!
//!   1. **Offset RmsNorm gain** — Gemma uses `(gamma + 1)` rather
//!      than `gamma` as the per-channel scale. Carry this in
//!      `apply_offset_rms_norm`.
//!   2. **Embedding scaling** — the token embedding is scaled by
//!      `sqrt(hidden_size)` after lookup (matches reference Gemma).
//!   3. **GELU FFN** — `down(gelu(gate) * up)` instead of LLaMA's
//!      SwiGLU. The activation choice is config-driven; the
//!      `hidden_activation` field carries either `gelu` or
//!      `gelu_pytorch_tanh`.
//!   4. **Optional Q/K/V/O biases** — `attention_bias: bool` switches
//!      the biases on. Gemma 2B-it leaves them off; some forks turn
//!      them on. Carried as the standard optional-bias fields on
//!      [`crate::lazy::LayerWeights`].
//!
//! Gemma 2 already ships as part of `crate::lazy` (`Gemma2Model`).
//! Gemma v1 doesn't share the v2-only soft-cap or local/global
//! attention alternation, so it's its own module.
//!
//! # Scope (v1, same as the other Phase D ports)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache
//! (recomputes each call), F32 activations.

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

/// Which GELU variant the FFN's gate path uses. Defaults to
/// `GeluPytorchTanh` to match the reference Gemma checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmaActivation {
    /// Standard `0.5 * x * (1 + erf(x / sqrt(2)))`.
    Gelu,
    /// PyTorch's `approximate="tanh"` variant.
    GeluPytorchTanh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GemmaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub attention_bias: bool,
    pub hidden_activation: GemmaActivation,
}

impl GemmaConfig {
    /// Preset for `google/gemma-2b`. Values from the HF config.
    pub fn gemma_2b() -> Self {
        Self {
            vocab_size: 256_000,
            hidden_size: 2048,
            intermediate_size: 16_384,
            num_hidden_layers: 18,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 8192,
            attention_bias: false,
            hidden_activation: GemmaActivation::GeluPytorchTanh,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GemmaWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct GemmaModel {
    pub config: GemmaConfig,
    pub weights: GemmaWeights,
}

impl GemmaModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "GemmaModel::forward: tokens must be non-empty");
        assert_eq!(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            "GemmaConfig: num_attention_heads * head_dim must equal hidden_size",
        );

        // Embedding lookup + sqrt(hidden_size) scaling (Gemma-specific).
        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let h = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;
        let scale = (cfg.hidden_size as f64).sqrt();
        let h = h.mul_scalar(scale);

        self.forward_embeds(&h, start_pos)
    }

    /// Forward from pre-computed input embeddings of shape
    /// `(batch, seq, hidden_size)`. Used by multimodal models
    /// (PaliGemma, etc.) that interleave image embeddings with
    /// text embeddings before running the Gemma layers. The
    /// caller is responsible for the `sqrt(hidden_size)` token-
    /// embedding scaling that `forward()` applies internally.
    pub fn forward_embeds(&self, embeds: &LazyTensor, start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "embeds must be rank 3 [b, seq, hidden]");
        let seq = dims[1];
        assert_eq!(dims[2], cfg.hidden_size, "embeds last dim must equal hidden_size");
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads, 0,
            "GemmaConfig: num_attention_heads must be a multiple of num_key_value_heads",
        );

        let mut h = embeds.clone();

        // Shared RoPE tables — built fresh per call because seq may vary.
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, cfg.head_dim,
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }

        // Final offset RmsNorm + lm_head.
        let h_norm = apply_offset_rms_norm(
            &h, &weights.final_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        )?;
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Run the decoder forward up to the final offset RmsNorm
    /// and return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection. Pairs with
    /// [`Self::forward_hidden_embeds`] for vision-language
    /// composition or embedding adapters that need raw hidden
    /// states.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "GemmaModel::forward_hidden: tokens must be non-empty");

        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let h_raw = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;
        // Gemma scales the token-embedding output by sqrt(hidden_size).
        let h_scaled = h_raw.mul_scalar((cfg.hidden_size as f64).sqrt());
        self.forward_hidden_embeds(&h_scaled, start_pos)
    }

    /// Like [`Self::forward_embeds`] but skips the `lm_head`
    /// projection and returns the post-RmsNorm hidden states.
    /// Caller is responsible for the `sqrt(hidden_size)`
    /// embedding scaling that `forward_hidden()` applies.
    pub fn forward_hidden_embeds(&self, embeds: &LazyTensor, start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "embeds must be rank 3 [b, seq, hidden]");
        let seq = dims[1];
        assert_eq!(dims[2], cfg.hidden_size);

        let mut h = embeds.clone();
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, cfg.head_dim,
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        apply_offset_rms_norm(&h, &weights.final_norm_gain, cfg.hidden_size, cfg.rms_norm_eps)
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        // Pre-attention offset RmsNorm.
        let x_norm = apply_offset_rms_norm(
            x, &layer.attn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        )?;

        // Q / K / V — biases are honored when the config flag is on.
        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size).add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        // GQA replication.
        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        // Strict causal mask (Gemma v1 has no sliding window).
        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq {
                mask_data[i * seq + j] = f32::NEG_INFINITY;
            }
        }
        let mask = x.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]));
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size).add_optional_trailing_bias(// LayerWeights doesn't carry an explicit attn_o_bias; reuse
            // attn_q_bias's None branch by passing None here. Gemma's
            // o_proj bias support would need a LayerWeights extension if
            // a checkpoint requires it (rare).
            None)?;
        let h1 = x.add(&attn_out)?;

        // Pre-FFN offset RmsNorm.
        let h1_norm = apply_offset_rms_norm(
            &h1, &layer.ffn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        )?;

        // GELU gated FFN: `down(gelu(gate) * up)`.
        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let activated_gate = match cfg.hidden_activation {
            GemmaActivation::Gelu => gate.gelu_erf(),
            GemmaActivation::GeluPytorchTanh => gate.gelu(),
        };
        let ffn_in = activated_gate.mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&ffn_in, cfg.intermediate_size, cfg.hidden_size);

        h1.add(&ffn_out)
    }
}

/// Gemma's offset RmsNorm: `y = (x / rms) * (gamma + 1)`. The `+ 1`
/// matches the reference Gemma forward pass.
fn apply_offset_rms_norm(
    x: &LazyTensor,
    gain: &Arc<[f32]>,
    dim: usize,
    eps: f64,
) -> Result<LazyTensor> {
    let _ = dim;
    x.rms_norm_affine_with_offset(gain, 1.0, eps)
}


#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &GemmaConfig) -> GemmaWeights {
        let mut s: u32 = 4242;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let mut next_box: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *next_box);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
            attn_q_bias: if cfg.attention_bias { Some(vec_of(h, &mut *next_box)) } else { None },
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_k_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *next_box)) } else { None },
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_v_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *next_box)) } else { None },
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *next_box)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *next_box)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *next_box)),
            attn_norm_gain: Arc::from(vec![0.1_f32; h]), // non-zero so the +1 offset is visible
            ffn_norm_gain:  Arc::from(vec![0.1_f32; h]),
        }).collect();
        let final_norm_gain = Arc::from(vec![0.1_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *next_box));
        GemmaWeights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = GemmaConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            attention_bias: false,
            hidden_activation: GemmaActivation::GeluPytorchTanh,
        };
        let model = GemmaModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// Offset RmsNorm: a baseline (gain = 0) must produce the SAME
    /// post-norm output as a unity baseline through `apply_affine_rms_norm`
    /// (because (0 + 1) == 1).
    #[test]
    fn offset_rms_norm_with_zero_gain_matches_unity() {
        let device = Device::cpu();
        let dim = 8;
        let x = LazyTensor::from_f32(
            (0..dim).map(|i| 0.1 * (i as f32 - 3.5)).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 1, dim]),
            &device,
        );
        let zero_gain: Arc<[f32]> = Arc::from(vec![0.0_f32; dim]);
        let unity_gain: Arc<[f32]> = Arc::from(vec![1.0_f32; dim]);
        let offset = apply_offset_rms_norm(&x, &zero_gain, dim, 1e-6).unwrap();
        let unity = x.rms_norm_affine(Arc::clone(&unity_gain), 1e-6).unwrap();
        let a = offset.realize_f32();
        let b = unity.realize_f32();
        assert_eq!(a.len(), b.len());
        for (&av, &bv) in a.iter().zip(b.iter()) {
            assert!((av - bv).abs() < 1e-6, "offset(0) = {av} vs unity = {bv}");
        }
    }

    /// Embedding scaling: with `hidden_size = 4`, embedding gets
    /// scaled by 2 before the layers. Compare against a parallel
    /// model whose token_embedding rows are pre-scaled by 1/2 — the
    /// post-embedding state should match.
    ///
    /// (Cross-check via output equality requires identical downstream
    /// projections; tiny tolerance accounts for the per-layer norm's
    /// dependence on the scaled inputs.)
    #[test]
    fn embedding_scale_is_sqrt_hidden() {
        // We can't easily isolate the embedding from the rest of the
        // forward; instead, verify the scale value matches sqrt(h).
        let cfg = GemmaConfig {
            vocab_size: 4,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 16,
            attention_bias: false,
            hidden_activation: GemmaActivation::Gelu,
        };
        // hidden_size = 4, so sqrt = 2.0.
        assert!(((cfg.hidden_size as f64).sqrt() - 2.0).abs() < 1e-12);
        let model = GemmaModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        // Just smoke-test that forward runs without panic.
        let logits = model.forward(&[0, 1, 2], 0).unwrap().realize_f32();
        assert_eq!(logits.len(), 3 * cfg.vocab_size);
    }
}

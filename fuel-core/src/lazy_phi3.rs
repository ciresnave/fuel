//! Phi-3 decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Phi-3 (Phi-3-mini-4k-instruct etc.) is a
//! standard GQA transformer with HuggingFace's "fused projection"
//! quirk: a single `qkv_proj` packs Q + K + V along the last dim,
//! and a single `gate_up_proj` packs gate + up. On disk the
//! safetensors stores them fused; in lazy we store them split
//! (matching [`crate::lazy::LayerWeights`]), with the safetensors
//! loader doing the narrow at load time.
//!
//! **Deferred to a follow-up** (don't block other ports on it):
//!   - LongRoPE long-context scaling (short_factor / long_factor /
//!     `original_max_position_embeddings`). Phi-3-mini-4k doesn't
//!     use it; Phi-3-mini-128k does.
//!   - `partial_rotary_factor` < 1.0 (apply RoPE to only a prefix of
//!     each head's dim). Default is 1.0 (full rotary) — that's what
//!     this port assumes.
//!
//! Both can be added by augmenting `Phi3Config` + the RoPE table
//! builder when a Phi-3-128k checkpoint needs to run.
//!
//! # Scope (v1, same as the other Phase D ports)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache, F32
//! activations. Strict lower-triangular causal mask.

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Phi3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
}

impl Phi3Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `microsoft/Phi-3-mini-4k-instruct`.
    pub fn phi3_mini_4k() -> Self {
        Self {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 4096,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Phi3Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Phi3Model {
    pub config: Phi3Config,
    pub weights: Phi3Weights,
}

impl Phi3Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection. Mirrors the
    /// `forward_hidden` pattern shipped across the LLM family.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Shared backbone: embed → RoPE → per-layer attn + MLP →
    /// final RmsNorm. Used by both `forward` (then matmuls
    /// with `lm_head`) and `forward_hidden`.
    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "Phi3Model: tokens must be non-empty");
        let head_dim = cfg.head_dim();
        assert_eq!(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            "Phi3Config: num_attention_heads * head_dim must equal hidden_size",
        );
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads, 0,
            "Phi3Config: num_attention_heads must be a multiple of num_key_value_heads",
        );

        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;

        let (cos_data, sin_data) =
            fuel_graph::build_rope_tables(cfg.rope_theta, start_pos, seq, head_dim);
        let rope_shape = Shape::from_dims(&[seq, head_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        Ok(crate::lazy::apply_affine_rms_norm_pub(
            &h, &weights.final_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        ))
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * head_dim;

        let x_norm = crate::lazy::apply_affine_rms_norm_pub(
            x, &layer.attn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );

        // Bias-free Q / K / V (Phi-3 uses linear_no_bias for all).
        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size);
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim);
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim);

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let (k_full, v_full) = if n_rep == 1 {
            (k_r, v)
        } else {
            let expand = |t: LazyTensor| -> Result<LazyTensor> {
                let s5 = t.reshape(Shape::from_dims(&[
                    batch,
                    cfg.num_key_value_heads,
                    1,
                    seq,
                    head_dim,
                ]))?;
                let bcast = s5.broadcast_to(Shape::from_dims(&[
                    batch,
                    cfg.num_key_value_heads,
                    n_rep,
                    seq,
                    head_dim,
                ]))?;
                bcast.reshape(Shape::from_dims(&[
                    batch,
                    cfg.num_attention_heads,
                    seq,
                    head_dim,
                ]))
            };
            (expand(k_r)?, expand(v)?)
        };

        // Strict causal mask.
        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
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
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size);

        let h1 = x.add(&attn_out)?;
        let h1_norm = crate::lazy::apply_affine_rms_norm_pub(
            &h1, &layer.ffn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );

        // SwiGLU FFN (Phi-3's MLP is SwiGLU even though it stores
        // gate+up fused on disk).
        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size);

        h1.add(&ffn_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &Phi3Config) -> Phi3Weights {
        let mut s: u32 = 8888;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim();
        let mut next_box: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *next_box);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
            attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *next_box)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *next_box)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *next_box)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *next_box));
        Phi3Weights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = Phi3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let model = Phi3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// `forward_hidden` returns post-RmsNorm hidden states
    /// `(1, seq, hidden_size)` without the lm_head matmul.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = Phi3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let model = Phi3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

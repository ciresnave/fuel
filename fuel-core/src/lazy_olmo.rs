//! OLMo decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. OLMo is "LLaMA with LayerNorm-no-bias":
//! - GQA + RoPE + SwiGLU FFN — same as LLaMA.
//! - **Pre-attention and pre-FFN LayerNorm** (subtract-mean + scale)
//!   with **no bias** — distinct from LLaMA's RmsNorm and from
//!   Falcon's full LayerNorm-with-bias.
//! - Optional Q/K/V/O biases via `cfg.attention_bias`.
//!
//! `apply_layer_norm_no_bias` is a local helper that mean-centres
//! then divides by stddev (using `layer_norm_last_dim` which builds
//! the mean+variance reduction) and scales by gain only.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct OlmoConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub layer_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub attention_bias: bool,
}

impl OlmoConfig {
    /// `allenai/OLMo-7B-hf` ballpark.
    pub fn olmo_7b() -> Self {
        Self {
            vocab_size: 50_304,
            hidden_size: 4096,
            intermediate_size: 11_008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            head_dim: 128,
            layer_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 2048,
            attention_bias: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OlmoLayerWeights {
    pub attn_norm_gain: Arc<[f32]>,
    pub ffn_norm_gain: Arc<[f32]>,
    pub attn_q: WeightStorage,
    pub attn_q_bias: Option<Arc<[f32]>>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Option<Arc<[f32]>>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Option<Arc<[f32]>>,
    pub attn_o: WeightStorage,
    pub attn_o_bias: Option<Arc<[f32]>>,
    pub ffn_gate: WeightStorage,
    pub ffn_up: WeightStorage,
    pub ffn_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct OlmoWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<OlmoLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct OlmoModel {
    pub config: OlmoConfig,
    pub weights: OlmoWeights,
}

impl OlmoModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Run the decoder forward up to the final LayerNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection. OLMo uses LayerNorm
    /// without bias for the final norm.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert_eq!(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size);

        let mut h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;

        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, cfg.head_dim,
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        apply_layer_norm_no_bias(
            &h, &weights.final_norm_gain, cfg.hidden_size, cfg.layer_norm_eps,
        )
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &OlmoLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let x_norm = apply_layer_norm_no_bias(
            x, &layer.attn_norm_gain, cfg.hidden_size, cfg.layer_norm_eps,
        )?;
        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size).add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = LazyTensor::additive_causal_mask_like(x, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size).add_optional_trailing_bias(layer.attn_o_bias.as_ref())?;

        let h1 = x.add(&attn_out)?;
        let h1_norm = apply_layer_norm_no_bias(
            &h1, &layer.ffn_norm_gain, cfg.hidden_size, cfg.layer_norm_eps,
        )?;
        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size);
        h1.add(&ffn_out)
    }
}

/// `y = ((x - mean) / sqrt(var + eps)) * gain`. Same as
/// [`crate::lazy::apply_affine_layer_norm`] minus the additive bias.
fn apply_layer_norm_no_bias(
    x: &LazyTensor,
    gain: &Arc<[f32]>,
    dim: usize,
    eps: f64,
) -> Result<LazyTensor> {
    assert_eq!(gain.len(), dim);
    let normalized = x.layer_norm_last_dim(eps)?;
    let gain_t = x.const_f32_like(Arc::clone(gain), Shape::from_dims(&[dim]));
    normalized.broadcast_mul(&gain_t)
}


#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_weights(cfg: &OlmoConfig) -> OlmoWeights {
        let mut s: u32 = 11111;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size; let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<OlmoLayerWeights> = (0..cfg.num_hidden_layers).map(|_| OlmoLayerWeights {
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            attn_q_bias: if cfg.attention_bias { Some(vec_of(h, &mut *nb)) } else { None },
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_k_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_v_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            attn_o_bias: if cfg.attention_bias { Some(vec_of(h, &mut *nb)) } else { None },
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        OlmoWeights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = OlmoConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, layer_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 64, attention_bias: false,
        };
        let model = OlmoModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3, 4], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 4, cfg.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }

    /// `forward_hidden` returns post-LayerNorm-no-bias hidden
    /// states `(1, seq, hidden_size)` without the lm_head matmul.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = OlmoConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, layer_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 64, attention_bias: false,
        };
        let model = OlmoModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

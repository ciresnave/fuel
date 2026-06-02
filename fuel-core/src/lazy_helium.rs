//! Helium decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Helium (Kyutai / 2B) is a clean Qwen2-shape
//! transformer: GQA + RmsNorm + SwiGLU FFN + RoPE with optional
//! Q/K/V/O biases gated by `cfg.attention_bias`. No sliding window,
//! no MoE, no QK-norm.
//!
//! Reuses `crate::lazy::LayerWeights`. Forward is essentially
//! `Qwen2Model::forward` with sliding window forced off.

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct HeliumConfig {
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
    pub tie_word_embeddings: bool,
}

impl HeliumConfig {
    /// `kyutai/helium-1-preview-2b` ballpark.
    pub fn helium_2b() -> Self {
        Self {
            vocab_size: 32_000,
            hidden_size: 2560,
            intermediate_size: 7040,
            num_hidden_layers: 32,
            num_attention_heads: 20,
            num_key_value_heads: 20,
            head_dim: 128,
            rms_norm_eps: 1e-8,
            rope_theta: 100_000.0,
            max_position_embeddings: 4096,
            attention_bias: false,
            tie_word_embeddings: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HeliumWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct HeliumModel {
    pub config: HeliumConfig,
    pub weights: HeliumWeights,
}

impl HeliumModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert_eq!(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size);

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
            fuel_graph::build_rope_tables(cfg.rope_theta, start_pos, seq, cfg.head_dim);
        let rope_shape = Shape::from_dims(&[seq, cfg.head_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }

        let h_norm = crate::lazy::apply_affine_rms_norm_pub(
            &h, &weights.final_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
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

        let x_norm = crate::lazy::apply_affine_rms_norm_pub(
            x, &layer.attn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );

        let q = opt_bias(
            layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size),
            layer.attn_q_bias.as_ref(), cfg.hidden_size,
        )?;
        let k = opt_bias(
            layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim),
            layer.attn_k_bias.as_ref(), kv_dim,
        )?;
        let v = opt_bias(
            layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim),
            layer.attn_v_bias.as_ref(), kv_dim,
        )?;

        let q = q.reshape(Shape::from_dims(&[batch, seq, cfg.num_attention_heads, cfg.head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let k = k.reshape(Shape::from_dims(&[batch, seq, cfg.num_key_value_heads, cfg.head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let v = v.reshape(Shape::from_dims(&[batch, seq, cfg.num_key_value_heads, cfg.head_dim]))?
            .permute([0, 2, 1, 3_usize])?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let (k_full, v_full) = if n_rep == 1 { (k_r, v) } else {
            let expand = |t: LazyTensor| -> Result<LazyTensor> {
                let s5 = t.reshape(Shape::from_dims(&[
                    batch, cfg.num_key_value_heads, 1, seq, cfg.head_dim,
                ]))?;
                let bc = s5.broadcast_to(Shape::from_dims(&[
                    batch, cfg.num_key_value_heads, n_rep, seq, cfg.head_dim,
                ]))?;
                bc.reshape(Shape::from_dims(&[
                    batch, cfg.num_attention_heads, seq, cfg.head_dim,
                ]))
            };
            (expand(k_r)?, expand(v)?)
        };

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq { mask_data[i * seq + j] = f32::NEG_INFINITY; }
        }
        let mask = x.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]));
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.permute([0, 2, 1, 3_usize])?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size);

        let h1 = x.add(&attn_out)?;
        let h1_norm = crate::lazy::apply_affine_rms_norm_pub(
            &h1, &layer.ffn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size);
        h1.add(&ffn_out)
    }
}

fn opt_bias(x: LazyTensor, b: Option<&Arc<[f32]>>, n: usize) -> Result<LazyTensor> {
    match b {
        None => Ok(x),
        Some(bv) => {
            assert_eq!(bv.len(), n);
            let bt = x.const_f32_like(Arc::clone(bv), Shape::from_dims(&[n]));
            x.broadcast_add(&bt)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_weights(cfg: &HeliumConfig) -> HeliumWeights {
        let mut s: u32 = 33333;
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
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            attn_q_bias: if cfg.attention_bias { Some(vec_of(h, &mut *nb)) } else { None },
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_k_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_v_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        HeliumWeights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = HeliumConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, rms_norm_eps: 1e-8, rope_theta: 100_000.0,
            max_position_embeddings: 64, attention_bias: false,
            tie_word_embeddings: false,
        };
        let model = HeliumModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }
}

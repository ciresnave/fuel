//! SmolLM3 decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. SmolLM3 (HuggingFace small-model line) is a
//! Qwen2-shape transformer with two notable extras:
//! - **Per-layer RoPE gating** — `no_rope_layers[i] == 1` skips RoPE
//!   for layer `i`. Useful for hybrid attention patterns where some
//!   layers run position-agnostic.
//! - **Optional sliding window** (Mistral-style).
//!
//! Otherwise: GQA + RmsNorm + SwiGLU FFN + optional Q/K/V/O biases
//! via `cfg.attention_bias`. Reuses `crate::lazy::LayerWeights`.

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct SmolLm3Config {
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
    pub sliding_window: Option<usize>,
    /// One entry per layer: `1` = use RoPE on that layer, `0` = skip
    /// RoPE. `None` = use RoPE on every layer (Llama default).
    pub no_rope_layers: Option<Vec<usize>>,
}

impl SmolLm3Config {
    fn layer_uses_rope(&self, layer_idx: usize) -> bool {
        match &self.no_rope_layers {
            Some(v) => v.get(layer_idx).copied().unwrap_or(1) == 1,
            None => true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SmolLm3Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct SmolLm3Model {
    pub config: SmolLm3Config,
    pub weights: SmolLm3Weights,
}

impl SmolLm3Model {
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

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let uses_rope = cfg.layer_uses_rope(layer_idx);
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, uses_rope)?;
        }

        let h_norm = crate::lazy::apply_affine_rms_norm_pub(
            &h, &weights.final_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    fn build_mask(&self, anchor: &LazyTensor, seq: usize) -> LazyTensor {
        let cfg = &self.config;
        let window = cfg.sliding_window.unwrap_or(seq + 1);
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i || j + window <= i {
                    mask_data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        anchor.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]))
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        uses_rope: bool,
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

        // Conditional RoPE — skip for layers in `no_rope_layers`.
        let (q_r, k_r) = if uses_rope {
            (
                q.rope_with_tables(rope_cos, rope_sin)?,
                k.rope_with_tables(rope_cos, rope_sin)?,
            )
        } else {
            (q, k)
        };

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
        let mask = self.build_mask(x, seq);
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
    fn tiny_weights(cfg: &SmolLm3Config) -> SmolLm3Weights {
        let mut s: u32 = 55555;
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
        SmolLm3Weights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_and_finite_all_rope() {
        let cfg = SmolLm3Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, rms_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 64, attention_bias: false,
            sliding_window: None, no_rope_layers: None,
        };
        let model = SmolLm3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }

    /// `no_rope_layers = [0, 1]` skips RoPE on layer 0 only.
    /// Output must differ from the all-RoPE configuration.
    #[test]
    fn skipping_rope_on_one_layer_changes_output() {
        let mut cfg = SmolLm3Config {
            vocab_size: 16, hidden_size: 8, intermediate_size: 16,
            num_hidden_layers: 2, num_attention_heads: 2, num_key_value_heads: 2,
            head_dim: 4, rms_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 32, attention_bias: false,
            sliding_window: None, no_rope_layers: None,
        };
        let weights = tiny_weights(&cfg);
        let out_all = SmolLm3Model { config: cfg.clone(), weights: weights.clone() }
            .forward(&[1, 2, 3, 4], 0).unwrap().realize_f32();
        cfg.no_rope_layers = Some(vec![0, 1]); // skip RoPE on layer 0
        let out_partial = SmolLm3Model { config: cfg, weights }
            .forward(&[1, 2, 3, 4], 0).unwrap().realize_f32();
        let any_diff = out_all.iter().zip(out_partial.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7);
        assert!(any_diff, "skipping RoPE on layer 0 must change output");
    }
}

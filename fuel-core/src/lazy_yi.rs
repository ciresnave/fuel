//! Yi decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Yi (01-AI / 6B / 34B) is architecturally a
//! standard LLaMA: GQA + RmsNorm + SwiGLU FFN + RoPE with
//! `rope_theta = 5e6` (longer-context tuning). No special twists.
//!
//! Reuses [`crate::lazy::LayerWeights`] (LLaMA's per-layer struct).
//! Forward mirrors `crate::lazy_mistral::MistralModel` minus the
//! sliding-window mask. No KV cache (v1).

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct YiConfig {
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
}

impl YiConfig {
    /// `01-ai/Yi-6B`.
    pub fn yi_6b() -> Self {
        Self {
            vocab_size: 64_000,
            hidden_size: 4096,
            intermediate_size: 11_008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 5_000_000.0,
            max_position_embeddings: 4096,
        }
    }
}

#[derive(Debug, Clone)]
pub struct YiWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct YiModel {
    pub config: YiConfig,
    pub weights: YiWeights,
}

impl YiModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection.
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
        assert_eq!(cfg.num_attention_heads % cfg.num_key_value_heads, 0);

        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;

        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, cfg.head_dim,
        );

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
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let x_norm = crate::lazy::apply_affine_rms_norm_pub(
            x, &layer.attn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size);
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim);
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim);

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let (k_full, v_full) = if n_rep == 1 {
            (k_r, v)
        } else {
            let expand = |t: LazyTensor| -> Result<LazyTensor> {
                let s5 = t.reshape(Shape::from_dims(&[
                    batch, cfg.num_key_value_heads, 1, seq, cfg.head_dim,
                ]))?;
                let bcast = s5.broadcast_to(Shape::from_dims(&[
                    batch, cfg.num_key_value_heads, n_rep, seq, cfg.head_dim,
                ]))?;
                bcast.reshape(Shape::from_dims(&[
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

        let merged = attn_v.merge_heads()?;
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

#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_weights(cfg: &YiConfig) -> YiWeights {
        let mut s: u32 = 65537;
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
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)), attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        YiWeights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = YiConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2,
            head_dim: 4, rms_norm_eps: 1e-5, rope_theta: 5_000_000.0,
            max_position_embeddings: 64,
        };
        let model = YiModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for &v in &out { assert!(v.is_finite()); }
    }

    /// `forward_hidden` returns post-RmsNorm hidden states
    /// `(1, seq, hidden_size)` without the lm_head matmul.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = YiConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2,
            head_dim: 4, rms_norm_eps: 1e-5, rope_theta: 5_000_000.0,
            max_position_embeddings: 64,
        };
        let model = YiModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

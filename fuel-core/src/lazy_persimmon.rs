//! Persimmon decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Persimmon (Adept AI) combines several quirks:
//! - **LayerNorm with bias** (not RmsNorm) on pre-attention +
//!   pre-FFN paths.
//! - **QK-LayerNorm** — separate LayerNorm applied to projected Q
//!   and K BEFORE head reshape (`q_norm` shape `[hidden_size]`,
//!   `k_norm` shape `[num_kv_heads * head_dim]`).
//! - **Partial rotary** (factor 0.5 for the 8B base model). Reuses
//!   `crate::lazy_stablelm::apply_partial_rotary`.
//! - **ReLU MLP** — `down(relu(up(x)))`, no gate path.
//! - **Q/K/V/O biases** always present (no gating flag).
//!
//! Custom `PersimmonLayerWeights` because the LN+bias surface and
//! QK-LN structure don't fit `LayerWeights`.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_stablelm::apply_partial_rotary;
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct PersimmonConfig {
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
    pub partial_rotary_factor: f64,
    pub qk_layernorm: bool,
}

impl PersimmonConfig {
    pub fn rope_dim(&self) -> usize {
        let rd = (self.head_dim as f64 * self.partial_rotary_factor) as usize;
        (rd / 2) * 2
    }
}

#[derive(Debug, Clone)]
pub struct PersimmonLayerWeights {
    pub input_ln_gain: Arc<[f32]>,
    pub input_ln_bias: Arc<[f32]>,
    pub post_attn_ln_gain: Arc<[f32]>,
    pub post_attn_ln_bias: Arc<[f32]>,
    pub attn_q: WeightStorage,
    pub attn_q_bias: Arc<[f32]>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Arc<[f32]>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Arc<[f32]>,
    pub attn_o: WeightStorage,
    pub attn_o_bias: Arc<[f32]>,
    /// QK-LN gain + bias (present iff `cfg.qk_layernorm`).
    pub q_norm: Option<(Arc<[f32]>, Arc<[f32]>)>,
    pub k_norm: Option<(Arc<[f32]>, Arc<[f32]>)>,
    /// `[hidden_size, intermediate_size]`.
    pub mlp_up: WeightStorage,
    pub mlp_up_bias: Arc<[f32]>,
    /// `[intermediate_size, hidden_size]`.
    pub mlp_down: WeightStorage,
    pub mlp_down_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct PersimmonWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<PersimmonLayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct PersimmonModel {
    pub config: PersimmonConfig,
    pub weights: PersimmonWeights,
}

impl PersimmonModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Run the decoder forward up to the final LayerNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Persimmon uses partial RoPE + LayerNorm (gain+bias).
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

        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;

        let rope_dim = cfg.rope_dim();
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, rope_dim,
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        Ok(h.layer_norm_affine(std::sync::Arc::clone(&weights.final_ln_gain), std::sync::Arc::clone(&weights.final_ln_bias), cfg.layer_norm_eps)?)
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &PersimmonLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * head_dim;

        let x_norm = x.layer_norm_affine(std::sync::Arc::clone(&layer.input_ln_gain), std::sync::Arc::clone(&layer.input_ln_bias), cfg.layer_norm_eps)?;

        // Q/K/V projections — always have biases on Persimmon.
        let q = bias_add(
            layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size),
            &layer.attn_q_bias, cfg.hidden_size,
        )?;
        let k = bias_add(
            layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim),
            &layer.attn_k_bias, kv_dim,
        )?;
        let v = bias_add(
            layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim),
            &layer.attn_v_bias, kv_dim,
        )?;

        // QK-LayerNorm BEFORE head reshape.
        let (q, k) = match (&layer.q_norm, &layer.k_norm) {
            (Some((qg, qb)), Some((kg, kb))) => {
                let q = q.layer_norm_affine(std::sync::Arc::clone(&qg), std::sync::Arc::clone(&qb), cfg.layer_norm_eps)?;
                let k = k.layer_norm_affine(std::sync::Arc::clone(&kg), std::sync::Arc::clone(&kb), cfg.layer_norm_eps)?;
                (q, k)
            }
            _ => (q, k),
        };

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, head_dim)?;

        let q_r = apply_partial_rotary(&q, rope_cos, rope_sin, head_dim, cfg.rope_dim())?;
        let k_r = apply_partial_rotary(&k, rope_cos, rope_sin, head_dim, cfg.rope_dim())?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = LazyTensor::additive_causal_mask_like(x, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = bias_add(
            layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size),
            &layer.attn_o_bias, cfg.hidden_size,
        )?;

        let h1 = x.add(&attn_out)?;
        let h1_norm = h1.layer_norm_affine(std::sync::Arc::clone(&layer.post_attn_ln_gain), std::sync::Arc::clone(&layer.post_attn_ln_bias), cfg.layer_norm_eps)?;
        // MLP: simple `down(relu(up(x)))`.
        let up = bias_add(
            layer.mlp_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size),
            &layer.mlp_up_bias, cfg.intermediate_size,
        )?;
        let up_act = up.relu();
        let ffn_out = bias_add(
            layer.mlp_down.apply_linear(&up_act, cfg.intermediate_size, cfg.hidden_size),
            &layer.mlp_down_bias, cfg.hidden_size,
        )?;
        h1.add(&ffn_out)
    }
}

fn bias_add(x: LazyTensor, b: &Arc<[f32]>, n: usize) -> Result<LazyTensor> {
    let _ = n;
    x.add_trailing_bias(Arc::clone(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_weights(cfg: &PersimmonConfig) -> PersimmonWeights {
        let mut s: u32 = 77777;
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
        let layers: Vec<PersimmonLayerWeights> = (0..cfg.num_hidden_layers).map(|_| PersimmonLayerWeights {
            input_ln_gain:     Arc::from(vec![1.0_f32; h]),
            input_ln_bias:     Arc::from(vec![0.0_f32; h]),
            post_attn_ln_gain: Arc::from(vec![1.0_f32; h]),
            post_attn_ln_bias: Arc::from(vec![0.0_f32; h]),
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            attn_q_bias: vec_of(h, &mut *nb),
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_k_bias: vec_of(kv, &mut *nb),
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_v_bias: vec_of(kv, &mut *nb),
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            attn_o_bias: vec_of(h, &mut *nb),
            q_norm: if cfg.qk_layernorm {
                Some((Arc::from(vec![1.0_f32; h]), Arc::from(vec![0.0_f32; h])))
            } else { None },
            k_norm: if cfg.qk_layernorm {
                Some((Arc::from(vec![1.0_f32; kv]), Arc::from(vec![0.0_f32; kv])))
            } else { None },
            mlp_up: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            mlp_up_bias: vec_of(i, &mut *nb),
            mlp_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            mlp_down_bias: vec_of(h, &mut *nb),
        }).collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        PersimmonWeights { token_embedding, layers, final_ln_gain, final_ln_bias, output }
    }

    #[test]
    fn forward_with_qk_layernorm_and_partial_rotary() {
        let cfg = PersimmonConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, layer_norm_eps: 1e-5, rope_theta: 25_000.0,
            max_position_embeddings: 64, partial_rotary_factor: 1.0,
            qk_layernorm: true,
        };
        let model = PersimmonModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = PersimmonConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, layer_norm_eps: 1e-5, rope_theta: 25_000.0,
            max_position_embeddings: 64, partial_rotary_factor: 1.0,
            qk_layernorm: true,
        };
        let model = PersimmonModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

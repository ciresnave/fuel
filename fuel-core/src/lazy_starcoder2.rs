//! StarCoder2 decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. StarCoder2 is GQA + RoPE + LayerNorm + plain
//! `c_proj(gelu(c_fc(x)))` MLP. Closest cousin in this batch is
//! Falcon's serial-attention mode — same shared LN-with-bias
//! pattern — but StarCoder2 uses RoPE (not Falcon-style halfsplit
//! rotary on the heads-flattened view) and has standard
//! `[input_ln, attn, post_attn_ln, mlp]` sublayer ordering.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache, F32
//! activations. Sliding-window mask when `cfg.sliding_window` is
//! `Some(N)`; strict causal otherwise.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct StarCoder2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub norm_epsilon: f64,
    pub rope_theta: f64,
    pub use_bias: bool,
    pub sliding_window: Option<usize>,
}

impl StarCoder2Config {
    /// `bigcode/starcoder2-3b` ballpark.
    pub fn starcoder2_3b() -> Self {
        Self {
            vocab_size: 49_152,
            hidden_size: 3072,
            intermediate_size: 12_288,
            num_hidden_layers: 30,
            num_attention_heads: 24,
            num_key_value_heads: 2,
            head_dim: 128,
            max_position_embeddings: 16_384,
            norm_epsilon: 1e-5,
            rope_theta: 999_999.0,
            use_bias: true,
            sliding_window: Some(4096),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StarCoder2LayerWeights {
    pub input_ln_gain: Arc<[f32]>,
    pub input_ln_bias: Arc<[f32]>,
    pub post_attn_ln_gain: Arc<[f32]>,
    pub post_attn_ln_bias: Arc<[f32]>,
    pub attn_q: WeightStorage,
    pub attn_q_bias: Option<Arc<[f32]>>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Option<Arc<[f32]>>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Option<Arc<[f32]>>,
    pub attn_o: WeightStorage,
    pub attn_o_bias: Option<Arc<[f32]>>,
    pub mlp_fc: WeightStorage,
    pub mlp_fc_bias: Option<Arc<[f32]>>,
    pub mlp_proj: WeightStorage,
    pub mlp_proj_bias: Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct StarCoder2Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<StarCoder2LayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct StarCoder2Model {
    pub config: StarCoder2Config,
    pub weights: StarCoder2Weights,
}

impl StarCoder2Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Run the decoder forward up to the final LayerNorm and
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
        Ok(crate::lazy::apply_affine_layer_norm_pub(
            &h, &weights.final_ln_gain, &weights.final_ln_bias,
            cfg.hidden_size, cfg.norm_epsilon,
        ))
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &StarCoder2LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let x_norm = crate::lazy::apply_affine_layer_norm_pub(
            x, &layer.input_ln_gain, &layer.input_ln_bias,
            cfg.hidden_size, cfg.norm_epsilon,
        );

        let q = optional_bias(
            layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size),
            layer.attn_q_bias.as_ref(), cfg.hidden_size,
        )?;
        let k = optional_bias(
            layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim),
            layer.attn_k_bias.as_ref(), kv_dim,
        )?;
        let v = optional_bias(
            layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim),
            layer.attn_v_bias.as_ref(), kv_dim,
        )?;

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
        let mask = self.build_mask(x, seq);
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = optional_bias(
            layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size),
            layer.attn_o_bias.as_ref(), cfg.hidden_size,
        )?;

        let h1 = x.add(&attn_out)?;
        let h1_norm = crate::lazy::apply_affine_layer_norm_pub(
            &h1, &layer.post_attn_ln_gain, &layer.post_attn_ln_bias,
            cfg.hidden_size, cfg.norm_epsilon,
        );

        // MLP: c_proj(gelu(c_fc(x))). Standard GELU, not GeluPyTorchTanh.
        let mid = optional_bias(
            layer.mlp_fc.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size),
            layer.mlp_fc_bias.as_ref(), cfg.intermediate_size,
        )?;
        let mid_act = mid.gelu_erf();
        let ffn_out = optional_bias(
            layer.mlp_proj.apply_linear(&mid_act, cfg.intermediate_size, cfg.hidden_size),
            layer.mlp_proj_bias.as_ref(), cfg.hidden_size,
        )?;

        h1.add(&ffn_out)
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
}

fn optional_bias(x: LazyTensor, bias: Option<&Arc<[f32]>>, last_dim: usize) -> Result<LazyTensor> {
    match bias {
        None => Ok(x),
        Some(b) => {
            assert_eq!(b.len(), last_dim);
            let b_t = x.const_f32_like(Arc::clone(b), Shape::from_dims(&[last_dim]));
            x.broadcast_add(&b_t)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &StarCoder2Config) -> StarCoder2Weights {
        let mut s: u32 = 27182;
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
        let layers: Vec<StarCoder2LayerWeights> = (0..cfg.num_hidden_layers).map(|_| StarCoder2LayerWeights {
            input_ln_gain:     Arc::from(vec![1.0_f32; h]),
            input_ln_bias:     Arc::from(vec![0.0_f32; h]),
            post_attn_ln_gain: Arc::from(vec![1.0_f32; h]),
            post_attn_ln_bias: Arc::from(vec![0.0_f32; h]),
            attn_q:        WeightStorage::F32(vec_of(h * h, &mut *next_box)),
            attn_q_bias:   if cfg.use_bias { Some(vec_of(h, &mut *next_box)) } else { None },
            attn_k:        WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_k_bias:   if cfg.use_bias { Some(vec_of(kv, &mut *next_box)) } else { None },
            attn_v:        WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_v_bias:   if cfg.use_bias { Some(vec_of(kv, &mut *next_box)) } else { None },
            attn_o:        WeightStorage::F32(vec_of(h * h, &mut *next_box)),
            attn_o_bias:   if cfg.use_bias { Some(vec_of(h, &mut *next_box)) } else { None },
            mlp_fc:        WeightStorage::F32(vec_of(h * i, &mut *next_box)),
            mlp_fc_bias:   if cfg.use_bias { Some(vec_of(i, &mut *next_box)) } else { None },
            mlp_proj:      WeightStorage::F32(vec_of(i * h, &mut *next_box)),
            mlp_proj_bias: if cfg.use_bias { Some(vec_of(h, &mut *next_box)) } else { None },
        }).collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *next_box));
        StarCoder2Weights { token_embedding, layers, final_ln_gain, final_ln_bias, output }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = StarCoder2Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2,
            head_dim: 4, max_position_embeddings: 64, norm_epsilon: 1e-5,
            rope_theta: 10_000.0, use_bias: true, sliding_window: Some(4),
        };
        let model = StarCoder2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// `forward_hidden` returns post-LayerNorm hidden states.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = StarCoder2Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2,
            head_dim: 4, max_position_embeddings: 64, norm_epsilon: 1e-5,
            rope_theta: 10_000.0, use_bias: true, sliding_window: None,
        };
        let model = StarCoder2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

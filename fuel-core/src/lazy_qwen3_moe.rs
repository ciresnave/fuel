//! Qwen3-MoE decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Qwen3-MoE = Qwen3 attention (per-head QK-norm
//! + per-layer sliding-window gating + optional Q/K/V/O biases) +
//! per-layer FFN alternation between a dense SwiGLU MLP and a
//! Mixtral-style sparse MoE. `decoder_sparse_step` controls the
//! cadence: layer `i` uses MoE when `(i + 1) % decoder_sparse_step
//! == 0`; other layers run a single SwiGLU.
//!
//! v1 uses **dense routing** for the MoE layers (every expert
//! evaluated, weighted by full router softmax) — same trade-off
//! as Mixtral. No shared expert.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    /// Layer `i` uses MoE iff `(i + 1) % decoder_sparse_step == 0`.
    /// `1` → every layer is MoE; `2` → every other; etc.
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
}

impl Qwen3MoeConfig {
    pub fn layer_uses_moe(&self, layer_idx: usize) -> bool {
        self.num_experts > 0 && (layer_idx + 1) % self.decoder_sparse_step == 0
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3MoeExpertWeights {
    pub gate_w: WeightStorage,
    pub up_w: WeightStorage,
    pub down_w: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Qwen3MoeLayerWeights {
    pub attn_norm_gain: Arc<[f32]>,
    pub ffn_norm_gain: Arc<[f32]>,
    pub attn_q: WeightStorage,
    pub attn_q_bias: Option<Arc<[f32]>>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Option<Arc<[f32]>>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Option<Arc<[f32]>>,
    pub attn_o: WeightStorage,
    /// Per-head QK-norm gains (`[head_dim]` each).
    pub q_norm_gain: Arc<[f32]>,
    pub k_norm_gain: Arc<[f32]>,
    /// FFN variant. `Dense` → single SwiGLU; `Moe` → router + experts.
    pub ffn: Qwen3MoeFfn,
}

#[derive(Debug, Clone)]
pub enum Qwen3MoeFfn {
    Dense {
        gate_w: WeightStorage,
        up_w: WeightStorage,
        down_w: WeightStorage,
    },
    Moe {
        /// `[hidden_size, num_experts]` router.
        router_w: Arc<[f32]>,
        experts: Vec<Qwen3MoeExpertWeights>,
    },
}

#[derive(Debug, Clone)]
pub struct Qwen3MoeWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Qwen3MoeLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Qwen3MoeModel {
    pub config: Qwen3MoeConfig,
    pub weights: Qwen3MoeWeights,
}

impl Qwen3MoeModel {
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
            let uses_window = cfg.use_sliding_window && layer_idx < cfg.max_window_layers;
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, uses_window)?;
        }

        let h_norm = crate::lazy::apply_affine_rms_norm_pub(
            &h, &weights.final_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    fn build_layer_mask(&self, anchor: &LazyTensor, seq: usize, uses_window: bool) -> LazyTensor {
        let cfg = &self.config;
        let window = if uses_window { cfg.sliding_window.unwrap_or(seq + 1) } else { seq + 1 };
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i || j + window <= i { mask_data[i * seq + j] = f32::NEG_INFINITY; }
            }
        }
        anchor.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]))
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &Qwen3MoeLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        uses_window: bool,
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

        // Per-head QK-norm.
        let q = crate::lazy::apply_affine_rms_norm_pub(
            &q, &layer.q_norm_gain, cfg.head_dim, cfg.rms_norm_eps,
        );
        let k = crate::lazy::apply_affine_rms_norm_pub(
            &k, &layer.k_norm_gain, cfg.head_dim, cfg.rms_norm_eps,
        );

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
        let mask = self.build_layer_mask(x, seq, uses_window);
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

        let ffn_out = self.apply_ffn(&h1_norm, &layer.ffn, batch, seq)?;
        h1.add(&ffn_out)
    }

    fn apply_ffn(
        &self,
        x: &LazyTensor,
        ffn: &Qwen3MoeFfn,
        batch: usize,
        seq: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        match ffn {
            Qwen3MoeFfn::Dense { gate_w, up_w, down_w } => {
                let inter = cfg.intermediate_size;
                let gate = gate_w.apply_linear(x, h, inter);
                let up = up_w.apply_linear(x, h, inter);
                let swiglu = gate.silu().mul(&up)?;
                Ok(down_w.apply_linear(&swiglu, inter, h))
            }
            Qwen3MoeFfn::Moe { router_w, experts } => {
                let inter = cfg.moe_intermediate_size;
                let router_w_t = x.const_f32_like(
                    router_w.clone(),
                    Shape::from_dims(&[h, cfg.num_experts]),
                );
                let router_logits = x.matmul(&router_w_t)?;
                let router_weights = router_logits.softmax_last_dim()?;

                let mut routed_sum: Option<LazyTensor> = None;
                for (ei, ew) in experts.iter().enumerate() {
                    let gate = ew.gate_w.apply_linear(x, h, inter);
                    let up = ew.up_w.apply_linear(x, h, inter);
                    let swiglu = gate.silu().mul(&up)?;
                    let expert_out = ew.down_w.apply_linear(&swiglu, inter, h);

                    let w_col = router_weights.slice(2_usize, ei, 1)?;
                    let w_bc = w_col.broadcast_to(Shape::from_dims(&[batch, seq, h]))?;
                    let gated = expert_out.mul(&w_bc)?;
                    routed_sum = Some(match routed_sum {
                        Some(s) => s.add(&gated)?,
                        None => gated,
                    });
                }
                Ok(routed_sum.expect("Qwen3-MoE: must have at least one expert"))
            }
        }
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
    fn tiny_weights(cfg: &Qwen3MoeConfig) -> Qwen3MoeWeights {
        let mut s: u32 = 13579;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<Qwen3MoeLayerWeights> = (0..cfg.num_hidden_layers).map(|li| {
            let ffn = if cfg.layer_uses_moe(li) {
                let router_w = vec_of(h * cfg.num_experts, &mut *nb);
                let experts: Vec<Qwen3MoeExpertWeights> = (0..cfg.num_experts).map(|_| {
                    Qwen3MoeExpertWeights {
                        gate_w: WeightStorage::F32(vec_of(h * moe_inter, &mut *nb)),
                        up_w:   WeightStorage::F32(vec_of(h * moe_inter, &mut *nb)),
                        down_w: WeightStorage::F32(vec_of(moe_inter * h, &mut *nb)),
                    }
                }).collect();
                Qwen3MoeFfn::Moe { router_w, experts }
            } else {
                Qwen3MoeFfn::Dense {
                    gate_w: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                    up_w:   WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                    down_w: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
                }
            };
            Qwen3MoeLayerWeights {
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
                attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                attn_q_bias: if cfg.attention_bias { Some(vec_of(h, &mut *nb)) } else { None },
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_k_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_v_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
                attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                q_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
                k_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
                ffn,
            }
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        Qwen3MoeWeights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_with_alternating_dense_and_moe() {
        // decoder_sparse_step = 2 → layers 1 and 3 (0-indexed) use MoE,
        // layer 0 and 2 use dense.
        let cfg = Qwen3MoeConfig {
            vocab_size: 16, hidden_size: 8, intermediate_size: 16,
            num_hidden_layers: 4, num_attention_heads: 2, head_dim: 4,
            attention_bias: false, num_key_value_heads: 2,
            max_position_embeddings: 32,
            sliding_window: None, max_window_layers: 0, use_sliding_window: false,
            rope_theta: 10_000.0, rms_norm_eps: 1e-5,
            decoder_sparse_step: 2, moe_intermediate_size: 8,
            num_experts: 2, num_experts_per_tok: 1,
        };
        // Confirm the FFN-mode mapping is what we expect.
        assert_eq!(cfg.layer_uses_moe(0), false);
        assert_eq!(cfg.layer_uses_moe(1), true);
        assert_eq!(cfg.layer_uses_moe(2), false);
        assert_eq!(cfg.layer_uses_moe(3), true);
        let model = Qwen3MoeModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }
}

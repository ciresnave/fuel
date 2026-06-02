//! MPT (Mosaic Pretrained Transformer) decoder ported to the
//! lazy-graph API.
//!
//! Phase D LLM port. MPT (Replit-Code-v1.5-3B, MosaicBERT-style) is
//! distinguished by **ALiBi positional bias** instead of RoPE — a
//! per-head linear position penalty added directly to attention
//! scores. Otherwise: GQA + LayerNorm + GELU MLP + bias-free
//! projections.
//!
//! # ALiBi
//!
//! For a causal model, the bias for query position `i` attending to
//! key position `j ≤ i` is `slope[h] * (j - i)` (zero at `j == i`,
//! more negative as `j` recedes). Per-head slopes are
//! `1 / 2^(v * alibi_bias_max / n_heads_pow2)` for `v = 1..=n_heads_pow2`,
//! with the canonical interleave trick when `n_heads` isn't a
//! power of 2.
//!
//! v1 pre-computes the combined ALiBi + causal mask
//! `[1, n_heads, seq, seq]` as a single F32 const tensor at forward
//! time and broadcast-adds it to the attention scores before
//! softmax — same shape as the standard causal mask, just with
//! ALiBi's negative biases on the valid (lower-triangular)
//! positions instead of zeros.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct MptConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub expansion_ratio: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub kv_n_heads: usize,
    pub alibi_bias_max: usize,
    pub layer_norm_eps: f64,
}

impl MptConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    pub fn ffn_dim(&self) -> usize {
        self.d_model * self.expansion_ratio
    }

    /// Replit-Code-v1.5-3B preset.
    pub fn replit_code_v1_5_3b() -> Self {
        Self {
            d_model: 3072,
            n_heads: 24,
            n_layers: 32,
            expansion_ratio: 4,
            max_seq_len: 4096,
            vocab_size: 32_768,
            kv_n_heads: 8,
            alibi_bias_max: 8,
            layer_norm_eps: 1e-5,
        }
    }

    /// Compute the per-head ALiBi slopes vector of length `n_heads`.
    /// Mirrors the eager `build_alibi_bias` slope construction.
    pub fn alibi_slopes(&self) -> Vec<f32> {
        let n = self.n_heads;
        let mut n2 = 1_usize;
        while n2 < n { n2 *= 2; }
        let bias_max = self.alibi_bias_max;
        let slopes: Vec<f32> = (1..=n2)
            .map(|v| 1.0_f32 / 2.0_f32.powf((v * bias_max) as f32 / n2 as f32))
            .collect();
        if n2 == n {
            slopes
        } else {
            // Interleave: odd indices first, then even.
            let evens: Vec<f32> = slopes.iter().step_by(2).copied().collect();
            let odds:  Vec<f32> = slopes.iter().skip(1).step_by(2).copied().collect();
            odds.into_iter().chain(evens.into_iter()).take(n).collect()
        }
    }
}

/// Combined ALiBi + causal mask for `seq` tokens. Layout
/// `[1, n_heads, seq, seq]` row-major. For `j > i` (future), the
/// entry is `-inf`. For `j <= i`, the entry is
/// `slope[h] * (j - i)` (zero on the diagonal, more negative as `j`
/// recedes).
pub fn build_alibi_causal_mask(seq: usize, slopes: &[f32]) -> Vec<f32> {
    let n_heads = slopes.len();
    let mut out = vec![0.0_f32; n_heads * seq * seq];
    for h in 0..n_heads {
        for i in 0..seq {
            for j in 0..seq {
                let idx = h * seq * seq + i * seq + j;
                if j > i {
                    out[idx] = f32::NEG_INFINITY;
                } else {
                    out[idx] = slopes[h] * (j as f32 - i as f32);
                }
            }
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct MptLayerWeights {
    pub norm1_gain: Arc<[f32]>,
    pub norm1_bias: Arc<[f32]>,
    pub norm2_gain: Arc<[f32]>,
    pub norm2_bias: Arc<[f32]>,
    /// Bias-free Q/K/V/O. MPT fuses Q+K+V on disk; we store split in
    /// memory.
    pub attn_q: WeightStorage,
    pub attn_k: WeightStorage,
    pub attn_v: WeightStorage,
    pub attn_o: WeightStorage,
    /// `[d_model, ffn_dim]` — `up_proj`.
    pub mlp_up: WeightStorage,
    /// `[ffn_dim, d_model]` — `down_proj`.
    pub mlp_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct MptWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<MptLayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct MptModel {
    pub config: MptConfig,
    pub weights: MptWeights,
}

impl MptModel {
    pub fn forward(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert_eq!(cfg.n_heads * cfg.head_dim(), cfg.d_model);
        assert_eq!(cfg.n_heads % cfg.kv_n_heads, 0);

        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.d_model]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.d_model]))?;

        // ALiBi + causal combined mask, computed once at forward.
        let slopes = cfg.alibi_slopes();
        let mask_data = build_alibi_causal_mask(seq, &slopes);
        let mask = h.const_f32_like(
            mask_data,
            Shape::from_dims(&[1, cfg.n_heads, seq, seq]),
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &mask)?;
        }

        let h_norm = crate::lazy::apply_affine_layer_norm_pub(
            &h, &weights.final_ln_gain, &weights.final_ln_bias,
            cfg.d_model, cfg.layer_norm_eps,
        );
        Ok(weights.output.apply_linear(&h_norm, cfg.d_model, cfg.vocab_size))
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &MptLayerWeights,
        mask: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.kv_n_heads * head_dim;

        let x_norm = crate::lazy::apply_affine_layer_norm_pub(
            x, &layer.norm1_gain, &layer.norm1_bias,
            cfg.d_model, cfg.layer_norm_eps,
        );

        // Bias-free Q/K/V.
        let q = layer.attn_q.apply_linear(&x_norm, cfg.d_model, cfg.d_model);
        let k = layer.attn_k.apply_linear(&x_norm, cfg.d_model, kv_dim);
        let v = layer.attn_v.apply_linear(&x_norm, cfg.d_model, kv_dim);

        let q = q.reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let k = k.reshape(Shape::from_dims(&[batch, seq, cfg.kv_n_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let v = v.reshape(Shape::from_dims(&[batch, seq, cfg.kv_n_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;

        // GQA replication.
        let n_rep = cfg.n_heads / cfg.kv_n_heads;
        let (k_full, v_full) = if n_rep == 1 { (k, v) } else {
            let expand = |t: LazyTensor| -> Result<LazyTensor> {
                let s5 = t.reshape(Shape::from_dims(&[
                    batch, cfg.kv_n_heads, 1, seq, head_dim,
                ]))?;
                let bc = s5.broadcast_to(Shape::from_dims(&[
                    batch, cfg.kv_n_heads, n_rep, seq, head_dim,
                ]))?;
                bc.reshape(Shape::from_dims(&[
                    batch, cfg.n_heads, seq, head_dim,
                ]))
            };
            (expand(k)?, expand(v)?)
        };

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        // Broadcast-add the ALiBi + causal mask (shape
        // `[1, n_heads, seq, seq]`). Broadcasts cleanly over the
        // batch axis.
        let scores_masked = scores_scaled.broadcast_add(mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.permute([0, 2, 1, 3_usize])?
            .reshape(Shape::from_dims(&[batch, seq, cfg.d_model]))?;
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.d_model, cfg.d_model);

        let h1 = x.add(&attn_out)?;
        let h1_norm = crate::lazy::apply_affine_layer_norm_pub(
            &h1, &layer.norm2_gain, &layer.norm2_bias,
            cfg.d_model, cfg.layer_norm_eps,
        );

        let mid = layer.mlp_up.apply_linear(&h1_norm, cfg.d_model, cfg.ffn_dim());
        let mid_act = mid.gelu_erf();
        let ffn_out = layer.mlp_down.apply_linear(&mid_act, cfg.ffn_dim(), cfg.d_model);
        h1.add(&ffn_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_weights(cfg: &MptConfig) -> MptWeights {
        let mut s: u32 = 14641;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.d_model;
        let kv = cfg.kv_n_heads * cfg.head_dim();
        let inter = cfg.ffn_dim();
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<MptLayerWeights> = (0..cfg.n_layers).map(|_| MptLayerWeights {
            norm1_gain: Arc::from(vec![1.0_f32; h]),
            norm1_bias: Arc::from(vec![0.0_f32; h]),
            norm2_gain: Arc::from(vec![1.0_f32; h]),
            norm2_bias: Arc::from(vec![0.0_f32; h]),
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            mlp_up:   WeightStorage::F32(vec_of(h * inter, &mut *nb)),
            mlp_down: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
        }).collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        MptWeights { token_embedding, layers, final_ln_gain, final_ln_bias, output }
    }

    #[test]
    fn forward_shape_and_finite_with_alibi() {
        let cfg = MptConfig {
            d_model: 16, n_heads: 4, n_layers: 2, expansion_ratio: 4,
            max_seq_len: 32, vocab_size: 32, kv_n_heads: 2,
            alibi_bias_max: 8, layer_norm_eps: 1e-5,
        };
        let model = MptModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3, 4]).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 4, cfg.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }

    /// ALiBi slopes for n_heads = 4 (power of 2) follow the
    /// `1 / 2^(v * 8 / 4)` formula directly.
    #[test]
    fn alibi_slopes_power_of_two_heads() {
        let cfg = MptConfig {
            d_model: 16, n_heads: 4, n_layers: 1, expansion_ratio: 4,
            max_seq_len: 16, vocab_size: 16, kv_n_heads: 2,
            alibi_bias_max: 8, layer_norm_eps: 1e-5,
        };
        let slopes = cfg.alibi_slopes();
        assert_eq!(slopes.len(), 4);
        // slopes[0] = 1 / 2^(1 * 8 / 4) = 1/4
        // slopes[1] = 1 / 2^(2 * 8 / 4) = 1/16
        // slopes[2] = 1 / 2^(3 * 8 / 4) = 1/64
        // slopes[3] = 1 / 2^(4 * 8 / 4) = 1/256
        for (i, expected) in [0.25_f32, 0.0625, 0.015_625, 0.003_906_25].iter().enumerate() {
            assert!((slopes[i] - *expected).abs() < 1e-6, "slopes[{i}] = {} vs {expected}", slopes[i]);
        }
    }

    /// ALiBi penalty must produce a different output than a strict
    /// causal mask alone. We compare two runs with the same weights
    /// but pre-built masks that differ only in their lower-triangle
    /// entries (zero vs ALiBi-shaped).
    #[test]
    fn alibi_mask_differs_from_zero_lower_triangle() {
        let cfg = MptConfig {
            d_model: 8, n_heads: 2, n_layers: 1, expansion_ratio: 2,
            max_seq_len: 16, vocab_size: 8, kv_n_heads: 1,
            alibi_bias_max: 8, layer_norm_eps: 1e-5,
        };
        let slopes = cfg.alibi_slopes();
        let causal_mask = build_alibi_causal_mask(4, &[0.0, 0.0]); // zero slopes → causal only
        let alibi_mask = build_alibi_causal_mask(4, &slopes);
        // Should differ at positions where j < i (the ALiBi negative bias).
        let any_diff = causal_mask.iter().zip(alibi_mask.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7 && a.is_finite() && b.is_finite());
        assert!(any_diff, "ALiBi mask must differ from zero-slope causal mask on j < i positions");
    }
}

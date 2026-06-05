//! Falcon (7B and similar) decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Falcon is architecturally distinct from
//! LLaMA-cousins:
//!   1. **Parallel attention + FFN** — `out = attn(ln(x)) + mlp(ln(x))
//!      + x` with a shared LayerNorm input. Two branches sum into one
//!      residual instead of LLaMA's serial two-residual flow.
//!   2. **LayerNorm** (with bias) — not RmsNorm. Both `gamma` and
//!      `beta` live per layer.
//!   3. **Multi-query attention** (n_head_kv == 1) by default — one
//!      shared K and one shared V for all attention heads. Implemented
//!      via the existing GQA replication code with `num_kv_heads = 1`.
//!   4. **Standard GELU MLP** — `down(gelu(up(x)))`, no gate path
//!      (h → 4h → h, two projections).
//!   5. **No final LayerNorm** post-decoder per the eager
//!      reference — wait, yes there is: `ln_f` after all decoder
//!      blocks. So: input embed → N × decoder block → final LN →
//!      lm_head.
//!   6. **Optional projection biases** — `cfg.bias` flag enables
//!      biases on Q/K/V/O/MLP linears.
//!
//! Custom [`FalconLayerWeights`] because LayerNorm has both gain and
//! bias (vs LLaMA RmsNorm's gain-only), and the MLP shape differs
//! (no gate path).
//!
//! # Scope (v1)
//!
//! - Forward-only, single sequence (`batch == 1`), no KV cache.
//! - Multi-query attention with `n_head_kv = 1` (the Falcon 7B
//!   default). The struct carries `n_head_kv` so non-MQA variants
//!   can be plugged in.
//! - No ALiBi, no `new_decoder_architecture` (Falcon-180B uses it;
//!   add when needed).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct FalconConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Per-layer K/V head count. `1` for default MQA Falcon-7B.
    pub n_head_kv: usize,
    pub layer_norm_epsilon: f64,
    pub max_position_embeddings: usize,
    /// True for Falcon-7B/40B/180B — parallel attention + FFN with a
    /// shared input LayerNorm.
    pub parallel_attn: bool,
    /// True when projection weights have additive biases.
    pub bias: bool,
    pub rope_theta: f64,
}

impl FalconConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `tiiuae/falcon-7b` defaults.
    pub fn falcon_7b() -> Self {
        Self {
            vocab_size: 65_024,
            hidden_size: 4544,
            num_hidden_layers: 32,
            num_attention_heads: 71,
            n_head_kv: 1,
            layer_norm_epsilon: 1e-5,
            max_position_embeddings: 2048,
            parallel_attn: true,
            bias: false,
            rope_theta: 10_000.0,
        }
    }
}

/// Falcon per-layer weights. Distinct from `crate::lazy::LayerWeights`
/// because:
///   - Input LN has both `gain` and `bias` (and an optional
///     post-attention LN when `parallel_attn == false`).
///   - MLP is just `up + down` (no gate).
#[derive(Debug, Clone)]
pub struct FalconLayerWeights {
    pub input_ln_gain: Arc<[f32]>,
    pub input_ln_bias: Arc<[f32]>,
    /// Present only when `parallel_attn == false` (Falcon-7B leaves
    /// this `None`).
    pub post_attn_ln: Option<(Arc<[f32]>, Arc<[f32]>)>,
    pub attn_q: WeightStorage,
    pub attn_q_bias: Option<Arc<[f32]>>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Option<Arc<[f32]>>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Option<Arc<[f32]>>,
    pub attn_dense: WeightStorage,
    pub attn_dense_bias: Option<Arc<[f32]>>,
    /// `[hidden_size, 4 * hidden_size]`.
    pub mlp_up: WeightStorage,
    pub mlp_up_bias: Option<Arc<[f32]>>,
    /// `[4 * hidden_size, hidden_size]`.
    pub mlp_down: WeightStorage,
    pub mlp_down_bias: Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct FalconWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<FalconLayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct FalconModel {
    pub config: FalconConfig,
    pub weights: FalconWeights,
}

impl FalconModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Run the decoder forward up to the final LayerNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection. Mirrors the
    /// `forward_hidden` pattern across the LLM family —
    /// Falcon-specific bit is the final-LN uses gain+bias
    /// affine (LayerNorm, not RmsNorm).
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Shared backbone: embed → RoPE → per-layer parallel
    /// attn + MLP (Falcon's parallel structure) → final
    /// LayerNorm.
    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "FalconModel: tokens must be non-empty");
        let head_dim = cfg.head_dim();
        assert_eq!(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            "FalconConfig: num_attention_heads * head_dim must equal hidden_size",
        );
        assert!(
            cfg.n_head_kv >= 1 && cfg.num_attention_heads % cfg.n_head_kv == 0,
            "FalconConfig: num_attention_heads must be a positive multiple of n_head_kv",
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
        Ok(crate::lazy::apply_affine_layer_norm_pub(
            &h, &weights.final_ln_gain, &weights.final_ln_bias,
            cfg.hidden_size, cfg.layer_norm_epsilon,
        ))
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &FalconLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_head_kv * head_dim;

        // Shared input LayerNorm for attention (and FFN in parallel mode).
        let x_ln = crate::lazy::apply_affine_layer_norm_pub(
            x, &layer.input_ln_gain, &layer.input_ln_bias,
            cfg.hidden_size, cfg.layer_norm_epsilon,
        );

        let attn_output = self.attention(&x_ln, layer, rope_cos, rope_sin, batch, seq, head_dim, kv_dim)?;

        if cfg.parallel_attn {
            // `out = attn(ln(x)) + mlp(ln(x)) + x` — both branches use
            // the SAME ln(x) input, and a single residual sums them.
            let mlp_output = self.mlp(&x_ln, layer, batch, seq)?;
            let summed = attn_output.add(&mlp_output)?;
            x.add(&summed)
        } else {
            // Serial: `h1 = attn(ln(x)) + x; out = mlp(ln'(h1)) + h1`.
            let h1 = x.add(&attn_output)?;
            let h1_ln = match &layer.post_attn_ln {
                Some((g, b)) => crate::lazy::apply_affine_layer_norm_pub(
                    &h1, g, b, cfg.hidden_size, cfg.layer_norm_epsilon,
                ),
                None => h1.clone(),
            };
            let mlp_output = self.mlp(&h1_ln, layer, batch, seq)?;
            h1.add(&mlp_output)
        }
    }

    fn attention(
        &self,
        x_ln: &LazyTensor,
        layer: &FalconLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        batch: usize,
        seq: usize,
        head_dim: usize,
        kv_dim: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let q = optional_bias(
            layer.attn_q.apply_linear(x_ln, cfg.hidden_size, cfg.hidden_size),
            layer.attn_q_bias.as_ref(), cfg.hidden_size,
        )?;
        let k = optional_bias(
            layer.attn_k.apply_linear(x_ln, cfg.hidden_size, kv_dim),
            layer.attn_k_bias.as_ref(), kv_dim,
        )?;
        let v = optional_bias(
            layer.attn_v.apply_linear(x_ln, cfg.hidden_size, kv_dim),
            layer.attn_v_bias.as_ref(), kv_dim,
        )?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.n_head_kv, head_dim)?;
        let v = v.split_heads(cfg.n_head_kv, head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        // Multi-query attention: broadcast K/V from n_head_kv → num_heads.
        let n_rep = cfg.num_attention_heads / cfg.n_head_kv;
        let (k_full, v_full) = if n_rep == 1 {
            (k_r, v)
        } else {
            let expand = |t: LazyTensor| -> Result<LazyTensor> {
                let s5 = t.reshape(Shape::from_dims(&[
                    batch, cfg.n_head_kv, 1, seq, head_dim,
                ]))?;
                let bcast = s5.broadcast_to(Shape::from_dims(&[
                    batch, cfg.n_head_kv, n_rep, seq, head_dim,
                ]))?;
                bcast.reshape(Shape::from_dims(&[
                    batch, cfg.num_attention_heads, seq, head_dim,
                ]))
            };
            (expand(k_r)?, expand(v)?)
        };

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = LazyTensor::additive_causal_mask_like(&x_ln, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        optional_bias(
            layer.attn_dense.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size),
            layer.attn_dense_bias.as_ref(), cfg.hidden_size,
        )
    }

    fn mlp(
        &self,
        x_ln: &LazyTensor,
        layer: &FalconLayerWeights,
        _batch: usize,
        _seq: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let inter = 4 * cfg.hidden_size;
        let up = optional_bias(
            layer.mlp_up.apply_linear(x_ln, cfg.hidden_size, inter),
            layer.mlp_up_bias.as_ref(), inter,
        )?;
        let up_act = up.gelu();
        optional_bias(
            layer.mlp_down.apply_linear(&up_act, inter, cfg.hidden_size),
            layer.mlp_down_bias.as_ref(), cfg.hidden_size,
        )
    }
}

fn optional_bias(x: LazyTensor, bias: Option<&Arc<[f32]>>, last_dim: usize) -> Result<LazyTensor> {
    let _ = last_dim;
    x.add_optional_trailing_bias(bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &FalconConfig) -> FalconWeights {
        let mut s: u32 = 5555;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let kv = cfg.n_head_kv * cfg.head_dim();
        let inter = 4 * h;
        let mut next_box: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *next_box);
        let layers: Vec<FalconLayerWeights> = (0..cfg.num_hidden_layers).map(|_| FalconLayerWeights {
            input_ln_gain: Arc::from(vec![1.0_f32; h]),
            input_ln_bias: Arc::from(vec![0.0_f32; h]),
            post_attn_ln: if cfg.parallel_attn {
                None
            } else {
                Some((Arc::from(vec![1.0_f32; h]), Arc::from(vec![0.0_f32; h])))
            },
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
            attn_q_bias: if cfg.bias { Some(vec_of(h, &mut *next_box)) } else { None },
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_k_bias: if cfg.bias { Some(vec_of(kv, &mut *next_box)) } else { None },
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_v_bias: if cfg.bias { Some(vec_of(kv, &mut *next_box)) } else { None },
            attn_dense: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
            attn_dense_bias: if cfg.bias { Some(vec_of(h, &mut *next_box)) } else { None },
            mlp_up: WeightStorage::F32(vec_of(h * inter, &mut *next_box)),
            mlp_up_bias: if cfg.bias { Some(vec_of(inter, &mut *next_box)) } else { None },
            mlp_down: WeightStorage::F32(vec_of(inter * h, &mut *next_box)),
            mlp_down_bias: if cfg.bias { Some(vec_of(h, &mut *next_box)) } else { None },
        }).collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *next_box));
        FalconWeights { token_embedding, layers, final_ln_gain, final_ln_bias, output }
    }

    #[test]
    fn forward_shape_and_finite_parallel_attn() {
        let cfg = FalconConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            n_head_kv: 1,
            layer_norm_epsilon: 1e-5,
            max_position_embeddings: 64,
            parallel_attn: true,
            bias: false,
            rope_theta: 10_000.0,
        };
        let model = FalconModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// Serial mode (parallel_attn = false): exercises the
    /// post-attention LayerNorm path.
    #[test]
    fn forward_shape_and_finite_serial_attn() {
        let cfg = FalconConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            n_head_kv: 2,
            layer_norm_epsilon: 1e-5,
            max_position_embeddings: 32,
            parallel_attn: false,
            bias: true,
            rope_theta: 10_000.0,
        };
        let model = FalconModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![3, 1, 4, 1, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for &v in out.iter() {
            assert!(v.is_finite());
        }
    }

    /// Parallel-attn output must differ from serial-attn output for
    /// the same weights — they're different computations.
    #[test]
    fn parallel_and_serial_attn_diverge() {
        let cfg_p = FalconConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            n_head_kv: 2,
            layer_norm_epsilon: 1e-5,
            max_position_embeddings: 16,
            parallel_attn: true,
            bias: false,
            rope_theta: 10_000.0,
        };
        let weights = tiny_weights(&cfg_p);
        let mut cfg_s = cfg_p.clone();
        cfg_s.parallel_attn = false;
        // For serial mode the tiny_weights doesn't add post_attn_ln
        // (it checks `parallel_attn` at the moment of construction);
        // build a serial-shaped weight set instead.
        let weights_s = {
            let mut w = weights.clone();
            for l in &mut w.layers {
                l.post_attn_ln = Some((
                    Arc::from(vec![1.0_f32; cfg_p.hidden_size]),
                    Arc::from(vec![0.0_f32; cfg_p.hidden_size]),
                ));
            }
            w
        };
        let out_p = FalconModel { config: cfg_p, weights }
            .forward(&[1, 2, 3, 4], 0).unwrap().realize_f32();
        let out_s = FalconModel { config: cfg_s, weights: weights_s }
            .forward(&[1, 2, 3, 4], 0).unwrap().realize_f32();
        let any_diff = out_p.iter().zip(out_s.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(any_diff, "parallel vs serial attention must diverge");
    }

    /// `forward_hidden` returns post-final-LN hidden states
    /// `(1, seq, hidden_size)` without the lm_head matmul.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = FalconConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            n_head_kv: 1,
            layer_norm_epsilon: 1e-5,
            max_position_embeddings: 64,
            parallel_attn: true,
            bias: false,
            rope_theta: 10_000.0,
        };
        let model = FalconModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

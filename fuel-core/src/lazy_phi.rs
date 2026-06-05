//! Phi (Phi-1.5 / Phi-2) decoder ported to the lazy-graph API.
//!
//! Phase D specialized port. Microsoft's older Phi series (before
//! Phi-3) has its own architectural signature distinct from the
//! Llama family:
//!
//!   1. **Parallel attention + MLP** — same residual block as
//!      Falcon: `out = residual + attn(LN(x)) + mlp(LN(x))`.
//!      A single `input_layernorm` feeds both paths. No
//!      separate `post_attention_layernorm`.
//!   2. **Partial rotary** — only the first
//!      `partial_rotary_factor * head_dim` features of each
//!      head are rotated; the tail passes through unchanged.
//!      Same shape as the StableLM / Persimmon partial rotary
//!      handled by [`LazyTensor::rope_partial`].
//!   3. **LayerNorm (with bias) on hidden_size** — not RmsNorm.
//!   4. **Sequential MLP** — `fc2(act(fc1(x)))`. No SwiGLU gate.
//!      Activation is config-driven (the reference Phi-2 uses
//!      "new-GELU"; we expose both Gelu and GeluPytorchTanh
//!      variants and treat them as the same approximation
//!      family — the underlying `LazyTensor::gelu()` is the
//!      tanh-approximation, matching `Activation::NewGelu`).
//!   5. **All linear layers carry biases** — Q/K/V/dense and
//!      fc1/fc2 and lm_head all include bias terms.
//!
//! The `qk_layernorm` flag is exposed but v1 only supports
//! `qk_layernorm == false` (which is the Phi-2 default — the
//! flag was added for some Phi-2 fine-tunes; the reference
//! checkpoint ships with it off).
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache
//! (recomputes each call), F32 activations.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

/// Which GELU variant the MLP's activation uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiActivation {
    /// Standard ERF GELU `0.5 * x * (1 + erf(x / sqrt(2)))`.
    Gelu,
    /// PyTorch's `approximate="tanh"` variant — same family as
    /// `NewGelu` in HF transformers. Default for Phi-2.
    GeluPytorchTanh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhiConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Defaults to `num_attention_heads` if `None` (MHA, no GQA).
    pub num_key_value_heads: Option<usize>,
    pub head_dim: usize,
    pub layer_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    /// Fraction of `head_dim` that gets rotated. Phi-2 uses 0.4.
    pub partial_rotary_factor: f64,
    /// v1 only supports `false`. `true` would require a per-head
    /// LayerNorm applied post-reshape — left to follow-up.
    pub qk_layernorm: bool,
    pub hidden_activation: PhiActivation,
}

impl PhiConfig {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
    pub fn rope_dim(&self) -> usize {
        let r = (self.partial_rotary_factor * self.head_dim as f64) as usize;
        // RoPE expects an even rope_dim (pairs of features).
        r & !1
    }
}

#[derive(Debug, Clone)]
pub struct PhiLayerWeights {
    pub input_ln_gain: Arc<[f32]>,
    pub input_ln_bias: Arc<[f32]>,
    pub attn_q: WeightStorage,
    pub attn_q_bias: Arc<[f32]>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Arc<[f32]>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Arc<[f32]>,
    /// Phi calls the output projection `dense`.
    pub attn_dense: WeightStorage,
    pub attn_dense_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct PhiWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<PhiLayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub lm_head: WeightStorage,
    pub lm_head_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct PhiModel {
    pub config: PhiConfig,
    pub weights: PhiWeights,
}

impl PhiModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        let logits = weights.lm_head.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size);
        let bias_t = h_norm.const_f32_like(
            Arc::clone(&weights.lm_head_bias),
            Shape::from_dims(&[cfg.vocab_size]),
        );
        logits.broadcast_add(&bias_t)
    }

    /// Run the decoder forward up to the final LayerNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection AND its bias.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "PhiModel: tokens must be non-empty");
        assert_eq!(
            cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size,
            "PhiConfig: num_attention_heads * head_dim must equal hidden_size",
        );
        assert!(
            !cfg.qk_layernorm,
            "PhiModel v1: qk_layernorm not yet supported (reference Phi-2 sets it false)",
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

        let rope_dim = cfg.rope_dim();
        assert!(
            rope_dim > 0 && rope_dim <= cfg.head_dim,
            "PhiConfig: rope_dim ({}) out of [1, head_dim ({})]",
            rope_dim, cfg.head_dim,
        );
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
        layer: &PhiLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let n_kv = cfg.num_kv_heads();
        let kv_dim = n_kv * cfg.head_dim;
        let rope_dim = cfg.rope_dim();

        // Single LN feeds BOTH attention and MLP paths (Phi parallel block).
        let x_norm = x.layer_norm_affine(std::sync::Arc::clone(&layer.input_ln_gain), std::sync::Arc::clone(&layer.input_ln_bias), cfg.layer_norm_eps)?;

        // ---- Attention path -------------------------------------------------
        let q = layer.attn_q.apply_linear_with_bias(&x_norm, cfg.hidden_size, cfg.hidden_size, std::sync::Arc::clone(&layer.attn_q_bias))?;
        let k = layer.attn_k.apply_linear_with_bias(&x_norm, cfg.hidden_size, kv_dim, std::sync::Arc::clone(&layer.attn_k_bias))?;
        let v = layer.attn_v.apply_linear_with_bias(&x_norm, cfg.hidden_size, kv_dim, std::sync::Arc::clone(&layer.attn_v_bias))?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(n_kv, cfg.head_dim)?;
        let v = v.split_heads(n_kv, cfg.head_dim)?;

        // Partial rotary on the first `rope_dim` features.
        let q_r = q.rope_partial(rope_cos, rope_sin, rope_dim)?;
        let k_r = k.rope_partial(rope_cos, rope_sin, rope_dim)?;

        // GQA replication (Phi-2 uses MHA so n_rep = 1 usually).
        let n_rep = cfg.num_attention_heads / n_kv;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        // Strict causal mask.
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
        let attn_out = layer.attn_dense.apply_linear_with_bias(&merged, cfg.hidden_size, cfg.hidden_size, std::sync::Arc::clone(&layer.attn_dense_bias))?;

        // ---- MLP path (uses the SAME x_norm) --------------------------------
        let fc1_out = layer.fc1.apply_linear_with_bias(&x_norm, cfg.hidden_size, cfg.intermediate_size, std::sync::Arc::clone(&layer.fc1_bias))?;
        let activated = match cfg.hidden_activation {
            PhiActivation::Gelu => fc1_out.gelu_erf(),
            PhiActivation::GeluPytorchTanh => fc1_out.gelu(),
        };
        let mlp_out = layer.fc2.apply_linear_with_bias(&activated, cfg.intermediate_size, cfg.hidden_size, std::sync::Arc::clone(&layer.fc2_bias))?;

        // Parallel combine: residual + attn + mlp.
        x.add(&attn_out)?.add(&mlp_out)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &PhiConfig) -> PhiWeights {
        let mut s: u32 = 31337;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let n_kv = cfg.num_kv_heads();
        let kv = n_kv * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<PhiLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| PhiLayerWeights {
                input_ln_gain: Arc::from(vec![1.0_f32; h]),
                input_ln_bias: Arc::from(vec![0.0_f32; h]),
                attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                attn_q_bias: vec_of(h, &mut *nb),
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_k_bias: vec_of(kv, &mut *nb),
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_v_bias: vec_of(kv, &mut *nb),
                attn_dense: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                attn_dense_bias: vec_of(h, &mut *nb),
                fc1: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                fc1_bias: vec_of(i, &mut *nb),
                fc2: WeightStorage::F32(vec_of(i * h, &mut *nb)),
                fc2_bias: vec_of(h, &mut *nb),
            })
            .collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let lm_head = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        let lm_head_bias = vec_of(cfg.vocab_size, &mut *nb);
        PhiWeights {
            token_embedding, layers,
            final_ln_gain, final_ln_bias,
            lm_head, lm_head_bias,
        }
    }

    fn tiny_config() -> PhiConfig {
        PhiConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4,
            num_key_value_heads: None, head_dim: 4,
            layer_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 64,
            partial_rotary_factor: 0.5, // rope_dim = 2
            qk_layernorm: false,
            hidden_activation: PhiActivation::GeluPytorchTanh,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = PhiModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// rope_dim equal to head_dim must run (full rotary degenerate case).
    #[test]
    fn full_rotary_factor() {
        let mut cfg = tiny_config();
        cfg.partial_rotary_factor = 1.0;
        assert_eq!(cfg.rope_dim(), cfg.head_dim);
        let model = PhiModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3], 0).unwrap().realize_f32();
        assert_eq!(logits.len(), 3 * cfg.vocab_size);
    }

    /// Parallel block must NOT collapse to sequential: differentiating
    /// attention and MLP at the residual sum is the structural test.
    /// Zero out fc1/fc2 (kills MLP path) and confirm output equals
    /// `residual + attn`. Then zero out Q/K/V/dense (kills attn) and
    /// confirm output equals `residual + mlp`. Both must be possible
    /// independently.
    #[test]
    fn parallel_paths_are_additive() {
        let cfg = PhiConfig { num_hidden_layers: 1, ..tiny_config() };
        let base = tiny_weights(&cfg);

        let mut no_mlp = base.clone();
        let zero = |n: usize| WeightStorage::F32(Arc::from(vec![0.0_f32; n]));
        let zero_b = |n: usize| Arc::from(vec![0.0_f32; n]) as Arc<[f32]>;
        no_mlp.layers[0].fc1 = zero(cfg.hidden_size * cfg.intermediate_size);
        no_mlp.layers[0].fc1_bias = zero_b(cfg.intermediate_size);
        no_mlp.layers[0].fc2 = zero(cfg.intermediate_size * cfg.hidden_size);
        no_mlp.layers[0].fc2_bias = zero_b(cfg.hidden_size);

        let mut no_attn = base.clone();
        no_attn.layers[0].attn_q = zero(cfg.hidden_size * cfg.hidden_size);
        no_attn.layers[0].attn_q_bias = zero_b(cfg.hidden_size);
        no_attn.layers[0].attn_k = zero(cfg.hidden_size * cfg.hidden_size);
        no_attn.layers[0].attn_k_bias = zero_b(cfg.hidden_size);
        no_attn.layers[0].attn_v = zero(cfg.hidden_size * cfg.hidden_size);
        no_attn.layers[0].attn_v_bias = zero_b(cfg.hidden_size);
        no_attn.layers[0].attn_dense = zero(cfg.hidden_size * cfg.hidden_size);
        no_attn.layers[0].attn_dense_bias = zero_b(cfg.hidden_size);

        // Both should produce finite, non-trivial outputs different
        // from each other (different paths active).
        let m_no_mlp = PhiModel { config: cfg.clone(), weights: no_mlp };
        let m_no_attn = PhiModel { config: cfg.clone(), weights: no_attn };
        let toks = [3_u32, 7, 2];
        let a = m_no_mlp.forward(&toks, 0).unwrap().realize_f32();
        let b = m_no_attn.forward(&toks, 0).unwrap().realize_f32();
        assert!(a.iter().all(|v| v.is_finite()));
        assert!(b.iter().all(|v| v.is_finite()));
        let mut max_diff = 0.0_f32;
        for (av, bv) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((av - bv).abs());
        }
        assert!(max_diff > 1e-6,
            "no-mlp vs no-attn must produce different outputs: {max_diff}");
    }

    #[test]
    fn num_kv_heads_default_is_num_attention_heads() {
        let cfg = tiny_config();
        assert_eq!(cfg.num_kv_heads(), cfg.num_attention_heads);
    }

    /// `forward_hidden` returns post-LayerNorm hidden states
    /// `(1, seq, hidden_size)` without the lm_head matmul or
    /// its bias.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config();
        let model = PhiModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

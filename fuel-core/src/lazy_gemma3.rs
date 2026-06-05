//! Gemma 3 decoder ported to the lazy-graph API.
//!
//! Phase D specialized port. Gemma 3 keeps Gemma's family flavor
//! (offset RmsNorm, sqrt(hidden_size) embedding scale, GELU gated
//! FFN) and adds four architectural twists over Gemma 1/2:
//!
//!   1. **Alternating local/global attention** — layers cycle
//!      through `sliding_window_pattern` slots. Layer `i` uses a
//!      sliding-window mask + the *local* RoPE base when
//!      `(i + 1) % sliding_window_pattern > 0`, and a full causal
//!      mask + the global RoPE base otherwise. This matches the
//!      eager Gemma3 forward in `gemma3.rs`.
//!   2. **Dual RoPE bases** — `rope_theta` for global layers and
//!      `rope_local_base_freq` for sliding layers. We precompute
//!      both tables once and pick per-layer.
//!   3. **Attention-score soft-capping** — when
//!      `attn_logit_softcapping` is `Some(sc)`, scaled scores are
//!      passed through `((scores / sc).tanh() * sc)` before the
//!      mask add. Same shape of soft-cap as Gemma 2, exposed via
//!      config rather than hardcoded.
//!   4. **Final-logit soft-capping** — same shape applied to the
//!      output of `lm_head` when `final_logit_softcapping` is set.
//!
//! Other carries from Gemma 1: offset RmsNorm `(gain + 1)`, embed
//! scaled by `sqrt(hidden_size)`, GELU (configurable variant) gated
//! FFN, optional Q/K/V/O biases via `attention_bias`.
//!
//! Gemma 3 also adds per-head Q/K RmsNorm (post-reshape, on
//! `head_dim`), four norms per block (input + post-attn + pre-FFN
//! + post-FFN), and tied lm_head/embeddings. `num_heads * head_dim`
//! is **not** required to equal `hidden_size` — Gemma 3 uses
//! independent attention head/embedding sizes (e.g. 1B has
//! `hidden_size=1152`, `num_heads=4`, `head_dim=256`).
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache
//! (recomputes each call), F32 activations. The lm_head reuses
//! `token_embedding` as a tied projection.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

pub use crate::lazy_gemma::GemmaActivation;

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_local_base_freq: f64,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    /// Layers `i` where `(i + 1) % sliding_window_pattern == 0` use
    /// full causal attention + the global RoPE base; the others
    /// use sliding-window + the local RoPE base. The reference
    /// 4B/12B/27B checkpoints set this to 6 (5 local + 1 global).
    pub sliding_window_pattern: usize,
    pub attention_bias: bool,
    pub hidden_activation: GemmaActivation,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct Gemma3LayerWeights {
    pub attn_q: WeightStorage,
    pub attn_q_bias: Option<Arc<[f32]>>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Option<Arc<[f32]>>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Option<Arc<[f32]>>,
    pub attn_o: WeightStorage,
    pub attn_o_bias: Option<Arc<[f32]>>,
    /// Per-head Q RmsNorm gain on `head_dim` (offset `(gain + 1)`).
    pub q_norm_gain: Arc<[f32]>,
    /// Per-head K RmsNorm gain on `head_dim` (offset `(gain + 1)`).
    pub k_norm_gain: Arc<[f32]>,
    pub input_norm_gain: Arc<[f32]>,
    pub post_attn_norm_gain: Arc<[f32]>,
    pub pre_ffn_norm_gain: Arc<[f32]>,
    pub post_ffn_norm_gain: Arc<[f32]>,
    pub ffn_gate: WeightStorage,
    pub ffn_up: WeightStorage,
    pub ffn_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Gemma3Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Gemma3LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Gemma3Model {
    pub config: Gemma3Config,
    pub weights: Gemma3Weights,
}

impl Gemma3Model {
    /// True if layer `i` uses sliding-window + local RoPE.
    fn layer_uses_sliding(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.config.sliding_window_pattern > 0
    }

    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        let lm_head = WeightStorage::F32(weights.token_embedding.clone());
        let logits = lm_head.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size);
        let logits = match cfg.final_logit_softcapping {
            None => logits,
            Some(sc) => logits.mul_scalar(1.0 / sc).tanh().mul_scalar(sc),
        };
        Ok(logits)
    }

    /// Run the decoder forward up to the final offset RmsNorm
    /// and return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the tied `lm_head` matmul AND the final logit
    /// softcapping. Gemma3-specific: dual-RoPE (global + local)
    /// + per-layer sliding-window pattern + sqrt(hidden_size)
    /// embedding scaling — all honored by the shared backbone.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "Gemma3Model: tokens must be non-empty");
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads, 0,
            "Gemma3Config: num_attention_heads must be a multiple of num_key_value_heads",
        );
        assert!(
            cfg.sliding_window_pattern > 0,
            "Gemma3Config: sliding_window_pattern must be > 0",
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
        h = h.mul_scalar((cfg.hidden_size as f64).sqrt());

        let (rope_cos_g, rope_sin_g) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, cfg.head_dim,
        );
        let (rope_cos_l, rope_sin_l) = h.rope_tables_const(
            cfg.rope_local_base_freq, start_pos, seq, cfg.head_dim,
        );

        let full_mask = self.build_mask(&h, seq, None);
        let sliding_mask = self.build_mask(&h, seq, Some(cfg.sliding_window));

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let uses_window = self.layer_uses_sliding(layer_idx);
            let (rope_cos, rope_sin) = if uses_window {
                (&rope_cos_l, &rope_sin_l)
            } else {
                (&rope_cos_g, &rope_sin_g)
            };
            let mask = if uses_window { &sliding_mask } else { &full_mask };
            h = self.apply_layer(&h, layer, rope_cos, rope_sin, mask)?;
        }
        h.rms_norm_affine_with_offset(&weights.final_norm_gain, 1.0, cfg.rms_norm_eps)
    }

    fn build_mask(&self, anchor: &LazyTensor, seq: usize, sliding: Option<usize>) -> LazyTensor {
        let window = sliding.unwrap_or(seq + 1);
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
        layer: &Gemma3LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        // Pre-attention offset RmsNorm.
        let residual = x.clone();
        let x_norm = x.rms_norm_affine_with_offset(&layer.input_norm_gain, 1.0, cfg.rms_norm_eps)?;

        // Q / K / V projections; note Q goes to num_heads*head_dim
        // which is NOT necessarily equal to hidden_size.
        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, q_dim).add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        // (b, seq, n_heads, head_dim) -> (b, n_heads, seq, head_dim).
        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        // Per-head Q/K RmsNorm on head_dim (POST-reshape, like eager Gemma3).
        let q = q.rms_norm_affine_with_offset(&layer.q_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let k = k.rms_norm_affine_with_offset(&layer.k_norm_gain, 1.0, cfg.rms_norm_eps)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        // GQA replication: expand K, V to num_attention_heads.
        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        // Attention-score soft-cap (per-layer config).
        let scores_capped = match cfg.attn_logit_softcapping {
            None => scores_scaled,
            Some(sc) => scores_scaled.mul_scalar(1.0 / sc).tanh().mul_scalar(sc),
        };
        let scores_masked = scores_capped.broadcast_add(mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer.attn_o.apply_linear(&merged, q_dim, cfg.hidden_size).add_optional_trailing_bias(layer.attn_o_bias.as_ref())?;
        // post_attention_layernorm wraps the attn output BEFORE the residual add.
        let attn_out_norm = attn_out.rms_norm_affine_with_offset(&layer.post_attn_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let h1 = residual.add(&attn_out_norm)?;

        // Pre-FFN offset RmsNorm.
        let residual2 = h1.clone();
        let h1_norm = h1.rms_norm_affine_with_offset(&layer.pre_ffn_norm_gain, 1.0, cfg.rms_norm_eps)?;

        // GELU gated FFN.
        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let activated = match cfg.hidden_activation {
            GemmaActivation::Gelu => gate.gelu_erf(),
            GemmaActivation::GeluPytorchTanh => gate.gelu(),
        };
        let ffn_in = activated.mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&ffn_in, cfg.intermediate_size, cfg.hidden_size);
        // post_feedforward_layernorm wraps the FFN output BEFORE the residual add.
        let ffn_out_norm = ffn_out.rms_norm_affine_with_offset(&layer.post_ffn_norm_gain, 1.0, cfg.rms_norm_eps)?;

        residual2.add(&ffn_out_norm)
    }
}

/// Gemma's offset RmsNorm: `y = (x / rms) * (gain + 1)`.


#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &Gemma3Config) -> Gemma3Weights {
        let mut s: u32 = 5151;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let mut next_box: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *next_box);
        let layers: Vec<Gemma3LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| Gemma3LayerWeights {
                attn_q: WeightStorage::F32(vec_of(h * q_dim, &mut *next_box)),
                attn_q_bias: if cfg.attention_bias { Some(vec_of(q_dim, &mut *next_box)) } else { None },
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
                attn_k_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *next_box)) } else { None },
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
                attn_v_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *next_box)) } else { None },
                attn_o: WeightStorage::F32(vec_of(q_dim * h, &mut *next_box)),
                attn_o_bias: if cfg.attention_bias { Some(vec_of(h, &mut *next_box)) } else { None },
                q_norm_gain: Arc::from(vec![0.05_f32; cfg.head_dim]),
                k_norm_gain: Arc::from(vec![0.05_f32; cfg.head_dim]),
                input_norm_gain: Arc::from(vec![0.05_f32; h]),
                post_attn_norm_gain: Arc::from(vec![0.05_f32; h]),
                pre_ffn_norm_gain: Arc::from(vec![0.05_f32; h]),
                post_ffn_norm_gain: Arc::from(vec![0.05_f32; h]),
                ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *next_box)),
                ffn_up: WeightStorage::F32(vec_of(h * i, &mut *next_box)),
                ffn_down: WeightStorage::F32(vec_of(i * h, &mut *next_box)),
            })
            .collect();
        let final_norm_gain = Arc::from(vec![0.05_f32; h]);
        Gemma3Weights { token_embedding, layers, final_norm_gain }
    }

    fn tiny_config() -> Gemma3Config {
        Gemma3Config {
            vocab_size: 32,
            // Pick num_heads * head_dim != hidden_size to exercise
            // independent attention/embedding dims like real Gemma3.
            hidden_size: 24,
            intermediate_size: 32,
            num_hidden_layers: 4, // exercise both global + local layers (pattern=3)
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,           // q_dim=16, kv_dim=8 — neither matches hidden_size.
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_local_base_freq: 10_000.0, // same as global for the "tables match" test
            max_position_embeddings: 64,
            sliding_window: 3,
            sliding_window_pattern: 3,
            attention_bias: false,
            hidden_activation: GemmaActivation::GeluPytorchTanh,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = Gemma3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// The sliding-window pattern should actually change the
    /// behavior: forcing pattern=1 makes every layer use FULL
    /// causal + global RoPE; pattern=N where N != 1 introduces
    /// local layers. With identical weights and different
    /// pattern values, outputs must differ.
    #[test]
    fn pattern_changes_output() {
        let mut cfg_a = tiny_config();
        cfg_a.sliding_window_pattern = 1; // all global
        let mut cfg_b = tiny_config();
        cfg_b.sliding_window_pattern = 3; // 2 local + 1 global per cycle
        // Force the local RoPE base to differ from the global one
        // so picking the wrong table changes the output.
        cfg_a.rope_local_base_freq = 50_000.0;
        cfg_b.rope_local_base_freq = 50_000.0;
        // Reuse the SAME weights for both.
        let weights = tiny_weights(&cfg_a);
        let m_a = Gemma3Model { config: cfg_a.clone(), weights: weights.clone() };
        let m_b = Gemma3Model { config: cfg_b.clone(), weights };
        let toks: Vec<u32> = vec![3, 7, 2, 9, 1];
        let a = m_a.forward(&toks, 0).unwrap().realize_f32();
        let b = m_b.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (av, bv) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((av - bv).abs());
        }
        assert!(max_diff > 1e-6,
            "pattern change must alter output, max_diff = {max_diff}");
    }

    /// With sliding_window_pattern=1 (all global), and local RoPE
    /// base equal to global RoPE base, the two RoPE table sets
    /// are identical — so the result must match an equivalent
    /// "no soft-cap, no sliding" baseline up to soft-cap effect.
    ///
    /// We assert here that the soft-cap is active (changing it
    /// changes the output).
    #[test]
    fn attn_softcap_changes_output() {
        let mut cfg_no = tiny_config();
        cfg_no.attn_logit_softcapping = None;
        let mut cfg_yes = tiny_config();
        cfg_yes.attn_logit_softcapping = Some(20.0);
        let weights = tiny_weights(&cfg_no);
        let m_no = Gemma3Model { config: cfg_no, weights: weights.clone() };
        let m_yes = Gemma3Model { config: cfg_yes, weights };
        let toks: Vec<u32> = vec![1, 2, 3];
        let a = m_no.forward(&toks, 0).unwrap().realize_f32();
        let b = m_yes.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (av, bv) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((av - bv).abs());
        }
        assert!(max_diff > 1e-6,
            "attn soft-cap must alter output, max_diff = {max_diff}");
    }

    /// Final-logit soft-cap must change output (bounds the
    /// pre-softmax logits).
    #[test]
    fn final_softcap_changes_output() {
        let mut cfg_no = tiny_config();
        cfg_no.final_logit_softcapping = None;
        let mut cfg_yes = tiny_config();
        cfg_yes.final_logit_softcapping = Some(5.0);
        let weights = tiny_weights(&cfg_no);
        let m_no = Gemma3Model { config: cfg_no, weights: weights.clone() };
        let m_yes = Gemma3Model { config: cfg_yes, weights };
        let toks: Vec<u32> = vec![4, 5, 6];
        let a = m_no.forward(&toks, 0).unwrap().realize_f32();
        let b = m_yes.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (av, bv) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((av - bv).abs());
        }
        assert!(max_diff > 1e-6,
            "final soft-cap must alter output, max_diff = {max_diff}");
    }

    /// With sliding_window_pattern=2 and 4 layers, layers 0 and 2
    /// are local (sliding) and layers 1 and 3 are global. Verify
    /// `layer_uses_sliding` matches.
    #[test]
    fn layer_pattern_assignment() {
        let mut cfg = tiny_config();
        cfg.sliding_window_pattern = 2;
        let model = Gemma3Model { config: cfg, weights: tiny_weights(&tiny_config()) };
        // (i + 1) % 2 > 0  →  i is even (0, 2 → local) ; odd (1, 3 → global)
        assert!(model.layer_uses_sliding(0));
        assert!(!model.layer_uses_sliding(1));
        assert!(model.layer_uses_sliding(2));
        assert!(!model.layer_uses_sliding(3));
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config();
        let model = Gemma3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

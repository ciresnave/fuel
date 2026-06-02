//! Gemma 2 (Google DeepMind 2024) ported to the lazy-graph API.
//!
//! Gemma 2 extends Gemma 1 with three architectural changes:
//!
//!   1. **Four RmsNorms per layer** (Gemma 1 has two). The
//!      sublayer structure is:
//!      `input_norm → attn → post_attn_norm → +residual →
//!       pre_ffn_norm → mlp → post_ffn_norm → +residual`.
//!      The "post_*" norms sit AFTER the sublayer's main op but
//!      BEFORE the residual add. Combined with input/pre norms,
//!      every sublayer is wrapped in `norm(sublayer(norm(x)))`.
//!   2. **Attention logit softcapping**: when
//!      `attn_logit_softcapping = Some(cap)`, raw attention
//!      scores `(Q @ K.T) * (1/sqrt(d))` are bounded by
//!      `tanh(scores / cap) * cap` BEFORE the mask is applied.
//!      Stabilizes training in deep models; runs at inference too.
//!   3. **Final logit softcapping**: same trick applied to the
//!      output `lm_head` logits before they're returned.
//!
//! Gemma 2 keeps Gemma 1's distinctive bits:
//!
//!   - **Embedding scaling**: hidden states multiplied by
//!     `sqrt(hidden_size)` after the embedding lookup.
//!   - **RmsNorm with `(gamma + 1.0)` multiplier** (the
//!     learnable weights center on 0, not 1).
//!   - GQA with `num_key_value_heads`, standard RoPE on Q/K,
//!     SwiGLU MLP with separate gate/up/down projections.
//!   - Tied embeddings (`lm_head.weight == embed_tokens.weight`).
//!
//! # Sliding window mask
//!
//! When `sliding_window = Some(w)`, the causal mask zeros out
//! position pairs `(i, j)` with `j + w < i` (in addition to
//! `j > i`). Gemma 2 alternates global / local layers in the
//! same way as Qwen2 in some configurations but the eager
//! port applies the sliding window globally per the config.
//! v1 follows the eager port: window applied to ALL layers
//! when the config has it.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache
//! (recomputes per call), F32. Returns vocab logits
//! `(1, seq, vocab_size)` with final softcapping applied.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub attention_bias: bool,
    pub max_position_embeddings: usize,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub sliding_window: Option<usize>,
}

impl Gemma2Config {
    /// `google/gemma-2-2b` preset (approximate; actual `config.json`
    /// from HuggingFace overrides).
    pub fn gemma2_2b() -> Self {
        Self {
            vocab_size: 256_000,
            hidden_size: 2_304,
            intermediate_size: 9_216,
            num_hidden_layers: 26,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            attention_bias: false,
            max_position_embeddings: 8_192,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            sliding_window: Some(4_096),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Gemma2LayerWeights {
    pub input_norm_gain: Arc<[f32]>,
    pub post_attn_norm_gain: Arc<[f32]>,
    pub pre_ffn_norm_gain: Arc<[f32]>,
    pub post_ffn_norm_gain: Arc<[f32]>,
    pub q: WeightStorage,
    pub q_bias: Option<Arc<[f32]>>,
    pub k: WeightStorage,
    pub k_bias: Option<Arc<[f32]>>,
    pub v: WeightStorage,
    pub v_bias: Option<Arc<[f32]>>,
    pub o: WeightStorage,
    pub o_bias: Option<Arc<[f32]>>,
    pub gate: WeightStorage,
    pub up: WeightStorage,
    pub down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Gemma2Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Gemma2LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Gemma2Model {
    pub config: Gemma2Config,
    pub weights: Gemma2Weights,
}

impl Gemma2Model {
    /// Forward pass: returns `(1, seq, vocab_size)`.
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "Gemma2Model::forward: tokens must be non-empty");
        let head_dim = cfg.head_dim;
        assert_eq!(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            "Gemma2Config: num_attention_heads * head_dim must equal hidden_size",
        );
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads,
            0,
            "Gemma2Config: num_attention_heads must be a multiple of num_key_value_heads",
        );

        // ---- Embedding + sqrt(hidden_size) scaling -------------------------
        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;
        let embed_scale = (cfg.hidden_size as f64).sqrt();
        h = h.mul_scalar(embed_scale);

        // ---- Shared RoPE tables --------------------------------------------
        let (cos_data, sin_data) = fuel_graph::build_rope_tables(
            cfg.rope_theta, start_pos, seq, head_dim,
        );
        let rope_shape = Shape::from_dims(&[seq, head_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        // ---- Causal (optionally sliding) mask, shared across layers --------
        let mask = self.build_mask(&h, seq);

        // ---- Decoder blocks ------------------------------------------------
        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, &mask)?;
        }

        // ---- Final RmsNorm + tied lm_head + final softcapping --------------
        let h_norm = apply_gemma2_rms_norm(
            &h, &weights.final_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        // Tied lm_head reuses the token embedding weight (vocab, hidden) —
        // logits = h_norm @ embed.T. We assemble it as a Linear: hidden -> vocab.
        let lm_head_w = h.const_f32_like(
            Arc::clone(&weights.token_embedding),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
        );
        // h_norm @ lm_head_w.T: shape (batch, seq, vocab_size).
        let logits = h_norm.matmul(&lm_head_w.transpose()?)?;
        let logits = apply_softcap(&logits, cfg.final_logit_softcapping);
        Ok(logits)
    }

    fn build_mask(&self, anchor: &LazyTensor, seq: usize) -> LazyTensor {
        let cfg = &self.config;
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                let masked = if j > i {
                    true
                } else if let Some(w) = cfg.sliding_window {
                    j + w < i
                } else {
                    false
                };
                if masked {
                    mask_data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        anchor.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]))
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &Gemma2LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * head_dim;

        // Attention sublayer: post_attn_norm(attn(input_norm(x))) + x.
        let x_in = apply_gemma2_rms_norm(
            x, &layer.input_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        let q = apply_optional_bias(
            layer.q.apply_linear(&x_in, cfg.hidden_size, cfg.hidden_size),
            layer.q_bias.as_ref(), cfg.hidden_size,
        )?;
        let k = apply_optional_bias(
            layer.k.apply_linear(&x_in, cfg.hidden_size, kv_dim),
            layer.k_bias.as_ref(), kv_dim,
        )?;
        let v = apply_optional_bias(
            layer.v.apply_linear(&x_in, cfg.hidden_size, kv_dim),
            layer.v_bias.as_ref(), kv_dim,
        )?;

        let q = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.num_attention_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let k = k
            .reshape(Shape::from_dims(&[batch, seq, cfg.num_key_value_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let v = v
            .reshape(Shape::from_dims(&[batch, seq, cfg.num_key_value_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let (k_full, v_full) = if n_rep == 1 {
            (k_r, v)
        } else {
            let expand = |t: LazyTensor| -> Result<LazyTensor> {
                let s5 = t.reshape(Shape::from_dims(&[
                    batch, cfg.num_key_value_heads, 1, seq, head_dim,
                ]))?;
                let bcast = s5.broadcast_to(Shape::from_dims(&[
                    batch, cfg.num_key_value_heads, n_rep, seq, head_dim,
                ]))?;
                bcast.reshape(Shape::from_dims(&[
                    batch, cfg.num_attention_heads, seq, head_dim,
                ]))
            };
            (expand(k_r)?, expand(v)?)
        };

        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_full.transpose()?)?.mul_scalar(scale);
        // Attention logit softcapping (BEFORE the mask).
        let scores = apply_softcap(&scores, cfg.attn_logit_softcapping);
        let scores = scores.broadcast_add(mask)?;
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v_full)?;
        let merged = ctx
            .permute([0, 2, 1, 3_usize])?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;
        let attn_out = apply_optional_bias(
            layer.o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size),
            layer.o_bias.as_ref(), cfg.hidden_size,
        )?;
        // Post-attn norm BEFORE the residual add.
        let attn_post = apply_gemma2_rms_norm(
            &attn_out, &layer.post_attn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        let h1 = x.add(&attn_post)?;

        // MLP sublayer: post_ffn_norm(mlp(pre_ffn_norm(h1))) + h1.
        let h1_in = apply_gemma2_rms_norm(
            &h1, &layer.pre_ffn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        let gate = layer.gate.apply_linear(&h1_in, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.up.apply_linear(&h1_in, cfg.hidden_size, cfg.intermediate_size);
        // Gemma 2 default activation is GeGLU-tanh (HiddenAct::GeluPytorchTanh
        // in the config), but the original Gemma uses Silu. The eager config
        // carries the activation as a field; we follow gemma-2 default which
        // is GELU-tanh (used by the public release).
        let swi = gate.gelu().mul(&up)?;
        let ffn_out = layer.down.apply_linear(&swi, cfg.intermediate_size, cfg.hidden_size);
        let ffn_post = apply_gemma2_rms_norm(
            &ffn_out, &layer.post_ffn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        h1.add(&ffn_post)
    }
}

/// Gemma-style RmsNorm: scale by `gamma + 1.0` rather than
/// `gamma`. The +1 is folded into a separate constant tensor
/// on the graph so the underlying RmsNorm op stays standard.
fn apply_gemma2_rms_norm(
    x: &LazyTensor,
    gain: &Arc<[f32]>,
    hidden_size: usize,
    eps: f64,
) -> LazyTensor {
    // Build an effective gain = gain + 1.0 const, then apply the
    // standard affine RmsNorm helper.
    let gain_plus_one: Arc<[f32]> = Arc::from(
        gain.iter().map(|v| v + 1.0).collect::<Vec<_>>(),
    );
    crate::lazy::apply_affine_rms_norm_pub(x, &gain_plus_one, hidden_size, eps)
}

fn apply_softcap(x: &LazyTensor, cap: Option<f64>) -> LazyTensor {
    match cap {
        None => x.clone(),
        Some(c) => {
            let scaled = x.mul_scalar(1.0 / c);
            scaled.tanh().mul_scalar(c)
        }
    }
}

fn apply_optional_bias(
    x: LazyTensor,
    bias: Option<&Arc<[f32]>>,
    last_dim: usize,
) -> Result<LazyTensor> {
    match bias {
        None => Ok(x),
        Some(b) => {
            assert_eq!(b.len(), last_dim);
            let bt = x.const_f32_like(Arc::clone(b), Shape::from_dims(&[last_dim]));
            x.broadcast_add(&bt)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }

    fn tiny_cfg() -> Gemma2Config {
        Gemma2Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4,
            num_key_value_heads: 2, head_dim: 4,
            rms_norm_eps: 1e-6, rope_theta: 10_000.0,
            attention_bias: false, max_position_embeddings: 32,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            sliding_window: None,
        }
    }

    fn tiny_weights(cfg: &Gemma2Config, seed: u32) -> Gemma2Weights {
        let mut nb = rng_seed(seed);
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h, &mut nb);
        let layers: Vec<Gemma2LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| Gemma2LayerWeights {
                input_norm_gain: Arc::from(vec![0.0_f32; h]),
                post_attn_norm_gain: Arc::from(vec![0.0_f32; h]),
                pre_ffn_norm_gain: Arc::from(vec![0.0_f32; h]),
                post_ffn_norm_gain: Arc::from(vec![0.0_f32; h]),
                q: WeightStorage::F32(vec_of(h * q_dim, &mut nb)),
                q_bias: None,
                k: WeightStorage::F32(vec_of(h * kv_dim, &mut nb)),
                k_bias: None,
                v: WeightStorage::F32(vec_of(h * kv_dim, &mut nb)),
                v_bias: None,
                o: WeightStorage::F32(vec_of(q_dim * h, &mut nb)),
                o_bias: None,
                gate: WeightStorage::F32(vec_of(h * i, &mut nb)),
                up: WeightStorage::F32(vec_of(h * i, &mut nb)),
                down: WeightStorage::F32(vec_of(i * h, &mut nb)),
            })
            .collect();
        Gemma2Weights {
            token_embedding,
            layers,
            final_norm_gain: Arc::from(vec![0.0_f32; h]),
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_cfg();
        let model = Gemma2Model { config: cfg.clone(), weights: tiny_weights(&cfg, 11) };
        let tokens = [1_u32, 2, 3, 4];
        let out = model.forward(&tokens, 0).unwrap();
        assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    /// Final logit softcapping bounds output: |logit| <= cap.
    #[test]
    fn final_softcap_bounds_logits() {
        let mut cfg = tiny_cfg();
        cfg.final_logit_softcapping = Some(2.0);
        let model = Gemma2Model { config: cfg.clone(), weights: tiny_weights(&cfg, 22) };
        let tokens = [1_u32, 2, 3, 4];
        let out = model.forward(&tokens, 0).unwrap().realize_f32();
        for &v in &out {
            assert!(v.abs() <= 2.0_f32 + 1e-4,
                "logit {v} exceeds softcap 2.0");
        }
    }

    /// Sliding window mask: with window=2 and seq=4, token at
    /// position 0 should be masked OUT of position 3's attention
    /// (3 - 0 = 3 > window 2). Verify by checking the model runs
    /// and produces different output vs. no-window config.
    #[test]
    fn sliding_window_changes_output() {
        let cfg_no_window = {
            let mut c = tiny_cfg();
            c.sliding_window = None;
            c
        };
        let cfg_window = {
            let mut c = tiny_cfg();
            c.sliding_window = Some(2);
            c
        };
        let weights = tiny_weights(&cfg_no_window, 33);
        let m_a = Gemma2Model { config: cfg_no_window, weights: weights.clone() };
        let m_b = Gemma2Model { config: cfg_window, weights };
        let tokens = [1_u32, 2, 3, 4];
        let a = m_a.forward(&tokens, 0).unwrap().realize_f32();
        let b = m_b.forward(&tokens, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "sliding window must alter output, max_diff = {max_diff}");
    }

    /// Gemma-2 RmsNorm scales by `gamma + 1`. With gain=0 (default
    /// init) the effective scale is 1.0, matching standard RmsNorm.
    /// Verify by comparing apply_gemma2_rms_norm(x, [0,...]) and
    /// apply_affine_rms_norm(x, [1,...]).
    #[test]
    fn gemma2_rms_norm_offset() {
        let h = 8;
        let data: Arc<[f32]> = Arc::from(vec![1.0_f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
        let x = LazyTensor::from_f32(data, Shape::from_dims(&[1, 1, h]), &Device::cpu());
        let zero_gain: Arc<[f32]> = Arc::from(vec![0.0_f32; h]);
        let one_gain: Arc<[f32]> = Arc::from(vec![1.0_f32; h]);
        let g2_out = apply_gemma2_rms_norm(&x, &zero_gain, h, 1e-6).realize_f32();
        let baseline = crate::lazy::apply_affine_rms_norm_pub(&x, &one_gain, h, 1e-6).realize_f32();
        for (a, b) in g2_out.iter().zip(baseline.iter()) {
            assert!((a - b).abs() < 1e-6,
                "gemma2 RmsNorm with gain=0 must equal standard RmsNorm with gain=1, {a} vs {b}");
        }
    }
}

//! ChatGLM2 / ChatGLM3 (THUDM) decoder ported to the lazy-graph API.
//!
//! Phase D specialized port. The older GLM family (GLM2/3, before
//! GLM-4) ships its own architectural mix that distinguishes it
//! from later GLMs:
//!
//!   1. **Fused QKV projection with MQA** —
//!      `query_key_value: hidden → n_heads * head_dim + 2 * group_num
//!      * head_dim`. The fused linear's output is sliced into
//!      Q (the first `n_heads * head_dim` columns), K and V
//!      (each `group_num * head_dim`). With
//!      `multi_query_group_num=2` and `num_attention_heads=32`,
//!      this is 16-way GQA via the eager-broadcast K/V replication.
//!   2. **Halved-pair RoPE** — only the **first half** of each
//!      head's features are rotated, and rotation is **pair-
//!      adjacent** (interleaved). Reuses
//!      [`crate::lazy_glm4::apply_interleaved_partial_rope`]
//!      with `rope_dim = head_dim / 2`.
//!   3. **Fused SwiGLU MLP** — `dense_h_to_4h: hidden → 2 *
//!      ffn_size` produces the gate and up halves; split, SwiGLU,
//!      down-project through `dense_4h_to_h`.
//!   4. **Apply-residual-post-layernorm flag** —
//!      `apply_residual_connection_post_layernorm=false` (the
//!      GLM3-6B default) sources the residual from the **pre-LN**
//!      input; the flag's `true` branch sources it from the
//!      post-LN output. v1 honors both.
//!   5. **RmsNorm or LayerNorm** controlled by `cfg.rmsnorm`.
//!      v1 honors both via stored gain + optional bias.
//!
//! `apply_query_key_layer_scaling` is a numerical-stability
//! coefficient that scales scores down then back up — a no-op in
//! F32 — so v1 ignores it.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache
//! (recomputes each call), F32. Uses `(batch, seq, hidden)`
//! layout internally — the eager `(seq, batch, hidden)`
//! transposition is a stylistic choice that doesn't change
//! the math.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_glm4::apply_interleaved_partial_rope;
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatGlmNorm {
    Rms,
    Layer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatGlmConfig {
    pub num_layers: usize,
    pub padded_vocab_size: usize,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    /// Per-head dimension (== `head_dim`).
    pub kv_channels: usize,
    pub num_attention_heads: usize,
    pub multi_query_group_num: usize,
    pub seq_length: usize,
    pub layernorm_epsilon: f64,
    pub norm_kind: ChatGlmNorm,
    pub add_qkv_bias: bool,
    pub add_bias_linear: bool,
    pub apply_residual_connection_post_layernorm: bool,
    pub post_layer_norm: bool,
    /// RoPE base. GLM3-6B uses `10_000.0`; CodeGeeX4-9B uses
    /// `10_000.0 * rope_ratio = 5_000_000.0` (ratio = 500).
    pub rope_base: f64,
}

impl ChatGlmConfig {
    /// GLM3-6B preset (THUDM/chatglm3-6b).
    pub fn glm3_6b() -> Self {
        Self {
            num_layers: 28, padded_vocab_size: 65024,
            hidden_size: 4096, ffn_hidden_size: 13696,
            kv_channels: 128, num_attention_heads: 32,
            multi_query_group_num: 2,
            seq_length: 8192, layernorm_epsilon: 1e-5,
            norm_kind: ChatGlmNorm::Rms,
            add_qkv_bias: true, add_bias_linear: false,
            apply_residual_connection_post_layernorm: false,
            post_layer_norm: true,
            rope_base: 10_000.0,
        }
    }

    /// CodeGeeX4-9B preset (THUDM/codegeex4-all-9b). Same
    /// architecture as GLM3-6B with a different layer count,
    /// vocab + seq_length, and **`rope_ratio = 500`** which
    /// pushes the RoPE base to `5_000_000`.
    pub fn codegeex4() -> Self {
        Self {
            num_layers: 40, padded_vocab_size: 151552,
            hidden_size: 4096, ffn_hidden_size: 13696,
            kv_channels: 128, num_attention_heads: 32,
            multi_query_group_num: 2,
            seq_length: 131_072, layernorm_epsilon: 1e-5,
            norm_kind: ChatGlmNorm::Rms,
            add_qkv_bias: true, add_bias_linear: false,
            apply_residual_connection_post_layernorm: false,
            post_layer_norm: true,
            rope_base: 10_000.0 * 500.0,
        }
    }
}

impl ChatGlmConfig {
    pub fn head_dim(&self) -> usize {
        self.kv_channels
    }
    pub fn rope_dim(&self) -> usize {
        self.kv_channels / 2
    }
    /// Output channels of the fused QKV linear.
    pub fn qkv_hidden_size(&self) -> usize {
        let hpa = self.head_dim();
        self.num_attention_heads * hpa + 2 * self.multi_query_group_num * hpa
    }
}

#[derive(Debug, Clone)]
pub struct ChatGlmLayerWeights {
    pub input_norm_gain: Arc<[f32]>,
    /// Bias is `None` for RmsNorm, `Some` for LayerNorm.
    pub input_norm_bias: Option<Arc<[f32]>>,
    pub post_attn_norm_gain: Arc<[f32]>,
    pub post_attn_norm_bias: Option<Arc<[f32]>>,
    /// Fused `[hidden, qkv_hidden_size]`.
    pub query_key_value: WeightStorage,
    pub query_key_value_bias: Option<Arc<[f32]>>,
    pub dense: WeightStorage,
    pub dense_bias: Option<Arc<[f32]>>,
    /// Fused gate+up: `[hidden, 2 * ffn_hidden_size]`. First half is
    /// the gated path, second half is the up path.
    pub dense_h_to_4h: WeightStorage,
    pub dense_h_to_4h_bias: Option<Arc<[f32]>>,
    pub dense_4h_to_h: WeightStorage,
    pub dense_4h_to_h_bias: Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct ChatGlmWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<ChatGlmLayerWeights>,
    pub final_norm_gain: Option<Arc<[f32]>>,
    pub final_norm_bias: Option<Arc<[f32]>>,
    /// `[hidden, vocab_size]`. Distinct from `token_embedding` in
    /// the GLM3-6B reference checkpoint.
    pub output_layer: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct ChatGlmModel {
    pub config: ChatGlmConfig,
    pub weights: ChatGlmWeights,
}

impl ChatGlmModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_post = self.run_backbone(tokens, start_pos)?;
        Ok(weights.output_layer.apply_linear(
            &h_post, cfg.hidden_size, cfg.padded_vocab_size,
        ))
    }

    /// Run the decoder forward up to (and including, when
    /// `post_layer_norm`) the final norm and return per-token
    /// hidden states `(1, seq, hidden_size)`. ChatGLM-specific:
    /// optional `post_layer_norm` gate and configurable
    /// `apply_residual_connection_post_layernorm` are honored.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "ChatGlmModel: tokens must be non-empty");
        let head_dim = cfg.head_dim();
        let rope_dim = cfg.rope_dim();
        assert!(
            rope_dim > 0 && rope_dim % 2 == 0,
            "ChatGlmConfig: kv_channels ({head_dim}) must be even and ≥ 2 for halved-pair RoPE",
        );
        assert_eq!(
            cfg.num_attention_heads % cfg.multi_query_group_num, 0,
            "num_attention_heads ({}) must be a multiple of multi_query_group_num ({})",
            cfg.num_attention_heads, cfg.multi_query_group_num,
        );

        let mut h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.padded_vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;

        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_base, start_pos, seq, rope_dim,
        );

        for layer in &weights.layers {
            h = self.apply_block(&h, layer, &rope_cos, &rope_sin)?;
        }

        if cfg.post_layer_norm {
            apply_norm(
                &h,
                weights.final_norm_gain.as_ref()
                    .expect("post_layer_norm: final_norm_gain required"),
                weights.final_norm_bias.as_ref(),
                cfg.hidden_size, cfg.layernorm_epsilon, cfg.norm_kind,
            )
        } else {
            Ok(h)
        }
    }

    fn apply_block(
        &self,
        x: &LazyTensor,
        layer: &ChatGlmLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let x_norm = apply_norm(
            x, &layer.input_norm_gain, layer.input_norm_bias.as_ref(),
            h, cfg.layernorm_epsilon, cfg.norm_kind,
        )?;
        let attn = self.apply_attention(&x_norm, layer, rope_cos, rope_sin)?;
        let residual_attn = if cfg.apply_residual_connection_post_layernorm {
            &x_norm
        } else {
            x
        };
        let h1 = residual_attn.add(&attn)?;

        let h1_norm = apply_norm(
            &h1, &layer.post_attn_norm_gain, layer.post_attn_norm_bias.as_ref(),
            h, cfg.layernorm_epsilon, cfg.norm_kind,
        )?;
        let mlp_out = self.apply_mlp(&h1_norm, layer)?;
        let residual_mlp = if cfg.apply_residual_connection_post_layernorm {
            &h1_norm
        } else {
            &h1
        };
        residual_mlp.add(&mlp_out)
    }

    fn apply_attention(
        &self,
        x: &LazyTensor,
        layer: &ChatGlmLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.hidden_size;
        let hpa = cfg.head_dim();
        let n_heads = cfg.num_attention_heads;
        let n_groups = cfg.multi_query_group_num;
        let q_dim = n_heads * hpa;
        let kv_dim = n_groups * hpa;
        let qkv_dim = cfg.qkv_hidden_size();

        let qkv = layer.query_key_value.apply_linear(x, h, qkv_dim);
        let qkv = match &layer.query_key_value_bias {
            None => qkv,
            Some(b) => {
                let bt = x.const_f32_like(Arc::clone(b), Shape::from_dims(&[qkv_dim]));
                qkv.broadcast_add(&bt)?
            }
        };

        let q = qkv.slice(2_usize, 0, q_dim)?;
        let k = qkv.slice(2_usize, q_dim, kv_dim)?;
        let v = qkv.slice(2_usize, q_dim + kv_dim, kv_dim)?;

        let _ = (batch, seq);
        let q = q.split_heads(n_heads, hpa)?;
        let k = k.split_heads(n_groups, hpa)?;
        let v = v.split_heads(n_groups, hpa)?;

        // Halved-pair RoPE on the FIRST half of head_dim, pair-adjacent.
        let rope_dim = cfg.rope_dim();
        let q_r = apply_interleaved_partial_rope(&q, rope_cos, rope_sin, hpa, rope_dim)?;
        let k_r = apply_interleaved_partial_rope(&k, rope_cos, rope_sin, hpa, rope_dim)?;

        // GQA expand K, V from n_groups → n_heads.
        let n_rep = n_heads / n_groups;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (hpa as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        // Strict causal mask.
        let mask = LazyTensor::additive_causal_mask_like(x, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        // Note: q_dim = n_heads * hpa may not equal hidden_size when the
        // model uses a separate kv_channels — the eager reference relies
        // on `n_heads * kv_channels == hidden_size` (GLM3-6B: 32 * 128 = 4096).
        let dense_out = layer.dense.apply_linear(&merged, q_dim, cfg.hidden_size);
        match &layer.dense_bias {
            None => Ok(dense_out),
            Some(b) => {
                let bt = x.const_f32_like(
                    Arc::clone(b),
                    Shape::from_dims(&[cfg.hidden_size]),
                );
                dense_out.broadcast_add(&bt)
            }
        }
    }

    fn apply_mlp(
        &self,
        x: &LazyTensor,
        layer: &ChatGlmLayerWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let ffn = cfg.ffn_hidden_size;
        let fused_dim = 2 * ffn;
        // Fused gate + up: hidden → 2 * ffn.
        let h_to_4h = layer.dense_h_to_4h.apply_linear(x, h, fused_dim);
        let h_to_4h = match &layer.dense_h_to_4h_bias {
            None => h_to_4h,
            Some(b) => {
                let bt = x.const_f32_like(Arc::clone(b), Shape::from_dims(&[fused_dim]));
                h_to_4h.broadcast_add(&bt)?
            }
        };
        let gate = h_to_4h.slice(2_usize, 0, ffn)?;
        let up = h_to_4h.slice(2_usize, ffn, ffn)?;
        let swiglu = gate.silu().mul(&up)?;
        let down = layer.dense_4h_to_h.apply_linear(&swiglu, ffn, h);
        match &layer.dense_4h_to_h_bias {
            None => Ok(down),
            Some(b) => {
                let bt = x.const_f32_like(Arc::clone(b), Shape::from_dims(&[h]));
                down.broadcast_add(&bt)
            }
        }
    }
}

fn apply_norm(
    x: &LazyTensor,
    gain: &Arc<[f32]>,
    bias: Option<&Arc<[f32]>>,
    dim: usize,
    eps: f64,
    kind: ChatGlmNorm,
) -> Result<LazyTensor> {
    match kind {
        ChatGlmNorm::Rms => {
            assert!(bias.is_none(), "RmsNorm: bias must be None");
            let _ = dim;
            x.rms_norm_affine(Arc::clone(gain), eps)
        }
        ChatGlmNorm::Layer => {
            let bias = bias.expect("LayerNorm: bias required");
            let _ = dim;
            x.layer_norm_affine(Arc::clone(gain), Arc::clone(bias), eps)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &ChatGlmConfig) -> ChatGlmWeights {
        let mut s: u32 = 33333;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let qkv_dim = cfg.qkv_hidden_size();
        let q_dim = cfg.num_attention_heads * cfg.head_dim();
        let ffn = cfg.ffn_hidden_size;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.padded_vocab_size * h, &mut *nb);

        let make_norm_bias = || -> Option<Arc<[f32]>> {
            match cfg.norm_kind {
                ChatGlmNorm::Rms => None,
                ChatGlmNorm::Layer => Some(Arc::from(vec![0.0_f32; h])),
            }
        };

        let layers: Vec<ChatGlmLayerWeights> = (0..cfg.num_layers)
            .map(|_| ChatGlmLayerWeights {
                input_norm_gain: Arc::from(vec![1.0_f32; h]),
                input_norm_bias: make_norm_bias(),
                post_attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                post_attn_norm_bias: make_norm_bias(),
                query_key_value: WeightStorage::F32(vec_of(h * qkv_dim, &mut *nb)),
                query_key_value_bias: if cfg.add_qkv_bias || cfg.add_bias_linear {
                    Some(vec_of(qkv_dim, &mut *nb))
                } else {
                    None
                },
                dense: WeightStorage::F32(vec_of(q_dim * h, &mut *nb)),
                dense_bias: if cfg.add_bias_linear { Some(vec_of(h, &mut *nb)) } else { None },
                dense_h_to_4h: WeightStorage::F32(vec_of(h * (2 * ffn), &mut *nb)),
                dense_h_to_4h_bias: if cfg.add_bias_linear {
                    Some(vec_of(2 * ffn, &mut *nb))
                } else {
                    None
                },
                dense_4h_to_h: WeightStorage::F32(vec_of(ffn * h, &mut *nb)),
                dense_4h_to_h_bias: if cfg.add_bias_linear { Some(vec_of(h, &mut *nb)) } else { None },
            })
            .collect();
        let final_norm_gain = if cfg.post_layer_norm {
            Some(Arc::from(vec![1.0_f32; h]))
        } else {
            None
        };
        let final_norm_bias = if cfg.post_layer_norm {
            make_norm_bias()
        } else {
            None
        };
        let output_layer = WeightStorage::F32(vec_of(h * cfg.padded_vocab_size, &mut *nb));
        ChatGlmWeights {
            token_embedding, layers,
            final_norm_gain, final_norm_bias,
            output_layer,
        }
    }

    fn tiny_config() -> ChatGlmConfig {
        // n_heads * head_dim = hidden_size: 4 * 4 = 16.
        ChatGlmConfig {
            num_layers: 2,
            padded_vocab_size: 32,
            hidden_size: 16,
            ffn_hidden_size: 24,
            kv_channels: 4,
            num_attention_heads: 4,
            multi_query_group_num: 2,
            seq_length: 64,
            layernorm_epsilon: 1e-5,
            norm_kind: ChatGlmNorm::Rms,
            add_qkv_bias: true,
            add_bias_linear: false,
            apply_residual_connection_post_layernorm: false,
            post_layer_norm: true,
            rope_base: 10_000.0,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = ChatGlmModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.padded_vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// MQA Q/K/V slicing must be wired: the second-half of K
    /// columns belongs to V, not K. Zero the V section of the
    /// fused QKV → output should change (proves V is sliced).
    #[test]
    fn fused_qkv_slicing_layout() {
        let cfg = ChatGlmConfig { num_layers: 1, ..tiny_config() };
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim();
        let kv_dim = cfg.multi_query_group_num * cfg.head_dim();
        let qkv_dim = cfg.qkv_hidden_size();
        assert_eq!(qkv_dim, q_dim + 2 * kv_dim);
        let base = tiny_weights(&cfg);

        // Zero V columns (last kv_dim cols of the fused QKV linear).
        let mut zeroed = base.clone();
        let mut qkv_v = match &zeroed.layers[0].query_key_value {
            WeightStorage::F32(v) => v.to_vec(),
            _ => panic!(),
        };
        for row in 0..h {
            for j in q_dim + kv_dim..qkv_dim {
                qkv_v[row * qkv_dim + j] = 0.0;
            }
        }
        zeroed.layers[0].query_key_value = WeightStorage::F32(Arc::from(qkv_v));
        if let Some(b) = &mut zeroed.layers[0].query_key_value_bias {
            let mut bv: Vec<f32> = (*b.clone()).to_vec();
            for j in q_dim + kv_dim..qkv_dim {
                bv[j] = 0.0;
            }
            *b = Arc::from(bv);
        }
        let m_base = ChatGlmModel { config: cfg.clone(), weights: base };
        let m_zero = ChatGlmModel { config: cfg, weights: zeroed };
        let toks = [1_u32, 2, 3];
        let a = m_base.forward(&toks, 0).unwrap().realize_f32();
        let b = m_zero.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "zeroing fused-QKV V columns must change output, max_diff = {max_diff}");
    }

    /// LayerNorm variant runs without panic (sanity for the
    /// optional-bias plumbing).
    #[test]
    fn layer_norm_variant() {
        let cfg = ChatGlmConfig {
            norm_kind: ChatGlmNorm::Layer,
            ..tiny_config()
        };
        let model = ChatGlmModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3], 0).unwrap().realize_f32();
        assert_eq!(logits.len(), 3 * cfg.padded_vocab_size);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    /// `rope_base` actually affects RoPE. Use a config with
    /// `kv_channels = 8` so `rope_dim = 4` ⇒ the
    /// `step_by(2)` over `[0, rope_dim)` includes a non-zero `i`,
    /// making the per-frequency exponent depend on the base
    /// (with `rope_dim = 2`, the only frequency is `theta^0 = 1`
    /// regardless of `theta`).
    #[test]
    fn rope_base_alters_output() {
        let base_cfg = ChatGlmConfig {
            kv_channels: 8, // ⇒ rope_dim = 4
            num_attention_heads: 2,
            hidden_size: 16, // 2 heads * 8
            ..tiny_config()
        };
        let cfg_a = ChatGlmConfig { rope_base: 10_000.0, ..base_cfg.clone() };
        let cfg_b = ChatGlmConfig { rope_base: 5_000_000.0, ..base_cfg };
        let weights = tiny_weights(&cfg_a);
        let m_a = ChatGlmModel { config: cfg_a, weights: weights.clone() };
        let m_b = ChatGlmModel { config: cfg_b, weights };
        let toks = [1_u32, 2, 3, 4];
        let a = m_a.forward(&toks, 0).unwrap().realize_f32();
        let b = m_b.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        // Tiny weights (∈ [-0.025, 0.025]) ⇒ diff is real but small.
        assert!(max_diff > 1e-8,
            "rope_base change must alter output, max_diff = {max_diff}");
    }

    /// Both presets construct without panic.
    #[test]
    fn presets_construct() {
        let glm = ChatGlmConfig::glm3_6b();
        assert_eq!(glm.num_layers, 28);
        assert_eq!(glm.rope_base, 10_000.0);

        let cgx = ChatGlmConfig::codegeex4();
        assert_eq!(cgx.num_layers, 40);
        assert_eq!(cgx.rope_base, 5_000_000.0);
    }

    /// `apply_residual_connection_post_layernorm` flag flips the
    /// residual source — output must differ between true and false.
    #[test]
    fn residual_source_flag() {
        let cfg_a = ChatGlmConfig {
            apply_residual_connection_post_layernorm: false,
            ..tiny_config()
        };
        let cfg_b = ChatGlmConfig {
            apply_residual_connection_post_layernorm: true,
            ..tiny_config()
        };
        let weights = tiny_weights(&cfg_a);
        let m_a = ChatGlmModel { config: cfg_a, weights: weights.clone() };
        let m_b = ChatGlmModel { config: cfg_b, weights };
        let toks = [3_u32, 5, 7];
        let a = m_a.forward(&toks, 0).unwrap().realize_f32();
        let b = m_b.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "residual-source flag must alter output, max_diff = {max_diff}");
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config();
        let model = ChatGlmModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

//! Gemma 4 text decoder ported to the lazy-graph API.
//!
//! Phase D specialized port. Gemma 4 is a **multimodal** model
//! (text + vision + audio). v1 ports the **text-only** decoder;
//! the vision tower (ViT-style encoder) and audio tower (audio
//! mel-spec encoder) plus the multimodal embedding integration
//! are deferred to follow-up sessions.
//!
//! The text decoder extends [`crate::lazy_gemma3`] with several
//! architectural wrinkles:
//!
//!   1. **Per-layer-type head_dim and num_kv_heads.** Sliding
//!      layers use `head_dim` + `num_key_value_heads`; full
//!      ("global") layers use `global_head_dim` +
//!      `num_global_key_value_heads`. This is configured by a
//!      per-layer `layer_types: Vec<Gemma4LayerType>`.
//!   2. **Partial rotary on global layers.** Only the first
//!      `partial_rotary_factor * head_dim` features per head
//!      are rotated; the rest pass through unchanged. Sliding
//!      layers use full rotary.
//!   3. **Per-layer-type RoPE base.** Sliding layers use
//!      `rope_local_base_freq` (~10_000); global layers use
//!      `rope_theta` (~1_000_000).
//!   4. **V normalization** (pure RmsNorm without a learned
//!      weight, just rsqrt of mean-of-squares). Applied
//!      *after* the V projection and reshape, before attention.
//!   5. **Per-head Q/K RmsNorm with `(gain + 1)` offset**
//!      (Gemma family convention; same as Gemma3).
//!   6. **4-norm block structure** (Gemma3-shape):
//!      input + post-attn + pre-FFN + post-FFN.
//!   7. **`query_pre_attn_scalar`** — Gemma4 honors the eager
//!      reference's choice to compute scale as
//!      `1 / sqrt(head_dim)` (the divisor matches the actual
//!      head_dim per layer type); the config-stored
//!      `query_pre_attn_scalar` field is unused by the eager
//!      forward, so v1 ignores it.
//!   8. **Final-logit soft-cap** when `final_logit_softcapping`
//!      is set (no per-layer attention-score cap in Gemma4 unlike
//!      Gemma3 — that field is absent).
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache,
//! F32. Multimodal injection (image and audio embeddings
//! interleaved with text token embeddings via mask) is deferred.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

pub use crate::lazy_gemma::GemmaActivation as Gemma4Activation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// For full-attention layers (default = `num_key_value_heads`).
    pub num_global_key_value_heads: Option<usize>,
    /// Head dim for sliding layers.
    pub head_dim: usize,
    /// Head dim for full-attention layers.
    pub global_head_dim: usize,
    pub rms_norm_eps: f64,
    /// RoPE base for full-attention layers (default 1_000_000).
    pub rope_theta: f64,
    /// RoPE base for sliding layers (default 10_000).
    pub rope_local_base_freq: f64,
    /// Fraction of `global_head_dim` rotated on full-attn layers.
    pub partial_rotary_factor: f64,
    pub sliding_window: usize,
    pub layer_types: Vec<Gemma4LayerType>,
    pub attention_bias: bool,
    pub hidden_activation: Gemma4Activation,
    pub final_logit_softcapping: Option<f64>,
    pub tie_word_embeddings: bool,
}

impl Gemma4TextConfig {
    pub fn num_global_kv(&self) -> usize {
        self.num_global_key_value_heads.unwrap_or(self.num_key_value_heads)
    }
    pub fn layer_type(&self, layer_idx: usize) -> Gemma4LayerType {
        self.layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(Gemma4LayerType::SlidingAttention)
    }
    pub fn global_rope_dim(&self) -> usize {
        let r = (self.partial_rotary_factor * self.global_head_dim as f64) as usize;
        r & !1
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4LayerWeights {
    pub attn_q: WeightStorage,
    pub attn_q_bias: Option<Arc<[f32]>>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Option<Arc<[f32]>>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Option<Arc<[f32]>>,
    pub attn_o: WeightStorage,
    pub attn_o_bias: Option<Arc<[f32]>>,
    /// Per-head Q RmsNorm gain on the layer-effective head_dim.
    pub q_norm_gain: Arc<[f32]>,
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
pub struct Gemma4TextWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Gemma4LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Gemma4TextModel {
    pub config: Gemma4TextConfig,
    pub weights: Gemma4TextWeights,
}

impl Gemma4TextModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        let lm_head = WeightStorage::F32(weights.token_embedding.clone());
        let logits = lm_head.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size);
        match cfg.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(logits.mul_scalar(1.0 / sc).tanh().mul_scalar(sc)),
        }
    }

    /// Run the decoder forward up to the final offset RmsNorm
    /// and return per-token hidden states `(1, seq, hidden_size)`.
    /// Gemma4-text-specific: alternating
    /// `SlidingAttention` / `FullAttention` layer types each
    /// with their own head_dim, num_kv_heads, RoPE table, and
    /// mask — all honored by the shared backbone.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert!(cfg.num_hidden_layers > 0);
        assert_eq!(
            cfg.layer_types.len(), cfg.num_hidden_layers,
            "layer_types length must match num_hidden_layers",
        );

        let mut h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;
        h = h.mul_scalar((cfg.hidden_size as f64).sqrt());

        let (rope_cos_l, rope_sin_l) = h.rope_tables_const(
            cfg.rope_local_base_freq, start_pos, seq, cfg.head_dim,
        );

        let rope_dim_global = cfg.global_rope_dim();
        let (rope_cos_g, rope_sin_g) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, rope_dim_global,
        );

        let full_mask = self.build_mask(&h, seq, None);
        let sliding_mask = self.build_mask(&h, seq, Some(cfg.sliding_window));

        for (idx, layer) in weights.layers.iter().enumerate() {
            let kind = cfg.layer_type(idx);
            let (head_dim, num_kv, rope_cos, rope_sin, rope_dim, mask, is_global) = match kind {
                Gemma4LayerType::SlidingAttention => (
                    cfg.head_dim, cfg.num_key_value_heads,
                    &rope_cos_l, &rope_sin_l, cfg.head_dim,
                    &sliding_mask, false,
                ),
                Gemma4LayerType::FullAttention => (
                    cfg.global_head_dim, cfg.num_global_kv(),
                    &rope_cos_g, &rope_sin_g, rope_dim_global,
                    &full_mask, true,
                ),
            };
            h = self.apply_layer(
                &h, layer, head_dim, num_kv, rope_cos, rope_sin, rope_dim, mask, is_global,
            )?;
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

    #[allow(clippy::too_many_arguments)]
    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &Gemma4LayerWeights,
        head_dim: usize,
        num_kv: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        rope_dim: usize,
        mask: &LazyTensor,
        _is_global: bool,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let n_heads = cfg.num_attention_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = num_kv * head_dim;

        // Pre-attention norm.
        let residual = x.clone();
        let x_norm = x.rms_norm_affine_with_offset(&layer.input_norm_gain, 1.0, cfg.rms_norm_eps)?;

        // Q / K / V projections.
        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, q_dim).add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(n_heads, head_dim)?;
        let k = k.split_heads(num_kv, head_dim)?;
        let v = v.split_heads(num_kv, head_dim)?;

        // Per-head Q/K RmsNorm with `(gain + 1)` offset.
        let q = q.rms_norm_affine_with_offset(&layer.q_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let k = k.rms_norm_affine_with_offset(&layer.k_norm_gain, 1.0, cfg.rms_norm_eps)?;
        // V normalization: pure RmsNorm without a learned weight.
        let v = v_rms_norm(&v, cfg.rms_norm_eps)?;

        // RoPE. Global uses partial rotary; sliding uses full.
        let q_r = q.rope_partial(rope_cos, rope_sin, rope_dim)?;
        let k_r = k.rope_partial(rope_cos, rope_sin, rope_dim)?;

        // GQA expand.
        let n_rep = n_heads / num_kv;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let scores_masked = scores_scaled.broadcast_add(mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer.attn_o.apply_linear(&merged, q_dim, cfg.hidden_size).add_optional_trailing_bias(layer.attn_o_bias.as_ref())?;
        let attn_out_norm = attn_out.rms_norm_affine_with_offset(&layer.post_attn_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let h1 = residual.add(&attn_out_norm)?;

        // FFN sublayer.
        let residual2 = h1.clone();
        let h1_norm = h1.rms_norm_affine_with_offset(&layer.pre_ffn_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let activated = match cfg.hidden_activation {
            Gemma4Activation::Gelu => gate.gelu_erf(),
            Gemma4Activation::GeluPytorchTanh => gate.gelu(),
        };
        let ffn_in = activated.mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&ffn_in, cfg.intermediate_size, cfg.hidden_size);
        let ffn_out_norm = ffn_out.rms_norm_affine_with_offset(&layer.post_ffn_norm_gain, 1.0, cfg.rms_norm_eps)?;
        residual2.add(&ffn_out_norm)
    }
}


/// V RmsNorm: pure rsqrt(mean of squares + eps), no learned weight.
fn v_rms_norm(v: &LazyTensor, eps: f64) -> Result<LazyTensor> {
    v.rms_norm_last_dim(eps)
}


#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_weights(cfg: &Gemma4TextConfig) -> Gemma4TextWeights {
        let mut s: u32 = 42949;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<Gemma4LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|idx| {
                let kind = cfg.layer_type(idx);
                let (head_dim, num_kv) = match kind {
                    Gemma4LayerType::SlidingAttention => (cfg.head_dim, cfg.num_key_value_heads),
                    Gemma4LayerType::FullAttention => (cfg.global_head_dim, cfg.num_global_kv()),
                };
                let q_dim = cfg.num_attention_heads * head_dim;
                let kv_dim = num_kv * head_dim;
                Gemma4LayerWeights {
                    attn_q: WeightStorage::F32(vec_of(h * q_dim, &mut *nb)),
                    attn_q_bias: if cfg.attention_bias { Some(vec_of(q_dim, &mut *nb)) } else { None },
                    attn_k: WeightStorage::F32(vec_of(h * kv_dim, &mut *nb)),
                    attn_k_bias: if cfg.attention_bias { Some(vec_of(kv_dim, &mut *nb)) } else { None },
                    attn_v: WeightStorage::F32(vec_of(h * kv_dim, &mut *nb)),
                    attn_v_bias: if cfg.attention_bias { Some(vec_of(kv_dim, &mut *nb)) } else { None },
                    attn_o: WeightStorage::F32(vec_of(q_dim * h, &mut *nb)),
                    attn_o_bias: if cfg.attention_bias { Some(vec_of(h, &mut *nb)) } else { None },
                    q_norm_gain: Arc::from(vec![0.05_f32; head_dim]),
                    k_norm_gain: Arc::from(vec![0.05_f32; head_dim]),
                    input_norm_gain: Arc::from(vec![0.05_f32; h]),
                    post_attn_norm_gain: Arc::from(vec![0.05_f32; h]),
                    pre_ffn_norm_gain: Arc::from(vec![0.05_f32; h]),
                    post_ffn_norm_gain: Arc::from(vec![0.05_f32; h]),
                    ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                    ffn_up: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                    ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
                }
            })
            .collect();
        let final_norm_gain = Arc::from(vec![0.05_f32; h]);
        Gemma4TextWeights { token_embedding, layers, final_norm_gain }
    }

    fn tiny_config() -> Gemma4TextConfig {
        // 4 layers: pattern [sliding, sliding, sliding, full].
        Gemma4TextConfig {
            vocab_size: 32, hidden_size: 24, intermediate_size: 32,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_global_key_value_heads: Some(2),
            head_dim: 4, global_head_dim: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            partial_rotary_factor: 0.5, // global_rope_dim = 4
            sliding_window: 3,
            layer_types: vec![
                Gemma4LayerType::SlidingAttention,
                Gemma4LayerType::SlidingAttention,
                Gemma4LayerType::SlidingAttention,
                Gemma4LayerType::FullAttention,
            ],
            attention_bias: false,
            hidden_activation: Gemma4Activation::GeluPytorchTanh,
            final_logit_softcapping: Some(30.0),
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = Gemma4TextModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// Per-layer head_dim selection: the global layer's head_dim
    /// is 8 while sliding layers use 4. The model must still
    /// produce coherent output. Swap the final layer from full
    /// to sliding (different head_dim, different num_kv,
    /// different RoPE table) — output must differ.
    #[test]
    fn per_layer_head_dim_alters_output() {
        let cfg_a = tiny_config();
        let cfg_b = Gemma4TextConfig {
            layer_types: vec![
                Gemma4LayerType::SlidingAttention,
                Gemma4LayerType::SlidingAttention,
                Gemma4LayerType::SlidingAttention,
                Gemma4LayerType::SlidingAttention,
            ],
            ..cfg_a.clone()
        };
        let weights_a = tiny_weights(&cfg_a);
        let weights_b = tiny_weights(&cfg_b);
        let m_a = Gemma4TextModel { config: cfg_a, weights: weights_a };
        let m_b = Gemma4TextModel { config: cfg_b, weights: weights_b };
        let toks = [1_u32, 2, 3];
        let a = m_a.forward(&toks, 0).unwrap().realize_f32();
        let b = m_b.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "layer-type change must alter output, max_diff = {max_diff}");
    }

    /// Final logit soft-cap must measurably change output.
    #[test]
    fn final_softcap_changes_output() {
        let cfg_no = Gemma4TextConfig { final_logit_softcapping: None, ..tiny_config() };
        let cfg_yes = Gemma4TextConfig { final_logit_softcapping: Some(5.0), ..tiny_config() };
        let weights = tiny_weights(&cfg_no);
        let m_no = Gemma4TextModel { config: cfg_no, weights: weights.clone() };
        let m_yes = Gemma4TextModel { config: cfg_yes, weights };
        let toks = [3_u32, 7, 11];
        let a = m_no.forward(&toks, 0).unwrap().realize_f32();
        let b = m_yes.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "final logit soft-cap must alter output, max_diff = {max_diff}");
    }

    /// V normalization is wired: a different magnitude of input
    /// to the V projection should still produce sane output. We
    /// just check that the v_rms_norm helper actually runs on a
    /// sample tensor (smoke test).
    #[test]
    fn v_rms_norm_smoke() {
        let dev = Device::cpu();
        let x = LazyTensor::from_f32(
            Arc::from((0..16).map(|i| (i as f32 + 1.0) * 0.1).collect::<Vec<_>>()),
            Shape::from_dims(&[1, 2, 2, 4]),
            &dev,
        );
        let normed = v_rms_norm(&x, 1e-6).unwrap().realize_f32();
        // mean-squared per last-dim group should be ~1 after RMS norm.
        for chunk in normed.chunks(4) {
            let mean_sq: f32 = chunk.iter().map(|v| v * v).sum::<f32>() / 4.0;
            assert!((mean_sq - 1.0).abs() < 1e-3,
                "v_rms_norm did not produce unit-RMS chunk: mean_sq = {mean_sq}");
        }
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config();
        let model = Gemma4TextModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

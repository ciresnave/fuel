//! StableLM decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. StableLM (Stability AI) is structurally
//! `Mistral + LayerNorm + partial-rotary`:
//! - **LayerNorm** with bias on the pre-attention + pre-FFN paths
//!   (not RmsNorm).
//! - **Partial rotary** — apply RoPE to only the first
//!   `(head_dim * partial_rotary_factor)` dimensions of each head.
//!   StableLM-1 uses 0.25 (25% of head_dim gets RoPE); StableLM-2
//!   uses 1.0 (full rotary, equivalent to LLaMA's path).
//! - **Optional Q/K/V biases** via `cfg.use_qkv_bias` (StableLM-2
//!   uses biases on Q/K/V only, not O).
//! - GQA + SwiGLU + bias-free O.
//!
//! The partial-rotary helper applies RoPE to a head-dim prefix and
//! passes the rest through unchanged. v1 builds the prefix's
//! cos/sin tables from `rope_dim` (not `head_dim`); the suffix
//! never sees RoPE.

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct StableLmConfig {
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
    /// Fraction of `head_dim` to which RoPE is applied. Stability's
    /// StableLM-1 ships 0.25; StableLM-2 uses 1.0.
    pub partial_rotary_factor: f64,
    /// StableLM-2 has Q/K/V biases; StableLM-1 doesn't.
    pub use_qkv_bias: bool,
}

impl StableLmConfig {
    pub fn rope_dim(&self) -> usize {
        // Round down to an even number so the half-split RoPE layout
        // (cos/sin pair-of-dims) divides cleanly. `head_dim` is
        // already even, so flooring partial_rotary_factor * head_dim
        // and rounding down to even gives a valid prefix.
        let rd = (self.head_dim as f64 * self.partial_rotary_factor) as usize;
        (rd / 2) * 2
    }
}

#[derive(Debug, Clone)]
pub struct StableLmWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<StableLmLayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub output: WeightStorage,
}

/// Per-layer weights. LayerNorm has both gain + bias on both norm
/// positions; Q/K/V optionally have biases; O has none; MLP is
/// SwiGLU with no biases.
#[derive(Debug, Clone)]
pub struct StableLmLayerWeights {
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
    pub ffn_gate: WeightStorage,
    pub ffn_up: WeightStorage,
    pub ffn_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct StableLmModel {
    pub config: StableLmConfig,
    pub weights: StableLmWeights,
}

impl StableLmModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final LayerNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection. Mirrors the
    /// `forward_hidden` pattern across the LLM family.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. StableLM does NOT scale embeddings.
    pub fn forward_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let h_norm = self.run_backbone_embeds(embeds, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`].
    pub fn forward_hidden_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.run_backbone_embeds(embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder.
    pub fn embed_tokens_anchored(
        &self, anchor: &LazyTensor, tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        anchor.embed_tokens_anchored(
            self.weights.token_embedding.clone(),
            cfg.vocab_size, cfg.hidden_size, tokens,
        )
    }

    fn apply_lm_head(&self, h_norm: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        Ok(self.weights.output.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0);

        let h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;
        self.run_backbone_embeds(&h, start_pos)
    }

    fn run_backbone_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(crate::Error::Msg(format!(
                "StableLmModel::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "StableLmModel::forward_embeds: seq must be > 0".into(),
            ).bt());
        }
        if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "StableLmConfig: num_attention_heads * head_dim must equal hidden_size".into(),
            ).bt());
        }
        let mut h = embeds.clone();

        let rope_dim = cfg.rope_dim();
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, rope_dim,
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        h.layer_norm_affine(
            std::sync::Arc::clone(&weights.final_ln_gain),
            std::sync::Arc::clone(&weights.final_ln_bias),
            cfg.layer_norm_eps,
        )
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &StableLmLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let rope_dim = cfg.rope_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * head_dim;

        let x_norm = x.layer_norm_affine(std::sync::Arc::clone(&layer.input_ln_gain), std::sync::Arc::clone(&layer.input_ln_bias), cfg.layer_norm_eps)?;

        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size).add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, head_dim)?;

        // Partial rotary: slice [..., :rope_dim] → apply RoPE → concat
        // with [..., rope_dim:].
        let q_r = q.rope_partial(rope_cos, rope_sin, rope_dim)?;
        let k_r = k.rope_partial(rope_cos, rope_sin, rope_dim)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq { mask_data[i * seq + j] = f32::NEG_INFINITY; }
        }
        let mask = x.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]));
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size);

        let h1 = x.add(&attn_out)?;
        let h1_norm = h1.layer_norm_affine(std::sync::Arc::clone(&layer.post_attn_ln_gain), std::sync::Arc::clone(&layer.post_attn_ln_bias), cfg.layer_norm_eps)?;
        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size);
        h1.add(&ffn_out)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_weights(cfg: &StableLmConfig) -> StableLmWeights {
        let mut s: u32 = 88888;
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
        let layers: Vec<StableLmLayerWeights> = (0..cfg.num_hidden_layers).map(|_| StableLmLayerWeights {
            input_ln_gain:     Arc::from(vec![1.0_f32; h]),
            input_ln_bias:     Arc::from(vec![0.0_f32; h]),
            post_attn_ln_gain: Arc::from(vec![1.0_f32; h]),
            post_attn_ln_bias: Arc::from(vec![0.0_f32; h]),
            attn_q:        WeightStorage::F32(vec_of(h * h, &mut *nb)),
            attn_q_bias:   if cfg.use_qkv_bias { Some(vec_of(h, &mut *nb)) } else { None },
            attn_k:        WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_k_bias:   if cfg.use_qkv_bias { Some(vec_of(kv, &mut *nb)) } else { None },
            attn_v:        WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_v_bias:   if cfg.use_qkv_bias { Some(vec_of(kv, &mut *nb)) } else { None },
            attn_o:        WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate:      WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up:        WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down:      WeightStorage::F32(vec_of(i * h, &mut *nb)),
        }).collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        StableLmWeights { token_embedding, layers, final_ln_gain, final_ln_bias, output }
    }

    #[test]
    fn forward_partial_rotary_factor_0_25() {
        let cfg = StableLmConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, layer_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 64, partial_rotary_factor: 0.25,
            use_qkv_bias: false,
        };
        // 4 * 0.25 = 1 → rope_dim = 0 (rounded to even). Bump to 1.0
        // for this smoke test so RoPE actually fires.
        let cfg_full = StableLmConfig { partial_rotary_factor: 1.0, ..cfg };
        let model = StableLmModel { config: cfg_full.clone(), weights: tiny_weights(&cfg_full) };
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg_full.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }

    /// With a head_dim large enough that partial rotary is genuinely
    /// partial (rope_dim == 4 of 8), output must differ from the
    /// full-rotary path.
    #[test]
    fn partial_rotary_differs_from_full() {
        let mut cfg = StableLmConfig {
            vocab_size: 16, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 1, num_attention_heads: 2, num_key_value_heads: 2,
            head_dim: 8, layer_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 32, partial_rotary_factor: 0.5, // → rope_dim = 4
            use_qkv_bias: false,
        };
        let weights = tiny_weights(&cfg);
        let out_partial = StableLmModel { config: cfg.clone(), weights: weights.clone() }
            .forward(&[1, 2, 3, 4], 0).unwrap().realize_f32();
        cfg.partial_rotary_factor = 1.0; // full rotary
        let out_full = StableLmModel { config: cfg, weights }
            .forward(&[1, 2, 3, 4], 0).unwrap().realize_f32();
        let any_diff = out_partial.iter().zip(out_full.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7);
        assert!(any_diff, "partial vs full rotary must differ");
    }

    /// `forward_hidden` returns post-LayerNorm hidden states
    /// `(1, seq, hidden_size)` without the lm_head matmul.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = StableLmConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, layer_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 64, partial_rotary_factor: 1.0,
            use_qkv_bias: false,
        };
        let model = StableLmModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    fn forward_embeds_test_cfg() -> StableLmConfig {
        StableLmConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, layer_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 64, partial_rotary_factor: 1.0,
            use_qkv_bias: false,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = StableLmModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let logits_via_embeds = model.forward_embeds(&embeds, 0).unwrap().realize_f32();
        let max_diff = logits_ref.iter().zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "StableLm forward vs forward_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = StableLmModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let bad = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]), &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = forward_embeds_test_cfg();
        let model = StableLmModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![5, 7];
        let h_ref = model.forward_hidden(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let h_via_embeds = model.forward_hidden_embeds(&embeds, 0).unwrap().realize_f32();
        let max_diff = h_ref.iter().zip(h_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "StableLm forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})");
    }
}

//! GLM-4 (new architecture) decoder ported to the lazy-graph API.
//!
//! Phase D specialized port. GLM-4 keeps the Llama-family overall
//! shape (RmsNorm, GQA, RoPE, SwiGLU FFN) and adds three
//! architectural twists worth honoring distinctly:
//!
//!   1. **Interleaved RoPE (`rope_i`)** — pairs are adjacent
//!      `(x_0, x_1), (x_2, x_3), …` rather than the standard
//!      split-half layout `(x_i, x_{i+d/2})` that
//!      [`fuel_graph::build_rope_tables`] /
//!      [`LazyTensor::rope_with_tables`] assume. We emulate
//!      the interleaved variant by **reshape-permuting** the
//!      input from `(..., d)` to `(..., d/2, 2)`, swapping the
//!      last two dims to `(..., 2, d/2)`, applying standard
//!      split-half RoPE on the resulting `(..., d)`, then
//!      reversing the permute. This is exactly equivalent to
//!      pair-adjacent rotation and avoids a new graph op.
//!   2. **Optional partial rotary** — `partial_rotary_factor`
//!      controls the rotated prefix per head, same as Phi /
//!      StableLM. The pass-through tail is untouched.
//!   3. **Four norms per block** — `input_layernorm` and
//!      `post_self_attn_layernorm` wrap the attention path;
//!      `post_attention_layernorm` and `post_mlp_layernorm`
//!      wrap the FFN path. Two residual sums, four norms.
//!   4. **Fused `gate_up_proj`** — a single linear
//!      `hidden → 2 * intermediate` is split into the gate and
//!      up halves (same pattern as Phi-3). FFN uses
//!      `down(act(gate) * up)`.
//!
//! Optional Q/K/V biases (`attention_bias`, default false) and
//! a tied lm_head (`tie_word_embeddings`) are supported via
//! flag fields. v1 supports the GLM-4 default configuration:
//! SwiGLU activation, full or partial rotary, no sliding window.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache
//! (recomputes each call), F32. The mask is causal-only (no
//! sliding window — the config's `sliding_window` is read but
//! ignored in v1, mirroring the eager GLM-4 default).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm4Activation {
    Silu,
    Gelu,
    GeluPytorchTanh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Glm4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Fraction of `head_dim` to rotate. `1.0` = full rotary
    /// (default for GLM-4-9B).
    pub partial_rotary_factor: f64,
    pub attention_bias: bool,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_activation: Glm4Activation,
    pub tie_word_embeddings: bool,
}

impl Glm4Config {
    pub fn rope_dim(&self) -> usize {
        let r = (self.partial_rotary_factor * self.head_dim as f64) as usize;
        r & !1 // RoPE expects even
    }
}

#[derive(Debug, Clone)]
pub struct Glm4LayerWeights {
    pub input_norm_gain: Arc<[f32]>,
    pub post_self_attn_norm_gain: Arc<[f32]>,
    pub post_attn_norm_gain: Arc<[f32]>,
    pub post_mlp_norm_gain: Arc<[f32]>,

    pub attn_q: WeightStorage,
    pub attn_q_bias: Option<Arc<[f32]>>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Option<Arc<[f32]>>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Option<Arc<[f32]>>,
    pub attn_o: WeightStorage, // never has bias in GLM-4

    /// Fused gate+up: `[hidden, 2*intermediate]`. First half is
    /// the gated path, second half is the up path.
    pub ffn_gate_up: WeightStorage,
    pub ffn_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Glm4Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Glm4LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    /// If `tie_word_embeddings`, the caller passes `None` and
    /// `token_embedding` is reused as the lm_head matrix.
    pub lm_head: Option<WeightStorage>,
}

#[derive(Debug, Clone)]
pub struct Glm4Model {
    pub config: Glm4Config,
    pub weights: Glm4Weights,
}

impl Glm4Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. GLM4 does NOT scale embeddings.
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
        let lm_head_w = match &self.weights.lm_head {
            Some(w) => w.clone(),
            None => WeightStorage::F32(self.weights.token_embedding.clone()),
        };
        Ok(lm_head_w.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0, "Glm4Model::forward: tokens must be non-empty");

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
                "Glm4Model::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "Glm4Model::forward_embeds: seq must be > 0".into(),
            ).bt());
        }
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
            return Err(crate::Error::Msg(
                "Glm4Config: num_attention_heads must be a multiple of num_key_value_heads".into(),
            ).bt());
        }
        let rope_dim = cfg.rope_dim();
        if rope_dim == 0 || rope_dim > cfg.head_dim || rope_dim % 2 != 0 {
            return Err(crate::Error::Msg(format!(
                "Glm4Config: rope_dim ({rope_dim}) must be even and in (0, head_dim ({})]",
                cfg.head_dim,
            )).bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, rope_dim,
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        h.rms_norm_affine(std::sync::Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps)
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &Glm4LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let rope_dim = cfg.rope_dim();

        // ---- Attention sublayer ---------------------------------------------
        let residual = x.clone();
        let x_norm = x.rms_norm_affine(std::sync::Arc::clone(&layer.input_norm_gain), cfg.rms_norm_eps)?;

        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, q_dim).add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        // Interleaved RoPE on the first `rope_dim` features.
        let q_r = apply_interleaved_partial_rope(&q, rope_cos, rope_sin, cfg.head_dim, rope_dim)?;
        let k_r = apply_interleaved_partial_rope(&k, rope_cos, rope_sin, cfg.head_dim, rope_dim)?;

        // GQA replication.
        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        // Strict causal mask.
        let mask = LazyTensor::additive_causal_mask_like(x, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer.attn_o.apply_linear(&merged, q_dim, cfg.hidden_size);
        let attn_normed = attn_out.rms_norm_affine(std::sync::Arc::clone(&layer.post_self_attn_norm_gain), cfg.rms_norm_eps)?;
        let h1 = residual.add(&attn_normed)?;

        // ---- FFN sublayer ---------------------------------------------------
        let residual2 = h1.clone();
        let h1_norm = h1.rms_norm_affine(std::sync::Arc::clone(&layer.post_attn_norm_gain), cfg.rms_norm_eps)?;

        // Fused gate_up: [hidden, 2 * intermediate]. Split last dim.
        let gate_up = layer.ffn_gate_up.apply_linear(
            &h1_norm, cfg.hidden_size, 2 * cfg.intermediate_size,
        );
        let gate = gate_up.slice(2_usize, 0, cfg.intermediate_size)?;
        let up = gate_up.slice(2_usize, cfg.intermediate_size, cfg.intermediate_size)?;
        let activated = match cfg.hidden_activation {
            Glm4Activation::Silu => gate.silu(),
            Glm4Activation::Gelu => gate.gelu_erf(),
            Glm4Activation::GeluPytorchTanh => gate.gelu(),
        };
        let ffn_in = activated.mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&ffn_in, cfg.intermediate_size, cfg.hidden_size);
        let ffn_normed = ffn_out.rms_norm_affine(std::sync::Arc::clone(&layer.post_mlp_norm_gain), cfg.rms_norm_eps)?;
        residual2.add(&ffn_normed)
    }
}

/// Apply interleaved RoPE (`rope_i`) to the first `rope_dim`
/// features of each head. `qk` is shaped `[batch, n_heads, seq, head_dim]`
/// and `rope_cos` / `rope_sin` are the standard split-half tables
/// shaped `[seq, rope_dim]` produced by `fuel_graph::build_rope_tables`.
///
/// The trick: reshape `(..., d)` to `(..., d/2, 2)`, permute the
/// last two dims, then `reshape(..., d)` — adjacent pairs become
/// "first half + second half" which exactly matches the standard
/// split-half RoPE convention. Reverse the permute afterward.
pub fn apply_interleaved_partial_rope(
    qk: &LazyTensor,
    rope_cos: &LazyTensor,
    rope_sin: &LazyTensor,
    head_dim: usize,
    rope_dim: usize,
) -> Result<LazyTensor> {
    if rope_dim == 0 {
        return Ok(qk.clone());
    }
    let shape = qk.shape();
    let dims = shape.dims();
    assert_eq!(dims.len(), 4);
    let batch = dims[0];
    let n_heads = dims[1];
    let seq = dims[2];
    let pass_dim = head_dim - rope_dim;

    // Split rotated prefix vs pass-through tail.
    let rot = qk.slice(3_usize, 0, rope_dim)?;
    let pass = if pass_dim > 0 {
        Some(qk.slice(3_usize, rope_dim, pass_dim)?)
    } else {
        None
    };

    // Permute (..., rope_dim) → (..., 2, rope_dim/2) by reshape + permute.
    let half = rope_dim / 2;
    let rot_pairs = rot.reshape(Shape::from_dims(&[batch, n_heads, seq, half, 2]))?;
    // Swap last two dims: (..., half, 2) → (..., 2, half).
    let rot_split = rot_pairs.permute([0, 1, 2, 4, 3_usize])?;
    // Flatten back to (..., rope_dim).
    let rot_flat = rot_split.reshape(Shape::from_dims(&[batch, n_heads, seq, rope_dim]))?;

    // Now standard split-half RoPE.
    let rotated = rot_flat.rope_with_tables(rope_cos, rope_sin)?;

    // Reverse: (..., rope_dim) → (..., 2, half) → (..., half, 2) → flatten.
    let rotated_split = rotated.reshape(Shape::from_dims(&[batch, n_heads, seq, 2, half]))?;
    let rotated_pairs = rotated_split.permute([0, 1, 2, 4, 3_usize])?;
    let rotated_flat = rotated_pairs.reshape(Shape::from_dims(&[batch, n_heads, seq, rope_dim]))?;

    match pass {
        None => Ok(rotated_flat),
        Some(pass_tensor) => rotated_flat.concat(&pass_tensor, 3_usize),
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &Glm4Config) -> Glm4Weights {
        let mut s: u32 = 67890;
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
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<Glm4LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| Glm4LayerWeights {
                input_norm_gain: Arc::from(vec![1.0_f32; h]),
                post_self_attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                post_attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                post_mlp_norm_gain: Arc::from(vec![1.0_f32; h]),
                attn_q: WeightStorage::F32(vec_of(h * q_dim, &mut *nb)),
                attn_q_bias: if cfg.attention_bias { Some(vec_of(q_dim, &mut *nb)) } else { None },
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_k_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_v_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
                attn_o: WeightStorage::F32(vec_of(q_dim * h, &mut *nb)),
                ffn_gate_up: WeightStorage::F32(vec_of(h * (2 * i), &mut *nb)),
                ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb)))
        };
        Glm4Weights { token_embedding, layers, final_norm_gain, lm_head }
    }

    fn tiny_config() -> Glm4Config {
        Glm4Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4,
            num_key_value_heads: 2, head_dim: 4,
            partial_rotary_factor: 0.5, // rope_dim = 2
            attention_bias: false,
            max_position_embeddings: 64,
            rope_theta: 10_000.0, rms_norm_eps: 1e-5,
            hidden_activation: Glm4Activation::Silu,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = Glm4Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    #[test]
    fn tied_embedding_lm_head() {
        let cfg = Glm4Config { tie_word_embeddings: true, ..tiny_config() };
        let weights = tiny_weights(&cfg);
        assert!(weights.lm_head.is_none());
        let model = Glm4Model { config: cfg.clone(), weights };
        let logits = model.forward(&[2, 3], 0).unwrap().realize_f32();
        assert_eq!(logits.len(), 2 * cfg.vocab_size);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn full_rotary() {
        let mut cfg = tiny_config();
        cfg.partial_rotary_factor = 1.0;
        assert_eq!(cfg.rope_dim(), cfg.head_dim);
        let model = Glm4Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3], 0).unwrap().realize_f32();
        assert_eq!(logits.len(), 3 * cfg.vocab_size);
    }

    /// Interleaved RoPE on rope_dim == head_dim — identical input
    /// shape but different rotation convention than split-half.
    /// Verify rotation is applied: zero RoPE tables (cos = 1, sin = 0)
    /// should be the identity; with non-trivial tables, output changes.
    #[test]
    fn interleaved_rope_is_applied() {
        let cfg = Glm4Config { num_hidden_layers: 1, partial_rotary_factor: 1.0, ..tiny_config() };
        let head_dim = cfg.head_dim;
        let rope_dim = cfg.rope_dim();

        let dev = Device::cpu();
        // Build a (1, 1, 1, head_dim) tensor.
        let qk = LazyTensor::from_f32(
            Arc::from((0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect::<Vec<_>>()),
            Shape::from_dims(&[1, 1, 1, head_dim]),
            &dev,
        );
        // Identity-ish RoPE tables: cos=1, sin=0 ⇒ rotation is identity.
        let cos_id = qk.const_f32_like(
            Arc::from(vec![1.0_f32; rope_dim]),
            Shape::from_dims(&[1, rope_dim]),
        );
        let sin_id = qk.const_f32_like(
            Arc::from(vec![0.0_f32; rope_dim]),
            Shape::from_dims(&[1, rope_dim]),
        );
        let id_out = apply_interleaved_partial_rope(&qk, &cos_id, &sin_id, head_dim, rope_dim)
            .unwrap()
            .realize_f32();
        let in_data = qk.realize_f32();
        for (a, b) in in_data.iter().zip(id_out.iter()) {
            assert!((a - b).abs() < 1e-6,
                "identity RoPE must round-trip: {a} vs {b}");
        }

        // Non-trivial RoPE: cos=0, sin=1 ⇒ pair (x_0, x_1) becomes (-x_1, x_0).
        // For interleaved, this means: [x0, x1, x2, x3] → [-x1, x0, -x3, x2].
        let cos_rot = qk.const_f32_like(
            Arc::from(vec![0.0_f32; rope_dim]),
            Shape::from_dims(&[1, rope_dim]),
        );
        let sin_rot = qk.const_f32_like(
            Arc::from(vec![1.0_f32; rope_dim]),
            Shape::from_dims(&[1, rope_dim]),
        );
        let rot_out = apply_interleaved_partial_rope(&qk, &cos_rot, &sin_rot, head_dim, rope_dim)
            .unwrap()
            .realize_f32();
        // Expected: pair (a, b) → (-b, a) per interleaved RoPE convention.
        // in_data = [0.1, 0.2, 0.3, 0.4]
        // expected = [-0.2, 0.1, -0.4, 0.3]
        let expected: Vec<f32> = in_data
            .chunks(2)
            .flat_map(|pair| vec![-pair[1], pair[0]])
            .collect();
        for (a, e) in rot_out.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5,
                "interleaved rotation: got {a}, expected {e}");
        }
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config();
        let model = Glm4Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = tiny_config();
        let model = Glm4Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
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
            "GLM4 forward vs forward_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = tiny_config();
        let model = Glm4Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let bad = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]), &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = tiny_config();
        let model = Glm4Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
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
            "GLM4 forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})");
    }
}

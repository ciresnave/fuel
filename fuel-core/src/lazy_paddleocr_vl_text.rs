//! PaddleOCR-VL text language model ported to the lazy-graph API.
//!
//! Sub-port 1 of three for PaddleOCR-VL. The text stack is an
//! ERNIE-4.5-0.3B style decoder: GQA, RmsNorm pre-norm, SwiGLU FFN,
//! bias-free Q/K/V/O projections, and **Multimodal Rotary Position
//! Embedding (M-RoPE)**. M-RoPE is the only structural deviation from
//! a stock LLaMA / Mistral decoder — everything else lines up with
//! `crate::lazy::LayerWeights`.
//!
//! # M-RoPE in one paragraph
//!
//! Standard 1D RoPE pairs each position with a scalar `pos`. M-RoPE
//! pairs each token with a 3D position `(t, h, w)` (temporal /
//! height / width) so vision tokens can carry their 2D grid location.
//! Within a single attention head the `head_dim` is split into
//! `mrope_section * 2 = [16, 24, 24, 16, 24, 24]` (Python list-
//! repetition) chunks; chunk `i` uses dimension `i % 3` of the 3D
//! position. For text tokens all three positions coincide so M-RoPE
//! reduces to standard 1D RoPE — that property powers the
//! `forward_embeds_matches_forward` test below.
//!
//! The cos/sin tables for M-RoPE are built host-side once per forward
//! and fed into the existing fused `rope_with_tables` op. That keeps
//! the on-graph attention dispatch identical to llama / mistral; the
//! only M-RoPE-specific work is the table-building helper
//! `build_mrope_tables` and the 3D position-id surface.
//!
//! # Scope
//!
//! - Forward-only, batch = 1, F32 activations.
//! - No KV cache.
//! - Weight names mirror the eager safetensors layout for downstream
//!   loader interop:
//!     - `model.embed_tokens.weight` → [`PaddleOcrVlTextWeights::token_embedding`]
//!     - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` → `attn_{q,k,v,o}`
//!     - `model.layers.{i}.mlp.{gate,up,down}_proj.weight` → `ffn_{gate,up,down}`
//!     - `model.layers.{i}.input_layernorm.weight` → `attn_norm_gain`
//!     - `model.layers.{i}.post_attention_layernorm.weight` → `ffn_norm_gain`
//!     - `model.norm.weight` → `final_norm_gain`
//!     - `lm_head.weight` → `output` (or tied to embedding if
//!       `tie_word_embeddings`).

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

/// PaddleOCR-VL text-model configuration. Mirrors the HuggingFace
/// `config.json` shape so a downloaded config can deserialize
/// directly when the safetensors loader is wired. `mrope_section`
/// must sum to `head_dim / 2` (PaddleOCR-VL ships
/// `[16, 24, 24]` for `head_dim = 128`).
#[derive(Debug, Clone, PartialEq)]
pub struct PaddleOcrVlTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub use_bias: bool,
    pub tie_word_embeddings: bool,
    /// Multimodal RoPE section sizes `[temporal, height, width]`.
    /// Repeated twice to fill `head_dim` (see module docs).
    pub mrope_section: Vec<usize>,
}

impl PaddleOcrVlTextConfig {
    /// Stock PaddleOCR-VL text-stack preset. Matches the HuggingFace
    /// defaults in the eager `paddleocr_vl::config::Config::default`.
    pub fn paddleocr_vl_default() -> Self {
        Self {
            vocab_size: 103_424,
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 18,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            max_position_embeddings: 131_072,
            use_bias: false,
            tie_word_embeddings: false,
            mrope_section: vec![16, 24, 24],
        }
    }
}

/// Weight bundle for [`PaddleOcrVlTextModel`]. Per-layer fields reuse
/// the shared [`LayerWeights`] shape (identical to LLaMA / Mistral
/// bias-free GQA).
#[derive(Debug, Clone)]
pub struct PaddleOcrVlTextWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    /// `[hidden_size, vocab_size]` lm_head. When
    /// `tie_word_embeddings` is true the caller should leave this
    /// `None`; the forward path then wraps `token_embedding` as the
    /// lm_head projection.
    pub output: Option<WeightStorage>,
}

#[derive(Debug, Clone)]
pub struct PaddleOcrVlTextModel {
    pub config: PaddleOcrVlTextConfig,
    pub weights: PaddleOcrVlTextWeights,
}

impl PaddleOcrVlTextModel {
    /// Standard token-in / logits-out entry point. Uses 1D positions
    /// `start_pos..start_pos+seq` for all three M-RoPE axes — i.e.
    /// pure text input. Returns logits `(1, seq, vocab_size)`.
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = LazyTensor::embed_tokens(
            self.weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
            &Device::cpu(),
        )?;
        self.forward_embeds(&h, start_pos)
    }

    /// Forward from pre-computed embeddings `(1, seq, hidden_size)`.
    /// Used by the PaddleOCR-VL composition layer (sub-port 3) after
    /// it interleaves vision-tile embeddings into the text stream.
    /// Uses standard 1D positions; multimodal callers needing true
    /// 3D M-RoPE should use [`Self::forward_embeds_with_mrope`].
    pub fn forward_embeds(&self, embeds: &LazyTensor, start_pos: usize) -> Result<LazyTensor> {
        let h_norm = self.forward_hidden_embeds(embeds, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Like [`Self::forward_embeds`] but skips the lm_head projection
    /// and returns post-final-RmsNorm hidden states. Used by
    /// PaddleOCR-VL composition tests that need to inspect text-stack
    /// hidden states before the logit projection.
    pub fn forward_hidden_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let seq = self.validate_embeds(embeds)?;
        let position_ids = host_text_position_ids(seq, start_pos);
        self.run_backbone(embeds, &position_ids)?
            .rms_norm_affine(Arc::clone(&self.weights.final_norm_gain), cfg.rms_norm_eps)
    }

    /// M-RoPE entry point. `position_ids` is a host-side
    /// `[seq][3]` table: index `[s][0]` is temporal, `[s][1]` is
    /// height, `[s][2]` is width for token `s` (batch = 1).
    /// Returns logits `(1, seq, vocab_size)`.
    ///
    /// Vision-token positions are filled by the host-side
    /// `compute_mrope_position_ids` family in the eager port; for
    /// the lazy text-only path we expose the table directly so the
    /// composition layer can build it once and feed it in.
    pub fn forward_embeds_with_mrope(
        &self, embeds: &LazyTensor, position_ids: &[[i64; 3]],
    ) -> Result<LazyTensor> {
        let h_norm = self.forward_hidden_embeds_with_mrope(embeds, position_ids)?;
        self.apply_lm_head(&h_norm)
    }

    /// Like [`Self::forward_embeds_with_mrope`] but skips the
    /// lm_head and returns post-RmsNorm hidden states.
    pub fn forward_hidden_embeds_with_mrope(
        &self, embeds: &LazyTensor, position_ids: &[[i64; 3]],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let seq = self.validate_embeds(embeds)?;
        if position_ids.len() != seq {
            return Err(crate::Error::Msg(format!(
                "PaddleOcrVlTextModel: position_ids length ({}) must match seq ({})",
                position_ids.len(), seq,
            )).bt());
        }
        self.run_backbone(embeds, position_ids)?
            .rms_norm_affine(Arc::clone(&self.weights.final_norm_gain), cfg.rms_norm_eps)
    }

    fn validate_embeds(&self, embeds: &LazyTensor) -> Result<usize> {
        let cfg = &self.config;
        let dims = embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(crate::Error::Msg(format!(
                "PaddleOcrVlTextModel: embeds shape must be (1, seq, {}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        if dims[1] == 0 {
            return Err(crate::Error::Msg(
                "PaddleOcrVlTextModel: seq must be > 0".into(),
            ).bt());
        }
        if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "PaddleOcrVlTextConfig: num_attention_heads * head_dim must equal hidden_size".into(),
            ).bt());
        }
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
            return Err(crate::Error::Msg(format!(
                "PaddleOcrVlTextConfig: num_attention_heads ({}) must be a multiple of num_key_value_heads ({})",
                cfg.num_attention_heads, cfg.num_key_value_heads,
            )).bt());
        }
        let mrope_sum: usize = cfg.mrope_section.iter().sum();
        if mrope_sum * 2 != cfg.head_dim {
            return Err(crate::Error::Msg(format!(
                "PaddleOcrVlTextConfig: mrope_section {:?} sums to {}, expected head_dim/2 = {}",
                cfg.mrope_section, mrope_sum, cfg.head_dim / 2,
            )).bt());
        }
        Ok(dims[1])
    }

    fn run_backbone(
        &self, embeds: &LazyTensor, position_ids: &[[i64; 3]],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = position_ids.len();

        let (rope_cos_data, rope_sin_data) =
            build_mrope_tables(cfg.rope_theta, cfg.head_dim, &cfg.mrope_section, position_ids);
        let rope_shape = Shape::from_dims(&[seq, cfg.head_dim]);
        let rope_cos = embeds.const_f32_like(rope_cos_data, rope_shape.clone());
        let rope_sin = embeds.const_f32_like(rope_sin_data, rope_shape);
        let mask = build_causal_mask(embeds, seq);

        let mut h = embeds.clone();
        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, &mask)?;
        }
        Ok(h)
    }

    fn apply_lm_head(&self, h_norm: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        match &self.weights.output {
            Some(w) => Ok(w.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size)),
            None => {
                // Tied lm_head: project against the transposed
                // embedding table. Materialize the table as a
                // `WeightStorage::F32` then `apply_linear`.
                let tied = WeightStorage::F32(Arc::clone(&self.weights.token_embedding));
                Ok(tied.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size))
            }
        }
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let x_norm = x.rms_norm_affine(Arc::clone(&layer.attn_norm_gain), cfg.rms_norm_eps)?;

        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size)
            .add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim)
            .add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim)
            .add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let scores_masked = scores_scaled.broadcast_add(mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size);

        let h1 = x.add(&attn_out)?;
        let h1_norm = h1.rms_norm_affine(Arc::clone(&layer.ffn_norm_gain), cfg.rms_norm_eps)?;

        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size);

        h1.add(&ffn_out)
    }
}

/// Host-side 1D position-id table matching the standard causal
/// `start_pos..start_pos+seq` layout — all three M-RoPE axes equal
/// the same scalar position.
pub fn host_text_position_ids(seq: usize, start_pos: usize) -> Vec<[i64; 3]> {
    (0..seq)
        .map(|s| {
            let p = (start_pos + s) as i64;
            [p, p, p]
        })
        .collect()
}

/// Build the half-split-layout M-RoPE cos/sin tables for a sequence
/// of 3D positions. Each output row is `head_dim` wide and encodes
/// the per-feature `cos(theta)` / `sin(theta)` where `theta` is
/// `inv_freq[feature] * position[mrope_dim(feature)]` —
/// `mrope_dim(feature)` selects temporal / height / width per the
/// repeated mrope_section chunks `[T, H, W, T, H, W]`.
///
/// The returned data is fed directly into `rope_with_tables`, which
/// expects a `[seq, head_dim]` layout with the standard LLaMA half-
/// split property `cos[i] == cos[i + head_dim/2]`.
///
/// # Half-split layout reasoning
///
/// Eager builds three per-dim tables of shape `[seq, head_dim/2]`,
/// concats each to itself (`cat(x, x, -1)`) along the feature axis,
/// then takes 6 chunks of widths `[T, H, W, T, H, W]` from the
/// appropriate axis. Because each axis-table was doubled, feature
/// `i` in the first half and feature `i + head_dim/2` in the second
/// half come from the *same* `(axis, freq_index)` pair — i.e. the
/// classic half-split layout. The output of this helper preserves
/// that property by construction.
pub fn build_mrope_tables(
    theta: f64,
    head_dim: usize,
    mrope_section: &[usize],
    positions: &[[i64; 3]],
) -> (Vec<f32>, Vec<f32>) {
    assert!(head_dim.is_multiple_of(2), "build_mrope_tables: head_dim must be even");
    let half = head_dim / 2;
    let mrope_sum: usize = mrope_section.iter().sum();
    assert_eq!(
        mrope_sum, half,
        "build_mrope_tables: mrope_section must sum to head_dim/2 ({}), got {}",
        half, mrope_sum,
    );

    // Precompute per-feature (axis_idx, freq_index_into_half) so the
    // hot loop is two multiplies per (position, feature).
    //
    // The eager flow:
    //   - per-axis cos_dim[axis] is shape [seq, half] with cell
    //     [s, i] = cos(inv_freq[i] * positions[s][axis]).
    //   - per-axis cos_full[axis] of shape [seq, head_dim] is
    //     cat(cos_dim, cos_dim, -1), so cell [s, j] for j in [0,half)
    //     uses freq_index = j, and j in [half, head_dim) uses
    //     freq_index = j - half.
    //   - the final cos[s, j] is taken from cos_full[axis(j)] where
    //     axis(j) is determined by which of the 6 mrope_section
    //     chunks j falls into.
    let mut axis_for_feat: Vec<usize> = Vec::with_capacity(head_dim);
    let mut freq_for_feat: Vec<usize> = Vec::with_capacity(head_dim);
    let mut offset = 0_usize;
    // First half: chunks 0..3 with mrope_section widths, axis = i % 3.
    for (i, &width) in mrope_section.iter().enumerate() {
        for j in 0..width {
            axis_for_feat.push(i % 3);
            freq_for_feat.push(offset + j);
        }
        offset += width;
    }
    // Second half: chunks 3..6, axis = (i + 3) % 3 = i % 3 again,
    // but freq_index resets per chunk so this mirrors the first half.
    // Note: `freq_for_feat` here must reset to track the second half
    // of the doubled axis-table, so freq is `offset_in_second_half`.
    let mut offset = 0_usize;
    for (i, &width) in mrope_section.iter().enumerate() {
        for j in 0..width {
            axis_for_feat.push(i % 3);
            freq_for_feat.push(offset + j);
        }
        offset += width;
    }
    debug_assert_eq!(axis_for_feat.len(), head_dim);

    // Precompute inv_freq[i] = theta^(-i/head_dim) for i in 0..half.
    let inv_freq: Vec<f64> = (0..half)
        .map(|i| theta.powf(-(i as f64) / (head_dim as f64) * 2.0))
        .collect();
    // Note: eager uses `(0..head_dim).step_by(2)` with exponent
    // `i / head_dim` — equivalent because `i = 2*k` so `2k/head_dim
    // = 2 * (k / head_dim)`. We index by k in [0, half).

    let seq = positions.len();
    let mut cos_data = vec![0.0_f32; seq * head_dim];
    let mut sin_data = vec![0.0_f32; seq * head_dim];

    for (s, pos3) in positions.iter().enumerate() {
        let row = s * head_dim;
        for f in 0..head_dim {
            let axis = axis_for_feat[f];
            let k = freq_for_feat[f];
            let pos = pos3[axis] as f64;
            let angle = pos * inv_freq[k];
            cos_data[row + f] = angle.cos() as f32;
            sin_data[row + f] = angle.sin() as f32;
        }
    }
    (cos_data, sin_data)
}

fn build_causal_mask(anchor: &LazyTensor, seq: usize) -> LazyTensor {
    let mut mask_data = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask_data[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    anchor.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> PaddleOcrVlTextConfig {
        // mrope_section sums to head_dim/2 = 8 (so [2, 3, 3]).
        PaddleOcrVlTextConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            use_bias: false,
            tie_word_embeddings: false,
            // head_dim/2 = 2, so single-chunk axis pattern: only the
            // temporal axis is used. Use [2] to keep section length
            // legal for tiny head_dim.
            mrope_section: vec![1, 1],
        }
    }

    fn tiny_weights(cfg: &PaddleOcrVlTextConfig) -> PaddleOcrVlTextWeights {
        let mut s: u32 = 13579;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| LayerWeights {
                attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                attn_q_bias: None,
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_k_bias: None,
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_v_bias: None,
                attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                ffn_up: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        PaddleOcrVlTextWeights {
            token_embedding,
            layers,
            final_norm_gain,
            output: Some(output),
        }
    }

    /// Tiny 2-layer forward produces the expected logits shape with
    /// no NaNs / infinities.
    #[test]
    fn forward_shape_and_finite_tiny() {
        let cfg = tiny_cfg();
        let model = PaddleOcrVlTextModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        assert_eq!(out.len(), tokens.len() * cfg.vocab_size);
        for (idx, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{idx}] = {v} not finite");
        }
    }

    /// `forward_embeds` produces byte-identical logits to `forward`
    /// when fed the same token embeddings — the two paths only differ
    /// in whether the caller pre-embeds the tokens.
    #[test]
    fn forward_embeds_matches_forward() {
        let cfg = tiny_cfg();
        let model = PaddleOcrVlTextModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![3, 1, 4, 1, 5];

        let direct = model.forward(&tokens, 0).unwrap().realize_f32();

        let embeds = LazyTensor::embed_tokens(
            model.weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            &tokens,
            &Device::cpu(),
        ).unwrap();
        let via_embeds = model.forward_embeds(&embeds, 0).unwrap().realize_f32();

        assert_eq!(direct.len(), via_embeds.len());
        for (i, (a, b)) in direct.iter().zip(via_embeds.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "logits[{i}]: forward={a} vs forward_embeds={b}",
            );
        }
    }

    /// PaddleOCR-VL deviation test. The text LM uses M-RoPE rather
    /// than standard 1D RoPE — verify the cos/sin table builder
    /// preserves the half-split property
    /// (`cos[s, i] == cos[s, i + head_dim/2]`) and that text-only
    /// positions (all 3 axes equal) reduce M-RoPE to standard RoPE.
    ///
    /// The text-only-reduces-to-1D property is what makes
    /// `forward_embeds_matches_forward` (above) a valid smoke for
    /// the M-RoPE path — without it, `forward` (which uses
    /// host_text_position_ids) and the eager standard-RoPE
    /// equivalent would not match.
    #[test]
    fn ernie_mrope_specific_deviation() {
        // 1. Half-split property: build M-RoPE tables for arbitrary
        // 3D positions and check feature pairing (i, i + head_dim/2).
        let head_dim = 8;
        let mrope_section = vec![2, 1, 1];
        let positions = vec![[3_i64, 7, 11], [1, 2, 4]];
        let (cos, sin) = build_mrope_tables(
            10_000.0, head_dim, &mrope_section, &positions,
        );
        let seq = positions.len();
        let half = head_dim / 2;
        for s in 0..seq {
            for i in 0..half {
                let c0 = cos[s * head_dim + i];
                let c1 = cos[s * head_dim + i + half];
                let s0 = sin[s * head_dim + i];
                let s1 = sin[s * head_dim + i + half];
                assert!(
                    (c0 - c1).abs() < 1e-6,
                    "half-split cos mismatch at s={s} i={i}: {c0} vs {c1}",
                );
                assert!(
                    (s0 - s1).abs() < 1e-6,
                    "half-split sin mismatch at s={s} i={i}: {s0} vs {s1}",
                );
            }
        }

        // 2. Text-only positions (all 3 axes equal) → M-RoPE ≡ 1D RoPE.
        let text_positions: Vec<[i64; 3]> =
            (0..seq).map(|s| { let p = s as i64; [p, p, p] }).collect();
        let (cos_mrope, sin_mrope) = build_mrope_tables(
            10_000.0, head_dim, &mrope_section, &text_positions,
        );
        let (cos_std, sin_std) =
            fuel_graph::build_rope_tables(10_000.0, 0, seq, head_dim);
        for (i, (a, b)) in cos_mrope.iter().zip(cos_std.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "text-only cos[{i}]: mrope={a} vs 1d={b}",
            );
        }
        for (i, (a, b)) in sin_mrope.iter().zip(sin_std.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "text-only sin[{i}]: mrope={a} vs 1d={b}",
            );
        }

        // 3. Different M-RoPE 3D positions produce different cos/sin
        // tables (i.e. the multimodal axis selection actually matters).
        let mixed_positions = vec![[0_i64, 0, 0], [0, 1, 0], [0, 0, 1]];
        let (cos_mixed, _) = build_mrope_tables(
            10_000.0, head_dim, &mrope_section, &mixed_positions,
        );
        let row1 = &cos_mixed[head_dim..2 * head_dim];
        let row2 = &cos_mixed[2 * head_dim..3 * head_dim];
        let any_diff = row1.iter().zip(row2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            any_diff,
            "M-RoPE rows for distinct 3D positions should differ — got identical",
        );

        // 4. Forward with explicit M-RoPE 3D positions runs cleanly
        // and matches the text-only forward when fed text-axis
        // positions.
        let cfg = tiny_cfg();
        let model = PaddleOcrVlTextModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![2, 4, 6];
        let embeds = LazyTensor::embed_tokens(
            model.weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            &tokens,
            &Device::cpu(),
        ).unwrap();
        let text_pos = host_text_position_ids(tokens.len(), 0);
        let via_mrope = model
            .forward_embeds_with_mrope(&embeds, &text_pos)
            .unwrap()
            .realize_f32();
        let via_1d = model.forward_embeds(&embeds, 0).unwrap().realize_f32();
        for (i, (a, b)) in via_1d.iter().zip(via_mrope.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "M-RoPE with text positions should match standard 1D RoPE at logit[{i}]: 1d={a} mrope={b}",
            );
        }
    }

    /// `tie_word_embeddings` swaps the lm_head for the transposed
    /// embedding table — exercises the `output: None` branch of the
    /// weight layout.
    #[test]
    fn tied_embeddings_runs() {
        let mut cfg = tiny_cfg();
        cfg.tie_word_embeddings = true;
        let mut weights = tiny_weights(&cfg);
        weights.output = None;
        let model = PaddleOcrVlTextModel { config: cfg.clone(), weights };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// Hidden-state entry point returns post-final-RmsNorm states of
    /// shape `(1, seq, hidden_size)`, not logits.
    #[test]
    fn forward_hidden_embeds_skips_lm_head() {
        let cfg = tiny_cfg();
        let model = PaddleOcrVlTextModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let embeds = LazyTensor::embed_tokens(
            model.weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            &tokens,
            &Device::cpu(),
        ).unwrap();
        let hidden = model.forward_hidden_embeds(&embeds, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite());
        }
    }
}

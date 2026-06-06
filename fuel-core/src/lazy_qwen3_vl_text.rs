//! Qwen3-VL text language model ported to the lazy-graph API.
//!
//! Qwen3-VL extends the Qwen3 decoder (see [`crate::lazy_qwen3`]) with
//! two VL-specific changes:
//!
//! 1. **MROPE** (multi-axis RoPE): three independent rotary position
//!    streams `(t, h, w)` are interleaved along `head_dim`. The
//!    `mrope_section = [t_dim, h_dim, w_dim]` slice (summing to
//!    `head_dim/2`) is doubled into six repeated sections
//!    `[t, h, w, t, h, w]` (summing to `head_dim`) so that the standard
//!    rotate-half RoPE op consumes the combined `[seq, head_dim]` cos/sin
//!    tables unchanged. Section `i` draws its position from axis
//!    `i % 3` (temporal / height / width).
//! 2. **Per-token MROPE positions** instead of a scalar `start_pos`:
//!    `forward_embeds_with_mrope_positions(embeds, &positions)` takes
//!    a slice of `(t, h, w)` triples, one per sequence position. Vision
//!    tokens carry the patch's `(h, w)` grid coordinate; text tokens
//!    repeat the same scalar in all three axes.
//!
//! Everything else — per-head QK-norm, GQA, SwiGLU, optional Q/K/V/O
//! biases, sliding-window gate — is inherited from the Qwen3 shape and
//! re-implemented here without disturbing [`crate::lazy_qwen3`].

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

/// Per-token MROPE position tuple: `(temporal, height, width)`.
pub type MropePos = [u32; 3];

/// Qwen3-VL text-side configuration. Mirrors [`crate::lazy_qwen3::Qwen3Config`]
/// with one addition: `mrope_section` controls the per-axis dimension
/// split for the multi-axis rotary table.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3VlTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    /// Per-axis section sizes `[t_dim, h_dim, w_dim]`. Must sum to
    /// `head_dim / 2`. The repeated layout `[t, h, w, t, h, w]` covers
    /// the full `head_dim` so that the standard rotate-half RoPE rule
    /// (`x[j+half]` paired with `x[j]`) lands on a section with the
    /// same axis owner and the same `freq_idx = j % (head_dim / 2)`.
    pub mrope_section: [usize; 3],
}

#[derive(Debug, Clone)]
pub struct Qwen3VlTextLayerExtras {
    /// `[head_dim]` — per-head RmsNorm gain for Q.
    pub q_norm_gain: Arc<[f32]>,
    /// `[head_dim]` — per-head RmsNorm gain for K.
    pub k_norm_gain: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Qwen3VlTextWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub layer_extras: Vec<Qwen3VlTextLayerExtras>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Qwen3VlTextModel {
    pub config: Qwen3VlTextConfig,
    pub weights: Qwen3VlTextWeights,
}

impl Qwen3VlTextModel {
    /// Run the LM end-to-end from token ids. `mrope_positions` carries
    /// one `(t, h, w)` triple per token; for pure-text inputs all three
    /// axes share the same scalar position (so MROPE collapses to
    /// 1D RoPE).
    pub fn forward(
        &self,
        tokens: &[u32],
        mrope_positions: &[MropePos],
    ) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens, mrope_positions)?;
        self.apply_lm_head(&h_norm)
    }

    /// Pre-embedded entry point. `embeds` shape `(1, seq, hidden_size)`,
    /// `mrope_positions.len() == seq`. Used by the Qwen3-VL composition
    /// after image tokens have been substituted into the text embedding
    /// stream.
    pub fn forward_embeds(
        &self,
        embeds: &LazyTensor,
        mrope_positions: &[MropePos],
    ) -> Result<LazyTensor> {
        let h_norm = self.run_backbone_embeds(embeds, mrope_positions)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`]. Returns the
    /// post-final-RmsNorm states `(1, seq, hidden_size)` for downstream
    /// hosts that do not want the `lm_head` projection.
    pub fn forward_hidden_embeds(
        &self,
        embeds: &LazyTensor,
        mrope_positions: &[MropePos],
    ) -> Result<LazyTensor> {
        self.run_backbone_embeds_with_deepstack(embeds, mrope_positions, &[])
    }

    /// Deepstack-aware forward. `deepstack_per_layer[i]` (if `Some`) is
    /// added into the post-layer hidden states after decoder layer `i`,
    /// matching the eager Qwen3-VL injection convention where vision
    /// residuals captured at vision layers `deepstack_visual_indexes[i]`
    /// flow into the text-side hidden stream at text layer `i`. The
    /// residual must be shaped `(1, seq, hidden_size)` and already
    /// positioned so non-visual rows are zero. The slice length may be
    /// shorter than the number of layers; remaining layers receive no
    /// injection (mirrors `if i < deepstack.len()` in eager).
    pub fn forward_embeds_with_deepstack(
        &self,
        embeds: &LazyTensor,
        mrope_positions: &[MropePos],
        deepstack_per_layer: &[Option<LazyTensor>],
    ) -> Result<LazyTensor> {
        let h_norm =
            self.run_backbone_embeds_with_deepstack(embeds, mrope_positions, deepstack_per_layer)?;
        self.apply_lm_head(&h_norm)
    }

    /// Token-embedding lookup anchored on `anchor`'s graph. Lets the
    /// composition host concatenate vision features with the resulting
    /// text embeddings before calling [`Self::forward_embeds`].
    pub fn embed_tokens_anchored(
        &self,
        anchor: &LazyTensor,
        tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        anchor.embed_tokens_anchored(
            self.weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
        )
    }

    fn apply_lm_head(&self, h_norm: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        Ok(self
            .weights
            .output
            .apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    fn run_backbone(
        &self,
        tokens: &[u32],
        mrope_positions: &[MropePos],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        if seq == 0 {
            return Err(crate::Error::Msg(
                "Qwen3VlTextModel::forward: tokens must be non-empty".into(),
            )
            .bt());
        }

        let h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
            &Device::cpu(),
        )?;
        self.run_backbone_embeds(&h, mrope_positions)
    }

    fn run_backbone_embeds(
        &self,
        embeds: &LazyTensor,
        mrope_positions: &[MropePos],
    ) -> Result<LazyTensor> {
        self.run_backbone_embeds_with_deepstack(embeds, mrope_positions, &[])
    }

    fn run_backbone_embeds_with_deepstack(
        &self,
        embeds: &LazyTensor,
        mrope_positions: &[MropePos],
        deepstack_per_layer: &[Option<LazyTensor>],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(crate::Error::Msg(format!(
                "Qwen3VlTextModel::forward_embeds: expected embeds shape \
                 (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            ))
            .bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "Qwen3VlTextModel::forward_embeds: seq must be > 0".into(),
            )
            .bt());
        }
        if mrope_positions.len() != seq {
            return Err(crate::Error::Msg(format!(
                "Qwen3VlTextModel::forward_embeds: mrope_positions.len() ({}) \
                 must equal seq ({seq})",
                mrope_positions.len(),
            ))
            .bt());
        }
        if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "Qwen3VlTextConfig: num_attention_heads * head_dim must equal hidden_size"
                    .into(),
            )
            .bt());
        }
        if weights.layers.len() != weights.layer_extras.len() {
            return Err(crate::Error::Msg(format!(
                "Qwen3VlTextWeights: layers ({}) must have matching layer_extras ({})",
                weights.layers.len(),
                weights.layer_extras.len(),
            ))
            .bt());
        }
        validate_mrope_section(&cfg.mrope_section, cfg.head_dim)?;

        let (cos_data, sin_data) = build_mrope_tables(
            cfg.rope_theta,
            mrope_positions,
            cfg.head_dim,
            &cfg.mrope_section,
        );
        let rope_shape = Shape::from_dims(&[seq, cfg.head_dim]);
        let rope_cos = embeds.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = embeds.const_f32_like(sin_data, rope_shape);

        let mut h = embeds.clone();
        for (layer_idx, (layer, extras)) in weights
            .layers
            .iter()
            .zip(weights.layer_extras.iter())
            .enumerate()
        {
            let uses_window = cfg.use_sliding_window && layer_idx < cfg.max_window_layers;
            h = self.apply_layer(&h, layer, extras, &rope_cos, &rope_sin, uses_window)?;
            if let Some(Some(residual)) = deepstack_per_layer.get(layer_idx) {
                h = h.add(residual)?;
            }
        }
        h.rms_norm_affine(
            std::sync::Arc::clone(&weights.final_norm_gain),
            cfg.rms_norm_eps,
        )
    }

    fn build_layer_mask(
        &self,
        anchor: &LazyTensor,
        seq: usize,
        uses_window: bool,
    ) -> LazyTensor {
        let cfg = &self.config;
        let window = if uses_window {
            cfg.sliding_window.unwrap_or(seq + 1)
        } else {
            seq + 1
        };
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
        layer: &LayerWeights,
        extras: &Qwen3VlTextLayerExtras,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        uses_window: bool,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let x_norm = x.rms_norm_affine(
            std::sync::Arc::clone(&layer.attn_norm_gain),
            cfg.rms_norm_eps,
        )?;

        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size)
            .add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer
            .attn_k
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)
            .add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer
            .attn_v
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)
            .add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        let q = q.rms_norm_affine(
            std::sync::Arc::clone(&extras.q_norm_gain),
            cfg.rms_norm_eps,
        )?;
        let k = k.rms_norm_affine(
            std::sync::Arc::clone(&extras.k_norm_gain),
            cfg.rms_norm_eps,
        )?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = self.build_layer_mask(x, seq, uses_window);
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out =
            layer
                .attn_o
                .apply_linear(&merged, cfg.hidden_size, cfg.hidden_size);

        let h1 = x.add(&attn_out)?;
        let h1_norm = h1.rms_norm_affine(
            std::sync::Arc::clone(&layer.ffn_norm_gain),
            cfg.rms_norm_eps,
        )?;
        let gate = layer
            .ffn_gate
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer
            .ffn_up
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out =
            layer
                .ffn_down
                .apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size);
        h1.add(&ffn_out)
    }
}

fn validate_mrope_section(section: &[usize; 3], head_dim: usize) -> Result<()> {
    let half = head_dim / 2;
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(crate::Error::Msg(format!(
            "mrope_section: head_dim ({head_dim}) must be a positive even number",
        ))
        .bt());
    }
    let sum: usize = section.iter().sum();
    if sum != half {
        return Err(crate::Error::Msg(format!(
            "mrope_section {section:?} must sum to head_dim/2 ({half}), got {sum}",
        ))
        .bt());
    }
    Ok(())
}

/// Build the MROPE cos/sin tables for `seq` tokens at the supplied
/// `(t, h, w)` positions. Output is laid out as `[seq, head_dim]` so it
/// drops straight into [`LazyTensor::rope_with_tables`].
///
/// Section layout follows the HF convention: `mrope_section * 2`
/// produces six repeated chunks `[t, h, w, t, h, w]` summing to
/// `head_dim`. Section `i` is owned by axis `i % 3`. Because each of
/// the two halves replays the same `[t, h, w]` pattern in the same
/// order with the same sizes, the section owner and the
/// `freq_idx = j % (head_dim / 2)` both match between `j` and `j + half`
/// — exactly what rotate-half RoPE expects.
fn build_mrope_tables(
    base: f64,
    mrope_positions: &[MropePos],
    head_dim: usize,
    mrope_section: &[usize; 3],
) -> (Vec<f32>, Vec<f32>) {
    let seq = mrope_positions.len();
    let half = head_dim / 2;
    let mut cos = vec![0.0_f32; seq * head_dim];
    let mut sin = vec![0.0_f32; seq * head_dim];

    // Inverse frequencies sized to head_dim/2 (matches the
    // duplicated-half layout the rest of the codebase uses).
    let inv_freq: Vec<f64> = (0..half)
        .map(|i| base.powf(-2.0 * (i as f64) / (head_dim as f64)))
        .collect();

    // Map j in 0..head_dim to its section owner (which axis: 0/1/2).
    // Sections come in order [t, h, w, t, h, w] with sizes
    // [s0, s1, s2, s0, s1, s2].
    let sections_repeated = [
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
    ];
    let mut axis_of_j = vec![0_usize; head_dim];
    let mut offset = 0_usize;
    for (i, &sec_size) in sections_repeated.iter().enumerate() {
        for k in 0..sec_size {
            axis_of_j[offset + k] = i % 3;
        }
        offset += sec_size;
    }
    debug_assert_eq!(offset, head_dim);

    for (p, pos) in mrope_positions.iter().enumerate() {
        for j in 0..head_dim {
            let axis = axis_of_j[j];
            let pos_val = pos[axis] as f64;
            let freq_idx = j % half;
            let theta = pos_val * inv_freq[freq_idx];
            cos[p * head_dim + j] = theta.cos() as f32;
            sin[p * head_dim + j] = theta.sin() as f32;
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> Qwen3VlTextConfig {
        Qwen3VlTextConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 64,
            sliding_window: None,
            max_window_layers: 0,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            attention_bias: false,
            tie_word_embeddings: false,
            mrope_section: [1, 1, 0],
        }
    }

    fn tiny_weights(cfg: &Qwen3VlTextConfig) -> Qwen3VlTextWeights {
        let mut s: u32 = 24680;
        let mut next = move || -> f32 {
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
        let mut layers = Vec::new();
        let mut layer_extras = Vec::new();
        for _ in 0..cfg.num_hidden_layers {
            layers.push(LayerWeights {
                attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                attn_q_bias: if cfg.attention_bias {
                    Some(vec_of(h, &mut *nb))
                } else {
                    None
                },
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_k_bias: if cfg.attention_bias {
                    Some(vec_of(kv, &mut *nb))
                } else {
                    None
                },
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_v_bias: if cfg.attention_bias {
                    Some(vec_of(kv, &mut *nb))
                } else {
                    None
                },
                attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                ffn_up: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
            });
            layer_extras.push(Qwen3VlTextLayerExtras {
                q_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
                k_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
            });
        }
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        Qwen3VlTextWeights {
            token_embedding,
            layers,
            layer_extras,
            final_norm_gain,
            output,
        }
    }

    fn scalar_positions(seq: usize) -> Vec<MropePos> {
        (0..seq as u32).map(|p| [p, p, p]).collect()
    }

    #[test]
    fn forward_shape_and_finite_tiny() {
        let cfg = tiny_cfg();
        let model = Qwen3VlTextModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens = vec![1_u32, 2, 3];
        let positions = scalar_positions(tokens.len());
        let logits = model.forward(&tokens, &positions).unwrap();
        assert_eq!(
            logits.shape().dims(),
            &[1, tokens.len(), cfg.vocab_size]
        );
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn mrope_positions_change_attention() {
        // Two MROPE position grids that differ on the height axis in
        // a way RoPE's translation-invariance can't absorb (varying
        // pairwise distances). `mrope_section = [1, 1, 0]` gives the
        // height axis its own RoPE slot, so changing pairwise h
        // distances MUST change the attention output. Anchor case for
        // verifying MROPE actually wires the per-axis positions
        // through, instead of collapsing to temporal-axis-only RoPE.
        let cfg = tiny_cfg();
        let model = Qwen3VlTextModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens = vec![1_u32, 2, 3, 4];

        // Both grids share the same temporal axis. Grid `a` has h
        // positions matching the temporal axis (uniform spacing);
        // grid `b` clusters two h positions together so the pairwise
        // distance pattern is *different*, not just translated.
        let positions_a: Vec<MropePos> = vec![[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]];
        let positions_b: Vec<MropePos> = vec![[0, 0, 0], [1, 1, 1], [2, 7, 2], [3, 8, 3]];

        let out_a = model.forward(&tokens, &positions_a).unwrap().realize_f32();
        let out_b = model.forward(&tokens, &positions_b).unwrap().realize_f32();
        let max_diff = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "different MROPE height pairwise distances must yield different \
             logits (max diff {max_diff})",
        );
    }

    #[test]
    fn forward_embeds_matches_forward() {
        let cfg = tiny_cfg();
        let model = Qwen3VlTextModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens = vec![1_u32, 2, 3];
        let positions = scalar_positions(tokens.len());
        let logits_ref = model.forward(&tokens, &positions).unwrap().realize_f32();

        let anchor = LazyTensor::from_f32(
            vec![0.0_f32],
            Shape::from_dims(&[1]),
            &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let logits_via_embeds = model
            .forward_embeds(&embeds, &positions)
            .unwrap()
            .realize_f32();
        assert_eq!(logits_ref.len(), logits_via_embeds.len());
        let max_diff = logits_ref
            .iter()
            .zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "forward vs forward_embeds must agree (max diff {max_diff})",
        );
    }

    #[test]
    fn forward_embeds_rejects_mismatched_positions() {
        let cfg = tiny_cfg();
        let model = Qwen3VlTextModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32],
            Shape::from_dims(&[1]),
            &Device::cpu(),
        );
        let embeds = model
            .embed_tokens_anchored(&anchor, &[1_u32, 2, 3])
            .unwrap();
        let bad_positions = scalar_positions(2); // seq is 3, only 2 supplied
        assert!(model.forward_embeds(&embeds, &bad_positions).is_err());
    }

    #[test]
    fn mrope_section_validation_catches_bad_sum() {
        let mut cfg = tiny_cfg();
        cfg.mrope_section = [2, 2, 0]; // sum 4 != head_dim/2 == 2
        let model = Qwen3VlTextModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens = vec![1_u32, 2];
        let positions = scalar_positions(tokens.len());
        assert!(model.forward(&tokens, &positions).is_err());
    }
}

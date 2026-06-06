//! Granite-MoE Hybrid (IBM Granite long-context) ported to the
//! lazy-graph API.
//!
//! Phase D specialized port. The eager reference declares a
//! "hybrid attention/Mamba" layer scheme via `layer_types:
//! Vec<TemporalKind>` but **only the attention branch is
//! implemented** (the Mamba branch bails). v1 mirrors that
//! scope: attention layers ship, Mamba layers panic. The
//! architecture is otherwise:
//!
//!   1. **Granite-style RoPE scaling** — frequencies above
//!      `low_freq_wavelen` get divided by `factor`, frequencies
//!      below `high_freq_wavelen` pass through unchanged, and
//!      frequencies in between get a smooth mix. This is the
//!      same shape as the LLaMA-3 / "YaRN-like" frequency
//!      interpolation, but lives natively in the Granite config.
//!   2. **Four scalar multipliers** as Granite-specific
//!      arithmetic glue:
//!        - `embedding_multiplier`: scales the token embedding
//!          immediately after lookup.
//!        - `attention_multiplier`: replaces the standard
//!          `1/sqrt(head_dim)` score scaling.
//!        - `residual_multiplier`: scales `attn_out` and `mlp_out`
//!          **before** the residual add (not after — matches
//!          eager `scale_tensor(x, residual_multiplier)`).
//!        - `logits_scaling`: divides the final logits by this
//!          (eager stores it as `1.0 / logits_scaling`).
//!   3. **Fused gated MLP** — `input_linear: hidden → 2 *
//!      shared_intermediate`, chunked into `(left, right)`,
//!      output = `output_linear(silu(left) * right)`. Same
//!      shape as Phi-3's fused gate_up.
//!   4. **GQA attention** with standard split-half RoPE applied
//!      with the granite-rescaled frequencies.
//!   5. **Pre-LN** with RmsNorm (no offset), two norms per
//!      block (input_layernorm + post_attention_layernorm).
//!   6. **Tied lm_head** to `embed_tokens`.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache,
//! F32, attention-only layers (eager's Mamba branch bails too).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraniteLayerType {
    Attention,
    Mamba,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraniteRopeScaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraniteMoeHybridConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub shared_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub rope_scaling: Option<GraniteRopeScaling>,
    pub layer_types: Vec<GraniteLayerType>,
    pub attention_multiplier: f32,
    pub embedding_multiplier: f32,
    pub residual_multiplier: f32,
    pub logits_scaling: f32,
}

impl GraniteMoeHybridConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    /// Effective logits divisor (eager stores `1.0 / logits_scaling`
    /// and **multiplies** by it; we keep the divisor view here for
    /// clarity).
    pub fn logits_divisor(&self) -> f32 {
        if self.logits_scaling == 0.0 { 1.0 } else { self.logits_scaling }
    }
}

#[derive(Debug, Clone)]
pub struct GraniteMoeHybridAttnWeights {
    pub q_proj: WeightStorage,
    pub k_proj: WeightStorage,
    pub v_proj: WeightStorage,
    pub o_proj: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct GraniteMoeHybridMlpWeights {
    /// `[hidden, 2 * shared_intermediate]`.
    pub input_linear: WeightStorage,
    /// `[shared_intermediate, hidden]`.
    pub output_linear: WeightStorage,
}

#[derive(Debug, Clone)]
pub enum GraniteMoeHybridLayerWeights {
    Attention {
        input_norm_gain: Arc<[f32]>,
        attn: GraniteMoeHybridAttnWeights,
        post_attn_norm_gain: Arc<[f32]>,
        mlp: GraniteMoeHybridMlpWeights,
    },
    /// Reserved for future Mamba expansion — v1 panics if encountered
    /// (matches eager).
    Mamba,
}

#[derive(Debug, Clone)]
pub struct GraniteMoeHybridWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<GraniteMoeHybridLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct GraniteMoeHybridModel {
    pub config: GraniteMoeHybridConfig,
    pub weights: GraniteMoeHybridWeights,
}

impl GraniteMoeHybridWeights {
    /// Load Granite-MoE-Hybrid weights from a `MmapedSafetensors` file
    /// using the standard HuggingFace naming. Granite-MoE-Hybrid is
    /// declared as a hybrid attention/Mamba layer scheme via
    /// `cfg.layer_types`; per the v1 forward (and matching eager
    /// scope), only the Attention branch carries loadable weights —
    /// Mamba layers materialize as the unit
    /// [`GraniteMoeHybridLayerWeights::Mamba`] variant and the forward
    /// pass returns `Err` if any Mamba layer is exercised.
    ///
    /// Tied lm_head: Granite-MoE-Hybrid ties `lm_head.weight` to
    /// `model.embed_tokens.weight` (the eager model projects via
    /// `embeddings().t()`), so no `lm_head.*` tensor is loaded — the
    /// lazy model holds only the token embedding and reuses it on the
    /// output side at forward time.
    ///
    /// Tensor names mirrored from
    /// `fuel_transformers::models::llm::granitemoehybrid`:
    ///   - `model.embed_tokens.weight` → [`GraniteMoeHybridWeights::token_embedding`]
    ///   - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` → `attn.{q,k,v,o}_proj`
    ///   - `model.layers.{i}.shared_mlp.input_linear.weight` → `mlp.input_linear`
    ///   - `model.layers.{i}.shared_mlp.output_linear.weight` → `mlp.output_linear`
    ///   - `model.layers.{i}.input_layernorm.weight` → `input_norm_gain`
    ///   - `model.layers.{i}.post_attention_layernorm.weight` → `post_attn_norm_gain`
    ///   - `model.norm.weight` → `final_norm_gain`
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &GraniteMoeHybridConfig,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};

        if cfg.layer_types.len() != cfg.num_hidden_layers {
            return Err(crate::Error::Msg(format!(
                "GraniteMoeHybridWeights::load_from_mmapped: \
                 cfg.layer_types length ({}) must match num_hidden_layers ({})",
                cfg.layer_types.len(), cfg.num_hidden_layers,
            )).bt());
        }

        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let inter = cfg.shared_intermediate_size;

        let token_embedding = load_tensor_as_f32(st, "model.embed_tokens.weight")?;
        if token_embedding.len() != cfg.vocab_size * h {
            crate::bail!(
                "model.embed_tokens.weight: {} elts, expected {} ({}×{})",
                token_embedding.len(), cfg.vocab_size * h, cfg.vocab_size, h,
            );
        }

        let mut layers: Vec<GraniteMoeHybridLayerWeights> =
            Vec::with_capacity(cfg.num_hidden_layers);
        for (li, kind) in cfg.layer_types.iter().enumerate() {
            match kind {
                GraniteLayerType::Attention => {
                    let p = format!("model.layers.{li}");
                    let q_proj = load_transposed_matrix_preserve_dtype(
                        st, &format!("{p}.self_attn.q_proj.weight"), q_dim, h,
                    )?;
                    let k_proj = load_transposed_matrix_preserve_dtype(
                        st, &format!("{p}.self_attn.k_proj.weight"), kv_dim, h,
                    )?;
                    let v_proj = load_transposed_matrix_preserve_dtype(
                        st, &format!("{p}.self_attn.v_proj.weight"), kv_dim, h,
                    )?;
                    let o_proj = load_transposed_matrix_preserve_dtype(
                        st, &format!("{p}.self_attn.o_proj.weight"), h, q_dim,
                    )?;
                    let input_linear = load_transposed_matrix_preserve_dtype(
                        st, &format!("{p}.shared_mlp.input_linear.weight"), 2 * inter, h,
                    )?;
                    let output_linear = load_transposed_matrix_preserve_dtype(
                        st, &format!("{p}.shared_mlp.output_linear.weight"), h, inter,
                    )?;
                    let input_norm_gain = load_tensor_as_f32(
                        st, &format!("{p}.input_layernorm.weight"),
                    )?;
                    let post_attn_norm_gain = load_tensor_as_f32(
                        st, &format!("{p}.post_attention_layernorm.weight"),
                    )?;
                    layers.push(GraniteMoeHybridLayerWeights::Attention {
                        input_norm_gain: Arc::from(input_norm_gain),
                        attn: GraniteMoeHybridAttnWeights {
                            q_proj, k_proj, v_proj, o_proj,
                        },
                        post_attn_norm_gain: Arc::from(post_attn_norm_gain),
                        mlp: GraniteMoeHybridMlpWeights {
                            input_linear, output_linear,
                        },
                    });
                }
                GraniteLayerType::Mamba => {
                    // v1 scope: Mamba layers are loaded as the unit
                    // variant; the forward pass returns Err if one is
                    // exercised. Matches eager (which also bails on
                    // Mamba). Tensor names like
                    // `{p}.mamba.{A_log,D,in_proj,...}` are documented
                    // here for the future expansion that will materialize
                    // them, but skipped today.
                    layers.push(GraniteMoeHybridLayerWeights::Mamba);
                }
            }
        }

        let final_norm_gain = load_tensor_as_f32(st, "model.norm.weight")?;

        Ok(GraniteMoeHybridWeights {
            token_embedding: Arc::from(token_embedding),
            layers,
            final_norm_gain: Arc::from(final_norm_gain),
        })
    }
}

impl GraniteMoeHybridModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Granite-specific: `embedding_multiplier` is applied,
    /// `logits_divisor` is NOT (it sits past the tied lm_head).
    /// Granite-rescaled RoPE tables and per-layer Attention vs.
    /// Mamba selection are honored.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. The caller MUST apply Granite's
    /// `embedding_multiplier` scaling before invoking — matches the
    /// convention used by lazy_paligemma / lazy_llava / lazy_voxtral
    /// so the multimodal composition layer owns the scaling decision.
    ///
    /// Layers of kind `Mamba` return Err (matches eager scope).
    pub fn forward_embeds(
        &self, scaled_embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let h_norm = self.run_backbone_embeds(scaled_embeds, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`].
    pub fn forward_hidden_embeds(
        &self, scaled_embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.run_backbone_embeds(scaled_embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder. NOTE:
    /// the `embedding_multiplier` scaling is NOT applied — caller is
    /// responsible (matches the scaled-embeds contract above).
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
        let lm_head = WeightStorage::F32(self.weights.token_embedding.clone());
        let logits = lm_head.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size);
        let div = cfg.logits_divisor();
        if (div - 1.0).abs() < f32::EPSILON {
            Ok(logits)
        } else {
            Ok(logits.mul_scalar(1.0 / div as f64))
        }
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0);

        let mut h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;
        if (cfg.embedding_multiplier - 1.0).abs() > f32::EPSILON {
            h = h.mul_scalar(cfg.embedding_multiplier as f64);
        }
        self.run_backbone_embeds(&h, start_pos)
    }

    fn run_backbone_embeds(
        &self, scaled_embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = scaled_embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(crate::Error::Msg(format!(
                "GraniteMoeHybridModel::forward_embeds: expected scaled_embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "GraniteMoeHybridModel::forward_embeds: seq must be > 0".into(),
            ).bt());
        }
        if cfg.num_attention_heads * cfg.head_dim() != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "num_attention_heads * head_dim must equal hidden_size".into(),
            ).bt());
        }
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
            return Err(crate::Error::Msg(
                "num_attention_heads must be a multiple of num_key_value_heads".into(),
            ).bt());
        }
        if weights.layers.len() != cfg.num_hidden_layers {
            return Err(crate::Error::Msg(format!(
                "weights.layers length ({}) must match num_hidden_layers ({})",
                weights.layers.len(), cfg.num_hidden_layers,
            )).bt());
        }
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            return Err(crate::Error::Msg(format!(
                "layer_types length ({}) must match num_hidden_layers ({})",
                cfg.layer_types.len(), cfg.num_hidden_layers,
            )).bt());
        }
        let mut h = scaled_embeds.clone();

        let head_dim = cfg.head_dim();
        let (cos_data, sin_data) = build_granite_rope_tables(
            cfg.rope_theta as f64, start_pos, seq, head_dim,
            cfg.rope_scaling.as_ref(),
        );
        let rope_shape = Shape::from_dims(&[seq, head_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        for (idx, (layer, kind)) in weights.layers.iter()
            .zip(cfg.layer_types.iter()).enumerate() {
            match (layer, kind) {
                (
                    GraniteMoeHybridLayerWeights::Attention { input_norm_gain, attn, post_attn_norm_gain, mlp },
                    GraniteLayerType::Attention,
                ) => {
                    h = self.apply_attn_block(&h, input_norm_gain, attn, post_attn_norm_gain, mlp, &rope_cos, &rope_sin)?;
                }
                (GraniteMoeHybridLayerWeights::Mamba, GraniteLayerType::Mamba) => {
                    return Err(crate::Error::Msg(format!(
                        "GraniteMoeHybrid layer {idx}: Mamba layers not yet supported \
                         (matches eager scope); use a non-Mamba layer kind",
                    )).bt());
                }
                _ => return Err(crate::Error::Msg(format!(
                    "GraniteMoeHybrid layer {idx}: weight kind does not match \
                     layer_types[{idx}] — config + weights are inconsistent",
                )).bt()),
            }
        }

        Ok(h.rms_norm_affine(std::sync::Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps)?)
    }

    fn apply_attn_block(
        &self,
        x: &LazyTensor,
        input_norm_gain: &Arc<[f32]>,
        attn: &GraniteMoeHybridAttnWeights,
        post_attn_norm_gain: &Arc<[f32]>,
        mlp: &GraniteMoeHybridMlpWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;

        // Pre-attention norm.
        let x_norm = x.rms_norm_affine(Arc::clone(input_norm_gain), cfg.rms_norm_eps)?;
        let attn_out = self.attention(&x_norm, attn, rope_cos, rope_sin)?;
        // Residual multiplier (scales attn_out BEFORE the residual add).
        let attn_scaled = if (cfg.residual_multiplier - 1.0).abs() > f32::EPSILON {
            attn_out.mul_scalar(cfg.residual_multiplier as f64)
        } else {
            attn_out
        };
        let h1 = x.add(&attn_scaled)?;

        // Pre-MLP norm.
        let h1_norm = h1.rms_norm_affine(Arc::clone(post_attn_norm_gain), cfg.rms_norm_eps)?;
        let mlp_out = self.apply_mlp(&h1_norm, mlp)?;
        let mlp_scaled = if (cfg.residual_multiplier - 1.0).abs() > f32::EPSILON {
            mlp_out.mul_scalar(cfg.residual_multiplier as f64)
        } else {
            mlp_out
        };
        h1.add(&mlp_scaled)
    }

    fn attention(
        &self,
        x: &LazyTensor,
        w: &GraniteMoeHybridAttnWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;

        let q = w.q_proj.apply_linear(x, cfg.hidden_size, q_dim);
        let k = w.k_proj.apply_linear(x, cfg.hidden_size, kv_dim);
        let v = w.v_proj.apply_linear(x, cfg.hidden_size, kv_dim);

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        // Granite uses `attention_multiplier` as the scaling factor
        // INSTEAD OF `1/sqrt(head_dim)`.
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(cfg.attention_multiplier as f64);
        let mask = LazyTensor::additive_causal_mask_like(x, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let probs = scores_masked.softmax_last_dim()?;
        let ctx = probs.matmul(&v_full)?;
        let merged = ctx.merge_heads()?;
        Ok(w.o_proj.apply_linear(&merged, q_dim, cfg.hidden_size))
    }

    fn apply_mlp(
        &self,
        x: &LazyTensor,
        w: &GraniteMoeHybridMlpWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let inter = cfg.shared_intermediate_size;
        let fused = w.input_linear.apply_linear(x, h, 2 * inter);
        let left = fused.slice(2_usize, 0, inter)?;
        let right = fused.slice(2_usize, inter, inter)?;
        let gated = left.silu().mul(&right)?;
        Ok(w.output_linear.apply_linear(&gated, inter, h))
    }
}

/// Build cos/sin RoPE tables with Granite's per-frequency
/// rescaling. The split-half convention (features `i` and
/// `i + half` share a frequency) matches
/// [`fuel_graph::build_rope_tables`].
fn build_granite_rope_tables(
    base: f64,
    start_pos: usize,
    seq: usize,
    head_dim: usize,
    rope_scaling: Option<&GraniteRopeScaling>,
) -> (Vec<f32>, Vec<f32>) {
    assert!(head_dim % 2 == 0);
    let half = head_dim / 2;
    // Compute per-i base frequencies.
    let mut inv_freqs: Vec<f32> = (0..half)
        .map(|i| (base.powf(-2.0 * (i as f64) / (head_dim as f64))) as f32)
        .collect();

    // Granite scaling: rebucket each frequency by wavelength.
    if let Some(s) = rope_scaling {
        let low_freq_wavelen = s.original_max_position_embeddings as f32 / s.low_freq_factor;
        let high_freq_wavelen = s.original_max_position_embeddings as f32 / s.high_freq_factor;
        for freq in inv_freqs.iter_mut() {
            let wavelen = 2.0 * std::f32::consts::PI / *freq;
            *freq = if wavelen < high_freq_wavelen {
                *freq
            } else if wavelen > low_freq_wavelen {
                *freq / s.factor
            } else {
                let smooth = (s.original_max_position_embeddings as f32 / wavelen
                    - s.low_freq_factor)
                    / (s.high_freq_factor - s.low_freq_factor);
                (1.0 - smooth) * *freq / s.factor + smooth * *freq
            };
        }
    }

    let mut cos = vec![0.0_f32; seq * head_dim];
    let mut sin = vec![0.0_f32; seq * head_dim];
    for p in 0..seq {
        let pos = (start_pos + p) as f32;
        for i in 0..half {
            let theta = pos * inv_freqs[i];
            let c = theta.cos();
            let s_v = theta.sin();
            cos[p * head_dim + i] = c;
            cos[p * head_dim + i + half] = c;
            sin[p * head_dim + i] = s_v;
            sin[p * head_dim + i + half] = s_v;
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &GraniteMoeHybridConfig) -> GraniteMoeHybridWeights {
        let mut s: u32 = 13131;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let inter = cfg.shared_intermediate_size;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<GraniteMoeHybridLayerWeights> = cfg.layer_types
            .iter()
            .map(|kind| match kind {
                GraniteLayerType::Attention => {
                    GraniteMoeHybridLayerWeights::Attention {
                        input_norm_gain: Arc::from(vec![1.0_f32; h]),
                        attn: GraniteMoeHybridAttnWeights {
                            q_proj: WeightStorage::F32(vec_of(h * q_dim, &mut *nb)),
                            k_proj: WeightStorage::F32(vec_of(h * kv_dim, &mut *nb)),
                            v_proj: WeightStorage::F32(vec_of(h * kv_dim, &mut *nb)),
                            o_proj: WeightStorage::F32(vec_of(q_dim * h, &mut *nb)),
                        },
                        post_attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                        mlp: GraniteMoeHybridMlpWeights {
                            input_linear: WeightStorage::F32(vec_of(h * (2 * inter), &mut *nb)),
                            output_linear: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
                        },
                    }
                }
                GraniteLayerType::Mamba => GraniteMoeHybridLayerWeights::Mamba,
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        GraniteMoeHybridWeights { token_embedding, layers, final_norm_gain }
    }

    fn tiny_config() -> GraniteMoeHybridConfig {
        GraniteMoeHybridConfig {
            vocab_size: 16, hidden_size: 8,
            intermediate_size: 16, shared_intermediate_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 2, num_key_value_heads: 1,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5, rope_theta: 10_000.0,
            rope_scaling: None,
            layer_types: vec![GraniteLayerType::Attention, GraniteLayerType::Attention],
            attention_multiplier: 0.25, // = 1/sqrt(head_dim=16) is irrelevant — Granite chooses freely
            embedding_multiplier: 2.0,
            residual_multiplier: 0.5,
            logits_scaling: 4.0,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = GraniteMoeHybridModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// Each of the 4 scalar multipliers must measurably affect
    /// the output. Toggle them one at a time relative to a
    /// neutral (==1.0) baseline.
    #[test]
    fn scalar_multipliers_are_wired() {
        let cfg_neutral = GraniteMoeHybridConfig {
            embedding_multiplier: 1.0,
            attention_multiplier: 1.0,
            residual_multiplier: 1.0,
            logits_scaling: 1.0,
            ..tiny_config()
        };
        let weights = tiny_weights(&cfg_neutral);
        let toks = [1_u32, 2, 3];
        let baseline = GraniteMoeHybridModel { config: cfg_neutral.clone(), weights: weights.clone() }
            .forward(&toks, 0).unwrap().realize_f32();

        let check = |cfg: GraniteMoeHybridConfig, label: &str| {
            let m = GraniteMoeHybridModel { config: cfg, weights: weights.clone() };
            let out = m.forward(&toks, 0).unwrap().realize_f32();
            let mut max_diff = 0.0_f32;
            for (a, b) in baseline.iter().zip(out.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
            assert!(max_diff > 1e-6, "{label} must alter output, max_diff = {max_diff}");
        };

        check(GraniteMoeHybridConfig { embedding_multiplier: 2.5, ..cfg_neutral.clone() },
              "embedding_multiplier");
        check(GraniteMoeHybridConfig { attention_multiplier: 0.5, ..cfg_neutral.clone() },
              "attention_multiplier");
        check(GraniteMoeHybridConfig { residual_multiplier: 2.0, ..cfg_neutral.clone() },
              "residual_multiplier");
        check(GraniteMoeHybridConfig { logits_scaling: 3.0, ..cfg_neutral.clone() },
              "logits_scaling");
    }

    /// Granite RoPE scaling must measurably alter the table
    /// for a config that uses the scaled regime.
    #[test]
    fn granite_rope_scaling_alters_tables() {
        let (cos_a, sin_a) = build_granite_rope_tables(
            10_000.0, 0, 4, 8, None,
        );
        let (cos_b, sin_b) = build_granite_rope_tables(
            10_000.0, 0, 4, 8,
            Some(&GraniteRopeScaling {
                factor: 4.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8,
            }),
        );
        // Some frequencies should differ.
        let mut max_diff = 0.0_f32;
        for (a, b) in cos_a.iter().zip(cos_b.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        for (a, b) in sin_a.iter().zip(sin_b.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(max_diff > 1e-6,
            "Granite RoPE scaling must change the tables, max_diff = {max_diff}");
    }

    /// Mamba layers return Err in v1 (matches eager scope; was a panic
    /// pre-2026-06, retired per "never panic in production").
    #[test]
    fn mamba_layer_returns_err() {
        let cfg = GraniteMoeHybridConfig {
            layer_types: vec![GraniteLayerType::Mamba, GraniteLayerType::Attention],
            ..tiny_config()
        };
        let model = GraniteMoeHybridModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let res = model.forward(&[1, 2], 0);
        assert!(res.is_err(), "Mamba layer should return Err, got Ok");
        let err = format!("{}", res.unwrap_err());
        assert!(err.contains("Mamba layers not yet supported"),
            "unexpected error message: {err}");
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config();
        let model = GraniteMoeHybridModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        // forward_embeds takes scaled embeds (caller applies
        // embedding_multiplier) — mirrors the Gemma convention.
        let cfg = tiny_config();
        let model = GraniteMoeHybridModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let scaled = if (cfg.embedding_multiplier - 1.0).abs() > f32::EPSILON {
            embeds.mul_scalar(cfg.embedding_multiplier as f64)
        } else {
            embeds
        };
        let logits_via_embeds = model.forward_embeds(&scaled, 0).unwrap().realize_f32();
        let max_diff = logits_ref.iter().zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "GraniteMoeHybrid forward vs forward_embeds (post-scale) must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = tiny_config();
        let model = GraniteMoeHybridModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let bad = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]), &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = tiny_config();
        let model = GraniteMoeHybridModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![5, 7];
        let h_ref = model.forward_hidden(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let scaled = if (cfg.embedding_multiplier - 1.0).abs() > f32::EPSILON {
            embeds.mul_scalar(cfg.embedding_multiplier as f64)
        } else {
            embeds
        };
        let h_via_embeds = model.forward_hidden_embeds(&scaled, 0).unwrap().realize_f32();
        let max_diff = h_ref.iter().zip(h_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "GraniteMoeHybrid forward_hidden vs forward_hidden_embeds (post-scale) must agree (max diff {max_diff})");
    }
}

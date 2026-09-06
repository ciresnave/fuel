// SPDX-License-Identifier: MIT OR Apache-2.0
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
//! `head_dim`), four norms per block (input + post-attn + pre-FFN +
//! post-FFN), and tied lm_head/embeddings. `num_heads * head_dim`
//! is **not** required to equal `hidden_size` — Gemma 3 uses
//! independent attention head/embedding sizes (e.g. 1B has
//! `hidden_size=1152`, `num_heads=4`, `head_dim=256`).
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache
//! (recomputes each call), F32 activations. The lm_head reuses
//! `token_embedding` as a tied projection.

use fuel_core::inference_context::{DecodeSession, DecodeTokenData, InferenceContext, KvCache};
use fuel_core::lazy::{Tensor, WeightStorage};
use fuel_core::persistent_decode::{
    DecodeBackbone, DecodeDims, DecodeLayerInputs, MaskPlan, PersistentDecodeModel, RopePlan,
};
use fuel_core::{Device, Result};
use fuel_ir::{DType, Shape};
use std::sync::Arc;

pub use crate::models::lazy_gemma::GemmaActivation;

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

/// Map a Gemma-3 `hidden_activation` string to [`GemmaActivation`]. Gemma-3
/// configs carry the CORRECTED key `hidden_activation` with `gelu_pytorch_tanh`
/// (unlike Gemma-1, whose `hidden_act: "gelu"` is a known misnomer). Unknown
/// values error rather than silently defaulting.
fn gemma3_activation_from_str(s: &str) -> fuel_core::Result<GemmaActivation> {
    match s {
        "gelu" => Ok(GemmaActivation::Gelu),
        "gelu_pytorch_tanh" | "gelu_new" => Ok(GemmaActivation::GeluPytorchTanh),
        other => Err(fuel_core::Error::Msg(format!(
            "unsupported Gemma-3 hidden_activation {other:?} (expected gelu/gelu_pytorch_tanh)"
        ))),
    }
}

// ROADMAP item 8 (II): config-from-path on the #57 template. Gemma-3 is a FLAT
// artifact: a `serde` raw with HF field names + Gemma-3's own constant defaults,
// then `resolve` routes kv heads + head_dim through the shared `fuel_core::hf_config`
// rules (Gemma-3 ships an explicit, decoupled head_dim, e.g. 256 vs 1152/4=288).
// The two logit-softcappings are `Option` (null in the reference configs). The
// non-serde `GemmaActivation` enum is parsed from the `hidden_activation` string.
#[derive(Debug, Clone, serde::Deserialize)]
struct Gemma3ConfigRaw {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_gemma3_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_gemma3_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_gemma3_rope_local_base_freq")]
    rope_local_base_freq: f64,
    max_position_embeddings: usize,
    #[serde(default = "default_gemma3_sliding_window")]
    sliding_window: usize,
    #[serde(default = "default_gemma3_sliding_window_pattern")]
    sliding_window_pattern: usize,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    hidden_activation: Option<String>,
    #[serde(default)]
    attn_logit_softcapping: Option<f64>,
    #[serde(default)]
    final_logit_softcapping: Option<f64>,
}

fn default_gemma3_rms_norm_eps() -> f64 {
    1e-6
}
fn default_gemma3_rope_theta() -> f64 {
    1_000_000.0
}
fn default_gemma3_rope_local_base_freq() -> f64 {
    10_000.0
}
fn default_gemma3_sliding_window() -> usize {
    4096
}
fn default_gemma3_sliding_window_pattern() -> usize {
    6
}

impl Gemma3ConfigRaw {
    fn from_json_str(json: &str) -> fuel_core::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| fuel_core::Error::Msg(format!("parsing Gemma-3 config.json: {e}")))
    }

    fn resolve(self) -> fuel_core::Result<Gemma3Config> {
        let hidden_activation = match self.hidden_activation.as_deref() {
            None => GemmaActivation::GeluPytorchTanh,
            Some(s) => gemma3_activation_from_str(s)?,
        };
        Ok(Gemma3Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: fuel_core::hf_config::num_key_value_heads(
                self.num_key_value_heads,
                self.num_attention_heads,
            )?,
            head_dim: fuel_core::hf_config::head_dim(
                self.head_dim,
                self.hidden_size,
                self.num_attention_heads,
            ),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            rope_local_base_freq: self.rope_local_base_freq,
            max_position_embeddings: self.max_position_embeddings,
            sliding_window: self.sliding_window,
            sliding_window_pattern: self.sliding_window_pattern,
            attention_bias: self.attention_bias,
            hidden_activation,
            attn_logit_softcapping: self.attn_logit_softcapping,
            final_logit_softcapping: self.final_logit_softcapping,
        })
    }
}

impl Gemma3Config {
    /// Parse a HuggingFace `config.json` string into a [`Gemma3Config`].
    ///
    /// ROADMAP item 8 (II): reads the artifact rather than returning a preset —
    /// see the born-red `gemma3_config_from_hf_json_parses_the_artifact`.
    pub fn from_hf_json_str(json: &str) -> fuel_core::Result<Self> {
        Gemma3ConfigRaw::from_json_str(json)?.resolve()
    }
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
    /// Process-unique identity for THIS weight set — what lets a held decode
    /// plan tell two same-architecture models apart (GAP-029). Mint with
    /// [`fuel_core::decode_shape::ModelInstanceId::next`].
    pub instance: fuel_core::decode_shape::ModelInstanceId,
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
        !(layer_idx + 1).is_multiple_of(self.config.sliding_window_pattern)
    }

    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final offset RmsNorm
    /// and return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the tied `lm_head` matmul AND the final logit
    /// softcapping. Gemma3-specific: dual-RoPE (global + local) +
    /// per-layer sliding-window pattern + sqrt(hidden_size)
    /// embedding scaling — all honored by the shared backbone.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips the token-embedding step and
    /// runs the decoder over pre-embedded inputs (typically the
    /// concatenation of vision-projected embeddings + text token
    /// embeddings).
    ///
    /// `scaled_embeds` shape: `(1, seq, hidden_size)`. The caller
    /// must apply Gemma's `sqrt(hidden_size)` scaling before
    /// invoking — matching the convention used by lazy_paligemma /
    /// lazy_llava / lazy_voxtral so the multimodal composition
    /// layer owns the scaling decision.
    pub fn forward_embeds(&self, scaled_embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let h_norm = self.decode_from_scaled_embeds(scaled_embeds, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`]. Returns the
    /// post-final-RmsNorm states `(1, seq, hidden_size)`. Used by
    /// retrieval / embedding consumers.
    pub fn forward_hidden_embeds(
        &self,
        scaled_embeds: &Tensor,
        start_pos: usize,
    ) -> Result<Tensor> {
        self.decode_from_scaled_embeds(scaled_embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder. Used by
    /// multimodal compositions to obtain text-side embeddings that
    /// will be spliced with vision features before
    /// [`Self::forward_embeds`].
    ///
    /// Returns shape `(1, seq, hidden_size)`. The caller is responsible
    /// for the `sqrt(hidden_size)` scaling.
    pub fn embed_tokens_anchored(&self, anchor: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        let cfg = &self.config;
        anchor.embed_tokens_anchored(
            self.weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
        )
    }

    fn apply_lm_head(&self, h_norm: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let lm_head = WeightStorage::F32(self.weights.token_embedding.clone());
        let logits = lm_head.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size)?;
        match cfg.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(logits.mul_scalar(1.0 / sc).tanh().mul_scalar(sc)),
        }
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0, "Gemma3Model: tokens must be non-empty");

        let h = Tensor::embed_tokens(
            weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
            &Device::cpu(),
        )?;
        let h = h.mul_scalar((cfg.hidden_size as f64).sqrt());
        self.decode_from_scaled_embeds(&h, start_pos)
    }

    fn decode_from_scaled_embeds(
        &self,
        scaled_embeds: &Tensor,
        start_pos: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = scaled_embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(fuel_core::Error::Msg(format!(
                "Gemma3Model::forward_embeds: expected scaled_embeds shape \
                 (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            ))
            .bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(fuel_core::Error::Msg(
                "Gemma3Model::forward_embeds: seq must be > 0".into(),
            )
            .bt());
        }
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(fuel_core::Error::Msg(
                "Gemma3Config: num_attention_heads must be a multiple of num_key_value_heads"
                    .into(),
            )
            .bt());
        }
        if cfg.sliding_window_pattern == 0 {
            return Err(fuel_core::Error::Msg(
                "Gemma3Config: sliding_window_pattern must be > 0".into(),
            )
            .bt());
        }
        let mut h = scaled_embeds.clone();

        let (rope_cos_g, rope_sin_g) =
            h.rope_tables_const(cfg.rope_theta, start_pos, seq, cfg.head_dim);
        let (rope_cos_l, rope_sin_l) =
            h.rope_tables_const(cfg.rope_local_base_freq, start_pos, seq, cfg.head_dim);

        let full_mask = self.build_mask(&h, seq, None);
        let sliding_mask = self.build_mask(&h, seq, Some(cfg.sliding_window));

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let uses_window = self.layer_uses_sliding(layer_idx);
            let (rope_cos, rope_sin) = if uses_window {
                (&rope_cos_l, &rope_sin_l)
            } else {
                (&rope_cos_g, &rope_sin_g)
            };
            let mask = if uses_window {
                &sliding_mask
            } else {
                &full_mask
            };
            h = self.apply_layer(&h, layer, rope_cos, rope_sin, mask)?;
        }
        h.rms_norm_affine_with_offset(&weights.final_norm_gain, 1.0, cfg.rms_norm_eps)
    }

    fn build_mask(&self, anchor: &Tensor, seq: usize, sliding: Option<usize>) -> Tensor {
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
        x: &Tensor,
        layer: &Gemma3LayerWeights,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        // Pre-attention offset RmsNorm.
        let residual = x.clone();
        let x_norm =
            x.rms_norm_affine_with_offset(&layer.input_norm_gain, 1.0, cfg.rms_norm_eps)?;

        // Q / K / V projections; note Q goes to num_heads*head_dim
        // which is NOT necessarily equal to hidden_size.
        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.hidden_size, q_dim)?
            .add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer
            .attn_k
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer
            .attn_v
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

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
        let attn_out = layer
            .attn_o
            .apply_linear(&merged, q_dim, cfg.hidden_size)?
            .add_optional_trailing_bias(layer.attn_o_bias.as_ref())?;
        // post_attention_layernorm wraps the attn output BEFORE the residual add.
        let attn_out_norm = attn_out.rms_norm_affine_with_offset(
            &layer.post_attn_norm_gain,
            1.0,
            cfg.rms_norm_eps,
        )?;
        let h1 = residual.add(&attn_out_norm)?;

        // Pre-FFN offset RmsNorm.
        let residual2 = h1.clone();
        let h1_norm =
            h1.rms_norm_affine_with_offset(&layer.pre_ffn_norm_gain, 1.0, cfg.rms_norm_eps)?;

        // GELU gated FFN.
        let gate = layer
            .ffn_gate
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let up = layer
            .ffn_up
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let activated = match cfg.hidden_activation {
            GemmaActivation::Gelu => gate.gelu_erf(),
            GemmaActivation::GeluPytorchTanh => gate.gelu(),
        };
        let ffn_in = activated.mul(&up)?;
        let ffn_out =
            layer
                .ffn_down
                .apply_linear(&ffn_in, cfg.intermediate_size, cfg.hidden_size)?;
        // post_feedforward_layernorm wraps the FFN output BEFORE the residual add.
        let ffn_out_norm = ffn_out.rms_norm_affine_with_offset(
            &layer.post_ffn_norm_gain,
            1.0,
            cfg.rms_norm_eps,
        )?;

        residual2.add(&ffn_out_norm)
    }
}

// Gemma's offset RmsNorm: `y = (x / rms) * (gain + 1)`.

// ---- GAP-029 · persistent-KV decode ----------------------------------------
//
// Gemma3 is the family that forced the seam's two newest capabilities, and it
// is the only one so far that varies BOTH axes per layer:
//
//   mask : sliding window vs full causal, on a MODULAR pattern
//   rope : `rope_local_base_freq` vs `rope_theta` — DIFFERENT TABLE BYTES
//
// The RoPE base is the interesting one. SmolLm3 also varies RoPE per layer, by
// SKIPPING it, and needed nothing from the seam: skipping consumes no different
// data. Gemma3 needs different bytes per variant, which is what `RopePlan`
// exists for. Variation in DATA needs a variant axis; variation in
// WHETHER-TO-APPLY does not.
//
// It also scales embeddings by sqrt(hidden_size) — `DecodeDims::embed_scale`,
// applied by the seam BEFORE the activation dtype cast (prefill scales in f32).

impl Gemma3Model {
    /// Per-layer mask: sliding-window on `layer_uses_sliding` layers, full
    /// causal elsewhere. The predicate is **modular**, not a prefix split, which
    /// is why this is [`MaskPlan::per_layer_window`] rather than
    /// `split_window`.
    pub fn decode_mask_plan(&self) -> MaskPlan {
        let cfg = &self.config;
        MaskPlan::per_layer_window(cfg.num_hidden_layers, cfg.sliding_window, |i| {
            self.layer_uses_sliding(i)
        })
    }

    /// Per-layer RoPE **base** — the axis no other family needs.
    ///
    /// ⚠️ Collapses to a single variant when the two bases are equal, and
    /// Gemma3's own shipped fixtures do exactly that
    /// (`rope_local_base_freq: 10_000.0` alongside `rope_theta: 10_000.0`,
    /// deliberately, per the comment at its definition). A decode test on an
    /// unmodified `tiny_config()` therefore exercises **one** RoPE table and
    /// would pass under a single-base port — which is why the windowed decode
    /// test asserts the two bases differ before it asserts anything else.
    pub fn decode_rope_plan(&self) -> RopePlan {
        let cfg = &self.config;
        RopePlan::per_layer_base(
            cfg.num_hidden_layers,
            cfg.rope_local_base_freq, // sliding layers
            cfg.rope_theta,           // full-causal layers
            |i| self.layer_uses_sliding(i),
        )
    }

    /// Identity a held decode plan is baked against. Both plans contribute
    /// STRUCTURE (variant counts + per-layer assignment); the window width and
    /// the base values are data, rebound per token, and excluded.
    pub fn decode_shape_key(&self) -> u64 {
        let cfg = &self.config;
        let mut h = fuel_core::decode_shape::ShapeKeyHasher::new();
        h.mix_str("gemma3")
            .mix_instance(self.weights.instance)
            .mix_u64(cfg.num_hidden_layers as u64)
            .mix_u64(cfg.num_attention_heads as u64)
            .mix_u64(cfg.num_key_value_heads as u64)
            .mix_u64(cfg.head_dim as u64)
            .mix_u64(cfg.hidden_size as u64)
            .mix_u64(cfg.intermediate_size as u64)
            .mix_u64(cfg.vocab_size as u64)
            .mix_f64(cfg.rms_norm_eps);
        // Softcapping changes the graph's SHAPE (extra tanh/mul nodes), so
        // presence must key — a plan built with it must not be reused without.
        h.mix_present(cfg.attn_logit_softcapping.is_some());
        h.mix_present(cfg.final_logit_softcapping.is_some());
        self.decode_mask_plan().mix_into(&mut h);
        self.decode_rope_plan().mix_into(&mut h);
        h.finish()
    }

    /// Decode/prefill through a pre-allocated [`KvCache`], rebuilding the graph
    /// each step.
    pub fn forward_with_kv_context(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
    ) -> Result<Vec<f32>> {
        fuel_core::persistent_decode::forward_with_kv_context(self, tokens, cache, ctx, false, None)
    }

    /// Plan-once persistent decode.
    pub fn forward_with_kv_context_persistent(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<DecodeSession>,
    ) -> Result<Vec<f32>> {
        fuel_core::persistent_decode::forward_with_kv_context_persistent(
            self, tokens, cache, ctx, session, None,
        )
    }

    /// Persistent decode with the session owned by the `InferenceContext`.
    pub fn forward_decode_step(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
    ) -> Result<Vec<f32>> {
        let mut session = ctx.take_decode_session();
        let out = self.forward_with_kv_context_persistent(tokens, cache, ctx, &mut session);
        ctx.put_decode_session(session);
        out
    }

    /// One Gemma3 layer against the pre-allocated KV buffers.
    ///
    /// Same math as [`Self::apply_layer`] — four offset RmsNorms per block,
    /// per-head QK-norm, attention-score softcap, GELU-gated FFN, and
    /// `attn_o: q_dim -> hidden_size` (Gemma3 is the first family on this seam
    /// where `num_attention_heads * head_dim != hidden_size`). The two decode
    /// differences are the seam's usual ones: GQA rides `matmul`'s head
    /// broadcast, and no flash-decode arm is offered — its single
    /// `k_len = cached_len + seq` cannot express a sliding window, so on a
    /// windowed layer it would silently drop the window on bf16/CUDA.
    fn apply_layer_with_kv_writes(
        &self,
        layer: &Gemma3LayerWeights,
        inputs: &DecodeLayerInputs<'_>,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let x = inputs.x;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let act_dtype = x.dtype();

        let residual = x.clone();
        let x_norm =
            x.rms_norm_affine_with_offset(&layer.input_norm_gain, 1.0, cfg.rms_norm_eps)?;

        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.hidden_size, q_dim)?
            .add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer
            .attn_k
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer
            .attn_v
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v_h = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        // Per-head offset QK-norm, POST-reshape and PRE-RoPE — the prefill
        // order, which is what makes decode agree with it.
        let q = q.rms_norm_affine_with_offset(&layer.q_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let k = k.rms_norm_affine_with_offset(&layer.k_norm_gain, 1.0, cfg.rms_norm_eps)?;

        let q_r = q
            .to_dtype(DType::F32)?
            .rope_with_tables(inputs.rope_cos, inputs.rope_sin)?
            .to_dtype(act_dtype)?;
        let k_r = k
            .to_dtype(DType::F32)?
            .rope_with_tables(inputs.rope_cos, inputs.rope_sin)?
            .to_dtype(act_dtype)?;

        let write_ranges = vec![
            (0, batch),
            (0, cfg.num_key_value_heads),
            (0, seq), // axis-2 start is dynamic; width = seq
            (0, cfg.head_dim),
        ];
        let (full_k, full_v) = match inputs.offset {
            Some(off) => (
                inputs
                    .k_cache
                    .write_slice_doff(&k_r, off, 2, write_ranges.clone())?,
                inputs
                    .v_cache
                    .write_slice_doff(&v_h, off, 2, write_ranges)?,
            ),
            None => {
                let dyn_off = fuel_ir::DynScalar::Sym(inputs.cached_len_sym);
                (
                    inputs
                        .k_cache
                        .write_slice_dyn(&k_r, write_ranges.clone(), 2, dyn_off)?,
                    inputs
                        .v_cache
                        .write_slice_dyn(&v_h, write_ranges, 2, dyn_off)?,
                )
            }
        };

        let k_t = full_k.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let scores_capped = match cfg.attn_logit_softcapping {
            None => scores_scaled,
            Some(sc) => scores_scaled.mul_scalar(1.0 / sc).tanh().mul_scalar(sc),
        };
        let scores_masked = scores_capped.broadcast_add(inputs.mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&full_v)?;

        // `merge_heads()` inlined as permute + reshape so `attn_v`'s SOLE
        // consumer (the permute) can be named as the flash arm's reconverge —
        // arm-0 runnability requires the merge to read arm 0.
        let attn_v_permuted = attn_v.permute([0, 2, 1, 3_usize])?;
        fuel_core::lazy::offer_flash_decode_arm_for_region(
            q_r.graph(),
            q_r.node_id(),
            full_k.node_id(),
            full_v.node_id(),
            attn_v.node_id(),
            attn_v_permuted.node_id(),
            scale as f32,
            inputs.attended_len_sym,
            // GAP-194: this layer's own window — sliding layers are declined.
            inputs.attn_window,
            // ⚠️ AND THE SOFTCAP, which is the second false field the original
            // offer site hardcoded. Gemma3 is the family that has one: passing
            // `None` here would let the arm compute UNCAPPED attention while
            // the decomposed path caps it. It disqualifies (no kernel support),
            // which is correct — the point is that it is STATED.
            cfg.attn_logit_softcapping.map(|v| v as f32),
            fuel_dispatch::decode_flash::FlashArmCapability::production(),
        )?;
        let merged = attn_v_permuted.reshape(Shape::from_dims(&[batch, seq, q_dim]))?;
        let attn_out = layer
            .attn_o
            .apply_linear(&merged, q_dim, cfg.hidden_size)?
            .add_optional_trailing_bias(layer.attn_o_bias.as_ref())?;
        // post_attention_layernorm wraps the attn output BEFORE the residual add.
        let attn_out_norm = attn_out.rms_norm_affine_with_offset(
            &layer.post_attn_norm_gain,
            1.0,
            cfg.rms_norm_eps,
        )?;
        let h1 = residual.add(&attn_out_norm)?;

        let residual2 = h1.clone();
        let h1_norm =
            h1.rms_norm_affine_with_offset(&layer.pre_ffn_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let gate = layer
            .ffn_gate
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let up = layer
            .ffn_up
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let activated = match cfg.hidden_activation {
            GemmaActivation::Gelu => gate.gelu_erf(),
            GemmaActivation::GeluPytorchTanh => gate.gelu(),
        };
        let ffn_in = activated.mul(&up)?;
        let ffn_out =
            layer
                .ffn_down
                .apply_linear(&ffn_in, cfg.intermediate_size, cfg.hidden_size)?;
        let ffn_out_norm = ffn_out.rms_norm_affine_with_offset(
            &layer.post_ffn_norm_gain,
            1.0,
            cfg.rms_norm_eps,
        )?;

        residual2.add(&ffn_out_norm)
    }
}

// ===========================================================================
// BORN-RED RECORD — GEMMA3 (GAP-029, 2026-08-14)
//
// Gemma3 varies BOTH per-layer axes, so each was born red SEPARATELY: a test
// that only proves "some variation matters" cannot tell you which half works.
// Config: `decode_cfg()` — 4 layers, `sliding_window_pattern: 3`,
// `sliding_window: 3`, `rope_local_base_freq: 1_000` vs `rope_theta: 10_000`.
// Prefill 3, decode 3. Both runs carried `Compiling fuel-core`.
//
//   MASK axis  — `decode_mask_plan` -> MaskPlan::dense(..) (single-mask port)
//     per-step max|diff| at abs pos 3..=5 = [5.876e-2, 3.466e-2, 9.661e-2]
//
//   RoPE axis  — `decode_rope_plan` -> RopePlan::single(rope_theta, ..)
//     per-step max|diff| at abs pos 3..=5 = [1.235e-3, 8.673e-4, 1.031e-3]
//
//   Correct (both plans) = 0.0 at every step; control (pattern 1) = 0.0.
//
// ⚠️ THE RoPE NUMBER IS THE ONE TO READ. At ~1e-3 it is ~100x the 1e-5 oracle
// — and comfortably UNDER the `diff < 5e-3 || rel < 1e-2` of the decode test
// this port would naturally have been modelled on. THAT TEMPLATE GOES GREEN ON
// A DUAL-BASE DEFECT. The tolerance being measured rather than inherited is the
// only reason this axis is covered at all.
//
// ⚠️ AND UNLIKE QWEN2, ALL THREE POSITIONS DIVERGE ON THE MASK AXIS. Qwen2's
// leading step was clean because its window (4) could not exclude anything
// until absolute position 4; Gemma3's window is 3, so it bites from position 3.
// The "leading zero proves discrimination" reasoning is therefore CONFIG-
// specific, not a property of the seam — do not port that inference across
// families without re-deriving it from the window and the prefill length.
//
// PER-FAMILY SABOTAGE (increment 3 constraint (1)): drop the FFN residual in
// Gemma3's OWN `apply_layer_with_kv_writes` ->
//   lazy_gemma3 x2 FAILED; test result: 135 passed; 2 failed
// EXACTLY ONE family red. Qwen2, Qwen3, Qwen3Moe, Phi3, SmolLm3, Glm4, Llama,
// Llama3, Phi and DeepSeek2 all stayed green — including the six that share the
// build path and the two RoPE/mask plan constructors this family introduced.
//
// NOTE ON WHAT THE SHIPPED TEST ACTUALLY CATCHES: with the guards restored,
// both sabotages are caught by the `n_variants() == 2` assertions BEFORE the
// logits comparison runs — a structural failure that names the broken axis
// instead of a number that says only "something diverged". The magnitudes above
// were measured with those guards temporarily bypassed, and are recorded here
// because the threshold's calibration is otherwise unverifiable.
// ===========================================================================

impl PersistentDecodeModel for Gemma3Model {
    fn decode_n_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn build_decode_token_data(
        &self,
        device: &Device,
        cached_len: usize,
        tokens: &[u32],
        session: &DecodeSession,
        cache: &KvCache,
        rope_inv_freq: Option<&[f64]>,
    ) -> Result<DecodeTokenData> {
        let host = fuel_core::persistent_decode::compute_decode_token_host(
            self,
            cached_len,
            tokens,
            session.max_seq_len(),
            rope_inv_freq,
        );
        fuel_core::persistent_decode::upload_decode_token_data(
            device,
            &host,
            cache.dtype.unwrap_or(DType::F32),
            session.offset_node().is_some().then_some(cached_len),
        )
    }
}

impl DecodeBackbone for Gemma3Model {
    fn decode_family(&self) -> &'static str {
        "Gemma3Model"
    }

    fn decode_dims(&self) -> DecodeDims {
        let cfg = &self.config;
        DecodeDims {
            n_layers: cfg.num_hidden_layers,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            // ⚠️ `hidden_size`, which for Gemma3 is NOT
            // `num_attention_heads * head_dim` — the first family on this seam
            // where those differ. The seam uses this only for the embed reshape
            // and the LM-head width, both of which want `hidden_size`.
            hidden: cfg.hidden_size,
            vocab: cfg.vocab_size,
            rope_width: cfg.head_dim,
            // Gemma scales embeddings by sqrt(hidden_size). The SEAM applies it,
            // before the dtype cast — prefill scales in f32, so applying it
            // after the cast would round in bf16 and diverge invisibly on an
            // f32 gate.
            embed_scale: Some((cfg.hidden_size as f64).sqrt()),
        }
    }

    fn decode_shape_key(&self) -> u64 {
        Gemma3Model::decode_shape_key(self)
    }

    fn decode_mask_plan(&self) -> MaskPlan {
        Gemma3Model::decode_mask_plan(self)
    }

    fn decode_rope_plan(&self) -> RopePlan {
        Gemma3Model::decode_rope_plan(self)
    }

    fn decode_token_embedding(&self) -> Arc<[f32]> {
        self.weights.token_embedding.clone()
    }

    fn decode_apply_layer(
        &self,
        layer_idx: usize,
        inputs: &DecodeLayerInputs<'_>,
    ) -> Result<Tensor> {
        self.apply_layer_with_kv_writes(&self.weights.layers[layer_idx], inputs)
    }

    /// Offset final norm + the **tied** LM head (Gemma3 has no separate
    /// `output` weight) + final logit softcapping.
    fn decode_final_norm_and_head(&self, h: &Tensor) -> Result<Tensor> {
        let h_norm = h.rms_norm_affine_with_offset(
            &self.weights.final_norm_gain,
            1.0,
            self.config.rms_norm_eps,
        )?;
        self.apply_lm_head(&h_norm)
    }
}

// ---- Safetensors loader ----------------------------------------------------

impl Gemma3Weights {
    /// Load Gemma 3 weights from a `MmapedSafetensors` file using the
    /// standard HuggingFace naming. Gemma 3 ties `lm_head.weight` to
    /// `model.embed_tokens.weight` — there is no separate output
    /// projection field on `Gemma3Weights` (the `apply_lm_head` path
    /// reuses the embedding table directly).
    ///
    /// Tensor names mirrored from `fuel_transformers::models::llm::gemma3`:
    ///   - `model.embed_tokens.weight` → [`Gemma3Weights::token_embedding`]
    ///   - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
    ///   - `model.layers.{i}.self_attn.{q,k,v,o}_proj.bias`
    ///     (loaded only when `attention_bias == true`)
    ///   - `model.layers.{i}.self_attn.q_norm.weight` → `q_norm_gain`
    ///   - `model.layers.{i}.self_attn.k_norm.weight` → `k_norm_gain`
    ///   - `model.layers.{i}.input_layernorm.weight` → `input_norm_gain`
    ///   - `model.layers.{i}.post_attention_layernorm.weight` → `post_attn_norm_gain`
    ///   - `model.layers.{i}.pre_feedforward_layernorm.weight` → `pre_ffn_norm_gain`
    ///   - `model.layers.{i}.post_feedforward_layernorm.weight` → `post_ffn_norm_gain`
    ///   - `model.layers.{i}.mlp.{gate,up,down}_proj.weight` → `ffn_{gate,up,down}`
    ///   - `model.norm.weight` → `final_norm_gain`
    ///
    /// Gemma 3 uses independent attention/embedding dims —
    /// `num_attention_heads * head_dim` is NOT required to equal
    /// `hidden_size`. The Q projection is `[q_dim, hidden_size]` where
    /// `q_dim = num_attention_heads * head_dim`, and o_proj inverts
    /// that to `[hidden_size, q_dim]`.
    pub fn load_from_mmapped(
        st: &fuel_core::safetensors::MmapedSafetensors,
        cfg: &Gemma3Config,
    ) -> Result<Self> {
        use fuel_core::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};

        let h = cfg.hidden_size;
        let i_dim = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let token_embedding = load_tensor_as_f32(st, "model.embed_tokens.weight")?;
        if token_embedding.len() != cfg.vocab_size * h {
            fuel_core::bail!(
                "model.embed_tokens.weight: {} elts, expected {} ({}×{})",
                token_embedding.len(),
                cfg.vocab_size * h,
                cfg.vocab_size,
                h,
            );
        }

        let mut layers: Vec<Gemma3LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for li in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{li}");
            let attn_q = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.q_proj.weight"),
                q_dim,
                h,
            )?;
            let attn_k = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                h,
            )?;
            let attn_v = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.v_proj.weight"),
                kv_dim,
                h,
            )?;
            let attn_o = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.o_proj.weight"),
                h,
                q_dim,
            )?;
            let attn_q_bias = if cfg.attention_bias {
                load_tensor_as_f32(st, &format!("{p}.self_attn.q_proj.bias"))
                    .ok()
                    .map(Arc::from)
            } else {
                None
            };
            let attn_k_bias = if cfg.attention_bias {
                load_tensor_as_f32(st, &format!("{p}.self_attn.k_proj.bias"))
                    .ok()
                    .map(Arc::from)
            } else {
                None
            };
            let attn_v_bias = if cfg.attention_bias {
                load_tensor_as_f32(st, &format!("{p}.self_attn.v_proj.bias"))
                    .ok()
                    .map(Arc::from)
            } else {
                None
            };
            let attn_o_bias = if cfg.attention_bias {
                load_tensor_as_f32(st, &format!("{p}.self_attn.o_proj.bias"))
                    .ok()
                    .map(Arc::from)
            } else {
                None
            };
            let q_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.self_attn.q_norm.weight"),
            )?);
            let k_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.self_attn.k_norm.weight"),
            )?);
            let input_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.input_layernorm.weight"),
            )?);
            let post_attn_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.post_attention_layernorm.weight"),
            )?);
            let pre_ffn_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.pre_feedforward_layernorm.weight"),
            )?);
            let post_ffn_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.post_feedforward_layernorm.weight"),
            )?);
            let ffn_gate = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.mlp.gate_proj.weight"),
                i_dim,
                h,
            )?;
            let ffn_up = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.mlp.up_proj.weight"),
                i_dim,
                h,
            )?;
            let ffn_down = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.mlp.down_proj.weight"),
                h,
                i_dim,
            )?;
            layers.push(Gemma3LayerWeights {
                attn_q,
                attn_q_bias,
                attn_k,
                attn_k_bias,
                attn_v,
                attn_v_bias,
                attn_o,
                attn_o_bias,
                q_norm_gain,
                k_norm_gain,
                input_norm_gain,
                post_attn_norm_gain,
                pre_ffn_norm_gain,
                post_ffn_norm_gain,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.norm.weight")?);

        Ok(Gemma3Weights {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding: Arc::from(token_embedding),
            layers,
            final_norm_gain,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ROADMAP item 8 (II). Golden values from google/gemma-3-1b-it's real
    // config.json (model_type "gemma3_text"). No size preset, so a second
    // distinct config gives the constant-parser discrimination. head_dim is
    // EXPLICIT and DECOUPLED (256 vs 1152/4 = 288).
    const GEMMA3_1B_CONFIG_JSON: &str = r#"{
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3_text",
        "vocab_size": 262144,
        "hidden_size": 1152,
        "intermediate_size": 6912,
        "num_hidden_layers": 26,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-06,
        "rope_local_base_freq": 10000,
        "rope_theta": 1000000,
        "sliding_window": 512,
        "sliding_window_pattern": 6,
        "hidden_activation": "gelu_pytorch_tanh",
        "attn_logit_softcapping": null,
        "final_logit_softcapping": null,
        "attention_bias": false
    }"#;

    #[test]
    fn gemma3_config_from_hf_json_parses_the_artifact() {
        let cfg = Gemma3Config::from_hf_json_str(GEMMA3_1B_CONFIG_JSON).unwrap();
        assert_eq!(cfg.hidden_size, 1152);
        assert_eq!(cfg.num_hidden_layers, 26);
        assert_eq!(cfg.num_attention_heads, 4);
        assert_eq!(cfg.vocab_size, 262_144);
        assert_eq!(cfg.intermediate_size, 6912);
        // GQA: default would be num_attention_heads (4); 1 proves the key was READ.
        assert_eq!(cfg.num_key_value_heads, 1);
        // head_dim EXPLICIT and decoupled — 256, NOT 1152/4 = 288.
        assert_eq!(cfg.head_dim, 256);
        assert_ne!(cfg.head_dim, 1152 / 4);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.rope_local_base_freq, 10_000.0);
        assert_eq!(cfg.sliding_window, 512);
        assert_eq!(cfg.sliding_window_pattern, 6);
        assert_eq!(cfg.hidden_activation, GemmaActivation::GeluPytorchTanh);
        // null softcaps → None
        assert_eq!(cfg.attn_logit_softcapping, None);
        assert_eq!(cfg.final_logit_softcapping, None);
    }

    /// A SECOND distinct config: distinct sizes, softcaps PRESENT (→ Some), an
    /// explicit `hidden_activation: "gelu"` (→ Gelu, exercising the map's other
    /// arm), and omitted rope_local_base_freq/sliding_window_pattern (→ defaults).
    #[test]
    fn gemma3_config_reads_a_second_distinct_config() {
        let json = r#"{
            "model_type": "gemma3_text",
            "vocab_size": 262144,
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "num_hidden_layers": 34,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 131072,
            "sliding_window": 1024,
            "hidden_activation": "gelu",
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0
        }"#;
        let cfg = Gemma3Config::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        // explicit "gelu" → Gelu (the map's non-default arm)
        assert_eq!(cfg.hidden_activation, GemmaActivation::Gelu);
        assert_eq!(cfg.attn_logit_softcapping, Some(50.0));
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
        // omitted → defaults
        assert_eq!(cfg.rope_local_base_freq, 10_000.0);
        assert_eq!(cfg.sliding_window_pattern, 6);
        assert_ne!(cfg.hidden_size, 1152);
    }

    /// `num_key_value_heads` ABSENT → defaults to `num_attention_heads`.
    #[test]
    fn gemma3_config_gqa_defaults_to_num_heads_when_absent() {
        let json = r#"{
            "model_type": "gemma3_text",
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "head_dim": 16,
            "max_position_embeddings": 128
        }"#;
        let cfg = Gemma3Config::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.num_key_value_heads, 8); // absent → num_attention_heads
        // absent hidden_activation → Gemma-3's default GeluPytorchTanh
        assert_eq!(cfg.hidden_activation, GemmaActivation::GeluPytorchTanh);
    }

    /// TRUE MQA (`num_key_value_heads = 1`) survives, not collapsed.
    #[test]
    fn gemma3_config_preserves_true_mqa() {
        let json = r#"{
            "model_type": "gemma3_text",
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "head_dim": 16,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128
        }"#;
        let cfg = Gemma3Config::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.num_key_value_heads, 1);
    }

    /// An UNKNOWN `hidden_activation` ERRORS rather than silently defaulting.
    #[test]
    fn gemma3_config_rejects_unknown_hidden_activation() {
        let json = r#"{
            "model_type": "gemma3_text",
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "head_dim": 16,
            "max_position_embeddings": 128,
            "hidden_activation": "silu"
        }"#;
        assert!(Gemma3Config::from_hf_json_str(json).is_err());
    }

    fn tiny_weights(cfg: &Gemma3Config) -> Gemma3Weights {
        let mut s: u32 = 5151;
        let next = move || -> f32 {
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
                attn_q_bias: if cfg.attention_bias {
                    Some(vec_of(q_dim, &mut *next_box))
                } else {
                    None
                },
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
                attn_k_bias: if cfg.attention_bias {
                    Some(vec_of(kv, &mut *next_box))
                } else {
                    None
                },
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
                attn_v_bias: if cfg.attention_bias {
                    Some(vec_of(kv, &mut *next_box))
                } else {
                    None
                },
                attn_o: WeightStorage::F32(vec_of(q_dim * h, &mut *next_box)),
                attn_o_bias: if cfg.attention_bias {
                    Some(vec_of(h, &mut *next_box))
                } else {
                    None
                },
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
        Gemma3Weights {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            final_norm_gain,
        }
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
            head_dim: 4, // q_dim=16, kv_dim=8 — neither matches hidden_size.
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
        let model = Gemma3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
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
        let m_a = Gemma3Model {
            config: cfg_a.clone(),
            weights: weights.clone(),
        };
        let m_b = Gemma3Model {
            config: cfg_b.clone(),
            weights,
        };
        let toks: Vec<u32> = vec![3, 7, 2, 9, 1];
        let a = m_a.forward(&toks, 0).unwrap().realize_f32();
        let b = m_b.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (av, bv) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((av - bv).abs());
        }
        assert!(
            max_diff > 1e-6,
            "pattern change must alter output, max_diff = {max_diff}"
        );
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
        let m_no = Gemma3Model {
            config: cfg_no,
            weights: weights.clone(),
        };
        let m_yes = Gemma3Model {
            config: cfg_yes,
            weights,
        };
        let toks: Vec<u32> = vec![1, 2, 3];
        let a = m_no.forward(&toks, 0).unwrap().realize_f32();
        let b = m_yes.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (av, bv) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((av - bv).abs());
        }
        assert!(
            max_diff > 1e-6,
            "attn soft-cap must alter output, max_diff = {max_diff}"
        );
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
        let m_no = Gemma3Model {
            config: cfg_no,
            weights: weights.clone(),
        };
        let m_yes = Gemma3Model {
            config: cfg_yes,
            weights,
        };
        let toks: Vec<u32> = vec![4, 5, 6];
        let a = m_no.forward(&toks, 0).unwrap().realize_f32();
        let b = m_yes.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (av, bv) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((av - bv).abs());
        }
        assert!(
            max_diff > 1e-6,
            "final soft-cap must alter output, max_diff = {max_diff}"
        );
    }

    /// With sliding_window_pattern=2 and 4 layers, layers 0 and 2
    /// are local (sliding) and layers 1 and 3 are global. Verify
    /// `layer_uses_sliding` matches.
    #[test]
    fn layer_pattern_assignment() {
        let mut cfg = tiny_config();
        cfg.sliding_window_pattern = 2;
        let model = Gemma3Model {
            config: cfg,
            weights: tiny_weights(&tiny_config()),
        };
        // (i + 1) % 2 > 0  →  i is even (0, 2 → local) ; odd (1, 3 → global)
        assert!(model.layer_uses_sliding(0));
        assert!(!model.layer_uses_sliding(1));
        assert!(model.layer_uses_sliding(2));
        assert!(!model.layer_uses_sliding(3));
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config();
        let model = Gemma3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
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
        let model = Gemma3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();
        let anchor = Tensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu());
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let scaled = embeds.mul_scalar((cfg.hidden_size as f64).sqrt());
        let logits_via_embeds = model.forward_embeds(&scaled, 0).unwrap().realize_f32();
        assert_eq!(logits_ref.len(), logits_via_embeds.len());
        let max_diff = logits_ref
            .iter()
            .zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "Gemma3 forward vs forward_embeds (post-scale) must agree (max diff {max_diff})"
        );
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = tiny_config();
        let model = Gemma3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let bad_embeds = Tensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]),
            &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad_embeds, 0).is_err());
        let rank2 = Tensor::from_f32(
            vec![0.0_f32; cfg.hidden_size],
            Shape::from_dims(&[1, cfg.hidden_size]),
            &Device::cpu(),
        );
        assert!(model.forward_embeds(&rank2, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = tiny_config();
        let model = Gemma3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![5, 7];
        let h_ref = model.forward_hidden(&tokens, 0).unwrap().realize_f32();
        let anchor = Tensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu());
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let scaled = embeds.mul_scalar((cfg.hidden_size as f64).sqrt());
        let h_via_embeds = model
            .forward_hidden_embeds(&scaled, 0)
            .unwrap()
            .realize_f32();
        let max_diff = h_ref
            .iter()
            .zip(h_via_embeds.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "Gemma3 forward_hidden vs forward_hidden_embeds (post-scale) must agree (max diff {max_diff})"
        );
    }

    // ---- GAP-029 · persistent decode ---------------------------------------

    /// Measured, not inherited. The natural template
    /// (`forward_with_kv_context_decode_matches_non_cached_forward`) asserts
    /// `diff < 5e-3 || rel < 1e-2`, which sits ABOVE every divergence this
    /// program has measured for a mis-ported decode axis.
    const DECODE_ORACLE_ABS: f32 = 1e-5;

    /// ⚠️ **`tiny_config()` IS DEGENERATE ON THE RoPE AXIS and this config
    /// exists to fix that.**
    ///
    /// The shipped fixture sets `rope_local_base_freq: 10_000.0` alongside
    /// `rope_theta: 10_000.0` — deliberately, per its own comment, "for the
    /// tables match test". With the two bases equal, `RopePlan::per_layer_base`
    /// correctly collapses to ONE variant, every layer reads identical tables,
    /// and **a decode test on the unmodified fixture passes under a single-base
    /// port.** The tolerance would be fine, the assertion would be fine, and the
    /// INPUT would have removed the axis under test.
    ///
    /// So this overrides the local base. `sliding_window_pattern: 3` over 4
    /// layers keeps the mask axis mixed too (layers 0/1/3 sliding, layer 2 full).
    fn decode_cfg() -> Gemma3Config {
        Gemma3Config {
            rope_local_base_freq: 1_000.0,
            ..tiny_config()
        }
    }

    /// Max |logit diff| per decode step against the non-cached forward at the
    /// same absolute position. `>= 3` decode steps so assertions reach the
    /// per-token REBIND path rather than only the held-graph build.
    fn decode_vs_forward_max_abs(cfg: &Gemma3Config, tokens: &[u32], prefill: usize) -> Vec<f32> {
        let n_decode = tokens.len() - prefill;
        assert!(
            n_decode >= 3,
            "need >= 3 decode tokens to reach the rebind path"
        );
        let model = Gemma3Model {
            config: cfg.clone(),
            weights: tiny_weights(cfg),
        };

        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            tokens.len(),
            DType::F32,
            &dev,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(dev);
        let mut session: Option<DecodeSession> = None;

        model
            .forward_with_kv_context_persistent(
                &tokens[..prefill],
                &mut cache,
                &mut ctx,
                &mut session,
            )
            .expect("prefill");
        assert!(
            session.is_none(),
            "prefill (seq > 1) must NOT build the held session"
        );

        let mut out = Vec::with_capacity(n_decode);
        for pos in prefill..tokens.len() {
            let got = model
                .forward_with_kv_context_persistent(
                    &tokens[pos..=pos],
                    &mut cache,
                    &mut ctx,
                    &mut session,
                )
                .expect("decode");
            assert!(session.is_some(), "decode must hold a session from token 1");
            let full = model.forward(&tokens[..=pos], 0).unwrap().realize_f32();
            let expected = &full[pos * cfg.vocab_size..(pos + 1) * cfg.vocab_size];
            out.push(
                got.iter()
                    .zip(expected.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max),
            );
        }
        assert_eq!(
            cache.cached_len,
            tokens.len(),
            "cache must advance every step"
        );
        out
    }

    /// ⚠️ **NON-DISCRIMINATION CONTROL.** `sliding_window_pattern: 1` makes
    /// every layer full-causal + global-base, so BOTH plans collapse to one
    /// variant. This passes under a correct port AND under one that ignores
    /// per-layer variation entirely — it certifies the seam, the embed scale,
    /// the four offset norms, QK-norm, softcapping, the tied LM head and the
    /// rebind path, and it certifies **nothing** about per-layer variation.
    #[test]
    fn gemma3_decode_matches_forward_when_no_layer_varies() {
        let cfg = Gemma3Config {
            sliding_window_pattern: 1,
            ..decode_cfg()
        };
        let model = Gemma3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        assert_eq!(
            model.decode_mask_plan().n_variants(),
            1,
            "control must be uniform"
        );
        assert_eq!(
            model.decode_rope_plan().n_variants(),
            1,
            "control must be uniform"
        );

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let diffs = decode_vs_forward_max_abs(&cfg, &tokens, 3);
        let worst = diffs.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            worst < DECODE_ORACLE_ABS,
            "uniform Gemma3 decode diverged: per-step max|diff| = {diffs:?}. This is \
             the CONTROL — the varying test proves nothing until it is green.",
        );
    }

    /// **GAP-029 — Gemma3 decode with BOTH axes varying per layer.**
    ///
    /// ⚠️ **The first four assertions are the non-vacuity guard, and they are
    /// the point of this test.** Gemma3 is the only family whose per-layer RoPE
    /// *base* is genuine DATA variation, and its shipped fixture collapses that
    /// axis. Asserting the derived `n_variants() == 2` — not merely that the
    /// config fields differ — is what makes a green here mean the dual-base path
    /// actually ran.
    ///
    /// **Born red, observed, on each axis separately** (numbers in the
    /// sabotage record below `decode_from_scaled_embeds`).
    #[test]
    fn gemma3_per_layer_mask_and_rope_decode_matches_forward() {
        let cfg = decode_cfg();
        let model = Gemma3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };

        // --- non-vacuity of the INPUT, then of the DERIVED structure ---
        assert_ne!(
            cfg.rope_local_base_freq, cfg.rope_theta,
            "fixture guard: with equal bases the RoPE axis collapses and this test \
             passes under a single-base port",
        );
        assert_eq!(
            model.decode_rope_plan().n_variants(),
            2,
            "the dual-base path must actually be live, not merely configured",
        );
        assert_eq!(
            model.decode_mask_plan().n_variants(),
            2,
            "the per-layer mask path must actually be live",
        );
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        assert!(
            tokens.len() > cfg.sliding_window,
            "non-vacuity: the sliding window must actually exclude something",
        );

        let diffs = decode_vs_forward_max_abs(&cfg, &tokens, 3);
        let worst = diffs.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            worst < DECODE_ORACLE_ABS,
            "Gemma3 decode diverged from the per-layer forward: per-step max|diff| at \
             absolute positions 3..=5 = {diffs:?} (limit {DECODE_ORACLE_ABS:e}). A \
             single-mask OR single-RoPE-base port produces this signature.",
        );
    }

    /// The degeneracy itself, asserted rather than left as a comment: with the
    /// shipped fixture's equal bases the RoPE plan MUST collapse to one variant.
    ///
    /// This is what makes the guard in the sibling test meaningful — it pins
    /// that `n_variants()` genuinely tracks the fixture, so asserting `== 2`
    /// there is a real check and not a tautology.
    #[test]
    fn gemma3_equal_rope_bases_collapse_to_one_variant() {
        let cfg = tiny_config();
        assert_eq!(
            cfg.rope_local_base_freq, cfg.rope_theta,
            "the shipped fixture is expected to be RoPE-degenerate; if this changes, \
             `decode_cfg`'s override may no longer be needed",
        );
        let model = Gemma3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        assert_eq!(
            model.decode_rope_plan().n_variants(),
            1,
            "equal bases must collapse — otherwise the graph would carry two \
             identical tables and `n_variants` would stop being a vacuity signal",
        );
        // The mask axis is independent and stays mixed at pattern 3.
        assert_eq!(model.decode_mask_plan().n_variants(), 2);
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0
//! Qwen2 (non-MoE) decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Qwen2 = Mistral + Q/K/V biases + per-layer
//! sliding-window gating (`use_sliding_window` + `max_window_layers`).
//! Everything else (GQA, RmsNorm, SwiGLU FFN, RoPE) matches LLaMA /
//! Mistral so we reuse [`fuel_core::lazy::LayerWeights`] directly — the
//! optional `attn_{q,k,v}_bias` fields on `LayerWeights` already
//! handle Qwen2's bias layout.
//!
//! # Sliding window
//!
//! Qwen2's `Config` carries:
//!   - `sliding_window: usize` (always set, e.g. 32768 for 7B)
//!   - `use_sliding_window: bool` — global switch
//!   - `max_window_layers: usize` — first N layers use the window;
//!     remaining layers run dense. Mixed-mode is the canonical Qwen2
//!     setup.
//!
//! The lazy port honors this by building per-layer masks: layer `i`
//! uses the sliding-window mask iff
//! `use_sliding_window && i < max_window_layers`; otherwise dense
//! causal.
//!
//! # Scope
//!
//! Same as the Mistral port — forward-only, single sequence
//! (`batch == 1`), no KV cache (recomputes each call), F32
//! activations, sliding-window mask built per-forward as a const.
//!
//! # Weight names (HuggingFace safetensors)
//!
//! Mirrors eager `fuel_transformers::models::llm::qwen2`:
//!   - `model.embed_tokens.weight` → `token_embedding`
//!   - `model.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}` →
//!     `attn_{q,k,v}` + `attn_{q,k,v}_bias` (biases ARE present)
//!   - `model.layers.{i}.self_attn.o_proj.weight` → `attn_o`
//!     (no bias)
//!   - `model.layers.{i}.mlp.{gate,up,down}_proj.weight` →
//!     `ffn_{gate,up,down}`
//!   - `model.layers.{i}.input_layernorm.weight` → `attn_norm_gain`
//!   - `model.layers.{i}.post_attention_layernorm.weight` →
//!     `ffn_norm_gain`
//!   - `model.norm.weight` → `final_norm_gain`
//!   - `lm_head.weight` → `output` (or tied to `token_embedding` when
//!     `tie_word_embeddings == true`; safetensors loader resolves it)

use fuel_core::inference_context::{DecodeSession, DecodeTokenData, InferenceContext, KvCache};
use fuel_core::lazy::{LayerWeights, Tensor, WeightStorage};
use fuel_core::persistent_decode::{
    DecodeBackbone, DecodeDims, DecodeLayerInputs, MaskPlan, PersistentDecodeModel,
};
use fuel_core::{Device, Result};
use fuel_ir::{DType, Shape};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    /// First `max_window_layers` layers use the sliding-window mask
    /// when `use_sliding_window == true`; remaining layers run dense.
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
}

impl Qwen2Config {
    /// `head_dim = hidden_size / num_attention_heads`. Convenience
    /// accessor — every Qwen2 size derives head_dim this way.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Preset for `Qwen/Qwen2-7B`. Field values from the HF config.
    pub fn qwen2_7b() -> Self {
        Self {
            vocab_size: 152_064,
            hidden_size: 3584,
            intermediate_size: 18_944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            max_position_embeddings: 131_072,
            sliding_window: 131_072,
            max_window_layers: 28,
            use_sliding_window: false,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
        }
    }
}

// ROADMAP item 8 (II): config-from-path, as a capability of the config TYPE
// (not a registry service — it survives whatever happens to increment I).
// Convention mirrors the ~30 models that already parse HF config.json (gemma2,
// bert, llama_full, phi, …): a `serde` raw carrying the file's field names plus
// constant defaults, then `resolve` fills the sibling-derived values.
#[derive(Debug, Clone, serde::Deserialize)]
struct Qwen2ConfigRaw {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    max_position_embeddings: usize,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    max_window_layers: Option<usize>,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default = "default_qwen2_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_qwen2_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
}

fn default_qwen2_rope_theta() -> f64 {
    1_000_000.0
}
fn default_qwen2_rms_norm_eps() -> f64 {
    1e-6
}

impl Qwen2ConfigRaw {
    fn from_json_str(json: &str) -> fuel_core::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| fuel_core::Error::Msg(format!("parsing Qwen2 config.json: {e}")))
    }

    fn resolve(self) -> Qwen2Config {
        Qwen2Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: fuel_core::hf_config::num_key_value_heads(
                self.num_key_value_heads,
                self.num_attention_heads,
            ),
            max_position_embeddings: self.max_position_embeddings,
            // absent/null sliding_window: fall back to the full context length.
            sliding_window: self.sliding_window.unwrap_or(self.max_position_embeddings),
            // HF default: all layers windowed, i.e. num_hidden_layers.
            max_window_layers: self.max_window_layers.unwrap_or(self.num_hidden_layers),
            use_sliding_window: self.use_sliding_window,
            rope_theta: self.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            tie_word_embeddings: self.tie_word_embeddings,
        }
    }
}

impl Qwen2Config {
    /// Parse a HuggingFace `config.json` string into a [`Qwen2Config`].
    ///
    /// ROADMAP item 8 (II): reads the artifact rather than returning a preset —
    /// see the born-red `qwen2_config_from_hf_json_parses_the_artifact_not_a_preset`.
    pub fn from_hf_json_str(json: &str) -> fuel_core::Result<Self> {
        Ok(Qwen2ConfigRaw::from_json_str(json)?.resolve())
    }
}

#[derive(Debug, Clone)]
pub struct Qwen2Weights {
    /// Process-unique identity for THIS weight set — the component that lets a
    /// held decode plan tell two same-architecture models apart (GAP-029).
    ///
    /// On the weights rather than on the model for the reason `LlamaWeights`
    /// gives: the weights are what a held graph bakes as `Const`s, so two models
    /// over one weight set may legitimately share a plan while two with distinct
    /// weights must not. Mint with
    /// [`fuel_core::decode_shape::ModelInstanceId::next`].
    pub instance: fuel_core::decode_shape::ModelInstanceId,
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Qwen2Model {
    pub config: Qwen2Config,
    pub weights: Qwen2Weights,
}

impl Qwen2Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        weights
            .output
            .apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size)
    }

    /// Run the encoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection — useful for embedding
    /// adapters (Stella-en-v5, etc.) that swap the causal LM
    /// head for a custom projector or pooler.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Like [`Self::forward_hidden`] but takes pre-computed
    /// `embeds` of shape `(1, seq, hidden_size)` and a
    /// caller-supplied `(1, 1, seq, seq)` additive mask. The
    /// mask is used for ALL layers — Qwen2's per-layer
    /// sliding-window gating is skipped. Both `embeds` and
    /// `attention_mask` MUST live on the same graph (build
    /// the mask via `embeds.const_f32_like(...)`).
    ///
    /// Useful for bidirectional encoder mode (mask is just
    /// the pad-only `(1 - mask[j]) * MIN` broadcast). Pass
    /// `0` for keep and `-inf` (or a large negative) for mask.
    pub fn forward_hidden_embeds_with_mask(
        &self,
        embeds: &Tensor,
        attention_mask: &Tensor,
        start_pos: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "embeds must be rank 3 [b, seq, hidden]");
        let seq = dims[1];
        assert_eq!(dims[2], cfg.hidden_size);
        let head_dim = cfg.head_dim();
        assert_eq!(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            "Qwen2Config: num_attention_heads * head_dim must equal hidden_size",
        );
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads,
            0,
            "Qwen2Config: num_attention_heads must be a multiple of num_key_value_heads",
        );

        let mut h = embeds.clone();
        let (rope_cos, rope_sin) = h.rope_tables_const(cfg.rope_theta, start_pos, seq, head_dim);

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, attention_mask)?;
        }
        h.rms_norm_affine(
            std::sync::Arc::clone(&weights.final_norm_gain),
            cfg.rms_norm_eps,
        )
    }

    /// Shared backbone for the causal-mask paths
    /// (`forward` and `forward_hidden`). Embed → RoPE →
    /// per-layer attn+MLP → final RmsNorm. Builds a
    /// sliding-window or strict-causal mask per layer based
    /// on the config. For non-causal use (bidirectional
    /// encoder mode), see [`Self::forward_hidden_embeds_with_mask`].
    /// Like [`Self::forward_hidden`] but takes pre-computed
    /// `embeds` of shape `(1, seq, hidden_size)` and uses the
    /// standard per-layer (sliding-window / strict-causal)
    /// mask construction. Skips the LM head. Use this from
    /// multimodal hosts that interleave image embeddings into
    /// the text stream (LLaVA-style consumers) and want hidden
    /// states without the lm_head projection.
    pub fn forward_hidden_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        self.run_backbone_embeds(embeds, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let _batch = 1;
        assert!(seq > 0, "Qwen2Model: tokens must be non-empty");

        let h = Tensor::embed_tokens(
            weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
            &Device::cpu(),
        )?;
        self.run_backbone_embeds(&h, start_pos)
    }

    fn run_backbone_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "embeds must be rank 3 [b, seq, hidden]");
        let seq = dims[1];
        assert_eq!(dims[2], cfg.hidden_size);
        let head_dim = cfg.head_dim();
        assert_eq!(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            "Qwen2Config: num_attention_heads * head_dim must equal hidden_size",
        );
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads,
            0,
            "Qwen2Config: num_attention_heads ({}) must be a multiple of num_key_value_heads ({})",
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
        );

        let mut h = embeds.clone();
        let (rope_cos, rope_sin) = h.rope_tables_const(cfg.rope_theta, start_pos, seq, head_dim);

        let causal_window = if cfg.use_sliding_window {
            Some(self.build_layer_mask(&h, seq, true))
        } else {
            None
        };
        let causal_strict = self.build_layer_mask(&h, seq, false);

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let uses_window = cfg.use_sliding_window && layer_idx < cfg.max_window_layers;
            let mask = if uses_window {
                causal_window
                    .as_ref()
                    .expect("windowed mask built when use_sliding_window")
            } else {
                &causal_strict
            };
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, mask)?;
        }
        h.rms_norm_affine(
            std::sync::Arc::clone(&weights.final_norm_gain),
            cfg.rms_norm_eps,
        )
    }

    /// Build the attention mask for one layer. `uses_window == true`
    /// produces the sliding-window causal mask; `false` produces a
    /// strict lower-triangular causal mask.
    fn build_layer_mask(&self, anchor: &Tensor, seq: usize, uses_window: bool) -> Tensor {
        let cfg = &self.config;
        let window = if uses_window {
            cfg.sliding_window
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
        x: &Tensor,
        layer: &LayerWeights,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * head_dim;

        let x_norm = x.rms_norm_affine(
            std::sync::Arc::clone(&layer.attn_norm_gain),
            cfg.rms_norm_eps,
        )?;

        // Q / K / V projections with optional biases (Qwen2 has them).
        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size)?
            .add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer
            .attn_k
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer
            .attn_v
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        // GQA replication.
        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        // Scaled dot-product attention with caller-supplied mask.
        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let _ = seq; // silence unused after refactor; mask already sized for seq.
        let scores_masked = scores_scaled.broadcast_add(mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer
            .attn_o
            .apply_linear(&merged, cfg.hidden_size, cfg.hidden_size)?;

        let h1 = x.add(&attn_out)?;
        let h1_norm = h1.rms_norm_affine(
            std::sync::Arc::clone(&layer.ffn_norm_gain),
            cfg.rms_norm_eps,
        )?;

        let gate = layer
            .ffn_gate
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let up = layer
            .ffn_up
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out =
            layer
                .ffn_down
                .apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size)?;

        h1.add(&ffn_out)
    }
}

// ---- GAP-029 increment 3 · persistent-KV decode -----------------------------
//
// Qwen2 is family 1 of the eleven. It does NOT get its own build path: the
// embed → RoPE → mask → per-layer → norm → logits body lives in
// `fuel_core::persistent_decode` and serves D1 and D2 from one parameterised
// source. What lives here is the part that is genuinely Qwen2: the attention
// block, and the per-layer mask variation the rest of the tree has no analogue
// for.

impl Qwen2Model {
    /// **Per-layer attention variation.** Qwen2 gates on
    /// `use_sliding_window && layer_idx < max_window_layers`
    /// (`run_backbone_embeds`), so a decode port that builds ONE mask and hands
    /// it to every layer is silently wrong for any mixed config — measured at
    /// **7.9e-3** max |logit diff| on the live 2-layer/window-4 config, ~800x
    /// the f32 noise floor.
    ///
    /// Expressed through [`MaskPlan::split_window`], which Qwen3 and Qwen3Moe
    /// share verbatim (same predicate, same fields).
    pub fn decode_mask_plan(&self) -> MaskPlan {
        let cfg = &self.config;
        if cfg.use_sliding_window {
            MaskPlan::split_window(
                cfg.num_hidden_layers,
                cfg.max_window_layers,
                cfg.sliding_window,
            )
        } else {
            MaskPlan::dense(cfg.num_hidden_layers)
        }
    }

    /// Identity a held decode plan for THIS model is baked against: family +
    /// the config values that change graph structure + this weight set.
    ///
    /// `rope_theta` and `sliding_window` are deliberately absent — both are
    /// per-token *data* (RoPE tables and mask bytes are rebound every step), so
    /// baking them would forfeit plan reuse across a change that is already
    /// handled correctly. What the *plan* contributes is structural: how many
    /// mask variants exist and which layer reads which.
    pub fn decode_shape_key(&self) -> u64 {
        let cfg = &self.config;
        let mut h = fuel_core::decode_shape::ShapeKeyHasher::new();
        h.mix_str("qwen2")
            .mix_instance(self.weights.instance)
            .mix_u64(cfg.num_hidden_layers as u64)
            .mix_u64(cfg.num_attention_heads as u64)
            .mix_u64(cfg.num_key_value_heads as u64)
            .mix_u64(cfg.head_dim() as u64)
            .mix_u64(cfg.hidden_size as u64)
            .mix_u64(cfg.intermediate_size as u64)
            .mix_u64(cfg.vocab_size as u64)
            .mix_f64(cfg.rms_norm_eps);
        self.decode_mask_plan().mix_into(&mut h);
        h.finish()
    }

    /// Decode/prefill through a pre-allocated [`KvCache`], rebuilding the graph
    /// each step. The primitive the persistent path falls back to.
    pub fn forward_with_kv_context(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
    ) -> Result<Vec<f32>> {
        fuel_core::persistent_decode::forward_with_kv_context(self, tokens, cache, ctx, false, None)
    }

    /// Plan-once persistent decode: the first `seq == 1` token builds and
    /// optimizes the graph, every later token rebinds data and skips optimize.
    /// `seq != 1` falls back to [`Self::forward_with_kv_context`].
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

    /// [`Self::forward_with_kv_context_persistent`] at the ergonomic call shape
    /// — the session rides in the `InferenceContext`, so a hand-rolled decode
    /// loop gets plan reuse without knowing `DecodeSession` exists. The result
    /// is bound BEFORE the session is put back so an error path cannot silently
    /// downgrade every later token to re-planning.
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

    /// One Qwen2 layer against the pre-allocated KV buffers: the fresh K/V slab
    /// is written at the runtime offset `cached_len`, then attention reads the
    /// **full fixed-capacity** buffers under a `[1, 1, seq, max_seq_len]` mask
    /// that excludes both future positions and the unwritten tail. Nothing in
    /// the graph's shape depends on `cached_len`, which is what makes the decode
    /// step reusable across tokens.
    ///
    /// Same math as [`Self::apply_layer`] — biased Q/K/V, GQA, SwiGLU — with
    /// two deliberate differences from the prefill twin:
    ///
    /// - **GQA is left to `matmul`'s head broadcast** rather than materialised
    ///   with `repeat_interleave`. Replicating K/V here would mean expanding the
    ///   whole `max_seq_len` cache every token, not just this step's row.
    /// - **No flash-decode arm is offered, and that is a correctness decision,
    ///   not caution.** The CUDA flash arm expresses its key range as a single
    ///   `k_len = cached_len + seq`, which cannot represent a sliding window: on
    ///   a windowed layer it would attend to the whole prefix and **silently
    ///   drop the window** on bf16/CUDA, the exact defect this port exists to
    ///   fix — while being invisible to this lane's f32/CPU gate, where the arm
    ///   declines anyway. Offering it needs a windowed `k` range first.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_with_kv_writes(
        &self,
        x: &Tensor,
        layer: &LayerWeights,
        k_cache_const: &Tensor,
        v_cache_const: &Tensor,
        cached_len_sym: fuel_ir::SymId,
        // The live attended prefix (`cached_len + seq`) — the flash arm's
        // `k_len`. Inert on the f32 decode graph, where the arm declines.
        attended_len_sym: fuel_ir::SymId,
        offset: Option<&Tensor>,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        mask: &Tensor,
        // GAP-194: this layer's own window, from the same plan entry `mask`
        // came from — so the arm and the mask cannot disagree.
        attn_window: Option<usize>,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let act_dtype = x.dtype();

        let x_norm = x.rms_norm_affine(Arc::clone(&layer.attn_norm_gain), cfg.rms_norm_eps)?;

        // Q / K / V projections with Qwen2's biases.
        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size)?
            .add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer
            .attn_k
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer
            .attn_v
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let q_h = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k_h = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v_h = v.split_heads(cfg.num_key_value_heads, head_dim)?;
        // RoPE runs in f32 (its build-time requirement); the casts are no-ops
        // when the activation dtype is already f32.
        let q_r = q_h
            .to_dtype(DType::F32)?
            .rope_with_tables(rope_cos, rope_sin)?
            .to_dtype(act_dtype)?;
        let k_r = k_h
            .to_dtype(DType::F32)?
            .rope_with_tables(rope_cos, rope_sin)?
            .to_dtype(act_dtype)?;

        // Write this step's K/V into the pre-allocated buffers at the runtime
        // offset. The returned tensor's Storage Arc IS the cache const's Arc —
        // the write mutates in place and everything downstream reads the
        // post-write buffer.
        let write_ranges = vec![
            (0, batch),
            (0, cfg.num_key_value_heads),
            (0, seq), // axis-2 start is dynamic; width = seq
            (0, head_dim),
        ];
        let (full_k, full_v) = match offset {
            // Device-resident offset (`Op::WriteSliceDoff`, CPU/CUDA): read at
            // kernel launch, so the step is CUDA-graph-capturable.
            Some(off) => (
                k_cache_const.write_slice_doff(&k_r, off, 2, write_ranges.clone())?,
                v_cache_const.write_slice_doff(&v_h, off, 2, write_ranges)?,
            ),
            // Backend-generic `SymEnv` offset (Vulkan). Bit-identical write.
            None => {
                let dyn_off = fuel_ir::DynScalar::Sym(cached_len_sym);
                (
                    k_cache_const.write_slice_dyn(&k_r, write_ranges.clone(), 2, dyn_off)?,
                    v_cache_const.write_slice_dyn(&v_h, write_ranges, 2, dyn_off)?,
                )
            }
        };

        let k_t = full_k.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let scores_masked = scores_scaled.broadcast_add(mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&full_v)?;

        // `merge_heads()` inlined as permute + reshape so `attn_v`'s SOLE
        // consumer (the permute) can be named as the flash arm's reconverge —
        // arm-0 runnability requires the merge to read arm 0. The same split
        // `LlamaModel` makes, for the same reason.
        let attn_v_permuted = attn_v.permute([0, 2, 1, 3_usize])?;
        fuel_core::lazy::offer_flash_decode_arm_for_region(
            q_r.graph(),
            q_r.node_id(),
            full_k.node_id(),
            full_v.node_id(),
            attn_v.node_id(),
            attn_v_permuted.node_id(),
            scale as f32,
            attended_len_sym,
            // GAP-194: this layer's OWN window. A windowed layer is declined by
            // the admissibility gate (`flash_decoding` implements no local
            // attention); a dense layer is eligible. Stating it is what makes
            // the offer honest — asserting `None` here would be the defect.
            attn_window,
            None, // Qwen2 has no attention-logit softcap
            fuel_dispatch::decode_flash::FlashArmCapability::production(),
        )?;
        let merged = attn_v_permuted.reshape(Shape::from_dims(&[
            batch,
            seq,
            cfg.num_attention_heads * head_dim,
        ]))?;
        let attn_out = layer
            .attn_o
            .apply_linear(&merged, cfg.hidden_size, cfg.hidden_size)?;

        let h1 = x.add(&attn_out)?;
        let h1_norm = h1.rms_norm_affine(Arc::clone(&layer.ffn_norm_gain), cfg.rms_norm_eps)?;
        let gate = layer
            .ffn_gate
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let up = layer
            .ffn_up
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out =
            layer
                .ffn_down
                .apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size)?;
        h1.add(&ffn_out)
    }
}

impl PersistentDecodeModel for Qwen2Model {
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
            // Only the device-offset path carries an offset buffer; a SymEnv
            // session gets its offset through `cached_len_sym` instead.
            session.offset_node().is_some().then_some(cached_len),
        )
    }
}

impl DecodeBackbone for Qwen2Model {
    fn decode_family(&self) -> &'static str {
        "Qwen2Model"
    }

    fn decode_dims(&self) -> DecodeDims {
        let cfg = &self.config;
        DecodeDims {
            n_layers: cfg.num_hidden_layers,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
            hidden: cfg.hidden_size,
            vocab: cfg.vocab_size,
            // Full rotary — Qwen2 has no `rotary_dim`/`partial_rotary_factor`.
            rope_width: cfg.head_dim(),
            embed_scale: None,
        }
    }

    fn decode_shape_key(&self) -> u64 {
        Qwen2Model::decode_shape_key(self)
    }

    fn decode_mask_plan(&self) -> MaskPlan {
        Qwen2Model::decode_mask_plan(self)
    }

    fn decode_rope_plan(&self) -> fuel_core::persistent_decode::RopePlan {
        fuel_core::persistent_decode::RopePlan::single(
            self.config.rope_theta,
            self.decode_dims().n_layers,
        )
    }

    fn decode_token_embedding(&self) -> Arc<[f32]> {
        self.weights.token_embedding.clone()
    }

    fn decode_apply_layer(
        &self,
        layer_idx: usize,
        inputs: &DecodeLayerInputs<'_>,
    ) -> Result<Tensor> {
        self.apply_layer_with_kv_writes(
            inputs.x,
            &self.weights.layers[layer_idx],
            inputs.k_cache,
            inputs.v_cache,
            inputs.cached_len_sym,
            inputs.attended_len_sym,
            inputs.offset,
            inputs.rope_cos,
            inputs.rope_sin,
            inputs.mask,
            inputs.attn_window,
        )
    }

    fn decode_final_norm_and_head(&self, h: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let h_norm =
            h.rms_norm_affine(Arc::clone(&self.weights.final_norm_gain), cfg.rms_norm_eps)?;
        self.weights
            .output
            .apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size)
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl Qwen2Weights {
    /// Load Qwen2 weights from HF safetensors (e.g. `Qwen/Qwen2-7B`).
    /// Qwen2 has biases on Q/K/V but NOT on the output projection.
    pub fn load_from_mmapped(
        st: &fuel_core::safetensors::MmapedSafetensors,
        cfg: &Qwen2Config,
    ) -> Result<Self> {
        use fuel_core::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let inter = cfg.intermediate_size;

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);
        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let attn_q = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.q_proj.weight"),
                q_dim,
                h,
            )?;
            let attn_q_bias = Some(Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.self_attn.q_proj.bias"),
            )?));
            let attn_k = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                h,
            )?;
            let attn_k_bias = Some(Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.self_attn.k_proj.bias"),
            )?));
            let attn_v = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.v_proj.weight"),
                kv_dim,
                h,
            )?;
            let attn_v_bias = Some(Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.self_attn.v_proj.bias"),
            )?));
            let attn_o = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.o_proj.weight"),
                h,
                q_dim,
            )?;
            let ffn_gate = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.mlp.gate_proj.weight"),
                inter,
                h,
            )?;
            let ffn_up = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.mlp.up_proj.weight"),
                inter,
                h,
            )?;
            let ffn_down = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.mlp.down_proj.weight"),
                h,
                inter,
            )?;
            let attn_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.input_layernorm.weight"),
            )?);
            let ffn_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.post_attention_layernorm.weight"),
            )?);
            layers.push(LayerWeights {
                attn_q,
                attn_q_bias,
                attn_k,
                attn_k_bias,
                attn_v,
                attn_v_bias,
                attn_o,
                ffn_gate,
                ffn_up,
                ffn_down,
                attn_norm_gain,
                ffn_norm_gain,
            });
        }
        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.norm.weight")?);
        let output = if cfg.tie_word_embeddings {
            crate::models::lazy_llama_full::tied_lm_head_from_embeddings(
                &token_embedding,
                cfg.vocab_size,
                h,
            )
        } else {
            load_transposed_matrix_preserve_dtype(st, "lm_head.weight", cfg.vocab_size, h)?
        };
        Ok(Self {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            final_norm_gain,
            output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ROADMAP item 8 (II). Golden values from Qwen/Qwen2-0.5B's real config.json
    // (huggingface.co/Qwen/Qwen2-0.5B/blob/main/config.json).
    const QWEN2_0_5B_CONFIG_JSON: &str = r#"{
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": 151936,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "max_position_embeddings": 131072,
        "max_window_layers": 24,
        "sliding_window": 131072,
        "use_sliding_window": false,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": true
    }"#;

    #[test]
    fn qwen2_config_from_hf_json_parses_the_artifact_not_a_preset() {
        let cfg = Qwen2Config::from_hf_json_str(QWEN2_0_5B_CONFIG_JSON).unwrap();
        // POSITIVE goldens — Qwen2-0.5B, none coinciding with a resolve default:
        assert_eq!(cfg.hidden_size, 896); // required field, no default
        assert_eq!(cfg.num_hidden_layers, 24); // required
        assert_eq!(cfg.num_attention_heads, 14); // required
        assert_eq!(cfg.vocab_size, 151_936); // required
        assert_eq!(cfg.intermediate_size, 4864); // required
        // GQA: default is num_attention_heads (14); 2 proves the key was READ.
        assert_eq!(cfg.num_key_value_heads, 2);
        // 0.5B ties; the `#[serde(default)]` bool default is false and the 7B preset
        // is false, so `true` proves the key was READ (not defaulted, not the preset).
        assert!(cfg.tie_word_embeddings);
        // Sabotage sibling (WEAKER): not the 7B preset. The `==` goldens above are primary.
        assert_ne!(cfg, Qwen2Config::qwen2_7b());
    }

    /// Retained sabotage sibling: a SECOND distinct config must parse to ITS OWN
    /// values, so a parser that returns a constant/preset fails one of the two.
    /// Also exercises the default path (sliding_window/rope_theta/etc. omitted).
    #[test]
    fn qwen2_config_from_hf_json_reads_a_second_distinct_config() {
        let json = r#"{
            "model_type": "qwen2",
            "vocab_size": 152064,
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "max_position_embeddings": 131072,
            "tie_word_embeddings": false
        }"#;
        let cfg = Qwen2Config::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert!(!cfg.tie_word_embeddings);
        // omitted defaults resolved
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.max_window_layers, 28); // defaulted to num_hidden_layers
        // distinct from the 0.5B parse — a constant parser cannot satisfy both tests
        assert_ne!(cfg.hidden_size, 896);
    }

    /// Retained sabotage sibling: with `num_key_value_heads` ABSENT, GQA defaults
    /// to `num_attention_heads`. Paired with the 0.5B golden (present → 2), this
    /// distinguishes "read the key" from "never looked".
    #[test]
    fn qwen2_config_gqa_defaults_to_num_heads_when_absent() {
        let json = r#"{
            "model_type": "qwen2",
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "max_position_embeddings": 128
        }"#;
        let cfg = Qwen2Config::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.num_key_value_heads, 8); // absent → defaults to num_attention_heads
    }

    fn tiny_weights(cfg: &Qwen2Config) -> Qwen2Weights {
        let mut s: u32 = 7777;
        let next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim();
        let mut next_box: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *next_box);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| LayerWeights {
                attn_q: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
                attn_q_bias: Some(vec_of(h, &mut *next_box)),
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
                attn_k_bias: Some(vec_of(kv, &mut *next_box)),
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
                attn_v_bias: Some(vec_of(kv, &mut *next_box)),
                attn_o: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
                ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *next_box)),
                ffn_up: WeightStorage::F32(vec_of(h * i, &mut *next_box)),
                ffn_down: WeightStorage::F32(vec_of(i * h, &mut *next_box)),
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *next_box));
        Qwen2Weights {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            final_norm_gain,
            output,
        }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = Qwen2Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            sliding_window: 4,
            max_window_layers: 1,
            use_sliding_window: true,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let model = Qwen2Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        assert_eq!(out.len(), tokens.len() * cfg.vocab_size);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// `use_sliding_window = false` and `use_sliding_window = true`
    /// with `max_window_layers = 0` must produce identical outputs
    /// (no layer actually uses the window in either case).
    #[test]
    fn no_window_paths_match() {
        let cfg_a = Qwen2Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            sliding_window: 2,
            max_window_layers: 2,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let mut cfg_b = cfg_a.clone();
        cfg_b.use_sliding_window = true;
        cfg_b.max_window_layers = 0; // every layer is "past" the window cutoff → dense
        let weights = tiny_weights(&cfg_a);
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let out_a = Qwen2Model {
            config: cfg_a,
            weights: weights.clone(),
        }
        .forward(&tokens, 0)
        .unwrap()
        .realize_f32();
        let out_b = Qwen2Model {
            config: cfg_b,
            weights,
        }
        .forward(&tokens, 0)
        .unwrap()
        .realize_f32();
        assert_eq!(out_a, out_b);
    }

    /// `max_window_layers > 0` with a real window MUST diverge from
    /// the all-dense run on sequences longer than the window.
    #[test]
    fn window_layers_diverge_from_dense() {
        let cfg_window = Qwen2Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            sliding_window: 2,
            max_window_layers: 2, // both layers use the window
            use_sliding_window: true,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let mut cfg_dense = cfg_window.clone();
        cfg_dense.use_sliding_window = false;
        let weights = tiny_weights(&cfg_window);
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let out_window = Qwen2Model {
            config: cfg_window.clone(),
            weights: weights.clone(),
        }
        .forward(&tokens, 0)
        .unwrap()
        .realize_f32();
        let _ = cfg_window;
        let out_dense = Qwen2Model {
            config: cfg_dense,
            weights,
        }
        .forward(&tokens, 0)
        .unwrap()
        .realize_f32();
        let any_diff = out_window
            .iter()
            .zip(out_dense.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(any_diff, "sliding window should diverge from dense run");
    }

    /// Q/K/V biases must be honored. Compare a run with all-zero
    /// biases against one with random biases — outputs must differ.
    #[test]
    fn qkv_biases_affect_output() {
        let cfg = Qwen2Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            sliding_window: 32,
            max_window_layers: 0,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let mut wt_zero = tiny_weights(&cfg);
        let zero_h: Arc<[f32]> = Arc::from(vec![0.0_f32; cfg.hidden_size]);
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim();
        let zero_kv: Arc<[f32]> = Arc::from(vec![0.0_f32; kv_dim]);
        for l in &mut wt_zero.layers {
            l.attn_q_bias = Some(zero_h.clone());
            l.attn_k_bias = Some(zero_kv.clone());
            l.attn_v_bias = Some(zero_kv.clone());
        }
        let wt_random = tiny_weights(&cfg);
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let out_zero = Qwen2Model {
            config: cfg.clone(),
            weights: wt_zero,
        }
        .forward(&tokens, 0)
        .unwrap()
        .realize_f32();
        let out_random = Qwen2Model {
            config: cfg,
            weights: wt_random,
        }
        .forward(&tokens, 0)
        .unwrap()
        .realize_f32();
        let any_diff = out_zero
            .iter()
            .zip(out_random.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(any_diff, "non-zero Q/K/V biases must change output");
    }

    /// `forward_hidden_embeds_with_mask` accepts pre-computed
    /// embeds plus a caller-supplied `(1, 1, seq, seq)`
    /// additive mask. A bidirectional pad mask (all zeros)
    /// produces different hidden states than the strict-causal
    /// `forward_hidden` because the bidirectional path lets
    /// earlier tokens attend to later ones too.
    #[test]
    fn forward_hidden_embeds_with_bidirectional_mask() {
        let cfg = Qwen2Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            sliding_window: 32,
            max_window_layers: 0,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let model = Qwen2Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let h_causal = model.forward_hidden(&tokens, 0).unwrap().realize_f32();

        // Build embeds externally and the bidirectional mask
        // anchored on the same graph as embeds.
        let embed_table = Tensor::from_f32(
            model.weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids =
            embed_table.const_u32_like(tokens.clone(), Shape::from_dims(&[tokens.len()]));
        let embeds = embed_table
            .index_select(0_usize, &token_ids)
            .unwrap()
            .reshape(Shape::from_dims(&[1, tokens.len(), cfg.hidden_size]))
            .unwrap();
        let zero_mask: Arc<[f32]> = Arc::from(vec![0.0_f32; tokens.len() * tokens.len()]);
        let mask = embeds.const_f32_like(
            zero_mask,
            Shape::from_dims(&[1, 1, tokens.len(), tokens.len()]),
        );
        let h_bidir = model
            .forward_hidden_embeds_with_mask(&embeds, &mask, 0)
            .unwrap()
            .realize_f32();
        assert_eq!(h_causal.len(), h_bidir.len());
        let mut max_diff = 0.0_f32;
        for (x, y) in h_causal.iter().zip(h_bidir.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(
            max_diff > 1e-7,
            "bidirectional hidden states must differ from causal, max_diff = {max_diff}"
        );
        for &v in &h_bidir {
            assert!(v.is_finite(), "non-finite bidirectional hidden: {v}");
        }
    }

    /// `forward_hidden_embeds(embeds, start_pos)` must produce
    /// the same hidden states as `forward_hidden(tokens, start_pos)`
    /// when the embeds are built from the token-embedding table —
    /// proves the embed-lookup is the only difference and the
    /// per-layer mask construction matches the tokens path.
    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = Qwen2Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            sliding_window: 32,
            max_window_layers: 0,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let model = Qwen2Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let from_tokens = model.forward_hidden(&tokens, 0).unwrap().realize_f32();

        let embed_table = Tensor::from_f32(
            model.weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids =
            embed_table.const_u32_like(tokens.clone(), Shape::from_dims(&[tokens.len()]));
        let embeds = embed_table
            .index_select(0_usize, &token_ids)
            .unwrap()
            .reshape(Shape::from_dims(&[1, tokens.len(), cfg.hidden_size]))
            .unwrap();
        let from_embeds = model
            .forward_hidden_embeds(&embeds, 0)
            .unwrap()
            .realize_f32();
        assert_eq!(from_tokens.len(), from_embeds.len());
        for (a, b) in from_tokens.iter().zip(from_embeds.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "forward_hidden_embeds must match forward_hidden: {a} vs {b}"
            );
        }
    }

    /// The config every windowed-decode design rests on: 2 layers, window 4,
    /// `max_window_layers: 1` — layer 0 windowed, layer 1 dense.
    fn mixed_window_cfg() -> Qwen2Config {
        Qwen2Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            sliding_window: 4,
            max_window_layers: 1,
            use_sliding_window: true,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        }
    }

    /// Logits from the SHIPPED per-layer gating (`forward`), and logits from a
    /// SINGLE mask applied to every layer — which is exactly what a one-mask
    /// decode port computes. `window` selects the single mask's width, so the
    /// same helper serves both the discriminating case and its control.
    fn per_layer_vs_single_mask(
        cfg: &Qwen2Config,
        tokens: &[u32],
        single_window: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let model = Qwen2Model {
            config: cfg.clone(),
            weights: tiny_weights(cfg),
        };
        let per_layer = model.forward(tokens, 0).unwrap().realize_f32();

        let seq = tokens.len();
        let embeds = Tensor::embed_tokens(
            model.weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
            &Device::cpu(),
        )
        .unwrap();
        // Same predicate as `build_layer_mask`, so the only thing that varies
        // between the two paths is WHICH layers see WHICH width.
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i || j + single_window <= i {
                    mask_data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = embeds.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]));
        let hidden = model
            .forward_hidden_embeds_with_mask(&embeds, &mask, 0)
            .unwrap();
        let single = model
            .weights
            .output
            .apply_linear(&hidden, cfg.hidden_size, cfg.vocab_size)
            .unwrap()
            .realize_f32();
        (per_layer, single)
    }

    // ---- GAP-029 increment 3, family 1: persistent decode ------------------

    /// Prefill `tokens[..prefill]`, then decode the rest ONE token at a time
    /// through the persistent path, returning each decode step's logits.
    ///
    /// **The `>= 3` decode steps are load-bearing, not padding.** One decode
    /// token exercises only the held-graph BUILD path; the per-token REBIND path
    /// — where the mask bytes for the new position are recomputed and written
    /// into the held Const — is first reached on step 2. A 1-step test passes on
    /// a model that is wrong from token 2 onward. The assert lives here rather
    /// than in a caller so a later test cannot quietly weaken it.
    fn decode_steps(model: &Qwen2Model, tokens: &[u32], prefill: usize) -> Vec<Vec<f32>> {
        let cfg = &model.config;
        let n_decode = tokens.len() - prefill;
        assert!(
            n_decode >= 3,
            "decode_steps: need >= 3 decode tokens to reach the rebind path \
             (got {n_decode}); a 1-token test only walks the build path",
        );
        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
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
        for i in prefill..tokens.len() {
            out.push(
                model
                    .forward_with_kv_context_persistent(
                        &tokens[i..=i],
                        &mut cache,
                        &mut ctx,
                        &mut session,
                    )
                    .expect("decode"),
            );
            assert!(session.is_some(), "decode must hold a session from token 1");
        }
        assert_eq!(
            cache.cached_len,
            tokens.len(),
            "cache must advance every step"
        );
        out
    }

    /// Max |logit diff| between each decode step and the **shipped, per-layer
    /// gated** non-cached forward at the same absolute position.
    ///
    /// `forward` is the oracle because it is the path that already honours
    /// `use_sliding_window && layer_idx < max_window_layers`; decode agreeing
    /// with it is the whole claim.
    fn decode_vs_forward_max_abs(cfg: &Qwen2Config, tokens: &[u32], prefill: usize) -> Vec<f32> {
        let model = Qwen2Model {
            config: cfg.clone(),
            weights: tiny_weights(cfg),
        };
        let steps = decode_steps(&model, tokens, prefill);
        steps
            .iter()
            .enumerate()
            .map(|(k, got)| {
                let pos = prefill + k;
                let full = model.forward(&tokens[..=pos], 0).unwrap().realize_f32();
                let expected = &full[pos * cfg.vocab_size..(pos + 1) * cfg.vocab_size];
                assert_eq!(got.len(), expected.len());
                got.iter()
                    .zip(expected.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max)
            })
            .collect()
    }

    /// The oracle's threshold. **Not inherited — measured.**
    ///
    /// The obvious template, `forward_with_kv_context_decode_matches_non_cached_
    /// forward` (`lazy.rs`), asserts `diff < 5e-3 || rel < 1e-2`. GAP-029's
    /// measured single-mask divergence is **7.9e-3**: only 1.6x that abs bound,
    /// and the `||` means the `rel` arm alone can pass it outright. **A Qwen2
    /// decode test written on that template would go GREEN on a single-mask
    /// port** — a vacuous oracle arriving through the tolerance rather than
    /// through the assertion target.
    ///
    /// Measured separation on this config (prefill 3, decode 3):
    ///
    /// ```text
    /// correct windowed decode vs forward : 0.0, 0.0, 0.0     (bit-exact)
    /// single-mask decode      vs forward : 0.0, 7.04e-3, 7.95e-3
    /// ```
    ///
    /// So the margin is not a factor of two — it is total. `1e-5` sits ~790x
    /// below the divergence while leaving headroom for a legitimate future
    /// reassociation (a fused kernel, a different reduction order). Asserting
    /// the measured `0.0` would be tighter but would fire on non-defects, and a
    /// test that fires on non-defects is a test somebody weakens.
    const DECODE_ORACLE_ABS: f32 = 1e-5;

    /// ⚠️ NON-DISCRIMINATION CONTROL — RUN THIS FIRST WHEN READING.
    ///
    /// The two paths under comparison are DIFFERENT FUNCTIONS
    /// (`run_backbone_embeds` vs `forward_hidden_embeds_with_mask`), so a
    /// logits difference between them is NOT self-evidently caused by masking:
    /// it could come from any unrelated divergence in those bodies. This pins
    /// that down. With `max_window_layers: 0` no layer uses the window, so the
    /// per-layer path applies the strict-causal mask everywhere — which is
    /// exactly the single mask supplied here — and the two MUST agree.
    ///
    /// If this test ever fails, the sibling below proves NOTHING: the
    /// instrument would be measuring path divergence, not window divergence.
    #[test]
    fn control_the_two_paths_agree_when_no_layer_uses_the_window() {
        let cfg = Qwen2Config {
            max_window_layers: 0,
            ..mixed_window_cfg()
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        // `seq + 1` is `build_layer_mask`'s own strict-causal width.
        let (per_layer, single) = per_layer_vs_single_mask(&cfg, &tokens, tokens.len() + 1);
        assert_eq!(per_layer.len(), single.len());
        for (i, (a, b)) in per_layer.iter().zip(single.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "control: the two paths must agree when no layer is windowed, but logit {i} is {a} vs {b}. The sibling test is then measuring PATH divergence, not WINDOW divergence.",
            );
        }
    }

    /// GAP-029: converts the row's central claim from INFERENCE to MEASUREMENT.
    ///
    /// The row records *"a single-mask decode port is silently wrong for
    /// Qwen2/Qwen3/Qwen3Moe"* as an inference from the mask's SHAPE — the
    /// existing mask tests compare BYTES and no forward pass had been run. This
    /// asserts it at the LOGITS level (never sampled tokens, which are a vacuous
    /// oracle for a tiny model) on the live mixed config.
    ///
    /// `seq = 6 > sliding_window = 4`, deliberately: at `seq <= window` the two
    /// masks are byte-identical and this test would pass vacuously — the same
    /// non-discrimination trap `window_wider_than_capacity_is_byte_identical_to_
    /// the_dense_mask` records for the mask primitive.
    ///
    /// A FAILURE HERE IS A REAL FINDING, not a broken test: it would mean
    /// per-layer window gating does not change this model's output, and the
    /// N=2-variant decode machinery would need re-examining before being built.
    #[test]
    fn single_mask_diverges_from_per_layer_gating_at_the_logits_level() {
        let cfg = mixed_window_cfg();
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        assert!(
            tokens.len() > cfg.sliding_window,
            "non-vacuity: the window must actually bite",
        );
        let (per_layer, single) = per_layer_vs_single_mask(&cfg, &tokens, tokens.len() + 1);
        assert_eq!(per_layer.len(), single.len());

        // Far above f32 noise: the measured divergence is ~8e-3, ~800x this.
        const THRESHOLD: f32 = 1e-5;
        let max_abs = per_layer
            .iter()
            .zip(single.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs > THRESHOLD,
            "a SINGLE mask reproduced the per-layer-gated logits to within {max_abs} (threshold {THRESHOLD:e}). The GAP-029 inference that a one-mask decode port is silently wrong for this family would then be FALSE, and the N=2 mask machinery must be re-examined before it is built.",
        );
    }

    /// Node count of the held decode graph on the MIXED (two-variant) config —
    /// see [`qwen2_held_decode_graph_has_not_grown`]. Measured, not predicted.
    const QWEN2_DECODE_GRAPH_NODES: usize = 150;

    fn gap029_qwen2_decode_graph_nodes() -> usize {
        let cfg = mixed_window_cfg();
        let model = Qwen2Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
            6,
            DType::F32,
            &dev,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(dev);
        let mut session: Option<DecodeSession> = None;
        model
            .forward_with_kv_context_persistent(&[1, 2, 3], &mut cache, &mut ctx, &mut session)
            .expect("prefill");
        model
            .forward_with_kv_context_persistent(&[4], &mut cache, &mut ctx, &mut session)
            .expect("decode");
        session
            .expect("session built on the first decode token")
            .graph_node_count()
    }

    /// **STRUCTURAL baseline for the TWO-VARIANT case**, captured 2026-08-13
    /// before the Gemma3 seam work. The Llama/Phi3 siblings pin `n == 1`; this
    /// pins that a genuinely windowed family's graph does not grow either when
    /// the RoPE-variant machinery arrives (Qwen2 has one RoPE base, so its
    /// count must be untouched by it — only its MASK is multi-variant).
    ///
    /// A logits golden cannot see node growth. This can.
    #[test]
    fn qwen2_held_decode_graph_has_not_grown() {
        assert_eq!(
            gap029_qwen2_decode_graph_nodes(),
            QWEN2_DECODE_GRAPH_NODES,
            "Qwen2's held decode graph changed size",
        );
    }

    /// ⚠️ **NON-DISCRIMINATION CONTROL for the windowed decode test below —
    /// and it is what makes that test's red mean anything.**
    ///
    /// With `max_window_layers: 0` no layer uses the window, so the mask plan
    /// collapses to a single dense variant and persistent decode must reproduce
    /// the non-cached forward. **This passes under BOTH a correct windowed mask
    /// builder and one that ignores the window entirely** — so it certifies that
    /// the decode seam, the KV writes, the held graph and the per-token rebind
    /// are sound, and it must NOT be read as evidence that windowing works.
    ///
    /// Its job is to separate two explanations of the sibling's failure: "the
    /// mask is wrong" versus "Qwen2 decode is simply broken". Without it, a red
    /// there is uninterpretable.
    #[test]
    fn qwen2_decode_matches_forward_when_no_layer_is_windowed() {
        let cfg = Qwen2Config {
            max_window_layers: 0,
            ..mixed_window_cfg()
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        for (k, diff) in decode_vs_forward_max_abs(&cfg, &tokens, 3)
            .iter()
            .enumerate()
        {
            assert!(
                *diff < DECODE_ORACLE_ABS,
                "unwindowed decode step {k} (absolute position {}) diverged from the \
                 non-cached forward by {diff} (limit {DECODE_ORACLE_ABS:e}). This is the \
                 CONTROL: the seam itself is broken, and the windowed test below \
                 proves nothing until this is green.",
                3 + k,
            );
        }
    }

    /// **GAP-029 family 1 — the measurement the whole increment exists for.**
    ///
    /// Qwen2's shipped prefill gates per layer (`use_sliding_window &&
    /// layer_idx < max_window_layers`). Persistent decode must agree with it
    /// position-for-position, or the model silently changes behaviour at the
    /// prefill→decode boundary.
    ///
    /// **Born red, observed rather than asserted.** The mask builder first
    /// returned the DENSE mask for every variant — byte-identically what a
    /// single-mask decode port computes — giving `[0.0, 7.04e-3, 7.95e-3]`
    /// against this oracle, i.e. failing on both steps where the window bites.
    /// Making [`fuel_core::lazy::build_decode_causal_mask_windowed`] the windowed
    /// variant's source took it to `[0.0, 0.0, 0.0]`.
    ///
    /// **The zeros are not vacuous, and the shape of the result is the proof.**
    /// Absolute position 3 agrees under BOTH bodies, because a window of 4
    /// cannot exclude anything until position 4 — so a degenerate oracle would
    /// have shown three zeros in the red run too, and it showed one. The
    /// divergence appears exactly where the predicate says it must.
    ///
    /// **Both failing steps are REBIND steps** (the session is built on decode
    /// step 0, at position 3). The defect therefore could not have been seen by
    /// a single-decode-token test, which walks only the held-graph build path.
    #[test]
    fn qwen2_windowed_decode_matches_per_layer_gated_forward() {
        let cfg = mixed_window_cfg();
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        assert!(
            tokens.len() > cfg.sliding_window,
            "non-vacuity: the sequence must outrun the window or both masks are \
             byte-identical and this test asserts nothing",
        );
        let diffs = decode_vs_forward_max_abs(&cfg, &tokens, 3);
        // The window first bites at absolute position 4, so this is where a
        // single-mask port and a windowed one must part company. Stated as an
        // assertion rather than a comment: if the config ever changes so that no
        // tested step is windowed, this test silently stops discriminating.
        assert!(
            3 + diffs.len() > cfg.sliding_window,
            "non-vacuity: no decoded position is far enough in for the window to \
             exclude anything",
        );
        for (k, diff) in diffs.iter().enumerate() {
            assert!(
                *diff < DECODE_ORACLE_ABS,
                "windowed decode step {k} (absolute position {}) diverged from the \
                 per-layer-gated forward by {diff} (limit {DECODE_ORACLE_ABS:e}). \
                 A single mask applied to every layer produces exactly this \
                 signature — see `MaskPlan` and `build_decode_causal_mask_windowed`.",
                3 + k,
            );
        }
    }

    /// The persistent path must not be the only one that is right: `seq != 1`
    /// falls back to D1, and D1 builds the same stacked mask from the same plan.
    /// A windowed family whose prefill-through-the-cache disagreed with its own
    /// non-cached forward would corrupt the KV state every later decode reads.
    #[test]
    fn qwen2_windowed_multi_token_prefill_through_the_cache_matches_forward() {
        let cfg = mixed_window_cfg();
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let model = Qwen2Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };

        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
            tokens.len(),
            DType::F32,
            &dev,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(dev);
        // One 6-token D1 pass — `seq > window`, so the window genuinely bites
        // inside a single call rather than across steps.
        let got = model
            .forward_with_kv_context(&tokens, &mut cache, &mut ctx)
            .expect("prefill through the cache");

        let full = model.forward(&tokens, 0).unwrap().realize_f32();
        let last = tokens.len() - 1;
        let expected = &full[last * cfg.vocab_size..(last + 1) * cfg.vocab_size];
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < DECODE_ORACLE_ABS,
                "D1 prefill logit[{i}]: cached={a}, non-cached={b}",
            );
        }
    }
}

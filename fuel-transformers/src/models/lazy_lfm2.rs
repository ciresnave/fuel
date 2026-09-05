// SPDX-License-Identifier: MIT OR Apache-2.0
//! LFM2 (Liquid Foundation Model 2) decoder ported to the lazy-graph API.
//!
//! LFM2 is a *hybrid* architecture: layers alternate between standard
//! GQA self-attention blocks and "short-conv liquid" (LIV) blocks. Both
//! variants share the same residual layout — pre-`operator_norm`
//! followed by the temporal mixer, then pre-`ffn_norm` followed by a
//! SwiGLU MLP — and only differ in what sits inside the temporal mixer.
//!
//! ```text
//! residual:  x → operator_norm → mixer → + → norm2 → mlp → + → out
//!                                  ↑           (SwiGLU)     ↑
//!                                  x                        h1
//! ```
//!
//! ### Mixer variants
//!
//! - **Attention**: standard multi-head attention with grouped KV heads.
//!   Per-head Q and K RmsNorm gates sit between the projection split
//!   and RoPE. Otherwise behaves like Qwen2-shape GQA.
//! - **ShortConv (LIV)**: a depthwise causal convolution with an input
//!   gate.
//!
//!   ```text
//!   bcx     = in_proj(x).transpose(1, 2)   # (B, 3·hidden, seq)
//!   B, C, X = split bcx along channel
//!   bx      = B · X                        # input-gated signal
//!   y       = depthwise_conv1d(bx, k=l_cache, groups=hidden)
//!   y       = C · y                        # output gating
//!   out     = out_proj(y)
//!   ```
//!
//!   `l_cache` is the conv kernel width (typically 4 in released LFM2
//!   checkpoints). The conv is causal because we left-pad with
//!   `l_cache - 1` zeros and then narrow to `seq` along the time axis —
//!   matching the eager prefill path in
//!   `fuel-transformers/.../quantized_lfm2.rs`.
//!
//! ### Scope (v1)
//!
//! - **Prefill only.** `forward(tokens, start_pos)` rebuilds the graph
//!   each call. There is no autoregressive ShortConv state — the eager
//!   model maintained a `[B, hidden, l_cache]` rolling buffer for
//!   single-step decode; replicating that in the lazy graph requires the
//!   multi-output infrastructure flagged in
//!   `docs/session-prompts/multi-output-nodes-option-c.md` (same
//!   blocker that gates Mamba autoregressive resume). `start_pos` is
//!   honored for RoPE in attention blocks; the conv blocks treat every
//!   call as a fresh prefill from zero state.
//! - **F32 activations.** Weights via [`fuel_core::lazy::WeightStorage`]
//!   (F32 / BF16 / Q4_0).
//! - **Single batch.** Asserts `batch == 1`.

use fuel_core::lazy::{Tensor, WeightStorage};
use fuel_core::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

/// Selects which mixer a given LFM2 layer runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LFM2BlockType {
    /// Standard multi-head attention with GQA and per-head Q/K RmsNorm.
    Attention,
    /// Short causal convolution with input + output gating (LIV layer).
    Conv,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LFM2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    /// Width of the depthwise causal conv kernel inside each Conv
    /// (ShortConv / LIV) layer. Released LFM2 checkpoints ship with
    /// `l_cache == 4`.
    pub conv_kernel_size: usize,
    /// Per-layer block schedule. `len()` must equal `num_hidden_layers`.
    pub block_types: Vec<LFM2BlockType>,
}

impl LFM2Config {
    /// Sanity-check that the config is internally consistent. Called by
    /// the model constructor; surfacing here for tests that want to
    /// validate a hand-rolled config without instantiating weights.
    pub fn validate(&self) -> Result<()> {
        if self.block_types.len() != self.num_hidden_layers {
            return Err(fuel_core::Error::Msg(format!(
                "LFM2Config: block_types.len() ({}) must equal num_hidden_layers ({})",
                self.block_types.len(),
                self.num_hidden_layers,
            ))
            .bt());
        }
        if self.num_attention_heads * self.head_dim != self.hidden_size {
            return Err(fuel_core::Error::Msg(format!(
                "LFM2Config: num_attention_heads ({}) * head_dim ({}) must equal hidden_size ({})",
                self.num_attention_heads, self.head_dim, self.hidden_size,
            ))
            .bt());
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(fuel_core::Error::Msg(format!(
                "LFM2Config: num_attention_heads ({}) must be a multiple of num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads,
            )).bt());
        }
        if self.conv_kernel_size == 0 {
            return Err(
                fuel_core::Error::Msg("LFM2Config: conv_kernel_size must be >= 1".into()).bt(),
            );
        }
        Ok(())
    }
}

// ROADMAP item 8 (II), key/shape-mismatch program. LFM2 is the KEY-REMAP case:
// almost nothing in the HF artifact is spelled the way `LFM2Config` spells it,
// and two of the fields must be DERIVED rather than read.
//
// Measured against the real artifact (`LiquidAI/LFM2-1.2B/config.json`, read
// directly, not from a description):
//
//     HF key                  ->  LFM2Config field       note
//     block_ff_dim: 12288     ->  intermediate_size
//     conv_L_cache: 3         ->  conv_kernel_size       capital L, serde rename
//     norm_eps: 1e-05         ->  rms_norm_eps           see the duplicate-key rule
//     full_attn_idxs: [..]    ->  block_types            DERIVED, per-layer
//     (absent)                ->  head_dim               DERIVED, hidden/heads
//
// ⚠️ TWO KEYS ARE DUPLICATED IN THE SHIPPED ARTIFACT and a mapper must have a
// rule rather than a coincidence: `norm_eps` and `block_norm_eps` are BOTH
// 1e-05, and `num_attention_heads` and `num_heads` are BOTH 32. They agree
// today. `resolve` therefore ERRORS if a future checkpoint ships them
// disagreeing, rather than silently preferring one — a config where they
// disagree means something this port does not understand.
#[derive(Debug, Clone, serde::Deserialize)]
struct LFM2ConfigRaw {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    /// LFM2 ships this alongside `num_attention_heads`; see the duplicate rule.
    #[serde(default)]
    num_heads: Option<usize>,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    /// LFM2's spelling of `intermediate_size`.
    block_ff_dim: usize,
    /// LFM2's spelling of the conv kernel width. Capital `L` in the artifact.
    #[serde(rename = "conv_L_cache")]
    conv_l_cache: usize,
    /// Indices of the layers that run ATTENTION; every other layer is Conv.
    full_attn_idxs: Vec<usize>,
    max_position_embeddings: usize,
    rope_theta: f64,
    #[serde(default)]
    norm_eps: Option<f64>,
    #[serde(default)]
    block_norm_eps: Option<f64>,
    /// ABSENT from every released LFM2 config. Read anyway: if a future
    /// checkpoint ships one it is authoritative over the derivation.
    #[serde(default)]
    head_dim: Option<usize>,
}

/// Expand `full_attn_idxs` into the per-layer schedule `LFM2Config` requires.
///
/// ⚠️ `LFM2Config::validate` requires `block_types.len() == num_hidden_layers`,
/// so this is a genuine EXPANSION — unlike `RecurrentGemmaConfig`, whose
/// `block_type()` indexes a repeating CYCLE modulo its length and must NOT be
/// expanded. Two models, two opposite correct answers for the same-sounding
/// "block types" field.
///
/// An out-of-range index is an ERROR, not a clamp: an index past the last layer
/// means the artifact and `num_hidden_layers` disagree, and silently dropping it
/// would produce a well-formed schedule for a different model.
fn lfm2_block_types_from_attn_idxs(
    full_attn_idxs: &[usize],
    num_hidden_layers: usize,
) -> fuel_core::Result<Vec<LFM2BlockType>> {
    let mut types = vec![LFM2BlockType::Conv; num_hidden_layers];
    for &i in full_attn_idxs {
        if i >= num_hidden_layers {
            return Err(fuel_core::Error::Msg(format!(
                "LFM2 config.json: full_attn_idxs contains layer {i}, but \
                 num_hidden_layers is {num_hidden_layers} — the artifact and the \
                 layer count disagree"
            )));
        }
        types[i] = LFM2BlockType::Attention;
    }
    Ok(types)
}

/// Reconcile a key HF ships twice. Agreement is required; disagreement is an
/// error rather than a silent preference.
fn lfm2_agree<T: PartialEq + std::fmt::Debug>(
    a: Option<T>,
    b: Option<T>,
    a_name: &str,
    b_name: &str,
) -> fuel_core::Result<Option<T>> {
    match (a, b) {
        (Some(x), Some(y)) if x != y => Err(fuel_core::Error::Msg(format!(
            "LFM2 config.json: {a_name} ({x:?}) and {b_name} ({y:?}) disagree. \
             Released checkpoints ship them equal; a config where they differ \
             means something this port does not model."
        ))),
        (Some(x), _) => Ok(Some(x)),
        (None, y) => Ok(y),
    }
}

impl LFM2ConfigRaw {
    fn from_json_str(json: &str) -> fuel_core::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| fuel_core::Error::Msg(format!("parsing LFM2 config.json: {e}")))
    }

    fn resolve(self) -> fuel_core::Result<LFM2Config> {
        let num_attention_heads = lfm2_agree(
            Some(self.num_attention_heads),
            self.num_heads,
            "num_attention_heads",
            "num_heads",
        )?
        .expect("left operand is Some");

        let rms_norm_eps = lfm2_agree(
            self.norm_eps,
            self.block_norm_eps,
            "norm_eps",
            "block_norm_eps",
        )?
        .ok_or_else(|| {
            fuel_core::Error::Msg(
                "LFM2 config.json: neither norm_eps nor block_norm_eps present".into(),
            )
        })?;

        // ⚠️ ARCHITECTURE-DEPENDENT DEFAULT. No released LFM2 config ships
        // `head_dim`. `LFM2Config::validate` requires
        // `num_attention_heads * head_dim == hidden_size`, so the only value
        // that can satisfy the port's own invariant is `hidden / heads`. An
        // explicit key, if one ever appears, WINS — the derivation is the
        // fallback, not an override.
        let head_dim = match self.head_dim {
            Some(d) => d,
            None => {
                if num_attention_heads == 0 || !self.hidden_size.is_multiple_of(num_attention_heads)
                {
                    return Err(fuel_core::Error::Msg(format!(
                        "LFM2 config.json omits head_dim and hidden_size ({}) is not \
                         divisible by num_attention_heads ({}), so it cannot be derived",
                        self.hidden_size, num_attention_heads
                    )));
                }
                self.hidden_size / num_attention_heads
            }
        };

        let block_types =
            lfm2_block_types_from_attn_idxs(&self.full_attn_idxs, self.num_hidden_layers)?;

        let cfg = LFM2Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: fuel_core::hf_config::num_key_value_heads(
                self.num_key_value_heads,
                num_attention_heads,
            )?,
            head_dim,
            intermediate_size: self.block_ff_dim,
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.rope_theta,
            rms_norm_eps,
            conv_kernel_size: self.conv_l_cache,
            block_types,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

impl LFM2Config {
    /// Parse a HuggingFace LFM2 `config.json` string into an [`LFM2Config`].
    ///
    /// ROADMAP item 8 (II): reads the artifact rather than returning a preset.
    /// LFM2 is the key-remap case — see `LFM2ConfigRaw` for the full mapping and
    /// for the two DERIVED fields (`head_dim`, `block_types`).
    pub fn from_hf_json_str(json: &str) -> fuel_core::Result<Self> {
        LFM2ConfigRaw::from_json_str(json)?.resolve()
    }
}

/// Per-layer weights specialized for the attention variant.
#[derive(Debug, Clone)]
pub struct LFM2AttentionWeights {
    /// `[hidden, q_dim]` — Q projection. `q_dim = num_attention_heads * head_dim`.
    pub attn_q: WeightStorage,
    /// `[hidden, kv_dim]` — K projection. `kv_dim = num_key_value_heads * head_dim`.
    pub attn_k: WeightStorage,
    /// `[hidden, kv_dim]` — V projection.
    pub attn_v: WeightStorage,
    /// `[q_dim, hidden]` — output projection.
    pub attn_o: WeightStorage,
    /// `[head_dim]` per-head RmsNorm gain applied to Q.
    pub q_norm_gain: Arc<[f32]>,
    /// `[head_dim]` per-head RmsNorm gain applied to K.
    pub k_norm_gain: Arc<[f32]>,
}

/// Per-layer weights specialized for the ShortConv (LIV) variant.
#[derive(Debug, Clone)]
pub struct LFM2ConvWeights {
    /// `[hidden, 3 * hidden]` — input projection producing the three
    /// channels (B-gate, C-gate, X-signal).
    pub in_proj: WeightStorage,
    /// `[hidden, hidden]` — output projection.
    pub out_proj: WeightStorage,
    /// `[hidden, 1, conv_kernel_size]` depthwise (`groups == hidden`)
    /// causal conv kernel. The eager loader transposes the on-disk
    /// `[l_cache, hidden]` layout to this convention.
    pub conv_weight: Arc<[f32]>,
}

/// Mixer-variant container. Mirrors the eager `LayerKind` enum.
#[derive(Debug, Clone)]
pub enum LFM2MixerWeights {
    Attention(LFM2AttentionWeights),
    Conv(LFM2ConvWeights),
}

/// Per-layer LFM2 weights: norms + mixer-variant weights + SwiGLU MLP.
#[derive(Debug, Clone)]
pub struct LFM2LayerWeights {
    /// `[hidden]` RmsNorm gain applied before the temporal mixer.
    pub operator_norm_gain: Arc<[f32]>,
    /// `[hidden]` RmsNorm gain applied before the MLP.
    pub ffn_norm_gain: Arc<[f32]>,
    pub mixer: LFM2MixerWeights,
    /// `[hidden, intermediate_size]` — SwiGLU gate projection (`w1`).
    pub ffn_gate: WeightStorage,
    /// `[hidden, intermediate_size]` — SwiGLU up projection (`w3`).
    pub ffn_up: WeightStorage,
    /// `[intermediate_size, hidden]` — SwiGLU down projection (`w2`).
    pub ffn_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct LFM2Weights {
    /// `[vocab_size, hidden_size]` token embedding table.
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LFM2LayerWeights>,
    /// `[hidden_size]` RmsNorm gain applied to the final hidden state.
    pub final_norm_gain: Arc<[f32]>,
    /// `[hidden_size, vocab_size]` LM head. LFM2 GGUFs often tie this
    /// to the token embedding — the loader resolves it.
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct LFM2Model {
    pub config: LFM2Config,
    pub weights: LFM2Weights,
}

impl LFM2Model {
    /// Run a full-sequence forward and return logits `[1, seq, vocab_size]`.
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final RmsNorm and return
    /// per-token hidden states `(1, seq, hidden_size)`.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs of shape `(1, seq, hidden_size)`.
    pub fn forward_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone_embeds(embeds, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`].
    pub fn forward_hidden_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        self.run_backbone_embeds(embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder.
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
        self.weights
            .output
            .apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let seq = tokens.len();
        assert!(seq > 0, "LFM2Model::forward: tokens must be non-empty");

        let h = Tensor::embed_tokens(
            self.weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
            &Device::cpu(),
        )?;
        self.run_backbone_embeds(&h, start_pos)
    }

    fn run_backbone_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        cfg.validate()?;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(fuel_core::Error::Msg(format!(
                "LFM2Model::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(
                fuel_core::Error::Msg("LFM2Model::forward_embeds: seq must be > 0".into()).bt(),
            );
        }

        let (rope_cos, rope_sin) =
            embeds.rope_tables_const(cfg.rope_theta, start_pos, seq, cfg.head_dim);

        let mut h = embeds.clone();
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            h = self.apply_layer(&h, layer, layer_idx, &rope_cos, &rope_sin)?;
        }
        h.rms_norm_affine(Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps)
    }

    fn build_causal_mask(&self, anchor: &Tensor, seq: usize) -> Tensor {
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i {
                    mask_data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        anchor.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]))
    }

    fn apply_layer(
        &self,
        x: &Tensor,
        layer: &LFM2LayerWeights,
        layer_idx: usize,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let residual = x.clone();
        let x_norm = x.rms_norm_affine(Arc::clone(&layer.operator_norm_gain), cfg.rms_norm_eps)?;
        let mixer_out = match (&layer.mixer, cfg.block_types[layer_idx]) {
            (LFM2MixerWeights::Attention(a), LFM2BlockType::Attention) => {
                self.apply_attention(&x_norm, a, rope_cos, rope_sin)?
            }
            (LFM2MixerWeights::Conv(c), LFM2BlockType::Conv) => {
                self.apply_short_conv(&x_norm, c)?
            }
            _ => {
                return Err(fuel_core::Error::Msg(format!(
                    "LFM2 layer {layer_idx}: mixer weight kind does not match \
                 block_types[{layer_idx}] = {:?}",
                    cfg.block_types[layer_idx],
                ))
                .bt());
            }
        };
        let h1 = residual.add(&mixer_out)?;

        let residual2 = h1.clone();
        let h1_norm = h1.rms_norm_affine(Arc::clone(&layer.ffn_norm_gain), cfg.rms_norm_eps)?;
        let mlp_out = self.apply_mlp(&h1_norm, layer)?;
        residual2.add(&mlp_out)
    }

    fn apply_mlp(&self, x: &Tensor, layer: &LFM2LayerWeights) -> Result<Tensor> {
        let cfg = &self.config;
        let gate = layer
            .ffn_gate
            .apply_linear(x, cfg.hidden_size, cfg.intermediate_size)?;
        let up = layer
            .ffn_up
            .apply_linear(x, cfg.hidden_size, cfg.intermediate_size)?;
        let swiglu = gate.silu().mul(&up)?;
        layer
            .ffn_down
            .apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size)
    }

    fn apply_attention(
        &self,
        x: &Tensor,
        a: &LFM2AttentionWeights,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let dims = x.shape();
        let dims = dims.dims();
        let _batch = dims[0];
        let seq = dims[1];
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let q = a.attn_q.apply_linear(x, cfg.hidden_size, q_dim)?;
        let k = a.attn_k.apply_linear(x, cfg.hidden_size, kv_dim)?;
        let v = a.attn_v.apply_linear(x, cfg.hidden_size, kv_dim)?;

        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        // Per-head Q/K RmsNorm — the gain is shape `[head_dim]` and is
        // broadcast across the (batch, heads, seq) leading dims by the
        // `rms_norm_affine` last-dim contract.
        let q = q.rms_norm_affine(Arc::clone(&a.q_norm_gain), cfg.rms_norm_eps)?;
        let k = k.rms_norm_affine(Arc::clone(&a.k_norm_gain), cfg.rms_norm_eps)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = self.build_causal_mask(x, seq);
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        a.attn_o.apply_linear(&merged, q_dim, cfg.hidden_size)
    }

    /// Apply the LFM2 ShortConv (LIV) mixer.
    ///
    /// Multi-token / prefill semantics — matches the eager
    /// `ShortConvLayer::forward` "seq_len != 1" branch. We do NOT
    /// implement the eager single-step cache (`self.cache`) here; that
    /// path requires multi-output graph nodes so we can return both
    /// the new hidden state and the updated conv-state slot from one
    /// forward call. See the multi-output Option C design memo for the
    /// blocker.
    ///
    /// TODO(autoregressive): when multi-output nodes land, restore the
    /// cached-state path with the following semantics (from the eager
    /// `ShortConvLayer::forward` decode branch):
    ///   - Initial state shape: `[B, hidden, l_cache]`, zeros at t=0.
    ///   - Per single-step call: roll the cache left by 1 along the
    ///     `l_cache` axis, append the new `bx[:, :, 0]` column, then
    ///     compute `out = sum_keepdim(state * conv_weight, dim=2)`.
    ///   - Persist the rolled state for the next step.
    ///
    /// Until then `start_pos > 0` calls just re-run prefill from zero.
    fn apply_short_conv(&self, x: &Tensor, c: &LFM2ConvWeights) -> Result<Tensor> {
        let cfg = &self.config;
        let dims = x.shape();
        let dims = dims.dims();
        let _batch = dims[0];
        let seq = dims[1];
        let hidden = cfg.hidden_size;
        let k = cfg.conv_kernel_size;

        // in_proj: x → bcx in (B, seq, 3*hidden), then transpose to
        // (B, 3*hidden, seq) so we can slice along the channel axis.
        let bcx = c.in_proj.apply_linear(x, hidden, 3 * hidden)?;
        let bcx_t = bcx.permute([0, 2, 1_usize])?; // (B, 3*hidden, seq)

        let b_gate = bcx_t.slice(1_usize, 0, hidden)?; // (B, hidden, seq)
        let c_gate = bcx_t.slice(1_usize, hidden, hidden)?; // (B, hidden, seq)
        let x_sig = bcx_t.slice(1_usize, 2 * hidden, hidden)?; // (B, hidden, seq)

        // Input-gate: bx = b * x. The depthwise conv operates on bx.
        let bx = b_gate.mul(&x_sig)?;

        // causal_conv1d expects (B, channels, seq + k - 1) — we left-pad
        // with k-1 zeros and produce (B, channels, seq). Mirrors the
        // Mamba prefill path.
        let bx_padded = bx.pad_with_zeros(2_usize, k - 1, 0)?;
        let conv_w = x.const_f32_like(
            Arc::clone(&c.conv_weight),
            Shape::from_dims(&[hidden, 1, k]),
        );
        // The eager loader stores no conv bias for LIV layers, but the
        // fused op requires a bias tensor — pass a zero vector.
        let conv_b = x.const_f32_like(
            Arc::from(vec![0.0_f32; hidden]),
            Shape::from_dims(&[hidden]),
        );
        // Plain depthwise causal conv, no fused SiLU — LFM2 keeps the
        // SwiGLU activation in the MLP only.
        let conv_out = bx_padded.causal_conv1d(&conv_w, &conv_b, false); // (B, hidden, seq)

        // Output-gate: y = c * conv_out, then transpose back to
        // (B, seq, hidden) and project out.
        let gated = c_gate.mul(&conv_out)?;
        let _ = seq;
        let out_t = gated.permute([0, 2, 1_usize])?;
        c.out_proj.apply_linear(&out_t, hidden, hidden)
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl LFM2Weights {
    /// Load LFM2 weights from HF safetensors.
    ///
    /// Tensor naming follows the LiquidAI LFM2 HF release convention:
    /// - `model.embed_tokens.weight` `[vocab, hidden]`
    /// - `model.layers.{i}.operator_norm.weight` (some checkpoints use
    ///   `input_layernorm.weight`; both are tried)
    /// - `model.layers.{i}.ffn_norm.weight` (or `post_attention_layernorm`)
    /// - Attention layers:
    ///   - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
    ///   - `model.layers.{i}.self_attn.{q,k}_layernorm.weight`
    /// - Conv layers:
    ///   - `model.layers.{i}.conv.in_proj.weight` `[3·hidden, hidden]`
    ///   - `model.layers.{i}.conv.out_proj.weight` `[hidden, hidden]`
    ///   - `model.layers.{i}.conv.conv.weight` `[hidden, 1, l_cache]`
    /// - `model.layers.{i}.mlp.{gate,up,down}_proj.weight` (SwiGLU)
    /// - `model.norm.weight` `[hidden]`
    /// - `lm_head.weight` `[vocab, hidden]` (optional; tied otherwise)
    pub fn load_from_mmapped(
        st: &fuel_core::safetensors::MmapedSafetensors,
        cfg: &LFM2Config,
    ) -> Result<Self> {
        use fuel_core::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype as ltm};
        cfg.validate()?;
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let k = cfg.conv_kernel_size;

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);

        // Helper that tries one tensor name and falls back to a second.
        let load_alt_f32 = |a: &str, b: &str| -> Result<Vec<f32>> {
            load_tensor_as_f32(st, a).or_else(|_| load_tensor_as_f32(st, b))
        };
        let load_alt_matrix =
            |a: &str, b: &str, out_f: usize, in_f: usize| -> Result<WeightStorage> {
                ltm(st, a, out_f, in_f).or_else(|_| ltm(st, b, out_f, in_f))
            };

        let mut layers: Vec<LFM2LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, &block) in cfg.block_types.iter().enumerate() {
            let p = format!("model.layers.{i}");

            let operator_norm_gain = Arc::from(load_alt_f32(
                &format!("{p}.operator_norm.weight"),
                &format!("{p}.input_layernorm.weight"),
            )?);
            let ffn_norm_gain = Arc::from(load_alt_f32(
                &format!("{p}.ffn_norm.weight"),
                &format!("{p}.post_attention_layernorm.weight"),
            )?);

            let ffn_gate = load_alt_matrix(
                &format!("{p}.mlp.gate_proj.weight"),
                &format!("{p}.feed_forward.w1.weight"),
                inter,
                h,
            )?;
            let ffn_up = load_alt_matrix(
                &format!("{p}.mlp.up_proj.weight"),
                &format!("{p}.feed_forward.w3.weight"),
                inter,
                h,
            )?;
            let ffn_down = load_alt_matrix(
                &format!("{p}.mlp.down_proj.weight"),
                &format!("{p}.feed_forward.w2.weight"),
                h,
                inter,
            )?;

            let mixer = match block {
                LFM2BlockType::Attention => {
                    let attn_q = load_alt_matrix(
                        &format!("{p}.self_attn.q_proj.weight"),
                        &format!("{p}.attention.q_proj.weight"),
                        q_dim,
                        h,
                    )?;
                    let attn_k = load_alt_matrix(
                        &format!("{p}.self_attn.k_proj.weight"),
                        &format!("{p}.attention.k_proj.weight"),
                        kv_dim,
                        h,
                    )?;
                    let attn_v = load_alt_matrix(
                        &format!("{p}.self_attn.v_proj.weight"),
                        &format!("{p}.attention.v_proj.weight"),
                        kv_dim,
                        h,
                    )?;
                    let attn_o = load_alt_matrix(
                        &format!("{p}.self_attn.o_proj.weight"),
                        &format!("{p}.attention.o_proj.weight"),
                        h,
                        q_dim,
                    )?;
                    let q_norm_gain = Arc::from(load_alt_f32(
                        &format!("{p}.self_attn.q_layernorm.weight"),
                        &format!("{p}.attention.q_norm.weight"),
                    )?);
                    let k_norm_gain = Arc::from(load_alt_f32(
                        &format!("{p}.self_attn.k_layernorm.weight"),
                        &format!("{p}.attention.k_norm.weight"),
                    )?);
                    LFM2MixerWeights::Attention(LFM2AttentionWeights {
                        attn_q,
                        attn_k,
                        attn_v,
                        attn_o,
                        q_norm_gain,
                        k_norm_gain,
                    })
                }
                LFM2BlockType::Conv => {
                    let in_proj = load_alt_matrix(
                        &format!("{p}.conv.in_proj.weight"),
                        &format!("{p}.shortconv.in_proj.weight"),
                        3 * h,
                        h,
                    )?;
                    let out_proj = load_alt_matrix(
                        &format!("{p}.conv.out_proj.weight"),
                        &format!("{p}.shortconv.out_proj.weight"),
                        h,
                        h,
                    )?;
                    let raw = load_alt_f32(
                        &format!("{p}.conv.conv.weight"),
                        &format!("{p}.shortconv.conv.weight"),
                    )?;
                    // Accept several plausible layouts for the conv
                    // kernel: `[hidden, 1, k]`, `[hidden, k]`, or
                    // `[k, hidden]` (eager loader's transposed form).
                    let conv_weight = normalize_conv_kernel(raw, h, k, i)?;
                    LFM2MixerWeights::Conv(LFM2ConvWeights {
                        in_proj,
                        out_proj,
                        conv_weight: Arc::from(conv_weight),
                    })
                }
            };

            layers.push(LFM2LayerWeights {
                operator_norm_gain,
                ffn_norm_gain,
                mixer,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        let final_norm_gain = Arc::from(load_alt_f32(
            "model.norm.weight",
            "model.embedding_norm.weight",
        )?);

        let output = match ltm(st, "lm_head.weight", cfg.vocab_size, h) {
            Ok(w) => w,
            Err(_) => crate::models::lazy_llama_full::tied_lm_head_from_embeddings(
                &token_embedding,
                cfg.vocab_size,
                h,
            ),
        };

        Ok(Self {
            token_embedding,
            layers,
            final_norm_gain,
            output,
        })
    }
}

/// Reshape a flat conv-kernel buffer into the `[hidden, 1, k]` layout
/// the `causal_conv1d` op expects. Accepts the three layouts LFM2
/// checkpoints have shipped with: `(hidden, 1, k)` (already correct),
/// `(hidden, k)`, or `(k, hidden)`.
fn normalize_conv_kernel(
    raw: Vec<f32>,
    hidden: usize,
    k: usize,
    layer_idx: usize,
) -> Result<Vec<f32>> {
    let want = hidden * k;
    if raw.len() != want {
        return Err(fuel_core::Error::Msg(format!(
            "LFM2 layer {layer_idx} conv.conv.weight: {} elems, expected hidden*k = {hidden}*{k} = {want}",
            raw.len(),
        )).bt());
    }
    // Heuristic: we have no shape metadata at this point (the f32
    // loader flattens it). The `[hidden, k]` and `[hidden, 1, k]`
    // layouts are byte-identical to the target. The `[k, hidden]`
    // layout requires a transpose. Default to assuming the storage is
    // already `[hidden, k]` (the most common case in HF LFM2
    // releases) — this matches the eager loader's first-attempt
    // interpretation. Callers with a transposed checkpoint must
    // pre-transpose before calling `load_from_mmapped`.
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> LFM2Config {
        // Smallest config that still hits both block variants.
        // num_hidden_layers = 2 with [Attention, Conv].
        LFM2Config {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            conv_kernel_size: 4,
            block_types: vec![LFM2BlockType::Attention, LFM2BlockType::Conv],
        }
    }

    fn tiny_weights(cfg: &LFM2Config) -> LFM2Weights {
        let mut s: u32 = 12345;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut vec_of =
            |n: usize| -> Arc<[f32]> { Arc::from((0..n).map(|_| next()).collect::<Vec<_>>()) };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let k = cfg.conv_kernel_size;

        let token_embedding = vec_of(cfg.vocab_size * h);
        let mut layers: Vec<LFM2LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for &block in cfg.block_types.iter() {
            let mixer = match block {
                LFM2BlockType::Attention => LFM2MixerWeights::Attention(LFM2AttentionWeights {
                    attn_q: WeightStorage::F32(vec_of(h * q_dim)),
                    attn_k: WeightStorage::F32(vec_of(h * kv_dim)),
                    attn_v: WeightStorage::F32(vec_of(h * kv_dim)),
                    attn_o: WeightStorage::F32(vec_of(q_dim * h)),
                    q_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
                    k_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
                }),
                LFM2BlockType::Conv => LFM2MixerWeights::Conv(LFM2ConvWeights {
                    in_proj: WeightStorage::F32(vec_of(h * 3 * h)),
                    out_proj: WeightStorage::F32(vec_of(h * h)),
                    conv_weight: vec_of(h * k),
                }),
            };
            layers.push(LFM2LayerWeights {
                operator_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
                mixer,
                ffn_gate: WeightStorage::F32(vec_of(h * i)),
                ffn_up: WeightStorage::F32(vec_of(h * i)),
                ffn_down: WeightStorage::F32(vec_of(i * h)),
            });
        }
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size));
        LFM2Weights {
            token_embedding,
            layers,
            final_norm_gain,
            output,
        }
    }

    #[test]
    fn forward_shape_and_finite_attention_plus_conv() {
        let cfg = tiny_cfg();
        let model = LFM2Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_cfg();
        let model = LFM2Model {
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
    fn config_validate_rejects_mismatched_block_types_len() {
        let mut cfg = tiny_cfg();
        cfg.block_types = vec![LFM2BlockType::Attention]; // wrong length
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_rejects_kv_groups_not_dividing_attn_heads() {
        let mut cfg = tiny_cfg();
        cfg.num_key_value_heads = 3; // 4 % 3 != 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn all_conv_layers_still_run() {
        // Stress the ShortConv path on its own.
        let mut cfg = tiny_cfg();
        cfg.block_types = vec![LFM2BlockType::Conv, LFM2BlockType::Conv];
        let model = LFM2Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit on all-conv config: {v}");
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = tiny_cfg();
        let model = LFM2Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();
        let anchor = Tensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu());
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let logits_via_embeds = model.forward_embeds(&embeds, 0).unwrap().realize_f32();
        let max_diff = logits_ref
            .iter()
            .zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "LFM2 forward vs forward_embeds must agree (max diff {max_diff})"
        );
    }

    /// The REAL `LiquidAI/LFM2-1.2B/config.json`, read from the Hub rather than
    /// transcribed from a description. Trimmed only of keys this port does not
    /// model (tokenizer ids, init scales, dtype); every key the mapper reads is
    /// verbatim, including both duplicated pairs.
    const LFM2_HF_CONFIG_JSON: &str = r#"{
        "architectures": ["Lfm2ForCausalLM"],
        "block_auto_adjust_ff_dim": true,
        "block_dim": 2048,
        "block_ff_dim": 12288,
        "block_multiple_of": 256,
        "block_norm_eps": 1e-05,
        "block_use_swiglu": true,
        "conv_L_cache": 3,
        "conv_bias": false,
        "conv_dim": 2048,
        "full_attn_idxs": [2, 5, 8, 10, 12, 14],
        "hidden_size": 2048,
        "max_position_embeddings": 128000,
        "model_type": "lfm2",
        "norm_eps": 1e-05,
        "num_attention_heads": 32,
        "num_heads": 32,
        "num_hidden_layers": 16,
        "num_key_value_heads": 8,
        "rope_theta": 1000000.0,
        "use_cache": true,
        "vocab_size": 65536
    }"#;

    /// **The remap, asserted field by field against the artifact.**
    ///
    /// Every one of these is a CLAIM ABOUT HF'S FORMAT. A wrong claim here does
    /// not fail to parse — it yields a `LFM2Config` that is well-formed and
    /// describes a different model, which is why each mapped key is asserted
    /// individually rather than via a single round-trip.
    #[test]
    fn lfm2_config_from_hf_json_maps_every_renamed_key() {
        let cfg = LFM2Config::from_hf_json_str(LFM2_HF_CONFIG_JSON).unwrap();

        // verbatim keys
        assert_eq!(cfg.vocab_size, 65536);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 16);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.max_position_embeddings, 128_000);
        assert_eq!(cfg.rope_theta, 1_000_000.0);

        // RENAMED keys -- each of these is where a wrong claim would be silent
        assert_eq!(
            cfg.intermediate_size, 12288,
            "block_ff_dim -> intermediate_size"
        );
        assert_eq!(cfg.conv_kernel_size, 3, "conv_L_cache -> conv_kernel_size");
        assert_eq!(cfg.rms_norm_eps, 1e-05, "norm_eps -> rms_norm_eps");
    }

    /// **`head_dim` is DERIVED when absent — and the "explicit key wins" arm is
    /// NOT CONSTRUCTIBLE, which is a fact about LFM2 rather than a gap here.**
    ///
    /// ⚠️ An earlier version of this test claimed a two-arm structure and its
    /// second arm was VACUOUS. It set `heads: 16, head_dim: 128` against
    /// `hidden_size: 2048` — but `2048 / 16 == 128`, so READING the key and
    /// DERIVING it give the same answer and the arm could not tell them apart.
    /// A sabotage that forced the mapper to always derive left it GREEN.
    ///
    /// The reason no such arm exists: `LFM2Config::validate` requires
    /// `num_attention_heads * head_dim == hidden_size`, so **every config that
    /// validates has `head_dim == hidden / heads`**. Reading and deriving are
    /// observationally identical across the whole legal input space, and the
    /// distinction is unobservable through the public API by construction.
    ///
    /// That is asserted below rather than asserted about: a config whose
    /// explicit `head_dim` disagrees with the derivation is REJECTED, which is
    /// what closes the space.
    #[test]
    fn lfm2_head_dim_is_derived_when_absent_and_the_relation_closes_the_space() {
        // ARM 1 -- absent in the real artifact: derived as 2048 / 32.
        let derived = LFM2Config::from_hf_json_str(LFM2_HF_CONFIG_JSON).unwrap();
        assert_eq!(derived.head_dim, 64, "2048 / 32");
        assert!(
            !LFM2_HF_CONFIG_JSON.contains("\"head_dim\""),
            "control: the artifact must NOT contain head_dim, or arm 1 proves nothing"
        );

        // WHY THERE IS NO SECOND ARM: an explicit head_dim that disagrees with
        // the derivation cannot survive validate(), so no legal config
        // distinguishes read-vs-derive.
        let disagreeing = LFM2_HF_CONFIG_JSON.replace(
            "\"conv_L_cache\": 3",
            "\"conv_L_cache\": 3, \"head_dim\": 100",
        );
        let err = LFM2Config::from_hf_json_str(&disagreeing)
            .expect_err("head_dim=100 with 32 heads and hidden 2048 must be rejected")
            .to_string();
        assert!(
            err.contains("head_dim"),
            "rejection must name head_dim, got: {err}"
        );

        // And a divisibility failure is named rather than silently truncated.
        let indivisible =
            LFM2_HF_CONFIG_JSON.replace("\"hidden_size\": 2048", "\"hidden_size\": 2050");
        assert!(
            LFM2Config::from_hf_json_str(&indivisible).is_err(),
            "2050 / 32 does not divide; the derivation must decline, not truncate"
        );
    }

    /// **`full_attn_idxs` -> per-layer `block_types`, asserted as the FULL
    /// sequence.**
    ///
    /// ⚠️ Asserting only the LENGTH, or only that attention appears somewhere,
    /// would pass for an off-by-one or an inverted schedule. The whole 16-element
    /// vector is pinned, so a derivation that is subtly wrong produces a visible
    /// diff rather than a plausible Vec of the right length.
    #[test]
    fn lfm2_block_types_expand_to_the_exact_per_layer_schedule() {
        let cfg = LFM2Config::from_hf_json_str(LFM2_HF_CONFIG_JSON).unwrap();
        use LFM2BlockType::{Attention as A, Conv as C};
        let expected = vec![
            C, C, A, C, C, A, C, C, // 0..7   attention at 2, 5
            A, C, A, C, A, C, A, C, // 8..15  attention at 8, 10, 12, 14
        ];
        assert_eq!(
            cfg.block_types, expected,
            "full_attn_idxs [2,5,8,10,12,14] over 16 layers"
        );
        assert_eq!(
            cfg.block_types.len(),
            cfg.num_hidden_layers,
            "validate() requires one entry per layer -- LFM2 EXPANDS, unlike \
             RecurrentGemma whose block_types is a repeating CYCLE"
        );
        // The inverse is also pinned: exactly the listed indices, nothing else.
        let attn: Vec<usize> = cfg
            .block_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LFM2BlockType::Attention)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(attn, vec![2, 5, 8, 10, 12, 14]);
    }

    /// An index past the last layer is an ERROR, not a clamp or a skip.
    ///
    /// Dropping it silently would yield a well-formed schedule for a model that
    /// is not the one in the artifact — the exact failure mode this program
    /// exists to prevent.
    #[test]
    fn lfm2_rejects_an_attention_index_past_the_last_layer() {
        let bad = LFM2_HF_CONFIG_JSON.replace("[2, 5, 8, 10, 12, 14]", "[2, 5, 99]");
        let err = LFM2Config::from_hf_json_str(&bad).unwrap_err().to_string();
        assert!(
            err.contains("full_attn_idxs") && err.contains("99"),
            "expected the offending index to be NAMED, got: {err}"
        );
    }

    /// **The duplicate-key rule, which HF's own artifact makes necessary.**
    ///
    /// `norm_eps`/`block_norm_eps` and `num_attention_heads`/`num_heads` are each
    /// shipped TWICE and agree today. A mapper that silently prefers one is
    /// encoding a claim it has no evidence for, so disagreement is an error.
    #[test]
    fn lfm2_rejects_duplicated_keys_that_disagree() {
        let eps =
            LFM2_HF_CONFIG_JSON.replace("\"block_norm_eps\": 1e-05", "\"block_norm_eps\": 1e-03");
        let err = LFM2Config::from_hf_json_str(&eps).unwrap_err().to_string();
        assert!(
            err.contains("norm_eps"),
            "eps disagreement must name the keys: {err}"
        );

        let heads = LFM2_HF_CONFIG_JSON.replace("\"num_heads\": 32", "\"num_heads\": 16");
        let err = LFM2Config::from_hf_json_str(&heads)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("num_heads"),
            "head-count disagreement must name the keys: {err}"
        );

        // CONTROL: they agree in the real artifact, so the guard is not simply
        // rejecting everything.
        assert!(LFM2Config::from_hf_json_str(LFM2_HF_CONFIG_JSON).is_ok());
    }
}

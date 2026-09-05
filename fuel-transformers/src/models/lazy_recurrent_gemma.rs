// SPDX-License-Identifier: MIT OR Apache-2.0
//! Recurrent Gemma decoder ported to the lazy-graph API.
//!
//! Phase D specialized port. Recurrent Gemma alternates between
//! attention layers (Gemma-shaped GQA with halved-feature partial
//! rotary) and **recurrent** layers built around a Real-Gated
//! Linear Recurrent Unit (RG-LRU). Block types are configured by
//! a `block_types: Vec<TemporalBlockType>` cycle, e.g.
//! `[R, R, A]` repeats: two recurrent then one attention.
//!
//! # Recurrent block (RecurrentBlock)
//!
//!   ```text
//!   y = act(linear_y(x))                          ; gating branch
//!   x_branch = linear_x(x)                        ; recurrence input
//!   x_branch = causal_conv1d(x_branch, w=4)       ; depthwise, kernel 4
//!   x_branch = rg_lru(x_branch)                   ; recurrent unit
//!   out      = linear_out(x_branch * y)
//!   ```
//!
//! ## RG-LRU
//!
//! Per layer, a block-structured recurrence with `n_heads` blocks
//! each of `block_width = lru_width / n_heads` features:
//!
//!   ```text
//!   i_gate    = sigmoid(per_head_W_i @ x + b_i)       ; input gate
//!   r_gate    = sigmoid(per_head_W_r @ x + b_r)       ; recurrent gate
//!   log_decay = -8 * r_gate * softplus(recurrent_param)
//!   decay     = exp(log_decay)
//!   a_square  = exp(2 * log_decay)
//!   gated_x   = x * i_gate
//!   mult      = reset + (1 - reset) * sqrt(1 - a_square)
//!   x_in      = gated_x * mult
//!   state[t]  = decay[t] * (1 - reset[t]) * state[t-1] + x_in[t]
//!   ```
//!
//! At `pos == 0` reset is `1` so `state[0] = x_in[0]`; thereafter
//! reset is `0`. v1 only supports prefill from zero state — no
//! cross-call state resumption — so the model only sees the
//! "first chunk" reset path.
//!
//! # Attention block
//!
//! Standard Gemma GQA. Partial rotary with `partial_rotary_factor
//! == 0.5` is hard-coded by the eager reference (and asserted by
//! this port): only the first half of each head's features are
//! rotated. The `attention_window_size` is read but applied as a
//! sliding-window mask (a v1 simplification — eager uses local
//! causal masking).
//!
//! # MLP
//!
//! `down(act(gate(x)) * up(x))`, with intermediate size
//! `intermediate_size / 2` per the eager reference (a Gemma-RG
//! quirk — the config's `intermediate_size` is the *fused* width,
//! halved for SwiGLU's two-branch path).
//!
//! # Other carries
//!
//! Offset RmsNorm `(gain + 1)` (Gemma family convention).
//! Tied lm_head to `token_embedding`. Soft-cap on final logits via
//! `logits_soft_cap`.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache,
//! no cross-call state, F32. The recurrent scan is unrolled at
//! graph-build time (same shape as the RWKV-5 port — long
//! prompts produce large but well-formed graphs).

use fuel_core::lazy::{Tensor, WeightStorage};
use fuel_core::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

pub use crate::models::lazy_gemma::GemmaActivation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalBlockType {
    Attention,
    Recurrent,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentGemmaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// LRU width; defaults to `hidden_size` if `None`.
    pub lru_width: Option<usize>,
    pub attention_window_size: usize,
    pub conv1d_width: usize,
    pub logits_soft_cap: f64,
    pub hidden_activation: GemmaActivation,
    /// Must equal 0.5 to match the eager reference (asserted).
    pub partial_rotary_factor: f64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub block_types: Vec<TemporalBlockType>,
    pub attention_bias: bool,
    pub max_seq_len: usize,
}

impl RecurrentGemmaConfig {
    pub fn lru_width_or_default(&self) -> usize {
        self.lru_width.unwrap_or(self.hidden_size)
    }
    pub fn block_width(&self) -> usize {
        self.lru_width_or_default() / self.num_attention_heads
    }
    pub fn mlp_intermediate(&self) -> usize {
        self.intermediate_size / 2
    }
    pub fn block_type(&self, layer_idx: usize) -> TemporalBlockType {
        self.block_types[layer_idx % self.block_types.len()]
    }
}

// ROADMAP item 8 (II), key/shape-mismatch program. RecurrentGemma is the
// SHAPE case — but not the shape the dispatch described, and the difference is
// the whole point of reading the code before writing the mapper.
//
// Measured against the real artifact (`google/recurrentgemma-2b/config.json`,
// read through the authenticated Hub connector because the repo is gated):
//
//     _block_types: ["recurrent","recurrent","attention"]   3 elements
//     num_hidden_layers: 26
//
// ⚠️ THE DISPATCH SAID THIS NEEDS PATTERN -> PER-LAYER EXPANSION. IT DOES NOT.
// `RecurrentGemmaConfig::block_type()` indexes
// `self.block_types[layer_idx % self.block_types.len()]` — a CYCLE — and this
// module's own header says so: *"Block types are configured by a
// `block_types: Vec<TemporalBlockType>` cycle, e.g. `[R, R, A]` repeats"*.
// The existing fixtures use lengths 1 and 3 against larger layer counts.
// So `_block_types` maps DIRECTLY and expanding it would encode a claim the
// struct does not make.
//
// ⚠️ CONTRAST WITH LFM2, which IS an expansion: `LFM2Config::validate`
// requires `block_types.len() == num_hidden_layers`. Two models, one
// same-sounding field, opposite correct answers.
//
// ⚠️ AND `attention_window_size` IS ITS OWN FIELD on the struct (read at the
// attention site), NOT a source for `max_seq_len`. Mapping one into the other
// would conflate a sliding window with a sequence bound. `max_seq_len` has no
// counterpart in the artifact — see `resolve` for what it is set to and why.
#[derive(Debug, Clone, serde::Deserialize)]
struct RecurrentGemmaConfigRaw {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    head_dim: usize,
    #[serde(default)]
    lru_width: Option<usize>,
    attention_window_size: usize,
    conv1d_width: usize,
    logits_soft_cap: f64,
    hidden_activation: String,
    partial_rotary_factor: f64,
    rms_norm_eps: f64,
    rope_theta: f64,
    /// Underscore-prefixed in the artifact. A repeating CYCLE, not a per-layer
    /// list — mapped verbatim, see the module note above.
    #[serde(rename = "_block_types")]
    block_types: Vec<String>,
    #[serde(default)]
    attention_bias: bool,
}

/// Map RecurrentGemma's `_block_types` strings to [`TemporalBlockType`].
///
/// Unknown values ERROR rather than defaulting: a block kind this port does not
/// implement is a fact worth surfacing, and defaulting it to either variant
/// would produce a model that runs and is wrong.
fn recurrent_gemma_block_type_from_str(s: &str) -> fuel_core::Result<TemporalBlockType> {
    match s {
        "attention" => Ok(TemporalBlockType::Attention),
        "recurrent" => Ok(TemporalBlockType::Recurrent),
        other => Err(fuel_core::Error::Msg(format!(
            "unsupported RecurrentGemma _block_types entry {other:?} \
             (expected \"attention\" or \"recurrent\")"
        ))),
    }
}

/// Map `hidden_activation` to [`GemmaActivation`]. Unknown values ERROR.
fn recurrent_gemma_activation_from_str(s: &str) -> fuel_core::Result<GemmaActivation> {
    match s {
        "gelu_pytorch_tanh" => Ok(GemmaActivation::GeluPytorchTanh),
        "gelu" => Ok(GemmaActivation::Gelu),
        other => Err(fuel_core::Error::Msg(format!(
            "unsupported RecurrentGemma hidden_activation {other:?} \
             (expected gelu/gelu_pytorch_tanh)"
        ))),
    }
}

impl RecurrentGemmaConfigRaw {
    fn from_json_str(json: &str) -> fuel_core::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| fuel_core::Error::Msg(format!("parsing RecurrentGemma config.json: {e}")))
    }

    fn resolve(self) -> fuel_core::Result<RecurrentGemmaConfig> {
        if self.block_types.is_empty() {
            return Err(fuel_core::Error::Msg(
                "RecurrentGemma config.json: _block_types is empty, so block_type() \
                 would divide by zero"
                    .into(),
            ));
        }
        let block_types = self
            .block_types
            .iter()
            .map(|s| recurrent_gemma_block_type_from_str(s))
            .collect::<fuel_core::Result<Vec<_>>>()?;

        Ok(RecurrentGemmaConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: fuel_core::hf_config::num_key_value_heads(
                self.num_key_value_heads,
                self.num_attention_heads,
            )?,
            head_dim: self.head_dim,
            lru_width: self.lru_width,
            attention_window_size: self.attention_window_size,
            conv1d_width: self.conv1d_width,
            logits_soft_cap: self.logits_soft_cap,
            hidden_activation: recurrent_gemma_activation_from_str(&self.hidden_activation)?,
            partial_rotary_factor: self.partial_rotary_factor,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            block_types,
            attention_bias: self.attention_bias,
            // ⚠️ NO HF COUNTERPART. RecurrentGemma's config ships no
            // `max_position_embeddings` — a recurrent model has no positional
            // table to bound, and its attention layers are windowed instead.
            // `max_seq_len` is set to the attention window, which is the largest
            // span any attention layer in this model attends, and is the only
            // number in the artifact with a defensible claim to the name.
            //
            // This is a DERIVED value, not a read one, and the field is
            // currently unread by the port (declared and fixtured only), so the
            // choice is unobservable today. Recorded here rather than left to be
            // rediscovered: if `max_seq_len` ever becomes load-bearing, this
            // line is a decision that needs re-making, not a fact that was read.
            max_seq_len: self.attention_window_size,
        })
    }
}

impl RecurrentGemmaConfig {
    /// Parse a HuggingFace RecurrentGemma `config.json` string.
    ///
    /// ROADMAP item 8 (II): reads the artifact rather than returning a preset.
    /// See `RecurrentGemmaConfigRaw` for why `_block_types` is mapped VERBATIM
    /// (it is a cycle, not a per-layer list) and what `max_seq_len` is set to.
    pub fn from_hf_json_str(json: &str) -> fuel_core::Result<Self> {
        RecurrentGemmaConfigRaw::from_json_str(json)?.resolve()
    }
}

#[derive(Debug, Clone)]
pub struct RgluWeights {
    /// `[lru_width]` — softplus'd per-feature parameter for the decay.
    pub recurrent_param: Arc<[f32]>,
    /// `[n_heads, block_width, block_width]` — per-head input gate matrix.
    pub input_gate_weight: Arc<[f32]>,
    /// `[n_heads, block_width]` — per-head input gate bias.
    pub input_gate_bias: Arc<[f32]>,
    pub recurrent_gate_weight: Arc<[f32]>,
    pub recurrent_gate_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct RecurrentBlockWeights {
    pub linear_y_w: WeightStorage, // hidden → lru_width
    pub linear_y_b: Arc<[f32]>,
    pub linear_x_w: WeightStorage, // hidden → lru_width
    pub linear_x_b: Arc<[f32]>,
    pub linear_out_w: WeightStorage, // lru_width → hidden
    pub linear_out_b: Arc<[f32]>,
    /// `[lru_width, 1, conv1d_width]` depthwise kernel.
    pub conv1d_w: Arc<[f32]>,
    /// `[lru_width]`.
    pub conv1d_b: Arc<[f32]>,
    pub rg_lru: RgluWeights,
}

#[derive(Debug, Clone)]
pub struct AttentionBlockWeights {
    pub q_w: WeightStorage,
    pub q_b: Option<Arc<[f32]>>,
    pub k_w: WeightStorage,
    pub k_b: Option<Arc<[f32]>>,
    pub v_w: WeightStorage,
    pub v_b: Option<Arc<[f32]>>,
    pub o_w: WeightStorage,
    pub o_b: Arc<[f32]>, // o_proj always has bias in recurrent_gemma
}

#[derive(Debug, Clone)]
pub enum TemporalBlockWeights {
    Attention(AttentionBlockWeights),
    Recurrent(RecurrentBlockWeights),
}

#[derive(Debug, Clone)]
pub struct RecurrentGemmaLayerWeights {
    pub temporal_pre_norm_gain: Arc<[f32]>,
    pub channel_pre_norm_gain: Arc<[f32]>,
    pub temporal: TemporalBlockWeights,
    pub mlp_gate_w: WeightStorage,
    pub mlp_gate_b: Arc<[f32]>,
    pub mlp_up_w: WeightStorage,
    pub mlp_up_b: Arc<[f32]>,
    pub mlp_down_w: WeightStorage,
    pub mlp_down_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct RecurrentGemmaWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<RecurrentGemmaLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    // lm_head is tied to token_embedding.
}

#[derive(Debug, Clone)]
pub struct RecurrentGemmaModel {
    pub config: RecurrentGemmaConfig,
    pub weights: RecurrentGemmaWeights,
}

impl RecurrentGemmaModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final offset RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// RecurrentGemma-specific: per-layer Attention vs. Recurrent
    /// (LRU) temporal block selection from `block_types`.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. RecurrentGemma does NOT scale
    /// embeddings (unlike Gemma which applies sqrt(hidden_size) —
    /// RecurrentGemma's eager port omits it).
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
        let lm_head = WeightStorage::F32(self.weights.token_embedding.clone());
        let logits = lm_head.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size)?;
        let sc = cfg.logits_soft_cap;
        if sc > 0.0 {
            Ok(logits.mul_scalar(1.0 / sc).tanh().mul_scalar(sc))
        } else {
            Ok(logits)
        }
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0, "RecurrentGemmaModel: tokens must be non-empty");

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
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(fuel_core::Error::Msg(format!(
                "RecurrentGemmaModel::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(fuel_core::Error::Msg(
                "RecurrentGemmaModel::forward_embeds: seq must be > 0".into(),
            )
            .bt());
        }
        if (cfg.partial_rotary_factor - 0.5).abs() >= 1e-9 {
            return Err(fuel_core::Error::Msg(format!(
                "RecurrentGemmaConfig: partial_rotary_factor must be exactly 0.5 (got {})",
                cfg.partial_rotary_factor,
            ))
            .bt());
        }
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(fuel_core::Error::Msg(
                "num_attention_heads must be a multiple of num_key_value_heads".into(),
            )
            .bt());
        }
        let lru_width = cfg.lru_width_or_default();
        if !lru_width.is_multiple_of(cfg.num_attention_heads) {
            return Err(fuel_core::Error::Msg(format!(
                "lru_width ({lru_width}) must be a multiple of num_attention_heads ({})",
                cfg.num_attention_heads,
            ))
            .bt());
        }
        let mut h = embeds.clone();

        let rope_dim = cfg.head_dim / 2;
        let (rope_cos, rope_sin) = h.rope_tables_const(cfg.rope_theta, start_pos, seq, rope_dim);

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            h = self.apply_layer(&h, layer, layer_idx, &rope_cos, &rope_sin)?;
        }
        h.rms_norm_affine_with_offset(&weights.final_norm_gain, 1.0, cfg.rms_norm_eps)
    }

    fn apply_layer(
        &self,
        x: &Tensor,
        layer: &RecurrentGemmaLayerWeights,
        layer_idx: usize,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let _h = cfg.hidden_size;

        // Temporal sublayer: pre_norm → temporal_block → residual add.
        let residual = x.clone();
        let x_norm =
            x.rms_norm_affine_with_offset(&layer.temporal_pre_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let temporal_out = match (&layer.temporal, cfg.block_type(layer_idx)) {
            (TemporalBlockWeights::Attention(a), TemporalBlockType::Attention) => {
                self.apply_attention(&x_norm, a, rope_cos, rope_sin)?
            }
            (TemporalBlockWeights::Recurrent(r), TemporalBlockType::Recurrent) => {
                self.apply_recurrent(&x_norm, r)?
            }
            _ => {
                return Err(fuel_core::Error::Msg(format!(
                    "RecurrentGemma layer {layer_idx}: weight kind does not match \
                 block_types[{layer_idx} % {}] — config + weights are inconsistent",
                    cfg.block_types.len(),
                ))
                .bt());
            }
        };
        let h1 = residual.add(&temporal_out)?;

        // Channel sublayer: pre_norm → MLP → residual add.
        let residual2 = h1.clone();
        let h1_norm =
            h1.rms_norm_affine_with_offset(&layer.channel_pre_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let mlp_out = self.apply_mlp(&h1_norm, layer)?;
        residual2.add(&mlp_out)
    }

    fn apply_attention(
        &self,
        x: &Tensor,
        a: &AttentionBlockWeights,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let rope_dim = cfg.head_dim / 2;
        let window = cfg.attention_window_size;

        let q = a
            .q_w
            .apply_linear(x, cfg.hidden_size, q_dim)?
            .add_optional_trailing_bias(a.q_b.as_ref())?;
        let k = a
            .k_w
            .apply_linear(x, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(a.k_b.as_ref())?;
        let v = a
            .v_w
            .apply_linear(x, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(a.v_b.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        // Partial rotary on first head_dim/2 features.
        let q_r = q.rope_partial(rope_cos, rope_sin, rope_dim)?;
        let k_r = k.rope_partial(rope_cos, rope_sin, rope_dim)?;

        // GQA expand.
        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        // Sliding-window causal mask.
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i || (window > 0 && j + window <= i) {
                    mask_data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = x.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]));
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = a.o_w.apply_linear(&merged, q_dim, cfg.hidden_size)?;
        attn_out.add_trailing_bias(std::sync::Arc::clone(&a.o_b))
    }

    fn apply_recurrent(&self, x: &Tensor, r: &RecurrentBlockWeights) -> Result<Tensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.hidden_size;
        let lru_width = cfg.lru_width_or_default();
        let n_heads = cfg.num_attention_heads;
        let block_width = cfg.block_width();
        let kernel = cfg.conv1d_width;

        // Gating branch.
        let y = r.linear_y_w.apply_linear_with_bias(
            x,
            h,
            lru_width,
            std::sync::Arc::clone(&r.linear_y_b),
        )?;
        let y_act = match cfg.hidden_activation {
            GemmaActivation::Gelu => y.gelu_erf(),
            GemmaActivation::GeluPytorchTanh => y.gelu(),
        };

        // Recurrence input.
        let x_branch = r.linear_x_w.apply_linear_with_bias(
            x,
            h,
            lru_width,
            std::sync::Arc::clone(&r.linear_x_b),
        )?;

        // Causal conv1d: (batch, seq, lru_width) → (batch, lru_width, seq),
        // pad left with (kernel - 1) zeros, run causal_conv1d, transpose back.
        let x_b_t = x_branch.permute([0, 2, 1_usize])?; // (b, lru, seq)
        let pad_zeros = x.const_f32_like(
            Arc::from(vec![0.0_f32; batch * lru_width * (kernel - 1)]),
            Shape::from_dims(&[batch, lru_width, kernel - 1]),
        );
        let x_b_padded = pad_zeros.concat(&x_b_t, 2_usize)?;
        let conv_w = x.const_f32_like(
            Arc::clone(&r.conv1d_w),
            Shape::from_dims(&[lru_width, 1, kernel]),
        );
        let conv_b = x.const_f32_like(Arc::clone(&r.conv1d_b), Shape::from_dims(&[lru_width]));
        let x_conv = x_b_padded.causal_conv1d(&conv_w, &conv_b, false); // (b, lru, seq)
        let x_back = x_conv.permute([0, 2, 1_usize])?; // (b, seq, lru_width)

        // RG-LRU.
        let x_lru = self.apply_rg_lru(&x_back, &r.rg_lru, batch, seq, n_heads, block_width)?;

        // Gate × output.
        let gated = x_lru.mul(&y_act)?;
        let out_proj = r.linear_out_w.apply_linear(&gated, lru_width, h)?;
        out_proj.add_trailing_bias(std::sync::Arc::clone(&r.linear_out_b))
    }

    fn apply_rg_lru(
        &self,
        x: &Tensor,
        rg: &RgluWeights,
        batch: usize,
        seq: usize,
        n_heads: usize,
        block_width: usize,
    ) -> Result<Tensor> {
        let lru_width = n_heads * block_width;

        // Reshape x to (b, seq, n_heads, block_width).
        let xh = x.reshape(Shape::from_dims(&[batch, seq, n_heads, block_width]))?;

        // Per-head gate projection: for each head h, compute
        //   gate[..., h, :] = sigmoid(W[h] @ x[..., h, :] + b[h])
        // The per-head W has shape (block_width, block_width). We do
        // batched matmul: reshape x to (..., n_heads, 1, block_width) and
        // W to (1, 1, n_heads, block_width, block_width), then matmul.
        let project = |w: &Arc<[f32]>, b: &Arc<[f32]>| -> Result<Tensor> {
            let w_t = x.const_f32_like(
                Arc::clone(w),
                Shape::from_dims(&[1, 1, n_heads, block_width, block_width]),
            );
            let w_bc = w_t.broadcast_to(Shape::from_dims(&[
                batch,
                seq,
                n_heads,
                block_width,
                block_width,
            ]))?;
            let x_row = xh.reshape(Shape::from_dims(&[batch, seq, n_heads, 1, block_width]))?;
            let res = x_row.matmul(&w_bc)?; // (b, seq, n_heads, 1, block_width)
            let res = res.reshape(Shape::from_dims(&[batch, seq, n_heads, block_width]))?;
            // Add per-head bias (n_heads, block_width).
            let b_t = x.const_f32_like(
                Arc::clone(b),
                Shape::from_dims(&[1, 1, n_heads, block_width]),
            );
            let b_bc = b_t.broadcast_to(Shape::from_dims(&[batch, seq, n_heads, block_width]))?;
            res.add(&b_bc)
        };
        let input_gate = project(&rg.input_gate_weight, &rg.input_gate_bias)?.sigmoid();
        let recurrent_gate = project(&rg.recurrent_gate_weight, &rg.recurrent_gate_bias)?.sigmoid();

        // Flatten back to (b, seq, lru_width).
        let input_gate = input_gate.reshape(Shape::from_dims(&[batch, seq, lru_width]))?;
        let recurrent_gate = recurrent_gate.reshape(Shape::from_dims(&[batch, seq, lru_width]))?;

        // log_decay = -8 * recurrent_gate * softplus(recurrent_param)
        // softplus(y) = log(exp(y) + 1)
        let rp = x.const_f32_like(
            Arc::clone(&rg.recurrent_param),
            Shape::from_dims(&[lru_width]),
        );
        let softplus_rp = rp.exp().add_scalar(1.0).log();
        // broadcast (lru_width) → (1, 1, lru_width)
        let softplus_rp_bc = softplus_rp
            .reshape(Shape::from_dims(&[1, 1, lru_width]))?
            .broadcast_to(Shape::from_dims(&[batch, seq, lru_width]))?;
        let log_decay = recurrent_gate.mul_scalar(-8.0).mul(&softplus_rp_bc)?;
        let decay = log_decay.exp();
        let a_square = log_decay.mul_scalar(2.0).exp();

        // gated_x = x * input_gate
        let gated_x = x.mul(&input_gate)?;
        // mult = reset + (1 - reset) * sqrt(1 - a_square)
        //   at t=0, reset=1 ⇒ mult=1
        //   at t>0, reset=0 ⇒ mult = sqrt(1 - a_square)
        // Build reset mask shape (1, seq, 1): [1.0, 0.0, 0.0, ...].
        let mut reset_data = vec![0.0_f32; seq];
        reset_data[0] = 1.0;
        let reset = x.const_f32_like(Arc::from(reset_data), Shape::from_dims(&[1, seq, 1]));
        let one_minus_reset = reset.mul_scalar(-1.0).add_scalar(1.0); // 1 - reset
        let one_minus_reset_bc =
            one_minus_reset.broadcast_to(Shape::from_dims(&[batch, seq, lru_width]))?;
        // 1 - a_square (clamp away from negatives via straight subtraction; in
        // valid range a_square ∈ (0, 1] so 1 - a_square ≥ 0).
        let one_minus_a_square = a_square.mul_scalar(-1.0).add_scalar(1.0);
        let sqrt_term = one_minus_a_square.sqrt();
        let reset_bc = reset.broadcast_to(Shape::from_dims(&[batch, seq, lru_width]))?;
        let mult = reset_bc.add(&one_minus_reset_bc.mul(&sqrt_term)?)?;
        let normalized_x = gated_x.mul(&mult)?;

        // Effective decay = decay * (1 - reset).
        let decay_eff = decay.mul(&one_minus_reset_bc)?;

        // Sequential recurrence:
        //   state[t] = decay_eff[t] * state[t-1] + normalized_x[t]
        // Stack states into (b, seq, lru_width). State starts at zeros.
        let mut state: Option<Tensor> = None;
        let mut out_steps: Vec<Tensor> = Vec::with_capacity(seq);
        for t in 0..seq {
            let x_t = normalized_x.slice(1_usize, t, 1)?; // (b, 1, lru_width)
            let d_t = decay_eff.slice(1_usize, t, 1)?; // (b, 1, lru_width)
            let new_state = match state {
                None => x_t,
                Some(s) => d_t.mul(&s)?.add(&x_t)?,
            };
            state = Some(new_state.clone());
            out_steps.push(new_state);
        }
        // Concat along seq axis.
        let mut all: Option<Tensor> = None;
        for step in out_steps.into_iter() {
            all = Some(match all {
                None => step,
                Some(s) => s.concat(&step, 1_usize)?,
            });
        }
        Ok(all.expect("at least one step"))
    }

    fn apply_mlp(&self, x: &Tensor, layer: &RecurrentGemmaLayerWeights) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let inter = cfg.mlp_intermediate();
        let gate = layer.mlp_gate_w.apply_linear_with_bias(
            x,
            h,
            inter,
            std::sync::Arc::clone(&layer.mlp_gate_b),
        )?;
        let up = layer.mlp_up_w.apply_linear_with_bias(
            x,
            h,
            inter,
            std::sync::Arc::clone(&layer.mlp_up_b),
        )?;
        let activated = match cfg.hidden_activation {
            GemmaActivation::Gelu => gate.gelu_erf(),
            GemmaActivation::GeluPytorchTanh => gate.gelu(),
        };
        let inner = activated.mul(&up)?;
        let down = layer.mlp_down_w.apply_linear(&inner, inter, h)?;
        down.add_trailing_bias(std::sync::Arc::clone(&layer.mlp_down_b))
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl RecurrentGemmaWeights {
    /// Load RecurrentGemma (google/recurrentgemma-*) weights from HF safetensors.
    /// Layer kind alternates between recurrent and attention per
    /// `cfg.block_types`. lm_head is tied to token embedding.
    pub fn load_from_mmapped(
        st: &fuel_core::safetensors::MmapedSafetensors,
        cfg: &RecurrentGemmaConfig,
    ) -> Result<Self> {
        use fuel_core::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype as ltm};
        let h = cfg.hidden_size;
        let inter = cfg.mlp_intermediate();
        let lru = cfg.lru_width_or_default();
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);

        let opt_bias = |name: String| -> Option<Arc<[f32]>> {
            load_tensor_as_f32(st, &name).ok().map(Arc::from)
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let temporal_pre_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.temporal_pre_norm.weight"),
            )?);
            let channel_pre_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.channel_pre_norm.weight"),
            )?);

            let temporal = match cfg.block_type(i) {
                TemporalBlockType::Attention => {
                    let tp = format!("{p}.temporal_block");
                    let q_w = ltm(st, &format!("{tp}.q_proj.weight"), q_dim, h)?;
                    let q_b = if cfg.attention_bias {
                        opt_bias(format!("{tp}.q_proj.bias"))
                    } else {
                        None
                    };
                    let k_w = ltm(st, &format!("{tp}.k_proj.weight"), kv_dim, h)?;
                    let k_b = if cfg.attention_bias {
                        opt_bias(format!("{tp}.k_proj.bias"))
                    } else {
                        None
                    };
                    let v_w = ltm(st, &format!("{tp}.v_proj.weight"), kv_dim, h)?;
                    let v_b = if cfg.attention_bias {
                        opt_bias(format!("{tp}.v_proj.bias"))
                    } else {
                        None
                    };
                    let o_w = ltm(st, &format!("{tp}.o_proj.weight"), h, q_dim)?;
                    let o_b = Arc::from(load_tensor_as_f32(st, &format!("{tp}.o_proj.bias"))?);
                    TemporalBlockWeights::Attention(AttentionBlockWeights {
                        q_w,
                        q_b,
                        k_w,
                        k_b,
                        v_w,
                        v_b,
                        o_w,
                        o_b,
                    })
                }
                TemporalBlockType::Recurrent => {
                    let tp = format!("{p}.temporal_block");
                    let linear_y_w = ltm(st, &format!("{tp}.linear_y.weight"), lru, h)?;
                    let linear_y_b =
                        Arc::from(load_tensor_as_f32(st, &format!("{tp}.linear_y.bias"))?);
                    let linear_x_w = ltm(st, &format!("{tp}.linear_x.weight"), lru, h)?;
                    let linear_x_b =
                        Arc::from(load_tensor_as_f32(st, &format!("{tp}.linear_x.bias"))?);
                    let linear_out_w = ltm(st, &format!("{tp}.linear_out.weight"), h, lru)?;
                    let linear_out_b =
                        Arc::from(load_tensor_as_f32(st, &format!("{tp}.linear_out.bias"))?);
                    let conv1d_w =
                        Arc::from(load_tensor_as_f32(st, &format!("{tp}.conv_1d.weight"))?);
                    let conv1d_b =
                        Arc::from(load_tensor_as_f32(st, &format!("{tp}.conv_1d.bias"))?);
                    let recurrent_param = Arc::from(load_tensor_as_f32(
                        st,
                        &format!("{tp}.rg_lru.recurrent_param"),
                    )?);
                    let input_gate_weight = Arc::from(load_tensor_as_f32(
                        st,
                        &format!("{tp}.rg_lru.input_gate_weight"),
                    )?);
                    let input_gate_bias = Arc::from(load_tensor_as_f32(
                        st,
                        &format!("{tp}.rg_lru.input_gate_bias"),
                    )?);
                    let recurrent_gate_weight = Arc::from(load_tensor_as_f32(
                        st,
                        &format!("{tp}.rg_lru.recurrent_gate_weight"),
                    )?);
                    let recurrent_gate_bias = Arc::from(load_tensor_as_f32(
                        st,
                        &format!("{tp}.rg_lru.recurrent_gate_bias"),
                    )?);
                    TemporalBlockWeights::Recurrent(RecurrentBlockWeights {
                        linear_y_w,
                        linear_y_b,
                        linear_x_w,
                        linear_x_b,
                        linear_out_w,
                        linear_out_b,
                        conv1d_w,
                        conv1d_b,
                        rg_lru: RgluWeights {
                            recurrent_param,
                            input_gate_weight,
                            input_gate_bias,
                            recurrent_gate_weight,
                            recurrent_gate_bias,
                        },
                    })
                }
            };

            let mp = format!("{p}.mlp_block");
            let mlp_gate_w = ltm(st, &format!("{mp}.gate_proj.weight"), inter, h)?;
            let mlp_gate_b = Arc::from(load_tensor_as_f32(st, &format!("{mp}.gate_proj.bias"))?);
            let mlp_up_w = ltm(st, &format!("{mp}.up_proj.weight"), inter, h)?;
            let mlp_up_b = Arc::from(load_tensor_as_f32(st, &format!("{mp}.up_proj.bias"))?);
            let mlp_down_w = ltm(st, &format!("{mp}.down_proj.weight"), h, inter)?;
            let mlp_down_b = Arc::from(load_tensor_as_f32(st, &format!("{mp}.down_proj.bias"))?);

            layers.push(RecurrentGemmaLayerWeights {
                temporal_pre_norm_gain,
                channel_pre_norm_gain,
                temporal,
                mlp_gate_w,
                mlp_gate_b,
                mlp_up_w,
                mlp_up_b,
                mlp_down_w,
                mlp_down_b,
            });
        }

        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.final_norm.weight")?);

        Ok(Self {
            token_embedding,
            layers,
            final_norm_gain,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &RecurrentGemmaConfig) -> RecurrentGemmaWeights {
        let mut s: u32 = 24681;
        let next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let lru = cfg.lru_width_or_default();
        let n_heads = cfg.num_attention_heads;
        let block_w = cfg.block_width();
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let inter = cfg.mlp_intermediate();
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<RecurrentGemmaLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|li| {
                let temporal = match cfg.block_type(li) {
                    TemporalBlockType::Attention => {
                        TemporalBlockWeights::Attention(AttentionBlockWeights {
                            q_w: WeightStorage::F32(vec_of(h * q_dim, &mut *nb)),
                            q_b: if cfg.attention_bias {
                                Some(vec_of(q_dim, &mut *nb))
                            } else {
                                None
                            },
                            k_w: WeightStorage::F32(vec_of(h * kv_dim, &mut *nb)),
                            k_b: if cfg.attention_bias {
                                Some(vec_of(kv_dim, &mut *nb))
                            } else {
                                None
                            },
                            v_w: WeightStorage::F32(vec_of(h * kv_dim, &mut *nb)),
                            v_b: if cfg.attention_bias {
                                Some(vec_of(kv_dim, &mut *nb))
                            } else {
                                None
                            },
                            o_w: WeightStorage::F32(vec_of(q_dim * h, &mut *nb)),
                            o_b: vec_of(h, &mut *nb),
                        })
                    }
                    TemporalBlockType::Recurrent => {
                        TemporalBlockWeights::Recurrent(RecurrentBlockWeights {
                            linear_y_w: WeightStorage::F32(vec_of(h * lru, &mut *nb)),
                            linear_y_b: vec_of(lru, &mut *nb),
                            linear_x_w: WeightStorage::F32(vec_of(h * lru, &mut *nb)),
                            linear_x_b: vec_of(lru, &mut *nb),
                            linear_out_w: WeightStorage::F32(vec_of(lru * h, &mut *nb)),
                            linear_out_b: vec_of(h, &mut *nb),
                            conv1d_w: vec_of(lru * cfg.conv1d_width, &mut *nb),
                            conv1d_b: vec_of(lru, &mut *nb),
                            rg_lru: RgluWeights {
                                recurrent_param: vec_of(lru, &mut *nb),
                                input_gate_weight: vec_of(n_heads * block_w * block_w, &mut *nb),
                                input_gate_bias: vec_of(n_heads * block_w, &mut *nb),
                                recurrent_gate_weight: vec_of(
                                    n_heads * block_w * block_w,
                                    &mut *nb,
                                ),
                                recurrent_gate_bias: vec_of(n_heads * block_w, &mut *nb),
                            },
                        })
                    }
                };
                RecurrentGemmaLayerWeights {
                    temporal_pre_norm_gain: Arc::from(vec![0.05_f32; h]),
                    channel_pre_norm_gain: Arc::from(vec![0.05_f32; h]),
                    temporal,
                    mlp_gate_w: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                    mlp_gate_b: vec_of(inter, &mut *nb),
                    mlp_up_w: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                    mlp_up_b: vec_of(inter, &mut *nb),
                    mlp_down_w: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
                    mlp_down_b: vec_of(h, &mut *nb),
                }
            })
            .collect();
        let final_norm_gain = Arc::from(vec![0.05_f32; h]);
        RecurrentGemmaWeights {
            token_embedding,
            layers,
            final_norm_gain,
        }
    }

    fn tiny_config() -> RecurrentGemmaConfig {
        RecurrentGemmaConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 3,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            lru_width: Some(8),
            attention_window_size: 8,
            conv1d_width: 4,
            logits_soft_cap: 30.0,
            hidden_activation: GemmaActivation::GeluPytorchTanh,
            partial_rotary_factor: 0.5,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            block_types: vec![
                TemporalBlockType::Recurrent,
                TemporalBlockType::Recurrent,
                TemporalBlockType::Attention,
            ],
            attention_bias: false,
            max_seq_len: 32,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = RecurrentGemmaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    #[test]
    fn single_token() {
        let cfg = tiny_config();
        let model = RecurrentGemmaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let logits = model.forward(&[3], 0).unwrap().realize_f32();
        assert_eq!(logits.len(), cfg.vocab_size);
    }

    /// Recurrent state propagates: first-token swap changes
    /// last-token output via the LRU recurrence.
    #[test]
    fn recurrent_state_propagates() {
        let cfg = RecurrentGemmaConfig {
            // Force layer 0 to be Recurrent so state actually flows.
            block_types: vec![TemporalBlockType::Recurrent],
            num_hidden_layers: 1,
            ..tiny_config()
        };
        let weights = tiny_weights(&cfg);
        let model = RecurrentGemmaModel {
            config: cfg.clone(),
            weights,
        };
        let a = model.forward(&[0, 5, 5, 5], 0).unwrap().realize_f32();
        let b = model.forward(&[7, 5, 5, 5], 0).unwrap().realize_f32();
        let last_a = &a[a.len() - cfg.vocab_size..];
        let last_b = &b[b.len() - cfg.vocab_size..];
        let mut max_diff = 0.0_f32;
        for (x, y) in last_a.iter().zip(last_b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        // Tiny-weight test (weights ∈ [-0.025, 0.025]) — the
        // recurrent contribution is real but small; we just
        // require it to be measurably non-zero.
        assert!(
            max_diff > 1e-8,
            "recurrent state must propagate first→last, max_diff = {max_diff}"
        );
    }

    /// Soft-cap on logits is wired: removing it changes output.
    #[test]
    fn logits_soft_cap_changes_output() {
        let cfg_a = RecurrentGemmaConfig {
            logits_soft_cap: 0.0,
            ..tiny_config()
        };
        let cfg_b = RecurrentGemmaConfig {
            logits_soft_cap: 5.0,
            ..tiny_config()
        };
        let weights = tiny_weights(&cfg_a);
        let m_a = RecurrentGemmaModel {
            config: cfg_a,
            weights: weights.clone(),
        };
        let m_b = RecurrentGemmaModel {
            config: cfg_b,
            weights,
        };
        let a = m_a.forward(&[1, 2, 3], 0).unwrap().realize_f32();
        let b = m_b.forward(&[1, 2, 3], 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(
            max_diff > 1e-6,
            "logits soft-cap must alter output, max_diff = {max_diff}"
        );
    }

    #[test]
    fn block_type_alternation() {
        let cfg = tiny_config();
        // block_types = [R, R, A], num_hidden_layers = 3
        assert_eq!(cfg.block_type(0), TemporalBlockType::Recurrent);
        assert_eq!(cfg.block_type(1), TemporalBlockType::Recurrent);
        assert_eq!(cfg.block_type(2), TemporalBlockType::Attention);
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config();
        let model = RecurrentGemmaModel {
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
        let model = RecurrentGemmaModel {
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
            max_diff < 1e-5,
            "RecurrentGemma forward vs forward_embeds must agree (max diff {max_diff})"
        );
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = tiny_config();
        let model = RecurrentGemmaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let bad = Tensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]),
            &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = tiny_config();
        let model = RecurrentGemmaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![5, 7];
        let h_ref = model.forward_hidden(&tokens, 0).unwrap().realize_f32();
        let anchor = Tensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu());
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let h_via_embeds = model
            .forward_hidden_embeds(&embeds, 0)
            .unwrap()
            .realize_f32();
        let max_diff = h_ref
            .iter()
            .zip(h_via_embeds.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "RecurrentGemma forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})"
        );
    }

    /// The REAL `google/recurrentgemma-2b/config.json`. The repo is GATED, so
    /// this was read through the authenticated Hub connector rather than
    /// transcribed from a description — which matters, because the dispatch's
    /// description of two of these fields was wrong and only the artifact
    /// settled it.
    const RG_HF_CONFIG_JSON: &str = r#"{
        "_block_types": ["recurrent", "recurrent", "attention"],
        "architectures": ["RecurrentGemmaForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attention_window_size": 2048,
        "conv1d_width": 4,
        "embeddings_scale_by_sqrt_dim": true,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 2560,
        "intermediate_size": 15360,
        "logits_soft_cap": 30.0,
        "lru_width": 2560,
        "model_type": "recurrent_gemma",
        "num_attention_heads": 10,
        "num_hidden_layers": 26,
        "num_key_value_heads": 1,
        "partial_rotary_factor": 0.5,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "vocab_size": 256000
    }"#;

    #[test]
    fn recurrent_gemma_config_from_hf_json_maps_the_artifact() {
        let cfg = RecurrentGemmaConfig::from_hf_json_str(RG_HF_CONFIG_JSON).unwrap();
        assert_eq!(cfg.vocab_size, 256_000);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.intermediate_size, 15360);
        assert_eq!(cfg.num_hidden_layers, 26);
        assert_eq!(cfg.num_attention_heads, 10);
        assert_eq!(cfg.num_key_value_heads, 1);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.lru_width, Some(2560));
        assert_eq!(cfg.conv1d_width, 4);
        assert_eq!(cfg.logits_soft_cap, 30.0);
        assert_eq!(cfg.partial_rotary_factor, 0.5);
        assert_eq!(cfg.rms_norm_eps, 1e-06);
        assert_eq!(cfg.rope_theta, 10000.0);
        assert_eq!(cfg.hidden_activation, GemmaActivation::GeluPytorchTanh);
        assert!(!cfg.attention_bias);
    }

    /// **`_block_types` is a CYCLE and is mapped VERBATIM — the dispatch said it
    /// needed per-layer expansion, and the code says otherwise.**
    ///
    /// `RecurrentGemmaConfig::block_type()` indexes
    /// `block_types[layer_idx % block_types.len()]`, and this module's own header
    /// documents it as a cycle. Expanding a 3-element pattern to 26 entries would
    /// produce the same answers for layers 0..25 and a DIFFERENT `len()`, so the
    /// error message at the weight-shape check would name a modulus that does not
    /// match the config. Verbatim is the faithful mapping.
    ///
    /// ⚠️ The per-layer SCHEDULE is asserted anyway — as the sequence
    /// `block_type()` produces — because that is the observable a wrong tiling
    /// would corrupt. An off-by-one or a repeat-instead-of-cycle fails here.
    #[test]
    fn recurrent_gemma_block_types_are_a_cycle_not_an_expansion() {
        let cfg = RecurrentGemmaConfig::from_hf_json_str(RG_HF_CONFIG_JSON).unwrap();
        use TemporalBlockType::{Attention as A, Recurrent as R};

        // Mapped verbatim: THREE entries, not twenty-six.
        assert_eq!(
            cfg.block_types,
            vec![R, R, A],
            "the pattern is stored as-is"
        );
        assert_eq!(
            cfg.block_types.len(),
            3,
            "must NOT be expanded to num_hidden_layers -- block_type() cycles"
        );

        // The observable: the full 26-layer schedule the cycle produces.
        let schedule: Vec<TemporalBlockType> = (0..cfg.num_hidden_layers)
            .map(|i| cfg.block_type(i))
            .collect();
        let expected = vec![
            R, R, A, R, R, A, R, R, A, // 0..8
            R, R, A, R, R, A, R, R, A, // 9..17
            R, R, A, R, R, A, R, R, // 18..25
        ];
        assert_eq!(schedule, expected, "26 layers from [R,R,A]");

        // Pinned as indices too: an off-by-one moves every one of these.
        let attn: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|i| cfg.block_type(*i) == TemporalBlockType::Attention)
            .collect();
        assert_eq!(attn, vec![2, 5, 8, 11, 14, 17, 20, 23]);
        assert_eq!(attn.len(), 8, "26 layers, every third from index 2");
    }

    /// An unknown block kind ERRORS rather than defaulting to either variant.
    ///
    /// Defaulting would give a model that runs and is wrong — the silent failure
    /// this whole program exists to prevent.
    #[test]
    fn recurrent_gemma_rejects_an_unknown_block_type() {
        let bad = RG_HF_CONFIG_JSON.replace("\"attention\"]", "\"mamba\"]");
        let err = RecurrentGemmaConfig::from_hf_json_str(&bad)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("mamba"),
            "the unsupported value must be NAMED, got: {err}"
        );

        let empty =
            RG_HF_CONFIG_JSON.replace("[\"recurrent\", \"recurrent\", \"attention\"]", "[]");
        assert!(
            RecurrentGemmaConfig::from_hf_json_str(&empty).is_err(),
            "an empty cycle would make block_type() divide by zero"
        );
    }

    /// An unknown activation ERRORS rather than defaulting.
    #[test]
    fn recurrent_gemma_rejects_an_unknown_activation() {
        let bad = RG_HF_CONFIG_JSON.replace("gelu_pytorch_tanh", "swiglu");
        let err = RecurrentGemmaConfig::from_hf_json_str(&bad)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("swiglu"),
            "must name the activation, got: {err}"
        );
    }

    /// **`attention_window_size` is its own field and is NOT `max_seq_len`.**
    ///
    /// The dispatch described `attention_window_size` as the source for
    /// `max_seq_len`. They are separate fields on the struct with different
    /// meanings — the window is read at the attention site; `max_seq_len` has no
    /// counterpart in the artifact at all.
    ///
    /// This pins the distinction so a later "simplification" that folds one into
    /// the other has to delete an assertion that says why not.
    #[test]
    fn recurrent_gemma_attention_window_is_read_as_itself() {
        let cfg = RecurrentGemmaConfig::from_hf_json_str(RG_HF_CONFIG_JSON).unwrap();
        assert_eq!(
            cfg.attention_window_size, 2048,
            "read verbatim from the artifact"
        );
        // max_seq_len is DERIVED from it (documented in `resolve`), so changing
        // the window moves both -- which is exactly what a reader should be able
        // to see, rather than inferring that the artifact supplied a max length.
        let narrower = RG_HF_CONFIG_JSON.replace(
            "\"attention_window_size\": 2048",
            "\"attention_window_size\": 512",
        );
        let cfg2 = RecurrentGemmaConfig::from_hf_json_str(&narrower).unwrap();
        assert_eq!(cfg2.attention_window_size, 512);
        assert_eq!(
            cfg2.max_seq_len, 512,
            "max_seq_len has NO HF counterpart and is set to the window -- if this \
             ever becomes load-bearing it is a decision to re-make, not a fact read \
             from the artifact"
        );
    }
}

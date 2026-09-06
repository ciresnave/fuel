// SPDX-License-Identifier: MIT OR Apache-2.0
//! LLaMA-3 long-context RoPE scaling on top of the lazy
//! [`LlamaModel`](fuel_core::lazy::LlamaModel).
//!
//! `lazy::LlamaModel` is the canonical lazy-graph LLaMA decoder used
//! by LLaVA, anchor-oracle tests, and the rest of the LLaMA-family
//! lazy ports. It handles standard (unscaled) RoPE via
//! `rope_base = cfg.rope_theta` and a uniform `theta^(-2i/d)`
//! frequency table.
//!
//! LLaMA-3.1 introduced a long-context RoPE scaling scheme that
//! splits the rotary frequencies into three bands:
//!
//!   - wavelen `<` high_freq_wavelen → unscaled
//!   - wavelen `>` low_freq_wavelen → divided by `factor`
//!   - in between → smoothed interpolation
//!
//! where `*_freq_wavelen = original_max_position_embeddings / *_freq_factor`.
//!
//! This module adds:
//!   - [`Llama3RopeConfig`] / [`Llama3RopeType`] — HF-shape config types.
//!   - [`LlamaEosToks`] — single-or-multiple EOS token id enum (LLaMA-3
//!     ships three EOS tokens).
//!   - [`LlamaFullConfig`] — full HF `config.json` shape (rope_scaling,
//!     eos_token_id, tie_word_embeddings, etc.) with a serde-driven
//!     [`from_hf_json_str`](LlamaFullConfig::from_hf_json_str)
//!     deserializer.
//!   - [`build_llama3_rope_tables`] — host-side RoPE cos/sin builder
//!     that applies the three-band scaling when scaling is present and
//!     reduces to the standard unscaled tables when it's `None`.
//!   - [`Llama3Model`] — a thin wrapper over `lazy::LlamaModel` that
//!     injects scaled RoPE tables into the standard backbone via
//!     `LlamaModel::run_backbone_with_rope_tables`.
//!
//! The unscaled forward path goes through `lazy::LlamaModel::forward`
//! directly — `Llama3Model` only adds value when `rope_scaling.is_some()`.
//! That's intentional: `Llama3Model::forward` produces bit-for-bit
//! identical output to `LlamaModel::forward` when scaling is absent,
//! so users can keep a single code path regardless of model variant.
//!
//! ## EOS tokens
//!
//! `LlamaEosToks` carries the tokens; it does not enforce them.
//! Generation loops (Lightbulb) consult
//! [`LlamaEosToks::is_eos`](LlamaEosToks::is_eos) per token.

use fuel_core::lazy::{LlamaConfig, LlamaModel, LlamaWeights, Tensor};
use fuel_core::{Error, Result};
use fuel_ir::Shape;
use serde::Deserialize;
use std::f64::consts::PI;
use std::sync::Arc;

/// Selects the RoPE scaling algorithm. `Default` is plain unscaled
/// RoPE; `Llama3` is the LLaMA-3.1 three-band long-context scaling.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Llama3RopeType {
    /// LLaMA-3.1 long-context three-band frequency scaling.
    Llama3,
    /// Standard unscaled RoPE (default).
    #[default]
    Default,
}

/// LLaMA-3.1 long-context RoPE scaling parameters.
///
/// All fields are HF `rope_scaling` keys verbatim:
///   - `factor`: global divisor for low-frequency rotations.
///   - `low_freq_factor` / `high_freq_factor`: define the band
///     boundaries via `wavelen_band = original_max_pos / *_freq_factor`.
///   - `original_max_position_embeddings`: pretraining max seq length.
///   - `rope_type`: which scaling algorithm to use.
///
/// LLaMA-3.1-8B canonical values: `factor=8.0`,
/// `low_freq_factor=1.0`, `high_freq_factor=4.0`,
/// `original_max_position_embeddings=8192`, `rope_type=Llama3`.
#[derive(Debug, Clone, PartialEq)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Llama3RopeType,
}

/// End-of-sequence token id(s) for a LLaMA model.
///
/// HF `eos_token_id` is either a single integer (`Single(t)`) or a
/// list (`Multiple(vec)`). LLaMA-3.1 ships `[128001, 128008, 128009]`.
#[derive(Debug, Clone, PartialEq)]
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

impl LlamaEosToks {
    /// `true` if `tok` matches any of the EOS tokens this enum carries.
    pub fn is_eos(&self, tok: u32) -> bool {
        match self {
            Self::Single(s) => *s == tok,
            Self::Multiple(v) => v.contains(&tok),
        }
    }
}

/// Full HuggingFace `config.json` shape for LLaMA family models
/// (LLaMA-1 / LLaMA-2 / LLaMA-3.x / Mistral-shape-LLaMAs).
///
/// Use [`from_hf_json_str`](Self::from_hf_json_str) to parse a
/// downloaded `config.json`, then [`to_lazy_config`](Self::to_lazy_config)
/// to obtain the lower-level [`LlamaConfig`] the model's forward
/// path consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaFullConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub tie_word_embeddings: bool,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_max_position_embeddings() -> usize {
    4096
}

/// HF's `eos_token_id` is either a single integer or a list, and anything
/// else is treated as absent rather than as an error.
///
/// Kept as a hand parse rather than `#[serde(untagged)]` because the two are
/// not equivalent: `untagged` REJECTS a third shape, and this accepts it as
/// `None`. Preserving that tolerance is the point — the differential compares
/// against the parser that shipped, not against a stricter one.
fn parse_eos_token_id(v: &serde_json::Value) -> Option<LlamaEosToks> {
    if let Some(n) = v.as_u64() {
        Some(LlamaEosToks::Single(n as u32))
    } else if let Some(arr) = v.as_array() {
        let vec: Option<Vec<u32>> = arr.iter().map(|e| e.as_u64().map(|n| n as u32)).collect();
        vec.map(LlamaEosToks::Multiple)
    } else {
        None
    }
}

/// A Llama `config.json` as HuggingFace ships it.
///
/// ⚠️ TWO FIELDS ARE HELD AS RAW `serde_json::Value` ON PURPOSE, and this is
/// the first config in the set that needs it. `eos_token_id` and
/// `rope_scaling` are parsed by functions that are DELIBERATELY TOLERANT:
/// `parse_llama3_rope_scaling` returns `None` if any field is missing,
/// accepts `rope_type` OR `type` as the key, and falls back to
/// `Llama3RopeType::Default` on an unrecognised value. Every one of those is
/// a hard error under `#[derive(Deserialize)]`.
///
/// Making them typed here would be a behaviour change wearing a refactor's
/// clothes: configs that parse today would start failing. `serde` handles the
/// flat fields; the tolerant shapes stay with the code that was written to be
/// tolerant.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct LlamaFullConfigRaw {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    tie_word_embeddings: bool,
}

impl LlamaFullConfigRaw {
    fn from_json_str(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            Error::Msg(format!(
                "LlamaFullConfig::from_hf_json_str: parsing config.json: {e}"
            ))
        })
    }

    fn resolve(self) -> Result<LlamaFullConfig> {
        Ok(LlamaFullConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
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
            )?,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id.as_ref().and_then(parse_eos_token_id),
            rope_scaling: self
                .rope_scaling
                .as_ref()
                .and_then(parse_llama3_rope_scaling),
            tie_word_embeddings: self.tie_word_embeddings,
        })
    }
}

impl LlamaFullConfig {
    /// Parse HF `config.json` text. Missing optional fields fall back
    /// to LLaMA's documented defaults (`rope_theta=10000.0`,
    /// `rms_norm_eps=1e-5`, `tie_word_embeddings=false`, no scaling).
    ///
    /// [`LlamaFullConfigRaw`] is the wire shape; [`LlamaFullConfigRaw::resolve`]
    /// applies the two cross-field defaults and the two TOLERANT sub-object
    /// parses that `serde` deliberately is not asked to perform.
    pub fn from_hf_json_str(json: &str) -> Result<Self> {
        LlamaFullConfigRaw::from_json_str(json)?.resolve()
    }

    /// Convert to the lower-level [`LlamaConfig`] consumed by the model
    /// forward path. Drops the HF-only fields (eos, bos, tie_word_embeddings,
    /// max_position_embeddings) — those are carried separately by
    /// [`Llama3Model`].
    pub fn to_lazy_config(&self) -> LlamaConfig {
        LlamaConfig {
            vocab_size: self.vocab_size,
            dim: self.hidden_size,
            n_layers: self.num_hidden_layers,
            n_heads: self.num_attention_heads,
            n_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            ffn_dim: self.intermediate_size,
            norm_eps: self.rms_norm_eps,
            rope_base: self.rope_theta,
        }
    }
}

fn parse_llama3_rope_scaling(v: &serde_json::Value) -> Option<Llama3RopeConfig> {
    let factor = v.get("factor").and_then(|x| x.as_f64())? as f32;
    let low_freq_factor = v.get("low_freq_factor").and_then(|x| x.as_f64())? as f32;
    let high_freq_factor = v.get("high_freq_factor").and_then(|x| x.as_f64())? as f32;
    let original_max_position_embeddings = v
        .get("original_max_position_embeddings")
        .and_then(|x| x.as_u64())? as usize;
    let rope_type = match v.get("rope_type").and_then(|x| x.as_str()) {
        Some("llama3") => Llama3RopeType::Llama3,
        // HF sometimes carries `type` instead of `rope_type`.
        None => match v.get("type").and_then(|x| x.as_str()) {
            Some("llama3") => Llama3RopeType::Llama3,
            _ => Llama3RopeType::Default,
        },
        _ => Llama3RopeType::Default,
    };
    Some(Llama3RopeConfig {
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_position_embeddings,
        rope_type,
    })
}

/// Build interleaved-pairs RoPE cos/sin tables for `seq` positions
/// starting at `start_pos`, applying LLaMA-3.1 three-band scaling
/// when `scaling.is_some() && rope_type == Llama3`.
///
/// Returns `(cos, sin)` each laid out as `[seq, head_dim]` in row-
/// major order — the same layout the lazy graph's `rope_with_tables`
/// expects (and what `fuel_graph::build_rope_tables` produces for
/// the unscaled case).
///
/// `head_dim` must be even.
pub fn build_llama3_rope_tables(
    rope_base: f64,
    scaling: Option<&Llama3RopeConfig>,
    start_pos: usize,
    seq: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    // GAP-281: a CONFIG property -- reject it rather than panicking. The
    // half-dimension split below assumes an even head_dim.
    if !head_dim.is_multiple_of(2) {
        return Err(Error::Msg(format!(
            "build_llama3_rope_tables: head_dim ({head_dim}) must be even"
        )));
    }
    let inv_freq = compute_llama3_inv_freq(rope_base, scaling, head_dim);
    let half = head_dim / 2;
    let mut cos_data = vec![0.0_f32; seq * head_dim];
    let mut sin_data = vec![0.0_f32; seq * head_dim];
    for p in 0..seq {
        let pos = (start_pos + p) as f64;
        for i in 0..half {
            let theta = pos * inv_freq[i];
            let c = theta.cos() as f32;
            let s = theta.sin() as f32;
            cos_data[p * head_dim + i] = c;
            cos_data[p * head_dim + i + half] = c;
            sin_data[p * head_dim + i] = s;
            sin_data[p * head_dim + i + half] = s;
        }
    }
    Ok((cos_data, sin_data))
}

fn compute_llama3_inv_freq(
    rope_base: f64,
    scaling: Option<&Llama3RopeConfig>,
    head_dim: usize,
) -> Vec<f64> {
    let half = head_dim / 2;
    let base: Vec<f64> = (0..half)
        .map(|i| rope_base.powf(-2.0 * (i as f64) / (head_dim as f64)))
        .collect();

    match scaling {
        None
        | Some(Llama3RopeConfig {
            rope_type: Llama3RopeType::Default,
            ..
        }) => base,
        Some(cfg) => {
            let orig = cfg.original_max_position_embeddings as f64;
            let low_freq_wavelen = orig / cfg.low_freq_factor as f64;
            let high_freq_wavelen = orig / cfg.high_freq_factor as f64;
            base.into_iter()
                .map(|freq| {
                    let wavelen = 2.0 * PI / freq;
                    if wavelen < high_freq_wavelen {
                        freq
                    } else if wavelen > low_freq_wavelen {
                        freq / cfg.factor as f64
                    } else {
                        let smooth = (orig / wavelen - cfg.low_freq_factor as f64)
                            / (cfg.high_freq_factor as f64 - cfg.low_freq_factor as f64);
                        (1.0 - smooth) * freq / cfg.factor as f64 + smooth * freq
                    }
                })
                .collect()
        }
    }
}

/// LLaMA model with optional LLaMA-3.1 long-context RoPE scaling.
///
/// Wraps [`fuel_core::lazy::LlamaModel`] and injects scaled RoPE tables
/// into the standard backbone via
/// [`LlamaModel::run_backbone_with_rope_tables`]. When
/// `rope_scaling.is_none()` the forward path produces bit-for-bit
/// identical output to `LlamaModel::forward`.
#[derive(Debug, Clone)]
pub struct Llama3Model {
    pub inner: LlamaModel,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub eos_token_id: Option<LlamaEosToks>,
}

impl Llama3Model {
    /// Construct from an existing [`LlamaModel`] plus optional scaling +
    /// EOS metadata. The model's `cfg.rope_base` stays unchanged —
    /// scaling is applied on top.
    pub fn new(
        inner: LlamaModel,
        rope_scaling: Option<Llama3RopeConfig>,
        eos_token_id: Option<LlamaEosToks>,
    ) -> Self {
        Self {
            inner,
            rope_scaling,
            eos_token_id,
        }
    }

    /// This model's RoPE inverse frequencies — `config.rope_base`'s defaults
    /// when unscaled, the LLaMA-3.1 long-context transform when scaled.
    ///
    /// The scaling is *entirely* a per-dimension transform of this vector (see
    /// [`compute_llama3_inv_freq`]), which is why the decode entry points below
    /// are thin calls into [`LlamaModel`]'s own path with the vector passed
    /// down, rather than a parallel implementation of it.
    fn rope_inv_freq(&self) -> Vec<f64> {
        compute_llama3_inv_freq(
            self.inner.config.rope_base,
            self.rope_scaling.as_ref(),
            self.inner.config.head_dim,
        )
    }

    /// Incremental KV-cache forward — the decode surface this model was missing
    /// entirely. `new`/`forward`/`forward_embeds`/`forward_hidden_embeds` were
    /// the whole method set, which meant serving a GGUF checkpoint through
    /// [`crate::models::lazy_quantized_llama::QuantizedLlama3Model`] re-ran the full
    /// prefix for every token.
    ///
    /// Delegates to [`LlamaModel::forward_with_kv_context_inv_freq`] with this
    /// model's frequencies, so a *scaled* model decodes correctly rather than
    /// silently inheriting unscaled RoPE — the failure a plain delegation to
    /// `self.inner` would have produced, and one that shows up as slightly
    /// wrong long-context attention rather than an error.
    pub fn forward_with_kv_context(
        &self,
        tokens: &[u32],
        cache: &mut fuel_core::inference_context::KvCache,
        ctx: &mut fuel_core::inference_context::InferenceContext,
    ) -> Result<Vec<f32>> {
        let inv = self.rope_inv_freq();
        self.inner
            .forward_with_kv_context_inv_freq(tokens, cache, ctx, Some(&inv))
    }

    /// Persistent-KV forward: build + optimize the decode graph once, then
    /// rebind it per token.
    ///
    /// **This is the variant that makes the model servable**, not
    /// [`Self::forward_with_kv_context`]. The non-persistent arm re-optimizes
    /// the graph every token; the Lightbulb port measured 5,901 ms/token
    /// against 26.47 ms/token with a caller-held `DecodeSession` on the same
    /// hardware (CUDA, TinyLlama-1.1B f32, debug). A KV cache without the held
    /// plan closes the asymptotic gap and leaves the constant that dominates.
    pub fn forward_with_kv_context_persistent(
        &self,
        tokens: &[u32],
        cache: &mut fuel_core::inference_context::KvCache,
        ctx: &mut fuel_core::inference_context::InferenceContext,
        session: &mut Option<fuel_core::inference_context::DecodeSession>,
    ) -> Result<Vec<f32>> {
        let inv = self.rope_inv_freq();
        self.inner.forward_with_kv_context_persistent_inv_freq(
            tokens,
            cache,
            ctx,
            session,
            Some(&inv),
        )
    }

    /// Forward from token ids. Returns logits `[1, seq, vocab_size]`.
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let cfg = &self.inner.config;
        let weights = &self.inner.weights;
        let seq = tokens.len();
        assert!(seq > 0, "Llama3Model::forward: tokens must be non-empty");

        let embed = Tensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &fuel_core::Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let h = embed
            .index_select(0, &token_ids)?
            .reshape(Shape::from_dims(&[1, seq, cfg.dim]))?;
        self.forward_embeds(&h, start_pos)
    }

    /// Forward from pre-computed embeddings `[batch, seq, dim]`.
    pub fn forward_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let cfg = &self.inner.config;
        let weights = &self.inner.weights;
        let h_norm = self.run_backbone_embeds(embeds, start_pos)?;
        weights
            .output
            .apply_linear(&h_norm, cfg.dim, cfg.vocab_size)
    }

    /// Forward from pre-computed embeddings; skip the LM head and
    /// return post-final-RMSNorm hidden states `[batch, seq, dim]`.
    pub fn forward_hidden_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        self.run_backbone_embeds(embeds, start_pos)
    }

    fn run_backbone_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let cfg = &self.inner.config;
        let dims = embeds.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "embeds must be rank 3 [b, seq, dim]");
        let seq = dims[1];
        assert_eq!(dims[2], cfg.dim);

        let (cos_data, sin_data) = build_llama3_rope_tables(
            cfg.rope_base,
            self.rope_scaling.as_ref(),
            start_pos,
            seq,
            cfg.head_dim,
        )?;
        let rope_shape = Shape::from_dims(&[seq, cfg.head_dim]);
        let rope_cos = embeds.const_f32_like(Arc::from(cos_data), rope_shape.clone());
        let rope_sin = embeds.const_f32_like(Arc::from(sin_data), rope_shape);

        let mask = Tensor::additive_causal_mask_like(embeds, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;

        self.inner
            .run_backbone_with_rope_tables(embeds, &rope_cos, &rope_sin, &mask)
    }
}

/// Construct a tied-embedding [`LlamaWeights::output`] when the HF
/// `tie_word_embeddings` flag is set. The lm_head's weight is the
/// transposed token embedding matrix; this helper materializes that
/// view as a flat F32 [`fuel_core::lazy::WeightStorage`].
///
/// For untied checkpoints the safetensors loader produces a separate
/// `lm_head.weight` tensor — this helper isn't used in that path.
pub fn tied_lm_head_from_embeddings(
    token_embedding: &Arc<[f32]>,
    vocab_size: usize,
    hidden_size: usize,
) -> fuel_core::lazy::WeightStorage {
    // `apply_linear`'s convention is weight shape `[in_features,
    // out_features]` for a `(B, T, in) → (B, T, out)` projection.
    // Token embedding is stored as `[vocab_size, hidden]` i.e.
    // `[out_features, in_features]`. The lm_head wants
    // `[hidden, vocab_size]` i.e. `[in, out]` — same data when
    // transposed. Since `apply_linear` (via WeightStorage::F32)
    // emits the matmul with the weight in `[in, out]` orientation,
    // and the embedding lookup uses `index_select` (which works on
    // either orientation), we can reuse the embedding bytes ONLY by
    // performing a transpose copy. There's no zero-copy shortcut
    // because the matmul kernel needs contiguous `[in, out]`.
    debug_assert_eq!(token_embedding.len(), vocab_size * hidden_size);
    let mut out = vec![0.0_f32; hidden_size * vocab_size];
    for v in 0..vocab_size {
        for h in 0..hidden_size {
            out[h * vocab_size + v] = token_embedding[v * hidden_size + h];
        }
    }
    fuel_core::lazy::WeightStorage::F32(Arc::from(out))
}

/// Build a complete [`Llama3Model`] from parsed HF config + weights.
///
/// Convenience constructor for the common case: caller has already
/// produced [`LlamaWeights`] via a safetensors loader and a parsed
/// [`LlamaFullConfig`] from `config.json`. The output projection is
/// expected to be in `weights.output` — if `tie_word_embeddings` is
/// set and the safetensors file did not ship a separate `lm_head`,
/// callers can use [`tied_lm_head_from_embeddings`] to materialize
/// the tied projection before calling this.
pub fn build_llama3_model(cfg: &LlamaFullConfig, weights: LlamaWeights) -> Llama3Model {
    let inner = LlamaModel {
        config: cfg.to_lazy_config(),
        weights,
    };
    Llama3Model::new(inner, cfg.rope_scaling.clone(), cfg.eos_token_id.clone())
}

#[cfg(test)]
impl LlamaFullConfig {
    /// The hand-rolled parser this type used before the `serde` split, kept
    /// verbatim and test-only as the differential's oracle.
    pub(crate) fn from_hf_json_str_legacy(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            Error::Msg(format!(
                "LlamaFullConfig::from_hf_json_str: parsing config.json: {e}"
            ))
        })?;

        let get_usize = |key: &str| -> Result<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| Error::Msg(format!("config.json: missing/invalid field {key:?}")))
        };
        let opt_usize = |key: &str| -> Option<usize> {
            v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
        };
        let opt_u32 =
            |key: &str| -> Option<u32> { v.get(key).and_then(|x| x.as_u64()).map(|x| x as u32) };
        let opt_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };
        let opt_bool = |key: &str| -> Option<bool> { v.get(key).and_then(|x| x.as_bool()) };

        let hidden_size = get_usize("hidden_size")?;
        let num_attention_heads = get_usize("num_attention_heads")?;
        let num_key_value_heads = opt_usize("num_key_value_heads").unwrap_or(num_attention_heads);
        let head_dim = opt_usize("head_dim").unwrap_or(hidden_size / num_attention_heads);

        // eos_token_id is either a u64 or an array of u64.
        let eos_token_id = v.get("eos_token_id").and_then(|x| {
            if let Some(n) = x.as_u64() {
                Some(LlamaEosToks::Single(n as u32))
            } else if let Some(arr) = x.as_array() {
                let vec: Option<Vec<u32>> =
                    arr.iter().map(|e| e.as_u64().map(|n| n as u32)).collect();
                vec.map(LlamaEosToks::Multiple)
            } else {
                None
            }
        });

        let rope_scaling = v.get("rope_scaling").and_then(parse_llama3_rope_scaling);

        Ok(Self {
            hidden_size,
            intermediate_size: get_usize("intermediate_size")?,
            vocab_size: get_usize("vocab_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps: opt_f64("rms_norm_eps").unwrap_or(1e-5),
            rope_theta: opt_f64("rope_theta").unwrap_or(10_000.0),
            max_position_embeddings: opt_usize("max_position_embeddings").unwrap_or(4096),
            bos_token_id: opt_u32("bos_token_id"),
            eos_token_id,
            rope_scaling,
            tie_word_embeddings: opt_bool("tie_word_embeddings").unwrap_or(false),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core::lazy::{LayerWeights, WeightStorage};

    fn tiny_weights(cfg: &LlamaConfig) -> LlamaWeights {
        let mut s: u32 = 12345;
        let next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let h = cfg.dim;
        let i = cfg.ffn_dim;
        let kv = cfg.n_kv_heads * cfg.head_dim;
        let buf = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let mut next_box: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = buf(cfg.vocab_size * h, &mut *next_box);
        let layers: Vec<LayerWeights> = (0..cfg.n_layers)
            .map(|_| LayerWeights {
                attn_q: WeightStorage::F32(buf(h * h, &mut *next_box)),
                attn_q_bias: None,
                attn_k: WeightStorage::F32(buf(h * kv, &mut *next_box)),
                attn_k_bias: None,
                attn_v: WeightStorage::F32(buf(h * kv, &mut *next_box)),
                attn_v_bias: None,
                attn_o: WeightStorage::F32(buf(h * h, &mut *next_box)),
                ffn_gate: WeightStorage::F32(buf(h * i, &mut *next_box)),
                ffn_up: WeightStorage::F32(buf(h * i, &mut *next_box)),
                ffn_down: WeightStorage::F32(buf(i * h, &mut *next_box)),
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(buf(h * cfg.vocab_size, &mut *next_box));
        LlamaWeights {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            final_norm_gain,
            output,
        }
    }

    fn tiny_cfg() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
        }
    }

    /// Without `rope_scaling`, `Llama3Model::forward` must produce
    /// bit-for-bit identical logits to the underlying `LlamaModel::forward`
    /// — the wrapper path uses the same RoPE base and the same
    /// strict-causal mask the backbone uses internally.
    #[test]
    fn unscaled_matches_inner_forward() {
        let cfg = tiny_cfg();
        let weights = tiny_weights(&cfg);
        let inner = LlamaModel {
            config: cfg.clone(),
            weights,
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let baseline = inner.forward(&tokens, 0).unwrap().realize_f32();

        let wrapped = Llama3Model::new(inner, None, None);
        let scaled = wrapped.forward(&tokens, 0).unwrap().realize_f32();

        assert_eq!(baseline.len(), scaled.len());
        for (i, (a, b)) in baseline.iter().zip(scaled.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "logits[{i}]: baseline={a}, scaled={b}, diff={}",
                (a - b).abs(),
            );
        }
    }

    /// With LLaMA-3.1 scaling, the RoPE inv-freq table must differ
    /// from the unscaled baseline on at least the low-freq band.
    #[test]
    fn llama3_scaling_changes_inv_freq() {
        // Pick a head_dim large enough that the lowest-frequency
        // wavelength exceeds `low_freq_wavelen` and triggers division.
        let head_dim = 128;
        let rope_base = 500_000.0_f64; // LLaMA-3's base
        let scaling = Llama3RopeConfig {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
            rope_type: Llama3RopeType::Llama3,
        };

        let plain = compute_llama3_inv_freq(rope_base, None, head_dim);
        let scaled = compute_llama3_inv_freq(rope_base, Some(&scaling), head_dim);

        assert_eq!(plain.len(), scaled.len());
        // Lowest-frequency entries get divided by factor.
        let last = head_dim / 2 - 1;
        assert!(
            (scaled[last] - plain[last] / 8.0).abs() < 1e-12,
            "lowest-freq band should be divided by factor=8: plain={}, scaled={}",
            plain[last],
            scaled[last],
        );
        // Highest-frequency entries (i=0) — wavelen = 2*PI/1.0 = 6.28 —
        // are below high_freq_wavelen = 8192/4 = 2048 → unscaled.
        assert!(
            (plain[0] - scaled[0]).abs() < 1e-12,
            "highest-freq band should be unscaled: plain={}, scaled={}",
            plain[0],
            scaled[0],
        );
    }

    /// With `rope_type=Default`, scaling should be a no-op even if
    /// the other fields are filled in.
    #[test]
    fn llama3_default_rope_type_is_noop() {
        let head_dim = 128;
        let scaling = Llama3RopeConfig {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
            rope_type: Llama3RopeType::Default,
        };
        let plain = compute_llama3_inv_freq(500_000.0, None, head_dim);
        let scaled = compute_llama3_inv_freq(500_000.0, Some(&scaling), head_dim);
        for i in 0..(head_dim / 2) {
            assert!(
                (plain[i] - scaled[i]).abs() < 1e-12,
                "rope_type=Default should not scale: i={i}",
            );
        }
    }

    /// Smoke: scaled-RoPE forward produces finite logits of the
    /// expected shape on a tiny config. Doesn't validate numerical
    /// match against eager — scaled-RoPE numerics are exercised by
    /// `llama3_scaling_changes_inv_freq`.
    #[test]
    fn scaled_forward_shape_and_finite() {
        let cfg = tiny_cfg();
        let weights = tiny_weights(&cfg);
        let inner = LlamaModel {
            config: cfg.clone(),
            weights,
        };
        let scaling = Llama3RopeConfig {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
            rope_type: Llama3RopeType::Llama3,
        };
        let model = Llama3Model::new(inner, Some(scaling), None);
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// Round-trip a realistic LLaMA-3.1-8B `config.json` fragment.
    /// Every `config.json` fixture this module exercises, as ONE corpus.
    ///
    /// **Corpus: 5 fixtures — 2 pre-existing, 3 added.** Per-rule coverage,
    /// stated because it travels neither between configs nor between rules:
    ///
    /// | rule / behaviour        | discriminated by a PRE-EXISTING fixture?          |
    /// |-------------------------|--------------------------------------------------|
    /// | `num_key_value_heads`   | YES — llama3.1-8b ships 8 against 32 heads        |
    /// | `head_dim`              | NO — llama3.1-8b ships 128 against a quotient of 4096/32 = 128, so the axis is COLLAPSED, and llama2 omits it entirely. Fixture 3 added. |
    /// | `eos_token_id` scalar   | YES — llama2 ships `2`                            |
    /// | `eos_token_id` array    | YES — llama3.1-8b ships `[128001, 128008, 128009]`|
    /// | rope_scaling TOLERANCE  | NO — nothing exercised a malformed or alternately-keyed block. Fixtures 4 and 5 added. |
    ///
    /// A COLLAPSED axis is the dangerous one: the fixture exercises the rule
    /// and cannot tell a correct implementation from one that ignores it,
    /// because the two answers coincide.
    const DIFFERENTIAL_CORPUS: &[(&str, &str)] = &[
        (
            "llama3.1-8b (kv_heads 8 != 32 discriminates; head_dim 128 == 4096/32 does NOT)",
            r#"{
                "architectures": ["LlamaForCausalLM"], "bos_token_id": 128000,
                "eos_token_id": [128001, 128008, 128009],
                "hidden_size": 4096, "intermediate_size": 14336,
                "num_attention_heads": 32, "num_hidden_layers": 32,
                "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 128256,
                "max_position_embeddings": 131072, "rms_norm_eps": 1.0e-5,
                "rope_theta": 500000.0, "tie_word_embeddings": false,
                "rope_scaling": {
                    "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192, "rope_type": "llama3"
                }
            }"#,
        ),
        (
            "llama2 (no GQA, no head_dim, no rope_scaling, scalar eos)",
            r#"{
                "architectures": ["LlamaForCausalLM"], "bos_token_id": 1,
                "eos_token_id": 2, "hidden_size": 4096, "intermediate_size": 11008,
                "num_attention_heads": 32, "num_hidden_layers": 32, "vocab_size": 32000,
                "max_position_embeddings": 4096, "rms_norm_eps": 1.0e-6,
                "tie_word_embeddings": false
            }"#,
        ),
        (
            "ADDED: explicit head_dim DISAGREEING with hidden_size/num_attention_heads",
            r#"{
                "hidden_size": 4096, "intermediate_size": 11008,
                "num_attention_heads": 32, "num_hidden_layers": 32,
                "vocab_size": 32000, "head_dim": 96
            }"#,
        ),
        (
            "ADDED: rope_scaling MISSING a field - must yield None, not an error",
            r#"{
                "hidden_size": 4096, "intermediate_size": 11008,
                "num_attention_heads": 32, "num_hidden_layers": 32, "vocab_size": 32000,
                "rope_scaling": { "factor": 8.0, "rope_type": "llama3" }
            }"#,
        ),
        (
            "ADDED: rope_scaling keyed `type` instead of `rope_type` (HF ships both)",
            r#"{
                "hidden_size": 4096, "intermediate_size": 11008,
                "num_attention_heads": 32, "num_hidden_layers": 32, "vocab_size": 32000,
                "rope_scaling": {
                    "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192, "type": "llama3"
                }
            }"#,
        ),
    ];

    #[test]
    fn serde_path_agrees_with_the_legacy_parser_on_every_fixture() {
        assert_eq!(DIFFERENTIAL_CORPUS.len(), 5, "corpus shrank");
        for (name, json) in DIFFERENTIAL_CORPUS {
            let new = LlamaFullConfig::from_hf_json_str(json);
            let old = LlamaFullConfig::from_hf_json_str_legacy(json);
            match (new, old) {
                (Ok(a), Ok(b)) => assert_eq!(a, b, "differential mismatch on {name}"),
                (Err(_), Err(_)) => {}
                (Ok(_), Err(e)) => panic!("{name}: serde accepted, legacy rejected: {e}"),
                (Err(e), Ok(_)) => panic!("{name}: serde rejected, legacy accepted: {e}"),
            }
        }
    }

    #[test]
    fn explicit_head_dim_survives_both_paths() {
        let json = DIFFERENTIAL_CORPUS[2].1;
        let cfg = LlamaFullConfig::from_hf_json_str(json).unwrap();
        assert_eq!(
            cfg.head_dim, 96,
            "explicit head_dim must not be overwritten"
        );
        assert_ne!(
            cfg.head_dim,
            4096 / 32,
            "the pre-existing fixture cannot make this distinction"
        );
    }

    #[test]
    fn rope_scaling_stays_tolerant_under_serde() {
        // The whole reason `rope_scaling` is held as a raw Value: these two
        // shapes are ACCEPTED by the parser that shipped. Under a derived
        // `Deserialize` the first is a missing-field error and the second an
        // unknown-key/missing-key error, so configs that load today would
        // start failing - a behaviour change wearing a refactor's clothes.
        let missing_field = LlamaFullConfig::from_hf_json_str(DIFFERENTIAL_CORPUS[3].1).unwrap();
        assert!(
            missing_field.rope_scaling.is_none(),
            "an incomplete rope_scaling block degrades to None, it does not error",
        );
        let alt_key = LlamaFullConfig::from_hf_json_str(DIFFERENTIAL_CORPUS[4].1).unwrap();
        assert!(
            alt_key.rope_scaling.is_some(),
            "HF ships `type` as well as `rope_type` and both must parse",
        );
    }

    #[test]
    fn from_hf_json_str_parses_llama3_1_8b() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128008, 128009],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1.0e-5,
            "rope_theta": 500000.0,
            "tie_word_embeddings": false,
            "rope_scaling": {
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            }
        }"#;
        let cfg = LlamaFullConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 128256);
        assert!((cfg.rope_theta - 500_000.0).abs() < 1e-6);
        assert_eq!(cfg.bos_token_id, Some(128000));
        match cfg.eos_token_id {
            Some(LlamaEosToks::Multiple(ref v)) => assert_eq!(v, &vec![128001, 128008, 128009]),
            other => panic!("expected Multiple EOS, got {:?}", other),
        }
        let scaling = cfg.rope_scaling.as_ref().expect("rope_scaling must parse");
        assert_eq!(scaling.rope_type, Llama3RopeType::Llama3);
        assert!((scaling.factor - 8.0).abs() < 1e-6);
        assert_eq!(scaling.original_max_position_embeddings, 8192);
        assert!(!cfg.tie_word_embeddings);

        // to_lazy_config drops HF-only fields without losing model shape.
        let lazy = cfg.to_lazy_config();
        assert_eq!(lazy.dim, 4096);
        assert_eq!(lazy.n_kv_heads, 8);
        assert_eq!(lazy.head_dim, 128);
        assert!((lazy.rope_base - 500_000.0).abs() < 1e-6);
    }

    /// LLaMA-2 style config (single EOS, no rope_scaling).
    #[test]
    fn from_hf_json_str_parses_llama2() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1.0e-6,
            "tie_word_embeddings": false
        }"#;
        let cfg = LlamaFullConfig::from_hf_json_str(json).unwrap();
        assert_eq!(
            cfg.num_key_value_heads, 32,
            "LLaMA-2 had no GQA, should default to num_attention_heads"
        );
        assert!(
            (cfg.rope_theta - 10_000.0).abs() < 1e-6,
            "rope_theta defaults to 10000.0"
        );
        assert_eq!(cfg.bos_token_id, Some(1));
        assert_eq!(cfg.eos_token_id, Some(LlamaEosToks::Single(2)));
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn llama_eos_toks_is_eos_single() {
        let eos = LlamaEosToks::Single(2);
        assert!(eos.is_eos(2));
        assert!(!eos.is_eos(3));
    }

    #[test]
    fn llama_eos_toks_is_eos_multiple() {
        let eos = LlamaEosToks::Multiple(vec![128001, 128008, 128009]);
        assert!(eos.is_eos(128001));
        assert!(eos.is_eos(128008));
        assert!(eos.is_eos(128009));
        assert!(!eos.is_eos(128000));
    }

    /// `tied_lm_head_from_embeddings` matches `wte^T @ x` semantics.
    /// Tested by checking a single (vocab, hidden) entry against
    /// the corresponding transposed coordinate.
    #[test]
    fn tied_lm_head_is_transpose_of_embeddings() {
        let vocab_size = 5;
        let hidden_size = 3;
        let wte: Arc<[f32]> = Arc::from(
            (0..vocab_size * hidden_size)
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
        );
        let lm = tied_lm_head_from_embeddings(&wte, vocab_size, hidden_size);
        let lm_data = match &lm {
            fuel_core::lazy::WeightStorage::F32(a) => a.clone(),
            _ => panic!("expected F32"),
        };
        // lm shape: [hidden_size, vocab_size]; wte shape: [vocab_size, hidden_size].
        for v in 0..vocab_size {
            for h in 0..hidden_size {
                let from_wte = wte[v * hidden_size + h];
                let from_lm = lm_data[h * vocab_size + v];
                assert!(
                    (from_wte - from_lm).abs() < 1e-9,
                    "transpose mismatch at (v={v}, h={h}): wte={from_wte}, lm={from_lm}",
                );
            }
        }
    }

    /// A `head_dim` large enough that the lowest-frequency wavelength exceeds
    /// `low_freq_wavelen`, so the scaling actually engages. `tiny_cfg`'s
    /// `head_dim: 4` does NOT — a parity test built on it would compare the
    /// scaled path against itself and pass no matter what the override did.
    fn scaled_cfg() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 32,
            dim: 32,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 16,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 500_000.0,
        }
    }

    /// LLaMA-3.1's band boundaries are `original_max_position_embeddings /
    /// *_freq_factor`, so the real 8192 puts every band edge far beyond any
    /// position a unit test can afford to decode to: with the production
    /// numbers, scaled and unscaled decode at positions 0..6 differ by 1.1e-7,
    /// and a parity test built on them passes whether the override is threaded
    /// or not (measured — that is how this constant got chosen).
    ///
    /// `original_max_position_embeddings: 8` moves the same band arithmetic
    /// into the positions a test actually visits. The code path is identical;
    /// only where the bands fall changes.
    /// Parity tolerance for the scaled-decode oracle. Deliberately far below
    /// the scaled-vs-unscaled separation the negative control measures — if
    /// the two were comparable, "parity holds" and "the override was dropped"
    /// would be indistinguishable outcomes.
    const PARITY_TOL: f32 = 1e-5;

    fn llama3_scaling() -> Llama3RopeConfig {
        Llama3RopeConfig {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8,
            rope_type: Llama3RopeType::Llama3,
        }
    }

    /// Decode `n` tokens one at a time through the persistent KV path,
    /// returning the LAST step's logits.
    fn decode_persistent(m: &Llama3Model, prompt: &[u32], extra: &[u32]) -> Vec<f32> {
        let cfg = &m.inner.config;
        let dev = fuel_core::Device::cpu();
        let mut cache = fuel_core::inference_context::KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            prompt.len() + extra.len() + 1,
            fuel_ir::DType::F32,
            &dev,
        )
        .unwrap();
        let mut ctx = fuel_core::inference_context::InferenceContext::new(dev);
        let mut session = None;
        let mut logits = m
            .forward_with_kv_context_persistent(prompt, &mut cache, &mut ctx, &mut session)
            .unwrap();
        for &t in extra {
            logits = m
                .forward_with_kv_context_persistent(&[t], &mut cache, &mut ctx, &mut session)
                .unwrap();
        }
        logits
    }

    /// The oracle: incremental persistent decode must agree with a full-prefix
    /// `forward` at the same final position, **with scaling engaged**.
    ///
    /// Compares LOGITS, not sampled tokens — per the standing warning in
    /// `fuel-inference/src/multi_session.rs`, a tiny random model's greedy
    /// argmax is a fixed point and would pass even if the KV state were wrong.
    ///
    /// Several decode steps, not one, on purpose: the RoPE override has to
    /// reach BOTH the first token's held-graph build and every later rebind.
    /// A version threading only the build would produce a correct step 1 and
    /// wrong steps 2..n — position-dependent, and invisible to a 1-token test.
    #[test]
    fn scaled_persistent_decode_matches_full_prefix_forward() {
        let cfg = scaled_cfg();
        let inner = LlamaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let m = Llama3Model::new(inner, Some(llama3_scaling()), None);

        let prompt = [1u32, 2, 3, 4];
        let extra = [5u32, 6, 7];
        let got = decode_persistent(&m, &prompt, &extra);

        // Oracle: one forward over the whole sequence; take the last position.
        let all: Vec<u32> = prompt.iter().chain(extra.iter()).copied().collect();
        let full = m.forward(&all, 0).unwrap().realize_f32();
        let want = &full[(all.len() - 1) * cfg.vocab_size..];

        assert_eq!(
            got.len(),
            cfg.vocab_size,
            "decode returns last-position logits"
        );
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).abs() < PARITY_TOL,
                "scaled persistent decode diverged at logit {i}: {a} vs {b}",
            );
        }
    }

    /// The negative control for the test above. If the override were dropped
    /// and `Llama3Model` decoded through plain unscaled `LlamaModel`, the
    /// result would still be *finite and plausible* — it just would not match
    /// the scaled oracle. Assert the two are actually distinguishable, or the
    /// parity assertion above proves nothing about scaling.
    #[test]
    fn unscaled_decode_is_distinguishable_from_scaled() {
        let cfg = scaled_cfg();
        let weights = tiny_weights(&cfg);
        let scaled = Llama3Model::new(
            LlamaModel {
                config: cfg.clone(),
                weights: weights.clone(),
            },
            Some(llama3_scaling()),
            None,
        );
        let unscaled = Llama3Model::new(
            LlamaModel {
                config: cfg.clone(),
                weights,
            },
            None,
            None,
        );

        let prompt = [1u32, 2, 3, 4];
        let extra = [5u32, 6, 7];
        let a = decode_persistent(&scaled, &prompt, &extra);
        let b = decode_persistent(&unscaled, &prompt, &extra);

        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 10.0 * PARITY_TOL,
            "scaled and unscaled decode are indistinguishable (max diff {max_diff}) — the parity test above would pass with the override removed",
        );
    }

    /// GAP-281: an odd `head_dim` must DECLINE, not panic. `half = head_dim / 2`
    /// truncates silently otherwise, so it is a CONFIG property.
    ///
    /// Asserts on the MESSAGE, not merely that an `Err` arrived: a later guard
    /// on the same path can supply an `Err` of its own (measured in
    /// `lazy_mamba2`), which makes an `expect_err`-only arm pass under
    /// sabotage while reporting the guard as live.
    #[test]
    fn llama3_odd_head_dim_declines_rather_than_panicking() {
        // CONTROL: the conforming even value must still build.
        build_llama3_rope_tables(10_000.0, None, 0, 4, 8)
            .expect("control: an even head_dim must still build");

        let err = build_llama3_rope_tables(10_000.0, None, 0, 4, 7)
            .expect_err("an odd head_dim must DECLINE, not panic");
        let msg = format!("{err}");
        assert!(
            msg.contains("head_dim") && msg.contains('7'),
            "the decline must name the operand and its value, got: {msg}"
        );
    }
}

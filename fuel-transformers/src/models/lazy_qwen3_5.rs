// SPDX-License-Identifier: MIT OR Apache-2.0
//! Qwen3.5 (the Gated DeltaNet / full-attention hybrid; HF `model_type` `qwen3_5`) config and
//! weight loader. ROADMAP row 40, stage 1.
//!
//! The text decoder alternates two token mixers, chosen per layer by `layer_types`:
//!
//! - `linear_attention`: a Gated DeltaNet layer (`linear_attn.*`): one fused `in_proj_qkv`
//!   projection, a depthwise causal `conv1d` over the q/k/v channels, per-value-head decay
//!   (`A_log`, `dt_bias`, `in_proj_a`) and write strength (`in_proj_b`), an output gate
//!   (`in_proj_z`) applied through a gated RMS norm (`norm`), and `out_proj`.
//! - `full_attention`: grouped-query attention with a doubled `q_proj` (the second half is an
//!   output gate), per-head q/k RMS norms and partial rotary embeddings.
//!
//! Every layer then has the same SwiGLU MLP. The checkpoint is a multimodal wrapper
//! (`Qwen3_5ForConditionalGeneration`): the text tower lives under `model.language_model.`, and
//! `model.visual.*` and the speculative `mtp.*` head are not read here.
//!
//! # Scope
//!
//! This is the loader only. [`Qwen3_5Model::forward`] returns a typed error: the Gated DeltaNet
//! recurrence is not built yet (ROADMAP row 40 stages 2 and 3, `docs/gaps.md` GAP-368).
//!
//! # Norm convention
//!
//! The decoder RMS norms (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, the
//! final `norm`) are zero-centred: the output is `x_hat * (1 + w)`. Measured on
//! `Qwen/Qwen3.5-0.8B`: `layers.0.input_layernorm.weight` has mean 0.238 and min -0.115, and the
//! final norm has mean 3.31, so a plain `x_hat * w` reading would collapse the early layers. The
//! loader therefore stores `1 + w`, and the rest of Fuel can apply the gain with its ordinary
//! `rms_norm_affine`. The gated norm inside a Gated DeltaNet layer (`linear_attn.norm.weight`,
//! mean 0.95) is NOT zero-centred and is stored as is.

use fuel_ir::error::{Error, Result};
use fuel_tensor::lazy::{
    Tensor, WeightStorage, load_tensor_as_f32, load_transposed_matrix_preserve_dtype,
};
use std::sync::Arc;

/// Which token mixer a decoder layer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3_5LayerKind {
    /// Gated DeltaNet (`linear_attn.*`).
    LinearAttention,
    /// Grouped-query attention with an output gate (`self_attn.*`).
    FullAttention,
}

/// The text decoder's configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3_5Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<Qwen3_5LayerKind>,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
    // Full attention.
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// `q_proj` is twice as wide and its second half gates the attention output.
    pub attn_output_gate: bool,
    pub rope_theta: f64,
    /// Fraction of `head_dim` that is rotated.
    pub partial_rotary_factor: f64,
    // Gated DeltaNet.
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
}

impl Qwen3_5Config {
    /// `linear_num_key_heads * linear_key_head_dim`: the width of q (and of k) in a GDN layer.
    pub fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// `linear_num_value_heads * linear_value_head_dim`: the width of v (and of the output gate).
    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Channels the depthwise conv runs over: q, k and v concatenated.
    pub fn linear_conv_dim(&self) -> usize {
        2 * self.linear_key_dim() + self.linear_value_dim()
    }

    /// Width of the full-attention query (before the gate doubling).
    pub fn attn_q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Parse a HuggingFace `config.json`. The text fields are read from `text_config` when the
    /// file is the multimodal wrapper, and from the top level otherwise.
    pub fn from_hf_json_str(json: &str) -> Result<Self> {
        let root: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| Error::Msg(format!("Qwen3_5Config: config.json is not JSON: {e}")).bt())?;
        let text = root.get("text_config").unwrap_or(&root);
        let raw: Raw = serde_json::from_value(text.clone())
            .map_err(|e| Error::Msg(format!("Qwen3_5Config: {e}")).bt())?;
        let tie = raw
            .tie_word_embeddings
            .or_else(|| root.get("tie_word_embeddings").and_then(|v| v.as_bool()))
            .unwrap_or(false);
        raw.resolve(tie)
    }

    fn validate(&self) -> Result<()> {
        let bad = |msg: String| Err(Error::Msg(format!("Qwen3_5Config: {msg}")).bt());
        if self.layer_types.len() != self.num_hidden_layers {
            return bad(format!(
                "layer_types has {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if self.linear_num_key_heads == 0
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
        {
            return bad(format!(
                "linear_num_value_heads ({}) must be a positive multiple of linear_num_key_heads ({})",
                self.linear_num_value_heads, self.linear_num_key_heads
            ));
        }
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return bad(format!(
                "num_attention_heads ({}) must be a positive multiple of num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if !(self.partial_rotary_factor > 0.0 && self.partial_rotary_factor <= 1.0) {
            return bad(format!(
                "partial_rotary_factor {} is outside (0, 1]",
                self.partial_rotary_factor
            ));
        }
        for (name, v) in [
            ("hidden_size", self.hidden_size),
            ("head_dim", self.head_dim),
            ("linear_key_head_dim", self.linear_key_head_dim),
            ("linear_value_head_dim", self.linear_value_head_dim),
            ("linear_conv_kernel_dim", self.linear_conv_kernel_dim),
        ] {
            if v == 0 {
                return bad(format!("{name} is zero"));
            }
        }
        Ok(())
    }
}

#[derive(serde::Deserialize)]
struct RopeParameters {
    rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    partial_rotary_factor: f64,
}

fn default_partial_rotary() -> f64 {
    1.0
}

#[derive(serde::Deserialize)]
struct Raw {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    layer_types: Vec<String>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f64,
    #[serde(default)]
    attn_output_gate: bool,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    max_position_embeddings: usize,
    rope_parameters: RopeParameters,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel_dim: usize,
}

fn default_eps() -> f64 {
    1e-6
}

impl Raw {
    fn resolve(self, tie_word_embeddings: bool) -> Result<Qwen3_5Config> {
        let layer_types = self
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "linear_attention" => Ok(Qwen3_5LayerKind::LinearAttention),
                "full_attention" => Ok(Qwen3_5LayerKind::FullAttention),
                other => Err(Error::Msg(format!(
                    "Qwen3_5Config: unknown layer type {other:?} (expected linear_attention or full_attention)"
                ))
                .bt()),
            })
            .collect::<Result<Vec<_>>>()?;
        let cfg = Qwen3_5Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            layer_types,
            rms_norm_eps: self.rms_norm_eps,
            tie_word_embeddings,
            max_position_embeddings: self.max_position_embeddings,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            attn_output_gate: self.attn_output_gate,
            rope_theta: self.rope_parameters.rope_theta,
            partial_rotary_factor: self.rope_parameters.partial_rotary_factor,
            linear_num_key_heads: self.linear_num_key_heads,
            linear_num_value_heads: self.linear_num_value_heads,
            linear_key_head_dim: self.linear_key_head_dim,
            linear_value_head_dim: self.linear_value_head_dim,
            linear_conv_kernel_dim: self.linear_conv_kernel_dim,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

/// A Gated DeltaNet layer's weights.
#[derive(Debug, Clone)]
pub struct GatedDeltaNetWeights {
    /// `hidden -> 2 * key_dim + value_dim`: q, k and v, in that order.
    pub in_proj_qkv: WeightStorage,
    /// `hidden -> value_dim`: the output gate.
    pub in_proj_z: WeightStorage,
    /// `hidden -> num_value_heads`: the decay input.
    pub in_proj_a: WeightStorage,
    /// `hidden -> num_value_heads`: the write strength (`beta`) input.
    pub in_proj_b: WeightStorage,
    /// Depthwise causal conv, `[conv_dim, kernel]` row-major (the checkpoint's middle `1` is dropped).
    pub conv1d_weight: Arc<[f32]>,
    /// `[num_value_heads]`, kept in f32: the decay is `-exp(A_log) * softplus(a + dt_bias)`.
    pub a_log: Arc<[f32]>,
    /// `[num_value_heads]`.
    pub dt_bias: Arc<[f32]>,
    /// `[value_head_dim]`: gain of the gated RMS norm (NOT zero-centred).
    pub norm_gain: Arc<[f32]>,
    /// `value_dim -> hidden`.
    pub out_proj: WeightStorage,
}

/// A full-attention layer's weights.
#[derive(Debug, Clone)]
pub struct FullAttentionWeights {
    /// `hidden -> q_dim * 2` when the config has an output gate, else `hidden -> q_dim`.
    pub q_proj: WeightStorage,
    pub k_proj: WeightStorage,
    pub v_proj: WeightStorage,
    pub o_proj: WeightStorage,
    /// `[head_dim]`, already `1 + w`.
    pub q_norm_gain: Arc<[f32]>,
    /// `[head_dim]`, already `1 + w`.
    pub k_norm_gain: Arc<[f32]>,
}

/// A layer's token mixer.
#[derive(Debug, Clone)]
pub enum Qwen3_5Mixer {
    GatedDeltaNet(GatedDeltaNetWeights),
    FullAttention(FullAttentionWeights),
}

/// One decoder layer.
#[derive(Debug, Clone)]
pub struct Qwen3_5Layer {
    /// `[hidden]`, already `1 + w`.
    pub input_norm_gain: Arc<[f32]>,
    pub mixer: Qwen3_5Mixer,
    /// `[hidden]`, already `1 + w`.
    pub post_attn_norm_gain: Arc<[f32]>,
    pub ffn_gate: WeightStorage,
    pub ffn_up: WeightStorage,
    pub ffn_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Qwen3_5Weights {
    pub instance: fuel_tensor::decode_shape::ModelInstanceId,
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Qwen3_5Layer>,
    /// `[hidden]`, already `1 + w`.
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

type St = fuel_loaders::safetensors::MmapedSafetensors;

fn msg(text: String) -> fuel_ir::error::Error {
    Error::Msg(text).bt()
}

/// The shape a tensor has in the checkpoint, or an error naming the tensor.
fn shape_of(st: &St, name: &str) -> Result<Vec<usize>> {
    Ok(st
        .get(name)
        .map_err(|e| msg(format!("Qwen3_5Weights: tensor {name:?}: {e}")))?
        .shape()
        .to_vec())
}

fn expect_shape(st: &St, name: &str, want: &[usize]) -> Result<()> {
    let got = shape_of(st, name)?;
    if got != want {
        return Err(msg(format!(
            "Qwen3_5Weights: tensor {name:?} has shape {got:?}, the config implies {want:?}"
        )));
    }
    Ok(())
}

/// A 1-D tensor of `len` elements as f32.
fn vector(st: &St, name: &str, len: usize) -> Result<Arc<[f32]>> {
    expect_shape(st, name, &[len])?;
    Ok(Arc::from(load_tensor_as_f32(st, name)?))
}

/// A zero-centred RMS norm gain: stored as `1 + w`.
fn zero_centred_gain(st: &St, name: &str, len: usize) -> Result<Arc<[f32]>> {
    let w = vector(st, name, len)?;
    Ok(w.iter().map(|x| 1.0 + x).collect::<Vec<f32>>().into())
}

/// A `[out, in]` projection weight.
fn projection(
    st: &St,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> Result<WeightStorage> {
    expect_shape(st, name, &[out_features, in_features])?;
    load_transposed_matrix_preserve_dtype(st, name, out_features, in_features)
}

impl Qwen3_5Weights {
    /// Load the text decoder from a HuggingFace `Qwen/Qwen3.5-*` checkpoint. Every tensor's
    /// shape is checked against the config before it is read.
    pub fn load_from_mmapped(st: &St, cfg: &Qwen3_5Config) -> Result<Self> {
        let prefix = Self::text_prefix(st)?;
        let h = cfg.hidden_size;
        let embed_name = format!("{prefix}embed_tokens.weight");
        expect_shape(st, &embed_name, &[cfg.vocab_size, h])?;
        let token_embedding: Arc<[f32]> = Arc::from(load_tensor_as_f32(st, &embed_name)?);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Self::load_layer(st, cfg, &format!("{prefix}layers.{i}"), i))
            .collect::<Result<Vec<_>>>()?;
        let final_norm_gain = zero_centred_gain(st, &format!("{prefix}norm.weight"), h)?;
        let output = if cfg.tie_word_embeddings {
            crate::models::lazy_llama_full::tied_lm_head_from_embeddings(
                &token_embedding,
                cfg.vocab_size,
                h,
            )
        } else {
            projection(st, "lm_head.weight", cfg.vocab_size, h)?
        };
        Ok(Self {
            instance: fuel_tensor::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            final_norm_gain,
            output,
        })
    }

    /// `model.language_model.` in the multimodal wrapper, `model.` in a text-only export.
    fn text_prefix(st: &St) -> Result<&'static str> {
        for prefix in ["model.language_model.", "model."] {
            if st.get(&format!("{prefix}embed_tokens.weight")).is_ok() {
                return Ok(prefix);
            }
        }
        Err(msg(
            "Qwen3_5Weights: no `embed_tokens.weight` under `model.language_model.` or `model.`"
                .to_string(),
        ))
    }

    fn load_layer(st: &St, cfg: &Qwen3_5Config, p: &str, i: usize) -> Result<Qwen3_5Layer> {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let mixer = match cfg.layer_types[i] {
            Qwen3_5LayerKind::LinearAttention => Qwen3_5Mixer::GatedDeltaNet(
                Self::load_gated_delta_net(st, cfg, &format!("{p}.linear_attn"))?,
            ),
            Qwen3_5LayerKind::FullAttention => Qwen3_5Mixer::FullAttention(
                Self::load_full_attention(st, cfg, &format!("{p}.self_attn"))?,
            ),
        };
        Ok(Qwen3_5Layer {
            input_norm_gain: zero_centred_gain(st, &format!("{p}.input_layernorm.weight"), h)?,
            mixer,
            post_attn_norm_gain: zero_centred_gain(
                st,
                &format!("{p}.post_attention_layernorm.weight"),
                h,
            )?,
            ffn_gate: projection(st, &format!("{p}.mlp.gate_proj.weight"), inter, h)?,
            ffn_up: projection(st, &format!("{p}.mlp.up_proj.weight"), inter, h)?,
            ffn_down: projection(st, &format!("{p}.mlp.down_proj.weight"), h, inter)?,
        })
    }

    fn load_gated_delta_net(st: &St, cfg: &Qwen3_5Config, p: &str) -> Result<GatedDeltaNetWeights> {
        let h = cfg.hidden_size;
        let nv = cfg.linear_num_value_heads;
        let conv_dim = cfg.linear_conv_dim();
        let conv_name = format!("{p}.conv1d.weight");
        expect_shape(st, &conv_name, &[conv_dim, 1, cfg.linear_conv_kernel_dim])?;
        Ok(GatedDeltaNetWeights {
            in_proj_qkv: projection(st, &format!("{p}.in_proj_qkv.weight"), conv_dim, h)?,
            in_proj_z: projection(
                st,
                &format!("{p}.in_proj_z.weight"),
                cfg.linear_value_dim(),
                h,
            )?,
            in_proj_a: projection(st, &format!("{p}.in_proj_a.weight"), nv, h)?,
            in_proj_b: projection(st, &format!("{p}.in_proj_b.weight"), nv, h)?,
            conv1d_weight: Arc::from(load_tensor_as_f32(st, &conv_name)?),
            a_log: vector(st, &format!("{p}.A_log"), nv)?,
            dt_bias: vector(st, &format!("{p}.dt_bias"), nv)?,
            norm_gain: vector(st, &format!("{p}.norm.weight"), cfg.linear_value_head_dim)?,
            out_proj: projection(
                st,
                &format!("{p}.out_proj.weight"),
                h,
                cfg.linear_value_dim(),
            )?,
        })
    }

    fn load_full_attention(st: &St, cfg: &Qwen3_5Config, p: &str) -> Result<FullAttentionWeights> {
        let h = cfg.hidden_size;
        let q_dim = cfg.attn_q_dim();
        let q_out = if cfg.attn_output_gate {
            2 * q_dim
        } else {
            q_dim
        };
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        Ok(FullAttentionWeights {
            q_proj: projection(st, &format!("{p}.q_proj.weight"), q_out, h)?,
            k_proj: projection(st, &format!("{p}.k_proj.weight"), kv_dim, h)?,
            v_proj: projection(st, &format!("{p}.v_proj.weight"), kv_dim, h)?,
            o_proj: projection(st, &format!("{p}.o_proj.weight"), h, q_dim)?,
            q_norm_gain: zero_centred_gain(st, &format!("{p}.q_norm.weight"), cfg.head_dim)?,
            k_norm_gain: zero_centred_gain(st, &format!("{p}.k_norm.weight"), cfg.head_dim)?,
        })
    }
}

/// The loaded model. There is no forward pass yet: see the module docs.
#[derive(Debug, Clone)]
pub struct Qwen3_5Model {
    pub config: Qwen3_5Config,
    pub weights: Qwen3_5Weights,
}

impl Qwen3_5Model {
    /// Always an error until the Gated DeltaNet recurrence exists (ROADMAP row 40 stages 2-3,
    /// `docs/gaps.md` GAP-368). A typed refusal, not a wrong answer.
    pub fn forward(&self, _tokens: &[u32], _start_pos: usize) -> Result<Tensor> {
        Err(msg(
            "Qwen3_5Model::forward: the Gated DeltaNet recurrence is not implemented (GAP-368, \
             ROADMAP row 40); only the loader exists"
                .to_string(),
        ))
    }

    /// Number of Gated DeltaNet layers.
    pub fn num_linear_layers(&self) -> usize {
        self.config
            .layer_types
            .iter()
            .filter(|k| **k == Qwen3_5LayerKind::LinearAttention)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Qwen/Qwen3.5-0.8B` `config.json`, abridged to the keys this parser reads plus the
    /// multimodal wrapper around them (the real file, fetched 2026-10-23).
    const CONFIG_0_8B: &str = r#"{
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "attn_output_gate": true, "head_dim": 256, "hidden_size": 1024,
            "intermediate_size": 3584,
            "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention",
                            "linear_attention","linear_attention","linear_attention","full_attention",
                            "linear_attention","linear_attention","linear_attention","full_attention",
                            "linear_attention","linear_attention","linear_attention","full_attention",
                            "linear_attention","linear_attention","linear_attention","full_attention",
                            "linear_attention","linear_attention","linear_attention","full_attention"],
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
            "linear_num_value_heads": 16, "linear_value_head_dim": 128,
            "max_position_embeddings": 262144, "num_attention_heads": 8, "num_hidden_layers": 24,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "tie_word_embeddings": true,
            "vocab_size": 248320,
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25,
                                "rope_type": "default", "mrope_interleaved": true,
                                "mrope_section": [11, 11, 10]}
        },
        "tie_word_embeddings": true
    }"#;

    #[test]
    fn parses_the_real_0_8b_config() {
        let c = Qwen3_5Config::from_hf_json_str(CONFIG_0_8B).unwrap();
        assert_eq!(c.num_hidden_layers, 24);
        assert_eq!(c.layer_types.len(), 24);
        assert_eq!(
            c.layer_types
                .iter()
                .filter(|k| **k == Qwen3_5LayerKind::FullAttention)
                .count(),
            6
        );
        assert!(c.tie_word_embeddings && c.attn_output_gate);
        assert_eq!(c.rope_theta, 1.0e7);
        assert_eq!(c.partial_rotary_factor, 0.25);
        // The derived widths the real checkpoint's tensors have: conv1d [6144, 1, 4],
        // in_proj_z [2048, 1024], q_proj [4096, 1024] (= 2 * 8 * 256).
        assert_eq!(c.linear_conv_dim(), 6144);
        assert_eq!(c.linear_value_dim(), 2048);
        assert_eq!(2 * c.attn_q_dim(), 4096);
    }

    /// The untied 27B differs in head counts (48 value heads on 16 key heads) and has
    /// `tie_word_embeddings` at the top level only.
    #[test]
    fn reads_untied_top_level_tie_and_unequal_head_counts() {
        let mut v: serde_json::Value = serde_json::from_str(CONFIG_0_8B).unwrap();
        v["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("tie_word_embeddings");
        v["text_config"]["linear_num_value_heads"] = 48.into();
        v["tie_word_embeddings"] = false.into();
        let json = v.to_string();
        let c = Qwen3_5Config::from_hf_json_str(&json).unwrap();
        assert!(!c.tie_word_embeddings);
        assert_eq!(c.linear_value_dim(), 48 * 128);
        assert_eq!(c.linear_conv_dim(), 2 * 16 * 128 + 48 * 128);
    }

    #[test]
    fn refuses_inconsistent_configs() {
        let bad_layers =
            CONFIG_0_8B.replace("\"num_hidden_layers\": 24", "\"num_hidden_layers\": 25");
        assert!(Qwen3_5Config::from_hf_json_str(&bad_layers).is_err());
        let bad_heads = CONFIG_0_8B.replace(
            "\"linear_num_value_heads\": 16",
            "\"linear_num_value_heads\": 24",
        );
        assert!(Qwen3_5Config::from_hf_json_str(&bad_heads).is_err());
        let bad_kind = CONFIG_0_8B.replacen("full_attention", "sliding_attention", 1);
        assert!(Qwen3_5Config::from_hf_json_str(&bad_kind).is_err());
        assert!(Qwen3_5Config::from_hf_json_str("not json").is_err());
    }

    /// The flat (text-only export) form parses too.
    #[test]
    fn parses_a_flat_text_only_config() {
        let v: serde_json::Value = serde_json::from_str(CONFIG_0_8B).unwrap();
        let flat = v["text_config"].to_string();
        let c = Qwen3_5Config::from_hf_json_str(&flat).unwrap();
        assert_eq!(c.hidden_size, 1024);
    }
}

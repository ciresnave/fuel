// SPDX-License-Identifier: MIT OR Apache-2.0
//! OLMo decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. OLMo is "LLaMA with LayerNorm-no-bias":
//! - GQA + RoPE + SwiGLU FFN — same as LLaMA.
//! - **Pre-attention and pre-FFN LayerNorm** (subtract-mean + scale)
//!   with **no bias** — distinct from LLaMA's RmsNorm and from
//!   Falcon's full LayerNorm-with-bias.
//! - Optional Q/K/V/O biases via `cfg.attention_bias`.
//!
//! `apply_layer_norm_no_bias` is a local helper that mean-centres
//! then divides by stddev (using `layer_norm_last_dim` which builds
//! the mean+variance reduction) and scales by gain only.

use fuel_core::lazy::{Tensor, WeightStorage};
use fuel_core::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct OlmoConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub layer_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub attention_bias: bool,
}

impl OlmoConfig {
    /// `allenai/OLMo-7B-hf` ballpark.
    pub fn olmo_7b() -> Self {
        Self {
            vocab_size: 50_304,
            hidden_size: 4096,
            intermediate_size: 11_008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            head_dim: 128,
            layer_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 2048,
            attention_bias: false,
        }
    }
}

// ROADMAP item 8 (II): config-from-path on the #57 template. A `serde` raw with
// HF field names + OLMo's own constant defaults, then `resolve` routes kv heads
// + head_dim through the shared `fuel_core::hf_config` rules. OLMo v1 uses
// non-parametric LayerNorm, so its config carries no `layer_norm_eps` — the field
// defaults to the port's 1e-5. `clip_qkv` (null in the artifact) is unmodelled.
#[derive(Debug, Clone, serde::Deserialize)]
struct OlmoConfigRaw {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_olmo_layer_norm_eps")]
    layer_norm_eps: f64,
    #[serde(default = "default_olmo_rope_theta")]
    rope_theta: f64,
    max_position_embeddings: usize,
    #[serde(default)]
    attention_bias: bool,
}

fn default_olmo_layer_norm_eps() -> f64 {
    1e-5
}
fn default_olmo_rope_theta() -> f64 {
    10_000.0
}

impl OlmoConfigRaw {
    fn from_json_str(json: &str) -> fuel_core::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| fuel_core::Error::Msg(format!("parsing OLMo config.json: {e}")))
    }

    fn resolve(self) -> Result<OlmoConfig> {
        Ok(OlmoConfig {
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
            layer_norm_eps: self.layer_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            attention_bias: self.attention_bias,
        })
    }
}

impl OlmoConfig {
    /// Parse a HuggingFace `config.json` string into an [`OlmoConfig`].
    ///
    /// ROADMAP item 8 (II): reads the artifact rather than returning a preset —
    /// see the born-red `olmo_config_from_hf_json_parses_the_artifact_not_a_preset`.
    pub fn from_hf_json_str(json: &str) -> fuel_core::Result<Self> {
        OlmoConfigRaw::from_json_str(json)?.resolve()
    }
}

#[derive(Debug, Clone)]
pub struct OlmoLayerWeights {
    pub attn_norm_gain: Arc<[f32]>,
    pub ffn_norm_gain: Arc<[f32]>,
    pub attn_q: WeightStorage,
    pub attn_q_bias: Option<Arc<[f32]>>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Option<Arc<[f32]>>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Option<Arc<[f32]>>,
    pub attn_o: WeightStorage,
    pub attn_o_bias: Option<Arc<[f32]>>,
    pub ffn_gate: WeightStorage,
    pub ffn_up: WeightStorage,
    pub ffn_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct OlmoWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<OlmoLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct OlmoModel {
    pub config: OlmoConfig,
    pub weights: OlmoWeights,
}

impl OlmoModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final LayerNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection. OLMo uses LayerNorm
    /// without bias for the final norm.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. OLMo does NOT scale embeddings.
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
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0);

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
                "OlmoModel::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(
                fuel_core::Error::Msg("OlmoModel::forward_embeds: seq must be > 0".into()).bt(),
            );
        }
        if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size {
            return Err(fuel_core::Error::Msg(
                "OlmoConfig: num_attention_heads * head_dim must equal hidden_size".into(),
            )
            .bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) =
            h.rope_tables_const(cfg.rope_theta, start_pos, seq, cfg.head_dim);

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        apply_layer_norm_no_bias(
            &h,
            &weights.final_norm_gain,
            cfg.hidden_size,
            cfg.layer_norm_eps,
        )
    }

    fn apply_layer(
        &self,
        x: &Tensor,
        layer: &OlmoLayerWeights,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let x_norm = apply_layer_norm_no_bias(
            x,
            &layer.attn_norm_gain,
            cfg.hidden_size,
            cfg.layer_norm_eps,
        )?;
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
        let mask = Tensor::additive_causal_mask_like(x, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer
            .attn_o
            .apply_linear(&merged, cfg.hidden_size, cfg.hidden_size)?
            .add_optional_trailing_bias(layer.attn_o_bias.as_ref())?;

        let h1 = x.add(&attn_out)?;
        let h1_norm = apply_layer_norm_no_bias(
            &h1,
            &layer.ffn_norm_gain,
            cfg.hidden_size,
            cfg.layer_norm_eps,
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

/// `y = ((x - mean) / sqrt(var + eps)) * gain`. Same as
/// [`fuel_core::lazy::apply_affine_layer_norm`] minus the additive bias.
fn apply_layer_norm_no_bias(x: &Tensor, gain: &Arc<[f32]>, dim: usize, eps: f64) -> Result<Tensor> {
    assert_eq!(gain.len(), dim);
    let normalized = x.layer_norm_last_dim(eps)?;
    let gain_t = x.const_f32_like(Arc::clone(gain), Shape::from_dims(&[dim]));
    normalized.broadcast_mul(&gain_t)
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl OlmoWeights {
    /// Load OLMo weights from HF safetensors (e.g. `allenai/OLMo-7B-hf`).
    /// HF naming follows LLaMA conventions: model.embed_tokens / model.layers.{i}
    /// / model.norm / lm_head.
    pub fn load_from_mmapped(
        st: &fuel_core::safetensors::MmapedSafetensors,
        cfg: &OlmoConfig,
    ) -> Result<Self> {
        use fuel_core::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);

        let mut layers: Vec<OlmoLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let attn_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.input_layernorm.weight"),
            )?);
            let ffn_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.post_attention_layernorm.weight"),
            )?);
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
            let (attn_q_bias, attn_k_bias, attn_v_bias, attn_o_bias) = if cfg.attention_bias {
                (
                    Some(Arc::from(load_tensor_as_f32(
                        st,
                        &format!("{p}.self_attn.q_proj.bias"),
                    )?)),
                    Some(Arc::from(load_tensor_as_f32(
                        st,
                        &format!("{p}.self_attn.k_proj.bias"),
                    )?)),
                    Some(Arc::from(load_tensor_as_f32(
                        st,
                        &format!("{p}.self_attn.v_proj.bias"),
                    )?)),
                    Some(Arc::from(load_tensor_as_f32(
                        st,
                        &format!("{p}.self_attn.o_proj.bias"),
                    )?)),
                )
            } else {
                (None, None, None, None)
            };
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
            layers.push(OlmoLayerWeights {
                attn_norm_gain,
                ffn_norm_gain,
                attn_q,
                attn_q_bias,
                attn_k,
                attn_k_bias,
                attn_v,
                attn_v_bias,
                attn_o,
                attn_o_bias,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }
        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.norm.weight")?);
        let output =
            load_transposed_matrix_preserve_dtype(st, "lm_head.weight", cfg.vocab_size, h)?;
        Ok(Self {
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

    // ROADMAP item 8 (II). Golden values from allenai/OLMo-1B-hf's real
    // config.json — the 1B, not the 7B the preset is named after, so parse !=
    // preset. OLMo v1 config omits layer_norm_eps (non-parametric LN) → defaulted.
    const OLMO_1B_CONFIG_JSON: &str = r#"{
        "architectures": ["OlmoForCausalLM"],
        "model_type": "olmo",
        "vocab_size": 50304,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": 2048,
        "rope_theta": 10000.0,
        "attention_bias": false,
        "clip_qkv": null
    }"#;

    #[test]
    fn olmo_config_from_hf_json_parses_the_artifact_not_a_preset() {
        let cfg = OlmoConfig::from_hf_json_str(OLMO_1B_CONFIG_JSON).unwrap();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 16);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.vocab_size, 50_304);
        assert_eq!(cfg.intermediate_size, 8192);
        assert_eq!(cfg.num_key_value_heads, 16);
        // head_dim absent → derived 2048/16 = 128.
        assert_eq!(cfg.head_dim, 128);
        // layer_norm_eps absent from the artifact → defaulted to 1e-5.
        assert_eq!(cfg.layer_norm_eps, 1e-5);
        assert_eq!(cfg.rope_theta, 10_000.0);
        assert!(!cfg.attention_bias);
        // Sabotage sibling (WEAKER): distinct from the 7B preset (hidden differs).
        assert_ne!(cfg, OlmoConfig::olmo_7b());
    }

    /// A SECOND distinct config, exercising the take-if-present head_dim branch
    /// (explicit 96 ≠ 2560/20 = 128) with GQA.
    #[test]
    fn olmo_config_reads_a_second_distinct_config_with_explicit_head_dim() {
        let json = r#"{
            "model_type": "olmo",
            "vocab_size": 40000,
            "hidden_size": 2560,
            "intermediate_size": 6912,
            "num_hidden_layers": 20,
            "num_attention_heads": 20,
            "num_key_value_heads": 5,
            "head_dim": 96,
            "max_position_embeddings": 2048,
            "rope_theta": 500000.0
        }"#;
        let cfg = OlmoConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.num_key_value_heads, 5);
        // explicit head_dim WINS over the quotient (2560/20 = 128).
        assert_eq!(cfg.head_dim, 96);
        assert_ne!(cfg.head_dim, 2560 / 20);
        assert_eq!(cfg.rope_theta, 500_000.0);
        assert_eq!(cfg.layer_norm_eps, 1e-5); // defaulted
        assert_ne!(cfg.hidden_size, 2048);
    }

    /// `num_key_value_heads` ABSENT → defaults to `num_attention_heads`.
    #[test]
    fn olmo_config_gqa_defaults_to_num_heads_when_absent() {
        let json = r#"{
            "model_type": "olmo",
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "max_position_embeddings": 128
        }"#;
        let cfg = OlmoConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.num_key_value_heads, 8);
    }

    /// TRUE MQA (`num_key_value_heads = 1`) survives, not collapsed.
    #[test]
    fn olmo_config_preserves_true_mqa() {
        let json = r#"{
            "model_type": "olmo",
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128
        }"#;
        let cfg = OlmoConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.num_key_value_heads, 1);
    }

    fn tiny_weights(cfg: &OlmoConfig) -> OlmoWeights {
        let mut s: u32 = 11111;
        let next = || -> f32 {
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
        let layers: Vec<OlmoLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| OlmoLayerWeights {
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
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
                attn_o_bias: if cfg.attention_bias {
                    Some(vec_of(h, &mut *nb))
                } else {
                    None
                },
                ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                ffn_up: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        OlmoWeights {
            token_embedding,
            layers,
            final_norm_gain,
            output,
        }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = OlmoConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 4,
            layer_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            attention_bias: false,
        };
        let model = OlmoModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let logits = model.forward(&[1, 2, 3, 4], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 4, cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// `forward_hidden` returns post-LayerNorm-no-bias hidden
    /// states `(1, seq, hidden_size)` without the lm_head matmul.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = OlmoConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 4,
            layer_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            attention_bias: false,
        };
        let model = OlmoModel {
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

    fn forward_embeds_test_cfg() -> OlmoConfig {
        OlmoConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 4,
            layer_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            attention_bias: false,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = OlmoModel {
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
            "OLMo forward vs forward_embeds must agree (max diff {max_diff})"
        );
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = OlmoModel {
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
        let cfg = forward_embeds_test_cfg();
        let model = OlmoModel {
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
            "OLMo forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})"
        );
    }
}

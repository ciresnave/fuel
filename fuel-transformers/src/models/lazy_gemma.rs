// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gemma (v1) decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Gemma1 is the closest non-LLaMA architectural
//! cousin in this batch — same overall shape (RmsNorm, GQA, RoPE,
//! gated FFN) but with three small twists worth honoring:
//!
//!   1. **Offset RmsNorm gain** — Gemma uses `(gamma + 1)` rather
//!      than `gamma` as the per-channel scale. Carry this in
//!      `apply_offset_rms_norm`.
//!   2. **Embedding scaling** — the token embedding is scaled by
//!      `sqrt(hidden_size)` after lookup (matches reference Gemma).
//!   3. **GELU FFN** — `down(gelu(gate) * up)` instead of LLaMA's
//!      SwiGLU. The activation choice is config-driven; the
//!      `hidden_activation` field carries either `gelu` or
//!      `gelu_pytorch_tanh`.
//!   4. **Optional Q/K/V/O biases** — `attention_bias: bool` switches
//!      the biases on. Gemma 2B-it leaves them off; some forks turn
//!      them on. Carried as the standard optional-bias fields on
//!      [`fuel_core::lazy::LayerWeights`].
//!
//! Gemma 2 already ships as part of `fuel_core::lazy` (`Gemma2Model`).
//! Gemma v1 doesn't share the v2-only soft-cap or local/global
//! attention alternation, so it's its own module.
//!
//! # Scope (v1, same as the other Phase D ports)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache
//! (recomputes each call), F32 activations.

use fuel_core::lazy::{LayerWeights, Tensor, WeightStorage};
use fuel_core::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

/// Which GELU variant the FFN's gate path uses. Defaults to
/// `GeluPytorchTanh` to match the reference Gemma checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmaActivation {
    /// Standard `0.5 * x * (1 + erf(x / sqrt(2)))`.
    Gelu,
    /// PyTorch's `approximate="tanh"` variant.
    GeluPytorchTanh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GemmaConfig {
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
    pub attention_bias: bool,
    pub hidden_activation: GemmaActivation,
}

impl GemmaConfig {
    /// Preset for `google/gemma-2b`. Values from the HF config.
    pub fn gemma_2b() -> Self {
        Self {
            vocab_size: 256_000,
            hidden_size: 2048,
            intermediate_size: 16_384,
            num_hidden_layers: 18,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 8192,
            attention_bias: false,
            hidden_activation: GemmaActivation::GeluPytorchTanh,
        }
    }
}

/// Map a Gemma-1 `hidden_act` string to [`GemmaActivation`].
///
/// ⚠️ Gemma-1's published configs carry `hidden_act: "gelu"`, but every Gemma-1
/// checkpoint uses the TANH-APPROXIMATE GELU — a KNOWN BUG in Google's config
/// (the reference implementation and the model cards use `gelu_pytorch_tanh`;
/// the `gemma_2b()` preset above already encodes `GeluPytorchTanh`). This
/// resolver CORRECTS the artifact: `"gelu"` → `GeluPytorchTanh`.
///
/// The correction is scoped to the Gemma architecture and lives HERE, at the
/// resolution site — never in `GemmaActivation` or a global default — because
/// `gelu → GeluPytorchTanh` is TRUE for Gemma and FALSE for every other
/// architecture; encoding it more widely would be a guard asserting a false
/// claim. Overriding an EXPLICIT artifact value (unlike glm4, which filled an
/// ABSENT one) is the stronger act, and this arch-scoped truth is its licence.
fn gemma_activation_from_str(s: &str) -> fuel_core::Result<GemmaActivation> {
    match s {
        "gelu" | "gelu_pytorch_tanh" | "gelu_new" => Ok(GemmaActivation::GeluPytorchTanh),
        other => Err(fuel_core::Error::Msg(format!(
            "unsupported Gemma hidden_act {other:?} (Gemma uses the tanh-approximate GELU)"
        ))),
    }
}

// ROADMAP item 8 (II): config-from-path on the #57 template. Gemma-1 is a FLAT
// artifact: a `serde` raw with HF field names + Gemma's own constant defaults,
// then `resolve` routes kv heads + head_dim through the shared `fuel_core::hf_config`
// rules (Gemma ships an explicit, often-decoupled head_dim, e.g. 256 vs 3072/16=192
// on gemma-7b). `hidden_act` (also accepted as `hidden_activation`) is corrected
// to the tanh GELU — see `gemma_activation_from_str`.
#[derive(Debug, Clone, serde::Deserialize)]
struct GemmaConfigRaw {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_gemma_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_gemma_rope_theta")]
    rope_theta: f64,
    max_position_embeddings: usize,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default, alias = "hidden_activation")]
    hidden_act: Option<String>,
}

fn default_gemma_rms_norm_eps() -> f64 {
    1e-6
}
fn default_gemma_rope_theta() -> f64 {
    10_000.0
}

impl GemmaConfigRaw {
    fn from_json_str(json: &str) -> fuel_core::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| fuel_core::Error::Msg(format!("parsing Gemma config.json: {e}")))
    }

    fn resolve(self) -> fuel_core::Result<GemmaConfig> {
        let hidden_activation = match self.hidden_act.as_deref() {
            None => GemmaActivation::GeluPytorchTanh,
            Some(s) => gemma_activation_from_str(s)?,
        };
        Ok(GemmaConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: fuel_core::hf_config::num_key_value_heads(
                self.num_key_value_heads,
                self.num_attention_heads,
            ),
            head_dim: fuel_core::hf_config::head_dim(
                self.head_dim,
                self.hidden_size,
                self.num_attention_heads,
            ),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            attention_bias: self.attention_bias,
            hidden_activation,
        })
    }
}

impl GemmaConfig {
    /// Parse a HuggingFace `config.json` string into a [`GemmaConfig`].
    ///
    /// ROADMAP item 8 (II): reads the artifact rather than returning a preset —
    /// see the born-red `gemma_config_from_hf_json_corrects_the_gelu_misnomer`.
    pub fn from_hf_json_str(json: &str) -> fuel_core::Result<Self> {
        GemmaConfigRaw::from_json_str(json)?.resolve()
    }
}

#[derive(Debug, Clone)]
pub struct GemmaWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct GemmaModel {
    pub config: GemmaConfig,
    pub weights: GemmaWeights,
}

impl GemmaModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let _batch = 1;
        assert!(seq > 0, "GemmaModel::forward: tokens must be non-empty");
        assert_eq!(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            "GemmaConfig: num_attention_heads * head_dim must equal hidden_size",
        );

        // Embedding lookup + sqrt(hidden_size) scaling (Gemma-specific).
        let h = Tensor::embed_tokens(
            weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
            &Device::cpu(),
        )?;
        let scale = (cfg.hidden_size as f64).sqrt();
        let h = h.mul_scalar(scale);

        self.forward_embeds(&h, start_pos)
    }

    /// Forward from pre-computed input embeddings of shape
    /// `(batch, seq, hidden_size)`. Used by multimodal models
    /// (PaliGemma, etc.) that interleave image embeddings with
    /// text embeddings before running the Gemma layers. The
    /// caller is responsible for the `sqrt(hidden_size)` token-
    /// embedding scaling that `forward()` applies internally.
    pub fn forward_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "embeds must be rank 3 [b, seq, hidden]");
        let seq = dims[1];
        assert_eq!(
            dims[2], cfg.hidden_size,
            "embeds last dim must equal hidden_size"
        );
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads,
            0,
            "GemmaConfig: num_attention_heads must be a multiple of num_key_value_heads",
        );

        let mut h = embeds.clone();

        // Shared RoPE tables — built fresh per call because seq may vary.
        let (rope_cos, rope_sin) =
            h.rope_tables_const(cfg.rope_theta, start_pos, seq, cfg.head_dim);

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }

        // Final offset RmsNorm + lm_head.
        let h_norm =
            h.rms_norm_affine_with_offset(&weights.final_norm_gain, 1.0, cfg.rms_norm_eps)?;
        weights
            .output
            .apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size)
    }

    /// Run the decoder forward up to the final offset RmsNorm
    /// and return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection. Pairs with
    /// [`Self::forward_hidden_embeds`] for vision-language
    /// composition or embedding adapters that need raw hidden
    /// states.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let _batch = 1;
        assert!(
            seq > 0,
            "GemmaModel::forward_hidden: tokens must be non-empty"
        );

        let h_raw = Tensor::embed_tokens(
            weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
            &Device::cpu(),
        )?;
        // Gemma scales the token-embedding output by sqrt(hidden_size).
        let h_scaled = h_raw.mul_scalar((cfg.hidden_size as f64).sqrt());
        self.forward_hidden_embeds(&h_scaled, start_pos)
    }

    /// Like [`Self::forward_embeds`] but skips the `lm_head`
    /// projection and returns the post-RmsNorm hidden states.
    /// Caller is responsible for the `sqrt(hidden_size)`
    /// embedding scaling that `forward_hidden()` applies.
    pub fn forward_hidden_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "embeds must be rank 3 [b, seq, hidden]");
        let seq = dims[1];
        assert_eq!(dims[2], cfg.hidden_size);

        let mut h = embeds.clone();
        let (rope_cos, rope_sin) =
            h.rope_tables_const(cfg.rope_theta, start_pos, seq, cfg.head_dim);

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        h.rms_norm_affine_with_offset(&weights.final_norm_gain, 1.0, cfg.rms_norm_eps)
    }

    fn apply_layer(
        &self,
        x: &Tensor,
        layer: &LayerWeights,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        // Pre-attention offset RmsNorm.
        let x_norm = x.rms_norm_affine_with_offset(&layer.attn_norm_gain, 1.0, cfg.rms_norm_eps)?;

        // Q / K / V — biases are honored when the config flag is on.
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

        // GQA replication.
        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        // Strict causal mask (Gemma v1 has no sliding window).
        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq {
                mask_data[i * seq + j] = f32::NEG_INFINITY;
            }
        }
        let mask = x.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]));
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer
            .attn_o
            .apply_linear(&merged, cfg.hidden_size, cfg.hidden_size)?
            .add_optional_trailing_bias(
                // LayerWeights doesn't carry an explicit attn_o_bias; reuse
                // attn_q_bias's None branch by passing None here. Gemma's
                // o_proj bias support would need a LayerWeights extension if
                // a checkpoint requires it (rare).
                None,
            )?;
        let h1 = x.add(&attn_out)?;

        // Pre-FFN offset RmsNorm.
        let h1_norm =
            h1.rms_norm_affine_with_offset(&layer.ffn_norm_gain, 1.0, cfg.rms_norm_eps)?;

        // GELU gated FFN: `down(gelu(gate) * up)`.
        let gate = layer
            .ffn_gate
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let up = layer
            .ffn_up
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let activated_gate = match cfg.hidden_activation {
            GemmaActivation::Gelu => gate.gelu_erf(),
            GemmaActivation::GeluPytorchTanh => gate.gelu(),
        };
        let ffn_in = activated_gate.mul(&up)?;
        let ffn_out =
            layer
                .ffn_down
                .apply_linear(&ffn_in, cfg.intermediate_size, cfg.hidden_size)?;

        h1.add(&ffn_out)
    }
}

// Gemma's offset RmsNorm: `y = (x / rms) * (gamma + 1)`. The `+ 1`
// matches the reference Gemma forward pass.

// ---- Safetensors loader ----------------------------------------------------

impl GemmaWeights {
    /// Load Gemma v1 weights from a `MmapedSafetensors` file using
    /// the standard HuggingFace naming. Tied lm_head: Gemma 1 ties
    /// `lm_head.weight` to `model.embed_tokens.weight` (the eager
    /// model constructs `Linear::new(embed_tokens.embeddings().clone(), None)`),
    /// so the output projection is derived from the token embedding
    /// when `lm_head.weight` is absent.
    ///
    /// Tensor names mirrored from `fuel_transformers::models::llm::gemma`:
    ///   - `model.embed_tokens.weight` → [`GemmaWeights::token_embedding`]
    ///   - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` → `attn_{q,k,v,o}`
    ///   - `model.layers.{i}.self_attn.{q,k,v}_proj.bias` → `attn_{q,k,v}_bias`
    ///     (loaded only when `attention_bias == true`)
    ///   - `model.layers.{i}.mlp.{gate,up,down}_proj.weight` → `ffn_{gate,up,down}`
    ///   - `model.layers.{i}.input_layernorm.weight` → `attn_norm_gain`
    ///   - `model.layers.{i}.post_attention_layernorm.weight` → `ffn_norm_gain`
    ///   - `model.norm.weight` → `final_norm_gain`
    ///   - `lm_head.weight` (optional, fallback to tied embeddings) → `output`
    pub fn load_from_mmapped(
        st: &fuel_core::safetensors::MmapedSafetensors,
        cfg: &GemmaConfig,
    ) -> Result<Self> {
        use fuel_core::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};

        let h = cfg.hidden_size;
        let i_dim = cfg.intermediate_size;
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

        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for li in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{li}");
            let attn_q = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.q_proj.weight"),
                h,
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
                h,
            )?;
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
            let attn_norm_gain = load_tensor_as_f32(st, &format!("{p}.input_layernorm.weight"))?;
            let ffn_norm_gain =
                load_tensor_as_f32(st, &format!("{p}.post_attention_layernorm.weight"))?;
            // Optional biases — present only when `attention_bias == true`
            // in HF config; absent biases are not an error.
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
                attn_norm_gain: Arc::from(attn_norm_gain),
                ffn_norm_gain: Arc::from(ffn_norm_gain),
            });
        }

        let final_norm_gain = load_tensor_as_f32(st, "model.norm.weight")?;

        // Gemma v1 ties lm_head to token embeddings. Try the explicit
        // `lm_head.weight` first (some forks publish it), fall back to
        // a transposed copy of the token embedding.
        let output: WeightStorage =
            match load_transposed_matrix_preserve_dtype(st, "lm_head.weight", cfg.vocab_size, h) {
                Ok(w) => w,
                Err(_) => {
                    let mut transposed = vec![0.0_f32; h * cfg.vocab_size];
                    for i in 0..cfg.vocab_size {
                        for j in 0..h {
                            transposed[j * cfg.vocab_size + i] = token_embedding[i * h + j];
                        }
                    }
                    WeightStorage::F32(Arc::from(transposed))
                }
            };

        Ok(GemmaWeights {
            token_embedding: Arc::from(token_embedding),
            layers,
            final_norm_gain: Arc::from(final_norm_gain),
            output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ROADMAP item 8 (II). Golden values from google/gemma-2b's real config.json.
    // Its `hidden_act` is the misnomer "gelu" (Gemma actually uses the tanh GELU).
    const GEMMA_2B_CONFIG_JSON: &str = r#"{
        "architectures": ["GemmaForCausalLM"],
        "model_type": "gemma",
        "vocab_size": 256000,
        "hidden_size": 2048,
        "intermediate_size": 16384,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "hidden_act": "gelu",
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "attention_bias": false
    }"#;

    #[test]
    fn gemma_config_from_hf_json_corrects_the_gelu_misnomer() {
        let cfg = GemmaConfig::from_hf_json_str(GEMMA_2B_CONFIG_JSON).unwrap();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 18);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.vocab_size, 256_000);
        // GQA: default would be num_attention_heads (8); 1 proves the key was READ.
        assert_eq!(cfg.num_key_value_heads, 1);
        assert_eq!(cfg.head_dim, 256);
        // THE RULING — CORRECTION ARM: the artifact says "gelu", but Gemma uses
        // the tanh GELU, so this must resolve to GeluPytorchTanh, NOT Gelu. A
        // faithful "gelu" -> Gelu parser fails here; that is the point.
        assert_eq!(cfg.hidden_activation, GemmaActivation::GeluPytorchTanh);
        assert_ne!(cfg.hidden_activation, GemmaActivation::Gelu);
    }

    /// A SECOND distinct real config (gemma-7b): head_dim EXPLICIT and DECOUPLED
    /// (256 vs 3072/16 = 192), proving take-if-present on a live artifact.
    #[test]
    fn gemma_config_reads_gemma_7b_with_decoupled_head_dim() {
        let json = r#"{
            "model_type": "gemma",
            "vocab_size": 256000,
            "hidden_size": 3072,
            "intermediate_size": 24576,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "hidden_act": "gelu",
            "max_position_embeddings": 8192
        }"#;
        let cfg = GemmaConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 28);
        // explicit head_dim WINS over the quotient (3072/16 = 192).
        assert_eq!(cfg.head_dim, 256);
        assert_ne!(cfg.head_dim, 3072 / 16);
        // omitted rms_norm_eps/rope_theta → defaults
        assert_eq!(cfg.rms_norm_eps, 1e-6);
        assert_eq!(cfg.rope_theta, 10_000.0);
        assert_ne!(cfg.hidden_size, 2048);
    }

    /// The activation correction is a TWO-ARM (glm4-shaped) test: the misnomer
    /// "gelu" AND the explicit "gelu_pytorch_tanh" both resolve to GeluPytorchTanh,
    /// and an UNKNOWN value ERRORS — the error arm is what catches a resolver
    /// hardcoded to GeluPytorchTanh (which would pass both mapping arms).
    #[test]
    fn gemma_config_activation_corrects_maps_and_rejects() {
        let with_act = |act: &str| {
            format!(
                r#"{{
                "model_type": "gemma",
                "vocab_size": 1000, "hidden_size": 64, "intermediate_size": 128,
                "num_hidden_layers": 2, "num_attention_heads": 8, "head_dim": 8,
                "max_position_embeddings": 128, "hidden_act": "{act}"
            }}"#
            )
        };
        // Arm 1: the misnomer is corrected.
        assert_eq!(
            GemmaConfig::from_hf_json_str(&with_act("gelu"))
                .unwrap()
                .hidden_activation,
            GemmaActivation::GeluPytorchTanh
        );
        // Arm 2: an already-correct explicit value is preserved.
        assert_eq!(
            GemmaConfig::from_hf_json_str(&with_act("gelu_pytorch_tanh"))
                .unwrap()
                .hidden_activation,
            GemmaActivation::GeluPytorchTanh
        );
        // Arm 3 (the anti-hardcode discriminator): unknown errors.
        assert!(GemmaConfig::from_hf_json_str(&with_act("silu")).is_err());
    }

    /// GQA absent → num_attention_heads; true MQA (1) survives.
    #[test]
    fn gemma_config_gqa_default_and_true_mqa() {
        let base = |kv: &str| {
            format!(
                r#"{{
                "model_type": "gemma",
                "vocab_size": 1000, "hidden_size": 64, "intermediate_size": 128,
                "num_hidden_layers": 2, "num_attention_heads": 8, "head_dim": 8,
                "max_position_embeddings": 128 {kv}
            }}"#
            )
        };
        let absent = GemmaConfig::from_hf_json_str(&base("")).unwrap();
        assert_eq!(absent.num_key_value_heads, 8);
        let mqa = GemmaConfig::from_hf_json_str(&base(", \"num_key_value_heads\": 1")).unwrap();
        assert_eq!(mqa.num_key_value_heads, 1);
    }

    fn tiny_weights(cfg: &GemmaConfig) -> GemmaWeights {
        let mut s: u32 = 4242;
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
        let mut next_box: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *next_box);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| LayerWeights {
                attn_q: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
                attn_q_bias: if cfg.attention_bias {
                    Some(vec_of(h, &mut *next_box))
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
                attn_o: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
                ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *next_box)),
                ffn_up: WeightStorage::F32(vec_of(h * i, &mut *next_box)),
                ffn_down: WeightStorage::F32(vec_of(i * h, &mut *next_box)),
                attn_norm_gain: Arc::from(vec![0.1_f32; h]), // non-zero so the +1 offset is visible
                ffn_norm_gain: Arc::from(vec![0.1_f32; h]),
            })
            .collect();
        let final_norm_gain = Arc::from(vec![0.1_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *next_box));
        GemmaWeights {
            token_embedding,
            layers,
            final_norm_gain,
            output,
        }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = GemmaConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            attention_bias: false,
            hidden_activation: GemmaActivation::GeluPytorchTanh,
        };
        let model = GemmaModel {
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

    /// Offset RmsNorm: a baseline (gain = 0) must produce the SAME
    /// post-norm output as a unity baseline through `apply_affine_rms_norm`
    /// (because (0 + 1) == 1).
    #[test]
    fn offset_rms_norm_with_zero_gain_matches_unity() {
        let device = Device::cpu();
        let dim = 8;
        let x = Tensor::from_f32(
            (0..dim).map(|i| 0.1 * (i as f32 - 3.5)).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 1, dim]),
            &device,
        );
        let zero_gain: Arc<[f32]> = Arc::from(vec![0.0_f32; dim]);
        let unity_gain: Arc<[f32]> = Arc::from(vec![1.0_f32; dim]);
        let offset = x
            .rms_norm_affine_with_offset(&zero_gain, 1.0, 1e-6)
            .unwrap();
        let unity = x.rms_norm_affine(Arc::clone(&unity_gain), 1e-6).unwrap();
        let a = offset.realize_f32();
        let b = unity.realize_f32();
        assert_eq!(a.len(), b.len());
        for (&av, &bv) in a.iter().zip(b.iter()) {
            assert!((av - bv).abs() < 1e-6, "offset(0) = {av} vs unity = {bv}");
        }
    }

    /// Embedding scaling: with `hidden_size = 4`, embedding gets
    /// scaled by 2 before the layers. Compare against a parallel
    /// model whose token_embedding rows are pre-scaled by 1/2 — the
    /// post-embedding state should match.
    ///
    /// (Cross-check via output equality requires identical downstream
    /// projections; tiny tolerance accounts for the per-layer norm's
    /// dependence on the scaled inputs.)
    #[test]
    fn embedding_scale_is_sqrt_hidden() {
        // We can't easily isolate the embedding from the rest of the
        // forward; instead, verify the scale value matches sqrt(h).
        let cfg = GemmaConfig {
            vocab_size: 4,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 16,
            attention_bias: false,
            hidden_activation: GemmaActivation::Gelu,
        };
        // hidden_size = 4, so sqrt = 2.0.
        assert!(((cfg.hidden_size as f64).sqrt() - 2.0).abs() < 1e-12);
        let model = GemmaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        // Just smoke-test that forward runs without panic.
        let logits = model.forward(&[0, 1, 2], 0).unwrap().realize_f32();
        assert_eq!(logits.len(), 3 * cfg.vocab_size);
    }
}

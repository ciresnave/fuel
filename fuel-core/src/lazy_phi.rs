//! Phi (Phi-1.5 / Phi-2) decoder ported to the lazy-graph API.
//!
//! Phase D specialized port. Microsoft's older Phi series (before
//! Phi-3) has its own architectural signature distinct from the
//! Llama family:
//!
//!   1. **Parallel attention + MLP** — same residual block as
//!      Falcon: `out = residual + attn(LN(x)) + mlp(LN(x))`.
//!      A single `input_layernorm` feeds both paths. No
//!      separate `post_attention_layernorm`.
//!   2. **Partial rotary** — only the first
//!      `partial_rotary_factor * head_dim` features of each
//!      head are rotated; the tail passes through unchanged.
//!      Same shape as the StableLM / Persimmon partial rotary
//!      handled by [`LazyTensor::rope_partial`].
//!   3. **LayerNorm (with bias) on hidden_size** — not RmsNorm.
//!   4. **Sequential MLP** — `fc2(act(fc1(x)))`. No SwiGLU gate.
//!      Activation is config-driven (the reference Phi-2 uses
//!      "new-GELU"; we expose both Gelu and GeluPytorchTanh
//!      variants and treat them as the same approximation
//!      family — the underlying `LazyTensor::gelu()` is the
//!      tanh-approximation, matching `Activation::NewGelu`).
//!   5. **All linear layers carry biases** — Q/K/V/dense and
//!      fc1/fc2 and lm_head all include bias terms.
//!
//! The `qk_layernorm` flag is exposed but v1 only supports
//! `qk_layernorm == false` (which is the Phi-2 default — the
//! flag was added for some Phi-2 fine-tunes; the reference
//! checkpoint ships with it off).
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache
//! (recomputes each call), F32 activations.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

/// Which GELU variant the MLP's activation uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiActivation {
    /// Standard ERF GELU `0.5 * x * (1 + erf(x / sqrt(2)))`.
    Gelu,
    /// PyTorch's `approximate="tanh"` variant — same family as
    /// `NewGelu` in HF transformers. Default for Phi-2.
    GeluPytorchTanh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhiConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Defaults to `num_attention_heads` if `None` (MHA, no GQA).
    pub num_key_value_heads: Option<usize>,
    pub head_dim: usize,
    pub layer_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    /// Fraction of `head_dim` that gets rotated. Phi-2 uses 0.4.
    pub partial_rotary_factor: f64,
    /// v1 only supports `false`. `true` would require a per-head
    /// LayerNorm applied post-reshape — left to follow-up.
    pub qk_layernorm: bool,
    pub hidden_activation: PhiActivation,
}

impl PhiConfig {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
    pub fn rope_dim(&self) -> usize {
        let r = (self.partial_rotary_factor * self.head_dim as f64) as usize;
        // RoPE expects an even rope_dim (pairs of features).
        r & !1
    }
}

#[derive(Debug, Clone)]
pub struct PhiLayerWeights {
    pub input_ln_gain: Arc<[f32]>,
    pub input_ln_bias: Arc<[f32]>,
    pub attn_q: WeightStorage,
    pub attn_q_bias: Arc<[f32]>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Arc<[f32]>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Arc<[f32]>,
    /// Phi calls the output projection `dense`.
    pub attn_dense: WeightStorage,
    pub attn_dense_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct PhiWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<PhiLayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub lm_head: WeightStorage,
    /// Optional `lm_head` bias. HF Phi-2 ships with this bias present;
    /// some Phi forks (Dolphin-Phi, fine-tunes) drop it. Loaders MUST
    /// produce `None` rather than zero-padded all-zeros when the bias
    /// is absent on disk — this preserves the no-bias semantics
    /// instead of doing a redundant broadcast-add of zeros and avoids
    /// a panic on checkpoints that genuinely lack the parameter.
    pub lm_head_bias: Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct PhiModel {
    pub config: PhiConfig,
    pub weights: PhiWeights,
}

impl PhiModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        match &weights.lm_head_bias {
            Some(b) => weights.lm_head.apply_linear_with_bias(
                &h_norm, cfg.hidden_size, cfg.vocab_size, Arc::clone(b),
            ),
            None => Ok(weights.lm_head.apply_linear(
                &h_norm, cfg.hidden_size, cfg.vocab_size,
            )),
        }
    }

    /// Run the decoder forward up to the final LayerNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection AND its bias.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "PhiModel: tokens must be non-empty");
        assert_eq!(
            cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size,
            "PhiConfig: num_attention_heads * head_dim must equal hidden_size",
        );
        assert!(
            !cfg.qk_layernorm,
            "PhiModel v1: qk_layernorm not yet supported (reference Phi-2 sets it false)",
        );

        let mut h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;

        let rope_dim = cfg.rope_dim();
        assert!(
            rope_dim > 0 && rope_dim <= cfg.head_dim,
            "PhiConfig: rope_dim ({}) out of [1, head_dim ({})]",
            rope_dim, cfg.head_dim,
        );
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, rope_dim,
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        Ok(h.layer_norm_affine(std::sync::Arc::clone(&weights.final_ln_gain), std::sync::Arc::clone(&weights.final_ln_bias), cfg.layer_norm_eps)?)
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &PhiLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let n_kv = cfg.num_kv_heads();
        let kv_dim = n_kv * cfg.head_dim;
        let rope_dim = cfg.rope_dim();

        // Single LN feeds BOTH attention and MLP paths (Phi parallel block).
        let x_norm = x.layer_norm_affine(std::sync::Arc::clone(&layer.input_ln_gain), std::sync::Arc::clone(&layer.input_ln_bias), cfg.layer_norm_eps)?;

        // ---- Attention path -------------------------------------------------
        let q = layer.attn_q.apply_linear_with_bias(&x_norm, cfg.hidden_size, cfg.hidden_size, std::sync::Arc::clone(&layer.attn_q_bias))?;
        let k = layer.attn_k.apply_linear_with_bias(&x_norm, cfg.hidden_size, kv_dim, std::sync::Arc::clone(&layer.attn_k_bias))?;
        let v = layer.attn_v.apply_linear_with_bias(&x_norm, cfg.hidden_size, kv_dim, std::sync::Arc::clone(&layer.attn_v_bias))?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(n_kv, cfg.head_dim)?;
        let v = v.split_heads(n_kv, cfg.head_dim)?;

        // Partial rotary on the first `rope_dim` features.
        let q_r = q.rope_partial(rope_cos, rope_sin, rope_dim)?;
        let k_r = k.rope_partial(rope_cos, rope_sin, rope_dim)?;

        // GQA replication (Phi-2 uses MHA so n_rep = 1 usually).
        let n_rep = cfg.num_attention_heads / n_kv;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        // Strict causal mask.
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
        let attn_out = layer.attn_dense.apply_linear_with_bias(&merged, cfg.hidden_size, cfg.hidden_size, std::sync::Arc::clone(&layer.attn_dense_bias))?;

        // ---- MLP path (uses the SAME x_norm) --------------------------------
        let fc1_out = layer.fc1.apply_linear_with_bias(&x_norm, cfg.hidden_size, cfg.intermediate_size, std::sync::Arc::clone(&layer.fc1_bias))?;
        let activated = match cfg.hidden_activation {
            PhiActivation::Gelu => fc1_out.gelu_erf(),
            PhiActivation::GeluPytorchTanh => fc1_out.gelu(),
        };
        let mlp_out = layer.fc2.apply_linear_with_bias(&activated, cfg.intermediate_size, cfg.hidden_size, std::sync::Arc::clone(&layer.fc2_bias))?;

        // Parallel combine: residual + attn + mlp.
        x.add(&attn_out)?.add(&mlp_out)
    }
}

// ---- HuggingFace config + safetensors loading ----------------------------

impl PhiConfig {
    /// Parse a Phi-2 `config.json` from HuggingFace.
    ///
    /// HF native names mapped to standalone field names:
    ///   `hidden_size` → `hidden_size`,
    ///   `intermediate_size` → `intermediate_size`,
    ///   `num_hidden_layers` → `num_hidden_layers`,
    ///   `num_attention_heads` → `num_attention_heads`,
    ///   `num_key_value_heads` → `num_key_value_heads` (Option — falls
    ///   back to `num_attention_heads` for MHA configs),
    ///   `head_dim` → `head_dim` (defaults to `hidden_size / num_attention_heads`),
    ///   `layer_norm_eps` → `layer_norm_eps` (defaults to 1e-5),
    ///   `rope_theta` → `rope_theta` (defaults to 10000.0),
    ///   `max_position_embeddings` → `max_position_embeddings`,
    ///   `partial_rotary_factor` → `partial_rotary_factor` (defaults to 0.4 — Phi-2's),
    ///   `qk_layernorm` → `qk_layernorm` (defaults to false; v1 only
    ///   supports false — set to true bails at load time).
    pub fn from_hf_json_str(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| crate::Error::Msg(format!("parsing Phi config.json: {e}")))?;
        let get_usize = |key: &str| -> Result<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| {
                    crate::Error::Msg(format!("Phi config.json: missing/invalid {key:?}"))
                })
        };
        let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };
        let get_bool = |key: &str| -> Option<bool> { v.get(key).and_then(|x| x.as_bool()) };

        let vocab_size = get_usize("vocab_size")?;
        let hidden_size = get_usize("hidden_size")?;
        let intermediate_size = get_usize("intermediate_size")?;
        let num_hidden_layers = get_usize("num_hidden_layers")?;
        let num_attention_heads = get_usize("num_attention_heads")?;
        let num_key_value_heads = v
            .get("num_key_value_heads")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize);
        let head_dim = v
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(hidden_size / num_attention_heads);
        let layer_norm_eps = get_f64("layer_norm_eps").unwrap_or(1e-5);
        let rope_theta = get_f64("rope_theta").unwrap_or(10_000.0);
        let max_position_embeddings = get_usize("max_position_embeddings")?;
        let partial_rotary_factor = get_f64("partial_rotary_factor").unwrap_or(0.4);
        let qk_layernorm = get_bool("qk_layernorm").unwrap_or(false);
        if qk_layernorm {
            return Err(crate::Error::Msg(
                "Phi: qk_layernorm=true is not yet supported by the standalone port (v1)".into(),
            ).bt());
        }

        Ok(PhiConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            layer_norm_eps,
            rope_theta,
            max_position_embeddings,
            partial_rotary_factor,
            qk_layernorm,
            hidden_activation: PhiActivation::GeluPytorchTanh,
        })
    }
}

impl PhiWeights {
    /// Load Phi-2 weights from memory-mapped safetensors. Loads weights
    /// as `WeightStorage::F32` after on-load transposition from HF's
    /// `[out, in]` layout to fuel's `[in, out]`.
    ///
    /// `lm_head.bias` is loaded as `None` when absent on disk — matches
    /// the `Option<Arc<[f32]>>` field type and avoids the previous
    /// panicking design (closed in commit b723ddf8).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &PhiConfig,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let n_kv = cfg.num_kv_heads();
        let kv_dim = n_kv * cfg.head_dim;

        let token_embedding = Arc::from(
            crate::lazy::load_tensor_as_f32(st, "model.embed_tokens.weight")?,
        );

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for li in 0..cfg.num_hidden_layers {
            let load_lin = |suffix: &str, out_f: usize, in_f: usize| -> Result<WeightStorage> {
                let name = format!("model.layers.{li}.{suffix}");
                let v = crate::lazy::load_transposed_matrix(st, &name, out_f, in_f)?;
                Ok(WeightStorage::F32(Arc::from(v)))
            };
            let load_vec = |suffix: &str| -> Result<Arc<[f32]>> {
                let name = format!("model.layers.{li}.{suffix}");
                Ok(Arc::from(crate::lazy::load_tensor_as_f32(st, &name)?))
            };

            layers.push(PhiLayerWeights {
                input_ln_gain: load_vec("input_layernorm.weight")?,
                input_ln_bias: load_vec("input_layernorm.bias")?,
                attn_q: load_lin("self_attn.q_proj.weight", h, h)?,
                attn_q_bias: load_vec("self_attn.q_proj.bias")?,
                attn_k: load_lin("self_attn.k_proj.weight", kv_dim, h)?,
                attn_k_bias: load_vec("self_attn.k_proj.bias")?,
                attn_v: load_lin("self_attn.v_proj.weight", kv_dim, h)?,
                attn_v_bias: load_vec("self_attn.v_proj.bias")?,
                attn_dense: load_lin("self_attn.dense.weight", h, h)?,
                attn_dense_bias: load_vec("self_attn.dense.bias")?,
                fc1: load_lin("mlp.fc1.weight", i, h)?,
                fc1_bias: load_vec("mlp.fc1.bias")?,
                fc2: load_lin("mlp.fc2.weight", h, i)?,
                fc2_bias: load_vec("mlp.fc2.bias")?,
            });
        }

        let final_ln_gain = Arc::from(
            crate::lazy::load_tensor_as_f32(st, "model.final_layernorm.weight")?,
        );
        let final_ln_bias = Arc::from(
            crate::lazy::load_tensor_as_f32(st, "model.final_layernorm.bias")?,
        );
        let lm_head = WeightStorage::F32(Arc::from(
            crate::lazy::load_transposed_matrix(st, "lm_head.weight", cfg.vocab_size, h)?,
        ));
        let lm_head_bias = crate::lazy::load_tensor_as_f32(st, "lm_head.bias")
            .ok()
            .map(Arc::from);

        Ok(PhiWeights {
            token_embedding,
            layers,
            final_ln_gain,
            final_ln_bias,
            lm_head,
            lm_head_bias,
        })
    }
}

impl PhiModel {
    /// Download a Phi-2 checkpoint from the HuggingFace Hub and build
    /// a ready-to-forward `PhiModel`.
    pub fn from_hub(repo_id: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| crate::Error::Msg(format!("hf-hub config.json: {e}")))?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let config = PhiConfig::from_hf_json_str(&config_str)?;

        let weight_paths: Vec<std::path::PathBuf> = match repo.get("model.safetensors.index.json") {
            Ok(index_path) => {
                let index_str = std::fs::read_to_string(&index_path)?;
                let index: serde_json::Value = serde_json::from_str(&index_str)
                    .map_err(|e| crate::Error::Msg(format!("parsing index: {e}")))?;
                let weight_map = index
                    .get("weight_map")
                    .and_then(|x| x.as_object())
                    .ok_or_else(|| crate::Error::Msg("index: missing weight_map".into()))?;
                let mut unique = std::collections::HashSet::new();
                for v in weight_map.values() {
                    if let Some(s) = v.as_str() {
                        unique.insert(s.to_string());
                    }
                }
                let mut paths = Vec::new();
                for name in &unique {
                    paths.push(
                        repo.get(name)
                            .map_err(|e| crate::Error::Msg(format!("hf-hub {name}: {e}")))?,
                    );
                }
                paths
            }
            Err(_) => vec![repo
                .get("model.safetensors")
                .map_err(|e| {
                    crate::Error::Msg(format!("hf-hub model.safetensors: {e}"))
                })?],
        };

        let st = unsafe { crate::safetensors::MmapedSafetensors::multi(&weight_paths) }?;
        let weights = PhiWeights::load_from_mmapped(&st, &config)?;

        Ok(PhiModel { config, weights })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &PhiConfig) -> PhiWeights {
        let mut s: u32 = 31337;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let n_kv = cfg.num_kv_heads();
        let kv = n_kv * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<PhiLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| PhiLayerWeights {
                input_ln_gain: Arc::from(vec![1.0_f32; h]),
                input_ln_bias: Arc::from(vec![0.0_f32; h]),
                attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                attn_q_bias: vec_of(h, &mut *nb),
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_k_bias: vec_of(kv, &mut *nb),
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_v_bias: vec_of(kv, &mut *nb),
                attn_dense: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                attn_dense_bias: vec_of(h, &mut *nb),
                fc1: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                fc1_bias: vec_of(i, &mut *nb),
                fc2: WeightStorage::F32(vec_of(i * h, &mut *nb)),
                fc2_bias: vec_of(h, &mut *nb),
            })
            .collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let lm_head = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        let lm_head_bias = Some(vec_of(cfg.vocab_size, &mut *nb));
        PhiWeights {
            token_embedding, layers,
            final_ln_gain, final_ln_bias,
            lm_head, lm_head_bias,
        }
    }

    #[test]
    fn from_hf_json_str_parses_canonical_phi2_fields() {
        // Excerpt from microsoft/phi-2/config.json (excluding the
        // wrapper fields like `model_type`, `transformers_version`,
        // etc. — only the architecture-relevant subset is needed).
        let json = r#"{
            "vocab_size": 51200,
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "head_dim": 80,
            "layer_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 2048,
            "partial_rotary_factor": 0.4
        }"#;
        let cfg = PhiConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.vocab_size, 51200);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.intermediate_size, 10240);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, Some(32));
        assert_eq!(cfg.head_dim, 80);
        assert!((cfg.layer_norm_eps - 1e-5).abs() < 1e-12);
        assert!((cfg.rope_theta - 10000.0).abs() < 1e-9);
        assert_eq!(cfg.max_position_embeddings, 2048);
        assert!((cfg.partial_rotary_factor - 0.4).abs() < 1e-9);
        assert!(!cfg.qk_layernorm);
        // rope_dim = (0.4 * 80) & !1 = 32 — even
        assert_eq!(cfg.rope_dim(), 32);
    }

    #[test]
    fn from_hf_json_str_applies_optional_defaults() {
        // Minimal Phi-2-like config (no rope_theta, no head_dim,
        // no partial_rotary_factor, no GQA).
        let json = r#"{
            "vocab_size": 51200,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "max_position_embeddings": 2048
        }"#;
        let cfg = PhiConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.num_key_value_heads, None);  // MHA — no GQA
        assert_eq!(cfg.head_dim, 64);  // 1024 / 16
        assert!((cfg.layer_norm_eps - 1e-5).abs() < 1e-12);
        assert!((cfg.rope_theta - 10000.0).abs() < 1e-9);
        assert!((cfg.partial_rotary_factor - 0.4).abs() < 1e-9);
        assert!(!cfg.qk_layernorm);
    }

    #[test]
    fn from_hf_json_str_bails_on_qk_layernorm_true() {
        // v1 only supports qk_layernorm = false; setting it true must
        // fail explicitly (not silently produce wrong results).
        let json = r#"{
            "vocab_size": 51200,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "max_position_embeddings": 2048,
            "qk_layernorm": true
        }"#;
        let result = PhiConfig::from_hf_json_str(json);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("qk_layernorm"),
            "expected error to mention qk_layernorm, got: {err}",
        );
    }

    fn tiny_config() -> PhiConfig {
        PhiConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4,
            num_key_value_heads: None, head_dim: 4,
            layer_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 64,
            partial_rotary_factor: 0.5, // rope_dim = 2
            qk_layernorm: false,
            hidden_activation: PhiActivation::GeluPytorchTanh,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = PhiModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// No-lm-head-bias checkpoints (some Phi fine-tunes drop the
    /// bias) must run without panic via the `None` branch.
    /// Regression guard for the previous panicking design where
    /// `lm_head_bias` was a non-Optional `Arc<[f32]>`.
    #[test]
    fn forward_without_lm_head_bias() {
        let cfg = tiny_config();
        let mut weights = tiny_weights(&cfg);
        weights.lm_head_bias = None;
        let model = PhiModel { config: cfg.clone(), weights };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// With lm_head_bias = Some(zeros), output must equal the
    /// no-bias forward. Verifies the Option branch produces
    /// arithmetically equivalent results.
    #[test]
    fn lm_head_bias_zero_matches_none() {
        let cfg = tiny_config();
        let base = tiny_weights(&cfg);

        // Path A: bias = None.
        let mut weights_none = base.clone();
        weights_none.lm_head_bias = None;
        let m_none = PhiModel { config: cfg.clone(), weights: weights_none };
        let a = m_none.forward(&[1, 2, 3], 0).unwrap().realize_f32();

        // Path B: bias = Some(zeros).
        let mut weights_zero = base.clone();
        weights_zero.lm_head_bias = Some(Arc::from(vec![0.0_f32; cfg.vocab_size]));
        let m_zero = PhiModel { config: cfg.clone(), weights: weights_zero };
        let b = m_zero.forward(&[1, 2, 3], 0).unwrap().realize_f32();

        assert_eq!(a.len(), b.len());
        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            assert!((av - bv).abs() < 1e-5, "logit[{i}]: none={av} vs zero-bias={bv}");
        }
    }

    /// rope_dim equal to head_dim must run (full rotary degenerate case).
    #[test]
    fn full_rotary_factor() {
        let mut cfg = tiny_config();
        cfg.partial_rotary_factor = 1.0;
        assert_eq!(cfg.rope_dim(), cfg.head_dim);
        let model = PhiModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3], 0).unwrap().realize_f32();
        assert_eq!(logits.len(), 3 * cfg.vocab_size);
    }

    /// Parallel block must NOT collapse to sequential: differentiating
    /// attention and MLP at the residual sum is the structural test.
    /// Zero out fc1/fc2 (kills MLP path) and confirm output equals
    /// `residual + attn`. Then zero out Q/K/V/dense (kills attn) and
    /// confirm output equals `residual + mlp`. Both must be possible
    /// independently.
    #[test]
    fn parallel_paths_are_additive() {
        let cfg = PhiConfig { num_hidden_layers: 1, ..tiny_config() };
        let base = tiny_weights(&cfg);

        let mut no_mlp = base.clone();
        let zero = |n: usize| WeightStorage::F32(Arc::from(vec![0.0_f32; n]));
        let zero_b = |n: usize| Arc::from(vec![0.0_f32; n]) as Arc<[f32]>;
        no_mlp.layers[0].fc1 = zero(cfg.hidden_size * cfg.intermediate_size);
        no_mlp.layers[0].fc1_bias = zero_b(cfg.intermediate_size);
        no_mlp.layers[0].fc2 = zero(cfg.intermediate_size * cfg.hidden_size);
        no_mlp.layers[0].fc2_bias = zero_b(cfg.hidden_size);

        let mut no_attn = base.clone();
        no_attn.layers[0].attn_q = zero(cfg.hidden_size * cfg.hidden_size);
        no_attn.layers[0].attn_q_bias = zero_b(cfg.hidden_size);
        no_attn.layers[0].attn_k = zero(cfg.hidden_size * cfg.hidden_size);
        no_attn.layers[0].attn_k_bias = zero_b(cfg.hidden_size);
        no_attn.layers[0].attn_v = zero(cfg.hidden_size * cfg.hidden_size);
        no_attn.layers[0].attn_v_bias = zero_b(cfg.hidden_size);
        no_attn.layers[0].attn_dense = zero(cfg.hidden_size * cfg.hidden_size);
        no_attn.layers[0].attn_dense_bias = zero_b(cfg.hidden_size);

        // Both should produce finite, non-trivial outputs different
        // from each other (different paths active).
        let m_no_mlp = PhiModel { config: cfg.clone(), weights: no_mlp };
        let m_no_attn = PhiModel { config: cfg.clone(), weights: no_attn };
        let toks = [3_u32, 7, 2];
        let a = m_no_mlp.forward(&toks, 0).unwrap().realize_f32();
        let b = m_no_attn.forward(&toks, 0).unwrap().realize_f32();
        assert!(a.iter().all(|v| v.is_finite()));
        assert!(b.iter().all(|v| v.is_finite()));
        let mut max_diff = 0.0_f32;
        for (av, bv) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((av - bv).abs());
        }
        assert!(max_diff > 1e-6,
            "no-mlp vs no-attn must produce different outputs: {max_diff}");
    }

    #[test]
    fn num_kv_heads_default_is_num_attention_heads() {
        let cfg = tiny_config();
        assert_eq!(cfg.num_kv_heads(), cfg.num_attention_heads);
    }

    /// `forward_hidden` returns post-LayerNorm hidden states
    /// `(1, seq, hidden_size)` without the lm_head matmul or
    /// its bias.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config();
        let model = PhiModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

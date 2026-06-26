//! Gemma 2 (Google DeepMind 2024) ported to the lazy-graph API.
//!
//! Gemma 2 extends Gemma 1 with three architectural changes:
//!
//!   1. **Four RmsNorms per layer** (Gemma 1 has two). The
//!      sublayer structure is:
//!      `input_norm → attn → post_attn_norm → +residual →
//!       pre_ffn_norm → mlp → post_ffn_norm → +residual`.
//!      The "post_*" norms sit AFTER the sublayer's main op but
//!      BEFORE the residual add. Combined with input/pre norms,
//!      every sublayer is wrapped in `norm(sublayer(norm(x)))`.
//!   2. **Attention logit softcapping**: when
//!      `attn_logit_softcapping = Some(cap)`, raw attention
//!      scores `(Q @ K.T) * (1/sqrt(d))` are bounded by
//!      `tanh(scores / cap) * cap` BEFORE the mask is applied.
//!      Stabilizes training in deep models; runs at inference too.
//!   3. **Final logit softcapping**: same trick applied to the
//!      output `lm_head` logits before they're returned.
//!
//! Gemma 2 keeps Gemma 1's distinctive bits:
//!
//!   - **Embedding scaling**: hidden states multiplied by
//!     `sqrt(hidden_size)` after the embedding lookup.
//!   - **RmsNorm with `(gamma + 1.0)` multiplier** (the
//!     learnable weights center on 0, not 1).
//!   - GQA with `num_key_value_heads`, standard RoPE on Q/K,
//!     SwiGLU MLP with separate gate/up/down projections.
//!   - Tied embeddings (`lm_head.weight == embed_tokens.weight`).
//!
//! # Sliding window mask
//!
//! When `sliding_window = Some(w)`, the causal mask zeros out
//! position pairs `(i, j)` with `j + w < i` (in addition to
//! `j > i`). Gemma 2 alternates global / local layers in the
//! same way as Qwen2 in some configurations but the eager
//! port applies the sliding window globally per the config.
//! v1 follows the eager port: window applied to ALL layers
//! when the config has it.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache
//! (recomputes per call), F32. Returns vocab logits
//! `(1, seq, vocab_size)` with final softcapping applied.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub attention_bias: bool,
    pub max_position_embeddings: usize,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub sliding_window: Option<usize>,
}

impl Gemma2Config {
    /// `google/gemma-2-2b` preset (approximate; actual `config.json`
    /// from HuggingFace overrides).
    pub fn gemma2_2b() -> Self {
        Self {
            vocab_size: 256_000,
            hidden_size: 2_304,
            intermediate_size: 9_216,
            num_hidden_layers: 26,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            attention_bias: false,
            max_position_embeddings: 8_192,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            sliding_window: Some(4_096),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Gemma2LayerWeights {
    pub input_norm_gain: Arc<[f32]>,
    pub post_attn_norm_gain: Arc<[f32]>,
    pub pre_ffn_norm_gain: Arc<[f32]>,
    pub post_ffn_norm_gain: Arc<[f32]>,
    pub q: WeightStorage,
    pub q_bias: Option<Arc<[f32]>>,
    pub k: WeightStorage,
    pub k_bias: Option<Arc<[f32]>>,
    pub v: WeightStorage,
    pub v_bias: Option<Arc<[f32]>>,
    pub o: WeightStorage,
    pub o_bias: Option<Arc<[f32]>>,
    pub gate: WeightStorage,
    pub up: WeightStorage,
    pub down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Gemma2Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Gemma2LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Gemma2Model {
    pub config: Gemma2Config,
    pub weights: Gemma2Weights,
}

impl Gemma2Model {
    /// Forward pass: returns `(1, seq, vocab_size)`.
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant — returns the post-final-RmsNorm states
    /// `(1, seq, hidden_size)`. Skips the tied lm_head matmul and the
    /// final logit softcapping. Used by retrieval / embedding consumers.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips the token-embedding step and
    /// runs the decoder over pre-embedded inputs.
    ///
    /// `scaled_embeds` shape: `(1, seq, hidden_size)`. The caller
    /// must apply Gemma's `sqrt(hidden_size)` scaling before
    /// invoking — matches lazy_paligemma / lazy_voxtral convention.
    pub fn forward_embeds(
        &self, scaled_embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let h_norm = self.decode_from_scaled_embeds(scaled_embeds, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`]. Returns the
    /// post-final-RmsNorm states `(1, seq, hidden_size)`.
    pub fn forward_hidden_embeds(
        &self, scaled_embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.decode_from_scaled_embeds(scaled_embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder. Used by
    /// multimodal compositions. Returns shape `(1, seq, hidden_size)`;
    /// caller is responsible for the `sqrt(hidden_size)` scaling.
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
        // Tied lm_head reuses the token embedding weight (vocab, hidden) —
        // logits = h_norm @ embed.T. We assemble it as a Linear: hidden -> vocab.
        let lm_head_w = h_norm.const_f32_like(
            Arc::clone(&self.weights.token_embedding),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
        );
        let logits = h_norm.matmul(&lm_head_w.transpose()?)?;
        Ok(logits.softcap_optional(cfg.final_logit_softcapping))
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0, "Gemma2Model::forward: tokens must be non-empty");

        // ---- Embedding + sqrt(hidden_size) scaling -------------------------
        let h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;
        let h = h.mul_scalar((cfg.hidden_size as f64).sqrt());
        self.decode_from_scaled_embeds(&h, start_pos)
    }

    fn decode_from_scaled_embeds(
        &self, scaled_embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = scaled_embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(crate::Error::Msg(format!(
                "Gemma2Model::forward_embeds: expected scaled_embeds shape \
                 (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "Gemma2Model::forward_embeds: seq must be > 0".into(),
            ).bt());
        }
        let head_dim = cfg.head_dim;
        if cfg.num_attention_heads * head_dim != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "Gemma2Config: num_attention_heads * head_dim must equal hidden_size".into(),
            ).bt());
        }
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
            return Err(crate::Error::Msg(
                "Gemma2Config: num_attention_heads must be a multiple of num_key_value_heads".into(),
            ).bt());
        }
        let mut h = scaled_embeds.clone();

        // ---- Shared RoPE tables --------------------------------------------
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, head_dim,
        );

        // ---- Causal (optionally sliding) mask, shared across layers --------
        let mask = self.build_mask(&h, seq);

        // ---- Decoder blocks ------------------------------------------------
        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, &mask)?;
        }

        // ---- Final RmsNorm ------------------------------------------------
        h.rms_norm_affine_with_offset(&weights.final_norm_gain, 1.0, cfg.rms_norm_eps)
    }

    fn build_mask(&self, anchor: &LazyTensor, seq: usize) -> LazyTensor {
        let cfg = &self.config;
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                let masked = if j > i {
                    true
                } else if let Some(w) = cfg.sliding_window {
                    j + w < i
                } else {
                    false
                };
                if masked {
                    mask_data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        anchor.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]))
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &Gemma2LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * head_dim;

        // Attention sublayer: post_attn_norm(attn(input_norm(x))) + x.
        let x_in = x.rms_norm_affine_with_offset(&layer.input_norm_gain, 1.0, cfg.rms_norm_eps).unwrap();
        let q = layer.q.apply_linear(&x_in, cfg.hidden_size, cfg.hidden_size).add_optional_trailing_bias(layer.q_bias.as_ref())?;
        let k = layer.k.apply_linear(&x_in, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.k_bias.as_ref())?;
        let v = layer.v.apply_linear(&x_in, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.v_bias.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_full.transpose()?)?.mul_scalar(scale);
        // Attention logit softcapping (BEFORE the mask).
        let scores = scores.softcap_optional(cfg.attn_logit_softcapping);
        let scores = scores.broadcast_add(mask)?;
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v_full)?;
        let merged = ctx.merge_heads()?;
        let attn_out = layer.o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size).add_optional_trailing_bias(layer.o_bias.as_ref())?;
        // Post-attn norm BEFORE the residual add.
        let attn_post = attn_out.rms_norm_affine_with_offset(&layer.post_attn_norm_gain, 1.0, cfg.rms_norm_eps).unwrap();
        let h1 = x.add(&attn_post)?;

        // MLP sublayer: post_ffn_norm(mlp(pre_ffn_norm(h1))) + h1.
        let h1_in = h1.rms_norm_affine_with_offset(&layer.pre_ffn_norm_gain, 1.0, cfg.rms_norm_eps).unwrap();
        let gate = layer.gate.apply_linear(&h1_in, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.up.apply_linear(&h1_in, cfg.hidden_size, cfg.intermediate_size);
        // Gemma 2 default activation is GeGLU-tanh (HiddenAct::GeluPytorchTanh
        // in the config), but the original Gemma uses Silu. The eager config
        // carries the activation as a field; we follow gemma-2 default which
        // is GELU-tanh (used by the public release).
        let swi = gate.gelu().mul(&up)?;
        let ffn_out = layer.down.apply_linear(&swi, cfg.intermediate_size, cfg.hidden_size);
        let ffn_post = ffn_out.rms_norm_affine_with_offset(&layer.post_ffn_norm_gain, 1.0, cfg.rms_norm_eps).unwrap();
        h1.add(&ffn_post)
    }
}

/// Gemma-style RmsNorm: scale by `gamma + 1.0` rather than
/// `gamma`. The +1 is folded into a separate constant tensor
/// on the graph so the underlying RmsNorm op stays standard.

// ---- HuggingFace config + weight loading ----------------------------------

impl Gemma2Config {
    /// Parse a Gemma-2 `config.json` from HuggingFace.
    ///
    /// Field map (HF → standalone Gemma2Config):
    ///   `hidden_size` → `hidden_size`,
    ///   `intermediate_size` → `intermediate_size`,
    ///   `num_hidden_layers` → `num_hidden_layers`,
    ///   `num_attention_heads` → `num_attention_heads`,
    ///   `num_key_value_heads` → `num_key_value_heads` (defaults to
    ///   `num_attention_heads` when absent — Gemma 2 always sets it),
    ///   `head_dim` → `head_dim` (defaults to
    ///   `hidden_size / num_attention_heads` when absent),
    ///   `rms_norm_eps` → `rms_norm_eps` (defaults to 1e-6),
    ///   `rope_theta` → `rope_theta` (defaults to 10000.0),
    ///   `attention_bias` → `attention_bias` (defaults to false),
    ///   `max_position_embeddings` → `max_position_embeddings`,
    ///   `attn_logit_softcapping` → `attn_logit_softcapping`,
    ///   `final_logit_softcapping` → `final_logit_softcapping`,
    ///   `sliding_window` → `sliding_window`.
    pub fn from_hf_json_str(json: &str) -> crate::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| crate::Error::Msg(format!("parsing Gemma2 config.json: {e}")))?;
        let get_usize = |key: &str| -> crate::Result<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| {
                    crate::Error::Msg(format!("Gemma2 config.json: missing/invalid {key:?}"))
                })
        };
        let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };
        let get_bool = |key: &str| -> Option<bool> { v.get(key).and_then(|x| x.as_bool()) };

        let vocab_size = get_usize("vocab_size")?;
        let hidden_size = get_usize("hidden_size")?;
        let num_hidden_layers = get_usize("num_hidden_layers")?;
        let num_attention_heads = get_usize("num_attention_heads")?;
        let num_key_value_heads = v
            .get("num_key_value_heads")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(num_attention_heads);
        let intermediate_size = get_usize("intermediate_size")?;
        let head_dim = v
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(hidden_size / num_attention_heads);
        let rms_norm_eps = get_f64("rms_norm_eps").unwrap_or(1e-6);
        let rope_theta = get_f64("rope_theta").unwrap_or(10000.0);
        let attention_bias = get_bool("attention_bias").unwrap_or(false);
        let max_position_embeddings = get_usize("max_position_embeddings")?;
        let attn_logit_softcapping = get_f64("attn_logit_softcapping");
        let final_logit_softcapping = get_f64("final_logit_softcapping");
        let sliding_window = v
            .get("sliding_window")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize);

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            attention_bias,
            max_position_embeddings,
            attn_logit_softcapping,
            final_logit_softcapping,
            sliding_window,
        })
    }
}

impl Gemma2Weights {
    /// Load Gemma-2 weights from a memory-mapped safetensors file
    /// (or sharded set thereof). Weights are stored as
    /// `WeightStorage::F32` after on-load transposition from the
    /// HuggingFace `[out, in]` layout to fuel's `[in, out]` layout.
    ///
    /// Gemma 2 doesn't carry attention biases in the published
    /// checkpoints (`attention_bias=false`), so q/k/v/o biases are
    /// always loaded as `None`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &Gemma2Config,
    ) -> crate::Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let token_embedding = Arc::from(
            crate::lazy::load_tensor_as_f32(st, "model.embed_tokens.weight")?,
        );

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for li in 0..cfg.num_hidden_layers {
            let load_lin = |suffix: &str, out_f: usize, in_f: usize| -> crate::Result<WeightStorage> {
                let name = format!("model.layers.{li}.{suffix}");
                let v = crate::lazy::load_transposed_matrix(st, &name, out_f, in_f)?;
                Ok(WeightStorage::F32(Arc::from(v)))
            };
            let load_norm = |suffix: &str| -> crate::Result<Arc<[f32]>> {
                let name = format!("model.layers.{li}.{suffix}");
                Ok(Arc::from(crate::lazy::load_tensor_as_f32(st, &name)?))
            };

            layers.push(Gemma2LayerWeights {
                input_norm_gain: load_norm("input_layernorm.weight")?,
                post_attn_norm_gain: load_norm("post_attention_layernorm.weight")?,
                pre_ffn_norm_gain: load_norm("pre_feedforward_layernorm.weight")?,
                post_ffn_norm_gain: load_norm("post_feedforward_layernorm.weight")?,
                q: load_lin("self_attn.q_proj.weight", q_dim, h)?,
                q_bias: None,
                k: load_lin("self_attn.k_proj.weight", kv_dim, h)?,
                k_bias: None,
                v: load_lin("self_attn.v_proj.weight", kv_dim, h)?,
                v_bias: None,
                o: load_lin("self_attn.o_proj.weight", h, q_dim)?,
                o_bias: None,
                gate: load_lin("mlp.gate_proj.weight", i, h)?,
                up: load_lin("mlp.up_proj.weight", i, h)?,
                down: load_lin("mlp.down_proj.weight", h, i)?,
            });
        }

        let final_norm_gain = Arc::from(
            crate::lazy::load_tensor_as_f32(st, "model.norm.weight")?,
        );

        Ok(Gemma2Weights { token_embedding, layers, final_norm_gain })
    }
}

impl Gemma2Model {
    /// Download a Gemma-2 checkpoint from the HuggingFace Hub and
    /// build a ready-to-forward `Gemma2Model`. Requires
    /// `HF_TOKEN` for gated repos like `google/gemma-2-2b-it`.
    pub fn from_hub(repo_id: &str) -> crate::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| crate::Error::Msg(format!("hf-hub config.json: {e}")))?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let config = Gemma2Config::from_hf_json_str(&config_str)?;

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
        let weights = Gemma2Weights::load_from_mmapped(&st, &config)?;

        Ok(Gemma2Model { config, weights })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }

    #[test]
    fn from_hf_json_str_parses_canonical_gemma2_2b_fields() {
        // Excerpt from google/gemma-2-2b config.json plus a fake
        // sliding_window field to verify all the optional-default
        // branches are exercised.
        let json = r#"{
            "vocab_size": 256000,
            "hidden_size": 2304,
            "intermediate_size": 9216,
            "num_hidden_layers": 26,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "attention_bias": false,
            "max_position_embeddings": 8192,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "sliding_window": 4096
        }"#;
        let cfg = Gemma2Config::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.vocab_size, 256000);
        assert_eq!(cfg.hidden_size, 2304);
        assert_eq!(cfg.intermediate_size, 9216);
        assert_eq!(cfg.num_hidden_layers, 26);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!((cfg.rope_theta - 10000.0).abs() < 1e-9);
        assert!(!cfg.attention_bias);
        assert_eq!(cfg.max_position_embeddings, 8192);
        assert_eq!(cfg.attn_logit_softcapping, Some(50.0));
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
        assert_eq!(cfg.sliding_window, Some(4096));
    }

    #[test]
    fn from_hf_json_str_applies_optional_defaults() {
        // Minimal config — every optional defaults applied.
        let json = r#"{
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "max_position_embeddings": 512
        }"#;
        let cfg = Gemma2Config::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.num_key_value_heads, 4);  // defaults to num_attention_heads
        assert_eq!(cfg.head_dim, 16);            // defaults to hidden / heads
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!((cfg.rope_theta - 10000.0).abs() < 1e-9);
        assert!(!cfg.attention_bias);
        assert_eq!(cfg.attn_logit_softcapping, None);
        assert_eq!(cfg.final_logit_softcapping, None);
        assert_eq!(cfg.sliding_window, None);
    }

    fn tiny_cfg() -> Gemma2Config {
        Gemma2Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4,
            num_key_value_heads: 2, head_dim: 4,
            rms_norm_eps: 1e-6, rope_theta: 10_000.0,
            attention_bias: false, max_position_embeddings: 32,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            sliding_window: None,
        }
    }

    fn tiny_weights(cfg: &Gemma2Config, seed: u32) -> Gemma2Weights {
        let mut nb = rng_seed(seed);
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h, &mut nb);
        let layers: Vec<Gemma2LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| Gemma2LayerWeights {
                input_norm_gain: Arc::from(vec![0.0_f32; h]),
                post_attn_norm_gain: Arc::from(vec![0.0_f32; h]),
                pre_ffn_norm_gain: Arc::from(vec![0.0_f32; h]),
                post_ffn_norm_gain: Arc::from(vec![0.0_f32; h]),
                q: WeightStorage::F32(vec_of(h * q_dim, &mut nb)),
                q_bias: None,
                k: WeightStorage::F32(vec_of(h * kv_dim, &mut nb)),
                k_bias: None,
                v: WeightStorage::F32(vec_of(h * kv_dim, &mut nb)),
                v_bias: None,
                o: WeightStorage::F32(vec_of(q_dim * h, &mut nb)),
                o_bias: None,
                gate: WeightStorage::F32(vec_of(h * i, &mut nb)),
                up: WeightStorage::F32(vec_of(h * i, &mut nb)),
                down: WeightStorage::F32(vec_of(i * h, &mut nb)),
            })
            .collect();
        Gemma2Weights {
            token_embedding,
            layers,
            final_norm_gain: Arc::from(vec![0.0_f32; h]),
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_cfg();
        let model = Gemma2Model { config: cfg.clone(), weights: tiny_weights(&cfg, 11) };
        let tokens = [1_u32, 2, 3, 4];
        let out = model.forward(&tokens, 0).unwrap();
        assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    /// Final logit softcapping bounds output: |logit| <= cap.
    #[test]
    fn final_softcap_bounds_logits() {
        let mut cfg = tiny_cfg();
        cfg.final_logit_softcapping = Some(2.0);
        let model = Gemma2Model { config: cfg.clone(), weights: tiny_weights(&cfg, 22) };
        let tokens = [1_u32, 2, 3, 4];
        let out = model.forward(&tokens, 0).unwrap().realize_f32();
        for &v in &out {
            assert!(v.abs() <= 2.0_f32 + 1e-4,
                "logit {v} exceeds softcap 2.0");
        }
    }

    /// Sliding window mask: with window=2 and seq=4, token at
    /// position 0 should be masked OUT of position 3's attention
    /// (3 - 0 = 3 > window 2). Verify by checking the model runs
    /// and produces different output vs. no-window config.
    #[test]
    fn sliding_window_changes_output() {
        let cfg_no_window = {
            let mut c = tiny_cfg();
            c.sliding_window = None;
            c
        };
        let cfg_window = {
            let mut c = tiny_cfg();
            c.sliding_window = Some(2);
            c
        };
        let weights = tiny_weights(&cfg_no_window, 33);
        let m_a = Gemma2Model { config: cfg_no_window, weights: weights.clone() };
        let m_b = Gemma2Model { config: cfg_window, weights };
        let tokens = [1_u32, 2, 3, 4];
        let a = m_a.forward(&tokens, 0).unwrap().realize_f32();
        let b = m_b.forward(&tokens, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "sliding window must alter output, max_diff = {max_diff}");
    }

    /// Gemma-2 RmsNorm scales by `gamma + 1`. With gain=0 (default
    /// init) the effective scale is 1.0, matching standard RmsNorm.
    /// Verify by comparing apply_gemma2_rms_norm(x, [0,...]) and
    /// apply_affine_rms_norm(x, [1,...]).
    #[test]
    fn gemma2_rms_norm_offset() {
        let h = 8;
        let data: Arc<[f32]> = Arc::from(vec![1.0_f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
        let x = LazyTensor::from_f32(data, Shape::from_dims(&[1, 1, h]), &Device::cpu());
        let zero_gain: Arc<[f32]> = Arc::from(vec![0.0_f32; h]);
        let one_gain: Arc<[f32]> = Arc::from(vec![1.0_f32; h]);
        let g2_out = x.rms_norm_affine_with_offset(&zero_gain, 1.0, 1e-6).unwrap().realize_f32();
        let baseline = x.rms_norm_affine(Arc::clone(&one_gain), 1e-6).unwrap().realize_f32();
        for (a, b) in g2_out.iter().zip(baseline.iter()) {
            assert!((a - b).abs() < 1e-6,
                "gemma2 RmsNorm with gain=0 must equal standard RmsNorm with gain=1, {a} vs {b}");
        }
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_cfg();
        let model = Gemma2Model { config: cfg.clone(), weights: tiny_weights(&cfg, 42) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = tiny_cfg();
        let model = Gemma2Model { config: cfg.clone(), weights: tiny_weights(&cfg, 42) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let scaled = embeds.mul_scalar((cfg.hidden_size as f64).sqrt());
        let logits_via_embeds = model.forward_embeds(&scaled, 0).unwrap().realize_f32();
        assert_eq!(logits_ref.len(), logits_via_embeds.len());
        let max_diff = logits_ref.iter().zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "Gemma2 forward vs forward_embeds (post-scale) must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = tiny_cfg();
        let model = Gemma2Model { config: cfg.clone(), weights: tiny_weights(&cfg, 42) };
        let bad_embeds = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]),
            &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad_embeds, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = tiny_cfg();
        let model = Gemma2Model { config: cfg.clone(), weights: tiny_weights(&cfg, 42) };
        let tokens: Vec<u32> = vec![2, 5];
        let h_ref = model.forward_hidden(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let scaled = embeds.mul_scalar((cfg.hidden_size as f64).sqrt());
        let h_via_embeds = model.forward_hidden_embeds(&scaled, 0).unwrap().realize_f32();
        let max_diff = h_ref.iter().zip(h_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "Gemma2 forward_hidden vs forward_hidden_embeds (post-scale) must agree (max diff {max_diff})");
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0
//! Granite decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Granite (IBM) is a pure LLaMA-clone: bias-free
//! Q/K/V/O + RmsNorm + SwiGLU FFN + RoPE. No sliding window, no
//! MoE, no QK-norm, no embedding scaling. Mirrors the existing
//! `Yi`/`Mistral` shells with `head_dim` derived as
//! `hidden_size / num_attention_heads`.
//!
//! Reuses `fuel_core::lazy::LayerWeights`.

use fuel_core::lazy::{LayerWeights, Tensor, WeightStorage};
use fuel_core::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct GraniteConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
}

impl GraniteConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ROADMAP item 8 (II): config-from-path on the #57 template. A `serde` raw with
// HF field names + constant defaults, then `resolve` routes kv heads through the
// shared `fuel_core::hf_config` rule. GraniteConfig carries NO explicit head_dim
// (it derives `hidden_size / num_attention_heads` in `head_dim()`), so only the
// kv-head rule is routed. Granite's scaling multipliers
// (attention/embedding/residual/logits) are not modelled by GraniteConfig and are
// ignored by serde — an existing port limitation, not introduced here.
#[derive(Debug, Clone, serde::Deserialize)]
struct GraniteConfigRaw {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default = "default_granite_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_granite_rope_theta")]
    rope_theta: f64,
    max_position_embeddings: usize,
}

fn default_granite_rms_norm_eps() -> f64 {
    1e-5
}
fn default_granite_rope_theta() -> f64 {
    10_000.0
}

impl GraniteConfigRaw {
    fn from_json_str(json: &str) -> fuel_core::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| fuel_core::Error::Msg(format!("parsing Granite config.json: {e}")))
    }

    fn resolve(self) -> Result<GraniteConfig> {
        Ok(GraniteConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: fuel_core::hf_config::num_key_value_heads(
                self.num_key_value_heads,
                self.num_attention_heads,
            )?,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
        })
    }
}

impl GraniteConfig {
    /// Parse a HuggingFace `config.json` string into a [`GraniteConfig`].
    ///
    /// ROADMAP item 8 (II): reads the artifact rather than returning a preset —
    /// see the born-red `granite_config_from_hf_json_parses_the_artifact`.
    pub fn from_hf_json_str(json: &str) -> fuel_core::Result<Self> {
        GraniteConfigRaw::from_json_str(json)?.resolve()
    }
}

#[derive(Debug, Clone)]
pub struct GraniteWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct GraniteModel {
    pub config: GraniteConfig,
    pub weights: GraniteWeights,
}

impl GraniteModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. Granite does NOT scale embeddings.
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
                "GraniteModel::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(fuel_core::Error::Msg(
                "GraniteModel::forward_embeds: seq must be > 0".into(),
            )
            .bt());
        }
        let head_dim = cfg.head_dim();
        if cfg.num_attention_heads * head_dim != cfg.hidden_size {
            return Err(fuel_core::Error::Msg(
                "GraniteConfig: num_attention_heads * head_dim must equal hidden_size".into(),
            )
            .bt());
        }
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(fuel_core::Error::Msg(
                "GraniteConfig: num_attention_heads must be a multiple of num_key_value_heads"
                    .into(),
            )
            .bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) = h.rope_tables_const(cfg.rope_theta, start_pos, seq, head_dim);

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        h.rms_norm_affine(
            std::sync::Arc::clone(&weights.final_norm_gain),
            cfg.rms_norm_eps,
        )
    }

    fn apply_layer(
        &self,
        x: &Tensor,
        layer: &LayerWeights,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
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
        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size)?;
        let k = layer
            .attn_k
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?;
        let v = layer
            .attn_v
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
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

// ---- HuggingFace safetensors loader ----------------------------------------

impl GraniteWeights {
    /// Load Granite (ibm-granite/granite-*-instruct) weights from HF safetensors.
    /// Standard LLaMA-shape naming, no biases.
    pub fn load_from_mmapped(
        st: &fuel_core::safetensors::MmapedSafetensors,
        cfg: &GraniteConfig,
    ) -> Result<Self> {
        use fuel_core::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype as ltm};
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim();
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim();
        let inter = cfg.intermediate_size;

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);
        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let attn_q = ltm(st, &format!("{p}.self_attn.q_proj.weight"), q_dim, h)?;
            let attn_k = ltm(st, &format!("{p}.self_attn.k_proj.weight"), kv_dim, h)?;
            let attn_v = ltm(st, &format!("{p}.self_attn.v_proj.weight"), kv_dim, h)?;
            let attn_o = ltm(st, &format!("{p}.self_attn.o_proj.weight"), h, q_dim)?;
            let ffn_gate = ltm(st, &format!("{p}.mlp.gate_proj.weight"), inter, h)?;
            let ffn_up = ltm(st, &format!("{p}.mlp.up_proj.weight"), inter, h)?;
            let ffn_down = ltm(st, &format!("{p}.mlp.down_proj.weight"), h, inter)?;
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
                attn_q_bias: None,
                attn_k,
                attn_k_bias: None,
                attn_v,
                attn_v_bias: None,
                attn_o,
                ffn_gate,
                ffn_up,
                ffn_down,
                attn_norm_gain,
                ffn_norm_gain,
            });
        }
        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.norm.weight")?);
        // Granite ties lm_head to token embeddings on the small models.
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

#[cfg(test)]
mod tests {
    use super::*;

    // ROADMAP item 8 (II). Golden values from ibm-granite/granite-3.1-2b-base's
    // real config.json. Granite has no size preset, so a second distinct real
    // config (granite-3.1-8b) provides the constant-parser discrimination. The
    // scaling multipliers are in the JSON to prove serde tolerates them.
    const GRANITE_3_1_2B_CONFIG_JSON: &str = r#"{
        "architectures": ["GraniteForCausalLM"],
        "model_type": "granite",
        "vocab_size": 49152,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-05,
        "rope_theta": 5000000.0,
        "attention_multiplier": 0.015625,
        "embedding_multiplier": 12.0,
        "logits_scaling": 8.0,
        "residual_multiplier": 0.22
    }"#;

    #[test]
    fn granite_config_from_hf_json_parses_the_artifact() {
        let cfg = GraniteConfig::from_hf_json_str(GRANITE_3_1_2B_CONFIG_JSON).unwrap();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.vocab_size, 49_152);
        assert_eq!(cfg.intermediate_size, 8192);
        // GQA: default would be num_attention_heads (32); 8 proves the key was READ.
        assert_eq!(cfg.num_key_value_heads, 8);
        // head_dim is DERIVED (no field): 2048/32 = 64.
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.rope_theta, 5_000_000.0);
        assert_eq!(cfg.max_position_embeddings, 131_072);
        // The four scaling multipliers in the JSON above are unmodelled and were
        // ignored without error — reaching this line proves it.
    }

    /// A SECOND distinct real config (granite-3.1-8b-base): distinct
    /// hidden/intermediate/rope_theta. A constant parser fails one of the two.
    #[test]
    fn granite_config_reads_a_second_distinct_config() {
        let json = r#"{
            "model_type": "granite",
            "vocab_size": 49152,
            "hidden_size": 4096,
            "intermediate_size": 12800,
            "num_hidden_layers": 40,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000000.0
        }"#;
        let cfg = GraniteConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 12_800);
        assert_eq!(cfg.rope_theta, 10_000_000.0);
        assert_ne!(cfg.hidden_size, 2048); // distinct from the 2b parse
    }

    /// `num_key_value_heads` ABSENT → defaults to `num_attention_heads`.
    #[test]
    fn granite_config_gqa_defaults_to_num_heads_when_absent() {
        let json = r#"{
            "model_type": "granite",
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "max_position_embeddings": 128
        }"#;
        let cfg = GraniteConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.num_key_value_heads, 8); // absent → num_attention_heads
    }

    /// TRUE MQA (`num_key_value_heads = 1`) survives, not collapsed.
    #[test]
    fn granite_config_preserves_true_mqa() {
        let json = r#"{
            "model_type": "granite",
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128
        }"#;
        let cfg = GraniteConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.num_key_value_heads, 1);
    }

    fn tiny_weights(cfg: &GraniteConfig) -> GraniteWeights {
        let mut s: u32 = 44444;
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
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| LayerWeights {
                attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                attn_q_bias: None,
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_k_bias: None,
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_v_bias: None,
                attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                ffn_up: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        GraniteWeights {
            token_embedding,
            layers,
            final_norm_gain,
            output,
        }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = GraniteConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
        };
        let model = GraniteModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = GraniteConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
        };
        let model = GraniteModel {
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

    fn forward_embeds_test_cfg() -> GraniteConfig {
        GraniteConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = GraniteModel {
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
            "Granite forward vs forward_embeds must agree (max diff {max_diff})"
        );
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = GraniteModel {
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
        let model = GraniteModel {
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
            "Granite forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})"
        );
    }
}

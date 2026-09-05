// SPDX-License-Identifier: MIT OR Apache-2.0
//! StarCoder2 decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. StarCoder2 is GQA + RoPE + LayerNorm + plain
//! `c_proj(gelu(c_fc(x)))` MLP. Closest cousin in this batch is
//! Falcon's serial-attention mode — same shared LN-with-bias
//! pattern — but StarCoder2 uses RoPE (not Falcon-style halfsplit
//! rotary on the heads-flattened view) and has standard
//! `[input_ln, attn, post_attn_ln, mlp]` sublayer ordering.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache, F32
//! activations. Sliding-window mask when `cfg.sliding_window` is
//! `Some(N)`; strict causal otherwise.

use fuel_core::lazy::{Tensor, WeightStorage};
use fuel_core::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct StarCoder2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub norm_epsilon: f64,
    pub rope_theta: f64,
    pub use_bias: bool,
    pub sliding_window: Option<usize>,
}

impl StarCoder2Config {
    /// `bigcode/starcoder2-3b` ballpark.
    pub fn starcoder2_3b() -> Self {
        Self {
            vocab_size: 49_152,
            hidden_size: 3072,
            intermediate_size: 12_288,
            num_hidden_layers: 30,
            num_attention_heads: 24,
            num_key_value_heads: 2,
            head_dim: 128,
            max_position_embeddings: 16_384,
            norm_epsilon: 1e-5,
            rope_theta: 999_999.0,
            use_bias: true,
            sliding_window: Some(4096),
        }
    }
}

// ROADMAP item 8 (II): config-from-path, as a capability of the config TYPE.
// A `serde` raw carrying HF's field names + constant defaults, then `resolve`
// routes the two sibling-derived values (kv heads, head_dim) through the shared
// `fuel_core::hf_config` rules. StarCoder2 ships an explicit `head_dim` only for
// padded-head variants; the take-if-present rule honors it and derives the
// quotient otherwise.
#[derive(Debug, Clone, serde::Deserialize)]
struct StarCoder2ConfigRaw {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    max_position_embeddings: usize,
    #[serde(default = "default_starcoder2_norm_epsilon")]
    norm_epsilon: f64,
    #[serde(default = "default_starcoder2_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_starcoder2_use_bias")]
    use_bias: bool,
    #[serde(default)]
    sliding_window: Option<usize>,
}

fn default_starcoder2_norm_epsilon() -> f64 {
    1e-5
}
fn default_starcoder2_rope_theta() -> f64 {
    10_000.0
}
fn default_starcoder2_use_bias() -> bool {
    true
}

impl StarCoder2ConfigRaw {
    fn from_json_str(json: &str) -> fuel_core::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| fuel_core::Error::Msg(format!("parsing StarCoder2 config.json: {e}")))
    }

    fn resolve(self) -> StarCoder2Config {
        StarCoder2Config {
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
            max_position_embeddings: self.max_position_embeddings,
            norm_epsilon: self.norm_epsilon,
            rope_theta: self.rope_theta,
            use_bias: self.use_bias,
            sliding_window: self.sliding_window,
        }
    }
}

impl StarCoder2Config {
    /// Parse a HuggingFace `config.json` string into a [`StarCoder2Config`].
    ///
    /// ROADMAP item 8 (II): reads the artifact rather than returning a preset —
    /// see the born-red `starcoder2_config_from_hf_json_parses_the_artifact_not_a_preset`.
    pub fn from_hf_json_str(json: &str) -> fuel_core::Result<Self> {
        Ok(StarCoder2ConfigRaw::from_json_str(json)?.resolve())
    }
}

#[derive(Debug, Clone)]
pub struct StarCoder2LayerWeights {
    pub input_ln_gain: Arc<[f32]>,
    pub input_ln_bias: Arc<[f32]>,
    pub post_attn_ln_gain: Arc<[f32]>,
    pub post_attn_ln_bias: Arc<[f32]>,
    pub attn_q: WeightStorage,
    pub attn_q_bias: Option<Arc<[f32]>>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Option<Arc<[f32]>>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Option<Arc<[f32]>>,
    pub attn_o: WeightStorage,
    pub attn_o_bias: Option<Arc<[f32]>>,
    pub mlp_fc: WeightStorage,
    pub mlp_fc_bias: Option<Arc<[f32]>>,
    pub mlp_proj: WeightStorage,
    pub mlp_proj_bias: Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct StarCoder2Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<StarCoder2LayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct StarCoder2Model {
    pub config: StarCoder2Config,
    pub weights: StarCoder2Weights,
}

impl StarCoder2Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final LayerNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. Starcoder2 does NOT scale embeddings.
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
                "StarCoder2Model::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(fuel_core::Error::Msg(
                "StarCoder2Model::forward_embeds: seq must be > 0".into(),
            )
            .bt());
        }
        if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size {
            return Err(fuel_core::Error::Msg(
                "StarCoder2Config: num_attention_heads * head_dim must equal hidden_size".into(),
            )
            .bt());
        }
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(fuel_core::Error::Msg(
                "StarCoder2Config: num_attention_heads must be a multiple of num_key_value_heads"
                    .into(),
            )
            .bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) =
            h.rope_tables_const(cfg.rope_theta, start_pos, seq, cfg.head_dim);

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        h.layer_norm_affine(
            std::sync::Arc::clone(&weights.final_ln_gain),
            std::sync::Arc::clone(&weights.final_ln_bias),
            cfg.norm_epsilon,
        )
    }

    fn apply_layer(
        &self,
        x: &Tensor,
        layer: &StarCoder2LayerWeights,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let x_norm = x.layer_norm_affine(
            std::sync::Arc::clone(&layer.input_ln_gain),
            std::sync::Arc::clone(&layer.input_ln_bias),
            cfg.norm_epsilon,
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
        let mask = self.build_mask(x, seq);
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer
            .attn_o
            .apply_linear(&merged, cfg.hidden_size, cfg.hidden_size)?
            .add_optional_trailing_bias(layer.attn_o_bias.as_ref())?;

        let h1 = x.add(&attn_out)?;
        let h1_norm = h1.layer_norm_affine(
            std::sync::Arc::clone(&layer.post_attn_ln_gain),
            std::sync::Arc::clone(&layer.post_attn_ln_bias),
            cfg.norm_epsilon,
        )?;

        // MLP: c_proj(gelu(c_fc(x))). Standard GELU, not GeluPyTorchTanh.
        let mid = layer
            .mlp_fc
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?
            .add_optional_trailing_bias(layer.mlp_fc_bias.as_ref())?;
        let mid_act = mid.gelu_erf();
        let ffn_out = layer
            .mlp_proj
            .apply_linear(&mid_act, cfg.intermediate_size, cfg.hidden_size)?
            .add_optional_trailing_bias(layer.mlp_proj_bias.as_ref())?;

        h1.add(&ffn_out)
    }

    fn build_mask(&self, anchor: &Tensor, seq: usize) -> Tensor {
        let cfg = &self.config;
        let window = cfg.sliding_window.unwrap_or(seq + 1);
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
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl StarCoder2Weights {
    /// Load StarCoder2 weights from HF safetensors (e.g. `bigcode/starcoder2-3b`).
    /// StarCoder2 has biases throughout when `use_bias=true`.
    pub fn load_from_mmapped(
        st: &fuel_core::safetensors::MmapedSafetensors,
        cfg: &StarCoder2Config,
    ) -> Result<Self> {
        use fuel_core::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let opt_bias =
            |st: &fuel_core::safetensors::MmapedSafetensors, n: &str| -> Option<Arc<[f32]>> {
                if cfg.use_bias {
                    load_tensor_as_f32(st, n).ok().map(Arc::from)
                } else {
                    None
                }
            };

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);
        let mut layers: Vec<StarCoder2LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let input_ln_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.input_layernorm.weight"),
            )?);
            let input_ln_bias = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.input_layernorm.bias"),
            )?);
            let post_attn_ln_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.post_attention_layernorm.weight"),
            )?);
            let post_attn_ln_bias = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.post_attention_layernorm.bias"),
            )?);
            let attn_q = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.q_proj.weight"),
                q_dim,
                h,
            )?;
            let attn_q_bias = opt_bias(st, &format!("{p}.self_attn.q_proj.bias"));
            let attn_k = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                h,
            )?;
            let attn_k_bias = opt_bias(st, &format!("{p}.self_attn.k_proj.bias"));
            let attn_v = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.v_proj.weight"),
                kv_dim,
                h,
            )?;
            let attn_v_bias = opt_bias(st, &format!("{p}.self_attn.v_proj.bias"));
            let attn_o = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.o_proj.weight"),
                h,
                q_dim,
            )?;
            let attn_o_bias = opt_bias(st, &format!("{p}.self_attn.o_proj.bias"));
            let mlp_fc = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.mlp.c_fc.weight"),
                inter,
                h,
            )?;
            let mlp_fc_bias = opt_bias(st, &format!("{p}.mlp.c_fc.bias"));
            let mlp_proj = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.mlp.c_proj.weight"),
                h,
                inter,
            )?;
            let mlp_proj_bias = opt_bias(st, &format!("{p}.mlp.c_proj.bias"));
            layers.push(StarCoder2LayerWeights {
                input_ln_gain,
                input_ln_bias,
                post_attn_ln_gain,
                post_attn_ln_bias,
                attn_q,
                attn_q_bias,
                attn_k,
                attn_k_bias,
                attn_v,
                attn_v_bias,
                attn_o,
                attn_o_bias,
                mlp_fc,
                mlp_fc_bias,
                mlp_proj,
                mlp_proj_bias,
            });
        }
        let final_ln_gain = Arc::from(load_tensor_as_f32(st, "model.norm.weight")?);
        let final_ln_bias = Arc::from(load_tensor_as_f32(st, "model.norm.bias")?);
        let output =
            load_transposed_matrix_preserve_dtype(st, "lm_head.weight", cfg.vocab_size, h)?;
        Ok(Self {
            token_embedding,
            layers,
            final_ln_gain,
            final_ln_bias,
            output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ROADMAP item 8 (II). Golden values from bigcode/starcoder2-3b's real
    // config.json (huggingface.co/bigcode/starcoder2-3b/blob/main/config.json).
    const STARCODER2_3B_CONFIG_JSON: &str = r#"{
        "architectures": ["Starcoder2ForCausalLM"],
        "model_type": "starcoder2",
        "vocab_size": 49152,
        "hidden_size": 3072,
        "intermediate_size": 12288,
        "num_hidden_layers": 30,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "max_position_embeddings": 16384,
        "norm_epsilon": 1e-05,
        "rope_theta": 999999.4420358813,
        "sliding_window": 4096,
        "use_bias": true
    }"#;

    #[test]
    fn starcoder2_config_from_hf_json_parses_the_artifact_not_a_preset() {
        let cfg = StarCoder2Config::from_hf_json_str(STARCODER2_3B_CONFIG_JSON).unwrap();
        // POSITIVE goldens — starcoder2-3b, required fields (no default):
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 30);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.vocab_size, 49_152);
        assert_eq!(cfg.intermediate_size, 12_288);
        // GQA: default is num_attention_heads (24); 2 proves the key was READ.
        assert_eq!(cfg.num_key_value_heads, 2);
        // head_dim absent from the artifact → derived 3072/24 = 128.
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.max_position_embeddings, 16_384);
        assert_eq!(cfg.sliding_window, Some(4096));
        assert!(cfg.use_bias);
        // The REAL rope_theta (~999999.442), distinct from the preset's rounded
        // 999_999.0 — the single field that separates this parse from
        // `starcoder2_3b()`, so it is a PRIMARY discriminator. Asserted within a
        // tolerance because serde_json and rustc round the artifact's 17th decimal
        // to adjacent f64 values (1 ULP apart); the point is that it is the
        // artifact's value (0.442 away from the preset), not an exact bit pattern.
        assert!((cfg.rope_theta - 999_999.442).abs() < 0.01);
        assert_ne!(cfg.rope_theta, 999_999.0);
        // Sabotage sibling (WEAKER): not the 3b preset. The `==` goldens are primary.
        assert_ne!(cfg, StarCoder2Config::starcoder2_3b());
    }

    /// A SECOND distinct config parses to ITS OWN values, exercising the default
    /// path (norm_epsilon/rope_theta/use_bias/sliding_window omitted) AND the
    /// take-if-present head_dim branch (explicit 96 ≠ 4096/32 = 128). A constant
    /// or preset parser fails one of the two configs.
    #[test]
    fn starcoder2_config_reads_a_second_distinct_config_with_explicit_head_dim() {
        let json = r#"{
            "model_type": "starcoder2",
            "vocab_size": 49152,
            "hidden_size": 4096,
            "intermediate_size": 16384,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 96,
            "max_position_embeddings": 16384
        }"#;
        let cfg = StarCoder2Config::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        // explicit head_dim WINS over the quotient (4096/32 = 128).
        assert_eq!(cfg.head_dim, 96);
        assert_ne!(cfg.head_dim, 4096 / 32);
        // omitted → resolve defaults
        assert_eq!(cfg.rope_theta, 10_000.0);
        assert_eq!(cfg.norm_epsilon, 1e-5);
        assert!(cfg.use_bias); // default true
        assert_eq!(cfg.sliding_window, None);
        // distinct from the 3b parse — a constant parser cannot satisfy both
        assert_ne!(cfg.hidden_size, 3072);
    }

    /// With `num_key_value_heads` ABSENT, GQA defaults to `num_attention_heads`.
    /// Paired with the 3b golden (present → 2), this distinguishes "read the key"
    /// from "never looked".
    #[test]
    fn starcoder2_config_gqa_defaults_to_num_heads_when_absent() {
        let json = r#"{
            "model_type": "starcoder2",
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "max_position_embeddings": 128
        }"#;
        let cfg = StarCoder2Config::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.num_key_value_heads, 8); // absent → num_attention_heads
    }

    /// Load-bearing behavioural row (hf_config take-if-present-else-derive): a
    /// config STATING `num_key_value_heads = 1` is TRUE MQA and must survive as 1,
    /// never collapsed to `num_attention_heads`. Passes only because resolve routes
    /// through `hf_config::num_key_value_heads`; rewrite it to fork and this reds.
    #[test]
    fn starcoder2_config_preserves_true_mqa() {
        let json = r#"{
            "model_type": "starcoder2",
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128
        }"#;
        let cfg = StarCoder2Config::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.num_key_value_heads, 1); // TRUE MQA survives, not collapsed to 8
    }

    fn tiny_weights(cfg: &StarCoder2Config) -> StarCoder2Weights {
        let mut s: u32 = 27182;
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
        let layers: Vec<StarCoder2LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| StarCoder2LayerWeights {
                input_ln_gain: Arc::from(vec![1.0_f32; h]),
                input_ln_bias: Arc::from(vec![0.0_f32; h]),
                post_attn_ln_gain: Arc::from(vec![1.0_f32; h]),
                post_attn_ln_bias: Arc::from(vec![0.0_f32; h]),
                attn_q: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
                attn_q_bias: if cfg.use_bias {
                    Some(vec_of(h, &mut *next_box))
                } else {
                    None
                },
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
                attn_k_bias: if cfg.use_bias {
                    Some(vec_of(kv, &mut *next_box))
                } else {
                    None
                },
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
                attn_v_bias: if cfg.use_bias {
                    Some(vec_of(kv, &mut *next_box))
                } else {
                    None
                },
                attn_o: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
                attn_o_bias: if cfg.use_bias {
                    Some(vec_of(h, &mut *next_box))
                } else {
                    None
                },
                mlp_fc: WeightStorage::F32(vec_of(h * i, &mut *next_box)),
                mlp_fc_bias: if cfg.use_bias {
                    Some(vec_of(i, &mut *next_box))
                } else {
                    None
                },
                mlp_proj: WeightStorage::F32(vec_of(i * h, &mut *next_box)),
                mlp_proj_bias: if cfg.use_bias {
                    Some(vec_of(h, &mut *next_box))
                } else {
                    None
                },
            })
            .collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *next_box));
        StarCoder2Weights {
            token_embedding,
            layers,
            final_ln_gain,
            final_ln_bias,
            output,
        }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = StarCoder2Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 64,
            norm_epsilon: 1e-5,
            rope_theta: 10_000.0,
            use_bias: true,
            sliding_window: Some(4),
        };
        let model = StarCoder2Model {
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

    /// `forward_hidden` returns post-LayerNorm hidden states.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = StarCoder2Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 64,
            norm_epsilon: 1e-5,
            rope_theta: 10_000.0,
            use_bias: true,
            sliding_window: None,
        };
        let model = StarCoder2Model {
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

    fn forward_embeds_test_cfg() -> StarCoder2Config {
        StarCoder2Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 64,
            norm_epsilon: 1e-5,
            rope_theta: 10_000.0,
            use_bias: true,
            sliding_window: None,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = StarCoder2Model {
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
            "StarCoder2 forward vs forward_embeds must agree (max diff {max_diff})"
        );
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = StarCoder2Model {
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
        let model = StarCoder2Model {
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
            "StarCoder2 forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})"
        );
    }
}

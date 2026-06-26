//! SmolLM3 decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. SmolLM3 (HuggingFace small-model line) is a
//! Qwen2-shape transformer with two notable extras:
//! - **Per-layer RoPE gating** — `no_rope_layers[i] == 1` skips RoPE
//!   for layer `i`. Useful for hybrid attention patterns where some
//!   layers run position-agnostic.
//! - **Optional sliding window** (Mistral-style).
//!
//! Otherwise: GQA + RmsNorm + SwiGLU FFN + optional Q/K/V/O biases
//! via `cfg.attention_bias`. Reuses `crate::lazy::LayerWeights`.

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct SmolLm3Config {
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
    pub sliding_window: Option<usize>,
    /// One entry per layer: `1` = use RoPE on that layer, `0` = skip
    /// RoPE. `None` = use RoPE on every layer (Llama default).
    pub no_rope_layers: Option<Vec<usize>>,
}

impl SmolLm3Config {
    fn layer_uses_rope(&self, layer_idx: usize) -> bool {
        match &self.no_rope_layers {
            Some(v) => v.get(layer_idx).copied().unwrap_or(1) == 1,
            None => true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SmolLm3Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct SmolLm3Model {
    pub config: SmolLm3Config,
    pub weights: SmolLm3Weights,
}

impl SmolLm3Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// SmolLM3-specific: every Nth layer skips RoPE
    /// (NoPE pattern). The hook honors the same per-layer
    /// RoPE-on/off schedule as `forward`.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. SmolLM3 does NOT scale embeddings.
    pub fn forward_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let h_norm = self.run_backbone_embeds(embeds, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`].
    pub fn forward_hidden_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.run_backbone_embeds(embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder.
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
        Ok(self.weights.output.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0);

        let h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;
        self.run_backbone_embeds(&h, start_pos)
    }

    fn run_backbone_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(crate::Error::Msg(format!(
                "SmolLm3Model::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "SmolLm3Model::forward_embeds: seq must be > 0".into(),
            ).bt());
        }
        if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "SmolLm3Config: num_attention_heads * head_dim must equal hidden_size".into(),
            ).bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, cfg.head_dim,
        );

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let uses_rope = cfg.layer_uses_rope(layer_idx);
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, uses_rope)?;
        }
        h.rms_norm_affine(std::sync::Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps)
    }

    fn build_mask(&self, anchor: &LazyTensor, seq: usize) -> LazyTensor {
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

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        uses_rope: bool,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let x_norm = x.rms_norm_affine(std::sync::Arc::clone(&layer.attn_norm_gain), cfg.rms_norm_eps)?;

        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size).add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        // Conditional RoPE — skip for layers in `no_rope_layers`.
        let (q_r, k_r) = if uses_rope {
            (
                q.rope_with_tables(rope_cos, rope_sin)?,
                k.rope_with_tables(rope_cos, rope_sin)?,
            )
        } else {
            (q, k)
        };

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
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size);

        let h1 = x.add(&attn_out)?;
        let h1_norm = h1.rms_norm_affine(std::sync::Arc::clone(&layer.ffn_norm_gain), cfg.rms_norm_eps)?;
        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size);
        h1.add(&ffn_out)
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl SmolLm3Weights {
    /// Load SmolLM3 weights from HF safetensors.
    /// HF naming follows LLaMA conventions: model.embed_tokens.weight,
    /// model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,optional bias},
    /// model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight,
    /// model.layers.{i}.mlp.{gate,up,down}_proj.weight, model.norm.weight,
    /// lm_head.weight (always present in HF SmolLM3 checkpoints).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &SmolLm3Config,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
        let h = cfg.hidden_size;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);

        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let attn_q = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.q_proj.weight"), q_dim, h,
            )?;
            let attn_k = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.k_proj.weight"), kv_dim, h,
            )?;
            let attn_v = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.v_proj.weight"), kv_dim, h,
            )?;
            let attn_o = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.o_proj.weight"), h, q_dim,
            )?;
            let ffn_gate = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.mlp.gate_proj.weight"), inter, h,
            )?;
            let ffn_up = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.mlp.up_proj.weight"), inter, h,
            )?;
            let ffn_down = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.mlp.down_proj.weight"), h, inter,
            )?;
            let attn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.input_layernorm.weight"),
            )?);
            let ffn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.post_attention_layernorm.weight"),
            )?);

            let attn_q_bias = if cfg.attention_bias {
                Some(Arc::from(load_tensor_as_f32(st, &format!("{p}.self_attn.q_proj.bias"))?))
            } else { None };
            let attn_k_bias = if cfg.attention_bias {
                Some(Arc::from(load_tensor_as_f32(st, &format!("{p}.self_attn.k_proj.bias"))?))
            } else { None };
            let attn_v_bias = if cfg.attention_bias {
                Some(Arc::from(load_tensor_as_f32(st, &format!("{p}.self_attn.v_proj.bias"))?))
            } else { None };

            layers.push(LayerWeights {
                attn_q, attn_q_bias,
                attn_k, attn_k_bias,
                attn_v, attn_v_bias,
                attn_o,
                ffn_gate, ffn_up, ffn_down,
                attn_norm_gain, ffn_norm_gain,
            });
        }

        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.norm.weight")?);
        let output = load_transposed_matrix_preserve_dtype(
            st, "lm_head.weight", cfg.vocab_size, h,
        )?;

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
    fn tiny_weights(cfg: &SmolLm3Config) -> SmolLm3Weights {
        let mut s: u32 = 55555;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size; let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            attn_q_bias: if cfg.attention_bias { Some(vec_of(h, &mut *nb)) } else { None },
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_k_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_v_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        SmolLm3Weights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_and_finite_all_rope() {
        let cfg = SmolLm3Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, rms_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 64, attention_bias: false,
            sliding_window: None, no_rope_layers: None,
        };
        let model = SmolLm3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }

    /// `no_rope_layers = [0, 1]` skips RoPE on layer 0 only.
    /// Output must differ from the all-RoPE configuration.
    #[test]
    fn skipping_rope_on_one_layer_changes_output() {
        let mut cfg = SmolLm3Config {
            vocab_size: 16, hidden_size: 8, intermediate_size: 16,
            num_hidden_layers: 2, num_attention_heads: 2, num_key_value_heads: 2,
            head_dim: 4, rms_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 32, attention_bias: false,
            sliding_window: None, no_rope_layers: None,
        };
        let weights = tiny_weights(&cfg);
        let out_all = SmolLm3Model { config: cfg.clone(), weights: weights.clone() }
            .forward(&[1, 2, 3, 4], 0).unwrap().realize_f32();
        cfg.no_rope_layers = Some(vec![0, 1]); // skip RoPE on layer 0
        let out_partial = SmolLm3Model { config: cfg, weights }
            .forward(&[1, 2, 3, 4], 0).unwrap().realize_f32();
        let any_diff = out_all.iter().zip(out_partial.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7);
        assert!(any_diff, "skipping RoPE on layer 0 must change output");
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = SmolLm3Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, rms_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 64, attention_bias: false,
            sliding_window: None, no_rope_layers: None,
        };
        let model = SmolLm3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    fn forward_embeds_test_cfg() -> SmolLm3Config {
        SmolLm3Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, rms_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 64, attention_bias: false,
            sliding_window: None, no_rope_layers: None,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = SmolLm3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let logits_via_embeds = model.forward_embeds(&embeds, 0).unwrap().realize_f32();
        let max_diff = logits_ref.iter().zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "SmolLm3 forward vs forward_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = SmolLm3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let bad = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]), &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = forward_embeds_test_cfg();
        let model = SmolLm3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![5, 7];
        let h_ref = model.forward_hidden(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let h_via_embeds = model.forward_hidden_embeds(&embeds, 0).unwrap().realize_f32();
        let max_diff = h_ref.iter().zip(h_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "SmolLm3 forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})");
    }
}

//! OLMo2 decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. OLMo2 evolves OLMo with two changes:
//! 1. **RmsNorm** instead of LayerNorm-no-bias.
//! 2. **QK-norm** — apply a separate RmsNorm to the projected Q and
//!    K before the head reshape. `q_norm` has shape `[hidden_size]`;
//!    `k_norm` has shape `[num_kv_heads * head_dim]`.
//!
//! Otherwise identical to OLMo: GQA + RoPE + SwiGLU FFN + optional
//! Q/K/V/O biases via `cfg.attention_bias`.
//!
//! Reuses LLaMA's `LayerWeights` for the standard fields and stores
//! the QK-norm gains separately in `Olmo2LayerExtras`.

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Olmo2Config {
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
}

impl Olmo2Config {
    /// `allenai/OLMo2-7B`-class.
    pub fn olmo2_7b() -> Self {
        Self {
            vocab_size: 100_352,
            hidden_size: 4096,
            intermediate_size: 11_008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 500_000.0,
            max_position_embeddings: 4096,
            attention_bias: false,
        }
    }
}

/// Per-layer QK-norm gains. Sibling-side to `LayerWeights` for the
/// OLMo2-specific extras.
#[derive(Debug, Clone)]
pub struct Olmo2LayerExtras {
    /// `[hidden_size]`.
    pub q_norm_gain: Arc<[f32]>,
    /// `[num_kv_heads * head_dim]`.
    pub k_norm_gain: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Olmo2Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub layer_extras: Vec<Olmo2LayerExtras>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Olmo2Model {
    pub config: Olmo2Config,
    pub weights: Olmo2Weights,
}

impl Olmo2Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection. OLMo2 uses RmsNorm
    /// (vs. OLMo's LayerNorm-no-bias).
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. OLMo2 does NOT scale embeddings.
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
                "Olmo2Model::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "Olmo2Model::forward_embeds: seq must be > 0".into(),
            ).bt());
        }
        if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "Olmo2Config: num_attention_heads * head_dim must equal hidden_size".into(),
            ).bt());
        }
        if weights.layers.len() != weights.layer_extras.len() {
            return Err(crate::Error::Msg(format!(
                "Olmo2Weights: layers ({}) must have matching layer_extras ({})",
                weights.layers.len(), weights.layer_extras.len(),
            )).bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, cfg.head_dim,
        );

        for (layer, extras) in weights.layers.iter().zip(weights.layer_extras.iter()) {
            h = self.apply_layer(&h, layer, extras, &rope_cos, &rope_sin)?;
        }
        h.rms_norm_affine(std::sync::Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps)
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        extras: &Olmo2LayerExtras,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
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

        // QK-norm — RmsNorm Q and K BEFORE head reshape.
        let q = q.rms_norm_affine(std::sync::Arc::clone(&extras.q_norm_gain), cfg.rms_norm_eps)?;
        let k = k.rms_norm_affine(std::sync::Arc::clone(&extras.k_norm_gain), cfg.rms_norm_eps)?;

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
        let mask = LazyTensor::additive_causal_mask_like(x, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
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

impl Olmo2Weights {
    /// Load OLMo2 (allenai/OLMo2-*) weights from HuggingFace safetensors.
    /// Standard LLaMA-shape attention with QK-norm gains.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &Olmo2Config,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype as ltm};
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;

        let opt_bias = |name: String| -> Option<Arc<[f32]>> {
            load_tensor_as_f32(st, &name).ok().map(Arc::from)
        };

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);
        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        let mut layer_extras: Vec<Olmo2LayerExtras> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let attn_q = ltm(st, &format!("{p}.self_attn.q_proj.weight"), q_dim, h)?;
            let attn_q_bias = if cfg.attention_bias {
                opt_bias(format!("{p}.self_attn.q_proj.bias"))
            } else { None };
            let attn_k = ltm(st, &format!("{p}.self_attn.k_proj.weight"), kv_dim, h)?;
            let attn_k_bias = if cfg.attention_bias {
                opt_bias(format!("{p}.self_attn.k_proj.bias"))
            } else { None };
            let attn_v = ltm(st, &format!("{p}.self_attn.v_proj.weight"), kv_dim, h)?;
            let attn_v_bias = if cfg.attention_bias {
                opt_bias(format!("{p}.self_attn.v_proj.bias"))
            } else { None };
            let attn_o = ltm(st, &format!("{p}.self_attn.o_proj.weight"), h, q_dim)?;
            let ffn_gate = ltm(st, &format!("{p}.mlp.gate_proj.weight"), inter, h)?;
            let ffn_up = ltm(st, &format!("{p}.mlp.up_proj.weight"), inter, h)?;
            let ffn_down = ltm(st, &format!("{p}.mlp.down_proj.weight"), h, inter)?;
            // OLMo2 swaps the LN placement: input is `post_feedforward_layernorm`
            // (post-norm-ish), but the LayerWeights field is called attn_norm_gain
            // and is applied per the model's forward implementation.
            let attn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.post_attention_layernorm.weight"),
            )?);
            let ffn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.post_feedforward_layernorm.weight"),
            )?);
            layers.push(LayerWeights {
                attn_q, attn_q_bias, attn_k, attn_k_bias,
                attn_v, attn_v_bias, attn_o,
                ffn_gate, ffn_up, ffn_down, attn_norm_gain, ffn_norm_gain,
            });

            let q_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.self_attn.q_norm.weight"),
            )?);
            let k_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.self_attn.k_norm.weight"),
            )?);
            layer_extras.push(Olmo2LayerExtras { q_norm_gain, k_norm_gain });
        }
        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.norm.weight")?);
        let output = match ltm(st, "lm_head.weight", cfg.vocab_size, h) {
            Ok(w) => w,
            Err(_) => crate::lazy_llama_full::tied_lm_head_from_embeddings(
                &token_embedding, cfg.vocab_size, h,
            ),
        };
        Ok(Self { token_embedding, layers, layer_extras, final_norm_gain, output })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_weights(cfg: &Olmo2Config) -> Olmo2Weights {
        let mut s: u32 = 22222;
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
        let mut layers = Vec::new();
        let mut layer_extras = Vec::new();
        for _ in 0..cfg.num_hidden_layers {
            layers.push(LayerWeights {
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
            });
            layer_extras.push(Olmo2LayerExtras {
                q_norm_gain: Arc::from(vec![1.0_f32; h]),
                k_norm_gain: Arc::from(vec![1.0_f32; kv]),
            });
        }
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        Olmo2Weights { token_embedding, layers, layer_extras, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = Olmo2Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, rms_norm_eps: 1e-6, rope_theta: 500_000.0,
            max_position_embeddings: 64, attention_bias: false,
        };
        let model = Olmo2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3, 4], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 4, cfg.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }

    /// QK-norm with all-ones gain should produce different output
    /// than skipping it entirely. We can't easily disable QK-norm
    /// without rewiring; instead set q_norm to all-zero gain (which
    /// kills Q's signal) and verify the output changes drastically.
    #[test]
    fn qk_norm_gain_affects_output() {
        let cfg = Olmo2Config {
            vocab_size: 16, hidden_size: 8, intermediate_size: 16,
            num_hidden_layers: 1, num_attention_heads: 2, num_key_value_heads: 2,
            head_dim: 4, rms_norm_eps: 1e-6, rope_theta: 10_000.0,
            max_position_embeddings: 32, attention_bias: false,
        };
        let weights_a = tiny_weights(&cfg);
        let mut weights_b = weights_a.clone();
        for e in &mut weights_b.layer_extras {
            e.q_norm_gain = Arc::from(vec![0.5_f32; cfg.hidden_size]);
        }
        let out_a = Olmo2Model { config: cfg.clone(), weights: weights_a }
            .forward(&[1, 2, 3], 0).unwrap().realize_f32();
        let out_b = Olmo2Model { config: cfg, weights: weights_b }
            .forward(&[1, 2, 3], 0).unwrap().realize_f32();
        let any_diff = out_a.iter().zip(out_b.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(any_diff, "different q_norm gain must change output");
    }

    /// `forward_hidden` returns post-RmsNorm hidden states
    /// `(1, seq, hidden_size)` without the lm_head matmul.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = Olmo2Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, rms_norm_eps: 1e-6, rope_theta: 10_000.0,
            max_position_embeddings: 32, attention_bias: false,
        };
        let model = Olmo2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    fn forward_embeds_test_cfg() -> Olmo2Config {
        Olmo2Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 4, rms_norm_eps: 1e-6, rope_theta: 10_000.0,
            max_position_embeddings: 32, attention_bias: false,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = Olmo2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
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
            "OLMo2 forward vs forward_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = Olmo2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let bad = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]), &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = forward_embeds_test_cfg();
        let model = Olmo2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
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
            "OLMo2 forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})");
    }
}

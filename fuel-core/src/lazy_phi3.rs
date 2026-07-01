//! Phi-3 decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Phi-3 (Phi-3-mini-4k-instruct etc.) is a
//! standard GQA transformer with HuggingFace's "fused projection"
//! quirk: a single `qkv_proj` packs Q + K + V along the last dim,
//! and a single `gate_up_proj` packs gate + up. On disk the
//! safetensors stores them fused; in lazy we store them split
//! (matching [`crate::lazy::LayerWeights`]), with the safetensors
//! loader doing the narrow at load time.
//!
//! **Deferred to a follow-up** (don't block other ports on it):
//!   - LongRoPE long-context scaling (short_factor / long_factor /
//!     `original_max_position_embeddings`). Phi-3-mini-4k doesn't
//!     use it; Phi-3-mini-128k does.
//!   - `partial_rotary_factor` < 1.0 (apply RoPE to only a prefix of
//!     each head's dim). Default is 1.0 (full rotary) — that's what
//!     this port assumes.
//!
//! Both can be added by augmenting `Phi3Config` + the RoPE table
//! builder when a Phi-3-128k checkpoint needs to run.
//!
//! # Scope (v1, same as the other Phase D ports)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache, F32
//! activations. Strict lower-triangular causal mask.

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Phi3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
}

impl Phi3Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `microsoft/Phi-3-mini-4k-instruct`.
    pub fn phi3_mini_4k() -> Self {
        Self {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 4096,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Phi3Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Phi3Model {
    pub config: Phi3Config,
    pub weights: Phi3Weights,
}

impl Phi3Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection. Mirrors the
    /// `forward_hidden` pattern shipped across the LLM family.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips the token-embedding step and runs
    /// the decoder over pre-embedded inputs — the precursor for a
    /// future Phi-3-Vision / Phi-3.5-V lazy composition. Phi3 does
    /// NOT scale embeddings — `embeds` is passed raw.
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

    /// Shared backbone: embed → RoPE → per-layer attn + MLP →
    /// final RmsNorm. Used by both `forward` (then matmuls
    /// with `lm_head`) and `forward_hidden`.
    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0, "Phi3Model: tokens must be non-empty");

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
                "Phi3Model::forward_embeds: expected embeds shape \
                 (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "Phi3Model::forward_embeds: seq must be > 0".into(),
            ).bt());
        }
        let head_dim = cfg.head_dim();
        if cfg.num_attention_heads * head_dim != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "Phi3Config: num_attention_heads * head_dim must equal hidden_size".into(),
            ).bt());
        }
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
            return Err(crate::Error::Msg(
                "Phi3Config: num_attention_heads must be a multiple of num_key_value_heads".into(),
            ).bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, head_dim,
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        h.rms_norm_affine(std::sync::Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps)
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * head_dim;

        let x_norm = x.rms_norm_affine(std::sync::Arc::clone(&layer.attn_norm_gain), cfg.rms_norm_eps)?;

        // Bias-free Q / K / V (Phi-3 uses linear_no_bias for all).
        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size);
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim);
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim);

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        // Strict causal mask.
        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
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
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size);

        let h1 = x.add(&attn_out)?;
        let h1_norm = h1.rms_norm_affine(std::sync::Arc::clone(&layer.ffn_norm_gain), cfg.rms_norm_eps)?;

        // SwiGLU FFN (Phi-3's MLP is SwiGLU even though it stores
        // gate+up fused on disk).
        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size);

        h1.add(&ffn_out)
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

/// Split a fused QKV transposed matrix (shape [hidden_size, qkv_out]) into Q/K/V.
/// Phi3 uses MQA-like qkv_out = q_dim + 2*kv_dim with Q occupying the first
/// q_dim columns then K then V.
fn split_phi3_qkv(
    transposed: &[f32],
    hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let qkv_out = q_dim + 2 * kv_dim;
    let mut q = vec![0.0_f32; hidden_size * q_dim];
    let mut k = vec![0.0_f32; hidden_size * kv_dim];
    let mut v = vec![0.0_f32; hidden_size * kv_dim];
    for row in 0..hidden_size {
        let src = &transposed[row * qkv_out..(row + 1) * qkv_out];
        q[row * q_dim..(row + 1) * q_dim].copy_from_slice(&src[0..q_dim]);
        k[row * kv_dim..(row + 1) * kv_dim].copy_from_slice(&src[q_dim..q_dim + kv_dim]);
        v[row * kv_dim..(row + 1) * kv_dim].copy_from_slice(&src[q_dim + kv_dim..]);
    }
    (q, k, v)
}

/// Split fused gate_up_proj [hidden_size, 2*intermediate] into gate and up.
fn split_phi3_gate_up(transposed: &[f32], hidden_size: usize, inter: usize) -> (Vec<f32>, Vec<f32>) {
    let out_dim = 2 * inter;
    let mut gate = vec![0.0_f32; hidden_size * inter];
    let mut up = vec![0.0_f32; hidden_size * inter];
    for row in 0..hidden_size {
        let src = &transposed[row * out_dim..(row + 1) * out_dim];
        gate[row * inter..(row + 1) * inter].copy_from_slice(&src[0..inter]);
        up[row * inter..(row + 1) * inter].copy_from_slice(&src[inter..]);
    }
    (gate, up)
}

impl Phi3Weights {
    /// Load Phi-3 weights from HF safetensors (e.g. `microsoft/Phi-3-mini-4k-instruct`).
    /// Phi-3 uses fused qkv_proj + fused gate_up_proj — split at load time.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &Phi3Config,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix};
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let inter = cfg.intermediate_size;

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);

        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let qkv = load_transposed_matrix(
                st, &format!("{p}.self_attn.qkv_proj.weight"), q_dim + 2 * kv_dim, h,
            )?;
            let (q, k, v) = split_phi3_qkv(&qkv, h, q_dim, kv_dim);
            let attn_o = crate::lazy::load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.o_proj.weight"), h, q_dim,
            )?;

            let gate_up = load_transposed_matrix(
                st, &format!("{p}.mlp.gate_up_proj.weight"), 2 * inter, h,
            )?;
            let (gate, up) = split_phi3_gate_up(&gate_up, h, inter);
            let ffn_down = crate::lazy::load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.mlp.down_proj.weight"), h, inter,
            )?;

            let attn_norm_gain = Arc::from(load_tensor_as_f32(st, &format!("{p}.input_layernorm.weight"))?);
            let ffn_norm_gain = Arc::from(load_tensor_as_f32(st, &format!("{p}.post_attention_layernorm.weight"))?);

            layers.push(LayerWeights {
                attn_q: WeightStorage::F32(Arc::from(q)),
                attn_q_bias: None,
                attn_k: WeightStorage::F32(Arc::from(k)),
                attn_k_bias: None,
                attn_v: WeightStorage::F32(Arc::from(v)),
                attn_v_bias: None,
                attn_o,
                ffn_gate: WeightStorage::F32(Arc::from(gate)),
                ffn_up: WeightStorage::F32(Arc::from(up)),
                ffn_down,
                attn_norm_gain,
                ffn_norm_gain,
            });
        }

        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.norm.weight")?);
        let output = if cfg.tie_word_embeddings {
            crate::lazy_llama_full::tied_lm_head_from_embeddings(&token_embedding, cfg.vocab_size, h)
        } else {
            crate::lazy::load_transposed_matrix_preserve_dtype(
                st, "lm_head.weight", cfg.vocab_size, h,
            )?
        };

        Ok(Self { token_embedding, layers, final_norm_gain, output })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &Phi3Config) -> Phi3Weights {
        let mut s: u32 = 8888;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim();
        let mut next_box: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *next_box);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
            attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *next_box)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *next_box)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *next_box)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *next_box));
        Phi3Weights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = Phi3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let model = Phi3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// `forward_hidden` returns post-RmsNorm hidden states
    /// `(1, seq, hidden_size)` without the lm_head matmul.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = Phi3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let model = Phi3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    fn forward_embeds_test_cfg() -> Phi3Config {
        Phi3Config {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0, rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = Phi3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
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
            "Phi3 forward vs forward_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = Phi3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let bad = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]), &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = forward_embeds_test_cfg();
        let model = Phi3Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
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
            "Phi3 forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})");
    }
}

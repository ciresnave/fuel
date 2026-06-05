//! BigCode (StarCoder-1) decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. BigCode is a GPT-2-style transformer:
//! - **Learned absolute position embeddings** (`wpe`) added to the
//!   token embedding (`wte`) at the start. No RoPE, no ALiBi.
//! - **Multi-query attention** — single shared K and V across all
//!   attention heads (`multi_query == true` by default).
//! - LayerNorm with bias on input + post-attention paths.
//! - **GELU MLP** — `down(gelu(up(x)))`, no gate path.
//! - Q/K/V/O and MLP projections all have biases.
//!
//! The learned-position path is the only thing fundamentally
//! different from StarCoder2 — replace the RoPE step with a single
//! `index_select(wpe, [start_pos..start_pos+seq])` + `broadcast_add`
//! to the token embedding.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct BigCodeConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub layer_norm_epsilon: f64,
    pub intermediate_size: usize,
    /// `true` for StarCoder-1 — single shared K/V across all
    /// attention heads.
    pub multi_query: bool,
}

impl BigCodeConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn kv_dim(&self) -> usize {
        if self.multi_query { self.head_dim() }
        else { self.hidden_size }
    }

    /// StarCoder-1 ~1B preset (HuggingFace `bigcode/starcoder`).
    pub fn starcoder_1b() -> Self {
        Self {
            vocab_size: 49_152,
            max_position_embeddings: 8192,
            num_hidden_layers: 24,
            hidden_size: 2048,
            num_attention_heads: 16,
            layer_norm_epsilon: 1e-5,
            intermediate_size: 8192,
            multi_query: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BigCodeLayerWeights {
    pub input_ln_gain: Arc<[f32]>,
    pub input_ln_bias: Arc<[f32]>,
    pub post_attn_ln_gain: Arc<[f32]>,
    pub post_attn_ln_bias: Arc<[f32]>,
    pub attn_q: WeightStorage,
    pub attn_q_bias: Arc<[f32]>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Arc<[f32]>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Arc<[f32]>,
    pub attn_o: WeightStorage,
    pub attn_o_bias: Arc<[f32]>,
    pub mlp_fc: WeightStorage,
    pub mlp_fc_bias: Arc<[f32]>,
    pub mlp_proj: WeightStorage,
    pub mlp_proj_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct BigCodeWeights {
    /// `[vocab_size, hidden_size]` — wte.
    pub token_embedding: Arc<[f32]>,
    /// `[max_position_embeddings, hidden_size]` — wpe (learned
    /// positional embedding).
    pub position_embedding: Arc<[f32]>,
    pub layers: Vec<BigCodeLayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    /// Often tied to `token_embedding` in BigCode checkpoints; the
    /// safetensors loader handles the tying.
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct BigCodeModel {
    pub config: BigCodeConfig,
    pub weights: BigCodeWeights,
}

impl BigCodeModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Run the decoder forward up to the final LayerNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection. BigCode (StarCoder1)
    /// uses learned absolute position embeddings + LayerNorm
    /// final norm — same backbone is run for both `forward`
    /// and `forward_hidden`.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert_eq!(cfg.num_attention_heads * cfg.head_dim(), cfg.hidden_size);
        assert!(
            start_pos + seq <= cfg.max_position_embeddings,
            "BigCodeModel: start_pos + seq ({}) exceeds max_position_embeddings ({})",
            start_pos + seq, cfg.max_position_embeddings,
        );

        let token_emb = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;

        let wpe = token_emb.const_f32_like(
            weights.position_embedding.clone(),
            Shape::from_dims(&[cfg.max_position_embeddings, cfg.hidden_size]),
        );
        let pos_ids: Vec<u32> = (0..seq).map(|i| (start_pos + i) as u32).collect();
        let pos_ids_t = token_emb.const_u32_like(pos_ids, Shape::from_dims(&[seq]));
        let pos_emb = wpe
            .index_select(0_usize, &pos_ids_t)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;
        let mut h = token_emb.add(&pos_emb)?;

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer)?;
        }
        Ok(h.layer_norm_affine(std::sync::Arc::clone(&weights.final_ln_gain), std::sync::Arc::clone(&weights.final_ln_bias), cfg.layer_norm_epsilon)?)
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &BigCodeLayerWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.kv_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];

        let x_norm = x.layer_norm_affine(std::sync::Arc::clone(&layer.input_ln_gain), std::sync::Arc::clone(&layer.input_ln_bias), cfg.layer_norm_epsilon)?;

        let q = layer.attn_q.apply_linear_with_bias(&x_norm, cfg.hidden_size, cfg.hidden_size, std::sync::Arc::clone(&layer.attn_q_bias))?;
        let k = layer.attn_k.apply_linear_with_bias(&x_norm, cfg.hidden_size, kv_dim, std::sync::Arc::clone(&layer.attn_k_bias))?;
        let v = layer.attn_v.apply_linear_with_bias(&x_norm, cfg.hidden_size, kv_dim, std::sync::Arc::clone(&layer.attn_v_bias))?;

        let n_kv_heads = if cfg.multi_query { 1 } else { cfg.num_attention_heads };
        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(n_kv_heads, head_dim)?;
        let v = v.split_heads(n_kv_heads, head_dim)?;

        // MQA replication: broadcast K and V from 1 → num_heads.
        let n_rep = cfg.num_attention_heads / n_kv_heads;
        let k_full = k.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = LazyTensor::additive_causal_mask_like(x, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer.attn_o.apply_linear_with_bias(&merged, cfg.hidden_size, cfg.hidden_size, std::sync::Arc::clone(&layer.attn_o_bias))?;

        let h1 = x.add(&attn_out)?;
        let h1_norm = h1.layer_norm_affine(std::sync::Arc::clone(&layer.post_attn_ln_gain), std::sync::Arc::clone(&layer.post_attn_ln_bias), cfg.layer_norm_epsilon)?;

        // GELU MLP.
        let mid = layer.mlp_fc.apply_linear_with_bias(&h1_norm, cfg.hidden_size, cfg.intermediate_size, std::sync::Arc::clone(&layer.mlp_fc_bias))?;
        let mid_act = mid.gelu_erf();
        let ffn_out = layer.mlp_proj.apply_linear_with_bias(&mid_act, cfg.intermediate_size, cfg.hidden_size, std::sync::Arc::clone(&layer.mlp_proj_bias))?;

        h1.add(&ffn_out)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_weights(cfg: &BigCodeConfig) -> BigCodeWeights {
        let mut s: u32 = 12121;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size; let i = cfg.intermediate_size;
        let kv = cfg.kv_dim();
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let position_embedding = vec_of(cfg.max_position_embeddings * h, &mut *nb);
        let layers: Vec<BigCodeLayerWeights> = (0..cfg.num_hidden_layers).map(|_| BigCodeLayerWeights {
            input_ln_gain:     Arc::from(vec![1.0_f32; h]),
            input_ln_bias:     Arc::from(vec![0.0_f32; h]),
            post_attn_ln_gain: Arc::from(vec![1.0_f32; h]),
            post_attn_ln_bias: Arc::from(vec![0.0_f32; h]),
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)), attn_q_bias: vec_of(h, &mut *nb),
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_k_bias: vec_of(kv, &mut *nb),
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_v_bias: vec_of(kv, &mut *nb),
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)), attn_o_bias: vec_of(h, &mut *nb),
            mlp_fc: WeightStorage::F32(vec_of(h * i, &mut *nb)), mlp_fc_bias: vec_of(i, &mut *nb),
            mlp_proj: WeightStorage::F32(vec_of(i * h, &mut *nb)), mlp_proj_bias: vec_of(h, &mut *nb),
        }).collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        BigCodeWeights { token_embedding, position_embedding, layers, final_ln_gain, final_ln_bias, output }
    }

    #[test]
    fn forward_shape_and_finite_mqa() {
        let cfg = BigCodeConfig {
            vocab_size: 32, max_position_embeddings: 64, num_hidden_layers: 2,
            hidden_size: 16, num_attention_heads: 4, layer_norm_epsilon: 1e-5,
            intermediate_size: 32, multi_query: true,
        };
        let model = BigCodeModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }

    /// Different start_pos must produce different output (the
    /// learned position embedding pulls different rows).
    #[test]
    fn different_start_pos_changes_output() {
        let cfg = BigCodeConfig {
            vocab_size: 16, max_position_embeddings: 32, num_hidden_layers: 1,
            hidden_size: 8, num_attention_heads: 2, layer_norm_epsilon: 1e-5,
            intermediate_size: 16, multi_query: true,
        };
        let weights = tiny_weights(&cfg);
        let out_0 = BigCodeModel { config: cfg.clone(), weights: weights.clone() }
            .forward(&[1, 2, 3], 0).unwrap().realize_f32();
        let out_5 = BigCodeModel { config: cfg, weights }
            .forward(&[1, 2, 3], 5).unwrap().realize_f32();
        let any_diff = out_0.iter().zip(out_5.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7);
        assert!(any_diff, "different start_pos must change the learned-position output");
    }

    /// `forward_hidden` returns post-LayerNorm hidden states.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = BigCodeConfig {
            vocab_size: 32, max_position_embeddings: 64, num_hidden_layers: 2,
            hidden_size: 16, num_attention_heads: 4, layer_norm_epsilon: 1e-5,
            intermediate_size: 32, multi_query: true,
        };
        let model = BigCodeModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

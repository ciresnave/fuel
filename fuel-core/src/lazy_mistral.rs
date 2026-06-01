//! Mistral decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. The Mistral architecture is "LLaMA with sliding-
//! window attention" — every other piece (GQA, RmsNorm, SwiGLU FFN,
//! RoPE, bias-free Q/K/V/O projections) is shared with LLaMA. We
//! reuse [`crate::lazy::LayerWeights`] directly since the per-layer
//! parameter shape is identical; only the attention mask differs.
//!
//! # Sliding-window mask
//!
//! Eager Mistral builds a `[seq, seq]` (or `[seq, total_seq]` for
//! decode with KV cache) mask where `mask[i, j] = -inf` if either:
//!   - `j > i` (causal — can't attend to future)
//!   - `j + sliding_window <= i` (sliding-window — can't attend past
//!     `sliding_window` tokens back)
//!
//! With `sliding_window = None`, this reduces to a strict lower-
//! triangular causal mask (LLaMA semantics). The lazy port mirrors
//! this — every other piece of `apply_layer` matches the LLaMA path
//! (`crate::lazy::LlamaModel::apply_layer`).
//!
//! # KV cache
//!
//! v1 of this port does NOT use a KV cache — every `forward` call
//! recomputes attention over the full input. This matches the LLaMA
//! lazy port's v1 contract and keeps the prefill / decode patterns
//! uniform. Adding [`crate::lazy_kv_cache::LazyKvCache::append_rotating`]
//! (Phase C) on top is orthogonal plumbing for a later session;
//! `MistralModel` will gain a `forward_with_cache` then, mirroring
//! the LLaMA pair.
//!
//! # Scope
//!
//! - Forward-only. Autograd through Mistral is not exercised here.
//! - Single sequence (`batch == 1`). Multi-batch is a future addition
//!   when InferenceContext starts batching real workloads.
//! - F32 activations; weights via [`crate::lazy::WeightStorage`]
//!   (F32 / BF16 / Q4_0 all work through `apply_linear`).
//!
//! # Weight names (HuggingFace safetensors)
//!
//! Mirrors the eager `fuel_transformers::models::llm::mistral` field
//! ordering for safetensors loader interop:
//!   - `model.embed_tokens.weight` → [`MistralWeights::token_embedding`]
//!   - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` → `attn_{q,k,v,o}`
//!   - `model.layers.{i}.mlp.{gate,up,down}_proj.weight` → `ffn_{gate,up,down}`
//!   - `model.layers.{i}.input_layernorm.weight` → `attn_norm_gain`
//!   - `model.layers.{i}.post_attention_layernorm.weight` → `ffn_norm_gain`
//!   - `model.norm.weight` → `final_norm_gain`
//!   - `lm_head.weight` → `output`
//!
//! All Q/K/V/O biases are `None` (Mistral uses `linear_no_bias`).

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

/// Mistral / Mixtral model configuration.
///
/// Mirrors HuggingFace `config.json` naming where practical so a
/// downloaded config can be deserialized directly (when the
/// safetensors loader is wired). Fields:
///   - `sliding_window`: window size for the sliding-window mask.
///     `Some(4096)` for Mistral-7B-v0.1; `None` reverts to a strict
///     causal mask (i.e. dense attention, equivalent to LLaMA).
#[derive(Debug, Clone, PartialEq)]
pub struct MistralConfig {
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
    pub sliding_window: Option<usize>,
}

impl MistralConfig {
    /// Preset for `mistralai/Mistral-7B-v0.1`. Same layout the eager
    /// `Config::config_7b_v0_1` returns.
    pub fn mistral_7b_v0_1() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 32_768,
            sliding_window: Some(4096),
        }
    }
}

/// Mistral weights. Per-layer K/V/Q/O/FFN tensors reuse
/// [`crate::lazy::LayerWeights`] (the LLaMA shape works unchanged
/// because Mistral's per-layer parameter layout is identical to
/// bias-free LLaMA).
#[derive(Debug, Clone)]
pub struct MistralWeights {
    /// `[vocab_size, hidden_size]` token embedding table.
    pub token_embedding: Arc<[f32]>,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
    /// `[hidden_size]` RmsNorm gain before the lm_head.
    pub final_norm_gain: Arc<[f32]>,
    /// `[hidden_size, vocab_size]` lm_head projection.
    pub output: WeightStorage,
}

/// Mistral LM, lazy-graph form.
#[derive(Debug, Clone)]
pub struct MistralModel {
    pub config: MistralConfig,
    pub weights: MistralWeights,
}

impl MistralModel {
    /// Run a forward pass on `tokens` and return the final logits
    /// `[1, seq, vocab_size]` as a [`LazyTensor`]. Call `.realize_f32()`
    /// on the result to materialize.
    ///
    /// `start_pos` offsets the RoPE frequencies. Pass `0` for the
    /// first forward call of a sequence; for v1 (no KV cache) callers
    /// always re-feed the full prefix so `start_pos` is typically `0`.
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "MistralModel::forward: tokens must be non-empty");
        assert_eq!(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            "MistralConfig: num_attention_heads * head_dim must equal hidden_size",
        );
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads, 0,
            "MistralConfig: num_attention_heads ({}) must be a multiple of num_key_value_heads ({})",
            cfg.num_attention_heads, cfg.num_key_value_heads,
        );

        // Embedding lookup.
        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;

        // Shared RoPE tables (one alloc per forward; reused across layers).
        let (cos_data, sin_data) = fuel_graph::build_rope_tables(
            cfg.rope_theta,
            start_pos,
            seq,
            cfg.head_dim,
        );
        let rope_shape = Shape::from_dims(&[seq, cfg.head_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        // Per-layer decode.
        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }

        // Final RmsNorm + lm_head projection.
        let h_norm = crate::lazy::apply_affine_rms_norm_pub(
            &h, &weights.final_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Build the sliding-window causal mask for a single forward
    /// (v1, no cache). Shape `[1, 1, seq, seq]`, broadcast-ready for
    /// `scores`.
    ///
    /// `mask[i, j] = -inf` if `j > i` (causal) OR
    /// `j + sliding_window <= i` (sliding-window). With
    /// `sliding_window == None` this reduces to a strict lower-tri
    /// causal mask.
    fn build_sliding_window_mask(&self, anchor: &LazyTensor, seq: usize) -> LazyTensor {
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

    /// Single transformer layer. Mirrors
    /// `crate::lazy::LlamaModel::apply_layer` except the attention
    /// mask uses sliding-window semantics.
    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        // Pre-attention RmsNorm (affine).
        let x_norm = crate::lazy::apply_affine_rms_norm_pub(
            x, &layer.attn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );

        // Q / K / V projections (bias-free for Mistral).
        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size);
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim);
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim);

        // Reshape to per-head and transpose to [batch, heads, seq, head_dim].
        let q = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.num_attention_heads, cfg.head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let k = k
            .reshape(Shape::from_dims(&[batch, seq, cfg.num_key_value_heads, cfg.head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let v = v
            .reshape(Shape::from_dims(&[batch, seq, cfg.num_key_value_heads, cfg.head_dim]))?
            .permute([0, 2, 1, 3_usize])?;

        // Apply RoPE on Q and K only.
        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        // GQA replication: bring K and V from `n_kv_heads` to `n_heads`
        // by repeating each KV head `n_rep` times along the head axis.
        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let (k_full, v_full) = if n_rep == 1 {
            (k_r, v)
        } else {
            let expand = |t: LazyTensor| -> Result<LazyTensor> {
                let s5 = t.reshape(Shape::from_dims(&[
                    batch,
                    cfg.num_key_value_heads,
                    1,
                    seq,
                    cfg.head_dim,
                ]))?;
                let bcast = s5.broadcast_to(Shape::from_dims(&[
                    batch,
                    cfg.num_key_value_heads,
                    n_rep,
                    seq,
                    cfg.head_dim,
                ]))?;
                bcast.reshape(Shape::from_dims(&[
                    batch,
                    cfg.num_attention_heads,
                    seq,
                    cfg.head_dim,
                ]))
            };
            (expand(k_r)?, expand(v)?)
        };

        // Scaled dot-product attention with the sliding-window mask.
        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = self.build_sliding_window_mask(x, seq);
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        // Merge heads + output projection.
        let merged = attn_v
            .permute([0, 2, 1, 3_usize])?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size);

        // First residual.
        let h1 = x.add(&attn_out)?;

        // Pre-FFN RmsNorm (affine).
        let h1_norm = crate::lazy::apply_affine_rms_norm_pub(
            &h1, &layer.ffn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );

        // SwiGLU FFN: `down(silu(gate) * up)`.
        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size);

        // Second residual.
        h1.add(&ffn_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LCG-seeded weights so the shape test is deterministic; values
    /// kept small so the forward stays in normal-float range.
    fn tiny_weights(cfg: &MistralConfig) -> MistralWeights {
        let mut s: u32 = 12345;
        let mut next = || -> f32 {
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
        MistralWeights { token_embedding, layers, final_norm_gain, output }
    }

    /// Smoke: a tiny 2-layer Mistral-shape forward produces logits of
    /// the expected shape and contains no NaNs.
    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = MistralConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            sliding_window: Some(4),
        };
        let model = MistralModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        assert_eq!(out.len(), tokens.len() * cfg.vocab_size);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// With `sliding_window = None`, the mask reduces to strict
    /// lower-triangular causal — the first row should attend only to
    /// itself, the last row should attend to every prior + itself.
    /// We verify by comparing the model output against a parallel
    /// run with `sliding_window = Some(seq + 1)` (large enough to
    /// not actually mask anything past the causal cutoff). They must
    /// be element-wise identical.
    #[test]
    fn sliding_window_none_matches_large_window() {
        let mut cfg = MistralConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 32,
            sliding_window: None,
        };
        let weights = tiny_weights(&cfg);
        let tokens: Vec<u32> = vec![3, 1, 4, 1, 5, 9];
        let out_no_window = MistralModel { config: cfg.clone(), weights: weights.clone() }
            .forward(&tokens, 0).unwrap().realize_f32();
        cfg.sliding_window = Some(tokens.len() + 1);
        let out_large_window = MistralModel { config: cfg, weights }
            .forward(&tokens, 0).unwrap().realize_f32();
        assert_eq!(out_no_window, out_large_window);
    }

    /// `sliding_window = Some(2)` must NOT match a strict-causal run
    /// — for seq_len > 2 the window actually restricts attention.
    #[test]
    fn sliding_window_2_differs_from_strict_causal() {
        let cfg_small = MistralConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 32,
            sliding_window: Some(2),
        };
        let mut cfg_full = cfg_small.clone();
        cfg_full.sliding_window = None;
        let weights = tiny_weights(&cfg_small);
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let out_small = MistralModel { config: cfg_small, weights: weights.clone() }
            .forward(&tokens, 0).unwrap().realize_f32();
        let out_full = MistralModel { config: cfg_full, weights }
            .forward(&tokens, 0).unwrap().realize_f32();
        // The two must differ — sliding_window=2 masks history past 2 tokens
        // while full causal attends to everything below.
        let any_diff = out_small.iter().zip(out_full.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(any_diff, "sliding_window=2 should differ from full causal");
    }
}

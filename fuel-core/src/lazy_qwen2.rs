//! Qwen2 (non-MoE) decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Qwen2 = Mistral + Q/K/V biases + per-layer
//! sliding-window gating (`use_sliding_window` + `max_window_layers`).
//! Everything else (GQA, RmsNorm, SwiGLU FFN, RoPE) matches LLaMA /
//! Mistral so we reuse [`crate::lazy::LayerWeights`] directly — the
//! optional `attn_{q,k,v}_bias` fields on `LayerWeights` already
//! handle Qwen2's bias layout.
//!
//! # Sliding window
//!
//! Qwen2's `Config` carries:
//!   - `sliding_window: usize` (always set, e.g. 32768 for 7B)
//!   - `use_sliding_window: bool` — global switch
//!   - `max_window_layers: usize` — first N layers use the window;
//!     remaining layers run dense. Mixed-mode is the canonical Qwen2
//!     setup.
//!
//! The lazy port honors this by building per-layer masks: layer `i`
//! uses the sliding-window mask iff
//! `use_sliding_window && i < max_window_layers`; otherwise dense
//! causal.
//!
//! # Scope
//!
//! Same as the Mistral port — forward-only, single sequence
//! (`batch == 1`), no KV cache (recomputes each call), F32
//! activations, sliding-window mask built per-forward as a const.
//!
//! # Weight names (HuggingFace safetensors)
//!
//! Mirrors eager `fuel_transformers::models::llm::qwen2`:
//!   - `model.embed_tokens.weight` → `token_embedding`
//!   - `model.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}` →
//!     `attn_{q,k,v}` + `attn_{q,k,v}_bias` (biases ARE present)
//!   - `model.layers.{i}.self_attn.o_proj.weight` → `attn_o`
//!     (no bias)
//!   - `model.layers.{i}.mlp.{gate,up,down}_proj.weight` →
//!     `ffn_{gate,up,down}`
//!   - `model.layers.{i}.input_layernorm.weight` → `attn_norm_gain`
//!   - `model.layers.{i}.post_attention_layernorm.weight` →
//!     `ffn_norm_gain`
//!   - `model.norm.weight` → `final_norm_gain`
//!   - `lm_head.weight` → `output` (or tied to `token_embedding` when
//!     `tie_word_embeddings == true`; safetensors loader resolves it)

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    /// First `max_window_layers` layers use the sliding-window mask
    /// when `use_sliding_window == true`; remaining layers run dense.
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
}

impl Qwen2Config {
    /// `head_dim = hidden_size / num_attention_heads`. Convenience
    /// accessor — every Qwen2 size derives head_dim this way.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Preset for `Qwen/Qwen2-7B`. Field values from the HF config.
    pub fn qwen2_7b() -> Self {
        Self {
            vocab_size: 152_064,
            hidden_size: 3584,
            intermediate_size: 18_944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            max_position_embeddings: 131_072,
            sliding_window: 131_072,
            max_window_layers: 28,
            use_sliding_window: false,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen2Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Qwen2Model {
    pub config: Qwen2Config,
    pub weights: Qwen2Weights,
}

impl Qwen2Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "Qwen2Model::forward: tokens must be non-empty");
        let head_dim = cfg.head_dim();
        assert_eq!(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            "Qwen2Config: num_attention_heads * head_dim must equal hidden_size",
        );
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads,
            0,
            "Qwen2Config: num_attention_heads ({}) must be a multiple of num_key_value_heads ({})",
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

        // Shared RoPE tables.
        let (cos_data, sin_data) =
            fuel_graph::build_rope_tables(cfg.rope_theta, start_pos, seq, head_dim);
        let rope_shape = Shape::from_dims(&[seq, head_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        // Per-layer decode. Layer index drives sliding-window gating.
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let uses_window =
                cfg.use_sliding_window && layer_idx < cfg.max_window_layers;
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, uses_window)?;
        }

        // Final RmsNorm + lm_head projection.
        let h_norm = crate::lazy::apply_affine_rms_norm_pub(
            &h, &weights.final_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Run the encoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection — useful for embedding
    /// adapters (Stella-en-v5, NV-Embed v2, etc.) that swap
    /// the causal LM head for a custom projector or pooler.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "Qwen2Model::forward_hidden: tokens must be non-empty");
        let head_dim = cfg.head_dim();
        assert_eq!(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            "Qwen2Config: num_attention_heads * head_dim must equal hidden_size",
        );

        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;

        let (cos_data, sin_data) =
            fuel_graph::build_rope_tables(cfg.rope_theta, start_pos, seq, head_dim);
        let rope_shape = Shape::from_dims(&[seq, head_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let uses_window =
                cfg.use_sliding_window && layer_idx < cfg.max_window_layers;
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, uses_window)?;
        }
        Ok(crate::lazy::apply_affine_rms_norm_pub(
            &h, &weights.final_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        ))
    }

    /// Build the attention mask for one layer. `uses_window == true`
    /// produces the sliding-window causal mask; `false` produces a
    /// strict lower-triangular causal mask.
    fn build_layer_mask(&self, anchor: &LazyTensor, seq: usize, uses_window: bool) -> LazyTensor {
        let cfg = &self.config;
        let window = if uses_window { cfg.sliding_window } else { seq + 1 };
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
        uses_window: bool,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * head_dim;

        let x_norm = crate::lazy::apply_affine_rms_norm_pub(
            x, &layer.attn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );

        // Q / K / V projections with optional biases (Qwen2 has them).
        let q = apply_optional_bias(
            layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size),
            layer.attn_q_bias.as_ref(),
            cfg.hidden_size,
        )?;
        let k = apply_optional_bias(
            layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim),
            layer.attn_k_bias.as_ref(),
            kv_dim,
        )?;
        let v = apply_optional_bias(
            layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim),
            layer.attn_v_bias.as_ref(),
            kv_dim,
        )?;

        let q = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.num_attention_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let k = k
            .reshape(Shape::from_dims(&[batch, seq, cfg.num_key_value_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let v = v
            .reshape(Shape::from_dims(&[batch, seq, cfg.num_key_value_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        // GQA replication.
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
                    head_dim,
                ]))?;
                let bcast = s5.broadcast_to(Shape::from_dims(&[
                    batch,
                    cfg.num_key_value_heads,
                    n_rep,
                    seq,
                    head_dim,
                ]))?;
                bcast.reshape(Shape::from_dims(&[
                    batch,
                    cfg.num_attention_heads,
                    seq,
                    head_dim,
                ]))
            };
            (expand(k_r)?, expand(v)?)
        };

        // Scaled dot-product attention.
        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = self.build_layer_mask(x, seq, uses_window);
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v
            .permute([0, 2, 1, 3_usize])?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size);

        let h1 = x.add(&attn_out)?;
        let h1_norm = crate::lazy::apply_affine_rms_norm_pub(
            &h1, &layer.ffn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );

        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size);

        h1.add(&ffn_out)
    }
}

/// Add an optional `[last_dim]` bias to `x`'s last dim via
/// `broadcast_add`. Returns `x` unchanged when `bias` is `None`.
/// Sibling of `crate::lazy::apply_optional_bias` (which is module-
/// private); duplicated here to keep `lazy.rs` API unchanged.
fn apply_optional_bias(
    x: LazyTensor,
    bias: Option<&Arc<[f32]>>,
    last_dim: usize,
) -> Result<LazyTensor> {
    match bias {
        None => Ok(x),
        Some(b) => {
            assert_eq!(
                b.len(), last_dim,
                "apply_optional_bias: bias length {} != last_dim {last_dim}",
                b.len(),
            );
            let b_t = x.const_f32_like(Arc::clone(b), Shape::from_dims(&[last_dim]));
            x.broadcast_add(&b_t)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &Qwen2Config) -> Qwen2Weights {
        let mut s: u32 = 7777;
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
            attn_q_bias: Some(vec_of(h, &mut *next_box)),
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_k_bias: Some(vec_of(kv, &mut *next_box)),
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_v_bias: Some(vec_of(kv, &mut *next_box)),
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *next_box)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *next_box)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *next_box)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *next_box));
        Qwen2Weights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = Qwen2Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            sliding_window: 4,
            max_window_layers: 1,
            use_sliding_window: true,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let model = Qwen2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        assert_eq!(out.len(), tokens.len() * cfg.vocab_size);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// `use_sliding_window = false` and `use_sliding_window = true`
    /// with `max_window_layers = 0` must produce identical outputs
    /// (no layer actually uses the window in either case).
    #[test]
    fn no_window_paths_match() {
        let cfg_a = Qwen2Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            sliding_window: 2,
            max_window_layers: 2,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let mut cfg_b = cfg_a.clone();
        cfg_b.use_sliding_window = true;
        cfg_b.max_window_layers = 0; // every layer is "past" the window cutoff → dense
        let weights = tiny_weights(&cfg_a);
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let out_a = Qwen2Model { config: cfg_a, weights: weights.clone() }
            .forward(&tokens, 0).unwrap().realize_f32();
        let out_b = Qwen2Model { config: cfg_b, weights }
            .forward(&tokens, 0).unwrap().realize_f32();
        assert_eq!(out_a, out_b);
    }

    /// `max_window_layers > 0` with a real window MUST diverge from
    /// the all-dense run on sequences longer than the window.
    #[test]
    fn window_layers_diverge_from_dense() {
        let mut cfg_window = Qwen2Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            sliding_window: 2,
            max_window_layers: 2, // both layers use the window
            use_sliding_window: true,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let mut cfg_dense = cfg_window.clone();
        cfg_dense.use_sliding_window = false;
        let weights = tiny_weights(&cfg_window);
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let out_window = Qwen2Model { config: cfg_window.clone(), weights: weights.clone() }
            .forward(&tokens, 0).unwrap().realize_f32();
        let _ = cfg_window;
        let out_dense = Qwen2Model { config: cfg_dense, weights }
            .forward(&tokens, 0).unwrap().realize_f32();
        let any_diff = out_window.iter().zip(out_dense.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(any_diff, "sliding window should diverge from dense run");
    }

    /// Q/K/V biases must be honored. Compare a run with all-zero
    /// biases against one with random biases — outputs must differ.
    #[test]
    fn qkv_biases_affect_output() {
        let cfg = Qwen2Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            sliding_window: 32,
            max_window_layers: 0,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let mut wt_zero = tiny_weights(&cfg);
        let zero_h: Arc<[f32]> = Arc::from(vec![0.0_f32; cfg.hidden_size]);
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim();
        let zero_kv: Arc<[f32]> = Arc::from(vec![0.0_f32; kv_dim]);
        for l in &mut wt_zero.layers {
            l.attn_q_bias = Some(zero_h.clone());
            l.attn_k_bias = Some(zero_kv.clone());
            l.attn_v_bias = Some(zero_kv.clone());
        }
        let wt_random = tiny_weights(&cfg);
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let out_zero = Qwen2Model { config: cfg.clone(), weights: wt_zero }
            .forward(&tokens, 0).unwrap().realize_f32();
        let out_random = Qwen2Model { config: cfg, weights: wt_random }
            .forward(&tokens, 0).unwrap().realize_f32();
        let any_diff = out_zero.iter().zip(out_random.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(any_diff, "non-zero Q/K/V biases must change output");
    }
}

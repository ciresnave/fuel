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
        let h_norm = self.run_backbone(tokens, start_pos)?;
        Ok(weights.output.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Run the encoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection — useful for embedding
    /// adapters (Stella-en-v5, etc.) that swap the causal LM
    /// head for a custom projector or pooler.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Like [`Self::forward_hidden`] but takes pre-computed
    /// `embeds` of shape `(1, seq, hidden_size)` and a
    /// caller-supplied `(1, 1, seq, seq)` additive mask. The
    /// mask is used for ALL layers — Qwen2's per-layer
    /// sliding-window gating is skipped. Both `embeds` and
    /// `attention_mask` MUST live on the same graph (build
    /// the mask via `embeds.const_f32_like(...)`).
    ///
    /// Useful for bidirectional encoder mode (mask is just
    /// the pad-only `(1 - mask[j]) * MIN` broadcast). Pass
    /// `0` for keep and `-inf` (or a large negative) for mask.
    pub fn forward_hidden_embeds_with_mask(
        &self,
        embeds: &LazyTensor,
        attention_mask: &LazyTensor,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "embeds must be rank 3 [b, seq, hidden]");
        let seq = dims[1];
        assert_eq!(dims[2], cfg.hidden_size);
        let head_dim = cfg.head_dim();
        assert_eq!(
            cfg.num_attention_heads * head_dim, cfg.hidden_size,
            "Qwen2Config: num_attention_heads * head_dim must equal hidden_size",
        );
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads, 0,
            "Qwen2Config: num_attention_heads must be a multiple of num_key_value_heads",
        );

        let mut h = embeds.clone();
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, head_dim,
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, attention_mask)?;
        }
        Ok(h.rms_norm_affine(std::sync::Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps)?)
    }

    /// Shared backbone for the causal-mask paths
    /// (`forward` and `forward_hidden`). Embed → RoPE →
    /// per-layer attn+MLP → final RmsNorm. Builds a
    /// sliding-window or strict-causal mask per layer based
    /// on the config. For non-causal use (bidirectional
    /// encoder mode), see [`Self::forward_hidden_embeds_with_mask`].
    /// Like [`Self::forward_hidden`] but takes pre-computed
    /// `embeds` of shape `(1, seq, hidden_size)` and uses the
    /// standard per-layer (sliding-window / strict-causal)
    /// mask construction. Skips the LM head. Use this from
    /// multimodal hosts that interleave image embeddings into
    /// the text stream (LLaVA-style consumers) and want hidden
    /// states without the lm_head projection.
    pub fn forward_hidden_embeds(
        &self,
        embeds: &LazyTensor,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        self.run_backbone_embeds(embeds, start_pos)
    }

    fn run_backbone(
        &self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "Qwen2Model: tokens must be non-empty");

        let h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;
        self.run_backbone_embeds(&h, start_pos)
    }

    fn run_backbone_embeds(
        &self,
        embeds: &LazyTensor,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "embeds must be rank 3 [b, seq, hidden]");
        let seq = dims[1];
        assert_eq!(dims[2], cfg.hidden_size);
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

        let mut h = embeds.clone();
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, head_dim,
        );

        let causal_window = if cfg.use_sliding_window {
            Some(self.build_layer_mask(&h, seq, true))
        } else {
            None
        };
        let causal_strict = self.build_layer_mask(&h, seq, false);

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let uses_window =
                cfg.use_sliding_window && layer_idx < cfg.max_window_layers;
            let mask = if uses_window {
                causal_window.as_ref().expect("windowed mask built when use_sliding_window")
            } else {
                &causal_strict
            };
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, mask)?;
        }
        Ok(h.rms_norm_affine(std::sync::Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps)?)
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
        mask: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * head_dim;

        let x_norm = x.rms_norm_affine(std::sync::Arc::clone(&layer.attn_norm_gain), cfg.rms_norm_eps)?;

        // Q / K / V projections with optional biases (Qwen2 has them).
        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size).add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        // GQA replication.
        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        // Scaled dot-product attention with caller-supplied mask.
        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let _ = seq; // silence unused after refactor; mask already sized for seq.
        let scores_masked = scores_scaled.broadcast_add(mask)?;
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

/// Add an optional `[last_dim]` bias to `x`'s last dim via
/// `broadcast_add`. Returns `x` unchanged when `bias` is `None`.
/// Delegates to `LazyTensor::add_optional_trailing_bias` — the
/// per-port wrapper is preserved so call sites inside this module
/// keep their existing signature.

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

    /// `forward_hidden_embeds_with_mask` accepts pre-computed
    /// embeds plus a caller-supplied `(1, 1, seq, seq)`
    /// additive mask. A bidirectional pad mask (all zeros)
    /// produces different hidden states than the strict-causal
    /// `forward_hidden` because the bidirectional path lets
    /// earlier tokens attend to later ones too.
    #[test]
    fn forward_hidden_embeds_with_bidirectional_mask() {
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
        let model = Qwen2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let h_causal = model.forward_hidden(&tokens, 0).unwrap().realize_f32();

        // Build embeds externally and the bidirectional mask
        // anchored on the same graph as embeds.
        let embed_table = LazyTensor::from_f32(
            model.weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = embed_table.const_u32_like(
            tokens.clone(), Shape::from_dims(&[tokens.len()]),
        );
        let embeds = embed_table
            .index_select(0_usize, &token_ids).unwrap()
            .reshape(Shape::from_dims(&[1, tokens.len(), cfg.hidden_size])).unwrap();
        let zero_mask: Arc<[f32]> = Arc::from(vec![0.0_f32; tokens.len() * tokens.len()]);
        let mask = embeds.const_f32_like(
            zero_mask, Shape::from_dims(&[1, 1, tokens.len(), tokens.len()]),
        );
        let h_bidir = model.forward_hidden_embeds_with_mask(&embeds, &mask, 0).unwrap().realize_f32();
        assert_eq!(h_causal.len(), h_bidir.len());
        let mut max_diff = 0.0_f32;
        for (x, y) in h_causal.iter().zip(h_bidir.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "bidirectional hidden states must differ from causal, max_diff = {max_diff}");
        for &v in &h_bidir {
            assert!(v.is_finite(), "non-finite bidirectional hidden: {v}");
        }
    }

    /// `forward_hidden_embeds(embeds, start_pos)` must produce
    /// the same hidden states as `forward_hidden(tokens, start_pos)`
    /// when the embeds are built from the token-embedding table —
    /// proves the embed-lookup is the only difference and the
    /// per-layer mask construction matches the tokens path.
    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = Qwen2Config {
            vocab_size: 16, hidden_size: 8, intermediate_size: 16,
            num_hidden_layers: 1, num_attention_heads: 2, num_key_value_heads: 1,
            max_position_embeddings: 32, sliding_window: 32, max_window_layers: 0,
            use_sliding_window: false, rope_theta: 10_000.0, rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let model = Qwen2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let from_tokens = model.forward_hidden(&tokens, 0).unwrap().realize_f32();

        let embed_table = LazyTensor::from_f32(
            model.weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = embed_table.const_u32_like(
            tokens.clone(), Shape::from_dims(&[tokens.len()]),
        );
        let embeds = embed_table
            .index_select(0_usize, &token_ids).unwrap()
            .reshape(Shape::from_dims(&[1, tokens.len(), cfg.hidden_size])).unwrap();
        let from_embeds = model.forward_hidden_embeds(&embeds, 0).unwrap().realize_f32();
        assert_eq!(from_tokens.len(), from_embeds.len());
        for (a, b) in from_tokens.iter().zip(from_embeds.iter()) {
            assert!((a - b).abs() < 1e-6,
                "forward_hidden_embeds must match forward_hidden: {a} vs {b}");
        }
    }
}

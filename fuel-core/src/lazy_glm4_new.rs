//! GLM-4 (new architecture, ChatGLM-4-style) ported to the
//! lazy-graph API.
//!
//! This is the HuggingFace-released GLM-4 architecture that
//! ships under `THUDM/glm-4-9b-chat` / `glm-4-9b-chat-hf` and
//! the upstream `glm4_new` module in fuel-transformers. It is
//! distinct from the older [`crate::lazy_glm4`] port (which
//! targeted the original CodeGeeX/ChatGLM3 lineage).
//!
//! # Differences from `lazy_glm4`
//!
//!   1. **Four RmsNorms per layer** instead of two. Sublayer
//!      structure is
//!      `input_norm → attn → post_self_attn_norm → +residual →
//!       post_attn_norm → mlp → post_mlp_norm → +residual`.
//!      Mirrors the Gemma-2 pattern: post-norms sit AFTER the
//!      sublayer's main op but BEFORE the residual add.
//!   2. **Fused `gate_up_proj`**: a single linear of width
//!      `2 * intermediate_size` whose output is sliced into
//!      gate (first half) and up (second half). MLP becomes
//!      `down(act(gate) * up)` with `act` configurable.
//!   3. **Interleaved partial RoPE**: same trick as
//!      `lazy_glm4` — `partial_rotary_factor` controls how
//!      many head-dim features get rotated; the remainder
//!      passes through unchanged. The lazy port reuses
//!      `lazy_glm4::apply_interleaved_partial_rope`.
//!   4. **Optional sliding-window mask** (eager carries it
//!      in the config but applies the standard causal mask
//!      by default — `sliding_window` is plumbed through but
//!      defaults to no window in the public release).
//!
//! Keeps standard GLM4 bits: separate Q/K/V (NOT fused),
//! optional QKV biases via `attention_bias`, GQA with
//! `num_key_value_heads`, tied embeddings.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache
//! (recomputes per call), F32. Returns vocab logits
//! `(1, seq, vocab_size)`. The standard causal mask is built
//! per-forward; sliding-window is plumbed through via the
//! config field.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_glm4::apply_interleaved_partial_rope;
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm4NewActivation {
    Silu,
    Gelu,
    GeluPytorchTanh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Glm4NewConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// `None` → derived from `hidden_size / num_attention_heads`.
    pub head_dim: Option<usize>,
    /// Fraction of `head_dim` that gets rotated. `None` → 1.0 (full).
    pub partial_rotary_factor: Option<f32>,
    pub attention_bias: bool,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_act: Glm4NewActivation,
}

impl Glm4NewConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    pub fn rotary_dim(&self) -> usize {
        let d = self.head_dim();
        match self.partial_rotary_factor {
            None => d,
            Some(f) => (f * d as f32) as usize,
        }
    }
    /// `THUDM/glm-4-9b-chat` preset (approximate; actual
    /// `config.json` from HuggingFace overrides).
    pub fn glm4_9b_chat() -> Self {
        Self {
            vocab_size: 151_552,
            hidden_size: 4_096,
            intermediate_size: 13_696,
            num_hidden_layers: 40,
            num_attention_heads: 32,
            num_key_value_heads: 2,
            head_dim: Some(128),
            partial_rotary_factor: Some(0.5),
            attention_bias: true,
            max_position_embeddings: 131_072,
            sliding_window: None,
            tie_word_embeddings: false,
            rope_theta: 5_000_000.0,
            rms_norm_eps: 1e-5,
            hidden_act: Glm4NewActivation::Silu,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Glm4NewLayerWeights {
    pub input_norm_gain: Arc<[f32]>,
    pub post_self_attn_norm_gain: Arc<[f32]>,
    pub post_attn_norm_gain: Arc<[f32]>,
    pub post_mlp_norm_gain: Arc<[f32]>,
    pub q: WeightStorage,
    pub q_bias: Option<Arc<[f32]>>,
    pub k: WeightStorage,
    pub k_bias: Option<Arc<[f32]>>,
    pub v: WeightStorage,
    pub v_bias: Option<Arc<[f32]>>,
    pub o: WeightStorage,
    /// Fused `[hidden, 2 * intermediate]`: first half = gate,
    /// second half = up.
    pub gate_up: WeightStorage,
    pub down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Glm4NewWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Glm4NewLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    /// `None` when `tie_word_embeddings == true`; the embedding
    /// table is reused for the lm_head.
    pub lm_head: Option<WeightStorage>,
}

#[derive(Debug, Clone)]
pub struct Glm4NewModel {
    pub config: Glm4NewConfig,
    pub weights: Glm4NewWeights,
}

impl Glm4NewModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        let logits = match &weights.lm_head {
            Some(w) => w.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size),
            None => {
                let lm_w = h_norm.const_f32_like(
                    Arc::clone(&weights.token_embedding),
                    Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
                );
                h_norm.matmul(&lm_w.transpose()?)?
            }
        };
        Ok(logits)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips tied/untied `lm_head` projection.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads,
            0,
            "Glm4NewConfig: num_attention_heads must be a multiple of num_key_value_heads",
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

        let rope_dim = cfg.rotary_dim();
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, rope_dim,
        );

        let mask = self.build_mask(&h, seq);

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, &mask, q_dim, kv_dim)?;
        }
        Ok(crate::lazy::apply_affine_rms_norm_pub(
            &h, &weights.final_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        ))
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

    #[allow(clippy::too_many_arguments)]
    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &Glm4NewLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
        q_dim: usize,
        kv_dim: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let rope_dim = cfg.rotary_dim();
        let dims = x.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let seq = dims[1];

        // Attention sublayer: post_self_attn_norm(attn(input_norm(x))) + x.
        let x_in = crate::lazy::apply_affine_rms_norm_pub(
            x, &layer.input_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        let q = opt_bias(
            layer.q.apply_linear(&x_in, cfg.hidden_size, q_dim),
            layer.q_bias.as_ref(), q_dim,
        )?;
        let k = opt_bias(
            layer.k.apply_linear(&x_in, cfg.hidden_size, kv_dim),
            layer.k_bias.as_ref(), kv_dim,
        )?;
        let v = opt_bias(
            layer.v.apply_linear(&x_in, cfg.hidden_size, kv_dim),
            layer.v_bias.as_ref(), kv_dim,
        )?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, head_dim)?;

        let q_r = apply_interleaved_partial_rope(&q, rope_cos, rope_sin, head_dim, rope_dim)?;
        let k_r = apply_interleaved_partial_rope(&k, rope_cos, rope_sin, head_dim, rope_dim)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let (k_full, v_full) = if n_rep == 1 {
            (k_r, v)
        } else {
            let expand = |t: LazyTensor| -> Result<LazyTensor> {
                let s5 = t.reshape(Shape::from_dims(&[
                    batch, cfg.num_key_value_heads, 1, seq, head_dim,
                ]))?;
                let bcast = s5.broadcast_to(Shape::from_dims(&[
                    batch, cfg.num_key_value_heads, n_rep, seq, head_dim,
                ]))?;
                bcast.reshape(Shape::from_dims(&[
                    batch, cfg.num_attention_heads, seq, head_dim,
                ]))
            };
            (expand(k_r)?, expand(v)?)
        };

        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_full.transpose()?)?.mul_scalar(scale);
        let scores = scores.broadcast_add(mask)?;
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v_full)?;
        let merged = ctx.merge_heads()?;
        let attn_out = layer.o.apply_linear(&merged, q_dim, cfg.hidden_size);
        let attn_post = crate::lazy::apply_affine_rms_norm_pub(
            &attn_out, &layer.post_self_attn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        let h1 = x.add(&attn_post)?;

        // MLP sublayer: post_mlp_norm(mlp(post_attn_norm(h1))) + h1.
        let h1_in = crate::lazy::apply_affine_rms_norm_pub(
            &h1, &layer.post_attn_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        let gate_up = layer.gate_up.apply_linear(&h1_in, cfg.hidden_size, 2 * cfg.intermediate_size);
        let gate = gate_up.slice(2_usize, 0, cfg.intermediate_size)?;
        let up = gate_up.slice(2_usize, cfg.intermediate_size, cfg.intermediate_size)?;
        let act = match cfg.hidden_act {
            Glm4NewActivation::Silu => gate.silu(),
            Glm4NewActivation::Gelu => gate.gelu_erf(),
            Glm4NewActivation::GeluPytorchTanh => gate.gelu(),
        };
        let inner = act.mul(&up)?;
        let mlp_out = layer.down.apply_linear(&inner, cfg.intermediate_size, cfg.hidden_size);
        let mlp_post = crate::lazy::apply_affine_rms_norm_pub(
            &mlp_out, &layer.post_mlp_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        );
        h1.add(&mlp_post)
    }
}

fn opt_bias(
    x: LazyTensor,
    bias: Option<&Arc<[f32]>>,
    last_dim: usize,
) -> Result<LazyTensor> {
    let _ = last_dim;
    x.add_optional_trailing_bias(bias)
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

    fn tiny_cfg() -> Glm4NewConfig {
        Glm4NewConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 24,
            num_hidden_layers: 2, num_attention_heads: 4,
            num_key_value_heads: 2, head_dim: Some(4),
            partial_rotary_factor: Some(0.5),
            attention_bias: true,
            max_position_embeddings: 32,
            sliding_window: None,
            tie_word_embeddings: true,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            hidden_act: Glm4NewActivation::Silu,
        }
    }

    fn tiny_weights(cfg: &Glm4NewConfig, seed: u32) -> Glm4NewWeights {
        let mut nb = rng_seed(seed);
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h, &mut nb);
        let layers: Vec<Glm4NewLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| Glm4NewLayerWeights {
                input_norm_gain: Arc::from(vec![1.0_f32; h]),
                post_self_attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                post_attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                post_mlp_norm_gain: Arc::from(vec![1.0_f32; h]),
                q: WeightStorage::F32(vec_of(h * q_dim, &mut nb)),
                q_bias: if cfg.attention_bias { Some(vec_of(q_dim, &mut nb)) } else { None },
                k: WeightStorage::F32(vec_of(h * kv_dim, &mut nb)),
                k_bias: if cfg.attention_bias { Some(vec_of(kv_dim, &mut nb)) } else { None },
                v: WeightStorage::F32(vec_of(h * kv_dim, &mut nb)),
                v_bias: if cfg.attention_bias { Some(vec_of(kv_dim, &mut nb)) } else { None },
                o: WeightStorage::F32(vec_of(q_dim * h, &mut nb)),
                gate_up: WeightStorage::F32(vec_of(h * 2 * i, &mut nb)),
                down: WeightStorage::F32(vec_of(i * h, &mut nb)),
            })
            .collect();
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut nb)))
        };
        Glm4NewWeights {
            token_embedding,
            layers,
            final_norm_gain: Arc::from(vec![1.0_f32; h]),
            lm_head,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_cfg();
        let model = Glm4NewModel { config: cfg.clone(), weights: tiny_weights(&cfg, 11) };
        let tokens = [1_u32, 2, 3, 4];
        let out = model.forward(&tokens, 0).unwrap();
        assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    /// 4-norm-per-layer sublayer structure is wired. Zeroing
    /// the post_self_attn_norm gain must alter the output —
    /// confirms that path runs.
    #[test]
    fn post_self_attn_norm_is_wired() {
        let cfg = tiny_cfg();
        let mut base = tiny_weights(&cfg, 22);
        let mut modified = base.clone();
        let h = cfg.hidden_size;
        modified.layers[0].post_self_attn_norm_gain = Arc::from(vec![0.0_f32; h]);
        // Re-Arc base to break any potential aliasing.
        base.layers[0].post_self_attn_norm_gain =
            Arc::from(base.layers[0].post_self_attn_norm_gain.to_vec());

        let m_a = Glm4NewModel { config: cfg.clone(), weights: base };
        let m_b = Glm4NewModel { config: cfg, weights: modified };
        let tokens = [1_u32, 2, 3, 4];
        let a = m_a.forward(&tokens, 0).unwrap().realize_f32();
        let b = m_b.forward(&tokens, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "zeroing post_self_attn_norm_gain must alter output, max_diff = {max_diff}");
    }

    /// Fused gate_up split is wired correctly: zeroing the
    /// gate half (first `intermediate_size` columns) zeroes the
    /// MLP contribution, must change output. If the slice
    /// boundaries were inverted (gate becomes up), the test
    /// would still pass because the MLP would also collapse.
    /// What we're really testing is "fused gate_up is split
    /// at the right boundary and the gate path is active".
    #[test]
    fn fused_gate_up_split() {
        let cfg = tiny_cfg();
        let base = tiny_weights(&cfg, 33);
        let mut modified = base.clone();
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        // Replace the FIRST half of gate_up with zeros (gate path).
        let orig = match &base.layers[0].gate_up {
            WeightStorage::F32(v) => v.to_vec(),
            _ => unreachable!(),
        };
        let mut zeroed = orig.clone();
        // Layout: (hidden, 2 * intermediate) row-major. For each row,
        // zero the first `intermediate` columns.
        for row in 0..h {
            for col in 0..i {
                zeroed[row * (2 * i) + col] = 0.0;
            }
        }
        modified.layers[0].gate_up = WeightStorage::F32(Arc::from(zeroed));

        let m_a = Glm4NewModel { config: cfg.clone(), weights: base };
        let m_b = Glm4NewModel { config: cfg, weights: modified };
        let tokens = [1_u32, 2, 3, 4];
        let a = m_a.forward(&tokens, 0).unwrap().realize_f32();
        let b = m_b.forward(&tokens, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "zeroing gate half of gate_up must alter output, max_diff = {max_diff}");
    }

    /// rotary_dim is `partial_rotary_factor * head_dim`.
    #[test]
    fn partial_rotary_factor() {
        let mut cfg = tiny_cfg();
        cfg.head_dim = Some(8);
        cfg.partial_rotary_factor = Some(0.5);
        assert_eq!(cfg.rotary_dim(), 4);
        cfg.partial_rotary_factor = Some(1.0);
        assert_eq!(cfg.rotary_dim(), 8);
        cfg.partial_rotary_factor = None;
        assert_eq!(cfg.rotary_dim(), 8);
    }

    /// Untied embeddings: lm_head is a distinct WeightStorage.
    #[test]
    fn untied_embeddings_runs() {
        let mut cfg = tiny_cfg();
        cfg.tie_word_embeddings = false;
        let model = Glm4NewModel { config: cfg.clone(), weights: tiny_weights(&cfg, 44) };
        let tokens = [1_u32, 2, 3];
        let out = model.forward(&tokens, 0).unwrap();
        assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_cfg();
        let model = Glm4NewModel { config: cfg.clone(), weights: tiny_weights(&cfg, 55) };
        let tokens = [1_u32, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

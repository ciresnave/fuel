//! Parler-TTS audio token decoder — lazy port.
//!
//! Multi-codebook transformer decoder that generates per-codebook
//! audio token logits conditioned on a text encoder's hidden
//! states. Architecture is BART-style with Post-LN inside each
//! sublayer wrapped in residual:
//!   self_attn(causal) → +res → cross_attn → +res → FFN → +res
//!
//! Inputs:
//!   - `input_ids`: U32 LazyTensor of shape `(1, num_codebooks, T)`.
//!     Per-codebook embeddings are looked up and summed.
//!   - `prompt_embeds` (optional): `(1, P, hidden_size)` — when
//!     present, prepended to the embedded codebook tokens before
//!     the decoder layers.
//!   - `encoder_states`: text-encoder output for cross-attention.
//!     For Parler, this is T5's `forward_encoder` output (already
//!     optionally projected via `enc_to_dec_proj` to match
//!     `hidden_size`).
//!   - `start_pos`: position-embedding offset for resume-style
//!     decoding (prefill = 0).
//!
//! Output: `Vec<LazyTensor>` of length `num_codebooks`, each
//! `(1, P+T or 1, vocab_size)` per-codebook lm-head logits.
//!
//! v1 scope:
//!   - F32, batch == 1, prefill only (no KV cache).
//!   - GQA: `num_kv_heads` / `num_cross_kv_heads` can differ from
//!     `num_heads`; K/V are tiled to match Q's head count.
//!   - Learned absolute positional embeddings (RoPE deferred).
//!   - Configurable activation (Gelu / GeluPytorchTanh / Relu /
//!     Silu — matches the eager `Activation` enum subset).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParlerActivation {
    Gelu,
    GeluPytorchTanh,
    Relu,
    Silu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParlerDecoderConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub ffn_dim: usize,
    pub num_attention_heads: usize,
    /// Defaults to `num_attention_heads`.
    pub num_kv_heads: Option<usize>,
    /// Defaults to `num_kv_heads`.
    pub num_cross_kv_heads: Option<usize>,
    pub activation_function: ParlerActivation,
    pub hidden_size: usize,
    pub num_codebooks: usize,
    /// True when `enc_to_dec_proj` is present (text encoder
    /// `d_model` ≠ decoder `hidden_size`).
    pub has_enc_proj: bool,
    /// True if `embed_prompts` table is present and prompt
    /// embeddings are computed by looking up `prompt_tokens`.
    /// v1 takes precomputed `prompt_embeds` directly so this flag
    /// is informational only.
    pub has_prompt_embedding: bool,
}

impl ParlerDecoderConfig {
    pub fn num_kv_heads_resolved(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_attention_heads)
    }
    pub fn num_cross_kv_heads_resolved(&self) -> usize {
        self.num_cross_kv_heads.unwrap_or(self.num_kv_heads_resolved())
    }
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ---- Weight structures ------------------------------------------------------

/// Linear with optional bias.
#[derive(Debug, Clone)]
pub struct LinearWeights {
    pub w: WeightStorage,
    pub b: Option<Arc<[f32]>>,
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Debug, Clone)]
pub struct AttnProjections {
    pub q_proj: WeightStorage,
    pub k_proj: WeightStorage,
    pub v_proj: WeightStorage,
    pub out_proj: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    pub gain: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct ParlerDecoderLayerWeights {
    pub self_attn: AttnProjections,
    pub self_attn_ln: LayerNormWeights,
    pub cross_attn: AttnProjections,
    pub cross_attn_ln: LayerNormWeights,
    pub fc1: WeightStorage,
    pub fc2: WeightStorage,
    pub final_ln: LayerNormWeights,
}

#[derive(Debug, Clone)]
pub struct ParlerDecoderWeights {
    /// One per codebook: `[vocab_size + 1, hidden_size]`.
    pub embed_tokens: Vec<Arc<[f32]>>,
    /// `[max_position_embeddings, hidden_size]`.
    pub embed_positions: Arc<[f32]>,
    pub layers: Vec<ParlerDecoderLayerWeights>,
    pub final_ln: LayerNormWeights,
    /// One per codebook: `[hidden_size, vocab_size]`.
    pub lm_heads: Vec<WeightStorage>,
    /// Optional `enc_to_dec_proj`: `[text_encoder_d_model, hidden_size]` with bias.
    pub enc_to_dec_proj: Option<LinearWeights>,
}

#[derive(Debug, Clone)]
pub struct ParlerDecoderModel {
    pub config: ParlerDecoderConfig,
    pub weights: ParlerDecoderWeights,
}

// ---- Forward ---------------------------------------------------------------

impl ParlerDecoderModel {
    /// Prefill forward pass returning per-codebook logits.
    ///
    /// * `input_ids` — U32 LazyTensor `(1, num_codebooks, T)`.
    /// * `prompt_embeds` — optional `(1, P, hidden_size)` prepended
    ///   to the codebook embeddings before the decoder layers.
    /// * `encoder_states` — text-encoder output. When the model has
    ///   `enc_to_dec_proj`, it is applied here.
    /// * `start_pos` — position-embedding offset (0 for fresh prefill).
    ///
    /// Returns a `Vec<LazyTensor>` of length `num_codebooks`, each
    /// shaped `(1, P+T, vocab_size)` (or `(1, T, vocab_size)` if no
    /// prompt).
    pub fn forward(
        &self,
        input_ids: &LazyTensor,
        prompt_embeds: Option<&LazyTensor>,
        encoder_states: &LazyTensor,
        start_pos: usize,
    ) -> Result<Vec<LazyTensor>> {
        let cfg = &self.config;
        let w = &self.weights;

        let dims = input_ids.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "input_ids must be rank 3 [B, num_codebooks, T]");
        assert_eq!(dims[0], 1, "v1 supports batch == 1");
        assert_eq!(dims[1], cfg.num_codebooks);
        let t = dims[2];

        // Anchor on input_ids — this is the most-common graph the
        // caller threads through (codes are derived from a shared
        // source). The embedding tables build off of input_ids so
        // index_select stays on-graph; encoder_states is expected
        // to live on the same graph.
        let anchor = input_ids;
        let h_dim = cfg.hidden_size;

        // Sum per-codebook embeddings.
        let mut embed_sum: Option<LazyTensor> = None;
        for (cb, tbl) in w.embed_tokens.iter().enumerate() {
            let ids = input_ids
                .narrow(1_usize, cb, 1)?
                .reshape(Shape::from_dims(&[t]))?;
            let table = input_ids.const_f32_like(
                Arc::clone(tbl),
                Shape::from_dims(&[cfg.vocab_size + 1, h_dim]),
            );
            let lookup = table
                .index_select(0_usize, &ids)?
                .reshape(Shape::from_dims(&[1, t, h_dim]))?;
            embed_sum = Some(match embed_sum {
                None => lookup,
                Some(s) => s.add(&lookup)?,
            });
        }
        let codebook_embeds = embed_sum.expect("num_codebooks must be ≥ 1");

        // Optional prompt prepend.
        let mut x = if let Some(p) = prompt_embeds {
            p.concat(&codebook_embeds, 1_usize)?
        } else {
            codebook_embeds
        };

        // Add learned positional embeddings.
        let total_len = x.shape().dims()[1];
        let pos_table = anchor.const_f32_like(
            Arc::clone(&w.embed_positions),
            Shape::from_dims(&[cfg.max_position_embeddings, h_dim]),
        );
        let pos_ids: Vec<u32> = (0..total_len).map(|i| (i + start_pos) as u32).collect();
        let pos_idx = anchor.const_u32_like(
            pos_ids, Shape::from_dims(&[total_len]),
        );
        let pos = pos_table
            .index_select(0_usize, &pos_idx)?
            .reshape(Shape::from_dims(&[1, total_len, h_dim]))?;
        x = x.add(&pos)?;

        // Project encoder states if needed.
        let enc_proj = if let Some(p) = &w.enc_to_dec_proj {
            apply_linear_with_bias(encoder_states, p, anchor)?
        } else {
            encoder_states.clone()
        };

        // Strict causal mask `(1, 1, total_len, total_len)`.
        let mut mask_data = vec![0.0_f32; total_len * total_len];
        for i in 0..total_len {
            for j in (i + 1)..total_len {
                mask_data[i * total_len + j] = f32::NEG_INFINITY;
            }
        }
        let causal_mask = anchor.const_f32_like(
            mask_data, Shape::from_dims(&[1, 1, total_len, total_len]),
        );

        for layer in &w.layers {
            x = apply_decoder_layer(&x, layer, &enc_proj, &causal_mask, cfg, anchor)?;
        }
        // Final LN.
        let x = apply_layer_norm(&x, &w.final_ln, h_dim, 1e-5)?;

        // Per-codebook LM heads.
        let mut logits = Vec::with_capacity(cfg.num_codebooks);
        for lm in &w.lm_heads {
            logits.push(lm.apply_linear(&x, h_dim, cfg.vocab_size));
        }
        Ok(logits)
    }
}

fn apply_decoder_layer(
    x: &LazyTensor,
    w: &ParlerDecoderLayerWeights,
    enc_states: &LazyTensor,
    causal_mask: &LazyTensor,
    cfg: &ParlerDecoderConfig,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let h_dim = cfg.hidden_size;

    // Self-attention with causal mask.
    let residual = x.clone();
    let normed = apply_layer_norm(x, &w.self_attn_ln, h_dim, 1e-5)?;
    let attn_out = apply_attention(
        &normed, None, &w.self_attn,
        cfg.num_attention_heads, cfg.num_kv_heads_resolved(), cfg.head_dim(),
        h_dim, h_dim, Some(causal_mask), anchor,
    )?;
    let x = residual.add(&attn_out)?;

    // Cross-attention to encoder states.
    let residual = x.clone();
    let normed = apply_layer_norm(&x, &w.cross_attn_ln, h_dim, 1e-5)?;
    let cross_out = apply_attention(
        &normed, Some(enc_states), &w.cross_attn,
        cfg.num_attention_heads, cfg.num_cross_kv_heads_resolved(), cfg.head_dim(),
        h_dim, h_dim, None, anchor,
    )?;
    let x = residual.add(&cross_out)?;

    // FFN.
    let residual = x.clone();
    let normed = apply_layer_norm(&x, &w.final_ln, h_dim, 1e-5)?;
    let h = w.fc1.apply_linear(&normed, h_dim, cfg.ffn_dim);
    let h = match cfg.activation_function {
        ParlerActivation::Gelu => h.gelu(),
        ParlerActivation::GeluPytorchTanh => h.gelu(),
        ParlerActivation::Relu => h.relu(),
        ParlerActivation::Silu => h.silu(),
    };
    let h = w.fc2.apply_linear(&h, cfg.ffn_dim, h_dim);
    residual.add(&h)
}

/// Scaled-softmax multi-head attention with optional GQA. When
/// `kv_input` is `None`, K/V come from `q_input` (self-attn).
#[allow(clippy::too_many_arguments)]
fn apply_attention(
    q_input: &LazyTensor,
    kv_input: Option<&LazyTensor>,
    w: &AttnProjections,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_in_dim: usize,
    kv_in_dim: usize,
    mask: Option<&LazyTensor>,
    _anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let q_dims = q_input.shape();
    let q_dims = q_dims.dims();
    let b = q_dims[0];
    let q_len = q_dims[1];
    let kv_src = kv_input.unwrap_or(q_input);
    let kv_dims = kv_src.shape();
    let kv_dims = kv_dims.dims();
    let kv_len = kv_dims[1];

    let q_out_dim = num_heads * head_dim;
    let kv_out_dim = num_kv_heads * head_dim;

    let q = w.q_proj.apply_linear(q_input, q_in_dim, q_out_dim);
    let k = w.k_proj.apply_linear(kv_src, kv_in_dim, kv_out_dim);
    let v = w.v_proj.apply_linear(kv_src, kv_in_dim, kv_out_dim);

    let scaling = 1.0_f64 / (head_dim as f64).sqrt();
    let q = q.mul_scalar(scaling);

    let q = q.reshape(Shape::from_dims(&[b, q_len, num_heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;
    let k = k.reshape(Shape::from_dims(&[b, kv_len, num_kv_heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;
    let v = v.reshape(Shape::from_dims(&[b, kv_len, num_kv_heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;

    // GQA: repeat K/V along the head dim to match Q's head count.
    let (k, v) = if num_kv_heads == num_heads {
        (k, v)
    } else {
        assert_eq!(num_heads % num_kv_heads, 0);
        let n_rep = num_heads / num_kv_heads;
        let k = repeat_along_head_dim(&k, b, num_kv_heads, n_rep, kv_len, head_dim)?;
        let v = repeat_along_head_dim(&v, b, num_kv_heads, n_rep, kv_len, head_dim)?;
        (k, v)
    };

    let kt = k.permute([0, 1, 3, 2_usize])?;
    let mut scores = q.matmul(&kt)?;
    if let Some(m) = mask {
        let mb = m.broadcast_to(Shape::from_dims(&[b, num_heads, q_len, kv_len]))?;
        scores = scores.add(&mb)?;
    }
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?;
    let ctx = ctx
        .permute([0, 2, 1, 3_usize])?
        .reshape(Shape::from_dims(&[b, q_len, q_out_dim]))?;
    Ok(w.out_proj.apply_linear(&ctx, q_out_dim, q_out_dim))
}

/// Repeat K/V along the head dim: `(B, n_kv, L, D) → (B, n_kv * n_rep, L, D)`
/// via unsqueeze + broadcast_to + reshape.
fn repeat_along_head_dim(
    x: &LazyTensor, b: usize, n_kv: usize, n_rep: usize, l: usize, d: usize,
) -> Result<LazyTensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let unsq = x.reshape(Shape::from_dims(&[b, n_kv, 1, l, d]))?;
    let bc = unsq.broadcast_to(Shape::from_dims(&[b, n_kv, n_rep, l, d]))?;
    bc.reshape(Shape::from_dims(&[b, n_kv * n_rep, l, d]))
}

fn apply_layer_norm(
    x: &LazyTensor,
    ln: &LayerNormWeights,
    hidden: usize,
    eps: f64,
) -> Result<LazyTensor> {
    let _ = hidden;
    x.layer_norm_affine(Arc::clone(&ln.gain), Arc::clone(&ln.bias), eps)
}

fn apply_linear_with_bias(
    x: &LazyTensor,
    w: &LinearWeights,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let projected = w.w.apply_linear(x, w.in_features, w.out_features);
    if let Some(b) = &w.b {
        let bias_t = anchor.const_f32_like(
            Arc::clone(b), Shape::from_dims(&[w.out_features]),
        );
        projected.broadcast_add(&bias_t)
    } else {
        Ok(projected)
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }
    fn vec_of(n: usize, nb: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| nb()).collect::<Vec<_>>())
    }
    fn ws(n: usize, nb: &mut dyn FnMut() -> f32) -> WeightStorage {
        WeightStorage::F32(vec_of(n, nb))
    }
    fn ln_w(c: usize) -> LayerNormWeights {
        LayerNormWeights {
            gain: Arc::from(vec![1.0_f32; c]),
            bias: Arc::from(vec![0.0_f32; c]),
        }
    }
    fn attn_w(
        q_in: usize, kv_in: usize, q_out: usize, kv_out: usize, nb: &mut dyn FnMut() -> f32,
    ) -> AttnProjections {
        AttnProjections {
            q_proj: ws(q_in * q_out, nb),
            k_proj: ws(kv_in * kv_out, nb),
            v_proj: ws(kv_in * kv_out, nb),
            out_proj: ws(q_out * q_out, nb),
        }
    }
    fn layer_w(cfg: &ParlerDecoderConfig, nb: &mut dyn FnMut() -> f32) -> ParlerDecoderLayerWeights {
        let h = cfg.hidden_size;
        let n_kv = cfg.num_kv_heads_resolved();
        let n_xkv = cfg.num_cross_kv_heads_resolved();
        let head_dim = cfg.head_dim();
        ParlerDecoderLayerWeights {
            self_attn: attn_w(h, h, h, n_kv * head_dim, nb),
            self_attn_ln: ln_w(h),
            cross_attn: attn_w(h, h, h, n_xkv * head_dim, nb),
            cross_attn_ln: ln_w(h),
            fc1: ws(h * cfg.ffn_dim, nb),
            fc2: ws(cfg.ffn_dim * h, nb),
            final_ln: ln_w(h),
        }
    }

    fn tiny_config() -> ParlerDecoderConfig {
        ParlerDecoderConfig {
            vocab_size: 32,
            max_position_embeddings: 64,
            num_hidden_layers: 2,
            ffn_dim: 16,
            num_attention_heads: 4,
            num_kv_heads: Some(2),
            num_cross_kv_heads: Some(2),
            activation_function: ParlerActivation::Gelu,
            hidden_size: 8,
            num_codebooks: 2,
            has_enc_proj: false,
            has_prompt_embedding: false,
        }
    }

    fn tiny_weights(cfg: &ParlerDecoderConfig) -> ParlerDecoderWeights {
        let mut nb = rng_seed(2026);
        let h = cfg.hidden_size;
        let embed_tokens: Vec<Arc<[f32]>> = (0..cfg.num_codebooks)
            .map(|_| vec_of((cfg.vocab_size + 1) * h, &mut nb))
            .collect();
        let embed_positions = vec_of(cfg.max_position_embeddings * h, &mut nb);
        let layers: Vec<ParlerDecoderLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| layer_w(cfg, &mut nb))
            .collect();
        let final_ln = ln_w(h);
        let lm_heads: Vec<WeightStorage> = (0..cfg.num_codebooks)
            .map(|_| ws(h * cfg.vocab_size, &mut nb))
            .collect();
        ParlerDecoderWeights {
            embed_tokens, embed_positions, layers, final_ln, lm_heads,
            enc_to_dec_proj: None,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = ParlerDecoderModel { config: cfg.clone(), weights };
        let dev = Device::cpu();
        // (1, num_codebooks, T) U32.
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32; 1], Shape::from_dims(&[1]), &dev,
        );
        let input_ids = anchor.const_u32_like(
            vec![1_u32, 2, 3, 4, 5, 6],
            Shape::from_dims(&[1, cfg.num_codebooks, 3]),
        );
        let encoder_states = anchor.const_f32_like(
            Arc::<[f32]>::from((0..(1 * 5 * cfg.hidden_size))
                .map(|i| (i as f32) * 0.01).collect::<Vec<_>>()),
            Shape::from_dims(&[1, 5, cfg.hidden_size]),
        );
        let logits = model.forward(&input_ids, None, &encoder_states, 0).unwrap();
        assert_eq!(logits.len(), cfg.num_codebooks);
        for (cb, l) in logits.iter().enumerate() {
            assert_eq!(l.shape().dims(), &[1, 3, cfg.vocab_size]);
            for &v in &l.realize_f32() {
                assert!(v.is_finite(), "non-finite logit at codebook {cb}: {v}");
            }
        }
    }

    /// Causal mask: changing a later token must not affect logits
    /// at earlier positions.
    #[test]
    fn causal_mask_enforced() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = ParlerDecoderModel { config: cfg.clone(), weights };
        let dev = Device::cpu();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32; 1], Shape::from_dims(&[1]), &dev,
        );
        let encoder_states = anchor.const_f32_like(
            Arc::<[f32]>::from(vec![0.05_f32; 1 * 4 * cfg.hidden_size]),
            Shape::from_dims(&[1, 4, cfg.hidden_size]),
        );
        // Row-major layout: codebook 0 at slots 0..4, codebook 1 at 4..8.
        // Changing slot 3 (codebook 0 position 3) and slot 7 (codebook
        // 1 position 3) leaves positions 0..2 unchanged.
        let ids_a = anchor.const_u32_like(
            vec![1_u32, 2, 3, 4, 5, 6, 7, 8],
            Shape::from_dims(&[1, cfg.num_codebooks, 4]),
        );
        let ids_b = anchor.const_u32_like(
            vec![1_u32, 2, 3, 9, 5, 6, 7, 9], // only last position of each codebook changed
            Shape::from_dims(&[1, cfg.num_codebooks, 4]),
        );
        let a = model.forward(&ids_a, None, &encoder_states, 0).unwrap();
        let b = model.forward(&ids_b, None, &encoder_states, 0).unwrap();
        for cb in 0..cfg.num_codebooks {
            let av = a[cb].realize_f32();
            let bv = b[cb].realize_f32();
            // Positions 0..=2 must match across the two runs.
            for t in 0..3 {
                for c in 0..cfg.vocab_size {
                    let i = t * cfg.vocab_size + c;
                    assert!((av[i] - bv[i]).abs() < 1e-5,
                        "causal mask violated cb={cb} t={t} c={c}: {} vs {}",
                        av[i], bv[i]);
                }
            }
        }
    }

    /// Cross-attention condition: changing the encoder states must
    /// change the decoder logits.
    #[test]
    fn cross_attention_is_wired() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = ParlerDecoderModel { config: cfg.clone(), weights };
        let dev = Device::cpu();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32; 1], Shape::from_dims(&[1]), &dev,
        );
        let ids = anchor.const_u32_like(
            vec![1_u32, 2, 3, 4],
            Shape::from_dims(&[1, cfg.num_codebooks, 2]),
        );
        let enc_a = anchor.const_f32_like(
            Arc::<[f32]>::from((0..(1 * 4 * cfg.hidden_size))
                .map(|i| (i as f32) * 0.01).collect::<Vec<_>>()),
            Shape::from_dims(&[1, 4, cfg.hidden_size]),
        );
        let enc_b = anchor.const_f32_like(
            Arc::<[f32]>::from((0..(1 * 4 * cfg.hidden_size))
                .map(|i| (i as f32) * 0.01 + 0.5).collect::<Vec<_>>()),
            Shape::from_dims(&[1, 4, cfg.hidden_size]),
        );
        let a = model.forward(&ids, None, &enc_a, 0).unwrap();
        let b = model.forward(&ids, None, &enc_b, 0).unwrap();
        let av = a[0].realize_f32();
        let bv = b[0].realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in av.iter().zip(bv.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "cross-attention must condition decoder on encoder, max_diff = {max_diff}");
    }

    /// With prompt_embeds, output length is P + T.
    #[test]
    fn prompt_prepend_lengthens_output() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = ParlerDecoderModel { config: cfg.clone(), weights };
        let dev = Device::cpu();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32; 1], Shape::from_dims(&[1]), &dev,
        );
        let ids = anchor.const_u32_like(
            vec![1_u32, 2, 3, 4],
            Shape::from_dims(&[1, cfg.num_codebooks, 2]),
        );
        let prompt = anchor.const_f32_like(
            Arc::<[f32]>::from(vec![0.05_f32; 1 * 3 * cfg.hidden_size]),
            Shape::from_dims(&[1, 3, cfg.hidden_size]),
        );
        let enc = anchor.const_f32_like(
            Arc::<[f32]>::from(vec![0.05_f32; 1 * 4 * cfg.hidden_size]),
            Shape::from_dims(&[1, 4, cfg.hidden_size]),
        );
        let logits = model.forward(&ids, Some(&prompt), &enc, 0).unwrap();
        // With prompt P=3, output token length is P + T = 3 + 2 = 5.
        assert_eq!(logits[0].shape().dims(), &[1, 5, cfg.vocab_size]);
    }
}

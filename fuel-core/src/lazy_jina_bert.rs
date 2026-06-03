//! Jina-BERT (jinaai/jina-embeddings-v2-base-en) ported to
//! the lazy-graph API.
//!
//! Jina embedding-v2 is a BERT-family encoder tuned for long
//! contexts. Departures from classic BERT:
//!
//!   1. **ALiBi** (Attention with Linear Biases) instead of
//!      absolute or rotary position embeddings. A learned-free
//!      slope per head is multiplied by `-|i - j|` and broadcast-
//!      added to attention scores before softmax. No position
//!      embedding table; positions extend losslessly past the
//!      training length.
//!   2. **GeGLU FFN with post-LN residual**: gate/non-gate are
//!      `gated_layers(x).chunk(2)`, the inner product passes
//!      through `wo`, and the residual + LN happens AFTER the
//!      MLP (BERT-shape post-LN, not ModernBERT's pre-LN).
//!   3. **Separate Q/K/V** linears (NO fused QKV).
//!   4. **BERT-style post-LN attention output**:
//!      `out = LN(x + dense(attn(x)))`.
//!   5. Token type embeddings (used with all-0 ids by default).
//!   6. Standard absolute embedding LayerNorm.
//!
//! # ALiBi slope construction
//!
//! Let `m` = next power of 2 of `n_heads`. Compute
//! `s_v = -1 / 2^(8v/m)` for `v ∈ [1..=m]`. If `n_heads == m`,
//! use the slopes directly. Otherwise interleave: take odd-
//! indexed slopes then even-indexed ones, capped at `n_heads`.
//! The slopes multiply a precomputed `|i - j|` matrix to give
//! the per-head bias `(n_heads, seq, seq)` that's broadcast-
//! added to attention scores in every layer.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. Returns per-token hidden
//! states `(1, seq, hidden_size)`. Optional additive pad mask
//! shaped `(1, 1, seq, seq)` (broadcast-added on top of ALiBi).
//! Pooling / projection heads stay out of v1 — caller can
//! mean-pool and L2-normalize.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JinaActivation {
    Gelu,
    GeluPytorchTanh,
    Relu,
    Silu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JinaBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub hidden_activation: JinaActivation,
    pub layer_norm_eps: f64,
}

impl JinaBertConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    /// `jinaai/jina-embeddings-v2-base-en` preset.
    pub fn jina_v2_base() -> Self {
        Self {
            vocab_size: 30528,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 8192,
            type_vocab_size: 2,
            hidden_activation: JinaActivation::Gelu,
            layer_norm_eps: 1e-12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct JinaLayerWeights {
    pub q: WeightStorage,
    pub q_bias: Arc<[f32]>,
    pub k: WeightStorage,
    pub k_bias: Arc<[f32]>,
    pub v: WeightStorage,
    pub v_bias: Arc<[f32]>,
    pub attn_out: WeightStorage,
    pub attn_out_bias: Arc<[f32]>,
    pub attn_ln_gain: Arc<[f32]>,
    pub attn_ln_bias: Arc<[f32]>,
    /// GeGLU fused gate||non-gate `[hidden, 2 * intermediate]` (no bias).
    pub gated_layers: WeightStorage,
    /// `[intermediate, hidden]` MLP down-projection.
    pub mlp_wo: WeightStorage,
    pub mlp_wo_bias: Arc<[f32]>,
    pub mlp_ln_gain: Arc<[f32]>,
    pub mlp_ln_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct JinaBertWeights {
    pub word_embedding: Arc<[f32]>,
    pub token_type_embedding: Arc<[f32]>,
    pub embed_ln_gain: Arc<[f32]>,
    pub embed_ln_bias: Arc<[f32]>,
    pub layers: Vec<JinaLayerWeights>,
}

#[derive(Debug, Clone)]
pub struct JinaBertModel {
    pub config: JinaBertConfig,
    pub weights: JinaBertWeights,
}

/// Compute the ALiBi slope per head, following the eager Fuel
/// (and original ALiBi paper) recipe: next power of 2 of
/// `n_heads`, evenly spaced negative-power-of-2 slopes, then
/// interleave to recover odd head counts.
fn alibi_slopes(n_heads: usize) -> Vec<f32> {
    let mut n2 = 1;
    while n2 < n_heads {
        n2 *= 2;
    }
    let base: Vec<f32> = (1..=n2)
        .map(|v| -1.0_f32 / 2.0_f32.powf((v * 8) as f32 / n2 as f32))
        .collect();
    if n2 == n_heads {
        base
    } else {
        // Take odd indices first (skip(1).step_by(2)), then even indices.
        let odds = base.iter().skip(1).step_by(2).copied();
        let evens = base.iter().step_by(2).copied();
        odds.chain(evens).take(n_heads).collect()
    }
}

/// Precompute the ALiBi additive bias as a row-major
/// `(n_heads * seq * seq)` flat buffer. `bias[h, i, j] = -|i - j| * slope[h]`.
fn build_alibi_bias(n_heads: usize, seq: usize) -> Arc<[f32]> {
    let slopes = alibi_slopes(n_heads);
    let mut out = vec![0.0_f32; n_heads * seq * seq];
    for h in 0..n_heads {
        let s = slopes[h];
        for i in 0..seq {
            for j in 0..seq {
                let d = (i as isize - j as isize).unsigned_abs() as f32;
                out[h * seq * seq + i * seq + j] = s * d;
            }
        }
    }
    Arc::from(out)
}

impl JinaBertModel {
    /// Run a forward pass.
    ///
    /// - `tokens`: input ids, length `seq`.
    /// - `attention_mask`: optional additive pad mask shaped
    ///   `(1, 1, seq, seq)` (broadcast-added on top of ALiBi).
    ///
    /// Returns per-token hidden states `(1, seq, hidden_size)`.
    pub fn forward(
        &self,
        tokens: &[u32],
        attention_mask: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        assert!(seq > 0);
        assert!(seq <= cfg.max_position_embeddings);
        assert!(
            n_heads * cfg.head_dim() == h,
            "num_attention_heads * head_dim must equal hidden_size",
        );

        // ---- Embeddings ----------------------------------------------------
        let word_emb_t = LazyTensor::from_f32(
            weights.word_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, h]),
            &Device::cpu(),
        );
        let token_ids = word_emb_t.const_u32_like(
            tokens.to_vec(),
            Shape::from_dims(&[seq]),
        );
        let word_embeds = word_emb_t
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, h]))?;

        // Token type embeddings (default all-0 ids; row 0 of the table).
        let tte_t = word_emb_t.const_f32_like(
            Arc::clone(&weights.token_type_embedding),
            Shape::from_dims(&[cfg.type_vocab_size, h]),
        );
        let tt_ids = word_emb_t.const_u32_like(vec![0_u32; seq], Shape::from_dims(&[seq]));
        let tt_embeds = tte_t
            .index_select(0_usize, &tt_ids)?
            .reshape(Shape::from_dims(&[batch, seq, h]))?;

        let mut x = crate::lazy::apply_affine_layer_norm_pub(
            &word_embeds.add(&tt_embeds)?,
            &weights.embed_ln_gain,
            &weights.embed_ln_bias,
            h,
            cfg.layer_norm_eps,
        );

        // ---- ALiBi bias (shared across layers) -----------------------------
        let alibi_data = build_alibi_bias(n_heads, seq);
        let alibi_t = x
            .const_f32_like(alibi_data, Shape::from_dims(&[n_heads, seq, seq]))
            .reshape(Shape::from_dims(&[1, n_heads, seq, seq]))?;
        // Optionally fold the pad mask onto ALiBi once, so each
        // layer just broadcast-adds a single bias tensor.
        let bias = match attention_mask {
            None => alibi_t,
            Some(mask) => alibi_t.broadcast_add(mask)?,
        };

        // ---- Encoder blocks ------------------------------------------------
        for layer in &weights.layers {
            x = self.apply_layer(&x, layer, &bias)?;
        }
        Ok(x)
    }

    /// Extract per-token features at the requested layer
    /// indices. Output shape per layer: `(1, seq, hidden_size)`.
    /// Each capture is the post-block hidden state (Jina-BERT
    /// is post-LN throughout, so captures are already fully
    /// normalized).
    ///
    /// Jina-BERT specifics preserved:
    ///   - **ALiBi bias** built once and (when an
    ///     `attention_mask` is supplied) pre-folded with the
    ///     pad mask, so each layer just broadcast-adds a
    ///     single bias tensor — same path the public `forward`
    ///     takes.
    ///   - **Token-type embeddings** at row 0 (default).
    ///   - **Embedding LayerNorm** (with bias) applied before
    ///     the first encoder block.
    ///
    /// Layer-id contract: 0-based, strictly increasing, all in
    /// `[0, num_hidden_layers)`. Mirrors the BERT / DistilBERT
    /// / XLM-R / NomicBert / ModernBERT hooks.
    pub fn forward_intermediate_layers(
        &self,
        tokens: &[u32],
        layer_ids: &[usize],
        attention_mask: Option<&LazyTensor>,
    ) -> Result<Vec<LazyTensor>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        assert!(seq > 0);
        assert!(seq <= cfg.max_position_embeddings);
        assert!(
            n_heads * cfg.head_dim() == h,
            "num_attention_heads * head_dim must equal hidden_size",
        );
        assert!(!layer_ids.is_empty(), "layer_ids must not be empty");
        for w in layer_ids.windows(2) {
            assert!(w[0] < w[1], "layer_ids must be strictly increasing");
        }
        let depth = weights.layers.len();
        assert!(
            *layer_ids.last().unwrap() < depth,
            "layer_ids must all be in [0, num_hidden_layers = {depth})",
        );

        let word_emb_t = LazyTensor::from_f32(
            weights.word_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, h]),
            &Device::cpu(),
        );
        let token_ids = word_emb_t.const_u32_like(
            tokens.to_vec(), Shape::from_dims(&[seq]),
        );
        let word_embeds = word_emb_t
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, h]))?;
        let tte_t = word_emb_t.const_f32_like(
            Arc::clone(&weights.token_type_embedding),
            Shape::from_dims(&[cfg.type_vocab_size, h]),
        );
        let tt_ids = word_emb_t.const_u32_like(
            vec![0_u32; seq], Shape::from_dims(&[seq]),
        );
        let tt_embeds = tte_t
            .index_select(0_usize, &tt_ids)?
            .reshape(Shape::from_dims(&[batch, seq, h]))?;
        let mut x = crate::lazy::apply_affine_layer_norm_pub(
            &word_embeds.add(&tt_embeds)?,
            &weights.embed_ln_gain, &weights.embed_ln_bias,
            h, cfg.layer_norm_eps,
        );

        // Shared ALiBi bias (optionally folded with pad mask).
        let alibi_data = build_alibi_bias(n_heads, seq);
        let alibi_t = x
            .const_f32_like(alibi_data, Shape::from_dims(&[n_heads, seq, seq]))
            .reshape(Shape::from_dims(&[1, n_heads, seq, seq]))?;
        let bias = match attention_mask {
            None => alibi_t,
            Some(mask) => alibi_t.broadcast_add(mask)?,
        };

        let mut out = Vec::with_capacity(layer_ids.len());
        let mut next_capture = 0;
        for (idx, layer) in weights.layers.iter().enumerate() {
            x = self.apply_layer(&x, layer, &bias)?;
            if next_capture < layer_ids.len() && layer_ids[next_capture] == idx {
                out.push(x.clone());
                next_capture += 1;
            }
        }
        Ok(out)
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &JinaLayerWeights,
        bias: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = x.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();

        // ---- Self-attention (separate Q/K/V) ------------------------------
        let q = layer.q.apply_linear(x, h, h);
        let q = bias_add(q, &layer.q_bias, h, x)?;
        let k = layer.k.apply_linear(x, h, h);
        let k = bias_add(k, &layer.k_bias, h, x)?;
        let v = layer.v.apply_linear(x, h, h);
        let v = bias_add(v, &layer.v_bias, h, x)?;

        let q = q
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let k = k
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let v = v
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose()?)?.mul_scalar(scale);
        let scores = scores.broadcast_add(bias)?;
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx
            .permute([0, 2, 1, 3_usize])?
            .reshape(Shape::from_dims(&[batch, seq, h]))?;
        let attn_out = layer.attn_out.apply_linear(&merged, h, h);
        let attn_out = bias_add(attn_out, &layer.attn_out_bias, h, x)?;

        // Post-LN attention residual: LN(x + attn).
        let h1 = crate::lazy::apply_affine_layer_norm_pub(
            &x.add(&attn_out)?,
            &layer.attn_ln_gain, &layer.attn_ln_bias,
            h, cfg.layer_norm_eps,
        );

        // ---- GeGLU MLP -----------------------------------------------------
        let i = cfg.intermediate_size;
        let up = layer.gated_layers.apply_linear(&h1, h, 2 * i);
        let gate = up.slice(2_usize, 0, i)?;
        let value = up.slice(2_usize, i, i)?;
        let gated = match cfg.hidden_activation {
            JinaActivation::Gelu => gate.gelu_erf(),
            JinaActivation::GeluPytorchTanh => gate.gelu(),
            JinaActivation::Relu => gate.relu(),
            JinaActivation::Silu => gate.silu(),
        };
        let inner = gated.mul(&value)?;
        let down = layer.mlp_wo.apply_linear(&inner, i, h);
        let down = bias_add(down, &layer.mlp_wo_bias, h, x)?;

        // Post-LN MLP residual: LN(h1 + mlp).
        Ok(crate::lazy::apply_affine_layer_norm_pub(
            &h1.add(&down)?,
            &layer.mlp_ln_gain, &layer.mlp_ln_bias,
            h, cfg.layer_norm_eps,
        ))
    }
}

fn bias_add(
    x: LazyTensor,
    bias: &Arc<[f32]>,
    n: usize,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    assert_eq!(bias.len(), n);
    let bt = anchor.const_f32_like(Arc::clone(bias), Shape::from_dims(&[n]));
    x.broadcast_add(&bt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_cfg() -> JinaBertConfig {
        JinaBertConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 24,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            hidden_activation: JinaActivation::Gelu,
            layer_norm_eps: 1e-12,
        }
    }

    fn tiny_weights(cfg: &JinaBertConfig) -> JinaBertWeights {
        let mut s: u32 = 86420;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let word_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let token_type_embedding = vec_of(cfg.type_vocab_size * h, &mut *nb);
        let embed_ln_gain = Arc::from(vec![1.0_f32; h]);
        let embed_ln_bias = Arc::from(vec![0.0_f32; h]);

        let layers: Vec<JinaLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| JinaLayerWeights {
                q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                q_bias: vec_of(h, &mut *nb),
                k: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                k_bias: vec_of(h, &mut *nb),
                v: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                v_bias: vec_of(h, &mut *nb),
                attn_out: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                attn_out_bias: vec_of(h, &mut *nb),
                attn_ln_gain: Arc::from(vec![1.0_f32; h]),
                attn_ln_bias: Arc::from(vec![0.0_f32; h]),
                gated_layers: WeightStorage::F32(vec_of(h * 2 * i, &mut *nb)),
                mlp_wo: WeightStorage::F32(vec_of(i * h, &mut *nb)),
                mlp_wo_bias: vec_of(h, &mut *nb),
                mlp_ln_gain: Arc::from(vec![1.0_f32; h]),
                mlp_ln_bias: Arc::from(vec![0.0_f32; h]),
            })
            .collect();
        JinaBertWeights {
            word_embedding,
            token_type_embedding,
            embed_ln_gain,
            embed_ln_bias,
            layers,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_cfg();
        let model = JinaBertModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens = [1_u32, 2, 3, 4, 5, 6];
        let out = model.forward(&tokens, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    /// Bidirectional attention with ALiBi bias.
    #[test]
    fn bidirectional_attention() {
        let cfg = tiny_cfg();
        let model = JinaBertModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let toks_a = [1_u32, 2, 3, 4];
        let toks_b = [1_u32, 2, 3, 15];
        let a = model.forward(&toks_a, None).unwrap().realize_f32();
        let b = model.forward(&toks_b, None).unwrap().realize_f32();
        let h = cfg.hidden_size;
        let mut max_diff = 0.0_f32;
        for i in 0..h {
            max_diff = max_diff.max((a[i] - b[i]).abs());
        }
        assert!(max_diff > 1e-7,
            "last-token change must affect position 0 (bidirectional), max_diff = {max_diff}");
    }

    /// ALiBi slope table for n_heads = power of 2 follows the
    /// closed-form `-1 / 2^(8v/n)` recipe.
    #[test]
    fn alibi_slopes_pow2() {
        let s = alibi_slopes(8);
        // Negative, monotonically decreasing in magnitude (head 0
        // has the strongest negative slope).
        for &v in &s {
            assert!(v < 0.0, "slopes must be negative, got {v}");
        }
        for w in s.windows(2) {
            assert!(w[0].abs() > w[1].abs(),
                "slope magnitudes must shrink across heads: {} → {}",
                w[0], w[1]);
        }
        // s[0] = -1/2^1 = -0.5 exactly.
        assert!((s[0] + 0.5).abs() < 1e-7, "head-0 slope expected -0.5, got {}", s[0]);
    }

    /// ALiBi slope table for non-power-of-2 head counts uses
    /// the interleaved fallback. n_heads = 12 has the
    /// next-power-of-2 = 16; we just verify the right count
    /// and that all slopes are negative.
    #[test]
    fn alibi_slopes_non_pow2() {
        let s = alibi_slopes(12);
        assert_eq!(s.len(), 12);
        for &v in &s {
            assert!(v < 0.0, "slopes must be negative, got {v}");
        }
    }

    /// ALiBi bias affects output: with one distinct token at
    /// position 5 (the rest identical), position 0 (distance 5
    /// from the distinct token) and position 4 (distance 1) get
    /// different ALiBi-weighted attention to that token, so
    /// their outputs must differ. If ALiBi were not wired, the
    /// inverse-bag-of-words symmetry between positions 0..4
    /// would make their outputs identical.
    #[test]
    fn alibi_distinguishes_positions() {
        let cfg = tiny_cfg();
        let model = JinaBertModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let toks = [5_u32, 5, 5, 5, 5, 30];
        let out = model.forward(&toks, None).unwrap().realize_f32();
        let h = cfg.hidden_size;
        let mut max_diff = 0.0_f32;
        // Position 0 (ALiBi distance 5 from the distinct token at idx 5)
        // vs position 4 (ALiBi distance 1).
        for j in 0..h {
            let p0 = out[j];
            let p4 = out[4 * h + j];
            max_diff = max_diff.max((p0 - p4).abs());
        }
        assert!(max_diff > 1e-7,
            "ALiBi should weigh the distinct token differently at distance 5 \
             vs distance 1, but position 0 and 4 outputs are identical: max_diff = {max_diff}");
    }

    /// `forward_intermediate_layers` returns one tensor per
    /// requested layer index, each shaped `(1, seq, hidden_size)`.
    /// Mirrors the BERT-family hooks.
    #[test]
    fn forward_intermediate_layers_shape() {
        let cfg = tiny_cfg();
        let model = JinaBertModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let toks = [1_u32, 2, 3, 4];
        let outs = model.forward_intermediate_layers(&toks, &[0_usize, 1], None).unwrap();
        assert_eq!(outs.len(), 2);
        for out in &outs {
            assert_eq!(out.shape().dims(), &[1, toks.len(), cfg.hidden_size]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite intermediate: {v}");
            }
        }
        let a = outs[0].realize_f32();
        let b = outs[1].realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "layer 0 and layer 1 intermediates must differ, max_diff = {max_diff}");
    }
}

//! DeBERTa-v2 / v3 — lazy port.
//!
//! Encoder-only Transformer with **disentangled attention**: in
//! addition to standard content-to-content (Q·K) scores, the
//! attention also incorporates a content-to-position (c2p) term
//! `Q · pos_key^T` gathered by a log-bucketed relative-position
//! table, and a position-to-content (p2c) term `pos_query · K^T`
//! gathered by the (sign-flipped) same table.
//!
//! v1 scope targets the canonical DeBERTa-v3 setup:
//!   - `share_att_key = true` (Q/K projections reused on rel_embeddings)
//!   - `pos_att_type = ["p2c", "c2p"]` (both bias terms enabled)
//!   - `position_buckets = 256`
//!   - `norm_rel_ebd = "layer_norm"` (LN on rel_embeddings)
//!   - `position_biased_input = false` (no absolute pos-embed in inputs)
//!   - F32, batch == 1, prefill only, no token-type / conv-layer.
//!
//! These are the settings for `microsoft/deberta-v3-base`,
//! `deberta-v3-large`, `deberta-v3-small`, and `mdeberta-v3-base`.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct DebertaV2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    /// Bucket count for the log-bucket relative position table.
    pub position_buckets: usize,
    /// Clamp on |relative_pos| before bucketing.
    pub max_relative_positions: usize,
}

impl DebertaV2Config {
    /// `microsoft/deberta-v3-base`.
    pub fn v3_base() -> Self {
        Self {
            vocab_size: 128_100,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-7,
            position_buckets: 256,
            max_relative_positions: 512,
        }
    }
    /// `microsoft/deberta-v3-large`.
    pub fn v3_large() -> Self {
        Self {
            vocab_size: 128_100,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-7,
            position_buckets: 256,
            max_relative_positions: 512,
        }
    }
    /// `microsoft/deberta-v3-small`.
    pub fn v3_small() -> Self {
        Self {
            vocab_size: 128_100,
            hidden_size: 768,
            num_hidden_layers: 6,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-7,
            position_buckets: 256,
            max_relative_positions: 512,
        }
    }
    pub fn head_dim(&self) -> usize { self.hidden_size / self.num_attention_heads }
}

// ---- Weight structures ------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    pub gain: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct LinearWeights {
    pub w: WeightStorage,
    pub b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct DebertaV2AttentionWeights {
    pub query_proj: LinearWeights,
    pub key_proj: LinearWeights,
    pub value_proj: LinearWeights,
    /// Post-attention dense + LN (BERT-style Post-LN).
    pub out_dense: LinearWeights,
    pub out_ln: LayerNormWeights,
}

#[derive(Debug, Clone)]
pub struct DebertaV2FfnWeights {
    pub intermediate: LinearWeights,
    pub output: LinearWeights,
    pub output_ln: LayerNormWeights,
}

#[derive(Debug, Clone)]
pub struct DebertaV2LayerWeights {
    pub attn: DebertaV2AttentionWeights,
    pub ffn: DebertaV2FfnWeights,
}

#[derive(Debug, Clone)]
pub struct DebertaV2Weights {
    /// `(vocab_size, hidden_size)`.
    pub word_embedding: Arc<[f32]>,
    pub embed_ln: LayerNormWeights,
    /// `(2 * position_buckets, hidden_size)`. Indexed by bucket id.
    pub rel_embeddings: Arc<[f32]>,
    /// Optional LayerNorm on `rel_embeddings` (norm_rel_ebd == "layer_norm").
    pub rel_emb_ln: Option<LayerNormWeights>,
    pub layers: Vec<DebertaV2LayerWeights>,
}

#[derive(Debug, Clone)]
pub struct DebertaV2Model {
    pub config: DebertaV2Config,
    pub weights: DebertaV2Weights,
}

// ---- Relative-position bucketing (precomputed at forward time) -------------

/// Build the c2p bucket-index table.
///
/// Returns a flat `seq_len * seq_len` `Vec<u32>` of values in
/// `[0, 2 * position_buckets)` where entry `(q, k)` is the bucket
/// index for `k - q`, *plus* `position_buckets` (the `att_span`
/// offset applied by the eager `c2p_pos` computation).
pub fn build_c2p_indices(
    seq_len: usize, position_buckets: usize, max_relative_positions: usize,
) -> Vec<u32> {
    let bucket_size = position_buckets as isize;
    let max_position = max_relative_positions as isize;
    let mid = bucket_size / 2;
    let att_span = bucket_size;
    let clamp_high = (att_span * 2 - 1) as u32;
    let mut out = Vec::with_capacity(seq_len * seq_len);
    for q in 0..seq_len {
        for k in 0..seq_len {
            let rel = (k as isize) - (q as isize); // sign matches eager `k_ids - q_ids`
            let bucket = log_bucket(rel, bucket_size, max_position);
            // c2p_pos = bucket + att_span, clamped to [0, 2*att_span-1].
            let v = (bucket + att_span as i64).max(0);
            let v = (v as u32).min(clamp_high);
            out.push(v);
        }
    }
    let _ = mid;
    out
}

/// Build the p2c bucket-index table.
///
/// p2c uses `-(k - q)` (sign-flipped relative_pos) plus att_span.
/// Square `(seq_len, seq_len)` indices indexed by `(k, q)` →
/// transposed naturally by the matmul layout downstream.
pub fn build_p2c_indices(
    seq_len: usize, position_buckets: usize, max_relative_positions: usize,
) -> Vec<u32> {
    let bucket_size = position_buckets as isize;
    let max_position = max_relative_positions as isize;
    let att_span = bucket_size;
    let clamp_high = (att_span * 2 - 1) as u32;
    let mut out = Vec::with_capacity(seq_len * seq_len);
    // p2c table is indexed identically to c2p but with the relative-pos
    // sign flipped: eager does `-r_pos + att_span`.
    for q in 0..seq_len {
        for k in 0..seq_len {
            let rel = (k as isize) - (q as isize);
            let bucket = log_bucket(rel, bucket_size, max_position);
            // p2c_pos = -bucket + att_span, clamped to [0, 2*att_span-1].
            let v = (-bucket + att_span as i64).max(0);
            let v = (v as u32).min(clamp_high);
            out.push(v);
        }
    }
    out
}

fn log_bucket(rel: isize, bucket_size: isize, max_position: isize) -> i64 {
    let sign: isize = if rel > 0 { 1 } else if rel < 0 { -1 } else { 0 };
    let mid = bucket_size / 2;
    let abs = if rel.unsigned_abs() < mid as usize {
        (mid - 1) as f64
    } else {
        rel.unsigned_abs() as f64
    };
    if abs <= mid as f64 {
        rel as i64
    } else {
        let first = (abs / mid as f64).ln();
        let second = ((max_position as f64 - 1.0) / mid as f64).ln();
        let ratio = first / second;
        let v = (ratio * (mid - 1) as f64).ceil() + mid as f64;
        (v as i64) * sign as i64
    }
}

// ---- Forward ---------------------------------------------------------------

impl DebertaV2Model {
    /// Run prefill and return final hidden states `(1, T, hidden)`.
    pub fn forward(&self, input_ids: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let w = &self.weights;
        let t = input_ids.len();
        assert!(t > 0);
        let h = cfg.hidden_size;
        let anchor_ids: Vec<u32> = input_ids.to_vec();
        let ids = LazyTensor::from_u32(
            anchor_ids,
            Shape::from_dims(&[t]),
            &crate::Device::cpu(),
        );

        // Word embedding lookup + LN.
        let table = ids.const_f32_like(
            Arc::clone(&w.word_embedding),
            Shape::from_dims(&[cfg.vocab_size, h]),
        );
        let x = table
            .index_select(0_usize, &ids)?
            .reshape(Shape::from_dims(&[1, t, h]))?;
        let mut x = apply_layer_norm(&x, &w.embed_ln, h, cfg.layer_norm_eps)?;

        // Build / normalize rel_embeddings once per forward.
        let rel_table = ids.const_f32_like(
            Arc::clone(&w.rel_embeddings),
            Shape::from_dims(&[2 * cfg.position_buckets, h]),
        );
        let rel_table = match &w.rel_emb_ln {
            None => rel_table,
            Some(ln) => apply_layer_norm(
                &rel_table.reshape(Shape::from_dims(&[1, 2 * cfg.position_buckets, h]))?,
                ln, h, cfg.layer_norm_eps,
            )?.reshape(Shape::from_dims(&[2 * cfg.position_buckets, h]))?,
        };

        // Build c2p / p2c gather index tables (depend on T).
        let c2p_idx = build_c2p_indices(
            t, cfg.position_buckets, cfg.max_relative_positions,
        );
        let p2c_idx = build_p2c_indices(
            t, cfg.position_buckets, cfg.max_relative_positions,
        );
        let c2p_idx = ids.const_u32_like(
            c2p_idx, Shape::from_dims(&[1, t, t]),
        );
        let p2c_idx = ids.const_u32_like(
            p2c_idx, Shape::from_dims(&[1, t, t]),
        );

        for layer in &w.layers {
            x = apply_layer(&x, layer, &rel_table, &c2p_idx, &p2c_idx, cfg, &ids)?;
        }
        Ok(x)
    }
}

fn apply_layer(
    x: &LazyTensor,
    w: &DebertaV2LayerWeights,
    rel_table: &LazyTensor,
    c2p_idx: &LazyTensor,
    p2c_idx: &LazyTensor,
    cfg: &DebertaV2Config,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let attn_out = apply_attention(x, &w.attn, rel_table, c2p_idx, p2c_idx, cfg, anchor)?;
    // Post-LN inside the attention sublayer.
    let projected = apply_linear(&attn_out, &w.attn.out_dense, anchor)?;
    let x = apply_layer_norm(&projected.add(x)?, &w.attn.out_ln, cfg.hidden_size, cfg.layer_norm_eps)?;

    // FFN.
    let inter = apply_linear(&x, &w.ffn.intermediate, anchor)?;
    let inter = inter.gelu();
    let out = apply_linear(&inter, &w.ffn.output, anchor)?;
    apply_layer_norm(&out.add(&x)?, &w.ffn.output_ln, cfg.hidden_size, cfg.layer_norm_eps)
}

fn apply_attention(
    x: &LazyTensor,
    w: &DebertaV2AttentionWeights,
    rel_table: &LazyTensor,
    c2p_idx: &LazyTensor,
    p2c_idx: &LazyTensor,
    cfg: &DebertaV2Config,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let t = dims[1]; let h = dims[2];
    let heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim();

    let q = apply_linear(x, &w.query_proj, anchor)?;
    let k = apply_linear(x, &w.key_proj, anchor)?;
    let v = apply_linear(x, &w.value_proj, anchor)?;

    // (B, T, H) → (B, heads, T, head_dim) → flatten to (B*heads, T, head_dim).
    let q = q.reshape(Shape::from_dims(&[b, t, heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?
        .reshape(Shape::from_dims(&[b * heads, t, head_dim]))?;
    let k = k.reshape(Shape::from_dims(&[b, t, heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?
        .reshape(Shape::from_dims(&[b * heads, t, head_dim]))?;
    let v = v.reshape(Shape::from_dims(&[b, t, heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?
        .reshape(Shape::from_dims(&[b * heads, t, head_dim]))?;

    // Disentangled-attn scale factor: scale_factor = 1 (content) + 1 (c2p) + 1 (p2c) = 3.
    let scale_factor = 3.0_f64;
    let scale = 1.0_f64 / ((head_dim as f64) * scale_factor).sqrt();

    // Content-to-content scores: (b*heads, T, T)
    let kt = k.permute([0, 2, 1_usize])?;
    let scores = q.matmul(&kt)?.mul_scalar(scale);

    // share_att_key: project rel_table through Q and K linears.
    let rel_q = apply_linear(rel_table, &w.query_proj, anchor)?
        .reshape(Shape::from_dims(&[2 * cfg.position_buckets, heads, head_dim]))?
        .permute([1, 0, 2_usize])?;
    let rel_k = apply_linear(rel_table, &w.key_proj, anchor)?
        .reshape(Shape::from_dims(&[2 * cfg.position_buckets, heads, head_dim]))?
        .permute([1, 0, 2_usize])?;
    // Tile across the B dim (b == 1 in v1 so this is just heads).
    debug_assert_eq!(b, 1);
    let rel_q = rel_q
        .reshape(Shape::from_dims(&[b * heads, 2 * cfg.position_buckets, head_dim]))?;
    let rel_k = rel_k
        .reshape(Shape::from_dims(&[b * heads, 2 * cfg.position_buckets, head_dim]))?;

    // c2p: Q · rel_k^T → (b*heads, T, 2*att_span), then gather along last dim by c2p_idx.
    let rel_k_t = rel_k.permute([0, 2, 1_usize])?;
    let c2p_att = q.matmul(&rel_k_t)?.mul_scalar(scale);
    let c2p_idx_b = c2p_idx
        .broadcast_to(Shape::from_dims(&[b * heads, t, t]))?;
    let c2p_term = c2p_att.gather(2_usize, &c2p_idx_b)?;

    // p2c: rel_q · K^T → (b*heads, 2*att_span, T). Gather along the
    // "rel_q" dim (dim 1) using p2c indices, then transpose the (q, k)
    // pair so it matches scores' (T, T) layout.
    //
    // Eager does k.matmul(pos_query^T).gather(-1, p2c_pos).t() — same as
    // (rel_q · k^T).gather(dim=1, p2c_indices).
    let rel_q_kt = rel_q.matmul(&kt)?.mul_scalar(scale);
    // rel_q_kt shape: (b*heads, 2*att_span, T). Gather along dim=1 needs
    // an index tensor of shape (b*heads, T, T) where index[bh, q, k] picks
    // a row in dim 1 (the 2*att_span axis). The eager code's p2c table
    // is (q, k) → bucket id (k indexes the key/query position pair) — for
    // a square self-attention this is just the transposed c2p table.
    let p2c_idx_b = p2c_idx
        .broadcast_to(Shape::from_dims(&[b * heads, t, t]))?;
    // gather over dim 1 returns (bh, T, T) where output[bh, q, k] =
    // rel_q_kt[bh, p2c_idx[bh, q, k], k]. We want output[bh, q, k] =
    // rel_q_kt[bh, p2c_idx[bh, k, q], q] in eager terms (the .t() at the
    // end of the eager p2c path). Pre-transpose the index table.
    let p2c_idx_t = p2c_idx_b.permute([0, 2, 1_usize])?;
    // Index tensor must address dim 1 of rel_q_kt; we also need to remap
    // the "k" axis: rel_q_kt's last dim is T (the K tokens), same as the
    // attention's k axis. After gather, dim 1 becomes "scattered" by
    // p2c_idx_t which is shaped (bh, T, T). Result: (bh, T_q, T_k).
    let p2c_term = rel_q_kt.gather(1_usize, &p2c_idx_t)?;
    // Match eager `.t()`: transpose the gathered (q, k) → (k, q), then
    // since q==k for self-attention, the transpose corresponds to
    // swapping the last two dims of the gathered result.
    let p2c_term = p2c_term.permute([0, 2, 1_usize])?;

    let scores = scores.add(&c2p_term)?.add(&p2c_term)?;
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?;
    // (b*heads, T, head_dim) → (b, heads, T, head_dim) → (b, T, H)
    let ctx = ctx
        .reshape(Shape::from_dims(&[b, heads, t, head_dim]))?
        .permute([0, 2, 1, 3_usize])?
        .reshape(Shape::from_dims(&[b, t, h]))?;
    Ok(ctx)
}

fn apply_layer_norm(
    x: &LazyTensor, ln: &LayerNormWeights, hidden: usize, eps: f64,
) -> Result<LazyTensor> {
    let _ = hidden;
    x.layer_norm_affine(Arc::clone(&ln.gain), Arc::clone(&ln.bias), eps)
}

fn apply_linear(
    x: &LazyTensor, lw: &LinearWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let _ = anchor;
    let dims = x.shape();
    let dims = dims.dims();
    let in_features = dims[dims.len() - 1];
    let out_features = lw.b.len();
    lw.w.apply_linear_with_bias(x, in_features, out_features, Arc::clone(&lw.b))
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
    fn linear_w(
        in_features: usize, out_features: usize, nb: &mut dyn FnMut() -> f32,
    ) -> LinearWeights {
        LinearWeights { w: ws(in_features * out_features, nb), b: vec_of(out_features, nb) }
    }

    fn tiny_config() -> DebertaV2Config {
        DebertaV2Config {
            vocab_size: 64,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 32,
            max_position_embeddings: 32,
            layer_norm_eps: 1e-7,
            position_buckets: 8,
            max_relative_positions: 16,
        }
    }

    fn build_weights(cfg: &DebertaV2Config) -> DebertaV2Weights {
        let mut nb = rng_seed(2026);
        let h = cfg.hidden_size;
        let layers: Vec<DebertaV2LayerWeights> = (0..cfg.num_hidden_layers).map(|_| {
            DebertaV2LayerWeights {
                attn: DebertaV2AttentionWeights {
                    query_proj: linear_w(h, h, &mut nb),
                    key_proj: linear_w(h, h, &mut nb),
                    value_proj: linear_w(h, h, &mut nb),
                    out_dense: linear_w(h, h, &mut nb),
                    out_ln: ln_w(h),
                },
                ffn: DebertaV2FfnWeights {
                    intermediate: linear_w(h, cfg.intermediate_size, &mut nb),
                    output: linear_w(cfg.intermediate_size, h, &mut nb),
                    output_ln: ln_w(h),
                },
            }
        }).collect();
        DebertaV2Weights {
            word_embedding: vec_of(cfg.vocab_size * h, &mut nb),
            embed_ln: ln_w(h),
            rel_embeddings: vec_of(2 * cfg.position_buckets * h, &mut nb),
            rel_emb_ln: Some(ln_w(h)),
            layers,
        }
    }

    #[test]
    fn c2p_indices_are_within_bounds() {
        let idx = build_c2p_indices(8, 8, 16);
        assert_eq!(idx.len(), 8 * 8);
        for &v in &idx {
            assert!((v as usize) < 2 * 8, "c2p idx out of range: {v}");
        }
    }

    #[test]
    fn p2c_indices_are_within_bounds() {
        let idx = build_p2c_indices(8, 8, 16);
        assert_eq!(idx.len(), 8 * 8);
        for &v in &idx {
            assert!((v as usize) < 2 * 8, "p2c idx out of range: {v}");
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let weights = build_weights(&cfg);
        let model = DebertaV2Model { config: cfg.clone(), weights };
        let ids = vec![1_u32, 2, 3, 4, 5, 6, 7, 8];
        let out = model.forward(&ids).unwrap();
        assert_eq!(out.shape().dims(), &[1, ids.len(), cfg.hidden_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    #[test]
    fn forward_responds_to_input() {
        let cfg = tiny_config();
        let weights = build_weights(&cfg);
        let model = DebertaV2Model { config: cfg, weights };
        let a = model.forward(&[1_u32, 2, 3, 4, 5, 6, 7, 8]).unwrap().realize_f32();
        let b = model.forward(&[1_u32, 2, 3, 4, 5, 6, 7, 9]).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "DeBERTa-v2 must respond to input changes, max_diff = {max_diff}");
    }

    #[test]
    fn presets_construct() {
        let base = DebertaV2Config::v3_base();
        assert_eq!(base.position_buckets, 256);
        let large = DebertaV2Config::v3_large();
        assert_eq!(large.hidden_size, 1024);
        let small = DebertaV2Config::v3_small();
        assert_eq!(small.num_hidden_layers, 6);
    }
}

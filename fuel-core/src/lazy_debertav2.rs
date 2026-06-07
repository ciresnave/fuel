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
use crate::{DType, Result};
use fuel_core_types::Shape;
use std::collections::HashMap;
use std::sync::Arc;

/// Activation / weight dtype used by the lazy DeBERTa-v2 port. The
/// binary references this constant to pick a tokenizer/dtype-aware
/// loader path, matching the eager port that fixed it to `f32`.
pub const DTYPE: DType = DType::F32;

/// HuggingFace-style `id2label` map. DeBERTa fine-tunes for token-
/// or sequence-classification ship one of these in their `config.json`
/// (string keys like `"0"`, `"1"`, … in the JSON; deserialized to a
/// `u32 → String` map here).
pub type Id2Label = HashMap<u32, String>;

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
    /// Optional `id2label` map (present on most fine-tuned task
    /// checkpoints; absent on raw encoder bases).
    pub id2label: Option<Id2Label>,
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
            id2label: None,
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
            id2label: None,
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
            id2label: None,
        }
    }
    pub fn head_dim(&self) -> usize { self.hidden_size / self.num_attention_heads }

    /// Parse a HuggingFace `config.json` string. Strict on the fields
    /// the lazy port needs (`vocab_size`, `hidden_size`, …) but tolerant
    /// to the long tail of dropout / initializer / activation knobs the
    /// eager port consumed (this lazy v1 hardcodes them).
    ///
    /// Optional fields:
    /// - `position_buckets` (defaults to 256, matching v3-base/large)
    /// - `max_relative_positions` (defaults to `max_position_embeddings`)
    /// - `layer_norm_eps` (defaults to 1e-7)
    /// - `id2label` (string-keyed JSON object → `u32 → String`)
    pub fn from_hf_json_str(s: &str) -> crate::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| crate::Error::Msg(format!("parsing deberta-v2 config.json: {e}")).bt())?;

        let get_usize = |key: &str| -> crate::Result<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| crate::Error::Msg(format!(
                    "deberta-v2 config.json: missing/invalid field {key:?}",
                )).bt())
        };

        let vocab_size = get_usize("vocab_size")?;
        let hidden_size = get_usize("hidden_size")?;
        let num_hidden_layers = get_usize("num_hidden_layers")?;
        let num_attention_heads = get_usize("num_attention_heads")?;
        let intermediate_size = get_usize("intermediate_size")?;
        let max_position_embeddings = get_usize("max_position_embeddings")?;

        let layer_norm_eps = v.get("layer_norm_eps")
            .and_then(|x| x.as_f64()).unwrap_or(1e-7);
        let position_buckets = v.get("position_buckets")
            .and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(256);
        // HF ships -1 to mean "use max_position_embeddings"; accept that.
        let max_relative_positions = match v.get("max_relative_positions").and_then(|x| x.as_i64()) {
            Some(n) if n >= 1 => n as usize,
            _ => max_position_embeddings,
        };

        let id2label = parse_id2label(v.get("id2label"));

        Ok(Self {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            layer_norm_eps,
            position_buckets,
            max_relative_positions,
            id2label,
        })
    }
}

fn parse_id2label(v: Option<&serde_json::Value>) -> Option<Id2Label> {
    let obj = v?.as_object()?;
    let mut out = HashMap::new();
    for (k, val) in obj {
        let id = k.parse::<u32>().ok()?;
        let label = val.as_str()?.to_string();
        out.insert(id, label);
    }
    Some(out)
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
        let mut x = x.layer_norm_affine(Arc::clone(&w.embed_ln.gain), Arc::clone(&w.embed_ln.bias), cfg.layer_norm_eps)?;

        // Build / normalize rel_embeddings once per forward.
        let rel_table = ids.const_f32_like(
            Arc::clone(&w.rel_embeddings),
            Shape::from_dims(&[2 * cfg.position_buckets, h]),
        );
        let rel_table = match &w.rel_emb_ln {
            None => rel_table,
            Some(ln) => rel_table.reshape(Shape::from_dims(&[1, 2 * cfg.position_buckets, h]))?.layer_norm_affine(Arc::clone(&ln.gain), Arc::clone(&ln.bias), cfg.layer_norm_eps)?.reshape(Shape::from_dims(&[2 * cfg.position_buckets, h]))?,
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
    let x = &projected.add(x)?.layer_norm_affine(Arc::clone(&w.attn.out_ln.gain), Arc::clone(&w.attn.out_ln.bias), cfg.layer_norm_eps)?;

    // FFN.
    let inter = apply_linear(&x, &w.ffn.intermediate, anchor)?;
    let inter = inter.gelu();
    let out = apply_linear(&inter, &w.ffn.output, anchor)?;
    out.add(&x)?.layer_norm_affine(Arc::clone(&w.ffn.output_ln.gain), Arc::clone(&w.ffn.output_ln.bias), cfg.layer_norm_eps)
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

// ---- HuggingFace safetensors loader ----------------------------------------

impl DebertaV2Weights {
    /// Load DeBERTa-v2/v3 (microsoft/deberta-v3-*) weights from HF safetensors.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &DebertaV2Config,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype as ltm};
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;

        let prefix = if load_tensor_as_f32(st, "deberta.embeddings.word_embeddings.weight").is_ok() {
            "deberta."
        } else { "" };

        let word_embedding = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}embeddings.word_embeddings.weight"),
        )?);
        let embed_ln = LayerNormWeights {
            gain: Arc::from(load_tensor_as_f32(
                st, &format!("{prefix}embeddings.LayerNorm.weight"),
            )?),
            bias: Arc::from(load_tensor_as_f32(
                st, &format!("{prefix}embeddings.LayerNorm.bias"),
            )?),
        };
        let rel_embeddings = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}encoder.rel_embeddings.weight"),
        )?);
        let rel_emb_ln = load_tensor_as_f32(
            st, &format!("{prefix}encoder.LayerNorm.weight"),
        ).ok().map(|gain| LayerNormWeights {
            gain: Arc::from(gain),
            bias: Arc::from(load_tensor_as_f32(
                st, &format!("{prefix}encoder.LayerNorm.bias"),
            ).unwrap_or_else(|_| vec![0.0_f32; h])),
        });

        let load_lin = |p: &str, in_f: usize, out_f: usize| -> Result<LinearWeights> {
            let w = ltm(st, &format!("{p}.weight"), out_f, in_f)?;
            let b = Arc::from(load_tensor_as_f32(st, &format!("{p}.bias"))?);
            Ok(LinearWeights { w, b })
        };
        let load_ln = |p: &str| -> Result<LayerNormWeights> {
            Ok(LayerNormWeights {
                gain: Arc::from(load_tensor_as_f32(st, &format!("{p}.weight"))?),
                bias: Arc::from(load_tensor_as_f32(st, &format!("{p}.bias"))?),
            })
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("{prefix}encoder.layer.{i}");
            let attn = DebertaV2AttentionWeights {
                query_proj: load_lin(&format!("{p}.attention.self.query_proj"), h, h)?,
                key_proj: load_lin(&format!("{p}.attention.self.key_proj"), h, h)?,
                value_proj: load_lin(&format!("{p}.attention.self.value_proj"), h, h)?,
                out_dense: load_lin(&format!("{p}.attention.output.dense"), h, h)?,
                out_ln: load_ln(&format!("{p}.attention.output.LayerNorm"))?,
            };
            let ffn = DebertaV2FfnWeights {
                intermediate: load_lin(&format!("{p}.intermediate.dense"), h, inter)?,
                output: load_lin(&format!("{p}.output.dense"), inter, h)?,
                output_ln: load_ln(&format!("{p}.output.LayerNorm"))?,
            };
            layers.push(DebertaV2LayerWeights { attn, ffn });
        }

        Ok(Self {
            word_embedding, embed_ln, rel_embeddings, rel_emb_ln, layers,
        })
    }
}

// ---- Task heads ------------------------------------------------------------

/// Per-token NER prediction. Mirrors the eager port's `NERItem`
/// (the binary's `--task=ner` path returns `Vec<Vec<NERItem>>`,
/// one inner Vec per input sentence).
#[derive(Debug, Clone, PartialEq)]
pub struct NERItem {
    /// Predicted entity label (looked up via `Id2Label`).
    pub entity: String,
    /// Softmax probability of the chosen label.
    pub score: f32,
    /// Surface form of the corresponding sub-word token.
    pub word: String,
    /// Token index inside the input sequence (after CLS/SEP framing).
    pub index: usize,
}

/// Per-sequence classification prediction. Mirrors the eager port's
/// `TextClassificationItem` (the binary's `--task=text-classification`
/// path returns `Vec<TextClassificationItem>`, one per input sentence).
#[derive(Debug, Clone, PartialEq)]
pub struct TextClassificationItem {
    pub label: String,
    pub score: f32,
}

// ---- DebertaV2NERModel -----------------------------------------------------

/// Weights for the NER (token-classification) head: a single linear
/// `classifier` layer applied to every token's encoder hidden state.
#[derive(Debug, Clone)]
pub struct DebertaV2NERWeights {
    pub encoder: DebertaV2Weights,
    /// `[hidden_size, num_labels]` after the load-time transpose.
    pub classifier: LinearWeights,
}

/// Token-classification model. Wraps the encoder and a per-token
/// linear classifier — the canonical `AutoModelForTokenClassification`
/// shape on the HuggingFace side.
///
/// Forward returns logits of shape `(1, seq, num_labels)`. Batch size
/// is fixed to 1 in line with the rest of the lazy port.
#[derive(Debug, Clone)]
pub struct DebertaV2NERModel {
    pub config: DebertaV2Config,
    pub weights: DebertaV2NERWeights,
    pub num_labels: usize,
}

impl DebertaV2NERModel {
    /// Build a NER model from already-loaded weights. `num_labels` is
    /// the number of entity classes (length of `id2label`).
    pub fn new(
        config: DebertaV2Config,
        weights: DebertaV2NERWeights,
        num_labels: usize,
    ) -> Self {
        Self { config, weights, num_labels }
    }

    /// Run prefill + the NER head. `token_type_ids` / `attention_mask`
    /// are accepted to match the binary's call shape but are currently
    /// unused (the lazy encoder doesn't yet ingest either — see the
    /// module docstring for v1 scope).
    pub fn forward(
        &self,
        input_ids: &[u32],
        token_type_ids: Option<&[u32]>,
        attention_mask: Option<&[u32]>,
    ) -> Result<LazyTensor> {
        let _ = token_type_ids;
        let _ = attention_mask;
        let encoder = DebertaV2Model {
            config: self.config.clone(),
            weights: self.weights.encoder.clone(),
        };
        let hidden = encoder.forward(input_ids)?;
        // hidden: (1, seq, hidden). Anchor for the classifier consts.
        let anchor = LazyTensor::from_u32(
            input_ids.to_vec(),
            Shape::from_dims(&[input_ids.len()]),
            &crate::Device::cpu(),
        );
        // Apply classifier to every token. apply_linear_with_bias
        // works on the trailing dim — exactly what we want here.
        apply_linear(&hidden, &self.weights.classifier, &anchor)
    }
}

impl DebertaV2NERWeights {
    /// Load encoder + token-classification head from a HuggingFace
    /// safetensors checkpoint.
    ///
    /// Expected tensor names (in addition to the encoder prefix used
    /// by [`DebertaV2Weights::load_from_mmapped`]):
    /// - `classifier.weight`  — shape `[num_labels, hidden_size]`
    /// - `classifier.bias`    — shape `[num_labels]` (optional; zeros
    ///   are substituted when absent — matches HF's `linear_no_bias`
    ///   variant in the older code paths).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &DebertaV2Config,
        num_labels: usize,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype as ltm};
        let encoder = DebertaV2Weights::load_from_mmapped(st, cfg)?;
        let w = ltm(st, "classifier.weight", num_labels, cfg.hidden_size)?;
        let b = match load_tensor_as_f32(st, "classifier.bias") {
            Ok(v) => Arc::from(v),
            Err(_) => Arc::from(vec![0.0_f32; num_labels]),
        };
        Ok(Self {
            encoder,
            classifier: LinearWeights { w, b },
        })
    }
}

// ---- DebertaV2SeqClassificationModel ---------------------------------------

/// Weights for the sequence-classification head: a pooler `dense`
/// (hidden → hidden) followed by tanh, then a `classifier` linear
/// (hidden → num_labels). The pooler reads the `[CLS]` token only.
#[derive(Debug, Clone)]
pub struct DebertaV2SeqClassificationWeights {
    pub encoder: DebertaV2Weights,
    pub pooler_dense: LinearWeights,
    pub classifier: LinearWeights,
}

/// Sequence-classification model. Mirrors the eager
/// `DebertaV2SeqClassificationModel`: pool the first-token hidden
/// state via dense + tanh, then project to per-class logits.
///
/// Forward returns logits of shape `(1, num_labels)`.
#[derive(Debug, Clone)]
pub struct DebertaV2SeqClassificationModel {
    pub config: DebertaV2Config,
    pub weights: DebertaV2SeqClassificationWeights,
    pub num_labels: usize,
}

impl DebertaV2SeqClassificationModel {
    pub fn new(
        config: DebertaV2Config,
        weights: DebertaV2SeqClassificationWeights,
        num_labels: usize,
    ) -> Self {
        Self { config, weights, num_labels }
    }

    /// Run prefill + sequence-classification head.
    pub fn forward(
        &self,
        input_ids: &[u32],
        token_type_ids: Option<&[u32]>,
        attention_mask: Option<&[u32]>,
    ) -> Result<LazyTensor> {
        let _ = token_type_ids;
        let _ = attention_mask;
        let encoder = DebertaV2Model {
            config: self.config.clone(),
            weights: self.weights.encoder.clone(),
        };
        let hidden = encoder.forward(input_ids)?; // (1, seq, hidden)
        // First-token ([CLS]) hidden state: (1, 1, hidden) → (1, hidden).
        let cls = hidden.slice(1_usize, 0, 1)?.squeeze(1_usize)?;
        let anchor = LazyTensor::from_u32(
            input_ids.to_vec(),
            Shape::from_dims(&[input_ids.len()]),
            &crate::Device::cpu(),
        );
        // pooler.dense → tanh → classifier.
        let pooled = apply_linear(&cls, &self.weights.pooler_dense, &anchor)?;
        let pooled = pooled.tanh();
        apply_linear(&pooled, &self.weights.classifier, &anchor)
    }
}

impl DebertaV2SeqClassificationWeights {
    /// Load encoder + pooler + classifier head from a HuggingFace
    /// safetensors checkpoint.
    ///
    /// Expected tensor names (in addition to the encoder prefix):
    /// - `pooler.dense.weight`   — `[hidden_size, hidden_size]`
    /// - `pooler.dense.bias`     — `[hidden_size]`
    /// - `classifier.weight`     — `[num_labels, hidden_size]`
    /// - `classifier.bias`       — `[num_labels]` (optional; zeros
    ///   substituted on absence)
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &DebertaV2Config,
        num_labels: usize,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype as ltm};
        let h = cfg.hidden_size;
        let encoder = DebertaV2Weights::load_from_mmapped(st, cfg)?;
        let pooler_w = ltm(st, "pooler.dense.weight", h, h)?;
        let pooler_b = Arc::from(load_tensor_as_f32(st, "pooler.dense.bias")?);
        let classifier_w = ltm(st, "classifier.weight", num_labels, h)?;
        let classifier_b = match load_tensor_as_f32(st, "classifier.bias") {
            Ok(v) => Arc::from(v),
            Err(_) => Arc::from(vec![0.0_f32; num_labels]),
        };
        Ok(Self {
            encoder,
            pooler_dense: LinearWeights { w: pooler_w, b: pooler_b },
            classifier: LinearWeights { w: classifier_w, b: classifier_b },
        })
    }
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
            id2label: None,
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

    // ---- Task head tests ---------------------------------------------------

    fn build_ner_weights(
        cfg: &DebertaV2Config, num_labels: usize,
    ) -> DebertaV2NERWeights {
        let mut nb = rng_seed(2027);
        let encoder = build_weights(cfg);
        let classifier = linear_w(cfg.hidden_size, num_labels, &mut nb);
        DebertaV2NERWeights { encoder, classifier }
    }

    fn build_seq_cls_weights(
        cfg: &DebertaV2Config, num_labels: usize,
    ) -> DebertaV2SeqClassificationWeights {
        let mut nb = rng_seed(2028);
        let encoder = build_weights(cfg);
        let pooler_dense = linear_w(cfg.hidden_size, cfg.hidden_size, &mut nb);
        let classifier = linear_w(cfg.hidden_size, num_labels, &mut nb);
        DebertaV2SeqClassificationWeights {
            encoder, pooler_dense, classifier,
        }
    }

    #[test]
    fn ner_forward_shape_per_token_logits() {
        let cfg = tiny_config();
        let num_labels = 5;
        let weights = build_ner_weights(&cfg, num_labels);
        let model = DebertaV2NERModel::new(cfg.clone(), weights, num_labels);
        let ids: Vec<u32> = (1..9).collect();
        let logits = model.forward(&ids, None, None).unwrap();
        assert_eq!(logits.shape().dims(), &[1, ids.len(), num_labels]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite NER logit: {v}");
        }
    }

    #[test]
    fn seq_classification_forward_shape_one_per_sequence() {
        let cfg = tiny_config();
        let num_labels = 3;
        let weights = build_seq_cls_weights(&cfg, num_labels);
        let model = DebertaV2SeqClassificationModel::new(cfg, weights, num_labels);
        let ids: Vec<u32> = (1..9).collect();
        let logits = model.forward(&ids, None, None).unwrap();
        assert_eq!(logits.shape().dims(), &[1, num_labels]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite seq-class logit: {v}");
        }
    }

    // ---- Safetensors fixture round-trip -----------------------------------

    fn push_f32(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        name: &str,
        shape: Vec<usize>,
        nb: &mut dyn FnMut() -> f32,
    ) {
        let n: usize = shape.iter().product();
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n { bytes.extend_from_slice(&nb().to_le_bytes()); }
        owned.push((name.to_string(), shape, bytes));
    }

    fn push_ln(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        c: usize,
        nb: &mut dyn FnMut() -> f32,
    ) {
        // HF DeBERTa LayerNorm fields are `.weight` / `.bias`.
        push_f32(owned, &format!("{prefix}.weight"), vec![c], nb);
        push_f32(owned, &format!("{prefix}.bias"), vec![c], nb);
    }

    fn push_lin(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        nb: &mut dyn FnMut() -> f32,
    ) {
        // HF stores Linear weight as [out, in].
        push_f32(
            owned, &format!("{prefix}.weight"),
            vec![out_features, in_features], nb,
        );
        push_f32(owned, &format!("{prefix}.bias"), vec![out_features], nb);
    }

    fn push_encoder(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        cfg: &DebertaV2Config,
        nb: &mut dyn FnMut() -> f32,
    ) {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        // Embeddings.
        push_f32(owned, "embeddings.word_embeddings.weight",
            vec![cfg.vocab_size, h], nb);
        push_ln(owned, "embeddings.LayerNorm", h, nb);
        // Relative-position embeddings.
        push_f32(owned, "encoder.rel_embeddings.weight",
            vec![2 * cfg.position_buckets, h], nb);
        push_ln(owned, "encoder.LayerNorm", h, nb);
        // Transformer stack.
        for i in 0..cfg.num_hidden_layers {
            let p = format!("encoder.layer.{i}");
            push_lin(owned, &format!("{p}.attention.self.query_proj"), h, h, nb);
            push_lin(owned, &format!("{p}.attention.self.key_proj"), h, h, nb);
            push_lin(owned, &format!("{p}.attention.self.value_proj"), h, h, nb);
            push_lin(owned, &format!("{p}.attention.output.dense"), h, h, nb);
            push_ln(owned, &format!("{p}.attention.output.LayerNorm"), h, nb);
            push_lin(owned, &format!("{p}.intermediate.dense"), h, inter, nb);
            push_lin(owned, &format!("{p}.output.dense"), inter, h, nb);
            push_ln(owned, &format!("{p}.output.LayerNorm"), h, nb);
        }
    }

    fn build_safetensors_file(
        owned: Vec<(String, Vec<usize>, Vec<u8>)>,
        tag: &str,
    ) -> std::path::PathBuf {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        for (name, shape, bytes) in &owned {
            let view = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                .expect("TensorView::new");
            tensors.insert(name.clone(), view);
        }
        let serialized = safetensors::serialize(&tensors, None)
            .expect("safetensors::serialize");
        let tmp = std::env::temp_dir().join(format!(
            "fuel_debertav2_load_test_{}_{tag}.safetensors",
            std::process::id(),
        ));
        std::fs::write(&tmp, &serialized).expect("write tmp");
        tmp
    }

    #[test]
    fn ner_weights_round_trip_through_safetensors_fixture() {
        let cfg = tiny_config();
        let num_labels = 4;
        let mut nb = rng_seed(31);

        let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        push_encoder(&mut owned, &cfg, &mut nb);
        push_lin(&mut owned, "classifier", cfg.hidden_size, num_labels, &mut nb);

        let tmp = build_safetensors_file(owned, "ner");
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&tmp) }
            .expect("MmapedSafetensors::new");

        let weights = DebertaV2NERWeights::load_from_mmapped(&st, &cfg, num_labels)
            .expect("load NER weights");
        let model = DebertaV2NERModel::new(cfg.clone(), weights, num_labels);

        let ids = vec![1_u32, 2, 3, 4, 5, 6, 7, 8];
        let logits = model.forward(&ids, None, None).unwrap();
        assert_eq!(logits.shape().dims(), &[1, ids.len(), num_labels]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite NER logit after load: {v}");
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn seq_classification_weights_round_trip_through_safetensors_fixture() {
        let cfg = tiny_config();
        let num_labels = 2;
        let mut nb = rng_seed(41);

        let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        push_encoder(&mut owned, &cfg, &mut nb);
        push_lin(&mut owned, "pooler.dense",
            cfg.hidden_size, cfg.hidden_size, &mut nb);
        push_lin(&mut owned, "classifier",
            cfg.hidden_size, num_labels, &mut nb);

        let tmp = build_safetensors_file(owned, "seqcls");
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&tmp) }
            .expect("MmapedSafetensors::new");

        let weights = DebertaV2SeqClassificationWeights::load_from_mmapped(
            &st, &cfg, num_labels,
        ).expect("load seq-cls weights");
        let model = DebertaV2SeqClassificationModel::new(
            cfg, weights, num_labels,
        );

        let ids = vec![1_u32, 2, 3, 4, 5, 6, 7, 8];
        let logits = model.forward(&ids, None, None).unwrap();
        assert_eq!(logits.shape().dims(), &[1, num_labels]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite seq-class logit after load: {v}");
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn from_hf_json_str_parses_id2label() {
        let json = r#"{
            "vocab_size": 128100,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "layer_norm_eps": 1e-7,
            "position_buckets": 256,
            "max_relative_positions": -1,
            "id2label": {"0": "O", "1": "B-PER", "2": "I-PER"}
        }"#;
        let cfg = DebertaV2Config::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.vocab_size, 128_100);
        assert_eq!(cfg.position_buckets, 256);
        // -1 → defaults to max_position_embeddings.
        assert_eq!(cfg.max_relative_positions, 512);
        let labels = cfg.id2label.expect("id2label parsed");
        assert_eq!(labels.len(), 3);
        assert_eq!(labels.get(&0).map(String::as_str), Some("O"));
        assert_eq!(labels.get(&1).map(String::as_str), Some("B-PER"));
        assert_eq!(labels.get(&2).map(String::as_str), Some("I-PER"));
    }

    #[test]
    fn from_hf_json_str_id2label_optional() {
        let json = r#"{
            "vocab_size": 100,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "max_position_embeddings": 32
        }"#;
        let cfg = DebertaV2Config::from_hf_json_str(json).unwrap();
        assert!(cfg.id2label.is_none());
        assert_eq!(cfg.position_buckets, 256);
    }
}

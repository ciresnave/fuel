// SPDX-License-Identifier: MIT OR Apache-2.0
//! The gated full-attention token mixer of Qwen3.5, composed from base lazy ops. ROADMAP row 115
//! (`docs/design-notes/2026-10-30-qwen35-composed-forward-row-115.md`, section 3).
//!
//! [`gated_full_attention`] is a prefill from position 0. [`gated_full_attention_cached`] takes the
//! keys and values of earlier tokens as host data and returns the grown ones.

use crate::models::lazy_qwen3_5::{FullAttentionWeights, Qwen3_5Config};
use fuel_ir::Shape;
use fuel_ir::error::{Error, Result};
use fuel_tensor::lazy::{Tensor, WeightStorage};
use std::sync::Arc;

/// Rotated width of each head: `int(head_dim * partial_rotary_factor)`.
pub fn rotary_dim(cfg: &Qwen3_5Config) -> usize {
    (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize
}

/// Additive causal mask `[1, 1, t, past + t]` for `t` new queries after `past` cached tokens:
/// query `i` (absolute position `past + i`) sees keys `0..=past + i`; the rest are `-inf`.
fn causal_mask(anchor: &Tensor, t: usize, past: usize) -> Result<Tensor> {
    let total = past + t;
    let mut data = vec![0.0_f32; t * total];
    for i in 0..t {
        for j in (past + i + 1)..total {
            data[i * total + j] = f32::NEG_INFINITY;
        }
    }
    anchor.const_f32_like(data, Shape::from_dims(&[1, 1, t, total]))
}

/// The keys and values of the tokens already seen by one attention layer, as host data in the
/// `[n_kv_heads, len, head_dim]` layout (post-norm, post-rotary keys).
#[derive(Debug, Clone)]
pub struct AttnPast<'a> {
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub len: usize,
}

/// A key or value projection `[T, hidden] -> [1, nkv, T, hd]`.
fn head_projection(x: &Tensor, proj: &WeightStorage, cfg: &Qwen3_5Config) -> Result<Tensor> {
    let t = x.shape().dims()[0];
    let (nkv, hd) = (cfg.num_key_value_heads, cfg.head_dim);
    proj.apply_linear(x, cfg.hidden_size, nkv * hd)?
        .reshape(Shape::from_dims(&[1, t, nkv * hd]))?
        .split_heads(nkv, hd)
}

/// The query, gate, key and value of one layer, each in the attention layout:
/// `q [1, nh, T, hd]`, `k` and `v [1, nkv, T, hd]` (q and k normed and rotated), and the output
/// gate `[T, nh * hd]`.
fn project(
    x: &Tensor,
    w: &FullAttentionWeights,
    cfg: &Qwen3_5Config,
    start_pos: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let t = x.shape().dims()[0];
    let (nh, hd) = (cfg.num_attention_heads, cfg.head_dim);
    let q_dim = cfg.attn_q_dim();
    if !cfg.attn_output_gate {
        return Err(Error::Msg(
            "gated_full_attention: attn_output_gate = false is not supported by this layer".into(),
        )
        .bt());
    }
    // `q_proj` is `[T, nh * 2 * hd]`; per head, the first `hd` is the query and the next `hd` the gate.
    let qg = w
        .q_proj
        .apply_linear(x, cfg.hidden_size, 2 * q_dim)?
        .reshape(Shape::from_dims(&[t, nh, 2 * hd]))?;
    let q = qg
        .narrow(2, 0, hd)?
        .reshape(Shape::from_dims(&[1, t, q_dim]))?
        .split_heads(nh, hd)?;
    let gate = qg
        .narrow(2, hd, hd)?
        .reshape(Shape::from_dims(&[t, q_dim]))?;
    let k = head_projection(x, &w.k_proj, cfg)?;
    let v = head_projection(x, &w.v_proj, cfg)?;
    let q = q.rms_norm_affine(Arc::clone(&w.q_norm_gain), cfg.rms_norm_eps)?;
    let k = k.rms_norm_affine(Arc::clone(&w.k_norm_gain), cfg.rms_norm_eps)?;
    // Rotate-half over the first `rotary_dim` of each head; the rest passes through.
    let rd = rotary_dim(cfg);
    let (cos, sin) = x.rope_tables_const(cfg.rope_theta, start_pos, t, rd);
    Ok((
        q.rope_partial(&cos, &sin, rd)?,
        k.rope_partial(&cos, &sin, rd)?,
        v,
        gate,
    ))
}

/// One gated full-attention token mixer over `x [T, hidden]` (already input-normed), as the
/// HuggingFace `Qwen3_5Attention.forward` computes it for text: GQA, scale `head_dim^-0.5`,
/// causal, output multiplied by `sigmoid(gate)`, then `o_proj`. Returns `[T, hidden]`.
pub fn gated_full_attention(
    x: &Tensor,
    w: &FullAttentionWeights,
    cfg: &Qwen3_5Config,
) -> Result<Tensor> {
    Ok(gated_full_attention_cached(x, w, cfg, None)?.0)
}

/// The keys and values with the cached ones in front: `[1, n_kv_heads, past.len + T, head_dim]`.
fn with_past(
    x: &Tensor,
    k: Tensor,
    v: Tensor,
    past: Option<&AttnPast<'_>>,
    cfg: &Qwen3_5Config,
) -> Result<(Tensor, Tensor)> {
    let Some(p) = past else { return Ok((k, v)) };
    let dims = Shape::from_dims(&[1, cfg.num_key_value_heads, p.len, cfg.head_dim]);
    let past_k = x.const_f32_like(p.k.to_vec(), dims.clone())?;
    let past_v = x.const_f32_like(p.v.to_vec(), dims)?;
    Ok((past_k.concat(&k, 2)?, past_v.concat(&v, 2)?))
}

/// Causal softmax attention of `q` over the (already grown) `k`, `v`, heads merged: `[T, nh * hd]`.
fn attend(
    x: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    start: usize,
    cfg: &Qwen3_5Config,
) -> Result<Tensor> {
    let t = x.shape().dims()[0];
    let (nh, hd) = (cfg.num_attention_heads, cfg.head_dim);
    let n_rep = nh / cfg.num_key_value_heads;
    let k_rep = k.repeat_interleave(1_usize, n_rep)?;
    let v_rep = v.repeat_interleave(1_usize, n_rep)?;
    let scores = q
        .matmul(&k_rep.transpose()?)?
        .mul_scalar(1.0 / (hd as f64).sqrt())
        .broadcast_add(&causal_mask(x, t, start)?)?;
    scores
        .softmax_last_dim()?
        .matmul(&v_rep)?
        .merge_heads()?
        .reshape(Shape::from_dims(&[t, nh * hd]))
}

/// [`gated_full_attention`] for `T` new tokens after `past`: the new tokens take positions
/// `past.len..past.len + T`. Returns `(y [T, hidden], k, v)` where `k` and `v` are the grown
/// `[1, n_kv_heads, past.len + T, head_dim]` keys and values to store for the next call.
pub fn gated_full_attention_cached(
    x: &Tensor,
    w: &FullAttentionWeights,
    cfg: &Qwen3_5Config,
    past: Option<&AttnPast<'_>>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let start = past.map_or(0, |p| p.len);
    let (q, k, v, gate) = project(x, w, cfg, start)?;
    let (k, v) = with_past(x, k, v, past, cfg)?;
    let ctx = attend(x, &q, &k, &v, start, cfg)?;
    let nhd = cfg.num_attention_heads * cfg.head_dim;
    let y = w
        .o_proj
        .apply_linear(&ctx.mul(&gate.sigmoid())?, nhd, cfg.hidden_size)?;
    Ok((y, k, v))
}

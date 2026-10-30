// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Gated DeltaNet recurrence of Qwen3.5, composed from base lazy ops. ROADMAP row 115,
//! step 1 (`docs/design-notes/2026-10-30-qwen35-composed-forward-row-115.md`).
//!
//! This is the correct-but-slow form: the token loop is unrolled into graph nodes (about 25 per
//! token), so it is meant for oracles and short sequences. Chunkwise prefill is row 116 and a
//! fused kernel is row 122.

use crate::models::lazy_qwen3_5::{GatedDeltaNetWeights, Qwen3_5Config};
use fuel_ir::Shape;
use fuel_ir::error::{Error, Result};
use fuel_tensor::lazy::Tensor;
use std::sync::Arc;

/// `x / sqrt(sum(x^2) + eps)` over the last axis (the reference's `l2norm`, `eps = 1e-6`).
pub fn l2norm_last(x: &Tensor, eps: f64) -> Result<Tensor> {
    let ss = x.sqr().sum_keepdim(x.shape().rank() - 1)?;
    let inv = ss.add_scalar(eps).rsqrt();
    x.broadcast_mul(&inv)
}

/// The delta-rule recurrence over `T` tokens, per value head, in the reference's order:
///
/// ```text
/// q, k <- l2norm(q), l2norm(k);  q <- q / sqrt(dk)
/// for each token t:  S <- S * exp(g_t);  delta <- (v_t - S^T k_t) * beta_t;
///                    S <- S + k_t delta^T;  o_t <- S^T q_t
/// ```
///
/// Shapes (no batch axis): `q`, `k` are `[T, nv, dk]` (already repeated to the value-head count),
/// `v` is `[T, nv, dv]`, `g` and `beta` are `[T, nv]`, `state` is `[nv, dk, dv]` or `None` for a
/// zero start. Returns `(out [T, nv, dv], final_state [nv, dk, dv])`.
pub fn gated_delta_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let qd = q.shape().dims().to_vec();
    let vd = v.shape().dims().to_vec();
    if qd.len() != 3 || vd.len() != 3 {
        return Err(Error::Msg(format!(
            "gated_delta_recurrence: q and v must be rank 3 [T, nv, d], got {qd:?} and {vd:?}"
        ))
        .bt());
    }
    let (t_len, nv, dk, dv) = (qd[0], qd[1], qd[2], vd[2]);
    if t_len == 0 {
        return Err(Error::Msg("gated_delta_recurrence: zero tokens".into()).bt());
    }
    let q = l2norm_last(q, 1e-6)?.mul_scalar(1.0 / (dk as f64).sqrt());
    let k = l2norm_last(k, 1e-6)?;
    let mut s = match state {
        Some(s) => s.clone(),
        None => q.const_f32_like(vec![0.0_f32; nv * dk * dv], Shape::from_dims(&[nv, dk, dv]))?,
    };
    let mut out: Option<Tensor> = None;
    for t in 0..t_len {
        let q_t = q.narrow(0, t, 1)?.reshape(Shape::from_dims(&[nv, 1, dk]))?;
        let k_t = k.narrow(0, t, 1)?.reshape(Shape::from_dims(&[nv, 1, dk]))?;
        let k_col = k_t.transpose_last_two()?;
        let v_t = v.narrow(0, t, 1)?.reshape(Shape::from_dims(&[nv, 1, dv]))?;
        let decay = g
            .narrow(0, t, 1)?
            .reshape(Shape::from_dims(&[nv, 1, 1]))?
            .exp();
        let beta_t = beta
            .narrow(0, t, 1)?
            .reshape(Shape::from_dims(&[nv, 1, 1]))?;
        s = s.broadcast_mul(&decay)?;
        let kv = k_t.matmul(&s)?;
        let delta = v_t.sub(&kv)?.broadcast_mul(&beta_t)?;
        s = s.add(&k_col.matmul(&delta)?)?;
        let o_t = q_t.matmul(&s)?.reshape(Shape::from_dims(&[1, nv, dv]))?;
        out = Some(match out {
            None => o_t,
            Some(prev) => prev.concat(&o_t, 0)?,
        });
    }
    // `t_len > 0` was checked above, so `out` is `Some`.
    let out = out.ok_or_else(|| Error::Msg("gated_delta_recurrence: no output".into()).bt())?;
    Ok((out, s))
}

/// `softplus(x) = max(x, 0) + ln(1 + exp(-|x|))`: exact in f32 and branch-free (the naive
/// `ln(1 + exp(x))` overflows for large `x`).
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    x.relu().add(&x.abs().neg().exp().add_scalar(1.0).log())
}

/// The state a Gated DeltaNet layer carries between calls: the recurrent matrix `S` and the last
/// `K - 1` inputs of the depthwise conv. Both are fixed size (they do not grow with the context).
#[derive(Debug, Clone)]
pub struct GdnState {
    /// `[nv, dk, dv]`, f32.
    pub recurrent: Tensor,
    /// `[1, conv_dim, K - 1]`, f32.
    pub conv_window: Tensor,
}

/// One Gated DeltaNet token mixer over `x [T, hidden]` (already input-normed), as the HuggingFace
/// `Qwen3_5GatedDeltaNet.forward` computes it. Returns `(y [T, hidden], new_state)`; `state` is
/// `None` for the start of a sequence.
pub fn gated_delta_net_mixer(
    x: &Tensor,
    w: &GatedDeltaNetWeights,
    cfg: &Qwen3_5Config,
    state: Option<&GdnState>,
) -> Result<(Tensor, GdnState)> {
    let t_len = x.shape().dims()[0];
    let h = cfg.hidden_size;
    let (nk, nv) = (cfg.linear_num_key_heads, cfg.linear_num_value_heads);
    let (dk, dv) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
    let (key_dim, value_dim, conv_dim) = (
        cfg.linear_key_dim(),
        cfg.linear_value_dim(),
        cfg.linear_conv_dim(),
    );
    let kern = cfg.linear_conv_kernel_dim;

    let mixed = w.in_proj_qkv.apply_linear(x, h, conv_dim)?;
    let z = w.in_proj_z.apply_linear(x, h, value_dim)?;
    let b = w.in_proj_b.apply_linear(x, h, nv)?;
    let a = w.in_proj_a.apply_linear(x, h, nv)?;

    // Depthwise causal conv + SiLU over the channel axis, with the carried window as left context.
    let chan_major = mixed
        .transpose()?
        .reshape(Shape::from_dims(&[1, conv_dim, t_len]))?;
    let window = match state {
        Some(s) => s.conv_window.clone(),
        None => x.const_f32_like(
            vec![0.0_f32; conv_dim * (kern - 1)],
            Shape::from_dims(&[1, conv_dim, kern - 1]),
        )?,
    };
    let padded = window.concat(&chan_major, 2)?;
    let conv_w = x.const_f32_like(
        Arc::clone(&w.conv1d_weight),
        Shape::from_dims(&[conv_dim, 1, kern]),
    )?;
    let conv_b = x.const_f32_like(vec![0.0_f32; conv_dim], Shape::from_dims(&[conv_dim]))?;
    let conv_out = padded.causal_conv1d(&conv_w, &conv_b, true); // [1, conv_dim, T]
    let new_window = padded.narrow(2, t_len, kern - 1)?;
    let mixed = conv_out
        .reshape(Shape::from_dims(&[conv_dim, t_len]))?
        .transpose()?; // [T, conv_dim]

    let q = mixed
        .narrow(1, 0, key_dim)?
        .reshape(Shape::from_dims(&[t_len, nk, dk]))?;
    let k = mixed
        .narrow(1, key_dim, key_dim)?
        .reshape(Shape::from_dims(&[t_len, nk, dk]))?;
    let v = mixed
        .narrow(1, 2 * key_dim, value_dim)?
        .reshape(Shape::from_dims(&[t_len, nv, dv]))?;
    // Each key head serves `nv / nk` consecutive value heads (`repeat_interleave`, not a tile).
    let (q, k) = if nv > nk {
        (
            q.repeat_interleave(1_usize, nv / nk)?,
            k.repeat_interleave(1_usize, nv / nk)?,
        )
    } else {
        (q, k)
    };

    let beta = b.sigmoid();
    let dt_bias = x.const_f32_like(Arc::clone(&w.dt_bias), Shape::from_dims(&[nv]))?;
    let neg_exp_a: Vec<f32> = w.a_log.iter().map(|v| -v.exp()).collect();
    let neg_exp_a = x.const_f32_like(neg_exp_a, Shape::from_dims(&[nv]))?;
    let g = softplus(&a.broadcast_add(&dt_bias)?)?.broadcast_mul(&neg_exp_a)?;

    let prev = state.map(|s| &s.recurrent);
    let (core, recurrent) = gated_delta_recurrence(&q, &k, &v, &g, &beta, prev)?;

    // Gated RMS norm per value head: weight * rms_norm(o) * silu(z).
    let ms = core.sqr().mean_keepdim(2_usize)?;
    let normed = core.broadcast_mul(&ms.add_scalar(cfg.rms_norm_eps).rsqrt())?;
    let gain = x.const_f32_like(Arc::clone(&w.norm_gain), Shape::from_dims(&[dv]))?;
    let z = z.reshape(Shape::from_dims(&[t_len, nv, dv]))?;
    let gated = normed.broadcast_mul(&gain)?.mul(&z.silu())?;
    let y = w.out_proj.apply_linear(
        &gated.reshape(Shape::from_dims(&[t_len, value_dim]))?,
        value_dim,
        h,
    )?;
    Ok((
        y,
        GdnState {
            recurrent,
            conv_window: new_window,
        },
    ))
}

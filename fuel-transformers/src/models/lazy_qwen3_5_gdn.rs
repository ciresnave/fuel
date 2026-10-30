// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Gated DeltaNet recurrence of Qwen3.5, composed from base lazy ops. ROADMAP row 115,
//! step 1 (`docs/design-notes/2026-10-30-qwen35-composed-forward-row-115.md`).
//!
//! This is the correct-but-slow form: the token loop is unrolled into graph nodes (about 25 per
//! token), so it is meant for oracles and short sequences. Chunkwise prefill is row 116 and a
//! fused kernel is row 122.

use fuel_ir::Shape;
use fuel_ir::error::{Error, Result};
use fuel_tensor::lazy::Tensor;

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

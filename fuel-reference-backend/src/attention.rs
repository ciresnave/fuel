//! Attention reference implementations (Phase 8 Tier 1).
//!
//! Two equivalent forms of multi-head scaled-dot-product attention:
//!
//! - [`attention_naive`] — the math definition. Builds the full
//!   `[B, H, Sq, Sk]` attention matrix, applies mask, softmax, matmuls
//!   with `V`. O(N²) memory, O(N²·d) compute. The textbook oracle.
//! - [`attention_flash`] — the FlashAttention-v2 forward algorithm
//!   in pure Rust. Tiles `K`/`V` into blocks, processes them with
//!   online softmax, never materializes the full attention matrix.
//!   O(N·d) memory, O(N²·d) compute. The Tier 1 deliverable that
//!   Tiers 2/3 must match.
//!
//! Backward via recompute is in [`attention_flash_backward`]. The
//! recompute approach (vs saving the attention matrix) is what
//! actual FA kernels do — it costs an extra forward but lets the
//! backward fit in O(N·d) memory.
//!
//! All shapes are `[B, H, S, D]` — batch-first, heads-second. GQA
//! is expressed by `H_q > H_kv`; the implementations broadcast each
//! KV head across the corresponding group of Q heads.
//!
//! Optional features wired (any combination):
//! - **Causal masking** (`p.causal`).
//! - **Sliding-window attention** (`p.window_size_left/right`).
//!   Position `j` is masked unless `i - left ≤ j ≤ i + right`.
//! - **ALiBi slopes** (per-head bias `slope_h * (j - i)`).
//! - **Softcap** (`tanh(x / softcap) * softcap`) before softmax.

use crate::RefTensor;
use fuel_core_types::Shape;
use num_traits::Float;

/// Parameters that don't depend on the input tensors.
#[derive(Clone, Copy, Debug)]
pub struct AttentionParams {
    /// Multiplier applied to `Q · K^T` before any masking. Typically
    /// `1 / sqrt(head_dim)`. The kernel doesn't infer this — callers
    /// pass it explicitly so non-standard scales (Mistral's pre-norm
    /// scale, MQA experiments) work without a special case.
    pub softmax_scale: f32,
    /// Apply causal (lower-triangular) mask: `j > i` is masked.
    pub causal: bool,
    /// Sliding-window left half-width. `Some(w)` masks positions with
    /// `j < i - w`. `None` disables.
    pub window_size_left: Option<usize>,
    /// Sliding-window right half-width. `Some(w)` masks positions with
    /// `j > i + w`. `None` disables.
    pub window_size_right: Option<usize>,
    /// If `Some(c)`, apply `tanh(x / c) * c` to the pre-softmax scores
    /// to prevent extreme values (Gemma-style logit cap).
    pub softcap: Option<f32>,
}

impl Default for AttentionParams {
    fn default() -> Self {
        Self {
            softmax_scale: 1.0,
            causal: false,
            window_size_left: None,
            window_size_right: None,
            softcap: None,
        }
    }
}

/// Decompose `[B, H, S, D]` shape, asserting rank 4.
fn dims_bhsd<T: Clone>(t: &RefTensor<T>, label: &str) -> (usize, usize, usize, usize) {
    let d = t.shape().dims();
    assert_eq!(d.len(), 4, "{label}: expected rank 4 [B, H, S, D], got {d:?}");
    (d[0], d[1], d[2], d[3])
}

/// Whether `(query_pos, key_pos)` is admissible under mask + window.
#[inline]
fn position_admissible(
    qi: usize,
    kj: usize,
    p: &AttentionParams,
) -> bool {
    if p.causal && kj > qi {
        return false;
    }
    if let Some(w) = p.window_size_left {
        if kj + w < qi {
            return false;
        }
    }
    if let Some(w) = p.window_size_right {
        if kj > qi + w {
            return false;
        }
    }
    true
}

/// Naive multi-head scaled-dot-product attention. Builds the full
/// `[B, H, Sq, Sk]` attention matrix. The math-definition oracle.
pub fn attention_naive<T: Float>(
    q: &RefTensor<T>,
    k: &RefTensor<T>,
    v: &RefTensor<T>,
    alibi_slopes: Option<&RefTensor<T>>,
    p: &AttentionParams,
) -> RefTensor<T> {
    let (b_q, h_q, sq, d_q) = dims_bhsd(q, "attention_naive q");
    let (b_k, h_k, sk, d_k) = dims_bhsd(k, "attention_naive k");
    let (b_v, h_v, sk_v, d_v) = dims_bhsd(v, "attention_naive v");
    assert_eq!(b_q, b_k, "B mismatch q vs k");
    assert_eq!(b_k, b_v, "B mismatch k vs v");
    assert_eq!(h_k, h_v, "Hk mismatch k vs v");
    assert_eq!(sk, sk_v, "Sk mismatch k vs v");
    assert_eq!(d_q, d_k, "head_dim mismatch q vs k");
    assert_eq!(d_k, d_v, "head_dim mismatch k vs v");
    assert_eq!(
        h_q % h_k, 0,
        "Hq={h_q} must be a multiple of Hk={h_k} (GQA group size)",
    );
    let groups = h_q / h_k;
    if let Some(slopes) = alibi_slopes {
        assert_eq!(
            slopes.shape().dims(), &[h_q],
            "alibi_slopes must be shape [Hq={h_q}]",
        );
    }

    let q_data = q.as_slice();
    let k_data = k.as_slice();
    let v_data = v.as_slice();
    let alibi_data = alibi_slopes.map(|t| t.as_slice());

    let scale = T::from(p.softmax_scale).expect("softmax_scale must convert to T");
    let softcap = p.softcap.and_then(|c| T::from(c).map(|t| (t, T::one() / t)));

    let mut out = vec![T::zero(); b_q * h_q * sq * d_q];

    // Strides for [B, H, S, D] contiguous layout.
    let q_h_stride = sq * d_q;
    let q_b_stride = h_q * q_h_stride;
    let k_h_stride = sk * d_k;
    let k_b_stride = h_k * k_h_stride;
    let o_h_stride = sq * d_q;
    let o_b_stride = h_q * o_h_stride;

    for bi in 0..b_q {
        for hi in 0..h_q {
            let kv_h = hi / groups;
            let q_off = bi * q_b_stride + hi * q_h_stride;
            let k_off = bi * k_b_stride + kv_h * k_h_stride;
            let v_off = k_off; // same shape as K
            let o_off = bi * o_b_stride + hi * o_h_stride;
            let alibi_h = alibi_data.map(|a| a[hi]);
            for qi in 0..sq {
                // Compute raw scores S[qi, kj] for this (b, h, qi).
                let mut scores = vec![T::neg_infinity(); sk];
                let mut max_score = T::neg_infinity();
                for kj in 0..sk {
                    if !position_admissible(qi, kj, p) {
                        continue;
                    }
                    let mut acc = T::zero();
                    let q_row = &q_data[q_off + qi * d_q .. q_off + (qi + 1) * d_q];
                    let k_row = &k_data[k_off + kj * d_k .. k_off + (kj + 1) * d_k];
                    for (qx, kx) in q_row.iter().zip(k_row.iter()) {
                        acc = acc + (*qx) * (*kx);
                    }
                    let mut s = acc * scale;
                    if let Some((c, inv_c)) = softcap {
                        s = (s * inv_c).tanh() * c;
                    }
                    if let Some(slope) = alibi_h {
                        // ALiBi adds `slope * (kj - qi)` (Press et al.).
                        let delta = T::from(kj as f32 - qi as f32)
                            .expect("alibi delta must convert to T");
                        s = s + slope * delta;
                    }
                    scores[kj] = s;
                    if s > max_score { max_score = s; }
                }
                if !max_score.is_finite() {
                    // No admissible keys — output row is all zeros (matches
                    // FA's "fully masked" handling).
                    continue;
                }
                // Stable softmax.
                let mut sum = T::zero();
                for s in scores.iter_mut() {
                    if s.is_finite() {
                        *s = (*s - max_score).exp();
                        sum = sum + *s;
                    } else {
                        *s = T::zero();
                    }
                }
                let inv_sum = T::one() / sum;
                // out[qi, :] = sum_j (scores[j] * inv_sum) * v[kv_h, j, :]
                for kj in 0..sk {
                    let p_ij = scores[kj] * inv_sum;
                    if p_ij == T::zero() { continue; }
                    let v_row = &v_data[v_off + kj * d_v .. v_off + (kj + 1) * d_v];
                    for (od, vd) in
                        out[o_off + qi * d_q .. o_off + (qi + 1) * d_q]
                            .iter_mut()
                            .zip(v_row.iter())
                    {
                        *od = *od + p_ij * (*vd);
                    }
                }
            }
        }
    }

    RefTensor::from_vec(out, Shape::from_dims(&[b_q, h_q, sq, d_q]))
}

/// FlashAttention-v2 forward in pure Rust. Tiles K/V into blocks of
/// `BC` columns at a time, runs online softmax over them. The full
/// `[B, H, Sq, Sk]` attention matrix is never materialized — only
/// per-Q-row running statistics `m` (running max) and `l` (running
/// denominator) plus the partial output `O`.
///
/// Output is bit-for-bit equal to [`attention_naive`] up to f32
/// associativity drift in the partial sums. Tier 2/3 backends must
/// match this within the same drift envelope.
///
/// `BR` (Q-row block size) and `BC` (K-col block size) are
/// constants — small for cache friendliness, but correctness is
/// independent of the choice.
pub fn attention_flash<T: Float>(
    q: &RefTensor<T>,
    k: &RefTensor<T>,
    v: &RefTensor<T>,
    alibi_slopes: Option<&RefTensor<T>>,
    p: &AttentionParams,
) -> RefTensor<T> {
    const BR: usize = 16;
    const BC: usize = 16;

    let (b_q, h_q, sq, d_q) = dims_bhsd(q, "attention_flash q");
    let (b_k, h_k, sk, d_k) = dims_bhsd(k, "attention_flash k");
    let (b_v, h_v, sk_v, d_v) = dims_bhsd(v, "attention_flash v");
    assert_eq!(b_q, b_k);
    assert_eq!(b_k, b_v);
    assert_eq!(h_k, h_v);
    assert_eq!(sk, sk_v);
    assert_eq!(d_q, d_k);
    assert_eq!(d_k, d_v);
    assert_eq!(h_q % h_k, 0);
    let groups = h_q / h_k;
    if let Some(slopes) = alibi_slopes {
        assert_eq!(slopes.shape().dims(), &[h_q]);
    }

    let q_data = q.as_slice();
    let k_data = k.as_slice();
    let v_data = v.as_slice();
    let alibi_data = alibi_slopes.map(|t| t.as_slice());

    let scale = T::from(p.softmax_scale).expect("softmax_scale must convert to T");
    let softcap = p.softcap.and_then(|c| T::from(c).map(|t| (t, T::one() / t)));

    let q_h_stride = sq * d_q;
    let q_b_stride = h_q * q_h_stride;
    let k_h_stride = sk * d_k;
    let k_b_stride = h_k * k_h_stride;

    let mut out = vec![T::zero(); b_q * h_q * sq * d_q];

    for bi in 0..b_q {
        for hi in 0..h_q {
            let kv_h = hi / groups;
            let q_off = bi * q_b_stride + hi * q_h_stride;
            let k_off = bi * k_b_stride + kv_h * k_h_stride;
            let v_off = k_off;
            let o_off = q_off; // same indexing as Q (B, Hq stride)
            let alibi_h = alibi_data.map(|a| a[hi]);

            // Process Q rows in BR-sized tiles. Online softmax state
            // (m_i, l_i) and partial output O_i are local to each Q
            // row — initialize per row, finalize after K/V sweep.
            for q_tile_start in (0..sq).step_by(BR) {
                let q_tile_end = (q_tile_start + BR).min(sq);
                let q_tile_rows = q_tile_end - q_tile_start;

                // Per-Q-row state for this tile.
                let mut m = vec![T::neg_infinity(); q_tile_rows];
                let mut l = vec![T::zero(); q_tile_rows];
                let mut o_acc = vec![T::zero(); q_tile_rows * d_q];

                for k_tile_start in (0..sk).step_by(BC) {
                    let k_tile_end = (k_tile_start + BC).min(sk);
                    let k_tile_cols = k_tile_end - k_tile_start;

                    // 1. Compute S_ij = Q_i K_j^T * scale (+ softcap +
                    //    alibi) and per-row max over this K tile.
                    let mut s = vec![T::neg_infinity(); q_tile_rows * k_tile_cols];
                    let mut tile_max = vec![T::neg_infinity(); q_tile_rows];
                    for qi_local in 0..q_tile_rows {
                        let qi = q_tile_start + qi_local;
                        let q_row = &q_data[q_off + qi * d_q .. q_off + (qi + 1) * d_q];
                        for kj_local in 0..k_tile_cols {
                            let kj = k_tile_start + kj_local;
                            if !position_admissible(qi, kj, p) {
                                continue;
                            }
                            let k_row = &k_data[k_off + kj * d_k .. k_off + (kj + 1) * d_k];
                            let mut acc = T::zero();
                            for (qx, kx) in q_row.iter().zip(k_row.iter()) {
                                acc = acc + (*qx) * (*kx);
                            }
                            let mut s_ij = acc * scale;
                            if let Some((c, inv_c)) = softcap {
                                s_ij = (s_ij * inv_c).tanh() * c;
                            }
                            if let Some(slope) = alibi_h {
                                let delta = T::from(kj as f32 - qi as f32)
                                    .expect("alibi delta must convert to T");
                                s_ij = s_ij + slope * delta;
                            }
                            s[qi_local * k_tile_cols + kj_local] = s_ij;
                            if s_ij > tile_max[qi_local] {
                                tile_max[qi_local] = s_ij;
                            }
                        }
                    }

                    // 2. Online softmax update per Q row.
                    for qi_local in 0..q_tile_rows {
                        let m_old = m[qi_local];
                        let m_new = if tile_max[qi_local] > m_old {
                            tile_max[qi_local]
                        } else {
                            m_old
                        };
                        if !m_new.is_finite() {
                            // No admissible keys for this Q row in any tile yet —
                            // skip.
                            continue;
                        }
                        let scale_old = if m_old.is_finite() {
                            (m_old - m_new).exp()
                        } else {
                            T::zero()
                        };
                        // Recompute P_ij = exp(S_ij - m_new), accumulate row sum.
                        let mut row_sum = T::zero();
                        for kj_local in 0..k_tile_cols {
                            let s_ij = s[qi_local * k_tile_cols + kj_local];
                            let p_ij = if s_ij.is_finite() {
                                (s_ij - m_new).exp()
                            } else {
                                T::zero()
                            };
                            s[qi_local * k_tile_cols + kj_local] = p_ij;
                            row_sum = row_sum + p_ij;
                        }
                        l[qi_local] = scale_old * l[qi_local] + row_sum;

                        // O_i = scale_old · O_i + P_ij · V_j (block-vector matmul).
                        let o_row = &mut o_acc[qi_local * d_q .. (qi_local + 1) * d_q];
                        for od in o_row.iter_mut() {
                            *od = *od * scale_old;
                        }
                        for kj_local in 0..k_tile_cols {
                            let p_ij = s[qi_local * k_tile_cols + kj_local];
                            if p_ij == T::zero() { continue; }
                            let kj = k_tile_start + kj_local;
                            let v_row = &v_data[v_off + kj * d_v .. v_off + (kj + 1) * d_v];
                            for (od, vd) in o_row.iter_mut().zip(v_row.iter()) {
                                *od = *od + p_ij * (*vd);
                            }
                        }
                        m[qi_local] = m_new;
                    }
                }

                // Finalize: divide each Q row's accumulator by its denominator.
                for qi_local in 0..q_tile_rows {
                    if l[qi_local] == T::zero() {
                        continue; // no admissible keys -> output zero
                    }
                    let qi = q_tile_start + qi_local;
                    let inv_l = T::one() / l[qi_local];
                    let o_src = &o_acc[qi_local * d_q .. (qi_local + 1) * d_q];
                    let o_dst_off = o_off + qi * d_q;
                    for (out_d, &od) in
                        out[o_dst_off .. o_dst_off + d_q].iter_mut().zip(o_src.iter())
                    {
                        *out_d = od * inv_l;
                    }
                }
            }
        }
    }

    RefTensor::from_vec(out, Shape::from_dims(&[b_q, h_q, sq, d_q]))
}

/// Backward of attention via recompute. Given Q, K, V, the forward
/// output O, and the upstream gradient dO, returns (dQ, dK, dV).
///
/// Recompute approach: re-run forward to get per-row softmax state
/// `(m, l)`, then compute gradient walks. Costs an extra forward
/// pass, saves O(N²) memory of the attention matrix. Matches what
/// real FA backward kernels do.
///
/// dV[b, h_kv, j, :] = Σ_i P[b, h, i, j] · dO[b, h, i, :]   (summed over groups)
/// dP[b, h, i, j]    = dO[b, h, i, :] · V[b, h_kv, j, :]
/// dS[b, h, i, j]    = (dP[b, h, i, j] - Σ_j' P[..., j'] · dP[..., j']) · P[b, h, i, j]
/// dQ[b, h, i, :]    = scale · Σ_j dS[b, h, i, j] · K[b, h_kv, j, :]
/// dK[b, h_kv, j, :] = scale · Σ_i dS[b, h, i, j] · Q[b, h, i, :]   (summed over groups)
pub fn attention_flash_backward<T: Float>(
    q: &RefTensor<T>,
    k: &RefTensor<T>,
    v: &RefTensor<T>,
    do_grad: &RefTensor<T>,
    alibi_slopes: Option<&RefTensor<T>>,
    p: &AttentionParams,
) -> (RefTensor<T>, RefTensor<T>, RefTensor<T>) {
    let (b_q, h_q, sq, d_q) = dims_bhsd(q, "attention_flash_backward q");
    let (_, h_k, sk, _) = dims_bhsd(k, "attention_flash_backward k");
    let (_, _, _, d_v) = dims_bhsd(v, "attention_flash_backward v");
    let groups = h_q / h_k;

    let q_data = q.as_slice();
    let k_data = k.as_slice();
    let v_data = v.as_slice();
    let do_data = do_grad.as_slice();
    let alibi_data = alibi_slopes.map(|t| t.as_slice());

    let scale = T::from(p.softmax_scale).expect("softmax_scale must convert to T");
    let softcap = p.softcap.and_then(|c| T::from(c).map(|t| (t, T::one() / t)));

    let q_h_stride = sq * d_q;
    let q_b_stride = h_q * q_h_stride;
    let k_h_stride = sk * d_q;
    let k_b_stride = h_k * k_h_stride;

    let mut dq = vec![T::zero(); b_q * h_q * sq * d_q];
    let mut dk = vec![T::zero(); b_q * h_k * sk * d_q];
    let mut dv = vec![T::zero(); b_q * h_k * sk * d_v];

    for bi in 0..b_q {
        for hi in 0..h_q {
            let kv_h = hi / groups;
            let q_off = bi * q_b_stride + hi * q_h_stride;
            let k_off = bi * k_b_stride + kv_h * k_h_stride;
            let v_off = k_off;
            let do_off = q_off;
            let dq_off = q_off;
            let dk_off = k_off;
            let dv_off = k_off;
            let alibi_h = alibi_data.map(|a| a[hi]);

            for qi in 0..sq {
                // Recompute S_ij and softmax P[qi, :] for this row.
                let mut p_row = vec![T::zero(); sk];
                let mut max_s = T::neg_infinity();
                let mut s_row = vec![T::neg_infinity(); sk];
                let mut s_pre_softcap = vec![T::neg_infinity(); sk];
                for kj in 0..sk {
                    if !position_admissible(qi, kj, p) { continue; }
                    let q_row = &q_data[q_off + qi * d_q .. q_off + (qi + 1) * d_q];
                    let k_row = &k_data[k_off + kj * d_q .. k_off + (kj + 1) * d_q];
                    let mut acc = T::zero();
                    for (qx, kx) in q_row.iter().zip(k_row.iter()) {
                        acc = acc + (*qx) * (*kx);
                    }
                    let mut s = acc * scale;
                    s_pre_softcap[kj] = s;
                    if let Some((c, inv_c)) = softcap {
                        s = (s * inv_c).tanh() * c;
                    }
                    if let Some(slope) = alibi_h {
                        let delta = T::from(kj as f32 - qi as f32)
                            .expect("alibi delta must convert to T");
                        s = s + slope * delta;
                    }
                    s_row[kj] = s;
                    if s > max_s { max_s = s; }
                }
                if !max_s.is_finite() { continue; }
                let mut sum_exp = T::zero();
                for kj in 0..sk {
                    if s_row[kj].is_finite() {
                        let e = (s_row[kj] - max_s).exp();
                        p_row[kj] = e;
                        sum_exp = sum_exp + e;
                    }
                }
                let inv_sum = T::one() / sum_exp;
                for kj in 0..sk {
                    p_row[kj] = p_row[kj] * inv_sum;
                }

                // dV[..., j, :] += P[i, j] · dO[i, :]
                let do_row = &do_data[do_off + qi * d_v .. do_off + (qi + 1) * d_v];
                for kj in 0..sk {
                    let p_ij = p_row[kj];
                    if p_ij == T::zero() { continue; }
                    let dst_off = dv_off + kj * d_v;
                    for (dvd, &dod) in dv[dst_off .. dst_off + d_v].iter_mut().zip(do_row.iter()) {
                        *dvd = *dvd + p_ij * dod;
                    }
                }

                // dP[i, j] = dO[i, :] · V[j, :]
                let mut dp = vec![T::zero(); sk];
                for kj in 0..sk {
                    let v_row = &v_data[v_off + kj * d_v .. v_off + (kj + 1) * d_v];
                    let mut acc = T::zero();
                    for (dod, &vd) in do_row.iter().zip(v_row.iter()) {
                        acc = acc + (*dod) * vd;
                    }
                    dp[kj] = acc;
                }

                // softmax backward: dS[i, j] = (dP[i, j] - Σ_j' P[i, j'] · dP[i, j']) · P[i, j]
                let mut row_dot = T::zero();
                for kj in 0..sk {
                    row_dot = row_dot + p_row[kj] * dp[kj];
                }
                let mut ds = vec![T::zero(); sk];
                for kj in 0..sk {
                    ds[kj] = (dp[kj] - row_dot) * p_row[kj];
                    if let Some((_c, inv_c)) = softcap {
                        // d/dx (tanh(x/c) * c) = sech²(x/c) = 1 - tanh²(x/c)
                        let pre = s_pre_softcap[kj];
                        if pre.is_finite() {
                            let t = (pre * inv_c).tanh();
                            let dtanh = T::one() - t * t;
                            ds[kj] = ds[kj] * dtanh;
                        }
                    }
                }

                // dQ[i, :] += scale · Σ_j dS[i, j] · K[j, :]
                let dq_row_off = dq_off + qi * d_q;
                for kj in 0..sk {
                    let ds_ij = ds[kj] * scale;
                    if ds_ij == T::zero() { continue; }
                    let k_row = &k_data[k_off + kj * d_q .. k_off + (kj + 1) * d_q];
                    for (dqd, &kd) in dq[dq_row_off .. dq_row_off + d_q].iter_mut().zip(k_row.iter()) {
                        *dqd = *dqd + ds_ij * kd;
                    }
                }
                // dK[j, :] += scale · Σ_i dS[i, j] · Q[i, :]
                let q_row = &q_data[q_off + qi * d_q .. q_off + (qi + 1) * d_q];
                for kj in 0..sk {
                    let ds_ij = ds[kj] * scale;
                    if ds_ij == T::zero() { continue; }
                    let dst_off = dk_off + kj * d_q;
                    for (dkd, &qd) in dk[dst_off .. dst_off + d_q].iter_mut().zip(q_row.iter()) {
                        *dkd = *dkd + ds_ij * qd;
                    }
                }
            }
        }
    }

    (
        RefTensor::from_vec(dq, Shape::from_dims(&[b_q, h_q, sq, d_q])),
        RefTensor::from_vec(dk, Shape::from_dims(&[b_q, h_k, sk, d_q])),
        RefTensor::from_vec(dv, Shape::from_dims(&[b_q, h_k, sk, d_v])),
    )
}

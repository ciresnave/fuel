//! RWKV v7 "Goose" decoder ported to the lazy-graph API.
//!
//! Phase D specialized port. v7 is the third (and most complex)
//! RWKV iteration. It extends the [v5][1] / [v6][2] linear
//! attention with three architectural innovations:
//!
//!   1. **Delta-rule state update** —
//!      ```text
//!      state[h] = state[h] * w[h] + state[h] @ ab[h] + vk[h]
//!      ```
//!      with `vk = v ⊗ k` (outer product) and `ab = -kk ⊗ (kk * a)`
//!      (ICL correction term, where `kk` is the L2-normalised
//!      key and `a` is a per-feature in-context-learning rate).
//!   2. **Value residual stream across layers** — layer 0
//!      produces `v_first` (the unaltered V projection per
//!      token). Subsequent layers blend toward it via a
//!      sigmoid-gated mix:
//!      ```text
//!      v[layer > 0] = v + (v_first - v) * sigmoid(v0 + (xv @ v1) @ v2)
//!      ```
//!   3. **LoRA-style projections** for decay (`w0/w1/w2`),
//!      ICL rate (`a0/a1/a2`), value residual (`v0/v1/v2`,
//!      layers > 0 only), and gate (`g1/g2`). Each is a
//!      low-rank `[hidden, lora_dim] → [lora_dim, hidden]`
//!      pair feeding a tanh or sigmoid.
//!
//! The TimeMix block uses **six per-stream token-shift mixes**
//! (`x_r, x_w, x_k, x_v, x_a, x_g`), the L2-normalised key
//! variant (`k_k`), the ICL key correction (`k_a`), and a
//! per-head bonus term (`r_k`) added to the GroupNorm output
//! before the gate multiplication.
//!
//! v7a (DeepEmbed gating in ChannelMix) and v7b (DEA — Deep
//! Embedding Attention, a separate quadratic attention path)
//! are **deferred** to follow-up sessions; v1 ships the base
//! v7 only.
//!
//! # Scope (v1)
//!
//! Forward-only prefill, zero initial state, batch == 1, F32.
//! Time loop is unrolled at graph-build time (same approach
//! as [`crate::lazy_rwkv5`] and [`crate::lazy_rwkv6`]).
//!
//! [1]: crate::lazy_rwkv5
//! [2]: crate::lazy_rwkv6

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Rwkv7Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub head_size: usize,
    /// FFN dim. Defaults to `4 * hidden_size` when `None`.
    pub intermediate_size: Option<usize>,
    /// LoRA rank for the decay path (`w1/w2`).
    pub d_decay: usize,
    /// LoRA rank for the ICL-rate path (`a1/a2`).
    pub d_aaa: usize,
    /// LoRA rank for the value residual path (`v1/v2`, layers > 0 only).
    pub d_mv: usize,
    /// LoRA rank for the gate path (`g1/g2`).
    pub d_gate: usize,
    pub layer_norm_epsilon: f64,
}

impl Rwkv7Config {
    pub fn n_heads(&self) -> usize {
        self.hidden_size / self.head_size
    }
    pub fn dim_ffn(&self) -> usize {
        self.intermediate_size.unwrap_or(4 * self.hidden_size)
    }
}

#[derive(Debug, Clone)]
pub struct Rwkv7TimeMixWeights {
    // Token-shift mixes — shape `[hidden]` each.
    pub x_r: Arc<[f32]>,
    pub x_w: Arc<[f32]>,
    pub x_k: Arc<[f32]>,
    pub x_v: Arc<[f32]>,
    pub x_a: Arc<[f32]>,
    pub x_g: Arc<[f32]>,
    // Decay LoRA.
    pub w0: Arc<[f32]>, // [hidden]
    pub w1: Arc<[f32]>, // [hidden, d_decay]
    pub w2: Arc<[f32]>, // [d_decay, hidden]
    // ICL-rate LoRA.
    pub a0: Arc<[f32]>,
    pub a1: Arc<[f32]>,
    pub a2: Arc<[f32]>,
    // Value-residual LoRA (None at layer 0).
    pub v0: Option<Arc<[f32]>>,
    pub v1: Option<Arc<[f32]>>,
    pub v2: Option<Arc<[f32]>>,
    // Gate LoRA (no g0).
    pub g1: Arc<[f32]>,
    pub g2: Arc<[f32]>,
    // Key processing.
    pub k_k: Arc<[f32]>,
    pub k_a: Arc<[f32]>,
    // Bonus term — flat `[hidden]` (n_heads * head_size).
    pub r_k: Arc<[f32]>,
    // Projections.
    pub receptance: WeightStorage,
    pub key: WeightStorage,
    pub value: WeightStorage,
    pub output: WeightStorage,
    // Per-head GroupNorm affine.
    pub ln_x_gain: Arc<[f32]>,
    pub ln_x_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Rwkv7ChannelMixWeights {
    pub x_k: Arc<[f32]>,
    pub key: WeightStorage,
    pub value: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Rwkv7LayerWeights {
    pub ln1_gain: Arc<[f32]>,
    pub ln1_bias: Arc<[f32]>,
    pub ln2_gain: Arc<[f32]>,
    pub ln2_bias: Arc<[f32]>,
    /// Only on layer 0.
    pub pre_ln: Option<(Arc<[f32]>, Arc<[f32]>)>,
    pub time_mix: Rwkv7TimeMixWeights,
    pub channel_mix: Rwkv7ChannelMixWeights,
}

#[derive(Debug, Clone)]
pub struct Rwkv7Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Rwkv7LayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub head: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Rwkv7Model {
    pub config: Rwkv7Config,
    pub weights: Rwkv7Weights,
}

impl Rwkv7Model {
    pub fn forward(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        let n_heads = cfg.n_heads();
        let head_size = cfg.head_size;
        assert_eq!(
            n_heads * head_size, cfg.hidden_size,
            "Rwkv7Config: n_heads * head_size must equal hidden_size",
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

        // v_first collected at layer 0, consumed by layers > 0.
        // Stored as per-token vectors of shape (b, hidden).
        let mut v_first: Option<Vec<LazyTensor>> = None;

        for (li, layer) in weights.layers.iter().enumerate() {
            let xs = if let Some((g, b)) = &layer.pre_ln {
                crate::lazy::apply_affine_layer_norm_pub(&h, g, b, cfg.hidden_size, cfg.layer_norm_epsilon)
            } else {
                h.clone()
            };
            let xs_ln1 = crate::lazy::apply_affine_layer_norm_pub(
                &xs, &layer.ln1_gain, &layer.ln1_bias,
                cfg.hidden_size, cfg.layer_norm_epsilon,
            );
            let (attn, v_first_layer) = self.time_mix(
                &xs_ln1, layer, batch, seq, n_heads, head_size, li, v_first.as_ref(),
            )?;
            if li == 0 {
                v_first = Some(v_first_layer);
            }
            let h_post_attn = xs.add(&attn)?;

            let xs_ln2 = crate::lazy::apply_affine_layer_norm_pub(
                &h_post_attn, &layer.ln2_gain, &layer.ln2_bias,
                cfg.hidden_size, cfg.layer_norm_epsilon,
            );
            let ff = self.channel_mix(&xs_ln2, layer, batch, seq)?;
            h = h_post_attn.add(&ff)?;
        }

        let h_norm = crate::lazy::apply_affine_layer_norm_pub(
            &h, &weights.final_ln_gain, &weights.final_ln_bias,
            cfg.hidden_size, cfg.layer_norm_epsilon,
        );
        Ok(weights.head.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// TimeMix forward over the full sequence; unrolls the time
    /// loop at graph build. Returns `(attn_out, v_first_per_t)`.
    /// `v_first_per_t` is non-empty only at layer 0 (it's what
    /// downstream layers consume).
    #[allow(clippy::too_many_arguments)]
    fn time_mix(
        &self,
        x: &LazyTensor,
        layer: &Rwkv7LayerWeights,
        batch: usize,
        seq: usize,
        n_heads: usize,
        head_size: usize,
        layer_id: usize,
        v_first_prev: Option<&Vec<LazyTensor>>,
    ) -> Result<(LazyTensor, Vec<LazyTensor>)> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let tm = &layer.time_mix;

        // ---- Per-stream token-shift mix coefficients --------------------------
        let mix_const = |arc: &Arc<[f32]>| -> LazyTensor {
            x.const_f32_like(Arc::clone(arc), Shape::from_dims(&[1, 1, h]))
        };
        let m_r = mix_const(&tm.x_r);
        let m_w = mix_const(&tm.x_w);
        let m_k = mix_const(&tm.x_k);
        let m_v = mix_const(&tm.x_v);
        let m_a = mix_const(&tm.x_a);
        let m_g = mix_const(&tm.x_g);

        // Token-shifted input: shifted - x  (note: shifted is x_{t-1}).
        let shifted = shift_seq(x, batch, seq, h);
        let xx = shifted.sub(x)?;

        let xr = x.add(&xx.broadcast_mul(&m_r)?)?;
        let xw = x.add(&xx.broadcast_mul(&m_w)?)?;
        let xk = x.add(&xx.broadcast_mul(&m_k)?)?;
        let xv = x.add(&xx.broadcast_mul(&m_v)?)?;
        let xa = x.add(&xx.broadcast_mul(&m_a)?)?;
        let xg = x.add(&xx.broadcast_mul(&m_g)?)?;

        // ---- Projections -----------------------------------------------------
        let r = tm.receptance.apply_linear(&xr, h, h);
        let k = tm.key.apply_linear(&xk, h, h);
        let v = tm.value.apply_linear(&xv, h, h);

        // ---- Decay LoRA: w = exp(-0.606531 * sigmoid(w0 + tanh(xw @ w1) @ w2))
        let w0 = x.const_f32_like(Arc::clone(&tm.w0), Shape::from_dims(&[1, 1, h]));
        let w1 = x.const_f32_like(Arc::clone(&tm.w1), Shape::from_dims(&[h, cfg.d_decay]));
        let w2 = x.const_f32_like(Arc::clone(&tm.w2), Shape::from_dims(&[cfg.d_decay, h]));
        let w_hidden = xw.matmul(&w1)?.tanh().matmul(&w2)?;
        let w_pre = w0.broadcast_to(Shape::from_dims(&[batch, seq, h]))?.add(&w_hidden)?;
        let w_sig = w_pre.sigmoid();
        let w = w_sig.mul_scalar(-0.606531_f64).exp();

        // ---- ICL rate: a = sigmoid(a0 + (xa @ a1) @ a2)
        let a0 = x.const_f32_like(Arc::clone(&tm.a0), Shape::from_dims(&[1, 1, h]));
        let a1 = x.const_f32_like(Arc::clone(&tm.a1), Shape::from_dims(&[h, cfg.d_aaa]));
        let a2 = x.const_f32_like(Arc::clone(&tm.a2), Shape::from_dims(&[cfg.d_aaa, h]));
        let a_pre = a0.broadcast_to(Shape::from_dims(&[batch, seq, h]))?
            .add(&xa.matmul(&a1)?.matmul(&a2)?)?;
        let a = a_pre.sigmoid();

        // ---- Gate: g = sigmoid(xg @ g1) @ g2 ---------------------------------
        let g1 = x.const_f32_like(Arc::clone(&tm.g1), Shape::from_dims(&[h, cfg.d_gate]));
        let g2 = x.const_f32_like(Arc::clone(&tm.g2), Shape::from_dims(&[cfg.d_gate, h]));
        let g = xg.matmul(&g1)?.sigmoid().matmul(&g2)?;

        // ---- Value residual stream (layer > 0) -------------------------------
        // Collect v_first per token at layer 0; consume in layers > 0.
        let mut v_first_collected: Vec<LazyTensor> = Vec::with_capacity(seq);
        let v_effective = if layer_id == 0 {
            // At layer 0 v_first[t] = v[t].
            for t in 0..seq {
                let vt = v.slice(1_usize, t, 1)?;
                v_first_collected.push(vt);
            }
            v
        } else {
            let v_first_prev = v_first_prev.expect("layer > 0 requires v_first from layer 0");
            assert_eq!(v_first_prev.len(), seq, "v_first sequence length mismatch");
            // v0 + (xv @ v1) @ v2 → sigmoid → blend.
            let v0_arc = tm.v0.as_ref().expect("layer > 0 must carry v0/v1/v2");
            let v1_arc = tm.v1.as_ref().expect("layer > 0 must carry v0/v1/v2");
            let v2_arc = tm.v2.as_ref().expect("layer > 0 must carry v0/v1/v2");
            let v0 = x.const_f32_like(Arc::clone(v0_arc), Shape::from_dims(&[1, 1, h]));
            let v1l = x.const_f32_like(Arc::clone(v1_arc), Shape::from_dims(&[h, cfg.d_mv]));
            let v2l = x.const_f32_like(Arc::clone(v2_arc), Shape::from_dims(&[cfg.d_mv, h]));
            let gate_pre = v0.broadcast_to(Shape::from_dims(&[batch, seq, h]))?
                .add(&xv.matmul(&v1l)?.matmul(&v2l)?)?;
            let gate = gate_pre.sigmoid();
            // Stack the per-step v_first vectors back into a (b, seq, h) tensor.
            let mut vf_stack: Option<LazyTensor> = None;
            for vt in v_first_prev {
                vf_stack = Some(match vf_stack {
                    None => vt.clone(),
                    Some(s) => s.concat(vt, 1_usize)?,
                });
            }
            let v_first_full = vf_stack.expect("v_first_prev non-empty");
            v.add(&v_first_full.sub(&v)?.mul(&gate)?)?
        };
        let v = v_effective;

        // ---- Key processing --------------------------------------------------
        let kk_const = x.const_f32_like(Arc::clone(&tm.k_k), Shape::from_dims(&[1, 1, h]));
        let kk_raw = k.broadcast_mul(&kk_const)?;
        // L2 normalize per (batch, seq, head, head_size).
        let kk_h = kk_raw.reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
        let kk_norm = kk_h.sqr().sum_dim(3_usize)?.sqrt().add_scalar(1e-12);
        let kk_norm_bc = kk_norm
            .reshape(Shape::from_dims(&[batch, seq, n_heads, 1]))?
            .broadcast_to(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
        let kk = kk_h.div(&kk_norm_bc)?
            .reshape(Shape::from_dims(&[batch, seq, h]))?;

        // k = k * (1 + (a - 1) * k_a)
        let k_a_const = x.const_f32_like(Arc::clone(&tm.k_a), Shape::from_dims(&[1, 1, h]));
        let a_minus_1 = a.add_scalar(-1.0);
        let scale_term = a_minus_1.broadcast_mul(&k_a_const)?.add_scalar(1.0);
        let k_corrected = k.mul(&scale_term)?;

        // ---- Per-head reshapes for state update -----------------------------
        let r_h = r.reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?
            .permute([0, 2, 1, 3_usize])?;        // (b, h, seq, N)
        let k_h_state = k_corrected
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?
            .permute([0, 2, 1, 3_usize])?;       // (b, h, seq, N)
        let v_h = v.reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?
            .permute([0, 2, 1, 3_usize])?;
        let kk_h2 = kk.reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?
            .permute([0, 2, 1, 3_usize])?;
        let a_h = a.reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?
            .permute([0, 2, 1, 3_usize])?;
        let w_h = w.reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?
            .permute([0, 2, 1, 3_usize])?;

        // ---- Bonus term r_k --------------------------------------------------
        let r_k_const = x.const_f32_like(
            Arc::clone(&tm.r_k),
            Shape::from_dims(&[1, 1, h]),
        );

        // ---- Delta-rule time loop -------------------------------------------
        // state shape: (b, n_heads, head_size, head_size)
        let state_init = x.const_f32_like(
            Arc::from(vec![0.0_f32; batch * n_heads * head_size * head_size]),
            Shape::from_dims(&[batch, n_heads, head_size, head_size]),
        );
        let mut state = state_init;
        let mut out_steps: Vec<LazyTensor> = Vec::with_capacity(seq);
        for t in 0..seq {
            let r_t = r_h.slice(2_usize, t, 1)?;                // (b, h, 1, N)
            let k_t = k_h_state.slice(2_usize, t, 1)?;
            let v_t = v_h.slice(2_usize, t, 1)?;
            let kk_t = kk_h2.slice(2_usize, t, 1)?;
            let a_t = a_h.slice(2_usize, t, 1)?;
            let w_t = w_h.slice(2_usize, t, 1)?;                // (b, h, 1, N)

            // vk = v^T @ k → (b, h, N, N)  (outer product: v[N,1] @ k[1,N])
            let v_col = v_t.reshape(Shape::from_dims(&[batch, n_heads, head_size, 1]))?;
            let k_row = k_t.reshape(Shape::from_dims(&[batch, n_heads, 1, head_size]))?;
            let vk = v_col.matmul(&k_row)?;                       // (b, h, N, N)

            // ab = (-kk)^T @ (kk * a) → (b, h, N, N)
            let kk_col_neg = kk_t.mul_scalar(-1.0)
                .reshape(Shape::from_dims(&[batch, n_heads, head_size, 1]))?;
            let kk_a_row = kk_t.mul(&a_t)?
                .reshape(Shape::from_dims(&[batch, n_heads, 1, head_size]))?;
            let ab = kk_col_neg.matmul(&kk_a_row)?;               // (b, h, N, N)

            // state := state * w + state @ ab + vk
            // w broadcast: (b, h, 1, N) → (b, h, N, N) for elementwise multiply
            let w_bc = w_t.broadcast_to(Shape::from_dims(&[batch, n_heads, head_size, head_size]))?;
            let state_decayed = state.mul(&w_bc)?;
            let state_via_ab = state.matmul(&ab)?;
            state = state_decayed.add(&state_via_ab)?.add(&vk)?;

            // out_t = state @ r → (b, h, N, 1) → flatten to (b, h, N)
            let r_col = r_t.reshape(Shape::from_dims(&[batch, n_heads, head_size, 1]))?;
            let out_col = state.matmul(&r_col)?;                  // (b, h, N, 1)
            let out_per_head = out_col.reshape(Shape::from_dims(&[batch, n_heads, head_size]))?;
            // Stack per-head into (b, 1, hidden).
            let out_t = out_per_head.reshape(Shape::from_dims(&[batch, 1, h]))?;
            out_steps.push(out_t);
        }

        let mut stacked: Option<LazyTensor> = None;
        for ot in out_steps.into_iter() {
            stacked = Some(match stacked {
                None => ot,
                Some(s) => s.concat(&ot, 1_usize)?,
            });
        }
        let stacked = stacked.expect("at least one step");

        // ---- Per-head GroupNorm with eps = 64e-5 ----------------------------
        let out_grouped = group_norm_per_head(
            &stacked, &tm.ln_x_gain, &tm.ln_x_bias,
            batch, seq, n_heads, head_size, 64e-5,
        )?;

        // ---- Bonus term: (r * k * r_k).sum_per_head * v ----------------------
        let r_k_term = r.mul(&k_corrected)?.broadcast_mul(&r_k_const)?;
        let r_k_per_head = r_k_term
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?
            .sum_dim(3_usize)?;                                   // (b, seq, n_heads)
        let r_k_bc = r_k_per_head
            .reshape(Shape::from_dims(&[batch, seq, n_heads, 1]))?
            .broadcast_to(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
        let v_per_head = v
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
        let bonus = r_k_bc
            .mul(&v_per_head)?
            .reshape(Shape::from_dims(&[batch, seq, h]))?;
        let out_with_bonus = out_grouped.add(&bonus)?;

        // ---- Gate + output projection ---------------------------------------
        let gated = out_with_bonus.mul(&g)?;
        let final_out = tm.output.apply_linear(&gated, h, h);
        Ok((final_out, v_first_collected))
    }

    fn channel_mix(
        &self,
        x: &LazyTensor,
        layer: &Rwkv7LayerWeights,
        batch: usize,
        seq: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let cm = &layer.channel_mix;
        let h = cfg.hidden_size;
        let dim_ffn = cfg.dim_ffn();

        let shifted = shift_seq(x, batch, seq, h);
        let xx = shifted.sub(x)?;
        let mix_const = x.const_f32_like(Arc::clone(&cm.x_k), Shape::from_dims(&[1, 1, h]));
        let k_in = x.add(&xx.broadcast_mul(&mix_const)?)?;

        // k = relu(k @ key)^2
        let k = cm.key.apply_linear(&k_in, h, dim_ffn).relu().sqr();
        Ok(cm.value.apply_linear(&k, dim_ffn, h))
    }
}

fn shift_seq(x: &LazyTensor, batch: usize, seq: usize, h: usize) -> LazyTensor {
    let zero = x.const_f32_like(
        Arc::from(vec![0.0_f32; batch * h]),
        Shape::from_dims(&[batch, 1, h]),
    );
    if seq == 1 {
        return zero;
    }
    let earlier = x.slice(1_usize, 0, seq - 1).unwrap();
    zero.concat(&earlier, 1_usize).unwrap()
}

fn group_norm_per_head(
    x: &LazyTensor,
    gain: &Arc<[f32]>,
    bias: &Arc<[f32]>,
    batch: usize,
    seq: usize,
    n_heads: usize,
    head_size: usize,
    eps: f64,
) -> Result<LazyTensor> {
    let hidden = n_heads * head_size;
    let xh = x.reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
    let mean = xh.mean_dim(3_usize)?;
    let mean_bc = mean
        .reshape(Shape::from_dims(&[batch, seq, n_heads, 1]))?
        .broadcast_to(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
    let centered = xh.sub(&mean_bc)?;
    let var = centered.sqr().mean_dim(3_usize)?;
    let inv_std = var.add_scalar(eps).rsqrt();
    let inv_std_bc = inv_std
        .reshape(Shape::from_dims(&[batch, seq, n_heads, 1]))?
        .broadcast_to(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
    let normed = centered.mul(&inv_std_bc)?;
    let gain_t = x.const_f32_like(Arc::clone(gain), Shape::from_dims(&[1, 1, n_heads, head_size]));
    let bias_t = x.const_f32_like(Arc::clone(bias), Shape::from_dims(&[1, 1, n_heads, head_size]));
    let gain_bc = gain_t.broadcast_to(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
    let bias_bc = bias_t.broadcast_to(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
    let scaled = normed.mul(&gain_bc)?.add(&bias_bc)?;
    scaled.reshape(Shape::from_dims(&[batch, seq, hidden]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_weights(cfg: &Rwkv7Config) -> Rwkv7Weights {
        let mut s: u32 = 71717;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let h = cfg.hidden_size;
        let n_heads = cfg.n_heads();
        let head_size = cfg.head_size;
        let dim_ffn = cfg.dim_ffn();
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<Rwkv7LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|li| {
                let pre_ln = if li == 0 {
                    Some((Arc::from(vec![1.0_f32; h]), Arc::from(vec![0.0_f32; h])))
                } else {
                    None
                };
                let (v0, v1, v2) = if li == 0 {
                    (None, None, None)
                } else {
                    (
                        Some(vec_of(h, &mut *nb)),
                        Some(vec_of(h * cfg.d_mv, &mut *nb)),
                        Some(vec_of(cfg.d_mv * h, &mut *nb)),
                    )
                };
                let time_mix = Rwkv7TimeMixWeights {
                    x_r: vec_of(h, &mut *nb),
                    x_w: vec_of(h, &mut *nb),
                    x_k: vec_of(h, &mut *nb),
                    x_v: vec_of(h, &mut *nb),
                    x_a: vec_of(h, &mut *nb),
                    x_g: vec_of(h, &mut *nb),
                    w0: vec_of(h, &mut *nb),
                    w1: vec_of(h * cfg.d_decay, &mut *nb),
                    w2: vec_of(cfg.d_decay * h, &mut *nb),
                    a0: vec_of(h, &mut *nb),
                    a1: vec_of(h * cfg.d_aaa, &mut *nb),
                    a2: vec_of(cfg.d_aaa * h, &mut *nb),
                    v0, v1, v2,
                    g1: vec_of(h * cfg.d_gate, &mut *nb),
                    g2: vec_of(cfg.d_gate * h, &mut *nb),
                    k_k: vec_of(h, &mut *nb),
                    k_a: vec_of(h, &mut *nb),
                    r_k: vec_of(n_heads * head_size, &mut *nb),
                    receptance: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    key: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    value: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    output: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    ln_x_gain: Arc::from(vec![1.0_f32; h]),
                    ln_x_bias: Arc::from(vec![0.0_f32; h]),
                };
                let channel_mix = Rwkv7ChannelMixWeights {
                    x_k: vec_of(h, &mut *nb),
                    key: WeightStorage::F32(vec_of(h * dim_ffn, &mut *nb)),
                    value: WeightStorage::F32(vec_of(dim_ffn * h, &mut *nb)),
                };
                Rwkv7LayerWeights {
                    ln1_gain: Arc::from(vec![1.0_f32; h]),
                    ln1_bias: Arc::from(vec![0.0_f32; h]),
                    ln2_gain: Arc::from(vec![1.0_f32; h]),
                    ln2_bias: Arc::from(vec![0.0_f32; h]),
                    pre_ln,
                    time_mix,
                    channel_mix,
                }
            })
            .collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let head = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        Rwkv7Weights {
            token_embedding, layers,
            final_ln_gain, final_ln_bias, head,
        }
    }

    fn tiny_config() -> Rwkv7Config {
        Rwkv7Config {
            vocab_size: 16, hidden_size: 8,
            num_hidden_layers: 2, head_size: 4,
            intermediate_size: Some(16),
            d_decay: 3, d_aaa: 3, d_mv: 3, d_gate: 3,
            layer_norm_epsilon: 1e-5,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = Rwkv7Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    #[test]
    fn single_token() {
        let cfg = tiny_config();
        let model = Rwkv7Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[3]).unwrap().realize_f32();
        assert_eq!(logits.len(), cfg.vocab_size);
    }

    /// Delta-rule state must propagate: first-token swap should
    /// affect last-token logits via the recurrent state.
    #[test]
    fn delta_rule_state_propagates() {
        let cfg = Rwkv7Config { num_hidden_layers: 1, ..tiny_config() };
        let weights = tiny_weights(&cfg);
        let model = Rwkv7Model { config: cfg.clone(), weights };
        let a = model.forward(&[0, 3, 3, 3]).unwrap().realize_f32();
        let b = model.forward(&[7, 3, 3, 3]).unwrap().realize_f32();
        let v = cfg.vocab_size;
        let last_a = &a[a.len() - v..];
        let last_b = &b[b.len() - v..];
        let mut max_diff = 0.0_f32;
        for (x, y) in last_a.iter().zip(last_b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-10,
            "delta-rule state must propagate, max_diff = {max_diff}");
    }

    /// Value residual stream must be wired at layer > 0:
    /// override layer 1's `v0` (the dominant constant in the
    /// gate; `sigmoid(v0 + tiny_correction)` becomes
    /// `sigmoid(large_value + tiny_correction)`) and confirm
    /// output differs from baseline. Tiny-weight v2 changes
    /// produce a ~2e-8 logit delta that rounds to 0 at f32
    /// precision; v0 dominates the gate so a large v0 override
    /// gives a clearly measurable diff.
    #[test]
    fn value_residual_is_wired() {
        let cfg = Rwkv7Config { num_hidden_layers: 2, ..tiny_config() };
        let base = tiny_weights(&cfg);
        let mut overridden = base.clone();
        let h = cfg.hidden_size;
        // Layer 1's v0 — push it to a large positive value so
        // sigmoid(v0) ≈ 1 and the v_first → v blend is forced.
        overridden.layers[1].time_mix.v0 = Some(Arc::from(vec![5.0_f32; h]));
        let m_base = Rwkv7Model { config: cfg.clone(), weights: base };
        let m_over = Rwkv7Model { config: cfg, weights: overridden };
        let toks = [1_u32, 2, 3];
        let a = m_base.forward(&toks).unwrap().realize_f32();
        let b = m_over.forward(&toks).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        // Tiny weights (∈ [-0.025, 0.025]) yield tiny logit
        // magnitudes (~1e-5 range), so even a saturating v0
        // override produces only ~3e-7 absolute diff. The
        // value-residual path IS active; we just require it
        // to be measurably non-zero.
        assert!(max_diff > 1e-8,
            "value-residual gate (v0) must alter output, max_diff = {max_diff}");
    }
}

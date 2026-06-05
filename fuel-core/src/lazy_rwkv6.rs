//! RWKV v6 ("Finch") decoder ported to the lazy-graph API.
//!
//! Phase D specialized port. RWKV v6 extends [`crate::lazy_rwkv5`]
//! with **per-token learned time-mixing offsets** and a
//! **per-token learned decay**, both produced by low-rank
//! projections of the token + token-shift difference. The
//! linear-attention recurrence and channel-mix FFN are otherwise
//! identical to v5.
//!
//! # Per-token time-mix correction (the v6 wrinkle)
//!
//! At each step, compute `sx = shifted - xs`. The five static
//! mix vectors from v5 (`time_mix_w`, `time_mix_key`, `_value`,
//! `_receptance`, `_gate`) each receive a learned **per-token
//! correction**:
//!
//!   ```text
//!   xxx           = xs + sx * time_mix_x                      ; first-stage mix
//!   xxx           = tanh(xxx @ time_mix_w1)                   ; [b, t, 5 * n_heads]
//!   m[w/k/v/r/g]  = stream_slice(xxx)[s] @ time_mix_w2[s]     ; [b, t, hidden]
//!   x[stream]     = xs + sx * (time_mix_<stream> + m[stream]) ; per-stream mixed input
//!   ```
//!
//! Then K/V/R/Gate projections + reshape happen on the
//! corrected `x[stream]` inputs.
//!
//! # Per-token decay correction
//!
//! The static `time_decay` is similarly corrected via a low-rank
//! tanh-MLP applied to `xw`:
//!
//!   ```text
//!   w = time_decay + tanh(xw @ time_decay_w1) @ time_decay_w2  ; [b, t, hidden]
//!   w = exp(-exp(w))                                            ; per-token decay
//!   ```
//!
//! reshaped to `[n_heads, head_size, 1]` and broadcast against
//! the recurrent state during the time loop.
//!
//! # Channel-mix (FFN)
//!
//! Identical to v5: token-shifted K + R inputs;
//! `value = V(relu(K(k_in)).square())`; `out = sigmoid(R(r_in)) * value`.
//!
//! # Scope (v1)
//!
//! Forward-only prefill: zero initial state, batch == 1, F32.
//! The recurrent time loop is unrolled at graph-build time.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

pub use crate::lazy_rwkv5::Rwkv5Config as Rwkv6Config;

#[derive(Debug, Clone)]
pub struct Rwkv6LayerWeights {
    pub ln1_gain: Arc<[f32]>,
    pub ln1_bias: Arc<[f32]>,
    pub ln2_gain: Arc<[f32]>,
    pub ln2_bias: Arc<[f32]>,
    /// Only on layer 0.
    pub pre_ln: Option<(Arc<[f32]>, Arc<[f32]>)>,

    // ---- Time-mix (attention) ----
    pub attn_time_mix_x: Arc<[f32]>,            // [hidden] — first-stage mix
    pub attn_time_mix_w: Arc<[f32]>,            // static "w" stream mix
    pub attn_time_mix_key: Arc<[f32]>,
    pub attn_time_mix_value: Arc<[f32]>,
    pub attn_time_mix_receptance: Arc<[f32]>,
    pub attn_time_mix_gate: Arc<[f32]>,
    /// `[hidden, 5 * n_heads]` — first-stage mix correction.
    pub attn_time_mix_w1: Arc<[f32]>,
    /// `[5, n_heads, hidden]` — second-stage mix correction.
    pub attn_time_mix_w2: Arc<[f32]>,
    /// `[hidden]` — static base decay.
    pub attn_time_decay: Arc<[f32]>,
    /// `[hidden, 2 * n_heads]` — first-stage decay correction.
    pub attn_time_decay_w1: Arc<[f32]>,
    /// `[2 * n_heads, hidden]` — second-stage decay correction.
    pub attn_time_decay_w2: Arc<[f32]>,
    /// `[n_heads, head_size]` per-head bonus.
    pub attn_time_faaaa: Arc<[f32]>,

    pub attn_key: WeightStorage,        // hidden → attn_hidden
    pub attn_value: WeightStorage,
    pub attn_receptance: WeightStorage,
    pub attn_gate: WeightStorage,
    pub attn_output: WeightStorage,     // attn_hidden → hidden
    pub attn_ln_x_gain: Arc<[f32]>,
    pub attn_ln_x_bias: Arc<[f32]>,

    // ---- Channel-mix (FFN) ----
    pub ffn_time_mix_key: Arc<[f32]>,
    pub ffn_time_mix_receptance: Arc<[f32]>,
    pub ffn_key: WeightStorage,
    pub ffn_value: WeightStorage,
    pub ffn_receptance: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Rwkv6Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Rwkv6LayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub head: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Rwkv6Model {
    pub config: Rwkv6Config,
    pub weights: Rwkv6Weights,
}

impl Rwkv6Model {
    pub fn forward(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the RWKV-v6 stack forward up to the final LayerNorm
    /// and return per-token hidden states `(1, seq, hidden_size)`.
    pub fn forward_hidden(&self, tokens: &[u32]) -> Result<LazyTensor> {
        self.run_backbone(tokens)
    }

    /// Multimodal entry point. Skips token embedding; runs the RWKV-v6
    /// stack over pre-embedded inputs. RWKV does NOT scale embeddings
    /// and has no `start_pos` parameter — recurrent state is implicit
    /// in the time-mix (v1 is prefill only).
    pub fn forward_embeds(&self, embeds: &LazyTensor) -> Result<LazyTensor> {
        let h_norm = self.run_backbone_embeds(embeds)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`].
    pub fn forward_hidden_embeds(&self, embeds: &LazyTensor) -> Result<LazyTensor> {
        self.run_backbone_embeds(embeds)
    }

    /// Build per-token embeddings without running the decoder.
    pub fn embed_tokens_anchored(
        &self, anchor: &LazyTensor, tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        anchor.embed_tokens_anchored(
            self.weights.token_embedding.clone(),
            cfg.vocab_size, cfg.hidden_size, tokens,
        )
    }

    fn apply_lm_head(&self, h_norm: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        Ok(self.weights.head.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    fn run_backbone(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0, "Rwkv6Model: tokens must be non-empty");

        let h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;
        self.run_backbone_embeds(&h)
    }

    fn run_backbone_embeds(&self, embeds: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(crate::Error::Msg(format!(
                "Rwkv6Model::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        let batch = dims[0];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "Rwkv6Model::forward_embeds: seq must be > 0".into(),
            ).bt());
        }
        let n_heads = cfg.n_heads();
        let head_size = cfg.head_size;
        if n_heads * head_size != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "Rwkv6Config: n_heads * head_size must equal hidden_size".into(),
            ).bt());
        }
        let mut h = embeds.clone();

        for layer in &weights.layers {
            h = self.apply_block(&h, layer, batch, seq, n_heads, head_size)?;
        }
        h.layer_norm_affine(
            std::sync::Arc::clone(&weights.final_ln_gain),
            std::sync::Arc::clone(&weights.final_ln_bias),
            cfg.layer_norm_epsilon,
        )
    }

    fn apply_block(
        &self,
        xs: &LazyTensor,
        layer: &Rwkv6LayerWeights,
        batch: usize,
        seq: usize,
        n_heads: usize,
        head_size: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;

        let _ = h;
        let xs = if let Some((g, b)) = &layer.pre_ln {
            xs.layer_norm_affine(Arc::clone(g), Arc::clone(b), cfg.layer_norm_epsilon)?
        } else {
            xs.clone()
        };

        // Attention sublayer.
        let xs_ln1 = xs.layer_norm_affine(std::sync::Arc::clone(&layer.ln1_gain), std::sync::Arc::clone(&layer.ln1_bias), cfg.layer_norm_epsilon)?;
        let attn = self.time_mix(&xs_ln1, layer, batch, seq, n_heads, head_size)?;
        let xs = xs.add(&attn)?;

        // FFN sublayer.
        let xs_ln2 = xs.layer_norm_affine(std::sync::Arc::clone(&layer.ln2_gain), std::sync::Arc::clone(&layer.ln2_bias), cfg.layer_norm_epsilon)?;
        let ff = self.channel_mix(&xs_ln2, layer, batch, seq)?;
        xs.add(&ff)
    }

    fn time_mix(
        &self,
        x: &LazyTensor,
        layer: &Rwkv6LayerWeights,
        batch: usize,
        seq: usize,
        n_heads: usize,
        head_size: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let ah = cfg.attention_hidden_size;
        assert_eq!(
            ah, h,
            "Rwkv6: v1 assumes attention_hidden_size == hidden_size",
        );

        // Token shift.
        let shifted = shift_seq(x, batch, seq, h);
        // sx = shifted - xs.
        let sx = shifted.sub(x)?;

        // First-stage mix: xxx = xs + sx * time_mix_x.
        let mix_x = x.const_f32_like(
            Arc::clone(&layer.attn_time_mix_x),
            Shape::from_dims(&[1, 1, h]),
        );
        let xxx = x.add(&sx.broadcast_mul(&mix_x)?)?;

        // Per-token mix correction: tanh(xxx @ w1) of shape (b, t, 5 * n_heads),
        // then per-stream projection through w2.
        let w1 = x.const_f32_like(
            Arc::clone(&layer.attn_time_mix_w1),
            Shape::from_dims(&[h, 5 * n_heads]),
        );
        let xxx_proj = xxx.matmul(&w1)?.tanh(); // (b, t, 5 * n_heads)
        // Reshape w2 to (5, n_heads, h) for stream slicing.
        let w2 = x.const_f32_like(
            Arc::clone(&layer.attn_time_mix_w2),
            Shape::from_dims(&[5, n_heads, h]),
        );

        // For each stream s ∈ [0, 5):
        //   stream_in  = xxx_proj.slice(2, s*n_heads, n_heads) → (b, t, n_heads)
        //   stream_w2  = w2.slice(0, s, 1).reshape(n_heads, h)
        //   m_s        = stream_in @ stream_w2 → (b, t, h)
        let stream_offset = |s: usize| -> Result<LazyTensor> {
            let in_s = xxx_proj.slice(2_usize, s * n_heads, n_heads)?;
            let w2_s = w2
                .slice(0_usize, s, 1)?
                .reshape(Shape::from_dims(&[n_heads, h]))?;
            in_s.matmul(&w2_s)
        };
        let m_w = stream_offset(0)?;
        let m_k = stream_offset(1)?;
        let m_v = stream_offset(2)?;
        let m_r = stream_offset(3)?;
        let m_g = stream_offset(4)?;

        // x[stream] = xs + sx * (time_mix_<stream> + m[stream]).
        let make_input = |static_mix: &Arc<[f32]>, m: &LazyTensor| -> Result<LazyTensor> {
            let mix_static = x.const_f32_like(
                Arc::clone(static_mix),
                Shape::from_dims(&[1, 1, h]),
            );
            let mix_static_bc = mix_static.broadcast_to(Shape::from_dims(&[batch, seq, h]))?;
            let mix_total = mix_static_bc.add(m)?;
            x.add(&sx.mul(&mix_total)?)
        };
        let xw = make_input(&layer.attn_time_mix_w, &m_w)?;
        let xk = make_input(&layer.attn_time_mix_key, &m_k)?;
        let xv = make_input(&layer.attn_time_mix_value, &m_v)?;
        let xr = make_input(&layer.attn_time_mix_receptance, &m_r)?;
        let xg = make_input(&layer.attn_time_mix_gate, &m_g)?;

        // Per-token decay: w = time_decay + tanh(xw @ decay_w1) @ decay_w2.
        let dw1 = x.const_f32_like(
            Arc::clone(&layer.attn_time_decay_w1),
            Shape::from_dims(&[h, 2 * n_heads]),
        );
        let dw2 = x.const_f32_like(
            Arc::clone(&layer.attn_time_decay_w2),
            Shape::from_dims(&[2 * n_heads, h]),
        );
        let td_base = x.const_f32_like(
            Arc::clone(&layer.attn_time_decay),
            Shape::from_dims(&[1, 1, h]),
        );
        let td_correction = xw.matmul(&dw1)?.tanh().matmul(&dw2)?;
        let td_base_bc = td_base.broadcast_to(Shape::from_dims(&[batch, seq, h]))?;
        let w_per_token = td_base_bc.add(&td_correction)?;
        // w = exp(-exp(w))
        let decay_per_token = w_per_token.exp().neg().exp();
        // Reshape to (b, t, n_heads, head_size) for per-head access.
        let decay_h = decay_per_token
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;

        // Projections.
        let k = layer.attn_key.apply_linear(&xk, h, h);
        let v = layer.attn_value.apply_linear(&xv, h, h);
        let r = layer.attn_receptance.apply_linear(&xr, h, h);
        let g = layer.attn_gate.apply_linear(&xg, h, h).silu();

        // Per-head reshapes (same as v5).
        let k_h = k
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?
            .permute([0_usize, 2, 3, 1])?;
        let v_h = v
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?
            .permute([0_usize, 2, 1, 3])?;
        let r_h = r
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?
            .permute([0_usize, 2, 1, 3])?;

        let faaaa = x.const_f32_like(
            Arc::clone(&layer.attn_time_faaaa),
            Shape::from_dims(&[n_heads, head_size]),
        );
        let faaaa = faaaa.reshape(Shape::from_dims(&[n_heads, head_size, 1]))?;

        let state_init = x.const_f32_like(
            Arc::from(vec![0.0_f32; batch * n_heads * head_size * head_size]),
            Shape::from_dims(&[batch, n_heads, head_size, head_size]),
        );
        let mut state = state_init;
        let mut outs: Vec<LazyTensor> = Vec::with_capacity(seq);
        for t in 0..seq {
            let kt = k_h.slice(3_usize, t, 1)?; // (b, n_heads, head_size, 1)
            let vt = v_h.slice(2_usize, t, 1)?; // (b, n_heads, 1, head_size)
            let rt = r_h.slice(2_usize, t, 1)?;
            let at = kt.matmul(&vt)?; // (b, n_heads, head_size, head_size)
            let rhs = faaaa.broadcast_mul(&at)?.add(&state)?;
            let out_t = rt.matmul(&rhs)?.squeeze(2_usize)?;
            // Per-token decay: extract this token's decay (b, n_heads, head_size) and
            // reshape to (b, n_heads, head_size, 1) for state broadcast.
            let decay_t = decay_h
                .slice(1_usize, t, 1)?
                .reshape(Shape::from_dims(&[batch, n_heads, head_size, 1]))?;
            state = at.add(&decay_t.broadcast_mul(&state)?)?;
            outs.push(out_t);
        }

        // Stack outs along the seq dim.
        let mut stacked: Option<LazyTensor> = None;
        for out_t in outs.into_iter() {
            let with_seq = out_t.reshape(Shape::from_dims(&[batch, 1, n_heads, head_size]))?;
            stacked = Some(match stacked {
                None => with_seq,
                Some(s) => s.concat(&with_seq, 1_usize)?,
            });
        }
        let stacked = stacked.expect("at least one step");
        let stacked = stacked.reshape(Shape::from_dims(&[batch, seq, h]))?;

        let out = group_norm(
            &stacked, &layer.attn_ln_x_gain, &layer.attn_ln_x_bias,
            batch, seq, n_heads, head_size, 1e-5,
        )?;
        let gated = out.mul(&g)?;
        Ok(layer.attn_output.apply_linear(&gated, h, h))
    }

    fn channel_mix(
        &self,
        x: &LazyTensor,
        layer: &Rwkv6LayerWeights,
        batch: usize,
        seq: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let inter = cfg.ffn_intermediate();
        let shifted = shift_seq(x, batch, seq, h);
        // Eager v6 computes `shifted - xs` then `xs + shifted_diff * mix`,
        // which simplifies to `xs * (1 - mix) + shifted * mix`. v5 and v6
        // produce the same channel-mix structure once the static mix
        // coefficients are reinterpreted.
        let mix_key = x.const_f32_like(
            Arc::clone(&layer.ffn_time_mix_key),
            Shape::from_dims(&[1, 1, h]),
        );
        let mix_rec = x.const_f32_like(
            Arc::clone(&layer.ffn_time_mix_receptance),
            Shape::from_dims(&[1, 1, h]),
        );
        let sx = shifted.sub(x)?;
        let k_in = x.add(&sx.broadcast_mul(&mix_key)?)?;
        let r_in = x.add(&sx.broadcast_mul(&mix_rec)?)?;

        let k = layer.ffn_key.apply_linear(&k_in, h, inter).relu().sqr();
        let v = layer.ffn_value.apply_linear(&k, inter, h);
        let r = layer.ffn_receptance.apply_linear(&r_in, h, h).sigmoid();
        r.mul(&v)
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

fn group_norm(
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
    let gain_t = x.const_f32_like(
        Arc::clone(gain),
        Shape::from_dims(&[1, 1, n_heads, head_size]),
    );
    let bias_t = x.const_f32_like(
        Arc::clone(bias),
        Shape::from_dims(&[1, 1, n_heads, head_size]),
    );
    let gain_bc = gain_t.broadcast_to(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
    let bias_bc = bias_t.broadcast_to(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
    let scaled = normed.mul(&gain_bc)?.add(&bias_bc)?;
    scaled.reshape(Shape::from_dims(&[batch, seq, hidden]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &Rwkv6Config) -> Rwkv6Weights {
        let mut s: u32 = 17171;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let n_heads = cfg.n_heads();
        let inter = cfg.ffn_intermediate();
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<Rwkv6LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|li| {
                let pre_ln = if li == 0 {
                    Some((Arc::from(vec![1.0_f32; h]), Arc::from(vec![0.0_f32; h])))
                } else {
                    None
                };
                Rwkv6LayerWeights {
                    ln1_gain: Arc::from(vec![1.0_f32; h]),
                    ln1_bias: Arc::from(vec![0.0_f32; h]),
                    ln2_gain: Arc::from(vec![1.0_f32; h]),
                    ln2_bias: Arc::from(vec![0.0_f32; h]),
                    pre_ln,
                    attn_time_mix_x: vec_of(h, &mut *nb),
                    attn_time_mix_w: vec_of(h, &mut *nb),
                    attn_time_mix_key: vec_of(h, &mut *nb),
                    attn_time_mix_value: vec_of(h, &mut *nb),
                    attn_time_mix_receptance: vec_of(h, &mut *nb),
                    attn_time_mix_gate: vec_of(h, &mut *nb),
                    attn_time_mix_w1: vec_of(h * 5 * n_heads, &mut *nb),
                    attn_time_mix_w2: vec_of(5 * n_heads * h, &mut *nb),
                    attn_time_decay: vec_of(h, &mut *nb),
                    attn_time_decay_w1: vec_of(h * 2 * n_heads, &mut *nb),
                    attn_time_decay_w2: vec_of(2 * n_heads * h, &mut *nb),
                    attn_time_faaaa: vec_of(n_heads * cfg.head_size, &mut *nb),
                    attn_key: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    attn_value: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    attn_receptance: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    attn_gate: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    attn_output: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    attn_ln_x_gain: Arc::from(vec![1.0_f32; h]),
                    attn_ln_x_bias: Arc::from(vec![0.0_f32; h]),
                    ffn_time_mix_key: vec_of(h, &mut *nb),
                    ffn_time_mix_receptance: vec_of(h, &mut *nb),
                    ffn_key: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                    ffn_value: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
                    ffn_receptance: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                }
            })
            .collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let head = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        Rwkv6Weights { token_embedding, layers, final_ln_gain, final_ln_bias, head }
    }

    fn tiny_config() -> Rwkv6Config {
        Rwkv6Config {
            vocab_size: 16, hidden_size: 8,
            num_hidden_layers: 2, attention_hidden_size: 8,
            head_size: 4, num_attention_heads: 0,
            intermediate_size: Some(16),
            layer_norm_epsilon: 1e-5, rescale_every: 6,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = Rwkv6Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
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
        let model = Rwkv6Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[3]).unwrap().realize_f32();
        assert_eq!(logits.len(), cfg.vocab_size);
    }

    /// v6's per-token mix correction (time_mix_w1/w2) must be
    /// wired: zero out time_mix_w2 vs leave it default ⇒
    /// output must differ.
    #[test]
    fn time_mix_correction_is_wired() {
        let cfg = Rwkv6Config { num_hidden_layers: 1, ..tiny_config() };
        let base = tiny_weights(&cfg);
        let mut zeroed = base.clone();
        let h = cfg.hidden_size;
        let n_heads = cfg.n_heads();
        zeroed.layers[0].attn_time_mix_w2 = Arc::from(vec![0.0_f32; 5 * n_heads * h]);
        let m_base = Rwkv6Model { config: cfg.clone(), weights: base };
        let m_zero = Rwkv6Model { config: cfg, weights: zeroed };
        let toks = [1_u32, 2, 3];
        let a = m_base.forward(&toks).unwrap().realize_f32();
        let b = m_zero.forward(&toks).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        // Tiny weights (∈ [-0.025, 0.025]) ⇒ the rank-(2*n_heads or
        // 5*n_heads) correction is the product of two such matrices,
        // so the magnitude lands ~1e-9 here. We just require it to
        // be measurable; behaviorally it grows with real weights.
        assert!(max_diff > 1e-10,
            "v6 per-token mix correction must alter output, max_diff = {max_diff}");
    }

    /// v6's per-token decay correction (time_decay_w1/w2) must
    /// be wired: zero it out and confirm output changes.
    #[test]
    fn time_decay_correction_is_wired() {
        let cfg = Rwkv6Config { num_hidden_layers: 1, ..tiny_config() };
        let base = tiny_weights(&cfg);
        let mut zeroed = base.clone();
        let h = cfg.hidden_size;
        let n_heads = cfg.n_heads();
        zeroed.layers[0].attn_time_decay_w2 = Arc::from(vec![0.0_f32; 2 * n_heads * h]);
        let m_base = Rwkv6Model { config: cfg.clone(), weights: base };
        let m_zero = Rwkv6Model { config: cfg, weights: zeroed };
        let toks = [1_u32, 2, 3, 4];
        let a = m_base.forward(&toks).unwrap().realize_f32();
        let b = m_zero.forward(&toks).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-10,
            "v6 per-token decay correction must alter output, max_diff = {max_diff}");
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config();
        let model = Rwkv6Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = tiny_config();
        let model = Rwkv6Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let logits_via_embeds = model.forward_embeds(&embeds).unwrap().realize_f32();
        let max_diff = logits_ref.iter().zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "Rwkv6 forward vs forward_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = tiny_config();
        let model = Rwkv6Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let bad = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]), &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = tiny_config();
        let model = Rwkv6Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![5, 7];
        let h_ref = model.forward_hidden(&tokens).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let h_via_embeds = model.forward_hidden_embeds(&embeds).unwrap().realize_f32();
        let max_diff = h_ref.iter().zip(h_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "Rwkv6 forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})");
    }
}

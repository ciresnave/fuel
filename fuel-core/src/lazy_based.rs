//! Based (Stanford Hazy Research) decoder ported to the lazy-graph API.
//!
//! Phase D specialized port. Based mixes **three mixer types**
//! per layer, chosen by config-driven layer-index lists:
//!
//!   1. **`BasedConv`** (default) — gated depthwise 1-D causal
//!      convolution: project hidden to `4 * hidden`, chunk into
//!      `(u_conv, gate)` each `2 * hidden`; run `silu(conv(u_conv))
//!      * gate` then project back. Conv is depthwise (groups =
//!      channels), kernel 3, with `padding = 2` ⇒ caller left-pads
//!      with 2 zeros, runs causal_conv1d, then narrows to seq.
//!   2. **`LinearAttention`** — Taylor-expansion feature-map
//!      linear attention from the Based paper. Q, K are projected
//!      to `num_heads * feature_dim`, then lifted by the
//!      polynomial feature map
//!      `phi(x) = [1, x / d^(1/4), outer(x, x).flatten() /
//!      (sqrt(2) * sqrt(d))]` of expanded size `d^2 + d + 1`.
//!      Prefill computes
//!      `aqk = (phi(Q) @ phi(K)^T) * tril, @ V`, normalised by
//!      `z = 1 / (phi(Q) * cumsum(phi(K))).sum(-1) + eps`.
//!   3. **`SlidingWindowAttention`** — standard softmax attention
//!      with fused Wqkv (`hidden → 3 * hidden`), rotary, and a
//!      sliding-window causal mask of half-window radius.
//!
//! Per-layer routing: layer `i` is `LinearAttention` if
//! `i ∈ alt_mixer_layers`, `SlidingWindowAttention` if
//! `i ∈ alt_mixer_2_layers`, else `BasedConv`.
//!
//! Other carries:
//!   - Custom **swapped-order SwiGLU** MLP: `fc1: hidden → 4 *
//!     hidden`, chunked to `(left, right)` each `2 * hidden`;
//!     `silu(right) * left` then `fc2: 2*hidden → hidden`. The
//!     eager swaps the gate/value order vs. the standard
//!     `Activation::Swiglu`.
//!   - RmsNorm (no offset) pre-LN.
//!   - Tied lm_head to `embed_tokens`.
//!
//! # Scope (v1)
//!
//! Forward-only **prefill** (`seqlen_offset == 0` path only).
//! Streaming/decode paths for the stateful mixers are deferred
//! — they each maintain a different shape of recurrent state
//! and warrant their own session. batch=1, F32.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct BasedLinearAttentionParams {
    pub num_heads: usize,
    pub feature_dim: usize,
    /// Input dim used in the Taylor feature map's normalization.
    /// Eager uses `cfg.la.feature_map.input_dim` which equals
    /// `feature_dim` for shipped checkpoints; carried as a separate
    /// field for completeness.
    pub input_dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BasedSlidingWindowParams {
    pub num_heads: usize,
    pub window_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BasedConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// FFN intermediate; eager defaults this to `2 * hidden_size`
    /// (the fc1 output is `4*hidden`, chunked into 2 halves).
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub layer_norm_epsilon: f64,
    pub rope_theta: f64,
    pub alt_mixer_layers: Vec<usize>,
    pub alt_mixer_2_layers: Vec<usize>,
    pub la: BasedLinearAttentionParams,
    pub swa: BasedSlidingWindowParams,
}

impl BasedConfig {
    pub fn mixer_kind(&self, layer_idx: usize) -> BasedMixerKind {
        if self.alt_mixer_layers.contains(&layer_idx) {
            BasedMixerKind::Linear
        } else if self.alt_mixer_2_layers.contains(&layer_idx) {
            BasedMixerKind::Sliding
        } else {
            BasedMixerKind::Conv
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasedMixerKind {
    Conv,
    Linear,
    Sliding,
}

#[derive(Debug, Clone)]
pub struct BasedConvWeights {
    /// `[hidden, 4 * hidden]` projection.
    pub in_proj_w: WeightStorage,
    pub in_proj_b: Arc<[f32]>,
    /// `[2 * hidden, 1, 3]` depthwise kernel (groups = 2 * hidden).
    pub conv_w: Arc<[f32]>,
    /// `[2 * hidden]` conv bias (zeros for the bias-free
    /// `conv1d_no_bias` in eager but `causal_conv1d` requires bias —
    /// pass zeros if absent).
    pub conv_b: Arc<[f32]>,
    /// `[2 * hidden, hidden]`.
    pub out_proj_w: WeightStorage,
    pub out_proj_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct BasedLinearAttentionWeights {
    pub q_proj: WeightStorage, // hidden → num_heads * feature_dim
    pub k_proj: WeightStorage,
    pub v_proj: WeightStorage, // hidden → hidden
    pub out_proj: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct BasedSlidingWeights {
    /// Fused `[hidden, 3 * hidden]`.
    pub wqkv: WeightStorage,
    pub out_proj: WeightStorage,
}

#[derive(Debug, Clone)]
pub enum BasedMixerWeights {
    Conv(BasedConvWeights),
    Linear(BasedLinearAttentionWeights),
    Sliding(BasedSlidingWeights),
}

#[derive(Debug, Clone)]
pub struct BasedLayerWeights {
    pub norm1_gain: Arc<[f32]>,
    pub norm2_gain: Arc<[f32]>,
    pub mixer: BasedMixerWeights,
    /// `[hidden, 4 * hidden]` — eager `fc1` output is 4×hidden;
    /// the chunk-2 splits into `(left, right)` each `2 * hidden`.
    pub fc1: WeightStorage,
    /// `[intermediate (== 2 * hidden), hidden]`.
    pub fc2: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct BasedWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<BasedLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct BasedModel {
    pub config: BasedConfig,
    pub weights: BasedWeights,
}

impl BasedModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        let lm_head = WeightStorage::F32(weights.token_embedding.clone());
        Ok(lm_head.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Based-specific: per-layer mixer-type selection
    /// (sliding-window attention / linear-attention / short-conv)
    /// is honored. v1 = prefill only (start_pos = 0).
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert_eq!(
            weights.layers.len(), cfg.num_hidden_layers,
            "weights.layers length must match num_hidden_layers",
        );
        assert_eq!(start_pos, 0, "BasedModel v1: prefill only (start_pos must be 0)");

        let mut h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;

        let sliding_head_dim = cfg.hidden_size / cfg.swa.num_heads;
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, sliding_head_dim,
        );

        for (idx, layer) in weights.layers.iter().enumerate() {
            h = self.apply_block(&h, layer, idx, &rope_cos, &rope_sin)?;
        }
        Ok(h.rms_norm_affine(std::sync::Arc::clone(&weights.final_norm_gain), cfg.layer_norm_epsilon)?)
    }

    fn apply_block(
        &self,
        x: &LazyTensor,
        layer: &BasedLayerWeights,
        layer_idx: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let x_norm = x.rms_norm_affine(std::sync::Arc::clone(&layer.norm1_gain), cfg.layer_norm_epsilon)?;
        let mixed = self.apply_mixer(&x_norm, &layer.mixer, layer_idx, rope_cos, rope_sin)?;
        let h1 = x.add(&mixed)?;
        let h1_norm = h1.rms_norm_affine(std::sync::Arc::clone(&layer.norm2_gain), cfg.layer_norm_epsilon)?;
        let mlp_out = self.apply_mlp(&h1_norm, &layer.fc1, &layer.fc2)?;
        h1.add(&mlp_out)
    }

    fn apply_mixer(
        &self,
        x: &LazyTensor,
        mixer: &BasedMixerWeights,
        layer_idx: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let expected = cfg.mixer_kind(layer_idx);
        match (mixer, expected) {
            (BasedMixerWeights::Conv(w), BasedMixerKind::Conv) => self.apply_conv(x, w),
            (BasedMixerWeights::Linear(w), BasedMixerKind::Linear) => self.apply_linear(x, w),
            (BasedMixerWeights::Sliding(w), BasedMixerKind::Sliding) => {
                self.apply_sliding(x, w, rope_cos, rope_sin)
            }
            _ => Err(crate::Error::Msg(format!(
                "Based layer {layer_idx}: mixer weight kind does not match \
                 config-derived kind {expected:?} — config + weights are inconsistent",
            )).bt()),
        }
    }

    fn apply_conv(&self, x: &LazyTensor, w: &BasedConvWeights) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.hidden_size;
        let dim2 = 2 * h;
        let kernel = 3;

        // in_proj: hidden → 4 * hidden, then chunk into (u_conv, gate).
        let proj = w.in_proj_w.apply_linear(x, h, 4 * h);
        let proj_bias_t = x.const_f32_like(
            Arc::clone(&w.in_proj_b),
            Shape::from_dims(&[4 * h]),
        );
        let proj = proj.broadcast_add(&proj_bias_t)?;
        let u_conv_in = proj.slice(2_usize, 0, dim2)?;
        let gate = proj.slice(2_usize, dim2, dim2)?;

        // Causal conv1d: (b, seq, dim2) → (b, dim2, seq), pad-left (k-1)
        // zeros, run causal_conv1d, transpose back.
        let u_perm = u_conv_in.permute([0, 2, 1_usize])?;
        let pad_zeros = x.const_f32_like(
            Arc::from(vec![0.0_f32; batch * dim2 * (kernel - 1)]),
            Shape::from_dims(&[batch, dim2, kernel - 1]),
        );
        let u_padded = pad_zeros.concat(&u_perm, 2_usize)?;
        let conv_w = x.const_f32_like(
            Arc::clone(&w.conv_w),
            Shape::from_dims(&[dim2, 1, kernel]),
        );
        let conv_b = x.const_f32_like(
            Arc::clone(&w.conv_b),
            Shape::from_dims(&[dim2]),
        );
        // Note: causal_conv1d's optional fused-SiLU is off — we want
        // the conv output to be then multiplied by silu(...) elsewhere?
        // Looking again, the eager applies silu AFTER conv:
        //   u_conv = conv(u_conv_in).silu()
        // So we run conv1d WITHOUT fused silu, then silu.
        let u_conv_raw = u_padded.causal_conv1d(&conv_w, &conv_b, false);
        // back to (b, seq, dim2)
        let u_conv = u_conv_raw.permute([0, 2, 1_usize])?.silu();
        let v = u_conv.mul(&gate)?;
        let out = w.out_proj_w.apply_linear(&v, dim2, h);
        let out_bias_t = x.const_f32_like(
            Arc::clone(&w.out_proj_b),
            Shape::from_dims(&[h]),
        );
        out.broadcast_add(&out_bias_t)
    }

    fn apply_linear(
        &self,
        x: &LazyTensor,
        w: &BasedLinearAttentionWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.hidden_size;
        let n_heads = cfg.la.num_heads;
        let d_feat = cfg.la.feature_dim;
        // Eager v_dim: v_proj outputs hidden, reshaped to (b, l, num_heads, hidden/num_heads).
        assert_eq!(
            h % n_heads, 0,
            "BasedConfig: hidden_size must be divisible by la.num_heads",
        );
        let d_h_v = h / n_heads;
        let d_expanded = d_feat * d_feat + d_feat + 1;

        let q = w.q_proj.apply_linear(x, h, n_heads * d_feat);
        let k = w.k_proj.apply_linear(x, h, n_heads * d_feat);
        let v = w.v_proj.apply_linear(x, h, h);

        let _ = (batch, seq);
        let q = q.split_heads(n_heads, d_feat)?;
        let k = k.split_heads(n_heads, d_feat)?;
        let v_h = v.split_heads(n_heads, d_h_v)?;

        // Apply the Taylor feature map.
        let phi_q = taylor_feature_map(&q, d_feat, batch, n_heads, seq)?;
        let phi_k = taylor_feature_map(&k, d_feat, batch, n_heads, seq)?;
        // phi shapes: (b, n_heads, seq, d_expanded)
        debug_assert_eq!(
            phi_q.shape().dims(),
            &[batch, n_heads, seq, d_expanded],
            "phi_q shape mismatch",
        );

        // Prefill linear attention (causal):
        //   aqk = (phi_q @ phi_k^T) * tril, then @ v
        //   z   = 1 / ((phi_q * cumsum(phi_k, axis=seq)).sum(-1) + eps)
        //   out = aqk * z.unsqueeze(-1)
        let phi_k_t = phi_k.transpose()?;
        let aqk = phi_q.matmul(&phi_k_t)?; // (b, h, seq, seq)
        // Build causal tril mask anchored on x's graph.
        let mut tril_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..=i {
                tril_data[i * seq + j] = 1.0;
            }
        }
        let tril = x.const_f32_like(
            Arc::from(tril_data),
            Shape::from_dims(&[1, 1, seq, seq]),
        );
        let tril_bc = tril.broadcast_to(Shape::from_dims(&[batch, n_heads, seq, seq]))?;
        let aqk_masked = aqk.mul(&tril_bc)?;
        let aqk_v = aqk_masked.matmul(&v_h)?; // (b, h, seq, d_h_v)

        let phi_k_cumsum = phi_k.cumsum(2_usize)?; // along seq axis
        let prod = phi_q.mul(&phi_k_cumsum)?;
        let z_inv = prod
            .sum_dim(3_usize)?            // (b, h, seq)
            .add_scalar(1e-12);
        // z = 1 / z_inv. Use mul(-1) negate, exp, etc. — easier: just
        // divide.  We don't have direct scalar-divide on LazyTensor's
        // public API; do it via reciprocal: build ones, divide.
        let ones = x.const_f32_like(
            Arc::from(vec![1.0_f32; batch * n_heads * seq]),
            Shape::from_dims(&[batch, n_heads, seq]),
        );
        let z = ones.div(&z_inv)?;
        let z_bc = z
            .reshape(Shape::from_dims(&[batch, n_heads, seq, 1]))?
            .broadcast_to(Shape::from_dims(&[batch, n_heads, seq, d_h_v]))?;
        let out_h = aqk_v.mul(&z_bc)?;

        let merged = out_h.merge_heads()?;
        Ok(w.out_proj.apply_linear(&merged, h, h))
    }

    fn apply_sliding(
        &self,
        x: &LazyTensor,
        w: &BasedSlidingWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.hidden_size;
        let n_heads = cfg.swa.num_heads;
        let head_dim = h / n_heads;
        let window = cfg.swa.window_size / 2;

        // Fused Wqkv: hidden → 3 * hidden, slice into Q/K/V.
        let qkv = w.wqkv.apply_linear(x, h, 3 * h);
        let q = qkv.slice(2_usize, 0, h)?;
        let k = qkv.slice(2_usize, h, h)?;
        let v = qkv.slice(2_usize, 2 * h, h)?;

        let _ = batch;
        let q = q.split_heads(n_heads, head_dim)?;
        let k = k.split_heads(n_heads, head_dim)?;
        let v = v.split_heads(n_heads, head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let k_t = k_r.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        // Sliding causal mask.
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i || j + window < i {
                    mask_data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = x.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]));
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let probs = scores_masked.softmax_last_dim()?;
        let attn_v = probs.matmul(&v)?;
        let merged = attn_v.merge_heads()?;
        Ok(w.out_proj.apply_linear(&merged, h, h))
    }

    fn apply_mlp(
        &self,
        x: &LazyTensor,
        fc1: &WeightStorage,
        fc2: &WeightStorage,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        // fc1 out is 4 * hidden; chunked to (left, right) each 2 * hidden.
        let projected = fc1.apply_linear(x, h, 4 * h);
        let left = projected.slice(2_usize, 0, 2 * h)?;
        let right = projected.slice(2_usize, 2 * h, 2 * h)?;
        // Eager swap: `silu(right) * left`.
        let gated = right.silu().mul(&left)?;
        Ok(fc2.apply_linear(&gated, inter, h))
    }
}

/// Taylor feature map: `phi(x) = [ones, x / d^(1/4),
/// (x ⊗ x).flatten() / (sqrt(2) * sqrt(d))]` for input shape
/// `(b, h, seq, d)`. Output shape is `(b, h, seq, d^2 + d + 1)`.
fn taylor_feature_map(
    x: &LazyTensor,
    d: usize,
    batch: usize,
    n_heads: usize,
    seq: usize,
) -> Result<LazyTensor> {
    let r2 = std::f64::consts::SQRT_2;
    let rd = (d as f64).sqrt();
    let rrd = rd.sqrt();

    // Ones row of length 1 per (b, h, seq).
    let ones = x.const_f32_like(
        Arc::from(vec![1.0_f32; batch * n_heads * seq * 1]),
        Shape::from_dims(&[batch, n_heads, seq, 1]),
    );

    let x_scaled = x.mul_scalar(1.0 / rrd); // x / d^(1/4)

    // Outer product: (b, h, seq, d, 1) * (b, h, seq, 1, d) = (b, h, seq, d, d)
    let x_col = x
        .reshape(Shape::from_dims(&[batch, n_heads, seq, d, 1]))?;
    let x_row = x
        .reshape(Shape::from_dims(&[batch, n_heads, seq, 1, d]))?;
    let x_col_bc = x_col.broadcast_to(Shape::from_dims(&[batch, n_heads, seq, d, d]))?;
    let x_row_bc = x_row.broadcast_to(Shape::from_dims(&[batch, n_heads, seq, d, d]))?;
    let outer = x_col_bc.mul(&x_row_bc)?;
    let outer_flat = outer
        .reshape(Shape::from_dims(&[batch, n_heads, seq, d * d]))?
        .mul_scalar(1.0 / (r2 * rd));

    // Concat along last axis: [ones, x_scaled, outer_flat] → d^2 + d + 1
    let part_1 = ones.concat(&x_scaled, 3_usize)?;
    part_1.concat(&outer_flat, 3_usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_weights(cfg: &BasedConfig) -> BasedWeights {
        let mut s: u32 = 24680;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let h = cfg.hidden_size;
        let dim2 = 2 * h;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<BasedLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|i| {
                let kind = cfg.mixer_kind(i);
                let mixer = match kind {
                    BasedMixerKind::Conv => BasedMixerWeights::Conv(BasedConvWeights {
                        in_proj_w: WeightStorage::F32(vec_of(h * (4 * h), &mut *nb)),
                        in_proj_b: vec_of(4 * h, &mut *nb),
                        conv_w: vec_of(dim2 * 3, &mut *nb),
                        conv_b: Arc::from(vec![0.0_f32; dim2]),
                        out_proj_w: WeightStorage::F32(vec_of(dim2 * h, &mut *nb)),
                        out_proj_b: vec_of(h, &mut *nb),
                    }),
                    BasedMixerKind::Linear => BasedMixerWeights::Linear(BasedLinearAttentionWeights {
                        q_proj: WeightStorage::F32(vec_of(h * cfg.la.num_heads * cfg.la.feature_dim, &mut *nb)),
                        k_proj: WeightStorage::F32(vec_of(h * cfg.la.num_heads * cfg.la.feature_dim, &mut *nb)),
                        v_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                        out_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    }),
                    BasedMixerKind::Sliding => BasedMixerWeights::Sliding(BasedSlidingWeights {
                        wqkv: WeightStorage::F32(vec_of(h * (3 * h), &mut *nb)),
                        out_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    }),
                };
                BasedLayerWeights {
                    norm1_gain: Arc::from(vec![1.0_f32; h]),
                    norm2_gain: Arc::from(vec![1.0_f32; h]),
                    mixer,
                    fc1: WeightStorage::F32(vec_of(h * (4 * h), &mut *nb)),
                    fc2: WeightStorage::F32(vec_of(cfg.intermediate_size * h, &mut *nb)),
                }
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        BasedWeights { token_embedding, layers, final_norm_gain }
    }

    fn tiny_config() -> BasedConfig {
        BasedConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16, // = 2 * hidden
            num_hidden_layers: 3,
            num_attention_heads: 2,
            layer_norm_epsilon: 1e-5,
            rope_theta: 10_000.0,
            alt_mixer_layers: vec![1],     // layer 1 = LinearAttention
            alt_mixer_2_layers: vec![2],   // layer 2 = SlidingWindowAttention
                                            // layer 0 = BasedConv (default)
            la: BasedLinearAttentionParams { num_heads: 2, feature_dim: 4, input_dim: 4 },
            swa: BasedSlidingWindowParams { num_heads: 2, window_size: 4 },
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = BasedModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// All three mixer types must contribute distinctly: zero
    /// each one's output projection in turn and confirm the
    /// outputs differ from the baseline.
    #[test]
    fn all_three_mixers_active() {
        let cfg = tiny_config();
        let base = tiny_weights(&cfg);
        let baseline = BasedModel { config: cfg.clone(), weights: base.clone() }
            .forward(&[1, 2, 3], 0).unwrap().realize_f32();

        // Zero out the Conv mixer's output projection at layer 0.
        {
            let mut w = base.clone();
            if let BasedMixerWeights::Conv(c) = &mut w.layers[0].mixer {
                let h = cfg.hidden_size;
                c.out_proj_w = WeightStorage::F32(Arc::from(vec![0.0_f32; (2 * h) * h]));
                c.out_proj_b = Arc::from(vec![0.0_f32; h]);
            }
            let out = BasedModel { config: cfg.clone(), weights: w }
                .forward(&[1, 2, 3], 0).unwrap().realize_f32();
            let mut max_diff = 0.0_f32;
            for (a, b) in baseline.iter().zip(out.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
            assert!(max_diff > 1e-6, "BasedConv mixer must affect output");
        }

        // Zero out the Linear mixer's output projection at layer 1.
        {
            let mut w = base.clone();
            if let BasedMixerWeights::Linear(la) = &mut w.layers[1].mixer {
                let h = cfg.hidden_size;
                la.out_proj = WeightStorage::F32(Arc::from(vec![0.0_f32; h * h]));
            }
            let out = BasedModel { config: cfg.clone(), weights: w }
                .forward(&[1, 2, 3], 0).unwrap().realize_f32();
            let mut max_diff = 0.0_f32;
            for (a, b) in baseline.iter().zip(out.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
            assert!(max_diff > 1e-6, "LinearAttention mixer must affect output");
        }

        // Zero out the Sliding mixer's output projection at layer 2.
        {
            let mut w = base.clone();
            if let BasedMixerWeights::Sliding(sw) = &mut w.layers[2].mixer {
                let h = cfg.hidden_size;
                sw.out_proj = WeightStorage::F32(Arc::from(vec![0.0_f32; h * h]));
            }
            let out = BasedModel { config: cfg.clone(), weights: w }
                .forward(&[1, 2, 3], 0).unwrap().realize_f32();
            let mut max_diff = 0.0_f32;
            for (a, b) in baseline.iter().zip(out.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
            assert!(max_diff > 1e-6, "SlidingWindowAttention mixer must affect output");
        }
    }

    #[test]
    fn mixer_routing_via_config_lists() {
        let cfg = tiny_config();
        assert_eq!(cfg.mixer_kind(0), BasedMixerKind::Conv);
        assert_eq!(cfg.mixer_kind(1), BasedMixerKind::Linear);
        assert_eq!(cfg.mixer_kind(2), BasedMixerKind::Sliding);
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config();
        let model = BasedModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

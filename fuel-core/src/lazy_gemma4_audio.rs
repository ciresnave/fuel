//! Gemma 4 audio encoder (Conformer) ported to the lazy-graph API.
//!
//! Pipeline:
//!
//!   1. **SSCP front-end** (`SubSampleConvProjection`). Two strided
//!      Conv2D blocks turn the mel-spectrogram `(B, T_in, n_mels)`
//!      into `(B, T_sub, hidden_size)`. Each block is a Conv2D with
//!      semicausal padding + LayerNorm + ReLU. After the second
//!      block, the residual `(C_out, F_out)` channel/freq factors
//!      are flattened and linearly projected to `hidden_size`.
//!   2. **Conformer stack.** A stack of `depth` blocks, each:
//!      - half-step feed-forward (`residual + 0.5 · FF(x)`),
//!      - chunked multi-head self-attention with a host-built
//!        block-band mask + Shaw-style relative position bias,
//!      - lightweight depthwise Conv1D with GLU gating,
//!      - half-step feed-forward,
//!      - RmsNorm.
//!   3. **Output projection** (optional).
//!
//! # Chunked attention
//!
//! Self-attention is restricted to the chunk the query lives in plus
//! `left_chunks` whole chunks immediately preceding it. The mask is
//! built host-side at graph-build time from `chunk_size`,
//! `left_chunks`, and the post-SSCP sequence length, and emitted as a
//! `const_f32_like` tensor of `0.0` on attend / `-inf` on mask. It is
//! broadcast-added to the scaled QKᵀ scores before softmax.
//!
//! # Relative position bias
//!
//! A learnable Shaw-style table of shape `(2 * max_rel + 1, num_heads)`
//! indexed by the clipped offset `clip(i - j, -max_rel, +max_rel)`.
//! Reduces to a single `index_select` followed by a permute/broadcast
//! into `(num_heads, T, T)` and an add to the scores.
//!
//! `max_rel` is derived from `chunk_size + left_chunks * chunk_size`
//! so all reachable offsets are representable.
//!
//! # Scope (v1)
//!
//! Forward-only, F32, no input mask. The eager
//! [`fuel_transformers::models::llm::gemma4::audio`] additionally
//! takes an `audio_mel_mask` for padded-length batches; this lazy port
//! assumes the caller has already padded mel input to a uniform length
//! and that all frames are valid. Mask-aware audio batching is a
//! follow-up.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

// ── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4AudioConfig {
    pub input_feat_size: usize,
    pub hidden_size: usize,
    pub output_proj_dims: Option<usize>,
    pub conf_attention_chunk_size: usize,
    /// Number of whole chunks of left context (inclusive of the current
    /// chunk) the eager config exposes minus 1 — i.e. the number of
    /// strictly past chunks each query can see.
    pub conf_left_chunks: usize,
    pub conf_attention_logit_cap: f64,
    pub conf_num_attention_heads: usize,
    pub conf_num_hidden_layers: usize,
    pub conf_conv_kernel_size: usize,
    pub conf_reduction_factor: usize,
    pub conf_residual_weight: f64,
    pub sscp_conv_channel_size: [usize; 2],
    pub sscp_conv_kernel_size: [[usize; 2]; 2],
    pub sscp_conv_stride_size: [[usize; 2]; 2],
    pub rms_norm_eps: f64,
    pub gradient_clipping: f64,
}

impl Gemma4AudioConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.conf_num_attention_heads
    }

    /// Maximum representable relative position offset. Drives the
    /// `(2*max_rel + 1, num_heads)` Shaw table.
    pub fn max_rel(&self) -> usize {
        // Reachable offset under the block-band mask: a query at the
        // end of a chunk can see a key at offset
        //   -(left_chunks * chunk_size + (chunk_size - 1))
        // and a query at the start of a chunk can see a key at offset
        //   +(chunk_size - 1).
        // We bound by the larger so all reachable signed offsets
        // survive the clamp.
        self.conf_left_chunks * self.conf_attention_chunk_size
            + self.conf_attention_chunk_size
            - 1
    }
}

// ── Weights ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SscpBlockWeights {
    /// `[Cout, Cin, Kt, Kf]`.
    pub conv: WeightStorage,
    /// LayerNorm gain on `Cout`.
    pub norm_gain: Arc<[f32]>,
    /// LayerNorm bias on `Cout`.
    pub norm_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct ConformerLayerWeights {
    // FF1 (half-step).
    pub ff1_pre_norm: Arc<[f32]>,
    pub ff1_post_norm: Arc<[f32]>,
    pub ff1_w1: WeightStorage,
    pub ff1_w2: WeightStorage,

    // Attention.
    pub attn_pre_norm: Arc<[f32]>,
    pub attn_post_norm: Arc<[f32]>,
    pub attn_q: WeightStorage,
    pub attn_k: WeightStorage,
    pub attn_v: WeightStorage,
    pub attn_o: WeightStorage,
    /// Shaw-style learnable bias table `[2*max_rel + 1, num_heads]`.
    pub rel_pos_bias: Arc<[f32]>,

    // Light Conv1D.
    pub lconv_pre_norm: Arc<[f32]>,
    pub lconv_inner_norm: Arc<[f32]>,
    pub lconv_linear_start: WeightStorage,
    pub lconv_linear_end: WeightStorage,
    /// Depthwise conv `[hidden_size, 1, K]`.
    pub lconv_depthwise: WeightStorage,

    // FF2 (half-step).
    pub ff2_pre_norm: Arc<[f32]>,
    pub ff2_post_norm: Arc<[f32]>,
    pub ff2_w1: WeightStorage,
    pub ff2_w2: WeightStorage,

    pub out_norm: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Gemma4AudioWeights {
    pub sscp: [SscpBlockWeights; 2],
    /// `[Cout * F_out, hidden_size]`.
    pub sscp_input_proj: WeightStorage,
    pub layers: Vec<ConformerLayerWeights>,
    /// `[hidden_size, output_proj_dims]` if `output_proj_dims` is set.
    pub output_proj: Option<WeightStorage>,
}

#[derive(Debug, Clone)]
pub struct Gemma4AudioModel {
    pub config: Gemma4AudioConfig,
    pub weights: Gemma4AudioWeights,
}

impl Gemma4AudioModel {
    /// Encode a mel-spectrogram `(B, T_in, n_mels)` into Conformer
    /// hidden states `(B, T_out, d_model)`, where `d_model` is
    /// `output_proj_dims.unwrap_or(hidden_size)` and `T_out` is the
    /// SSCP-subsampled length divided by `conf_reduction_factor`.
    pub fn forward(&self, mel: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = mel.shape();
        let dims = dims.dims();
        if dims.len() != 3 {
            return Err(crate::Error::Msg(format!(
                "Gemma4AudioModel::forward: mel must be rank 3 (B, T, n_mels), got {dims:?}",
            )).bt());
        }
        let (b, t_in, n_mels) = (dims[0], dims[1], dims[2]);
        if n_mels != cfg.input_feat_size {
            return Err(crate::Error::Msg(format!(
                "Gemma4AudioModel::forward: mel n_mels={n_mels} != input_feat_size={}",
                cfg.input_feat_size,
            )).bt());
        }
        if t_in == 0 {
            return Err(crate::Error::Msg(
                "Gemma4AudioModel::forward: T_in must be > 0".into(),
            ).bt());
        }
        // (B, T, F) -> (B, 1, T, F)
        let x = mel.unsqueeze(1_usize)?;
        let (mut x, mut t_cur, mut f_cur) = (x, t_in, n_mels);
        for (idx, block) in self.weights.sscp.iter().enumerate() {
            let (xo, to, fo) = self.sscp_block_forward(&x, t_cur, f_cur, idx, block)?;
            x = xo;
            t_cur = to;
            f_cur = fo;
        }
        // x : (B, C_out, T_sub, F_out) → (B, T_sub, F_out * C_out)
        let c_out = cfg.sscp_conv_channel_size[1];
        let x = x
            .permute([0_usize, 2, 3, 1])?
            .reshape(Shape::from_dims(&[b, t_cur, f_cur * c_out]))?;
        let mut h = self.weights.sscp_input_proj.apply_linear(
            &x,
            f_cur * c_out,
            cfg.hidden_size,
        );

        // Chunked attention mask + rel-pos offsets — shared across layers.
        let t_seq = t_cur;
        let attn_mask = self.build_block_band_mask(&h, t_seq)?;
        let rel_pos_idx = self.build_rel_pos_indices(&h, t_seq)?;

        for layer in &self.weights.layers {
            h = self.conformer_block_forward(&h, layer, &attn_mask, &rel_pos_idx, t_seq)?;
        }

        // Reduction-factor subsampling along T.
        let t_after_blocks = h.dim(1_usize)?;
        let stride = cfg.conf_reduction_factor.max(1);
        let h = if stride > 1 {
            let reduced_len = t_after_blocks.div_ceil(stride);
            let mut idx_data = Vec::with_capacity(reduced_len);
            for i in 0..reduced_len {
                let pick = (i * stride).min(t_after_blocks - 1);
                idx_data.push(pick as u32);
            }
            let idx = h.const_u32_like(idx_data, Shape::from_dims(&[reduced_len]));
            h.index_select(1_usize, &idx)?
        } else {
            h
        };

        if let Some(ref out_proj) = self.weights.output_proj {
            let out_dim = cfg.output_proj_dims.expect(
                "output_proj weights present but output_proj_dims is None",
            );
            Ok(out_proj.apply_linear(&h, cfg.hidden_size, out_dim))
        } else {
            Ok(h)
        }
    }

    fn sscp_block_forward(
        &self,
        x: &LazyTensor,
        t_in: usize,
        f_in: usize,
        idx: usize,
        block: &SscpBlockWeights,
    ) -> Result<(LazyTensor, usize, usize)> {
        let cfg = &self.config;
        let kt = cfg.sscp_conv_kernel_size[idx][0];
        let kf = cfg.sscp_conv_kernel_size[idx][1];
        let st = cfg.sscp_conv_stride_size[idx][0];
        let sf = cfg.sscp_conv_stride_size[idx][1];
        let cin = if idx == 0 { 1 } else { cfg.sscp_conv_channel_size[idx - 1] };
        let cout = cfg.sscp_conv_channel_size[idx];

        // Semicausal-style padding: half on each side along T, +1 on each side along F.
        let pad_t = kt / 2;
        let pad_f = 1_usize;
        // x : (B, Cin, T, F). pad_with_zeros on last two dims.
        let x_padded = x
            .pad_with_zeros(2_usize, pad_t, pad_t)?
            .pad_with_zeros(3_usize, pad_f, pad_f)?;
        let w = block.conv.const_like(&x_padded, Shape::from_dims(&[cout, cin, kt, kf]))?;
        let conv_out = x_padded.conv2d(&w, None, (st, sf), (0, 0), 1)?;
        // Output T_out / F_out from conv arithmetic.
        let t_out = (t_in + 2 * pad_t - kt) / st + 1;
        let f_out = (f_in + 2 * pad_f - kf) / sf + 1;

        // LayerNorm over channel dim: permute (B, C, T, F) → (B, T, F, C),
        // layer-norm last dim, then permute back.
        let permuted = conv_out.permute([0_usize, 2, 3, 1])?;
        let normed = permuted.layer_norm_affine(
            Arc::clone(&block.norm_gain),
            Arc::clone(&block.norm_bias),
            cfg.rms_norm_eps,
        )?;
        let back = normed.permute([0_usize, 3, 1, 2])?;
        Ok((back.relu(), t_out, f_out))
    }

    fn conformer_block_forward(
        &self,
        x: &LazyTensor,
        layer: &ConformerLayerWeights,
        attn_mask: &LazyTensor,
        rel_pos_idx: &LazyTensor,
        t_seq: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let scale = cfg.conf_residual_weight;
        let clip = cfg.gradient_clipping;

        // FF1 (half-step) ────────────────────────────────────────────────
        let h = self.feed_forward(
            x,
            &layer.ff1_pre_norm,
            &layer.ff1_post_norm,
            &layer.ff1_w1,
            &layer.ff1_w2,
            scale,
            clip,
        )?;

        // Attention ────────────────────────────────────────────────────
        let h = self.attention(
            &h,
            layer,
            attn_mask,
            rel_pos_idx,
            t_seq,
            clip,
        )?;

        // Light Conv1D ──────────────────────────────────────────────────
        let h = self.light_conv1d(&h, layer, clip)?;

        // FF2 (half-step) ───────────────────────────────────────────────
        let h = self.feed_forward(
            &h,
            &layer.ff2_pre_norm,
            &layer.ff2_post_norm,
            &layer.ff2_w1,
            &layer.ff2_w2,
            scale,
            clip,
        )?;
        let h = h.clamp(-clip, clip);

        // Final RmsNorm.
        h.rms_norm_affine(Arc::clone(&layer.out_norm), cfg.rms_norm_eps)
    }

    fn feed_forward(
        &self,
        x: &LazyTensor,
        pre_gain: &Arc<[f32]>,
        post_gain: &Arc<[f32]>,
        w1: &WeightStorage,
        w2: &WeightStorage,
        scale: f64,
        clip: f64,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let residual = x.clone();
        let x = x.clamp(-clip, clip);
        let x_n = x.rms_norm_affine(Arc::clone(pre_gain), cfg.rms_norm_eps)?;
        let y = w1.apply_linear(&x_n, cfg.hidden_size, cfg.hidden_size * 4);
        let y = y.silu();
        let y = w2.apply_linear(&y, cfg.hidden_size * 4, cfg.hidden_size);
        let y = y.clamp(-clip, clip);
        let y_n = y.rms_norm_affine(Arc::clone(post_gain), cfg.rms_norm_eps)?;
        residual.add(&y_n.mul_scalar(scale))
    }

    fn attention(
        &self,
        x: &LazyTensor,
        layer: &ConformerLayerWeights,
        attn_mask: &LazyTensor,
        rel_pos_idx: &LazyTensor,
        t_seq: usize,
        clip: f64,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let n_heads = cfg.conf_num_attention_heads;
        let head_dim = cfg.head_dim();
        let hidden = cfg.hidden_size;
        let dims = x.shape();
        let dims = dims.dims();
        let b = dims[0];

        let residual = x.clone();
        let x = x.clamp(-clip, clip);
        let x_n = x.rms_norm_affine(Arc::clone(&layer.attn_pre_norm), cfg.rms_norm_eps)?;

        let q = layer.attn_q.apply_linear(&x_n, hidden, n_heads * head_dim);
        let k = layer.attn_k.apply_linear(&x_n, hidden, n_heads * head_dim);
        let v = layer.attn_v.apply_linear(&x_n, hidden, n_heads * head_dim);

        // (B, T, H, D) → (B, H, T, D)
        let q = q.reshape(Shape::from_dims(&[b, t_seq, n_heads, head_dim]))?
            .permute([0_usize, 2, 1, 3])?;
        let k = k.reshape(Shape::from_dims(&[b, t_seq, n_heads, head_dim]))?
            .permute([0_usize, 2, 1, 3])?;
        let v = v.reshape(Shape::from_dims(&[b, t_seq, n_heads, head_dim]))?
            .permute([0_usize, 2, 1, 3])?;

        // QKᵀ scaled.
        let scale = (head_dim as f64).powf(-0.5);
        let k_t = k.permute([0_usize, 1, 3, 2])?; // (B, H, D, T)
        let scores = q.matmul(&k_t)?.mul_scalar(scale);

        // Rel-pos bias: (T, T) flat index → look up rows in
        // (2*max_rel+1, H) → reshape (T, T, H) → permute (H, T, T) →
        // broadcast to (B, H, T, T).
        let max_rel = cfg.max_rel();
        let span = 2 * max_rel + 1;
        let rel_table = x.const_f32_like(
            Arc::clone(&layer.rel_pos_bias),
            Shape::from_dims(&[span, n_heads]),
        );
        let picked = rel_table.index_select(0_usize, rel_pos_idx)?; // (T*T, H)
        let bias = picked
            .reshape(Shape::from_dims(&[t_seq, t_seq, n_heads]))?
            .permute([2_usize, 0, 1])? // (H, T, T)
            .reshape(Shape::from_dims(&[1, n_heads, t_seq, t_seq]))?
            .broadcast_to(Shape::from_dims(&[b, n_heads, t_seq, t_seq]))?;
        let scores = scores.add(&bias)?;

        // Softcap.
        let cap = cfg.conf_attention_logit_cap;
        let scores = scores.mul_scalar(1.0 / cap).tanh().mul_scalar(cap);

        // Add the (1, 1, T, T) block-band mask (broadcast across B, H).
        let mask = attn_mask.broadcast_to(Shape::from_dims(&[b, n_heads, t_seq, t_seq]))?;
        let scores = scores.add(&mask)?;
        let probs = scores.softmax_last_dim()?;

        // probs : (B, H, T, T), v : (B, H, T, D) → (B, H, T, D)
        let ctx = probs.matmul(&v)?;
        let ctx = ctx.permute([0_usize, 2, 1, 3])?
            .reshape(Shape::from_dims(&[b, t_seq, hidden]))?;
        let out = layer.attn_o.apply_linear(&ctx, hidden, hidden).clamp(-clip, clip);
        let out_n = out.rms_norm_affine(Arc::clone(&layer.attn_post_norm), cfg.rms_norm_eps)?;
        residual.add(&out_n)
    }

    fn light_conv1d(
        &self,
        x: &LazyTensor,
        layer: &ConformerLayerWeights,
        clip: f64,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let k = cfg.conf_conv_kernel_size;

        let residual = x.clone();
        let x_n = x.rms_norm_affine(Arc::clone(&layer.lconv_pre_norm), cfg.rms_norm_eps)?;
        let y = layer.lconv_linear_start.apply_linear(&x_n, hidden, hidden * 2);
        // GLU: split last dim in half, mul by sigmoid of the other half.
        let half = hidden;
        let y1 = y.narrow(2_usize, 0, half)?;
        let y2 = y.narrow(2_usize, half, half)?;
        let y = y1.mul(&y2.sigmoid())?;

        // (B, T, C) → (B, C, T) for depthwise conv1d.
        let y_bct = y.permute([0_usize, 2, 1])?;
        // Causal pad (left = k-1) on the temporal axis.
        let y_padded = y_bct.pad_with_zeros(2_usize, k - 1, 0)?;
        let w = layer.lconv_depthwise.const_like(
            &y_padded,
            Shape::from_dims(&[hidden, 1, k]),
        )?;
        let y_conv = y_padded.conv1d(&w, None, 1, 0, hidden)?;
        // (B, C, T) → (B, T, C)
        let y_btc = y_conv.permute([0_usize, 2, 1])?.clamp(-clip, clip);
        let y_n = y_btc.rms_norm_affine(Arc::clone(&layer.lconv_inner_norm), cfg.rms_norm_eps)?;
        let y_act = y_n.silu();
        let y_out = layer.lconv_linear_end.apply_linear(&y_act, hidden, hidden);
        residual.add(&y_out)
    }

    /// Build the chunked attention mask: 0.0 where attendable, a large
    /// negative number where not. Block-band over T_seq.
    fn build_block_band_mask(&self, anchor: &LazyTensor, t_seq: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let mask = chunked_band_mask_values(t_seq, cfg.conf_attention_chunk_size, cfg.conf_left_chunks);
        Ok(anchor.const_f32_like(
            Arc::from(mask),
            Shape::from_dims(&[1, 1, t_seq, t_seq]),
        ))
    }

    /// Build a flat `(T*T,)` U32 index tensor selecting rows from the
    /// `(2*max_rel + 1, num_heads)` Shaw table: index `[i*T + j]` is
    /// `clip(i - j, -max_rel, +max_rel) + max_rel`.
    fn build_rel_pos_indices(&self, anchor: &LazyTensor, t_seq: usize) -> Result<LazyTensor> {
        let max_rel = self.config.max_rel() as isize;
        let mut data = Vec::with_capacity(t_seq * t_seq);
        for i in 0..t_seq {
            for j in 0..t_seq {
                let off = (i as isize) - (j as isize);
                let clipped = off.clamp(-max_rel, max_rel);
                let bucket = (clipped + max_rel) as u32;
                data.push(bucket);
            }
        }
        Ok(anchor.const_u32_like(data, Shape::from_dims(&[t_seq * t_seq])))
    }
}

/// Host-side block-band mask: rows are queries, cols are keys.
/// Returns a flat `t_seq * t_seq` vector of `0.0` on attend and a
/// large negative number on block. Public so the test suite can
/// verify the construction.
///
/// Query at frame `i` attends to keys at frame `j` iff
/// `j / chunk_size ∈ [q_chunk - left_chunks, q_chunk]`, where
/// `q_chunk = i / chunk_size`.
pub fn chunked_band_mask_values(t_seq: usize, chunk_size: usize, left_chunks: usize) -> Vec<f32> {
    let neg = -1.0e9_f32;
    let mut out = vec![neg; t_seq * t_seq];
    if t_seq == 0 || chunk_size == 0 {
        return out;
    }
    for i in 0..t_seq {
        let q_chunk = i / chunk_size;
        let lo_chunk = q_chunk.saturating_sub(left_chunks);
        let hi_chunk = q_chunk;
        let lo_j = lo_chunk * chunk_size;
        let hi_j = (hi_chunk + 1) * chunk_size;
        for j in lo_j..hi_j.min(t_seq) {
            out[i * t_seq + j] = 0.0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Gemma4AudioConfig {
        Gemma4AudioConfig {
            input_feat_size: 8,
            hidden_size: 16,
            output_proj_dims: None,
            conf_attention_chunk_size: 16,
            conf_left_chunks: 1,
            conf_attention_logit_cap: 50.0,
            conf_num_attention_heads: 4,
            conf_num_hidden_layers: 2,
            conf_conv_kernel_size: 5,
            conf_reduction_factor: 1,
            conf_residual_weight: 0.5,
            sscp_conv_channel_size: [16, 16],
            sscp_conv_kernel_size: [[3, 3], [3, 3]],
            sscp_conv_stride_size: [[2, 2], [2, 2]],
            rms_norm_eps: 1e-6,
            gradient_clipping: 1e10,
        }
    }

    fn make_rng(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }

    fn vec_arc(n: usize, rng: &mut impl FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| rng()).collect::<Vec<_>>())
    }

    fn vec_const(n: usize, v: f32) -> Arc<[f32]> {
        Arc::from(vec![v; n])
    }

    fn tiny_weights(cfg: &Gemma4AudioConfig) -> Gemma4AudioWeights {
        let mut rng = make_rng(42);
        let h = cfg.hidden_size;
        let c0 = cfg.sscp_conv_channel_size[0];
        let c1 = cfg.sscp_conv_channel_size[1];
        let kt = cfg.sscp_conv_kernel_size[0][0];
        let kf = cfg.sscp_conv_kernel_size[0][1];
        let kt1 = cfg.sscp_conv_kernel_size[1][0];
        let kf1 = cfg.sscp_conv_kernel_size[1][1];
        let k_lc = cfg.conf_conv_kernel_size;
        let n_heads = cfg.conf_num_attention_heads;
        let span = 2 * cfg.max_rel() + 1;

        let sscp = [
            SscpBlockWeights {
                conv: WeightStorage::F32(vec_arc(c0 * 1 * kt * kf, &mut rng)),
                norm_gain: vec_const(c0, 1.0),
                norm_bias: vec_const(c0, 0.0),
            },
            SscpBlockWeights {
                conv: WeightStorage::F32(vec_arc(c1 * c0 * kt1 * kf1, &mut rng)),
                norm_gain: vec_const(c1, 1.0),
                norm_bias: vec_const(c1, 0.0),
            },
        ];
        // After two strided convs on (T=64, F=8) with stride 2 and 3x3 + pad (kt/2, 1):
        //   T: 64 -> 32 -> 16
        //   F: 8 -> 5 -> 3   (8+2-3)/2+1=4? actually (8+2-3)/2+1 = 4; then (4+2-3)/2+1=2
        // We pre-compute the SSCP F_out for the proj layer.
        let mut current_f = cfg.input_feat_size;
        for i in 0..2 {
            let kf_i = cfg.sscp_conv_kernel_size[i][1];
            let sf_i = cfg.sscp_conv_stride_size[i][1];
            current_f = (current_f + 2 - kf_i) / sf_i + 1;
        }
        let f_out = current_f;
        let sscp_input_proj = WeightStorage::F32(vec_arc(c1 * f_out * h, &mut rng));

        let layers: Vec<ConformerLayerWeights> = (0..cfg.conf_num_hidden_layers)
            .map(|_| ConformerLayerWeights {
                ff1_pre_norm: vec_const(h, 1.0),
                ff1_post_norm: vec_const(h, 1.0),
                ff1_w1: WeightStorage::F32(vec_arc(h * h * 4, &mut rng)),
                ff1_w2: WeightStorage::F32(vec_arc(h * 4 * h, &mut rng)),

                attn_pre_norm: vec_const(h, 1.0),
                attn_post_norm: vec_const(h, 1.0),
                attn_q: WeightStorage::F32(vec_arc(h * h, &mut rng)),
                attn_k: WeightStorage::F32(vec_arc(h * h, &mut rng)),
                attn_v: WeightStorage::F32(vec_arc(h * h, &mut rng)),
                attn_o: WeightStorage::F32(vec_arc(h * h, &mut rng)),
                rel_pos_bias: vec_arc(span * n_heads, &mut rng),

                lconv_pre_norm: vec_const(h, 1.0),
                lconv_inner_norm: vec_const(h, 1.0),
                lconv_linear_start: WeightStorage::F32(vec_arc(h * h * 2, &mut rng)),
                lconv_linear_end: WeightStorage::F32(vec_arc(h * h, &mut rng)),
                lconv_depthwise: WeightStorage::F32(vec_arc(h * 1 * k_lc, &mut rng)),

                ff2_pre_norm: vec_const(h, 1.0),
                ff2_post_norm: vec_const(h, 1.0),
                ff2_w1: WeightStorage::F32(vec_arc(h * h * 4, &mut rng)),
                ff2_w2: WeightStorage::F32(vec_arc(h * 4 * h, &mut rng)),

                out_norm: vec_const(h, 1.0),
            })
            .collect();

        Gemma4AudioWeights {
            sscp,
            sscp_input_proj,
            layers,
            output_proj: None,
        }
    }

    #[test]
    fn forward_shape_and_finite_tiny() {
        let cfg = tiny_config();
        let model = Gemma4AudioModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let b = 1;
        let t_in = 64;
        let n_mels = cfg.input_feat_size;
        let mel_data: Vec<f32> = (0..b * t_in * n_mels)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let mel = LazyTensor::from_f32(
            Arc::from(mel_data),
            Shape::from_dims(&[b, t_in, n_mels]),
            &Device::cpu(),
        );
        let out = model.forward(&mel).unwrap();
        // T: 64 -> 32 -> 16 after two stride-2 convs.
        let dims = out.shape().dims().to_vec();
        assert_eq!(dims, vec![b, 16, cfg.hidden_size],
            "expected (1, 16, {}), got {:?}", cfg.hidden_size, dims);
        let realized = out.realize_f32();
        for &v in &realized {
            assert!(v.is_finite(), "got non-finite encoder output {v}");
        }
    }

    #[test]
    fn chunked_attention_mask_block_band() {
        // 4 chunks of 16 frames each, left_chunks = 1: queries in
        // chunk c attend to keys in chunks {c-1, c} (clamped at 0).
        let chunk = 16;
        let left = 1;
        let t = 4 * chunk;
        let m = chunked_band_mask_values(t, chunk, left);
        for i in 0..t {
            let q_chunk = i / chunk;
            let lo = if q_chunk == 0 { 0 } else { (q_chunk - left) * chunk };
            let hi = (q_chunk + 1) * chunk;
            for j in 0..t {
                let attend = j >= lo && j < hi;
                let val = m[i * t + j];
                if attend {
                    assert_eq!(val, 0.0,
                        "i={i} j={j}: expected attend (0.0), got {val}");
                } else {
                    assert!(val < -1.0e8,
                        "i={i} j={j}: expected block (large negative), got {val}");
                }
            }
        }
    }

    #[test]
    fn rel_pos_bias_table_lookup() {
        // Hand-trace the (i, j) → bucket assignment with tiny config.
        let cfg = tiny_config();
        let max_rel = cfg.max_rel() as isize;
        let span = (2 * max_rel + 1) as usize;
        let t_seq: usize = 4;

        // Mirror Gemma4AudioModel::build_rel_pos_indices.
        let mut expected = Vec::with_capacity(t_seq * t_seq);
        for i in 0..t_seq {
            for j in 0..t_seq {
                let off = (i as isize) - (j as isize);
                let clipped = off.clamp(-max_rel, max_rel);
                expected.push((clipped + max_rel) as u32);
            }
        }
        assert_eq!(expected.len(), t_seq * t_seq);
        // Buckets must be in range.
        for &b in &expected {
            assert!((b as usize) < span, "bucket {b} out of range {span}");
        }
        // Diagonal (i == j) must map to the "zero offset" bucket = max_rel.
        for i in 0..t_seq {
            assert_eq!(expected[i * t_seq + i], max_rel as u32,
                "diagonal at i={i} must be the zero-offset bucket {max_rel}");
        }
        // The (i=3, j=0) offset is +3 which is within max_rel → bucket = max_rel + 3.
        assert_eq!(expected[3 * t_seq + 0], (max_rel + 3) as u32);
        // The (i=0, j=3) offset is -3 → bucket = max_rel - 3.
        assert_eq!(expected[0 * t_seq + 3], (max_rel - 3) as u32);

        // Now build the same index tensor through the model and read
        // it back, asserting the realized values match.
        let model = Gemma4AudioModel { config: cfg, weights: tiny_weights(&tiny_config()) };
        let anchor = LazyTensor::from_f32(
            Arc::from(vec![0.0_f32; t_seq]),
            Shape::from_dims(&[t_seq]),
            &Device::cpu(),
        );
        let idx_t = model.build_rel_pos_indices(&anchor, t_seq).unwrap();
        let realized = idx_t.realize_u32();
        assert_eq!(realized, expected);
    }
}

//! Mamba-2 decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Mamba-2 generalizes Mamba's selective-scan
//! recurrence via the State Space Duality (SSD) algorithm, which
//! reformulates the SSM step as a chunked matrix scan amenable to
//! GPU parallelism. The lazy stack already encapsulates the SSD
//! recurrence as a single [`LazyTensor::ssd_chunk_scan`] op (Phase
//! 7.6 fused op; CPU + CUDA + Vulkan all wired) — the eager code's
//! `segsum` / `reshape_into_chunks` / explicit chunk recurrence
//! disappears.
//!
//! # Pipeline (per mixer block)
//!
//! ```text
//! in_proj   → [z, xbc, dt]
//! conv1d   on xbc → [x, b, c]   (depthwise; D_CONV = 4)
//! silu     on each path
//! dt        = softplus(dt + dt_bias) (handled by ssd_chunk_scan
//!             when delta_softplus stays internal — Mamba-2 has its
//!             own softplus path explicitly)
//! a         = -exp(a_log)
//! y         = ssd_chunk_scan(x.reshape([batch, seq, heads, head_dim]),
//!                            dt, a, b, c, chunk_size)
//! skip      = x * d (per-head)
//! gated     = (y + skip) * silu(z)
//! out_proj(gated)
//! ```
//!
//! # v1 scope
//!
//! - Prefill only; no autoregressive resume (`Op::SsdChunkScanWithInitState`
//!   is the same future addition the Mamba memo flags).
//! - F32 activations.
//! - `ngroups = 1` (the common Mamba-2 default). Multi-group
//!   variants need B/C reshape + per-group broadcast that's a
//!   mechanical extension when a checkpoint demands it.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

/// Mamba-2's conv kernel size. Fixed at 4 across released checkpoints.
pub const D_CONV: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct Mamba2Config {
    pub d_model: usize,
    pub n_layer: usize,
    pub vocab_size: usize,
    pub d_state: usize,
    pub expand: usize,
    pub head_dim: usize,
    pub ngroups: usize,
    pub pad_vocab_size_multiple: usize,
    pub chunk_size: usize,
    pub rms_norm_eps: f64,
}

impl Mamba2Config {
    pub fn vocab_size(&self) -> usize {
        let pad = self.pad_vocab_size_multiple.max(1);
        self.vocab_size.div_ceil(pad) * pad
    }
    pub fn d_inner(&self) -> usize { self.d_model * self.expand }
    pub fn n_heads(&self) -> usize { self.d_inner() / self.head_dim }
    /// Size of the `[x, b, c]` concatenation on the inner dim. The
    /// eager code calls this `d_xbc`.
    pub fn d_xbc(&self) -> usize { self.d_inner() + 2 * self.ngroups * self.d_state }

    /// `state-spaces/mamba2-130m`-class.
    pub fn mamba2_130m() -> Self {
        Self {
            d_model: 768,
            n_layer: 24,
            vocab_size: 50_277,
            d_state: 64,
            expand: 2,
            head_dim: 64,
            ngroups: 1,
            pad_vocab_size_multiple: 16,
            chunk_size: 256,
            rms_norm_eps: 1e-5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mamba2LayerWeights {
    pub norm_gain: Arc<[f32]>,
    /// `[d_model, d_inner + d_xbc + n_heads]` — projects to
    /// `[z, xbc, dt]`.
    pub in_proj: WeightStorage,
    /// `[d_xbc, 1, D_CONV]` depthwise conv kernel (over the `xbc`
    /// concatenation).
    pub conv1d_weight: Arc<[f32]>,
    /// `[d_xbc]` conv bias.
    pub conv1d_bias: Arc<[f32]>,
    /// `[n_heads]` log-eigenvalues. `a = -exp(a_log)`.
    pub a_log: Arc<[f32]>,
    /// `[n_heads]` per-head skip scale.
    pub d: Arc<[f32]>,
    /// `[n_heads]` learned bias added before softplus.
    pub dt_bias: Arc<[f32]>,
    /// `[d_inner]` — pre-out_proj norm gain.
    pub out_norm_gain: Arc<[f32]>,
    /// `[d_inner, d_model]`.
    pub out_proj: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Mamba2Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Mamba2LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Mamba2Model {
    pub config: Mamba2Config,
    pub weights: Mamba2Weights,
}

impl Mamba2Model {
    /// Prefill forward over `tokens`; returns logits
    /// `[1, seq, vocab_size_padded]`.
    ///
    /// `seq` must be divisible by `cfg.chunk_size` for the SSD scan.
    /// Callers that need non-aligned lengths can right-pad to the
    /// next multiple and ignore the trailing logits.
    pub fn forward(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens)?;
        Ok(weights.output.apply_linear(
            &h_norm, cfg.d_model, cfg.vocab_size(),
        ))
    }

    /// Run the Mamba-2 SSD stack forward up to the final
    /// RmsNorm and return per-token hidden states
    /// `(1, seq, d_model)`. v1 is prefill only; `seq` must be
    /// a multiple of `cfg.chunk_size`.
    pub fn forward_hidden(&self, tokens: &[u32]) -> Result<LazyTensor> {
        self.run_backbone(tokens)
    }

    fn run_backbone(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "Mamba2Model::forward: tokens must be non-empty");
        assert!(
            seq % cfg.chunk_size == 0,
            "Mamba2Model::forward: seq ({seq}) must be a multiple of chunk_size ({}). \
             Right-pad to the next multiple before calling.",
            cfg.chunk_size,
        );
        let vocab_padded = cfg.vocab_size();

        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[vocab_padded, cfg.d_model]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.d_model]))?;

        for layer in &weights.layers {
            h = self.apply_residual_block(&h, layer)?;
        }
        Ok(crate::lazy::apply_affine_rms_norm_pub(
            &h, &weights.final_norm_gain, cfg.d_model, cfg.rms_norm_eps,
        ))
    }

    fn apply_residual_block(
        &self,
        x: &LazyTensor,
        layer: &Mamba2LayerWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_norm = crate::lazy::apply_affine_rms_norm_pub(
            x, &layer.norm_gain, cfg.d_model, cfg.rms_norm_eps,
        );
        let mixer_out = self.apply_mixer(&x_norm, layer)?;
        x.add(&mixer_out)
    }

    fn apply_mixer(
        &self,
        x: &LazyTensor,
        layer: &Mamba2LayerWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let d_inner = cfg.d_inner();
        let d_xbc = cfg.d_xbc();
        let n_heads = cfg.n_heads();
        let d_state = cfg.d_state;
        let head_dim = cfg.head_dim;
        let x_shape = x.shape();
        let x_dims = x_shape.dims();
        let batch = x_dims[0];
        let seq = x_dims[1];

        // in_proj: x → [z, xbc, dt] concatenated.
        let proj_size = d_inner + d_xbc + n_heads;
        let in_out = layer.in_proj.apply_linear(x, cfg.d_model, proj_size);
        let z = in_out.slice(2_usize, 0, d_inner)?;
        let xbc = in_out.slice(2_usize, d_inner, d_xbc)?;
        let dt = in_out.slice(2_usize, d_inner + d_xbc, n_heads)?;

        // Conv1d on xbc (depthwise). Pad left D_CONV-1 zeros then call
        // causal_conv1d with use_silu=true (matches eager's silu post-conv).
        let xbc_t = xbc
            .permute([0, 2, 1_usize])?  // [batch, d_xbc, seq]
            .pad_with_zeros(2_usize, D_CONV - 1, 0)?;
        let conv_w = x.const_f32_like(
            layer.conv1d_weight.clone(),
            Shape::from_dims(&[d_xbc, 1, D_CONV]),
        );
        let conv_b = x.const_f32_like(
            layer.conv1d_bias.clone(),
            Shape::from_dims(&[d_xbc]),
        );
        let xbc_conv = xbc_t.causal_conv1d(&conv_w, &conv_b, /* use_silu */ true);
        // Back to [batch, seq, d_xbc].
        let xbc_conv = xbc_conv.permute([0, 2, 1_usize])?;

        // Split conv'd xbc → [x_path, b, c].
        let bc_dim = cfg.ngroups * d_state;
        let x_path = xbc_conv.slice(2_usize, 0, d_inner)?;
        let b = xbc_conv.slice(2_usize, d_inner, bc_dim)?;
        let c = xbc_conv.slice(2_usize, d_inner + bc_dim, bc_dim)?;

        // Reshape x_path for SSD: [batch, seq, n_heads, head_dim].
        let x_heads = x_path.reshape(Shape::from_dims(&[batch, seq, n_heads, head_dim]))?;
        // b, c: reshape to [batch, seq, ngroups, d_state] then broadcast
        // to per-head if ngroups < n_heads. For ngroups == 1, broadcast
        // to all heads via reshape + broadcast_to.
        let b_g = b.reshape(Shape::from_dims(&[batch, seq, cfg.ngroups, d_state]))?;
        let c_g = c.reshape(Shape::from_dims(&[batch, seq, cfg.ngroups, d_state]))?;
        let (b_heads, c_heads) = if cfg.ngroups == n_heads {
            (b_g, c_g)
        } else {
            assert!(
                n_heads % cfg.ngroups == 0,
                "Mamba2: n_heads ({n_heads}) must be a multiple of ngroups ({})",
                cfg.ngroups,
            );
            let n_per_group = n_heads / cfg.ngroups;
            let expand = |t: LazyTensor| -> Result<LazyTensor> {
                // [batch, seq, ngroups, d_state] → [batch, seq, ngroups, 1, d_state]
                // → broadcast to [batch, seq, ngroups, n_per_group, d_state]
                // → reshape to [batch, seq, n_heads, d_state].
                let s5 = t.reshape(Shape::from_dims(&[
                    batch, seq, cfg.ngroups, 1, d_state,
                ]))?;
                let bcast = s5.broadcast_to(Shape::from_dims(&[
                    batch, seq, cfg.ngroups, n_per_group, d_state,
                ]))?;
                bcast.reshape(Shape::from_dims(&[batch, seq, n_heads, d_state]))
            };
            (expand(b_g)?, expand(c_g)?)
        };

        // dt: add learned bias + softplus. The ssd_chunk_scan op takes
        // dt as `[batch, seq, n_heads]`; the eager code applies softplus
        // BEFORE passing into the scan. We replicate that here.
        let dt_bias_t = x.const_f32_like(
            layer.dt_bias.clone(),
            Shape::from_dims(&[n_heads]),
        );
        let dt_biased = dt.broadcast_add(&dt_bias_t)?;
        // softplus(x) = ln(1 + exp(x)). Use the existing primitive chain.
        let dt_soft = dt_biased.exp().add_scalar(1.0).log();

        // a = -exp(a_log). a_log is `[n_heads]`.
        let a_log = x.const_f32_like(
            layer.a_log.clone(),
            Shape::from_dims(&[n_heads]),
        );
        let a = a_log.exp().neg();

        // SSD scan: y = ssd_chunk_scan(x_heads, dt, a, b, c, chunk_size).
        // Returns `[batch, seq, n_heads, head_dim]`.
        let y = x_heads.ssd_chunk_scan(&dt_soft, &a, &b_heads, &c_heads, cfg.chunk_size);

        // Skip path: y + x_heads * d (per-head).
        let d_t = x.const_f32_like(
            layer.d.clone(),
            Shape::from_dims(&[n_heads]),
        );
        // Broadcast d from [n_heads] across [batch, seq, n_heads, head_dim]
        // (last-axis broadcast multiplication via reshape).
        let d_per_head = d_t.reshape(Shape::from_dims(&[1, 1, n_heads, 1]))?;
        let skip = x_heads.broadcast_mul(&d_per_head)?;
        let y_with_skip = y.add(&skip)?;

        // Flatten head dims back to [batch, seq, d_inner].
        let y_flat = y_with_skip.reshape(Shape::from_dims(&[batch, seq, d_inner]))?;

        // RMS-normed gate path: out_norm(y_flat) * silu(z).
        let y_normed = crate::lazy::apply_affine_rms_norm_pub(
            &y_flat, &layer.out_norm_gain, d_inner, cfg.rms_norm_eps,
        );
        let gated = y_normed.mul(&z.silu())?;

        // out_proj: d_inner → d_model.
        Ok(layer.out_proj.apply_linear(&gated, d_inner, cfg.d_model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &Mamba2Config) -> Mamba2Weights {
        let mut s: u32 = 31415;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let d_model = cfg.d_model;
        let d_inner = cfg.d_inner();
        let d_xbc = cfg.d_xbc();
        let n_heads = cfg.n_heads();
        let vocab = cfg.vocab_size();
        let mut next_box: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(vocab * d_model, &mut *next_box);
        let proj_size = d_inner + d_xbc + n_heads;
        let layers: Vec<Mamba2LayerWeights> = (0..cfg.n_layer).map(|_| Mamba2LayerWeights {
            norm_gain:       Arc::from(vec![1.0_f32; d_model]),
            in_proj:         WeightStorage::F32(vec_of(d_model * proj_size, &mut *next_box)),
            conv1d_weight:   vec_of(d_xbc * D_CONV, &mut *next_box),
            conv1d_bias:     vec_of(d_xbc, &mut *next_box),
            a_log:           Arc::from((0..n_heads).map(|i| -1.0 - 0.01 * i as f32).collect::<Vec<_>>()),
            d:               vec_of(n_heads, &mut *next_box),
            dt_bias:         vec_of(n_heads, &mut *next_box),
            out_norm_gain:   Arc::from(vec![1.0_f32; d_inner]),
            out_proj:        WeightStorage::F32(vec_of(d_inner * d_model, &mut *next_box)),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; d_model]);
        let output = WeightStorage::F32(vec_of(d_model * vocab, &mut *next_box));
        Mamba2Weights { token_embedding, layers, final_norm_gain, output }
    }

    /// Smoke test on a chunk-aligned sequence. seq must be a multiple
    /// of `chunk_size`; we set both to 4 for the tiny config.
    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = Mamba2Config {
            d_model: 16,
            n_layer: 2,
            vocab_size: 32,
            d_state: 8,
            expand: 2,
            head_dim: 4,
            ngroups: 1,
            pad_vocab_size_multiple: 8,
            chunk_size: 4,
            rms_norm_eps: 1e-5,
        };
        let model = Mamba2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        // seq = 8 (multiple of chunk_size = 4).
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let logits = model.forward(&tokens).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size()]);
        let out = logits.realize_f32();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    #[test]
    fn config_derived_sizes_match_reference() {
        let cfg = Mamba2Config::mamba2_130m();
        // d_inner = 768 * 2 = 1536; n_heads = 1536 / 64 = 24
        assert_eq!(cfg.d_inner(), 1536);
        assert_eq!(cfg.n_heads(), 24);
        // d_xbc = d_inner + 2*ngroups*d_state = 1536 + 2*1*64 = 1664
        assert_eq!(cfg.d_xbc(), 1664);
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = Mamba2Config {
            d_model: 16, n_layer: 2, vocab_size: 32, d_state: 8,
            expand: 2, head_dim: 4, ngroups: 1, pad_vocab_size_multiple: 8,
            chunk_size: 4, rms_norm_eps: 1e-5,
        };
        let model = Mamba2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let hidden = model.forward_hidden(&tokens).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.d_model]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

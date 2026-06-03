//! Mamba (v1) decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Mamba is a state-space model (SSM), not an
//! attention model — completely different from the LLaMA-cousins.
//! Each layer is a selective-scan recurrence wrapped in a residual
//! block:
//!
//! ```text
//! residual:  x → norm → mixer → + → out
//!                                ↑
//!                                x
//! mixer:     in_proj → split → [x_path, z_path]
//!            x_path → causal_conv1d → silu → x_proj → split → [delta, b, c]
//!            delta  → dt_proj → softplus → broadcast
//!            a       = -exp(a_log)
//!            y       = selective_scan(x_path, delta, a, b, c)
//!            out     = (y + x_path * d) * silu(z_path)
//!            out_proj(out)
//! ```
//!
//! # What v1 ships
//!
//! - **Prefill only**: `forward(tokens) -> [1, seq, vocab]`. The
//!   recurrence starts at h = 0 each call.
//! - **No autoregressive resume**: a follow-up needs an
//!   `Op::SelectiveScanWithInitState` (or a 6-input variant of
//!   `selective_scan`) that takes the previous step's `last_state`
//!   as input. The multi-output session memo flagged this as the
//!   Mamba decode-loop blocker. `selective_scan_bundled` already
//!   produces `last_state` as a slot; consumer migration just needs
//!   the init-state input on the next forward.
//! - **F32 activations**; weights via [`crate::lazy::WeightStorage`].
//!
//! # Op surface used (already in lazy stack)
//!
//! - [`LazyTensor::causal_conv1d`] — Mamba-1 prefill conv with
//!   optional fused SiLU.
//! - [`LazyTensor::selective_scan`] — the SSM scan; returns the `y`
//!   slot of the bundled producer.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

/// Mamba's conv kernel size. Fixed at 4 in every released Mamba
/// checkpoint.
pub const D_CONV: usize = 4;

/// Mamba's hidden-state size per channel. Fixed at 16 in every
/// released checkpoint.
pub const D_STATE: usize = 16;

#[derive(Debug, Clone, PartialEq)]
pub struct MambaConfig {
    pub d_model: usize,
    pub n_layer: usize,
    pub vocab_size: usize,
    /// Vocab is rounded up to a multiple of this (typically 8) at
    /// load time so the lm_head matmul is aligned.
    pub pad_vocab_size_multiple: usize,
    pub rms_norm_eps: f64,
}

impl MambaConfig {
    /// Effective vocab size after padding.
    pub fn vocab_size(&self) -> usize {
        let pad = self.pad_vocab_size_multiple.max(1);
        self.vocab_size.div_ceil(pad) * pad
    }

    /// `delta`'s low-rank dim (rank of the dt projection).
    pub fn dt_rank(&self) -> usize {
        self.d_model.div_ceil(16)
    }

    /// Inner expanded dim: Mamba doubles the channel count internally.
    pub fn d_inner(&self) -> usize {
        self.d_model * 2
    }

    /// Default 130M-class config from the Mamba paper. Matches
    /// `state-spaces/mamba-130m`.
    pub fn mamba_130m() -> Self {
        Self {
            d_model: 768,
            n_layer: 24,
            vocab_size: 50_277,
            pad_vocab_size_multiple: 8,
            rms_norm_eps: 1e-5,
        }
    }
}

/// Per-layer weights for one Mamba mixer block.
#[derive(Debug, Clone)]
pub struct MambaLayerWeights {
    /// Pre-mixer RmsNorm gain. `[d_model]`.
    pub norm_gain: Arc<[f32]>,
    /// `[d_model, 2 * d_inner]` — projects to `[x_path, z_path]`.
    pub in_proj: WeightStorage,
    /// `[d_inner, 1, D_CONV]` causal-conv kernel; depthwise (groups
    /// = d_inner). `causal_conv1d` reshape convention.
    pub conv1d_weight: Arc<[f32]>,
    /// `[d_inner]` conv bias.
    pub conv1d_bias: Arc<[f32]>,
    /// `[d_inner, dt_rank + 2 * D_STATE]` — projects to
    /// `[delta_low_rank, b, c]`.
    pub x_proj: WeightStorage,
    /// `[dt_rank, d_inner]` — projects the low-rank delta up to
    /// d_inner. Has a bias (required per the Mamba reference).
    pub dt_proj: WeightStorage,
    pub dt_proj_bias: Arc<[f32]>,
    /// `[d_inner, D_STATE]` — `A_log`, the log of the negative-real
    /// SSM eigenvalues. `a = -exp(A_log)`.
    pub a_log: Arc<[f32]>,
    /// `[d_inner]` — the per-channel skip-path scaling.
    pub d: Arc<[f32]>,
    /// `[d_inner, d_model]` — projects back to d_model.
    pub out_proj: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct MambaWeights {
    /// `[vocab_size_padded, d_model]`.
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<MambaLayerWeights>,
    /// Final RmsNorm gain. `[d_model]`.
    pub final_norm_gain: Arc<[f32]>,
    /// `[d_model, vocab_size_padded]`. Often tied to `token_embedding`
    /// in real Mamba checkpoints; the safetensors loader resolves.
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct MambaModel {
    pub config: MambaConfig,
    pub weights: MambaWeights,
}

impl MambaModel {
    /// Run a full-sequence prefill forward and return the logits
    /// `[1, seq, vocab_size_padded]`.
    pub fn forward(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens)?;
        Ok(weights.output.apply_linear(
            &h_norm, cfg.d_model, cfg.vocab_size(),
        ))
    }

    /// Run the Mamba SSM stack forward up to the final RmsNorm
    /// and return per-token hidden states `(1, seq, d_model)`.
    /// No `start_pos` parameter — Mamba's recurrent state is
    /// implicit in the SSM scan; v1 is prefill only.
    pub fn forward_hidden(&self, tokens: &[u32]) -> Result<LazyTensor> {
        self.run_backbone(tokens)
    }

    fn run_backbone(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "MambaModel::forward: tokens must be non-empty");
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
        layer: &MambaLayerWeights,
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
        layer: &MambaLayerWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let d_inner = cfg.d_inner();
        let dt_rank = cfg.dt_rank();
        let x_shape = x.shape();
        let x_dims = x_shape.dims();
        let batch = x_dims[0];
        let seq = x_dims[1];

        // in_proj: x → [x_path, z_path] each `[batch, seq, d_inner]`.
        let xz = layer.in_proj.apply_linear(x, cfg.d_model, 2 * d_inner);
        let x_path = xz.slice(2_usize, 0, d_inner)?;
        let z_path = xz.slice(2_usize, d_inner, d_inner)?;

        // Causal 1d conv along the seq axis. `causal_conv1d` expects
        // `[batch, channels, seq + (kernel - 1)]` — caller left-pads.
        // For prefill we left-pad with `D_CONV - 1` zeros.
        let x_for_conv = x_path
            .permute([0, 2, 1_usize])?  // [batch, d_inner, seq]
            .pad_with_zeros(2_usize, D_CONV - 1, 0)?;  // pad LEFT with kernel-1 zeros

        let conv_w = x.const_f32_like(
            layer.conv1d_weight.clone(),
            Shape::from_dims(&[d_inner, 1, D_CONV]),
        );
        let conv_b = x.const_f32_like(
            layer.conv1d_bias.clone(),
            Shape::from_dims(&[d_inner]),
        );
        // Fused SiLU activation.
        let x_conv = x_for_conv.causal_conv1d(&conv_w, &conv_b, /* use_silu */ true);
        // x_conv shape: [batch, d_inner, seq]. Transpose to [batch, seq, d_inner].
        let x_conv = x_conv.permute([0, 2, 1_usize])?;

        // x_proj: x_conv → [delta_low_rank, b, c].
        let proj = layer.x_proj.apply_linear(&x_conv, d_inner, dt_rank + 2 * D_STATE);
        let delta_low = proj.slice(2_usize, 0, dt_rank)?;
        let b = proj.slice(2_usize, dt_rank, D_STATE)?;
        let c = proj.slice(2_usize, dt_rank + D_STATE, D_STATE)?;

        // dt_proj: [batch, seq, dt_rank] → [batch, seq, d_inner].
        // dt_proj has a bias.
        let delta = layer.dt_proj.apply_linear(&delta_low, dt_rank, d_inner);
        let dt_bias_t = x.const_f32_like(
            layer.dt_proj_bias.clone(),
            Shape::from_dims(&[d_inner]),
        );
        let delta = delta.broadcast_add(&dt_bias_t)?;

        // a = -exp(a_log). a_log is `[d_inner, D_STATE]`.
        let a_log = x.const_f32_like(
            layer.a_log.clone(),
            Shape::from_dims(&[d_inner, D_STATE]),
        );
        let a = a_log.exp().neg();

        // d is `[d_inner]` — broadcast across batch + seq for the
        // skip-path addition.
        let d_t = x.const_f32_like(
            layer.d.clone(),
            Shape::from_dims(&[d_inner]),
        );

        // Selective scan: u = x_conv, returns y `[batch, seq, d_inner]`.
        // The `delta_softplus = true` flag tells the kernel to apply
        // softplus internally — mirroring the eager `softplus(delta)`.
        let y = x_conv.selective_scan(&delta, &a, &b, &c, /* delta_softplus */ true);

        // Skip path: y + x_conv * d (broadcast d across batch + seq).
        let skip = x_conv.broadcast_mul(&d_t)?;
        let y_with_skip = y.add(&skip)?;

        // Gate by z_path (silu(z)).
        let z_silu = z_path.silu();
        let gated = y_with_skip.mul(&z_silu)?;

        // out_proj: d_inner → d_model.
        Ok(layer.out_proj.apply_linear(&gated, d_inner, cfg.d_model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &MambaConfig) -> MambaWeights {
        let mut s: u32 = 1234;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let d_model = cfg.d_model;
        let d_inner = cfg.d_inner();
        let dt_rank = cfg.dt_rank();
        let vocab = cfg.vocab_size();
        let mut next_box: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(vocab * d_model, &mut *next_box);
        let layers: Vec<MambaLayerWeights> = (0..cfg.n_layer).map(|_| MambaLayerWeights {
            norm_gain: Arc::from(vec![1.0_f32; d_model]),
            in_proj: WeightStorage::F32(vec_of(d_model * 2 * d_inner, &mut *next_box)),
            conv1d_weight: vec_of(d_inner * 1 * D_CONV, &mut *next_box),
            conv1d_bias:   vec_of(d_inner, &mut *next_box),
            x_proj: WeightStorage::F32(vec_of(d_inner * (dt_rank + 2 * D_STATE), &mut *next_box)),
            dt_proj: WeightStorage::F32(vec_of(dt_rank * d_inner, &mut *next_box)),
            dt_proj_bias: vec_of(d_inner, &mut *next_box),
            // a_log: small negatives so a = -exp(a_log) stays stable.
            a_log: Arc::from((0..d_inner * D_STATE).map(|i| -1.0 - 0.01 * (i as f32)).collect::<Vec<_>>()),
            d: vec_of(d_inner, &mut *next_box),
            out_proj: WeightStorage::F32(vec_of(d_inner * d_model, &mut *next_box)),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; d_model]);
        let output = WeightStorage::F32(vec_of(d_model * vocab, &mut *next_box));
        MambaWeights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = MambaConfig {
            d_model: 16,
            n_layer: 2,
            vocab_size: 24,
            pad_vocab_size_multiple: 8,
            rms_norm_eps: 1e-5,
        };
        // After padding, vocab → 24 (already a multiple of 8).
        let model = MambaModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7];
        let logits = model.forward(&tokens).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size()]);
        let out = logits.realize_f32();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    #[test]
    fn vocab_padding_rounds_up() {
        let cfg = MambaConfig {
            d_model: 16, n_layer: 1, vocab_size: 50_277,
            pad_vocab_size_multiple: 8, rms_norm_eps: 1e-5,
        };
        // 50_277 → 50_280 (next multiple of 8).
        assert_eq!(cfg.vocab_size(), 50_280);
    }

    #[test]
    fn dt_rank_is_d_model_div_ceil_16() {
        let cfg = MambaConfig {
            d_model: 768, n_layer: 1, vocab_size: 256,
            pad_vocab_size_multiple: 8, rms_norm_eps: 1e-5,
        };
        assert_eq!(cfg.dt_rank(), 48);
        assert_eq!(cfg.d_inner(), 1536);
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = MambaConfig {
            d_model: 16, n_layer: 2, vocab_size: 24,
            pad_vocab_size_multiple: 8, rms_norm_eps: 1e-5,
        };
        let model = MambaModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.d_model]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}

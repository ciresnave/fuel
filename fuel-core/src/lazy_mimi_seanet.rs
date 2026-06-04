//! Mimi SeaNet — lazy port.
//!
//! Convolutional encoder/decoder that surrounds Mimi's
//! transformer + quantizer. The encoder progressively downsamples
//! raw audio to a latent sequence using strided dilated
//! convolutions separated by residual blocks; the decoder mirrors
//! the encoder with transposed convolutions to reconstruct audio.
//!
//! Block layout:
//!
//!   - **`SeaNetResnetBlock`** — `for conv in convs: y = conv(act(y))`,
//!     then `+ x` (or `+ shortcut(x)` if `true_skip = false`).
//!     Each block has two dilated convs (kernel × dilation) +
//!     optional 1×1 skip-shortcut conv.
//!
//!   - **`SeaNetEncoder`** — `init_conv → for each layer: [N
//!     residuals + activation + downsample (kernel = 2·ratio,
//!     stride = ratio)] → activation → final_conv`. Channel dim
//!     doubles at every downsample.
//!
//!   - **`SeaNetDecoder`** — `init_conv → for each layer:
//!     [activation + upsample (transpose conv, kernel = 2·ratio,
//!     stride = ratio) + N residuals] → activation → final_conv →
//!     optional final activation`. Channel dim halves at every
//!     upsample.
//!
//! All convs are **causal**: pad-left-only by
//! `(kernel - 1) · dilation` zeros (or replicated edge per
//! `PadMode`), no right-pad on the inference path. Dilation is
//! handled by **expanding the weight** with zero-interleaved
//! taps (via [`crate::lazy_dac::expand_conv1d_weight_for_dilation_if_needed`])
//! so plain non-dilated `conv1d` produces the dilated output.
//!
//! v1 scope: F32, batch == 1, forward-only inference. No
//! WeightNorm (assumed pre-baked into weights at load time;
//! WeightNorm renormalizes `g · v / ||v||` — a load-time
//! preprocess for inference). No streaming `step` API. No LSTM
//! (Mimi v0.1 has `lstm = 0`; the eager port also bails when
//! `lstm > 0`).

use crate::lazy::LazyTensor;
use crate::lazy_dac::expand_conv1d_weight_for_dilation_if_needed;
use crate::lazy_encodec::{pad1d, PadMode};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeaNetActivation {
    /// ELU with α=1.0 (Mimi v0.1 default).
    Elu1,
    Gelu,
    Relu,
    /// `silu(x) = x · sigmoid(x)`.
    Silu,
}

#[derive(Debug, Clone)]
pub struct SeaNetConfig {
    /// Output latent dim (= encoder output channels = decoder input channels).
    pub dimension: usize,
    /// Audio channel count (1 for mono Mimi).
    pub channels: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    /// Stride list for the encoder's downsample stages (decoder
    /// reverses this order for upsample).
    pub ratios: Vec<usize>,
    pub activation: SeaNetActivation,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub dilation_base: usize,
    pub pad_mode: PadMode,
    pub true_skip: bool,
    pub compress: usize,
    pub final_activation: Option<SeaNetActivation>,
}

impl SeaNetConfig {
    /// Mimi v0.1 SeaNet preset (24 kHz audio, ratios `[8, 6, 5, 4]`).
    pub fn mimi_v0_1() -> Self {
        Self {
            dimension: 512,
            channels: 1,
            n_filters: 64,
            n_residual_layers: 1,
            ratios: vec![8, 6, 5, 4],
            activation: SeaNetActivation::Elu1,
            kernel_size: 7,
            residual_kernel_size: 3,
            last_kernel_size: 3,
            dilation_base: 2,
            pad_mode: PadMode::Constant,
            true_skip: true,
            compress: 2,
            final_activation: None,
        }
    }
}

// ---- Weight structures ------------------------------------------------------

/// Forward-only `Conv1d` weights (no WeightNorm runtime renormalize).
#[derive(Debug, Clone)]
pub struct LazyConv1dWeights {
    /// Stored `(out_channels, in_channels / groups, kernel_size)`.
    pub weight: Arc<[f32]>,
    /// `(out_channels,)` — present unless this conv was built without bias.
    pub bias: Option<Arc<[f32]>>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

#[derive(Debug, Clone)]
pub struct LazyConvTranspose1dWeights {
    /// Stored `(in_channels, out_channels / groups, kernel_size)` to
    /// match PyTorch's `ConvTranspose1d.weight` layout.
    pub weight: Arc<[f32]>,
    pub bias: Option<Arc<[f32]>>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub groups: usize,
}

#[derive(Debug, Clone)]
pub struct SeaNetResnetBlockWeights {
    /// Two dilated convs per block (residual_kernel, then 1×1).
    pub convs: Vec<LazyConv1dWeights>,
    /// Optional 1×1 shortcut conv when `true_skip = false`.
    pub shortcut: Option<LazyConv1dWeights>,
}

#[derive(Debug, Clone)]
pub struct SeaNetEncoderLayerWeights {
    pub residuals: Vec<SeaNetResnetBlockWeights>,
    pub downsample: LazyConv1dWeights,
}

#[derive(Debug, Clone)]
pub struct SeaNetDecoderLayerWeights {
    pub upsample: LazyConvTranspose1dWeights,
    pub residuals: Vec<SeaNetResnetBlockWeights>,
}

#[derive(Debug, Clone)]
pub struct SeaNetEncoderWeights {
    pub init_conv: LazyConv1dWeights,
    pub layers: Vec<SeaNetEncoderLayerWeights>,
    pub final_conv: LazyConv1dWeights,
}

#[derive(Debug, Clone)]
pub struct SeaNetDecoderWeights {
    pub init_conv: LazyConv1dWeights,
    pub layers: Vec<SeaNetDecoderLayerWeights>,
    pub final_conv: LazyConv1dWeights,
}

// ---- Forward helpers -------------------------------------------------------

fn apply_activation(x: &LazyTensor, act: SeaNetActivation) -> LazyTensor {
    match act {
        SeaNetActivation::Elu1 => x.elu(1.0),
        SeaNetActivation::Gelu => x.gelu(),
        SeaNetActivation::Relu => x.relu(),
        SeaNetActivation::Silu => x.silu(),
    }
}

/// Causal conv1d forward: left-pad with `(kernel-1) · dilation`
/// then apply non-dilated conv1d (dilation is folded into the
/// weight via zero-interleave).
fn apply_causal_conv1d(
    x: &LazyTensor, w: &LazyConv1dWeights, pad_mode: PadMode,
) -> Result<LazyTensor> {
    let effective_k = (w.kernel_size - 1) * w.dilation + 1;
    let pad_total = effective_k.saturating_sub(w.stride);
    let padded = pad1d(x, pad_total, 0, pad_mode, x)?;
    let (weight_v, expanded_k) = expand_conv1d_weight_for_dilation_if_needed(
        &w.weight,
        w.out_channels, w.in_channels / w.groups, w.kernel_size,
        w.dilation,
    );
    debug_assert_eq!(expanded_k, effective_k);
    let weight_arc: Arc<[f32]> = Arc::from(weight_v);
    let weight = padded.const_f32_like(
        weight_arc,
        Shape::from_dims(&[w.out_channels, w.in_channels / w.groups, effective_k]),
    );
    let bias_t = w.bias.as_ref().map(|b| {
        padded.const_f32_like(
            Arc::clone(b), Shape::from_dims(&[w.out_channels]),
        )
    });
    padded.conv1d(&weight, bias_t.as_ref(), w.stride, 0, w.groups)
}

/// Causal conv_transpose1d for upsample. Mimi's upsample kernel is
/// `2 · stride`, with the natural transposed-conv output trimmed
/// to remove the trailing acausal tap region.
fn apply_causal_conv_transpose1d(
    x: &LazyTensor, w: &LazyConvTranspose1dWeights,
) -> Result<LazyTensor> {
    let weight = x.const_f32_like(
        Arc::clone(&w.weight),
        Shape::from_dims(&[w.in_channels, w.out_channels / w.groups, w.kernel_size]),
    );
    // Use LazyTensor::conv_transpose1d (composite shipped earlier this
    // session, layered over conv_transpose2d via rank-3 ↔ rank-4 lift).
    let y = x.conv_transpose1d(
        &weight, w.stride, /* padding */ 0, /* output_padding */ 0,
        /* dilation */ 1, w.groups,
    )?;
    // Apply bias via broadcast_add since conv_transpose1d doesn't take it.
    let y = match &w.bias {
        None => y,
        Some(b) => {
            let bias = x
                .const_f32_like(Arc::clone(b), Shape::from_dims(&[w.out_channels]))
                .reshape(Shape::from_dims(&[1, w.out_channels, 1]))?
                .broadcast_to(Shape::from_dims(y.shape().dims()))?;
            y.add(&bias)?
        }
    };
    // Causal trim: ConvTranspose1d's natural output length is
    // `(T_in - 1) · stride + kernel`. For Mimi's `kernel = 2·stride`
    // the causal output length is `T_in · stride`; we remove the
    // trailing `kernel - stride = stride` tail.
    let dims = y.shape().dims().to_vec();
    let t_out = dims[2];
    let trim_right = w.kernel_size.saturating_sub(w.stride);
    let keep = t_out.saturating_sub(trim_right);
    y.narrow(2_usize, 0, keep)
}

fn apply_resnet_block(
    x: &LazyTensor, w: &SeaNetResnetBlockWeights,
    activation: SeaNetActivation, pad_mode: PadMode,
) -> Result<LazyTensor> {
    let mut y = x.clone();
    for conv in &w.convs {
        y = apply_activation(&y, activation);
        y = apply_causal_conv1d(&y, conv, pad_mode)?;
    }
    let skip = match &w.shortcut {
        None => x.clone(),
        Some(shortcut) => apply_causal_conv1d(x, shortcut, pad_mode)?,
    };
    y.add(&skip)
}

// ---- Public model APIs -----------------------------------------------------

#[derive(Debug, Clone)]
pub struct SeaNetEncoderModel {
    pub config: SeaNetConfig,
    pub weights: SeaNetEncoderWeights,
}

#[derive(Debug, Clone)]
pub struct SeaNetDecoderModel {
    pub config: SeaNetConfig,
    pub weights: SeaNetDecoderWeights,
}

impl SeaNetEncoderModel {
    /// Encode raw audio `(1, channels, T)` to latent `(1, dimension, T_latent)`.
    pub fn forward(&self, audio: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let mut x = apply_causal_conv1d(audio, &self.weights.init_conv, cfg.pad_mode)?;
        for layer in &self.weights.layers {
            for res in &layer.residuals {
                x = apply_resnet_block(&x, res, cfg.activation, cfg.pad_mode)?;
            }
            x = apply_activation(&x, cfg.activation);
            x = apply_causal_conv1d(&x, &layer.downsample, cfg.pad_mode)?;
        }
        let x = apply_activation(&x, cfg.activation);
        apply_causal_conv1d(&x, &self.weights.final_conv, cfg.pad_mode)
    }
}

impl SeaNetDecoderModel {
    /// Decode latent `(1, dimension, T_latent)` to audio
    /// `(1, channels, T)`.
    pub fn forward(&self, latent: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let mut x = apply_causal_conv1d(latent, &self.weights.init_conv, cfg.pad_mode)?;
        for layer in &self.weights.layers {
            x = apply_activation(&x, cfg.activation);
            x = apply_causal_conv_transpose1d(&x, &layer.upsample)?;
            for res in &layer.residuals {
                x = apply_resnet_block(&x, res, cfg.activation, cfg.pad_mode)?;
            }
        }
        let x = apply_activation(&x, cfg.activation);
        let x = apply_causal_conv1d(&x, &self.weights.final_conv, cfg.pad_mode)?;
        Ok(match cfg.final_activation {
            None => x,
            Some(act) => apply_activation(&x, act),
        })
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }
    fn vec_of(n: usize, nb: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| nb()).collect::<Vec<_>>())
    }

    fn conv_w(
        in_c: usize, out_c: usize, k: usize, stride: usize, dilation: usize, groups: usize,
        bias: bool, nb: &mut dyn FnMut() -> f32,
    ) -> LazyConv1dWeights {
        LazyConv1dWeights {
            weight: vec_of(out_c * (in_c / groups) * k, nb),
            bias: if bias { Some(vec_of(out_c, nb)) } else { None },
            in_channels: in_c, out_channels: out_c,
            kernel_size: k, stride, dilation, groups,
        }
    }

    fn conv_tr_w(
        in_c: usize, out_c: usize, k: usize, stride: usize, groups: usize,
        bias: bool, nb: &mut dyn FnMut() -> f32,
    ) -> LazyConvTranspose1dWeights {
        LazyConvTranspose1dWeights {
            weight: vec_of(in_c * (out_c / groups) * k, nb),
            bias: if bias { Some(vec_of(out_c, nb)) } else { None },
            in_channels: in_c, out_channels: out_c,
            kernel_size: k, stride, groups,
        }
    }

    fn resnet_block_w(
        dim: usize, k: usize, dilation: usize, compress: usize, true_skip: bool,
        nb: &mut dyn FnMut() -> f32,
    ) -> SeaNetResnetBlockWeights {
        let hidden = dim / compress;
        SeaNetResnetBlockWeights {
            convs: vec![
                conv_w(dim, hidden, k, 1, dilation, 1, true, nb),
                conv_w(hidden, dim, 1, 1, 1, 1, true, nb),
            ],
            shortcut: if true_skip { None } else { Some(conv_w(dim, dim, 1, 1, 1, 1, true, nb)) },
        }
    }

    fn tiny_cfg() -> SeaNetConfig {
        SeaNetConfig {
            dimension: 8, channels: 1,
            n_filters: 2,
            n_residual_layers: 1,
            ratios: vec![2, 2],
            activation: SeaNetActivation::Elu1,
            kernel_size: 3, residual_kernel_size: 3, last_kernel_size: 3,
            dilation_base: 2, pad_mode: PadMode::Constant,
            true_skip: true, compress: 2,
            final_activation: None,
        }
    }

    fn build_encoder(cfg: &SeaNetConfig) -> SeaNetEncoderModel {
        let mut nb = rng_seed(2026);
        let mut mult = 1_usize;
        let init_conv = conv_w(cfg.channels, mult * cfg.n_filters, cfg.kernel_size, 1, 1, 1, true, &mut nb);
        let mut layers = Vec::with_capacity(cfg.ratios.len());
        for ratio in cfg.ratios.iter().rev() {
            let dim = mult * cfg.n_filters;
            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                residuals.push(resnet_block_w(
                    dim, cfg.residual_kernel_size,
                    cfg.dilation_base.pow(j as u32),
                    cfg.compress, cfg.true_skip, &mut nb,
                ));
            }
            let downsample = conv_w(dim, dim * 2, ratio * 2, *ratio, 1, 1, true, &mut nb);
            layers.push(SeaNetEncoderLayerWeights { residuals, downsample });
            mult *= 2;
        }
        let final_conv = conv_w(
            mult * cfg.n_filters, cfg.dimension, cfg.last_kernel_size, 1, 1, 1, true, &mut nb,
        );
        SeaNetEncoderModel {
            config: cfg.clone(),
            weights: SeaNetEncoderWeights { init_conv, layers, final_conv },
        }
    }

    fn build_decoder(cfg: &SeaNetConfig) -> SeaNetDecoderModel {
        let mut nb = rng_seed(2027);
        let mut mult = 1_usize << cfg.ratios.len();
        let init_conv = conv_w(cfg.dimension, mult * cfg.n_filters, cfg.kernel_size, 1, 1, 1, true, &mut nb);
        let mut layers = Vec::with_capacity(cfg.ratios.len());
        for ratio in cfg.ratios.iter() {
            let dim = mult * cfg.n_filters;
            let out_dim = dim / 2;
            let upsample = conv_tr_w(dim, out_dim, ratio * 2, *ratio, 1, true, &mut nb);
            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                residuals.push(resnet_block_w(
                    out_dim, cfg.residual_kernel_size,
                    cfg.dilation_base.pow(j as u32),
                    cfg.compress, cfg.true_skip, &mut nb,
                ));
            }
            layers.push(SeaNetDecoderLayerWeights { upsample, residuals });
            mult /= 2;
        }
        let final_conv = conv_w(
            cfg.n_filters, cfg.channels, cfg.last_kernel_size, 1, 1, 1, true, &mut nb,
        );
        SeaNetDecoderModel {
            config: cfg.clone(),
            weights: SeaNetDecoderWeights { init_conv, layers, final_conv },
        }
    }

    #[test]
    fn encoder_forward_shape_and_finite() {
        let cfg = tiny_cfg();
        let encoder = build_encoder(&cfg);
        // Input audio length must be divisible by total stride.
        let total_stride: usize = cfg.ratios.iter().product();
        let t_in = total_stride * 4;
        let audio = LazyTensor::from_f32(
            (0..(1 * cfg.channels * t_in)).map(|i| (i as f32) * 0.001).collect::<Vec<_>>(),
            Shape::from_dims(&[1, cfg.channels, t_in]),
            &Device::cpu(),
        );
        let latent = encoder.forward(&audio).unwrap();
        let dims = latent.shape();
        let dims = dims.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], cfg.dimension);
        assert_eq!(dims[2], t_in / total_stride);
        for &v in &latent.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn decoder_forward_shape_and_finite() {
        let cfg = tiny_cfg();
        let decoder = build_decoder(&cfg);
        let t_latent = 5;
        let latent = LazyTensor::from_f32(
            (0..(1 * cfg.dimension * t_latent)).map(|i| (i as f32) * 0.001).collect::<Vec<_>>(),
            Shape::from_dims(&[1, cfg.dimension, t_latent]),
            &Device::cpu(),
        );
        let audio = decoder.forward(&latent).unwrap();
        let dims = audio.shape();
        let dims = dims.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], cfg.channels);
        let total_stride: usize = cfg.ratios.iter().product();
        // Causal-trimmed upsample produces exactly t_latent · total_stride.
        assert_eq!(dims[2], t_latent * total_stride);
        for &v in &audio.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn encoder_decoder_round_trip_shape() {
        let cfg = tiny_cfg();
        let encoder = build_encoder(&cfg);
        let decoder = build_decoder(&cfg);
        let total_stride: usize = cfg.ratios.iter().product();
        let t_in = total_stride * 3;
        let audio = LazyTensor::from_f32(
            (0..(1 * cfg.channels * t_in)).map(|i| (i as f32) * 0.001).collect::<Vec<_>>(),
            Shape::from_dims(&[1, cfg.channels, t_in]),
            &Device::cpu(),
        );
        let latent = encoder.forward(&audio).unwrap();
        let recon = decoder.forward(&latent).unwrap();
        // Audio length preserved end-to-end (within the configured tolerance for
        // causal conv 1D — exact match here because kernel == 2·stride).
        assert_eq!(recon.shape().dims()[2], t_in);
    }

    #[test]
    fn preset_mimi_v0_1() {
        let p = SeaNetConfig::mimi_v0_1();
        assert_eq!(p.dimension, 512);
        assert_eq!(p.ratios, vec![8, 6, 5, 4]);
        let total: usize = p.ratios.iter().product();
        assert_eq!(total, 8 * 6 * 5 * 4); // 960× downsample
    }
}

//! EfficientNet (Tan & Le 2019, "EfficientNet: Rethinking
//! Model Scaling for Convolutional Neural Networks") ported
//! to the lazy-graph API. All eight depth variants B0–B7 are
//! reachable through width/depth multipliers on the seven-stage
//! MBConv schedule.
//!
//! # Architecture
//!
//! ```text
//! init_cna  : ConvNormSwish (3 → 32, k=3, stride=2)
//! stage_1   : MBConv × N₁  (expand=1, k=3, stride=1, 32 → 16)
//! stage_2   : MBConv × N₂  (expand=6, k=3, stride=2, 16 → 24)
//! stage_3   : MBConv × N₃  (expand=6, k=5, stride=2, 24 → 40)
//! stage_4   : MBConv × N₄  (expand=6, k=3, stride=2, 40 → 80)
//! stage_5   : MBConv × N₅  (expand=6, k=5, stride=1, 80 → 112)
//! stage_6   : MBConv × N₆  (expand=6, k=5, stride=2, 112 → 192)
//! stage_7   : MBConv × N₇  (expand=6, k=3, stride=1, 192 → 320)
//! final_cna : ConvNormSwish (320 → 1280, k=1, stride=1)
//! mean(H, W) → Linear(1280 → nclasses)
//! ```
//!
//! Width / depth multipliers:
//!
//! - B0: 1.0 / 1.0
//! - B1: 1.0 / 1.1
//! - B2: 1.1 / 1.2
//! - B3: 1.2 / 1.4
//! - B4: 1.4 / 1.8
//! - B5: 1.6 / 2.2
//! - B6: 1.8 / 2.6
//! - B7: 2.0 / 3.1
//!
//! Channel counts are width-multiplied then rounded to the
//! nearest multiple of 8 (the `make_divisible` helper). Block
//! counts are depth-multiplied then ceil()'d.
//!
//! # MBConv (mobile inverted bottleneck)
//!
//! ```text
//!   x → [expand] 1×1 conv → BN → Swish        # if exp_ratio > 1
//!     → depthwise k×k conv → BN → Swish      # groups = exp_channels
//!     → squeeze_excite                       # see below
//!     → project 1×1 conv → BN                # NO activation
//!     → (+ x)                                # iff stride==1 && c_in==c_out
//! ```
//!
//! When `exp_ratio == 1`, the expand 1×1 is omitted and the
//! depthwise conv runs on `c_in` channels directly.
//!
//! # Squeeze-and-excitation
//!
//! ```text
//!   gate = mean_keepdim(x, (H, W))         # (N, C, 1, 1)
//!        → 1×1 conv → Swish → 1×1 conv → sigmoid
//!   y    = x * gate                         # broadcast on (H, W)
//! ```
//!
//! Squeeze channels = `max(1, c_in / 4)`. The two 1×1 convs
//! are bias-enabled.
//!
//! # Asymmetric "same" padding
//!
//! Eager `Conv2DSame` computes
//! `pad_total = max(0, (oh - 1) * s + k - ih)` and splits as
//! `pad_total / 2` on the left and `pad_total - pad_total / 2`
//! on the right (asymmetric when total is odd). For stride=2
//! with even input sizes (32, 64, 224, …) and odd kernel
//! (3, 5), padding is asymmetric: `(0, 1)` on H and W. The
//! lazy `conv2d` op only supports symmetric padding, so the
//! port applies `pad_with_zeros` on dims 2/3 first and then
//! calls `conv2d` with padding `(0, 0)`.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. Returns class logits
//! `(1, nclasses)`. Input is `(1, 3, H, W)`; H and W are
//! free but typically 224 (B0) up to 600 (B7).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_convmixer::BatchNormParams;
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MBConvStageConfig {
    pub expand_ratio: f64,
    pub kernel: usize,
    pub stride: usize,
    pub input_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
}

#[derive(Debug, Clone)]
pub struct EfficientNetConfig {
    /// Seven stage configs (after width/depth-multiplier expansion).
    pub stages: Vec<MBConvStageConfig>,
    pub nclasses: usize,
    pub bn_eps: f64,
    /// `4 * stages.last().out_channels` (1280 for B0).
    pub final_channels: usize,
}

fn make_divisible(v: f64, divisor: usize) -> usize {
    let min_value = divisor;
    let new_v = usize::max(
        min_value,
        (v + divisor as f64 * 0.5) as usize / divisor * divisor,
    );
    if (new_v as f64) < 0.9 * v {
        new_v + divisor
    } else {
        new_v
    }
}

fn bneck_confs(width_mult: f64, depth_mult: f64) -> Vec<MBConvStageConfig> {
    let conf = |e, k, s, i, o, n| MBConvStageConfig {
        expand_ratio: e,
        kernel: k,
        stride: s,
        input_channels: make_divisible(i as f64 * width_mult, 8),
        out_channels: make_divisible(o as f64 * width_mult, 8),
        num_layers: (n as f64 * depth_mult).ceil() as usize,
    };
    vec![
        conf(1.0, 3, 1, 32,  16,  1),
        conf(6.0, 3, 2, 16,  24,  2),
        conf(6.0, 5, 2, 24,  40,  2),
        conf(6.0, 3, 2, 40,  80,  3),
        conf(6.0, 5, 1, 80,  112, 3),
        conf(6.0, 5, 2, 112, 192, 4),
        conf(6.0, 3, 1, 192, 320, 1),
    ]
}

impl EfficientNetConfig {
    fn from_mults(nclasses: usize, w: f64, d: f64) -> Self {
        let stages = bneck_confs(w, d);
        let last_out = stages.last().expect("at least one stage").out_channels;
        Self { stages, nclasses, bn_eps: 1e-3, final_channels: 4 * last_out }
    }
    pub fn b0(nclasses: usize) -> Self { Self::from_mults(nclasses, 1.0, 1.0) }
    pub fn b1(nclasses: usize) -> Self { Self::from_mults(nclasses, 1.0, 1.1) }
    pub fn b2(nclasses: usize) -> Self { Self::from_mults(nclasses, 1.1, 1.2) }
    pub fn b3(nclasses: usize) -> Self { Self::from_mults(nclasses, 1.2, 1.4) }
    pub fn b4(nclasses: usize) -> Self { Self::from_mults(nclasses, 1.4, 1.8) }
    pub fn b5(nclasses: usize) -> Self { Self::from_mults(nclasses, 1.6, 2.2) }
    pub fn b6(nclasses: usize) -> Self { Self::from_mults(nclasses, 1.8, 2.6) }
    pub fn b7(nclasses: usize) -> Self { Self::from_mults(nclasses, 2.0, 3.1) }
}

#[derive(Debug, Clone)]
pub struct ConvBN {
    /// `[c_out, c_in / groups, k, k]`.
    pub w: WeightStorage,
    pub bn: BatchNormParams,
    pub c_in: usize,
    pub c_out: usize,
    pub kernel: usize,
    pub stride: usize,
    pub groups: usize,
}

#[derive(Debug, Clone)]
pub struct ConvBias {
    /// `[c_out, c_in, 1, 1]`.
    pub w: WeightStorage,
    pub b: Arc<[f32]>,
    pub c_in: usize,
    pub c_out: usize,
}

#[derive(Debug, Clone)]
pub struct SqueezeExciteWeights {
    pub fc1: ConvBias,
    pub fc2: ConvBias,
}

#[derive(Debug, Clone)]
pub struct MBConvWeights {
    /// `None` when `expand_ratio == 1` and no expand step happens.
    pub expand: Option<ConvBN>,
    pub depthwise: ConvBN,
    pub se: SqueezeExciteWeights,
    /// Project conv has NO activation.
    pub project: ConvBN,
    /// Captured at weight-build time.
    pub config: MBConvStageConfig,
}

#[derive(Debug, Clone)]
pub struct EfficientNetWeights {
    pub init_cna: ConvBN,
    pub blocks: Vec<MBConvWeights>,
    pub final_cna: ConvBN,
    pub classifier_w: WeightStorage,
    pub classifier_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct EfficientNetModel {
    pub config: EfficientNetConfig,
    pub weights: EfficientNetWeights,
}

impl EfficientNetModel {
    /// Run a forward pass on `image` of shape `(1, 3, H, W)`.
    pub fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = image.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4, "image must be rank 4 [N, 3, H, W]");
        assert_eq!(dims[1], 3, "image must have 3 input channels");

        let mut x = self.apply_conv_bn(image, &self.weights.init_cna)?;
        x = swish(&x)?;
        for block in &self.weights.blocks {
            x = self.apply_mbconv(&x, block)?;
        }
        x = self.apply_conv_bn(&x, &self.weights.final_cna)?;
        x = swish(&x)?;

        // Global average pool over (H, W).
        let pooled_w = x.mean_dim(3_usize)?;     // (1, C, H)
        let pooled = pooled_w.mean_dim(2_usize)?; // (1, C)

        let logits = self.weights.classifier_w.apply_linear(
            &pooled, cfg.final_channels, cfg.nclasses,
        );
        let bias_t = pooled.const_f32_like(
            Arc::clone(&self.weights.classifier_b),
            Shape::from_dims(&[cfg.nclasses]),
        );
        logits.broadcast_add(&bias_t)
    }

    fn apply_conv_bn(&self, x: &LazyTensor, cb: &ConvBN) -> Result<LazyTensor> {
        let x_padded = pad_same(x, cb.kernel, cb.stride)?;
        let w = cb.w.const_like(
            x, Shape::from_dims(&[cb.c_out, cb.c_in / cb.groups, cb.kernel, cb.kernel]),
        );
        let conv = x_padded.conv2d(&w, None, (cb.stride, cb.stride), (0, 0), cb.groups)?;
        apply_bn(&conv, &cb.bn, cb.c_out)
    }

    fn apply_conv_bias(&self, x: &LazyTensor, cb: &ConvBias) -> Result<LazyTensor> {
        // 1×1 conv → broadcast-add bias on channel dim.
        let w = cb.w.const_like(
            x, Shape::from_dims(&[cb.c_out, cb.c_in, 1, 1]),
        );
        let conv = x.conv2d(&w, None, (1, 1), (0, 0), 1)?;
        let b_t = x
            .const_f32_like(Arc::clone(&cb.b), Shape::from_dims(&[cb.c_out]))
            .reshape(Shape::from_dims(&[1, cb.c_out, 1, 1]))?;
        conv.broadcast_add(&b_t)
    }

    fn apply_mbconv(&self, x: &LazyTensor, block: &MBConvWeights) -> Result<LazyTensor> {
        let cfg = &block.config;
        let use_residual = cfg.stride == 1 && cfg.input_channels == cfg.out_channels;
        let mut y = x.clone();
        if let Some(exp) = &block.expand {
            y = self.apply_conv_bn(&y, exp)?;
            y = swish(&y)?;
        }
        y = self.apply_conv_bn(&y, &block.depthwise)?;
        y = swish(&y)?;
        y = self.apply_se(&y, &block.se)?;
        y = self.apply_conv_bn(&y, &block.project)?;
        if use_residual {
            x.add(&y)
        } else {
            Ok(y)
        }
    }

    fn apply_se(&self, x: &LazyTensor, se: &SqueezeExciteWeights) -> Result<LazyTensor> {
        let pooled = x.mean_keepdim(2_usize)?.mean_keepdim(3_usize)?; // (N, C, 1, 1)
        let g = self.apply_conv_bias(&pooled, &se.fc1)?;
        let g = swish(&g)?;
        let g = self.apply_conv_bias(&g, &se.fc2)?;
        let g = g.sigmoid();
        x.broadcast_mul(&g)
    }
}

/// Asymmetric "same" padding identical to eager `Conv2DSame`:
/// `pad_total = max(0, (oh - 1) * s + k - ih)`, split as
/// `(pad_total / 2, pad_total - pad_total / 2)`. Computed
/// independently on dims 2 and 3.
fn pad_same(x: &LazyTensor, k: usize, s: usize) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims().to_vec();
    let (h, w) = (dims[2], dims[3]);
    let oh = h.div_ceil(s);
    let ow = w.div_ceil(s);
    let pad_h = (oh.saturating_sub(1) * s + k).saturating_sub(h);
    let pad_w = (ow.saturating_sub(1) * s + k).saturating_sub(w);
    let mut y = x.clone();
    if pad_h > 0 {
        y = y.pad_with_zeros(2_usize, pad_h / 2, pad_h - pad_h / 2)?;
    }
    if pad_w > 0 {
        y = y.pad_with_zeros(3_usize, pad_w / 2, pad_w - pad_w / 2)?;
    }
    Ok(y)
}

/// Apply fused-affine BN to a 4-D NCHW tensor.
fn apply_bn(x: &LazyTensor, bn: &BatchNormParams, channels: usize) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    assert_eq!(dims.len(), 4, "BN input must be rank 4");
    assert_eq!(dims[1], channels);
    let w_t = x
        .const_f32_like(Arc::clone(&bn.w), Shape::from_dims(&[channels]))
        .reshape(Shape::from_dims(&[1, channels, 1, 1]))?;
    let b_t = x
        .const_f32_like(Arc::clone(&bn.b), Shape::from_dims(&[channels]))
        .reshape(Shape::from_dims(&[1, channels, 1, 1]))?;
    x.broadcast_mul(&w_t)?.broadcast_add(&b_t)
}

/// Swish / SiLU: `x * sigmoid(x)`. EfficientNet's standard activation.
fn swish(x: &LazyTensor) -> Result<LazyTensor> {
    Ok(x.silu())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }

    fn tiny_bn(channels: usize, nb: &mut dyn FnMut() -> f32) -> BatchNormParams {
        let gain: Vec<f32> = (0..channels).map(|_| 1.0 + nb() * 0.1).collect();
        let bias: Vec<f32> = (0..channels).map(|_| nb() * 0.1).collect();
        let mean: Vec<f32> = (0..channels).map(|_| nb() * 0.05).collect();
        let var: Vec<f32> = (0..channels).map(|_| 1.0 + nb().abs() * 0.05).collect();
        BatchNormParams::from_raw(&gain, &bias, &mean, &var, 1e-3)
    }

    fn build_conv_bn(
        c_in: usize, c_out: usize, kernel: usize, stride: usize, groups: usize,
        nb: &mut dyn FnMut() -> f32,
    ) -> ConvBN {
        let w_len = c_out * (c_in / groups) * kernel * kernel;
        ConvBN {
            w: WeightStorage::F32(vec_of(w_len, nb)),
            bn: tiny_bn(c_out, nb),
            c_in, c_out, kernel, stride, groups,
        }
    }

    fn build_conv_bias(c_in: usize, c_out: usize, nb: &mut dyn FnMut() -> f32) -> ConvBias {
        ConvBias {
            w: WeightStorage::F32(vec_of(c_out * c_in, nb)),
            b: vec_of(c_out, nb),
            c_in, c_out,
        }
    }

    fn build_mbconv(stage: &MBConvStageConfig, c_in_override: Option<usize>, nb: &mut dyn FnMut() -> f32)
        -> MBConvWeights
    {
        let mut cfg = *stage;
        if let Some(c) = c_in_override { cfg.input_channels = c; cfg.stride = 1; }
        let exp = make_divisible(cfg.input_channels as f64 * cfg.expand_ratio, 8);
        let expand = if exp != cfg.input_channels {
            Some(build_conv_bn(cfg.input_channels, exp, 1, 1, 1, nb))
        } else {
            None
        };
        let depthwise = build_conv_bn(exp, exp, cfg.kernel, cfg.stride, exp, nb);
        let squeeze = usize::max(1, cfg.input_channels / 4);
        let se = SqueezeExciteWeights {
            fc1: build_conv_bias(exp, squeeze, nb),
            fc2: build_conv_bias(squeeze, exp, nb),
        };
        let project = build_conv_bn(exp, cfg.out_channels, 1, 1, 1, nb);
        MBConvWeights { expand, depthwise, se, project, config: cfg }
    }

    fn build_weights(cfg: &EfficientNetConfig, seed: u32) -> EfficientNetWeights {
        let mut nb = rng_seed(seed);
        let first_in = cfg.stages[0].input_channels;
        let init_cna = build_conv_bn(3, first_in, 3, 2, 1, &mut nb);
        let mut blocks = Vec::new();
        for stage in &cfg.stages {
            for r in 0..stage.num_layers {
                let c_in_override = if r == 0 { None } else { Some(stage.out_channels) };
                blocks.push(build_mbconv(stage, c_in_override, &mut nb));
            }
        }
        let last_out = cfg.stages.last().unwrap().out_channels;
        let final_cna = build_conv_bn(last_out, cfg.final_channels, 1, 1, 1, &mut nb);
        let classifier_w = WeightStorage::F32(vec_of(cfg.final_channels * cfg.nclasses, &mut nb));
        let classifier_b = vec_of(cfg.nclasses, &mut nb);
        EfficientNetWeights {
            init_cna, blocks, final_cna, classifier_w, classifier_b,
        }
    }

    fn tiny_image(h: usize) -> LazyTensor {
        let mut nb = rng_seed(1234);
        let data: Arc<[f32]> = Arc::from(
            (0..3 * h * h).map(|_| nb()).collect::<Vec<_>>()
        );
        LazyTensor::from_f32(data, Shape::from_dims(&[1, 3, h, h]), &Device::cpu())
    }

    /// pad_same with stride=2, k=3, even input → asymmetric (0, 1).
    /// stride=2, k=3, odd input → (1, 1) symmetric. stride=1, k=3,
    /// even input → (1, 1).
    #[test]
    fn pad_same_formula() {
        // (h, k, s) → expected (left, right)
        let cases: [(usize, usize, usize, (usize, usize)); 5] = [
            (32, 3, 2, (0, 1)),
            (33, 3, 2, (1, 1)),
            (32, 3, 1, (1, 1)),
            (32, 5, 1, (2, 2)),
            (32, 5, 2, (1, 2)),
        ];
        for (h, k, s, expected) in cases {
            // pad_total = max(0, ceil(h/s - 1) * s + k - h)
            let oh = h.div_ceil(s);
            let pad_total = (oh.saturating_sub(1) * s + k).saturating_sub(h);
            let left = pad_total / 2;
            let right = pad_total - left;
            assert_eq!((left, right), expected,
                "h={h}, k={k}, s={s}: expected {expected:?}, got ({left}, {right})");
        }
    }

    /// EfficientNet-B0 with shrunk channels — verify the full
    /// forward chain runs and returns finite logits. Real B0
    /// channels (32→16→24→40→80→112→192→320 → 1280) keep test
    /// fast at 32×32 input.
    #[test]
    fn efficientnet_b0_forward_shape() {
        let cfg = EfficientNetConfig::b0(10);
        let weights = build_weights(&cfg, 999);
        let model = EfficientNetModel { config: cfg, weights };
        let img = tiny_image(32);
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 10]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    /// Final channel count = 4 × last-stage out_channels (B0:
    /// 4 * 320 = 1280, the classic EfficientNet feature dim).
    #[test]
    fn b0_final_channels_is_1280() {
        let cfg = EfficientNetConfig::b0(10);
        assert_eq!(cfg.final_channels, 1280);
    }

    /// Squeeze channels = max(1, c_in / 4). For B0's first
    /// stage (c_in = 32), squeeze = 8. Verify with a small
    /// SqueezeExcite forward.
    #[test]
    fn squeeze_excite_squeezes_channels() {
        let mut nb = rng_seed(55);
        let se = SqueezeExciteWeights {
            fc1: build_conv_bias(32, 8, &mut nb),
            fc2: build_conv_bias(8, 32, &mut nb),
        };
        let img = LazyTensor::from_f32(
            Arc::from((0..1 * 32 * 4 * 4).map(|i| (i as f32) * 0.001).collect::<Vec<_>>()),
            Shape::from_dims(&[1, 32, 4, 4]),
            &Device::cpu(),
        );
        let cfg = EfficientNetConfig::b0(1);
        let weights = build_weights(&cfg, 1);
        let model = EfficientNetModel { config: cfg, weights };
        let out = model.apply_se(&img, &se).unwrap();
        assert_eq!(out.shape().dims(), &[1, 32, 4, 4]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite SE output: {v}");
        }
    }
}

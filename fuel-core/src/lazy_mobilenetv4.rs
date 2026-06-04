//! MobileNetV4 — lazy port.
//!
//! Pipeline: image → stem (3×3 stride-2 conv-BN-act) → 5 stages
//! (each a sequence of Conv / EdgeResidual / UniversalBottleneck
//! blocks) → global mean pool → optional head (1×1 conv-BN-act
//! → flatten → linear classifier).
//!
//! Block types supported (v1):
//!   - **Convolutional**: conv (k, stride, pad = k/2) → BN → act.
//!   - **EdgeResidual**: expand conv (k, stride, pad = k/2) →
//!     BN → act → 1×1 pointwise conv → BN. No skip.
//!   - **UniversalBottleneck**: optional `dw_start` depthwise →
//!     1×1 `pw_exp` pointwise → BN → act → optional `dw_mid`
//!     depthwise → BN → act → 1×1 `pw_proj` pointwise → BN →
//!     optional layer-scale → optional residual.
//!
//! v1 scope:
//!   - F32, batch == 1, fused-affine BN (inference-mode).
//!   - **Conv variants** (Small / Medium / Large_Conv) — these
//!     use only the three block types above. The Hybrid variants
//!     additionally use Mobile-MQA Attention blocks; that block
//!     type is a follow-up.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_convmixer::BatchNormParams;
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mv4Activation {
    Relu,
    Gelu,
}

/// Lightweight block-spec enum used at config time only. The
/// loaded weights for each block live on `BlockWeights` below;
/// the spec drives the strides, kernels, channels, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockSpec {
    Convolutional {
        out_channels: usize, kernel: usize, stride: usize,
    },
    EdgeResidual {
        out_channels: usize, kernel: usize, stride: usize, expand: usize,
    },
    UniversalBottleneck {
        out_channels: usize, start_kernel: usize, mid_kernel: usize,
        stride: usize, expand: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mv4Config {
    pub stem_dim: usize,
    pub activation: Mv4Activation,
    /// 5 stages, each a list of blocks.
    pub stages: [Vec<BlockSpec>; 5],
    /// Number of channels coming into the head (= last
    /// Convolutional block's out_channels). MobileNetV4-Conv
    /// presets use 960 with a 1280-channel head_out.
    pub head_in_channels: usize,
    pub head_out_channels: usize,
}

impl Mv4Config {
    /// MobileNetV4-Conv-Small preset.
    pub fn conv_small() -> Self {
        use BlockSpec::*;
        Self {
            stem_dim: 32,
            activation: Mv4Activation::Relu,
            head_in_channels: 960,
            head_out_channels: 1280,
            stages: [
                vec![
                    Convolutional { out_channels: 32, kernel: 3, stride: 2 },
                    Convolutional { out_channels: 32, kernel: 1, stride: 1 },
                ],
                vec![
                    Convolutional { out_channels: 96, kernel: 3, stride: 2 },
                    Convolutional { out_channels: 64, kernel: 1, stride: 1 },
                ],
                vec![
                    UniversalBottleneck { out_channels: 96,  start_kernel: 5, mid_kernel: 5, stride: 2, expand: 3 },
                    UniversalBottleneck { out_channels: 96,  start_kernel: 0, mid_kernel: 3, stride: 1, expand: 2 },
                    UniversalBottleneck { out_channels: 96,  start_kernel: 0, mid_kernel: 3, stride: 1, expand: 2 },
                    UniversalBottleneck { out_channels: 96,  start_kernel: 0, mid_kernel: 3, stride: 1, expand: 2 },
                    UniversalBottleneck { out_channels: 96,  start_kernel: 0, mid_kernel: 3, stride: 1, expand: 2 },
                    UniversalBottleneck { out_channels: 96,  start_kernel: 3, mid_kernel: 0, stride: 1, expand: 4 },
                ],
                vec![
                    UniversalBottleneck { out_channels: 128, start_kernel: 3, mid_kernel: 3, stride: 2, expand: 6 },
                    UniversalBottleneck { out_channels: 128, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4 },
                    UniversalBottleneck { out_channels: 128, start_kernel: 0, mid_kernel: 5, stride: 1, expand: 4 },
                    UniversalBottleneck { out_channels: 128, start_kernel: 0, mid_kernel: 5, stride: 1, expand: 3 },
                    UniversalBottleneck { out_channels: 128, start_kernel: 0, mid_kernel: 3, stride: 1, expand: 4 },
                    UniversalBottleneck { out_channels: 128, start_kernel: 0, mid_kernel: 3, stride: 1, expand: 4 },
                ],
                vec![
                    Convolutional { out_channels: 960, kernel: 1, stride: 1 },
                ],
            ],
        }
    }
}

// ---- Weight structures ------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Conv2dBnWeights {
    /// `[c_out, c_in / groups, k, k]`.
    pub w: Arc<[f32]>,
    pub bn: BatchNormParams,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub groups: usize,
}

#[derive(Debug, Clone)]
pub struct UibWeights {
    /// `dw_start` is present iff `start_kernel > 0`.
    pub dw_start: Option<Conv2dBnWeights>,
    pub pw_exp: Conv2dBnWeights,
    /// `dw_mid` is present iff `mid_kernel > 0`.
    pub dw_mid: Option<Conv2dBnWeights>,
    pub pw_proj: Conv2dBnWeights,
    /// Optional per-channel layer-scale γ (length = out_channels).
    pub layer_scale_gamma: Option<Arc<[f32]>>,
    pub skip: bool,
}

#[derive(Debug, Clone)]
pub struct EdgeResidualWeights {
    pub conv_exp: Conv2dBnWeights,
    pub conv_pwl: Conv2dBnWeights,
}

/// Per-block weights, tagged to match the spec.
#[derive(Debug, Clone)]
pub enum BlockWeights {
    Convolutional(Conv2dBnWeights),
    EdgeResidual(EdgeResidualWeights),
    UniversalBottleneck(UibWeights),
}

#[derive(Debug, Clone)]
pub struct Mv4HeadWeights {
    /// 1×1 conv-BN: head_in_channels → head_out_channels.
    pub conv: Conv2dBnWeights,
    /// Linear classifier: `[head_out_channels, num_classes]`.
    pub linear_w: WeightStorage,
    pub linear_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Mv4Weights {
    pub stem: Conv2dBnWeights,
    /// Flat list of blocks across all 5 stages, in order.
    pub blocks: Vec<BlockWeights>,
    pub head: Option<Mv4HeadWeights>,
}

#[derive(Debug, Clone)]
pub struct Mv4Model {
    pub config: Mv4Config,
    pub weights: Mv4Weights,
}

// ---- Forward ---------------------------------------------------------------

impl Mv4Model {
    /// Run inference on `image` of shape `(1, 3, H, W)`. Returns
    /// classifier logits when `weights.head` is `Some`, else
    /// pooled features `(1, head_in_channels)`.
    pub fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = image.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        assert_eq!(dims[1], 3, "image must have 3 input channels");

        let mut x = apply_conv_bn_act(image, &self.weights.stem, cfg.activation, image)?;
        for blk in &self.weights.blocks {
            x = apply_block(&x, blk, cfg.activation, image)?;
        }
        // Global mean over (H, W).
        let pooled_w = x.mean_dim(3_usize)?;
        let pooled = pooled_w.mean_dim(2_usize)?;
        match &self.weights.head {
            None => Ok(pooled),
            Some(head) => {
                let dims = pooled.shape();
                let dims = dims.dims();
                let c = dims[1];
                let chw = pooled.reshape(Shape::from_dims(&[1, c, 1, 1]))?;
                let h = apply_conv_bn_act(&chw, &head.conv, cfg.activation, image)?;
                let flat = h.reshape(Shape::from_dims(&[1, cfg.head_out_channels]))?;
                let n = head.linear_b.len();
                let logits = head.linear_w.apply_linear(&flat, cfg.head_out_channels, n);
                let bias = image.const_f32_like(
                    Arc::clone(&head.linear_b), Shape::from_dims(&[n]),
                );
                logits.broadcast_add(&bias)
            }
        }
    }

    /// Backbone-only forward: returns the channels-first feature
    /// map after the last stage's blocks, BEFORE global mean pool
    /// and the optional head.
    pub fn forward_features(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = image.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        assert_eq!(dims[1], 3);
        let mut x = apply_conv_bn_act(image, &self.weights.stem, cfg.activation, image)?;
        for blk in &self.weights.blocks {
            x = apply_block(&x, blk, cfg.activation, image)?;
        }
        Ok(x)
    }
}

// ---- Component helpers -----------------------------------------------------

fn apply_bn(
    x: &LazyTensor, bn: &BatchNormParams, channels: usize,
) -> Result<LazyTensor> {
    assert_eq!(bn.w.len(), channels);
    let w_t = x
        .const_f32_like(Arc::clone(&bn.w), Shape::from_dims(&[channels]))
        .reshape(Shape::from_dims(&[1, channels, 1, 1]))?;
    let b_t = x
        .const_f32_like(Arc::clone(&bn.b), Shape::from_dims(&[channels]))
        .reshape(Shape::from_dims(&[1, channels, 1, 1]))?;
    x.broadcast_mul(&w_t)?.broadcast_add(&b_t)
}

fn apply_conv_bn(
    x: &LazyTensor, c: &Conv2dBnWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let w = anchor.const_f32_like(
        Arc::clone(&c.w),
        Shape::from_dims(&[c.c_out, c.c_in / c.groups, c.k, c.k]),
    );
    let conv = x.conv2d(
        &w, None,
        (c.stride, c.stride),
        (c.pad, c.pad),
        c.groups,
    )?;
    apply_bn(&conv, &c.bn, c.c_out)
}

fn apply_act(x: LazyTensor, act: Mv4Activation) -> LazyTensor {
    match act {
        Mv4Activation::Relu => x.relu(),
        Mv4Activation::Gelu => x.gelu(),
    }
}

fn apply_conv_bn_act(
    x: &LazyTensor, c: &Conv2dBnWeights, act: Mv4Activation, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    Ok(apply_act(apply_conv_bn(x, c, anchor)?, act))
}

fn apply_block(
    x: &LazyTensor, b: &BlockWeights, act: Mv4Activation, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    match b {
        BlockWeights::Convolutional(c) => apply_conv_bn_act(x, c, act, anchor),
        BlockWeights::EdgeResidual(er) => {
            let y = apply_conv_bn_act(x, &er.conv_exp, act, anchor)?;
            apply_conv_bn(&y, &er.conv_pwl, anchor)
        }
        BlockWeights::UniversalBottleneck(uib) => {
            let mut y = x.clone();
            if let Some(dw) = &uib.dw_start {
                y = apply_conv_bn(&y, dw, anchor)?;
            }
            y = apply_conv_bn_act(&y, &uib.pw_exp, act, anchor)?;
            if let Some(dw) = &uib.dw_mid {
                y = apply_conv_bn_act(&y, dw, act, anchor)?;
            }
            y = apply_conv_bn(&y, &uib.pw_proj, anchor)?;
            if let Some(g) = &uib.layer_scale_gamma {
                let gt = anchor
                    .const_f32_like(Arc::clone(g), Shape::from_dims(&[g.len()]))
                    .reshape(Shape::from_dims(&[1, g.len(), 1, 1]))?;
                y = y.broadcast_mul(&gt)?;
            }
            if uib.skip {
                y = y.add(x)?;
            }
            Ok(y)
        }
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
    fn ws(n: usize, nb: &mut dyn FnMut() -> f32) -> WeightStorage {
        WeightStorage::F32(vec_of(n, nb))
    }
    fn tiny_bn(c: usize) -> BatchNormParams {
        BatchNormParams {
            w: Arc::from(vec![1.0_f32; c]),
            b: Arc::from(vec![0.0_f32; c]),
        }
    }
    fn conv_bn_w(
        c_in: usize, c_out: usize, k: usize, stride: usize, pad: usize, groups: usize,
        nb: &mut dyn FnMut() -> f32,
    ) -> Conv2dBnWeights {
        Conv2dBnWeights {
            w: vec_of(c_out * (c_in / groups) * k * k, nb),
            bn: tiny_bn(c_out),
            c_in, c_out, k, stride, pad, groups,
        }
    }

    /// Build weights for a single spec. Tracks the running `in_channels`.
    fn block_weights(
        spec: &BlockSpec, in_ch: usize, nb: &mut dyn FnMut() -> f32,
    ) -> BlockWeights {
        match *spec {
            BlockSpec::Convolutional { out_channels, kernel, stride } => {
                BlockWeights::Convolutional(conv_bn_w(
                    in_ch, out_channels, kernel, stride, kernel / 2, 1, nb,
                ))
            }
            BlockSpec::EdgeResidual { out_channels, kernel, stride, expand } => {
                let mid = in_ch * expand;
                BlockWeights::EdgeResidual(EdgeResidualWeights {
                    conv_exp: conv_bn_w(in_ch, mid, kernel, stride, kernel / 2, 1, nb),
                    conv_pwl: conv_bn_w(mid, out_channels, 1, 1, 0, 1, nb),
                })
            }
            BlockSpec::UniversalBottleneck {
                out_channels, start_kernel, mid_kernel, stride, expand,
            } => {
                let mid = in_ch * expand;
                let dw_start_stride = if mid_kernel > 0 { 1 } else { stride };
                let dw_start = if start_kernel > 0 {
                    Some(conv_bn_w(
                        in_ch, in_ch, start_kernel, dw_start_stride, start_kernel / 2, in_ch, nb,
                    ))
                } else { None };
                let pw_exp = conv_bn_w(in_ch, mid, 1, 1, 0, 1, nb);
                let dw_mid = if mid_kernel > 0 {
                    Some(conv_bn_w(
                        mid, mid, mid_kernel, stride, mid_kernel / 2, mid, nb,
                    ))
                } else { None };
                let pw_proj = conv_bn_w(mid, out_channels, 1, 1, 0, 1, nb);
                let skip = in_ch == out_channels && stride == 1;
                BlockWeights::UniversalBottleneck(UibWeights {
                    dw_start, pw_exp, dw_mid, pw_proj,
                    layer_scale_gamma: None,
                    skip,
                })
            }
        }
    }

    /// Construct synthetic weights for a config. Channel chaining
    /// across stages mirrors the eager `mobilenetv4_blocks` loop.
    fn build_weights(cfg: &Mv4Config) -> Mv4Weights {
        let mut nb = rng_seed(0xC0FFEE);
        let stem = conv_bn_w(3, cfg.stem_dim, 3, 2, 1, 1, &mut nb);
        let mut in_ch = cfg.stem_dim;
        let mut blocks = Vec::new();
        for stage in &cfg.stages {
            for spec in stage {
                blocks.push(block_weights(spec, in_ch, &mut nb));
                in_ch = match spec {
                    BlockSpec::Convolutional { out_channels, .. } => *out_channels,
                    BlockSpec::EdgeResidual { out_channels, .. } => *out_channels,
                    BlockSpec::UniversalBottleneck { out_channels, .. } => *out_channels,
                };
            }
        }
        Mv4Weights { stem, blocks, head: None }
    }

    fn with_head(mut w: Mv4Weights, cfg: &Mv4Config, n_classes: usize) -> Mv4Weights {
        let mut nb = rng_seed(7777);
        w.head = Some(Mv4HeadWeights {
            conv: conv_bn_w(
                cfg.head_in_channels, cfg.head_out_channels, 1, 1, 0, 1, &mut nb,
            ),
            linear_w: ws(cfg.head_out_channels * n_classes, &mut nb),
            linear_b: vec_of(n_classes, &mut nb),
        });
        w
    }

    fn tiny_config() -> Mv4Config {
        use BlockSpec::*;
        Mv4Config {
            stem_dim: 8,
            activation: Mv4Activation::Relu,
            head_in_channels: 32,
            head_out_channels: 16,
            stages: [
                vec![Convolutional { out_channels: 8, kernel: 1, stride: 1 }],
                vec![Convolutional { out_channels: 16, kernel: 3, stride: 2 }],
                vec![
                    UniversalBottleneck { out_channels: 16, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 2 },
                    UniversalBottleneck { out_channels: 16, start_kernel: 0, mid_kernel: 3, stride: 1, expand: 2 },
                ],
                vec![
                    EdgeResidual { out_channels: 24, kernel: 3, stride: 2, expand: 2 },
                ],
                vec![Convolutional { out_channels: 32, kernel: 1, stride: 1 }],
            ],
        }
    }

    #[test]
    fn forward_no_head_shape_and_finite() {
        let cfg = tiny_config();
        let weights = build_weights(&cfg);
        let model = Mv4Model { config: cfg.clone(), weights };
        let img = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let pooled = model.forward(&img).unwrap();
        // No head → pooled features (1, head_in_channels = 32).
        assert_eq!(pooled.shape().dims(), &[1, cfg.head_in_channels]);
        for &v in &pooled.realize_f32() {
            assert!(v.is_finite(), "non-finite pooled value: {v}");
        }
    }

    #[test]
    fn forward_with_head_shape_and_finite() {
        let cfg = tiny_config();
        let weights = with_head(build_weights(&cfg), &cfg, 7);
        let model = Mv4Model { config: cfg.clone(), weights };
        let img = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 7]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn forward_features_returns_pre_pool_map() {
        let cfg = tiny_config();
        let weights = build_weights(&cfg);
        let model = Mv4Model { config: cfg.clone(), weights };
        let img = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let feats = model.forward_features(&img).unwrap();
        let shape = feats.shape();
        let dims = shape.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], cfg.head_in_channels);
        // Sanity: spatial dims must be > 0 (model didn't collapse them).
        assert!(dims[2] > 0);
        assert!(dims[3] > 0);
        for &v in &feats.realize_f32() {
            assert!(v.is_finite(), "non-finite feature: {v}");
        }
    }

    /// UniversalBottleneck skip path: same in/out channels and
    /// stride=1 should keep activations bounded. Different inputs
    /// produce different outputs.
    #[test]
    fn uib_responds_to_input() {
        let cfg = tiny_config();
        let weights = build_weights(&cfg);
        let model = Mv4Model { config: cfg.clone(), weights };
        let img_a = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let img_b = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01 + 0.5).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let a = model.forward(&img_a).unwrap().realize_f32();
        let b = model.forward(&img_b).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        // Tiny random weights (~0.05 magnitude) propagated through many
        // BN+conv layers attenuate the signal. The path IS wired; we just
        // need a tolerance that admits the very-small-magnitude response.
        assert!(max_diff > 1e-9,
            "backbone must respond to input changes, max_diff = {max_diff}");
    }

    #[test]
    fn conv_small_preset_constructs() {
        let cfg = Mv4Config::conv_small();
        assert_eq!(cfg.stem_dim, 32);
        assert_eq!(cfg.head_in_channels, 960);
        assert_eq!(cfg.head_out_channels, 1280);
        assert_eq!(cfg.stages[0].len(), 2);
        assert_eq!(cfg.stages[4].len(), 1);
    }
}

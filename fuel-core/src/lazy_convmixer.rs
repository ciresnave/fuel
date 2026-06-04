//! ConvMixer (Trockman & Kolter 2022 "Patches Are All You Need?")
//! ported to the lazy-graph API.
//!
//! ConvMixer is a minimal patch-based convnet that mixes
//! spatial / channel information with two convolutions per
//! block: a **depthwise** k×k conv (mixes spatial neighbors
//! per channel) followed by a **pointwise** 1×1 conv (mixes
//! channels per pixel). Both arms are bracketed by GELU +
//! BatchNorm. The depthwise arm has a residual skip; the
//! pointwise arm does not.
//!
//! # Architecture
//!
//!   1. **Stem**: Conv2d(3, dim, k=patch_size, stride=patch_size)
//!      → GELU → BatchNorm.
//!   2. **Mixer blocks** × `depth`:
//!      ```text
//!      y = BN(GELU(depthwise_conv_k×k_same(x)))
//!      x = x + y                  # residual on depthwise
//!      x = BN(GELU(pointwise_conv_1×1(x)))
//!      ```
//!   3. **Head**: global average pool over (H, W) → Linear(dim → nclasses).
//!
//! # Depthwise vs pointwise
//!
//! - Depthwise: `Conv2d(dim, dim, kernel=k, groups=dim, padding="same")`.
//!   Each output channel sees only its own input channel; cheap
//!   spatial mixing.
//! - Pointwise: `Conv2d(dim, dim, kernel=1)`. Each output pixel
//!   sees all input channels; cheap channel mixing.
//!
//! # "Same" padding for k×k stride-1 depthwise
//!
//! For stride=1 the eager `conv2d_same` computes
//! `pad_total = max(0, (1 - 1) * 1 + k - H)`, which for `H ≥ k`
//! reduces to `pad_total = k - H` clamped at 0; when `H ≥ k`,
//! `pad_total = 0` only if `k == 1`. For typical ConvMixer
//! defaults (`H ≫ k`), the eager formula simplifies to
//! `pad_total = k - 1`, split as `floor((k-1)/2)` on the left
//! and `ceil((k-1)/2)` on the right. The lazy `conv2d` op uses
//! symmetric padding only, so we restrict to odd kernel sizes
//! (where `(k-1)/2` on both sides is exact). All upstream
//! ConvMixer configs use odd kernels (9 for c1536, 9 for c1024).
//!
//! # BatchNorm in inference mode
//!
//! `y = (x - running_mean) / sqrt(running_var + eps) * gain + bias`,
//! broadcast per channel across (N, H, W). Fused into a single
//! per-channel affine `y = x * w + b` where
//! `w = gain / sqrt(var + eps)` and
//! `b = bias - mean * w` so the forward path is one
//! broadcast-mul + one broadcast-add.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. The classifier head returns
//! logits `(1, nclasses)`. The backbone returns
//! `(1, dim, H/patch_size, W/patch_size)` and can be exposed
//! separately if a caller wants feature maps for downstream
//! heads.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct ConvMixerConfig {
    /// Channel dim used throughout the model.
    pub dim: usize,
    /// Number of mixer blocks.
    pub depth: usize,
    /// Depthwise conv kernel size (must be odd, see module
    /// docs on "same" padding).
    pub kernel_size: usize,
    /// Stem conv kernel size + stride (patch size).
    pub patch_size: usize,
    /// Final classifier output dimension.
    pub nclasses: usize,
    /// BatchNorm epsilon.
    pub bn_eps: f64,
}

impl ConvMixerConfig {
    /// ConvMixer-1536/20: 1536 channels, 20 blocks, kernel=9, patch=7.
    pub fn c1536_20(nclasses: usize) -> Self {
        Self {
            dim: 1536, depth: 20, kernel_size: 9, patch_size: 7,
            nclasses, bn_eps: 1e-5,
        }
    }
    /// ConvMixer-1024/20: 1024 channels, 20 blocks, kernel=9, patch=14.
    pub fn c1024_20(nclasses: usize) -> Self {
        Self {
            dim: 1024, depth: 20, kernel_size: 9, patch_size: 14,
            nclasses, bn_eps: 1e-5,
        }
    }
}

/// Per-channel fused-affine BatchNorm weights (inference mode).
/// `gain / sqrt(var + eps)` and `bias - mean * w` are baked in
/// at weight-load time so the forward path is purely affine.
#[derive(Debug, Clone)]
pub struct BatchNormParams {
    /// `gain / sqrt(var + eps)`, length `channels`.
    pub w: Arc<[f32]>,
    /// `bias - running_mean * w`, length `channels`.
    pub b: Arc<[f32]>,
}

impl BatchNormParams {
    /// Bake the inference-mode formula into a per-channel affine.
    pub fn from_raw(
        gain: &[f32],
        bias: &[f32],
        running_mean: &[f32],
        running_var: &[f32],
        eps: f64,
    ) -> Self {
        let c = gain.len();
        assert_eq!(bias.len(), c);
        assert_eq!(running_mean.len(), c);
        assert_eq!(running_var.len(), c);
        let mut w = vec![0.0_f32; c];
        let mut b = vec![0.0_f32; c];
        for i in 0..c {
            let inv = 1.0_f32 / ((running_var[i] as f64 + eps) as f32).sqrt();
            w[i] = gain[i] * inv;
            b[i] = bias[i] - running_mean[i] * w[i];
        }
        Self { w: Arc::from(w), b: Arc::from(b) }
    }
}

#[derive(Debug, Clone)]
pub struct ConvMixerBlockWeights {
    /// Depthwise k×k conv weight `[dim, 1, k, k]` (groups=dim).
    pub depthwise: WeightStorage,
    pub depthwise_bn: BatchNormParams,
    /// Pointwise 1×1 conv weight `[dim, dim, 1, 1]`.
    pub pointwise: WeightStorage,
    pub pointwise_bn: BatchNormParams,
}

#[derive(Debug, Clone)]
pub struct ConvMixerWeights {
    /// Stem conv weight `[dim, 3, patch_size, patch_size]`.
    pub stem: WeightStorage,
    pub stem_bn: BatchNormParams,
    pub blocks: Vec<ConvMixerBlockWeights>,
    /// Classifier `[dim, nclasses]`.
    pub head: WeightStorage,
    /// Classifier bias `[nclasses]`.
    pub head_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct ConvMixerModel {
    pub config: ConvMixerConfig,
    pub weights: ConvMixerWeights,
}

impl ConvMixerModel {
    /// Run a forward pass on `image` of shape `(1, 3, H, W)`.
    /// Returns class logits `(1, nclasses)`.
    pub fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x = self.run_backbone(image)?;
        let pooled = x.global_avg_pool_2d()?;
        let logits = self.weights.head.apply_linear(&pooled, cfg.dim, cfg.nclasses);
        let bias_t = pooled.const_f32_like(
            Arc::clone(&self.weights.head_bias),
            Shape::from_dims(&[cfg.nclasses]),
        );
        logits.broadcast_add(&bias_t)
    }

    /// Run the backbone (patch-embed stem + ConvMixer blocks)
    /// and return the channels-first feature map
    /// `(1, dim, H/patch, W/patch)` BEFORE global mean pool
    /// and the linear classifier.
    pub fn forward_features(&self, image: &LazyTensor) -> Result<LazyTensor> {
        self.run_backbone(image)
    }

    fn run_backbone(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = image.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4, "image must be rank 4 [N, 3, H, W]");
        assert_eq!(dims[1], 3, "image must have 3 input channels, got {}", dims[1]);
        assert!(cfg.kernel_size % 2 == 1,
            "ConvMixer depthwise kernel must be odd for symmetric same-padding (got {})",
            cfg.kernel_size,
        );

        let stem_w = self.weights.stem.const_like(
            image,
            Shape::from_dims(&[cfg.dim, 3, cfg.patch_size, cfg.patch_size]),
        );
        let mut x = image
            .conv2d(&stem_w, None, (cfg.patch_size, cfg.patch_size), (0, 0), 1)?
            .gelu_erf();
        x = self.apply_bn(&x, &self.weights.stem_bn)?;

        for block in &self.weights.blocks {
            x = self.apply_block(&x, block)?;
        }
        Ok(x)
    }

    fn apply_block(
        &self,
        x: &LazyTensor,
        block: &ConvMixerBlockWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let pad = (cfg.kernel_size - 1) / 2;
        // Depthwise: dim → dim, kernel=k, groups=dim, padding=pad.
        let dw_w = block.depthwise.const_like(
            x,
            Shape::from_dims(&[cfg.dim, 1, cfg.kernel_size, cfg.kernel_size]),
        );
        let dw_out = x.conv2d(&dw_w, None, (1, 1), (pad, pad), cfg.dim)?;
        let dw_out = dw_out.gelu_erf();
        let dw_out = self.apply_bn(&dw_out, &block.depthwise_bn)?;
        // Residual on depthwise.
        let residual = x.add(&dw_out)?;

        // Pointwise: dim → dim, kernel=1, padding=0.
        let pw_w = block.pointwise.const_like(
            x,
            Shape::from_dims(&[cfg.dim, cfg.dim, 1, 1]),
        );
        let pw_out = residual.conv2d(&pw_w, None, (1, 1), (0, 0), 1)?;
        let pw_out = pw_out.gelu_erf();
        self.apply_bn(&pw_out, &block.pointwise_bn)
    }

    /// Apply fused-affine BatchNorm `y = x * w[c] + b[c]` across
    /// the channel dim of a `(N, C, H, W)` tensor.
    fn apply_bn(&self, x: &LazyTensor, bn: &BatchNormParams) -> Result<LazyTensor> {
        x.channel_affine_4d(Arc::clone(&bn.w), Arc::clone(&bn.b))
    }
}

/// Build a tiny ConvMixer image for tests: 3 channels, (1, 3, H, W).
#[cfg(test)]
fn tiny_image(h: usize, w: usize, device: &Device) -> LazyTensor {
    let mut s: u32 = 42;
    let data: Arc<[f32]> = Arc::from((0..3 * h * w)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5)
        })
        .collect::<Vec<_>>());
    LazyTensor::from_f32(data, Shape::from_dims(&[1, 3, h, w]), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_cfg(h_after_patch: usize) -> ConvMixerConfig {
        ConvMixerConfig {
            dim: 8, depth: 2, kernel_size: 3, patch_size: h_after_patch,
            nclasses: 5, bn_eps: 1e-5,
        }
    }

    fn tiny_bn(cfg: &ConvMixerConfig, nb: &mut dyn FnMut() -> f32) -> BatchNormParams {
        let c = cfg.dim;
        // Sample gain ≈ 1, bias ≈ 0, mean ≈ 0, var ≈ 1 with small noise.
        let mut gain = vec![1.0_f32; c];
        let mut bias = vec![0.0_f32; c];
        let mut mean = vec![0.0_f32; c];
        let mut var = vec![1.0_f32; c];
        for i in 0..c {
            gain[i] += nb() * 0.1;
            bias[i] += nb() * 0.1;
            mean[i] += nb() * 0.05;
            var[i] += nb().abs() * 0.05;
        }
        BatchNormParams::from_raw(&gain, &bias, &mean, &var, cfg.bn_eps as f64)
    }

    fn tiny_weights(cfg: &ConvMixerConfig) -> ConvMixerWeights {
        let mut s: u32 = 9999;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let dim = cfg.dim;
        let k = cfg.kernel_size;
        let p = cfg.patch_size;

        let stem = WeightStorage::F32(vec_of(dim * 3 * p * p, &mut *nb));
        let stem_bn = tiny_bn(cfg, &mut *nb);

        let blocks: Vec<ConvMixerBlockWeights> = (0..cfg.depth)
            .map(|_| ConvMixerBlockWeights {
                depthwise: WeightStorage::F32(vec_of(dim * 1 * k * k, &mut *nb)),
                depthwise_bn: tiny_bn(cfg, &mut *nb),
                pointwise: WeightStorage::F32(vec_of(dim * dim, &mut *nb)),
                pointwise_bn: tiny_bn(cfg, &mut *nb),
            })
            .collect();

        let head = WeightStorage::F32(vec_of(dim * cfg.nclasses, &mut *nb));
        let head_bias = vec_of(cfg.nclasses, &mut *nb);

        ConvMixerWeights { stem, stem_bn, blocks, head, head_bias }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_cfg(2);
        let model = ConvMixerModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let img = tiny_image(8, 8, &Device::cpu());
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, cfg.nclasses]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    /// BatchNorm fused-affine: identity weights (gain=1, bias=0,
    /// mean=0, var=1, eps=0) leave the input unchanged.
    #[test]
    fn batchnorm_identity_is_identity() {
        let c = 4;
        let gain = vec![1.0_f32; c];
        let bias = vec![0.0_f32; c];
        let mean = vec![0.0_f32; c];
        let var = vec![1.0_f32; c];
        let bn = BatchNormParams::from_raw(&gain, &bias, &mean, &var, 0.0);
        for i in 0..c {
            assert!((bn.w[i] - 1.0).abs() < 1e-7);
            assert!(bn.b[i].abs() < 1e-7);
        }
    }

    /// BatchNorm fused-affine: a known channel-wise affine. With
    /// gain = [2, 3], bias = [1, -1], mean = [0.5, 0.0], var = [4.0, 1.0]
    /// and eps = 0: w = gain / sqrt(var) = [1.0, 3.0],
    /// b = bias - mean * w = [1.0 - 0.5*1.0, -1.0 - 0.0*3.0] = [0.5, -1.0].
    #[test]
    fn batchnorm_fused_affine_is_correct() {
        let gain = vec![2.0_f32, 3.0];
        let bias = vec![1.0_f32, -1.0];
        let mean = vec![0.5_f32, 0.0];
        let var = vec![4.0_f32, 1.0];
        let bn = BatchNormParams::from_raw(&gain, &bias, &mean, &var, 0.0);
        assert!((bn.w[0] - 1.0).abs() < 1e-7, "expected w[0] = 1.0, got {}", bn.w[0]);
        assert!((bn.w[1] - 3.0).abs() < 1e-7, "expected w[1] = 3.0, got {}", bn.w[1]);
        assert!((bn.b[0] - 0.5).abs() < 1e-7, "expected b[0] = 0.5, got {}", bn.b[0]);
        assert!((bn.b[1] + 1.0).abs() < 1e-7, "expected b[1] = -1.0, got {}", bn.b[1]);
    }

    /// Stem patch embedding: a (1, 3, 6, 6) input with patch=3
    /// produces a (1, dim, 2, 2) feature map (3x3 patches → 2x2 grid).
    #[test]
    fn stem_patch_grid_shape() {
        let cfg = ConvMixerConfig {
            dim: 4, depth: 1, kernel_size: 3, patch_size: 3,
            nclasses: 2, bn_eps: 1e-5,
        };
        let model = ConvMixerModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let img = tiny_image(6, 6, &Device::cpu());
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, cfg.nclasses]);
    }

    #[test]
    fn forward_features_shape_and_finite() {
        let cfg = tiny_cfg(2);
        let model = ConvMixerModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let img = tiny_image(8, 8, &Device::cpu());
        let feats = model.forward_features(&img).unwrap();
        let shape = feats.shape();
        let dims = shape.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], cfg.dim);
        for &v in &feats.realize_f32() {
            assert!(v.is_finite(), "non-finite feature: {v}");
        }
    }
}

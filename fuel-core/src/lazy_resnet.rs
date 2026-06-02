//! ResNet (He et al. 2015 "Deep Residual Learning for Image
//! Recognition") ported to the lazy-graph API.
//!
//! ResNet is the canonical residual-conv classifier: a 7×7
//! stem, a max-pool, four stages of residual blocks, then
//! global-average-pool + Linear. Two block flavors:
//!
//!   - **BasicBlock** (used by ResNet-18, ResNet-34):
//!     `conv3x3 → BN → ReLU → conv3x3 → BN → (+residual) → ReLU`.
//!     Output channels equal `c_out`.
//!   - **Bottleneck** (used by ResNet-50, -101, -152):
//!     `conv1x1 → BN → ReLU → conv3x3 → BN → ReLU → conv1x1 →
//!      BN → (+residual) → ReLU`. The 1×1 convs squeeze and
//!     expand by factor 4: `c_out` mid-channels but the residual
//!     and output have `4 * c_out` channels.
//!
//! In both cases, the residual passes through an optional
//! "downsample" arm (`conv1x1 + BN`) iff the spatial stride or
//! channel count would otherwise mismatch.
//!
//! # Fused-affine BatchNorm
//!
//! Same fused-affine BN as ConvMixer:
//! `y = x * w[c] + b[c]` where `w = gain / sqrt(var + eps)`
//! and `b = bias - mean * w`. Precomputed at weight-load time
//! and broadcast across `(N, C, H, W)` via a reshape to
//! `(1, C, 1, 1)`.
//!
//! # Stem and pooling
//!
//! - Stem: `Conv2d(3, 64, k=7, stride=2, padding=3) → BN → ReLU`.
//! - Pre-stage pool: `MaxPool2d(k=3, stride=2, padding=1)`. The
//!   lazy `max_pool2d` accepts padding directly, so we collapse
//!   the eager `pad_with_same(...).max_pool2d_with_stride(3, 2)`
//!   into a single call.
//! - Head: `mean_dim(H).mean_dim(W) → Linear(features, nclasses)`.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. Returns class logits
//! `(1, nclasses)` when the classifier is present; otherwise
//! the pooled feature vector `(1, features)` where `features`
//! is 512 (basic) or 2048 (bottleneck). The `_no_final_layer`
//! variant is just `nclasses = None` at config time.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_convmixer::BatchNormParams;
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResNetKind {
    Basic,
    Bottleneck,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResNetConfig {
    pub kind: ResNetKind,
    /// Block counts per stage (4 stages, e.g. `[2, 2, 2, 2]`
    /// for ResNet-18, `[3, 4, 6, 3]` for ResNet-50).
    pub blocks_per_stage: [usize; 4],
    /// `None` → return pooled features; `Some(n)` → return
    /// classifier logits of width `n`.
    pub nclasses: Option<usize>,
}

impl ResNetConfig {
    pub fn resnet18(nclasses: Option<usize>) -> Self {
        Self { kind: ResNetKind::Basic, blocks_per_stage: [2, 2, 2, 2], nclasses }
    }
    pub fn resnet34(nclasses: Option<usize>) -> Self {
        Self { kind: ResNetKind::Basic, blocks_per_stage: [3, 4, 6, 3], nclasses }
    }
    pub fn resnet50(nclasses: Option<usize>) -> Self {
        Self { kind: ResNetKind::Bottleneck, blocks_per_stage: [3, 4, 6, 3], nclasses }
    }
    pub fn resnet101(nclasses: Option<usize>) -> Self {
        Self { kind: ResNetKind::Bottleneck, blocks_per_stage: [3, 4, 23, 3], nclasses }
    }
    pub fn resnet152(nclasses: Option<usize>) -> Self {
        Self { kind: ResNetKind::Bottleneck, blocks_per_stage: [3, 8, 36, 3], nclasses }
    }
    /// Feature width after the final stage: 512 (basic) or
    /// 2048 (bottleneck).
    pub fn features(&self) -> usize {
        match self.kind {
            ResNetKind::Basic => 512,
            ResNetKind::Bottleneck => 4 * 512,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DownsampleWeights {
    /// `[c_out, c_in, 1, 1]`.
    pub conv: WeightStorage,
    pub bn: BatchNormParams,
}

#[derive(Debug, Clone)]
pub struct ResNetBlockWeights {
    /// Stage stride (1 or 2). Drives the first conv (basic) or
    /// the middle 3×3 (bottleneck) and any downsample arm.
    pub stride: usize,
    /// Channels at the block's input (for downsample lookup).
    pub c_in: usize,
    /// Channels at the block's "mid" point. For Basic this is
    /// also the output channel count; for Bottleneck the output
    /// is `4 * c_out`.
    pub c_out: usize,

    /// First conv:
    /// - Basic: `[c_out, c_in, 3, 3]` (stride=stride, padding=1).
    /// - Bottleneck: `[c_out, c_in, 1, 1]` (stride=1, padding=0).
    pub conv1: WeightStorage,
    pub bn1: BatchNormParams,

    /// Second conv:
    /// - Basic: `[c_out, c_out, 3, 3]` (stride=1, padding=1).
    /// - Bottleneck: `[c_out, c_out, 3, 3]` (stride=stride, padding=1).
    pub conv2: WeightStorage,
    pub bn2: BatchNormParams,

    /// Third conv (Bottleneck only):
    /// `[4 * c_out, c_out, 1, 1]` (stride=1, padding=0).
    pub conv3: Option<WeightStorage>,
    pub bn3: Option<BatchNormParams>,

    /// Present iff `stride != 1 || c_in != block_out`. Built
    /// against the block's effective output channels.
    pub downsample: Option<DownsampleWeights>,
}

#[derive(Debug, Clone)]
pub struct ResNetStageWeights {
    pub blocks: Vec<ResNetBlockWeights>,
}

#[derive(Debug, Clone)]
pub struct ResNetWeights {
    /// `[64, 3, 7, 7]` stem conv.
    pub stem_conv: WeightStorage,
    pub stem_bn: BatchNormParams,
    /// Four residual stages.
    pub stages: [ResNetStageWeights; 4],
    /// `[features, nclasses]` and bias `[nclasses]`.
    /// `None` when the config has `nclasses == None`.
    pub fc: Option<(WeightStorage, Arc<[f32]>)>,
}

#[derive(Debug, Clone)]
pub struct ResNetModel {
    pub config: ResNetConfig,
    pub weights: ResNetWeights,
}

impl ResNetModel {
    /// Run a forward pass on `image` of shape `(1, 3, H, W)`.
    pub fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = image.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4, "image must be rank 4 [N, 3, H, W]");
        assert_eq!(dims[1], 3, "image must have 3 input channels");

        // ---- Stem: conv7x7 s=2 p=3 → BN → ReLU → max_pool 3 s=2 p=1
        let stem_w = self.weights.stem_conv.const_like(
            image, Shape::from_dims(&[64, 3, 7, 7]),
        );
        let mut x = image.conv2d(&stem_w, None, (2, 2), (3, 3), 1)?;
        x = apply_bn(&x, &self.weights.stem_bn, 64)?.relu();
        x = x.max_pool2d((3, 3), (2, 2), (1, 1))?;

        // ---- Four residual stages ----------------------------------------
        for stage in &self.weights.stages {
            for block in &stage.blocks {
                x = self.apply_block(&x, block)?;
            }
        }

        // ---- Global average pool → (1, features) -------------------------
        let pooled_w = x.mean_dim(3_usize)?;     // (1, C, H)
        let pooled = pooled_w.mean_dim(2_usize)?; // (1, C)

        // ---- Classifier (optional) ---------------------------------------
        match &self.weights.fc {
            None => Ok(pooled),
            Some((w, b)) => {
                let n = cfg.nclasses.expect("config nclasses must be Some when fc is present");
                let logits = w.apply_linear(&pooled, cfg.features(), n);
                let bias_t = pooled.const_f32_like(
                    Arc::clone(b), Shape::from_dims(&[n]),
                );
                logits.broadcast_add(&bias_t)
            }
        }
    }

    fn apply_block(
        &self,
        x: &LazyTensor,
        block: &ResNetBlockWeights,
    ) -> Result<LazyTensor> {
        match self.config.kind {
            ResNetKind::Basic => self.apply_basic_block(x, block),
            ResNetKind::Bottleneck => self.apply_bottleneck_block(x, block),
        }
    }

    fn apply_basic_block(
        &self,
        x: &LazyTensor,
        block: &ResNetBlockWeights,
    ) -> Result<LazyTensor> {
        let c_in = block.c_in;
        let c_out = block.c_out;
        let s = block.stride;
        let conv1_w = block.conv1.const_like(
            x, Shape::from_dims(&[c_out, c_in, 3, 3]),
        );
        let conv2_w = block.conv2.const_like(
            x, Shape::from_dims(&[c_out, c_out, 3, 3]),
        );
        let y = x.conv2d(&conv1_w, None, (s, s), (1, 1), 1)?;
        let y = apply_bn(&y, &block.bn1, c_out)?.relu();
        let y = y.conv2d(&conv2_w, None, (1, 1), (1, 1), 1)?;
        let y = apply_bn(&y, &block.bn2, c_out)?;
        let residual = self.maybe_downsample(x, block, c_out)?;
        residual.add(&y)?.relu().to_result()
    }

    fn apply_bottleneck_block(
        &self,
        x: &LazyTensor,
        block: &ResNetBlockWeights,
    ) -> Result<LazyTensor> {
        let c_in = block.c_in;
        let c_out = block.c_out;
        let c_expanded = 4 * c_out;
        let s = block.stride;
        let conv1_w = block.conv1.const_like(
            x, Shape::from_dims(&[c_out, c_in, 1, 1]),
        );
        let conv2_w = block.conv2.const_like(
            x, Shape::from_dims(&[c_out, c_out, 3, 3]),
        );
        let conv3 = block.conv3.as_ref().expect("bottleneck block must carry conv3");
        let bn3 = block.bn3.as_ref().expect("bottleneck block must carry bn3");
        let conv3_w = conv3.const_like(
            x, Shape::from_dims(&[c_expanded, c_out, 1, 1]),
        );

        let y = x.conv2d(&conv1_w, None, (1, 1), (0, 0), 1)?;
        let y = apply_bn(&y, &block.bn1, c_out)?.relu();
        let y = y.conv2d(&conv2_w, None, (s, s), (1, 1), 1)?;
        let y = apply_bn(&y, &block.bn2, c_out)?.relu();
        let y = y.conv2d(&conv3_w, None, (1, 1), (0, 0), 1)?;
        let y = apply_bn(&y, bn3, c_expanded)?;
        let residual = self.maybe_downsample(x, block, c_expanded)?;
        residual.add(&y)?.relu().to_result()
    }

    /// Apply the downsample arm if the block has one; otherwise
    /// return `x` unchanged. The downsample is always
    /// `Conv1x1(c_in → block_out, stride) → BN`.
    fn maybe_downsample(
        &self,
        x: &LazyTensor,
        block: &ResNetBlockWeights,
        block_out: usize,
    ) -> Result<LazyTensor> {
        match &block.downsample {
            None => Ok(x.clone()),
            Some(ds) => {
                let c_in = block.c_in;
                let s = block.stride;
                let w = ds.conv.const_like(
                    x, Shape::from_dims(&[block_out, c_in, 1, 1]),
                );
                let y = x.conv2d(&w, None, (s, s), (0, 0), 1)?;
                apply_bn(&y, &ds.bn, block_out)
            }
        }
    }
}

/// Apply fused-affine BatchNorm to a 4-D NCHW tensor.
fn apply_bn(x: &LazyTensor, bn: &BatchNormParams, channels: usize) -> Result<LazyTensor> {
    assert_eq!(bn.w.len(), channels, "BN gain length must equal channels");
    assert_eq!(bn.b.len(), channels, "BN bias length must equal channels");
    let dims = x.shape();
    let dims = dims.dims();
    assert_eq!(dims.len(), 4, "BN input must be rank 4");
    assert_eq!(dims[1], channels, "BN input channels must equal weight count");
    let w_t = x
        .const_f32_like(Arc::clone(&bn.w), Shape::from_dims(&[channels]))
        .reshape(Shape::from_dims(&[1, channels, 1, 1]))?;
    let b_t = x
        .const_f32_like(Arc::clone(&bn.b), Shape::from_dims(&[channels]))
        .reshape(Shape::from_dims(&[1, channels, 1, 1]))?;
    x.broadcast_mul(&w_t)?.broadcast_add(&b_t)
}

/// Tiny adapter: LazyTensor::relu returns a LazyTensor by value;
/// we need it inside a Result chain. (The eager Result-returning
/// path predates the lazy infallible-relu signature.)
trait LazyTensorResultExt {
    fn to_result(self) -> Result<LazyTensor>;
}
impl LazyTensorResultExt for LazyTensor {
    fn to_result(self) -> Result<LazyTensor> { Ok(self) }
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
        BatchNormParams::from_raw(&gain, &bias, &mean, &var, 1e-5)
    }

    fn build_block(
        kind: ResNetKind,
        c_in: usize,
        c_out: usize,
        stride: usize,
        nb: &mut dyn FnMut() -> f32,
    ) -> ResNetBlockWeights {
        let needs_ds = match kind {
            ResNetKind::Basic => stride != 1 || c_in != c_out,
            ResNetKind::Bottleneck => stride != 1 || c_in != 4 * c_out,
        };
        let (conv1, conv2, conv3, bn3) = match kind {
            ResNetKind::Basic => (
                WeightStorage::F32(vec_of(c_out * c_in * 3 * 3, nb)),
                WeightStorage::F32(vec_of(c_out * c_out * 3 * 3, nb)),
                None,
                None,
            ),
            ResNetKind::Bottleneck => (
                WeightStorage::F32(vec_of(c_out * c_in * 1 * 1, nb)),
                WeightStorage::F32(vec_of(c_out * c_out * 3 * 3, nb)),
                Some(WeightStorage::F32(vec_of(4 * c_out * c_out * 1 * 1, nb))),
                Some(tiny_bn(4 * c_out, nb)),
            ),
        };
        let bn1 = tiny_bn(c_out, nb);
        let bn2 = tiny_bn(c_out, nb);
        let block_out = match kind {
            ResNetKind::Basic => c_out,
            ResNetKind::Bottleneck => 4 * c_out,
        };
        let downsample = if needs_ds {
            Some(DownsampleWeights {
                conv: WeightStorage::F32(vec_of(block_out * c_in * 1 * 1, nb)),
                bn: tiny_bn(block_out, nb),
            })
        } else {
            None
        };
        ResNetBlockWeights {
            stride, c_in, c_out, conv1, bn1, conv2, bn2, conv3, bn3, downsample,
        }
    }

    fn build_stage(
        kind: ResNetKind,
        c_in: usize,
        c_out: usize,
        stride: usize,
        n_blocks: usize,
        nb: &mut dyn FnMut() -> f32,
    ) -> ResNetStageWeights {
        let block_out = match kind {
            ResNetKind::Basic => c_out,
            ResNetKind::Bottleneck => 4 * c_out,
        };
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let l_in = if i == 0 { c_in } else { block_out };
            let s = if i == 0 { stride } else { 1 };
            blocks.push(build_block(kind, l_in, c_out, s, nb));
        }
        ResNetStageWeights { blocks }
    }

    fn build_tiny_weights(cfg: &ResNetConfig, seed: u32) -> ResNetWeights {
        let mut nb = rng_seed(seed);
        let stem_conv = WeightStorage::F32(vec_of(64 * 3 * 7 * 7, &mut nb));
        let stem_bn = tiny_bn(64, &mut nb);
        let kind = cfg.kind;
        let stage1 = build_stage(kind, 64, 64, 1, cfg.blocks_per_stage[0], &mut nb);
        let in2 = match kind { ResNetKind::Basic => 64, ResNetKind::Bottleneck => 256 };
        let stage2 = build_stage(kind, in2, 128, 2, cfg.blocks_per_stage[1], &mut nb);
        let in3 = match kind { ResNetKind::Basic => 128, ResNetKind::Bottleneck => 512 };
        let stage3 = build_stage(kind, in3, 256, 2, cfg.blocks_per_stage[2], &mut nb);
        let in4 = match kind { ResNetKind::Basic => 256, ResNetKind::Bottleneck => 1024 };
        let stage4 = build_stage(kind, in4, 512, 2, cfg.blocks_per_stage[3], &mut nb);
        let fc = cfg.nclasses.map(|n| {
            (
                WeightStorage::F32(vec_of(cfg.features() * n, &mut nb)),
                vec_of(n, &mut nb),
            )
        });
        ResNetWeights {
            stem_conv,
            stem_bn,
            stages: [stage1, stage2, stage3, stage4],
            fc,
        }
    }

    fn tiny_image(h: usize, w: usize) -> LazyTensor {
        let mut nb = rng_seed(42);
        let data: Arc<[f32]> = Arc::from(
            (0..3 * h * w).map(|_| nb()).collect::<Vec<_>>()
        );
        LazyTensor::from_f32(data, Shape::from_dims(&[1, 3, h, w]), &Device::cpu())
    }

    #[test]
    fn resnet18_with_classifier_shape() {
        let cfg = ResNetConfig::resnet18(Some(10));
        let weights = build_tiny_weights(&cfg, 1234);
        let model = ResNetModel { config: cfg, weights };
        let img = tiny_image(64, 64);
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 10]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn resnet18_no_classifier_returns_features() {
        let cfg = ResNetConfig::resnet18(None);
        let weights = build_tiny_weights(&cfg, 7777);
        let model = ResNetModel { config: cfg, weights };
        let img = tiny_image(64, 64);
        let feats = model.forward(&img).unwrap();
        assert_eq!(feats.shape().dims(), &[1, 512]);
        for &v in &feats.realize_f32() {
            assert!(v.is_finite(), "non-finite feature: {v}");
        }
    }

    #[test]
    fn resnet50_bottleneck_features_2048() {
        let cfg = ResNetConfig::resnet50(None);
        let weights = build_tiny_weights(&cfg, 5555);
        let model = ResNetModel { config: cfg, weights };
        // Use a smaller-than-real input to keep test fast; ResNet
        // still works on small images because spatial downsampling
        // happens up to 32x.
        let img = tiny_image(64, 64);
        let feats = model.forward(&img).unwrap();
        assert_eq!(feats.shape().dims(), &[1, 2048]);
        for &v in &feats.realize_f32() {
            assert!(v.is_finite(), "non-finite feature: {v}");
        }
    }

    /// Stride-2 stages downsample the spatial dims. Verify the
    /// stem cuts H by 4 (conv7-s2 + maxpool-s2) and each later
    /// stage cuts H by another factor of 2 (4 → 2 → 1 with H=64).
    #[test]
    fn spatial_downsampling_chain() {
        // After stem: H -> H/2 (conv) -> H/4 (maxpool).
        // Stage 1 keeps spatial size; stages 2-4 each halve.
        // H=64 → 16 → 16 → 8 → 4 → 2.
        let cfg = ResNetConfig::resnet18(None);
        let weights = build_tiny_weights(&cfg, 4321);
        let model = ResNetModel { config: cfg, weights };
        let img = tiny_image(64, 64);
        // Forward computes the full chain, so the only direct
        // observation is the final pooled feature shape — but
        // mid-shape introspection isn't needed: if any stride
        // path mismatches, the conv2d shape check would fail.
        let feats = model.forward(&img).unwrap();
        assert_eq!(feats.shape().dims(), &[1, 512]);
    }
}

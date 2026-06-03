//! VGG-13 / VGG-16 / VGG-19 (Simonyan & Zisserman 2014, "Very
//! Deep Convolutional Networks for Large-Scale Image
//! Recognition") ported to the lazy-graph API.
//!
//! VGG is the canonical "all-3×3 convs" classifier: stacks of
//! ReLU'd 3×3 convs separated by 2×2 max-pool downsamples, then
//! three fully-connected layers. No BatchNorm, no skip
//! connections, no fancy heads — the architectural simplicity
//! is the point.
//!
//! # Architecture
//!
//! ```text
//! conv_block_1 (N convs × [3, 64] or [64, 64], +MaxPool)
//! conv_block_2 (..., +MaxPool)
//! conv_block_3 (..., +MaxPool)
//! conv_block_4 (..., +MaxPool)
//! conv_block_5 (..., +MaxPool)
//! Flatten     (1, 512 * 7 * 7) at 224×224 input
//! FC(25088 → 4096) → ReLU → (Dropout disabled in inference)
//! FC(4096 → 4096)  → ReLU
//! FC(4096 → nclasses) → ReLU
//! ```
//!
//! VGG-13 has 10 convs (2+2+2+2+2), VGG-16 has 13 convs
//! (2+2+3+3+3), VGG-19 has 16 convs (2+2+4+4+4). All convs are
//! 3×3 with stride 1 and padding 1 ("same" for stride 1).
//! Pools are 2×2 with stride 2 and no padding.
//!
//! # Final ReLU on the classifier head
//!
//! Eager Fuel's VGG (mirroring the upstream timm impl) applies
//! `ReLU` after the FINAL FC. That's non-standard for a
//! classifier — typically the logits go straight into a softmax
//! / cross-entropy loss — but the lazy port reproduces it
//! verbatim so weights load identically. Callers that want raw
//! logits can drop the final ReLU.
//!
//! # Dropout in inference
//!
//! VGG carries three Dropout(p=0.5) instances in the FC head.
//! At inference (`train=false`), dropout is the identity; the
//! lazy port simply omits it.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. Input is `(1, 3, H, W)` with
//! `H == W` and divisible by 32 (typically 224). Returns
//! `(1, nclasses)` after the final FC + ReLU.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VggVariant {
    Vgg13,
    Vgg16,
    Vgg19,
}

impl VggVariant {
    /// Conv counts per block (5 blocks).
    pub fn convs_per_block(&self) -> [usize; 5] {
        match self {
            Self::Vgg13 => [2, 2, 2, 2, 2],
            Self::Vgg16 => [2, 2, 3, 3, 3],
            Self::Vgg19 => [2, 2, 4, 4, 4],
        }
    }
}

#[derive(Debug, Clone)]
pub struct VggConfig {
    pub variant: VggVariant,
    pub nclasses: usize,
    /// Expected post-pool spatial size (typically 7 for a 224×224 input).
    pub head_spatial: usize,
    /// FC head hidden width (4096 for canonical VGG).
    pub head_hidden: usize,
}

impl VggConfig {
    pub fn vgg13(nclasses: usize) -> Self {
        Self { variant: VggVariant::Vgg13, nclasses, head_spatial: 7, head_hidden: 4096 }
    }
    pub fn vgg16(nclasses: usize) -> Self {
        Self { variant: VggVariant::Vgg16, nclasses, head_spatial: 7, head_hidden: 4096 }
    }
    pub fn vgg19(nclasses: usize) -> Self {
        Self { variant: VggVariant::Vgg19, nclasses, head_spatial: 7, head_hidden: 4096 }
    }
}

/// One 3×3 conv + bias. Weight shape `[c_out, c_in, 3, 3]`.
#[derive(Debug, Clone)]
pub struct VggConvWeights {
    pub w: WeightStorage,
    pub b: Arc<[f32]>,
    pub c_in: usize,
    pub c_out: usize,
}

#[derive(Debug, Clone)]
pub struct VggHeadFc {
    /// `[in_features, out_features]`.
    pub w: WeightStorage,
    /// `[out_features]`.
    pub b: Arc<[f32]>,
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Debug, Clone)]
pub struct VggWeights {
    /// Conv blocks, in evaluation order. Each block is a list
    /// of convs (2-4) followed by an implicit max-pool.
    pub blocks: Vec<Vec<VggConvWeights>>,
    pub fc1: VggHeadFc,
    pub fc2: VggHeadFc,
    pub fc3: VggHeadFc,
}

#[derive(Debug, Clone)]
pub struct VggModel {
    pub config: VggConfig,
    pub weights: VggWeights,
}

impl VggModel {
    /// Run a forward pass on `image` of shape `(1, 3, H, W)`.
    /// Returns `(1, nclasses)`.
    pub fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x = self.run_backbone(image)?;

        let final_dims = x.shape();
        let final_dims = final_dims.dims();
        let n = final_dims[0];
        let c = final_dims[1];
        let h = final_dims[2];
        let w = final_dims[3];
        assert_eq!(h, cfg.head_spatial,
            "VGG head expects post-conv spatial size {}, got {}",
            cfg.head_spatial, h);
        assert_eq!(w, cfg.head_spatial,
            "VGG head expects post-conv spatial size {}, got {}",
            cfg.head_spatial, w);
        let flat_dim = c * h * w;
        assert_eq!(flat_dim, self.weights.fc1.in_features,
            "VGG fc1.in_features mismatch: flattened {flat_dim} vs weight {}",
            self.weights.fc1.in_features);
        let flat = x.reshape(Shape::from_dims(&[n, flat_dim]))?;

        let h1 = self.apply_fc(&flat, &self.weights.fc1)?.relu();
        let h2 = self.apply_fc(&h1, &self.weights.fc2)?.relu();
        let h3 = self.apply_fc(&h2, &self.weights.fc3)?.relu();
        Ok(h3)
    }

    /// Run the conv backbone (all 5 conv blocks + max-pools)
    /// and return the channels-first feature map
    /// `(1, last_block_channels, H/32, W/32)` BEFORE flatten
    /// and the FC head. Useful for downstream dense prediction
    /// or as a frozen feature extractor.
    pub fn forward_features(&self, image: &LazyTensor) -> Result<LazyTensor> {
        self.run_backbone(image)
    }

    fn run_backbone(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let dims = image.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4, "image must be rank 4 [N, 3, H, W]");
        assert_eq!(dims[1], 3, "image must have 3 input channels");
        assert_eq!(dims[2], dims[3], "VGG expects square inputs");
        assert!(
            dims[2] % 32 == 0,
            "VGG input H must be divisible by 32 (5 stride-2 pools), got {}",
            dims[2],
        );

        let mut x = image.clone();
        for block in &self.weights.blocks {
            for conv in block {
                x = self.apply_conv(&x, conv)?.relu();
            }
            x = x.max_pool2d((2, 2), (2, 2), (0, 0))?;
        }
        Ok(x)
    }

    fn apply_conv(&self, x: &LazyTensor, conv: &VggConvWeights) -> Result<LazyTensor> {
        let w = conv.w.const_like(
            x, Shape::from_dims(&[conv.c_out, conv.c_in, 3, 3]),
        );
        let out = x.conv2d(&w, None, (1, 1), (1, 1), 1)?;
        let bias_t = x
            .const_f32_like(Arc::clone(&conv.b), Shape::from_dims(&[conv.c_out]))
            .reshape(Shape::from_dims(&[1, conv.c_out, 1, 1]))?;
        out.broadcast_add(&bias_t)
    }

    fn apply_fc(&self, x: &LazyTensor, fc: &VggHeadFc) -> Result<LazyTensor> {
        let out = fc.w.apply_linear(x, fc.in_features, fc.out_features);
        let bias_t = x.const_f32_like(
            Arc::clone(&fc.b), Shape::from_dims(&[fc.out_features]),
        );
        out.broadcast_add(&bias_t)
    }
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

    fn tiny_cfg(variant: VggVariant) -> VggConfig {
        // Use a tiny "feature scale" — 32×32 input → 1×1 after 5 pools.
        // Channels follow the canonical VGG pattern but at small width.
        VggConfig {
            variant, nclasses: 10, head_spatial: 1, head_hidden: 16,
        }
    }

    /// Channel widths per block, scaled down from the canonical
    /// 64 / 128 / 256 / 512 / 512 to keep tests fast.
    fn tiny_block_channels() -> [usize; 5] {
        [4, 8, 16, 32, 32]
    }

    fn build_conv(c_in: usize, c_out: usize, nb: &mut dyn FnMut() -> f32) -> VggConvWeights {
        VggConvWeights {
            w: WeightStorage::F32(vec_of(c_out * c_in * 3 * 3, nb)),
            b: vec_of(c_out, nb),
            c_in, c_out,
        }
    }

    fn build_weights(cfg: &VggConfig, seed: u32) -> VggWeights {
        let mut nb = rng_seed(seed);
        let convs_per_block = cfg.variant.convs_per_block();
        let block_ch = tiny_block_channels();
        let mut blocks = Vec::with_capacity(5);
        let mut c_prev = 3;
        for (block_idx, &n_conv) in convs_per_block.iter().enumerate() {
            let mut block = Vec::with_capacity(n_conv);
            for conv_idx in 0..n_conv {
                let c_in = if conv_idx == 0 { c_prev } else { block_ch[block_idx] };
                let c_out = block_ch[block_idx];
                block.push(build_conv(c_in, c_out, &mut nb));
            }
            c_prev = block_ch[block_idx];
            blocks.push(block);
        }
        let flat_in = c_prev * cfg.head_spatial * cfg.head_spatial;
        let fc1 = VggHeadFc {
            w: WeightStorage::F32(vec_of(flat_in * cfg.head_hidden, &mut nb)),
            b: vec_of(cfg.head_hidden, &mut nb),
            in_features: flat_in,
            out_features: cfg.head_hidden,
        };
        let fc2 = VggHeadFc {
            w: WeightStorage::F32(vec_of(cfg.head_hidden * cfg.head_hidden, &mut nb)),
            b: vec_of(cfg.head_hidden, &mut nb),
            in_features: cfg.head_hidden,
            out_features: cfg.head_hidden,
        };
        let fc3 = VggHeadFc {
            w: WeightStorage::F32(vec_of(cfg.head_hidden * cfg.nclasses, &mut nb)),
            b: vec_of(cfg.nclasses, &mut nb),
            in_features: cfg.head_hidden,
            out_features: cfg.nclasses,
        };
        VggWeights { blocks, fc1, fc2, fc3 }
    }

    fn tiny_image(h: usize) -> LazyTensor {
        let mut nb = rng_seed(123);
        let data: Arc<[f32]> = Arc::from(
            (0..3 * h * h).map(|_| nb()).collect::<Vec<_>>()
        );
        LazyTensor::from_f32(data, Shape::from_dims(&[1, 3, h, h]), &Device::cpu())
    }

    #[test]
    fn vgg13_forward_shape() {
        let cfg = tiny_cfg(VggVariant::Vgg13);
        let weights = build_weights(&cfg, 11);
        let model = VggModel { config: cfg, weights };
        let img = tiny_image(32);
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 10]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
            // Final-layer ReLU means logits are >= 0.
            assert!(v >= 0.0, "VGG final ReLU should make logits >= 0, got {v}");
        }
    }

    #[test]
    fn vgg16_forward_shape() {
        let cfg = tiny_cfg(VggVariant::Vgg16);
        let weights = build_weights(&cfg, 22);
        let model = VggModel { config: cfg, weights };
        let img = tiny_image(32);
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 10]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn vgg19_forward_shape() {
        let cfg = tiny_cfg(VggVariant::Vgg19);
        let weights = build_weights(&cfg, 33);
        let model = VggModel { config: cfg, weights };
        let img = tiny_image(32);
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 10]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// Conv counts per variant follow the canonical VGG schedule.
    #[test]
    fn variant_conv_counts() {
        assert_eq!(VggVariant::Vgg13.convs_per_block(), [2, 2, 2, 2, 2]);
        assert_eq!(VggVariant::Vgg16.convs_per_block(), [2, 2, 3, 3, 3]);
        assert_eq!(VggVariant::Vgg19.convs_per_block(), [2, 2, 4, 4, 4]);
    }

    #[test]
    fn forward_features_shape_and_finite() {
        let cfg = tiny_cfg(VggVariant::Vgg13);
        let weights = build_weights(&cfg, 44);
        let model = VggModel { config: cfg.clone(), weights };
        let img = tiny_image(32);
        let feats = model.forward_features(&img).unwrap();
        let shape = feats.shape();
        let dims = shape.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[2], cfg.head_spatial);
        assert_eq!(dims[3], cfg.head_spatial);
        for &v in &feats.realize_f32() {
            assert!(v.is_finite(), "non-finite feature: {v}");
        }
    }
}

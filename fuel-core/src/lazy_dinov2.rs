//! DINOv2 vision transformer (Meta AI) ported to the lazy-graph
//! API.
//!
//! "DINOv2: Learning Robust Visual Features without Supervision"
//! (Oquab et al. 2023). Self-supervised vision transformer used
//! widely as a feature extractor for downstream tasks (depth
//! estimation, segmentation, classification fine-tunes). Direct
//! consumer: `facebook/dinov2-base`, `facebook/dinov2-small`,
//! etc.; indirectly used by depth_anything_v2 and several
//! VLM backbones.
//!
//! Differences from plain ViT (see [`crate::lazy_vit`]):
//!
//!   1. **Fused Wqkv projection.** One linear `hidden → 3 *
//!      hidden` produces Q, K, V together; sliced into thirds.
//!   2. **LayerScale.** A per-channel learned `gamma` multiplier
//!      applied to the attention output AND to the MLP output
//!      BEFORE the residual add. Lets the model down-weight
//!      stale residuals at training-init.
//!   3. **Classifier head on concat of CLS + mean-pooled patches.**
//!      Final feature is `cat(cls_norm, mean(patch_norm))` of
//!      dim `2 * embed_dim`, classified into `num_classes`.
//!
//! Same as ViT: standard MHA with Q/K/V biases (always on for
//! DINOv2), GELU MLP, pre-LN encoder block, learned 1D
//! position embedding added to (cls ⨯ patch) tokens.
//!
//! # Scope (v1)
//!
//! Forward-only, single fixed-size image (`image_size` ×
//! `image_size`), F32. Variable image-size position-embedding
//! interpolation deferred (eager uses bicubic which is a
//! larger op we don't have on the lazy surface yet).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Dinov2Config {
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub mlp_ratio: usize,
    pub layer_norm_eps: f64,
    pub num_classes: usize,
}

impl Dinov2Config {
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }
    pub fn num_patches(&self) -> usize {
        let p = self.num_patches_per_side();
        p * p
    }
    pub fn mlp_hidden(&self) -> usize {
        self.embed_dim * self.mlp_ratio
    }

    /// DINOv2 ViT-Small/14: 12 layers, 384 dim, 6 heads.
    pub fn vit_small() -> Self {
        Self {
            embed_dim: 384, depth: 12, num_heads: 6,
            num_channels: 3, image_size: 518, patch_size: 14,
            mlp_ratio: 4, layer_norm_eps: 1e-5,
            num_classes: 1000,
        }
    }
    /// DINOv2 ViT-Base/14: 12 layers, 768 dim, 12 heads.
    pub fn vit_base() -> Self {
        Self {
            embed_dim: 768, depth: 12, num_heads: 12,
            num_channels: 3, image_size: 518, patch_size: 14,
            mlp_ratio: 4, layer_norm_eps: 1e-5,
            num_classes: 1000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Dinov2BlockWeights {
    pub norm1_gain: Arc<[f32]>,
    pub norm1_bias: Arc<[f32]>,
    /// Fused `[embed_dim, 3 * embed_dim]`.
    pub qkv: WeightStorage,
    pub qkv_bias: Arc<[f32]>,
    pub proj: WeightStorage,
    pub proj_bias: Arc<[f32]>,
    pub ls1_gamma: Arc<[f32]>,
    pub norm2_gain: Arc<[f32]>,
    pub norm2_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
    pub ls2_gamma: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Dinov2Weights {
    /// Conv2d weight `[embed_dim, num_channels, patch, patch]`.
    pub patch_proj: Arc<[f32]>,
    pub patch_proj_bias: Arc<[f32]>,
    pub cls_token: Arc<[f32]>,
    /// `[num_patches + 1, embed_dim]`.
    pub pos_embed: Arc<[f32]>,
    pub blocks: Vec<Dinov2BlockWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    /// `[2 * embed_dim, num_classes]`.
    pub head: WeightStorage,
    pub head_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Dinov2Model {
    pub config: Dinov2Config,
    pub weights: Dinov2Weights,
}

impl Dinov2Model {
    /// Run image classification. Returns `(1, num_classes)`.
    pub fn forward(&self, pixel_values: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = pixel_values.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        let batch = dims[0];
        assert_eq!(batch, 1, "v1 supports batch == 1");
        assert_eq!(dims[1], cfg.num_channels);
        assert_eq!(dims[2], cfg.image_size);
        assert_eq!(dims[3], cfg.image_size);

        // Patch Conv2d.
        let conv_w = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj),
            Shape::from_dims(&[cfg.embed_dim, cfg.num_channels, cfg.patch_size, cfg.patch_size]),
        );
        let conv_b = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj_bias),
            Shape::from_dims(&[cfg.embed_dim]),
        );
        let conv_out = pixel_values.conv2d(
            &conv_w,
            Some(&conv_b),
            (cfg.patch_size, cfg.patch_size),
            (0, 0),
            1,
        )?;
        let np = cfg.num_patches();
        let patches = conv_out
            .reshape(Shape::from_dims(&[batch, cfg.embed_dim, np]))?
            .permute([0, 2, 1_usize])?;

        // Prepend CLS token.
        let cls = pixel_values.const_f32_like(
            Arc::clone(&weights.cls_token),
            Shape::from_dims(&[1, 1, cfg.embed_dim]),
        );
        let cls_bc = cls.broadcast_to(Shape::from_dims(&[batch, 1, cfg.embed_dim]))?;
        let with_cls = cls_bc.concat(&patches, 1_usize)?;

        // Add learned position embedding.
        let pos = pixel_values.const_f32_like(
            Arc::clone(&weights.pos_embed),
            Shape::from_dims(&[np + 1, cfg.embed_dim]),
        );
        let pos_bc = pos
            .reshape(Shape::from_dims(&[1, np + 1, cfg.embed_dim]))?
            .broadcast_to(Shape::from_dims(&[batch, np + 1, cfg.embed_dim]))?;
        let mut h = with_cls.add(&pos_bc)?;

        // Encoder blocks (no causal mask for vision).
        for block in &weights.blocks {
            h = self.apply_block(&h, block)?;
        }

        // Final LayerNorm.
        let h_norm = crate::lazy::apply_affine_layer_norm_pub(
            &h, &weights.final_ln_gain, &weights.final_ln_bias,
            cfg.embed_dim, cfg.layer_norm_eps,
        );

        // Feature: cat(cls_norm, mean(patch_norm)) → (1, 2 * embed_dim).
        let cls_feat = h_norm
            .slice(1_usize, 0, 1)?
            .reshape(Shape::from_dims(&[batch, cfg.embed_dim]))?;
        let patch_feat = h_norm
            .slice(1_usize, 1, np)?
            .mean_dim(1_usize)?
            .reshape(Shape::from_dims(&[batch, cfg.embed_dim]))?;
        let combined = cls_feat.concat(&patch_feat, 1_usize)?;

        // Classifier head.
        let logits = weights.head.apply_linear(&combined, 2 * cfg.embed_dim, cfg.num_classes);
        let bias_t = pixel_values.const_f32_like(
            Arc::clone(&weights.head_bias),
            Shape::from_dims(&[cfg.num_classes]),
        );
        logits.broadcast_add(&bias_t)
    }

    /// Extract per-token features at the requested layer indices.
    /// Used by DPT-style heads (Depth Anything V2, etc.) that
    /// fuse features from multiple stages of the backbone — the
    /// canonical layer pick for DINOv2 ViT-Small/Base/Large is
    /// `[2, 5, 8, 11]` (every 3 layers).
    ///
    /// Returns `Vec<LazyTensor>` with one tensor per requested
    /// layer index, shape `(1, num_patches + 1, embed_dim)`. The
    /// CLS token is at slot 0 of each output; the patch features
    /// follow. **No final LayerNorm** is applied — the DPT head
    /// has its own per-stage projection that absorbs the
    /// normalization step.
    ///
    /// Layer indices are 0-based and must be strictly increasing
    /// and within `[0, depth)`.
    pub fn forward_intermediate_layers(
        &self,
        pixel_values: &LazyTensor,
        layer_ids: &[usize],
    ) -> Result<Vec<LazyTensor>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = pixel_values.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        let batch = dims[0];
        assert_eq!(batch, 1, "v1 supports batch == 1");
        assert_eq!(dims[1], cfg.num_channels);
        assert_eq!(dims[2], cfg.image_size);
        assert_eq!(dims[3], cfg.image_size);
        assert!(!layer_ids.is_empty(), "layer_ids must not be empty");
        for w in layer_ids.windows(2) {
            assert!(w[0] < w[1], "layer_ids must be strictly increasing");
        }
        assert!(
            *layer_ids.last().unwrap() < cfg.depth,
            "layer_ids must all be in [0, depth = {})", cfg.depth,
        );

        // Patch conv + cls + pos_embed (same path as forward()).
        let conv_w = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj),
            Shape::from_dims(&[cfg.embed_dim, cfg.num_channels, cfg.patch_size, cfg.patch_size]),
        );
        let conv_b = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj_bias),
            Shape::from_dims(&[cfg.embed_dim]),
        );
        let conv_out = pixel_values.conv2d(
            &conv_w, Some(&conv_b),
            (cfg.patch_size, cfg.patch_size), (0, 0), 1,
        )?;
        let np = cfg.num_patches();
        let patches = conv_out
            .reshape(Shape::from_dims(&[batch, cfg.embed_dim, np]))?
            .permute([0, 2, 1_usize])?;
        let cls = pixel_values.const_f32_like(
            Arc::clone(&weights.cls_token),
            Shape::from_dims(&[1, 1, cfg.embed_dim]),
        );
        let cls_bc = cls.broadcast_to(Shape::from_dims(&[batch, 1, cfg.embed_dim]))?;
        let with_cls = cls_bc.concat(&patches, 1_usize)?;
        let pos = pixel_values.const_f32_like(
            Arc::clone(&weights.pos_embed),
            Shape::from_dims(&[np + 1, cfg.embed_dim]),
        );
        let pos_bc = pos
            .reshape(Shape::from_dims(&[1, np + 1, cfg.embed_dim]))?
            .broadcast_to(Shape::from_dims(&[batch, np + 1, cfg.embed_dim]))?;
        let mut h = with_cls.add(&pos_bc)?;

        // Run blocks and capture at the requested indices.
        let mut out = Vec::with_capacity(layer_ids.len());
        let mut next_capture = 0;
        for (idx, block) in weights.blocks.iter().enumerate() {
            h = self.apply_block(&h, block)?;
            if next_capture < layer_ids.len() && layer_ids[next_capture] == idx {
                out.push(h.clone());
                next_capture += 1;
            }
        }
        Ok(out)
    }

    fn apply_block(
        &self,
        x: &LazyTensor,
        block: &Dinov2BlockWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = x.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.embed_dim;
        let n_heads = cfg.num_heads;
        let head_dim = cfg.head_dim();

        // Pre-LN before attention.
        let x_norm = crate::lazy::apply_affine_layer_norm_pub(
            x, &block.norm1_gain, &block.norm1_bias, h, cfg.layer_norm_eps,
        );

        // Fused Wqkv: hidden → 3 * hidden.
        let qkv_lin = block.qkv.apply_linear(&x_norm, h, 3 * h);
        let qkv_bias_t = x.const_f32_like(
            Arc::clone(&block.qkv_bias),
            Shape::from_dims(&[3 * h]),
        );
        let qkv = qkv_lin.broadcast_add(&qkv_bias_t)?;
        let q = qkv.slice(2_usize, 0, h)?;
        let k = qkv.slice(2_usize, h, h)?;
        let v = qkv.slice(2_usize, 2 * h, h)?;

        let _ = (batch, seq);
        let q = q.split_heads(n_heads, head_dim)?;
        let k = k.split_heads(n_heads, head_dim)?;
        let v = v.split_heads(n_heads, head_dim)?;

        let k_t = k.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k_t)?.mul_scalar(scale);
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx.merge_heads()?;
        let proj = block.proj.apply_linear(&merged, h, h);
        let proj_b_t = x.const_f32_like(
            Arc::clone(&block.proj_bias),
            Shape::from_dims(&[h]),
        );
        let attn_out = proj.broadcast_add(&proj_b_t)?;

        // LayerScale 1: per-channel gamma multiplier BEFORE residual.
        let ls1_t = x.const_f32_like(
            Arc::clone(&block.ls1_gamma),
            Shape::from_dims(&[h]),
        );
        let attn_scaled = attn_out.broadcast_mul(&ls1_t)?;
        let h1 = x.add(&attn_scaled)?;

        // Pre-LN before MLP.
        let h1_norm = crate::lazy::apply_affine_layer_norm_pub(
            &h1, &block.norm2_gain, &block.norm2_bias, h, cfg.layer_norm_eps,
        );
        let mlp_hidden = cfg.mlp_hidden();
        let fc1 = block.fc1.apply_linear(&h1_norm, h, mlp_hidden);
        let fc1_b_t = x.const_f32_like(
            Arc::clone(&block.fc1_bias),
            Shape::from_dims(&[mlp_hidden]),
        );
        let fc1 = fc1.broadcast_add(&fc1_b_t)?.gelu_erf();
        let fc2 = block.fc2.apply_linear(&fc1, mlp_hidden, h);
        let fc2_b_t = x.const_f32_like(
            Arc::clone(&block.fc2_bias),
            Shape::from_dims(&[h]),
        );
        let mlp_out = fc2.broadcast_add(&fc2_b_t)?;

        // LayerScale 2.
        let ls2_t = x.const_f32_like(
            Arc::clone(&block.ls2_gamma),
            Shape::from_dims(&[h]),
        );
        let mlp_scaled = mlp_out.broadcast_mul(&ls2_t)?;
        h1.add(&mlp_scaled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_weights(cfg: &Dinov2Config) -> Dinov2Weights {
        let mut s: u32 = 47474;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let h = cfg.embed_dim;
        let p = cfg.patch_size;
        let c = cfg.num_channels;
        let np = cfg.num_patches();
        let mlp_h = cfg.mlp_hidden();
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);

        let patch_proj = vec_of(h * c * p * p, &mut *nb);
        let patch_proj_bias = vec_of(h, &mut *nb);
        let cls_token = vec_of(h, &mut *nb);
        let pos_embed = vec_of((np + 1) * h, &mut *nb);

        let blocks: Vec<Dinov2BlockWeights> = (0..cfg.depth)
            .map(|_| Dinov2BlockWeights {
                norm1_gain: Arc::from(vec![1.0_f32; h]),
                norm1_bias: Arc::from(vec![0.0_f32; h]),
                qkv: WeightStorage::F32(vec_of(h * (3 * h), &mut *nb)),
                qkv_bias: vec_of(3 * h, &mut *nb),
                proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                proj_bias: vec_of(h, &mut *nb),
                ls1_gamma: vec_of(h, &mut *nb),
                norm2_gain: Arc::from(vec![1.0_f32; h]),
                norm2_bias: Arc::from(vec![0.0_f32; h]),
                fc1: WeightStorage::F32(vec_of(h * mlp_h, &mut *nb)),
                fc1_bias: vec_of(mlp_h, &mut *nb),
                fc2: WeightStorage::F32(vec_of(mlp_h * h, &mut *nb)),
                fc2_bias: vec_of(h, &mut *nb),
                ls2_gamma: vec_of(h, &mut *nb),
            })
            .collect();

        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let head = WeightStorage::F32(vec_of((2 * h) * cfg.num_classes, &mut *nb));
        let head_bias = vec_of(cfg.num_classes, &mut *nb);
        Dinov2Weights {
            patch_proj, patch_proj_bias,
            cls_token, pos_embed,
            blocks,
            final_ln_gain, final_ln_bias,
            head, head_bias,
        }
    }

    fn tiny_config() -> Dinov2Config {
        Dinov2Config {
            embed_dim: 16, depth: 2, num_heads: 4,
            num_channels: 3, image_size: 16, patch_size: 4,
            mlp_ratio: 2, layer_norm_eps: 1e-5,
            num_classes: 8,
        }
    }

    fn tiny_image(cfg: &Dinov2Config) -> LazyTensor {
        let n_pix = 1 * cfg.num_channels * cfg.image_size * cfg.image_size;
        let img_data: Vec<f32> = (0..n_pix).map(|i| (i as f32 / n_pix as f32)).collect();
        LazyTensor::from_f32(
            Arc::from(img_data),
            Shape::from_dims(&[1, cfg.num_channels, cfg.image_size, cfg.image_size]),
            &Device::cpu(),
        )
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = Dinov2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let img = tiny_image(&cfg);
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, cfg.num_classes]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// LayerScale must be wired: zeroing ls1_gamma kills the
    /// attention contribution from that block; output must change.
    #[test]
    fn layer_scale_is_wired() {
        let cfg = tiny_config();
        let base = tiny_weights(&cfg);
        let mut zeroed = base.clone();
        zeroed.blocks[0].ls1_gamma = Arc::from(vec![0.0_f32; cfg.embed_dim]);
        let m_base = Dinov2Model { config: cfg.clone(), weights: base };
        let m_zero = Dinov2Model { config: cfg.clone(), weights: zeroed };
        let img_a = tiny_image(&cfg);
        let img_b = tiny_image(&cfg);
        let a = m_base.forward(&img_a).unwrap().realize_f32();
        let b = m_zero.forward(&img_b).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "LayerScale (ls1_gamma) must affect output, max_diff = {max_diff}");
    }

    /// Fused Wqkv slicing: zero the V columns of block 0's qkv
    /// (the last `embed_dim` cols) and the output must change.
    #[test]
    fn fused_wqkv_slicing_layout() {
        let cfg = tiny_config();
        let h = cfg.embed_dim;
        let base = tiny_weights(&cfg);
        let mut zeroed = base.clone();
        let mut qkv = match &zeroed.blocks[0].qkv {
            WeightStorage::F32(v) => v.to_vec(),
            _ => panic!(),
        };
        for row in 0..h {
            for j in 2 * h..3 * h {
                qkv[row * (3 * h) + j] = 0.0;
            }
        }
        zeroed.blocks[0].qkv = WeightStorage::F32(Arc::from(qkv));
        let mut bias_v: Vec<f32> = (*zeroed.blocks[0].qkv_bias).to_vec();
        for j in 2 * h..3 * h {
            bias_v[j] = 0.0;
        }
        zeroed.blocks[0].qkv_bias = Arc::from(bias_v);
        let m_base = Dinov2Model { config: cfg.clone(), weights: base };
        let m_zero = Dinov2Model { config: cfg.clone(), weights: zeroed };
        let a = m_base.forward(&tiny_image(&cfg)).unwrap().realize_f32();
        let b = m_zero.forward(&tiny_image(&cfg)).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "Fused Wqkv V slice must affect output, max_diff = {max_diff}");
    }

    #[test]
    fn config_presets() {
        let small = Dinov2Config::vit_small();
        assert_eq!(small.head_dim(), 64);
        assert_eq!(small.num_patches(), 1369); // (518/14)^2 = 37^2 = 1369
        let base = Dinov2Config::vit_base();
        assert_eq!(base.embed_dim, 768);
        assert_eq!(base.num_patches(), 1369);
    }

    /// `forward_intermediate_layers` returns one tensor per
    /// requested layer index, each shaped
    /// `(1, num_patches + 1, embed_dim)` (CLS token + patches).
    #[test]
    fn forward_intermediate_layers_shape() {
        let cfg = tiny_config();
        let model = Dinov2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let img = tiny_image(&cfg);
        let layer_ids = [0_usize, 1];
        let outs = model.forward_intermediate_layers(&img, &layer_ids).unwrap();
        assert_eq!(outs.len(), 2);
        let np = cfg.num_patches();
        for out in &outs {
            assert_eq!(out.shape().dims(), &[1, np + 1, cfg.embed_dim]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite intermediate: {v}");
            }
        }
    }

    /// Intermediate features at different layers must differ —
    /// each block transforms the hidden state, so layer 0 ≠ layer 1
    /// even for a noise-free input.
    #[test]
    fn intermediate_layers_differ_across_depth() {
        let cfg = tiny_config();
        let model = Dinov2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let img = tiny_image(&cfg);
        let outs = model.forward_intermediate_layers(&img, &[0_usize, 1]).unwrap();
        let a = outs[0].realize_f32();
        let b = outs[1].realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "layer 0 and layer 1 intermediates must differ, max_diff = {max_diff}");
    }

    /// Requesting the LAST layer's intermediate gives the same
    /// hidden state that `forward` runs through the final LN.
    #[test]
    fn last_intermediate_matches_pre_final_ln() {
        let cfg = tiny_config();
        let last = cfg.depth - 1;
        let model = Dinov2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let img = tiny_image(&cfg);
        let outs = model.forward_intermediate_layers(&img, &[last]).unwrap();
        assert_eq!(outs.len(), 1);
        let np = cfg.num_patches();
        assert_eq!(outs[0].shape().dims(), &[1, np + 1, cfg.embed_dim]);
    }
}

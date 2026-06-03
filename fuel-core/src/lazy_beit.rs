//! BEiT (BERT pre-training of Image Transformers) ported to the
//! lazy-graph API.
//!
//! Bao et al. 2021. ViT variant pre-trained with masked image
//! modeling. Distinguishing features from plain ViT (see
//! [`crate::lazy_vit`]) and DINOv2 (see [`crate::lazy_dinov2`]):
//!
//!   1. **Relative position bias.** A learned
//!      `[num_relative_distance, num_heads]` bias table is
//!      indexed by a fixed `relative_position_index` of shape
//!      `[NB_TOKENS, NB_TOKENS]`, then added to the attention
//!      scores. NO absolute position embedding.
//!   2. **Fused Wqkv** (like DINOv2): one linear `hidden → 3 *
//!      hidden` produces Q, K, V together.
//!   3. **LayerScale** (like DINOv2): per-channel `gamma`
//!      multiplier on attn output and MLP output BEFORE the
//!      residual add.
//!   4. **CLS-token prepended**, no learned absolute pos embed.
//!   5. **Mean-over-patches feature** (NOT CLS-pooled): final
//!      classifier head sees `mean(patches)` → LayerNorm →
//!      classifier.
//!
//! The relative-position-index construction:
//!
//!   - For each pair of patch tokens `(p_a, p_b)`, the bucket
//!     is `((row_a - row_b) + W - 1) * (2W - 1) + ((col_a -
//!     col_b) + W - 1)` where W = num_patches_per_side.
//!     This covers `(2W - 1) ** 2` distinct buckets.
//!   - Three additional buckets handle CLS↔token (and CLS↔CLS)
//!     pairs: indices `num_buckets - 3, num_buckets - 2,
//!     num_buckets - 1`.
//!
//! v1 precomputes the index table once at model construction
//! (it's a function of `num_patches_per_side` only) and stores
//! it as `Arc<[u32]>`.
//!
//! # Scope (v1)
//!
//! Forward-only, single fixed-size image (`image_size` ×
//! `image_size`), F32.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct BeitConfig {
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub mlp_ratio: usize,
    pub layer_norm_eps: f64,
    pub num_classes: usize,
    pub qkv_bias: bool,
    pub proj_bias: bool,
}

impl BeitConfig {
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
    pub fn num_tokens(&self) -> usize {
        self.num_patches() + 1
    }
    pub fn num_relative_distance(&self) -> usize {
        let w = self.num_patches_per_side();
        (2 * w - 1) * (2 * w - 1) + 3
    }

    /// BEiT ViT-Base/16 at 384×384.
    pub fn vit_base() -> Self {
        Self {
            embed_dim: 768, depth: 12, num_heads: 12,
            num_channels: 3, image_size: 384, patch_size: 16,
            mlp_ratio: 4, layer_norm_eps: 1e-6,
            num_classes: 1000, qkv_bias: true, proj_bias: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BeitBlockWeights {
    pub norm1_gain: Arc<[f32]>,
    pub norm1_bias: Arc<[f32]>,
    /// Fused `[embed_dim, 3 * embed_dim]`.
    pub qkv: WeightStorage,
    pub qkv_bias: Option<Arc<[f32]>>,
    pub proj: WeightStorage,
    pub proj_bias: Option<Arc<[f32]>>,
    pub ls1_gamma: Arc<[f32]>,
    /// `[num_relative_distance, num_heads]`.
    pub relative_position_bias_table: Arc<[f32]>,
    pub norm2_gain: Arc<[f32]>,
    pub norm2_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
    pub ls2_gamma: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct BeitWeights {
    /// Conv2d weight `[embed_dim, num_channels, patch, patch]`.
    pub patch_proj: Arc<[f32]>,
    pub patch_proj_bias: Arc<[f32]>,
    pub cls_token: Arc<[f32]>,
    pub blocks: Vec<BeitBlockWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub head: WeightStorage,
    pub head_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct BeitModel {
    pub config: BeitConfig,
    pub weights: BeitWeights,
    /// Precomputed `[NB_TOKENS * NB_TOKENS]` flat index of
    /// bucket numbers — function of `num_patches_per_side`
    /// only. Built once at construction.
    relative_position_index: Arc<[u32]>,
}

impl BeitModel {
    pub fn new(config: BeitConfig, weights: BeitWeights) -> Self {
        let relative_position_index = Arc::from(
            build_relative_position_index(config.num_patches_per_side()),
        );
        Self { config, weights, relative_position_index }
    }

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
        let mut h = cls_bc.concat(&patches, 1_usize)?;

        // Encoder blocks with relative position bias.
        for block in &weights.blocks {
            h = self.apply_block(&h, block, pixel_values)?;
        }

        // Mean over patch tokens (exclude CLS at position 0).
        let patch_mean = h
            .slice(1_usize, 1, np)?
            .mean_dim(1_usize)?
            .reshape(Shape::from_dims(&[batch, cfg.embed_dim]))?;
        // Final LayerNorm on the pooled vector.
        let pooled_ln = crate::lazy::apply_affine_layer_norm_pub(
            &patch_mean, &weights.final_ln_gain, &weights.final_ln_bias,
            cfg.embed_dim, cfg.layer_norm_eps,
        );
        // Classifier.
        let logits = weights.head.apply_linear(&pooled_ln, cfg.embed_dim, cfg.num_classes);
        let bias_t = pixel_values.const_f32_like(
            Arc::clone(&weights.head_bias),
            Shape::from_dims(&[cfg.num_classes]),
        );
        logits.broadcast_add(&bias_t)
    }

    /// Extract per-token features at the requested layer
    /// indices. Output shape per layer:
    /// `(1, num_patches + 1, embed_dim)` — CLS at slot 0,
    /// patches follow. **No final LayerNorm** applied — the
    /// final LN sits inside the classifier head, so the raw
    /// post-block hidden state is what gets returned (matches
    /// the convention used by the other ViT-shape hooks).
    ///
    /// BEiT specifics preserved:
    ///   - **Learned relative position bias** is built once
    ///     per layer block (the bias table is per-block in
    ///     BEiT) and applied inside the attention path —
    ///     same path the public `forward` takes.
    ///   - No absolute / learned 1D position embedding.
    ///
    /// Layer-id contract: 0-based, strictly increasing, all in
    /// `[0, num_blocks)`. Mirrors the ViT, DINOv2, DINOv2-reg4,
    /// SigLIP, and CLIP hooks.
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
        let depth = weights.blocks.len();
        assert!(
            *layer_ids.last().unwrap() < depth,
            "layer_ids must all be in [0, num_blocks = {depth})",
        );

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
        let mut h = cls_bc.concat(&patches, 1_usize)?;

        let mut out = Vec::with_capacity(layer_ids.len());
        let mut next_capture = 0;
        for (idx, block) in weights.blocks.iter().enumerate() {
            h = self.apply_block(&h, block, pixel_values)?;
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
        block: &BeitBlockWeights,
        anchor: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = x.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let seq = dims[1]; // num_tokens
        let h = cfg.embed_dim;
        let n_heads = cfg.num_heads;
        let head_dim = cfg.head_dim();

        // Pre-LN.
        let x_norm = crate::lazy::apply_affine_layer_norm_pub(
            x, &block.norm1_gain, &block.norm1_bias, h, cfg.layer_norm_eps,
        );

        // Fused Wqkv.
        let qkv_lin = block.qkv.apply_linear(&x_norm, h, 3 * h);
        let qkv = match &block.qkv_bias {
            None => qkv_lin,
            Some(b) => {
                let bt = anchor.const_f32_like(
                    Arc::clone(b),
                    Shape::from_dims(&[3 * h]),
                );
                qkv_lin.broadcast_add(&bt)?
            }
        };
        let q = qkv.slice(2_usize, 0, h)?;
        let k = qkv.slice(2_usize, h, h)?;
        let v = qkv.slice(2_usize, 2 * h, h)?;

        let q = q
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let k = k
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let v = v
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;

        // Relative position bias: look up indices in the
        // `[num_relative_distance, num_heads]` table, reshape
        // to `[1, num_heads, seq, seq]` to broadcast over batch.
        let rel_bias = self.build_relative_position_bias(
            anchor, &block.relative_position_bias_table,
            seq, n_heads,
        )?;

        let k_t = k.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k_t)?.mul_scalar(scale);
        let scores = scores.broadcast_add(&rel_bias)?;
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx
            .permute([0, 2, 1, 3_usize])?
            .reshape(Shape::from_dims(&[batch, seq, h]))?;
        let proj = block.proj.apply_linear(&merged, h, h);
        let attn_out = match &block.proj_bias {
            None => proj,
            Some(b) => {
                let bt = anchor.const_f32_like(
                    Arc::clone(b),
                    Shape::from_dims(&[h]),
                );
                proj.broadcast_add(&bt)?
            }
        };

        // LayerScale 1 + residual.
        let ls1_t = anchor.const_f32_like(
            Arc::clone(&block.ls1_gamma),
            Shape::from_dims(&[h]),
        );
        let h1 = x.add(&attn_out.broadcast_mul(&ls1_t)?)?;

        // Pre-MLP norm + MLP + LayerScale 2 + residual.
        let h1_norm = crate::lazy::apply_affine_layer_norm_pub(
            &h1, &block.norm2_gain, &block.norm2_bias, h, cfg.layer_norm_eps,
        );
        let mlp_h = cfg.embed_dim * cfg.mlp_ratio;
        let fc1 = block.fc1.apply_linear(&h1_norm, h, mlp_h);
        let fc1_bias_t = anchor.const_f32_like(
            Arc::clone(&block.fc1_bias),
            Shape::from_dims(&[mlp_h]),
        );
        let fc1 = fc1.broadcast_add(&fc1_bias_t)?.gelu_erf();
        let fc2 = block.fc2.apply_linear(&fc1, mlp_h, h);
        let fc2_bias_t = anchor.const_f32_like(
            Arc::clone(&block.fc2_bias),
            Shape::from_dims(&[h]),
        );
        let mlp_out = fc2.broadcast_add(&fc2_bias_t)?;
        let ls2_t = anchor.const_f32_like(
            Arc::clone(&block.ls2_gamma),
            Shape::from_dims(&[h]),
        );
        h1.add(&mlp_out.broadcast_mul(&ls2_t)?)
    }

    fn build_relative_position_bias(
        &self,
        anchor: &LazyTensor,
        bias_table: &Arc<[f32]>,
        seq: usize,
        n_heads: usize,
    ) -> Result<LazyTensor> {
        assert_eq!(seq, self.config.num_tokens());
        // Look up each (i, j)'s bucket and pull the n_heads vector.
        // Build the bias tensor directly as a const of shape
        // [1, n_heads, seq, seq].
        let nrd = self.config.num_relative_distance();
        assert_eq!(bias_table.len(), nrd * n_heads);
        let idx = &self.relative_position_index;
        assert_eq!(idx.len(), seq * seq);
        let mut bias_data = vec![0.0_f32; n_heads * seq * seq];
        for q in 0..seq {
            for kv in 0..seq {
                let bucket = idx[q * seq + kv] as usize;
                for h in 0..n_heads {
                    let src = bucket * n_heads + h;
                    let dst = h * seq * seq + q * seq + kv;
                    bias_data[dst] = bias_table[src];
                }
            }
        }
        let bias = anchor.const_f32_like(
            Arc::from(bias_data),
            Shape::from_dims(&[1, n_heads, seq, seq]),
        );
        Ok(bias)
    }
}

/// Build the `[NB_TOKENS, NB_TOKENS]` flat bucket index for BEiT.
///
/// `NB_TOKENS = W * W + 1` where W = `num_patches_per_side`.
/// - Patch-to-patch pairs at `(p_a, p_b) = (r_a, c_a) ↔
///   (r_b, c_b)`: bucket = `((r_a - r_b) + W - 1) * (2W - 1)
///   + ((c_a - c_b) + W - 1)`.
/// - CLS-to-patch, patch-to-CLS, CLS-to-CLS: the last three
///   buckets `(nrd - 3, nrd - 2, nrd - 1)`.
fn build_relative_position_index(num_patches_per_side: usize) -> Vec<u32> {
    let w = num_patches_per_side;
    let w_area = w * w;
    let nb = w_area + 1;
    let two_w_m1 = (2 * w - 1) as u32;
    let nrd = (2 * w - 1) * (2 * w - 1) + 3;
    let mut idx = vec![0_u32; nb * nb];

    // Patch-to-patch (rows 1..=w_area, cols 1..=w_area).
    for a in 0..w_area {
        let r_a = a / w;
        let c_a = a % w;
        for b in 0..w_area {
            let r_b = b / w;
            let c_b = b % w;
            let dr = (r_a as i32 - r_b as i32 + (w as i32 - 1)) as u32;
            let dc = (c_a as i32 - c_b as i32 + (w as i32 - 1)) as u32;
            let bucket = dr * two_w_m1 + dc;
            idx[(a + 1) * nb + (b + 1)] = bucket;
        }
    }
    // CLS-to-patch (row 0, cols 1..=w_area): bucket = nrd - 3.
    let cls_to_patch = (nrd - 3) as u32;
    for j in 1..nb {
        idx[0 * nb + j] = cls_to_patch;
    }
    // Patch-to-CLS (rows 1..=w_area, col 0): bucket = nrd - 2.
    let patch_to_cls = (nrd - 2) as u32;
    for i in 1..nb {
        idx[i * nb + 0] = patch_to_cls;
    }
    // CLS-to-CLS (0, 0): bucket = nrd - 1.
    idx[0] = (nrd - 1) as u32;
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_weights(cfg: &BeitConfig) -> BeitWeights {
        let mut s: u32 = 42424;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let h = cfg.embed_dim;
        let p = cfg.patch_size;
        let c = cfg.num_channels;
        let mlp_h = cfg.embed_dim * cfg.mlp_ratio;
        let nrd = cfg.num_relative_distance();

        let patch_proj = vec_of(h * c * p * p, &mut *nb);
        let patch_proj_bias = vec_of(h, &mut *nb);
        let cls_token = vec_of(h, &mut *nb);

        let blocks: Vec<BeitBlockWeights> = (0..cfg.depth).map(|_| BeitBlockWeights {
            norm1_gain: Arc::from(vec![1.0_f32; h]),
            norm1_bias: Arc::from(vec![0.0_f32; h]),
            qkv: WeightStorage::F32(vec_of(h * (3 * h), &mut *nb)),
            qkv_bias: if cfg.qkv_bias { Some(vec_of(3 * h, &mut *nb)) } else { None },
            proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            proj_bias: if cfg.proj_bias { Some(vec_of(h, &mut *nb)) } else { None },
            ls1_gamma: vec_of(h, &mut *nb),
            relative_position_bias_table: vec_of(nrd * cfg.num_heads, &mut *nb),
            norm2_gain: Arc::from(vec![1.0_f32; h]),
            norm2_bias: Arc::from(vec![0.0_f32; h]),
            fc1: WeightStorage::F32(vec_of(h * mlp_h, &mut *nb)),
            fc1_bias: vec_of(mlp_h, &mut *nb),
            fc2: WeightStorage::F32(vec_of(mlp_h * h, &mut *nb)),
            fc2_bias: vec_of(h, &mut *nb),
            ls2_gamma: vec_of(h, &mut *nb),
        }).collect();

        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let head = WeightStorage::F32(vec_of(h * cfg.num_classes, &mut *nb));
        let head_bias = vec_of(cfg.num_classes, &mut *nb);
        BeitWeights {
            patch_proj, patch_proj_bias,
            cls_token,
            blocks,
            final_ln_gain, final_ln_bias,
            head, head_bias,
        }
    }

    fn tiny_config() -> BeitConfig {
        BeitConfig {
            embed_dim: 16,
            depth: 2,
            num_heads: 4,
            num_channels: 3,
            image_size: 16,
            patch_size: 4,
            mlp_ratio: 2,
            layer_norm_eps: 1e-6,
            num_classes: 8,
            qkv_bias: true,
            proj_bias: true,
        }
    }

    fn tiny_image(cfg: &BeitConfig) -> LazyTensor {
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
        let model = BeitModel::new(cfg.clone(), tiny_weights(&cfg));
        let img = tiny_image(&cfg);
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, cfg.num_classes]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// Relative position bias is wired: zeroing the bias table
    /// in block 0 changes the output.
    #[test]
    fn rel_pos_bias_is_wired() {
        let cfg = tiny_config();
        let base = tiny_weights(&cfg);
        let mut zeroed = base.clone();
        let nrd = cfg.num_relative_distance();
        zeroed.blocks[0].relative_position_bias_table =
            Arc::from(vec![0.0_f32; nrd * cfg.num_heads]);
        let m_base = BeitModel::new(cfg.clone(), base);
        let m_zero = BeitModel::new(cfg.clone(), zeroed);
        let img_a = tiny_image(&cfg);
        let img_b = tiny_image(&cfg);
        let a = m_base.forward(&img_a).unwrap().realize_f32();
        let b = m_zero.forward(&img_b).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        // Tiny weights (∈ [-0.025, 0.025]) → bias contribution
        // is small. Just require it to be measurable.
        assert!(max_diff > 1e-8,
            "relative position bias must affect output, max_diff = {max_diff}");
    }

    /// The relative position index calculation is symmetric in
    /// the patch-to-patch quadrant and the same patch pair gets
    /// the same bucket. CLS↔* pairs get the special buckets.
    #[test]
    fn relative_position_index_structure() {
        let idx = build_relative_position_index(2);
        // 2×2 grid + 1 CLS = 5 tokens. NB = 5, idx is 25 entries.
        assert_eq!(idx.len(), 25);
        let nrd = (2 * 2 - 1) * (2 * 2 - 1) + 3; // = 12
        // CLS-CLS (position 0, 0).
        assert_eq!(idx[0], (nrd - 1) as u32);
        // CLS-to-patch (row 0, col 1..5).
        for j in 1..5 {
            assert_eq!(idx[j], (nrd - 3) as u32);
        }
        // Patch-to-CLS (col 0, row 1..5).
        for i in 1..5 {
            assert_eq!(idx[i * 5], (nrd - 2) as u32);
        }
        // Same-patch pairs (positions 1, 2, 3, 4 → diag of the
        // patch-to-patch quadrant) all have bucket
        // = (W - 1) * (2W - 1) + (W - 1) = 1 * 3 + 1 = 4.
        let same_bucket: u32 = (1 * 3 + 1) as u32;
        for k in 1..=4 {
            assert_eq!(idx[k * 5 + k], same_bucket);
        }
    }

    /// `forward_intermediate_layers` returns one tensor per
    /// requested layer index, each shaped
    /// `(1, num_patches + 1, embed_dim)`. CLS at slot 0,
    /// patches follow. Mirrors the ViT-shape vision hooks.
    #[test]
    fn forward_intermediate_layers_shape() {
        let cfg = tiny_config();
        let model = BeitModel::new(cfg.clone(), tiny_weights(&cfg));
        let img = tiny_image(&cfg);
        let outs = model.forward_intermediate_layers(&img, &[0_usize, 1]).unwrap();
        assert_eq!(outs.len(), 2);
        let np = cfg.num_patches();
        for out in &outs {
            assert_eq!(out.shape().dims(), &[1, np + 1, cfg.embed_dim]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite intermediate: {v}");
            }
        }
        let a = outs[0].realize_f32();
        let b = outs[1].realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "layer 0 and layer 1 intermediates must differ, max_diff = {max_diff}");
    }
}

//! EfficientViT-MSRA — lazy port.
//!
//! Pipeline: image → stem (4 strided conv-BN-ReLU) → 3 stages
//! (each = optional downsample + N blocks) → global mean pool
//! → optional BN + linear classifier.
//!
//! Each block: dw0 → ffn0 → cga_attn → dw1 → ffn1, with residual
//! after each sub-module. The attention is Cascaded Group
//! Attention (CGA): channels split into `heads` groups; each
//! head's input is the cumulative sum of all previous heads' +
//! its own slice; each head runs through a per-head pointwise
//! conv producing (Q, K, V) with `key_dim` for Q/K and
//! `c_in/heads` for V; Q is spatially smoothed by a depthwise
//! conv (per-head kernel size); standard scaled-softmax attention;
//! per-head outputs concatenated then ReLU + final 1×1 proj.
//!
//! Windowing: when `stage_resolutions[stage] > 7`, the input is
//! tiled into 7×7 windows before attention and untiled after.
//! v1 supports both windowed and non-windowed stages.
//!
//! v1 scope:
//!   - F32 weights and activations.
//!   - `batch == 1`.
//!   - BatchNorm is fused-affine (inference mode).
//!   - Forward-only (no autograd in scope).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_convmixer::BatchNormParams;
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

/// EfficientViT config. Channels, blocks-per-stage, heads-per-stage,
/// and per-head depthwise kernel sizes match the timm MSRA
/// reference. `stage_resolutions` records the *expected* spatial
/// resolution at each stage's attention block — drives the 7×7
/// windowing decision. Default presets compute this from the
/// standard `image_size = 224`.
#[derive(Debug, Clone, PartialEq)]
pub struct EfficientVitConfig {
    pub channels: [usize; 3],
    pub blocks: [usize; 3],
    pub heads: [usize; 3],
    /// Per-head depthwise kernel sizes (one entry per head, max
    /// across stages).
    pub kernels: Vec<usize>,
    /// Attention pre-windowing target resolution per stage. The
    /// standard image_size=224 path gives [14, 7, 4]; v1 lets
    /// callers override for non-standard inputs.
    pub stage_resolutions: [usize; 3],
    pub key_dim: usize,
}

impl EfficientVitConfig {
    /// EfficientViT-MSRA M0 (smallest variant). image_size=224.
    pub fn m0() -> Self {
        Self {
            channels: [64, 128, 192], blocks: [1, 2, 3],
            heads: [4, 4, 4], kernels: vec![5, 5, 5, 5],
            stage_resolutions: [14, 7, 4], key_dim: 16,
        }
    }
    /// EfficientViT-MSRA M1.
    pub fn m1() -> Self {
        Self {
            channels: [128, 144, 192], blocks: [1, 2, 3],
            heads: [2, 3, 3], kernels: vec![7, 5, 3, 3],
            stage_resolutions: [14, 7, 4], key_dim: 16,
        }
    }
    /// EfficientViT-MSRA M2.
    pub fn m2() -> Self {
        Self {
            channels: [128, 192, 224], blocks: [1, 2, 3],
            heads: [4, 3, 2], kernels: vec![7, 5, 3, 3],
            stage_resolutions: [14, 7, 4], key_dim: 16,
        }
    }
}

// ---- Weight structures ------------------------------------------------------

/// Single Conv+BN+ReLU stem block (4 of these stack in the stem).
#[derive(Debug, Clone)]
pub struct ConvBnWeights {
    /// `[c_out, c_in / groups, k, k]`.
    pub conv_w: Arc<[f32]>,
    pub bn: BatchNormParams,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub groups: usize,
}

#[derive(Debug, Clone)]
pub struct StemWeights {
    pub conv1: ConvBnWeights,
    pub conv2: ConvBnWeights,
    pub conv3: ConvBnWeights,
    pub conv4: ConvBnWeights,
}

/// Conv2d with bias (1×1 fc1/fc2 inside SE block).
#[derive(Debug, Clone)]
pub struct Conv1x1BiasWeights {
    pub w: Arc<[f32]>,
    pub b: Arc<[f32]>,
    pub c_in: usize,
    pub c_out: usize,
}

#[derive(Debug, Clone)]
pub struct SeWeights {
    pub fc1: Conv1x1BiasWeights,
    pub fc2: Conv1x1BiasWeights,
}

/// Conv-MLP block: pointwise → ReLU → pointwise, both BN-fused.
#[derive(Debug, Clone)]
pub struct ConvMlpWeights {
    pub pw1: ConvBnWeights,
    pub pw2: ConvBnWeights,
}

#[derive(Debug, Clone)]
pub struct PatchMergeWeights {
    pub conv1: ConvBnWeights,
    pub conv2: ConvBnWeights,
    pub conv3: ConvBnWeights,
    pub se: SeWeights,
}

#[derive(Debug, Clone)]
pub struct ResBlockWeights {
    pub dw: ConvBnWeights,
    pub mlp: ConvMlpWeights,
}

#[derive(Debug, Clone)]
pub struct DownsampleWeights {
    pub res1: ResBlockWeights,
    pub patchmerge: PatchMergeWeights,
    pub res2: ResBlockWeights,
}

/// Per-head CGA weights: pointwise from `c_in/heads` to
/// `c_in/heads + 2*key_dim`, plus a depthwise on the Q channels.
#[derive(Debug, Clone)]
pub struct CgaHeadWeights {
    pub qkv: ConvBnWeights,
    pub dw_q: ConvBnWeights,
}

#[derive(Debug, Clone)]
pub struct CgaWeights {
    pub heads: Vec<CgaHeadWeights>,
    pub proj: ConvBnWeights,
}

#[derive(Debug, Clone)]
pub struct EfficientVitBlockWeights {
    pub dw0: ConvBnWeights,
    pub ffn0: ConvMlpWeights,
    pub attn: CgaWeights,
    pub dw1: ConvBnWeights,
    pub ffn1: ConvMlpWeights,
}

#[derive(Debug, Clone)]
pub struct EfficientVitStageWeights {
    pub downsample: Option<DownsampleWeights>,
    pub blocks: Vec<EfficientVitBlockWeights>,
}

#[derive(Debug, Clone)]
pub struct EfficientVitWeights {
    pub stem: StemWeights,
    pub stages: [EfficientVitStageWeights; 3],
    /// Classification head: BN over channels[2] + linear.
    /// None means no classifier (returns pooled features).
    pub head: Option<(BatchNormParams, WeightStorage, Arc<[f32]>)>,
}

#[derive(Debug, Clone)]
pub struct EfficientVitModel {
    pub config: EfficientVitConfig,
    pub weights: EfficientVitWeights,
}

// ---- Forward ---------------------------------------------------------------

impl EfficientVitModel {
    /// Run inference on `image` of shape `(1, 3, H, W)`. Returns
    /// classifier logits when `weights.head` is `Some`, else
    /// pooled features `(1, channels[2])`.
    pub fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = image.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4, "image must be rank 4 [N, 3, H, W]");
        assert_eq!(dims[1], 3, "image must have 3 input channels");

        let mut x = self.run_stem(image)?;
        for (si, stage_w) in self.weights.stages.iter().enumerate() {
            if let Some(ds) = &stage_w.downsample {
                x = apply_downsample(&x, ds, image)?;
            }
            for block in &stage_w.blocks {
                x = apply_block(&x, block, cfg, si, image)?;
            }
        }
        // Global mean over (H, W).
        let pooled_w = x.mean_dim(3_usize)?;
        let pooled = pooled_w.mean_dim(2_usize)?;
        match &self.weights.head {
            None => Ok(pooled),
            Some((bn, lin_w, lin_b)) => {
                let c = cfg.channels[2];
                // BN on (1, C) — promote to (1, C, 1, 1), apply, squeeze back.
                let reshaped = pooled.reshape(Shape::from_dims(&[1, c, 1, 1]))?;
                let bn_out = apply_bn(&reshaped, bn, c)?;
                let flat = bn_out.reshape(Shape::from_dims(&[1, c]))?;
                let n_out = lin_b.len();
                let logits = lin_w.apply_linear(&flat, c, n_out);
                let bias = image.const_f32_like(
                    Arc::clone(lin_b), Shape::from_dims(&[n_out]),
                );
                logits.broadcast_add(&bias)
            }
        }
    }

    /// Run the backbone (stem + 3 stages) and return the channels-
    /// first feature map BEFORE global mean pool and the classifier.
    pub fn forward_features(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = image.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        assert_eq!(dims[1], 3);
        let mut x = self.run_stem(image)?;
        for (si, stage_w) in self.weights.stages.iter().enumerate() {
            if let Some(ds) = &stage_w.downsample {
                x = apply_downsample(&x, ds, image)?;
            }
            for block in &stage_w.blocks {
                x = apply_block(&x, block, cfg, si, image)?;
            }
        }
        Ok(x)
    }

    fn run_stem(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let stem = &self.weights.stem;
        let x = apply_conv_bn(image, &stem.conv1, image)?.relu();
        let x = apply_conv_bn(&x, &stem.conv2, image)?.relu();
        let x = apply_conv_bn(&x, &stem.conv3, image)?.relu();
        // Final stem conv has NO trailing ReLU (eager `efficientvit_stem`
        // chains relu only between conv1..3, not after conv4).
        apply_conv_bn(&x, &stem.conv4, image)
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
    x: &LazyTensor, c: &ConvBnWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let w = anchor.const_f32_like(
        Arc::clone(&c.conv_w),
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

fn apply_conv1x1_bias(
    x: &LazyTensor, c: &Conv1x1BiasWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let w = anchor.const_f32_like(
        Arc::clone(&c.w), Shape::from_dims(&[c.c_out, c.c_in, 1, 1]),
    );
    let conv = x.conv2d(&w, None, (1, 1), (0, 0), 1)?;
    let bias = anchor
        .const_f32_like(Arc::clone(&c.b), Shape::from_dims(&[c.c_out]))
        .reshape(Shape::from_dims(&[1, c.c_out, 1, 1]))?;
    conv.broadcast_add(&bias)
}

fn apply_conv_mlp(
    x: &LazyTensor, m: &ConvMlpWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let h = apply_conv_bn(x, &m.pw1, anchor)?.relu();
    apply_conv_bn(&h, &m.pw2, anchor)
}

fn apply_se(
    x: &LazyTensor, se: &SeWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    // Mean over (H, W) keeping dims: reshape from (B, C, H, W) →
    // (B, C, 1, 1).
    let dims = x.shape();
    let dims = dims.dims();
    let c = dims[1];
    let pooled_w = x.mean_dim(3_usize)?;
    let pooled = pooled_w.mean_dim(2_usize)?;
    let pooled = pooled.reshape(Shape::from_dims(&[dims[0], c, 1, 1]))?;
    let g = apply_conv1x1_bias(&pooled, &se.fc1, anchor)?.relu();
    let g = apply_conv1x1_bias(&g, &se.fc2, anchor)?.sigmoid();
    let g_b = g.broadcast_to(Shape::from_dims(dims))?;
    x.mul(&g_b)
}

fn apply_patchmerge(
    x: &LazyTensor, p: &PatchMergeWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let x = apply_conv_bn(x, &p.conv1, anchor)?.relu();
    let x = apply_conv_bn(&x, &p.conv2, anchor)?.relu();
    let x = apply_se(&x, &p.se, anchor)?;
    apply_conv_bn(&x, &p.conv3, anchor)
}

fn apply_res_block(
    x: &LazyTensor, r: &ResBlockWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let y = apply_conv_bn(x, &r.dw, anchor)?;
    let x = x.add(&y)?;
    let y = apply_conv_mlp(&x, &r.mlp, anchor)?;
    x.add(&y)
}

fn apply_downsample(
    x: &LazyTensor, d: &DownsampleWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let x = apply_res_block(x, &d.res1, anchor)?;
    let x = apply_patchmerge(&x, &d.patchmerge, anchor)?;
    apply_res_block(&x, &d.res2, anchor)
}

fn apply_block(
    x: &LazyTensor, b: &EfficientVitBlockWeights,
    cfg: &EfficientVitConfig, stage: usize, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let y = apply_conv_bn(x, &b.dw0, anchor)?;
    let x = x.add(&y)?;
    let y = apply_conv_mlp(&x, &b.ffn0, anchor)?;
    let x = x.add(&y)?;
    let y = apply_cga_attn(&x, &b.attn, cfg, stage, anchor)?;
    let x = x.add(&y)?;
    let y = apply_conv_bn(&x, &b.dw1, anchor)?;
    let x = x.add(&y)?;
    let y = apply_conv_mlp(&x, &b.ffn1, anchor)?;
    x.add(&y)
}

/// CGA + optional 7×7 windowing.
fn apply_cga_attn(
    x: &LazyTensor, w: &CgaWeights,
    cfg: &EfficientVitConfig, stage: usize, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let win_res = 7;
    let need_windowing = cfg.stage_resolutions[stage] > win_res;
    if !need_windowing {
        return cga_core(x, w, cfg, anchor);
    }
    // Windowing path: pad to multiple of win_res, tile, run, untile.
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0];
    let c = dims[1];
    let h = dims[2];
    let w_ = dims[3];
    let pad_b = (win_res - h % win_res) % win_res;
    let pad_r = (win_res - w_ % win_res) % win_res;
    let ph = h + pad_b;
    let pw = w_ + pad_r;
    let nh = ph / win_res;
    let nw = pw / win_res;
    // Permute to NHWC for padding then back.
    let xs = x.permute([0, 2, 3, 1_usize])?;
    // Pad along H (dim 1) and W (dim 2).
    let xs = pad_dim_with_zeros(&xs, 2, pad_r)?;
    let xs = pad_dim_with_zeros(&xs, 1, pad_b)?;
    // (B, nh, win, nw, win, C)
    let xs = xs.reshape(Shape::from_dims(&[b, nh, win_res, nw, win_res, c]))?;
    // Transpose dims 2 and 3 → (B, nh, nw, win, win, C)
    let xs = xs.permute([0, 1, 3, 2, 4, 5_usize])?;
    // (B * nh * nw, win, win, C) → permute → (B*nh*nw, C, win, win)
    let xs = xs
        .reshape(Shape::from_dims(&[b * nh * nw, win_res, win_res, c]))?
        .permute([0, 3, 1, 2_usize])?;
    let ys = cga_core(&xs, w, cfg, anchor)?;
    // Reverse the tiling: (B*nh*nw, C, win, win) → (B, nh, nw, win, win, C)
    let ys = ys.permute([0, 2, 3, 1_usize])?;
    let ys = ys.reshape(Shape::from_dims(&[b, nh, nw, win_res, win_res, c]))?;
    let ys = ys.permute([0, 1, 3, 2, 4, 5_usize])?;
    let ys = ys.reshape(Shape::from_dims(&[b, ph, pw, c]))?;
    // Crop back to (h, w) — narrow along H and W.
    let ys = ys.narrow(1_usize, 0, h)?;
    let ys = ys.narrow(2_usize, 0, w_)?;
    // NHWC → NCHW.
    Ok(ys.permute([0, 3, 1, 2_usize])?)
}

/// CGA forward without windowing — assumes input fits as-is.
fn cga_core(
    x: &LazyTensor, w: &CgaWeights, cfg: &EfficientVitConfig,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0];
    let c = dims[1];
    let h = dims[2];
    let w_ = dims[3];
    let heads = w.heads.len();
    let val_dim = c / heads;
    let key_dim = cfg.key_dim;
    let scale = 1.0_f64 / (key_dim as f64).sqrt();
    let chunk_size = val_dim;
    // Split x along channel into `heads` chunks of `val_dim` channels.
    let mut feats_in: Vec<LazyTensor> = Vec::with_capacity(heads);
    for i in 0..heads {
        feats_in.push(x.narrow(1_usize, i * chunk_size, chunk_size)?);
    }
    let mut feat = feats_in[0].clone();
    let mut feats_out: Vec<LazyTensor> = Vec::with_capacity(heads);
    for i in 0..heads {
        if i > 0 {
            feat = feat.add(&feats_in[i])?;
        }
        // qkv: pointwise from val_dim → val_dim + 2*key_dim.
        let qkv_out = apply_conv_bn(&feat, &w.heads[i].qkv, anchor)?;
        let q = qkv_out.narrow(1_usize, 0, key_dim)?;
        let k = qkv_out.narrow(1_usize, key_dim, key_dim)?;
        let v = qkv_out.narrow(1_usize, 2 * key_dim, val_dim)?;
        // Depthwise smoothing on Q.
        let q = apply_conv_bn(&q, &w.heads[i].dw_q, anchor)?;
        // Flatten (B, c, H, W) → (B, c, H*W).
        let q = q.reshape(Shape::from_dims(&[b, key_dim, h * w_]))?;
        let k = k.reshape(Shape::from_dims(&[b, key_dim, h * w_]))?;
        let v = v.reshape(Shape::from_dims(&[b, val_dim, h * w_]))?;
        let q = q.mul_scalar(scale);
        // Attention: q^T @ k → (B, H*W, H*W). softmax over last dim.
        let q_t = q.permute([0, 2, 1_usize])?;
        let att = q_t.matmul(&k)?;
        let att = att.softmax_last_dim()?;
        // v @ att^T → (B, val_dim, H*W).
        let att_t = att.permute([0, 2, 1_usize])?;
        let out = v.matmul(&att_t)?;
        let out = out.reshape(Shape::from_dims(&[b, val_dim, h, w_]))?;
        feats_out.push(out);
        feat = feats_out.last().unwrap().clone();
    }
    // Concat heads along channel dim.
    let mut concat = feats_out[0].clone();
    for f in &feats_out[1..] {
        concat = concat.concat(f, 1_usize)?;
    }
    let xs = concat.relu();
    apply_conv_bn(&xs, &w.proj, anchor)
}

/// Pad a single dim with `right` zeros at the end. Composite using
/// `concat` against a freshly-built zero tensor of matching shape.
fn pad_dim_with_zeros(
    x: &LazyTensor, dim: usize, right: usize,
) -> Result<LazyTensor> {
    if right == 0 {
        return Ok(x.clone());
    }
    let dims = x.shape();
    let mut shape = dims.dims().to_vec();
    shape[dim] = right;
    let n: usize = shape.iter().product();
    let zeros = x.const_f32_like(
        Arc::<[f32]>::from(vec![0.0_f32; n]),
        Shape::from_dims(&shape),
    );
    x.concat(&zeros, dim)
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
    fn arc_ones(n: usize) -> Arc<[f32]> { Arc::from(vec![1.0_f32; n]) }
    fn arc_zeros(n: usize) -> Arc<[f32]> { Arc::from(vec![0.0_f32; n]) }
    fn tiny_bn(c: usize) -> BatchNormParams {
        BatchNormParams { w: arc_ones(c), b: arc_zeros(c) }
    }

    fn conv_bn_w(
        c_in: usize, c_out: usize, k: usize, stride: usize, pad: usize,
        groups: usize, nb: &mut dyn FnMut() -> f32,
    ) -> ConvBnWeights {
        ConvBnWeights {
            conv_w: vec_of(c_out * (c_in / groups) * k * k, nb),
            bn: tiny_bn(c_out),
            c_in, c_out, k, stride, pad, groups,
        }
    }

    fn conv1x1_bias_w(
        c_in: usize, c_out: usize, nb: &mut dyn FnMut() -> f32,
    ) -> Conv1x1BiasWeights {
        Conv1x1BiasWeights {
            w: vec_of(c_out * c_in, nb),
            b: vec_of(c_out, nb),
            c_in, c_out,
        }
    }

    fn conv_mlp_w(dim: usize, hidden: usize, nb: &mut dyn FnMut() -> f32) -> ConvMlpWeights {
        ConvMlpWeights {
            pw1: conv_bn_w(dim, hidden, 1, 1, 0, 1, nb),
            pw2: conv_bn_w(hidden, dim, 1, 1, 0, 1, nb),
        }
    }

    fn cga_w(
        c_in: usize, n_heads: usize, key_dim: usize, kernels: &[usize],
        nb: &mut dyn FnMut() -> f32,
    ) -> CgaWeights {
        let val_dim = c_in / n_heads;
        let mut heads = Vec::with_capacity(n_heads);
        for i in 0..n_heads {
            let k = kernels[i % kernels.len()];
            heads.push(CgaHeadWeights {
                qkv: conv_bn_w(val_dim, val_dim + 2 * key_dim, 1, 1, 0, 1, nb),
                dw_q: conv_bn_w(key_dim, key_dim, k, 1, k / 2, key_dim, nb),
            });
        }
        let proj = conv_bn_w(c_in, c_in, 1, 1, 0, 1, nb);
        CgaWeights { heads, proj }
    }

    fn block_w(
        dim: usize, n_heads: usize, key_dim: usize, kernels: &[usize],
        nb: &mut dyn FnMut() -> f32,
    ) -> EfficientVitBlockWeights {
        EfficientVitBlockWeights {
            dw0: conv_bn_w(dim, dim, 3, 1, 1, dim, nb),
            ffn0: conv_mlp_w(dim, dim * 2, nb),
            attn: cga_w(dim, n_heads, key_dim, kernels, nb),
            dw1: conv_bn_w(dim, dim, 3, 1, 1, dim, nb),
            ffn1: conv_mlp_w(dim, dim * 2, nb),
        }
    }

    fn res_block_w(dim: usize, nb: &mut dyn FnMut() -> f32) -> ResBlockWeights {
        ResBlockWeights {
            dw: conv_bn_w(dim, dim, 3, 1, 1, dim, nb),
            mlp: conv_mlp_w(dim, dim * 2, nb),
        }
    }

    fn patchmerge_w(
        in_c: usize, out_c: usize, nb: &mut dyn FnMut() -> f32,
    ) -> PatchMergeWeights {
        let hid = in_c * 4;
        PatchMergeWeights {
            conv1: conv_bn_w(in_c, hid, 1, 1, 0, 1, nb),
            conv2: conv_bn_w(hid, hid, 3, 2, 1, hid, nb),
            conv3: conv_bn_w(hid, out_c, 1, 1, 0, 1, nb),
            se: SeWeights {
                fc1: conv1x1_bias_w(hid, hid / 4, nb),
                fc2: conv1x1_bias_w(hid / 4, hid, nb),
            },
        }
    }

    fn downsample_w(
        in_c: usize, out_c: usize, nb: &mut dyn FnMut() -> f32,
    ) -> DownsampleWeights {
        DownsampleWeights {
            res1: res_block_w(in_c, nb),
            patchmerge: patchmerge_w(in_c, out_c, nb),
            res2: res_block_w(out_c, nb),
        }
    }

    fn stem_w(dim: usize, nb: &mut dyn FnMut() -> f32) -> StemWeights {
        StemWeights {
            // Each stride-2 stem block uses kernel=3 padding=1.
            conv1: conv_bn_w(3, dim / 8, 3, 2, 1, 1, nb),
            conv2: conv_bn_w(dim / 8, dim / 4, 3, 2, 1, 1, nb),
            conv3: conv_bn_w(dim / 4, dim / 2, 3, 2, 1, 1, nb),
            conv4: conv_bn_w(dim / 2, dim, 3, 2, 1, 1, nb),
        }
    }

    fn tiny_config() -> EfficientVitConfig {
        EfficientVitConfig {
            channels: [16, 32, 48],
            blocks: [1, 1, 1],
            heads: [2, 2, 2],
            kernels: vec![3, 3],
            stage_resolutions: [4, 2, 1], // image_size=64 → /16 = 4 → /2 = 2 → /2 = 1
            key_dim: 4,
        }
    }

    fn tiny_weights(cfg: &EfficientVitConfig) -> EfficientVitWeights {
        let mut nb = rng_seed(2026);
        let ch = cfg.channels;
        let head_kernels = &cfg.kernels;
        let stem = stem_w(ch[0], &mut nb);
        let stage0 = EfficientVitStageWeights {
            downsample: None,
            blocks: (0..cfg.blocks[0])
                .map(|_| block_w(ch[0], cfg.heads[0], cfg.key_dim, head_kernels, &mut nb))
                .collect(),
        };
        let stage1 = EfficientVitStageWeights {
            downsample: Some(downsample_w(ch[0], ch[1], &mut nb)),
            blocks: (0..cfg.blocks[1])
                .map(|_| block_w(ch[1], cfg.heads[1], cfg.key_dim, head_kernels, &mut nb))
                .collect(),
        };
        let stage2 = EfficientVitStageWeights {
            downsample: Some(downsample_w(ch[1], ch[2], &mut nb)),
            blocks: (0..cfg.blocks[2])
                .map(|_| block_w(ch[2], cfg.heads[2], cfg.key_dim, head_kernels, &mut nb))
                .collect(),
        };
        EfficientVitWeights {
            stem,
            stages: [stage0, stage1, stage2],
            head: None,
        }
    }

    #[test]
    fn forward_features_shape_and_finite() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = EfficientVitModel { config: cfg.clone(), weights };
        let img = LazyTensor::from_f32(
            (0..(3 * 64 * 64)).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 64, 64]), &Device::cpu(),
        );
        let feats = model.forward_features(&img).unwrap();
        let shape = feats.shape();
        let dims = shape.dims();
        // After stem (/16): 4. After 2 downsamples (/2 each): 1.
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], cfg.channels[2]);
        assert_eq!(dims[2], 1);
        assert_eq!(dims[3], 1);
        for &v in &feats.realize_f32() {
            assert!(v.is_finite(), "non-finite feature: {v}");
        }
    }

    /// Windowing: tiny config with stage_resolutions[0] > 7
    /// forces the windowing path. Verify the path executes
    /// end-to-end and the output is finite. Use a stage 0
    /// resolution of 8 (need image 8*16 = 128, but to keep
    /// the test cheap we test the windowing kernel directly
    /// via cga_core's wrapper on a hand-built input).
    #[test]
    fn windowing_path_runs() {
        let mut cfg = tiny_config();
        cfg.stage_resolutions[0] = 8; // > 7 → triggers windowing
        let weights = tiny_weights(&cfg);
        let model = EfficientVitModel { config: cfg.clone(), weights };
        // image_size=128 → stem /16 = 8 at stage 0. Windowing
        // will tile into (8/7) → pad to 14 → 2x2 windows of 7x7.
        let img = LazyTensor::from_f32(
            (0..(3 * 128 * 128)).map(|i| (i as f32) * 0.001).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 128, 128]), &Device::cpu(),
        );
        let feats = model.forward_features(&img).unwrap();
        let shape = feats.shape();
        let dims = shape.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], cfg.channels[2]);
        for &v in &feats.realize_f32() {
            assert!(v.is_finite(), "non-finite windowed feature: {v}");
        }
    }

    /// `cga_core` (no windowing) responds to input changes — proves
    /// the cascade + Q/K/V/proj path is wired through to the output.
    #[test]
    fn cga_responds_to_input() {
        let mut nb = rng_seed(99);
        let dim = 16;
        let cga = cga_w(dim, 2, 4, &[3, 3], &mut nb);
        let cfg = EfficientVitConfig {
            channels: [dim, dim, dim], blocks: [1, 1, 1],
            heads: [2, 2, 2], kernels: vec![3, 3],
            stage_resolutions: [4, 2, 1], key_dim: 4,
        };
        let a = LazyTensor::from_f32(
            (0..(dim * 4 * 4)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, dim, 4, 4]), &Device::cpu(),
        );
        let b = LazyTensor::from_f32(
            (0..(dim * 4 * 4)).map(|i| (i as f32) * 0.01 + 0.5).collect::<Vec<_>>(),
            Shape::from_dims(&[1, dim, 4, 4]), &Device::cpu(),
        );
        let out_a = cga_core(&a, &cga, &cfg, &a).unwrap().realize_f32();
        let out_b = cga_core(&b, &cga, &cfg, &b).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in out_a.iter().zip(out_b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "CGA must respond to input changes, max_diff = {max_diff}");
    }

    #[test]
    fn preset_constructs() {
        let m0 = EfficientVitConfig::m0();
        assert_eq!(m0.channels, [64, 128, 192]);
        assert_eq!(m0.stage_resolutions, [14, 7, 4]);
        let m1 = EfficientVitConfig::m1();
        assert_eq!(m1.kernels, vec![7, 5, 3, 3]);
    }
}

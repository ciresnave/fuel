//! FastViT — lazy port.
//!
//! Hierarchical 4-stage hybrid backbone: stem → 4 stages, each
//! consisting of a patch-embedding downsample + positional
//! encoding + N blocks. Stages 0-2 use RepMixer blocks; stage 3
//! uses Attention blocks when `cfg.attn` is true, otherwise also
//! RepMixer.
//!
//! v1 assumes weights are already **reparameterized** at load
//! time — the eager port fuses each multi-branch MobileOne block
//! into a single Conv2d with bias before runtime. This lazy port
//! takes the fused single-conv form directly.
//!
//! Block summary (all on `(B, C, H, W)` channel-first tensors):
//!   - **MobileOne (reparam'd)** = Conv2d + bias + optional SE + optional GELU(erf).
//!   - **ConvNorm** = depthwise conv with BN absorbed at load time
//!     → represented as Conv2d + bias.
//!   - **ConvMLP** = ConvNorm (7×7 dw) → fc1 → GELU → fc2 (1×1 convs).
//!   - **RepMixer** = γ ⊙ (mixer(x) − norm(x)) + x, where
//!     `mixer` and `norm` are reparam'd MobileOne blocks.
//!   - **RepMixerBlock** = `y = token_mixer(x); y + γ · mlp(y)`.
//!   - **AttentionBlock** = `x + γ1 · attn(LN(x))`, then
//!     `x + γ2 · mlp(x)`.
//!   - **PositionalEncoding** = `x + dwconv7(x)` residual.
//!   - **PatchEmbed** = `(lk(x) + sk(x))` → optional SE → optional
//!     GELU → 1×1 MobileOne block.
//!
//! v1 scope: F32, batch == 1, forward-only inference.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct FastVitConfig {
    /// Stem output channels (= stage 0 input channels).
    pub in_channels: usize,
    /// Number of blocks per stage (length 4).
    pub blocks: [usize; 4],
    /// MLP expansion ratio (for the FFN inside each block).
    pub exp_ratio: usize,
    /// If true and `idx == 3`, the last stage uses Attention blocks
    /// instead of RepMixer blocks.
    pub attn: bool,
    /// Whether the patch-embed downsamples apply a GELU(erf) between
    /// the SE block and the trailing MobileOne 1×1 conv.
    pub lkc_use_act: bool,
    /// Per-head feature dim for the attention block (timm default = 32).
    pub head_dim: usize,
    /// Image size for forward assertions (presets use 256).
    pub image_size: usize,
    /// Number of classes for the optional head.
    pub num_classes: Option<usize>,
}

impl FastVitConfig {
    /// `fastvit_t8` (timm).
    pub fn t8() -> Self {
        Self {
            in_channels: 48, blocks: [2, 2, 4, 2],
            exp_ratio: 3, attn: false, lkc_use_act: true,
            head_dim: 32, image_size: 256,
            num_classes: Some(1000),
        }
    }
    /// `fastvit_sa12` — hybrid variant with attention in the last stage.
    pub fn sa12() -> Self {
        Self {
            in_channels: 64, blocks: [2, 2, 6, 2],
            exp_ratio: 4, attn: true, lkc_use_act: true,
            head_dim: 32, image_size: 256,
            num_classes: Some(1000),
        }
    }
    /// FastViT-MCI0 — vision backbone for MobileCLIP-S1 / MetaCLIP.
    pub fn mci0() -> Self {
        Self {
            in_channels: 64, blocks: [2, 6, 10, 2],
            exp_ratio: 3, attn: true, lkc_use_act: true,
            head_dim: 32, image_size: 256,
            num_classes: None,
        }
    }
    /// FastViT-MCI1 — vision backbone for MobileCLIP-S2.
    pub fn mci1() -> Self {
        Self {
            in_channels: 64, blocks: [4, 12, 20, 4],
            exp_ratio: 3, attn: true, lkc_use_act: true,
            head_dim: 32, image_size: 256,
            num_classes: None,
        }
    }
    /// FastViT-MCI2 — vision backbone for MobileCLIP-S3.
    pub fn mci2() -> Self {
        Self {
            in_channels: 80, blocks: [4, 12, 24, 4],
            exp_ratio: 3, attn: true, lkc_use_act: true,
            head_dim: 32, image_size: 256,
            num_classes: None,
        }
    }
}

// ---- Weight structures ------------------------------------------------------

/// Plain Conv2d with bias (post-reparameterization / BN-absorbed form).
#[derive(Debug, Clone)]
pub struct Conv2dBiasWeights {
    pub w: Arc<[f32]>,
    pub b: Arc<[f32]>,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub groups: usize,
}

/// Squeeze-and-Excitation block: two 1×1 convs + sigmoid gate.
#[derive(Debug, Clone)]
pub struct SeWeights {
    pub fc1: Conv2dBiasWeights,
    pub fc2: Conv2dBiasWeights,
}

/// Post-reparameterization MobileOne block: a single fused conv +
/// optional SE + optional GELU(erf).
#[derive(Debug, Clone)]
pub struct ReparamMobileOneWeights {
    pub conv: Conv2dBiasWeights,
    pub se: Option<SeWeights>,
    pub use_act: bool,
}

/// `ConvMLP` weights: a 7×7 depthwise `conv_norm` + two 1×1 convs.
#[derive(Debug, Clone)]
pub struct ConvMlpWeights {
    /// 7×7 depthwise conv with BN absorbed.
    pub conv_norm: Conv2dBiasWeights,
    /// 1×1 conv to `dim * exp_ratio`.
    pub fc1: Conv2dBiasWeights,
    /// 1×1 conv back to `dim`.
    pub fc2: Conv2dBiasWeights,
}

#[derive(Debug, Clone)]
pub struct RepMixerWeights {
    /// Per-channel layer-scale γ (length = dim).
    pub gamma: Arc<[f32]>,
    /// Reparam'd MobileOne block used as the "mixer" branch.
    pub mixer: ReparamMobileOneWeights,
    /// Reparam'd MobileOne block used as the "norm" branch.
    pub norm: ReparamMobileOneWeights,
}

#[derive(Debug, Clone)]
pub struct RepMixerBlockWeights {
    pub gamma_mlp: Arc<[f32]>,
    pub token_mixer: RepMixerWeights,
    pub mlp: ConvMlpWeights,
}

#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    pub gain: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct FastVitAttentionWeights {
    /// Fused QKV: `[hidden, 3·hidden]` (loaded directly into
    /// `WeightStorage::apply_linear` convention).
    pub qkv: WeightStorage,
    pub proj: WeightStorage,
    pub proj_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct AttentionBlockWeights {
    pub gamma1: Arc<[f32]>,
    pub gamma2: Arc<[f32]>,
    pub norm_bn: BnWeights,
    pub token_mixer: FastVitAttentionWeights,
    pub mlp: ConvMlpWeights,
}

/// Fused-affine BatchNorm (inference mode): per-channel `w` and `b`.
#[derive(Debug, Clone)]
pub struct BnWeights {
    /// gain / sqrt(var + eps).
    pub w: Arc<[f32]>,
    /// bias - mean · w.
    pub b: Arc<[f32]>,
}

/// One stage's blocks (either all RepMixer or all Attention).
#[derive(Debug, Clone)]
pub enum FastVitStageBlocks {
    RepMixer(Vec<RepMixerBlockWeights>),
    Attention(Vec<AttentionBlockWeights>),
}

#[derive(Debug, Clone)]
pub struct PatchEmbedWeights {
    pub large_conv: Conv2dBiasWeights,
    pub small_conv: Conv2dBiasWeights,
    pub se: Option<SeWeights>,
    /// Trailing 1×1 MobileOne block.
    pub mobileone_1x1: ReparamMobileOneWeights,
}

#[derive(Debug, Clone)]
pub struct StageWeights {
    /// `None` for stage 0 (the stem already handled the 4×
    /// downsample).
    pub downsample: Option<PatchEmbedWeights>,
    /// 7×7 depthwise conv added as a residual: `x + dw7(x)`.
    pub pos_emb: Option<Conv2dBiasWeights>,
    pub blocks: FastVitStageBlocks,
}

#[derive(Debug, Clone)]
pub struct FastVitHeadWeights {
    /// 1×1 conv-BN: in_channels → final_features.
    pub conv: Conv2dBiasWeights,
    pub linear_w: WeightStorage,
    pub linear_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct FastVitWeights {
    /// Stem: 3 reparam'd MobileOne blocks.
    pub stem: [ReparamMobileOneWeights; 3],
    pub stages: [StageWeights; 4],
    pub head: Option<FastVitHeadWeights>,
}

#[derive(Debug, Clone)]
pub struct FastVitModel {
    pub config: FastVitConfig,
    pub weights: FastVitWeights,
}

// ---- Forward ---------------------------------------------------------------

impl FastVitModel {
    /// Run the backbone (stem + 4 stages) and return the channels-
    /// first feature map BEFORE global mean pool and the head.
    pub fn forward_features(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let dims = image.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        assert_eq!(dims[1], 3);
        let mut x = self.run_stem(image)?;
        for (si, stage) in self.weights.stages.iter().enumerate() {
            x = run_stage(&x, stage, image)?;
            let _ = si;
        }
        Ok(x)
    }

    /// Full forward: backbone → global mean → optional head
    /// (1×1 conv-BN-GELU → flatten → linear classifier).
    pub fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let feats = self.forward_features(image)?;
        match &self.weights.head {
            None => Ok(feats),
            Some(head) => {
                let h = apply_conv2d_bias(&feats, &head.conv, image)?;
                let h = h.gelu_erf();
                let pooled = h.mean_dim(3_usize)?.mean_dim(2_usize)?;
                let dims = pooled.shape();
                let dims = dims.dims();
                let c = dims[1];
                let n = head.linear_b.len();
                let logits = head.linear_w.apply_linear(&pooled, c, n);
                let bias = image.const_f32_like(
                    Arc::clone(&head.linear_b), Shape::from_dims(&[n]),
                );
                logits.broadcast_add(&bias)
            }
        }
    }

    fn run_stem(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let mut x = apply_reparam_mobileone(image, &self.weights.stem[0], image)?;
        x = apply_reparam_mobileone(&x, &self.weights.stem[1], image)?;
        apply_reparam_mobileone(&x, &self.weights.stem[2], image)
    }
}

fn run_stage(
    x: &LazyTensor, stage: &StageWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let mut x = if let Some(ds) = &stage.downsample {
        apply_patch_embed(x, ds, anchor)?
    } else {
        x.clone()
    };
    if let Some(pos) = &stage.pos_emb {
        let res = apply_conv2d_bias(&x, pos, anchor)?;
        x = x.add(&res)?;
    }
    match &stage.blocks {
        FastVitStageBlocks::RepMixer(blocks) => {
            for blk in blocks {
                x = apply_repmixer_block(&x, blk, anchor)?;
            }
        }
        FastVitStageBlocks::Attention(blocks) => {
            for blk in blocks {
                x = apply_attention_block(&x, blk, anchor)?;
            }
        }
    }
    Ok(x)
}

// ---- Block helpers ---------------------------------------------------------

fn apply_conv2d_bias(
    x: &LazyTensor, c: &Conv2dBiasWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let w = anchor.const_f32_like(
        Arc::clone(&c.w),
        Shape::from_dims(&[c.c_out, c.c_in / c.groups, c.k, c.k]),
    );
    let bias = anchor.const_f32_like(
        Arc::clone(&c.b), Shape::from_dims(&[c.c_out]),
    );
    x.conv2d(
        &w, Some(&bias),
        (c.stride, c.stride),
        (c.pad, c.pad),
        c.groups,
    )
}

fn apply_bn_fused(
    x: &LazyTensor, bn: &BnWeights, channels: usize,
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

fn apply_se(
    x: &LazyTensor, se: &SeWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let c = dims[1];
    let pooled = x
        .mean_dim(3_usize)?
        .mean_dim(2_usize)?
        .reshape(Shape::from_dims(&[dims[0], c, 1, 1]))?;
    let g = apply_conv2d_bias(&pooled, &se.fc1, anchor)?.relu();
    let g = apply_conv2d_bias(&g, &se.fc2, anchor)?.sigmoid();
    let g_b = g.broadcast_to(Shape::from_dims(dims))?;
    x.mul(&g_b)
}

fn apply_reparam_mobileone(
    x: &LazyTensor, m: &ReparamMobileOneWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let mut y = apply_conv2d_bias(x, &m.conv, anchor)?;
    if let Some(se) = &m.se {
        y = apply_se(&y, se, anchor)?;
    }
    if m.use_act {
        y = y.gelu_erf();
    }
    Ok(y)
}

fn apply_conv_mlp(
    x: &LazyTensor, m: &ConvMlpWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let x = apply_conv2d_bias(x, &m.conv_norm, anchor)?;
    let x = apply_conv2d_bias(&x, &m.fc1, anchor)?;
    let x = x.gelu_erf();
    apply_conv2d_bias(&x, &m.fc2, anchor)
}

fn apply_repmixer(
    x: &LazyTensor, r: &RepMixerWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let mixer = apply_reparam_mobileone(x, &r.mixer, anchor)?;
    let norm = apply_reparam_mobileone(x, &r.norm, anchor)?;
    let diff = mixer.sub(&norm)?;
    let gamma = anchor
        .const_f32_like(Arc::clone(&r.gamma), Shape::from_dims(&[r.gamma.len()]))
        .reshape(Shape::from_dims(&[1, r.gamma.len(), 1, 1]))?;
    let scaled = diff.broadcast_mul(&gamma)?;
    x.add(&scaled)
}

fn apply_repmixer_block(
    x: &LazyTensor, b: &RepMixerBlockWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let y = apply_repmixer(x, &b.token_mixer, anchor)?;
    let mlp_out = apply_conv_mlp(&y, &b.mlp, anchor)?;
    let gamma = anchor
        .const_f32_like(Arc::clone(&b.gamma_mlp), Shape::from_dims(&[b.gamma_mlp.len()]))
        .reshape(Shape::from_dims(&[1, b.gamma_mlp.len(), 1, 1]))?;
    let scaled = mlp_out.broadcast_mul(&gamma)?;
    y.add(&scaled)
}

fn apply_fastvit_attention(
    x: &LazyTensor, w: &FastVitAttentionWeights,
    head_dim: usize, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let c = dims[1]; let h = dims[2]; let ww = dims[3];
    let n = h * ww;
    let num_heads = c / head_dim;
    let scale = 1.0_f64 / (head_dim as f64).sqrt();
    // (B, C, H, W) → (B, N, C)
    let x_seq = x
        .reshape(Shape::from_dims(&[b, c, n]))?
        .permute([0, 2, 1_usize])?;
    // qkv: (B, N, 3C)
    let qkv = w.qkv.apply_linear(&x_seq, c, 3 * c);
    let q = qkv.narrow(2_usize, 0, c)?;
    let k = qkv.narrow(2_usize, c, c)?;
    let v = qkv.narrow(2_usize, 2 * c, c)?;
    let q = q.reshape(Shape::from_dims(&[b, n, num_heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;
    let k = k.reshape(Shape::from_dims(&[b, n, num_heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;
    let v = v.reshape(Shape::from_dims(&[b, n, num_heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;
    let q = q.mul_scalar(scale);
    let kt = k.permute([0, 1, 3, 2_usize])?;
    let scores = q.matmul(&kt)?;
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?;
    let ctx = ctx
        .permute([0, 2, 1, 3_usize])?
        .reshape(Shape::from_dims(&[b, n, c]))?;
    let projected = w.proj.apply_linear(&ctx, c, c);
    let bias_t = anchor.const_f32_like(
        Arc::clone(&w.proj_bias), Shape::from_dims(&[c]),
    );
    let out = projected.broadcast_add(&bias_t)?;
    // Back to (B, C, H, W).
    Ok(out
        .permute([0, 2, 1_usize])?
        .reshape(Shape::from_dims(&[b, c, h, ww]))?)
}

fn apply_attention_block(
    x: &LazyTensor, b: &AttentionBlockWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let c = dims[1];
    let head_dim = 32;
    // norm + attn, scaled by gamma1, +residual.
    let normed = apply_bn_fused(x, &b.norm_bn, c)?;
    let attn = apply_fastvit_attention(&normed, &b.token_mixer, head_dim, anchor)?;
    let gamma1 = anchor
        .const_f32_like(Arc::clone(&b.gamma1), Shape::from_dims(&[b.gamma1.len()]))
        .reshape(Shape::from_dims(&[1, b.gamma1.len(), 1, 1]))?;
    let attn_scaled = attn.broadcast_mul(&gamma1)?;
    let x = x.add(&attn_scaled)?;

    // mlp + γ2 + residual.
    let mlp_out = apply_conv_mlp(&x, &b.mlp, anchor)?;
    let gamma2 = anchor
        .const_f32_like(Arc::clone(&b.gamma2), Shape::from_dims(&[b.gamma2.len()]))
        .reshape(Shape::from_dims(&[1, b.gamma2.len(), 1, 1]))?;
    let mlp_scaled = mlp_out.broadcast_mul(&gamma2)?;
    x.add(&mlp_scaled)
}

fn apply_patch_embed(
    x: &LazyTensor, p: &PatchEmbedWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let lk = apply_conv2d_bias(x, &p.large_conv, anchor)?;
    let sk = apply_conv2d_bias(x, &p.small_conv, anchor)?;
    let mut x = lk.add(&sk)?;
    if let Some(se) = &p.se {
        x = apply_se(&x, se, anchor)?;
    }
    // The eager port unconditionally absorbs the SE block; the GELU
    // gating is controlled by `lkc_use_act` at the model level —
    // here we always apply it since the timm reference does.
    x = x.gelu_erf();
    apply_reparam_mobileone(&x, &p.mobileone_1x1, anchor)
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

    fn conv_w(
        c_in: usize, c_out: usize, k: usize, stride: usize, pad: usize, groups: usize,
        nb: &mut dyn FnMut() -> f32,
    ) -> Conv2dBiasWeights {
        Conv2dBiasWeights {
            w: vec_of(c_out * (c_in / groups) * k * k, nb),
            b: vec_of(c_out, nb),
            c_in, c_out, k, stride, pad, groups,
        }
    }

    fn se_w(c: usize, nb: &mut dyn FnMut() -> f32) -> SeWeights {
        let sq = (c / 16).max(1);
        SeWeights {
            fc1: conv_w(c, sq, 1, 1, 0, 1, nb),
            fc2: conv_w(sq, c, 1, 1, 0, 1, nb),
        }
    }

    fn reparam_mobileone(
        c_in: usize, c_out: usize, k: usize, stride: usize, groups: usize,
        with_se: bool, use_act: bool, nb: &mut dyn FnMut() -> f32,
    ) -> ReparamMobileOneWeights {
        let pad = k / 2;
        ReparamMobileOneWeights {
            conv: conv_w(c_in, c_out, k, stride, pad, groups, nb),
            se: if with_se { Some(se_w(c_out, nb)) } else { None },
            use_act,
        }
    }

    fn conv_mlp_w(
        dim: usize, exp: usize, nb: &mut dyn FnMut() -> f32,
    ) -> ConvMlpWeights {
        ConvMlpWeights {
            conv_norm: conv_w(dim, dim, 7, 1, 3, dim, nb),
            fc1: conv_w(dim, dim * exp, 1, 1, 0, 1, nb),
            fc2: conv_w(dim * exp, dim, 1, 1, 0, 1, nb),
        }
    }

    fn repmixer_block_w(dim: usize, exp: usize, nb: &mut dyn FnMut() -> f32) -> RepMixerBlockWeights {
        RepMixerBlockWeights {
            gamma_mlp: vec_of(dim, nb),
            token_mixer: RepMixerWeights {
                gamma: vec_of(dim, nb),
                mixer: reparam_mobileone(dim, dim, 3, 1, dim, false, false, nb),
                norm: reparam_mobileone(dim, dim, 3, 1, dim, false, false, nb),
            },
            mlp: conv_mlp_w(dim, exp, nb),
        }
    }

    fn attention_block_w(
        dim: usize, exp: usize, nb: &mut dyn FnMut() -> f32,
    ) -> AttentionBlockWeights {
        AttentionBlockWeights {
            gamma1: vec_of(dim, nb),
            gamma2: vec_of(dim, nb),
            norm_bn: BnWeights {
                w: Arc::from(vec![1.0_f32; dim]),
                b: Arc::from(vec![0.0_f32; dim]),
            },
            token_mixer: FastVitAttentionWeights {
                qkv: ws(dim * 3 * dim, nb),
                proj: ws(dim * dim, nb),
                proj_bias: vec_of(dim, nb),
            },
            mlp: conv_mlp_w(dim, exp, nb),
        }
    }

    fn patch_embed_w(
        c_in: usize, c_out: usize, nb: &mut dyn FnMut() -> f32,
    ) -> PatchEmbedWeights {
        PatchEmbedWeights {
            large_conv: conv_w(c_in, c_out, 7, 2, 3, 1, nb),
            small_conv: conv_w(c_in, c_out, 3, 2, 1, 1, nb),
            se: Some(se_w(c_out, nb)),
            mobileone_1x1: reparam_mobileone(c_out, c_out, 1, 1, 1, false, true, nb),
        }
    }

    fn tiny_config() -> FastVitConfig {
        FastVitConfig {
            in_channels: 8,
            blocks: [1, 1, 1, 1],
            exp_ratio: 2,
            attn: true,
            lkc_use_act: true,
            head_dim: 4,
            image_size: 32,
            num_classes: Some(10),
        }
    }

    fn build_weights(cfg: &FastVitConfig) -> FastVitWeights {
        let mut nb = rng_seed(2026);
        let c0 = cfg.in_channels;
        // Stem: 3 reparam'd MobileOne blocks. Conv -> /2 -> /2 → matches eager
        // (stride 2 for first two, stride 1 for the third).
        let stem = [
            reparam_mobileone(3, c0, 3, 2, 1, false, true, &mut nb),
            reparam_mobileone(c0, c0, 3, 2, c0, false, true, &mut nb),
            reparam_mobileone(c0, c0, 1, 1, 1, false, true, &mut nb),
        ];

        // Each stage's dim doubles: stage idx i → dim = c0 * 2^i.
        let mut stages: [_; 4] = [
            StageWeights {
                downsample: None,
                pos_emb: Some(conv_w(c0, c0, 7, 1, 3, c0, &mut nb)),
                blocks: FastVitStageBlocks::RepMixer(
                    (0..cfg.blocks[0]).map(|_| repmixer_block_w(c0, cfg.exp_ratio, &mut nb)).collect(),
                ),
            },
            StageWeights {
                downsample: Some(patch_embed_w(c0, c0 * 2, &mut nb)),
                pos_emb: Some(conv_w(c0 * 2, c0 * 2, 7, 1, 3, c0 * 2, &mut nb)),
                blocks: FastVitStageBlocks::RepMixer(
                    (0..cfg.blocks[1]).map(|_| repmixer_block_w(c0 * 2, cfg.exp_ratio, &mut nb)).collect(),
                ),
            },
            StageWeights {
                downsample: Some(patch_embed_w(c0 * 2, c0 * 4, &mut nb)),
                pos_emb: Some(conv_w(c0 * 4, c0 * 4, 7, 1, 3, c0 * 4, &mut nb)),
                blocks: FastVitStageBlocks::RepMixer(
                    (0..cfg.blocks[2]).map(|_| repmixer_block_w(c0 * 4, cfg.exp_ratio, &mut nb)).collect(),
                ),
            },
            StageWeights {
                downsample: Some(patch_embed_w(c0 * 4, c0 * 8, &mut nb)),
                pos_emb: Some(conv_w(c0 * 8, c0 * 8, 7, 1, 3, c0 * 8, &mut nb)),
                blocks: if cfg.attn {
                    FastVitStageBlocks::Attention(
                        (0..cfg.blocks[3]).map(|_| attention_block_w(c0 * 8, cfg.exp_ratio, &mut nb)).collect(),
                    )
                } else {
                    FastVitStageBlocks::RepMixer(
                        (0..cfg.blocks[3]).map(|_| repmixer_block_w(c0 * 8, cfg.exp_ratio, &mut nb)).collect(),
                    )
                },
            },
        ];
        let final_c = c0 * 8;
        let head = cfg.num_classes.map(|n| FastVitHeadWeights {
            conv: conv_w(final_c, final_c, 1, 1, 0, 1, &mut nb),
            linear_w: ws(final_c * n, &mut nb),
            linear_b: vec_of(n, &mut nb),
        });
        // Avoid taking references during the assignment dance.
        let s3 = std::mem::replace(&mut stages[3], StageWeights {
            downsample: None, pos_emb: None,
            blocks: FastVitStageBlocks::RepMixer(vec![]),
        });
        let s2 = std::mem::replace(&mut stages[2], StageWeights {
            downsample: None, pos_emb: None,
            blocks: FastVitStageBlocks::RepMixer(vec![]),
        });
        let s1 = std::mem::replace(&mut stages[1], StageWeights {
            downsample: None, pos_emb: None,
            blocks: FastVitStageBlocks::RepMixer(vec![]),
        });
        let s0 = std::mem::replace(&mut stages[0], StageWeights {
            downsample: None, pos_emb: None,
            blocks: FastVitStageBlocks::RepMixer(vec![]),
        });

        FastVitWeights { stem, stages: [s0, s1, s2, s3], head }
    }

    #[test]
    fn forward_with_head_shape_and_finite() {
        let cfg = tiny_config();
        let weights = build_weights(&cfg);
        let model = FastVitModel { config: cfg.clone(), weights };
        let img = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 10]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn forward_features_returns_feature_map() {
        let cfg = tiny_config();
        let weights = build_weights(&cfg);
        let model = FastVitModel { config: cfg.clone(), weights };
        let img = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let feats = model.forward_features(&img).unwrap();
        let shape = feats.shape();
        let dims = shape.dims();
        assert_eq!(dims[0], 1);
        // Final stage channel dim = c0 * 8.
        assert_eq!(dims[1], cfg.in_channels * 8);
        for &v in &feats.realize_f32() {
            assert!(v.is_finite(), "non-finite feature: {v}");
        }
    }

    #[test]
    fn forward_responds_to_input() {
        let cfg = tiny_config();
        let weights = build_weights(&cfg);
        let model = FastVitModel { config: cfg, weights };
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
        // Tiny random weights (~0.025) attenuate through stem + 4 stages of
        // (conv + SE-sigmoid + GELU + 7x7-dwconv + MLP) blocks; the signal
        // survives but is heavily damped.
        assert!(max_diff > 1e-10,
            "FastViT must respond to input changes, max_diff = {max_diff}");
    }

    #[test]
    fn presets_construct() {
        let t8 = FastVitConfig::t8();
        assert_eq!(t8.in_channels, 48);
        assert!(!t8.attn);
        let sa12 = FastVitConfig::sa12();
        assert_eq!(sa12.in_channels, 64);
        assert!(sa12.attn);
    }
}

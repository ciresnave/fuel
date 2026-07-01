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

use crate::lazy::{load_tensor_as_f32, LazyTensor, WeightStorage};
use crate::lazy_convmixer::BatchNormParams;
use crate::Result;
use fuel_ir::Shape;
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
    /// Mobile multi-query attention block. Q has `heads` separate
    /// streams (each `kv_dim` channels); K/V each have a single
    /// stream (`kv_dim` channels) broadcast across all heads. When
    /// `kv_stride > 1`, K/V get a depthwise downsample by
    /// `kernel`/`kv_stride` before the projections.
    ///
    /// v1: `stride` must equal 1.
    Attention {
        out_channels: usize, heads: usize, kernel: usize, stride: usize,
        kv_dim: usize, kv_stride: usize,
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

/// Mobile multi-query attention weights. The depthwise
/// downsample (+ BN) on K/V is present iff `kv_stride > 1`.
#[derive(Debug, Clone)]
pub struct MqaWeights {
    /// BatchNorm applied to the residual input before Q/K/V.
    pub input_norm: BatchNormParams,
    /// Q projection: 1×1 conv `in_channels → kv_dim · heads`.
    pub query_proj: Conv2dBnWeights,
    /// Optional K depthwise downsample (groups=in_channels).
    pub key_down: Option<Conv2dBnWeights>,
    /// K projection: 1×1 conv `in_channels → kv_dim`.
    pub key_proj: Conv2dBnWeights,
    /// Optional V depthwise downsample.
    pub value_down: Option<Conv2dBnWeights>,
    /// V projection: 1×1 conv `in_channels → kv_dim`.
    pub value_proj: Conv2dBnWeights,
    /// Output projection: 1×1 conv `kv_dim · heads → out_channels`.
    pub output_proj: Conv2dBnWeights,
    pub heads: usize,
    pub kv_dim: usize,
    pub layer_scale_gamma: Option<Arc<[f32]>>,
    pub skip: bool,
}

/// Per-block weights, tagged to match the spec.
#[derive(Debug, Clone)]
pub enum BlockWeights {
    Convolutional(Conv2dBnWeights),
    EdgeResidual(EdgeResidualWeights),
    UniversalBottleneck(UibWeights),
    Attention(MqaWeights),
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
        let pooled = x.global_avg_pool_2d()?;
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
    let _ = channels;
    x.channel_affine_4d(Arc::clone(&bn.w), Arc::clone(&bn.b))
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
        BlockWeights::Attention(mqa) => apply_mqa(x, mqa, anchor),
    }
}

/// Mobile multi-query attention forward. Input `(1, C, H, W)`;
/// output `(1, C_out, H, W)`.
///
///   q = query_proj(BN(x))                                            shape (1, kv_dim·heads, H, W)
///   k = key_proj(optional key_down(BN(x)))                           shape (1, kv_dim, H_kv, W_kv)
///   v = value_proj(optional value_down(BN(x)))                       shape (1, kv_dim, H_kv, W_kv)
///
///   Reshape into multi-query attention layout:
///     q' = q.reshape(1, heads, kv_dim, H·W).transpose(-1,-2)         (1, heads, H·W, kv_dim)
///     kv = reshape_kv (1, kv_dim, H_kv·W_kv) → transpose → unsqueeze (1, 1, H_kv·W_kv, kv_dim)
///
///   attn = softmax(q' · scale · k^T)                                 (1, heads, H·W, H_kv·W_kv)
///   o    = attn · v                                                  (1, heads, H·W, kv_dim)
///   reshape back to (1, kv_dim·heads, H, W), output_proj, then optional
///   layer-scale + optional residual.
fn apply_mqa(
    x: &LazyTensor, mqa: &MqaWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let h = dims[2]; let w = dims[3];
    let heads = mqa.heads;
    let kv_dim = mqa.kv_dim;
    let scale = 1.0_f64 / (kv_dim as f64).sqrt();

    let normed = apply_bn(x, &mqa.input_norm, mqa.input_norm.w.len())?;

    // Q: 1×1 conv → (B, kv_dim*heads, H, W)
    let q = apply_conv_bn(&normed, &mqa.query_proj, anchor)?;
    let q = q
        .reshape(Shape::from_dims(&[b, heads, kv_dim, h * w]))?
        .permute([0, 1, 3, 2_usize])?; // (B, heads, H·W, kv_dim)
    let q = q.mul_scalar(scale);

    // K
    let mut k_input = normed.clone();
    if let Some(kd) = &mqa.key_down {
        k_input = apply_conv_bn(&k_input, kd, anchor)?;
    }
    let k = apply_conv_bn(&k_input, &mqa.key_proj, anchor)?;
    let k_dims = k.shape();
    let k_dims = k_dims.dims();
    let h_kv = k_dims[2]; let w_kv = k_dims[3];
    let kv_len = h_kv * w_kv;
    // (B, kv_dim, H_kv·W_kv) → (B, H_kv·W_kv, kv_dim) → unsqueeze head dim
    let k_seq = k
        .reshape(Shape::from_dims(&[b, kv_dim, kv_len]))?
        .permute([0, 2, 1_usize])?
        .reshape(Shape::from_dims(&[b, 1, kv_len, kv_dim]))?;
    // Broadcast to (B, heads, kv_len, kv_dim) so matmul shapes match.
    let k_bc = k_seq.broadcast_to(Shape::from_dims(&[b, heads, kv_len, kv_dim]))?;

    // V (same shape gymnastics)
    let mut v_input = normed.clone();
    if let Some(vd) = &mqa.value_down {
        v_input = apply_conv_bn(&v_input, vd, anchor)?;
    }
    let v = apply_conv_bn(&v_input, &mqa.value_proj, anchor)?;
    let v_seq = v
        .reshape(Shape::from_dims(&[b, kv_dim, kv_len]))?
        .permute([0, 2, 1_usize])?
        .reshape(Shape::from_dims(&[b, 1, kv_len, kv_dim]))?;
    let v_bc = v_seq.broadcast_to(Shape::from_dims(&[b, heads, kv_len, kv_dim]))?;

    // attn = q @ k^T → (B, heads, H·W, kv_len)
    let k_t = k_bc.permute([0, 1, 3, 2_usize])?;
    let scores = q.matmul(&k_t)?;
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v_bc)?; // (B, heads, H·W, kv_dim)

    // Reshape back to (B, kv_dim·heads, H, W):
    //   (B, heads, H·W, kv_dim) → permute (0, 2, 1, 3) → (B, H·W, heads, kv_dim)
    //   reshape (B, H, W, kv_dim·heads) → permute (0, 3, 1, 2) → (B, kv_dim·heads, H, W)
    let o = ctx
        .permute([0, 2, 1, 3_usize])?
        .reshape(Shape::from_dims(&[b, h, w, kv_dim * heads]))?
        .permute([0, 3, 1, 2_usize])?;

    let mut y = apply_conv_bn(&o, &mqa.output_proj, anchor)?;
    if let Some(g) = &mqa.layer_scale_gamma {
        let gt = anchor
            .const_f32_like(Arc::clone(g), Shape::from_dims(&[g.len()]))
            .reshape(Shape::from_dims(&[1, g.len(), 1, 1]))?;
        y = y.broadcast_mul(&gt)?;
    }
    if mqa.skip {
        y = y.add(x)?;
    }
    Ok(y)
}

// ---- HuggingFace safetensors loading ---------------------------------------

/// timm BatchNorm eps used throughout the MobileNetV4 byobnet config.
const MV4_BN_EPS: f64 = 1e-5;

impl Mv4Weights {
    /// Load MobileNetV4 weights from a timm-format safetensors checkpoint.
    /// Layout (top-level):
    ///
    /// - Stem: `conv_stem.weight` (`[stem_dim, 3, 3, 3]`),
    ///   `bn1.{weight,bias,running_mean,running_var}` (`[stem_dim]`).
    /// - Blocks at `blocks.{stage}.{block}` per the `Mv4Config::stages`:
    ///   - `Convolutional`: `conv.weight` + `bn1.*`.
    ///   - `UniversalBottleneck`: optional `dw_start.{conv.weight,bn.*}`,
    ///     `pw_exp.{conv.weight,bn.*}`, optional `dw_mid.{conv.weight,bn.*}`,
    ///     `pw_proj.{conv.weight,bn.*}`, optional `layer_scale.gamma`.
    ///   - `EdgeResidual`: `conv_exp.weight` + `bn1.*`,
    ///     `conv_pwl.weight` + `bn2.*`.
    ///   - `Attention`: `norm.*`, then under `attn.`: `query.proj.weight`,
    ///     optional `key.{down_conv.weight,norm.*}`, `key.proj.weight`,
    ///     optional `value.{down_conv.weight,norm.*}`, `value.proj.weight`,
    ///     `output.proj.weight`. Optional `layer_scale.gamma`.
    /// - Head (when `with_head == true`): `conv_head.weight` + `norm_head.*`,
    ///   `classifier.{weight,bias}` (`[nclasses, head_out_channels]` →
    ///   transposed to `[in, out]`).
    ///
    /// BN parameters fold into the layer's per-channel affine at load
    /// time via [`BatchNormParams::from_raw`].
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &Mv4Config,
        with_head: Option<usize>,
    ) -> crate::Result<Self> {
        // Stem.
        let stem_w = mv4_load_check(st, "conv_stem.weight",
            cfg.stem_dim * 3 * 3 * 3)?;
        let stem_bn = mv4_load_bn(st, "bn1", cfg.stem_dim)?;
        let stem = Conv2dBnWeights {
            w: Arc::from(stem_w),
            bn: stem_bn,
            c_in: 3, c_out: cfg.stem_dim,
            k: 3, stride: 2, pad: 1, groups: 1,
        };

        // Blocks.
        let mut blocks: Vec<BlockWeights> = Vec::new();
        let mut in_ch = cfg.stem_dim;
        for (stage_idx, stage) in cfg.stages.iter().enumerate() {
            for (block_idx, spec) in stage.iter().enumerate() {
                let prefix = format!("blocks.{stage_idx}.{block_idx}");
                let (bw, next_in) = mv4_load_block(st, &prefix, spec, in_ch)?;
                blocks.push(bw);
                in_ch = next_in;
            }
        }

        // Head.
        let head = if let Some(nclasses) = with_head {
            let conv_w = mv4_load_check(
                st, "conv_head.weight",
                cfg.head_out_channels * cfg.head_in_channels,
            )?;
            let bn = mv4_load_bn(st, "norm_head", cfg.head_out_channels)?;
            let conv = Conv2dBnWeights {
                w: Arc::from(conv_w),
                bn,
                c_in: cfg.head_in_channels,
                c_out: cfg.head_out_channels,
                k: 1, stride: 1, pad: 0, groups: 1,
            };
            let linear_w_t = mv4_load_transposed(
                st, "classifier.weight", nclasses, cfg.head_out_channels,
            )?;
            let linear_b = mv4_load_check(st, "classifier.bias", nclasses)?;
            Some(Mv4HeadWeights {
                conv,
                linear_w: WeightStorage::F32(Arc::from(linear_w_t)),
                linear_b: Arc::from(linear_b),
            })
        } else {
            None
        };

        Ok(Self { stem, blocks, head })
    }
}

fn mv4_load_block(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    spec: &BlockSpec,
    in_ch: usize,
) -> crate::Result<(BlockWeights, usize)> {
    match *spec {
        BlockSpec::Convolutional { out_channels, kernel, stride } => {
            let pad = kernel / 2;
            let conv = mv4_load_conv_bn(
                st, &format!("{prefix}.conv"), &format!("{prefix}.bn1"),
                in_ch, out_channels, kernel, stride, pad, 1,
            )?;
            Ok((BlockWeights::Convolutional(conv), out_channels))
        }
        BlockSpec::EdgeResidual { out_channels, kernel, stride, expand } => {
            let mid = in_ch * expand;
            let pad = kernel / 2;
            let conv_exp = mv4_load_conv_bn(
                st, &format!("{prefix}.conv_exp"), &format!("{prefix}.bn1"),
                in_ch, mid, kernel, stride, pad, 1,
            )?;
            let conv_pwl = mv4_load_conv_bn(
                st, &format!("{prefix}.conv_pwl"), &format!("{prefix}.bn2"),
                mid, out_channels, 1, 1, 0, 1,
            )?;
            Ok((BlockWeights::EdgeResidual(EdgeResidualWeights { conv_exp, conv_pwl }), out_channels))
        }
        BlockSpec::UniversalBottleneck {
            out_channels, start_kernel, mid_kernel, stride, expand,
        } => {
            let mid = in_ch * expand;
            let dw_start_stride = if mid_kernel > 0 { 1 } else { stride };
            let dw_start = if start_kernel > 0 {
                Some(mv4_load_conv_bn(
                    st,
                    &format!("{prefix}.dw_start.conv"),
                    &format!("{prefix}.dw_start.bn"),
                    in_ch, in_ch, start_kernel, dw_start_stride, start_kernel / 2, in_ch,
                )?)
            } else { None };
            let pw_exp = mv4_load_conv_bn(
                st, &format!("{prefix}.pw_exp.conv"), &format!("{prefix}.pw_exp.bn"),
                in_ch, mid, 1, 1, 0, 1,
            )?;
            let dw_mid = if mid_kernel > 0 {
                Some(mv4_load_conv_bn(
                    st,
                    &format!("{prefix}.dw_mid.conv"),
                    &format!("{prefix}.dw_mid.bn"),
                    mid, mid, mid_kernel, stride, mid_kernel / 2, mid,
                )?)
            } else { None };
            let pw_proj = mv4_load_conv_bn(
                st, &format!("{prefix}.pw_proj.conv"), &format!("{prefix}.pw_proj.bn"),
                mid, out_channels, 1, 1, 0, 1,
            )?;
            let layer_scale_gamma = if st.get(&format!("{prefix}.layer_scale.gamma")).is_ok() {
                let g = mv4_load_check(st, &format!("{prefix}.layer_scale.gamma"), out_channels)?;
                Some(Arc::<[f32]>::from(g))
            } else {
                None
            };
            let skip = in_ch == out_channels && stride == 1;
            Ok((BlockWeights::UniversalBottleneck(UibWeights {
                dw_start, pw_exp, dw_mid, pw_proj, layer_scale_gamma, skip,
            }), out_channels))
        }
        BlockSpec::Attention {
            out_channels, heads, kernel, stride, kv_dim, kv_stride,
        } => {
            if stride != 1 {
                return Err(crate::Error::Msg(
                    "Mobile-MQA v1: stride must be 1".into(),
                ).bt());
            }
            // Input residual BN: in eager Fuel the BN dim is `out_channels`
            // (relies on in_channels == out_channels for Attention blocks).
            let input_norm = mv4_load_bn(
                st, &format!("{prefix}.norm"), out_channels,
            )?;
            let query_proj = mv4_load_proj(
                st, &format!("{prefix}.attn.query.proj"),
                in_ch, kv_dim * heads,
            )?;
            let (key_down, value_down) = if kv_stride > 1 {
                let kd = mv4_load_dw(
                    st,
                    &format!("{prefix}.attn.key.down_conv"),
                    &format!("{prefix}.attn.key.norm"),
                    in_ch, kernel, kv_stride,
                )?;
                let vd = mv4_load_dw(
                    st,
                    &format!("{prefix}.attn.value.down_conv"),
                    &format!("{prefix}.attn.value.norm"),
                    in_ch, kernel, kv_stride,
                )?;
                (Some(kd), Some(vd))
            } else {
                (None, None)
            };
            let key_proj = mv4_load_proj(
                st, &format!("{prefix}.attn.key.proj"), in_ch, kv_dim,
            )?;
            let value_proj = mv4_load_proj(
                st, &format!("{prefix}.attn.value.proj"), in_ch, kv_dim,
            )?;
            let output_proj = mv4_load_proj(
                st, &format!("{prefix}.attn.output.proj"),
                kv_dim * heads, out_channels,
            )?;
            let layer_scale_gamma = if st.get(&format!("{prefix}.layer_scale.gamma")).is_ok() {
                let g = mv4_load_check(st, &format!("{prefix}.layer_scale.gamma"), out_channels)?;
                Some(Arc::<[f32]>::from(g))
            } else {
                None
            };
            let skip = in_ch == out_channels;
            Ok((BlockWeights::Attention(MqaWeights {
                input_norm, query_proj, key_down, key_proj,
                value_down, value_proj, output_proj,
                heads, kv_dim, layer_scale_gamma, skip,
            }), out_channels))
        }
    }
}

/// Load a `conv2d_no_bias` + `batch_norm` pair into a `Conv2dBnWeights`,
/// baking BN at load time.
#[allow(clippy::too_many_arguments)]
fn mv4_load_conv_bn(
    st: &crate::safetensors::MmapedSafetensors,
    conv_prefix: &str,
    bn_prefix: &str,
    c_in: usize,
    c_out: usize,
    k: usize,
    stride: usize,
    pad: usize,
    groups: usize,
) -> crate::Result<Conv2dBnWeights> {
    let w = mv4_load_check(
        st, &format!("{conv_prefix}.weight"),
        c_out * (c_in / groups) * k * k,
    )?;
    let bn = mv4_load_bn(st, bn_prefix, c_out)?;
    Ok(Conv2dBnWeights {
        w: Arc::from(w),
        bn,
        c_in, c_out, k, stride, pad, groups,
    })
}

/// MQA 1×1 projection (no stride/pad). Uses BN that fuses into the
/// per-channel affine.
fn mv4_load_proj(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c_in: usize,
    c_out: usize,
) -> crate::Result<Conv2dBnWeights> {
    // Eager Fuel uses `conv2d_no_bias` for the MQA projections WITHOUT
    // a BN; the lazy Conv2dBnWeights still carries a BN, so we bake an
    // identity BN (gain=1, bias=0, mean=0, var=1) here. The lazy
    // `apply_conv_bn` helper assumes BN always exists.
    let w = mv4_load_check(st, &format!("{prefix}.weight"), c_out * c_in)?;
    let identity_bn = BatchNormParams {
        w: Arc::from(vec![1.0_f32; c_out]),
        b: Arc::from(vec![0.0_f32; c_out]),
    };
    Ok(Conv2dBnWeights {
        w: Arc::from(w),
        bn: identity_bn,
        c_in, c_out,
        k: 1, stride: 1, pad: 0, groups: 1,
    })
}

/// MQA depthwise downsample (`groups = in_channels`) with BN.
fn mv4_load_dw(
    st: &crate::safetensors::MmapedSafetensors,
    conv_prefix: &str,
    bn_prefix: &str,
    channels: usize,
    kernel: usize,
    stride: usize,
) -> crate::Result<Conv2dBnWeights> {
    let w = mv4_load_check(
        st, &format!("{conv_prefix}.weight"),
        channels * kernel * kernel,
    )?;
    let bn = mv4_load_bn(st, bn_prefix, channels)?;
    Ok(Conv2dBnWeights {
        w: Arc::from(w),
        bn,
        c_in: channels, c_out: channels,
        k: kernel, stride, pad: kernel / 2, groups: channels,
    })
}

fn mv4_load_bn(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    channels: usize,
) -> crate::Result<BatchNormParams> {
    let gain = mv4_load_check(st, &format!("{prefix}.weight"), channels)?;
    let bias = mv4_load_check(st, &format!("{prefix}.bias"),   channels)?;
    let mean = mv4_load_check(st, &format!("{prefix}.running_mean"), channels)?;
    let var  = mv4_load_check(st, &format!("{prefix}.running_var"),  channels)?;
    Ok(BatchNormParams::from_raw(&gain, &bias, &mean, &var, MV4_BN_EPS))
}

fn mv4_load_check(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    expected_len: usize,
) -> crate::Result<Vec<f32>> {
    let v = load_tensor_as_f32(st, name)?;
    if v.len() != expected_len {
        return Err(crate::Error::Msg(format!(
            "MobileNetV4 load {name:?}: got {} elements, expected {}",
            v.len(), expected_len,
        ))
        .bt());
    }
    Ok(v)
}

fn mv4_load_transposed(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> crate::Result<Vec<f32>> {
    let flat = mv4_load_check(st, name, out_features * in_features)?;
    let mut out = vec![0.0_f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            out[j * out_features + i] = flat[i * in_features + j];
        }
    }
    Ok(out)
}

impl Mv4Model {
    /// Download a timm-format MobileNetV4 safetensors checkpoint and load
    /// it into a model. `nclasses` selects whether the classifier head is
    /// loaded.
    pub fn from_hub_with_config(
        repo_id: &str, config: Mv4Config, nclasses: Option<usize>,
    ) -> Result<Self> {
        Self::from_hub_with_filename(repo_id, "model.safetensors", config, nclasses)
    }

    /// Explicit-filename variant.
    pub fn from_hub_with_filename(
        repo_id: &str,
        filename: &str,
        config: Mv4Config,
        nclasses: Option<usize>,
    ) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let weights_path = repo
            .get(filename)
            .map_err(|e| crate::Error::Msg(format!("hf-hub mv4 safetensors: {e}")))?;
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&weights_path) }?;
        let weights = Mv4Weights::load_from_mmapped(&st, &config, nclasses)?;
        Ok(Self { config, weights })
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
            BlockSpec::Attention {
                out_channels, heads, kernel, stride, kv_dim, kv_stride,
            } => {
                assert_eq!(stride, 1, "Mobile-MQA v1: stride must be 1");
                let key_down = if kv_stride > 1 {
                    Some(conv_bn_w(in_ch, in_ch, kernel, kv_stride, kernel / 2, in_ch, nb))
                } else { None };
                let value_down = if kv_stride > 1 {
                    Some(conv_bn_w(in_ch, in_ch, kernel, kv_stride, kernel / 2, in_ch, nb))
                } else { None };
                BlockWeights::Attention(MqaWeights {
                    input_norm: BatchNormParams {
                        w: Arc::from(vec![1.0_f32; in_ch]),
                        b: Arc::from(vec![0.0_f32; in_ch]),
                    },
                    query_proj: conv_bn_w(in_ch, kv_dim * heads, 1, 1, 0, 1, nb),
                    key_down,
                    key_proj: conv_bn_w(in_ch, kv_dim, 1, 1, 0, 1, nb),
                    value_down,
                    value_proj: conv_bn_w(in_ch, kv_dim, 1, 1, 0, 1, nb),
                    output_proj: conv_bn_w(kv_dim * heads, out_channels, 1, 1, 0, 1, nb),
                    heads,
                    kv_dim,
                    layer_scale_gamma: None,
                    skip: in_ch == out_channels,
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
                    BlockSpec::Attention { out_channels, .. } => *out_channels,
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

    /// Tiny config with a Mobile-MQA Attention block. Exercise both
    /// the `kv_stride > 1` (key/value depthwise downsample) and
    /// `kv_stride == 1` (direct) paths.
    fn tiny_hybrid_config(kv_stride: usize) -> Mv4Config {
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
                    Attention { out_channels: 16, heads: 2, kernel: 3, stride: 1, kv_dim: 4, kv_stride },
                ],
                vec![
                    EdgeResidual { out_channels: 24, kernel: 3, stride: 2, expand: 2 },
                ],
                vec![Convolutional { out_channels: 32, kernel: 1, stride: 1 }],
            ],
        }
    }

    #[test]
    fn mqa_kv_stride_2_runs() {
        let cfg = tiny_hybrid_config(2);
        let weights = build_weights(&cfg);
        let model = Mv4Model { config: cfg.clone(), weights };
        let img = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let pooled = model.forward(&img).unwrap();
        assert_eq!(pooled.shape().dims(), &[1, cfg.head_in_channels]);
        for &v in &pooled.realize_f32() {
            assert!(v.is_finite(), "kv_stride=2 produced non-finite pooled: {v}");
        }
    }

    #[test]
    fn mqa_kv_stride_1_runs() {
        let cfg = tiny_hybrid_config(1);
        let weights = build_weights(&cfg);
        let model = Mv4Model { config: cfg.clone(), weights };
        let img = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let pooled = model.forward(&img).unwrap();
        assert_eq!(pooled.shape().dims(), &[1, cfg.head_in_channels]);
        for &v in &pooled.realize_f32() {
            assert!(v.is_finite(), "kv_stride=1 produced non-finite pooled: {v}");
        }
    }

    /// Different input images must yield different pooled outputs
    /// when an Attention block is present.
    #[test]
    fn mqa_responds_to_input() {
        let cfg = tiny_hybrid_config(2);
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
        assert!(max_diff > 1e-9,
            "Mobile-MQA path must respond to input changes, max_diff = {max_diff}");
    }

    // ---- load_from_mmapped round-trip ---------------------------------------

    fn raw_f32(len: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        let mut out = Vec::with_capacity(len * 4);
        for _ in 0..len {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let v = ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05;
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn raw_f32_const(len: usize, value: f32) -> Vec<u8> {
        let mut out = Vec::with_capacity(len * 4);
        for _ in 0..len {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn push_identity_bn(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        channels: usize,
    ) {
        for (suffix, raw) in [
            ("weight",       raw_f32_const(channels, 1.0)),
            ("bias",         raw_f32_const(channels, 0.0)),
            ("running_mean", raw_f32_const(channels, 0.0)),
            ("running_var",  raw_f32_const(channels, 1.0)),
        ] {
            owned.push((format!("{prefix}.{suffix}"), vec![channels], raw));
        }
    }

    /// Build a tiny synthetic MV4 safetensors blob for `tiny_config()`
    /// (no Attention, only Conv / EdgeResidual / UniversalBottleneck),
    /// load it via the loader, and verify the stem fused conv weight
    /// equals the raw conv weight (since identity BN ⇒ scale = 1/√(1+eps)).
    #[test]
    fn load_from_mmapped_round_trip_mv4_tiny_no_head() {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;

        let cfg = tiny_config();
        let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();

        // Stem.
        owned.push((
            "conv_stem.weight".into(),
            vec![cfg.stem_dim, 3, 3, 3],
            raw_f32(cfg.stem_dim * 3 * 9, 0xC0FFEE),
        ));
        push_identity_bn(&mut owned, "bn1", cfg.stem_dim);

        let mut in_ch = cfg.stem_dim;
        for (stage_idx, stage) in cfg.stages.iter().enumerate() {
            for (block_idx, spec) in stage.iter().enumerate() {
                let prefix = format!("blocks.{stage_idx}.{block_idx}");
                in_ch = build_block_tensors(&mut owned, &prefix, *spec, in_ch);
            }
        }

        // Serialize.
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        for (name, shape, bytes) in &owned {
            let view = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                .expect("TensorView::new");
            tensors.insert(name.clone(), view);
        }
        let metadata: Option<HashMap<String, String>> = None;
        let serialized = safetensors::serialize(&tensors, metadata)
            .expect("safetensors::serialize");

        let tmp = std::env::temp_dir().join(format!(
            "fuel_mv4_load_test_{}.safetensors", std::process::id(),
        ));
        std::fs::write(&tmp, &serialized).expect("write tmp");
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&tmp) }
            .expect("MmapedSafetensors::new");

        let loaded = Mv4Weights::load_from_mmapped(&st, &cfg, /* with_head = */ None)
            .expect("Mv4Weights::load_from_mmapped");

        // Stem conv weight matches the raw bytes (BN is identity ⇒ no scaling)
        // exactly; verify each element matches.
        let raw_stem = &owned.iter()
            .find(|(n, _, _)| n == "conv_stem.weight").unwrap().2;
        let raw_stem_f: Vec<f32> = raw_stem.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for (i, expected) in raw_stem_f.iter().enumerate() {
            let got = loaded.stem.w[i];
            assert!((got - expected).abs() < 1e-6,
                "stem conv weight[{i}]: expected {expected}, got {got}");
        }
        // BN gain/bias fused from (gain=1, bias=0, mean=0, var=1, eps=1e-5)
        // is `w = 1/√(1+eps)`, `b = 0`. The model APPLIES BN as a per-channel
        // affine, so verifying `bn.w` is the right scalar covers it.
        let bn_scale = 1.0_f32 / (1.0_f32 + MV4_BN_EPS as f32).sqrt();
        for c in 0..cfg.stem_dim {
            assert!((loaded.stem.bn.w[c] - bn_scale).abs() < 1e-6,
                "stem.bn.w[{c}] expected ~{bn_scale}, got {}", loaded.stem.bn.w[c]);
            assert!(loaded.stem.bn.b[c].abs() < 1e-6);
        }

        // Block count + structure.
        let total_blocks: usize = cfg.stages.iter().map(|s| s.len()).sum();
        assert_eq!(loaded.blocks.len(), total_blocks);
        assert!(loaded.head.is_none());

        // Forward chain runs end-to-end.
        let model = Mv4Model { config: cfg.clone(), weights: loaded };
        let img = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let pooled = model.forward(&img).unwrap();
        assert_eq!(pooled.shape().dims(), &[1, cfg.head_in_channels]);

        let _ = std::fs::remove_file(&tmp);
    }

    /// Walk one block spec and push the synthetic tensors it requires
    /// into `owned`. Returns the layer's output channel count for the
    /// next iteration's `in_ch` chaining.
    fn build_block_tensors(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        spec: BlockSpec,
        in_ch: usize,
    ) -> usize {
        match spec {
            BlockSpec::Convolutional { out_channels, kernel, stride: _ } => {
                let _pad = kernel / 2;
                owned.push((
                    format!("{prefix}.conv.weight"),
                    vec![out_channels, in_ch, kernel, kernel],
                    raw_f32(out_channels * in_ch * kernel * kernel, 1),
                ));
                push_identity_bn(owned, &format!("{prefix}.bn1"), out_channels);
                out_channels
            }
            BlockSpec::EdgeResidual { out_channels, kernel, stride: _, expand } => {
                let mid = in_ch * expand;
                owned.push((
                    format!("{prefix}.conv_exp.weight"),
                    vec![mid, in_ch, kernel, kernel],
                    raw_f32(mid * in_ch * kernel * kernel, 2),
                ));
                push_identity_bn(owned, &format!("{prefix}.bn1"), mid);
                owned.push((
                    format!("{prefix}.conv_pwl.weight"),
                    vec![out_channels, mid, 1, 1],
                    raw_f32(out_channels * mid, 3),
                ));
                push_identity_bn(owned, &format!("{prefix}.bn2"), out_channels);
                out_channels
            }
            BlockSpec::UniversalBottleneck {
                out_channels, start_kernel, mid_kernel, stride: _, expand,
            } => {
                let mid = in_ch * expand;
                if start_kernel > 0 {
                    owned.push((
                        format!("{prefix}.dw_start.conv.weight"),
                        vec![in_ch, 1, start_kernel, start_kernel],
                        raw_f32(in_ch * start_kernel * start_kernel, 4),
                    ));
                    push_identity_bn(owned, &format!("{prefix}.dw_start.bn"), in_ch);
                }
                owned.push((
                    format!("{prefix}.pw_exp.conv.weight"),
                    vec![mid, in_ch, 1, 1],
                    raw_f32(mid * in_ch, 5),
                ));
                push_identity_bn(owned, &format!("{prefix}.pw_exp.bn"), mid);
                if mid_kernel > 0 {
                    owned.push((
                        format!("{prefix}.dw_mid.conv.weight"),
                        vec![mid, 1, mid_kernel, mid_kernel],
                        raw_f32(mid * mid_kernel * mid_kernel, 6),
                    ));
                    push_identity_bn(owned, &format!("{prefix}.dw_mid.bn"), mid);
                }
                owned.push((
                    format!("{prefix}.pw_proj.conv.weight"),
                    vec![out_channels, mid, 1, 1],
                    raw_f32(out_channels * mid, 7),
                ));
                push_identity_bn(owned, &format!("{prefix}.pw_proj.bn"), out_channels);
                out_channels
            }
            BlockSpec::Attention { .. } => {
                // Not exercised in this round-trip test; the tiny_config()
                // used above intentionally avoids Attention blocks. The
                // loader's Attention path is structurally identical to the
                // others, just with more name probing.
                panic!("Attention block not exercised in load_from_mmapped tiny round-trip");
            }
        }
    }

    /// Smoke test that documents the canonical `from_hub_with_config`
    /// usage. Ignored because it hits the HF Hub.
    #[test]
    #[ignore]
    fn from_hub_smoke_mv4_conv_small() {
        let cfg = Mv4Config::conv_small();
        let model = Mv4Model::from_hub_with_config(
            "timm/mobilenetv4_conv_small.e2400_r224_in1k",
            cfg,
            Some(1000),
        ).expect("from_hub_with_config");
        assert!(model.weights.head.is_some());
    }
}

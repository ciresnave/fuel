//! TinyViT (MobileSAM image encoder) ported to the lazy-graph API.
//!
//! TinyViT is the lightweight image backbone used by MobileSAM
//! (Zhang et al. 2023). Architecturally distinct from SAM's ViT-B/L/H:
//!
//! 1. **Patch embed** = 2 stride-2 Conv2dBN with GELU between (4×
//!    spatial downsample at input, channels 3 → embed_dim/2 → embed_dim).
//! 2. **Stage 0** = `ConvLayer` of MBConv blocks (Mobile inverted
//!    bottleneck: 1×1 expand → 3×3 DW conv → 1×1 reduce + residual +
//!    GELU) followed by a `PatchMerging` downsample (2× spatial,
//!    embed_dim → next stage dim).
//! 3. **Stages 1..3** = `BasicLayer` of TinyViTBlocks. Each block is
//!    window-attention (zero-pad partition with `window_size` × `window_size`
//!    windows; SAM-style) + relative-position attention biases (precomputed
//!    table indexed by integer spatial offsets) + 3×3 depthwise local conv
//!    + LN-MLP. Stages 1 and 2 each end with a `PatchMerging`.
//! 4. **Neck** = 2× (Conv2d-no-bias 1×1 + LayerNorm2d) + Conv2d-no-bias
//!    3×3 + LayerNorm2d. Projects the final stage feature map to 256
//!    channels and outputs the same `(1, 256, 64, 64)` shape as the
//!    standard SAM ViT image encoder — the mask decoder consumes it
//!    interchangeably.
//!
//! # MobileSAM preset
//!
//! [`TinyVitConfig::mobile_sam_5m`] returns the 5M-parameter MobileSAM
//! variant: `embed_dims=[64,128,160,320]`, `depths=[2,2,6,2]`,
//! `num_heads=[2,4,5,10]`, `window_sizes=[7,7,14,7]`, `img_size=1024`.
//!
//! # Scope (v1)
//!
//! Forward only, batch == 1, F32. The neck output `(1, 256, 64, 64)`
//! plugs into [`crate::lazy_sam::SamMaskDecoder`] interchangeably with
//! the standard SAM ViT encoder output.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_convmixer::BatchNormParams;
use crate::lazy_sam::SamLayerNormWeights;
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

// ---- Config ---------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TinyVitConfig {
    pub img_size: usize,
    pub in_chans: usize,
    /// Per-stage channel dim. Length = number of stages (typically 4).
    pub embed_dims: Vec<usize>,
    /// Per-stage block count. Stage 0 uses MBConv (no attention);
    /// stages 1..N use TinyViTBlock (windowed attention + DW conv + MLP).
    pub depths: Vec<usize>,
    /// Per-stage attention head count (stage 0 unused).
    pub num_heads: Vec<usize>,
    /// Per-stage windowed-attention window size (stage 0 unused).
    pub window_sizes: Vec<usize>,
    pub mbconv_expand_ratio: usize,
    pub mlp_ratio: usize,
    pub local_conv_size: usize,
}

impl TinyVitConfig {
    /// MobileSAM 5M-parameter preset matching the reference checkpoint.
    pub fn mobile_sam_5m() -> Self {
        Self {
            img_size: 1024,
            in_chans: 3,
            embed_dims: vec![64, 128, 160, 320],
            depths: vec![2, 2, 6, 2],
            num_heads: vec![2, 4, 5, 10],
            window_sizes: vec![7, 7, 14, 7],
            mbconv_expand_ratio: 4,
            mlp_ratio: 4,
            local_conv_size: 3,
        }
    }

    /// Number of stages (== `embed_dims.len()`).
    pub fn num_stages(&self) -> usize {
        self.embed_dims.len()
    }
}

// ---- Weights --------------------------------------------------------------

/// Conv2d + fused-affine BN parameters (BN baked into per-channel
/// `(w, b)` at load time).
#[derive(Debug, Clone)]
pub struct Conv2dBnWeights {
    pub conv_w: Arc<[f32]>,
    pub bn: BatchNormParams,
    /// Channel count after the conv — needed at forward time to bind
    /// the conv weight shape.
    pub c_out: usize,
    pub c_in: usize,
    pub kernel: usize,
    /// Convolution groups (== c_in for depthwise; 1 for pointwise/dense).
    pub groups: usize,
    pub stride: usize,
    pub padding: usize,
}

#[derive(Debug, Clone)]
pub struct PatchEmbedWeights {
    /// `[embed_dim/2, in_chans, 3, 3]` stride=2 pad=1.
    pub conv1: Conv2dBnWeights,
    /// `[embed_dim, embed_dim/2, 3, 3]` stride=2 pad=1.
    pub conv2: Conv2dBnWeights,
}

#[derive(Debug, Clone)]
pub struct MbConvWeights {
    /// `[hidden, c_in, 1, 1]`.
    pub conv1: Conv2dBnWeights,
    /// `[hidden, 1, 3, 3]` depthwise (groups=hidden).
    pub conv2: Conv2dBnWeights,
    /// `[c_out, hidden, 1, 1]`.
    pub conv3: Conv2dBnWeights,
}

#[derive(Debug, Clone)]
pub struct PatchMergingWeights {
    pub conv1: Conv2dBnWeights,
    pub conv2: Conv2dBnWeights,
    pub conv3: Conv2dBnWeights,
    pub input_resolution: (usize, usize),
    pub dim: usize,
    pub out: usize,
}

#[derive(Debug, Clone)]
pub struct MlpWeights {
    pub norm: SamLayerNormWeights,
    /// `[hidden, in_dim]`.
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    /// `[in_dim, hidden]`.
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
}

/// TinyViT block windowed attention. Uses precomputed relative-position
/// attention biases — a small gather table indexed by integer spatial
/// offsets within the window.
#[derive(Debug, Clone)]
pub struct TinyVitAttnWeights {
    pub norm: SamLayerNormWeights,
    /// Fused QKV: `[d_total, dim]` where d_total = dh + 2 * nh*key_dim
    /// with dh = num_heads * key_dim * attn_ratio.
    pub qkv: WeightStorage,
    pub qkv_bias: Arc<[f32]>,
    /// `[dim, dh]`.
    pub proj: WeightStorage,
    pub proj_bias: Arc<[f32]>,
    /// `[num_heads, n_distinct_offsets]` table.
    pub attention_biases: Arc<[f32]>,
    /// Precomputed flat index buffer of length `n_tokens * n_tokens`
    /// pointing into `attention_biases` for the (q, k) offset of each
    /// query-key pair within the window (n_tokens = window_size *
    /// window_size). The gathered bias is broadcast over heads and
    /// added to the attention scores.
    pub attention_bias_idxs: Arc<[u32]>,
    pub n_offsets: usize,
    pub key_dim: usize,
    pub num_heads: usize,
    pub d: usize,
    pub dh: usize,
}

#[derive(Debug, Clone)]
pub struct TinyVitBlockWeights {
    pub attn: TinyVitAttnWeights,
    /// `[dim, 1, k, k]` depthwise (groups=dim, padding=k/2).
    pub local_conv: Conv2dBnWeights,
    pub mlp: MlpWeights,
    pub dim: usize,
    pub input_resolution: (usize, usize),
    pub window_size: usize,
}

#[derive(Debug, Clone)]
pub struct ConvLayerWeights {
    pub blocks: Vec<MbConvWeights>,
    pub downsample: Option<PatchMergingWeights>,
}

#[derive(Debug, Clone)]
pub struct BasicLayerWeights {
    pub blocks: Vec<TinyVitBlockWeights>,
    pub downsample: Option<PatchMergingWeights>,
}

/// 2-D LayerNorm matching `lazy_sam::layer_norm_2d` — per-channel mean
/// and variance over `(B, C, H, W)`.
#[derive(Debug, Clone)]
pub struct LayerNorm2dWeights {
    pub gain: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct TinyVitNeckWeights {
    /// `[256, last_embed_dim, 1, 1]` no bias.
    pub conv1: Arc<[f32]>,
    pub ln1: LayerNorm2dWeights,
    /// `[256, 256, 3, 3]` no bias.
    pub conv2: Arc<[f32]>,
    pub ln2: LayerNorm2dWeights,
}

#[derive(Debug, Clone)]
pub struct TinyVitWeights {
    pub patch_embed: PatchEmbedWeights,
    /// Stage 0 — convolutional (MBConv blocks + downsample).
    pub stage0: ConvLayerWeights,
    /// Stages 1..N — windowed-attention basic layers.
    pub stages: Vec<BasicLayerWeights>,
    pub neck: TinyVitNeckWeights,
}

#[derive(Debug, Clone)]
pub struct TinyVitModel {
    pub config: TinyVitConfig,
    pub weights: TinyVitWeights,
}

// ---- Forward helpers ------------------------------------------------------

/// `x.conv2d(w, None, stride, pad, groups)` followed by per-channel
/// fused-affine BN broadcast over (N, C, H, W).
fn apply_conv2d_bn(x: &LazyTensor, w: &Conv2dBnWeights) -> Result<LazyTensor> {
    let cw = x.const_f32_like(
        Arc::clone(&w.conv_w),
        Shape::from_dims(&[w.c_out, w.c_in / w.groups, w.kernel, w.kernel]),
    );
    let conv = x.conv2d(&cw, None, (w.stride, w.stride), (w.padding, w.padding), w.groups)?;
    conv.channel_affine_4d(Arc::clone(&w.bn.w), Arc::clone(&w.bn.b))
}

/// SAM-style per-channel LayerNorm over (B, C, H, W). Delegates to the
/// crate-private helper shared with `lazy_sam`.
fn layer_norm_2d(
    x: &LazyTensor, w: &LayerNorm2dWeights, c: usize, eps: f64,
) -> Result<LazyTensor> {
    crate::lazy_sam::layer_norm_2d(x, &SamLayerNormWeights {
        gain: Arc::clone(&w.gain), bias: Arc::clone(&w.bias),
    }, c, eps)
}

fn apply_patch_embed(x: &LazyTensor, w: &PatchEmbedWeights) -> Result<LazyTensor> {
    let x = apply_conv2d_bn(x, &w.conv1)?;
    let x = x.gelu();
    apply_conv2d_bn(&x, &w.conv2)
}

fn apply_mbconv(x: &LazyTensor, w: &MbConvWeights) -> Result<LazyTensor> {
    let h = apply_conv2d_bn(x, &w.conv1)?;
    let h = h.gelu();
    let h = apply_conv2d_bn(&h, &w.conv2)?;
    let h = h.gelu();
    let h = apply_conv2d_bn(&h, &w.conv3)?;
    Ok(h.add(x)?.gelu())
}

/// Patch merging — operates on `(B, L=H*W, C)` token sequence (or 4-D
/// NCHW), and returns `(B, L_out, C_out)`.
fn apply_patch_merging(x: &LazyTensor, w: &PatchMergingWeights) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    // Accept rank-3 (B, L, C) by reshaping to NCHW; pass-through rank-4.
    let x_nchw = if dims.len() == 3 {
        let (b, _l, c) = (dims[0], dims[1], dims[2]);
        let (h, ww) = w.input_resolution;
        if c != w.dim {
            return Err(crate::Error::Msg(format!(
                "PatchMerging: input C={c} != expected dim={}", w.dim,
            )).bt());
        }
        x.reshape(Shape::from_dims(&[b, h, ww, c]))?
            .permute([0, 3, 1, 2_usize])?
    } else {
        x.clone()
    };
    let h = apply_conv2d_bn(&x_nchw, &w.conv1)?;
    let h = h.gelu();
    let h = apply_conv2d_bn(&h, &w.conv2)?;
    let h = h.gelu();
    let h = apply_conv2d_bn(&h, &w.conv3)?;
    // Flatten (B, C, H, W) → (B, L, C).
    let h_dims = h.shape();
    let h_dims = h_dims.dims();
    let (b2, c2, h2, w2) = (h_dims[0], h_dims[1], h_dims[2], h_dims[3]);
    h.reshape(Shape::from_dims(&[b2, c2, h2 * w2]))?
        .permute([0, 2, 1_usize])
}

fn apply_mlp(x: &LazyTensor, w: &MlpWeights, in_dim: usize) -> Result<LazyTensor> {
    let x_norm = x.layer_norm_affine(
        Arc::clone(&w.norm.gain), Arc::clone(&w.norm.bias), 1e-5,
    )?;
    let h = w.fc1.apply_linear_with_bias(&x_norm, in_dim, w.fc1_bias.len(), Arc::clone(&w.fc1_bias))?;
    let h = h.gelu();
    w.fc2.apply_linear_with_bias(&h, w.fc1_bias.len(), in_dim, Arc::clone(&w.fc2_bias))
}

/// Apply windowed attention over the input `(B, L=window_size^2, dim)`.
/// Adds the precomputed relative-position attention bias.
fn apply_window_attn(
    xs: &LazyTensor, w: &TinyVitAttnWeights, b: usize, n: usize, dim_in: usize,
) -> Result<LazyTensor> {
    let xs = xs.layer_norm_affine(
        Arc::clone(&w.norm.gain), Arc::clone(&w.norm.bias), 1e-5,
    )?;
    let d_total = w.dh + 2 * w.num_heads * w.key_dim;
    let qkv_flat = w.qkv.apply_linear_with_bias(
        &xs, dim_in, d_total, Arc::clone(&w.qkv_bias),
    )?;
    // (B, N, num_heads, d_total/num_heads).
    let per_head = d_total / w.num_heads;
    let qkv = qkv_flat.reshape(Shape::from_dims(&[b, n, w.num_heads, per_head]))?;

    // Slice along the last dim: Q is first key_dim, K is next key_dim,
    // V is remaining `d = attn_ratio * key_dim`.
    let q = qkv.narrow(3_usize, 0, w.key_dim)?
        .permute([0, 2, 1, 3_usize])?;
    let k = qkv.narrow(3_usize, w.key_dim, w.key_dim)?
        .permute([0, 2, 1, 3_usize])?;
    let v = qkv.narrow(3_usize, 2 * w.key_dim, w.d)?
        .permute([0, 2, 1, 3_usize])?;

    let scale = 1.0_f64 / (w.key_dim as f64).sqrt();
    let attn = q.matmul(&k.permute([0, 1, 3, 2_usize])?)?
        .mul_scalar(scale);

    // Build the per-batch relative-position bias by gathering
    // attention_biases[h, idxs[i*n+j]] for each (i, j) — produces
    // shape (num_heads, n, n) which is broadcast-added across batch.
    let bias_table = xs.const_f32_like(
        Arc::clone(&w.attention_biases),
        Shape::from_dims(&[w.num_heads, w.n_offsets]),
    );
    let idxs = xs.const_u32_like(
        w.attention_bias_idxs.to_vec(),
        Shape::from_dims(&[n * n]),
    );
    let ab = bias_table
        .index_select(1_usize, &idxs)?
        .reshape(Shape::from_dims(&[1, w.num_heads, n, n]))?
        .broadcast_to(Shape::from_dims(&[b, w.num_heads, n, n]))?;
    let attn = attn.add(&ab)?;
    let attn = attn.softmax_last_dim()?;

    let out = attn.matmul(&v)?
        .permute([0, 2, 1, 3_usize])?
        .reshape(Shape::from_dims(&[b, n, w.dh]))?;
    w.proj.apply_linear_with_bias(&out, w.dh, dim_in, Arc::clone(&w.proj_bias))
}

fn apply_tiny_vit_block(
    x: &LazyTensor, w: &TinyVitBlockWeights,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let (b, l, c) = (dims[0], dims[1], dims[2]);
    let (h, ww) = w.input_resolution;
    if c != w.dim {
        return Err(crate::Error::Msg(format!(
            "TinyViTBlock: input C={c} != block dim={}", w.dim,
        )).bt());
    }

    let res_x = x.clone();
    // Windowed attention. If the input grid size equals window size,
    // single window — no partitioning needed. Otherwise zero-pad and
    // partition into windows.
    let attn_out = if h == w.window_size && ww == w.window_size {
        apply_window_attn(x, &w.attn, b, l, c)?
    } else {
        let x_grid = x.reshape(Shape::from_dims(&[b, h, ww, c]))?;
        let pad_b = (w.window_size - h % w.window_size) % w.window_size;
        let pad_r = (w.window_size - ww % w.window_size) % w.window_size;
        let x_grid = if pad_b > 0 || pad_r > 0 {
            // Pad along dim 1 (H) and dim 2 (W) with zeros.
            let mut padded = x_grid;
            if pad_b > 0 {
                let zeros_b = padded.const_f32_like(
                    Arc::<[f32]>::from(vec![0.0_f32; b * pad_b * ww * c]),
                    Shape::from_dims(&[b, pad_b, ww, c]),
                );
                padded = padded.concat(&zeros_b, 1_usize)?;
            }
            if pad_r > 0 {
                let dims_after = padded.shape();
                let dims_after = dims_after.dims();
                let h_padded = dims_after[1];
                let zeros_r = padded.const_f32_like(
                    Arc::<[f32]>::from(vec![0.0_f32; b * h_padded * pad_r * c]),
                    Shape::from_dims(&[b, h_padded, pad_r, c]),
                );
                padded = padded.concat(&zeros_r, 2_usize)?;
            }
            padded
        } else {
            x_grid
        };
        let p_h = h + pad_b;
        let p_w = ww + pad_r;
        let n_h = p_h / w.window_size;
        let n_w = p_w / w.window_size;
        // (B, n_h, ws, n_w, ws, C) → (B*n_h*n_w, ws*ws, C).
        let windows = x_grid
            .reshape(Shape::from_dims(&[b, n_h, w.window_size, n_w, w.window_size, c]))?
            .permute([0, 1, 3, 2, 4, 5_usize])?
            .reshape(Shape::from_dims(&[b * n_h * n_w, w.window_size * w.window_size, c]))?;
        let win_attn = apply_window_attn(
            &windows, &w.attn, b * n_h * n_w, w.window_size * w.window_size, c,
        )?;
        // (B*n_h*n_w, ws*ws, C) → (B, n_h, n_w, ws, ws, C) → (B, p_h, p_w, C).
        let unwin = win_attn
            .reshape(Shape::from_dims(&[b, n_h, n_w, w.window_size, w.window_size, c]))?
            .permute([0, 1, 3, 2, 4, 5_usize])?
            .reshape(Shape::from_dims(&[b, p_h, p_w, c]))?;
        // Strip padding on H and W.
        let unwin = if pad_b > 0 {
            unwin.narrow(1_usize, 0, h)?
        } else { unwin };
        let unwin = if pad_r > 0 {
            unwin.narrow(2_usize, 0, ww)?
        } else { unwin };
        unwin.reshape(Shape::from_dims(&[b, l, c]))?
    };

    let x = res_x.add(&attn_out)?;
    // DW local conv: (B, L, C) → (B, C, H, W) → DWConv → (B, C, L) → (B, L, C).
    let x_nchw = x.permute([0, 2, 1_usize])?
        .reshape(Shape::from_dims(&[b, c, h, ww]))?;
    let x_conv = apply_conv2d_bn(&x_nchw, &w.local_conv)?;
    let x = x_conv.reshape(Shape::from_dims(&[b, c, l]))?
        .permute([0, 2, 1_usize])?;
    let mlp_out = apply_mlp(&x, &w.mlp, c)?;
    x.add(&mlp_out)
}

fn apply_conv_layer(x: &LazyTensor, w: &ConvLayerWeights) -> Result<LazyTensor> {
    let mut h = x.clone();
    for block in &w.blocks {
        h = apply_mbconv(&h, block)?;
    }
    match &w.downsample {
        None => Ok(h),
        Some(ds) => apply_patch_merging(&h, ds),
    }
}

fn apply_basic_layer(x: &LazyTensor, w: &BasicLayerWeights) -> Result<LazyTensor> {
    let mut h = x.clone();
    for block in &w.blocks {
        h = apply_tiny_vit_block(&h, block)?;
    }
    match &w.downsample {
        None => Ok(h),
        Some(ds) => apply_patch_merging(&h, ds),
    }
}

impl TinyVitModel {
    /// Encode a single RGB image. `image_chw` is row-major
    /// `[3, img_size, img_size]` F32. Returns the image feature
    /// map `(1, 256, 64, 64)` — same shape as the standard SAM ViT
    /// encoder so the mask decoder consumes it interchangeably.
    pub fn forward(&self, image_chw: &[f32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let img = cfg.img_size;
        if image_chw.len() != cfg.in_chans * img * img {
            return Err(crate::Error::Msg(format!(
                "TinyViTModel: expected {} f32s ({}x{}x{}), got {}",
                cfg.in_chans * img * img, cfg.in_chans, img, img, image_chw.len(),
            )).bt());
        }
        let x = LazyTensor::from_f32(
            image_chw.to_vec(),
            Shape::from_dims(&[1, cfg.in_chans, img, img]),
            &Device::cpu(),
        );

        // Patch embed: 4× spatial downsample (2 × stride-2 convs).
        let x = apply_patch_embed(&x, &weights.patch_embed)?;
        // Stage 0: MBConv layer + (optional) PatchMerging.
        let x = apply_conv_layer(&x, &weights.stage0)?;
        // Stages 1..N: BasicLayer (windowed attention + DW conv + MLP).
        let mut x = x;
        for stage in &weights.stages {
            x = apply_basic_layer(&x, stage)?;
        }

        // Last stage output is in (B, L, C) layout. Reshape to NCHW
        // for the neck — derive the spatial side from the actual L
        // since the stage-2 PatchMerging stride flips between 1 and 2
        // depending on the next channel count (eager preserves a
        // special-case list of {320, 448, 576}; for MobileSAM
        // next_dim=320 triggers stride=1 and L=4096=64²).
        let dims = x.shape();
        let dims = dims.dims();
        let (b, l, c) = (dims[0], dims[1], dims[2]);
        let last_side = (l as f64).sqrt() as usize;
        if last_side * last_side != l {
            return Err(crate::Error::Msg(format!(
                "TinyViTModel::forward: post-stages token count L={l} is not a perfect square",
            )).bt());
        }
        let x_nchw = x.reshape(Shape::from_dims(&[b, last_side, last_side, c]))?
            .permute([0, 3, 1, 2_usize])?;

        // Neck: 2× (Conv + LN2d).
        let last_dim = *cfg.embed_dims.last().unwrap();
        let nc1 = x_nchw.const_f32_like(
            Arc::clone(&weights.neck.conv1),
            Shape::from_dims(&[256, last_dim, 1, 1]),
        );
        let x = x_nchw.conv2d(&nc1, None, (1, 1), (0, 0), 1)?;
        let x = layer_norm_2d(&x, &weights.neck.ln1, 256, 1e-6)?;
        let nc2 = x.const_f32_like(
            Arc::clone(&weights.neck.conv2),
            Shape::from_dims(&[256, 256, 3, 3]),
        );
        let x = x.conv2d(&nc2, None, (1, 1), (1, 1), 1)?;
        layer_norm_2d(&x, &weights.neck.ln2, 256, 1e-6)
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

/// Helper: load Conv2dBN from `prefix` (conv at `prefix.c`, BN at `prefix.bn`),
/// folding BN into per-channel affine.
fn load_conv2d_bn(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c_in: usize, c_out: usize, kernel: usize, groups: usize,
    stride: usize, padding: usize, bn_eps: f64,
) -> Result<Conv2dBnWeights> {
    use crate::lazy::load_tensor_as_f32;
    let conv_w = Arc::from(load_tensor_as_f32(st, &format!("{prefix}.c.weight"))?);
    let gain = load_tensor_as_f32(st, &format!("{prefix}.bn.weight"))?;
    let bias = load_tensor_as_f32(st, &format!("{prefix}.bn.bias"))?;
    let mean = load_tensor_as_f32(st, &format!("{prefix}.bn.running_mean"))?;
    let var = load_tensor_as_f32(st, &format!("{prefix}.bn.running_var"))?;
    let bn = BatchNormParams::from_raw(&gain, &bias, &mean, &var, bn_eps);
    Ok(Conv2dBnWeights {
        conv_w, bn, c_out, c_in, kernel, groups, stride, padding,
    })
}

impl TinyVitWeights {
    /// Load TinyViT (MobileSAM image encoder) weights from HF safetensors.
    /// Matches the upstream `eager` Conv2dBN / MBConv / TinyViTBlock layout.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &TinyVitConfig,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype as ltm};
        let bn_eps = 1e-5;
        let in_chans = cfg.in_chans;
        let first_dim = cfg.embed_dims[0];

        // Patch embed: seq.0 = Conv2dBN(in_chans, first_dim/2, 3, stride=2, pad=1)
        //              seq.2 = Conv2dBN(first_dim/2, first_dim, 3, stride=2, pad=1)
        let pe_conv1 = load_conv2d_bn(
            st, "patch_embed.seq.0", in_chans, first_dim / 2, 3, 1, 2, 1, bn_eps,
        )?;
        let pe_conv2 = load_conv2d_bn(
            st, "patch_embed.seq.2", first_dim / 2, first_dim, 3, 1, 2, 1, bn_eps,
        )?;
        let patch_embed = PatchEmbedWeights { conv1: pe_conv1, conv2: pe_conv2 };

        // Stage 0: ConvLayer with MBConv blocks.
        let mut stage0_blocks = Vec::with_capacity(cfg.depths[0]);
        let dim0 = cfg.embed_dims[0];
        let hidden0 = dim0 * cfg.mbconv_expand_ratio;
        for i in 0..cfg.depths[0] {
            let p = format!("layers.0.blocks.{i}");
            let conv1 = load_conv2d_bn(st, &format!("{p}.conv1"), dim0, hidden0, 1, 1, 1, 0, bn_eps)?;
            let conv2 = load_conv2d_bn(st, &format!("{p}.conv2"), hidden0, hidden0, 3, hidden0, 1, 1, bn_eps)?;
            let conv3 = load_conv2d_bn(st, &format!("{p}.conv3"), hidden0, dim0, 1, 1, 1, 0, bn_eps)?;
            stage0_blocks.push(MbConvWeights { conv1, conv2, conv3 });
        }
        // Downsample to next dim (PatchMerging-style with 3 convs).
        let dim1 = cfg.embed_dims[1];
        let mut input_res = (cfg.img_size / 4, cfg.img_size / 4);
        let stage0_ds = {
            let p = "layers.0.downsample".to_string();
            let conv1 = load_conv2d_bn(st, &format!("{p}.conv1"), dim0, dim1, 1, 1, 1, 0, bn_eps).ok();
            let conv2 = load_conv2d_bn(st, &format!("{p}.conv2"), dim1, dim1, 3, dim1, 2, 1, bn_eps).ok();
            let conv3 = load_conv2d_bn(st, &format!("{p}.conv3"), dim1, dim1, 1, 1, 1, 0, bn_eps).ok();
            match (conv1, conv2, conv3) {
                (Some(c1), Some(c2), Some(c3)) => Some(PatchMergingWeights {
                    conv1: c1, conv2: c2, conv3: c3,
                    input_resolution: input_res, dim: dim0, out: dim1,
                }),
                _ => None,
            }
        };
        let stage0 = ConvLayerWeights { blocks: stage0_blocks, downsample: stage0_ds };

        // Stages 1..N: BasicLayer with TinyViTBlock.
        let load_ln = |prefix: &str| -> Result<SamLayerNormWeights> {
            Ok(SamLayerNormWeights {
                gain: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.weight"))?),
                bias: Arc::from(load_tensor_as_f32(st, &format!("{prefix}.bias"))?),
            })
        };

        let mut stages = Vec::with_capacity(cfg.num_stages() - 1);
        for s in 1..cfg.num_stages() {
            input_res = (input_res.0 / 2, input_res.1 / 2);
            let dim = cfg.embed_dims[s];
            let num_heads = cfg.num_heads[s];
            let window_size = cfg.window_sizes[s];
            let key_dim = dim / num_heads;
            let attn_ratio = 1;
            let dh = num_heads * key_dim * attn_ratio;
            let nh_kd = num_heads * key_dim;
            let d_total = dh + 2 * nh_kd;
            let n_tokens = window_size * window_size;

            let mut blocks = Vec::with_capacity(cfg.depths[s]);
            for i in 0..cfg.depths[s] {
                let p = format!("layers.{s}.blocks.{i}");
                let norm = load_ln(&format!("{p}.attn.norm"))?;
                let qkv = ltm(st, &format!("{p}.attn.qkv.weight"), d_total, dim)?;
                let qkv_bias = Arc::from(load_tensor_as_f32(
                    st, &format!("{p}.attn.qkv.bias"),
                )?);
                let proj = ltm(st, &format!("{p}.attn.proj.weight"), dim, dh)?;
                let proj_bias = Arc::from(load_tensor_as_f32(
                    st, &format!("{p}.attn.proj.bias"),
                )?);
                // attention_biases shape: [num_heads, n_distinct_offsets]
                let attention_biases: Arc<[f32]> = Arc::from(load_tensor_as_f32(
                    st, &format!("{p}.attn.attention_biases"),
                )?);
                let n_offsets = attention_biases.len() / num_heads;
                // Build flat indexing table.
                let attention_bias_idxs: Arc<[u32]> = Arc::from(
                    build_attention_bias_idxs(window_size, n_offsets)
                );
                let attn = TinyVitAttnWeights {
                    norm, qkv, qkv_bias, proj, proj_bias,
                    attention_biases, attention_bias_idxs, n_offsets,
                    key_dim, num_heads, d: dim, dh,
                };
                let local_conv = load_conv2d_bn(
                    st, &format!("{p}.local_conv"),
                    dim, dim, cfg.local_conv_size, dim, 1, cfg.local_conv_size / 2, bn_eps,
                )?;
                let mlp_norm = load_ln(&format!("{p}.mlp.norm"))?;
                let fc1 = ltm(st, &format!("{p}.mlp.fc1.weight"), dim * cfg.mlp_ratio, dim)?;
                let fc1_bias = Arc::from(load_tensor_as_f32(
                    st, &format!("{p}.mlp.fc1.bias"),
                )?);
                let fc2 = ltm(st, &format!("{p}.mlp.fc2.weight"), dim, dim * cfg.mlp_ratio)?;
                let fc2_bias = Arc::from(load_tensor_as_f32(
                    st, &format!("{p}.mlp.fc2.bias"),
                )?);
                let mlp = MlpWeights {
                    norm: mlp_norm, fc1, fc1_bias, fc2, fc2_bias,
                };
                blocks.push(TinyVitBlockWeights {
                    attn, local_conv, mlp,
                    dim, input_resolution: input_res, window_size,
                });
                let _ = n_tokens;
            }

            let downsample = if s + 1 < cfg.num_stages() {
                let dim_next = cfg.embed_dims[s + 1];
                let p = format!("layers.{s}.downsample");
                let conv1 = load_conv2d_bn(st, &format!("{p}.conv1"), dim, dim_next, 1, 1, 1, 0, bn_eps).ok();
                let conv2 = load_conv2d_bn(st, &format!("{p}.conv2"), dim_next, dim_next, 3, dim_next, 2, 1, bn_eps).ok();
                let conv3 = load_conv2d_bn(st, &format!("{p}.conv3"), dim_next, dim_next, 1, 1, 1, 0, bn_eps).ok();
                match (conv1, conv2, conv3) {
                    (Some(c1), Some(c2), Some(c3)) => Some(PatchMergingWeights {
                        conv1: c1, conv2: c2, conv3: c3,
                        input_resolution: input_res, dim, out: dim_next,
                    }),
                    _ => None,
                }
            } else { None };

            stages.push(BasicLayerWeights { blocks, downsample });
        }

        // Neck: 256-channel projection used by MobileSAM's image encoder.
        let last_dim = cfg.embed_dims[cfg.num_stages() - 1];
        let neck = TinyVitNeckWeights {
            conv1: Arc::from(load_tensor_as_f32(st, "neck.0.weight")?),
            ln1: LayerNorm2dWeights {
                gain: Arc::from(load_tensor_as_f32(st, "neck.1.weight")?),
                bias: Arc::from(load_tensor_as_f32(st, "neck.1.bias")?),
            },
            conv2: Arc::from(load_tensor_as_f32(st, "neck.2.weight")?),
            ln2: LayerNorm2dWeights {
                gain: Arc::from(load_tensor_as_f32(st, "neck.3.weight")?),
                bias: Arc::from(load_tensor_as_f32(st, "neck.3.bias")?),
            },
        };
        let _ = last_dim;

        Ok(Self { patch_embed, stage0, stages, neck })
    }
}

fn build_attention_bias_idxs(window_size: usize, n_offsets: usize) -> Vec<u32> {
    let n_tokens = window_size * window_size;
    let mut idxs = vec![0_u32; n_tokens * n_tokens];
    // Build attention_offsets like the eager code: set of distinct (dy, dx) abs-offsets.
    // For TinyViT, offsets are (|q.y - k.y|, |q.x - k.x|) — quadrant-collapsed.
    let mut offsets: Vec<(usize, usize)> = Vec::new();
    for qy in 0..window_size {
        for qx in 0..window_size {
            for ky in 0..window_size {
                for kx in 0..window_size {
                    let dy = qy.abs_diff(ky);
                    let dx = qx.abs_diff(kx);
                    if !offsets.contains(&(dy, dx)) {
                        offsets.push((dy, dx));
                    }
                }
            }
        }
    }
    for qi in 0..n_tokens {
        let (qy, qx) = (qi / window_size, qi % window_size);
        for ki in 0..n_tokens {
            let (ky, kx) = (ki / window_size, ki % window_size);
            let dy = qy.abs_diff(ky);
            let dx = qx.abs_diff(kx);
            let off_idx = offsets.iter().position(|&o| o == (dy, dx))
                .unwrap_or(0) as u32;
            idxs[qi * n_tokens + ki] = off_idx;
        }
    }
    // Cap to n_offsets to ensure no OOB indexing.
    for i in idxs.iter_mut() {
        if (*i as usize) >= n_offsets { *i = (n_offsets - 1) as u32; }
    }
    idxs
}

// ---- Tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.02
        }
    }

    fn tiny_bn(channels: usize, nb: &mut dyn FnMut() -> f32) -> BatchNormParams {
        let gain: Vec<f32> = (0..channels).map(|_| 1.0 + nb() * 0.1).collect();
        let bias: Vec<f32> = (0..channels).map(|_| nb() * 0.1).collect();
        let mean: Vec<f32> = (0..channels).map(|_| nb() * 0.05).collect();
        let var: Vec<f32> = (0..channels).map(|_| 1.0 + nb().abs() * 0.05).collect();
        BatchNormParams::from_raw(&gain, &bias, &mean, &var, 1e-5)
    }

    fn vec_of(n: usize, nb: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| nb()).collect::<Vec<_>>())
    }

    fn conv_bn(
        c_in: usize, c_out: usize, kernel: usize, stride: usize, padding: usize, groups: usize,
        nb: &mut dyn FnMut() -> f32,
    ) -> Conv2dBnWeights {
        let nw = c_out * (c_in / groups) * kernel * kernel;
        Conv2dBnWeights {
            conv_w: vec_of(nw, nb),
            bn: tiny_bn(c_out, nb),
            c_out, c_in, kernel, groups, stride, padding,
        }
    }

    fn make_attn_idxs(window: usize) -> (Vec<u32>, usize) {
        // Build attention_offsets map matching the eager loop.
        let mut offsets = std::collections::HashMap::new();
        let n = window * window;
        let mut idxs = Vec::with_capacity(n * n);
        let mut points = Vec::with_capacity(n);
        for x in 0..window {
            for y in 0..window {
                points.push((x as i64, y as i64));
            }
        }
        for &(x1, y1) in points.iter() {
            for &(x2, y2) in points.iter() {
                let off = ((x2 - x1).abs(), (y2 - y1).abs());
                let l = offsets.len();
                let idx = offsets.entry(off).or_insert(l);
                idxs.push(*idx as u32);
            }
        }
        (idxs, offsets.len())
    }

    fn ln_w(c: usize) -> SamLayerNormWeights {
        SamLayerNormWeights {
            gain: Arc::from(vec![1.0_f32; c]),
            bias: Arc::from(vec![0.0_f32; c]),
        }
    }

    fn ln2d_w(c: usize) -> LayerNorm2dWeights {
        LayerNorm2dWeights {
            gain: Arc::from(vec![1.0_f32; c]),
            bias: Arc::from(vec![0.0_f32; c]),
        }
    }

    fn tiny_config() -> TinyVitConfig {
        TinyVitConfig {
            img_size: 32,
            in_chans: 3,
            embed_dims: vec![8, 16, 24, 32],
            depths: vec![1, 1, 1, 1],
            num_heads: vec![1, 2, 3, 4],
            window_sizes: vec![2, 2, 2, 2],
            mbconv_expand_ratio: 2,
            mlp_ratio: 2,
            local_conv_size: 3,
        }
    }

    fn tiny_weights(cfg: &TinyVitConfig) -> TinyVitWeights {
        let mut nb = rng(42);
        let in_c = cfg.in_chans;
        let e0 = cfg.embed_dims[0];
        let pe1 = conv_bn(in_c, e0 / 2, 3, 2, 1, 1, &mut nb);
        let pe2 = conv_bn(e0 / 2, e0, 3, 2, 1, 1, &mut nb);
        let patch_embed = PatchEmbedWeights { conv1: pe1, conv2: pe2 };

        // Stage 0: MBConv blocks + PatchMerging downsample to stage 1's dim.
        let pps0 = cfg.img_size / 4;
        let mb_hidden = e0 * cfg.mbconv_expand_ratio;
        let stage0_blocks: Vec<MbConvWeights> = (0..cfg.depths[0]).map(|_| {
            MbConvWeights {
                conv1: conv_bn(e0, mb_hidden, 1, 1, 0, 1, &mut nb),
                conv2: conv_bn(mb_hidden, mb_hidden, 3, 1, 1, mb_hidden, &mut nb),
                conv3: conv_bn(mb_hidden, e0, 1, 1, 0, 1, &mut nb),
            }
        }).collect();
        let e1 = cfg.embed_dims[1];
        let stage0_ds = PatchMergingWeights {
            conv1: conv_bn(e0, e1, 1, 1, 0, 1, &mut nb),
            conv2: conv_bn(e1, e1, 3, 2, 1, e1, &mut nb),
            conv3: conv_bn(e1, e1, 1, 1, 0, 1, &mut nb),
            input_resolution: (pps0, pps0), dim: e0, out: e1,
        };
        let stage0 = ConvLayerWeights {
            blocks: stage0_blocks,
            downsample: Some(stage0_ds),
        };

        let n_stages = cfg.num_stages();
        let mut stages = Vec::with_capacity(n_stages - 1);
        // Track actual spatial resolution through the stage chain. Stage
        // 0 downsamples by 2× via its PatchMerging (stride may flip
        // based on the next dim), so stage 1 sees pps0 / 2.
        let mut current_res = pps0 / 2; // after stage 0 ds (stride=2 for non-special e1)
        for i in 1..n_stages {
            let res = current_res;
            let dim = cfg.embed_dims[i];
            let nh = cfg.num_heads[i];
            let ws = cfg.window_sizes[i];
            let key_dim = dim / nh;
            let d = key_dim * 1; // attn_ratio=1
            let dh = d * nh;
            let d_total = dh + 2 * nh * key_dim;
            let (attn_idxs, n_offsets) = make_attn_idxs(ws);

            let blocks: Vec<TinyVitBlockWeights> = (0..cfg.depths[i]).map(|_| {
                let attn = TinyVitAttnWeights {
                    norm: ln_w(dim),
                    qkv: WeightStorage::F32(vec_of(dim * d_total, &mut nb)),
                    qkv_bias: vec_of(d_total, &mut nb),
                    proj: WeightStorage::F32(vec_of(dh * dim, &mut nb)),
                    proj_bias: vec_of(dim, &mut nb),
                    attention_biases: vec_of(nh * n_offsets, &mut nb),
                    attention_bias_idxs: Arc::from(attn_idxs.clone()),
                    n_offsets, key_dim, num_heads: nh, d, dh,
                };
                TinyVitBlockWeights {
                    attn,
                    local_conv: conv_bn(dim, dim, cfg.local_conv_size, 1, cfg.local_conv_size/2, dim, &mut nb),
                    mlp: MlpWeights {
                        norm: ln_w(dim),
                        fc1: WeightStorage::F32(vec_of(dim * dim * cfg.mlp_ratio, &mut nb)),
                        fc1_bias: vec_of(dim * cfg.mlp_ratio, &mut nb),
                        fc2: WeightStorage::F32(vec_of(dim * cfg.mlp_ratio * dim, &mut nb)),
                        fc2_bias: vec_of(dim, &mut nb),
                    },
                    dim, input_resolution: (res, res), window_size: ws,
                }
            }).collect();
            let ds = if i < n_stages - 1 {
                let next_dim = cfg.embed_dims[i + 1];
                let stride = if [320, 448, 576].contains(&next_dim) { 1 } else { 2 };
                let ds_w = PatchMergingWeights {
                    conv1: conv_bn(dim, next_dim, 1, 1, 0, 1, &mut nb),
                    // Stride mirrors eager logic: 1 for {320,448,576} else 2.
                    conv2: conv_bn(next_dim, next_dim, 3, stride, 1, next_dim, &mut nb),
                    conv3: conv_bn(next_dim, next_dim, 1, 1, 0, 1, &mut nb),
                    input_resolution: (res, res), dim, out: next_dim,
                };
                // Advance current_res for the next stage's blocks.
                current_res = if stride == 2 { res / 2 } else { res };
                Some(ds_w)
            } else { None };
            stages.push(BasicLayerWeights { blocks, downsample: ds });
        }

        let last_dim = *cfg.embed_dims.last().unwrap();
        let neck = TinyVitNeckWeights {
            conv1: vec_of(256 * last_dim * 1 * 1, &mut nb),
            ln1: ln2d_w(256),
            conv2: vec_of(256 * 256 * 3 * 3, &mut nb),
            ln2: ln2d_w(256),
        };

        TinyVitWeights { patch_embed, stage0, stages, neck }
    }

    #[test]
    fn forward_shape_and_finite_tiny() {
        // tiny_config: img=32, 4 stages each downsampling by 2 except
        // the last → 32/4 (patch_embed) → 8 → 4 → 2 → 1 (no ds).
        // Final spatial side: 1. Neck output: (1, 256, 1, 1).
        let cfg = tiny_config();
        let model = TinyVitModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let img: Vec<f32> = (0..cfg.in_chans * cfg.img_size * cfg.img_size)
            .map(|i| ((i as f32) * 0.001) - 0.05).collect();
        let out = model.forward(&img).unwrap();
        assert_eq!(out.shape().dims(), &[1, 256, 1, 1]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite neck output: {v}");
        }
    }

    #[test]
    fn mobile_sam_preset_dims() {
        let cfg = TinyVitConfig::mobile_sam_5m();
        assert_eq!(cfg.img_size, 1024);
        assert_eq!(cfg.in_chans, 3);
        assert_eq!(cfg.embed_dims, vec![64, 128, 160, 320]);
        assert_eq!(cfg.depths, vec![2, 2, 6, 2]);
        assert_eq!(cfg.num_heads, vec![2, 4, 5, 10]);
        assert_eq!(cfg.window_sizes, vec![7, 7, 14, 7]);
    }
}

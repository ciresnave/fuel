//! SegFormer — lazy port.
//!
//! Hierarchical 4-stage transformer encoder with two heads:
//! ImageClassificationModel (mean-pool of last stage → linear)
//! and SemanticSegmentationModel (per-stage MLP → arbitrary-scale
//! interpolate to stage 0 resolution → concat → 1×1 conv + BN +
//! ReLU → 1×1 classifier).
//!
//! Each encoder stage:
//!   1. Overlap patch embedding (Conv2d k=patch_size, stride
//!      from `cfg.strides[i]`, pad = patch_size/2) → LayerNorm.
//!   2. N SegformerLayer blocks:
//!      Pre-LN1 → Efficient Self-Attention (Q from input, K/V
//!      from input after optional Sequence Reduction conv with
//!      stride = sr_ratio[i] + LN) → +residual
//!      → Pre-LN2 → Mix-FFN (Dense1 → 3×3 DWConv → activation
//!      → Dense2) → +residual.
//!   3. Stage-final LayerNorm.
//!
//! Mix-FFN's 3×3 depthwise conv embeds 2D spatial smoothing into
//! the channel-wise FFN, replacing explicit positional embeddings
//! — one of SegFormer's distinguishing design choices.
//!
//! v1 scope: F32, batch == 1, fused-affine BN at the decode head.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_convmixer::BatchNormParams;
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegformerActivation {
    Gelu,
    Relu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegformerConfig {
    pub num_channels: usize,
    pub num_encoder_blocks: usize,
    pub depths: Vec<usize>,
    pub sr_ratios: Vec<usize>,
    pub hidden_sizes: Vec<usize>,
    pub patch_sizes: Vec<usize>,
    pub strides: Vec<usize>,
    pub num_attention_heads: Vec<usize>,
    pub mlp_ratios: Vec<usize>,
    pub hidden_act: SegformerActivation,
    pub layer_norm_eps: f64,
    pub decoder_hidden_size: usize,
}

impl SegformerConfig {
    /// Build a SegFormer MIT preset. The variants differ only in
    /// `hidden_sizes`, `depths`, and `decoder_hidden_size` — the
    /// other fields are shared across all `nvidia/mit-b{0..5}` checkpoints.
    fn mit_preset(
        hidden_sizes: [usize; 4],
        depths: [usize; 4],
        decoder_hidden_size: usize,
    ) -> Self {
        Self {
            num_channels: 3,
            num_encoder_blocks: 4,
            depths: depths.to_vec(),
            sr_ratios: vec![8, 4, 2, 1],
            hidden_sizes: hidden_sizes.to_vec(),
            patch_sizes: vec![7, 3, 3, 3],
            strides: vec![4, 2, 2, 2],
            num_attention_heads: vec![1, 2, 5, 8],
            mlp_ratios: vec![4, 4, 4, 4],
            hidden_act: SegformerActivation::Gelu,
            layer_norm_eps: 1e-6,
            decoder_hidden_size,
        }
    }

    /// HuggingFace MIT-B0 preset (matches `nvidia/mit-b0`).
    pub fn mit_b0() -> Self {
        Self::mit_preset([32, 64, 160, 256], [2, 2, 2, 2], 256)
    }
    /// HuggingFace MIT-B1 preset (matches `nvidia/mit-b1`).
    pub fn mit_b1() -> Self {
        Self::mit_preset([64, 128, 320, 512], [2, 2, 2, 2], 256)
    }
    /// HuggingFace MIT-B2 preset (matches `nvidia/mit-b2`).
    pub fn mit_b2() -> Self {
        Self::mit_preset([64, 128, 320, 512], [3, 4, 6, 3], 768)
    }
    /// HuggingFace MIT-B3 preset (matches `nvidia/mit-b3`).
    pub fn mit_b3() -> Self {
        Self::mit_preset([64, 128, 320, 512], [3, 4, 18, 3], 768)
    }
    /// HuggingFace MIT-B4 preset (matches `nvidia/mit-b4`).
    pub fn mit_b4() -> Self {
        Self::mit_preset([64, 128, 320, 512], [3, 8, 27, 3], 768)
    }
    /// HuggingFace MIT-B5 preset (matches `nvidia/mit-b5`).
    pub fn mit_b5() -> Self {
        Self::mit_preset([64, 128, 320, 512], [3, 6, 40, 3], 768)
    }
}

// ---- Weight structures ------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Conv2dWeights {
    /// `[c_out, c_in / groups, k, k]`.
    pub w: Arc<[f32]>,
    pub b: Option<Arc<[f32]>>,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub groups: usize,
}

/// Affine LayerNorm: gain + bias per channel.
#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    pub gain: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct OverlapPatchEmbeddingWeights {
    pub projection: Conv2dWeights,
    pub layer_norm: LayerNormWeights,
}

#[derive(Debug, Clone)]
pub struct EfficientSelfAttentionWeights {
    /// All three Q/K/V projections: `[hidden, hidden]`.
    pub query: WeightStorage,
    pub query_bias: Arc<[f32]>,
    pub key: WeightStorage,
    pub key_bias: Arc<[f32]>,
    pub value: WeightStorage,
    pub value_bias: Arc<[f32]>,
    /// Optional sequence-reduction Conv2d (k=stride=sr_ratio) +
    /// LayerNorm. Present iff sr_ratio > 1.
    pub sr: Option<Conv2dWeights>,
    pub sr_norm: Option<LayerNormWeights>,
}

#[derive(Debug, Clone)]
pub struct AttentionOutputWeights {
    pub dense: WeightStorage,
    pub dense_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct MixFfnWeights {
    pub dense1: WeightStorage,
    pub dense1_bias: Arc<[f32]>,
    /// 3×3 depthwise conv (groups = hidden_features).
    pub dw_conv: Conv2dWeights,
    pub dense2: WeightStorage,
    pub dense2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct SegformerLayerWeights {
    pub layer_norm_1: LayerNormWeights,
    pub attention: EfficientSelfAttentionWeights,
    pub attention_output: AttentionOutputWeights,
    pub layer_norm_2: LayerNormWeights,
    pub mlp: MixFfnWeights,
    /// Stage's hidden size — load-time value, distinct per stage.
    pub hidden_size: usize,
    /// Stage's attention head count.
    pub num_heads: usize,
}

#[derive(Debug, Clone)]
pub struct SegformerStageWeights {
    pub patch_embedding: OverlapPatchEmbeddingWeights,
    pub layers: Vec<SegformerLayerWeights>,
    pub final_ln: LayerNormWeights,
}

#[derive(Debug, Clone)]
pub struct SegformerEncoderWeights {
    pub stages: Vec<SegformerStageWeights>,
}

/// Decode-head weights for semantic segmentation.
#[derive(Debug, Clone)]
pub struct SegformerDecodeHeadWeights {
    /// Per-stage MLP: hidden_sizes[i] → decoder_hidden_size.
    pub linear_c: Vec<(WeightStorage, Arc<[f32]>)>,
    /// 1×1 conv: 4·decoder_hidden_size → decoder_hidden_size.
    pub linear_fuse: Conv2dWeights,
    pub batch_norm: BatchNormParams,
    /// 1×1 conv: decoder_hidden_size → num_labels.
    pub classifier: Conv2dWeights,
}

/// Classification-head weights.
#[derive(Debug, Clone)]
pub struct SegformerClassifierWeights {
    /// `[hidden_sizes[-1], num_labels]`.
    pub w: WeightStorage,
    pub b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct SemanticSegmentationModel {
    pub config: SegformerConfig,
    pub encoder: SegformerEncoderWeights,
    pub decode_head: SegformerDecodeHeadWeights,
    pub num_labels: usize,
}

#[derive(Debug, Clone)]
pub struct ImageClassificationModel {
    pub config: SegformerConfig,
    pub encoder: SegformerEncoderWeights,
    pub classifier: SegformerClassifierWeights,
}

// ---- Forward ---------------------------------------------------------------

impl SemanticSegmentationModel {
    /// Run the segmentation head. Returns logits
    /// `(1, num_labels, h_stage0, w_stage0)` — caller is expected
    /// to upsample to the input image resolution.
    pub fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let hidden_states = encoder_forward(image, &self.config, &self.encoder)?;
        decode_head_forward(image, &hidden_states, &self.config, &self.decode_head)
    }
}

impl ImageClassificationModel {
    /// Run the encoder and return classification logits
    /// `(1, num_labels)` from the mean-pooled last-stage features.
    pub fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let hidden_states = encoder_forward(image, cfg, &self.encoder)?;
        let last = hidden_states.last().expect("encoder produced no states");
        let dims = last.shape();
        let dims = dims.dims();
        let c = dims[1]; let h = dims[2]; let w = dims[3];
        let flat = last
            .reshape(Shape::from_dims(&[1, c, h * w]))?
            .permute([0, 2, 1_usize])?;
        let pooled = flat.mean_dim(1_usize)?;
        let n = self.classifier.b.len();
        let logits = self.classifier.w.apply_linear(&pooled, c, n);
        let bias = image.const_f32_like(
            Arc::clone(&self.classifier.b), Shape::from_dims(&[n]),
        );
        logits.broadcast_add(&bias)
    }
}

fn encoder_forward(
    image: &LazyTensor,
    cfg: &SegformerConfig,
    enc: &SegformerEncoderWeights,
) -> Result<Vec<LazyTensor>> {
    assert_eq!(enc.stages.len(), cfg.num_encoder_blocks);
    let mut all = Vec::with_capacity(enc.stages.len());
    let mut x = image.clone();
    for stage in &enc.stages {
        // Patch embedding: conv + LN.
        x = apply_conv2d(&x, &stage.patch_embedding.projection, image)?;
        x = layer_norm_chw(&x, &stage.patch_embedding.layer_norm, cfg.layer_norm_eps)?;

        for layer in &stage.layers {
            x = apply_segformer_layer(&x, layer, cfg, image)?;
        }

        // Stage-final LN.
        x = layer_norm_chw(&x, &stage.final_ln, cfg.layer_norm_eps)?;

        all.push(x.clone());
    }
    Ok(all)
}

/// LayerNorm applied along the channel axis of a (B, C, H, W)
/// tensor: permute to (B, H*W, C), normalize last dim, scale +
/// shift, then permute back.
fn layer_norm_chw(
    x: &LazyTensor,
    ln: &LayerNormWeights,
    eps: f64,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let c = dims[1]; let h = dims[2]; let w = dims[3];
    let seq = x
        .reshape(Shape::from_dims(&[b, c, h * w]))?
        .permute([0, 2, 1_usize])?;
    let normed = seq.layer_norm_affine(Arc::clone(&ln.gain), Arc::clone(&ln.bias), eps)?;
    Ok(normed
        .permute([0, 2, 1_usize])?
        .reshape(Shape::from_dims(&[b, c, h, w]))?)
}

fn apply_segformer_layer(
    x: &LazyTensor,
    w: &SegformerLayerWeights,
    cfg: &SegformerConfig,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let c = dims[1]; let h = dims[2]; let w_sp = dims[3];

    // (B, C, H, W) → (B, H*W, C) for the residual + LN1 path.
    let seq_orig = x
        .reshape(Shape::from_dims(&[b, c, h * w_sp]))?
        .permute([0, 2, 1_usize])?;

    // LN1 on the [B, H*W, C] form.
    let normed = seq_orig.layer_norm_affine(Arc::clone(&w.layer_norm_1.gain), Arc::clone(&w.layer_norm_1.bias), cfg.layer_norm_eps)?;
    let norm_chw = normed
        .permute([0, 2, 1_usize])?
        .reshape(Shape::from_dims(&[b, c, h, w_sp]))?;
    let attn_out = apply_efficient_attention(
        &norm_chw, &w.attention, &w.attention_output,
        w.hidden_size, w.num_heads, cfg.layer_norm_eps, anchor,
    )?;
    // Residual on (B, H*W, C).
    let hidden = attn_out.add(&seq_orig)?;

    // LN2 on (B, H*W, C).
    let normed = hidden.layer_norm_affine(Arc::clone(&w.layer_norm_2.gain), Arc::clone(&w.layer_norm_2.bias), cfg.layer_norm_eps)?;
    let norm_chw = normed
        .permute([0, 2, 1_usize])?
        .reshape(Shape::from_dims(&[b, c, h, w_sp]))?;
    let mlp_out = apply_mix_ffn(&norm_chw, &w.mlp, cfg, c, anchor)?;
    let mlp_seq = mlp_out
        .reshape(Shape::from_dims(&[b, c, h * w_sp]))?
        .permute([0, 2, 1_usize])?;
    let out_seq = hidden.add(&mlp_seq)?;
    Ok(out_seq
        .permute([0, 2, 1_usize])?
        .reshape(Shape::from_dims(&[b, c, h, w_sp]))?)
}

/// Efficient self-attention. Input is (B, C, H, W); output is
/// (B, H*W, C). Q from (B, H*W, C); K/V from a sequence-reduced
/// view (Conv2d stride=sr_ratio + LN) when `attn.sr` is Some.
#[allow(clippy::too_many_arguments)]
fn apply_efficient_attention(
    x: &LazyTensor,
    attn: &EfficientSelfAttentionWeights,
    out: &AttentionOutputWeights,
    hidden_size: usize,
    num_heads: usize,
    eps: f64,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let h = dims[2]; let w_sp = dims[3];
    let head_dim = hidden_size / num_heads;
    let scale = 1.0_f64 / (head_dim as f64).sqrt();

    let x_seq = x
        .reshape(Shape::from_dims(&[b, hidden_size, h * w_sp]))?
        .permute([0, 2, 1_usize])?;
    let q_len = h * w_sp;

    let q = attn.query.apply_linear_with_bias(&x_seq, hidden_size, hidden_size, std::sync::Arc::clone(&attn.query_bias))?;

    let kv_seq = if let (Some(sr), Some(sr_norm)) = (&attn.sr, &attn.sr_norm) {
        let sr_out = apply_conv2d(x, sr, anchor)?;
        let sr_dims = sr_out.shape();
        let sr_dims = sr_dims.dims();
        let h2 = sr_dims[2]; let w2 = sr_dims[3];
        let flat = sr_out
            .reshape(Shape::from_dims(&[b, hidden_size, h2 * w2]))?
            .permute([0, 2, 1_usize])?;
        flat.layer_norm_affine(Arc::clone(&sr_norm.gain), Arc::clone(&sr_norm.bias), eps)?
    } else {
        x_seq.clone()
    };

    let kv_dims = kv_seq.shape();
    let kv_len = kv_dims.dims()[1];
    let k = attn.key.apply_linear_with_bias(&kv_seq, hidden_size, hidden_size, std::sync::Arc::clone(&attn.key_bias))?;
    let v = attn.value.apply_linear_with_bias(&kv_seq, hidden_size, hidden_size, std::sync::Arc::clone(&attn.value_bias))?;

    let _ = (q_len, kv_len);
    let q = q.split_heads(num_heads, head_dim)?;
    let k = k.split_heads(num_heads, head_dim)?;
    let v = v.split_heads(num_heads, head_dim)?;

    let kt = k.permute([0, 1, 3, 2_usize])?;
    let scores = q.matmul(&kt)?.mul_scalar(scale);
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?.merge_heads()?;
    let _ = (b, hidden_size);
    out.dense.apply_linear_with_bias(&ctx, hidden_size, hidden_size, std::sync::Arc::clone(&out.dense_bias))
}

fn apply_mix_ffn(
    x: &LazyTensor,
    m: &MixFfnWeights,
    cfg: &SegformerConfig,
    hidden_size: usize,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let h = dims[2]; let w_sp = dims[3];
    let hidden_features = m.dense1_bias.len();

    let seq = x
        .reshape(Shape::from_dims(&[b, hidden_size, h * w_sp]))?
        .permute([0, 2, 1_usize])?;
    let h1 = m.dense1.apply_linear_with_bias(&seq, hidden_size, hidden_features, std::sync::Arc::clone(&m.dense1_bias))?;
    let chw = h1
        .permute([0, 2, 1_usize])?
        .reshape(Shape::from_dims(&[b, hidden_features, h, w_sp]))?;
    let h2 = apply_conv2d(&chw, &m.dw_conv, anchor)?;
    let h2 = match cfg.hidden_act {
        SegformerActivation::Gelu => h2.gelu(),
        SegformerActivation::Relu => h2.relu(),
    };
    let seq = h2
        .reshape(Shape::from_dims(&[b, hidden_features, h * w_sp]))?
        .permute([0, 2, 1_usize])?;
    let h3 = m.dense2.apply_linear_with_bias(&seq, hidden_features, hidden_size, std::sync::Arc::clone(&m.dense2_bias))?;
    Ok(h3
        .permute([0, 2, 1_usize])?
        .reshape(Shape::from_dims(&[b, hidden_size, h, w_sp]))?)
}

fn decode_head_forward(
    anchor: &LazyTensor,
    states: &[LazyTensor],
    cfg: &SegformerConfig,
    head: &SegformerDecodeHeadWeights,
) -> Result<LazyTensor> {
    assert_eq!(states.len(), head.linear_c.len());
    let dims0 = states[0].shape();
    let dims0 = dims0.dims();
    let target_h = dims0[2];
    let target_w = dims0[3];

    let mut feats: Vec<LazyTensor> = Vec::with_capacity(states.len());
    for (i, hs) in states.iter().enumerate() {
        let (w, bias) = &head.linear_c[i];
        let dims = hs.shape();
        let dims = dims.dims();
        let b = dims[0]; let c = dims[1]; let h = dims[2]; let w_sp = dims[3];
        let seq = hs
            .reshape(Shape::from_dims(&[b, c, h * w_sp]))?
            .permute([0, 2, 1_usize])?;
        let projected = w.apply_linear_with_bias(&seq, c, cfg.decoder_hidden_size, std::sync::Arc::clone(&bias))?;
        let chw = projected
            .permute([0, 2, 1_usize])?
            .reshape(Shape::from_dims(&[b, cfg.decoder_hidden_size, h, w_sp]))?;
        let upsampled = chw.interpolate2d(target_h, target_w)?;
        feats.push(upsampled);
    }
    feats.reverse();
    let mut cat = feats[0].clone();
    for f in &feats[1..] {
        cat = cat.concat(f, 1_usize)?;
    }
    let fused = apply_conv2d(&cat, &head.linear_fuse, anchor)?;
    let bn = apply_bn(&fused, &head.batch_norm, cfg.decoder_hidden_size)?;
    let relu = bn.relu();
    apply_conv2d(&relu, &head.classifier, anchor)
}

// ---- Primitives ------------------------------------------------------------



fn apply_conv2d(
    x: &LazyTensor,
    c: &Conv2dWeights,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let w = anchor.const_f32_like(
        Arc::clone(&c.w),
        Shape::from_dims(&[c.c_out, c.c_in / c.groups, c.k, c.k]),
    );
    let bias = c.b.as_ref().map(|b| {
        anchor.const_f32_like(Arc::clone(b), Shape::from_dims(&[c.c_out]))
    });
    x.conv2d(
        &w, bias.as_ref(),
        (c.stride, c.stride),
        (c.pad, c.pad),
        c.groups,
    )
}

fn apply_bn(
    x: &LazyTensor, bn: &BatchNormParams, channels: usize,
) -> Result<LazyTensor> {
    let _ = channels;
    x.channel_affine_4d(Arc::clone(&bn.w), Arc::clone(&bn.b))
}

// ---- Safetensors loaders ---------------------------------------------------

/// Load a 1-D F32 tensor as `Arc<[f32]>`.
fn load_arc_f32(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
) -> Result<Arc<[f32]>> {
    Ok(Arc::from(crate::lazy::load_tensor_as_f32(st, name)?))
}

/// Load a HuggingFace LayerNorm (`<prefix>.weight`, `<prefix>.bias`).
fn load_ln(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
) -> Result<LayerNormWeights> {
    Ok(LayerNormWeights {
        gain: load_arc_f32(st, &format!("{prefix}.weight"))?,
        bias: load_arc_f32(st, &format!("{prefix}.bias"))?,
    })
}

/// Load a HuggingFace Conv2d (`<prefix>.weight`, optional `<prefix>.bias`).
/// The on-disk layout `[c_out, c_in / groups, k, k]` is the same as ours,
/// so this is a flat F32 read.
fn load_conv2d(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c_in: usize,
    c_out: usize,
    k: usize,
    stride: usize,
    pad: usize,
    groups: usize,
    has_bias: bool,
) -> Result<Conv2dWeights> {
    let w = load_arc_f32(st, &format!("{prefix}.weight"))?;
    let expected = c_out * (c_in / groups) * k * k;
    if w.len() != expected {
        return Err(crate::Error::Msg(format!(
            "load_conv2d {prefix:?}: weight has {} elements, expected {expected} \
             ([{c_out}, {} = {c_in}/{groups}, {k}, {k}])",
            w.len(), c_in / groups,
        )).bt());
    }
    let b = if has_bias {
        let bias = load_arc_f32(st, &format!("{prefix}.bias"))?;
        if bias.len() != c_out {
            return Err(crate::Error::Msg(format!(
                "load_conv2d {prefix:?}: bias has {} elements, expected {c_out}",
                bias.len(),
            )).bt());
        }
        Some(bias)
    } else {
        None
    };
    Ok(Conv2dWeights { w, b, c_in, c_out, k, stride, pad, groups })
}

/// Load a HuggingFace Linear into a `WeightStorage` (+ bias). HF stores
/// `[out, in]`; we transpose to `[in, out]` to match
/// `WeightStorage::apply_linear`'s convention.
fn load_linear(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    in_features: usize,
    out_features: usize,
) -> Result<(WeightStorage, Arc<[f32]>)> {
    let w = crate::lazy::load_transposed_matrix_preserve_dtype(
        st, &format!("{prefix}.weight"), out_features, in_features,
    )?;
    let bias = load_arc_f32(st, &format!("{prefix}.bias"))?;
    if bias.len() != out_features {
        return Err(crate::Error::Msg(format!(
            "load_linear {prefix:?}: bias has {} elements, expected {out_features}",
            bias.len(),
        )).bt());
    }
    Ok((w, bias))
}

/// Load a HuggingFace BatchNorm prefix (`{weight,bias,running_mean,
/// running_var}`) and bake into our fused-affine form.
fn load_bn(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    channels: usize,
    eps: f64,
) -> Result<BatchNormParams> {
    let gain = crate::lazy::load_tensor_as_f32(st, &format!("{prefix}.weight"))?;
    let bias = crate::lazy::load_tensor_as_f32(st, &format!("{prefix}.bias"))?;
    let mean = crate::lazy::load_tensor_as_f32(st, &format!("{prefix}.running_mean"))?;
    let var  = crate::lazy::load_tensor_as_f32(st, &format!("{prefix}.running_var"))?;
    if gain.len() != channels || bias.len() != channels
        || mean.len() != channels || var.len() != channels {
        return Err(crate::Error::Msg(format!(
            "load_bn {prefix:?}: expected {channels} elements per stat, \
             got gain={} bias={} mean={} var={}",
            gain.len(), bias.len(), mean.len(), var.len(),
        )).bt());
    }
    Ok(BatchNormParams::from_raw(&gain, &bias, &mean, &var, eps))
}

impl OverlapPatchEmbeddingWeights {
    fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        c_in: usize,
        c_out: usize,
        patch_size: usize,
        stride: usize,
    ) -> Result<Self> {
        Ok(Self {
            projection: load_conv2d(
                st, &format!("{prefix}.proj"),
                c_in, c_out, patch_size, stride, patch_size / 2, 1, true,
            )?,
            layer_norm: load_ln(st, &format!("{prefix}.layer_norm"))?,
        })
    }
}

impl EfficientSelfAttentionWeights {
    fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        hidden_size: usize,
        sr_ratio: usize,
    ) -> Result<Self> {
        let (q, qb) = load_linear(st, &format!("{prefix}.query"), hidden_size, hidden_size)?;
        let (k, kb) = load_linear(st, &format!("{prefix}.key"),   hidden_size, hidden_size)?;
        let (v, vb) = load_linear(st, &format!("{prefix}.value"), hidden_size, hidden_size)?;
        let (sr, sr_norm) = if sr_ratio > 1 {
            (
                Some(load_conv2d(
                    st, &format!("{prefix}.sr"),
                    hidden_size, hidden_size, sr_ratio, sr_ratio, 0, 1, true,
                )?),
                Some(load_ln(st, &format!("{prefix}.layer_norm"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            query: q, query_bias: qb,
            key: k, key_bias: kb,
            value: v, value_bias: vb,
            sr, sr_norm,
        })
    }
}

impl AttentionOutputWeights {
    fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        hidden_size: usize,
    ) -> Result<Self> {
        let (w, b) = load_linear(st, &format!("{prefix}.dense"), hidden_size, hidden_size)?;
        Ok(Self { dense: w, dense_bias: b })
    }
}

impl MixFfnWeights {
    fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        hidden_size: usize,
        mlp_ratio: usize,
    ) -> Result<Self> {
        let hidden_features = hidden_size * mlp_ratio;
        let (d1, d1b) = load_linear(
            st, &format!("{prefix}.dense1"), hidden_size, hidden_features,
        )?;
        // HF nests Conv2d twice: SegformerDWConv wraps Conv2d at `.dwconv`,
        // and MixFFN places SegformerDWConv at `.dwconv` → final key is
        // `<prefix>.dwconv.dwconv.{weight,bias}`.
        let dw = load_conv2d(
            st, &format!("{prefix}.dwconv.dwconv"),
            hidden_features, hidden_features, 3, 1, 1, hidden_features, true,
        )?;
        let (d2, d2b) = load_linear(
            st, &format!("{prefix}.dense2"), hidden_features, hidden_size,
        )?;
        Ok(Self {
            dense1: d1, dense1_bias: d1b,
            dw_conv: dw,
            dense2: d2, dense2_bias: d2b,
        })
    }
}

impl SegformerLayerWeights {
    fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        hidden_size: usize,
        num_heads: usize,
        sr_ratio: usize,
        mlp_ratio: usize,
    ) -> Result<Self> {
        Ok(Self {
            layer_norm_1: load_ln(st, &format!("{prefix}.layer_norm_1"))?,
            attention: EfficientSelfAttentionWeights::load_from_mmapped(
                st, &format!("{prefix}.attention.self"), hidden_size, sr_ratio,
            )?,
            attention_output: AttentionOutputWeights::load_from_mmapped(
                st, &format!("{prefix}.attention.output"), hidden_size,
            )?,
            layer_norm_2: load_ln(st, &format!("{prefix}.layer_norm_2"))?,
            mlp: MixFfnWeights::load_from_mmapped(
                st, &format!("{prefix}.mlp"), hidden_size, mlp_ratio,
            )?,
            hidden_size,
            num_heads,
        })
    }
}

impl SegformerEncoderWeights {
    /// Load the SegFormer hierarchical encoder from a HuggingFace
    /// `nvidia/mit-b{0..5}` or `nvidia/segformer-*` safetensors blob.
    ///
    /// `prefix` lets callers point at either bare encoder checkpoints
    /// (`encoder.`) or the wrapped `SegformerModel` (`segformer.encoder.`)
    /// — pass `""` for the former and `"segformer.encoder."` for the latter.
    /// Final prefix is `<root>encoder.` regardless: this method appends
    /// `patch_embeddings.{i}.*`, `block.{i}.{j}.*`, and `layer_norm.{i}.*`
    /// after the supplied prefix.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &SegformerConfig,
        prefix: &str,
    ) -> Result<Self> {
        let mut stages = Vec::with_capacity(cfg.num_encoder_blocks);
        for i in 0..cfg.num_encoder_blocks {
            let c_in = if i == 0 { cfg.num_channels } else { cfg.hidden_sizes[i - 1] };
            let c_out = cfg.hidden_sizes[i];
            let patch_embedding = OverlapPatchEmbeddingWeights::load_from_mmapped(
                st, &format!("{prefix}patch_embeddings.{i}"),
                c_in, c_out, cfg.patch_sizes[i], cfg.strides[i],
            )?;
            let mut layers = Vec::with_capacity(cfg.depths[i]);
            for j in 0..cfg.depths[i] {
                layers.push(SegformerLayerWeights::load_from_mmapped(
                    st, &format!("{prefix}block.{i}.{j}"),
                    c_out,
                    cfg.num_attention_heads[i],
                    cfg.sr_ratios[i],
                    cfg.mlp_ratios[i],
                )?);
            }
            let final_ln = load_ln(st, &format!("{prefix}layer_norm.{i}"))?;
            stages.push(SegformerStageWeights {
                patch_embedding,
                layers,
                final_ln,
            });
        }
        Ok(Self { stages })
    }
}

impl SegformerDecodeHeadWeights {
    /// Load the all-MLP decode head. HF key prefix is `decode_head.`;
    /// pass `prefix = "decode_head."` for top-level segmentation
    /// checkpoints. `num_labels` is the output class count (e.g. 150
    /// for ADE20K, 19 for Cityscapes).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &SegformerConfig,
        prefix: &str,
        num_labels: usize,
    ) -> Result<Self> {
        // HF BatchNorm eps default is 1e-5 (independent of layer_norm_eps).
        const BN_EPS: f64 = 1e-5;
        let mut linear_c = Vec::with_capacity(cfg.num_encoder_blocks);
        for i in 0..cfg.num_encoder_blocks {
            let (w, b) = load_linear(
                st, &format!("{prefix}linear_c.{i}.proj"),
                cfg.hidden_sizes[i], cfg.decoder_hidden_size,
            )?;
            linear_c.push((w, b));
        }
        let linear_fuse = load_conv2d(
            st, &format!("{prefix}linear_fuse"),
            cfg.decoder_hidden_size * cfg.num_encoder_blocks,
            cfg.decoder_hidden_size, 1, 1, 0, 1, false,
        )?;
        let batch_norm = load_bn(
            st, &format!("{prefix}batch_norm"), cfg.decoder_hidden_size, BN_EPS,
        )?;
        // HF's `classifier` is `nn.Conv2d(..., kernel_size=1)`; bias=True by default.
        let classifier = load_conv2d(
            st, &format!("{prefix}classifier"),
            cfg.decoder_hidden_size, num_labels, 1, 1, 0, 1, true,
        )?;
        Ok(Self { linear_c, linear_fuse, batch_norm, classifier })
    }
}

impl SegformerClassifierWeights {
    /// Load the image-classification head (single linear from the last
    /// stage hidden size to `num_labels`). HF stores this as
    /// `classifier.{weight,bias}` at the top level of a
    /// `SegformerForImageClassification` checkpoint.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &SegformerConfig,
        prefix: &str,
        num_labels: usize,
    ) -> Result<Self> {
        let in_features = *cfg.hidden_sizes.last().expect("hidden_sizes is non-empty");
        let (w, b) = load_linear(
            st, &format!("{prefix}classifier"),
            in_features, num_labels,
        )?;
        Ok(Self { w, b })
    }
}

impl ImageClassificationModel {
    /// Load a full `SegformerForImageClassification` HF checkpoint.
    /// Naming: `segformer.encoder.*` for the backbone and `classifier.*`
    /// for the head.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: SegformerConfig,
        num_labels: usize,
    ) -> Result<Self> {
        let encoder = SegformerEncoderWeights::load_from_mmapped(
            st, &cfg, "segformer.encoder.",
        )?;
        let classifier = SegformerClassifierWeights::load_from_mmapped(
            st, &cfg, "", num_labels,
        )?;
        Ok(Self { config: cfg, encoder, classifier })
    }
}

impl SemanticSegmentationModel {
    /// Load a full `SegformerForSemanticSegmentation` HF checkpoint.
    /// Naming: `segformer.encoder.*` and `decode_head.*`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: SegformerConfig,
        num_labels: usize,
    ) -> Result<Self> {
        let encoder = SegformerEncoderWeights::load_from_mmapped(
            st, &cfg, "segformer.encoder.",
        )?;
        let decode_head = SegformerDecodeHeadWeights::load_from_mmapped(
            st, &cfg, "decode_head.", num_labels,
        )?;
        Ok(Self {
            config: cfg,
            encoder,
            decode_head,
            num_labels,
        })
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
    fn ln_w(c: usize) -> LayerNormWeights {
        LayerNormWeights {
            gain: Arc::from(vec![1.0_f32; c]),
            bias: Arc::from(vec![0.0_f32; c]),
        }
    }

    fn conv2d_w(
        c_in: usize, c_out: usize, k: usize, stride: usize, pad: usize, groups: usize,
        bias: bool, nb: &mut dyn FnMut() -> f32,
    ) -> Conv2dWeights {
        Conv2dWeights {
            w: vec_of(c_out * (c_in / groups) * k * k, nb),
            b: if bias { Some(vec_of(c_out, nb)) } else { None },
            c_in, c_out, k, stride, pad, groups,
        }
    }

    fn attn_w(
        hidden: usize, sr_ratio: usize, layer_norm_eps_present: bool,
        nb: &mut dyn FnMut() -> f32,
    ) -> EfficientSelfAttentionWeights {
        EfficientSelfAttentionWeights {
            query: ws(hidden * hidden, nb), query_bias: vec_of(hidden, nb),
            key: ws(hidden * hidden, nb), key_bias: vec_of(hidden, nb),
            value: ws(hidden * hidden, nb), value_bias: vec_of(hidden, nb),
            sr: if sr_ratio > 1 {
                Some(conv2d_w(hidden, hidden, sr_ratio, sr_ratio, 0, 1, true, nb))
            } else { None },
            sr_norm: if sr_ratio > 1 && layer_norm_eps_present {
                Some(ln_w(hidden))
            } else { None },
        }
    }

    fn mix_ffn_w(
        hidden: usize, mlp_ratio: usize, nb: &mut dyn FnMut() -> f32,
    ) -> MixFfnWeights {
        let hf = hidden * mlp_ratio;
        MixFfnWeights {
            dense1: ws(hidden * hf, nb), dense1_bias: vec_of(hf, nb),
            dw_conv: conv2d_w(hf, hf, 3, 1, 1, hf, true, nb),
            dense2: ws(hf * hidden, nb), dense2_bias: vec_of(hidden, nb),
        }
    }

    fn layer_w(
        hidden: usize, num_heads: usize, sr_ratio: usize, mlp_ratio: usize,
        nb: &mut dyn FnMut() -> f32,
    ) -> SegformerLayerWeights {
        SegformerLayerWeights {
            layer_norm_1: ln_w(hidden),
            attention: attn_w(hidden, sr_ratio, true, nb),
            attention_output: AttentionOutputWeights {
                dense: ws(hidden * hidden, nb),
                dense_bias: vec_of(hidden, nb),
            },
            layer_norm_2: ln_w(hidden),
            mlp: mix_ffn_w(hidden, mlp_ratio, nb),
            hidden_size: hidden,
            num_heads,
        }
    }

    fn tiny_config() -> SegformerConfig {
        SegformerConfig {
            num_channels: 3, num_encoder_blocks: 4,
            depths: vec![1, 1, 1, 1],
            sr_ratios: vec![4, 2, 1, 1],
            hidden_sizes: vec![8, 16, 32, 64],
            patch_sizes: vec![3, 3, 3, 3],
            strides: vec![2, 2, 2, 2],
            num_attention_heads: vec![1, 2, 4, 8],
            mlp_ratios: vec![2, 2, 2, 2],
            hidden_act: SegformerActivation::Gelu,
            layer_norm_eps: 1e-6,
            decoder_hidden_size: 16,
        }
    }

    fn tiny_encoder_weights(cfg: &SegformerConfig) -> SegformerEncoderWeights {
        let mut nb = rng_seed(31337);
        let mut stages = Vec::with_capacity(cfg.num_encoder_blocks);
        for i in 0..cfg.num_encoder_blocks {
            let c_in = if i == 0 { cfg.num_channels } else { cfg.hidden_sizes[i - 1] };
            let c_out = cfg.hidden_sizes[i];
            let patch_size = cfg.patch_sizes[i];
            let stride = cfg.strides[i];
            let pe = OverlapPatchEmbeddingWeights {
                projection: conv2d_w(c_in, c_out, patch_size, stride, patch_size / 2, 1, true, &mut nb),
                layer_norm: ln_w(c_out),
            };
            let mut layers = Vec::with_capacity(cfg.depths[i]);
            for _ in 0..cfg.depths[i] {
                layers.push(layer_w(c_out, cfg.num_attention_heads[i],
                    cfg.sr_ratios[i], cfg.mlp_ratios[i], &mut nb));
            }
            stages.push(SegformerStageWeights {
                patch_embedding: pe,
                layers,
                final_ln: ln_w(c_out),
            });
        }
        SegformerEncoderWeights { stages }
    }

    fn tiny_classifier_weights(cfg: &SegformerConfig, num_labels: usize) -> SegformerClassifierWeights {
        let mut nb = rng_seed(99);
        let c = *cfg.hidden_sizes.last().unwrap();
        SegformerClassifierWeights {
            w: ws(c * num_labels, &mut nb),
            b: vec_of(num_labels, &mut nb),
        }
    }

    fn tiny_decode_weights(cfg: &SegformerConfig, num_labels: usize) -> SegformerDecodeHeadWeights {
        let mut nb = rng_seed(55);
        let mut linear_c = Vec::with_capacity(cfg.num_encoder_blocks);
        for i in 0..cfg.num_encoder_blocks {
            linear_c.push((ws(cfg.hidden_sizes[i] * cfg.decoder_hidden_size, &mut nb),
                           vec_of(cfg.decoder_hidden_size, &mut nb)));
        }
        SegformerDecodeHeadWeights {
            linear_c,
            linear_fuse: conv2d_w(
                cfg.decoder_hidden_size * cfg.num_encoder_blocks, cfg.decoder_hidden_size,
                1, 1, 0, 1, false, &mut nb,
            ),
            batch_norm: BatchNormParams {
                w: Arc::from(vec![1.0_f32; cfg.decoder_hidden_size]),
                b: Arc::from(vec![0.0_f32; cfg.decoder_hidden_size]),
            },
            classifier: conv2d_w(
                cfg.decoder_hidden_size, num_labels, 1, 1, 0, 1, false, &mut nb,
            ),
        }
    }

    #[test]
    fn image_classification_shape_and_finite() {
        let cfg = tiny_config();
        let enc = tiny_encoder_weights(&cfg);
        let n_labels = 5;
        let cls = tiny_classifier_weights(&cfg, n_labels);
        let model = ImageClassificationModel {
            config: cfg, encoder: enc, classifier: cls,
        };
        // Image 32x32. Stride pipeline 2*2*2*2 = 16 → stage 0: 16,
        // stage 1: 8, stage 2: 4, stage 3: 2. Patch padding lifts
        // slightly but the shape is fine for the test.
        let img = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, n_labels]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn semantic_segmentation_shape_and_finite() {
        let cfg = tiny_config();
        let enc = tiny_encoder_weights(&cfg);
        let n_labels = 4;
        let dec = tiny_decode_weights(&cfg, n_labels);
        let model = SemanticSegmentationModel {
            config: cfg, encoder: enc, decode_head: dec,
            num_labels: n_labels,
        };
        let img = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let logits = model.forward(&img).unwrap();
        let shape = logits.shape();
        let dims = shape.dims();
        // Output is (1, num_labels, h_stage0, w_stage0).
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], n_labels);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite seg logit: {v}");
        }
    }

    /// Different images must produce different image-classification
    /// logits — proves the hierarchical encoder + classifier is wired.
    #[test]
    fn classification_responds_to_input() {
        let cfg = tiny_config();
        let enc = tiny_encoder_weights(&cfg);
        let cls = tiny_classifier_weights(&cfg, 4);
        let model = ImageClassificationModel {
            config: cfg, encoder: enc, classifier: cls,
        };
        let a = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let b = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01 + 0.7).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let la = model.forward(&a).unwrap().realize_f32();
        let lb = model.forward(&b).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in la.iter().zip(lb.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "classifier must respond to input, max_diff = {max_diff}");
    }

    #[test]
    fn mit_b0_preset_constructs() {
        let cfg = SegformerConfig::mit_b0();
        assert_eq!(cfg.hidden_sizes, vec![32, 64, 160, 256]);
        assert_eq!(cfg.sr_ratios, vec![8, 4, 2, 1]);
        assert_eq!(cfg.decoder_hidden_size, 256);
    }

    #[test]
    fn mit_b1_through_b5_presets_construct() {
        let presets = [
            (SegformerConfig::mit_b1(), [64usize, 128, 320, 512], vec![2usize, 2, 2, 2], 256),
            (SegformerConfig::mit_b2(), [64, 128, 320, 512], vec![3, 4, 6, 3], 768),
            (SegformerConfig::mit_b3(), [64, 128, 320, 512], vec![3, 4, 18, 3], 768),
            (SegformerConfig::mit_b4(), [64, 128, 320, 512], vec![3, 8, 27, 3], 768),
            (SegformerConfig::mit_b5(), [64, 128, 320, 512], vec![3, 6, 40, 3], 768),
        ];
        for (cfg, hs, depths, dh) in presets {
            assert_eq!(cfg.hidden_sizes, hs.to_vec());
            assert_eq!(cfg.depths, depths);
            assert_eq!(cfg.decoder_hidden_size, dh);
            assert_eq!(cfg.sr_ratios, vec![8, 4, 2, 1]);
            assert_eq!(cfg.patch_sizes, vec![7, 3, 3, 3]);
            assert_eq!(cfg.strides, vec![4, 2, 2, 2]);
            assert_eq!(cfg.num_attention_heads, vec![1, 2, 5, 8]);
            assert_eq!(cfg.mlp_ratios, vec![4, 4, 4, 4]);
            assert_eq!(cfg.layer_norm_eps, 1e-6);
        }
    }

    // ---- Safetensors round-trip ------------------------------------------

    /// Append `n` f32 values to `owned` under `name` as a 1-D shape.
    fn push_f32_1d(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        name: &str,
        values: &[f32],
    ) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values { bytes.extend_from_slice(&v.to_le_bytes()); }
        owned.push((name.to_string(), vec![values.len()], bytes));
    }

    /// Append a multi-dim f32 tensor of given shape, filled by `nb()`.
    fn push_f32(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        name: &str,
        shape: Vec<usize>,
        nb: &mut dyn FnMut() -> f32,
    ) {
        let n: usize = shape.iter().product();
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n { bytes.extend_from_slice(&nb().to_le_bytes()); }
        owned.push((name.to_string(), shape, bytes));
    }

    /// Push a HuggingFace LayerNorm prefix (`<prefix>.{weight,bias}`).
    fn push_ln(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        c: usize,
        nb: &mut dyn FnMut() -> f32,
    ) {
        push_f32(owned, &format!("{prefix}.weight"), vec![c], nb);
        push_f32(owned, &format!("{prefix}.bias"),   vec![c], nb);
    }

    /// Push a HuggingFace BatchNorm prefix (4 stats).
    fn push_bn(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        c: usize,
        nb: &mut dyn FnMut() -> f32,
    ) {
        // weight (gain), bias, running_mean, running_var.
        push_f32_1d(owned, &format!("{prefix}.weight"),
            &(0..c).map(|_| nb()).collect::<Vec<_>>());
        push_f32_1d(owned, &format!("{prefix}.bias"),
            &(0..c).map(|_| nb()).collect::<Vec<_>>());
        push_f32_1d(owned, &format!("{prefix}.running_mean"),
            &(0..c).map(|_| nb()).collect::<Vec<_>>());
        // running_var must be strictly positive for inv-sqrt to be finite.
        push_f32_1d(owned, &format!("{prefix}.running_var"),
            &(0..c).map(|_| nb().abs() + 0.1).collect::<Vec<_>>());
    }

    fn push_conv2d(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        c_in: usize,
        c_out: usize,
        k: usize,
        groups: usize,
        has_bias: bool,
        nb: &mut dyn FnMut() -> f32,
    ) {
        push_f32(
            owned, &format!("{prefix}.weight"),
            vec![c_out, c_in / groups, k, k], nb,
        );
        if has_bias {
            push_f32(owned, &format!("{prefix}.bias"), vec![c_out], nb);
        }
    }

    fn push_linear(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        nb: &mut dyn FnMut() -> f32,
    ) {
        // HF stores weight as [out, in].
        push_f32(
            owned, &format!("{prefix}.weight"),
            vec![out_features, in_features], nb,
        );
        push_f32(owned, &format!("{prefix}.bias"), vec![out_features], nb);
    }

    fn push_attention(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        hidden: usize,
        sr_ratio: usize,
        nb: &mut dyn FnMut() -> f32,
    ) {
        // attention.self.{query,key,value}
        push_linear(owned, &format!("{prefix}.self.query"), hidden, hidden, nb);
        push_linear(owned, &format!("{prefix}.self.key"),   hidden, hidden, nb);
        push_linear(owned, &format!("{prefix}.self.value"), hidden, hidden, nb);
        if sr_ratio > 1 {
            push_conv2d(
                owned, &format!("{prefix}.self.sr"),
                hidden, hidden, sr_ratio, 1, true, nb,
            );
            push_ln(owned, &format!("{prefix}.self.layer_norm"), hidden, nb);
        }
        // attention.output.dense
        push_linear(owned, &format!("{prefix}.output.dense"), hidden, hidden, nb);
    }

    fn push_mix_ffn(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        hidden: usize,
        mlp_ratio: usize,
        nb: &mut dyn FnMut() -> f32,
    ) {
        let hf = hidden * mlp_ratio;
        push_linear(owned, &format!("{prefix}.dense1"), hidden, hf, nb);
        // SegformerDWConv → conv2d at `<prefix>.dwconv.dwconv`.
        push_conv2d(
            owned, &format!("{prefix}.dwconv.dwconv"),
            hf, hf, 3, hf, true, nb,
        );
        push_linear(owned, &format!("{prefix}.dense2"), hf, hidden, nb);
    }

    fn push_layer(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        hidden: usize,
        sr_ratio: usize,
        mlp_ratio: usize,
        nb: &mut dyn FnMut() -> f32,
    ) {
        push_ln(owned, &format!("{prefix}.layer_norm_1"), hidden, nb);
        push_attention(owned, &format!("{prefix}.attention"), hidden, sr_ratio, nb);
        push_ln(owned, &format!("{prefix}.layer_norm_2"), hidden, nb);
        push_mix_ffn(owned, &format!("{prefix}.mlp"), hidden, mlp_ratio, nb);
    }

    fn push_encoder(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        cfg: &SegformerConfig,
        nb: &mut dyn FnMut() -> f32,
    ) {
        for i in 0..cfg.num_encoder_blocks {
            let c_in = if i == 0 { cfg.num_channels } else { cfg.hidden_sizes[i - 1] };
            let c_out = cfg.hidden_sizes[i];
            push_conv2d(
                owned, &format!("{prefix}patch_embeddings.{i}.proj"),
                c_in, c_out, cfg.patch_sizes[i], 1, true, nb,
            );
            push_ln(owned, &format!("{prefix}patch_embeddings.{i}.layer_norm"), c_out, nb);
            for j in 0..cfg.depths[i] {
                push_layer(
                    owned, &format!("{prefix}block.{i}.{j}"),
                    c_out, cfg.sr_ratios[i], cfg.mlp_ratios[i], nb,
                );
            }
            push_ln(owned, &format!("{prefix}layer_norm.{i}"), c_out, nb);
        }
    }

    fn build_safetensors_file(
        owned: Vec<(String, Vec<usize>, Vec<u8>)>,
        tag: &str,
    ) -> std::path::PathBuf {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        for (name, shape, bytes) in &owned {
            let view = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                .expect("TensorView::new");
            tensors.insert(name.clone(), view);
        }
        let serialized = safetensors::serialize(&tensors, None)
            .expect("safetensors::serialize");
        let tmp = std::env::temp_dir().join(format!(
            "fuel_segformer_load_test_{}_{tag}.safetensors",
            std::process::id(),
        ));
        std::fs::write(&tmp, &serialized).expect("write tmp");
        tmp
    }

    #[test]
    fn load_from_mmapped_classification_round_trip() {
        let cfg = tiny_config();
        let n_labels = 5;
        let mut nb = rng_seed(7);

        let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        push_encoder(&mut owned, "segformer.encoder.", &cfg, &mut nb);
        // classifier (top-level Linear hidden_sizes[-1] → num_labels).
        let c_last = *cfg.hidden_sizes.last().unwrap();
        push_linear(&mut owned, "classifier", c_last, n_labels, &mut nb);

        let tmp = build_safetensors_file(owned, "cls");
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&tmp) }
            .expect("MmapedSafetensors::new");

        let model = ImageClassificationModel::load_from_mmapped(&st, cfg, n_labels)
            .expect("load classification model");

        // Sanity: shape + finiteness on a tiny image.
        let img = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, n_labels]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit from loaded model: {v}");
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn load_from_mmapped_segmentation_round_trip() {
        let cfg = tiny_config();
        let n_labels = 4;
        let mut nb = rng_seed(11);

        let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        push_encoder(&mut owned, "segformer.encoder.", &cfg, &mut nb);
        // Decode head: linear_c.{i}.proj, linear_fuse (no bias), batch_norm,
        // classifier (Conv2d w/ bias).
        for i in 0..cfg.num_encoder_blocks {
            push_linear(
                &mut owned, &format!("decode_head.linear_c.{i}.proj"),
                cfg.hidden_sizes[i], cfg.decoder_hidden_size, &mut nb,
            );
        }
        push_conv2d(
            &mut owned, "decode_head.linear_fuse",
            cfg.decoder_hidden_size * cfg.num_encoder_blocks,
            cfg.decoder_hidden_size, 1, 1, false, &mut nb,
        );
        push_bn(&mut owned, "decode_head.batch_norm", cfg.decoder_hidden_size, &mut nb);
        push_conv2d(
            &mut owned, "decode_head.classifier",
            cfg.decoder_hidden_size, n_labels, 1, 1, true, &mut nb,
        );

        let tmp = build_safetensors_file(owned, "seg");
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&tmp) }
            .expect("MmapedSafetensors::new");

        let model = SemanticSegmentationModel::load_from_mmapped(&st, cfg, n_labels)
            .expect("load segmentation model");
        assert_eq!(model.num_labels, n_labels);

        let img = LazyTensor::from_f32(
            (0..(3 * 32 * 32)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 32, 32]), &Device::cpu(),
        );
        let logits = model.forward(&img).unwrap();
        let dims = logits.shape();
        let dims = dims.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], n_labels);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite seg logit from loaded model: {v}");
        }

        let _ = std::fs::remove_file(&tmp);
    }
}

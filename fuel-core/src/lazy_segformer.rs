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
    /// HuggingFace MIT-B0 preset (224×224).
    pub fn mit_b0() -> Self {
        Self {
            num_channels: 3,
            num_encoder_blocks: 4,
            depths: vec![2, 2, 2, 2],
            sr_ratios: vec![8, 4, 2, 1],
            hidden_sizes: vec![32, 64, 160, 256],
            patch_sizes: vec![7, 3, 3, 3],
            strides: vec![4, 2, 2, 2],
            num_attention_heads: vec![1, 2, 5, 8],
            mlp_ratios: vec![4, 4, 4, 4],
            hidden_act: SegformerActivation::Gelu,
            layer_norm_eps: 1e-6,
            decoder_hidden_size: 256,
        }
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
    let normed = apply_layer_norm(&seq, ln, c, eps)?;
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
    let normed = apply_layer_norm(&seq_orig, &w.layer_norm_1, c, cfg.layer_norm_eps)?;
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
    let normed = apply_layer_norm(&hidden, &w.layer_norm_2, c, cfg.layer_norm_eps)?;
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

    let q = apply_linear_with_bias(
        &x_seq, &attn.query, &attn.query_bias, hidden_size, hidden_size, anchor,
    )?;

    let kv_seq = if let (Some(sr), Some(sr_norm)) = (&attn.sr, &attn.sr_norm) {
        let sr_out = apply_conv2d(x, sr, anchor)?;
        let sr_dims = sr_out.shape();
        let sr_dims = sr_dims.dims();
        let h2 = sr_dims[2]; let w2 = sr_dims[3];
        let flat = sr_out
            .reshape(Shape::from_dims(&[b, hidden_size, h2 * w2]))?
            .permute([0, 2, 1_usize])?;
        apply_layer_norm(&flat, sr_norm, hidden_size, eps)?
    } else {
        x_seq.clone()
    };

    let kv_dims = kv_seq.shape();
    let kv_len = kv_dims.dims()[1];
    let k = apply_linear_with_bias(
        &kv_seq, &attn.key, &attn.key_bias, hidden_size, hidden_size, anchor,
    )?;
    let v = apply_linear_with_bias(
        &kv_seq, &attn.value, &attn.value_bias, hidden_size, hidden_size, anchor,
    )?;

    let q = q.reshape(Shape::from_dims(&[b, q_len, num_heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;
    let k = k.reshape(Shape::from_dims(&[b, kv_len, num_heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;
    let v = v.reshape(Shape::from_dims(&[b, kv_len, num_heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;

    let kt = k.permute([0, 1, 3, 2_usize])?;
    let scores = q.matmul(&kt)?.mul_scalar(scale);
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?;
    let ctx = ctx
        .permute([0, 2, 1, 3_usize])?
        .reshape(Shape::from_dims(&[b, q_len, hidden_size]))?;
    apply_linear_with_bias(
        &ctx, &out.dense, &out.dense_bias, hidden_size, hidden_size, anchor,
    )
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
    let h1 = apply_linear_with_bias(
        &seq, &m.dense1, &m.dense1_bias, hidden_size, hidden_features, anchor,
    )?;
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
    let h3 = apply_linear_with_bias(
        &seq, &m.dense2, &m.dense2_bias, hidden_features, hidden_size, anchor,
    )?;
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
        let projected = apply_linear_with_bias(
            &seq, w, bias, c, cfg.decoder_hidden_size, anchor,
        )?;
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

fn apply_layer_norm(
    x: &LazyTensor,
    ln: &LayerNormWeights,
    hidden: usize,
    eps: f64,
) -> Result<LazyTensor> {
    let last = x.shape().dims().last().copied().unwrap();
    assert_eq!(last, hidden);
    x.layer_norm_affine(Arc::clone(&ln.gain), Arc::clone(&ln.bias), eps)
}

fn apply_linear_with_bias(
    x: &LazyTensor,
    w: &WeightStorage,
    b: &Arc<[f32]>,
    in_features: usize,
    out_features: usize,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let _ = anchor;
    w.apply_linear_with_bias(x, in_features, out_features, Arc::clone(b))
}

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
    assert_eq!(bn.w.len(), channels);
    let w_t = x
        .const_f32_like(Arc::clone(&bn.w), Shape::from_dims(&[channels]))
        .reshape(Shape::from_dims(&[1, channels, 1, 1]))?;
    let b_t = x
        .const_f32_like(Arc::clone(&bn.b), Shape::from_dims(&[channels]))
        .reshape(Shape::from_dims(&[1, channels, 1, 1]))?;
    x.broadcast_mul(&w_t)?.broadcast_add(&b_t)
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
}

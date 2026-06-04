//! BLIP vision encoder — lazy port.
//!
//! Standard ViT-shape Pre-LN transformer that encodes an image
//! into per-token hidden states `(1, num_patches + 1, hidden_size)`.
//! The +1 accounts for the prepended CLS token.
//!
//! Distinctive from `lazy_vit`:
//!   - **Fused QKV** linear (single `[3·hidden, hidden]` weight)
//!     vs `lazy_vit`'s separate Q / K / V projections — matches
//!     PyTorch BLIP's `qkv` linear layer.
//!   - Output linear is named `projection` (not `out_proj`).
//!   - Optional attention-mask add path on the softmax probs
//!     (multiplied, not added — eager BLIP uses
//!     `attn_probs * attn_mask` rather than additive mask). v1
//!     omits the mask path since BLIP captioning never passes
//!     one through the vision side.
//!
//! Used by BlipForConditionalGeneration (image captioning).
//!
//! v1 scope: F32, batch == 1, prefill only.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlipVisionActivation {
    Gelu,
    GeluPytorchTanh,
    Relu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlipVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_activation: BlipVisionActivation,
    pub layer_norm_eps: f64,
}

impl BlipVisionConfig {
    /// `Salesforce/blip-image-captioning-base` (ViT-Base).
    pub fn image_captioning_base() -> Self {
        Self {
            hidden_size: 768, intermediate_size: 3072,
            num_hidden_layers: 12, num_attention_heads: 12,
            image_size: 384, patch_size: 16,
            hidden_activation: BlipVisionActivation::Gelu,
            layer_norm_eps: 1e-5,
        }
    }

    /// `Salesforce/blip-image-captioning-large` (ViT-Large).
    pub fn image_captioning_large() -> Self {
        Self {
            hidden_size: 1024, intermediate_size: 4096,
            num_hidden_layers: 24, num_attention_heads: 16,
            image_size: 384, patch_size: 16,
            hidden_activation: BlipVisionActivation::Gelu,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }
    pub fn num_patches(&self) -> usize {
        let p = self.num_patches_per_side();
        p * p
    }
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    pub gain: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct BlipVisionAttentionWeights {
    /// Fused QKV: `[3·hidden_size, hidden_size]` (stored as
    /// `[hidden_size, 3·hidden_size]` after load-time transpose
    /// to match `WeightStorage::apply_linear`'s convention).
    pub qkv: WeightStorage,
    pub qkv_bias: Arc<[f32]>,
    pub projection: WeightStorage,
    pub projection_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct BlipMlpWeights {
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct BlipVisionLayerWeights {
    pub ln1: LayerNormWeights,
    pub attn: BlipVisionAttentionWeights,
    pub ln2: LayerNormWeights,
    pub mlp: BlipMlpWeights,
}

#[derive(Debug, Clone)]
pub struct BlipVisionWeights {
    /// `[hidden_size, 3, patch_size, patch_size]`.
    pub patch_proj: Arc<[f32]>,
    /// `[hidden_size]`.
    pub patch_proj_bias: Arc<[f32]>,
    /// `[hidden_size]` — the CLS token.
    pub class_token: Arc<[f32]>,
    /// `[num_patches + 1, hidden_size]`.
    pub position_embedding: Arc<[f32]>,
    pub layers: Vec<BlipVisionLayerWeights>,
    /// Post-encoder LN.
    pub post_layernorm: LayerNormWeights,
}

#[derive(Debug, Clone)]
pub struct BlipVisionModel {
    pub config: BlipVisionConfig,
    pub weights: BlipVisionWeights,
}

impl BlipVisionModel {
    /// Encode an image `(1, 3, H, W)` (H = W = image_size) into
    /// per-token hidden states `(1, num_patches + 1, hidden_size)`.
    /// The first token is CLS; the rest are patch tokens in
    /// row-major order.
    pub fn forward(&self, pixel_values: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = pixel_values.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        assert_eq!(dims[0], 1, "v1 supports batch == 1");
        assert_eq!(dims[1], 3, "image must have 3 input channels");
        assert_eq!(dims[2], cfg.image_size);
        assert_eq!(dims[3], cfg.image_size);

        let weights = &self.weights;
        let np = cfg.num_patches();

        // Patch embedding via stride-= patch_size conv.
        let conv_w = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj),
            Shape::from_dims(&[cfg.hidden_size, 3, cfg.patch_size, cfg.patch_size]),
        );
        let conv_b = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj_bias),
            Shape::from_dims(&[cfg.hidden_size]),
        );
        let conv_out = pixel_values.conv2d(
            &conv_w, Some(&conv_b),
            (cfg.patch_size, cfg.patch_size),
            (0, 0),
            1,
        )?;
        // (1, hidden, num_patches_per_side, num_patches_per_side) → (1, hidden, np) → (1, np, hidden)
        let patches = conv_out
            .reshape(Shape::from_dims(&[1, cfg.hidden_size, np]))?
            .permute([0, 2, 1_usize])?;

        // CLS token + positional embedding.
        let cls = pixel_values
            .const_f32_like(
                Arc::clone(&weights.class_token),
                Shape::from_dims(&[1, 1, cfg.hidden_size]),
            );
        let with_cls = cls.concat(&patches, 1_usize)?;
        let pos = pixel_values.const_f32_like(
            Arc::clone(&weights.position_embedding),
            Shape::from_dims(&[1, np + 1, cfg.hidden_size]),
        );
        let mut x = with_cls.add(&pos)?;

        // Transformer encoder layers (Pre-LN).
        for layer in &weights.layers {
            x = apply_layer(&x, layer, cfg, pixel_values)?;
        }

        // Post-encoder LN.
        apply_layer_norm(&x, &weights.post_layernorm, cfg.hidden_size, cfg.layer_norm_eps)
    }
}

fn apply_layer(
    x: &LazyTensor,
    w: &BlipVisionLayerWeights,
    cfg: &BlipVisionConfig,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let residual = x.clone();
    let normed = apply_layer_norm(x, &w.ln1, cfg.hidden_size, cfg.layer_norm_eps)?;
    let attn_out = apply_attention(&normed, &w.attn, cfg, anchor)?;
    let x = residual.add(&attn_out)?;

    let residual = x.clone();
    let normed = apply_layer_norm(&x, &w.ln2, cfg.hidden_size, cfg.layer_norm_eps)?;
    let mlp_out = apply_mlp(&normed, &w.mlp, cfg, anchor)?;
    residual.add(&mlp_out)
}

fn apply_attention(
    x: &LazyTensor,
    w: &BlipVisionAttentionWeights,
    cfg: &BlipVisionConfig,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let seq = dims[1];
    let embed = cfg.hidden_size;
    let n_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim();
    let scale = 1.0_f64 / (head_dim as f64).sqrt();

    // Fused QKV: project to 3·hidden then split.
    let qkv = apply_linear_with_bias(x, &w.qkv, &w.qkv_bias, embed, 3 * embed, anchor)?;
    let q = qkv.narrow(2_usize, 0, embed)?;
    let k = qkv.narrow(2_usize, embed, embed)?;
    let v = qkv.narrow(2_usize, 2 * embed, embed)?;

    // (B, seq, embed) → (B, n_heads, seq, head_dim).
    let _ = (b, seq, embed);
    let q = q.split_heads(n_heads, head_dim)?;
    let k = k.split_heads(n_heads, head_dim)?;
    let v = v.split_heads(n_heads, head_dim)?;

    let kt = k.permute([0, 1, 3, 2_usize])?;
    let scores = q.matmul(&kt)?.mul_scalar(scale);
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?.merge_heads()?;
    apply_linear_with_bias(&ctx, &w.projection, &w.projection_bias, embed, embed, anchor)
}

fn apply_mlp(
    x: &LazyTensor,
    m: &BlipMlpWeights,
    cfg: &BlipVisionConfig,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let h1 = apply_linear_with_bias(
        x, &m.fc1, &m.fc1_bias, cfg.hidden_size, cfg.intermediate_size, anchor,
    )?;
    let h1 = match cfg.hidden_activation {
        BlipVisionActivation::Gelu => h1.gelu(),
        BlipVisionActivation::GeluPytorchTanh => h1.gelu_erf(),
        BlipVisionActivation::Relu => h1.relu(),
    };
    apply_linear_with_bias(
        &h1, &m.fc2, &m.fc2_bias, cfg.intermediate_size, cfg.hidden_size, anchor,
    )
}

fn apply_layer_norm(
    x: &LazyTensor,
    ln: &LayerNormWeights,
    hidden: usize,
    eps: f64,
) -> Result<LazyTensor> {
    let _ = hidden;
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

    fn tiny_config() -> BlipVisionConfig {
        BlipVisionConfig {
            hidden_size: 8, intermediate_size: 16,
            num_hidden_layers: 2, num_attention_heads: 2,
            image_size: 8, patch_size: 4,
            hidden_activation: BlipVisionActivation::Gelu,
            layer_norm_eps: 1e-5,
        }
    }

    fn tiny_weights(cfg: &BlipVisionConfig) -> BlipVisionWeights {
        let mut nb = rng_seed(2026);
        let h = cfg.hidden_size;
        let np = cfg.num_patches();
        let layers: Vec<BlipVisionLayerWeights> = (0..cfg.num_hidden_layers).map(|_| {
            BlipVisionLayerWeights {
                ln1: ln_w(h),
                attn: BlipVisionAttentionWeights {
                    qkv: ws(h * 3 * h, &mut nb),
                    qkv_bias: vec_of(3 * h, &mut nb),
                    projection: ws(h * h, &mut nb),
                    projection_bias: vec_of(h, &mut nb),
                },
                ln2: ln_w(h),
                mlp: BlipMlpWeights {
                    fc1: ws(h * cfg.intermediate_size, &mut nb),
                    fc1_bias: vec_of(cfg.intermediate_size, &mut nb),
                    fc2: ws(cfg.intermediate_size * h, &mut nb),
                    fc2_bias: vec_of(h, &mut nb),
                },
            }
        }).collect();
        BlipVisionWeights {
            patch_proj: vec_of(h * 3 * cfg.patch_size * cfg.patch_size, &mut nb),
            patch_proj_bias: vec_of(h, &mut nb),
            class_token: vec_of(h, &mut nb),
            position_embedding: vec_of((np + 1) * h, &mut nb),
            layers,
            post_layernorm: ln_w(h),
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = BlipVisionModel { config: cfg.clone(), weights };
        let img = LazyTensor::from_f32(
            (0..(3 * cfg.image_size * cfg.image_size))
                .map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, cfg.image_size, cfg.image_size]),
            &Device::cpu(),
        );
        let out = model.forward(&img).unwrap();
        let np = cfg.num_patches();
        assert_eq!(out.shape().dims(), &[1, np + 1, cfg.hidden_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    #[test]
    fn forward_responds_to_image() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = BlipVisionModel { config: cfg.clone(), weights };
        let img_a = LazyTensor::from_f32(
            (0..(3 * cfg.image_size * cfg.image_size))
                .map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, cfg.image_size, cfg.image_size]),
            &Device::cpu(),
        );
        let img_b = LazyTensor::from_f32(
            (0..(3 * cfg.image_size * cfg.image_size))
                .map(|i| (i as f32) * 0.01 + 0.5).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, cfg.image_size, cfg.image_size]),
            &Device::cpu(),
        );
        let a = model.forward(&img_a).unwrap().realize_f32();
        let b = model.forward(&img_b).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "BLIP vision must respond to input changes, max_diff = {max_diff}");
    }

    #[test]
    fn preset_constructs() {
        let base = BlipVisionConfig::image_captioning_base();
        assert_eq!(base.hidden_size, 768);
        assert_eq!(base.num_hidden_layers, 12);
        assert_eq!(base.image_size, 384);
        let large = BlipVisionConfig::image_captioning_large();
        assert_eq!(large.hidden_size, 1024);
        assert_eq!(large.num_hidden_layers, 24);
    }
}

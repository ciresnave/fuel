//! EVA-02 — lazy port.
//!
//! ViT-shape Pre-LN backbone with **interleaved RoPE** applied to
//! patch tokens of Q and K (CLS token skipped), and a **SwiGLU-LN**
//! MLP variant: `xs_g = silu(fc1_g(x)); xs_x = fc1_x(x);
//! xs = LN(xs_g * xs_x); out = fc2(xs)`.
//!
//! v1 scope: F32, batch == 1, 448×448 input only (canonical
//! `img_size`), forward-only inference.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct EvaConfig {
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub img_size: usize,
    pub patch_size: usize,
    pub num_classes: Option<usize>,
}

impl EvaConfig {
    /// EVA-02 ViT-Base/14 at 448×448.
    pub fn vit_base() -> Self {
        Self {
            embed_dim: 768, depth: 12, num_heads: 12,
            img_size: 448, patch_size: 14,
            num_classes: Some(1000),
        }
    }
    /// EVA-02 ViT-Large/14 at 448×448.
    pub fn vit_large() -> Self {
        Self {
            embed_dim: 1024, depth: 24, num_heads: 16,
            img_size: 448, patch_size: 14,
            num_classes: Some(1000),
        }
    }
    pub fn head_dim(&self) -> usize { self.embed_dim / self.num_heads }
    pub fn num_patches(&self) -> usize {
        let g = self.img_size / self.patch_size;
        g * g
    }
    pub fn hidden_mlp_dim(&self) -> usize { self.embed_dim * 4 * 2 / 3 }
}

// ---- Weight structures ------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    pub gain: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct EvaAttentionWeights {
    /// Q has bias, K has NO bias, V has bias. All `[embed, embed]`.
    pub q_w: WeightStorage, pub q_b: Arc<[f32]>,
    pub k_w: WeightStorage,
    pub v_w: WeightStorage, pub v_b: Arc<[f32]>,
    pub proj_w: WeightStorage, pub proj_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct EvaMlpWeights {
    pub fc1_g_w: WeightStorage, pub fc1_g_b: Arc<[f32]>,
    pub fc1_x_w: WeightStorage, pub fc1_x_b: Arc<[f32]>,
    pub norm: LayerNormWeights,
    pub fc2_w: WeightStorage, pub fc2_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct EvaBlockWeights {
    pub norm1: LayerNormWeights,
    pub attn: EvaAttentionWeights,
    pub norm2: LayerNormWeights,
    pub mlp: EvaMlpWeights,
}

#[derive(Debug, Clone)]
pub struct EvaWeights {
    /// Patch-embed conv `(patch_size×patch_size)` stride `patch_size`:
    /// 3 → embed_dim.
    pub patch_conv_w: Arc<[f32]>,
    pub patch_conv_b: Arc<[f32]>,
    /// `(1, 1, embed_dim)`.
    pub cls_token: Arc<[f32]>,
    /// `(1, num_patches + 1, embed_dim)`.
    pub pos_embed: Arc<[f32]>,
    /// `(num_patches, 2 * head_dim)`: first `head_dim` channels = sin,
    /// last `head_dim` channels = cos. Each (sin[2k], sin[2k+1]) and
    /// (cos[2k], cos[2k+1]) pair shares the same angle (interleaved
    /// RoPE layout).
    pub rot_pos_embed: Arc<[f32]>,
    pub blocks: Vec<EvaBlockWeights>,
    pub norm: LayerNormWeights,
    pub head_w: WeightStorage,
    pub head_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct EvaModel {
    pub config: EvaConfig,
    pub weights: EvaWeights,
}

// ---- Forward ---------------------------------------------------------------

impl EvaModel {
    pub fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = image.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        assert_eq!(dims[1], 3);
        assert_eq!(dims[2], cfg.img_size);
        assert_eq!(dims[3], cfg.img_size);
        let b = dims[0];
        let e = cfg.embed_dim;
        let n_patches = cfg.num_patches();
        let n_tokens = n_patches + 1;

        // Patch embed: conv stride patch_size → (B, E, gH, gW) → (B, gH*gW, E).
        let w = image.const_f32_like(
            Arc::clone(&self.weights.patch_conv_w),
            Shape::from_dims(&[e, 3, cfg.patch_size, cfg.patch_size]),
        );
        let bias = image.const_f32_like(
            Arc::clone(&self.weights.patch_conv_b), Shape::from_dims(&[e]),
        );
        let patches = image.conv2d(&w, Some(&bias), (cfg.patch_size, cfg.patch_size), (0, 0), 1)?;
        let p_dims = patches.shape();
        let p_dims = p_dims.dims();
        let gh = p_dims[2]; let gw = p_dims[3];
        debug_assert_eq!(gh * gw, n_patches);
        let patches = patches
            .reshape(Shape::from_dims(&[b, e, n_patches]))?
            .permute([0, 2, 1_usize])?;

        // Concat CLS token at position 0.
        let cls = image
            .const_f32_like(Arc::clone(&self.weights.cls_token), Shape::from_dims(&[1, 1, e]))
            .broadcast_to(Shape::from_dims(&[b, 1, e]))?;
        let mut x = cls.concat(&patches, 1_usize)?;

        // Add pos_embed.
        let pos = image
            .const_f32_like(Arc::clone(&self.weights.pos_embed), Shape::from_dims(&[1, n_tokens, e]))
            .broadcast_to(Shape::from_dims(&[b, n_tokens, e]))?;
        x = x.add(&pos)?;

        // Rotary embedding split into cos/sin (each `head_dim` wide).
        let head_dim = cfg.head_dim();
        let rot_full = image.const_f32_like(
            Arc::clone(&self.weights.rot_pos_embed),
            Shape::from_dims(&[n_patches, 2 * head_dim]),
        );
        let sin_emb = rot_full.narrow(1_usize, 0, head_dim)?;
        let cos_emb = rot_full.narrow(1_usize, head_dim, head_dim)?;

        for blk in &self.weights.blocks {
            x = apply_block(&x, blk, cfg, &cos_emb, &sin_emb, image)?;
        }

        // Mean over patch tokens (drop CLS), then norm + head.
        let patch_tokens = x.narrow(1_usize, 1, n_patches)?;
        let pooled = patch_tokens.mean_dim(1_usize)?;
        let normed = apply_layer_norm_last(&pooled, &self.weights.norm, e)?;
        let logits = self.weights.head_w.apply_linear(
            &normed, e, self.weights.head_b.len(),
        );
        let head_bias = image.const_f32_like(
            Arc::clone(&self.weights.head_b),
            Shape::from_dims(&[self.weights.head_b.len()]),
        );
        logits.broadcast_add(&head_bias)
    }
}

fn apply_block(
    x: &LazyTensor, blk: &EvaBlockWeights, cfg: &EvaConfig,
    cos_emb: &LazyTensor, sin_emb: &LazyTensor, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let e = cfg.embed_dim;
    let n1 = apply_layer_norm_last(x, &blk.norm1, e)?;
    let attn_out = apply_attention(&n1, blk, cfg, cos_emb, sin_emb, anchor)?;
    let after_attn = x.add(&attn_out)?;
    let n2 = apply_layer_norm_last(&after_attn, &blk.norm2, e)?;
    let mlp_out = apply_mlp(&n2, &blk.mlp, cfg, anchor)?;
    after_attn.add(&mlp_out)
}

fn apply_attention(
    x: &LazyTensor, blk: &EvaBlockWeights, cfg: &EvaConfig,
    cos_emb: &LazyTensor, sin_emb: &LazyTensor, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let n = dims[1]; let e = dims[2];
    debug_assert_eq!(e, cfg.embed_dim);
    let heads = cfg.num_heads;
    let head_dim = cfg.head_dim();
    let scale = 1.0_f64 / (head_dim as f64).sqrt();
    let a = &blk.attn;

    // Three independent linears (Q has bias, K NO bias, V has bias).
    let q = a.q_w.apply_linear(x, e, e);
    let q_bias = anchor.const_f32_like(Arc::clone(&a.q_b), Shape::from_dims(&[e]));
    let q = q.broadcast_add(&q_bias)?;
    let k = a.k_w.apply_linear(x, e, e);
    let v = a.v_w.apply_linear(x, e, e);
    let v_bias = anchor.const_f32_like(Arc::clone(&a.v_b), Shape::from_dims(&[e]));
    let v = v.broadcast_add(&v_bias)?;

    // (B, N, E) → (B, N, heads, head_dim) → (B, heads, N, head_dim)
    let q = q
        .reshape(Shape::from_dims(&[b, n, heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;
    let k = k
        .reshape(Shape::from_dims(&[b, n, heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;
    let v = v
        .reshape(Shape::from_dims(&[b, n, heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;

    // Apply RoPE to patch tokens of Q and K (skip CLS at position 0).
    let q = apply_rope_skip_cls(&q, cos_emb, sin_emb, b, heads, n, head_dim)?;
    let k = apply_rope_skip_cls(&k, cos_emb, sin_emb, b, heads, n, head_dim)?;

    let q = q.mul_scalar(scale);
    let batch = b * heads;
    let q3 = q.reshape(Shape::from_dims(&[batch, n, head_dim]))?;
    let k3 = k.reshape(Shape::from_dims(&[batch, n, head_dim]))?;
    let v3 = v.reshape(Shape::from_dims(&[batch, n, head_dim]))?;
    let kt = k3.permute([0, 2, 1_usize])?;
    let scores = q3.matmul(&kt)?;
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v3)?;
    // (batch, N, head_dim) → (B, heads, N, head_dim) → (B, N, E)
    let ctx = ctx
        .reshape(Shape::from_dims(&[b, heads, n, head_dim]))?
        .permute([0, 2, 1, 3_usize])?
        .reshape(Shape::from_dims(&[b, n, e]))?;
    let projected = a.proj_w.apply_linear(&ctx, e, e);
    let proj_bias = anchor.const_f32_like(Arc::clone(&a.proj_b), Shape::from_dims(&[e]));
    projected.broadcast_add(&proj_bias)
}

/// Apply interleaved RoPE to the patch tokens of `x` while leaving the
/// CLS token (position 0) untouched. `x` is `(B, heads, N, head_dim)`.
/// `cos_emb`/`sin_emb` are `(N - 1, head_dim)`.
fn apply_rope_skip_cls(
    x: &LazyTensor,
    cos_emb: &LazyTensor, sin_emb: &LazyTensor,
    b: usize, heads: usize, n: usize, head_dim: usize,
) -> Result<LazyTensor> {
    let cls = x.narrow(2_usize, 0, 1)?;
    let body = x.narrow(2_usize, 1, n - 1)?;
    let body_rot = apply_rope(&body, cos_emb, sin_emb, b, heads, n - 1, head_dim)?;
    cls.concat(&body_rot, 2_usize)
}

fn apply_rope(
    x: &LazyTensor,
    cos_emb: &LazyTensor, sin_emb: &LazyTensor,
    b: usize, heads: usize, n_body: usize, head_dim: usize,
) -> Result<LazyTensor> {
    // x: (B, heads, n_body, head_dim). emb: (n_body, head_dim).
    // Build rot(x): rot[2k] = -x[2k+1]; rot[2k+1] = x[2k].
    let half = head_dim / 2;
    let evens: Vec<u32> = (0..head_dim as u32).step_by(2).collect();
    let odds: Vec<u32> = (1..head_dim as u32).step_by(2).collect();
    debug_assert_eq!(evens.len(), half);
    debug_assert_eq!(odds.len(), half);

    let even_idx = x.const_u32_like(evens, Shape::from_dims(&[half]));
    let odd_idx = x.const_u32_like(odds, Shape::from_dims(&[half]));
    let x_even = x.index_select(3_usize, &even_idx)?;
    let x_odd_neg = x.index_select(3_usize, &odd_idx)?.mul_scalar(-1.0);

    // Stack along a new last dim, then reshape to interleave:
    // (..., half, 2) → (..., half * 2 = head_dim).
    let rot = LazyTensor::stack(&[&x_odd_neg, &x_even], 4_usize)?
        .reshape(Shape::from_dims(&[b, heads, n_body, head_dim]))?;

    // Broadcast emb (n_body, head_dim) to (B, heads, n_body, head_dim).
    let cos_b = cos_emb
        .reshape(Shape::from_dims(&[1, 1, n_body, head_dim]))?
        .broadcast_to(Shape::from_dims(&[b, heads, n_body, head_dim]))?;
    let sin_b = sin_emb
        .reshape(Shape::from_dims(&[1, 1, n_body, head_dim]))?
        .broadcast_to(Shape::from_dims(&[b, heads, n_body, head_dim]))?;

    let x_cos = x.mul(&cos_b)?;
    let rot_sin = rot.mul(&sin_b)?;
    x_cos.add(&rot_sin)
}

fn apply_mlp(
    x: &LazyTensor, m: &EvaMlpWeights, cfg: &EvaConfig, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let e = cfg.embed_dim;
    let hidden = cfg.hidden_mlp_dim();

    let g = m.fc1_g_w.apply_linear(x, e, hidden);
    let g_b = anchor.const_f32_like(Arc::clone(&m.fc1_g_b), Shape::from_dims(&[hidden]));
    let g = g.broadcast_add(&g_b)?.silu();

    let xv = m.fc1_x_w.apply_linear(x, e, hidden);
    let xv_b = anchor.const_f32_like(Arc::clone(&m.fc1_x_b), Shape::from_dims(&[hidden]));
    let xv = xv.broadcast_add(&xv_b)?;

    let gated = g.mul(&xv)?;
    let normed = apply_layer_norm_last(&gated, &m.norm, hidden)?;

    let out = m.fc2_w.apply_linear(&normed, hidden, e);
    let out_b = anchor.const_f32_like(Arc::clone(&m.fc2_b), Shape::from_dims(&[e]));
    out.broadcast_add(&out_b)
}

fn apply_layer_norm_last(
    x: &LazyTensor, ln: &LayerNormWeights, hidden: usize,
) -> Result<LazyTensor> {
    let _ = hidden;
    x.layer_norm_affine(Arc::clone(&ln.gain), Arc::clone(&ln.bias), 1e-6)
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

    fn tiny_config() -> EvaConfig {
        // 28×28 image, 7×7 patches → 4×4 = 16 patch tokens + 1 CLS = 17 tokens.
        // embed_dim 8, 2 heads → head_dim = 4. rot_pos_embed = (16, 8).
        EvaConfig {
            embed_dim: 8, depth: 2, num_heads: 2,
            img_size: 28, patch_size: 7,
            num_classes: Some(5),
        }
    }

    fn build_attn_w(e: usize, nb: &mut dyn FnMut() -> f32) -> EvaAttentionWeights {
        EvaAttentionWeights {
            q_w: ws(e * e, nb), q_b: vec_of(e, nb),
            k_w: ws(e * e, nb),
            v_w: ws(e * e, nb), v_b: vec_of(e, nb),
            proj_w: ws(e * e, nb), proj_b: vec_of(e, nb),
        }
    }

    fn build_mlp_w(e: usize, hidden: usize, nb: &mut dyn FnMut() -> f32) -> EvaMlpWeights {
        EvaMlpWeights {
            fc1_g_w: ws(e * hidden, nb), fc1_g_b: vec_of(hidden, nb),
            fc1_x_w: ws(e * hidden, nb), fc1_x_b: vec_of(hidden, nb),
            norm: ln_w(hidden),
            fc2_w: ws(hidden * e, nb), fc2_b: vec_of(e, nb),
        }
    }

    fn build_weights(cfg: &EvaConfig) -> EvaWeights {
        let mut nb = rng_seed(2026);
        let e = cfg.embed_dim;
        let head_dim = cfg.head_dim();
        let n_patches = cfg.num_patches();
        let hidden = cfg.hidden_mlp_dim();
        let n_tokens = n_patches + 1;
        let blocks: Vec<EvaBlockWeights> = (0..cfg.depth).map(|_| {
            EvaBlockWeights {
                norm1: ln_w(e),
                attn: build_attn_w(e, &mut nb),
                norm2: ln_w(e),
                mlp: build_mlp_w(e, hidden, &mut nb),
            }
        }).collect();

        // Build rot_pos_embed: (n_patches, 2*head_dim) with interleaved
        // (cos[2k]==cos[2k+1], sin[2k]==sin[2k+1]) pairs. First head_dim
        // channels = sin, next head_dim = cos.
        let mut rot = Vec::with_capacity(n_patches * 2 * head_dim);
        for p in 0..n_patches {
            for k in 0..(head_dim / 2) {
                let theta = (p as f32) / 10000.0_f32.powf(2.0 * k as f32 / head_dim as f32);
                let s = theta.sin(); let _c = theta.cos();
                // sin block at index 2k, 2k+1.
                rot.push(s); // sin[2k]
                rot.push(s); // sin[2k+1]
            }
            for k in 0..(head_dim / 2) {
                let theta = (p as f32) / 10000.0_f32.powf(2.0 * k as f32 / head_dim as f32);
                let c = theta.cos();
                rot.push(c);
                rot.push(c);
            }
        }
        EvaWeights {
            patch_conv_w: vec_of(e * 3 * cfg.patch_size * cfg.patch_size, &mut nb),
            patch_conv_b: vec_of(e, &mut nb),
            cls_token: vec_of(e, &mut nb),
            pos_embed: vec_of(n_tokens * e, &mut nb),
            rot_pos_embed: Arc::from(rot),
            blocks,
            norm: ln_w(e),
            head_w: ws(e * cfg.num_classes.unwrap(), &mut nb),
            head_b: vec_of(cfg.num_classes.unwrap(), &mut nb),
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let weights = build_weights(&cfg);
        let model = EvaModel { config: cfg.clone(), weights };
        let img = LazyTensor::from_f32(
            (0..(3 * 28 * 28)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 28, 28]), &Device::cpu(),
        );
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 5]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn forward_responds_to_input() {
        let cfg = tiny_config();
        let weights = build_weights(&cfg);
        let model = EvaModel { config: cfg, weights };
        let img_a = LazyTensor::from_f32(
            (0..(3 * 28 * 28)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 28, 28]), &Device::cpu(),
        );
        let img_b = LazyTensor::from_f32(
            (0..(3 * 28 * 28)).map(|i| (i as f32) * 0.01 + 0.3).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 28, 28]), &Device::cpu(),
        );
        let a = model.forward(&img_a).unwrap().realize_f32();
        let b = model.forward(&img_b).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "EVA-02 must respond to input changes, max_diff = {max_diff}");
    }

    #[test]
    fn presets_construct() {
        let base = EvaConfig::vit_base();
        assert_eq!(base.embed_dim, 768);
        assert_eq!(base.head_dim(), 64);
        assert_eq!(base.num_patches(), 32 * 32);
        let large = EvaConfig::vit_large();
        assert_eq!(large.head_dim(), 64);
    }

    #[test]
    fn rope_skip_cls_keeps_cls_unchanged() {
        let cfg = tiny_config();
        let weights = build_weights(&cfg);
        let head_dim = cfg.head_dim();
        let b = 1; let heads = cfg.num_heads; let n = cfg.num_patches() + 1;

        let x_data: Vec<f32> = (0..(b * heads * n * head_dim)).map(|i| (i as f32) * 0.01).collect();
        let x = LazyTensor::from_f32(
            x_data.clone(),
            Shape::from_dims(&[b, heads, n, head_dim]),
            &Device::cpu(),
        );

        let n_patches = n - 1;
        let rot = x.const_f32_like(
            Arc::clone(&weights.rot_pos_embed),
            Shape::from_dims(&[n_patches, 2 * head_dim]),
        );
        let sin_emb = rot.narrow(1_usize, 0, head_dim).unwrap();
        let cos_emb = rot.narrow(1_usize, head_dim, head_dim).unwrap();
        let out = apply_rope_skip_cls(&x, &cos_emb, &sin_emb, b, heads, n, head_dim).unwrap();
        let out_data = out.realize_f32();
        // First-token (CLS) entries should be byte-identical to input.
        for hh in 0..heads {
            for d in 0..head_dim {
                let i = ((hh * n) + 0) * head_dim + d;
                let i_x = ((hh * n) + 0) * head_dim + d;
                assert!((out_data[i] - x_data[i_x]).abs() < 1e-7,
                    "CLS-token RoPE leak at head={hh} d={d}: {} vs {}",
                    out_data[i], x_data[i_x]);
            }
        }
    }
}

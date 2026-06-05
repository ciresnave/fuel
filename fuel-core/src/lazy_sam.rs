//! Segment Anything Model (SAM) — ViT-B image encoder, lazy port.
//!
//! Phase D vision port. SAM (Meta AI 2023) provides a 3-stage
//! architecture for promptable image segmentation:
//!
//!   1. **Image encoder** — a ViT backbone that runs once per image
//!      and emits a dense image embedding (1024×1024 → 64×64×256).
//!   2. **Prompt encoder** — encodes positive/negative point and
//!      box prompts into shared-dim embeddings.
//!   3. **Mask decoder** — small transformer that fuses the image
//!      embedding with the prompt embeddings to produce a mask.
//!
//! This v1 ports **only the image encoder**, ViT-B preset. The
//! prompt encoder + mask decoder come in follow-up commits.
//!
//! # Architecture (`SamImageEncoderVit`)
//!
//!   ```text
//!   image (1, 3, 1024, 1024)                           // 0
//!     ── Conv2d(k=16, s=16)              → (1, 768, 64, 64)
//!     ── permute(0,2,3,1)                → (1, 64, 64, 768)
//!     ── + abs_pos_embed (1, 64, 64, 768)              // 1
//!     ── 12 × Block (768d, 12 heads)                    // 2-13
//!     ── permute(0,3,1,2)                → (1, 768, 64, 64)
//!     ── neck: Conv2d(768→256, k=1, nb)               → (1, 256, 64, 64)
//!              LayerNorm2d
//!              Conv2d(256→256, k=3, p=1, nb)
//!              LayerNorm2d                            // 14
//!   ```
//!
//! Each ViT block:
//!
//!   ```text
//!   x' = x + window_attn(LN1(x))
//!   x  = x' + MLP(LN2(x'))
//!   ```
//!
//! Attention has three quirks SAM ViT inherits from the original
//! design:
//!
//!   - **Windowed attention** on most layers (`window_size = 14`).
//!     Patches are partitioned into 14×14 windows; attention runs
//!     within each window. The 4 *global-attention* layers
//!     (indices 2, 5, 8, 11 for ViT-B) skip the window split and
//!     attend over the full 64×64 patch grid.
//!   - **Decomposed relative position bias** added to the attention
//!     scores. Two learned tables `rel_pos_h` and `rel_pos_w` of
//!     shape `(2·input_size − 1, head_dim)` are gathered per
//!     query/key offset and broadcast-added to the attention
//!     matrix. The lazy port uses the broadcast-add path; the
//!     eager `Add3` CPU custom op (a fused-add fast path) is
//!     replaced by the natural broadcast expression — slower per
//!     element but works on every backend.
//!   - **Fused QKV** — a single `qkv: Linear(dim → 3·dim)`
//!     projection that gets sliced into Q/K/V along the last dim
//!     after a reshape.
//!
//! The **neck** post-encoder is two 1×1 / 3×3 convolutions with
//! per-channel LayerNorm2d in between, reducing the 768-dim
//! patch embeddings to 256-dim image-feature embeddings that the
//! prompt encoder + mask decoder consume.
//!
//! # Scope (v1)
//!
//! - **Forward-only**, single image (`batch == 1`), F32 throughout.
//! - **ViT-B preset only** (12 layers, 768-dim, 12 heads). ViT-L
//!   (24 layers, 1024-dim, 16 heads) and ViT-H (32 layers,
//!   1280-dim, 16 heads) are parameter changes; trivial follow-ups.
//! - **`use_rel_pos = true`, `use_abs_pos = true`, `qkv_bias = true`** —
//!   the SAM defaults. Bias-free variants and no-rel-pos variants
//!   are config knobs that can be wired later.
//! - **Decomposed rel-pos interpolation deferred**: the eager
//!   `get_rel_pos` has a `todo!()` branch when `q_size` /
//!   `k_size` ≠ the stored table's first dimension. The ported
//!   path bails explicitly in that branch — currently only the
//!   `q_size = k_size = window_size` (=14) case for windowed
//!   layers and `q_size = k_size = img_size/patch_size` (=64)
//!   case for global layers are tested, both of which match the
//!   table dimensions exactly.
//! - **Prompt encoder + mask decoder + TinyViT image encoder** —
//!   deferred to follow-up commits. They depend on the image
//!   encoder being available first.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

// ---- Config ---------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct SamImageEncoderConfig {
    /// Side length of the input image (1024 for SAM).
    pub img_size: usize,
    /// Patch side (16 for SAM ViT — gives a 64×64 patch grid at
    /// img_size=1024).
    pub patch_size: usize,
    /// Input channels (3 for RGB).
    pub in_chans: usize,
    /// Transformer hidden dim (768 for ViT-B, 1024 for ViT-L, 1280 for ViT-H).
    pub embed_dim: usize,
    /// Number of transformer blocks (12 for ViT-B).
    pub depth: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Output channel count of the neck (256 for SAM — `PROMPT_EMBED_DIM`).
    pub out_chans: usize,
    /// QKV bias toggle. SAM uses `true`.
    pub qkv_bias: bool,
    /// Use decomposed relative-position bias in attention. SAM uses `true`.
    pub use_rel_pos: bool,
    /// Use absolute position embedding added after patch embedding. SAM uses `true`.
    pub use_abs_pos: bool,
    /// Window side for windowed-attention layers. SAM uses 14.
    pub window_size: usize,
    /// Layer indices that use full (non-windowed) attention. For ViT-B
    /// these are `[2, 5, 8, 11]`.
    pub global_attn_indexes: Vec<usize>,
}

impl SamImageEncoderConfig {
    /// SAM ViT-B preset (the smallest of the three SAM ViT sizes).
    pub fn vit_b() -> Self {
        Self {
            img_size: 1024,
            patch_size: 16,
            in_chans: 3,
            embed_dim: 768,
            depth: 12,
            num_heads: 12,
            out_chans: 256,
            qkv_bias: true,
            use_rel_pos: true,
            use_abs_pos: true,
            window_size: 14,
            global_attn_indexes: vec![2, 5, 8, 11],
        }
    }

    /// SAM ViT-L preset — 4× the parameter count of ViT-B (24 layers,
    /// 1024-dim, 16 heads). Global-attention layers at depth/4 strides.
    pub fn vit_l() -> Self {
        Self {
            img_size: 1024,
            patch_size: 16,
            in_chans: 3,
            embed_dim: 1024,
            depth: 24,
            num_heads: 16,
            out_chans: 256,
            qkv_bias: true,
            use_rel_pos: true,
            use_abs_pos: true,
            window_size: 14,
            global_attn_indexes: vec![5, 11, 17, 23],
        }
    }

    /// SAM ViT-H preset — the original SAM's largest (and most
    /// accurate) backbone (32 layers, 1280-dim, 16 heads).
    pub fn vit_h() -> Self {
        Self {
            img_size: 1024,
            patch_size: 16,
            in_chans: 3,
            embed_dim: 1280,
            depth: 32,
            num_heads: 16,
            out_chans: 256,
            qkv_bias: true,
            use_rel_pos: true,
            use_abs_pos: true,
            window_size: 14,
            global_attn_indexes: vec![7, 15, 23, 31],
        }
    }

    /// Patches per side (`img_size / patch_size`). For SAM ViT-B at
    /// `img_size = 1024` this is 64.
    pub fn patches_per_side(&self) -> usize {
        self.img_size / self.patch_size
    }

    /// Per-head dimension (`embed_dim / num_heads`).
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
}

// ---- Weights --------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SamLayerNormWeights {
    pub gain: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct SamAttentionWeights {
    /// `[embed_dim, 3 · embed_dim]` fused QKV projection.
    pub qkv: WeightStorage,
    pub qkv_bias: Arc<[f32]>,
    /// `[embed_dim, embed_dim]` output projection (always biased).
    pub proj: WeightStorage,
    pub proj_bias: Arc<[f32]>,
    /// `[(2·input_size − 1), head_dim]` rel-pos table for the H axis.
    /// `None` when `use_rel_pos = false`.
    pub rel_pos_h: Option<Arc<[f32]>>,
    /// `[(2·input_size − 1), head_dim]` rel-pos table for the W axis.
    /// `None` when `use_rel_pos = false`.
    pub rel_pos_w: Option<Arc<[f32]>>,
    /// Side of the attention input grid this rel-pos table sizes for.
    /// For windowed layers this is `window_size`; for global layers
    /// it's `patches_per_side`.
    pub input_size: usize,
}

#[derive(Debug, Clone)]
pub struct SamMlpWeights {
    /// `[embed_dim, 4 · embed_dim]`.
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    /// `[4 · embed_dim, embed_dim]`.
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct SamBlockWeights {
    pub norm1: SamLayerNormWeights,
    pub attn: SamAttentionWeights,
    pub norm2: SamLayerNormWeights,
    pub mlp: SamMlpWeights,
}

#[derive(Debug, Clone)]
pub struct SamImageEncoderWeights {
    /// Conv2d patch projection kernel `[embed_dim, in_chans, patch_size, patch_size]`.
    pub patch_embed_w: Arc<[f32]>,
    /// Patch projection bias `[embed_dim]`.
    pub patch_embed_b: Arc<[f32]>,
    /// Absolute position embedding `[1, patches_per_side, patches_per_side, embed_dim]`.
    /// `None` when `use_abs_pos = false`.
    pub pos_embed: Option<Arc<[f32]>>,
    pub blocks: Vec<SamBlockWeights>,
    /// Neck conv1: `[out_chans, embed_dim, 1, 1]` (no bias).
    pub neck_conv1_w: Arc<[f32]>,
    pub neck_ln1: SamLayerNormWeights,
    /// Neck conv2: `[out_chans, out_chans, 3, 3]` (no bias).
    pub neck_conv2_w: Arc<[f32]>,
    pub neck_ln2: SamLayerNormWeights,
}

#[derive(Debug, Clone)]
pub struct SamImageEncoderVit {
    pub config: SamImageEncoderConfig,
    pub weights: SamImageEncoderWeights,
}

// ---- Forward --------------------------------------------------------------

impl SamImageEncoderVit {
    /// Encode a single RGB image. `image_chw` is row-major
    /// `[3, img_size, img_size]` F32 pixel data (normalized by the
    /// caller — SAM divides by 255 and standardizes with ImageNet
    /// stats before calling forward). Returns the image feature
    /// map of shape `(1, out_chans, patches_per_side, patches_per_side)`.
    pub fn forward(&self, image_chw: &[f32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let img = cfg.img_size;
        let ch = cfg.in_chans;
        assert_eq!(
            image_chw.len(), ch * img * img,
            "SAM image: expected {} elements ({}×{}×{}), got {}",
            ch * img * img, ch, img, img, image_chw.len(),
        );

        // ---- Patch embedding: Conv2d(k=patch_size, s=patch_size) -----------
        let x = LazyTensor::from_f32(
            image_chw.to_vec(),
            Shape::from_dims(&[1, ch, img, img]),
            &Device::cpu(),
        );
        let pps = cfg.patches_per_side();
        let conv_w = x.const_f32_like(
            Arc::clone(&weights.patch_embed_w),
            Shape::from_dims(&[cfg.embed_dim, ch, cfg.patch_size, cfg.patch_size]),
        );
        let conv_b = x.const_f32_like(
            Arc::clone(&weights.patch_embed_b),
            Shape::from_dims(&[cfg.embed_dim]),
        );
        let patches = x.conv2d(
            &conv_w, Some(&conv_b),
            (cfg.patch_size, cfg.patch_size),  // stride
            (0, 0),                            // padding
            1,                                 // groups
        )?;
        // (1, embed_dim, pps, pps) → (1, pps, pps, embed_dim).
        let mut x = patches.permute([0, 2, 3, 1_usize])?;

        // ---- Absolute position embedding (optional) ------------------------
        if let Some(pos) = &weights.pos_embed {
            let pos_t = x.const_f32_like(
                Arc::clone(pos),
                Shape::from_dims(&[1, pps, pps, cfg.embed_dim]),
            );
            x = x.add(&pos_t)?;
        }

        // ---- 12 transformer blocks ----------------------------------------
        for blk in &weights.blocks {
            x = self.apply_block(&x, blk, pps)?;
        }

        // ---- Neck: permute → conv1 → LN2d → conv2 → LN2d -------------------
        let x = x.permute([0, 3, 1, 2_usize])?;  // (1, embed_dim, pps, pps)
        let neck1_w = x.const_f32_like(
            Arc::clone(&weights.neck_conv1_w),
            Shape::from_dims(&[cfg.out_chans, cfg.embed_dim, 1, 1]),
        );
        let x = x.conv2d(&neck1_w, None, (1, 1), (0, 0), 1)?;
        let x = layer_norm_2d(&x, &weights.neck_ln1, cfg.out_chans, 1e-6)?;
        let neck2_w = x.const_f32_like(
            Arc::clone(&weights.neck_conv2_w),
            Shape::from_dims(&[cfg.out_chans, cfg.out_chans, 3, 3]),
        );
        let x = x.conv2d(&neck2_w, None, (1, 1), (1, 1), 1)?;
        layer_norm_2d(&x, &weights.neck_ln2, cfg.out_chans, 1e-6)
    }

    fn apply_block(
        &self,
        x: &LazyTensor,
        blk: &SamBlockWeights,
        pps: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let embed_dim = cfg.embed_dim;

        // Pre-LN1 → window-partition → attention → window-unpartition.
        let normed = x.layer_norm_affine(
            Arc::clone(&blk.norm1.gain), Arc::clone(&blk.norm1.bias), 1e-6,
        )?;
        // Window-partition if windowed; else operate on the full grid.
        let window = blk.attn.input_size;
        let is_global = window == pps;
        let (input_for_attn, pad_hw) = if is_global {
            (normed.clone(), (pps, pps))
        } else {
            window_partition(&normed, window, pps, pps, embed_dim)?
        };
        let attn_out = apply_attention(
            &input_for_attn, &blk.attn, cfg.num_heads, cfg.head_dim(),
        )?;
        let attn_out = if is_global {
            attn_out
        } else {
            window_unpartition(&attn_out, window, pad_hw, (pps, pps), embed_dim)?
        };
        let x1 = x.add(&attn_out)?;

        // Pre-LN2 → MLP (Linear → GELU → Linear) → residual add.
        let normed = x1.layer_norm_affine(
            Arc::clone(&blk.norm2.gain), Arc::clone(&blk.norm2.bias), 1e-6,
        )?;
        let mlp_hidden = blk.mlp.fc1.apply_linear_with_bias(
            &normed, embed_dim, embed_dim * 4, Arc::clone(&blk.mlp.fc1_bias),
        )?.gelu();
        let mlp_out = blk.mlp.fc2.apply_linear_with_bias(
            &mlp_hidden, embed_dim * 4, embed_dim, Arc::clone(&blk.mlp.fc2_bias),
        )?;
        x1.add(&mlp_out)
    }
}

/// Per-channel LayerNorm for `(N, C, H, W)` tensors. Reduces over C
/// (axis 1), then applies a learnable per-channel gain + bias.
///
/// The eager `LayerNorm2d` does mean/var manually because the
/// affine has to broadcast against a 4-D tensor with the channel
/// axis NOT at the end. The lazy port uses the same manual
/// formulation rather than the `LazyTensor::layer_norm_affine`
/// method (which reduces over the LAST dim).
fn layer_norm_2d(
    x: &LazyTensor,
    ln: &SamLayerNormWeights,
    num_channels: usize,
    eps: f64,
) -> Result<LazyTensor> {
    // mean over dim 1 (channel), keepdim.
    let dims = x.shape();
    let dims = dims.dims();
    assert_eq!(dims.len(), 4, "layer_norm_2d: expected rank-4 input");
    let n = dims[0]; let h = dims[2]; let w = dims[3];

    let u = x.mean_dim(1_usize)?
        .reshape(Shape::from_dims(&[n, 1, h, w]))?
        .broadcast_to(Shape::from_dims(&[n, num_channels, h, w]))?;
    let xs = x.sub(&u)?;
    let s = xs.mul(&xs)?.mean_dim(1_usize)?
        .reshape(Shape::from_dims(&[n, 1, h, w]))?
        .broadcast_to(Shape::from_dims(&[n, num_channels, h, w]))?;
    let denom = s.add_scalar(eps).sqrt();
    let normalized = xs.div(&denom)?;

    let g = x
        .const_f32_like(Arc::clone(&ln.gain), Shape::from_dims(&[num_channels]))
        .reshape(Shape::from_dims(&[1, num_channels, 1, 1]))?
        .broadcast_to(Shape::from_dims(&[n, num_channels, h, w]))?;
    let b = x
        .const_f32_like(Arc::clone(&ln.bias), Shape::from_dims(&[num_channels]))
        .reshape(Shape::from_dims(&[1, num_channels, 1, 1]))?
        .broadcast_to(Shape::from_dims(&[n, num_channels, h, w]))?;
    normalized.mul(&g)?.add(&b)
}

/// SAM-style decomposed-rel-pos attention. Input shape:
/// `(1, h, w, embed_dim)`. For windowed layers `(h, w) = (window, window)`,
/// for global layers `(h, w) = (pps, pps)`. Output has the same shape.
fn apply_attention(
    x: &LazyTensor,
    w: &SamAttentionWeights,
    num_heads: usize,
    head_dim: usize,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    assert_eq!(dims.len(), 4, "SAM attn: expected rank-4 input");
    let batch = dims[0]; let h = dims[1]; let wid = dims[2]; let c = dims[3];
    let nh = num_heads;
    let hd = head_dim;
    assert_eq!(nh * hd, c, "SAM attn: num_heads * head_dim != embed_dim");

    // Flatten (h, w) into a single sequence dim for QKV projection:
    // (b, h, w, c) → (b, h*w, c).
    let x_flat = x.reshape(Shape::from_dims(&[batch, h * wid, c]))?;

    // qkv: Linear(c → 3c). Result (b, hw, 3c).
    let qkv = w.qkv.apply_linear_with_bias(
        &x_flat, c, 3 * c, Arc::clone(&w.qkv_bias),
    )?;
    // Reshape to (b, hw, 3, nh, hd) → permute to (3, b, nh, hw, hd).
    let qkv = qkv
        .reshape(Shape::from_dims(&[batch, h * wid, 3, nh, hd]))?
        .permute([2, 0, 3, 1, 4_usize])?;

    // Slice along the first dim for q/k/v.
    let q = qkv.slice(0_usize, 0, 1)?
        .reshape(Shape::from_dims(&[batch * nh, h * wid, hd]))?;
    let k = qkv.slice(0_usize, 1, 1)?
        .reshape(Shape::from_dims(&[batch * nh, h * wid, hd]))?;
    let v = qkv.slice(0_usize, 2, 1)?
        .reshape(Shape::from_dims(&[batch * nh, h * wid, hd]))?;

    // attn = (q * scale) @ k.T → (b*nh, hw, hw).
    let scale = 1.0_f64 / (hd as f64).sqrt();
    let q_scaled = q.mul_scalar(scale);
    let k_t = k.transpose()?;
    let attn = q_scaled.matmul(&k_t)?;

    // Decomposed rel-pos.
    let attn = match (&w.rel_pos_h, &w.rel_pos_w) {
        (Some(rph), Some(rpw)) => add_decomposed_rel_pos(
            &attn, &q, rph, rpw, batch * nh, h, wid, hd, w.input_size,
        )?,
        _ => attn,
    };

    // Softmax + matmul v.
    let attn = attn.softmax_last_dim()?;
    let ctx = attn.matmul(&v)?;

    // Reshape back: (b*nh, hw, hd) → (b, nh, h, w, hd) → (b, h, w, nh*hd).
    let ctx = ctx
        .reshape(Shape::from_dims(&[batch, nh, h, wid, hd]))?
        .permute([0, 2, 3, 1, 4_usize])?
        .reshape(Shape::from_dims(&[batch, h, wid, nh * hd]))?;
    // Flatten for output projection: (b, h*w, c) → proj → reshape back.
    let ctx_flat = ctx.reshape(Shape::from_dims(&[batch, h * wid, c]))?;
    let projected = w.proj.apply_linear_with_bias(
        &ctx_flat, c, c, Arc::clone(&w.proj_bias),
    )?;
    projected.reshape(Shape::from_dims(&[batch, h, wid, c]))
}

/// Add the decomposed relative-position bias to the attention scores.
/// `attn` is `(b*nh, q_h*q_w, k_h*k_w)`. `q` is the query
/// pre-matmul tensor `(b*nh, q_h*q_w, head_dim)`.
#[allow(clippy::too_many_arguments)]
fn add_decomposed_rel_pos(
    attn: &LazyTensor,
    q: &LazyTensor,
    rel_pos_h: &Arc<[f32]>,
    rel_pos_w: &Arc<[f32]>,
    b_nh: usize,
    q_h: usize,
    q_w: usize,
    head_dim: usize,
    input_size: usize,
) -> Result<LazyTensor> {
    // For SAM ViT-B all attention input grids are square AND match
    // the stored rel-pos table size; no interpolation needed.
    let max_rel_dist = 2 * input_size - 1;
    let rh = get_rel_pos(attn, q_h, q_h, rel_pos_h, max_rel_dist, head_dim)?;
    let rw = get_rel_pos(attn, q_w, q_w, rel_pos_w, max_rel_dist, head_dim)?;

    // r_q shape: (b*nh, q_h, q_w, head_dim).
    let r_q = q.reshape(Shape::from_dims(&[b_nh, q_h, q_w, head_dim]))?;
    // rel_h = einsum("bhwc,hkc->bhwk", r_q, rh)
    // rh has shape (q_h, q_h, head_dim) — transpose last two → (q_h, head_dim, q_h)
    // and broadcast to (b*nh, q_h, head_dim, q_h)? Simpler: matmul r_q @ rh.t().
    let rh_bc = rh
        .reshape(Shape::from_dims(&[1, q_h, q_h, head_dim]))?
        .broadcast_to(Shape::from_dims(&[b_nh, q_h, q_h, head_dim]))?;
    let rh_t = rh_bc.permute([0, 1, 3, 2_usize])?;  // (b*nh, q_h, head_dim, q_h)
    let rel_h = r_q.matmul(&rh_t)?;  // (b*nh, q_h, q_w, q_h)

    // rel_w = einsum("bhwc,wkc->bhwk", r_q, rw)
    // Reshape r_q via transpose to put w first: (b*nh, q_h, q_w, head_dim)
    // → (b*nh, q_w, q_h, head_dim).
    let r_q_w = r_q.permute([0, 2, 1, 3_usize])?;  // (b*nh, q_w, q_h, head_dim)
    let rw_bc = rw
        .reshape(Shape::from_dims(&[1, q_w, q_w, head_dim]))?
        .broadcast_to(Shape::from_dims(&[b_nh, q_w, q_w, head_dim]))?;
    let rw_t = rw_bc.permute([0, 1, 3, 2_usize])?;  // (b*nh, q_w, head_dim, q_w)
    let rel_w_pre = r_q_w.matmul(&rw_t)?;  // (b*nh, q_w, q_h, q_w)
    let rel_w = rel_w_pre.permute([0, 2, 1, 3_usize])?;  // (b*nh, q_h, q_w, q_w)

    // Final fused-add: attn (reshaped to b*nh, q_h, q_w, q_h, q_w) + broadcast(rel_h, rel_w)
    let attn_grid = attn.reshape(Shape::from_dims(&[b_nh, q_h, q_w, q_h, q_w]))?;
    let rel_h_bc = rel_h
        .reshape(Shape::from_dims(&[b_nh, q_h, q_w, q_h, 1]))?
        .broadcast_to(Shape::from_dims(&[b_nh, q_h, q_w, q_h, q_w]))?;
    let rel_w_bc = rel_w
        .reshape(Shape::from_dims(&[b_nh, q_h, q_w, 1, q_w]))?
        .broadcast_to(Shape::from_dims(&[b_nh, q_h, q_w, q_h, q_w]))?;
    let summed = attn_grid.add(&rel_h_bc)?.add(&rel_w_bc)?;
    summed.reshape(Shape::from_dims(&[b_nh, q_h * q_w, q_h * q_w]))
}

/// Gather `q_size × k_size` relative-position entries from the
/// `rel_pos` table. Returns shape `(q_size, k_size, head_dim)`.
fn get_rel_pos(
    anchor: &LazyTensor,
    q_size: usize,
    k_size: usize,
    rel_pos: &Arc<[f32]>,
    max_rel_dist: usize,
    head_dim: usize,
) -> Result<LazyTensor> {
    if 2 * std::cmp::max(q_size, k_size) - 1 != max_rel_dist {
        return Err(crate::Error::Msg(format!(
            "get_rel_pos: interpolation not yet supported (q_size={q_size}, \
             k_size={k_size}, max_rel_dist={max_rel_dist})",
        )).bt());
    }
    // Build the integer relative-coordinate index table host-side.
    let q_scale = f64::max(1.0, k_size as f64 / q_size as f64);
    let k_scale = f64::max(1.0, q_size as f64 / k_size as f64);
    let mut indices = vec![0_u32; q_size * k_size];
    for i in 0..q_size {
        for j in 0..k_size {
            let q_c = (i as f64) * q_scale;
            let k_c = (j as f64) * k_scale;
            let rel = q_c - k_c + (k_size as f64 - 1.0) * k_scale;
            indices[i * k_size + j] = rel as u32;
        }
    }
    let idx_t = anchor.const_u32_like(indices, Shape::from_dims(&[q_size * k_size]));
    let rel_pos_table = anchor.const_f32_like(
        Arc::clone(rel_pos),
        Shape::from_dims(&[max_rel_dist, head_dim]),
    );
    rel_pos_table
        .index_select(0_usize, &idx_t)?
        .reshape(Shape::from_dims(&[q_size, k_size, head_dim]))
}

/// Partition a `(b, h, w, c)` tensor into windows of side
/// `window`. Pads with zeros if `h`/`w` aren't divisible by
/// `window`. Returns `(windows, (padded_h, padded_w))` where
/// `windows` has shape `(num_windows·b, window, window, c)`.
fn window_partition(
    x: &LazyTensor,
    window: usize,
    h: usize,
    w: usize,
    c: usize,
) -> Result<(LazyTensor, (usize, usize))> {
    let pad_h = (window - h % window) % window;
    let pad_w = (window - w % window) % window;
    let h_p = h + pad_h;
    let w_p = w + pad_w;
    let xs = if pad_h > 0 || pad_w > 0 {
        // (b, h, w, c) — pad along H (axis 1) and W (axis 2) with zeros.
        // Use pad with mode=Constant 0.
        let padding: Vec<(usize, usize)> = vec![
            (0, 0),       // batch
            (0, pad_h),   // H
            (0, pad_w),   // W
            (0, 0),       // C
        ];
        x.pad(padding, fuel_graph::PadMode::Constant, 0.0)?
    } else {
        x.clone()
    };
    let dims = xs.shape();
    let b = dims.dims()[0];
    let windows = xs
        .reshape(Shape::from_dims(&[
            b,
            h_p / window, window,
            w_p / window, window,
            c,
        ]))?
        // (b, h_blocks, win, w_blocks, win, c) → (b, h_blocks, w_blocks, win, win, c)
        .permute([0, 1, 3, 2, 4, 5_usize])?
        .reshape(Shape::from_dims(&[
            b * (h_p / window) * (w_p / window),
            window, window, c,
        ]))?;
    Ok((windows, (h_p, w_p)))
}

/// Inverse of `window_partition`. Reassembles per-window features
/// back into a `(b, h, w, c)` tensor, trimming any zero padding.
fn window_unpartition(
    windows: &LazyTensor,
    window: usize,
    (h_p, w_p): (usize, usize),
    (h, w): (usize, usize),
    c: usize,
) -> Result<LazyTensor> {
    let nw_h = h_p / window;
    let nw_w = w_p / window;
    let total = windows.shape().dims()[0];
    let b = total / (nw_h * nw_w);
    let xs = windows
        .reshape(Shape::from_dims(&[b, nw_h, nw_w, window, window, c]))?
        .permute([0, 1, 3, 2, 4, 5_usize])?
        .reshape(Shape::from_dims(&[b, h_p, w_p, c]))?;
    // Trim the padding back to (h, w) if it was added.
    let xs = if h_p > h { xs.slice(1_usize, 0, h)? } else { xs };
    let xs = if w_p > w { xs.slice(2_usize, 0, w)? } else { xs };
    Ok(xs)
}

// ===========================================================================
// SAM Prompt Encoder
// ===========================================================================

/// Configuration for the SAM prompt encoder. The defaults match the
/// stock SAM publication: `embed_dim=256`, `image_embedding_size=(64, 64)`
/// (from the ViT image encoder), `input_image_size=(1024, 1024)`,
/// `mask_in_chans=16`.
#[derive(Debug, Clone, PartialEq)]
pub struct SamPromptEncoderConfig {
    pub embed_dim: usize,
    /// Side dimensions of the dense image embedding grid (the
    /// patch grid from the image encoder). For SAM ViT default
    /// this is `(64, 64)`.
    pub image_embedding_size: (usize, usize),
    /// Pre-resize input image side. Used to normalize point/box
    /// coordinates into `[0, 1]` before positional encoding.
    pub input_image_size: (usize, usize),
    /// Channel count of the intermediate mask-encoder stage.
    pub mask_in_chans: usize,
}

impl SamPromptEncoderConfig {
    /// Defaults matching SAM's official checkpoint.
    pub fn sam_default() -> Self {
        Self {
            embed_dim: 256,
            image_embedding_size: (64, 64),
            input_image_size: (1024, 1024),
            mask_in_chans: 16,
        }
    }
}

/// Weights for SAM's prompt encoder.
#[derive(Debug, Clone)]
pub struct SamPromptEncoderWeights {
    /// `[2, embed_dim/2]` random Gaussian projection matrix used
    /// by the positional encoder. The exact values are part of
    /// the trained checkpoint (not re-randomized at load time).
    pub positional_encoding_gaussian: Arc<[f32]>,
    /// 4 `[1, embed_dim]` per-prompt-type embeddings, in order:
    ///   `point_embeddings[0]` — background point (label = 0)
    ///   `point_embeddings[1]` — foreground point (label = 1)
    ///   `point_embeddings[2]` — box top-left corner
    ///   `point_embeddings[3]` — box bottom-right corner
    pub point_embeddings: [Arc<[f32]>; 4],
    /// `[1, embed_dim]` — added in place of a real point embedding
    /// when the caller passes a padding label (label = -1).
    pub not_a_point_embed: Arc<[f32]>,
    /// `[1, embed_dim]` — used as the dense embedding when no mask
    /// prompt is provided (broadcast across the full image grid).
    pub no_mask_embed: Arc<[f32]>,
    /// Mask downscaling stack: `[mask_in_chans/4, 1, 2, 2]` Conv2d
    /// (stride=2, no padding) + LayerNorm2d + GELU + `[mask_in_chans,
    /// mask_in_chans/4, 2, 2]` Conv2d (stride=2) + LayerNorm2d +
    /// GELU + `[embed_dim, mask_in_chans, 1, 1]` Conv2d.
    pub mask_conv1_w: Arc<[f32]>,
    pub mask_conv1_b: Arc<[f32]>,
    pub mask_ln1: SamLayerNormWeights,
    pub mask_conv2_w: Arc<[f32]>,
    pub mask_conv2_b: Arc<[f32]>,
    pub mask_ln2: SamLayerNormWeights,
    pub mask_conv3_w: Arc<[f32]>,
    pub mask_conv3_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct SamPromptEncoder {
    pub config: SamPromptEncoderConfig,
    pub weights: SamPromptEncoderWeights,
}

impl SamPromptEncoder {
    /// Compute the dense position-encoding grid for the image
    /// embedding. Returns shape `(1, embed_dim, h, w)` where
    /// `(h, w) = image_embedding_size`.
    ///
    /// `anchor` selects the graph the result lives on — constants
    /// are emitted via `anchor.const_f32_like`. Pass the image
    /// embedding tensor when composing with `SamMaskDecoder`.
    ///
    /// This is the broadcast positional encoding the mask decoder
    /// adds to image-encoder features during cross-attention.
    pub fn dense_pe(&self, anchor: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let (h, w) = cfg.image_embedding_size;
        // Build a (h, w, 2) tensor of normalized (x, y) cell-centers.
        let mut coords = Vec::with_capacity(h * w * 2);
        for yi in 0..h {
            for xi in 0..w {
                coords.push((xi as f32 + 0.5) / w as f32);
                coords.push((yi as f32 + 0.5) / h as f32);
            }
        }
        let coords_t = anchor.const_f32_like(
            Arc::<[f32]>::from(coords),
            Shape::from_dims(&[h, w, 2]),
        );
        // pe_encoding: project, scale, sin+cos cat, then transpose.
        let pe = self.pe_encoding(anchor, &coords_t)?;
        // (h, w, embed_dim) → (1, embed_dim, h, w).
        let pe_chw = pe.permute([2, 0, 1_usize])?;
        pe_chw.reshape(Shape::from_dims(&[1, cfg.embed_dim, h, w]))
    }

    /// Encode point prompts. `points` is `(N, 2)` of `(x, y)`
    /// coordinates in original image pixels; `labels` is `(N,)`
    /// with values in `{0, 1, -1}` meaning `{background,
    /// foreground, padding}`. The +0.5 cell-center shift the
    /// eager reference applies is handled internally.
    ///
    /// Returns shape `(1, N_padded, embed_dim)` where
    /// `N_padded = N + 1` if `pad=true` (a single zero-coord
    /// padding point with label=-1 is appended), else `N_padded = N`.
    /// Pass `pad=true` when only points (no boxes) are supplied —
    /// this matches the official SAM forward path.
    pub fn embed_points(
        &self, anchor: &LazyTensor, points_xy: &[f32], labels: &[f32], pad: bool,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let n = labels.len();
        assert_eq!(points_xy.len(), n * 2,
            "embed_points: {} points, expected {} (2 coords per point)",
            points_xy.len() / 2, n);

        // Apply +0.5 cell-center shift and optionally pad.
        let mut coords_owned: Vec<f32> = points_xy.iter().map(|&v| v + 0.5).collect();
        let mut labels_owned: Vec<f32> = labels.to_vec();
        if pad {
            coords_owned.push(0.0);
            coords_owned.push(0.0);
            labels_owned.push(-1.0);
        }
        let n_padded = labels_owned.len();

        // Normalize coords: x by W, y by H — same convention as
        // PositionEmbeddingRandom::forward_with_coords.
        for i in 0..n_padded {
            coords_owned[2 * i] /= cfg.input_image_size.1 as f32;
            coords_owned[2 * i + 1] /= cfg.input_image_size.0 as f32;
        }
        let coords_t = anchor.const_f32_like(
            Arc::<[f32]>::from(coords_owned),
            Shape::from_dims(&[1, n_padded, 2]),
        );
        let labels_t = anchor.const_f32_like(
            Arc::<[f32]>::from(labels_owned),
            Shape::from_dims(&[1, n_padded]),
        );

        // pe_encoding(coords) → (1, n_padded, embed_dim).
        let pos_emb = self.pe_encoding(anchor, &coords_t)?;

        // Per-label addition:
        //   label == -1 → swap in `not_a_point_embed` (replacement, not add)
        //   label ==  0 → add `point_embeddings[0]` (background)
        //   label ==  1 → add `point_embeddings[1]` (foreground)
        let labels_bc = labels_t
            .reshape(Shape::from_dims(&[1, n_padded, 1]))?
            .broadcast_to(Shape::from_dims(&[1, n_padded, cfg.embed_dim]))?;

        let not_a_point = self.broadcast_per_point_emb(
            anchor, &self.weights.not_a_point_embed, n_padded)?;
        let neg1_mask = labels_bc.eq(&labels_bc.const_f32_like(
            Arc::<[f32]>::from(vec![-1.0_f32; 1]),
            Shape::from_dims(&[1]),
        ).broadcast_to(Shape::from_dims(&[1, n_padded, cfg.embed_dim]))?)?;
        let pos_emb = neg1_mask.where_cond(&not_a_point, &pos_emb)?;

        let bg_emb = self.broadcast_per_point_emb(
            anchor, &self.weights.point_embeddings[0], n_padded)?;
        let zeros = pos_emb.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 1]),
            Shape::from_dims(&[1]),
        ).broadcast_to(Shape::from_dims(&[1, n_padded, cfg.embed_dim]))?;
        let zero_mask = labels_bc.eq(&labels_bc.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 1]),
            Shape::from_dims(&[1]),
        ).broadcast_to(Shape::from_dims(&[1, n_padded, cfg.embed_dim]))?)?;
        let label0_contrib = zero_mask.where_cond(&bg_emb, &zeros)?;
        let pos_emb = pos_emb.add(&label0_contrib)?;

        let fg_emb = self.broadcast_per_point_emb(
            anchor, &self.weights.point_embeddings[1], n_padded)?;
        let one_mask = labels_bc.eq(&labels_bc.const_f32_like(
            Arc::<[f32]>::from(vec![1.0_f32; 1]),
            Shape::from_dims(&[1]),
        ).broadcast_to(Shape::from_dims(&[1, n_padded, cfg.embed_dim]))?)?;
        let label1_contrib = one_mask.where_cond(&fg_emb, &zeros)?;
        pos_emb.add(&label1_contrib)
    }

    /// Encode box prompts. `boxes` is `(N, 4)` row-major
    /// `(x1, y1, x2, y2)` per box in original image pixels.
    /// Returns shape `(1, 2*N, embed_dim)` — two embeddings per
    /// box, one for each corner.
    pub fn embed_boxes(&self, anchor: &LazyTensor, boxes_xyxy: &[f32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        assert_eq!(boxes_xyxy.len() % 4, 0,
            "embed_boxes: input length {} not divisible by 4", boxes_xyxy.len());
        let n_boxes = boxes_xyxy.len() / 4;

        // Reshape to `(n_boxes, 2, 2)` and apply +0.5 cell-center.
        let mut corners = Vec::with_capacity(boxes_xyxy.len());
        for &v in boxes_xyxy {
            corners.push(v + 0.5);
        }
        // Normalize: x coords by W, y coords by H.
        for i in 0..n_boxes {
            corners[4 * i]     /= cfg.input_image_size.1 as f32;
            corners[4 * i + 1] /= cfg.input_image_size.0 as f32;
            corners[4 * i + 2] /= cfg.input_image_size.1 as f32;
            corners[4 * i + 3] /= cfg.input_image_size.0 as f32;
        }

        let coords_t = anchor.const_f32_like(
            Arc::<[f32]>::from(corners),
            Shape::from_dims(&[n_boxes, 2, 2]),
        );
        let pe = self.pe_encoding(anchor, &coords_t)?;  // (n_boxes, 2, embed_dim)

        // Add per-corner type embeddings:
        //   corner 0 (top-left) gets point_embeddings[2]
        //   corner 1 (bottom-right) gets point_embeddings[3]
        let tl = self.broadcast_per_point_emb(
            anchor, &self.weights.point_embeddings[2], 1)?
            .reshape(Shape::from_dims(&[1, 1, cfg.embed_dim]))?
            .broadcast_to(Shape::from_dims(&[n_boxes, 1, cfg.embed_dim]))?;
        let br = self.broadcast_per_point_emb(
            anchor, &self.weights.point_embeddings[3], 1)?
            .reshape(Shape::from_dims(&[1, 1, cfg.embed_dim]))?
            .broadcast_to(Shape::from_dims(&[n_boxes, 1, cfg.embed_dim]))?;
        let pe_tl = pe.slice(1_usize, 0, 1)?.add(&tl)?;
        let pe_br = pe.slice(1_usize, 1, 1)?.add(&br)?;
        let pe_both = pe_tl.concat(&pe_br, 1_usize)?;
        pe_both.reshape(Shape::from_dims(&[1, 2 * n_boxes, cfg.embed_dim]))
    }

    /// Encode an input mask via the 3-conv downscaling stack.
    /// `masks` is `(1, 1, H, W)` where `H = 4 * image_embedding_size.0`
    /// and `W = 4 * image_embedding_size.1` (SAM's input is 4× the
    /// embedding grid because two stride-2 convs reduce it).
    /// Returns `(1, embed_dim, image_embedding_size.0, image_embedding_size.1)`.
    pub fn embed_masks(&self, masks: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let w = &self.weights;
        let mi = cfg.mask_in_chans;
        let q = mi / 4;
        // Conv1: 1 → mi/4, k=2, s=2.
        let conv1_w = masks.const_f32_like(
            Arc::clone(&w.mask_conv1_w),
            Shape::from_dims(&[q, 1, 2, 2]),
        );
        let conv1_b = masks.const_f32_like(
            Arc::clone(&w.mask_conv1_b),
            Shape::from_dims(&[q]),
        );
        let x = masks.conv2d(&conv1_w, Some(&conv1_b), (2, 2), (0, 0), 1)?;
        let x = layer_norm_2d(&x, &w.mask_ln1, q, 1e-6)?;
        let x = x.gelu();
        // Conv2: mi/4 → mi, k=2, s=2.
        let conv2_w = masks.const_f32_like(
            Arc::clone(&w.mask_conv2_w),
            Shape::from_dims(&[mi, q, 2, 2]),
        );
        let conv2_b = masks.const_f32_like(
            Arc::clone(&w.mask_conv2_b),
            Shape::from_dims(&[mi]),
        );
        let x = x.conv2d(&conv2_w, Some(&conv2_b), (2, 2), (0, 0), 1)?;
        let x = layer_norm_2d(&x, &w.mask_ln2, mi, 1e-6)?;
        let x = x.gelu();
        // Conv3: mi → embed_dim, k=1.
        let conv3_w = masks.const_f32_like(
            Arc::clone(&w.mask_conv3_w),
            Shape::from_dims(&[cfg.embed_dim, mi, 1, 1]),
        );
        let conv3_b = masks.const_f32_like(
            Arc::clone(&w.mask_conv3_b),
            Shape::from_dims(&[cfg.embed_dim]),
        );
        x.conv2d(&conv3_w, Some(&conv3_b), (1, 1), (0, 0), 1)
    }

    /// Convenience: if no mask is supplied, return the
    /// `no_mask_embed` broadcast across the image embedding grid.
    /// Shape: `(1, embed_dim, h, w)`.
    pub fn no_mask_dense(&self, anchor: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let (h, w) = cfg.image_embedding_size;
        let no_mask = anchor.const_f32_like(
            Arc::clone(&self.weights.no_mask_embed),
            Shape::from_dims(&[1, cfg.embed_dim, 1, 1]),
        );
        no_mask.broadcast_to(Shape::from_dims(&[1, cfg.embed_dim, h, w]))
    }

    // -- Internal helpers ---------------------------------------------------

    /// Project coordinates through the Gaussian matrix and emit
    /// sin/cos features. Input shape `(..., 2)`, output shape
    /// `(..., embed_dim)`.
    fn pe_encoding(&self, anchor: &LazyTensor, coords: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        // Scale `coords` from [0, 1] to [-1, 1].
        let coords = coords.affine(2.0, -1.0);
        // Project: (..., 2) @ (2, embed_dim/2) → (..., embed_dim/2).
        let gaussian = anchor.const_f32_like(
            Arc::clone(&self.weights.positional_encoding_gaussian),
            Shape::from_dims(&[2, cfg.embed_dim / 2]),
        );
        let projected = coords.matmul(&gaussian)?;
        // Multiply by 2π then sin + cos concat along last dim.
        let scaled = projected.mul_scalar(2.0 * std::f64::consts::PI);
        let s = scaled.sin();
        let c = scaled.cos();
        let last = s.rank() - 1;
        s.concat(&c, last)
    }

    fn broadcast_per_point_emb(
        &self,
        anchor: &LazyTensor,
        emb_data: &Arc<[f32]>,
        n_points: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let e = anchor.const_f32_like(
            Arc::clone(emb_data),
            Shape::from_dims(&[1, 1, cfg.embed_dim]),
        );
        e.broadcast_to(Shape::from_dims(&[1, n_points, cfg.embed_dim]))
    }
}

// ===========================================================================
// SAM Mask Decoder + Two-Way Transformer
// ===========================================================================

/// One internal SAM-decoder attention head bank. Differs from the
/// image-encoder attention in three ways:
///   1. **No fused QKV** — separate `q_proj`, `k_proj`, `v_proj`.
///   2. **Downsample rate** — Q/K/V are projected to
///      `embedding_dim / downsample_rate` (1 for self-attn, 2 for
///      the cross-attentions); then `out_proj` lifts back to
///      `embedding_dim`.
///   3. **No rel-pos bias** — straight scaled-dot-product.
///
/// Inputs are `(b, n, c)` token sequences. Output is `(b, n, c)`.
#[derive(Debug, Clone)]
pub struct SamDecoderAttentionWeights {
    pub q_proj: WeightStorage,
    pub q_bias: Arc<[f32]>,
    pub k_proj: WeightStorage,
    pub k_bias: Arc<[f32]>,
    pub v_proj: WeightStorage,
    pub v_bias: Arc<[f32]>,
    pub out_proj: WeightStorage,
    pub out_bias: Arc<[f32]>,
    pub num_heads: usize,
    pub embedding_dim: usize,
    /// 1 for self-attention, 2 for the two cross-attention paths
    /// in `TwoWayAttentionBlock`.
    pub downsample_rate: usize,
}

fn sam_decoder_attention(
    w: &SamDecoderAttentionWeights,
    q_in: &LazyTensor,
    k_in: &LazyTensor,
    v_in: &LazyTensor,
) -> Result<LazyTensor> {
    let d = w.embedding_dim;
    let internal = d / w.downsample_rate;
    let hd = internal / w.num_heads;

    // Project to internal dim. Inputs are (b, n_*, d).
    let q = w.q_proj.apply_linear_with_bias(q_in, d, internal, Arc::clone(&w.q_bias))?;
    let k = w.k_proj.apply_linear_with_bias(k_in, d, internal, Arc::clone(&w.k_bias))?;
    let v = w.v_proj.apply_linear_with_bias(v_in, d, internal, Arc::clone(&w.v_bias))?;

    // (b, n, internal) → (b, num_heads, n, hd) via split_heads.
    let q = q.split_heads(w.num_heads, hd)?;
    let k = k.split_heads(w.num_heads, hd)?;
    let v = v.split_heads(w.num_heads, hd)?;

    // Scaled dot-product.
    let scale = 1.0_f64 / (hd as f64).sqrt();
    let k_t = k.transpose()?;
    let scores = q.matmul(&k_t)?.mul_scalar(scale);
    let attn = scores.softmax_last_dim()?;
    let ctx = attn.matmul(&v)?;

    // Merge heads back: (b, num_heads, n, hd) → (b, n, internal)
    let merged = ctx.merge_heads()?;
    // Output projection back to d.
    w.out_proj.apply_linear_with_bias(&merged, internal, d, Arc::clone(&w.out_bias))
}

/// LayerNorm weights with explicit hidden size carried alongside,
/// for compactness in the decoder's many per-block norms.
#[derive(Debug, Clone)]
pub struct SamSimpleLnWeights {
    pub gain: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct SamMlpBlockWeights {
    /// `[embedding_dim, mlp_dim]`
    pub lin1: WeightStorage,
    pub lin1_bias: Arc<[f32]>,
    /// `[mlp_dim, embedding_dim]`
    pub lin2: WeightStorage,
    pub lin2_bias: Arc<[f32]>,
    pub embedding_dim: usize,
    pub mlp_dim: usize,
    pub activation: SamMlpActivation,
}

/// Activations supported inside the SAM decoder MLPs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamMlpActivation {
    /// ReLU — used by the two-way transformer's MLP block.
    Relu,
    /// GELU — used by the image-encoder's MLP block.
    Gelu,
}

fn apply_sam_mlp(x: &LazyTensor, w: &SamMlpBlockWeights) -> Result<LazyTensor> {
    let h = w.lin1.apply_linear_with_bias(
        x, w.embedding_dim, w.mlp_dim, Arc::clone(&w.lin1_bias),
    )?;
    let h = match w.activation {
        SamMlpActivation::Relu => h.relu(),
        SamMlpActivation::Gelu => h.gelu(),
    };
    w.lin2.apply_linear_with_bias(
        &h, w.mlp_dim, w.embedding_dim, Arc::clone(&w.lin2_bias),
    )
}

/// One layer of the bidirectional cross-attention transformer.
#[derive(Debug, Clone)]
pub struct SamTwoWayAttentionBlockWeights {
    pub self_attn: SamDecoderAttentionWeights,
    pub norm1: SamSimpleLnWeights,
    pub cross_attn_token_to_image: SamDecoderAttentionWeights,
    pub norm2: SamSimpleLnWeights,
    pub mlp: SamMlpBlockWeights,
    pub norm3: SamSimpleLnWeights,
    pub norm4: SamSimpleLnWeights,
    pub cross_attn_image_to_token: SamDecoderAttentionWeights,
    /// First layer skips the query positional-encoding for
    /// self-attention (the queries ARE the position tokens that
    /// would otherwise be added). Subsequent layers don't.
    pub skip_first_layer_pe: bool,
}

fn apply_two_way_block(
    blk: &SamTwoWayAttentionBlockWeights,
    queries: &LazyTensor,
    keys: &LazyTensor,
    query_pe: &LazyTensor,
    key_pe: &LazyTensor,
) -> Result<(LazyTensor, LazyTensor)> {
    // Self-attention.
    let queries = if blk.skip_first_layer_pe {
        sam_decoder_attention(&blk.self_attn, queries, queries, queries)?
    } else {
        let q_in = queries.add(query_pe)?;
        let attn_out = sam_decoder_attention(&blk.self_attn, &q_in, &q_in, queries)?;
        queries.add(&attn_out)?
    };
    let queries = queries.layer_norm_affine(
        Arc::clone(&blk.norm1.gain), Arc::clone(&blk.norm1.bias), 1e-5,
    )?;

    // Cross-attention: tokens attending to image.
    let q_in = queries.add(query_pe)?;
    let k_in = keys.add(key_pe)?;
    let attn_out = sam_decoder_attention(&blk.cross_attn_token_to_image, &q_in, &k_in, keys)?;
    let queries = queries.add(&attn_out)?;
    let queries = queries.layer_norm_affine(
        Arc::clone(&blk.norm2.gain), Arc::clone(&blk.norm2.bias), 1e-5,
    )?;

    // MLP.
    let mlp_out = apply_sam_mlp(&queries, &blk.mlp)?;
    let queries = queries.add(&mlp_out)?;
    let queries = queries.layer_norm_affine(
        Arc::clone(&blk.norm3.gain), Arc::clone(&blk.norm3.bias), 1e-5,
    )?;

    // Cross-attention: image attending to tokens (note: keys is the
    // query side here, queries is the K/V side — the eager code
    // labels them deliberately backwards in this branch).
    let q_in = queries.add(query_pe)?;
    let k_in = keys.add(key_pe)?;
    let attn_out = sam_decoder_attention(
        &blk.cross_attn_image_to_token, &k_in, &q_in, &queries,
    )?;
    let keys = keys.add(&attn_out)?;
    let keys = keys.layer_norm_affine(
        Arc::clone(&blk.norm4.gain), Arc::clone(&blk.norm4.bias), 1e-5,
    )?;

    Ok((queries, keys))
}

#[derive(Debug, Clone)]
pub struct SamTwoWayTransformerWeights {
    pub layers: Vec<SamTwoWayAttentionBlockWeights>,
    pub final_attn_token_to_image: SamDecoderAttentionWeights,
    pub norm_final_attn: SamSimpleLnWeights,
}

/// Run SAM's two-way transformer. `image_embedding` is `(b, c, h, w)`,
/// `image_pe` is `(b, c, h, w)`, `point_embedding` is `(b, n_tokens, c)`.
/// Returns `(queries, keys)` where queries is `(b, n_tokens, c)` and
/// keys is `(b, h*w, c)`.
pub fn apply_two_way_transformer(
    w: &SamTwoWayTransformerWeights,
    image_embedding: &LazyTensor,
    image_pe: &LazyTensor,
    point_embedding: &LazyTensor,
) -> Result<(LazyTensor, LazyTensor)> {
    let ie_dims = image_embedding.shape();
    let ie_dims = ie_dims.dims();
    assert_eq!(ie_dims.len(), 4, "two-way transformer: image_embedding must be (b, c, h, w)");
    let b = ie_dims[0]; let c = ie_dims[1]; let h = ie_dims[2]; let w_dim = ie_dims[3];

    // (b, c, h, w) → (b, h*w, c).
    let ie = image_embedding
        .reshape(Shape::from_dims(&[b, c, h * w_dim]))?
        .permute([0, 2, 1_usize])?;
    let ipe = image_pe
        .reshape(Shape::from_dims(&[b, c, h * w_dim]))?
        .permute([0, 2, 1_usize])?;

    let mut queries = point_embedding.clone();
    let mut keys = ie;
    for blk in &w.layers {
        let (q, k) = apply_two_way_block(blk, &queries, &keys, point_embedding, &ipe)?;
        queries = q;
        keys = k;
    }

    // Final cross-attention + LN on queries.
    let q_in = queries.add(point_embedding)?;
    let k_in = keys.add(&ipe)?;
    let attn_out = sam_decoder_attention(
        &w.final_attn_token_to_image, &q_in, &k_in, &keys,
    )?;
    let queries = queries.add(&attn_out)?;
    let queries = queries.layer_norm_affine(
        Arc::clone(&w.norm_final_attn.gain),
        Arc::clone(&w.norm_final_attn.bias),
        1e-5,
    )?;
    Ok((queries, keys))
}

/// Multi-layer perceptron used in SAM's IoU prediction head and
/// per-mask-token hypernetwork.
#[derive(Debug, Clone)]
pub struct SamMlpMaskDecoderWeights {
    /// One `Linear` per layer. Lengths must agree:
    ///   layer 0: `input_dim → hidden_dim`
    ///   layer i in 1..n-1: `hidden_dim → hidden_dim`
    ///   layer n-1: `hidden_dim → output_dim`
    pub layers: Vec<WeightStorage>,
    pub biases: Vec<Arc<[f32]>>,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub sigmoid_output: bool,
}

fn apply_mlp_mask_decoder(
    w: &SamMlpMaskDecoderWeights,
    x: &LazyTensor,
) -> Result<LazyTensor> {
    let n = w.layers.len();
    assert_eq!(n, w.biases.len(), "SamMlpMaskDecoder: layers/biases length mismatch");
    assert!(n >= 1, "SamMlpMaskDecoder: at least 1 layer required");

    let mut h = x.clone();
    for i in 0..n {
        let in_dim = if i == 0 { w.input_dim } else { w.hidden_dim };
        let out_dim = if i + 1 == n { w.output_dim } else { w.hidden_dim };
        h = w.layers[i].apply_linear_with_bias(
            &h, in_dim, out_dim, Arc::clone(&w.biases[i]),
        )?;
        if i + 1 < n {
            h = h.relu();
        }
    }
    if w.sigmoid_output {
        Ok(h.sigmoid())
    } else {
        Ok(h)
    }
}

/// Configuration for the SAM mask decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct SamMaskDecoderConfig {
    /// `transformer_dim` from the eager code — equals `embed_dim`
    /// from the prompt encoder (256 in SAM default).
    pub transformer_dim: usize,
    /// 3 for SAM default — the model outputs 3 multi-task masks +
    /// 1 single mask = 4 total mask tokens.
    pub num_multimask_outputs: usize,
    pub iou_head_depth: usize,
    pub iou_head_hidden_dim: usize,
    /// Two-way transformer depth (2 for SAM default).
    pub transformer_depth: usize,
    /// Heads inside the two-way transformer (8 for SAM default).
    pub transformer_num_heads: usize,
    /// MLP hidden dim inside the two-way transformer (2048 for SAM default).
    pub transformer_mlp_dim: usize,
}

impl SamMaskDecoderConfig {
    /// SAM publication defaults.
    pub fn sam_default() -> Self {
        Self {
            transformer_dim: 256,
            num_multimask_outputs: 3,
            iou_head_depth: 3,
            iou_head_hidden_dim: 256,
            transformer_depth: 2,
            transformer_num_heads: 8,
            transformer_mlp_dim: 2048,
        }
    }

    /// Total number of mask tokens (multi-task + single mask).
    pub fn num_mask_tokens(&self) -> usize {
        self.num_multimask_outputs + 1
    }
}

#[derive(Debug, Clone)]
pub struct SamMaskDecoderWeights {
    /// `[1, transformer_dim]` — the IoU-prediction token (prepended
    /// to the prompt tokens before the transformer).
    pub iou_token: Arc<[f32]>,
    /// `[num_mask_tokens, transformer_dim]`.
    pub mask_tokens: Arc<[f32]>,
    pub transformer: SamTwoWayTransformerWeights,
    /// First upscaler: `ConvTranspose2d(transformer_dim, transformer_dim/4, k=2, s=2)`.
    pub upsample_conv1_w: Arc<[f32]>,
    pub upsample_conv1_b: Arc<[f32]>,
    /// LayerNorm2d between the two upscalers (channels = transformer_dim/4).
    pub upsample_ln: SamLayerNormWeights,
    /// Second upscaler: `ConvTranspose2d(transformer_dim/4, transformer_dim/8, k=2, s=2)`.
    pub upsample_conv2_w: Arc<[f32]>,
    pub upsample_conv2_b: Arc<[f32]>,
    /// Per-mask-token hypernetwork MLPs, one per mask token. Each
    /// projects `transformer_dim → transformer_dim/8` via a
    /// 3-layer MLP.
    pub hypernetwork_mlps: Vec<SamMlpMaskDecoderWeights>,
    /// IoU-prediction head (MLP that consumes the IoU-token output
    /// from the transformer).
    pub iou_prediction_head: SamMlpMaskDecoderWeights,
}

#[derive(Debug, Clone)]
pub struct SamMaskDecoder {
    pub config: SamMaskDecoderConfig,
    pub weights: SamMaskDecoderWeights,
}

impl SamMaskDecoder {
    /// Predict masks and IoU scores from image embeddings and prompts.
    ///
    /// Inputs (all `LazyTensor` on the same graph):
    ///   - `image_embeddings`: `(b, transformer_dim, h, w)` — from
    ///     the image encoder. `b` is typically 1.
    ///   - `image_pe`: `(b, transformer_dim, h, w)` — dense
    ///     positional encoding for the image grid (from the
    ///     prompt encoder's `dense_pe()`).
    ///   - `sparse_prompt_embeddings`: `(b, n_prompts, transformer_dim)`
    ///     — sparse prompt embeddings (points + boxes from the
    ///     prompt encoder).
    ///   - `dense_prompt_embeddings`: `(b, transformer_dim, h, w)`
    ///     — dense mask prompt (or `no_mask_dense()` if absent).
    ///   - `multimask_output`: when true, returns the 3 multi-task
    ///     masks; when false, returns just the single mask token.
    ///
    /// Output:
    ///   - `masks`: `(b, num_returned, h_out, w_out)` where
    ///     `(h_out, w_out) = (4·h, 4·w)` after the 2× upscalers.
    ///   - `iou_pred`: `(b, num_returned)` quality scores.
    pub fn forward(
        &self,
        image_embeddings: &LazyTensor,
        image_pe: &LazyTensor,
        sparse_prompt_embeddings: &LazyTensor,
        dense_prompt_embeddings: &LazyTensor,
        multimask_output: bool,
    ) -> Result<(LazyTensor, LazyTensor)> {
        let cfg = &self.config;
        let w = &self.weights;
        let nmt = cfg.num_mask_tokens();
        let td = cfg.transformer_dim;

        // Output tokens = [iou_token; mask_tokens] of shape (1 + nmt, td).
        let iou_t = sparse_prompt_embeddings.const_f32_like(
            Arc::clone(&w.iou_token), Shape::from_dims(&[1, td]),
        );
        let mask_t = sparse_prompt_embeddings.const_f32_like(
            Arc::clone(&w.mask_tokens), Shape::from_dims(&[nmt, td]),
        );
        let output_tokens = iou_t.concat(&mask_t, 0_usize)?;
        let sp_dims = sparse_prompt_embeddings.shape();
        let sp_dims = sp_dims.dims();
        let b = sp_dims[0];
        let output_tokens = output_tokens
            .reshape(Shape::from_dims(&[1, 1 + nmt, td]))?
            .broadcast_to(Shape::from_dims(&[b, 1 + nmt, td]))?;
        let tokens = output_tokens.concat(sparse_prompt_embeddings, 1_usize)?;

        // Expand image_embeddings + image_pe to match the token-batch
        // dimension (b — though typically 1 for SAM).
        let n_replica = tokens.shape().dims()[0];
        let src = if n_replica == b {
            image_embeddings.clone()
        } else {
            image_embeddings.repeat_interleave(0_usize, n_replica / b)?
        };
        let pos_src = if n_replica == b {
            image_pe.clone()
        } else {
            image_pe.repeat_interleave(0_usize, n_replica / b)?
        };
        // Fuse the dense prompt embeddings into the image-features
        // (broadcast-add — dense_prompt_embeddings is (b, td, h, w)).
        let src = src.add(dense_prompt_embeddings)?;
        let src_dims = src.shape();
        let src_dims = src_dims.dims();
        let (b_, c, h, w_dim) = (src_dims[0], src_dims[1], src_dims[2], src_dims[3]);

        // Run the two-way transformer.
        let (hs, src) = apply_two_way_transformer(&w.transformer, &src, &pos_src, &tokens)?;

        // Take the IoU token and each mask token from the queries.
        // hs shape: (b, 1 + nmt + n_prompts, td)
        let iou_token_out = hs.slice(1_usize, 0, 1)?
            .reshape(Shape::from_dims(&[b_, td]))?;
        let mask_tokens_out = hs.slice(1_usize, 1, nmt)?;  // (b, nmt, td)

        // Upscale the (now-transformed) keys back to (b, td, h, w) then
        // do the 2× ConvTranspose2d stack with LayerNorm2d in between.
        let src_grid = src
            .permute([0, 2, 1_usize])?
            .reshape(Shape::from_dims(&[b_, c, h, w_dim]))?;
        let ct1_w = src_grid.const_f32_like(
            Arc::clone(&w.upsample_conv1_w),
            Shape::from_dims(&[td, td / 4, 2, 2]),
        );
        let ct1_b = src_grid.const_f32_like(
            Arc::clone(&w.upsample_conv1_b),
            Shape::from_dims(&[td / 4]),
        );
        let up1 = src_grid.conv_transpose2d(
            &ct1_w, (2, 2), (0, 0), (0, 0), (1, 1), 1,
        )?;
        // Add bias (broadcast across spatial dims).
        let up1 = up1.broadcast_add(
            &ct1_b.reshape(Shape::from_dims(&[1, td / 4, 1, 1]))?,
        )?;
        let up1 = layer_norm_2d(&up1, &w.upsample_ln, td / 4, 1e-6)?;
        let up1 = up1.gelu();
        let ct2_w = src_grid.const_f32_like(
            Arc::clone(&w.upsample_conv2_w),
            Shape::from_dims(&[td / 4, td / 8, 2, 2]),
        );
        let ct2_b = src_grid.const_f32_like(
            Arc::clone(&w.upsample_conv2_b),
            Shape::from_dims(&[td / 8]),
        );
        let upscaled = up1.conv_transpose2d(
            &ct2_w, (2, 2), (0, 0), (0, 0), (1, 1), 1,
        )?;
        let upscaled = upscaled.broadcast_add(
            &ct2_b.reshape(Shape::from_dims(&[1, td / 8, 1, 1]))?,
        )?;
        let upscaled = upscaled.gelu();

        // Run each mask-token's hypernetwork MLP. Stack to
        // (b, nmt, td/8). Multiplying by the upscaled feature map
        // (flattened to (b, td/8, H·W) and reshaped back) yields
        // the predicted masks (b, nmt, H, W).
        let mut hyper_outs: Vec<LazyTensor> = Vec::with_capacity(nmt);
        for (i, mlp) in w.hypernetwork_mlps.iter().enumerate() {
            let mt_i = mask_tokens_out.slice(1_usize, i, 1)?
                .reshape(Shape::from_dims(&[b_, td]))?;
            let h_i = apply_mlp_mask_decoder(mlp, &mt_i)?;  // (b, td/8)
            hyper_outs.push(h_i.reshape(Shape::from_dims(&[b_, 1, td / 8]))?);
        }
        let mut hyper_in = hyper_outs[0].clone();
        for h in &hyper_outs[1..] {
            hyper_in = hyper_in.concat(h, 1_usize)?;
        }
        // hyper_in: (b, nmt, td/8). upscaled: (b, td/8, H, W).
        let up_dims = upscaled.shape();
        let up_dims = up_dims.dims();
        let (h_out, w_out) = (up_dims[2], up_dims[3]);
        let upscaled_flat = upscaled.reshape(
            Shape::from_dims(&[b_, td / 8, h_out * w_out]),
        )?;
        let masks_flat = hyper_in.matmul(&upscaled_flat)?;
        let masks = masks_flat.reshape(
            Shape::from_dims(&[b_, nmt, h_out, w_out]),
        )?;

        // IoU prediction head.
        let iou_pred = apply_mlp_mask_decoder(&w.iou_prediction_head, &iou_token_out)?;

        // Optionally slice to return either the top 3 multi-task masks
        // or just the single mask.
        if multimask_output {
            let masks = masks.slice(1_usize, 1, nmt - 1)?;
            let iou_pred = iou_pred.slice(1_usize, 1, nmt - 1)?;
            Ok((masks, iou_pred))
        } else {
            let masks = masks.slice(1_usize, 0, 1)?;
            let iou_pred = iou_pred.slice(1_usize, 0, 1)?;
            Ok((masks, iou_pred))
        }
    }
}

// ===========================================================================
// SAM Model — end-to-end composition
// ===========================================================================

/// Composed configuration for a full SAM model: image encoder + prompt
/// encoder + mask decoder, bundled so a single preset constructor (e.g.
/// `vit_b()`) yields a mutually-consistent triple.
#[derive(Debug, Clone)]
pub struct SamModelConfig {
    pub image_encoder: SamImageEncoderConfig,
    pub prompt_encoder: SamPromptEncoderConfig,
    pub mask_decoder: SamMaskDecoderConfig,
}

impl SamModelConfig {
    fn derive_from_image_encoder(image_encoder: SamImageEncoderConfig) -> Self {
        let pps = image_encoder.patches_per_side();
        let img = image_encoder.img_size;
        let oc = image_encoder.out_chans;
        let prompt_encoder = SamPromptEncoderConfig {
            embed_dim: oc,
            image_embedding_size: (pps, pps),
            input_image_size: (img, img),
            mask_in_chans: 16,
        };
        let mask_decoder = SamMaskDecoderConfig {
            transformer_dim: oc,
            ..SamMaskDecoderConfig::sam_default()
        };
        Self { image_encoder, prompt_encoder, mask_decoder }
    }

    /// SAM ViT-B preset (img_size=1024, depth=12, embed_dim=768).
    pub fn vit_b() -> Self {
        Self::derive_from_image_encoder(SamImageEncoderConfig::vit_b())
    }
    /// SAM ViT-L preset (img_size=1024, depth=24, embed_dim=1024).
    pub fn vit_l() -> Self {
        Self::derive_from_image_encoder(SamImageEncoderConfig::vit_l())
    }
    /// SAM ViT-H preset (img_size=1024, depth=32, embed_dim=1280).
    pub fn vit_h() -> Self {
        Self::derive_from_image_encoder(SamImageEncoderConfig::vit_h())
    }
}

/// End-to-end SAM model — wraps the three SAM sub-modules and exposes a
/// composed `forward` that takes raw image pixels + point prompts and
/// returns segmentation masks + IoU scores.
#[derive(Debug)]
pub struct SamModel {
    pub config: SamModelConfig,
    pub image_encoder: SamImageEncoderVit,
    pub prompt_encoder: SamPromptEncoder,
    pub mask_decoder: SamMaskDecoder,
}

impl SamModel {
    /// ImageNet RGB mean used by SAM preprocessing (Meta reference).
    pub const PIXEL_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
    /// ImageNet RGB std used by SAM preprocessing (Meta reference).
    pub const PIXEL_STD: [f32; 3] = [58.395, 57.12, 57.375];

    pub fn new(
        config: SamModelConfig,
        image_encoder_weights: SamImageEncoderWeights,
        prompt_encoder_weights: SamPromptEncoderWeights,
        mask_decoder_weights: SamMaskDecoderWeights,
    ) -> Self {
        let image_encoder = SamImageEncoderVit {
            config: config.image_encoder.clone(),
            weights: image_encoder_weights,
        };
        let prompt_encoder = SamPromptEncoder {
            config: config.prompt_encoder.clone(),
            weights: prompt_encoder_weights,
        };
        let mask_decoder = SamMaskDecoder {
            config: config.mask_decoder.clone(),
            weights: mask_decoder_weights,
        };
        Self { config, image_encoder, prompt_encoder, mask_decoder }
    }

    /// Host-side preprocess: subtract ImageNet RGB mean, divide by std,
    /// then zero-pad to `image_size × image_size`. `image_chw` is
    /// row-major `(3, h, w)` raw pixel f32 (typically `0..=255`).
    /// Returns the `(3, image_size, image_size)` f32 buffer the image
    /// encoder consumes.
    pub fn preprocess(&self, image_chw: &[f32], h: usize, w: usize) -> Result<Vec<f32>> {
        let s = self.config.image_encoder.img_size;
        if h > s || w > s {
            return Err(crate::Error::Msg(format!(
                "SAM preprocess: image ({w}x{h}) exceeds max side {s}",
            )).bt());
        }
        if image_chw.len() != 3 * h * w {
            return Err(crate::Error::Msg(format!(
                "SAM preprocess: expected {} f32s (3x{}x{}), got {}",
                3 * h * w, h, w, image_chw.len(),
            )).bt());
        }
        let mut out = vec![0.0_f32; 3 * s * s];
        for c in 0..3 {
            let mean = Self::PIXEL_MEAN[c];
            let std = Self::PIXEL_STD[c];
            let src_plane = &image_chw[c * h * w..(c + 1) * h * w];
            let dst_plane = &mut out[c * s * s..(c + 1) * s * s];
            for y in 0..h {
                let src_row = &src_plane[y * w..(y + 1) * w];
                let dst_row = &mut dst_plane[y * s..y * s + w];
                for (di, &px) in dst_row.iter_mut().zip(src_row.iter()) {
                    *di = (px - mean) / std;
                }
            }
        }
        Ok(out)
    }

    /// Encode the raw image into the dense feature map. Returns shape
    /// `(1, out_chans, patches_per_side, patches_per_side)`.
    pub fn embeddings(&self, image_chw: &[f32], h: usize, w: usize) -> Result<LazyTensor> {
        let padded = self.preprocess(image_chw, h, w)?;
        self.image_encoder.forward(&padded)
    }

    /// Prompt-encode + mask-decode + upsample + crop, starting from a
    /// precomputed image embedding.
    ///
    /// Inputs:
    ///   - `img_embeddings`: from `embeddings()`.
    ///   - `orig_h`, `orig_w`: pre-padding image size — the mask is
    ///     cropped to this extent after upsampling.
    ///   - `points_xy`: row-major `(N, 2)` pixel coordinates relative to
    ///     the original (pre-padding) image. The +0.5 cell-center shift
    ///     and the not-a-point padding marker are handled internally.
    ///   - `point_labels`: `(N,)` with values in `{0, 1}` (background /
    ///     foreground); the `-1` padding marker is appended internally.
    ///   - `multimask_output`: when true, returns the 3 multi-task masks;
    ///     when false, the single mask.
    ///
    /// Returns `(masks, iou_pred)`:
    ///   - `masks`: `(num_returned, orig_h, orig_w)` — batch dim squeezed.
    ///   - `iou_pred`: `(1, num_returned)` quality scores.
    pub fn forward_for_embeddings(
        &self,
        img_embeddings: &LazyTensor,
        orig_h: usize, orig_w: usize,
        points_xy: &[f32], point_labels: &[f32],
        multimask_output: bool,
    ) -> Result<(LazyTensor, LazyTensor)> {
        if point_labels.is_empty() {
            return Err(crate::Error::Msg(
                "SAM forward: no prompts supplied (point_labels is empty)".into(),
            ).bt());
        }
        let img_size = self.config.image_encoder.img_size;
        if orig_h == 0 || orig_w == 0 || orig_h > img_size || orig_w > img_size {
            return Err(crate::Error::Msg(format!(
                "SAM forward: orig ({orig_w}x{orig_h}) must be within (0, {img_size}]",
            )).bt());
        }
        let dense_pe = self.prompt_encoder.dense_pe(img_embeddings)?;
        let sparse = self.prompt_encoder.embed_points(
            img_embeddings, points_xy, point_labels, true,
        )?;
        let dense = self.prompt_encoder.no_mask_dense(img_embeddings)?;
        let (low_res, iou) = self.mask_decoder.forward(
            img_embeddings, &dense_pe, &sparse, &dense, multimask_output,
        )?;
        let pps = self.config.image_encoder.patches_per_side();
        let decoder_side = 4 * pps;
        if img_size % decoder_side != 0 {
            return Err(crate::Error::Msg(format!(
                "SAM upsample: img_size={img_size} not divisible by decoder spatial side {decoder_side}",
            )).bt());
        }
        let scale = img_size / decoder_side;
        let upsampled = low_res.upsample_nearest2d(scale)?;
        let cropped = upsampled
            .narrow(2_usize, 0, orig_h)?
            .narrow(3_usize, 0, orig_w)?;
        let masks = cropped.squeeze(0_usize)?;
        Ok((masks, iou))
    }

    /// End-to-end composed forward: preprocess → image encode → prompt
    /// encode → mask decode → upsample → crop.
    pub fn forward(
        &self,
        image_chw: &[f32], orig_h: usize, orig_w: usize,
        points_xy: &[f32], point_labels: &[f32],
        multimask_output: bool,
    ) -> Result<(LazyTensor, LazyTensor)> {
        let img_embeddings = self.embeddings(image_chw, orig_h, orig_w)?;
        self.forward_for_embeddings(
            &img_embeddings, orig_h, orig_w, points_xy, point_labels, multimask_output,
        )
    }
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

    fn tiny_cfg() -> SamImageEncoderConfig {
        // Tiny config — small enough to test forward shape & finiteness
        // without burning seconds per assertion.
        SamImageEncoderConfig {
            img_size: 32,     // 32 / 4 = 8 patches per side
            patch_size: 4,
            in_chans: 3,
            embed_dim: 16,    // 2 heads × 8 head_dim
            depth: 2,
            num_heads: 2,
            out_chans: 8,
            qkv_bias: true,
            use_rel_pos: true,
            use_abs_pos: true,
            window_size: 4,   // 4×4 window over 8×8 grid (2×2 windows)
            global_attn_indexes: vec![1],  // layer 1 is global; layer 0 is windowed
        }
    }

    fn tiny_weights(cfg: &SamImageEncoderConfig) -> SamImageEncoderWeights {
        let mut next = rng(12345);
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let e = cfg.embed_dim;
        let pps = cfg.patches_per_side();
        let blocks: Vec<SamBlockWeights> = (0..cfg.depth).map(|i| {
            let window = if cfg.global_attn_indexes.contains(&i) {
                pps  // global attention
            } else {
                cfg.window_size
            };
            let rel_dist = 2 * window - 1;
            SamBlockWeights {
                norm1: SamLayerNormWeights {
                    gain: Arc::from(vec![1.0_f32; e]),
                    bias: Arc::from(vec![0.0_f32; e]),
                },
                attn: SamAttentionWeights {
                    qkv: WeightStorage::F32(vec_of(e * 3 * e)),
                    qkv_bias: vec_of(3 * e),
                    proj: WeightStorage::F32(vec_of(e * e)),
                    proj_bias: vec_of(e),
                    rel_pos_h: Some(vec_of(rel_dist * cfg.head_dim())),
                    rel_pos_w: Some(vec_of(rel_dist * cfg.head_dim())),
                    input_size: window,
                },
                norm2: SamLayerNormWeights {
                    gain: Arc::from(vec![1.0_f32; e]),
                    bias: Arc::from(vec![0.0_f32; e]),
                },
                mlp: SamMlpWeights {
                    fc1: WeightStorage::F32(vec_of(e * e * 4)),
                    fc1_bias: vec_of(e * 4),
                    fc2: WeightStorage::F32(vec_of(e * 4 * e)),
                    fc2_bias: vec_of(e),
                },
            }
        }).collect();

        SamImageEncoderWeights {
            patch_embed_w: vec_of(e * cfg.in_chans * cfg.patch_size * cfg.patch_size),
            patch_embed_b: vec_of(e),
            pos_embed: Some(vec_of(pps * pps * e)),
            blocks,
            neck_conv1_w: vec_of(cfg.out_chans * e * 1 * 1),
            neck_ln1: SamLayerNormWeights {
                gain: Arc::from(vec![1.0_f32; cfg.out_chans]),
                bias: Arc::from(vec![0.0_f32; cfg.out_chans]),
            },
            neck_conv2_w: vec_of(cfg.out_chans * cfg.out_chans * 3 * 3),
            neck_ln2: SamLayerNormWeights {
                gain: Arc::from(vec![1.0_f32; cfg.out_chans]),
                bias: Arc::from(vec![0.0_f32; cfg.out_chans]),
            },
        }
    }

    #[test]
    fn forward_shape_and_finite_tiny() {
        let cfg = tiny_cfg();
        let weights = tiny_weights(&cfg);
        let encoder = SamImageEncoderVit { config: cfg.clone(), weights };
        let img: Vec<f32> = (0..cfg.in_chans * cfg.img_size * cfg.img_size)
            .map(|i| ((i as f32) * 0.001) - 0.05).collect();
        let out = encoder.forward(&img).unwrap();
        let pps = cfg.patches_per_side();
        assert_eq!(out.shape().dims(), &[1, cfg.out_chans, pps, pps]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite SAM encoder output: {v}");
        }
    }

    #[test]
    fn vit_b_preset_has_correct_parameters() {
        let cfg = SamImageEncoderConfig::vit_b();
        assert_eq!(cfg.img_size, 1024);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.embed_dim, 768);
        assert_eq!(cfg.depth, 12);
        assert_eq!(cfg.num_heads, 12);
        assert_eq!(cfg.out_chans, 256);
        assert_eq!(cfg.window_size, 14);
        assert_eq!(cfg.global_attn_indexes, vec![2, 5, 8, 11]);
        assert_eq!(cfg.patches_per_side(), 64);
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn vit_l_preset_has_correct_parameters() {
        let cfg = SamImageEncoderConfig::vit_l();
        assert_eq!(cfg.embed_dim, 1024);
        assert_eq!(cfg.depth, 24);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.global_attn_indexes, vec![5, 11, 17, 23]);
        assert_eq!(cfg.head_dim(), 64);  // 1024 / 16
        assert_eq!(cfg.patches_per_side(), 64);  // image+patch size unchanged
    }

    #[test]
    fn vit_h_preset_has_correct_parameters() {
        let cfg = SamImageEncoderConfig::vit_h();
        assert_eq!(cfg.embed_dim, 1280);
        assert_eq!(cfg.depth, 32);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.global_attn_indexes, vec![7, 15, 23, 31]);
        assert_eq!(cfg.head_dim(), 80);  // 1280 / 16
        assert_eq!(cfg.patches_per_side(), 64);
    }

    #[test]
    fn layer_norm_2d_is_per_channel() {
        // Constant-per-pixel input → variance is zero → output should
        // be the bias (since gain·0 + bias = bias).
        let n = 1; let c = 4; let h = 3; let w = 5;
        let data: Vec<f32> = (0..n * c * h * w).map(|i| {
            // Set each channel to a different constant.
            ((i / (h * w)) % c) as f32
        }).collect();
        let x = LazyTensor::from_f32(
            data, Shape::from_dims(&[n, c, h, w]), &Device::cpu(),
        );
        let ln = SamLayerNormWeights {
            gain: Arc::from(vec![2.0_f32; c]),
            bias: Arc::from(vec![1.0_f32; c]),
        };
        let out = layer_norm_2d(&x, &ln, c, 1e-6).unwrap().realize_f32();
        // For each pixel: mean across channels = (0+1+2+3)/4 = 1.5,
        // values are 0,1,2,3 per channel → centered = -1.5,-0.5,0.5,1.5
        // → variance = (1.5² + 0.5² + 0.5² + 1.5²)/4 = 1.25
        // → normalized ≈ centered / sqrt(1.25) ≈ -1.3416, -0.4472, 0.4472, 1.3416
        // → gain·normalized + bias = 2·... + 1
        // → -1.6833, 0.1056, 1.8944, 3.6833
        let expected = [-1.6833_f32, 0.1056, 1.8944, 3.6833];
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ci * h * w + hi * w + wi;
                    assert!(
                        (out[idx] - expected[ci]).abs() < 1e-2,
                        "channel {ci}: got {} expected {}", out[idx], expected[ci],
                    );
                }
            }
        }
    }

    fn tiny_prompt_cfg() -> SamPromptEncoderConfig {
        SamPromptEncoderConfig {
            embed_dim: 8,                       // 2 × num_pos_feats (4)
            image_embedding_size: (4, 4),
            input_image_size: (64, 64),
            mask_in_chans: 16,
        }
    }

    fn tiny_prompt_weights(cfg: &SamPromptEncoderConfig) -> SamPromptEncoderWeights {
        let mut next = rng(98765);
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let mi = cfg.mask_in_chans;
        let q = mi / 4;
        SamPromptEncoderWeights {
            positional_encoding_gaussian: vec_of(2 * (cfg.embed_dim / 2)),
            point_embeddings: [
                vec_of(cfg.embed_dim),
                vec_of(cfg.embed_dim),
                vec_of(cfg.embed_dim),
                vec_of(cfg.embed_dim),
            ],
            not_a_point_embed: vec_of(cfg.embed_dim),
            no_mask_embed: vec_of(cfg.embed_dim),
            mask_conv1_w: vec_of(q * 1 * 2 * 2),
            mask_conv1_b: vec_of(q),
            mask_ln1: SamLayerNormWeights {
                gain: Arc::from(vec![1.0_f32; q]),
                bias: Arc::from(vec![0.0_f32; q]),
            },
            mask_conv2_w: vec_of(mi * q * 2 * 2),
            mask_conv2_b: vec_of(mi),
            mask_ln2: SamLayerNormWeights {
                gain: Arc::from(vec![1.0_f32; mi]),
                bias: Arc::from(vec![0.0_f32; mi]),
            },
            mask_conv3_w: vec_of(cfg.embed_dim * mi * 1 * 1),
            mask_conv3_b: vec_of(cfg.embed_dim),
        }
    }

    fn dummy_anchor() -> LazyTensor {
        LazyTensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu())
    }

    #[test]
    fn dense_pe_shape_and_finite() {
        let cfg = tiny_prompt_cfg();
        let weights = tiny_prompt_weights(&cfg);
        let enc = SamPromptEncoder { config: cfg.clone(), weights };
        let anchor = dummy_anchor();
        let pe = enc.dense_pe(&anchor).unwrap();
        assert_eq!(pe.shape().dims(), &[1, cfg.embed_dim, 4, 4]);
        for &v in &pe.realize_f32() {
            assert!(v.is_finite(), "non-finite dense pe element: {v}");
        }
    }

    #[test]
    fn embed_points_no_pad_shape() {
        let cfg = tiny_prompt_cfg();
        let weights = tiny_prompt_weights(&cfg);
        let enc = SamPromptEncoder { config: cfg.clone(), weights };
        let anchor = dummy_anchor();
        // 3 points: 2 foreground (label 1) + 1 background (label 0).
        let points = vec![10.0_f32, 20.0, 30.0, 30.0, 50.0, 5.0];
        let labels = vec![1.0_f32, 1.0, 0.0];
        let out = enc.embed_points(&anchor, &points, &labels, /* pad */ false).unwrap();
        assert_eq!(out.shape().dims(), &[1, 3, cfg.embed_dim]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite point embedding: {v}");
        }
    }

    #[test]
    fn embed_points_with_pad_adds_padding_slot() {
        let cfg = tiny_prompt_cfg();
        let weights = tiny_prompt_weights(&cfg);
        let enc = SamPromptEncoder { config: cfg.clone(), weights };
        let anchor = dummy_anchor();
        let points = vec![10.0_f32, 20.0, 30.0, 30.0];
        let labels = vec![1.0_f32, 0.0];
        let out = enc.embed_points(&anchor, &points, &labels, /* pad */ true).unwrap();
        assert_eq!(out.shape().dims(), &[1, 3, cfg.embed_dim]);  // +1 padding row
    }

    #[test]
    fn embed_boxes_shape() {
        let cfg = tiny_prompt_cfg();
        let weights = tiny_prompt_weights(&cfg);
        let enc = SamPromptEncoder { config: cfg.clone(), weights };
        let anchor = dummy_anchor();
        // 2 boxes (4 corners total).
        let boxes = vec![
            5.0_f32, 10.0, 30.0, 40.0,
            15.0,    20.0, 50.0, 55.0,
        ];
        let out = enc.embed_boxes(&anchor, &boxes).unwrap();
        assert_eq!(out.shape().dims(), &[1, 4, cfg.embed_dim]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite box embedding: {v}");
        }
    }

    #[test]
    fn embed_masks_downscales_4x() {
        let cfg = tiny_prompt_cfg();
        let weights = tiny_prompt_weights(&cfg);
        let enc = SamPromptEncoder { config: cfg.clone(), weights };
        // Input masks: 4× image_embedding_size = 16 × 16.
        let (h_in, w_in) = (4 * cfg.image_embedding_size.0, 4 * cfg.image_embedding_size.1);
        let masks_data: Vec<f32> = (0..1 * 1 * h_in * w_in)
            .map(|i| ((i as f32) * 0.001) - 0.05).collect();
        let masks = LazyTensor::from_f32(
            masks_data,
            Shape::from_dims(&[1, 1, h_in, w_in]),
            &Device::cpu(),
        );
        let out = enc.embed_masks(&masks).unwrap();
        // After two stride-2 convs the spatial dims drop by 4×.
        assert_eq!(
            out.shape().dims(),
            &[1, cfg.embed_dim, cfg.image_embedding_size.0, cfg.image_embedding_size.1],
        );
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite mask embedding: {v}");
        }
    }

    #[test]
    fn no_mask_dense_broadcasts_correctly() {
        let cfg = tiny_prompt_cfg();
        let weights = tiny_prompt_weights(&cfg);
        let enc = SamPromptEncoder { config: cfg.clone(), weights };
        let anchor = dummy_anchor();
        let out = enc.no_mask_dense(&anchor).unwrap();
        assert_eq!(
            out.shape().dims(),
            &[1, cfg.embed_dim, cfg.image_embedding_size.0, cfg.image_embedding_size.1],
        );
        // Every spatial cell should equal the per-channel no_mask_embed
        // value (the broadcast is along the spatial axes).
        let no_mask = enc.weights.no_mask_embed.clone();
        let realized = out.realize_f32();
        let (h, w) = cfg.image_embedding_size;
        for ci in 0..cfg.embed_dim {
            for hi in 0..h {
                for wi in 0..w {
                    let got = realized[ci * h * w + hi * w + wi];
                    let want = no_mask[ci];
                    assert!(
                        (got - want).abs() < 1e-6,
                        "(c={ci}, h={hi}, w={wi}): got {got} expected {want}",
                    );
                }
            }
        }
    }

    fn tiny_decoder_cfg() -> SamMaskDecoderConfig {
        SamMaskDecoderConfig {
            transformer_dim: 8,         // 4 heads × 2 hd at downsample=1, or 4 × 1 at ds=2
            num_multimask_outputs: 3,
            iou_head_depth: 2,
            iou_head_hidden_dim: 8,
            transformer_depth: 2,
            transformer_num_heads: 4,
            transformer_mlp_dim: 16,
        }
    }

    fn tiny_decoder_attn_weights(
        next: &mut dyn FnMut() -> f32,
        embedding_dim: usize,
        num_heads: usize,
        downsample_rate: usize,
    ) -> SamDecoderAttentionWeights {
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let internal = embedding_dim / downsample_rate;
        SamDecoderAttentionWeights {
            q_proj: WeightStorage::F32(vec_of(embedding_dim * internal)),
            q_bias: vec_of(internal),
            k_proj: WeightStorage::F32(vec_of(embedding_dim * internal)),
            k_bias: vec_of(internal),
            v_proj: WeightStorage::F32(vec_of(embedding_dim * internal)),
            v_bias: vec_of(internal),
            out_proj: WeightStorage::F32(vec_of(internal * embedding_dim)),
            out_bias: vec_of(embedding_dim),
            num_heads,
            embedding_dim,
            downsample_rate,
        }
    }

    fn tiny_mlp_weights(
        next: &mut dyn FnMut() -> f32,
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        n: usize,
    ) -> SamMlpMaskDecoderWeights {
        let mut layers = Vec::with_capacity(n);
        let mut biases = Vec::with_capacity(n);
        for i in 0..n {
            let id = if i == 0 { in_dim } else { hidden_dim };
            let od = if i + 1 == n { out_dim } else { hidden_dim };
            let weights: Vec<f32> = (0..id * od).map(|_| next()).collect();
            layers.push(WeightStorage::F32(Arc::from(weights)));
            biases.push(Arc::from((0..od).map(|_| next()).collect::<Vec<_>>()));
        }
        SamMlpMaskDecoderWeights {
            layers, biases,
            input_dim: in_dim, hidden_dim, output_dim: out_dim,
            sigmoid_output: false,
        }
    }

    fn make_arc_vec(next: &mut dyn FnMut() -> f32, n: usize) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn make_block(
        td: usize,
        nh: usize,
        mlp_dim: usize,
        skip: bool,
        next: &mut dyn FnMut() -> f32,
    ) -> SamTwoWayAttentionBlockWeights {
        let self_attn = tiny_decoder_attn_weights(next, td, nh, 1);
        let cross_attn_t2i = tiny_decoder_attn_weights(next, td, nh, 2);
        let cross_attn_i2t = tiny_decoder_attn_weights(next, td, nh, 2);
        let mlp = SamMlpBlockWeights {
            lin1: WeightStorage::F32(make_arc_vec(next, td * mlp_dim)),
            lin1_bias: make_arc_vec(next, mlp_dim),
            lin2: WeightStorage::F32(make_arc_vec(next, mlp_dim * td)),
            lin2_bias: make_arc_vec(next, td),
            embedding_dim: td,
            mlp_dim,
            activation: SamMlpActivation::Relu,
        };
        SamTwoWayAttentionBlockWeights {
            self_attn, cross_attn_token_to_image: cross_attn_t2i,
            cross_attn_image_to_token: cross_attn_i2t,
            mlp,
            norm1: SamSimpleLnWeights {
                gain: Arc::from(vec![1.0_f32; td]),
                bias: Arc::from(vec![0.0_f32; td]),
            },
            norm2: SamSimpleLnWeights {
                gain: Arc::from(vec![1.0_f32; td]),
                bias: Arc::from(vec![0.0_f32; td]),
            },
            norm3: SamSimpleLnWeights {
                gain: Arc::from(vec![1.0_f32; td]),
                bias: Arc::from(vec![0.0_f32; td]),
            },
            norm4: SamSimpleLnWeights {
                gain: Arc::from(vec![1.0_f32; td]),
                bias: Arc::from(vec![0.0_f32; td]),
            },
            skip_first_layer_pe: skip,
        }
    }

    fn tiny_decoder_weights(cfg: &SamMaskDecoderConfig) -> SamMaskDecoderWeights {
        let mut next = rng(31415);
        let td = cfg.transformer_dim;
        let nh = cfg.transformer_num_heads;
        let nmt = cfg.num_mask_tokens();
        let mlp_dim = cfg.transformer_mlp_dim;

        let layers: Vec<_> = (0..cfg.transformer_depth)
            .map(|i| make_block(td, nh, mlp_dim, i == 0, &mut next))
            .collect();
        let final_attn = tiny_decoder_attn_weights(&mut next, td, nh, 2);

        let transformer = SamTwoWayTransformerWeights {
            layers,
            final_attn_token_to_image: final_attn,
            norm_final_attn: SamSimpleLnWeights {
                gain: Arc::from(vec![1.0_f32; td]),
                bias: Arc::from(vec![0.0_f32; td]),
            },
        };

        let mut hypernetwork_mlps = Vec::with_capacity(nmt);
        for _ in 0..nmt {
            hypernetwork_mlps.push(tiny_mlp_weights(&mut next, td, td, td / 8, 3));
        }
        let iou_prediction_head = tiny_mlp_weights(
            &mut next, td, cfg.iou_head_hidden_dim, nmt, cfg.iou_head_depth,
        );

        let iou_token = make_arc_vec(&mut next, td);
        let mask_tokens = make_arc_vec(&mut next, nmt * td);
        let upsample_conv1_w = make_arc_vec(&mut next, td * (td / 4) * 2 * 2);
        let upsample_conv1_b = make_arc_vec(&mut next, td / 4);
        let upsample_conv2_w = make_arc_vec(&mut next, (td / 4) * (td / 8) * 2 * 2);
        let upsample_conv2_b = make_arc_vec(&mut next, td / 8);

        SamMaskDecoderWeights {
            iou_token, mask_tokens,
            transformer,
            upsample_conv1_w, upsample_conv1_b,
            upsample_ln: SamLayerNormWeights {
                gain: Arc::from(vec![1.0_f32; td / 4]),
                bias: Arc::from(vec![0.0_f32; td / 4]),
            },
            upsample_conv2_w, upsample_conv2_b,
            hypernetwork_mlps,
            iou_prediction_head,
        }
    }

    #[test]
    fn mask_decoder_forward_shape_singlemask() {
        let cfg = tiny_decoder_cfg();
        let weights = tiny_decoder_weights(&cfg);
        let decoder = SamMaskDecoder { config: cfg.clone(), weights };

        // (1, td, 4, 4) image embedding, (1, td, 4, 4) PE, (1, n_prompts, td) prompts.
        let td = cfg.transformer_dim;
        let h = 4; let w = 4;
        let img_data: Vec<f32> = (0..1 * td * h * w).map(|i| ((i as f32) * 0.001) - 0.05).collect();
        let img = LazyTensor::from_f32(
            img_data, Shape::from_dims(&[1, td, h, w]), &Device::cpu(),
        );
        let pe_data: Vec<f32> = (0..1 * td * h * w).map(|i| ((i as f32) * 0.0007)).collect();
        let pe = img.const_f32_like(
            Arc::<[f32]>::from(pe_data), Shape::from_dims(&[1, td, h, w]),
        );
        let n_prompts = 2;
        let sparse_data: Vec<f32> = (0..1 * n_prompts * td).map(|i| (i as f32) * 0.01).collect();
        let sparse = img.const_f32_like(
            Arc::<[f32]>::from(sparse_data),
            Shape::from_dims(&[1, n_prompts, td]),
        );
        let dense_data: Vec<f32> = (0..1 * td * h * w).map(|i| ((i as f32) * 0.0005)).collect();
        let dense = img.const_f32_like(
            Arc::<[f32]>::from(dense_data), Shape::from_dims(&[1, td, h, w]),
        );

        let (masks, iou_pred) = decoder.forward(
            &img, &pe, &sparse, &dense, /* multimask_output */ false,
        ).unwrap();
        // After 2× ConvTranspose2d, spatial dims grow 4× (2 × 2). h=4→8→16.
        assert_eq!(masks.shape().dims(), &[1, 1, 16, 16]);
        assert_eq!(iou_pred.shape().dims(), &[1, 1]);
        for &v in &masks.realize_f32() {
            assert!(v.is_finite(), "non-finite single mask: {v}");
        }
    }

    #[test]
    fn mask_decoder_forward_shape_multimask() {
        let cfg = tiny_decoder_cfg();
        let weights = tiny_decoder_weights(&cfg);
        let decoder = SamMaskDecoder { config: cfg.clone(), weights };

        let td = cfg.transformer_dim;
        let h = 4; let w = 4;
        let img_data: Vec<f32> = (0..1 * td * h * w).map(|i| ((i as f32) * 0.001) - 0.05).collect();
        let img = LazyTensor::from_f32(
            img_data, Shape::from_dims(&[1, td, h, w]), &Device::cpu(),
        );
        let pe = img.const_f32_like(
            Arc::<[f32]>::from(vec![0.001_f32; 1 * td * h * w]),
            Shape::from_dims(&[1, td, h, w]),
        );
        let n_prompts = 2;
        let sparse = img.const_f32_like(
            Arc::<[f32]>::from(vec![0.01_f32; 1 * n_prompts * td]),
            Shape::from_dims(&[1, n_prompts, td]),
        );
        let dense = img.const_f32_like(
            Arc::<[f32]>::from(vec![0.0005_f32; 1 * td * h * w]),
            Shape::from_dims(&[1, td, h, w]),
        );

        let (masks, iou_pred) = decoder.forward(
            &img, &pe, &sparse, &dense, /* multimask_output */ true,
        ).unwrap();
        // 3 multi-task masks at 16×16.
        assert_eq!(masks.shape().dims(), &[1, cfg.num_multimask_outputs, 16, 16]);
        assert_eq!(iou_pred.shape().dims(), &[1, cfg.num_multimask_outputs]);
    }

    fn consistent_model_cfg() -> SamModelConfig {
        // pps = 32/4 = 8; decoder_side = 4*pps = 32 == img_size → scale=1.
        let image_encoder = SamImageEncoderConfig {
            img_size: 32,
            patch_size: 4,
            in_chans: 3,
            embed_dim: 16,
            depth: 2,
            num_heads: 2,
            out_chans: 8,
            qkv_bias: true,
            use_rel_pos: true,
            use_abs_pos: true,
            window_size: 4,
            global_attn_indexes: vec![1],
        };
        let pps = image_encoder.patches_per_side();
        let prompt_encoder = SamPromptEncoderConfig {
            embed_dim: image_encoder.out_chans,
            image_embedding_size: (pps, pps),
            input_image_size: (image_encoder.img_size, image_encoder.img_size),
            mask_in_chans: 16,
        };
        let mask_decoder = SamMaskDecoderConfig {
            transformer_dim: image_encoder.out_chans,
            num_multimask_outputs: 3,
            iou_head_depth: 2,
            iou_head_hidden_dim: 8,
            transformer_depth: 2,
            transformer_num_heads: 4,
            transformer_mlp_dim: 16,
        };
        SamModelConfig { image_encoder, prompt_encoder, mask_decoder }
    }

    #[test]
    fn sam_model_config_vit_b_is_consistent() {
        let cfg = SamModelConfig::vit_b();
        assert_eq!(cfg.image_encoder.out_chans, cfg.prompt_encoder.embed_dim);
        assert_eq!(cfg.image_encoder.out_chans, cfg.mask_decoder.transformer_dim);
        let pps = cfg.image_encoder.patches_per_side();
        assert_eq!(cfg.prompt_encoder.image_embedding_size, (pps, pps));
        assert_eq!(
            cfg.prompt_encoder.input_image_size,
            (cfg.image_encoder.img_size, cfg.image_encoder.img_size),
        );
    }

    #[test]
    fn sam_model_config_vit_l_and_h_share_neck() {
        let cfg_l = SamModelConfig::vit_l();
        let cfg_h = SamModelConfig::vit_h();
        // All three SAM ViT presets share out_chans=256 → same prompt/decoder dims.
        assert_eq!(cfg_l.prompt_encoder.embed_dim, 256);
        assert_eq!(cfg_h.prompt_encoder.embed_dim, 256);
        assert_eq!(cfg_l.mask_decoder.transformer_dim, 256);
        assert_eq!(cfg_h.mask_decoder.transformer_dim, 256);
    }

    #[test]
    fn sam_model_preprocess_normalizes_and_pads() {
        let model_cfg = consistent_model_cfg();
        let img_size = model_cfg.image_encoder.img_size;
        let model = SamModel::new(
            model_cfg.clone(),
            tiny_weights(&model_cfg.image_encoder),
            tiny_prompt_weights(&model_cfg.prompt_encoder),
            tiny_decoder_weights(&model_cfg.mask_decoder),
        );
        // 24x28 raw image filled with the per-channel mean → preprocess
        // result should be 0 in the populated region and 0 in the pad.
        let (h, w) = (24, 28);
        let mut raw = vec![0.0_f32; 3 * h * w];
        for c in 0..3 {
            for i in 0..h * w {
                raw[c * h * w + i] = SamModel::PIXEL_MEAN[c];
            }
        }
        let out = model.preprocess(&raw, h, w).unwrap();
        assert_eq!(out.len(), 3 * img_size * img_size);
        for c in 0..3 {
            for y in 0..img_size {
                for x in 0..img_size {
                    let v = out[c * img_size * img_size + y * img_size + x];
                    if y < h && x < w {
                        assert!((v - 0.0).abs() < 1e-6,
                            "(c={c},y={y},x={x}): expected 0 (mean-subtracted) got {v}");
                    } else {
                        assert_eq!(v, 0.0, "pad at (c={c},y={y},x={x}) should be 0");
                    }
                }
            }
        }
    }

    #[test]
    fn sam_model_preprocess_rejects_oversize_image() {
        let model_cfg = consistent_model_cfg();
        let model = SamModel::new(
            model_cfg.clone(),
            tiny_weights(&model_cfg.image_encoder),
            tiny_prompt_weights(&model_cfg.prompt_encoder),
            tiny_decoder_weights(&model_cfg.mask_decoder),
        );
        // 33 > img_size=32 → bail.
        let raw = vec![0.0_f32; 3 * 33 * 33];
        assert!(model.preprocess(&raw, 33, 33).is_err());
    }

    #[test]
    fn sam_model_forward_singlemask_shape_and_finite() {
        let model_cfg = consistent_model_cfg();
        let img_size = model_cfg.image_encoder.img_size;
        let model = SamModel::new(
            model_cfg.clone(),
            tiny_weights(&model_cfg.image_encoder),
            tiny_prompt_weights(&model_cfg.prompt_encoder),
            tiny_decoder_weights(&model_cfg.mask_decoder),
        );
        let (h, w) = (24, 28);
        let raw: Vec<f32> = (0..3 * h * w).map(|i| (i as f32) * 0.1).collect();
        let points_xy = vec![5.0_f32, 7.0, 12.0, 15.0];
        let point_labels = vec![1.0_f32, 0.0];
        let (masks, iou) = model.forward(
            &raw, h, w, &points_xy, &point_labels, false,
        ).unwrap();
        // singlemask → 1 mask channel, cropped to (h, w).
        // decoder_side = 4 * pps = 32 == img_size → scale=1.
        // upsample (1, 1, 32, 32) → narrow to (1, 1, 24, 28) → squeeze → (1, 24, 28).
        assert_eq!(masks.shape().dims(), &[1, h, w]);
        assert_eq!(iou.shape().dims(), &[1, 1]);
        let _ = img_size; // silence unused if the constant changes
        for &v in &masks.realize_f32() {
            assert!(v.is_finite(), "non-finite mask: {v}");
        }
        for &v in &iou.realize_f32() {
            assert!(v.is_finite(), "non-finite iou: {v}");
        }
    }

    #[test]
    fn sam_model_forward_multimask_shape() {
        let model_cfg = consistent_model_cfg();
        let model = SamModel::new(
            model_cfg.clone(),
            tiny_weights(&model_cfg.image_encoder),
            tiny_prompt_weights(&model_cfg.prompt_encoder),
            tiny_decoder_weights(&model_cfg.mask_decoder),
        );
        let (h, w) = (20, 22);
        let raw: Vec<f32> = (0..3 * h * w).map(|i| (i as f32) * 0.05).collect();
        let points_xy = vec![3.0_f32, 4.0];
        let point_labels = vec![1.0_f32];
        let (masks, iou) = model.forward(
            &raw, h, w, &points_xy, &point_labels, true,
        ).unwrap();
        // multimask → 3 mask channels.
        assert_eq!(masks.shape().dims(), &[3, h, w]);
        assert_eq!(iou.shape().dims(), &[1, 3]);
    }

    #[test]
    fn sam_model_forward_rejects_empty_prompts() {
        let model_cfg = consistent_model_cfg();
        let model = SamModel::new(
            model_cfg.clone(),
            tiny_weights(&model_cfg.image_encoder),
            tiny_prompt_weights(&model_cfg.prompt_encoder),
            tiny_decoder_weights(&model_cfg.mask_decoder),
        );
        let (h, w) = (16, 16);
        let raw = vec![0.0_f32; 3 * h * w];
        let res = model.forward(&raw, h, w, &[], &[], false);
        assert!(res.is_err(), "empty prompts should bail");
    }

    #[test]
    fn sam_model_forward_for_embeddings_skips_image_encode() {
        let model_cfg = consistent_model_cfg();
        let model = SamModel::new(
            model_cfg.clone(),
            tiny_weights(&model_cfg.image_encoder),
            tiny_prompt_weights(&model_cfg.prompt_encoder),
            tiny_decoder_weights(&model_cfg.mask_decoder),
        );
        let (h, w) = (24, 28);
        let raw: Vec<f32> = (0..3 * h * w).map(|i| (i as f32) * 0.1).collect();
        // Precompute embeddings once, then exercise mask decode twice with
        // different prompts off the same embeddings.
        let img_embeddings = model.embeddings(&raw, h, w).unwrap();
        let (m1, _) = model.forward_for_embeddings(
            &img_embeddings, h, w, &[5.0, 7.0], &[1.0], false,
        ).unwrap();
        let (m2, _) = model.forward_for_embeddings(
            &img_embeddings, h, w, &[12.0, 15.0], &[0.0], false,
        ).unwrap();
        assert_eq!(m1.shape().dims(), &[1, h, w]);
        assert_eq!(m2.shape().dims(), &[1, h, w]);
    }

    #[test]
    fn window_partition_then_unpartition_round_trips() {
        // 4×4×8 grid, window=2 → 2×2 = 4 windows, no padding.
        let b = 1; let h = 4; let w = 4; let c = 8;
        let data: Vec<f32> = (0..b * h * w * c).map(|i| i as f32).collect();
        let x = LazyTensor::from_f32(
            data.clone(), Shape::from_dims(&[b, h, w, c]), &Device::cpu(),
        );
        let (windows, (h_p, w_p)) = window_partition(&x, 2, h, w, c).unwrap();
        assert_eq!(windows.shape().dims(), &[4, 2, 2, c]);
        assert_eq!((h_p, w_p), (4, 4));
        let restored = window_unpartition(&windows, 2, (h_p, w_p), (h, w), c).unwrap();
        assert_eq!(restored.shape().dims(), &[b, h, w, c]);
        let r = restored.realize_f32();
        for (i, (&a, &b)) in r.iter().zip(data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "round-trip elem {i} differs: {a} vs {b}",
            );
        }
    }
}

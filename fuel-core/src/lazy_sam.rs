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

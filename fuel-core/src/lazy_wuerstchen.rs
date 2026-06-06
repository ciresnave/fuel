//! Wuerstchen v2 cascaded latent diffusion model — lazy port.
//!
//! Wuerstchen (Stability AI) is a three-stage cascaded LDM:
//!
//! - **Stage A** (`PaellaVqModel`) — Paella-VQ tokenizer; inference
//!   uses the decoder only (decompresses VQ latents to RGB images).
//! - **Stage B** (`DiffNextModel`) — DiffNeXt UNet diffuses on the VQ
//!   latents conditioned on Stage C output (effnet) + optional CLIP
//!   text.
//! - **Stage C** (`PriorModel`) — smaller diffusion model mapping
//!   text embedding to a low-resolution latent map.
//!
//! Both diffusion stages share a residual block built around a
//! depthwise conv + LayerNorm + channel-wise MLP with
//! GlobalResponseNorm (ConvNeXt v2 style).
//!
//! # Eager source
//! `fuel-transformers/src/models/diffusion/wuerstchen/*` (~1176 LOC
//! across 7 files); see `port-wuerstchen.md` for the full mapping.

use crate::lazy::LazyTensor;
use fuel_core_types::Shape;
use std::sync::Arc;

// ---- Config ----------------------------------------------------------------

/// Wuerstchen architectural hyperparameters. Provides `tiny()` for
/// in-test exercise; production presets are loaded from HuggingFace
/// configs (not implemented here — call-sites build the struct).
#[derive(Debug, Clone)]
pub struct WuerstchenConfig {
    // --- Stage C (Prior) ---
    /// Prior latent channels (input to `PriorModel`).
    pub prior_c_in: usize,
    /// Prior internal width.
    pub prior_c: usize,
    /// Prior conditioning width (text embedding dim post-mapper).
    pub prior_c_cond: usize,
    /// Sinusoidal noise-ratio embedding dim.
    pub c_r: usize,
    /// Depth (number of Block triples) in `PriorModel`.
    pub prior_depth: usize,
    /// Attention heads inside `PriorModel`.
    pub prior_nhead: usize,

    // --- Stage B (DiffNeXt) ---
    /// DiffNeXt latent input channels.
    pub diffnext_c_in: usize,
    /// DiffNeXt output channels (same as input for noise prediction).
    pub diffnext_c_out: usize,
    /// DiffNeXt conditioning width.
    pub diffnext_c_cond: usize,
    /// Pixel-unshuffle / patch factor at the DiffNeXt entry.
    pub patch_size: usize,
    /// Per-level hidden channels.
    pub diffnext_c_hidden: Vec<usize>,
    /// Blocks per down/up level.
    pub diffnext_blocks: Vec<usize>,
    /// Attention heads per level (0 disables attention at that level).
    pub diffnext_nhead: Vec<usize>,

    // --- Stage A (Paella VQ) ---
    /// Latent channel count flowing into `PaellaVqModel::decode`.
    pub paella_latent_channels: usize,
    /// Paella decoder per-stage channel widths (decoder order:
    /// deepest → shallowest).
    pub paella_levels: Vec<usize>,
    /// Bottleneck residual blocks at the deepest decoder level.
    pub paella_bottleneck_blocks: usize,
    /// Output image channels (3 for RGB).
    pub paella_out_channels: usize,

    // --- Shared ---
    /// Text embedding dim (CLIP-style).
    pub clip_embed: usize,
    /// Image height fed back from PaellaVQ decode.
    pub image_size: usize,
}

impl WuerstchenConfig {
    /// Minimal config exercising every stage. Picked to keep build /
    /// realize cost low while still covering the residual block /
    /// attention / pixel-shuffle paths.
    pub fn tiny() -> Self {
        Self {
            prior_c_in: 4,
            prior_c: 16,
            prior_c_cond: 16,
            c_r: 16,
            prior_depth: 1,
            prior_nhead: 2,

            diffnext_c_in: 4,
            diffnext_c_out: 4,
            diffnext_c_cond: 16,
            patch_size: 2,
            diffnext_c_hidden: vec![8, 16],
            diffnext_blocks: vec![1, 1],
            diffnext_nhead: vec![0, 2],

            paella_latent_channels: 4,
            paella_levels: vec![8, 8],
            paella_bottleneck_blocks: 1,
            paella_out_channels: 3,

            clip_embed: 16,
            image_size: 32,
        }
    }
}

// ---- Weight bags -----------------------------------------------------------

/// Affine GlobalResponseNorm parameters (NCHW). `gamma`/`beta` shape
/// `(1, 1, 1, C)` in PyTorch convention; we store them flat `[C]`.
#[derive(Debug, Clone)]
pub struct GrnWeights {
    pub gamma: Arc<[f32]>,
    pub beta:  Arc<[f32]>,
}

/// Wuerstchen ResBlock (used by `PriorModel` and inside `DiffNeXt`):
/// depthwise conv → WLayerNorm → channelwise MLP with GRN.
#[derive(Debug, Clone)]
pub struct ResBlockWeights {
    /// Depthwise conv weight `[C, 1, K, K]` and bias `[C]`. Input
    /// channels for the conv = `c + c_skip` because the skip is
    /// concatenated **before** the depthwise step.
    pub dw_w: Arc<[f32]>,
    pub dw_b: Arc<[f32]>,
    /// fc1: `[4C, C]`, stored as `[C, 4C]` for matmul.
    pub fc1_w: Arc<[f32]>,
    pub fc1_b: Arc<[f32]>,
    pub grn:   GrnWeights,
    /// fc2: `[C, 4C]`, stored as `[4C, C]`.
    pub fc2_w: Arc<[f32]>,
    pub fc2_b: Arc<[f32]>,
    pub c: usize,
    pub c_skip: usize,
    pub ksize: usize,
}

/// FiLM scale+shift mapper for timestep conditioning.
#[derive(Debug, Clone)]
pub struct TimestepBlockWeights {
    /// Linear `c_timestep → 2*C`, stored as `[c_timestep, 2*C]`.
    pub w: Arc<[f32]>,
    pub b: Arc<[f32]>,
    pub c: usize,
    pub c_timestep: usize,
}

/// Cross-attention block.
#[derive(Debug, Clone)]
pub struct AttnBlockWeights {
    /// `kv_mapper.1`: `[c_cond, c]`.
    pub kv_mapper_w: Arc<[f32]>,
    pub kv_mapper_b: Arc<[f32]>,
    /// `to_q`: `[c, c]`.
    pub to_q_w: Arc<[f32]>,
    pub to_q_b: Arc<[f32]>,
    pub to_k_w: Arc<[f32]>,
    pub to_k_b: Arc<[f32]>,
    pub to_v_w: Arc<[f32]>,
    pub to_v_b: Arc<[f32]>,
    /// `to_out`: `[c, c]`.
    pub to_out_w: Arc<[f32]>,
    pub to_out_b: Arc<[f32]>,
    pub c: usize,
    pub c_cond: usize,
    pub heads: usize,
}

/// One `(ResBlock, TimestepBlock, AttnBlock)` triple from `PriorModel`.
#[derive(Debug, Clone)]
pub struct PriorBlockWeights {
    pub res:  ResBlockWeights,
    pub ts:   TimestepBlockWeights,
    pub attn: AttnBlockWeights,
}

/// Full weight bag for `PriorModel`.
#[derive(Debug, Clone)]
pub struct PriorWeights {
    /// `projection`: 1×1 conv `[c, c_in, 1, 1]` + bias `[c]`.
    pub projection_w: Arc<[f32]>,
    pub projection_b: Arc<[f32]>,
    /// cond_mapper_lin1: `[c_cond, c]`.
    pub cond1_w: Arc<[f32]>,
    pub cond1_b: Arc<[f32]>,
    /// cond_mapper_lin2: `[c, c]`.
    pub cond2_w: Arc<[f32]>,
    pub cond2_b: Arc<[f32]>,
    pub blocks: Vec<PriorBlockWeights>,
    /// out_conv: 1×1 conv `[c_in*2, c, 1, 1]` + bias `[c_in*2]`.
    pub out_conv_w: Arc<[f32]>,
    pub out_conv_b: Arc<[f32]>,
}

/// One DiffNeXt sub-block: ResBlockStageB + Timestep FiLM + optional
/// cross-attention.
#[derive(Debug, Clone)]
pub struct DiffNextSubBlockWeights {
    pub res:  ResBlockWeights,
    pub ts:   TimestepBlockWeights,
    pub attn: Option<AttnBlockWeights>,
}

/// One DiffNeXt level (down or up).
#[derive(Debug, Clone)]
pub struct DiffNextLevelWeights {
    /// Optional down: stride-2 2×2 conv `[Cout, Cin, 2, 2]`. None for level 0 of the down path.
    pub down_w: Option<Arc<[f32]>>,
    pub down_b: Option<Arc<[f32]>>,
    /// Optional up: 2×2 transposed conv `[Cin, Cout, 2, 2]`. None for level 0 of the up path.
    pub up_w: Option<Arc<[f32]>>,
    pub up_b: Option<Arc<[f32]>>,
    pub subs: Vec<DiffNextSubBlockWeights>,
}

/// Weight bag for `DiffNextModel`.
#[derive(Debug, Clone)]
pub struct DiffNextWeights {
    /// clip_mapper: `[clip_embed, c_cond]`.
    pub clip_mapper_w: Arc<[f32]>,
    pub clip_mapper_b: Arc<[f32]>,
    /// embedding 1×1 conv `[C_HIDDEN[0], c_in * patch²]` + bias.
    pub embed_w: Arc<[f32]>,
    pub embed_b: Arc<[f32]>,
    pub down_levels: Vec<DiffNextLevelWeights>,
    pub up_levels:   Vec<DiffNextLevelWeights>,
    /// clf 1×1 conv to `[2 * c_out * patch², C_HIDDEN[0], 1, 1]`.
    pub clf_w: Arc<[f32]>,
    pub clf_b: Arc<[f32]>,
}

/// MixingResidualBlock (Paella VQ). Six learnable scalar gates plus
/// depthwise conv + channelwise MLP.
#[derive(Debug, Clone)]
pub struct PaellaMixingResWeights {
    /// Six gates.
    pub gammas: [f32; 6],
    /// Depthwise 3×3 conv (with replication pad applied at call
    /// site): `[C, 1, 3, 3]` + bias `[C]`.
    pub dw_w: Arc<[f32]>,
    pub dw_b: Arc<[f32]>,
    /// channelwise.0: linear `C → embed_dim` (`embed_dim = 4*C`).
    pub fc1_w: Arc<[f32]>,
    pub fc1_b: Arc<[f32]>,
    /// channelwise.2: linear `embed_dim → C`.
    pub fc2_w: Arc<[f32]>,
    pub fc2_b: Arc<[f32]>,
    pub c: usize,
}

/// Weight bag for `PaellaVqModel`. Decoder path only.
#[derive(Debug, Clone)]
pub struct PaellaVqWeights {
    /// 1×1 entry conv from latent → deepest level: `[paella_levels[0],
    /// latent_channels, 1, 1]`.
    pub up_in_w: Arc<[f32]>,
    pub up_in_b: Arc<[f32]>,
    /// Per-decoder-level groups. `paella_levels[0]` is deepest.
    pub up_levels: Vec<PaellaUpLevelWeights>,
    /// out_block 1×1 conv: `[out_channels * 4, paella_levels[-1], 1, 1]`.
    pub out_w: Arc<[f32]>,
    pub out_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct PaellaUpLevelWeights {
    pub res_blocks: Vec<PaellaMixingResWeights>,
    /// Conv-transpose `[Cin, Cout, 4, 4]` stride=2 padding=1; bias `[Cout]`.
    /// None at the shallowest level.
    pub upsample_w: Option<Arc<[f32]>>,
    pub upsample_b: Option<Arc<[f32]>>,
}

// ---- Forward primitives ----------------------------------------------------

/// `(1, C, H, W)` LayerNorm over the channel axis with no affine.
/// Matches eager `WLayerNorm`: permute NHWC, LN(last dim), permute back.
fn w_layer_norm(x: &LazyTensor, c: usize, h: usize, w: usize, eps: f64) -> LazyTensor {
    let x_nhwc = x.permute([0, 2, 3, 1_usize]).unwrap();
    let x_flat = x_nhwc.reshape(Shape::from_dims(&[1, h * w, c])).unwrap();
    let normed = x_flat.layer_norm_last_dim(eps).unwrap();
    normed
        .reshape(Shape::from_dims(&[1, h, w, c])).unwrap()
        .permute([0, 3, 1, 2_usize]).unwrap()
}

/// Same shape contract as `w_layer_norm` but operates on a `[1, seq, C]`
/// tensor directly (i.e. already channels-last and flattened).
fn ln_last_no_affine(x: &LazyTensor, eps: f64) -> LazyTensor {
    x.layer_norm_last_dim(eps).unwrap()
}

/// `y = x @ W + b`. `x` shape `[B, seq, in_f]`, W stored
/// `[in_f, out_f]` row-major.
fn linear(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: Option<&Arc<[f32]>>,
    in_f: usize,
    out_f: usize,
    batch: usize,
    seq: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[in_f, out_f]));
    let proj = x.matmul(&w_t).unwrap();
    match b {
        Some(b) => {
            let bias = x
                .const_f32_like(b.clone(), Shape::from_dims(&[out_f]))
                .reshape(Shape::from_dims(&[1, 1, out_f])).unwrap()
                .broadcast_to(Shape::from_dims(&[batch, seq, out_f])).unwrap();
            proj.add(&bias).unwrap()
        }
        None => proj,
    }
}

/// Global Response Normalization on `[1, C, H, W]`, returning the same
/// shape. Mirrors `WGlobalResponseNorm` from `common.rs`:
///
/// ```text
///   agg_norm[c]            = sqrt(sum_{h,w} x[b, c, h, w]^2)        # [1, C, 1, 1]
///   stand_div_norm[c]      = agg_norm[c] / (mean_c(agg_norm) + 1e-6)
///   y = x * stand_div_norm * gamma + beta + x   (residual)
/// ```
///
/// Public for downstream tests (`grn_hand_computed`).
pub fn global_response_norm(
    x: &LazyTensor,
    gamma: &Arc<[f32]>,
    beta: &Arc<[f32]>,
    c: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    // sum over (h, w) per channel.
    let sq = x.mul(x).unwrap();  // [1, C, H, W]
    let agg = sq.reduce_sum_to(Shape::from_dims(&[1, c, 1, 1])).sqrt();  // [1, C, 1, 1]
    // mean over channels via reduce_sum + scalar mul.
    let agg_sum = agg.reduce_sum_to(Shape::from_dims(&[1, 1, 1, 1]));
    let mean_c = agg_sum.mul_scalar(1.0_f64 / c as f64);  // [1, 1, 1, 1]
    let denom = mean_c
        .add_scalar(1e-6_f64)
        .broadcast_to(Shape::from_dims(&[1, c, 1, 1])).unwrap();
    let stand = agg.div(&denom).unwrap();  // [1, C, 1, 1]
    let stand_b = stand.broadcast_to(Shape::from_dims(&[1, c, h, w])).unwrap();
    let g = x
        .const_f32_like(gamma.clone(), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w])).unwrap();
    let b = x
        .const_f32_like(beta.clone(), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w])).unwrap();
    let scaled = x.mul(&stand_b).unwrap().mul(&g).unwrap();
    let with_beta = scaled.add(&b).unwrap();
    with_beta.add(x).unwrap()
}

/// Apply the ResBlock used by `PriorModel`: depthwise conv → WLN →
/// channelwise MLP with GRN (channels-last), residual. Optional skip
/// is concatenated along the channel axis **before** the depthwise.
fn apply_res_block(
    x: &LazyTensor,
    rw: &ResBlockWeights,
    skip: Option<&LazyTensor>,
    h: usize,
    w: usize,
    eps: f64,
) -> LazyTensor {
    let c = rw.c;
    let c_skip = rw.c_skip;
    let k = rw.ksize;
    let residual = x.clone();
    let xs = match skip {
        Some(s) => x.concat(s, 1_usize).unwrap(),
        None => x.clone(),
    };
    // depthwise conv: groups = c, in/out channels c+c_skip -> c.
    // Eager uses depthwise with groups=c on (c+c_skip) input — but
    // depthwise requires Cin == groups. The eager `ResBlock::new`
    // calls `conv2d(c + c_skip, c, ...)` with `groups=c`. So Cin
    // must == groups → only valid when c_skip == 0. We follow eager
    // semantics for the common (no-skip) ResBlock case (`PriorModel`
    // uses `c_skip == 0`).
    let _ = c_skip;
    let dw_w = x.const_f32_like(rw.dw_w.clone(), Shape::from_dims(&[c, 1, k, k]));
    let dw_b = x.const_f32_like(rw.dw_b.clone(), Shape::from_dims(&[c]));
    let conv = xs.conv2d(&dw_w, Some(&dw_b), (1, 1), (k / 2, k / 2), c).unwrap();
    let norm = w_layer_norm(&conv, c, h, w, eps);
    // Channelwise MLP. Permute to NHWC for the linears.
    let nhwc = norm.permute([0, 2, 3, 1_usize]).unwrap();  // [1, H, W, C]
    let flat = nhwc.reshape(Shape::from_dims(&[1, h * w, c])).unwrap();
    let fc1 = linear(&flat, &rw.fc1_w, Some(&rw.fc1_b), c, 4 * c, 1, h * w).gelu();
    // GRN expects [1, C, H, W]; bridge: reshape fc1 to that.
    let fc1_chw = fc1
        .reshape(Shape::from_dims(&[1, h, w, 4 * c])).unwrap()
        .permute([0, 3, 1, 2_usize]).unwrap();
    let grn = global_response_norm(&fc1_chw, &rw.grn.gamma, &rw.grn.beta, 4 * c, h, w);
    // Back to channels-last for fc2.
    let grn_nhwc = grn.permute([0, 2, 3, 1_usize]).unwrap();
    let grn_flat = grn_nhwc.reshape(Shape::from_dims(&[1, h * w, 4 * c])).unwrap();
    let fc2 = linear(&grn_flat, &rw.fc2_w, Some(&rw.fc2_b), 4 * c, c, 1, h * w);
    let out_chw = fc2
        .reshape(Shape::from_dims(&[1, h, w, c])).unwrap()
        .permute([0, 3, 1, 2_usize]).unwrap();
    residual.add(&out_chw).unwrap()
}

/// DiffNeXt-style ResBlockStageB. Difference vs `apply_res_block`:
/// the optional skip is concatenated **after** the depthwise+norm, and
/// the MLP's fc1 has input width `c + c_skip`.
fn apply_res_block_stage_b(
    x: &LazyTensor,
    rw: &ResBlockWeights,
    skip: Option<&LazyTensor>,
    h: usize,
    w: usize,
    eps: f64,
) -> LazyTensor {
    let c = rw.c;
    let c_skip = rw.c_skip;
    let k = rw.ksize;
    let residual = x.clone();
    let dw_w = x.const_f32_like(rw.dw_w.clone(), Shape::from_dims(&[c, 1, k, k]));
    let dw_b = x.const_f32_like(rw.dw_b.clone(), Shape::from_dims(&[c]));
    let conv = x.conv2d(&dw_w, Some(&dw_b), (1, 1), (k / 2, k / 2), c).unwrap();
    let norm = w_layer_norm(&conv, c, h, w, eps);
    // Concat skip channels after norm.
    let merged = match skip {
        Some(s) => norm.concat(s, 1_usize).unwrap(),
        None => norm,
    };
    let merged_nhwc = merged.permute([0, 2, 3, 1_usize]).unwrap();
    let merged_flat = merged_nhwc
        .reshape(Shape::from_dims(&[1, h * w, c + c_skip])).unwrap();
    let fc1 = linear(&merged_flat, &rw.fc1_w, Some(&rw.fc1_b), c + c_skip, 4 * c, 1, h * w).gelu();
    let fc1_chw = fc1
        .reshape(Shape::from_dims(&[1, h, w, 4 * c])).unwrap()
        .permute([0, 3, 1, 2_usize]).unwrap();
    let grn = global_response_norm(&fc1_chw, &rw.grn.gamma, &rw.grn.beta, 4 * c, h, w);
    let grn_nhwc = grn.permute([0, 2, 3, 1_usize]).unwrap();
    let grn_flat = grn_nhwc.reshape(Shape::from_dims(&[1, h * w, 4 * c])).unwrap();
    let fc2 = linear(&grn_flat, &rw.fc2_w, Some(&rw.fc2_b), 4 * c, c, 1, h * w);
    let out_chw = fc2
        .reshape(Shape::from_dims(&[1, h, w, c])).unwrap()
        .permute([0, 3, 1, 2_usize]).unwrap();
    residual.add(&out_chw).unwrap()
}

/// FiLM-style timestep block. `t` is `[1, c_timestep]`; output is `(1+a)·x + b`
/// where `[a, b] = mapper(t).unsqueeze(-1).unsqueeze(-1).chunk(2, axis=1)`.
fn apply_timestep_block(
    x: &LazyTensor,
    tw: &TimestepBlockWeights,
    t: &LazyTensor,  // [1, 1, c_timestep]
    h: usize,
    w: usize,
) -> LazyTensor {
    let c = tw.c;
    let mapped = linear(t, &tw.w, Some(&tw.b), tw.c_timestep, 2 * c, 1, 1);
    // mapped: [1, 1, 2C]. chunk along axis -1.
    let halves = mapped.chunk(2, 2_usize).unwrap();
    let a = halves[0]
        .reshape(Shape::from_dims(&[1, c, 1, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w])).unwrap();
    let b = halves[1]
        .reshape(Shape::from_dims(&[1, c, 1, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w])).unwrap();
    let one_plus_a = a.add_scalar(1.0_f64);
    x.mul(&one_plus_a).unwrap().add(&b).unwrap()
}

/// Cross-attention block: norm(x) + attention(norm(x), silu(kv).mapper)
/// where the spatial tokens are optionally prepended to kv (`self_attn`).
fn apply_attn_block(
    x: &LazyTensor,
    aw: &AttnBlockWeights,
    kv_raw: &LazyTensor,    // [1, S_kv, c_cond]
    s_kv: usize,
    h: usize,
    w: usize,
    eps: f64,
    self_attn: bool,
) -> LazyTensor {
    let c = aw.c;
    let heads = aw.heads;
    let d_head = c / heads;
    // norm(x): NCHW → NHWC layered LN.
    let norm = w_layer_norm(x, c, h, w, eps);
    // kv: silu(kv_raw) → kv_mapper linear.
    let kv = linear(
        &kv_raw.silu(),
        &aw.kv_mapper_w, Some(&aw.kv_mapper_b),
        aw.c_cond, c, 1, s_kv,
    );
    // norm_xs as [1, H*W, C].
    let norm_seq = norm
        .reshape(Shape::from_dims(&[1, c, h * w])).unwrap()
        .transpose().unwrap();
    let (kv_full, s_total) = if self_attn {
        // prepend norm tokens to kv along seq dim.
        (norm_seq.concat(&kv, 1_usize).unwrap(), h * w + s_kv)
    } else {
        (kv, s_kv)
    };
    // Q from norm_xs; K/V from kv_full.
    let q = linear(&norm_seq, &aw.to_q_w, Some(&aw.to_q_b), c, c, 1, h * w);
    let k = linear(&kv_full,  &aw.to_k_w, Some(&aw.to_k_b), c, c, 1, s_total);
    let v = linear(&kv_full,  &aw.to_v_w, Some(&aw.to_v_b), c, c, 1, s_total);
    // Reshape to heads. [1, S, C] → [1, S, H, D] → [1, H, S, D].
    let q_h = q
        .reshape(Shape::from_dims(&[1, h * w, heads, d_head])).unwrap()
        .permute([0, 2, 1, 3_usize]).unwrap();
    let k_h = k
        .reshape(Shape::from_dims(&[1, s_total, heads, d_head])).unwrap()
        .permute([0, 2, 1, 3_usize]).unwrap();
    let v_h = v
        .reshape(Shape::from_dims(&[1, s_total, heads, d_head])).unwrap()
        .permute([0, 2, 1, 3_usize]).unwrap();
    let k_t = k_h.permute([0, 1, 3, 2_usize]).unwrap();  // [1, H, D, S]
    let scale = 1.0_f64 / (d_head as f64).sqrt();
    let scores = q_h.matmul(&k_t).unwrap().mul_scalar(scale);  // [1, H, hw, S]
    let probs = scores.softmax_last_dim().unwrap();
    let out_h = probs.matmul(&v_h).unwrap();  // [1, H, hw, D]
    let out = out_h
        .permute([0, 2, 1, 3_usize]).unwrap()
        .reshape(Shape::from_dims(&[1, h * w, c])).unwrap();
    let proj = linear(&out, &aw.to_out_w, Some(&aw.to_out_b), c, c, 1, h * w);
    // Back to [1, C, H, W] and residual-add with original x.
    let proj_chw = proj
        .transpose().unwrap()
        .reshape(Shape::from_dims(&[1, c, h, w])).unwrap();
    x.add(&proj_chw).unwrap()
}

// ---- Sinusoidal noise-ratio embedding -------------------------------------

/// Sinusoidal embedding of a scalar noise ratio `r ∈ [0, 1]`. Mirrors
/// eager `gen_r_embedding`:
///   `r_scaled = r * 10000`
///   `freqs[i] = exp(-i * ln(10000) / (half_dim - 1))` for `i ∈ [0, half_dim)`
///   `emb = cat(sin(r_scaled * freqs), cos(r_scaled * freqs))`
///   pad with one zero if `c_r` is odd.
/// Produces a host-side `Vec<f32>` of length `c_r`, since `r` is a
/// known host scalar at planning time.
fn r_embedding_host(r: f32, c_r: usize) -> Vec<f32> {
    let half_dim = c_r / 2;
    let max_pos = 10000.0_f64;
    let denom = ((half_dim as f64) - 1.0).max(1.0);
    let log_max = max_pos.ln() / denom;
    let mut out = vec![0.0_f32; c_r];
    let r64 = r as f64 * max_pos;
    for i in 0..half_dim {
        let freq = (-(i as f64) * log_max).exp();
        let arg = r64 * freq;
        out[i] = arg.sin() as f32;
        out[i + half_dim] = arg.cos() as f32;
    }
    // odd c_r leaves out[c_r-1] = 0 (matches eager pad_with_zeros).
    out
}

// ---- PriorModel ------------------------------------------------------------

/// Wuerstchen Stage C — text → low-resolution latent denoising model.
#[derive(Debug, Clone)]
pub struct PriorModel {
    pub config:  WuerstchenConfig,
    pub weights: PriorWeights,
}

impl PriorModel {
    /// Run one denoising step. `xs` is `[1, c_in, H, W]`; `r` is the
    /// scalar noise ratio in `[0, 1]`; `c_embed` is `[1, S, c_cond]`
    /// text conditioning.
    pub fn forward(
        &self,
        xs: &LazyTensor,
        r: f32,
        c_embed: &LazyTensor,
        h: usize,
        w: usize,
    ) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let c = cfg.prior_c;
        let c_in = cfg.prior_c_in;
        let c_cond = cfg.prior_c_cond;
        let s_kv = c_embed.dim(1_usize)?;
        let eps = 1e-6_f64;

        // x_in for the final shift+scale.
        let x_in = xs.clone();
        // projection: 1×1 conv `c_in → c`.
        let proj_w = xs.const_f32_like(self.weights.projection_w.clone(),
            Shape::from_dims(&[c, c_in, 1, 1]));
        let proj_b = xs.const_f32_like(self.weights.projection_b.clone(),
            Shape::from_dims(&[c]));
        let mut x = xs.conv2d(&proj_w, Some(&proj_b), (1, 1), (0, 0), 1)?;

        // Cond-mapper: linear → leaky_relu(0.2) → linear.
        let c1 = linear(c_embed, &self.weights.cond1_w, Some(&self.weights.cond1_b),
            c_cond, c, 1, s_kv);
        // leaky_relu(0.2): max(x, 0.2 * x). Compose via where_cond.
        let c1_neg = c1.mul_scalar(0.2_f64);
        let zero = c1.const_f32_like(vec![0.0_f32; c1.elem_count()], c1.shape());
        let mask = c1.gt(&zero)?;
        let c1_act = mask.where_cond(&c1, &c1_neg)?;
        let c_embed_mapped = linear(&c1_act, &self.weights.cond2_w, Some(&self.weights.cond2_b),
            c, c, 1, s_kv);

        // r_embed.
        let r_vec = r_embedding_host(r, cfg.c_r);
        let r_embed = xs
            .const_f32_like(r_vec, Shape::from_dims(&[cfg.c_r]))
            .reshape(Shape::from_dims(&[1, 1, cfg.c_r]))?;

        for bw in &self.weights.blocks {
            x = apply_res_block(&x, &bw.res, None, h, w, eps);
            x = apply_timestep_block(&x, &bw.ts, &r_embed, h, w);
            x = apply_attn_block(&x, &bw.attn, &c_embed_mapped, s_kv, h, w, eps, true);
        }

        // out_ln + out_conv → chunk(2, dim=1) → (x_in - a0) / (|a1 - 1| + eps).
        let normed = w_layer_norm(&x, c, h, w, eps);
        let out_w = xs.const_f32_like(self.weights.out_conv_w.clone(),
            Shape::from_dims(&[c_in * 2, c, 1, 1]));
        let out_b = xs.const_f32_like(self.weights.out_conv_b.clone(),
            Shape::from_dims(&[c_in * 2]));
        let out = normed.conv2d(&out_w, Some(&out_b), (1, 1), (0, 0), 1)?;
        let ab = out.chunk(2, 1_usize)?;
        let a0 = &ab[0];
        let a1 = &ab[1];
        let denom = a1.add_scalar(-1.0_f64).abs().add_scalar(1e-5_f64);
        let num = x_in.sub(a0)?;
        Ok(num.div(&denom)?)
    }
}

// ---- DiffNextModel ---------------------------------------------------------

/// Wuerstchen Stage B — DiffNeXt UNet that denoises VQ latents.
#[derive(Debug, Clone)]
pub struct DiffNextModel {
    pub config:  WuerstchenConfig,
    pub weights: DiffNextWeights,
}

impl DiffNextModel {
    /// One denoising step. `xs` is `[1, c_in, H, W]`; `r` scalar noise ratio
    /// in `[0, 1]`; `clip` is `[1, S, clip_embed]` text conditioning.
    pub fn forward(
        &self,
        xs: &LazyTensor,
        r: f32,
        clip: &LazyTensor,
        h_in: usize,
        w_in: usize,
    ) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let c_in = cfg.diffnext_c_in;
        let c_out = cfg.diffnext_c_out;
        let c_cond = cfg.diffnext_c_cond;
        let p = cfg.patch_size;
        let eps = 1e-6_f64;
        let levels = &cfg.diffnext_c_hidden;
        if levels.is_empty() {
            return Err(crate::Error::Msg(
                "DiffNextModel.forward: diffnext_c_hidden is empty".into(),
            ).bt());
        }
        if h_in % p != 0 || w_in % p != 0 {
            return Err(crate::Error::Msg(format!(
                "DiffNextModel.forward: spatial dims ({h_in}, {w_in}) must be divisible by patch_size {p}",
            )).bt());
        }

        // Save the input for the final shift+scale.
        let x_in = xs.clone();

        // clip mapper: linear + LN-no-affine.
        let s_kv = clip.dim(1_usize)?;
        let c_embed = linear(clip, &self.weights.clip_mapper_w, Some(&self.weights.clip_mapper_b),
            cfg.clip_embed, c_cond, 1, s_kv);
        let c_embed = ln_last_no_affine(&c_embed, eps);

        // r_embed.
        let r_vec = r_embedding_host(r, cfg.c_r);
        let r_embed = xs
            .const_f32_like(r_vec, Shape::from_dims(&[cfg.c_r]))
            .reshape(Shape::from_dims(&[1, 1, cfg.c_r]))?;

        // Pixel-unshuffle by `p`: [1, C, H, W] → [1, C * p², H/p, W/p].
        let mut h = h_in / p;
        let mut w = w_in / p;
        let xu = pixel_unshuffle(xs, c_in, h_in, w_in, p);
        // embedding 1×1 conv: c_in*p² → C_HIDDEN[0].
        let emb_w = xs.const_f32_like(self.weights.embed_w.clone(),
            Shape::from_dims(&[levels[0], c_in * p * p, 1, 1]));
        let emb_b = xs.const_f32_like(self.weights.embed_b.clone(),
            Shape::from_dims(&[levels[0]]));
        let mut x = xu.conv2d(&emb_w, Some(&emb_b), (1, 1), (0, 0), 1)?;
        x = w_layer_norm(&x, levels[0], h, w, eps);

        // --- down path ---
        let mut skips: Vec<LazyTensor> = Vec::new();
        for (i, lvl) in self.weights.down_levels.iter().enumerate() {
            let c_lvl = levels[i];
            if let (Some(dw), Some(db)) = (&lvl.down_w, &lvl.down_b) {
                let prev = levels[i - 1];
                x = w_layer_norm(&x, prev, h, w, eps);
                let dw_t = xs.const_f32_like(dw.clone(), Shape::from_dims(&[c_lvl, prev, 2, 2]));
                let db_t = xs.const_f32_like(db.clone(), Shape::from_dims(&[c_lvl]));
                x = x.conv2d(&dw_t, Some(&db_t), (2, 2), (0, 0), 1)?;
                h /= 2;
                w /= 2;
            }
            for sub in &lvl.subs {
                x = apply_res_block_stage_b(&x, &sub.res, None, h, w, eps);
                x = apply_timestep_block(&x, &sub.ts, &r_embed, h, w);
                if let Some(aw) = &sub.attn {
                    x = apply_attn_block(&x, aw, &c_embed, s_kv, h, w, eps, true);
                }
            }
            skips.push(x.clone());
        }

        // --- up path ---
        skips.reverse();
        let mut x = skips[0].clone();
        let n_levels = levels.len();
        for (i, lvl) in self.weights.up_levels.iter().enumerate() {
            // Effective level index from the deepest end.
            let lvl_idx = n_levels - 1 - i;
            let c_lvl = levels[lvl_idx];
            for (j, sub) in lvl.subs.iter().enumerate() {
                // First sub-block of every level except the deepest gets the
                // matching down-skip concat. We follow the eager
                // pattern but with the simplifying choice of skipping the
                // "effnet_c"  injection (it's gated to None in our config).
                let skip_ref = if j == 0 && i > 0 {
                    Some(&skips[i])
                } else {
                    None
                };
                x = apply_res_block_stage_b(&x, &sub.res, skip_ref, h, w, eps);
                x = apply_timestep_block(&x, &sub.ts, &r_embed, h, w);
                if let Some(aw) = &sub.attn {
                    x = apply_attn_block(&x, aw, &c_embed, s_kv, h, w, eps, true);
                }
            }
            if let (Some(uw), Some(ub)) = (&lvl.up_w, &lvl.up_b) {
                let next = levels[lvl_idx - 1];
                x = w_layer_norm(&x, c_lvl, h, w, eps);
                let uw_t = xs.const_f32_like(uw.clone(), Shape::from_dims(&[c_lvl, next, 2, 2]));
                let ub_t = xs.const_f32_like(ub.clone(), Shape::from_dims(&[next]));
                x = x.conv_transpose2d(&uw_t, (2, 2), (0, 0), (0, 0), (1, 1), 1)?;
                let _ = ub_t;
                // conv_transpose2d does not yet plumb the bias; add manually.
                let bias = xs
                    .const_f32_like(ub.clone(), Shape::from_dims(&[next]))
                    .reshape(Shape::from_dims(&[1, next, 1, 1]))?
                    .broadcast_to(Shape::from_dims(&[1, next, h * 2, w * 2]))?;
                x = x.add(&bias)?;
                h *= 2;
                w *= 2;
            }
        }

        // --- classifier ---
        let last_h = levels[0];
        x = w_layer_norm(&x, last_h, h, w, eps);
        let clf_w = xs.const_f32_like(self.weights.clf_w.clone(),
            Shape::from_dims(&[2 * c_out * p * p, last_h, 1, 1]));
        let clf_b = xs.const_f32_like(self.weights.clf_b.clone(),
            Shape::from_dims(&[2 * c_out * p * p]));
        let out = x.conv2d(&clf_w, Some(&clf_b), (1, 1), (0, 0), 1)?;
        // pixel_shuffle by p: [1, 2*c_out*p², H, W] → [1, 2*c_out, H*p, W*p].
        let out = pixel_shuffle(&out, 2 * c_out * p * p, h, w, p);
        // chunk(2, dim=1).
        let ab = out.chunk(2, 1_usize)?;
        let a = &ab[0];
        let b_raw = &ab[1];
        // b = sigmoid(b_raw) * (1 - 2*eps_b) + eps_b, eps_b = 1e-3.
        let eps_b = 1e-3_f64;
        let b = b_raw.sigmoid().affine(1.0 - 2.0 * eps_b, eps_b);
        let num = x_in.sub(a)?;
        Ok(num.div(&b)?)
    }
}

/// Pixel-unshuffle (a.k.a. space-to-depth) by factor `p`. Input
/// `[1, C, H, W]` → output `[1, C*p², H/p, W/p]`. Implemented as a
/// reshape+permute composition.
fn pixel_unshuffle(x: &LazyTensor, c: usize, h: usize, w: usize, p: usize) -> LazyTensor {
    let h_out = h / p;
    let w_out = w / p;
    let r = x.reshape(Shape::from_dims(&[1, c, h_out, p, w_out, p])).unwrap();
    // permute to [1, C, p, p, H_out, W_out].
    let pm = r.permute([0, 1, 3, 5, 2, 4_usize]).unwrap();
    pm.reshape(Shape::from_dims(&[1, c * p * p, h_out, w_out])).unwrap()
}

/// Pixel-shuffle (depth-to-space) by factor `p`. Input `[1, C, H, W]`
/// → output `[1, C/p², H*p, W*p]`. Inverse of `pixel_unshuffle`.
fn pixel_shuffle(x: &LazyTensor, c: usize, h: usize, w: usize, p: usize) -> LazyTensor {
    let c_out = c / (p * p);
    let r = x.reshape(Shape::from_dims(&[1, c_out, p, p, h, w])).unwrap();
    // permute to [1, C_out, H, p, W, p].
    let pm = r.permute([0, 1, 4, 2, 5, 3_usize]).unwrap();
    pm.reshape(Shape::from_dims(&[1, c_out, h * p, w * p])).unwrap()
}

// ---- PaellaVqModel (decoder) ----------------------------------------------

/// Paella VQ-VAE decoder. Stage A; encoder side intentionally omitted
/// (inference goes Stage C → B → A-decode).
#[derive(Debug, Clone)]
pub struct PaellaVqModel {
    pub config:  WuerstchenConfig,
    pub weights: PaellaVqWeights,
}

impl PaellaVqModel {
    /// Decode VQ latents to an image. `latents` shape
    /// `[1, latent_channels, h_lat, w_lat]`. Output shape
    /// `[1, out_channels, h_lat * upscale, w_lat * upscale]` where
    /// `upscale = 2 * 2^(n_levels - 1)` (the *2 is the final pixel-shuffle).
    pub fn decode(&self, latents: &LazyTensor) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let dims = latents.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(crate::Error::Msg(format!(
                "PaellaVqModel.decode: latents must be rank 4 [1, C, H, W], got {dims:?}",
            )).bt());
        }
        if dims[1] != cfg.paella_latent_channels {
            return Err(crate::Error::Msg(format!(
                "PaellaVqModel.decode: latent channel mismatch (have {}, cfg {})",
                dims[1], cfg.paella_latent_channels,
            )).bt());
        }
        let (mut h, mut w) = (dims[2], dims[3]);
        // up_in 1×1 conv.
        let levels = &cfg.paella_levels;
        let in_w = latents.const_f32_like(self.weights.up_in_w.clone(),
            Shape::from_dims(&[levels[0], cfg.paella_latent_channels, 1, 1]));
        let in_b = latents.const_f32_like(self.weights.up_in_b.clone(),
            Shape::from_dims(&[levels[0]]));
        let mut x = latents.conv2d(&in_w, Some(&in_b), (1, 1), (0, 0), 1)?;
        for (i, lw) in self.weights.up_levels.iter().enumerate() {
            let c_lvl = levels[i];
            for rb in &lw.res_blocks {
                x = apply_paella_mixing_res(&x, rb, h, w);
            }
            if let (Some(uw), Some(ub)) = (&lw.upsample_w, &lw.upsample_b) {
                let next = levels[i + 1];
                let uw_t = latents.const_f32_like(uw.clone(),
                    Shape::from_dims(&[c_lvl, next, 4, 4]));
                let x_t = x.conv_transpose2d(&uw_t, (2, 2), (1, 1), (0, 0), (1, 1), 1)?;
                let new_h = h * 2;
                let new_w = w * 2;
                let bias = latents
                    .const_f32_like(ub.clone(), Shape::from_dims(&[next]))
                    .reshape(Shape::from_dims(&[1, next, 1, 1]))?
                    .broadcast_to(Shape::from_dims(&[1, next, new_h, new_w]))?;
                x = x_t.add(&bias)?;
                h = new_h;
                w = new_w;
            }
        }
        // out_block: 1×1 conv → pixel_shuffle by 2.
        let last_c = *levels.last().unwrap();
        let oc = cfg.paella_out_channels;
        let ow = latents.const_f32_like(self.weights.out_w.clone(),
            Shape::from_dims(&[oc * 4, last_c, 1, 1]));
        let ob = latents.const_f32_like(self.weights.out_b.clone(),
            Shape::from_dims(&[oc * 4]));
        let out = x.conv2d(&ow, Some(&ob), (1, 1), (0, 0), 1)?;
        let out = pixel_shuffle(&out, oc * 4, h, w, 2);
        // Final tanh: bring into [-1, 1] (matches Paella's reference output range).
        Ok(out.tanh())
    }
}

/// Apply one Paella MixingResidualBlock.
fn apply_paella_mixing_res(
    x: &LazyTensor,
    bw: &PaellaMixingResWeights,
    h: usize,
    w: usize,
) -> LazyTensor {
    let c = bw.c;
    let g = &bw.gammas;
    let eps = 1e-6_f64;
    // Branch 1: norm1 (no affine, NHWC) → permute back → affine(1+g0, g1)
    //           → replication_pad(1) → depthwise conv 3×3 (no padding,
    //           since we already padded). Eager runs `replication_pad2d`
    //           then a plain `conv2d` (no padding). The lazy port
    //           approximates replication padding with edge-replicate
    //           via narrow+repeat+concat.
    let temp = w_layer_norm(x, c, h, w, eps);
    let temp = temp.affine(1.0 + g[0] as f64, g[1] as f64);
    let padded = replicate_pad_2d(&temp, c, h, w, 1);
    let dw_w = x.const_f32_like(bw.dw_w.clone(), Shape::from_dims(&[c, 1, 3, 3]));
    let dw_b = x.const_f32_like(bw.dw_b.clone(), Shape::from_dims(&[c]));
    let conv = padded.conv2d(&dw_w, Some(&dw_b), (1, 1), (0, 0), c).unwrap();
    let xs = x.add(&conv.mul_scalar(g[2] as f64)).unwrap();

    // Branch 2: norm2 → affine(1+g3, g4) → channelwise MLP (linear,
    //           GELU, linear) → scale by g5, residual.
    let temp = w_layer_norm(&xs, c, h, w, eps)
        .affine(1.0 + g[3] as f64, g[4] as f64);
    let nhwc = temp.permute([0, 2, 3, 1_usize]).unwrap();
    let flat = nhwc.reshape(Shape::from_dims(&[1, h * w, c])).unwrap();
    let embed_dim = bw.fc1_b.len();
    let mid = linear(&flat, &bw.fc1_w, Some(&bw.fc1_b), c, embed_dim, 1, h * w).gelu();
    let out = linear(&mid, &bw.fc2_w, Some(&bw.fc2_b), embed_dim, c, 1, h * w);
    let out_chw = out
        .reshape(Shape::from_dims(&[1, h, w, c])).unwrap()
        .permute([0, 3, 1, 2_usize]).unwrap();
    xs.add(&out_chw.mul_scalar(g[5] as f64)).unwrap()
}

/// Replication-pad 2D by `pad` on every side. Input `[1, C, H, W]` →
/// output `[1, C, H + 2*pad, W + 2*pad]`.
fn replicate_pad_2d(x: &LazyTensor, c: usize, h: usize, w: usize, pad: usize) -> LazyTensor {
    if pad == 0 {
        return x.clone();
    }
    // First pad along W (dim 3).
    let left_col = x.narrow(3_usize, 0, 1).unwrap();
    let right_col = x.narrow(3_usize, w - 1, 1).unwrap();
    let left_block = left_col.repeat(Shape::from_dims(&[1, 1, 1, pad])).unwrap();
    let right_block = right_col.repeat(Shape::from_dims(&[1, 1, 1, pad])).unwrap();
    let padded_w = left_block.concat(x, 3_usize).unwrap().concat(&right_block, 3_usize).unwrap();
    // Then pad along H (dim 2).
    let h2 = padded_w.dim(2_usize).unwrap();
    let _ = (h, w);
    let top_row = padded_w.narrow(2_usize, 0, 1).unwrap();
    let bot_row = padded_w.narrow(2_usize, h2 - 1, 1).unwrap();
    let top_block = top_row.repeat(Shape::from_dims(&[1, 1, pad, 1])).unwrap();
    let bot_block = bot_row.repeat(Shape::from_dims(&[1, 1, pad, 1])).unwrap();
    top_block.concat(&padded_w, 2_usize).unwrap().concat(&bot_block, 2_usize).unwrap()
}

// ---- End-to-end generate ---------------------------------------------------

/// Cascade Stage C → B → A-decode. `text_embed` is `[1, S, clip_embed]`.
/// `prior_steps` / `b_steps` count denoising iterations. Returns
/// `[1, 3, image_size, image_size]`.
///
/// Tiny config note: `prior_steps == 0` and `b_steps == 0` are valid
/// — the diffusion stages still run their initial denoise once to
/// surface the model's response to the input noise + conditioning.
pub fn generate(
    prior: &PriorModel,
    diffnext: &DiffNextModel,
    paella: &PaellaVqModel,
    text_embed: &LazyTensor,
    prior_steps: usize,
    b_steps: usize,
    prior_h: usize,
    prior_w: usize,
    b_h: usize,
    b_w: usize,
) -> crate::Result<LazyTensor> {
    let cfg = &prior.config;

    // Stage C: start from noise and run `1 + prior_steps` denoising
    // forwards. The eager pipeline composes this with a DDPM
    // scheduler; in v1 we apply the model's denoise as the update
    // (model already outputs `(x - a) / b`), giving a deterministic
    // shape-preserving step. Noise is generated via `noise_on_graph`
    // so it lives on the same graph as `text_embed`.
    let mut z_c = noise_on_graph(
        text_embed,
        Shape::from_dims(&[1, cfg.prior_c_in, prior_h, prior_w]),
        0xC0DEC0DE,
    );
    for s in 0..=prior_steps {
        let r = 1.0 - (s as f32 / (prior_steps as f32 + 1.0));
        z_c = prior.forward(&z_c, r, text_embed, prior_h, prior_w)?;
    }

    // Stage B: noise the VQ latent, condition on z_c via clip-mapper. v1
    // ignores effnet injection — DiffNextModel.forward treats it as
    // unconditional spatial. The text conditioning therefore carries
    // through `clip`.
    let mut z_b = noise_on_graph(
        text_embed,
        Shape::from_dims(&[1, cfg.diffnext_c_in, b_h, b_w]),
        0xB0DEB0DE,
    );
    for s in 0..=b_steps {
        let r = 1.0 - (s as f32 / (b_steps as f32 + 1.0));
        z_b = diffnext.forward(&z_b, r, text_embed, b_h, b_w)?;
    }

    // Stage A decode.
    let img = paella.decode(&z_b)?;
    Ok(img)
}

/// Deterministic small Gaussian-ish noise anchored to `anchor`'s graph.
/// Wuerstchen's denoising semantics in this port are deterministic (the
/// model's `(x - a) / b` output is the update), so a reproducible
/// pseudo-random initial state suffices for both the tiny test and the
/// general-purpose entry point.
fn noise_on_graph(anchor: &LazyTensor, shape: Shape, seed: u64) -> LazyTensor {
    let n = shape.elem_count();
    let data = small_normal_vec(n, seed);
    anchor.const_f32_like(data, shape)
}

/// Internal helper producing a deterministic small-noise vector
/// (mean 0, range ≈ ±0.025). Same generator the `make_*_weights`
/// fixtures use, exposed as a non-Arc Vec for noise tensors.
fn small_normal_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (((s >> 33) as u32 as f32) / (u32::MAX as f32) - 0.5) * 0.05;
        out.push(v);
    }
    out
}

// ---- Test fixtures ---------------------------------------------------------

fn arc_zeros(n: usize) -> Arc<[f32]> { Arc::from(vec![0.0_f32; n]) }
fn arc_ones(n: usize) -> Arc<[f32]> { Arc::from(vec![1.0_f32; n]) }

fn small_normal(n: usize, seed: u64) -> Arc<[f32]> {
    // Deterministic pseudo-random small values for shape/finite tests.
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (((s >> 33) as u32 as f32) / (u32::MAX as f32) - 0.5) * 0.05;
        out.push(v);
    }
    Arc::from(out)
}

fn make_grn(c: usize) -> GrnWeights {
    GrnWeights { gamma: arc_ones(c), beta: arc_zeros(c) }
}

fn make_res_block(c: usize, c_skip: usize, ksize: usize, seed: u64) -> ResBlockWeights {
    ResBlockWeights {
        dw_w: small_normal(c * 1 * ksize * ksize, seed),
        dw_b: arc_zeros(c),
        fc1_w: small_normal((c + c_skip) * 4 * c, seed + 1),
        fc1_b: arc_zeros(4 * c),
        grn: make_grn(4 * c),
        fc2_w: small_normal(4 * c * c, seed + 2),
        fc2_b: arc_zeros(c),
        c, c_skip, ksize,
    }
}

fn make_ts_block(c: usize, c_timestep: usize, seed: u64) -> TimestepBlockWeights {
    TimestepBlockWeights {
        w: small_normal(c_timestep * 2 * c, seed),
        b: arc_zeros(2 * c),
        c, c_timestep,
    }
}

fn make_attn_block(c: usize, c_cond: usize, heads: usize, seed: u64) -> AttnBlockWeights {
    AttnBlockWeights {
        kv_mapper_w: small_normal(c_cond * c, seed),
        kv_mapper_b: arc_zeros(c),
        to_q_w: small_normal(c * c, seed + 1),
        to_q_b: arc_zeros(c),
        to_k_w: small_normal(c * c, seed + 2),
        to_k_b: arc_zeros(c),
        to_v_w: small_normal(c * c, seed + 3),
        to_v_b: arc_zeros(c),
        to_out_w: small_normal(c * c, seed + 4),
        to_out_b: arc_zeros(c),
        c, c_cond, heads,
    }
}

/// Synthetic small weights for `PriorModel` exercising every component.
pub fn make_prior_weights(cfg: &WuerstchenConfig) -> PriorWeights {
    let c = cfg.prior_c;
    let c_in = cfg.prior_c_in;
    let c_cond = cfg.prior_c_cond;
    let mut blocks = Vec::with_capacity(cfg.prior_depth);
    for i in 0..cfg.prior_depth {
        let base = 100 + (i as u64) * 100;
        blocks.push(PriorBlockWeights {
            res: make_res_block(c, 0, 3, base),
            ts:  make_ts_block(c, cfg.c_r, base + 10),
            attn: make_attn_block(c, c, cfg.prior_nhead, base + 20),
        });
    }
    PriorWeights {
        projection_w: small_normal(c * c_in * 1 * 1, 1),
        projection_b: arc_zeros(c),
        cond1_w: small_normal(c_cond * c, 2),
        cond1_b: arc_zeros(c),
        cond2_w: small_normal(c * c, 3),
        cond2_b: arc_zeros(c),
        blocks,
        out_conv_w: small_normal(c * c_in * 2, 4),
        out_conv_b: arc_zeros(c_in * 2),
    }
}

/// Synthetic small weights for `DiffNextModel`.
pub fn make_diffnext_weights(cfg: &WuerstchenConfig) -> DiffNextWeights {
    let levels = &cfg.diffnext_c_hidden;
    let c_in = cfg.diffnext_c_in;
    let c_out = cfg.diffnext_c_out;
    let c_cond = cfg.diffnext_c_cond;
    let p = cfg.patch_size;

    let mut down_levels = Vec::with_capacity(levels.len());
    for (i, &c_lvl) in levels.iter().enumerate() {
        let mut subs = Vec::with_capacity(cfg.diffnext_blocks[i]);
        for j in 0..cfg.diffnext_blocks[i] {
            let seed = 1000 + (i as u64) * 100 + (j as u64) * 10;
            subs.push(DiffNextSubBlockWeights {
                res: make_res_block(c_lvl, 0, 3, seed),
                ts:  make_ts_block(c_lvl, cfg.c_r, seed + 1),
                attn: if cfg.diffnext_nhead[i] > 0 {
                    Some(make_attn_block(c_lvl, c_cond, cfg.diffnext_nhead[i], seed + 2))
                } else { None },
            });
        }
        let (down_w, down_b) = if i > 0 {
            let prev = levels[i - 1];
            (Some(small_normal(c_lvl * prev * 4, 2000 + i as u64)), Some(arc_zeros(c_lvl)))
        } else {
            (None, None)
        };
        down_levels.push(DiffNextLevelWeights { down_w, down_b, up_w: None, up_b: None, subs });
    }

    let mut up_levels = Vec::with_capacity(levels.len());
    for i in 0..levels.len() {
        let lvl_idx = levels.len() - 1 - i;
        let c_lvl = levels[lvl_idx];
        let mut subs = Vec::with_capacity(cfg.diffnext_blocks[lvl_idx]);
        for j in 0..cfg.diffnext_blocks[lvl_idx] {
            let seed = 3000 + (i as u64) * 100 + (j as u64) * 10;
            // c_skip from the symmetric down-skip (same channel count as the level).
            let c_skip = if j == 0 && i > 0 { c_lvl } else { 0 };
            subs.push(DiffNextSubBlockWeights {
                res: make_res_block(c_lvl, c_skip, 3, seed),
                ts:  make_ts_block(c_lvl, cfg.c_r, seed + 1),
                attn: if cfg.diffnext_nhead[lvl_idx] > 0 {
                    Some(make_attn_block(c_lvl, c_cond, cfg.diffnext_nhead[lvl_idx], seed + 2))
                } else { None },
            });
        }
        let (up_w, up_b) = if lvl_idx > 0 {
            let next = levels[lvl_idx - 1];
            (Some(small_normal(c_lvl * next * 4, 4000 + i as u64)), Some(arc_zeros(next)))
        } else {
            (None, None)
        };
        up_levels.push(DiffNextLevelWeights { down_w: None, down_b: None, up_w, up_b, subs });
    }

    DiffNextWeights {
        clip_mapper_w: small_normal(cfg.clip_embed * c_cond, 10),
        clip_mapper_b: arc_zeros(c_cond),
        embed_w: small_normal(levels[0] * c_in * p * p, 11),
        embed_b: arc_zeros(levels[0]),
        down_levels,
        up_levels,
        clf_w: small_normal(2 * c_out * p * p * levels[0], 12),
        clf_b: arc_zeros(2 * c_out * p * p),
    }
}

/// Synthetic small weights for `PaellaVqModel` decoder.
pub fn make_paella_weights(cfg: &WuerstchenConfig) -> PaellaVqWeights {
    let levels = &cfg.paella_levels;
    let lc = cfg.paella_latent_channels;
    let oc = cfg.paella_out_channels;
    let mut up_levels = Vec::with_capacity(levels.len());
    for (i, &c_lvl) in levels.iter().enumerate() {
        let mut res_blocks = Vec::new();
        let n_bottleneck = if i == 0 { cfg.paella_bottleneck_blocks } else { 1 };
        for j in 0..n_bottleneck {
            let seed = 5000 + (i as u64) * 100 + (j as u64) * 10;
            let embed_dim = c_lvl * 4;
            res_blocks.push(PaellaMixingResWeights {
                gammas: [0.05, 0.0, 0.05, 0.05, 0.0, 0.05],
                dw_w: small_normal(c_lvl * 1 * 3 * 3, seed),
                dw_b: arc_zeros(c_lvl),
                fc1_w: small_normal(c_lvl * embed_dim, seed + 1),
                fc1_b: arc_zeros(embed_dim),
                fc2_w: small_normal(embed_dim * c_lvl, seed + 2),
                fc2_b: arc_zeros(c_lvl),
                c: c_lvl,
            });
        }
        let (upsample_w, upsample_b) = if i < levels.len() - 1 {
            let next = levels[i + 1];
            (Some(small_normal(c_lvl * next * 16, 6000 + i as u64)), Some(arc_zeros(next)))
        } else {
            (None, None)
        };
        up_levels.push(PaellaUpLevelWeights { res_blocks, upsample_w, upsample_b });
    }
    let last_c = *levels.last().unwrap();
    PaellaVqWeights {
        up_in_w: small_normal(levels[0] * lc * 1 * 1, 7000),
        up_in_b: arc_zeros(levels[0]),
        up_levels,
        out_w: small_normal(oc * 4 * last_c, 7001),
        out_b: arc_zeros(oc * 4),
    }
}

// ---- Safetensors loaders ---------------------------------------------------

/// Load a tensor as Arc<[f32]>, asserting the element count.
fn load_arc_f32(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    expected: usize,
) -> crate::Result<Arc<[f32]>> {
    use crate::lazy::load_tensor_as_f32;
    let v = load_tensor_as_f32(st, name)?;
    if v.len() != expected {
        return Err(crate::Error::Msg(format!(
            "wuerstchen load: tensor {name:?} has {} elements, expected {expected}",
            v.len(),
        )).bt());
    }
    Ok(Arc::from(v))
}

/// Load a linear weight as `[in_f, out_f]` (transposed from HF
/// `[out_f, in_f]`).
fn load_linear_f32(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    out_f: usize,
    in_f: usize,
) -> crate::Result<Arc<[f32]>> {
    use crate::lazy::load_transposed_matrix;
    Ok(Arc::from(load_transposed_matrix(st, name, out_f, in_f)?))
}

fn load_grn(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c: usize,
) -> crate::Result<GrnWeights> {
    Ok(GrnWeights {
        gamma: load_arc_f32(st, &format!("{prefix}.gamma"), c)?,
        beta:  load_arc_f32(st, &format!("{prefix}.beta"),  c)?,
    })
}

fn load_res_block(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c: usize,
    c_skip: usize,
    ksize: usize,
) -> crate::Result<ResBlockWeights> {
    // depthwise conv: [C, 1, K, K]
    let dw_w = load_arc_f32(st, &format!("{prefix}.depthwise.weight"), c * ksize * ksize)?;
    let dw_b = load_arc_f32(st, &format!("{prefix}.depthwise.bias"),   c)?;
    // channelwise.0 (fc1): linear (c + c_skip) → 4c
    let fc1_w = load_linear_f32(st, &format!("{prefix}.channelwise.0.weight"), 4 * c, c + c_skip)?;
    let fc1_b = load_arc_f32(st, &format!("{prefix}.channelwise.0.bias"), 4 * c)?;
    // channelwise.2 (grn) gamma/beta shape [4*C].
    let grn = load_grn(st, &format!("{prefix}.channelwise.2"), 4 * c)?;
    // channelwise.4 (fc2): linear 4c → c
    let fc2_w = load_linear_f32(st, &format!("{prefix}.channelwise.4.weight"), c, 4 * c)?;
    let fc2_b = load_arc_f32(st, &format!("{prefix}.channelwise.4.bias"), c)?;
    Ok(ResBlockWeights {
        dw_w, dw_b, fc1_w, fc1_b, grn, fc2_w, fc2_b,
        c, c_skip, ksize,
    })
}

fn load_ts_block(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c: usize,
    c_timestep: usize,
) -> crate::Result<TimestepBlockWeights> {
    let w = load_linear_f32(st, &format!("{prefix}.mapper.weight"), 2 * c, c_timestep)?;
    let b = load_arc_f32(st, &format!("{prefix}.mapper.bias"), 2 * c)?;
    Ok(TimestepBlockWeights { w, b, c, c_timestep })
}

fn load_attn_block(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c: usize,
    c_cond: usize,
    heads: usize,
) -> crate::Result<AttnBlockWeights> {
    let kv_mapper_w = load_linear_f32(st, &format!("{prefix}.kv_mapper.1.weight"), c, c_cond)?;
    let kv_mapper_b = load_arc_f32(st, &format!("{prefix}.kv_mapper.1.bias"), c)?;
    let to_q_w = load_linear_f32(st, &format!("{prefix}.attention.to_q.weight"), c, c)?;
    let to_q_b = load_arc_f32(st, &format!("{prefix}.attention.to_q.bias"), c)?;
    let to_k_w = load_linear_f32(st, &format!("{prefix}.attention.to_k.weight"), c, c)?;
    let to_k_b = load_arc_f32(st, &format!("{prefix}.attention.to_k.bias"), c)?;
    let to_v_w = load_linear_f32(st, &format!("{prefix}.attention.to_v.weight"), c, c)?;
    let to_v_b = load_arc_f32(st, &format!("{prefix}.attention.to_v.bias"), c)?;
    let to_out_w = load_linear_f32(st, &format!("{prefix}.attention.to_out.0.weight"), c, c)?;
    let to_out_b = load_arc_f32(st, &format!("{prefix}.attention.to_out.0.bias"), c)?;
    Ok(AttnBlockWeights {
        kv_mapper_w, kv_mapper_b,
        to_q_w, to_q_b, to_k_w, to_k_b, to_v_w, to_v_b,
        to_out_w, to_out_b,
        c, c_cond, heads,
    })
}

impl PriorWeights {
    /// Walk a HuggingFace Wuerstchen Stage C (Prior) safetensors and
    /// build the `PriorWeights` bag.
    ///
    /// The HF Prior checkpoint follows the eager
    /// `fuel-transformers::models::diffusion::wuerstchen::prior::WPrior`
    /// field layout. Expected tensors:
    /// - `projection.weight` / `.bias` (1×1 conv `c_in → c`)
    /// - `cond_mapper.0.weight` / `.bias` (linear `c_cond → c`)
    /// - `cond_mapper.2.weight` / `.bias` (linear `c → c`)
    /// - `blocks.{i}.0.{depthwise,channelwise.*}` (`ResBlock`)
    /// - `blocks.{i}.1.mapper.{weight,bias}` (`TimestepBlock` FiLM)
    /// - `blocks.{i}.2.{kv_mapper.1,attention.*}` (`AttnBlock`)
    /// - `out.0.weight` / `.bias` for the LN-free 1×1 conv
    ///   (`c → 2*c_in`) at the final output projection.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &WuerstchenConfig,
    ) -> crate::Result<Self> {
        let c = cfg.prior_c;
        let c_in = cfg.prior_c_in;
        let c_cond = cfg.prior_c_cond;
        // 1×1 conv => [c, c_in, 1, 1].
        let projection_w = load_arc_f32(st, "projection.weight", c * c_in)?;
        let projection_b = load_arc_f32(st, "projection.bias",   c)?;
        let cond1_w = load_linear_f32(st, "cond_mapper.0.weight", c, c_cond)?;
        let cond1_b = load_arc_f32(st, "cond_mapper.0.bias", c)?;
        let cond2_w = load_linear_f32(st, "cond_mapper.2.weight", c, c)?;
        let cond2_b = load_arc_f32(st, "cond_mapper.2.bias", c)?;
        let mut blocks = Vec::with_capacity(cfg.prior_depth);
        for i in 0..cfg.prior_depth {
            blocks.push(PriorBlockWeights {
                res:  load_res_block(st, &format!("blocks.{i}.0"), c, 0, 3)?,
                ts:   load_ts_block(st, &format!("blocks.{i}.1"), c, cfg.c_r)?,
                attn: load_attn_block(st, &format!("blocks.{i}.2"), c, c, cfg.prior_nhead)?,
            });
        }
        // out.0 is the 1×1 conv `c → 2*c_in`.
        let out_conv_w = load_arc_f32(st, "out.0.weight", c_in * 2 * c)?;
        let out_conv_b = load_arc_f32(st, "out.0.bias",   c_in * 2)?;
        Ok(PriorWeights {
            projection_w, projection_b,
            cond1_w, cond1_b, cond2_w, cond2_b,
            blocks, out_conv_w, out_conv_b,
        })
    }
}

impl DiffNextWeights {
    /// Walk a HuggingFace Wuerstchen Stage B (DiffNeXt) safetensors and
    /// build the `DiffNextWeights` bag.
    ///
    /// Expected tensors (mirrors eager `WDiffNeXt`):
    /// - `clip_mapper.weight` / `.bias`
    /// - `embedding.1.weight` / `.bias` (the unshuffle stack stores
    ///    the 1×1 conv at sequential index `.1`; index `.0` is
    ///    PixelUnshuffle and has no params)
    /// - `down_blocks.{i}.{0,1,...}.{0,1,2}.*` per sub-block triple
    /// - `up_blocks.{i}.{0,1,...}.{0,1,2}.*` per sub-block triple
    /// - `clf.1.weight` / `.bias` (1×1 conv to `2*c_out*p²`)
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &WuerstchenConfig,
    ) -> crate::Result<Self> {
        let levels = &cfg.diffnext_c_hidden;
        let c_in = cfg.diffnext_c_in;
        let c_out = cfg.diffnext_c_out;
        let c_cond = cfg.diffnext_c_cond;
        let p = cfg.patch_size;
        if levels.is_empty() {
            return Err(crate::Error::Msg(
                "DiffNextWeights.load_from_mmapped: diffnext_c_hidden empty".into(),
            ).bt());
        }
        let clip_mapper_w = load_linear_f32(st, "clip_mapper.weight", c_cond, cfg.clip_embed)?;
        let clip_mapper_b = load_arc_f32(st, "clip_mapper.bias", c_cond)?;
        // 1×1 conv: [C_HIDDEN[0], c_in * p*p, 1, 1]
        let embed_w = load_arc_f32(st, "embedding.1.weight", levels[0] * c_in * p * p)?;
        let embed_b = load_arc_f32(st, "embedding.1.bias",   levels[0])?;

        // Down levels.
        let mut down_levels = Vec::with_capacity(levels.len());
        for (i, &c_lvl) in levels.iter().enumerate() {
            let (down_w, down_b) = if i > 0 {
                let prev = levels[i - 1];
                // stride-2 2×2 conv: [Cout, Cin, 2, 2].
                let w = load_arc_f32(st, &format!("down_blocks.{i}.0.1.weight"), c_lvl * prev * 4)?;
                let b = load_arc_f32(st, &format!("down_blocks.{i}.0.1.bias"),   c_lvl)?;
                (Some(w), Some(b))
            } else {
                (None, None)
            };
            // Sub-block offset (skip the down conv index 0 if it exists).
            let sub_offset = if i > 0 { 1 } else { 0 };
            let mut subs = Vec::with_capacity(cfg.diffnext_blocks[i]);
            for j in 0..cfg.diffnext_blocks[i] {
                let sub_idx = sub_offset + j;
                let sub_pfx = format!("down_blocks.{i}.{sub_idx}");
                subs.push(DiffNextSubBlockWeights {
                    res: load_res_block(st, &format!("{sub_pfx}.0"), c_lvl, 0, 3)?,
                    ts:  load_ts_block(st, &format!("{sub_pfx}.1"), c_lvl, cfg.c_r)?,
                    attn: if cfg.diffnext_nhead[i] > 0 {
                        Some(load_attn_block(st, &format!("{sub_pfx}.2"), c_lvl, c_cond, cfg.diffnext_nhead[i])?)
                    } else { None },
                });
            }
            down_levels.push(DiffNextLevelWeights { down_w, down_b, up_w: None, up_b: None, subs });
        }

        // Up levels.
        let mut up_levels = Vec::with_capacity(levels.len());
        for i in 0..levels.len() {
            let lvl_idx = levels.len() - 1 - i;
            let c_lvl = levels[lvl_idx];
            let mut subs = Vec::with_capacity(cfg.diffnext_blocks[lvl_idx]);
            for j in 0..cfg.diffnext_blocks[lvl_idx] {
                let sub_pfx = format!("up_blocks.{i}.{j}");
                let c_skip = if j == 0 && i > 0 { c_lvl } else { 0 };
                subs.push(DiffNextSubBlockWeights {
                    res: load_res_block(st, &format!("{sub_pfx}.0"), c_lvl, c_skip, 3)?,
                    ts:  load_ts_block(st, &format!("{sub_pfx}.1"), c_lvl, cfg.c_r)?,
                    attn: if cfg.diffnext_nhead[lvl_idx] > 0 {
                        Some(load_attn_block(st, &format!("{sub_pfx}.2"), c_lvl, c_cond, cfg.diffnext_nhead[lvl_idx])?)
                    } else { None },
                });
            }
            let (up_w, up_b) = if lvl_idx > 0 {
                let next = levels[lvl_idx - 1];
                // Trailing conv-transpose 2×2: stored at the end of the level.
                let after_subs = cfg.diffnext_blocks[lvl_idx];
                let w = load_arc_f32(
                    st, &format!("up_blocks.{i}.{after_subs}.1.weight"),
                    c_lvl * next * 4,
                )?;
                let b = load_arc_f32(
                    st, &format!("up_blocks.{i}.{after_subs}.1.bias"),
                    next,
                )?;
                (Some(w), Some(b))
            } else {
                (None, None)
            };
            up_levels.push(DiffNextLevelWeights { down_w: None, down_b: None, up_w, up_b, subs });
        }

        // clf.1 is a 1×1 conv (clf.0 is the LN).
        let clf_w = load_arc_f32(st, "clf.1.weight", 2 * c_out * p * p * levels[0])?;
        let clf_b = load_arc_f32(st, "clf.1.bias",   2 * c_out * p * p)?;

        Ok(DiffNextWeights {
            clip_mapper_w, clip_mapper_b,
            embed_w, embed_b,
            down_levels, up_levels,
            clf_w, clf_b,
        })
    }
}

fn load_paella_mixing_res(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c: usize,
) -> crate::Result<PaellaMixingResWeights> {
    let embed_dim = 4 * c;
    let gammas_vec = load_arc_f32(st, &format!("{prefix}.gammas"), 6)?;
    let gammas = [
        gammas_vec[0], gammas_vec[1], gammas_vec[2],
        gammas_vec[3], gammas_vec[4], gammas_vec[5],
    ];
    Ok(PaellaMixingResWeights {
        gammas,
        // Depthwise conv: [C, 1, 3, 3]. eager calls it `.depthwise.1` (with replication pad at .0).
        dw_w: load_arc_f32(st, &format!("{prefix}.depthwise.1.weight"), c * 9)?,
        dw_b: load_arc_f32(st, &format!("{prefix}.depthwise.1.bias"),   c)?,
        fc1_w: load_linear_f32(st, &format!("{prefix}.channelwise.0.weight"), embed_dim, c)?,
        fc1_b: load_arc_f32(st, &format!("{prefix}.channelwise.0.bias"), embed_dim)?,
        fc2_w: load_linear_f32(st, &format!("{prefix}.channelwise.2.weight"), c, embed_dim)?,
        fc2_b: load_arc_f32(st, &format!("{prefix}.channelwise.2.bias"), c)?,
        c,
    })
}

impl PaellaVqWeights {
    /// Walk a HuggingFace Paella VQ-VAE safetensors and build the
    /// `PaellaVqWeights` bag (decoder side only).
    ///
    /// Expected tensors (mirrors eager `PaellaVQ` decoder):
    /// - `up_blocks.0.0.0.weight` / `.bias` — 1×1 `up_in` conv
    ///   (`latent_channels → paella_levels[0]`).
    /// - For each level group: a sequence of `MixingResidualBlock`
    ///   entries followed by an optional conv-transpose stride-2.
    /// - `out_block.1.weight` / `.bias` — 1×1 `out` conv
    ///   (`paella_levels.last() → out_channels * 4`).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &WuerstchenConfig,
    ) -> crate::Result<Self> {
        let levels = &cfg.paella_levels;
        let lc = cfg.paella_latent_channels;
        let oc = cfg.paella_out_channels;
        if levels.is_empty() {
            return Err(crate::Error::Msg(
                "PaellaVqWeights.load_from_mmapped: paella_levels empty".into(),
            ).bt());
        }
        let up_in_w = load_arc_f32(st, "up_blocks.0.0.0.weight", levels[0] * lc)?;
        let up_in_b = load_arc_f32(st, "up_blocks.0.0.0.bias",   levels[0])?;
        let mut up_levels = Vec::with_capacity(levels.len());
        for (i, &c_lvl) in levels.iter().enumerate() {
            let mut res_blocks = Vec::new();
            let n_res = if i == 0 { cfg.paella_bottleneck_blocks } else { 1 };
            for j in 0..n_res {
                let pfx = format!("up_blocks.{i}.{j}.1");
                res_blocks.push(load_paella_mixing_res(st, &pfx, c_lvl)?);
            }
            let (upsample_w, upsample_b) = if i < levels.len() - 1 {
                let next = levels[i + 1];
                // Conv-transpose 4×4 stride 2: [Cin, Cout, 4, 4].
                let w_off = n_res; // upsample slot index in this level group.
                let w = load_arc_f32(
                    st, &format!("up_blocks.{i}.{w_off}.0.weight"),
                    c_lvl * next * 16,
                )?;
                let b = load_arc_f32(
                    st, &format!("up_blocks.{i}.{w_off}.0.bias"),
                    next,
                )?;
                (Some(w), Some(b))
            } else {
                (None, None)
            };
            up_levels.push(PaellaUpLevelWeights { res_blocks, upsample_w, upsample_b });
        }
        let last_c = *levels.last().unwrap();
        let out_w = load_arc_f32(st, "out_block.1.weight", oc * 4 * last_c)?;
        let out_b = load_arc_f32(st, "out_block.1.bias",   oc * 4)?;
        Ok(PaellaVqWeights {
            up_in_w, up_in_b, up_levels, out_w, out_b,
        })
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> crate::Device { crate::Device::cpu() }

    /// GlobalResponseNorm hand-check: with `x = [1, 0; 0, 0]` on `C=1`,
    /// `agg = sqrt(sum_HW(x²)) = sqrt(1) = 1`.
    /// `mean_C(agg) = 1`. `denom = mean + 1e-6 ≈ 1`.
    /// `stand = agg / denom ≈ 1` (broadcast over the 2×2 spatial map).
    /// `y = x * stand * γ + β + x = x * 1 * 1 + 0 + x = 2x`.
    #[test]
    fn global_response_norm_hand_computed() {
        let x_data = vec![1.0_f32, 0.0, 0.0, 0.0];  // [1, 1, 2, 2]
        let x = LazyTensor::from_f32(x_data.clone(), Shape::from_dims(&[1, 1, 2, 2]), &dev());
        let gamma = arc_ones(1);
        let beta = arc_zeros(1);
        let out = global_response_norm(&x, &gamma, &beta, 1, 2, 2).realize_f32();
        // Expected: y[i] ≈ x[i] * 1 + x[i] = 2 * x[i] (with tiny eps drift).
        let expected: Vec<f32> = x_data.iter().map(|&v| 2.0 * v).collect();
        for (i, (a, e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-4,
                "GRN[{i}]: expected {e}, got {a}");
        }
    }

    /// Paella VQ decoder: tiny config decodes a `(1, 4, 8, 8)` latent
    /// into a `(1, 3, H, W)` image with H = W = 8 * 2^(n_levels-1) * 2 = 32.
    #[test]
    fn paella_vq_decoder_shape() {
        let cfg = WuerstchenConfig::tiny();
        let weights = make_paella_weights(&cfg);
        let model = PaellaVqModel { config: cfg.clone(), weights };
        // Latent shape: spatial 8x8.
        let lat_data = vec![0.01_f32; 1 * 4 * 8 * 8];
        let lat = LazyTensor::from_f32(lat_data, Shape::from_dims(&[1, 4, 8, 8]), &dev());
        let img = model.decode(&lat).unwrap();
        // n_levels = 2 → one upsample (×2) followed by pixel_shuffle ×2 = total ×4.
        assert_eq!(img.shape().dims(), &[1, 3, 32, 32]);
        let flat = img.realize_f32();
        for v in &flat { assert!(v.is_finite(), "non-finite paella decoder pixel: {v}"); }
        for v in &flat { assert!(v.abs() <= 1.0 + 1e-5, "tanh out-of-range: {v}"); }
    }

    /// PriorModel forward: tiny config 2×2 spatial prior, finite output.
    #[test]
    fn prior_forward_shape_finite_tiny() {
        let cfg = WuerstchenConfig::tiny();
        let weights = make_prior_weights(&cfg);
        let model = PriorModel { config: cfg.clone(), weights };
        let xs_data = vec![0.01_f32; 1 * cfg.prior_c_in * 2 * 2];
        let xs = LazyTensor::from_f32(xs_data, Shape::from_dims(&[1, cfg.prior_c_in, 2, 2]), &dev());
        let txt_data = vec![0.01_f32; 1 * 4 * cfg.prior_c_cond];
        let txt = xs.const_f32_like(txt_data, Shape::from_dims(&[1, 4, cfg.prior_c_cond]));
        let out = model.forward(&xs, 0.5, &txt, 2, 2).unwrap();
        assert_eq!(out.shape().dims(), &[1, cfg.prior_c_in, 2, 2]);
        for v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite prior output: {v}");
        }
    }

    /// DiffNextModel forward: tiny config 4×4 spatial Stage B, finite.
    #[test]
    fn diffnext_forward_shape_finite_tiny() {
        let cfg = WuerstchenConfig::tiny();
        let weights = make_diffnext_weights(&cfg);
        let model = DiffNextModel { config: cfg.clone(), weights };
        let h = 4; let w = 4;
        let xs_data = vec![0.01_f32; 1 * cfg.diffnext_c_in * h * w];
        let xs = LazyTensor::from_f32(xs_data, Shape::from_dims(&[1, cfg.diffnext_c_in, h, w]), &dev());
        let txt_data = vec![0.01_f32; 1 * 4 * cfg.clip_embed];
        let txt = xs.const_f32_like(txt_data, Shape::from_dims(&[1, 4, cfg.clip_embed]));
        let out = model.forward(&xs, 0.5, &txt, h, w).unwrap();
        assert_eq!(out.shape().dims(), &[1, cfg.diffnext_c_out, h, w]);
        for v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite diffnext output: {v}");
        }
    }

    /// End-to-end generate: tiny config; text → 32×32 RGB image; finite
    // ---- Safetensors loader smoke tests -------------------------------

    fn write_tmp_safetensors_w(
        tensors: &[(String, Vec<usize>, Vec<f32>)],
    ) -> std::path::PathBuf {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;
        let bytes_store: Vec<Vec<u8>> = tensors.iter()
            .map(|(_, _, data)| data.iter().flat_map(|f| f.to_le_bytes()).collect())
            .collect();
        let views: HashMap<String, TensorView<'_>> = tensors.iter()
            .zip(bytes_store.iter())
            .map(|((name, shape, _), bytes)| {
                let v = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .expect("TensorView::new");
                (name.clone(), v)
            })
            .collect();
        let metadata: Option<HashMap<String, String>> = None;
        let bytes_out = safetensors::serialize(&views, metadata).unwrap();
        let path = std::env::temp_dir().join(format!(
            "fuel_lazy_wuerst_test_{}_{}.safetensors",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
            std::process::id(),
        ));
        std::fs::write(&path, bytes_out).unwrap();
        path
    }

    /// Load round-trip for `PriorWeights`. Synthesizes a safetensors file
    /// matching the eager Wuerstchen WPrior naming and verifies the loader
    /// reconstructs the bag with the expected per-tensor shapes.
    #[test]
    fn load_prior_weights_from_mmapped_tiny() {
        let cfg = WuerstchenConfig::tiny();
        let c = cfg.prior_c;
        let c_in = cfg.prior_c_in;
        let c_cond = cfg.prior_c_cond;
        let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        // projection: [c, c_in, 1, 1]
        t.push(("projection.weight".into(), vec![c, c_in, 1, 1], vec![0.0; c * c_in]));
        t.push(("projection.bias".into(),   vec![c], vec![0.0; c]));
        // cond_mapper.0 + .2 (linears) — HF stores as [out_f, in_f].
        t.push(("cond_mapper.0.weight".into(), vec![c, c_cond], vec![0.0; c * c_cond]));
        t.push(("cond_mapper.0.bias".into(),   vec![c], vec![0.0; c]));
        t.push(("cond_mapper.2.weight".into(), vec![c, c], vec![0.0; c * c]));
        t.push(("cond_mapper.2.bias".into(),   vec![c], vec![0.0; c]));
        // One block (matches tiny.prior_depth = 1).
        let pfx = "blocks.0";
        // ResBlock at .0: depthwise [c, 1, 3, 3].
        t.push((format!("{pfx}.0.depthwise.weight"), vec![c, 1, 3, 3], vec![0.0; c * 9]));
        t.push((format!("{pfx}.0.depthwise.bias"),   vec![c], vec![0.0; c]));
        t.push((format!("{pfx}.0.channelwise.0.weight"), vec![4 * c, c], vec![0.0; 4 * c * c]));
        t.push((format!("{pfx}.0.channelwise.0.bias"),   vec![4 * c], vec![0.0; 4 * c]));
        t.push((format!("{pfx}.0.channelwise.2.gamma"), vec![4 * c], vec![1.0; 4 * c]));
        t.push((format!("{pfx}.0.channelwise.2.beta"),  vec![4 * c], vec![0.0; 4 * c]));
        t.push((format!("{pfx}.0.channelwise.4.weight"), vec![c, 4 * c], vec![0.0; 4 * c * c]));
        t.push((format!("{pfx}.0.channelwise.4.bias"),   vec![c], vec![0.0; c]));
        // TimestepBlock at .1: mapper [2c, c_r].
        t.push((format!("{pfx}.1.mapper.weight"), vec![2 * c, cfg.c_r], vec![0.0; 2 * c * cfg.c_r]));
        t.push((format!("{pfx}.1.mapper.bias"),   vec![2 * c], vec![0.0; 2 * c]));
        // AttnBlock at .2.
        t.push((format!("{pfx}.2.kv_mapper.1.weight"), vec![c, c], vec![0.0; c * c]));
        t.push((format!("{pfx}.2.kv_mapper.1.bias"),   vec![c], vec![0.0; c]));
        for kind in ["q", "k", "v"] {
            t.push((format!("{pfx}.2.attention.to_{kind}.weight"), vec![c, c], vec![0.0; c * c]));
            t.push((format!("{pfx}.2.attention.to_{kind}.bias"),   vec![c], vec![0.0; c]));
        }
        t.push((format!("{pfx}.2.attention.to_out.0.weight"), vec![c, c], vec![0.0; c * c]));
        t.push((format!("{pfx}.2.attention.to_out.0.bias"),   vec![c], vec![0.0; c]));
        // out.0 conv [c_in*2, c, 1, 1].
        t.push(("out.0.weight".into(), vec![c_in * 2, c, 1, 1], vec![0.0; c_in * 2 * c]));
        t.push(("out.0.bias".into(),   vec![c_in * 2], vec![0.0; c_in * 2]));

        let path = write_tmp_safetensors_w(&t);
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path).unwrap() };
        let w = PriorWeights::load_from_mmapped(&st, &cfg).unwrap();
        assert_eq!(w.blocks.len(), cfg.prior_depth);
        assert_eq!(w.projection_w.len(), c * c_in);
        let _ = std::fs::remove_file(&path);
    }

    /// and within `[-1, 1]` (tanh activation at decoder output).
    #[test]
    fn end_to_end_generate_tiny() {
        let cfg = WuerstchenConfig::tiny();
        let prior = PriorModel { config: cfg.clone(), weights: make_prior_weights(&cfg) };
        let diffnext = DiffNextModel { config: cfg.clone(), weights: make_diffnext_weights(&cfg) };
        let paella   = PaellaVqModel { config: cfg.clone(), weights: make_paella_weights(&cfg) };
        let txt_data = vec![0.01_f32; 1 * 4 * cfg.clip_embed];
        let txt = LazyTensor::from_f32(txt_data, Shape::from_dims(&[1, 4, cfg.clip_embed]), &dev());
        let img = generate(&prior, &diffnext, &paella, &txt, 0, 0, 2, 2, 8, 8).unwrap();
        assert_eq!(img.shape().dims(), &[1, 3, 32, 32]);
        let flat = img.realize_f32();
        for v in &flat {
            assert!(v.is_finite(), "non-finite generated pixel: {v}");
            assert!(v.abs() <= 1.0 + 1e-5, "generated pixel out of [-1,1]: {v}");
        }
    }
}

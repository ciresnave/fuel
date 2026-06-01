//! Stable Diffusion 1.5 UNet (denoising network) ported to the
//! lazy-graph API.
//!
//! Third component of Phase 6a anchor #6. The UNet is where almost
//! all of SD 1.5's ~860M parameters live — the other two components
//! (CLIP text encoder at ~125M, VAE decoder at ~50M) are small in
//! comparison. At inference this network runs 20-50 times per
//! image, each pass conditioned on:
//!
//! - The current noisy latent `[1, 4, H/8, W/8]`
//! - A scalar timestep embedded via sinusoidal features + a 2-layer MLP
//! - The CLIP text embedding `[1, 77, 768]` (cross-attention key/value)
//!
//! # Architectural firsts (vs prior SD 1.5 components)
//!
//! - **Sinusoidal timestep embedding** + 2-layer MLP conditioning.
//!   Every ResNet in the UNet consumes the time embedding via a
//!   per-block linear projection added mid-stack to the feature maps.
//! - **Spatial transformer blocks** (`Transformer2DModel`): reshape
//!   `[1, C, H, W]` to `[1, H·W, C]`, run N transformer blocks where
//!   each has (self-attn + cross-attn + GEGLU FFN) each pre-LN'd,
//!   reshape back. The cross-attention K/V source is the text
//!   embedding from the CLIP encoder.
//! - **GEGLU** activation: `x * gelu(gate)` with `(x, gate) =
//!   split(proj(input), dim=-1)`. Doubles the FFN's input projection
//!   width.
//! - **Strided 3×3 Conv2d** (`conv2d_k3_s2_p1`): the stride-2 case
//!   used by SD's Downsample2D. Dispatches to the native `Op::Conv2D`
//!   with `stride=(2,2)`.
//!
//! Everything else — GroupNorm, Conv2d 3×3 s=1, Conv2d 1×1,
//! 2× nearest upsample, multi-head attention — is reused from the
//! VAE decoder / Whisper decoder modules.
//!
//! # Scope / limitations
//!
//! - Forward-only; no autograd validation.
//! - No KV cache / no step-to-step reuse. Diffusion's schedule loops
//!   are outside this module's responsibility.
//! - **Performance**: every conv now routes through the native
//!   `Op::Conv2D`. The CPU backend still uses a reference 5-loop
//!   implementation, so generation remains slow but the composition
//!   overhead is gone; tiled/GPU kernels are later work.

use crate::lazy::LazyTensor;
use fuel_core_types::Shape;
use std::sync::Arc;

// ---- Config ----------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SdUnetConfig {
    pub in_channels:        usize,
    pub out_channels:       usize,
    /// Channel widths per down/up block. SD 1.5: `[320, 640, 1280, 1280]`.
    pub block_out_channels: Vec<usize>,
    /// Number of ResNet+Attention pairs per down block (and ResNet+Attention
    /// triples per up block). SD 1.5: 2.
    pub layers_per_block:   usize,
    /// Which down blocks use CrossAttnDownBlock2D (with attention) vs
    /// plain DownBlock2D (no attention). SD 1.5: `[true, true, true, false]`.
    pub down_has_attn:      Vec<bool>,
    /// Which up blocks use CrossAttnUpBlock2D vs plain UpBlock2D.
    /// SD 1.5: `[false, true, true, true]`. (Note: up blocks are
    /// indexed in execution order — 0 is the deepest/smallest.)
    pub up_has_attn:        Vec<bool>,
    pub cross_attention_dim: usize,  // 768 for SD 1.5 (CLIP-L)
    pub attention_head_dim:  usize,  // 8 for SD 1.5 (so heads = C/8 per level)
    pub time_embed_dim:      usize,  // 1280 for SD 1.5 (= 4 * block_out_channels[0])
    pub norm_num_groups:     usize,  // 32
    pub norm_eps:            f64,
}

impl SdUnetConfig {
    pub fn sd_v1() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![320, 640, 1280, 1280],
            layers_per_block: 2,
            down_has_attn: vec![true, true, true, false],
            up_has_attn:   vec![false, true, true, true],
            cross_attention_dim: 768,
            attention_head_dim: 8,
            time_embed_dim: 1280,
            norm_num_groups: 32,
            norm_eps: 1e-5,
        }
    }
}

// ---- Weight storage --------------------------------------------------------

/// ResNet block with time conditioning. `conv_shortcut` is present
/// when `c_in != c_out`.
#[derive(Debug, Clone)]
pub struct UResnetWeights {
    pub n1_g: Arc<[f32]>, pub n1_b: Arc<[f32]>,
    pub c1_w: Arc<[f32]>, pub c1_b: Arc<[f32]>,
    /// Time-embedding projection: `[c_out, time_embed_dim]`.
    pub te_w: Arc<[f32]>, pub te_b: Arc<[f32]>,
    pub n2_g: Arc<[f32]>, pub n2_b: Arc<[f32]>,
    pub c2_w: Arc<[f32]>, pub c2_b: Arc<[f32]>,
    pub shortcut_w: Option<Arc<[f32]>>,
    pub shortcut_b: Option<Arc<[f32]>>,
}

/// One transformer block inside a `Transformer2DModel`. Self-attn
/// then cross-attn then GEGLU FFN, each pre-LN'd and residual-wrapped.
#[derive(Debug, Clone)]
pub struct TransformerBlockWeights {
    // pre-LN for self-attn
    pub n1_g: Arc<[f32]>, pub n1_b: Arc<[f32]>,
    // self-attn (no biases on q/k/v, bias only on to_out.0)
    pub attn1_q_w: Arc<[f32]>,
    pub attn1_k_w: Arc<[f32]>,
    pub attn1_v_w: Arc<[f32]>,
    pub attn1_out_w: Arc<[f32]>, pub attn1_out_b: Arc<[f32]>,
    // pre-LN for cross-attn
    pub n2_g: Arc<[f32]>, pub n2_b: Arc<[f32]>,
    // cross-attn: q from x (channel dim C), k/v from text (cross_dim)
    pub attn2_q_w: Arc<[f32]>,
    pub attn2_k_w: Arc<[f32]>,  // [C, cross_dim]
    pub attn2_v_w: Arc<[f32]>,
    pub attn2_out_w: Arc<[f32]>, pub attn2_out_b: Arc<[f32]>,
    // pre-LN for FFN
    pub n3_g: Arc<[f32]>, pub n3_b: Arc<[f32]>,
    // GEGLU: `ff.net.0.proj` is `[2*4*C, C]`, `ff.net.2` is `[C, 4*C]`.
    pub ff_in_w: Arc<[f32]>, pub ff_in_b: Arc<[f32]>,
    pub ff_out_w: Arc<[f32]>, pub ff_out_b: Arc<[f32]>,
}

/// Spatial transformer (a.k.a. Transformer2DModel). Wraps N
/// transformer blocks with an entry/exit projection and a residual.
#[derive(Debug, Clone)]
pub struct SpatialTransformerWeights {
    pub norm_g: Arc<[f32]>, pub norm_b: Arc<[f32]>,
    /// 1×1 conv, `[C, C, 1, 1]`.
    pub proj_in_w: Arc<[f32]>, pub proj_in_b: Arc<[f32]>,
    pub blocks: Vec<TransformerBlockWeights>,
    pub proj_out_w: Arc<[f32]>, pub proj_out_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct DownBlockWeights {
    pub resnets: Vec<UResnetWeights>,
    /// `attentions[i]` runs after `resnets[i]`. Empty for DownBlock2D
    /// (no attention — the deepest stage of SD 1.5's UNet).
    pub attentions: Vec<SpatialTransformerWeights>,
    /// `[C, C, 3, 3]` strided conv for the downsample. None for the
    /// last down block.
    pub downsample_w: Option<Arc<[f32]>>,
    pub downsample_b: Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct UpBlockWeights {
    pub resnets: Vec<UResnetWeights>,
    pub attentions: Vec<SpatialTransformerWeights>,
    pub upsample_conv_w: Option<Arc<[f32]>>,
    pub upsample_conv_b: Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct SdUnetWeights {
    // Time embedding MLP
    pub time_mlp_1_w: Arc<[f32]>, pub time_mlp_1_b: Arc<[f32]>,
    pub time_mlp_2_w: Arc<[f32]>, pub time_mlp_2_b: Arc<[f32]>,
    // conv_in: [dim[0], in_channels, 3, 3]
    pub conv_in_w: Arc<[f32]>, pub conv_in_b: Arc<[f32]>,
    // down path
    pub down_blocks: Vec<DownBlockWeights>,
    // mid
    pub mid_resnet_1: UResnetWeights,
    pub mid_attn: SpatialTransformerWeights,
    pub mid_resnet_2: UResnetWeights,
    // up path
    pub up_blocks: Vec<UpBlockWeights>,
    // final
    pub conv_norm_out_g: Arc<[f32]>, pub conv_norm_out_b: Arc<[f32]>,
    pub conv_out_w: Arc<[f32]>, pub conv_out_b: Arc<[f32]>,
}

// ---- Model -----------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SdUnet {
    pub config:  SdUnetConfig,
    pub weights: SdUnetWeights,
}

impl SdUnet {
    /// One denoising step. Inputs:
    /// - `latent`: flat `[1, 4, h_lat, w_lat]`.
    /// - `timestep`: scalar — typically an integer in `[0, 1000)`
    ///   depending on the scheduler, but the function accepts any
    ///   float and sinusoidally embeds it.
    /// - `text_emb`: flat `[1, 77, 768]` CLIP text encoding.
    pub fn forward(
        &self,
        latent: &[f32],
        timestep: f32,
        text_emb: &[f32],
        h_lat: usize,
        w_lat: usize,
    ) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let c_in = cfg.in_channels;
        assert_eq!(latent.len(), c_in * h_lat * w_lat);
        assert_eq!(text_emb.len(), 1 * 77 * cfg.cross_attention_dim);

        // --- timestep embedding ----------------------------------
        let c_first = cfg.block_out_channels[0];
        let time_sin = sinusoidal_timestep_embedding(timestep, c_first);
        let x = LazyTensor::from_f32(latent.to_vec(), Shape::from_dims(&[1, c_in, h_lat, w_lat]), &crate::Device::cpu());
        let t_flat = x
            .const_f32_like(time_sin, Shape::from_dims(&[1, c_first]))
            .reshape(Shape::from_dims(&[1, 1, c_first])).unwrap();
        let t_emb = linear(&t_flat, &self.weights.time_mlp_1_w, Some(&self.weights.time_mlp_1_b), c_first, cfg.time_embed_dim, 1)
            .silu();
        let t_emb = linear(&t_emb, &self.weights.time_mlp_2_w, Some(&self.weights.time_mlp_2_b), cfg.time_embed_dim, cfg.time_embed_dim, 1);
        // t_emb shape: [1, 1, time_embed_dim].

        // --- text embedding (channel-last) -----------------------
        let te = x
            .const_f32_like(text_emb.to_vec(), Shape::from_dims(&[1, 77, cfg.cross_attention_dim]));

        // --- conv_in ----------------------------------------------
        let x = conv2d_k3_s1_p1(&x, &self.weights.conv_in_w, &self.weights.conv_in_b,
            c_in, c_first, h_lat, w_lat);

        // --- down path, collecting skips ------------------------
        let mut skips: Vec<LazyTensor> = vec![x.clone()];
        let mut x = x;
        let mut h = h_lat;
        let mut w = w_lat;
        for (bi, bw) in self.weights.down_blocks.iter().enumerate() {
            let c_in_block = if bi == 0 { c_first } else { cfg.block_out_channels[bi - 1] };
            let c_out_block = cfg.block_out_channels[bi];
            for ri in 0..cfg.layers_per_block {
                let in_c = if ri == 0 { c_in_block } else { c_out_block };
                x = u_resnet(&x, &bw.resnets[ri], &t_emb, cfg, in_c, c_out_block, h, w);
                if !bw.attentions.is_empty() {
                    x = spatial_transformer(&x, &bw.attentions[ri], &te, cfg, c_out_block, h, w);
                }
                skips.push(x.clone());
            }
            if let (Some(dw), Some(db)) = (&bw.downsample_w, &bw.downsample_b) {
                x = conv2d_k3_s2_p1(&x, dw, db, c_out_block, c_out_block, h, w);
                h /= 2;
                w /= 2;
                skips.push(x.clone());
            }
        }

        // --- mid block ------------------------------------------
        let c_mid = *cfg.block_out_channels.last().unwrap();
        let x = u_resnet(&x, &self.weights.mid_resnet_1, &t_emb, cfg, c_mid, c_mid, h, w);
        let x = spatial_transformer(&x, &self.weights.mid_attn, &te, cfg, c_mid, h, w);
        let x = u_resnet(&x, &self.weights.mid_resnet_2, &t_emb, cfg, c_mid, c_mid, h, w);

        // --- up path, consuming skips ---------------------------
        let mut x = x;
        for (bi, bw) in self.weights.up_blocks.iter().enumerate() {
            // Up blocks are listed in execution order (from smallest
            // resolution to largest). SD 1.5's up path has 3 ResNets per
            // block; the first consumes the deepest skip, then two more.
            let c_out_block = cfg.block_out_channels[cfg.block_out_channels.len() - 1 - bi];
            for ri in 0..(cfg.layers_per_block + 1) {
                let skip = skips.pop().expect("up: skip underflow");
                x = x.concat(&skip, 1);  // channel-axis concat
                let in_c = x.dims()[1];
                x = u_resnet(&x, &bw.resnets[ri], &t_emb, cfg, in_c, c_out_block, h, w);
                if !bw.attentions.is_empty() {
                    x = spatial_transformer(&x, &bw.attentions[ri], &te, cfg, c_out_block, h, w);
                }
            }
            if let (Some(uw), Some(ub)) = (&bw.upsample_conv_w, &bw.upsample_conv_b) {
                x = upsample_nearest_2x(&x, c_out_block, h, w);
                h *= 2;
                w *= 2;
                x = conv2d_k3_s1_p1(&x, uw, ub, c_out_block, c_out_block, h, w);
            }
        }

        // --- final norm + SiLU + conv_out -----------------------
        let x = group_norm(&x, &self.weights.conv_norm_out_g, &self.weights.conv_norm_out_b,
            cfg.norm_num_groups, cfg.norm_eps, c_first, h, w);
        let x = x.silu();
        Ok(conv2d_k3_s1_p1(&x, &self.weights.conv_out_w, &self.weights.conv_out_b,
            c_first, cfg.out_channels, h, w))
    }
}

// ---- ResNet with time conditioning -----------------------------------------

fn u_resnet(
    x: &LazyTensor,
    rw: &UResnetWeights,
    t_emb: &LazyTensor,  // [1, 1, time_embed_dim]
    cfg: &SdUnetConfig,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let h1 = group_norm(x, &rw.n1_g, &rw.n1_b, cfg.norm_num_groups, cfg.norm_eps, c_in, h, w);
    let h1 = h1.silu();
    let h1 = conv2d_k3_s1_p1(&h1, &rw.c1_w, &rw.c1_b, c_in, c_out, h, w);

    // Time projection, broadcast over spatial.
    let t = t_emb.silu();
    let t = linear(&t, &rw.te_w, Some(&rw.te_b), cfg.time_embed_dim, c_out, 1);
    // t shape: [1, 1, c_out] → broadcast to [1, c_out, h, w] via reshape.
    let t_bc = t
        .reshape(Shape::from_dims(&[1, c_out, 1, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, c_out, h, w])).unwrap();
    let h1 = h1.add(&t_bc);

    let h2 = group_norm(&h1, &rw.n2_g, &rw.n2_b, cfg.norm_num_groups, cfg.norm_eps, c_out, h, w);
    let h2 = h2.silu();
    let h2 = conv2d_k3_s1_p1(&h2, &rw.c2_w, &rw.c2_b, c_out, c_out, h, w);

    let shortcut = match (&rw.shortcut_w, &rw.shortcut_b) {
        (Some(sw), Some(sb)) => conv2d_k1_s1_p0(x, sw, sb, c_in, c_out, h, w),
        _ => x.clone(),
    };
    shortcut.add(&h2)
}

// ---- Spatial Transformer ---------------------------------------------------

fn spatial_transformer(
    x: &LazyTensor,  // [1, C, H, W]
    sw: &SpatialTransformerWeights,
    text_emb: &LazyTensor,  // [1, 77, cross_dim]
    cfg: &SdUnetConfig,
    c: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let n = h * w;
    let residual = x.clone();
    let x = group_norm(x, &sw.norm_g, &sw.norm_b, cfg.norm_num_groups, cfg.norm_eps, c, h, w);
    let x = conv2d_k1_s1_p0(&x, &sw.proj_in_w, &sw.proj_in_b, c, c, h, w);
    // Reshape [1, C, H, W] → [1, H·W, C] for the transformer blocks.
    let mut xf = x
        .permute(&[0, 2, 3, 1])
        .reshape(Shape::from_dims(&[1, n, c])).unwrap();
    for tb in &sw.blocks {
        xf = transformer_block(&xf, tb, text_emb, cfg, c, n);
    }
    // Back to [1, C, H, W].
    let out = xf
        .reshape(Shape::from_dims(&[1, h, w, c])).unwrap()
        .permute(&[0, 3, 1, 2]);
    let out = conv2d_k1_s1_p0(&out, &sw.proj_out_w, &sw.proj_out_b, c, c, h, w);
    residual.add(&out)
}

/// One transformer block: self-attn → cross-attn → GEGLU FFN.
fn transformer_block(
    x: &LazyTensor,  // [1, N, C]
    tb: &TransformerBlockWeights,
    text_emb: &LazyTensor,  // [1, 77, cross_dim]
    cfg: &SdUnetConfig,
    c: usize,
    n: usize,
) -> LazyTensor {
    let head_dim = cfg.attention_head_dim;
    assert_eq!(c % head_dim, 0);
    let n_heads = c / head_dim;
    let cross_dim = cfg.cross_attention_dim;

    // --- self-attn (pre-LN, no bias on q/k/v, bias on out) --------
    let x_ln = layer_norm_affine(x, &tb.n1_g, &tb.n1_b, 1e-5, c, n);
    let q = linear(&x_ln, &tb.attn1_q_w, None, c, c, n);
    let k = linear(&x_ln, &tb.attn1_k_w, None, c, c, n);
    let v = linear(&x_ln, &tb.attn1_v_w, None, c, c, n);
    let attn = multi_head_attn(&q, &k, &v, c, n, n, n_heads, head_dim);
    let attn = linear(&attn, &tb.attn1_out_w, Some(&tb.attn1_out_b), c, c, n);
    let x = x.add(&attn);

    // --- cross-attn (pre-LN, no bias on q/k/v, bias on out) -------
    let x_ln = layer_norm_affine(&x, &tb.n2_g, &tb.n2_b, 1e-5, c, n);
    let q = linear(&x_ln, &tb.attn2_q_w, None, c, c, n);
    let k = linear(text_emb, &tb.attn2_k_w, None, cross_dim, c, 77);
    let v = linear(text_emb, &tb.attn2_v_w, None, cross_dim, c, 77);
    let attn = multi_head_attn(&q, &k, &v, c, n, 77, n_heads, head_dim);
    let attn = linear(&attn, &tb.attn2_out_w, Some(&tb.attn2_out_b), c, c, n);
    let x = x.add(&attn);

    // --- GEGLU FFN (pre-LN) ---------------------------------------
    let x_ln = layer_norm_affine(&x, &tb.n3_g, &tb.n3_b, 1e-5, c, n);
    let mid = linear(&x_ln, &tb.ff_in_w, Some(&tb.ff_in_b), c, 2 * 4 * c, n);
    // GEGLU: split mid along last dim into [x, gate], compute x * gelu(gate).
    let half = 4 * c;
    let xv = mid.slice(2, 0, half);
    let gate = mid.slice(2, half, half);
    let gated = xv.mul(&gate.gelu());
    let ffn_out = linear(&gated, &tb.ff_out_w, Some(&tb.ff_out_b), 4 * c, c, n);
    x.add(&ffn_out)
}

fn multi_head_attn(
    q: &LazyTensor,  // [1, q_n, c]
    k: &LazyTensor,  // [1, kv_n, c]
    v: &LazyTensor,  // [1, kv_n, c]
    c: usize,
    q_n: usize,
    kv_n: usize,
    n_heads: usize,
    d_head: usize,
) -> LazyTensor {
    let q = q
        .reshape(Shape::from_dims(&[1, q_n, n_heads, d_head])).unwrap()
        .permute(&[0, 2, 1, 3]);  // [1, H, Q, D]
    let k = k
        .reshape(Shape::from_dims(&[1, kv_n, n_heads, d_head])).unwrap()
        .permute(&[0, 2, 1, 3]);
    let v = v
        .reshape(Shape::from_dims(&[1, kv_n, n_heads, d_head])).unwrap()
        .permute(&[0, 2, 1, 3]);
    let k_t = k.permute(&[0, 1, 3, 2]);  // [1, H, D, KV]
    let scale = 1.0 / (d_head as f64).sqrt();
    let scores = q.matmul(&k_t).mul_scalar(scale);  // [1, H, Q, KV]
    let probs = scores.softmax_last_dim();
    probs
        .matmul(&v)
        .permute(&[0, 2, 1, 3])
        .reshape(Shape::from_dims(&[1, q_n, c])).unwrap()
}

// ---- Sinusoidal timestep embedding ----------------------------------------

/// Standard sinusoidal positional encoding of a single scalar timestep.
/// Produces `dim` frequencies in `[sin(…), cos(…)]` interleaved halves
/// matching SD's `flip_sin_to_cos=true` convention.
fn sinusoidal_timestep_embedding(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let max_period = 10000.0_f32;
    let mut out = vec![0.0_f32; dim];
    for i in 0..half {
        let freq = (-(i as f32 / half as f32) * max_period.ln()).exp();
        let arg = t * freq;
        // cos first half, sin second half — the flip_sin_to_cos=true
        // order SD uses.
        out[i] = arg.cos();
        out[i + half] = arg.sin();
    }
    out
}

// ---- Primitives (shared with VAE; replicated here to keep the module
// self-contained while native Conv2d + GroupNorm ops land) ----------

fn layer_norm_affine(
    x: &LazyTensor,
    gamma: &Arc<[f32]>,
    beta: &Arc<[f32]>,
    eps: f64,
    hidden: usize,
    seq: usize,
) -> LazyTensor {
    let normed = x.layer_norm_last_dim(eps);
    let g = x
        .const_f32_like(gamma.clone(), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&[1, 1, hidden])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, seq, hidden])).unwrap();
    let b = x
        .const_f32_like(beta.clone(), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&[1, 1, hidden])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, seq, hidden])).unwrap();
    normed.mul(&g).add(&b)
}

fn group_norm(
    x: &LazyTensor,
    gamma: &Arc<[f32]>,
    beta: &Arc<[f32]>,
    groups: usize,
    eps: f64,
    c: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let cpg = c / groups;
    let m = cpg * h * w;
    let x_flat = x.reshape(Shape::from_dims(&[1, groups, m])).unwrap();
    let mean = x_flat.mean_dim(2);
    let mean_bc = mean
        .reshape(Shape::from_dims(&[1, groups, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, groups, m])).unwrap();
    let centered = x_flat.sub(&mean_bc);
    let sq = centered.mul(&centered);
    let var = sq.mean_dim(2);
    let std = var.add_scalar(eps).sqrt();
    let std_bc = std
        .reshape(Shape::from_dims(&[1, groups, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, groups, m])).unwrap();
    let normed = centered.div(&std_bc);
    let normed_chw = normed.reshape(Shape::from_dims(&[1, c, h, w])).unwrap();
    let g = x
        .const_f32_like(gamma.clone(), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w])).unwrap();
    let b = x
        .const_f32_like(beta.clone(), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w])).unwrap();
    normed_chw.mul(&g).add(&b)
}

/// 3×3 conv, stride 1, padding 1. Dispatches to the native `Op::Conv2D`.
fn conv2d_k3_s1_p1(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: &Arc<[f32]>,
    cin: usize,
    cout: usize,
    _h: usize,
    _w_sz: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[cout, cin, 3, 3]));
    let b_t = x.const_f32_like(b.clone(), Shape::from_dims(&[cout]));
    x.conv2d(&w_t, Some(&b_t), (1, 1), (1, 1), 1)
}

/// Stride-2 3×3 conv with padding 1. Dispatches to the native `Op::Conv2D`.
fn conv2d_k3_s2_p1(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: &Arc<[f32]>,
    cin: usize,
    cout: usize,
    _h: usize,
    _w_sz: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[cout, cin, 3, 3]));
    let b_t = x.const_f32_like(b.clone(), Shape::from_dims(&[cout]));
    x.conv2d(&w_t, Some(&b_t), (2, 2), (1, 1), 1)
}

/// 1×1 conv, stride 1, padding 0. Dispatches to the native `Op::Conv2D`.
fn conv2d_k1_s1_p0(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: &Arc<[f32]>,
    cin: usize,
    cout: usize,
    _h: usize,
    _w_sz: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[cout, cin, 1, 1]));
    let b_t = x.const_f32_like(b.clone(), Shape::from_dims(&[cout]));
    x.conv2d(&w_t, Some(&b_t), (1, 1), (0, 0), 1)
}

fn upsample_nearest_2x(x: &LazyTensor, c: usize, h: usize, w: usize) -> LazyTensor {
    let x6 = x.reshape(Shape::from_dims(&[1, c, h, 1, w, 1])).unwrap();
    let x6 = x6.concat(&x6, 3);
    let x6 = x6.concat(&x6, 5);
    x6.reshape(Shape::from_dims(&[1, c, 2 * h, 2 * w])).unwrap()
}

fn linear(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: Option<&Arc<[f32]>>,
    in_f: usize,
    out_f: usize,
    seq: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[in_f, out_f]));
    let proj = x.matmul(&w_t);
    match b {
        Some(b) => {
            let bias = x
                .const_f32_like(b.clone(), Shape::from_dims(&[out_f]))
                .reshape(Shape::from_dims(&[1, 1, out_f])).unwrap()
                .broadcast_to(Shape::from_dims(&[1, seq, out_f])).unwrap();
            proj.add(&bias)
        }
        None => proj,
    }
}

// ---- Safetensors loader ----------------------------------------------------

impl SdUnetWeights {
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &SdUnetConfig,
    ) -> crate::Result<Self> {
        let c_first = cfg.block_out_channels[0];

        // time embedding MLP
        let tm1_w = load_transposed(st, "time_embedding.linear_1.weight", cfg.time_embed_dim, c_first)?;
        let tm1_b = load_f32(st, "time_embedding.linear_1.bias")?;
        let tm2_w = load_transposed(st, "time_embedding.linear_2.weight", cfg.time_embed_dim, cfg.time_embed_dim)?;
        let tm2_b = load_f32(st, "time_embedding.linear_2.bias")?;

        // conv_in
        let conv_in_w = load_f32(st, "conv_in.weight")?;
        let conv_in_b = load_f32(st, "conv_in.bias")?;

        // down blocks
        let mut down_blocks = Vec::with_capacity(4);
        for bi in 0..4 {
            let c_in = if bi == 0 { c_first } else { cfg.block_out_channels[bi - 1] };
            let c_out = cfg.block_out_channels[bi];
            let has_attn = cfg.down_has_attn[bi];

            let mut resnets = Vec::with_capacity(cfg.layers_per_block);
            let mut attentions = Vec::with_capacity(cfg.layers_per_block);
            for ri in 0..cfg.layers_per_block {
                let in_c = if ri == 0 { c_in } else { c_out };
                let r = load_u_resnet(st, &format!("down_blocks.{bi}.resnets.{ri}"),
                    in_c, c_out, cfg.time_embed_dim)?;
                resnets.push(r);
                if has_attn {
                    let a = load_spatial_transformer(st,
                        &format!("down_blocks.{bi}.attentions.{ri}"),
                        c_out, cfg.cross_attention_dim)?;
                    attentions.push(a);
                }
            }
            let (downsample_w, downsample_b) = if bi < 3 {
                let w = load_f32(st, &format!("down_blocks.{bi}.downsamplers.0.conv.weight"))?;
                let b = load_f32(st, &format!("down_blocks.{bi}.downsamplers.0.conv.bias"))?;
                (Some(Arc::from(w)), Some(Arc::from(b)))
            } else {
                (None, None)
            };
            down_blocks.push(DownBlockWeights {
                resnets, attentions,
                downsample_w, downsample_b,
            });
        }

        // mid block
        let c_mid = *cfg.block_out_channels.last().unwrap();
        let mid_resnet_1 = load_u_resnet(st, "mid_block.resnets.0", c_mid, c_mid, cfg.time_embed_dim)?;
        let mid_attn = load_spatial_transformer(st, "mid_block.attentions.0", c_mid, cfg.cross_attention_dim)?;
        let mid_resnet_2 = load_u_resnet(st, "mid_block.resnets.1", c_mid, c_mid, cfg.time_embed_dim)?;

        // up blocks — layers_per_block + 1 resnets each, skip-connection concat
        let mut up_blocks = Vec::with_capacity(4);
        for bi in 0..4 {
            let n_res = cfg.layers_per_block + 1;
            let c_out = cfg.block_out_channels[cfg.block_out_channels.len() - 1 - bi];
            let c_prev_up = if bi == 0 { c_mid } else { cfg.block_out_channels[cfg.block_out_channels.len() - bi] };
            let has_attn = cfg.up_has_attn[bi];

            let mut resnets = Vec::with_capacity(n_res);
            let mut attentions = Vec::with_capacity(n_res);
            for ri in 0..n_res {
                // Compute this resnet's input channel count (x + skip).
                let c_skip = if ri < cfg.layers_per_block {
                    c_out
                } else {
                    // Last resnet consumes the skip from before the
                    // previous down block's downsample (which was at
                    // block_out_channels[len - bi - 1] before the
                    // downsample changed count — in SD 1.5 the
                    // downsampled dim equals c_out of that down block).
                    if bi == 3 { c_first } else { cfg.block_out_channels[cfg.block_out_channels.len() - 1 - bi - 1] }
                };
                let in_c = if ri == 0 { c_prev_up + c_skip } else { c_out + c_skip };
                let r = load_u_resnet(st, &format!("up_blocks.{bi}.resnets.{ri}"),
                    in_c, c_out, cfg.time_embed_dim)?;
                resnets.push(r);
                if has_attn {
                    let a = load_spatial_transformer(st,
                        &format!("up_blocks.{bi}.attentions.{ri}"),
                        c_out, cfg.cross_attention_dim)?;
                    attentions.push(a);
                }
            }
            let (upsample_conv_w, upsample_conv_b) = if bi < 3 {
                let w = load_f32(st, &format!("up_blocks.{bi}.upsamplers.0.conv.weight"))?;
                let b = load_f32(st, &format!("up_blocks.{bi}.upsamplers.0.conv.bias"))?;
                (Some(Arc::from(w)), Some(Arc::from(b)))
            } else {
                (None, None)
            };
            up_blocks.push(UpBlockWeights { resnets, attentions, upsample_conv_w, upsample_conv_b });
        }

        let conv_norm_out_g = load_f32(st, "conv_norm_out.weight")?;
        let conv_norm_out_b = load_f32(st, "conv_norm_out.bias")?;
        let conv_out_w = load_f32(st, "conv_out.weight")?;
        let conv_out_b = load_f32(st, "conv_out.bias")?;

        Ok(Self {
            time_mlp_1_w: Arc::from(tm1_w), time_mlp_1_b: Arc::from(tm1_b),
            time_mlp_2_w: Arc::from(tm2_w), time_mlp_2_b: Arc::from(tm2_b),
            conv_in_w: Arc::from(conv_in_w), conv_in_b: Arc::from(conv_in_b),
            down_blocks,
            mid_resnet_1, mid_attn, mid_resnet_2,
            up_blocks,
            conv_norm_out_g: Arc::from(conv_norm_out_g),
            conv_norm_out_b: Arc::from(conv_norm_out_b),
            conv_out_w: Arc::from(conv_out_w),
            conv_out_b: Arc::from(conv_out_b),
        })
    }
}

fn load_u_resnet(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c_in: usize,
    c_out: usize,
    time_dim: usize,
) -> crate::Result<UResnetWeights> {
    let n1_g = load_f32(st, &format!("{prefix}.norm1.weight"))?;
    let n1_b = load_f32(st, &format!("{prefix}.norm1.bias"))?;
    let c1_w = load_f32(st, &format!("{prefix}.conv1.weight"))?;
    let c1_b = load_f32(st, &format!("{prefix}.conv1.bias"))?;
    let te_w = load_transposed(st, &format!("{prefix}.time_emb_proj.weight"), c_out, time_dim)?;
    let te_b = load_f32(st, &format!("{prefix}.time_emb_proj.bias"))?;
    let n2_g = load_f32(st, &format!("{prefix}.norm2.weight"))?;
    let n2_b = load_f32(st, &format!("{prefix}.norm2.bias"))?;
    let c2_w = load_f32(st, &format!("{prefix}.conv2.weight"))?;
    let c2_b = load_f32(st, &format!("{prefix}.conv2.bias"))?;
    let (shortcut_w, shortcut_b) = if c_in != c_out {
        let sw = load_f32(st, &format!("{prefix}.conv_shortcut.weight"))?;
        let sb = load_f32(st, &format!("{prefix}.conv_shortcut.bias"))?;
        (Some(Arc::from(sw)), Some(Arc::from(sb)))
    } else { (None, None) };
    Ok(UResnetWeights {
        n1_g: Arc::from(n1_g), n1_b: Arc::from(n1_b),
        c1_w: Arc::from(c1_w), c1_b: Arc::from(c1_b),
        te_w: Arc::from(te_w), te_b: Arc::from(te_b),
        n2_g: Arc::from(n2_g), n2_b: Arc::from(n2_b),
        c2_w: Arc::from(c2_w), c2_b: Arc::from(c2_b),
        shortcut_w, shortcut_b,
    })
}

fn load_spatial_transformer(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c: usize,
    cross_dim: usize,
) -> crate::Result<SpatialTransformerWeights> {
    let norm_g = load_f32(st, &format!("{prefix}.norm.weight"))?;
    let norm_b = load_f32(st, &format!("{prefix}.norm.bias"))?;
    let proj_in_w = load_f32(st, &format!("{prefix}.proj_in.weight"))?;
    let proj_in_b = load_f32(st, &format!("{prefix}.proj_in.bias"))?;
    let proj_out_w = load_f32(st, &format!("{prefix}.proj_out.weight"))?;
    let proj_out_b = load_f32(st, &format!("{prefix}.proj_out.bias"))?;
    // SD 1.5 has 1 transformer block per Transformer2DModel.
    let p = format!("{prefix}.transformer_blocks.0");
    let tb = TransformerBlockWeights {
        n1_g: Arc::from(load_f32(st, &format!("{p}.norm1.weight"))?),
        n1_b: Arc::from(load_f32(st, &format!("{p}.norm1.bias"))?),
        attn1_q_w: Arc::from(load_transposed(st, &format!("{p}.attn1.to_q.weight"), c, c)?),
        attn1_k_w: Arc::from(load_transposed(st, &format!("{p}.attn1.to_k.weight"), c, c)?),
        attn1_v_w: Arc::from(load_transposed(st, &format!("{p}.attn1.to_v.weight"), c, c)?),
        attn1_out_w: Arc::from(load_transposed(st, &format!("{p}.attn1.to_out.0.weight"), c, c)?),
        attn1_out_b: Arc::from(load_f32(st, &format!("{p}.attn1.to_out.0.bias"))?),
        n2_g: Arc::from(load_f32(st, &format!("{p}.norm2.weight"))?),
        n2_b: Arc::from(load_f32(st, &format!("{p}.norm2.bias"))?),
        attn2_q_w: Arc::from(load_transposed(st, &format!("{p}.attn2.to_q.weight"), c, c)?),
        attn2_k_w: Arc::from(load_transposed(st, &format!("{p}.attn2.to_k.weight"), c, cross_dim)?),
        attn2_v_w: Arc::from(load_transposed(st, &format!("{p}.attn2.to_v.weight"), c, cross_dim)?),
        attn2_out_w: Arc::from(load_transposed(st, &format!("{p}.attn2.to_out.0.weight"), c, c)?),
        attn2_out_b: Arc::from(load_f32(st, &format!("{p}.attn2.to_out.0.bias"))?),
        n3_g: Arc::from(load_f32(st, &format!("{p}.norm3.weight"))?),
        n3_b: Arc::from(load_f32(st, &format!("{p}.norm3.bias"))?),
        ff_in_w: Arc::from(load_transposed(st, &format!("{p}.ff.net.0.proj.weight"), 2 * 4 * c, c)?),
        ff_in_b: Arc::from(load_f32(st, &format!("{p}.ff.net.0.proj.bias"))?),
        ff_out_w: Arc::from(load_transposed(st, &format!("{p}.ff.net.2.weight"), c, 4 * c)?),
        ff_out_b: Arc::from(load_f32(st, &format!("{p}.ff.net.2.bias"))?),
    };
    Ok(SpatialTransformerWeights {
        norm_g: Arc::from(norm_g), norm_b: Arc::from(norm_b),
        proj_in_w: Arc::from(proj_in_w), proj_in_b: Arc::from(proj_in_b),
        blocks: vec![tb],
        proj_out_w: Arc::from(proj_out_w), proj_out_b: Arc::from(proj_out_b),
    })
}

fn load_f32(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
) -> crate::Result<Vec<f32>> {
    use safetensors::Dtype;
    let view = st
        .get(name)
        .map_err(|e| crate::Error::Msg(format!("unet load_f32 {name:?}: {e}")).bt())?;
    let bytes = view.data();
    match view.dtype() {
        Dtype::F32 => {
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        Dtype::F16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::f16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        other => crate::bail!("unet load_f32: unsupported dtype {other:?} for {name:?}"),
    }
}

fn load_transposed(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> crate::Result<Vec<f32>> {
    let flat = load_f32(st, name)?;
    if flat.len() != out_features * in_features {
        crate::bail!(
            "unet load_transposed: {name:?} has {} elements, expected {}",
            flat.len(), out_features * in_features,
        );
    }
    let mut out = vec![0.0_f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            out[j * out_features + i] = flat[i * in_features + j];
        }
    }
    Ok(out)
}

impl SdUnet {
    pub fn from_hub(repo_id: &str) -> crate::Result<Self> {
        Self::from_hub_with_config(repo_id, SdUnetConfig::sd_v1())
    }
    pub fn from_hub_with_config(repo_id: &str, config: SdUnetConfig) -> crate::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let path = repo
            .get("unet/diffusion_pytorch_model.safetensors")
            .map_err(|e| crate::Error::Msg(format!("hf-hub unet safetensors: {e}")))?;
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path) }?;
        let weights = SdUnetWeights::load_from_mmapped(&st, &config)?;
        Ok(Self { config, weights })
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd_v1_config_shape() {
        let cfg = SdUnetConfig::sd_v1();
        assert_eq!(cfg.block_out_channels, vec![320, 640, 1280, 1280]);
        assert_eq!(cfg.cross_attention_dim, 768);
        assert_eq!(cfg.time_embed_dim, 1280);
    }

    #[test]
    fn sinusoidal_embedding_stable_shape() {
        let emb = sinusoidal_timestep_embedding(100.0, 320);
        assert_eq!(emb.len(), 320);
        assert!(emb.iter().all(|v| v.is_finite()));
    }
}

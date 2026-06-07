//! Stable Diffusion 3 VAE decoder ported to the lazy-graph API.
//!
//! SD3 (and SD 3.5) ships an AutoencoderKL whose latent space is
//! **16-channel** rather than the 4-channel space used by SD 1.5 /
//! SD 2.x / SDXL. The convolutional backbone of the decoder is
//! otherwise structurally identical to SD 1.5's VAE — same
//! `block_out_channels = [128, 256, 512, 512]`, same
//! `layers_per_block = 2` (→ 3 ResNets per up-block in the decoder),
//! same `norm_num_groups = 32`, same mid-block (ResNet + spatial
//! attention + ResNet). The only weight-shape difference is at the
//! decoder's `conv_in` (`[d_mid, 16, 3, 3]` instead of
//! `[d_mid, 4, 3, 3]`) and at the encoder's `conv_out` (which we don't
//! port — inference-time SD3 only ever runs the decoder).
//!
//! Two other deltas vs SD 1.5:
//!
//! - **No `post_quant_conv`.** SD3's checkpoint sets
//!   `use_post_quant_conv = false`, so the raw latent goes straight
//!   into the decoder's `conv_in`.
//! - **Latent scale + shift.** Diffusers / TAESD3 convention is to
//!   normalize SD3 latents as `(x / 1.5305) + 0.0609` before decoding.
//!   We apply that inline at the entry to [`SdVae3Decoder::decode`] so
//!   callers can hand the model the raw sampler output without any
//!   pre-arithmetic.
//!
//! All the heavy machinery (group_norm, conv2d_k3_s1_p1,
//! conv2d_k1_s1_p0, upsample_nearest_2x, spatial attention, ResNet
//! block) is identical to the SD 1.5 VAE port in
//! [`crate::lazy_sd_vae`] — see that file for op-level commentary.
//! This file deliberately copies those primitives rather than
//! depending on `lazy_sd_vae`'s private fns so the two VAE variants
//! stay independent (one can be edited or retired without touching
//! the other).

use crate::lazy::LazyTensor;
use fuel_core_types::Shape;
use std::sync::Arc;

// ---- Config ----------------------------------------------------------------

/// Hyperparameters for SD3's AutoencoderKL decoder. All values are
/// fixed by the trained checkpoint; this struct just collects them.
#[derive(Debug, Clone)]
pub struct SdVae3Config {
    /// Channel widths per decoder stage, **in decoder order** (i.e. the
    /// reverse of the `block_out_channels` list in the HF config).
    /// For SD3: `[512, 512, 512, 256, 128]` — 512 at the mid block,
    /// then 4 up blocks stepping down to 128. Structurally identical
    /// to SD 1.5.
    pub dims: Vec<usize>,
    /// Latent channel count (16 for SD3).
    pub latent_channels: usize,
    /// Output channel count (3 — RGB).
    pub out_channels: usize,
    /// ResNet blocks per up-block (3 for SD3 — `layers_per_block + 1`).
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
    /// Latent scale factor applied at decode entry as `x / scale`.
    /// `1.5305` for SD3 (TAESD3 / Diffusers convention).
    pub scale_factor: f64,
    /// Latent shift applied after scaling at decode entry as `x + shift`.
    /// `0.0609` for SD3.
    pub shift_factor: f64,
}

impl SdVae3Config {
    /// Stable Diffusion 3 / 3.5 AutoencoderKL decoder config.
    pub fn sd3() -> Self {
        Self {
            // block_out_channels = [128, 256, 512, 512], reversed =
            // [512, 512, 256, 128]. Mid block runs at 512.
            dims: vec![512, 512, 512, 256, 128],
            latent_channels: 16,
            out_channels: 3,
            layers_per_block: 3,
            norm_num_groups: 32,
            norm_eps: 1e-6,
            scale_factor: 1.5305,
            shift_factor: 0.0609,
        }
    }
}

// ---- Weight storage --------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Resnet3Weights {
    pub n1_g: Arc<[f32]>, pub n1_b: Arc<[f32]>,
    /// [Cout, Cin, 3, 3] (HF order, used by `conv2d_k3_s1_p1`).
    pub c1_w: Arc<[f32]>, pub c1_b: Arc<[f32]>,
    pub n2_g: Arc<[f32]>, pub n2_b: Arc<[f32]>,
    pub c2_w: Arc<[f32]>, pub c2_b: Arc<[f32]>,
    /// Optional 1×1 shortcut when in_channels != out_channels.
    /// Shape `[Cout, Cin, 1, 1]`.
    pub shortcut_w: Option<Arc<[f32]>>,
    pub shortcut_b: Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct Attn3Weights {
    pub gn_g: Arc<[f32]>, pub gn_b: Arc<[f32]>,
    /// Stored row-major `[C, C]` (not a conv). Load-time transpose
    /// gives `[C, C]` suitable for `x @ W`.
    pub q_w: Arc<[f32]>, pub q_b: Arc<[f32]>,
    pub k_w: Arc<[f32]>, pub k_b: Arc<[f32]>,
    pub v_w: Arc<[f32]>, pub v_b: Arc<[f32]>,
    pub out_w: Arc<[f32]>, pub out_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct UpBlock3Weights {
    pub resnets: Vec<Resnet3Weights>,
    /// Upsampler's 3×3 conv (applied after 2× nearest upsample). None
    /// for the last up-block (SD3's `up_blocks.3`).
    pub upsample_conv: Option<(Arc<[f32]>, Arc<[f32]>)>,
}

#[derive(Debug, Clone)]
pub struct SdVae3DecoderWeights {
    /// 3×3 conv (latent_ch → dim[0]). SD3 has no `post_quant_conv`,
    /// so the raw (scaled+shifted) latent feeds straight in here.
    pub conv_in_w: Arc<[f32]>,
    pub conv_in_b: Arc<[f32]>,
    /// Mid block: ResNet + Attention + ResNet, all at dim[0].
    pub mid_resnet_1: Resnet3Weights,
    pub mid_attn:     Attn3Weights,
    pub mid_resnet_2: Resnet3Weights,
    pub up_blocks: Vec<UpBlock3Weights>,
    pub conv_norm_out_g: Arc<[f32]>,
    pub conv_norm_out_b: Arc<[f32]>,
    pub conv_out_w: Arc<[f32]>,
    pub conv_out_b: Arc<[f32]>,
}

// ---- Model -----------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SdVae3Decoder {
    pub config:  SdVae3Config,
    pub weights: SdVae3DecoderWeights,
}

impl SdVae3Decoder {
    /// Decode a 16-channel SD3 latent tensor `[1, 16, H_lat, W_lat]`
    /// into an image `[1, 3, 8*H_lat, 8*W_lat]`. For SD3 the standard
    /// shape is `H_lat = W_lat = 64` giving a 512×512 output; bigger /
    /// smaller latents work at the same weights (convs are
    /// translation-invariant and the spatial attention runs on
    /// arbitrary H×W).
    ///
    /// The TAESD3 scale + shift constants (`/ 1.5305 + 0.0609`) are
    /// applied inline at entry, so callers pass the raw sampler output
    /// without pre-arithmetic.
    pub fn decode(&self, latent: &LazyTensor) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let lc = cfg.latent_channels;

        // Validate input shape: [1, latent_channels, H_lat, W_lat].
        let shape = latent.shape();
        let dims = shape.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[1] != lc {
            crate::bail!(
                "SdVae3Decoder::decode: expected latent shape [1, {lc}, H, W], got {dims:?}"
            );
        }
        let h_lat = dims[2];
        let w_lat = dims[3];

        // Apply SD3 latent scale + shift: x_in = (x / 1.5305) + 0.0609.
        // mul_scalar(1/k) matches eager `(x / k)`; the order matches the
        // diffusers / TAESD3 reference (scale before shift).
        let x = latent.mul_scalar(1.0 / cfg.scale_factor).add_scalar(cfg.shift_factor);

        // conv_in: 3×3 conv, [1, 16, H, W] → [1, dim[0], H, W].
        // (No post_quant_conv for SD3.)
        let d_mid = cfg.dims[0];
        let x = conv2d_k3_s1_p1(&x, &self.weights.conv_in_w, &self.weights.conv_in_b,
            lc, d_mid, h_lat, w_lat);

        // mid block: ResNet + Attention + ResNet (all at d_mid).
        let x = resnet(&x, &self.weights.mid_resnet_1, cfg, d_mid, d_mid, h_lat, w_lat);
        let x = vae_spatial_attention(&x, &self.weights.mid_attn, cfg, d_mid, h_lat, w_lat);
        let x = resnet(&x, &self.weights.mid_resnet_2, cfg, d_mid, d_mid, h_lat, w_lat);

        // up blocks: 4 stages. dims[1..] gives the output channel for each.
        let mut x = x;
        let mut c = d_mid;
        let mut h = h_lat;
        let mut w = w_lat;
        for (si, up) in self.weights.up_blocks.iter().enumerate() {
            let c_out = cfg.dims[1 + si];
            for (ri, rb) in up.resnets.iter().enumerate() {
                let c_in = if ri == 0 { c } else { c_out };
                x = resnet(&x, rb, cfg, c_in, c_out, h, w);
            }
            c = c_out;
            if let Some((uw, ub)) = &up.upsample_conv {
                x = upsample_nearest_2x(&x, c, h, w);
                h *= 2;
                w *= 2;
                x = conv2d_k3_s1_p1(&x, uw, ub, c, c, h, w);
            }
        }

        // Final norm + SiLU + 3×3 conv → [1, 3, 8H_lat, 8W_lat].
        let x = group_norm(&x, &self.weights.conv_norm_out_g, &self.weights.conv_norm_out_b,
            cfg.norm_num_groups, cfg.norm_eps, c, h, w);
        let x = x.silu();
        Ok(conv2d_k3_s1_p1(&x, &self.weights.conv_out_w, &self.weights.conv_out_b,
            c, cfg.out_channels, h, w))
    }
}

// ---- ResNet block ---------------------------------------------------------

fn resnet(
    x: &LazyTensor,
    rw: &Resnet3Weights,
    cfg: &SdVae3Config,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let h1 = group_norm(x, &rw.n1_g, &rw.n1_b, cfg.norm_num_groups, cfg.norm_eps, c_in, h, w);
    let h1 = h1.silu();
    let h1 = conv2d_k3_s1_p1(&h1, &rw.c1_w, &rw.c1_b, c_in, c_out, h, w);
    let h2 = group_norm(&h1, &rw.n2_g, &rw.n2_b, cfg.norm_num_groups, cfg.norm_eps, c_out, h, w);
    let h2 = h2.silu();
    let h2 = conv2d_k3_s1_p1(&h2, &rw.c2_w, &rw.c2_b, c_out, c_out, h, w);
    let shortcut = match (&rw.shortcut_w, &rw.shortcut_b) {
        (Some(w_s), Some(b_s)) => conv2d_k1_s1_p0(x, w_s, b_s, c_in, c_out, h, w),
        _ => x.clone(),
    };
    shortcut.add(&h2).unwrap()
}

// ---- VAE spatial attention ------------------------------------------------

/// Single-head self-attention over `[H*W]` positions. Inputs / outputs
/// are `[1, C, H, W]`; internally we permute to `[1, H*W, C]`, project
/// Q/K/V as plain linears, scaled dot-product, then project out and
/// reshape back. Mirrors SD 1.5 — SD3's VAE mid-block attention is
/// the same op.
fn vae_spatial_attention(
    x: &LazyTensor,
    aw: &Attn3Weights,
    cfg: &SdVae3Config,
    c: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let n = h * w;
    let x_norm = group_norm(x, &aw.gn_g, &aw.gn_b, cfg.norm_num_groups, cfg.norm_eps, c, h, w);
    // [1, C, H, W] → [1, H*W, C].
    let xf = x_norm
        .permute([0, 2, 3, 1_usize]).unwrap()
        .reshape(Shape::from_dims(&[1, n, c])).unwrap();
    let q = linear(&xf, &aw.q_w, Some(&aw.q_b), c, c, n);
    let k = linear(&xf, &aw.k_w, Some(&aw.k_b), c, c, n);
    let v = linear(&xf, &aw.v_w, Some(&aw.v_b), c, c, n);
    let k_t = k.permute([0, 2, 1_usize]).unwrap();  // [1, C, N]
    let scores = q.matmul(&k_t).unwrap().mul_scalar(1.0 / (c as f64).sqrt());  // [1, N, N]
    let probs = scores.softmax_last_dim().unwrap();
    let ctx = probs.matmul(&v).unwrap();  // [1, N, C]
    let out = linear(&ctx, &aw.out_w, Some(&aw.out_b), c, c, n);
    let out_chw = out
        .reshape(Shape::from_dims(&[1, h, w, c])).unwrap()
        .permute([0, 3, 1, 2_usize]).unwrap();
    x.add(&out_chw).unwrap()
}

// ---- Primitives -----------------------------------------------------------

/// GroupNorm with per-channel affine. Input `[1, C, H, W]`. Built from
/// mean_dim + manual variance + sqrt + broadcast. Same shape as
/// `lazy_sd_vae::group_norm`; kept private here so the two ports stay
/// independent.
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
    assert_eq!(c % groups, 0, "group_norm: C={c} not divisible by groups={groups}");
    let cpg = c / groups;
    let m = cpg * h * w;  // elements per group

    let x_flat = x.reshape(Shape::from_dims(&[1, groups, m])).unwrap();
    let mean = x_flat.mean_dim(2).unwrap();
    let mean_bc = mean
        .reshape(Shape::from_dims(&[1, groups, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, groups, m])).unwrap();
    let centered = x_flat.sub(&mean_bc).unwrap();
    let sq = centered.mul(&centered).unwrap();
    let var = sq.mean_dim(2).unwrap();
    let std = var.add_scalar(eps).sqrt();
    let std_bc = std
        .reshape(Shape::from_dims(&[1, groups, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, groups, m])).unwrap();
    let normed = centered.div(&std_bc).unwrap();
    let normed_chw = normed.reshape(Shape::from_dims(&[1, c, h, w])).unwrap();
    let g = x
        .const_f32_like(gamma.clone(), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w])).unwrap();
    let b = x
        .const_f32_like(beta.clone(), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w])).unwrap();
    normed_chw.mul(&g).unwrap().add(&b).unwrap()
}

/// General 3×3 conv, stride 1, padding 1. Input `[1, Cin, H, W]`,
/// kernel `[Cout, Cin, 3, 3]` in HF order, bias `[Cout]`. Output
/// `[1, Cout, H, W]`. Dispatches to the native `Op::Conv2D`.
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
    x.conv2d(&w_t, Some(&b_t), (1, 1), (1, 1), 1).unwrap()
}

/// 1×1 conv, stride 1, padding 0. Input `[1, Cin, H, W]`, kernel
/// `[Cout, Cin, 1, 1]`, bias `[Cout]`. Output `[1, Cout, H, W]`.
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
    x.conv2d(&w_t, Some(&b_t), (1, 1), (0, 0), 1).unwrap()
}

/// 2× nearest-neighbor upsample along both spatial axes.
fn upsample_nearest_2x(x: &LazyTensor, c: usize, h: usize, w: usize) -> LazyTensor {
    let x6 = x.reshape(Shape::from_dims(&[1, c, h, 1, w, 1])).unwrap();
    let x6 = x6.concat(&x6, 3).unwrap();
    let x6 = x6.concat(&x6, 5).unwrap();
    x6.reshape(Shape::from_dims(&[1, c, 2 * h, 2 * w])).unwrap()
}

/// `y = x @ W + b`. `x`: `[1, seq, in_f]`, `W`: `[in_f, out_f]`.
fn linear(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: Option<&Arc<[f32]>>,
    in_f: usize,
    out_f: usize,
    seq: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[in_f, out_f]));
    let proj = x.matmul(&w_t).unwrap();
    match b {
        Some(b) => {
            let bias = x
                .const_f32_like(b.clone(), Shape::from_dims(&[out_f]))
                .reshape(Shape::from_dims(&[1, 1, out_f])).unwrap()
                .broadcast_to(Shape::from_dims(&[1, seq, out_f])).unwrap();
            proj.add(&bias).unwrap()
        }
        None => proj,
    }
}

// ---- Safetensors loader ----------------------------------------------------

impl SdVae3DecoderWeights {
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &SdVae3Config,
    ) -> crate::Result<Self> {
        let _lc = cfg.latent_channels;
        let _oc = cfg.out_channels;
        let d_mid = cfg.dims[0];

        // SD3 has no post_quant_conv — `use_post_quant_conv = false`
        // on the trained checkpoint. The raw latent flows straight
        // into `decoder.conv_in`.

        let conv_in_w = load_f32(st, "decoder.conv_in.weight")?;
        let conv_in_b = load_f32(st, "decoder.conv_in.bias")?;

        let mid_resnet_1 = load_resnet(st, "decoder.mid_block.resnets.0", d_mid, d_mid)?;
        let mid_attn = load_attn(st, "decoder.mid_block.attentions.0", d_mid)?;
        let mid_resnet_2 = load_resnet(st, "decoder.mid_block.resnets.1", d_mid, d_mid)?;

        let mut up_blocks = Vec::with_capacity(4);
        for si in 0..4 {
            let c_in = cfg.dims[si];  // input channel for block si (= output of previous)
            let c_out = cfg.dims[si + 1];
            let mut resnets = Vec::with_capacity(cfg.layers_per_block);
            for ri in 0..cfg.layers_per_block {
                let in_c = if ri == 0 { c_in } else { c_out };
                let r = load_resnet(
                    st,
                    &format!("decoder.up_blocks.{si}.resnets.{ri}"),
                    in_c, c_out,
                )?;
                resnets.push(r);
            }
            // Last up_block has no upsampler.
            let upsample_conv = if si == 3 {
                None
            } else {
                let p = format!("decoder.up_blocks.{si}.upsamplers.0.conv");
                let uw = load_f32(st, &format!("{p}.weight"))?;
                let ub = load_f32(st, &format!("{p}.bias"))?;
                Some((Arc::from(uw), Arc::from(ub)))
            };
            up_blocks.push(UpBlock3Weights { resnets, upsample_conv });
        }

        let conv_norm_out_g = load_f32(st, "decoder.conv_norm_out.weight")?;
        let conv_norm_out_b = load_f32(st, "decoder.conv_norm_out.bias")?;
        let conv_out_w = load_f32(st, "decoder.conv_out.weight")?;
        let conv_out_b = load_f32(st, "decoder.conv_out.bias")?;

        Ok(Self {
            conv_in_w: Arc::from(conv_in_w),
            conv_in_b: Arc::from(conv_in_b),
            mid_resnet_1, mid_attn, mid_resnet_2,
            up_blocks,
            conv_norm_out_g: Arc::from(conv_norm_out_g),
            conv_norm_out_b: Arc::from(conv_norm_out_b),
            conv_out_w: Arc::from(conv_out_w),
            conv_out_b: Arc::from(conv_out_b),
        })
    }
}

fn load_resnet(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c_in: usize,
    c_out: usize,
) -> crate::Result<Resnet3Weights> {
    let n1_g = load_f32(st, &format!("{prefix}.norm1.weight"))?;
    let n1_b = load_f32(st, &format!("{prefix}.norm1.bias"))?;
    let c1_w = load_f32(st, &format!("{prefix}.conv1.weight"))?;
    let c1_b = load_f32(st, &format!("{prefix}.conv1.bias"))?;
    let n2_g = load_f32(st, &format!("{prefix}.norm2.weight"))?;
    let n2_b = load_f32(st, &format!("{prefix}.norm2.bias"))?;
    let c2_w = load_f32(st, &format!("{prefix}.conv2.weight"))?;
    let c2_b = load_f32(st, &format!("{prefix}.conv2.bias"))?;
    let (shortcut_w, shortcut_b) = if c_in != c_out {
        let sw = load_f32(st, &format!("{prefix}.conv_shortcut.weight"))?;
        let sb = load_f32(st, &format!("{prefix}.conv_shortcut.bias"))?;
        (Some(Arc::from(sw)), Some(Arc::from(sb)))
    } else {
        (None, None)
    };
    Ok(Resnet3Weights {
        n1_g: Arc::from(n1_g), n1_b: Arc::from(n1_b),
        c1_w: Arc::from(c1_w), c1_b: Arc::from(c1_b),
        n2_g: Arc::from(n2_g), n2_b: Arc::from(n2_b),
        c2_w: Arc::from(c2_w), c2_b: Arc::from(c2_b),
        shortcut_w, shortcut_b,
    })
}

fn load_attn(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c: usize,
) -> crate::Result<Attn3Weights> {
    let gn_g = load_f32(st, &format!("{prefix}.group_norm.weight"))?;
    let gn_b = load_f32(st, &format!("{prefix}.group_norm.bias"))?;
    let q_w = load_transposed(st, &format!("{prefix}.query.weight"), c, c)?;
    let q_b = load_f32(st, &format!("{prefix}.query.bias"))?;
    let k_w = load_transposed(st, &format!("{prefix}.key.weight"), c, c)?;
    let k_b = load_f32(st, &format!("{prefix}.key.bias"))?;
    let v_w = load_transposed(st, &format!("{prefix}.value.weight"), c, c)?;
    let v_b = load_f32(st, &format!("{prefix}.value.bias"))?;
    let out_w = load_transposed(st, &format!("{prefix}.proj_attn.weight"), c, c)?;
    let out_b = load_f32(st, &format!("{prefix}.proj_attn.bias"))?;
    Ok(Attn3Weights {
        gn_g: Arc::from(gn_g), gn_b: Arc::from(gn_b),
        q_w: Arc::from(q_w), q_b: Arc::from(q_b),
        k_w: Arc::from(k_w), k_b: Arc::from(k_b),
        v_w: Arc::from(v_w), v_b: Arc::from(v_b),
        out_w: Arc::from(out_w), out_b: Arc::from(out_b),
    })
}

fn load_f32(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
) -> crate::Result<Vec<f32>> {
    use safetensors::Dtype;
    let view = st
        .get(name)
        .map_err(|e| crate::Error::Msg(format!("sd3 vae load_f32 {name:?}: {e}")).bt())?;
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
        other => crate::bail!("sd3 vae load_f32: unsupported dtype {other:?} for {name:?}"),
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
            "sd3 vae load_transposed: {name:?} has {} elements, expected {}",
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

// ---- HuggingFace Hub integration -------------------------------------------

impl SdVae3Decoder {
    pub fn from_hub(repo_id: &str) -> crate::Result<Self> {
        Self::from_hub_with_config(repo_id, SdVae3Config::sd3())
    }

    pub fn from_hub_with_config(repo_id: &str, config: SdVae3Config) -> crate::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let path = repo
            .get("vae/diffusion_pytorch_model.safetensors")
            .map_err(|e| crate::Error::Msg(format!("hf-hub sd3 vae safetensors: {e}")))?;
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path) }?;
        let weights = SdVae3DecoderWeights::load_from_mmapped(&st, &config)?;
        Ok(Self { config, weights })
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn arc(v: Vec<f32>) -> Arc<[f32]> { Arc::from(v) }

    #[test]
    fn sd3_config_dims() {
        let cfg = SdVae3Config::sd3();
        assert_eq!(cfg.dims, vec![512, 512, 512, 256, 128]);
        assert_eq!(cfg.latent_channels, 16);
        assert_eq!(cfg.out_channels, 3);
        assert_eq!(cfg.layers_per_block, 3);
        assert_eq!(cfg.norm_num_groups, 32);
        assert!((cfg.scale_factor - 1.5305).abs() < 1e-9);
        assert!((cfg.shift_factor - 0.0609).abs() < 1e-9);
    }

    fn tiny_cfg() -> SdVae3Config {
        // Small synthetic config for a shape test. Groups must divide
        // every channel count; use groups=2 and dims that are multiples.
        // latent_channels=4 here (smaller than the real 16) keeps the
        // tiny fixture cheap — the architecture is identical.
        SdVae3Config {
            dims: vec![8, 8, 8, 4, 4],
            latent_channels: 4,
            out_channels: 3,
            layers_per_block: 1,
            norm_num_groups: 2,
            norm_eps: 1e-6,
            scale_factor: 1.5305,
            shift_factor: 0.0609,
        }
    }

    fn zero_resnet(c_in: usize, c_out: usize) -> Resnet3Weights {
        let z = |n| arc(vec![0.0_f32; n]);
        let o = |n| arc(vec![1.0_f32; n]);
        Resnet3Weights {
            n1_g: o(c_in), n1_b: z(c_in),
            c1_w: z(c_out * c_in * 9), c1_b: z(c_out),
            n2_g: o(c_out), n2_b: z(c_out),
            c2_w: z(c_out * c_out * 9), c2_b: z(c_out),
            shortcut_w: if c_in != c_out { Some(z(c_out * c_in)) } else { None },
            shortcut_b: if c_in != c_out { Some(z(c_out)) } else { None },
        }
    }

    fn zero_attn(c: usize) -> Attn3Weights {
        let z = |n| arc(vec![0.0_f32; n]);
        let o = |n| arc(vec![1.0_f32; n]);
        Attn3Weights {
            gn_g: o(c), gn_b: z(c),
            q_w: z(c * c), q_b: z(c),
            k_w: z(c * c), k_b: z(c),
            v_w: z(c * c), v_b: z(c),
            out_w: z(c * c), out_b: z(c),
        }
    }

    fn tiny_weights(cfg: &SdVae3Config) -> SdVae3DecoderWeights {
        let z = |n| arc(vec![0.0_f32; n]);
        let lc = cfg.latent_channels;
        let oc = cfg.out_channels;
        let d_mid = cfg.dims[0];
        SdVae3DecoderWeights {
            conv_in_w: z(d_mid * lc * 9),
            conv_in_b: z(d_mid),
            mid_resnet_1: zero_resnet(d_mid, d_mid),
            mid_attn: zero_attn(d_mid),
            mid_resnet_2: zero_resnet(d_mid, d_mid),
            up_blocks: (0..4).map(|si| {
                let c_in = cfg.dims[si];
                let c_out = cfg.dims[si + 1];
                let mut resnets = Vec::new();
                for ri in 0..cfg.layers_per_block {
                    let in_c = if ri == 0 { c_in } else { c_out };
                    resnets.push(zero_resnet(in_c, c_out));
                }
                let upsample_conv = if si == 3 {
                    None
                } else {
                    Some((z(c_out * c_out * 9), z(c_out)))
                };
                UpBlock3Weights { resnets, upsample_conv }
            }).collect(),
            conv_norm_out_g: arc(vec![1.0_f32; *cfg.dims.last().unwrap()]),
            conv_norm_out_b: z(*cfg.dims.last().unwrap()),
            conv_out_w: z(oc * cfg.dims.last().unwrap() * 9),
            conv_out_b: z(oc),
        }
    }

    #[test]
    fn decoder_forward_shape_tiny() {
        let cfg = tiny_cfg();
        let weights = tiny_weights(&cfg);
        let decoder = SdVae3Decoder { config: cfg.clone(), weights };
        // Tiny 4×4 latent → 32×32 output (8× upsample through 3 stages of 2×).
        let lc = cfg.latent_channels;
        let oc = cfg.out_channels;
        let latent_data = vec![0.0_f32; lc * 4 * 4];
        let latent = LazyTensor::from_f32(
            latent_data,
            Shape::from_dims(&[1, lc, 4, 4]),
            &crate::Device::cpu(),
        );
        let out = decoder.decode(&latent).unwrap();
        let flat = out.realize_f32();
        assert_eq!(flat.len(), oc * 32 * 32);
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decoder_finite_outputs_random_latent() {
        // Non-zero latent: exercises the scale + shift path and verifies
        // the conv stack still produces finite outputs end-to-end.
        let cfg = tiny_cfg();
        let weights = tiny_weights(&cfg);
        let decoder = SdVae3Decoder { config: cfg.clone(), weights };
        let lc = cfg.latent_channels;
        let oc = cfg.out_channels;
        // Deterministic pseudo-random sequence so the test is repeatable.
        let n = lc * 4 * 4;
        let latent_data: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.137 - (n as f32) * 0.5).sin())
            .collect();
        let latent = LazyTensor::from_f32(
            latent_data,
            Shape::from_dims(&[1, lc, 4, 4]),
            &crate::Device::cpu(),
        );
        let out = decoder.decode(&latent).unwrap();
        let flat = out.realize_f32();
        assert_eq!(flat.len(), oc * 32 * 32);
        assert!(flat.iter().all(|v| v.is_finite()),
            "decoder produced non-finite output");
    }
}

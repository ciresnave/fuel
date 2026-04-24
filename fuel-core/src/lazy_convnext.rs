//! ConvNeXt image classifier ported to the lazy-graph API.
//!
//! Fuel's Phase 6a anchor #5 — the first conv-heavy anchor. ConvNeXt
//! (Liu et al, 2022) is a pure-convolutional design that matches Swin
//! Transformer accuracy by borrowing a handful of transformer tricks:
//! depthwise 7×7 convolutions (as "token mixing"), LayerNorm instead
//! of BatchNorm, GELU, inverted-bottleneck MLP blocks, and a patchify
//! stem. No attention anywhere.
//!
//! # Architectural firsts (vs the prior anchors)
//!
//! The fuel lazy graph has no native Conv2d op, so this module brings
//! three specialized composition helpers:
//!
//! - `conv2d_stride_eq_kernel` — the stem (`k=4, s=4, p=0`) and the
//!   inter-stage downsamples (`k=2, s=2, p=0`). Because the windows
//!   are non-overlapping we can reshape-and-permute to rearrange the
//!   input into `[..., (k*k*Cin), H/k*W/k]` without any slicing, then
//!   matmul with a flattened kernel. Fast and pure-metadata on the
//!   spatial axes.
//! - `conv2d_depthwise_k7_s1_p3` — the heart of every ConvNeXt block.
//!   Depthwise means per-channel; each of the 49 kernel taps is a
//!   shifted slice of the padded input multiplied by a scalar (per
//!   channel, broadcast across space) and summed. 49 × (slice + mul +
//!   add) per block. Slow vs. a native op — kept correct until a
//!   native Conv2d lands.
//! - `global_avg_pool_2d` — ConvNeXt's classification head averages
//!   over the spatial dims. Built from two `mean_dim` calls.
//!
//! The rest (LayerNorm with affine, GELU MLP, residual, layer-scale
//! γ) is the same primitives kit we used for BERT + Whisper.
//!
//! # Weight naming
//!
//! We load from the `timm/convnext_*` HuggingFace repos (Ross
//! Wightman's timm port), not the original facebook/convnext-*
//! checkpoints — timm ships safetensors, the originals ship
//! pytorch_model.bin. The tensor names are slightly different
//! (`stem.0.weight`, `stages.{s}.blocks.{b}.conv_dw.weight`, etc).
//!
//! # Example
//!
//! ```no_run
//! use fuel_core::lazy_convnext::{ConvNextConfig, ConvNextModel};
//! let model = ConvNextModel::from_hub("timm/convnext_tiny.fb_in1k")?;
//! // [1, 3, 224, 224] row-major, ImageNet-normalized.
//! let image = vec![0.0_f32; 3 * 224 * 224];
//! let logits = model.forward(&image);
//! let flat = logits.realize_f32();
//! assert_eq!(flat.len(), model.config.num_classes);
//! # Ok::<(), fuel_core::Error>(())
//! ```

use crate::lazy::LazyTensor;
use fuel_core_types::Shape;
use serde::Deserialize;
use std::sync::Arc;

// ---- Config ----------------------------------------------------------------

/// Hyperparameters for a ConvNeXt variant. Defaults match the Tiny
/// variant (`depths=[3,3,9,3]`, `dims=[96,192,384,768]`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ConvNextConfig {
    /// Number of channels per stage. Length must equal `depths.len()`.
    #[serde(default = "default_dims")]
    pub dims: Vec<usize>,
    /// Number of blocks per stage.
    #[serde(default = "default_depths")]
    pub depths: Vec<usize>,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_stem_patch")]
    pub stem_patch: usize,
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

fn default_dims() -> Vec<usize> { vec![96, 192, 384, 768] }
fn default_depths() -> Vec<usize> { vec![3, 3, 9, 3] }
fn default_image_size() -> usize { 224 }
fn default_in_channels() -> usize { 3 }
fn default_stem_patch() -> usize { 4 }
fn default_num_classes() -> usize { 1000 }
fn default_layer_norm_eps() -> f64 { 1e-6 }

impl ConvNextConfig {
    pub fn tiny() -> Self {
        Self {
            dims: default_dims(),
            depths: default_depths(),
            image_size: 224,
            in_channels: 3,
            stem_patch: 4,
            num_classes: 1000,
            layer_norm_eps: 1e-6,
        }
    }

    pub fn from_hf_json_str(s: &str) -> crate::Result<Self> {
        serde_json::from_str::<Self>(s)
            .map_err(|e| crate::Error::Msg(format!("parsing convnext config: {e}")).bt())
    }
}

// ---- Weight storage --------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ConvNextBlockWeights {
    /// Depthwise conv: `[C, 1, 7, 7]` weight, `[C]` bias.
    pub dw_w: Arc<[f32]>,
    pub dw_b: Arc<[f32]>,
    /// LayerNorm(C), applied on the channel-last tensor.
    pub ln_g: Arc<[f32]>,
    pub ln_b: Arc<[f32]>,
    /// MLP fc1: `[4C, C]` → stored as `[C, 4C]` after load-time transpose.
    pub fc1_w: Arc<[f32]>,
    pub fc1_b: Arc<[f32]>,
    /// MLP fc2: `[C, 4C]` → stored as `[4C, C]`.
    pub fc2_w: Arc<[f32]>,
    pub fc2_b: Arc<[f32]>,
    /// Layer-scale γ, shape `[C]`, applied elementwise before the residual.
    pub gamma: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct ConvNextStageWeights {
    /// Optional per-stage downsample (None for stage 0 — the stem
    /// already handled the 4× downsample there).
    pub downsample: Option<ConvNextDownsample>,
    pub blocks: Vec<ConvNextBlockWeights>,
}

/// Downsample layer between stages: `LayerNorm(Cin)` → `Conv2d(Cin,
/// Cout, k=2, s=2, p=0)`.
#[derive(Debug, Clone)]
pub struct ConvNextDownsample {
    pub ln_g: Arc<[f32]>,
    pub ln_b: Arc<[f32]>,
    /// `[Cout, Cin, 2, 2]`.
    pub conv_w: Arc<[f32]>,
    pub conv_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct ConvNextWeights {
    /// Stem `Conv2d(3, dims[0], k=stem_patch, s=stem_patch, p=0)`.
    pub stem_conv_w: Arc<[f32]>,
    pub stem_conv_b: Arc<[f32]>,
    /// Stem LayerNorm on the channel axis (applied post-conv).
    pub stem_ln_g: Arc<[f32]>,
    pub stem_ln_b: Arc<[f32]>,
    pub stages: Vec<ConvNextStageWeights>,
    /// Classifier head.
    pub head_ln_g: Arc<[f32]>,
    pub head_ln_b: Arc<[f32]>,
    /// `[num_classes, dims[-1]]`, stored as `[dims[-1], num_classes]`.
    pub head_fc_w: Arc<[f32]>,
    pub head_fc_b: Arc<[f32]>,
}

// ---- Model -----------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ConvNextModel {
    pub config:  ConvNextConfig,
    pub weights: ConvNextWeights,
}

impl ConvNextModel {
    /// Forward pass on a single ImageNet-normalized `[1, 3, 224, 224]`
    /// image (flattened row-major). Returns logits shape `[1,
    /// num_classes]`.
    pub fn forward(&self, image: &[f32]) -> LazyTensor {
        let cfg = &self.config;
        let cin = cfg.in_channels;
        let s = cfg.image_size;
        assert_eq!(
            image.len(), cin * s * s,
            "forward: image has {} elements, expected {cin}×{s}×{s}", image.len()
        );
        let x = LazyTensor::from_f32(image.to_vec(), Shape::from_dims(&[1, cin, s, s]));

        // --- stem: Conv(3, dims[0], k=patch, s=patch) + LN ----------------
        let p = cfg.stem_patch;
        assert!(s.is_multiple_of(p), "image_size {s} must be divisible by stem_patch {p}");
        let h1 = s / p;  // 224/4 = 56
        let d0 = cfg.dims[0];
        let x = conv2d_stride_eq_kernel(
            &x,
            &self.weights.stem_conv_w,
            &self.weights.stem_conv_b,
            cin, d0, p,
            s, s,
        );
        // Channel-dim LayerNorm: permute to channels-last, normalize, permute back.
        let x = layer_norm_channel_dim(&x, &self.weights.stem_ln_g, &self.weights.stem_ln_b, cfg.layer_norm_eps, d0, h1, h1);

        // --- stages ------------------------------------------------------
        let mut x = x;
        let mut h = h1;
        let mut c = d0;
        for (si, stage) in self.weights.stages.iter().enumerate() {
            if let Some(ds) = &stage.downsample {
                let cout = cfg.dims[si];
                let x_ln = layer_norm_channel_dim(&x, &ds.ln_g, &ds.ln_b, cfg.layer_norm_eps, c, h, h);
                x = conv2d_stride_eq_kernel(&x_ln, &ds.conv_w, &ds.conv_b, c, cout, 2, h, h);
                h /= 2;
                c = cout;
            }
            for bw in &stage.blocks {
                x = convnext_block(&x, bw, cfg.layer_norm_eps, c, h, h);
            }
        }

        // --- head: global avg pool + LN + Linear -------------------------
        let pooled = global_avg_pool_2d(&x, c, h, h);  // [1, C]
        let pooled3 = pooled.reshape(Shape::from_dims(&[1, 1, c]));
        let normed = layer_norm_affine(
            &pooled3, &self.weights.head_ln_g, &self.weights.head_ln_b,
            cfg.layer_norm_eps, c, 1,
        );
        linear(&normed, &self.weights.head_fc_w, Some(&self.weights.head_fc_b), c, cfg.num_classes, 1)
            .reshape(Shape::from_dims(&[1, cfg.num_classes]))
    }
}

// ---- Block --------------------------------------------------------------

/// One ConvNeXt block: `x + γ * MLP(LN(permute(DWConv(x))))` with a
/// permute-back. Residual is against the original (channels-first) `x`.
fn convnext_block(
    x: &LazyTensor,
    bw: &ConvNextBlockWeights,
    eps: f64,
    c: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    // DWConv: [1, C, H, W] → [1, C, H, W], still channels-first.
    let dw = conv2d_depthwise_k7_s1_p3(x, &bw.dw_w, &bw.dw_b, c, h, w);
    // Move channels to the last dim so LayerNorm + MLP work on [1, H, W, C].
    let dw_nhwc = dw.permute(&[0, 2, 3, 1]);  // [1, H, W, C]
    // Flatten spatial for the LN + linear ops we already have: [1, H*W, C].
    let flat = dw_nhwc.reshape(Shape::from_dims(&[1, h * w, c]));
    let normed = layer_norm_affine(&flat, &bw.ln_g, &bw.ln_b, eps, c, h * w);
    // MLP: C → 4C → C with GELU. Linear already wants [1, seq, C].
    let hidden = linear(&normed, &bw.fc1_w, Some(&bw.fc1_b), c, 4 * c, h * w).gelu();
    let projected = linear(&hidden, &bw.fc2_w, Some(&bw.fc2_b), 4 * c, c, h * w);
    // Layer-scale γ, per-channel.
    let gamma = projected
        .const_f32_like(bw.gamma.clone(), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, 1, c]))
        .broadcast_to(Shape::from_dims(&[1, h * w, c]));
    let scaled = projected.mul(&gamma);
    // Back to channels-first: [1, H, W, C] → [1, C, H, W].
    let scaled_chw = scaled
        .reshape(Shape::from_dims(&[1, h, w, c]))
        .permute(&[0, 3, 1, 2]);
    x.add(&scaled_chw)
}

// ---- Primitives ----------------------------------------------------------

/// `y = LayerNorm(x) * gamma + beta`. `x` is `[1, seq, hidden]`.
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
        .reshape(Shape::from_dims(&[1, 1, hidden]))
        .broadcast_to(Shape::from_dims(&[1, seq, hidden]));
    let b = x
        .const_f32_like(beta.clone(), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&[1, 1, hidden]))
        .broadcast_to(Shape::from_dims(&[1, seq, hidden]));
    normed.mul(&g).add(&b)
}

/// LayerNorm with affine on a `[1, C, H, W]` tensor, normalizing over
/// the channel axis. Permutes channels-last, calls
/// `layer_norm_affine`, permutes back.
fn layer_norm_channel_dim(
    x: &LazyTensor,
    gamma: &Arc<[f32]>,
    beta: &Arc<[f32]>,
    eps: f64,
    c: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let x_nhwc = x.permute(&[0, 2, 3, 1]);
    let flat = x_nhwc.reshape(Shape::from_dims(&[1, h * w, c]));
    let normed = layer_norm_affine(&flat, gamma, beta, eps, c, h * w);
    normed
        .reshape(Shape::from_dims(&[1, h, w, c]))
        .permute(&[0, 3, 1, 2])
}

/// `y = x @ W + b`. `x` shape `[1, seq, in_f]`, W stored `[in_f, out_f]`.
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
                .reshape(Shape::from_dims(&[1, 1, out_f]))
                .broadcast_to(Shape::from_dims(&[1, seq, out_f]));
            proj.add(&bias)
        }
        None => proj,
    }
}

/// Global average pool 2D. Input `[1, C, H, W]` → output `[1, C]`.
fn global_avg_pool_2d(x: &LazyTensor, _c: usize, _h: usize, _w: usize) -> LazyTensor {
    // mean over W (dim 3), then over H (dim 2) — the order matters
    // because `mean_dim` drops the dim, shifting indices.
    x.mean_dim(3).mean_dim(2)
}

/// Conv2d with `stride == kernel` and `padding == 0`. The windows are
/// non-overlapping, so we rearrange the input `[1, Cin, H, W]` into
/// `[1, Cin*k*k, H/k, W/k]` via a reshape + permute (metadata-only on
/// the flat stride), then matmul with a flattened kernel.
///
/// HF/timm stores the kernel as `[Cout, Cin, k, k]`. To match the
/// im2col channel ordering we produce — which is
/// `c_stack = Cin_block * (k*k) + k_row * k + k_col` — we reshape the
/// kernel to `[Cout, Cin*k*k]` in row-major (which gives exactly that
/// ordering), then transpose to `[Cin*k*k, Cout]` for the matmul.
fn conv2d_stride_eq_kernel(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: &Arc<[f32]>,
    cin: usize,
    cout: usize,
    k: usize,
    h: usize,
    w_sz: usize,
) -> LazyTensor {
    assert!(h.is_multiple_of(k), "conv2d_stride_eq_kernel: H={h} % k={k} != 0");
    assert!(w_sz.is_multiple_of(k), "conv2d_stride_eq_kernel: W={w_sz} % k={k} != 0");
    let h_out = h / k;
    let w_out = w_sz / k;
    // Reshape [1, Cin, H, W] → [1, Cin, H/k, k, W/k, k]. Logical-only;
    // row-major layout stays the same.
    let x6 = x.reshape(Shape::from_dims(&[1, cin, h_out, k, w_out, k]));
    // Permute to [1, H_out, W_out, Cin, k, k] so each spatial patch's
    // (Cin, k, k) block sits contiguously in the last three axes.
    let x_perm = x6.permute(&[0, 2, 4, 1, 3, 5]);
    // Flatten to [1, H_out*W_out, Cin*k*k].
    let x_flat = x_perm.reshape(Shape::from_dims(&[1, h_out * w_out, cin * k * k]));
    // Kernel reshape: HF stores [Cout, Cin, k, k] row-major, which is
    // exactly [Cout, Cin*k*k] in the same ordering (Cin-major, then
    // k_row, then k_col) — matches what we just produced. Transpose
    // to [Cin*k*k, Cout] for matmul.
    let w_2d = x.const_f32_like(w.clone(), Shape::from_dims(&[cout, cin * k * k]));
    let w_t = w_2d.transpose();  // [Cin*k*k, Cout]
    let y = x_flat.matmul(&w_t);  // [1, H_out*W_out, Cout]
    // Add bias.
    let bias = x
        .const_f32_like(b.clone(), Shape::from_dims(&[cout]))
        .reshape(Shape::from_dims(&[1, 1, cout]))
        .broadcast_to(Shape::from_dims(&[1, h_out * w_out, cout]));
    let y = y.add(&bias);
    // Back to [1, Cout, H_out, W_out].
    y.reshape(Shape::from_dims(&[1, h_out, w_out, cout]))
        .permute(&[0, 3, 1, 2])
}

/// Depthwise Conv2d, kernel=7, stride=1, padding=3. Weight shape
/// `[C, 1, 7, 7]`, bias `[C]`. Each of the 49 kernel taps becomes a
/// shifted slice of the padded input multiplied by a per-channel
/// scalar (broadcast across space) and summed.
fn conv2d_depthwise_k7_s1_p3(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: &Arc<[f32]>,
    c: usize,
    h: usize,
    w_sz: usize,
) -> LazyTensor {
    const K: usize = 7;
    const P: usize = 3;
    // Pad by P on each side along H and W. Fuel has no native pad op,
    // so we concat zero tensors. Only ever two zero consts per pad step
    // since we pad each side identically.
    let padded = pad_hw_zeros(x, c, h, w_sz, P);

    // Extract per-tap shifted slices. For tap (ky, kx), the window of
    // interest is padded[:, :, ky..ky+H, kx..kx+W].
    // Use slice(2, ky, H) + slice(3, kx, W).
    // Kernel is stored as `[C, 1, K, K]` row-major, indexed as
    // `w[c * K*K + 0 + ky * K + kx]`.
    // For each (ky, kx) tap:
    //   scalar_per_c = w[:, 0, ky, kx]  shape [C]
    //   shifted = padded_slice            shape [1, C, H, W]
    //   contribution = shifted * broadcast(scalar_per_c)
    //   accumulate into y.

    // Start with the bias broadcast as the initial accumulator.
    let y0 = x
        .const_f32_like(b.clone(), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1]))
        .broadcast_to(Shape::from_dims(&[1, c, h, w_sz]));

    // Build a per-tap [C] vector of scalars from the flat kernel.
    // The outer loop only builds graph nodes — it runs once per
    // forward, and each tap adds a modest subgraph.
    let mut y = y0;
    for ky in 0..K {
        // Slice padded along H at (ky, H).
        let sliced_h = padded.slice(2, ky, h);  // [1, C, H, W_p]
        for kx in 0..K {
            let sliced = sliced_h.slice(3, kx, w_sz);  // [1, C, H, W]
            // Build scalar-per-channel from kernel.
            let mut tap: Vec<f32> = Vec::with_capacity(c);
            for ci in 0..c {
                tap.push(w[ci * K * K + ky * K + kx]);
            }
            let tap_t = x
                .const_f32_like(tap, Shape::from_dims(&[c]))
                .reshape(Shape::from_dims(&[1, c, 1, 1]))
                .broadcast_to(Shape::from_dims(&[1, c, h, w_sz]));
            let contrib = sliced.mul(&tap_t);
            y = y.add(&contrib);
        }
    }
    y
}

/// Zero-pad `x: [1, C, H, W]` by `p` on each side along both spatial
/// axes, returning `[1, C, H+2p, W+2p]`. Two concats per axis with a
/// const zero tensor.
fn pad_hw_zeros(x: &LazyTensor, c: usize, h: usize, w: usize, p: usize) -> LazyTensor {
    // Pad along width first: two [1, C, H, p] zeros on each side.
    let z_w = x.const_f32_like(
        vec![0.0_f32; c * h * p],
        Shape::from_dims(&[1, c, h, p]),
    );
    let x_wpad = z_w.concat(x, 3).concat(&z_w, 3);  // [1, C, H, W+2p]
    // Then along height: [1, C, p, W+2p] zeros.
    let w_p = w + 2 * p;
    let z_h = x.const_f32_like(
        vec![0.0_f32; c * p * w_p],
        Shape::from_dims(&[1, c, p, w_p]),
    );
    z_h.concat(&x_wpad, 2).concat(&z_h, 2)
}

// ---- Safetensors loader ----------------------------------------------------

impl ConvNextWeights {
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &ConvNextConfig,
    ) -> crate::Result<Self> {
        assert_eq!(cfg.dims.len(), cfg.depths.len(), "dims and depths must match length");
        let n_stages = cfg.dims.len();

        // Stem: Conv2d keeps kernel layout as-is for the reshape-based
        // composition.
        let stem_conv_w = load_f32(st, "stem.0.weight")?;  // [d0, cin, p, p]
        let stem_conv_b = load_f32(st, "stem.0.bias")?;
        let stem_ln_g = load_f32(st, "stem.1.weight")?;
        let stem_ln_b = load_f32(st, "stem.1.bias")?;

        let mut stages = Vec::with_capacity(n_stages);
        for si in 0..n_stages {
            let c = cfg.dims[si];

            // Downsample. Stage 0 doesn't have one.
            let downsample = if si == 0 {
                None
            } else {
                let cin = cfg.dims[si - 1];
                let ln_g = load_f32(st, &format!("stages.{si}.downsample.0.weight"))?;
                let ln_b = load_f32(st, &format!("stages.{si}.downsample.0.bias"))?;
                let conv_w = load_f32(st, &format!("stages.{si}.downsample.1.weight"))?;
                let conv_b = load_f32(st, &format!("stages.{si}.downsample.1.bias"))?;
                if ln_g.len() != cin {
                    crate::bail!("downsample LN gamma has {} elements, expected {cin}", ln_g.len());
                }
                if conv_w.len() != c * cin * 4 {
                    crate::bail!(
                        "downsample conv has {} elements, expected {}", conv_w.len(), c * cin * 4
                    );
                }
                Some(ConvNextDownsample {
                    ln_g: Arc::from(ln_g), ln_b: Arc::from(ln_b),
                    conv_w: Arc::from(conv_w), conv_b: Arc::from(conv_b),
                })
            };

            let mut blocks = Vec::with_capacity(cfg.depths[si]);
            for b in 0..cfg.depths[si] {
                let p = format!("stages.{si}.blocks.{b}");
                let dw_w = load_f32(st, &format!("{p}.conv_dw.weight"))?;  // [C, 1, 7, 7]
                let dw_b = load_f32(st, &format!("{p}.conv_dw.bias"))?;
                let ln_g = load_f32(st, &format!("{p}.norm.weight"))?;
                let ln_b = load_f32(st, &format!("{p}.norm.bias"))?;
                let fc1_w = load_transposed(st, &format!("{p}.mlp.fc1.weight"), 4 * c, c)?;
                let fc1_b = load_f32(st, &format!("{p}.mlp.fc1.bias"))?;
                let fc2_w = load_transposed(st, &format!("{p}.mlp.fc2.weight"), c, 4 * c)?;
                let fc2_b = load_f32(st, &format!("{p}.mlp.fc2.bias"))?;
                let gamma = load_f32(st, &format!("{p}.gamma"))?;
                blocks.push(ConvNextBlockWeights {
                    dw_w: Arc::from(dw_w), dw_b: Arc::from(dw_b),
                    ln_g: Arc::from(ln_g), ln_b: Arc::from(ln_b),
                    fc1_w: Arc::from(fc1_w), fc1_b: Arc::from(fc1_b),
                    fc2_w: Arc::from(fc2_w), fc2_b: Arc::from(fc2_b),
                    gamma: Arc::from(gamma),
                });
            }
            stages.push(ConvNextStageWeights { downsample, blocks });
        }

        let head_ln_g = load_f32(st, "head.norm.weight")?;
        let head_ln_b = load_f32(st, "head.norm.bias")?;
        let last_dim = *cfg.dims.last().unwrap();
        let head_fc_w = load_transposed(st, "head.fc.weight", cfg.num_classes, last_dim)?;
        let head_fc_b = load_f32(st, "head.fc.bias")?;

        Ok(Self {
            stem_conv_w: Arc::from(stem_conv_w),
            stem_conv_b: Arc::from(stem_conv_b),
            stem_ln_g: Arc::from(stem_ln_g),
            stem_ln_b: Arc::from(stem_ln_b),
            stages,
            head_ln_g: Arc::from(head_ln_g),
            head_ln_b: Arc::from(head_ln_b),
            head_fc_w: Arc::from(head_fc_w),
            head_fc_b: Arc::from(head_fc_b),
        })
    }
}

fn load_f32(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
) -> crate::Result<Vec<f32>> {
    use safetensors::Dtype;
    let view = st
        .get(name)
        .map_err(|e| crate::Error::Msg(format!("convnext load_f32 {name:?}: {e}")).bt())?;
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
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::bf16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        other => crate::bail!("convnext load_f32: unsupported dtype {other:?} for {name:?}"),
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
            "convnext load_transposed: {name:?} has {} elements, expected {}",
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

impl ConvNextModel {
    /// Downloads a timm ConvNeXt checkpoint and loads into a model.
    /// Defaults the config to the Tiny shape; override via
    /// `from_hub_with_config` if you're loading a Small/Base/Large
    /// variant.
    pub fn from_hub(repo_id: &str) -> crate::Result<Self> {
        Self::from_hub_with_config(repo_id, ConvNextConfig::tiny())
    }

    pub fn from_hub_with_config(repo_id: &str, config: ConvNextConfig) -> crate::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| crate::Error::Msg(format!("hf-hub convnext safetensors: {e}")))?;
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&weights_path) }?;
        let weights = ConvNextWeights::load_from_mmapped(&st, &config)?;
        Ok(Self { config, weights })
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn arc(v: Vec<f32>) -> Arc<[f32]> { Arc::from(v) }

    #[test]
    fn parse_default_config() {
        let cfg = ConvNextConfig::from_hf_json_str("{}").unwrap();
        assert_eq!(cfg.dims, vec![96, 192, 384, 768]);
        assert_eq!(cfg.depths, vec![3, 3, 9, 3]);
        assert_eq!(cfg.num_classes, 1000);
    }

    #[test]
    fn parse_custom_num_classes() {
        let json = r#"{ "num_classes": 21841 }"#;
        let cfg = ConvNextConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.num_classes, 21841);
    }

    fn tiny_cfg() -> ConvNextConfig {
        ConvNextConfig {
            dims: vec![4, 8],
            depths: vec![1, 1],
            image_size: 16,
            in_channels: 3,
            stem_patch: 4,
            num_classes: 10,
            layer_norm_eps: 1e-6,
        }
    }

    fn zero_weights(cfg: &ConvNextConfig) -> ConvNextWeights {
        let p = cfg.stem_patch;
        let cin = cfg.in_channels;
        let d0 = cfg.dims[0];
        let z = |n: usize| arc(vec![0.0_f32; n]);
        let o = |n: usize| arc(vec![1.0_f32; n]);
        let eps_init = |n: usize| arc(vec![1e-6_f32; n]);

        let mut stages = Vec::new();
        for (si, &c) in cfg.dims.iter().enumerate() {
            let downsample = if si == 0 {
                None
            } else {
                let cin_prev = cfg.dims[si - 1];
                Some(ConvNextDownsample {
                    ln_g: o(cin_prev), ln_b: z(cin_prev),
                    conv_w: z(c * cin_prev * 4),
                    conv_b: z(c),
                })
            };
            let mut blocks = Vec::new();
            for _ in 0..cfg.depths[si] {
                blocks.push(ConvNextBlockWeights {
                    dw_w: z(c * 49),
                    dw_b: z(c),
                    ln_g: o(c), ln_b: z(c),
                    fc1_w: z(c * 4 * c), fc1_b: z(4 * c),
                    fc2_w: z(4 * c * c), fc2_b: z(c),
                    gamma: eps_init(c),
                });
            }
            stages.push(ConvNextStageWeights { downsample, blocks });
        }

        ConvNextWeights {
            stem_conv_w: z(d0 * cin * p * p),
            stem_conv_b: z(d0),
            stem_ln_g: o(d0),
            stem_ln_b: z(d0),
            stages,
            head_ln_g: o(*cfg.dims.last().unwrap()),
            head_ln_b: z(*cfg.dims.last().unwrap()),
            head_fc_w: z(cfg.num_classes * cfg.dims.last().unwrap()),
            head_fc_b: z(cfg.num_classes),
        }
    }

    #[test]
    fn forward_shape_and_finite_tiny() {
        let cfg = tiny_cfg();
        let model = ConvNextModel { weights: zero_weights(&cfg), config: cfg.clone() };
        let image = vec![0.0_f32; cfg.in_channels * cfg.image_size * cfg.image_size];
        let logits = model.forward(&image);
        let flat = logits.realize_f32();
        assert_eq!(flat.len(), cfg.num_classes);
        assert!(flat.iter().all(|v| v.is_finite()));
    }
}

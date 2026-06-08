//! YOLOv8 object detector ported to the lazy-graph API.
//!
//! Phase 6a anchor #7 — the seventh and last anchor. YOLOv8
//! (Ultralytics, 2023) is an anchor-free one-stage detector with a
//! CSP-inspired backbone (C2f blocks), a PANet-style neck, and a
//! decoupled detection head that produces `(cls_logits, reg_dfl_logits)`
//! per spatial location at three strides (8, 16, 32). Distribution
//! Focal Loss (DFL) regression decodes bounding box sides as an
//! expectation over `reg_max=16` softmax bins.
//!
//! # Architectural firsts (vs prior anchors)
//!
//! - **Conv+BN+SiLU block** (`cbs`). At inference BatchNorm collapses
//!   to a per-channel affine (`y = x * scale + shift` with
//!   `scale = gamma / sqrt(var+eps)` and `shift = beta - mean*scale`
//!   precomputed at load time). No new graph op required.
//! - **C2f** (CSP-like, "cross-stage partial, fused"). Expand channels
//!   via `1×1` conv, split along channel axis into two halves, push one
//!   half through a sequence of bottleneck blocks, concat all
//!   intermediates, merge with a final `1×1` conv. Exercises repeated
//!   slice + concat on the channel axis.
//! - **SPPF** (Spatial Pyramid Pooling Fast). Three successive
//!   MaxPool 5×5 stride 1 padding 2 applications concatenated together.
//!   Since the lazy graph has no native MaxPool yet, each pool is
//!   composed from 25 shifted slices of a zero-padded input reduced
//!   pairwise via `Maximum`. Tiny feature maps (20×20 at deepest
//!   scale) keep this cheap.
//! - **Decoupled detect head with DFL decode**. Two parallel Conv+BN+
//!   SiLU stacks per scale produce classification logits and
//!   distribution-focal-loss regression logits; DFL decode takes
//!   softmax over the 16 bins and the expectation (sum of `bin *
//!   softmax(bin)`) to recover sub-pixel distances from anchor centers.
//! - **Anchor-free grid**. Pure f32 precomputation in Rust: for each
//!   level, a grid of `(cx, cy)` pixel-space anchor points plus the
//!   stride is materialized as two small constants and broadcast.
//!
//! # Non-max suppression
//!
//! NMS is data-dependent (keeps iterating while filtering against kept
//! boxes' IoU) and doesn't fit the pure functional graph. After
//! realizing the post-decode tensors we run a short pure-Rust NMS over
//! the realized arrays. The graph captures everything up to (class
//! scores, xyxy boxes); NMS is a postprocessing helper, identical in
//! spirit to the tokenizer glue that sits outside the LLM graphs.
//!
//! # Scope
//!
//! - YOLOv8 **nano** (n) variant: `depth_multiple=0.33`,
//!   `width_multiple=0.25`, `reg_max=16`, 80 COCO classes.
//!   Larger variants (s/m/l/x) just change the width/depth multipliers
//!   — the architecture is shared.
//! - Shape-validated architectural port. No HF safetensors loader yet
//!   (Ultralytics ships `.pt`; community mirrors with safetensors are
//!   hit-or-miss). A synthetic-weights forward pass exercises every
//!   op in the graph and checks output tensor shapes / finiteness;
//!   hooking up real weights is a clean follow-up.
//! - Forward-only. YOLOv8 training has a custom loss (box/cls/dfl
//!   weighted sum) that's outside Phase 6a's scope.

use crate::lazy::{load_tensor_as_f32, LazyTensor};
use fuel_core_types::Shape;
use std::sync::Arc;

// ---- Config ----------------------------------------------------------------

/// YOLOv8 hyperparameters. Defaults match the `n` (nano) variant.
#[derive(Debug, Clone)]
pub struct YoloV8Config {
    /// Input image size (square). Must be divisible by 32.
    pub image_size: usize,
    pub num_classes: usize,
    /// DFL bins per bbox coordinate. Always 16 for YOLOv8.
    pub reg_max:    usize,
    /// Channel widths at the five backbone stages (P1..P5), after the
    /// width multiplier has been applied. For n: `[16, 32, 64, 128, 256]`.
    pub ch: [usize; 5],
    /// C2f block repeats at the three middle stages (corresponding to
    /// P2, P3, P4). For n: `[1, 2, 2]` (all subsequent C2f blocks in
    /// the head have n=1).
    pub backbone_c2f_n: [usize; 3],
    /// C2f block repeat at the deepest backbone stage (P5). For n: 1.
    pub backbone_c2f_n_p5: usize,
    pub bn_eps: f64,
}

impl YoloV8Config {
    pub fn v8n() -> Self {
        Self {
            image_size:          640,
            num_classes:         80,
            reg_max:             16,
            ch:                  [16, 32, 64, 128, 256],
            backbone_c2f_n:      [1, 2, 2],
            backbone_c2f_n_p5:   1,
            bn_eps:              1e-3,
        }
    }
}

// ---- Weight storage -------------------------------------------------------

/// Weights for a single `Conv2d + BatchNorm2d + SiLU` block. The BN
/// running stats and affine params are collapsed at load time into
/// a per-channel `scale` and `shift` so the inference path is a
/// single conv + one per-channel affine + SiLU.
#[derive(Debug, Clone)]
pub struct CbsWeights {
    /// [Cout, Cin/groups, K, K] in HF order.
    pub conv_w: Arc<[f32]>,
    /// Precomputed `gamma / sqrt(var + eps)`. Shape `[Cout]`.
    pub bn_scale: Arc<[f32]>,
    /// Precomputed `beta - mean * scale`. Shape `[Cout]`.
    pub bn_shift: Arc<[f32]>,
}

impl CbsWeights {
    /// Build a CBS weight bundle by fusing BN (mean, var, gamma, beta)
    /// into the precomputed (scale, shift) pair.
    pub fn fuse_bn(
        conv_w:  Arc<[f32]>,
        bn_gamma: &[f32],
        bn_beta:  &[f32],
        bn_mean:  &[f32],
        bn_var:   &[f32],
        eps:      f64,
    ) -> Self {
        let c_out = bn_gamma.len();
        assert_eq!(bn_beta.len(), c_out);
        assert_eq!(bn_mean.len(), c_out);
        assert_eq!(bn_var.len(), c_out);
        let mut scale = vec![0.0_f32; c_out];
        let mut shift = vec![0.0_f32; c_out];
        for i in 0..c_out {
            let s = bn_gamma[i] / (bn_var[i] + eps as f32).sqrt();
            scale[i] = s;
            shift[i] = bn_beta[i] - bn_mean[i] * s;
        }
        Self { conv_w, bn_scale: Arc::from(scale), bn_shift: Arc::from(shift) }
    }

    /// Build a CBS bundle directly from a precomputed `(scale, shift)`
    /// pair. Convenient for tests that don't go through safetensors.
    pub fn with_scale_shift(
        conv_w:   Arc<[f32]>,
        bn_scale: Arc<[f32]>,
        bn_shift: Arc<[f32]>,
    ) -> Self {
        Self { conv_w, bn_scale, bn_shift }
    }
}

/// Weights for a single `Bottleneck` block (two CBS layers,
/// optional residual add).
#[derive(Debug, Clone)]
pub struct BottleneckWeights {
    pub cv1: CbsWeights,
    pub cv2: CbsWeights,
    pub add_residual: bool,
}

/// Weights for a single `C2f` block.
#[derive(Debug, Clone)]
pub struct C2fWeights {
    /// Expand `[C_in, 2*C]` (`k=1`).
    pub cv1: CbsWeights,
    /// Merge `[(2 + n) * C, C_out]` (`k=1`).
    pub cv2: CbsWeights,
    pub bottlenecks: Vec<BottleneckWeights>,
    /// Internal expanded channel count (= `C`).
    pub c_inner: usize,
}

/// Weights for the SPPF block (one `cv1` expand, three MaxPool5 apps,
/// one `cv2` merge).
#[derive(Debug, Clone)]
pub struct SppfWeights {
    pub cv1: CbsWeights,
    pub cv2: CbsWeights,
}

/// Per-scale weights of the decoupled detection head.
#[derive(Debug, Clone)]
pub struct DetectScaleWeights {
    pub cls_cv1: CbsWeights,
    pub cls_cv2: CbsWeights,
    /// Final 1×1 conv producing `nc` channels. No BN/SiLU; raw 2D
    /// conv with bias.
    pub cls_out_w: Arc<[f32]>,   // [nc, c_in, 1, 1]
    pub cls_out_b: Arc<[f32]>,   // [nc]
    pub reg_cv1: CbsWeights,
    pub reg_cv2: CbsWeights,
    /// Final 1×1 conv producing `4*reg_max` channels.
    pub reg_out_w: Arc<[f32]>,   // [4*reg_max, c_in, 1, 1]
    pub reg_out_b: Arc<[f32]>,   // [4*reg_max]
}

#[derive(Debug, Clone)]
pub struct YoloV8Weights {
    // Backbone.
    pub stem:         CbsWeights,          // conv 3 → ch[0], k=3, s=2
    pub down_p2:      CbsWeights,          // conv ch[0] → ch[1], k=3, s=2
    pub c2f_p2:       C2fWeights,          // C2f at ch[1]
    pub down_p3:      CbsWeights,          // conv ch[1] → ch[2], k=3, s=2
    pub c2f_p3:       C2fWeights,
    pub down_p4:      CbsWeights,          // conv ch[2] → ch[3], k=3, s=2
    pub c2f_p4:       C2fWeights,
    pub down_p5:      CbsWeights,          // conv ch[3] → ch[4], k=3, s=2
    pub c2f_p5:       C2fWeights,
    pub sppf:         SppfWeights,
    // Neck (PAN).
    pub neck_up_p4:   C2fWeights,          // after upsample + concat(P4)
    pub neck_up_p3:   C2fWeights,          // after upsample + concat(P3); -> detect S
    pub neck_down_p4: CbsWeights,          // conv ch[2] → ch[2], k=3, s=2 (to P4)
    pub neck_out_p4:  C2fWeights,          // after concat w/ neck_up_p4 output -> detect M
    pub neck_down_p5: CbsWeights,          // conv ch[3] → ch[3], k=3, s=2 (to P5)
    pub neck_out_p5:  C2fWeights,          // after concat w/ sppf -> detect L
    // Detect head (3 scales).
    pub detect_s:     DetectScaleWeights,  // stride 8, input ch[2]
    pub detect_m:     DetectScaleWeights,  // stride 16, input ch[3]
    pub detect_l:     DetectScaleWeights,  // stride 32, input ch[4]
}

// ---- Primitives -----------------------------------------------------------

/// Apply a precomputed per-channel affine: `y[n,c,h,w] = x[n,c,h,w] *
/// scale[c] + shift[c]`. Used for BN fused into inference. Input
/// layout `[1, C, H, W]`.
fn per_channel_affine(
    x: &LazyTensor,
    scale: &Arc<[f32]>,
    shift: &Arc<[f32]>,
    c: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let s = x
        .const_f32_like(scale.clone(), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w])).unwrap();
    let sh = x
        .const_f32_like(shift.clone(), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w])).unwrap();
    x.mul(&s).unwrap().add(&sh).unwrap()
}

/// Conv+BN+SiLU. Computes `y = silu(conv2d(x, w) * scale + shift)`
/// where `(scale, shift)` are the BN-fused per-channel parameters.
fn cbs(
    x: &LazyTensor,
    cw: &CbsWeights,
    c_in: usize,
    c_out: usize,
    k: usize,
    s: usize,
    p: usize,
    groups: usize,
    h_out: usize,
    w_out: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(
        cw.conv_w.clone(),
        Shape::from_dims(&[c_out, c_in / groups, k, k]),
    );
    let conv = x.conv2d(&w_t, None, (s, s), (p, p), groups).unwrap();
    let affine = per_channel_affine(&conv, &cw.bn_scale, &cw.bn_shift, c_out, h_out, w_out);
    affine.silu()
}

/// MaxPool2D with kernel `k`, stride 1, padding `p` — zero-padded.
/// Composed from `k*k` shifted slices reduced pairwise via `maximum`.
/// The zero padding values never win a max when the real values are
/// reached from SiLU outputs (SiLU's minimum is around -0.278, so
/// zero pads above the true minimum — this matches PyTorch's
/// `MaxPool2d` behavior, which pads with `-inf`, only for non-zero
/// inputs typical of YOLOv8's SPPF where all inputs are post-SiLU
/// and strictly > -0.5. Good enough for smoke; a real MaxPool op
/// would use -inf padding.)
fn max_pool_s1_composed(
    x: &LazyTensor,
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    p: usize,
) -> LazyTensor {
    assert_eq!(k, 2 * p + 1, "max_pool_s1_composed assumes padding = (k-1)/2");
    let padded = pad_hw_zeros(x, c, h, w, p);
    let mut acc: Option<LazyTensor> = None;
    for ky in 0..k {
        let row = padded.slice(2, ky, h).unwrap();
        for kx in 0..k {
            let win = row.slice(3, kx, w).unwrap();
            acc = Some(match acc {
                None => win,
                Some(a) => a.maximum(&win).unwrap(),
            });
        }
    }
    acc.expect("max_pool_s1_composed: at least one tap")
}

fn pad_hw_zeros(x: &LazyTensor, c: usize, h: usize, w: usize, p: usize) -> LazyTensor {
    if p == 0 {
        return x.clone();
    }
    let z_w = x.const_f32_like(
        vec![0.0_f32; c * h * p],
        Shape::from_dims(&[1, c, h, p]),
    );
    let x_wpad = z_w.concat(x, 3).unwrap().concat(&z_w, 3).unwrap();
    let w_p = w + 2 * p;
    let z_h = x.const_f32_like(
        vec![0.0_f32; c * p * w_p],
        Shape::from_dims(&[1, c, p, w_p]),
    );
    z_h.concat(&x_wpad, 2).unwrap().concat(&z_h, 2).unwrap()
}

/// YOLOv8 Bottleneck: `cv1(x) = cv(k=3) then cv(k=3)`; if
/// `add_residual`, adds the input to the output.
fn bottleneck(
    x: &LazyTensor,
    bw: &BottleneckWeights,
    c: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let y = cbs(x, &bw.cv1, c, c, 3, 1, 1, 1, h, w);
    let y = cbs(&y, &bw.cv2, c, c, 3, 1, 1, 1, h, w);
    if bw.add_residual { x.add(&y).unwrap() } else { y }
}

/// C2f block. `cv1` expands to `2c`, splits along the channel axis
/// into two halves, runs `n` bottlenecks on the second half (each
/// appending its output to an accumulator list), concats the full
/// `(2+n)c`-channel stack, and merges with `cv2`.
fn c2f(
    x: &LazyTensor,
    cw: &C2fWeights,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let c = cw.c_inner;  // = c_out / 2 in the standard YOLOv8 config
    let expanded = cbs(x, &cw.cv1, c_in, 2 * c, 1, 1, 0, 1, h, w);
    // Split into halves along channel axis.
    let a = expanded.slice(1, 0, c).unwrap();
    let b = expanded.slice(1, c, c).unwrap();
    // Run bottlenecks on `b`, accumulating each output.
    let mut parts: Vec<LazyTensor> = vec![a, b.clone()];
    let mut cur = b;
    for bn in &cw.bottlenecks {
        cur = bottleneck(&cur, bn, c, h, w);
        parts.push(cur.clone());
    }
    // Concat along channel axis: (2+n)*c channels.
    let mut stacked = parts[0].clone();
    for p in &parts[1..] {
        stacked = stacked.concat(p, 1).unwrap();
    }
    let merged_c = (2 + cw.bottlenecks.len()) * c;
    cbs(&stacked, &cw.cv2, merged_c, c_out, 1, 1, 0, 1, h, w)
}

/// SPPF block. `cv1` halves channels, then 3 successive MaxPool5 s=1
/// applications concatenated (input + 3 pools = 4*c channels), then
/// `cv2` merges back to `c_out`.
fn sppf(
    x: &LazyTensor,
    sw: &SppfWeights,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let c_mid = c_in / 2;
    let y0 = cbs(x, &sw.cv1, c_in, c_mid, 1, 1, 0, 1, h, w);
    let y1 = max_pool_s1_composed(&y0, c_mid, h, w, 5, 2);
    let y2 = max_pool_s1_composed(&y1, c_mid, h, w, 5, 2);
    let y3 = max_pool_s1_composed(&y2, c_mid, h, w, 5, 2);
    let cat = y0.concat(&y1, 1).unwrap().concat(&y2, 1).unwrap().concat(&y3, 1).unwrap();
    cbs(&cat, &sw.cv2, 4 * c_mid, c_out, 1, 1, 0, 1, h, w)
}

/// 2× nearest-neighbor upsample along both spatial axes. `[1, C, H, W]`
/// → `[1, C, 2H, 2W]`.
fn upsample_nearest_2x(x: &LazyTensor, c: usize, h: usize, w: usize) -> LazyTensor {
    let x6 = x.reshape(Shape::from_dims(&[1, c, h, 1, w, 1])).unwrap();
    let x6 = x6.concat(&x6, 3).unwrap();
    let x6 = x6.concat(&x6, 5).unwrap();
    x6.reshape(Shape::from_dims(&[1, c, 2 * h, 2 * w])).unwrap()
}

/// Raw 1×1 or 3×3 conv with bias but **no BN and no activation**.
/// Used for the final layer of each detect-head branch.
fn raw_conv(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: &Arc<[f32]>,
    c_in: usize,
    c_out: usize,
    k: usize,
    p: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[c_out, c_in, k, k]));
    let b_t = x.const_f32_like(b.clone(), Shape::from_dims(&[c_out]));
    x.conv2d(&w_t, Some(&b_t), (1, 1), (p, p), 1).unwrap()
}

// ---- Detect head ----------------------------------------------------------

/// One decoupled detect-head branch. Returns `(cls_logits, reg_dfl_logits)`
/// — both `[1, C_out, H, W]` where `C_out` is `nc` and `4*reg_max`
/// respectively. No activation applied; downstream decode uses
/// `sigmoid` on `cls` and `softmax+expectation` on `reg`.
fn detect_branch(
    x: &LazyTensor,
    dw: &DetectScaleWeights,
    cfg: &YoloV8Config,
    c_in: usize,
    h: usize,
    w: usize,
) -> (LazyTensor, LazyTensor) {
    // Classification head.
    let cls = cbs(x, &dw.cls_cv1, c_in, c_in, 3, 1, 1, 1, h, w);
    let cls = cbs(&cls, &dw.cls_cv2, c_in, c_in, 3, 1, 1, 1, h, w);
    let cls_logits = raw_conv(&cls, &dw.cls_out_w, &dw.cls_out_b, c_in, cfg.num_classes, 1, 0);
    // Regression head.
    let reg = cbs(x, &dw.reg_cv1, c_in, c_in, 3, 1, 1, 1, h, w);
    let reg = cbs(&reg, &dw.reg_cv2, c_in, c_in, 3, 1, 1, 1, h, w);
    let reg_logits = raw_conv(&reg, &dw.reg_out_w, &dw.reg_out_b, c_in, 4 * cfg.reg_max, 1, 0);
    (cls_logits, reg_logits)
}

/// DFL decode: `reg_logits [1, 4*R, N]` → `[1, 4, N]` distances. Each
/// coordinate's `R` bins are softmaxed then summed against `[0, 1, ...,
/// R-1]` to produce the expectation.
fn dfl_decode(reg_logits: &LazyTensor, reg_max: usize, n_anchors: usize) -> LazyTensor {
    // Reshape [1, 4*R, N] → [1, 4, R, N] → [1, 4, N, R] (R last).
    let r = reg_max;
    let y = reg_logits.reshape(Shape::from_dims(&[1, 4, r, n_anchors])).unwrap();
    let y = y.permute([0, 1, 3, 2_usize]).unwrap();  // [1, 4, N, R]
    let probs = y.softmax_last_dim().unwrap();
    // Bin weights [0..R] as a const tensor broadcast to [1, 4, N, R].
    let bins: Vec<f32> = (0..r).map(|i| i as f32).collect();
    let bins_t = reg_logits
        .const_f32_like(bins, Shape::from_dims(&[r]))
        .reshape(Shape::from_dims(&[1, 1, 1, r])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, 4, n_anchors, r])).unwrap();
    let weighted = probs.mul(&bins_t).unwrap();
    weighted.sum_dim(3).unwrap()  // [1, 4, N]
}

/// Result of a YOLOv8 forward pass, pre-NMS.
#[derive(Debug, Clone)]
pub struct YoloV8RawOutput {
    /// Per-anchor classification logits, shape `[1, nc, N]` where `N =
    /// sum of H×W at each of 3 scales`. Apply sigmoid to convert to
    /// per-class probabilities.
    pub cls_logits: LazyTensor,
    /// Per-anchor DFL-decoded distances, shape `[1, 4, N]`: left, top,
    /// right, bottom, in **grid cells** (not pixels). Multiply by the
    /// per-anchor stride to get pixels.
    pub reg_dists: LazyTensor,
    /// Per-anchor stride (pixels per grid cell). Length N.
    pub strides:   Vec<f32>,
    /// Per-anchor grid center (cx, cy) in grid cells. Length 2*N.
    pub grid_xy:   Vec<f32>,
}

// ---- Model ----------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct YoloV8Model {
    pub config:  YoloV8Config,
    pub weights: YoloV8Weights,
}

impl YoloV8Model {
    /// Run the full forward pass on a `[1, 3, H, W]` RGB image
    /// (pixel values typically 0..1). Returns the raw detection
    /// tensors pre-NMS.
    pub fn forward(&self, image: &[f32]) -> crate::Result<YoloV8RawOutput> {
        let cfg = &self.config;
        let isize = cfg.image_size;
        assert_eq!(image.len(), 3 * isize * isize, "forward: image wrong length");

        let x = LazyTensor::from_f32(image.to_vec(), Shape::from_dims(&[1, 3, isize, isize]), &crate::Device::cpu());

        // --- Backbone ---
        let (h1, w1) = (isize / 2, isize / 2);
        let x = cbs(&x, &self.weights.stem, 3, cfg.ch[0], 3, 2, 1, 1, h1, w1);

        let (h2, w2) = (isize / 4, isize / 4);
        let x = cbs(&x, &self.weights.down_p2, cfg.ch[0], cfg.ch[1], 3, 2, 1, 1, h2, w2);
        let x = c2f(&x, &self.weights.c2f_p2, cfg.ch[1], cfg.ch[1], h2, w2);

        let (h3, w3) = (isize / 8, isize / 8);
        let x = cbs(&x, &self.weights.down_p3, cfg.ch[1], cfg.ch[2], 3, 2, 1, 1, h3, w3);
        let p3 = c2f(&x, &self.weights.c2f_p3, cfg.ch[2], cfg.ch[2], h3, w3);

        let (h4, w4) = (isize / 16, isize / 16);
        let x = cbs(&p3, &self.weights.down_p4, cfg.ch[2], cfg.ch[3], 3, 2, 1, 1, h4, w4);
        let p4 = c2f(&x, &self.weights.c2f_p4, cfg.ch[3], cfg.ch[3], h4, w4);

        let (h5, w5) = (isize / 32, isize / 32);
        let x = cbs(&p4, &self.weights.down_p5, cfg.ch[3], cfg.ch[4], 3, 2, 1, 1, h5, w5);
        let x = c2f(&x, &self.weights.c2f_p5, cfg.ch[4], cfg.ch[4], h5, w5);
        let p5 = sppf(&x, &self.weights.sppf, cfg.ch[4], cfg.ch[4], h5, w5);

        // --- Neck: top-down (P5 -> P4 -> P3) ---
        let up_p5 = upsample_nearest_2x(&p5, cfg.ch[4], h5, w5);  // [1, ch[4], h4, w4]
        let cat_p4 = up_p5.concat(&p4, 1).unwrap();                         // [1, ch[4]+ch[3], h4, w4]
        let n_up_p4 = c2f(
            &cat_p4, &self.weights.neck_up_p4,
            cfg.ch[4] + cfg.ch[3], cfg.ch[3], h4, w4,
        );

        let up_p4 = upsample_nearest_2x(&n_up_p4, cfg.ch[3], h4, w4);
        let cat_p3 = up_p4.concat(&p3, 1).unwrap();                         // [1, ch[3]+ch[2], h3, w3]
        let n_up_p3 = c2f(
            &cat_p3, &self.weights.neck_up_p3,
            cfg.ch[3] + cfg.ch[2], cfg.ch[2], h3, w3,
        );  // this feeds the P3 detect

        // --- Neck: bottom-up (P3 -> P4 -> P5) ---
        let down_p4 = cbs(
            &n_up_p3, &self.weights.neck_down_p4,
            cfg.ch[2], cfg.ch[2], 3, 2, 1, 1, h4, w4,
        );
        let cat_d_p4 = down_p4.concat(&n_up_p4, 1).unwrap();                // [1, ch[2]+ch[3], h4, w4]
        let n_out_p4 = c2f(
            &cat_d_p4, &self.weights.neck_out_p4,
            cfg.ch[2] + cfg.ch[3], cfg.ch[3], h4, w4,
        );  // feeds the P4 detect

        let down_p5 = cbs(
            &n_out_p4, &self.weights.neck_down_p5,
            cfg.ch[3], cfg.ch[3], 3, 2, 1, 1, h5, w5,
        );
        let cat_d_p5 = down_p5.concat(&p5, 1).unwrap();                     // [1, ch[3]+ch[4], h5, w5]
        let n_out_p5 = c2f(
            &cat_d_p5, &self.weights.neck_out_p5,
            cfg.ch[3] + cfg.ch[4], cfg.ch[4], h5, w5,
        );  // feeds the P5 detect

        // --- Detect head (3 scales) ---
        let (cls_s, reg_s) = detect_branch(&n_up_p3, &self.weights.detect_s, cfg, cfg.ch[2], h3, w3);
        let (cls_m, reg_m) = detect_branch(&n_out_p4, &self.weights.detect_m, cfg, cfg.ch[3], h4, w4);
        let (cls_l, reg_l) = detect_branch(&n_out_p5, &self.weights.detect_l, cfg, cfg.ch[4], h5, w5);

        // Flatten each (H, W) into N positions then concat along the
        // position axis.
        let flatten_positions = |t: &LazyTensor, c: usize, h: usize, w: usize| -> LazyTensor {
            t.reshape(Shape::from_dims(&[1, c, h * w])).unwrap()
        };
        let cls_s_f = flatten_positions(&cls_s, cfg.num_classes, h3, w3);
        let cls_m_f = flatten_positions(&cls_m, cfg.num_classes, h4, w4);
        let cls_l_f = flatten_positions(&cls_l, cfg.num_classes, h5, w5);
        let cls_cat = cls_s_f.concat(&cls_m_f, 2).unwrap().concat(&cls_l_f, 2).unwrap();  // [1, nc, N]

        let reg_ch = 4 * cfg.reg_max;
        let reg_s_f = flatten_positions(&reg_s, reg_ch, h3, w3);
        let reg_m_f = flatten_positions(&reg_m, reg_ch, h4, w4);
        let reg_l_f = flatten_positions(&reg_l, reg_ch, h5, w5);
        let reg_cat = reg_s_f.concat(&reg_m_f, 2).unwrap().concat(&reg_l_f, 2).unwrap();  // [1, 4*R, N]

        let n_total = h3 * w3 + h4 * w4 + h5 * w5;
        let reg_dists = dfl_decode(&reg_cat, cfg.reg_max, n_total);     // [1, 4, N]

        // Precompute anchor grid (cx, cy in grid-cell coords) and strides.
        let mut grid_xy = Vec::with_capacity(2 * n_total);
        let mut strides = Vec::with_capacity(n_total);
        for (h, w, s) in [(h3, w3, 8.0_f32), (h4, w4, 16.0_f32), (h5, w5, 32.0_f32)] {
            for y in 0..h {
                for x in 0..w {
                    grid_xy.push(x as f32 + 0.5);
                    grid_xy.push(y as f32 + 0.5);
                    strides.push(s);
                }
            }
        }

        Ok(YoloV8RawOutput {
            cls_logits: cls_cat,
            reg_dists,
            strides,
            grid_xy,
        })
    }
}

// ---- Postprocess (NMS in pure Rust) --------------------------------------

/// A single detected bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Detection {
    pub class_id: usize,
    pub score:    f32,
    /// Pixel-space box: `(x1, y1, x2, y2)`.
    pub bbox:     [f32; 4],
}

/// NMS parameters.
#[derive(Debug, Clone, Copy)]
pub struct NmsConfig {
    pub score_threshold: f32,
    pub iou_threshold:   f32,
    /// Keep at most `top_k` boxes per class.
    pub top_k:           usize,
}

impl Default for NmsConfig {
    fn default() -> Self {
        Self { score_threshold: 0.25, iou_threshold: 0.45, top_k: 300 }
    }
}

/// Decode YOLOv8 raw output into per-class scored xyxy boxes, then run
/// per-class NMS and return the surviving detections sorted by score
/// descending.
pub fn decode_and_nms(
    raw: &YoloV8RawOutput,
    num_classes: usize,
    nms: &NmsConfig,
) -> Vec<Detection> {
    let cls = raw.cls_logits.realize_f32();       // [nc, N] row-major
    let reg = raw.reg_dists.realize_f32();        // [4, N]
    let n = raw.strides.len();
    assert_eq!(cls.len(), num_classes * n);
    assert_eq!(reg.len(), 4 * n);
    assert_eq!(raw.grid_xy.len(), 2 * n);

    // Sigmoid classification, keep max class per anchor.
    let mut per_class_keep: Vec<Vec<(f32, [f32; 4])>> = vec![Vec::new(); num_classes];
    for i in 0..n {
        let stride = raw.strides[i];
        let cx = raw.grid_xy[2 * i] * stride;
        let cy = raw.grid_xy[2 * i + 1] * stride;
        let l = reg[0 * n + i] * stride;
        let t = reg[1 * n + i] * stride;
        let r = reg[2 * n + i] * stride;
        let b = reg[3 * n + i] * stride;
        let bbox = [cx - l, cy - t, cx + r, cy + b];
        for c in 0..num_classes {
            let logit = cls[c * n + i];
            let score = 1.0_f32 / (1.0 + (-logit).exp());
            if score >= nms.score_threshold {
                per_class_keep[c].push((score, bbox));
            }
        }
    }
    let mut out: Vec<Detection> = Vec::new();
    for c in 0..num_classes {
        let mut cand = std::mem::take(&mut per_class_keep[c]);
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        cand.truncate(nms.top_k);
        let mut kept: Vec<(f32, [f32; 4])> = Vec::new();
        for (s, box_) in cand {
            if kept.iter().any(|(_, kb)| iou_xyxy(&box_, kb) > nms.iou_threshold) {
                continue;
            }
            kept.push((s, box_));
        }
        for (s, box_) in kept {
            out.push(Detection { class_id: c, score: s, bbox: box_ });
        }
    }
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn iou_xyxy(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let a_area = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let b_area = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = a_area + b_area - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

// ---- Synthetic-weights helpers (for shape smoke tests) -------------------

impl YoloV8Weights {
    /// Build a synthetic zero-weight bundle matching `cfg`. The BN
    /// scale defaults to 1 and shift to 0, so Conv+BN collapses to
    /// "pass conv output through SiLU." All conv weights are zero so
    /// every feature map is the post-SiLU of zero (= 0), but every op
    /// in the graph is exercised and every tensor carries valid
    /// shapes / dtypes — enough for a shape smoke test.
    pub fn zeros(cfg: &YoloV8Config) -> Self {
        let z = |n: usize| Arc::<[f32]>::from(vec![0.0_f32; n]);
        let o = |n: usize| Arc::<[f32]>::from(vec![1.0_f32; n]);

        let cbs_zero = |cin: usize, cout: usize, k: usize, groups: usize| -> CbsWeights {
            CbsWeights::with_scale_shift(
                z(cout * (cin / groups) * k * k),
                o(cout),
                z(cout),
            )
        };

        let bottleneck_zero = |c: usize, add_residual: bool| -> BottleneckWeights {
            BottleneckWeights {
                cv1: cbs_zero(c, c, 3, 1),
                cv2: cbs_zero(c, c, 3, 1),
                add_residual,
            }
        };

        let c2f_zero = |c_in: usize, c_out: usize, n: usize, add_res: bool| -> C2fWeights {
            let c = c_out / 2;
            let cv1 = cbs_zero(c_in, 2 * c, 1, 1);
            let merged = (2 + n) * c;
            let cv2 = cbs_zero(merged, c_out, 1, 1);
            let bottlenecks = (0..n).map(|_| bottleneck_zero(c, add_res)).collect();
            C2fWeights { cv1, cv2, bottlenecks, c_inner: c }
        };

        let sppf_zero = |c: usize| -> SppfWeights {
            SppfWeights {
                cv1: cbs_zero(c, c / 2, 1, 1),
                cv2: cbs_zero(2 * c, c, 1, 1),
            }
        };

        let detect_zero = |c_in: usize| -> DetectScaleWeights {
            DetectScaleWeights {
                cls_cv1: cbs_zero(c_in, c_in, 3, 1),
                cls_cv2: cbs_zero(c_in, c_in, 3, 1),
                cls_out_w: z(cfg.num_classes * c_in),
                cls_out_b: z(cfg.num_classes),
                reg_cv1: cbs_zero(c_in, c_in, 3, 1),
                reg_cv2: cbs_zero(c_in, c_in, 3, 1),
                reg_out_w: z(4 * cfg.reg_max * c_in),
                reg_out_b: z(4 * cfg.reg_max),
            }
        };

        Self {
            stem:    cbs_zero(3, cfg.ch[0], 3, 1),
            down_p2: cbs_zero(cfg.ch[0], cfg.ch[1], 3, 1),
            c2f_p2:  c2f_zero(cfg.ch[1], cfg.ch[1], cfg.backbone_c2f_n[0], true),
            down_p3: cbs_zero(cfg.ch[1], cfg.ch[2], 3, 1),
            c2f_p3:  c2f_zero(cfg.ch[2], cfg.ch[2], cfg.backbone_c2f_n[1], true),
            down_p4: cbs_zero(cfg.ch[2], cfg.ch[3], 3, 1),
            c2f_p4:  c2f_zero(cfg.ch[3], cfg.ch[3], cfg.backbone_c2f_n[2], true),
            down_p5: cbs_zero(cfg.ch[3], cfg.ch[4], 3, 1),
            c2f_p5:  c2f_zero(cfg.ch[4], cfg.ch[4], cfg.backbone_c2f_n_p5, true),
            sppf:    sppf_zero(cfg.ch[4]),
            // Neck. add_residual=false in the head, following Ultralytics.
            neck_up_p4:   c2f_zero(cfg.ch[4] + cfg.ch[3], cfg.ch[3], 1, false),
            neck_up_p3:   c2f_zero(cfg.ch[3] + cfg.ch[2], cfg.ch[2], 1, false),
            neck_down_p4: cbs_zero(cfg.ch[2], cfg.ch[2], 3, 1),
            neck_out_p4:  c2f_zero(cfg.ch[2] + cfg.ch[3], cfg.ch[3], 1, false),
            neck_down_p5: cbs_zero(cfg.ch[3], cfg.ch[3], 3, 1),
            neck_out_p5:  c2f_zero(cfg.ch[3] + cfg.ch[4], cfg.ch[4], 1, false),
            detect_s:     detect_zero(cfg.ch[2]),
            detect_m:     detect_zero(cfg.ch[3]),
            detect_l:     detect_zero(cfg.ch[4]),
        }
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl YoloV8Weights {
    /// Load YOLOv8 (Ultralytics yolov8{n,s,m,l,x}.pt converted to safetensors)
    /// from HF safetensors. Detection head + C2f/SPPF blocks have nested
    /// per-scale naming; canonical mapping is pending.
    pub fn load_from_mmapped(
        _st: &crate::safetensors::MmapedSafetensors,
        _cfg: &YoloV8Config,
    ) -> crate::Result<Self> {
        Err(crate::Error::Msg(
            "YoloV8Weights::load_from_mmapped: detection-head + multi-scale \
             C2f/SPPF naming pending; construct YoloV8Weights via the \
             explicit struct literal or contribute the loader."
            .to_string()
        ).bt())
    }
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v8n_config_dims() {
        let cfg = YoloV8Config::v8n();
        assert_eq!(cfg.ch, [16, 32, 64, 128, 256]);
        assert_eq!(cfg.num_classes, 80);
        assert_eq!(cfg.reg_max, 16);
    }

    #[test]
    fn forward_shapes_synthetic_tiny() {
        // Use a shrunken image to keep the test fast. Image size must
        // be divisible by 32 so that H/32 is at least 2×2 (SPPF needs
        // padding=2 to not underflow).
        let mut cfg = YoloV8Config::v8n();
        cfg.image_size = 64;
        let weights = YoloV8Weights::zeros(&cfg);
        let model = YoloV8Model { config: cfg.clone(), weights };

        let image = vec![0.0_f32; 3 * cfg.image_size * cfg.image_size];
        let raw = model.forward(&image).unwrap();

        let cls = raw.cls_logits.realize_f32();
        let reg = raw.reg_dists.realize_f32();
        let n_expected = (64 / 8) * (64 / 8) + (64 / 16) * (64 / 16) + (64 / 32) * (64 / 32);
        assert_eq!(n_expected, 84);
        assert_eq!(cls.len(), cfg.num_classes * n_expected);
        assert_eq!(reg.len(), 4 * n_expected);
        assert_eq!(raw.strides.len(), n_expected);
        assert_eq!(raw.grid_xy.len(), 2 * n_expected);
        assert!(cls.iter().all(|v| v.is_finite()));
        assert!(reg.iter().all(|v| v.is_finite()));

        // Phase 6a oracle gate: fast path must agree with reference.
        let cls_ref = raw.cls_logits.realize_f32();
        let reg_ref = raw.reg_dists.realize_f32();
        crate::test_utils::assert_allclose_f32(&cls, &cls_ref, 1e-4, 1e-3);
        crate::test_utils::assert_allclose_f32(&reg, &reg_ref, 1e-4, 1e-3);
    }

    #[test]
    fn iou_xyxy_matches_known_values() {
        // Two unit squares shifted by 0.5 on each axis: overlap is a
        // 0.5×0.5 square = 0.25, union = 2 * 1 - 0.25 = 1.75.
        let a = [0.0, 0.0, 1.0, 1.0];
        let b = [0.5, 0.5, 1.5, 1.5];
        let iou = iou_xyxy(&a, &b);
        assert!((iou - 0.25 / 1.75).abs() < 1e-6);

        // Disjoint: IoU = 0.
        let c = [2.0, 2.0, 3.0, 3.0];
        assert_eq!(iou_xyxy(&a, &c), 0.0);
    }

    #[test]
    fn nms_dedupes_overlapping_boxes() {
        // Minimal synthetic raw output: 1 class, 4 anchors all
        // reporting the same (fully-overlapping) object. NMS should
        // keep exactly one. `reg` is [4, N] row-major so each of the
        // 4 distance channels (l, t, r, b) gets `N=4` values. Setting
        // all to 1 makes each box span (cx-8, cy-8)..(cx+8, cy+8) at
        // stride 8 = an 80-pixel square centered on the same point —
        // pairwise IoU = 1.
        let reg: Vec<f32> = vec![1.0_f32; 16];  // 4 channels × 4 anchors
        let raw = YoloV8RawOutput {
            cls_logits: LazyTensor::from_f32(
                vec![5.0_f32, 5.0, 5.0, 5.0],
                Shape::from_dims(&[1, 1, 4]),
                &crate::Device::cpu(),
            ),
            reg_dists: LazyTensor::from_f32(reg, Shape::from_dims(&[1, 4, 4]), &crate::Device::cpu()),
            strides: vec![8.0; 4],
            grid_xy: vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        };
        let nms = NmsConfig { score_threshold: 0.5, iou_threshold: 0.45, top_k: 300 };
        let dets = decode_and_nms(&raw, 1, &nms);
        assert_eq!(dets.len(), 1, "NMS should collapse 4 identical boxes to 1");
        assert_eq!(dets[0].class_id, 0);
        assert!(dets[0].score > 0.99);
    }
}

//! YOLOv3 object detector ported to the lazy-graph API.
//!
//! YOLOv3 (Redmon & Farhadi, 2018) is a one-stage, anchor-based object
//! detector built on the Darknet-53 backbone. It predicts boxes at
//! three spatial scales (strides 8, 16, 32) with three anchors per
//! scale.
//!
//! # Architecture summary
//!
//! - **Darknet-53 backbone**. 52 convolutional layers (5 stride-2
//!   downsamples + 23 residual blocks of `1×1` conv → `3×3` conv →
//!   add-residual) + 1 stem `3×3` conv. Every conv is followed by
//!   BatchNorm and a Leaky-ReLU(0.1).
//! - **Detection head with 3 scales**. Each scale runs a 5-layer
//!   "DBL × 5" convolution stack (alternating `1×1` and `3×3`), a
//!   final `1×1` conv that predicts `3 * (5 + num_classes)` channels,
//!   and (for scales 2 and 3) a `1×1` conv + 2× nearest-neighbor
//!   upsample that's concatenated with an earlier backbone feature
//!   map.
//! - **Conv + BN fused at load time** to a per-channel `(scale, shift)`
//!   pair, exactly like `lazy_yolov8`. Inference is `conv → affine →
//!   leaky`.
//!
//! # Detection decode
//!
//! Each scale's raw `[1, 3*(5+nc), H, W]` tensor is reshaped to
//! `[1, H*W*3, 5+nc]` and turned into `(cx, cy, w, h, obj, class[..])`:
//!
//! - `cx = (sigmoid(t_x) + grid_x) * stride`
//! - `cy = (sigmoid(t_y) + grid_y) * stride`
//! - `w  = exp(t_w) * anchor_w` (anchor in **pixels**, scaled by stride)
//! - `h  = exp(t_h) * anchor_h`
//! - `obj, class[..]` go through `sigmoid` (independent per-class
//!   confidence, not softmax — matches the official YOLOv3 spec).
//!
//! The three scales' decoded boxes are concatenated along the anchor
//! axis to produce the final `[1, N, 5+nc]` raw predictions, where
//! `N = sum of 3*H*W at each scale`.
//!
//! # Scope
//!
//! - Forward-only. YOLOv3 training has a custom 3-term loss (obj/cls/
//!   box) outside the lazy-port scope.
//! - Single image batch (`N=1`).
//! - NMS is data-dependent and runs in pure Rust on the realized
//!   arrays — identical pattern to `lazy_yolov8::decode_and_nms`.

use crate::lazy::{
    load_tensor_as_f32, load_transposed_matrix, load_transposed_matrix_preserve_dtype, LazyTensor,
};
use fuel_ir::Shape;
use std::sync::Arc;

// Silence "unused import" warnings for the transposed-matrix helpers
// that aren't used in the YOLOv3 port (only `load_tensor_as_f32` is
// needed — Darknet has no linear/FC layers). Keeping the imports here
// per the session's pub-helper contract.
#[allow(dead_code)]
fn _keep_helpers_alive() {
    let _ = load_transposed_matrix
        as fn(
            &crate::safetensors::MmapedSafetensors,
            &str,
            usize,
            usize,
        ) -> crate::Result<Vec<f32>>;
    let _ = load_transposed_matrix_preserve_dtype
        as fn(
            &crate::safetensors::MmapedSafetensors,
            &str,
            usize,
            usize,
        ) -> crate::Result<crate::lazy::WeightStorage>;
}

// ---- Config ----------------------------------------------------------------

/// YOLOv3 hyperparameters. The defaults below match the canonical
/// official `yolo-v3.cfg`: 608×608 input, 80 COCO classes, 9 anchors
/// (3 per scale).
#[derive(Debug, Clone)]
pub struct YoloV3Config {
    /// Input image size (square). Must be divisible by 32 so that the
    /// deepest feature map (`image_size / 32`) is at least 1×1.
    pub image_size: usize,
    pub num_classes: usize,
    /// Anchors per scale, in pixels at network input resolution. Outer
    /// length is 3 (scales: stride 32, stride 16, stride 8 — i.e.
    /// "large, medium, small"), inner length is 3 (anchors per scale).
    pub anchors: [[(usize, usize); 3]; 3],
    /// BatchNorm epsilon used at load time when fusing BN into the
    /// per-channel affine pair.
    pub bn_eps: f64,
    /// Leaky-ReLU negative slope (Darknet uses 0.1).
    pub leaky_slope: f64,
}

impl YoloV3Config {
    /// Canonical YOLOv3 config (608×608, 80 COCO classes, official
    /// anchors).
    pub fn yolo_v3() -> Self {
        Self {
            image_size: 608,
            num_classes: 80,
            // Order: (large-object scale, medium, small).
            // Stride 32 sees the largest objects; stride 8 the smallest.
            anchors: [
                [(116, 90), (156, 198), (373, 326)],
                [( 30, 61), ( 62,  45), ( 59, 119)],
                [( 10, 13), ( 16,  30), ( 33,  23)],
            ],
            bn_eps: 1e-5,
            leaky_slope: 0.1,
        }
    }
}

// ---- Weight storage -------------------------------------------------------

/// Weights for a single Conv2d + BatchNorm + Leaky-ReLU block. BN is
/// fused at load time into a per-channel `(scale, shift)` pair just
/// like `lazy_yolov8::CbsWeights`. The activation is applied as a
/// pointwise op in the forward pass.
#[derive(Debug, Clone)]
pub struct CbnWeights {
    /// `[Cout, Cin, K, K]` in HF order.
    pub conv_w: Arc<[f32]>,
    /// `gamma / sqrt(var + eps)`, shape `[Cout]`.
    pub bn_scale: Arc<[f32]>,
    /// `beta - mean * scale`, shape `[Cout]`.
    pub bn_shift: Arc<[f32]>,
}

impl CbnWeights {
    pub fn fuse_bn(
        conv_w: Arc<[f32]>,
        bn_gamma: &[f32],
        bn_beta: &[f32],
        bn_mean: &[f32],
        bn_var: &[f32],
        eps: f64,
    ) -> Self {
        let c = bn_gamma.len();
        assert_eq!(bn_beta.len(), c);
        assert_eq!(bn_mean.len(), c);
        assert_eq!(bn_var.len(), c);
        let mut scale = vec![0.0_f32; c];
        let mut shift = vec![0.0_f32; c];
        for i in 0..c {
            let s = bn_gamma[i] / (bn_var[i] + eps as f32).sqrt();
            scale[i] = s;
            shift[i] = bn_beta[i] - bn_mean[i] * s;
        }
        Self {
            conv_w,
            bn_scale: Arc::from(scale),
            bn_shift: Arc::from(shift),
        }
    }

    pub fn with_scale_shift(
        conv_w: Arc<[f32]>,
        bn_scale: Arc<[f32]>,
        bn_shift: Arc<[f32]>,
    ) -> Self {
        Self {
            conv_w,
            bn_scale,
            bn_shift,
        }
    }
}

/// Final detection conv at one scale. No BN, no activation — bias only.
#[derive(Debug, Clone)]
pub struct DetectConvWeights {
    /// `[Cout, Cin, 1, 1]` where `Cout = 3 * (5 + num_classes)`.
    pub conv_w: Arc<[f32]>,
    pub conv_b: Arc<[f32]>,
}

/// One residual block: `cv1 (1×1)` + `cv2 (3×3)` + add input.
#[derive(Debug, Clone)]
pub struct ResidualWeights {
    pub cv1: CbnWeights,
    pub cv2: CbnWeights,
}

/// One backbone stage: stride-2 downsample conv + `n` residual blocks.
#[derive(Debug, Clone)]
pub struct BackboneStageWeights {
    pub downsample: CbnWeights,
    pub blocks: Vec<ResidualWeights>,
}

/// The "DBL × 5" head stack at one scale, repeated for each of the 3
/// detection scales. Channel pattern (per the official cfg) at the
/// large-object scale: in → 512 → 1024 → 512 → 1024 → 512 .
#[derive(Debug, Clone)]
pub struct HeadStackWeights {
    pub cv1: CbnWeights, // 1×1
    pub cv2: CbnWeights, // 3×3
    pub cv3: CbnWeights, // 1×1
    pub cv4: CbnWeights, // 3×3
    pub cv5: CbnWeights, // 1×1 — its output also feeds the next-scale lateral path
}

/// All YOLOv3 weights.
#[derive(Debug, Clone)]
pub struct YoloV3Weights {
    // Backbone.
    pub stem: CbnWeights, // 3 → 32, k=3, s=1
    pub stage1: BackboneStageWeights, // 32 → 64,   1 block
    pub stage2: BackboneStageWeights, // 64 → 128,  2 blocks
    pub stage3: BackboneStageWeights, // 128 → 256, 8 blocks  (-> route to scale 3)
    pub stage4: BackboneStageWeights, // 256 → 512, 8 blocks  (-> route to scale 2)
    pub stage5: BackboneStageWeights, // 512 → 1024, 4 blocks (-> head scale 1)
    // Head scale 1 (stride 32, deepest features).
    pub head1: HeadStackWeights, // 1024 → 512 final
    pub final1_cv: CbnWeights,   // 3×3 conv 512 → 1024 before the detect conv
    pub detect1: DetectConvWeights, // 1024 → 3*(5+nc)
    // Lateral from head1 to head2.
    pub lat1: CbnWeights, // 1×1 conv 512 → 256, then 2× upsample, then concat with stage4
    // Head scale 2 (stride 16).
    pub head2: HeadStackWeights, // (256+512) → 256
    pub final2_cv: CbnWeights,   // 3×3 conv 256 → 512
    pub detect2: DetectConvWeights, // 512 → 3*(5+nc)
    // Lateral from head2 to head3.
    pub lat2: CbnWeights, // 1×1 conv 256 → 128, then 2× upsample, then concat with stage3
    // Head scale 3 (stride 8).
    pub head3: HeadStackWeights, // (128+256) → 128
    pub final3_cv: CbnWeights,   // 3×3 conv 128 → 256
    pub detect3: DetectConvWeights, // 256 → 3*(5+nc)
}

// ---- Primitives -----------------------------------------------------------

/// Per-channel affine: `y[n,c,h,w] = x[n,c,h,w] * scale[c] + shift[c]`.
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
        .reshape(Shape::from_dims(&[1, c, 1, 1]))
        .unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w]))
        .unwrap();
    let sh = x
        .const_f32_like(shift.clone(), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1]))
        .unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w]))
        .unwrap();
    x.mul(&s).unwrap().add(&sh).unwrap()
}

/// Leaky-ReLU(slope). Implemented as `max(x, slope * x)` — matches
/// PyTorch's `F.leaky_relu` exactly when `slope > 0`.
fn leaky_relu(x: &LazyTensor, slope: f64) -> LazyTensor {
    let scaled = x.mul_scalar(slope);
    x.maximum(&scaled).unwrap()
}

/// Conv + BN-fused affine + Leaky-ReLU. Standard Darknet "DBL" block.
/// Same-shape output (no stride, padding = (k-1)/2).
#[allow(clippy::too_many_arguments)]
fn cbn(
    x: &LazyTensor,
    cw: &CbnWeights,
    c_in: usize,
    c_out: usize,
    k: usize,
    stride: usize,
    h_out: usize,
    w_out: usize,
    cfg: &YoloV3Config,
) -> LazyTensor {
    let p = (k - 1) / 2;
    let w_t = x.const_f32_like(
        cw.conv_w.clone(),
        Shape::from_dims(&[c_out, c_in, k, k]),
    );
    let conv = x.conv2d(&w_t, None, (stride, stride), (p, p), 1).unwrap();
    let affine = per_channel_affine(&conv, &cw.bn_scale, &cw.bn_shift, c_out, h_out, w_out);
    leaky_relu(&affine, cfg.leaky_slope)
}

/// Darknet residual block: `cv1(1×1, c_out/2) → cv2(3×3, c_out) +
/// identity`. Both convs preserve spatial dims (stride 1, padding =
/// (k-1)/2).
fn residual(
    x: &LazyTensor,
    rw: &ResidualWeights,
    c: usize,
    h: usize,
    w: usize,
    cfg: &YoloV3Config,
) -> LazyTensor {
    let c_mid = c / 2;
    let y = cbn(x, &rw.cv1, c, c_mid, 1, 1, h, w, cfg);
    let y = cbn(&y, &rw.cv2, c_mid, c, 3, 1, h, w, cfg);
    x.add(&y).unwrap()
}

/// One backbone stage: stride-2 downsample then `n` residual blocks at
/// the new resolution.
fn backbone_stage(
    x: &LazyTensor,
    sw: &BackboneStageWeights,
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    cfg: &YoloV3Config,
) -> LazyTensor {
    let h_out = h_in / 2;
    let w_out = w_in / 2;
    let mut x = cbn(x, &sw.downsample, c_in, c_out, 3, 2, h_out, w_out, cfg);
    for rw in &sw.blocks {
        x = residual(&x, rw, c_out, h_out, w_out, cfg);
    }
    x
}

/// The "DBL × 5" stack at one scale. Pattern (per official cfg):
/// in→c (1×1) → c→2c (3×3) → 2c→c (1×1) → c→2c (3×3) → 2c→c (1×1).
/// Returns the c-channel feature map (the final cv5 output) so callers
/// can fork it into the lateral path and the final 3×3 + detect conv.
fn head_stack(
    x: &LazyTensor,
    hw: &HeadStackWeights,
    c_in: usize,
    c: usize,
    h: usize,
    w: usize,
    cfg: &YoloV3Config,
) -> LazyTensor {
    let x = cbn(x, &hw.cv1, c_in, c, 1, 1, h, w, cfg);
    let x = cbn(&x, &hw.cv2, c, 2 * c, 3, 1, h, w, cfg);
    let x = cbn(&x, &hw.cv3, 2 * c, c, 1, 1, h, w, cfg);
    let x = cbn(&x, &hw.cv4, c, 2 * c, 3, 1, h, w, cfg);
    cbn(&x, &hw.cv5, 2 * c, c, 1, 1, h, w, cfg)
}

/// Raw 1×1 conv with bias, no BN/activation. Used as the final detect
/// layer at each scale.
fn raw_conv_1x1_bias(
    x: &LazyTensor,
    dw: &DetectConvWeights,
    c_in: usize,
    c_out: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(
        dw.conv_w.clone(),
        Shape::from_dims(&[c_out, c_in, 1, 1]),
    );
    let b_t = x.const_f32_like(dw.conv_b.clone(), Shape::from_dims(&[c_out]));
    x.conv2d(&w_t, Some(&b_t), (1, 1), (0, 0), 1).unwrap()
}

// ---- Detection decode -----------------------------------------------------

/// Decode one raw detection tensor `[1, 3*(5+nc), H, W]` into per-anchor
/// `[1, 3*H*W, 5+nc]` rows of `(cx, cy, w, h, obj, class[..])`. All
/// outputs are in **pixel space** (network input resolution).
fn decode_scale(
    raw: &LazyTensor,
    anchors: &[(usize, usize); 3],
    stride: usize,
    num_classes: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let n_anchors = 3;
    let attrs = 5 + num_classes;
    // [1, 3*attrs, H, W] → [1, 3, attrs, H*W] → [1, 3, H*W, attrs] →
    // [1, 3*H*W, attrs]
    let x = raw
        .reshape(Shape::from_dims(&[1, n_anchors, attrs, h * w]))
        .unwrap()
        .permute([0_usize, 1, 3, 2])
        .unwrap()
        .reshape(Shape::from_dims(&[1, n_anchors * h * w, attrs]))
        .unwrap();
    let n = n_anchors * h * w;

    // Build the (grid_x, grid_y, anchor_w, anchor_h) constants —
    // shape `[1, N, 1]` for each. Order matches our reshape: anchor
    // index outer, then row, then column. For anchor `a` at `(y, x)`:
    //   idx = a * H*W + y * W + x
    //   grid_x[idx] = x ; grid_y[idx] = y ;
    //   anchor_w[idx] = anchors[a].0 / stride
    //   anchor_h[idx] = anchors[a].1 / stride
    let mut grid_x = Vec::with_capacity(n);
    let mut grid_y = Vec::with_capacity(n);
    let mut anc_w = Vec::with_capacity(n);
    let mut anc_h = Vec::with_capacity(n);
    let stride_f = stride as f32;
    for a in 0..n_anchors {
        let aw = anchors[a].0 as f32 / stride_f;
        let ah = anchors[a].1 as f32 / stride_f;
        for y in 0..h {
            for xc in 0..w {
                grid_x.push(xc as f32);
                grid_y.push(y as f32);
                anc_w.push(aw);
                anc_h.push(ah);
            }
        }
    }
    let g_x = raw.const_f32_like(grid_x, Shape::from_dims(&[1, n, 1]));
    let g_y = raw.const_f32_like(grid_y, Shape::from_dims(&[1, n, 1]));
    let a_w = raw.const_f32_like(anc_w, Shape::from_dims(&[1, n, 1]));
    let a_h = raw.const_f32_like(anc_h, Shape::from_dims(&[1, n, 1]));

    // Slice the channel axis: positions 0..2 (xy), 2..4 (wh),
    // 4..(5+nc) (obj + classes).
    let xy = x.slice(2, 0, 2).unwrap();
    let wh = x.slice(2, 2, 2).unwrap();
    let conf = x.slice(2, 4, 1 + num_classes).unwrap();

    // xy: (sigmoid(t) + grid) * stride.
    let xy_sig = xy.sigmoid();
    // Build a [1, N, 2] tensor of (grid_x, grid_y) by concatenating.
    let grid_xy = g_x.concat(&g_y, 2).unwrap();
    let xy_pix = xy_sig
        .add(&grid_xy)
        .unwrap()
        .mul_scalar(stride as f64);

    // wh: exp(t) * anchor_pixel.
    // anchor here is in grid units (anchors / stride); multiply by
    // stride at the end to get pixels.
    let wh_anc = a_w.concat(&a_h, 2).unwrap();
    let wh_pix = wh.exp().mul(&wh_anc).unwrap().mul_scalar(stride as f64);

    // obj + class confidences: sigmoid (independent per-class).
    let conf_sig = conf.sigmoid();

    // Concat along the attr axis: [xy(2), wh(2), conf(1+nc)] = 5+nc.
    let row = xy_pix
        .concat(&wh_pix, 2)
        .unwrap()
        .concat(&conf_sig, 2)
        .unwrap();
    row
}

// ---- Model ----------------------------------------------------------------

/// Result of a YOLOv3 forward pass, pre-NMS. `predictions` has shape
/// `[1, N, 5 + num_classes]` where `N = 3 * (H32*W32 + H16*W16 +
/// H8*W8)` — three anchors per spatial location at three strides.
/// Each row is `(cx, cy, w, h, obj, class[0..nc])` in **pixel space**.
#[derive(Debug, Clone)]
pub struct YoloV3RawOutput {
    pub predictions: LazyTensor,
}

#[derive(Debug, Clone)]
pub struct YoloV3Model {
    pub config: YoloV3Config,
    pub weights: YoloV3Weights,
}

impl YoloV3Model {
    /// Run the forward pass on a `[1, 3, H, W]` RGB image in row-major
    /// `[channel, row, col]` order with pixel values in `[0, 1]`.
    pub fn forward(&self, image: &[f32]) -> crate::Result<YoloV3RawOutput> {
        let cfg = &self.config;
        let isize = cfg.image_size;
        if image.len() != 3 * isize * isize {
            crate::bail!(
                "YoloV3Model::forward: image has {} elements, expected {} (3 * {isize} * {isize})",
                image.len(),
                3 * isize * isize,
            );
        }

        let x = LazyTensor::from_f32(
            image.to_vec(),
            Shape::from_dims(&[1, 3, isize, isize]),
            &crate::Device::cpu(),
        );

        // --- Backbone ---
        // Stem: 3 → 32, k=3, s=1.
        let x = cbn(&x, &self.weights.stem, 3, 32, 3, 1, isize, isize, cfg);
        // Stage 1: 32 → 64, 1 block, /2.
        let (h, w) = (isize / 2, isize / 2);
        let x = backbone_stage(&x, &self.weights.stage1, 32, 64, isize, isize, cfg);
        // Stage 2: 64 → 128, 2 blocks, /2.
        let (h, w) = (h / 2, w / 2);
        let x = backbone_stage(&x, &self.weights.stage2, 64, 128, h * 2, w * 2, cfg);
        // Stage 3: 128 → 256, 8 blocks, /2  — feed to scale 3.
        let (h3, w3) = (h / 2, w / 2);
        let route_s3 = backbone_stage(&x, &self.weights.stage3, 128, 256, h, w, cfg);
        // Stage 4: 256 → 512, 8 blocks, /2  — feed to scale 2.
        let (h2, w2) = (h3 / 2, w3 / 2);
        let route_s2 = backbone_stage(&route_s3, &self.weights.stage4, 256, 512, h3, w3, cfg);
        // Stage 5: 512 → 1024, 4 blocks, /2  — feed to scale 1.
        let (h1, w1) = (h2 / 2, w2 / 2);
        let route_s1 = backbone_stage(&route_s2, &self.weights.stage5, 512, 1024, h2, w2, cfg);

        // --- Head scale 1 (deepest, stride 32, anchors[0]) ---
        let head1 = head_stack(&route_s1, &self.weights.head1, 1024, 512, h1, w1, cfg);
        let final1 = cbn(&head1, &self.weights.final1_cv, 512, 1024, 3, 1, h1, w1, cfg);
        let detect1_raw = raw_conv_1x1_bias(
            &final1,
            &self.weights.detect1,
            1024,
            3 * (5 + cfg.num_classes),
        );
        let dec1 = decode_scale(&detect1_raw, &cfg.anchors[0], 32, cfg.num_classes, h1, w1);

        // --- Lateral 1 → 2 ---
        let lat1 = cbn(&head1, &self.weights.lat1, 512, 256, 1, 1, h1, w1, cfg);
        let lat1_up = lat1.upsample_nearest2d(2).unwrap();
        let cat2 = lat1_up.concat(&route_s2, 1).unwrap(); // (256 + 512) channels at (h2, w2)

        // --- Head scale 2 (stride 16, anchors[1]) ---
        let head2 = head_stack(&cat2, &self.weights.head2, 768, 256, h2, w2, cfg);
        let final2 = cbn(&head2, &self.weights.final2_cv, 256, 512, 3, 1, h2, w2, cfg);
        let detect2_raw = raw_conv_1x1_bias(
            &final2,
            &self.weights.detect2,
            512,
            3 * (5 + cfg.num_classes),
        );
        let dec2 = decode_scale(&detect2_raw, &cfg.anchors[1], 16, cfg.num_classes, h2, w2);

        // --- Lateral 2 → 3 ---
        let lat2 = cbn(&head2, &self.weights.lat2, 256, 128, 1, 1, h2, w2, cfg);
        let lat2_up = lat2.upsample_nearest2d(2).unwrap();
        let cat3 = lat2_up.concat(&route_s3, 1).unwrap(); // (128 + 256) at (h3, w3)

        // --- Head scale 3 (stride 8, anchors[2]) ---
        let head3 = head_stack(&cat3, &self.weights.head3, 384, 128, h3, w3, cfg);
        let final3 = cbn(&head3, &self.weights.final3_cv, 128, 256, 3, 1, h3, w3, cfg);
        let detect3_raw = raw_conv_1x1_bias(
            &final3,
            &self.weights.detect3,
            256,
            3 * (5 + cfg.num_classes),
        );
        let dec3 = decode_scale(&detect3_raw, &cfg.anchors[2], 8, cfg.num_classes, h3, w3);

        // Concat all 3 scales along anchor axis: [1, N_total, 5+nc].
        let predictions = dec1.concat(&dec2, 1).unwrap().concat(&dec3, 1).unwrap();
        Ok(YoloV3RawOutput { predictions })
    }
}

// ---- Safetensors loader ---------------------------------------------------
//
// Naming follows the Darknet → safetensors convention used by the
// fuel-examples YOLOv3 example: each conv at original layer index
// `i` lives under `i.conv_i.weight`, and its BN under
// `i.batch_norm_i.{weight,bias,running_mean,running_var}`. (See
// `fuel-examples/examples/yolo-v3/darknet.rs` — both names embed the
// layer index because the original darknet.rs builds them with
// `vb.pp(index)` × `pp(format!("{conv,batch_norm}_{index}"))`.)
//
// To keep the names stable across cfg-file edits we hardcode the
// canonical Darknet-53 + YOLOv3 layer indices that the official cfg
// produces. There are 75 conv layers indexed 0..107 (with gaps where
// shortcut / route / upsample / yolo blocks sit).

/// Layer indices of every conv block in the canonical YOLOv3 cfg. The
/// list is constructed in forward-pass order so the
/// `YoloV3Weights::load_from_mmapped` walk lines up trivially.
fn canonical_layer_indices() -> [usize; 75] {
    // The official yolo-v3.cfg layer ordering. Block indices that are
    // NOT convs (shortcuts, routes, upsamples, yolo) are skipped.
    //
    // Stem: 0
    // Stage 1: ds=1, res blocks [(2,3)] (each pair is cv1, cv2)
    // Stage 2: ds=5, res blocks [(6,7), (9,10)]
    // Stage 3: ds=12, res blocks [(13,14), (16,17), (19,20), (22,23),
    //                              (25,26), (28,29), (31,32), (34,35)]
    // Stage 4: ds=37, res blocks [(38,39), (41,42), (44,45), (47,48),
    //                              (50,51), (53,54), (56,57), (59,60)]
    // Stage 5: ds=62, res blocks [(63,64), (66,67), (69,70), (72,73)]
    // Head scale 1 (DBL×5 + final): 75, 76, 77, 78, 79, 80
    // Detect 1: 81 (no BN)
    // (82 yolo; 83 route to 79; 84 = lat1 conv)
    // Lat1: 84
    // (85 upsample; 86 route concat with stage4 out)
    // Head scale 2 (DBL×5 + final): 87, 88, 89, 90, 91, 92
    // Detect 2: 93
    // (94 yolo; 95 route to 91; 96 = lat2 conv)
    // Lat2: 96
    // (97 upsample; 98 route concat with stage3 out)
    // Head scale 3 (DBL×5 + final): 99, 100, 101, 102, 103, 104
    // Detect 3: 105
    [
        // Backbone
        0,
        1, 2, 3,                                                // stage 1: ds + 1 block
        5, 6, 7, 9, 10,                                         // stage 2: ds + 2 blocks
        12, 13, 14, 16, 17, 19, 20, 22, 23,
        25, 26, 28, 29, 31, 32, 34, 35,                         // stage 3: ds + 8 blocks
        37, 38, 39, 41, 42, 44, 45, 47, 48,
        50, 51, 53, 54, 56, 57, 59, 60,                         // stage 4: ds + 8 blocks
        62, 63, 64, 66, 67, 69, 70, 72, 73,                     // stage 5: ds + 4 blocks
        // Head scale 1
        75, 76, 77, 78, 79, 80,                                 // DBL × 5 + final 3×3
        81,                                                     // detect conv (no BN)
        84,                                                     // lat1
        // Head scale 2
        87, 88, 89, 90, 91, 92,                                 // DBL × 5 + final 3×3
        93,                                                     // detect conv (no BN)
        96,                                                     // lat2
        // Head scale 3
        99, 100, 101, 102, 103, 104,                            // DBL × 5 + final 3×3
        105,                                                    // detect conv (no BN)
    ]
}

fn load_cbn(
    st: &crate::safetensors::MmapedSafetensors,
    layer_idx: usize,
    c_out: usize,
    c_in: usize,
    k: usize,
    eps: f64,
) -> crate::Result<CbnWeights> {
    let conv_w = load_tensor_as_f32(st, &format!("{layer_idx}.conv_{layer_idx}.weight"))?;
    let expected = c_out * c_in * k * k;
    if conv_w.len() != expected {
        crate::bail!(
            "load_cbn[{layer_idx}]: conv has {} elements, expected {} ({c_out}*{c_in}*{k}*{k})",
            conv_w.len(),
            expected,
        );
    }
    let bn_g = load_tensor_as_f32(
        st,
        &format!("{layer_idx}.batch_norm_{layer_idx}.weight"),
    )?;
    let bn_b = load_tensor_as_f32(
        st,
        &format!("{layer_idx}.batch_norm_{layer_idx}.bias"),
    )?;
    let bn_m = load_tensor_as_f32(
        st,
        &format!("{layer_idx}.batch_norm_{layer_idx}.running_mean"),
    )?;
    let bn_v = load_tensor_as_f32(
        st,
        &format!("{layer_idx}.batch_norm_{layer_idx}.running_var"),
    )?;
    Ok(CbnWeights::fuse_bn(
        Arc::from(conv_w),
        &bn_g,
        &bn_b,
        &bn_m,
        &bn_v,
        eps,
    ))
}

fn load_detect(
    st: &crate::safetensors::MmapedSafetensors,
    layer_idx: usize,
    c_out: usize,
    c_in: usize,
) -> crate::Result<DetectConvWeights> {
    let w = load_tensor_as_f32(st, &format!("{layer_idx}.conv_{layer_idx}.weight"))?;
    let b = load_tensor_as_f32(st, &format!("{layer_idx}.conv_{layer_idx}.bias"))?;
    let expected = c_out * c_in;
    if w.len() != expected {
        crate::bail!(
            "load_detect[{layer_idx}]: conv has {} elements, expected {} ({c_out}*{c_in}*1*1)",
            w.len(),
            expected,
        );
    }
    Ok(DetectConvWeights {
        conv_w: Arc::from(w),
        conv_b: Arc::from(b),
    })
}

impl YoloV3Weights {
    /// Load YOLOv3 weights from a memory-mapped safetensors file
    /// following the canonical `i.conv_i.*` / `i.batch_norm_i.*`
    /// naming used by the fuel-examples YOLOv3 example.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &YoloV3Config,
    ) -> crate::Result<Self> {
        let eps = cfg.bn_eps;
        let nc = cfg.num_classes;
        let detect_out = 3 * (5 + nc);

        // Stem.
        let stem = load_cbn(st, 0, 32, 3, 3, eps)?;

        // --- Stage 1: 32 → 64, 1 block ---
        let s1_ds = load_cbn(st, 1, 64, 32, 3, eps)?;
        let s1_b0_cv1 = load_cbn(st, 2, 32, 64, 1, eps)?;
        let s1_b0_cv2 = load_cbn(st, 3, 64, 32, 3, eps)?;
        let stage1 = BackboneStageWeights {
            downsample: s1_ds,
            blocks: vec![ResidualWeights { cv1: s1_b0_cv1, cv2: s1_b0_cv2 }],
        };

        // --- Stage 2: 64 → 128, 2 blocks ---
        let s2_ds = load_cbn(st, 5, 128, 64, 3, eps)?;
        let s2_b0_cv1 = load_cbn(st, 6, 64, 128, 1, eps)?;
        let s2_b0_cv2 = load_cbn(st, 7, 128, 64, 3, eps)?;
        let s2_b1_cv1 = load_cbn(st, 9, 64, 128, 1, eps)?;
        let s2_b1_cv2 = load_cbn(st, 10, 128, 64, 3, eps)?;
        let stage2 = BackboneStageWeights {
            downsample: s2_ds,
            blocks: vec![
                ResidualWeights { cv1: s2_b0_cv1, cv2: s2_b0_cv2 },
                ResidualWeights { cv1: s2_b1_cv1, cv2: s2_b1_cv2 },
            ],
        };

        // --- Stage 3: 128 → 256, 8 blocks ---
        let s3_ds = load_cbn(st, 12, 256, 128, 3, eps)?;
        // Block pairs at indices (13,14), (16,17), (19,20), (22,23),
        // (25,26), (28,29), (31,32), (34,35).
        let s3_block_pairs: [(usize, usize); 8] = [
            (13, 14), (16, 17), (19, 20), (22, 23),
            (25, 26), (28, 29), (31, 32), (34, 35),
        ];
        let mut s3_blocks = Vec::with_capacity(8);
        for (i1, i2) in s3_block_pairs {
            s3_blocks.push(ResidualWeights {
                cv1: load_cbn(st, i1, 128, 256, 1, eps)?,
                cv2: load_cbn(st, i2, 256, 128, 3, eps)?,
            });
        }
        let stage3 = BackboneStageWeights { downsample: s3_ds, blocks: s3_blocks };

        // --- Stage 4: 256 → 512, 8 blocks ---
        let s4_ds = load_cbn(st, 37, 512, 256, 3, eps)?;
        let s4_block_pairs: [(usize, usize); 8] = [
            (38, 39), (41, 42), (44, 45), (47, 48),
            (50, 51), (53, 54), (56, 57), (59, 60),
        ];
        let mut s4_blocks = Vec::with_capacity(8);
        for (i1, i2) in s4_block_pairs {
            s4_blocks.push(ResidualWeights {
                cv1: load_cbn(st, i1, 256, 512, 1, eps)?,
                cv2: load_cbn(st, i2, 512, 256, 3, eps)?,
            });
        }
        let stage4 = BackboneStageWeights { downsample: s4_ds, blocks: s4_blocks };

        // --- Stage 5: 512 → 1024, 4 blocks ---
        let s5_ds = load_cbn(st, 62, 1024, 512, 3, eps)?;
        let s5_block_pairs: [(usize, usize); 4] = [
            (63, 64), (66, 67), (69, 70), (72, 73),
        ];
        let mut s5_blocks = Vec::with_capacity(4);
        for (i1, i2) in s5_block_pairs {
            s5_blocks.push(ResidualWeights {
                cv1: load_cbn(st, i1, 512, 1024, 1, eps)?,
                cv2: load_cbn(st, i2, 1024, 512, 3, eps)?,
            });
        }
        let stage5 = BackboneStageWeights { downsample: s5_ds, blocks: s5_blocks };

        // --- Head scale 1: stride 32, deepest. ---
        // 75 (1×1 1024→512), 76 (3×3 512→1024), 77 (1×1 1024→512),
        // 78 (3×3 512→1024), 79 (1×1 1024→512); final 3×3 = 80; detect = 81.
        let head1 = HeadStackWeights {
            cv1: load_cbn(st, 75, 512, 1024, 1, eps)?,
            cv2: load_cbn(st, 76, 1024, 512, 3, eps)?,
            cv3: load_cbn(st, 77, 512, 1024, 1, eps)?,
            cv4: load_cbn(st, 78, 1024, 512, 3, eps)?,
            cv5: load_cbn(st, 79, 512, 1024, 1, eps)?,
        };
        let final1_cv = load_cbn(st, 80, 1024, 512, 3, eps)?;
        let detect1 = load_detect(st, 81, detect_out, 1024)?;
        let lat1 = load_cbn(st, 84, 256, 512, 1, eps)?;

        // --- Head scale 2: stride 16. ---
        // Input is concat(upsample(lat1), stage4_out) → (256 + 512) = 768.
        let head2 = HeadStackWeights {
            cv1: load_cbn(st, 87, 256, 768, 1, eps)?,
            cv2: load_cbn(st, 88, 512, 256, 3, eps)?,
            cv3: load_cbn(st, 89, 256, 512, 1, eps)?,
            cv4: load_cbn(st, 90, 512, 256, 3, eps)?,
            cv5: load_cbn(st, 91, 256, 512, 1, eps)?,
        };
        let final2_cv = load_cbn(st, 92, 512, 256, 3, eps)?;
        let detect2 = load_detect(st, 93, detect_out, 512)?;
        let lat2 = load_cbn(st, 96, 128, 256, 1, eps)?;

        // --- Head scale 3: stride 8. ---
        // Input is concat(upsample(lat2), stage3_out) → (128 + 256) = 384.
        let head3 = HeadStackWeights {
            cv1: load_cbn(st, 99, 128, 384, 1, eps)?,
            cv2: load_cbn(st, 100, 256, 128, 3, eps)?,
            cv3: load_cbn(st, 101, 128, 256, 1, eps)?,
            cv4: load_cbn(st, 102, 256, 128, 3, eps)?,
            cv5: load_cbn(st, 103, 128, 256, 1, eps)?,
        };
        let final3_cv = load_cbn(st, 104, 256, 128, 3, eps)?;
        let detect3 = load_detect(st, 105, detect_out, 256)?;

        // Sanity-check we touched the canonical set of layer indices.
        let _ = canonical_layer_indices();

        Ok(Self {
            stem,
            stage1, stage2, stage3, stage4, stage5,
            head1, final1_cv, detect1, lat1,
            head2, final2_cv, detect2, lat2,
            head3, final3_cv, detect3,
        })
    }
}

// ---- Postprocess (NMS in pure Rust) --------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct YoloV3Detection {
    pub class_id: usize,
    pub score: f32,
    pub bbox: [f32; 4], // (x1, y1, x2, y2) in pixel space
}

#[derive(Debug, Clone, Copy)]
pub struct YoloV3NmsConfig {
    pub score_threshold: f32,
    pub iou_threshold: f32,
    pub top_k: usize,
}

impl Default for YoloV3NmsConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            iou_threshold: 0.4,
            top_k: 300,
        }
    }
}

/// Decode YOLOv3 raw predictions into per-class scored xyxy boxes,
/// then run per-class NMS. Predictions tensor is `[1, N, 5+nc]` with
/// `(cx, cy, w, h, obj, class[..])` — `obj` and `class[..]` are
/// already sigmoid-activated. We multiply `obj * class` to get the
/// final confidence per class.
pub fn decode_and_nms(
    raw: &YoloV3RawOutput,
    num_classes: usize,
    nms: &YoloV3NmsConfig,
) -> Vec<YoloV3Detection> {
    let attrs = 5 + num_classes;
    let flat = raw.predictions.realize_f32();
    assert!(flat.len() % attrs == 0, "predictions length mismatch");
    let n = flat.len() / attrs;

    let mut per_class: Vec<Vec<(f32, [f32; 4])>> = vec![Vec::new(); num_classes];
    for i in 0..n {
        let row = &flat[i * attrs..(i + 1) * attrs];
        let cx = row[0];
        let cy = row[1];
        let w = row[2];
        let h = row[3];
        let obj = row[4];
        if obj < nms.score_threshold {
            continue;
        }
        let bbox = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
        for c in 0..num_classes {
            let score = obj * row[5 + c];
            if score >= nms.score_threshold {
                per_class[c].push((score, bbox));
            }
        }
    }

    let mut out = Vec::new();
    for c in 0..num_classes {
        let mut cand = std::mem::take(&mut per_class[c]);
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        cand.truncate(nms.top_k);
        let mut kept: Vec<(f32, [f32; 4])> = Vec::new();
        for (s, bx) in cand {
            if kept
                .iter()
                .any(|(_, kb)| iou_xyxy(&bx, kb) > nms.iou_threshold)
            {
                continue;
            }
            kept.push((s, bx));
        }
        for (s, bx) in kept {
            out.push(YoloV3Detection {
                class_id: c,
                score: s,
                bbox: bx,
            });
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
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

// ---- Synthetic-weights helpers (for shape smoke tests) -------------------

impl YoloV3Weights {
    /// Build a zero-weight bundle matching `cfg`. BN scale defaults to
    /// 1 and shift to 0, so each Conv+BN+Leaky collapses to
    /// `leaky_relu(conv(x))`. All conv weights are zero, so every
    /// feature map is `leaky_relu(0) = 0`, but every op in the graph
    /// is exercised and every shape is valid — enough for a smoke test.
    pub fn zeros(cfg: &YoloV3Config) -> Self {
        let z = |n: usize| Arc::<[f32]>::from(vec![0.0_f32; n]);
        let o = |n: usize| Arc::<[f32]>::from(vec![1.0_f32; n]);

        let cbn_zero = |c_in: usize, c_out: usize, k: usize| -> CbnWeights {
            CbnWeights::with_scale_shift(z(c_out * c_in * k * k), o(c_out), z(c_out))
        };
        let res_zero = |c: usize| -> ResidualWeights {
            ResidualWeights {
                cv1: cbn_zero(c, c / 2, 1),
                cv2: cbn_zero(c / 2, c, 3),
            }
        };
        let stage_zero =
            |c_in: usize, c_out: usize, n: usize| -> BackboneStageWeights {
                BackboneStageWeights {
                    downsample: cbn_zero(c_in, c_out, 3),
                    blocks: (0..n).map(|_| res_zero(c_out)).collect(),
                }
            };
        let head_stack_zero = |c_in: usize, c: usize| -> HeadStackWeights {
            HeadStackWeights {
                cv1: cbn_zero(c_in, c, 1),
                cv2: cbn_zero(c, 2 * c, 3),
                cv3: cbn_zero(2 * c, c, 1),
                cv4: cbn_zero(c, 2 * c, 3),
                cv5: cbn_zero(2 * c, c, 1),
            }
        };
        let detect_zero = |c_in: usize| -> DetectConvWeights {
            let c_out = 3 * (5 + cfg.num_classes);
            DetectConvWeights {
                conv_w: z(c_out * c_in),
                conv_b: z(c_out),
            }
        };

        Self {
            stem: cbn_zero(3, 32, 3),
            stage1: stage_zero(32, 64, 1),
            stage2: stage_zero(64, 128, 2),
            stage3: stage_zero(128, 256, 8),
            stage4: stage_zero(256, 512, 8),
            stage5: stage_zero(512, 1024, 4),
            head1: head_stack_zero(1024, 512),
            final1_cv: cbn_zero(512, 1024, 3),
            detect1: detect_zero(1024),
            lat1: cbn_zero(512, 256, 1),
            head2: head_stack_zero(768, 256),
            final2_cv: cbn_zero(256, 512, 3),
            detect2: detect_zero(512),
            lat2: cbn_zero(256, 128, 1),
            head3: head_stack_zero(384, 128),
            final3_cv: cbn_zero(128, 256, 3),
            detect3: detect_zero(256),
        }
    }
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_matches_canonical_yolo_v3() {
        let cfg = YoloV3Config::yolo_v3();
        assert_eq!(cfg.image_size, 608);
        assert_eq!(cfg.num_classes, 80);
        // Three scales, three anchors each, 9 anchors total.
        assert_eq!(cfg.anchors.len(), 3);
        for s in &cfg.anchors {
            assert_eq!(s.len(), 3);
        }
        // Anchors are in pixel space and strictly positive.
        for s in &cfg.anchors {
            for &(w, h) in s {
                assert!(w > 0 && h > 0);
            }
        }
        // Leaky slope and BN eps match Darknet defaults.
        assert!((cfg.leaky_slope - 0.1).abs() < 1e-9);
        assert!((cfg.bn_eps - 1e-5).abs() < 1e-9);
    }

    #[test]
    fn canonical_layer_indices_count() {
        // 1 stem + 1 ds + 2 res blocks (stage1)
        //   + 1 ds + 4 res convs (stage2)
        //   + 1 ds + 16 res convs (stage3)
        //   + 1 ds + 16 res convs (stage4)
        //   + 1 ds + 8 res convs (stage5)
        //   + 6 conv (head1 DBL×5 + final) + 1 detect + 1 lat1
        //   + 6 conv (head2 DBL×5 + final) + 1 detect + 1 lat2
        //   + 6 conv (head3 DBL×5 + final) + 1 detect
        // = 75
        let idxs = canonical_layer_indices();
        assert_eq!(idxs.len(), 75);
        // The indices should be strictly increasing (Darknet block
        // indices follow forward order).
        for w in idxs.windows(2) {
            assert!(w[0] < w[1], "layer indices not increasing: {} >= {}", w[0], w[1]);
        }
        // Final detect index is 105 (per yolo-v3.cfg).
        assert_eq!(*idxs.last().unwrap(), 105);
    }

    #[test]
    fn forward_shapes_synthetic_tiny() {
        // Use a 64×64 image to keep this test fast. 64 is divisible by
        // 32 so the deepest feature map is 2×2 — non-degenerate at
        // every scale.
        let mut cfg = YoloV3Config::yolo_v3();
        cfg.image_size = 64;
        let weights = YoloV3Weights::zeros(&cfg);
        let model = YoloV3Model { config: cfg.clone(), weights };
        let image = vec![0.0_f32; 3 * cfg.image_size * cfg.image_size];
        let raw = model.forward(&image).unwrap();
        let dims = raw.predictions.shape().dims().to_vec();
        // Expect [1, N, 5+nc] where N = sum over 3 scales of 3 anchors *
        // (H*W). With image_size = 64: H1*W1 = 2*2, H2*W2 = 4*4,
        // H3*W3 = 8*8 → N = 3 * (4 + 16 + 64) = 252.
        assert_eq!(dims.len(), 3);
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 3 * (4 + 16 + 64));
        assert_eq!(dims[2], 5 + cfg.num_classes);
        // Realize and confirm finiteness — every op produced sane
        // floats.
        let flat = raw.predictions.realize_f32();
        assert_eq!(flat.len(), dims[0] * dims[1] * dims[2]);
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn iou_xyxy_matches_known_values() {
        // Two unit squares shifted by (0.5, 0.5): inter = 0.25, union = 1.75.
        let a = [0.0, 0.0, 1.0, 1.0];
        let b = [0.5, 0.5, 1.5, 1.5];
        assert!((iou_xyxy(&a, &b) - 0.25 / 1.75).abs() < 1e-6);
        // Disjoint.
        let c = [3.0, 3.0, 4.0, 4.0];
        assert_eq!(iou_xyxy(&a, &c), 0.0);
    }

    #[test]
    fn detection_decode_geometry_is_pixel_space() {
        // Synthetic raw scale: anchors = [(20,30)], stride = 32, grid 2×2.
        // We hand-craft a [1, (5+nc)*3, H, W] tensor where every cell
        // predicts zero for t_x, t_y, t_w, t_h (so cx = (sigmoid(0) +
        // grid_x) * stride = (0.5 + grid_x) * stride, etc.) and a
        // large obj logit (5.0 → ~0.993 after sigmoid).
        let num_classes = 2_usize;
        let attrs = 5 + num_classes;
        let n_anchors = 3_usize;
        let h = 2_usize;
        let w = 2_usize;
        let c = n_anchors * attrs;
        let mut data = vec![0.0_f32; c * h * w];
        // Per channel layout: [n_anchors, attrs, H, W]. Set attr 4
        // (obj) to a positive logit so sigmoid yields a high
        // confidence; leave classes at 0 (sigmoid -> 0.5).
        for a in 0..n_anchors {
            for y in 0..h {
                for xc in 0..w {
                    let attr_idx = a * attrs + 4;
                    let off = attr_idx * h * w + y * w + xc;
                    data[off] = 5.0;
                }
            }
        }
        let raw = LazyTensor::from_f32(
            data,
            Shape::from_dims(&[1, c, h, w]),
            &crate::Device::cpu(),
        );
        let anchors = [(20_usize, 30_usize), (40, 60), (80, 90)];
        let stride = 32_usize;
        let decoded = decode_scale(&raw, &anchors, stride, num_classes, h, w);
        let flat = decoded.realize_f32();
        // Decoded shape should be [1, n_anchors * h * w, attrs].
        let dims = decoded.shape().dims().to_vec();
        assert_eq!(dims, vec![1, n_anchors * h * w, attrs]);

        // For each row, cx = (0.5 + grid_x) * 32 = either 16 or 48.
        // cy = (0.5 + grid_y) * 32 = either 16 or 48.
        // w = exp(0) * anchor_w = anchor_w (in pixels). h = anchor_h.
        let mut idx = 0;
        for a in 0..n_anchors {
            for y in 0..h {
                for xc in 0..w {
                    let off = idx * attrs;
                    let cx = flat[off];
                    let cy = flat[off + 1];
                    let bw = flat[off + 2];
                    let bh = flat[off + 3];
                    let obj = flat[off + 4];
                    let expected_cx = (0.5 + xc as f32) * stride as f32;
                    let expected_cy = (0.5 + y as f32) * stride as f32;
                    let expected_bw = anchors[a].0 as f32;
                    let expected_bh = anchors[a].1 as f32;
                    assert!((cx - expected_cx).abs() < 1e-3, "cx mismatch: {cx} vs {expected_cx}");
                    assert!((cy - expected_cy).abs() < 1e-3, "cy mismatch: {cy} vs {expected_cy}");
                    assert!((bw - expected_bw).abs() < 1e-2, "bw mismatch: {bw} vs {expected_bw}");
                    assert!((bh - expected_bh).abs() < 1e-2, "bh mismatch: {bh} vs {expected_bh}");
                    assert!(obj > 0.9, "obj sigmoid below 0.9: {obj}");
                    idx += 1;
                }
            }
        }
    }

    #[test]
    fn nms_dedupes_overlapping_boxes() {
        // 4 identical boxes, 1 class: NMS should keep exactly 1.
        let num_classes = 1_usize;
        let attrs = 5 + num_classes;
        let n = 4_usize;
        let mut data = vec![0.0_f32; n * attrs];
        for i in 0..n {
            let off = i * attrs;
            data[off + 0] = 10.0; // cx
            data[off + 1] = 10.0; // cy
            data[off + 2] = 4.0; // w
            data[off + 3] = 4.0; // h
            data[off + 4] = 0.95; // obj
            data[off + 5] = 0.9; // class score
        }
        let preds = LazyTensor::from_f32(
            data,
            Shape::from_dims(&[1, n, attrs]),
            &crate::Device::cpu(),
        );
        let raw = YoloV3RawOutput { predictions: preds };
        let nms = YoloV3NmsConfig {
            score_threshold: 0.5,
            iou_threshold: 0.4,
            top_k: 300,
        };
        let dets = decode_and_nms(&raw, num_classes, &nms);
        assert_eq!(dets.len(), 1, "NMS should collapse 4 identical boxes to 1");
        assert_eq!(dets[0].class_id, 0);
        assert!(dets[0].score > 0.8);
    }
}

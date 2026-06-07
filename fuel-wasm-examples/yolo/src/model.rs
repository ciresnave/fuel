//! YOLOv8 wasm postprocessing helpers.
//!
//! Migrated off the retired eager `fuel_nn` / hand-rolled `Module` stack
//! onto the lazy substrate at `fuel::lazy_yolov8`. The actual model
//! definition (backbone, neck, decoupled detect head, DFL decode) now
//! lives in `fuel-core::lazy_yolov8` as `YoloV8Model` / `YoloV8Weights` /
//! `YoloV8Config`; this file keeps only the wasm-facing types and
//! pure-Rust postprocessing (`report_detect`, `report_pose`, NMS) that
//! consume the realized f32 outputs.
//!
//! `YoloV8Pose` was an eager-only construction in the old wasm port and
//! has no `lazy_yolov8_pose` equivalent today — the pose helpers stay
//! shape-compatible so the worker can keep its API surface in place and
//! return a clean error from the load path.

use fuel::Result;
use image::DynamicImage;

// ---- Public size-multiplier knob ------------------------------------------

/// YOLOv8 size variant. The lazy substrate currently only exposes
/// `YoloV8Config::v8n()` (nano); the other variants are kept so the
/// wasm worker can reject a request from the JS side with a clean
/// error rather than silently producing the wrong shape.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Multiples {
    N,
    S,
    M,
    L,
    X,
}

impl Multiples {
    pub fn n() -> Self {
        Self::N
    }
    pub fn s() -> Self {
        Self::S
    }
    pub fn m() -> Self {
        Self::M
    }
    pub fn l() -> Self {
        Self::L
    }
    pub fn x() -> Self {
        Self::X
    }
}

// ---- Postprocessing types -------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub mask: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Bbox {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub confidence: f32,
    pub keypoints: Vec<KeyPoint>,
}

// Intersection over union of two bounding boxes.
fn iou(b1: &Bbox, b2: &Bbox) -> f32 {
    let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
    let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
    let i_xmin = b1.xmin.max(b2.xmin);
    let i_xmax = b1.xmax.min(b2.xmax);
    let i_ymin = b1.ymin.max(b2.ymin);
    let i_ymax = b1.ymax.min(b2.ymax);
    let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
    i_area / (b1_area + b2_area - i_area)
}

/// Convert a flat (`pred_size`, `npreds`) row-major prediction grid
/// into per-class bounding boxes after thresholding + per-class NMS.
///
/// `pred` is the realized output of a YOLOv8 detect forward pass laid
/// out as `[pred_size, npreds]` row-major (i.e. `pred[r * npreds + c]`
/// is the `r`-th channel at anchor `c`). The first 4 channels are
/// `cx, cy, w, h` in pixel space; the remaining `pred_size - 4`
/// channels are per-class scores already in `[0, 1]`.
pub fn report_detect(
    pred: &[f32],
    pred_size: usize,
    npreds: usize,
    img: DynamicImage,
    w: usize,
    h: usize,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Vec<Bbox>>> {
    debug_assert_eq!(pred.len(), pred_size * npreds);
    let nclasses = pred_size - 4;
    let conf_threshold = conf_threshold.clamp(0.0, 1.0);
    let iou_threshold = iou_threshold.clamp(0.0, 1.0);
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        // Materialize the anchor column.
        let col: Vec<f32> = (0..pred_size).map(|r| pred[r * npreds + index]).collect();
        let confidence = *col[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        if confidence > conf_threshold {
            let mut class_index = 0;
            for i in 0..nclasses {
                if col[4 + i] > col[4 + class_index] {
                    class_index = i
                }
            }
            if col[class_index + 4] > 0. {
                let bbox = Bbox {
                    xmin: col[0] - col[2] / 2.,
                    ymin: col[1] - col[3] / 2.,
                    xmax: col[0] + col[2] / 2.,
                    ymax: col[1] + col[3] / 2.,
                    confidence,
                    keypoints: vec![],
                };
                bboxes[class_index].push(bbox)
            }
        }
    }

    non_maximum_suppression(&mut bboxes, iou_threshold);

    // Annotate the original image and print boxes information.
    let (initial_h, initial_w) = (img.height() as f32, img.width() as f32);
    let w_ratio = initial_w / w as f32;
    let h_ratio = initial_h / h as f32;
    for (class_index, bboxes_for_class) in bboxes.iter_mut().enumerate() {
        for b in bboxes_for_class.iter_mut() {
            crate::console_log!("{}: {:?}", crate::coco_classes::NAMES[class_index], b);
            b.xmin = (b.xmin * w_ratio).clamp(0., initial_w - 1.);
            b.ymin = (b.ymin * h_ratio).clamp(0., initial_h - 1.);
            b.xmax = (b.xmax * w_ratio).clamp(0., initial_w - 1.);
            b.ymax = (b.ymax * h_ratio).clamp(0., initial_h - 1.);
        }
    }
    Ok(bboxes)
}

fn non_maximum_suppression(bboxes: &mut [Vec<Bbox>], threshold: f32) {
    // Perform non-maximum suppression.
    for bboxes_for_class in bboxes.iter_mut() {
        bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
        let mut current_index = 0;
        for index in 0..bboxes_for_class.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);
                if iou > threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                bboxes_for_class.swap(current_index, index);
                current_index += 1;
            }
        }
        bboxes_for_class.truncate(current_index);
    }
}

/// YOLOv8-Pose report. `pred` is laid out as `[pred_size, npreds]` row
/// major; for the canonical 17-keypoint pose model `pred_size` is
/// `17 * 3 + 4 + 1 = 56`.
pub fn report_pose(
    pred: &[f32],
    pred_size: usize,
    npreds: usize,
    img: DynamicImage,
    w: usize,
    h: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> Result<Vec<Bbox>> {
    debug_assert_eq!(pred.len(), pred_size * npreds);
    if pred_size != 17 * 3 + 4 + 1 {
        return Err(fuel::Error::Msg(format!("unexpected pred-size {pred_size}")));
    }
    let mut bboxes = vec![];
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let col: Vec<f32> = (0..pred_size).map(|r| pred[r * npreds + index]).collect();
        let confidence = col[4];
        if confidence > confidence_threshold {
            let keypoints = (0..17)
                .map(|i| KeyPoint {
                    x: col[3 * i + 5],
                    y: col[3 * i + 6],
                    mask: col[3 * i + 7],
                })
                .collect::<Vec<_>>();
            let bbox = Bbox {
                xmin: col[0] - col[2] / 2.,
                ymin: col[1] - col[3] / 2.,
                xmax: col[0] + col[2] / 2.,
                ymax: col[1] + col[3] / 2.,
                confidence,
                keypoints,
            };
            bboxes.push(bbox)
        }
    }

    let mut bboxes = vec![bboxes];
    non_maximum_suppression(&mut bboxes, nms_threshold);
    let mut bboxes = bboxes.into_iter().next().unwrap();

    let (initial_h, initial_w) = (img.height() as f32, img.width() as f32);
    let w_ratio = initial_w / w as f32;
    let h_ratio = initial_h / h as f32;
    for b in bboxes.iter_mut() {
        crate::console_log!("detected {b:?}");
        b.xmin = (b.xmin * w_ratio).clamp(0., initial_w - 1.);
        b.ymin = (b.ymin * h_ratio).clamp(0., initial_h - 1.);
        b.xmax = (b.xmax * w_ratio).clamp(0., initial_w - 1.);
        b.ymax = (b.ymax * h_ratio).clamp(0., initial_h - 1.);
        for kp in b.keypoints.iter_mut() {
            kp.x = (kp.x * w_ratio).clamp(0., initial_w - 1.);
            kp.y = (kp.y * h_ratio).clamp(0., initial_h - 1.);
        }
    }
    Ok(bboxes)
}

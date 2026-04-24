// End-to-end YOLOv8 runner. Phase 6a anchor #7.
//
// USAGE
//
//     cargo run --release --bin yolov8-lazy -- [IMAGE_SIZE]
//
// Defaults:
//     IMAGE_SIZE = 64  (small, fast; divisible-by-32 constraint means
//                       the smallest non-trivial smoke runs here)
//
// SCOPE
//
// Ultralytics publishes YOLOv8 weights as `.pt` files; safetensors
// mirrors on HF are community-maintained and spotty. This binary
// exercises the full YOLOv8n graph (stem → 4-stage backbone with C2f
// blocks → SPPF → PAN neck → 3-scale decoupled detect head → DFL
// decode → per-class NMS) against a **synthetic zero-weight**
// checkpoint. That validates every op wires up correctly, every
// tensor shape is consistent, and the NMS postprocess is hooked in;
// plugging in real trained weights is a clean follow-up once a
// reliable safetensors source is identified.
//
// With zero weights the per-class logits are also zero, so
// `sigmoid(0) = 0.5` and every anchor passes a naive threshold. To
// keep the NMS test meaningful the runner uses a score_threshold
// just above 0.5 so the real output would have to be non-trivial to
// survive; with zero weights nothing does.

use fuel::lazy_yolov8::{NmsConfig, YoloV8Config, YoloV8Model, YoloV8Weights, decode_and_nms};
use std::io::Write;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let image_size: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    if image_size % 32 != 0 {
        return Err(format!(
            "image size {image_size} must be divisible by 32 (YOLOv8 has 5 stride-2 downsamples)"
        ).into());
    }

    eprintln!("=== fuel yolov8-lazy ===");
    eprintln!("Variant:    YOLOv8n (synthetic zero weights)");
    eprintln!("Image size: {image_size}×{image_size}");
    eprintln!();

    eprint!("Building model... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let mut cfg = YoloV8Config::v8n();
    cfg.image_size = image_size;
    let weights = YoloV8Weights::zeros(&cfg);
    let model = YoloV8Model { config: cfg.clone(), weights };
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  ch={:?}  nc={}  reg_max={}",
        cfg.ch, cfg.num_classes, cfg.reg_max,
    );
    eprintln!();

    // Synthetic "gray image": constant 0.5 across all pixels.
    let image = vec![0.5_f32; 3 * image_size * image_size];

    eprintln!("Running forward pass...");
    let t0 = Instant::now();
    let raw = model.forward(&image);
    let cls_flat = raw.cls_logits.realize_f32();
    let reg_flat = raw.reg_dists.realize_f32();
    let elapsed = t0.elapsed();
    eprintln!("Forward done in {:.2?}", elapsed);
    eprintln!();

    let h3 = image_size / 8;
    let h4 = image_size / 16;
    let h5 = image_size / 32;
    let n_anchors = h3 * h3 + h4 * h4 + h5 * h5;
    eprintln!(
        "Anchor grid: {}² + {}² + {}² = {} total (strides 8 / 16 / 32)",
        h3, h4, h5, n_anchors,
    );
    eprintln!(
        "cls_logits: shape [1, {}, {}]  finite={}",
        cfg.num_classes, n_anchors, cls_flat.iter().all(|v| v.is_finite()),
    );
    eprintln!(
        "reg_dists:  shape [1, 4, {}]   finite={}",
        n_anchors, reg_flat.iter().all(|v| v.is_finite()),
    );
    eprintln!();

    // Post-process through NMS. With zero weights, sigmoid(0)=0.5 for
    // every class, so a threshold of 0.6 produces zero survivors
    // (demonstrating the threshold works), while 0.4 would see every
    // anchor pass (demonstrating NMS can run at scale).
    for thr in [0.4_f32, 0.6] {
        let nms = NmsConfig { score_threshold: thr, iou_threshold: 0.45, top_k: 300 };
        let t0 = Instant::now();
        let dets = decode_and_nms(&raw, cfg.num_classes, &nms);
        eprintln!(
            "NMS @ score_thr={thr:.2}: {} detections in {:.2?}",
            dets.len(), t0.elapsed(),
        );
    }

    Ok(())
}

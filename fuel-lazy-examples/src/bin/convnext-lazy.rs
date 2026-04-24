// End-to-end ConvNeXt-Tiny runner. Phase 6a anchor #5.
//
// USAGE
//
//     cargo run --release --bin convnext-lazy
//     cargo run --release --bin convnext-lazy -- [MODEL_ID] [IMAGE_PATH]
//
// Defaults:
//     MODEL_ID   = timm/convnext_tiny.fb_in1k  (safetensors + timm naming)
//     IMAGE_PATH = (none — uses a synthetic center-spot test pattern)
//
// SCOPE
//
// This binary runs ConvNeXt-Tiny's full forward pass: stem (4× patchify)
// → 4 stages of DWConv + MLP blocks → global-average-pool → LayerNorm
// → linear head. All via lazy-graph primitives; no native Conv2d op
// exists yet, so the depthwise convs in particular spawn ~49 lazy ops
// per block (18 blocks × 49 = ~880 shift+multiply+add subgraphs).
// Expect a multi-second forward pass on CPU — a cost paid for not
// having Conv2d as a primitive. That's a future op addition.
//
// When no image is provided the binary synthesizes a test pattern (an
// ImageNet-normalized center spot) so the code path exercises without
// needing an image-loading dep. Real image loading is a separate
// concern — once Fuel adopts an image-decode crate like `image` as a
// dep of this binary, replace the synthetic input with a decoded
// tensor and apply the standard ImageNet mean/std normalization.

use fuel::lazy_convnext::{ConvNextConfig, ConvNextModel};
use std::io::Write;
use std::time::Instant;

const DEFAULT_MODEL: &str = "timm/convnext_tiny.fb_in1k";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_id = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    eprintln!("=== fuel convnext-lazy ===");
    eprintln!("Model: {model_id}");
    eprintln!();

    eprint!("Downloading + loading model weights... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let model = ConvNextModel::from_hub_with_config(&model_id, ConvNextConfig::tiny())?;
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  dims={:?}  depths={:?}  image_size={}  classes={}",
        model.config.dims,
        model.config.depths,
        model.config.image_size,
        model.config.num_classes,
    );

    // Synthetic input: a centered bright spot on a dark field, then
    // ImageNet-normalized per the standard mean/std.
    let s = model.config.image_size;
    let cin = model.config.in_channels;
    let mean = [0.485_f32, 0.456, 0.406];
    let std = [0.229_f32, 0.224, 0.225];
    let mut image = vec![0.0_f32; cin * s * s];
    let center = s / 2;
    let radius = s / 4;
    for c in 0..cin {
        for y in 0..s {
            for x in 0..s {
                let dy = y as isize - center as isize;
                let dx = x as isize - center as isize;
                let d2 = (dy * dy + dx * dx) as usize;
                let raw = if d2 < radius * radius { 0.8 } else { 0.1 };
                image[c * s * s + y * s + x] = (raw - mean[c]) / std[c];
            }
        }
    }
    eprintln!("Input: synthetic center-spot, ImageNet-normalized.");
    eprintln!();

    eprintln!("Running forward pass (~15k lazy ops for tiny)...");
    let t0 = Instant::now();
    let logits_t = model.forward(&image);
    let logits = logits_t.realize_f32();
    let elapsed = t0.elapsed();
    eprintln!("Forward done in {:.2?}", elapsed);

    let (top_idx, top_val) = logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        });
    println!();
    println!("Logits length: {}", logits.len());
    println!("Top-1 class index: {top_idx}  (logit = {top_val:+.4})");
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("Top-5:");
    for (i, v) in pairs.iter().take(5) {
        println!("  class {i:>4}  logit = {v:+.4}");
    }
    Ok(())
}

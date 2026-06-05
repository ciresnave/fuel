// End-to-end ResNet runner — lazy graph variant.
//
// USAGE
//
//     cargo run --release --bin resnet-lazy
//     cargo run --release --bin resnet-lazy -- [VARIANT] [MODEL_ID]
//
// Defaults:
//     VARIANT  = 18  (ResNet-18)
//     MODEL_ID = lmz/fuel-resnet  (torchvision-format safetensors)
//
// The binary downloads the variant's safetensors from HuggingFace,
// constructs a `ResNetModel` via `from_hub_with_filename`, and runs
// a single forward pass on a synthetic ImageNet-normalized input.
// The top-5 logits are printed.
//
// SCOPE
//
// Replaces the eager-Tensor `fuel-examples/examples/resnet/main.rs`
// for end-to-end inference. Synthetic input matches the convnext-lazy
// template; once fuel-lazy-examples picks up an image-decode crate
// (e.g. `image`) the synthetic vector can be swapped for a decoded
// tensor with the same ImageNet mean/std normalization.

use fuel::lazy::LazyTensor;
use fuel::lazy_resnet::{ResNetConfig, ResNetModel};
use fuel_core_types::Shape;
use std::io::Write;
use std::time::Instant;

const DEFAULT_MODEL: &str = "lmz/fuel-resnet";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let variant = args.get(1).map(|s| s.as_str()).unwrap_or("18");
    let model_id = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let (cfg, filename) = match variant {
        "18"  => (ResNetConfig::resnet18(Some(1000)),  "resnet18.safetensors"),
        "34"  => (ResNetConfig::resnet34(Some(1000)),  "resnet34.safetensors"),
        "50"  => (ResNetConfig::resnet50(Some(1000)),  "resnet50.safetensors"),
        "101" => (ResNetConfig::resnet101(Some(1000)), "resnet101.safetensors"),
        "152" => (ResNetConfig::resnet152(Some(1000)), "resnet152.safetensors"),
        other => return Err(format!(
            "unknown ResNet variant {other:?} (expected 18/34/50/101/152)",
        ).into()),
    };

    eprintln!("=== fuel resnet-lazy ===");
    eprintln!("Variant: ResNet-{variant} ({filename})");
    eprintln!("Model: {model_id}");
    eprintln!();

    eprint!("Downloading + loading model weights... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let model = ResNetModel::from_hub_with_filename(&model_id, filename, cfg)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  kind={:?}  blocks_per_stage={:?}  classes={}",
        model.config.kind,
        model.config.blocks_per_stage,
        model.config.nclasses.unwrap_or(0),
    );

    // Synthetic input: a centered bright spot on a dark field, then
    // ImageNet-normalized per the standard mean/std. Matches the
    // convnext-lazy convention so both binaries can be smoke-tested
    // without an image-decode dependency.
    const IMG_SIZE: usize = 224;
    let mean = [0.485_f32, 0.456, 0.406];
    let std = [0.229_f32, 0.224, 0.225];
    let mut image = vec![0.0_f32; 3 * IMG_SIZE * IMG_SIZE];
    let center = IMG_SIZE / 2;
    let radius = IMG_SIZE / 4;
    for c in 0..3 {
        for y in 0..IMG_SIZE {
            for x in 0..IMG_SIZE {
                let dy = y as isize - center as isize;
                let dx = x as isize - center as isize;
                let d2 = (dy * dy + dx * dx) as usize;
                let raw = if d2 < radius * radius { 0.8 } else { 0.1 };
                image[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = (raw - mean[c]) / std[c];
            }
        }
    }
    eprintln!("Input: synthetic 224x224 center-spot, ImageNet-normalized.");
    eprintln!();

    eprintln!("Running forward pass...");
    let t0 = Instant::now();
    let img_lt = LazyTensor::from_f32(
        image,
        Shape::from_dims(&[1, 3, IMG_SIZE, IMG_SIZE]),
        &fuel::Device::cpu(),
    );
    let logits_t = model.forward(&img_lt)?;
    let logits = logits_t.realize_f32();
    let elapsed = t0.elapsed();
    eprintln!("Forward done in {:.2?}", elapsed);

    println!();
    println!("Logits length: {}", logits.len());
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("Top-5:");
    for (i, v) in pairs.iter().take(5) {
        println!("  class {i:>4}  logit = {v:+.4}");
    }
    Ok(())
}

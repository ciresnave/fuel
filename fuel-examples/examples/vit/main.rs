#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::Parser;

use fuel::lazy::LazyTensor;
use fuel::lazy_vit::{VitConfig, VitModel, VitWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;

/// Load an image from disk, resize to 224x224, CHW f32 with ImageNet
/// normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
/// Returns a flat row-major Vec<f32> for a single 224x224 RGB image.
fn load_image224_imagenet_norm<P: AsRef<std::path::Path>>(p: P) -> anyhow::Result<Vec<f32>> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let img = image::ImageReader::open(p)?
        .decode()?
        .resize_to_fill(224, 224, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let raw = img.into_raw();
    let h = 224usize;
    let w = 224usize;
    let mut out = vec![0.0_f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 3;
            for c in 0..3 {
                let v = raw[base + c] as f32 / 255.0;
                out[c * h * w + y * w + x] = (v - MEAN[c]) / STD[c];
            }
        }
    }
    Ok(out)
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Lazy realizes through CPU/router; `cpu` flag preserved for CLI parity.
    let _ = args.cpu;
    let device = Device::cpu();

    let pixels = load_image224_imagenet_norm(&args.image)?;
    let image = LazyTensor::from_f32(
        Arc::<[f32]>::from(pixels),
        Shape::from_dims(&[1, 3, 224, 224]),
        &device,
    );
    println!("loaded image");

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("google/vit-base-patch16-224".into());
            api.get("model.safetensors")?
        }
        Some(model) => model.into(),
    };

    let cfg = VitConfig::vit_base_patch16_224();
    let st = unsafe { MmapedSafetensors::multi(&[&model_file]) }
        .map_err(|e| E::msg(format!("mmap: {e}")))?;
    let weights = VitWeights::load_from_mmapped(&st, &cfg, Some(1000))
        .map_err(|e| E::msg(format!("weights: {e}")))?;
    let model = VitModel {
        config: cfg.clone(),
        weights,
    };
    println!("model built");

    let logits_t = model.forward(&image)?;
    let logits = logits_t.realize_f32();

    // Softmax over the class dim (logits shape: [1, num_classes]).
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let prs: Vec<f32> = exp.iter().map(|v| v / sum).collect();

    let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
    prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for &(category_idx, pr) in prs.iter().take(5) {
        println!(
            "{:24}: {:.2}%",
            fuel_examples::imagenet::CLASSES[category_idx],
            100. * pr
        );
    }
    Ok(())
}

//! BEiT: BERT Pre-Training of Image Transformers
//! https://github.com/microsoft/unilm/tree/master/beit

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::Parser;

use fuel::lazy::LazyTensor;
use fuel::lazy_beit::{BeitConfig, BeitModel, BeitWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;

/// Loads an image from disk and applies BEiT-specific normalization
/// (mean=0.5, std=0.5 per channel). Returns a flat row-major Vec<f32>
/// laid out as CHW for a single 384x384 RGB image.
pub fn load_image384_beit_norm<P: AsRef<std::path::Path>>(p: P) -> anyhow::Result<Vec<f32>> {
    let img = image::ImageReader::open(p)?
        .decode()?
        .resize_to_fill(384, 384, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let raw = img.into_raw();
    // raw is HWC u8 — convert to CHW f32 normalized with mean=std=0.5.
    let h = 384usize;
    let w = 384usize;
    let mut out = vec![0.0_f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 3;
            for c in 0..3 {
                let v = raw[base + c] as f32 / 255.0;
                out[c * h * w + y * w + x] = (v - 0.5) / 0.5;
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

    let pixels = load_image384_beit_norm(args.image)?;
    let image = LazyTensor::from_f32(
        Arc::<[f32]>::from(pixels),
        Shape::from_dims(&[1, 3, 384, 384]),
        &device,
    );
    println!("loaded image");

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("vincent-espitalier/fuel-beit".into());
            api.get("beit_base_patch16_384.in22k_ft_in22k_in1k.safetensors")?
        }
        Some(model) => model.into(),
    };

    let cfg = BeitConfig::vit_base();
    let st = unsafe { MmapedSafetensors::multi(&[&model_file]) }
        .map_err(|e| E::msg(format!("mmap: {e}")))?;
    let weights = BeitWeights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("weights: {e}")))?;
    let model = BeitModel::new(cfg, weights);
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

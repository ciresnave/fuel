//! EVA-02: Explore the limits of Visual representation at scAle
//! https://github.com/baaivision/EVA

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::Parser;

use fuel::lazy::LazyTensor;
use fuel::lazy_eva2::{EvaConfig, EvaModel, EvaWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;

/// Loads an image from disk and applies OpenAI/CLIP normalization, returning
/// a flat row-major Vec<f32> laid out as CHW for a single 448x448 RGB image.
pub fn load_image448_openai_norm<P: AsRef<std::path::Path>>(p: P) -> anyhow::Result<Vec<f32>> {
    let img = image::ImageReader::open(p)?
        .decode()?
        .resize_to_fill(448, 448, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let raw = img.into_raw();
    // raw is HWC u8 — convert to CHW f32 normalized with OpenAI mean/std.
    let h = 448usize;
    let w = 448usize;
    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    let std = [0.26862954f32, 0.261_302_6, 0.275_777_1];
    let mut out = vec![0.0_f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 3;
            for c in 0..3 {
                let v = raw[base + c] as f32 / 255.0;
                out[c * h * w + y * w + x] = (v - mean[c]) / std[c];
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

    let pixels = load_image448_openai_norm(args.image)?;
    let image = LazyTensor::from_f32(
        Arc::<[f32]>::from(pixels),
        Shape::from_dims(&[1, 3, 448, 448]),
        &device,
    );
    println!("loaded image");

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("vincent-espitalier/fuel-eva2".into());
            api.get("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k_adapted.safetensors")?
        }
        Some(model) => model.into(),
    };

    let cfg = EvaConfig::vit_base();
    let st = unsafe { MmapedSafetensors::multi(&[&model_file]) }
        .map_err(|e| E::msg(format!("mmap: {e}")))?;
    let weights = EvaWeights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("weights: {e}")))?;
    let model = EvaModel { config: cfg, weights };
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

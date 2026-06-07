//! DINOv2: Learning Robust Visual Features without Supervision
//! https://github.com/facebookresearch/dinov2

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;
use fuel::lazy::LazyTensor;
use fuel::lazy_dinov2::{Dinov2Config, Dinov2Model, Dinov2Weights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;

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

    // Lazy path realizes via CPU/router; `device` flag is preserved
    // for CLI parity with the eager binary.
    let _ = args.cpu;
    let device = Device::cpu();

    // The lazy DINOv2 v1 expects fixed image_size == 518 (no
    // position-embedding interpolation). Load with the 518 helper
    // and convert to a flat f32 vec for a (1, 3, 518, 518) lazy tensor.
    let cfg = Dinov2Config::vit_small();
    let eager_image = fuel_examples::imagenet::load_image518(&args.image)?;
    println!("loaded image {eager_image:?}");
    let image_vec: Vec<f32> = eager_image.flatten_all()?.to_vec1::<f32>()?;
    let image = LazyTensor::from_f32(
        Arc::<[f32]>::from(image_vec),
        Shape::from_dims(&[1, cfg.num_channels, cfg.image_size, cfg.image_size]),
        &device,
    );

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/fuel-dino-v2".into());
            api.get("dinov2_vits14.safetensors")?
        }
        Some(model) => model.into(),
    };

    let st = unsafe { MmapedSafetensors::new(&model_file) }?;
    let weights = Dinov2Weights::load_from_mmapped(&st, &cfg)?;
    let model = Dinov2Model { config: cfg, weights };
    println!("model built");

    let logits = model.forward(&image)?;
    let probs = logits.softmax_last_dim()?;
    let prs = probs.realize_f32();
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

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::sync::Arc;

use fuel::lazy::LazyTensor;
use fuel::lazy_convmixer::{ConvMixerConfig, ConvMixerModel, ConvMixerWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};

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

pub fn main() -> Result<()> {
    let args = Args::parse();

    // `--cpu` is preserved for parity; lazy realize lives on CPU by
    // default in this binary.
    let _ = args.cpu;
    let device = Device::cpu();

    // Decode + ImageNet-normalize via the shared helper, then unfold to
    // a flat row-major Vec<f32> and wrap as a lazy (1, 3, 224, 224) input.
    let image = fuel_examples::imagenet::load_image224(&args.image)?;
    println!("loaded image {image:?}");
    let image_vec = image.flatten_all()?.to_vec1::<f32>()?;
    let image_lazy = LazyTensor::from_f32(
        Arc::<[f32]>::from(image_vec),
        Shape::from_dims(&[1, 3, 224, 224]),
        &device,
    );

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/fuel-convmixer".into());
            api.get("convmixer_1024_20_ks9_p14.safetensors")?
        }
        Some(model) => model.into(),
    };

    let cfg = ConvMixerConfig::c1024_20(1000);
    let st = unsafe { MmapedSafetensors::multi(&[model_file]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = ConvMixerWeights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("load convmixer weights: {e}")))?;
    let model = ConvMixerModel {
        config: cfg,
        weights,
    };
    println!("model built");

    let logits_t = model.forward(&image_lazy)?;
    let logits = logits_t.realize_f32();

    // Softmax over the class dim.
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

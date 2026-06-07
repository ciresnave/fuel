//! EfficientNet implementation.
//!
//! https://arxiv.org/abs/1905.11946

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};

use fuel::lazy::LazyTensor;
use fuel::lazy_efficientnet::{EfficientNetConfig, EfficientNetModel, EfficientNetWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    B0,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
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

    /// Variant of the model to use.
    #[arg(value_enum, long, default_value_t = Which::B2)]
    which: Which,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Lazy realizes through CPU/router; `cpu` flag preserved for CLI parity.
    let _ = args.cpu;
    let device = Device::cpu();

    // Image loading still uses the eager imagenet helper (returns shape
    // (3, 224, 224)). Convert to a flat f32 vec and build a lazy
    // (1, 3, 224, 224) tensor.
    let eager_image = fuel_examples::imagenet::load_image224(&args.image)?;
    println!("loaded image {eager_image:?}");
    let image_vec: Vec<f32> = eager_image.flatten_all()?.to_vec1::<f32>()?;
    let image = LazyTensor::from_f32(
        Arc::<[f32]>::from(image_vec),
        Shape::from_dims(&[1, 3, 224, 224]),
        &device,
    );

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/fuel-efficientnet".into());
            let filename = match args.which {
                Which::B0 => "efficientnet-b0.safetensors",
                Which::B1 => "efficientnet-b1.safetensors",
                Which::B2 => "efficientnet-b2.safetensors",
                Which::B3 => "efficientnet-b3.safetensors",
                Which::B4 => "efficientnet-b4.safetensors",
                Which::B5 => "efficientnet-b5.safetensors",
                Which::B6 => "efficientnet-b6.safetensors",
                Which::B7 => "efficientnet-b7.safetensors",
            };
            api.get(filename)?
        }
        Some(model) => model.into(),
    };

    let nclasses = fuel_examples::imagenet::CLASS_COUNT as usize;
    let cfg = match args.which {
        Which::B0 => EfficientNetConfig::b0(nclasses),
        Which::B1 => EfficientNetConfig::b1(nclasses),
        Which::B2 => EfficientNetConfig::b2(nclasses),
        Which::B3 => EfficientNetConfig::b3(nclasses),
        Which::B4 => EfficientNetConfig::b4(nclasses),
        Which::B5 => EfficientNetConfig::b5(nclasses),
        Which::B6 => EfficientNetConfig::b6(nclasses),
        Which::B7 => EfficientNetConfig::b7(nclasses),
    };

    let st = unsafe { MmapedSafetensors::new(&model_file) }?;
    let weights = EfficientNetWeights::load_from_mmapped(&st, &cfg)?;
    let model = EfficientNetModel { config: cfg, weights };
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

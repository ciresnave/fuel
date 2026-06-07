#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};

use fuel::lazy::LazyTensor;
use fuel::lazy_efficientvit::{EfficientVitConfig, EfficientVitModel, EfficientVitWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    M0,
    M1,
    M2,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::M0 => "m0",
            Self::M1 => "m1",
            Self::M2 => "m2",
        };
        format!("timm/efficientvit_{name}.r224_in1k")
    }

    fn config(&self) -> EfficientVitConfig {
        match self {
            Self::M0 => EfficientVitConfig::m0(),
            Self::M1 => EfficientVitConfig::m1(),
            Self::M2 => EfficientVitConfig::m2(),
        }
    }
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

    #[arg(value_enum, long, default_value_t=Which::M0)]
    which: Which,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Lazy realizes through CPU/router; `cpu` flag preserved for CLI parity.
    let _ = args.cpu;
    let device = Device::cpu();

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
            let api = api.model(args.which.model_filename());
            api.get("model.safetensors")?
        }
        Some(model) => model.into(),
    };

    let st = unsafe { MmapedSafetensors::multi(&[&model_file]) }?;
    let config = args.which.config();
    let weights = EfficientVitWeights::load_from_mmapped(&st, &config)?;
    let model = EfficientVitModel { config, weights };
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

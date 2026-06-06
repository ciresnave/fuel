#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};

use fuel::lazy::LazyTensor;
use fuel::lazy_mobileone::{MobileOneConfig, MobileOneModel, MobileOneWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    S0,
    S1,
    S2,
    S3,
    S4,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::S0 => "s0",
            Self::S1 => "s1",
            Self::S2 => "s2",
            Self::S3 => "s3",
            Self::S4 => "s4",
        };
        format!("timm/mobileone_{name}.apple_in1k")
    }

    fn config(&self, nclasses: Option<usize>) -> MobileOneConfig {
        match self {
            Self::S0 => MobileOneConfig::s0(nclasses),
            Self::S1 => MobileOneConfig::s1(nclasses),
            Self::S2 => MobileOneConfig::s2(nclasses),
            Self::S3 => MobileOneConfig::s3(nclasses),
            Self::S4 => MobileOneConfig::s4(nclasses),
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

    #[arg(value_enum, long, default_value_t=Which::S0)]
    which: Which,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Lazy realizes through CPU/router; `cpu` flag preserved for CLI parity.
    let _ = args.cpu;
    let device = Device::cpu();

    // Image loading still uses the eager imagenet helper (returns
    // shape (3, 224, 224)). Convert to a flat f32 vec and build a
    // lazy (1, 3, 224, 224) tensor.
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
            let model_name = args.which.model_filename();
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model(model_name);
            api.get("model.safetensors")?
        }
        Some(model) => model.into(),
    };

    let nclasses = fuel_examples::imagenet::CLASS_COUNT as usize;
    let config = args.which.config(Some(nclasses));
    let st = unsafe { MmapedSafetensors::new(&model_file) }?;
    let weights = MobileOneWeights::load_from_mmapped(&st, &config)?;
    let model = MobileOneModel { config, weights };
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

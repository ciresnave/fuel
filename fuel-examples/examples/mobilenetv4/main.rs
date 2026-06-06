#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};

use fuel::lazy::LazyTensor;
use fuel::lazy_mobilenetv4::{Mv4Config, Mv4Model, Mv4Weights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    Small,
    Medium,
    Large,
    HybridMedium,
    HybridLarge,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::Small => "conv_small.e2400_r224",
            Self::Medium => "conv_medium.e500_r256",
            Self::HybridMedium => "hybrid_medium.ix_e550_r256",
            Self::Large => "conv_large.e600_r384",
            Self::HybridLarge => "hybrid_large.ix_e600_r384",
        };
        format!("timm/mobilenetv4_{name}_in1k")
    }

    fn resolution(&self) -> usize {
        match self {
            Self::Small => 224,
            Self::Medium => 256,
            Self::HybridMedium => 256,
            Self::Large => 384,
            Self::HybridLarge => 384,
        }
    }

    fn config(&self) -> anyhow::Result<Mv4Config> {
        match self {
            Self::Small => Ok(Mv4Config::conv_small()),
            // The lazy port currently only ships the `conv_small` preset.
            // Medium / Large / Hybrid variants are follow-ups (Hybrid pulls
            // in the Mobile-MQA attention block).
            Self::Medium | Self::Large | Self::HybridMedium | Self::HybridLarge => {
                anyhow::bail!(
                    "lazy_mobilenetv4 currently only supports the Small (conv_small) variant; \
                     {:?} is not yet ported. See fuel-core/src/lazy_mobilenetv4.rs::Mv4Config.",
                    self
                )
            }
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

    #[arg(value_enum, long, default_value_t=Which::Small)]
    which: Which,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Lazy realizes through CPU/router; `cpu` flag preserved for CLI parity.
    let _ = args.cpu;
    let device = Device::cpu();

    // Image loading still uses the eager imagenet helper (returns
    // shape (3, res, res)). Convert to a flat f32 vec and build a
    // lazy (1, 3, res, res) tensor.
    let res = args.which.resolution();
    let eager_image = fuel_examples::imagenet::load_image(&args.image, res)?;
    println!("loaded image {eager_image:?}");
    let image_vec: Vec<f32> = eager_image.flatten_all()?.to_vec1::<f32>()?;
    let image = LazyTensor::from_f32(
        Arc::<[f32]>::from(image_vec),
        Shape::from_dims(&[1, 3, res, res]),
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
    let config = args.which.config()?;
    let st = unsafe { MmapedSafetensors::new(&model_file) }?;
    let weights = Mv4Weights::load_from_mmapped(&st, &config, Some(nclasses))?;
    let model = Mv4Model { config, weights };
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

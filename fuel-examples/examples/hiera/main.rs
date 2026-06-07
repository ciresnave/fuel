#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::{Parser, ValueEnum};

use fuel::lazy::LazyTensor;
use fuel::lazy_hiera::{HieraConfig, HieraModel, HieraWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    Tiny,
    Small,
    Base,
    BasePlus,
    Large,
    Huge,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Base => "base",
            Self::BasePlus => "base_plus",
            Self::Large => "large",
            Self::Huge => "huge",
        };
        format!("timm/hiera_{name}_224.mae_in1k_ft_in1k")
    }

    fn config(&self) -> HieraConfig {
        match self {
            Self::Tiny => HieraConfig::tiny(),
            Self::Small => HieraConfig::small(),
            Self::Base => HieraConfig::base(),
            Self::BasePlus => HieraConfig::base_plus(),
            Self::Large => HieraConfig::large(),
            Self::Huge => HieraConfig::huge(),
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

    #[arg(value_enum, long, default_value_t=Which::Tiny)]
    which: Which,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Lazy path realizes via CPU/router; `cpu` flag preserved for CLI parity.
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

    let cfg = args.which.config();
    let st = unsafe { MmapedSafetensors::multi(&[&model_file]) }
        .map_err(|e| E::msg(format!("mmap: {e}")))?;
    let weights = HieraWeights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("weights: {e}")))?;
    let model = HieraModel { config: cfg, weights };
    println!("model built");

    let logits_t = model.forward(&image)?;
    let probs_t = logits_t.softmax_last_dim()?;
    let prs = probs_t.realize_f32();

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

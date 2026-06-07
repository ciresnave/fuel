#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use std::sync::Arc;

use fuel::lazy::LazyTensor;
use fuel::lazy_fastvit::{FastVitConfig, FastVitModel, FastVitWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    T8,
    T12,
    S12,
    SA12,
    SA24,
    SA36,
    MA36,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::T8 => "t8",
            Self::T12 => "t12",
            Self::S12 => "s12",
            Self::SA12 => "sa12",
            Self::SA24 => "sa24",
            Self::SA36 => "sa36",
            Self::MA36 => "ma36",
        };
        format!("timm/fastvit_{name}.apple_in1k")
    }

    fn config(&self) -> Result<FastVitConfig> {
        match self {
            Self::T8 => Ok(FastVitConfig::t8()),
            Self::SA12 => Ok(FastVitConfig::sa12()),
            Self::T12 | Self::S12 | Self::SA24 | Self::SA36 | Self::MA36 => Err(E::msg(format!(
                "FastViT variant {self:?} is not yet wired in `lazy_fastvit::FastVitConfig`. \
                 Supported lazy presets: T8, SA12 (plus MCI0/1/2 used by MobileCLIP)."
            ))),
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

    #[arg(value_enum, long, default_value_t=Which::S12)]
    which: Which,
}

pub fn main() -> Result<()> {
    let args = Args::parse();

    // Lazy realizes through CPU/router; `cpu` flag preserved for CLI parity.
    let _ = args.cpu;
    let device = Device::cpu();

    // Image loading still uses the eager imagenet helper (returns
    // shape (3, 256, 256)). Convert to a flat f32 vec and build a
    // lazy (1, 3, 256, 256) tensor.
    let eager_image = fuel_examples::imagenet::load_image(&args.image, 256)?;
    println!("loaded image {eager_image:?}");
    let image_vec: Vec<f32> = eager_image.flatten_all()?.to_vec1::<f32>()?;
    let image = LazyTensor::from_f32(
        Arc::<[f32]>::from(image_vec),
        Shape::from_dims(&[1, 3, 256, 256]),
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

    let config = args.which.config()?;
    let st = unsafe { MmapedSafetensors::multi(&[model_file]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = FastVitWeights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load fastvit weights: {e}")))?;
    let model = FastVitModel { config, weights };
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

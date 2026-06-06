#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};

use fuel::lazy::LazyTensor;
use fuel::lazy_repvgg::{RepVggConfig, RepVggModel, RepVggWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    A0,
    A1,
    A2,
    B0,
    B1,
    B2,
    B3,
    B1G4,
    B2G4,
    B3G4,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::A0 => "a0",
            Self::A1 => "a1",
            Self::A2 => "a2",
            Self::B0 => "b0",
            Self::B1 => "b1",
            Self::B2 => "b2",
            Self::B3 => "b3",
            Self::B1G4 => "b1g4",
            Self::B2G4 => "b2g4",
            Self::B3G4 => "b3g4",
        };
        format!("timm/repvgg_{name}.rvgg_in1k")
    }

    fn config(&self, nclasses: Option<usize>) -> RepVggConfig {
        match self {
            Self::A0 => RepVggConfig::a0(nclasses),
            Self::A1 => RepVggConfig::a1(nclasses),
            Self::A2 => RepVggConfig::a2(nclasses),
            Self::B0 => RepVggConfig::b0(nclasses),
            Self::B1 => RepVggConfig::b1(nclasses),
            Self::B2 => RepVggConfig::b2(nclasses),
            Self::B3 => RepVggConfig::b3(nclasses),
            Self::B1G4 => RepVggConfig::b1g4(nclasses),
            Self::B2G4 => RepVggConfig::b2g4(nclasses),
            Self::B3G4 => RepVggConfig::b3g4(nclasses),
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

    #[arg(value_enum, long, default_value_t=Which::A0)]
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
    let nclasses = 1000_usize;
    let config = args.which.config(Some(nclasses));
    let weights = RepVggWeights::load_from_mmapped(&st, &config)?;
    let model = RepVggModel { config, weights };
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

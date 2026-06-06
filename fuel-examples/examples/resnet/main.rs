#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};
use fuel::lazy::LazyTensor;
use fuel::lazy_resnet::{ResNetConfig, ResNetModel, ResNetWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    #[value(name = "18")]
    Resnet18,
    #[value(name = "34")]
    Resnet34,
    #[value(name = "50")]
    Resnet50,
    #[value(name = "101")]
    Resnet101,
    #[value(name = "152")]
    Resnet152,
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
    #[arg(value_enum, long, default_value_t = Which::Resnet18)]
    which: Which,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Lazy path realizes via CPU/router; `device` flag is preserved
    // for CLI parity with the eager binary.
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
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/fuel-resnet".into());
            let filename = match args.which {
                Which::Resnet18 => "resnet18.safetensors",
                Which::Resnet34 => "resnet34.safetensors",
                Which::Resnet50 => "resnet50.safetensors",
                Which::Resnet101 => "resnet101.safetensors",
                Which::Resnet152 => "resnet152.safetensors",
            };
            api.get(filename)?
        }
        Some(model) => model.into(),
    };

    let class_count = fuel_examples::imagenet::CLASS_COUNT as usize;
    let config = match args.which {
        Which::Resnet18 => ResNetConfig::resnet18(Some(class_count)),
        Which::Resnet34 => ResNetConfig::resnet34(Some(class_count)),
        Which::Resnet50 => ResNetConfig::resnet50(Some(class_count)),
        Which::Resnet101 => ResNetConfig::resnet101(Some(class_count)),
        Which::Resnet152 => ResNetConfig::resnet152(Some(class_count)),
    };
    let st = unsafe { MmapedSafetensors::new(&model_file) }?;
    let weights = ResNetWeights::load_from_mmapped(&st, &config)?;
    let model = ResNetModel { config, weights };
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

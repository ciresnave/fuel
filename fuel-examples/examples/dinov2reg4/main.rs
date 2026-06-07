//! DINOv2 reg4 finetuned on PlantCLEF 2024
//! https://arxiv.org/abs/2309.16588
//! https://huggingface.co/spaces/BVRA/PlantCLEF2024
//! https://zenodo.org/records/10848263

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;
use fuel::lazy::LazyTensor;
use fuel::lazy_dinov2reg4::{Dinov2Reg4Config, Dinov2Reg4Model, Dinov2Reg4Weights};
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

    // Image loading still uses the eager imagenet helper (returns
    // shape (3, 518, 518)). Convert to a flat f32 vec and build a
    // lazy (1, 3, 518, 518) tensor.
    let eager_image = fuel_examples::imagenet::load_image518(&args.image)?;
    println!("loaded image {eager_image:?}");
    let image_vec: Vec<f32> = eager_image.flatten_all()?.to_vec1::<f32>()?;
    let image = LazyTensor::from_f32(
        Arc::<[f32]>::from(image_vec),
        Shape::from_dims(&[1, 3, 518, 518]),
        &device,
    );

    let f_species_id_mapping = "fuel-examples/examples/dinov2reg4/species_id_mapping.txt";
    let classes: Vec<String> = std::fs::read_to_string(f_species_id_mapping)
        .expect("missing classes file")
        .split('\n')
        .map(|s| s.to_string())
        .collect();

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api =
                api.model("vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights".into());
            api.get(
                "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all.safetensors",
            )?
        }
        Some(model) => model.into(),
    };

    // The PlantCLEF2024 checkpoint is a ViT-Base/14 backbone with a
    // 7806-class classifier head. `Dinov2Reg4Config::vit_base()`
    // defaults to 1000 classes, so override `num_classes` here.
    let mut config = Dinov2Reg4Config::vit_base();
    config.num_classes = 7806;

    let st = unsafe { MmapedSafetensors::new(&model_file) }?;
    let weights = Dinov2Reg4Weights::load_from_mmapped(&st, &config)?;
    let model = Dinov2Reg4Model { config, weights };
    println!("model built");

    let logits = model.forward(&image)?;
    let probs = logits.softmax_last_dim()?;
    let prs = probs.realize_f32();
    let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
    prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for &(category_idx, pr) in prs.iter().take(5) {
        println!("{:24}: {:.2}%", classes[category_idx], 100. * pr);
    }
    Ok(())
}

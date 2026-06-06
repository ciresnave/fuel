#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};

use fuel::lazy_convnext::{ConvNextConfig, ConvNextModel, ConvNextWeights};
use fuel::safetensors::MmapedSafetensors;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    Atto,
    Femto,
    Pico,
    Nano,
    Tiny,
    Small,
    Base,
    Large,
    AttoV2,
    FemtoV2,
    PicoV2,
    NanoV2,
    TinyV2,
    BaseV2,
    LargeV2,
    XLarge,
    Huge,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::Atto => "convnext_atto.d2_in1k",
            Self::Femto => "convnext_femto.d1_in1k",
            Self::Pico => "convnext_pico.d1_in1k",
            Self::Nano => "convnext_nano.d1h_in1k",
            Self::Tiny => "convnext_tiny.fb_in1k",
            Self::Small => "convnext_small.fb_in1k",
            Self::Base => "convnext_base.fb_in1k",
            Self::Large => "convnext_large.fb_in1k",
            Self::AttoV2 => "convnextv2_atto.fcmae_ft_in1k",
            Self::FemtoV2 => "convnextv2_femto.fcmae_ft_in1k",
            Self::PicoV2 => "convnextv2_pico.fcmae_ft_in1k",
            Self::NanoV2 => "convnextv2_nano.fcmae_ft_in1k",
            Self::TinyV2 => "convnextv2_tiny.fcmae_ft_in1k",
            Self::BaseV2 => "convnextv2_base.fcmae_ft_in1k",
            Self::LargeV2 => "convnextv2_large.fcmae_ft_in1k",
            Self::XLarge => "convnext_xlarge.fb_in22k_ft_in1k",
            Self::Huge => "convnextv2_huge.fcmae_ft_in1k",
        };

        format!("timm/{name}")
    }

    fn config(&self) -> ConvNextConfig {
        // ConvNextConfig::tiny() is the only V1 preset in the lazy module;
        // build the other V1 variants inline by patching dims/depths.
        let mut cfg = ConvNextConfig::tiny();
        match self {
            // V1 variants (the lazy module ships these as ad-hoc configs).
            Self::Atto => {
                cfg.dims = vec![40, 80, 160, 320];
                cfg.depths = vec![2, 2, 6, 2];
            }
            Self::Femto => {
                cfg.dims = vec![48, 96, 192, 384];
                cfg.depths = vec![2, 2, 6, 2];
            }
            Self::Pico => {
                cfg.dims = vec![64, 128, 256, 512];
                cfg.depths = vec![2, 2, 6, 2];
            }
            Self::Nano => {
                cfg.dims = vec![80, 160, 320, 640];
                cfg.depths = vec![2, 2, 8, 2];
            }
            Self::Tiny => {
                // already correct via tiny()
            }
            Self::Small => {
                cfg.dims = vec![96, 192, 384, 768];
                cfg.depths = vec![3, 3, 27, 3];
            }
            Self::Base => {
                cfg.dims = vec![128, 256, 512, 1024];
                cfg.depths = vec![3, 3, 27, 3];
            }
            Self::Large => {
                cfg.dims = vec![192, 384, 768, 1536];
                cfg.depths = vec![3, 3, 27, 3];
            }
            Self::XLarge => {
                cfg.dims = vec![256, 512, 1024, 2048];
                cfg.depths = vec![3, 3, 27, 3];
            }
            // V2 variants ship as presets on ConvNextConfig.
            Self::AttoV2 => return ConvNextConfig::v2_atto(),
            Self::FemtoV2 => return ConvNextConfig::v2_femto(),
            Self::PicoV2 => return ConvNextConfig::v2_pico(),
            Self::NanoV2 => return ConvNextConfig::v2_nano(),
            Self::TinyV2 => return ConvNextConfig::v2_tiny(),
            Self::BaseV2 => return ConvNextConfig::v2_base(),
            Self::LargeV2 => return ConvNextConfig::v2_large(),
            Self::Huge => return ConvNextConfig::v2_huge(),
        }
        cfg
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

    // Image is decoded to an ImageNet-normalized eager Tensor (3, 224, 224)
    // by the shared examples helper, then unfolded to a flat row-major Vec<f32>
    // which is what `ConvNextModel::forward` expects.
    let image = fuel_examples::imagenet::load_image224(args.image)?;
    println!("loaded image {image:?}");
    let image_vec = image.flatten_all()?.to_vec1::<f32>()?;

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
    let st = unsafe { MmapedSafetensors::new(&model_file) }?;
    let weights = ConvNextWeights::load_from_mmapped(&st, &cfg)?;
    let model = ConvNextModel { config: cfg, weights };
    println!("model built");

    let logits_t = model.forward(&image_vec)?;
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

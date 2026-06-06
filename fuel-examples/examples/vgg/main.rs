#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};
use fuel::lazy::{
    load_tensor_as_f32, load_transposed_matrix, LazyTensor, WeightStorage,
};
use fuel::lazy_vgg::{
    VggConfig, VggConvWeights, VggHeadFc, VggModel, VggVariant, VggWeights,
};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    Vgg13,
    Vgg16,
    Vgg19,
}

impl Which {
    fn variant(&self) -> VggVariant {
        match self {
            Self::Vgg13 => VggVariant::Vgg13,
            Self::Vgg16 => VggVariant::Vgg16,
            Self::Vgg19 => VggVariant::Vgg19,
        }
    }

    fn hf_repo(&self) -> &'static str {
        match self {
            Self::Vgg13 => "timm/vgg13.tv_in1k",
            Self::Vgg16 => "timm/vgg16.tv_in1k",
            Self::Vgg19 => "timm/vgg19.tv_in1k",
        }
    }

    /// `features.<idx>` indices of conv layers in the timm
    /// safetensors. The other indices are ReLU / MaxPool, which
    /// are activation-only and have no weights.
    fn conv_feature_indices(&self) -> Vec<usize> {
        match self {
            // Vgg13: blocks [2,2,2,2,2], stride layout has a Pool
            // after every 2 convs, so conv indices are 0, 2, 5, 7,
            // 10, 12, 15, 17, 20, 22.
            Self::Vgg13 => vec![0, 2, 5, 7, 10, 12, 15, 17, 20, 22],
            // Vgg16: blocks [2,2,3,3,3]
            Self::Vgg16 => vec![0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28],
            // Vgg19: blocks [2,2,4,4,4]
            Self::Vgg19 => vec![
                0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34,
            ],
        }
    }
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Variant of the model to use.
    #[arg(value_enum, long, default_value_t = Which::Vgg13)]
    which: Which,
}

/// Channel widths per block (canonical VGG).
fn block_channels() -> [usize; 5] {
    [64, 128, 256, 512, 512]
}

fn load_vgg_weights(
    st: &MmapedSafetensors,
    which: Which,
    nclasses: usize,
) -> anyhow::Result<VggWeights> {
    let variant = which.variant();
    let convs_per_block = variant.convs_per_block();
    let ch = block_channels();
    let feature_ids = which.conv_feature_indices();

    let total_convs: usize = convs_per_block.iter().sum();
    assert_eq!(feature_ids.len(), total_convs,
        "feature-index list ({}) doesn't match total conv count ({total_convs})",
        feature_ids.len());

    let mut blocks: Vec<Vec<VggConvWeights>> = Vec::with_capacity(5);
    let mut c_prev = 3_usize;
    let mut conv_cursor = 0_usize;
    for (block_idx, &n_conv) in convs_per_block.iter().enumerate() {
        let c_out = ch[block_idx];
        let mut block = Vec::with_capacity(n_conv);
        for conv_in_block in 0..n_conv {
            let c_in = if conv_in_block == 0 { c_prev } else { c_out };
            let feat_id = feature_ids[conv_cursor];
            let w_name = format!("features.{feat_id}.weight");
            let b_name = format!("features.{feat_id}.bias");
            let w_flat = load_tensor_as_f32(st, &w_name)?;
            let expected = c_out * c_in * 3 * 3;
            anyhow::ensure!(
                w_flat.len() == expected,
                "{w_name}: {} elements, expected {expected} ({c_out}x{c_in}x3x3)",
                w_flat.len(),
            );
            let b_flat = load_tensor_as_f32(st, &b_name)?;
            block.push(VggConvWeights {
                w: WeightStorage::F32(Arc::from(w_flat)),
                b: Arc::from(b_flat),
                c_in,
                c_out,
            });
            conv_cursor += 1;
        }
        c_prev = c_out;
        blocks.push(block);
    }

    let head_hidden = 4096_usize;
    let head_spatial = 7_usize;
    let flat_in = c_prev * head_spatial * head_spatial;
    let fc1_w = WeightStorage::F32(Arc::from(load_transposed_matrix(
        st,
        "pre_logits.fc1.weight",
        head_hidden,
        flat_in,
    )?));
    let fc1_b = Arc::from(load_tensor_as_f32(st, "pre_logits.fc1.bias")?);
    let fc2_w = WeightStorage::F32(Arc::from(load_transposed_matrix(
        st,
        "pre_logits.fc2.weight",
        head_hidden,
        head_hidden,
    )?));
    let fc2_b = Arc::from(load_tensor_as_f32(st, "pre_logits.fc2.bias")?);
    let fc3_w = WeightStorage::F32(Arc::from(load_transposed_matrix(
        st,
        "head.fc.weight",
        nclasses,
        head_hidden,
    )?));
    let fc3_b = Arc::from(load_tensor_as_f32(st, "head.fc.bias")?);

    Ok(VggWeights {
        blocks,
        fc1: VggHeadFc {
            w: fc1_w,
            b: fc1_b,
            in_features: flat_in,
            out_features: head_hidden,
        },
        fc2: VggHeadFc {
            w: fc2_w,
            b: fc2_b,
            in_features: head_hidden,
            out_features: head_hidden,
        },
        fc3: VggHeadFc {
            w: fc3_w,
            b: fc3_b,
            in_features: head_hidden,
            out_features: nclasses,
        },
    })
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

    let api = hf_hub::api::sync::Api::new()?;
    let api = api.model(args.which.hf_repo().to_string());
    let model_file = api.get("model.safetensors")?;

    let st = unsafe { MmapedSafetensors::new(&model_file) }?;
    let nclasses = 1000_usize;
    let config = match args.which {
        Which::Vgg13 => VggConfig::vgg13(nclasses),
        Which::Vgg16 => VggConfig::vgg16(nclasses),
        Which::Vgg19 => VggConfig::vgg19(nclasses),
    };
    let weights = load_vgg_weights(&st, args.which, nclasses)?;
    let model = VggModel { config, weights };
    println!("model built");

    let logits = model.forward(&image)?;
    let probs = logits.softmax_last_dim()?;
    let prs = probs.realize_f32();
    let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
    prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for &(category_idx, pr) in prs.iter().take(5) {
        println!(
            "{:50}: {:.2}%",
            fuel_examples::imagenet::CLASSES[category_idx],
            100. * pr
        );
    }
    Ok(())
}

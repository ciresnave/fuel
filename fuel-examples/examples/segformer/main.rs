#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::{Args, Parser, Subcommand};
use imageproc::image::Rgb;
use imageproc::integral_image::ArrayData;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use fuel::lazy::LazyTensor;
use fuel::lazy_segformer::{
    ImageClassificationModel, SegformerActivation, SegformerConfig, SemanticSegmentationModel,
};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};

#[derive(Parser)]
#[clap(about, version, long_about = None)]
struct CliArgs {
    #[arg(long, help = "use cpu")]
    cpu: bool,
    #[command(subcommand)]
    command: Commands,
}
#[derive(Args, Debug)]
struct SegmentationArgs {
    #[arg(
        long,
        help = "name of the huggingface hub model",
        default_value = "nvidia/segformer-b0-finetuned-ade-512-512"
    )]
    model_name: String,
    #[arg(
        long,
        help = "path to the label file in json format",
        default_value = "fuel-examples/examples/segformer/assets/labels.json"
    )]
    label_path: PathBuf,
    #[arg(long, help = "path to for the output mask image")]
    output_path: PathBuf,
    #[arg(help = "path to image as input")]
    image: PathBuf,
}

#[derive(Args, Debug)]
struct ClassificationArgs {
    #[arg(
        long,
        help = "name of the huggingface hub model",
        default_value = "paolinox/segformer-finetuned-food101"
    )]
    model_name: String,
    #[arg(help = "path to image as input")]
    image: PathBuf,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Segment(SegmentationArgs),
    Classify(ClassificationArgs),
}

/// HF `config.json` shape, with the fields we care about. The lazy
/// `SegformerConfig` doesn't carry `id2label`, so we deserialize this
/// helper, keep `id2label` here, and project everything else into
/// `SegformerConfig`.
#[derive(Debug, Clone, serde::Deserialize)]
struct HfSegformerConfig {
    #[serde(default)]
    id2label: HashMap<String, String>,
    num_channels: usize,
    num_encoder_blocks: usize,
    depths: Vec<usize>,
    sr_ratios: Vec<usize>,
    hidden_sizes: Vec<usize>,
    patch_sizes: Vec<usize>,
    strides: Vec<usize>,
    num_attention_heads: Vec<usize>,
    mlp_ratios: Vec<usize>,
    hidden_act: String,
    layer_norm_eps: f64,
    decoder_hidden_size: usize,
}

impl HfSegformerConfig {
    fn to_lazy(&self) -> anyhow::Result<SegformerConfig> {
        let hidden_act = match self.hidden_act.as_str() {
            "gelu" | "gelu_new" | "gelu_pytorch_tanh" => SegformerActivation::Gelu,
            "relu" => SegformerActivation::Relu,
            other => {
                return Err(E::msg(format!(
                    "unsupported hidden_act for lazy segformer: {other:?}"
                )));
            }
        };
        Ok(SegformerConfig {
            num_channels: self.num_channels,
            num_encoder_blocks: self.num_encoder_blocks,
            depths: self.depths.clone(),
            sr_ratios: self.sr_ratios.clone(),
            hidden_sizes: self.hidden_sizes.clone(),
            patch_sizes: self.patch_sizes.clone(),
            strides: self.strides.clone(),
            num_attention_heads: self.num_attention_heads.clone(),
            mlp_ratios: self.mlp_ratios.clone(),
            hidden_act,
            layer_norm_eps: self.layer_norm_eps,
            decoder_hidden_size: self.decoder_hidden_size,
        })
    }
}

fn fetch_model_and_config(
    model_name: String,
) -> anyhow::Result<(PathBuf, HfSegformerConfig)> {
    println!("loading model {model_name} via huggingface hub");
    let api = hf_hub::api::sync::Api::new()?;
    let api = api.model(model_name.clone());
    let model_file = api.get("model.safetensors")?;
    println!("model {model_name} downloaded and loaded");
    let config = std::fs::read_to_string(api.get("config.json")?)?;
    let config: HfSegformerConfig = serde_json::from_str(&config)?;
    println!("{config:?}");
    Ok((model_file, config))
}

#[derive(Debug, serde::Deserialize)]
struct LabelItem {
    index: u32,
    color: String,
}

/// Load an image at 224x224 ImageNet preprocessing via the shared
/// helper, then convert to a lazy `(1, 3, 224, 224)` tensor.
fn load_image_lazy(path: PathBuf, device: &Device) -> anyhow::Result<LazyTensor> {
    let eager_image = fuel_examples::imagenet::load_image224(path)?;
    println!("loaded image {eager_image:?}");
    let image_vec: Vec<f32> = eager_image.flatten_all()?.to_vec1::<f32>()?;
    Ok(LazyTensor::from_f32(
        Arc::<[f32]>::from(image_vec),
        Shape::from_dims(&[1, 3, 224, 224]),
        device,
    ))
}

fn segmentation_task(args: SegmentationArgs, device: &Device) -> anyhow::Result<()> {
    let label_file = std::fs::read_to_string(&args.label_path)?;
    let label_items: Vec<LabelItem> = serde_json::from_str(&label_file)?;
    let label_colors: HashMap<u32, Rgb<u8>> = label_items
        .iter()
        .map(|x| {
            (x.index - 1, {
                let color = x.color.trim_start_matches('#');
                let r = u8::from_str_radix(&color[0..2], 16).unwrap();
                let g = u8::from_str_radix(&color[2..4], 16).unwrap();
                let b = u8::from_str_radix(&color[4..6], 16).unwrap();
                Rgb([r, g, b])
            })
        })
        .collect();

    let image = load_image_lazy(args.image, device)?;
    let (model_file, hf_cfg) = fetch_model_and_config(args.model_name)?;
    let num_labels = label_items.len();

    let cfg = hf_cfg.to_lazy()?;
    let st = unsafe { MmapedSafetensors::multi(&[&model_file]) }
        .map_err(|e| E::msg(format!("mmap: {e}")))?;
    let model = SemanticSegmentationModel::load_from_mmapped(&st, cfg, num_labels)
        .map_err(|e| E::msg(format!("weights: {e}")))?;

    let segmentations = model.forward(&image)?;

    // segmentations: (1, num_labels, H, W) → squeeze batch → argmax along
    // the class dim (axis 0 after the squeeze) → (H, W) u32 mask.
    let mask_t = segmentations.squeeze(0_usize)?.argmax_dim(0_usize)?;
    let mask_dims = mask_t.shape();
    let mask_dims = mask_dims.dims();
    let (h, w) = (mask_dims[0], mask_dims[1]);
    let mask = mask_t.realize_u32();
    let mask = mask
        .iter()
        .flat_map(|x| label_colors[x].data())
        .collect::<Vec<u8>>();
    let mask: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(w as u32, h as u32, mask).unwrap();
    // resize
    let mask = image::DynamicImage::from(mask);
    let mask = mask.resize_to_fill(
        w as u32 * 4,
        h as u32 * 4,
        image::imageops::FilterType::CatmullRom,
    );
    mask.save(args.output_path.clone())?;
    println!("mask image saved to {:?}", args.output_path);
    Ok(())
}

fn classification_task(args: ClassificationArgs, device: &Device) -> anyhow::Result<()> {
    let image = load_image_lazy(args.image, device)?;
    let (model_file, hf_cfg) = fetch_model_and_config(args.model_name)?;
    let num_labels = 7;
    let cfg = hf_cfg.to_lazy()?;
    let id2label = hf_cfg.id2label.clone();
    let st = unsafe { MmapedSafetensors::multi(&[&model_file]) }
        .map_err(|e| E::msg(format!("mmap: {e}")))?;
    let model = ImageClassificationModel::load_from_mmapped(&st, cfg, num_labels)
        .map_err(|e| E::msg(format!("weights: {e}")))?;

    let classification = model.forward(&image)?;
    let probs = classification.softmax_last_dim()?;
    let probs = probs.squeeze(0_usize)?;
    let probs_vec = probs.realize_f32();
    println!("classification logits {probs_vec:?}");

    let label_id = probs
        .argmax_dim(0_usize)?
        .realize_u32()
        .first()
        .copied()
        .ok_or_else(|| E::msg("argmax returned empty"))?;
    let label_id = format!("{label_id}");
    println!("label: {}", id2label[&label_id]);
    Ok(())
}

pub fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse();
    // Lazy path realizes via CPU/router; the `cpu` flag is preserved
    // for CLI parity but the device passed into LazyTensor::from_f32
    // is always CPU.
    let _ = args.cpu;
    let device = Device::cpu();
    if let Commands::Segment(args) = args.command {
        segmentation_task(args, &device)?
    } else if let Commands::Classify(args) = args.command {
        classification_task(args, &device)?
    }
    Ok(())
}

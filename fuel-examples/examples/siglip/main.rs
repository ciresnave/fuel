#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::Parser;

use fuel::lazy::LazyTensor;
use fuel::lazy_siglip::{
    SiglipActivation, SiglipModel, SiglipModelWeights, SiglipTextConfig, SiglipVisionConfig,
};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;

use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
enum Which {
    #[value(name = "v1-base-patch16-224")]
    V1BasePatch16_224,
    #[value(name = "v2-base-patch16-224")]
    V2BasePatch16_224,
    #[value(name = "v2-base-patch16-256")]
    V2BasePatch16_256,
    #[value(name = "v2-base-patch16-384")]
    V2BasePatch16_384,
    #[value(name = "v2-base-patch16-512")]
    V2BasePatch16_512,
    #[value(name = "v2-large-patch16-256")]
    V2LargePatch16_256,
    #[value(name = "v2-large-patch16-384")]
    V2LargePatch16_384,
    #[value(name = "v2-large-patch16-512")]
    V2LargePatch16_512,
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    config: Option<String>,

    #[arg(long)]
    hf_repo: Option<String>,

    #[arg(long, default_value = "v1-base-patch16-224")]
    which: Which,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long, use_value_delimiter = true)]
    images: Option<Vec<String>>,

    #[arg(long)]
    cpu: bool,

    #[arg(long, use_value_delimiter = true)]
    sequences: Option<Vec<String>>,

    #[arg(short, long)]
    image_size: Option<usize>,
}

/// Load an image and normalize to [-1, 1] CHW f32 tensor with shape (3, H, W).
fn load_image_chw<T: AsRef<std::path::Path>>(
    path: T,
    image_size: usize,
) -> anyhow::Result<Vec<f32>> {
    let img = image::ImageReader::open(path)?.decode()?;
    let img = img.resize_to_fill(
        image_size as u32,
        image_size as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8();
    let raw = img.into_raw();
    // raw is HWC u8 — convert to CHW f32 normalized to [-1, 1].
    let h = image_size;
    let w = image_size;
    let mut out = vec![0.0_f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 3;
            for c in 0..3 {
                out[c * h * w + y * w + x] = raw[base + c] as f32 / 255.0 * 2.0 - 1.0;
            }
        }
    }
    Ok(out)
}

/// SigLIP config fragment we need to drive the lazy model.
#[derive(serde::Deserialize, Debug)]
struct HfTextConfig {
    #[serde(default = "default_text_vocab_size")]
    vocab_size: usize,
    #[serde(default = "default_text_hidden_size")]
    hidden_size: usize,
    #[serde(default = "default_text_intermediate_size")]
    intermediate_size: usize,
    #[serde(default = "default_text_num_hidden_layers")]
    num_hidden_layers: usize,
    #[serde(default = "default_text_num_attention_heads")]
    num_attention_heads: usize,
    #[serde(default = "default_text_max_position_embeddings")]
    max_position_embeddings: usize,
    #[serde(default = "default_text_layer_norm_eps")]
    layer_norm_eps: f64,
    #[serde(default = "default_text_pad_token_id")]
    pad_token_id: u32,
}

fn default_text_vocab_size() -> usize {
    32000
}
fn default_text_hidden_size() -> usize {
    768
}
fn default_text_intermediate_size() -> usize {
    3072
}
fn default_text_num_hidden_layers() -> usize {
    12
}
fn default_text_num_attention_heads() -> usize {
    12
}
fn default_text_max_position_embeddings() -> usize {
    64
}
fn default_text_layer_norm_eps() -> f64 {
    1e-6
}
fn default_text_pad_token_id() -> u32 {
    1
}

#[derive(serde::Deserialize, Debug)]
struct HfVisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    hidden_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    intermediate_size: usize,
    #[serde(default = "default_vision_num_hidden_layers")]
    num_hidden_layers: usize,
    #[serde(default = "default_vision_num_attention_heads")]
    num_attention_heads: usize,
    #[serde(default = "default_vision_num_channels")]
    num_channels: usize,
    #[serde(default = "default_vision_image_size")]
    image_size: usize,
    #[serde(default = "default_vision_patch_size")]
    patch_size: usize,
    #[serde(default = "default_vision_layer_norm_eps")]
    layer_norm_eps: f64,
}

fn default_vision_hidden_size() -> usize {
    768
}
fn default_vision_intermediate_size() -> usize {
    3072
}
fn default_vision_num_hidden_layers() -> usize {
    12
}
fn default_vision_num_attention_heads() -> usize {
    12
}
fn default_vision_num_channels() -> usize {
    3
}
fn default_vision_image_size() -> usize {
    224
}
fn default_vision_patch_size() -> usize {
    16
}
fn default_vision_layer_norm_eps() -> f64 {
    1e-6
}

#[derive(serde::Deserialize, Debug)]
struct HfConfig {
    text_config: HfTextConfig,
    vision_config: HfVisionConfig,
}

fn text_cfg_from_hf(hf: &HfTextConfig) -> SiglipTextConfig {
    SiglipTextConfig {
        vocab_size: hf.vocab_size,
        hidden_size: hf.hidden_size,
        intermediate_size: hf.intermediate_size,
        num_hidden_layers: hf.num_hidden_layers,
        num_attention_heads: hf.num_attention_heads,
        max_position_embeddings: hf.max_position_embeddings,
        hidden_activation: SiglipActivation::GeluPytorchTanh,
        layer_norm_eps: hf.layer_norm_eps,
    }
}

fn vision_cfg_from_hf(hf: &HfVisionConfig) -> SiglipVisionConfig {
    SiglipVisionConfig {
        hidden_size: hf.hidden_size,
        intermediate_size: hf.intermediate_size,
        num_hidden_layers: hf.num_hidden_layers,
        num_attention_heads: hf.num_attention_heads,
        num_channels: hf.num_channels,
        image_size: hf.image_size,
        patch_size: hf.patch_size,
        hidden_activation: SiglipActivation::GeluPytorchTanh,
        layer_norm_eps: hf.layer_norm_eps,
    }
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let hf_repo = match args.hf_repo.as_ref() {
        Some(hf_repo) => hf_repo.to_string(),
        None => match args.which {
            Which::V1BasePatch16_224 => "google/siglip-base-patch16-224".to_string(),
            Which::V2BasePatch16_224 => "google/siglip2-base-patch16-224".to_string(),
            Which::V2BasePatch16_256 => "google/siglip2-base-patch16-256".to_string(),
            Which::V2BasePatch16_384 => "google/siglip2-base-patch16-384".to_string(),
            Which::V2BasePatch16_512 => "google/siglip2-base-patch16-512".to_string(),
            Which::V2LargePatch16_256 => "google/siglip2-large-patch16-256".to_string(),
            Which::V2LargePatch16_384 => "google/siglip2-large-patch16-384".to_string(),
            Which::V2LargePatch16_512 => "google/siglip2-large-patch16-512".to_string(),
        },
    };
    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model(hf_repo.clone());
            api.get("model.safetensors")?
        }
        Some(model) => model.into(),
    };
    let config_file = match args.config {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model(hf_repo.clone());
            api.get("config.json")?
        }
        Some(config) => config.into(),
    };
    let tokenizer = get_tokenizer(&hf_repo, args.tokenizer)?;
    let hf_config: HfConfig = serde_json::from_slice(&std::fs::read(&config_file)?)?;
    let text_config = text_cfg_from_hf(&hf_config.text_config);
    let vision_config = vision_cfg_from_hf(&hf_config.vision_config);

    // Lazy realizes through CPU/router; `cpu` flag preserved for CLI parity.
    let _ = args.cpu;
    let device = Device::cpu();

    let vec_imgs = match args.images {
        Some(imgs) => imgs,
        None => vec![
            "fuel-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg".to_string(),
            "fuel-examples/examples/yolo-v8/assets/bike.jpg".to_string(),
        ],
    };
    let image_size = args.image_size.unwrap_or(vision_config.image_size);

    let st = unsafe { MmapedSafetensors::multi(&[&model_file]) }?;
    let weights = SiglipModelWeights::load_from_mmapped(&st, &text_config, &vision_config)?;
    let model = SiglipModel {
        text_config: text_config.clone(),
        vision_config: vision_config.clone(),
        weights,
    };
    println!("model built");

    // Tokenize text sequences (padded to max_position_embeddings).
    let (token_seqs, vec_seq) = tokenize_sequences(&text_config, args.sequences, &tokenizer)?;

    // Encode every image independently → (1, hidden); collect into (N_img, hidden).
    let mut image_feats: Vec<LazyTensor> = Vec::with_capacity(vec_imgs.len());
    for path in &vec_imgs {
        let pixels = load_image_chw(path, image_size)?;
        let pixel_tensor = LazyTensor::from_f32(
            Arc::<[f32]>::from(pixels),
            Shape::from_dims(&[1, 3, image_size, image_size]),
            &device,
        );
        let feat = model.image_features(&pixel_tensor)?;
        image_feats.push(feat);
    }
    // Stack image features → (N_img, hidden).
    let mut img_acc = image_feats.remove(0);
    for next in image_feats.into_iter() {
        img_acc = img_acc.concat(&next, 0_usize)?;
    }
    let image_features_normalized = img_acc.l2_normalize(1_usize, 1e-12)?;

    // Encode every text sequence independently → (1, hidden); collect into (N_txt, hidden).
    let mut text_feats: Vec<LazyTensor> = Vec::with_capacity(token_seqs.len());
    for tokens in &token_seqs {
        let feat = model.text_features(tokens)?;
        text_feats.push(feat);
    }
    let mut txt_acc = text_feats.remove(0);
    for next in text_feats.into_iter() {
        txt_acc = txt_acc.concat(&next, 0_usize)?;
    }
    let text_features_normalized = txt_acc.l2_normalize(1_usize, 1e-12)?;

    // logits_per_text = (txt @ img^T) * exp(logit_scale) + logit_bias.
    let scale = (model.weights.logit_scale as f64).exp();
    let bias = model.weights.logit_bias as f64;
    let img_t = image_features_normalized.transpose()?;
    let logits_per_text = text_features_normalized
        .matmul(&img_t)?
        .mul_scalar(scale)
        .add_scalar(bias);
    let logits_per_image = logits_per_text.transpose()?;

    // Softmax over the text axis per image, matching the eager example.
    let softmax_image = logits_per_image.softmax_last_dim()?;
    let softmax_image_vec = softmax_image.realize_f32();
    println!("softmax_image_vec: {softmax_image_vec:?}");
    let probability_vec = softmax_image_vec
        .iter()
        .map(|v| v * 100.0)
        .collect::<Vec<f32>>();
    let probability_per_image = probability_vec.len() / vec_imgs.len();
    for (i, img) in vec_imgs.iter().enumerate() {
        let start = i * probability_per_image;
        let end = start + probability_per_image;
        let prob = &probability_vec[start..end];
        println!("\n\nResults for image: {img}\n");
        for (i, p) in prob.iter().enumerate() {
            println!("Probability: {:.4}% Text: {} ", p, vec_seq[i]);
        }
    }
    Ok(())
}

pub fn get_tokenizer(hf_repo: &str, tokenizer: Option<String>) -> anyhow::Result<Tokenizer> {
    let tokenizer = match tokenizer {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model(hf_repo.to_string());
            api.get("tokenizer.json")?
        }
        Some(file) => file.into(),
    };

    Tokenizer::from_file(tokenizer).map_err(E::msg)
}

pub fn tokenize_sequences(
    config: &SiglipTextConfig,
    sequences: Option<Vec<String>>,
    tokenizer: &Tokenizer,
) -> anyhow::Result<(Vec<Vec<u32>>, Vec<String>)> {
    // Default pad id from SigLIP HF config; the lazy text model accepts a
    // raw token slice so we pad here to keep parity with the eager example
    // (fixed-length text inputs).
    let pad_id: u32 = 1;
    let vec_seq = match sequences {
        Some(seq) => seq,
        None => vec![
            "a cycling race".to_string(),
            "a photo of two cats".to_string(),
            "a robot holding a fuel".to_string(),
        ],
    };
    let mut tokens: Vec<Vec<u32>> = vec![];
    for seq in vec_seq.clone() {
        let encoding = tokenizer.encode(seq, true).map_err(E::msg)?;
        tokens.push(encoding.get_ids().to_vec());
    }
    let max_len = config.max_position_embeddings;
    for token_vec in tokens.iter_mut() {
        let len_diff = max_len - token_vec.len();
        if len_diff > 0 {
            token_vec.extend(vec![pad_id; len_diff]);
        }
    }
    Ok((tokens, vec_seq))
}

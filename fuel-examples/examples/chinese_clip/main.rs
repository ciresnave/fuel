#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;

use fuel::lazy::LazyTensor;
use fuel::lazy_chinese_clip::{
    ChineseClipConfig, ChineseClipModel, ChineseClipWeights,
};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long, use_value_delimiter = true)]
    images: Option<Vec<String>>,

    #[arg(long)]
    cpu: bool,

    #[arg(long, use_value_delimiter = true)]
    sequences: Option<Vec<String>>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt::init();

    // `--cpu` preserved for CLI parity; lazy realize defaults to CPU.
    let _ = args.cpu;
    let device = Device::cpu();

    let config = ChineseClipConfig::clip_vit_base_patch16();

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let repo = hf_hub::Repo::with_revision(
                "OFA-Sys/chinese-clip-vit-base-patch16".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/3".to_string(),
            );
            let api = api.repo(repo);
            api.get("model.safetensors")?
        }
        Some(model) => model.into(),
    };

    let st = unsafe { MmapedSafetensors::multi(&[model_file]) }
        .map_err(|e| anyhow::Error::msg(format!("mmap safetensors: {e}")))?;
    let weights = ChineseClipWeights::load_from_mmapped(&st, &config)
        .map_err(|e| anyhow::Error::msg(format!("load chinese-clip weights: {e}")))?;
    let clip_model = ChineseClipModel {
        config: config.clone(),
        weights,
    };
    tracing::info!("Transformer loaded. ");

    let vec_imgs = match args.images.clone() {
        Some(imgs) => imgs,
        None => vec![
            "fuel-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg".to_string(),
            "fuel-examples/examples/yolo-v8/assets/bike.jpg".to_string(),
        ],
    };

    // Per-image features (1, projection_dim).
    let mut image_feats: Vec<Vec<f32>> = Vec::with_capacity(vec_imgs.len());
    for img_path in &vec_imgs {
        let pixels = load_image_as_vec(img_path, config.vision.image_size)?;
        let pixels = LazyTensor::from_f32(
            Arc::<[f32]>::from(pixels),
            Shape::from_dims(&[1, 3, config.vision.image_size, config.vision.image_size]),
            &device,
        );
        let f = clip_model.get_image_features(&pixels)?;
        image_feats.push(f.realize_f32());
    }
    tracing::info!("Images loaded. ");

    let tokenizer = load_tokenizer()?;
    let (token_lists, text_sequences) =
        tokenize_sequences(args.sequences, &tokenizer)?;

    // Per-text features (1, projection_dim).
    let mut text_feats: Vec<Vec<f32>> = Vec::with_capacity(token_lists.len());
    for tokens in &token_lists {
        let f = clip_model.get_text_features(tokens)?;
        text_feats.push(f.realize_f32());
    }

    tracing::info!("Computing ... ");

    // Contrastive logits: scale * (l2norm(text) @ l2norm(image).T).
    // We build logits_per_image[image_i][text_j] directly from the
    // already-realized feature vectors.
    let logit_scale = clip_model.weights.logit_scale.exp();
    let n_img = image_feats.len();
    let n_txt = text_feats.len();
    let mut logits_per_image = vec![0.0f32; n_img * n_txt];
    for (i, ifeat) in image_feats.iter().enumerate() {
        let i_norm = l2_norm(ifeat);
        for (j, tfeat) in text_feats.iter().enumerate() {
            let t_norm = l2_norm(tfeat);
            let dot: f32 = ifeat.iter().zip(tfeat.iter())
                .map(|(a, b)| a * b).sum();
            let denom = (i_norm * t_norm).max(1e-12);
            logits_per_image[i * n_txt + j] = logit_scale * dot / denom;
        }
    }

    // Softmax across text axis for each image.
    let mut probability_vec = Vec::with_capacity(n_img * n_txt);
    for i in 0..n_img {
        let row = &logits_per_image[i * n_txt..(i + 1) * n_txt];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        for e in exp {
            probability_vec.push(100.0 * e / sum.max(1e-30));
        }
    }

    let probability_per_image = probability_vec.len() / vec_imgs.len();

    for (i, img) in vec_imgs.iter().enumerate() {
        let start = i * probability_per_image;
        let end = start + probability_per_image;
        let prob = &probability_vec[start..end];
        tracing::info!("\n\nResults for image: {}\n", img);

        for (i, p) in prob.iter().enumerate() {
            tracing::info!("Probability: {:.4}% Text: {} ", p, text_sequences[i]);
        }
    }

    Ok(())
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn load_tokenizer() -> Result<Tokenizer> {
    let tokenizer_file = {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = hf_hub::Repo::with_revision(
            "OFA-Sys/chinese-clip-vit-base-patch16".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/3".to_string(),
        );
        let api = api.repo(repo);
        api.get("tokenizer.json")?
    };

    Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)
}

/// Tokenize each sequence and pad to `max_len` with [PAD]. Returns
/// the per-sequence token lists and the original strings.
pub fn tokenize_sequences(
    sequences: Option<Vec<String>>,
    tokenizer: &Tokenizer,
) -> Result<(Vec<Vec<u32>>, Vec<String>)> {
    let vec_seq = match sequences {
        Some(seq) => seq,
        None => vec![
            "自行车比赛".to_string(),
            "两只猫咪".to_string(),
            "拿着蜡烛的机器人".to_string(),
        ],
    };

    let mut input_ids: Vec<Vec<u32>> = vec![];
    let mut max_len = 0;

    for seq in vec_seq.clone() {
        let encoding = tokenizer.encode(seq, true).map_err(anyhow::Error::msg)?;
        input_ids.push(encoding.get_ids().to_vec());
        if encoding.get_ids().len() > max_len {
            max_len = encoding.get_ids().len();
        }
    }

    let pad_id = *tokenizer
        .get_vocab(true)
        .get("[PAD]")
        .ok_or(anyhow::Error::msg("No pad token"))?;

    let input_ids: Vec<Vec<u32>> = input_ids
        .iter_mut()
        .map(|item| {
            item.extend(vec![pad_id; max_len - item.len()]);
            item.to_vec()
        })
        .collect();

    Ok((input_ids, vec_seq))
}

/// Load image, resize to (image_size, image_size), apply OpenAI
/// normalization, return CHW row-major f32 vector.
fn load_image_as_vec<T: AsRef<std::path::Path>>(
    path: T, image_size: usize,
) -> Result<Vec<f32>> {
    let img = image::ImageReader::open(path)?.decode()?;
    let img = img.resize_to_fill(
        image_size as u32, image_size as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8().into_raw(); // HWC u8

    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    let std = [0.26862954f32, 0.261_302_6, 0.275_777_1];

    let h = image_size;
    let w = image_size;
    let mut out = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = img[(y * w + x) * 3 + c] as f32 / 255.0;
                let v = (v - mean[c]) / std[c];
                out[(c * h + y) * w + x] = v;
            }
        }
    }
    Ok(out)
}

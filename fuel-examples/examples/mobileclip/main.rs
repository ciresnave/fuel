#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use std::sync::Arc;

use fuel::lazy::LazyTensor;
use fuel::lazy_mobileclip::{MobileClipConfig, MobileClipModel, MobileClipWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};

use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    S1,
    S2,
}

impl Which {
    fn model_name(&self) -> String {
        let name = match self {
            Self::S1 => "S1",
            Self::S2 => "S2",
        };
        format!("apple/MobileCLIP-{name}-OpenCLIP")
    }

    fn config(&self) -> MobileClipConfig {
        match self {
            Self::S1 => MobileClipConfig::s1(),
            Self::S2 => MobileClipConfig::s2(),
        }
    }
}

#[derive(Parser)]
struct Args {
    #[arg(long, use_value_delimiter = true)]
    images: Option<Vec<String>>,

    #[arg(long)]
    cpu: bool,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long)]
    use_pth: bool,

    #[arg(long, use_value_delimiter = true)]
    sequences: Option<Vec<String>>,

    #[arg(value_enum, long, default_value_t=Which::S1)]
    which: Which,
}

/// Load image, resize to (image_size, image_size), CHW f32, no mean/std
/// normalization (matches the eager `load_image_with_std_mean(_, _, [0;3], [1;3])`
/// call used by the original binary).
fn load_image_as_vec<P: AsRef<std::path::Path>>(
    path: P, image_size: usize,
) -> Result<Vec<f32>> {
    let img = image::ImageReader::open(path)?
        .decode()
        .map_err(|e| E::msg(format!("decode image: {e}")))?;
    let img = img.resize_to_fill(
        image_size as u32, image_size as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8().into_raw(); // HWC u8
    let h = image_size;
    let w = image_size;
    let mut out = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = img[(y * w + x) * 3 + c] as f32 / 255.0;
                out[(c * h + y) * w + x] = v;
            }
        }
    }
    Ok(out)
}

pub fn main() -> Result<()> {
    let args = Args::parse();

    let model_name = args.which.model_name();
    let api = hf_hub::api::sync::Api::new()?;
    let api = api.model(model_name);
    if args.use_pth {
        anyhow::bail!("--use-pth is not supported by the lazy MobileCLIP loader");
    }
    let model_file = api.get("open_clip_model.safetensors")?;
    let tokenizer = api.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let config = args.which.config();

    // `--cpu` is preserved for parity; lazy realize lives on CPU by
    // default in this binary.
    let _ = args.cpu;
    let device = Device::cpu();

    let vec_imgs = match args.images {
        Some(imgs) => imgs,
        None => vec![
            "fuel-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg".to_string(),
            "fuel-examples/examples/yolo-v8/assets/bike.jpg".to_string(),
        ],
    };

    let st = unsafe { MmapedSafetensors::multi(&[model_file]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = MobileClipWeights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load mobileclip weights: {e}")))?;
    let model = MobileClipModel { config: config.clone(), weights };

    // Tokenize sequences (returns padded `Vec<Vec<u32>>` + sequences).
    let (token_lists, vec_seq) = tokenize_sequences(args.sequences, &tokenizer)?;

    // Per-image feature vectors (1, projection_dim).
    let mut image_feats: Vec<Vec<f32>> = Vec::with_capacity(vec_imgs.len());
    for img_path in &vec_imgs {
        let pixels = load_image_as_vec(img_path, config.image_size)?;
        let pixels = LazyTensor::from_f32(
            Arc::<[f32]>::from(pixels),
            Shape::from_dims(&[1, 3, config.image_size, config.image_size]),
            &device,
        );
        let f = model.get_image_features(&pixels)?;
        image_feats.push(f.realize_f32());
    }

    // Per-sequence feature vectors (1, projection_dim). The eot
    // position for OpenCLIP-style text encoders is the highest-id
    // token in the padded sequence (mirrors argmax(input_ids) in the
    // eager forward).
    let anchor = LazyTensor::from_f32(
        vec![0.0_f32], Shape::from_dims(&[1]), &device,
    );
    let mut text_feats: Vec<Vec<f32>> = Vec::with_capacity(token_lists.len());
    for tokens in &token_lists {
        let eot_pos = argmax_u32(tokens);
        let f = model.get_text_features(tokens, eot_pos, &anchor)?;
        text_feats.push(f.realize_f32());
    }

    // Build `logits_per_image[image][text]` from already-realized
    // feature vectors using the MobileCLIP contrastive scale.
    let logit_scale = model.weights.logit_scale.exp();
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

    // Softmax across the text axis for each image.
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
    println!("softmax_image_vec (percent): {probability_vec:?}");
    let probability_per_image = n_txt;
    for (i, img) in vec_imgs.iter().enumerate() {
        let start = i * probability_per_image;
        let end = start + probability_per_image;
        let prob = &probability_vec[start..end];
        println!("\n\nResults for image: {img}\n");
        for (i, p) in prob.iter().enumerate() {
            println!("Probability: {:.4}% Text: {}", p, vec_seq[i]);
        }
    }

    Ok(())
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn argmax_u32(tokens: &[u32]) -> usize {
    let mut best = 0usize;
    let mut best_val = tokens[0];
    for (i, &t) in tokens.iter().enumerate().skip(1) {
        if t > best_val {
            best_val = t;
            best = i;
        }
    }
    best
}

pub fn tokenize_sequences(
    sequences: Option<Vec<String>>,
    tokenizer: &Tokenizer,
) -> Result<(Vec<Vec<u32>>, Vec<String>)> {
    // The model does not work well if the text is padded using the <|endoftext|> token, using 0
    // as the original OpenCLIP code.
    let pad_id = 0;

    let vec_seq = match sequences {
        Some(seq) => seq,
        None => vec![
            "a cycling race".to_string(),
            "a photo of two cats".to_string(),
            "a robot holding a fuel".to_string(),
        ],
    };

    let mut tokens = vec![];
    for seq in vec_seq.clone() {
        let encoding = tokenizer.encode(seq, true).map_err(E::msg)?;
        tokens.push(encoding.get_ids().to_vec());
    }

    let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);
    // Pad the sequences to have the same length.
    for token_vec in tokens.iter_mut() {
        let len_diff = max_len - token_vec.len();
        if len_diff > 0 {
            token_vec.extend(vec![pad_id; len_diff]);
        }
    }
    Ok((tokens, vec_seq))
}

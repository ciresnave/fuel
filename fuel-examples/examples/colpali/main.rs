use anyhow::{Error as E, Result};
use clap::Parser;
use image::DynamicImage;
use pdf2image::{RenderOptionsBuilder, PDF};
use std::sync::Arc;
use tokenizers::Tokenizer;

use fuel::lazy::LazyTensor;
use fuel::lazy_colpali::{ColPaliModel, ColPaliWeights};
use fuel::lazy_gemma::{GemmaActivation, GemmaConfig};
use fuel::lazy_paligemma::PaligemmaConfig;
use fuel::lazy_siglip::SiglipVisionConfig;
use fuel::{Device, Shape};
use hf_hub::{api::sync::Api, Repo, RepoType};

fn paligemma_3b_448_config() -> PaligemmaConfig {
    // PaliGemma-3B 448 — SigLIP-So400m image encoder at 448×448 paired with
    // the Gemma 2B decoder. Mirrors fuel_transformers' Config::paligemma_3b_448.
    PaligemmaConfig {
        vision_config: SiglipVisionConfig {
            patch_size: 14,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            hidden_size: 1152,
            intermediate_size: 4304,
            image_size: 448,
            num_channels: 3,
            hidden_activation: fuel::lazy_siglip::SiglipActivation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        },
        text_config: GemmaConfig {
            vocab_size: 257_216,
            hidden_size: 2048,
            intermediate_size: 16_384,
            num_hidden_layers: 18,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 8192,
            attention_bias: false,
            hidden_activation: GemmaActivation::GeluPytorchTanh,
        },
        projection_dim: 2048,
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    prompt: String,

    /// number of top pages to show.
    #[arg(long, default_value_t = 3)]
    top_k: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    pdf: String,

    #[arg(long)]
    start: Option<u32>,

    #[arg(long)]
    end: Option<u32>,
}

fn image_to_chw(image: &DynamicImage, image_size: usize) -> Vec<f32> {
    let img = image.resize_to_fill(
        image_size as u32,
        image_size as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8();
    let raw = img.into_raw();
    // Same affine(2/255, -1) normalization as the eager PaliGemma path.
    let mut chw = vec![0f32; 3 * image_size * image_size];
    for h in 0..image_size {
        for w in 0..image_size {
            for c in 0..3 {
                let v = raw[(h * image_size + w) * 3 + c] as f32;
                chw[c * image_size * image_size + h * image_size + w] = v * (2.0 / 255.0) - 1.0;
            }
        }
    }
    chw
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    let _ = fuel_examples::device(args.cpu)?;

    let api = Api::new()?;
    let model_id = match &args.model_id {
        Some(model_id) => model_id.to_string(),
        None => "vidore/colpali-v1.2-merged".to_string(),
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));

    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => api
            .repo(Repo::with_revision(
                "vidore/colpali".to_string(),
                RepoType::Model,
                "main".to_string(),
            ))
            .get("tokenizer.json")?,
    };

    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };

    let start = std::time::Instant::now();
    let config = paligemma_3b_448_config();
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = ColPaliWeights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = ColPaliModel { config: config.clone(), weights };
    println!("loaded the model in {:?}", start.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let prompt_tokens = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let dummy_tokens = tokenizer
        .encode("Describe the image", true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let pdf = PDF::from_file(args.pdf)?;
    let range = if let (Some(start), Some(end)) = (args.start, args.end) {
        pdf2image::Pages::Range(start..=end)
    } else {
        pdf2image::Pages::Range(1..=pdf.page_count())
    };
    let pages = pdf.render(range, RenderOptionsBuilder::default().build()?)?;

    // Per-text-token text embeddings (run once).
    let text_emb = model
        .forward_text(&prompt_tokens)
        .map_err(|e| E::msg(format!("forward_text: {e}")))?
        .realize_f32();
    let text_dim = fuel::lazy_colpali::COLPALI_PROJ_DIM;
    let text_seq = prompt_tokens.len();

    let img_size = config.vision_config.image_size;
    let np = config.vision_config.num_patches();
    let img_seq = np + dummy_tokens.len();

    let mut all_scores: Vec<f32> = Vec::with_capacity(pages.len());
    for page in &pages {
        let chw = image_to_chw(page, img_size);
        let pixel_values = LazyTensor::from_f32(
            Arc::<[f32]>::from(chw),
            Shape::from_dims(&[1, 3, img_size, img_size]),
            &Device::cpu(),
        );
        let image_emb = model
            .forward_images(&pixel_values, &dummy_tokens)
            .map_err(|e| E::msg(format!("forward_images: {e}")))?
            .realize_f32();
        // ColBERT MaxSim score: sum over text tokens of max over image tokens
        // of inner product.
        let mut score = 0f32;
        for t in 0..text_seq {
            let t_off = t * text_dim;
            let t_vec = &text_emb[t_off..t_off + text_dim];
            let mut best = f32::NEG_INFINITY;
            for i in 0..img_seq {
                let i_off = i * text_dim;
                let i_vec = &image_emb[i_off..i_off + text_dim];
                let mut dot = 0f32;
                for k in 0..text_dim {
                    dot += t_vec[k] * i_vec[k];
                }
                if dot > best {
                    best = dot;
                }
            }
            score += best;
        }
        all_scores.push(score);
    }

    let mut indices: Vec<usize> = (0..all_scores.len()).collect();
    indices.sort_by(|a, b| all_scores[*b].partial_cmp(&all_scores[*a]).unwrap());
    let top = args.top_k.min(indices.len());
    let top_k_indices = &indices[0..top];

    println!("Prompt: {}", args.prompt);
    println!("top {} page numbers that contain similarity to the prompt", top);
    println!("-----------------------------------");
    for index in top_k_indices {
        println!("Page: {:?}", index + 1);
    }
    println!("-----------------------------------");
    Ok(())
}

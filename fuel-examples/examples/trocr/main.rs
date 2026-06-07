#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use std::sync::Arc;

use fuel::lazy::LazyTensor;
use fuel::lazy_trocr::{TrocrActivation, TrocrDecoderConfig, TrocrModel};
use fuel::lazy_vit::{VitActivation, VitConfig};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use fuel_examples::token_output_stream::TokenOutputStream;

use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, ValueEnum)]
enum Which {
    #[value(name = "base")]
    BaseHandwritten,
    #[value(name = "large")]
    LargeHandwritten,
    #[value(name = "base-printed")]
    BasePrinted,
    #[value(name = "large-printed")]
    LargePrinted,
}

impl Which {
    fn default_model_id(&self) -> &'static str {
        match self {
            Self::BaseHandwritten => "microsoft/trocr-base-handwritten",
            Self::LargeHandwritten => "microsoft/trocr-large-handwritten",
            Self::BasePrinted => "microsoft/trocr-base-printed",
            Self::LargePrinted => "microsoft/trocr-large-printed",
        }
    }

    fn default_revision(&self) -> &'static str {
        match self {
            Self::BaseHandwritten => "refs/pr/3",
            Self::LargeHandwritten => "refs/pr/6",
            Self::BasePrinted => "refs/pr/7",
            Self::LargePrinted => "main",
        }
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Path to a local `model.safetensors` (overrides --model-id).
    #[arg(long)]
    model: Option<String>,

    /// HuggingFace repo id (e.g. `microsoft/trocr-base-handwritten`).
    /// If omitted, the canonical repo for `--which` is used.
    #[arg(long)]
    model_id: Option<String>,

    /// HuggingFace revision (branch, tag, or PR ref).
    #[arg(long)]
    revision: Option<String>,

    /// Path to a local `tokenizer.json` (overrides the HF download).
    #[arg(long)]
    tokenizer: Option<String>,

    /// Choose the variant of the model to run.
    #[arg(long, default_value = "base")]
    which: Which,

    /// Run on CPU rather than on GPU.
    ///
    /// Preserved for CLI parity with the eager binary; lazy ports
    /// currently realize via the default executor (CPU / router).
    #[arg(long)]
    cpu: bool,

    /// The image file to be processed.
    #[arg(long)]
    image: String,

    /// Greedy-decoding maximum new tokens.
    #[arg(long, default_value_t = 1000)]
    max_new_tokens: usize,
}

// ---- HF config.json parsing -------------------------------------------------
//
// TrOCR ships a nested HF config.json:
//   {
//     "encoder": { ...ViT fields... },
//     "decoder": { ...BART-style fields... },
//     ...
//   }

#[derive(Debug, Clone, serde::Deserialize)]
struct HfConfig {
    encoder: HfVitConfig,
    decoder: HfTrocrDecoderConfig,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
struct HfVitConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    hidden_act: String,
    layer_norm_eps: f64,
    image_size: usize,
    patch_size: usize,
    num_channels: usize,
    qkv_bias: bool,
}

impl Default for HfVitConfig {
    fn default() -> Self {
        // microsoft/trocr-base-handwritten encoder defaults.
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            layer_norm_eps: 1e-12,
            image_size: 384,
            patch_size: 16,
            num_channels: 3,
            qkv_bias: false,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
struct HfTrocrDecoderConfig {
    vocab_size: usize,
    /// Some HF configs surface `decoder_vocab_size` distinct from
    /// `vocab_size`. When present we prefer it for the LM head /
    /// embedding table sizing.
    decoder_vocab_size: Option<usize>,
    d_model: usize,
    cross_attention_hidden_size: usize,
    decoder_layers: usize,
    decoder_attention_heads: usize,
    decoder_ffn_dim: usize,
    activation_function: String,
    max_position_embeddings: usize,
    scale_embedding: bool,
    tie_word_embeddings: bool,
    decoder_start_token_id: u32,
    eos_token_id: u32,
}

impl Default for HfTrocrDecoderConfig {
    fn default() -> Self {
        // microsoft/trocr-base-handwritten decoder defaults.
        Self {
            vocab_size: 50265,
            decoder_vocab_size: None,
            d_model: 1024,
            cross_attention_hidden_size: 768,
            decoder_layers: 12,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            activation_function: "gelu".to_string(),
            max_position_embeddings: 512,
            scale_embedding: false,
            tie_word_embeddings: true,
            decoder_start_token_id: 2,
            eos_token_id: 2,
        }
    }
}

fn map_vit_activation(name: &str) -> Result<VitActivation> {
    Ok(match name {
        "gelu" => VitActivation::Gelu,
        "gelu_pytorch_tanh" | "gelu_new" => VitActivation::GeluPytorchTanh,
        "relu" => VitActivation::Relu,
        "silu" | "swish" => VitActivation::Silu,
        other => anyhow::bail!("unsupported ViT activation: {other}"),
    })
}

fn map_trocr_activation(name: &str) -> Result<TrocrActivation> {
    Ok(match name {
        "gelu" => TrocrActivation::Gelu,
        "relu" => TrocrActivation::Relu,
        other => anyhow::bail!("unsupported TrOCR decoder activation: {other}"),
    })
}

fn vit_config_from_hf(c: &HfVitConfig) -> Result<VitConfig> {
    Ok(VitConfig {
        hidden_size: c.hidden_size,
        num_hidden_layers: c.num_hidden_layers,
        num_attention_heads: c.num_attention_heads,
        intermediate_size: c.intermediate_size,
        hidden_activation: map_vit_activation(&c.hidden_act)?,
        layer_norm_eps: c.layer_norm_eps,
        image_size: c.image_size,
        patch_size: c.patch_size,
        num_channels: c.num_channels,
        qkv_bias: c.qkv_bias,
    })
}

fn decoder_config_from_hf(c: &HfTrocrDecoderConfig) -> Result<TrocrDecoderConfig> {
    Ok(TrocrDecoderConfig {
        vocab_size: c.decoder_vocab_size.unwrap_or(c.vocab_size),
        d_model: c.d_model,
        cross_attention_hidden_size: c.cross_attention_hidden_size,
        decoder_layers: c.decoder_layers,
        decoder_attention_heads: c.decoder_attention_heads,
        decoder_ffn_dim: c.decoder_ffn_dim,
        activation_function: map_trocr_activation(&c.activation_function)?,
        max_position_embeddings: c.max_position_embeddings,
        // HF convention: learned positional embedding offset is 2.
        learned_pos_offset: 2,
        scale_embedding: c.scale_embedding,
        tie_word_embeddings: c.tie_word_embeddings,
    })
}

// ---- Image preprocessing ----------------------------------------------------

/// Loads an image from disk, resizes to (image_size x image_size),
/// rescales to [0, 1], normalizes with TrOCR mean/std = [0.5, 0.5, 0.5],
/// and returns a flat row-major `Vec<f32>` of length `3 * H * W`
/// laid out as (C, H, W). Mirrors the eager `ViTImageProcessor`
/// defaults (do_resize=true, do_rescale=true, do_normalize=true).
fn load_image_as_vec<P: AsRef<std::path::Path>>(p: P, image_size: usize) -> Result<Vec<f32>> {
    let img = image::ImageReader::open(p)?
        .decode()?
        .resize_exact(
            image_size as u32,
            image_size as u32,
            image::imageops::FilterType::Triangle,
        );
    let img = img.to_rgb8();
    let raw = img.into_raw(); // (H, W, C) row-major, u8

    let mean = [0.5_f32, 0.5, 0.5];
    let std = [0.5_f32, 0.5, 0.5];

    let h = image_size;
    let w = image_size;
    let mut out = vec![0.0_f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = raw[(y * w + x) * 3 + c] as f32 / 255.0;
                let v = (v - mean[c]) / std[c];
                out[(c * h + y) * w + x] = v;
            }
        }
    }
    Ok(out)
}

// ---- Main -------------------------------------------------------------------

pub fn main() -> Result<()> {
    let args = Args::parse();

    let api = hf_hub::api::sync::Api::new()?;
    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| args.which.default_model_id().to_string());
    let revision = args
        .revision
        .clone()
        .unwrap_or_else(|| args.which.default_revision().to_string());

    // ---- Tokenizer ----
    let tokenizer_file = match args.tokenizer.clone() {
        Some(file) => std::path::PathBuf::from(file),
        None => {
            let repo = api.repo(hf_hub::Repo::with_revision(
                model_id.clone(),
                hf_hub::RepoType::Model,
                revision.clone(),
            ));
            repo.get("tokenizer.json")?
        }
    };
    let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(E::msg)?;
    let mut tokenizer = TokenOutputStream::new(tokenizer);

    // ---- Lazy realizes via the default executor; --cpu is parity-only ----
    let _ = args.cpu;
    let device = Device::cpu();

    // ---- Model weights ----
    let model_file = match args.model.clone() {
        Some(model) => std::path::PathBuf::from(model),
        None => {
            let repo = api.repo(hf_hub::Repo::with_revision(
                model_id.clone(),
                hf_hub::RepoType::Model,
                revision.clone(),
            ));
            repo.get("model.safetensors")?
        }
    };
    println!("model: {model_file:?}");

    // ---- HF config.json ----
    let config_file = {
        let repo = api.repo(hf_hub::Repo::with_revision(
            model_id.clone(),
            hf_hub::RepoType::Model,
            revision.clone(),
        ));
        repo.get("config.json")?
    };
    let hf_config: HfConfig =
        serde_json::from_reader(std::fs::File::open(&config_file)?)?;
    let decoder_start_token_id = hf_config.decoder.decoder_start_token_id;
    let eos_token_id = hf_config.decoder.eos_token_id;

    let encoder_config = vit_config_from_hf(&hf_config.encoder)?;
    let decoder_config = decoder_config_from_hf(&hf_config.decoder)?;

    println!("loading model weights");
    let st = unsafe { MmapedSafetensors::multi(&[&model_file]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let model = TrocrModel::load_from_mmapped(
        &st,
        encoder_config.clone(),
        decoder_config.clone(),
    )
    .map_err(|e| E::msg(format!("load TrOCR weights: {e}")))?;
    println!("model built");

    // ---- Image ----
    let image_size = encoder_config.image_size;
    let image_vec = load_image_as_vec(&args.image, image_size)?;
    println!(
        "loaded image ({} f32 values, {image_size}x{image_size})",
        image_vec.len()
    );
    let pixel_values = LazyTensor::from_f32(
        Arc::<[f32]>::from(image_vec),
        Shape::from_dims(&[1, 3, image_size, image_size]),
        &device,
    );

    // Encode once; the lazy decoder takes the encoder output as a
    // graph anchor and re-uses it across autoregressive steps.
    let encoder_xs = model
        .forward_encoder(&pixel_values)
        .map_err(|e| E::msg(format!("encoder forward: {e}")))?;

    // ---- Greedy decoding ----
    let vocab_size = decoder_config.vocab_size;
    let mut token_ids: Vec<u32> = vec![decoder_start_token_id];
    for _ in 0..args.max_new_tokens {
        // The lazy decoder takes the FULL target sequence each step
        // (no KV cache yet). Greedy argmax over the last position —
        // matches blip's v1 sampling.
        let logits = model
            .forward_decoder(&token_ids, &encoder_xs)
            .map_err(|e| E::msg(format!("decoder forward: {e}")))?;
        let data = logits.realize_f32();
        let seq = token_ids.len();
        let off = (seq - 1) * vocab_size;
        let last_logits = &data[off..off + vocab_size];

        let mut best_i = 0usize;
        let mut best = last_logits[0];
        for (i, &v) in last_logits.iter().enumerate().skip(1) {
            if v > best {
                best = v;
                best_i = i;
            }
        }
        let token = best_i as u32;
        token_ids.push(token);

        if let Some(t) = tokenizer.next_token(token)? {
            use std::io::Write;
            print!("{t}");
            std::io::stdout().flush()?;
        }
        if token == eos_token_id {
            break;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    println!();
    Ok(())
}

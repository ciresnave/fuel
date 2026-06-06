//! Z-Image Text-to-Image Generation — lazy port migration.
//!
//! Z-Image is Alibaba's text-to-image Flow Matching model. The eager
//! pipeline ran through `fuel_transformers::models::z_image`. This
//! binary now constructs the model through the lazy port at
//! `fuel::lazy_z_image` and drives the full generate-image pipeline
//! through `ZImageModel::generate` (text encoder + DiT + VAE + Flow
//! Match Euler scheduler — all four components ship through one
//! `load_from_mmapped` per Weights struct).
//!
//! # Running
//!
//! ```bash
//! cargo run --features metal --example z_image --release -- \
//!     --model turbo \
//!     --prompt "A beautiful landscape with mountains and a lake" \
//!     --height 1024 --width 1024 --num-steps 9
//! ```
//!
//! # Model files
//!
//! Models auto-download from <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>.

use anyhow::{Error as E, Result};
use clap::Parser;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use fuel::lazy_z_image::{
    AutoEncoderKL, TextEncoderConfig, TextEncoderWeights, VaeConfig, VaeWeights,
    ZImageConfig, ZImageModel, ZImageTextEncoder, ZImageTransformer2DModel,
    ZImageTransformerWeights,
};

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum Model {
    /// Z-Image-Turbo: optimized for fast inference (8-9 steps)
    Turbo,
}

impl Model {
    fn repo(&self) -> &'static str {
        match self {
            Self::Turbo => "Tongyi-MAI/Z-Image-Turbo",
        }
    }

    fn default_steps(&self) -> usize {
        match self {
            Self::Turbo => 9,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A beautiful landscape with mountains and a lake"
    )]
    prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The height in pixels of the generated image.
    #[arg(long, default_value_t = 1024)]
    height: usize,

    /// The width in pixels of the generated image.
    #[arg(long, default_value_t = 1024)]
    width: usize,

    /// Number of inference steps.
    #[arg(long)]
    num_steps: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value = "turbo")]
    model: Model,

    /// Override path to the model weights directory (uses HuggingFace by default).
    #[arg(long)]
    model_path: Option<String>,

    /// Output image filename.
    #[arg(long, default_value = "z_image_output.png")]
    output: String,
}

/// Format user prompt for Qwen3 chat template
/// (add_generation_prompt=True, enable_thinking=True).
fn format_prompt_for_qwen3(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    )
}

fn run(args: Args) -> Result<()> {
    let num_steps = args.num_steps.unwrap_or_else(|| args.model.default_steps());

    println!("Z-Image Text-to-Image Generation");
    println!("================================");
    println!("Model: {:?}", args.model);
    println!("Prompt: {}", args.prompt);
    println!("Size: {}x{}", args.width, args.height);
    println!("Steps: {}", num_steps);

    let _device = fuel_examples::device(args.cpu)?;

    let api = Api::new()?;
    let repo = api.model(args.model.repo().to_string());
    let use_local = args.model_path.is_some();
    let model_path = args.model_path.map(std::path::PathBuf::from);

    if use_local {
        println!(
            "\nLoading models from local path: {}",
            model_path.as_ref().unwrap().display()
        );
    } else {
        println!(
            "\nDownloading model from HuggingFace: {}",
            args.model.repo()
        );
    }

    // ==================== Load tokenizer ====================
    println!("Loading tokenizer...");
    let tokenizer_path = if use_local {
        model_path
            .as_ref()
            .unwrap()
            .join("tokenizer")
            .join("tokenizer.json")
    } else {
        repo.get("tokenizer/tokenizer.json")?
    };
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

    // ==================== Load text encoder ====================
    println!("Loading text encoder...");
    let text_encoder_cfg = TextEncoderConfig::z_image();
    let text_encoder_files: Vec<std::path::PathBuf> = if use_local {
        (1..=3)
            .map(|i| {
                model_path
                    .as_ref()
                    .unwrap()
                    .join("text_encoder")
                    .join(format!("model-{:05}-of-00003.safetensors", i))
            })
            .filter(|p| p.exists())
            .collect()
    } else {
        (1..=3)
            .map(|i| repo.get(&format!("text_encoder/model-{:05}-of-00003.safetensors", i)))
            .filter_map(|r| r.ok())
            .collect()
    };
    if text_encoder_files.is_empty() {
        anyhow::bail!("Text encoder weights not found");
    }
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&text_encoder_files)? };
    let text_encoder_weights = TextEncoderWeights::load_from_mmapped(&st, &text_encoder_cfg)?;
    let text_encoder = ZImageTextEncoder {
        config: text_encoder_cfg,
        weights: text_encoder_weights,
    };

    // ==================== Load transformer ====================
    println!("Loading transformer...");
    let transformer_cfg = ZImageConfig::z_image_turbo();
    let transformer_files: Vec<std::path::PathBuf> = if use_local {
        (1..=3)
            .map(|i| {
                model_path
                    .as_ref()
                    .unwrap()
                    .join("transformer")
                    .join(format!(
                        "diffusion_pytorch_model-{:05}-of-00003.safetensors",
                        i
                    ))
            })
            .filter(|p| p.exists())
            .collect()
    } else {
        (1..=3)
            .map(|i| {
                repo.get(&format!(
                    "transformer/diffusion_pytorch_model-{:05}-of-00003.safetensors",
                    i
                ))
            })
            .filter_map(|r| r.ok())
            .collect()
    };
    if transformer_files.is_empty() {
        anyhow::bail!("Transformer weights not found");
    }
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&transformer_files)? };
    let transformer_weights = ZImageTransformerWeights::load_from_mmapped(&st, &transformer_cfg)?;
    let transformer = ZImageTransformer2DModel {
        config: transformer_cfg.clone(),
        weights: transformer_weights,
    };

    // ==================== Load VAE ====================
    println!("Loading VAE...");
    let vae_cfg = VaeConfig::z_image();
    let vae_path = if use_local {
        let path = model_path
            .as_ref()
            .unwrap()
            .join("vae")
            .join("diffusion_pytorch_model.safetensors");
        if !path.exists() {
            anyhow::bail!("VAE weights not found at {:?}", path);
        }
        path
    } else {
        repo.get("vae/diffusion_pytorch_model.safetensors")?
    };
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[vae_path])? };
    let vae_weights = VaeWeights::load_from_mmapped(&st, &vae_cfg)?;
    let vae = AutoEncoderKL {
        config: vae_cfg,
        weights: vae_weights,
    };

    // ==================== Build top-level model ====================
    let model = ZImageModel {
        text_encoder,
        transformer,
        vae,
    };

    // ==================== Tokenize prompt ====================
    println!("\nTokenizing prompt...");
    let formatted_prompt = format_prompt_for_qwen3(&args.prompt);
    let tokens = tokenizer
        .encode(formatted_prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    println!("Token count: {}", tokens.len());

    // ==================== Calculate latent dimensions ====================
    let patch_size = transformer_cfg.patch_size;
    let vae_align = 16; // vae_scale_factor * 2 = 8 * 2
    if !args.height.is_multiple_of(vae_align) || !args.width.is_multiple_of(vae_align) {
        anyhow::bail!(
            "Image dimensions must be divisible by {}. Got {}x{}.",
            vae_align,
            args.width,
            args.height,
        );
    }
    let latent_h = 2 * (args.height / vae_align);
    let latent_w = 2 * (args.width / vae_align);
    let _ = patch_size;
    println!("Latent size: {}x{}", latent_w, latent_h);

    // ==================== Generate ====================
    println!("\nGenerating image ({} steps, seed {})...", num_steps, args.seed);
    let image = model.generate(&tokens, latent_h, latent_w, num_steps, args.seed)?;
    let image_data = image.realize_f32();
    let dims = image.shape().dims().to_vec();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
        anyhow::bail!("Unexpected output shape: {:?}", dims);
    }
    let (c, h, w) = (dims[1], dims[2], dims[3]);

    // Post-process: [-1, 1] -> [0, 255].
    let mut chw_u8 = vec![0_u8; c * h * w];
    for (i, &v) in image_data.iter().enumerate() {
        let scaled = ((v.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round() as u8;
        chw_u8[i] = scaled;
    }

    let eager_u8 = fuel::Tensor::from_vec(chw_u8, (c, h, w), &fuel::Device::cpu())?;
    println!("Saving image to {}...", args.output);
    fuel_examples::save_image(&eager_u8, &args.output)?;

    println!("\nDone! Image saved to {}", args.output);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

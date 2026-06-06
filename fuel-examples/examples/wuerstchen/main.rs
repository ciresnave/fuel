//! Wuerstchen — lazy port migration.
//!
//! This binary used to drive the full Wuerstchen v2 cascaded LDM
//! pipeline (Stage A/B/C + CLIP text encoders) through eager
//! `fuel-transformers`. Round-9a migrates the model-construction
//! and weight-loading surface to the lazy port at
//! `fuel::lazy_wuerstchen`. The Stage C/B/A weight bags now load
//! through `Weights::load_from_mmapped`; the diffusion loop itself
//! is left as a TODO once a lazy CLIP text encoder is shipped (the
//! eager `stable_diffusion::clip` text encoder is not yet ported).

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use clap::Parser;

use fuel::lazy_wuerstchen::{
    DiffNextModel, DiffNextWeights, PaellaVqModel, PaellaVqWeights, PriorModel,
    PriorWeights, WuerstchenConfig,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: String,

    #[arg(long, default_value = "")]
    uncond_prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The decoder weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    decoder_weights: Option<String>,

    /// The prior weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    prior_weights: Option<String>,

    /// The VQGAN weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    vqgan_weights: Option<String>,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE", default_value = "sd_final.png")]
    final_image: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Decoder,
    VqGan,
    Prior,
}

impl ModelFile {
    fn get(&self, filename: Option<String>) -> Result<std::path::PathBuf> {
        use hf_hub::api::sync::Api;
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let repo_main = "warp-ai/wuerstchen";
                let repo_prior = "warp-ai/wuerstchen-prior";
                let (repo, path) = match self {
                    Self::Decoder => (repo_main, "decoder/diffusion_pytorch_model.safetensors"),
                    Self::VqGan => (repo_main, "vqgan/diffusion_pytorch_model.safetensors"),
                    Self::Prior => (repo_prior, "prior/diffusion_pytorch_model.safetensors"),
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

/// Production Wuerstchen config from
/// <https://huggingface.co/warp-ai/wuerstchen-prior/blob/main/prior/config.json>
/// and the matching decoder/vqgan configs. Tracks the constants the
/// eager binary hard-coded inside `run`.
fn wuerstchen_config() -> WuerstchenConfig {
    WuerstchenConfig {
        // Stage C (Prior).
        prior_c_in: 16,
        prior_c: 1536,
        prior_c_cond: 1280,
        c_r: 64,
        prior_depth: 32,
        prior_nhead: 24,

        // Stage B (DiffNeXt).
        diffnext_c_in: 4,
        diffnext_c_out: 4,
        diffnext_c_cond: 1024,
        patch_size: 2,
        diffnext_c_hidden: vec![320, 640, 1280, 1280],
        diffnext_blocks: vec![2, 6, 28, 6],
        diffnext_nhead: vec![0, 0, 20, 20],

        // Stage A (Paella VQ).
        paella_latent_channels: 4,
        paella_levels: vec![384, 384, 192, 192],
        paella_bottleneck_blocks: 12,
        paella_out_channels: 3,

        clip_embed: 1024,
        image_size: 1024,
    }
}

fn run(args: Args) -> Result<()> {
    let _device = fuel_examples::device(args.cpu)?;
    let cfg = wuerstchen_config();

    println!("Prompt: {}", args.prompt);
    println!("Uncond: {}", args.uncond_prompt);

    println!("Loading the prior (Stage C).");
    let prior_file = ModelFile::Prior.get(args.prior_weights)?;
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[prior_file])? };
    let prior_weights = PriorWeights::load_from_mmapped(&st, &cfg)?;
    let _prior = PriorModel {
        config: cfg.clone(),
        weights: prior_weights,
    };

    println!("Loading the decoder (Stage B).");
    let decoder_file = ModelFile::Decoder.get(args.decoder_weights)?;
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[decoder_file])? };
    let decoder_weights = DiffNextWeights::load_from_mmapped(&st, &cfg)?;
    let _decoder = DiffNextModel {
        config: cfg.clone(),
        weights: decoder_weights,
    };

    println!("Loading the VQGAN (Stage A).");
    let vqgan_file = ModelFile::VqGan.get(args.vqgan_weights)?;
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[vqgan_file])? };
    let vqgan_weights = PaellaVqWeights::load_from_mmapped(&st, &cfg)?;
    let _vqgan = PaellaVqModel {
        config: cfg.clone(),
        weights: vqgan_weights,
    };

    // TODO(lazy-wuerstchen-pipeline): wire the prior-denoise → decoder-denoise
    // → vqgan-decode loop once a lazy CLIP-style text encoder is available
    // (the eager `stable_diffusion::clip` text encoder is not yet ported).
    println!(
        "Models loaded; lazy text-encoder + denoising loop is the next migration step."
    );
    println!("Target output: {}", args.final_image);

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

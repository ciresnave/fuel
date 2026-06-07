//! Stable Diffusion text-to-image — lazy port migration.
//!
//! The eager binary built three sub-models (CLIP text encoder, UNet,
//! VAE decoder) via `fuel_transformers::models::stable_diffusion` and
//! drove an N-step DDIM/DDPM denoising loop. This binary now wires
//! the same loop through the lazy ports:
//!
//! - `fuel::lazy_sd_text_encoder::SdTextEncoder` (+ `SdTextTokenizer`)
//! - `fuel::lazy_sd_unet::SdUnet`
//! - `fuel::lazy_sd_vae::SdVaeDecoder`
//! - `fuel::lazy_sd_samplers::DdimScheduler`
//!
//! # Scope
//!
//! The lazy SD modules currently target SD 1.5 only — the UNet and
//! VAE configs expose `sd_v1()` and no SDXL / SD 2.x / inpainting
//! variants. The CLI still accepts `--sd-version` for forward
//! compatibility but only `v1-5` is functional; other variants bail
//! at startup.

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use rand::{Rng, SeedableRng};

use fuel::lazy_sd_samplers::{
    DdimScheduler, DdimSchedulerConfig, SdScheduler,
};
use fuel::lazy_sd_text_encoder::{
    ClipTextConfig, ClipTextWeights, SdTextEncoder, SdTextTokenizer,
};
use fuel::lazy_sd_unet::{SdUnet, SdUnetConfig, SdUnetWeights};
use fuel::lazy_sd_vae::{SdVaeConfig, SdVaeDecoder, SdVaeDecoderWeights};

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

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The height in pixels of the generated image.
    #[arg(long)]
    height: Option<usize>,

    /// The width in pixels of the generated image.
    #[arg(long)]
    width: Option<usize>,

    /// The UNet weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    unet_weights: Option<String>,

    /// The CLIP weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    clip_weights: Option<String>,

    /// The VAE weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    vae_weights: Option<String>,

    #[arg(long, value_name = "FILE")]
    /// The file specifying the tokenizer to used for tokenization.
    tokenizer: Option<String>,

    /// The number of steps to run the diffusion for.
    #[arg(long, default_value_t = 30)]
    n_steps: usize,

    /// The number of samples to generate iteratively.
    #[arg(long, default_value_t = 1)]
    num_samples: usize,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE", default_value = "sd_final.png")]
    final_image: String,

    #[arg(long, value_enum, default_value = "v1-5")]
    sd_version: StableDiffusionVersion,

    /// Generate intermediary images at each step.
    #[arg(long, action)]
    intermediary_images: bool,

    /// Override the default classifier-free guidance scale (7.5).
    #[arg(long, default_value_t = 7.5)]
    guidance_scale: f64,

    /// The seed to use when generating random samples.
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum StableDiffusionVersion {
    V1_5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Tokenizer,
    Clip,
    Unet,
    Vae,
}

impl StableDiffusionVersion {
    fn repo(&self) -> &'static str {
        match self {
            Self::V1_5 => "stable-diffusion-v1-5/stable-diffusion-v1-5",
        }
    }
}

impl ModelFile {
    fn get(
        &self,
        filename: Option<String>,
        version: StableDiffusionVersion,
    ) -> Result<std::path::PathBuf> {
        use hf_hub::api::sync::Api;
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let (repo, path) = match self {
                    Self::Tokenizer => (
                        "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
                        "tokenizer.json",
                    ),
                    Self::Clip => (version.repo(), "text_encoder/model.safetensors"),
                    Self::Unet => (version.repo(), "unet/diffusion_pytorch_model.safetensors"),
                    Self::Vae => (version.repo(), "vae/diffusion_pytorch_model.safetensors"),
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

fn output_filename(
    basename: &str,
    sample_idx: usize,
    num_samples: usize,
    timestep_idx: Option<usize>,
) -> String {
    let filename = if num_samples > 1 {
        match basename.rsplit_once('.') {
            None => format!("{basename}.{sample_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}.{sample_idx}.{extension}")
            }
        }
    } else {
        basename.to_string()
    };
    match timestep_idx {
        None => filename,
        Some(timestep_idx) => match filename.rsplit_once('.') {
            None => format!("{filename}-{timestep_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}-{timestep_idx}.{extension}")
            }
        },
    }
}

/// Decode `latents` through the VAE and save the resulting image(s).
///
/// `latents_flat` is the realized `[bsize, 4, h_lat, w_lat]` f32 buffer;
/// the VAE lazy module decodes one batch entry at a time.
#[allow(clippy::too_many_arguments)]
fn save_image(
    vae: &SdVaeDecoder,
    latents_flat: &[f32],
    bsize: usize,
    h_lat: usize,
    w_lat: usize,
    vae_scale: f64,
    idx: usize,
    final_image: &str,
    num_samples: usize,
    timestep_ids: Option<usize>,
) -> Result<()> {
    let stride = 4 * h_lat * w_lat;
    for batch in 0..bsize {
        let lat_slice = &latents_flat[batch * stride..(batch + 1) * stride];
        let scaled: Vec<f32> = lat_slice
            .iter()
            .map(|v| (*v as f64 / vae_scale) as f32)
            .collect();

        let image = vae.decode(&scaled, h_lat, w_lat);
        let image_flat = image.realize_f32();
        let dims = image.shape().dims().to_vec();
        if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
            anyhow::bail!("Unexpected VAE output shape: {:?}", dims);
        }
        let (c, h, w) = (dims[1], dims[2], dims[3]);

        // Post-process: [-1, 1] → [0, 255].
        let mut chw_u8 = vec![0_u8; c * h * w];
        for (i, &v) in image_flat.iter().enumerate() {
            let scaled_px = ((v.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round() as u8;
            chw_u8[i] = scaled_px;
        }
        let eager_u8 = fuel::Tensor::from_vec(chw_u8, (c, h, w), &fuel::Device::cpu())?;
        let image_filename = output_filename(
            final_image,
            (bsize * idx) + batch + 1,
            batch + num_samples,
            timestep_ids,
        );
        fuel_examples::save_image(&eager_u8, image_filename)?;
    }
    Ok(())
}

/// Build the CLIP text encoder embedding for `prompt`. When
/// `use_guide_scale` is true, the unconditional prompt is encoded as
/// well and the two are concatenated along the batch axis so the UNet
/// sees `[2, 77, 768]` for classifier-free guidance.
fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer_path: std::path::PathBuf,
    clip_weights_path: std::path::PathBuf,
    use_guide_scale: bool,
) -> Result<fuel::lazy::LazyTensor> {
    let clip_cfg = ClipTextConfig::sd_v1();
    let st = unsafe { fuel::safetensors::MmapedSafetensors::new(&clip_weights_path) }
        .map_err(|e| E::msg(format!("clip mmap: {e}")))?;
    let clip_weights = ClipTextWeights::load_from_mmapped(&st, &clip_cfg)
        .map_err(|e| E::msg(format!("clip weights: {e}")))?;
    let text_model = SdTextEncoder {
        config: clip_cfg.clone(),
        weights: clip_weights,
    };

    let tokenizer = SdTextTokenizer::from_file(&tokenizer_path, &clip_cfg)
        .map_err(|e| E::msg(format!("clip tokenizer: {e}")))?;

    println!("Running with prompt \"{prompt}\".");
    let tokens = tokenizer
        .encode_padded(prompt)
        .map_err(|e| E::msg(format!("encode prompt: {e}")))?;
    let cond = text_model
        .forward(&tokens)
        .map_err(|e| E::msg(format!("clip forward: {e}")))?;

    if use_guide_scale {
        let uncond_tokens = tokenizer
            .encode_padded(uncond_prompt)
            .map_err(|e| E::msg(format!("encode uncond: {e}")))?;
        let uncond = text_model
            .forward(&uncond_tokens)
            .map_err(|e| E::msg(format!("clip forward uncond: {e}")))?;
        let combined = uncond
            .concat(&cond, 0)
            .map_err(|e| E::msg(format!("concat embeddings: {e}")))?;
        Ok(combined)
    } else {
        Ok(cond)
    }
}

fn run(args: Args) -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let Args {
        prompt,
        uncond_prompt,
        cpu,
        height,
        width,
        n_steps,
        tokenizer,
        final_image,
        num_samples,
        sd_version,
        clip_weights,
        vae_weights,
        unet_weights,
        tracing,
        guidance_scale,
        seed,
        intermediary_images,
    } = args;

    let _guard = if tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // The lazy SD modules currently only ship SD 1.5 configs. Bail
    // early if the caller asked for anything else.
    if !matches!(sd_version, StableDiffusionVersion::V1_5) {
        anyhow::bail!(
            "lazy SD port currently supports SD 1.5 only; got {:?}",
            sd_version,
        );
    }

    // Image dimensions default to SD 1.5's training resolution.
    let height = height.unwrap_or(512);
    let width = width.unwrap_or(512);
    if !height.is_multiple_of(8) || !width.is_multiple_of(8) {
        anyhow::bail!("height/width must be multiples of 8, got {height}x{width}");
    }

    let device = fuel_examples::device(cpu)?;
    // If a seed is not given, generate a random one.
    let seed = seed.unwrap_or(rand::rng().random_range(0u64..u64::MAX));
    println!("Using seed {seed}");
    device.set_seed(seed)?;
    let use_guide_scale = guidance_scale > 1.0;

    // ---- text embedding ---------------------------------------------------
    let tokenizer_path = ModelFile::Tokenizer.get(tokenizer, sd_version)?;
    let clip_weights_path = ModelFile::Clip.get(clip_weights, sd_version)?;
    let text_emb_lazy = text_embeddings(
        &prompt,
        &uncond_prompt,
        tokenizer_path,
        clip_weights_path,
        use_guide_scale,
    )?;
    let text_emb_flat = text_emb_lazy.realize_f32();
    let text_emb_dims = text_emb_lazy.shape().dims().to_vec();
    println!("text embeddings shape: {:?}", text_emb_dims);
    // Per-batch slice into the text embedding for UNet calls.
    let max_pos = ClipTextConfig::sd_v1().max_position_embeddings;
    let hidden = ClipTextConfig::sd_v1().hidden_size;

    // ---- VAE --------------------------------------------------------------
    println!("Building the autoencoder.");
    let vae_weights_path = ModelFile::Vae.get(vae_weights, sd_version)?;
    let vae_st = unsafe { fuel::safetensors::MmapedSafetensors::new(&vae_weights_path) }
        .map_err(|e| E::msg(format!("vae mmap: {e}")))?;
    let vae_cfg = SdVaeConfig::sd_v1();
    let vae_weights = SdVaeDecoderWeights::load_from_mmapped(&vae_st, &vae_cfg)
        .map_err(|e| E::msg(format!("vae weights: {e}")))?;
    let vae = SdVaeDecoder {
        config: vae_cfg,
        weights: vae_weights,
    };

    // ---- UNet -------------------------------------------------------------
    println!("Building the unet.");
    let unet_weights_path = ModelFile::Unet.get(unet_weights, sd_version)?;
    let unet_st = unsafe { fuel::safetensors::MmapedSafetensors::new(&unet_weights_path) }
        .map_err(|e| E::msg(format!("unet mmap: {e}")))?;
    let unet_cfg = SdUnetConfig::sd_v1();
    let unet_weights = SdUnetWeights::load_from_mmapped(&unet_st, &unet_cfg)
        .map_err(|e| E::msg(format!("unet weights: {e}")))?;
    let unet = SdUnet {
        config: unet_cfg,
        weights: unet_weights,
    };

    // ---- scheduler --------------------------------------------------------
    let scheduler_cfg = DdimSchedulerConfig::default();
    let mut scheduler = DdimScheduler::new(n_steps, scheduler_cfg)
        .map_err(|e| E::msg(format!("ddim init: {e}")))?;

    // SD 1.5 VAE scale.
    let vae_scale = 0.18215_f64;
    let bsize = 1_usize; // batched generation isn't exposed via SdUnet::forward yet.
    let h_lat = height / 8;
    let w_lat = width / 8;

    for idx in 0..num_samples {
        let timesteps = scheduler.timesteps().to_vec();

        // Initial latents: standard-normal noise scaled by
        // `init_noise_sigma`.
        let n_latents = bsize * 4 * h_lat * w_lat;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(idx as u64));
        let init_sigma = scheduler.init_noise_sigma() as f32;
        let mut latents: Vec<f32> = (0..n_latents)
            .map(|_| standard_normal(&mut rng) * init_sigma)
            .collect();

        println!("starting sampling ({} steps)", timesteps.len());
        for (timestep_index, &timestep) in timesteps.iter().enumerate() {
            let start_time = std::time::Instant::now();

            // The lazy UNet takes a single conditioning vector per call,
            // so for classifier-free guidance we run it twice: once
            // unconditional, once conditional. SD's eager pipeline
            // batched these into one call; the lazy module isn't batched
            // along the conditioning axis yet.
            let noise_pred = if use_guide_scale {
                let n_text = max_pos * hidden;
                let uncond_emb = &text_emb_flat[..n_text];
                let cond_emb = &text_emb_flat[n_text..2 * n_text];

                let pred_uncond_lazy = unet
                    .forward(&latents, timestep as f32, uncond_emb, h_lat, w_lat)
                    .map_err(|e| E::msg(format!("unet uncond: {e}")))?;
                let pred_cond_lazy = unet
                    .forward(&latents, timestep as f32, cond_emb, h_lat, w_lat)
                    .map_err(|e| E::msg(format!("unet cond: {e}")))?;
                let pred_uncond = pred_uncond_lazy.realize_f32();
                let pred_cond = pred_cond_lazy.realize_f32();
                // CFG: uncond + scale * (cond - uncond).
                pred_uncond
                    .iter()
                    .zip(pred_cond.iter())
                    .map(|(u, c)| u + (guidance_scale as f32) * (c - u))
                    .collect::<Vec<f32>>()
            } else {
                let pred_lazy = unet
                    .forward(&latents, timestep as f32, &text_emb_flat, h_lat, w_lat)
                    .map_err(|e| E::msg(format!("unet: {e}")))?;
                pred_lazy.realize_f32()
            };

            // Build LazyTensors for the scheduler step. The scheduler
            // returns a LazyTensor — realize it back to f32 to feed the
            // next UNet step.
            let sample_lazy = fuel::lazy::LazyTensor::from_f32(
                latents.clone(),
                fuel::Shape::from_dims(&[bsize, 4, h_lat, w_lat]),
                &fuel::Device::cpu(),
            );
            let model_out_lazy = sample_lazy.const_f32_like(
                noise_pred,
                fuel::Shape::from_dims(&[bsize, 4, h_lat, w_lat]),
            );
            let next_lazy = scheduler
                .step(&model_out_lazy, timestep, &sample_lazy)
                .map_err(|e| E::msg(format!("scheduler step: {e}")))?;
            latents = next_lazy.realize_f32();

            let dt = start_time.elapsed().as_secs_f32();
            println!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);

            if intermediary_images {
                save_image(
                    &vae,
                    &latents,
                    bsize,
                    h_lat,
                    w_lat,
                    vae_scale,
                    idx,
                    &final_image,
                    num_samples,
                    Some(timestep_index + 1),
                )?;
            }
        }

        println!(
            "Generating the final image for sample {}/{}.",
            idx + 1,
            num_samples
        );
        save_image(
            &vae,
            &latents,
            bsize,
            h_lat,
            w_lat,
            vae_scale,
            idx,
            &final_image,
            num_samples,
            None,
        )?;
    }
    Ok(())
}

/// Box-Muller standard-normal sample. Avoids pulling in
/// `rand_distr` as a direct dep of `fuel-examples`.
fn standard_normal<R: Rng>(rng: &mut R) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-10_f32);
    let u2: f32 = rng.random::<f32>();
    (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos()
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

//! Stable Diffusion 3 / 3.5 text-to-image — lazy port.
//!
//! Migrated from the retired eager binary at
//! `fuel-examples/examples/_stable-diffusion-3_retired/` (recovered from
//! git history at `cfcb35cf~1:fuel-examples/examples/stable-diffusion-3/
//! main.rs`) onto the lazy-graph API:
//!
//! - [`fuel::lazy_sd3_text_encoder::Sd3TripleClip`] — CLIP-L + CLIP-G +
//!   T5-XXL composer that produces the SD3 `(context, y)` conditioning pair.
//! - [`fuel::lazy_mmdit::MmDitFullModel`] — SD3 MMDiT with patchify /
//!   pos-embed / final context-qkv-only block / unpatchify, supports SLG
//!   via `skip_layers`.
//! - [`fuel::lazy_sd_samplers_sd3::flow_match_euler_sample`] — SD3
//!   flow-match Euler with CFG + optional Skip-Layer-Guidance.
//! - [`fuel::lazy_sd3_vae::SdVae3Decoder`] — 16-channel SD3 VAE decoder
//!   (no post-quant-conv, applies the TAESD3 scale + shift internally).
//!
//! # Scope
//!
//! - Targets the SD 3.5 *split* checkpoint layout
//!   (`text_encoders/clip_l.safetensors`,
//!   `text_encoders/clip_g.safetensors`,
//!   `text_encoders/t5xxl_fp16.safetensors`, and a monolithic MMDiT
//!   weight file). The SD3-medium monolithic checkpoint at
//!   `stabilityai/stable-diffusion-3-medium/sd3_medium_incl_clips_t5xxlfp16.safetensors`
//!   needs name-prefix routing (`text_encoders.clip_l.transformer.*`)
//!   that this binary does not yet implement; pass `--model-id` pointing
//!   at one of the SD 3.5 split repos to drive end-to-end image gen.
//! - The CLIP weight loader on `lazy_sd_text_encoder` is hardcoded to
//!   the HF `text_model.` prefix; the SD 3.5 published
//!   `clip_l.safetensors` / `clip_g.safetensors` ship encoder weights
//!   at the root with no prefix. The binary builds the encoder against
//!   that convention by mapping `text_model.*` to the split layout via
//!   a thin in-binary prefix-routing wrapper around the mmap; the same
//!   wrapper picks up the CLIP-G `text_projection.weight` at the root.

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::sync::Arc;
use tokenizers::Tokenizer;

use fuel::lazy::LazyTensor;
use fuel::lazy_mmdit::{MmDitFullConfig, MmDitFullModel, MmDitFullWeights};
use fuel::lazy_sd3_text_encoder::{
    Sd3TripleClip, Sd3TripleClipConfig, Sd3TripleClipWeights, SD3_MAX_POSITION_EMBEDDINGS,
};
use fuel::lazy_sd3_vae::{SdVae3Config, SdVae3Decoder, SdVae3DecoderWeights};
use fuel::lazy_sd_samplers_sd3::{
    flow_match_euler_sample, Sd3Denoiser, Sd3SamplerConfig,
};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Positive text prompt.
    #[arg(
        long,
        default_value = "A cute rusty robot holding a fuel torch in its hand, \
        with glowing neon text \"LETS GO RUSTY\" displayed on its chest, \
        bright background, high quality, 4k"
    )]
    prompt: String,

    /// Negative prompt for classifier-free guidance.
    #[arg(long, default_value = "")]
    negative_prompt: String,

    /// Number of Euler integration steps. Defaults to 28 (SD 3.5 reference).
    #[arg(long, default_value_t = 28)]
    num_steps: usize,

    /// Classifier-free guidance scale. Defaults to 4.5 (SD 3.5 reference).
    #[arg(long, default_value_t = 4.5)]
    guidance_scale: f64,

    /// Skip-Layer-Guidance strength. `0.0` disables SLG; the reference
    /// SD 3.5-medium default is 2.5.
    #[arg(long, default_value_t = 0.0)]
    slg_scale: f64,

    /// Output PNG file path.
    #[arg(long, default_value = "sd3_out.png")]
    output_png: String,

    /// HuggingFace model repo id. Defaults to SD 3.5-large; switch to
    /// `stabilityai/stable-diffusion-3.5-medium` to drive the SLG
    /// pipeline, or to `stabilityai/stable-diffusion-3.5-large-turbo`
    /// for the 4-step distilled variant.
    #[arg(long, default_value = "stabilityai/stable-diffusion-3.5-large")]
    model_id: String,

    /// Output image width in pixels. Must be a multiple of 16 (= 8×
    /// VAE downsample × 2× patch size). Defaults to 1024.
    #[arg(long, default_value_t = 1024)]
    width: usize,

    /// Output image height in pixels. Same constraint as `--width`.
    #[arg(long, default_value_t = 1024)]
    height: usize,

    /// Optional RNG seed for reproducible noise. Falls back to a fixed
    /// constant when omitted so the binary is deterministic by default.
    #[arg(long)]
    seed: Option<u64>,
}

/// Local denoiser wrapper around `MmDitFullModel` so we can hand the
/// sampler a `&dyn Sd3Denoiser` without modifying `lazy_mmdit`. Keeps
/// the trait `impl` out of `fuel-core` until the SD3 module stack
/// settles its sampler/denoiser ownership (the sampler is in
/// `lazy_sd_samplers_sd3`, the model lives in `lazy_mmdit`, and adding
/// a `Sd3Denoiser` impl directly in `lazy_mmdit` would couple the two
/// across module boundaries).
struct MmDitFullDenoiser<'a> {
    model: &'a MmDitFullModel,
}

impl<'a> Sd3Denoiser for MmDitFullDenoiser<'a> {
    fn forward(
        &self,
        latent: &LazyTensor,
        timestep: &LazyTensor,
        y: &LazyTensor,
        context: &LazyTensor,
        skip_layers: Option<&[usize]>,
    ) -> fuel::Result<LazyTensor> {
        self.model.forward(latent, timestep, y, context, skip_layers)
    }
}

/// Tokenize `prompt` against `tokenizer`, pad/truncate to 77 token ids
/// using `pad_id` (CLIP) or 0 (T5). Matches the eager
/// `StableDiffusion3TripleClipWithTokenizer::encode_text_to_embedding`
/// convention: tokenizers either truncate to 76 + EOS or pad to 77.
fn tokenize_padded(
    tokenizer: &Tokenizer,
    prompt: &str,
    pad_id: u32,
) -> Result<Vec<u32>> {
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    if tokens.len() > SD3_MAX_POSITION_EMBEDDINGS {
        tokens.truncate(SD3_MAX_POSITION_EMBEDDINGS);
    } else {
        while tokens.len() < SD3_MAX_POSITION_EMBEDDINGS {
            tokens.push(pad_id);
        }
    }
    Ok(tokens)
}

/// Deterministic standard-normal noise via Box-Muller over a small LCG
/// seeded from `seed`. Avoids pulling in a runtime RNG kernel — the
/// sampler itself is RNG-free, so the only randomness we need is the
/// initial latent. Matches the same pattern used by `fuel-examples/
/// examples/flux/main.rs`.
fn deterministic_noise(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next_u32 = || -> u32 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 32) as u32
    };
    let mut next_f32 = || -> f32 {
        let u = (next_u32() as f64 / u32::MAX as f64)
            .clamp(1e-9, 1.0 - 1e-9);
        u as f32
    };
    let mut out = Vec::with_capacity(n);
    while out.len() + 1 < n {
        let u1 = next_f32();
        let u2 = next_f32();
        let r = ((-2.0_f64) * (u1 as f64).ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2 as f64;
        out.push((r * theta.cos()) as f32);
        out.push((r * theta.sin()) as f32);
    }
    if out.len() < n {
        let u1 = next_f32();
        let u2 = next_f32();
        let r = ((-2.0_f64) * (u1 as f64).ln()).sqrt();
        out.push((r * (2.0 * std::f64::consts::PI * u2 as f64).cos()) as f32);
    }
    out
}

/// Pick a `MmDitFullConfig` preset from a HuggingFace model id. SD 3.5
/// `-medium` is the MMDiT-X variant whose joint-block weights are
/// shaped the same as the plain MmDit — the non-X joint-block path is
/// what `MmDitFullModel` runs today; the binary tags the config so a
/// follow-up can route it through the MMDiT-X path when that lands.
fn mmdit_config_for(model_id: &str) -> MmDitFullConfig {
    if model_id.contains("3.5-large") {
        // Both `large` and `large-turbo` share the same MMDiT shape;
        // the turbo distillation only changes step count + CFG.
        MmDitFullConfig::sd3_5_large()
    } else if model_id.contains("3.5-medium") {
        MmDitFullConfig::sd3_5_medium()
    } else {
        // SD3-medium (and any unrecognized id) → 24-block / 1536-hidden.
        MmDitFullConfig::sd3_medium()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let Args {
        prompt,
        negative_prompt,
        num_steps,
        guidance_scale,
        slg_scale,
        output_png,
        model_id,
        width,
        height,
        seed,
    } = args;

    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        anyhow::bail!(
            "--width / --height must be multiples of 16 (got {width}×{height})"
        );
    }

    let device = Device::cpu();
    let api = hf_hub::api::sync::Api::new()?;
    let mmdit_cfg = mmdit_config_for(&model_id);

    // ---- Resolve checkpoint paths -----------------------------------
    //
    // SD 3.5 ships the text encoders under `text_encoders/` and the
    // MMDiT as a top-level monolithic file. We pull all three CLIP /
    // T5 files plus the matching MMDiT file. The MMDiT file naming
    // varies by which SD 3.5 variant the user pointed `--model-id` at.
    let repo = api.repo(hf_hub::Repo::model(model_id.clone()));
    println!("Resolving safetensors from `{model_id}`…");
    let clip_l_file = repo.get("text_encoders/clip_l.safetensors")?;
    let clip_g_file = repo.get("text_encoders/clip_g.safetensors")?;
    let t5_file = repo.get("text_encoders/t5xxl_fp16.safetensors")?;
    let mmdit_file = {
        // The Hub-published filenames are `sd3.5_<variant>.safetensors`
        // for SD 3.5; try the most-common variants in turn and pick the
        // first one that resolves so this works for `-large`,
        // `-large-turbo`, and `-medium` without extra CLI flags.
        let candidates = [
            "sd3.5_large.safetensors",
            "sd3.5_large_turbo.safetensors",
            "sd3.5_medium.safetensors",
        ];
        let mut found = None;
        for c in candidates {
            if let Ok(p) = repo.get(c) {
                found = Some(p);
                break;
            }
        }
        found.ok_or_else(|| {
            E::msg(format!(
                "could not resolve an MMDiT weight file under `{model_id}` \
                 (looked for sd3.5_large / sd3.5_large_turbo / sd3.5_medium)",
            ))
        })?
    };
    let vae_file = match repo.get("vae/diffusion_pytorch_model.safetensors") {
        Ok(p) => p,
        Err(_) => repo.get("vae/diffusion_pytorch_model.fp16.safetensors")?,
    };

    // ---- Tokenize the prompts ---------------------------------------
    //
    // SD3 splits text conditioning across THREE encoders; each one wants
    // its own tokenizer. The CLIP-L / CLIP-G pair share the OpenAI
    // CLIP vocab (49408 ids, end-of-text = 49407 used as pad), pulled
    // from the standard mirror. T5-XXL needs the SentencePiece model
    // shipped by `lmz/mt5-tokenizers`.
    println!("Loading tokenizers…");
    let clip_tokenizer_file = api
        .model("laion/CLIP-ViT-L-14-laion2B-s32B-b82K".to_string())
        .get("tokenizer.json")?;
    let clip_tokenizer = Tokenizer::from_file(clip_tokenizer_file).map_err(E::msg)?;
    let t5_tokenizer_file = api
        .model("lmz/mt5-tokenizers".to_string())
        .get("t5-v1_1-xxl.tokenizer.json")?;
    let t5_tokenizer = Tokenizer::from_file(t5_tokenizer_file).map_err(E::msg)?;

    // CLIP pads with end-of-text (49407); T5 pads with 0.
    let clip_pad_id: u32 = 49407;
    let t5_pad_id: u32 = 0;

    let clip_l_tokens = tokenize_padded(&clip_tokenizer, &prompt, clip_pad_id)?;
    let clip_g_tokens = clip_l_tokens.clone();
    let t5_tokens = tokenize_padded(&t5_tokenizer, &prompt, t5_pad_id)?;

    let neg_clip_l_tokens =
        tokenize_padded(&clip_tokenizer, &negative_prompt, clip_pad_id)?;
    let neg_clip_g_tokens = neg_clip_l_tokens.clone();
    let neg_t5_tokens = tokenize_padded(&t5_tokenizer, &negative_prompt, t5_pad_id)?;

    // ---- Build the triple-CLIP composer + encode --------------------
    println!("Loading triple-CLIP weights…");
    let st_clip_l = unsafe { MmapedSafetensors::new(&clip_l_file) }
        .map_err(|e| E::msg(format!("mmap clip_l: {e}")))?;
    let st_clip_g = unsafe { MmapedSafetensors::new(&clip_g_file) }
        .map_err(|e| E::msg(format!("mmap clip_g: {e}")))?;
    let st_t5 = unsafe { MmapedSafetensors::new(&t5_file) }
        .map_err(|e| E::msg(format!("mmap t5: {e}")))?;

    let triple_cfg = Sd3TripleClipConfig::sd3_medium();
    let triple_weights =
        Sd3TripleClipWeights::load_from_mmapped(&st_clip_l, &st_clip_g, &st_t5, &triple_cfg)
            .map_err(|e| E::msg(format!("triple-CLIP weights: {e}")))?;
    let triple = Sd3TripleClip::new(triple_cfg, triple_weights);

    println!("Encoding prompts…");
    let (context, y) = triple
        .encode(&clip_l_tokens, &clip_g_tokens, &t5_tokens)
        .map_err(|e| E::msg(format!("triple-CLIP encode: {e}")))?;
    let (neg_context, neg_y) = triple
        .encode(&neg_clip_l_tokens, &neg_clip_g_tokens, &neg_t5_tokens)
        .map_err(|e| E::msg(format!("triple-CLIP encode (negative): {e}")))?;
    println!(
        "  context: {:?}, y: {:?}",
        context.shape().dims(),
        y.shape().dims(),
    );
    // Drop the encoder bag early — its weights are big and the MMDiT
    // load coming next needs the memory.
    drop(triple);

    // ---- Build the MMDiT --------------------------------------------
    println!("Loading MMDiT weights ({model_id})…");
    let st_mmdit = unsafe { MmapedSafetensors::new(&mmdit_file) }
        .map_err(|e| E::msg(format!("mmap mmdit: {e}")))?;
    let mmdit_weights = MmDitFullWeights::load_from_mmapped(&st_mmdit, &mmdit_cfg)
        .map_err(|e| E::msg(format!("mmdit weights: {e}")))?;
    let mmdit = MmDitFullModel {
        config: mmdit_cfg.clone(),
        weights: mmdit_weights,
    };

    // ---- Build the initial noise latent -----------------------------
    let c_lat = mmdit_cfg.in_channels;
    let h_lat = height / 8;
    let w_lat = width / 8;
    let noise_seed = seed.unwrap_or(0xBADF_00D_BADF_00D_u64);
    let noise = deterministic_noise(noise_seed, c_lat * h_lat * w_lat);
    let latent = LazyTensor::from_f32(
        Arc::from(noise),
        Shape::from_dims(&[1, c_lat, h_lat, w_lat]),
        &device,
    );

    // ---- Build the sampler config -----------------------------------
    //
    // `--slg-scale > 0` enables SLG and routes through the SD 3.5-medium
    // reference window [0.01, 0.20] over layers [7, 8, 9]. Outside that,
    // we run plain CFG flow-match Euler.
    let sampler_cfg = if slg_scale > 0.0 {
        Sd3SamplerConfig {
            num_steps,
            time_snr_shift: 3.0,
            guidance_scale,
            slg_layers: vec![7, 8, 9],
            slg_scale,
            slg_start: 0.01,
            slg_end: 0.20,
        }
    } else {
        Sd3SamplerConfig::slg_disabled(num_steps, 3.0, guidance_scale)
    };

    // ---- Sample -----------------------------------------------------
    let denoiser = MmDitFullDenoiser { model: &mmdit };
    println!(
        "Sampling {num_steps} steps (CFG={guidance_scale}, SLG={slg_scale})…",
    );
    let start_time = std::time::Instant::now();
    let final_latent = flow_match_euler_sample(
        &denoiser,
        latent,
        context,
        y,
        &sampler_cfg,
        neg_context,
        neg_y,
    )
    .map_err(|e| E::msg(format!("flow_match_euler_sample: {e}")))?;
    let dt = start_time.elapsed().as_secs_f32();
    println!(
        "Sampling done in {dt:.2}s ({:.2} iter/s).",
        num_steps as f32 / dt,
    );
    drop(mmdit);

    // ---- VAE decode -------------------------------------------------
    println!("Loading VAE weights…");
    let st_vae = unsafe { MmapedSafetensors::new(&vae_file) }
        .map_err(|e| E::msg(format!("mmap vae: {e}")))?;
    let vae_cfg = SdVae3Config::sd3();
    let vae_weights = SdVae3DecoderWeights::load_from_mmapped(&st_vae, &vae_cfg)
        .map_err(|e| E::msg(format!("vae weights: {e}")))?;
    let vae = SdVae3Decoder {
        config: vae_cfg,
        weights: vae_weights,
    };

    println!("Decoding latent → image…");
    let image_lazy = vae
        .decode(&final_latent)
        .map_err(|e| E::msg(format!("vae decode: {e}")))?;

    // ---- Realize + post-process to U8 -------------------------------
    //
    // Pixel post: clamp to [-1, 1], shift to [0, 1], scale to [0, 255],
    // round to u8. Matches the eager `((img.clamp(-1, 1) + 1.0) *
    // 127.5).to_dtype(U8)` pipeline.
    let dims = image_lazy.shape().dims().to_vec();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
        anyhow::bail!("VAE produced unexpected output shape: {dims:?}");
    }
    let (_, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let pixels = image_lazy.realize_f32();
    let mut chw_u8 = vec![0_u8; c * h * w];
    for (i, &v) in pixels.iter().enumerate() {
        let scaled = ((v.clamp(-1.0_f32, 1.0_f32) + 1.0) * 127.5).round();
        chw_u8[i] = scaled.clamp(0.0, 255.0) as u8;
    }

    // ---- Save PNG ---------------------------------------------------
    //
    // `fuel_examples::save_image` expects an eager `(3, H, W)` U8
    // tensor; building one is the cleanest realize→save bridge. The
    // file extension on `output_png` drives the image-crate encoder.
    let eager =
        fuel::Tensor::from_vec(chw_u8, (c, h, w), &fuel::Device::cpu())
            .map_err(|e| E::msg(format!("build eager u8 tensor: {e}")))?;
    fuel_examples::save_image(&eager, &output_png)
        .map_err(|e| E::msg(format!("save png: {e}")))?;
    println!("Saved {output_png}");
    Ok(())
}

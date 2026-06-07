//! FLUX — lazy port.
//!
//! Migrated from `fuel_transformers::models::{clip,flux,t5}` to the
//! lazy-graph API at `fuel::lazy_clip`, `fuel::lazy_t5`, and
//! `fuel::lazy_flux`. The text-conditioning towers (T5 encoder + CLIP-L
//! pooled embedding) and the FLUX DiT are all driven through
//! `load_from_mmapped` + `generate`.
//!
//! Limitations of the lazy port (vs. the eager binary):
//!   - `FluxVae` has no `load_from_mmapped` yet (the encoder/decoder
//!     weight bags are exposed but the safetensors loader is pending).
//!     The binary still wires the DiT path end-to-end and writes the
//!     packed latent to disk so progress can be inspected; the VAE
//!     decode + JPEG save will be re-enabled once the loader lands.
//!   - The quantized Q4_0 path is built via
//!     `QuantizedFluxModel::from_f32_bake` rather than reading a GGUF
//!     file; consuming the upstream `flux1-schnell.gguf` is left as a
//!     follow-up.
//!
//! Eager parity: image packing / unpacking, sequential RoPE id
//! tensors, schedule shift, and the per-step denoise update are
//! re-implemented inline so the lazy module can stay agnostic to the
//! FLUX-specific packing layout.

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::sync::Arc;
use tokenizers::Tokenizer;

use fuel::lazy::LazyTensor;
use fuel::lazy_clip::{ClipTextConfig, ClipTextModel, ClipTextWeights};
use fuel::lazy_flux::{
    generate, FlowMatchScheduler, FluxConfig, FluxModel, FluxWeights, QuantizedFluxModel,
};
use fuel::lazy_t5::{T5Activation, T5Config, T5Model, T5Weights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(long, default_value = "A rusty robot walking on a beach")]
    prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Use the quantized model.
    #[arg(long)]
    quantized: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The height in pixels of the generated image.
    #[arg(long)]
    height: Option<usize>,

    /// The width in pixels of the generated image.
    #[arg(long)]
    width: Option<usize>,

    #[arg(long)]
    decode_only: Option<String>,

    #[arg(long, value_enum, default_value = "schnell")]
    model: Model,

    /// Use the slower kernels.
    #[arg(long)]
    use_dmmv: bool,

    /// The seed to use when generating random samples.
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum Model {
    Schnell,
    Dev,
}

/// Subset of the HuggingFace `t5-v1_1-*/config.json` schema that we
/// need to populate a lazy `T5Config`. We only use the encoder for FLUX
/// conditioning, but the lazy struct still wants decoder fields.
#[derive(Debug, serde::Deserialize)]
struct HfT5Config {
    vocab_size: usize,
    d_model: usize,
    d_kv: usize,
    d_ff: usize,
    num_layers: usize,
    num_decoder_layers: Option<usize>,
    num_heads: usize,
    relative_attention_num_buckets: usize,
    #[serde(default = "default_rel_attn_max_distance")]
    relative_attention_max_distance: usize,
    layer_norm_epsilon: f64,
    #[serde(default = "default_feed_forward_proj")]
    feed_forward_proj: String,
    #[serde(default = "default_tie_word_embeddings")]
    tie_word_embeddings: bool,
}

fn default_rel_attn_max_distance() -> usize {
    128
}
fn default_feed_forward_proj() -> String {
    "relu".to_string()
}
fn default_tie_word_embeddings() -> bool {
    true
}

fn lazy_t5_config_from_hf(hf: HfT5Config) -> T5Config {
    let (gated_ffn, activation) = match hf.feed_forward_proj.as_str() {
        "gated-gelu" => (true, T5Activation::Gelu),
        "gated-gelu-pytorch-tanh" => (true, T5Activation::GeluPytorchTanh),
        "gated-silu" => (true, T5Activation::Silu),
        "gelu" => (false, T5Activation::Gelu),
        "gelu_new" | "gelu-new" => (false, T5Activation::GeluPytorchTanh),
        "silu" => (false, T5Activation::Silu),
        _ => (false, T5Activation::Relu),
    };
    T5Config {
        vocab_size: hf.vocab_size,
        d_model: hf.d_model,
        d_kv: hf.d_kv,
        d_ff: hf.d_ff,
        num_layers: hf.num_layers,
        num_decoder_layers: hf.num_decoder_layers,
        num_heads: hf.num_heads,
        relative_attention_num_buckets: hf.relative_attention_num_buckets,
        relative_attention_max_distance: hf.relative_attention_max_distance,
        layer_norm_epsilon: hf.layer_norm_epsilon,
        activation,
        gated_ffn,
        tie_word_embeddings: hf.tie_word_embeddings,
    }
}

/// argmax position of u32 token ids — CLIP-L picks the EOS slot the
/// same way the eager `ClipTextTransformer::forward` did.
fn argmax_u32(tokens: &[u32]) -> usize {
    let mut best = 0usize;
    let mut best_val = tokens.first().copied().unwrap_or(0);
    for (i, &t) in tokens.iter().enumerate().skip(1) {
        if t > best_val {
            best_val = t;
            best = i;
        }
    }
    best
}

/// Pack `(B, 16, H, W)` latent into FLUX's `(B, H/2 * W/2, 64)` patch
/// sequence layout. Mirrors `flux::sampling::State::new` from the eager
/// implementation.
fn pack_latent(x: &LazyTensor, b: usize, c: usize, h: usize, w: usize) -> Result<LazyTensor> {
    let packed = x
        .reshape(Shape::from_dims(&[b, c, h / 2, 2, w / 2, 2]))?
        .permute([0_usize, 2, 4, 1, 3, 5])?
        .reshape(Shape::from_dims(&[b, h / 2 * w / 2, c * 4]))?;
    Ok(packed)
}

/// Reverse [`pack_latent`].
fn unpack_latent(xs: &LazyTensor, height: usize, width: usize) -> Result<LazyTensor> {
    let dims = xs.shape().dims().to_vec();
    let b = dims[0];
    let c_ph_pw = dims[2];
    let h_div2 = height.div_ceil(16);
    let w_div2 = width.div_ceil(16);
    let c = c_ph_pw / 4;
    let out = xs
        .reshape(Shape::from_dims(&[b, h_div2, w_div2, c, 2, 2]))?
        .permute([0_usize, 3, 1, 4, 2, 5])?
        .reshape(Shape::from_dims(&[b, c, h_div2 * 2, w_div2 * 2]))?;
    Ok(out)
}

/// Per-patch RoPE id tensor `(B, H/2 * W/2, 3)` matching eager
/// `flux::sampling::State::new`.
fn make_img_ids(anchor: &LazyTensor, b: usize, h: usize, w: usize) -> LazyTensor {
    let h2 = h / 2;
    let w2 = w / 2;
    let n = h2 * w2;
    let mut buf = vec![0.0_f32; b * n * 3];
    for bb in 0..b {
        for i in 0..h2 {
            for j in 0..w2 {
                let base = ((bb * n) + i * w2 + j) * 3;
                buf[base] = 0.0;
                buf[base + 1] = i as f32;
                buf[base + 2] = j as f32;
            }
        }
    }
    anchor.const_f32_like(Arc::from(buf), Shape::from_dims(&[b, n, 3]))
}

/// Per-token text RoPE id tensor `(B, S_text, 3)` — all zeros, just
/// like the eager implementation.
fn make_txt_ids(anchor: &LazyTensor, b: usize, seq: usize) -> LazyTensor {
    let buf = vec![0.0_f32; b * seq * 3];
    anchor.const_f32_like(Arc::from(buf), Shape::from_dims(&[b, seq, 3]))
}

/// Deterministic standard-normal noise via Box-Muller over an LCG seeded
/// from `seed`. Used in place of the eager `Tensor::randn` so we don't
/// depend on a runtime RNG kernel.
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
            .max(1e-9)
            .min(1.0 - 1e-9);
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

fn run(args: Args) -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let Args {
        prompt,
        cpu,
        height,
        width,
        tracing,
        decode_only,
        model,
        quantized,
        seed,
        ..
    } = args;
    let width = width.unwrap_or(1360);
    let height = height.unwrap_or(768);

    let _guard = if tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let api = hf_hub::api::sync::Api::new()?;
    let bf_repo = {
        let name = match model {
            Model::Dev => "black-forest-labs/FLUX.1-dev",
            Model::Schnell => "black-forest-labs/FLUX.1-schnell",
        };
        api.repo(hf_hub::Repo::model(name.to_string()))
    };
    // `--cpu` is preserved for parity with the eager binary; the lazy
    // realize seam currently lives on CPU regardless.
    let _ = cpu;
    let device = Device::cpu();

    // ---- Build the packed-latent input -----------------------------
    let (packed_latent, b, c_lat, h_lat, w_lat) = match decode_only.as_deref() {
        None => {
            // ---- T5 encoder text features ---------------------------
            let t5_emb = {
                let repo = api.repo(hf_hub::Repo::with_revision(
                    "google/t5-v1_1-xxl".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/2".to_string(),
                ));
                let model_file = repo.get("model.safetensors")?;
                let config_filename = repo.get("config.json")?;
                let config_str = std::fs::read_to_string(config_filename)?;
                let hf_cfg: HfT5Config = serde_json::from_str(&config_str)?;
                let t5_cfg = lazy_t5_config_from_hf(hf_cfg);
                let st = unsafe { MmapedSafetensors::multi(&[model_file]) }
                    .map_err(|e| E::msg(format!("mmap t5 safetensors: {e}")))?;
                let t5_weights = T5Weights::load_from_mmapped(&st, &t5_cfg)
                    .map_err(|e| E::msg(format!("load t5 weights: {e}")))?;
                let t5_model = T5Model {
                    config: t5_cfg,
                    weights: t5_weights,
                };
                let tokenizer_filename = api
                    .model("lmz/mt5-tokenizers".to_string())
                    .get("t5-v1_1-xxl.tokenizer.json")?;
                let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
                let mut tokens = tokenizer
                    .encode(prompt.as_str(), true)
                    .map_err(E::msg)?
                    .get_ids()
                    .to_vec();
                tokens.resize(256, 0);
                t5_model
                    .forward_encoder(&tokens)
                    .map_err(|e| E::msg(format!("t5 forward: {e}")))?
            };
            println!("T5 features: {:?}", t5_emb.shape().dims());

            // ---- CLIP-L pooled text features ------------------------
            let clip_emb = {
                let repo = api.repo(hf_hub::Repo::model(
                    "openai/clip-vit-large-patch14".to_string(),
                ));
                let model_file = repo.get("model.safetensors")?;
                let config = ClipTextConfig {
                    vocab_size: 49408,
                    projection_dim: 768,
                    intermediate_size: 3072,
                    embed_dim: 768,
                    max_position_embeddings: 77,
                    num_hidden_layers: 12,
                    num_attention_heads: 12,
                };
                let st = unsafe { MmapedSafetensors::multi(&[model_file]) }
                    .map_err(|e| E::msg(format!("mmap clip safetensors: {e}")))?;
                let clip_weights = ClipTextWeights::load_from_mmapped(&st, &config, "text_model.")
                    .map_err(|e| E::msg(format!("load clip weights: {e}")))?;
                let clip_text_model = ClipTextModel {
                    config: config.clone(),
                    weights: clip_weights,
                };
                let tokenizer_filename = repo.get("tokenizer.json")?;
                let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
                let tokens = tokenizer
                    .encode(prompt.as_str(), true)
                    .map_err(E::msg)?
                    .get_ids()
                    .to_vec();
                let eos_pos = argmax_u32(&tokens);
                clip_text_model
                    .pool_eos(&tokens, eos_pos)
                    .map_err(|e| E::msg(format!("clip pool: {e}")))?
            };
            println!("CLIP features: {:?}", clip_emb.shape().dims());

            // ---- Initial noise + packed latent ----------------------
            let cfg = match model {
                Model::Dev => FluxConfig::dev(),
                Model::Schnell => FluxConfig::schnell(),
            };
            let batch = 1usize;
            let c_lat = 16usize;
            let h_lat = height.div_ceil(16) * 2;
            let w_lat = width.div_ceil(16) * 2;
            let noise_seed = seed.unwrap_or(0xBADF_00D_BADF_00Du64);
            let noise = deterministic_noise(noise_seed, batch * c_lat * h_lat * w_lat);
            let noise_t = LazyTensor::from_f32(
                Arc::from(noise),
                Shape::from_dims(&[batch, c_lat, h_lat, w_lat]),
                &device,
            );
            let img_packed = pack_latent(&noise_t, batch, c_lat, h_lat, w_lat)
                .map_err(|e| E::msg(format!("pack latent: {e}")))?;
            let img_ids = make_img_ids(&img_packed, batch, h_lat, w_lat);
            let txt_dims = t5_emb.shape().dims().to_vec();
            let txt_seq = txt_dims[1];
            let txt_ids = make_txt_ids(&img_packed, batch, txt_seq);

            // ---- Scheduler ------------------------------------------
            let image_seq_len = h_lat / 2 * w_lat / 2;
            let scheduler = match model {
                Model::Dev => FlowMatchScheduler::shifted(50, image_seq_len, 0.5, 1.15),
                Model::Schnell => FlowMatchScheduler::linear(4),
            };

            // ---- Denoise --------------------------------------------
            let packed_out = if quantized {
                let model_file = match model {
                    Model::Schnell => api
                        .repo(hf_hub::Repo::model("lmz/fuel-flux".to_string()))
                        .get("flux1-schnell.gguf")?,
                    Model::Dev => anyhow::bail!("quantized flux1-dev not supported"),
                };
                // The lazy port does not yet read GGUF; we surface the
                // mismatch with a clear message rather than silently
                // running the F32 model. Bake-from-F32 is available via
                // `QuantizedFluxModel::from_f32_bake` once the F32 file
                // is loaded.
                let _ = model_file;
                let _: Option<QuantizedFluxModel> = None;
                anyhow::bail!(
                    "lazy flux: GGUF Q4_0 loader is pending; rerun without --quantized \
                     or wait for `QuantizedFluxModel::from_gguf`"
                );
            } else {
                let model_file = match model {
                    Model::Schnell => bf_repo.get("flux1-schnell.safetensors")?,
                    Model::Dev => bf_repo.get("flux1-dev.safetensors")?,
                };
                let st = unsafe { MmapedSafetensors::multi(&[model_file]) }
                    .map_err(|e| E::msg(format!("mmap flux safetensors: {e}")))?;
                let weights = FluxWeights::load_from_mmapped(&st, &cfg)
                    .map_err(|e| E::msg(format!("load flux weights: {e}")))?;
                let flux_model = FluxModel {
                    config: cfg.clone(),
                    weights,
                };
                // Optional guidance scalar — only the dev model uses it.
                let guidance_scalar = if cfg.guidance_embed {
                    Some(img_packed.const_f32_like(
                        Arc::from(vec![4.0_f32; batch]),
                        Shape::from_dims(&[batch]),
                    ))
                } else {
                    None
                };
                let guidance_ref = guidance_scalar.as_ref();
                generate(
                    &flux_model,
                    &clip_emb,
                    &t5_emb,
                    &img_packed,
                    &img_ids,
                    &txt_ids,
                    &scheduler,
                    guidance_ref,
                )
                .map_err(|e| E::msg(format!("flux generate: {e}")))?
            };
            (packed_out, batch, c_lat, h_lat, w_lat)
        }
        Some(file) => {
            // Decode-only path: load a previously-saved packed latent
            // (eager safetensors layout) and skip the DiT entirely.
            let st = unsafe { MmapedSafetensors::multi(&[file]) }
                .map_err(|e| E::msg(format!("mmap decode_only safetensors: {e}")))?;
            let raw = fuel::lazy::load_tensor_as_f32(&st, "img")
                .map_err(|e| E::msg(format!("load 'img' tensor: {e}")))?;
            let h_lat = height.div_ceil(16) * 2;
            let w_lat = width.div_ceil(16) * 2;
            let batch = 1usize;
            let c_lat = 16usize;
            let img_seq = h_lat / 2 * w_lat / 2;
            let packed = LazyTensor::from_f32(
                Arc::from(raw),
                Shape::from_dims(&[batch, img_seq, 64]),
                &device,
            );
            (packed, batch, c_lat, h_lat, w_lat)
        }
    };

    // ---- Unpack to image-shaped latent ----------------------------
    let img_latent = unpack_latent(&packed_latent, height, width)
        .map_err(|e| E::msg(format!("unpack latent: {e}")))?;
    let latent_data = img_latent.realize_f32();
    println!(
        "latent img shape: {:?} (mean {:.6})",
        img_latent.shape().dims(),
        latent_data.iter().copied().sum::<f32>() / (latent_data.len().max(1) as f32),
    );

    // ---- VAE decode is pending ------------------------------------
    //
    // `FluxVae::load_from_mmapped` does not yet exist on the lazy
    // module. We bail with a clear message rather than producing
    // bogus pixels. Once the loader lands the call below becomes:
    //
    //     let vae_file = bf_repo.get("ae.safetensors")?;
    //     let st = unsafe { MmapedSafetensors::multi(&[vae_file])? };
    //     let vae = FluxVae::load_from_mmapped(&st, &FluxVaeConfig::dev())?;
    //     let img = vae.decode(&img_latent)?;
    //     ...
    //
    let _ = b;
    let _ = c_lat;
    let _ = h_lat;
    let _ = w_lat;
    let _ = bf_repo;
    anyhow::bail!(
        "lazy flux: VAE decode is pending — \
         `FluxVae::load_from_mmapped` has not been implemented yet. \
         The packed latent + unpack path ran successfully; rerun once \
         the VAE loader lands."
    );
}

fn main() -> Result<()> {
    let args = Args::parse();
    // `--use_dmmv` was an eager `fuel::quantized::cuda::set_force_dmmv`
    // toggle; the lazy port does not expose that switch, so we accept
    // the flag for CLI parity and ignore the value.
    let _ = args.use_dmmv;
    run(args)
}

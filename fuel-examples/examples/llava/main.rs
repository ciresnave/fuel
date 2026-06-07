//! LLaVA (Large Language-and-Vision Assistant) — lazy-graph port.
//!
//! v1 of this binary is intentionally minimal:
//!
//! * Single image, single resolution. The eager port did anyres
//!   tile splitting on top of [`select_best_resolution`]; here we
//!   just resize the image to `vision_config.image_size` and feed
//!   it as one tile. The `select_best_resolution` helper is still
//!   wired in so you can pass a multi-resolution grid (e.g. from
//!   `image_grid_pinpoints`) to pick the closest match before the
//!   resize, but the resize itself collapses to a single tile.
//! * "linear" projector only — matching what
//!   `fuel::lazy_llava::LlavaModel::forward` supports.
//! * Per-patch CLIP features ("patch" select_feature) only.
//! * Greedy argmax decode; no KV cache (every step re-runs the
//!   whole graph), mirroring the BLIP / PaliGemma lazy binaries.
//! * No `<image>` token splice — image features are prepended to
//!   the text embeddings, identical to what `LlavaModel::forward`
//!   already does internally. The HF prompt convention of inserting
//!   `<image>` somewhere in the text and replacing it with the
//!   image features in-place is left to a v2 of this binary.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::io::Write;
use std::sync::Arc;

use fuel::lazy::LazyTensor;
use fuel::lazy_llava::{
    select_best_resolution, HFLlavaConfig, LlavaModel, LlavaWeights,
};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};

use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to a local image file.
    #[arg(long)]
    image: String,

    /// User prompt. For v1 we do NOT replace a `<image>` token —
    /// the image features are always prepended.
    #[arg(long)]
    prompt: String,

    /// Hugging Face model id. Defaults to the canonical
    /// llava-1.5-7b-hf checkpoint.
    #[arg(long)]
    model_id: Option<String>,

    /// HF revision (branch / commit / tag).
    #[arg(long, default_value = "main")]
    revision: String,

    /// Override `tokenizer.json` path.
    #[arg(long)]
    tokenizer_file: Option<String>,

    /// Override the safetensors weight files (comma-separated).
    #[arg(long)]
    weight_files: Option<String>,

    /// Override the `config.json` path. By default the binary
    /// pulls it from the hub and parses it via
    /// [`HFLlavaConfig::from_hf_json_str`].
    #[arg(long)]
    config_file: Option<String>,

    /// Run on CPU rather than GPU. Lazy realize routes through the
    /// default backend chooser; this flag is preserved for parity.
    #[arg(long)]
    cpu: bool,

    /// Greedy decode length cap (tokens after the prompt).
    #[arg(long, default_value_t = 256)]
    sample_len: usize,

    /// Optional supported-resolution pinpoints, formatted as
    /// `w1xh1,w2xh2,...`. When provided, the helper picks the best
    /// match before resizing to `image_size x image_size`; this is
    /// purely informational for v1 (the lazy forward still works at
    /// a single tile).
    #[arg(long)]
    grid_pinpoints: Option<String>,
}

/// Parse a `w1xh1,w2xh2,...` pinpoints string into a `Vec<(u32, u32)>`.
fn parse_pinpoints(s: &str) -> Result<Vec<(u32, u32)>> {
    s.split(',')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (w, h) = p
                .split_once('x')
                .ok_or_else(|| E::msg(format!("pinpoint {p:?} not in WxH form")))?;
            let w: u32 = w.trim().parse().map_err(E::msg)?;
            let h: u32 = h.trim().parse().map_err(E::msg)?;
            Ok((w, h))
        })
        .collect()
}

/// Load `path` and produce a CHW f32 vector of length
/// `3 * image_size * image_size` using the OpenAI CLIP
/// normalization the LLaVA / LLaVA-NeXT preprocessor uses.
fn load_image_chw<P: AsRef<std::path::Path>>(
    path: P,
    image_size: usize,
) -> Result<Vec<f32>> {
    let img = image::ImageReader::open(path)?.decode()?;
    let img = img.resize_to_fill(
        image_size as u32,
        image_size as u32,
        image::imageops::FilterType::Triangle,
    );
    let raw = img.to_rgb8().into_raw();
    let mean = [0.481_454_66_f32, 0.457_827_5, 0.408_210_73];
    let std = [0.268_629_54_f32, 0.261_302_6, 0.275_777_1];
    let h = image_size;
    let w = image_size;
    let mut out = vec![0.0_f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = raw[(y * w + x) * 3 + c] as f32 / 255.0;
                out[(c * h + y) * w + x] = (v - mean[c]) / std[c];
            }
        }
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _ = args.cpu; // parity flag — lazy realize chooses backend by router

    // ---- Resolve hub repo / files ------------------------------------------
    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| "llava-hf/llava-1.5-7b-hf".to_string());
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.clone(),
        RepoType::Model,
        args.revision.clone(),
    ));

    let config_path = match args.config_file {
        Some(p) => std::path::PathBuf::from(p),
        None => repo.get("config.json")?,
    };
    let tokenizer_path = match args.tokenizer_file {
        Some(p) => std::path::PathBuf::from(p),
        None => repo.get("tokenizer.json")?,
    };
    let weight_files: Vec<std::path::PathBuf> = match &args.weight_files {
        Some(files) => files.split(',').map(std::path::PathBuf::from).collect(),
        None => fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };

    // ---- Config ------------------------------------------------------------
    let config_str = std::fs::read_to_string(&config_path)?;
    let config = HFLlavaConfig::from_hf_json_str(&config_str)
        .map_err(|e| E::msg(format!("parse llava config: {e}")))?;
    let img_size = config.vision_config.image_size;
    println!(
        "config: vision {}x{} (patch {}), text dim {} / {} layers / vocab {}",
        img_size,
        img_size,
        config.vision_config.patch_size,
        config.text_config.dim,
        config.text_config.n_layers,
        config.text_config.vocab_size,
    );

    // ---- Optional pinpoint selection (info-only for v1) --------------------
    if let Some(s) = args.grid_pinpoints.as_deref() {
        let pins = parse_pinpoints(s)?;
        if !pins.is_empty() {
            let (orig_w, orig_h) = {
                let img = image::ImageReader::open(&args.image)?.decode()?;
                (img.width(), img.height())
            };
            let pick = select_best_resolution((orig_w, orig_h), &pins);
            println!(
                "pinpoints: original {orig_w}x{orig_h} → best fit {}x{} \
                 (v1 still resizes to {img_size}x{img_size} after this)",
                pick.0, pick.1,
            );
        }
    }

    // ---- Tokenizer ---------------------------------------------------------
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

    // ---- Image -------------------------------------------------------------
    let pixel_chw = load_image_chw(&args.image, img_size)?;
    println!("loaded image (1, 3, {img_size}, {img_size})");
    let pixel_values = LazyTensor::from_f32(
        Arc::<[f32]>::from(pixel_chw),
        Shape::from_dims(&[1, 3, img_size, img_size]),
        &Device::cpu(),
    );

    // ---- Weights -----------------------------------------------------------
    let load_start = std::time::Instant::now();
    let st = unsafe { MmapedSafetensors::multi(&weight_files) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = LlavaWeights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load llava weights: {e}")))?;
    let model = LlavaModel {
        config: config.clone(),
        weights,
    };
    println!("loaded weights in {:?}", load_start.elapsed());

    // ---- Prompt → tokens ---------------------------------------------------
    let prompt = args.prompt.clone();
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut tokens: Vec<u32> = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let np = config.vision_config.num_patches();
    let vocab_size = config.text_config.vocab_size;
    let eos_token_id = tokenizer
        .token_to_id("</s>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"));

    // ---- Greedy decode -----------------------------------------------------
    let gen_start = std::time::Instant::now();
    let mut generated = 0_usize;
    for _ in 0..args.sample_len {
        let logits = model
            .forward(&pixel_values, &tokens)
            .map_err(|e| E::msg(format!("llava forward: {e}")))?;
        let data = logits.realize_f32();
        // logits shape (1, num_patches + text_len, vocab); pick the last row.
        let seq = np + tokens.len();
        let off = (seq - 1) * vocab_size;
        let last = &data[off..off + vocab_size];
        let mut best_i = 0_usize;
        let mut best = last[0];
        for (i, &v) in last.iter().enumerate().skip(1) {
            if v > best {
                best = v;
                best_i = i;
            }
        }
        let next = best_i as u32;
        if Some(next) == eos_token_id {
            break;
        }
        tokens.push(next);
        generated += 1;
        let piece = tokenizer.decode(&[next], true).map_err(E::msg)?;
        print!("{piece}");
        std::io::stdout().flush()?;
    }
    let dt = gen_start.elapsed();
    println!(
        "\n{generated} tokens generated in {:?} ({:.2} tok/s)",
        dt,
        generated as f64 / dt.as_secs_f64(),
    );
    Ok(())
}

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::sync::Arc;

use fuel::lazy::LazyTensor;
use fuel::lazy_blip::{BlipConfig, BlipForConditionalGeneration, BlipWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use fuel_examples::token_output_stream::TokenOutputStream;

use tokenizers::Tokenizer;

// TODO: Maybe add support for the conditional prompt.
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Use the quantized version of the model.
    ///
    /// The lazy port ships F32 safetensors only; the eager-only
    /// GGUF quantized path is not yet wired through lazy_blip.
    /// Passing this flag returns an error so users know to fall
    /// back to the eager binary (or wait for the lazy GGUF port).
    #[arg(long)]
    quantized: bool,
}

const SEP_TOKEN_ID: u32 = 102;

/// Loads an image from disk, resizes to 384x384, applies OpenAI
/// normalization, and returns a flat row-major Vec<f32> of length
/// `3 * 384 * 384` laid out as (C, H, W).
fn load_image_as_vec<P: AsRef<std::path::Path>>(p: P) -> Result<Vec<f32>> {
    let img = image::ImageReader::open(p)?
        .decode()?
        .resize_to_fill(384, 384, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let raw = img.into_raw(); // (H, W, C) row-major, u8

    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    let std = [0.26862954f32, 0.261_302_6, 0.275_777_1];

    // Convert HWC u8 → CHW f32 with normalization.
    let h = 384usize;
    let w = 384usize;
    let mut out = vec![0.0f32; 3 * h * w];
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

pub fn main() -> Result<()> {
    let args = Args::parse();

    if args.quantized {
        return Err(E::msg(
            "The lazy blip port does not yet support the quantized GGUF \
             checkpoint; only the F32 safetensors path is wired. Drop \
             --quantized or use the eager binary for the GGUF path.",
        ));
    }

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.repo(hf_hub::Repo::with_revision(
                "Salesforce/blip-image-captioning-large".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/18".to_string(),
            ));
            api.get("model.safetensors")?
        }
        Some(model) => model.into(),
    };
    let tokenizer = match args.tokenizer {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("Salesforce/blip-image-captioning-large".to_string());
            api.get("tokenizer.json")?
        }
        Some(file) => file.into(),
    };
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let mut tokenizer = TokenOutputStream::new(tokenizer);

    // Lazy path currently realizes via the default executor (CPU /
    // router). The `--cpu` flag is preserved for CLI parity with the
    // eager binary but has no effect here.
    let _ = args.cpu;
    let device = Device::cpu();

    let config = BlipConfig::image_captioning_large();

    let image_vec = load_image_as_vec(&args.image)?;
    println!("loaded image ({} f32 values)", image_vec.len());
    let pixel_values = LazyTensor::from_f32(
        Arc::<[f32]>::from(image_vec),
        Shape::from_dims(&[1, 3, 384, 384]),
        &device,
    );

    let st = unsafe { MmapedSafetensors::multi(&[model_file]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = BlipWeights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load blip weights: {e}")))?;
    let model = BlipForConditionalGeneration {
        config: config.clone(),
        weights,
    };
    println!("model built");

    let vocab_size = config.text_config.vocab_size;
    let mut token_ids: Vec<u32> = vec![30522u32];
    for _ in 0..1000 {
        // Lazy text decoder has no KV cache: re-run the full sequence
        // each step. forward() also re-runs vision; on CPU this is
        // O(N) image-encoder evaluations which is slow but correct.
        let logits = model.forward(&pixel_values, &token_ids, 0)?;
        let data = logits.realize_f32();
        let seq = token_ids.len();
        // logits has shape (1, T, vocab); pick the LAST token's row.
        let off = (seq - 1) * vocab_size;
        let last_logits = &data[off..off + vocab_size];

        // Greedy argmax sampler (matches the lazy port's lack of a
        // LogitsProcessor + lack of KV cache — we surface a simple
        // deterministic decode for v1).
        let mut best_i = 0usize;
        let mut best = last_logits[0];
        for (i, &v) in last_logits.iter().enumerate().skip(1) {
            if v > best {
                best = v;
                best_i = i;
            }
        }
        let token = best_i as u32;

        if token == SEP_TOKEN_ID {
            break;
        }
        token_ids.push(token);
        if let Some(t) = tokenizer.next_token(token)? {
            use std::io::Write;
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    println!();
    Ok(())
}

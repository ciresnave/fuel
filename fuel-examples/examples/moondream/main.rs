//! Moondream — lazy port.
//!
//! The eager binary ran a custom incremental generation loop with the
//! Moondream text decoder's `forward` / `forward_with_img` split and a
//! per-step KV cache. The lazy port at `fuel::lazy_moondream` currently
//! exposes a single-pass `forward` that consumes `(pixel_values, &[u32]
//! text_tokens)` and returns logits for the concatenated `[image_features;
//! text_embeds]` sequence. Per-step KV cache reuse is deferred to follow-up
//! work on the lazy module, so this binary re-runs the full forward pass
//! per generated token (correct, just not yet optimal).
//!
//! What this binary does today:
//!   1. Parses CLI args + downloads the Moondream-v1 safetensors and tokenizer.
//!   2. Loads the image, preprocesses it to (1, 3, 378, 378) f32 (NCHW), and
//!      wraps the result in a lazy tensor.
//!   3. Builds a `MoondreamConfig` for v2 (1152-dim vision tower, 2048-dim
//!      Phi-1.5 text decoder).
//!   4. Loads weights via `MoondreamWeights::load_from_mmapped`. NOTE: the
//!      lazy loader is presently a stub — see `lazy_moondream.rs`. The binary
//!      compiles + the error surfaces at runtime when the user invokes it.
//!   5. Runs `MoondreamModel::forward(&pixel_values, &tokens)` per step and
//!      greedy / sampled-decodes the next token using a locally-defined
//!      sampler (mirrors the `helium` lazy binary pattern).
//!
//! Deferrals vs the eager binary:
//!   - Quantized GGUF (q4_0) variant: no `lazy_quantized_moondream` yet;
//!     `--quantized` flag is now rejected.
//!   - `--f16`: the lazy module is F32-only.
//!   - KV-cache: re-runs full forward each step (correctness > perf for v1).

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{bail, Error as E, Result};
use clap::Parser;

use fuel::lazy::LazyTensor;
use fuel::lazy_mixformer::{MixFormerConfig, MixFormerActivation};
use fuel::lazy_moondream::{
    MoondreamConfig, MoondreamModel, MoondreamProjectionConfig, MoondreamVisionConfig,
    MoondreamWeights,
};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Display the token for the specified prompt.
    #[arg(long)]
    verbose_prompt: bool,

    #[arg(long)]
    prompt: String,

    #[arg(long)]
    image: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(long, default_value_t = 5000)]
    sample_len: usize,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// Use the quantized (GGUF q4_0) variant of the model. Currently
    /// rejected — no `lazy_quantized_moondream` module yet.
    #[arg(long)]
    quantized: bool,

    /// Use f16 precision for all the computations rather than f32. Currently
    /// rejected — the lazy Moondream module is F32-only.
    #[arg(long)]
    f16: bool,

    #[arg(long)]
    model_file: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,
}

/// Loads an image from disk using the image crate, this returns a NCHW f32
/// vector with shape `(1, 3, 378, 378)` plus the (channels, height, width)
/// triple. Mean/std are baked into the float buffer; pixels are
/// `(x / 255 - 0.5) / 0.5` so the result lives in `[-1, 1]`.
pub fn load_image_nchw<P: AsRef<std::path::Path>>(p: P) -> Result<Vec<f32>> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(|e| E::msg(format!("decode image: {e}")))?
        .resize_to_fill(378, 378, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    // image::RgbImage gives interleaved HxWxC; we need CHW.
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.into_raw(); // length = h * w * 3, layout HxWxC
    let mut nchw = vec![0.0_f32; 3 * h * w];
    // mean/std: 0.5 for all channels.
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let src = (y * w + x) * 3 + c;
                let dst = c * h * w + y * w + x;
                let v = raw[src] as f32 / 255.0;
                nchw[dst] = (v - 0.5) / 0.5;
            }
        }
    }
    Ok(nchw)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        fuel::utils::with_avx(),
        fuel::utils::with_neon(),
        fuel::utils::with_simd128(),
        fuel::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    if args.quantized {
        bail!(
            "the lazy moondream binary does not yet support --quantized \
             (no lazy_quantized_moondream module). Drop the flag to use the \
             safetensors weights."
        );
    }
    if args.f16 {
        bail!(
            "the lazy moondream binary does not yet support --f16 (lazy_moondream \
             is F32-only). Drop the flag."
        );
    }

    // The lazy realize path runs on CPU/router today; preserve CLI parity but
    // ignore the flag.
    let _ = args.cpu;
    let device = Device::cpu();

    let start = std::time::Instant::now();
    let api = hf_hub::api::tokio::Api::new()?;
    let (model_id, revision) = match args.model_id {
        Some(model_id) => (model_id.to_string(), None),
        None => (
            "vikhyatk/moondream1".to_string(),
            Some("f6e9da68e8f1b78b8f3ee10905d56826db7a5802"),
        ),
    };
    let revision = match (args.revision, revision) {
        (Some(r), _) => r,
        (None, Some(r)) => r.to_string(),
        (None, None) => "main".to_string(),
    };
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id,
        hf_hub::RepoType::Model,
        revision,
    ));
    let model_file = match args.model_file {
        Some(m) => m.into(),
        None => repo.get("model.safetensors").await?,
    };
    let tokenizer = match args.tokenizer_file {
        Some(m) => m.into(),
        None => repo.get("tokenizer.json").await?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let cfg = moondream_v2_lazy_config();
    let st = unsafe { MmapedSafetensors::multi(&[model_file]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = MoondreamWeights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = MoondreamModel {
        config: cfg.clone(),
        weights,
    };
    println!("loaded the model in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let image_data = load_image_nchw(&args.image)?;
    let pixel_values = LazyTensor::from_f32(
        Arc::<[f32]>::from(image_data),
        Shape::from_dims(&[1, cfg.vision.num_channels, cfg.vision.image_size, cfg.vision.image_size]),
        &device,
    );
    println!(
        "loaded image (1, {}, {}, {}) in {:?}",
        cfg.vision.num_channels,
        cfg.vision.image_size,
        cfg.vision.image_size,
        start.elapsed()
    );

    let prompt = format!("\n\nQuestion: {0}\n\nAnswer:", args.prompt);
    let encoded = tokenizer.encode(prompt.as_str(), true).map_err(E::msg)?;
    if encoded.get_ids().is_empty() {
        bail!("Empty prompts are not supported in the Moondream model.")
    }
    if args.verbose_prompt {
        for (token, id) in encoded.get_tokens().iter().zip(encoded.get_ids().iter()) {
            let token = token.replace('▁', " ").replace("<0x0A>", "\n");
            println!("{id:7} -> '{token}'");
        }
    }

    // Moondream tokenizer bos_token and eos_token is "<|endoftext|>".
    let special_token = match tokenizer.get_vocab(true).get("<|endoftext|>") {
        Some(token) => *token,
        None => bail!("cannot find the special token"),
    };
    let (bos_token, eos_token) = (special_token, special_token);

    let mut tokens: Vec<u32> = std::iter::once(bos_token)
        .chain(encoded.get_ids().iter().copied())
        .collect();
    let mut generated_tokens = 0_usize;

    println!("starting the inference loop");
    use std::io::Write;
    let vocab_size = cfg.text.vocab_size;
    let start_gen = std::time::Instant::now();
    for index in 0..args.sample_len {
        // The lazy v1 forward always re-runs the full prefix; no KV cache.
        let logits = model
            .forward(&pixel_values, &tokens)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let logits_data = logits.realize_f32();
        // logits shape: (1, num_patches + text_len, vocab) — pick the last text row.
        let seq = cfg.vision.num_patches + tokens.len();
        let last_off = (seq - 1) * vocab_size;
        let mut last_logits: Vec<f32> = logits_data[last_off..last_off + vocab_size].to_vec();
        if args.repeat_penalty != 1.0 {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            apply_repeat_penalty(&mut last_logits, args.repeat_penalty, &tokens[start_at..]);
        }
        let next_token = sample(
            &last_logits,
            args.temperature.unwrap_or(0.0) as f32,
            args.top_p.map(|p| p as f32),
            args.seed.wrapping_add(index as u64),
        );
        tokens.push(next_token);
        generated_tokens += 1;
        if next_token == eos_token
            || tokens.ends_with(&[27, 10619, 29] /* <END> */)
        {
            break;
        }
        let token_str = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{token_str}");
        std::io::stdout().flush()?;
    }
    let dt = start_gen.elapsed();
    println!(
        "\ngenerated in {} seconds\n{generated_tokens} tokens generated ({:.2} token/s)",
        dt.as_secs_f64(),
        generated_tokens as f64 / dt.as_secs_f64()
    );

    Ok(())
}

/// Moondream-v2 config in the lazy layout. Values mirror the eager
/// `moondream::Config::v2()` (vision + Phi-1.5 text decoder + projection).
fn moondream_v2_lazy_config() -> MoondreamConfig {
    MoondreamConfig {
        vision: MoondreamVisionConfig::v2(),
        projection: MoondreamProjectionConfig::v2(),
        text: MixFormerConfig {
            vocab_size: 51200,
            hidden_size: 2048,
            n_inner: None, // 4 * 2048
            num_hidden_layers: 24,
            num_attention_heads: 32,
            rotary_dim: 32,
            layer_norm_eps: 1e-5,
            max_position_embeddings: 2048,
            rope_theta: 10_000.0,
            hidden_activation: MixFormerActivation::GeluPytorchTanh,
            tie_word_embeddings: false,
        },
    }
}

/// Local repeat penalty (the eager `fuel_transformers::utils` helper expects
/// an eager `Tensor`).
fn apply_repeat_penalty(logits: &mut [f32], penalty: f32, context: &[u32]) {
    let mut seen = std::collections::HashSet::new();
    for &t in context {
        if !seen.insert(t) {
            continue;
        }
        let idx = t as usize;
        if idx < logits.len() {
            let v = logits[idx];
            logits[idx] = if v >= 0.0 { v / penalty } else { v * penalty };
        }
    }
}

/// Local sampler — mirrors the `helium` lazy binary. `temperature <= 0.0`
/// is treated as greedy.
fn sample(
    logits: &[f32],
    temperature: f32,
    top_p: Option<f32>,
    seed: u64,
) -> u32 {
    if temperature <= 0.0 {
        let mut best_i = 0usize;
        let mut best = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > best {
                best = v;
                best_i = i;
            }
        }
        return best_i as u32;
    }
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let inv_t = 1.0 / temperature.max(1e-6);
    let mut probs: Vec<f32> = logits.iter().map(|&x| ((x - max_l) * inv_t).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum.max(1e-30);
    }
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let mut keep_mask: Vec<bool> = vec![true; probs.len()];
    if let Some(p_cut) = top_p {
        let mut cum2 = 0.0;
        let mut allow = true;
        for &i in &idx {
            if !keep_mask[i] {
                continue;
            }
            if !allow {
                keep_mask[i] = false;
                continue;
            }
            cum2 += probs[i];
            if cum2 >= p_cut {
                allow = false;
            }
        }
    }
    let mut filtered: Vec<f32> = probs
        .iter()
        .enumerate()
        .map(|(i, p)| if keep_mask[i] { *p } else { 0.0 })
        .collect();
    let s: f32 = filtered.iter().sum();
    if s > 0.0 {
        for v in &mut filtered {
            *v /= s;
        }
    } else {
        return 0;
    }
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    state ^= state >> 33;
    state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
    state ^= state >> 33;
    let r = (state as f32) / (u64::MAX as f32);
    let mut cum = 0.0;
    for (i, p) in filtered.iter().enumerate() {
        cum += *p;
        if r <= cum {
            return i as u32;
        }
    }
    (filtered.len() - 1) as u32
}


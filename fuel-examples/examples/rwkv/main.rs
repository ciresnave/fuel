//! RWKV (v5 / v7) inference binary on the lazy-graph API.
//!
//! Revival of the retired eager `fuel-rwkv` example. Wires the freshly
//! ported `fuel::lazy_rwkv_tokenizer::Tokenizer` (RWKV's byte-pair
//! tokenizer — *not* a Hugging Face `tokenizers` JSON) to the lazy
//! `Rwkv5Model` / `Rwkv7Model` ports.
//!
//! Scope mirrors the lazy ports themselves: prefill-only, F32, batch=1.
//! Each generated token re-runs the full sequence through the model
//! (same pattern as the lazy `mamba` example) because the lazy modules
//! do not yet expose a resume-from-state entry point.
//!
//! ```bash
//! cargo run --example rwkv --release -- \
//!   --variant 5 --prompt "The smallest prime is "
//!
//! cargo run --example rwkv --release -- \
//!   --variant 7 --prompt "The Eiffel tower is in the city of"
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use std::io::Write;

use fuel::lazy_rwkv5::{Rwkv5Config, Rwkv5Model, Rwkv5Weights};
use fuel::lazy_rwkv7::{Rwkv7Config, Rwkv7Model, Rwkv7Weights};
use fuel::lazy_rwkv_tokenizer::Tokenizer;
use hf_hub::{api::sync::Api, Repo, RepoType};

/// Which RWKV variant to run. Selected via `--variant 5` or `--variant 7`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Variant {
    #[value(name = "5", aliases = ["v5", "rwkv5"])]
    V5,
    #[value(name = "7", aliases = ["v7", "rwkv7"])]
    V7,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// RWKV variant: `5` (Eagle / World) or `7` (Goose).
    #[arg(long, default_value = "5")]
    variant: Variant,

    /// Prompt to seed generation.
    #[arg(long)]
    prompt: String,

    /// Number of tokens to sample after the prompt.
    #[arg(long, short = 'n', default_value_t = 100)]
    sample_len: usize,

    /// RNG seed for temperature sampling.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Sampling temperature. `0` (default) means argmax / greedy.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// HF model repo id. Defaults to a variant-appropriate small model.
    #[arg(long)]
    model_id: Option<String>,

    /// HF model repo revision.
    #[arg(long, default_value = "main")]
    revision: String,

    /// Override the safetensors weight file(s) (comma-separated).
    #[arg(long)]
    weight_files: Option<String>,

    /// Override the config.json path.
    #[arg(long)]
    config_file: Option<String>,

    /// Override the RWKV vocab JSON path. Defaults to the upstream
    /// `lmz/fuel-rwkv` mirror of `rwkv_vocab_v20230424.json`.
    #[arg(long)]
    tokenizer_file: Option<String>,
}

fn default_model_id(v: Variant) -> &'static str {
    match v {
        // World-1B5 — small enough to load on a laptop, exercises the v5 path.
        Variant::V5 => "RWKV/rwkv-5-world-1b5",
        // RWKV-7 G1d 0.1B — smallest published v7 checkpoint.
        Variant::V7 => "BlinkDL/rwkv-7-world",
    }
}

fn default_weight_file(v: Variant) -> &'static str {
    match v {
        Variant::V5 => "model.safetensors",
        // BlinkDL publishes v7 weights as a single safetensors file
        // alongside the json config; this is the smallest g1d release.
        Variant::V7 => "rwkv7-g1d-0.1b-20260129-ctx8192.safetensors",
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let api = Api::new()?;
    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| default_model_id(args.variant).to_string());
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision.clone(),
    ));

    // RWKV tokenizer is *not* a HF `tokenizers` JSON — it's a flat
    // `{"token": id}` byte-pair vocabulary mirrored at lmz/fuel-rwkv.
    let tokenizer_path = match args.tokenizer_file.as_ref() {
        Some(f) => std::path::PathBuf::from(f),
        None => api
            .model("lmz/fuel-rwkv".to_string())
            .get("rwkv_vocab_v20230424.json")?,
    };
    let tokenizer =
        Tokenizer::new(&tokenizer_path).map_err(|e| E::msg(format!("tokenizer: {e}")))?;

    let weight_files: Vec<std::path::PathBuf> = match args.weight_files.as_ref() {
        Some(files) => files.split(',').map(std::path::PathBuf::from).collect(),
        None => vec![repo.get(default_weight_file(args.variant))?],
    };

    let config_file = match args.config_file.as_ref() {
        Some(f) => Some(std::path::PathBuf::from(f)),
        None => match args.variant {
            // v5 ships a HF-style config.json next to the weights.
            Variant::V5 => Some(repo.get("config.json")?),
            // v7 BlinkDL repos publish a json under the same stem as
            // the safetensors file, but config.json may be missing.
            // We try and fall through to a default if absent.
            Variant::V7 => repo.get("config.json").ok(),
        },
    };

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&weight_files) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;

    let start_load = std::time::Instant::now();
    let (vocab_size, logits_step): (usize, Box<dyn Fn(&[u32]) -> Result<Vec<f32>>>) =
        match args.variant {
            Variant::V5 => {
                let cfg = rwkv5_config_from_hf(config_file.as_deref())?;
                let weights = Rwkv5Weights::load_from_mmapped(&st, &cfg)
                    .map_err(|e| E::msg(format!("load v5 weights: {e}")))?;
                let model = Rwkv5Model {
                    config: cfg.clone(),
                    weights,
                };
                let v = cfg.vocab_size;
                let step: Box<dyn Fn(&[u32]) -> Result<Vec<f32>>> = Box::new(move |toks| {
                    let logits = model
                        .forward(toks)
                        .map_err(|e| E::msg(format!("forward v5: {e}")))?;
                    Ok(logits.realize_f32())
                });
                (v, step)
            }
            Variant::V7 => {
                let cfg = rwkv7_config_from_hf(config_file.as_deref(), &st)?;
                let weights = Rwkv7Weights::load_from_mmapped(&st, &cfg)
                    .map_err(|e| E::msg(format!("load v7 weights: {e}")))?;
                let model = Rwkv7Model {
                    config: cfg.clone(),
                    weights,
                };
                let v = cfg.vocab_size;
                let step: Box<dyn Fn(&[u32]) -> Result<Vec<f32>>> = Box::new(move |toks| {
                    let logits = model
                        .forward(toks)
                        .map_err(|e| E::msg(format!("forward v7: {e}")))?;
                    Ok(logits.realize_f32())
                });
                (v, step)
            }
        };
    eprintln!("loaded model in {:?}", start_load.elapsed());

    // Encode prompt, echo it, then sample.
    let mut tokens = tokenizer
        .encode(&args.prompt)
        .map_err(|e| E::msg(format!("encode prompt: {e}")))?;
    print!("{}", args.prompt);
    std::io::stdout().flush()?;

    let mut generated_count = 0usize;
    // Track decoded byte offset so we only print bytes for newly produced
    // tokens — RWKV's byte-pair encoding can split multi-byte UTF-8 across
    // tokens, so we decode the full suffix bytes-string each step.
    let mut last_decoded_bytes = tokenizer.decode_bytes(&tokens).len();
    let start_gen = std::time::Instant::now();

    for index in 0..args.sample_len {
        let logits_full = logits_step(&tokens)?;
        let seq = tokens.len();
        let last_off = (seq - 1) * vocab_size;
        let last_logits = &logits_full[last_off..last_off + vocab_size];
        let next = sample(
            last_logits,
            args.temperature,
            args.seed.wrapping_add(index as u64),
        );
        tokens.push(next);
        generated_count += 1;

        // Stream out any bytes we can now decode. The accumulated tokens
        // may decode to invalid UTF-8 mid-codepoint; in that case we hold
        // the bytes back until the next token produces a valid suffix.
        let all_bytes = tokenizer.decode_bytes(&tokens);
        if all_bytes.len() > last_decoded_bytes {
            let new_bytes = &all_bytes[last_decoded_bytes..];
            // Try to print only the longest valid-utf8 prefix of new_bytes.
            match std::str::from_utf8(new_bytes) {
                Ok(s) => {
                    print!("{s}");
                    last_decoded_bytes = all_bytes.len();
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to > 0 {
                        // SAFETY: bounds-checked by valid_up_to().
                        let s = std::str::from_utf8(&new_bytes[..valid_up_to]).unwrap();
                        print!("{s}");
                        last_decoded_bytes += valid_up_to;
                    }
                }
            }
            std::io::stdout().flush()?;
        }

        // RWKV vocab id 0 is the end-of-text sentinel in BlinkDL's
        // released checkpoints; the retired eager example also stops on 261
        // but treating 0 as the universal EOS is the safer cross-version
        // default for the revived binary.
        if next == 0 {
            break;
        }
    }

    // Flush any held-back trailing bytes (replacement char shows partial UTF-8).
    let all_bytes = tokenizer.decode_bytes(&tokens);
    if all_bytes.len() > last_decoded_bytes {
        let trailing = String::from_utf8_lossy(&all_bytes[last_decoded_bytes..]);
        print!("{trailing}");
    }

    let dt = start_gen.elapsed();
    println!(
        "\n{generated_count} tokens generated ({:.2} token/s)",
        generated_count as f64 / dt.as_secs_f64()
    );
    Ok(())
}

// ── HF config.json parsing ──────────────────────────────────────────────────

fn read_json(path: Option<&std::path::Path>) -> Result<Option<serde_json::Value>> {
    match path {
        None => Ok(None),
        Some(p) => {
            let bytes = std::fs::read(p)?;
            let v: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|e| E::msg(format!("parse config.json: {e}")))?;
            Ok(Some(v))
        }
    }
}

fn rwkv5_config_from_hf(path: Option<&std::path::Path>) -> Result<Rwkv5Config> {
    let v = read_json(path)?
        .ok_or_else(|| E::msg("rwkv v5 requires a config.json (got none)"))?;
    let get_usize = |k: &str| -> Option<usize> { v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize) };
    let get_f64 = |k: &str| -> Option<f64> { v.get(k).and_then(|x| x.as_f64()) };

    let vocab_size = get_usize("vocab_size")
        .ok_or_else(|| E::msg("config.json: missing vocab_size"))?;
    let hidden_size = get_usize("hidden_size")
        .ok_or_else(|| E::msg("config.json: missing hidden_size"))?;
    let num_hidden_layers = get_usize("num_hidden_layers")
        .ok_or_else(|| E::msg("config.json: missing num_hidden_layers"))?;
    let attention_hidden_size = get_usize("attention_hidden_size").unwrap_or(hidden_size);
    let head_size = get_usize("head_size").unwrap_or(64);
    let num_attention_heads = get_usize("num_attention_heads").unwrap_or(0);
    let intermediate_size = get_usize("intermediate_size");
    let layer_norm_epsilon = get_f64("layer_norm_epsilon").unwrap_or(1e-5);
    let rescale_every = get_usize("rescale_every").unwrap_or(6);

    Ok(Rwkv5Config {
        vocab_size,
        hidden_size,
        num_hidden_layers,
        attention_hidden_size,
        head_size,
        num_attention_heads,
        intermediate_size,
        layer_norm_epsilon,
        rescale_every,
    })
}

/// Build a v7 config from the optional `config.json`, falling back to
/// values inferred from the safetensors header when fields are missing
/// (BlinkDL's RWKV-7 releases sometimes ship without a HF-format config).
fn rwkv7_config_from_hf(
    path: Option<&std::path::Path>,
    st: &fuel::safetensors::MmapedSafetensors,
) -> Result<Rwkv7Config> {
    let v = read_json(path)?;
    let get_usize = |k: &str| -> Option<usize> {
        v.as_ref()
            .and_then(|v| v.get(k))
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
    };
    let get_f64 = |k: &str| -> Option<f64> {
        v.as_ref().and_then(|v| v.get(k)).and_then(|x| x.as_f64())
    };

    // First try config.json, then infer from weight shapes.
    let (hidden_size, num_hidden_layers, vocab_size) = match (
        get_usize("hidden_size"),
        get_usize("num_hidden_layers"),
        get_usize("vocab_size"),
    ) {
        (Some(h), Some(l), Some(v)) => (h, l, v),
        _ => infer_v7_shape(st)?,
    };
    let head_size = get_usize("head_size").unwrap_or(64);
    let intermediate_size = get_usize("intermediate_size");
    let layer_norm_epsilon = get_f64("layer_norm_epsilon").unwrap_or(1e-5);

    // LoRA ranks: infer from per-layer weight shapes (`w1: [hidden, d_decay]`
    // etc). The retired eager loader does the same — it's more robust than a
    // formula since BlinkDL tunes these per checkpoint size.
    let (d_decay, d_aaa, d_mv, d_gate) = infer_v7_lora_dims(st, hidden_size)?;

    Ok(Rwkv7Config {
        vocab_size,
        hidden_size,
        num_hidden_layers,
        head_size,
        intermediate_size,
        d_decay,
        d_aaa,
        d_mv,
        d_gate,
        layer_norm_epsilon,
    })
}

/// Infer (hidden_size, num_hidden_layers, vocab_size) from the safetensors
/// header. Walks `rwkv.blocks.{i}.ln1.weight` to count layers and reads
/// `rwkv.embeddings.weight`'s shape for hidden / vocab.
fn infer_v7_shape(
    st: &fuel::safetensors::MmapedSafetensors,
) -> Result<(usize, usize, usize)> {
    let names: Vec<String> = st
        .tensors()
        .iter()
        .map(|(n, _)| n.clone())
        .collect();

    let emb = names
        .iter()
        .find(|n| n.as_str() == "rwkv.embeddings.weight")
        .ok_or_else(|| E::msg("safetensors: missing rwkv.embeddings.weight"))?;
    let view = st
        .get(emb)
        .map_err(|e| E::msg(format!("load {emb}: {e}")))?;
    let dims = view.shape();
    if dims.len() != 2 {
        return Err(E::msg(format!(
            "rwkv.embeddings.weight: expected 2-D, got {dims:?}",
        )));
    }
    let vocab_size = dims[0];
    let hidden_size = dims[1];

    let mut max_layer: i64 = -1;
    for n in &names {
        if let Some(rest) = n.strip_prefix("rwkv.blocks.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(i) = rest[..dot].parse::<i64>() {
                    if i > max_layer {
                        max_layer = i;
                    }
                }
            }
        }
    }
    if max_layer < 0 {
        return Err(E::msg("safetensors: no rwkv.blocks.* tensors found"));
    }
    let num_hidden_layers = (max_layer + 1) as usize;
    Ok((hidden_size, num_hidden_layers, vocab_size))
}

fn infer_v7_lora_dims(
    st: &fuel::safetensors::MmapedSafetensors,
    hidden_size: usize,
) -> Result<(usize, usize, usize, usize)> {
    let dim1 = |name: &str| -> Result<usize> {
        let view = st
            .get(name)
            .map_err(|e| E::msg(format!("load {name}: {e}")))?;
        let dims = view.shape();
        if dims.len() != 2 {
            return Err(E::msg(format!(
                "{name}: expected 2-D, got {dims:?}",
            )));
        }
        if dims[0] != hidden_size {
            return Err(E::msg(format!(
                "{name}: expected first dim == hidden_size={hidden_size}, got {dims:?}",
            )));
        }
        Ok(dims[1])
    };
    let d_decay = dim1("rwkv.blocks.0.attention.w1")?;
    let d_aaa = dim1("rwkv.blocks.0.attention.a1")?;
    // v1 of the layer 0 block has no v1/v2 (value-residual stream starts
    // at layer 1) — read from block 1 if present, else fall back to d_decay.
    let d_mv = dim1("rwkv.blocks.1.attention.v1").unwrap_or(d_decay);
    let d_gate = dim1("rwkv.blocks.0.attention.g1")?;
    Ok((d_decay, d_aaa, d_mv, d_gate))
}

// ── Sampling ────────────────────────────────────────────────────────────────

/// Argmax (`temperature <= 0`) or temperature-only multinomial sampling.
/// Mirrors the lightweight sampler used by other lazy example binaries
/// (yi, mamba) — top-k/top-p are out of scope for the revival.
fn sample(logits: &[f32], temperature: f32, seed: u64) -> u32 {
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
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    // SplitMix64-ish PRNG seeded from `seed` for reproducibility.
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    state ^= state >> 33;
    state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
    state ^= state >> 33;
    let r = (state as f32) / (u64::MAX as f32);
    let mut cum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

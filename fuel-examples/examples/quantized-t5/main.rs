#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use fuel::lazy_quantized_t5::QuantizedT5Model;
use fuel::lazy_t5::{T5Activation, T5Config};
use hf_hub::{api::sync::Api, api::sync::ApiRepo, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, ValueEnum)]
enum Which {
    T5Small,
    FlanT5Small,
    FlanT5Base,
    FlanT5Large,
    FlanT5Xl,
    FlanT5Xxl,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model repository to use on the HuggingFace hub.
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    weight_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    // Enable/disable decoding.
    #[arg(long, default_value = "false")]
    disable_cache: bool,

    /// Use this prompt, otherwise compute sentence similarities.
    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The model size to use.
    #[arg(long, default_value = "t5-small")]
    which: Which,
}

/// Parsed HF `config.json` for T5 / Flan-T5. Mirrors the fields used by the
/// eager `quantized_t5::Config`. The lazy `T5Config` only needs a strict
/// subset, so we hold on to the auxiliary fields (`pad_token_id`,
/// `eos_token_id`, `decoder_start_token_id`) here.
struct LoadedT5 {
    cfg: T5Config,
    pad_token_id: u32,
    eos_token_id: u32,
    decoder_start_token_id: Option<u32>,
}

fn parse_t5_config(json: &str) -> Result<LoadedT5> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| E::msg(format!("parsing T5 config.json: {e}")))?;
    let get_usize = |key: &str| -> Result<usize> {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .ok_or_else(|| E::msg(format!("T5 config.json: missing/invalid field {key:?}")))
    };
    let get_usize_opt = |key: &str| -> Option<usize> {
        v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_f64 = |key: &str, default: f64| -> f64 {
        v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
    };
    let get_bool = |key: &str, default: bool| -> bool {
        v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
    };

    let vocab_size = get_usize("vocab_size")?;
    let d_model = get_usize("d_model")?;
    let d_kv = get_usize("d_kv")?;
    let d_ff = get_usize("d_ff")?;
    let num_layers = get_usize("num_layers")?;
    let num_decoder_layers = get_usize_opt("num_decoder_layers");
    let num_heads = get_usize("num_heads")?;
    let relative_attention_num_buckets = get_usize("relative_attention_num_buckets")?;
    let relative_attention_max_distance =
        get_usize_opt("relative_attention_max_distance").unwrap_or(128);
    let layer_norm_epsilon = get_f64("layer_norm_epsilon", 1e-6);
    let tie_word_embeddings = get_bool("tie_word_embeddings", true);

    // Parse `feed_forward_proj` — HF uses strings like "relu", "gated-gelu",
    // "gated-silu", "gelu_new", etc.
    let ffp: &str = v
        .get("feed_forward_proj")
        .and_then(|x| x.as_str())
        .unwrap_or("relu");
    let (gated_ffn, activation) = match ffp {
        "gated-gelu" => (true, T5Activation::GeluPytorchTanh),
        "gated-silu" => (true, T5Activation::Silu),
        "relu" => (false, T5Activation::Relu),
        "silu" | "swish" => (false, T5Activation::Silu),
        "gelu" => (false, T5Activation::Gelu),
        "gelu_new" | "gelu_pytorch_tanh" => (false, T5Activation::GeluPytorchTanh),
        other => {
            // Tolerate "gated-X" generic shape.
            if let Some(inner) = other.strip_prefix("gated-") {
                let act = match inner {
                    "gelu" => T5Activation::GeluPytorchTanh,
                    "silu" | "swish" => T5Activation::Silu,
                    "relu" => T5Activation::Relu,
                    _ => T5Activation::GeluPytorchTanh,
                };
                (true, act)
            } else {
                (false, T5Activation::Relu)
            }
        }
    };

    let pad_token_id = get_usize_opt("pad_token_id").unwrap_or(0) as u32;
    let eos_token_id = get_usize_opt("eos_token_id").unwrap_or(1) as u32;
    let decoder_start_token_id = get_usize_opt("decoder_start_token_id").map(|x| x as u32);

    Ok(LoadedT5 {
        cfg: T5Config {
            vocab_size,
            d_model,
            d_kv,
            d_ff,
            num_layers,
            num_decoder_layers,
            num_heads,
            relative_attention_num_buckets,
            relative_attention_max_distance,
            layer_norm_epsilon,
            activation,
            gated_ffn,
            tie_word_embeddings,
        },
        pad_token_id,
        eos_token_id,
        decoder_start_token_id,
    })
}

struct T5ModelBuilder {
    loaded: LoadedT5,
    weights_filename: PathBuf,
}

impl T5ModelBuilder {
    pub fn load(args: &Args) -> Result<(Self, Tokenizer)> {
        let default_model = "lmz/fuel-quantized-t5".to_string();
        let (model_id, revision) = match (args.model_id.to_owned(), args.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, "main".to_string()),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = match &args.config_file {
            Some(filename) => Self::get_local_or_remote_file(filename, &api)?,
            None => match args.which {
                Which::T5Small => api.get("config.json")?,
                Which::FlanT5Small => api.get("config-flan-t5-small.json")?,
                Which::FlanT5Base => api.get("config-flan-t5-base.json")?,
                Which::FlanT5Large => api.get("config-flan-t5-large.json")?,
                Which::FlanT5Xl => api.get("config-flan-t5-xl.json")?,
                Which::FlanT5Xxl => api.get("config-flan-t5-xxl.json")?,
            },
        };
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = match &args.weight_file {
            Some(filename) => Self::get_local_or_remote_file(filename, &api)?,
            None => match args.which {
                Which::T5Small => api.get("model.gguf")?,
                Which::FlanT5Small => api.get("model-flan-t5-small.gguf")?,
                Which::FlanT5Base => api.get("model-flan-t5-base.gguf")?,
                Which::FlanT5Large => api.get("model-flan-t5-large.gguf")?,
                Which::FlanT5Xl => api.get("model-flan-t5-xl.gguf")?,
                Which::FlanT5Xxl => api.get("model-flan-t5-xxl.gguf")?,
            },
        };
        let config_json = std::fs::read_to_string(config_filename)?;
        let loaded = parse_t5_config(&config_json)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        Ok((
            Self {
                loaded,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_model(&self) -> Result<QuantizedT5Model> {
        QuantizedT5Model::from_gguf(&self.weights_filename, &self.loaded.cfg)
            .map_err(|e| E::msg(format!("load quantized t5 weights: {e}")))
    }

    fn get_local_or_remote_file(filename: &str, api: &ApiRepo) -> Result<PathBuf> {
        let local_filename = std::path::PathBuf::from(filename);
        if local_filename.exists() {
            Ok(local_filename)
        } else {
            Ok(api.get(filename)?)
        }
    }
}

fn main() -> Result<()> {
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

    // `--disable-cache` retained for CLI compatibility; the lazy port runs
    // on the graph executor (no per-step KV cache yet) so it is effectively
    // a no-op.
    let _ = args.disable_cache;

    let (builder, mut tokenizer) = T5ModelBuilder::load(&args)?;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    let tokens: Vec<u32> = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let model = builder.build_model()?;
    let decoder_start = builder
        .loaded
        .decoder_start_token_id
        .unwrap_or(builder.loaded.pad_token_id);
    let mut output_token_ids: Vec<u32> = vec![decoder_start];
    let temperature = if args.temperature <= 0. {
        None
    } else {
        Some(args.temperature)
    };
    let encoder_out = model
        .forward_encoder(&tokens)
        .map_err(|e| E::msg(format!("encoder forward: {e}")))?;
    let start = std::time::Instant::now();
    let vocab_size = builder.loaded.cfg.vocab_size;
    let eos_token_id = builder.loaded.eos_token_id;

    for index in 0.. {
        if output_token_ids.len() > 512 {
            break;
        }
        // Lazy port has no KV cache — each step re-runs the decoder over
        // the full target prefix.
        let logits = model
            .forward_decoder(&output_token_ids, &encoder_out)
            .map_err(|e| E::msg(format!("decoder forward: {e}")))?;
        let logits_data = logits.realize_f32();
        let tgt_len = output_token_ids.len();
        let last_off = (tgt_len - 1) * vocab_size;
        let mut last_logits: Vec<f32> = logits_data[last_off..last_off + vocab_size].to_vec();
        if args.repeat_penalty != 1.0 {
            let start_at = output_token_ids.len().saturating_sub(args.repeat_last_n);
            apply_repeat_penalty(
                &mut last_logits,
                args.repeat_penalty,
                &output_token_ids[start_at..],
            );
        }
        let next_token_id = sample(
            &last_logits,
            temperature.map(|t| t as f32).unwrap_or(0.0),
            args.top_p.map(|p| p as f32),
            args.seed.wrapping_add(index as u64),
        );
        if next_token_id == eos_token_id {
            break;
        }
        output_token_ids.push(next_token_id);
        if let Some(text) = tokenizer.id_to_token(next_token_id) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            print!("{text}");
            std::io::stdout().flush()?;
        }
    }
    let dt = start.elapsed();
    println!(
        "\n{} tokens generated ({:.2} token/s)\n",
        output_token_ids.len(),
        output_token_ids.len() as f64 / dt.as_secs_f64(),
    );
    Ok(())
}

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

fn sample(logits: &[f32], temperature: f32, top_p: Option<f32>, seed: u64) -> u32 {
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

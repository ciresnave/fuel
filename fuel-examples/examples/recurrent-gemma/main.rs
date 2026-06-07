#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::io::Write;

use fuel::lazy_recurrent_gemma::{
    GemmaActivation, RecurrentGemmaConfig, RecurrentGemmaModel, RecurrentGemmaWeights,
    TemporalBlockType,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "2b")]
    Base2B,
    #[value(name = "2b-it")]
    Instruct2B,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long, default_value_t = 250)]
    top_k: usize,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 8000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The model to use.
    #[arg(long, default_value = "2b")]
    which: Which,

    /// Use a GGUF quantized variant. Not yet supported in the lazy port.
    #[arg(long)]
    quantized: bool,
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
    let _ = fuel_examples::device(args.cpu)?;
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    if args.quantized {
        anyhow::bail!(
            "the lazy recurrent-gemma port does not yet support --quantized; \
             lazy_quantized_recurrent_gemma is not implemented in fuel-core"
        );
    }

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match &args.model_id {
        Some(model_id) => model_id.to_string(),
        None => match args.which {
            Which::Base2B => "google/recurrentgemma-2b".to_string(),
            Which::Instruct2B => "google/recurrentgemma-2b-it".to_string(),
        },
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let config_filename = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("config.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config_json = std::fs::read_to_string(&config_filename)?;
    let config = recurrent_gemma_config_from_hf_json_str(&config_json)?;
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = RecurrentGemmaWeights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load recurrent-gemma weights: {e}")))?;
    let model = RecurrentGemmaModel { config: config.clone(), weights };
    println!("loaded the model in {:?}", start.elapsed());

    let eos_token_id = parse_eos_token_id(&config_json)
        .or_else(|| tokenizer.token_to_id("<eos>"));

    let mut tok_stream = fuel_examples::token_output_stream::TokenOutputStream::new(tokenizer);

    let mut tokens = tok_stream
        .tokenizer()
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    for &t in tokens.iter() {
        if let Some(s) = tok_stream.next_token(t)? {
            print!("{s}");
        }
    }
    std::io::stdout().flush()?;

    let vocab_size = config.vocab_size;
    let mut generated_tokens = 0usize;
    let start_gen = std::time::Instant::now();
    for index in 0..args.sample_len {
        let logits = model
            .forward(&tokens, 0)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let logits_data = logits.realize_f32();
        let seq = tokens.len();
        let last_off = (seq - 1) * vocab_size;
        let mut last_logits: Vec<f32> = logits_data[last_off..last_off + vocab_size].to_vec();
        if args.repeat_penalty != 1.0 {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            apply_repeat_penalty(&mut last_logits, args.repeat_penalty, &tokens[start_at..]);
        }
        let next_token = sample(
            &last_logits,
            args.temperature.map(|t| t as f32).unwrap_or(0.0),
            Some(args.top_k),
            args.top_p.map(|p| p as f32),
            args.seed.wrapping_add(index as u64),
        );
        tokens.push(next_token);
        generated_tokens += 1;
        if Some(next_token) == eos_token_id {
            break;
        }
        if let Some(t) = tok_stream.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    let dt = start_gen.elapsed();
    if let Some(rest) = tok_stream.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    std::io::stdout().flush()?;
    println!(
        "\n{generated_tokens} tokens generated ({:.2} token/s)",
        generated_tokens as f64 / dt.as_secs_f64(),
    );
    Ok(())
}

fn recurrent_gemma_config_from_hf_json_str(json: &str) -> Result<RecurrentGemmaConfig> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let get_usize = |key: &str| -> Result<usize> {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .ok_or_else(|| E::msg(format!("config.json: missing/invalid field {key:?}")))
    };
    let get_usize_or = |key: &str, default: usize| -> usize {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(default)
    };
    let get_usize_opt = |key: &str| -> Option<usize> {
        v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_f64_or = |key: &str, default: f64| -> f64 {
        v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
    };
    let get_bool_or = |key: &str, default: bool| -> bool {
        v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
    };

    let vocab_size = get_usize("vocab_size")?;
    let hidden_size = get_usize("hidden_size")?;
    let intermediate_size = get_usize("intermediate_size")?;
    let num_hidden_layers = get_usize("num_hidden_layers")?;
    let num_attention_heads = get_usize("num_attention_heads")?;
    let num_key_value_heads = get_usize_or("num_key_value_heads", num_attention_heads);
    let head_dim = get_usize_or("head_dim", hidden_size / num_attention_heads);
    let lru_width = get_usize_opt("lru_width");
    let attention_window_size = get_usize_or("attention_window_size", 2048);
    let conv1d_width = get_usize_or("conv1d_width", 4);
    let logits_soft_cap = get_f64_or("logits_soft_cap", 0.0);
    let partial_rotary_factor = get_f64_or("partial_rotary_factor", 0.5);
    let rms_norm_eps = get_f64_or("rms_norm_eps", 1e-6);
    let rope_theta = get_f64_or("rope_theta", 10_000.0);
    let attention_bias = get_bool_or("attention_bias", false);
    let max_seq_len = get_usize_or("max_seq_len", 8192);

    let hidden_activation = match v.get("hidden_activation").and_then(|x| x.as_str()) {
        Some("gelu_pytorch_tanh") => GemmaActivation::GeluPytorchTanh,
        Some("gelu") => GemmaActivation::Gelu,
        Some(other) => {
            return Err(E::msg(format!(
                "config.json: unsupported hidden_activation {other:?}"
            )));
        }
        // The recurrent-gemma reference uses gelu_pytorch_tanh by default.
        None => GemmaActivation::GeluPytorchTanh,
    };

    let block_types = parse_block_types(&v)?;

    Ok(RecurrentGemmaConfig {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        lru_width,
        attention_window_size,
        conv1d_width,
        logits_soft_cap,
        hidden_activation,
        partial_rotary_factor,
        rms_norm_eps,
        rope_theta,
        block_types,
        attention_bias,
        max_seq_len,
    })
}

fn parse_block_types(v: &serde_json::Value) -> Result<Vec<TemporalBlockType>> {
    let arr = v
        .get("block_types")
        .or_else(|| v.get("_block_types"))
        .and_then(|x| x.as_array())
        .ok_or_else(|| E::msg("config.json: missing field 'block_types'"))?;
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let s = entry.as_str().ok_or_else(|| {
            E::msg("config.json: block_types entries must be strings")
        })?;
        let bt = match s {
            "attention" => TemporalBlockType::Attention,
            "recurrent" => TemporalBlockType::Recurrent,
            other => {
                return Err(E::msg(format!(
                    "config.json: unknown block type {other:?}"
                )));
            }
        };
        out.push(bt);
    }
    if out.is_empty() {
        return Err(E::msg("config.json: block_types must be non-empty"));
    }
    Ok(out)
}

fn parse_eos_token_id(json: &str) -> Option<u32> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    v.get("eos_token_id").and_then(|x| x.as_u64()).map(|x| x as u32)
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

fn sample(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
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
    if let Some(k) = top_k {
        for &i in idx.iter().skip(k) {
            keep_mask[i] = false;
        }
    }
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
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

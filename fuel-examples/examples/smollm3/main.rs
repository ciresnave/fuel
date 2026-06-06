#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::io::Write;

use fuel::lazy::{LlamaConfig, LlamaWeights};
use fuel::lazy_smollm3::{SmolLm3Config, SmolLm3Model, SmolLm3Weights};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

const DEFAULT_PROMPT: &str = "Write a Rust function to calculate the factorial of a given number.";

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum WhichModel {
    #[value(name = "3b")]
    W3b,
    #[value(name = "3b-base")]
    W3bBase,
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

    /// Which model variant to use.
    #[arg(long, default_value = "3b")]
    model: WhichModel,

    /// Path to model file (optional, will auto-download if not provided).
    #[arg(long)]
    model_path: Option<String>,

    /// Path to tokenizer file (optional, will auto-download if not provided).
    #[arg(long)]
    tokenizer: Option<String>,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 1000)]
    sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
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

    let _device = fuel_examples::device(args.cpu)?;
    let api = Api::new()?;
    let model_id = match args.model {
        WhichModel::W3b => "HuggingFaceTB/SmolLM3-3B",
        WhichModel::W3bBase => "HuggingFaceTB/SmolLM3-3B-Base",
    };
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    let tokenizer_filename = match args.tokenizer.as_ref() {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames: Vec<std::path::PathBuf> = match args.model_path.as_ref() {
        Some(path) => vec![std::path::PathBuf::from(path)],
        None => fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let config_file = repo.get("config.json")?;
    let config_json = std::fs::read_to_string(&config_file)?;
    let cfg: SmolLm3Config = smollm3_config_from_hf_json_str(&config_json)?;
    let eos_token_id = parse_eos_token_id(&config_json);

    let llama_cfg = LlamaConfig {
        vocab_size: cfg.vocab_size,
        dim:        cfg.hidden_size,
        n_layers:   cfg.num_hidden_layers,
        n_heads:    cfg.num_attention_heads,
        n_kv_heads: cfg.num_key_value_heads,
        head_dim:   cfg.head_dim,
        ffn_dim:    cfg.intermediate_size,
        norm_eps:   cfg.rms_norm_eps,
        rope_base:  cfg.rope_theta,
    };
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let llama_weights: LlamaWeights = LlamaWeights::load_from_mmapped(&st, &llama_cfg)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let weights = SmolLm3Weights {
        token_embedding: llama_weights.token_embedding,
        layers: llama_weights.layers,
        final_norm_gain: llama_weights.final_norm_gain,
        output: llama_weights.output,
    };
    let model = SmolLm3Model { config: cfg.clone(), weights };

    let prompt = args.prompt.clone().unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    let mut tok_stream = fuel_examples::token_output_stream::TokenOutputStream::new(tokenizer);
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut tokens = tok_stream
        .tokenizer()
        .encode(prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let mut generated_tokens: usize = 0;
    let start_gen = std::time::Instant::now();
    for index in 0..args.sample_len {
        let logits = model
            .forward(&tokens, 0)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let logits_data = logits.realize_f32();
        let vocab_size = cfg.vocab_size;
        let seq = tokens.len();
        let last_off = (seq - 1) * vocab_size;
        let mut last_logits: Vec<f32> = logits_data[last_off..last_off + vocab_size].to_vec();
        if args.repeat_penalty != 1.0 {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            apply_repeat_penalty(&mut last_logits, args.repeat_penalty, &tokens[start_at..]);
        }
        let next_token = sample(
            &last_logits,
            args.temperature as f32,
            args.top_k,
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

fn smollm3_config_from_hf_json_str(json: &str) -> Result<SmolLm3Config> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let get_usize = |key: &str| -> Option<usize> {
        v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };
    let get_bool = |key: &str| -> Option<bool> { v.get(key).and_then(|x| x.as_bool()) };
    let vocab_size = get_usize("vocab_size").unwrap_or(128_256);
    let hidden_size = get_usize("hidden_size").unwrap_or(2048);
    let intermediate_size = get_usize("intermediate_size").unwrap_or(11_008);
    let num_hidden_layers = get_usize("num_hidden_layers").unwrap_or(36);
    let num_attention_heads = get_usize("num_attention_heads").unwrap_or(16);
    let num_key_value_heads = get_usize("num_key_value_heads").unwrap_or(4);
    let head_dim = get_usize("head_dim").unwrap_or(hidden_size / num_attention_heads);
    let rms_norm_eps = get_f64("rms_norm_eps").unwrap_or(1e-6);
    let rope_theta = get_f64("rope_theta").unwrap_or(5_000_000.0);
    let max_position_embeddings = get_usize("max_position_embeddings").unwrap_or(65_536);
    let attention_bias = get_bool("attention_bias").unwrap_or(false);
    let sliding_window = get_usize("sliding_window");
    let no_rope_layers = v
        .get("no_rope_layers")
        .and_then(|arr| arr.as_array())
        .map(|arr| arr.iter()
            .filter_map(|x| x.as_u64().map(|v| v as usize))
            .collect::<Vec<_>>());
    Ok(SmolLm3Config {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps,
        rope_theta,
        max_position_embeddings,
        attention_bias,
        sliding_window,
        no_rope_layers,
    })
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

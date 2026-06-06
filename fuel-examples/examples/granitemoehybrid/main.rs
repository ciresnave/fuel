// Granite 4.0 Micro text generation example (GraniteMoeHybrid).

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};
use clap::Parser;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use fuel::lazy_granitemoehybrid::{
    GraniteLayerType, GraniteMoeHybridConfig, GraniteMoeHybridModel, GraniteMoeHybridWeights,
    GraniteRopeScaling,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

const EOS_TOKEN_ID: u32 = 100257;
const DEFAULT_PROMPT: &str = "How Fault Tolerant Quantum Computers will help humanity?";
const DEFAULT_MODEL_ID: &str = "ibm-granite/granite-4.0-micro";

fn build_chat_prompt(user_prompt: &str) -> String {
    format!(
        "<|start_of_role|>user<|end_of_role|>{user_prompt}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
    )
}

fn init_tracing(enable: bool) {
    if !enable {
        return;
    }
    let (chrome_layer, _) = ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The temperature used to generate samples.
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

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 4096)]
    sample_len: usize,

    #[arg(long)]
    no_kv_cache: bool,

    #[arg(long)]
    prompt: Option<String>,

    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Override the model identifier or directory.
    #[arg(long)]
    model_id: Option<String>,

    /// Use a specific revision when loading from the Hugging Face Hub.
    #[arg(long)]
    revision: Option<String>,

    /// Enable Flash-Attention kernels when compiled with the feature.
    #[arg(long)]
    use_flash_attn: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();
    init_tracing(args.tracing);
    let _ = fuel_examples::device(args.cpu)?;
    let _ = args.dtype;
    let _ = args.use_flash_attn;
    let _ = args.no_kv_cache;

    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());
    println!("Loading the model weights from {model_id}");

    let (tokenizer_filename, config_filename, filenames) = if Path::new(&model_id).exists() {
        let model_path = Path::new(&model_id);
        let tokenizer_filename = model_path.join("tokenizer.json");
        let config_filename = model_path.join("config.json");
        let filenames = fuel_examples::hub_load_local_safetensors(
            model_path,
            "model.safetensors.index.json",
        )?;
        (tokenizer_filename, config_filename, filenames)
    } else {
        let api = Api::new()?;
        let revision = args.revision.clone().unwrap_or_else(|| "main".to_string());
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let config_filename = repo.get("config.json")?;
        let filenames =
            fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;
        (tokenizer_filename, config_filename, filenames)
    };

    let config_json = std::fs::read_to_string(&config_filename)?;
    let config = granitemoehybrid_config_from_hf_json_str(&config_json)?;
    let eos_token_id = parse_eos_token_id(&config_json).unwrap_or(EOS_TOKEN_ID);

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = GraniteMoeHybridWeights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = GraniteMoeHybridModel { config: config.clone(), weights };

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let user_prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let chat_prompt = build_chat_prompt(user_prompt);
    let mut tokens = tokenizer
        .encode(chat_prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    println!("Starting the inference loop:");
    println!("User: {user_prompt}\n");
    print!("Assistant: ");
    std::io::stdout().flush()?;

    let mut start_gen = Instant::now();
    let mut token_generated = 0usize;

    for index in 0..args.sample_len {
        if index == 1 {
            start_gen = Instant::now();
        }
        let logits = model
            .forward(&tokens, 0)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let logits_data = logits.realize_f32();
        let vocab_size = config.vocab_size;
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
        token_generated += 1;
        tokens.push(next_token);
        if next_token == eos_token_id {
            break;
        }
        let tok = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{tok}");
        std::io::stdout().flush()?;
    }

    let duration = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated.saturating_sub(1)) as f64 / duration.as_secs_f64(),
    );
    Ok(())
}

fn granitemoehybrid_config_from_hf_json_str(json: &str) -> Result<GraniteMoeHybridConfig> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let get_usize = |key: &str| -> Option<usize> {
        v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };
    let get_f32 = |key: &str| -> Option<f32> { get_f64(key).map(|x| x as f32) };

    let vocab_size = get_usize("vocab_size")
        .ok_or_else(|| E::msg("missing vocab_size"))?;
    let hidden_size = get_usize("hidden_size")
        .ok_or_else(|| E::msg("missing hidden_size"))?;
    let intermediate_size = get_usize("intermediate_size")
        .ok_or_else(|| E::msg("missing intermediate_size"))?;
    let shared_intermediate_size =
        get_usize("shared_intermediate_size").unwrap_or(intermediate_size);
    let num_hidden_layers = get_usize("num_hidden_layers")
        .ok_or_else(|| E::msg("missing num_hidden_layers"))?;
    let num_attention_heads = get_usize("num_attention_heads")
        .ok_or_else(|| E::msg("missing num_attention_heads"))?;
    let num_key_value_heads = get_usize("num_key_value_heads").unwrap_or(num_attention_heads);
    let max_position_embeddings = get_usize("max_position_embeddings").unwrap_or(4096);
    let rms_norm_eps = get_f64("rms_norm_eps").unwrap_or(1e-5);
    let rope_theta = get_f32("rope_theta").unwrap_or(10_000.0);

    // Layer types: parse "layer_types" list or fall back to all-attention.
    let layer_types: Vec<GraniteLayerType> = v
        .get("layer_types")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .map(|val| match val.as_str() {
                    Some("mamba") | Some("Mamba") => GraniteLayerType::Mamba,
                    _ => GraniteLayerType::Attention,
                })
                .collect()
        })
        .unwrap_or_else(|| vec![GraniteLayerType::Attention; num_hidden_layers]);

    if layer_types.len() != num_hidden_layers {
        bail!(
            "layer_types length ({}) does not match num_hidden_layers ({})",
            layer_types.len(),
            num_hidden_layers,
        );
    }

    let attention_multiplier = get_f32("attention_multiplier").unwrap_or(1.0);
    let embedding_multiplier = get_f32("embedding_multiplier").unwrap_or(1.0);
    let residual_multiplier = get_f32("residual_multiplier").unwrap_or(1.0);
    let logits_scaling = get_f32("logits_scaling").unwrap_or(1.0);

    // Parse rope_scaling if present.
    let rope_scaling = v.get("rope_scaling").and_then(|rs| {
        let factor = rs.get("factor").and_then(|x| x.as_f64())? as f32;
        let low_freq_factor = rs.get("low_freq_factor").and_then(|x| x.as_f64())? as f32;
        let high_freq_factor = rs.get("high_freq_factor").and_then(|x| x.as_f64())? as f32;
        let original_max_position_embeddings = rs
            .get("original_max_position_embeddings")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)?;
        Some(GraniteRopeScaling {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        })
    });

    Ok(GraniteMoeHybridConfig {
        vocab_size,
        hidden_size,
        intermediate_size,
        shared_intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        rope_scaling,
        layer_types,
        attention_multiplier,
        embedding_multiplier,
        residual_multiplier,
        logits_scaling,
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

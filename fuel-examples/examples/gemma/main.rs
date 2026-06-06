#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::io::Write;

use fuel::lazy_gemma::{GemmaActivation, GemmaConfig, GemmaModel, GemmaWeights};
use fuel::lazy_gemma2::{Gemma2Config, Gemma2Model, Gemma2Weights};
use fuel::lazy_gemma3::{Gemma3Config, Gemma3Model, Gemma3Weights};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "2b")]
    Base2B,
    #[value(name = "7b")]
    Base7B,
    #[value(name = "2b-it")]
    Instruct2B,
    #[value(name = "7b-it")]
    Instruct7B,
    #[value(name = "1.1-2b-it")]
    InstructV1_1_2B,
    #[value(name = "1.1-7b-it")]
    InstructV1_1_7B,
    #[value(name = "code-2b")]
    CodeBase2B,
    #[value(name = "code-7b")]
    CodeBase7B,
    #[value(name = "code-2b-it")]
    CodeInstruct2B,
    #[value(name = "code-7b-it")]
    CodeInstruct7B,
    #[value(name = "2-2b")]
    BaseV2_2B,
    #[value(name = "2-2b-it")]
    InstructV2_2B,
    #[value(name = "2-9b")]
    BaseV2_9B,
    #[value(name = "2-9b-it")]
    InstructV2_9B,
    #[value(name = "3-1b")]
    BaseV3_1B,
    #[value(name = "3-1b-it")]
    InstructV3_1B,
}

enum LazyModel {
    V1(GemmaModel, usize),
    V2(Gemma2Model, usize),
    V3(Gemma3Model, usize),
}

impl LazyModel {
    fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        match self {
            Self::V1(m, _) => m
                .forward(tokens, 0)
                .map(|l| l.realize_f32())
                .map_err(|e| E::msg(format!("forward: {e}"))),
            Self::V2(m, _) => m
                .forward(tokens, 0)
                .map(|l| l.realize_f32())
                .map_err(|e| E::msg(format!("forward: {e}"))),
            Self::V3(m, _) => m
                .forward(tokens, 0)
                .map(|l| l.realize_f32())
                .map_err(|e| E::msg(format!("forward: {e}"))),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::V1(_, v) | Self::V2(_, v) | Self::V3(_, v) => *v,
        }
    }
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

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
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
    #[arg(long, default_value = "2-2b")]
    which: Which,

    #[arg(long)]
    use_flash_attn: bool,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _ = args.use_flash_attn;
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    let _ = fuel_examples::device(args.cpu)?;

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match &args.model_id {
        Some(model_id) => model_id.to_string(),
        None => match args.which {
            Which::InstructV1_1_2B => "google/gemma-1.1-2b-it".to_string(),
            Which::InstructV1_1_7B => "google/gemma-1.1-7b-it".to_string(),
            Which::Base2B => "google/gemma-2b".to_string(),
            Which::Base7B => "google/gemma-7b".to_string(),
            Which::Instruct2B => "google/gemma-2b-it".to_string(),
            Which::Instruct7B => "google/gemma-7b-it".to_string(),
            Which::CodeBase2B => "google/codegemma-2b".to_string(),
            Which::CodeBase7B => "google/codegemma-7b".to_string(),
            Which::CodeInstruct2B => "google/codegemma-2b-it".to_string(),
            Which::CodeInstruct7B => "google/codegemma-7b-it".to_string(),
            Which::BaseV2_2B => "google/gemma-2-2b".to_string(),
            Which::InstructV2_2B => "google/gemma-2-2b-it".to_string(),
            Which::BaseV2_9B => "google/gemma-2-9b".to_string(),
            Which::InstructV2_9B => "google/gemma-2-9b-it".to_string(),
            Which::BaseV3_1B => "google/gemma-3-1b-pt".to_string(),
            Which::InstructV3_1B => "google/gemma-3-1b-it".to_string(),
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
        None => match args.which {
            Which::BaseV3_1B | Which::InstructV3_1B => vec![repo.get("model.safetensors")?],
            _ => fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
        },
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let config_json = std::fs::read_to_string(&config_filename)?;
    let eos_token_id = parse_eos_token_id(&config_json);

    let model = match args.which {
        Which::Base2B
        | Which::Base7B
        | Which::Instruct2B
        | Which::Instruct7B
        | Which::InstructV1_1_2B
        | Which::InstructV1_1_7B
        | Which::CodeBase2B
        | Which::CodeBase7B
        | Which::CodeInstruct2B
        | Which::CodeInstruct7B => {
            let config = gemma_config_from_hf_json_str(&config_json)?;
            let weights = GemmaWeights::load_from_mmapped(&st, &config)
                .map_err(|e| E::msg(format!("load gemma weights: {e}")))?;
            let vocab = config.vocab_size;
            LazyModel::V1(GemmaModel { config, weights }, vocab)
        }
        Which::BaseV2_2B | Which::InstructV2_2B | Which::BaseV2_9B | Which::InstructV2_9B => {
            let config = gemma2_config_from_hf_json_str(&config_json)?;
            let weights = Gemma2Weights::load_from_mmapped(&st, &config)
                .map_err(|e| E::msg(format!("load gemma2 weights: {e}")))?;
            let vocab = config.vocab_size;
            LazyModel::V2(Gemma2Model { config, weights }, vocab)
        }
        Which::BaseV3_1B | Which::InstructV3_1B => {
            let config = gemma3_config_from_hf_json_str(&config_json)?;
            let weights = Gemma3Weights::load_from_mmapped(&st, &config)
                .map_err(|e| E::msg(format!("load gemma3 weights: {e}")))?;
            let vocab = config.vocab_size;
            LazyModel::V3(Gemma3Model { config, weights }, vocab)
        }
    };
    println!("loaded the model in {:?}", start.elapsed());

    let prompt = match args.which {
        Which::InstructV3_1B => {
            format!(
                "<start_of_turn> user\n{}<end_of_turn>\n<start_of_turn> model\n",
                args.prompt
            )
        }
        _ => args.prompt.clone(),
    };

    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let eos_token_id = eos_token_id.or_else(|| tokenizer.token_to_id("<eos>"));
    let eot_token_id = tokenizer.token_to_id("<end_of_turn>").or(eos_token_id);

    let mut generated_tokens = 0usize;
    let start_gen = std::time::Instant::now();
    for index in 0..args.sample_len {
        let logits_data = model.forward(&tokens)?;
        let vocab_size = model.vocab_size();
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
            args.top_p.map(|p| p as f32),
            args.seed.wrapping_add(index as u64),
        );
        tokens.push(next_token);
        generated_tokens += 1;
        if Some(next_token) == eos_token_id || Some(next_token) == eot_token_id {
            break;
        }
        let tok = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{tok}");
        std::io::stdout().flush()?;
    }
    let dt = start_gen.elapsed();
    println!(
        "\n{generated_tokens} tokens generated ({:.2} token/s)",
        generated_tokens as f64 / dt.as_secs_f64(),
    );
    Ok(())
}

fn gemma_config_from_hf_json_str(json: &str) -> Result<GemmaConfig> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    let get_usize = |key: &str, default: usize| -> usize {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(default)
    };
    let get_f64 = |key: &str, default: f64| -> f64 {
        v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
    };
    let get_bool = |key: &str, default: bool| -> bool {
        v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
    };
    Ok(GemmaConfig {
        vocab_size: get_usize("vocab_size", 256_000),
        hidden_size: get_usize("hidden_size", 2048),
        intermediate_size: get_usize("intermediate_size", 16_384),
        num_hidden_layers: get_usize("num_hidden_layers", 18),
        num_attention_heads: get_usize("num_attention_heads", 8),
        num_key_value_heads: get_usize("num_key_value_heads", 1),
        head_dim: get_usize("head_dim", 256),
        rms_norm_eps: get_f64("rms_norm_eps", 1e-6),
        rope_theta: get_f64("rope_theta", 10_000.0),
        max_position_embeddings: get_usize("max_position_embeddings", 8192),
        attention_bias: get_bool("attention_bias", false),
        hidden_activation: GemmaActivation::GeluPytorchTanh,
    })
}

fn gemma2_config_from_hf_json_str(json: &str) -> Result<Gemma2Config> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    let get_usize = |key: &str, default: usize| -> usize {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(default)
    };
    let get_usize_opt = |key: &str| -> Option<usize> {
        v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_f64 = |key: &str, default: f64| -> f64 {
        v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
    };
    let get_f64_opt = |key: &str| -> Option<f64> {
        v.get(key).and_then(|x| x.as_f64())
    };
    let get_bool = |key: &str, default: bool| -> bool {
        v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
    };
    Ok(Gemma2Config {
        vocab_size: get_usize("vocab_size", 256_000),
        hidden_size: get_usize("hidden_size", 2304),
        intermediate_size: get_usize("intermediate_size", 9216),
        num_hidden_layers: get_usize("num_hidden_layers", 26),
        num_attention_heads: get_usize("num_attention_heads", 8),
        num_key_value_heads: get_usize("num_key_value_heads", 4),
        head_dim: get_usize("head_dim", 256),
        rms_norm_eps: get_f64("rms_norm_eps", 1e-6),
        rope_theta: get_f64("rope_theta", 10_000.0),
        attention_bias: get_bool("attention_bias", false),
        max_position_embeddings: get_usize("max_position_embeddings", 8192),
        attn_logit_softcapping: get_f64_opt("attn_logit_softcapping"),
        final_logit_softcapping: get_f64_opt("final_logit_softcapping"),
        sliding_window: get_usize_opt("sliding_window"),
    })
}

fn gemma3_config_from_hf_json_str(json: &str) -> Result<Gemma3Config> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    // gemma3 sometimes embeds text_config; promote it if present.
    let v = v.get("text_config").cloned().unwrap_or(v);
    let get_usize = |key: &str, default: usize| -> usize {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(default)
    };
    let get_f64 = |key: &str, default: f64| -> f64 {
        v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
    };
    let get_f64_opt = |key: &str| -> Option<f64> {
        v.get(key).and_then(|x| x.as_f64())
    };
    let get_bool = |key: &str, default: bool| -> bool {
        v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
    };
    Ok(Gemma3Config {
        vocab_size: get_usize("vocab_size", 262_144),
        hidden_size: get_usize("hidden_size", 1152),
        intermediate_size: get_usize("intermediate_size", 6912),
        num_hidden_layers: get_usize("num_hidden_layers", 26),
        num_attention_heads: get_usize("num_attention_heads", 4),
        num_key_value_heads: get_usize("num_key_value_heads", 1),
        head_dim: get_usize("head_dim", 256),
        rms_norm_eps: get_f64("rms_norm_eps", 1e-6),
        rope_theta: get_f64("rope_theta", 1_000_000.0),
        rope_local_base_freq: get_f64("rope_local_base_freq", 10_000.0),
        max_position_embeddings: get_usize("max_position_embeddings", 32_768),
        sliding_window: get_usize("sliding_window", 512),
        sliding_window_pattern: get_usize("sliding_window_pattern", 6),
        attention_bias: get_bool("attention_bias", false),
        hidden_activation: GemmaActivation::GeluPytorchTanh,
        attn_logit_softcapping: get_f64_opt("attn_logit_softcapping"),
        final_logit_softcapping: get_f64_opt("final_logit_softcapping"),
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

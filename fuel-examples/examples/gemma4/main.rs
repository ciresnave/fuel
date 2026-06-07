#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::io::Write;

use fuel::lazy_gemma::GemmaActivation;
use fuel::lazy_gemma4_text::{
    Gemma4LayerType, Gemma4TextConfig, Gemma4TextModel, Gemma4TextWeights,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

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
    use_flash_attn: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

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

    /// Load the multimodal model (vision + audio encoders).
    ///
    /// The lazy port currently only ships the text decoder
    /// (`lazy_gemma4_text`). Passing `--multimodal` only affects
    /// the config-parsing path (read the `text_config` sub-object
    /// from the multimodal config), the vision and audio towers
    /// are not wired through yet.
    #[arg(long)]
    multimodal: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Use the slower dmmv cuda kernel.
    #[arg(long)]
    force_dmmv: bool,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _ = args.use_flash_attn;
    #[cfg(feature = "cuda")]
    fuel::quantized::cuda::set_force_dmmv(args.force_dmmv);
    let _ = args.force_dmmv;

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

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| "google/gemma-4-E4B-it".to_string());
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
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
    let _ = fuel_examples::device(args.cpu)?;

    let config_filename = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("config.json")?,
    };
    let config_json = std::fs::read_to_string(&config_filename)?;
    let (config, eos_token_id, eot_token_id) =
        gemma4_text_config_from_hf_json_str(&config_json, args.multimodal)?;

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = Gemma4TextWeights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load gemma4 text weights: {e}")))?;
    let vocab_size = config.vocab_size;
    let model = Gemma4TextModel { config, weights };

    println!("loaded the model in {:?}", start.elapsed());

    print!("{}", args.prompt);
    std::io::stdout().flush()?;
    let mut tokens = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let eos_token_id = eos_token_id
        .or_else(|| tokenizer.token_to_id("</s>"))
        .or_else(|| tokenizer.token_to_id("<eos>"));
    let eot_token_id = eot_token_id.or_else(|| tokenizer.token_to_id("<end_of_turn>"));

    let mut generated_tokens = 0usize;
    let start_gen = std::time::Instant::now();
    for index in 0..args.sample_len {
        let logits_data = model
            .forward(&tokens, 0)
            .map_err(|e| E::msg(format!("forward: {e}")))?
            .realize_f32();
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
            args.top_k,
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

/// Parse the HF JSON config into a lazy `Gemma4TextConfig`.
///
/// When `multimodal` is true (or when the JSON has a `text_config`
/// sub-object), the text-decoder config is taken from the inner
/// `text_config` field; otherwise the top-level JSON is parsed as
/// the text config directly.
///
/// Returns `(config, eos_token_id, end_of_turn_token_id)`.
fn gemma4_text_config_from_hf_json_str(
    json: &str,
    multimodal: bool,
) -> Result<(Gemma4TextConfig, Option<u32>, Option<u32>)> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    let eos_token_id = v
        .get("eos_token_id")
        .and_then(|x| x.as_u64())
        .map(|x| x as u32);
    let eot_token_id = v
        .get("eoi_token_id")
        .and_then(|x| x.as_u64())
        .map(|x| x as u32);

    // Pick the text-config sub-object when present (multimodal config)
    // or when the user explicitly asked for it.
    let text_v = match (multimodal, v.get("text_config")) {
        (_, Some(tc)) => tc.clone(),
        (true, None) => v.clone(),
        (false, None) => v.clone(),
    };

    let get_usize = |key: &str, default: usize| -> usize {
        text_v
            .get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(default)
    };
    let get_usize_opt = |key: &str| -> Option<usize> {
        text_v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_f64 = |key: &str, default: f64| -> f64 {
        text_v
            .get(key)
            .and_then(|x| x.as_f64())
            .unwrap_or(default)
    };
    let get_f64_opt = |key: &str| -> Option<f64> {
        text_v.get(key).and_then(|x| x.as_f64())
    };
    let get_bool = |key: &str, default: bool| -> bool {
        text_v
            .get(key)
            .and_then(|x| x.as_bool())
            .unwrap_or(default)
    };

    // Layer-type strings → enum vec.
    let num_hidden_layers = get_usize("num_hidden_layers", 26);
    let layer_types: Vec<Gemma4LayerType> = text_v
        .get("layer_types")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .map(|item| match item.as_str() {
                    Some("full_attention") => Gemma4LayerType::FullAttention,
                    _ => Gemma4LayerType::SlidingAttention,
                })
                .collect()
        })
        .unwrap_or_else(|| {
            vec![Gemma4LayerType::SlidingAttention; num_hidden_layers]
        });

    // rope_parameters parsing.
    let rope_params = text_v.get("rope_parameters");
    let rope_local_base_freq = rope_params
        .and_then(|rp| rp.get("sliding_attention"))
        .and_then(|sa| sa.get("rope_theta"))
        .and_then(|x| x.as_f64())
        .unwrap_or(10_000.0);
    let partial_rotary_factor = rope_params
        .and_then(|rp| rp.get("full_attention"))
        .and_then(|fa| fa.get("partial_rotary_factor"))
        .and_then(|x| x.as_f64())
        .unwrap_or(0.25);

    let config = Gemma4TextConfig {
        vocab_size: get_usize("vocab_size", 262_144),
        hidden_size: get_usize("hidden_size", 2048),
        intermediate_size: get_usize("intermediate_size", 16_384),
        num_hidden_layers,
        num_attention_heads: get_usize("num_attention_heads", 8),
        num_key_value_heads: get_usize("num_key_value_heads", 4),
        num_global_key_value_heads: get_usize_opt("num_global_key_value_heads"),
        head_dim: get_usize("head_dim", 256),
        global_head_dim: get_usize("global_head_dim", 512),
        rms_norm_eps: get_f64("rms_norm_eps", 1e-6),
        rope_theta: get_f64("rope_theta", 1_000_000.0),
        rope_local_base_freq,
        partial_rotary_factor,
        sliding_window: get_usize("sliding_window", 4096),
        layer_types,
        attention_bias: get_bool("attention_bias", false),
        hidden_activation: GemmaActivation::GeluPytorchTanh,
        final_logit_softcapping: get_f64_opt("final_logit_softcapping"),
        tie_word_embeddings: get_bool("tie_word_embeddings", true),
    };

    Ok((config, eos_token_id, eot_token_id))
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
    top_p: Option<f32>,
    top_k: Option<usize>,
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

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};

use fuel::lazy_based::{
    BasedConfig, BasedLinearAttentionParams, BasedModel, BasedSlidingWindowParams, BasedWeights,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    #[value(name = "360m")]
    W360m,
    #[value(name = "1b")]
    W1b,
    #[value(name = "1b-50b")]
    W1b50b,
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
    #[arg(long, default_value_t = 0.0)]
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
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "refs/pr/1")]
    revision: String,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long, default_value = "360m")]
    which: Which,
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
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature, args.repeat_penalty, args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match args.model_id {
        Some(model_id) => model_id,
        None => match args.which {
            Which::W360m => "hazyresearch/based-360m".to_string(),
            Which::W1b => "hazyresearch/based-1b".to_string(),
            Which::W1b50b => "hazyresearch/based-1b-50b".to_string(),
        },
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let config_file = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("config.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => vec![repo.get("model.safetensors")?],
    };

    let tok_repo = api.model("openai-community/gpt2".to_string());
    let tokenizer_file = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => tok_repo.get("tokenizer.json")?,
    };

    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config_json = std::fs::read_to_string(&config_file)?;
    let cfg = based_config_from_hf_json_str(&config_json)?;

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = BasedWeights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = BasedModel {
        config: cfg.clone(),
        weights,
    };

    println!("loaded the model in {:?}", start.elapsed());

    let mut tok_stream = fuel_examples::token_output_stream::TokenOutputStream::new(tokenizer);
    let eos_token = tok_stream
        .get_token("<|endoftext|>")
        .ok_or_else(|| E::msg("cannot find the <|endoftext|> token"))?;

    let mut tokens = tok_stream
        .tokenizer()
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    for &t in tokens.iter() {
        if let Some(t) = tok_stream.next_token(t)? {
            print!("{t}");
        }
    }
    std::io::stdout().flush()?;

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
        if next_token == eos_token {
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

/// Parse a HuggingFace-style Based `config.json` into a [`BasedConfig`].
///
/// The eager `fuel_transformers::models::based::Config` uses serde renames
/// to map HF field names (`n_embd`, `n_inner`, `n_layer`, `n_head`,
/// `rotary_emb_base`) onto its struct fields; the lazy `BasedConfig` is a
/// pure data struct with no serde derives, so we translate by hand.
fn based_config_from_hf_json_str(json: &str) -> Result<BasedConfig> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let get_usize = |key: &str| -> Result<usize> {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .ok_or_else(|| E::msg(format!("config.json: missing/invalid field {key:?}")))
    };
    let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };
    let get_vec_usize = |key: &str| -> Result<Vec<usize>> {
        let arr = v
            .get(key)
            .and_then(|x| x.as_array())
            .ok_or_else(|| E::msg(format!("config.json: missing/invalid field {key:?}")))?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            let n = item
                .as_u64()
                .ok_or_else(|| E::msg(format!("config.json: {key:?} contains non-integer")))?;
            out.push(n as usize);
        }
        Ok(out)
    };

    let vocab_size = get_usize("vocab_size")?;
    let hidden_size = get_usize("n_embd")?;
    let intermediate_size = get_usize("n_inner")?;
    let num_hidden_layers = get_usize("n_layer")?;
    let num_attention_heads = get_usize("n_head")?;
    let layer_norm_epsilon = get_f64("layer_norm_epsilon")
        .ok_or_else(|| E::msg("config.json: missing/invalid field \"layer_norm_epsilon\""))?;
    let rope_theta = get_f64("rotary_emb_base").unwrap_or(10_000.0);

    let alt_mixer_layers = get_vec_usize("alt_mixer_layers")?;
    let alt_mixer_2_layers = get_vec_usize("alt_mixer_2_layers")?;

    // alt_mixer is the linear-attention block; alt_mixer_2 is sliding window.
    let la_node = v
        .get("alt_mixer")
        .ok_or_else(|| E::msg("config.json: missing field \"alt_mixer\""))?;
    let la_num_heads = la_node
        .get("num_heads")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| E::msg("config.json: alt_mixer.num_heads"))?;
    let la_feature_dim = la_node
        .get("feature_dim")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| E::msg("config.json: alt_mixer.feature_dim"))?;
    let la_input_dim = la_node
        .get("feature_map")
        .and_then(|fm| fm.get("input_dim"))
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(la_feature_dim);

    let swa_node = v
        .get("alt_mixer_2")
        .ok_or_else(|| E::msg("config.json: missing field \"alt_mixer_2\""))?;
    let swa_num_heads = swa_node
        .get("num_heads")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| E::msg("config.json: alt_mixer_2.num_heads"))?;
    let swa_window_size = swa_node
        .get("window_size")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| E::msg("config.json: alt_mixer_2.window_size"))?;

    Ok(BasedConfig {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        layer_norm_epsilon,
        rope_theta,
        alt_mixer_layers,
        alt_mixer_2_layers,
        la: BasedLinearAttentionParams {
            num_heads: la_num_heads,
            feature_dim: la_feature_dim,
            input_dim: la_input_dim,
        },
        swa: BasedSlidingWindowParams {
            num_heads: swa_num_heads,
            window_size: swa_window_size,
        },
    })
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

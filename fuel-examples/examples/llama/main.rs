// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};
use clap::{Parser, ValueEnum};

use fuel::lazy::{LlamaConfig, LlamaModel, LlamaWeights};
use fuel::lazy_llama_full::{
    build_llama3_model, Llama3Model, LlamaEosToks, LlamaFullConfig,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;

const EOS_TOKEN: &str = "</s>";
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    V1,
    V2,
    V3,
    V31,
    V3Instruct,
    V31Instruct,
    V32_1b,
    V32_1bInstruct,
    V32_3b,
    V32_3bInstruct,
    #[value(name = "solar-10.7b")]
    Solar10_7B,
    #[value(name = "tiny-llama-1.1b-chat")]
    TinyLlama1_1BChat,
    #[value(name = "SmoLM2-1.7B")]
    SmolLM2_1B,
    #[value(name = "SmoLM2-1.7B-Instruct")]
    SmolLM2_1BInstruct,
    #[value(name = "SmoLM2-360M")]
    SmolLM2_360M,
    #[value(name = "SmoLM2-360M-Instruct")]
    SmolLM2_360MInstruct,
    #[value(name = "SmoLM2-135M")]
    SmolLM2_135M,
    #[value(name = "SmoLM2-135M-Instruct")]
    SmolLM2_135MInstruct,
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
    #[arg(short = 'n', long, default_value_t = 10000)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long)]
    no_kv_cache: bool,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use different dtype than f16 (ignored in lazy port — always f32).
    #[arg(long)]
    dtype: Option<String>,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// The model size to use.
    #[arg(long, default_value = "v3")]
    which: Which,

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
    let _ = args.use_flash_attn;
    let _ = args.no_kv_cache;
    let _ = args.dtype;

    // Pick model repo id.
    let api = Api::new()?;
    let model_id = args.model_id.unwrap_or_else(|| {
        let str = match args.which {
            Which::V1 => "Narsil/amall-7b",
            Which::V2 => "meta-llama/Llama-2-7b-hf",
            Which::V3 => "meta-llama/Meta-Llama-3-8B",
            Which::V3Instruct => "meta-llama/Meta-Llama-3-8B-Instruct",
            Which::V31 => "meta-llama/Llama-3.1-8B",
            Which::V31Instruct => "meta-llama/Llama-3.1-8B-Instruct",
            Which::V32_1b => "meta-llama/Llama-3.2-1B",
            Which::V32_1bInstruct => "meta-llama/Llama-3.2-1B-Instruct",
            Which::V32_3b => "meta-llama/Llama-3.2-3B",
            Which::V32_3bInstruct => "meta-llama/Llama-3.2-3B-Instruct",
            Which::Solar10_7B => "upstage/SOLAR-10.7B-v1.0",
            Which::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Which::SmolLM2_135M => "HuggingFaceTB/SmolLM2-135M",
            Which::SmolLM2_135MInstruct => "HuggingFaceTB/SmolLM2-135M-Instruct",
            Which::SmolLM2_360M => "HuggingFaceTB/SmolLM2-360M",
            Which::SmolLM2_360MInstruct => "HuggingFaceTB/SmolLM2-360M-Instruct",
            Which::SmolLM2_1B => "HuggingFaceTB/SmolLM2-1.7B",
            Which::SmolLM2_1BInstruct => "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        };
        str.to_string()
    });
    println!("loading the model weights from {model_id}");
    let revision = args.revision.unwrap_or("main".to_string());
    let api = api.repo(Repo::with_revision(model_id.clone(), RepoType::Model, revision));

    let tokenizer_filename = api.get("tokenizer.json")?;

    // Load HF config.json — gives us rope_scaling + eos_token_id for
    // LLaMA-3.x. Falls back to LlamaConfig::from_hf_json_str for the
    // base unscaled fields.
    let config_path = api.get("config.json")?;
    let config_str = std::fs::read_to_string(&config_path)?;
    let full_cfg = LlamaFullConfig::from_hf_json_str(&config_str)
        .map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let lazy_cfg: LlamaConfig = full_cfg.to_lazy_config();

    // Resolve safetensors shard list (sharded → index.json, single → model.safetensors).
    let weight_paths: Vec<std::path::PathBuf> = match args.which {
        Which::V1
        | Which::V2
        | Which::V3
        | Which::V3Instruct
        | Which::V31
        | Which::V31Instruct
        | Which::V32_3b
        | Which::V32_3bInstruct
        | Which::Solar10_7B => {
            fuel_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?
        }
        Which::SmolLM2_360M
        | Which::SmolLM2_360MInstruct
        | Which::SmolLM2_135M
        | Which::SmolLM2_135MInstruct
        | Which::SmolLM2_1B
        | Which::SmolLM2_1BInstruct
        | Which::V32_1b
        | Which::V32_1bInstruct
        | Which::TinyLlama1_1BChat => {
            vec![api.get("model.safetensors")?]
        }
    };

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&weight_paths) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights: LlamaWeights = LlamaWeights::load_from_mmapped(&st, &lazy_cfg)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let llama_inner = LlamaModel { config: lazy_cfg.clone(), weights };
    let llama = build_llama3_model(&full_cfg, llama_inner.weights.clone());
    let llama = Llama3Model {
        inner: LlamaModel { config: lazy_cfg, weights: llama.inner.weights },
        rope_scaling: llama.rope_scaling,
        eos_token_id: llama.eos_token_id,
    };

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let eos_token_id: Option<LlamaEosToks> = full_cfg.eos_token_id.clone().or_else(|| {
        tokenizer
            .token_to_id(EOS_TOKEN)
            .map(LlamaEosToks::Single)
    });
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let mut tokenizer = fuel_examples::token_output_stream::TokenOutputStream::new(tokenizer);

    println!("starting the inference loop");
    print!("{prompt}");

    let mut start_gen = std::time::Instant::now();
    let mut token_generated: usize = 0;
    for index in 0..args.sample_len {
        if index == 1 {
            start_gen = std::time::Instant::now();
        }

        // The lazy port v1 does not maintain a KV cache between calls;
        // we re-feed the full prefix each step (O(n²) — fine for a
        // correctness demo, matches lazy_llama2c's CLI runner).
        let logits = llama
            .forward(&tokens, 0)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let logits_data = logits.realize_f32();
        // logits shape: [1, seq, vocab_size]. Take the last position.
        let vocab_size = llama.inner.config.vocab_size;
        let seq = tokens.len();
        let last_offset = (seq - 1) * vocab_size;
        let mut last_logits: Vec<f32> =
            logits_data[last_offset..last_offset + vocab_size].to_vec();

        // Repeat penalty.
        if args.repeat_penalty != 1.0 {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            apply_repeat_penalty(&mut last_logits, args.repeat_penalty, &tokens[start_at..]);
        }

        // Sampling: temperature + optional top-k / top-p.
        let next_token = sample(
            &last_logits,
            args.temperature as f32,
            args.top_k,
            args.top_p.map(|p| p as f32),
            args.seed.wrapping_add(index as u64),
        );

        token_generated += 1;
        tokens.push(next_token);

        match &eos_token_id {
            Some(LlamaEosToks::Single(eos_tok_id)) if next_token == *eos_tok_id => {
                break;
            }
            Some(LlamaEosToks::Multiple(eos_ids)) if eos_ids.contains(&next_token) => {
                break;
            }
            _ => (),
        }
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated.saturating_sub(1)) as f64 / dt.as_secs_f64(),
    );

    let _ = bail_if_invalid_dtype as fn(&str) -> Result<()>;
    Ok(())
}

fn bail_if_invalid_dtype(dtype: &str) -> Result<()> {
    match dtype {
        "f16" | "bf16" | "f32" => Ok(()),
        other => bail!("Unsupported dtype {other}"),
    }
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
        // Argmax.
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

    // Temperature scale + softmax.
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let inv_t = 1.0 / temperature.max(1e-6);
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&x| ((x - max_l) * inv_t).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum.max(1e-30);
    }

    // Indices sorted by descending probability.
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let mut keep_mask: Vec<bool> = vec![true; probs.len()];

    if let Some(k) = top_k {
        for &i in idx.iter().skip(k) {
            keep_mask[i] = false;
        }
    }
    if let Some(p_cut) = top_p {
        let mut cum = 0.0;
        for &i in &idx {
            if !keep_mask[i] {
                continue;
            }
            cum += probs[i];
            if cum > p_cut {
                // Keep this one and stop.
                continue;
            }
        }
        // Apply: drop tokens once cumulative > p_cut.
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
        // Fall back to uniform argmax.
        return 0;
    }

    // Deterministic LCG over `seed`.
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

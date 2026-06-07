#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use fuel::lazy_t5::{T5Activation, T5Config, T5Model, T5Weights};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, ValueEnum)]
enum Which {
    T5Base,
    T5Small,
    T5Large,
    T5_3B,
    Mt5Base,
    Mt5Small,
    Mt5Large,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model repository to use on the HuggingFace hub.
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    model_file: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    /// Enable decoding.
    #[arg(long)]
    decode: bool,

    // Enable/disable decoding.
    #[arg(long, default_value = "false")]
    disable_cache: bool,

    /// Use this prompt, otherwise compute sentence similarities.
    #[arg(long)]
    prompt: Option<String>,

    /// If set along with --decode, will use this prompt to initialize the decoder.
    #[arg(long)]
    decoder_prompt: Option<String>,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

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

    /// The model to be used.
    #[arg(long, default_value = "t5-small")]
    which: Which,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
}

/// Parsed HF `config.json` for T5 / mT5. Mirrors the fields used by the
/// eager `t5::Config`. The lazy `T5Config` only needs a strict subset, so
/// we hold on to the auxiliary fields (`pad_token_id`, `eos_token_id`,
/// `decoder_start_token_id`) here.
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
    let relative_attention_max_distance = get_usize_opt("relative_attention_max_distance").unwrap_or(128);
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
    weights_filename: Vec<PathBuf>,
}

impl T5ModelBuilder {
    pub fn load(args: &Args) -> Result<(Self, Tokenizer)> {
        let (default_model, default_revision) = match args.which {
            Which::T5Base => ("t5-base", "main"),
            Which::T5Small => ("t5-small", "refs/pr/15"),
            Which::T5Large => ("t5-large", "main"),
            Which::T5_3B => ("t5-3b", "main"),
            Which::Mt5Base => ("google/mt5-base", "refs/pr/5"),
            Which::Mt5Small => ("google/mt5-small", "refs/pr/6"),
            Which::Mt5Large => ("google/mt5-large", "refs/pr/2"),
        };
        let default_model = default_model.to_string();
        let default_revision = default_revision.to_string();
        let (model_id, revision) = match (args.model_id.to_owned(), args.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id.clone(), RepoType::Model, revision);
        let api = Api::new()?;
        let repo = api.repo(repo);
        let config_filename = match &args.config_file {
            None => repo.get("config.json")?,
            Some(f) => f.into(),
        };
        let tokenizer_filename = match &args.tokenizer_file {
            None => match args.which {
                Which::Mt5Base => api
                    .model("lmz/mt5-tokenizers".into())
                    .get("mt5-base.tokenizer.json")?,
                Which::Mt5Small => api
                    .model("lmz/mt5-tokenizers".into())
                    .get("mt5-small.tokenizer.json")?,
                Which::Mt5Large => api
                    .model("lmz/mt5-tokenizers".into())
                    .get("mt5-large.tokenizer.json")?,
                _ => repo.get("tokenizer.json")?,
            },
            Some(f) => f.into(),
        };
        let weights_filename = match &args.model_file {
            Some(f) => f.split(',').map(|v| v.into()).collect::<Vec<_>>(),
            None => {
                if model_id == "google/flan-t5-xxl" || model_id == "google/flan-ul2" {
                    fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
                } else {
                    vec![repo.get("model.safetensors")?]
                }
            }
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

    pub fn build(&self) -> Result<T5Model> {
        let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&self.weights_filename) }
            .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
        let weights = T5Weights::load_from_mmapped(&st, &self.loaded.cfg)
            .map_err(|e| E::msg(format!("load t5 weights: {e}")))?;
        Ok(T5Model {
            config: self.loaded.cfg.clone(),
            weights,
        })
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

    // `--cpu` and `--disable-cache` retained for CLI compatibility; the lazy
    // port runs on the graph executor (no per-step KV cache yet) so both are
    // effectively no-ops.
    let _ = args.cpu;
    let _ = args.disable_cache;

    let (builder, mut tokenizer) = T5ModelBuilder::load(&args)?;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    match args.prompt.clone() {
        Some(prompt) => {
            let tokens: Vec<u32> = tokenizer
                .encode(prompt, true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            if !args.decode {
                let model = builder.build()?;
                let start = std::time::Instant::now();
                let ys = model
                    .forward_encoder(&tokens)
                    .map_err(|e| E::msg(format!("encoder forward: {e}")))?;
                let data = ys.realize_f32();
                println!("encoder output shape: {:?}", ys.shape().dims());
                println!("first 8 values: {:?}", &data[..data.len().min(8)]);
                println!("Took {:?}", start.elapsed());
            } else {
                let model = builder.build()?;
                let decoder_start = builder
                    .loaded
                    .decoder_start_token_id
                    .unwrap_or(builder.loaded.pad_token_id);
                let mut output_token_ids: Vec<u32> = vec![decoder_start];
                if let Some(decoder_prompt) = &args.decoder_prompt {
                    print!("{decoder_prompt}");
                    output_token_ids.extend(
                        tokenizer
                            .encode(decoder_prompt.to_string(), false)
                            .map_err(E::msg)?
                            .get_ids()
                            .to_vec(),
                    );
                }
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
                    // Lazy port has no KV cache — each step re-runs the
                    // decoder over the full target prefix.
                    let logits = model
                        .forward_decoder(&output_token_ids, &encoder_out)
                        .map_err(|e| E::msg(format!("decoder forward: {e}")))?;
                    let logits_data = logits.realize_f32();
                    let tgt_len = output_token_ids.len();
                    let last_off = (tgt_len - 1) * vocab_size;
                    let mut last_logits: Vec<f32> =
                        logits_data[last_off..last_off + vocab_size].to_vec();
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
            }
        }
        None => {
            let model = builder.build()?;
            let sentences = [
                "The cat sits outside",
                "A man is playing guitar",
                "I love pasta",
                "The new movie is awesome",
                "The cat plays in the garden",
                "A woman watches TV",
                "The new movie is so great",
                "Do you like pizza?",
            ];
            let n_sentences = sentences.len();
            let d_model = builder.loaded.cfg.d_model;
            // Each entry is a `[d_model]` pooled embedding for one sentence.
            let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(n_sentences);
            for sentence in sentences {
                let tokens: Vec<u32> = tokenizer
                    .encode(sentence, true)
                    .map_err(E::msg)?
                    .get_ids()
                    .to_vec();
                let embeddings = model
                    .forward_encoder(&tokens)
                    .map_err(|e| E::msg(format!("encoder forward: {e}")))?;
                let dims = embeddings.shape().dims().to_vec();
                println!("generated embeddings {:?}", dims);
                // Shape is (1, n_tokens, d_model); mean-pool across token axis.
                let data = embeddings.realize_f32();
                let n_tokens = dims[1];
                let mut pooled = vec![0.0_f32; d_model];
                for t in 0..n_tokens {
                    for h in 0..d_model {
                        pooled[h] += data[t * d_model + h];
                    }
                }
                let inv = 1.0_f32 / (n_tokens as f32);
                for v in &mut pooled {
                    *v *= inv;
                }
                if args.normalize_embeddings {
                    normalize_l2(&mut pooled);
                }
                println!("pooled embeddings ({} dims)", pooled.len());
                all_embeddings.push(pooled);
            }

            let mut similarities = vec![];
            for (i, e_i) in all_embeddings.iter().enumerate() {
                for (j, e_j) in all_embeddings
                    .iter()
                    .enumerate()
                    .take(n_sentences)
                    .skip(i + 1)
                {
                    let sum_ij: f32 = e_i.iter().zip(e_j.iter()).map(|(a, b)| a * b).sum();
                    let sum_i2: f32 = e_i.iter().map(|a| a * a).sum();
                    let sum_j2: f32 = e_j.iter().map(|a| a * a).sum();
                    let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
                    similarities.push((cosine_similarity, i, j))
                }
            }
            similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
            for &(score, i, j) in similarities[..5].iter() {
                println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
            }
        }
    }
    Ok(())
}

fn normalize_l2(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v {
            *x *= inv;
        }
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

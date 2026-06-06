#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use std::io::Write;
use std::sync::Arc;

use fuel::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype, WeightStorage};
use fuel::lazy_mamba2::{
    Mamba2Config, Mamba2LayerWeights, Mamba2Model, Mamba2Weights, D_CONV,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum Which {
    Mamba2_130m,
    Mamba2_370m,
    Mamba2_780m,
    Mamba2_1_3b,
    Mamba2_2_7b,
}

impl std::fmt::Display for Which {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Self::Mamba2_130m => "AntonV/mamba2-130m-hf",
            Self::Mamba2_370m => "AntonV/mamba2-370m-hf",
            Self::Mamba2_780m => "AntonV/mamba2-780m-hf",
            Self::Mamba2_1_3b => "AntonV/mamba2-1.3b-hf",
            Self::Mamba2_2_7b => "AntonV/mamba2-2.7b-hf",
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
    #[arg(long, short = 'n', default_value_t = 5000)]
    sample_len: usize,

    #[arg(long, default_value = "mamba2-130m")]
    which: Which,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long, default_value = "f32")]
    dtype: String,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Use chunked prefill for processing the initial prompt.
    #[arg(long)]
    use_prefill: bool,

    /// Chunk size for prefill (default 256).
    #[arg(long, default_value_t = 256)]
    chunk_size: usize,
}

fn mamba2_config_from_hf_json_str(json: &str, chunk_size: usize) -> Result<Mamba2Config> {
    // Config contains `Infinity` which is not valid JSON, replace.
    let json = json.replace("Infinity", "1e30");
    let v: serde_json::Value =
        serde_json::from_str(&json).map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let get_usize = |key: &str, alias: &str, default: Option<usize>| -> Result<usize> {
        if let Some(x) = v.get(key).and_then(|x| x.as_u64()) {
            Ok(x as usize)
        } else if let Some(x) = v.get(alias).and_then(|x| x.as_u64()) {
            Ok(x as usize)
        } else if let Some(d) = default {
            Ok(d)
        } else {
            Err(E::msg(format!(
                "config.json: missing {key:?} (alias {alias:?})"
            )))
        }
    };
    let d_model = get_usize("d_model", "hidden_size", None)?;
    let n_layer = get_usize("n_layer", "num_hidden_layers", None)?;
    let vocab_size = get_usize("vocab_size", "vocab_size", None)?;
    let d_state = get_usize("d_state", "state_size", Some(128))?;
    let expand = get_usize("expand", "expand", Some(2))?;
    let head_dim = get_usize("head_dim", "headdim", Some(64))?;
    let ngroups = get_usize("ngroups", "n_groups", Some(1))?;
    let pad_vocab_size_multiple = v
        .get("pad_vocab_size_multiple")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(16);
    let rms_norm_eps = v
        .get("rms_norm_eps")
        .and_then(|x| x.as_f64())
        .or_else(|| v.get("layer_norm_epsilon").and_then(|x| x.as_f64()))
        .unwrap_or(1e-5);
    Ok(Mamba2Config {
        d_model,
        n_layer,
        vocab_size,
        d_state,
        expand,
        head_dim,
        ngroups,
        pad_vocab_size_multiple,
        chunk_size,
        rms_norm_eps,
    })
}

fn load_mamba2_weights(
    st: &fuel::safetensors::MmapedSafetensors,
    cfg: &Mamba2Config,
) -> Result<Mamba2Weights> {
    let vocab_padded = cfg.vocab_size();
    let d_model = cfg.d_model;
    let d_inner = cfg.d_inner();
    let d_xbc = cfg.d_xbc();
    let n_heads = cfg.n_heads();
    let proj_size = d_inner + d_xbc + n_heads;

    // HF mamba2 uses "backbone.embeddings.weight" (plural).
    let token_embedding = load_tensor_as_f32(st, "backbone.embeddings.weight")
        .or_else(|_| load_tensor_as_f32(st, "backbone.embedding.weight"))?;

    let mut layers: Vec<Mamba2LayerWeights> = Vec::with_capacity(cfg.n_layer);
    for i in 0..cfg.n_layer {
        let base = format!("backbone.layers.{i}");
        let norm_gain = load_tensor_as_f32(st, &format!("{base}.norm.weight"))?;
        let in_proj = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.mixer.in_proj.weight"),
            proj_size,
            d_model,
        )?;
        let conv1d_weight = load_tensor_as_f32(st, &format!("{base}.mixer.conv1d.weight"))?;
        let conv1d_bias = load_tensor_as_f32(st, &format!("{base}.mixer.conv1d.bias"))?;
        let a_log = load_tensor_as_f32(st, &format!("{base}.mixer.A_log"))?;
        let d = load_tensor_as_f32(st, &format!("{base}.mixer.D"))?;
        let dt_bias = load_tensor_as_f32(st, &format!("{base}.mixer.dt_bias"))?;
        let out_norm_gain = load_tensor_as_f32(st, &format!("{base}.mixer.norm.weight"))?;
        let out_proj = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.mixer.out_proj.weight"),
            d_model,
            d_inner,
        )?;
        if conv1d_weight.len() != d_xbc * D_CONV {
            return Err(E::msg(format!(
                "{base}.mixer.conv1d.weight: {} elements, expected {} (d_xbc={d_xbc} × D_CONV={D_CONV})",
                conv1d_weight.len(),
                d_xbc * D_CONV
            )));
        }
        layers.push(Mamba2LayerWeights {
            norm_gain: Arc::from(norm_gain),
            in_proj,
            conv1d_weight: Arc::from(conv1d_weight),
            conv1d_bias: Arc::from(conv1d_bias),
            a_log: Arc::from(a_log),
            d: Arc::from(d),
            dt_bias: Arc::from(dt_bias),
            out_norm_gain: Arc::from(out_norm_gain),
            out_proj,
        });
    }
    let final_norm_gain = load_tensor_as_f32(st, "backbone.norm_f.weight")?;
    let output: WeightStorage = match load_transposed_matrix_preserve_dtype(
        st,
        "lm_head.weight",
        vocab_padded,
        d_model,
    ) {
        Ok(w) => w,
        Err(_) => {
            let mut transposed = vec![0.0_f32; d_model * vocab_padded];
            for i in 0..vocab_padded {
                for j in 0..d_model {
                    transposed[j * vocab_padded + i] = token_embedding[i * d_model + j];
                }
            }
            WeightStorage::F32(Arc::from(transposed))
        }
    };
    Ok(Mamba2Weights {
        token_embedding: Arc::from(token_embedding),
        layers,
        final_norm_gain: Arc::from(final_norm_gain),
        output,
    })
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
    let _ = args.cpu;
    let _ = args.dtype;
    let _ = args.use_prefill;
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
        .unwrap_or_else(|| args.which.model_id().to_string());
    let repo = api.repo(Repo::new(model_id.clone(), RepoType::Model));
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
        None => vec![repo.get("model.safetensors")?],
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config_json = std::fs::read_to_string(&config_filename)?;
    let config = mamba2_config_from_hf_json_str(&config_json, args.chunk_size)?;
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights =
        load_mamba2_weights(&st, &config).map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = Mamba2Model {
        config: config.clone(),
        weights,
    };
    println!("loaded the model in {:?}", start.elapsed());

    let mut tok_stream = fuel_examples::token_output_stream::TokenOutputStream::new(tokenizer);
    let eos_token = tok_stream
        .get_token("<|endoftext|>")
        .ok_or_else(|| anyhow::anyhow!("cannot find the <|endoftext|> token"))?;

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

    let vocab_padded = config.vocab_size();
    let chunk_size = config.chunk_size;
    let mut generated_tokens: usize = 0;
    let start_gen = std::time::Instant::now();
    for index in 0..args.sample_len {
        // Pad tokens to the next multiple of chunk_size with the EOS token
        // — the lazy mamba2 forward requires `seq % chunk_size == 0`.
        let real_seq = tokens.len();
        let padded_len = real_seq.div_ceil(chunk_size) * chunk_size;
        let mut padded = tokens.clone();
        padded.resize(padded_len, eos_token);
        let logits = model
            .forward(&padded)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let logits_data = logits.realize_f32();
        // The "real" last token is at index real_seq - 1.
        let last_off = (real_seq - 1) * vocab_padded;
        let mut last_logits: Vec<f32> = logits_data[last_off..last_off + vocab_padded].to_vec();
        if args.repeat_penalty != 1.0 {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            apply_repeat_penalty(&mut last_logits, args.repeat_penalty, &tokens[start_at..]);
        }
        let next_token = sample(
            &last_logits,
            args.temperature.unwrap_or(0.) as f32,
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

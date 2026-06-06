#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use std::io::Write;
use std::sync::Arc;

use fuel::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype, WeightStorage};
use fuel::lazy_mamba::{MambaConfig, MambaLayerWeights, MambaModel, MambaWeights, D_CONV, D_STATE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum Which {
    Mamba130m,
    Mamba370m,
    Mamba790m,
    Mamba1_4b,
    Mamba2_8b,
    Mamba2_8bSlimPj,
}

impl std::fmt::Display for Which {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Self::Mamba130m => "state-spaces/mamba-130m",
            Self::Mamba370m => "state-spaces/mamba-370m",
            Self::Mamba790m => "state-spaces/mamba-790m",
            Self::Mamba1_4b => "state-spaces/mamba-1.4b",
            Self::Mamba2_8b => "state-spaces/mamba-2.8b",
            Self::Mamba2_8bSlimPj => "state-spaces/mamba-2.8b-slimpj'",
        }
    }

    fn revision(&self) -> &'static str {
        match self {
            Self::Mamba130m
            | Self::Mamba370m
            | Self::Mamba790m
            | Self::Mamba1_4b
            | Self::Mamba2_8bSlimPj => "refs/pr/1",
            Self::Mamba2_8b => "refs/pr/4",
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

    #[arg(long, default_value = "mamba130m")]
    which: Which,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

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
}

fn mamba_config_from_hf_json_str(json: &str) -> Result<MambaConfig> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let get_usize = |key: &str| -> Result<usize> {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .ok_or_else(|| E::msg(format!("config.json: missing/invalid field {key:?}")))
    };
    let d_model = get_usize("d_model")?;
    let n_layer = get_usize("n_layer")?;
    let vocab_size = get_usize("vocab_size")?;
    let pad_vocab_size_multiple = v
        .get("pad_vocab_size_multiple")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(8);
    let rms_norm_eps = v
        .get("rms_norm_eps")
        .and_then(|x| x.as_f64())
        .unwrap_or(1e-5);
    Ok(MambaConfig {
        d_model,
        n_layer,
        vocab_size,
        pad_vocab_size_multiple,
        rms_norm_eps,
    })
}

fn load_mamba_weights(
    st: &fuel::safetensors::MmapedSafetensors,
    cfg: &MambaConfig,
) -> Result<MambaWeights> {
    let vocab_padded = cfg.vocab_size();
    let d_model = cfg.d_model;
    let d_inner = cfg.d_inner();
    let dt_rank = cfg.dt_rank();

    let token_embedding = load_tensor_as_f32(st, "backbone.embedding.weight")?;
    if token_embedding.len() != vocab_padded * d_model {
        return Err(E::msg(format!(
            "backbone.embedding.weight: {} elements, expected {} ({}×{})",
            token_embedding.len(),
            vocab_padded * d_model,
            vocab_padded,
            d_model
        )));
    }

    let mut layers: Vec<MambaLayerWeights> = Vec::with_capacity(cfg.n_layer);
    for i in 0..cfg.n_layer {
        let base = format!("backbone.layers.{i}");
        let norm_gain = load_tensor_as_f32(st, &format!("{base}.norm.weight"))?;
        // in_proj: HF [2*d_inner, d_model] → transposed to [d_model, 2*d_inner].
        let in_proj = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.mixer.in_proj.weight"),
            2 * d_inner,
            d_model,
        )?;
        // Conv1d weight/bias are kept as raw shapes — no transpose.
        let conv1d_weight = load_tensor_as_f32(st, &format!("{base}.mixer.conv1d.weight"))?;
        let conv1d_bias = load_tensor_as_f32(st, &format!("{base}.mixer.conv1d.bias"))?;
        // x_proj: HF [dt_rank + 2*D_STATE, d_inner] → transposed to [d_inner, dt_rank + 2*D_STATE].
        let x_proj = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.mixer.x_proj.weight"),
            dt_rank + 2 * D_STATE,
            d_inner,
        )?;
        // dt_proj: HF [d_inner, dt_rank] → transposed to [dt_rank, d_inner].
        let dt_proj = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.mixer.dt_proj.weight"),
            d_inner,
            dt_rank,
        )?;
        let dt_proj_bias = load_tensor_as_f32(st, &format!("{base}.mixer.dt_proj.bias"))?;
        let a_log = load_tensor_as_f32(st, &format!("{base}.mixer.A_log"))?;
        let d = load_tensor_as_f32(st, &format!("{base}.mixer.D"))?;
        // out_proj: HF [d_model, d_inner] → transposed to [d_inner, d_model].
        let out_proj = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.mixer.out_proj.weight"),
            d_model,
            d_inner,
        )?;
        if conv1d_weight.len() != d_inner * D_CONV {
            return Err(E::msg(format!(
                "{base}.mixer.conv1d.weight: {} elements, expected {} (d_inner={d_inner} × D_CONV={D_CONV})",
                conv1d_weight.len(),
                d_inner * D_CONV
            )));
        }
        layers.push(MambaLayerWeights {
            norm_gain: Arc::from(norm_gain),
            in_proj,
            conv1d_weight: Arc::from(conv1d_weight),
            conv1d_bias: Arc::from(conv1d_bias),
            x_proj,
            dt_proj,
            dt_proj_bias: Arc::from(dt_proj_bias),
            a_log: Arc::from(a_log),
            d: Arc::from(d),
            out_proj,
        });
    }
    let final_norm_gain = load_tensor_as_f32(st, "backbone.norm_f.weight")?;
    // lm_head is tied to embedding in Mamba — transpose embedding for matmul.
    // Try loading lm_head.weight first; fall back to tied.
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

    Ok(MambaWeights {
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
    let repo = api.repo(Repo::with_revision(
        args.model_id
            .unwrap_or_else(|| args.which.model_id().to_string()),
        RepoType::Model,
        args.revision
            .unwrap_or_else(|| args.which.revision().to_string()),
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => api
            .model("EleutherAI/gpt-neox-20b".to_string())
            .get("tokenizer.json")?,
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
    let config = mamba_config_from_hf_json_str(&config_json)?;
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights =
        load_mamba_weights(&st, &config).map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = MambaModel {
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
    let mut generated_tokens: usize = 0;
    let start_gen = std::time::Instant::now();
    for index in 0..args.sample_len {
        // Prefill-only: re-run the whole sequence each step. The lazy
        // mamba module's selective scan is prefill-only — no resume-from-
        // state op yet — so we mirror the eager binary's per-step pattern,
        // which already recomputes from scratch each iteration in the
        // mamba-minimal version.
        let logits = model
            .forward(&tokens)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let logits_data = logits.realize_f32();
        let seq = tokens.len();
        let last_off = (seq - 1) * vocab_padded;
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

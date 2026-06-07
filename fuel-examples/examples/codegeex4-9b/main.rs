#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use fuel::lazy_chatglm::{ChatGlmConfig, ChatGlmModel, ChatGlmNorm, ChatGlmWeights};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(name = "cache", short)]
    cache_path: Option<String>,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Display the token for the specified prompt.
    #[arg(long)]
    prompt: String,

    /// Display the tokens for the specified prompt and outputs.
    #[arg(long)]
    verbose: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.95)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value_t = 0.8)]
    top_p: f64,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 8192)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    weight_path: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.2)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        fuel::utils::with_avx(),
        fuel::utils::with_neon(),
        fuel::utils::with_simd128(),
        fuel::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature, args.repeat_penalty, args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = match args.cache_path.as_ref() {
        None => Api::new()?,
        Some(path) => {
            hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(path.to_string().into()))
                .build()
                .map_err(anyhow::Error::msg)?
        }
    };
    let model_id = match args.model_id {
        Some(model_id) => model_id.to_string(),
        None => "THUDM/codegeex4-all-9b".to_string(),
    };
    let revision = match args.revision {
        Some(rev) => rev.to_string(),
        None => "main".to_string(),
    };
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let tokenizer_filename = match args.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => api
            .model("THUDM/codegeex4-all-9b".to_string())
            .get("tokenizer.json")
            .map_err(anyhow::Error::msg)?,
    };
    let config_filename = match &args.weight_path {
        Some(path) => std::path::Path::new(path).join("config.json"),
        None => repo.get("config.json")?,
    };

    let filenames = match &args.weight_path {
        Some(path) => {
            fuel_examples::hub_load_local_safetensors(path, "model.safetensors.index.json")?
        }
        _ => fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config_json = std::fs::read_to_string(&config_filename)?;
    let cfg: ChatGlmConfig = chatglm_config_from_hf_json_str(&config_json)?;

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap: {e}")))?;
    let weights = ChatGlmWeights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("weights: {e}")))?;
    let model = ChatGlmModel { config: cfg.clone(), weights };

    println!("loaded the model in {:?}", start.elapsed());

    let eos_token = match tokenizer.get_vocab(true).get("<|endoftext|>") {
        Some(token) => *token,
        None => panic!("cannot find the endoftext token"),
    };

    let encoded = tokenizer.encode(args.prompt.clone(), true).map_err(E::msg)?;
    if encoded.is_empty() {
        panic!("Empty prompts are not supported in the chatglm model.")
    }
    if args.verbose {
        for (token, id) in encoded.get_tokens().iter().zip(encoded.get_ids().iter()) {
            let token = token.replace('\u{2581}', " ").replace("<0x0A>", "\n");
            println!("{id:7} -> '{token}'");
        }
    }

    let mut tokens = encoded.get_ids().to_vec();
    let mut generated = 0usize;

    print!("{}", args.prompt);
    std::io::stdout().flush()?;
    let start_gen = std::time::Instant::now();
    println!("\n start_gen");
    println!("samplelen {}", args.sample_len);
    let mut result: Vec<String> = vec![];
    for index in 0..args.sample_len {
        let logits = model
            .forward(&tokens, 0)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let data = logits.realize_f32();
        let v = cfg.padded_vocab_size;
        let seq = tokens.len();
        let last_off = (seq - 1) * v;
        let mut last: Vec<f32> = data[last_off..last_off + v].to_vec();
        if args.repeat_penalty != 1.0 {
            let s = tokens.len().saturating_sub(args.repeat_last_n);
            apply_repeat_penalty(&mut last, args.repeat_penalty, &tokens[s..]);
        }
        let next = sample(
            &last,
            args.temperature as f32,
            None,
            Some(args.top_p as f32),
            args.seed.wrapping_add(index as u64),
        );
        tokens.push(next);
        generated += 1;
        if next == eos_token {
            break;
        }
        let token = tokenizer.decode(&[next], true).map_err(E::msg)?;
        if args.verbose {
            println!(
                "[Count: {}] [Raw Token: {next}] [Decode Token: {token}]",
                index + 1
            );
        }
        result.push(token);
        std::io::stdout().flush()?;
    }
    let dt = start_gen.elapsed();
    println!(
        "\n{generated} tokens generated ({:.2} token/s)",
        generated as f64 / dt.as_secs_f64()
    );
    println!("Result:");
    for tokens in result {
        print!("{tokens}");
    }
    Ok(())
}

fn chatglm_config_from_hf_json_str(json: &str) -> Result<ChatGlmConfig> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| E::msg(format!("parse config: {e}")))?;
    let get_usize = |k: &str| -> Option<usize> {
        v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_f64 = |k: &str| -> Option<f64> { v.get(k).and_then(|x| x.as_f64()) };
    let get_bool = |k: &str| -> Option<bool> { v.get(k).and_then(|x| x.as_bool()) };
    let preset = ChatGlmConfig::codegeex4();
    let rmsnorm = get_bool("rmsnorm").unwrap_or(matches!(preset.norm_kind, ChatGlmNorm::Rms));
    let rope_ratio = get_usize("rope_ratio").unwrap_or(500);
    Ok(ChatGlmConfig {
        num_layers: get_usize("num_layers").unwrap_or(preset.num_layers),
        padded_vocab_size: get_usize("padded_vocab_size").unwrap_or(preset.padded_vocab_size),
        hidden_size: get_usize("hidden_size").unwrap_or(preset.hidden_size),
        ffn_hidden_size: get_usize("ffn_hidden_size").unwrap_or(preset.ffn_hidden_size),
        kv_channels: get_usize("kv_channels").unwrap_or(preset.kv_channels),
        num_attention_heads: get_usize("num_attention_heads").unwrap_or(preset.num_attention_heads),
        multi_query_group_num: get_usize("multi_query_group_num")
            .unwrap_or(preset.multi_query_group_num),
        seq_length: get_usize("seq_length").unwrap_or(preset.seq_length),
        layernorm_epsilon: get_f64("layernorm_epsilon").unwrap_or(preset.layernorm_epsilon),
        norm_kind: if rmsnorm { ChatGlmNorm::Rms } else { ChatGlmNorm::Layer },
        add_qkv_bias: get_bool("add_qkv_bias").unwrap_or(preset.add_qkv_bias),
        add_bias_linear: get_bool("add_bias_linear").unwrap_or(preset.add_bias_linear),
        apply_residual_connection_post_layernorm: get_bool(
            "apply_residual_connection_post_layernorm",
        )
        .unwrap_or(preset.apply_residual_connection_post_layernorm),
        post_layer_norm: get_bool("post_layer_norm").unwrap_or(preset.post_layer_norm),
        rope_base: 10_000.0 * rope_ratio as f64,
    })
}

fn apply_repeat_penalty(logits: &mut [f32], p: f32, ctx: &[u32]) {
    let mut seen = std::collections::HashSet::new();
    for &t in ctx {
        if !seen.insert(t) {
            continue;
        }
        let i = t as usize;
        if i < logits.len() {
            let v = logits[i];
            logits[i] = if v >= 0.0 { v / p } else { v * p };
        }
    }
}

fn sample(
    logits: &[f32],
    temp: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: u64,
) -> u32 {
    if temp <= 0.0 {
        let mut bi = 0usize;
        let mut b = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > b {
                b = v;
                bi = i;
            }
        }
        return bi as u32;
    }
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let inv = 1.0 / temp.max(1e-6);
    let mut probs: Vec<f32> = logits.iter().map(|&x| ((x - m) * inv).exp()).collect();
    let s: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= s.max(1e-30);
    }
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let mut keep = vec![true; probs.len()];
    if let Some(k) = top_k {
        for &i in idx.iter().skip(k) {
            keep[i] = false;
        }
    }
    if let Some(pc) = top_p {
        let mut c = 0.0;
        let mut allow = true;
        for &i in &idx {
            if !keep[i] {
                continue;
            }
            if !allow {
                keep[i] = false;
                continue;
            }
            c += probs[i];
            if c >= pc {
                allow = false;
            }
        }
    }
    let mut f: Vec<f32> = probs
        .iter()
        .enumerate()
        .map(|(i, p)| if keep[i] { *p } else { 0.0 })
        .collect();
    let ss: f32 = f.iter().sum();
    if ss > 0.0 {
        for v in &mut f {
            *v /= ss;
        }
    } else {
        return 0;
    }
    let mut st = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    st ^= st >> 33;
    st = st.wrapping_mul(0xff51_afd7_ed55_8ccd);
    st ^= st >> 33;
    let r = (st as f32) / (u64::MAX as f32);
    let mut c = 0.0;
    for (i, p) in f.iter().enumerate() {
        c += *p;
        if r <= c {
            return i as u32;
        }
    }
    (f.len() - 1) as u32
}

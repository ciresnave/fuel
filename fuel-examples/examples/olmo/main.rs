#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use std::io::Write;
use std::sync::Arc;

use fuel::lazy::{
    load_tensor_as_f32, load_transposed_matrix_preserve_dtype, WeightStorage,
};
use fuel::lazy_olmo::{OlmoConfig, OlmoLayerWeights, OlmoModel, OlmoWeights};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum Which {
    #[value(name = "1b")]
    W1b,
    #[value(name = "7b")]
    W7b,
    #[value(name = "7b-twin-2t")]
    W7bTwin2T,
    #[value(name = "1.7-7b")]
    V1_7W7b,
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

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 1000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long, default_value = "1b")]
    model: Which,

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

    let _device = fuel_examples::device(args.cpu)?;
    let api = Api::new()?;
    let model_id = match args.model_id.clone() {
        Some(model_id) => model_id,
        None => match args.model {
            Which::W1b => "allenai/OLMo-1B-hf".to_string(),
            Which::W7b => "allenai/OLMo-7B-hf".to_string(),
            Which::W7bTwin2T => "allenai/OLMo-7B-Twin-2T-hf".to_string(),
            Which::V1_7W7b => "allenai/OLMo-1.7-7B-hf".to_string(),
        },
    };

    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision.clone(),
    ));
    let tokenizer_filename = match args.tokenizer_file.as_ref() {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames: Vec<std::path::PathBuf> = match args.weight_files.as_ref() {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => match args.model {
            Which::W1b => vec![repo.get("model.safetensors")?],
            _ => fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
        },
    };

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let config_file = repo.get("config.json")?;
    let config_json = std::fs::read_to_string(&config_file)?;
    let cfg = olmo_config_from_hf_json_str(&config_json)?;
    let eos_token_id = parse_eos_token_id(&config_json);

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = load_olmo_weights(&st, &cfg)?;
    let model = OlmoModel { config: cfg.clone(), weights };

    let mut tok_stream = fuel_examples::token_output_stream::TokenOutputStream::new(tokenizer);
    print!("{}", args.prompt);
    std::io::stdout().flush()?;
    let mut tokens = tok_stream
        .tokenizer()
        .encode(args.prompt.clone(), true)
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
            args.temperature.map(|t| t as f32).unwrap_or(0.0),
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

fn load_olmo_weights(
    st: &fuel::safetensors::MmapedSafetensors,
    cfg: &OlmoConfig,
) -> Result<OlmoWeights> {
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    let token_embedding: Arc<[f32]> = Arc::from(
        load_tensor_as_f32(st, "model.embed_tokens.weight")
            .map_err(|e| E::msg(format!("embed_tokens: {e}")))?,
    );
    let mut layers: Vec<OlmoLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let attn_q = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.self_attn.q_proj.weight"),
            cfg.hidden_size, cfg.hidden_size,
        ).map_err(|e| E::msg(format!("q_proj L{i}: {e}")))?;
        let attn_k = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.self_attn.k_proj.weight"),
            kv_dim, cfg.hidden_size,
        ).map_err(|e| E::msg(format!("k_proj L{i}: {e}")))?;
        let attn_v = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.self_attn.v_proj.weight"),
            kv_dim, cfg.hidden_size,
        ).map_err(|e| E::msg(format!("v_proj L{i}: {e}")))?;
        let attn_o = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.self_attn.o_proj.weight"),
            cfg.hidden_size, cfg.hidden_size,
        ).map_err(|e| E::msg(format!("o_proj L{i}: {e}")))?;
        let ffn_gate = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.mlp.gate_proj.weight"),
            cfg.intermediate_size, cfg.hidden_size,
        ).map_err(|e| E::msg(format!("gate L{i}: {e}")))?;
        let ffn_up = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.mlp.up_proj.weight"),
            cfg.intermediate_size, cfg.hidden_size,
        ).map_err(|e| E::msg(format!("up L{i}: {e}")))?;
        let ffn_down = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.mlp.down_proj.weight"),
            cfg.hidden_size, cfg.intermediate_size,
        ).map_err(|e| E::msg(format!("down L{i}: {e}")))?;
        let attn_norm_gain: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("model.layers.{i}.input_layernorm.weight"))
                .map_err(|e| E::msg(format!("input_ln L{i}: {e}")))?,
        );
        let ffn_norm_gain: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("model.layers.{i}.post_attention_layernorm.weight"))
                .map_err(|e| E::msg(format!("post_attention_ln L{i}: {e}")))?,
        );
        let attn_q_bias = load_tensor_as_f32(st, &format!("model.layers.{i}.self_attn.q_proj.bias"))
            .ok().map(Arc::from);
        let attn_k_bias = load_tensor_as_f32(st, &format!("model.layers.{i}.self_attn.k_proj.bias"))
            .ok().map(Arc::from);
        let attn_v_bias = load_tensor_as_f32(st, &format!("model.layers.{i}.self_attn.v_proj.bias"))
            .ok().map(Arc::from);
        let attn_o_bias = load_tensor_as_f32(st, &format!("model.layers.{i}.self_attn.o_proj.bias"))
            .ok().map(Arc::from);
        layers.push(OlmoLayerWeights {
            attn_norm_gain, ffn_norm_gain,
            attn_q, attn_q_bias,
            attn_k, attn_k_bias,
            attn_v, attn_v_bias,
            attn_o, attn_o_bias,
            ffn_gate, ffn_up, ffn_down,
        });
    }
    let final_norm_gain: Arc<[f32]> = Arc::from(
        load_tensor_as_f32(st, "model.norm.weight")
            .map_err(|e| E::msg(format!("final_norm: {e}")))?,
    );
    // OLMo ties lm_head to embeddings by default; fall back to embedding.
    let output = match load_transposed_matrix_preserve_dtype(
        st, "lm_head.weight", cfg.vocab_size, cfg.hidden_size,
    ) {
        Ok(w) => w,
        Err(_) => WeightStorage::F32(token_embedding.clone()),
    };
    Ok(OlmoWeights { token_embedding, layers, final_norm_gain, output })
}

fn olmo_config_from_hf_json_str(json: &str) -> Result<OlmoConfig> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let get_usize = |key: &str| -> Option<usize> {
        v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };
    let get_bool = |key: &str| -> Option<bool> { v.get(key).and_then(|x| x.as_bool()) };
    let vocab_size = get_usize("vocab_size").unwrap_or(50_304);
    let hidden_size = get_usize("hidden_size").unwrap_or(2048);
    let intermediate_size = get_usize("intermediate_size").unwrap_or(8192);
    let num_hidden_layers = get_usize("num_hidden_layers").unwrap_or(16);
    let num_attention_heads = get_usize("num_attention_heads").unwrap_or(16);
    let num_key_value_heads = get_usize("num_key_value_heads").unwrap_or(num_attention_heads);
    let head_dim = get_usize("head_dim").unwrap_or(hidden_size / num_attention_heads);
    let layer_norm_eps = get_f64("layer_norm_eps").or_else(|| get_f64("rms_norm_eps")).unwrap_or(1e-5);
    let rope_theta = get_f64("rope_theta").unwrap_or(10_000.0);
    let max_position_embeddings = get_usize("max_position_embeddings").unwrap_or(2048);
    let attention_bias = get_bool("attention_bias").unwrap_or(false);
    Ok(OlmoConfig {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        layer_norm_eps,
        rope_theta,
        max_position_embeddings,
        attention_bias,
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

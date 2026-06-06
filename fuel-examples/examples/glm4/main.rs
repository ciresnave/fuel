use anyhow::{Error as E, Result};
use clap::Parser;
use std::io::Write;
use std::sync::Arc;

use fuel::lazy::{
    load_tensor_as_f32, load_transposed_matrix_preserve_dtype, WeightStorage,
};
use fuel::lazy_glm4_new::{
    Glm4NewActivation, Glm4NewConfig, Glm4NewLayerWeights, Glm4NewModel, Glm4NewWeights,
};
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    /// GLM4 architectures using the new (lazy_glm4_new) lazy port.
    #[value(name = "glm4-new")]
    GLM4New,
    /// GLM4 old variant — eager-only; skipped under the lazy migration.
    #[value(name = "glm4-old")]
    GLM4Old,
}

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
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value_t = 0.8)]
    top_p: f64,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

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

    /// Which GLM4 variant. The lazy port currently supports
    /// `glm4-new` only; `glm4-old` errors out.
    #[arg(long)]
    which: Which,
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

    if matches!(args.which, Which::GLM4Old) {
        anyhow::bail!(
            "glm4-old is not currently supported by the lazy GLM4 port. \
             Use --which glm4-new instead."
        );
    }

    let _device = fuel_examples::device(args.cpu)?;
    let api = match args.cache_path.as_ref() {
        None => hf_hub::api::sync::Api::new()?,
        Some(path) => hf_hub::api::sync::ApiBuilder::from_cache(
            hf_hub::Cache::new(path.to_string().into()),
        )
        .build()
        .map_err(anyhow::Error::msg)?,
    };

    let model_id = args.model_id.clone().unwrap_or_else(|| "THUDM/GLM-4-9B-0414".to_string());
    let revision = args.revision.clone().unwrap_or_else(|| "main".to_string());
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

    let tokenizer_filename = match (args.weight_path.as_ref(), args.tokenizer.as_ref()) {
        (Some(_), Some(file)) => std::path::PathBuf::from(file),
        (None, Some(file)) => std::path::PathBuf::from(file),
        (Some(path), None) => std::path::Path::new(path).join("tokenizer.json"),
        (None, None) => repo.get("tokenizer.json")?,
    };
    let config_filename = match &args.weight_path {
        Some(path) => std::path::Path::new(path).join("config.json"),
        _ => repo.get("config.json")?,
    };
    let filenames = match &args.weight_path {
        Some(path) => {
            fuel_examples::hub_load_local_safetensors(path, "model.safetensors.index.json")?
        }
        _ => fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };

    let tokenizer = Tokenizer::from_file(tokenizer_filename).expect("Tokenizer Error");

    let config_json = std::fs::read_to_string(&config_filename)?;
    let cfg = glm4_new_config_from_hf_json_str(&config_json)?;
    let eos_token_ids: Vec<u32> = match parse_eos_token_ids(&config_json) {
        Some(ids) => ids,
        None => {
            let fallback = tokenizer
                .get_vocab(true)
                .get("<|user|>")
                .copied()
                .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
                .ok_or_else(|| E::msg("cannot find an EOS token for glm4-new"))?;
            vec![fallback]
        }
    };

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = load_glm4_new_weights(&st, &cfg)?;
    let model = Glm4NewModel { config: cfg.clone(), weights };

    let mut tok_stream = fuel_examples::token_output_stream::TokenOutputStream::new(tokenizer);
    if args.verbose {
        println!("Starting the inference loop:");
    } else {
        print!("{}", &args.prompt);
        std::io::stdout().flush()?;
    }
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
            args.temperature as f32,
            args.top_k,
            Some(args.top_p as f32),
            args.seed.wrapping_add(index as u64),
        );
        tokens.push(next_token);
        generated_tokens += 1;
        if eos_token_ids.contains(&next_token) {
            break;
        }
        if let Some(t) = tok_stream.next_token(next_token)? {
            if args.verbose {
                println!(
                    "[Count: {generated_tokens}] [Raw Token: {next_token}] [Decode Token: {t}]"
                );
            } else {
                print!("{t}");
                std::io::stdout().flush()?;
            }
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

fn load_glm4_new_weights(
    st: &fuel::safetensors::MmapedSafetensors,
    cfg: &Glm4NewConfig,
) -> Result<Glm4NewWeights> {
    let head_dim = cfg.head_dim();
    let q_dim = cfg.num_attention_heads * head_dim;
    let kv_dim = cfg.num_key_value_heads * head_dim;
    let token_embedding: Arc<[f32]> = Arc::from(
        load_tensor_as_f32(st, "model.embed_tokens.weight")
            .map_err(|e| E::msg(format!("embed_tokens: {e}")))?,
    );
    let mut layers: Vec<Glm4NewLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let input_norm_gain: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("model.layers.{i}.input_layernorm.weight"))
                .map_err(|e| E::msg(format!("input_norm L{i}: {e}")))?,
        );
        let post_self_attn_norm_gain: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("model.layers.{i}.post_self_attn_layernorm.weight"))
                .map_err(|e| E::msg(format!("post_self_attn_norm L{i}: {e}")))?,
        );
        let post_attn_norm_gain: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("model.layers.{i}.post_attention_layernorm.weight"))
                .map_err(|e| E::msg(format!("post_attn_norm L{i}: {e}")))?,
        );
        let post_mlp_norm_gain: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("model.layers.{i}.post_mlp_layernorm.weight"))
                .map_err(|e| E::msg(format!("post_mlp_norm L{i}: {e}")))?,
        );
        let q = load_transposed_matrix_preserve_dtype(
            st, &format!("model.layers.{i}.self_attn.q_proj.weight"),
            q_dim, cfg.hidden_size,
        ).map_err(|e| E::msg(format!("q L{i}: {e}")))?;
        let q_bias = load_tensor_as_f32(st, &format!("model.layers.{i}.self_attn.q_proj.bias"))
            .ok().map(Arc::from);
        let k = load_transposed_matrix_preserve_dtype(
            st, &format!("model.layers.{i}.self_attn.k_proj.weight"),
            kv_dim, cfg.hidden_size,
        ).map_err(|e| E::msg(format!("k L{i}: {e}")))?;
        let k_bias = load_tensor_as_f32(st, &format!("model.layers.{i}.self_attn.k_proj.bias"))
            .ok().map(Arc::from);
        let v = load_transposed_matrix_preserve_dtype(
            st, &format!("model.layers.{i}.self_attn.v_proj.weight"),
            kv_dim, cfg.hidden_size,
        ).map_err(|e| E::msg(format!("v L{i}: {e}")))?;
        let v_bias = load_tensor_as_f32(st, &format!("model.layers.{i}.self_attn.v_proj.bias"))
            .ok().map(Arc::from);
        let o = load_transposed_matrix_preserve_dtype(
            st, &format!("model.layers.{i}.self_attn.o_proj.weight"),
            cfg.hidden_size, q_dim,
        ).map_err(|e| E::msg(format!("o L{i}: {e}")))?;
        let gate_up = load_transposed_matrix_preserve_dtype(
            st, &format!("model.layers.{i}.mlp.gate_up_proj.weight"),
            2 * cfg.intermediate_size, cfg.hidden_size,
        ).map_err(|e| E::msg(format!("gate_up L{i}: {e}")))?;
        let down = load_transposed_matrix_preserve_dtype(
            st, &format!("model.layers.{i}.mlp.down_proj.weight"),
            cfg.hidden_size, cfg.intermediate_size,
        ).map_err(|e| E::msg(format!("down L{i}: {e}")))?;
        layers.push(Glm4NewLayerWeights {
            input_norm_gain,
            post_self_attn_norm_gain,
            post_attn_norm_gain,
            post_mlp_norm_gain,
            q, q_bias,
            k, k_bias,
            v, v_bias,
            o,
            gate_up,
            down,
        });
    }
    let final_norm_gain: Arc<[f32]> = Arc::from(
        load_tensor_as_f32(st, "model.norm.weight")
            .map_err(|e| E::msg(format!("final_norm: {e}")))?,
    );
    let lm_head = if cfg.tie_word_embeddings {
        None
    } else {
        Some(load_transposed_matrix_preserve_dtype(
            st, "lm_head.weight", cfg.vocab_size, cfg.hidden_size,
        ).unwrap_or_else(|_| WeightStorage::F32(token_embedding.clone())))
    };
    Ok(Glm4NewWeights { token_embedding, layers, final_norm_gain, lm_head })
}

fn glm4_new_config_from_hf_json_str(json: &str) -> Result<Glm4NewConfig> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let get_usize = |key: &str| -> Option<usize> {
        v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };
    let get_bool = |key: &str| -> Option<bool> { v.get(key).and_then(|x| x.as_bool()) };
    let vocab_size = get_usize("vocab_size").unwrap_or(151_552);
    let hidden_size = get_usize("hidden_size").unwrap_or(4_096);
    let intermediate_size = get_usize("intermediate_size").unwrap_or(13_696);
    let num_hidden_layers = get_usize("num_hidden_layers").unwrap_or(40);
    let num_attention_heads = get_usize("num_attention_heads").unwrap_or(32);
    let num_key_value_heads = get_usize("num_key_value_heads").unwrap_or(num_attention_heads);
    let head_dim = get_usize("head_dim");
    let partial_rotary_factor = v
        .get("partial_rotary_factor")
        .and_then(|x| x.as_f64())
        .map(|x| x as f32);
    let attention_bias = get_bool("attention_bias").unwrap_or(true);
    let max_position_embeddings = get_usize("max_position_embeddings").unwrap_or(131_072);
    let sliding_window = get_usize("sliding_window");
    let tie_word_embeddings = get_bool("tie_word_embeddings").unwrap_or(false);
    let rope_theta = get_f64("rope_theta").unwrap_or(5_000_000.0);
    let rms_norm_eps = get_f64("rms_norm_eps").unwrap_or(1e-5);
    let hidden_act = match v.get("hidden_act").and_then(|x| x.as_str()) {
        Some("gelu") => Glm4NewActivation::Gelu,
        Some("gelu_pytorch_tanh") => Glm4NewActivation::GeluPytorchTanh,
        _ => Glm4NewActivation::Silu,
    };
    Ok(Glm4NewConfig {
        vocab_size, hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, num_key_value_heads, head_dim, partial_rotary_factor,
        attention_bias, max_position_embeddings, sliding_window, tie_word_embeddings,
        rope_theta, rms_norm_eps, hidden_act,
    })
}

fn parse_eos_token_ids(json: &str) -> Option<Vec<u32>> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let raw = v.get("eos_token_id")?;
    if let Some(single) = raw.as_u64() {
        Some(vec![single as u32])
    } else if let Some(arr) = raw.as_array() {
        let ids = arr.iter()
            .filter_map(|x| x.as_u64().map(|v| v as u32))
            .collect::<Vec<_>>();
        if ids.is_empty() { None } else { Some(ids) }
    } else {
        None
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

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::io::Write;

use fuel::lazy_deepseek2::{
    DeepSeek2Activation, DeepSeek2Config, DeepSeek2Model, DeepSeek2Weights,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "lite")]
    Lite,
    #[value(name = "lite-chat")]
    LiteChat,
    #[value(name = "coder-lite-chat")]
    CoderLiteChat,
    #[value(name = "v2")]
    V2,
    #[value(name = "v2-chat")]
    V2Chat,
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

    /// The model size to use.
    #[arg(long, default_value = "lite")]
    which: Which,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

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
    let model_id = match args.model_id {
        Some(model_id) => model_id,
        None => match args.which {
            Which::CoderLiteChat => "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct".to_string(),
            Which::LiteChat => "deepseek-ai/DeepSeek-V2-Lite-Chat".to_string(),
            Which::Lite => "deepseek-ai/DeepSeek-V2-Lite".to_string(),
            Which::V2 => "deepseek-ai/DeepSeek-V2".to_string(),
            Which::V2Chat => "deepseek-ai/DeepSeek-V2-Chat".to_string(),
        },
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let config_filename = repo.get("config.json")?;
    let filenames = fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config_json = std::fs::read_to_string(&config_filename)?;
    let config = deepseek2_config_from_hf_json_str(&config_json)?;
    let eos_token_id = parse_eos_token_id(&config_json)
        .or_else(|| tokenizer.token_to_id("<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>"));

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = DeepSeek2Weights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = DeepSeek2Model { config: config.clone(), weights };
    println!("loaded the model in {:?}", start.elapsed());

    print!("{}", args.prompt);
    std::io::stdout().flush()?;
    let mut tokens = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let start_gen = std::time::Instant::now();
    let mut generated_tokens = 0usize;
    for index in 0..args.sample_len {
        let logits = model
            .forward(&tokens, 0)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let logits_data = logits.realize_f32();
        let vocab_size = config.vocab_size;
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

fn deepseek2_config_from_hf_json_str(json: &str) -> Result<DeepSeek2Config> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let get_usize = |key: &str| -> Option<usize> {
        v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };
    let get_bool = |key: &str| -> Option<bool> { v.get(key).and_then(|x| x.as_bool()) };
    let get_str = |key: &str| -> Option<String> {
        v.get(key).and_then(|x| x.as_str()).map(|x| x.to_string())
    };

    let vocab_size = get_usize("vocab_size")
        .ok_or_else(|| E::msg("missing vocab_size"))?;
    let hidden_size = get_usize("hidden_size")
        .ok_or_else(|| E::msg("missing hidden_size"))?;
    let intermediate_size = get_usize("intermediate_size")
        .ok_or_else(|| E::msg("missing intermediate_size"))?;
    let moe_intermediate_size = get_usize("moe_intermediate_size").unwrap_or(intermediate_size);
    let num_hidden_layers = get_usize("num_hidden_layers")
        .ok_or_else(|| E::msg("missing num_hidden_layers"))?;
    let num_attention_heads = get_usize("num_attention_heads")
        .ok_or_else(|| E::msg("missing num_attention_heads"))?;
    let n_shared_experts = get_usize("n_shared_experts");
    let n_routed_experts = get_usize("n_routed_experts");
    let num_experts_per_tok = get_usize("num_experts_per_tok");
    let moe_layer_freq = get_usize("moe_layer_freq").unwrap_or(1);
    let first_k_dense_replace = get_usize("first_k_dense_replace").unwrap_or(0);
    let norm_topk_prob = get_bool("norm_topk_prob").unwrap_or(false);
    let hidden_activation = match get_str("hidden_act").as_deref() {
        Some("gelu") | Some("gelu_new") | Some("gelu_pytorch_tanh") => DeepSeek2Activation::Gelu,
        _ => DeepSeek2Activation::Silu,
    };
    let max_position_embeddings = get_usize("max_position_embeddings").unwrap_or(4096);
    let rms_norm_eps = get_f64("rms_norm_eps").unwrap_or(1e-6);
    let tie_word_embeddings = get_bool("tie_word_embeddings").unwrap_or(false);
    let rope_theta = get_f64("rope_theta").unwrap_or(10_000.0);
    let attention_bias = get_bool("attention_bias").unwrap_or(false);
    let q_lora_rank = get_usize("q_lora_rank");
    let qk_rope_head_dim = get_usize("qk_rope_head_dim").unwrap_or(64);
    let kv_lora_rank = get_usize("kv_lora_rank").unwrap_or(512);
    let v_head_dim = get_usize("v_head_dim").unwrap_or(128);
    let qk_nope_head_dim = get_usize("qk_nope_head_dim").unwrap_or(128);

    Ok(DeepSeek2Config {
        vocab_size,
        hidden_size,
        intermediate_size,
        moe_intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        n_shared_experts,
        n_routed_experts,
        num_experts_per_tok,
        moe_layer_freq,
        first_k_dense_replace,
        norm_topk_prob,
        hidden_activation,
        max_position_embeddings,
        rms_norm_eps,
        tie_word_embeddings,
        rope_theta,
        attention_bias,
        q_lora_rank,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        qk_nope_head_dim,
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

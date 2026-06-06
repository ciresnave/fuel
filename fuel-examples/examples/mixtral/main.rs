#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::io::Write;
use std::sync::Arc;

use fuel::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype, WeightStorage};
use fuel::lazy_mixtral::{
    MixtralConfig, MixtralExpertWeights, MixtralLayerWeights, MixtralModel, MixtralWeights,
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

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    #[arg(long, default_value = "mistralai/Mixtral-8x7B-v0.1")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

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

fn mixtral_config_from_hf_json_str(json: &str) -> Result<MixtralConfig> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let get_usize = |key: &str| -> Result<usize> {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .ok_or_else(|| E::msg(format!("config.json: missing/invalid field {key:?}")))
    };
    let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };
    let vocab_size = get_usize("vocab_size")?;
    let hidden_size = get_usize("hidden_size")?;
    let intermediate_size = get_usize("intermediate_size")?;
    let num_hidden_layers = get_usize("num_hidden_layers")?;
    let num_attention_heads = get_usize("num_attention_heads")?;
    let num_key_value_heads = v
        .get("num_key_value_heads")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(num_attention_heads);
    let head_dim = v
        .get("head_dim")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(hidden_size / num_attention_heads);
    let rms_norm_eps = get_f64("rms_norm_eps").unwrap_or(1e-5);
    let rope_theta = get_f64("rope_theta").unwrap_or(1e6);
    let max_position_embeddings = v
        .get("max_position_embeddings")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(32_768);
    let sliding_window = v
        .get("sliding_window")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize);
    let num_experts_per_tok = v
        .get("num_experts_per_tok")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(2);
    let num_local_experts = v
        .get("num_local_experts")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(8);
    Ok(MixtralConfig {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        sliding_window,
        num_experts_per_tok,
        num_local_experts,
    })
}

fn load_mixtral_weights(
    st: &fuel::safetensors::MmapedSafetensors,
    cfg: &MixtralConfig,
) -> Result<MixtralWeights> {
    let token_embedding = load_tensor_as_f32(st, "model.embed_tokens.weight")?;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    let mut layers: Vec<MixtralLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let base = format!("model.layers.{i}");
        let attn_q = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.self_attn.q_proj.weight"),
            cfg.hidden_size,
            cfg.hidden_size,
        )?;
        let attn_k = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.self_attn.k_proj.weight"),
            kv_dim,
            cfg.hidden_size,
        )?;
        let attn_v = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.self_attn.v_proj.weight"),
            kv_dim,
            cfg.hidden_size,
        )?;
        let attn_o = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.self_attn.o_proj.weight"),
            cfg.hidden_size,
            cfg.hidden_size,
        )?;
        let attn_norm_gain = load_tensor_as_f32(st, &format!("{base}.input_layernorm.weight"))?;
        let ffn_norm_gain =
            load_tensor_as_f32(st, &format!("{base}.post_attention_layernorm.weight"))?;
        // Router gate: HF stores `[num_experts, hidden]` but we want
        // `[hidden, num_experts]` for `x @ gate_w`.
        let gate_flat = load_tensor_as_f32(
            st,
            &format!("{base}.block_sparse_moe.gate.weight"),
        )?;
        // Transpose [num_experts, hidden] → [hidden, num_experts].
        let e = cfg.num_local_experts;
        let h = cfg.hidden_size;
        if gate_flat.len() != e * h {
            return Err(E::msg(format!(
                "router gate at layer {i}: expected {} elements got {}",
                e * h,
                gate_flat.len()
            )));
        }
        let mut gate_w_t = vec![0.0_f32; e * h];
        for ei in 0..e {
            for hi in 0..h {
                gate_w_t[hi * e + ei] = gate_flat[ei * h + hi];
            }
        }
        let mut experts: Vec<MixtralExpertWeights> = Vec::with_capacity(e);
        for ei in 0..e {
            let ebase = format!("{base}.block_sparse_moe.experts.{ei}");
            // HF naming: w1 = gate_proj, w3 = up_proj, w2 = down_proj.
            let gate_w = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{ebase}.w1.weight"),
                cfg.intermediate_size,
                cfg.hidden_size,
            )?;
            let up_w = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{ebase}.w3.weight"),
                cfg.intermediate_size,
                cfg.hidden_size,
            )?;
            let down_w = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{ebase}.w2.weight"),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?;
            experts.push(MixtralExpertWeights {
                gate_w,
                up_w,
                down_w,
            });
        }
        layers.push(MixtralLayerWeights {
            attn_norm_gain: Arc::from(attn_norm_gain),
            ffn_norm_gain: Arc::from(ffn_norm_gain),
            attn_q,
            attn_k,
            attn_v,
            attn_o,
            gate_w: Arc::from(gate_w_t),
            experts,
        });
    }
    let final_norm_gain = load_tensor_as_f32(st, "model.norm.weight")?;
    let output: WeightStorage = match load_transposed_matrix_preserve_dtype(
        st,
        "lm_head.weight",
        cfg.vocab_size,
        cfg.hidden_size,
    ) {
        Ok(w) => w,
        Err(_) => {
            // Tied weights — transpose embed_tokens.
            let mut transposed = vec![0.0_f32; cfg.hidden_size * cfg.vocab_size];
            for i in 0..cfg.vocab_size {
                for j in 0..cfg.hidden_size {
                    transposed[j * cfg.vocab_size + i] =
                        token_embedding[i * cfg.hidden_size + j];
                }
            }
            WeightStorage::F32(Arc::from(transposed))
        }
    };
    Ok(MixtralWeights {
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
    let _ = args.use_flash_attn;
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
        args.model_id.clone(),
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames: Vec<std::path::PathBuf> = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config_file = repo.get("config.json")?;
    let config_json = std::fs::read_to_string(&config_file)?;
    let config = mixtral_config_from_hf_json_str(&config_json)?;
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights =
        load_mixtral_weights(&st, &config).map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = MixtralModel {
        config: config.clone(),
        weights,
    };
    println!("loaded the model in {:?}", start.elapsed());

    let mut tok_stream = fuel_examples::token_output_stream::TokenOutputStream::new(tokenizer);
    let eos_token = match tok_stream.get_token("</s>") {
        Some(token) => token,
        None => anyhow::bail!("cannot find the </s> token"),
    };

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
            args.temperature.unwrap_or(0.) as f32,
            None,
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

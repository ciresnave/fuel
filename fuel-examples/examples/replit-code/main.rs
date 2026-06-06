// Replit-Code (MPT-3B) — lazy-graph port.
//
// Forward-only inference using `fuel::lazy_mpt::MptModel`. Builds the
// graph from token IDs each step. v1 does not maintain a KV cache —
// we recompute over the full prefix per generated token (O(n²) but
// correct).

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{anyhow, Error as E, Result};
use clap::Parser;

use fuel::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype, WeightStorage};
use fuel::lazy_mpt::{MptConfig, MptLayerWeights, MptModel, MptWeights};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::sync::Arc;
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

    /// Display the token for the specified prompt.
    #[arg(long)]
    verbose_prompt: bool,

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

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    weight_file: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

/// Load MPT weights from a memory-mapped safetensors checkpoint.
///
/// Tensor names follow the MosaicML / replit-code convention:
///   - `transformer.wte.weight`                                 — token embedding
///   - `transformer.blocks.{i}.norm_1.{weight,bias}`            — pre-attn LN
///   - `transformer.blocks.{i}.attn.Wqkv.weight`                — fused QKV
///                                                                ([d_model + 2*kv_dim, d_model])
///   - `transformer.blocks.{i}.attn.out_proj.weight`
///   - `transformer.blocks.{i}.norm_2.{weight,bias}`            — pre-FFN LN
///   - `transformer.blocks.{i}.ffn.up_proj.weight`
///   - `transformer.blocks.{i}.ffn.down_proj.weight`
///   - `transformer.norm_f.{weight,bias}`                       — final LN
///   - `transformer.wte.weight` (tied)                          — output head
fn load_mpt_weights(
    st: &fuel::safetensors::MmapedSafetensors,
    cfg: &MptConfig,
) -> Result<MptWeights> {
    let h = cfg.d_model;
    let kv_dim = cfg.kv_n_heads * cfg.head_dim();
    let inter = cfg.ffn_dim();

    let token_embedding =
        load_tensor_as_f32(st, "transformer.wte.weight").map_err(|e| anyhow!("{e}"))?;
    if token_embedding.len() != cfg.vocab_size * h {
        anyhow::bail!(
            "wte: {} elements, expected {} ({}×{})",
            token_embedding.len(),
            cfg.vocab_size * h,
            cfg.vocab_size,
            h,
        );
    }

    let mut layers: Vec<MptLayerWeights> = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let norm1_gain =
            load_tensor_as_f32(st, &format!("transformer.blocks.{i}.norm_1.weight"))
                .map_err(|e| anyhow!("{e}"))?;
        // norm_1.bias is absent in some MPT checkpoints; fall back to zeros.
        let norm1_bias = load_tensor_as_f32(st, &format!("transformer.blocks.{i}.norm_1.bias"))
            .unwrap_or_else(|_| vec![0.0_f32; h]);
        let norm2_gain =
            load_tensor_as_f32(st, &format!("transformer.blocks.{i}.norm_2.weight"))
                .map_err(|e| anyhow!("{e}"))?;
        let norm2_bias = load_tensor_as_f32(st, &format!("transformer.blocks.{i}.norm_2.bias"))
            .unwrap_or_else(|_| vec![0.0_f32; h]);

        // Wqkv: HF stores `[out=d_model + 2*kv_dim, in=d_model]` flat row-major.
        // We split that into Q (d_model rows), K (kv_dim rows), V (kv_dim rows)
        // and transpose each into our `[in, out]` layout.
        let wqkv_flat = load_tensor_as_f32(
            st,
            &format!("transformer.blocks.{i}.attn.Wqkv.weight"),
        )
        .map_err(|e| anyhow!("{e}"))?;
        let expected = (h + 2 * kv_dim) * h;
        if wqkv_flat.len() != expected {
            anyhow::bail!(
                "blocks.{i}.attn.Wqkv: {} elements, expected {}",
                wqkv_flat.len(),
                expected,
            );
        }
        // Q is rows [0..d_model), K is rows [d_model..d_model+kv_dim), V is rows
        // [d_model+kv_dim..d_model+2*kv_dim).
        let mut attn_q_flat = vec![0.0_f32; h * h];
        let mut attn_k_flat = vec![0.0_f32; kv_dim * h];
        let mut attn_v_flat = vec![0.0_f32; kv_dim * h];
        for row in 0..h {
            for col in 0..h {
                attn_q_flat[row * h + col] = wqkv_flat[row * h + col];
            }
        }
        for row in 0..kv_dim {
            for col in 0..h {
                attn_k_flat[row * h + col] = wqkv_flat[(h + row) * h + col];
            }
        }
        for row in 0..kv_dim {
            for col in 0..h {
                attn_v_flat[row * h + col] = wqkv_flat[(h + kv_dim + row) * h + col];
            }
        }
        // Transpose Q [h, h] (HF) → [h, h] (fuel layout: `[in, out]`).
        let mut attn_q = vec![0.0_f32; h * h];
        for o in 0..h {
            for inn in 0..h {
                attn_q[inn * h + o] = attn_q_flat[o * h + inn];
            }
        }
        let mut attn_k = vec![0.0_f32; h * kv_dim];
        for o in 0..kv_dim {
            for inn in 0..h {
                attn_k[inn * kv_dim + o] = attn_k_flat[o * h + inn];
            }
        }
        let mut attn_v = vec![0.0_f32; h * kv_dim];
        for o in 0..kv_dim {
            for inn in 0..h {
                attn_v[inn * kv_dim + o] = attn_v_flat[o * h + inn];
            }
        }

        let attn_o = load_transposed_matrix_preserve_dtype(
            st,
            &format!("transformer.blocks.{i}.attn.out_proj.weight"),
            h,
            h,
        )
        .map_err(|e| anyhow!("{e}"))?;

        let mlp_up = load_transposed_matrix_preserve_dtype(
            st,
            &format!("transformer.blocks.{i}.ffn.up_proj.weight"),
            inter,
            h,
        )
        .map_err(|e| anyhow!("{e}"))?;
        let mlp_down = load_transposed_matrix_preserve_dtype(
            st,
            &format!("transformer.blocks.{i}.ffn.down_proj.weight"),
            h,
            inter,
        )
        .map_err(|e| anyhow!("{e}"))?;

        layers.push(MptLayerWeights {
            norm1_gain: Arc::from(norm1_gain),
            norm1_bias: Arc::from(norm1_bias),
            norm2_gain: Arc::from(norm2_gain),
            norm2_bias: Arc::from(norm2_bias),
            attn_q: WeightStorage::F32(Arc::from(attn_q)),
            attn_k: WeightStorage::F32(Arc::from(attn_k)),
            attn_v: WeightStorage::F32(Arc::from(attn_v)),
            attn_o,
            mlp_up,
            mlp_down,
        });
    }

    let final_ln_gain =
        load_tensor_as_f32(st, "transformer.norm_f.weight").map_err(|e| anyhow!("{e}"))?;
    let final_ln_bias = load_tensor_as_f32(st, "transformer.norm_f.bias")
        .unwrap_or_else(|_| vec![0.0_f32; h]);

    // MPT/replit-code ties the output head to the input embedding —
    // there is no separate `lm_head.weight`. Transpose wte for the
    // matmul layout `[hidden, vocab]`.
    let mut tied = vec![0.0_f32; h * cfg.vocab_size];
    for v in 0..cfg.vocab_size {
        for j in 0..h {
            tied[j * cfg.vocab_size + v] = token_embedding[v * h + j];
        }
    }
    let output = WeightStorage::F32(Arc::from(tied));

    Ok(MptWeights {
        token_embedding: Arc::from(token_embedding),
        layers,
        final_ln_gain: Arc::from(final_ln_gain),
        final_ln_bias: Arc::from(final_ln_bias),
        output,
    })
}

fn main() -> Result<()> {
    use std::io::Write;
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

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = args.model_id.clone().unwrap_or_else(|| "lmz/fuel-replit-code".to_string());
    let revision = args.revision.clone().unwrap_or_else(|| "main".to_string());
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let tokenizer_filename = match args.tokenizer.clone() {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filename = match args.weight_file.clone() {
        Some(weight_file) => std::path::PathBuf::from(weight_file),
        None => repo.get("model.safetensors")?,
    };
    println!("retrieved the files in {:?}", start.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let cfg = MptConfig::replit_code_v1_5_3b();
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[filename]) }
        .map_err(|e| anyhow!("mmap safetensors: {e}"))?;
    let weights = load_mpt_weights(&st, &cfg)?;
    let model = MptModel { config: cfg.clone(), weights };
    println!("loaded the model in {:?}", start.elapsed());

    let tokens = tokenizer.encode(args.prompt.clone(), true).map_err(E::msg)?;
    if tokens.is_empty() {
        anyhow::bail!("Empty prompts are not supported.")
    }
    if args.verbose_prompt {
        for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
            let token = token.replace('▁', " ").replace("<0x0A>", "\n");
            println!("{id:7} -> '{token}'");
        }
    }
    let mut tokens = tokens.get_ids().to_vec();
    let eos_token = match tokenizer.get_vocab(true).get("<|endoftext|>") {
        Some(token) => *token,
        None => anyhow::bail!("cannot find the endoftext token"),
    };

    print!("{}", args.prompt);
    std::io::stdout().flush()?;
    let start_gen = std::time::Instant::now();
    let mut generated_tokens = 0_usize;

    for index in 0..args.sample_len {
        // No KV cache in v1 — recompute the whole prefix each step.
        let logits = model.forward(&tokens)?;
        let logits_data = logits.realize_f32();
        let vocab_size = cfg.vocab_size;
        let seq = tokens.len();
        let last_offset = (seq - 1) * vocab_size;
        let mut last_logits: Vec<f32> = logits_data[last_offset..last_offset + vocab_size].to_vec();

        if args.repeat_penalty != 1.0 {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            apply_repeat_penalty(&mut last_logits, args.repeat_penalty, &tokens[start_at..]);
        }
        let next_token = sample(
            &last_logits,
            args.temperature.unwrap_or(0.0) as f32,
            args.top_k,
            args.top_p.map(|p| p as f32),
            args.seed.wrapping_add(index as u64),
        );
        tokens.push(next_token);
        generated_tokens += 1;
        if next_token == eos_token {
            break;
        }
        let token = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{token}");
        std::io::stdout().flush()?;
    }
    let dt = start_gen.elapsed();
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
        let mut best_i = 0_usize;
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

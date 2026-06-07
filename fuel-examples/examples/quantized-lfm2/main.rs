//! GGUF-quantized LFM2 (Liquid Foundation Model 2) example.
//!
//! Loads an LFM2 GGUF and runs greedy text generation via the lazy
//! [`fuel::lazy_quantized_lfm2::QuantizedLFM2Model`]. Mirrors the
//! `quantized-qwen3` example's CLI shape (model path, prompt,
//! sample length, sampling params) so the binary keeps the same
//! ergonomics across the quantized-model family.
//!
//! v1 caveats — see the lazy port for details:
//! - Prefill is the only mode that exercises the ShortConv (LIV)
//!   block correctly; the autoregressive cache for ShortConv is
//!   not yet wired and is gated on the multi-output-node design.
//!   Greedy decode here still works because the lazy
//!   `forward(&[next_token], pos)` call rebuilds the graph from
//!   scratch per step — slow but correct.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::io::Write;
use tokenizers::Tokenizer;

use fuel::lazy_lfm2::{LFM2BlockType, LFM2Config};
use fuel::lazy_quantized_lfm2::QuantizedLFM2Model;
use fuel::quantized::gguf_mmap::MmapedContent;
use fuel_transformers::generation::{LogitsProcessor, Sampling};

use fuel_examples::token_output_stream::TokenOutputStream;

const DEFAULT_PROMPT: &str = "Tell me a story in 100 words.";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to an LFM2 GGUF file. Required — the example does not
    /// hard-code a HuggingFace repo for LFM2.
    #[arg(long)]
    model: String,

    /// Path to a tokenizer.json. Required — LFM2 uses a model-specific
    /// tokenizer; we don't fetch it automatically.
    #[arg(long)]
    tokenizer: String,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Number of tokens to generate (counting the first sampled one).
    #[arg(short = 'n', long, default_value_t = 128)]
    sample_len: usize,

    /// Temperature for sampling. 0.0 = greedy / argmax.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// Nucleus-sampling top-p (combined with `temperature`).
    #[arg(long)]
    top_p: Option<f64>,

    /// Sample only among the top-k logits (combined with `temperature`).
    #[arg(long)]
    top_k: Option<usize>,

    /// Seed for the sampler.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Repeat penalty (1.0 = off).
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// Last N tokens to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Process prompt tokens one at a time (slower; useful for
    /// debugging start_pos plumbing).
    #[arg(long)]
    split_prompt: bool,

    /// Forces CPU even if a GPU is available (no-op for the lazy
    /// path today — the lazy module picks its own device).
    #[arg(long)]
    cpu: bool,
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{size_in_bytes}B")
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

/// Build an `LFM2Config` from the GGUF metadata. Mirrors the eager
/// `quantized_lfm2` loader's keyset (`lfm2.*`).
fn lfm2_cfg_from_gguf(mc: &MmapedContent) -> Result<LFM2Config> {
    let md = mc.metadata();
    let get = |k: &str| -> Result<&fuel::quantized::gguf_file::Value> {
        md.get(k).ok_or_else(|| E::msg(format!("gguf metadata: missing key {k:?}")))
    };
    let num_attention_heads = get("lfm2.attention.head_count")?.to_u32()? as usize;
    let num_hidden_layers = get("lfm2.block_count")?.to_u32()? as usize;
    let hidden_size = get("lfm2.embedding_length")?.to_u32()? as usize;
    let intermediate_size = match md.get("lfm2.feed_forward_length") {
        Some(v) => v.to_u32()? as usize,
        None => get("lfm2.intermediate_size")?.to_u32()? as usize,
    };
    let max_position_embeddings = get("lfm2.context_length")?.to_u32()? as usize;
    let rms_norm_eps = get("lfm2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
    let rope_theta = md.get("lfm2.rope.freq_base")
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(1_000_000.0) as f64;
    let conv_kernel_size = get("lfm2.shortconv.l_cache")?.to_u32()? as usize;

    // head_count_kv may be a per-layer array or a scalar. Per-layer
    // arrays signal LFM2's "Attention if kv_heads > 0 else Conv"
    // schedule; a scalar means uniform attention across layers.
    let block_types: Vec<LFM2BlockType> = match md.get("lfm2.attention.head_count_kv") {
        Some(fuel::quantized::gguf_file::Value::Array(arr)) => {
            arr.iter().map(|v| {
                let n = v.to_u32().unwrap_or(0) as usize;
                if n > 0 { LFM2BlockType::Attention } else { LFM2BlockType::Conv }
            }).collect()
        }
        Some(v) => {
            let n = v.to_u32()? as usize;
            vec![if n > 0 { LFM2BlockType::Attention } else { LFM2BlockType::Conv }; num_hidden_layers]
        }
        None => return Err(E::msg("gguf metadata: missing lfm2.attention.head_count_kv")),
    };
    if block_types.len() != num_hidden_layers {
        return Err(E::msg(format!(
            "gguf: lfm2.attention.head_count_kv has {} entries, expected {num_hidden_layers}",
            block_types.len(),
        )));
    }
    let num_key_value_heads = match md.get("lfm2.attention.head_count_kv") {
        Some(fuel::quantized::gguf_file::Value::Array(arr)) => {
            // First non-zero entry — all attention layers share the
            // same KV-head count in LFM2 releases.
            arr.iter().filter_map(|v| v.to_u32().ok()).find(|&n| n > 0)
                .ok_or_else(|| E::msg("gguf: no attention layer in head_count_kv (every entry is 0)"))? as usize
        }
        Some(v) => v.to_u32()? as usize,
        None => unreachable!(),
    };
    let head_dim = hidden_size / num_attention_heads;

    // Vocab — derive from token-embedding tensor if not explicitly listed.
    let vocab_size = match md.get("lfm2.vocab_size") {
        Some(v) => v.to_u32()? as usize,
        None => {
            let info = mc.content().tensor_infos
                .get("token_embd.weight")
                .ok_or_else(|| E::msg("gguf: missing token_embd.weight"))?;
            let dims = info.shape.dims();
            if dims.is_empty() {
                return Err(E::msg("gguf: token_embd.weight has empty shape"));
            }
            dims[0]
        }
    };

    Ok(LFM2Config {
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        intermediate_size,
        max_position_embeddings,
        rope_theta,
        rms_norm_eps,
        conv_kernel_size,
        block_types,
    })
}

fn main() -> anyhow::Result<()> {
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

    let model_path = std::path::PathBuf::from(&args.model);
    let _device = fuel_examples::device(args.cpu)?;
    let start = std::time::Instant::now();

    let mmaped = MmapedContent::from_path(&model_path).map_err(|e| e.with_path(&model_path))?;
    let metadata_done = start.elapsed();
    let mut total_size_in_bytes = 0;
    for (_, tensor) in mmaped.content().tensor_infos.iter() {
        let elem_count = tensor.shape.elem_count();
        total_size_in_bytes +=
            elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
    }
    println!(
        "mmapped {:?} tensors ({}); header in {:.2}s",
        mmaped.content().tensor_infos.len(),
        &format_size(total_size_in_bytes),
        metadata_done.as_secs_f32(),
    );

    let cfg = lfm2_cfg_from_gguf(&mmaped)?;
    println!(
        "config: hidden={} layers={} heads={} kv_heads={} head_dim={} ffn={} conv_k={} vocab={}",
        cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads,
        cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size,
        cfg.conv_kernel_size, cfg.vocab_size,
    );
    let n_attn = cfg.block_types.iter().filter(|b| matches!(b, LFM2BlockType::Attention)).count();
    let n_conv = cfg.block_types.iter().filter(|b| matches!(b, LFM2BlockType::Conv)).count();
    println!("block schedule: {n_attn} attention + {n_conv} short-conv (LIV)");
    drop(mmaped);

    let model = QuantizedLFM2Model::from_gguf(&model_path, &cfg)
        .map_err(|e| E::msg(format!("from_gguf: {e}")))?;
    println!(
        "model built in {:.2}s total",
        start.elapsed().as_secs_f32()
    );

    let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(anyhow::Error::msg)?;
    let mut tos = TokenOutputStream::new(tokenizer);
    let prompt_str = args.prompt.clone().unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    print!("prompt: {prompt_str}\n");

    let encoded = tos.tokenizer()
        .encode(prompt_str.as_str(), true)
        .map_err(anyhow::Error::msg)?;
    let tokens = encoded.get_ids().to_vec();
    if tokens.is_empty() {
        return Err(E::msg("tokenizer produced an empty token sequence"));
    }

    let to_sample = args.sample_len.saturating_sub(1);
    let mut all_tokens: Vec<u32> = Vec::with_capacity(args.sample_len);

    let mut logits_processor = {
        let temperature = args.temperature;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (args.top_k, args.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    let vocab_size = cfg.vocab_size;
    let last_logits_to_tensor = |logits_flat: Vec<f32>, seq: usize| -> Result<Vec<f32>> {
        let last_off = (seq - 1) * vocab_size;
        Ok(logits_flat[last_off..last_off + vocab_size].to_vec())
    };

    // Prompt processing.
    let start_prompt_processing = std::time::Instant::now();
    let mut next_token = if !args.split_prompt {
        let logits_lazy = model
            .forward(&tokens, 0)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let logits_flat = logits_lazy.realize_f32();
        let logits = last_logits_to_tensor(logits_flat, tokens.len())?;
        logits_processor.sample(&logits)?
    } else {
        let mut next_token = 0;
        for (pos, token) in tokens.iter().enumerate() {
            let logits_lazy = model
                .forward(&[*token], pos)
                .map_err(|e| E::msg(format!("forward split: {e}")))?;
            let logits_flat = logits_lazy.realize_f32();
            let logits = last_logits_to_tensor(logits_flat, 1)?;
            next_token = logits_processor.sample(&logits)?;
        }
        next_token
    };
    let prompt_dt = start_prompt_processing.elapsed();
    all_tokens.push(next_token);
    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        std::io::stdout().flush()?;
    }

    // Decode loop.
    let eos_id = {
        let vocab = tos.tokenizer().get_vocab(true);
        vocab.get("</s>")
            .or_else(|| vocab.get("<|endoftext|>"))
            .or_else(|| vocab.get("<|im_end|>"))
            .copied()
    };
    let start_post_prompt = std::time::Instant::now();
    let mut sampled = 0;
    for index in 0..to_sample {
        let logits_lazy = model
            .forward(&[next_token], tokens.len() + index)
            .map_err(|e| E::msg(format!("forward decode: {e}")))?;
        let logits_flat = logits_lazy.realize_f32();
        let mut logits = last_logits_to_tensor(logits_flat, 1)?;
        if args.repeat_penalty != 1. {
            let start_at = all_tokens.len().saturating_sub(args.repeat_last_n);
            fuel_transformers::utils::apply_repeat_penalty(
                &mut logits,
                args.repeat_penalty,
                &all_tokens[start_at..],
            );
        }
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
        sampled += 1;
        if Some(next_token) == eos_id {
            break;
        }
    }

    if let Some(rest) = tos.decode_rest().map_err(fuel::Error::msg)? {
        print!("{rest}");
    }
    std::io::stdout().flush()?;
    let dt = start_post_prompt.elapsed();
    println!(
        "\n\n{:4} prompt tokens processed: {:.2} token/s",
        tokens.len(),
        tokens.len() as f64 / prompt_dt.as_secs_f64(),
    );
    println!(
        "{sampled:4} tokens generated: {:.2} token/s",
        sampled as f64 / dt.as_secs_f64(),
    );
    Ok(())
}

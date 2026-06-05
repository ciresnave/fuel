// End-to-end LLaMA-family model runner that uses fuel's Phase 6a
// lazy graph layer and the gemm-backed fast CPU executor.
//
// USAGE
//
//     cargo run --release --bin llama-lazy
//     cargo run --release --bin llama-lazy -- [MODEL_ID] [PROMPT] [MAX_NEW_TOKENS]
//
// Defaults:
//     MODEL_ID        = TinyLlama/TinyLlama-1.1B-Chat-v1.0
//     PROMPT          = "Once upon a time"
//     MAX_NEW_TOKENS  = 32
//
// MODEL CHOICE
//
// The default is TinyLlama 1.1B — a non-gated, downloadable-without-
// authentication model that uses the LLaMA architecture. Small enough
// to download in a few minutes and to run through the current executor
// in a reasonable time.
//
// For Llama 3 8B, pass `meta-llama/Meta-Llama-3-8B` as the first arg.
// That model is gated: you must accept the license on Hugging Face
// and set HF_TOKEN (or run `huggingface-cli login`) first. The
// download is ~16GB and takes substantially longer.
//
// PERFORMANCE EXPECTATIONS
//
// The fast CPU executor uses `gemm` under the hood (~50-200x faster
// than the reference path) but is still pure CPU and has no KV cache.
// TinyLlama on a modern desktop CPU will land somewhere between "a
// second per token" and "a few seconds per token" depending on
// sequence length. Slow compared to production serving stacks, but
// fast enough to demonstrate correctness and observe real generated
// text.
//
// The next major speedup after the fast executor comes from adding
// a KV cache (which cuts per-step work from O(seq²) to O(seq)) and,
// later, vendor-tuned BLAS through future `fuel-intelcpu-backend`
// / `fuel-amdcpu-backend` crates.

use fuel::lazy::{LlamaTokenizer, SamplingStrategy};
use fuel::lazy_llama2c::Llama2cModel;
use fuel::{DType, Device};
use std::io::Write;
use std::time::Instant;

const DEFAULT_MODEL: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
const DEFAULT_PROMPT: &str = "Once upon a time";
const DEFAULT_MAX_NEW: usize = 32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _trace_guard = if std::env::var("FUEL_TRACE").is_ok() {
        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file("trace.json")
            .include_args(true)
            .build();
        use tracing_subscriber::prelude::*;
        tracing_subscriber::registry()
            .with(chrome_layer)
            .init();
        eprintln!("Tracing enabled → trace.json");
        Some(guard)
    } else {
        None
    };

    let args: Vec<String> = std::env::args().collect();
    let model_id = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let max_new: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_NEW);

    eprintln!("=== fuel llama-lazy ===");
    eprintln!("Model:  {model_id}");
    eprintln!("Prompt: {prompt:?}");
    eprintln!("Max new tokens: {max_new}");
    eprintln!();

    eprint!("Downloading + loading model weights... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let model = Llama2cModel::from_hub(&model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  config: dim={}  layers={}  heads={}  kv_heads={}  vocab={}  rope_theta={}",
        model.config.dim,
        model.config.n_layers,
        model.config.n_heads,
        model.config.n_kv_heads,
        model.config.vocab_size,
        model.config.rope_theta,
    );

    eprint!("Loading tokenizer...             ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let tokenizer = LlamaTokenizer::from_hub(&model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());

    let prompt_tokens = tokenizer.encode(&prompt, true)?;
    eprintln!("Prompt tokens: {}", prompt_tokens.len());
    eprintln!();

    eprintln!("Generating (this may take a while on CPU)...");
    eprintln!("---");
    // Stream tokens as they're produced. BPE pieces can emit partial
    // UTF-8 (one token may span a multi-byte codepoint with a
    // neighbour), so we can't just decode each token on its own —
    // instead we decode the whole sequence so far and print whatever
    // is newly appended to the prior decode. That's how every
    // production streaming-decoder handles this.
    print!("{prompt}");
    std::io::stdout().flush().ok();
    let mut streamed: Vec<u32> = prompt_tokens.clone();
    let mut printed_text = tokenizer.decode(&streamed, true)?;
    let device = Device::cpu();
    let t0 = Instant::now();
    let output_tokens = model.generate_streaming_with_kv_context(
        &prompt_tokens,
        max_new,
        SamplingStrategy::Temperature { temp: 0.8, seed: 42 },
        tokenizer.eos_id(),
        &device,
        DType::F32,
        |tok| {
            streamed.push(tok);
            if let Ok(full) = tokenizer.decode(&streamed, true) {
                if let Some(delta) = full.strip_prefix(&printed_text) {
                    print!("{delta}");
                    std::io::stdout().flush().ok();
                }
                printed_text = full;
            }
        },
    )?;
    let elapsed = t0.elapsed();
    println!();

    let new_tokens = output_tokens.len().saturating_sub(prompt_tokens.len());
    eprintln!();
    eprintln!("---");
    eprintln!(
        "Generated {new_tokens} tokens in {:.2?} ({:.2}s/token avg)",
        elapsed,
        elapsed.as_secs_f64() / new_tokens.max(1) as f64,
    );
    if let Some(eos) = tokenizer.eos_id() {
        if output_tokens.last() == Some(&eos) {
            eprintln!("(stopped early on EOS token {eos})");
        }
    }

    Ok(())
}

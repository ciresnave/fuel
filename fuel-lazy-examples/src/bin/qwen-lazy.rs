// End-to-end Qwen2 runner built on the same LazyTensor path as
// `llama-lazy`, proving the LLaMA-family forward code also handles
// architectures with Q/K/V attention biases.
//
// USAGE
//
//     cargo run --release --bin qwen-lazy
//     cargo run --release --bin qwen-lazy -- [MODEL_ID] [PROMPT] [MAX_NEW_TOKENS]
//
// Defaults:
//     MODEL_ID        = Qwen/Qwen2-0.5B-Instruct
//     PROMPT          = "The quick brown fox"
//     MAX_NEW_TOKENS  = 32
//
// WHY A SEPARATE BINARY
//
// Qwen2's safetensors carry q_proj.bias / k_proj.bias / v_proj.bias
// tensors that LLaMA doesn't have. The loader inside fuel-core
// detects them automatically and wires them into the attention block
// via a broadcast-add after each projection; no config flag needed.
// This binary is just the end-to-end proof that the wiring works on
// real weights: nothing here is Qwen-specific, so pointing it at a
// LLaMA-family repo also works.
//
// Qwen2 0.5B is ~1GB on disk and runs in roughly the same
// time-per-token range as TinyLlama on a modern desktop CPU.

use fuel::lazy::{LlamaModel, LlamaTokenizer, SamplingStrategy};
use std::io::Write;
use std::time::Instant;

const DEFAULT_MODEL: &str = "Qwen/Qwen2-0.5B-Instruct";
const DEFAULT_PROMPT: &str = "The quick brown fox";
const DEFAULT_MAX_NEW: usize = 32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    eprintln!("=== fuel qwen-lazy ===");
    eprintln!("Model:  {model_id}");
    eprintln!("Prompt: {prompt:?}");
    eprintln!("Max new tokens: {max_new}");
    eprintln!();

    eprint!("Downloading + loading model weights... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let model = LlamaModel::from_hub(&model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  config: dim={}  layers={}  heads={}  kv_heads={}  vocab={}  rope_base={}",
        model.config.dim,
        model.config.n_layers,
        model.config.n_heads,
        model.config.n_kv_heads,
        model.config.vocab_size,
        model.config.rope_base,
    );
    let layer0 = &model.weights.layers[0];
    eprintln!(
        "  qkv biases present: q={} k={} v={}",
        layer0.attn_q_bias.is_some(),
        layer0.attn_k_bias.is_some(),
        layer0.attn_v_bias.is_some(),
    );

    eprint!("Loading tokenizer...             ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let tokenizer = LlamaTokenizer::from_hub(&model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());

    let prompt_tokens = tokenizer.encode(&prompt, true)?;
    eprintln!("Prompt tokens: {}", prompt_tokens.len());
    eprintln!();

    eprintln!("Generating...");
    eprintln!("---");
    print!("{prompt}");
    std::io::stdout().flush().ok();
    let mut streamed: Vec<u32> = prompt_tokens.clone();
    let mut printed_text = tokenizer.decode(&streamed, true)?;
    let t0 = Instant::now();
    let output_tokens = model.generate_streaming(
        &prompt_tokens,
        max_new,
        SamplingStrategy::Temperature { temp: 0.8, seed: 42 },
        tokenizer.eos_id(),
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

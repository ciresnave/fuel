// End-to-end Gemma 2 runner. Exercises all the Gemma 2-specific
// graph-layer features: GeGLU activation, embedding scaling,
// attention + final logit softcapping, 4 norms per layer,
// offset-RmsNorm, decoupled head_dim, sliding-window attention.
//
// USAGE
//
//     cargo run --release --bin gemma-lazy
//     cargo run --release --bin gemma-lazy -- [MODEL_ID] [PROMPT] [MAX_NEW_TOKENS]
//
// Defaults to `google/gemma-2-2b-it` (instruction-tuned Gemma 2 2B).
// This model is gated — accept the license at
// https://huggingface.co/google/gemma-2-2b-it and set HF_TOKEN first.

use fuel::lazy::{Gemma2Model, LlamaTokenizer, SamplingStrategy};
use std::io::Write;
use std::time::Instant;

const DEFAULT_MODEL: &str = "google/gemma-2-2b-it";
const DEFAULT_PROMPT: &str = "The meaning of life is";
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

    eprintln!("=== fuel gemma-lazy ===");
    eprintln!("Model:  {model_id}");
    eprintln!("Prompt: {prompt:?}");
    eprintln!("Max new tokens: {max_new}");
    eprintln!();

    eprint!("Downloading + loading model weights... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let model = Gemma2Model::from_hub(&model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  config: dim={}  layers={}  heads={}  kv_heads={}  head_dim={}  vocab={}",
        model.config.dim,
        model.config.n_layers,
        model.config.n_heads,
        model.config.n_kv_heads,
        model.config.head_dim,
        model.config.vocab_size,
    );
    eprintln!(
        "  attn_softcap={:?}  final_softcap={:?}  sliding_window={:?}",
        model.config.attn_logit_softcapping,
        model.config.final_logit_softcapping,
        model.config.sliding_window,
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

    // Gemma 2 doesn't have generate_streaming on its own model struct
    // (only LlamaModel does), so we use the manual loop here. This
    // also serves as a demonstration of driving forward() directly.
    print!("{prompt}");
    std::io::stdout().flush().ok();
    let mut all_tokens = prompt_tokens.clone();
    let mut printed_text = tokenizer.decode(&all_tokens, true)?;
    let mut rng_state: u64 = 42;
    let t0 = Instant::now();
    for _ in 0..max_new {
        let logits = model.forward(&all_tokens, 0);
        let last_pos = all_tokens.len() - 1;
        let last_logits = logits
            .slice(1, last_pos, 1)
            .reshape(fuel::Shape::from_dims(&[model.config.vocab_size]))
            .realize_f32();
        let next = fuel::lazy::sample_logits(
            &last_logits,
            SamplingStrategy::Temperature { temp: 0.7, seed: 42 },
            &mut rng_state,
        );
        all_tokens.push(next);
        if let Ok(full) = tokenizer.decode(&all_tokens, true) {
            if let Some(delta) = full.strip_prefix(&printed_text) {
                print!("{delta}");
                std::io::stdout().flush().ok();
            }
            printed_text = full;
        }
        if let Some(eos) = tokenizer.eos_id() {
            if next == eos {
                break;
            }
        }
    }
    let elapsed = t0.elapsed();
    println!();

    let new_tokens = all_tokens.len().saturating_sub(prompt_tokens.len());
    eprintln!();
    eprintln!("---");
    eprintln!(
        "Generated {new_tokens} tokens in {:.2?} ({:.2}s/token avg)",
        elapsed,
        elapsed.as_secs_f64() / new_tokens.max(1) as f64,
    );

    Ok(())
}

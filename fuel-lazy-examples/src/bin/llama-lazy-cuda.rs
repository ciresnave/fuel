// CUDA GPU-accelerated LLaMA runner.
//
// USAGE
//
//     cargo run --release --bin llama-lazy-cuda --features cuda
//     cargo run --release --bin llama-lazy-cuda --features cuda -- [MODEL_ID] [PROMPT] [MAX_NEW_TOKENS]
//
// Requires: CUDA toolkit, NVIDIA GPU, and MSVC build tools (for kernel
// compilation on first build).

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires the `cuda` feature.");
    eprintln!("Run: cargo run --release --bin llama-lazy-cuda --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fuel::lazy::{LlamaModel, LlamaTokenizer, SamplingStrategy};
    use std::io::Write;
    use std::time::Instant;

    // Tracing: set FUEL_TRACE=1 to write a Chrome-compatible trace file.
    // Open the resulting trace.json in chrome://tracing or ui.perfetto.dev
    // to see a flame chart of every op, const upload, and D2H transfer.
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

    const DEFAULT_MODEL: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
    const DEFAULT_PROMPT: &str = "Once upon a time";
    const DEFAULT_MAX_NEW: usize = 32;

    let args: Vec<String> = std::env::args().collect();
    let model_id = args.get(1).cloned().unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let prompt = args.get(2).cloned().unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let max_new: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_MAX_NEW);

    eprintln!("=== fuel llama-lazy-cuda ===");
    eprintln!("Model:  {model_id}");
    eprintln!("Prompt: {prompt:?}");
    eprintln!("Max new tokens: {max_new}");
    eprintln!();

    eprint!("Initializing CUDA device... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let cuda_device = fuel::CudaDevice::new(0)?;
    let backend = fuel_graph_cuda::CudaBackend::new(cuda_device);
    let mut executor = fuel_graph_executor::GraphExecutor::new(backend);
    eprintln!("done in {:.2?}", t0.elapsed());

    eprint!("Downloading + loading model weights... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let model = LlamaModel::from_hub(&model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  config: dim={}  layers={}  heads={}  kv_heads={}  vocab={}  rope_base={}",
        model.config.dim, model.config.n_layers, model.config.n_heads,
        model.config.n_kv_heads, model.config.vocab_size, model.config.rope_base,
    );

    eprint!("Loading tokenizer...             ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let tokenizer = LlamaTokenizer::from_hub(&model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());

    let prompt_tokens = tokenizer.encode(&prompt, true)?;
    eprintln!("Prompt tokens: {}", prompt_tokens.len());
    eprintln!();

    eprintln!("Generating on CUDA...");
    eprintln!("---");
    print!("{prompt}");
    std::io::stdout().flush().ok();
    let mut streamed: Vec<u32> = prompt_tokens.clone();
    let mut printed_text = tokenizer.decode(&streamed, true)?;
    let t0 = Instant::now();
    let output_tokens = model.generate_streaming_cuda(
        &prompt_tokens,
        max_new,
        SamplingStrategy::Temperature { temp: 0.8, seed: 42 },
        tokenizer.eos_id(),
        &mut executor,
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

    Ok(())
}

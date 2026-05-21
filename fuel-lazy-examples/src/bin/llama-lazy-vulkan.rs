// Vulkan GPU-accelerated LLaMA runner.
//
// USAGE
//
//     cargo run --release --bin llama-lazy-vulkan --features vulkan
//     cargo run --release --bin llama-lazy-vulkan --features vulkan -- [MODEL_ID] [PROMPT] [MAX_NEW_TOKENS]
//
// DEVICE SELECTION
//
//     --list-devices     List all Vulkan physical devices and exit
//     --device=N         Use device index N (default: prefer discrete GPU)
//     --device=name      Match device by substring (e.g. "4070", "AMD")
//
// Requires: Vulkan-capable GPU with up-to-date drivers.

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("This binary requires the `vulkan` feature.");
    eprintln!("Run: cargo run --release --bin llama-lazy-vulkan --features vulkan");
    std::process::exit(1);
}

#[cfg(feature = "vulkan")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fuel::lazy::{LlamaModel, LlamaTokenizer, SamplingStrategy};
    use fuel_graph_executor::GraphExecutor;
    use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};
    use std::io::Write;
    use std::time::Instant;

    // Tracing support.
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

    // --list-devices
    if args.iter().any(|a| a == "--list-devices") {
        eprintln!("Vulkan physical devices:");
        for (idx, name, dtype) in VulkanBackend::list_devices()? {
            eprintln!("  [{idx}] {name} ({dtype})");
        }
        return Ok(());
    }

    // Parse device selection.
    let selection = args.iter()
        .find(|a| a.starts_with("--device="))
        .map(|a| {
            let val = &a["--device=".len()..];
            match val.parse::<usize>() {
                Ok(idx) => DeviceSelection::Index(idx),
                Err(_) => DeviceSelection::ByName(val.to_string()),
            }
        })
        .unwrap_or(DeviceSelection::PreferDiscrete);

    // Filter out our flags for positional args.
    let positional: Vec<&str> = args.iter()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .collect();

    let model_id = positional.first().copied().unwrap_or("TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    let prompt = positional.get(1).copied().unwrap_or("Once upon a time");
    let max_new: usize = positional.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);

    eprintln!("=== fuel llama-lazy-vulkan ===");
    eprintln!("Model:  {model_id}");
    eprintln!("Prompt: {prompt:?}");
    eprintln!("Max new tokens: {max_new}");
    eprintln!();

    eprint!("Initializing Vulkan device... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let backend = VulkanBackend::with_selection(selection)?;
    eprintln!("done in {:.2?} — {}", t0.elapsed(), backend.device_name);

    let mut executor = GraphExecutor::new(backend);

    eprint!("Downloading + loading model weights... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let model = LlamaModel::from_hub(model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  config: dim={}  layers={}  heads={}  kv_heads={}  vocab={}",
        model.config.dim, model.config.n_layers, model.config.n_heads,
        model.config.n_kv_heads, model.config.vocab_size,
    );

    eprint!("Loading tokenizer...             ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let tokenizer = LlamaTokenizer::from_hub(model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());

    let prompt_tokens = tokenizer.encode(prompt, true)?;
    eprintln!("Prompt tokens: {}", prompt_tokens.len());
    eprintln!();

    eprintln!("Generating on Vulkan (device-resident KV cache)...");
    eprintln!("---");
    // Stream through the backend-agnostic generate_streaming_gpu_on.
    // `KVCache<VulkanBackend>` keeps K/V on the GPU between decode
    // steps — no D2H/H2D round-trip per token.
    print!("{prompt}");
    std::io::stdout().flush().ok();
    let mut streamed = prompt_tokens.clone();
    let mut printed_text = tokenizer.decode(&streamed, true)?;
    let t0 = Instant::now();
    let output_tokens = model.generate_streaming_gpu_on(
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

    eprintln!();
    eprintln!("Vulkan op stats (host-side submit time):");
    for (name, s) in executor.backend.op_stats_snapshot() {
        let avg_us = if s.count == 0 { 0 } else { (s.total_ns / s.count as u128) / 1000 };
        eprintln!(
            "  {name:20} count={:>7} total={:>7}ms avg={avg_us}us",
            s.count,
            s.total_ns / 1_000_000
        );
    }

    Ok(())
}

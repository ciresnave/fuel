// Vulkan GPU-accelerated Phi-2 runner.
//
// Phi-2 (microsoft/phi-2, 2.7B params) uses a different architecture
// from LLaMA:
//   - LayerNorm with bias (not RMSNorm)
//   - Standard MLP: fc1 → GELU → fc2 (not SwiGLU)
//   - Parallel attention + MLP residual structure
//   - Partial RoPE (first 32 of 80 head_dim entries rotate)
//
// USAGE
//     cargo run --release --bin phi-lazy-vulkan --features vulkan
//     cargo run --release --bin phi-lazy-vulkan --features vulkan -- [MODEL_ID] [PROMPT] [MAX_NEW_TOKENS]
//
// Defaults to microsoft/phi-2. ~5.4 GB at bf16 — tight on 8 GB cards.

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("This binary requires the `vulkan` feature.");
    eprintln!("Run: cargo run --release --bin phi-lazy-vulkan --features vulkan");
    std::process::exit(1);
}

#[cfg(feature = "vulkan")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fuel::lazy::{LlamaTokenizer, PhiModel, SamplingStrategy};
    use fuel::{DType, Device};
    use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};
    use std::io::Write;
    use std::time::Instant;

    let _trace_guard = if std::env::var("FUEL_TRACE").is_ok() {
        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file("trace.json")
            .include_args(true)
            .build();
        use tracing_subscriber::prelude::*;
        tracing_subscriber::registry().with(chrome_layer).init();
        eprintln!("Tracing enabled → trace.json");
        Some(guard)
    } else {
        None
    };

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--list-devices") {
        eprintln!("Vulkan physical devices:");
        for (idx, name, dtype) in VulkanBackend::list_devices()? {
            eprintln!("  [{idx}] {name} ({dtype})");
        }
        return Ok(());
    }

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

    // --gguf=<path> loads a Q4_0-quantized Phi-2 from a local GGUF file
    // (e.g. TheBloke/phi-2-GGUF's phi-2.Q4_0.gguf). When set, tokenizer
    // still loads from the safetensors repo since GGUF tokenizer extract
    // is a separate plumbing exercise.
    let gguf_path: Option<String> = args.iter()
        .find(|a| a.starts_with("--gguf="))
        .map(|a| a["--gguf=".len()..].to_string());
    let tokenizer_repo: String = args.iter()
        .find(|a| a.starts_with("--tokenizer="))
        .map(|a| a["--tokenizer=".len()..].to_string())
        .unwrap_or_else(|| "microsoft/phi-2".to_string());

    let positional: Vec<&str> = args.iter()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .collect();

    let model_id = positional.first().copied().unwrap_or("microsoft/phi-2");
    let prompt = positional.get(1).copied().unwrap_or("Once upon a time");
    let max_new: usize = positional.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);

    eprintln!("=== fuel phi-lazy-vulkan ===");
    eprintln!("Model:  {model_id}");
    eprintln!("Prompt: {prompt:?}");
    eprintln!("Max new tokens: {max_new}");
    eprintln!();

    eprint!("Initializing Vulkan device... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let backend = VulkanBackend::with_selection(selection)?;
    eprintln!("done in {:.2?} — {}", t0.elapsed(), backend.device_name);

    // Phase 7.6 step 9c E.3.3/E.3.4: the pipelined executor handles
    // backend dispatch via `Device`; no GraphBackend executor needed.
    // KvCache + InferenceContext live on this device.
    let device: Device = backend.into();

    let t0 = Instant::now();
    let model = match &gguf_path {
        Some(p) => {
            eprint!("Loading Phi-2 from GGUF ({p})... ");
            std::io::stderr().flush().ok();
            PhiModel::from_gguf(p)?
        }
        None => {
            eprint!("Downloading + loading model weights... ");
            std::io::stderr().flush().ok();
            PhiModel::from_hub(model_id)?
        }
    };
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  config: dim={}  layers={}  heads={}  ffn={}  rotary_dim={}  vocab={}",
        model.config.dim, model.config.n_layers, model.config.n_heads,
        model.config.ffn_dim, model.config.rotary_dim, model.config.vocab_size,
    );

    eprint!("Loading tokenizer...             ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let tokenizer = LlamaTokenizer::from_hub(&tokenizer_repo)?;
    eprintln!("done in {:.2?}", t0.elapsed());

    let prompt_tokens = tokenizer.encode(prompt, true)?;
    eprintln!("Prompt tokens: {}", prompt_tokens.len());
    eprintln!();

    eprintln!("Generating on Vulkan (device-resident KV cache)...");
    eprintln!("---");
    print!("{prompt}");
    std::io::stdout().flush().ok();
    let mut streamed = prompt_tokens.clone();
    let mut printed_text = tokenizer.decode(&streamed, true)?;
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

    Ok(())
}

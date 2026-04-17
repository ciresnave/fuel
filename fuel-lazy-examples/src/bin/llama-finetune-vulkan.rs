// TinyLlama fine-tuning example — trains only the output head (lm_head).
//
// USAGE
//
//     cargo run --release --bin llama-finetune-vulkan --features vulkan
//
// Demonstrates Fuel's training loop with a real pretrained model.
// Fine-tunes ONLY the lm_head (output projection) while keeping
// all 22 transformer layers frozen as constants. This is the
// simplest possible fine-tune — LoRA, adapter layers, and full-
// parameter training all build on the same TrainState + backward +
// optimizer machinery shown here.
//
// WHAT IT DOES
//
// 1. Downloads TinyLlama from HuggingFace (cached after first run)
// 2. Extracts the lm_head weight as a trainable Parameter
// 3. Runs 20 steps of AdamW on a single training sentence
// 4. Prints per-step loss to show convergence
//
// The training data is deliberately tiny so the model overfits
// quickly — that's the point. Real fine-tuning would use a larger
// dataset, more steps, possibly LoRA adapters, and gradient
// checkpointing for larger models.

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("This binary requires the `vulkan` feature.");
    eprintln!("Run: cargo run --release --bin llama-finetune-vulkan --features vulkan");
    std::process::exit(1);
}

#[cfg(feature = "vulkan")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fuel::lazy::{LlamaModel, LlamaTokenizer, WeightStorage};
    use fuel::train::{self, OptimizerConfig, Parameter, TrainState};
    use fuel_graph_executor::GraphExecutor;
    use fuel_graph_vulkan::{DeviceSelection, VulkanBackend};
    use fuel::Shape;
    use std::io::Write;
    use std::sync::Arc;
    use std::time::Instant;

    eprintln!("=== fuel llama-finetune-vulkan ===\n");

    // ---- Load model + tokenizer ----
    let model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
    eprint!("Loading model... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let model = LlamaModel::from_hub(model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    let cfg = &model.config;
    eprintln!("  dim={} layers={} vocab={}", cfg.dim, cfg.n_layers, cfg.vocab_size);

    eprint!("Loading tokenizer... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let tokenizer = LlamaTokenizer::from_hub(model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());

    // ---- Set up Vulkan ----
    eprint!("Initializing Vulkan... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let backend = VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)?;
    eprintln!("done in {:.2?} — {}", t0.elapsed(), backend.device_name);
    let mut executor = GraphExecutor::new(backend);

    // ---- Extract trainable lm_head ----
    // Everything else stays frozen — the forward pass uses
    // forward_hidden() which runs all 22 layers using const nodes,
    // then we apply the trainable lm_head manually.
    let lm_head_data: Vec<f32> = match &model.weights.output {
        WeightStorage::F32(a) => a.to_vec(),
        WeightStorage::BF16(a) => a.iter().map(|x| x.to_f32()).collect(),
    };
    let params = vec![
        Parameter::new_f32(
            "lm_head",
            Shape::from_dims(&[cfg.dim, cfg.vocab_size]),
            lm_head_data,
        ),
    ];
    let mut state = TrainState::new(&params, &mut executor, OptimizerConfig::adam_w(1e-4))?;

    // ---- Training data ----
    let train_text = "The meaning of life is to find happiness in helping others grow.";
    let train_tokens = tokenizer.encode(train_text, true)?;
    let seq = train_tokens.len();
    eprintln!("\nTraining on {seq} tokens: \"{train_text}\"");
    eprintln!("  (next-token prediction: input = tokens[0..n-1], target = tokens[1..n])\n");

    // Next-token prediction: input[t] predicts target[t] = tokens[t+1].
    let input_ids: Vec<u32> = train_tokens[..seq - 1].to_vec();
    let target_ids: Vec<u32> = train_tokens[1..seq].to_vec();
    let input_seq = input_ids.len();

    // One-hot target [input_seq, vocab_size].
    let mut target_onehot = vec![0.0f32; input_seq * cfg.vocab_size];
    for (i, &tid) in target_ids.iter().enumerate() {
        target_onehot[i * cfg.vocab_size + tid as usize] = 1.0;
    }
    let target_arc: Arc<[f32]> = target_onehot.into();

    // ---- Training loop ----
    let n_steps = 20;
    eprintln!("Training ({n_steps} steps, AdamW lr=1e-4):");
    let t0 = Instant::now();
    let model_ref = &model;
    for step in 0..n_steps {
        let tgt = target_arc.clone();
        let ids = input_ids.clone();
        let vocab = cfg.vocab_size;
        let dim = cfg.dim;
        let iseq = input_seq;

        let loss = state.step(&mut executor, move |_graph, params| {
            let lm_head = &params["lm_head"];

            // Forward: all 22 frozen layers → hidden state.
            // Pass lm_head as the graph anchor so all nodes land on
            // the same graph the parameters live on.
            let hidden = model_ref.forward_hidden(&ids, 0, lm_head);
            // [1, input_seq, dim] → [input_seq, dim]
            let hidden = hidden.reshape(Shape::from_dims(&[iseq, dim]));

            // Trainable output head → logits [input_seq, vocab_size]
            let logits = hidden.matmul(lm_head);

            // Cross-entropy loss against next-token targets.
            let target = lm_head.const_f32_like(tgt, Shape::from_dims(&[iseq, vocab]));
            train::loss::cross_entropy_with_logits(&logits, &target)
        })?;

        eprintln!("  step {step:>2}: loss = {loss:.4}");
    }
    let elapsed = t0.elapsed();
    eprintln!("\nDone in {elapsed:.2?} ({:.1} steps/sec)", n_steps as f64 / elapsed.as_secs_f64());

    // ---- Inspect result ----
    let final_lm_head = state.param_to_host("lm_head", &executor)?;
    let orig_lm_head: Vec<f32> = match &model.weights.output {
        WeightStorage::F32(a) => a.to_vec(),
        WeightStorage::BF16(a) => a.iter().map(|x| x.to_f32()).collect(),
    };
    let max_delta = final_lm_head.iter().zip(&orig_lm_head)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Max weight delta from pretrained: {max_delta:.6}");

    Ok(())
}

// End-to-end BERT encoder runner built on the LazyTensor path. Proves
// Fuel's Phase 6a lazy frontend handles encoder-only models with
// absolute position embeddings, bidirectional attention, and GELU
// FFNs — architecturally distinct from the LLaMA / Qwen2 decoder-only
// anchors already landed.
//
// USAGE
//
//     cargo run --release --bin bert-lazy
//     cargo run --release --bin bert-lazy -- [MODEL_ID] [PROMPT]
//
// Defaults:
//     MODEL_ID = bert-base-uncased
//     PROMPT   = "The quick brown fox jumps over the lazy dog."
//
// Prints the first 8 values of the `[CLS]` hidden state plus timing
// stats. A real consumer (sentence embedding, classifier, MLM) layers
// a task head on top of the returned `[1, seq, hidden]` tensor; this
// binary stops at the encoder output so the diagnostic stays short.

use fuel::lazy_bert::{BertModel, BertTokenizer};
use std::io::Write;
use std::time::Instant;

const DEFAULT_MODEL: &str = "bert-base-uncased";
const DEFAULT_PROMPT: &str = "The quick brown fox jumps over the lazy dog.";

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

    eprintln!("=== fuel bert-lazy ===");
    eprintln!("Model:  {model_id}");
    eprintln!("Prompt: {prompt:?}");
    eprintln!();

    eprint!("Downloading + loading model weights... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let model = BertModel::from_hub(&model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  config: hidden={}  layers={}  heads={}  ff={}  vocab={}  max_pos={}",
        model.config.hidden_size,
        model.config.num_hidden_layers,
        model.config.num_attention_heads,
        model.config.intermediate_size,
        model.config.vocab_size,
        model.config.max_position_embeddings,
    );

    eprint!("Loading tokenizer...                    ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let tokenizer = BertTokenizer::from_hub(&model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  [CLS]={:?}  [SEP]={:?}  [PAD]={:?}",
        tokenizer.cls_id(),
        tokenizer.sep_id(),
        tokenizer.pad_id(),
    );

    let token_ids = tokenizer.encode(&prompt, true)?;
    eprintln!("Prompt tokens ({}): {:?}", token_ids.len(), &token_ids);
    eprintln!();

    eprintln!("Running forward pass...");
    let t0 = Instant::now();
    let hidden = model.forward(&token_ids)?;
    let flat = hidden.realize_f32();
    let elapsed = t0.elapsed();

    let h = model.config.hidden_size;
    let seq = token_ids.len();
    assert_eq!(flat.len(), seq * h);
    let cls_hidden = &flat[..h];

    eprintln!(
        "Forward done in {:.2?}  ({:.4?}/token)",
        elapsed,
        elapsed / seq.max(1) as u32,
    );
    eprintln!();
    println!("[CLS] hidden state (first 8 of {h}):");
    for (i, v) in cls_hidden.iter().take(8).enumerate() {
        println!("  [{i:>2}] = {v:+.6}");
    }
    println!();
    println!("‖[CLS]‖₂ = {:.4}", l2_norm(cls_hidden));
    println!("Per-token ‖h_i‖₂ (first 4 tokens):");
    for i in 0..seq.min(4) {
        let tok_hidden = &flat[i * h..(i + 1) * h];
        println!("  tok[{i}] = {:.4}", l2_norm(tok_hidden));
    }

    Ok(())
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

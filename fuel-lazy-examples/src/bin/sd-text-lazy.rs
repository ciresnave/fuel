// SD 1.5 text encoder runner. First component of Phase 6a anchor #6.
//
// USAGE
//
//     cargo run --release --bin sd-text-lazy
//     cargo run --release --bin sd-text-lazy -- [REPO_ID] "PROMPT"
//
// Defaults:
//     REPO_ID = stable-diffusion-v1-5/stable-diffusion-v1-5
//     PROMPT  = "a photo of an astronaut riding a horse on mars"
//
// SCOPE
//
// This binary runs SD 1.5's CLIP-ViT-L/14 text encoder on a prompt
// string. Output is the `[1, 77, 768]` hidden-state tensor the UNet
// cross-attends into at every down/mid/up block during diffusion.
// The binary prints summary statistics (norm of `[EOS]` token,
// per-token norms) so you can sanity-check the conditioning before
// handing it off to a UNet that doesn't exist yet in Fuel's lazy
// graph.

use fuel::lazy_sd_text_encoder::{SdTextEncoder, SdTextTokenizer};
use std::io::Write;
use std::time::Instant;

const DEFAULT_REPO: &str = "stable-diffusion-v1-5/stable-diffusion-v1-5";
const DEFAULT_PROMPT: &str = "a photo of an astronaut riding a horse on mars";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let repo_id = args.get(1).cloned().unwrap_or_else(|| DEFAULT_REPO.to_string());
    let prompt = args.get(2).cloned().unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    eprintln!("=== fuel sd-text-lazy ===");
    eprintln!("Repo:   {repo_id}");
    eprintln!("Prompt: {prompt:?}");
    eprintln!();

    eprint!("Downloading + loading text encoder weights... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let model = SdTextEncoder::from_hub(&repo_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  hidden={} layers={} heads={} max_pos={} vocab={}",
        model.config.hidden_size,
        model.config.num_hidden_layers,
        model.config.num_attention_heads,
        model.config.max_position_embeddings,
        model.config.vocab_size,
    );

    eprint!("Loading tokenizer...                       ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let tokenizer = SdTextTokenizer::from_hub_with_config(&repo_id, &model.config)?;
    eprintln!("done in {:.2?}", t0.elapsed());

    let tokens = tokenizer.encode_padded(&prompt)?;
    let nonpad = tokens.iter().take_while(|&&t| t != model.config.pad_token_id).count();
    eprintln!("Prompt tokens ({} non-pad of {} padded): {:?}", nonpad, tokens.len(), &tokens[..nonpad.min(20)]);
    eprintln!();

    eprintln!("Running forward pass...");
    let t0 = Instant::now();
    let hidden = model.forward(&tokens);
    let flat = hidden.realize_f32();
    eprintln!("Forward done in {:.2?}", t0.elapsed());
    eprintln!();

    let h = model.config.hidden_size;
    let seq = model.config.max_position_embeddings;
    assert_eq!(flat.len(), seq * h);

    // Per-token L2 norms of the first 8 tokens + the EOS position.
    println!("Per-token ‖h_i‖₂ (first 8 tokens):");
    for i in 0..8.min(seq) {
        let row = &flat[i * h..(i + 1) * h];
        println!("  tok[{i}] = {:.4}", l2(row));
    }
    let eos_pos = if nonpad == 0 { 0 } else { nonpad - 1 };
    let eos_row = &flat[eos_pos * h..(eos_pos + 1) * h];
    println!();
    println!("‖[EOS]‖₂ (position {eos_pos}) = {:.4}", l2(eos_row));
    println!("First 8 of the [EOS] hidden state:");
    for (i, v) in eos_row.iter().take(8).enumerate() {
        println!("  [{i:>2}] = {v:+.6}");
    }
    Ok(())
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#![allow(dead_code)]
// https://huggingface.co/facebook/musicgen-small/tree/main
// https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/models/musicgen/modeling_musicgen.py
//
// Lazy-graph migration: this binary now uses `fuel::lazy_musicgen`,
// which bundles the MusicGen decoder + a built-in text adapter. The
// historic eager binary (parked alongside this file in
// `musicgen_model.rs`) only loaded the model and ran the T5 text
// encoder on the prompt as a smoke test; this lazy version mirrors
// that scope — load weights, run the decoder forward over a tiny
// placeholder audio-token window — without pulling the full T5 +
// EnCodec stack.
//
// TODO: Add an offline mode.
// TODO: Add a KV cache.
// TODO: Wire a real T5 encoder via `forward_with_encoder_states`.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};

use fuel::lazy_musicgen::{MusicGenConfig, MusicGenModel, MusicGenWeights};
use fuel::safetensors::MmapedSafetensors;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model weight file, in safetensor format.
    #[arg(long)]
    model: Option<String>,

    /// The tokenizer config.
    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(
        long,
        default_value = "90s rock song with loud guitars and heavy drums"
    )]
    prompt: String,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();
    // `--cpu` is preserved for parity; the lazy realize path lives on
    // CPU by default in this binary.
    let _ = args.cpu;

    let tokenizer = match args.tokenizer {
        Some(tokenizer) => std::path::PathBuf::from(tokenizer),
        None => Api::new()?
            .model("facebook/musicgen-small".to_string())
            .get("tokenizer.json")?,
    };
    let mut tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;

    let model_path = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => Api::new()?
            .repo(Repo::with_revision(
                "facebook/musicgen-small".to_string(),
                RepoType::Model,
                "refs/pr/13".to_string(),
            ))
            .get("model.safetensors")?,
    };

    let cfg = MusicGenConfig::musicgen_small();
    let st = unsafe { MmapedSafetensors::multi(&[model_path]) }
        .map_err(|e| E::msg(format!("mmap: {e}")))?;
    let weights = MusicGenWeights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("weights: {e}")))?;
    let model = MusicGenModel {
        config: cfg.clone(),
        weights,
    };

    let tokens = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    println!("tokens: {tokens:?}");

    // Smoke-test forward: feed the prompt tokens as the (built-in)
    // text-adapter input and run the decoder over a one-step
    // placeholder audio-token window (zeros across all codebooks),
    // matching the eager stub's "just run a forward and print" shape.
    let seq_len: usize = 1;
    let audio_tokens: Vec<u32> = vec![0_u32; cfg.num_codebooks * seq_len];
    let logits = model
        .forward(&tokens, &audio_tokens, 0)
        .map_err(|e| E::msg(format!("forward: {e}")))?;
    let logits_shape = logits.shape();
    println!("logits shape: {:?}", logits_shape.dims());
    let realized = logits.realize_f32();
    println!("logits first 8 = {:?}", &realized[..realized.len().min(8)]);

    Ok(())
}

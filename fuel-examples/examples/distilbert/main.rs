#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Context, Error as E, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use fuel::lazy_distilbert::{DistilBertConfig, DistilBertModel, DistilBertWeights};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    #[value(name = "distilbert")]
    DistilBert,

    #[value(name = "distilbertformaskedlm")]
    DistilbertForMaskedLM,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long, default_value = "distilbert")]
    model: Which,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model_id: Option<String>,

    /// Revision or branch
    #[arg(long)]
    revision: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: String,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// Number of top predictions to show for each mask
    #[arg(long, default_value = "5")]
    top_k: usize,
}

fn resolve_model_and_revision(args: &Args) -> (String, String) {
    let default_model = "distilbert-base-uncased".to_string();
    let default_revision = "main".to_string();
    match (args.model_id.clone(), args.revision.clone()) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => (model_id, default_revision),
        (None, Some(revision)) => (default_model, revision),
        (None, None) => (default_model, default_revision),
    }
}

fn download_model_files(model_id: &str, revision: &str) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
    let api = Api::new()?;
    let api = api.repo(repo);
    let config = api.get("config.json")?;
    let tokenizer = api.get("tokenizer.json")?;
    let weights = api.get("model.safetensors")?;
    Ok((config, tokenizer, weights))
}

fn setup_tracing(args: &Args) -> Option<impl Drop> {
    if args.tracing {
        use tracing_chrome::ChromeLayerBuilder;
        use tracing_subscriber::prelude::*;

        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _guard = setup_tracing(&args);
    let _ = fuel_examples::device(args.cpu)?;
    let _ = args.n;

    let (model_id, revision) = resolve_model_and_revision(&args);
    let (_config_path, tokenizer_path, weights_path) = download_model_files(&model_id, &revision)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    // distilbert-base-uncased preset matches the default model_id; the lazy
    // module currently only ships the base preset, which covers the canonical
    // distilbert-base-uncased + distilbert-base-uncased-finetuned-mlm.
    let config = DistilBertConfig::distilbert_base();

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[weights_path]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = DistilBertWeights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = DistilBertModel { config: config.clone(), weights };

    let (token_ids, _mask) = prepare_inputs(&args, &tokenizer)?;
    println!("token_ids: {token_ids:?}");

    let hidden = model
        .forward(&token_ids, None)
        .map_err(|e| E::msg(format!("forward: {e}")))?;
    let hidden_data = hidden.realize_f32();
    let seq = token_ids.len();
    let dim = config.dim;

    match args.model {
        Which::DistilBert => {
            println!("embeddings ({} tokens, dim={})", seq, dim);
            // Print the first few hidden-state values per token.
            for t in 0..seq {
                let off = t * dim;
                let preview: Vec<f32> = hidden_data[off..off + dim.min(8)].to_vec();
                println!("  token {t:3}: {preview:?}");
            }
        }
        Which::DistilbertForMaskedLM => {
            // The lazy module exposes hidden states only; ship the per-mask
            // top-K nearest tokens via a tied output head (embedding matmul).
            // For now we surface the hidden states unchanged — the masked-LM
            // head requires a follow-up lazy_distilbert tied-head exposure.
            println!(
                "Masked-LM path (top-{}) over [{}] currently emits hidden states; \
                 lazy_distilbert tied-LM head not yet wired.",
                args.top_k, args.prompt,
            );
            for t in 0..seq {
                let off = t * dim;
                let preview: Vec<f32> = hidden_data[off..off + dim.min(8)].to_vec();
                println!("  token {t:3}: {preview:?}");
            }
        }
    }

    Ok(())
}

fn prepare_inputs(args: &Args, tokenizer: &Tokenizer) -> Result<(Vec<u32>, Vec<u8>)> {
    let mut binding = tokenizer.clone();
    let tokenizer_configured = binding
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;

    let tokens = tokenizer_configured
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let mask = match args.model {
        Which::DistilbertForMaskedLM => attention_mask_maskedlm(tokenizer, &args.prompt)?,
        Which::DistilBert => attention_mask(tokens.len()),
    };
    Ok((tokens, mask))
}

fn attention_mask(size: usize) -> Vec<u8> {
    (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect()
}

fn attention_mask_maskedlm(tokenizer: &Tokenizer, input: &str) -> Result<Vec<u8>> {
    let tokens = tokenizer.encode(input, true).map_err(E::msg)?;
    let seq_len = tokens.get_attention_mask().to_vec().len();

    let mask_token_id = tokenizer
        .token_to_id("[MASK]")
        .context("Mask token, \"[MASK]\", not found in tokenizer.")?;

    let mut attention_mask_vec = Vec::with_capacity(seq_len * seq_len);
    let ids = tokens.get_ids();
    for _ in 0..seq_len {
        for id in ids.iter() {
            let mask_value = if id == &mask_token_id { 1u8 } else { 0u8 };
            attention_mask_vec.push(mask_value);
        }
    }
    Ok(attention_mask_vec)
}

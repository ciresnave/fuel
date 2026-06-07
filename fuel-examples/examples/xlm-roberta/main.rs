//! XLM-RoBERTa example wired against the lazy-graph encoder.
//!
//! Supports three task heads:
//!
//!   * `fill-mask`     — `XlmrForMaskedLM`, argmax at the `<mask>` token
//!     and decode the predicted id.
//!   * `reranker`      — `XlmrForSequenceClassification` with
//!     `num_labels = 1`; sigmoid of the single logit gives a relevance
//!     score.
//!   * `classification` — `XlmrForSequenceClassification` with the
//!     model's native label count; softmax + argmax over labels.
//!
//! Only the happy path is reproduced from the historical eager binary:
//! batch size 1, no padding inside the prompt, F32 weights, default
//! `xlm-roberta-base` config.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Context, Error as E, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use fuel::lazy_xlm_roberta::{
    XlmrConfig, XlmrForMaskedLM, XlmrForSequenceClassification,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Task {
    /// Predict the token at the `<mask>` position.
    FillMask,
    /// Single-logit relevance score (BAAI/bge-reranker-* checkpoints).
    Reranker,
    /// Multi-label softmax classification (e.g. xlmr-formality).
    Classification,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Which head to attach on top of the base encoder.
    #[arg(long, value_enum, default_value_t = Task::FillMask)]
    task: Task,

    /// HuggingFace model id. Sensible per-task defaults are used when
    /// omitted.
    #[arg(long)]
    model_id: Option<String>,

    /// Branch / revision to download from the hub.
    #[arg(long, default_value = "main")]
    revision: String,

    /// Prompt to run through the model. Task-specific defaults are used
    /// when omitted.
    #[arg(long)]
    prompt: Option<String>,

    /// For `classification`: number of output labels. Defaults to 2.
    #[arg(long, default_value_t = 2)]
    num_labels: usize,

    /// Run on CPU rather than GPU (only CPU is wired in the lazy path
    /// today; flag kept for parity with sibling examples).
    #[arg(long)]
    cpu: bool,
}

fn default_model_id(task: Task) -> &'static str {
    match task {
        Task::FillMask       => "FacebookAI/xlm-roberta-base",
        Task::Reranker       => "BAAI/bge-reranker-base",
        Task::Classification => "s-nlp/xlmr_formality_classifier",
    }
}

fn default_prompt(task: Task) -> &'static str {
    match task {
        Task::FillMask       => "Hello I'm a <mask> model.",
        Task::Reranker       => "what is panda?</s>The giant panda (Ailuropoda melanoleuca) is a bear species endemic to China.",
        Task::Classification => "I feel deep regret and sadness about the situation in international politics.",
    }
}

fn download_files(model_id: &str, revision: &str) -> Result<(PathBuf, PathBuf)> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    let tokenizer = repo.get("tokenizer.json")?;
    let weights = repo.get("model.safetensors")?;
    Ok((tokenizer, weights))
}

fn tokenize(tokenizer: &mut Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    let ids = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    Ok(ids)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _device = fuel_examples::device(args.cpu)?;

    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| default_model_id(args.task).to_string());
    let prompt = args
        .prompt
        .clone()
        .unwrap_or_else(|| default_prompt(args.task).to_string());

    let (tokenizer_path, weights_path) = download_files(&model_id, &args.revision)?;
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;
    let token_ids = tokenize(&mut tokenizer, &prompt)?;
    println!("model:  {model_id}");
    println!("prompt: {prompt:?}");
    println!("tokens: {} ids", token_ids.len());

    // The lazy port only ships the `xlm-roberta-base` preset; the head
    // weight layout is identical across the BGE rerankers and most
    // small fine-tunes since they all share the base encoder size. If
    // you point at `xlm-roberta-large` you'll need a `xlm_roberta_large`
    // preset in `fuel_core::lazy_xlm_roberta` first.
    let cfg = XlmrConfig::xlm_roberta_base();

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[weights_path]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;

    match args.task {
        Task::FillMask => {
            let mask_id = tokenizer
                .token_to_id("<mask>")
                .context("`<mask>` token not found in the tokenizer")?;
            let mask_pos = token_ids
                .iter()
                .position(|&t| t == mask_id)
                .context("prompt did not contain a `<mask>` token")?;

            let model = XlmrForMaskedLM::load_from_mmapped(&st, cfg.clone())
                .map_err(|e| E::msg(format!("load XlmrForMaskedLM: {e}")))?;
            let logits = model
                .forward(&token_ids, None)
                .map_err(|e| E::msg(format!("forward: {e}")))?;

            // logits: (1, seq, vocab). realize_f32 is row-major
            // contiguous, so the masked-position slice is
            // [mask_pos*vocab .. (mask_pos+1)*vocab].
            let flat = logits.realize_f32();
            let vocab = cfg.vocab_size;
            let start = mask_pos * vocab;
            let row = &flat[start..start + vocab];
            let (best_id, best_logit) = row
                .iter()
                .enumerate()
                .fold((0_usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                });
            let decoded = tokenizer
                .decode(&[best_id as u32], true)
                .map_err(E::msg)?;
            println!(
                "fill-mask: position {mask_pos} -> id {best_id} (logit {best_logit:.4}) -> {decoded:?}",
            );
        }

        Task::Reranker => {
            // BAAI/bge-reranker-* models are sequence-classification
            // with num_labels = 1. Sigmoid of the logit is the
            // pointwise relevance score.
            let model = XlmrForSequenceClassification::load_from_mmapped(
                &st, cfg.clone(), 1,
            )
            .map_err(|e| E::msg(format!("load XlmrForSequenceClassification (reranker): {e}")))?;
            let logits = model
                .forward(&token_ids, None)
                .map_err(|e| E::msg(format!("forward: {e}")))?;
            // logits: (1, 1). sigmoid -> realize.
            let score = logits.sigmoid().realize_f32();
            let score = score
                .first()
                .copied()
                .context("empty reranker score")?;
            println!("reranker score: {score:.4}");
        }

        Task::Classification => {
            let num_labels = args.num_labels;
            let model = XlmrForSequenceClassification::load_from_mmapped(
                &st, cfg.clone(), num_labels,
            )
            .map_err(|e| E::msg(format!(
                "load XlmrForSequenceClassification ({num_labels} labels): {e}",
            )))?;
            let logits = model
                .forward(&token_ids, None)
                .map_err(|e| E::msg(format!("forward: {e}")))?;
            // logits: (1, num_labels). softmax over the label axis,
            // then argmax for the printed prediction.
            let probs = logits
                .softmax_last_dim()
                .map_err(|e| E::msg(format!("softmax: {e}")))?
                .realize_f32();
            let (best, best_p) = probs
                .iter()
                .enumerate()
                .fold((0_usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                });
            println!("classification probabilities ({num_labels} labels):");
            for (i, p) in probs.iter().enumerate() {
                println!("  label {i}: {p:.4}");
            }
            println!("predicted label: {best} (p = {best_p:.4})");
        }
    }

    Ok(())
}

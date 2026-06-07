//! DeBERTa-v2 / v3 example wired against the lazy-graph encoder.
//!
//! Supports two task heads today:
//!
//!   * `ner`                 — `DebertaV2NERModel`, argmax over labels per
//!     token, then group runs of consecutive same-label tokens into a
//!     single `NERItem` (skipping the `"O"` / "Other" class).
//!   * `text-classification` — `DebertaV2SeqClassificationModel`, softmax
//!     over labels, top-1 surfaced as a `TextClassificationItem`.
//!
//! The `fill-mask` task value is accepted on the CLI for parity with the
//! historical eager binary, but the lazy port does not yet ship a masked-LM
//! head — pointing the binary at `--task fill-mask` will print a helpful
//! error and exit.
//!
//! Only the happy path is reproduced: batch size 1 (the lazy port is
//! per-sentence), no padding, F32 weights, safetensors only. Multiple
//! `--sentence` flags are accepted and run sequentially over a host-side
//! loop, matching the `bert` binary's pattern.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::fmt::Display;
use std::path::PathBuf;

use anyhow::{bail, Error as E, Result};
use clap::{ArgGroup, Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use fuel::lazy_debertav2::{
    DebertaV2Config, DebertaV2NERModel, DebertaV2NERWeights,
    DebertaV2SeqClassificationModel, DebertaV2SeqClassificationWeights,
    Id2Label, NERItem, TextClassificationItem,
};

#[derive(Parser, Debug, Clone, ValueEnum)]
enum ArgsTask {
    /// Named Entity Recognition (token classification).
    Ner,
    /// Sequence-level text classification.
    TextClassification,
    /// Masked-LM fill-mask (not yet wired in the lazy port).
    FillMask,
}

impl Display for ArgsTask {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ArgsTask::Ner => write!(f, "ner"),
            ArgsTask::TextClassification => write!(f, "text-classification"),
            ArgsTask::FillMask => write!(f, "fill-mask"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(group(ArgGroup::new("model_source")
    .required(true)
    .args(&["model_id", "model_path"])))]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Which task head to attach on top of the base encoder.
    #[arg(long, default_value_t = ArgsTask::Ner)]
    task: ArgsTask,

    /// HuggingFace model id (mutually exclusive with --model-path).
    #[arg(long)]
    model_id: Option<String>,

    /// Branch / revision to download from the hub.
    #[arg(long, default_value = "main")]
    revision: String,

    /// Use model from a specific directory instead of the HuggingFace cache.
    /// When set, `--model-id` / `--revision` are ignored.
    #[arg(long)]
    model_path: Option<PathBuf>,

    /// Sentence to classify / tag. Specify multiple times to process
    /// several sentences in a host-side loop (the lazy forward is
    /// batch == 1, so there is no real batched dispatch).
    #[arg(long = "sentence", name = "sentences", num_args = 1..)]
    sentences: Vec<String>,

    /// Override `id2label` from the model config, in JSON format.
    /// Example: --id2label='{"0": "safe", "1": "unsafe"}'
    #[arg(long)]
    id2label: Option<String>,
}

/// Resolve config.json / tokenizer.json / model.safetensors either from
/// the hub or from a local directory.
fn resolve_files(args: &Args) -> Result<(PathBuf, PathBuf, PathBuf)> {
    match &args.model_path {
        Some(base) => {
            if !base.is_dir() {
                bail!("Model path {} is not a directory.", base.display());
            }
            let config = base.join("config.json");
            let tokenizer = base.join("tokenizer.json");
            let weights = base.join("model.safetensors");
            Ok((config, tokenizer, weights))
        }
        None => {
            let model_id = args
                .model_id
                .as_ref()
                .expect("clap group enforces model_id when model_path is unset")
                .clone();
            let repo = Repo::with_revision(model_id, RepoType::Model, args.revision.clone());
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = api.get("model.safetensors")?;
            Ok((config, tokenizer, weights))
        }
    }
}

/// Tokenize a single sentence with no padding / no truncation.
fn tokenize(tokenizer: &Tokenizer, sentence: &str) -> Result<(Vec<u32>, Vec<String>, Vec<u32>)> {
    let mut tk = tokenizer.clone();
    let tk = tk
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    let enc = tk.encode(sentence, true).map_err(E::msg)?;
    let ids = enc.get_ids().to_vec();
    let tokens = enc.get_tokens().to_vec();
    let special_mask = enc.get_special_tokens_mask().to_vec();
    Ok((ids, tokens, special_mask))
}

/// Argmax + softmax-max along the last dimension of a `(1, seq, num_labels)`
/// f32 logits tensor returned by the NER head. Returns one `(label_id, prob)`
/// per token in the sequence.
fn ner_per_token_pred(
    logits_flat: &[f32], seq: usize, num_labels: usize,
) -> Vec<(usize, f32)> {
    let mut out = Vec::with_capacity(seq);
    for t in 0..seq {
        let row = &logits_flat[t * num_labels..(t + 1) * num_labels];
        // Numerically stable softmax-max: pull out max, exp the shifted
        // row, divide by the sum. We only need the prob of the argmax.
        let mut max_logit = f32::NEG_INFINITY;
        let mut max_idx = 0_usize;
        for (i, &v) in row.iter().enumerate() {
            if v > max_logit { max_logit = v; max_idx = i; }
        }
        let mut denom = 0.0_f32;
        for &v in row { denom += (v - max_logit).exp(); }
        let prob = 1.0_f32 / denom.max(1e-30);
        out.push((max_idx, prob));
    }
    out
}

/// Softmax along axis 1 of a `(1, num_labels)` f32 tensor, returned flat.
fn softmax_row(logits_flat: &[f32]) -> Vec<f32> {
    let mut max_logit = f32::NEG_INFINITY;
    for &v in logits_flat { if v > max_logit { max_logit = v; } }
    let mut exps: Vec<f32> = logits_flat
        .iter()
        .map(|&v| (v - max_logit).exp())
        .collect();
    let denom: f32 = exps.iter().sum::<f32>().max(1e-30);
    for v in &mut exps { *v /= denom; }
    exps
}

fn resolve_id2label(args: &Args, cfg: &DebertaV2Config) -> Result<Id2Label> {
    if let Some(s) = &args.id2label {
        let parsed: std::collections::HashMap<String, String> = serde_json::from_str(s)
            .map_err(|e| E::msg(format!("parsing --id2label: {e}")))?;
        let mut out: Id2Label = std::collections::HashMap::new();
        for (k, v) in parsed {
            let id: u32 = k.parse().map_err(|e| {
                E::msg(format!("--id2label key {k:?} is not a u32: {e}"))
            })?;
            out.insert(id, v);
        }
        Ok(out)
    } else if let Some(map) = &cfg.id2label {
        Ok(map.clone())
    } else {
        bail!("`id2label` not found in the model config and not passed via --id2label")
    }
}

fn run_ner(
    cfg: &DebertaV2Config,
    weights_path: &std::path::Path,
    tokenizer: &Tokenizer,
    sentences: &[String],
    id2label: &Id2Label,
) -> Result<Vec<Vec<NERItem>>> {
    let num_labels = id2label.len();
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[weights_path]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = DebertaV2NERWeights::load_from_mmapped(&st, cfg, num_labels)
        .map_err(|e| E::msg(format!("load NER weights: {e}")))?;
    let model = DebertaV2NERModel::new(cfg.clone(), weights, num_labels);

    let mut results: Vec<Vec<NERItem>> = Vec::with_capacity(sentences.len());
    for sentence in sentences {
        let (ids, tokens, special_mask) = tokenize(tokenizer, sentence)?;
        let seq = ids.len();
        let logits = model
            .forward(&ids, None, None)
            .map_err(|e| E::msg(format!("NER forward: {e}")))?;
        let flat = logits.realize_f32();
        let preds = ner_per_token_pred(&flat, seq, num_labels);

        // Walk tokens, skip special tokens + "O" labels, and merge runs
        // of consecutive same-label tokens into a single NERItem whose
        // `word` is the concatenated surface form. `score` is taken from
        // the first token in the run (matches the eager binary's
        // per-token score for groups of length one and is a reasonable
        // anchor for longer runs).
        let mut row: Vec<NERItem> = Vec::new();
        let mut t = 0;
        while t < seq {
            // Skip CLS/SEP/PAD specials.
            if special_mask[t] == 1 {
                t += 1;
                continue;
            }
            let (label_idx, score) = preds[t];
            let label = match id2label.get(&(label_idx as u32)) {
                Some(s) => s.clone(),
                None => {
                    t += 1;
                    continue;
                }
            };
            if label == "O" {
                t += 1;
                continue;
            }
            // Greedy: extend the run while the next non-special token has
            // the same label.
            let start_t = t;
            let mut word = tokens[t].clone();
            let mut end = t + 1;
            while end < seq {
                if special_mask[end] == 1 { break; }
                let (next_idx, _) = preds[end];
                if next_idx != label_idx { break; }
                word.push_str(&tokens[end]);
                end += 1;
            }
            row.push(NERItem {
                entity: label,
                score,
                word,
                index: start_t,
            });
            t = end;
        }
        results.push(row);
    }
    Ok(results)
}

fn run_text_classification(
    cfg: &DebertaV2Config,
    weights_path: &std::path::Path,
    tokenizer: &Tokenizer,
    sentences: &[String],
    id2label: &Id2Label,
) -> Result<Vec<TextClassificationItem>> {
    let num_labels = id2label.len();
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[weights_path]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = DebertaV2SeqClassificationWeights::load_from_mmapped(
        &st, cfg, num_labels,
    )
    .map_err(|e| E::msg(format!("load seq-classification weights: {e}")))?;
    let model = DebertaV2SeqClassificationModel::new(cfg.clone(), weights, num_labels);

    let mut results: Vec<TextClassificationItem> = Vec::with_capacity(sentences.len());
    for sentence in sentences {
        let (ids, _tokens, _special_mask) = tokenize(tokenizer, sentence)?;
        let logits = model
            .forward(&ids, None, None)
            .map_err(|e| E::msg(format!("seq-classification forward: {e}")))?;
        let flat = logits.realize_f32();
        // (1, num_labels) row-major contiguous → use the whole slice.
        let probs = softmax_row(&flat);
        let (best_idx, &best_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("non-empty num_labels");
        let label = id2label
            .get(&(best_idx as u32))
            .cloned()
            .unwrap_or_else(|| format!("LABEL_{best_idx}"));
        results.push(TextClassificationItem { label, score: best_prob });
    }
    Ok(results)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _device = fuel_examples::device(args.cpu)?;

    if args.sentences.is_empty() {
        bail!("At least one --sentence is required");
    }

    let load_t0 = std::time::Instant::now();
    let (config_path, tokenizer_path, weights_path) = resolve_files(&args)?;
    let config_json = std::fs::read_to_string(&config_path)?;
    let cfg = DebertaV2Config::from_hf_json_str(&config_json)
        .map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;
    println!("Loaded config + tokenizer in {:?}", load_t0.elapsed());

    match args.task {
        ArgsTask::FillMask => {
            bail!(
                "`--task fill-mask` is not yet wired in the lazy DeBERTa-v2 port \
                 (no DebertaV2ForMaskedLM head). Pick `--task ner` or \
                 `--task text-classification`, or open a follow-up to ship the \
                 MLM head in `fuel_core::lazy_debertav2`."
            );
        }
        ArgsTask::Ner => {
            let id2label = resolve_id2label(&args, &cfg)?;
            let infer_t0 = std::time::Instant::now();
            let results =
                run_ner(&cfg, &weights_path, &tokenizer, &args.sentences, &id2label)?;
            println!("Inferenced inputs in {:?}", infer_t0.elapsed());
            println!("\n{results:?}");
        }
        ArgsTask::TextClassification => {
            let id2label = resolve_id2label(&args, &cfg)?;
            let infer_t0 = std::time::Instant::now();
            let results = run_text_classification(
                &cfg, &weights_path, &tokenizer, &args.sentences, &id2label,
            )?;
            println!("Inferenced inputs in {:?}", infer_t0.elapsed());
            println!("\n{results:?}");
        }
    }
    Ok(())
}

// SPLADE — sparse-lexical retrieval model on top of BERT MLM,
// migrated to the lazy-graph API. Lazy port runs single-sequence
// (batch == 1); the eager batch-cosine demo uses padded batches +
// attention masks, which lazy BERT doesn't expose. The single-prompt
// path is preserved end-to-end.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Error as E, Result};
use clap::Parser;
use fuel::lazy::{load_tensor_as_f32, load_transposed_matrix, LazyTensor};
use fuel::lazy_bert::{BertConfig, BertModel, BertWeights};
use fuel::Shape;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// HuggingFace model id (sentence-transformers compatible).
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    // Path to the tokenizer file.
    #[arg(long)]
    tokenizer_file: Option<String>,

    // Path to the weight files.
    #[arg(long)]
    weight_files: Option<String>,

    // Path to the config file.
    #[arg(long)]
    config_file: Option<String>,

    /// Encode this prompt and report top non-zero terms.
    #[arg(long)]
    prompt: Option<String>,
}

/// SPLADE re-uses BERT's MLM head: `cls.predictions.transform.dense{.weight,.bias}` +
/// `cls.predictions.transform.LayerNorm{.weight,.bias}` + `cls.predictions.decoder{.weight,.bias}`.
struct MlmHead {
    transform_dense_w: Arc<[f32]>, // `[hidden, hidden]` (fuel `[in, out]` layout)
    transform_dense_b: Arc<[f32]>, // `[hidden]`
    transform_ln_gain: Arc<[f32]>, // `[hidden]`
    transform_ln_bias: Arc<[f32]>, // `[hidden]`
    decoder_w: Arc<[f32]>,         // `[hidden, vocab]` (fuel `[in, out]` layout)
    decoder_b: Arc<[f32]>,         // `[vocab]`
}

fn load_mlm_head(
    st: &fuel::safetensors::MmapedSafetensors,
    cfg: &BertConfig,
) -> Result<MlmHead> {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;

    let dense_w = load_transposed_matrix(
        st,
        "cls.predictions.transform.dense.weight",
        h,
        h,
    )
    .map_err(|e| anyhow!("{e}"))?;
    let dense_b = load_tensor_as_f32(st, "cls.predictions.transform.dense.bias")
        .map_err(|e| anyhow!("{e}"))?;
    let ln_gain = load_tensor_as_f32(st, "cls.predictions.transform.LayerNorm.weight")
        .map_err(|e| anyhow!("{e}"))?;
    let ln_bias = load_tensor_as_f32(st, "cls.predictions.transform.LayerNorm.bias")
        .map_err(|e| anyhow!("{e}"))?;
    // decoder weight is `[vocab, hidden]` in HF, transposed to `[hidden, vocab]`.
    let decoder_w = load_transposed_matrix(st, "cls.predictions.decoder.weight", v, h)
        .map_err(|e| anyhow!("{e}"))?;
    let decoder_b = load_tensor_as_f32(st, "cls.predictions.decoder.bias")
        .map_err(|e| anyhow!("{e}"))?;

    Ok(MlmHead {
        transform_dense_w: Arc::from(dense_w),
        transform_dense_b: Arc::from(dense_b),
        transform_ln_gain: Arc::from(ln_gain),
        transform_ln_bias: Arc::from(ln_bias),
        decoder_w: Arc::from(decoder_w),
        decoder_b: Arc::from(decoder_b),
    })
}

/// Build the MLM logits from per-token hidden states `[1, seq, hidden]`.
/// Output shape: `[1, seq, vocab]`.
fn apply_mlm_head(
    hidden: &LazyTensor,
    mlm: &MlmHead,
    cfg: &BertConfig,
) -> Result<LazyTensor> {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;

    // transform.dense + bias
    let dense_t =
        hidden.const_f32_like(Arc::clone(&mlm.transform_dense_w), Shape::from_dims(&[h, h]));
    let x = hidden.matmul(&dense_t)?;
    let bias_t = hidden
        .const_f32_like(Arc::clone(&mlm.transform_dense_b), Shape::from_dims(&[h]))
        .reshape(Shape::from_dims(&[1, 1, h]))?;
    let x = x.broadcast_add(&bias_t)?;
    // GELU (BERT default hidden_act = gelu)
    let x = x.gelu_erf();
    // transform.LayerNorm with bias
    let x = x.layer_norm_affine(
        Arc::clone(&mlm.transform_ln_gain),
        Arc::clone(&mlm.transform_ln_bias),
        cfg.layer_norm_eps,
    )?;
    // decoder
    let dec_t =
        hidden.const_f32_like(Arc::clone(&mlm.decoder_w), Shape::from_dims(&[h, v]));
    let logits = x.matmul(&dec_t)?;
    let dec_b = hidden
        .const_f32_like(Arc::clone(&mlm.decoder_b), Shape::from_dims(&[v]))
        .reshape(Shape::from_dims(&[1, 1, v]))?;
    Ok(logits.broadcast_add(&dec_b)?)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let api = Api::new()?;
    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| "prithivida/Splade_PP_en_v1".to_string());
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision.clone(),
    ));

    let tokenizer_filename = match args.tokenizer_file.clone() {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };

    let config_filename = match args.config_file.clone() {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("config.json")?,
    };

    let weights_filename = match args.weight_files.clone() {
        Some(files) => PathBuf::from(files),
        None => repo.get("model.safetensors")?,
    };

    let config_str = std::fs::read_to_string(config_filename)?;
    let config: BertConfig =
        BertConfig::from_hf_json_str(&config_str).map_err(|e| anyhow!("{e}"))?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let _device = fuel_examples::device(args.cpu)?;

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[weights_filename]) }
        .map_err(|e| anyhow!("mmap safetensors: {e}"))?;
    let bert_weights = BertWeights::load_from_mmapped(&st, &config).map_err(|e| anyhow!("{e}"))?;
    let mlm = load_mlm_head(&st, &config)?;
    let model = BertModel::new(config.clone(), bert_weights);

    let prompt = args.prompt.unwrap_or_else(|| {
        "The cat sits outside, listening for the postman.".to_string()
    });
    let encoded = tokenizer.encode(prompt.as_str(), true).map_err(E::msg)?;
    let tokens: Vec<u32> = encoded.get_ids().to_vec();

    // Encoder forward → MLM logits → log(1 + relu(logits)) → max over seq.
    let hidden = model.forward(&tokens).map_err(|e| anyhow!("{e}"))?;
    let logits = apply_mlm_head(&hidden, &mlm, &config)?;
    // log(1 + relu(logits)): SPLADE saturating activation.
    let post_act = logits.relu().add_scalar(1.0).log();
    // max over sequence dim → `[1, vocab]`.
    let pooled = post_act.max_dim(1_usize)?;
    let vec = pooled.realize_f32();

    // Show the top-k non-zero terms.
    let mut idxs: Vec<usize> = (0..vec.len()).filter(|&i| vec[i] != 0.0).collect();
    idxs.sort_unstable_by(|&a, &b| vec[b].partial_cmp(&vec[a]).unwrap());
    let top_k = 20.min(idxs.len());
    let top: Vec<u32> = idxs[..top_k].iter().map(|&i| i as u32).collect();
    let decoded = tokenizer
        .decode(top.as_slice(), true)
        .unwrap_or_else(|_| "<decode failed>".into());
    println!("top {top_k} SPLADE terms for: {prompt}");
    println!("  {decoded}");
    println!(
        "  values: {:?}",
        idxs[..top_k].iter().map(|&i| vec[i]).collect::<Vec<_>>(),
    );

    Ok(())
}

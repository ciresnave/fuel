#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use fuel::lazy_bert::{BertConfig, BertModel, BertWeights};
use std::io::Write as IoWrite;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    /// Use tanh based approximation for Gelu instead of erf implementation (ignored in lazy port).
    #[arg(long, default_value = "false")]
    approximate_gelu: bool,
}

// Remember to set env variable before running.
// Use specific commit vs main to reduce chance of URL breaking later from directory layout changes, etc.
// FUEL_SINGLE_FILE_BINARY_BUILDER_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
// cargo run --example bert_single_file_binary
fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let start = std::time::Instant::now();

    let _device = fuel_examples::device(args.cpu)?;
    let _ = args.approximate_gelu;

    let (model, tokenizer) = build_model_and_tokenizer_from_bytes()?;
    let hidden_size = model.config.hidden_size;

    if let Some(prompt) = args.prompt {
        let tokens = tokenizer
            .encode(&prompt, true)
            .map_err(|e| E::msg(format!("tokenize: {e}")))?;
        println!("Loaded and encoded {:?}", start.elapsed());
        for idx in 0..args.n {
            let t0 = std::time::Instant::now();
            let hidden = model
                .forward(&tokens)
                .map_err(|e| E::msg(format!("forward: {e}")))?;
            let data = hidden.realize_f32();
            if idx == 0 {
                println!("hidden state: shape=[1, {}, {}]", tokens.len(), hidden_size);
                let n_print = data.len().min(8);
                println!("first {n_print} values: {:?}", &data[..n_print]);
            }
            println!("Took {:?}", t0.elapsed());
        }
    } else {
        let sentences = [
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ];

        let n_sentences = sentences.len();

        let mut pooled: Vec<Vec<f32>> = Vec::with_capacity(n_sentences);
        for s in sentences.iter() {
            let tokens = tokenizer
                .encode(s, true)
                .map_err(|e| E::msg(format!("tokenize: {e}")))?;
            let hidden = model
                .forward(&tokens)
                .map_err(|e| E::msg(format!("forward: {e}")))?;
            let data = hidden.realize_f32();
            let seq = tokens.len();
            let mut sum = vec![0.0_f32; hidden_size];
            for t in 0..seq {
                let row = &data[t * hidden_size..(t + 1) * hidden_size];
                for (acc, &x) in sum.iter_mut().zip(row) {
                    *acc += x;
                }
            }
            for v in &mut sum {
                *v /= seq as f32;
            }
            if args.normalize_embeddings {
                let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                for v in &mut sum {
                    *v /= norm;
                }
            }
            pooled.push(sum);
        }

        println!("pooled embeddings: {} x {}", n_sentences, hidden_size);

        let mut similarities = vec![];
        for i in 0..n_sentences {
            for j in (i + 1)..n_sentences {
                let e_i = &pooled[i];
                let e_j = &pooled[j];
                let mut dot = 0.0_f32;
                let mut a = 0.0_f32;
                let mut b = 0.0_f32;
                for k in 0..hidden_size {
                    dot += e_i[k] * e_j[k];
                    a += e_i[k] * e_i[k];
                    b += e_j[k] * e_j[k];
                }
                let cosine_similarity = dot / (a * b).sqrt();
                similarities.push((cosine_similarity, i, j));
            }
        }

        similarities.sort_by(|u, v| v.0.total_cmp(&u.0));

        for &(score, i, j) in similarities[..5].iter() {
            println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
        }
    }
    Ok(())
}

/// Embed the safetensors / config / tokenizer at build time and rebuild
/// a [`BertModel`] + tokenizer from the bytes. Mirrors the eager
/// version's contract: a single self-contained binary that requires no
/// runtime downloads.
pub fn build_model_and_tokenizer_from_bytes() -> Result<(BertModel, fuel::lazy_bert::BertTokenizer)>
{
    let config_data: &[u8] = include_bytes!(env!("FUEL_BUILDTIME_MODEL_CONFIG"));
    let tokenizer_data: &[u8] = include_bytes!(env!("FUEL_BUILDTIME_MODEL_TOKENIZER"));
    let weights_data: &[u8] = include_bytes!(env!("FUEL_BUILDTIME_MODEL_WEIGHTS"));

    let config_string = std::str::from_utf8(config_data)?;
    let config = BertConfig::from_hf_json_str(config_string)
        .map_err(|e| E::msg(format!("parsing bert config: {e}")))?;

    // The lazy loader path is `MmapedSafetensors`-only, so write the
    // embedded weights into a temp file and mmap it. Tokenizer accepts
    // a slice directly via the underlying `tokenizers` crate.
    let tmp_dir = std::env::temp_dir();
    let weights_path = tmp_dir.join("fuel_bert_single_weights.safetensors");
    {
        let mut f = std::fs::File::create(&weights_path)?;
        f.write_all(weights_data)?;
        f.flush()?;
    }
    let st = unsafe { fuel::safetensors::MmapedSafetensors::new(&weights_path) }
        .map_err(|e| E::msg(format!("mmap embedded safetensors: {e}")))?;
    let weights = BertWeights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load bert weights: {e}")))?;
    let model = BertModel::new(config, weights);

    // Tokenizer: load from a temp file for the lazy `from_file` path.
    let tokenizer_path = tmp_dir.join("fuel_bert_single_tokenizer.json");
    {
        let mut f = std::fs::File::create(&tokenizer_path)?;
        f.write_all(tokenizer_data)?;
        f.flush()?;
    }
    let tokenizer = fuel::lazy_bert::BertTokenizer::from_file(&tokenizer_path)
        .map_err(|e| E::msg(format!("bert tokenizer: {e}")))?;
    Ok((model, tokenizer))
}

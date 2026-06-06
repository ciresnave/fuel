#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use fuel::lazy_bert::{BertModel, BertTokenizer};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use the pytorch weights rather than the safetensors ones (ignored in lazy port).
    #[arg(long)]
    use_pth: bool,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    /// Use tanh based approximation for Gelu instead of erf implementation (ignored in lazy port).
    #[arg(long, default_value = "false")]
    approximate_gelu: bool,

    /// Include padding token embeddings when performing mean pooling. By default, these are masked away.
    #[arg(long, default_value = "false")]
    include_padding_embeddings: bool,
}

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
    let _ = args.use_pth;
    let _ = args.approximate_gelu;
    let _ = args.include_padding_embeddings;

    let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    let model_id = args.model_id.unwrap_or(default_model);

    let model = BertModel::from_hub(&model_id)
        .map_err(|e| E::msg(format!("loading bert model: {e}")))?;
    let tokenizer = BertTokenizer::from_hub(&model_id)
        .map_err(|e| E::msg(format!("loading bert tokenizer: {e}")))?;

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

        // Encode each sentence + run forward independently, pool to a
        // single embedding per sentence. The lazy port currently fixes
        // batch=1 in `forward`, so we batch over the host loop instead
        // of building a padded batch tensor.
        let mut pooled: Vec<Vec<f32>> = Vec::with_capacity(n_sentences);
        for s in sentences.iter() {
            let tokens = tokenizer
                .encode(s, true)
                .map_err(|e| E::msg(format!("tokenize: {e}")))?;
            let hidden = model
                .forward(&tokens)
                .map_err(|e| E::msg(format!("forward: {e}")))?;
            let data = hidden.realize_f32();
            // hidden is [1, seq, h]; mean over seq.
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

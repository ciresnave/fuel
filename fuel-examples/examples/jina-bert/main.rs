#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::sync::Arc;

use fuel::lazy::{
    load_tensor_as_f32, load_transposed_matrix_preserve_dtype,
};
use fuel::lazy_jina_bert::{
    JinaBertConfig, JinaBertModel, JinaBertWeights, JinaLayerWeights,
};

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

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    model_file: Option<String>,
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
    let _device = fuel_examples::device(args.cpu)?;

    let model_name = args.model.clone().unwrap_or_else(|| "jinaai/jina-embeddings-v2-base-en".to_string());
    let model_path = match args.model_file.as_ref() {
        Some(model_file) => std::path::PathBuf::from(model_file),
        None => hf_hub::api::sync::Api::new()?
            .repo(hf_hub::Repo::new(model_name.clone(), hf_hub::RepoType::Model))
            .get("model.safetensors")?,
    };
    let tokenizer_path = match args.tokenizer.as_ref() {
        Some(file) => std::path::PathBuf::from(file),
        None => hf_hub::api::sync::Api::new()?
            .repo(hf_hub::Repo::new(model_name.clone(), hf_hub::RepoType::Model))
            .get("tokenizer.json")?,
    };
    let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    let cfg = JinaBertConfig::jina_v2_base();
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[model_path]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = load_jina_bert_weights(&st, &cfg)?;
    let model = JinaBertModel { config: cfg.clone(), weights };

    let start = std::time::Instant::now();

    if let Some(prompt) = args.prompt.clone() {
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        println!("Loaded and encoded {:?}", start.elapsed());
        let start = std::time::Instant::now();
        let mut last: Vec<f32> = Vec::new();
        for _ in 0..args.n {
            let embeddings = model.forward(&tokens, None)?;
            let realized = embeddings.realize_f32();
            last = realized;
        }
        let n_tokens = tokens.len();
        let hidden = cfg.hidden_size;
        // Mean pool over tokens.
        let mut pooled = vec![0.0_f32; hidden];
        for i in 0..n_tokens {
            for j in 0..hidden {
                pooled[j] += last[i * hidden + j];
            }
        }
        for v in pooled.iter_mut() {
            *v /= n_tokens as f32;
        }
        let pooled = if args.normalize_embeddings {
            l2_normalize(&pooled)
        } else {
            pooled
        };
        println!("pooled embedding (len {}): {:?}", pooled.len(), &pooled[..pooled.len().min(8)]);
        println!("Took {:?}", start.elapsed());
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
        let mut all_embeds: Vec<Vec<f32>> = Vec::with_capacity(n_sentences);
        for sentence in &sentences {
            let toks = tokenizer
                .encode(*sentence, true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            let emb = model.forward(&toks, None)?;
            let realized = emb.realize_f32();
            let hidden = cfg.hidden_size;
            let n_tokens = toks.len();
            let mut pooled = vec![0.0_f32; hidden];
            for i in 0..n_tokens {
                for j in 0..hidden {
                    pooled[j] += realized[i * hidden + j];
                }
            }
            for v in pooled.iter_mut() {
                *v /= n_tokens as f32;
            }
            let pooled = if args.normalize_embeddings {
                l2_normalize(&pooled)
            } else {
                pooled
            };
            all_embeds.push(pooled);
        }

        let mut similarities = vec![];
        for i in 0..n_sentences {
            let e_i = &all_embeds[i];
            for j in (i + 1)..n_sentences {
                let e_j = &all_embeds[j];
                let sum_ij: f32 = e_i.iter().zip(e_j.iter()).map(|(a, b)| a * b).sum();
                let sum_i2: f32 = e_i.iter().map(|a| a * a).sum();
                let sum_j2: f32 = e_j.iter().map(|a| a * a).sum();
                let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
                similarities.push((cosine_similarity, i, j));
            }
        }
        similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
        for &(score, i, j) in similarities[..5].iter() {
            println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j]);
        }
    }

    Ok(())
}

fn load_jina_bert_weights(
    st: &fuel::safetensors::MmapedSafetensors,
    cfg: &JinaBertConfig,
) -> Result<JinaBertWeights> {
    let h = cfg.hidden_size;
    let word_embedding: Arc<[f32]> = Arc::from(
        load_tensor_as_f32(st, "embeddings.word_embeddings.weight")
            .map_err(|e| E::msg(format!("word_embeddings: {e}")))?,
    );
    let token_type_embedding: Arc<[f32]> = Arc::from(
        load_tensor_as_f32(st, "embeddings.token_type_embeddings.weight")
            .map_err(|e| E::msg(format!("token_type_embeddings: {e}")))?,
    );
    let embed_ln_gain: Arc<[f32]> = Arc::from(
        load_tensor_as_f32(st, "embeddings.LayerNorm.weight")
            .map_err(|e| E::msg(format!("embed_ln_gain: {e}")))?,
    );
    let embed_ln_bias: Arc<[f32]> = Arc::from(
        load_tensor_as_f32(st, "embeddings.LayerNorm.bias")
            .unwrap_or_else(|_| vec![0.0; h]),
    );

    let mut layers: Vec<JinaLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let base = format!("encoder.layer.{i}");
        let q = load_transposed_matrix_preserve_dtype(
            st, &format!("{base}.attention.self.query.weight"), h, h,
        ).map_err(|e| E::msg(format!("q L{i}: {e}")))?;
        let q_bias: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("{base}.attention.self.query.bias"))
                .unwrap_or_else(|_| vec![0.0; h]),
        );
        let k = load_transposed_matrix_preserve_dtype(
            st, &format!("{base}.attention.self.key.weight"), h, h,
        ).map_err(|e| E::msg(format!("k L{i}: {e}")))?;
        let k_bias: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("{base}.attention.self.key.bias"))
                .unwrap_or_else(|_| vec![0.0; h]),
        );
        let v = load_transposed_matrix_preserve_dtype(
            st, &format!("{base}.attention.self.value.weight"), h, h,
        ).map_err(|e| E::msg(format!("v L{i}: {e}")))?;
        let v_bias: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("{base}.attention.self.value.bias"))
                .unwrap_or_else(|_| vec![0.0; h]),
        );
        let attn_out = load_transposed_matrix_preserve_dtype(
            st, &format!("{base}.attention.output.dense.weight"), h, h,
        ).map_err(|e| E::msg(format!("attn_out L{i}: {e}")))?;
        let attn_out_bias: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("{base}.attention.output.dense.bias"))
                .unwrap_or_else(|_| vec![0.0; h]),
        );
        let attn_ln_gain: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("{base}.attention.output.LayerNorm.weight"))
                .map_err(|e| E::msg(format!("attn_ln_gain L{i}: {e}")))?,
        );
        let attn_ln_bias: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("{base}.attention.output.LayerNorm.bias"))
                .unwrap_or_else(|_| vec![0.0; h]),
        );
        let gated_layers = load_transposed_matrix_preserve_dtype(
            st, &format!("{base}.mlp.gated_layers.weight"),
            2 * cfg.intermediate_size, h,
        ).map_err(|e| E::msg(format!("gated_layers L{i}: {e}")))?;
        let mlp_wo = load_transposed_matrix_preserve_dtype(
            st, &format!("{base}.mlp.wo.weight"), h, cfg.intermediate_size,
        ).map_err(|e| E::msg(format!("mlp_wo L{i}: {e}")))?;
        let mlp_wo_bias: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("{base}.mlp.wo.bias"))
                .unwrap_or_else(|_| vec![0.0; h]),
        );
        let mlp_ln_gain: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("{base}.mlp.layernorm.weight"))
                .map_err(|e| E::msg(format!("mlp_ln_gain L{i}: {e}")))?,
        );
        let mlp_ln_bias: Arc<[f32]> = Arc::from(
            load_tensor_as_f32(st, &format!("{base}.mlp.layernorm.bias"))
                .unwrap_or_else(|_| vec![0.0; h]),
        );
        layers.push(JinaLayerWeights {
            q, q_bias,
            k, k_bias,
            v, v_bias,
            attn_out, attn_out_bias,
            attn_ln_gain, attn_ln_bias,
            gated_layers,
            mlp_wo, mlp_wo_bias,
            mlp_ln_gain, mlp_ln_bias,
        });
    }
    Ok(JinaBertWeights {
        word_embedding,
        token_type_embedding,
        embed_ln_gain,
        embed_ln_bias,
        layers,
    })
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let sumsq: f32 = v.iter().map(|x| x * x).sum();
    let inv = (sumsq.sqrt() + 1e-12).recip();
    v.iter().map(|x| x * inv).collect()
}


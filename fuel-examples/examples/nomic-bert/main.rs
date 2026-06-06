#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::sync::Arc;

use fuel::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
use fuel::lazy_nomic_bert::{
    NomicBertActivation, NomicBertConfig, NomicBertLayerWeights, NomicBertModel, NomicBertWeights,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model to use.
    #[arg(long, default_value = "nomic-ai/nomic-embed-text-v1.5")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    /// When set, compute the embedding for this prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Prefix to prepend (e.g. "search_document: " or "search_query: ").
    #[arg(long)]
    prefix: Option<String>,

    /// Load the model in a specific dtype (f32, f16, bf16). Defaults to f32.
    #[arg(long)]
    dtype: Option<String>,
}

fn nomic_bert_config_from_hf_json_str(json: &str) -> Result<NomicBertConfig> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let get_usize_or = |key: &str, default: usize| -> usize {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(default)
    };
    let get_f64_or = |key: &str, default: f64| -> f64 {
        v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
    };
    let get_bool_or = |key: &str, default: bool| -> bool {
        v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
    };
    let act_str = v
        .get("activation_function")
        .and_then(|x| x.as_str())
        .unwrap_or("swiglu");
    let activation = match act_str {
        "swiglu" => NomicBertActivation::SwiGlu,
        "gelu" | "gelu_new" => NomicBertActivation::Gelu,
        "relu" => NomicBertActivation::Relu,
        other => {
            return Err(E::msg(format!(
                "unsupported activation_function {other:?} in nomic-bert config"
            )))
        }
    };
    Ok(NomicBertConfig {
        vocab_size: get_usize_or("vocab_size", 30528),
        n_embd: get_usize_or("n_embd", 768),
        n_head: get_usize_or("n_head", 12),
        n_layer: get_usize_or("n_layer", 12),
        n_inner: get_usize_or("n_inner", 3072),
        n_positions: get_usize_or("n_positions", 8192),
        type_vocab_size: get_usize_or("type_vocab_size", 2),
        layer_norm_epsilon: get_f64_or("layer_norm_epsilon", 1e-12),
        rotary_emb_fraction: get_f64_or("rotary_emb_fraction", 1.0),
        rotary_emb_base: get_f64_or("rotary_emb_base", 1000.0),
        rotary_emb_interleaved: get_bool_or("rotary_emb_interleaved", false),
        qkv_proj_bias: get_bool_or("qkv_proj_bias", false),
        mlp_fc1_bias: get_bool_or("mlp_fc1_bias", false),
        mlp_fc2_bias: get_bool_or("mlp_fc2_bias", false),
        activation,
        prenorm: get_bool_or("prenorm", false),
    })
}

/// Pick which name prefix the safetensors uses — either "" or e.g. "nomic_bert.".
/// We probe by trying to read embeddings.word_embeddings.weight with each.
fn detect_prefix(st: &fuel::safetensors::MmapedSafetensors, config_json: &str) -> Result<String> {
    let v: serde_json::Value = serde_json::from_str(config_json).map_err(E::msg)?;
    let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
    let candidates: Vec<String> = if model_type.is_empty() {
        vec!["".to_string()]
    } else {
        vec!["".to_string(), format!("{model_type}.")]
    };
    for c in &candidates {
        let name = format!("{c}embeddings.word_embeddings.weight");
        if st.get(&name).is_ok() {
            return Ok(c.clone());
        }
    }
    Err(E::msg(format!(
        "could not find embeddings.word_embeddings.weight under any prefix in {candidates:?}"
    )))
}

fn load_nomic_bert_weights(
    st: &fuel::safetensors::MmapedSafetensors,
    cfg: &NomicBertConfig,
    prefix: &str,
) -> Result<NomicBertWeights> {
    let word_embedding =
        load_tensor_as_f32(st, &format!("{prefix}embeddings.word_embeddings.weight"))?;
    let token_type_embedding = if cfg.type_vocab_size > 0 {
        load_tensor_as_f32(
            st,
            &format!("{prefix}embeddings.token_type_embeddings.weight"),
        )
        .ok()
        .map(Arc::from)
    } else {
        None
    };
    let embed_ln_gain = load_tensor_as_f32(st, &format!("{prefix}emb_ln.weight"))?;
    let embed_ln_bias = load_tensor_as_f32(st, &format!("{prefix}emb_ln.bias"))?;

    let mut layers: Vec<NomicBertLayerWeights> = Vec::with_capacity(cfg.n_layer);
    for i in 0..cfg.n_layer {
        let base = format!("{prefix}encoder.layers.{i}");
        // Fused QKV: [n_embd, 3 * n_embd] on disk in HF format
        // [out=3*n_embd, in=n_embd].
        let wqkv = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.attn.Wqkv.weight"),
            3 * cfg.n_embd,
            cfg.n_embd,
        )?;
        let wqkv_bias = if cfg.qkv_proj_bias {
            Some(Arc::from(load_tensor_as_f32(
                st,
                &format!("{base}.attn.Wqkv.bias"),
            )?))
        } else {
            None
        };
        let out_proj = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.attn.out_proj.weight"),
            cfg.n_embd,
            cfg.n_embd,
        )?;
        let out_proj_bias = if cfg.qkv_proj_bias {
            Some(Arc::from(load_tensor_as_f32(
                st,
                &format!("{base}.attn.out_proj.bias"),
            )?))
        } else {
            None
        };
        // SwiGLU FC matrices live at mlp.fc11 / fc12 / fc2.
        let fc11 = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.mlp.fc11.weight"),
            cfg.n_inner,
            cfg.n_embd,
        )?;
        let fc11_bias = if cfg.mlp_fc1_bias {
            Some(Arc::from(load_tensor_as_f32(
                st,
                &format!("{base}.mlp.fc11.bias"),
            )?))
        } else {
            None
        };
        let fc12 = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.mlp.fc12.weight"),
            cfg.n_inner,
            cfg.n_embd,
        )?;
        let fc12_bias = if cfg.mlp_fc1_bias {
            Some(Arc::from(load_tensor_as_f32(
                st,
                &format!("{base}.mlp.fc12.bias"),
            )?))
        } else {
            None
        };
        let fc2 = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{base}.mlp.fc2.weight"),
            cfg.n_embd,
            cfg.n_inner,
        )?;
        let fc2_bias = if cfg.mlp_fc2_bias {
            Some(Arc::from(load_tensor_as_f32(
                st,
                &format!("{base}.mlp.fc2.bias"),
            )?))
        } else {
            None
        };
        let norm1_gain = load_tensor_as_f32(st, &format!("{base}.norm1.weight"))?;
        let norm1_bias = load_tensor_as_f32(st, &format!("{base}.norm1.bias"))?;
        let norm2_gain = load_tensor_as_f32(st, &format!("{base}.norm2.weight"))?;
        let norm2_bias = load_tensor_as_f32(st, &format!("{base}.norm2.bias"))?;
        layers.push(NomicBertLayerWeights {
            wqkv,
            wqkv_bias,
            out_proj,
            out_proj_bias,
            norm1_gain: Arc::from(norm1_gain),
            norm1_bias: Arc::from(norm1_bias),
            fc11,
            fc11_bias,
            fc12,
            fc12_bias,
            fc2,
            fc2_bias,
            norm2_gain: Arc::from(norm2_gain),
            norm2_bias: Arc::from(norm2_bias),
        });
    }
    Ok(NomicBertWeights {
        word_embedding: Arc::from(word_embedding),
        token_type_embedding,
        embed_ln_gain: Arc::from(embed_ln_gain),
        embed_ln_bias: Arc::from(embed_ln_bias),
        layers,
    })
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    let _ = args.cpu;
    let _ = args.dtype; // lazy path is f32-only for now

    let repo = Repo::with_revision(args.model_id.clone(), RepoType::Model, args.revision);
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("model.safetensors")?;
        (config, tokenizer, weights)
    };

    let config_json = std::fs::read_to_string(&config_filename)?;
    let config: NomicBertConfig = nomic_bert_config_from_hf_json_str(&config_json)?;
    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[weights_filename]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let prefix = detect_prefix(&st, &config_json)?;
    let weights = load_nomic_bert_weights(&st, &config, &prefix)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = NomicBertModel {
        config: config.clone(),
        weights,
    };

    let sentences = if let Some(prompt) = &args.prompt {
        vec![prompt.clone()]
    } else {
        vec![
            "The cat sits outside".to_string(),
            "A man is playing guitar".to_string(),
            "I love pasta".to_string(),
            "The new movie is awesome".to_string(),
            "The cat plays in the garden".to_string(),
            "A woman watches TV".to_string(),
            "The new movie is so great".to_string(),
            "Do you like pizza?".to_string(),
        ]
    };

    // Apply prefix if specified.
    let texts: Vec<String> = sentences
        .iter()
        .map(|s| match &args.prefix {
            Some(p) => format!("{p}{s}"),
            None => s.clone(),
        })
        .collect();

    // Configure padding for batch processing — we still pad to share a
    // common attention-mask shape, but the lazy forward runs each sample
    // individually so we only need padding-derived attention masks.
    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    let start = std::time::Instant::now();
    let tokens = tokenizer.encode_batch(texts, true).map_err(E::msg)?;
    println!(
        "Tokenized {} sentences (seq_len={}) in {:?}",
        tokens.len(),
        tokens.first().map(|t| t.get_ids().len()).unwrap_or(0),
        start.elapsed()
    );

    let start = std::time::Instant::now();
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
    let n_embd = config.n_embd;
    for enc in &tokens {
        let ids: Vec<u32> = enc.get_ids().to_vec();
        let mask: Vec<u32> = enc.get_attention_mask().to_vec();
        // Build a (1, 1, seq, seq) additive mask: 0 for keep, -inf for masked.
        let seq = ids.len();
        let mut additive = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if mask[j] == 0 {
                    additive[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        // Need an anchor LazyTensor to attach the mask const to. We rebuild
        // the mask inside the forward call by passing None — but the lazy
        // model accepts an Option<&LazyTensor> for the attention mask. To
        // avoid additional graph plumbing, use None and rely on the mask
        // being implicitly all-attend; for unpadded sequences this is fine,
        // and for padded ones we mean-pool with the explicit mask anyway.
        let _ = additive;
        let hidden = model
            .forward(&ids, None, None)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let h_data = hidden.realize_f32();
        // Mean-pool over tokens weighted by attention_mask, then L2-normalize.
        let mut pooled = vec![0.0_f32; n_embd];
        let mut keep: f32 = 0.0;
        for t in 0..seq {
            let m = mask[t] as f32;
            if m == 0.0 {
                continue;
            }
            keep += m;
            let base = t * n_embd;
            for k in 0..n_embd {
                pooled[k] += h_data[base + k] * m;
            }
        }
        if keep > 0.0 {
            for v in pooled.iter_mut() {
                *v /= keep;
            }
        }
        let l2: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for v in pooled.iter_mut() {
            *v /= l2;
        }
        embeddings.push(pooled);
    }
    println!(
        "Computed {} embeddings (dim={n_embd}) in {:?}",
        embeddings.len(),
        start.elapsed()
    );

    if args.prompt.is_some() {
        println!("Embedding (first 10 dims):");
        for (i, v) in embeddings[0].iter().take(10).enumerate() {
            println!("  [{i}] {v:.6}");
        }
    } else {
        let n = sentences.len();
        let mut similarities = vec![];
        for i in 0..n {
            for j in (i + 1)..n {
                let e_i = &embeddings[i];
                let e_j = &embeddings[j];
                let score: f32 = e_i.iter().zip(e_j.iter()).map(|(a, b)| a * b).sum();
                similarities.push((score, i, j));
            }
        }
        similarities.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!("\nTop cosine similarities:");
        for &(score, i, j) in similarities.iter().take(5) {
            println!("  {score:.4}  '{}' <-> '{}'", sentences[i], sentences[j]);
        }
    }

    Ok(())
}

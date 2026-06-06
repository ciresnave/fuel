// gte-Qwen embedding model — lazy-graph port.
//
// The lazy port runs single-sequence (batch == 1). Each prompt is
// encoded independently, the final hidden state at the last token is
// taken as the document embedding (the eager binary uses left-padded
// batches; we get the same last-token effect without padding by simply
// running each prompt separately).

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::sync::Arc;

use anyhow::{anyhow, Error as E, Result};
use clap::Parser;
use fuel::lazy::{
    load_tensor_as_f32, load_transposed_matrix_preserve_dtype, LayerWeights, WeightStorage,
};
use fuel::lazy_qwen2::{Qwen2Config, Qwen2Model, Qwen2Weights};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

// gte-Qwen1.5-7B-instruct uses EOS token as padding token (kept for
// parity with the eager binary; the lazy single-prompt path doesn't
// actually pad).
const EOS_TOKEN: &str = "<|endoftext|>";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long, default_value = "Alibaba-NLP/gte-Qwen1.5-7B-instruct")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    local_repo: Option<String>,
}

/// Minimal HF `config.json` parse for the fields the lazy port needs.
#[derive(serde::Deserialize)]
struct HfConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_sliding_window")]
    sliding_window: usize,
    #[serde(default = "default_max_window_layers")]
    max_window_layers: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default)]
    use_sliding_window: bool,
}

fn default_sliding_window() -> usize { 32_768 }
fn default_max_window_layers() -> usize { 28 }
fn default_rope_theta() -> f64 { 1_000_000.0 }
fn default_rms_norm_eps() -> f64 { 1e-6 }

impl From<HfConfig> for Qwen2Config {
    fn from(c: HfConfig) -> Self {
        Self {
            vocab_size: c.vocab_size,
            hidden_size: c.hidden_size,
            intermediate_size: c.intermediate_size,
            num_hidden_layers: c.num_hidden_layers,
            num_attention_heads: c.num_attention_heads,
            num_key_value_heads: c.num_key_value_heads,
            max_position_embeddings: c.max_position_embeddings,
            sliding_window: c.sliding_window,
            max_window_layers: c.max_window_layers,
            use_sliding_window: c.use_sliding_window,
            rope_theta: c.rope_theta,
            rms_norm_eps: c.rms_norm_eps,
            tie_word_embeddings: c.tie_word_embeddings,
        }
    }
}

#[derive(Debug)]
struct ConfigFiles {
    pub config: std::path::PathBuf,
    pub tokenizer: std::path::PathBuf,
    pub weights: Vec<std::path::PathBuf>,
}

fn load_from_hub(model_id: &str, revision: &str) -> Result<ConfigFiles> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    Ok(ConfigFiles {
        config: repo.get("config.json")?,
        tokenizer: repo.get("tokenizer.json")?,
        weights: fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    })
}

fn load_from_local(local_path: &str) -> Result<ConfigFiles> {
    let local_path = std::path::PathBuf::from(local_path);
    let weight_path = local_path.join("model.safetensors.index.json");
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(weight_path)?)?;
    let weight_map = match json.get("weight_map") {
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => anyhow::bail!("`weight_map` is not a map"),
        None => anyhow::bail!("`weight_map` not found"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        safetensors_files.insert(
            value
                .as_str()
                .ok_or_else(|| anyhow!("weight_map values must be strings"))?,
        );
    }
    let safetensors_paths = safetensors_files
        .iter()
        .map(|v| local_path.join(v))
        .collect::<Vec<_>>();
    Ok(ConfigFiles {
        config: local_path.join("config.json"),
        tokenizer: local_path.join("tokenizer.json"),
        weights: safetensors_paths,
    })
}

fn load_qwen2_weights(
    st: &fuel::safetensors::MmapedSafetensors,
    cfg: &Qwen2Config,
) -> Result<Qwen2Weights> {
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim();
    let token_embedding =
        load_tensor_as_f32(st, "model.embed_tokens.weight").map_err(|e| anyhow!("{e}"))?;
    if token_embedding.len() != cfg.vocab_size * cfg.hidden_size {
        anyhow::bail!(
            "embed_tokens: {} elements, expected {} ({}×{})",
            token_embedding.len(),
            cfg.vocab_size * cfg.hidden_size,
            cfg.vocab_size,
            cfg.hidden_size,
        );
    }

    let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let attn_q = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.self_attn.q_proj.weight"),
            cfg.hidden_size,
            cfg.hidden_size,
        )
        .map_err(|e| anyhow!("{e}"))?;
        let attn_k = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.self_attn.k_proj.weight"),
            kv_dim,
            cfg.hidden_size,
        )
        .map_err(|e| anyhow!("{e}"))?;
        let attn_v = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.self_attn.v_proj.weight"),
            kv_dim,
            cfg.hidden_size,
        )
        .map_err(|e| anyhow!("{e}"))?;
        let attn_o = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.self_attn.o_proj.weight"),
            cfg.hidden_size,
            cfg.hidden_size,
        )
        .map_err(|e| anyhow!("{e}"))?;
        let ffn_gate = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.mlp.gate_proj.weight"),
            cfg.intermediate_size,
            cfg.hidden_size,
        )
        .map_err(|e| anyhow!("{e}"))?;
        let ffn_up = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.mlp.up_proj.weight"),
            cfg.intermediate_size,
            cfg.hidden_size,
        )
        .map_err(|e| anyhow!("{e}"))?;
        let ffn_down = load_transposed_matrix_preserve_dtype(
            st,
            &format!("model.layers.{i}.mlp.down_proj.weight"),
            cfg.hidden_size,
            cfg.intermediate_size,
        )
        .map_err(|e| anyhow!("{e}"))?;
        let attn_norm_gain =
            load_tensor_as_f32(st, &format!("model.layers.{i}.input_layernorm.weight"))
                .map_err(|e| anyhow!("{e}"))?;
        let ffn_norm_gain = load_tensor_as_f32(
            st,
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
        )
        .map_err(|e| anyhow!("{e}"))?;
        // Qwen2 has Q/K/V biases.
        let attn_q_bias = load_tensor_as_f32(
            st,
            &format!("model.layers.{i}.self_attn.q_proj.bias"),
        )
        .ok()
        .map(Arc::from);
        let attn_k_bias = load_tensor_as_f32(
            st,
            &format!("model.layers.{i}.self_attn.k_proj.bias"),
        )
        .ok()
        .map(Arc::from);
        let attn_v_bias = load_tensor_as_f32(
            st,
            &format!("model.layers.{i}.self_attn.v_proj.bias"),
        )
        .ok()
        .map(Arc::from);
        layers.push(LayerWeights {
            attn_q,
            attn_q_bias,
            attn_k,
            attn_k_bias,
            attn_v,
            attn_v_bias,
            attn_o,
            ffn_gate,
            ffn_up,
            ffn_down,
            attn_norm_gain: Arc::from(attn_norm_gain),
            ffn_norm_gain: Arc::from(ffn_norm_gain),
        });
    }

    let final_norm_gain =
        load_tensor_as_f32(st, "model.norm.weight").map_err(|e| anyhow!("{e}"))?;
    // gte-Qwen has lm_head; if it's tied to embedding we fall back to
    // transposing the embedding matrix into `[hidden, vocab]` layout.
    // (gte-Qwen never reads logits — only hidden states — so this path
    // is only exercised when constructing the model.)
    let output: WeightStorage = match load_transposed_matrix_preserve_dtype(
        st,
        "lm_head.weight",
        cfg.vocab_size,
        cfg.hidden_size,
    ) {
        Ok(w) => w,
        Err(_) => {
            let mut transposed = vec![0.0_f32; cfg.hidden_size * cfg.vocab_size];
            for i in 0..cfg.vocab_size {
                for j in 0..cfg.hidden_size {
                    transposed[j * cfg.vocab_size + i] =
                        token_embedding[i * cfg.hidden_size + j];
                }
            }
            WeightStorage::F32(Arc::from(transposed))
        }
    };

    Ok(Qwen2Weights {
        token_embedding: Arc::from(token_embedding),
        layers,
        final_norm_gain: Arc::from(final_norm_gain),
        output,
    })
}

/// L2-normalize a flat `[hidden]` vector in-place semantics: returns a
/// fresh `Vec<f32>` whose Euclidean norm is 1.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let s: f32 = v.iter().map(|x| x * x).sum();
    let inv = 1.0_f32 / s.sqrt().max(1e-30);
    v.iter().map(|x| x * inv).collect()
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

    println!("Fetching model files...");
    let start = std::time::Instant::now();
    let config_files = match args.local_repo {
        Some(local_path) => load_from_local(&local_path)?,
        None => load_from_hub(&args.model_id, &args.revision)?,
    };
    println!("Model file retrieved in {:?}", start.elapsed());

    // Tokenizer setup (no padding — we run one prompt at a time).
    let tokenizer = Tokenizer::from_file(config_files.tokenizer).map_err(E::msg)?;

    let _device = fuel_examples::device(args.cpu)?;
    let config_str = std::fs::read_to_string(config_files.config)?;
    let hf_cfg: HfConfig = serde_json::from_str(&config_str)?;
    let cfg: Qwen2Config = hf_cfg.into();

    let start = std::time::Instant::now();
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&config_files.weights) }
        .map_err(|e| anyhow!("mmap safetensors: {e}"))?;
    let weights = load_qwen2_weights(&st, &cfg)?;
    let model = Qwen2Model { config: cfg.clone(), weights };
    println!("Model loaded in {:?}", start.elapsed());

    // Encode the queries and the targets
    let instruct = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ";
    let documents = vec![
        format!("{instruct}how much protein should a female eat{EOS_TOKEN}"),
        format!("{instruct}summit define{EOS_TOKEN}"),
        format!("As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.{EOS_TOKEN}"),
        format!("Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.{EOS_TOKEN}"),
    ];

    let start_gen = std::time::Instant::now();
    // Per-document forward; pull the last hidden state and L2-normalize.
    let h = cfg.hidden_size;
    let mut embeds: Vec<Vec<f32>> = Vec::with_capacity(documents.len());
    for doc in &documents {
        let encoded = tokenizer.encode(doc.as_str(), true).map_err(E::msg)?;
        let tokens: Vec<u32> = encoded.get_ids().to_vec();
        let hidden = model.forward_hidden(&tokens, 0)?;
        let data = hidden.realize_f32();
        let seq = tokens.len();
        // hidden shape: [1, seq, hidden]
        let last_offset = (seq - 1) * h;
        let last = &data[last_offset..last_offset + h];
        embeds.push(l2_normalize(last));
    }

    // scores = queries[0..2] @ targets[2..4].T → 2x2 matrix.
    let mut scores = Vec::with_capacity(4);
    for i in 0..2 {
        let mut row = Vec::with_capacity(2);
        for j in 0..2 {
            let q = &embeds[i];
            let t = &embeds[2 + j];
            let dot: f32 = q.iter().zip(t.iter()).map(|(a, b)| a * b).sum();
            row.push(dot);
        }
        scores.push(row);
    }

    println!("Embedding done in {:?}", start_gen.elapsed());
    println!("Scores: {scores:?}");

    Ok(())
}

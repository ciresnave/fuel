// Stella-en-v5 embedding model — lazy-graph port.
//
// v1 of the lazy port supports the 1.5B Large variant only.
// The 400M Small variant uses a BERT-RoPE backbone with token-type
// embeddings and absolute position scaling; its lazy port is a
// separate follow-up.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::path::Path;

use anyhow::{anyhow, Error as E, Result};
use clap::Parser;

use fuel::lazy::{
    load_tensor_as_f32, load_transposed_matrix_preserve_dtype, LayerWeights, WeightStorage,
};
use fuel::lazy_qwen2::{Qwen2Config, Qwen2Weights};
use fuel::lazy_stella_v5::{
    StellaEmbedDim, StellaV5Config, StellaV5Model, StellaV5Weights,
};
use hf_hub::{api::sync::Api, Repo};
use std::sync::Arc;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
pub enum EmbedDim {
    #[value(name = "256")]
    Dim256,
    #[value(name = "768")]
    Dim768,
    #[value(name = "1024")]
    Dim1024,
    #[value(name = "2048")]
    Dim2048,
    #[value(name = "4096")]
    Dim4096,
    #[value(name = "6144")]
    Dim6144,
    #[value(name = "8192")]
    Dim8192,
}

impl EmbedDim {
    fn embed_dim_default_dir(&self) -> &'static str {
        match self {
            Self::Dim256 => "2_Dense_256",
            Self::Dim768 => "2_Dense_768",
            Self::Dim1024 => "2_Dense_1024",
            Self::Dim2048 => "2_Dense_2048",
            Self::Dim4096 => "2_Dense_4096",
            Self::Dim6144 => "2_Dense_6144",
            Self::Dim8192 => "2_Dense_8192",
        }
    }
    fn to_lazy(self) -> StellaEmbedDim {
        match self {
            Self::Dim256 => StellaEmbedDim::Dim256,
            Self::Dim768 => StellaEmbedDim::Dim768,
            Self::Dim1024 => StellaEmbedDim::Dim1024,
            Self::Dim2048 => StellaEmbedDim::Dim2048,
            Self::Dim4096 => StellaEmbedDim::Dim4096,
            Self::Dim6144 => StellaEmbedDim::Dim6144,
            Self::Dim8192 => StellaEmbedDim::Dim8192,
        }
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
pub enum EncodeTask {
    /// `s2p` is the retrieval task (default).
    #[value(name = "s2p")]
    S2P,
    /// `s2s` is the semantic similarity task.
    #[value(name = "s2s")]
    S2S,
}

impl EncodeTask {
    pub fn query_preproc(&self, txt: &[String]) -> Vec<String> {
        let instruct = match self {
            Self::S2P => {
                "Given a web search query, retrieve relevant passages that answer the query."
            }
            Self::S2S => "Retrieve semantically similar text.",
        };
        txt.iter()
            .map(|s| format!("Instruct: {instruct}\nQuery: {s}"))
            .collect()
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "1.5b")]
    Large,
    #[value(name = "400m")]
    Small,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value = "1.5b")]
    which: Which,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    query: Option<String>,

    #[arg(long, default_value = "1024")]
    embed_dim: Option<EmbedDim>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    base_weight_files: Option<String>,

    #[arg(long)]
    embed_head_weight_files: Option<String>,

    /// `s2s`: Semantic textual similarity; `s2p`: Retrieval task (default).
    #[arg(long, default_value = "s2p")]
    task: Option<EncodeTask>,
}

// Stella's Large variant uses left padding with <|endoftext|> as the
// pad token (the model card asks for last-token mean pool over the
// right-most valid positions).
fn create_tokenizer(tokenizer_file: &Path, which: Which) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;
    if which == Which::Large {
        let pad_id = if let Some(pad_id) = tokenizer.token_to_id("<|endoftext|>") {
            pad_id
        } else {
            return Err(anyhow!(
                "Tokenizer doesn't contain expected `<|endoftext|>` token"
            ));
        };
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Left,
            pad_id,
            pad_token: "<|endoftext|>".to_string(),
            ..Default::default()
        }));
    } else {
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            ..Default::default()
        }));
    }
    Ok(tokenizer)
}

/// Build [`Qwen2Weights`] from a memory-mapped Stella safetensors
/// checkpoint. Stella-1.5B uses the standard Qwen2 HF tensor names
/// (`model.embed_tokens.weight`, `model.layers.{i}.self_attn.*.weight/bias`,
/// `model.layers.{i}.mlp.*.weight`, `model.norm.weight`, `lm_head.weight`).
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
    // Stella keeps lm_head as a passthrough scaffold; the embedding
    // projection happens through embed_head. Fall back to tied
    // embeddings if lm_head is absent.
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
    let _device = fuel_examples::device(args.cpu)?;

    if args.which == Which::Small {
        anyhow::bail!(
            "Stella 400M variant (Small) uses a BERT-RoPE backbone that the lazy port \
             does not yet support; only the 1.5B Large variant is migrated.",
        );
    }

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let embed_dim = args.embed_dim.unwrap_or(EmbedDim::Dim1024);

    let repo_id = "dunzhang/stella_en_1.5B_v5";
    let cfg = StellaV5Config::stella_en_1_5b_v5(embed_dim.to_lazy());

    let repo = api.repo(Repo::model(repo_id.to_string()));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };

    let base_weight_files = match args.base_weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => vec![repo.get("model.safetensors")?],
    };

    let embed_weight_files = match args.embed_head_weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            let head_w_path = format!("{}/model.safetensors", embed_dim.embed_dim_default_dir());
            vec![repo.get(&head_w_path)?]
        }
    };

    println!("retrieved the files in {:?}", start.elapsed());

    let tokenizer = create_tokenizer(tokenizer_filename.as_path(), args.which)?;

    let start = std::time::Instant::now();

    let base_st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&base_weight_files) }
        .map_err(|e| anyhow!("mmap base safetensors: {e}"))?;
    let embed_st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&embed_weight_files) }
        .map_err(|e| anyhow!("mmap embed_head safetensors: {e}"))?;

    let backbone = load_qwen2_weights(&base_st, &cfg.backbone)?;
    // embed_head — Stella stores the Matryoshka projection in
    // `linear.weight` of the per-dim Dense module.
    let out_features = embed_dim.to_lazy().out_features();
    let embed_head = load_transposed_matrix_preserve_dtype(
        &embed_st,
        "linear.weight",
        out_features,
        cfg.backbone.hidden_size,
    )
    .map_err(|e| anyhow!("load embed_head: {e}"))?;

    let weights = StellaV5Weights { backbone, embed_head };
    let model = StellaV5Model { config: cfg, weights };

    println!("loaded the model in {:?}", start.elapsed());

    let task = args.task.unwrap_or(EncodeTask::S2P);

    if let Some(text) = args.query {
        let qry = task.query_preproc(&[text]);
        let encoding = tokenizer
            .encode(qry[0].clone(), true)
            .map_err(|e| anyhow!(e))?;
        let tokens: Vec<u32> = encoding.get_ids().to_vec();
        let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let out = model.forward_with_mask(&tokens, &mask)?;
        let out_data = out.realize_f32();
        println!("embeddings: {:?}", &out_data[..out_data.len().min(8)]);
        println!("(showing first 8 of {} embedding dims)", out_data.len());
    } else {
        // Example queries from the model card.
        let queries = [
            "What are some ways to reduce stress?".to_string(),
            "What are the benefits of drinking green tea?".to_string(),
        ];
        let qry = task.query_preproc(&queries);
        for q in &qry {
            let encoding = tokenizer.encode(q.clone(), true).map_err(|e| anyhow!(e))?;
            let tokens: Vec<u32> = encoding.get_ids().to_vec();
            let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
            let out = model.forward_with_mask(&tokens, &mask)?;
            let out_data = out.realize_f32();
            println!(
                "query: {}\nembedding (first 8 of {}): {:?}",
                q,
                out_data.len(),
                &out_data[..out_data.len().min(8)],
            );
        }
    }

    Ok(())
}

// ModernBERT — lazy-graph port (fill-mask demo).
//
// The lazy port exposes the encoder backbone (`forward(...)` returns
// per-token hidden states). The MLM head — dense + LN + tied-embedding
// decoder + decoder.bias — is reconstructed here in main.rs from the
// raw safetensors weights.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Error as E, Result};
use clap::{Parser, ValueEnum};
use fuel::lazy::{load_tensor_as_f32, load_transposed_matrix, LazyTensor, WeightStorage};
use fuel::lazy_modernbert::{
    ModernBertConfig, ModernBertLayerWeights, ModernBertModel, ModernBertWeights,
};
use fuel::Shape;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, ValueEnum)]
enum Model {
    ModernBertBase,
    ModernBertLarge,
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

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long, default_value = "modern-bert-base")]
    model: Model,

    // Path to the tokenizer file.
    #[arg(long)]
    tokenizer_file: Option<String>,

    // Path to the weight files.
    #[arg(long)]
    weight_files: Option<String>,

    // Path to the config file.
    #[arg(long)]
    config_file: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,
}

/// Minimal HF `config.json` parse for the fields the lazy port needs.
#[derive(serde::Deserialize)]
struct HfConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_layer_norm_eps")]
    layer_norm_eps: f64,
    #[serde(default = "default_global_attn_every_n_layers")]
    global_attn_every_n_layers: usize,
    #[serde(default = "default_global_rope_theta")]
    global_rope_theta: f64,
    #[serde(default = "default_local_attention")]
    local_attention: usize,
    #[serde(default = "default_local_rope_theta")]
    local_rope_theta: f64,
}

fn default_layer_norm_eps() -> f64 { 1e-5 }
fn default_global_attn_every_n_layers() -> usize { 3 }
fn default_global_rope_theta() -> f64 { 160_000.0 }
fn default_local_attention() -> usize { 128 }
fn default_local_rope_theta() -> f64 { 10_000.0 }

impl From<HfConfig> for ModernBertConfig {
    fn from(c: HfConfig) -> Self {
        Self {
            vocab_size: c.vocab_size,
            hidden_size: c.hidden_size,
            num_hidden_layers: c.num_hidden_layers,
            num_attention_heads: c.num_attention_heads,
            intermediate_size: c.intermediate_size,
            max_position_embeddings: c.max_position_embeddings,
            layer_norm_eps: c.layer_norm_eps,
            global_attn_every_n_layers: c.global_attn_every_n_layers,
            global_rope_theta: c.global_rope_theta,
            local_attention: c.local_attention,
            local_rope_theta: c.local_rope_theta,
        }
    }
}

/// Load the encoder weights for the lazy `ModernBertModel`.
fn load_modernbert_weights(
    st: &fuel::safetensors::MmapedSafetensors,
    cfg: &ModernBertConfig,
) -> Result<ModernBertWeights> {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;

    let word_embedding =
        load_tensor_as_f32(st, "model.embeddings.tok_embeddings.weight")
            .map_err(|e| anyhow!("{e}"))?;
    let embed_ln_gain = load_tensor_as_f32(st, "model.embeddings.norm.weight")
        .map_err(|e| anyhow!("{e}"))?;

    let mut layers: Vec<ModernBertLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        // Layer 0 has no attn_norm in ModernBERT — the embedding LayerNorm
        // does the pre-norm. Try to load; absent → None.
        let attn_norm_gain = load_tensor_as_f32(
            st,
            &format!("model.layers.{i}.attn_norm.weight"),
        )
        .ok()
        .map(Arc::from);

        // Wqkv: `[3 * hidden, hidden]` HF → `[hidden, 3 * hidden]` fuel.
        let wqkv = load_transposed_matrix(
            st,
            &format!("model.layers.{i}.attn.Wqkv.weight"),
            3 * h,
            h,
        )
        .map_err(|e| anyhow!("{e}"))?;
        // Wo: `[hidden, hidden]` HF → `[hidden, hidden]` (square, transposed).
        let wo = load_transposed_matrix(
            st,
            &format!("model.layers.{i}.attn.Wo.weight"),
            h,
            h,
        )
        .map_err(|e| anyhow!("{e}"))?;

        let mlp_norm_gain =
            load_tensor_as_f32(st, &format!("model.layers.{i}.mlp_norm.weight"))
                .map_err(|e| anyhow!("{e}"))?;

        // Wi: `[2 * intermediate, hidden]` HF → `[hidden, 2 * intermediate]` fuel.
        let mlp_wi = load_transposed_matrix(
            st,
            &format!("model.layers.{i}.mlp.Wi.weight"),
            2 * inter,
            h,
        )
        .map_err(|e| anyhow!("{e}"))?;
        // Wo: `[hidden, intermediate]` HF → `[intermediate, hidden]` fuel.
        let mlp_wo = load_transposed_matrix(
            st,
            &format!("model.layers.{i}.mlp.Wo.weight"),
            h,
            inter,
        )
        .map_err(|e| anyhow!("{e}"))?;

        layers.push(ModernBertLayerWeights {
            attn_norm_gain,
            wqkv: WeightStorage::F32(Arc::from(wqkv)),
            wo: WeightStorage::F32(Arc::from(wo)),
            mlp_norm_gain: Arc::from(mlp_norm_gain),
            mlp_wi: WeightStorage::F32(Arc::from(mlp_wi)),
            mlp_wo: WeightStorage::F32(Arc::from(mlp_wo)),
        });
    }

    let final_norm_gain = load_tensor_as_f32(st, "model.final_norm.weight")
        .map_err(|e| anyhow!("{e}"))?;

    Ok(ModernBertWeights {
        word_embedding: Arc::from(word_embedding),
        embed_ln_gain: Arc::from(embed_ln_gain),
        layers,
        final_norm_gain: Arc::from(final_norm_gain),
    })
}

/// MLM head weights: dense + norm + decoder.bias. Decoder weights are
/// tied to the input token embedding.
struct MlmHead {
    head_dense: Arc<[f32]>,    // `[hidden, hidden]` in fuel `[in, out]` layout
    head_norm_gain: Arc<[f32]>, // `[hidden]`
    decoder_bias: Arc<[f32]>,  // `[vocab]`
}

fn load_mlm_head(
    st: &fuel::safetensors::MmapedSafetensors,
    cfg: &ModernBertConfig,
) -> Result<MlmHead> {
    let h = cfg.hidden_size;
    let head_dense = load_transposed_matrix(st, "head.dense.weight", h, h)
        .map_err(|e| anyhow!("{e}"))?;
    let head_norm_gain = load_tensor_as_f32(st, "head.norm.weight")
        .map_err(|e| anyhow!("{e}"))?;
    let decoder_bias = load_tensor_as_f32(st, "decoder.bias")
        .map_err(|e| anyhow!("{e}"))?;
    Ok(MlmHead {
        head_dense: Arc::from(head_dense),
        head_norm_gain: Arc::from(head_norm_gain),
        decoder_bias: Arc::from(decoder_bias),
    })
}

/// Apply MLM head to per-token hidden states `[1, seq, hidden]`.
/// Returns logits `[1, seq, vocab]`.
fn apply_mlm_head(
    hidden: &LazyTensor,
    mlm: &MlmHead,
    word_embedding: &Arc<[f32]>,
    cfg: &ModernBertConfig,
) -> Result<LazyTensor> {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;

    // head.dense: linear (no bias)
    let dense_t = hidden.const_f32_like(
        Arc::clone(&mlm.head_dense),
        Shape::from_dims(&[h, h]),
    );
    let x = hidden.matmul(&dense_t)?;
    // GELU
    let x = x.gelu_erf();
    // head.norm: layer norm, no bias
    let zero_bias: Arc<[f32]> = Arc::from(vec![0.0_f32; h]);
    let x = x.layer_norm_affine(
        Arc::clone(&mlm.head_norm_gain),
        zero_bias,
        cfg.layer_norm_eps,
    )?;
    // Decoder: tied to input embedding. word_embedding has layout
    // `[vocab, hidden]`. The matmul `x @ W^T` is what eager Linear does,
    // but our LazyTensor.matmul expects `[hidden, vocab]`. We transpose
    // word_embedding once into a `[hidden, vocab]` const.
    let mut decoder_w = vec![0.0_f32; h * v];
    for vi in 0..v {
        for j in 0..h {
            decoder_w[j * v + vi] = word_embedding[vi * h + j];
        }
    }
    let decoder_w_t = hidden.const_f32_like(
        Arc::<[f32]>::from(decoder_w),
        Shape::from_dims(&[h, v]),
    );
    let logits = x.matmul(&decoder_w_t)?;
    // Add decoder.bias broadcast over [1, seq].
    let bias_t = hidden
        .const_f32_like(Arc::clone(&mlm.decoder_bias), Shape::from_dims(&[v]))
        .reshape(Shape::from_dims(&[1, 1, v]))?;
    let logits = logits.broadcast_add(&bias_t)?;
    Ok(logits)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let api = Api::new()?;
    let model_id = match &args.model_id {
        Some(model_id) => model_id.to_string(),
        None => match args.model {
            Model::ModernBertBase => "answerdotai/ModernBERT-base".to_string(),
            Model::ModernBertLarge => "answerdotai/ModernBERT-large".to_string(),
        },
    };
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

    let _device = fuel_examples::device(args.cpu)?;
    let config_str = std::fs::read_to_string(config_filename)?;
    let hf_cfg: HfConfig = serde_json::from_str(&config_str)?;
    let cfg: ModernBertConfig = hf_cfg.into();

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[weights_filename]) }
        .map_err(|e| anyhow!("mmap safetensors: {e}"))?;
    let weights = load_modernbert_weights(&st, &cfg)?;
    let mlm = load_mlm_head(&st, &cfg)?;
    let model = ModernBertModel { config: cfg.clone(), weights };

    // Lazy port runs batch=1; we iterate prompts one at a time.
    let prompts: Vec<String> = match args.prompt {
        Some(p) => vec![p],
        None => vec![
            "Hello I'm a [MASK] model.".to_string(),
            "I'm a [MASK] boy.".to_string(),
            "I'm [MASK] in berlin.".to_string(),
            "The capital of France is [MASK].".to_string(),
        ],
    };

    for (i, prompt) in prompts.iter().enumerate() {
        let encoded = tokenizer.encode(prompt.as_str(), true).map_err(E::msg)?;
        let tokens: Vec<u32> = encoded.get_ids().to_vec();

        let hidden = model.forward(&tokens, None)?;
        let logits = apply_mlm_head(
            &hidden,
            &mlm,
            &model.weights.word_embedding,
            &cfg,
        )?;
        let argmax = logits.argmax_dim(2_usize)?;
        let argmax_data = argmax.realize_u32();

        let decoded = tokenizer
            .decode(argmax_data.as_slice(), true)
            .map_err(E::msg)?;
        println!("Sentence: {} : {}", i + 1, decoded);
    }

    Ok(())
}

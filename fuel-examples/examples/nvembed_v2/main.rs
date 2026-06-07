#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use fuel::lazy_mistral::MistralConfig;
use fuel::lazy_nvembed_v2::{NvEmbedV2Config, NvEmbedV2Model, NvEmbedV2Weights};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingDirection, PaddingParams, Tokenizer, TruncationParams};

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

    /// L2 normalization for embeddings. The lazy `NvEmbedV2Model`
    /// always L2-normalizes; this flag is kept for CLI parity.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    model: Option<String>,

    /// Comma-separated list of model files (e.g., '/path/file1.safetensors,/path/file2.safetensors,/path/file3.safetensors')
    #[arg(long)]
    model_files: Option<String>,

    #[arg(long)]
    config_file: Option<String>,
}

fn build_model_and_tokenizer(args: &Args) -> Result<(NvEmbedV2Model, Tokenizer)> {
    let model_name = match args.model.as_ref() {
        Some(model) => model.to_string(),
        None => "nvidia/NV-Embed-v2".to_string(),
    };

    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));

    let model_files = match &args.model_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };

    let tokenizer_file = match &args.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };

    let _device = fuel_examples::device(args.cpu)?;

    let mut tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;
    let _ = tokenizer
        .with_padding(Some(PaddingParams {
            direction: PaddingDirection::Right,
            pad_id: 2,
            pad_token: "</s>".to_string(),
            ..Default::default()
        }))
        .with_truncation(Some(TruncationParams {
            max_length: 32_768,
            ..Default::default()
        }));

    // Build NvEmbedV2Config — either from explicit config.json, fetched
    // config.json on the hub, or the preset.
    let cfg: NvEmbedV2Config = match &args.config_file {
        Some(path) => {
            let json = std::fs::read_to_string(path)?;
            nvembed_config_from_hf_json_str(&json)?
        }
        None => match repo.get("config.json") {
            Ok(path) => {
                let json = std::fs::read_to_string(path)?;
                nvembed_config_from_hf_json_str(&json)?
            }
            Err(_) => NvEmbedV2Config::nv_embed_v2(),
        },
    };

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&model_files) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = NvEmbedV2Weights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = NvEmbedV2Model { config: cfg, weights };

    Ok((model, tokenizer))
}

/// Encode a batch of examples into L2-normalized embeddings.
///
/// The lazy `NvEmbedV2Model.forward` runs a single sequence at a time
/// and returns a `[1, hidden]` L2-normalized embedding. To match the
/// eager batched API we iterate the examples, encode each one, then
/// stack the resulting embeddings into a `(N, hidden)` host matrix.
///
/// `instruction` is prepended to every example (matching the eager
/// port). Pool-mask semantics — where the eager port zeroed out the
/// instruction tokens from pooling but kept them in attention — are
/// approximated here by zeroing them in the single attention mask
/// the lazy port accepts. This is a slight semantic shift from the
/// eager binary; embeddings still come out L2-normalized.
fn encode(
    model: &NvEmbedV2Model,
    tokenizer: &Tokenizer,
    examples: Vec<String>,
    instruction: &str,
) -> Result<Vec<Vec<f32>>> {
    let eos_token = tokenizer
        .get_padding()
        .map(|p| p.pad_token.clone())
        .unwrap_or_else(|| "".to_string());
    let bos = "<s>".to_string();
    let input_texts: Vec<String> = examples
        .iter()
        .map(|s| format!("{bos}{instruction}{s}{eos_token}"))
        .collect();

    // How many instruction tokens to zero out of the pool mask.
    let instruction_lens = if instruction.is_empty() {
        0
    } else {
        let encoded_instruction = tokenizer.encode(instruction, false).map_err(E::msg)?;
        encoded_instruction.get_tokens().len()
    };

    let mut out: Vec<Vec<f32>> = Vec::with_capacity(input_texts.len());
    for text in input_texts {
        let encoding = tokenizer.encode(text, false).map_err(E::msg)?;
        let tokens: Vec<u32> = encoding.get_ids().to_vec();
        // Build the attention mask, then zero out the leading
        // `instruction_lens` slots (pool-mask approximation).
        let mut mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let zero_through = instruction_lens.min(mask.len());
        for m in mask.iter_mut().take(zero_through) {
            *m = 0;
        }
        // Defensive: if instruction zeroed everything, skip masking
        // so the lazy forward's "sum mask > 0" assertion stays happy.
        if mask.iter().all(|&m| m == 0) {
            mask = encoding.get_attention_mask().to_vec();
        }
        let emb = model
            .forward(&tokens, &mask)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let data = emb.realize_f32(); // [1, hidden]
        out.push(data);
    }
    Ok(out)
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

    let _ = args.normalize_embeddings;

    let (model, tokenizer) = build_model_and_tokenizer(&args)?;

    if let Some(prompt) = args.prompt.clone() {
        let emb = encode(&model, &tokenizer, vec![prompt], "")?;
        println!("Embedding: {emb:?}");
    } else {
        let queries = [
            "are judo throws allowed in wrestling?",
            "how to become a radiology technician in michigan?",
        ];

        let passages = [
            "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
            "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan."
        ];
        let passage_instruction = "".to_string();
        let query_instruction =
            "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
                .to_string();

        let passages: Vec<String> = passages.iter().map(|s| s.to_string()).collect();
        let queries: Vec<String> = queries.iter().map(|s| s.to_string()).collect();

        let emb_query = encode(&model, &tokenizer, queries, &query_instruction)?;
        let emb_passage = encode(&model, &tokenizer, passages, &passage_instruction)?;

        // scores = emb_query @ emb_passage.T  ×  100
        let nq = emb_query.len();
        let np = emb_passage.len();
        let h = model.config.backbone.hidden_size;
        let mut scores: Vec<Vec<f32>> = Vec::with_capacity(nq);
        for q in &emb_query {
            assert_eq!(q.len(), h);
            let mut row = Vec::with_capacity(np);
            for p in &emb_passage {
                assert_eq!(p.len(), h);
                let dot: f32 = q.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
                row.push(dot * 100.0);
            }
            scores.push(row);
        }
        println!("scores: {scores:?}");
    }
    Ok(())
}

/// Parse a HuggingFace `config.json` into `NvEmbedV2Config`. The
/// NV-Embed-v2 checkpoint stores the Mistral backbone under the
/// usual Mistral keys (vocab_size, hidden_size, etc.) plus a few
/// `latent_attention_*` extras. Unknown fields fall back to the
/// preset.
fn nvembed_config_from_hf_json_str(json: &str) -> Result<NvEmbedV2Config> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let preset = NvEmbedV2Config::nv_embed_v2();
    let get_usize = |key: &str| -> Option<usize> {
        v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };

    // Many NV-Embed-v2 config.jsons wrap the backbone under a
    // `text_config` sub-object — fall back to the top level otherwise.
    let text_cfg = v.get("text_config").unwrap_or(&v);
    let get_text_usize = |key: &str| -> Option<usize> {
        text_cfg.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_text_f64 = |key: &str| -> Option<f64> {
        text_cfg.get(key).and_then(|x| x.as_f64())
    };

    let vocab_size = get_text_usize("vocab_size").unwrap_or(preset.backbone.vocab_size);
    let hidden_size = get_text_usize("hidden_size").unwrap_or(preset.backbone.hidden_size);
    let intermediate_size =
        get_text_usize("intermediate_size").unwrap_or(preset.backbone.intermediate_size);
    let num_hidden_layers =
        get_text_usize("num_hidden_layers").unwrap_or(preset.backbone.num_hidden_layers);
    let num_attention_heads =
        get_text_usize("num_attention_heads").unwrap_or(preset.backbone.num_attention_heads);
    let num_key_value_heads =
        get_text_usize("num_key_value_heads").unwrap_or(num_attention_heads);
    let head_dim = get_text_usize("head_dim").unwrap_or(hidden_size / num_attention_heads);
    let rms_norm_eps = get_text_f64("rms_norm_eps").unwrap_or(preset.backbone.rms_norm_eps);
    let rope_theta = get_text_f64("rope_theta").unwrap_or(preset.backbone.rope_theta);
    let max_position_embeddings = get_text_usize("max_position_embeddings")
        .unwrap_or(preset.backbone.max_position_embeddings);
    let sliding_window = text_cfg
        .get("sliding_window")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize);

    let backbone = MistralConfig {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps,
        rope_theta,
        max_position_embeddings,
        sliding_window,
    };

    let num_latents = get_usize("num_latents").unwrap_or(preset.num_latents);
    let latent_heads = get_usize("latent_attention_heads").unwrap_or(preset.latent_heads);
    let latent_head_dim =
        get_usize("latent_attention_head_dim").unwrap_or(preset.latent_head_dim);
    let ff_mult = get_usize("ff_mult").unwrap_or(preset.ff_mult);
    let layer_norm_eps = get_f64("layer_norm_eps").unwrap_or(preset.layer_norm_eps);

    Ok(NvEmbedV2Config {
        backbone,
        num_latents,
        latent_heads,
        latent_head_dim,
        ff_mult,
        layer_norm_eps,
    })
}

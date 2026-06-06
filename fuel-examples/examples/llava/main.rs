//! LLaVA — lazy port.
//!
//! The eager binary ran a full conversation loop with KV-cache iteration,
//! conversation templates, and image-token interleaving. The lazy port at
//! `fuel::lazy_llava` currently exposes a single-pass `forward` that
//! consumes `(pixel_values, &[u32] text_tokens)` and returns logits for
//! the concatenated `[image_features; text_embeds]` sequence. The
//! generation loop / KV cache / conversation templates are deferred to
//! follow-up work on the lazy module.
//!
//! What this binary does today:
//!   1. Loads the HF LLaVA config + tokenizer + image preprocessor.
//!   2. Loads the image, preprocesses it, and wraps the result in a
//!      lazy `(1, 3, H, W)` f32 tensor.
//!   3. Builds a lazy `LlavaConfig` from the HF JSON.
//!   4. Loads weights from safetensors via `LlavaWeights::load_from_mmapped`.
//!   5. Tokenizes the prompt (with `<image>` markers stripped — the
//!      v1 lazy path splices the image features in front automatically).
//!   6. Runs a single `LlavaModel::forward` and greedy-decodes the
//!      max-logit next token as a smoke test.

pub mod constants;
pub mod conversation;
pub mod image_processor;

use anyhow::{bail, Error as E, Result};
use clap::Parser;
use constants::*;
use conversation::Conversation;
use fuel::lazy::{LazyTensor, LlamaConfig};
use fuel::lazy_clip::ClipVisionConfig;
use fuel::lazy_llava::{LlavaConfig, LlavaModel, LlavaWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use fuel_transformers::models::llava::config::{
    HFGenerationConfig, HFLLaVAConfig, HFPreProcessorConfig, LLaVAConfig,
};
use hf_hub::api::sync::Api;
use image_processor::{process_image, ImageProcessor};
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "llava-hf/llava-v1.6-vicuna-7b-hf")]
    model_path: String,
    #[arg(long, default_value = "tokenizer/tokenizer.json")]
    tokenizer_path: String,
    #[arg(long)]
    image_file: String,
    #[arg(long)]
    conv_mode: Option<String>,
    #[arg(long, action)]
    hf: bool,
    #[arg(long, action)]
    cpu: bool,
    #[arg(long)]
    prompt: String,
}

fn get_model_name_from_path(model_path: &str) -> String {
    let model_paths: Vec<String> = model_path
        .trim_matches('/')
        .split('/')
        .map(|s| s.to_string())
        .collect();
    if model_paths.last().unwrap().starts_with("checkpoint-") {
        format!(
            "{}_{}",
            model_paths[model_paths.len() - 2],
            model_paths.last().unwrap()
        )
    } else {
        model_paths.last().unwrap().to_string()
    }
}

fn lazy_llava_config_from_hf(hf: &HFLLaVAConfig) -> LlavaConfig {
    let vision_config = ClipVisionConfig {
        embed_dim: hf.vision_config.hidden_size,
        intermediate_size: hf.vision_config.intermediate_size,
        num_hidden_layers: hf.vision_config.num_hidden_layers,
        num_attention_heads: hf.vision_config.num_attention_heads,
        projection_dim: hf.vision_config.projection_dim,
        num_channels: 3,
        image_size: hf.vision_config.image_size,
        patch_size: hf.vision_config.patch_size,
    };
    let dim = hf.text_config.hidden_size;
    let n_heads = hf.text_config.num_attention_heads;
    let text_config = LlamaConfig {
        vocab_size: hf.vocab_size,
        dim,
        n_layers: hf.text_config.num_hidden_layers,
        n_heads,
        n_kv_heads: hf.text_config.num_key_value_heads,
        head_dim: dim / n_heads,
        ffn_dim: hf.text_config.intermediate_size,
        norm_eps: hf.text_config.rms_norm_eps as f64,
        rope_base: hf.text_config.rope_theta as f64,
    };
    LlavaConfig {
        vision_config,
        text_config,
        // v1 only supports the "linear" projector where projection_dim == text_dim.
        projection_dim: dim,
    }
}

fn lazy_llava_config_from_local(local: &LLaVAConfig, image_size: usize) -> LlavaConfig {
    // The non-HF (liuhaotian) config doesn't carry a separate vision-config
    // block; use CLIP ViT-L/14 defaults plus the dynamic image size.
    let vision_config = ClipVisionConfig {
        embed_dim: 1024,
        intermediate_size: 4096,
        num_hidden_layers: 24,
        num_attention_heads: 16,
        projection_dim: 768,
        num_channels: 3,
        image_size,
        patch_size: 14,
    };
    let dim = local.hidden_size;
    let n_heads = local.num_attention_heads;
    let text_config = LlamaConfig {
        vocab_size: local.vocab_size,
        dim,
        n_layers: local.num_hidden_layers,
        n_heads,
        n_kv_heads: local.num_key_value_heads,
        head_dim: dim / n_heads,
        ffn_dim: local.intermediate_size,
        norm_eps: local.rms_norm_eps as f64,
        rope_base: local.rope_theta as f64,
    };
    LlavaConfig {
        vision_config,
        text_config,
        projection_dim: dim,
    }
}

fn main() -> Result<()> {
    let mut args = Args::parse();

    // Lazy realizes through CPU/router; `cpu` flag preserved for CLI parity.
    let _ = args.cpu;
    let device = Device::cpu();

    println!("Start loading model");
    let api = Api::new()?;
    let api = api.model(args.model_path.clone());
    let (llava_config, tokenizer, image_processor) = if args.hf {
        let config_filename = api.get("config.json")?;
        let hf_llava_config: HFLLaVAConfig =
            serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let generation_config_filename = api.get("generation_config.json")?;
        let generation_config: HFGenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_filename)?)?;
        let preprocessor_config_filename = api.get("preprocessor_config.json")?;
        let preprocessor_config: HFPreProcessorConfig =
            serde_json::from_slice(&std::fs::read(preprocessor_config_filename)?)?;
        let llava_config =
            hf_llava_config.to_llava_config(&generation_config, &preprocessor_config);
        let tokenizer_filename = api.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let proc = ImageProcessor::from_hf_preprocessor_config(&preprocessor_config);
        (llava_config, tokenizer, proc)
    } else {
        let config_filename = api.get("config.json")?;
        let llava_config: LLaVAConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(&args.tokenizer_path)
            .map_err(|e| E::msg(format!("Error loading {}: {}", &args.tokenizer_path, e)))?;
        let vt = llava_config
            .mm_vision_tower
            .clone()
            .ok_or_else(|| E::msg("non-HF config missing mm_vision_tower"))?;
        let proc = ImageProcessor::from_pretrained(&vt)?;
        (llava_config, tokenizer, proc)
    };

    let eos_token_id = llava_config.eos_token_id as u32;

    // Build the lazy LlavaConfig.
    let lazy_cfg = if args.hf {
        // Re-fetch + re-parse the HF JSON so we can read the raw vision/
        // text sub-configs (the LLaVAConfig wrapper flattens them).
        let config_filename = api.get("config.json")?;
        let hf_llava_config: HFLLaVAConfig =
            serde_json::from_slice(&std::fs::read(config_filename)?)?;
        lazy_llava_config_from_hf(&hf_llava_config)
    } else {
        lazy_llava_config_from_local(&llava_config, 336)
    };

    println!("loading model weights");
    let weight_filenames =
        fuel_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?;
    let st = unsafe { MmapedSafetensors::multi(&weight_filenames) }?;
    let weights = LlavaWeights::load_from_mmapped(&st, &lazy_cfg)?;
    let model = LlavaModel {
        config: lazy_cfg.clone(),
        weights,
    };

    println!("generating conv template");
    let image_token_se =
        format!("{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}");
    let qs = if args.prompt.contains(IMAGE_PLACEHOLDER) {
        if llava_config.mm_use_im_start_end {
            args.prompt.replace(IMAGE_PLACEHOLDER, &image_token_se)
        } else {
            args.prompt.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
        }
    } else if llava_config.mm_use_im_start_end {
        format!("{}\n{}", image_token_se, args.prompt)
    } else {
        format!("{}\n{}", DEFAULT_IMAGE_TOKEN, args.prompt)
    };

    let model_name = get_model_name_from_path(&args.model_path).to_lowercase();
    let conv_mode = if model_name.contains("llama-2") {
        "llava_llama_2"
    } else if model_name.contains("mistral") {
        "mistral_instruct"
    } else if model_name.contains("v1.6-34b") {
        "chatml_direct"
    } else if model_name.contains("v1") {
        "llava_v1"
    } else if model_name.contains("mpt") {
        "mpt"
    } else {
        "llava_v0"
    };
    if args.conv_mode.is_some() && args.conv_mode.as_deref() != Some(conv_mode) {
        println!(
            "Warning: the model is trained with {}, but you are using {}",
            conv_mode,
            args.conv_mode.as_deref().unwrap()
        );
    } else {
        args.conv_mode = Some(conv_mode.to_string());
    }

    let mut conv = match args.conv_mode {
        Some(conv_mode) => match conv_mode.as_str() {
            "chatml_direct" => Conversation::conv_chatml_direct(),
            "llava_v1" => Conversation::conv_llava_v1(),
            _ => bail!("conversation mode not implemented in the lazy v1 binary"),
        },
        None => bail!("conv_mode is required"),
    };
    conv.append_user_message(Some(&qs));
    conv.append_assistant_message(None);
    let prompt = conv.get_prompt();

    println!("loading image");
    let img = image::ImageReader::open(&args.image_file)?.decode()?;
    let image_tensor = process_image(&img, &image_processor, &llava_config)?;
    // process_image returns shape (1, 3, H, W) — flatten and rebuild as lazy.
    let dims = image_tensor.dims().to_vec();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
        bail!(
            "expected (1, 3, H, W) image tensor; got {:?}. Anyres/pad aspect \
             ratios are not supported by the lazy v1 binary.",
            dims
        );
    }
    let image_vec: Vec<f32> = image_tensor.flatten_all()?.to_vec1::<f32>()?;
    let pixel_values = LazyTensor::from_f32(
        Arc::<[f32]>::from(image_vec),
        Shape::from_dims(&dims),
        &device,
    );

    // Tokenize the prompt. The lazy v1 path doesn't yet support
    // image-token interleaving (it always prepends image features),
    // so we strip the `<image>` placeholder from the prompt before
    // tokenizing.
    let plain_prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, "");
    let token_ids: Vec<u32> = tokenizer
        .encode(plain_prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    if token_ids.is_empty() {
        bail!("tokenizer produced empty token list for prompt");
    }

    // Single-pass forward → logits over (1, np + text_len, vocab).
    let logits = model.forward(&pixel_values, &token_ids)?;
    let logits_vec = logits.realize_f32();
    let dims = logits.shape();
    let dims = dims.dims();
    if dims.len() != 3 {
        bail!("expected (1, seq, vocab) logits; got {:?}", dims);
    }
    let vocab = dims[2];
    let seq = dims[1];
    // Greedy argmax over the last position.
    let last = &logits_vec[(seq - 1) * vocab..seq * vocab];
    let (next_tok_idx, _) = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("non-empty logits row");
    let next_token = next_tok_idx as u32;
    println!("greedy next token id: {next_token}");
    if next_token == eos_token_id {
        println!("(was EOS)");
    } else if let Ok(decoded) = tokenizer.decode(&[next_token], true) {
        println!("decoded: {decoded}");
    }
    Ok(())
}

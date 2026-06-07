#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::{Parser, ValueEnum};
use std::sync::Arc;

use fuel::lazy::LazyTensor;
use fuel::lazy_trocr::{
    TrocrActivation, TrocrDecoderConfig, TrocrModel,
};
use fuel::lazy_vit::{VitActivation, VitConfig};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use fuel_examples::token_output_stream::TokenOutputStream;
use fuel_transformers::models::{trocr, vit};

use tokenizers::Tokenizer;
mod image_processor;

#[derive(Clone, Debug, Copy, ValueEnum)]
enum Which {
    #[value(name = "base")]
    BaseHandwritten,
    #[value(name = "large")]
    LargeHandwritten,
    BasePrinted,
    LargePrinted,
}

impl Which {
    fn repo_and_branch_name(&self) -> (&str, &str) {
        match self {
            Self::BaseHandwritten => ("microsoft/trocr-base-handwritten", "refs/pr/3"),
            Self::LargeHandwritten => ("microsoft/trocr-large-handwritten", "refs/pr/6"),
            Self::BasePrinted => ("microsoft/trocr-base-printed", "refs/pr/7"),
            Self::LargePrinted => ("microsoft/trocr-large-printed", "main"),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Config {
    encoder: vit::Config,
    decoder: trocr::TrOCRConfig,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    /// Choose the variant of the model to run.
    #[arg(long, default_value = "base")]
    which: Which,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The image file to be processed.
    #[arg(long)]
    image: String,

    /// Tokenization config.
    #[arg(long)]
    tokenizer: Option<String>,
}

/// Map an eager `fuel_nn::Activation` to the lazy `VitActivation`
/// (subset supported by `lazy_vit`).
fn map_vit_activation(act: fuel_nn::Activation) -> anyhow::Result<VitActivation> {
    Ok(match act {
        fuel_nn::Activation::Gelu => VitActivation::Gelu,
        fuel_nn::Activation::GeluPytorchTanh => VitActivation::GeluPytorchTanh,
        fuel_nn::Activation::Relu => VitActivation::Relu,
        fuel_nn::Activation::Silu => VitActivation::Silu,
        other => anyhow::bail!("unsupported ViT activation in lazy_vit: {other:?}"),
    })
}

/// Map an eager `fuel_nn::Activation` to the lazy `TrocrActivation`.
fn map_trocr_activation(act: fuel_nn::Activation) -> anyhow::Result<TrocrActivation> {
    Ok(match act {
        fuel_nn::Activation::Gelu => TrocrActivation::Gelu,
        fuel_nn::Activation::Relu => TrocrActivation::Relu,
        other => anyhow::bail!("unsupported TrOCR decoder activation: {other:?}"),
    })
}

fn lazy_vit_config_from_eager(cfg: &vit::Config) -> anyhow::Result<VitConfig> {
    Ok(VitConfig {
        hidden_size: cfg.hidden_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        intermediate_size: cfg.intermediate_size,
        hidden_activation: map_vit_activation(cfg.hidden_act)?,
        layer_norm_eps: cfg.layer_norm_eps,
        image_size: cfg.image_size,
        patch_size: cfg.patch_size,
        num_channels: cfg.num_channels,
        qkv_bias: cfg.qkv_bias,
    })
}

fn lazy_trocr_config_from_eager(cfg: &trocr::TrOCRConfig) -> anyhow::Result<TrocrDecoderConfig> {
    Ok(TrocrDecoderConfig {
        vocab_size: cfg.decoder_vocab_size.unwrap_or(cfg.vocab_size),
        d_model: cfg.d_model,
        cross_attention_hidden_size: cfg.cross_attention_hidden_size,
        decoder_layers: cfg.decoder_layers,
        decoder_attention_heads: cfg.decoder_attention_heads,
        decoder_ffn_dim: cfg.decoder_ffn_dim,
        activation_function: map_trocr_activation(cfg.activation_function)?,
        max_position_embeddings: cfg.max_position_embeddings,
        // HF convention: learned positional embedding offset is 2.
        learned_pos_offset: 2,
        scale_embedding: cfg.scale_embedding,
        tie_word_embeddings: cfg.tie_word_embeddings,
    })
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let api = hf_hub::api::sync::Api::new()?;

    let mut tokenizer_dec = {
        let tokenizer_file = match args.tokenizer {
            None => api
                .model(String::from("ToluClassics/fuel-trocr-tokenizer"))
                .get("tokenizer.json")?,
            Some(tokenizer) => std::path::PathBuf::from(tokenizer),
        };
        let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(E::msg)?;
        TokenOutputStream::new(tokenizer)
    };

    // Lazy realizes via the default executor (CPU/router). The
    // `--cpu` flag is preserved for CLI parity with the eager
    // binary but has no effect here.
    let _ = args.cpu;
    let device = Device::cpu();

    let model_file = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => {
            let (repo, branch) = args.which.repo_and_branch_name();
            api.repo(hf_hub::Repo::with_revision(
                repo.to_string(),
                hf_hub::RepoType::Model,
                branch.to_string(),
            ))
            .get("model.safetensors")?
        }
    };
    println!("model: {model_file:?}");

    let (encoder_config_eager, decoder_config_eager) = {
        let (repo, branch) = args.which.repo_and_branch_name();
        let config_filename = api
            .repo(hf_hub::Repo::with_revision(
                repo.to_string(),
                hf_hub::RepoType::Model,
                branch.to_string(),
            ))
            .get("config.json")?;
        let config: Config = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        (config.encoder, config.decoder)
    };
    let decoder_start_token_id = decoder_config_eager.decoder_start_token_id;
    let eos_token_id = decoder_config_eager.eos_token_id;

    let encoder_config = lazy_vit_config_from_eager(&encoder_config_eager)?;
    let decoder_config = lazy_trocr_config_from_eager(&decoder_config_eager)?;

    println!("loading model weights");
    let st = unsafe { MmapedSafetensors::multi(&[model_file]) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let model = TrocrModel::load_from_mmapped(&st, encoder_config.clone(), decoder_config.clone())
        .map_err(|e| E::msg(format!("load TrOCR weights: {e}")))?;

    let processor_config = image_processor::ProcessorConfig::default();
    let processor = image_processor::ViTImageProcessor::new(&processor_config);

    // Preprocess the image with the eager pipeline (still uses
    // `fuel::Tensor` for CV ops), then convert to a lazy tensor.
    let image_eager = processor.preprocess(vec![args.image.as_str()])?;
    let image_dims = image_eager.dims().to_vec();
    if image_dims.len() != 4 || image_dims[0] != 1 || image_dims[1] != 3 {
        anyhow::bail!(
            "expected (1, 3, H, W) image tensor; got {:?}",
            image_dims
        );
    }
    let image_vec: Vec<f32> = image_eager.flatten_all()?.to_vec1::<f32>()?;
    let pixel_values = LazyTensor::from_f32(
        Arc::<[f32]>::from(image_vec),
        Shape::from_dims(&image_dims),
        &device,
    );

    // Encode once; the lazy decoder takes the encoder output as a
    // graph anchor and re-uses it across autoregressive steps.
    let encoder_xs = model
        .forward_encoder(&pixel_values)
        .map_err(|e| E::msg(format!("encoder forward: {e}")))?;

    let vocab_size = decoder_config.vocab_size;
    let mut token_ids: Vec<u32> = vec![decoder_start_token_id];
    for _ in 0..1000 {
        // The lazy decoder takes the FULL target sequence each step
        // (it has no KV cache yet). Greedy argmax over the last
        // position — matches blip's v1 sampling.
        let logits = model
            .forward_decoder(&token_ids, &encoder_xs)
            .map_err(|e| E::msg(format!("decoder forward: {e}")))?;
        let data = logits.realize_f32();
        let seq = token_ids.len();
        let off = (seq - 1) * vocab_size;
        let last_logits = &data[off..off + vocab_size];

        let mut best_i = 0usize;
        let mut best = last_logits[0];
        for (i, &v) in last_logits.iter().enumerate().skip(1) {
            if v > best {
                best = v;
                best_i = i;
            }
        }
        let token = best_i as u32;
        token_ids.push(token);

        if let Some(t) = tokenizer_dec.next_token(token)? {
            use std::io::Write;
            print!("{t}");
            std::io::stdout().flush()?;
        }
        if token == eos_token_id {
            break;
        }
    }

    if let Some(rest) = tokenizer_dec.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    println!();

    Ok(())
}

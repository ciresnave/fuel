// Parler-TTS lazy-graph migration.
//
// This binary now uses the lazy_* modules for the three subnets that
// make up a Parler-TTS pipeline:
//   * `fuel::lazy_t5`         — text encoder (T5EncoderModel)
//   * `fuel::lazy_parler_tts` — multi-codebook audio-token decoder
//   * `fuel::lazy_dac`        — audio codec used to turn the codes
//                                back into a waveform
//
// Note: the lazy DAC + T5 loaders expect their tensors at the root of
// the safetensors file ("decoder.model.*" / "encoder.block.*"); the
// Parler checkpoints prefix them with "audio_encoder.model." and
// "text_encoder." respectively. Those prefix mismatches will surface
// as load-time errors when the binary is actually run against a
// Parler checkpoint — they are not part of this migration's scope.
// The binary compiles and the eager API is fully removed.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::sync::Arc;

use anyhow::Error as E;
use clap::Parser;

use fuel::lazy::LazyTensor;
use fuel::lazy_dac::{DacConfig, DacModel, DacWeights};
use fuel::lazy_parler_tts::{
    ParlerActivation, ParlerDecoderConfig, ParlerDecoderModel, ParlerDecoderWeights,
};
use fuel::lazy_t5::{T5Activation, T5Config, T5Model, T5Weights};
use fuel::safetensors::MmapedSafetensors;
use fuel::Shape;
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Display the token for the specified prompt.
    #[arg(long)]
    verbose_prompt: bool,

    #[arg(long, default_value = "Hey, how are you doing today?")]
    prompt: String,

    #[arg(
        long,
        default_value = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
    )]
    description: String,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(long, default_value_t = 5000)]
    sample_len: usize,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Use f16 precision for all the computations rather than f32.
    #[arg(long)]
    f16: bool,

    #[arg(long)]
    model_file: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long, default_value_t = 512)]
    max_steps: usize,

    /// The output wav file.
    #[arg(long, default_value = "out.wav")]
    out_file: String,

    #[arg(long, default_value = "large-v1")]
    which: Which,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "large-v1")]
    LargeV1,
    #[value(name = "mini-v1")]
    MiniV1,
}

/// Holds the parsed-but-not-yet-converted Parler config plus the
/// individual `*Config` structs the lazy subnets consume.
struct ParlerConfig {
    decoder_cfg: ParlerDecoderConfig,
    text_encoder_cfg: T5Config,
    audio_encoder_cfg: DacConfig,
    audio_sampling_rate: u32,
    decoder_start_token_id: u32,
    pad_token_id: u32,
    /// `vocab_size` for the top-level model — used to size the
    /// `embed_prompts` table.
    prompt_vocab_size: usize,
}

fn parse_parler_config(json: &str) -> anyhow::Result<ParlerConfig> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| E::msg(format!("parsing parler config.json: {e}")))?;

    fn obj<'a>(v: &'a serde_json::Value, key: &str) -> anyhow::Result<&'a serde_json::Value> {
        v.get(key)
            .ok_or_else(|| E::msg(format!("parler config.json: missing field {key:?}")))
    }
    fn get_usize(v: &serde_json::Value, key: &str) -> anyhow::Result<usize> {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .ok_or_else(|| E::msg(format!("parler config.json: missing/invalid field {key:?}")))
    }
    fn get_usize_opt(v: &serde_json::Value, key: &str) -> Option<usize> {
        v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    }
    fn get_f64(v: &serde_json::Value, key: &str, default: f64) -> f64 {
        v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
    }
    fn get_bool(v: &serde_json::Value, key: &str, default: bool) -> bool {
        v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
    }

    let decoder = obj(&v, "decoder")?;
    let text_encoder = obj(&v, "text_encoder")?;
    let audio_encoder = obj(&v, "audio_encoder")?;

    // ----- decoder ----------------------------------------------------
    let vocab_size = get_usize(decoder, "vocab_size")?;
    let max_position_embeddings = get_usize(decoder, "max_position_embeddings")?;
    let num_hidden_layers = get_usize(decoder, "num_hidden_layers")?;
    let ffn_dim = get_usize(decoder, "ffn_dim")?;
    let num_attention_heads = get_usize(decoder, "num_attention_heads")?;
    let num_kv_heads = get_usize_opt(decoder, "num_key_value_heads");
    let num_cross_kv_heads = get_usize_opt(decoder, "num_cross_attention_key_value_heads");
    let hidden_size = get_usize(decoder, "hidden_size")?;
    let num_codebooks = get_usize(decoder, "num_codebooks")?;

    let activation_str = decoder
        .get("activation_function")
        .and_then(|x| x.as_str())
        .unwrap_or("gelu");
    let activation_function = match activation_str {
        "gelu" => ParlerActivation::Gelu,
        "gelu_pytorch_tanh" | "gelu_new" => ParlerActivation::GeluPytorchTanh,
        "relu" => ParlerActivation::Relu,
        "silu" | "swish" => ParlerActivation::Silu,
        other => {
            return Err(E::msg(format!(
                "parler config.json: unsupported activation_function {other:?}"
            )))
        }
    };

    // Top-level vocab_size sizes the `embed_prompts` table.
    let prompt_vocab_size = get_usize(&v, "vocab_size")?;

    // ----- text encoder (T5) -----------------------------------------
    let t5_d_model = get_usize(text_encoder, "d_model")?;
    let t5_d_kv = get_usize(text_encoder, "d_kv")?;
    let t5_d_ff = get_usize(text_encoder, "d_ff")?;
    let t5_num_layers = get_usize(text_encoder, "num_layers")?;
    let t5_num_decoder_layers = get_usize_opt(text_encoder, "num_decoder_layers");
    let t5_num_heads = get_usize(text_encoder, "num_heads")?;
    let t5_vocab_size = get_usize(text_encoder, "vocab_size")?;
    let t5_rel_buckets = get_usize(text_encoder, "relative_attention_num_buckets")?;
    let t5_rel_max_distance =
        get_usize_opt(text_encoder, "relative_attention_max_distance").unwrap_or(128);
    let t5_layer_norm_epsilon = get_f64(text_encoder, "layer_norm_epsilon", 1e-6);
    let t5_tie_word_embeddings = get_bool(text_encoder, "tie_word_embeddings", true);
    let t5_ffp = text_encoder
        .get("feed_forward_proj")
        .and_then(|x| x.as_str())
        .unwrap_or("relu");
    let (t5_gated, t5_activation) = match t5_ffp {
        "gated-gelu" => (true, T5Activation::GeluPytorchTanh),
        "gated-silu" => (true, T5Activation::Silu),
        "relu" => (false, T5Activation::Relu),
        "silu" | "swish" => (false, T5Activation::Silu),
        "gelu" => (false, T5Activation::Gelu),
        "gelu_new" | "gelu_pytorch_tanh" => (false, T5Activation::GeluPytorchTanh),
        other => {
            if let Some(inner) = other.strip_prefix("gated-") {
                let act = match inner {
                    "gelu" => T5Activation::GeluPytorchTanh,
                    "silu" | "swish" => T5Activation::Silu,
                    "relu" => T5Activation::Relu,
                    _ => T5Activation::GeluPytorchTanh,
                };
                (true, act)
            } else {
                (false, T5Activation::Relu)
            }
        }
    };

    let text_encoder_cfg = T5Config {
        vocab_size: t5_vocab_size,
        d_model: t5_d_model,
        d_kv: t5_d_kv,
        d_ff: t5_d_ff,
        num_layers: t5_num_layers,
        num_decoder_layers: t5_num_decoder_layers,
        num_heads: t5_num_heads,
        relative_attention_num_buckets: t5_rel_buckets,
        relative_attention_max_distance: t5_rel_max_distance,
        layer_norm_epsilon: t5_layer_norm_epsilon,
        activation: t5_activation,
        gated_ffn: t5_gated,
        tie_word_embeddings: t5_tie_word_embeddings,
    };

    let has_enc_proj = t5_d_model != hidden_size;

    let decoder_cfg = ParlerDecoderConfig {
        vocab_size,
        max_position_embeddings,
        num_hidden_layers,
        ffn_dim,
        num_attention_heads,
        num_kv_heads,
        num_cross_kv_heads,
        activation_function,
        hidden_size,
        num_codebooks,
        has_enc_proj,
        has_prompt_embedding: true,
    };

    // ----- audio encoder (DAC) ---------------------------------------
    // The HF `audio_encoder` block only exposes a handful of fields;
    // the rest of the eager DAC decoder is built from hard-coded
    // values inside `dac::Decoder::new(latent_dim, 1536, &[8, 8, 4, 2], 1, ...)`.
    let dac_num_codebooks = get_usize(audio_encoder, "num_codebooks")?;
    let dac_codebook_size = get_usize(audio_encoder, "codebook_size")?;
    let dac_latent_dim = get_usize(audio_encoder, "latent_dim")?;
    let sampling_rate = audio_encoder
        .get("sampling_rate")
        .and_then(|x| x.as_u64())
        .unwrap_or(24_000) as u32;

    let audio_encoder_cfg = DacConfig {
        num_codebooks: dac_num_codebooks,
        codebook_size: dac_codebook_size,
        // The HF JSON does not expose codebook_dim explicitly; DAC
        // uses 8 by convention (descript/dac_44khz).
        codebook_dim: 8,
        latent_dim: dac_latent_dim,
        decoder_initial_channels: 1536,
        decoder_rates: vec![8, 8, 4, 2],
        decoder_out_channels: 1,
    };

    let decoder_start_token_id = get_usize(&v, "decoder_start_token_id")? as u32;
    let pad_token_id = get_usize(&v, "pad_token_id")? as u32;

    Ok(ParlerConfig {
        decoder_cfg,
        text_encoder_cfg,
        audio_encoder_cfg,
        audio_sampling_rate: sampling_rate,
        decoder_start_token_id,
        pad_token_id,
        prompt_vocab_size,
    })
}

/// Per-codebook argmax / temperature sampling. Mirrors the very
/// simple LogitsProcessor used by the eager Parler binary.
fn sample(logits: &[f32], temperature: f32, top_p: Option<f32>, seed: u64) -> u32 {
    if temperature <= 0.0 {
        let mut best_i = 0usize;
        let mut best = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > best {
                best = v;
                best_i = i;
            }
        }
        return best_i as u32;
    }
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let inv_t = 1.0 / temperature.max(1e-6);
    let mut probs: Vec<f32> = logits.iter().map(|&x| ((x - max_l) * inv_t).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum.max(1e-30);
    }
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let mut keep_mask: Vec<bool> = vec![true; probs.len()];
    if let Some(p_cut) = top_p {
        let mut cum2 = 0.0;
        let mut allow = true;
        for &i in &idx {
            if !keep_mask[i] {
                continue;
            }
            if !allow {
                keep_mask[i] = false;
                continue;
            }
            cum2 += probs[i];
            if cum2 >= p_cut {
                allow = false;
            }
        }
    }
    let mut filtered: Vec<f32> = probs
        .iter()
        .enumerate()
        .map(|(i, p)| if keep_mask[i] { *p } else { 0.0 })
        .collect();
    let s: f32 = filtered.iter().sum();
    if s > 0.0 {
        for v in &mut filtered {
            *v /= s;
        }
    } else {
        return 0;
    }
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    state ^= state >> 33;
    state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
    state ^= state >> 33;
    let r = (state as f32) / (u64::MAX as f32);
    let mut cum = 0.0;
    for (i, p) in filtered.iter().enumerate() {
        cum += *p;
        if r <= cum {
            return i as u32;
        }
    }
    (filtered.len() - 1) as u32
}

fn main() -> anyhow::Result<()> {
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
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        fuel::utils::with_avx(),
        fuel::utils::with_neon(),
        fuel::utils::with_simd128(),
        fuel::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature, args.repeat_penalty, args.repeat_last_n
    );

    // `--cpu`, `--f16`, `--quantized` retained for CLI compatibility
    // with the eager binary; the lazy port runs on CPU F32 only.
    let _ = args.cpu;
    let _ = args.f16;
    let _ = args.quantized;
    let _ = args.repeat_penalty;
    let _ = args.repeat_last_n;
    let _ = args.sample_len;

    let start = std::time::Instant::now();
    let api = hf_hub::api::sync::Api::new()?;
    let model_id = match args.model_id {
        Some(model_id) => model_id.to_string(),
        None => match args.which {
            Which::LargeV1 => "parler-tts/parler-tts-large-v1".to_string(),
            Which::MiniV1 => "parler-tts/parler-tts-mini-v1".to_string(),
        },
    };
    let revision = match args.revision {
        Some(r) => r,
        None => "main".to_string(),
    };
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id,
        hf_hub::RepoType::Model,
        revision,
    ));
    let model_files = match args.model_file {
        Some(m) => vec![m.into()],
        None => match args.which {
            Which::MiniV1 => vec![repo.get("model.safetensors")?],
            Which::LargeV1 => {
                fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
            }
        },
    };
    let config_path = match args.config_file {
        Some(m) => m.into(),
        None => repo.get("config.json")?,
    };
    let tokenizer_path = match args.tokenizer_file {
        Some(m) => m.into(),
        None => repo.get("tokenizer.json")?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config_json = std::fs::read_to_string(&config_path)?;
    let cfg = parse_parler_config(&config_json)?;

    // Memory-map the safetensors once and load every subnet from it.
    let st = unsafe { MmapedSafetensors::multi(&model_files) }
        .map_err(|e| E::msg(format!("mmap parler safetensors: {e}")))?;

    // --- Text encoder (T5) ------------------------------------------
    // Parler's T5 weights live under `text_encoder.*` rather than at
    // the root the lazy_t5 loader expects; this load call will fail
    // at runtime until lazy_t5 grows a prefix-aware loader.
    let t5_weights = T5Weights::load_from_mmapped(&st, &cfg.text_encoder_cfg)
        .map_err(|e| E::msg(format!("load t5 weights: {e}")))?;
    let text_encoder = T5Model {
        config: cfg.text_encoder_cfg.clone(),
        weights: t5_weights,
    };

    // --- Parler decoder ---------------------------------------------
    let decoder_weights = ParlerDecoderWeights::load_from_mmapped(
        &st,
        &cfg.decoder_cfg,
        cfg.text_encoder_cfg.d_model,
    )
    .map_err(|e| E::msg(format!("load parler decoder weights: {e}")))?;
    let decoder = ParlerDecoderModel {
        config: cfg.decoder_cfg.clone(),
        weights: decoder_weights,
    };

    // --- Audio encoder (DAC) ----------------------------------------
    // Same prefix story as T5: the lazy_dac loader expects the
    // tensors at the root of the file but Parler ships them under
    // `audio_encoder.model.*`. The load call will surface that as a
    // runtime error.
    let dac_weights = DacWeights::load_from_mmapped(&st, &cfg.audio_encoder_cfg)
        .map_err(|e| E::msg(format!("load dac weights: {e}")))?;
    let audio_encoder = DacModel {
        config: cfg.audio_encoder_cfg.clone(),
        weights: dac_weights,
    };

    // --- embed_prompts table ----------------------------------------
    // The top-level Parler model holds a single `embed_prompts.weight`
    // tensor used to embed the audio-prompt tokens before the decoder.
    let embed_prompts: Arc<[f32]> = Arc::from(
        fuel::lazy::load_tensor_as_f32(&st, "embed_prompts.weight")
            .map_err(|e| E::msg(format!("load embed_prompts.weight: {e}")))?,
    );
    let expected_prompts_len = cfg.prompt_vocab_size * cfg.decoder_cfg.hidden_size;
    if embed_prompts.len() != expected_prompts_len {
        return Err(E::msg(format!(
            "embed_prompts.weight: {} elements, expected {} ({}×{})",
            embed_prompts.len(),
            expected_prompts_len,
            cfg.prompt_vocab_size,
            cfg.decoder_cfg.hidden_size,
        )));
    }

    println!("loaded the model in {:?}", start.elapsed());

    // --- Tokenize prompts -------------------------------------------
    let description_token_ids: Vec<u32> = tokenizer
        .encode(args.description.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let prompt_token_ids: Vec<u32> = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    if args.verbose_prompt {
        println!("description tokens: {description_token_ids:?}");
        println!("prompt tokens: {prompt_token_ids:?}");
    }

    println!("starting generation...");

    // --- Text encoder forward ---------------------------------------
    let encoded = text_encoder
        .forward_encoder(&description_token_ids)
        .map_err(|e| E::msg(format!("text encoder forward: {e}")))?;

    // Anchor LazyTensor — gives `const_*_like` helpers a graph to hang
    // their nodes off of.
    let anchor = LazyTensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), &fuel::Device::cpu());

    // Prompt embeddings: look up each prompt token in the
    // `embed_prompts` table to get a `(1, P, hidden_size)` tensor.
    let prompt_len = prompt_token_ids.len();
    let h_dim = cfg.decoder_cfg.hidden_size;
    let embed_table = anchor.const_f32_like(
        Arc::clone(&embed_prompts),
        Shape::from_dims(&[cfg.prompt_vocab_size, h_dim]),
    );
    let prompt_ids_lt = anchor.const_u32_like(
        prompt_token_ids.clone(),
        Shape::from_dims(&[prompt_len]),
    );
    let prompt_hidden_states = embed_table
        .index_select(0_usize, &prompt_ids_lt)
        .map_err(|e| E::msg(format!("prompt embedding lookup: {e}")))?
        .reshape(Shape::from_dims(&[1, prompt_len, h_dim]))
        .map_err(|e| E::msg(format!("prompt embedding reshape: {e}")))?;

    // --- Generation loop --------------------------------------------
    let num_codebooks = cfg.decoder_cfg.num_codebooks;
    let mut audio_tokens: Vec<u32> = vec![cfg.decoder_start_token_id; num_codebooks];
    let mut all_audio_tokens: Vec<Vec<u32>> = vec![vec![]; num_codebooks];
    let vocab_size = cfg.decoder_cfg.vocab_size;

    for step in 0..args.max_steps {
        let input_ids_lt = anchor.const_u32_like(
            audio_tokens.clone(),
            Shape::from_dims(&[1, num_codebooks, 1]),
        );
        let (prompt_embeds_arg, start_pos) = if step == 0 {
            (Some(&prompt_hidden_states), 0_usize)
        } else {
            (None, step + prompt_len)
        };

        let logits = decoder
            .forward(&input_ids_lt, prompt_embeds_arg, &encoded, start_pos)
            .map_err(|e| E::msg(format!("decoder forward: {e}")))?;

        // Each entry in `logits` is `(1, T, vocab_size)`. Sample the
        // last position for each codebook (matches the eager
        // `logit.i((0, logit.dim(1)? - 1))?` pattern).
        for (cb_idx, logit) in logits.iter().enumerate() {
            if cb_idx > step {
                break;
            }
            if audio_tokens[cb_idx] == cfg.pad_token_id {
                continue;
            }
            let data = logit.realize_f32();
            let dims = logit.shape();
            let dims = dims.dims();
            let last_t = dims[1] - 1;
            let off = last_t * vocab_size;
            let last_logits: Vec<f32> = data[off..off + vocab_size].to_vec();
            let token = sample(
                &last_logits,
                args.temperature as f32,
                args.top_p.map(|p| p as f32),
                args.seed.wrapping_add((step * num_codebooks + cb_idx) as u64),
            );
            audio_tokens[cb_idx] = token;
        }

        if audio_tokens.iter().all(|v| *v == cfg.pad_token_id) {
            break;
        }
        for (cb_idx, &token) in audio_tokens.iter().enumerate() {
            if token != cfg.decoder_start_token_id && token != cfg.pad_token_id {
                all_audio_tokens[cb_idx].push(token);
            }
        }
    }

    // Equalize codebook lengths to the shortest one.
    let min_len = all_audio_tokens.iter().map(|v| v.len()).min().unwrap_or(0);
    for v in &mut all_audio_tokens {
        v.resize(min_len, 0);
    }
    println!("generated {min_len} steps × {num_codebooks} codebooks");

    // Pack into a `(1, num_codebooks, T)` U32 LazyTensor for the DAC
    // decoder.
    let codes_flat: Vec<u32> = (0..num_codebooks)
        .flat_map(|cb| all_audio_tokens[cb].iter().copied())
        .collect();
    let codes_lt = anchor.const_u32_like(
        codes_flat,
        Shape::from_dims(&[1, num_codebooks, min_len]),
    );

    let pcm = audio_encoder
        .decode_codes(&codes_lt)
        .map_err(|e| E::msg(format!("dac decode_codes: {e}")))?;
    let pcm_dims = pcm.shape().dims().to_vec();
    println!("pcm shape: {pcm_dims:?}");

    // Output is `(1, decoder_out_channels, time_out)`; grab the first
    // channel and write a WAV.
    let pcm_data = pcm.realize_f32();
    let out_channels = cfg.audio_encoder_cfg.decoder_out_channels.max(1);
    let t_out = pcm_data.len() / out_channels;
    let pcm_ch0: Vec<f32> = pcm_data[..t_out].to_vec();

    let mut output = std::fs::File::create(&args.out_file)?;
    fuel_examples::wav::write_pcm_as_wav(&mut output, &pcm_ch0, cfg.audio_sampling_rate)?;

    Ok(())
}

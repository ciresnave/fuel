//! Voxtral — lazy port.
//!
//! The eager binary ran a chunked audio-conditioned generation loop
//! using `fuel_transformers::models::voxtral`. The lazy port at
//! `fuel::lazy_voxtral` ships forward-only, single-chunk semantics:
//!
//!   * `VoxtralModel::forward_with_audio` takes one `[1, n_mel, mel_time]`
//!     mel spectrogram (mel_time must be even) and a token sequence with
//!     `audio_token_id` placeholders, and returns logits
//!     `(1, seq, vocab_size)`.
//!   * No KV cache: every greedy-decode step re-runs the full forward.
//!   * Multi-chunk audio (audio > 30 s) is **not** yet wired through
//!     the lazy port; the binary truncates to 30 s and warns.
//!
//! What this binary does today:
//!   1. Loads the HF Voxtral config + Tekken tokenizer + safetensors.
//!   2. Pre-processes audio to a single 30 s log-mel spectrogram
//!      via `fuel::lazy_whisper_audio::pcm_to_mel` (trimmed to an
//!      even `mel_time` = 3000 frames).
//!   3. Constructs the prompt with the exact HF token pattern
//!      `<s>[INST][BEGIN_AUDIO][AUDIO]×N[/INST]lang:en[TRANSCRIBE]`
//!      where N is `mel_time/2 * audio_hidden / intermediate_size`
//!      (= 375 for Voxtral 3B).
//!   4. Greedy-decodes one token per step, calling
//!      `forward_with_audio` each step.

use anyhow::{Context, Error as E, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use hf_hub::api::sync::Api;
use std::io::Cursor;
use std::path::PathBuf;
use tekken::Tekkenizer;

use fuel::lazy_voxtral::{
    VoxtralConfig, VoxtralEncoderConfig, VoxtralModel, VoxtralTextConfig, VoxtralWeights,
};
use fuel::lazy_whisper_audio;
use fuel::safetensors::MmapedSafetensors;

mod download;

const SAMPLE_RATE: u32 = 16000;
/// Voxtral encoder takes 30 s of audio per chunk; the lazy encoder
/// expects an even `mel_time`. The whisper-style `pcm_to_mel` returns
/// 3001 frames (= N_SAMPLES / HOP_LENGTH + 1); trim to 3000.
const MEL_FRAMES_PER_CHUNK: usize = 3000;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU. Lazy realizes through the
    /// default router today; this flag is preserved for CLI parity.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// The input to be processed, in wav format, will default to `jfk.wav`. Alternatively
    /// this can be set to sample:jfk, sample:gb1, ... to fetch a sample from the following
    /// repo: https://huggingface.co/datasets/Narsil/fuel_demo/
    #[arg(long)]
    input: Option<String>,

    #[arg(long, default_value = "mistralai/Voxtral-Mini-3B-2507")]
    model_id: Option<String>,

    /// Maximum number of new tokens to decode after the prompt.
    #[arg(long, default_value_t = 1000)]
    max_new_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Lazy realizes through CPU/router; the `cpu` flag is preserved
    // for CLI parity with the eager binary but has no effect today.
    let _ = args.cpu;

    let model_id = args.model_id.unwrap();

    println!("Loading Voxtral model");
    let (model_files, tokenizer_file) = download::model_files(&model_id)?;
    let config_path = model_files.0;
    let safetensor_paths = model_files.1;

    let cfg = load_voxtral_config(&config_path)?;
    let tokenizer = Tekkenizer::from_file(tokenizer_file).map_err(E::msg)?;

    let st = unsafe { MmapedSafetensors::multi(&safetensor_paths) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = VoxtralWeights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("load voxtral weights: {e}")))?;
    let model = VoxtralModel {
        config: cfg.clone(),
        weights,
    };
    println!("Model loaded successfully");

    let api = Api::new()?;
    let dataset = api.dataset("Narsil/fuel-examples".to_string());
    let audio_file = if let Some(input) = args.input {
        if let Some(sample) = input.strip_prefix("sample:") {
            dataset.get(&format!("samples_{sample}.wav"))?
        } else {
            std::path::PathBuf::from(input)
        }
    } else {
        println!("No audio file submitted: Downloading https://huggingface.co/datasets/Narsil/fuel_demo/blob/main/samples_jfk.wav");
        dataset.get("samples_jfk.wav")?
    };

    let (audio_data, sample_rate) =
        fuel_examples::audio::pcm_decode(audio_file).context("Failed to decode audio file")?;

    let result = transcribe(&model, &tokenizer, &audio_data, sample_rate, args.max_new_tokens)?;

    println!("\n===================================================\n");
    println!("{}", result);

    Ok(())
}

/// Greedy-decode a single 30-second chunk of audio.
fn transcribe(
    model: &VoxtralModel,
    tokenizer: &Tekkenizer,
    audio_data: &[f32],
    sample_rate: u32,
    max_new_tokens: usize,
) -> Result<String> {
    // 1) Resample to 16 kHz if needed.
    let audio: Vec<f32> = if sample_rate == SAMPLE_RATE {
        audio_data.to_vec()
    } else {
        fuel_examples::audio::resample(audio_data, sample_rate, SAMPLE_RATE)
            .context("Failed to resample audio")?
    };

    // 2) Truncate / pad to a single 30-second chunk. The lazy port
    //    currently supports single-chunk audio only; multi-chunk
    //    would require either looping forward_with_audio with
    //    different mel slices or extending the lazy projector to
    //    accept batched audio.
    let chunk_samples: usize = 30 * SAMPLE_RATE as usize;
    if audio.len() > chunk_samples {
        println!(
            "Warning: audio is {} samples (> 30 s); lazy port truncates to 30 s.",
            audio.len()
        );
    }
    let audio_clipped: Vec<f32> = audio.iter().take(chunk_samples).copied().collect();

    // 3) Load the 128-bin mel filterbank.
    let mel_bytes = include_bytes!("melfilters128.bytes");
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    let mut cursor = Cursor::new(&mel_bytes[..]);
    cursor.read_f32_into::<LittleEndian>(&mut mel_filters)?;

    // 4) Compute log-mel spectrogram via the host-side whisper pipeline.
    //    pcm_to_mel pads / trims to N_SAMPLES (= 480 000) and returns a
    //    flat `(n_mels, 3001)` row-major vec. We need an even mel_time,
    //    so trim to 3000 columns.
    let n_mel = cfg_n_mels(&model.config);
    let mel_full = lazy_whisper_audio::pcm_to_mel(&audio_clipped, &mel_filters, n_mel)
        .map_err(|e| E::msg(format!("pcm_to_mel: {e}")))?;
    let mel_total_frames = mel_full.len() / n_mel;
    let mel_time = MEL_FRAMES_PER_CHUNK.min(mel_total_frames);
    let mel_time = mel_time - (mel_time % 2);
    if mel_time == 0 {
        anyhow::bail!("transcribe: empty mel after trimming to even length");
    }
    // Build a `(n_mel, mel_time)` flat row-major vec by slicing the
    // first `mel_time` columns out of `(n_mel, mel_total_frames)`.
    let mut mel: Vec<f32> = Vec::with_capacity(n_mel * mel_time);
    for m in 0..n_mel {
        let row = &mel_full[m * mel_total_frames..(m + 1) * mel_total_frames];
        mel.extend_from_slice(&row[..mel_time]);
    }

    // 5) Compute the number of audio embedding slots produced by the
    //    encoder + projector. Encoder halves mel_time (stride-2 conv).
    //    Projector reshapes `(1, mel_time/2, audio_hidden)` into
    //    `(num_audio_tokens, audio_intermediate_size)` where
    //    `num_audio_tokens = (mel_time/2 * audio_hidden) /
    //                          audio_intermediate_size`.
    let acfg = &model.config.audio;
    let a_hidden = acfg.hidden_size;
    let intermediate = acfg.intermediate_size;
    let after_conv = mel_time / 2;
    let total_audio_features = after_conv * a_hidden;
    if !total_audio_features.is_multiple_of(intermediate) {
        anyhow::bail!(
            "Voxtral projector reshape: total {total_audio_features} not divisible by intermediate_size {intermediate}",
        );
    }
    let num_audio_tokens = total_audio_features / intermediate;

    // 6) Build the prompt token sequence. The lazy `forward_with_audio`
    //    will substitute projected audio embeddings at every
    //    `audio_token_id` position.
    //
    //    Pattern (matches the HF Voxtral processor):
    //    `<s>[INST][BEGIN_AUDIO][AUDIO]×N[/INST]lang:en[TRANSCRIBE]`
    //    where N = `num_audio_tokens`.
    let audio_token_id = model.config.audio_token_id;
    let mut tokens: Vec<u32> = Vec::with_capacity(num_audio_tokens + 8);
    tokens.push(1u32); // BOS: <s>
    tokens.push(3u32); // [INST]
    tokens.push(25u32); // [BEGIN_AUDIO]
    for _ in 0..num_audio_tokens {
        tokens.push(audio_token_id);
    }
    tokens.push(4u32); // [/INST]
    tokens.push(9909u32); // lang
    tokens.push(1058u32); // :
    tokens.push(1262u32); // en
    tokens.push(34u32); // [TRANSCRIBE]
    let prompt_len = tokens.len();

    // 7) Greedy generation loop. No KV cache yet; every step re-runs
    //    the full audio encoder + decoder forward.
    //
    //    EOS tokens follow the eager binary: 2, 128001, 128009, 128256.
    let vocab_size = model.config.text.vocab_size;
    let eos_set = [2u32, 128001, 128009, 128256];
    for _ in 0..max_new_tokens {
        let logits = model
            .forward_with_audio(&mel, mel_time, &tokens, 0)
            .map_err(|e| E::msg(format!("forward_with_audio: {e}")))?;
        let data = logits.realize_f32();
        let seq = tokens.len();
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
        let next_token = best_i as u32;
        if eos_set.contains(&next_token) {
            break;
        }
        tokens.push(next_token);
    }

    // 8) Decode only the newly generated tail.
    let new_tokens = if tokens.len() > prompt_len {
        &tokens[prompt_len..]
    } else {
        &tokens[..]
    };
    let decoded = tokenizer
        .decode(new_tokens, tekken::SpecialTokenPolicy::Ignore)
        .map_err(|e| E::msg(format!("Failed to decode tokens: {e}")))?;
    Ok(decoded)
}

fn cfg_n_mels(cfg: &VoxtralConfig) -> usize {
    cfg.audio.num_mel_bins
}

/// Load Voxtral configuration from the HF `config.json` and build a
/// lazy `VoxtralConfig`.
fn load_voxtral_config(config_file: &PathBuf) -> Result<VoxtralConfig> {
    let config_str = std::fs::read_to_string(config_file)?;
    let json: serde_json::Value =
        serde_json::from_str(&config_str).context("Failed to parse config.json")?;

    let audio_token_id = json
        .get("audio_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(24) as u32;

    let audio_json = json
        .get("audio_config")
        .ok_or_else(|| E::msg("Missing audio_config in configuration"))?;
    let audio = VoxtralEncoderConfig {
        hidden_size: audio_json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(1280) as usize,
        intermediate_size: audio_json
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(5120) as usize,
        num_hidden_layers: audio_json
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize,
        num_attention_heads: audio_json
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(20) as usize,
        num_mel_bins: audio_json
            .get("num_mel_bins")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize,
        max_source_positions: audio_json
            .get("max_source_positions")
            .and_then(|v| v.as_u64())
            .unwrap_or(1500) as usize,
    };

    let text_json = json
        .get("text_config")
        .ok_or_else(|| E::msg("Missing text_config in configuration"))?;
    let text = VoxtralTextConfig {
        vocab_size: text_json
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(131072) as usize,
        hidden_size: text_json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(3072) as usize,
        intermediate_size: text_json
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(8192) as usize,
        num_hidden_layers: text_json
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(30) as usize,
        num_attention_heads: text_json
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize,
        num_key_value_heads: text_json
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize,
        head_dim: text_json
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize,
        rms_norm_eps: text_json
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5),
        rope_theta: text_json
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(100_000_000.0),
        max_position_embeddings: text_json
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(131072) as usize,
        tie_word_embeddings: text_json
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    };

    Ok(VoxtralConfig {
        audio,
        text,
        audio_token_id,
    })
}


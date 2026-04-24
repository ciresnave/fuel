// End-to-end Whisper encoder-decoder runner. Phase 6a anchor #4.
//
// USAGE
//
//     cargo run --release --bin whisper-lazy
//     cargo run --release --bin whisper-lazy -- [MODEL_ID] [MAX_NEW_TOKENS]
//
// Defaults:
//     MODEL_ID       = openai/whisper-tiny
//     MAX_NEW_TOKENS = 16
//
// SCOPE
//
// This binary exercises Whisper's encoder + decoder + greedy decode
// on a ZERO mel spectrogram. It proves the architecture wiring end-
// to-end (conv stem → positional → 4 encoder layers → cross-attn into
// 4 decoder layers → tied output projection), checks the weights load
// correctly from the HF safetensors, and prints the first decoded
// tokens.
//
// Audio preprocessing (waveform → 80×3000 log-mel spectrogram via
// STFT + mel filterbank + log-scale) is deferred — that's a pure
// audio-DSP pipeline that lives outside the lazy graph. When it
// lands, replace the `vec![0.0; 80*3000]` below with the real mel
// features and the same generate_greedy path produces real
// transcriptions.

use fuel::lazy_whisper::{WhisperModel, WhisperTokenizer};
use std::io::Write;
use std::time::Instant;

const DEFAULT_MODEL: &str = "openai/whisper-tiny";
const DEFAULT_MAX_NEW: usize = 16;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_id = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let max_new: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_NEW);

    eprintln!("=== fuel whisper-lazy ===");
    eprintln!("Model:  {model_id}");
    eprintln!("Max new tokens: {max_new}");
    eprintln!();

    eprint!("Downloading + loading model weights... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let model = WhisperModel::from_hub(&model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  encoder: layers={}  heads={}  ffn={}  d_model={}",
        model.config.encoder_layers,
        model.config.encoder_attention_heads,
        model.config.encoder_ffn_dim,
        model.config.d_model,
    );
    eprintln!(
        "  decoder: layers={}  heads={}  ffn={}  vocab={}",
        model.config.decoder_layers,
        model.config.decoder_attention_heads,
        model.config.decoder_ffn_dim,
        model.config.vocab_size,
    );
    eprintln!(
        "  mel_bins={}  max_src={}  max_tgt={}",
        model.config.num_mel_bins,
        model.config.max_source_positions,
        model.config.max_target_positions,
    );

    eprint!("Loading tokenizer...                    ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let tokenizer = WhisperTokenizer::from_hub(&model_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    let sot = tokenizer.token_to_id("<|startoftranscript|>");
    let lang_en = tokenizer.token_to_id("<|en|>");
    let transcribe = tokenizer.token_to_id("<|transcribe|>");
    let notimestamps = tokenizer.token_to_id("<|notimestamps|>");
    eprintln!(
        "  <|startoftranscript|>={sot:?}  <|en|>={lang_en:?}  <|transcribe|>={transcribe:?}  <|notimestamps|>={notimestamps:?}",
    );
    eprintln!();

    // Build the English-transcribe decoder prompt. Fall back to just
    // the decoder_start_token_id if any special token is missing.
    let mut prompt: Vec<u32> = Vec::new();
    if let (Some(s), Some(l), Some(t), Some(n)) = (sot, lang_en, transcribe, notimestamps) {
        prompt.extend([s, l, t, n]);
    } else {
        prompt.push(model.config.decoder_start_token_id);
    }
    eprintln!("Decoder prompt: {prompt:?}");

    // Placeholder mel: [1, 80, 3000] zero spectrogram. With zero input
    // and real weights, argmax of the first-step logits is meaningful
    // (it reflects each layer's LN bias + token-embedding geometry)
    // but the text is not — hence the "zero-mel fingerprint" framing.
    let mel_bins = model.config.num_mel_bins;
    let mel_time = 3000;
    let mel = vec![0.0_f32; mel_bins * mel_time];
    eprintln!("Mel input: zero [{mel_bins}, {mel_time}] (placeholder)");
    eprintln!();

    eprintln!("Running encoder + greedy decode...");
    let t0 = Instant::now();
    let tokens = model.generate_greedy(&mel, &prompt, max_new)?;
    let elapsed = t0.elapsed();
    let new_tokens = tokens.len() - prompt.len();
    eprintln!(
        "Generated {new_tokens} tokens in {:.2?}  ({:.2?}/token avg)",
        elapsed,
        elapsed / new_tokens.max(1) as u32,
    );
    eprintln!();
    eprintln!("Tokens: {tokens:?}");
    eprintln!();
    let text = tokenizer.decode(&tokens, false)?;
    eprintln!("Decoded (keep special): {text}");
    let text_clean = tokenizer.decode(&tokens, true)?;
    eprintln!("Decoded (skip special): {text_clean}");

    Ok(())
}

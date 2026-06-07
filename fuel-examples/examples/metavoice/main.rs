// MetaVoice — lazy-port revival.
//
// This binary is a fresh write that wires the newly ported
// `fuel::lazy_metavoice` (stage-2 multi-codebook transformer) to the
// newly ported `fuel::lazy_encodec` decoder. It deliberately diverges
// from the retired eager binary in a few places that the lazy port does
// not (yet) cover. Each gap is called out inline near the code that
// stands in for it so the next session has a clear picture of what is
// load-bearing for actual TTS quality vs. what is here only to make the
// binary build and run end-to-end.
//
// Gaps (relative to the eager
// `fuel-transformers/src/_models_retired/audio/metavoice.rs` binary):
//
//   * **Stage-1 multi-codebook GPT** — eager `gpt::Model` is not lazy-
//     ported. The eager pipeline first runs a stage-1 GPT (text +
//     speaker → 2-codebook stream + adapter-extracted text ids), then
//     feeds the result into the stage-2 transformer + tilted-encodec
//     adapter to expand to 6 codebooks. The lazy port only has the
//     stage-2 transformer + the EnCodec decoder, so this binary drives
//     the stage-2 transformer directly on the text-token prompt and
//     skips the stage-1 hierarchy entirely. The output is therefore
//     `lazy_metavoice.num_codebooks` codes per position (default 4),
//     fed straight into encodec. Expected quality is well below the
//     eager pipeline — see README. Wiring stage-1 in is gated on the
//     lazy port of `metavoice::gpt::Model`.
//
//   * **BPE tokenizer** — the eager binary loads
//     `first_stage.meta.json` (a tiktoken-format BPE) via
//     `tokenizers::BPE`, which lives in the retired
//     `fuel-transformers/src/_models_retired/audio/metavoice.rs`. The
//     lazy `lazy_metavoice` ships no tokenizer. We fall back to a
//     byte-level encoding (each prompt byte clamped to
//     `cfg.vocab_size`). This is enough for the model to *run* but the
//     resulting tokens are not meaningful — re-porting `tokenizers::BPE`
//     to lazy-land is a follow-up.
//
//   * **Speaker embedding** — the eager binary downloads a pre-baked
//     `spk_emb.safetensors` from the HF repo. `--speaker-encoder` here
//     accepts either a safetensors file containing a `spk_emb` tensor
//     (matches the eager convention) **or** the path-prefix of a
//     lazy_metavoice_speaker_encoder checkpoint. When neither is
//     supplied (or load fails), we fall back to a zero speaker vector
//     — which still drives the model forward, just without any voice
//     conditioning.
//
//   * **Loudness normalization** — `fuel_examples::audio::normalize_loudness`
//     operates on the eager `Tensor` API. We mirror the host-side
//     implementation that the lazy encodec example uses (BS.1770 +
//     tanh compressor on a plain `&[f32]`) so the lazy path doesn't
//     have to round-trip through an eager Tensor.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::io::Write;

use fuel::lazy::LazyTensor;
use fuel::lazy_encodec::{EncodecConfig, EncodecModel, EncodecWeights};
use fuel::lazy_metavoice::{MetaVoiceConfig, MetaVoiceModel, MetaVoiceWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Text to synthesize.
    #[arg(long, default_value = "This is a demo of text to speech by MetaVoice-1B.")]
    prompt: String,

    /// Path to the speaker encoder safetensors. Accepts either:
    ///   - a file containing a top-level `spk_emb` tensor (eager
    ///     convention), or
    ///   - a `lazy_metavoice_speaker_encoder`-loadable checkpoint
    ///     (used only to size the speaker vector — we don't run the
    ///     full mel→LSTM pipeline because the lazy port doesn't ship
    ///     the upstream mel-spectrogram extraction).
    ///
    /// If missing or load fails, falls back to a zero vector.
    #[arg(long)]
    speaker_encoder: Option<String>,

    /// Path to the EnCodec decoder weights (the "first stage" of audio
    /// synthesis — codes → waveform). Defaults to the
    /// `facebook/encodec_24khz/model.safetensors` checkpoint.
    ///
    /// Note: in the retired eager binary `--first-stage-weights` named
    /// the stage-1 GPT model. Because the lazy port does not include
    /// the stage-1 GPT, we re-purpose this flag to point at the
    /// EnCodec weights — which **are** the first stage of audio
    /// synthesis from a coded representation. See the file-level
    /// comment for the broader gap.
    #[arg(long)]
    first_stage: Option<String>,

    /// Path to the MetaVoice stage-2 transformer safetensors. This is
    /// the model that `fuel::lazy_metavoice::MetaVoiceModel` wraps.
    #[arg(long)]
    second_stage: Option<String>,

    /// Output wav file path.
    #[arg(long, default_value = "out.wav")]
    output_wav: String,

    /// The maximum number of decoder steps to generate (each step
    /// produces `num_codebooks` codes — one per EnCodec codebook).
    #[arg(long, default_value_t = 256)]
    max_steps: usize,

    /// EnCodec stop-token id. The eager binary uses 2048 as the
    /// stage-1 GPT EOA marker; here we use it as an early-stop
    /// sentinel on the codebook-0 stream.
    #[arg(long, default_value_t = 2048)]
    stop_token: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 1) Resolve weight paths. The HF Hub fallbacks mirror the eager
    //    binary's default repos so a no-flag invocation still pulls
    //    real weights down.
    let api = hf_hub::api::sync::Api::new()?;

    let second_stage_path = match &args.second_stage {
        Some(p) => std::path::PathBuf::from(p),
        None => api
            .model("lmz/fuel-metavoice".to_string())
            .get("second_stage.safetensors")?,
    };
    let first_stage_path = match &args.first_stage {
        Some(p) => std::path::PathBuf::from(p),
        None => api
            .model("facebook/encodec_24khz".to_string())
            .get("model.safetensors")?,
    };

    // 2) Load the stage-2 MetaVoice transformer.
    println!("loading metavoice stage-2 weights from {second_stage_path:?}");
    let mv_st = unsafe { MmapedSafetensors::multi(&[&second_stage_path]) }
        .map_err(|e| E::msg(format!("mmap metavoice safetensors: {e}")))?;
    let mv_cfg = MetaVoiceConfig::metavoice_1b_v0_1();
    let mv_weights = MetaVoiceWeights::load_from_mmapped(&mv_st, &mv_cfg)
        .map_err(|e| E::msg(format!("load metavoice weights: {e}")))?;
    let mv_model = MetaVoiceModel {
        config: mv_cfg.clone(),
        weights: mv_weights,
    };

    // 3) Load the EnCodec decoder.
    println!("loading encodec weights from {first_stage_path:?}");
    let enc_st = unsafe { MmapedSafetensors::multi(&[&first_stage_path]) }
        .map_err(|e| E::msg(format!("mmap encodec safetensors: {e}")))?;
    let enc_cfg = EncodecConfig::default_preset();
    let enc_weights = EncodecWeights::load_from_mmapped(
        &enc_st,
        &enc_cfg,
        enc_cfg.sampling_rate,
        &enc_cfg.target_bandwidths,
    )
    .map_err(|e| E::msg(format!("load encodec weights: {e}")))?;
    let enc_model = EncodecModel {
        config: enc_cfg.clone(),
        weights: enc_weights,
    };

    // 4) Build a speaker embedding LazyTensor on CPU.
    let speaker_embed = load_speaker_embed(&args, &mv_cfg)?;

    // 5) Encode the prompt. **Gap:** the eager BPE tokenizer is not
    //    lazy-ported. We use a byte-level fallback that bounds every
    //    token to `vocab_size`. This is enough to exercise the forward
    //    path; meaningful TTS would require the real BPE.
    println!("prompt: '{}'", args.prompt);
    let prompt_tokens = byte_level_encode(&args.prompt, mv_cfg.vocab_size);
    println!(
        "encoded prompt into {} tokens (byte-level fallback)",
        prompt_tokens.len()
    );
    if prompt_tokens.is_empty() {
        anyhow::bail!("empty prompt produced no tokens — pass --prompt");
    }

    // 6) Autoregressive generation. Each step yields per-codebook
    //    logits at the final position; we greedy-pick one id per
    //    codebook and append codebook-0 to the rolling token sequence
    //    (which keeps the LM grounded). The full multi-codebook tuple
    //    at every step is appended to the codes buffer for encodec.
    let num_codebooks = mv_cfg.num_codebooks;
    let mut tokens = prompt_tokens.clone();
    // codes_per_cb[i] = sequence of ids generated for codebook i.
    let mut codes_per_cb: Vec<Vec<u32>> = (0..num_codebooks).map(|_| Vec::new()).collect();

    println!(
        "starting generation: max_steps={}, num_codebooks={}",
        args.max_steps, num_codebooks
    );
    let started = std::time::Instant::now();
    for step in 0..args.max_steps {
        let logits = mv_model
            .forward(&tokens, &speaker_embed, 0)
            .map_err(|e| E::msg(format!("metavoice forward step {step}: {e}")))?;
        // logits shape: (1, num_codebooks, vocab_size)
        let dims = logits.shape().dims().to_vec();
        if dims.len() != 3 || dims[0] != 1 || dims[1] != num_codebooks {
            anyhow::bail!(
                "unexpected metavoice logits shape {:?}, expected [1, {}, vocab]",
                dims,
                num_codebooks
            );
        }
        let vocab = dims[2];
        let flat = logits.realize_f32();

        // Greedy argmax per codebook.
        let mut next_per_cb: Vec<u32> = Vec::with_capacity(num_codebooks);
        for cb in 0..num_codebooks {
            let row = &flat[cb * vocab..(cb + 1) * vocab];
            let mut best_i = 0usize;
            let mut best = row[0];
            for (i, &v) in row.iter().enumerate().skip(1) {
                if v > best {
                    best = v;
                    best_i = i;
                }
            }
            next_per_cb.push(best_i as u32);
        }

        for (cb, &t) in next_per_cb.iter().enumerate() {
            codes_per_cb[cb].push(t);
        }

        // Feed codebook-0 back into the LM as the next text-like token.
        // (The stage-2 transformer was trained on a unified token
        // stream, so this keeps the autoregressive context coherent
        // enough to exercise the forward path even without the
        // stage-1 hierarchy.)
        let next_lm_token = next_per_cb[0];
        tokens.push(next_lm_token);
        print!(".");
        std::io::stdout().flush().ok();

        if next_lm_token == args.stop_token {
            println!("\nreached stop token at step {step}");
            break;
        }
    }
    println!();
    println!("generation finished in {:?}", started.elapsed());

    let gen_len = codes_per_cb[0].len();
    if gen_len == 0 {
        anyhow::bail!("generation produced zero codes");
    }
    println!("generated {gen_len} codes per codebook");

    // 7) Build a (1, num_encodec_codebooks, T) code tensor for encodec.
    //    The EnCodec decoder expects as many codebook rows as
    //    `enc_weights.quantizers.len()` (derived from the
    //    target-bandwidth ladder). The MetaVoice model only emits
    //    `mv_cfg.num_codebooks` rows — typically fewer than EnCodec
    //    accepts. We pad the missing codebooks with zeros so the
    //    decoder runs end-to-end. Quality is determined by the
    //    populated rows.
    let enc_num_cb = enc_model.weights.quantizers.len();
    let mut codes_flat: Vec<u32> = Vec::with_capacity(enc_num_cb * gen_len);
    for cb in 0..enc_num_cb {
        if cb < num_codebooks {
            // Clamp to encodec vocabulary just in case the metavoice
            // vocab is wider than encodec's codebook size.
            let cap = enc_cfg.codebook_size as u32;
            for &c in &codes_per_cb[cb] {
                codes_flat.push(c.min(cap.saturating_sub(1)));
            }
        } else {
            codes_flat.extend(std::iter::repeat(0_u32).take(gen_len));
        }
    }
    let anchor = LazyTensor::from_f32(vec![0.0_f32; 1], Shape::from_dims(&[1]), &Device::cpu());
    let codes = anchor.const_u32_like(codes_flat, Shape::from_dims(&[1, enc_num_cb, gen_len]));

    // 8) Decode to waveform.
    println!("decoding {gen_len} codes through encodec");
    let pcm_lazy = enc_model
        .decode_codes(&codes)
        .map_err(|e| E::msg(format!("encodec decode: {e}")))?;
    let pcm_shape = pcm_lazy.shape().dims().to_vec();
    println!("pcm shape: {:?}", pcm_shape);
    if pcm_shape.len() != 3 || pcm_shape[0] != 1 {
        anyhow::bail!("encodec returned unexpected shape {:?}", pcm_shape);
    }
    let pcm_flat = pcm_lazy.realize_f32();
    // (1, audio_channels, T) — keep channel 0.
    let channels = pcm_shape[1];
    let t = pcm_shape[2];
    let mut pcm: Vec<f32> = Vec::with_capacity(t);
    for i in 0..t {
        // Average across channels (audio_channels == 1 for the default
        // encodec preset, so this is the identity).
        let mut acc = 0.0f32;
        for c in 0..channels {
            acc += pcm_flat[c * t + i];
        }
        pcm.push(acc / channels.max(1) as f32);
    }

    let pcm = normalize_loudness_host(&pcm, enc_cfg.sampling_rate as u32, true);

    // 9) Write the wav file.
    let mut output = std::fs::File::create(&args.output_wav)?;
    fuel_examples::wav::write_pcm_as_wav(&mut output, &pcm, enc_cfg.sampling_rate as u32)?;
    println!("wrote {} samples to {}", pcm.len(), args.output_wav);
    Ok(())
}

// ---- Helpers ---------------------------------------------------------------

/// Build a speaker embedding LazyTensor of shape
/// `(1, 1, speaker_emb_dim)`. Prefers a pre-baked `spk_emb` tensor (the
/// eager convention) over running the lazy speaker encoder, since the
/// upstream mel-spectrogram extraction needed to drive
/// `SpeakerEncoderModel::forward` is not available here. Falls back to
/// zeros when no file is provided or load fails.
fn load_speaker_embed(args: &Args, cfg: &MetaVoiceConfig) -> Result<LazyTensor> {
    let device = Device::cpu();
    let shape = Shape::from_dims(&[1, 1, cfg.speaker_emb_dim]);

    if let Some(path) = &args.speaker_encoder {
        let pathbuf = std::path::PathBuf::from(path);
        match unsafe { MmapedSafetensors::multi(&[&pathbuf]) } {
            Ok(st) => {
                if let Ok(view) = st.get("spk_emb") {
                    let bytes = view.data();
                    let mut out: Vec<f32> = Vec::with_capacity(bytes.len() / 4);
                    for chunk in bytes.chunks_exact(4) {
                        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                    if out.len() == cfg.speaker_emb_dim {
                        println!("loaded spk_emb tensor ({} floats)", out.len());
                        return Ok(LazyTensor::from_f32(out, shape, &device));
                    } else {
                        eprintln!(
                            "speaker-encoder file has spk_emb with {} elts, \
                             expected {}; falling back to zeros",
                            out.len(),
                            cfg.speaker_emb_dim
                        );
                    }
                } else {
                    eprintln!(
                        "speaker-encoder file {path} has no `spk_emb` tensor; \
                         the lazy speaker encoder needs a mel-spectrogram \
                         pipeline we don't yet have, so we cannot derive one \
                         from the LSTM weights alone. Falling back to zeros."
                    );
                }
            }
            Err(e) => {
                eprintln!("could not mmap speaker-encoder file {path}: {e}; falling back to zeros");
            }
        }
    } else {
        println!("no --speaker-encoder given; using a zero speaker vector");
    }

    Ok(LazyTensor::from_f32(
        vec![0.0_f32; cfg.speaker_emb_dim],
        shape,
        &device,
    ))
}

/// Byte-level fallback tokenizer. Each prompt byte becomes a token id
/// modulo `vocab_size`. This stands in for the eager `tokenizers::BPE`
/// (tiktoken-format) tokenizer that the retired binary loaded out of
/// `first_stage.meta.json` — that loader lives in the retired
/// fuel-transformers tree and has not been lazy-ported.
fn byte_level_encode(text: &str, vocab_size: usize) -> Vec<u32> {
    let cap = vocab_size as u32;
    text.bytes()
        .map(|b| (b as u32) % cap.max(1))
        .collect()
}

/// Host-side BS.1770 loudness normalization, mirroring the encodec
/// example. The eager `fuel_examples::audio::normalize_loudness` helper
/// expects an eager `Tensor`; the lazy pipeline already has a flat
/// `Vec<f32>`, so we do the EBU R128 gate + tanh compressor here.
///
/// Logic mirrors:
///   <https://github.com/facebookresearch/audiocraft/blob/69fea8b290ad1b4b40d28f92d1dfc0ab01dbab85/audiocraft/data/audio_utils.py#L57>
fn normalize_loudness_host(wav: &[f32], sample_rate: u32, loudness_compressor: bool) -> Vec<f32> {
    use fuel_examples::bs1770;

    if wav.is_empty() {
        return Vec::new();
    }

    let mut sumsq: f64 = 0.0;
    for &v in wav.iter() {
        sumsq += (v as f64) * (v as f64);
    }
    let energy = (sumsq / wav.len() as f64).sqrt() as f32;
    if energy < 2e-3 {
        return wav.to_vec();
    }

    let mut meter = bs1770::ChannelLoudnessMeter::new(sample_rate);
    meter.push(wav.iter().copied());
    let power = meter.as_100ms_windows();
    let loudness = match bs1770::gated_mean(power) {
        None => return wav.to_vec(),
        Some(gp) => gp.loudness_lkfs() as f64,
    };
    let delta_loudness = -14.0 - loudness;
    let gain = 10f64.powf(delta_loudness / 20.0) as f32;
    let mut out: Vec<f32> = wav.iter().map(|&v| v * gain).collect();
    if loudness_compressor {
        for v in out.iter_mut() {
            *v = v.tanh();
        }
    }
    out
}

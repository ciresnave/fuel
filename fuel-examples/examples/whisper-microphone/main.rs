//! Whisper microphone — lazy port.
//!
//! Streams audio from the default input device, resamples to 16 kHz,
//! buffers 10-second windows, runs the Whisper encoder + greedy /
//! temperature-fallback decoder via `fuel::lazy_whisper`, and prints
//! the running transcription.
//!
//! Migration notes vs the eager binary:
//!   * `fuel_transformers::models::whisper` → `fuel::lazy_whisper`
//!     (`WhisperConfig` / `WhisperModel` / `WhisperWeights`) and
//!     `fuel::lazy_whisper_audio` for the host-side pcm→log-mel
//!     pipeline.
//!   * The eager `Model::encoder_forward` / `decoder_forward` /
//!     `decoder_final_linear` triad collapses into a single
//!     `forward_decoder(&tokens, &encoder_out)` that returns logits of
//!     shape `[1, seq, vocab]`. The no-speech / sample-row logic
//!     extracts the relevant row from the realized f32 vec instead of
//!     using `Tensor::i(..)`.
//!   * `mel` rides as a flat `Vec<f32>` of shape `[num_mel_bins, T]`
//!     (no Tensor wrapper) — narrowing along the time axis is a
//!     row-major slice copy. The lazy encoder requires an even
//!     `mel_time`; oddities are trimmed away.
//!   * `suppress_tokens` is loaded from the raw config.json — the lazy
//!     `WhisperConfig` doesn't carry that field.
//!   * `--quantized` is preserved in the CLI and dispatches to
//!     `lazy_quantized_whisper::QuantizedWhisperModel::from_gguf`. That
//!     loader is currently a stub and errors at construction time;
//!     normal (safetensors) mode runs end-to-end.

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::{distr::Distribution, SeedableRng};
use tokenizers::Tokenizer;

mod multilingual;

use fuel::lazy::LazyTensor;
use fuel::lazy_whisper::{WhisperConfig, WhisperModel, WhisperWeights};
use fuel::lazy_quantized_whisper::QuantizedWhisperModel;
use fuel::lazy_whisper_audio as audio;
use fuel::safetensors::MmapedSafetensors;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

// ---- Whisper audio + tokenizer constants -----------------------------------
//
// These live alongside the eager `fuel_transformers::models::whisper`
// constants but the lazy port doesn't re-export them. The values match
// the OpenAI Whisper reference verbatim.
const SAMPLE_RATE: usize = audio::SAMPLE_RATE;
const HOP_LENGTH: usize = audio::HOP_LENGTH;
const N_FRAMES: usize = audio::N_SAMPLES / HOP_LENGTH; // 3000 frames per 30-second chunk

const NO_SPEECH_THRESHOLD: f64 = 0.6;
const LOGPROB_THRESHOLD: f64 = -1.0;
const TEMPERATURES: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;

const SOT_TOKEN: &str = "<|startoftranscript|>";
const TRANSCRIBE_TOKEN: &str = "<|transcribe|>";
const TRANSLATE_TOKEN: &str = "<|translate|>";
const NO_TIMESTAMPS_TOKEN: &str = "<|notimestamps|>";
const EOT_TOKEN: &str = "<|endoftext|>";
const NO_SPEECH_TOKENS: [&str; 2] = ["<|nocaptions|>", "<|nospeech|>"];

pub enum Model {
    Normal(WhisperModel),
    Quantized(QuantizedWhisperModel),
}

impl Model {
    pub fn config(&self) -> &WhisperConfig {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    /// Run the encoder on a row-major `(num_mel_bins, mel_time)` mel
    /// spectrogram and return the `[1, mel_time/2, d_model]` encoder
    /// context lazy tensor.
    pub fn encoder_forward(&self, mel: &[f32], mel_time: usize) -> fuel::Result<LazyTensor> {
        match self {
            Self::Normal(m) => m.forward_encoder(mel, mel_time),
            Self::Quantized(m) => m.forward_encoder(mel, mel_time),
        }
    }

    /// Run the decoder for the full `tokens` prefix and return the
    /// `[1, seq, vocab_size]` logits lazy tensor.
    pub fn decoder_forward(
        &self,
        tokens: &[u32],
        encoder_out: &LazyTensor,
    ) -> fuel::Result<LazyTensor> {
        match self {
            Self::Normal(m) => m.forward_decoder(tokens, encoder_out),
            Self::Quantized(m) => m.forward_decoder(tokens, encoder_out),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

struct Decoder {
    model: Model,
    /// Optional per-vocab additive mask. Indices in `suppress_tokens`
    /// (and `<|notimestamps|>` when `timestamps` is true) are set to
    /// `-inf`; everything else is `0.0`.
    suppress_tokens: Vec<f32>,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    verbose: bool,
    tokenizer: Tokenizer,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        suppress_token_ids: &[u32],
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        verbose: bool,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let vocab_size = model.config().vocab_size;
        let suppress_set: std::collections::HashSet<u32> = suppress_token_ids.iter().copied().collect();
        let suppress_tokens: Vec<f32> = (0..vocab_size as u32)
            .map(|i| {
                if suppress_set.contains(&i) || (timestamps && i == no_timestamps_token) {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let sot_token = token_id(&tokenizer, SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, EOT_TOKEN)?;
        let no_speech_token = NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &[f32], mel_time: usize, t: f64) -> Result<DecodingResult> {
        let model = &self.model;
        let audio_features = model
            .encoder_forward(mel, mel_time)
            .map_err(|e| E::msg(format!("encoder: {e}")))?;
        if self.verbose {
            println!("audio features mel_time/2: {}", mel_time / 2);
        }
        let cfg = model.config();
        let vocab = cfg.vocab_size;
        let max_target = cfg.max_target_positions;
        let sample_len = max_target / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            // forward_decoder returns logits of shape [1, seq, vocab].
            let logits = model
                .decoder_forward(&tokens, &audio_features)
                .map_err(|e| E::msg(format!("decoder: {e}")))?;
            let flat = logits.realize_f32();
            let seq = tokens.len();
            if flat.len() != seq * vocab {
                anyhow::bail!(
                    "decoder logits unexpected size: got {} expected {} (= seq {seq} * vocab {vocab})",
                    flat.len(), seq * vocab,
                );
            }

            // Extract the no_speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let first_row = &flat[0..vocab];
                let row_max = first_row
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f64;
                for &v in first_row {
                    sum += ((v - row_max) as f64).exp();
                }
                let ns_logit = first_row[self.no_speech_token as usize];
                no_speech_prob = (((ns_logit - row_max) as f64).exp()) / sum;
            }

            // Last-row logits for the next-token decision.
            let last_row_start = (seq - 1) * vocab;
            let mut row: Vec<f32> = flat[last_row_start..last_row_start + vocab].to_vec();
            // Apply additive suppress mask.
            for (r, s) in row.iter_mut().zip(self.suppress_tokens.iter()) {
                *r += *s;
            }

            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let next_token = if t > 0f64 {
                let scaled: Vec<f32> = row.iter().map(|v| *v / t as f32).collect();
                let row_max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = scaled
                    .iter()
                    .map(|v| ((*v - row_max) as f64).exp() as f32)
                    .collect();
                let sum: f32 = probs.iter().sum();
                if sum > 0.0 {
                    for p in probs.iter_mut() {
                        *p /= sum;
                    }
                }
                let distr = rand::distr::weighted::WeightedIndex::new(&probs)?;
                distr.sample(&mut self.rng) as u32
            } else {
                row.iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);

            // Per-token probability of `next_token` from a softmax over the masked row (used for
            // avg_logprob — matches the eager binary).
            let row_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f64;
            for &v in &row {
                sum += ((v - row_max) as f64).exp();
            }
            let prob = (((row[next_token as usize] - row_max) as f64).exp()) / sum;
            if next_token == self.eot_token || tokens.len() > max_target {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, mel: &[f32], mel_time: usize) -> Result<DecodingResult> {
        for (i, &t) in TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(mel, mel_time, t);
            if i == TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn run(&mut self, mel: &[f32], mel_time: usize, times: Option<(f64, f64)>) -> Result<Vec<Segment>> {
        let num_mel_bins = self.model.config().num_mel_bins;
        let content_frames = mel_time;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let start = std::time::Instant::now();
            let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let mut segment_size = usize::min(content_frames - seek, N_FRAMES);
            // The lazy encoder requires an even mel_time (stride-2 conv).
            if !segment_size.is_multiple_of(2) {
                segment_size -= 1;
            }
            if segment_size == 0 {
                break;
            }
            let mel_segment =
                narrow_time_axis(mel, num_mel_bins, content_frames, seek, segment_size);
            let segment_duration = (segment_size * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment, segment_size)?;
            seek += segment_size;
            if dr.no_speech_prob > NO_SPEECH_THRESHOLD && dr.avg_logprob < LOGPROB_THRESHOLD {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            if self.timestamps {
                println!(
                    "{:.1}s -- {:.1}s",
                    segment.start,
                    segment.start + segment.duration,
                );
                let mut tokens_to_decode = vec![];
                let mut prev_timestamp_s = 0f32;
                for &token in segment.dr.tokens.iter() {
                    if token == self.sot_token || token == self.eot_token {
                        continue;
                    }
                    // The no_timestamp_token is the last before the timestamp ones.
                    if token > self.no_timestamps_token {
                        let timestamp_s = (token - self.no_timestamps_token + 1) as f32 / 50.;
                        if !tokens_to_decode.is_empty() {
                            let text = self
                                .tokenizer
                                .decode(&tokens_to_decode, true)
                                .map_err(E::msg)?;
                            println!("  {:.1}s-{:.1}s: {}", prev_timestamp_s, timestamp_s, text);
                            tokens_to_decode.clear()
                        }
                        prev_timestamp_s = timestamp_s;
                    } else {
                        tokens_to_decode.push(token)
                    }
                }
                if !tokens_to_decode.is_empty() {
                    let text = self
                        .tokenizer
                        .decode(&tokens_to_decode, true)
                        .map_err(E::msg)?;
                    if !text.is_empty() {
                        println!("  {:.1}s-...: {}", prev_timestamp_s, text);
                    }
                    tokens_to_decode.clear()
                }
            } else {
                match times {
                    Some((start, end)) => {
                        println!("{:.1}s -- {:.1}s: {}", start, end, segment.dr.text)
                    }
                    None => {
                        println!(
                            "{:.1}s -- {:.1}s: {}",
                            segment.start,
                            segment.start + segment.duration,
                            segment.dr.text,
                        )
                    }
                }
            }
            if self.verbose {
                println!("{seek}: {segment:?}, in {:?}", start.elapsed());
            }
            segments.push(segment)
        }
        Ok(segments)
    }

    fn set_language_token(&mut self, language_token: Option<u32>) {
        self.language_token = language_token;
    }

    fn model(&self) -> &Model {
        &self.model
    }
}

/// Narrow a row-major `(num_mel_bins, total_frames)` mel tensor along
/// the time axis to `(num_mel_bins, segment_size)` starting at frame
/// `start`. Mirrors `Tensor::narrow(2, start, segment_size)` for the
/// `(1, num_mel_bins, total_frames)` layout the eager binary used.
pub(crate) fn narrow_time_axis(
    mel: &[f32],
    num_mel_bins: usize,
    total_frames: usize,
    start: usize,
    segment_size: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_mel_bins * segment_size);
    for m in 0..num_mel_bins {
        let row_off = m * total_frames + start;
        out.extend_from_slice(&mel[row_off..row_off + segment_size]);
    }
    out
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> fuel::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => fuel::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Task {
    Transcribe,
    Translate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WhichModel {
    Tiny,
    #[value(name = "tiny.en")]
    TinyEn,
    Base,
    #[value(name = "base.en")]
    BaseEn,
    Small,
    #[value(name = "small.en")]
    SmallEn,
    Medium,
    #[value(name = "medium.en")]
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
    #[value(name = "distil-medium.en")]
    DistilMediumEn,
    #[value(name = "distil-large-v2")]
    DistilLargeV2,
}

impl WhichModel {
    fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::LargeV3
            | Self::LargeV3Turbo
            | Self::DistilLargeV2 => true,
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn | Self::DistilMediumEn => {
                false
            }
        }
    }

    fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::LargeV3Turbo => ("openai/whisper-large-v3-turbo", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU. The lazy port realizes through
    /// the default router today; this flag is preserved for CLI parity.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    model_id: Option<String>,

    /// The model to use, check out available models:
    /// https://huggingface.co/models?search=whisper
    #[arg(long)]
    revision: Option<String>,

    /// The model to be used, can be tiny, small, medium.
    #[arg(long, default_value = "tiny.en")]
    model: WhichModel,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    quantized: bool,

    /// Language.
    #[arg(long)]
    language: Option<String>,

    /// Task, when no task is specified, the input tokens contain only the sot token which can
    /// improve things when in no-timestamp mode.
    #[arg(long)]
    task: Option<Task>,

    /// Timestamps mode, this is not fully implemented yet.
    #[arg(long)]
    timestamps: bool,

    /// Print the full DecodingResult structure rather than just the text.
    #[arg(long)]
    verbose: bool,

    /// The input device to use.
    #[arg(long)]
    device: Option<String>,
}

pub fn main() -> Result<()> {
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
    // `cpu` flag is preserved for CLI parity; lazy currently routes
    // through the default backend pipeline.
    let _ = args.cpu;

    let (default_model, default_revision) = if args.quantized {
        ("lmz/fuel-whisper", "main")
    } else {
        args.model.model_and_revision()
    };
    let default_model = default_model.to_string();
    let default_revision = default_revision.to_string();
    let (model_id, revision) = match (args.model_id, args.revision) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => (model_id, "main".to_string()),
        (None, Some(revision)) => (default_model, revision),
        (None, None) => (default_model, default_revision),
    };

    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let (config, tokenizer, model) = if args.quantized {
            let ext = match args.model {
                WhichModel::TinyEn => "tiny-en",
                WhichModel::Tiny => "tiny",
                _ => unimplemented!("no quantized support for {:?}", args.model),
            };
            (
                repo.get(&format!("config-{ext}.json"))?,
                repo.get(&format!("tokenizer-{ext}.json"))?,
                repo.get(&format!("model-{ext}-q80.gguf"))?,
            )
        } else {
            let config = repo.get("config.json")?;
            let tokenizer = repo.get("tokenizer.json")?;
            let model = repo.get("model.safetensors")?;
            (config, tokenizer, model)
        };
        (config, tokenizer, model)
    };
    let config_str = std::fs::read_to_string(config_filename)?;
    let config: WhisperConfig = WhisperConfig::from_hf_json_str(&config_str)
        .map_err(|e| E::msg(format!("parse config: {e}")))?;
    // suppress_tokens is not declared on the lazy WhisperConfig; pull
    // it out of the raw JSON so the decoder still applies the original
    // Whisper suppression mask.
    let suppress_token_ids: Vec<u32> = {
        let v: serde_json::Value = serde_json::from_str(&config_str)?;
        v.get("suppress_tokens")
            .and_then(|t| t.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_i64().map(|i| i as u32))
                    .collect()
            })
            .unwrap_or_default()
    };
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let model = if args.quantized {
        Model::Quantized(
            QuantizedWhisperModel::from_gguf(&weights_filename)
                .map_err(|e| E::msg(format!("quantized whisper from_gguf: {e}")))?,
        )
    } else {
        let st = unsafe { MmapedSafetensors::multi(&[weights_filename]) }
            .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
        let weights = WhisperWeights::load_from_mmapped(&st, &config)
            .map_err(|e| E::msg(format!("load whisper weights: {e}")))?;
        Model::Normal(WhisperModel {
            config: config.clone(),
            weights,
        })
    };

    let mut decoder = Decoder::new(
        model,
        tokenizer.clone(),
        args.seed,
        &suppress_token_ids,
        /* language_token */ None,
        args.task,
        args.timestamps,
        args.verbose,
    )?;

    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("../whisper/melfilters.bytes").as_slice(),
        128 => include_bytes!("../whisper/melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    // Set up the input device and stream with the default input config.
    let host = cpal::default_host();
    let audio_device = match args.device.as_ref() {
        None => host.default_input_device(),
        Some(device) => host
            .input_devices()?
            .find(|x| x.name().map_or(false, |y| &y == device)),
    }
    .expect("failed to find the audio input device");

    let audio_config = audio_device
        .default_input_config()
        .expect("Failed to get default input config");
    println!("audio config {audio_config:?}");

    let channel_count = audio_config.channels() as usize;
    let in_sample_rate = audio_config.sample_rate().0 as usize;
    let resample_ratio = 16000. / in_sample_rate as f64;
    let mut resampler = rubato::FastFixedIn::new(
        resample_ratio,
        10.,
        rubato::PolynomialDegree::Septic,
        1024,
        1,
    )?;
    let (tx, rx) = std::sync::mpsc::channel();
    let stream = audio_device.build_input_stream(
        &audio_config.config(),
        move |pcm: &[f32], _: &cpal::InputCallbackInfo| {
            let pcm = pcm
                .iter()
                .step_by(channel_count)
                .copied()
                .collect::<Vec<f32>>();
            if !pcm.is_empty() {
                tx.send(pcm).unwrap()
            }
        },
        move |err| {
            eprintln!("an error occurred on stream: {}", err);
        },
        None,
    )?;
    stream.play()?;

    // loop to process the audio data forever (until the user stops the program)
    println!("transcribing audio...");
    let mut buffered_pcm = vec![];
    let mut language_token_set = false;
    while let Ok(pcm) = rx.recv() {
        use rubato::Resampler;

        buffered_pcm.extend_from_slice(&pcm);
        if buffered_pcm.len() < 10 * in_sample_rate {
            continue;
        }
        let mut resampled_pcm = vec![];
        // resample the audio, one chunk of 1024 samples at a time.
        // in case the audio input failed to produce an exact multiple of 1024 samples,
        // process the remainder on the next iteration of the loop.
        let full_chunks = buffered_pcm.len() / 1024;
        let remainder = buffered_pcm.len() % 1024;
        for chunk in 0..full_chunks {
            let buffered_pcm = &buffered_pcm[chunk * 1024..(chunk + 1) * 1024];
            let pcm = resampler.process(&[&buffered_pcm], None)?;
            resampled_pcm.extend_from_slice(&pcm[0]);
        }
        let pcm = resampled_pcm;
        println!("{} {}", buffered_pcm.len(), pcm.len());
        if remainder == 0 {
            buffered_pcm.clear();
        } else {
            // efficiently copy the remainder to the beginning of the `buffered_pcm` buffer and
            // truncate it.  That's more efficient then allocating a new vector and copying into it
            println!("audio device produced partial chunk with {remainder} samples; processing the remainder on the next iteration of the loop");
            buffered_pcm.copy_within(full_chunks * 1024.., 0);
            buffered_pcm.truncate(remainder);
        }
        let num_mel_bins = config.num_mel_bins;
        let mel = audio::pcm_to_mel(&pcm, &mel_filters, num_mel_bins)
            .map_err(|e| E::msg(format!("pcm_to_mel: {e}")))?;
        let mel_total = mel.len() / num_mel_bins;
        // Lazy encoder requires an even mel_time (stride-2 conv).
        let mel_time = mel_total - (mel_total % 2);
        if mel_time == 0 {
            continue;
        }
        // Trim each row to mel_time columns.
        let mel: Vec<f32> = if mel_time == mel_total {
            mel
        } else {
            let mut trimmed = Vec::with_capacity(num_mel_bins * mel_time);
            for m in 0..num_mel_bins {
                let row = &mel[m * mel_total..(m + 1) * mel_total];
                trimmed.extend_from_slice(&row[..mel_time]);
            }
            trimmed
        };

        // on the first iteration, we detect the language and set the language token.
        if !language_token_set {
            let language_token = match (args.model.is_multilingual(), args.language.clone()) {
                (true, None) => Some(multilingual::detect_language(
                    decoder.model(),
                    &tokenizer,
                    &mel,
                    mel_time,
                )?),
                (false, None) => None,
                (true, Some(language)) => match token_id(&tokenizer, &format!("<|{language}|>")) {
                    Ok(token_id) => Some(token_id),
                    Err(_) => anyhow::bail!("language {language} is not supported"),
                },
                (false, Some(_)) => {
                    anyhow::bail!("a language cannot be set for non-multilingual models")
                }
            };
            println!("language_token: {:?}", language_token);
            decoder.set_language_token(language_token);
            language_token_set = true;
        }
        decoder.run(&mel, mel_time, None)?;
    }

    Ok(())
}

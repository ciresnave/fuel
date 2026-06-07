// https://github.com/openai/whisper/blob/main/whisper/model.py/rgs
// Migrated to the lazy-graph API.
// TODO:
// - Batch size greater than 1.
// - More token filters (SuppressBlanks, ApplyTimestampRules).

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::SeedableRng;
use tokenizers::Tokenizer;

mod multilingual;

use fuel::lazy_whisper::{WhisperConfig, WhisperModel, WhisperWeights};
use fuel::lazy_quantized_whisper::QuantizedWhisperModel;

// Audio / decoding constants — formerly re-exported via
// `fuel_transformers::models::whisper::*`.
pub const SAMPLE_RATE: usize = 16000;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_LENGTH: usize = 30;
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE;
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000 frames per 30 s chunk

pub const NO_SPEECH_THRESHOLD: f64 = 0.6;
pub const LOGPROB_THRESHOLD: f64 = -1.0;
pub const TEMPERATURES: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
pub const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;

pub const SOT_TOKEN: &str = "<|startoftranscript|>";
pub const TRANSCRIBE_TOKEN: &str = "<|transcribe|>";
pub const TRANSLATE_TOKEN: &str = "<|translate|>";
pub const NO_TIMESTAMPS_TOKEN: &str = "<|notimestamps|>";
pub const EOT_TOKEN: &str = "<|endoftext|>";
pub const NO_SPEECH_TOKENS: [&str; 2] = ["<|nocaptions|>", "<|nospeech|>"];

/// HF Whisper `config.json` carries a `suppress_tokens` field that the
/// lazy `WhisperConfig` doesn't model — pull it out side-band so the
/// decoder can mask the logits at sample time.
fn parse_suppress_tokens(json_str: &str) -> Vec<u32> {
    let v: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    v.get("suppress_tokens")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_i64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default()
}

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

    /// Run the encoder on a flat mel `[n_mels, mel_time]` slice and
    /// return the realized `[1, mel_time/2, d_model]` encoder context as
    /// a `Vec<f32>`. The decoder re-wraps it as a fresh lazy tensor on
    /// each step.
    pub fn encoder_forward(&self, mel_flat: &[f32], mel_time: usize) -> Result<Vec<f32>> {
        match self {
            Self::Normal(m) => {
                let enc = m.forward_encoder(mel_flat, mel_time)
                    .map_err(|e| E::msg(format!("encoder: {e}")))?;
                Ok(enc.realize_f32())
            }
            Self::Quantized(m) => {
                let enc = m.forward_encoder(mel_flat, mel_time)
                    .map_err(|e| E::msg(format!("encoder: {e}")))?;
                Ok(enc.realize_f32())
            }
        }
    }

    /// Run the decoder against a pre-realized encoder context (re-wrapped
    /// here as a fresh lazy tensor) and return realized logits of shape
    /// `[1, seq, vocab]` flattened row-major.
    pub fn decoder_logits(
        &self,
        tokens: &[u32],
        encoder_flat: &[f32],
        mel_time: usize,
    ) -> Result<Vec<f32>> {
        let cfg = self.config();
        let t_half = mel_time / 2;
        let enc_shape = fuel::Shape::from_dims(&[1, t_half, cfg.d_model]);
        let enc_t = fuel::lazy::LazyTensor::from_f32(
            encoder_flat.to_vec(),
            enc_shape,
            &fuel::Device::cpu(),
        );
        match self {
            Self::Normal(m) => {
                let logits = m
                    .forward_decoder(tokens, &enc_t)
                    .map_err(|e| E::msg(format!("decoder: {e}")))?;
                Ok(logits.realize_f32())
            }
            Self::Quantized(m) => {
                let logits = m
                    .forward_decoder(tokens, &enc_t)
                    .map_err(|e| E::msg(format!("decoder: {e}")))?;
                Ok(logits.realize_f32())
            }
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
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    max_initial_timestamp_index: Option<u32>,
    verbose: bool,
    tokenizer: Tokenizer,
    /// Additive mask applied to logits to suppress non-speech tokens.
    suppress_tokens: Vec<f32>,
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
        suppress_tokens_ids: &[u32],
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        max_initial_timestamp_index: Option<u32>,
        verbose: bool,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let vocab = model.config().vocab_size as u32;
        let suppress_set: std::collections::HashSet<u32> =
            suppress_tokens_ids.iter().copied().collect();
        let suppress_tokens: Vec<f32> = (0..vocab)
            .map(|i| {
                if suppress_set.contains(&i) || timestamps && i == no_timestamps_token {
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
            max_initial_timestamp_index,
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

    fn decode(&mut self, mel_segment: &[f32], mel_time: usize, t: f64) -> Result<DecodingResult> {
        // Run the encoder once per segment.
        let audio_features = self.model.encoder_forward(mel_segment, mel_time)?;
        if self.verbose {
            let t_half = mel_time / 2;
            println!(
                "audio features: [1, {}, {}]",
                t_half,
                self.model.config().d_model
            );
        }
        let sample_len = self.model.config().max_target_positions / 2;
        let vocab = self.model.config().vocab_size;
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
            // Run the decoder over all current tokens. Returns flat
            // `[1, seq, vocab]` logits.
            let logits_flat =
                self.model.decoder_logits(&tokens, &audio_features, mel_time)?;
            let seq = tokens.len();
            assert_eq!(logits_flat.len(), seq * vocab);

            // Extract the no_speech probability on the first iteration
            // by looking at the first token's logits and the probability
            // for the according token.
            if i == 0 {
                let first_row = &logits_flat[0..vocab];
                let probs = softmax_vec(first_row);
                no_speech_prob = probs[self.no_speech_token as usize] as f64;
            }

            // Last-row logits — `[vocab]`.
            let last_off = (seq - 1) * vocab;
            let mut logits: Vec<f32> = logits_flat[last_off..last_off + vocab].to_vec();

            // Apply timestamp rules when timestamps are enabled
            if self.timestamps {
                self.apply_timestamp_rules(&mut logits, &tokens);
            }

            // Apply the cached suppress-token additive mask.
            for (l, s) in logits.iter_mut().zip(self.suppress_tokens.iter()) {
                *l += *s;
            }

            let next_token = if t > 0f64 {
                let scaled: Vec<f32> = logits.iter().map(|&v| v / (t as f32)).collect();
                let prs = softmax_vec(&scaled);
                let distr = WeightedIndex::new(&prs)?;
                distr.sample(&mut self.rng) as u32
            } else {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let probs = softmax_vec(&logits);
            let prob = probs[next_token as usize] as f64;
            if next_token == self.eot_token
                || tokens.len() > self.model.config().max_target_positions
            {
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

    fn decode_with_fallback(
        &mut self,
        mel_segment: &[f32],
        mel_time: usize,
    ) -> Result<DecodingResult> {
        for (i, &t) in TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(mel_segment, mel_time, t);
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

    /// In-place additive masks on `logits` implementing the timestamp
    /// rules from openai/whisper. Operates on a flat `[vocab]` slice.
    fn apply_timestamp_rules(&self, logits: &mut [f32], tokens: &[u32]) {
        let timestamp_begin = self.no_timestamps_token + 1;
        let vocab_size = self.model.config().vocab_size as u32;

        // ========== SETUP: Extract sampled tokens for analysis ==========
        let sample_begin = if self.language_token.is_some() { 3 } else { 2 };
        let sampled_tokens: &[u32] = if tokens.len() > sample_begin {
            &tokens[sample_begin..]
        } else {
            &[]
        };

        // ========== RULE 1: Timestamp pairing constraints ==========
        // Timestamps must come in pairs, except directly before EOT
        if !sampled_tokens.is_empty() {
            let last_was_timestamp = sampled_tokens
                .last()
                .map(|&t| t >= timestamp_begin)
                .unwrap_or(false);
            let penultimate_was_timestamp = if sampled_tokens.len() >= 2 {
                sampled_tokens[sampled_tokens.len() - 2] >= timestamp_begin
            } else {
                false
            };
            if last_was_timestamp {
                if penultimate_was_timestamp {
                    // Has to be non-timestamp — suppress timestamp tokens
                    for i in 0..vocab_size {
                        if i >= timestamp_begin {
                            logits[i as usize] = f32::NEG_INFINITY;
                        }
                    }
                } else {
                    // Cannot be normal text tokens — suppress everything before EOT
                    for i in 0..vocab_size {
                        if i < self.eot_token {
                            logits[i as usize] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // ========== RULE 2: Non-decreasing timestamp constraint ==========
            let timestamp_tokens: Vec<u32> = sampled_tokens
                .iter()
                .filter(|&&t| t >= timestamp_begin)
                .copied()
                .collect();
            if !timestamp_tokens.is_empty() {
                let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
                    *timestamp_tokens.last().unwrap()
                } else {
                    timestamp_tokens.last().unwrap() + 1
                };
                for i in 0..vocab_size {
                    if i >= timestamp_begin && i < timestamp_last {
                        logits[i as usize] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // ========== RULE 3: Force initial timestamp ==========
        if tokens.len() == sample_begin {
            for i in 0..vocab_size {
                if i < timestamp_begin {
                    logits[i as usize] = f32::NEG_INFINITY;
                }
            }
            if let Some(max_initial_timestamp_index) = self.max_initial_timestamp_index {
                let last_allowed = timestamp_begin + max_initial_timestamp_index;
                if last_allowed < vocab_size {
                    for i in 0..vocab_size {
                        if i > last_allowed {
                            logits[i as usize] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }

        // ========== RULE 4: Probability-based timestamp preference ==========
        let log_probs = log_softmax_vec(logits);
        let timestamp_logprob = logsumexp(&log_probs[timestamp_begin as usize..]);
        let max_text_token_logprob = log_probs[..timestamp_begin as usize]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if timestamp_logprob > max_text_token_logprob {
            // Only consider timestamp tokens.
            for i in 0..vocab_size {
                if i < timestamp_begin {
                    logits[i as usize] = f32::NEG_INFINITY;
                }
            }
        }
    }

    fn run(&mut self, mel: &[f32], n_mels: usize, total_frames: usize) -> Result<Vec<Segment>> {
        let mut seek = 0;
        let mut segments = vec![];
        while seek < total_frames {
            let start = std::time::Instant::now();
            let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let segment_size = usize::min(total_frames - seek, N_FRAMES);
            // The encoder requires an even mel_time (stride-2 conv).
            let segment_size = if segment_size.is_multiple_of(2) {
                segment_size
            } else {
                segment_size - 1
            };
            if segment_size == 0 {
                break;
            }
            let mel_segment = extract_mel_segment(mel, n_mels, total_frames, seek, segment_size);
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
                        let timestamp_s =
                            (token - self.no_timestamps_token + 1) as f32 / 50.;
                        if !tokens_to_decode.is_empty() {
                            let text = self
                                .tokenizer
                                .decode(&tokens_to_decode, true)
                                .map_err(E::msg)?;
                            println!(
                                "  {:.1}s-{:.1}s: {}",
                                prev_timestamp_s, timestamp_s, text
                            );
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
                println!(
                    "{:.1}s -- {:.1}s: {}",
                    segment.start,
                    segment.start + segment.duration,
                    segment.dr.text,
                )
            }
            if self.verbose {
                println!("{seek}: {segment:?}, in {:?}", start.elapsed());
            }
            segments.push(segment)
        }
        Ok(segments)
    }
}

/// Extract a `[n_mels, segment_size]` flat slice from a `[n_mels, total_frames]`
/// row-major mel buffer.
fn extract_mel_segment(
    mel: &[f32],
    n_mels: usize,
    total_frames: usize,
    seek: usize,
    segment_size: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_mels * segment_size);
    for m in 0..n_mels {
        let row = &mel[m * total_frames + seek..m * total_frames + seek + segment_size];
        out.extend_from_slice(row);
    }
    out
}

fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    let s = if s == 0.0 { 1.0 } else { s };
    exps.into_iter().map(|v| v / s).collect()
}

fn log_softmax_vec(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    let log_s = s.ln();
    logits.iter().map(|&v| (v - m) - log_s).collect()
}

fn logsumexp(vals: &[f32]) -> f32 {
    if vals.is_empty() {
        return f32::NEG_INFINITY;
    }
    let m = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return m;
    }
    let s: f32 = vals.iter().map(|&v| (v - m).exp()).sum();
    m + s.ln()
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
    #[value(name = "distil-large-v3")]
    DistilLargeV3,
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
            | Self::DistilLargeV2
            | Self::DistilLargeV3 => true,
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
            Self::DistilLargeV3 => ("distil-whisper/distil-large-v3", "main"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
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

    /// The input to be processed, in wav format, will default to `jfk.wav`. Alternatively
    /// this can be set to sample:jfk, sample:gb1, ... to fetch a sample from the following
    /// repo: https://huggingface.co/datasets/Narsil/fuel_demo/
    #[arg(long)]
    input: Option<String>,

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

    /// Timestamps mode.
    #[arg(long, default_value_t = true)]
    timestamps: bool,

    /// Maximum initial timestamp index to consider.
    #[arg(long)]
    max_initial_timestamp_index: Option<u32>,

    /// Print the full DecodingResult structure rather than just the text.
    #[arg(long)]
    verbose: bool,
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

    let (config_filename, tokenizer_filename, weights_filename, input) = {
        let api = Api::new()?;
        let dataset = api.dataset("Narsil/fuel-examples".to_string());
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let sample = if let Some(input) = args.input {
            if let Some(sample) = input.strip_prefix("sample:") {
                dataset.get(&format!("samples_{sample}.wav"))?
            } else {
                std::path::PathBuf::from(input)
            }
        } else {
            println!("No audio file submitted: Downloading https://huggingface.co/datasets/Narsil/fuel_demo/blob/main/samples_jfk.wav");
            dataset.get("samples_jfk.wav")?
        };
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
        (config, tokenizer, model, sample)
    };
    let config_json = std::fs::read_to_string(&config_filename)?;
    let config: WhisperConfig = WhisperConfig::from_hf_json_str(&config_json)
        .map_err(|e| E::msg(format!("parse whisper config: {e}")))?;
    let suppress_tokens_ids = parse_suppress_tokens(&config_json);
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    let (pcm_data, sample_rate) = fuel_examples::audio::pcm_decode(input)?;
    if sample_rate != SAMPLE_RATE as u32 {
        anyhow::bail!("input file must have a {} sampling rate", SAMPLE_RATE)
    }
    println!("pcm data loaded {}", pcm_data.len());
    // Build the flat `[n_mels, n_frames]` mel via the lazy-graph-free
    // audio helper. The eager path used `m::audio::pcm_to_mel(config,
    // pcm_data, mel_filters)`; the lazy helper takes the explicit
    // `n_mels` instead of the config.
    let mel = fuel::lazy_whisper_audio::pcm_to_mel(&pcm_data, &mel_filters, config.num_mel_bins)
        .map_err(|e| E::msg(format!("pcm_to_mel: {e}")))?;
    let total_frames = mel.len() / config.num_mel_bins;
    println!("loaded mel: [{}, {}]", config.num_mel_bins, total_frames);

    let model = if args.quantized {
        let m = QuantizedWhisperModel::from_gguf(&weights_filename)
            .map_err(|e| E::msg(format!("quantized whisper: {e}")))?;
        Model::Quantized(m)
    } else {
        let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&[weights_filename]) }
            .map_err(|e| E::msg(format!("mmap: {e}")))?;
        let weights = WhisperWeights::load_from_mmapped(&st, &config)
            .map_err(|e| E::msg(format!("weights: {e}")))?;
        Model::Normal(WhisperModel { config: config.clone(), weights })
    };

    let language_token = match (args.model.is_multilingual(), args.language) {
        (true, None) => Some(multilingual::detect_language(
            &model,
            &tokenizer,
            &mel,
            config.num_mel_bins,
            total_frames,
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
    let mut dc = Decoder::new(
        model,
        tokenizer,
        args.seed,
        &suppress_tokens_ids,
        language_token,
        args.task,
        args.timestamps,
        args.max_initial_timestamp_index,
        args.verbose,
    )?;
    dc.run(&mel, config.num_mel_bins, total_frames)?;
    Ok(())
}

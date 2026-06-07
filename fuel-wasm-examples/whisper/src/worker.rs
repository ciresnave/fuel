use crate::languages::LANGUAGES;
use anyhow::Error as E;
use fuel::lazy::LazyTensor;
use fuel::lazy_quantized_whisper::QuantizedWhisperModel;
use fuel::lazy_whisper::{WhisperConfig, WhisperModel, WhisperWeights};
use fuel::lazy_whisper_audio as audio;
use fuel::safetensors::BufferedSafetensors;
use fuel::Shape;
use rand::{distr::Distribution, rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use yew_agent::{HandlerId, Public, WorkerLink};

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => ($crate::worker::log(&format_args!($($t)*).to_string()))
}

// ---- Whisper audio + decoding constants -----------------------------------
//
// Formerly re-exported via `fuel_transformers::models::whisper`. The
// lazy `lazy_whisper` module deliberately keeps these out of its
// surface, so we inline them here at their reference (OpenAI Whisper)
// values.
pub const SAMPLE_RATE: usize = audio::SAMPLE_RATE;
pub const HOP_LENGTH: usize = audio::HOP_LENGTH;
pub const N_FRAMES: usize = audio::N_SAMPLES / HOP_LENGTH; // 3000 frames per 30 s

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

    /// Run the encoder on a flat `[n_mels, mel_time]` slice and return
    /// the realized `[1, mel_time/2, d_model]` encoder context as a
    /// `Vec<f32>`. The decoder re-wraps it as a fresh lazy tensor each
    /// step.
    pub fn encoder_forward(&self, mel: &[f32], mel_time: usize) -> anyhow::Result<Vec<f32>> {
        match self {
            Self::Normal(m) => {
                let enc = m
                    .forward_encoder(mel, mel_time)
                    .map_err(|e| E::msg(format!("encoder: {e}")))?;
                Ok(enc.realize_f32())
            }
            Self::Quantized(m) => {
                let enc = m
                    .forward_encoder(mel, mel_time)
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
    ) -> anyhow::Result<Vec<f32>> {
        let cfg = self.config();
        let t_half = mel_time / 2;
        let enc_shape = Shape::from_dims(&[1, t_half, cfg.d_model]);
        let enc_t = LazyTensor::from_f32(
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
}

pub struct Decoder {
    model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    language: Option<String>,
    is_multilingual: bool,
    mel_filters: Vec<f32>,
    timestamps: bool,
    tokenizer: Tokenizer,
    /// Additive mask applied to logits to suppress non-speech tokens.
    suppress_tokens: Vec<f32>,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        mel_filters: Vec<f32>,
        task: Option<Task>,
        language: Option<String>,
        is_multilingual: bool,
        timestamps: bool,
    ) -> anyhow::Result<Self> {
        // Without a per-config suppress_tokens list (the lazy
        // `WhisperConfig` doesn't carry that field), apply an empty
        // additive mask. The wasm worker historically read this list
        // from `config.suppress_tokens`; if needed it can be threaded
        // through `ModelData` later.
        let suppress_tokens: Vec<f32> = vec![0.0_f32; model.config().vocab_size];
        let no_timestamps_token = token_id(&tokenizer, NO_TIMESTAMPS_TOKEN)?;
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
        let seed = 299792458;
        Ok(Self {
            model,
            rng: StdRng::seed_from_u64(seed),
            tokenizer,
            mel_filters,
            task,
            timestamps,
            language,
            is_multilingual,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            no_timestamps_token,
        })
    }

    fn decode(
        &mut self,
        mel_segment: &[f32],
        mel_time: usize,
        t: f64,
    ) -> anyhow::Result<DecodingResult> {
        // Resolve language (if multilingual) on the actual segment.
        let language_token = match (self.is_multilingual, &self.language) {
            (true, None) => Some(detect_language(
                &self.model,
                &self.tokenizer,
                mel_segment,
                mel_time,
            )?),
            (false, None) => None,
            (true, Some(language)) => {
                match token_id(&self.tokenizer, &format!("<|{:?}|>", self.language)) {
                    Ok(token_id) => Some(token_id),
                    Err(_) => anyhow::bail!("language {language} is not supported"),
                }
            }
            (false, Some(_)) => {
                anyhow::bail!("a language cannot be set for non-multilingual models")
            }
        };

        let audio_features = self.model.encoder_forward(mel_segment, mel_time)?;
        let cfg_d_model = self.model.config().d_model;
        let t_half = mel_time / 2;
        console_log!("audio features: [1, {}, {}]", t_half, cfg_d_model);
        let sample_len = self.model.config().max_target_positions / 2;
        let vocab = self.model.config().vocab_size;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = language_token {
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
            let logits_flat =
                self.model.decoder_logits(&tokens, &audio_features, mel_time)?;
            let seq = tokens.len();
            debug_assert_eq!(logits_flat.len(), seq * vocab);

            // Extract the no_speech probability on the first iteration by
            // looking at the first token's logits.
            if i == 0 {
                let first_row = &logits_flat[0..vocab];
                let probs = softmax_vec(first_row);
                no_speech_prob = probs[self.no_speech_token as usize] as f64;
            }

            // Last-row logits — `[vocab]`.
            let last_off = (seq - 1) * vocab;
            let mut logits: Vec<f32> = logits_flat[last_off..last_off + vocab].to_vec();

            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            for (l, s) in logits.iter_mut().zip(self.suppress_tokens.iter()) {
                *l += *s;
            }
            let next_token = if t > 0f64 {
                let scaled: Vec<f32> = logits.iter().map(|&v| v / (t as f32)).collect();
                let prs = softmax_vec(&scaled);
                let distr = rand::distr::weighted::WeightedIndex::new(&prs)?;
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
    ) -> anyhow::Result<DecodingResult> {
        for (i, &t) in TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult, _> = self.decode(mel_segment, mel_time, t);
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
                    console_log!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn run(
        &mut self,
        mel: &[f32],
        n_mels: usize,
        total_frames: usize,
    ) -> anyhow::Result<Vec<Segment>> {
        let mut seek = 0;
        let mut segments = vec![];
        while seek < total_frames {
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
                console_log!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            console_log!("{seek}: {segment:?}");
            segments.push(segment)
        }
        Ok(segments)
    }

    pub fn load(md: ModelData) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_bytes(&md.tokenizer).map_err(E::msg)?;

        // Mel filters are stored as a flat `mel_80` f32 tensor in a
        // dedicated safetensors file. Wrap the bytes via the buffered
        // (non-mmap) entry point and convert to a `Vec<f32>` via the
        // lazy graph's `realize_f32`.
        let mel_st = BufferedSafetensors::new(md.mel_filters).map_err(|e| E::msg(e.to_string()))?;
        let mel_view = mel_st.get("mel_80").map_err(|e| E::msg(e.to_string()))?;
        let mel_filters = view_to_f32_vec(&mel_view)?;
        console_log!("loaded mel filters ({} elements)", mel_filters.len());

        let config: WhisperConfig = WhisperConfig::from_hf_json_str(
            std::str::from_utf8(&md.config).map_err(|e| E::msg(e.to_string()))?,
        )
        .map_err(|e| E::msg(format!("config: {e}")))?;

        let model = if md.quantized {
            // `QuantizedWhisperModel::from_gguf` currently stubs on
            // disk; there is no buffer constructor yet. Surface the
            // upstream error message to the worker caller.
            anyhow::bail!(
                "quantized whisper: lazy `QuantizedWhisperModel` has no buffer (GGUF \
                 in-memory) loader yet. Wire one in `fuel-core::lazy_quantized_whisper` \
                 to enable quantized models in the wasm worker."
            )
        } else {
            let st = BufferedSafetensors::new(md.weights).map_err(|e| E::msg(e.to_string()))?;
            let weights = load_whisper_weights(&st, &config)
                .map_err(|e| E::msg(format!("weights: {e}")))?;
            Model::Normal(WhisperModel {
                config: config.clone(),
                weights,
            })
        };
        console_log!("done loading model");

        let task = match md.task.as_deref() {
            Some("translate") => Some(Task::Translate),
            _ => Some(Task::Transcribe),
        };

        let decoder = Self::new(
            model,
            tokenizer,
            mel_filters,
            task,
            md.language,
            md.is_multilingual,
            md.timestamps,
        )?;
        Ok(decoder)
    }

    pub fn convert_and_run(&mut self, wav_input: &[u8]) -> anyhow::Result<Vec<Segment>> {
        let mut wav_input = std::io::Cursor::new(wav_input);
        let wav_reader = hound::WavReader::new(&mut wav_input)?;
        let spec = wav_reader.spec();
        console_log!("loaded wav data: {spec:?}");
        if spec.sample_rate != SAMPLE_RATE as u32 {
            anyhow::bail!("wav file must have a {} sampling rate", SAMPLE_RATE);
        }
        let mut data = wav_reader.into_samples::<i16>().collect::<Vec<_>>();
        data.truncate(data.len() / spec.channels as usize);
        let mut pcm_data = Vec::with_capacity(data.len());
        for d in data.into_iter() {
            let d = d?;
            pcm_data.push(d as f32 / 32768.)
        }
        console_log!("pcm data loaded {}", pcm_data.len());
        let n_mels = self.model.config().num_mel_bins;
        let mel = audio::pcm_to_mel(&pcm_data, &self.mel_filters, n_mels)
            .map_err(|e| E::msg(format!("pcm_to_mel: {e}")))?;
        let total_frames = mel.len() / n_mels;
        console_log!("loaded mel: [{}, {}]", n_mels, total_frames);
        let segments = self.run(&mel, n_mels, total_frames)?;
        Ok(segments)
    }
}

/// Detect the spoken language. Runs the encoder + a single decoder
/// step over the first `<= 2 * max_source_positions` frames of the
/// segment, takes the softmax over the language-token logits, and
/// returns the most likely language's token id.
pub fn detect_language(
    model: &Model,
    tokenizer: &Tokenizer,
    mel_segment: &[f32],
    mel_time: usize,
) -> anyhow::Result<u32> {
    console_log!("detecting language");
    let max_src = model.config().max_source_positions;
    // The encoder downsamples mel_time by 2; cap at 2 * max_src and
    // round down to an even value (stride-2 conv requirement).
    let cap = 2 * max_src;
    let n_mels = model.config().num_mel_bins;
    let mut seg_size = usize::min(mel_time, cap);
    if !seg_size.is_multiple_of(2) {
        seg_size -= 1;
    }
    if seg_size == 0 {
        anyhow::bail!("detect_language: empty mel segment");
    }
    // Extract the prefix segment from the row-major mel buffer.
    let mut mel_prefix = Vec::with_capacity(n_mels * seg_size);
    for m in 0..n_mels {
        let row = &mel_segment[m * mel_time..m * mel_time + seg_size];
        mel_prefix.extend_from_slice(row);
    }

    let encoder_out = model.encoder_forward(&mel_prefix, seg_size)?;
    let sot_token = token_id(tokenizer, SOT_TOKEN)?;
    let tokens = vec![sot_token];
    let logits_flat = model.decoder_logits(&tokens, &encoder_out, seg_size)?;
    let vocab = model.config().vocab_size;
    let last_off = (tokens.len() - 1) * vocab;
    let logits = &logits_flat[last_off..last_off + vocab];

    let language_token_ids = LANGUAGES
        .iter()
        .map(|(t, _)| token_id(tokenizer, &format!("<|{t}|>")))
        .collect::<fuel::Result<Vec<_>>>()
        .map_err(|e| anyhow::Error::msg(format!("language token id: {e}")))?;
    let picked: Vec<f32> = language_token_ids
        .iter()
        .map(|&i| logits[i as usize])
        .collect();
    let probs = softmax_vec(&picked);
    let mut probs_lang: Vec<((&str, &str), f32)> =
        LANGUAGES.iter().copied().zip(probs.into_iter()).collect();
    probs_lang.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for ((_, language), p) in probs_lang.iter().take(5) {
        console_log!("{language}: {p}")
    }
    let token = format!("<|{}|>", probs_lang[0].0 .0);
    let language = token_id(tokenizer, &token)?;
    console_log!("detected language: {language} {token}");
    Ok(language)
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> fuel::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => fuel::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum Task {
    Transcribe,
    Translate,
}

// Communication to the worker happens through bincode, the model weights and configs are fetched
// on the main thread and transferred via the following structure.
#[derive(Serialize, Deserialize)]
pub struct ModelData {
    pub weights: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub mel_filters: Vec<u8>,
    pub config: Vec<u8>,
    pub quantized: bool,
    pub timestamps: bool,
    pub is_multilingual: bool,
    pub language: Option<String>,
    pub task: Option<String>,
}

pub struct Worker {
    link: WorkerLink<Self>,
    decoder: Option<Decoder>,
}

#[derive(Serialize, Deserialize)]
pub enum WorkerInput {
    ModelData(ModelData),
    DecodeTask { wav_bytes: Vec<u8> },
}

#[derive(Serialize, Deserialize)]
pub enum WorkerOutput {
    Decoded(Vec<Segment>),
    WeightsLoaded,
}

impl yew_agent::Worker for Worker {
    type Input = WorkerInput;
    type Message = ();
    type Output = Result<WorkerOutput, String>;
    type Reach = Public<Self>;

    fn create(link: WorkerLink<Self>) -> Self {
        Self {
            link,
            decoder: None,
        }
    }

    fn update(&mut self, _msg: Self::Message) {
        // no messaging
    }

    fn handle_input(&mut self, msg: Self::Input, id: HandlerId) {
        let output = match msg {
            WorkerInput::ModelData(md) => match Decoder::load(md) {
                Ok(decoder) => {
                    self.decoder = Some(decoder);
                    Ok(WorkerOutput::WeightsLoaded)
                }
                Err(err) => Err(format!("model creation error {err:?}")),
            },
            WorkerInput::DecodeTask { wav_bytes } => match &mut self.decoder {
                None => Err("model has not been set".to_string()),
                Some(decoder) => decoder
                    .convert_and_run(&wav_bytes)
                    .map(WorkerOutput::Decoded)
                    .map_err(|e| e.to_string()),
            },
        };
        self.link.respond(id, output);
    }

    fn name_of_resource() -> &'static str {
        "worker.js"
    }

    fn resource_path_is_relative() -> bool {
        true
    }
}

// ---- Local helpers ---------------------------------------------------------

/// Build a `WhisperWeights` from a `BufferedSafetensors`. Mirrors the
/// `WhisperWeights::load_from_mmapped` loader in `fuel-core::lazy_whisper`,
/// but speaks `BufferedSafetensors` (owning `Vec<u8>`) so it works
/// under wasm32 where `memmap2`-backed `MmapedSafetensors` doesn't.
fn load_whisper_weights(
    st: &BufferedSafetensors,
    cfg: &WhisperConfig,
) -> anyhow::Result<WhisperWeights> {
    use fuel::lazy_whisper::{
        WhisperDecoderLayerWeights, WhisperDecoderWeights, WhisperEncoderLayerWeights,
        WhisperEncoderWeights,
    };

    let d = cfg.d_model;

    // --- encoder ------------------------------------------------
    let conv1_w = load_f32(st, "model.encoder.conv1.weight")?;
    let conv1_b = load_f32(st, "model.encoder.conv1.bias")?;
    let conv2_w = load_f32(st, "model.encoder.conv2.weight")?;
    let conv2_b = load_f32(st, "model.encoder.conv2.bias")?;
    let positional = load_f32(st, "model.encoder.embed_positions.weight")?;

    let mut enc_layers = Vec::with_capacity(cfg.encoder_layers);
    for i in 0..cfg.encoder_layers {
        let p = format!("model.encoder.layers.{i}");
        let self_attn_ln_g = load_f32(st, &format!("{p}.self_attn_layer_norm.weight"))?;
        let self_attn_ln_b = load_f32(st, &format!("{p}.self_attn_layer_norm.bias"))?;
        let q_w = load_transposed(st, &format!("{p}.self_attn.q_proj.weight"), d, d)?;
        let q_b = load_f32(st, &format!("{p}.self_attn.q_proj.bias"))?;
        let k_w = load_transposed(st, &format!("{p}.self_attn.k_proj.weight"), d, d)?;
        let v_w = load_transposed(st, &format!("{p}.self_attn.v_proj.weight"), d, d)?;
        let v_b = load_f32(st, &format!("{p}.self_attn.v_proj.bias"))?;
        let out_w = load_transposed(st, &format!("{p}.self_attn.out_proj.weight"), d, d)?;
        let out_b = load_f32(st, &format!("{p}.self_attn.out_proj.bias"))?;
        let final_ln_g = load_f32(st, &format!("{p}.final_layer_norm.weight"))?;
        let final_ln_b = load_f32(st, &format!("{p}.final_layer_norm.bias"))?;
        let fc1_w = load_transposed(st, &format!("{p}.fc1.weight"), cfg.encoder_ffn_dim, d)?;
        let fc1_b = load_f32(st, &format!("{p}.fc1.bias"))?;
        let fc2_w = load_transposed(st, &format!("{p}.fc2.weight"), d, cfg.encoder_ffn_dim)?;
        let fc2_b = load_f32(st, &format!("{p}.fc2.bias"))?;
        enc_layers.push(WhisperEncoderLayerWeights {
            self_attn_ln_g: Arc::from(self_attn_ln_g),
            self_attn_ln_b: Arc::from(self_attn_ln_b),
            q_w: Arc::from(q_w),
            q_b: Arc::from(q_b),
            k_w: Arc::from(k_w),
            v_w: Arc::from(v_w),
            v_b: Arc::from(v_b),
            out_w: Arc::from(out_w),
            out_b: Arc::from(out_b),
            final_ln_g: Arc::from(final_ln_g),
            final_ln_b: Arc::from(final_ln_b),
            fc1_w: Arc::from(fc1_w),
            fc1_b: Arc::from(fc1_b),
            fc2_w: Arc::from(fc2_w),
            fc2_b: Arc::from(fc2_b),
        });
    }
    let enc_final_ln_g = load_f32(st, "model.encoder.layer_norm.weight")?;
    let enc_final_ln_b = load_f32(st, "model.encoder.layer_norm.bias")?;

    // --- decoder ------------------------------------------------
    let dec_embed_tokens = load_f32(st, "model.decoder.embed_tokens.weight")?;
    let dec_embed_positions = load_f32(st, "model.decoder.embed_positions.weight")?;

    let mut dec_layers = Vec::with_capacity(cfg.decoder_layers);
    for i in 0..cfg.decoder_layers {
        let p = format!("model.decoder.layers.{i}");
        let self_ln_g = load_f32(st, &format!("{p}.self_attn_layer_norm.weight"))?;
        let self_ln_b = load_f32(st, &format!("{p}.self_attn_layer_norm.bias"))?;
        let self_q_w = load_transposed(st, &format!("{p}.self_attn.q_proj.weight"), d, d)?;
        let self_q_b = load_f32(st, &format!("{p}.self_attn.q_proj.bias"))?;
        let self_k_w = load_transposed(st, &format!("{p}.self_attn.k_proj.weight"), d, d)?;
        let self_v_w = load_transposed(st, &format!("{p}.self_attn.v_proj.weight"), d, d)?;
        let self_v_b = load_f32(st, &format!("{p}.self_attn.v_proj.bias"))?;
        let self_out_w = load_transposed(st, &format!("{p}.self_attn.out_proj.weight"), d, d)?;
        let self_out_b = load_f32(st, &format!("{p}.self_attn.out_proj.bias"))?;

        let cross_ln_g = load_f32(st, &format!("{p}.encoder_attn_layer_norm.weight"))?;
        let cross_ln_b = load_f32(st, &format!("{p}.encoder_attn_layer_norm.bias"))?;
        let cross_q_w = load_transposed(st, &format!("{p}.encoder_attn.q_proj.weight"), d, d)?;
        let cross_q_b = load_f32(st, &format!("{p}.encoder_attn.q_proj.bias"))?;
        let cross_k_w = load_transposed(st, &format!("{p}.encoder_attn.k_proj.weight"), d, d)?;
        let cross_v_w = load_transposed(st, &format!("{p}.encoder_attn.v_proj.weight"), d, d)?;
        let cross_v_b = load_f32(st, &format!("{p}.encoder_attn.v_proj.bias"))?;
        let cross_out_w = load_transposed(st, &format!("{p}.encoder_attn.out_proj.weight"), d, d)?;
        let cross_out_b = load_f32(st, &format!("{p}.encoder_attn.out_proj.bias"))?;

        let final_ln_g = load_f32(st, &format!("{p}.final_layer_norm.weight"))?;
        let final_ln_b = load_f32(st, &format!("{p}.final_layer_norm.bias"))?;
        let fc1_w = load_transposed(st, &format!("{p}.fc1.weight"), cfg.decoder_ffn_dim, d)?;
        let fc1_b = load_f32(st, &format!("{p}.fc1.bias"))?;
        let fc2_w = load_transposed(st, &format!("{p}.fc2.weight"), d, cfg.decoder_ffn_dim)?;
        let fc2_b = load_f32(st, &format!("{p}.fc2.bias"))?;

        dec_layers.push(WhisperDecoderLayerWeights {
            self_ln_g: Arc::from(self_ln_g),
            self_ln_b: Arc::from(self_ln_b),
            self_q_w: Arc::from(self_q_w),
            self_q_b: Arc::from(self_q_b),
            self_k_w: Arc::from(self_k_w),
            self_v_w: Arc::from(self_v_w),
            self_v_b: Arc::from(self_v_b),
            self_out_w: Arc::from(self_out_w),
            self_out_b: Arc::from(self_out_b),
            cross_ln_g: Arc::from(cross_ln_g),
            cross_ln_b: Arc::from(cross_ln_b),
            cross_q_w: Arc::from(cross_q_w),
            cross_q_b: Arc::from(cross_q_b),
            cross_k_w: Arc::from(cross_k_w),
            cross_v_w: Arc::from(cross_v_w),
            cross_v_b: Arc::from(cross_v_b),
            cross_out_w: Arc::from(cross_out_w),
            cross_out_b: Arc::from(cross_out_b),
            final_ln_g: Arc::from(final_ln_g),
            final_ln_b: Arc::from(final_ln_b),
            fc1_w: Arc::from(fc1_w),
            fc1_b: Arc::from(fc1_b),
            fc2_w: Arc::from(fc2_w),
            fc2_b: Arc::from(fc2_b),
        });
    }
    let dec_final_ln_g = load_f32(st, "model.decoder.layer_norm.weight")?;
    let dec_final_ln_b = load_f32(st, "model.decoder.layer_norm.bias")?;

    Ok(WhisperWeights {
        encoder: WhisperEncoderWeights {
            conv1_w: Arc::from(conv1_w),
            conv1_b: Arc::from(conv1_b),
            conv2_w: Arc::from(conv2_w),
            conv2_b: Arc::from(conv2_b),
            positional: Arc::from(positional),
            layers: enc_layers,
            final_ln_g: Arc::from(enc_final_ln_g),
            final_ln_b: Arc::from(enc_final_ln_b),
        },
        decoder: WhisperDecoderWeights {
            embed_tokens: Arc::from(dec_embed_tokens),
            embed_positions: Arc::from(dec_embed_positions),
            layers: dec_layers,
            final_ln_g: Arc::from(dec_final_ln_g),
            final_ln_b: Arc::from(dec_final_ln_b),
        },
    })
}

fn load_f32(st: &BufferedSafetensors, name: &str) -> anyhow::Result<Vec<f32>> {
    let view = st
        .get(name)
        .map_err(|e| anyhow::Error::msg(format!("load_f32 {name:?}: {e}")))?;
    view_to_f32_vec(&view)
}

fn view_to_f32_vec(view: &safetensors::tensor::TensorView<'_>) -> anyhow::Result<Vec<f32>> {
    use safetensors::Dtype;
    let bytes = view.data();
    match view.dtype() {
        Dtype::F32 => {
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        Dtype::F64 => {
            let mut out = Vec::with_capacity(bytes.len() / 8);
            for chunk in bytes.chunks_exact(8) {
                let arr: [u8; 8] = chunk.try_into().unwrap();
                out.push(f64::from_le_bytes(arr) as f32);
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::bf16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        Dtype::F16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::f16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        other => anyhow::bail!("view_to_f32_vec: unsupported dtype {other:?}"),
    }
}

fn load_transposed(
    st: &BufferedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> anyhow::Result<Vec<f32>> {
    let flat = load_f32(st, name)?;
    if flat.len() != out_features * in_features {
        anyhow::bail!(
            "load_transposed: tensor {name:?} has {} elements, expected {} ({out_features} × {in_features})",
            flat.len(),
            out_features * in_features,
        );
    }
    let mut out = vec![0.0_f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            out[j * out_features + i] = flat[i * in_features + j];
        }
    }
    Ok(out)
}

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

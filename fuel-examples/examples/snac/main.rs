#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use fuel::lazy::LazyTensor;
use fuel::lazy_snac::{SnacConfig, SnacModel, SnacWeights};
use fuel_core_types::Shape;
use hf_hub::api::sync::Api;
use serde::Deserialize;

mod audio_io;

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Action {
    AudioToCode,
    CodeToAudio,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "24khz")]
    S24khz,
    #[value(name = "32khz")]
    S32khz,
    #[value(name = "44khz")]
    S44khz,
}

impl Which {
    fn sample_rate(&self) -> u32 {
        match self {
            Which::S24khz => 24000,
            Which::S32khz => 32000,
            Which::S44khz => 44000,
        }
    }

    fn config_repo(&self) -> &'static str {
        match self {
            Which::S24khz => "hubertsiuzdak/snac_24khz",
            Which::S32khz => "hubertsiuzdak/snac_32khz",
            Which::S44khz => "hubertsiuzdak/snac_44khz",
        }
    }

    fn model_file(&self) -> &'static str {
        match self {
            Which::S24khz => "snac_24khz.safetensors",
            Which::S32khz => "snac_32khz.safetensors",
            Which::S44khz => "snac_44khz.safetensors",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The action to be performed (audio-to-code or code-to-audio).
    action: Action,

    /// The input file (audio file for audio-to-code, or codes safetensors for code-to-audio).
    in_file: String,

    /// The output file (codes safetensors for audio-to-code, or wave audio file for code-to-audio).
    out_file: String,

    /// The model size to use.
    #[arg(long, default_value = "24khz")]
    which: Which,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model weight file, in safetensor format.
    #[arg(long)]
    model: Option<String>,

    /// The config file, JSON.
    #[arg(long)]
    config: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HfSnacConfig {
    #[serde(default = "default_audio_channels")]
    audio_channels: usize,
    encoder_dim: usize,
    decoder_dim: usize,
    decoder_rates: Vec<usize>,
    #[serde(default)]
    attn_window_size: Option<usize>,
    codebook_size: usize,
    codebook_dim: usize,
    vq_strides: Vec<usize>,
    #[serde(default)]
    noise: bool,
    #[serde(default)]
    depthwise: bool,
}

fn default_audio_channels() -> usize {
    1
}

impl From<HfSnacConfig> for SnacConfig {
    fn from(c: HfSnacConfig) -> Self {
        SnacConfig {
            audio_channels: c.audio_channels,
            encoder_dim: c.encoder_dim,
            decoder_dim: c.decoder_dim,
            decoder_rates: c.decoder_rates,
            attn_window_size: c.attn_window_size,
            codebook_size: c.codebook_size,
            codebook_dim: c.codebook_dim,
            vq_strides: c.vq_strides,
            noise: c.noise,
            depthwise: c.depthwise,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _ = fuel_examples::device(args.cpu)?;
    let model_sample_rate = args.which.sample_rate();

    let config_path = match args.config {
        Some(c) => std::path::PathBuf::from(c),
        None => Api::new()?
            .model(args.which.config_repo().to_string())
            .get("config.json")?,
    };
    let cfg_json: HfSnacConfig = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let cfg: SnacConfig = cfg_json.into();

    let model_path = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => Api::new()?
            .model("lmz/fuel-snac".to_string())
            .get(args.which.model_file())?,
    };

    let st = unsafe { fuel::safetensors::MmapedSafetensors::new(&model_path) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = SnacWeights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let num_codebooks = weights.quantizers.len();
    let model = SnacModel {
        config: cfg.clone(),
        weights,
    };

    match args.action {
        Action::CodeToAudio => {
            // Load the codes from the safetensors file.
            let codes_st = unsafe {
                fuel::safetensors::MmapedSafetensors::new(&args.in_file)
            }
            .map_err(|e| E::msg(format!("mmap codes safetensors: {e}")))?;
            let mut codes: Vec<LazyTensor> = Vec::with_capacity(num_codebooks);
            for i in 0..num_codebooks {
                let name = format!("codes-{i}");
                let view = codes_st
                    .get(&name)
                    .map_err(|e| E::msg(format!("missing tensor {name}: {e}")))?;
                let shape: Vec<usize> = view.shape().to_vec();
                if shape.len() != 2 {
                    anyhow::bail!("codes-{i}: expected rank 2, got {:?}", shape);
                }
                // Decode as u32 codes. Safetensors stores as bytes — interpret accordingly.
                let raw = view.data();
                let dt = view.dtype();
                let u32_data: Vec<u32> = match dt {
                    safetensors::Dtype::U32 => raw
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect(),
                    safetensors::Dtype::I64 => raw
                        .chunks_exact(8)
                        .map(|b| {
                            i64::from_le_bytes([
                                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                            ]) as u32
                        })
                        .collect(),
                    other => anyhow::bail!("codes-{i}: unexpected dtype {other:?}"),
                };
                // Anchor onto the model graph via a temporary LazyTensor.
                let anchor = LazyTensor::from_f32(
                    vec![0.0_f32],
                    Shape::from_dims(&[1]),
                    &fuel::Device::Cpu,
                );
                let lt = anchor.const_u32_like(u32_data, Shape::from_dims(&shape));
                codes.push(lt);
                println!("codes-{i} shape: {:?}", shape);
            }
            let pcm = model
                .decode_codes(&codes)
                .map_err(|e| E::msg(format!("decode: {e}")))?;
            println!("output pcm shape: {:?}", pcm.shape().dims());
            let pcm_data = pcm.realize_f32();
            // Output shape is (1, audio_channels, T). Extract first batch, first channel.
            let total = pcm_data.len();
            let t = total / (cfg.audio_channels.max(1));
            let pcm_ch0: Vec<f32> = pcm_data[..t].to_vec();
            let pcm_norm = normalize_loudness_f32(&pcm_ch0, model_sample_rate, true);

            let mut output = std::fs::File::create(&args.out_file)?;
            fuel_examples::wav::write_pcm_as_wav(&mut output, &pcm_norm, model_sample_rate)?;
            println!("wrote {} samples to {}", pcm_norm.len(), args.out_file);
        }
        Action::AudioToCode => {
            // Encode path is not yet supported on lazy_snac (decode-only v1).
            // We still allow file loading to keep the binary usable end-to-end
            // for the decode workflow.
            anyhow::bail!(
                "AudioToCode is not yet implemented on the lazy SNAC port \
                 (lazy_snac v1 is decode-only). \
                 Use CodeToAudio with an existing codes safetensors file."
            );
        }
    }

    Ok(())
}

/// Simplified normalize_loudness on a Vec<f32>, mirroring
/// fuel_examples::audio::normalize_loudness without the Tensor dependency.
fn normalize_loudness_f32(wav: &[f32], sample_rate: u32, loudness_compressor: bool) -> Vec<f32> {
    if wav.is_empty() {
        return Vec::new();
    }
    let n = wav.len();
    let energy = (wav.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
    if energy < 2e-3 {
        return wav.to_vec();
    }
    let mut meter = fuel_examples::bs1770::ChannelLoudnessMeter::new(sample_rate);
    meter.push(wav.iter().copied());
    let power = meter.as_100ms_windows();
    let loudness = match fuel_examples::bs1770::gated_mean(power) {
        None => return wav.to_vec(),
        Some(gp) => gp.loudness_lkfs() as f64,
    };
    let delta_loudness = -14.0 - loudness;
    let gain = 10f64.powf(delta_loudness / 20.0) as f32;
    let out: Vec<f32> = wav.iter().map(|x| x * gain).collect();
    if loudness_compressor {
        out.into_iter().map(|x| x.tanh()).collect()
    } else {
        out
    }
}

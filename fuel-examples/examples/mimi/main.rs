#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use fuel::lazy::LazyTensor;
use fuel::lazy_mimi::{MimiConfig, MimiModel, MimiWeights};
use fuel_ir::Shape;
use hf_hub::api::sync::Api;

mod audio_io;

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Action {
    AudioToAudio,
    AudioToCode,
    CodeToAudio,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The action to be performed, specifies the format for the input and output data.
    action: Action,

    /// The input file, either an audio file or some mimi tokens stored as safetensors.
    in_file: String,

    /// The output file, either a wave audio file or some mimi tokens stored as safetensors.
    out_file: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model weight file, in safetensor format.
    #[arg(long)]
    model: Option<String>,

    /// Whether to use streaming or not, when streaming slices of data of the given size are passed
    /// to the encoder/decoder one at a time.
    ///
    /// NOTE: streaming is not supported by the lazy Mimi port (no `decode_step` / `reset_state`
    /// in lazy_mimi v1). Passing this flag results in an error.
    #[arg(long)]
    streaming: Option<usize>,
}

/// Mimi v0.1 resampler stride: `encoder_frame_rate / frame_rate`
/// = (sample_rate / prod(ratios)) / frame_rate
/// = (24000 / (8*6*5*4)) / 12.5 = 25 / 12.5 = 2.
const MIMI_V0_1_RESAMPLER_STRIDE: usize = 2;

fn main() -> Result<()> {
    let args = Args::parse();
    let _ = fuel_examples::device(args.cpu)?;

    if args.streaming.is_some() {
        anyhow::bail!(
            "streaming is not yet implemented on the lazy Mimi port \
             (lazy_mimi v1 is single-call encode/decode only). \
             Re-run without --streaming."
        );
    }

    let model_path = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => Api::new()?
            .model("kyutai/mimi".to_string())
            .get("model.safetensors")?,
    };

    let config = MimiConfig::v0_1(None);
    let st = unsafe { fuel::safetensors::MmapedSafetensors::new(&model_path) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = MimiWeights::load_from_mmapped(&st, &config, MIMI_V0_1_RESAMPLER_STRIDE)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = MimiModel {
        config: config.clone(),
        weights,
    };

    // Returned as an owned `Vec<u32>` of codes shaped (1, n_q, T_codes).
    let (codes_data, codes_shape): (Vec<u32>, Vec<usize>) = match args.action {
        Action::CodeToAudio => {
            // Load the codes from a safetensors file (saved by AudioToCode).
            let codes_st = unsafe {
                fuel::safetensors::MmapedSafetensors::new(&args.in_file)
            }
            .map_err(|e| E::msg(format!("mmap codes safetensors: {e}")))?;
            let view = codes_st
                .get("codes")
                .map_err(|e| E::msg(format!("missing tensor `codes`: {e}")))?;
            let shape: Vec<usize> = view.shape().to_vec();
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
                other => anyhow::bail!("codes: unexpected dtype {other:?}"),
            };
            (u32_data, shape)
        }
        Action::AudioToCode | Action::AudioToAudio => {
            let pcm = if args.in_file == "-" {
                println!(">>>> RECORDING AUDIO, PRESS ENTER ONCE DONE <<<<");
                let (stream, input_audio) = audio_io::setup_input_stream()?;
                let mut pcms = vec![];
                let stdin = std::thread::spawn(|| {
                    let mut s = String::new();
                    std::io::stdin().read_line(&mut s)
                });
                while !stdin.is_finished() {
                    let input = input_audio.lock().unwrap().take_all();
                    if input.is_empty() {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        continue;
                    }
                    pcms.push(input)
                }
                drop(stream);
                pcms.concat()
            } else {
                let (pcm, sample_rate) = audio_io::pcm_decode(args.in_file)?;
                if sample_rate != 24_000 {
                    println!("WARNING: mimi uses a 24khz sample rate, input uses {sample_rate}, resampling...");
                    audio_io::resample(&pcm, sample_rate as usize, 24_000)?
                } else {
                    pcm
                }
            };
            let pcm_len = pcm.len();
            let pcm_lt = LazyTensor::from_f32(
                pcm,
                Shape::from_dims(&[1, 1, pcm_len]),
                &fuel::Device::cpu(),
            );
            println!("input pcm shape: {:?}", pcm_lt.shape().dims());
            let codes = model
                .encode(&pcm_lt)
                .map_err(|e| E::msg(format!("encode: {e}")))?;
            let shape: Vec<usize> = codes.shape().dims().to_vec();
            let data = codes.realize_u32();
            (data, shape)
        }
    };
    println!("codes shape: {:?}", codes_shape);

    match args.action {
        Action::AudioToCode => {
            // Save the u32 codes back into a single-tensor safetensors file under name "codes".
            use safetensors::tensor::TensorView;
            use std::collections::HashMap;
            let bytes: Vec<u8> = codes_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let view = TensorView::new(
                safetensors::Dtype::U32,
                codes_shape.clone(),
                &bytes,
            )
            .map_err(|e| E::msg(format!("TensorView::new: {e}")))?;
            let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
            views.insert("codes".to_string(), view);
            let bytes_out = safetensors::serialize(&views, None)
                .map_err(|e| E::msg(format!("safetensors::serialize: {e}")))?;
            std::fs::write(&args.out_file, bytes_out)?;
            println!("wrote codes to {}", args.out_file);
        }
        Action::AudioToAudio | Action::CodeToAudio => {
            // Rehydrate the codes as a LazyTensor on a fresh graph so the decode
            // call has somewhere to anchor.
            let anchor = LazyTensor::from_f32(
                vec![0.0_f32],
                Shape::from_dims(&[1]),
                &fuel::Device::cpu(),
            );
            let codes_lt =
                anchor.const_u32_like(codes_data, Shape::from_dims(&codes_shape));
            let pcm = model
                .decode(&codes_lt)
                .map_err(|e| E::msg(format!("decode: {e}")))?;
            println!("output pcm shape: {:?}", pcm.shape().dims());
            let pcm_dims = pcm.shape().dims().to_vec();
            // Expect (1, channels, T_audio). Extract first batch / first channel.
            if pcm_dims.len() != 3 {
                anyhow::bail!("decoded pcm: expected rank-3, got {:?}", pcm_dims);
            }
            let channels = pcm_dims[1];
            let t = pcm_dims[2];
            let pcm_data = pcm.realize_f32();
            if channels == 0 || pcm_data.len() < t {
                anyhow::bail!(
                    "decoded pcm: empty/short buffer (channels={channels}, t={t}, total={})",
                    pcm_data.len(),
                );
            }
            // First batch, first channel: pcm_data[0..t].
            let pcm_ch0: Vec<f32> = pcm_data[..t].to_vec();
            let pcm_norm = normalize_loudness_f32(&pcm_ch0, 24_000, true);
            if args.out_file == "-" {
                let (stream, ad) = audio_io::setup_output_stream()?;
                {
                    let mut ad = ad.lock().unwrap();
                    ad.push_samples(&pcm_norm)?;
                }
                loop {
                    let ad = ad.lock().unwrap();
                    if ad.is_empty() {
                        break;
                    }
                    // That's very weird, calling thread::sleep here triggers the stream to stop
                    // playing (the callback doesn't seem to be called anymore).
                    // std::thread::sleep(std::time::Duration::from_millis(100));
                }
                drop(stream)
            } else {
                let mut output = std::fs::File::create(&args.out_file)?;
                fuel_examples::wav::write_pcm_as_wav(&mut output, &pcm_norm, 24_000)?;
            }
        }
    }
    Ok(())
}

/// Simplified normalize_loudness on a Vec<f32>, mirroring
/// fuel_examples::audio::normalize_loudness without the eager Tensor dependency.
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

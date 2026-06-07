//! EnCodec — lazy port.
//!
//! Wraps `fuel::lazy_encodec::EncodecModel`, which today implements the
//! decoder side only (`decode_codes`). The eager binary supported three
//! actions:
//!
//!   * `AudioToCode`   — encode PCM → discrete codes.
//!   * `AudioToAudio`  — encode then decode (round-trip).
//!   * `CodeToAudio`   — decode discrete codes → PCM.
//!
//! Only `CodeToAudio` is wired through the lazy port; the two encode
//! paths bail with a clear "lazy encoder not yet ported" error so the
//! binary still builds and the CLI shape is preserved.

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api;

use fuel::lazy::LazyTensor;
use fuel::lazy_encodec::{EncodecConfig, EncodecModel, EncodecWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};

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

    /// The input file, either an audio file or some encodec tokens stored as safetensors.
    in_file: String,

    /// The output file, either a wave audio file or some encodec tokens stored as safetensors.
    out_file: String,

    /// Run on CPU rather than on GPU. The lazy port realizes through the
    /// default router today; this flag is preserved for CLI parity with
    /// the eager binary but has no effect.
    #[arg(long)]
    cpu: bool,

    /// The model weight file, in safetensor format.
    #[arg(long)]
    model: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _ = args.cpu;

    // 1) Resolve + mmap the encodec weights.
    let model_path = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => Api::new()?
            .model("facebook/encodec_24khz".to_string())
            .get("model.safetensors")?,
    };

    let cfg = EncodecConfig::default_preset();
    // facebook/encodec_24khz: 24 kHz audio, 5 supported bandwidths.
    let sampling_rate: usize = 24_000;
    let target_bandwidths: Vec<f64> = vec![1.5, 3.0, 6.0, 12.0, 24.0];

    let st = unsafe { MmapedSafetensors::multi(&[model_path]) }
        .map_err(|e| E::msg(format!("mmap encodec safetensors: {e}")))?;
    let weights = EncodecWeights::load_from_mmapped(&st, &cfg, sampling_rate, &target_bandwidths)
        .map_err(|e| E::msg(format!("load encodec weights: {e}")))?;
    let model = EncodecModel {
        config: cfg.clone(),
        weights,
    };

    match args.action {
        Action::AudioToCode | Action::AudioToAudio => {
            anyhow::bail!(
                "lazy_encodec does not yet implement the encoder; \
                 only the `code-to-audio` action is supported by this binary today",
            );
        }
        Action::CodeToAudio => {
            // 2) Load the codes safetensors file. We don't go through
            //    the eager `fuel::safetensors::load` helper (which
            //    returns eager `Tensor`s); instead we mmap the file
            //    ourselves and read the raw u32 bytes for the `codes`
            //    tensor, then materialise it as a LazyTensor.
            let codes_st = unsafe { MmapedSafetensors::new(&args.in_file) }
                .map_err(|e| E::msg(format!("mmap codes safetensors {}: {e}", args.in_file)))?;
            let codes_view = codes_st
                .get("codes")
                .map_err(|e| E::msg(format!("no `codes` tensor in {}: {e}", args.in_file)))?;
            let codes_shape: Vec<usize> = codes_view.shape().to_vec();
            if codes_shape.len() != 3 {
                anyhow::bail!(
                    "codes tensor must be rank 3 [1, num_codebooks, T], got {:?}",
                    codes_shape,
                );
            }
            // Decode u32 little-endian bytes.
            let bytes = codes_view.data();
            if bytes.len() % 4 != 0 {
                anyhow::bail!(
                    "codes tensor byte length {} not a multiple of 4 (expected u32)",
                    bytes.len(),
                );
            }
            let mut codes_data: Vec<u32> = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                codes_data.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }

            println!("codes shape: {:?}", codes_shape);

            // 3) Anchor + materialise the codes as a LazyTensor on CPU.
            //    `const_u32_like` reuses the anchor's device; the
            //    anchor is a tiny f32 scalar on CPU.
            let anchor = LazyTensor::from_f32(
                vec![0.0_f32; 1],
                Shape::from_dims(&[1]),
                &Device::cpu(),
            );
            let codes = anchor.const_u32_like(codes_data, Shape::from_dims(&codes_shape));

            // 4) Run the decoder.
            let pcm_lazy = model
                .decode_codes(&codes)
                .map_err(|e| E::msg(format!("decode_codes: {e}")))?;
            let pcm_shape_holder = pcm_lazy.shape();
            let pcm_shape: Vec<usize> = pcm_shape_holder.dims().to_vec();
            println!("output pcm shape: {:?}", pcm_shape);
            if pcm_shape.len() != 3 || pcm_shape[0] != 1 || pcm_shape[1] != 1 {
                anyhow::bail!(
                    "decoder output must be [1, 1, T], got {:?}",
                    pcm_shape,
                );
            }
            let pcm_flat: Vec<f32> = pcm_lazy.realize_f32();
            // The shape is (1, 1, T); the flat row-major layout already
            // matches the T-long mono channel.
            let pcm = normalize_loudness_host(&pcm_flat, 24_000, true);

            // 5) Write to file or live audio.
            if args.out_file == "-" {
                let (stream, ad) = audio_io::setup_output_stream()?;
                {
                    let mut ad = ad.lock().unwrap();
                    ad.push_samples(&pcm)?;
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
                fuel_examples::wav::write_pcm_as_wav(&mut output, &pcm, 24_000)?;
            }
        }
    }

    Ok(())
}

/// Host-side mirror of `fuel_examples::audio::normalize_loudness`. The
/// eager helper requires an eager `Tensor`; the lazy decode path has
/// already realized to a `Vec<f32>`, so we operate directly on the host
/// slice using the same BS.1770 meter.
///
/// Logic mirrors:
///   <https://github.com/facebookresearch/audiocraft/blob/69fea8b290ad1b4b40d28f92d1dfc0ab01dbab85/audiocraft/data/audio_utils.py#L57>
fn normalize_loudness_host(wav: &[f32], sample_rate: u32, loudness_compressor: bool) -> Vec<f32> {
    use fuel_examples::bs1770;

    if wav.is_empty() {
        return Vec::new();
    }

    // RMS energy.
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

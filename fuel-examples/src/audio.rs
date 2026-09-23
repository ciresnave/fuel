// SPDX-License-Identifier: MIT OR Apache-2.0
use fuel::Result;

// https://github.com/facebookresearch/audiocraft/blob/69fea8b290ad1b4b40d28f92d1dfc0ab01dbab85/audiocraft/data/audio_utils.py#L57
//
// B6 note: this took and returned an eager `Tensor`, but every step is
// per-sample host arithmetic over a mono waveform — and it already round-tripped
// through `to_vec1::<f32>()` in the middle to feed the loudness meter. It now
// works on `&[f32]` directly.
pub fn normalize_loudness(
    wav: &[f32],
    sample_rate: u32,
    loudness_compressor: bool,
) -> Result<Vec<f32>> {
    let energy = (wav.iter().map(|v| v * v).sum::<f32>() / wav.len().max(1) as f32).sqrt();
    if energy < 2e-3 {
        return Ok(wav.to_vec());
    }
    let mut meter = crate::bs1770::ChannelLoudnessMeter::new(sample_rate);
    meter.push(wav.iter().copied());
    let power = meter.as_100ms_windows();
    let loudness = match crate::bs1770::gated_mean(power) {
        None => return Ok(wav.to_vec()),
        Some(gp) => gp.loudness_lkfs() as f64,
    };
    let delta_loudness = -14. - loudness;
    let gain = 10f64.powf(delta_loudness / 20.) as f32;
    Ok(wav
        .iter()
        .map(|v| {
            let v = v * gain;
            if loudness_compressor { v.tanh() } else { v }
        })
        .collect())
}

// Ported to the symphonia 0.6 API (GAP-338): `AudioBufferRef`/`Signal` were
// replaced by `GenericAudioBufferRef` + the `Audio` trait, `CODEC_TYPE_NULL`/
// `DecoderOptions` by `AudioCodecId`-less audio params + `AudioDecoderOptions`,
// `Probe::format` by `Probe::probe` (returns the reader directly, no
// intermediate `ProbedMetadata`), `Track::codec_params` is now
// `Option<CodecParameters>` (an enum over audio/video/subtitle), and
// `FormatReader::next_packet` now returns `Ok(None)` at end-of-stream instead
// of an `Err`. `Packet::track_id` is a field, not a method.
#[cfg(feature = "symphonia")]
pub fn pcm_decode<P: AsRef<std::path::Path>>(path: P) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::conv::FromSample;
    use symphonia::core::audio::sample::Sample;
    use symphonia::core::audio::{Audio, AudioBuffer, GenericAudioBufferRef};
    use symphonia::core::codecs::audio::AudioDecoderOptions;

    fn conv<T>(samples: &mut Vec<f32>, buf: &AudioBuffer<T>)
    where
        T: Sample,
        f32: FromSample<T>,
    {
        if let Some(chan0) = buf.plane(0) {
            samples.extend(chan0.iter().map(|v| f32::from_sample(*v)))
        }
    }

    // Open the media source.
    let src = std::fs::File::open(path).map_err(fuel::Error::wrap)?;

    // Create the media source stream.
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());

    // Create a probe hint using the file's extension. [Optional]
    let hint = symphonia::core::formats::probe::Hint::new();

    // Use the default options for metadata and format readers.
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();

    // Probe the media source and get the instantiated format reader directly.
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, fmt_opts, meta_opts)
        .map_err(fuel::Error::wrap)?;

    // Find the first track with decodable audio codec parameters.
    let (track_id, audio_params) = format
        .tracks()
        .iter()
        .find_map(|t| Some((t.id, t.codec_params.as_ref()?.audio()?.clone())))
        .ok_or_else(|| fuel::Error::Msg("no supported audio tracks".to_string()))?;

    // Use the default options for the decoder.
    let dec_opts: AudioDecoderOptions = Default::default();

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&audio_params, &dec_opts)
        .map_err(|_| fuel::Error::Msg("unsupported codec".to_string()))?;
    let sample_rate = audio_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();
    // The decode loop. `next_packet` returns `Ok(None)` at end-of-stream.
    while let Some(packet) = format.next_packet().map_err(fuel::Error::wrap)? {
        // Consume any new metadata that has been read since the last packet.
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet).map_err(fuel::Error::wrap)? {
            GenericAudioBufferRef::F32(buf) => {
                if let Some(chan0) = buf.plane(0) {
                    pcm_data.extend(chan0);
                }
            }
            GenericAudioBufferRef::U8(buf) => conv(&mut pcm_data, buf),
            GenericAudioBufferRef::U16(buf) => conv(&mut pcm_data, buf),
            GenericAudioBufferRef::U24(buf) => conv(&mut pcm_data, buf),
            GenericAudioBufferRef::U32(buf) => conv(&mut pcm_data, buf),
            GenericAudioBufferRef::S8(buf) => conv(&mut pcm_data, buf),
            GenericAudioBufferRef::S16(buf) => conv(&mut pcm_data, buf),
            GenericAudioBufferRef::S24(buf) => conv(&mut pcm_data, buf),
            GenericAudioBufferRef::S32(buf) => conv(&mut pcm_data, buf),
            GenericAudioBufferRef::F64(buf) => conv(&mut pcm_data, buf),
        }
    }
    Ok((pcm_data, sample_rate))
}

#[cfg(all(test, feature = "symphonia"))]
mod tests {
    use super::pcm_decode;

    // Builds a minimal canonical PCM WAV file (mono, 16-bit) in memory: no
    // library dependency, just the RIFF/fmt/data chunks symphonia's WAV
    // demuxer parses.
    fn write_wav_i16(path: &std::path::Path, sample_rate: u32, samples: &[i16]) {
        let data_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let block_align: u16 = 2; // mono, 16-bit
        let byte_rate = sample_rate * block_align as u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&(data_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&data_bytes);
        std::fs::write(path, buf).unwrap();
    }

    #[test]
    fn pcm_decode_round_trips_a_synthetic_wav() {
        let samples: Vec<i16> = vec![0, 16384, -16384, 32767, -32768, 100, -100, 0];
        let dir = std::env::temp_dir();
        let path = dir.join(format!("fuel-pcm-decode-test-{}.wav", std::process::id()));
        write_wav_i16(&path, 16_000, &samples);

        let (pcm, sample_rate) = pcm_decode(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(sample_rate, 16_000);
        assert_eq!(pcm.len(), samples.len());
        for (decoded, original) in pcm.iter().zip(samples.iter()) {
            let expected = *original as f32 / 32768.0;
            assert!(
                (decoded - expected).abs() < 1e-4,
                "decoded {decoded} vs expected {expected}"
            );
        }
    }
}

//! Whisper audio preprocessing — pure host-side STFT + log-mel.
//!
//! Ports `fuel_transformers::models::audio::whisper::audio` to a
//! single-threaded, f32-only, lazy-graph-free utility. The output of
//! [`pcm_to_mel`] is a flat `Vec<f32>` of shape `(n_mels, n_frames)`
//! that the caller wraps with `LazyTensor::from_f32` before feeding
//! the Whisper encoder.
//!
//! The mel filterbank is **not** embedded here: callers pass it as a
//! `(n_mels, n_fft/2 + 1)` row-major slice. The eager
//! `fuel-transformers` crate keeps the 80- and 128-band filterbank
//! blobs alongside the Whisper model so audio preprocessing stays a
//! leaf utility without data-blob dependencies in `fuel-core`.
//!
//! Constants follow OpenAI Whisper / whisper.cpp:
//! `SAMPLE_RATE = 16000`, `N_FFT = 400`, `HOP_LENGTH = 160`,
//! `CHUNK_LENGTH = 30` seconds → `N_SAMPLES = 480000`,
//! `N_FRAMES = N_SAMPLES / HOP_LENGTH + 1 = 3001`.

use crate::Result;

pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_LENGTH: usize = 30;
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE;

/// Symmetric Hann window of length `n`:
/// `w[i] = 0.5 * (1 - cos(2*pi*i / n))`.
pub fn hann_window(n: usize) -> Vec<f32> {
    let n_t = n as f32;
    let two_pi = 2.0 * std::f32::consts::PI;
    (0..n).map(|i| 0.5 * (1.0 - (two_pi * (i as f32) / n_t).cos())).collect()
}

fn dft(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    let two_pi = 2.0 * std::f32::consts::PI;
    let n_t = n as f32;
    let mut out = Vec::with_capacity(2 * n);
    for k in 0..n {
        let k_t = k as f32;
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (j, &x) in inp.iter().enumerate() {
            let angle = two_pi * k_t * (j as f32) / n_t;
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        out.push(re);
        out.push(im);
    }
    out
}

fn fft(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    if n == 1 {
        return vec![inp[0], 0.0];
    }
    if n % 2 == 1 {
        return dft(inp);
    }
    let mut out = vec![0.0f32; n * 2];
    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);
    for (i, &x) in inp.iter().enumerate() {
        if i % 2 == 0 {
            even.push(x);
        } else {
            odd.push(x);
        }
    }
    let even_fft = fft(&even);
    let odd_fft = fft(&odd);
    let two_pi = 2.0 * std::f32::consts::PI;
    let n_t = n as f32;
    for k in 0..n / 2 {
        let theta = two_pi * (k as f32) / n_t;
        let re = theta.cos();
        let im = -theta.sin();
        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];
        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + n / 2)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
    out
}

/// STFT producing the one-sided power spectrum for each frame.
///
/// `samples` is the time-domain signal (typically pre-padded to a
/// fixed length by the caller). `n_fft` is the window / FFT size and
/// `hop` is the step between consecutive frames. The window is a
/// Hann window of length `n_fft`.
///
/// The output is `n_frames = samples.len() / hop + 1` frames, each
/// `n_fft / 2 + 1` non-negative power bins. Power is computed as
/// `|X[k]|^2 + |X[N-k]|^2` (one-sided, folded) for k in
/// `1..n_fft/2`, matching the eager whisper.cpp pipeline.
pub fn stft(samples: &[f32], n_fft: usize, hop: usize) -> Vec<Vec<f32>> {
    let hann = hann_window(n_fft);
    let n_bins = n_fft / 2 + 1;
    let n_samples = samples.len();
    let n_frames = n_samples / hop + 1;
    let mut frames = Vec::with_capacity(n_frames);
    let mut fft_in = vec![0.0f32; n_fft];

    for i in 0..n_frames {
        let offset = i * hop;
        if offset >= n_samples {
            frames.push(vec![0.0f32; n_bins]);
            continue;
        }
        let take = std::cmp::min(n_fft, n_samples - offset);
        for j in 0..take {
            fft_in[j] = hann[j] * samples[offset + j];
        }
        for j in take..n_fft {
            fft_in[j] = 0.0;
        }

        let fft_out = fft(&fft_in);

        let mut power = vec![0.0f32; n_fft];
        for j in 0..n_fft {
            let re = fft_out[2 * j];
            let im = fft_out[2 * j + 1];
            power[j] = re * re + im * im;
        }
        for j in 1..n_fft / 2 {
            power[j] += power[n_fft - j];
        }

        let mut bins = Vec::with_capacity(n_bins);
        bins.extend_from_slice(&power[..n_bins]);
        frames.push(bins);
    }

    frames
}

/// Apply a mel filterbank to a power spectrogram and produce the
/// log-mel features.
///
/// `power_spec` is a `[n_frames][n_fft/2 + 1]` slice of power bins
/// from [`stft`]. `mel_filters` is `(n_mels, n_fft/2 + 1)` row-major.
///
/// Returns a flat `(n_mels, n_frames)` row-major vector:
/// `out[m * n_frames + f]`. The post-processing matches eager
/// whisper.cpp:
///   1. `mel = log10(max(filterbank @ power, 1e-10))`,
///   2. `mel = max(mel, mel.max() - 8.0)`,
///   3. `mel = (mel + 4) / 4`.
pub fn log_mel(
    power_spec: &[Vec<f32>],
    mel_filters: &[f32],
    n_mels: usize,
    n_fft: usize,
) -> Result<Vec<f32>> {
    let n_bins = n_fft / 2 + 1;
    if mel_filters.len() != n_mels * n_bins {
        crate::bail!(
            "log_mel: mel_filters has {} entries, expected n_mels({}) * (n_fft/2+1)({}) = {}",
            mel_filters.len(),
            n_mels,
            n_bins,
            n_mels * n_bins,
        );
    }
    let n_frames = power_spec.len();
    let mut mel = vec![0.0f32; n_mels * n_frames];

    for (f, frame) in power_spec.iter().enumerate() {
        if frame.len() != n_bins {
            crate::bail!(
                "log_mel: frame {} has {} bins, expected {}",
                f,
                frame.len(),
                n_bins,
            );
        }
        for m in 0..n_mels {
            let filt = &mel_filters[m * n_bins..(m + 1) * n_bins];
            let mut sum = 0.0f32;
            for k in 0..n_bins {
                sum += frame[k] * filt[k];
            }
            mel[m * n_frames + f] = sum.max(1e-10).log10();
        }
    }

    let mmax = mel
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| if a > b { a } else { b });
    let floor = mmax - 8.0;
    for v in mel.iter_mut() {
        let clamped = if *v > floor { *v } else { floor };
        *v = (clamped + 4.0) / 4.0;
    }

    Ok(mel)
}

/// Top-level Whisper audio preprocessing.
///
/// Pads or truncates `samples` to exactly `N_SAMPLES` (30 seconds at
/// 16 kHz = 480000 samples), then runs [`stft`] with
/// `N_FFT = 400` / `HOP_LENGTH = 160` and [`log_mel`] with the
/// supplied filterbank. Returns a flat `Vec<f32>` of shape
/// `(n_mels, N_FRAMES)` where `N_FRAMES = N_SAMPLES / HOP_LENGTH + 1
/// = 3001`.
pub fn pcm_to_mel(
    samples: &[f32],
    mel_filters: &[f32],
    n_mels: usize,
) -> Result<Vec<f32>> {
    let mut padded = Vec::with_capacity(N_SAMPLES);
    if samples.len() >= N_SAMPLES {
        padded.extend_from_slice(&samples[..N_SAMPLES]);
    } else {
        padded.extend_from_slice(samples);
        padded.resize(N_SAMPLES, 0.0);
    }
    let spec = stft(&padded, N_FFT, HOP_LENGTH);
    log_mel(&spec, mel_filters, n_mels, N_FFT)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_mel_filters(n_mels: usize, n_bins: usize) -> Vec<f32> {
        let mut filt = vec![0.0f32; n_mels * n_bins];
        for m in 0..n_mels {
            let lo = (m * n_bins) / n_mels;
            let hi = ((m + 2) * n_bins).div_ceil(n_mels).min(n_bins);
            for k in lo..hi {
                filt[m * n_bins + k] = 1.0;
            }
        }
        filt
    }

    #[test]
    fn hann_window_endpoints_and_peak() {
        let w = hann_window(400);
        assert_eq!(w.len(), 400);
        assert!(w[0].abs() < 1e-6);
        let mid = w[200];
        assert!((mid - 1.0).abs() < 1e-6, "hann peak at center = {}", mid);
    }

    #[test]
    fn sine_wave_pcm_to_mel_finite() {
        let n_mels = 80;
        let n_bins = N_FFT / 2 + 1;
        let filt = fake_mel_filters(n_mels, n_bins);
        let sr = SAMPLE_RATE as f32;
        let freq = 440.0f32;
        let two_pi = 2.0 * std::f32::consts::PI;
        let samples: Vec<f32> = (0..SAMPLE_RATE)
            .map(|i| (two_pi * freq * (i as f32) / sr).sin())
            .collect();
        let mel = pcm_to_mel(&samples, &filt, n_mels).unwrap();
        assert_eq!(mel.len(), n_mels * 3001);
        for (i, v) in mel.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at index {}: {}", i, v);
        }
        let mx = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mn = mel.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(mx > mn, "mel output is constant ({} == {})", mx, mn);
    }

    #[test]
    fn shape_for_30s_input() {
        let n_mels = 80;
        let n_bins = N_FFT / 2 + 1;
        let filt = fake_mel_filters(n_mels, n_bins);
        let samples = vec![0.0f32; N_SAMPLES];
        let mel = pcm_to_mel(&samples, &filt, n_mels).unwrap();
        assert_eq!(mel.len(), n_mels * 3001, "expected (n_mels=80, n_frames=3001)");
    }

    #[test]
    fn silence_clamps_to_floor() {
        let n_mels = 80;
        let n_bins = N_FFT / 2 + 1;
        let filt = fake_mel_filters(n_mels, n_bins);
        let samples = vec![0.0f32; N_SAMPLES];
        let mel = pcm_to_mel(&samples, &filt, n_mels).unwrap();
        for (i, v) in mel.iter().enumerate() {
            assert!(v.is_finite(), "silence produced non-finite at {}: {}", i, v);
        }
        let first = mel[0];
        for v in mel.iter() {
            assert!((*v - first).abs() < 1e-5, "silence not uniform: {} vs {}", v, first);
        }
        assert!((first - (-1.5)).abs() < 1e-5, "silence floor expected -1.5, got {}", first);
    }
}

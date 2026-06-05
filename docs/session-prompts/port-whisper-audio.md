# Port: Whisper audio preprocessing (STFT + log-mel)

## Eager source

- `fuel-transformers/src/models/audio/whisper/audio.rs` (338 LOC)
  — Host-side audio I/O for Whisper:
    - Hann window construction.
    - STFT via direct DFT (no FFTW dependency — Whisper's window
      is small enough that the naive O(N log N) twiddle-loop is
      fine for the 30-second preprocessing budget).
    - Log-mel filterbank: 80 mel bands by default, configurable.
    - `pcm_to_mel(samples, mel_filters)` — top-level API.

## Lazy module name

`fuel-core/src/lazy_whisper_audio.rs` (new file).

## Architecture summary

Audio preprocessing is **pure host CPU computation** that produces
the input to the lazy Whisper encoder. No lazy graph involvement
on the preprocessing side — the output is a `Vec<f32>` of shape
`(n_mels, n_frames)` that the caller wraps with
`LazyTensor::from_vec` before feeding the encoder.

Steps (all f32 host-side):

1. **Resample** to 16 kHz (caller's responsibility; not in this
   port).
2. **Pad/truncate** to 30 seconds (480000 samples).
3. **STFT**: window with Hann (N=400), hop=160, FFT size=400.
4. **Power spectrum**: `|stft|^2`.
5. **Mel filterbank**: matmul against pre-computed mel filters
   `(n_mels, n_fft/2 + 1)` to get a mel spectrogram.
6. **Log compression**: `log10(max(mel, 1e-10))`, then clamp to
   `max(log_mel.max() - 8.0, log_mel)`, then `(log_mel + 4) / 4`.

## Primitives needed

- None on the lazy graph side. Host-side f32 only. The DFT can stay
  as the eager file's naive twiddle implementation; if it becomes
  a perf concern, swap to `rustfft` behind a feature flag — but
  not now (no consumer is bottlenecked on preprocessing yet).

## Reusable modules

- Mel filterbank table — eager file embeds 80-band and 128-band
  filterbanks as binary blobs included via `include_bytes!`. The
  lazy port re-includes the same blobs (they live at the
  `fuel-transformers` data path).
- `LazyTensor::from_vec` to wrap the resulting Vec into a tensor.

## Open questions

- The eager file's mel filterbank blob path is relative to the
  `fuel-transformers` crate. The lazy port lives in `fuel-core`
  which sits below `fuel-transformers`. Where do the blobs live?
  Options:
  1. Move the blobs into `fuel-core/data/` and include from there.
     Cleanest but adds blob to a lower crate.
  2. Make the lazy port take the filterbank tensor as a parameter
     and let `fuel-transformers` (or the caller) pass it. Cleanest
     architecturally — host-side preprocessing shouldn't ship blobs
     with the core lib.
  Preference: option 2.
- Should this even be in `fuel-core`? Audio preprocessing isn't
  ML-graph work. But every other lazy_* module is in `fuel-core`
  and consumers expect to import from there. Keep it in
  `fuel-core` for symmetry, but treat it as a leaf utility with no
  dependencies on the rest of `lazy_*`.

## Splits

Single session — ~350 LOC mechanical port.

## Test strategy

- **Bit-for-bit reproducibility against eager**: feed the same PCM
  buffer (sine wave at 440 Hz, 1 second) through eager
  `pcm_to_mel` and `lazy_whisper_audio::pcm_to_mel`, assert the
  output Vec<f32>s match within 1e-6.
- **Shape check**: 30-second buffer at 16 kHz with hop=160 → 3001
  frames; with `n_mels=80` → output shape `(80, 3001)`.
- **Log-mel clamp**: synthetic silent buffer → log-mel saturates at
  the clamp floor, not -inf.

## References

- Eager source: `fuel-transformers/src/models/audio/whisper/audio.rs`
- Whisper paper: <https://arxiv.org/abs/2212.04356>
- OpenAI reference: <https://github.com/openai/whisper/blob/main/whisper/audio.py>
- Sibling: nothing shipped yet on the lazy audio-preprocessing side
  — this is the first.

//! MetaVoice speaker encoder — lazy port.
//!
//! Multi-layer LSTM that maps a mel-spectrogram input
//! `(1, T, mel_n_channels)` to a normalized speaker d-vector
//! `(1, T, embedding_size)`. Used by MetaVoice for voice cloning
//! to produce a speaker conditioning vector that the TTS model
//! attends to.
//!
//! Forward pipeline (matches the eager port):
//!   `LSTM × N → linear(hidden, embedding) → ReLU → L2-norm
//!   along the embedding axis`
//!
//! Note: the eager `embed_utterance` helper that windows raw
//! audio into partials, computes mel features, and averages
//! across partials is signal-processing + I/O — not represented
//! here. Callers prepare the mel tensor in host code (e.g., via
//! the eager `models::whisper::audio::log_mel_spectrogram_`
//! pipeline) and call `forward` on the result.
//!
//! v1 scope: F32, batch == 1, forward-only inference.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_lstm::{LstmCellWeights, LstmStack};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerEncoderConfig {
    pub sampling_rate: usize,
    pub partial_n_frames: usize,
    pub model_hidden_size: usize,
    pub model_embedding_size: usize,
    pub model_num_layers: usize,
    pub mel_window_length: usize,
    pub mel_window_step: usize,
    pub mel_n_channels: usize,
}

impl SpeakerEncoderConfig {
    /// MetaVoice default speaker-encoder configuration.
    pub fn default_cfg() -> Self {
        Self {
            sampling_rate: 16_000,
            partial_n_frames: 160,
            model_hidden_size: 256,
            model_embedding_size: 256,
            model_num_layers: 3,
            mel_window_length: 25,
            mel_window_step: 10,
            mel_n_channels: 40,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpeakerEncoderWeights {
    pub lstm: LstmStack,
    /// `(hidden, embedding_size)` linear.
    pub linear: WeightStorage,
    pub linear_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct SpeakerEncoderModel {
    pub config: SpeakerEncoderConfig,
    pub weights: SpeakerEncoderWeights,
}

impl SpeakerEncoderModel {
    /// Forward pass: `(1, T, mel_n_channels)` → `(1, T, embedding_size)`,
    /// L2-normalized along the embedding axis.
    pub fn forward(&self, mels: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = mels.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "speaker-encoder input must be rank 3 [1, T, D]");
        let b = dims[0]; let t = dims[1]; let d = dims[2];
        assert_eq!(b, 1, "v1 supports batch == 1");
        assert_eq!(d, cfg.mel_n_channels);

        // Multi-layer LSTM stack.
        let lstm_out = self.weights.lstm.forward(mels)?;

        // Linear (hidden → embedding) + ReLU.
        let h = cfg.model_hidden_size;
        let e = cfg.model_embedding_size;
        let proj = self.weights.linear.apply_linear(&lstm_out, h, e);
        let bias = mels.const_f32_like(
            Arc::clone(&self.weights.linear_bias), Shape::from_dims(&[e]),
        );
        let with_bias = proj.broadcast_add(&bias)?;
        let activated = with_bias.relu();

        // L2 normalize along the embedding (last) axis.
        l2_normalize_last(&activated, b, t, e)
    }
}

fn l2_normalize_last(
    x: &LazyTensor, b: usize, t: usize, e: usize,
) -> Result<LazyTensor> {
    let _ = (b, t, e);
    x.l2_normalize(2_usize, 0.0)
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl SpeakerEncoderWeights {
    /// Load MetaVoice speaker-encoder weights from a HuggingFace
    /// `MmapedSafetensors` checkpoint.
    ///
    /// Naming convention (matches the eager port at
    /// `fuel-transformers/src/models/audio/metavoice.rs::speaker_encoder`):
    ///   - LSTM stack: `lstm.{layer_idx}.weight_ih_l{layer_idx}`,
    ///     `lstm.{layer_idx}.weight_hh_l{layer_idx}`,
    ///     `lstm.{layer_idx}.bias_ih_l{layer_idx}`,
    ///     `lstm.{layer_idx}.bias_hh_l{layer_idx}`
    ///     (PyTorch's `nn.LSTM` per-layer tensor names plus the
    ///     `fuel_nn::lstm` `layer_idx` sub-module prefix from
    ///     `vb.pp("lstm").pp(layer_idx)`).
    ///   - Linear: `linear.weight` `[embedding, hidden]`,
    ///     `linear.bias` `[embedding]`.
    ///
    /// LSTM gate ordering matches PyTorch's `[i, f, g, o]` layout
    /// along the leading axis — same as [`LstmCellWeights`], so no
    /// re-shuffle is needed.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &SpeakerEncoderConfig,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};

        let h = cfg.model_hidden_size;
        let e = cfg.model_embedding_size;
        let four_h = 4 * h;

        // LSTM stack — first layer takes mel_n_channels, the rest take
        // hidden_size as input.
        let mut layers: Vec<LstmCellWeights> = Vec::with_capacity(cfg.model_num_layers);
        for li in 0..cfg.model_num_layers {
            let in_dim = if li == 0 { cfg.mel_n_channels } else { h };
            let w_ih = load_tensor_as_f32(
                st, &format!("lstm.{li}.weight_ih_l{li}"),
            )?;
            let w_hh = load_tensor_as_f32(
                st, &format!("lstm.{li}.weight_hh_l{li}"),
            )?;
            let b_ih = load_tensor_as_f32(
                st, &format!("lstm.{li}.bias_ih_l{li}"),
            )?;
            let b_hh = load_tensor_as_f32(
                st, &format!("lstm.{li}.bias_hh_l{li}"),
            )?;
            if w_ih.len() != four_h * in_dim {
                crate::bail!(
                    "lstm.{li}.weight_ih_l{li}: {} elts, expected {}",
                    w_ih.len(), four_h * in_dim,
                );
            }
            if w_hh.len() != four_h * h {
                crate::bail!(
                    "lstm.{li}.weight_hh_l{li}: {} elts, expected {}",
                    w_hh.len(), four_h * h,
                );
            }
            if b_ih.len() != four_h {
                crate::bail!(
                    "lstm.{li}.bias_ih_l{li}: {} elts, expected {}",
                    b_ih.len(), four_h,
                );
            }
            if b_hh.len() != four_h {
                crate::bail!(
                    "lstm.{li}.bias_hh_l{li}: {} elts, expected {}",
                    b_hh.len(), four_h,
                );
            }
            layers.push(LstmCellWeights {
                w_ih: Arc::<[f32]>::from(w_ih),
                w_hh: Arc::<[f32]>::from(w_hh),
                b_ih: Arc::<[f32]>::from(b_ih),
                b_hh: Arc::<[f32]>::from(b_hh),
                input_dim: in_dim,
                hidden_dim: h,
            });
        }
        let lstm = LstmStack { layers };

        // Linear projection `(hidden, embedding)`. HF stores
        // `[out=embedding, in=hidden]`; we want the `[in, out]`
        // contiguous layout that `apply_linear(_, hidden, embedding)`
        // consumes.
        let linear = load_transposed_matrix_preserve_dtype(
            st, "linear.weight", e, h,
        )?;
        let linear_bias: Arc<[f32]> = load_tensor_as_f32(st, "linear.bias")
            .ok()
            .map(Arc::<[f32]>::from)
            .unwrap_or_else(|| Arc::<[f32]>::from(vec![0.0_f32; e]));

        Ok(Self { lstm, linear, linear_bias })
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }
    fn vec_of(n: usize, nb: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| nb()).collect::<Vec<_>>())
    }
    fn ws(n: usize, nb: &mut dyn FnMut() -> f32) -> WeightStorage {
        WeightStorage::F32(vec_of(n, nb))
    }

    fn tiny_cfg() -> SpeakerEncoderConfig {
        SpeakerEncoderConfig {
            sampling_rate: 16_000,
            partial_n_frames: 8,
            model_hidden_size: 8,
            model_embedding_size: 6,
            model_num_layers: 2,
            mel_window_length: 25,
            mel_window_step: 10,
            mel_n_channels: 4,
        }
    }

    fn build_lstm_layer(in_dim: usize, h: usize, nb: &mut dyn FnMut() -> f32) -> LstmCellWeights {
        LstmCellWeights {
            w_ih: vec_of(4 * h * in_dim, nb),
            w_hh: vec_of(4 * h * h, nb),
            b_ih: vec_of(4 * h, nb),
            b_hh: vec_of(4 * h, nb),
            input_dim: in_dim,
            hidden_dim: h,
        }
    }

    fn tiny_model() -> SpeakerEncoderModel {
        let cfg = tiny_cfg();
        let mut nb = rng_seed(2026);
        let mut layers = Vec::with_capacity(cfg.model_num_layers);
        // First layer: mel_n_channels → hidden. Remaining: hidden → hidden.
        layers.push(build_lstm_layer(cfg.mel_n_channels, cfg.model_hidden_size, &mut nb));
        for _ in 1..cfg.model_num_layers {
            layers.push(build_lstm_layer(cfg.model_hidden_size, cfg.model_hidden_size, &mut nb));
        }
        let weights = SpeakerEncoderWeights {
            lstm: LstmStack { layers },
            linear: ws(cfg.model_hidden_size * cfg.model_embedding_size, &mut nb),
            linear_bias: vec_of(cfg.model_embedding_size, &mut nb),
        };
        SpeakerEncoderModel { config: cfg, weights }
    }

    #[test]
    fn forward_shape_and_finite() {
        let model = tiny_model();
        let cfg = &model.config;
        let t = 5;
        let mels = LazyTensor::from_f32(
            (0..(1 * t * cfg.mel_n_channels)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, t, cfg.mel_n_channels]),
            &Device::cpu(),
        );
        let out = model.forward(&mels).unwrap();
        assert_eq!(out.shape().dims(), &[1, t, cfg.model_embedding_size]);
        for &v in &out.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn forward_l2_normalized_per_row() {
        let model = tiny_model();
        let cfg = &model.config;
        let t = 4;
        let mels = LazyTensor::from_f32(
            (0..(1 * t * cfg.mel_n_channels)).map(|i| (i as f32) * 0.01 + 0.1).collect::<Vec<_>>(),
            Shape::from_dims(&[1, t, cfg.mel_n_channels]),
            &Device::cpu(),
        );
        let out = model.forward(&mels).unwrap().realize_f32();
        let e = cfg.model_embedding_size;
        for row in 0..t {
            let start = row * e;
            let mut norm_sq = 0.0_f32;
            for d in 0..e {
                norm_sq += out[start + d].powi(2);
            }
            let norm = norm_sq.sqrt();
            // The relu pre-norm can leave a row at exact zero (all
            // hidden values clipped) — in that case the L2-norm is
            // also zero and we can't sensibly check. Skip those.
            if norm > 1e-7 {
                assert!((norm - 1.0).abs() < 1e-5,
                    "row {row} norm = {norm}, expected ~1.0");
            }
        }
    }

    #[test]
    fn preset_default_cfg() {
        let p = SpeakerEncoderConfig::default_cfg();
        assert_eq!(p.model_hidden_size, 256);
        assert_eq!(p.model_embedding_size, 256);
        assert_eq!(p.mel_n_channels, 40);
    }
}

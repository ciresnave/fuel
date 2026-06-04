//! Mimi — top-level Encodec model.
//!
//! Composes the four lazy Mimi building blocks shipped earlier
//! in this session into a single audio codec:
//!
//! ```text
//! encode(audio):
//!   audio                                  (1, channels, T_audio)
//!     → SeaNetEncoder                       (1, seanet_dim, T_seanet)
//!     → ProjectedTransformer (encoder side) (1, seanet_dim, T_seanet)
//!     → ConvDownsample1d                    (1, seanet_dim, T_codes)
//!     → SplitRVQ.encode                     (1, n_q, T_codes)
//!
//! decode(codes):
//!   codes                                   (1, n_q, T_codes)
//!     → SplitRVQ.decode                     (1, seanet_dim, T_codes)
//!     → ConvTrUpsample1d                    (1, seanet_dim, T_seanet)
//!     → ProjectedTransformer (decoder side) (1, seanet_dim, T_seanet)
//!     → SeaNetDecoder                       (1, channels, T_audio)
//! ```
//!
//! v1 scope: F32, batch == 1, forward-only inference, single-call
//! (no streaming).

use crate::lazy::LazyTensor;
use crate::lazy_mimi_quantization::{
    split_rvq_decode, split_rvq_encode, SplitResidualVectorQuantizerWeights,
};
use crate::lazy_mimi_resampler::{ConvDownsample1dModel, ConvTrUpsample1dModel};
use crate::lazy_mimi_seanet::{
    SeaNetConfig, SeaNetDecoderModel, SeaNetDecoderWeights, SeaNetEncoderModel,
    SeaNetEncoderWeights,
};
use crate::lazy_mimi_transformer::{
    MimiTransformerConfig, ProjectedTransformerModel, ProjectedTransformerWeights,
};
use crate::Result;

#[derive(Debug, Clone)]
pub struct MimiConfig {
    pub seanet: SeaNetConfig,
    pub transformer: MimiTransformerConfig,
    /// Audio channel count (1 for mono Mimi v0.1).
    pub channels: usize,
    /// Number of RVQ codebooks (1 semantic + `n_q - 1` acoustic).
    pub n_q: usize,
    /// Codebook size (== bins).
    pub quantizer_bins: usize,
    /// Pre-quantizer projection dim — the dim the SRVQ internally
    /// operates in. The encoder transformer's output is projected
    /// into this dim via the `output_projs` slot of the
    /// transformer.
    pub quantizer_dim: usize,
}

impl MimiConfig {
    /// Mimi v0.1 preset (24 kHz audio, 8-codebook RVQ, 2048-bin
    /// codebooks). Matches the eager `Config::v0_1(num_codebooks =
    /// None)` defaults.
    pub fn v0_1(num_codebooks: Option<usize>) -> Self {
        Self {
            seanet: SeaNetConfig::mimi_v0_1(),
            transformer: MimiTransformerConfig::mimi_v0_1(),
            channels: 1,
            n_q: num_codebooks.unwrap_or(16),
            quantizer_bins: 2048,
            quantizer_dim: 256,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MimiWeights {
    pub encoder: SeaNetEncoderWeights,
    pub decoder: SeaNetDecoderWeights,
    pub encoder_transformer: ProjectedTransformerWeights,
    pub decoder_transformer: ProjectedTransformerWeights,
    /// `(seanet_dim, seanet_dim, kernel = 2·stride)`.
    pub downsample_weight: std::sync::Arc<[f32]>,
    /// `(seanet_dim, 1, kernel = 2·stride)` — depthwise.
    pub upsample_weight: std::sync::Arc<[f32]>,
    pub quantizer: SplitResidualVectorQuantizerWeights,
    /// Stride of the down/upsample pair — set at build time from
    /// `encoder_frame_rate / frame_rate`.
    pub resampler_stride: usize,
}

#[derive(Debug, Clone)]
pub struct MimiModel {
    pub config: MimiConfig,
    pub weights: MimiWeights,
}

impl MimiModel {
    fn encoder(&self) -> SeaNetEncoderModel {
        SeaNetEncoderModel {
            config: self.config.seanet.clone(),
            weights: self.weights.encoder.clone(),
        }
    }
    fn decoder(&self) -> SeaNetDecoderModel {
        SeaNetDecoderModel {
            config: self.config.seanet.clone(),
            weights: self.weights.decoder.clone(),
        }
    }
    fn encoder_transformer(&self) -> ProjectedTransformerModel {
        ProjectedTransformerModel {
            config: self.config.transformer.clone(),
            input_dim: self.config.seanet.dimension,
            weights: self.weights.encoder_transformer.clone(),
        }
    }
    fn decoder_transformer(&self) -> ProjectedTransformerModel {
        ProjectedTransformerModel {
            config: self.config.transformer.clone(),
            input_dim: self.config.seanet.dimension,
            weights: self.weights.decoder_transformer.clone(),
        }
    }
    fn downsample(&self) -> ConvDownsample1dModel {
        ConvDownsample1dModel {
            weights: crate::lazy_mimi_resampler::ConvDownsample1dWeights {
                weight: std::sync::Arc::clone(&self.weights.downsample_weight),
                dim: self.config.seanet.dimension,
                stride: self.weights.resampler_stride,
            },
        }
    }
    fn upsample(&self) -> ConvTrUpsample1dModel {
        ConvTrUpsample1dModel {
            weights: crate::lazy_mimi_resampler::ConvTrUpsample1dWeights {
                weight: std::sync::Arc::clone(&self.weights.upsample_weight),
                dim: self.config.seanet.dimension,
                stride: self.weights.resampler_stride,
            },
        }
    }

    /// Full encode: audio waveform → discrete RVQ codes.
    /// Input `(1, channels, T_audio)`; output `(1, n_q, T_codes)`.
    pub fn encode(&self, audio: &LazyTensor) -> Result<LazyTensor> {
        let h = self.encoder().forward(audio)?;
        let h = self.encoder_transformer().forward(&h)?;
        let h = h.into_iter().next()
            .expect("encoder transformer must yield ≥1 output");
        let h = self.downsample().forward(&h)?;
        split_rvq_encode(&h, &self.weights.quantizer)
    }

    /// Full decode: RVQ codes → reconstructed audio.
    /// Input `(1, n_q, T_codes)`; output `(1, channels, T_audio)`.
    pub fn decode(&self, codes: &LazyTensor) -> Result<LazyTensor> {
        let h = split_rvq_decode(codes, &self.weights.quantizer)?;
        let h = self.upsample().forward(&h)?;
        let h = self.decoder_transformer().forward(&h)?;
        let h = h.into_iter().next()
            .expect("decoder transformer must yield ≥1 output");
        self.decoder().forward(&h)
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_v0_1_default() {
        let cfg = MimiConfig::v0_1(None);
        assert_eq!(cfg.channels, 1);
        assert_eq!(cfg.n_q, 16);
        assert_eq!(cfg.quantizer_bins, 2048);
        assert_eq!(cfg.quantizer_dim, 256);
        assert_eq!(cfg.seanet.dimension, 512);
        assert_eq!(cfg.transformer.d_model, 512);
    }

    #[test]
    fn preset_v0_1_codebook_override() {
        let cfg = MimiConfig::v0_1(Some(8));
        assert_eq!(cfg.n_q, 8);
        // Other fields unchanged.
        assert_eq!(cfg.quantizer_bins, 2048);
        let ratios_product: usize = cfg.seanet.ratios.iter().product();
        // 24 kHz audio / 960× downsample = 25 Hz; codec is then
        // resampled to 12.5 Hz via the resampler.
        assert_eq!(ratios_product, 8 * 6 * 5 * 4);
    }
}


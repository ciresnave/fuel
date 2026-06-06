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

// ---- HuggingFace safetensors loader ----------------------------------------

impl MimiWeights {
    /// Load Mimi weights from a HuggingFace `MmapedSafetensors`
    /// checkpoint (e.g. `kyutai/mimi/model.safetensors`). Composes
    /// the four sub-module loaders:
    ///
    /// - `encoder.layers.*` — [`SeaNetEncoderWeights::load_from_mmapped`]
    /// - `decoder.layers.*` — [`SeaNetDecoderWeights::load_from_mmapped`]
    /// - `encoder_transformer.*` — [`ProjectedTransformerWeights::load_from_mmapped`]
    /// - `decoder_transformer.*` — [`ProjectedTransformerWeights::load_from_mmapped`]
    /// - `downsample.conv.{weight, bias?}` — top-level Mimi
    ///   `ConvDownsample1d` (bias-less, kernel = `2·resampler_stride`)
    /// - `upsample.convtr.{weight, bias?}` — top-level Mimi
    ///   `ConvTrUpsample1d` (depthwise, bias-less)
    /// - `quantizer.{semantic, acoustic}_residual_vector_quantizer.*`
    ///   — [`SplitResidualVectorQuantizerWeights::load_from_mmapped`]
    ///
    /// `resampler_stride` is the integer `encoder_frame_rate /
    /// frame_rate` ratio used at build time by
    /// [`MimiEncodecConfig::resampler_stride`]; the loader uses it
    /// to size the down/upsample kernels (`2 · stride`) but does
    /// not re-derive it.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &MimiConfig,
        resampler_stride: usize,
    ) -> Result<Self> {
        use crate::lazy::load_tensor_as_f32;

        let encoder = SeaNetEncoderWeights::load_from_mmapped(
            st, "encoder", &cfg.seanet,
        )?;
        let decoder = SeaNetDecoderWeights::load_from_mmapped(
            st, "decoder", &cfg.seanet,
        )?;

        let dim = cfg.seanet.dimension;
        // Both ProjectedTransformers take `input_dim = dim` and a
        // single `output_dim = dim` slot in Mimi v0.1 (the eager
        // `Encodec::new` passes `&[dim]` for output_dims).
        let encoder_transformer = ProjectedTransformerWeights::load_from_mmapped(
            st, "encoder_transformer", &cfg.transformer, dim, &[dim],
        )?;
        let decoder_transformer = ProjectedTransformerWeights::load_from_mmapped(
            st, "decoder_transformer", &cfg.transformer, dim, &[dim],
        )?;

        // Top-level downsample / upsample. Both are bias-less in the
        // eager port (`bias = false` on the underlying
        // StreamableConv1d / StreamableConvTranspose1d).
        let kernel = 2 * resampler_stride;
        // downsample: `(dim, dim, kernel)` — full-channel mix.
        let dn = load_tensor_as_f32(st, "downsample.conv.weight")?;
        let expected_dn = dim * dim * kernel;
        if dn.len() != expected_dn {
            crate::bail!(
                "downsample.conv.weight: {} elements, expected {expected_dn} ({dim}×{dim}×{kernel})",
                dn.len(),
            );
        }
        // upsample: `(dim, 1, kernel)` — depthwise (`groups = dim`).
        let up = load_tensor_as_f32(st, "upsample.convtr.weight")?;
        let expected_up = dim * 1 * kernel;
        if up.len() != expected_up {
            crate::bail!(
                "upsample.convtr.weight: {} elements, expected {expected_up} ({dim}×1×{kernel})",
                up.len(),
            );
        }

        let quantizer = SplitResidualVectorQuantizerWeights::load_from_mmapped(
            st, "quantizer",
            cfg.quantizer_dim,
            dim,
            dim,
            cfg.n_q,
            cfg.quantizer_bins,
        )?;

        Ok(MimiWeights {
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            downsample_weight: std::sync::Arc::from(dn),
            upsample_weight: std::sync::Arc::from(up),
            quantizer,
            resampler_stride,
        })
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


//! Mimi Encodec — top-level neural audio codec.
//!
//! Composition port of `fuel-transformers/src/models/audio/mimi/encodec.rs`.
//! Bundles the already-shipped sub-modules into a single
//! `encode(pcm) → codes` / `decode(codes) → pcm` codec:
//!
//! - [`crate::lazy_mimi_seanet`] — convolutional encoder/decoder.
//! - [`crate::lazy_mimi_transformer`] — streaming transformer
//!   (encoder + decoder).
//! - [`crate::lazy_mimi_resampler`] — strided causal conv1d /
//!   transposed conv1d between the SeaNet frame rate and the
//!   user-configured `frame_rate` at the quantizer.
//! - [`crate::lazy_mimi_quantization`] — split residual vector
//!   quantizer.
//!
//! The host-side resampling control between encoder frame rate
//! (`sample_rate / Π ratios`) and quantizer `frame_rate` is exposed
//! through [`MimiEncodecConfig::frame_rate`] / `sample_rate`; the
//! stride between the two is derived once at build time and baked
//! into the [`crate::lazy_mimi::MimiWeights::resampler_stride`].
//!
//! ## Streaming
//!
//! [`MimiEncodecModel::encode_step`] / `decode_step` provide a
//! frame-by-frame API equivalent to one-shot encode/decode up to
//! floating-point reassociation in the underlying convs. The
//! streaming state simply accumulates the input chunks and re-runs
//! the one-shot forward, emitting only newly-produced frames; this
//! gives **streaming-equals-one-shot** by construction at the cost
//! of O(T²) work over the full sequence. Real-time low-latency
//! streaming for production inference will require per-layer state
//! plumbing through the SeaNet / transformer sub-modules — out of
//! scope for the composition port.

use crate::lazy::LazyTensor;
use crate::lazy_mimi::{MimiConfig, MimiModel, MimiWeights};
use crate::Result;
use fuel_ir::Shape;

/// How the codec resamples between the SeaNet encoder frame rate
/// and the quantizer frame rate. Mimi v0.1 uses [`ResampleMethod::Conv`].
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ResampleMethod {
    /// Learnt strided convolution / transposed convolution
    /// (the only mode the lazy resampler implements).
    Conv,
    /// Linear / nearest interpolation (not implemented in the
    /// lazy port — present only so the config matches the eager
    /// surface for downstream type-compatibility).
    Interpolate,
}

/// Top-level Mimi codec configuration. Wraps [`MimiConfig`] with the
/// host-side audio framing knobs (sample rate / frame rate / channel
/// count / resample method) needed to compute the resampler stride.
#[derive(Debug, Clone)]
pub struct MimiEncodecConfig {
    /// Audio channel count.
    pub channels: usize,
    /// Input audio sample rate in Hz (e.g. `24_000.0` for Mimi v0.1).
    pub sample_rate: f64,
    /// Quantizer frame rate in Hz (e.g. `12.5` for Mimi v0.1).
    pub frame_rate: f64,
    /// Whether the model expects renormalized audio. Honoured by the
    /// caller — the lazy codec does not apply renormalization
    /// itself.
    pub renormalize: bool,
    /// How the codec resamples between encoder and quantizer frame rates.
    pub resample_method: ResampleMethod,
    /// Inner config carrying SeaNet + transformer + quantizer dims.
    pub inner: MimiConfig,
}

impl MimiEncodecConfig {
    /// Mimi v0.1 preset (24 kHz audio, 12.5 Hz codes, 8-codebook RVQ
    /// when `num_codebooks = None` is overridden to 8 by the
    /// canonical Kyutai release; the default carried through here
    /// is 16 to match the lazy `MimiConfig::v0_1`).
    pub fn v0_1(num_codebooks: Option<usize>) -> Self {
        Self {
            channels: 1,
            sample_rate: 24_000.0,
            frame_rate: 12.5,
            renormalize: true,
            resample_method: ResampleMethod::Conv,
            inner: MimiConfig::v0_1(num_codebooks),
        }
    }

    /// Encoder frame rate in Hz — the natural rate produced by the
    /// SeaNet conv stack before the resampler. Equals
    /// `sample_rate / Π ratios`.
    pub fn encoder_frame_rate(&self) -> f64 {
        let ratios_product: usize = self.inner.seanet.ratios.iter().product();
        self.sample_rate / ratios_product as f64
    }

    /// Stride between the SeaNet frame rate and the quantizer
    /// `frame_rate`. Equals `encoder_frame_rate / frame_rate` as an
    /// integer.
    pub fn resampler_stride(&self) -> usize {
        (self.encoder_frame_rate() / self.frame_rate) as usize
    }

    /// Total audio→code downsample factor: `Π ratios · resampler_stride`.
    pub fn total_audio_stride(&self) -> usize {
        let ratios_product: usize = self.inner.seanet.ratios.iter().product();
        ratios_product * self.resampler_stride()
    }
}

/// Mimi neural audio codec. Thin owning wrapper around
/// [`MimiModel`] that exposes the encodec encode / decode / streaming
/// API surface.
#[derive(Debug, Clone)]
pub struct MimiEncodecModel {
    pub config: MimiEncodecConfig,
    pub inner: MimiModel,
}

impl MimiEncodecModel {
    /// Build the codec from a top-level config + the four sub-module
    /// weight bundles. Verifies the resampler stride captured in
    /// `weights.resampler_stride` matches the stride computed from
    /// the config so that downstream `decode` consumers cannot
    /// silently disagree with the encoder framing.
    pub fn new(config: MimiEncodecConfig, weights: MimiWeights) -> Result<Self> {
        let expected_stride = config.resampler_stride();
        if weights.resampler_stride != expected_stride {
            return Err(crate::Error::Msg(format!(
                "MimiEncodecModel: weights.resampler_stride {} does not match \
                 config-derived stride {} (encoder_frame_rate {} / frame_rate {})",
                weights.resampler_stride,
                expected_stride,
                config.encoder_frame_rate(),
                config.frame_rate,
            )));
        }
        let inner = MimiModel {
            config: config.inner.clone(),
            weights,
        };
        Ok(Self { config, inner })
    }

    /// One-shot encode: PCM `(1, channels, T_audio)` → codes
    /// `(1, n_q, T_codes)` where `T_codes = T_audio / total_audio_stride`.
    pub fn encode(&self, pcm: &LazyTensor) -> Result<LazyTensor> {
        let dims = pcm.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[1] != self.config.channels {
            return Err(crate::Error::Msg(format!(
                "MimiEncodecModel::encode: expected (1, {}, T), got {:?}",
                self.config.channels, dims,
            )));
        }
        self.inner.encode(pcm)
    }

    /// One-shot decode: codes `(1, n_q, T_codes)` → PCM
    /// `(1, channels, T_audio)` where `T_audio = T_codes · total_audio_stride`.
    pub fn decode(&self, codes: &LazyTensor) -> Result<LazyTensor> {
        let dims = codes.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[1] != self.config.inner.n_q {
            return Err(crate::Error::Msg(format!(
                "MimiEncodecModel::decode: expected (1, {}, T), got {:?}",
                self.config.inner.n_q, dims,
            )));
        }
        self.inner.decode(codes)
    }

    /// Streaming encode step. Append `pcm_chunk` to the carried
    /// state's input buffer; if the accumulated buffer now contains
    /// at least one full encoder frame more than was previously
    /// emitted, run encode on the full buffer and return only the
    /// new codes. Returns `(new_state, None)` if not enough new
    /// samples have accumulated yet.
    ///
    /// The chunk-then-concatenate output recovers
    /// [`Self::encode`] bit-for-bit up to floating-point
    /// reassociation in the conv stack. PCM samples are buffered on
    /// the host so successive chunks may originate on independent
    /// graphs (the typical case for caller-driven streaming).
    pub fn encode_step(
        &self,
        mut state: MimiEncodecEncodeState,
        pcm_chunk: &LazyTensor,
    ) -> Result<(MimiEncodecEncodeState, Option<LazyTensor>)> {
        let dims = pcm_chunk.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[1] != self.config.channels {
            return Err(crate::Error::Msg(format!(
                "MimiEncodecModel::encode_step: expected (1, {}, L), got {:?}",
                self.config.channels, dims,
            )));
        }
        let chunk_len = dims[2];
        let chunk_host = pcm_chunk.realize_f32();
        if chunk_host.len() != self.config.channels * chunk_len {
            return Err(crate::Error::Msg(format!(
                "MimiEncodecModel::encode_step: realized PCM length {} != channels·L = {}",
                chunk_host.len(),
                self.config.channels * chunk_len,
            )));
        }
        state.buffer.extend_from_slice(&chunk_host);
        state.buffered_samples += chunk_len;

        // Snap to the largest prefix that is a whole number of
        // codec frames; otherwise the SeaNet downsample drops a
        // tail and the next step would see a phase shift.
        let stride = self.config.total_audio_stride();
        let total_frames = state.buffered_samples / stride;
        if total_frames <= state.emitted_frames {
            return Ok((state, None));
        }
        let usable_len = total_frames * stride;
        let pcm_for_run: Vec<f32> = if self.config.channels == 1 {
            state.buffer[..usable_len].to_vec()
        } else {
            // Interleaved by channel: state.buffer holds
            // (channels, T) row-major (channel-major). Slice columns 0..usable_len.
            let mut v = Vec::with_capacity(self.config.channels * usable_len);
            for c in 0..self.config.channels {
                let base = c * state.buffered_samples;
                v.extend_from_slice(&state.buffer[base..base + usable_len]);
            }
            v
        };
        let pcm_full = LazyTensor::from_f32(
            pcm_for_run,
            Shape::from_dims(&[1, self.config.channels, usable_len]),
            &crate::Device::cpu(),
        );
        let codes = self.inner.encode(&pcm_full)?;
        let new_frames = total_frames - state.emitted_frames;
        let emitted_so_far = state.emitted_frames;
        state.emitted_frames = total_frames;
        let new_codes = codes.narrow(2_usize, emitted_so_far, new_frames)?;
        Ok((state, Some(new_codes)))
    }

    /// Streaming decode step. Append `codes_chunk` to the carried
    /// state's code buffer and emit PCM for any newly-decodable
    /// frames. Chunk-then-concatenate output recovers
    /// [`Self::decode`] bit-for-bit up to floating-point
    /// reassociation. Codes are buffered on the host so successive
    /// chunks may originate on independent graphs.
    pub fn decode_step(
        &self,
        mut state: MimiEncodecDecodeState,
        codes_chunk: &LazyTensor,
    ) -> Result<(MimiEncodecDecodeState, Option<LazyTensor>)> {
        let dims = codes_chunk.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[1] != self.config.inner.n_q {
            return Err(crate::Error::Msg(format!(
                "MimiEncodecModel::decode_step: expected (1, {}, L), got {:?}",
                self.config.inner.n_q, dims,
            )));
        }
        let chunk_frames = dims[2];
        if chunk_frames == 0 {
            return Ok((state, None));
        }
        let n_q = self.config.inner.n_q;
        let chunk_host = crate::pipelined_bridge::realize_one_as::<u32>(
            codes_chunk.graph_tensor().graph(),
            codes_chunk.graph_tensor().id(),
            &crate::Device::cpu(),
        )
        .map_err(|e| crate::Error::Msg(format!("decode_step: realize codes: {e:?}")))?;
        if chunk_host.len() != n_q * chunk_frames {
            return Err(crate::Error::Msg(format!(
                "MimiEncodecModel::decode_step: realized code length {} != n_q·L = {}",
                chunk_host.len(),
                n_q * chunk_frames,
            )));
        }
        // chunk_host layout: (n_q, chunk_frames) row-major. Append
        // each codebook row to the per-codebook host buffer.
        if state.buffer.is_empty() {
            state.buffer.resize(n_q, Vec::new());
        }
        for q in 0..n_q {
            let row = &chunk_host[q * chunk_frames..(q + 1) * chunk_frames];
            state.buffer[q].extend_from_slice(row);
        }
        state.buffered_frames += chunk_frames;
        if state.buffered_frames <= state.emitted_frames {
            return Ok((state, None));
        }
        let total_frames = state.buffered_frames;
        let mut flat = Vec::with_capacity(n_q * total_frames);
        for q in 0..n_q {
            flat.extend_from_slice(&state.buffer[q]);
        }
        let codes_full = LazyTensor::from_u32(
            flat,
            Shape::from_dims(&[1, n_q, total_frames]),
            &crate::Device::cpu(),
        );
        let pcm = self.inner.decode(&codes_full)?;
        let stride = self.config.total_audio_stride();
        let new_samples = (state.buffered_frames - state.emitted_frames) * stride;
        let emitted_samples = state.emitted_frames * stride;
        state.emitted_frames = state.buffered_frames;
        let new_pcm = pcm.narrow(2_usize, emitted_samples, new_samples)?;
        Ok((state, Some(new_pcm)))
    }
}

/// Streaming-encode carry. Holds accumulated PCM samples on the host
/// (channel-major: `buffer[c * buffered_samples + t]`) plus a count
/// of emitted code frames so `encode_step` can return only the
/// newly-produced codes on each call.
#[derive(Debug, Clone, Default)]
pub struct MimiEncodecEncodeState {
    pub buffer: Vec<f32>,
    pub buffered_samples: usize,
    pub emitted_frames: usize,
}

impl MimiEncodecEncodeState {
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Streaming-decode carry. Holds accumulated codes on the host as
/// per-codebook rows (`buffer[q][t]`) plus a count of emitted code
/// frames so `decode_step` can return only the newly-produced PCM
/// samples on each call.
#[derive(Debug, Clone, Default)]
pub struct MimiEncodecDecodeState {
    pub buffer: Vec<Vec<u32>>,
    pub buffered_frames: usize,
    pub emitted_frames: usize,
}

impl MimiEncodecDecodeState {
    pub fn empty() -> Self {
        Self::default()
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::WeightStorage;
    use crate::lazy_mimi_quantization::{
        EuclideanCodebookWeights, ResidualVectorQuantizationWeights,
        ResidualVectorQuantizerWeights, SplitResidualVectorQuantizerWeights,
        VectorQuantizationWeights,
    };
    use crate::lazy_mimi_seanet::{
        LazyConv1dWeights, LazyConvTranspose1dWeights, SeaNetActivation, SeaNetConfig,
        SeaNetDecoderLayerWeights, SeaNetDecoderWeights, SeaNetEncoderLayerWeights,
        SeaNetEncoderWeights, SeaNetResnetBlockWeights,
    };
    use crate::lazy_mimi_transformer::{
        LayerNormWeights, MimiAttentionWeights, MimiMlpWeights, MimiTransformerConfig,
        MimiTransformerLayerWeights, MimiTransformerWeights, ProjectedTransformerWeights,
    };
    use crate::lazy_encodec::PadMode;
    use crate::Device;
    use std::sync::Arc;

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

    fn conv_w(
        in_c: usize, out_c: usize, k: usize, stride: usize, dilation: usize, groups: usize,
        bias: bool, nb: &mut dyn FnMut() -> f32,
    ) -> LazyConv1dWeights {
        LazyConv1dWeights {
            weight: vec_of(out_c * (in_c / groups) * k, nb),
            bias: if bias { Some(vec_of(out_c, nb)) } else { None },
            in_channels: in_c, out_channels: out_c,
            kernel_size: k, stride, dilation, groups,
        }
    }

    fn conv_tr_w(
        in_c: usize, out_c: usize, k: usize, stride: usize, groups: usize,
        bias: bool, nb: &mut dyn FnMut() -> f32,
    ) -> LazyConvTranspose1dWeights {
        LazyConvTranspose1dWeights {
            weight: vec_of(in_c * (out_c / groups) * k, nb),
            bias: if bias { Some(vec_of(out_c, nb)) } else { None },
            in_channels: in_c, out_channels: out_c,
            kernel_size: k, stride, groups,
        }
    }

    fn resnet_block_w(
        dim: usize, k: usize, dilation: usize, compress: usize, true_skip: bool,
        nb: &mut dyn FnMut() -> f32,
    ) -> SeaNetResnetBlockWeights {
        let hidden = dim / compress;
        SeaNetResnetBlockWeights {
            convs: vec![
                conv_w(dim, hidden, k, 1, dilation, 1, true, nb),
                conv_w(hidden, dim, 1, 1, 1, 1, true, nb),
            ],
            shortcut: if true_skip { None } else { Some(conv_w(dim, dim, 1, 1, 1, 1, true, nb)) },
        }
    }

    fn tiny_seanet_cfg() -> SeaNetConfig {
        SeaNetConfig {
            dimension: 8, channels: 1,
            n_filters: 2,
            n_residual_layers: 1,
            ratios: vec![2, 2],
            activation: SeaNetActivation::Elu1,
            kernel_size: 3, residual_kernel_size: 3, last_kernel_size: 3,
            dilation_base: 2, pad_mode: PadMode::Constant,
            true_skip: true, compress: 2,
            final_activation: None,
        }
    }

    fn tiny_transformer_cfg() -> MimiTransformerConfig {
        MimiTransformerConfig {
            d_model: 8, num_heads: 2, num_layers: 1,
            dim_feedforward: 16,
            max_period: 10_000.0,
            conv_layout: true,
            layer_norm_eps: 1e-5,
        }
    }

    fn ws(n: usize, nb: &mut dyn FnMut() -> f32) -> WeightStorage {
        WeightStorage::F32(vec_of(n, nb))
    }

    fn ln_w(c: usize) -> LayerNormWeights {
        LayerNormWeights {
            gain: Arc::from(vec![1.0_f32; c]),
            bias: Arc::from(vec![0.0_f32; c]),
        }
    }

    fn build_transformer_layer(
        d: usize, ff: usize, nb: &mut dyn FnMut() -> f32,
    ) -> MimiTransformerLayerWeights {
        MimiTransformerLayerWeights {
            norm1: ln_w(d),
            norm2: ln_w(d),
            attn: MimiAttentionWeights {
                q_proj: ws(d * d, nb),
                k_proj: ws(d * d, nb),
                v_proj: ws(d * d, nb),
                o_proj: ws(d * d, nb),
            },
            mlp: MimiMlpWeights {
                fc1: ws(d * ff, nb),
                fc2: ws(ff * d, nb),
            },
            layer_scale_1: vec_of(d, nb),
            layer_scale_2: vec_of(d, nb),
        }
    }

    fn build_projected_transformer(
        cfg: &MimiTransformerConfig, nb: &mut dyn FnMut() -> f32,
    ) -> ProjectedTransformerWeights {
        let d = cfg.d_model;
        let ff = cfg.dim_feedforward;
        ProjectedTransformerWeights {
            transformer: MimiTransformerWeights {
                layers: (0..cfg.num_layers).map(|_| build_transformer_layer(d, ff, nb)).collect(),
            },
            // input_dim == d → no input projection.
            input_proj: None,
            // single output, output_dim == d → no output projection.
            output_projs: vec![(None, d)],
        }
    }

    fn build_seanet_encoder(cfg: &SeaNetConfig, nb: &mut dyn FnMut() -> f32) -> SeaNetEncoderWeights {
        let mut mult = 1_usize;
        let init_conv = conv_w(cfg.channels, mult * cfg.n_filters, cfg.kernel_size, 1, 1, 1, true, nb);
        let mut layers = Vec::with_capacity(cfg.ratios.len());
        for ratio in cfg.ratios.iter().rev() {
            let dim = mult * cfg.n_filters;
            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                residuals.push(resnet_block_w(
                    dim, cfg.residual_kernel_size,
                    cfg.dilation_base.pow(j as u32),
                    cfg.compress, cfg.true_skip, nb,
                ));
            }
            let downsample = conv_w(dim, dim * 2, ratio * 2, *ratio, 1, 1, true, nb);
            layers.push(SeaNetEncoderLayerWeights { residuals, downsample });
            mult *= 2;
        }
        let final_conv = conv_w(
            mult * cfg.n_filters, cfg.dimension, cfg.last_kernel_size, 1, 1, 1, true, nb,
        );
        SeaNetEncoderWeights { init_conv, layers, final_conv }
    }

    fn build_seanet_decoder(cfg: &SeaNetConfig, nb: &mut dyn FnMut() -> f32) -> SeaNetDecoderWeights {
        let mut mult = 1_usize << cfg.ratios.len();
        let init_conv = conv_w(cfg.dimension, mult * cfg.n_filters, cfg.kernel_size, 1, 1, 1, true, nb);
        let mut layers = Vec::with_capacity(cfg.ratios.len());
        for ratio in cfg.ratios.iter() {
            let dim = mult * cfg.n_filters;
            let out_dim = dim / 2;
            let upsample = conv_tr_w(dim, out_dim, ratio * 2, *ratio, 1, true, nb);
            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                residuals.push(resnet_block_w(
                    out_dim, cfg.residual_kernel_size,
                    cfg.dilation_base.pow(j as u32),
                    cfg.compress, cfg.true_skip, nb,
                ));
            }
            layers.push(SeaNetDecoderLayerWeights { upsample, residuals });
            mult /= 2;
        }
        let final_conv = conv_w(
            cfg.n_filters, cfg.channels, cfg.last_kernel_size, 1, 1, 1, true, nb,
        );
        SeaNetDecoderWeights { init_conv, layers, final_conv }
    }

    fn tiny_codebook(
        codebook_size: usize, codebook_dim: usize, nb: &mut dyn FnMut() -> f32,
    ) -> EuclideanCodebookWeights {
        let emb = vec_of(codebook_size * codebook_dim, nb);
        let mut c2_v = Vec::with_capacity(codebook_size);
        for i in 0..codebook_size {
            let mut s = 0.0_f32;
            for j in 0..codebook_dim {
                let e = emb[i * codebook_dim + j];
                s += e * e;
            }
            c2_v.push(s / 2.0);
        }
        EuclideanCodebookWeights {
            embedding: emb,
            c2: Arc::from(c2_v),
            codebook_size,
            codebook_dim,
        }
    }

    fn tiny_vq(dim: usize, codebook_size: usize, nb: &mut dyn FnMut() -> f32) -> VectorQuantizationWeights {
        VectorQuantizationWeights {
            codebook: tiny_codebook(codebook_size, dim, nb),
            project_in_w: None, project_in_b: None,
            project_out_w: None, project_out_b: None,
            dim,
        }
    }

    fn tiny_rvq_quantizer(
        n_q: usize, internal_dim: usize, external_dim: usize, codebook_size: usize,
        nb: &mut dyn FnMut() -> f32,
    ) -> ResidualVectorQuantizerWeights {
        // 1x1 projections wired when internal_dim != external_dim.
        let input_proj_w = if internal_dim == external_dim {
            None
        } else {
            Some(vec_of(internal_dim * external_dim * 1, nb))
        };
        let output_proj_w = if internal_dim == external_dim {
            None
        } else {
            Some(vec_of(external_dim * internal_dim * 1, nb))
        };
        ResidualVectorQuantizerWeights {
            vq: ResidualVectorQuantizationWeights {
                layers: (0..n_q).map(|_| tiny_vq(internal_dim, codebook_size, nb)).collect(),
            },
            input_proj_w,
            output_proj_w,
            dim: internal_dim,
            input_dim: external_dim,
            output_dim: external_dim,
        }
    }

    /// Build a minimal end-to-end tiny model. Note: `internal_dim`
    /// for the RVQ is also the SeaNet dim here so we can skip the
    /// projections.
    fn build_tiny_model(n_q: usize, resampler_stride: usize) -> MimiEncodecModel {
        let mut nb = rng_seed(2026);
        let seanet = tiny_seanet_cfg();
        let mut tcfg = tiny_transformer_cfg();
        tcfg.d_model = seanet.dimension;
        let inner_cfg = MimiConfig {
            seanet: seanet.clone(),
            transformer: tcfg.clone(),
            channels: seanet.channels,
            n_q,
            quantizer_bins: 4,
            quantizer_dim: seanet.dimension,
        };
        let encoder_w = build_seanet_encoder(&seanet, &mut nb);
        let decoder_w = build_seanet_decoder(&seanet, &mut nb);
        let enc_xformer_w = build_projected_transformer(&tcfg, &mut nb);
        let dec_xformer_w = build_projected_transformer(&tcfg, &mut nb);
        let dim = seanet.dimension;
        // Downsample / upsample weight buffers (kernel = 2·stride).
        let dn_w_len = dim * dim * (2 * resampler_stride);
        let up_w_len = dim * 1 * (2 * resampler_stride);
        let downsample_weight: Arc<[f32]> = vec_of(dn_w_len, &mut nb);
        let upsample_weight: Arc<[f32]> = vec_of(up_w_len, &mut nb);
        let quantizer = SplitResidualVectorQuantizerWeights {
            rvq_first: tiny_rvq_quantizer(1, dim, dim, 4, &mut nb),
            rvq_rest: tiny_rvq_quantizer(n_q.saturating_sub(1).max(1), dim, dim, 4, &mut nb),
            n_q,
        };
        let weights = MimiWeights {
            encoder: encoder_w,
            decoder: decoder_w,
            encoder_transformer: enc_xformer_w,
            decoder_transformer: dec_xformer_w,
            downsample_weight,
            upsample_weight,
            quantizer,
            resampler_stride,
        };
        // Synthesize a config whose derived stride matches.
        // ratios_product = 4, so encoder_frame_rate = sample_rate / 4.
        // For resampler_stride = 2 we need frame_rate = encoder_frame_rate / 2.
        let sample_rate = 4.0 * 2.0 * resampler_stride as f64;
        let encoder_frame_rate = sample_rate / 4.0;
        let frame_rate = encoder_frame_rate / resampler_stride as f64;
        let config = MimiEncodecConfig {
            channels: 1,
            sample_rate,
            frame_rate,
            renormalize: false,
            resample_method: ResampleMethod::Conv,
            inner: inner_cfg,
        };
        MimiEncodecModel::new(config, weights).expect("tiny build")
    }

    #[test]
    fn v0_1_config_matches_kyutai_canonical_dims() {
        let cfg = MimiEncodecConfig::v0_1(None);
        // Top-level Kyutai canonical dims (eager `Config::v0_1`).
        assert_eq!(cfg.channels, 1);
        assert_eq!(cfg.sample_rate, 24_000.0);
        assert_eq!(cfg.frame_rate, 12.5);
        assert!(cfg.renormalize);
        assert_eq!(cfg.resample_method, ResampleMethod::Conv);
        assert_eq!(cfg.inner.n_q, 16);
        assert_eq!(cfg.inner.quantizer_bins, 2048);
        assert_eq!(cfg.inner.quantizer_dim, 256);
        assert_eq!(cfg.inner.seanet.dimension, 512);
        assert_eq!(cfg.inner.transformer.d_model, 512);
        assert_eq!(cfg.inner.transformer.num_heads, 8);
        assert_eq!(cfg.inner.transformer.num_layers, 8);
        assert_eq!(cfg.inner.seanet.ratios, vec![8, 6, 5, 4]);

        // Derived framing: 24_000 / (8·6·5·4) = 24_000 / 960 = 25 Hz
        // encoder rate; 25 / 12.5 = 2 resampler stride.
        let efr = cfg.encoder_frame_rate();
        assert!((efr - 25.0).abs() < 1e-9, "encoder_frame_rate = {efr}");
        assert_eq!(cfg.resampler_stride(), 2);
        // total audio→code stride = 960 · 2 = 1920.
        assert_eq!(cfg.total_audio_stride(), 1920);
    }

    #[test]
    fn v0_1_codebook_override_propagates() {
        let cfg = MimiEncodecConfig::v0_1(Some(8));
        assert_eq!(cfg.inner.n_q, 8);
        // Other fields unchanged.
        assert_eq!(cfg.inner.quantizer_bins, 2048);
        assert_eq!(cfg.sample_rate, 24_000.0);
    }

    #[test]
    fn new_rejects_stride_mismatch() {
        let model = build_tiny_model(2, 2);
        let mut bad = model.config.clone();
        bad.frame_rate *= 2.0; // Halves the derived stride.
        let mut bad_weights = model.inner.weights.clone();
        bad_weights.resampler_stride = 2;
        let err = MimiEncodecModel::new(bad, bad_weights);
        assert!(err.is_err());
    }

    #[test]
    fn tiny_encode_decode_round_trip_shape() {
        let n_q = 2;
        let resampler_stride = 2;
        let model = build_tiny_model(n_q, resampler_stride);
        let total_stride = model.config.total_audio_stride();
        // A "few-second" synthetic PCM scaled to the tiny config —
        // the only invariant is divisibility by total_stride.
        let t_audio = total_stride * 6;
        let pcm: Vec<f32> = (0..t_audio).map(|i| ((i as f32) * 0.01).sin() * 0.1).collect();
        let pcm = LazyTensor::from_f32(
            pcm, Shape::from_dims(&[1, 1, t_audio]), &Device::cpu(),
        );
        let codes = model.encode(&pcm).unwrap();
        let code_dims = codes.shape().dims().to_vec();
        assert_eq!(code_dims, vec![1, n_q, t_audio / total_stride]);
        let recon = model.decode(&codes).unwrap();
        let recon_dims = recon.shape().dims().to_vec();
        assert_eq!(recon_dims, vec![1, 1, t_audio]);
        for &v in &recon.realize_f32() {
            assert!(v.is_finite(), "recon contains non-finite sample");
        }
    }

    #[test]
    fn streaming_decode_equals_one_shot() {
        let n_q = 2;
        let resampler_stride = 2;
        let model = build_tiny_model(n_q, resampler_stride);
        let total_stride = model.config.total_audio_stride();
        // Build a small code sequence directly so the test only
        // exercises the decode streaming path. 4 code frames.
        let t_codes = 4;
        let codes: Vec<u32> = (0..(1 * n_q * t_codes)).map(|i| (i as u32) % 4).collect();
        let codes_t = LazyTensor::from_u32(
            codes, Shape::from_dims(&[1, n_q, t_codes]), &Device::cpu(),
        );

        let one_shot = model.decode(&codes_t).unwrap().realize_f32();
        assert_eq!(one_shot.len(), t_codes * total_stride);

        // Stream one code frame at a time.
        let mut state = MimiEncodecDecodeState::empty();
        let mut emitted = Vec::with_capacity(one_shot.len());
        for t in 0..t_codes {
            let chunk = codes_t.narrow(2_usize, t, 1).unwrap();
            let (new_state, out) = model.decode_step(state, &chunk).unwrap();
            state = new_state;
            if let Some(y) = out {
                let v = y.realize_f32();
                emitted.extend(v);
            }
        }
        assert_eq!(emitted.len(), one_shot.len(),
            "streamed length must equal one-shot length");
        // Tolerance covers re-association inside the conv stack — the
        // streaming path re-runs forward on the cumulative input, so
        // results match up to floating-point noise.
        for (i, (s, o)) in emitted.iter().zip(one_shot.iter()).enumerate() {
            assert!((s - o).abs() < 1e-5,
                "streaming/one-shot mismatch at {i}: streaming={s} one_shot={o}");
        }
    }

    #[test]
    fn streaming_encode_equals_one_shot() {
        let n_q = 2;
        let resampler_stride = 2;
        let model = build_tiny_model(n_q, resampler_stride);
        let total_stride = model.config.total_audio_stride();
        // 3 frames of synthetic PCM.
        let t_codes = 3;
        let t_audio = t_codes * total_stride;
        let pcm: Vec<f32> = (0..t_audio).map(|i| ((i as f32) * 0.03).cos() * 0.1).collect();
        let pcm_t = LazyTensor::from_f32(
            pcm.clone(), Shape::from_dims(&[1, 1, t_audio]), &Device::cpu(),
        );
        let one_shot = model.encode(&pcm_t).unwrap().realize_u32();
        assert_eq!(one_shot.len(), 1 * n_q * t_codes);

        // Stream PCM in chunks smaller than total_stride so the
        // first chunk yields no codes.
        let chunk_size = total_stride / 2;
        assert!(chunk_size >= 1);
        let mut state = MimiEncodecEncodeState::empty();
        // Per-codebook collected frame rows so we can reassemble the
        // (q, frame) layout the one-shot realize produces.
        let mut per_q: Vec<Vec<u32>> = vec![Vec::new(); n_q];
        let mut cursor = 0;
        while cursor < t_audio {
            let take = chunk_size.min(t_audio - cursor);
            let chunk_data: Vec<f32> = pcm[cursor..cursor + take].to_vec();
            let chunk = LazyTensor::from_f32(
                chunk_data, Shape::from_dims(&[1, 1, take]), &Device::cpu(),
            );
            let (new_state, out) = model.encode_step(state, &chunk).unwrap();
            state = new_state;
            if let Some(codes) = out {
                let dims = codes.shape().dims().to_vec();
                let new_frames = dims[2];
                let flat = codes.realize_u32();
                // codes are (1, n_q, new_frames) row-major. Append each
                // codebook row to its per-q bucket.
                for q in 0..n_q {
                    let row = &flat[q * new_frames..(q + 1) * new_frames];
                    per_q[q].extend_from_slice(row);
                }
            }
            cursor += take;
        }
        // Flatten (q, frame) → matches the one-shot (n_q, t_codes)
        // row-major layout.
        let mut emitted: Vec<u32> = Vec::with_capacity(n_q * t_codes);
        for q in 0..n_q {
            assert_eq!(per_q[q].len(), t_codes);
            emitted.extend_from_slice(&per_q[q]);
        }
        assert_eq!(emitted.len(), one_shot.len());
        for (i, (s, o)) in emitted.iter().zip(one_shot.iter()).enumerate() {
            assert_eq!(s, o, "streaming/one-shot encode mismatch at {i}");
        }
    }
}

//! Implementation of the Descript Audio Codec (DAC) model
//!
//! See: [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)
//!
//! DAC is an efficient neural audio codec that compresses waveforms into discrete
//! residual codes and reconstructs them with high fidelity.

use crate::models::encodec;
use candle::{IndexOp, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

/// Configuration parameters for a DAC model.
///
/// # Example
///
/// ```no_run
/// # let cfg: candle_transformers::models::dac::Config = unimplemented!(); // Typically loaded from JSON
/// ```
#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    /// Number of residual codebooks used in the quantizer.
    pub num_codebooks: usize,
    /// Target bit-rate in kbps.
    pub model_bitrate: u32,
    /// Number of entries per codebook.
    pub codebook_size: usize,
    /// Dimension of each codebook embedding vector.
    pub latent_dim: usize,
    /// Frame rate at the encoder output (in Hz).
    pub frame_rate: u32,
    /// Input audio sampling rate (in Hz).
    pub sampling_rate: u32,
}

/// Learnable periodic activation function used in place of ReLU throughout DAC.
///
/// $ \text{Snake}(x) = x + \frac{\sin^2(\alpha x)}{\alpha} $
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::dac::Snake1d;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let snake = Snake1d::new(64, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    /// Load the `alpha` parameter for `channels` channels from a [`VarBuilder`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::dac::Snake1d;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let snake = Snake1d::new(64, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }
}

impl candle::Module for Snake1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_shape = xs.shape();
        let xs = xs.flatten_from(2)?;
        let sin = self.alpha.broadcast_mul(&xs)?.sin()?;
        let sin = (&sin * &sin)?;
        (xs + (&self.alpha + 1e-9)?.recip()?.broadcast_mul(&sin)?)?.reshape(xs_shape)
    }
}

/// Dilated residual unit with two weight-normed convolutions.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::dac::ResidualUnit;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let unit = ResidualUnit::new(64, 1, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct ResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d,
    snake2: Snake1d,
    conv2: Conv1d,
}

impl ResidualUnit {
    /// Build a [`ResidualUnit`] with the given channel `dim` and `dilation`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::dac::ResidualUnit;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let unit = ResidualUnit::new(64, 1, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let pad = ((7 - 1) * dilation) / 2;
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(dim, vb.pp(0))?;
        let cfg1 = Conv1dConfig {
            dilation,
            padding: pad,
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(dim, dim, 7, cfg1, vb.pp(1))?;
        let snake2 = Snake1d::new(dim, vb.pp(2))?;
        let conv2 = encodec::conv1d_weight_norm(dim, dim, 1, Default::default(), vb.pp(3))?;
        Ok(Self {
            snake1,
            conv1,
            snake2,
            conv2,
        })
    }
}

impl candle::Module for ResidualUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = xs
            .apply(&self.snake1)?
            .apply(&self.conv1)?
            .apply(&self.snake2)?
            .apply(&self.conv2)?;
        let pad = (xs.dim(D::Minus1)? - ys.dim(D::Minus1)?) / 2;
        if pad > 0 {
            &ys + xs.narrow(D::Minus1, pad, ys.dim(D::Minus1)?)
        } else {
            ys + xs
        }
    }
}

/// One strided-downsampling block in the DAC encoder: 3 residual units + strided conv.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::dac::EncoderBlock;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let block = EncoderBlock::new(128, 4, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct EncoderBlock {
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
    snake1: Snake1d,
    conv1: Conv1d,
}

impl EncoderBlock {
    /// Build an [`EncoderBlock`] that doubles the channel count and downsamples by `stride`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::dac::EncoderBlock;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let block = EncoderBlock::new(128, 4, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let res1 = ResidualUnit::new(dim / 2, 1, vb.pp(0))?;
        let res2 = ResidualUnit::new(dim / 2, 3, vb.pp(1))?;
        let res3 = ResidualUnit::new(dim / 2, 9, vb.pp(2))?;
        let snake1 = Snake1d::new(dim / 2, vb.pp(3))?;
        let cfg1 = Conv1dConfig {
            stride,
            padding: stride.div_ceil(2),
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(dim / 2, dim, 2 * stride, cfg1, vb.pp(4))?;
        Ok(Self {
            res1,
            res2,
            res3,
            snake1,
            conv1,
        })
    }
}

impl candle::Module for EncoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.res1)?
            .apply(&self.res2)?
            .apply(&self.res3)?
            .apply(&self.snake1)?
            .apply(&self.conv1)
    }
}

/// Full DAC encoder stack: input conv → N [`EncoderBlock`]s → output conv → latent projection.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::dac::Encoder;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let encoder = Encoder::new(64, &[2, 4, 8, 8], 64, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Encoder {
    conv1: Conv1d,
    blocks: Vec<EncoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl candle::Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in self.blocks.iter() {
            xs = xs.apply(block)?
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)
    }
}

impl Encoder {
    /// Build the encoder. `strides` controls each block's downsampling factor;
    /// the channel count doubles at each stage, starting from `d_model`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::dac::Encoder;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let encoder = Encoder::new(64, &[2, 4, 8, 8], 64, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(
        mut d_model: usize,
        strides: &[usize],
        d_latent: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("block");
        let cfg1 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(1, d_model, 7, cfg1, vb.pp(0))?;
        let mut blocks = Vec::with_capacity(strides.len());
        for (block_idx, stride) in strides.iter().enumerate() {
            d_model *= 2;
            let block = EncoderBlock::new(d_model, *stride, vb.pp(block_idx + 1))?;
            blocks.push(block)
        }
        let snake1 = Snake1d::new(d_model, vb.pp(strides.len() + 1))?;
        let cfg2 = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv2 =
            encodec::conv1d_weight_norm(d_model, d_latent, 3, cfg2, vb.pp(strides.len() + 2))?;
        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }
}

/// One up-sampling block in the DAC decoder: transposed conv + 3 [`ResidualUnit`]s.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::dac::DecoderBlock;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let block = DecoderBlock::new(128, 64, 4, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct DecoderBlock {
    snake1: Snake1d,
    conv_tr1: ConvTranspose1d,
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
}

impl DecoderBlock {
    /// Build a [`DecoderBlock`] that upsamples from `in_dim` to `out_dim` by `stride`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::dac::DecoderBlock;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let block = DecoderBlock::new(128, 64, 4, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(in_dim: usize, out_dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(in_dim, vb.pp(0))?;
        let cfg = ConvTranspose1dConfig {
            stride,
            padding: stride.div_ceil(2),
            ..Default::default()
        };
        let conv_tr1 = encodec::conv_transpose1d_weight_norm(
            in_dim,
            out_dim,
            2 * stride,
            true,
            cfg,
            vb.pp(1),
        )?;
        let res1 = ResidualUnit::new(out_dim, 1, vb.pp(2))?;
        let res2 = ResidualUnit::new(out_dim, 3, vb.pp(3))?;
        let res3 = ResidualUnit::new(out_dim, 9, vb.pp(4))?;
        Ok(Self {
            snake1,
            conv_tr1,
            res1,
            res2,
            res3,
        })
    }
}

impl candle_nn::Module for DecoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.snake1)?
            .apply(&self.conv_tr1)?
            .apply(&self.res1)?
            .apply(&self.res2)?
            .apply(&self.res3)
    }
}

/// Full DAC decoder stack: input conv → N [`DecoderBlock`]s → output conv.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::dac::Decoder;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let decoder = Decoder::new(64, 1536, &[8, 8, 4, 2], 1, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Decoder {
    conv1: Conv1d,
    blocks: Vec<DecoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl Decoder {
    /// Build the decoder. `rates` controls each block's upsampling factor.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::dac::Decoder;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let decoder = Decoder::new(64, 1536, &[8, 8, 4, 2], 1, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(
        in_c: usize,
        mut channels: usize,
        rates: &[usize],
        d_out: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("model");
        let cfg1 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(in_c, channels, 7, cfg1, vb.pp(0))?;
        let mut blocks = Vec::with_capacity(rates.len());
        for (idx, stride) in rates.iter().enumerate() {
            let block = DecoderBlock::new(channels, channels / 2, *stride, vb.pp(idx + 1))?;
            channels /= 2;
            blocks.push(block)
        }
        let snake1 = Snake1d::new(channels, vb.pp(rates.len() + 1))?;
        let conv2 = encodec::conv1d_weight_norm(channels, d_out, 7, cfg1, vb.pp(rates.len() + 2))?;
        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }
}

impl candle::Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in self.blocks.iter() {
            xs = xs.apply(block)?
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)
    }
}

/// Single-codebook vector quantizer used by DAC.
///
/// Projects the input to codebook dimension, finds the nearest entry, and
/// projects the result back to the original dimension.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::dac::VectorQuantizer;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let vq = VectorQuantizer::new(64, 1024, 8, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[allow(unused)]
#[derive(Clone, Debug)]
pub struct VectorQuantizer {
    in_proj: Conv1d,
    out_proj: Conv1d,
    codebook: candle_nn::Embedding,
}

impl VectorQuantizer {
    /// Load a [`VectorQuantizer`] with input dimension `in_dim`, `cb_size` codebook entries
    /// each of dimension `cb_dim`, from a [`VarBuilder`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::dac::VectorQuantizer;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let vq = VectorQuantizer::new(64, 1024, 8, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(in_dim: usize, cb_size: usize, cb_dim: usize, vb: VarBuilder) -> Result<Self> {
        let in_proj =
            encodec::conv1d_weight_norm(in_dim, cb_dim, 1, Default::default(), vb.pp("in_proj"))?;
        let out_proj =
            encodec::conv1d_weight_norm(cb_dim, in_dim, 1, Default::default(), vb.pp("out_proj"))?;
        let codebook = candle_nn::embedding(cb_size, cb_dim, vb.pp("codebook"))?;
        Ok(Self {
            in_proj,
            out_proj,
            codebook,
        })
    }

    /// Look up the embedding vectors for the given codebook indices.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::dac::VectorQuantizer;
    /// # use candle::{Device, DType, Tensor};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let vq = VectorQuantizer::new(64, 1024, 8, vb)?;
    /// let ids = Tensor::zeros(16_usize, DType::U32, &Device::Cpu)?;
    /// let emb = vq.embed_code(&ids)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn embed_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        embed_id.apply(&self.codebook)
    }

    /// Decode codebook indices to feature vectors `(batch, channels, time)`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::dac::VectorQuantizer;
    /// # use candle::{Device, DType, Tensor};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let vq = VectorQuantizer::new(64, 1024, 8, vb)?;
    /// let ids = Tensor::zeros(16_usize, DType::U32, &Device::Cpu)?;
    /// let feats = vq.decode_code(&ids)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn decode_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        self.embed_code(embed_id)?.transpose(1, 2)
    }
}

/// Residual vector quantizer: a stack of [`VectorQuantizer`]s.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::dac::ResidualVectorQuantizer;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let rvq = ResidualVectorQuantizer::new(64, 12, 1024, 8, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct ResidualVectorQuantizer {
    quantizers: Vec<VectorQuantizer>,
}

impl ResidualVectorQuantizer {
    /// Load `n_codebooks` quantizer layers.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::dac::ResidualVectorQuantizer;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let rvq = ResidualVectorQuantizer::new(64, 12, 1024, 8, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(
        input_dim: usize,
        n_codebooks: usize,
        cb_size: usize,
        cb_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = &vb.pp("quantizers");
        let quantizers = (0..n_codebooks)
            .map(|i| VectorQuantizer::new(input_dim, cb_size, cb_dim, vb.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { quantizers })
    }

    /// Accumulate quantized embeddings from all codebooks for the given code matrix.
    ///
    /// * `codes` – integer tensor of shape `(batch, n_codebooks, time)`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::dac::ResidualVectorQuantizer;
    /// # use candle::{Device, DType, Tensor};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let rvq = ResidualVectorQuantizer::new(64, 12, 1024, 8, vb)?;
    /// let codes = Tensor::zeros((1, 12, 50), DType::U32, &Device::Cpu)?;
    /// let feats = rvq.from_codes(&codes)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    #[allow(clippy::wrong_self_convention)]
    pub fn from_codes(&self, codes: &Tensor) -> Result<Tensor> {
        let mut sum = None;
        for (idx, quantizer) in self.quantizers.iter().enumerate() {
            let z_p_i = quantizer.decode_code(&codes.i((.., idx))?)?;
            let z_q_i = z_p_i.apply(&quantizer.out_proj)?;
            let s = match sum {
                None => z_q_i,
                Some(s) => (s + z_q_i)?,
            };
            sum = Some(s)
        }
        match sum {
            Some(s) => Ok(s),
            None => candle::bail!("empty codebooks"),
        }
    }
}

/// The complete DAC model: [`Encoder`] + [`ResidualVectorQuantizer`] + [`Decoder`].
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::dac::{Config, Model};
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// # let cfg: Config = unimplemented!();
/// let model = Model::new(&cfg, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Model {
    pub encoder: Encoder,
    pub quantizer: ResidualVectorQuantizer,
    pub decoder: Decoder,
}

impl Model {
    /// Build the DAC model from a [`Config`] and a [`VarBuilder`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::dac::{Config, Model};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// # let cfg: Config = unimplemented!();
    /// let model = Model::new(&cfg, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(64, &[2, 4, 8, 8], cfg.latent_dim, vb.pp("encoder"))?;
        let quantizer = ResidualVectorQuantizer::new(
            cfg.latent_dim,
            cfg.num_codebooks,
            cfg.codebook_size,
            8,
            vb.pp("quantizer"),
        )?;
        let decoder = Decoder::new(cfg.latent_dim, 1536, &[8, 8, 4, 2], 1, vb.pp("decoder"))?;
        Ok(Self {
            encoder,
            decoder,
            quantizer,
        })
    }

    /// Decode discrete codes `(batch, n_codebooks, frames)` back to a waveform.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::dac::{Config, Model};
    /// # use candle::{Device, DType, Tensor};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// # let cfg: Config = unimplemented!();
    /// let model = Model::new(&cfg, vb)?;
    /// let codes = Tensor::zeros((1, 12, 50), DType::U32, &Device::Cpu)?;
    /// let audio = model.decode_codes(&codes)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn decode_codes(&self, audio_codes: &Tensor) -> Result<Tensor> {
        let audio_values = self.quantizer.from_codes(audio_codes)?;
        audio_values.apply(&self.decoder)
    }
}

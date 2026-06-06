//! Descript Audio Codec (DAC) — lazy port.
//!
//! Discrete codes `(batch, n_codebooks, time)` → waveform
//! `(batch, 1, time_out)` via:
//!   1. Residual vector quantizer reconstruction:
//!      `sum_i quantizer_i.out_proj(codebook_i[codes[:, i]])`.
//!   2. Decoder: initial Conv1d → N DecoderBlocks (each is
//!      Snake → ConvTranspose1d → 3 ResidualUnits with
//!      dilations 1, 3, 9) → Snake → final Conv1d.
//!
//! The Snake activation is a learnable periodic nonlinearity
//! `x + sin²(α·x) / (α + 1e-9)` with a per-channel `α`.
//!
//! Dilation handling: `LazyTensor::conv1d` doesn't yet take a
//! `dilation` parameter. Since DAC's dilated convs (k=7, d∈{1,3,9})
//! all use **constant** kernel weights, we lift dilation into the
//! weight tensor: expand `[Cout, Cin, K]` → `[Cout, Cin, K + (K-1)·(D-1)]`
//! by interleaving (D-1) zeros between adjacent kernel taps, then
//! call a plain (non-dilated) conv1d. This is mathematically
//! equivalent and incurs no runtime overhead beyond a larger
//! constant — typical DAC residual units only inflate `K=7` to at
//! most `K' = 7 + 6·8 = 55` for the deepest dilation = 9.
//!
//! v1 scope:
//!   - **Decoder-only path** (`decode_codes(codes) → audio`). The
//!     encoder is symmetric and not required for inference.
//!   - F32 weights and activations.
//!   - `batch == 1`.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

/// DAC config. The standard preset (`Config::default_preset`)
/// uses 12 codebooks, codebook_size 1024, latent_dim 1024.
#[derive(Debug, Clone, PartialEq)]
pub struct DacConfig {
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub latent_dim: usize,
    /// Mirrors eager `Decoder::new(64, 1536, &[8, 8, 4, 2], 1, ...)`.
    pub decoder_initial_channels: usize,
    pub decoder_rates: Vec<usize>,
    pub decoder_out_channels: usize,
}

impl DacConfig {
    /// Standard 44.1 kHz DAC preset.
    pub fn default_preset() -> Self {
        Self {
            num_codebooks: 12,
            codebook_size: 1024,
            codebook_dim: 8,
            latent_dim: 1024,
            decoder_initial_channels: 1536,
            decoder_rates: vec![8, 8, 4, 2],
            decoder_out_channels: 1,
        }
    }
}

// ---- Weight structs --------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Conv1dWeights {
    /// `[Cout, Cin, K]` (or, post-dilation expansion, `[Cout, Cin, K + (K-1)·(D-1)]`).
    pub w: Arc<[f32]>,
    pub b: Option<Arc<[f32]>>,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub dilation: usize,
}

#[derive(Debug, Clone)]
pub struct ConvTranspose1dWeights {
    /// `[Cin, Cout, K]` (PyTorch convention).
    pub w: Arc<[f32]>,
    pub b: Option<Arc<[f32]>>,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
}

#[derive(Debug, Clone)]
pub struct Snake1dWeights {
    /// `[1, C, 1]`.
    pub alpha: Arc<[f32]>,
    pub channels: usize,
}

#[derive(Debug, Clone)]
pub struct ResidualUnitWeights {
    pub snake1: Snake1dWeights,
    pub conv1: Conv1dWeights,
    pub snake2: Snake1dWeights,
    pub conv2: Conv1dWeights,
}

#[derive(Debug, Clone)]
pub struct DecoderBlockWeights {
    pub snake1: Snake1dWeights,
    pub conv_tr1: ConvTranspose1dWeights,
    pub res1: ResidualUnitWeights,
    pub res2: ResidualUnitWeights,
    pub res3: ResidualUnitWeights,
}

#[derive(Debug, Clone)]
pub struct DecoderWeights {
    pub conv1: Conv1dWeights,
    pub blocks: Vec<DecoderBlockWeights>,
    pub snake1: Snake1dWeights,
    pub conv2: Conv1dWeights,
}

#[derive(Debug, Clone)]
pub struct VectorQuantizerWeights {
    pub in_proj: Conv1dWeights,
    pub out_proj: Conv1dWeights,
    /// `[codebook_size, codebook_dim]` — embedded as a const tensor at lookup time.
    pub codebook: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct DacWeights {
    pub quantizers: Vec<VectorQuantizerWeights>,
    pub decoder: DecoderWeights,
}

#[derive(Debug, Clone)]
pub struct DacModel {
    pub config: DacConfig,
    pub weights: DacWeights,
}

// ---- Forward ---------------------------------------------------------------

impl DacModel {
    /// Decode discrete codes back to a waveform.
    ///
    /// * `codes` — U32 LazyTensor of shape `(1, num_codebooks, time)`.
    /// * Returns F32 audio `(1, decoder_out_channels, time_out)` where
    ///   `time_out = time · prod(decoder_rates)` modulo per-stage
    ///   conv padding edge effects.
    pub fn decode_codes(&self, codes: &LazyTensor) -> Result<LazyTensor> {
        let dims = codes.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "codes must be rank 3 [B, num_codebooks, T]");
        assert_eq!(dims[0], 1, "v1 supports batch == 1");
        assert_eq!(
            dims[1], self.config.num_codebooks,
            "codes must have {} codebooks, got {}",
            self.config.num_codebooks, dims[1],
        );
        let latent = self.rvq_from_codes(codes)?;
        self.decoder_forward(&latent)
    }

    /// `latent_sum = sum_i quantizers[i].out_proj(codebook_i[codes[:, i]])`.
    fn rvq_from_codes(&self, codes: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = codes.shape();
        let dims = dims.dims();
        let time = dims[2];
        let mut sum: Option<LazyTensor> = None;
        for (idx, q) in self.weights.quantizers.iter().enumerate() {
            // codes[:, idx, :] → (1, T) U32.
            let ids = codes
                .narrow(1_usize, idx, 1)?
                .reshape(Shape::from_dims(&[time]))?;
            // Embedding lookup: codebook[ids] → (T, codebook_dim).
            let codebook = codes.const_f32_like(
                Arc::clone(&q.codebook),
                Shape::from_dims(&[cfg.codebook_size, cfg.codebook_dim]),
            );
            let z_p = codebook
                .index_select(0_usize, &ids)?
                .reshape(Shape::from_dims(&[1, time, cfg.codebook_dim]))?
                .permute([0, 2, 1_usize])?;
            // out_proj: codebook_dim → latent_dim, k=1.
            let z_q = apply_conv1d(&z_p, &q.out_proj, codes)?;
            sum = Some(match sum {
                None => z_q,
                Some(s) => s.add(&z_q)?,
            });
        }
        sum.ok_or_else(|| {
            fuel_core_types::Error::Msg("DAC RVQ: no codebooks".into()).bt()
        })
    }

    fn decoder_forward(&self, latent: &LazyTensor) -> Result<LazyTensor> {
        let dec = &self.weights.decoder;
        let mut x = apply_conv1d(latent, &dec.conv1, latent)?;
        for block in &dec.blocks {
            x = apply_decoder_block(&x, block, latent)?;
        }
        x = apply_snake1d(&x, &dec.snake1, latent)?;
        apply_conv1d(&x, &dec.conv2, latent)
    }
}

// ---- Component helpers -----------------------------------------------------

/// Build the dilation-expanded weight tensor: pad each kernel tap
/// with (dilation - 1) zeros between consecutive taps. For
/// dilation = 1 this is the identity.
///
/// Shared infrastructure: also used by the EnCodec / SNAC / Mimi
/// audio codec ports — they all dispatch dilated Conv1d through
/// the plain Conv1d primitive using this weight-expansion trick.
pub fn expand_conv1d_weight_for_dilation_if_needed(
    w: &[f32], c_out: usize, c_in: usize, k: usize, dilation: usize,
) -> (Vec<f32>, usize) {
    expand_conv1d_weight_for_dilation(w, c_out, c_in, k, dilation)
}

fn expand_conv1d_weight_for_dilation(
    w: &[f32], c_out: usize, c_in: usize, k: usize, dilation: usize,
) -> (Vec<f32>, usize) {
    if dilation <= 1 {
        return (w.to_vec(), k);
    }
    let k_expanded = k + (k - 1) * (dilation - 1);
    let mut out = vec![0.0_f32; c_out * c_in * k_expanded];
    for o in 0..c_out {
        for i in 0..c_in {
            for j in 0..k {
                let src = (o * c_in + i) * k + j;
                let dst = (o * c_in + i) * k_expanded + j * dilation;
                out[dst] = w[src];
            }
        }
    }
    (out, k_expanded)
}

fn apply_conv1d(
    x: &LazyTensor,
    c: &Conv1dWeights,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let (w_data, k_eff) = expand_conv1d_weight_for_dilation(
        &c.w, c.c_out, c.c_in, c.k, c.dilation,
    );
    let w = anchor.const_f32_like(
        Arc::<[f32]>::from(w_data),
        Shape::from_dims(&[c.c_out, c.c_in, k_eff]),
    );
    let bias = c.b.as_ref().map(|b| {
        anchor.const_f32_like(Arc::clone(b), Shape::from_dims(&[c.c_out]))
    });
    x.conv1d(&w, bias.as_ref(), c.stride, c.pad, 1)
}

fn apply_conv_transpose1d(
    x: &LazyTensor,
    c: &ConvTranspose1dWeights,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let w = anchor.const_f32_like(
        Arc::clone(&c.w),
        Shape::from_dims(&[c.c_in, c.c_out, c.k]),
    );
    let mut out = x.conv_transpose1d(&w, c.stride, c.pad, 0, 1, 1)?;
    if let Some(b) = &c.b {
        let bias = anchor
            .const_f32_like(Arc::clone(b), Shape::from_dims(&[c.c_out]))
            .reshape(Shape::from_dims(&[1, c.c_out, 1]))?;
        out = out.broadcast_add(&bias)?;
    }
    Ok(out)
}

/// `Snake(x) = x + sin²(α · x) / (α + 1e-9)` with per-channel α.
fn apply_snake1d(
    x: &LazyTensor,
    s: &Snake1dWeights,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    assert_eq!(dims[1], s.channels,
        "snake1d: channel mismatch {} vs {}", dims[1], s.channels);
    let alpha = anchor
        .const_f32_like(Arc::clone(&s.alpha), Shape::from_dims(&[s.channels]))
        .reshape(Shape::from_dims(&[1, s.channels, 1]))?
        .broadcast_to(Shape::from_dims(dims))?;
    let scaled = x.mul(&alpha)?;
    let sin_v = scaled.sin();
    let sin_sq = sin_v.mul(&sin_v)?;
    let alpha_eps = alpha.add_scalar(1e-9);
    let recip = alpha_eps.recip();
    let correction = recip.mul(&sin_sq)?;
    x.add(&correction)
}

fn apply_residual_unit(
    x: &LazyTensor,
    r: &ResidualUnitWeights,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let y = apply_snake1d(x, &r.snake1, anchor)?;
    let y = apply_conv1d(&y, &r.conv1, anchor)?;
    let y = apply_snake1d(&y, &r.snake2, anchor)?;
    let y = apply_conv1d(&y, &r.conv2, anchor)?;
    // Eager `ResidualUnit::forward` narrows xs to ys.len() along
    // the last dim and adds. Our padding scheme keeps the lengths
    // equal in the common path; if they diverge, narrow.
    let y_t = y.shape().dims()[2];
    let x_t = x.shape().dims()[2];
    if x_t == y_t {
        x.add(&y)
    } else {
        let pad = (x_t - y_t) / 2;
        let x_narrow = x.narrow(2_usize, pad, y_t)?;
        x_narrow.add(&y)
    }
}

fn apply_decoder_block(
    x: &LazyTensor,
    b: &DecoderBlockWeights,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let y = apply_snake1d(x, &b.snake1, anchor)?;
    let y = apply_conv_transpose1d(&y, &b.conv_tr1, anchor)?;
    let y = apply_residual_unit(&y, &b.res1, anchor)?;
    let y = apply_residual_unit(&y, &b.res2, anchor)?;
    apply_residual_unit(&y, &b.res3, anchor)
}

// ---- Safetensors loader ----------------------------------------------------

/// Fuse weight-norm parameters for a Conv1d into a plain `[c_out, c_in, k]`
/// weight tensor: `w[o, i, j] = weight_v[o, i, j] · weight_g[o] / norm_v[o]`
/// where `norm_v[o] = sqrt(sum_{i, j} weight_v[o, i, j]^2)`.
///
/// `weight_v` is stored as `[c_out, c_in, k]` and `weight_g` as `[c_out, 1, 1]`
/// (flattened to length `c_out`).
fn fuse_weight_norm_conv1d(
    weight_v: &[f32], weight_g: &[f32], c_out: usize, c_in: usize, k: usize,
) -> crate::Result<Vec<f32>> {
    if weight_v.len() != c_out * c_in * k {
        crate::bail!(
            "fuse_weight_norm_conv1d: weight_v has {} elements, expected {} ({c_out}×{c_in}×{k})",
            weight_v.len(), c_out * c_in * k,
        );
    }
    if weight_g.len() != c_out {
        crate::bail!(
            "fuse_weight_norm_conv1d: weight_g has {} elements, expected {c_out}",
            weight_g.len(),
        );
    }
    let mut out = vec![0.0_f32; c_out * c_in * k];
    for o in 0..c_out {
        let base = o * c_in * k;
        let mut sumsq = 0.0_f64;
        for j in 0..c_in * k {
            let v = weight_v[base + j] as f64;
            sumsq += v * v;
        }
        let norm = sumsq.sqrt().max(1e-12) as f32;
        let scale = weight_g[o] / norm;
        for j in 0..c_in * k {
            out[base + j] = weight_v[base + j] * scale;
        }
    }
    Ok(out)
}

/// Same fusion as [`fuse_weight_norm_conv1d`] but with the storage
/// convention used by `ConvTranspose1d`: `weight_v` is `[c_in, c_out, k]`,
/// `weight_g` is `[c_in, 1, 1]`. The normalisation is over the last two
/// dims (per-input-channel), matching the eager
/// `weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?` recipe.
fn fuse_weight_norm_conv_transpose1d(
    weight_v: &[f32], weight_g: &[f32], c_in: usize, c_out: usize, k: usize,
) -> crate::Result<Vec<f32>> {
    if weight_v.len() != c_in * c_out * k {
        crate::bail!(
            "fuse_weight_norm_conv_transpose1d: weight_v has {} elements, expected {} ({c_in}×{c_out}×{k})",
            weight_v.len(), c_in * c_out * k,
        );
    }
    if weight_g.len() != c_in {
        crate::bail!(
            "fuse_weight_norm_conv_transpose1d: weight_g has {} elements, expected {c_in}",
            weight_g.len(),
        );
    }
    let mut out = vec![0.0_f32; c_in * c_out * k];
    for i in 0..c_in {
        let base = i * c_out * k;
        let mut sumsq = 0.0_f64;
        for j in 0..c_out * k {
            let v = weight_v[base + j] as f64;
            sumsq += v * v;
        }
        let norm = sumsq.sqrt().max(1e-12) as f32;
        let scale = weight_g[i] / norm;
        for j in 0..c_out * k {
            out[base + j] = weight_v[base + j] * scale;
        }
    }
    Ok(out)
}

/// Load one weight-normed `Conv1dWeights` at the given safetensors path
/// prefix. Reads `<prefix>.weight_g`, `<prefix>.weight_v`, `<prefix>.bias`
/// and fuses them into a plain conv weight.
fn load_wn_conv1d(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c_in: usize, c_out: usize, k: usize,
    stride: usize, pad: usize, dilation: usize,
) -> crate::Result<Conv1dWeights> {
    let weight_g = crate::lazy::load_tensor_as_f32(st, &format!("{prefix}.weight_g"))?;
    let weight_v = crate::lazy::load_tensor_as_f32(st, &format!("{prefix}.weight_v"))?;
    let fused = fuse_weight_norm_conv1d(&weight_v, &weight_g, c_out, c_in, k)?;
    let bias = crate::lazy::load_tensor_as_f32(st, &format!("{prefix}.bias"))?;
    if bias.len() != c_out {
        crate::bail!(
            "load_wn_conv1d {prefix:?}: bias has {} elements, expected {c_out}",
            bias.len(),
        );
    }
    Ok(Conv1dWeights {
        w: Arc::from(fused),
        b: Some(Arc::from(bias)),
        c_in, c_out, k, stride, pad, dilation,
    })
}

/// Load one weight-normed `ConvTranspose1dWeights` at the given prefix.
fn load_wn_conv_transpose1d(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    c_in: usize, c_out: usize, k: usize,
    stride: usize, pad: usize,
) -> crate::Result<ConvTranspose1dWeights> {
    let weight_g = crate::lazy::load_tensor_as_f32(st, &format!("{prefix}.weight_g"))?;
    let weight_v = crate::lazy::load_tensor_as_f32(st, &format!("{prefix}.weight_v"))?;
    let fused = fuse_weight_norm_conv_transpose1d(&weight_v, &weight_g, c_in, c_out, k)?;
    let bias = crate::lazy::load_tensor_as_f32(st, &format!("{prefix}.bias"))?;
    if bias.len() != c_out {
        crate::bail!(
            "load_wn_conv_transpose1d {prefix:?}: bias has {} elements, expected {c_out}",
            bias.len(),
        );
    }
    Ok(ConvTranspose1dWeights {
        w: Arc::from(fused),
        b: Some(Arc::from(bias)),
        c_in, c_out, k, stride, pad,
    })
}

fn load_snake(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    channels: usize,
) -> crate::Result<Snake1dWeights> {
    let alpha = crate::lazy::load_tensor_as_f32(st, &format!("{prefix}.alpha"))?;
    if alpha.len() != channels {
        crate::bail!(
            "load_snake {prefix:?}: alpha has {} elements, expected {channels}",
            alpha.len(),
        );
    }
    Ok(Snake1dWeights {
        alpha: Arc::from(alpha),
        channels,
    })
}

fn load_residual_unit(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    dim: usize,
    dilation: usize,
) -> crate::Result<ResidualUnitWeights> {
    let pad = ((7 - 1) * dilation) / 2;
    // Eager `ResidualUnit::new` calls `vb.pp("block")` then numbers
    // children 0..3: snake1, conv1 (k=7, dilated), snake2, conv2 (k=1).
    let p = format!("{prefix}.block");
    Ok(ResidualUnitWeights {
        snake1: load_snake(st, &format!("{p}.0"), dim)?,
        conv1: load_wn_conv1d(st, &format!("{p}.1"), dim, dim, 7, 1, pad, dilation)?,
        snake2: load_snake(st, &format!("{p}.2"), dim)?,
        conv2: load_wn_conv1d(st, &format!("{p}.3"), dim, dim, 1, 1, 0, 1)?,
    })
}

fn load_decoder_block(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
    stride: usize,
) -> crate::Result<DecoderBlockWeights> {
    let pad = stride.div_ceil(2);
    // Eager `DecoderBlock::new` uses `vb.pp("block")` then 0..4:
    // snake1, conv_tr1 (weight-norm transposed), res1, res2, res3.
    let p = format!("{prefix}.block");
    Ok(DecoderBlockWeights {
        snake1: load_snake(st, &format!("{p}.0"), in_dim)?,
        conv_tr1: load_wn_conv_transpose1d(
            st, &format!("{p}.1"), in_dim, out_dim, 2 * stride, stride, pad,
        )?,
        res1: load_residual_unit(st, &format!("{p}.2"), out_dim, 1)?,
        res2: load_residual_unit(st, &format!("{p}.3"), out_dim, 3)?,
        res3: load_residual_unit(st, &format!("{p}.4"), out_dim, 9)?,
    })
}

impl DacWeights {
    /// Load DAC weights from a memory-mapped safetensors file using the
    /// HuggingFace naming convention used by the
    /// `descript/dac_44khz` / `descript/dac_16khz` checkpoints.
    ///
    /// Tensor layout (decoder path + RVQ):
    /// - `quantizer.quantizers.{i}.in_proj.{weight_g,weight_v,bias}`
    /// - `quantizer.quantizers.{i}.out_proj.{weight_g,weight_v,bias}`
    /// - `quantizer.quantizers.{i}.codebook.weight` (shape `[cb_size, cb_dim]`)
    /// - `decoder.model.0.{weight_g,weight_v,bias}` — initial Conv1d (k=7, pad=3)
    /// - For each upsampling stage `s` (1..=N):
    ///   - `decoder.model.{s}.block.0.alpha` — Snake1d
    ///   - `decoder.model.{s}.block.1.{weight_g,weight_v,bias}` — ConvTranspose1d
    ///   - `decoder.model.{s}.block.{2,3,4}` — three ResidualUnits, each with
    ///     `block.0.alpha` / `block.1.{wg,wv,b}` (k=7 dilated)
    ///     / `block.2.alpha` / `block.3.{wg,wv,b}` (k=1)
    /// - `decoder.model.{N+1}.alpha` — final Snake1d
    /// - `decoder.model.{N+2}.{weight_g,weight_v,bias}` — final Conv1d
    ///
    /// All weight-normed convs are fused on the fly into a plain
    /// `[c_out, c_in, k]` (or `[c_in, c_out, k]` for transposes) weight
    /// tensor at load-time — same trick the eager port uses to keep
    /// inference paths un-cluttered by weight-norm bookkeeping.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &DacConfig,
    ) -> Result<Self> {
        // RVQ: one VectorQuantizer per codebook.
        let mut quantizers = Vec::with_capacity(cfg.num_codebooks);
        for i in 0..cfg.num_codebooks {
            let p = format!("quantizer.quantizers.{i}");
            let in_proj = load_wn_conv1d(
                st, &format!("{p}.in_proj"),
                cfg.latent_dim, cfg.codebook_dim, 1, 1, 0, 1,
            )?;
            let out_proj = load_wn_conv1d(
                st, &format!("{p}.out_proj"),
                cfg.codebook_dim, cfg.latent_dim, 1, 1, 0, 1,
            )?;
            let codebook = crate::lazy::load_tensor_as_f32(
                st, &format!("{p}.codebook.weight"),
            )?;
            let expected = cfg.codebook_size * cfg.codebook_dim;
            if codebook.len() != expected {
                crate::bail!(
                    "DAC codebook {i}: {} elements, expected {expected}",
                    codebook.len(),
                );
            }
            quantizers.push(VectorQuantizerWeights {
                in_proj, out_proj,
                codebook: Arc::from(codebook),
            });
        }

        // Decoder: `decoder.model.{i}` for i=0..N+2.
        let n = cfg.decoder_rates.len();
        let mut channels = cfg.decoder_initial_channels;
        let conv1 = load_wn_conv1d(
            st, "decoder.model.0",
            cfg.latent_dim, channels, 7, 1, 3, 1,
        )?;
        let mut blocks = Vec::with_capacity(n);
        for (idx, &stride) in cfg.decoder_rates.iter().enumerate() {
            let in_dim = channels;
            let out_dim = channels / 2;
            blocks.push(load_decoder_block(
                st, &format!("decoder.model.{}", idx + 1),
                in_dim, out_dim, stride,
            )?);
            channels = out_dim;
        }
        let snake1 = load_snake(
            st, &format!("decoder.model.{}", n + 1), channels,
        )?;
        let conv2 = load_wn_conv1d(
            st, &format!("decoder.model.{}", n + 2),
            channels, cfg.decoder_out_channels, 7, 1, 3, 1,
        )?;

        Ok(Self {
            quantizers,
            decoder: DecoderWeights { conv1, blocks, snake1, conv2 },
        })
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

    fn conv1d_w(
        c_in: usize, c_out: usize, k: usize, stride: usize, pad: usize, dilation: usize,
        bias: bool, nb: &mut dyn FnMut() -> f32,
    ) -> Conv1dWeights {
        Conv1dWeights {
            w: vec_of(c_out * c_in * k, nb),
            b: if bias { Some(vec_of(c_out, nb)) } else { None },
            c_in, c_out, k, stride, pad, dilation,
        }
    }

    fn conv_transpose1d_w(
        c_in: usize, c_out: usize, k: usize, stride: usize, pad: usize, bias: bool,
        nb: &mut dyn FnMut() -> f32,
    ) -> ConvTranspose1dWeights {
        ConvTranspose1dWeights {
            w: vec_of(c_in * c_out * k, nb),
            b: if bias { Some(vec_of(c_out, nb)) } else { None },
            c_in, c_out, k, stride, pad,
        }
    }

    fn snake_w(channels: usize, nb: &mut dyn FnMut() -> f32) -> Snake1dWeights {
        Snake1dWeights {
            alpha: vec_of(channels, nb),
            channels,
        }
    }

    fn res_unit_w(dim: usize, dilation: usize, nb: &mut dyn FnMut() -> f32) -> ResidualUnitWeights {
        let pad = ((7 - 1) * dilation) / 2;
        ResidualUnitWeights {
            snake1: snake_w(dim, nb),
            conv1: conv1d_w(dim, dim, 7, 1, pad, dilation, true, nb),
            snake2: snake_w(dim, nb),
            conv2: conv1d_w(dim, dim, 1, 1, 0, 1, true, nb),
        }
    }

    fn decoder_block_w(
        in_dim: usize, out_dim: usize, stride: usize, nb: &mut dyn FnMut() -> f32,
    ) -> DecoderBlockWeights {
        let pad = stride.div_ceil(2);
        DecoderBlockWeights {
            snake1: snake_w(in_dim, nb),
            conv_tr1: conv_transpose1d_w(in_dim, out_dim, 2 * stride, stride, pad, true, nb),
            res1: res_unit_w(out_dim, 1, nb),
            res2: res_unit_w(out_dim, 3, nb),
            res3: res_unit_w(out_dim, 9, nb),
        }
    }

    fn tiny_dac_config() -> DacConfig {
        DacConfig {
            num_codebooks: 2,
            codebook_size: 8,
            codebook_dim: 4,
            latent_dim: 16,
            decoder_initial_channels: 32,
            decoder_rates: vec![2, 2],
            decoder_out_channels: 1,
        }
    }

    fn tiny_dac_weights(cfg: &DacConfig) -> DacWeights {
        let mut nb = rng_seed(2026);
        let quantizers: Vec<VectorQuantizerWeights> = (0..cfg.num_codebooks).map(|_| {
            VectorQuantizerWeights {
                in_proj: conv1d_w(cfg.latent_dim, cfg.codebook_dim, 1, 1, 0, 1, true, &mut nb),
                out_proj: conv1d_w(cfg.codebook_dim, cfg.latent_dim, 1, 1, 0, 1, true, &mut nb),
                codebook: vec_of(cfg.codebook_size * cfg.codebook_dim, &mut nb),
            }
        }).collect();

        let mut channels = cfg.decoder_initial_channels;
        let conv1 = conv1d_w(cfg.latent_dim, channels, 7, 1, 3, 1, true, &mut nb);
        let mut blocks = Vec::with_capacity(cfg.decoder_rates.len());
        for &stride in &cfg.decoder_rates {
            let next = channels / 2;
            blocks.push(decoder_block_w(channels, next, stride, &mut nb));
            channels = next;
        }
        let snake1 = snake_w(channels, &mut nb);
        let conv2 = conv1d_w(channels, cfg.decoder_out_channels, 7, 1, 3, 1, true, &mut nb);

        DacWeights {
            quantizers,
            decoder: DecoderWeights { conv1, blocks, snake1, conv2 },
        }
    }

    #[test]
    fn dilation_expansion_correctness() {
        // K=3, D=3 → K' = 3 + 2*2 = 7.
        let w = vec![1.0_f32, 2.0, 3.0]; // c_out=1, c_in=1
        let (out, k_eff) = expand_conv1d_weight_for_dilation(&w, 1, 1, 3, 3);
        assert_eq!(k_eff, 7);
        assert_eq!(out, vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn dilation_expansion_identity_for_d_eq_1() {
        let w = vec![0.5_f32, 1.5, -1.0, 0.25];
        let (out, k_eff) = expand_conv1d_weight_for_dilation(&w, 2, 1, 2, 1);
        assert_eq!(k_eff, 2);
        assert_eq!(out, w);
    }

    #[test]
    fn dilation_expansion_multi_channel() {
        // c_out=2, c_in=1, K=2, D=2 → K' = 3.
        let w = vec![1.0_f32, 2.0, 3.0, 4.0];
        let (out, k_eff) = expand_conv1d_weight_for_dilation(&w, 2, 1, 2, 2);
        assert_eq!(k_eff, 3);
        assert_eq!(out, vec![1.0, 0.0, 2.0, 3.0, 0.0, 4.0]);
    }

    #[test]
    fn snake1d_alpha_zero_is_identity() {
        // α = 0 → sin(0) = 0 → correction = 0/(0+ε) = 0 → output = x.
        let dev = Device::cpu();
        let x = LazyTensor::from_f32(
            vec![0.5_f32, -0.25, 0.75, 1.0],
            Shape::from_dims(&[1, 2, 2]),
            &dev,
        );
        let snake_w = Snake1dWeights {
            alpha: Arc::from(vec![0.0_f32; 2]),
            channels: 2,
        };
        let out = apply_snake1d(&x, &snake_w, &x).unwrap().realize_f32();
        let in_data = x.realize_f32();
        for (a, b) in out.iter().zip(in_data.iter()) {
            assert!((a - b).abs() < 1e-5, "α=0 should be identity: {a} vs {b}");
        }
    }

    #[test]
    fn decode_codes_shape_and_finite() {
        let cfg = tiny_dac_config();
        let weights = tiny_dac_weights(&cfg);
        let model = DacModel { config: cfg.clone(), weights };
        let time = 4_usize;
        // codes shape: (1, num_codebooks, time), each entry in [0, codebook_size).
        let mut data: Vec<u32> = Vec::with_capacity(cfg.num_codebooks * time);
        for c in 0..cfg.num_codebooks {
            for t in 0..time {
                data.push(((c + t) % cfg.codebook_size) as u32);
            }
        }
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32; 1], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let codes = anchor.const_u32_like(
            data, Shape::from_dims(&[1, cfg.num_codebooks, time]),
        );
        let audio = model.decode_codes(&codes).unwrap();
        let dims = audio.shape();
        let dims = dims.dims();
        // Output channels = decoder_out_channels = 1.
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], cfg.decoder_out_channels);
        // Output time = time * product(rates) modulo edge padding.
        // With rates [2, 2] and time=4: time * 4 = 16, give or take.
        assert!(dims[2] > 0, "audio output must have positive length");
        for &v in &audio.realize_f32() {
            assert!(v.is_finite(), "non-finite audio sample: {v}");
        }
    }

    // ---- Safetensors loader round-trip --------------------------------

    fn write_tmp_safetensors(
        tensors: &[(String, Vec<usize>, Vec<f32>)],
    ) -> std::path::PathBuf {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;
        let bytes_store: Vec<Vec<u8>> = tensors.iter()
            .map(|(_, _, data)| data.iter().flat_map(|f| f.to_le_bytes()).collect())
            .collect();
        let views: HashMap<String, TensorView<'_>> = tensors.iter()
            .zip(bytes_store.iter())
            .map(|((name, shape, _), bytes)| {
                let v = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .expect("TensorView::new");
                (name.clone(), v)
            })
            .collect();
        let metadata: Option<HashMap<String, String>> = None;
        let bytes_out = safetensors::serialize(&views, metadata).unwrap();
        let path = std::env::temp_dir().join(format!(
            "fuel_lazy_dac_test_{}.safetensors",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
        ));
        std::fs::write(&path, bytes_out).unwrap();
        path
    }

    /// Emit a weight-normed conv1d at `prefix` plus an explicit bias.
    /// Uses constant weights so the test mainly exercises shape/name wiring.
    fn wn_conv1d_tensors(
        prefix: &str, c_in: usize, c_out: usize, k: usize, seed: u32,
    ) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut s = seed;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let wg: Vec<f32> = (0..c_out).map(|_| next()).collect();
        let wv: Vec<f32> = (0..c_out * c_in * k).map(|_| next()).collect();
        let b: Vec<f32> = (0..c_out).map(|_| next()).collect();
        vec![
            (format!("{prefix}.weight_g"), vec![c_out, 1, 1], wg),
            (format!("{prefix}.weight_v"), vec![c_out, c_in, k], wv),
            (format!("{prefix}.bias"), vec![c_out], b),
        ]
    }

    fn wn_conv_transpose1d_tensors(
        prefix: &str, c_in: usize, c_out: usize, k: usize, seed: u32,
    ) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut s = seed;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let wg: Vec<f32> = (0..c_in).map(|_| next()).collect();
        let wv: Vec<f32> = (0..c_in * c_out * k).map(|_| next()).collect();
        let b: Vec<f32> = (0..c_out).map(|_| next()).collect();
        vec![
            (format!("{prefix}.weight_g"), vec![c_in, 1, 1], wg),
            (format!("{prefix}.weight_v"), vec![c_in, c_out, k], wv),
            (format!("{prefix}.bias"), vec![c_out], b),
        ]
    }

    fn snake_tensors(prefix: &str, channels: usize, seed: u32)
        -> Vec<(String, Vec<usize>, Vec<f32>)>
    {
        let mut s = seed;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let alpha: Vec<f32> = (0..channels).map(|_| next()).collect();
        vec![(format!("{prefix}.alpha"), vec![1, channels, 1], alpha)]
    }

    fn res_unit_tensors(prefix: &str, dim: usize, seed: u32)
        -> Vec<(String, Vec<usize>, Vec<f32>)>
    {
        let mut out = Vec::new();
        out.extend(snake_tensors(&format!("{prefix}.block.0"), dim, seed));
        // k=7 dilated conv. dilation is irrelevant for the on-disk tensor.
        out.extend(wn_conv1d_tensors(&format!("{prefix}.block.1"), dim, dim, 7, seed + 1));
        out.extend(snake_tensors(&format!("{prefix}.block.2"), dim, seed + 2));
        out.extend(wn_conv1d_tensors(&format!("{prefix}.block.3"), dim, dim, 1, seed + 3));
        out
    }

    fn decoder_block_tensors(
        prefix: &str, in_dim: usize, out_dim: usize, stride: usize, seed: u32,
    ) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut out = Vec::new();
        out.extend(snake_tensors(&format!("{prefix}.block.0"), in_dim, seed));
        out.extend(wn_conv_transpose1d_tensors(
            &format!("{prefix}.block.1"), in_dim, out_dim, 2 * stride, seed + 10,
        ));
        out.extend(res_unit_tensors(&format!("{prefix}.block.2"), out_dim, seed + 20));
        out.extend(res_unit_tensors(&format!("{prefix}.block.3"), out_dim, seed + 30));
        out.extend(res_unit_tensors(&format!("{prefix}.block.4"), out_dim, seed + 40));
        out
    }

    /// Round-trip a tiny DAC config through a synthesized safetensors
    /// file and confirm the loader picks up all named tensors with
    /// correct shapes.
    #[test]
    fn load_from_mmapped_round_trip_tiny() {
        let cfg = tiny_dac_config();
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        // RVQ.
        for i in 0..cfg.num_codebooks {
            let p = format!("quantizer.quantizers.{i}");
            tensors.extend(wn_conv1d_tensors(
                &format!("{p}.in_proj"), cfg.latent_dim, cfg.codebook_dim, 1, 100 + i as u32,
            ));
            tensors.extend(wn_conv1d_tensors(
                &format!("{p}.out_proj"), cfg.codebook_dim, cfg.latent_dim, 1, 200 + i as u32,
            ));
            // codebook.weight has shape [cb_size, cb_dim].
            let cb: Vec<f32> = (0..cfg.codebook_size * cfg.codebook_dim)
                .map(|j| (j as f32 + i as f32) * 0.01).collect();
            tensors.push((
                format!("{p}.codebook.weight"),
                vec![cfg.codebook_size, cfg.codebook_dim],
                cb,
            ));
        }

        // Decoder.
        let n = cfg.decoder_rates.len();
        let mut channels = cfg.decoder_initial_channels;
        tensors.extend(wn_conv1d_tensors(
            "decoder.model.0", cfg.latent_dim, channels, 7, 500,
        ));
        for (idx, &stride) in cfg.decoder_rates.iter().enumerate() {
            let in_dim = channels;
            let out_dim = channels / 2;
            tensors.extend(decoder_block_tensors(
                &format!("decoder.model.{}", idx + 1),
                in_dim, out_dim, stride, 1000 + idx as u32 * 100,
            ));
            channels = out_dim;
        }
        tensors.extend(snake_tensors(
            &format!("decoder.model.{}", n + 1), channels, 7000,
        ));
        tensors.extend(wn_conv1d_tensors(
            &format!("decoder.model.{}", n + 2),
            channels, cfg.decoder_out_channels, 7, 8000,
        ));

        let path = write_tmp_safetensors(&tensors);
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path).unwrap() };
        let weights = DacWeights::load_from_mmapped(&st, &cfg).unwrap();
        assert_eq!(weights.quantizers.len(), cfg.num_codebooks);
        for (i, q) in weights.quantizers.iter().enumerate() {
            assert_eq!(q.in_proj.c_in, cfg.latent_dim);
            assert_eq!(q.in_proj.c_out, cfg.codebook_dim);
            assert_eq!(q.out_proj.c_in, cfg.codebook_dim);
            assert_eq!(q.out_proj.c_out, cfg.latent_dim);
            assert_eq!(q.codebook.len(), cfg.codebook_size * cfg.codebook_dim,
                "codebook {i} length");
        }
        assert_eq!(weights.decoder.blocks.len(), n);
        assert_eq!(weights.decoder.conv1.c_in, cfg.latent_dim);
        assert_eq!(weights.decoder.conv1.c_out, cfg.decoder_initial_channels);
        let mut ch = cfg.decoder_initial_channels;
        for (i, blk) in weights.decoder.blocks.iter().enumerate() {
            assert_eq!(blk.conv_tr1.c_in, ch, "block {i} conv_tr1.c_in");
            assert_eq!(blk.conv_tr1.c_out, ch / 2, "block {i} conv_tr1.c_out");
            assert_eq!(blk.res1.conv1.dilation, 1);
            assert_eq!(blk.res2.conv1.dilation, 3);
            assert_eq!(blk.res3.conv1.dilation, 9);
            ch /= 2;
        }
        assert_eq!(weights.decoder.snake1.channels, ch);
        assert_eq!(weights.decoder.conv2.c_in, ch);
        assert_eq!(weights.decoder.conv2.c_out, cfg.decoder_out_channels);

        let _ = std::fs::remove_file(&path);
    }

    /// Smoke-test that the loader produces a model that can decode
    /// codes end-to-end and produce a finite waveform.
    #[test]
    fn load_from_mmapped_decode_codes_smoke() {
        let cfg = tiny_dac_config();
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for i in 0..cfg.num_codebooks {
            let p = format!("quantizer.quantizers.{i}");
            tensors.extend(wn_conv1d_tensors(
                &format!("{p}.in_proj"), cfg.latent_dim, cfg.codebook_dim, 1, 100 + i as u32,
            ));
            tensors.extend(wn_conv1d_tensors(
                &format!("{p}.out_proj"), cfg.codebook_dim, cfg.latent_dim, 1, 200 + i as u32,
            ));
            let cb: Vec<f32> = (0..cfg.codebook_size * cfg.codebook_dim)
                .map(|j| (j as f32 + i as f32) * 0.01).collect();
            tensors.push((
                format!("{p}.codebook.weight"),
                vec![cfg.codebook_size, cfg.codebook_dim],
                cb,
            ));
        }
        let n = cfg.decoder_rates.len();
        let mut channels = cfg.decoder_initial_channels;
        tensors.extend(wn_conv1d_tensors(
            "decoder.model.0", cfg.latent_dim, channels, 7, 500,
        ));
        for (idx, &stride) in cfg.decoder_rates.iter().enumerate() {
            let in_dim = channels;
            let out_dim = channels / 2;
            tensors.extend(decoder_block_tensors(
                &format!("decoder.model.{}", idx + 1),
                in_dim, out_dim, stride, 1000 + idx as u32 * 100,
            ));
            channels = out_dim;
        }
        tensors.extend(snake_tensors(
            &format!("decoder.model.{}", n + 1), channels, 7000,
        ));
        tensors.extend(wn_conv1d_tensors(
            &format!("decoder.model.{}", n + 2),
            channels, cfg.decoder_out_channels, 7, 8000,
        ));

        let path = write_tmp_safetensors(&tensors);
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path).unwrap() };
        let weights = DacWeights::load_from_mmapped(&st, &cfg).unwrap();
        let model = DacModel { config: cfg.clone(), weights };

        let time = 4_usize;
        let dev = Device::cpu();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32; 1], Shape::from_dims(&[1]), &dev,
        );
        let codes = anchor.const_u32_like(
            vec![1_u32; cfg.num_codebooks * time],
            Shape::from_dims(&[1, cfg.num_codebooks, time]),
        );
        let audio = model.decode_codes(&codes).unwrap().realize_f32();
        assert!(!audio.is_empty(), "decoded audio must have samples");
        for &v in &audio {
            assert!(v.is_finite(), "non-finite sample: {v}");
        }

        let _ = std::fs::remove_file(&path);
    }

    /// Different codes must produce different audio — verifies
    /// the codebook path is wired through the decoder.
    #[test]
    fn decode_codes_responds_to_codes() {
        let cfg = tiny_dac_config();
        let weights = tiny_dac_weights(&cfg);
        let model = DacModel { config: cfg.clone(), weights };
        let time = 4_usize;
        let dev = Device::cpu();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32; 1], Shape::from_dims(&[1]), &dev,
        );
        let codes_a = anchor.const_u32_like(
            vec![0_u32; cfg.num_codebooks * time],
            Shape::from_dims(&[1, cfg.num_codebooks, time]),
        );
        let codes_b = anchor.const_u32_like(
            vec![3_u32; cfg.num_codebooks * time],
            Shape::from_dims(&[1, cfg.num_codebooks, time]),
        );
        let a = model.decode_codes(&codes_a).unwrap().realize_f32();
        let b = model.decode_codes(&codes_b).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "decoded audio must respond to code changes, max_diff = {max_diff}");
    }
}

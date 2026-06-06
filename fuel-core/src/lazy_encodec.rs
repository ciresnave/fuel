//! EnCodec — lazy port (decoder + RVQ).
//!
//! Discrete codes `(1, n_codebooks, T)` → waveform via:
//!   1. ResidualVectorQuantizer reconstruction: per-codebook
//!      embedding lookup + `out_proj` summed.
//!   2. Decoder:
//!      - init_conv (Conv1d) → init_lstm (with stack residual)
//!      - For each upsampling ratio:
//!          ELU → ConvTranspose1d (stride = ratio) → N ResnetBlocks
//!      - ELU → final_conv → waveform (B, audio_channels, T_out)
//!
//! Padding: EnCodec uses left-only causal padding (when
//! `use_causal_conv = true`, the default) or symmetric padding.
//! Both implemented via narrow + concat composites with one of:
//!   - Constant (zero) padding
//!   - Replicate (repeat edge value) padding
//!
//! Reflect padding is upstream-deferred (rare in EnCodec configs).
//!
//! ResnetBlock: ELU → Conv1d (dim → dim/compress, dilated) → ELU
//! → Conv1d (dim/compress → dim) → optional 1×1 shortcut conv on
//! the residual path, then add.
//!
//! v1 scope:
//!   - F32, batch == 1.
//!   - decode_codes (decoder + RVQ).
//!   - Dilated conv handled by the same expanded-const-weight
//!     trick as lazy_dac (kernel `K` with dilation `D` becomes a
//!     plain conv with kernel `K + (K-1)·(D-1)` and zero-interleaved
//!     weights — all DAC/EnCodec weights are constants at
//!     load-time).
//!   - GroupNorm and weight-norm trained variants both load
//!     through the same Conv1dWeights since norm is fused into
//!     the conv weight pre-realize.

use crate::lazy::{load_tensor_as_f32, LazyTensor, WeightStorage};
use crate::lazy_dac::expand_conv1d_weight_for_dilation_if_needed;
use crate::lazy_lstm::{LstmCellWeights, LstmStack};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Constant,
    Replicate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncodecConfig {
    pub audio_channels: usize,
    pub num_filters: usize,
    pub num_residual_layers: usize,
    /// Per-ratio downsampling/upsampling factor. The decoder iterates
    /// in the listed order (eager `cfg.upsampling_ratios.iter()`).
    pub upsampling_ratios: Vec<usize>,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub residual_kernel_size: usize,
    pub dilation_growth_rate: usize,
    pub use_causal_conv: bool,
    pub pad_mode: PadMode,
    pub compress: usize,
    pub num_lstm_layers: usize,
    pub use_conv_shortcut: bool,
    pub hidden_size: usize,
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
}

impl EncodecConfig {
    /// `facebook/encodec_24khz` preset.
    pub fn default_preset() -> Self {
        Self {
            audio_channels: 1,
            num_filters: 32,
            num_residual_layers: 1,
            upsampling_ratios: vec![8, 5, 4, 2],
            kernel_size: 7,
            last_kernel_size: 7,
            residual_kernel_size: 3,
            dilation_growth_rate: 2,
            use_causal_conv: true,
            pad_mode: PadMode::Replicate,
            compress: 2,
            num_lstm_layers: 2,
            use_conv_shortcut: true,
            hidden_size: 128,
            num_codebooks: 32,
            codebook_size: 1024,
            codebook_dim: 128,
        }
    }
}

// ---- Weight structs --------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Conv1dWeights {
    pub w: Arc<[f32]>,
    pub b: Option<Arc<[f32]>>,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
    pub stride: usize,
    pub dilation: usize,
}

#[derive(Debug, Clone)]
pub struct ConvTranspose1dWeights {
    /// `[c_in, c_out, K]` (PyTorch convention).
    pub w: Arc<[f32]>,
    pub b: Option<Arc<[f32]>>,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
    pub stride: usize,
}

#[derive(Debug, Clone)]
pub struct ResnetBlockWeights {
    pub conv1: Conv1dWeights,
    pub conv2: Conv1dWeights,
    /// 1×1 conv on the residual branch when `use_conv_shortcut`.
    pub shortcut: Option<Conv1dWeights>,
}

#[derive(Debug, Clone)]
pub struct UpsampleStageWeights {
    pub up_conv: ConvTranspose1dWeights,
    pub resnets: Vec<ResnetBlockWeights>,
}

#[derive(Debug, Clone)]
pub struct DecoderWeights {
    pub init_conv: Conv1dWeights,
    pub init_lstm: Vec<LstmCellWeights>,
    pub stages: Vec<UpsampleStageWeights>,
    pub final_conv: Conv1dWeights,
}

#[derive(Debug, Clone)]
pub struct VectorQuantizerWeights {
    /// `[codebook_size, codebook_dim]` — embedded as a const tensor at lookup.
    pub codebook: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct EncodecWeights {
    pub quantizers: Vec<VectorQuantizerWeights>,
    pub decoder: DecoderWeights,
}

#[derive(Debug, Clone)]
pub struct EncodecModel {
    pub config: EncodecConfig,
    pub weights: EncodecWeights,
}

// ---- Forward ---------------------------------------------------------------

impl EncodecModel {
    /// Decode discrete codes `(1, num_codebooks, T)` to a waveform
    /// `(1, audio_channels, T_out)`. T_out depends on the per-stage
    /// transposed conv strides and padding edge effects.
    pub fn decode_codes(&self, codes: &LazyTensor) -> Result<LazyTensor> {
        let dims = codes.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "codes must be rank 3 [1, num_codebooks, T]");
        assert_eq!(dims[0], 1, "v1 supports batch == 1");
        assert_eq!(
            dims[1], self.weights.quantizers.len(),
            "codes codebook count {} must match weights {}",
            dims[1], self.weights.quantizers.len(),
        );
        let latent = self.rvq_from_codes(codes)?;
        self.decoder_forward(&latent)
    }

    /// `latent = sum_i codebook_i[codes[:, i]]` projected to
    /// hidden_size space. EnCodec's RVQ uses a per-codebook
    /// embedding lookup; there's no out_proj (unlike DAC where
    /// out_proj is a 1×1 conv) — the eager EnCodec quantizer is
    /// `embed[codes]` directly summed across codebooks. (Reference:
    /// `transformers/models/encodec/modeling_encodec.py` —
    /// `EncodecResidualVectorQuantizer.decode`.)
    fn rvq_from_codes(&self, codes: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = codes.shape();
        let dims = dims.dims();
        let time = dims[2];
        let mut sum: Option<LazyTensor> = None;
        for (idx, q) in self.weights.quantizers.iter().enumerate() {
            let ids = codes
                .narrow(1_usize, idx, 1)?
                .reshape(Shape::from_dims(&[time]))?;
            let codebook = codes.const_f32_like(
                Arc::clone(&q.codebook),
                Shape::from_dims(&[cfg.codebook_size, cfg.codebook_dim]),
            );
            // (T, codebook_dim) → (1, codebook_dim, T)
            let z_p = codebook
                .index_select(0_usize, &ids)?
                .reshape(Shape::from_dims(&[1, time, cfg.codebook_dim]))?
                .permute([0, 2, 1_usize])?;
            sum = Some(match sum {
                None => z_p,
                Some(s) => s.add(&z_p)?,
            });
        }
        sum.ok_or_else(|| {
            fuel_core_types::Error::Msg("EnCodec RVQ: no codebooks".into()).bt()
        })
    }

    fn decoder_forward(&self, latent: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dec = &self.weights.decoder;

        let mut x = apply_encodec_conv1d(latent, &dec.init_conv, cfg, latent)?;
        // (B, C, T) → (B, T, C) for LSTM, with residual on (B, T, C),
        // then back to (B, C, T).
        let dims = x.shape();
        let dims = dims.dims();
        let b = dims[0]; let c = dims[1]; let t = dims[2];
        let x_btc = x
            .reshape(Shape::from_dims(&[b, c, t]))?
            .permute([0, 2, 1_usize])?;
        let lstm_stack = LstmStack { layers: dec.init_lstm.clone() };
        let lstm_out = lstm_stack.forward_with_residual(&x_btc)?;
        x = lstm_out
            .permute([0, 2, 1_usize])?
            .reshape(Shape::from_dims(&[b, c, t]))?;

        for stage in &dec.stages {
            x = x.elu(1.0);
            x = apply_encodec_conv_transpose1d(&x, &stage.up_conv, cfg, latent)?;
            for r in &stage.resnets {
                x = apply_resnet_block(&x, r, cfg, latent)?;
            }
        }
        x = x.elu(1.0);
        apply_encodec_conv1d(&x, &dec.final_conv, cfg, latent)
    }
}

// ---- Component helpers -----------------------------------------------------

fn apply_resnet_block(
    x: &LazyTensor,
    r: &ResnetBlockWeights,
    cfg: &EncodecConfig,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let residual = if let Some(sc) = &r.shortcut {
        apply_encodec_conv1d(x, sc, cfg, anchor)?
    } else {
        x.clone()
    };
    let y = x.elu(1.0);
    let y = apply_encodec_conv1d(&y, &r.conv1, cfg, anchor)?;
    let y = y.elu(1.0);
    let y = apply_encodec_conv1d(&y, &r.conv2, cfg, anchor)?;
    // The eager block narrows the residual to the post-conv length
    // when they differ (the dilated convs with causal padding
    // preserve length, but the symmetric case can produce mismatch).
    let y_dims = y.shape();
    let y_dims = y_dims.dims();
    let r_dims = residual.shape();
    let r_dims = r_dims.dims();
    let y_t = y_dims[2];
    let r_t = r_dims[2];
    let res = if y_t == r_t {
        residual
    } else {
        let pad = (r_t - y_t) / 2;
        residual.narrow(2_usize, pad, y_t)?
    };
    res.add(&y)
}

fn apply_encodec_conv1d(
    x: &LazyTensor,
    c: &Conv1dWeights,
    cfg: &EncodecConfig,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    // Effective kernel size with dilation.
    let k_eff = (c.k - 1) * c.dilation + 1;
    let padding_total = k_eff.saturating_sub(c.stride);
    let extra = extra_padding_for_conv1d(x, k_eff, c.stride, padding_total)?;
    let x_padded = if cfg.use_causal_conv {
        pad1d(x, padding_total, extra, cfg.pad_mode, anchor)?
    } else {
        let right = padding_total / 2;
        let left = padding_total - right;
        pad1d(x, left, right + extra, cfg.pad_mode, anchor)?
    };
    // Expand dilated weight if needed (dilation handled at weight level).
    let (w_data, k_used) =
        expand_conv1d_weight_for_dilation_if_needed(&c.w, c.c_out, c.c_in, c.k, c.dilation);
    let w = anchor.const_f32_like(
        Arc::<[f32]>::from(w_data),
        Shape::from_dims(&[c.c_out, c.c_in, k_used]),
    );
    let bias = c.b.as_ref().map(|b| {
        anchor.const_f32_like(Arc::clone(b), Shape::from_dims(&[c.c_out]))
    });
    x_padded.conv1d(&w, bias.as_ref(), c.stride, 0, 1)
}

fn apply_encodec_conv_transpose1d(
    x: &LazyTensor,
    c: &ConvTranspose1dWeights,
    cfg: &EncodecConfig,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let w = anchor.const_f32_like(
        Arc::clone(&c.w),
        Shape::from_dims(&[c.c_in, c.c_out, c.k]),
    );
    let mut out = x.conv_transpose1d(&w, c.stride, 0, 0, 1, 1)?;
    if let Some(b) = &c.b {
        let bias = anchor
            .const_f32_like(Arc::clone(b), Shape::from_dims(&[c.c_out]))
            .reshape(Shape::from_dims(&[1, c.c_out, 1]))?;
        out = out.broadcast_add(&bias)?;
    }
    // EnCodec causal transposed conv trims the tail by
    // `padding_total = k - stride` (with `trim_right_ratio = 1.0`).
    if cfg.use_causal_conv {
        let padding_total = c.k.saturating_sub(c.stride);
        let dims = out.shape();
        let dims = dims.dims();
        let t_out = dims[2];
        let keep = t_out.saturating_sub(padding_total);
        if keep > 0 && keep < t_out {
            out = out.narrow(2_usize, 0, keep)?;
        }
    }
    Ok(out)
}

fn extra_padding_for_conv1d(
    x: &LazyTensor, k_eff: usize, stride: usize, padding_total: usize,
) -> Result<usize> {
    let dims = x.shape();
    let dims = dims.dims();
    let t = dims[2];
    let n_frames = ((t + padding_total).saturating_sub(k_eff)) as f64 / stride as f64 + 1.0;
    let ideal_len = (n_frames.ceil() as usize - 1) * stride + k_eff;
    Ok(ideal_len.saturating_sub(t + padding_total))
}

/// Pad a (B, C, T) tensor along the last dim. Implements
/// Constant (zero) and Replicate (edge-repeat) modes via concat
/// composites. Causal callers pass `right = 0`.
pub fn pad1d(
    x: &LazyTensor, left: usize, right: usize, mode: PadMode, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let c = dims[1]; let t = dims[2];
    let make_const = |n: usize, anchor_t: &LazyTensor| -> LazyTensor {
        anchor_t.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; b * c * n]),
            Shape::from_dims(&[b, c, n]),
        )
    };
    let (left_pad, right_pad) = match mode {
        PadMode::Constant => {
            let lp = if left > 0 { Some(make_const(left, anchor)) } else { None };
            let rp = if right > 0 { Some(make_const(right, anchor)) } else { None };
            (lp, rp)
        }
        PadMode::Replicate => {
            // Replicate-left = x[:,:,0:1] repeated `left` times.
            // Replicate-right = x[:,:,-1:] repeated `right` times.
            let lp = if left > 0 {
                let edge = x.narrow(2_usize, 0, 1)?;
                let mut acc = edge.clone();
                for _ in 1..left {
                    acc = acc.concat(&edge, 2_usize)?;
                }
                Some(acc)
            } else { None };
            let rp = if right > 0 {
                let edge = x.narrow(2_usize, t - 1, 1)?;
                let mut acc = edge.clone();
                for _ in 1..right {
                    acc = acc.concat(&edge, 2_usize)?;
                }
                Some(acc)
            } else { None };
            (lp, rp)
        }
    };
    let mut acc = match left_pad {
        Some(lp) => lp.concat(x, 2_usize)?,
        None => x.clone(),
    };
    if let Some(rp) = right_pad {
        acc = acc.concat(&rp, 2_usize)?;
    }
    Ok(acc)
}

// ---- Safetensors loader ----------------------------------------------------

/// Recompose a weight-normalised conv1d kernel from the legacy
/// PyTorch `weight_g` / `weight_v` decomposition.
///
/// HuggingFace EnCodec checkpoints store every Conv1d / ConvTranspose1d
/// weight as `weight_g` (`[out_c, 1, 1]`) plus `weight_v`
/// (`[out_c, in_c, k]`). PyTorch's `nn.utils.weight_norm` reconstructs
/// the effective weight as
///
/// ```text
///   weight = weight_v * (weight_g / ||weight_v||_{dim=(1,2)})
/// ```
///
/// where the norm is taken over the input-channel and kernel axes
/// (i.e. one scalar per output channel). We fuse the norm back into
/// the dense weight at load time so the rest of the lazy pipeline can
/// treat the conv as a plain constant kernel.
fn fuse_weight_norm_conv1d(
    st: &crate::safetensors::MmapedSafetensors,
    name_prefix: &str,
    out_c: usize,
    in_c: usize,
    k: usize,
) -> Result<Vec<f32>> {
    let weight_g = load_tensor_as_f32(st, &format!("{name_prefix}.weight_g"))?;
    let weight_v = load_tensor_as_f32(st, &format!("{name_prefix}.weight_v"))?;
    if weight_g.len() != out_c {
        crate::bail!(
            "{name_prefix}.weight_g: {} elts, expected {} (= out_c)",
            weight_g.len(), out_c,
        );
    }
    let expected_v = out_c * in_c * k;
    if weight_v.len() != expected_v {
        crate::bail!(
            "{name_prefix}.weight_v: {} elts, expected {} (= {out_c} * {in_c} * {k})",
            weight_v.len(), expected_v,
        );
    }
    let mut out = vec![0.0_f32; expected_v];
    for o in 0..out_c {
        let row_start = o * in_c * k;
        let row_end = row_start + in_c * k;
        let v_row = &weight_v[row_start..row_end];
        let norm_sq: f32 = v_row.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt().max(1e-30);
        let scale = weight_g[o] / norm;
        for (dst, &src) in out[row_start..row_end].iter_mut().zip(v_row.iter()) {
            *dst = src * scale;
        }
    }
    Ok(out)
}

fn load_encodec_conv1d(
    st: &crate::safetensors::MmapedSafetensors,
    name_prefix: &str,
    c_in: usize,
    c_out: usize,
    k: usize,
    stride: usize,
    dilation: usize,
) -> Result<Conv1dWeights> {
    let w = fuse_weight_norm_conv1d(st, name_prefix, c_out, c_in, k)?;
    let b = load_tensor_as_f32(st, &format!("{name_prefix}.bias"))?;
    if b.len() != c_out {
        crate::bail!(
            "{name_prefix}.bias: {} elts, expected {}", b.len(), c_out,
        );
    }
    Ok(Conv1dWeights {
        w: Arc::from(w),
        b: Some(Arc::from(b)),
        c_in, c_out, k, stride, dilation,
    })
}

fn load_encodec_conv_transpose1d(
    st: &crate::safetensors::MmapedSafetensors,
    name_prefix: &str,
    c_in: usize,
    c_out: usize,
    k: usize,
    stride: usize,
) -> Result<ConvTranspose1dWeights> {
    // PyTorch ConvTranspose1d weight layout is `[c_in, c_out, k]` —
    // exactly what `weight_v` stores. Re-use the same fusion helper
    // by treating the leading axis as out_c for normalisation
    // purposes: PyTorch's weight_norm on ConvTranspose1d normalises
    // along dims (1, 2), which here means (c_out, k). Each "row" of
    // size `c_out * k` is associated with one weight_g scalar.
    let w = fuse_weight_norm_conv1d(st, name_prefix, c_in, c_out, k)?;
    let b = load_tensor_as_f32(st, &format!("{name_prefix}.bias"))?;
    if b.len() != c_out {
        crate::bail!(
            "{name_prefix}.bias: {} elts, expected {}", b.len(), c_out,
        );
    }
    Ok(ConvTranspose1dWeights {
        w: Arc::from(w),
        b: Some(Arc::from(b)),
        c_in, c_out, k, stride,
    })
}

/// Load a PyTorch `nn.LSTM` block as `num_layers` flat
/// [`LstmCellWeights`]. PyTorch stores its gates in the order
/// `[i, f, g, o]` along the leading axis — which matches the layout
/// `LstmCellWeights` documents, so we copy without re-shuffling.
fn load_encodec_lstm(
    st: &crate::safetensors::MmapedSafetensors,
    name_prefix: &str,
    dim: usize,
    num_layers: usize,
) -> Result<Vec<LstmCellWeights>> {
    let four_h = 4 * dim;
    let mut out = Vec::with_capacity(num_layers);
    for li in 0..num_layers {
        let w_ih = load_tensor_as_f32(
            st, &format!("{name_prefix}.weight_ih_l{li}"),
        )?;
        let w_hh = load_tensor_as_f32(
            st, &format!("{name_prefix}.weight_hh_l{li}"),
        )?;
        let b_ih = load_tensor_as_f32(
            st, &format!("{name_prefix}.bias_ih_l{li}"),
        )?;
        let b_hh = load_tensor_as_f32(
            st, &format!("{name_prefix}.bias_hh_l{li}"),
        )?;
        if w_ih.len() != four_h * dim {
            crate::bail!(
                "{name_prefix}.weight_ih_l{li}: {} elts, expected {}",
                w_ih.len(), four_h * dim,
            );
        }
        if w_hh.len() != four_h * dim {
            crate::bail!(
                "{name_prefix}.weight_hh_l{li}: {} elts, expected {}",
                w_hh.len(), four_h * dim,
            );
        }
        if b_ih.len() != four_h {
            crate::bail!(
                "{name_prefix}.bias_ih_l{li}: {} elts, expected {}",
                b_ih.len(), four_h,
            );
        }
        if b_hh.len() != four_h {
            crate::bail!(
                "{name_prefix}.bias_hh_l{li}: {} elts, expected {}",
                b_hh.len(), four_h,
            );
        }
        out.push(LstmCellWeights {
            w_ih: Arc::from(w_ih),
            w_hh: Arc::from(w_hh),
            b_ih: Arc::from(b_ih),
            b_hh: Arc::from(b_hh),
            input_dim: dim,
            hidden_dim: dim,
        });
    }
    Ok(out)
}

/// Number of residual quantizers EnCodec expects at a given target
/// bandwidth. Mirrors the eager `Config::num_quantizers` helper.
///
/// `num_quantizers = (1000 * max_bandwidth) / (frame_rate * 10)`
/// where `frame_rate = ceil(sampling_rate / prod(upsampling_ratios))`.
pub fn encodec_num_quantizers(
    sampling_rate: usize,
    upsampling_ratios: &[usize],
    target_bandwidths: &[f64],
) -> usize {
    let hop_length: usize = upsampling_ratios.iter().product();
    let frame_rate = sampling_rate.div_ceil(hop_length);
    let max_bw = target_bandwidths.last().copied().unwrap_or(0.0);
    let num = 1000.0_f64 * max_bw;
    (num as usize) / (frame_rate * 10)
}

impl EncodecWeights {
    /// Load EnCodec weights from a HuggingFace `MmapedSafetensors`
    /// checkpoint (e.g. `facebook/encodec_24khz/model.safetensors`).
    ///
    /// The HF EnCodec checkpoint stores each `EncodecConv1d` /
    /// `EncodecConvTranspose1d` with a PyTorch `weight_norm`
    /// parametrisation (`weight_g` + `weight_v`), which we fuse into
    /// a single dense kernel at load time. The naming follows the
    /// `EncodecDecoder.layers` `nn.ModuleList` indexing:
    ///
    /// - `decoder.layers.0` = init Conv1d
    /// - `decoder.layers.1` = init LSTM
    /// - For each upsampling ratio (in `cfg.upsampling_ratios` order):
    ///     - `+1` ELU (no params)
    ///     - `+1` ConvTranspose1d
    ///     - `+1` ResnetBlock for each of `cfg.num_residual_layers`
    /// - Final ELU (no params)
    /// - Final Conv1d
    ///
    /// Residual blocks expose two convs at `block.1.conv.*` and
    /// `block.3.conv.*` (their `block` is `[ELU, Conv, ELU, Conv]`),
    /// plus an optional `shortcut.conv.*` 1×1 conv when
    /// `cfg.use_conv_shortcut` is true.
    ///
    /// Quantizer layers are named `quantizer.layers.{i}.codebook.embed`
    /// and the count is derived from the maximum target bandwidth via
    /// [`encodec_num_quantizers`].
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &EncodecConfig,
        sampling_rate: usize,
        target_bandwidths: &[f64],
    ) -> Result<Self> {
        let num_codebooks = encodec_num_quantizers(
            sampling_rate, &cfg.upsampling_ratios, target_bandwidths,
        );
        if num_codebooks == 0 {
            crate::bail!(
                "EncodecWeights::load_from_mmapped: zero quantizers \
                 — check target_bandwidths and upsampling_ratios",
            );
        }

        let mut scaling = 1_usize << cfg.upsampling_ratios.len();

        // decoder.layers.0 — init conv (hidden_size → num_filters * scaling).
        let init_conv = load_encodec_conv1d(
            st, "decoder.layers.0.conv",
            cfg.hidden_size, cfg.num_filters * scaling,
            cfg.last_kernel_size, 1, 1,
        )?;

        // decoder.layers.1 — init LSTM at width `num_filters * scaling`.
        let init_lstm = load_encodec_lstm(
            st, "decoder.layers.1.lstm",
            cfg.num_filters * scaling, cfg.num_lstm_layers,
        )?;

        let mut idx = 2_usize;
        let mut stages: Vec<UpsampleStageWeights> =
            Vec::with_capacity(cfg.upsampling_ratios.len());

        for &ratio in &cfg.upsampling_ratios {
            let current = scaling * cfg.num_filters;
            // ELU has no params, but reserves an index in nn.ModuleList.
            idx += 1;
            let up_conv = load_encodec_conv_transpose1d(
                st, &format!("decoder.layers.{idx}.conv"),
                current, current / 2,
                ratio * 2, ratio,
            )?;
            idx += 1;
            let mut resnets: Vec<ResnetBlockWeights> =
                Vec::with_capacity(cfg.num_residual_layers);
            for j in 0..cfg.num_residual_layers {
                let dim = current / 2;
                let h = dim / cfg.compress;
                let dilation = cfg.dilation_growth_rate.pow(j as u32);
                let conv1 = load_encodec_conv1d(
                    st, &format!("decoder.layers.{idx}.block.1.conv"),
                    dim, h, cfg.residual_kernel_size, 1, dilation,
                )?;
                let conv2 = load_encodec_conv1d(
                    st, &format!("decoder.layers.{idx}.block.3.conv"),
                    h, dim, 1, 1, 1,
                )?;
                let shortcut = if cfg.use_conv_shortcut {
                    Some(load_encodec_conv1d(
                        st, &format!("decoder.layers.{idx}.shortcut.conv"),
                        dim, dim, 1, 1, 1,
                    )?)
                } else {
                    None
                };
                resnets.push(ResnetBlockWeights { conv1, conv2, shortcut });
                idx += 1;
            }
            stages.push(UpsampleStageWeights { up_conv, resnets });
            scaling /= 2;
        }
        // Final ELU.
        idx += 1;
        let final_conv = load_encodec_conv1d(
            st, &format!("decoder.layers.{idx}.conv"),
            cfg.num_filters, cfg.audio_channels,
            cfg.last_kernel_size, 1, 1,
        )?;

        // RVQ codebooks.
        let mut quantizers: Vec<VectorQuantizerWeights> =
            Vec::with_capacity(num_codebooks);
        for i in 0..num_codebooks {
            let embed = load_tensor_as_f32(
                st, &format!("quantizer.layers.{i}.codebook.embed"),
            )?;
            let expected = cfg.codebook_size * cfg.codebook_dim;
            if embed.len() != expected {
                crate::bail!(
                    "quantizer.layers.{i}.codebook.embed: {} elts, expected {}",
                    embed.len(), expected,
                );
            }
            quantizers.push(VectorQuantizerWeights {
                codebook: Arc::from(embed),
            });
        }

        Ok(EncodecWeights {
            quantizers,
            decoder: DecoderWeights {
                init_conv, init_lstm, stages, final_conv,
            },
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
        c_in: usize, c_out: usize, k: usize, stride: usize, dilation: usize,
        bias: bool, nb: &mut dyn FnMut() -> f32,
    ) -> Conv1dWeights {
        Conv1dWeights {
            w: vec_of(c_out * c_in * k, nb),
            b: if bias { Some(vec_of(c_out, nb)) } else { None },
            c_in, c_out, k, stride, dilation,
        }
    }
    fn conv_transpose1d_w(
        c_in: usize, c_out: usize, k: usize, stride: usize, bias: bool,
        nb: &mut dyn FnMut() -> f32,
    ) -> ConvTranspose1dWeights {
        ConvTranspose1dWeights {
            w: vec_of(c_in * c_out * k, nb),
            b: if bias { Some(vec_of(c_out, nb)) } else { None },
            c_in, c_out, k, stride,
        }
    }
    fn resnet_w(dim: usize, cfg: &EncodecConfig, nb: &mut dyn FnMut() -> f32) -> ResnetBlockWeights {
        let h = dim / cfg.compress;
        ResnetBlockWeights {
            conv1: conv1d_w(dim, h, cfg.residual_kernel_size, 1, 1, true, nb),
            conv2: conv1d_w(h, dim, 1, 1, 1, true, nb),
            shortcut: if cfg.use_conv_shortcut {
                Some(conv1d_w(dim, dim, 1, 1, 1, true, nb))
            } else { None },
        }
    }
    fn lstm_cell_w(d: usize, nb: &mut dyn FnMut() -> f32) -> LstmCellWeights {
        let four_h = 4 * d;
        LstmCellWeights {
            w_ih: vec_of(four_h * d, nb),
            w_hh: vec_of(four_h * d, nb),
            b_ih: vec_of(four_h, nb),
            b_hh: vec_of(four_h, nb),
            input_dim: d, hidden_dim: d,
        }
    }

    fn tiny_config() -> EncodecConfig {
        EncodecConfig {
            audio_channels: 1,
            num_filters: 4,
            num_residual_layers: 1,
            upsampling_ratios: vec![2, 2],
            kernel_size: 3,
            last_kernel_size: 3,
            residual_kernel_size: 3,
            dilation_growth_rate: 2,
            use_causal_conv: true,
            pad_mode: PadMode::Constant,
            compress: 2,
            num_lstm_layers: 1,
            use_conv_shortcut: false,
            hidden_size: 16,
            num_codebooks: 2,
            codebook_size: 8,
            codebook_dim: 16,
        }
    }

    fn tiny_weights(cfg: &EncodecConfig) -> EncodecWeights {
        let mut nb = rng_seed(0xE);
        // Decoder mirror of the eager Decoder::new loop:
        // scaling = 2^len(upsampling_ratios) at the start; init_conv goes from
        // hidden_size to num_filters * scaling.
        let mut scaling = 1_usize << cfg.upsampling_ratios.len();
        let init_conv = conv1d_w(
            cfg.hidden_size, cfg.num_filters * scaling,
            cfg.last_kernel_size, 1, 1, true, &mut nb,
        );
        let init_lstm: Vec<LstmCellWeights> = (0..cfg.num_lstm_layers)
            .map(|_| lstm_cell_w(cfg.num_filters * scaling, &mut nb))
            .collect();
        let mut stages = Vec::with_capacity(cfg.upsampling_ratios.len());
        for &ratio in &cfg.upsampling_ratios {
            let current = scaling * cfg.num_filters;
            let up = conv_transpose1d_w(current, current / 2, ratio * 2, ratio, true, &mut nb);
            let resnets: Vec<ResnetBlockWeights> = (0..cfg.num_residual_layers)
                .map(|_| resnet_w(current / 2, cfg, &mut nb))
                .collect();
            stages.push(UpsampleStageWeights { up_conv: up, resnets });
            scaling /= 2;
        }
        let final_conv = conv1d_w(
            cfg.num_filters, cfg.audio_channels,
            cfg.last_kernel_size, 1, 1, true, &mut nb,
        );

        let quantizers: Vec<VectorQuantizerWeights> = (0..cfg.num_codebooks)
            .map(|_| VectorQuantizerWeights {
                codebook: vec_of(cfg.codebook_size * cfg.codebook_dim, &mut nb),
            })
            .collect();

        EncodecWeights {
            quantizers,
            decoder: DecoderWeights {
                init_conv, init_lstm, stages, final_conv,
            },
        }
    }

    #[test]
    fn decode_codes_shape_and_finite() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = EncodecModel { config: cfg.clone(), weights };
        let time = 4_usize;
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
        let shape = audio.shape();
        let dims = shape.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], cfg.audio_channels);
        assert!(dims[2] > 0);
        for &v in &audio.realize_f32() {
            assert!(v.is_finite(), "non-finite audio sample: {v}");
        }
    }

    /// Different codes must produce different audio.
    #[test]
    fn decode_codes_responds_to_codes() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = EncodecModel { config: cfg.clone(), weights };
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
        assert!(max_diff > 1e-9,
            "decoded audio must respond to code changes, max_diff = {max_diff}");
    }

    /// Replicate padding sanity check: edge value repeats.
    #[test]
    fn pad1d_replicate_edges() {
        let dev = Device::cpu();
        let x = LazyTensor::from_f32(
            vec![1.0_f32, 2.0, 3.0, 4.0],
            Shape::from_dims(&[1, 1, 4]), &dev,
        );
        let y = pad1d(&x, 2, 2, PadMode::Replicate, &x).unwrap();
        let got = y.realize_f32();
        // Left pad 2 = [1, 1]; right pad 2 = [4, 4].
        assert_eq!(got, vec![1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn pad1d_constant_zero() {
        let dev = Device::cpu();
        let x = LazyTensor::from_f32(
            vec![1.0_f32, 2.0, 3.0, 4.0],
            Shape::from_dims(&[1, 1, 4]), &dev,
        );
        let y = pad1d(&x, 1, 1, PadMode::Constant, &x).unwrap();
        let got = y.realize_f32();
        assert_eq!(got, vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0]);
    }

    #[test]
    fn preset_constructs() {
        let cfg = EncodecConfig::default_preset();
        assert_eq!(cfg.upsampling_ratios, vec![8, 5, 4, 2]);
        assert_eq!(cfg.num_filters, 32);
        assert_eq!(cfg.hidden_size, 128);
        assert_eq!(cfg.num_lstm_layers, 2);
    }
}

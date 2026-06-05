//! SNAC — lazy port (decoder + per-stride RVQ).
//!
//! Multi-scale codes (one stream per stride level) → waveform via:
//!   1. Per-stride RVQ reconstruction: for each codebook,
//!      embedding lookup + 1×1 out_proj, then upsample (via
//!      repeat-interleave along time) to match the highest-rate
//!      stream's resolution, summed across codebooks.
//!   2. Decoder: init_conv (depthwise+pointwise or standard) →
//!      optional LocalMHA → N DecoderBlocks → Snake1d → final_conv
//!      → waveform `(1, audio_channels, T_out)`.
//!
//! DecoderBlock: Snake1d → ConvTranspose1d (stride=ratio) →
//! optional NoiseBlock → 3 ResidualUnits (dilations 1, 3, 9).
//!
//! v1 scope:
//!   - F32, batch == 1, decode-only.
//!   - NoiseBlock is deterministic-skipped (matches `noise=false`
//!     configs); injecting Gaussian noise needs a graph-level
//!     `randn` primitive which is a follow-up.
//!   - LocalMHA without rotary positional embeddings (matches the
//!     `use_rotary_pos_emb=false` path; rotary is a follow-up
//!     when SNAC checkpoints that use it materialize).
//!   - Depthwise and standard init_conv both supported.
//!   - Per-stride RVQ: each codebook has its own stride; codes are
//!     upsampled via repeat-interleave to the highest stride before
//!     summation.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_dac::expand_conv1d_weight_for_dilation_if_needed;
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct SnacConfig {
    pub audio_channels: usize,
    pub encoder_dim: usize,
    /// Per-decoder-stage upsampling stride.
    pub decoder_rates: Vec<usize>,
    /// Optional local-attention window size at the bottleneck.
    pub attn_window_size: Option<usize>,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    /// One stride per codebook (per-codebook temporal resolution).
    pub vq_strides: Vec<usize>,
    /// Whether the decoder injects noise (set false for v1
    /// deterministic decode — eager `noise=true` configs add a
    /// Gaussian draw scaled by a 1×1 conv per DecoderBlock).
    pub noise: bool,
    pub depthwise: bool,
    pub decoder_dim: usize,
}

impl SnacConfig {
    /// `hubertsiuzdak/snac_24khz` preset.
    pub fn snac_24khz() -> Self {
        Self {
            audio_channels: 1,
            encoder_dim: 48,
            decoder_dim: 1024,
            decoder_rates: vec![8, 8, 4, 2],
            attn_window_size: None,
            codebook_size: 4096,
            codebook_dim: 8,
            vq_strides: vec![8, 4, 2, 1],
            noise: true,
            depthwise: true,
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
    pub pad: usize,
    pub groups: usize,
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
    pub pad: usize,
    pub out_pad: usize,
}

#[derive(Debug, Clone)]
pub struct Snake1dWeights {
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

/// LocalMHA-without-rotary. norm → qkv linear → reshape → softmax
/// attention → out linear → +residual.
#[derive(Debug, Clone)]
pub struct LocalMhaWeights {
    pub norm_gain: Arc<[f32]>,
    pub norm_bias: Arc<[f32]>,
    pub to_qkv: WeightStorage,
    pub to_out: WeightStorage,
    pub num_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone)]
pub struct DecoderBlockWeights {
    pub snake1: Snake1dWeights,
    pub conv_tr1: ConvTranspose1dWeights,
    /// Present iff `cfg.noise == true`. The 1×1 conv that scales the
    /// Gaussian noise draw. v1 forward currently skips this path
    /// (deterministic decode), but the weights are kept for parity
    /// with the eager loader.
    pub noise_linear: Option<Conv1dWeights>,
    pub res1: ResidualUnitWeights,
    pub res2: ResidualUnitWeights,
    pub res3: ResidualUnitWeights,
}

/// Decoder init Conv1d — either a single standard conv or a
/// depthwise + pointwise pair.
#[derive(Debug, Clone)]
pub enum InitConvWeights {
    Standard(Conv1dWeights),
    Depthwise(Conv1dWeights, Conv1dWeights),
}

#[derive(Debug, Clone)]
pub struct DecoderWeights {
    pub init_conv: InitConvWeights,
    pub local_mha: Option<LocalMhaWeights>,
    pub blocks: Vec<DecoderBlockWeights>,
    pub final_snake: Snake1dWeights,
    pub final_conv: Conv1dWeights,
}

#[derive(Debug, Clone)]
pub struct VectorQuantizerWeights {
    /// `[codebook_size, codebook_dim]`.
    pub codebook: Arc<[f32]>,
    /// 1×1 conv `[in_dim, codebook_dim]` (PyTorch shape).
    pub out_proj: Conv1dWeights,
    /// Per-codebook temporal stride. Codes at this resolution must
    /// be upsampled by `stride` before summation.
    pub stride: usize,
}

#[derive(Debug, Clone)]
pub struct SnacWeights {
    pub quantizers: Vec<VectorQuantizerWeights>,
    pub decoder: DecoderWeights,
}

#[derive(Debug, Clone)]
pub struct SnacModel {
    pub config: SnacConfig,
    pub weights: SnacWeights,
}

// ---- Forward ---------------------------------------------------------------

impl SnacModel {
    /// Decode multi-stride codes back to a waveform. `codes` is a
    /// list, one per codebook, each a U32 LazyTensor of shape
    /// `(1, T_i)` where `T_i = T_max / vq_strides[i]`.
    ///
    /// `T_max` is the highest-resolution time axis (= length of
    /// `codes[k]` for the codebook with the smallest stride).
    pub fn decode_codes(&self, codes: &[LazyTensor]) -> Result<LazyTensor> {
        assert_eq!(codes.len(), self.weights.quantizers.len(),
            "codes count {} must match codebooks {}",
            codes.len(), self.weights.quantizers.len());
        let anchor = &codes[0];
        let latent = self.rvq_from_codes(codes, anchor)?;
        self.decoder_forward(&latent)
    }

    /// Compute the latent as the sum of per-codebook decoded streams,
    /// each upsampled (via repeat-interleave) to the highest-rate
    /// resolution.
    fn rvq_from_codes(
        &self, codes: &[LazyTensor], anchor: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        // Find the max time resolution (smallest stride → most frames).
        let max_t = codes
            .iter()
            .zip(self.weights.quantizers.iter())
            .map(|(c, q)| c.shape().dims()[1] * q.stride)
            .max()
            .unwrap_or(0);

        let mut sum: Option<LazyTensor> = None;
        for (c, q) in codes.iter().zip(self.weights.quantizers.iter()) {
            let dims = c.shape();
            let dims = dims.dims();
            let t_i = dims[1];
            // Embedding lookup: codebook[ids] → (T_i, codebook_dim).
            let codebook = anchor.const_f32_like(
                Arc::clone(&q.codebook),
                Shape::from_dims(&[cfg.codebook_size, cfg.codebook_dim]),
            );
            let ids_flat = c.reshape(Shape::from_dims(&[t_i]))?;
            // (T_i, cb_dim) → (1, cb_dim, T_i)
            let z_p = codebook
                .index_select(0_usize, &ids_flat)?
                .reshape(Shape::from_dims(&[1, t_i, cfg.codebook_dim]))?
                .permute([0, 2, 1_usize])?;
            // out_proj 1×1 conv: codebook_dim → input_dim.
            let z_q = apply_conv1d(&z_p, &q.out_proj, anchor)?;
            // Upsample by `q.stride` if needed via repeat-interleave
            // along the last (time) axis.
            let z_q = z_q.repeat_interleave(2_usize, q.stride)?;
            // Sanity: shapes match the target resolution.
            let z_dims = z_q.shape();
            let z_dims = z_dims.dims();
            assert_eq!(z_dims[2], max_t,
                "after upsampling, codebook stream length {} != max {}",
                z_dims[2], max_t);
            sum = Some(match sum {
                None => z_q,
                Some(s) => s.add(&z_q)?,
            });
        }
        sum.ok_or_else(|| {
            fuel_core_types::Error::Msg("SNAC RVQ: no codebooks".into()).bt()
        })
    }

    fn decoder_forward(&self, latent: &LazyTensor) -> Result<LazyTensor> {
        let dec = &self.weights.decoder;
        let mut x = match &dec.init_conv {
            InitConvWeights::Standard(c) => apply_conv1d(latent, c, latent)?,
            InitConvWeights::Depthwise(dw, pw) => {
                let h = apply_conv1d(latent, dw, latent)?;
                apply_conv1d(&h, pw, latent)?
            }
        };
        if let Some(mha) = &dec.local_mha {
            x = apply_local_mha(&x, mha, latent)?;
        }
        for block in &dec.blocks {
            x = apply_decoder_block(&x, block, latent, self.config.noise)?;
        }
        x = apply_snake1d(&x, &dec.final_snake, latent)?;
        apply_conv1d(&x, &dec.final_conv, latent)
    }
}

// ---- Component helpers -----------------------------------------------------

fn apply_conv1d(
    x: &LazyTensor, c: &Conv1dWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let (w_data, k_eff) = expand_conv1d_weight_for_dilation_if_needed(
        &c.w, c.c_out, c.c_in / c.groups, c.k, c.dilation,
    );
    let w = anchor.const_f32_like(
        Arc::<[f32]>::from(w_data),
        Shape::from_dims(&[c.c_out, c.c_in / c.groups, k_eff]),
    );
    let bias = c.b.as_ref().map(|b| {
        anchor.const_f32_like(Arc::clone(b), Shape::from_dims(&[c.c_out]))
    });
    x.conv1d(&w, bias.as_ref(), c.stride, c.pad, c.groups)
}

fn apply_conv_transpose1d(
    x: &LazyTensor, c: &ConvTranspose1dWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let w = anchor.const_f32_like(
        Arc::clone(&c.w),
        Shape::from_dims(&[c.c_in, c.c_out, c.k]),
    );
    let mut out = x.conv_transpose1d(&w, c.stride, c.pad, c.out_pad, 1, 1)?;
    if let Some(b) = &c.b {
        let bias = anchor
            .const_f32_like(Arc::clone(b), Shape::from_dims(&[c.c_out]))
            .reshape(Shape::from_dims(&[1, c.c_out, 1]))?;
        out = out.broadcast_add(&bias)?;
    }
    Ok(out)
}

/// `Snake(x) = x + sin²(α · x) / (α + 1e-9)` per-channel.
fn apply_snake1d(
    x: &LazyTensor, s: &Snake1dWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    assert_eq!(dims[1], s.channels);
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
    x: &LazyTensor, r: &ResidualUnitWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let y = apply_snake1d(x, &r.snake1, anchor)?;
    let y = apply_conv1d(&y, &r.conv1, anchor)?;
    let y = apply_snake1d(&y, &r.snake2, anchor)?;
    let y = apply_conv1d(&y, &r.conv2, anchor)?;
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
    x: &LazyTensor, b: &DecoderBlockWeights, anchor: &LazyTensor,
    noise: bool,
) -> Result<LazyTensor> {
    let y = apply_snake1d(x, &b.snake1, anchor)?;
    let mut y = apply_conv_transpose1d(&y, &b.conv_tr1, anchor)?;
    if noise {
        // v1: deterministic decode — noise term skipped. The NoiseBlock
        // adds `noise · (1×1 conv)(y)` where noise ∼ N(0,1). Adding a
        // graph-level randn primitive is a follow-up.
        let _ = &b.noise_linear; // weights are kept for the eventual port.
    }
    y = apply_residual_unit(&y, &b.res1, anchor)?;
    y = apply_residual_unit(&y, &b.res2, anchor)?;
    apply_residual_unit(&y, &b.res3, anchor)
}

/// Local multi-head attention without rotary. Pre-LN → qkv linear
/// → softmax attention → out linear → +residual.
fn apply_local_mha(
    x: &LazyTensor, w: &LocalMhaWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let c = dims[1]; let t = dims[2];
    let residual = x.clone();
    // (B, C, T) → (B, T, C) for LN.
    let x_btc = x.permute([0, 2, 1_usize])?;
    let _ = c;
    let normed = x_btc.layer_norm_affine(
        Arc::clone(&w.norm_gain), Arc::clone(&w.norm_bias), 1e-5,
    )?;
    // qkv: linear B,T,C → B,T,3C.
    let qkv = w.to_qkv.apply_linear(&normed, c, 3 * c);
    let q = qkv.narrow(2_usize, 0, c)?;
    let k = qkv.narrow(2_usize, c, c)?;
    let v = qkv.narrow(2_usize, 2 * c, c)?;
    let n_heads = w.num_heads;
    let head_dim = w.head_dim;
    let scale = 1.0_f64 / (head_dim as f64).sqrt();
    let _ = (b, t, c);
    let q = q.split_heads(n_heads, head_dim)?;
    let k = k.split_heads(n_heads, head_dim)?;
    let v = v.split_heads(n_heads, head_dim)?;
    let kt = k.permute([0, 1, 3, 2_usize])?;
    let scores = q.matmul(&kt)?.mul_scalar(scale);
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?.merge_heads()?;
    let out = w.to_out.apply_linear(&ctx, c, c);
    // (B, T, C) → (B, C, T) for residual add.
    let out_chw = out.permute([0, 2, 1_usize])?;
    out_chw.add(&residual)
}

// `repeat_interleave_last_dim` retired — call sites now use the
// public `LazyTensor::repeat_interleave(dim, repeats)` method
// shipped earlier this session.

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
    fn conv1d_w(
        c_in: usize, c_out: usize, k: usize, stride: usize, pad: usize, groups: usize,
        dilation: usize, bias: bool, nb: &mut dyn FnMut() -> f32,
    ) -> Conv1dWeights {
        Conv1dWeights {
            w: vec_of(c_out * (c_in / groups) * k, nb),
            b: if bias { Some(vec_of(c_out, nb)) } else { None },
            c_in, c_out, k, stride, pad, groups, dilation,
        }
    }
    fn conv_transpose1d_w(
        c_in: usize, c_out: usize, k: usize, stride: usize, pad: usize, out_pad: usize,
        bias: bool, nb: &mut dyn FnMut() -> f32,
    ) -> ConvTranspose1dWeights {
        ConvTranspose1dWeights {
            w: vec_of(c_in * c_out * k, nb),
            b: if bias { Some(vec_of(c_out, nb)) } else { None },
            c_in, c_out, k, stride, pad, out_pad,
        }
    }
    fn snake_w(channels: usize, nb: &mut dyn FnMut() -> f32) -> Snake1dWeights {
        Snake1dWeights { alpha: vec_of(channels, nb), channels }
    }
    fn res_unit_w(
        dim: usize, dilation: usize, groups: usize, nb: &mut dyn FnMut() -> f32,
    ) -> ResidualUnitWeights {
        let pad = ((7 - 1) * dilation) / 2;
        ResidualUnitWeights {
            snake1: snake_w(dim, nb),
            conv1: conv1d_w(dim, dim, 7, 1, pad, groups, dilation, true, nb),
            snake2: snake_w(dim, nb),
            conv2: conv1d_w(dim, dim, 1, 1, 0, 1, 1, true, nb),
        }
    }
    fn dec_block_w(
        in_dim: usize, out_dim: usize, stride: usize, groups: usize, noise: bool,
        nb: &mut dyn FnMut() -> f32,
    ) -> DecoderBlockWeights {
        DecoderBlockWeights {
            snake1: snake_w(in_dim, nb),
            conv_tr1: conv_transpose1d_w(
                in_dim, out_dim, 2 * stride, stride, stride.div_ceil(2), stride % 2, true, nb,
            ),
            noise_linear: if noise {
                Some(conv1d_w(out_dim, out_dim, 1, 1, 0, 1, 1, false, nb))
            } else { None },
            res1: res_unit_w(out_dim, 1, groups, nb),
            res2: res_unit_w(out_dim, 3, groups, nb),
            res3: res_unit_w(out_dim, 9, groups, nb),
        }
    }

    fn tiny_config() -> SnacConfig {
        SnacConfig {
            audio_channels: 1,
            encoder_dim: 8,
            decoder_dim: 32,
            decoder_rates: vec![2, 2],
            attn_window_size: None,
            codebook_size: 8,
            codebook_dim: 4,
            vq_strides: vec![2, 1],
            noise: false,
            depthwise: false,
        }
    }

    fn tiny_weights(cfg: &SnacConfig) -> SnacWeights {
        let mut nb = rng_seed(0x5);
        // Decoder mirror: init_conv goes from sum-of-out_proj to decoder_dim.
        // For v1 simplicity, in_dim = max(codebook_dim, decoder_dim_in) — eager
        // SNAC's out_proj projects to a per-codebook in_dim that equals the
        // hidden in_dim (typically = decoder_dim's input channel count).
        let in_dim = cfg.decoder_dim;
        let mut channels = cfg.decoder_dim;
        let init_conv = InitConvWeights::Standard(
            conv1d_w(in_dim, channels, 7, 1, 3, 1, 1, true, &mut nb),
        );
        let mut blocks = Vec::with_capacity(cfg.decoder_rates.len());
        for &stride in &cfg.decoder_rates {
            let next = channels / 2;
            blocks.push(dec_block_w(channels, next, stride, 1, cfg.noise, &mut nb));
            channels = next;
        }
        let final_snake = snake_w(channels, &mut nb);
        let final_conv = conv1d_w(channels, cfg.audio_channels, 7, 1, 3, 1, 1, true, &mut nb);

        let quantizers: Vec<VectorQuantizerWeights> = cfg.vq_strides.iter()
            .map(|&s| VectorQuantizerWeights {
                codebook: vec_of(cfg.codebook_size * cfg.codebook_dim, &mut nb),
                out_proj: conv1d_w(cfg.codebook_dim, in_dim, 1, 1, 0, 1, 1, true, &mut nb),
                stride: s,
            })
            .collect();

        SnacWeights {
            quantizers,
            decoder: DecoderWeights {
                init_conv,
                local_mha: None,
                blocks,
                final_snake,
                final_conv,
            },
        }
    }

    #[test]
    fn repeat_interleave_last_dim_via_public_api() {
        let dev = Device::cpu();
        // (1, 2, 3) → repeat last dim by 2 → (1, 2, 6).
        let x = LazyTensor::from_f32(
            vec![1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0],
            Shape::from_dims(&[1, 2, 3]), &dev,
        );
        let y = x.repeat_interleave(2_usize, 2).unwrap();
        assert_eq!(y.shape().dims(), &[1, 2, 6]);
        let got = y.realize_f32();
        assert_eq!(got, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0]);
    }

    #[test]
    fn decode_codes_shape_and_finite() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = SnacModel { config: cfg.clone(), weights };
        // For vq_strides = [2, 1], the max-resolution T = T_q0 * 2 = T_q1 * 1.
        // Use T_q0 = 2 → T_q1 = 4 → max_t = 4.
        let dev = Device::cpu();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32; 1], Shape::from_dims(&[1]), &dev,
        );
        let c0 = anchor.const_u32_like(
            vec![0_u32, 1], Shape::from_dims(&[1, 2]),
        );
        let c1 = anchor.const_u32_like(
            vec![2_u32, 3, 4, 5], Shape::from_dims(&[1, 4]),
        );
        let audio = model.decode_codes(&[c0, c1]).unwrap();
        let dims = audio.shape();
        let dims = dims.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], cfg.audio_channels);
        assert!(dims[2] > 0);
        for &v in &audio.realize_f32() {
            assert!(v.is_finite(), "non-finite audio sample: {v}");
        }
    }

    #[test]
    fn decode_codes_responds_to_codes() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = SnacModel { config: cfg.clone(), weights };
        let dev = Device::cpu();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32; 1], Shape::from_dims(&[1]), &dev,
        );
        let codes_a = vec![
            anchor.const_u32_like(vec![0_u32; 2], Shape::from_dims(&[1, 2])),
            anchor.const_u32_like(vec![0_u32; 4], Shape::from_dims(&[1, 4])),
        ];
        let codes_b = vec![
            anchor.const_u32_like(vec![3_u32; 2], Shape::from_dims(&[1, 2])),
            anchor.const_u32_like(vec![5_u32; 4], Shape::from_dims(&[1, 4])),
        ];
        let a = model.decode_codes(&codes_a).unwrap().realize_f32();
        let b = model.decode_codes(&codes_b).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-9,
            "audio must respond to code changes, max_diff = {max_diff}");
    }

    #[test]
    fn local_mha_shape_and_finite() {
        let mut nb = rng_seed(11);
        let c = 8; let t = 4;
        let mha = LocalMhaWeights {
            norm_gain: Arc::from(vec![1.0_f32; c]),
            norm_bias: Arc::from(vec![0.0_f32; c]),
            to_qkv: ws(c * 3 * c, &mut nb),
            to_out: ws(c * c, &mut nb),
            num_heads: 2, head_dim: c / 2,
        };
        let x = LazyTensor::from_f32(
            (0..(1 * c * t)).map(|i| (i as f32) * 0.05).collect::<Vec<_>>(),
            Shape::from_dims(&[1, c, t]), &Device::cpu(),
        );
        let out = apply_local_mha(&x, &mha, &x).unwrap();
        assert_eq!(out.shape().dims(), &[1, c, t]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite mha output: {v}");
        }
    }

    #[test]
    fn snake_alpha_zero_is_identity() {
        let x = LazyTensor::from_f32(
            vec![0.5_f32, -0.25, 0.75, 1.0],
            Shape::from_dims(&[1, 2, 2]), &Device::cpu(),
        );
        let snake = Snake1dWeights {
            alpha: Arc::from(vec![0.0_f32; 2]), channels: 2,
        };
        let out = apply_snake1d(&x, &snake, &x).unwrap().realize_f32();
        let in_data = x.realize_f32();
        for (a, b) in out.iter().zip(in_data.iter()) {
            assert!((a - b).abs() < 1e-5,
                "α=0 should be identity: {a} vs {b}");
        }
    }

    #[test]
    fn snac_24khz_preset_constructs() {
        let cfg = SnacConfig::snac_24khz();
        assert_eq!(cfg.decoder_rates, vec![8, 8, 4, 2]);
        assert_eq!(cfg.vq_strides, vec![8, 4, 2, 1]);
        assert_eq!(cfg.codebook_size, 4096);
    }
}

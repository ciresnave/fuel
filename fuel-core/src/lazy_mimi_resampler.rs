//! Mimi resampler — lazy port.
//!
//! Thin wrappers around a strided causal `Conv1d` (for downsampling)
//! and a strided causal `ConvTranspose1d` (for upsampling). Mimi
//! uses these between the SeaNet encoder's natural frame rate
//! (`sample_rate / Π ratios`) and the user-configured `frame_rate`
//! (e.g. 12.5 Hz at the RVQ).
//!
//! Both convs use:
//!   - `kernel = 2 × stride`
//!   - `causal` left-padding (`Replicate` mode for the downsample,
//!     no padding for the upsample with causal trim of `kernel −
//!     stride` from the right)
//!   - `bias = false`
//!   - `learnt = true` (the only mode supported by the eager port;
//!     `learnt = false` static linear resample is not implemented)
//!   - `ConvDownsample1d` uses `groups = 1` (full-channel mix)
//!   - `ConvTrUpsample1d` uses `groups = dim` (depthwise)
//!
//! Forward-only inference. Streaming `step` API not supported here
//! — see [`crate::lazy_mimi_seanet`] for the same design choice.

use crate::lazy::LazyTensor;
use crate::lazy_encodec::{pad1d, PadMode};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ConvDownsample1dWeights {
    /// `(dim, dim, kernel = 2·stride)`.
    pub weight: Arc<[f32]>,
    pub dim: usize,
    pub stride: usize,
}

#[derive(Debug, Clone)]
pub struct ConvTrUpsample1dWeights {
    /// `(dim, 1, kernel = 2·stride)` — depthwise transposed conv.
    pub weight: Arc<[f32]>,
    pub dim: usize,
    pub stride: usize,
}

#[derive(Debug, Clone)]
pub struct ConvDownsample1dModel {
    pub weights: ConvDownsample1dWeights,
}

#[derive(Debug, Clone)]
pub struct ConvTrUpsample1dModel {
    pub weights: ConvTrUpsample1dWeights,
}

impl ConvDownsample1dModel {
    /// `(1, dim, T)` → `(1, dim, T / stride)`. Left-pad with
    /// `(kernel - stride)` samples in `Replicate` mode (the only
    /// pad_mode the eager port uses for this layer).
    pub fn forward(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let w = &self.weights;
        let kernel = 2 * w.stride;
        let pad_total = kernel.saturating_sub(w.stride);
        let padded = pad1d(x, pad_total, 0, PadMode::Replicate, x)?;
        let weight = padded.const_f32_like(
            Arc::clone(&w.weight),
            Shape::from_dims(&[w.dim, w.dim, kernel]),
        );
        padded.conv1d(&weight, None, w.stride, 0, 1)
    }
}

impl ConvTrUpsample1dModel {
    /// `(1, dim, T)` → `(1, dim, T · stride)`. Depthwise causal
    /// transpose-conv: natural output length is `(T - 1) · stride
    /// + kernel`; trim the trailing `kernel - stride` samples for
    /// causality.
    pub fn forward(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let w = &self.weights;
        let kernel = 2 * w.stride;
        let weight = x.const_f32_like(
            Arc::clone(&w.weight),
            Shape::from_dims(&[w.dim, 1, kernel]),
        );
        let y = x.conv_transpose1d(
            &weight, w.stride, /* padding */ 0, /* output_padding */ 0,
            /* dilation */ 1, /* groups */ w.dim,
        )?;
        // Causal trim.
        let dims = y.shape().dims().to_vec();
        let t_out = dims[2];
        let trim = kernel.saturating_sub(w.stride);
        let keep = t_out.saturating_sub(trim);
        y.narrow(2_usize, 0, keep)
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl ConvDownsample1dWeights {
    /// Load `ConvDownsample1dWeights` from a HuggingFace
    /// `MmapedSafetensors` checkpoint at `{prefix}` (typically
    /// `"downsample"`). The eager `ConvDownsample1d` wraps a
    /// `StreamableConv1d` with `groups = 1`, `bias = false`, `norm =
    /// None` and `kernel = 2 · stride`. The on-disk path collapses
    /// to `{prefix}.conv.weight` — the single `pp("conv")` step from
    /// `NormConv1d` is the only prefix the `fuel_nn::conv1d_no_bias`
    /// `vb` sees. Matches the path already used by
    /// `MimiWeights::load_from_mmapped` (`"downsample.conv.weight"`).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        dim: usize,
        stride: usize,
    ) -> Result<Self> {
        use crate::lazy::load_tensor_as_f32;
        let kernel = 2 * stride;
        let expected = dim * dim * kernel;
        let w = load_tensor_as_f32(st, &format!("{prefix}.conv.weight"))?;
        if w.len() != expected {
            crate::bail!(
                "{prefix}.conv.weight: {} elements, expected {expected} ({dim}×{dim}×{kernel})",
                w.len(),
            );
        }
        let weight: Arc<[f32]> = Arc::from(w);
        Ok(ConvDownsample1dWeights {
            weight,
            dim,
            stride,
        })
    }
}

impl ConvTrUpsample1dWeights {
    /// Load `ConvTrUpsample1dWeights` from a HuggingFace
    /// `MmapedSafetensors` checkpoint at `{prefix}` (typically
    /// `"upsample"`). The eager `ConvTrUpsample1d` wraps a
    /// `StreamableConvTranspose1d` with `groups = dim` (depthwise),
    /// `bias = false`, `norm = None`, so PyTorch lays the weight out
    /// as `[in_c, out_c / groups, k] = [dim, 1, 2·stride]`. The
    /// `NormConvTranspose1d::new` `let vb = vb.pp("conv")` step is
    /// the only conv-prefix the `vb.get(..., "weight")` call sees, so
    /// the on-disk path is `{prefix}.convtr.weight` — matching the
    /// path already used by `MimiWeights::load_from_mmapped`
    /// (`"upsample.convtr.weight"`) and the HuggingFace
    /// `kyutai/mimi` checkpoint.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        dim: usize,
        stride: usize,
    ) -> Result<Self> {
        use crate::lazy::load_tensor_as_f32;
        let kernel = 2 * stride;
        let expected = dim * 1 * kernel;
        let w = load_tensor_as_f32(st, &format!("{prefix}.convtr.weight"))?;
        if w.len() != expected {
            crate::bail!(
                "{prefix}.convtr.weight: {} elements, expected {expected} ({dim}×1×{kernel})",
                w.len(),
            );
        }
        let weight: Arc<[f32]> = Arc::from(w);
        Ok(ConvTrUpsample1dWeights {
            weight,
            dim,
            stride,
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

    #[test]
    fn downsample_shapes() {
        let mut nb = rng_seed(2026);
        let dim = 4; let stride = 2;
        let model = ConvDownsample1dModel {
            weights: ConvDownsample1dWeights {
                weight: vec_of(dim * dim * 2 * stride, &mut nb),
                dim, stride,
            },
        };
        let t_in = 8;
        let x = LazyTensor::from_f32(
            (0..(1 * dim * t_in)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, dim, t_in]), &Device::cpu(),
        );
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[1, dim, t_in / stride]);
        for &v in &y.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn upsample_shapes() {
        let mut nb = rng_seed(2027);
        let dim = 4; let stride = 2;
        let model = ConvTrUpsample1dModel {
            weights: ConvTrUpsample1dWeights {
                weight: vec_of(dim * 1 * 2 * stride, &mut nb),
                dim, stride,
            },
        };
        let t_in = 5;
        let x = LazyTensor::from_f32(
            (0..(1 * dim * t_in)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, dim, t_in]), &Device::cpu(),
        );
        let y = model.forward(&x).unwrap();
        // Causal-trimmed output length = T · stride exactly.
        assert_eq!(y.shape().dims(), &[1, dim, t_in * stride]);
        for &v in &y.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn downsample_then_upsample_shape_round_trip() {
        let mut nb = rng_seed(2028);
        let dim = 4; let stride = 2;
        let dn = ConvDownsample1dModel {
            weights: ConvDownsample1dWeights {
                weight: vec_of(dim * dim * 2 * stride, &mut nb),
                dim, stride,
            },
        };
        let up = ConvTrUpsample1dModel {
            weights: ConvTrUpsample1dWeights {
                weight: vec_of(dim * 1 * 2 * stride, &mut nb),
                dim, stride,
            },
        };
        let t_in = 6;
        let x = LazyTensor::from_f32(
            (0..(1 * dim * t_in)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, dim, t_in]), &Device::cpu(),
        );
        let mid = dn.forward(&x).unwrap();
        assert_eq!(mid.shape().dims(), &[1, dim, t_in / stride]);
        let back = up.forward(&mid).unwrap();
        assert_eq!(back.shape().dims(), &[1, dim, t_in]);
    }
}

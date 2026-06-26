//! Conv3D for Qwen3-VL temporal patch embedding, decomposed into
//! two parallel Conv2Ds.
//!
//! Qwen3-VL receives video as `(B, in_c, T, H, W)` and applies a 3D
//! patch embedding with kernel `(2, kH, kW)` and stride `(2, kH, kW)`.
//! Because the temporal kernel depth is 2 (i.e. each output frame
//! consumes exactly two input frames), the convolution decomposes
//! cleanly into a pair of 2D convs whose outputs are summed:
//!
//! ```text
//!     y[..., t_out, :, :] = conv2d(x[..., 2*t_out + 0, :, :], w[..., 0, :, :])
//!                         + conv2d(x[..., 2*t_out + 1, :, :], w[..., 1, :, :])
//! ```
//!
//! There is no native lazy Conv3D op — adding one would be
//! significantly more involved than the single (Qwen3-VL) consumer
//! warrants. This module ports
//! `fuel_transformers::models::multimodal::qwen3_vl::conv3d_temporal_2`
//! using only the existing [`LazyTensor::conv2d`] primitive plus
//! `narrow` / `squeeze` / `add` / `unsqueeze`.
//!
//! ## Scope
//!
//! - **v1 (this module)**: kernel depth exactly 2. Input must have
//!   `T == 2` along the temporal axis; output has `T == 1`. The eager
//!   `Conv3dNoBias::forward` is bit-for-bit identical in scope.
//! - **Future**: generalize to arbitrary even kernel depth `N` if a
//!   second consumer needs it (sum of N parallel 2D convs over chunks
//!   of N consecutive frames, stride N).

use crate::lazy::LazyTensor;
use crate::Result;
use fuel_ir::Shape;
use std::sync::Arc;

/// 3D convolution config restricted to the temporal-patch-2 case.
/// Matches eager `Conv3dConfig` byte-for-byte; `dilation` is accepted
/// only when equal to 1 since the underlying lazy Conv2D primitive
/// doesn't expose dilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv3dTemporal2Config {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv3dTemporal2Config {
    fn default() -> Self {
        Self { padding: 0, stride: 1, dilation: 1, groups: 1 }
    }
}

/// Weight storage for a temporal-patch-2 Conv3D.
///
/// Eager stores the raw 3D weight `(Cout, Cin/groups, 2, kH, kW)`
/// and slices it at construction time to produce the two 2D weights
/// `w1 = ws[:, :, 0, :, :]` and `w2 = ws[:, :, 1, :, :]`. The lazy
/// port does the same split at construction so each forward pass
/// just emits two const nodes (cheap Arc-clones) — no per-step
/// slicing in the graph.
#[derive(Debug, Clone)]
pub struct Conv3dTemporal2Weights {
    /// `(Cout, Cin/groups, kH, kW)` — temporal slice index 0.
    pub w1: Arc<[f32]>,
    /// `(Cout, Cin/groups, kH, kW)` — temporal slice index 1.
    pub w2: Arc<[f32]>,
    pub out_channels: usize,
    pub in_channels_per_group: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub cfg: Conv3dTemporal2Config,
}

impl Conv3dTemporal2Weights {
    /// Build from a flat raw weight buffer in PyTorch
    /// `(Cout, Cin/groups, 2, kH, kW)` row-major order.
    ///
    /// Splits the temporal axis into two `(Cout, Cin/groups, kH, kW)`
    /// slabs by interleaved indexing: the temporal axis is stride
    /// `kH * kW` within each `(Cout, Cin/groups, 2, ...)` block of
    /// length `2 * kH * kW`.
    pub fn from_raw_weight(
        raw_weight: &[f32],
        out_channels: usize,
        in_channels_per_group: usize,
        kernel_h: usize,
        kernel_w: usize,
        cfg: Conv3dTemporal2Config,
    ) -> Result<Self> {
        let expected = out_channels * in_channels_per_group * 2 * kernel_h * kernel_w;
        if raw_weight.len() != expected {
            return Err(crate::Error::Msg(format!(
                "Conv3dTemporal2Weights::from_raw_weight: expected {expected} floats \
                 ({out_channels} * {in_channels_per_group} * 2 * {kernel_h} * {kernel_w}), got {}",
                raw_weight.len(),
            )));
        }
        if cfg.dilation != 1 {
            return Err(crate::Error::Msg(format!(
                "Conv3dTemporal2Weights: dilation must be 1 (lazy Conv2D primitive does not support \
                 dilation), got {}",
                cfg.dilation,
            )));
        }
        let plane = kernel_h * kernel_w;
        let outer = out_channels * in_channels_per_group;
        let mut w1 = Vec::with_capacity(outer * plane);
        let mut w2 = Vec::with_capacity(outer * plane);
        for o in 0..outer {
            let base = o * 2 * plane;
            w1.extend_from_slice(&raw_weight[base..base + plane]);
            w2.extend_from_slice(&raw_weight[base + plane..base + 2 * plane]);
        }
        Ok(Self {
            w1: Arc::from(w1),
            w2: Arc::from(w2),
            out_channels,
            in_channels_per_group,
            kernel_h,
            kernel_w,
            cfg,
        })
    }

    /// Apply the decomposed Conv3D to a temporal-pair input.
    ///
    /// `input` must have shape `(B, in_channels, 2, H, W)` where
    /// `in_channels = in_channels_per_group * cfg.groups`. Returns
    /// `(B, out_channels, 1, H_out, W_out)` where
    /// `H_out = (H + 2*padding - kernel_h) / stride + 1` (and
    /// likewise for W).
    pub fn apply(&self, input: &LazyTensor) -> Result<LazyTensor> {
        let dims = input.shape();
        let dims = dims.dims();
        if dims.len() != 5 {
            return Err(crate::Error::Msg(format!(
                "Conv3dTemporal2Weights::apply: input must be rank 5 (B, Cin, T, H, W), \
                 got rank {}",
                dims.len(),
            )));
        }
        let in_channels = self.in_channels_per_group * self.cfg.groups;
        if dims[1] != in_channels {
            return Err(crate::Error::Msg(format!(
                "Conv3dTemporal2Weights::apply: input has Cin={}, expected {in_channels} \
                 ({} per group * {} groups)",
                dims[1], self.in_channels_per_group, self.cfg.groups,
            )));
        }
        if dims[2] != 2 {
            return Err(crate::Error::Msg(format!(
                "Conv3dTemporal2Weights::apply: temporal-patch-2 requires T=2, got T={}",
                dims[2],
            )));
        }

        // Slice along temporal axis (dim=2) and squeeze.
        let xs1 = input.narrow(2_usize, 0, 1)?.squeeze(2_usize)?;
        let xs2 = input.narrow(2_usize, 1, 1)?.squeeze(2_usize)?;

        let w_shape = Shape::from_dims(&[
            self.out_channels,
            self.in_channels_per_group,
            self.kernel_h,
            self.kernel_w,
        ]);
        let w1 = input.const_f32_like(Arc::clone(&self.w1), w_shape.clone());
        let w2 = input.const_f32_like(Arc::clone(&self.w2), w_shape);

        let stride = (self.cfg.stride, self.cfg.stride);
        let padding = (self.cfg.padding, self.cfg.padding);
        let y1 = xs1.conv2d(&w1, None, stride, padding, self.cfg.groups)?;
        let y2 = xs2.conv2d(&w2, None, stride, padding, self.cfg.groups)?;
        let y = y1.add(&y2)?;
        // Re-insert temporal axis of size 1.
        y.unsqueeze(2_usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn raw_weight(out_c: usize, in_c_per_g: usize, kh: usize, kw: usize) -> Vec<f32> {
        // Deterministic small values keep all arithmetic in
        // normal-float range and let us hand-compute expected.
        (0..out_c * in_c_per_g * 2 * kh * kw)
            .map(|i| (i as f32) * 0.01 - 0.5)
            .collect()
    }

    #[test]
    fn from_raw_weight_splits_temporal_axis() {
        // (out=2, in=3, T=2, kH=1, kW=1) → 12 floats, w1/w2 each 6.
        let raw: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let w = Conv3dTemporal2Weights::from_raw_weight(
            &raw, 2, 3, 1, 1, Conv3dTemporal2Config::default(),
        ).unwrap();
        // outer iteration order: o in 0..(out * in/g) = 6 outer slabs of (2 * 1 * 1) = 2 floats each.
        // For outer o=0: w1 picks raw[0], w2 picks raw[1].
        // outer o=1: w1 picks raw[2], w2 picks raw[3]. … etc.
        let expected_w1 = vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
        let expected_w2 = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
        assert_eq!(&*w.w1, expected_w1.as_slice());
        assert_eq!(&*w.w2, expected_w2.as_slice());
    }

    #[test]
    fn from_raw_weight_rejects_dilation_other_than_1() {
        let raw = raw_weight(2, 3, 1, 1);
        let cfg = Conv3dTemporal2Config { dilation: 2, ..Default::default() };
        assert!(Conv3dTemporal2Weights::from_raw_weight(&raw, 2, 3, 1, 1, cfg).is_err());
    }

    #[test]
    fn from_raw_weight_rejects_size_mismatch() {
        let raw = raw_weight(2, 3, 1, 1);
        // Asking for out_channels=4 needs 24 floats, but raw has 12.
        let r = Conv3dTemporal2Weights::from_raw_weight(
            &raw, 4, 3, 1, 1, Conv3dTemporal2Config::default(),
        );
        assert!(r.is_err());
    }

    #[test]
    fn apply_kernel_1x1_matches_hand_computed() {
        // The simplest decomposition test: kernel (1, 1) so the
        // Conv2D is just per-pixel matmul along channels, no
        // spatial structure to worry about.
        // out_channels=1, in_channels=2, T=2, kH=kW=1.
        // Weight raw: 4 floats laid out (1, 2, 2, 1, 1) row-major:
        //   [w[0,0,0,0,0], w[0,0,1,0,0], w[0,1,0,0,0], w[0,1,1,0,0]]
        // So per-outer-slab (out=0, in=0): w1=[raw[0]], w2=[raw[1]];
        //              (out=0, in=1): w1=[raw[2]], w2=[raw[3]].
        let raw = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w = Conv3dTemporal2Weights::from_raw_weight(
            &raw, 1, 2, 1, 1, Conv3dTemporal2Config::default(),
        ).unwrap();
        // Input (B=1, C=2, T=2, H=1, W=1): just 4 scalars.
        //   x[0, 0, 0, 0, 0] = 10
        //   x[0, 0, 1, 0, 0] = 20
        //   x[0, 1, 0, 0, 0] = 30
        //   x[0, 1, 1, 0, 0] = 40
        let x_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let input = LazyTensor::from_f32(
            Arc::from(x_data),
            Shape::from_dims(&[1, 2, 2, 1, 1]),
            &Device::cpu(),
        );
        let y = w.apply(&input).unwrap();
        assert_eq!(y.shape().dims(), &[1, 1, 1, 1, 1]);
        // Expected:
        //   y[0,0,0,0,0] = conv2d_1(xs1) + conv2d_2(xs2)
        //     where xs1 = x[:,:,0] = [10, 30] (channels in dim 1, single pixel)
        //           xs2 = x[:,:,1] = [20, 40]
        //   conv2d_1 with w1 = [1.0 (cin=0), 3.0 (cin=1)]:
        //     y1[0,0,0,0] = 1.0*10 + 3.0*30 = 10 + 90 = 100
        //   conv2d_2 with w2 = [2.0, 4.0]:
        //     y2[0,0,0,0] = 2.0*20 + 4.0*40 = 40 + 160 = 200
        //   sum = 300
        let got = y.realize_f32();
        assert_eq!(got.len(), 1);
        assert!((got[0] - 300.0).abs() < 1e-5, "expected 300.0, got {}", got[0]);
    }

    #[test]
    fn apply_rejects_t_other_than_2() {
        let raw = vec![0.1_f32; 4];
        let w = Conv3dTemporal2Weights::from_raw_weight(
            &raw, 1, 2, 1, 1, Conv3dTemporal2Config::default(),
        ).unwrap();
        // Input shape (1, 2, 1, 1, 1) → 2 elements; T=1 should error.
        let input = LazyTensor::from_f32(
            Arc::from(vec![1.0_f32; 2]),
            Shape::from_dims(&[1, 2, 1, 1, 1]),
            &Device::cpu(),
        );
        assert!(w.apply(&input).is_err());
    }

    #[test]
    fn apply_strided_qwen3_vl_shape_smoke() {
        // Qwen3-VL canonical patch: kernel=(2, 14, 14), stride=(2, 14, 14).
        // Tiny version: out_c=4, in_c=3, kH=kW=2, stride=2. Input H=W=4 →
        // H_out = W_out = (4 - 2)/2 + 1 = 2.
        let raw = raw_weight(4, 3, 2, 2);
        let cfg = Conv3dTemporal2Config { stride: 2, ..Default::default() };
        let w = Conv3dTemporal2Weights::from_raw_weight(&raw, 4, 3, 2, 2, cfg).unwrap();

        let x_len = 1 * 3 * 2 * 4 * 4;
        let x_data: Vec<f32> = (0..x_len).map(|i| (i as f32) * 0.01).collect();
        let input = LazyTensor::from_f32(
            Arc::from(x_data),
            Shape::from_dims(&[1, 3, 2, 4, 4]),
            &Device::cpu(),
        );

        let y = w.apply(&input).unwrap();
        assert_eq!(y.shape().dims(), &[1, 4, 1, 2, 2]);
        for &v in &y.realize_f32() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}

//! Lazy `Conv1d` / `Conv2d` Module wrappers over `LazyTensor`.
//!
//! Mirrors the eager `fuel-nn::{Conv1d, Conv2d}` surface: each layer
//! holds a [`WeightStorage`] weight plus an optional bias and a config
//! struct controlling padding / stride / dilation / groups. `forward`
//! materializes the weight (and bias) as graph constants on the
//! activation's graph and delegates to [`LazyTensor::conv1d`] /
//! [`LazyTensor::conv2d`].
//!
//! Dilation: the LazyTensor conv primitives do not yet accept a
//! `dilation` argument, so configs that request dilation other than
//! `1` (or `(1, 1)`) are rejected at `forward` time rather than
//! silently dropped. This matches the "no deferrals — surface the
//! gap" convention used elsewhere in the lazy port.

use crate::Result;
use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_nn::{LazyBatchNorm2d, LazyModule};
use fuel_core_types::Shape;
use std::sync::Arc;

/// Configuration for [`LazyConv1d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LazyConv1dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for LazyConv1dConfig {
    fn default() -> Self {
        Self { padding: 0, stride: 1, dilation: 1, groups: 1 }
    }
}

impl LazyConv1dConfig {
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }
}

/// Configuration for [`LazyConv2d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LazyConv2dConfig {
    pub padding: (usize, usize),
    pub stride: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl Default for LazyConv2dConfig {
    fn default() -> Self {
        Self {
            padding: (0, 0),
            stride: (1, 1),
            dilation: (1, 1),
            groups: 1,
        }
    }
}

impl LazyConv2dConfig {
    pub fn with_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }
    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }
    pub fn with_dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }
}

/// 1-D convolution layer over `LazyTensor`.
#[derive(Debug, Clone)]
pub struct LazyConv1d {
    weight: WeightStorage,
    bias: Option<Arc<[f32]>>,
    cfg: LazyConv1dConfig,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
}

impl LazyConv1d {
    /// Build a 1-D convolution from a weight storage and optional bias.
    ///
    /// `weight` must have `out_channels * (in_channels / groups) * kernel_size`
    /// elements, in the conv-canonical `[Cout, Cin/groups, K]` layout.
    /// `bias`, if present, must have length `out_channels`.
    pub fn new(
        weight: WeightStorage,
        bias: Option<Arc<[f32]>>,
        cfg: LazyConv1dConfig,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        if cfg.groups < 1 {
            return Err(crate::Error::Msg(format!(
                "LazyConv1d::new: groups must be >= 1, got {}", cfg.groups,
            )).bt());
        }
        if in_channels % cfg.groups != 0 {
            return Err(crate::Error::Msg(format!(
                "LazyConv1d::new: in_channels ({}) must be divisible \
                 by groups ({})",
                in_channels, cfg.groups,
            )).bt());
        }
        if out_channels % cfg.groups != 0 {
            return Err(crate::Error::Msg(format!(
                "LazyConv1d::new: out_channels ({}) must be divisible \
                 by groups ({})",
                out_channels, cfg.groups,
            )).bt());
        }
        let expected = out_channels * (in_channels / cfg.groups) * kernel_size;
        if weight.elem_count() != expected {
            return Err(crate::Error::Msg(format!(
                "LazyConv1d::new: weight has {} elements but \
                 out_channels * (in_channels / groups) * kernel_size = \
                 {} * {} * {} = {}",
                weight.elem_count(),
                out_channels,
                in_channels / cfg.groups,
                kernel_size,
                expected,
            )).bt());
        }
        if let Some(b) = bias.as_ref() {
            if b.len() != out_channels {
                return Err(crate::Error::Msg(format!(
                    "LazyConv1d::new: bias has length {} but \
                     out_channels = {}",
                    b.len(), out_channels,
                )).bt());
            }
        }
        Ok(Self {
            weight, bias, cfg, in_channels, out_channels, kernel_size,
        })
    }

    pub fn cfg(&self) -> &LazyConv1dConfig { &self.cfg }
    pub fn weight(&self) -> &WeightStorage { &self.weight }
    pub fn bias(&self) -> Option<&Arc<[f32]>> { self.bias.as_ref() }
    pub fn in_channels(&self) -> usize { self.in_channels }
    pub fn out_channels(&self) -> usize { self.out_channels }
    pub fn kernel_size(&self) -> usize { self.kernel_size }
}

impl LazyModule for LazyConv1d {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        if self.cfg.dilation != 1 {
            return Err(crate::Error::Msg(format!(
                "LazyConv1d::forward: dilation = {} is not supported; \
                 LazyTensor::conv1d only takes stride/padding/groups. \
                 Use dilation == 1.",
                self.cfg.dilation,
            )).bt());
        }
        let w_shape = Shape::from_dims(&[
            self.out_channels,
            self.in_channels / self.cfg.groups,
            self.kernel_size,
        ]);
        let w_t = self.weight.const_like(xs, w_shape)?;
        let bias_t = self.bias.as_ref().map(|b| {
            xs.const_f32_like(
                Arc::clone(b), Shape::from_dims(&[self.out_channels]),
            )
        });
        xs.conv1d(
            &w_t,
            bias_t.as_ref(),
            self.cfg.stride,
            self.cfg.padding,
            self.cfg.groups,
        )
    }
}

/// 2-D convolution layer over `LazyTensor`.
#[derive(Debug, Clone)]
pub struct LazyConv2d {
    weight: WeightStorage,
    bias: Option<Arc<[f32]>>,
    cfg: LazyConv2dConfig,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
}

impl LazyConv2d {
    /// Build a 2-D convolution from a weight storage and optional bias.
    ///
    /// `weight` must have `out_channels * (in_channels / groups) * kernel_h
    /// * kernel_w` elements, in the conv-canonical
    /// `[Cout, Cin/groups, Kh, Kw]` layout. `bias`, if present, must have
    /// length `out_channels`.
    pub fn new(
        weight: WeightStorage,
        bias: Option<Arc<[f32]>>,
        cfg: LazyConv2dConfig,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
    ) -> Result<Self> {
        if cfg.groups < 1 {
            return Err(crate::Error::Msg(format!(
                "LazyConv2d::new: groups must be >= 1, got {}", cfg.groups,
            )).bt());
        }
        if in_channels % cfg.groups != 0 {
            return Err(crate::Error::Msg(format!(
                "LazyConv2d::new: in_channels ({}) must be divisible \
                 by groups ({})",
                in_channels, cfg.groups,
            )).bt());
        }
        if out_channels % cfg.groups != 0 {
            return Err(crate::Error::Msg(format!(
                "LazyConv2d::new: out_channels ({}) must be divisible \
                 by groups ({})",
                out_channels, cfg.groups,
            )).bt());
        }
        let expected = out_channels
            * (in_channels / cfg.groups)
            * kernel_h
            * kernel_w;
        if weight.elem_count() != expected {
            return Err(crate::Error::Msg(format!(
                "LazyConv2d::new: weight has {} elements but \
                 out_channels * (in_channels / groups) * kernel_h * kernel_w \
                 = {} * {} * {} * {} = {}",
                weight.elem_count(),
                out_channels,
                in_channels / cfg.groups,
                kernel_h, kernel_w,
                expected,
            )).bt());
        }
        if let Some(b) = bias.as_ref() {
            if b.len() != out_channels {
                return Err(crate::Error::Msg(format!(
                    "LazyConv2d::new: bias has length {} but \
                     out_channels = {}",
                    b.len(), out_channels,
                )).bt());
            }
        }
        Ok(Self {
            weight, bias, cfg, in_channels, out_channels, kernel_h, kernel_w,
        })
    }

    pub fn cfg(&self) -> &LazyConv2dConfig { &self.cfg }
    pub fn weight(&self) -> &WeightStorage { &self.weight }
    pub fn bias(&self) -> Option<&Arc<[f32]>> { self.bias.as_ref() }
    pub fn in_channels(&self) -> usize { self.in_channels }
    pub fn out_channels(&self) -> usize { self.out_channels }
    pub fn kernel_h(&self) -> usize { self.kernel_h }
    pub fn kernel_w(&self) -> usize { self.kernel_w }

    /// Fold a following [`LazyBatchNorm2d`] into this conv's weight and
    /// bias, returning a new [`LazyConv2d`] that produces the same
    /// activations as `bn(self(x))` in BN eval mode.
    ///
    /// Math (mirrors `CbsWeights::fuse_bn` in `lazy_yolov8`):
    ///   `scale[c] = γ[c] / sqrt(running_var[c] + eps)`
    ///   `new_weight[c, ...] = old_weight[c, ...] · scale[c]`
    ///   `new_bias[c]   = (old_bias[c] - running_mean[c]) · scale[c] + β[c]`
    /// where `(γ, β, running_mean, running_var, eps)` come from `bn`,
    /// and `old_bias[c]` defaults to 0 when this conv has no bias.
    ///
    /// `bn.num_features()` must equal this conv's `out_channels`. The
    /// returned conv reuses this conv's config (padding / stride /
    /// dilation / groups) verbatim; the weight dtype is preserved
    /// (F32 stays F32, BF16 stays BF16 via a host-side round-trip).
    ///
    /// Errors:
    /// - if `bn.num_features() != self.out_channels`
    /// - if the weight storage is [`WeightStorage::Q4_0`] or
    ///   [`WeightStorage::WithLoRA`] — BN absorption requires a host
    ///   f32 view, which neither variant exposes losslessly.
    pub fn absorb_bn(&self, bn: &LazyBatchNorm2d) -> Result<LazyConv2d> {
        if bn.num_features() != self.out_channels {
            return Err(crate::Error::Msg(format!(
                "LazyConv2d::absorb_bn: bn.num_features ({}) must equal \
                 conv.out_channels ({})",
                bn.num_features(), self.out_channels,
            )).bt());
        }

        // Read conv weight as host f32. F32 is direct; BF16 round-trips
        // through f32 for the math, then back to bf16 for storage so
        // the fused module keeps its native dtype. Q4_0/WithLoRA can't
        // be folded losslessly — surface that rather than silently
        // dequantizing.
        let c_out = self.out_channels;
        let per_out = (self.in_channels / self.cfg.groups) * self.kernel_h * self.kernel_w;
        let weight_f32: Vec<f32> = match &self.weight {
            WeightStorage::F32(a) => a.iter().copied().collect(),
            WeightStorage::BF16(a) => a.iter().map(|v| v.to_f32()).collect(),
            WeightStorage::Q4_0 { .. } => {
                return Err(crate::Error::Msg(
                    "LazyConv2d::absorb_bn: Q4_0 weights cannot be folded \
                     with a following BatchNorm (no lossless host f32 \
                     view). Dequantize first, fold, then requantize if \
                     desired.".into(),
                ).bt());
            }
            WeightStorage::WithLoRA { .. } => {
                return Err(crate::Error::Msg(
                    "LazyConv2d::absorb_bn: LoRA-wrapped weights cannot be \
                     folded with a following BatchNorm (the adapter must \
                     be applied to activations, not weights). Merge LoRA \
                     into the base first, then fold.".into(),
                ).bt());
            }
        };
        debug_assert_eq!(weight_f32.len(), c_out * per_out);

        let gamma = bn.weight();
        let beta = bn.bias();
        let mean = bn.running_mean();
        let var = bn.running_var();
        let eps32 = bn.eps() as f32;

        // Per-channel scale and the fused bias term.
        let mut new_weight = vec![0.0_f32; c_out * per_out];
        let mut new_bias = vec![0.0_f32; c_out];
        for c in 0..c_out {
            let scale = gamma[c] / (var[c] + eps32).sqrt();
            let base = c * per_out;
            for j in 0..per_out {
                new_weight[base + j] = weight_f32[base + j] * scale;
            }
            let old_b = self.bias.as_ref().map(|b| b[c]).unwrap_or(0.0_f32);
            new_bias[c] = (old_b - mean[c]) * scale + beta[c];
        }

        // Preserve the original weight dtype.
        let folded_weight = match &self.weight {
            WeightStorage::F32(_) => WeightStorage::F32(Arc::from(new_weight)),
            WeightStorage::BF16(_) => {
                let as_bf16: Vec<half::bf16> =
                    new_weight.into_iter().map(half::bf16::from_f32).collect();
                WeightStorage::BF16(Arc::from(as_bf16))
            }
            // The Q4_0 / WithLoRA arms returned early above.
            WeightStorage::Q4_0 { .. } | WeightStorage::WithLoRA { .. } => unreachable!(),
        };

        LazyConv2d::new(
            folded_weight,
            Some(Arc::from(new_bias)),
            self.cfg,
            self.in_channels,
            self.out_channels,
            self.kernel_h,
            self.kernel_w,
        )
    }
}

impl LazyModule for LazyConv2d {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        if self.cfg.dilation != (1, 1) {
            return Err(crate::Error::Msg(format!(
                "LazyConv2d::forward: dilation = {:?} is not supported; \
                 LazyTensor::conv2d only takes stride/padding/groups. \
                 Use dilation == (1, 1).",
                self.cfg.dilation,
            )).bt());
        }
        let w_shape = Shape::from_dims(&[
            self.out_channels,
            self.in_channels / self.cfg.groups,
            self.kernel_h,
            self.kernel_w,
        ]);
        let w_t = self.weight.const_like(xs, w_shape)?;
        let bias_t = self.bias.as_ref().map(|b| {
            xs.const_f32_like(
                Arc::clone(b), Shape::from_dims(&[self.out_channels]),
            )
        });
        xs.conv2d(
            &w_t,
            bias_t.as_ref(),
            self.cfg.stride,
            self.cfg.padding,
            self.cfg.groups,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn ramp_f32(n: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * scale + offset).collect()
    }

    #[test]
    fn conv1d_forward_shape_for_strided_input() {
        let n = 2;
        let cin = 3;
        let cout = 4;
        let l = 9;
        let k = 3;
        let cfg = LazyConv1dConfig { padding: 1, stride: 2, dilation: 1, groups: 1 };

        let w: Vec<f32> = ramp_f32(cout * cin * k, 0.05, -0.2);
        let bias: Vec<f32> = ramp_f32(cout, 0.1, 0.0);
        let layer = LazyConv1d::new(
            WeightStorage::F32(Arc::from(w)),
            Some(Arc::from(bias)),
            cfg, cin, cout, k,
        ).unwrap();

        let x_data: Vec<f32> = ramp_f32(n * cin * l, 0.03, -0.4);
        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[n, cin, l]), &Device::cpu(),
        );
        let y = layer.forward(&x).unwrap();
        let l_out = (l + 2 * cfg.padding - k) / cfg.stride + 1;
        assert_eq!(y.shape().dims(), &[n, cout, l_out]);
        let got = y.realize_f32();
        assert_eq!(got.len(), n * cout * l_out);
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "conv1d out[{i}] = {v} not finite");
        }
    }

    #[test]
    fn conv1d_no_bias_matches_lazy_tensor_conv1d_directly() {
        let n = 1;
        let cin = 2;
        let cout = 3;
        let l = 7;
        let k = 3;
        let cfg = LazyConv1dConfig { padding: 1, stride: 1, dilation: 1, groups: 1 };

        let w: Vec<f32> = ramp_f32(cout * cin * k, 0.04, 0.1);
        let x_data: Vec<f32> = ramp_f32(n * cin * l, 0.02, -0.3);

        let weight_arc: Arc<[f32]> = Arc::from(w.clone());
        let layer = LazyConv1d::new(
            WeightStorage::F32(Arc::clone(&weight_arc)),
            None,
            cfg, cin, cout, k,
        ).unwrap();
        let x = LazyTensor::from_f32(
            x_data.clone(),
            Shape::from_dims(&[n, cin, l]),
            &Device::cpu(),
        );
        let via_module = layer.forward(&x).unwrap().realize_f32();

        let x2 = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[n, cin, l]), &Device::cpu(),
        );
        let w_t = x2.const_f32_like(
            Arc::clone(&weight_arc),
            Shape::from_dims(&[cout, cin, k]),
        );
        let direct = x2.conv1d(&w_t, None, cfg.stride, cfg.padding, cfg.groups)
            .unwrap()
            .realize_f32();

        assert_eq!(via_module.len(), direct.len());
        for (i, (a, d)) in via_module.iter().zip(direct.iter()).enumerate() {
            assert!(
                (a - d).abs() < 1e-5,
                "conv1d_no_bias[{i}] module {a} != direct {d}",
            );
        }
    }

    #[test]
    fn conv2d_forward_shape_for_strided_input() {
        let n = 2;
        let cin = 3;
        let cout = 5;
        let h = 8;
        let w_in = 8;
        let kh = 3;
        let kw = 3;
        let cfg = LazyConv2dConfig {
            padding: (1, 1),
            stride: (2, 2),
            dilation: (1, 1),
            groups: 1,
        };

        let weight: Vec<f32> = ramp_f32(cout * cin * kh * kw, 0.02, -0.1);
        let bias: Vec<f32> = ramp_f32(cout, 0.05, 0.2);
        let layer = LazyConv2d::new(
            WeightStorage::F32(Arc::from(weight)),
            Some(Arc::from(bias)),
            cfg, cin, cout, kh, kw,
        ).unwrap();

        let x_data: Vec<f32> = ramp_f32(n * cin * h * w_in, 0.01, -0.5);
        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[n, cin, h, w_in]), &Device::cpu(),
        );
        let y = layer.forward(&x).unwrap();
        let h_out = (h + 2 * cfg.padding.0 - kh) / cfg.stride.0 + 1;
        let w_out = (w_in + 2 * cfg.padding.1 - kw) / cfg.stride.1 + 1;
        assert_eq!(y.shape().dims(), &[n, cout, h_out, w_out]);
        let got = y.realize_f32();
        assert_eq!(got.len(), n * cout * h_out * w_out);
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "conv2d out[{i}] = {v} not finite");
        }
    }

    #[test]
    fn conv2d_with_bias_matches_lazy_tensor_conv2d_plus_broadcast() {
        let n = 1;
        let cin = 2;
        let cout = 3;
        let h = 5;
        let w_in = 5;
        let kh = 3;
        let kw = 3;
        let cfg = LazyConv2dConfig {
            padding: (1, 1),
            stride: (1, 1),
            dilation: (1, 1),
            groups: 1,
        };

        let weight: Vec<f32> = ramp_f32(cout * cin * kh * kw, 0.03, 0.0);
        let bias: Vec<f32> = ramp_f32(cout, 0.5, -0.2);
        let x_data: Vec<f32> = ramp_f32(n * cin * h * w_in, 0.02, -0.4);

        let weight_arc: Arc<[f32]> = Arc::from(weight.clone());
        let bias_arc: Arc<[f32]> = Arc::from(bias.clone());

        let layer = LazyConv2d::new(
            WeightStorage::F32(Arc::clone(&weight_arc)),
            Some(Arc::clone(&bias_arc)),
            cfg, cin, cout, kh, kw,
        ).unwrap();
        let x = LazyTensor::from_f32(
            x_data.clone(),
            Shape::from_dims(&[n, cin, h, w_in]),
            &Device::cpu(),
        );
        let via_module = layer.forward(&x).unwrap().realize_f32();

        let x2 = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[n, cin, h, w_in]), &Device::cpu(),
        );
        let w_t = x2.const_f32_like(
            Arc::clone(&weight_arc),
            Shape::from_dims(&[cout, cin, kh, kw]),
        );
        let b_t = x2.const_f32_like(
            Arc::clone(&bias_arc), Shape::from_dims(&[cout]),
        );
        let direct = x2.conv2d(
            &w_t, Some(&b_t), cfg.stride, cfg.padding, cfg.groups,
        ).unwrap().realize_f32();

        assert_eq!(via_module.len(), direct.len());
        for (i, (a, d)) in via_module.iter().zip(direct.iter()).enumerate() {
            assert!(
                (a - d).abs() < 1e-5,
                "conv2d_with_bias[{i}] module {a} != direct {d}",
            );
        }
    }

    #[test]
    fn conv2d_absorb_bn_matches_conv_then_bn() {
        // (a) conv2d → batch_norm on a small fixture must equal a
        //     single conv2d whose weights have absorbed the BN.
        let n = 2;
        let cin = 3;
        let cout = 4;
        let h = 5;
        let w_in = 5;
        let kh = 3;
        let kw = 3;
        let cfg = LazyConv2dConfig {
            padding: (1, 1),
            stride: (1, 1),
            dilation: (1, 1),
            groups: 1,
        };

        let weight: Vec<f32> = ramp_f32(cout * cin * kh * kw, 0.02, -0.15);
        // BN affine + running stats — pick distinct ramps per buffer so
        // the math actually exercises each term.
        let gamma: Vec<f32> = ramp_f32(cout, 0.1, 0.5);
        let beta: Vec<f32> = ramp_f32(cout, 0.07, -0.3);
        let r_mean: Vec<f32> = ramp_f32(cout, 0.05, 0.2);
        // running_var must stay strictly positive.
        let r_var: Vec<f32> = (0..cout).map(|i| 0.3 + (i as f32) * 0.1).collect();
        let eps = 1e-5_f64;

        // No-bias conv to keep the no-bias absorb path on test (a).
        let conv = LazyConv2d::new(
            WeightStorage::F32(Arc::from(weight)),
            None,
            cfg, cin, cout, kh, kw,
        ).unwrap();
        let bn = LazyBatchNorm2d::new(
            Arc::from(gamma),
            Arc::from(beta),
            Arc::from(r_mean),
            Arc::from(r_var),
            eps,
            cout,
        ).unwrap();

        let x_data: Vec<f32> = ramp_f32(n * cin * h * w_in, 0.013, -0.4);

        // Path 1: conv → bn.
        let x1 = LazyTensor::from_f32(
            x_data.clone(),
            Shape::from_dims(&[n, cin, h, w_in]), &Device::cpu(),
        );
        let y1 = bn.forward(&conv.forward(&x1).unwrap()).unwrap().realize_f32();

        // Path 2: absorb_bn → single conv.
        let fused = conv.absorb_bn(&bn).unwrap();
        let x2 = LazyTensor::from_f32(
            x_data,
            Shape::from_dims(&[n, cin, h, w_in]), &Device::cpu(),
        );
        let y2 = fused.forward(&x2).unwrap().realize_f32();

        assert_eq!(y1.len(), y2.len());
        for (i, (a, b)) in y1.iter().zip(y2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "absorb_bn[{i}] conv→bn = {a}, fused = {b}",
            );
        }

        // Sanity: fused conv must now carry a bias even though the
        // original conv had none (BN's β alone produces a non-zero
        // per-channel offset).
        assert!(fused.bias().is_some());
    }

    #[test]
    fn conv2d_absorb_bn_merges_existing_bias() {
        // (b) absorb_bn on a conv that already has a bias correctly
        //     merges the existing bias into the new bias.
        //     With c_out = 1 the math collapses to scalars we can
        //     check by hand:
        //       scale = γ / sqrt(var + eps)
        //       new_bias = (old_bias - mean) · scale + β
        //       new_weight = old_weight · scale
        let cin = 2;
        let cout = 1;
        let kh = 2;
        let kw = 2;
        let cfg = LazyConv2dConfig::default();

        let weight: Vec<f32> = vec![0.5, -0.25, 0.75, 0.1, -0.3, 0.4, 0.2, -0.6];
        debug_assert_eq!(weight.len(), cout * cin * kh * kw);
        let old_bias_val = 0.7_f32;
        let conv = LazyConv2d::new(
            WeightStorage::F32(Arc::from(weight.clone())),
            Some(Arc::from(vec![old_bias_val])),
            cfg, cin, cout, kh, kw,
        ).unwrap();

        let gamma = 2.0_f32;
        let beta = -0.5_f32;
        let mean = 0.25_f32;
        let var = 0.75_f32;
        let eps = 1e-4_f64;

        let bn = LazyBatchNorm2d::new(
            Arc::from(vec![gamma]),
            Arc::from(vec![beta]),
            Arc::from(vec![mean]),
            Arc::from(vec![var]),
            eps,
            cout,
        ).unwrap();

        let fused = conv.absorb_bn(&bn).unwrap();

        let scale = gamma / (var + eps as f32).sqrt();
        let expected_bias = (old_bias_val - mean) * scale + beta;

        let folded_bias = fused.bias().expect("absorb_bn must produce a bias").clone();
        assert_eq!(folded_bias.len(), 1);
        assert!(
            (folded_bias[0] - expected_bias).abs() < 1e-6,
            "merged bias mismatch: got {}, expected {}",
            folded_bias[0], expected_bias,
        );

        // Weight: each element multiplied by scale.
        let folded_w = match fused.weight() {
            WeightStorage::F32(a) => a.clone(),
            _ => panic!("expected F32 weight after absorb_bn on F32 conv"),
        };
        assert_eq!(folded_w.len(), weight.len());
        for (i, (orig, got)) in weight.iter().zip(folded_w.iter()).enumerate() {
            let expected = *orig * scale;
            assert!(
                (got - expected).abs() < 1e-6,
                "folded_weight[{i}] = {got}, expected {expected}",
            );
        }

        // End-to-end: conv→bn must equal fused conv on a small input,
        // even with the existing bias on the original conv.
        let n = 1;
        let h = 3;
        let w_in = 3;
        let x_data: Vec<f32> = ramp_f32(n * cin * h * w_in, 0.1, -0.2);
        let x1 = LazyTensor::from_f32(
            x_data.clone(),
            Shape::from_dims(&[n, cin, h, w_in]), &Device::cpu(),
        );
        let y1 = bn.forward(&conv.forward(&x1).unwrap()).unwrap().realize_f32();
        let x2 = LazyTensor::from_f32(
            x_data,
            Shape::from_dims(&[n, cin, h, w_in]), &Device::cpu(),
        );
        let y2 = fused.forward(&x2).unwrap().realize_f32();
        assert_eq!(y1.len(), y2.len());
        for (i, (a, b)) in y1.iter().zip(y2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "absorb_bn_with_bias[{i}] conv→bn = {a}, fused = {b}",
            );
        }
    }

    #[test]
    fn conv2d_absorb_bn_rejects_channel_mismatch() {
        let cin = 2;
        let cout = 3;
        let kh = 3;
        let kw = 3;
        let weight = ramp_f32(cout * cin * kh * kw, 0.01, 0.0);
        let conv = LazyConv2d::new(
            WeightStorage::F32(Arc::from(weight)),
            None,
            LazyConv2dConfig::default(),
            cin, cout, kh, kw,
        ).unwrap();
        let wrong_features = cout + 1;
        let bn = LazyBatchNorm2d::new(
            Arc::from(vec![1.0_f32; wrong_features]),
            Arc::from(vec![0.0_f32; wrong_features]),
            Arc::from(vec![0.0_f32; wrong_features]),
            Arc::from(vec![1.0_f32; wrong_features]),
            1e-5,
            wrong_features,
        ).unwrap();
        let err = conv.absorb_bn(&bn).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("absorb_bn") && msg.contains("num_features"),
            "expected channel-mismatch message, got: {msg}",
        );
    }

    #[test]
    fn conv2d_depthwise_groups_equals_in_channels() {
        let n = 1;
        let c = 4;
        let h = 5;
        let w_in = 5;
        let kh = 3;
        let kw = 3;
        let cfg = LazyConv2dConfig {
            padding: (1, 1),
            stride: (1, 1),
            dilation: (1, 1),
            groups: c,
        };

        let weight: Vec<f32> = ramp_f32(c * 1 * kh * kw, 0.07, -0.1);
        let layer = LazyConv2d::new(
            WeightStorage::F32(Arc::from(weight)),
            None,
            cfg, c, c, kh, kw,
        ).unwrap();
        let x_data: Vec<f32> = ramp_f32(n * c * h * w_in, 0.02, 0.3);
        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[n, c, h, w_in]), &Device::cpu(),
        );
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[n, c, h, w_in]);
        let got = y.realize_f32();
        assert_eq!(got.len(), n * c * h * w_in);
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "depthwise conv2d out[{i}] = {v} not finite");
        }
    }
}

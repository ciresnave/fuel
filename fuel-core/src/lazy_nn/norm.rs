//! Lazy normalization Module wrappers — `LayerNorm`, `RmsNorm`,
//! `GroupNorm`, `BatchNorm2d`.
//!
//! Each wrapper holds its affine parameters (plus running statistics for
//! BatchNorm) as `Arc<[f32]>` buffers and lowers `forward` to existing
//! `LazyTensor` primitives — `layer_norm_affine`, `rms_norm_affine`,
//! `layer_norm_last_dim`, and `channel_affine_4d`. No new graph ops.
//!
//! `LazyBatchNorm2d` is inference-only: it absorbs `running_mean`,
//! `running_var`, and `eps` into a fused per-channel `gain · x + bias`,
//! matching eager `BatchNorm::forward_eval`'s `scale = γ / sqrt(var+ε)`,
//! `offset = β − μ · scale` pre-computation. Training-mode batch
//! statistics tracking is out of scope for the lazy-graph inference
//! path.

use crate::Result;
use crate::lazy::LazyTensor;
use crate::lazy_nn::LazyModule;
use fuel_ir::Shape;
use std::sync::Arc;

/// LayerNorm over the last dim with optional bias.
///
/// `y = ((x − mean) / sqrt(var + eps)) · gain (+ bias)`. Both `gain`
/// and `bias`, when present, must have length `last_dim`. The
/// normalization expects `xs.shape().dims().last() == Some(last_dim)`.
#[derive(Debug, Clone)]
pub struct LazyLayerNorm {
    gain: Arc<[f32]>,
    bias: Option<Arc<[f32]>>,
    eps: f64,
    last_dim: usize,
}

impl LazyLayerNorm {
    /// Build a LayerNorm wrapper. `gain.len()` and (when present)
    /// `bias.len()` must equal `last_dim`.
    pub fn new(
        gain: Arc<[f32]>,
        bias: Option<Arc<[f32]>>,
        eps: f64,
        last_dim: usize,
    ) -> Result<Self> {
        if gain.len() != last_dim {
            return Err(crate::Error::Msg(format!(
                "LazyLayerNorm::new: gain has length {} but last_dim = {}",
                gain.len(), last_dim,
            )).bt());
        }
        if let Some(b) = bias.as_ref() {
            if b.len() != last_dim {
                return Err(crate::Error::Msg(format!(
                    "LazyLayerNorm::new: bias has length {} but last_dim = {}",
                    b.len(), last_dim,
                )).bt());
            }
        }
        Ok(Self { gain, bias, eps, last_dim })
    }

    /// Reference to the gain (scale) buffer.
    pub fn gain(&self) -> &Arc<[f32]> { &self.gain }

    /// Reference to the bias buffer, if present.
    pub fn bias(&self) -> Option<&Arc<[f32]>> { self.bias.as_ref() }

    /// Epsilon for numerical stability.
    pub fn eps(&self) -> f64 { self.eps }

    /// Normalized last-dim size.
    pub fn last_dim(&self) -> usize { self.last_dim }
}

impl LazyModule for LazyLayerNorm {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        match &self.bias {
            Some(b) => xs.layer_norm_affine(
                Arc::clone(&self.gain), Arc::clone(b), self.eps,
            ),
            None => {
                let normed = xs.layer_norm_last_dim(self.eps)?;
                let g = normed.const_f32_like(
                    Arc::clone(&self.gain),
                    Shape::from_dims(&[self.last_dim]),
                );
                normed.broadcast_mul(&g)
            }
        }
    }
}

/// RmsNorm over the last dim: `y = (x / sqrt(mean(x²) + eps)) · gain`.
///
/// No bias term — RmsNorm has no β by construction. `gain.len()` must
/// equal `last_dim`.
#[derive(Debug, Clone)]
pub struct LazyRmsNorm {
    gain: Arc<[f32]>,
    eps: f64,
    last_dim: usize,
}

impl LazyRmsNorm {
    /// Build an RmsNorm wrapper. `gain.len()` must equal `last_dim`.
    pub fn new(gain: Arc<[f32]>, eps: f64, last_dim: usize) -> Result<Self> {
        if gain.len() != last_dim {
            return Err(crate::Error::Msg(format!(
                "LazyRmsNorm::new: gain has length {} but last_dim = {}",
                gain.len(), last_dim,
            )).bt());
        }
        Ok(Self { gain, eps, last_dim })
    }

    /// Reference to the gain buffer.
    pub fn gain(&self) -> &Arc<[f32]> { &self.gain }

    /// Epsilon for numerical stability.
    pub fn eps(&self) -> f64 { self.eps }

    /// Normalized last-dim size.
    pub fn last_dim(&self) -> usize { self.last_dim }
}

impl LazyModule for LazyRmsNorm {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        xs.rms_norm_affine(Arc::clone(&self.gain), self.eps)
    }
}

/// GroupNorm with learnable per-channel affine. Input shape is
/// `(B, C, ...)` with `C == num_channels` and `num_channels % num_groups == 0`.
/// Normalization is computed over each group's `(C/G) · spatial`
/// elements.
///
/// Lowering: reshape `(B, C, S)` → `(B, G, (C/G)·S)`, normalize the last
/// dim with `layer_norm_last_dim`, reshape back to `(B, C, S)`, and
/// apply the per-channel affine via a broadcast `gain · x + bias` over
/// dim 1.
#[derive(Debug, Clone)]
pub struct LazyGroupNorm {
    gain: Arc<[f32]>,
    bias: Arc<[f32]>,
    num_groups: usize,
    num_channels: usize,
    eps: f64,
}

impl LazyGroupNorm {
    /// Build a GroupNorm wrapper. `gain` and `bias` must have length
    /// `num_channels`, and `num_channels` must be divisible by
    /// `num_groups`.
    pub fn new(
        gain: Arc<[f32]>,
        bias: Arc<[f32]>,
        num_groups: usize,
        num_channels: usize,
        eps: f64,
    ) -> Result<Self> {
        if num_groups == 0 {
            return Err(crate::Error::Msg(
                "LazyGroupNorm::new: num_groups must be ≥ 1".into(),
            ).bt());
        }
        if num_channels % num_groups != 0 {
            return Err(crate::Error::Msg(format!(
                "LazyGroupNorm::new: num_groups ({num_groups}) must divide \
                 num_channels ({num_channels})",
            )).bt());
        }
        if gain.len() != num_channels {
            return Err(crate::Error::Msg(format!(
                "LazyGroupNorm::new: gain has length {} but num_channels = {}",
                gain.len(), num_channels,
            )).bt());
        }
        if bias.len() != num_channels {
            return Err(crate::Error::Msg(format!(
                "LazyGroupNorm::new: bias has length {} but num_channels = {}",
                bias.len(), num_channels,
            )).bt());
        }
        Ok(Self { gain, bias, num_groups, num_channels, eps })
    }

    /// Reference to the gain buffer.
    pub fn gain(&self) -> &Arc<[f32]> { &self.gain }

    /// Reference to the bias buffer.
    pub fn bias(&self) -> &Arc<[f32]> { &self.bias }

    /// Number of channel groups.
    pub fn num_groups(&self) -> usize { self.num_groups }

    /// Total channel count (`gain.len() == bias.len()`).
    pub fn num_channels(&self) -> usize { self.num_channels }

    /// Epsilon for numerical stability.
    pub fn eps(&self) -> f64 { self.eps }
}

impl LazyModule for LazyGroupNorm {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let in_dims: Vec<usize> = xs.shape().dims().to_vec();
        if in_dims.len() < 3 {
            return Err(crate::Error::Msg(format!(
                "LazyGroupNorm::forward: input rank must be ≥ 3, got {in_dims:?}",
            )).bt());
        }
        let b_sz = in_dims[0];
        let c = in_dims[1];
        if c != self.num_channels {
            return Err(crate::Error::Msg(format!(
                "LazyGroupNorm::forward: input channel dim {c} != num_channels = {}",
                self.num_channels,
            )).bt());
        }
        let spatial: usize = in_dims[2..].iter().product();
        let channels_per_group = self.num_channels / self.num_groups;
        let hidden_per_group = channels_per_group * spatial;

        let grouped = xs.reshape(
            Shape::from_dims(&[b_sz, self.num_groups, hidden_per_group]),
        )?;
        let normed = grouped.layer_norm_last_dim(self.eps)?;
        let restored = normed.reshape(Shape::from_dims(&in_dims))?;

        let mut affine_shape = vec![1_usize; in_dims.len()];
        affine_shape[1] = self.num_channels;
        let g_t = restored
            .const_f32_like(
                Arc::clone(&self.gain), Shape::from_dims(&[self.num_channels]),
            )
            .reshape(Shape::from_dims(&affine_shape))?;
        let b_t = restored
            .const_f32_like(
                Arc::clone(&self.bias), Shape::from_dims(&[self.num_channels]),
            )
            .reshape(Shape::from_dims(&affine_shape))?;
        restored.broadcast_mul(&g_t)?.broadcast_add(&b_t)
    }
}

/// BatchNorm2d (eval mode): `y = ((x − running_mean) / sqrt(running_var + eps)) · weight + bias`,
/// applied per channel over a `(N, C, H, W)` input.
///
/// Lazy graphs serve inference only; running statistics are used
/// directly and absorbed into a single per-channel affine
/// `gain · x + bias`, matching eager `BatchNorm::forward_eval`'s fused
/// scale/offset pre-computation. All four buffers must have length
/// `num_features`.
#[derive(Debug, Clone)]
pub struct LazyBatchNorm2d {
    weight: Arc<[f32]>,
    bias: Arc<[f32]>,
    running_mean: Arc<[f32]>,
    running_var: Arc<[f32]>,
    eps: f64,
    num_features: usize,
}

impl LazyBatchNorm2d {
    /// Build a BatchNorm2d wrapper. All four buffers must have length
    /// `num_features`.
    pub fn new(
        weight: Arc<[f32]>,
        bias: Arc<[f32]>,
        running_mean: Arc<[f32]>,
        running_var: Arc<[f32]>,
        eps: f64,
        num_features: usize,
    ) -> Result<Self> {
        for (name, buf) in [
            ("weight", &weight),
            ("bias", &bias),
            ("running_mean", &running_mean),
            ("running_var", &running_var),
        ] {
            if buf.len() != num_features {
                return Err(crate::Error::Msg(format!(
                    "LazyBatchNorm2d::new: {name} has length {} but num_features = {}",
                    buf.len(), num_features,
                )).bt());
            }
        }
        if eps < 0.0 {
            return Err(crate::Error::Msg(format!(
                "LazyBatchNorm2d::new: eps must be ≥ 0, got {eps}",
            )).bt());
        }
        Ok(Self {
            weight, bias, running_mean, running_var, eps, num_features,
        })
    }

    /// Reference to the per-channel weight (γ) buffer.
    pub fn weight(&self) -> &Arc<[f32]> { &self.weight }

    /// Reference to the per-channel bias (β) buffer.
    pub fn bias(&self) -> &Arc<[f32]> { &self.bias }

    /// Reference to the running mean buffer.
    pub fn running_mean(&self) -> &Arc<[f32]> { &self.running_mean }

    /// Reference to the running variance buffer.
    pub fn running_var(&self) -> &Arc<[f32]> { &self.running_var }

    /// Epsilon used inside `sqrt(var + eps)`.
    pub fn eps(&self) -> f64 { self.eps }

    /// Channel count (`C`).
    pub fn num_features(&self) -> usize { self.num_features }

    fn fused_gain_bias(&self) -> (Arc<[f32]>, Arc<[f32]>) {
        let eps32 = self.eps as f32;
        let mut gain = Vec::with_capacity(self.num_features);
        let mut bias = Vec::with_capacity(self.num_features);
        for c in 0..self.num_features {
            let scale = self.weight[c] / (self.running_var[c] + eps32).sqrt();
            gain.push(scale);
            bias.push(self.bias[c] - self.running_mean[c] * scale);
        }
        (Arc::from(gain), Arc::from(bias))
    }
}

impl LazyModule for LazyBatchNorm2d {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let dims: Vec<usize> = xs.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(crate::Error::Msg(format!(
                "LazyBatchNorm2d::forward: input must be rank 4 (N, C, H, W), got {dims:?}",
            )).bt());
        }
        if dims[1] != self.num_features {
            return Err(crate::Error::Msg(format!(
                "LazyBatchNorm2d::forward: input channel dim {} != num_features = {}",
                dims[1], self.num_features,
            )).bt());
        }
        let (gain, bias) = self.fused_gain_bias();
        xs.channel_affine_4d(gain, bias)
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
    fn layer_norm_forward_shape_and_finite() {
        let last_dim = 6;
        let seq = 4;
        let gain: Vec<f32> = ramp_f32(last_dim, 0.05, 0.7);
        let bias: Vec<f32> = ramp_f32(last_dim, 0.02, -0.1);
        let x_data: Vec<f32> = ramp_f32(seq * last_dim, 0.03, -0.5);

        let ln = LazyLayerNorm::new(
            Arc::from(gain),
            Some(Arc::from(bias)),
            1e-5,
            last_dim,
        ).unwrap();
        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[seq, last_dim]), &Device::cpu(),
        );
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[seq, last_dim]);
        let got = y.realize_f32();
        assert_eq!(got.len(), seq * last_dim);
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "layer_norm out[{i}] = {v} not finite");
        }
    }

    #[test]
    fn layer_norm_zero_eps_unit_gain_zero_bias_normalizes_to_unit_variance() {
        let last_dim = 5;
        let seq = 3;
        let gain = vec![1.0_f32; last_dim];
        let bias = vec![0.0_f32; last_dim];
        let x_data: Vec<f32> = ramp_f32(seq * last_dim, 0.4, -1.0);

        let ln = LazyLayerNorm::new(
            Arc::from(gain),
            Some(Arc::from(bias)),
            0.0,
            last_dim,
        ).unwrap();
        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[seq, last_dim]), &Device::cpu(),
        );
        let y = ln.forward(&x).unwrap();
        let got = y.realize_f32();
        assert_eq!(got.len(), seq * last_dim);
        for row in 0..seq {
            let slice = &got[row * last_dim..(row + 1) * last_dim];
            let mean: f32 = slice.iter().sum::<f32>() / (last_dim as f32);
            let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                / (last_dim as f32);
            assert!(mean.abs() < 1e-5, "row {row} mean = {mean}");
            assert!((var - 1.0).abs() < 1e-4, "row {row} var = {var}");
        }
    }

    #[test]
    fn rms_norm_matches_rms_norm_affine_directly() {
        let last_dim = 7;
        let seq = 4;
        let gain: Vec<f32> = ramp_f32(last_dim, 0.07, 0.3);
        let x_data: Vec<f32> = ramp_f32(seq * last_dim, 0.05, -0.4);

        let rn = LazyRmsNorm::new(
            Arc::from(gain.clone()), 1e-6, last_dim,
        ).unwrap();
        let x = LazyTensor::from_f32(
            x_data.clone(),
            Shape::from_dims(&[seq, last_dim]), &Device::cpu(),
        );
        let y = rn.forward(&x).unwrap();
        let got = y.realize_f32();

        let x2 = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[seq, last_dim]), &Device::cpu(),
        );
        let expected = x2
            .rms_norm_affine(Arc::from(gain), 1e-6)
            .unwrap()
            .realize_f32();

        assert_eq!(got.len(), expected.len());
        for (i, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "rms_norm[{i}] expected {e}, got {a}",
            );
        }
    }

    #[test]
    fn group_norm_forward_shape() {
        let num_groups = 2;
        let num_channels = 4;
        let b = 2;
        let h = 3;
        let w = 3;
        let gain: Vec<f32> = ramp_f32(num_channels, 0.1, 0.8);
        let bias: Vec<f32> = ramp_f32(num_channels, 0.03, -0.2);
        let x_data: Vec<f32> = ramp_f32(b * num_channels * h * w, 0.02, -0.3);

        let gn = LazyGroupNorm::new(
            Arc::from(gain),
            Arc::from(bias),
            num_groups,
            num_channels,
            1e-5,
        ).unwrap();
        let x = LazyTensor::from_f32(
            x_data,
            Shape::from_dims(&[b, num_channels, h, w]), &Device::cpu(),
        );
        let y = gn.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[b, num_channels, h, w]);
        let got = y.realize_f32();
        assert_eq!(got.len(), b * num_channels * h * w);
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "group_norm out[{i}] = {v} not finite");
        }
    }

    #[test]
    fn batch_norm_eval_uses_running_stats() {
        // x = 2, mean = 1, var = 1, eps = 0, weight = 1, bias = 0.5
        //   -> y = (2 - 1) / sqrt(1 + 0) * 1 + 0.5 = 1.5
        let num_features = 1;
        let n = 1;
        let h = 2;
        let w = 2;
        let weight = vec![1.0_f32; num_features];
        let bias = vec![0.5_f32; num_features];
        let running_mean = vec![1.0_f32; num_features];
        let running_var = vec![1.0_f32; num_features];
        let x_data = vec![2.0_f32; n * num_features * h * w];

        let bn = LazyBatchNorm2d::new(
            Arc::from(weight),
            Arc::from(bias),
            Arc::from(running_mean),
            Arc::from(running_var),
            0.0,
            num_features,
        ).unwrap();
        let x = LazyTensor::from_f32(
            x_data,
            Shape::from_dims(&[n, num_features, h, w]), &Device::cpu(),
        );
        let y = bn.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[n, num_features, h, w]);
        let got = y.realize_f32();
        for (i, v) in got.iter().enumerate() {
            assert!(
                (v - 1.5).abs() < 1e-6,
                "batch_norm_eval[{i}] expected 1.5, got {v}",
            );
        }
    }

    #[test]
    fn batch_norm_eval_per_channel_with_two_features() {
        // Channel 0: x=4, mean=0, var=4, eps=0, weight=2, bias=1
        //   -> scale = 2/2 = 1, offset = 1 - 0*1 = 1, y = 1*4 + 1 = 5
        // Channel 1: x=10, mean=5, var=4, eps=0, weight=2, bias=-1
        //   -> scale = 2/2 = 1, offset = -1 - 5*1 = -6, y = 1*10 - 6 = 4
        let num_features = 2;
        let n = 1;
        let h = 2;
        let w = 1;
        let weight = vec![2.0_f32, 2.0_f32];
        let bias = vec![1.0_f32, -1.0_f32];
        let running_mean = vec![0.0_f32, 5.0_f32];
        let running_var = vec![4.0_f32, 4.0_f32];
        // Layout (N, C, H, W) = (1, 2, 2, 1): ch0 row0, ch0 row1, ch1 row0, ch1 row1
        let x_data = vec![4.0_f32, 4.0_f32, 10.0_f32, 10.0_f32];

        let bn = LazyBatchNorm2d::new(
            Arc::from(weight),
            Arc::from(bias),
            Arc::from(running_mean),
            Arc::from(running_var),
            0.0,
            num_features,
        ).unwrap();
        let x = LazyTensor::from_f32(
            x_data,
            Shape::from_dims(&[n, num_features, h, w]), &Device::cpu(),
        );
        let got = bn.forward(&x).unwrap().realize_f32();
        let expected = vec![5.0_f32, 5.0_f32, 4.0_f32, 4.0_f32];
        for (i, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "batch_norm[{i}] expected {e}, got {a}",
            );
        }
    }
}

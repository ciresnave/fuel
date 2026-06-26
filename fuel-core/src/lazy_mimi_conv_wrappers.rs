//! Mimi learnable down-/up-sampling wrappers (sub-port 3 of
//! port-mimi-conv.md).
//!
//! Ports [`ConvDownsample1d`] / [`ConvTrUpsample1d`] from
//! `fuel_transformers::models::audio::mimi::conv` to the lazy-graph
//! API. Both are thin specializations:
//!
//! - [`ConvDownsample1dWeights`] wraps a
//!   [`StreamableConv1dWeights`] with `kernel = 2 · stride`,
//!   `stride = stride`, `groups = 1`, no bias, no norm,
//!   [`LazyPadMode::Replicate`] padding.
//! - [`ConvTrUpsample1dWeights`] wraps a
//!   [`StreamableConvTranspose1dWeights`] with `kernel = 2 · stride`,
//!   `stride = stride`, `groups = dim` (depthwise), no bias, no norm.
//!
//! `learnt = true` is the only supported mode (eager bails on
//! `learnt = false`); the lazy port mirrors that by simply not
//! exposing the flag — the weights *are* the learnt parameters.

use crate::lazy::LazyTensor;
use crate::lazy_mimi_conv::{LazyPadMode, StreamConv1dState, StreamableConv1dWeights};
use crate::lazy_mimi_conv_transpose::{
    StreamConvTranspose1dState, StreamableConvTranspose1dWeights,
};
use crate::Result;
use std::sync::Arc;

/// Learnable downsampling block — a single strided
/// [`StreamableConv1dWeights`] with replicate padding.
#[derive(Debug, Clone)]
pub struct ConvDownsample1dWeights {
    pub conv: StreamableConv1dWeights,
}

impl ConvDownsample1dWeights {
    /// Build a downsampler with the given `stride`, operating on
    /// `dim` channels. `raw_weight` is the effective conv kernel
    /// (call [`crate::lazy_mimi_conv::bake_weight_norm`] first if the
    /// checkpoint stores the `(weight_g, weight_v)` pair).
    /// Layout `(dim, dim, 2 · stride)` row-major.
    pub fn new(
        stride: usize,
        dim: usize,
        causal: bool,
        raw_weight: Arc<[f32]>,
    ) -> Result<Self> {
        let conv = StreamableConv1dWeights::new(
            raw_weight,
            None,
            dim,
            dim,
            2 * stride,
            stride,
            1,
            causal,
            LazyPadMode::Replicate,
        )?;
        Ok(Self { conv })
    }

    pub fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        self.conv.forward(xs)
    }

    pub fn step(
        &self,
        state: StreamConv1dState,
        xs: &LazyTensor,
    ) -> Result<(StreamConv1dState, Option<LazyTensor>)> {
        self.conv.step(state, xs)
    }
}

/// Learnable upsampling block — a single strided depthwise
/// [`StreamableConvTranspose1dWeights`].
#[derive(Debug, Clone)]
pub struct ConvTrUpsample1dWeights {
    pub convtr: StreamableConvTranspose1dWeights,
}

impl ConvTrUpsample1dWeights {
    /// Build an upsampler with the given `stride`, operating on `dim`
    /// channels (depthwise, `groups = dim`). `raw_weight` is the
    /// effective transposed-conv kernel
    /// (call [`crate::lazy_mimi_conv_transpose::bake_weight_norm_transpose`]
    /// first if the checkpoint stores the `(weight_g, weight_v)` pair).
    /// Layout `(dim, 1, 2 · stride)` row-major — out-per-group is 1.
    pub fn new(
        stride: usize,
        dim: usize,
        causal: bool,
        raw_weight: Arc<[f32]>,
    ) -> Result<Self> {
        let convtr = StreamableConvTranspose1dWeights::new(
            raw_weight,
            None,
            dim,
            dim,
            2 * stride,
            stride,
            dim,
            causal,
        )?;
        Ok(Self { convtr })
    }

    pub fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        self.convtr.forward(xs)
    }

    pub fn step(
        &self,
        state: StreamConvTranspose1dState,
        xs: &LazyTensor,
    ) -> Result<(StreamConvTranspose1dState, Option<LazyTensor>)> {
        self.convtr.step(state, xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use fuel_ir::Shape;

    fn const_xs(b: usize, c: usize, t: usize, src: &[f32]) -> LazyTensor {
        assert_eq!(src.len(), b * c * t);
        LazyTensor::from_f32(
            Arc::from(src.to_vec()),
            Shape::from_dims(&[b, c, t]),
            &Device::cpu(),
        )
    }

    fn ramp_weight(out_c: usize, in_per_group: usize, k: usize) -> Arc<[f32]> {
        let n = out_c * in_per_group * k;
        Arc::from(
            (0..n)
                .map(|i| 0.01 + (i as f32) * 0.03)
                .collect::<Vec<f32>>(),
        )
    }

    fn ramp_xs(b: usize, c: usize, t: usize) -> LazyTensor {
        let n = b * c * t;
        let data: Vec<f32> = (0..n).map(|i| 0.05 + (i as f32) * 0.017).collect();
        const_xs(b, c, t, &data)
    }

    #[test]
    fn downsample_forward_shape_stride_2() {
        let dim = 2;
        let stride = 2;
        let w = ramp_weight(dim, dim, 2 * stride);
        let down = ConvDownsample1dWeights::new(stride, dim, true, w).unwrap();
        let t_in = 16;
        let xs = ramp_xs(1, dim, t_in);
        let y = down.forward(&xs).unwrap();
        assert_eq!(y.shape().dims(), &[1, dim, t_in / stride]);
        for v in &y.realize_f32() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    #[test]
    fn downsample_forward_shape_stride_4() {
        let dim = 3;
        let stride = 4;
        let w = ramp_weight(dim, dim, 2 * stride);
        let down = ConvDownsample1dWeights::new(stride, dim, true, w).unwrap();
        let t_in = 24;
        let xs = ramp_xs(1, dim, t_in);
        let y = down.forward(&xs).unwrap();
        assert_eq!(y.shape().dims(), &[1, dim, t_in / stride]);
    }

    #[test]
    fn upsample_forward_shape_stride_2() {
        let dim = 2;
        let stride = 2;
        // Depthwise: weight shape is (dim, 1, 2*stride).
        let w = ramp_weight(dim, 1, 2 * stride);
        let up = ConvTrUpsample1dWeights::new(stride, dim, true, w).unwrap();
        let t_in = 8;
        let xs = ramp_xs(1, dim, t_in);
        let y = up.forward(&xs).unwrap();
        assert_eq!(y.shape().dims(), &[1, dim, t_in * stride]);
        for v in &y.realize_f32() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    #[test]
    fn upsample_forward_shape_stride_4() {
        let dim = 2;
        let stride = 4;
        let w = ramp_weight(dim, 1, 2 * stride);
        let up = ConvTrUpsample1dWeights::new(stride, dim, true, w).unwrap();
        let t_in = 5;
        let xs = ramp_xs(1, dim, t_in);
        let y = up.forward(&xs).unwrap();
        assert_eq!(y.shape().dims(), &[1, dim, t_in * stride]);
    }

    #[test]
    fn downsample_streaming_equals_one_shot() {
        let dim = 2;
        let stride = 2;
        let w = ramp_weight(dim, dim, 2 * stride);
        let down = ConvDownsample1dWeights::new(stride, dim, true, w).unwrap();
        let t_in = 12;
        let xs = ramp_xs(1, dim, t_in);
        let one_shot = down.forward(&xs).unwrap().realize_f32();

        let chunk = 3;
        let mut state = StreamConv1dState::empty();
        let mut pieces: Vec<LazyTensor> = Vec::new();
        let mut t = 0;
        while t < t_in {
            let len = chunk.min(t_in - t);
            let xc = xs.narrow(2_usize, t, len).unwrap();
            let (s2, y) = down.step(state, &xc).unwrap();
            state = s2;
            if let Some(y) = y {
                pieces.push(y);
            }
            t += len;
        }
        let mut streamed = pieces[0].clone();
        for p in pieces.iter().skip(1) {
            streamed = streamed.concat(p, 2_usize).unwrap();
        }
        let streamed = streamed.realize_f32();

        // Same channel-major flat layout: cout = dim.
        let cout = dim;
        let t_one = one_shot.len() / cout;
        let t_str = streamed.len() / cout;
        let common = t_one.min(t_str);
        assert!(common >= 1, "expected shared frames, got {common}");
        for c in 0..cout {
            for k in 0..common {
                let a = streamed[c * t_str + k];
                let b = one_shot[c * t_one + k];
                assert!(
                    (a - b).abs() < 1e-5,
                    "ch={c} t={k}: streamed={a} one_shot={b}",
                );
            }
        }
    }

    #[test]
    fn upsample_streaming_equals_one_shot() {
        let dim = 2;
        let stride = 2;
        let w = ramp_weight(dim, 1, 2 * stride);
        let up = ConvTrUpsample1dWeights::new(stride, dim, true, w).unwrap();
        let t_in = 8;
        let xs = ramp_xs(1, dim, t_in);
        let one_shot = up.forward(&xs).unwrap().realize_f32();

        let chunk = 2;
        let mut state = StreamConvTranspose1dState::empty();
        let mut pieces: Vec<LazyTensor> = Vec::new();
        let mut t = 0;
        while t < t_in {
            let len = chunk.min(t_in - t);
            let xc = xs.narrow(2_usize, t, len).unwrap();
            let (s2, y) = up.step(state, &xc).unwrap();
            state = s2;
            if let Some(y) = y {
                pieces.push(y);
            }
            t += len;
        }
        let mut streamed = pieces[0].clone();
        for p in pieces.iter().skip(1) {
            streamed = streamed.concat(p, 2_usize).unwrap();
        }
        let streamed = streamed.realize_f32();

        let cout = dim;
        let t_one = one_shot.len() / cout;
        let t_str = streamed.len() / cout;
        let common = t_one.min(t_str);
        assert!(common >= stride, "expected at least {stride} shared per-ch samples, got {common}");
        for c in 0..cout {
            for k in 0..common {
                let a = streamed[c * t_str + k];
                let b = one_shot[c * t_one + k];
                assert!(
                    (a - b).abs() < 1e-5,
                    "ch={c} t={k}: streamed={a} one_shot={b}",
                );
            }
        }
    }
}

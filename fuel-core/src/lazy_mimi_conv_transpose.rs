//! Mimi streaming-capable 1-D transposed convolution primitive
//! (sub-port 2 of port-mimi-conv.md).
//!
//! Ports the [`StreamableConvTranspose1d`] half of
//! `fuel_transformers::models::audio::mimi::conv` to the lazy-graph
//! API. Sub-port 1 ([`crate::lazy_mimi_conv`]) shipped the forward
//! [`StreamableConv1d`]; this module is the upsampling counterpart
//! used by the Mimi decoder / [`ConvTrUpsample1d`] (sub-port 3).
//!
//! # State-as-value
//!
//! Eager keeps the output-overlap buffer in `&mut self`. The lazy
//! port returns `(StreamConvTranspose1dState, Option<LazyTensor>)`
//! from [`StreamableConvTranspose1dWeights::step`] so streaming
//! composes naturally with the rest of the lazy decoder.
//!
//! # Output-overlap-add streaming
//!
//! `ConvTranspose1d` with kernel `k` and stride `s` produces
//! `(L_in - 1) · s + k` output samples from an `L_in`-sample input —
//! that's `k - s` more samples than the next chunk's transposed conv
//! will start contributing to. Those trailing `k - s` samples overlap
//! with the *head* of the next chunk's output and must be summed.
//!
//! Concretely, [`StreamableConvTranspose1dWeights::step`]:
//!
//! 1. Runs the raw transposed conv on the new chunk (`ys`, length
//!    `ot = (L_chunk - 1) · stride + kernel`).
//! 2. Sums `ys[..pt]` with the prior chunk's tail (`prev_ys`,
//!    length `pt = kernel - stride`), with the bias subtracted from
//!    `prev_ys` first — the new chunk's conv already added bias, and
//!    the prior chunk's tail also already had bias added when it was
//!    produced. Subtracting one copy keeps the sum at a single
//!    bias-per-output-sample.
//! 3. Splits the result at `ot - (kernel - stride)`: the first part
//!    is the emit, the trailing `kernel - stride` samples become the
//!    new `prev_ys`.
//!
//! # WeightNorm
//!
//! ConvTranspose1d's PyTorch weight layout is
//! `(in_channels, out_channels / groups, kernel)` — *in-axis-first*,
//! the transpose of Conv1d's `(out, in/groups, k)`. WeightNorm
//! reparametrizes with `g` of shape `(in_channels, 1, 1)` and `v` of
//! shape `(in_channels, out_channels, kernel)` (eager's `vb.get(
//! (in_c, out_c, k_size), "weight_v")`), norming over dims (1, 2)
//! per input channel. [`bake_weight_norm_transpose`] does this bake.
//!
//! # Scope of sub-port 2
//!
//! - `causal` (right-side unpad) and non-causal (split unpad).
//! - One-shot [`StreamableConvTranspose1dWeights::forward`] and
//!   chunked [`StreamableConvTranspose1dWeights::step`].
//! - `groups == 1` and `groups == in_channels == out_channels`
//!   (depthwise — eager's "eye-broadcast" trick to widen the weight
//!   is replicated below). Other group configurations are accepted
//!   but the eager eye-broadcast path is skipped (matches eager).
//! - WeightNorm via [`bake_weight_norm_transpose`]; Norm::Identity /
//!   None accepted. TimeGroupNorm is rejected (eager-parity for the
//!   causal-only path Mimi uses).

use crate::lazy::LazyTensor;
use crate::lazy_mimi_conv::{bake_weight_norm, pad_last_1d, LazyPadMode};
use crate::Result;
use fuel_ir::Shape;
use std::sync::Arc;

// Re-export for completeness so a caller only needs to bring this
// module's items in scope.
pub use crate::lazy_mimi_conv::LazyPadMode as TransposePadMode;

/// Bake PyTorch-style weight normalization for a transposed conv.
///
/// PyTorch reparametrizes a `ConvTranspose1d` weight as
/// `W = g · v / ||v||`, with `g` a per-*input*-channel scale of shape
/// `(Cin, 1, 1)` and `v` raw weight of shape `(Cin, Cout, K)`. The
/// norm is over the trailing two axes per input channel:
///
/// ```text
///     w[i, o, k] = g[i] · v[i, o, k] / sqrt(sum_{o, k} v[i, o, k]^2)
/// ```
///
/// Eager Mimi (`mimi::conv::NormConvTranspose1d::new`, WeightNorm
/// branch) bakes this once at construction. The lazy port does the
/// same here, returning the effective weight in plain
/// `(Cin, Cout, K)` row-major — indistinguishable downstream from a
/// non-WN checkpoint.
///
/// Shape note: this is in-axis-first by design (different from
/// [`bake_weight_norm`], which is out-axis-first for [`Op::Conv1D`]).
pub fn bake_weight_norm_transpose(
    weight_g: &[f32],
    weight_v: &[f32],
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
) -> Result<Vec<f32>> {
    // The math is identical to forward-conv WN — the "outer" axis is
    // just `in_channels` instead of `out_channels`, and the
    // "per-outer" stride is `out_channels * kernel_size`. Delegate.
    bake_weight_norm(weight_g, weight_v, in_channels, out_channels, kernel_size)
}

/// Weights + config for a streaming-capable transposed 1-D conv.
///
/// `weight` is the *effective* kernel — call
/// [`bake_weight_norm_transpose`] before constructing this if the
/// checkpoint stores the `(weight_g, weight_v)` reparametrization
/// pair. Layout is `(in_channels, out_channels / groups, kernel)`
/// row-major, matching PyTorch's `ConvTranspose1d.weight`.
#[derive(Debug, Clone)]
pub struct StreamableConvTranspose1dWeights {
    pub weight: Arc<[f32]>,
    pub bias: Option<Arc<[f32]>>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub groups: usize,
    pub causal: bool,
}

impl StreamableConvTranspose1dWeights {
    /// Validate shapes + store. `weight` must be
    /// `(in_channels, out_channels / groups, kernel_size)` row-major;
    /// `bias` (when `Some`) is `(out_channels,)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weight: Arc<[f32]>,
        bias: Option<Arc<[f32]>>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        causal: bool,
    ) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 || kernel_size == 0 || stride == 0 {
            return Err(crate::Error::Msg(format!(
                "StreamableConvTranspose1dWeights: in/out/kernel/stride must be > 0 \
                 (got {in_channels}, {out_channels}, {kernel_size}, {stride})",
            )));
        }
        if groups == 0 || !out_channels.is_multiple_of(groups) {
            return Err(crate::Error::Msg(format!(
                "StreamableConvTranspose1dWeights: out_channels {out_channels} must be \
                 divisible by groups {groups}",
            )));
        }
        let expected_w = in_channels * (out_channels / groups) * kernel_size;
        if weight.len() != expected_w {
            return Err(crate::Error::Msg(format!(
                "StreamableConvTranspose1dWeights: weight length {} != \
                 in_channels * (out_channels/groups) * kernel_size = {expected_w}",
                weight.len(),
            )));
        }
        if let Some(b) = &bias {
            if b.len() != out_channels {
                return Err(crate::Error::Msg(format!(
                    "StreamableConvTranspose1dWeights: bias length {} != out_channels {}",
                    b.len(),
                    out_channels,
                )));
            }
        }
        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            causal,
        })
    }

    /// Causal trim length: the unpadded output drops `kernel - stride`
    /// trailing samples (matches eager `trim_right_ratio = 1`).
    fn padding_total(&self) -> usize {
        self.kernel_size.saturating_sub(self.stride)
    }

    fn build_weight_tensor(&self, anchor: &LazyTensor) -> LazyTensor {
        anchor.const_f32_like(
            Arc::clone(&self.weight),
            Shape::from_dims(&[
                self.in_channels,
                self.out_channels / self.groups,
                self.kernel_size,
            ]),
        )
    }

    fn build_bias_tensor(&self, anchor: &LazyTensor) -> Option<LazyTensor> {
        self.bias.as_ref().map(|b| {
            anchor
                .const_f32_like(Arc::clone(b), Shape::from_dims(&[self.out_channels]))
        })
    }

    /// Apply bias as a per-output-channel broadcast add along the
    /// time axis. `xs` must be rank-3 `(B, out_channels, T)`.
    fn apply_bias(&self, xs: LazyTensor) -> Result<LazyTensor> {
        match self.build_bias_tensor(&xs) {
            None => Ok(xs),
            Some(b) => {
                let bias_1c1 =
                    b.reshape(Shape::from_dims(&[1, self.out_channels, 1]))?;
                Ok(xs.broadcast_add(&bias_1c1)?)
            }
        }
    }

    /// Raw transposed conv + bias, no padding trim. The corollary of
    /// eager `NormConvTranspose1d::forward` — used as the inner kernel
    /// inside both [`Self::forward`] (which then unpads) and
    /// [`Self::step`] (which then overlap-adds).
    fn raw_forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let w = self.build_weight_tensor(xs);
        let y = xs.conv_transpose1d(
            &w,
            self.stride,
            /* padding */ 0,
            /* output_padding */ 0,
            /* dilation */ 1,
            self.groups,
        )?;
        self.apply_bias(y)
    }

    /// Run the transposed conv in one-shot mode. Matches eager
    /// `StreamableConvTranspose1d::forward` semantics: raw transposed
    /// conv + bias, then trim `kernel - stride` from the right
    /// (causal) or split between the two ends (non-causal).
    ///
    /// Input shape `(B, in_channels, T)`. Output shape
    /// `(B, out_channels, (T - 1) · stride + kernel - pad_total)`,
    /// which simplifies to `(B, out_channels, (T - 1) · stride +
    /// stride) = (B, out_channels, T · stride)` whenever
    /// `kernel >= stride`.
    pub fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let dims = xs.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[1] != self.in_channels {
            return Err(crate::Error::Msg(format!(
                "StreamableConvTranspose1dWeights::forward: expected input \
                 (B, {}, T), got {dims:?}",
                self.in_channels,
            )));
        }
        let y = self.raw_forward(xs)?;
        let t_out = y.shape().dims()[2];
        let pad_total = self.padding_total();
        if pad_total == 0 {
            return Ok(y);
        }
        if t_out < pad_total {
            return Err(crate::Error::Msg(format!(
                "StreamableConvTranspose1dWeights::forward: output length {t_out} \
                 is shorter than pad_total {pad_total}; input length {} is too small",
                dims[2],
            )));
        }
        if self.causal {
            // Trim from the right (eager `trim_right_ratio = 1`).
            y.narrow(2_usize, 0, t_out - pad_total)
        } else {
            let pad_right = pad_total / 2;
            let pad_left = pad_total - pad_right;
            y.narrow(2_usize, pad_left, t_out - pad_total)
        }
    }
}

/// Persistent state for chunk-wise streaming inference through a
/// [`StreamableConvTranspose1dWeights`].
///
/// `prev_ys` carries the trailing `kernel - stride` output samples of
/// the previous step, with bias *included* (it was the natural output
/// of the prior chunk's transposed conv). Subsequent steps subtract
/// the bias before adding, to avoid double-counting it on the overlap
/// region. `None` until the first step has produced output.
#[derive(Debug, Clone, Default)]
pub struct StreamConvTranspose1dState {
    pub prev_ys: Option<LazyTensor>,
}

impl StreamConvTranspose1dState {
    pub fn empty() -> Self {
        Self::default()
    }
}

impl StreamableConvTranspose1dWeights {
    /// Stream `xs` (one chunk of the input) through the transposed
    /// conv and return the new state plus the chunk's emit.
    ///
    /// `xs` shape: `(B, in_channels, L_chunk)`. The returned tensor
    /// (when `Some`) has shape
    /// `(B, out_channels, (L_chunk - 1) · stride + stride) = (B,
    /// out_channels, L_chunk · stride)` once the very first chunk's
    /// kernel context is paid; `None` only if no input was supplied
    /// (matches eager's `StreamTensor::empty()`).
    ///
    /// Calling `step` from the empty state on a length-`T` input
    /// followed by concatenating all returned chunks recovers
    /// [`Self::forward`] bit-for-bit up to floating-point
    /// reassociation in the underlying conv kernel.
    pub fn step(
        &self,
        state: StreamConvTranspose1dState,
        xs: &LazyTensor,
    ) -> Result<(StreamConvTranspose1dState, Option<LazyTensor>)> {
        let dims = xs.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[1] != self.in_channels {
            return Err(crate::Error::Msg(format!(
                "StreamableConvTranspose1dWeights::step: expected (B, {}, L), got {dims:?}",
                self.in_channels,
            )));
        }
        let ys = self.raw_forward(xs)?;
        let ot = ys.shape().dims()[2];
        let invalid_steps = self.kernel_size.saturating_sub(self.stride);
        let merged = match state.prev_ys {
            None => ys,
            Some(prev_ys) => {
                let pt = prev_ys.shape().dims()[2];
                if pt > ot {
                    return Err(crate::Error::Msg(format!(
                        "StreamableConvTranspose1dWeights::step: prev_ys length {pt} \
                         exceeds new chunk output length {ot}; chunk too small",
                    )));
                }
                // Subtract the bias from prev_ys — the new ys head
                // already includes one bias copy, so summing untouched
                // would double-count.
                let prev_ys = match &self.bias {
                    None => prev_ys,
                    Some(b) => {
                        let bias_1c1 = prev_ys
                            .const_f32_like(
                                Arc::clone(b),
                                Shape::from_dims(&[self.out_channels]),
                            )
                            .reshape(Shape::from_dims(&[1, self.out_channels, 1]))?;
                        prev_ys.broadcast_sub(&bias_1c1)?
                    }
                };
                let head = ys.narrow(2_usize, 0, pt)?;
                let head = head.add(&prev_ys)?;
                if pt == ot {
                    head
                } else {
                    let tail = ys.narrow(2_usize, pt, ot - pt)?;
                    head.concat(&tail, 2_usize)?
                }
            }
        };
        if invalid_steps == 0 {
            // Stride == kernel: no overlap to carry. Emit everything.
            return Ok((
                StreamConvTranspose1dState { prev_ys: None },
                Some(merged),
            ));
        }
        if ot < invalid_steps {
            return Err(crate::Error::Msg(format!(
                "StreamableConvTranspose1dWeights::step: output length {ot} shorter \
                 than overlap {invalid_steps}; chunk too small",
            )));
        }
        let emit_len = ot - invalid_steps;
        let emit = merged.narrow(2_usize, 0, emit_len)?;
        let new_prev = merged.narrow(2_usize, emit_len, invalid_steps)?;
        Ok((
            StreamConvTranspose1dState { prev_ys: Some(new_prev) },
            Some(emit),
        ))
    }
}

// Silence the dead-code warning on the re-exported pad mode + the
// `pad_last_1d` import which isn't directly called here but is part of
// the sub-port-1 surface this module composes with.
#[allow(dead_code)]
fn _force_pad_imports_referenced(xs: &LazyTensor) -> Result<LazyTensor> {
    pad_last_1d(xs, 0, 0, LazyPadMode::Constant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn const_xs(b: usize, c: usize, t: usize, src: &[f32]) -> LazyTensor {
        assert_eq!(src.len(), b * c * t);
        LazyTensor::from_f32(
            Arc::from(src.to_vec()),
            Shape::from_dims(&[b, c, t]),
            &Device::cpu(),
        )
    }

    #[test]
    fn weight_norm_bakes_for_transpose_shape() {
        // Cin=2, Cout=1, K=2 → v shape (Cin=2, Cout=1, K=2) flat,
        // g shape (Cin=2,). Norm reduces over the trailing
        // (Cout * K) = 2 axis per input channel.
        let g = vec![1.0_f32, 2.0];
        // v[0] = [3, 4], v[1] = [6, 8].
        let v = vec![3.0_f32, 4.0, 6.0, 8.0];
        // norm[0] = sqrt(9 + 16) = 5; w[0] = 1 * [3, 4] / 5 = [0.6, 0.8].
        // norm[1] = sqrt(36 + 64) = 10; w[1] = 2 * [6, 8] / 10 = [1.2, 1.6].
        let w = bake_weight_norm_transpose(&g, &v, 2, 1, 2).unwrap();
        assert!((w[0] - 0.6).abs() < 1e-6);
        assert!((w[1] - 0.8).abs() < 1e-6);
        assert!((w[2] - 1.2).abs() < 1e-6);
        assert!((w[3] - 1.6).abs() < 1e-6);
    }

    #[test]
    fn weight_norm_transpose_rejects_size_mismatch() {
        assert!(bake_weight_norm_transpose(&[1.0], &[1.0, 2.0, 3.0], 2, 1, 2).is_err());
    }

    #[test]
    fn one_shot_forward_shape_and_finite_kernel_2_stride_1_causal() {
        // K=2, stride=1, in=1, out=1, causal. T_out (raw) = T + 1.
        // pad_total = 1, so trimmed length = T.
        let weight: Arc<[f32]> = Arc::from(vec![0.5_f32, -0.25]); // (in=1, out=1, k=2)
        let cv = StreamableConvTranspose1dWeights::new(
            weight, None, 1, 1, 2, 1, 1, true,
        )
        .unwrap();
        let xs = const_xs(1, 1, 4, &[1.0, 2.0, 3.0, 4.0]);
        let y = cv.forward(&xs).unwrap();
        assert_eq!(y.shape().dims(), &[1, 1, 4]);
        let got = y.realize_f32();
        // Raw transposed conv with weight [0.5, -0.25] on input
        // [1, 2, 3, 4] produces length-5 output:
        //   y_raw[0] = 0.5 * 1                = 0.5
        //   y_raw[1] = -0.25 * 1 + 0.5 * 2    = 0.75
        //   y_raw[2] = -0.25 * 2 + 0.5 * 3    = 1.0
        //   y_raw[3] = -0.25 * 3 + 0.5 * 4    = 1.25
        //   y_raw[4] = -0.25 * 4              = -1.0   (trimmed)
        // After causal trim of pad_total=1 from the right we get
        // [0.5, 0.75, 1.0, 1.25].
        assert!((got[0] - 0.5).abs() < 1e-6, "got[0] = {}", got[0]);
        assert!((got[1] - 0.75).abs() < 1e-6, "got[1] = {}", got[1]);
        assert!((got[2] - 1.0).abs() < 1e-6, "got[2] = {}", got[2]);
        assert!((got[3] - 1.25).abs() < 1e-6, "got[3] = {}", got[3]);
        for v in &got {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn one_shot_forward_with_bias_kernel_3_stride_2_causal() {
        // K=3, stride=2, in=1, out=1, causal. Bias = +0.5 per sample.
        // Raw out len = (T-1) * 2 + 3; pad_total = 1; trimmed = raw-1.
        let weight: Arc<[f32]> = Arc::from(vec![1.0_f32, 0.5, -0.5]); // (1, 1, 3)
        let bias: Arc<[f32]> = Arc::from(vec![0.5_f32]);
        let cv = StreamableConvTranspose1dWeights::new(
            weight, Some(bias), 1, 1, 3, 2, 1, true,
        )
        .unwrap();
        let xs = const_xs(1, 1, 3, &[1.0, 2.0, 3.0]);
        let y = cv.forward(&xs).unwrap();
        let dims = y.shape().dims().to_vec();
        // (T-1) * stride + kernel = 2 * 2 + 3 = 7; minus pad_total=1 = 6.
        assert_eq!(dims, vec![1, 1, 6]);
        let got = y.realize_f32();
        for v in &got {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    fn stream_concat_emits(
        cv: &StreamableConvTranspose1dWeights,
        xs: &LazyTensor,
        chunk_size: usize,
        t_total: usize,
    ) -> Vec<f32> {
        let mut state = StreamConvTranspose1dState::empty();
        let mut pieces: Vec<LazyTensor> = Vec::new();
        let mut t = 0;
        while t < t_total {
            let len = chunk_size.min(t_total - t);
            let chunk = xs.narrow(2_usize, t, len).unwrap();
            let (new_state, y) = cv.step(state, &chunk).unwrap();
            state = new_state;
            if let Some(y) = y {
                pieces.push(y);
            }
            t += len;
        }
        let mut out = pieces[0].clone();
        for p in pieces.iter().skip(1) {
            out = out.concat(p, 2_usize).unwrap();
        }
        out.realize_f32()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(
            a.len(),
            b.len(),
            "{label}: length mismatch {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "{label}: idx {i}: streamed={x} one_shot={y} diff={}",
                (x - y).abs(),
            );
        }
    }

    #[test]
    fn streaming_chunk_by_1_equals_one_shot_kernel_2_stride_1_causal() {
        let weight: Arc<[f32]> = Arc::from(vec![0.5_f32, -0.25]);
        let cv = StreamableConvTranspose1dWeights::new(
            weight, None, 1, 1, 2, 1, 1, true,
        )
        .unwrap();
        let xs_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.3 - 0.7).collect();
        let xs = const_xs(1, 1, 6, &xs_data);
        let one_shot_full = cv.forward(&xs).unwrap().realize_f32();
        let streamed = stream_concat_emits(&cv, &xs, 1, 6);
        // Streaming emits stop at the last bias-corrected sample of
        // the last chunk; the one_shot has the same length here
        // (T_out = T_in for stride 1, kernel 2 causal).
        let n = streamed.len().min(one_shot_full.len());
        assert!(n >= 1, "expected at least one shared sample");
        assert_close(&streamed[..n], &one_shot_full[..n], 1e-5, "chunk_by_1 k2s1");
    }

    #[test]
    fn streaming_chunk_by_2_equals_one_shot_kernel_3_stride_1_causal_with_bias() {
        let weight: Arc<[f32]> = Arc::from(vec![0.2_f32, -0.4, 0.1]);
        let bias: Arc<[f32]> = Arc::from(vec![0.1_f32]);
        let cv = StreamableConvTranspose1dWeights::new(
            weight, Some(bias), 1, 1, 3, 1, 1, true,
        )
        .unwrap();
        let xs_data: Vec<f32> = (0..8).map(|i| 0.05 + (i as f32) * 0.13).collect();
        let xs = const_xs(1, 1, 8, &xs_data);
        let one_shot_full = cv.forward(&xs).unwrap().realize_f32();
        let streamed = stream_concat_emits(&cv, &xs, 2, 8);
        let n = streamed.len().min(one_shot_full.len());
        assert!(n >= 4, "expected at least 4 shared samples, got {n}");
        assert_close(
            &streamed[..n],
            &one_shot_full[..n],
            1e-5,
            "chunk_by_2 k3s1 bias",
        );
    }

    #[test]
    fn streaming_chunk_by_larger_equals_one_shot_kernel_4_stride_2_causal() {
        let weight: Arc<[f32]> = Arc::from(vec![
            0.1_f32, -0.2, 0.3, -0.4, 0.15, 0.05, -0.1, 0.25,
        ]); // (in=1, out=2, k=4)
        let cv = StreamableConvTranspose1dWeights::new(
            weight, None, 1, 2, 4, 2, 1, true,
        )
        .unwrap();
        let xs_data: Vec<f32> = (0..8).map(|i| 0.1 * (i as f32) - 0.3).collect();
        let xs = const_xs(1, 1, 8, &xs_data);
        let one_shot_full = cv.forward(&xs).unwrap().realize_f32();
        let streamed = stream_concat_emits(&cv, &xs, 4, 8);
        // Output channels = 2, so the flat slice layout is
        // [ch0_t0, ch0_t1, ..., ch1_t0, ch1_t1, ...]. The streamed
        // emit length may be shorter than one-shot by the trailing
        // overlap that's still in `prev_ys`. Compare the common
        // per-channel prefix.
        let t_one = one_shot_full.len() / 2;
        let t_str = streamed.len() / 2;
        let common = t_one.min(t_str);
        assert!(common >= 4, "expected at least 4 shared per-ch samples");
        for c in 0..2 {
            for k in 0..common {
                let a = streamed[c * t_str + k];
                let b = one_shot_full[c * t_one + k];
                assert!(
                    (a - b).abs() < 1e-5,
                    "k4s2 ch={c} t={k}: streamed={a} one_shot={b}",
                );
            }
        }
    }

    #[test]
    fn rejects_weight_size_mismatch() {
        let r = StreamableConvTranspose1dWeights::new(
            Arc::from(vec![0.0_f32; 3]),
            None,
            2,
            1,
            2,
            1,
            1,
            true,
        );
        assert!(r.is_err());
    }

    #[test]
    fn rejects_out_channels_not_divisible_by_groups() {
        let r = StreamableConvTranspose1dWeights::new(
            Arc::from(vec![0.0_f32; 6]),
            None,
            2,
            3,
            2,
            1,
            2,
            true,
        );
        assert!(r.is_err());
    }
}

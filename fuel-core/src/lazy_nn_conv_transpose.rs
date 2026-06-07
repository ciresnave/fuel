//! Lazy `ConvTranspose1d` / `ConvTranspose2d` Module wrappers over
//! [`LazyTensor`].
//!
//! Mirrors the eager [`fuel_nn::ConvTranspose1d`] /
//! [`fuel_nn::ConvTranspose2d`] surface: each layer holds a
//! [`WeightStorage`] weight plus an optional bias and a config struct
//! controlling `padding` / `output_padding` / `stride` / `dilation` /
//! `groups`. `forward` materializes the weight (and bias) as graph
//! constants on the activation's graph at lazy-graph-build time and
//! delegates to [`LazyTensor::conv_transpose1d`] /
//! [`LazyTensor::conv_transpose2d`].
//!
//! # Lazy-graph semantics
//!
//! No computation happens inside `forward` — the weight is wrapped as
//! a `Const` node on the activation's graph, the transposed conv is
//! appended as a single [`fuel_graph::Op::ConvTranspose2D`] node (the
//! 1-D variant lifts to rank-4 transparently), and the bias add is
//! appended as a broadcast add. Validation (rank, channel divisibility,
//! stride / dilation > 0) surfaces as a typed [`crate::Error`] at
//! build time, matching the project rule that every check that *can*
//! run at graph-build time *must*.
//!
//! # Scope (v1)
//!
//! - F32 (and BF16 / Q4_0 via [`WeightStorage::const_like`]) weights;
//!   the activation must be float — the same constraint as the eager
//!   `Module` impls.
//! - The bias broadcast assumes an `f32` bias tensor — matches eager;
//!   bf16 bias support follows the same shape if a checkpoint ever
//!   needs it.
//! - PyTorch weight layout: `[Cin, Cout / groups, K]` for 1-D and
//!   `[Cin, Cout / groups, Kh, Kw]` for 2-D (note the *transposed*
//!   channel order vs forward `Conv{1,2}d`).
//!
//! # Limitations
//!
//! - 1-D path supports `groups >= 1`; the 2-D LazyTensor primitive
//!   accepts `groups` too, kept in the config for symmetry with the
//!   eager API even though the eager `ConvTranspose2dConfig` doesn't
//!   carry it yet (the eager `// TODO: support groups.` comment will
//!   land here too when the eager side catches up).
//! - Dilation is forwarded as-is to the underlying primitive — the
//!   IR carries it; no extra layer-level guard needed.

use crate::Result;
use crate::lazy::{LazyTensor, WeightStorage};
use fuel_core_types::Shape;
use std::sync::Arc;

// ===========================================================================
// 1-D transposed conv
// ===========================================================================

/// Configuration for [`ConvTranspose1d`].
///
/// Default: `padding=0`, `output_padding=0`, `stride=1`, `dilation=1`,
/// `groups=1` — matches the eager [`fuel_nn::ConvTranspose1dConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose1dConfig {
    /// Zero-padding implicitly removed from both sides of the output.
    pub padding: usize,
    /// Extra zero-padding appended to one side of the output, used to
    /// disambiguate the two possible output sizes for `stride > 1`.
    pub output_padding: usize,
    /// Stride of the (fractionally-strided) convolution.
    pub stride: usize,
    /// Spacing between kernel elements.
    pub dilation: usize,
    /// Number of blocked connections from input to output channels.
    pub groups: usize,
}

impl Default for ConvTranspose1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            output_padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

impl ConvTranspose1dConfig {
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }
    pub fn with_output_padding(mut self, output_padding: usize) -> Self {
        self.output_padding = output_padding;
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

/// 1-D transposed convolution layer over [`LazyTensor`].
///
/// Holds a [`WeightStorage`] weight in PyTorch's
/// `[Cin, Cout / groups, K]` layout (transposed channel order vs
/// forward [`crate::lazy_nn::LazyConv1d`]) plus an optional bias of
/// length `out_channels`.
#[derive(Debug, Clone)]
pub struct ConvTranspose1d {
    weight: WeightStorage,
    bias: Option<Arc<[f32]>>,
    config: ConvTranspose1dConfig,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
}

impl ConvTranspose1d {
    /// Build a 1-D transposed convolution from a weight storage and an
    /// optional bias.
    ///
    /// `weight` must have `in_channels * (out_channels / groups) *
    /// kernel_size` elements in `[Cin, Cout / groups, K]` row-major
    /// order. `bias`, when `Some`, must have length `out_channels`.
    ///
    /// Validates groups divisibility and shape at build time — bad
    /// inputs surface as typed errors here rather than panicking
    /// later inside [`LazyTensor::conv_transpose1d`].
    pub fn new(
        weight: WeightStorage,
        bias: Option<Arc<[f32]>>,
        config: ConvTranspose1dConfig,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        if config.groups < 1 {
            return Err(crate::Error::Msg(format!(
                "ConvTranspose1d::new: groups must be >= 1, got {}",
                config.groups,
            ))
            .bt());
        }
        if config.stride < 1 {
            return Err(crate::Error::Msg(format!(
                "ConvTranspose1d::new: stride must be >= 1, got {}",
                config.stride,
            ))
            .bt());
        }
        if config.dilation < 1 {
            return Err(crate::Error::Msg(format!(
                "ConvTranspose1d::new: dilation must be >= 1, got {}",
                config.dilation,
            ))
            .bt());
        }
        if !out_channels.is_multiple_of(config.groups) {
            return Err(crate::Error::Msg(format!(
                "ConvTranspose1d::new: out_channels ({}) must be \
                 divisible by groups ({})",
                out_channels, config.groups,
            ))
            .bt());
        }
        if !in_channels.is_multiple_of(config.groups) {
            return Err(crate::Error::Msg(format!(
                "ConvTranspose1d::new: in_channels ({}) must be \
                 divisible by groups ({})",
                in_channels, config.groups,
            ))
            .bt());
        }
        let expected = in_channels * (out_channels / config.groups) * kernel_size;
        if weight.elem_count() != expected {
            return Err(crate::Error::Msg(format!(
                "ConvTranspose1d::new: weight has {} elements but \
                 in_channels * (out_channels / groups) * kernel_size \
                 = {} * {} * {} = {}",
                weight.elem_count(),
                in_channels,
                out_channels / config.groups,
                kernel_size,
                expected,
            ))
            .bt());
        }
        if let Some(b) = bias.as_ref() {
            if b.len() != out_channels {
                return Err(crate::Error::Msg(format!(
                    "ConvTranspose1d::new: bias has length {} but \
                     out_channels = {}",
                    b.len(),
                    out_channels,
                ))
                .bt());
            }
        }
        Ok(Self {
            weight,
            bias,
            config,
            in_channels,
            out_channels,
            kernel_size,
        })
    }

    pub fn config(&self) -> &ConvTranspose1dConfig {
        &self.config
    }
    pub fn weight(&self) -> &WeightStorage {
        &self.weight
    }
    pub fn bias(&self) -> Option<&Arc<[f32]>> {
        self.bias.as_ref()
    }
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Build the forward graph: `ConvTranspose1d(x) + bias`.
    ///
    /// `x` must be rank-3 `[N, Cin, L]`. Returns rank-3
    /// `[N, Cout, Lout]` with
    /// `Lout = (L - 1) * stride - 2 * padding + dilation * (K - 1) +
    /// output_padding + 1`.
    pub fn forward(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let w_shape = Shape::from_dims(&[
            self.in_channels,
            self.out_channels / self.config.groups,
            self.kernel_size,
        ]);
        let w_t = self.weight.const_like(x, w_shape)?;
        let y = x.conv_transpose1d(
            &w_t,
            self.config.stride,
            self.config.padding,
            self.config.output_padding,
            self.config.dilation,
            self.config.groups,
        )?;
        match &self.bias {
            None => Ok(y),
            Some(b) => {
                let bias_t = y
                    .const_f32_like(
                        Arc::clone(b),
                        Shape::from_dims(&[self.out_channels]),
                    )
                    .reshape(Shape::from_dims(&[1, self.out_channels, 1]))?;
                Ok(y.broadcast_add(&bias_t)?)
            }
        }
    }

    /// Load weights from a HuggingFace `MmapedSafetensors` checkpoint.
    /// `prefix` is the parameter prefix (without trailing dot), e.g.
    /// `"decoder.upsample"`. Reads `{prefix}.weight` as a flat
    /// `[Cin, Cout / groups, K]` tensor and `{prefix}.bias` (optional)
    /// as `[Cout]`. Source dtype is upcast to f32 via
    /// [`crate::lazy::load_tensor_as_f32`].
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        config: ConvTranspose1dConfig,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        use crate::lazy::load_tensor_as_f32;
        let w = load_tensor_as_f32(st, &format!("{prefix}.weight"))?;
        let weight_arc: Arc<[f32]> = Arc::<[f32]>::from(w);
        let bias = match load_tensor_as_f32(st, &format!("{prefix}.bias")) {
            Ok(b) => Some(Arc::<[f32]>::from(b)),
            Err(_) => None,
        };
        Self::new(
            WeightStorage::F32(weight_arc),
            bias,
            config,
            in_channels,
            out_channels,
            kernel_size,
        )
    }
}

// ===========================================================================
// 2-D transposed conv
// ===========================================================================

/// Configuration for [`ConvTranspose2d`].
///
/// Default: `padding=0`, `output_padding=0`, `stride=1`,
/// `dilation=1`, `groups=1`. Mirrors the eager
/// [`fuel_nn::ConvTranspose2dConfig`] surface, plus a `groups` field
/// (the eager side has a `TODO: support groups.` and the underlying
/// [`LazyTensor::conv_transpose2d`] already takes it).
///
/// All `(usize, usize)` fields are `(height, width)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose2dConfig {
    pub padding: (usize, usize),
    pub output_padding: (usize, usize),
    pub stride: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl Default for ConvTranspose2dConfig {
    fn default() -> Self {
        Self {
            padding: (0, 0),
            output_padding: (0, 0),
            stride: (1, 1),
            dilation: (1, 1),
            groups: 1,
        }
    }
}

impl ConvTranspose2dConfig {
    pub fn with_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }
    pub fn with_output_padding(
        mut self,
        output_padding: (usize, usize),
    ) -> Self {
        self.output_padding = output_padding;
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

/// 2-D transposed convolution layer over [`LazyTensor`].
///
/// Weight layout is PyTorch's `[Cin, Cout / groups, Kh, Kw]` (transposed
/// channel order vs forward [`crate::lazy_nn::LazyConv2d`]).
#[derive(Debug, Clone)]
pub struct ConvTranspose2d {
    weight: WeightStorage,
    bias: Option<Arc<[f32]>>,
    config: ConvTranspose2dConfig,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
}

impl ConvTranspose2d {
    /// Build a 2-D transposed convolution from a weight storage and an
    /// optional bias.
    ///
    /// `weight` must have `in_channels * (out_channels / groups) *
    /// kernel_h * kernel_w` elements in `[Cin, Cout / groups, Kh, Kw]`
    /// row-major order. `bias`, when `Some`, must have length
    /// `out_channels`.
    pub fn new(
        weight: WeightStorage,
        bias: Option<Arc<[f32]>>,
        config: ConvTranspose2dConfig,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
    ) -> Result<Self> {
        if config.groups < 1 {
            return Err(crate::Error::Msg(format!(
                "ConvTranspose2d::new: groups must be >= 1, got {}",
                config.groups,
            ))
            .bt());
        }
        if config.stride.0 < 1 || config.stride.1 < 1 {
            return Err(crate::Error::Msg(format!(
                "ConvTranspose2d::new: stride must be >= 1, got {:?}",
                config.stride,
            ))
            .bt());
        }
        if config.dilation.0 < 1 || config.dilation.1 < 1 {
            return Err(crate::Error::Msg(format!(
                "ConvTranspose2d::new: dilation must be >= 1, got {:?}",
                config.dilation,
            ))
            .bt());
        }
        if !out_channels.is_multiple_of(config.groups) {
            return Err(crate::Error::Msg(format!(
                "ConvTranspose2d::new: out_channels ({}) must be \
                 divisible by groups ({})",
                out_channels, config.groups,
            ))
            .bt());
        }
        if !in_channels.is_multiple_of(config.groups) {
            return Err(crate::Error::Msg(format!(
                "ConvTranspose2d::new: in_channels ({}) must be \
                 divisible by groups ({})",
                in_channels, config.groups,
            ))
            .bt());
        }
        let expected = in_channels
            * (out_channels / config.groups)
            * kernel_h
            * kernel_w;
        if weight.elem_count() != expected {
            return Err(crate::Error::Msg(format!(
                "ConvTranspose2d::new: weight has {} elements but \
                 in_channels * (out_channels / groups) * kernel_h * \
                 kernel_w = {} * {} * {} * {} = {}",
                weight.elem_count(),
                in_channels,
                out_channels / config.groups,
                kernel_h,
                kernel_w,
                expected,
            ))
            .bt());
        }
        if let Some(b) = bias.as_ref() {
            if b.len() != out_channels {
                return Err(crate::Error::Msg(format!(
                    "ConvTranspose2d::new: bias has length {} but \
                     out_channels = {}",
                    b.len(),
                    out_channels,
                ))
                .bt());
            }
        }
        Ok(Self {
            weight,
            bias,
            config,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
        })
    }

    pub fn config(&self) -> &ConvTranspose2dConfig {
        &self.config
    }
    pub fn weight(&self) -> &WeightStorage {
        &self.weight
    }
    pub fn bias(&self) -> Option<&Arc<[f32]>> {
        self.bias.as_ref()
    }
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
    pub fn kernel_h(&self) -> usize {
        self.kernel_h
    }
    pub fn kernel_w(&self) -> usize {
        self.kernel_w
    }

    /// Build the forward graph: `ConvTranspose2d(x) + bias`.
    ///
    /// `x` must be rank-4 `[N, Cin, H, W]`. Returns rank-4
    /// `[N, Cout, Hout, Wout]`.
    pub fn forward(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let w_shape = Shape::from_dims(&[
            self.in_channels,
            self.out_channels / self.config.groups,
            self.kernel_h,
            self.kernel_w,
        ]);
        let w_t = self.weight.const_like(x, w_shape)?;
        let y = x.conv_transpose2d(
            &w_t,
            self.config.stride,
            self.config.padding,
            self.config.output_padding,
            self.config.dilation,
            self.config.groups,
        )?;
        match &self.bias {
            None => Ok(y),
            Some(b) => {
                let bias_t = y
                    .const_f32_like(
                        Arc::clone(b),
                        Shape::from_dims(&[self.out_channels]),
                    )
                    .reshape(Shape::from_dims(&[1, self.out_channels, 1, 1]))?;
                Ok(y.broadcast_add(&bias_t)?)
            }
        }
    }

    /// Load weights from a HuggingFace `MmapedSafetensors` checkpoint.
    /// `prefix` is the parameter prefix (without trailing dot), e.g.
    /// `"decoder.upconv"`. Reads `{prefix}.weight` as a flat
    /// `[Cin, Cout / groups, Kh, Kw]` tensor and `{prefix}.bias`
    /// (optional) as `[Cout]`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        config: ConvTranspose2dConfig,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
    ) -> Result<Self> {
        use crate::lazy::load_tensor_as_f32;
        let w = load_tensor_as_f32(st, &format!("{prefix}.weight"))?;
        let weight_arc: Arc<[f32]> = Arc::<[f32]>::from(w);
        let bias = match load_tensor_as_f32(st, &format!("{prefix}.bias")) {
            Ok(b) => Some(Arc::<[f32]>::from(b)),
            Err(_) => None,
        };
        Self::new(
            WeightStorage::F32(weight_arc),
            bias,
            config,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn ramp_f32(n: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * scale + offset).collect()
    }

    #[test]
    fn conv_transpose1d_forward_shape_default_config() {
        // ConvTranspose1d output length =
        //   (L - 1) * stride - 2*padding + dilation*(K-1) + output_padding + 1
        // With default config (stride=1, padding=0, dilation=1,
        // output_padding=0): Lout = L + K - 1.
        let n = 2;
        let cin = 3;
        let cout = 4;
        let l = 5;
        let k = 3;
        let cfg = ConvTranspose1dConfig::default();

        let w: Vec<f32> = ramp_f32(cin * cout * k, 0.05, -0.2);
        let bias: Vec<f32> = ramp_f32(cout, 0.1, 0.0);
        let layer = ConvTranspose1d::new(
            WeightStorage::F32(Arc::<[f32]>::from(w)),
            Some(Arc::<[f32]>::from(bias)),
            cfg,
            cin,
            cout,
            k,
        )
        .unwrap();

        let x_data: Vec<f32> = ramp_f32(n * cin * l, 0.03, -0.4);
        let x = LazyTensor::from_f32(
            x_data,
            Shape::from_dims(&[n, cin, l]),
            &Device::cpu(),
        );
        let y = layer.forward(&x).unwrap();
        let l_out = l + k - 1;
        assert_eq!(y.shape().dims(), &[n, cout, l_out]);
        let got = y.realize_f32();
        assert_eq!(got.len(), n * cout * l_out);
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "conv_transpose1d out[{i}] = {v} not finite");
        }
    }

    #[test]
    fn conv_transpose1d_strided_matches_direct_call() {
        // Module forward must agree byte-for-byte with hand-built
        // LazyTensor::conv_transpose1d + broadcast_add.
        let n = 1;
        let cin = 2;
        let cout = 3;
        let l = 4;
        let k = 3;
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride: 2,
            dilation: 1,
            groups: 1,
        };

        let weight_vec: Vec<f32> = ramp_f32(cin * cout * k, 0.04, 0.1);
        let bias_vec: Vec<f32> = ramp_f32(cout, 0.5, -0.2);
        let weight_arc: Arc<[f32]> = Arc::<[f32]>::from(weight_vec);
        let bias_arc: Arc<[f32]> = Arc::<[f32]>::from(bias_vec);

        let layer = ConvTranspose1d::new(
            WeightStorage::F32(Arc::clone(&weight_arc)),
            Some(Arc::clone(&bias_arc)),
            cfg,
            cin,
            cout,
            k,
        )
        .unwrap();

        let x_data: Vec<f32> = ramp_f32(n * cin * l, 0.02, -0.3);
        let x = LazyTensor::from_f32(
            x_data.clone(),
            Shape::from_dims(&[n, cin, l]),
            &Device::cpu(),
        );
        let via_module = layer.forward(&x).unwrap().realize_f32();

        let x2 = LazyTensor::from_f32(
            x_data,
            Shape::from_dims(&[n, cin, l]),
            &Device::cpu(),
        );
        let w_t = x2.const_f32_like(
            Arc::clone(&weight_arc),
            Shape::from_dims(&[cin, cout, k]),
        );
        let direct_raw = x2
            .conv_transpose1d(
                &w_t,
                cfg.stride,
                cfg.padding,
                cfg.output_padding,
                cfg.dilation,
                cfg.groups,
            )
            .unwrap();
        let b_t = direct_raw
            .const_f32_like(
                Arc::clone(&bias_arc),
                Shape::from_dims(&[cout]),
            )
            .reshape(Shape::from_dims(&[1, cout, 1]))
            .unwrap();
        let direct = direct_raw.broadcast_add(&b_t).unwrap().realize_f32();

        assert_eq!(via_module.len(), direct.len());
        for (i, (a, d)) in via_module.iter().zip(direct.iter()).enumerate() {
            assert!(
                (a - d).abs() < 1e-5,
                "conv_transpose1d strided[{i}] module {a} != direct {d}",
            );
        }
    }

    #[test]
    fn conv_transpose1d_rejects_weight_size_mismatch() {
        let bad_weight: Arc<[f32]> = Arc::<[f32]>::from(vec![0.0_f32; 5]);
        let r = ConvTranspose1d::new(
            WeightStorage::F32(bad_weight),
            None,
            ConvTranspose1dConfig::default(),
            /* in */ 2, /* out */ 3, /* k */ 4,
        );
        assert!(r.is_err());
    }

    #[test]
    fn conv_transpose1d_rejects_bad_groups() {
        // out_channels = 3 not divisible by groups = 2.
        let w: Arc<[f32]> = Arc::<[f32]>::from(vec![0.0_f32; 2 * 3 * 2]);
        let cfg = ConvTranspose1dConfig {
            groups: 2,
            ..ConvTranspose1dConfig::default()
        };
        let r = ConvTranspose1d::new(
            WeightStorage::F32(w),
            None,
            cfg,
            2,
            3,
            2,
        );
        assert!(r.is_err());
    }

    #[test]
    fn conv_transpose2d_forward_shape_default_config() {
        // With defaults (stride=1, padding=0, dilation=1,
        // output_padding=0): Hout = H + Kh - 1, Wout = W + Kw - 1.
        let n = 1;
        let cin = 2;
        let cout = 3;
        let h = 4;
        let w_in = 4;
        let kh = 3;
        let kw = 3;
        let cfg = ConvTranspose2dConfig::default();

        let weight: Vec<f32> = ramp_f32(cin * cout * kh * kw, 0.02, -0.1);
        let bias: Vec<f32> = ramp_f32(cout, 0.05, 0.2);
        let layer = ConvTranspose2d::new(
            WeightStorage::F32(Arc::<[f32]>::from(weight)),
            Some(Arc::<[f32]>::from(bias)),
            cfg,
            cin,
            cout,
            kh,
            kw,
        )
        .unwrap();

        let x_data: Vec<f32> = ramp_f32(n * cin * h * w_in, 0.01, -0.5);
        let x = LazyTensor::from_f32(
            x_data,
            Shape::from_dims(&[n, cin, h, w_in]),
            &Device::cpu(),
        );
        let y = layer.forward(&x).unwrap();
        let h_out = h + kh - 1;
        let w_out = w_in + kw - 1;
        assert_eq!(y.shape().dims(), &[n, cout, h_out, w_out]);
        let got = y.realize_f32();
        assert_eq!(got.len(), n * cout * h_out * w_out);
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "conv_transpose2d out[{i}] = {v} not finite");
        }
    }

    #[test]
    fn conv_transpose2d_strided_matches_direct_call() {
        // Stride 2 upsamples spatially; verify the module's bias
        // broadcast lands on the same result the direct call produces.
        let n = 1;
        let cin = 2;
        let cout = 2;
        let h = 3;
        let w_in = 3;
        let kh = 2;
        let kw = 2;
        let cfg = ConvTranspose2dConfig {
            padding: (0, 0),
            output_padding: (0, 0),
            stride: (2, 2),
            dilation: (1, 1),
            groups: 1,
        };

        let weight_vec: Vec<f32> = ramp_f32(cin * cout * kh * kw, 0.03, 0.0);
        let bias_vec: Vec<f32> = ramp_f32(cout, 0.5, -0.2);
        let weight_arc: Arc<[f32]> = Arc::<[f32]>::from(weight_vec);
        let bias_arc: Arc<[f32]> = Arc::<[f32]>::from(bias_vec);

        let layer = ConvTranspose2d::new(
            WeightStorage::F32(Arc::clone(&weight_arc)),
            Some(Arc::clone(&bias_arc)),
            cfg,
            cin,
            cout,
            kh,
            kw,
        )
        .unwrap();

        let x_data: Vec<f32> = ramp_f32(n * cin * h * w_in, 0.02, -0.4);
        let x = LazyTensor::from_f32(
            x_data.clone(),
            Shape::from_dims(&[n, cin, h, w_in]),
            &Device::cpu(),
        );
        let via_module = layer.forward(&x).unwrap().realize_f32();

        let x2 = LazyTensor::from_f32(
            x_data,
            Shape::from_dims(&[n, cin, h, w_in]),
            &Device::cpu(),
        );
        let w_t = x2.const_f32_like(
            Arc::clone(&weight_arc),
            Shape::from_dims(&[cin, cout, kh, kw]),
        );
        let direct_raw = x2
            .conv_transpose2d(
                &w_t,
                cfg.stride,
                cfg.padding,
                cfg.output_padding,
                cfg.dilation,
                cfg.groups,
            )
            .unwrap();
        let b_t = direct_raw
            .const_f32_like(
                Arc::clone(&bias_arc),
                Shape::from_dims(&[cout]),
            )
            .reshape(Shape::from_dims(&[1, cout, 1, 1]))
            .unwrap();
        let direct = direct_raw.broadcast_add(&b_t).unwrap().realize_f32();

        assert_eq!(via_module.len(), direct.len());
        for (i, (a, d)) in via_module.iter().zip(direct.iter()).enumerate() {
            assert!(
                (a - d).abs() < 1e-5,
                "conv_transpose2d strided[{i}] module {a} != direct {d}",
            );
        }
    }

    #[test]
    fn conv_transpose2d_rejects_weight_size_mismatch() {
        let bad_weight: Arc<[f32]> = Arc::<[f32]>::from(vec![0.0_f32; 7]);
        let r = ConvTranspose2d::new(
            WeightStorage::F32(bad_weight),
            None,
            ConvTranspose2dConfig::default(),
            /* in */ 2, /* out */ 3, /* kh */ 2, /* kw */ 2,
        );
        assert!(r.is_err());
    }

    #[test]
    fn conv_transpose1d_no_bias_skips_broadcast() {
        // The no-bias path should produce the same numbers as the raw
        // LazyTensor::conv_transpose1d call.
        let n = 1;
        let cin = 2;
        let cout = 3;
        let l = 4;
        let k = 2;
        let cfg = ConvTranspose1dConfig::default();

        let weight_vec: Vec<f32> = ramp_f32(cin * cout * k, 0.07, 0.1);
        let weight_arc: Arc<[f32]> = Arc::<[f32]>::from(weight_vec);

        let layer = ConvTranspose1d::new(
            WeightStorage::F32(Arc::clone(&weight_arc)),
            None,
            cfg,
            cin,
            cout,
            k,
        )
        .unwrap();
        let x_data: Vec<f32> = ramp_f32(n * cin * l, 0.04, -0.2);
        let x = LazyTensor::from_f32(
            x_data.clone(),
            Shape::from_dims(&[n, cin, l]),
            &Device::cpu(),
        );
        let via_module = layer.forward(&x).unwrap().realize_f32();

        let x2 = LazyTensor::from_f32(
            x_data,
            Shape::from_dims(&[n, cin, l]),
            &Device::cpu(),
        );
        let w_t = x2.const_f32_like(
            Arc::clone(&weight_arc),
            Shape::from_dims(&[cin, cout, k]),
        );
        let direct = x2
            .conv_transpose1d(
                &w_t,
                cfg.stride,
                cfg.padding,
                cfg.output_padding,
                cfg.dilation,
                cfg.groups,
            )
            .unwrap()
            .realize_f32();

        assert_eq!(via_module.len(), direct.len());
        for (i, (a, d)) in via_module.iter().zip(direct.iter()).enumerate() {
            assert!(
                (a - d).abs() < 1e-6,
                "conv_transpose1d no_bias[{i}] module {a} != direct {d}",
            );
        }
    }
}

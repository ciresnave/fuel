//! Convolution Layers.
//!
//! This module provides 1-D and 2-D convolution and transposed-convolution layers:
//!
//! - [`Conv1d`] / [`Conv2d`] -- standard (forward) convolutions.
//! - [`ConvTranspose1d`] / [`ConvTranspose2d`] -- transposed (fractionally-strided) convolutions.
//!
//! Each layer type has an associated config struct ([`Conv1dConfig`], [`Conv2dConfig`], etc.)
//! controlling padding, stride, dilation, and groups.
//!
//! Layers can be constructed directly from weight/bias tensors via `::new`, or loaded from a
//! [`VarBuilder`](crate::VarBuilder) using the free functions [`conv1d`], [`conv2d`], etc.
use crate::BatchNorm;
use candle::{conv::CudnnFwdAlgo, Context, Result, Tensor};

/// Configuration for [`Conv1d`].
///
/// Default: `padding=0`, `stride=1`, `dilation=1`, `groups=1`, no cuDNN algorithm hint.
///
/// # Example
///
/// ```rust
/// use candle_nn::Conv1dConfig;
///
/// let cfg = Conv1dConfig::default();
/// assert_eq!(cfg.padding, 0);
/// assert_eq!(cfg.stride, 1);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1dConfig {
    /// Zero-padding added to both sides of the input. Default: `0`.
    pub padding: usize,
    /// Stride of the convolution. Default: `1`.
    pub stride: usize,
    /// Spacing between kernel elements. Default: `1`.
    pub dilation: usize,
    /// Number of blocked connections from input to output channels. Default: `1`.
    pub groups: usize,
    /// Optional cuDNN forward algorithm selection hint.
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        }
    }
}

impl Conv1dConfig {
    /// Set the zero-padding on both sides of the input.
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Set the convolution stride.
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set the dilation (spacing between kernel elements).
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Set the number of convolution groups (for grouped/depthwise convolutions).
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }
}
///
/// Applies a 1-D convolution over an input tensor of shape `(N, C_in, L)` and produces
/// output of shape `(N, C_out, L_out)`. Implements [`Module`](crate::Module).
///
/// # Example
///
/// ```rust
/// use candle::{Tensor, Device, Module};
/// use candle_nn::{Conv1d, Conv1dConfig};
///
/// let w = Tensor::ones((1, 1, 3), candle::DType::F32, &Device::Cpu)?; // (C_out, C_in, K)
/// let conv = Conv1d::new(w, None, Conv1dConfig { padding: 1, ..Default::default() });
/// let x = Tensor::ones((1, 1, 5), candle::DType::F32, &Device::Cpu)?; // (N, C_in, L)
/// let y = conv.forward(&x)?;
/// assert_eq!(y.dims(), &[1, 1, 5]);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    bias_reshaped: Option<Tensor>,
    config: Conv1dConfig,
}

impl Conv1d {
    /// Creates a new `Conv1d` from a weight tensor of shape `(C_out, C_in / groups, K)` and
    /// an optional bias of shape `(C_out,)`.
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: Conv1dConfig) -> Self {
        let bias_reshaped = bias.as_ref().map(|b| {
            let dim = b.dim(0).unwrap();
            b.reshape((1, dim, 1)).unwrap()
        });
        Self {
            weight,
            bias,
            bias_reshaped,
            config,
        }
    }

    /// Returns the convolution configuration.
    pub fn config(&self) -> &Conv1dConfig {
        &self.config
    }

    /// Returns a reference to the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns a reference to the bias tensor, if present.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl crate::Module for Conv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x
            .conv1d_with_algo(
                &self.weight,
                self.config.padding,
                self.config.stride,
                self.config.dilation,
                self.config.groups,
                self.config.cudnn_fwd_algo,
            )
            .with_context(|| {
                format!(
                    "Conv1d(in={}, out={}, kernel={}): input shape {:?}",
                    self.weight.dim(1).unwrap_or(0) * self.config.groups,
                    self.weight.dim(0).unwrap_or(0),
                    self.weight.dim(2).unwrap_or(0),
                    x.shape()
                )
            })?;
        match &self.bias_reshaped {
            None => Ok(x),
            Some(bias) => Ok(x.broadcast_add(bias)?),
        }
    }
}

/// Configuration for [`ConvTranspose1d`].
///
/// Default: `padding=0`, `output_padding=0`, `stride=1`, `dilation=1`, `groups=1`.
///
/// # Example
///
/// ```rust
/// use candle_nn::ConvTranspose1dConfig;
///
/// let cfg = ConvTranspose1dConfig::default();
/// assert_eq!(cfg.stride, 1);
/// assert_eq!(cfg.output_padding, 0);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose1dConfig {
    /// Zero-padding added to both sides of the input. Default: `0`.
    pub padding: usize,
    /// Additional size added to the output shape. Default: `0`.
    pub output_padding: usize,
    /// Stride of the convolution. Default: `1`.
    pub stride: usize,
    /// Spacing between kernel elements. Default: `1`.
    pub dilation: usize,
    /// Number of blocked connections from input to output channels. Default: `1`.
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

/// A 1-D transposed convolution (deconvolution) layer.
///
/// Applies a transposed 1-D convolution over an input of shape `(N, C_in, L)`.
/// Implements [`Module`](crate::Module).
///
/// # Example
///
/// ```no_run
/// use candle_nn::{conv_transpose1d, ConvTranspose1dConfig, VarBuilder};
///
/// // let layer = conv_transpose1d(8, 4, 3, ConvTranspose1dConfig::default(), vb)?;
/// ```
#[derive(Clone, Debug)]
pub struct ConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvTranspose1dConfig,
}

impl ConvTranspose1d {
    /// Creates a new `ConvTranspose1d` from a weight tensor of shape
    /// `(C_in, C_out / groups, K)` and an optional bias of shape `(C_out,)`.
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: ConvTranspose1dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    /// Returns the transposed convolution configuration.
    pub fn config(&self) -> &ConvTranspose1dConfig {
        &self.config
    }

    /// Returns a reference to the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns a reference to the bias tensor, if present.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl crate::Module for ConvTranspose1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_shape = x.shape().clone();
        let x = x.conv_transpose1d(
            &self.weight,
            self.config.padding,
            self.config.output_padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
        )
        .with_context(|| {
            format!(
                "ConvTranspose1d(in={}, out={}, kernel={}): input shape {input_shape:?}",
                self.weight.dim(0).unwrap_or(0),
                self.weight.dim(1).unwrap_or(0) * self.config.groups,
                self.weight.dim(2).unwrap_or(0),
            )
        })?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

/// Configuration for [`Conv2d`].
///
/// Default: `padding=0`, `stride=1`, `dilation=1`, `groups=1`, no cuDNN algorithm hint.
///
/// # Example
///
/// ```rust
/// use candle_nn::Conv2dConfig;
///
/// let cfg = Conv2dConfig { padding: 1, stride: 2, ..Default::default() };
/// assert_eq!(cfg.padding, 1);
/// assert_eq!(cfg.stride, 2);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dConfig {
    /// Zero-padding added to both sides of the input in each spatial dimension. Default: `0`.
    pub padding: usize,
    /// Stride of the convolution. Default: `1`.
    pub stride: usize,
    /// Spacing between kernel elements. Default: `1`.
    pub dilation: usize,
    /// Number of blocked connections from input to output channels. Default: `1`.
    pub groups: usize,
    /// Optional cuDNN forward algorithm selection hint.
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        }
    }
}

impl Conv2dConfig {
    /// Set the zero-padding on both sides of the input in each spatial dimension.
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Set the convolution stride.
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set the dilation (spacing between kernel elements).
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Set the number of convolution groups.
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }
}

/// A 2-D convolution layer.
///
/// Applies a 2-D convolution over an input tensor of shape `(N, C_in, H, W)` and produces
/// output of shape `(N, C_out, H_out, W_out)`. Implements [`Module`](crate::Module).
///
/// # Example
///
/// ```rust
/// use candle::{Tensor, Device, Module};
/// use candle_nn::{Conv2d, Conv2dConfig};
///
/// let w = Tensor::ones((1, 1, 3, 3), candle::DType::F32, &Device::Cpu)?;
/// let conv = Conv2d::new(w, None, Conv2dConfig { padding: 1, ..Default::default() });
/// let x = Tensor::ones((1, 1, 4, 4), candle::DType::F32, &Device::Cpu)?;
/// let y = conv.forward(&x)?;
/// assert_eq!(y.dims(), &[1, 1, 4, 4]);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv2dConfig,
}

impl Conv2d {
    /// Creates a new `Conv2d` from a weight tensor of shape `(C_out, C_in / groups, kH, kW)`
    /// and an optional bias of shape `(C_out,)`.
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: Conv2dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    /// Returns the convolution configuration.
    pub fn config(&self) -> &Conv2dConfig {
        &self.config
    }

    /// Returns a reference to the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns a reference to the bias tensor, if present.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Fuses a [`BatchNorm`] layer into this convolution, returning a new `Conv2d` whose
    /// weight and bias absorb the batch-norm scaling and shifting. This is useful for
    /// inference optimization.
    pub fn absorb_bn(&self, bn: &BatchNorm) -> Result<Self> {
        if let Some((w_bn, b_bn)) = bn.weight_and_bias() {
            let std_ = w_bn.div(&((bn.running_var() + bn.eps())?.sqrt()?))?;
            let weight = self
                .weight()
                .broadcast_mul(&(std_.reshape((self.weight().dims4()?.0, 1, 1, 1))?))?;
            let bias = match &self.bias {
                None => b_bn.sub(&(std_.mul(bn.running_mean())?))?,
                Some(bias) => b_bn.add(&(std_.mul(&bias.sub(bn.running_mean())?)?))?,
            };
            Ok(Self {
                weight,
                bias: Some(bias),
                config: self.config,
            })
        } else {
            candle::bail!("batch norm does not have weight_and_bias")
        }
    }
}

impl crate::Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x
            .conv2d_with_algo(
                &self.weight,
                self.config.padding,
                self.config.stride,
                self.config.dilation,
                self.config.groups,
                self.config.cudnn_fwd_algo,
            )
            .with_context(|| {
                format!(
                    "Conv2d(in={}, out={}, kernel={}x{}): input shape {:?}",
                    self.weight.dim(1).unwrap_or(0) * self.config.groups,
                    self.weight.dim(0).unwrap_or(0),
                    self.weight.dim(2).unwrap_or(0),
                    self.weight.dim(3).unwrap_or(0),
                    x.shape()
                )
            })?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

/// Configuration for [`ConvTranspose2d`].
///
/// Default: `padding=0`, `output_padding=0`, `stride=1`, `dilation=1`.
///
/// # Example
///
/// ```rust
/// use candle_nn::ConvTranspose2dConfig;
///
/// let cfg = ConvTranspose2dConfig::default();
/// assert_eq!(cfg.stride, 1);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose2dConfig {
    /// Zero-padding added to both sides of the input in each spatial dimension. Default: `0`.
    pub padding: usize,
    /// Additional size added to the output spatial dimensions. Default: `0`.
    pub output_padding: usize,
    /// Stride of the convolution. Default: `1`.
    pub stride: usize,
    /// Spacing between kernel elements. Default: `1`.
    pub dilation: usize,
    // TODO: support groups.
}

impl Default for ConvTranspose2dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            output_padding: 0,
            stride: 1,
            dilation: 1,
        }
    }
}

/// A 2-D transposed convolution (deconvolution) layer.
///
/// Applies a transposed 2-D convolution over an input of shape `(N, C_in, H, W)`.
/// Implements [`Module`](crate::Module).
///
/// # Example
///
/// ```no_run
/// use candle_nn::{conv_transpose2d, ConvTranspose2dConfig, VarBuilder};
///
/// // let layer = conv_transpose2d(8, 4, 3, ConvTranspose2dConfig::default(), vb)?;
/// ```
#[derive(Clone, Debug)]
pub struct ConvTranspose2d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvTranspose2dConfig,
}

impl ConvTranspose2d {
    /// Creates a new `ConvTranspose2d` from a weight tensor of shape
    /// `(C_in, C_out, kH, kW)` and an optional bias of shape `(C_out,)`.
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: ConvTranspose2dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    /// Returns the transposed convolution configuration.
    pub fn config(&self) -> &ConvTranspose2dConfig {
        &self.config
    }

    /// Returns a reference to the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns a reference to the bias tensor, if present.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl crate::Module for ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_shape = x.shape().clone();
        let x = x.conv_transpose2d(
            &self.weight,
            self.config.padding,
            self.config.output_padding,
            self.config.stride,
            self.config.dilation,
        )
        .with_context(|| {
            format!(
                "ConvTranspose2d(in={}, out={}, kernel={}x{}): input shape {input_shape:?}",
                self.weight.dim(0).unwrap_or(0),
                self.weight.dim(1).unwrap_or(0),
                self.weight.dim(2).unwrap_or(0),
                self.weight.dim(3).unwrap_or(0),
            )
        })?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

/// Creates a [`Conv1d`] layer with bias, loading parameters from a [`VarBuilder`](crate::VarBuilder).
///
/// # Example
///
/// ```no_run
/// use candle_nn::{conv1d, Conv1dConfig, VarBuilder};
///
/// // let layer = conv1d(3, 16, 3, Conv1dConfig::default(), vb)?;
/// ```
pub fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv1d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (out_channels, in_channels / cfg.groups, kernel_size),
        "weight",
        init_ws,
    )?;
    let bound = 1. / (in_channels as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_channels, "bias", init_bs)?;
    Ok(Conv1d::new(ws, Some(bs), cfg))
}

/// Creates a [`Conv1d`] layer without bias, loading the weight from a [`VarBuilder`](crate::VarBuilder).
///
/// # Example
///
/// ```no_run
/// use candle_nn::{conv1d_no_bias, Conv1dConfig, VarBuilder};
///
/// // let layer = conv1d_no_bias(3, 16, 3, Conv1dConfig::default(), vb)?;
/// ```
pub fn conv1d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv1d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (out_channels, in_channels / cfg.groups, kernel_size),
        "weight",
        init_ws,
    )?;
    Ok(Conv1d::new(ws, None, cfg))
}

/// Creates a [`ConvTranspose1d`] layer with bias, loading parameters from a [`VarBuilder`](crate::VarBuilder).
///
/// # Example
///
/// ```no_run
/// use candle_nn::{conv_transpose1d, ConvTranspose1dConfig, VarBuilder};
///
/// // let layer = conv_transpose1d(16, 3, 3, ConvTranspose1dConfig::default(), vb)?;
/// ```
pub fn conv_transpose1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose1dConfig,
    vb: crate::VarBuilder,
) -> Result<ConvTranspose1d> {
    let bound = 1. / (out_channels as f64 * kernel_size as f64).sqrt();
    let init = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vb.get_with_hints(
        (in_channels, out_channels / cfg.groups, kernel_size),
        "weight",
        init,
    )?;
    let bs = vb.get_with_hints(out_channels, "bias", init)?;
    Ok(ConvTranspose1d::new(ws, Some(bs), cfg))
}

/// Creates a [`ConvTranspose1d`] layer without bias, loading the weight from a [`VarBuilder`](crate::VarBuilder).
///
/// # Example
///
/// ```no_run
/// use candle_nn::{conv_transpose1d_no_bias, ConvTranspose1dConfig, VarBuilder};
///
/// // let layer = conv_transpose1d_no_bias(16, 3, 3, ConvTranspose1dConfig::default(), vb)?;
/// ```
pub fn conv_transpose1d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose1dConfig,
    vb: crate::VarBuilder,
) -> Result<ConvTranspose1d> {
    let bound = 1. / (out_channels as f64 * kernel_size as f64).sqrt();
    let init = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vb.get_with_hints(
        (in_channels, out_channels / cfg.groups, kernel_size),
        "weight",
        init,
    )?;
    Ok(ConvTranspose1d::new(ws, None, cfg))
}

/// Creates a [`Conv2d`] layer with bias, loading parameters from a [`VarBuilder`](crate::VarBuilder).
///
/// # Example
///
/// ```no_run
/// use candle_nn::{conv2d, Conv2dConfig, VarBuilder};
///
/// // let layer = conv2d(3, 16, 3, Conv2dConfig::default(), vb)?;
/// ```
pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv2d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
        init_ws,
    )?;
    let bound = 1. / (in_channels as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_channels, "bias", init_bs)?;
    Ok(Conv2d::new(ws, Some(bs), cfg))
}

/// Creates a [`Conv2d`] layer without bias, loading the weight from a [`VarBuilder`](crate::VarBuilder).
///
/// # Example
///
/// ```no_run
/// use candle_nn::{conv2d_no_bias, Conv2dConfig, VarBuilder};
///
/// // let layer = conv2d_no_bias(3, 16, 3, Conv2dConfig::default(), vb)?;
/// ```
pub fn conv2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv2d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
        init_ws,
    )?;
    Ok(Conv2d::new(ws, None, cfg))
}

/// Creates a [`ConvTranspose2d`] layer with bias, loading parameters from a [`VarBuilder`](crate::VarBuilder).
///
/// # Example
///
/// ```no_run
/// use candle_nn::{conv_transpose2d, ConvTranspose2dConfig, VarBuilder};
///
/// // let layer = conv_transpose2d(16, 3, 3, ConvTranspose2dConfig::default(), vb)?;
/// ```
pub fn conv_transpose2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose2dConfig,
    vb: crate::VarBuilder,
) -> Result<ConvTranspose2d> {
    let bound = 1. / (out_channels as f64).sqrt() / kernel_size as f64;
    let init = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vb.get_with_hints(
        (in_channels, out_channels, kernel_size, kernel_size),
        "weight",
        init,
    )?;
    let bs = vb.get_with_hints(out_channels, "bias", init)?;
    Ok(ConvTranspose2d::new(ws, Some(bs), cfg))
}

/// Creates a [`ConvTranspose2d`] layer without bias, loading the weight from a [`VarBuilder`](crate::VarBuilder).
///
/// # Example
///
/// ```no_run
/// use candle_nn::{conv_transpose2d_no_bias, ConvTranspose2dConfig, VarBuilder};
///
/// // let layer = conv_transpose2d_no_bias(16, 3, 3, ConvTranspose2dConfig::default(), vb)?;
/// ```
pub fn conv_transpose2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose2dConfig,
    vb: crate::VarBuilder,
) -> Result<ConvTranspose2d> {
    let bound = 1. / (out_channels as f64).sqrt() / kernel_size as f64;
    let init = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vb.get_with_hints(
        (in_channels, out_channels, kernel_size, kernel_size),
        "weight",
        init,
    )?;
    Ok(ConvTranspose2d::new(ws, None, cfg))
}

//! Layer Normalization.
//!
//! This layer applies Layer Normalization over a mini-batch of inputs as described in [`Layer
//! Normalization`]. The input is expected to have three dimensions: a batch dimension, a length,
//! and a hidden size, the normalization is applied over the last dimension.
//!
//! # Example
//!
//! ```rust
//! use candle::{Tensor, Device::Cpu, test_utils::to_vec3_round};
//! use candle_nn::{LayerNorm, Module};
//! # fn main() -> candle::Result<()> {
//!
//! let w = Tensor::new(&[1f32, 1f32, 1f32], &Cpu)?;
//! let b = Tensor::new(&[0f32, 0f32, 0f32], &Cpu)?;
//! let layer = LayerNorm::new(w, b, 1e-5);
//!
//! let xs = Tensor::new(
//!     &[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]],
//!     &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(
//!     to_vec3_round(&ys, 4)?,
//!     &[[[-1.2247, 0.0,  1.2247],
//!        [-1.2247, 0.0,  1.2247],
//!        [ 1.2247, 0.0, -1.2247]]]);
//! # Ok(()) }
//! ```
//!
//! [`Layer Normalization`]: https://arxiv.org/abs/1607.06450
use candle::{DType, Module, Result, Tensor, D};

/// Configuration for [`LayerNorm`].
///
/// Can be constructed from an `f64` epsilon value via `From<f64>`, or use
/// `Default::default()` for standard settings (`eps=1e-5`, `remove_mean=true`,
/// `affine=true`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerNormConfig {
    /// Small constant added to the variance for numerical stability. Default: `1e-5`.
    pub eps: f64,
    /// Whether to subtract the mean during normalization. Default: `true`.
    /// When set to `false`, this layer behaves as RMS normalization.
    pub remove_mean: bool,
    /// When `true` (the default), a learnable `bias` parameter is created. When `false`,
    /// only the `weight` parameter is used.
    pub affine: bool,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        }
    }
}

impl From<f64> for LayerNormConfig {
    fn from(eps: f64) -> Self {
        Self {
            eps,
            remove_mean: true,
            affine: true,
        }
    }
}

impl LayerNormConfig {
    /// Set the epsilon for numerical stability (default: `1e-5`).
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Disable mean subtraction; equivalent to RMS normalization.
    pub fn no_mean_removal(mut self) -> Self {
        self.remove_mean = false;
        self
    }

    /// Disable the learnable `bias` parameter.
    pub fn no_bias(mut self) -> Self {
        self.affine = false;
        self
    }
}

/// Layer Normalization layer.
///
/// Normalizes the input over the last dimension using the mean and variance. Optionally
/// applies a learnable affine transformation (scale and shift). When `remove_mean` is
/// `false`, this behaves as RMS normalization. Implements [`Module`].
#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    remove_mean: bool,
    eps: f64,
}

impl LayerNorm {
    /// Creates a new `LayerNorm` with both `weight` and `bias`, subtracting the mean.
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: Some(bias),
            remove_mean: true,
            eps,
        }
    }

    /// Creates a new `LayerNorm` with only a `weight` parameter (no bias), subtracting the mean.
    pub fn new_no_bias(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: None,
            remove_mean: true,
            eps,
        }
    }

    /// Creates a `LayerNorm` that acts as RMS normalization (no mean removal, no bias).
    pub fn rms_norm(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: None,
            remove_mean: false,
            eps,
        }
    }

    /// Returns a reference to the weight (scale) tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns a reference to the bias tensor, if present.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Returns the epsilon value used for numerical stability.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Returns whether mean removal is enabled (`true` for LayerNorm, `false` for RmsNorm).
    pub fn remove_mean(&self) -> bool {
        self.remove_mean
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use candle::Context;
        let norm_size = self.weight.dim(0).unwrap_or(0);
        let input_shape = x.shape().clone();
        if x.is_contiguous() && self.remove_mean {
            if let Some(bias) = self.bias.as_ref() {
                return crate::ops::layer_norm(x, &self.weight, bias, self.eps as f32)
                    .with_context(|| {
                        format!("LayerNorm(size={norm_size}): input shape {input_shape:?}")
                    });
            }
        }
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let x = if self.remove_mean {
            let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
        .with_context(|| format!("LayerNorm(size={norm_size}): input shape {input_shape:?}"))
    }
}

/// Creates a [`LayerNorm`] layer by loading parameters from a [`VarBuilder`](crate::VarBuilder).
///
/// Loads `weight` and (when `affine=true`) `bias` from the variable store.
pub fn layer_norm<C: Into<LayerNormConfig>>(
    size: usize,
    config: C,
    vb: crate::VarBuilder,
) -> Result<LayerNorm> {
    let config = config.into();
    let weight = vb.get_with_hints(size, "weight", crate::Init::Const(1.))?;
    let bias = if config.affine {
        Some(vb.get_with_hints(size, "bias", crate::Init::Const(0.))?)
    } else {
        None
    };
    Ok(LayerNorm {
        weight,
        bias,
        remove_mean: config.remove_mean,
        eps: config.eps,
    })
}

/// Creates a [`LayerNorm`] layer without a bias parameter via a [`VarBuilder`](crate::VarBuilder).
pub fn layer_norm_no_bias(size: usize, eps: f64, vb: crate::VarBuilder) -> Result<LayerNorm> {
    let config = LayerNormConfig {
        eps,
        remove_mean: true,
        affine: false,
    };
    layer_norm(size, config, vb)
}

/// RmsNorm is a specialized version of the LayerNorm module.
#[derive(Clone, Debug)]
pub struct RmsNorm(LayerNorm);

impl RmsNorm {
    /// Creates a new `RmsNorm` layer with the given weight and epsilon.
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self(LayerNorm::rms_norm(weight, eps))
    }

    /// Unwraps the inner [`LayerNorm`].
    pub fn into_inner(self) -> LayerNorm {
        self.0
    }

    /// Returns a reference to the weight (scale) tensor.
    pub fn weight(&self) -> &Tensor {
        self.0.weight()
    }

    /// Returns the epsilon value used for numerical stability.
    pub fn eps(&self) -> f64 {
        self.0.eps()
    }

    /// Faster variant of the forward kernel, this can only be used on contiguous tensors though.
    pub fn forward_diff(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.is_contiguous() {
            crate::ops::rms_norm(xs, &self.0.weight, self.0.eps as f32)
        } else {
            self.0.forward(xs)
        }
    }
}

/// Creates an [`RmsNorm`] layer by loading the weight from a [`VarBuilder`](crate::VarBuilder).
pub fn rms_norm(size: usize, eps: f64, vb: crate::VarBuilder) -> Result<RmsNorm> {
    let config = LayerNormConfig {
        eps,
        remove_mean: false,
        affine: false,
    };
    Ok(RmsNorm(layer_norm(size, config, vb)?))
}

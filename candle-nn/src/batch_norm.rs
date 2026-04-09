//! Batch Normalization.
//!
//! This layer applies [Batch Normalization](https://arxiv.org/abs/1502.03167) over a mini-batch
//! of inputs. The input tensor is expected to have at least three dimensions `(N, C, ...)` where
//! `N` is the batch size and `C` is the number of channels (features). Normalization is performed
//! per-channel across the batch and spatial dimensions.
//!
//! During training ([`BatchNorm::forward_train`]), the layer computes batch statistics and updates
//! exponential moving averages stored in `running_mean` and `running_var`. During evaluation
//! (`forward_t` with `train=false`), it uses the stored running statistics for
//! normalization.
//!
//! When `affine` is enabled (the default), learnable scale (`weight`) and shift (`bias`)
//! parameters are applied after normalization.
//!
//! Use [`batch_norm`] to construct a `BatchNorm` from a [`VarBuilder`](crate::VarBuilder), or
//! use [`BatchNorm::new`] to construct one directly from tensors.
use candle::{Context, DType, Result, Tensor, Var};

/// Configuration for [`BatchNorm`].
///
/// Can be constructed from an `f64` epsilon value via the `From<f64>` implementation,
/// or use `Default::default()` for standard settings (`eps=1e-5`, `affine=true`,
/// `remove_mean=true`, `momentum=0.1`).
///
/// # Example
///
/// ```rust
/// use candle_nn::BatchNormConfig;
///
/// let cfg = BatchNormConfig::default();
/// assert_eq!(cfg.eps, 1e-5);
/// assert!(cfg.affine);
/// assert_eq!(cfg.momentum, 0.1);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchNormConfig {
    /// Small constant added to the variance for numerical stability. Default: `1e-5`.
    pub eps: f64,
    /// Whether to subtract the mean during normalization. Default: `true`.
    pub remove_mean: bool,

    /// When `true`, learnable `weight` and `bias` parameters are used. When `false`, no
    /// learnable parameters are created (gamma is fixed to 1, beta to 0). Note that this
    /// differs from [`LayerNormConfig::affine`](crate::LayerNormConfig::affine) where
    /// `false` still creates a weight but omits the bias.
    pub affine: bool,

    /// Controls exponential moving average of running stats. Default: `0.1`.
    ///
    /// Updated as: `running_stat * (1.0 - momentum) + batch_stat * momentum`.
    pub momentum: f64,
}

impl Default for BatchNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        }
    }
}

impl From<f64> for BatchNormConfig {
    fn from(eps: f64) -> Self {
        Self {
            eps,
            ..Default::default()
        }
    }
}

impl BatchNormConfig {
    /// Set the epsilon for numerical stability (default: `1e-5`).
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Disable mean subtraction during normalization.
    pub fn no_mean_removal(mut self) -> Self {
        self.remove_mean = false;
        self
    }

    /// Disable learnable affine parameters (`weight` and `bias`).
    pub fn no_affine(mut self) -> Self {
        self.affine = false;
        self
    }

    /// Set the momentum for the running statistics update (default: `0.1`).
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }
}

/// Batch Normalization layer.
///
/// Normalizes each channel across the batch and spatial dimensions. Maintains running
/// statistics (`running_mean`, `running_var`) that are updated during training and used
/// for normalization during evaluation. Implements [`ModuleT`](crate::ModuleT) so that
/// the caller can select training vs. evaluation behavior.
///
/// # Example
///
/// ```no_run
/// use candle_nn::{batch_norm, BatchNormConfig, VarBuilder};
///
/// // Typically constructed via VarBuilder:
/// // let bn = batch_norm(16, BatchNormConfig::default(), vb.pp("bn"))?;
/// // let out = candle_nn::ModuleT::forward_t(&bn, &input, true)?; // training mode
/// ```
#[derive(Clone, Debug)]
pub struct BatchNorm {
    running_mean: Var,
    running_var: Var,
    weight_and_bias: Option<(Tensor, Tensor)>,
    remove_mean: bool,
    eps: f64,
    momentum: f64,
}

impl BatchNorm {
    fn check_validity(&self, num_features: usize) -> Result<()> {
        if self.eps < 0. {
            candle::bail!("batch-norm eps cannot be negative {}", self.eps)
        }
        if !(0.0..=1.0).contains(&self.momentum) {
            candle::bail!(
                "batch-norm momentum must be between 0 and 1, is {}",
                self.momentum
            )
        }
        if self.running_mean.dims() != [num_features] {
            candle::bail!(
                "batch-norm running mean has unexpected shape {:?} should have shape [{num_features}]",
                self.running_mean.shape(),
            )
        }
        if self.running_var.dims() != [num_features] {
            candle::bail!(
                "batch-norm running variance has unexpected shape {:?} should have shape [{num_features}]",
                self.running_var.shape(),
            )
        }
        if let Some((weight, bias)) = self.weight_and_bias.as_ref() {
            if weight.dims() != [num_features] {
                candle::bail!(
                    "batch-norm weight has unexpected shape {:?} should have shape [{num_features}]",
                    weight.shape(),
                )
            }
            if bias.dims() != [num_features] {
                candle::bail!(
                    "batch-norm weight has unexpected shape {:?} should have shape [{num_features}]",
                    bias.shape(),
                )
            }
        }
        Ok(())
    }

    /// Creates a new `BatchNorm` layer with learnable `weight` and `bias`.
    ///
    /// All tensors must have shape `[num_features]`. Uses default momentum of `0.1`.
    pub fn new(
        num_features: usize,
        running_mean: Tensor,
        running_var: Tensor,
        weight: Tensor,
        bias: Tensor,
        eps: f64,
    ) -> Result<Self> {
        let out = Self {
            running_mean: Var::from_tensor(&running_mean)?,
            running_var: Var::from_tensor(&running_var)?,
            weight_and_bias: Some((weight, bias)),
            remove_mean: true,
            eps,
            momentum: 0.1,
        };
        out.check_validity(num_features)?;
        Ok(out)
    }

    /// Creates a new `BatchNorm` layer without learnable `weight`/`bias` parameters.
    ///
    /// The running statistics tensors must have shape `[num_features]`.
    pub fn new_no_bias(
        num_features: usize,
        running_mean: Tensor,
        running_var: Tensor,
        eps: f64,
    ) -> Result<Self> {
        let out = Self {
            running_mean: Var::from_tensor(&running_mean)?,
            running_var: Var::from_tensor(&running_var)?,
            weight_and_bias: None,
            remove_mean: true,
            eps,
            momentum: 0.1,
        };
        out.check_validity(num_features)?;
        Ok(out)
    }

    /// Creates a new `BatchNorm` layer with learnable parameters and a custom `momentum`.
    pub fn new_with_momentum(
        num_features: usize,
        running_mean: Tensor,
        running_var: Tensor,
        weight: Tensor,
        bias: Tensor,
        eps: f64,
        momentum: f64,
    ) -> Result<Self> {
        let out = Self {
            running_mean: Var::from_tensor(&running_mean)?,
            running_var: Var::from_tensor(&running_var)?,
            weight_and_bias: Some((weight, bias)),
            remove_mean: true,
            eps,
            momentum,
        };
        out.check_validity(num_features)?;
        Ok(out)
    }

    /// Creates a new `BatchNorm` layer without learnable parameters and with a custom `momentum`.
    pub fn new_no_bias_with_momentum(
        num_features: usize,
        running_mean: Tensor,
        running_var: Tensor,
        eps: f64,
        momentum: f64,
    ) -> Result<Self> {
        let out = Self {
            running_mean: Var::from_tensor(&running_mean)?,
            running_var: Var::from_tensor(&running_var)?,
            weight_and_bias: None,
            remove_mean: true,
            eps,
            momentum,
        };
        out.check_validity(num_features)?;
        Ok(out)
    }

    /// Returns a reference to the running mean tensor.
    pub fn running_mean(&self) -> &Tensor {
        self.running_mean.as_tensor()
    }

    /// Returns a reference to the running variance tensor.
    pub fn running_var(&self) -> &Tensor {
        self.running_var.as_tensor()
    }

    /// Returns the epsilon value used for numerical stability.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Returns the learnable weight and bias tensors, if present.
    pub fn weight_and_bias(&self) -> Option<(&Tensor, &Tensor)> {
        self.weight_and_bias.as_ref().map(|v| (&v.0, &v.1))
    }

    /// Returns the momentum value for running statistics updates.
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// Runs the forward pass in training mode, computing batch statistics and updating
    /// the running mean and variance via exponential moving average.
    pub fn forward_train(&self, x: &Tensor) -> Result<Tensor> {
        let num_features = self.running_mean.as_tensor().dim(0)?;
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        if x.rank() < 2 {
            candle::bail!(
                "batch-norm input tensor must have at least two dimensions ({:?})",
                x.shape()
            )
        }
        if x.dim(1)? != num_features {
            candle::bail!(
                "batch-norm input doesn't have the expected number of features ({:?} <> {})",
                x.shape(),
                num_features
            )
        }
        let x = x.to_dtype(internal_dtype)?;
        let x = x.transpose(0, 1)?;
        let x_dims_post_transpose = x.dims();
        // Flatten all the dimensions exception the channel one as this performs a Spatial Batch
        // Normalization.
        let x = x.flatten_from(1)?.contiguous()?;
        let x = if self.remove_mean {
            // The mean is taken over dim 1 as this is the batch dim after the transpose(0, 1) above.
            let mean_x = x.mean_keepdim(1)?;
            let updated_running_mean = ((self.running_mean.as_tensor() * (1.0 - self.momentum))?
                + (mean_x.flatten_all()? * self.momentum)?)?;
            self.running_mean.set(&updated_running_mean)?;
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };
        // The mean is taken over dim 1 as this is the batch dim after the transpose(0, 1) above.
        let norm_x = x.sqr()?.mean_keepdim(1)?;
        let updated_running_var = {
            let batch_size = x.dim(1)? as f64;
            let running_var_weight = 1.0 - self.momentum;
            let norm_x_weight = self.momentum * batch_size / (batch_size - 1.0);
            ((self.running_var.as_tensor() * running_var_weight)?
                + (&norm_x.flatten_all()? * norm_x_weight)?)?
        };
        self.running_var.set(&updated_running_var)?;
        let x = x
            .broadcast_div(&(norm_x + self.eps)?.sqrt()?)?
            .to_dtype(x_dtype)?;
        let x = match &self.weight_and_bias {
            None => x,
            Some((weight, bias)) => {
                let weight = weight.reshape(((), 1))?;
                let bias = bias.reshape(((), 1))?;
                x.broadcast_mul(&weight)?.broadcast_add(&bias)?
            }
        };
        x.reshape(x_dims_post_transpose)?.transpose(0, 1)
    }

    fn forward_eval(&self, x: &Tensor) -> Result<Tensor> {
        let target_shape: Vec<usize> = x
            .dims()
            .iter()
            .enumerate()
            .map(|(idx, v)| if idx == 1 { *v } else { 1 })
            .collect();
        let target_shape = target_shape.as_slice();

        let mean = self.running_mean.as_detached_tensor().reshape(target_shape)?;
        let std = (self
            .running_var
            .as_detached_tensor()
            .reshape(target_shape)?
            + self.eps)?
            .sqrt()?;

        match &self.weight_and_bias {
            None => x.broadcast_sub(&mean)?.broadcast_div(&std),
            Some((weight, bias)) => {
                // Pre-compute combined scale and offset so we only need 2 passes
                // over x instead of 4:
                //   y = weight/std * x + (bias - mean * weight/std)
                //     = scale * x + offset
                let weight = weight.reshape(target_shape)?;
                let bias = bias.reshape(target_shape)?;
                let scale = (&weight / &std)?;
                let offset = (&bias - (&mean * &scale)?)?;
                x.broadcast_mul(&scale)?.broadcast_add(&offset)
            }
        }
    }
}

impl crate::ModuleT for BatchNorm {
    fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let input_shape = x.shape().clone();
        let num_features = self.running_mean.as_tensor().dim(0).unwrap_or(0);
        let result = if train {
            self.forward_train(x)
        } else {
            self.forward_eval(x)
        };
        result.with_context(|| {
            format!(
                "BatchNorm(num_features={num_features}): input shape {input_shape:?}"
            )
        })
    }
}

/// Creates a [`BatchNorm`] layer by loading parameters from a [`VarBuilder`](crate::VarBuilder).
///
/// Loads `running_mean`, `running_var`, and (when `affine=true`) `weight` and `bias`
/// tensors from the variable store.
///
/// # Example
///
/// ```no_run
/// use candle_nn::{batch_norm, BatchNormConfig, VarBuilder};
/// use candle::DType;
/// // vb: VarBuilder with "running_mean", "running_var", "weight", "bias" tensors
/// # let vb: VarBuilder = unimplemented!();
/// let bn = batch_norm(64, BatchNormConfig::default(), vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
pub fn batch_norm<C: Into<BatchNormConfig>>(
    num_features: usize,
    config: C,
    vb: crate::VarBuilder,
) -> Result<BatchNorm> {
    use crate::Init;
    let config = config.into();
    if config.eps < 0. {
        candle::bail!("batch-norm eps cannot be negative {}", config.eps)
    }
    let running_mean = vb.get_with_hints(num_features, "running_mean", Init::Const(0.))?;
    let running_var = vb.get_with_hints(num_features, "running_var", Init::Const(1.))?;
    let weight_and_bias = if config.affine {
        let weight = vb.get_with_hints(num_features, "weight", Init::Const(1.))?;
        let bias = vb.get_with_hints(num_features, "bias", Init::Const(0.))?;
        Some((weight, bias))
    } else {
        None
    };
    Ok(BatchNorm {
        running_mean: Var::from_tensor(&running_mean)?,
        running_var: Var::from_tensor(&running_var)?,
        weight_and_bias,
        remove_mean: config.remove_mean,
        eps: config.eps,
        momentum: config.momentum,
    })
}

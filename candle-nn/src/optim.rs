//! Optimization algorithms for training neural networks.
//!
//! This module provides the [`Optimizer`] trait and concrete implementations including
//! [`SGD`] (stochastic gradient descent) and [`AdamW`] (Adam with decoupled weight decay).
//!
//! Optimizers operate on [`Var`] tensors and update them in-place based on
//! computed gradients. The typical training loop calls [`Optimizer::backward_step`] which
//! computes gradients via backpropagation and then applies the parameter update.
use candle::{Result, Tensor, Var};

/// The interface optimizers should implement.
///
/// Provides a common API for creating optimizers from a set of [`Var`] parameters,
/// performing gradient-based updates, and adjusting the learning rate.
///
/// # Required Methods
///
/// - [`new`](Optimizer::new) - Create an optimizer for the given variables and config.
/// - [`step`](Optimizer::step) - Apply one optimization step using precomputed gradients.
/// - [`learning_rate`](Optimizer::learning_rate) - Return the current learning rate.
/// - [`set_learning_rate`](Optimizer::set_learning_rate) - Update the learning rate.
///
/// # Provided Methods
///
/// - [`backward_step`](Optimizer::backward_step) - Compute gradients from a loss tensor
///   and apply one step (convenience wrapper around `backward` + `step`).
/// - [`empty`](Optimizer::empty) - Create an optimizer with no variables.
/// - [`from_slice`](Optimizer::from_slice) - Create an optimizer from a slice of `&Var`.
///
/// # Example
///
/// ```no_run
/// use candle::{Tensor, Var, Device, DType};
/// use candle_nn::{SGD, Optimizer};
///
/// let p = Var::from_tensor(&Tensor::ones(4, DType::F32, &Device::Cpu)?)?;
/// let loss = p.as_tensor().mean_all()?;
/// let mut opt = SGD::new(vec![p], 0.01)?;
/// opt.backward_step(&loss)?;
/// # Ok::<(), candle::Error>(())
/// ```
pub trait Optimizer: Sized {
    type Config: Sized;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self>;

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()>;

    fn learning_rate(&self) -> f64;

    fn set_learning_rate(&mut self, lr: f64);

    fn empty(config: Self::Config) -> Result<Self> {
        Self::new(vec![], config)
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    fn from_slice(vars: &[&Var], config: Self::Config) -> Result<Self> {
        let vars: Vec<_> = vars.iter().map(|&v| v.clone()).collect();
        Self::new(vars, config)
    }
}

/// Optimizer for Stochastic Gradient Descent.
///
/// Applies the update rule `theta = theta - lr * grad` to each parameter. Contrary to the
/// PyTorch implementation of SGD, this version does not support momentum.
///
/// The `Config` type is `f64`, representing the learning rate.
///
/// Non-float parameters are automatically filtered out during construction.
///
/// # Example
///
/// ```no_run
/// use candle::{Tensor, Var, Device, DType};
/// use candle_nn::{SGD, Optimizer};
///
/// let p = Var::from_tensor(&Tensor::ones(4, DType::F32, &Device::Cpu)?)?;
/// let mut opt = SGD::new(vec![p.clone()], 0.1)?;
/// let loss = p.as_tensor().mean_all()?;
/// opt.backward_step(&loss)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug)]
pub struct SGD {
    vars: Vec<Var>,
    learning_rate: f64,
}

impl Optimizer for SGD {
    type Config = f64;

    fn new(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        Ok(Self {
            vars,
            learning_rate,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        for var in self.vars.iter() {
            if let Some(grad) = grads.get(var) {
                var.set(&var.sub(&(grad * self.learning_rate)?)?)?;
            }
        }
        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr
    }
}

impl SGD {
    /// Consume the optimizer and return the owned list of variables.
    pub fn into_inner(self) -> Vec<Var> {
        self.vars
    }

    /// Add an additional variable to be optimized.
    pub fn push(&mut self, var: &Var) {
        self.vars.push(var.clone())
    }
}

/// Configuration parameters for the [`AdamW`] optimizer.
///
/// # Defaults
///
/// | Parameter      | Default |
/// |----------------|----------|
/// | `lr`           | 0.001   |
/// | `beta1`        | 0.9     |
/// | `beta2`        | 0.999   |
/// | `eps`          | 1e-8    |
/// | `weight_decay` | 0.01    |
///
/// # Example
///
/// ```rust
/// use candle_nn::ParamsAdamW;
///
/// let p = ParamsAdamW::default();
/// assert_eq!(p.lr, 0.001);
/// assert_eq!(p.weight_decay, 0.01);
/// # Ok::<(), candle::Error>(())
/// ```
/// Type alias for [`ParamsAdamW`] following the `<Optimizer>Config` naming convention
/// used by other configuration structs in this crate (e.g., `LayerNormConfig`, `GRUConfig`).
pub type AdamWConfig = ParamsAdamW;

#[derive(Clone, Debug)]
pub struct ParamsAdamW {
    /// Learning rate.
    pub lr: f64,
    /// Exponential decay rate for the first moment estimates.
    pub beta1: f64,
    /// Exponential decay rate for the second moment estimates.
    pub beta2: f64,
    /// Small constant for numerical stability in the denominator.
    pub eps: f64,
    /// Decoupled weight decay coefficient.
    pub weight_decay: f64,
}

impl Default for ParamsAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

impl ParamsAdamW {
    /// Set the learning rate.
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set the first moment decay coefficient (default: `0.9`).
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set the second moment decay coefficient (default: `0.999`).
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set the numerical stability epsilon (default: `1e-8`).
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the decoupled weight decay coefficient (default: `0.01`).
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

#[derive(Debug)]
struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

/// Adam optimizer with decoupled weight decay (AdamW).
///
/// Implements the algorithm from [Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101).
/// Maintains per-parameter first and second moment estimates and applies bias correction.
/// Weight decay is applied directly to the parameters rather than to the gradients.
///
/// Non-float parameters are automatically filtered out during construction.
///
/// # Example
///
/// ```no_run
/// use candle::{Tensor, Var, Device, DType};
/// use candle_nn::{AdamW, ParamsAdamW, Optimizer};
///
/// let p = Var::from_tensor(&Tensor::ones(4, DType::F32, &Device::Cpu)?)?;
/// let params = ParamsAdamW { lr: 1e-3, ..Default::default() };
/// let mut opt = AdamW::new(vec![p.clone()], params)?;
/// let loss = p.as_tensor().mean_all()?;
/// opt.backward_step(&loss)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug)]
pub struct AdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

impl Optimizer for AdamW {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        // Fuse the first-moment bias correction (scale_m) with the learning rate into a
        // single scalar. This eliminates a separate m_hat intermediate tensor and a
        // subsequent multiply-by-lr tensor, saving two tensor allocations per parameter.
        let lr_scale_m = lr * scale_m;
        // Factor the second-moment bias correction out of v_hat so that
        // sqrt(v / (1 - beta2^t)) + eps can be computed as
        // sqrt(v) * sqrt(scale_v) + eps in a single affine() call, avoiding
        // a separate v_hat tensor allocation.
        let sqrt_scale_v = scale_v.sqrt();
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                // This involves locking 3 RWLocks per params, if the parameters are large this
                // should not be an issue but this may be problematic with models with lots of
                // small parameters.
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                // Denominator: sqrt(next_v) * sqrt(scale_v) + eps via a single affine(),
                // mathematically equivalent to sqrt(next_v * scale_v) + eps.
                let denom = next_v.sqrt()?.affine(sqrt_scale_v, self.params.eps)?;
                // Numerator with fused lr and bias correction, then divide.
                let update = ((&next_m * lr_scale_m)? / denom)?;
                let next_theta = ((theta.as_tensor() * (1f64 - lr_lambda))? - update)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

impl AdamW {
    /// Create a new AdamW optimizer with a custom learning rate and default parameters.
    ///
    /// This is a convenience constructor equivalent to:
    /// ```ignore
    /// AdamW::new(vars, ParamsAdamW { lr: learning_rate, ..Default::default() })
    /// ```
    pub fn new_lr(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let params = ParamsAdamW {
            lr: learning_rate,
            ..ParamsAdamW::default()
        };
        Self::new(vars, params)
    }

    /// Return a reference to the current optimizer parameters.
    pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    /// Replace the optimizer parameters (learning rate, betas, etc.) for subsequent steps.
    pub fn set_params(&mut self, params: ParamsAdamW) {
        self.params = params;
    }
}

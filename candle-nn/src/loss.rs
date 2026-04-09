//! Loss functions for training neural networks.
//!
//! This module provides common loss functions including cross-entropy, MSE,
//! binary cross-entropy, and Huber loss.
use candle::{Result, Tensor};

/// The negative log likelihood loss.
///
/// Computes the negative log likelihood between log-probabilities and integer class labels.
/// This is typically used after applying [`log_softmax`](crate::ops::log_softmax) to model
/// outputs. For raw logits, use [`cross_entropy`] which applies log-softmax internally.
///
/// # Arguments
///
/// * `inp` - The input tensor of dimensions `[N, C]` where `N` is the batch size and `C` the
///   number of categories. This is expected to contain log probabilities.
/// * `target` - The ground truth labels as a tensor of u32 of dimension `[N]`.
///
/// # Returns
///
/// A scalar tensor containing the average NLL over the batch.
///
/// # Examples
///
/// ```
/// use candle::{Tensor, Device};
/// // Two samples, three classes — log-probabilities (already log-softmaxed)
/// let log_probs = Tensor::new(&[[-0.1054f32, -2.3026, -6.9078],
///                                [-2.3026, -0.1054, -6.9078]], &Device::Cpu)?;
/// let targets = Tensor::new(&[0u32, 1], &Device::Cpu)?;
/// let loss = candle_nn::loss::nll(&log_probs, &targets)?;
/// // Loss should be close to 0.1054 (average of the two correct-class log-probs, negated)
/// assert!(loss.to_scalar::<f32>()? < 0.2);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn nll(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    let b_sz = match target.dims() {
        &[b_sz] => b_sz,
        dims => candle::bail!("the target tensor should have a single dimension ({dims:?})"),
    };
    match inp.dims() {
        &[inp_b_sz, _] => {
            if inp_b_sz != b_sz {
                candle::bail!("batch size mismatch between inp ({inp_b_sz}) and target ({b_sz})")
            }
        }
        dims => candle::bail!("the target tensor should have two dimensions ({dims:?})"),
    }
    inp.gather(&target.unsqueeze(1)?, 1)?
        .sum_all()?
        .affine(-1f64 / b_sz as f64, 0.)
}

/// The negative log likelihood loss (descriptive alias for [`nll`]).
///
/// This is identical to [`nll`] — see its documentation for full details.
///
/// # Example
///
/// ```
/// use candle::{Tensor, Device};
/// let log_probs = Tensor::new(&[[-0.1054f32, -2.3026, -6.9078],
///                                [-2.3026, -0.1054, -6.9078]], &Device::Cpu)?;
/// let targets = Tensor::new(&[0u32, 1], &Device::Cpu)?;
/// let loss = candle_nn::loss::negative_log_likelihood(&log_probs, &targets)?;
/// assert!(loss.to_scalar::<f32>()? < 0.2);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn negative_log_likelihood(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    nll(inp, target)
}

/// The cross-entropy loss.
///
/// Applies [`log_softmax`](crate::ops::log_softmax) to the input logits and then computes the
/// negative log likelihood against integer class labels. This is the standard loss for
/// multi-class classification.
///
/// # Arguments
///
/// * `inp` - The input tensor of dimensions `[N, C]` where `N` is the batch size and `C` the
///   number of categories. This is expected to be raw (unnormalized) logits.
/// * `target` - The ground truth labels as a tensor of u32 of dimension `[N]`.
///
/// # Returns
///
/// A scalar tensor containing the average cross-entropy loss over the batch.
///
/// # Examples
///
/// ```
/// use candle::{Tensor, Device};
/// // Raw logits for 2 samples over 3 classes
/// let logits = Tensor::new(&[[2.0f32, 1.0, 0.1],
///                             [0.5, 2.5, 0.3]], &Device::Cpu)?;
/// let targets = Tensor::new(&[0u32, 1], &Device::Cpu)?;
/// let loss = candle_nn::loss::cross_entropy(&logits, &targets)?;
/// let loss_val = loss.to_scalar::<f32>()?;
/// assert!(loss_val > 0.0);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn cross_entropy(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    if inp.rank() != 2 {
        candle::bail!("cross_entropy expects an input tensor of rank 2")
    }
    let inp = crate::ops::log_softmax(inp, 1)?;
    nll(&inp, target)
}

/// The mean squared error loss.
///
/// Computes the element-wise squared difference between `inp` and `target`, then returns
/// the mean over all elements. Commonly used for regression tasks.
///
/// # Arguments
///
/// * `inp` - Predictions tensor of any shape.
/// * `target` - Ground truth tensor with the same shape as `inp`.
///
/// # Returns
///
/// A scalar tensor containing the mean squared error.
///
/// # Examples
///
/// ```
/// use candle::{Tensor, Device};
/// let predictions = Tensor::new(&[0.5f32, 0.8, 0.1], &Device::Cpu)?;
/// let targets = Tensor::new(&[1.0f32, 1.0, 0.0], &Device::Cpu)?;
/// let loss = candle_nn::loss::mse(&predictions, &targets)?;
/// let loss_val = loss.to_scalar::<f32>()?;
/// // MSE = ((0.5)^2 + (0.2)^2 + (0.1)^2) / 3 = 0.1
/// assert!((loss_val - 0.1).abs() < 1e-6);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn mse(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    (inp - target)?.sqr()?.mean_all()
}

/// The binary cross-entropy with logit loss.
///
/// Computes numerically stable binary cross-entropy from raw logits using the identity:
/// `max(x, 0) - x*t + log(1 + exp(-|x|))`. This avoids overflow when computing `exp(x)`
/// for large positive logits.
///
/// Suitable for multi-label classification where each element is an independent binary
/// prediction.
///
/// # Arguments
///
/// * `inp` - The input tensor containing raw logits (before sigmoid). Can be any shape.
/// * `target` - The ground truth labels (0.0 or 1.0) with the same shape as `inp`.
///
/// # Returns
///
/// A scalar tensor containing the mean binary cross-entropy loss.
///
/// # Examples
///
/// ```
/// use candle::{Tensor, Device};
/// let logits = Tensor::new(&[1.0f32, -1.0, 0.0], &Device::Cpu)?;
/// let targets = Tensor::new(&[1.0f32, 0.0, 1.0], &Device::Cpu)?;
/// let loss = candle_nn::loss::binary_cross_entropy_with_logit(&logits, &targets)?;
/// let loss_val = loss.to_scalar::<f32>()?;
/// assert!(loss_val > 0.0);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn binary_cross_entropy_with_logit(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    // Numerically stable form: max(x,0) - x*t + log(1 + exp(-|x|))
    let relu_inp = inp.relu()?;
    let neg_abs_inp = inp.abs()?.neg()?;
    let loss = ((&relu_inp - inp.mul(target)?)? + (neg_abs_inp.exp()? + 1.0)?.log()?)?;
    loss.mean_all()
}

/// Huber loss (smooth L1 loss).
///
/// A robust loss function that combines MAE and MSE losses, making it less sensitive to
/// outliers than pure MSE:
///
/// - When `|x - y| < delta`: uses the squared term `0.5 * (x - y)^2` (like MSE).
/// - When `|x - y| >= delta`: uses the linear term `delta * (|x - y| - 0.5 * delta)` (like MAE).
///
/// # Arguments
///
/// * `inp` - Predictions tensor of any shape.
/// * `target` - Ground truth tensor with the same shape as `inp`.
/// * `delta` - Threshold at which the loss transitions from quadratic to linear.
///
/// # Returns
///
/// A scalar tensor containing the mean Huber loss.
///
/// # Examples
///
/// ```
/// use candle::{Tensor, Device};
/// let predictions = Tensor::new(&[0.5f32, 3.0], &Device::Cpu)?;
/// let targets = Tensor::new(&[1.0f32, 1.0], &Device::Cpu)?;
/// let loss = candle_nn::loss::huber(&predictions, &targets, 1.0)?;
/// let loss_val = loss.to_scalar::<f32>()?;
/// // First element: |0.5-1.0|=0.5 < 1.0 => 0.5*0.25 = 0.125
/// // Second element: |3.0-1.0|=2.0 >= 1.0 => 1.0*(2.0-0.5) = 1.5
/// // Mean = (0.125 + 1.5) / 2 = 0.8125
/// assert!((loss_val - 0.8125).abs() < 1e-4);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn huber(inp: &Tensor, target: &Tensor, delta: f64) -> Result<Tensor> {
    if inp.dims() != target.dims() {
        candle::bail!(
            "input and target must have the same shape, got inp: {:?}, target: {:?}",
            inp.dims(),
            target.dims()
        );
    }
    let diff = (inp - target)?;
    let abs_diff = diff.abs()?;
    let mask = abs_diff.le(delta)?;
    let squared_loss = ((&diff * &diff)? * 0.5)?;
    let linear_loss = ((abs_diff * delta)? - 0.5 * delta.powi(2))?;
    let loss = mask.where_cond(&squared_loss, &linear_loss)?;
    loss.mean_all()
}

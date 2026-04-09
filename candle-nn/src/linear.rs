//! Linear layer
//!
//! This layer applies a linear transformation to the incoming data, `y = x@w.t() + b`.
//! The bias is optional. The `forward` method can be used to apply the layer, it supports input
//! with a batch dimension (so of shape `(b_sz, in_c)`) or without (of shape `(in_c,)`), the
//! output has shape `(b_sz, out_c)` and `(out_c,)` respectively.
//!
//! ```rust
//! use candle::{Tensor, Device::Cpu};
//! use candle_nn::{Linear, Module};
//! # fn main() -> candle::Result<()> {
//!
//! let w = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Cpu)?;
//! let layer = Linear::new(w, None); // Use no bias.
//! let xs = Tensor::new(&[[10f32, 100.]], &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(ys.to_vec2::<f32>()?, &[[210.0, 430.0, 650.0]]);
//! # Ok(()) }
//! ```
use candle::{Context, Result, Tensor};

/// A linear (fully connected) layer that applies `y = x @ weight^T + bias`.
///
/// The weight tensor has shape `(out_features, in_features)`. The optional bias has
/// shape `(out_features,)`. The layer supports batched inputs with 2, 3, or 4
/// dimensions.
///
/// # Example
///
/// ```rust
/// use candle::{Tensor, Device};
/// use candle_nn::{Linear, Module};
///
/// let w = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Device::Cpu)?;
/// let b = Tensor::new(&[0.5f32, 1.0, 1.5], &Device::Cpu)?;
/// let layer = Linear::new(w, Some(b));
/// let x = Tensor::new(&[[10f32, 100.]], &Device::Cpu)?;
/// let y = layer.forward(&x)?;
/// assert_eq!(y.to_vec2::<f32>()?, &[[210.5, 431.0, 651.5]]);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    /// Creates a new linear layer from a weight tensor and an optional bias tensor.
    ///
    /// The weight should have shape `(out_features, in_features)`.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
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

impl super::Module for Linear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let in_features = self.weight.dim(1)?;
        let out_features = self.weight.dim(0)?;
        let input_shape = x.shape().clone();
        // When possible, we avoid using a broadcasted matmul as it is much slower
        // than the standard matmul for the cuda and cpu backends.
        let x = match *x.dims() {
            [b1, b2, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((b1 * b2 * m, k))?
                        .matmul(&w)?
                        .reshape((b1, b2, m, ()))?
                } else {
                    let w = self.weight.broadcast_left((b1, b2))?.t()?;
                    x.matmul(&w)?
                }
            }
            [bsize, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((bsize * m, k))?
                        .matmul(&w)?
                        .reshape((bsize, m, ()))?
                } else {
                    let w = self.weight.broadcast_left(bsize)?.t()?;
                    x.matmul(&w)?
                }
            }
            _ => {
                let w = self.weight.t()?;
                x.matmul(&w)?
            }
        };
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
        .with_context(|| format!(
            "Linear({in_features}->{out_features}): input shape {input_shape:?}",
        ))
    }
}

/// Creates a new linear layer with bias from a [`VarBuilder`](crate::VarBuilder).
///
/// Initializes weight with Kaiming normal and bias with uniform distribution.
/// The weight and bias are loaded from or stored in `vb` under the names `"weight"`
/// and `"bias"`.
///
/// # Example
///
/// ```text
/// // Linear layers are typically constructed from a VarBuilder:
/// // let linear = candle_nn::linear(in_dim, out_dim, vb)?;
/// ```
pub fn linear(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}

/// Creates a new linear layer without bias from a [`VarBuilder`](crate::VarBuilder).
///
/// Initializes weight with Kaiming normal. The weight is loaded from or stored in
/// `vb` under the name `"weight"`.
///
/// # Example
///
/// ```text
/// // let linear = candle_nn::linear_no_bias(in_dim, out_dim, vb)?;
/// ```
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(Linear::new(ws, None))
}

/// Creates a new linear layer, optionally with bias, from a [`VarBuilder`](crate::VarBuilder).
///
/// If `bias` is true, behaves like [`linear`]; otherwise like [`linear_no_bias`].
///
/// # Example
///
/// ```text
/// // let layer = candle_nn::linear_b(in_dim, out_dim, true, vb)?;
/// ```
pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: crate::VarBuilder,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

//! Activation functions for neural networks.
//!
//! This module provides the [`Activation`] enum, which can represent and apply common
//! activation functions such as ReLU, GELU, SiLU, and others. The enum implements the
//! [`Module`](candle::Module) trait so it can be used directly in model forward passes.
//!
//! ```rust
//! use candle::{Tensor, Device};
//! use candle_nn::{Activation, Module};
//!
//! let act = Activation::Gelu;
//! let x = Tensor::new(&[-1.0f32, 0.0, 1.0], &Device::Cpu)?;
//! let y = act.forward(&x)?;
//! # Ok::<(), candle::Error>(())
//! ```

use candle::{Result, Tensor};

/// Common activation functions for neural network layers.
///
/// Each variant applies a different non-linear function element-wise to its input.
/// The enum implements [`Module`](candle::Module), so you can call `.forward(&tensor)`
/// to apply the activation.
///
/// # Supported activations
///
/// | Variant | Formula |
/// |---------|---------|
/// | `Gelu` | Gaussian Error Linear Unit (erf-based) |
/// | `NewGelu` | GELU with tanh approximation |
/// | `Relu` | `max(0, x)` |
/// | `Relu2` | `max(0, x)^2` |
/// | `Relu6` | `clamp(x, 0, 6)` |
/// | `Silu` | `x * sigmoid(x)` |
/// | `Sigmoid` | `1 / (1 + exp(-x))` |
/// | `Swiglu` | Split-and-gate: `silu(x1) * x2` |
/// | `Elu(alpha)` | Exponential Linear Unit |
/// | `LeakyRelu(slope)` | Leaky ReLU with configurable negative slope |
///
/// ```rust
/// use candle::{Tensor, Device, test_utils::to_vec1_round};
/// use candle_nn::{Activation, Module};
///
/// let relu = Activation::Relu;
/// let x = Tensor::new(&[-1.0f32, 0.0, 1.0], &Device::Cpu)?;
/// let y = relu.forward(&x)?;
/// assert_eq!(y.to_vec1::<f32>()?, &[0.0, 0.0, 1.0]);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Relu2,
    Relu6,
    Silu,
    Sigmoid,
    HardSigmoid,
    Swiglu,
    Swish,
    Mish,
    HardSwish,
    Elu(f64),
    LeakyRelu(f64),
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
}

impl super::Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu_erf(),
            // https://github.com/huggingface/transformers/blob/12f043eaeaabfef6f6efea411d98e6f6d3c094b7/src/transformers/activations.py#L49-L78
            Self::NewGelu => xs.gelu(),
            Self::Relu => xs.relu(),
            Self::Relu2 => xs.relu()?.sqr(),
            Self::Relu6 => xs.clamp(0f32, 6f32),
            Self::Silu => xs.silu(),
            Self::Sigmoid => crate::ops::sigmoid(xs),
            Self::HardSigmoid => crate::ops::hard_sigmoid(xs),
            Self::Swiglu => crate::ops::swiglu(xs),
            Self::Swish => xs * crate::ops::sigmoid(xs)?,
            Self::HardSwish => xs * crate::ops::hard_sigmoid(xs)?,
            Self::Mish => crate::ops::mish(xs),
            &Self::Elu(alpha) => xs.elu(alpha),
            &Self::LeakyRelu(negative_slope) => crate::ops::leaky_relu(xs, negative_slope),
            Self::GeluPytorchTanh => xs.gelu(),
        }
    }
}

/// Parametric ReLU activation: `max(0, x) + weight * min(0, x)`.
///
/// Unlike [`Activation::LeakyRelu`], the negative slope is a learned parameter.
/// The weight can be a scalar (shared across all channels) or a 1D vector with
/// one value per channel.
///
/// # Example
///
/// ```rust
/// use candle::{Tensor, Device, Module};
/// use candle_nn::activation::PReLU;
///
/// // Scalar weight of 0.1 — negative inputs are multiplied by 0.1.
/// let w = Tensor::new(&[0.1f32], &Device::Cpu)?;
/// let act = PReLU::new(w, true);
/// let x = Tensor::new(&[-2.0f32, -1.0, 0.0, 1.0], &Device::Cpu)?;
/// let y = act.forward(&x)?;
/// assert_eq!(y.to_vec1::<f32>()?, &[-0.2, -0.1, 0.0, 1.0]);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct PReLU {
    weight: Tensor,
    is_scalar: bool,
}

impl PReLU {
    /// Creates a new PReLU activation with the given weight tensor and scalar flag.
    pub fn new(weight: Tensor, is_scalar: bool) -> Self {
        Self { weight, is_scalar }
    }

    /// Returns a reference to the learnable weight parameter.
    /// Returns a reference to the learnable weight parameter.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns `true` if this PReLU uses a single shared scalar weight for all channels.
    pub fn is_scalar(&self) -> bool {
        self.is_scalar
    }
}

impl candle::Module for PReLU {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let weight = if self.is_scalar {
            self.weight.reshape(())?
        } else if xs.shape() == self.weight.shape() {
            self.weight.clone()
        } else if xs.rank() >= 2 {
            let num_channels = xs.dim(1)?;
            let num_weights = self.weight.elem_count();
            if num_weights != num_channels {
                candle::bail!("error in prelu: unexpected number of channels for the input, got {num_channels}, weight dim is {num_weights}")
            }
            let mut s = vec![1; xs.rank()];
            s[1] = num_weights;
            self.weight.reshape(s)?
        } else {
            self.weight.clone()
        };
        let zeros = xs.zeros_like()?;
        xs.maximum(&zeros)? + xs.minimum(&zeros)?.broadcast_mul(&weight)?
    }
}

/// Create or initialize a new PReLU layer.
///
/// This uses some default name for weights, namely `"weight"`.
///
/// # Arguments
///
/// * `num_channels` - The number of channels. Use `None` to have as single trainable value and
///   `Some` for a 1D vector with the appropriate number of channels. When applying the `forward`
///   function, the input tensor shape `s` should either be one dimension with this number of
///   channels or if `s.len() >= 2` it should have `s[1]` equal to this number.
///
/// # Example
///
/// ```no_run
/// use candle_nn::{prelu, VarBuilder};
///
/// // Scalar PReLU (shared weight across all channels):
/// // let act = prelu(None, vb)?;
///
/// // Per-channel PReLU for 16 channels:
/// // let act = prelu(Some(16), vb.pp("prelu"))?;
/// ```
pub fn prelu(num_channels: Option<usize>, vs: crate::VarBuilder) -> Result<PReLU> {
    let init_ws = crate::init::Init::Const(0.25);
    // When using a scalar weight, the PyTorch encoding is to use a 1d vector of length 1.
    let ws = vs.get_with_hints((num_channels.unwrap_or(1),), "weight", init_ws)?;
    Ok(PReLU::new(ws, num_channels.is_none()))
}

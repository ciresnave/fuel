//! A linear layer that transparently handles both full-precision and GGUF
//! quantized weights through a single `Module`-compatible interface.
//!
//! This allows inference code to be written once and work with both
//! safetensors checkpoints (float weights) and GGUF files (quantized weights)
//! without per-callsite conditionals.

use candle::quantized::QMatMul;
use candle::{Module, Result, Tensor};

use crate::Linear;

/// A linear layer that is either full-precision or quantized.
///
/// `QuantizableLinear` wraps either a [`Linear`] layer (fp32/fp16/bf16 weights
/// from a safetensors checkpoint) or a [`QMatMul`] layer (Q4_0, Q4_K, Q8_0,
/// etc. from a GGUF file). Both variants implement [`Module`] identically:
/// `forward` always returns outputs in the same floating-point precision as
/// the input tensor.
///
/// # Usage pattern
///
/// Design model structs using `QuantizableLinear` for all projection layers.
/// Load from safetensors or GGUF using the appropriate constructor, then run
/// the same `forward` pass regardless of which format was loaded:
///
/// ```no_run
/// use candle_nn::{QuantizableLinear, Module};
/// use candle::{Device, DType, Tensor};
///
/// # fn main() -> candle::Result<()> {
/// # let device = Device::Cpu;
/// // Float weights (from safetensors):
/// let w = Tensor::randn(0f32, 1., (64, 128), &device)?;
/// let float_layer = QuantizableLinear::from_float(w, None);
///
/// let x = Tensor::ones((4, 128), DType::F32, &device)?;
/// let out = float_layer.forward(&x)?;  // shape: [4, 64]
/// # Ok(())
/// # }
/// ```
///
/// # Bias handling
///
/// The `Quantized` variant does not carry an optional bias because GGUF models
/// typically encode bias-less projections. If a quantized layer needs a bias,
/// apply it manually after `forward`.
#[derive(Clone, Debug)]
pub enum QuantizableLinear {
    /// Full-precision linear layer (fp32 / fp16 / bf16 weights from safetensors).
    Float(Linear),
    /// Quantized linear layer (GGUF Q4_0, Q4_K, Q8_0, etc.).
    Quantized(QMatMul),
}

impl QuantizableLinear {
    /// Construct a full-precision `QuantizableLinear` from a weight tensor and
    /// an optional bias.
    ///
    /// `weight` must have shape `(out_features, in_features)` — the same
    /// convention as [`Linear`].
    pub fn from_float(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self::Float(Linear::new(weight, bias))
    }

    /// Construct a quantized `QuantizableLinear` from a [`QMatMul`].
    pub fn from_quantized(weight: QMatMul) -> Self {
        Self::Quantized(weight)
    }

    /// Returns `true` if this layer holds quantized weights.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }

    /// Dequantize to a plain `Tensor` weight.
    /// Useful for weight inspection, export, or when the caller needs a plain
    /// tensor regardless of the storage format.
    ///
    /// Dispatch:
    /// - `Float` — returns the weight tensor directly.
    /// - `Quantized(QTensor)` — dequantizes to `F32` via `QTensor::dequantize`.
    /// - `Quantized(Tensor)` — already full-precision; returned as-is.
    /// - `Quantized(TensorF16)` — converted `F16` → `F32`.
    pub fn dequantized_weight(&self) -> Result<Tensor> {
        match self {
            Self::Float(l) => Ok(l.weight().clone()),
            Self::Quantized(q) => match q {
                QMatMul::QTensor(qt) => qt.dequantize(&qt.device()),
                QMatMul::Tensor(t) => Ok(t.clone()),
                QMatMul::TensorF16(t) => t.to_dtype(candle::DType::F32),
            },
        }
    }
}

impl Module for QuantizableLinear {
    /// Run the linear transformation.
    ///
    /// For float weights: `out = input @ weight.T + bias`
    ///
    /// For quantized weights: equivalent matmul in the quantised format, output
    /// promoted to the same dtype as `xs` by `QMatMul::forward`.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Float(l) => l.forward(xs),
            Self::Quantized(q) => q.forward(xs),
        }
    }
}

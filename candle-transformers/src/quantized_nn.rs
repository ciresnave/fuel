//! Utilities for quanitized network layers
//!
//! This module contains various implementations of standard neural network layers, modules and
//! utilities including embedding, linear layers, and various normalization techniques.
//! Most implementations provide quantized weights support.

use crate::models::with_tracing::QMatMul;
use crate::quantized_var_builder::VarBuilder;
use candle::quantized::QTensor;
use candle::{Module, Result, Tensor};

/// An embedding layer that loads its weights from quantized tensors.
///
/// The quantized weight is dequantized once at construction time so that
/// forward passes run in full floating-point precision via the wrapped
/// [`candle_nn::Embedding`].
#[derive(Debug, Clone)]
pub struct Embedding {
    inner: candle_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    /// Creates a new `Embedding` layer by loading and dequantizing the weight matrix.
    ///
    /// # Arguments
    /// * `d1` – vocabulary size (number of distinct token ids).
    /// * `d2` – embedding dimension per token.
    /// * `vb` – quantized variable builder scoped to the embedding's `weight` tensor.
    pub fn new(d1: usize, d2: usize, vb: VarBuilder) -> Result<Self> {
        let embeddings = vb.get((d1, d2), "weight")?.dequantize(vb.device())?;
        let inner = candle_nn::Embedding::new(embeddings, d2);
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    /// Returns a reference to the dequantized embedding weight tensor of shape `(d1, d2)`.
    pub fn embeddings(&self) -> &Tensor {
        self.inner.embeddings()
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

/// A linear (fully-connected) layer backed by a quantized weight matrix.
///
/// The forward pass is performed through [`QMatMul`], which keeps the weights
/// in their compressed quantized form and dequantizes them on-the-fly during
/// matrix-multiplication.  An optional bias tensor stored in full precision
/// may be added after the matrix multiply.
#[derive(Debug, Clone)]
pub struct Linear {
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl Linear {
    /// Constructs a `Linear` layer from a shared [`QTensor`] reference and an optional bias.
    ///
    /// The `Arc<QTensor>` is converted into a [`QMatMul`] internally.
    pub fn from_arc(weight: std::sync::Arc<QTensor>, bias: Option<Tensor>) -> Result<Self> {
        let weight = QMatMul::from_weights(weight)?;
        Ok(Self { weight, bias })
    }

    /// Constructs a `Linear` layer from an already-created [`QMatMul`] and an optional bias.
    pub fn from_weights(weight: QMatMul, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let x = x.apply(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

/// Creates a linear layer with configurable bias, loading weights from a quantized [`VarBuilder`].
///
/// # Arguments
/// * `in_dim`  – input feature dimension.
/// * `out_dim` – output feature dimension.
/// * `bias`    – if `true`, a `bias` tensor is loaded and dequantized from `vb`.
pub fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?.dequantize(vb.device())?)
    } else {
        None
    };
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear { weight, bias })
}

/// Creates a linear layer with a bias term, loading weights from a quantized [`VarBuilder`].
pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let bias = vb.get(out_dim, "bias")?.dequantize(vb.device())?;
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear {
        weight,
        bias: Some(bias),
    })
}

/// Creates a layer-normalization module by loading dequantized `weight` and `bias` tensors.
pub fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::LayerNorm> {
    let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
    let bias = vb.get(size, "bias")?.dequantize(vb.device())?;
    Ok(candle_nn::LayerNorm::new(weight, bias, eps))
}

/// Creates a bias-free layer-normalization module by loading a dequantized `weight` tensor.
pub fn layer_norm_no_bias(size: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::LayerNorm> {
    let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
    Ok(candle_nn::LayerNorm::new_no_bias(weight, eps))
}

/// Creates a linear layer without a bias term, loading weights from a quantized [`VarBuilder`].
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear { weight, bias: None })
}

/// Root-mean-square normalisation layer backed by a dequantized scale vector.
///
/// Functionally equivalent to [`candle_nn::RmsNorm`] but loads its parameters
/// from a quantized [`VarBuilder`] or directly from a [`QTensor`],
/// dequantizing them once at construction time.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl RmsNorm {
    /// Creates an `RmsNorm` by loading and dequantizing the scale vector from a [`VarBuilder`].
    ///
    /// # Arguments
    /// * `size` – length of the normalisation vector (must equal the last tensor dimension).
    /// * `eps`  – small constant added to the RMS for numerical stability.
    /// * `vb`   – quantized variable builder scoped to the `weight` tensor.
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
        Ok(Self { weight, eps, span })
    }

    /// Constructs an `RmsNorm` directly from a [`QTensor`].
    ///
    /// The tensor is dequantized eagerly; its device is inferred from the tensor itself.
    pub fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps, span })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}

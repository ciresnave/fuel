use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// A traced embedding layer that wraps `candle_nn::Embedding` with a tracing span.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::with_tracing::Embedding;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let emb = Embedding::new(1000, 128, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Embedding {
    inner: candle_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    /// Create a new embedding layer of shape `(d1, d2)`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::with_tracing::Embedding;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let emb = Embedding::new(1000, 128, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(d1: usize, d2: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::embedding(d1, d2, vb)?;
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    /// Create an embedding layer directly from a weight tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use candle_transformers::models::with_tracing::Embedding;
    /// use candle::{Device, DType, Tensor};
    /// let weights = Tensor::zeros((1000, 128), DType::F32, &Device::Cpu)?;
    /// let emb = Embedding::from_weights(weights)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn from_weights(weights: Tensor) -> Result<Self> {
        let (_in_size, out_size) = weights.dims2()?;
        let inner = candle_nn::Embedding::new(weights, out_size);
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    /// Return a reference to the underlying embedding weight tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use candle_transformers::models::with_tracing::Embedding;
    /// use candle::{Device, DType, Tensor};
    /// let weights = Tensor::zeros((1000, 128), DType::F32, &Device::Cpu)?;
    /// let emb = Embedding::from_weights(weights)?;
    /// let w = emb.embeddings();
    /// assert_eq!(w.dims(), &[1000, 128]);
    /// # Ok::<(), candle::Error>(())
    /// ```
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

/// A traced linear layer that wraps `candle_nn::Linear` with a tracing span.
///
/// # Example
///
/// ```
/// use candle_transformers::models::with_tracing::Linear;
/// use candle::{Device, DType, Tensor};
/// let w = Tensor::zeros((64, 64), DType::F32, &Device::Cpu)?;
/// let layer = Linear::from_weights(w, None);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Linear {
    inner: candle_nn::Linear,
    span: tracing::Span,
}

impl Linear {
    /// Create a linear layer directly from weight and optional bias tensors.
    ///
    /// # Example
    ///
    /// ```
    /// use candle_transformers::models::with_tracing::Linear;
    /// use candle::{Device, DType, Tensor};
    /// let w = Tensor::zeros((64, 64), DType::F32, &Device::Cpu)?;
    /// let layer = Linear::from_weights(w, None);
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn from_weights(weights: Tensor, bias: Option<Tensor>) -> Self {
        let inner = candle_nn::Linear::new(weights, bias);
        let span = tracing::span!(tracing::Level::TRACE, "linear");
        Self { inner, span }
    }
}

/// Create a traced linear layer with optional bias.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::with_tracing::linear_b;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let layer = linear_b(64, 128, true, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
pub fn linear_b(d1: usize, d2: usize, b: bool, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear_b(d1, d2, b, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

/// Create a traced linear layer with bias.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::with_tracing::linear;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let layer = linear(64, 128, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
pub fn linear(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

/// Create a traced linear layer without bias.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::with_tracing::linear_no_bias;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let layer = linear_no_bias(64, 128, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
pub fn linear_no_bias(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear_no_bias(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

/// A traced 2D convolution layer that wraps `candle_nn::Conv2d` with a tracing span.
// Wrap the conv2d op to provide some tracing.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::with_tracing::conv2d;
/// # use candle_nn::{Conv2dConfig, VarBuilder};
/// # let vb: VarBuilder = unimplemented!();
/// let layer = conv2d(3, 16, 3, Conv2dConfig::default(), vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Conv2d {
    inner: candle_nn::Conv2d,
    span: tracing::Span,
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

/// Create a traced 2D convolution layer.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::with_tracing::conv2d;
/// # use candle_nn::{Conv2dConfig, VarBuilder};
/// # let vb: VarBuilder = unimplemented!();
/// let layer = conv2d(3, 16, 3, Conv2dConfig::default(), vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vs: candle_nn::VarBuilder,
) -> Result<Conv2d> {
    let span = tracing::span!(tracing::Level::TRACE, "conv2d");
    let inner = candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vs)?;
    Ok(Conv2d { inner, span })
}

/// A traced quantized matrix multiply layer.
// QMatMul wrapper adding some tracing.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::with_tracing::QMatMul;
/// # use candle_transformers::quantized_var_builder::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let layer = QMatMul::new(128, 64, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Clone)]
pub struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    /// Load a quantized linear layer from a variable store.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::with_tracing::QMatMul;
    /// # use candle_transformers::quantized_var_builder::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let layer = QMatMul::new(128, 64, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(
        out_dim: usize,
        in_dim: usize,
        vb: crate::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let ws = vb.get((in_dim, out_dim), "weight")?;
        let inner = candle::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    /// Create a `QMatMul` directly from a pre-loaded quantized tensor.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::with_tracing::QMatMul;
    /// # use std::sync::Arc;
    /// # use candle::quantized::QTensor;
    /// # let ws: Arc<QTensor> = unimplemented!();
    /// let layer = QMatMul::from_weights(ws)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn from_weights(ws: std::sync::Arc<candle::quantized::QTensor>) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }
}

impl Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

impl std::fmt::Debug for QMatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QMatMul")
    }
}

/// A traced layer norm that wraps `candle_nn::LayerNorm` with a tracing span.
///
/// # Example
///
/// ```
/// use candle_transformers::models::with_tracing::LayerNorm;
/// use candle::{Device, DType, Tensor};
/// let w = Tensor::ones(64usize, DType::F32, &Device::Cpu)?;
/// let b = Tensor::zeros(64usize, DType::F32, &Device::Cpu)?;
/// let norm = LayerNorm::new(w, b, 1e-5);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct LayerNorm {
    inner: candle_nn::LayerNorm,
    span: tracing::Span,
}

impl LayerNorm {
    /// Create a layer-norm from explicit weight, bias, and epsilon.
    ///
    /// # Example
    ///
    /// ```
    /// use candle_transformers::models::with_tracing::LayerNorm;
    /// use candle::{Device, DType, Tensor};
    /// let w = Tensor::ones(64usize, DType::F32, &Device::Cpu)?;
    /// let b = Tensor::zeros(64usize, DType::F32, &Device::Cpu)?;
    /// let norm = LayerNorm::new(w, b, 1e-5);
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        let inner = candle_nn::LayerNorm::new(weight, bias, eps);
        let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
        Self { inner, span }
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

/// Create a traced layer-norm from a config and variable store.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::with_tracing::layer_norm;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let norm = layer_norm(64, 1e-5, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
pub fn layer_norm<C: Into<candle_nn::LayerNormConfig>>(
    size: usize,
    c: C,
    vb: VarBuilder,
) -> Result<LayerNorm> {
    let inner = candle_nn::layer_norm(size, c, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
    Ok(LayerNorm { inner, span })
}

/// A traced RMS normalization layer that wraps `candle_nn::RmsNorm` with a tracing span.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::with_tracing::RmsNorm;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let norm = RmsNorm::new(64, 1e-5, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct RmsNorm {
    inner: candle_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    /// Create an RMS norm layer of the given hidden size and epsilon.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::with_tracing::RmsNorm;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let norm = RmsNorm::new(64, 1e-5, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm(size, eps, vb)?;
        Ok(Self { inner, span })
    }

    /// Run RMS normalization and return the differenced output (for gradient computations).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::with_tracing::RmsNorm;
    /// # use candle::{Device, DType, Tensor};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let norm = RmsNorm::new(64, 1e-5, vb)?;
    /// let x = Tensor::zeros((1, 64), DType::F32, &Device::Cpu)?;
    /// let out = norm.forward_diff(&x)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn forward_diff(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward_diff(x)
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

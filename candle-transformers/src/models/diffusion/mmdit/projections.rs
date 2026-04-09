use candle::{Module, Result, Tensor};
use candle_nn as nn;

/// Container holding query, key, and value tensors for attention computation.
pub struct Qkv {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

/// Two-layer MLP with GELU activation used in transformer feed-forward blocks.
pub struct Mlp {
    fc1: nn::Linear,
    act: nn::Activation,
    fc2: nn::Linear,
}

impl Mlp {
    /// Creates a new `Mlp` with the given input and hidden feature dimensions.
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let fc1 = nn::linear(in_features, hidden_features, vb.pp("fc1"))?;
        let act = nn::Activation::GeluPytorchTanh;
        let fc2 = nn::linear(hidden_features, in_features, vb.pp("fc2"))?;

        Ok(Self { fc1, act, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = self.act.forward(&x)?;
        self.fc2.forward(&x)
    }
}

/// Attention projections that only compute QKV (no output/projection layer).
pub struct QkvOnlyAttnProjections {
    qkv: nn::Linear,
    head_dim: usize,
}

impl QkvOnlyAttnProjections {
    /// Creates a new `QkvOnlyAttnProjections` for the given dimension and number of heads.
    pub fn new(dim: usize, num_heads: usize, vb: nn::VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = nn::linear(dim, dim * 3, vb.pp("qkv"))?;
        Ok(Self { qkv, head_dim })
    }

    /// Projects input to QKV tensors split by head dimension.
    pub fn pre_attention(&self, x: &Tensor) -> Result<Qkv> {
        let qkv = self.qkv.forward(x)?;
        split_qkv(&qkv, self.head_dim)
    }
}

/// Full attention projections including QKV computation, optional QK norm, and output projection.
pub struct AttnProjections {
    head_dim: usize,
    qkv: nn::Linear,
    ln_k: Option<candle_nn::RmsNorm>,
    ln_q: Option<candle_nn::RmsNorm>,
    proj: nn::Linear,
}

impl AttnProjections {
    /// Creates a new `AttnProjections` with optional per-head QK layer normalization.
    pub fn new(dim: usize, num_heads: usize, vb: nn::VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = nn::linear(dim, dim * 3, vb.pp("qkv"))?;
        let proj = nn::linear(dim, dim, vb.pp("proj"))?;
        let (ln_k, ln_q) = if vb.contains_tensor("ln_k.weight") {
            let ln_k = candle_nn::rms_norm(head_dim, 1e-6, vb.pp("ln_k"))?;
            let ln_q = candle_nn::rms_norm(head_dim, 1e-6, vb.pp("ln_q"))?;
            (Some(ln_k), Some(ln_q))
        } else {
            (None, None)
        };
        Ok(Self {
            head_dim,
            qkv,
            proj,
            ln_k,
            ln_q,
        })
    }

    /// Projects input to normalized QKV tensors; call before attention.
    pub fn pre_attention(&self, x: &Tensor) -> Result<Qkv> {
        let qkv = self.qkv.forward(x)?;
        let Qkv { q, k, v } = split_qkv(&qkv, self.head_dim)?;
        let q = match self.ln_q.as_ref() {
            None => q,
            Some(l) => {
                let (b, t, h) = q.dims3()?;
                l.forward(&q.reshape((b, t, (), self.head_dim))?)?
                    .reshape((b, t, h))?
            }
        };
        let k = match self.ln_k.as_ref() {
            None => k,
            Some(l) => {
                let (b, t, h) = k.dims3()?;
                l.forward(&k.reshape((b, t, (), self.head_dim))?)?
                    .reshape((b, t, h))?
            }
        };
        Ok(Qkv { q, k, v })
    }

    /// Projects the aggregated attention output through the output linear layer.
    pub fn post_attention(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)
    }
}

fn split_qkv(qkv: &Tensor, head_dim: usize) -> Result<Qkv> {
    let (batch_size, seq_len, _) = qkv.dims3()?;
    let qkv = qkv.reshape((batch_size, seq_len, 3, (), head_dim))?;
    let q = qkv.get_on_dim(2, 0)?;
    let q = q.reshape((batch_size, seq_len, ()))?;
    let k = qkv.get_on_dim(2, 1)?;
    let k = k.reshape((batch_size, seq_len, ()))?;
    let v = qkv.get_on_dim(2, 2)?;
    Ok(Qkv { q, k, v })
}

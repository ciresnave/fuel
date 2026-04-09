//! Module implementing the MPT (Multi-Purpose Transformer) model
//!
//! References:
//! - [MPT Model used by replit-code-v1_5-3b](https://huggingface.co/replit/replit-code-v1_5-3b/blob/main/modeling_mpt.py)
//! - [Configuration](https://huggingface.co/replit/replit-code-v1_5-3b/blob/main/configuration_mpt.py)
//!
//! The model uses grouped query attention and alibi positional embeddings.

use crate::models::with_tracing::{linear_no_bias, Embedding, Linear};
use crate::utils::masked_fill;
/// MPT model used by replit-code-v1_5-3b
/// https://huggingface.co/replit/replit-code-v1_5-3b/blob/main/modeling_mpt.py
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{kv_cache::KvCache, layer_norm, LayerNorm, VarBuilder};

// https://huggingface.co/replit/replit-code-v1_5-3b/blob/main/configuration_mpt.py
/// MPT model configuration.
///
/// Deserializable from a HuggingFace-style `config.json`.
///
/// # Example
///
/// ```
/// use candle_transformers::models::mpt::Config;
/// let cfg = Config::replit_code_v1_5_3b();
/// assert!(cfg.is_causal());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub(crate) d_model: usize,
    pub(crate) n_heads: usize,
    pub(crate) n_layers: usize,
    pub(crate) expansion_ratio: usize,
    pub(crate) max_seq_len: usize,
    pub(crate) vocab_size: usize,
    pub(crate) kv_n_heads: usize,
    pub(crate) attn_prefix_lm: bool,
    pub(crate) attn_alibi: bool,
    pub(crate) attn_alibi_bias_max: usize,
}

impl Config {
    /// Return the preset configuration for the Replit-Code-v1.5-3B model.
    ///
    /// # Example
    ///
    /// ```
    /// use candle_transformers::models::mpt::Config;
    /// let cfg = Config::replit_code_v1_5_3b();
    /// assert!(cfg.is_causal());
    /// ```
    pub fn replit_code_v1_5_3b() -> Self {
        Self {
            d_model: 3072,
            n_heads: 24,
            n_layers: 32,
            expansion_ratio: 4,
            max_seq_len: 4096,
            vocab_size: 32768,
            kv_n_heads: 8,
            attn_prefix_lm: false,
            attn_alibi: true,
            attn_alibi_bias_max: 8,
        }
    }

    /// Return `true` if the model uses a causal (autoregressive) attention mask.
    ///
    /// # Example
    ///
    /// ```
    /// use candle_transformers::models::mpt::Config;
    /// let cfg = Config::replit_code_v1_5_3b();
    /// assert!(cfg.is_causal());
    /// ```
    pub fn is_causal(&self) -> bool {
        !self.attn_prefix_lm
    }
}

#[derive(Debug, Clone)]
struct GroupedQueryAttention {
    wqkv: Linear,
    out_proj: Linear,
    kv_cache: KvCache,
    softmax_scale: f64,
    head_dim: usize,
    d_model: usize,
    n_heads: usize,
    kv_n_heads: usize,
    attn_bias: Tensor,
    span: tracing::Span,
}

impl GroupedQueryAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.d_model / cfg.n_heads;
        let wqkv_size = cfg.d_model + 2 * cfg.kv_n_heads * head_dim;
        let wqkv = linear_no_bias(cfg.d_model, wqkv_size, vb.pp("Wqkv"))?;
        let softmax_scale = 1f64 / (head_dim as f64).sqrt();
        let out_proj = linear_no_bias(cfg.d_model, cfg.d_model, vb.pp("out_proj"))?;
        let attn_bias = build_alibi_bias(cfg)?.to_device(vb.device())?;
        Ok(Self {
            wqkv,
            out_proj,
            kv_cache: KvCache::new(2, cfg.max_seq_len),
            softmax_scale,
            head_dim,
            d_model: cfg.d_model,
            n_heads: cfg.n_heads,
            kv_n_heads: cfg.kv_n_heads,
            attn_bias,
            span: tracing::span!(tracing::Level::TRACE, "gqa"),
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_size, seq_len, _n_embd) = xs.dims3()?;
        let qkv = self.wqkv.forward(xs)?;
        let query = qkv.narrow(2, 0, self.d_model)?;
        let kv_size = self.kv_n_heads * self.head_dim;
        let key = qkv.narrow(2, self.d_model, kv_size)?;
        let value = qkv.narrow(2, self.d_model + kv_size, kv_size)?;
        // scaled_multihead_dot_product_attention
        let query = query
            .reshape((b_size, seq_len, self.n_heads, ()))?
            .transpose(1, 2)?; // b,h,s,d
        let key = key
            .reshape((b_size, seq_len, self.kv_n_heads, ()))?
            .transpose(1, 2)?; // b,h,s,d
        let value = value
            .reshape((b_size, seq_len, self.kv_n_heads, ()))?
            .transpose(1, 2)?; // b,h,s,d
        let (key, value) = self.kv_cache.append(&key.contiguous()?, &value.contiguous()?)?;
        let query = query.contiguous()?;
        let key = crate::utils::repeat_kv(key, self.n_heads / self.kv_n_heads)?;
        let value = crate::utils::repeat_kv(value, self.n_heads / self.kv_n_heads)?;
        let attn_weights = (query.matmul(&key.transpose(2, 3)?)? * self.softmax_scale)?;
        let attn_bias = {
            let s_q = query.dim(D::Minus2)?;
            let s_k = key.dim(D::Minus1)?;
            let (_, _, a_q, a_k) = self.attn_bias.dims4()?;
            let start_q = a_q.saturating_sub(s_q);
            let start_k = a_k.saturating_sub(s_k);
            self.attn_bias.i((.., .., start_q.., start_k..))?
        };
        let attn_weights = attn_weights.broadcast_add(&attn_bias)?;
        let attn_weights = match mask {
            None => attn_weights,
            Some(mask) => masked_fill(
                &attn_weights,
                &mask.broadcast_as(attn_weights.shape())?,
                f32::NEG_INFINITY,
            )?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights
            .matmul(&value)?
            .transpose(1, 2)?
            .flatten_from(D::Minus2)?;
        let out = attn_output.apply(&self.out_proj)?;
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct Ffn {
    up_proj: Linear,
    down_proj: Linear,
}

impl Ffn {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.d_model * cfg.expansion_ratio;
        let up_proj = linear_no_bias(cfg.d_model, hidden, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(hidden, cfg.d_model, vb.pp("down_proj"))?;
        Ok(Self { up_proj, down_proj })
    }
}

impl Module for Ffn {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.up_proj)?.gelu_erf()?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct MPTBlock {
    norm1: LayerNorm, // Do we need the low-precision variant?
    attn: GroupedQueryAttention,
    norm2: LayerNorm,
    ffn: Ffn,
}

impl MPTBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln_cfg = candle_nn::LayerNormConfig {
            affine: false,
            ..Default::default()
        };
        let norm1 = layer_norm(cfg.d_model, ln_cfg, vb.pp("norm_1"))?;
        let norm2 = layer_norm(cfg.d_model, ln_cfg, vb.pp("norm_2"))?;
        let attn = GroupedQueryAttention::new(cfg, vb.pp("attn"))?;
        let ffn = Ffn::new(cfg, vb.pp("ffn"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            ffn,
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.norm1)?;
        let xs = self.attn.forward(&xs, mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.norm2)?.apply(&self.ffn)?;
        xs + residual
    }
}

pub(crate) fn build_alibi_bias(cfg: &Config) -> Result<Tensor> {
    let full = !cfg.is_causal();
    let seq_len = cfg.max_seq_len;
    let alibi_bias = Tensor::arange(1 - seq_len as i64, 1, &Device::Cpu)?;
    let alibi_bias = if full {
        let a1 = alibi_bias.reshape((1, 1, 1, seq_len))?;
        let a2 = alibi_bias.reshape((1, 1, seq_len, 1))?;
        a1.broadcast_sub(&a2)?.abs()?.neg()?
    } else {
        alibi_bias.reshape((1, 1, 1, seq_len))?
    };
    let mut n_heads2 = 1;
    while n_heads2 < cfg.n_heads {
        n_heads2 *= 2
    }
    let slopes = (1..=n_heads2)
        .map(|v| 1f32 / 2f32.powf((v * cfg.attn_alibi_bias_max) as f32 / n_heads2 as f32))
        .collect::<Vec<_>>();
    let slopes = if n_heads2 == cfg.n_heads {
        slopes
    } else {
        slopes
            .iter()
            .skip(1)
            .step_by(2)
            .chain(slopes.iter().step_by(2))
            .take(cfg.n_heads)
            .cloned()
            .collect::<Vec<f32>>()
    };
    let slopes = Tensor::new(slopes, &Device::Cpu)?.reshape((1, (), 1, 1))?;
    alibi_bias.to_dtype(DType::F32)?.broadcast_mul(&slopes)
}

/// MPT (MosaicML Pretrained Transformer) causal language model.
///
/// Uses multi-head attention with ALiBi positional biases and
/// grouped-query attention. Compatible with `replit/replit-code-v1_5-3b`.
///
/// # Example
///
/// ```no_run
/// use candle_transformers::models::mpt::{Model, Config};
/// use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let cfg = Config::replit_code_v1_5_3b();
/// let model = Model::new(&cfg, vb)?;
/// # Ok::<_, candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Model {
    wte: Embedding,
    blocks: Vec<MPTBlock>,
    norm_f: LayerNorm,
}

impl Model {
    /// Construct a new MPT model, loading weights from `vb`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use candle_transformers::models::mpt::{Model, Config};
    /// use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let cfg = Config::replit_code_v1_5_3b();
    /// let model = Model::new(&cfg, vb)?;
    /// # Ok::<_, candle::Error>(())
    /// ```
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let wte = Embedding::new(cfg.vocab_size, cfg.d_model, vb.pp("wte"))?;
        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let block = MPTBlock::new(cfg, vb_b.pp(i))?;
            blocks.push(block)
        }
        let ln_cfg = candle_nn::LayerNormConfig {
            affine: false,
            ..Default::default()
        };
        let norm_f = candle_nn::layer_norm(cfg.d_model, ln_cfg, vb.pp("norm_f"))?;
        Ok(Self {
            wte,
            blocks,
            norm_f,
        })
    }

    /// Run a forward pass given token ids.
    ///
    /// Returns logits of shape `(batch, vocab_size)` for the last token position.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use candle_transformers::models::mpt::Model;
    /// use candle::{Device, Tensor};
    /// # let mut model: Model = unimplemented!();
    /// let ids = Tensor::zeros((1, 8), candle::DType::U32, &Device::Cpu)?;
    /// let logits = model.forward(&ids)?;
    /// # Ok::<_, candle::Error>(())
    /// ```
    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (_b_size, seq_len) = xs.dims2()?;
        let mut xs = xs.apply(&self.wte)?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(get_mask(seq_len, xs.device())?)
        };
        for block in self.blocks.iter_mut() {
            xs = block.forward(&xs, mask.as_ref())?;
        }
        let xs = xs.apply(&self.norm_f)?;
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .squeeze(1)?
            .matmul(&self.wte.embeddings().t()?)?
            .squeeze(1)?;
        Ok(logits)
    }
}

pub(crate) fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}


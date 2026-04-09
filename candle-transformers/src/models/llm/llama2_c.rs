//! Llama2 inference implementation.
//!
//! See ["LLaMA 2: Open Foundation and Fine-Tuned Chat Models"](https://arxiv.org/abs/2307.09288)
//!
//! - ⚡ [Interactive Wasm Example](https://huggingface.co/spaces/lmz/candle-llama2)
//! - 💻 llama2.c [GH Link](https://github.com/karpathy/llama2.c)
//!

use crate::utils::masked_fill;
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::linear_no_bias as linear;
use candle_nn::{embedding, kv_cache::KvCache, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder};
use std::collections::HashMap;

/// Configuration for the Llama 2 model.
///
/// # Example
///
/// ```
/// use candle_transformers::models::llama2_c::Config;
/// let cfg = Config::tiny_15m();
/// assert_eq!(cfg.dim, 288);
/// ```
#[derive(Debug, Clone)]
pub struct Config {
    pub dim: usize,        // transformer dimension
    pub hidden_dim: usize, // for ffn layers
    pub n_layers: usize,   // number of layers
    pub n_heads: usize,    // number of query heads
    pub n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    pub seq_len: usize,    // max sequence length
    pub norm_eps: f64,
}

impl Config {
    /// Return the 260k-parameter tiny Llama 2 configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use candle_transformers::models::llama2_c::Config;
    /// let cfg = Config::tiny_260k();
    /// assert_eq!(cfg.dim, 64);
    /// ```
    pub fn tiny_260k() -> Self {
        Self {
            dim: 64,
            hidden_dim: 768,
            n_layers: 5,
            n_heads: 8,
            n_kv_heads: 4,
            vocab_size: 32000,
            seq_len: 512,
            norm_eps: 1e-5,
        }
    }

    /// Return the 15M-parameter tiny Llama 2 configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use candle_transformers::models::llama2_c::Config;
    /// let cfg = Config::tiny_15m();
    /// assert_eq!(cfg.dim, 288);
    /// ```
    pub fn tiny_15m() -> Self {
        Self {
            dim: 288,
            hidden_dim: 768,
            n_layers: 6,
            n_heads: 6,
            n_kv_heads: 6,
            vocab_size: 32000,
            seq_len: 256,
            norm_eps: 1e-5,
        }
    }

    /// Return the 42M-parameter tiny Llama 2 configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use candle_transformers::models::llama2_c::Config;
    /// let cfg = Config::tiny_42m();
    /// assert_eq!(cfg.dim, 512);
    /// ```
    pub fn tiny_42m() -> Self {
        Self {
            dim: 512,
            hidden_dim: 768,
            n_layers: 8,
            n_heads: 8,
            n_kv_heads: 8,
            vocab_size: 32000,
            seq_len: 1024,
            norm_eps: 1e-5,
        }
    }

    /// Return the 110M-parameter tiny Llama 2 configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use candle_transformers::models::llama2_c::Config;
    /// let cfg = Config::tiny_110m();
    /// assert_eq!(cfg.dim, 768);
    /// ```
    pub fn tiny_110m() -> Self {
        Self {
            dim: 768,
            hidden_dim: 768,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 12,
            vocab_size: 32000,
            seq_len: 1024,
            norm_eps: 1e-5,
        }
    }
}

/// RoPE cache and KV-cache for Llama 2 inference.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::llama2_c::{Cache, Config};
/// # use candle::{DType, Device};
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let cfg = Config::tiny_15m();
/// let mut cache = Cache::new(true, &cfg, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    pub kvs: Vec<KvCache>,
    pub cos: Tensor,
    pub sin: Tensor,
    device: Device,
}

impl Cache {
    /// Create a new cache with precomputed RoPE frequencies.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::llama2_c::{Cache, Config};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let cfg = Config::tiny_15m();
    /// let mut cache = Cache::new(true, &cfg, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(use_kv_cache: bool, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let n_elem = cfg.dim / cfg.n_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / n_elem as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), vb.device())?;
        let idx_theta = Tensor::arange(0, cfg.seq_len as u32, vb.device())?
            .to_dtype(DType::F32)?
            .reshape((cfg.seq_len, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let precomputed_cos = idx_theta.cos()?;
        let precomputed_sin = idx_theta.sin()?;

        let freq_cis_real = vb
            .get((cfg.seq_len, cfg.head_size() / 2), "freq_cis_real")
            .unwrap_or(precomputed_cos);
        let freq_cis_imag = vb
            .get((cfg.seq_len, cfg.head_size() / 2), "freq_cis_imag")
            .unwrap_or(precomputed_sin);
        let cos = freq_cis_real.reshape((cfg.seq_len, cfg.head_size() / 2, 1))?;
        let sin = freq_cis_imag.reshape((cfg.seq_len, cfg.head_size() / 2, 1))?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: (0..cfg.n_layers)
                .map(|_| KvCache::new(1, cfg.seq_len))
                .collect(),
            cos,
            sin,
            device: vb.device().clone(),
        })
    }

    /// Return (or compute and cache) a causal attention mask for the given sequence length.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::llama2_c::{Cache, Config};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let cfg = Config::tiny_15m();
    /// let mut cache = Cache::new(true, &cfg, vb)?;
    /// let mask = cache.mask(8, 0)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn mask(&mut self, seq_len: usize, index_pos: usize) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = crate::utils::build_causal_mask(seq_len, index_pos, &self.device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }
}

fn silu(xs: &Tensor) -> Result<Tensor> {
    xs / (xs.neg()?.exp()? + 1.0)?
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_head: usize,
    n_key_value_head: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
        let (b_sz, seq_len, h, n_embd) = x.dims4()?;
        let cos = cache.cos.i(index_pos..index_pos + seq_len)?;
        let sin = cache.sin.i(index_pos..index_pos + seq_len)?;
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;
        let cos = cos.broadcast_as((b_sz, seq_len, 1, n_embd / 2, 1))?;
        let sin = sin.broadcast_as((b_sz, seq_len, 1, n_embd / 2, 1))?;
        let x = x.reshape((b_sz, seq_len, h, n_embd / 2, 2))?;
        let x0 = x.narrow(D::Minus1, 0, 1)?;
        let x1 = x.narrow(D::Minus1, 1, 1)?;
        let dst0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        let dst1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
        let rope = Tensor::cat(&[&dst0, &dst1], D::Minus1)?.reshape((b_sz, seq_len, h, n_embd))?;
        Ok(rope)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.n_key_value_head, self.head_dim))?;
        let mut v = v.reshape((b_sz, seq_len, self.n_key_value_head, self.head_dim))?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            let (new_k, new_v) = cache.kvs[block_idx].append(&k.contiguous()?, &v.contiguous()?)?;
            k = new_k;
            v = new_v;
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = if seq_len <= 1 {
            att
        } else {
            let mask = cache.mask(seq_len, index_pos)?.broadcast_as(att.shape())?;
            masked_fill(&att, &mask, f32::NEG_INFINITY)?
        };
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        // Convert to contiguous as matmul doesn't support strided vs for now.
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.n_head / self.n_key_value_head;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, seq_len, n_kv_head, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(3)?
                .expand((b_sz, seq_len, n_kv_head, n_rep, head_dim))?
                .reshape((b_sz, seq_len, n_kv_head * n_rep, head_dim))?;
            Ok(x)
        }
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let size_in = cfg.dim;
        let size_q = (cfg.dim / cfg.n_heads) * cfg.n_heads;
        let size_kv = (cfg.dim / cfg.n_heads) * cfg.n_kv_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_head: cfg.n_heads,
            n_key_value_head: cfg.n_kv_heads,
            head_dim: cfg.dim / cfg.n_heads,
        })
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn new(c_fc1: Linear, c_fc2: Linear, c_proj: Linear) -> Self {
        Self {
            c_fc1,
            c_fc2,
            c_proj,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h_size = cfg.dim;
        let i_size = cfg.hidden_dim;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self::new(c_fc1, c_fc2, c_proj))
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn new(rms_1: RmsNorm, attn: CausalSelfAttention, rms_2: RmsNorm, mlp: Mlp) -> Self {
        Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let input_layernorm = rms_norm(cfg.dim, cfg.norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            rms_norm(cfg.dim, cfg.norm_eps, vb.pp("post_attention_layernorm"))?;
        Ok(Self::new(
            input_layernorm,
            attn,
            post_attention_layernorm,
            mlp,
        ))
    }
}

/// The Llama 2 language model.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::llama2_c::{Cache, Config, Llama};
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let cfg = Config::tiny_15m();
/// let model = Llama::load(vb, cfg)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    pub config: Config,
}

impl Llama {
    /// Run the forward pass and return vocabulary logits.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::llama2_c::{Cache, Config, Llama};
    /// # use candle::{Device, DType, Tensor};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let cfg = Config::tiny_15m();
    /// let model = Llama::load(vb.clone(), cfg.clone())?;
    /// let mut cache = Cache::new(true, &cfg, vb)?;
    /// let input = Tensor::zeros((1, 8), DType::U32, &Device::Cpu)?;
    /// let logits = model.forward(&input, 0, &mut cache)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let (_b_sz, _seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    /// Load the Llama 2 model from a variable store.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::llama2_c::{Config, Llama};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let cfg = Config::tiny_15m();
    /// let model = Llama::load(vb, cfg)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn load(vb: VarBuilder, cfg: Config) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.dim, vb.pp("model.embed_tokens"))?;
        let lm_head = linear(cfg.dim, cfg.vocab_size, vb.pp("lm_head"))?;
        let ln_f = rms_norm(cfg.dim, cfg.norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.n_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), &cfg).unwrap())
            .collect();
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            config: cfg,
        })
    }
}

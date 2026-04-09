//! Llama inference implementation.
//!
//! See ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971)
//!
//! Implementation based on Hugging Face's [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

use super::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
use crate::utils::masked_fill;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{embedding, kv_cache::KvCache, Embedding, Module, VarBuilder};
use std::{collections::HashMap, f32::consts::PI};

pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;

/// RoPE scaling type used by LLaMA-3 and its derivatives.
#[derive(Debug, Clone, serde::Deserialize, Default)]
pub enum Llama3RopeType {
    /// LLaMA-3 long-context RoPE scaling.
    #[serde(rename = "llama3")]
    Llama3,
    /// Standard unscaled RoPE (default).
    #[default]
    #[serde(rename = "default")]
    Default,
}

/// Parameters for LLaMA-3 long-context RoPE frequency scaling.
#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct Llama3RopeConfig {
    /// Global frequency scaling factor.
    pub factor: f32,
    /// Frequencies below this wavelen threshold are scaled by `factor`.
    pub low_freq_factor: f32,
    /// Frequencies above this wavelen threshold are left unscaled.
    pub high_freq_factor: f32,
    /// Maximum sequence length the base model was pretrained with.
    pub original_max_position_embeddings: usize,
    /// Selects the scaling algorithm; use `Llama3` for LLaMA-3 long-context.
    pub rope_type: Llama3RopeType,
}
/// End-of-sequence token id(s) for LLaMA models.
///
/// Some model variants specify a single EOS token while others list several;
/// this enum handles both cases during tokenizer configuration.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum LlamaEosToks {
    /// A single EOS token id.
    Single(u32),
    /// Multiple EOS token ids (any of these signals end-of-generation).
    Multiple(Vec<u32>),
}

/// Raw HuggingFace `config.json` fields for LLaMA / LLaMA-2 / LLaMA-3.
///
/// Deserialize this from the Hub `config.json` and call
/// [`into_config`](LlamaConfig::into_config) to obtain the internal [`Config`].
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: Option<bool>,
}

impl LlamaConfig {
    /// Returns the number of key-value heads, defaulting to `num_attention_heads`
    /// when `num_key_value_heads` is absent (i.e., no GQA).
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::llama::LlamaConfig;
    /// # let cfg: LlamaConfig = unimplemented!();
    /// let kv_heads = cfg.num_key_value_heads();
    /// assert!(kv_heads > 0);
    /// ```
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    /// Converts the HuggingFace config into the internal [`Config`] used by the model.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::llama::LlamaConfig;
    /// # let llama_cfg: LlamaConfig = unimplemented!();
    /// let cfg = llama_cfg.into_config(false);
    /// assert_eq!(cfg.use_flash_attn, false);
    /// ```
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            use_flash_attn,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        }
    }
}

/// Internal runtime configuration for the LLaMA model.
///
/// Consumed-only at construction time; use [`LlamaConfig`] to deserialize from
/// a HuggingFace `config.json` and then call [`LlamaConfig::into_config`].
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl Config {
    /// Returns the LLaMA-7B v1 preset configuration.
    ///
    /// # Example
    /// ```
    /// use candle_transformers::models::llama::Config;
    /// let cfg = Config::config_7b_v1(false);
    /// assert_eq!(cfg.hidden_size, 4096);
    /// assert_eq!(cfg.num_hidden_layers, 32);
    /// assert_eq!(cfg.vocab_size, 32000);
    /// ```
    pub fn config_7b_v1(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
        }
    }

    /// Returns the LLaMA-7B v2 preset configuration.
    ///
    /// Identical to v1 but uses `rms_norm_eps = 1e-5` instead of `1e-6`.
    ///
    /// # Example
    /// ```
    /// use candle_transformers::models::llama::Config;
    /// let cfg = Config::config_7b_v2(false);
    /// assert_eq!(cfg.hidden_size, 4096);
    /// assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-10);
    /// ```
    pub fn config_7b_v2(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
        }
    }
}

/// Pre-computed RoPE cosine/sine tables and KV-cache for one generation run.
///
/// Create one [`Cache`] per inference session and pass it mutably to each
/// [`Llama::forward`] call.  Set `use_kv_cache = false` to disable caching
/// (useful for benchmarking or when the full context fits in a single batch).
#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<KvCache>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    /// Allocates RoPE tables and empty KV caches for all transformer layers.
    ///
    /// # Arguments
    /// * `use_kv_cache` – enable incremental decoding with a key-value cache.
    /// * `dtype`        – dtype for the RoPE cosine/sine tensors.
    /// * `config`       – model configuration.
    /// * `device`       – target device for all pre-computed tensors.
    ///
    /// # Example
    /// ```
    /// use candle_transformers::models::llama::{Cache, Config};
    /// use candle::{DType, Device};
    /// # fn main() -> candle::Result<()> {
    /// let cfg = Config::config_7b_v1(false);
    /// let cache = Cache::new(true, DType::F32, &cfg, &Device::Cpu)?;
    /// assert!(cache.use_kv_cache);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let theta = match &config.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => calculate_default_inv_freq(config),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                calculate_default_inv_freq(config)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / rope_scaling.factor
                        } else {
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelen
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

        let theta = Tensor::new(theta, device)?;

        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: (0..config.num_hidden_layers)
                .map(|_| KvCache::new(2, config.max_position_embeddings))
                .collect(),
            device: device.clone(),
            cos,
            sin,
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize) -> Result<Tensor> {
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

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            let (new_k, new_v) = cache.kvs[block_idx].append(&k.contiguous()?, &v.contiguous()?)?;
            k = new_k;
            v = new_v;
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache.mask(seq_len, index_pos)?.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };

            let att = candle_nn::ops::softmax_last_dim(&att)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        crate::utils::repeat_kv(x, self.num_attention_heads / self.num_key_value_heads)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            span,
            span_rot,
        })
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

/// The LLaMA causal language model.
///
/// Wraps the token embedding, all transformer blocks, the final RMS-Norm, and
/// the LM head projection.  The [`Cache`] (pre-computed RoPE tables + KV caches)
/// must be created separately and passed to each [`forward`](Llama::forward) call.
#[derive(Debug, Clone)]
pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl Llama {
    // required by LLaVA
    /// Returns the embedding tensor for `x` without running the transformer blocks.
    ///
    /// Used by multi-modal models (e.g. LLaVA) that fuse visual tokens into the
    /// embedding space before passing them to the main model.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::llama::Llama;
    /// # use candle::{Device, DType, Tensor};
    /// # fn main() -> candle::Result<()> {
    /// # let model: Llama = unimplemented!();
    /// let ids = Tensor::zeros((1, 8), DType::U32, &Device::Cpu)?;
    /// let embeds = model.embed(&ids)?; // shape [1, 8, hidden_size]
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed(&self, x: &Tensor) -> Result<Tensor> {
        self.wte.forward(x)
    }
    // required by LLaVA
    /// Runs the transformer on a pre-computed input embedding instead of token ids.
    ///
    /// Used by multi-modal models (e.g. LLaVA) that inject visual features.
    ///
    /// # Arguments
    /// * `input_embed` – pre-computed embeddings of shape `(batch, seq_len, hidden_size)`.
    /// * `index_pos`   – position offset of the first token in `input_embed` within the KV cache.
    /// * `cache`       – mutable reference to the session's [`Cache`].
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::llama::{Cache, Config, Llama};
    /// # use candle_nn::VarBuilder;
    /// # use candle::{Device, DType, Tensor};
    /// # fn main() -> candle::Result<()> {
    /// # let model: Llama = unimplemented!();
    /// # let cfg = Config::config_7b_v1(false);
    /// # let mut cache = Cache::new(true, DType::F32, &cfg, &Device::Cpu)?;
    /// let embeds = Tensor::zeros((1, 8, 4096), DType::F32, &Device::Cpu)?;
    /// let logits = model.forward_input_embed(&embeds, 0, &mut cache)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward_input_embed(
        &self,
        input_embed: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = input_embed.dims3()?;
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    /// Runs a standard forward pass given token ids.
    ///
    /// # Arguments
    /// * `x`         – token id tensor of shape `(batch, seq_len)`.
    /// * `index_pos` – position offset of the first token (for incremental decoding).
    /// * `cache`     – mutable reference to the session's [`Cache`].
    ///
    /// Returns logits of shape `(batch, vocab_size)` for the last token position.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::llama::{Cache, Config, Llama};
    /// # use candle::{Device, DType, Tensor};
    /// # fn main() -> candle::Result<()> {
    /// # let model: Llama = unimplemented!();
    /// # let cfg = Config::config_7b_v1(false);
    /// # let mut cache = Cache::new(true, DType::F32, &cfg, &Device::Cpu)?;
    /// let ids = Tensor::zeros((1, 8), DType::U32, &Device::Cpu)?;
    /// let logits = model.forward(&ids, 0, &mut cache)?;
    /// assert_eq!(logits.dims()[0], 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    /// Loads model weights from a [`VarBuilder`] using the given configuration.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::llama::{Config, Llama};
    /// # use candle_nn::VarBuilder;
    /// # fn main() -> candle::Result<()> {
    /// # let vb: VarBuilder = unimplemented!();
    /// let cfg = Config::config_7b_v1(false);
    /// let model = Llama::load(vb, &cfg)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(wte.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg).unwrap())
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }
}

//! Mixtral Model, based on the Mistral architecture
//!
//! See Mistral and Mixtral at:
//! - [Hugging Face](https://huggingface.co/docs/transformers/model_doc/mixtral)
//! - [GitHub](https://github.com/mistralai/mistral-src)
//!

use crate::models::with_tracing::{linear_no_bias, Linear, RmsNorm};
/// Mistral LLM, https://github.com/mistralai/mistral-src
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{kv_cache::KvCache, Activation, VarBuilder};
use std::collections::HashMap;
use std::sync::Arc;

fn default_num_attention_heads() -> usize {
    32
}

fn default_use_flash_attn() -> bool {
    false
}

fn default_hidden_act() -> candle_nn::Activation {
    candle_nn::Activation::Silu
}

/// Mistral/Mixtral model configuration.
///
/// All fields mirror the HuggingFace `config.json` naming so that a config
/// downloaded from the Hub can be deserialized directly.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    pub head_dim: Option<usize>,
    pub num_key_value_heads: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    #[serde(default = "default_use_flash_attn")]
    pub use_flash_attn: bool,
}

impl Config {
    /// Returns the preset configuration for Mistral-7B-v0.1.
    ///
    /// # Example
    /// ```
    /// use candle_transformers::models::mistral::Config;
    /// let cfg = Config::config_7b_v0_1(false);
    /// assert_eq!(cfg.hidden_size, 4096);
    /// assert_eq!(cfg.num_key_value_heads, 8);
    /// ```
    // https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
    pub fn config_7b_v0_1(use_flash_attn: bool) -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            head_dim: None,
            num_key_value_heads: 8,
            hidden_act: Activation::Silu,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            sliding_window: Some(4096),
            use_flash_attn,
        }
    }

    /// Returns the preset configuration for OpenHermes-2.5-Mistral (ChatML prompt format).
    ///
    /// # Example
    /// ```
    /// use candle_transformers::models::mistral::Config;
    /// let cfg = Config::config_chat_ml(false);
    /// assert_eq!(cfg.vocab_size, 32002);
    /// ```
    // https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca/blob/main/config.json
    pub fn config_chat_ml(use_flash_attn: bool) -> Self {
        Self {
            vocab_size: 32002,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            head_dim: None,
            num_key_value_heads: 8,
            hidden_act: Activation::Silu,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            sliding_window: Some(4096),
            use_flash_attn,
        }
    }

    /// Returns the preset configuration for amazon/MistralLite.
    ///
    /// Extended vocabulary (32003 tokens) for Amazon's fine-tuned variant.
    ///
    /// # Example
    /// ```
    /// use candle_transformers::models::mistral::Config;
    /// let cfg = Config::config_amazon_mistral_lite(false);
    /// assert_eq!(cfg.vocab_size, 32003);
    /// ```
    // https://huggingface.co/amazon/MistralLite/blob/main/config.json
    pub fn config_amazon_mistral_lite(use_flash_attn: bool) -> Self {
        Self {
            vocab_size: 32003,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            head_dim: None,
            num_key_value_heads: 8,
            hidden_act: Activation::Silu,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            sliding_window: Some(4096),
            use_flash_attn,
        }
    }

    fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let rope_theta = cfg.rope_theta as f32;
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    _use_flash_attn: bool,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim();
        let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            kv_cache: KvCache::new(2, cfg.max_position_embeddings),
            _use_flash_attn: cfg.use_flash_attn,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        if seqlen_offset == 0 {
            self.kv_cache.reset();
        }
        let (key_states, value_states) = self.kv_cache.append(&key_states.contiguous()?, &value_states.contiguous()?)?;

        let key_states = crate::utils::repeat_kv(key_states, self.num_kv_groups)?;
        let value_states = crate::utils::repeat_kv(value_states, self.num_kv_groups)?;

        let scale = 1f32 / (self.head_dim as f32).sqrt();
        let attn_output =
            candle_nn::ops::sdpa(&query_states, &key_states, &value_states, attention_mask, false, scale, 1.)?;
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset()
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

/// The Mistral/Mixtral causal language model.
///
/// Supports both prompt-processing (prefill) and autoregressive generation
/// (decode) via an internal per-layer KV cache.  Call [`clear_kv_cache`]
/// between independent generation requests.
///
/// [`clear_kv_cache`]: Model::clear_kv_cache
#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    sliding_window: Option<usize>,
    device: Device,
    dtype: DType,
    masks: HashMap<(usize, usize), Tensor>,
}

impl Model {
    /// Constructs a new `Model` by loading weights from `vb`.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::mistral::{Config, Model};
    /// # use candle_nn::VarBuilder;
    /// # fn main() -> candle::Result<()> {
    /// # let vb: VarBuilder = unimplemented!();
    /// let model = Model::new(&Config::config_7b_v0_1(false), vb)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            masks: HashMap::new(),
        })
    }

    fn prepare_decoder_attention_mask(
        &mut self,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let key = (tgt_len, seqlen_offset);
        if let Some(mask) = self.masks.get(&key) {
            return Ok(mask.clone());
        }
        let sliding_window = self.sliding_window.unwrap_or(tgt_len + 1);
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        let mask = mask
            .expand((1, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)?;
        self.masks.insert(key, mask.clone());
        Ok(mask)
    }

    /// Returns a reference to the token-embedding table.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::mistral::Model;
    /// # fn main() {
    /// # let model: Model = unimplemented!();
    /// let emb = model.embed_tokens();
    /// # }
    /// ```
    pub fn embed_tokens(&self) -> &candle_nn::Embedding {
        &self.embed_tokens
    }

    /// Runs a forward pass given token ids.
    ///
    /// Returns logits of shape `(batch, vocab_size)` for the last token position.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::mistral::Model;
    /// # use candle::{Device, DType, Tensor};
    /// # fn main() -> candle::Result<()> {
    /// # let mut model: Model = unimplemented!();
    /// let ids = Tensor::zeros((1, 8), DType::U32, &Device::Cpu)?;
    /// let logits = model.forward(&ids, 0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(seq_len, seqlen_offset)?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }

    /// Runs a forward pass given pre-computed input embeddings.
    ///
    /// Used by multi-modal models that inject non-text embeddings (e.g. images)
    /// before passing them to the main transformer.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::mistral::Model;
    /// # use candle::{Device, DType, Tensor};
    /// # fn main() -> candle::Result<()> {
    /// # let mut model: Model = unimplemented!();
    /// let embeds = Tensor::zeros((1, 8, 4096), DType::F32, &Device::Cpu)?;
    /// let logits = model.forward_embeds(&embeds, None, 0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward_embeds(
        &mut self,
        xs: &Tensor,
        attn_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (_b_size, seq_len, _) = xs.dims3()?;
        let mut xs = xs.clone();
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attn_mask, seqlen_offset)?
        }
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }

    /// Resets the KV cache of every layer (call between independent generation requests).
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::mistral::Model;
    /// # fn main() {
    /// # let mut model: Model = unimplemented!();
    /// model.clear_kv_cache();
    /// # }
    /// ```
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

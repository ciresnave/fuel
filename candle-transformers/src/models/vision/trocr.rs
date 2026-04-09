//! TrOCR model implementation.
//!
//! TrOCR is a Transformer-based OCR model that uses a Vision Transformer encoder
//! and a BART-like decoder for optical character recognition.
//!
//! Key characteristics:
//! - Vision Transformer encoder for image processing
//! - BART-style decoder for text generation
//! - Learned positional embeddings
//! - Layer normalization and self-attention
//!
//! References:
//! - [Paper](https://arxiv.org/abs/2109.10282)
//! - [Model Card](https://huggingface.co/microsoft/trocr-base-handwritten)
//!

use crate::models::vit::{Config, Embeddings, Encoder};
use candle::{DType, Result, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_nn::{
    embedding, layer_norm, linear_no_bias, Embedding, LayerNorm, Linear, Module, VarBuilder,
};

fn default_tie_word_embeddings() -> bool {
    true
}
fn default_use_learned_position_embeddings() -> bool {
    true
}

/// Configuration for the TrOCR decoder (cross-attention transformer).
///
/// # Example
///
/// ```
/// use candle_transformers::models::trocr::TrOCRConfig;
/// let cfg = TrOCRConfig::default();
/// assert_eq!(cfg.vocab_size, 50265);
/// ```
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct TrOCRConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub cross_attention_hidden_size: usize,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub activation_function: candle_nn::Activation,
    pub max_position_embeddings: usize,
    pub dropout: f64,
    pub attention_dropout: f64,
    pub activation_dropout: f64,
    pub decoder_start_token_id: u32,
    pub init_std: f64,
    pub decoder_layerdrop: f64,
    pub use_cache: bool,
    pub scale_embedding: bool,
    pub pad_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: u32,
    pub decoder_vocab_size: Option<usize>,
    #[serde(default = "default_use_learned_position_embeddings")]
    pub use_learned_position_embeddings: bool,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

impl Default for TrOCRConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            d_model: 1024,
            cross_attention_hidden_size: 768,
            decoder_layers: 12,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            activation_function: candle_nn::Activation::Gelu,
            max_position_embeddings: 512,
            dropout: 0.1,
            attention_dropout: 0.0,
            activation_dropout: 0.0,
            decoder_start_token_id: 2,
            init_std: 0.02,
            decoder_layerdrop: 0.0,
            use_cache: true,
            scale_embedding: false,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            decoder_vocab_size: Some(50265),
            use_learned_position_embeddings: true,
            tie_word_embeddings: true,
        }
    }
}

#[derive(Debug, Clone)]
struct TrOCRLearnedPositionalEmbedding {
    offset: usize,
    weights: Embedding,
}

impl TrOCRLearnedPositionalEmbedding {
    fn load(vb: VarBuilder, cfg: &TrOCRConfig) -> Result<Self> {
        let offset: usize = 2;
        let num_embeddings = cfg.max_position_embeddings;
        let embedding_dim = cfg.d_model;
        let weights = embedding(num_embeddings + offset, embedding_dim, vb)?;

        Ok(Self { offset, weights })
    }

    fn new_sinusoidal(vb: VarBuilder, cfg: &TrOCRConfig) -> Result<Self> {
        // https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/src/transformers/models/trocr/modeling_trocr.py#L81
        let embedding_dim = cfg.d_model;
        let half_dim = embedding_dim / 2;
        let num_positions = cfg.max_position_embeddings + cfg.pad_token_id + 1;
        let dev = vb.device();
        let inv_freq: Vec<_> = (0..half_dim)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / (half_dim - 1) as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, num_positions as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((num_positions, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let emb = Tensor::cat(&[freqs.sin()?, freqs.cos()?], 1)?;
        let emb = Tensor::cat(
            &[
                emb.narrow(0, 0, cfg.pad_token_id)?,
                Tensor::zeros((1, embedding_dim), DType::F32, dev)?,
                emb.narrow(0, cfg.pad_token_id + 1, cfg.max_position_embeddings)?,
            ],
            0,
        )?
        .contiguous()?;
        let emb = Embedding::new(emb, embedding_dim);
        Ok(Self {
            offset: cfg.pad_token_id + 1,
            weights: emb,
        })
    }

    fn forward(&mut self, input_ids: &Tensor, past_key_values_length: u32) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;

        let positions = Tensor::arange(
            past_key_values_length,
            seq_len as u32 + past_key_values_length,
            input_ids.device(),
        )?
        .expand((b_sz, seq_len))?;

        let positions =
            positions.broadcast_add(&Tensor::new(self.offset as u32, input_ids.device())?)?;
        self.weights.forward(&positions)
    }
}

#[derive(Debug, Clone)]
struct TrOCRAttention {
    head_dim: usize,
    num_heads: usize,
    is_decoder: bool,
    scaling: f64,
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    kv_cache: KvCache,
}

impl TrOCRAttention {
    fn load(
        vb: VarBuilder,
        cfg: &TrOCRConfig,
        kdim: Option<usize>,
        vdim: Option<usize>,
    ) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let num_heads = cfg.decoder_attention_heads;
        let head_dim = embed_dim / num_heads;
        let kdim = kdim.unwrap_or(embed_dim);
        let vdim = vdim.unwrap_or(embed_dim);

        let k_proj = linear_no_bias(kdim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(vdim, embed_dim, vb.pp("v_proj"))?;
        let q_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("q_proj"))?;

        let out_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self {
            head_dim,
            num_heads,
            is_decoder: true,
            scaling: 1. / (head_dim as f64).sqrt(),
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            kv_cache: KvCache::new(2, cfg.max_position_embeddings),
        })
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache.reset()
    }

    fn _shape(&self, tensor: &Tensor, bsz: usize) -> Result<Tensor> {
        tensor
            .reshape((bsz, (), self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        kv_states: Option<&Tensor>,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, tgt_len, _) = xs.dims3()?;
        let query_states = (xs.apply(&self.q_proj)? * self.scaling)?;
        let (key_states, value_states) = match kv_states {
            None => {
                let key_states = self._shape(&xs.apply(&self.k_proj)?, b_sz)?;
                let value_states = self._shape(&xs.apply(&self.v_proj)?, b_sz)?;
                if self.is_decoder {
                    self.kv_cache
                        .append(&key_states.contiguous()?, &value_states.contiguous()?)?
                } else {
                    (key_states, value_states)
                }
            }
            Some(kv_states) => {
                let key_states = self._shape(&kv_states.apply(&self.k_proj)?, b_sz)?;
                let value_states = self._shape(&kv_states.apply(&self.v_proj)?, b_sz)?;
                (key_states, value_states)
            }
        };
        let proj_shape = (b_sz * self.num_heads, (), self.head_dim);
        let query_states = self._shape(&query_states, b_sz)?.reshape(proj_shape)?;
        let key_states = key_states.reshape(proj_shape)?;
        let value_states = value_states.reshape(proj_shape)?;
        let attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;
        let attn_weights = match attn_mask {
            None => attn_weights,
            Some(attn_mask) => attn_weights.broadcast_add(attn_mask)?,
        };
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_probs.matmul(&value_states)?;
        attn_output
            .reshape((b_sz, self.num_heads, tgt_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, tgt_len, self.head_dim * self.num_heads))?
            .apply(&self.out_proj)
    }
}

#[derive(Debug, Clone)]
struct TrOCRDecoderLayer {
    self_attn: TrOCRAttention,
    activation_fn: candle_nn::Activation,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: TrOCRAttention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl TrOCRDecoderLayer {
    fn load(vb: VarBuilder, cfg: &TrOCRConfig) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let self_attn = TrOCRAttention::load(vb.pp("self_attn"), cfg, None, None)?;
        let self_attn_layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let encoder_attn = TrOCRAttention::load(
            vb.pp("encoder_attn"),
            cfg,
            Some(cfg.cross_attention_hidden_size),
            Some(cfg.cross_attention_hidden_size),
        )?;
        let encoder_attn_layer_norm =
            layer_norm(embed_dim, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
        let fc1 = linear_no_bias(embed_dim, cfg.decoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear_no_bias(cfg.decoder_ffn_dim, embed_dim, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            activation_fn: cfg.activation_function,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    fn reset_kv_cache(&mut self) {
        self.self_attn.reset_kv_cache();
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.self_attn.forward(xs, None, Some(attention_mask))?;
        let xs = (xs + residual)?;
        let mut xs = self.self_attn_layer_norm.forward(&xs)?;

        if let Some(encoder_hidden_states) = &encoder_hidden_states {
            let residual = xs.clone();
            let encoder_attention_mask = attention_mask.clone(); // TODO
            xs = self.encoder_attn.forward(
                &xs,
                Some(encoder_hidden_states),
                Some(&encoder_attention_mask),
            )?;
            xs = (xs + residual)?;
            xs = self.encoder_attn_layer_norm.forward(&xs)?
        }

        let residual = xs.clone();
        let xs = self.fc1.forward(&xs)?;
        let xs = self.activation_fn.forward(&xs)?;
        let xs = self.fc2.forward(&xs)?;
        let xs = (xs + residual)?;
        let xs = self.final_layer_norm.forward(&xs)?;

        Ok(xs)
    }
}

/// The TrOCR auto-regressive transformer decoder.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::trocr::{TrOCRConfig, TrOCRModel};
/// # use candle_transformers::models::vit::Config;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// # let enc_cfg: Config = unimplemented!();
/// // TrOCRDecoder is constructed internally by TrOCRModel::new.
/// let model = TrOCRModel::new(&enc_cfg, &TrOCRConfig::default(), vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct TrOCRDecoder {
    layers: Vec<TrOCRDecoderLayer>,
    embed_scale: Option<f64>,
    embed_tokens: Embedding,
    embed_positions: TrOCRLearnedPositionalEmbedding,
}

impl TrOCRDecoder {
    fn new(cfg: &TrOCRConfig, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("decoder.model.decoder");

        let embed_tokens = embedding(cfg.vocab_size, cfg.d_model, vb.pp("embed_tokens"))?;
        let embed_positions = if cfg.use_learned_position_embeddings {
            TrOCRLearnedPositionalEmbedding::load(vb.pp("embed_positions"), cfg)?
        } else {
            TrOCRLearnedPositionalEmbedding::new_sinusoidal(vb.pp("embed_positions"), cfg)?
        };
        let mut layers = Vec::with_capacity(cfg.decoder_layers);
        let vb_l = vb.pp("layers");
        for idx in 0..cfg.decoder_layers {
            let layer = TrOCRDecoderLayer::load(vb_l.pp(idx), cfg)?;
            layers.push(layer)
        }
        let embed_scale = if cfg.scale_embedding {
            Some((cfg.d_model as f64).sqrt())
        } else {
            None
        };

        Ok(Self {
            layers,
            embed_scale,
            embed_tokens,
            embed_positions,
        })
    }

    fn reset_kv_cache(&mut self) {
        self.layers.iter_mut().for_each(|l| l.reset_kv_cache())
    }

    /// Run the decoder forward pass and return hidden states.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::trocr::{TrOCRConfig, TrOCRModel};
    /// # use candle_transformers::models::vit::Config;
    /// # use candle::{Device, DType, Tensor};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// # let enc_cfg: Config = unimplemented!();
    /// let mut model = TrOCRModel::new(&enc_cfg, &TrOCRConfig::default(), vb)?;
    /// let decoder = model.decoder();
    /// // Call decoder.forward(...) with appropriate tensors.
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn forward(
        &mut self,
        xs: &Tensor,
        encoder_xs: Option<&Tensor>,
        past_kv_len: usize,
        attn_mask: &Tensor,
    ) -> Result<Tensor> {
        let embed_pos = self.embed_positions.forward(xs, past_kv_len as u32)?;
        let xs = xs.apply(&self.embed_tokens)?;

        let xs = match self.embed_scale {
            None => xs,
            Some(scale) => (xs * scale)?,
        };

        let mut xs = xs.broadcast_add(&embed_pos)?;

        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attn_mask, encoder_xs)?;
        }
        Ok(xs)
    }
}

/// The TrOCR vision encoder built on a pretrained ViT-style backbone.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::trocr::TrOCREncoder;
/// # use candle_transformers::models::vit::Config;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// # let enc_cfg: Config = unimplemented!();
/// let encoder = TrOCREncoder::new(&enc_cfg, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct TrOCREncoder {
    embeddings: Embeddings,
    encoder: Encoder,
    layernorm: LayerNorm,
}

impl TrOCREncoder {
    /// Create a new encoder from a vision config and variable store.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::trocr::TrOCREncoder;
    /// # use candle_transformers::models::vit::Config;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// # let enc_cfg: Config = unimplemented!();
    /// let encoder = TrOCREncoder::new(&enc_cfg, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_v = vb.pp("encoder");

        let embeddings = Embeddings::new(cfg, false, vb_v.pp("embeddings"))?;

        let encoder = Encoder::new(cfg, vb_v.pp("encoder"))?;
        let layernorm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb_v.pp("layernorm"))?;

        Ok(Self {
            embeddings,
            encoder,
            layernorm,
        })
    }

    /// Encode an image tensor and return contextualized visual features.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::trocr::TrOCREncoder;
    /// # use candle_transformers::models::vit::Config;
    /// # use candle::{Device, DType, Tensor};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// # let enc_cfg: Config = unimplemented!();
    /// let encoder = TrOCREncoder::new(&enc_cfg, vb)?;
    /// let img = Tensor::zeros((1, 3, 384, 384), DType::F32, &Device::Cpu)?;
    /// let feats = encoder.forward(&img)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(xs, None, false)?;
        let encoder_outputs = self.encoder.forward(&embedding_output)?;

        self.layernorm.forward(&encoder_outputs)
    }
}

/// TrOCR decoder head that wraps `TrOCRDecoder` and applies the output projection.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::trocr::{TrOCRConfig, TrOCRForCausalLM};
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// let head = TrOCRForCausalLM::new(&TrOCRConfig::default(), vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct TrOCRForCausalLM {
    decoder: TrOCRDecoder,
    output_projection: Linear,
}

impl TrOCRForCausalLM {
    /// Create a new causal LM decoder head from config and variable store.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::trocr::{TrOCRConfig, TrOCRForCausalLM};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let head = TrOCRForCausalLM::new(&TrOCRConfig::default(), vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(decoder_cfg: &TrOCRConfig, vb: VarBuilder) -> Result<Self> {
        let decoder = TrOCRDecoder::new(decoder_cfg, vb.clone())?;
        let output_projection = if decoder_cfg.tie_word_embeddings {
            candle_nn::Linear::new(decoder.embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(
                decoder_cfg.d_model,
                decoder_cfg.vocab_size,
                vb.pp("decoder.output_projection"),
            )?
        };
        Ok(Self {
            decoder,
            output_projection,
        })
    }

    /// Run the decoder forward pass and return vocabulary logits.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::trocr::{TrOCRConfig, TrOCRForCausalLM};
    /// # use candle::{Device, DType, Tensor};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// let mut head = TrOCRForCausalLM::new(&TrOCRConfig::default(), vb)?;
    /// let ids = Tensor::zeros((1, 5), DType::U32, &Device::Cpu)?;
    /// let mask = Tensor::zeros((5, 5), DType::F32, &Device::Cpu)?;
    /// let logits = head.forward(&ids, None, 0, &mask)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn forward(
        &mut self,
        xs: &Tensor,
        encoder_xs: Option<&Tensor>,
        past_kv_len: usize,
        attn_mask: &Tensor,
    ) -> Result<Tensor> {
        let xs = self
            .decoder
            .forward(xs, encoder_xs, past_kv_len, attn_mask)?;
        let xs = xs.apply(&self.output_projection)?;

        Ok(xs)
    }

    fn reset_kv_cache(&mut self) {
        self.decoder.reset_kv_cache();
    }
}

/// Full TrOCR OCR model combining a vision encoder and a causal language model decoder.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::trocr::{TrOCRConfig, TrOCRModel};
/// # use candle_transformers::models::vit::Config;
/// # use candle_nn::VarBuilder;
/// # let vb: VarBuilder = unimplemented!();
/// # let enc_cfg: Config = unimplemented!();
/// let model = TrOCRModel::new(&enc_cfg, &TrOCRConfig::default(), vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct TrOCRModel {
    encoder: TrOCREncoder,
    decoder: TrOCRForCausalLM,
}

impl TrOCRModel {
    /// Create a new TrOCR model from encoder and decoder configs and a variable store.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::trocr::{TrOCRConfig, TrOCRModel};
    /// # use candle_transformers::models::vit::Config;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// # let enc_cfg: Config = unimplemented!();
    /// let model = TrOCRModel::new(&enc_cfg, &TrOCRConfig::default(), vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(encoder_cfg: &Config, decoder_cfg: &TrOCRConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = TrOCREncoder::new(encoder_cfg, vb.clone())?;
        let decoder = TrOCRForCausalLM::new(decoder_cfg, vb)?;
        Ok(Self { encoder, decoder })
    }

    /// Return a mutable reference to the encoder.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::trocr::{TrOCRConfig, TrOCRModel};
    /// # use candle_transformers::models::vit::Config;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// # let enc_cfg: Config = unimplemented!();
    /// let mut model = TrOCRModel::new(&enc_cfg, &TrOCRConfig::default(), vb)?;
    /// let _enc = model.encoder();
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn encoder(&mut self) -> &mut TrOCREncoder {
        &mut self.encoder
    }

    /// Return a mutable reference to the decoder head.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::trocr::{TrOCRConfig, TrOCRModel};
    /// # use candle_transformers::models::vit::Config;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// # let enc_cfg: Config = unimplemented!();
    /// let mut model = TrOCRModel::new(&enc_cfg, &TrOCRConfig::default(), vb)?;
    /// let _dec = model.decoder();
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn decoder(&mut self) -> &mut TrOCRForCausalLM {
        &mut self.decoder
    }

    /// Decode `xs` with cross-attention over `encoder_xs` and return logits.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::trocr::{TrOCRConfig, TrOCRModel};
    /// # use candle_transformers::models::vit::Config;
    /// # use candle::{Device, DType, Tensor};
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// # let enc_cfg: Config = unimplemented!();
    /// let mut model = TrOCRModel::new(&enc_cfg, &TrOCRConfig::default(), vb)?;
    /// let ids = Tensor::zeros((1, 5), DType::U32, &Device::Cpu)?;
    /// let enc_out = Tensor::zeros((1, 577, 768), DType::F32, &Device::Cpu)?;
    /// let logits = model.decode(&ids, &enc_out, 0)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn decode(
        &mut self,
        xs: &Tensor,
        encoder_xs: &Tensor,
        past_kv_len: usize,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(1)?;
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_vec(mask, (seq_len, seq_len), xs.device())?;

        self.decoder
            .forward(xs, Some(encoder_xs), past_kv_len, &mask)
    }

    /// Reset the decoder's key-value cache (e.g., between inference calls).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::trocr::{TrOCRConfig, TrOCRModel};
    /// # use candle_transformers::models::vit::Config;
    /// # use candle_nn::VarBuilder;
    /// # let vb: VarBuilder = unimplemented!();
    /// # let enc_cfg: Config = unimplemented!();
    /// let mut model = TrOCRModel::new(&enc_cfg, &TrOCRConfig::default(), vb)?;
    /// model.reset_kv_cache();
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn reset_kv_cache(&mut self) {
        self.decoder.reset_kv_cache();
    }
}

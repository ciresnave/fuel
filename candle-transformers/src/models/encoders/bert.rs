//! BERT (Bidirectional Encoder Representations from Transformers)
//!
//! Bert is a general large language model that can be used for various language tasks:
//! - Compute sentence embeddings for a prompt.
//! - Compute similarities between a set of sentences.
//! - [Arxiv](https://arxiv.org/abs/1810.04805) "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
//! - Upstream [GitHub repo](https://github.com/google-research/bert).
//! - See bert in [candle-examples](https://github.com/huggingface/candle/tree/main/candle-examples/) for runnable code
//!
use super::with_tracing::{layer_norm, linear, LayerNorm, Linear};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use serde::Deserialize;

pub const DTYPE: DType = DType::F32;

/// Activation function used inside BERT's feed-forward intermediate layer.
///
/// # Example
/// ```
/// use candle_transformers::models::bert::HiddenAct;
/// let act = HiddenAct::Gelu;
/// assert_eq!(act, HiddenAct::Gelu);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    /// GeLU with exact `erf`-based computation.
    Gelu,
    /// Faster approximate GeLU using a `tanh` approximation.
    GeluApproximate,
    /// Rectified Linear Unit.
    Relu,
}

#[derive(Clone)]
struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }

    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::GeluApproximate => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

/// Strategy used to encode token position information.
///
/// # Example
/// ```
/// use candle_transformers::models::bert::PositionEmbeddingType;
/// let pos = PositionEmbeddingType::Absolute;
/// assert_eq!(pos, PositionEmbeddingType::default());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    /// Standard learned absolute position embeddings added to the input.
    #[default]
    Absolute,
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/configuration_bert.py#L1
/// BERT model configuration.
///
/// All fields mirror the HuggingFace `BertConfig` naming so that a
/// `config.json` downloaded from the Hub can be deserialized directly.
///
/// # Example
/// ```
/// use candle_transformers::models::bert::Config;
/// let cfg = Config::default();
/// assert_eq!(cfg.vocab_size, 30522);
/// assert_eq!(cfg.hidden_size, 768);
/// assert_eq!(cfg.num_hidden_layers, 12);
/// ```
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    pub use_cache: bool,
    pub classifier_dropout: Option<f64>,
    pub model_type: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        }
    }
}

impl Config {
    fn _all_mini_lm_l6_v2() -> Self {
        // https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/config.json
        Self {
            vocab_size: 30522,
            hidden_size: 384,
            num_hidden_layers: 6,
            num_attention_heads: 12,
            intermediate_size: 1536,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        }
    }
}

#[derive(Clone)]
struct Dropout {
    #[allow(dead_code)]
    pr: f64,
}

impl Dropout {
    fn new(pr: f64) -> Self {
        Self { pr }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L180
struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl BertEmbeddings {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_bsize, seq_len) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (&input_embeddings + token_type_embeddings)?;
        if let Some(position_embeddings) = &self.position_embeddings {
            // TODO: Proper absolute positions?
            let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
            let position_ids = Tensor::new(&position_ids[..], input_ids.device())?;
            embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?
        }
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings)?;
        Ok(embeddings)
    }
}

#[derive(Clone)]
struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
    span_softmax: tracing::Span,
}

impl BertSelfAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_scores = attention_scores.broadcast_add(attention_mask)?;
        let attention_probs = {
            let _enter_sm = self.span_softmax.enter();
            candle_nn::ops::softmax(&attention_scores, candle::D::Minus1)?
        };
        let attention_probs = self.dropout.forward(&attention_probs)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle::D::Minus2)?;
        Ok(context_layer)
    }
}

#[derive(Clone)]
struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl BertSelfOutput {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "self-out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L392
#[derive(Clone)]
struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
    span: tracing::Span,
}

impl BertAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let self_attention = BertSelfAttention::load(vb.pp("self"), config)?;
        let self_output = BertSelfOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            self_attention,
            self_output,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let self_outputs = self.self_attention.forward(hidden_states, attention_mask)?;
        let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L441
#[derive(Clone)]
struct BertIntermediate {
    dense: Linear,
    intermediate_act: HiddenActLayer,
    span: tracing::Span,
}

impl BertIntermediate {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act: HiddenActLayer::new(config.hidden_act),
            span: tracing::span!(tracing::Level::TRACE, "inter"),
        })
    }
}

impl Module for BertIntermediate {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L456
#[derive(Clone)]
struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl BertOutput {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L470
/// A single BERT transformer layer: self-attention followed by a position-wise FFN.
///
/// Each layer applies:
/// 1. Multi-head self-attention with residual connection and layer norm.
/// 2. A two-layer feed-forward network with residual connection and layer norm.
#[derive(Clone)]
pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
    span: tracing::Span,
}

impl BertLayer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = BertAttention::load(vb.pp("attention"), config)?;
        let intermediate = BertIntermediate::load(vb.pp("intermediate"), config)?;
        let output = BertOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        // TODO: Support cross-attention?
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L523
        // TODO: Support something similar to `apply_chunking_to_forward`?
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L556
/// The BERT encoder: a stack of [`BertLayer`] transformer blocks.
///
/// # Example
/// ```no_run
/// # use candle_transformers::models::bert::{BertEncoder, Config};
/// # use candle_nn::VarBuilder;
/// # fn main() -> candle::Result<()> {
/// # let vb: VarBuilder = unimplemented!();
/// let config = Config::default();
/// let encoder = BertEncoder::load(vb, &config)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct BertEncoder {
    pub layers: Vec<BertLayer>,
    span: tracing::Span,
}

impl BertEncoder {
    /// Loads all encoder layers from a `VarBuilder` scoped to `bert.encoder`.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::bert::{BertEncoder, Config};
    /// # use candle_nn::VarBuilder;
    /// # fn main() -> candle::Result<()> {
    /// # let vb: VarBuilder = unimplemented!();
    /// let config = Config::default();
    /// let encoder = BertEncoder::load(vb, &config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| BertLayer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(BertEncoder { layers, span })
    }

    /// Passes `hidden_states` through all encoder layers applying `attention_mask`.
    ///
    /// # Arguments
    /// * `hidden_states` - Input of shape `[batch, seq_len, hidden_size]`.
    /// * `attention_mask` - Extended mask of shape `[batch, 1, 1, seq_len]` where
    ///   masked positions contain a large negative value.
    ///
    /// # Returns
    /// Contextualised representations of shape `[batch, seq_len, hidden_size]`.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::bert::{BertEncoder, Config};
    /// # use candle_nn::VarBuilder;
    /// # use candle::{Device, DType, Tensor};
    /// # fn main() -> candle::Result<()> {
    /// # let vb: VarBuilder = unimplemented!();
    /// # let encoder = BertEncoder::load(vb, &Config::default())?;
    /// let hidden = Tensor::zeros((1, 8, 768), DType::F32, &Device::Cpu)?;
    /// let mask = Tensor::zeros((1, 1, 1, 8), DType::F32, &Device::Cpu)?;
    /// let out = encoder.forward(&hidden, &mask)?;
    /// assert_eq!(out.dims(), &[1, 8, 768]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut hidden_states = hidden_states.clone();
        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_mask)?
        }
        Ok(hidden_states)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L874
/// The base BERT model (no task-specific head).
///
/// Maps token ids and token-type ids to a sequence of contextualised hidden
/// states.  An optional attention mask can be passed to ignore padding tokens.
///
/// # Example
/// ```no_run
/// # use candle_transformers::models::bert::{BertModel, Config};
/// # use candle_nn::VarBuilder;
/// # fn main() -> candle::Result<()> {
/// # let vb: VarBuilder = unimplemented!();
/// let config = Config::default();
/// let model = BertModel::load(vb, &config)?;
/// # Ok(())
/// # }
/// ```
pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pub device: Device,
    span: tracing::Span,
}

impl BertModel {
    /// Loads the BERT model weights from a `VarBuilder`.
    ///
    /// Supports both flat weight layouts (e.g. `bert-base-uncased`) and
    /// model-type-prefixed layouts (e.g. `roberta.embeddings`).
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::bert::{BertModel, Config};
    /// # use candle_nn::VarBuilder;
    /// # fn main() -> candle::Result<()> {
    /// # let vb: VarBuilder = unimplemented!();
    /// let config = Config::default();
    /// let model = BertModel::load(vb, &config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let (embeddings, encoder) = match (
            BertEmbeddings::load(vb.pp("embeddings"), config),
            BertEncoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let Some(model_type) = &config.model_type {
                    if let (Ok(embeddings), Ok(encoder)) = (
                        BertEmbeddings::load(vb.pp(format!("{model_type}.embeddings")), config),
                        BertEncoder::load(vb.pp(format!("{model_type}.encoder")), config),
                    ) {
                        (embeddings, encoder)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    /// Encodes a batch of token-id sequences into contextualised hidden states.
    ///
    /// # Arguments
    /// * `input_ids` - Token ids of shape `[batch, seq_len]`.
    /// * `token_type_ids` - Segment ids of shape `[batch, seq_len]` (all zeros for
    ///   single-sentence tasks).
    /// * `attention_mask` - Optional binary mask of shape `[batch, seq_len]`;
    ///   `1` for real tokens, `0` for padding. Defaults to all-ones when `None`.
    ///
    /// # Returns
    /// Sequence of hidden states of shape `[batch, seq_len, hidden_size]`.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::bert::{BertModel, Config};
    /// # use candle_nn::VarBuilder;
    /// # use candle::{Device, DType, Tensor};
    /// # fn main() -> candle::Result<()> {
    /// # let vb: VarBuilder = unimplemented!();
    /// # let model = BertModel::load(vb, &Config::default())?;
    /// let input_ids = Tensor::zeros((1, 8), DType::U32, &Device::Cpu)?;
    /// let type_ids = Tensor::zeros((1, 8), DType::U32, &Device::Cpu)?;
    /// let out = model.forward(&input_ids, &type_ids, None)?;
    /// assert_eq!(out.dims(), &[1, 8, 768]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        let attention_mask = match attention_mask {
            Some(attention_mask) => attention_mask.clone(),
            None => input_ids.ones_like()?,
        };
        let dtype = embedding_output.dtype();
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L995
        let attention_mask = get_extended_attention_mask(&attention_mask, dtype)?;
        let sequence_output = self.encoder.forward(&embedding_output, &attention_mask)?;
        Ok(sequence_output)
    }
}

fn get_extended_attention_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let attention_mask = match attention_mask.rank() {
        3 => attention_mask.unsqueeze(1)?,
        2 => attention_mask.unsqueeze(1)?.unsqueeze(1)?,
        _ => candle::bail!("Wrong shape for input_ids or attention_mask"),
    };
    let attention_mask = attention_mask.to_dtype(dtype)?;
    // torch.finfo(dtype).min
    (attention_mask.ones_like()? - &attention_mask)?.broadcast_mul(
        &Tensor::try_from(f32::MIN)?
            .to_device(attention_mask.device())?
            .to_dtype(dtype)?,
    )
}

//https://github.com/huggingface/transformers/blob/1bd604d11c405dfb8b78bda4062d88fc75c17de0/src/transformers/models/bert/modeling_bert.py#L752-L766
struct BertPredictionHeadTransform {
    dense: Linear,
    activation: HiddenActLayer,
    layer_norm: LayerNorm,
}

impl BertPredictionHeadTransform {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let activation = HiddenActLayer::new(config.hidden_act);
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            dense,
            activation,
            layer_norm,
        })
    }
}

impl Module for BertPredictionHeadTransform {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self
            .activation
            .forward(&self.dense.forward(hidden_states)?)?;
        self.layer_norm.forward(&hidden_states)
    }
}

// https://github.com/huggingface/transformers/blob/1bd604d11c405dfb8b78bda4062d88fc75c17de0/src/transformers/models/bert/modeling_bert.py#L769C1-L790C1
/// The masked-language-modelling prediction head: a dense transform followed by a
/// linear projection from `hidden_size` to `vocab_size`.
///
/// # Example
/// ```no_run
/// # use candle_transformers::models::bert::{BertLMPredictionHead, Config};
/// # use candle_nn::VarBuilder;
/// # fn main() -> candle::Result<()> {
/// # let vb: VarBuilder = unimplemented!();
/// let head = BertLMPredictionHead::load(vb, &Config::default())?;
/// # Ok(())
/// # }
/// ```
pub struct BertLMPredictionHead {
    transform: BertPredictionHeadTransform,
    decoder: Linear,
}

impl BertLMPredictionHead {
    /// Loads the MLM prediction head weights from `cls.predictions`.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::bert::{BertLMPredictionHead, Config};
    /// # use candle_nn::VarBuilder;
    /// # fn main() -> candle::Result<()> {
    /// # let vb: VarBuilder = unimplemented!();
    /// let head = BertLMPredictionHead::load(vb, &Config::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let transform = BertPredictionHeadTransform::load(vb.pp("transform"), config)?;
        let decoder = linear(config.hidden_size, config.vocab_size, vb.pp("decoder"))?;
        Ok(Self { transform, decoder })
    }
}

impl Module for BertLMPredictionHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.decoder
            .forward(&self.transform.forward(hidden_states)?)
    }
}

// https://github.com/huggingface/transformers/blob/1bd604d11c405dfb8b78bda4062d88fc75c17de0/src/transformers/models/bert/modeling_bert.py#L792
/// The complete BERT MLM head wrapping [`BertLMPredictionHead`].
///
/// Suitable for masked-language-modelling fine-tuning or as a feature
/// extractor when only the `predictions` sub-module is needed.
///
/// # Example
/// ```no_run
/// # use candle_transformers::models::bert::{BertOnlyMLMHead, Config};
/// # use candle_nn::VarBuilder;
/// # fn main() -> candle::Result<()> {
/// # let vb: VarBuilder = unimplemented!();
/// let head = BertOnlyMLMHead::load(vb, &Config::default())?;
/// # Ok(())
/// # }
/// ```
pub struct BertOnlyMLMHead {
    predictions: BertLMPredictionHead,
}

impl BertOnlyMLMHead {
    /// Loads the MLM head from `cls`.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::bert::{BertOnlyMLMHead, Config};
    /// # use candle_nn::VarBuilder;
    /// # fn main() -> candle::Result<()> {
    /// # let vb: VarBuilder = unimplemented!();
    /// let head = BertOnlyMLMHead::load(vb, &Config::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let predictions = BertLMPredictionHead::load(vb.pp("predictions"), config)?;
        Ok(Self { predictions })
    }
}

impl Module for BertOnlyMLMHead {
    fn forward(&self, sequence_output: &Tensor) -> Result<Tensor> {
        self.predictions.forward(sequence_output)
    }
}

/// Complete BERT model with a masked-language-modelling head.
///
/// Wraps [`BertModel`] with a [`BertOnlyMLMHead`] to produce per-token
/// vocabulary logits, suitable for fine-tuning on masked-language-modelling.
///
/// # Example
/// ```no_run
/// # use candle_transformers::models::bert::{BertForMaskedLM, Config};
/// # use candle_nn::VarBuilder;
/// # fn main() -> candle::Result<()> {
/// # let vb: VarBuilder = unimplemented!();
/// let model = BertForMaskedLM::load(vb, &Config::default())?;
/// # Ok(())
/// # }
/// ```
pub struct BertForMaskedLM {
    bert: BertModel,
    cls: BertOnlyMLMHead,
}

impl BertForMaskedLM {
    /// Loads the full BERT-for-MLM model from a `VarBuilder`.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::bert::{BertForMaskedLM, Config};
    /// # use candle_nn::VarBuilder;
    /// # fn main() -> candle::Result<()> {
    /// # let vb: VarBuilder = unimplemented!();
    /// let model = BertForMaskedLM::load(vb, &Config::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let bert = BertModel::load(vb.pp("bert"), config)?;
        let cls = BertOnlyMLMHead::load(vb.pp("cls"), config)?;
        Ok(Self { bert, cls })
    }

    /// Runs a forward pass and returns per-token vocabulary logits.
    ///
    /// # Arguments
    /// * `input_ids` - Token ids of shape `[batch, seq_len]`.
    /// * `token_type_ids` - Segment ids of shape `[batch, seq_len]`.
    /// * `attention_mask` - Optional binary mask; `None` attends to all tokens.
    ///
    /// # Returns
    /// Logits of shape `[batch, seq_len, vocab_size]`.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_transformers::models::bert::{BertForMaskedLM, Config};
    /// # use candle_nn::VarBuilder;
    /// # use candle::{Device, DType, Tensor};
    /// # fn main() -> candle::Result<()> {
    /// # let vb: VarBuilder = unimplemented!();
    /// # let model = BertForMaskedLM::load(vb, &Config::default())?;
    /// let ids = Tensor::zeros((1, 8), DType::U32, &Device::Cpu)?;
    /// let type_ids = Tensor::zeros_like(&ids)?;
    /// let logits = model.forward(&ids, &type_ids, None)?;
    /// assert_eq!(logits.dims()[0..2], [1, 8]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let sequence_output = self
            .bert
            .forward(input_ids, token_type_ids, attention_mask)?;
        self.cls.forward(&sequence_output)
    }
}

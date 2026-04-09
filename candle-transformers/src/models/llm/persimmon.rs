//! Persimmon Model
//!
//! A transformer language model for efficient inference and general-purpose tasks. The model uses a standard transformer architecture with:
//! - Layer normalization for Q/K attention
//! - RoPE embeddings with partial rotary factor
//! - ReLU activation
//! - Separate number of attention heads and KV heads
//!
//! References:
//! - 💻 [Hugging Face Implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/persimmon/modeling_persimmon.py)
//! - 💻 [Persimmon Config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/persimmon/configuration_persimmon.py)
//! - 🤗 [Hugging Face](https://huggingface.co/adept/persimmon-8b-base)
//!

use candle::DType;
use serde::Deserialize;

/// Default dtype for the Persimmon model weights.
pub const DTYPE: DType = DType::F32;

/// Position embedding strategy for the Persimmon model.
///
/// # Example
///
/// ```
/// use candle_transformers::models::persimmon::PositionEmbeddingType;
/// let emb_type = PositionEmbeddingType::Absolute;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    Absolute,
    Alibi,
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/persimmon/configuration_persimmon.py
/// Configuration for the Persimmon language model.
///
/// # Example
///
/// ```
/// use candle_transformers::models::persimmon::Config;
/// let cfg = Config::base_8b();
/// assert_eq!(cfg.vocab_size, 262144);
/// ```
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: candle_nn::Activation,
    pub max_position_embeddings: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub rms_norm_eps: f64,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub qk_layernorm: bool,
    pub partial_rotary_factor: f64,
}

impl Config {
    /// Return the default configuration for Persimmon-8B-Base.
    ///
    /// # Example
    ///
    /// ```
    /// use candle_transformers::models::persimmon::Config;
    /// let cfg = Config::base_8b();
    /// assert_eq!(cfg.vocab_size, 262144);
    /// assert_eq!(cfg.num_hidden_layers, 36);
    /// assert_eq!(cfg.hidden_size, 4096);
    /// ```
    pub fn base_8b() -> Self {
        // https://huggingface.co/adept/persimmon-8b-base/blob/main/config.json
        Self {
            hidden_act: candle_nn::Activation::Relu,
            hidden_size: 4096,
            initializer_range: 0.02,
            intermediate_size: 16384,
            layer_norm_eps: 1e-05,
            max_position_embeddings: 16384,
            num_attention_heads: 64,
            num_hidden_layers: 36,
            num_key_value_heads: 64,
            qk_layernorm: true,
            rms_norm_eps: 1e-06,
            rope_theta: 25000.0,
            tie_word_embeddings: false,
            use_cache: true,
            vocab_size: 262144,
            partial_rotary_factor: 0.5,
        }
    }
}

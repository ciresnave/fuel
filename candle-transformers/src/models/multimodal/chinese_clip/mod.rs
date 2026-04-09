//! Chinese contrastive Language-Image Pre-Training
//!
//! Chinese contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
//! pairs of images with related texts.
//!
//! - 💻 [GH Link](https://github.com/OFA-Sys/Chinese-CLIP)
//! - 💻 Transformers Python [reference implementation](https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/chinese_clip/modeling_chinese_clip.py)
//!
use candle::{Module, Result, Tensor, D};
use candle_nn as nn;

use text_model::ChineseClipTextTransformer;
use vision_model::ChineseClipVisionTransformer;

pub mod text_model;
pub mod vision_model;

/// Activation functions available in the Chinese CLIP model.
///
/// # Example
/// ```
/// use candle::Tensor;
/// use candle_transformers::models::chinese_clip::Activation;
/// let act = Activation::Gelu;
/// ```
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    QuickGelu,
    Gelu,
    GeluNew,
    Relu,
}

impl From<String> for Activation {
    fn from(value: String) -> Self {
        match value.as_str() {
            "quick_gelu" => Activation::QuickGelu,
            "gelu" => Activation::Gelu,
            "gelu_new" => Activation::GeluNew,
            "relu" => Activation::Relu,
            _ => panic!("Invalid activation function: {value}"),
        }
    }
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Activation::QuickGelu => xs * nn::ops::sigmoid(&(xs * 1.702f64)?)?,
            Activation::Gelu => xs.gelu_erf(),
            Activation::GeluNew => xs.gelu(),
            Activation::Relu => xs.relu(),
        }
    }
}

/// Top-level configuration for a Chinese CLIP model combining text and vision encoders.
///
/// # Example
/// ```
/// use candle_transformers::models::chinese_clip::ChineseClipConfig;
/// let config = ChineseClipConfig::clip_vit_base_patch16();
/// assert_eq!(config.projection_dim, 512);
/// ```
#[derive(Clone, Debug)]
pub struct ChineseClipConfig {
    pub text_config: text_model::ChineseClipTextConfig,
    pub vision_config: vision_model::ChineseClipVisionConfig,
    pub projection_dim: usize,
    pub logit_scale_init_value: f32,
    pub image_size: usize,
}

impl ChineseClipConfig {
    /// referer: <https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/blob/main/config.json>
    /// Returns the default Chinese CLIP ViT-B/16 configuration.
    ///
    /// # Example
    /// ```
    /// use candle_transformers::models::chinese_clip::ChineseClipConfig;
    /// let config = ChineseClipConfig::clip_vit_base_patch16();
    /// assert_eq!(config.projection_dim, 512);
    /// ```
    pub fn clip_vit_base_patch16() -> Self {
        let text_config = text_model::ChineseClipTextConfig::clip_vit_base_patch16();
        let vision_config = vision_model::ChineseClipVisionConfig::clip_vit_base_patch16();

        Self {
            text_config,
            vision_config,
            projection_dim: 512,
            logit_scale_init_value: 2.6592,
            image_size: 512,
        }
    }
}

/// A unified encoder configuration that holds either text or vision settings.
///
/// # Example
/// ```
/// use candle_transformers::models::chinese_clip::{EncoderConfig, ChineseClipConfig};
/// let cfg = ChineseClipConfig::clip_vit_base_patch16();
/// let enc = EncoderConfig::Vision(cfg.vision_config);
/// assert_eq!(enc.num_hidden_layers(), 12);
/// ```
#[derive(Clone, Debug)]
pub enum EncoderConfig {
    Text(text_model::ChineseClipTextConfig),
    Vision(vision_model::ChineseClipVisionConfig),
}

impl EncoderConfig {
    /// Returns the embedding dimension of the underlying text or vision config.
    ///
    /// # Example
    /// ```
    /// use candle_transformers::models::chinese_clip::{EncoderConfig, ChineseClipConfig};
    /// let cfg = ChineseClipConfig::clip_vit_base_patch16();
    /// let enc = EncoderConfig::Text(cfg.text_config);
    /// assert_eq!(enc.embed_dim(), 768);
    /// ```
    pub fn embed_dim(&self) -> usize {
        match self {
            Self::Text(c) => c.hidden_size,
            Self::Vision(c) => c.hidden_size,
        }
    }

    /// Returns the number of attention heads in the underlying config.
    pub fn num_attention_heads(&self) -> usize {
        match self {
            Self::Text(c) => c.num_attention_heads,
            Self::Vision(c) => c.num_attention_heads,
        }
    }

    /// Returns the intermediate (FFN) hidden size in the underlying config.
    pub fn intermediate_size(&self) -> usize {
        match self {
            Self::Text(c) => c.intermediate_size,
            Self::Vision(c) => c.intermediate_size,
        }
    }

    /// Returns the number of transformer encoder layers in the underlying config.
    pub fn num_hidden_layers(&self) -> usize {
        match self {
            Self::Text(c) => c.num_hidden_layers,
            Self::Vision(c) => c.num_hidden_layers,
        }
    }

    /// Returns the activation function specified in the underlying config.
    pub fn activation(&self) -> Activation {
        match self {
            Self::Text(c) => c.hidden_act,
            Self::Vision(c) => c.hidden_act,
        }
    }

    /// Returns the layer-norm epsilon from the underlying config.
    pub fn layer_norm_eps(&self) -> f64 {
        match self {
            Self::Text(c) => c.layer_norm_eps,
            Self::Vision(c) => c.layer_norm_eps,
        }
    }
}

/// The full Chinese CLIP model with text and vision encoders and dual projection heads.
///
/// # Example
/// ```no_run
/// use candle::Device;
/// use candle_nn::VarBuilder;
/// use candle_transformers::models::chinese_clip::{ChineseClipModel, ChineseClipConfig};
/// let config = ChineseClipConfig::clip_vit_base_patch16();
/// let vb = VarBuilder::zeros(candle::DType::F32, &Device::Cpu);
/// let model = ChineseClipModel::new(vb, &config)?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct ChineseClipModel {
    text_model: ChineseClipTextTransformer,
    vision_model: ChineseClipVisionTransformer,
    visual_projection: nn::Linear,
    text_projection: nn::Linear,
    logit_scale: Tensor,
}

impl ChineseClipModel {
    /// Creates a new `ChineseClipModel` loading weights from `vs`.
    ///
    /// # Example
    /// ```no_run
    /// use candle::Device;
    /// use candle_nn::VarBuilder;
    /// use candle_transformers::models::chinese_clip::{ChineseClipModel, ChineseClipConfig};
    /// let config = ChineseClipConfig::clip_vit_base_patch16();
    /// let vb = VarBuilder::zeros(candle::DType::F32, &Device::Cpu);
    /// let model = ChineseClipModel::new(vb, &config)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(vs: nn::VarBuilder, c: &ChineseClipConfig) -> Result<Self> {
        let text_model = ChineseClipTextTransformer::new(vs.pp("text_model"), &c.text_config)?;

        let vision_model =
            ChineseClipVisionTransformer::new(vs.pp("vision_model"), &c.vision_config)?;

        let vision_embed_dim = c.vision_config.hidden_size;
        let vision_projection = nn::linear_no_bias(
            vision_embed_dim,
            c.projection_dim,
            vs.pp("visual_projection"),
        )?;

        let text_embed_dim = c.text_config.hidden_size;
        let text_projection =
            nn::linear_no_bias(text_embed_dim, c.projection_dim, vs.pp("text_projection"))?;

        let logit_scale = if vs.contains_tensor("logit_scale") {
            vs.get(&[], "logit_scale")?
        } else {
            Tensor::new(&[c.logit_scale_init_value], vs.device())?
        };

        Ok(Self {
            text_model,
            vision_model,
            visual_projection: vision_projection,
            text_projection,
            logit_scale,
        })
    }

    /// Encodes text tokens and projects them to the shared embedding space.
    ///
    /// Returns L2-unnormalised embeddings; pass the output through [`div_l2_norm`] to normalise.
    pub fn get_text_features(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let output = self
            .text_model
            .forward(input_ids, token_type_ids, attention_mask)?
            .contiguous()?;
        self.text_projection.forward(&output)
    }

    /// Encodes an image and projects it to the shared embedding space.
    ///
    /// Returns L2-unnormalised embeddings; pass the output through [`div_l2_norm`] to normalise.
    pub fn get_image_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        pixel_values
            .apply(&self.vision_model)?
            .apply(&self.visual_projection)
    }

    /// Computes cross-modal similarity logits for a batch of images and text tokens.
    ///
    /// Returns `(logits_per_text, logits_per_image)` where each tensor has shape `[text, image]`
    /// and `[image, text]` respectively, scaled by a learnable `logit_scale`.
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let image_features = self.get_image_features(pixel_values)?;
        let text_features = self.get_text_features(input_ids, token_type_ids, attention_mask)?;

        let image_features_normalized = div_l2_norm(&image_features)?;
        let text_features_normalized = div_l2_norm(&text_features)?;

        let logits_per_text = text_features_normalized.matmul(&image_features_normalized.t()?)?;
        let logit_scale = self.logit_scale.exp()?;
        let logits_per_text = logits_per_text.broadcast_mul(&logit_scale)?;
        let logits_per_image = logits_per_text.t()?;
        Ok((logits_per_text, logits_per_image))
    }
}

/// Divides each row of `v` by its L2 norm, producing unit-length vectors.
///
/// # Example
/// ```
/// # fn main() -> candle::Result<()> {
/// use candle::{Device, Tensor};
/// use candle_transformers::models::chinese_clip::div_l2_norm;
/// let v = Tensor::new(&[[3f32, 4f32]], &Device::Cpu)?;
/// let normed = div_l2_norm(&v)?;
/// let vals = normed.to_vec2::<f32>()?;
/// // L2 norm of [3, 4] is 5; result should be [0.6, 0.8]
/// assert!((vals[0][0] - 0.6).abs() < 1e-5);
/// # Ok(())
/// # }
/// ```
pub fn div_l2_norm(v: &Tensor) -> Result<Tensor> {
    let l2_norm = v.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    v.broadcast_div(&l2_norm)
}

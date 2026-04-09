//! Colpali Model for text/image similarity scoring.
//!
//! Colpali combines a vision encoder with an efficient LM for retrieving content.
//!

use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::paligemma;
use candle_nn::{linear, Linear};

/// ColPali model combining PaliGemma with a projection head for retrieval.
///
/// # Example
///
/// ```no_run
/// # use candle_transformers::models::colpali::Model;
/// # use candle_transformers::models::paligemma::Config;
/// # use candle_nn::VarBuilder;
/// # let config: Config = unimplemented!();
/// # let vb: VarBuilder = unimplemented!();
/// let model = Model::new(&config, vb)?;
/// # Ok::<(), candle::Error>(())
/// ```
pub struct Model {
    pub model: paligemma::Model,
    pub custom_text_projection: Linear,
}

impl Model {
    /// Create a new ColPali model from the given PaliGemma config and variable store.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_transformers::models::colpali::Model;
    /// # use candle_transformers::models::paligemma::Config;
    /// # use candle_nn::VarBuilder;
    /// # let config: Config = unimplemented!();
    /// # let vb: VarBuilder = unimplemented!();
    /// let model = Model::new(&config, vb)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn new(config: &paligemma::Config, vb: VarBuilder) -> Result<Self> {
        let model = paligemma::Model::new(config, vb.pp("model"))?;
        let custom_text_projection = linear(
            config.text_config.hidden_size,
            128,
            vb.pp("custom_text_proj"),
        )?;

        Ok(Self {
            model,
            custom_text_projection,
        })
    }

    /// Embed images into L2-normalised dense vectors for retrieval.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle::{DType, Device, Tensor};
    /// # use candle_transformers::models::colpali::Model;
    /// # let mut model: Model = unimplemented!();
    /// let pixel_values = Tensor::zeros((1, 3, 448, 448), DType::F32, &Device::Cpu)?;
    /// let input_ids = Tensor::zeros((1, 256), DType::U32, &Device::Cpu)?;
    /// let embeddings = model.forward_images(&pixel_values, &input_ids)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn forward_images(&mut self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        let outputs = self
            .model
            .setup_without_projection(pixel_values, input_ids)?;
        let outputs = self.custom_text_projection.forward(&outputs)?;
        let outputs = outputs.broadcast_div(&outputs.sqr()?.sum_keepdim(2)?.sqrt()?)?;
        Ok(outputs)
    }

    /// Embed text tokens into L2-normalised dense vectors for retrieval.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle::{DType, Device, Tensor};
    /// # use candle_transformers::models::colpali::Model;
    /// # let mut model: Model = unimplemented!();
    /// let input_ids = Tensor::zeros((1, 32), DType::U32, &Device::Cpu)?;
    /// let embeddings = model.forward_text(&input_ids)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn forward_text(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let outputs = self.model.forward_without_projection(input_ids)?;
        let outputs = self.custom_text_projection.forward(&outputs)?;
        let outputs = outputs.broadcast_div(&outputs.sqr()?.sum_keepdim(2)?.sqrt()?)?;
        Ok(outputs)
    }
}

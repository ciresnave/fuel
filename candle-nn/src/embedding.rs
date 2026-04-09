//! Embedding layer.
//!
//! Maps integer token indices to dense vectors by looking up rows in a weight matrix.
use candle::{Result, Tensor};

/// A simple lookup-table embedding layer.
///
/// Stores an embedding matrix of shape `[vocab_size, hidden_size]` and retrieves rows
/// corresponding to the input token indices. The [`Module`](crate::Module) implementation
/// accepts an index tensor of any shape and appends `hidden_size` as the last dimension.
///
/// # Examples
///
/// ```
/// use candle::{Tensor, Device, DType};
/// use candle_nn::{Embedding, Module};
///
/// // Vocabulary of 5 tokens, embedding dimension 3
/// let weights = Tensor::randn(0f32, 1.0, (5, 3), &Device::Cpu)?;
/// let emb = Embedding::new(weights, 3);
///
/// let token_ids = Tensor::new(&[0u32, 3, 1], &Device::Cpu)?;
/// let output = emb.forward(&token_ids)?;
/// assert_eq!(output.dims(), &[3, 3]);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    /// Create an embedding layer from an existing weight tensor.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - The weight matrix of shape `[vocab_size, hidden_size]`.
    /// * `hidden_size` - The embedding dimension (must match the second axis of `embeddings`).
    pub fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    /// Return a reference to the underlying embedding weight tensor.
    pub fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }

    /// Return the hidden size (embedding dimension).
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl crate::Module for Embedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        use candle::Context;
        let vocab_size = self.embeddings.dim(0).unwrap_or(0);
        let input_shape = indexes.shape().clone();
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = self
            .embeddings
            .index_select(&indexes, 0)
            .with_context(|| {
                format!(
                    "Embedding(vocab={vocab_size}, hidden={}): indices shape {input_shape:?}",
                    self.hidden_size,
                )
            })?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

/// Create an [`Embedding`] layer using a [`VarBuilder`](crate::VarBuilder).
///
/// Initializes the embedding weight matrix with random normal values (mean 0, stdev 1)
/// and stores it under the name `"weight"` in the variable builder.
///
/// # Arguments
///
/// * `in_size` - Vocabulary size (number of embedding rows).
/// * `out_size` - Hidden size (embedding dimension).
/// * `vb` - Variable builder used to create or load the weight tensor.
pub fn embedding(in_size: usize, out_size: usize, vb: crate::VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get_with_hints(
        (in_size, out_size),
        "weight",
        crate::Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
    )?;
    Ok(Embedding::new(embeddings, out_size))
}

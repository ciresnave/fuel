//! Sequential Layer
//!
//! A sequential layer used to chain multiple layers and closures.
use candle::{Module, Result, Tensor};

/// A sequential layer combining multiple other layers.
///
/// Layers are applied in the order they were added. The output of each layer becomes
/// the input to the next. Build a `Sequential` using the [`seq`] function, then
/// chain layers with [`add`](Self::add) or closures with [`add_fn`](Self::add_fn).
///
/// # Example
///
/// ```rust
/// use candle::{Tensor, Device, Module};
/// use candle_nn::seq;
///
/// let model = seq()
///     .add_fn(|x| x.relu())
///     .add_fn(|x| x * 2.0);
/// let x = Tensor::new(&[-1.0f32, 0.0, 1.0], &Device::Cpu)?;
/// let y = model.forward(&x)?;
/// assert_eq!(y.to_vec1::<f32>()?, &[0.0, 0.0, 2.0]);
/// # Ok::<(), candle::Error>(())
/// ```
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

/// Creates a new empty sequential layer.
///
/// Use the returned [`Sequential`] to chain layers via [`add`](Sequential::add)
/// or [`add_fn`](Sequential::add_fn).
///
/// # Example
///
/// ```rust
/// use candle::{Tensor, Device, Module};
/// use candle_nn::seq;
///
/// let model = seq().add_fn(|x| x.relu());
/// let x = Tensor::new(&[-2.0f32, -1.0, 0.0, 1.0], &Device::Cpu)?;
/// let y = model.forward(&x)?;
/// assert_eq!(y.to_vec1::<f32>()?, &[0.0, 0.0, 0.0, 1.0]);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn seq() -> Sequential {
    Sequential { layers: vec![] }
}

impl Sequential {
    /// The number of sub-layers embedded in this layer.
    pub fn len(&self) -> i64 {
        self.layers.len() as i64
    }

    /// Returns true if this layer does not have any sub-layer.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?
        }
        Ok(xs)
    }
}

impl Sequential {
    /// Appends a layer after all the current layers.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor) -> Result<Tensor> + Send + Sync,
    {
        self.add(super::func(f))
    }

    /// Applies the forward pass and returns the output for each layer.
    pub fn forward_all(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let mut vec = Vec::with_capacity(self.layers.len());
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?;
            vec.push(xs.clone())
        }
        Ok(vec)
    }
}

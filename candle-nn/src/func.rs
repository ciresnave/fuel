//! Layers defined by closures.
//!
//! [`Func`] and [`FuncT`] wrap closures so they can be used anywhere a [`Module`](super::Module)
//! or [`ModuleT`](super::ModuleT) is expected. This is handy for ad-hoc layers like activation
//! functions or reshape operations that don't need learnable parameters.
use candle::{Result, Tensor};
use std::sync::Arc;

/// A layer defined by a simple closure, implementing [`Module`](super::Module).
///
/// This is useful for wrapping stateless operations (activations, reshapes, etc.) so they
/// can be composed with other `Module` layers in a sequential pipeline.
///
/// # Example
///
/// ```rust
/// use candle::{Tensor, Device, DType};
/// use candle_nn::{Func, Module};
///
/// let relu = Func::new(|xs| xs.relu());
/// let input = Tensor::new(&[-1.0f32, 0.0, 1.0], &Device::Cpu)?;
/// let output = relu.forward(&input)?;
/// assert_eq!(output.to_vec1::<f32>()?, &[0.0, 0.0, 1.0]);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Clone)]
pub struct Func<'a> {
    #[allow(clippy::type_complexity)]
    f: Arc<dyn 'a + Fn(&Tensor) -> Result<Tensor> + Send + Sync>,
}

impl std::fmt::Debug for Func<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "func")
    }
}

/// Creates a [`Func`] from a closure. This is a convenience shorthand for [`Func::new`].
pub fn func<'a, F>(f: F) -> Func<'a>
where
    F: 'a + Fn(&Tensor) -> Result<Tensor> + Send + Sync,
{
    Func { f: Arc::new(f) }
}

impl super::Module for Func<'_> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        (*self.f)(xs)
    }
}

impl<'a> Func<'a> {
    /// Creates a new `Func` from the given closure.
    pub fn new<F>(f: F) -> Self
    where
        F: 'a + Fn(&Tensor) -> Result<Tensor> + Send + Sync,
    {
        Self { f: Arc::new(f) }
    }
}

/// A layer defined by a closure that also receives a `train` flag, implementing
/// [`ModuleT`](super::ModuleT).
///
/// This is the training-aware counterpart of [`Func`]. The boolean `train` parameter
/// lets the closure behave differently during training and evaluation (e.g. applying
/// dropout only during training).
#[derive(Clone)]
pub struct FuncT<'a> {
    #[allow(clippy::type_complexity)]
    f: Arc<dyn 'a + Fn(&Tensor, bool) -> Result<Tensor> + Send + Sync>,
}

impl std::fmt::Debug for FuncT<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "func")
    }
}

/// Creates a [`FuncT`] from a closure. This is a convenience shorthand for [`FuncT::new`].
pub fn func_t<'a, F>(f: F) -> FuncT<'a>
where
    F: 'a + Fn(&Tensor, bool) -> Result<Tensor> + Send + Sync,
{
    FuncT { f: Arc::new(f) }
}

impl super::ModuleT for FuncT<'_> {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        (*self.f)(xs, train)
    }
}

impl<'a> FuncT<'a> {
    /// Creates a new `FuncT` from the given closure that receives a `train` flag.
    pub fn new<F>(f: F) -> Self
    where
        F: 'a + Fn(&Tensor, bool) -> Result<Tensor> + Send + Sync,
    {
        Self { f: Arc::new(f) }
    }
}

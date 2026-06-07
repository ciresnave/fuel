//! Lazy-graph Module wrappers.
//!
//! Port of `fuel-nn` over `LazyTensor`. Each module is a thin wrapper
//! that holds weights and implements `LazyModule::forward`, delegating
//! to the matching `LazyTensor` primitive. The eager `Module` trait in
//! `fuel-core` is built around `Tensor`; this module mirrors that
//! shape for `LazyTensor` so downstream lazy ports can build their
//! layer graphs out of named building blocks rather than ad-hoc
//! per-port helpers.

pub mod activation;
pub mod conv;
pub mod embedding;
pub mod init;
pub mod linear;
pub mod lora;
pub mod moe;
pub mod norm;
pub mod quantizable_linear;
pub mod sampling;
pub mod sequential;

pub use activation::{
    LazyElu, LazyGelu, LazyGeluPytorchTanh, LazyLeakyRelu, LazyRelu, LazySigmoid, LazySilu,
    LazyTanh,
};
pub use conv::{LazyConv1d, LazyConv1dConfig, LazyConv2d, LazyConv2dConfig};
pub use embedding::LazyEmbedding;
pub use linear::{LazyLinear, linear, linear_no_bias};
pub use lora::LazyLoraLinear;
pub use moe::{LazyMoeExpert, LazyMoeLayer, LazyMoeRouter};
pub use norm::{LazyBatchNorm2d, LazyGroupNorm, LazyLayerNorm, LazyRmsNorm};
pub use quantizable_linear::LazyQuantizableLinear;
pub use sequential::LazySequential;

use crate::Result;
use crate::lazy::LazyTensor;

/// Single-input `forward` over the lazy-graph tensor. Analogous to
/// the eager [`crate::Module`] trait, retargeted at `LazyTensor`.
pub trait LazyModule {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor>;
}

impl<F: Fn(&LazyTensor) -> Result<LazyTensor>> LazyModule for F {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        self(xs)
    }
}

impl<M: LazyModule> LazyModule for Option<&M> {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}

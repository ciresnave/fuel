//! Lazy-graph Module wrappers.
//!
//! Port of `fuel-nn` over `LazyTensor`. Each module is a thin wrapper
//! that holds weights and implements `LazyModule::forward`, delegating
//! to the matching `LazyTensor` primitive. The eager `Module` trait in
//! `fuel-core` is built around `Tensor`; this module mirrors that
//! shape for `LazyTensor` so downstream lazy ports can build their
//! layer graphs out of named building blocks rather than ad-hoc
//! per-port helpers.
//!
//! Sub-port 1: `LazyModule` trait + `LazyLinear` + `LazyEmbedding`.
//! Conv / norm / activation / sequential / lora / moe / sampling
//! ship as separate sub-ports.

pub mod embedding;
pub mod linear;

pub use embedding::LazyEmbedding;
pub use linear::LazyLinear;

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

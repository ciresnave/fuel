// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lazy-graph Module wrappers.
//!
//! Port of `fuel-nn` over `Tensor`. Each module is a thin wrapper
//! that holds weights and implements `Module::forward`, delegating
//! to the matching `Tensor` primitive. The eager `Module` trait in
//! `fuel-core` is built around `Tensor`; this module mirrors that
//! shape for `Tensor` so downstream lazy ports can build their
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
pub mod two_proj_attention;

pub use activation::{Elu, Gelu, GeluPytorchTanh, LeakyRelu, Relu, Sigmoid, Silu, Tanh};
pub use conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
pub use embedding::Embedding;
pub use linear::{Linear, linear, linear_no_bias};
pub use lora::LoraLinear;
pub use moe::{MoeExpert, MoeLayer, MoeRouter};
pub use norm::{BatchNorm2d, GroupNorm, LayerNorm, RmsNorm};
pub use quantizable_linear::QuantizableLinear;
pub use sequential::Sequential;
pub use two_proj_attention::TwoProjAttention;

use fuel_core::Result;
use fuel_core::lazy::Tensor;

/// Single-input `forward` over the lazy-graph tensor. Analogous to
/// the eager [`crate::Module`] trait, retargeted at `Tensor`.
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

impl<F: Fn(&Tensor) -> Result<Tensor>> Module for F {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self(xs)
    }
}

impl<M: Module> Module for Option<&M> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}

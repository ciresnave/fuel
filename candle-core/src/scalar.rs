//! Scalar values and the [`TensorOrScalar`] trait for ops that accept either tensors or scalars.
//!
//! The [`Scalar`] enum wraps a single typed value, and [`TensorOrScalar`] enables
//! functions like [`Tensor::maximum`] to accept both `Tensor` and numeric arguments.

// Scalar enum, its methods, and `impl<T: WithDType> From<T> for Scalar` are all
// provided by candle-core-types.
pub use candle_core_types::scalar::*;

use crate::{Result, Tensor, WithDType};

/// The result of converting a [`TensorOrScalar`] input: either a full tensor or a
/// scalar promoted to a single-element tensor.
pub enum TensorScalar {
    /// The input was already a tensor.
    Tensor(Tensor),
    /// The input was a scalar, converted to a 0-d tensor.
    Scalar(Tensor),
}

/// Trait for function arguments that accept either a [`Tensor`] reference or a scalar value.
///
/// This allows operations like [`Tensor::cmp`] and [`Tensor::where_cond`] to accept
/// both tensors and numeric literals as operands.
pub trait TensorOrScalar {
    /// Converts this value into a [`TensorScalar`].
    fn to_tensor_scalar(self) -> Result<TensorScalar>;
}

impl TensorOrScalar for &Tensor {
    fn to_tensor_scalar(self) -> Result<TensorScalar> {
        Ok(TensorScalar::Tensor(self.clone()))
    }
}

impl<T: WithDType> TensorOrScalar for T {
    fn to_tensor_scalar(self) -> Result<TensorScalar> {
        let scalar = Tensor::new(self, &crate::Device::Cpu)?;
        Ok(TensorScalar::Scalar(scalar))
    }
}

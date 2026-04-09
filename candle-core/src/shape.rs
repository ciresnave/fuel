//! Shapes describe the dimensionality of tensors.
//!
//! This module re-exports shape types from `candle-core-types` and adds
//! [`Tensor`](crate::Tensor) dimension-extraction convenience methods.
//!
//! ```rust
//! use candle_core::Shape;
//! let s = Shape::from((2, 3, 4));
//! assert_eq!(s.rank(), 3);
//! assert_eq!(s.elem_count(), 24);
//! assert_eq!(s.dims(), &[2, 3, 4]);
//! ```

// Re-export all shape types, traits, and free functions from the types crate.
pub use candle_core_types::shape::*;

// ---------------------------------------------------------------------------
// Tensor dimension-extraction methods
//
// These mirror the Shape::dimsN() methods on Tensor for convenience.
// The `?` operator handles the candle_core_types::Error → candle_core::Error
// conversion via the From impl in error.rs.
// ---------------------------------------------------------------------------

impl crate::Tensor {
    /// Extracts dimensions from a scalar (rank-0) tensor.
    ///
    /// Returns an error if the tensor does not have exactly 0 dimensions.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let t = Tensor::new(42f32, &Device::Cpu)?;
    /// assert_eq!(t.dims0()?, ());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn dims0(&self) -> crate::Result<()> {
        Ok(self.shape().dims0()?)
    }

    /// Extracts the single dimension from a rank-1 tensor.
    ///
    /// Returns an error if the tensor does not have exactly 1 dimension.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let t = Tensor::zeros(5, DType::F32, &Device::Cpu)?;
    /// assert_eq!(t.dims1()?, 5);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn dims1(&self) -> crate::Result<usize> {
        Ok(self.shape().dims1()?)
    }

    /// Extracts the two dimensions from a rank-2 tensor.
    ///
    /// Returns an error if the tensor does not have exactly 2 dimensions.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let t = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// assert_eq!(t.dims2()?, (2, 3));
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn dims2(&self) -> crate::Result<(usize, usize)> {
        Ok(self.shape().dims2()?)
    }

    /// Extracts the three dimensions from a rank-3 tensor.
    ///
    /// Returns an error if the tensor does not have exactly 3 dimensions.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let t = Tensor::zeros((2, 3, 4), DType::F32, &Device::Cpu)?;
    /// assert_eq!(t.dims3()?, (2, 3, 4));
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn dims3(&self) -> crate::Result<(usize, usize, usize)> {
        Ok(self.shape().dims3()?)
    }

    /// Extracts the four dimensions from a rank-4 tensor.
    ///
    /// Returns an error if the tensor does not have exactly 4 dimensions.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let t = Tensor::zeros((2, 3, 4, 5), DType::F32, &Device::Cpu)?;
    /// assert_eq!(t.dims4()?, (2, 3, 4, 5));
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn dims4(&self) -> crate::Result<(usize, usize, usize, usize)> {
        Ok(self.shape().dims4()?)
    }

    /// Extracts the five dimensions from a rank-5 tensor.
    ///
    /// Returns an error if the tensor does not have exactly 5 dimensions.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let t = Tensor::zeros((2, 3, 4, 5, 6), DType::F32, &Device::Cpu)?;
    /// assert_eq!(t.dims5()?, (2, 3, 4, 5, 6));
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn dims5(&self) -> crate::Result<(usize, usize, usize, usize, usize)> {
        Ok(self.shape().dims5()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride() {
        let shape = Shape::from(());
        assert_eq!(shape.stride_contiguous().to_vec(), Vec::<usize>::new());
        let shape = Shape::from(42);
        assert_eq!(shape.stride_contiguous().to_vec(), [1]);
        let shape = Shape::from((42, 1337));
        assert_eq!(shape.stride_contiguous().to_vec(), [1337, 1]);
        let shape = Shape::from((299, 792, 458));
        assert_eq!(shape.stride_contiguous().to_vec(), [458 * 792, 458, 1]);
    }

    #[test]
    fn test_from_tuple() {
        let shape = Shape::from((2,));
        assert_eq!(shape.dims(), &[2]);
        let shape = Shape::from((2, 3));
        assert_eq!(shape.dims(), &[2, 3]);
        let shape = Shape::from((2, 3, 4));
        assert_eq!(shape.dims(), &[2, 3, 4]);
        let shape = Shape::from((2, 3, 4, 5));
        assert_eq!(shape.dims(), &[2, 3, 4, 5]);
        let shape = Shape::from((2, 3, 4, 5, 6));
        assert_eq!(shape.dims(), &[2, 3, 4, 5, 6]);
        let shape = Shape::from((2, 3, 4, 5, 6, 7));
        assert_eq!(shape.dims(), &[2, 3, 4, 5, 6, 7]);
    }
}

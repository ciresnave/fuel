//! Shapes describe the dimensionality of tensors.
//!
//! This module re-exports shape types from `fuel-core-types` and adds
//! [`Tensor`](crate::Tensor) dimension-extraction convenience methods.
//!
//! ```rust
//! use fuel_core::Shape;
//! let s = Shape::from((2, 3, 4));
//! assert_eq!(s.rank(), 3);
//! assert_eq!(s.elem_count(), 24);
//! assert_eq!(s.dims(), &[2, 3, 4]);
//! ```

// Re-export all shape types, traits, and free functions from the types crate.
pub use fuel_ir::shape::*;



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride() {
        let shape = Shape::from(());
        assert_eq!(shape.stride_contiguous().to_vec(), Vec::<isize>::new());
        let shape = Shape::from(42);
        assert_eq!(shape.stride_contiguous().to_vec(), [1_isize]);
        let shape = Shape::from((42, 1337));
        assert_eq!(shape.stride_contiguous().to_vec(), [1337_isize, 1]);
        let shape = Shape::from((299, 792, 458));
        assert_eq!(shape.stride_contiguous().to_vec(), [458_isize * 792, 458, 1]);
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

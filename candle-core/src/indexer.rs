//! Tensor indexing operations for Python-like slicing syntax.
//!
//! This module provides [`TensorIndexer`], the core enum that powers the
//! [`IndexOp::i`] method on tensors. It supports three styles of indexing:
//!
//! - **Scalar selection** (`usize`) -- picks a single index along one dimension,
//!   removing that dimension from the result.
//! - **Range slicing** (e.g. `0..3`, `2..`, `..`, `..=4`) -- narrows a dimension
//!   to a contiguous sub-range, preserving the dimension.
//! - **Gather / advanced indexing** (`&[u32]`, `Vec<u32>`, or a 1-D `Tensor`) --
//!   selects arbitrary indices along a dimension via [`Tensor::index_select`].
//!
//! Multi-dimensional indexing is expressed with tuples of up to 7 elements:
//!
//! ```
//! # use candle_core::{Tensor, Device, IndexOp};
//! let t = Tensor::arange(0f32, 24f32, &Device::Cpu)?.reshape((2, 3, 4))?;
//!
//! // Scalar select on dim 0 removes that dimension.
//! let row = t.i(0)?;
//! assert_eq!(row.dims(), &[3, 4]);
//!
//! // Range on dim 0, scalar on dim 1.
//! let slice = t.i((0..1, 2))?;
//! assert_eq!(slice.dims(), &[1, 4]);
//!
//! // Gather with a u32 slice on dim 0.
//! let gathered = t.i(&[1u32, 0u32][..])?;
//! assert_eq!(gathered.dims(), &[2, 3, 4]);
//! # Ok::<(), candle_core::Error>(())
//! ```

use crate::{Error, Tensor};
use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

impl Tensor {
    /// Intended to be use by the trait `.i()`
    ///
    /// ```
    /// # use candle_core::{Tensor, DType, Device, IndexOp};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = a.i(0..1)?;
    /// assert_eq!(c.shape().dims(), &[1, 3]);
    ///
    /// let c = a.i(0)?;
    /// assert_eq!(c.shape().dims(), &[3]);
    ///
    /// let c = a.i((.., ..2) )?;
    /// assert_eq!(c.shape().dims(), &[2, 2]);
    ///
    /// let c = a.i((.., ..=2))?;
    /// assert_eq!(c.shape().dims(), &[2, 3]);
    ///
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    fn index(&self, indexers: &[TensorIndexer]) -> Result<Self, Error> {
        let mut x = self.clone();
        let dims = self.shape().dims();
        let mut current_dim = 0;
        for (i, indexer) in indexers.iter().enumerate() {
            x = match indexer {
                TensorIndexer::Select(n) => x.narrow(current_dim, *n, 1)?.squeeze(current_dim)?,
                TensorIndexer::Narrow(left_bound, right_bound) => {
                    let start = match left_bound {
                        Bound::Included(n) => *n,
                        Bound::Excluded(n) => *n + 1,
                        Bound::Unbounded => 0,
                    };
                    let stop = match right_bound {
                        Bound::Included(n) => *n + 1,
                        Bound::Excluded(n) => *n,
                        Bound::Unbounded => dims[i],
                    };
                    let out = x.narrow(current_dim, start, stop.saturating_sub(start))?;
                    current_dim += 1;
                    out
                }
                TensorIndexer::IndexSelect(indexes) => {
                    if indexes.rank() != 1 {
                        crate::bail!("multi-dimensional tensor indexing is not supported")
                    }
                    let out = x.index_select(&indexes.to_device(x.device())?, current_dim)?;
                    current_dim += 1;
                    out
                }
                TensorIndexer::Err(e) => crate::bail!("indexing error {e:?}"),
            };
        }
        Ok(x)
    }
}

/// Describes how to index into a single dimension of a [`Tensor`].
///
/// `TensorIndexer` is not normally constructed directly. Instead, the
/// [`IndexOp::i`] method accepts values that implement `Into<TensorIndexer>`
/// (integers, ranges, slices, and tensors) and converts them automatically.
///
/// # Variants
///
/// | Variant | Created from | Effect |
/// |---------|-------------|--------|
/// | [`Select`](TensorIndexer::Select) | `usize` | Picks one index, removes the dimension |
/// | [`Narrow`](TensorIndexer::Narrow) | Any `Range*<usize>` or `..` | Slices a contiguous sub-range |
/// | [`IndexSelect`](TensorIndexer::IndexSelect) | `&[u32]`, `Vec<u32>`, `&Tensor` | Gathers arbitrary indices via a 1-D index tensor |
///
/// # Examples
///
/// ```
/// # use candle_core::{Tensor, Device, IndexOp};
/// let a = Tensor::arange(0f32, 24f32, &Device::Cpu)?.reshape((2, 3, 4))?;
///
/// // Select -- picks index 0 on dim 0, result shape is [3, 4]
/// let b = a.i(0)?;
/// assert_eq!(b.dims(), &[3, 4]);
///
/// // Narrow -- keeps a range on dim 0, result shape is [1, 3, 4]
/// let c = a.i(0..1)?;
/// assert_eq!(c.dims(), &[1, 3, 4]);
///
/// // IndexSelect -- gathers indices [1, 0] on dim 0
/// let d = a.i(&[1u32, 0u32][..])?;
/// assert_eq!(d.dims(), &[2, 3, 4]);
/// # Ok::<(), candle_core::Error>(())
/// ```
#[derive(Debug)]
pub enum TensorIndexer {
    /// Selects a single index along the current dimension, removing that
    /// dimension from the output shape. Created from a `usize` value.
    Select(usize),

    /// Narrows the current dimension to a contiguous sub-range defined by
    /// start and end bounds. Created from Rust range expressions (`0..3`,
    /// `2..`, `..`, `..=4`, etc.).
    Narrow(Bound<usize>, Bound<usize>),

    /// Gathers arbitrary indices along the current dimension using a 1-D
    /// index tensor. Created from `&[u32]`, `Vec<u32>`, or `&Tensor`.
    /// The index tensor must be rank-1; multi-dimensional index tensors
    /// are not supported.
    IndexSelect(Tensor),

    /// Internal variant that carries a conversion error (e.g. when building
    /// the index tensor from a slice fails). Not constructed by user code.
    Err(Error),
}

/// Converts a `usize` into [`TensorIndexer::Select`], which picks a single
/// index along one dimension and removes that dimension from the result.
impl From<usize> for TensorIndexer {
    fn from(index: usize) -> Self {
        TensorIndexer::Select(index)
    }
}

/// Converts a `&[u32]` slice into [`TensorIndexer::IndexSelect`] by creating
/// a 1-D CPU tensor from the slice.
impl From<&[u32]> for TensorIndexer {
    fn from(index: &[u32]) -> Self {
        match Tensor::new(index, &crate::Device::Cpu) {
            Ok(tensor) => TensorIndexer::IndexSelect(tensor),
            Err(e) => TensorIndexer::Err(e),
        }
    }
}

/// Converts a `Vec<u32>` into [`TensorIndexer::IndexSelect`] by creating
/// a 1-D CPU tensor from the vector contents.
impl From<Vec<u32>> for TensorIndexer {
    fn from(index: Vec<u32>) -> Self {
        let len = index.len();
        match Tensor::from_vec(index, len, &crate::Device::Cpu) {
            Ok(tensor) => TensorIndexer::IndexSelect(tensor),
            Err(e) => TensorIndexer::Err(e),
        }
    }
}

/// Converts a `&Tensor` reference into [`TensorIndexer::IndexSelect`].
/// The tensor is cloned (a cheap reference-count bump).
impl From<&Tensor> for TensorIndexer {
    fn from(tensor: &Tensor) -> Self {
        TensorIndexer::IndexSelect(tensor.clone())
    }
}

trait RB: RangeBounds<usize> {}
impl RB for Range<usize> {}
impl RB for RangeFrom<usize> {}
impl RB for RangeFull {}
impl RB for RangeInclusive<usize> {}
impl RB for RangeTo<usize> {}
impl RB for RangeToInclusive<usize> {}

/// Converts any Rust range type (`Range<usize>`, `RangeFrom<usize>`,
/// `RangeFull`, `RangeInclusive<usize>`, `RangeTo<usize>`,
/// `RangeToInclusive<usize>`) into [`TensorIndexer::Narrow`].
impl<T: RB> From<T> for TensorIndexer {
    fn from(range: T) -> Self {
        use std::ops::Bound::*;
        let start = match range.start_bound() {
            Included(idx) => Included(*idx),
            Excluded(idx) => Excluded(*idx),
            Unbounded => Unbounded,
        };
        let end = match range.end_bound() {
            Included(idx) => Included(*idx),
            Excluded(idx) => Excluded(*idx),
            Unbounded => Unbounded,
        };
        TensorIndexer::Narrow(start, end)
    }
}

/// Trait used to implement multiple signatures for ease of use of the slicing
/// of a tensor
pub trait IndexOp<T> {
    /// Returns a slicing iterator which are the chunks of data necessary to
    /// reconstruct the desired tensor.
    fn i(&self, index: T) -> Result<Tensor, Error>;
}

impl<T> IndexOp<T> for Tensor
where
    T: Into<TensorIndexer>,
{
    ///```rust
    /// use candle_core::{Tensor, DType, Device, IndexOp};
    /// let a = Tensor::new(&[
    ///     [0., 1.],
    ///     [2., 3.],
    ///     [4., 5.]
    /// ], &Device::Cpu)?;
    ///
    /// let b = a.i(0)?;
    /// assert_eq!(b.shape().dims(), &[2]);
    /// assert_eq!(b.to_vec1::<f64>()?, &[0., 1.]);
    ///
    /// let c = a.i(..2)?;
    /// assert_eq!(c.shape().dims(), &[2, 2]);
    /// assert_eq!(c.to_vec2::<f64>()?, &[
    ///     [0., 1.],
    ///     [2., 3.]
    /// ]);
    ///
    /// let d = a.i(1..)?;
    /// assert_eq!(d.shape().dims(), &[2, 2]);
    /// assert_eq!(d.to_vec2::<f64>()?, &[
    ///     [2., 3.],
    ///     [4., 5.]
    /// ]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    fn i(&self, index: T) -> Result<Tensor, Error> {
        self.index(&[index.into()])
    }
}

impl<A> IndexOp<(A,)> for Tensor
where
    A: Into<TensorIndexer>,
{
    ///```rust
    /// use candle_core::{Tensor, DType, Device, IndexOp};
    /// let a = Tensor::new(&[
    ///     [0f32, 1.],
    ///     [2.  , 3.],
    ///     [4.  , 5.]
    /// ], &Device::Cpu)?;
    ///
    /// let b = a.i((0,))?;
    /// assert_eq!(b.shape().dims(), &[2]);
    /// assert_eq!(b.to_vec1::<f32>()?, &[0., 1.]);
    ///
    /// let c = a.i((..2,))?;
    /// assert_eq!(c.shape().dims(), &[2, 2]);
    /// assert_eq!(c.to_vec2::<f32>()?, &[
    ///     [0., 1.],
    ///     [2., 3.]
    /// ]);
    ///
    /// let d = a.i((1..,))?;
    /// assert_eq!(d.shape().dims(), &[2, 2]);
    /// assert_eq!(d.to_vec2::<f32>()?, &[
    ///     [2., 3.],
    ///     [4., 5.]
    /// ]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    fn i(&self, (a,): (A,)) -> Result<Tensor, Error> {
        self.index(&[a.into()])
    }
}
#[allow(non_snake_case)]
impl<A, B> IndexOp<(A, B)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
{
    ///```rust
    /// use candle_core::{Tensor, DType, Device, IndexOp};
    /// let a = Tensor::new(&[[0f32, 1., 2.], [3., 4., 5.], [6., 7., 8.]], &Device::Cpu)?;
    ///
    /// let b = a.i((1, 0))?;
    /// assert_eq!(b.to_vec0::<f32>()?, 3.);
    ///
    /// let c = a.i((..2, 1))?;
    /// assert_eq!(c.shape().dims(), &[2]);
    /// assert_eq!(c.to_vec1::<f32>()?, &[1., 4.]);
    ///
    /// let d = a.i((2.., ..))?;
    /// assert_eq!(d.shape().dims(), &[1, 3]);
    /// assert_eq!(d.to_vec2::<f32>()?, &[[6., 7., 8.]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    fn i(&self, (a, b): (A, B)) -> Result<Tensor, Error> {
        self.index(&[a.into(), b.into()])
    }
}

macro_rules! index_op_tuple {
    ($doc:tt, $($t:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($t),*> IndexOp<($($t,)*)> for Tensor
        where
            $($t: Into<TensorIndexer>,)*
        {
            #[doc=$doc]
            fn i(&self, ($($t,)*): ($($t,)*)) -> Result<Tensor, Error> {
                self.index(&[$($t.into(),)*])
            }
        }
    };
}

index_op_tuple!("see [`IndexOp::i`]", A, B, C);
index_op_tuple!("see [`IndexOp::i`]", A, B, C, D);
index_op_tuple!("see [`IndexOp::i`]", A, B, C, D, E);
index_op_tuple!("see [`IndexOp::i`]", A, B, C, D, E, F);
index_op_tuple!("see [`IndexOp::i`]", A, B, C, D, E, F, G);

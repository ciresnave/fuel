//! Primitives for streaming (incremental) tensor operations.
//!
//! When processing sequential data one chunk at a time (e.g. audio frames or token
//! sequences), modules often need to buffer partial results and emit output only when
//! enough input has accumulated. This module provides the core abstractions for that
//! pattern:
//!
//! - [`StreamTensor`] -- an `Option<Tensor>` wrapper that represents either a real tensor
//!   or the absence of data. It supports concatenation, narrowing, and splitting along an
//!   arbitrary dimension while correctly handling the empty case.
//! - [`StreamingModule`] -- a trait for modules that consume and produce `StreamTensor`
//!   values, with internal state that persists across calls to [`StreamingModule::step`].
//! - [`StreamingBinOp`] -- a streaming binary operator (add, mul, sub, div) that buffers
//!   left/right operands until both sides have matching lengths along the streaming
//!   dimension.
//! - [`BinOp`] -- an enum of supported element-wise binary operations.
//! - [`Map`] -- a simple adapter that wraps any [`crate::Module`] as a
//!   [`StreamingModule`] with no internal buffering.
//!
use crate::{Result, Shape, Tensor};

/// Convenience bound combining [`crate::shape::Dim`] and `Copy` for dimension arguments in
/// streaming operations.
pub trait Dim: crate::shape::Dim + Copy {}
impl<T: crate::shape::Dim + Copy> Dim for T {}

/// A stream tensor is used in streaming module. It can either contain an actual tensor or be
/// empty.
///
/// # Example
///
/// ```rust
/// use candle_core::streaming::StreamTensor;
/// use candle_core::{Tensor, Device, DType};
/// let t = Tensor::zeros((1, 4), DType::F32, &Device::Cpu)?;
/// let st = StreamTensor::from_tensor(t);
/// assert!(st.as_option().is_some());
/// let empty = StreamTensor::empty();
/// assert!(empty.as_option().is_none());
/// # Ok::<(), candle_core::Error>(())
/// ```
#[derive(Clone)]
pub struct StreamTensor(Option<Tensor>);

impl std::fmt::Debug for StreamTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(t) => write!(f, "{:?}", t.shape()),
            None => write!(f, "Empty"),
        }
    }
}

impl std::convert::From<Option<Tensor>> for StreamTensor {
    fn from(value: Option<Tensor>) -> Self {
        Self(value)
    }
}

impl std::convert::From<Tensor> for StreamTensor {
    fn from(value: Tensor) -> Self {
        Self(Some(value))
    }
}

impl std::convert::From<()> for StreamTensor {
    fn from(_value: ()) -> Self {
        Self(None)
    }
}

impl StreamTensor {
    /// Create an empty stream tensor (no data).
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::streaming::StreamTensor;
    /// let st = StreamTensor::empty();
    /// assert!(st.as_option().is_none());
    /// ```
    pub fn empty() -> Self {
        Self(None)
    }

    /// Wrap an existing tensor in a stream tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::streaming::StreamTensor;
    /// use candle_core::{Tensor, Device, DType};
    /// let t = Tensor::zeros((1, 4), DType::F32, &Device::Cpu)?;
    /// let st = StreamTensor::from_tensor(t);
    /// assert!(st.as_option().is_some());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn from_tensor(tensor: Tensor) -> Self {
        Self(Some(tensor))
    }

    /// Return the shape of the contained tensor, or `None` if empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::streaming::StreamTensor;
    /// use candle_core::{Tensor, Device, DType};
    /// let t = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let st = StreamTensor::from_tensor(t);
    /// assert_eq!(st.shape().unwrap().dims(), &[2, 3]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn shape(&self) -> Option<&Shape> {
        self.0.as_ref().map(|t| t.shape())
    }

    /// Concatenate two stream tensors along `dim`. If either side is empty, the other is
    /// returned unchanged. If both are empty, the result is empty.
    pub fn cat2<D: Dim>(&self, rhs: &Self, dim: D) -> Result<Self> {
        let xs = match (&self.0, &rhs.0) {
            (Some(lhs), Some(rhs)) => {
                let xs = Tensor::cat(&[lhs, rhs], dim)?;
                Some(xs)
            }
            (Some(xs), None) | (None, Some(xs)) => Some(xs.clone()),
            (None, None) => None,
        };
        Ok(Self(xs))
    }

    /// Return the size along `dim`, or 0 if the stream tensor is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::streaming::StreamTensor;
    /// use candle_core::{Tensor, Device, DType};
    /// let t = Tensor::zeros((2, 5), DType::F32, &Device::Cpu)?;
    /// let st = StreamTensor::from_tensor(t);
    /// assert_eq!(st.seq_len(1)?, 5);
    /// assert_eq!(StreamTensor::empty().seq_len(0)?, 0);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn seq_len<D: Dim>(&self, dim: D) -> Result<usize> {
        match &self.0 {
            None => Ok(0),
            Some(v) => v.dim(dim),
        }
    }

    /// Discard the contained tensor, making this stream tensor empty.
    pub fn reset(&mut self) {
        self.0 = None
    }

    /// Narrow (slice) along `dim` starting at `offset` for up to `len` elements.
    ///
    /// Returns empty if the stream tensor is empty or if `offset` is beyond the current
    /// size along `dim`.
    pub fn narrow<D: Dim>(&self, dim: D, offset: usize, len: usize) -> Result<StreamTensor> {
        let t = match &self.0 {
            None => None,
            Some(t) => {
                let seq_len = t.dim(dim)?;
                if seq_len <= offset {
                    None
                } else {
                    let t = t.narrow(dim, offset, usize::min(len, seq_len - offset))?;
                    Some(t)
                }
            }
        };
        Ok(Self(t))
    }

    /// Splits the Streaming Tensor on the time axis `dim` with the first `lhs_len` elements
    /// returned in the first output and the remaining in the second output.
    pub fn split<D: Dim>(&self, dim: D, lhs_len: usize) -> Result<(Self, Self)> {
        match &self.0 {
            None => Ok((Self::empty(), Self::empty())),
            Some(t) => {
                let seq_len = t.dim(dim)?;
                let lhs_len = usize::min(seq_len, lhs_len);
                if lhs_len == 0 {
                    Ok((Self::empty(), t.clone().into()))
                } else {
                    let lhs = Self::from_tensor(t.narrow(dim, 0, lhs_len)?);
                    let rhs_len = seq_len - lhs_len;
                    let rhs = if rhs_len == 0 {
                        Self::empty()
                    } else {
                        Self::from_tensor(t.narrow(dim, lhs_len, rhs_len)?)
                    };
                    Ok((lhs, rhs))
                }
            }
        }
    }

    /// Borrow the inner tensor, if present.
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::streaming::StreamTensor;
    /// use candle_core::{Tensor, Device, DType};
    /// let t = Tensor::zeros((1,), DType::F32, &Device::Cpu)?;
    /// let st = StreamTensor::from_tensor(t);
    /// assert!(st.as_option().is_some());
    /// assert!(StreamTensor::empty().as_option().is_none());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn as_option(&self) -> Option<&Tensor> {
        self.0.as_ref()
    }

    /// Apply a [`crate::Module`] to the inner tensor if present, returning empty
    /// if this stream tensor is empty.
    pub fn apply<M: crate::Module>(&self, m: &M) -> Result<Self> {
        match &self.0 {
            None => Ok(Self::empty()),
            Some(t) => Ok(Self::from_tensor(t.apply(m)?)),
        }
    }
}

/// A module that processes data incrementally one chunk at a time.
///
/// Implementations may maintain internal buffers so that enough data accumulates before
/// producing output. Call [`StreamingModule::step`] repeatedly with incoming chunks and
/// [`StreamingModule::reset_state`] to clear any buffered data between sequences.
///
/// # Example
///
/// ```no_run
/// use candle_core::streaming::{StreamTensor, StreamingModule};
/// use candle_core::Result;
/// struct Buffer { stored: StreamTensor }
/// impl StreamingModule for Buffer {
///     fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
///         self.stored = xs.clone();
///         Ok(xs.clone())
///     }
///     fn reset_state(&mut self) { self.stored = StreamTensor::empty(); }
/// }
/// ```
pub trait StreamingModule {
    /// Process the next chunk of input and return any output that is ready.
    // TODO: Should we also have a flush method?
    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor>;

    /// Clear all internal buffers and state, preparing the module for a new sequence.
    fn reset_state(&mut self);
}

/// Element-wise binary operations supported by [`StreamingBinOp`].
///
/// # Example
///
/// ```rust
/// use candle_core::streaming::BinOp;
/// let op = BinOp::Add;
/// assert_eq!(op, BinOp::Add);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Mul,
    Sub,
    Div,
}

/// A streaming binary operator that buffers left and right operands until they have
/// matching lengths along the streaming dimension, then applies the operation.
///
/// This is useful when two branches of a streaming pipeline produce chunks at different
/// rates. The operator internally accumulates whichever side is ahead and only emits
/// output for the portion where both sides overlap.
///
/// # Example
///
/// ```rust
/// use candle_core::streaming::{BinOp, StreamingBinOp, StreamTensor};
/// use candle_core::{Tensor, Device, DType, D};
/// let op = StreamingBinOp::new(BinOp::Add, D::Minus1);
/// assert_eq!(op.op, BinOp::Add);
/// ```
#[derive(Debug, Clone)]
pub struct StreamingBinOp {
    prev_lhs: StreamTensor,
    prev_rhs: StreamTensor,
    pub op: BinOp,
    pub dim: crate::D,
}

impl StreamingBinOp {
    /// Create a new streaming binary operator for the given operation and streaming dimension.
    pub fn new(op: BinOp, dim: crate::D) -> Self {
        Self {
            prev_lhs: StreamTensor::empty(),
            prev_rhs: StreamTensor::empty(),
            op,
            dim,
        }
    }

    /// Clear the internal left/right buffers.
    pub fn reset_state(&mut self) {
        self.prev_lhs.reset();
        self.prev_rhs.reset();
    }

    /// Apply the binary operation to two fully-aligned tensors (non-streaming path).
    pub fn forward(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        match self.op {
            BinOp::Add => Tensor::add(lhs, rhs),
            BinOp::Mul => Tensor::mul(lhs, rhs),
            BinOp::Sub => Tensor::sub(lhs, rhs),
            BinOp::Div => Tensor::div(lhs, rhs),
        }
    }

    /// Feed the next chunks of left and right operands. Returns the result for the portion
    /// where both sides overlap; any excess is buffered internally for the next call.
    pub fn step(&mut self, lhs: &StreamTensor, rhs: &StreamTensor) -> Result<StreamTensor> {
        let lhs = StreamTensor::cat2(&self.prev_lhs, lhs, self.dim)?;
        let rhs = StreamTensor::cat2(&self.prev_rhs, rhs, self.dim)?;
        let lhs_len = lhs.seq_len(self.dim)?;
        let rhs_len = rhs.seq_len(self.dim)?;
        let common_len = usize::min(lhs_len, rhs_len);
        let (lhs, prev_lhs) = lhs.split(self.dim, common_len)?;
        let (rhs, prev_rhs) = rhs.split(self.dim, common_len)?;
        let ys = match (lhs.0, rhs.0) {
            (Some(lhs), Some(rhs)) => {
                let ys = self.forward(&lhs, &rhs)?;
                StreamTensor::from_tensor(ys)
            }
            (None, None) => StreamTensor::empty(),
            (lhs, rhs) => crate::bail!("INTERNAL ERROR inconsistent lhs and rhs {lhs:?} {rhs:?}"),
        };
        self.prev_lhs = prev_lhs;
        self.prev_rhs = prev_rhs;
        Ok(ys)
    }
}

/// A [`StreamingModule`] adapter that wraps any [`crate::Module`] without buffering.
///
/// Each call to [`StreamingModule::step`] simply applies the inner module to the stream
/// tensor. This is useful for point-wise or stateless operations (e.g. activations,
/// layer norms) that do not need to accumulate data across steps.
pub struct Map<T: crate::Module>(T);

impl<T: crate::Module> StreamingModule for Map<T> {
    fn reset_state(&mut self) {}

    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
        xs.apply(&self.0)
    }
}

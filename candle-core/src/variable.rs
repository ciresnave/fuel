// Variables are wrappers around tensors that can be modified, they are typically used for holding
// weights and being modified by gradient descent.
// We do not expose a public way to create variables as this would break the invariant that the
// tensor within a variable is actually with `is_variable` set to `true`.
use crate::{DType, Device, Error, Result, Shape, Tensor};

/// A variable is a wrapper around a tensor, however variables can have their content modified
/// whereas tensors are immutable. Variables are the primary mechanism for holding model weights
/// during training: they track gradients through backpropagation and allow in-place updates
/// via optimizers.
///
/// `Var` dereferences to [`Tensor`], so all read-only tensor operations (e.g., `dims()`,
/// `matmul()`, `to_vec2()`) are available directly on a `Var`.
///
/// # Examples
///
/// ```rust
/// use candle_core::{Var, DType, Device};
///
/// // Create a variable, use it in a computation, and compute gradients.
/// let x = Var::new(&[3.0f32, 1.0, 4.0], &Device::Cpu)?;
/// let y = x.as_tensor().sqr()?;
/// let grads = y.sum_all()?.backward()?;
/// let grad_x = grads.get(x.as_tensor()).unwrap();
/// // dy/dx = 2*x, so gradients should be [6.0, 2.0, 8.0]
/// assert_eq!(grad_x.to_vec1::<f32>()?, vec![6.0, 2.0, 8.0]);
/// # Ok::<(), candle_core::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Var(Tensor);

/// Displays the variable using the same format as [`Tensor`].
impl std::fmt::Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::ops::Deref for Var {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl Var {
    /// Creates a variable filled with zeros.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, DType, Device};
    ///
    /// let v = Var::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// assert_eq!(v.dims(), &[2, 3]);
    /// assert_eq!(v.to_vec2::<f32>()?, vec![vec![0.0; 3]; 2]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let inner = Tensor::zeros_impl(shape, dtype, device, true)?;
        Ok(Self(inner))
    }

    /// Creates a variable filled with ones.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, DType, Device};
    ///
    /// let v = Var::ones((1, 4), DType::F32, &Device::Cpu)?;
    /// assert_eq!(v.to_vec2::<f32>()?, vec![vec![1.0; 4]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let inner = Tensor::ones_impl(shape, dtype, device, true)?;
        Ok(Self(inner))
    }

    /// Converts a tensor to a variable. If the tensor is already a variable it is returned
    /// as-is; otherwise a new variable-flagged copy is created. This is useful when loading
    /// pretrained weights that need to become trainable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, Tensor, Device};
    ///
    /// let t = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
    /// let v = Var::from_tensor(&t)?;
    /// assert_eq!(v.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn from_tensor(t: &Tensor) -> Result<Self> {
        if t.is_variable() {
            Ok(Self(t.clone()))
        } else {
            let inner = t.make_var()?;
            Ok(Self(inner))
        }
    }

    /// Creates a variable with random values sampled uniformly in `[lo, up)` using `f64`
    /// bounds. For a version that infers the dtype from the bound type, see [`Var::rand`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, DType, Device};
    ///
    /// let v = Var::rand_f64(0.0, 1.0, (2, 3), DType::F32, &Device::Cpu)?;
    /// assert_eq!(v.dims(), &[2, 3]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn rand_f64<S: Into<Shape>>(
        lo: f64,
        up: f64,
        s: S,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let inner = Tensor::rand_f64_impl(lo, up, s, dtype, device, true)?;
        Ok(Self(inner))
    }

    /// Creates a variable with random values sampled from a normal distribution with the
    /// given `mean` and `std` (standard deviation) using `f64` bounds. For a version that
    /// infers the dtype from the bound type, see [`Var::randn`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, DType, Device};
    ///
    /// let v = Var::randn_f64(0.0, 1.0, (3, 3), DType::F32, &Device::Cpu)?;
    /// assert_eq!(v.dims(), &[3, 3]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn randn_f64<S: Into<Shape>>(
        mean: f64,
        std: f64,
        s: S,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let inner = Tensor::randn_f64_impl(mean, std, s, dtype, device, true)?;
        Ok(Self(inner))
    }

    /// Creates a variable with random values sampled uniformly in `[lo, up)`. The dtype is
    /// inferred from the bound type `T` (e.g., `f32` yields `DType::F32`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, Device};
    ///
    /// let v = Var::rand(-1.0f32, 1.0f32, (4,), &Device::Cpu)?;
    /// assert_eq!(v.dims(), &[4]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn rand<S: Into<Shape>, T: crate::FloatDType>(
        lo: T,
        up: T,
        s: S,
        device: &Device,
    ) -> Result<Self> {
        let inner = Tensor::rand_impl(lo, up, s, device, true)?;
        Ok(Self(inner))
    }

    /// Creates a variable with random values sampled from a normal distribution with the
    /// given `mean` and `std`. The dtype is inferred from the bound type `T`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, Device};
    ///
    /// let v = Var::randn(0.0f32, 1.0f32, (2, 2), &Device::Cpu)?;
    /// assert_eq!(v.dims(), &[2, 2]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn randn<S: Into<Shape>, T: crate::FloatDType>(
        mean: T,
        std: T,
        s: S,
        device: &Device,
    ) -> Result<Self> {
        let inner = Tensor::randn_impl(mean, std, s, device, true)?;
        Ok(Self(inner))
    }

    /// Creates a new variable on the specified device from an array-like value. Accepts
    /// scalars, slices, `Vec`, and nested arrays (up to 6D). This is the most common way
    /// to create a variable with known initial data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, Device};
    ///
    /// // From a 1-D slice
    /// let v = Var::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
    /// assert_eq!(v.dims(), &[3]);
    ///
    /// // From a 2-D array
    /// let v = Var::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu)?;
    /// assert_eq!(v.dims(), &[2, 2]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn new<A: crate::device::NdArray>(array: A, device: &Device) -> Result<Self> {
        let shape = array.shape()?;
        let inner = Tensor::new_impl(array, shape, device, true)?;
        Ok(Self(inner))
    }

    /// Creates a variable from a `Vec` of data and a shape. The length of `data` must
    /// match the number of elements implied by `shape`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, Device};
    ///
    /// let v = Var::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)?;
    /// assert_eq!(v.to_vec2::<f32>()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn from_vec<S: Into<Shape>, D: crate::WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let inner = Tensor::from_vec_impl(data, shape, device, true)?;
        Ok(Self(inner))
    }

    /// Creates a variable from a slice of data and a shape. Similar to [`Var::from_vec`]
    /// but borrows the data instead of taking ownership.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, Device};
    ///
    /// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let v = Var::from_slice(&data, (2, 3), &Device::Cpu)?;
    /// assert_eq!(v.dims(), &[2, 3]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn from_slice<S: Into<Shape>, D: crate::WithDType>(
        array: &[D],
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let inner = Tensor::new_impl(array, shape.into(), device, true)?;
        Ok(Self(inner))
    }

    /// Returns a detached copy of the underlying tensor. The returned tensor is **not** a
    /// variable and will not track gradients. This is useful when you need to use the
    /// current value of a variable without affecting the computation graph.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, Device};
    ///
    /// let v = Var::new(&[1.0f32, 2.0], &Device::Cpu)?;
    /// let t = v.as_detached_tensor();
    /// assert!(!t.is_variable());
    /// assert_eq!(t.to_vec1::<f32>()?, vec![1.0, 2.0]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn as_detached_tensor(&self) -> Tensor {
        self.0.detach()
    }

    /// Returns a reference to the underlying tensor. Unlike [`Var::as_detached_tensor`],
    /// this keeps the variable's gradient tracking intact.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, DType, Device};
    ///
    /// let v = Var::zeros(3, DType::F32, &Device::Cpu)?;
    /// let t = v.as_tensor();
    /// assert!(t.is_variable());
    /// assert_eq!(t.dims(), &[3]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn as_tensor(&self) -> &Tensor {
        &self.0
    }

    /// Consumes this `Var` and returns the underlying tensor (still marked as a variable).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, DType, Device};
    ///
    /// let v = Var::ones((2,), DType::F32, &Device::Cpu)?;
    /// let t = v.into_inner();
    /// assert!(t.is_variable());
    /// assert_eq!(t.to_vec1::<f32>()?, vec![1.0, 1.0]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn into_inner(self) -> Tensor {
        self.0
    }

    /// Sets the content of the variable to match `src`. This uses interior mutability so no
    /// `&mut self` is needed. The source tensor must have the same shape as the variable and
    /// must **not** be derived from this variable's storage (to avoid aliasing issues).
    ///
    /// This is the mechanism optimizers use to update weights in-place after computing
    /// gradient steps.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Var, Tensor, DType, Device};
    ///
    /// let v = Var::zeros(3, DType::F32, &Device::Cpu)?;
    /// let new_vals = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
    /// v.set(&new_vals)?;
    /// assert_eq!(v.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn set(&self, src: &Tensor) -> Result<()> {
        if self.same_storage(src) {
            let msg = "cannot set a variable to a tensor that is derived from its value";
            Err(Error::CannotSetVar { msg }.bt())?
        }
        let (mut dst, layout) = self.storage_mut_and_layout();
        if !layout.is_contiguous() {
            let msg = "cannot set a non-contiguous variable";
            Err(Error::CannotSetVar { msg }.bt())?
        }
        let (src, src_l) = src.storage_and_layout();
        if layout.shape() != src_l.shape() {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: layout.shape().clone(),
                rhs: src_l.shape().clone(),
                op: "set",
            }
            .bt())?
        }
        src.copy_strided_src(&mut dst, layout.start_offset(), src_l)?;
        Ok(())
    }
}

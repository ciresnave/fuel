//! Core backend traits for device-agnostic tensor operations.
//!
//! This module defines the two traits that every compute backend must implement:
//!
//! - [`BackendStorage`] — a typed buffer of tensor data that lives on a
//!   specific device.
//!
//! - [`BackendDevice`] — a handle to a physical or virtual compute device.
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};

/// A typed, device-resident buffer that implements all tensor operations for
/// one backend.
///
/// Each method receives one or more [`Layout`]s that describe how the logical
/// tensor is mapped onto the flat backing buffer (offsets, strides, shape).
pub trait BackendStorage: Sized {
    /// The device type that owns storage of this kind.
    type Device: BackendDevice;

    /// Creates a new storage containing a copy of the elements described by
    /// `layout`.
    fn try_clone(&self, _: &Layout) -> Result<Self>;

    /// Returns the element [`DType`] stored in this buffer.
    fn dtype(&self) -> DType;

    /// Returns a reference to the device that owns this storage.
    fn device(&self) -> &Self::Device;

    /// Copies the data to a [`CpuStorage`].
    fn to_cpu_storage(&self) -> Result<CpuStorage>;

    /// Applies the elementwise affine map `x ↦ x * mul + add`.
    fn affine(&self, _: &Layout, _mul: f64, _add: f64) -> Result<Self>;

    /// Raises every element to the power `exp`.
    fn powf(&self, _: &Layout, _exp: f64) -> Result<Self>;

    /// Applies the ELU activation.
    fn elu(&self, _: &Layout, _alpha: f64) -> Result<Self>;

    /// Reduces elements over `axes` using `op`.
    fn reduce_op(&self, _: ReduceOp, _: &Layout, _axes: &[usize]) -> Result<Self>;

    /// Applies a pointwise comparison `op`.
    fn cmp(&self, _: CmpOp, _rhs: &Self, _lhs_layout: &Layout, _rhs_layout: &Layout)
        -> Result<Self>;

    /// Converts every element to `dtype`.
    fn to_dtype(&self, _: &Layout, _dtype: DType) -> Result<Self>;

    /// Applies a unary operation.
    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self>;

    /// Applies a binary operation.
    fn binary_impl<B: BinaryOpT>(
        &self,
        _rhs: &Self,
        _lhs_layout: &Layout,
        _rhs_layout: &Layout,
    ) -> Result<Self>;

    /// Returns on_true where condition is non-zero, on_false elsewhere.
    fn where_cond(
        &self,
        _cond_layout: &Layout,
        _on_true: &Self,
        _on_true_layout: &Layout,
        _on_false: &Self,
        _on_false_layout: &Layout,
    ) -> Result<Self>;

    /// Performs a 1-D convolution.
    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self>;

    /// Performs a 1-D transposed convolution.
    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self>;

    /// Performs a 2-D convolution.
    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self>;

    /// Performs a 2-D transposed convolution.
    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self>;

    /// Applies 2-D average pooling.
    fn avg_pool2d(
        &self,
        _layout: &Layout,
        _kernel: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self>;

    /// Applies 2-D max pooling.
    fn max_pool2d(
        &self,
        _layout: &Layout,
        _kernel: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self>;

    /// Upsamples the 1-D spatial dimension using nearest-neighbor interpolation.
    fn upsample_nearest1d(&self, _: &Layout, _target_size: usize) -> Result<Self>;

    /// Upsamples the 2-D spatial dimensions using nearest-neighbor interpolation.
    fn upsample_nearest2d(&self, _: &Layout, _target_h: usize, _target_w: usize) -> Result<Self>;

    /// Upsamples the 2-D spatial dimensions using bilinear interpolation.
    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _target_h: usize,
        _target_w: usize,
        _align_corners: bool,
        _scale_h: Option<f64>,
        _scale_w: Option<f64>,
    ) -> Result<Self>;

    /// Gathers elements from `self` along `dim` using integer indices.
    fn gather(
        &self,
        _src_layout: &Layout,
        _ids: &Self,
        _ids_layout: &Layout,
        _dim: usize,
    ) -> Result<Self>;

    /// Scatters `src` into `self` along `dim` at positions given by `ids`.
    fn scatter_set(
        &mut self,
        _self_layout: &Layout,
        _src: &Self,
        _src_layout: &Layout,
        _ids: &Self,
        _ids_layout: &Layout,
        _dim: usize,
    ) -> Result<()>;

    /// Atomically adds `src` into `self` along `dim` at positions given by `ids`.
    fn scatter_add_set(
        &mut self,
        _self_layout: &Layout,
        _src: &Self,
        _src_layout: &Layout,
        _ids: &Self,
        _ids_layout: &Layout,
        _dim: usize,
    ) -> Result<()>;

    /// Selects slices from `self` along `dim` using integer indices.
    fn index_select(
        &self,
        _ids: &Self,
        _src_layout: &Layout,
        _ids_layout: &Layout,
        _dim: usize,
    ) -> Result<Self>;

    /// Accumulates `src` slices into `self` along `dim` at the positions
    /// specified by `ids`.
    fn index_add(
        &self,
        _self_layout: &Layout,
        _ids: &Self,
        _ids_layout: &Layout,
        _src: &Self,
        _src_layout: &Layout,
        _dim: usize,
    ) -> Result<Self>;

    /// Computes a (batched) matrix product `lhs @ rhs`.
    fn matmul(
        &self,
        _rhs: &Self,
        _bmnk: (usize, usize, usize, usize),
        _lhs_layout: &Layout,
        _rhs_layout: &Layout,
    ) -> Result<Self>;

    /// Copies elements from `self` into `dst` starting at `dst_offset`.
    fn copy_strided_src(
        &self,
        _dst: &mut Self,
        _dst_offset: usize,
        _src_layout: &Layout,
    ) -> Result<()>;

    /// Copies a 2-D tile of elements from `self` into `dst`.
    #[allow(clippy::too_many_arguments)]
    fn copy2d(
        &self,
        _dst: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_stride1: usize,
        _dst_stride1: usize,
        _src_offset: usize,
        _dst_offset: usize,
    ) -> Result<()>;

    /// Sets every element addressed by `layout` to `value`.
    fn const_set(&mut self, _value: crate::scalar::Scalar, _: &Layout) -> Result<()>;
}

/// A handle to a physical or virtual compute device.
///
/// `BackendDevice` manufactures [`BackendStorage`] instances and manages
/// device-level resources.
pub trait BackendDevice: Sized + std::fmt::Debug + Clone {
    /// The storage type associated with this device.
    type Storage: BackendStorage;

    /// Creates a handle for the n-th device of this backend type.
    fn new(_ordinal: usize) -> Result<Self>;

    /// Returns the canonical [`DeviceLocation`](crate::DeviceLocation) that
    /// identifies this device.
    fn location(&self) -> crate::DeviceLocation;

    /// Returns `true` if `self` and `other` refer to the same physical device.
    fn same_device(&self, _other: &Self) -> bool;

    /// Allocates a zero-initialized storage.
    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    /// Allocates a storage with undefined contents.
    ///
    /// # Safety
    ///
    /// The caller must initialize every element before any read occurs.
    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    /// Creates a storage by copying elements from a host slice.
    fn storage_from_slice<T: crate::WithDType>(&self, _data: &[T]) -> Result<Self::Storage>;

    /// Creates a storage by copying data from a [`CpuStorage`].
    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage>;

    /// Creates a storage from a [`CpuStorage`], taking ownership.
    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage>;

    /// Generates a storage filled with uniform random values in `[lo, hi)`.
    fn rand_uniform(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _lo: f64,
        _hi: f64,
    ) -> Result<Self::Storage>;

    /// Generates a storage filled with normally distributed random values.
    fn rand_normal(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _mean: f64,
        _std: f64,
    ) -> Result<Self::Storage>;

    /// Sets the random-number generator seed.
    fn set_seed(&self, _seed: u64) -> Result<()>;

    /// Returns the current RNG seed.
    fn get_current_seed(&self) -> Result<u64>;

    /// Blocks until all pending operations are complete.
    fn synchronize(&self) -> Result<()>;
}

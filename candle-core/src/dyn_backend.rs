//! Object-safe backend traits for dynamic dispatch.
//!
//! [`DynBackendStorage`] and [`DynBackendDevice`] are the object-safe
//! counterparts of [`BackendStorage`](crate::backend::BackendStorage) and
//! [`BackendDevice`](crate::backend::BackendDevice). They replace `Self`
//! return types with `Box<dyn DynBackendStorage>`, eliminate generic type
//! parameters, and drop the `Sized` bound so that `dyn DynBackendStorage`
//! and `dyn DynBackendDevice` are legal trait objects.
//!
//! These traits power the `Device::Custom` and `Storage::Custom` variants,
//! enabling third-party backends without modifying `candle-core`.
//!
//! # Implementing a custom backend
//!
//! ```rust,ignore
//! use candle_core::dyn_backend::{DynBackendDevice, DynBackendStorage};
//! use candle_core::Device;
//! use std::sync::Arc;
//!
//! struct MyDevice { /* ... */ }
//! impl DynBackendDevice for MyDevice { /* ... */ }
//!
//! struct MyStorage { /* ... */ }
//! impl DynBackendStorage for MyStorage { /* ... */ }
//!
//! let device = Device::custom(Arc::new(MyDevice::new()));
//! ```
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use crate::{CpuStorage, DType, DeviceLocation, Layout, Result, Shape};
use std::any::Any;
use std::sync::Arc;

/// Object-safe counterpart of [`BackendStorage`](crate::backend::BackendStorage).
///
/// Every method mirrors the corresponding `BackendStorage` method but returns
/// `Box<dyn DynBackendStorage>` instead of `Self` and accepts `&dyn
/// DynBackendStorage` instead of `&Self` for multi-operand operations.
///
/// For binary and multi-operand methods (`cmp`, `binary_op`, `where_cond`,
/// etc.), the implementation must downcast the `&dyn DynBackendStorage`
/// argument to its concrete type. Operations between different custom
/// backends (or between a custom backend and a built-in one) are treated as
/// a device mismatch.
pub trait DynBackendStorage: Send + Sync + std::fmt::Debug {
    /// Clone the elements described by `layout` into a new storage.
    fn try_clone_dyn(&self, layout: &Layout) -> Result<Box<dyn DynBackendStorage>>;

    /// The element dtype of this storage.
    fn dtype_dyn(&self) -> DType;

    /// The device that owns this storage.
    fn device_dyn(&self) -> &dyn DynBackendDevice;

    /// Return a cloned `Arc` handle to the owning device.
    ///
    /// This is used by `Storage::device()` to reconstruct
    /// `Device::Custom(arc)` without requiring the `Storage` enum to carry
    /// a redundant copy of the device reference.
    fn device_arc_dyn(&self) -> Arc<dyn DynBackendDevice>;

    /// Copy the entire storage to CPU.
    fn to_cpu_storage_dyn(&self) -> Result<CpuStorage>;

    /// Elementwise affine: `x * mul + add`.
    fn affine_dyn(&self, layout: &Layout, mul: f64, add: f64)
        -> Result<Box<dyn DynBackendStorage>>;

    /// Raise every element to the power `e`.
    fn powf_dyn(&self, layout: &Layout, e: f64) -> Result<Box<dyn DynBackendStorage>>;

    /// ELU activation.
    fn elu_dyn(&self, layout: &Layout, alpha: f64) -> Result<Box<dyn DynBackendStorage>>;

    /// Reduction (sum, min, max, …) over `axes`.
    fn reduce_op_dyn(
        &self,
        op: ReduceOp,
        layout: &Layout,
        axes: &[usize],
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Elementwise comparison, returning a `U8` buffer.
    fn cmp_dyn(
        &self,
        op: CmpOp,
        rhs: &dyn DynBackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Cast to a different dtype.
    fn to_dtype_dyn(&self, layout: &Layout, dtype: DType) -> Result<Box<dyn DynBackendStorage>>;

    /// Elementwise unary operation.
    fn unary_op_dyn(
        &self,
        layout: &Layout,
        op: UnaryOp,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Elementwise binary operation.
    fn binary_op_dyn(
        &self,
        rhs: &dyn DynBackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: BinaryOp,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Select `on_true` where condition is nonzero, `on_false` elsewhere.
    fn where_cond_dyn(
        &self,
        cond_layout: &Layout,
        on_true: &dyn DynBackendStorage,
        on_true_layout: &Layout,
        on_false: &dyn DynBackendStorage,
        on_false_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 1-D convolution.
    fn conv1d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 1-D transposed convolution.
    fn conv_transpose1d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 2-D convolution.
    fn conv2d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 2-D transposed convolution.
    fn conv_transpose2d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 2-D average pooling.
    fn avg_pool2d_dyn(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 2-D max pooling.
    fn max_pool2d_dyn(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 1-D nearest-neighbor upsampling.
    fn upsample_nearest1d_dyn(
        &self,
        layout: &Layout,
        target_size: usize,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 2-D nearest-neighbor upsampling.
    fn upsample_nearest2d_dyn(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 2-D bilinear upsampling.
    #[allow(clippy::too_many_arguments)]
    fn upsample_bilinear2d_dyn(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Gather elements along `dim` using `ids`.
    fn gather_dyn(
        &self,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Scatter-set `src` into `self` along `dim` at positions in `ids`.
    fn scatter_set_dyn(
        &mut self,
        self_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<()>;

    /// Scatter-add `src` into `self` along `dim` at positions in `ids`.
    fn scatter_add_set_dyn(
        &mut self,
        self_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<()>;

    /// Select slices along `dim` at positions in `ids`.
    fn index_select_dyn(
        &self,
        ids: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Index-add: accumulate `src` slices into `self` along `dim` at `ids`.
    fn index_add_dyn(
        &self,
        self_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Matrix multiply.
    fn matmul_dyn(
        &self,
        rhs: &dyn DynBackendStorage,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Copy strided source into `dst` starting at `dst_offset`.
    fn copy_strided_src_dyn(
        &self,
        dst: &mut dyn DynBackendStorage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> Result<()>;

    /// Copy a 2-D tile from `self` into `dst`.
    #[allow(clippy::too_many_arguments)]
    fn copy2d_dyn(
        &self,
        dst: &mut dyn DynBackendStorage,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()>;

    /// Set every element described by `layout` to `value`.
    fn const_set_dyn(&mut self, value: crate::scalar::Scalar, layout: &Layout) -> Result<()>;

    /// Downcast to the concrete storage type.
    fn as_any(&self) -> &dyn Any;

    /// Downcast to the concrete storage type (mutable).
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Object-safe counterpart of [`BackendDevice`](crate::backend::BackendDevice).
///
/// All factory methods return `Box<dyn DynBackendStorage>` instead of
/// `Self::Storage`, and there is no `storage_from_slice` (use
/// [`storage_from_cpu_storage`](DynBackendDevice::storage_from_cpu_storage_dyn)
/// after constructing a [`CpuStorage`] on the host side).
pub trait DynBackendDevice: Send + Sync + std::fmt::Debug {
    /// Canonical device location.
    fn location_dyn(&self) -> DeviceLocation;

    /// Whether `self` and `other` refer to the same physical device.
    fn same_device_dyn(&self, other: &dyn DynBackendDevice) -> bool;

    /// Allocate zero-initialized storage.
    fn zeros_impl_dyn(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn DynBackendStorage>>;

    /// Allocate uninitialized storage.
    ///
    /// # Safety
    ///
    /// Caller must initialize all elements before reading.
    unsafe fn alloc_uninit_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Create storage from a CPU buffer (borrowed).
    fn storage_from_cpu_storage_dyn(
        &self,
        cpu: &CpuStorage,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Create storage from a CPU buffer (owned).
    fn storage_from_cpu_storage_owned_dyn(
        &self,
        cpu: CpuStorage,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Fill storage with uniform random values in `[lo, hi)`.
    fn rand_uniform_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        hi: f64,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Fill storage with normally distributed random values.
    fn rand_normal_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Set the RNG seed.
    fn set_seed_dyn(&self, seed: u64) -> Result<()>;

    /// Get the current RNG seed.
    fn get_current_seed_dyn(&self) -> Result<u64>;

    /// Block until all pending operations complete.
    fn synchronize_dyn(&self) -> Result<()>;

    /// Downcast to the concrete device type.
    fn as_any(&self) -> &dyn Any;
}

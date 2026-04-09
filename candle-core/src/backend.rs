//! Core backend traits for device-agnostic tensor operations.
//!
//! This module defines the two traits that every compute backend must implement:
//!
//! - [`BackendStorage`] — a typed buffer of tensor data that lives on a
//!   specific device. It knows its own [`DType`] and its parent
//!   [`BackendDevice`], and it carries out all tensor operations (arithmetic,
//!   reductions, contractions, memory copies, …) in terms of the backend's
//!   own primitive types and kernels.
//!
//! - [`BackendDevice`] — a handle to a physical or virtual compute device
//!   (CPU core, GPU ordinal, Metal command queue, …). It manufactures blank
//!   storage tensors (zeros, uninitialized, from host data) and manages
//!   device-level resources such as the random-number-generator seed.
//!
//! # Extension point
//!
//! These traits are the stable seam for adding new backends without modifying
//! `candle-core`. A third-party backend crate implements both traits, wraps
//! the storage type in `Storage::Custom` / `Device::Custom`, and passes a
//! constructed `Device` to `Tensor::new` — all without forking this crate.
//! (The `Custom` variant is planned for a future release; for now the first
//! step is stabilizing the documented contracts below.)
//!
//! [`BackendStorage`]: BackendStorage
//! [`BackendDevice`]: BackendDevice
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};

/// A typed, device-resident buffer that implements all tensor operations for
/// one backend.
///
/// Each method receives one or more [`Layout`]s that describe how the logical
/// tensor is mapped onto the flat backing buffer (offsets, strides, shape).
/// Implementations must read and write only the elements that the layout
/// addresses; they must not assume contiguity unless they explicitly check for
/// it.
///
/// Every method returns a **new** `Self` (except the in-place mutating methods
/// `scatter_set`, `scatter_add_set`, `copy_strided_src`, `copy2d`, and
/// `const_set`).  The caller is responsible for ensuring that layouts are
/// valid for the storage dimensions; passing an out-of-bounds layout is
/// undefined behaviour.
pub trait BackendStorage: Sized {
    /// The device type that owns storage of this kind.
    type Device: BackendDevice;

    /// Creates a new storage containing a copy of the elements described by
    /// `layout`.
    ///
    /// The result is always contiguous, regardless of whether the source
    /// layout is contiguous.  Returns an error if the target device cannot
    /// allocate the required memory.
    fn try_clone(&self, _: &Layout) -> Result<Self>;

    /// Returns the element [`DType`] stored in this buffer.
    fn dtype(&self) -> DType;

    /// Returns a reference to the device that owns this storage.
    fn device(&self) -> &Self::Device;

    /// Copies the data described by the entire storage to a [`CpuStorage`].
    ///
    /// For CPU backends this may be a shallow reference-counted copy (no
    /// allocation).  For GPU/Metal backends this triggers device
    /// synchronization and a device-to-host transfer; the call blocks until
    /// the transfer is complete.
    ///
    /// The result is always a contiguous, row-major buffer.
    fn to_cpu_storage(&self) -> Result<CpuStorage>;

    /// Applies the elementwise affine map `x ↦ x * mul + add` to every
    /// element addressed by `layout`, returning a new contiguous storage.
    fn affine(&self, _: &Layout, _mul: f64, _add: f64) -> Result<Self>;

    /// Raises every element addressed by `layout` to the power `exp`,
    /// returning a new contiguous storage.
    fn powf(&self, _: &Layout, _exp: f64) -> Result<Self>;

    /// Applies the ELU activation `x ↦ max(x, 0) + alpha * (exp(min(x, 0)) - 1)`
    /// to every element addressed by `layout`, returning a new contiguous
    /// storage.
    fn elu(&self, _: &Layout, _alpha: f64) -> Result<Self>;

    /// Reduces elements over `axes` using `op` (sum, max, min, …).
    ///
    /// The axes in `axes` are reduced away; the resulting shape has those
    /// dimensions removed.  The layout describes the input view; the axes
    /// indices are relative to the layout's logical shape.
    fn reduce_op(&self, _: ReduceOp, _: &Layout, _axes: &[usize]) -> Result<Self>;

    /// Applies a pointwise comparison `op` between `self` (described by
    /// `lhs_layout`) and `rhs` (described by `rhs_layout`).
    ///
    /// Returns a storage of the same logical shape with dtype `U8`; each
    /// element is `1u8` where the comparison holds and `0u8` otherwise.
    /// The two layouts must describe tensors with the same logical shape (or
    /// broadcastable shapes if the backend supports broadcasting here).
    fn cmp(&self, _: CmpOp, _rhs: &Self, _lhs_layout: &Layout, _rhs_layout: &Layout)
        -> Result<Self>;

    /// Converts every element addressed by `layout` to `dtype`, returning a
    /// new contiguous storage of the target dtype.
    ///
    /// Precision is preserved to the extent the target dtype allows.  Casting
    /// an out-of-range floating-point value to an integer is implementation-
    /// defined but must not panic.
    fn to_dtype(&self, _: &Layout, _dtype: DType) -> Result<Self>;

    /// Applies the stateless unary operation `B` (as defined by [`UnaryOpT`])
    /// to every element addressed by `layout`, returning a new contiguous
    /// storage.
    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self>;

    /// Applies the stateless binary operation `B` (as defined by
    /// [`BinaryOpT`]) elementwise between `self` (described by `lhs_layout`)
    /// and `rhs` (described by `rhs_layout`), returning a new contiguous
    /// storage.
    ///
    /// Both layouts must describe tensors with the same logical shape.
    fn binary_impl<B: BinaryOpT>(
        &self,
        _rhs: &Self,
        _lhs_layout: &Layout,
        _rhs_layout: &Layout,
    ) -> Result<Self>;

    /// Returns a new storage that contains `on_true` where `self` (the
    /// condition, described by `cond_layout`) is non-zero and `on_false`
    /// elsewhere.
    ///
    /// All three participants (`self`, `on_true`, `on_false`) must describe
    /// tensors with the same logical shape.  The condition storage must have
    /// dtype `U8`.
    fn where_cond(
        &self,
        _cond_layout: &Layout,
        _on_true: &Self,
        _on_true_layout: &Layout,
        _on_false: &Self,
        _on_false_layout: &Layout,
    ) -> Result<Self>;

    /// Performs a 1-D convolution of `self` (input, described by `l`) with
    /// `kernel` (described by `kernel_l`) using the parameters in `params`.
    ///
    /// Input shape: `(batch, in_channels, length)`.
    /// Kernel shape: `(out_channels, in_channels/groups, kernel_size)`.
    /// Output shape is determined by `params`.
    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self>;

    /// Performs a 1-D transposed convolution (also called *deconvolution*)
    /// of `self` with `kernel` using the parameters in `params`.
    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self>;

    /// Performs a 2-D convolution of `self` (input, described by `l`) with
    /// `kernel` (described by `kernel_l`) using the parameters in `params`.
    ///
    /// Input shape: `(batch, in_channels, height, width)`.
    /// Kernel shape: `(out_channels, in_channels/groups, kH, kW)`.
    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self>;

    /// Performs a 2-D transposed convolution of `self` with `kernel` using
    /// the parameters in `params`.
    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self>;

    /// Applies 2-D average pooling over `self` (described by `layout`).
    ///
    /// `kernel` is `(kH, kW)` and `stride` is `(sH, sW)`.
    fn avg_pool2d(
        &self,
        _layout: &Layout,
        _kernel: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self>;

    /// Applies 2-D max pooling over `self` (described by `layout`).
    ///
    /// `kernel` is `(kH, kW)` and `stride` is `(sH, sW)`.
    fn max_pool2d(
        &self,
        _layout: &Layout,
        _kernel: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self>;

    /// Upsamples the 1-D spatial dimension of `self` to `target_size`
    /// elements using nearest-neighbor interpolation.
    ///
    /// Input shape: `(batch, channels, length)`.
    fn upsample_nearest1d(&self, _: &Layout, _target_size: usize) -> Result<Self>;

    /// Upsamples the 2-D spatial dimensions of `self` to `(target_h,
    /// target_w)` using nearest-neighbor interpolation.
    ///
    /// Input shape: `(batch, channels, height, width)`.
    fn upsample_nearest2d(&self, _: &Layout, _target_h: usize, _target_w: usize) -> Result<Self>;

    /// Upsamples the 2-D spatial dimensions of `self` to `(target_h,
    /// target_w)` using bilinear interpolation.
    ///
    /// `align_corners` matches the PyTorch / ONNX flag of the same name.
    /// `scale_h` and `scale_w`, when `Some`, override the computed scale
    /// factor; pass `None` to derive the scale from the target and source
    /// sizes.
    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _target_h: usize,
        _target_w: usize,
        _align_corners: bool,
        _scale_h: Option<f64>,
        _scale_w: Option<f64>,
    ) -> Result<Self>;

    /// Gathers elements from `self` along `dim` using the integer indices in
    /// `ids`.
    ///
    /// Equivalent to NumPy `take` along an axis.  `src_layout` describes
    /// `self`; `ids_layout` describes `ids`.  The output shape matches the
    /// shape of `ids`.
    fn gather(
        &self,
        _src_layout: &Layout,
        _ids: &Self,
        _ids_layout: &Layout,
        _dim: usize,
    ) -> Result<Self>;

    /// Scatters `src` into `self` along `dim` at positions given by `ids`.
    ///
    /// For each index `i` in `ids`, sets
    /// `self[..., ids[i], ...] = src[..., i, ...]` (schematically).
    ///
    /// This is an in-place operation; `self` is modified directly.  The
    /// caller must ensure that `self` is not shared (i.e., has unique
    /// ownership of its buffer) before calling this method.
    fn scatter_set(
        &mut self,
        _self_layout: &Layout,
        _src: &Self,
        _src_layout: &Layout,
        _ids: &Self,
        _ids_layout: &Layout,
        _dim: usize,
    ) -> Result<()>;

    /// Atomically adds `src` into `self` along `dim` at positions given by
    /// `ids`.
    ///
    /// Like [`scatter_set`] but accumulates instead of overwriting, so
    /// repeated indices are summed.
    ///
    /// This is an in-place operation; `self` is modified directly.
    ///
    /// [`scatter_set`]: BackendStorage::scatter_set
    fn scatter_add_set(
        &mut self,
        _self_layout: &Layout,
        _src: &Self,
        _src_layout: &Layout,
        _ids: &Self,
        _ids_layout: &Layout,
        _dim: usize,
    ) -> Result<()>;

    /// Selects slices from `self` along `dim` using the integer indices in
    /// `ids`, returning a new storage whose size along `dim` equals
    /// `ids.len()`.
    ///
    /// Equivalent to `self.index_select(dim, ids)` in PyTorch.
    fn index_select(
        &self,
        _ids: &Self,
        _src_layout: &Layout,
        _ids_layout: &Layout,
        _dim: usize,
    ) -> Result<Self>;

    /// Accumulates `src` slices into `self` along `dim` at the positions
    /// specified by `ids`, returning a new storage of the same shape as
    /// `self`.
    ///
    /// If `k` appears multiple times in `ids`, the corresponding `src` slices
    /// are all added to position `k` in the output.
    fn index_add(
        &self,
        _self_layout: &Layout,
        _ids: &Self,
        _ids_layout: &Layout,
        _src: &Self,
        _src_layout: &Layout,
        _dim: usize,
    ) -> Result<Self>;

    /// Computes a (batched) matrix product `lhs @ rhs` and returns the result
    /// as a new contiguous storage.
    ///
    /// The four-tuple `bmnk` is `(batch_size, m, n, k)`:
    /// - `lhs` has logical shape `(batch_size, m, k)` (described by `lhs_layout`).
    /// - `rhs` has logical shape `(batch_size, k, n)` (described by `rhs_layout`).
    /// - Result has shape `(batch_size, m, n)`.
    ///
    /// Both layouts may be non-contiguous; the implementation is responsible
    /// for handling strides correctly.
    fn matmul(
        &self,
        _rhs: &Self,
        _bmnk: (usize, usize, usize, usize),
        _lhs_layout: &Layout,
        _rhs_layout: &Layout,
    ) -> Result<Self>;

    /// Copies elements from `self` (described by `src_layout`) into `dst`
    /// starting at element offset `dst_offset`.
    ///
    /// The destination is always written in logical (row-major) order
    /// regardless of the source layout.  `dst` must be large enough to hold
    /// `dst_offset + src_layout.num_elements()` elements.
    fn copy_strided_src(
        &self,
        _dst: &mut Self,
        _dst_offset: usize,
        _src_layout: &Layout,
    ) -> Result<()>;

    /// Copies a 2-D tile of elements from `self` into `dst`.
    ///
    /// All sizes and offsets are in **elements**, not bytes (unlike CUDA's
    /// `cudaMemcpy2D` which uses bytes).
    ///
    /// - `d1` × `d2`: logical extent of the tile.
    /// - `src_stride1`: stride in elements between consecutive rows in `self`.
    /// - `dst_stride1`: stride in elements between consecutive rows in `dst`.
    /// - `src_offset`: element offset of the top-left corner in `self`.
    /// - `dst_offset`: element offset of the top-left corner in `dst`.
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

    /// Sets every element addressed by `layout` to the scalar constant
    /// `value`.
    ///
    /// This is an in-place operation.  `value` is cast to the storage's dtype
    /// before writing; the caller is responsible for ensuring the value is
    /// representable in that dtype.
    fn const_set(&mut self, _value: crate::scalar::Scalar, _: &Layout) -> Result<()>;
}

/// A handle to a physical or virtual compute device.
///
/// `BackendDevice` manufactures [`BackendStorage`] instances and manages
/// device-level resources (memory pool, RNG state, command queue, …).  Every
/// backend provides exactly one concrete type that implements this trait, and
/// a paired storage type that implements [`BackendStorage`].
///
/// # Ordinals
///
/// Devices are addressed by a zero-based *ordinal*.  Ordinal `0` selects the
/// primary or sole device of that type (e.g. the first GPU).  Calling
/// [`new`](BackendDevice::new) with an ordinal that does not exist returns an
/// error; it never panics.
///
/// # Thread safety
///
/// Implementations should be `Send + Sync` so that tensors (which hold an
/// `Arc<dyn BackendDevice>` internally) can be shared across threads.  The
/// trait itself does not enforce this bound because some Metal/Objective-C
/// types are not `Send`, but new implementations should strive for it.
pub trait BackendDevice: Sized + std::fmt::Debug + Clone {
    /// The storage type associated with this device.
    type Storage: BackendStorage;

    /// Creates a handle for the n-th device of this backend type.
    ///
    /// `ordinal = 0` selects the first (or only) available device.  Returns
    /// an error if the ordinal is out of range or if the device cannot be
    /// initialized (e.g., driver not installed, GPU not present).
    // TODO: Make the ordinal generic and part of a generic DeviceLocation.
    fn new(_ordinal: usize) -> Result<Self>;

    /// Returns the canonical [`DeviceLocation`](crate::DeviceLocation) that
    /// identifies this device (e.g., `DeviceLocation::Cuda { gpu_id: 0 }`).
    fn location(&self) -> crate::DeviceLocation;

    /// Returns `true` if `self` and `other` refer to the same physical
    /// device.
    ///
    /// Two handles with equal ordinals must return `true`.  Two handles with
    /// different ordinals must return `false`.  Comparing handles of different
    /// backend types is only possible via the higher-level `Device::same`
    /// method on `candle-core`.
    fn same_device(&self, _other: &Self) -> bool;

    /// Allocates a zero-initialized storage of the given shape and dtype on
    /// this device.
    ///
    /// Returns an error if the allocation fails (out of memory, unsupported
    /// dtype, …).
    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    /// Allocates a storage of the given shape and dtype whose contents are
    /// **undefined**.
    ///
    /// # Safety
    ///
    /// The caller **must** initialize every element before any read occurs.
    /// Reading uninitialized data is undefined behaviour (on the CPU backend
    /// this means UB in the Rust or C sense; on GPU backends the values are
    /// simply unpredictable but not memory-unsafe).
    ///
    /// This method exists for performance: allocating without zeroing can save
    /// significant time when the caller will overwrite all elements
    /// immediately (e.g., via [`copy_strided_src`](BackendStorage::copy_strided_src)).
    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    /// Creates a storage by copying elements from the host slice `data`.
    ///
    /// `T` must implement [`WithDType`](crate::WithDType), which ties the
    /// Rust type to the corresponding [`DType`].  For GPU/Metal backends this
    /// triggers a host-to-device transfer; the call blocks until the transfer
    /// is complete.
    fn storage_from_slice<T: crate::WithDType>(&self, _data: &[T]) -> Result<Self::Storage>;

    /// Creates a storage by copying data from an existing [`CpuStorage`].
    ///
    /// For GPU/Metal backends this performs a host-to-device transfer.  For
    /// the CPU backend this copies the data; use
    /// [`storage_from_cpu_storage_owned`](BackendDevice::storage_from_cpu_storage_owned)
    /// when ownership can be transferred to avoid the copy.
    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage>;

    /// Creates a storage from a [`CpuStorage`], taking ownership.
    ///
    /// When `self` is a CPU device, implementations may reuse the buffer
    /// directly (zero-copy).  When `self` is a GPU/Metal device, the call
    /// behaves the same as [`storage_from_cpu_storage`](BackendDevice::storage_from_cpu_storage).
    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage>;

    /// Generates a storage filled with uniform random values in `[lo, hi)`.
    ///
    /// Uses the device's internal RNG state, which can be seeded with
    /// [`set_seed`](BackendDevice::set_seed).  Only `F32` and `F64` dtypes
    /// are required to be supported; other dtypes may return an error.
    fn rand_uniform(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _lo: f64,
        _hi: f64,
    ) -> Result<Self::Storage>;

    /// Generates a storage filled with normally distributed random values
    /// with the given `mean` and standard deviation `std`.
    ///
    /// Uses the device's internal RNG state.  Only `F32` and `F64` dtypes
    /// are required to be supported; other dtypes may return an error.
    fn rand_normal(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _mean: f64,
        _std: f64,
    ) -> Result<Self::Storage>;

    /// Sets the random-number generator seed for this device to `seed`.
    ///
    /// The same seed must produce the same sequence of random values on
    /// subsequent calls to [`rand_uniform`](BackendDevice::rand_uniform) and
    /// [`rand_normal`](BackendDevice::rand_normal), subject to the constraint
    /// that only a single logical RNG stream is required (i.e., parallel GPU
    /// kernels may produce values in an unspecified order within a single call,
    /// but the total multiset of values is deterministic given the seed).
    fn set_seed(&self, _seed: u64) -> Result<()>;

    /// Returns the current RNG seed for this device.
    ///
    /// The returned value reflects the most recent call to
    /// [`set_seed`](BackendDevice::set_seed), not any position within the
    /// generated sequence.
    fn get_current_seed(&self) -> Result<u64>;

    /// Blocks until all pending operations on this device are complete.
    ///
    /// On CPU backends this is a no-op (all operations are already
    /// synchronous).  On GPU/Metal backends this flushes the command queue
    /// and waits for the device to become idle — equivalent to
    /// `cudaDeviceSynchronize()` or `MTLCommandBuffer.waitUntilCompleted()`.
    ///
    /// Call this before reading device outputs from the host side when the
    /// ordering guarantee of `to_cpu_storage` is not sufficient.
    fn synchronize(&self) -> Result<()>;
}

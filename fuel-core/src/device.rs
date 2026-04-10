//! Device abstraction for CPU, CUDA, and Metal backends.
//!
//! ```rust
//! use fuel_core::Device;
//! let dev = Device::cpu();
//! assert!(dev.is_cpu());
//! assert_eq!(dev.location(), fuel_core::DeviceLocation::Cpu);
//! ```
use crate::dyn_backend::DynBackendDevice;
use crate::{CpuStorage, DType, Error, Result, Shape, Storage, WithDType};
use fuel_cpu_backend::dyn_impl::CpuBackendDevice;
use std::sync::Arc;

pub use fuel_core_types::DeviceLocation;

/// A device on which tensors can be created and computations performed.
///
/// Internally this is an `Arc<dyn DynBackendDevice>` — a trait-object handle
/// to the concrete backend device.  CPU, CUDA, Metal, and arbitrary third-party
/// backends are all accessed through this single type.
///
/// # Example
///
/// ```rust
/// use fuel_core::{Device, Tensor, DType};
/// let dev = Device::cpu();
/// let t = Tensor::zeros((2, 3), DType::F32, &dev)?;
/// assert_eq!(t.dims(), &[2, 3]);
/// # Ok::<(), fuel_core::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Device {
    pub(crate) inner: Arc<dyn DynBackendDevice>,
}

/// Trait for types that can be converted to tensor storage, providing shape and CPU data.
///
/// Implemented for scalars, arrays, slices, and nested vecs up to 4 dimensions.
/// This trait is what allows [`Tensor::new`](crate::Tensor::new) to accept many different
/// Rust types directly.
///
/// # Example
///
/// ```rust
/// use fuel_core::{Device, Tensor};
/// // Scalars, arrays, and nested arrays all implement NdArray
/// let scalar = Tensor::new(3.14f32, &Device::cpu())?;
/// let vec1d = Tensor::new(&[1f32, 2., 3.], &Device::cpu())?;
/// let mat2d = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::cpu())?;
/// assert_eq!(scalar.dims(), &[] as &[usize]);
/// assert_eq!(vec1d.dims(), &[3]);
/// assert_eq!(mat2d.dims(), &[2, 2]);
/// # Ok::<(), fuel_core::Error>(())
/// ```
pub trait NdArray {
    /// Returns the shape determined by this array-like value.
    fn shape(&self) -> Result<Shape>;

    /// Converts this value into CPU storage.
    fn to_cpu_storage(&self) -> CpuStorage;
}

impl<S: WithDType> NdArray for S {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(&[*self])
    }
}

impl<S: WithDType, const N: usize> NdArray for &[S; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self.as_slice())
    }
}

impl<S: WithDType> NdArray for &[S] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self)
    }
}

impl<S: WithDType, const N: usize, const M: usize> NdArray for &[[S; N]; M] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((M, N)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage_owned(self.concat())
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize> NdArray
    for &[[[S; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let mut vec = Vec::with_capacity(N1 * N2 * N3);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                vec.extend(self[i1][i2])
            }
        }
        S::to_cpu_storage_owned(vec)
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize, const N4: usize> NdArray
    for &[[[[S; N4]; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3, N4)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let mut vec = Vec::with_capacity(N1 * N2 * N3 * N4);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                for i3 in 0..N3 {
                    vec.extend(self[i1][i2][i3])
                }
            }
        }
        S::to_cpu_storage_owned(vec)
    }
}

impl<S: WithDType> NdArray for Vec<S> {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self.as_slice())
    }
}

impl<S: WithDType> NdArray for Vec<&[S]> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let n = self.len();
        let m = self[0].len();
        for v in self.iter() {
            if v.len() != m {
                crate::bail!("two elements have different len {m} {}", v.len())
            }
        }
        Ok(Shape::from((n, m)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let data = self.iter().copied().flatten().copied().collect::<Vec<_>>();
        S::to_cpu_storage_owned(data)
    }
}

impl<S: WithDType> NdArray for Vec<Vec<S>> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let n = self.len();
        let m = self[0].len();
        for v in self.iter() {
            if v.len() != m {
                crate::bail!("two elements have different len {m} {}", v.len())
            }
        }
        Ok(Shape::from((n, m)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let len: usize = self.iter().map(|v| v.len()).sum();
        let mut dst = Vec::with_capacity(len);
        for v in self.iter() {
            dst.extend(v.iter().copied());
        }
        S::to_cpu_storage_owned(dst)
    }
}

impl<S: WithDType> NdArray for Vec<Vec<Vec<S>>> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let shape0 = self[0].shape()?;
        let n = self.len();
        for v in self.iter() {
            let shape = v.shape()?;
            if shape != shape0 {
                crate::bail!("two elements have different shapes {shape:?} {shape0:?}")
            }
        }
        Ok(Shape::from([[n].as_slice(), shape0.dims()].concat()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        if self.is_empty() {
            return S::to_cpu_storage_owned(vec![]);
        }
        let len: usize = self
            .iter()
            .map(|v| v.iter().map(|v| v.len()).sum::<usize>())
            .sum();
        let mut dst = Vec::with_capacity(len);
        for v1 in self.iter() {
            for v2 in v1.iter() {
                dst.extend(v2.iter().copied());
            }
        }
        S::to_cpu_storage_owned(dst)
    }
}

impl<S: WithDType> NdArray for Vec<Vec<Vec<Vec<S>>>> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let shape0 = self[0].shape()?;
        let n = self.len();
        for v in self.iter() {
            let shape = v.shape()?;
            if shape != shape0 {
                crate::bail!("two elements have different shapes {shape:?} {shape0:?}")
            }
        }
        Ok(Shape::from([[n].as_slice(), shape0.dims()].concat()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let len: usize = self
            .iter()
            .map(|v| {
                v.iter()
                    .map(|v| v.iter().map(|v| v.len()).sum::<usize>())
                    .sum::<usize>()
            })
            .sum();
        let mut dst = Vec::with_capacity(len);
        for v1 in self.iter() {
            for v2 in v1.iter() {
                for v3 in v2.iter() {
                    dst.extend(v3.iter().copied());
                }
            }
        }
        S::to_cpu_storage_owned(dst)
    }
}

impl Device {
    /// Returns a CPU device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use fuel_core::Device;
    /// let dev = Device::cpu();
    /// assert!(dev.is_cpu());
    /// ```
    pub fn cpu() -> Self {
        Device {
            inner: Arc::new(CpuBackendDevice),
        }
    }

    /// Creates a new CUDA device with the given GPU ordinal.
    ///
    /// Requires CUDA support compiled in and a compatible GPU.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use fuel_core::Device;
    /// let dev = Device::new_cuda(0)?;
    /// assert!(dev.is_cuda());
    /// # Ok::<(), fuel_core::Error>(())
    /// ```
    pub fn new_cuda(ordinal: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let dev = crate::CudaDevice::new(ordinal)?;
            Ok(Device {
                inner: Arc::new(fuel_cuda::CudaBackendDevice(dev)),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ordinal;
            Err(Error::NotCompiledWithCudaSupport.bt())
        }
    }

    /// Returns the underlying CUDA device, or an error if this is not a CUDA device.
    pub fn as_cuda_device(&self) -> Result<&crate::CudaDevice> {
        #[cfg(feature = "cuda")]
        {
            self.inner
                .as_any()
                .downcast_ref::<fuel_cuda::CudaBackendDevice>()
                .map(|d| &d.0)
                .ok_or_else(|| Error::Msg("expected a cuda device".into()).bt())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::NotCompiledWithCudaSupport.bt())
        }
    }

    /// Returns the underlying Metal device, or an error if this is not a Metal device.
    pub fn as_metal_device(&self) -> Result<&crate::MetalDevice> {
        #[cfg(feature = "metal")]
        {
            self.inner
                .as_any()
                .downcast_ref::<fuel_metal::MetalBackendDevice>()
                .map(|d| &d.0)
                .ok_or_else(|| Error::Msg("expected a metal device".into()).bt())
        }
        #[cfg(not(feature = "metal"))]
        {
            Err(Error::NotCompiledWithMetalSupport.bt())
        }
    }

    /// Creates a new CUDA device with a dedicated stream.
    pub fn new_cuda_with_stream(ordinal: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let dev = crate::CudaDevice::new_with_stream(ordinal)?;
            Ok(Device {
                inner: Arc::new(fuel_cuda::CudaBackendDevice(dev)),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ordinal;
            Err(Error::NotCompiledWithCudaSupport.bt())
        }
    }

    /// Creates a new Metal device with the given ordinal.
    pub fn new_metal(ordinal: usize) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            let dev = crate::MetalDevice::new(ordinal)?;
            Ok(Device {
                inner: Arc::new(fuel_metal::MetalBackendDevice(dev)),
            })
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = ordinal;
            Err(Error::NotCompiledWithMetalSupport.bt())
        }
    }

    /// Creates a device backed by a custom [`DynBackendDevice`].
    ///
    /// The device uses dynamic dispatch for all operations, enabling
    /// third-party backends without modifying `fuel-core`.
    pub fn custom(device: Arc<dyn DynBackendDevice>) -> Self {
        Device { inner: device }
    }

    /// Wraps an existing [`CudaDevice`](crate::CudaDevice) into a `Device`.
    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_device(dev: crate::CudaDevice) -> Self {
        Device {
            inner: Arc::new(fuel_cuda::CudaBackendDevice(dev)),
        }
    }

    /// Wraps an existing [`MetalDevice`](crate::MetalDevice) into a `Device`.
    #[cfg(feature = "metal")]
    pub(crate) fn from_metal_device(dev: crate::MetalDevice) -> Self {
        Device {
            inner: Arc::new(fuel_metal::MetalBackendDevice(dev)),
        }
    }

    /// Returns `true` if this is a custom (third-party) device.
    pub fn is_custom(&self) -> bool {
        !self.is_cpu() && !self.is_cuda() && !self.is_metal()
    }

    /// Sets the random seed for this device's random number generator.
    pub fn set_seed(&self, seed: u64) -> Result<()> {
        self.inner.set_seed_dyn(seed)
    }

    /// Returns the current random seed for this device.
    pub fn get_current_seed(&self) -> Result<u64> {
        self.inner.get_current_seed_dyn()
    }

    /// Returns `true` if both devices refer to the same physical device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use fuel_core::Device;
    /// assert!(Device::cpu().same_device(&Device::cpu()));
    /// ```
    pub fn same_device(&self, rhs: &Self) -> bool {
        self.inner.same_device_dyn(&*rhs.inner)
    }

    /// Returns the physical [`DeviceLocation`] for this device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use fuel_core::{Device, DeviceLocation};
    /// assert_eq!(Device::cpu().location(), DeviceLocation::Cpu);
    /// ```
    pub fn location(&self) -> DeviceLocation {
        self.inner.location_dyn()
    }

    /// Returns `true` if this is the CPU device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use fuel_core::Device;
    /// assert!(Device::cpu().is_cpu());
    /// ```
    pub fn is_cpu(&self) -> bool {
        matches!(self.location(), DeviceLocation::Cpu)
    }

    /// Returns `true` if this is a CUDA device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use fuel_core::Device;
    /// assert!(!Device::cpu().is_cuda());
    /// ```
    pub fn is_cuda(&self) -> bool {
        matches!(self.location(), DeviceLocation::Cuda { .. })
    }

    /// Returns `true` if this is a Metal device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use fuel_core::Device;
    /// assert!(!Device::cpu().is_metal());
    /// ```
    pub fn is_metal(&self) -> bool {
        matches!(self.location(), DeviceLocation::Metal { .. })
    }

    /// Returns `true` if this device has native BF16 support.
    ///
    /// # Example
    ///
    /// ```rust
    /// use fuel_core::Device;
    /// // CPU does not have native BF16 support
    /// assert!(!Device::cpu().supports_bf16());
    /// ```
    pub fn supports_bf16(&self) -> bool {
        self.inner.supports_bf16()
    }

    /// Returns [`DType::BF16`] if supported, otherwise [`DType::F32`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use fuel_core::{Device, DType};
    /// assert_eq!(Device::cpu().bf16_default_to_f32(), DType::F32);
    /// ```
    pub fn bf16_default_to_f32(&self) -> DType {
        if self.supports_bf16() {
            DType::BF16
        } else {
            DType::F32
        }
    }

    /// Returns a CUDA device if available, otherwise falls back to CPU.
    pub fn cuda_if_available(ordinal: usize) -> Result<Self> {
        if crate::utils::cuda_is_available() {
            Self::new_cuda(ordinal)
        } else {
            Ok(Self::cpu())
        }
    }

    /// Returns a Metal device if available, otherwise falls back to CPU.
    pub fn metal_if_available(ordinal: usize) -> Result<Self> {
        if crate::utils::metal_is_available() {
            Self::new_metal(ordinal)
        } else {
            Ok(Self::cpu())
        }
    }

    pub(crate) fn rand_uniform_f64(
        &self,
        lo: f64,
        up: f64,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Storage> {
        // CUDA doesn't support generating F16/BF16 directly; generate as F32 then convert.
        if self.is_cuda() && (dtype == DType::F16 || dtype == DType::BF16) {
            let storage = Storage(self.inner.rand_uniform_dyn(shape, DType::F32, lo, up)?);
            storage.to_dtype(&crate::Layout::contiguous(shape), dtype)
        } else {
            Ok(Storage(self.inner.rand_uniform_dyn(shape, dtype, lo, up)?))
        }
    }

    pub(crate) fn rand_uniform<T: crate::FloatDType>(
        &self,
        lo: T,
        up: T,
        shape: &Shape,
    ) -> Result<Storage> {
        self.rand_uniform_f64(lo.to_f64(), up.to_f64(), shape, T::DTYPE)
    }

    pub(crate) fn rand_normal_f64(
        &self,
        mean: f64,
        std: f64,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Storage> {
        // CUDA doesn't support generating F16/BF16 directly; generate as F32 then convert.
        if self.is_cuda() && (dtype == DType::F16 || dtype == DType::BF16) {
            let storage = Storage(self.inner.rand_normal_dyn(shape, DType::F32, mean, std)?);
            storage.to_dtype(&crate::Layout::contiguous(shape), dtype)
        } else {
            Ok(Storage(self.inner.rand_normal_dyn(shape, dtype, mean, std)?))
        }
    }

    pub(crate) fn rand_normal<T: crate::FloatDType>(
        &self,
        mean: T,
        std: T,
        shape: &Shape,
    ) -> Result<Storage> {
        self.rand_normal_f64(mean.to_f64(), std.to_f64(), shape, T::DTYPE)
    }

    pub(crate) fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        Ok(Storage(self.inner.zeros_impl_dyn(shape, dtype)?))
    }

    pub(crate) unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        Ok(Storage(unsafe {
            self.inner.alloc_uninit_dyn(shape, dtype)?
        }))
    }

    pub(crate) fn storage_from_slice<D: WithDType>(&self, data: &[D]) -> Result<Storage> {
        let cpu = data.to_cpu_storage();
        Ok(Storage(self.inner.storage_from_cpu_storage_owned_dyn(cpu)?))
    }

    pub(crate) fn storage<A: NdArray>(&self, array: A) -> Result<Storage> {
        let cpu = array.to_cpu_storage();
        Ok(Storage(self.inner.storage_from_cpu_storage_owned_dyn(cpu)?))
    }

    pub(crate) fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Storage> {
        let cpu = S::to_cpu_storage_owned(data);
        Ok(Storage(self.inner.storage_from_cpu_storage_owned_dyn(cpu)?))
    }

    /// Synchronizes the device, waiting for all pending operations to complete.
    ///
    /// This is a no-op on CPU.
    ///
    /// ```rust
    /// use fuel_core::Device;
    /// Device::cpu().synchronize()?;
    /// # Ok::<(), fuel_core::Error>(())
    /// ```
    pub fn synchronize(&self) -> Result<()> {
        self.inner.synchronize_dyn()
    }
}

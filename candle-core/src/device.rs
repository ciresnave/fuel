//! Device abstraction for CPU, CUDA, and Metal backends.
//!
//! ```rust
//! use candle_core::Device;
//! let dev = Device::Cpu;
//! assert!(dev.is_cpu());
//! assert_eq!(dev.location(), candle_core::DeviceLocation::Cpu);
//! ```
use crate::backend::BackendDevice;
use crate::cpu_backend::CpuDevice;
use crate::dyn_backend::DynBackendDevice;
use crate::{CpuStorage, DType, Result, Shape, Storage, WithDType};
use std::sync::Arc;

pub use candle_core_types::DeviceLocation;

/// A device on which tensors can be created and computations performed.
///
/// # Example
///
/// ```rust
/// use candle_core::{Device, Tensor, DType};
/// let dev = Device::Cpu;
/// let t = Tensor::zeros((2, 3), DType::F32, &dev)?;
/// assert_eq!(t.dims(), &[2, 3]);
/// # Ok::<(), candle_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub enum Device {
    /// The CPU backend.
    Cpu,
    /// A CUDA GPU backend.
    Cuda(crate::CudaDevice),
    /// An Apple Metal GPU backend.
    Metal(crate::MetalDevice),
    /// A third-party backend, accessed through dynamic dispatch.
    ///
    /// The `Cpu`, `Cuda`, and `Metal` arms retain zero-overhead static dispatch.
    /// The `Custom` arm pays `Arc<dyn>` indirection, which is acceptable because
    /// the alternative is forking `candle-core`.
    Custom(Arc<dyn DynBackendDevice>),
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
/// use candle_core::{Device, Tensor};
/// // Scalars, arrays, and nested arrays all implement NdArray
/// let scalar = Tensor::new(3.14f32, &Device::Cpu)?;
/// let vec1d = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
/// let mat2d = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;
/// assert_eq!(scalar.dims(), &[] as &[usize]);
/// assert_eq!(vec1d.dims(), &[3]);
/// assert_eq!(mat2d.dims(), &[2, 2]);
/// # Ok::<(), candle_core::Error>(())
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
    /// Creates a new CUDA device with the given GPU ordinal.
    ///
    /// Requires CUDA support compiled in and a compatible GPU.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use candle_core::Device;
    /// let dev = Device::new_cuda(0)?;
    /// assert!(dev.is_cuda());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn new_cuda(ordinal: usize) -> Result<Self> {
        Ok(Self::Cuda(crate::CudaDevice::new(ordinal)?))
    }

    /// Returns the underlying CUDA device, or an error if this is not a CUDA device.
    pub fn as_cuda_device(&self) -> Result<&crate::CudaDevice> {
        match self {
            Self::Cuda(d) => Ok(d),
            Self::Cpu => crate::bail!("expected a cuda device, got cpu"),
            Self::Metal(_) => crate::bail!("expected a cuda device, got Metal"),
            Self::Custom(_) => crate::bail!("expected a cuda device, got custom"),
        }
    }

    /// Returns the underlying Metal device, or an error if this is not a Metal device.
    pub fn as_metal_device(&self) -> Result<&crate::MetalDevice> {
        match self {
            Self::Cuda(_) => crate::bail!("expected a metal device, got cuda"),
            Self::Cpu => crate::bail!("expected a metal device, got cpu"),
            Self::Metal(d) => Ok(d),
            Self::Custom(_) => crate::bail!("expected a metal device, got custom"),
        }
    }

    /// Creates a new CUDA device with a dedicated stream.
    pub fn new_cuda_with_stream(ordinal: usize) -> Result<Self> {
        Ok(Self::Cuda(crate::CudaDevice::new_with_stream(ordinal)?))
    }

    /// Creates a new Metal device with the given ordinal.
    pub fn new_metal(ordinal: usize) -> Result<Self> {
        Ok(Self::Metal(crate::MetalDevice::new(ordinal)?))
    }

    /// Creates a device backed by a custom [`DynBackendDevice`].
    ///
    /// The device uses dynamic dispatch for all operations, enabling
    /// third-party backends without modifying `candle-core`.
    pub fn custom(device: Arc<dyn DynBackendDevice>) -> Self {
        Self::Custom(device)
    }

    /// Returns `true` if this is a custom (third-party) device.
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom(_))
    }

    /// Sets the random seed for this device's random number generator.
    ///
    /// Only supported on CUDA and Metal devices.
    pub fn set_seed(&self, seed: u64) -> Result<()> {
        match self {
            Self::Cpu => CpuDevice.set_seed(seed),
            Self::Cuda(c) => c.set_seed(seed),
            Self::Metal(m) => m.set_seed(seed),
            Self::Custom(d) => d.set_seed_dyn(seed),
        }
    }

    /// Returns the current random seed for this device.
    ///
    /// Only supported on CUDA and Metal devices.
    pub fn get_current_seed(&self) -> Result<u64> {
        match self {
            Self::Cpu => CpuDevice.get_current_seed(),
            Self::Cuda(c) => c.get_current_seed(),
            Self::Metal(m) => m.get_current_seed(),
            Self::Custom(d) => d.get_current_seed_dyn(),
        }
    }

    /// Returns `true` if both devices refer to the same physical device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::Device;
    /// assert!(Device::Cpu.same_device(&Device::Cpu));
    /// ```
    pub fn same_device(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Cpu, Self::Cpu) => true,
            (Self::Cuda(lhs), Self::Cuda(rhs)) => lhs.same_device(rhs),
            (Self::Metal(lhs), Self::Metal(rhs)) => lhs.same_device(rhs),
            _ => false,
        }
    }

    /// Returns the physical [`DeviceLocation`] for this device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::{Device, DeviceLocation};
    /// assert_eq!(Device::Cpu.location(), DeviceLocation::Cpu);
    /// ```
    pub fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,
            Self::Cuda(device) => device.location(),
            Device::Metal(device) => device.location(),
            Device::Custom(device) => device.location_dyn(),
        }
    }

    /// Returns `true` if this is the CPU device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::Device;
    /// assert!(Device::Cpu.is_cpu());
    /// ```
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Returns `true` if this is a CUDA device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::Device;
    /// assert!(!Device::Cpu.is_cuda());
    /// ```
    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    /// Returns `true` if this is a Metal device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::Device;
    /// assert!(!Device::Cpu.is_metal());
    /// ```
    pub fn is_metal(&self) -> bool {
        matches!(self, Self::Metal(_))
    }

    /// Returns `true` if this device has native BF16 support.
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::Device;
    /// // CPU does not have native BF16 support
    /// assert!(!Device::Cpu.supports_bf16());
    /// ```
    pub fn supports_bf16(&self) -> bool {
        match self {
            Self::Cuda(_) | Self::Metal(_) => true,
            Self::Cpu => false,
            Self::Custom(_) => false,
        }
    }

    /// Returns [`DType::BF16`] if supported, otherwise [`DType::F32`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_core::{Device, DType};
    /// assert_eq!(Device::Cpu.bf16_default_to_f32(), DType::F32);
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
            Ok(Self::Cpu)
        }
    }

    /// Returns a Metal device if available, otherwise falls back to CPU.
    pub fn metal_if_available(ordinal: usize) -> Result<Self> {
        if crate::utils::metal_is_available() {
            Self::new_metal(ordinal)
        } else {
            Ok(Self::Cpu)
        }
    }

    pub(crate) fn rand_uniform_f64(
        &self,
        lo: f64,
        up: f64,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.rand_uniform(shape, dtype, lo, up)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                // TODO: Remove the special case if we start supporting generating f16/bf16 directly.
                if dtype == DType::F16 || dtype == DType::BF16 {
                    let storage = device.rand_uniform(shape, DType::F32, lo, up)?;
                    Storage::Cuda(storage).to_dtype(&crate::Layout::contiguous(shape), dtype)
                } else {
                    let storage = device.rand_uniform(shape, dtype, lo, up)?;
                    Ok(Storage::Cuda(storage))
                }
            }
            Device::Metal(device) => {
                let storage = device.rand_uniform(shape, dtype, lo, up)?;
                Ok(Storage::Metal(storage))
            }
            Device::Custom(device) => {
                let storage = device.rand_uniform_dyn(shape, dtype, lo, up)?;
                Ok(Storage::Custom(storage))
            }
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
        match self {
            Device::Cpu => {
                let storage = CpuDevice.rand_normal(shape, dtype, mean, std)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                // TODO: Remove the special case if we start supporting generating f16/bf16 directly.
                if dtype == DType::F16 || dtype == DType::BF16 {
                    let storage = device.rand_normal(shape, DType::F32, mean, std)?;
                    Storage::Cuda(storage).to_dtype(&crate::Layout::contiguous(shape), dtype)
                } else {
                    let storage = device.rand_normal(shape, dtype, mean, std)?;
                    Ok(Storage::Cuda(storage))
                }
            }
            Device::Metal(device) => {
                let storage = device.rand_normal(shape, dtype, mean, std)?;
                Ok(Storage::Metal(storage))
            }
            Device::Custom(device) => {
                let storage = device.rand_normal_dyn(shape, dtype, mean, std)?;
                Ok(Storage::Custom(storage))
            }
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
        match self {
            Device::Cpu => {
                let storage = CpuDevice.zeros_impl(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.zeros_impl(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.zeros_impl(shape, dtype)?;
                Ok(Storage::Metal(storage))
            }
            Device::Custom(device) => {
                let storage = device.zeros_impl_dyn(shape, dtype)?;
                Ok(Storage::Custom(storage))
            }
        }
    }

    pub(crate) unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = unsafe { CpuDevice.alloc_uninit(shape, dtype)? };
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = unsafe { device.alloc_uninit(shape, dtype)? };
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = unsafe { device.alloc_uninit(shape, dtype)? };
                Ok(Storage::Metal(storage))
            }
            Device::Custom(device) => {
                let storage = unsafe { device.alloc_uninit_dyn(shape, dtype)? };
                Ok(Storage::Custom(storage))
            }
        }
    }

    pub(crate) fn storage_from_slice<D: WithDType>(&self, data: &[D]) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(data.to_cpu_storage())),
            Device::Cuda(device) => {
                let storage = device.storage_from_slice(data)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.storage_from_slice(data)?;
                Ok(Storage::Metal(storage))
            }
            Device::Custom(device) => {
                let cpu = data.to_cpu_storage();
                let storage = device.storage_from_cpu_storage_owned_dyn(cpu)?;
                Ok(Storage::Custom(storage))
            }
        }
    }

    pub(crate) fn storage<A: NdArray>(&self, array: A) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(array.to_cpu_storage())),
            Device::Cuda(device) => {
                let storage = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Metal(storage))
            }
            Device::Custom(device) => {
                let cpu = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage_owned_dyn(cpu)?;
                Ok(Storage::Custom(storage))
            }
        }
    }

    pub(crate) fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(S::to_cpu_storage_owned(data))),
            Device::Cuda(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Metal(storage))
            }
            Device::Custom(device) => {
                let cpu = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage_owned_dyn(cpu)?;
                Ok(Storage::Custom(storage))
            }
        }
    }

    /// Synchronizes the device, waiting for all pending operations to complete.
    ///
    /// This is a no-op on CPU.
    ///
    /// ```rust
    /// use candle_core::Device;
    /// Device::Cpu.synchronize()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn synchronize(&self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Cuda(d) => d.synchronize(),
            Self::Metal(d) => d.synchronize(),
            Self::Custom(d) => d.synchronize_dyn(),
        }
    }
}

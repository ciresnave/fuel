use crate::{HostBuffer, DType, Device, Result, Shape, Storage, Tensor, D};
use k_quants::*;
use std::borrow::Cow;

#[cfg(target_feature = "avx2")]
pub mod avx;
mod dummy_cuda;
mod dummy_metal;
pub mod arch;
pub mod ggml_file;
pub mod gguf_file;
#[cfg(not(target_arch = "wasm32"))]
pub mod gguf_mmap;
pub mod imatrix_file;
pub mod k_quants;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(not(target_arch = "wasm32"))]
pub mod tokenizer;
#[cfg(not(feature = "metal"))]
mod metal {
    pub use super::dummy_metal::*;
}
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(not(feature = "cuda"))]
mod cuda {
    pub use super::dummy_cuda::*;
}

#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(target_feature = "simd128")]
pub mod simd128;
pub mod utils;
use half::{bf16, f16};

pub use k_quants::GgmlType;

// ---------------------------------------------------------------------------
// Internal downcast helpers (replace the deleted public Storage::as_*_storage
// accessors). These are file-local; quantized/mod.rs is the only legacy
// fuel-core consumer that still needs to peel back to concrete backend
// storage types.
// ---------------------------------------------------------------------------

fn as_cpu(s: &Storage) -> Result<&HostBuffer> {
    s.downcast_ref::<fuel_cpu_backend::CpuStorage>()
        .map(|s| &s.0)
        .ok_or_else(|| crate::Error::Msg("expected cpu storage".into()).bt())
}

#[cfg(feature = "cuda")]
fn as_cuda(s: &Storage) -> Option<&crate::CudaStorage> {
    s.downcast_ref::<crate::CudaStorage>()
}

#[cfg(feature = "metal")]
fn as_metal(s: &Storage) -> Option<&crate::MetalStorage> {
    s.downcast_ref::<crate::MetalStorage>()
}

fn as_t_slice<T>(data: Cow<'_, [u8]>) -> &[T] {
    let size = std::mem::size_of::<T>();
    assert_eq!(
        data.len() % size,
        0,
        "Data length must be a multiple of T's size"
    );
    let ptr = data.as_ptr();
    assert_eq!(
        (ptr as usize) % std::mem::align_of::<T>(),
        0,
        "Data pointer must be aligned to T's alignment"
    );
    unsafe { std::slice::from_raw_parts(ptr as *const T, data.len() / size) }
}

pub struct QTensor {
    storage: QStorage,
    shape: Shape,
}

impl Device {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
        if self.is_cpu() {
            let storage = dtype.cpu_zeros(elem_count);
            return Ok(QStorage::Cpu(storage));
        }
        #[cfg(feature = "metal")]
        if let Ok(metal) = self.as_metal_device() {
            let storage = metal::QMetalStorage::zeros(metal, elem_count, dtype)?;
            return Ok(QStorage::Metal(storage));
        }
        #[cfg(feature = "cuda")]
        if let Ok(cuda) = self.as_cuda_device() {
            let storage = cuda::QCudaStorage::zeros(cuda, elem_count, dtype)?;
            return Ok(QStorage::Cuda(storage));
        }
        Err(crate::Error::Msg(
            "quantized tensors are not supported on this backend".to_string(),
        ))
    }
}

pub enum QStorage {
    Cpu(Box<dyn QuantizedType>),
    Metal(metal::QMetalStorage),
    Cuda(cuda::QCudaStorage),
}

impl QStorage {
    pub fn from_data(data: Cow<'_, [u8]>, device: &Device, dtype: GgmlDType) -> Result<Self> {
        if device.is_cpu() {
            return Ok(Self::Cpu(dtype.from_data(data)));
        }
        #[cfg(feature = "metal")]
        if let Ok(d) = device.as_metal_device() {
            return match dtype {
                GgmlDType::F32 => metal::load_quantized(d, as_t_slice::<f32>(data)),
                GgmlDType::F16 => metal::load_quantized(d, as_t_slice::<f16>(data)),
                GgmlDType::Q4_0 => metal::load_quantized(d, as_t_slice::<BlockQ4_0>(data)),
                GgmlDType::Q4_1 => metal::load_quantized(d, as_t_slice::<BlockQ4_1>(data)),
                GgmlDType::Q5_0 => metal::load_quantized(d, as_t_slice::<BlockQ5_0>(data)),
                GgmlDType::Q5_1 => metal::load_quantized(d, as_t_slice::<BlockQ5_1>(data)),
                GgmlDType::Q8_0 => metal::load_quantized(d, as_t_slice::<BlockQ8_0>(data)),
                GgmlDType::Q8_1 => metal::load_quantized(d, as_t_slice::<BlockQ8_1>(data)),
                GgmlDType::Q2K => metal::load_quantized(d, as_t_slice::<BlockQ2K>(data)),
                GgmlDType::Q3K => metal::load_quantized(d, as_t_slice::<BlockQ3K>(data)),
                GgmlDType::Q4K => metal::load_quantized(d, as_t_slice::<BlockQ4K>(data)),
                GgmlDType::Q5K => metal::load_quantized(d, as_t_slice::<BlockQ5K>(data)),
                GgmlDType::Q6K => metal::load_quantized(d, as_t_slice::<BlockQ6K>(data)),
                GgmlDType::Q8K => metal::load_quantized(d, as_t_slice::<BlockQ8K>(data)),
                GgmlDType::BF16 => metal::load_quantized(d, as_t_slice::<bf16>(data)),
            };
        }
        #[cfg(feature = "cuda")]
        if let Ok(d) = device.as_cuda_device() {
            return match dtype {
                GgmlDType::F32 => cuda::load_quantized(d, as_t_slice::<f32>(data)),
                GgmlDType::F16 => cuda::load_quantized(d, as_t_slice::<f16>(data)),
                GgmlDType::Q4_0 => cuda::load_quantized(d, as_t_slice::<BlockQ4_0>(data)),
                GgmlDType::Q4_1 => cuda::load_quantized(d, as_t_slice::<BlockQ4_1>(data)),
                GgmlDType::Q5_0 => cuda::load_quantized(d, as_t_slice::<BlockQ5_0>(data)),
                GgmlDType::Q5_1 => cuda::load_quantized(d, as_t_slice::<BlockQ5_1>(data)),
                GgmlDType::Q8_0 => cuda::load_quantized(d, as_t_slice::<BlockQ8_0>(data)),
                GgmlDType::Q8_1 => cuda::load_quantized(d, as_t_slice::<BlockQ8_1>(data)),
                GgmlDType::Q2K => cuda::load_quantized(d, as_t_slice::<BlockQ2K>(data)),
                GgmlDType::Q3K => cuda::load_quantized(d, as_t_slice::<BlockQ3K>(data)),
                GgmlDType::Q4K => cuda::load_quantized(d, as_t_slice::<BlockQ4K>(data)),
                GgmlDType::Q5K => cuda::load_quantized(d, as_t_slice::<BlockQ5K>(data)),
                GgmlDType::Q6K => cuda::load_quantized(d, as_t_slice::<BlockQ6K>(data)),
                GgmlDType::Q8K => cuda::load_quantized(d, as_t_slice::<BlockQ8K>(data)),
                GgmlDType::BF16 => cuda::load_quantized(d, as_t_slice::<bf16>(data)),
            };
        }
        Err(crate::Error::Msg(
            "quantized tensors are not supported on this backend".to_string(),
        ))
    }

    fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.block_size(),
            QStorage::Metal(storage) => storage.dtype().block_size(),
            QStorage::Cuda(storage) => storage.dtype().block_size(),
        }
    }

    fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
            QStorage::Metal(storage) => storage.dtype(),
            QStorage::Cuda(storage) => storage.dtype(),
        }
    }

    fn device(&self) -> Device {
        match self {
            QStorage::Cpu(_storage) => Device::cpu(),
            QStorage::Metal(storage) => {
                #[cfg(feature = "metal")]
                { Device::from_metal_device(storage.device().clone()) }
                #[cfg(not(feature = "metal"))]
                { let _ = storage; unreachable!() }
            }
            QStorage::Cuda(storage) => {
                #[cfg(feature = "cuda")]
                { Device::from_cuda_device(storage.device().clone()) }
                #[cfg(not(feature = "cuda"))]
                { let _ = storage; unreachable!() }
            }
        }
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
            QStorage::Metal(storage) => storage.storage_size_in_bytes(),
            QStorage::Cuda(storage) => storage.storage_size_in_bytes(),
        }
    }

    fn quantize(&mut self, src: &Storage) -> Result<()> {
        match self {
            QStorage::Cpu(storage) => {
                let src = as_cpu(src)?;
                storage.from_float(src.as_slice::<f32>()?);
            }
            QStorage::Metal(storage) => {
                #[cfg(feature = "metal")]
                {
                    let metal_src = as_metal(src)
                        .ok_or_else(|| crate::Error::Msg("quantize: expected metal storage".into()))?;
                    storage.quantize(metal_src)?;
                }
                #[cfg(not(feature = "metal"))]
                { let _ = (storage, src); unreachable!(); }
            }
            QStorage::Cuda(storage) => {
                #[cfg(feature = "cuda")]
                {
                    let cuda_src = as_cuda(src)
                        .ok_or_else(|| crate::Error::Msg("quantize: expected cuda storage".into()))?;
                    storage.quantize(cuda_src)?;
                }
                #[cfg(not(feature = "cuda"))]
                { let _ = (storage, src); unreachable!(); }
            }
        }
        Ok(())
    }

    fn quantize_imatrix(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match self {
            QStorage::Cpu(storage) => {
                let src = as_cpu(src)?;
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            QStorage::Metal(storage) => {
                #[cfg(feature = "metal")]
                {
                    let metal_src = as_metal(src)
                        .ok_or_else(|| crate::Error::Msg("quantize_imatrix: expected metal storage".into()))?;
                    storage.quantize_imatrix(metal_src, imatrix_weights, n_per_row)?;
                }
                #[cfg(not(feature = "metal"))]
                { let _ = (storage, src, imatrix_weights, n_per_row); unreachable!(); }
            }
            QStorage::Cuda(storage) => {
                #[cfg(feature = "cuda")]
                {
                    let cuda_src = as_cuda(src)
                        .ok_or_else(|| crate::Error::Msg("quantize_imatrix: expected cuda storage".into()))?;
                    storage.quantize_imatrix(cuda_src, imatrix_weights, n_per_row)?;
                }
                #[cfg(not(feature = "cuda"))]
                { let _ = (storage, src, imatrix_weights, n_per_row); unreachable!(); }
            }
        }
        Ok(())
    }

    fn quantize_onto(&mut self, src: &Storage) -> Result<()> {
        let cpu_src = as_cpu(src)?;
        match self {
            QStorage::Cpu(storage) => {
                storage.from_float(cpu_src.as_slice::<f32>()?);
            }
            QStorage::Metal(storage) => {
                #[cfg(feature = "metal")]
                storage.quantize_onto(cpu_src)?;
                #[cfg(not(feature = "metal"))]
                { let _ = storage; unreachable!(); }
            }
            QStorage::Cuda(storage) => {
                #[cfg(feature = "cuda")]
                storage.quantize_onto(cpu_src)?;
                #[cfg(not(feature = "cuda"))]
                { let _ = storage; unreachable!(); }
            }
        }
        Ok(())
    }

    fn quantize_imatrix_onto(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let cpu_src = as_cpu(src)?;
        match self {
            QStorage::Cpu(storage) => {
                storage.from_float_imatrix(cpu_src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            QStorage::Metal(storage) => {
                #[cfg(feature = "metal")]
                storage.quantize_imatrix_onto(cpu_src, imatrix_weights, n_per_row)?;
                #[cfg(not(feature = "metal"))]
                { let _ = (storage, imatrix_weights, n_per_row); unreachable!(); }
            }
            QStorage::Cuda(storage) => {
                #[cfg(feature = "cuda")]
                storage.quantize_imatrix_onto(cpu_src, imatrix_weights, n_per_row)?;
                #[cfg(not(feature = "cuda"))]
                { let _ = (storage, imatrix_weights, n_per_row); unreachable!(); }
            }
        }
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> Result<Storage> {
        match self {
            QStorage::Cpu(storage) => Ok(Storage::new(fuel_cpu_backend::CpuStorage(storage.dequantize(elem_count)?))),
            QStorage::Metal(storage) => {
                #[cfg(feature = "metal")]
                { Ok(Storage::new(storage.dequantize(elem_count)?)) }
                #[cfg(not(feature = "metal"))]
                { let _ = (storage, elem_count); unreachable!() }
            }
            QStorage::Cuda(storage) => {
                #[cfg(feature = "cuda")]
                { Ok(Storage::new(storage.dequantize(elem_count)?)) }
                #[cfg(not(feature = "cuda"))]
                { let _ = (storage, elem_count); unreachable!() }
            }
        }
    }

    fn data(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            QStorage::Cpu(storage) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
            QStorage::Cuda(storage) => Ok(Cow::from(storage.data()?)),
            QStorage::Metal(storage) => Ok(Cow::from(storage.data()?)),
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) => {
                crate::bail!("not implemented");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlDType {
    pub(crate) fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => Self::BF16,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    pub(crate) fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            Self::BF16 => 30,
        }
    }

    /// The block dtype
    pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
        match self {
            Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
            Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
            Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
            Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
            Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
            Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
            Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
            Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
            Self::Q2K => Box::new(vec![BlockQ2K::zeros(); elem_count / BlockQ2K::BLCK_SIZE]),
            Self::Q3K => Box::new(vec![BlockQ3K::zeros(); elem_count / BlockQ3K::BLCK_SIZE]),
            Self::Q4K => Box::new(vec![BlockQ4K::zeros(); elem_count / BlockQ4K::BLCK_SIZE]),
            Self::Q5K => Box::new(vec![BlockQ5K::zeros(); elem_count / BlockQ5K::BLCK_SIZE]),
            Self::Q6K => Box::new(vec![BlockQ6K::zeros(); elem_count / BlockQ6K::BLCK_SIZE]),
            Self::Q8K => Box::new(vec![BlockQ8K::zeros(); elem_count / BlockQ8K::BLCK_SIZE]),
            Self::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
        }
    }

    pub fn from_data(&self, data: Cow<'_, [u8]>) -> Box<dyn QuantizedType> {
        match self {
            Self::F32 => Box::new(as_t_slice::<f32>(data).to_vec()),
            Self::F16 => Box::new(as_t_slice::<f16>(data).to_vec()),
            Self::Q4_0 => Box::new(as_t_slice::<BlockQ4_0>(data).to_vec()),
            Self::Q4_1 => Box::new(as_t_slice::<BlockQ4_1>(data).to_vec()),
            Self::Q5_0 => Box::new(as_t_slice::<BlockQ5_0>(data).to_vec()),
            Self::Q5_1 => Box::new(as_t_slice::<BlockQ5_1>(data).to_vec()),
            Self::Q8_0 => Box::new(as_t_slice::<BlockQ8_0>(data).to_vec()),
            Self::Q8_1 => Box::new(as_t_slice::<BlockQ8_1>(data).to_vec()),
            Self::Q2K => Box::new(as_t_slice::<BlockQ2K>(data).to_vec()),
            Self::Q3K => Box::new(as_t_slice::<BlockQ3K>(data).to_vec()),
            Self::Q4K => Box::new(as_t_slice::<BlockQ4K>(data).to_vec()),
            Self::Q5K => Box::new(as_t_slice::<BlockQ5K>(data).to_vec()),
            Self::Q6K => Box::new(as_t_slice::<BlockQ6K>(data).to_vec()),
            Self::Q8K => Box::new(as_t_slice::<BlockQ8K>(data).to_vec()),
            Self::BF16 => Box::new(as_t_slice::<bf16>(data).to_vec()),
        }
    }

    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 | Self::BF16 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => k_quants::QK_K,
        }
    }
}

// A version of GgmlType without `vec_dot` so that it can be dyn boxed.
pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()>;
    fn dequantize(&self, elem_count: usize) -> Result<HostBuffer>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn block_size(&self) -> usize;
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]);
    #[allow(clippy::wrong_self_convention)]
    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize);
    fn size(&self) -> usize;
}

impl<T: k_quants::GgmlType + Send + Sync> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()> {
        k_quants::matmul_f16(mkn, lhs, self.as_slice(), dst)
    }

    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }

    fn from_float(&mut self, xs: &[f32]) {
        T::from_float(xs, self)
    }

    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize) {
        T::from_float_imatrix(xs, self, imatrix_weights, n_per_row)
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn dequantize(&self, elem_count: usize) -> Result<HostBuffer> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys);
        Ok(HostBuffer::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }
}

impl std::fmt::Debug for QTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.shape, self.dtype())
    }
}

fn check_shape(shape: &Shape, block_size: usize) -> Result<()> {
    let dims = shape.dims();
    if dims.is_empty() {
        crate::bail!("scalar tensor cannot be quantized {shape:?}")
    }
    if !dims[dims.len() - 1].is_multiple_of(block_size) {
        crate::bail!(
            "quantized tensor must have their last dim divisible by block size {shape:?} {}",
            block_size
        )
    }
    Ok(())
}

impl QTensor {
    pub fn new<S: Into<Shape>>(storage: QStorage, shape: S) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, storage.block_size())?;
        Ok(Self { storage, shape })
    }

    pub fn quantize(src: &Tensor, dtype: GgmlDType) -> Result<Self> {
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    pub fn quantize_imatrix(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
    ) -> Result<Self> {
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }

        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            );
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize_imatrix(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_imatrix_onto(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
        dev: &Device,
    ) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_imatrix_onto(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_onto(src: &Tensor, dtype: GgmlDType, dev: &Device) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_onto(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let storage = self.storage.dequantize(self.shape.elem_count())?;
        let none = crate::op::BackpropOp::none();
        crate::tensor::from_storage(storage, self.shape.clone(), none, false).to_device(device)
    }

    pub fn dequantize_f16(&self, device: &Device) -> Result<Tensor> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/huggingface/fuel/issues/2136
        match &self.storage {
            QStorage::Cuda(s) => {
                let s = s.dequantize_f16(self.shape.elem_count())?;
                let none = crate::op::BackpropOp::none();
                #[cfg(feature = "cuda")]
                {
                    crate::tensor::from_storage(
                        Storage::new(s),
                        self.shape.clone(),
                        none,
                        false,
                    )
                    .to_device(device)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (s, none, device);
                    unreachable!()
                }
            }
            _ => {
                let s = self.dequantize(device)?.to_dtype(crate::DType::F16)?;
                Ok(s)
            }
        }
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        self.storage.data()
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match &self.storage {
            QStorage::Cuda(s) => {
                #[cfg(feature = "cuda")]
                {
                    let x_guard = x.storage();
                    let ids_guard = ids.storage();
                    let x_storage = as_cuda(&*x_guard)
                        .ok_or_else(|| crate::Error::Msg("indexed_moe_forward: x must be on CUDA".into()))?;
                    let ids_storage = as_cuda(&*ids_guard)
                        .ok_or_else(|| crate::Error::Msg("indexed_moe_forward: ids must be on CUDA".into()))?;
                    let (storage, out_shape) = s.indexed_moe_forward(
                        self.shape(),
                        x_storage,
                        x.layout(),
                        ids_storage,
                        ids.layout(),
                    )?;
                    Ok(crate::tensor::from_storage(
                        Storage::new(storage),
                        out_shape,
                        crate::op::BackpropOp::none(),
                        false,
                    ))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (s, x, ids);
                    unreachable!()
                }
            }
            _ => {
                panic!("indexed_moe_forward is not implemented in this platform!");
            }
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match &self.storage {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) => {
                crate::bail!("not implemented");
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum QMatMul {
    QTensor(std::sync::Arc<QTensor>),
    Tensor(Tensor),
    TensorF16(Tensor),
}

thread_local! {
    static DEQUANTIZE_ALL: bool = {
        match std::env::var("FUEL_DEQUANTIZE_ALL") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_F16: bool = {
        match std::env::var("FUEL_DEQUANTIZE_ALL_F16") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl QMatMul {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
            _ => DEQUANTIZE_ALL.with(|b| *b),
        };
        let t = if dequantize {
            let tensor = qtensor.dequantize(&qtensor.device())?;
            Self::Tensor(tensor)
        } else if DEQUANTIZE_ALL_F16.with(|b| *b) {
            let tensor = qtensor.dequantize_f16(&qtensor.device())?;
            Self::TensorF16(tensor)
        } else {
            Self::QTensor(qtensor)
        };
        Ok(t)
    }

    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Self::from_arc(std::sync::Arc::new(qtensor))
    }

    pub fn dequantize_f16(&self) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.dequantize_f16(&t.device()),
            Self::Tensor(t) => t.to_dtype(DType::F16),
            Self::TensorF16(t) => Ok(t.clone()),
        }
    }

    pub fn forward_via_f16(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.dequantize_f16()?;
        let in_dtype = xs.dtype();
        let w = match *xs.dims() {
            [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
            _ => w.t()?,
        };
        xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.indexed_moe_forward(x, ids),
            _ => {
                panic!("Not implemented!")
            }
        }
    }
}

impl crate::CustomOp1 for QTensor {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn fwd(
        &self,
        storage: &dyn crate::dyn_backend::DynBackendStorage,
        layout: &crate::Layout,
    ) -> Result<(Box<dyn crate::dyn_backend::DynBackendStorage>, Shape)> {
        if let Some(cpu) = storage
            .as_any()
            .downcast_ref::<fuel_cpu_backend::dyn_impl::CpuStorage>()
        {
            let storage = &cpu.0;
            if !layout.is_contiguous() {
                crate::bail!("input tensor is not contiguous {layout:?}")
            }
            let src_shape = layout.shape();
            // self is transposed so n is first then k.
            let (n, k) = self.shape.dims2()?;
            if src_shape.rank() < 2 {
                crate::bail!("input tensor has only one dimension {layout:?}")
            }
            let mut dst_shape = src_shape.dims().to_vec();
            let last_k = dst_shape.pop().unwrap();
            if last_k != k {
                crate::bail!("input tensor {layout:?} incompatible with {:?}", self.shape)
            }
            dst_shape.push(n);
            let dst_shape = Shape::from(dst_shape);
            #[allow(clippy::infallible_destructuring_match)]
            let self_storage = match &self.storage {
                QStorage::Cpu(storage) => storage,
                QStorage::Metal(_) | QStorage::Cuda(_) => crate::bail!("Invalid storage"),
            };
            let out = match storage.dtype() {
                DType::F32 => {
                    let slice = storage.as_slice::<f32>()?;
                    let slice = &slice
                        [layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                    let mut dst_storage = vec![0f32; dst_shape.elem_count()];
                    self_storage.matmul_t(
                        (dst_shape.elem_count() / n, k, n),
                        slice,
                        &mut dst_storage,
                    )?;
                    crate::HostBuffer::F32(dst_storage)
                }
                DType::F16 => {
                    let slice = storage.as_slice::<f16>()?;
                    let slice = &slice
                        [layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                    let mut dst_storage = vec![f16::ZERO; dst_shape.elem_count()];
                    self_storage.matmul_t_f16(
                        (dst_shape.elem_count() / n, k, n),
                        slice,
                        &mut dst_storage,
                    )?;
                    crate::HostBuffer::F16(dst_storage)
                }
                _ => crate::bail!("Expected f32/f16"),
            };
            return Ok((
                Box::new(fuel_cpu_backend::dyn_impl::CpuStorage(out)),
                dst_shape,
            ));
        }

        #[cfg(feature = "metal")]
        if let Some(metal) = storage
            .as_any()
            .downcast_ref::<fuel_metal::MetalStorage>()
        {
            let self_storage = match &self.storage {
                QStorage::Metal(metal) => metal,
                _ => unreachable!("Cannot call metal matmul on non metal QTensor"),
            };
            let (dst, shape) = self_storage.fwd(&self.shape, metal, layout)?;
            return Ok((Box::new(dst), shape));
        }

        #[cfg(feature = "cuda")]
        if let Some(cuda) = storage
            .as_any()
            .downcast_ref::<fuel_graph_cuda::CudaStorage>()
        {
            let self_storage = match &self.storage {
                QStorage::Cuda(cuda) => cuda,
                _ => unreachable!("Cannot call cuda matmul on non cuda QTensor"),
            };
            let (dst, shape) = self_storage.fwd(&self.shape, cuda, layout)?;
            return Ok((Box::new(dst), shape));
        }

        crate::bail!("qmatmul: unsupported backend")
    }
}

impl crate::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => xs.apply_op1_no_bwd(t.as_ref()),
            Self::Tensor(w) => {
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::TensorF16(w) => {
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
            }
        }
    }
}

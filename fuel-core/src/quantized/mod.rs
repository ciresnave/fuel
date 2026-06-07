//! Quantized (ggml/gguf block-format) tensor support for fuel-core.
//!
//! Per-backend kernels live in their own crates:
//! - CPU: `fuel-cpu-backend::quantized::CpuQStorage`
//! - CUDA: `fuel-cuda-backend::quantized::QCudaStorage`
//! - Metal: `fuel-metal-backend::quantized::QMetalStorage`
//!
//! fuel-core holds the polymorphic [`QTensor`] / [`QMatMul`] front-end and
//! the file-format readers (gguf/ggml/imatrix). All per-backend dispatch
//! goes through [`fuel_core_types::quantized::DynQuantizedStorage`] and
//! [`fuel_core_types::quantized::QuantizedDeviceKernels`] — adding a new
//! backend requires zero edits in this module.

use crate::tensor::Tensor;
use crate::{DType, Device, Result, Shape, D};
pub use fuel_core_types::quantized::{
    DynQuantizedStorage, GgmlDType, QuantizedDeviceKernels,
};
pub use fuel_quantized::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1, BlockQ6K,
    BlockQ8K, BlockQ8_0, BlockQ8_1, GgmlType, QuantizedType,
};

pub mod arch;
pub mod ggml_file;
pub mod gguf_file;
#[cfg(not(target_arch = "wasm32"))]
pub mod gguf_mmap;
pub mod imatrix_file;
#[cfg(not(target_arch = "wasm32"))]
pub mod tokenizer;

use std::borrow::Cow;

/// Storage component of a [`QTensor`] — a backend-agnostic
/// trait object owned by exactly one device.
pub type QStorage = Box<dyn DynQuantizedStorage>;

impl Device {
    /// Allocate a zero-initialised quantized storage of `dtype` for
    /// `elem_count` elements on this device. Returns an error if the
    /// device's backend doesn't implement [`QuantizedDeviceKernels`].
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
        let kernels = self.inner.as_quantized_kernels().ok_or_else(|| {
            crate::Error::Msg(
                "quantized tensors are not supported on this backend".into(),
            )
            .bt()
        })?;
        kernels.qzeros(elem_count, dtype)
    }
}

/// Load pre-quantized block-format bytes onto `device`. Format-aware
/// helper used by ggml/gguf readers; dispatches through the device's
/// [`QuantizedDeviceKernels`] adapter.
pub fn load_quantized(
    data: Cow<'_, [u8]>,
    device: &Device,
    dtype: GgmlDType,
) -> Result<QStorage> {
    let kernels = device.inner.as_quantized_kernels().ok_or_else(|| {
        crate::Error::Msg(
            "quantized tensors are not supported on this backend".into(),
        )
        .bt()
    })?;
    kernels.load_quantized(dtype, data)
}

pub struct QTensor {
    storage: QStorage,
    shape: Shape,
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
        let src_storage = src.storage()?;
        let src_guard = src_storage.read().unwrap();
        storage.quantize(src_guard.as_dyn())?;
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
        let src_storage = src.storage()?;
        let src_guard = src_storage.read().unwrap();
        storage.quantize_imatrix(src_guard.as_dyn(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`.
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
        let mut storage = dev.qzeros(elem_count, dtype)?;
        let src_storage = src.storage()?;
        let src_guard = src_storage.read().unwrap();
        storage.quantize_imatrix_onto(src_guard.as_dyn(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`.
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
        let mut storage = dev.qzeros(elem_count, dtype)?;
        let src_storage = src.storage()?;
        let src_guard = src_storage.read().unwrap();
        storage.quantize_onto(src_guard.as_dyn())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        Device::custom(self.storage.device_arc_dyn())
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
        crate::tensor::from_storage(
            crate::Storage::from_dyn(storage),
            self.shape.clone(),
            none,
            false,
        )
        .to_device(device)
    }

    pub fn dequantize_f16(&self, device: &Device) -> Result<Tensor> {
        // Try the backend's native f16 fast path; fall through to f32 +
        // cast on backends that don't override `dequantize_f16` (default
        // trait impl returns Err).
        match self.storage.dequantize_f16(self.shape.elem_count()) {
            Ok(storage) => {
                let none = crate::op::BackpropOp::none();
                crate::tensor::from_storage(
                    crate::Storage::from_dyn(storage),
                    self.shape.clone(),
                    none,
                    false,
                )
                .to_device(device)
            }
            Err(_) => {
                let s = self.dequantize(device)?.to_dtype(crate::DType::F16)?;
                Ok(s)
            }
        }
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.storage_size_in_bytes()
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        self.storage.data()
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let x_arc = x.storage()?;
        let ids_arc = ids.storage()?;
        let x_guard = x_arc.read().unwrap();
        let ids_guard = ids_arc.read().unwrap();
        let (storage, out_shape) = self.storage.indexed_moe_forward(
            self.shape(),
            x_guard.as_dyn(),
            x.layout(),
            ids_guard.as_dyn(),
            ids.layout(),
        )?;
        Ok(crate::tensor::from_storage(
            crate::Storage::from_dyn(storage),
            out_shape,
            crate::op::BackpropOp::none(),
            false,
        ))
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        self.storage.device_ptr()
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
            Ok(s) => !s.is_empty() && s != "0",
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_F16: bool = {
        match std::env::var("FUEL_DEQUANTIZE_ALL_F16") {
            Ok(s) => !s.is_empty() && s != "0",
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
            _ => panic!("Not implemented!"),
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
        self.storage.fwd(&self.shape, storage, layout)
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

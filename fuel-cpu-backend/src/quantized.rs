//! CPU-side quantized storage adapter — bridges `Box<dyn QuantizedType>`
//! (per-block-format scalar ops from fuel-quantized) to the
//! backend-agnostic `DynQuantizedStorage` trait that fuel-core dispatches
//! against.
//!
//! Owns the `QuantizedDeviceKernels` impl on `CpuBackendDevice` so
//! `Device::qzeros` / `Device::load_quantized` reach CPU through the
//! standard `DynBackendDevice::as_quantized_kernels` accessor.

use crate::CpuStorage;
use crate::dyn_impl::CpuBackendDevice;
use fuel_backend_contract::dyn_backend::DynBackendStorage;
use fuel_backend_contract::quantized::{DynQuantizedStorage, QuantizedDeviceKernels};
use fuel_ir::quantized::GgmlDType;
use fuel_ir::{DType, Error, HostBuffer, Layout, Result, Shape};
use fuel_quantized::{QuantizedType, cpu_from_data, cpu_zeros};
use half::f16;
use std::any::Any;
use std::borrow::Cow;

pub struct CpuQStorage(pub Box<dyn QuantizedType>);

impl std::fmt::Debug for CpuQStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "CpuQStorage[{:?}]", self.0.dtype())
    }
}

fn cpu_src_as_slice<'a>(src: &'a dyn DynBackendStorage) -> Result<&'a [f32]> {
    let cpu = src
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg("expected cpu storage for quantize".into()).bt())?;
    cpu.0.as_slice::<f32>()
}

impl DynQuantizedStorage for CpuQStorage {
    fn dtype(&self) -> GgmlDType {
        self.0.dtype()
    }
    fn block_size(&self) -> usize {
        self.0.block_size()
    }
    fn storage_size_in_bytes(&self) -> usize {
        self.0.storage_size_in_bytes()
    }
    fn quantize(&mut self, src: &dyn DynBackendStorage) -> Result<()> {
        self.0.from_float(cpu_src_as_slice(src)?);
        Ok(())
    }
    fn quantize_imatrix(
        &mut self,
        src: &dyn DynBackendStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        self.0
            .from_float_imatrix(cpu_src_as_slice(src)?, imatrix_weights, n_per_row);
        Ok(())
    }
    fn quantize_onto(&mut self, src: &dyn DynBackendStorage) -> Result<()> {
        self.0.from_float(cpu_src_as_slice(src)?);
        Ok(())
    }
    fn quantize_imatrix_onto(
        &mut self,
        src: &dyn DynBackendStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        self.0
            .from_float_imatrix(cpu_src_as_slice(src)?, imatrix_weights, n_per_row);
        Ok(())
    }
    fn dequantize(&self, elem_count: usize) -> Result<Box<dyn DynBackendStorage>> {
        let buf = self.0.dequantize(elem_count)?;
        Ok(Box::new(CpuStorage(buf)))
    }
    fn data(&self) -> Result<Cow<'_, [u8]>> {
        let data_ptr = self.0.as_ptr();
        let size_in_bytes = self.0.storage_size_in_bytes();
        let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
        Ok(Cow::from(data))
    }
    fn fwd(
        &self,
        self_shape: &Shape,
        input: &dyn DynBackendStorage,
        layout: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)> {
        let cpu = input
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::Msg("qmatmul: expected cpu storage".into()).bt())?;
        let storage = &cpu.0;
        if !layout.is_contiguous() {
            return Err(Error::Msg(format!("input tensor is not contiguous {layout:?}")).bt());
        }
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        if src_shape.rank() < 2 {
            return Err(Error::Msg(format!("input tensor has only one dimension {layout:?}")).bt());
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            return Err(Error::Msg(format!(
                "input tensor {layout:?} incompatible with {self_shape:?}"
            ))
            .bt());
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let out = match storage.dtype() {
            DType::F32 => {
                let slice = storage.as_slice::<f32>()?;
                let slice = &slice
                    [layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![0f32; dst_shape.elem_count()];
                self.0.matmul_t(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                HostBuffer::F32(dst_storage)
            }
            DType::F16 => {
                let slice = storage.as_slice::<f16>()?;
                let slice = &slice
                    [layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![f16::ZERO; dst_shape.elem_count()];
                self.0.matmul_t_f16(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                HostBuffer::F16(dst_storage)
            }
            _ => return Err(Error::Msg("Expected f32/f16".into()).bt()),
        };
        Ok((Box::new(CpuStorage(out)), dst_shape))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn device_arc_dyn(&self) -> std::sync::Arc<dyn fuel_backend_contract::dyn_backend::DynBackendDevice> {
        std::sync::Arc::new(CpuBackendDevice)
    }
}

impl QuantizedDeviceKernels for CpuBackendDevice {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<Box<dyn DynQuantizedStorage>> {
        Ok(Box::new(CpuQStorage(cpu_zeros(dtype, elem_count))))
    }
    fn load_quantized(
        &self,
        dtype: GgmlDType,
        data: Cow<'_, [u8]>,
    ) -> Result<Box<dyn DynQuantizedStorage>> {
        Ok(Box::new(CpuQStorage(cpu_from_data(dtype, data))))
    }
}

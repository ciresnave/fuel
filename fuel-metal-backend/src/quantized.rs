use crate::{DType, MetalDevice, MetalStorage, Result, Shape, D};
use fuel_backend_contract::dyn_backend::DynBackendStorage;
use fuel_backend_contract::quantized::{DynQuantizedStorage, QuantizedDeviceKernels};
use fuel_ir::quantized::GgmlDType;
use fuel_quantized::GgmlType;
use fuel_metal_kernels::metal::Buffer;
use std::any::Any;
use std::borrow::Cow;
use std::sync::Arc;

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
    buffer: Arc<Buffer>,
}

impl QMetalStorage {
    pub fn zeros(device: &MetalDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        let buffer = device.allocate_zeros(size)?;
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        use fuel_quantized::GgmlType;

        let buffer = self.device.allocate_buffer(self.buffer.length())?;
        let blit = self.device.blit_command_encoder()?;
        blit.set_label("blit_to_cpu");
        blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        blit.end_encoding();
        self.device.wait_until_completed()?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => {
                let vec: Vec<f32> = read_to_vec(&buffer, block_len);
                f32::to_float(&vec, &mut out);
            }
            GgmlDType::F16 => {
                let vec: Vec<half::f16> = read_to_vec(&buffer, block_len);
                half::f16::to_float(&vec, &mut out);
            }
            GgmlDType::BF16 => {
                let vec: Vec<half::bf16> = read_to_vec(&buffer, block_len);
                half::bf16::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_0 => {
                let vec: Vec<fuel_quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                fuel_quantized::BlockQ4_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<fuel_quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                fuel_quantized::BlockQ4_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<fuel_quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                fuel_quantized::BlockQ5_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<fuel_quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                fuel_quantized::BlockQ5_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_0 => {
                let vec: Vec<fuel_quantized::BlockQ8_0> = read_to_vec(&buffer, block_len);
                fuel_quantized::BlockQ8_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<fuel_quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                fuel_quantized::BlockQ8_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q2K => {
                let vec: Vec<fuel_quantized::BlockQ2K> = read_to_vec(&buffer, block_len);
                fuel_quantized::BlockQ2K::to_float(&vec, &mut out);
            }
            GgmlDType::Q3K => {
                let vec: Vec<fuel_quantized::BlockQ3K> = read_to_vec(&buffer, block_len);
                fuel_quantized::BlockQ3K::to_float(&vec, &mut out);
            }
            GgmlDType::Q4K => {
                let vec: Vec<fuel_quantized::BlockQ4K> = read_to_vec(&buffer, block_len);
                fuel_quantized::BlockQ4K::to_float(&vec, &mut out);
            }
            GgmlDType::Q5K => {
                let vec: Vec<fuel_quantized::BlockQ5K> = read_to_vec(&buffer, block_len);
                fuel_quantized::BlockQ5K::to_float(&vec, &mut out);
            }
            GgmlDType::Q6K => {
                let vec: Vec<fuel_quantized::BlockQ6K> = read_to_vec(&buffer, block_len);
                fuel_quantized::BlockQ6K::to_float(&vec, &mut out);
            }
            GgmlDType::Q8K => {
                let vec: Vec<fuel_quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                fuel_quantized::BlockQ8K::to_float(&vec, &mut out);
            }
        }

        let buffer = self.device.new_buffer_with_data(&out)?;
        Ok(MetalStorage::new(
            buffer,
            self.device.clone(),
            elem_count,
            DType::F32,
        ))
    }

    fn quantize_from_f32(
        &mut self,
        src: &[f32],
        imatrix: Option<(&[f32], usize)>,
    ) -> Result<()> {
        let mut qcpu = fuel_quantized::cpu_zeros(self.dtype, src.len());
        match imatrix {
            None => qcpu.from_float(src),
            Some((iw, n_per_row)) => qcpu.from_float_imatrix(src, iw, n_per_row),
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(qcpu.as_ptr(), qcpu.storage_size_in_bytes())
        };
        let buffer = self.device.new_buffer_with_data(bytes)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize(&mut self, src: &MetalStorage) -> Result<()> {
        let src = src.to_cpu::<f32>()?;
        self.quantize_from_f32(&src, None)
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &MetalStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let src = src.to_cpu::<f32>()?;
        self.quantize_from_f32(&src, Some((imatrix_weights, n_per_row)))
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &fuel_ir::HostBuffer,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        self.quantize_from_f32(src.as_slice::<f32>()?, Some((imatrix_weights, n_per_row)))
    }

    pub fn quantize_onto(&mut self, src: &fuel_ir::HostBuffer) -> Result<()> {
        self.quantize_from_f32(src.as_slice::<f32>()?, None)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.buffer.length()
    }

    fn fwd_mv(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            fuel_ir::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            fuel_ir::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();

        // We always use a single batch dimension and stack all the tensors in the batch on the
        // second dimension as the implementation in fuel-metal-kernels doesn't handle batch
        // properly.
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            n => fuel_ir::bail!("Invalid rank {n} for quantized matmul metal"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            fuel_ir::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul")?;
        let encoder = device.command_encoder()?;
        // In some cases it would be better to use the mm variant, though it has its drawbacks
        // around memory alignment.
        for batch_id in 0..m {
            fuel_metal_kernels::call_quantized_matmul_mv_t(
                device.device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                (1, 1, n, k),
                storage.buffer(),
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes(),
                &self.buffer,
                batch_id * n * DType::F32.size_in_bytes(),
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage = crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            fuel_ir::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            fuel_ir::bail!("input tensor has only one dimension {layout:?}")
        }
        let n = self_shape.dim(D::Minus2)?;
        let k = self_shape.dim(D::Minus1)?;
        let mut dst_shape = src_shape.dims().to_vec();

        if src_shape.rank() < self_shape.rank() {
            fuel_ir::bail!(
                "input rank ({}) must be >= weight rank ({})",
                src_shape.rank(),
                self_shape.rank()
            )
        }

        if src_shape.dim(D::Minus2)? == 1 {
            return self.fwd_mv(self_shape, storage, layout);
        }

        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            fuel_ir::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul")?;
        let encoder = device.command_encoder()?;

        assert_eq!(storage.dtype(), DType::F32);

        if self_shape.rank() > 4 {
            fuel_ir::bail!("weight rank ({}) must be <= 4", self_shape.rank())
        }
        let src0_l = crate::Layout::contiguous(
            [vec![1; 4 - self_shape.rank()], self_shape.dims().to_vec()].concat(),
        );
        let src0_stride = src0_l
            .stride()
            .iter()
            .map(|x| {
                (*x as f32 * (self.dtype.type_size() as f32 / self.dtype.block_size() as f32))
                    as usize
            })
            .collect::<Vec<_>>();

        if src_shape.rank() > 4 {
            fuel_ir::bail!("weight rank ({}) must be <= 4", src_shape.rank())
        }
        let src1_l = crate::Layout::contiguous(
            [vec![1; 4 - src_shape.rank()], src_shape.dims().to_vec()].concat(),
        );

        fuel_metal_kernels::call_quantized_matmul_mm_t(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            src0_l.dims(),
            &src0_stride,
            &self.buffer,
            src1_l.dims(),
            &src1_l
                .stride()
                .iter()
                .map(|x| x * DType::F32.size_in_bytes())
                .collect::<Vec<_>>(),
            storage.buffer(),
            src1_l.start_offset() * storage.dtype().size_in_bytes(),
            dst_shape.dims(),
            0,
            &dst,
        )
        .map_err(MetalError::from)?;

        let dst_storage = crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let buffer = self.device.allocate_buffer(self.buffer.length())?;
        {
            let blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
            blit.end_encoding();
        }
        self.device.wait_until_completed()?;
        Ok(read_to_vec::<u8>(&buffer, self.storage_size_in_bytes()))
    }
}

/// Build a `QMetalStorage` from raw block-format bytes already laid out for
/// `dtype`. Returned as a typed `Box<dyn DynQuantizedStorage>`.
pub fn load_quantized_bytes(
    device: &MetalDevice,
    dtype: GgmlDType,
    data: &[u8],
) -> Result<Box<dyn DynQuantizedStorage>> {
    let buffer = device.new_buffer_with_data(data)?;
    Ok(Box::new(QMetalStorage {
        dtype,
        device: device.clone(),
        buffer,
    }))
}

// ---------------------------------------------------------------------------
// DynQuantizedStorage / QuantizedDeviceKernels
// ---------------------------------------------------------------------------

impl std::fmt::Debug for QMetalStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QMetalStorage[{:?}]", self.dtype)
    }
}

impl DynQuantizedStorage for QMetalStorage {
    fn dtype(&self) -> GgmlDType {
        self.dtype
    }
    fn block_size(&self) -> usize {
        self.dtype.block_size()
    }
    fn storage_size_in_bytes(&self) -> usize {
        QMetalStorage::storage_size_in_bytes(self)
    }
    fn quantize(&mut self, src: &dyn DynBackendStorage) -> Result<()> {
        let metal = src.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
            fuel_ir::Error::Msg("quantize: expected metal storage".into()).bt()
        })?;
        QMetalStorage::quantize(self, metal)
    }
    fn quantize_imatrix(
        &mut self,
        src: &dyn DynBackendStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let metal = src.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
            fuel_ir::Error::Msg("quantize_imatrix: expected metal storage".into()).bt()
        })?;
        QMetalStorage::quantize_imatrix(self, metal, imatrix_weights, n_per_row)
    }
    fn quantize_onto(&mut self, src: &dyn DynBackendStorage) -> Result<()> {
        let cpu = src
            .as_any()
            .downcast_ref::<fuel_cpu_backend::CpuStorage>()
            .ok_or_else(|| {
                fuel_ir::Error::Msg("quantize_onto: expected cpu storage".into()).bt()
            })?;
        QMetalStorage::quantize_onto(self, &cpu.0)
    }
    fn quantize_imatrix_onto(
        &mut self,
        src: &dyn DynBackendStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let cpu = src
            .as_any()
            .downcast_ref::<fuel_cpu_backend::CpuStorage>()
            .ok_or_else(|| {
                fuel_ir::Error::Msg("quantize_imatrix_onto: expected cpu storage".into())
                    .bt()
            })?;
        QMetalStorage::quantize_imatrix_onto(self, &cpu.0, imatrix_weights, n_per_row)
    }
    fn dequantize(&self, elem_count: usize) -> Result<Box<dyn DynBackendStorage>> {
        Ok(Box::new(QMetalStorage::dequantize(self, elem_count)?))
    }
    fn data(&self) -> Result<Cow<'_, [u8]>> {
        Ok(Cow::Owned(QMetalStorage::data(self)?))
    }
    fn fwd(
        &self,
        self_shape: &Shape,
        input: &dyn DynBackendStorage,
        layout: &crate::Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)> {
        let metal = input.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
            fuel_ir::Error::Msg("qmatmul: expected metal storage".into()).bt()
        })?;
        let (s, sh) = QMetalStorage::fwd(self, self_shape, metal, layout)?;
        Ok((Box::new(s), sh))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn device_arc_dyn(&self) -> std::sync::Arc<dyn fuel_backend_contract::dyn_backend::DynBackendDevice> {
        std::sync::Arc::new(self.device.clone())
    }
}

impl QuantizedDeviceKernels for MetalDevice {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<Box<dyn DynQuantizedStorage>> {
        Ok(Box::new(QMetalStorage::zeros(self, elem_count, dtype)?))
    }
    fn load_quantized(
        &self,
        dtype: GgmlDType,
        data: Cow<'_, [u8]>,
    ) -> Result<Box<dyn DynQuantizedStorage>> {
        load_quantized_bytes(self, dtype, &data)
    }
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

impl From<GgmlDType> for fuel_metal_kernels::GgmlDType {
    fn from(value: GgmlDType) -> Self {
        match value {
            GgmlDType::Q4_0 => fuel_metal_kernels::GgmlDType::Q4_0,
            GgmlDType::Q4_1 => fuel_metal_kernels::GgmlDType::Q4_1,
            GgmlDType::Q5_0 => fuel_metal_kernels::GgmlDType::Q5_0,
            GgmlDType::Q5_1 => fuel_metal_kernels::GgmlDType::Q5_1,
            GgmlDType::Q8_0 => fuel_metal_kernels::GgmlDType::Q8_0,
            GgmlDType::Q8_1 => fuel_metal_kernels::GgmlDType::Q8_1,
            GgmlDType::Q2K => fuel_metal_kernels::GgmlDType::Q2K,
            GgmlDType::Q3K => fuel_metal_kernels::GgmlDType::Q3K,
            GgmlDType::Q4K => fuel_metal_kernels::GgmlDType::Q4K,
            GgmlDType::Q5K => fuel_metal_kernels::GgmlDType::Q5K,
            GgmlDType::Q6K => fuel_metal_kernels::GgmlDType::Q6K,
            GgmlDType::Q8K => fuel_metal_kernels::GgmlDType::Q8K,
            GgmlDType::F16 => fuel_metal_kernels::GgmlDType::F16,
            GgmlDType::F32 => fuel_metal_kernels::GgmlDType::F32,
            GgmlDType::BF16 => fuel_metal_kernels::GgmlDType::F16,
        }
    }
}

//! Vulkan GPU executor for fuel-graph computation graphs.
//!
//! Uses the Vulkane crate for Vulkan device management, buffer
//! allocation, and compute shader dispatch. WGSL shaders are compiled
//! to SPIR-V at runtime via naga.
//!
//! This is the third backend for fuel's generic `GraphExecutor<B>`,
//! validating the backend-agnostic architecture alongside CPU and CUDA.

use fuel_core_types::{DType, Layout, Shape};
use fuel_graph_executor::{BinaryOp, GraphBackend, UnaryOp};
use vulkane::safe::*;

/// Vulkan storage: a device-local buffer + its memory allocation,
/// plus the element count and dtype (Vulkan buffers are untyped bytes).
pub struct VulkanStorage {
    pub buffer: Buffer,
    pub memory: DeviceMemory,
    pub elem_count: usize,
    pub dtype: DType,
}

impl VulkanStorage {
    fn byte_size(&self) -> u64 {
        (self.elem_count * dtype_size(self.dtype)) as u64
    }
}

/// Vulkan backend: device + queue + command pool for compute dispatch.
pub struct VulkanBackend {
    pub device: Device,
    pub physical: PhysicalDevice,
    pub queue: Queue,
    pub queue_family: u32,
}

impl VulkanBackend {
    /// Initialize a Vulkan compute backend on the first available GPU.
    pub fn new() -> fuel_core_types::Result<Self> {
        let instance = Instance::new(InstanceCreateInfo {
            application_name: Some("fuel"),
            application_version: ApiVersion::V1_0,
            engine_name: Some("fuel-graph-vulkan"),
            engine_version: ApiVersion::V1_0,
            api_version: ApiVersion::V1_2,
            ..Default::default()
        }).map_err(|e| fuel_core_types::Error::Msg(format!("Vulkan instance: {e:?}")))?;

        let physicals = instance.enumerate_physical_devices()
            .map_err(|e| fuel_core_types::Error::Msg(format!("enumerate devices: {e:?}")))?;
        let physical = physicals.into_iter().next()
            .ok_or_else(|| fuel_core_types::Error::Msg("no Vulkan device found".into()))?;

        let queue_family = physical
            .find_queue_family(QueueFlags::COMPUTE)
            .ok_or_else(|| fuel_core_types::Error::Msg("no compute queue family".into()))?;

        let device = physical.create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo::single(queue_family)],
            ..Default::default()
        }).map_err(|e| fuel_core_types::Error::Msg(format!("create device: {e:?}")))?;

        let queue = device.get_queue(queue_family, 0);

        Ok(Self { device, physical, queue, queue_family })
    }

    /// Upload a typed slice to a device-local storage buffer.
    fn upload_slice<T: Copy + 'static>(
        &self, data: &[T], dtype: DType,
    ) -> fuel_core_types::Result<VulkanStorage> {
        let usage = BufferUsage::STORAGE_BUFFER
            | BufferUsage::TRANSFER_SRC
            | BufferUsage::TRANSFER_DST;
        let (buffer, memory) = self.queue.upload_buffer(
            &self.device,
            &self.physical,
            self.queue_family,
            data,
            usage,
        ).map_err(|e| fuel_core_types::Error::Msg(format!("upload: {e:?}")))?;
        Ok(VulkanStorage {
            buffer,
            memory,
            elem_count: data.len(),
            dtype,
        })
    }

    /// Download a device-local buffer to a host Vec<T>.
    fn download_slice<T: Copy + Default + 'static>(
        &self, storage: &VulkanStorage,
    ) -> fuel_core_types::Result<Vec<T>> {
        let byte_size = storage.byte_size();
        // Create a host-visible staging buffer.
        let (staging_buf, mut staging_mem) = Buffer::new_bound(
            &self.device,
            &self.physical,
            BufferCreateInfo {
                size: byte_size,
                usage: BufferUsage::TRANSFER_DST,
            },
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        ).map_err(|e| fuel_core_types::Error::Msg(format!("staging alloc: {e:?}")))?;

        // Copy device → staging.
        self.queue.one_shot(&self.device, self.queue_family, |cmd| {
            cmd.copy_buffer(&storage.buffer, &staging_buf, &[BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: byte_size,
            }]);
            Ok(())
        }).map_err(|e| fuel_core_types::Error::Msg(format!("copy D2H: {e:?}")))?;

        // Map and read.
        let mapped = staging_mem.map()
            .map_err(|e| fuel_core_types::Error::Msg(format!("map: {e:?}")))?;
        let bytes = mapped.as_slice();
        let n = storage.elem_count;
        let mut out = vec![T::default(); n];
        let out_bytes = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * std::mem::size_of::<T>())
        };
        out_bytes.copy_from_slice(&bytes[..out_bytes.len()]);
        Ok(out)
    }
}

impl GraphBackend for VulkanBackend {
    type Storage = VulkanStorage;

    fn alloc_zeros(&self, shape: &Shape, dtype: DType) -> fuel_core_types::Result<Self::Storage> {
        let n = shape.elem_count();
        let byte_size = (n * dtype_size(dtype)) as u64;
        let (buffer, memory) = Buffer::new_bound(
            &self.device,
            &self.physical,
            BufferCreateInfo {
                size: byte_size,
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
            },
            MemoryPropertyFlags::DEVICE_LOCAL,
        ).map_err(|e| fuel_core_types::Error::Msg(format!("alloc: {e:?}")))?;

        // Zero-fill via vkCmdFillBuffer.
        self.queue.one_shot(&self.device, self.queue_family, |cmd| {
            cmd.fill_buffer(&buffer, 0, byte_size, 0);
            Ok(())
        }).map_err(|e| fuel_core_types::Error::Msg(format!("zero fill: {e:?}")))?;

        Ok(VulkanStorage { buffer, memory, elem_count: n, dtype })
    }

    fn upload(
        &self,
        buf: &fuel_core_types::HostBuffer,
        _shape: &Shape,
    ) -> fuel_core_types::Result<Self::Storage> {
        use fuel_core_types::HostBuffer;
        match buf {
            HostBuffer::F32(v) => self.upload_slice(v, DType::F32),
            HostBuffer::F64(v) => self.upload_slice(v, DType::F64),
            HostBuffer::U32(v) => self.upload_slice(v, DType::U32),
            _ => fuel_core_types::bail!("VulkanBackend::upload: unsupported dtype"),
        }
    }

    fn download(
        &self, storage: &Self::Storage,
    ) -> fuel_core_types::Result<fuel_core_types::HostBuffer> {
        use fuel_core_types::HostBuffer;
        match storage.dtype {
            DType::F32 => Ok(HostBuffer::F32(self.download_slice::<f32>(storage)?)),
            DType::F64 => Ok(HostBuffer::F64(self.download_slice::<f64>(storage)?)),
            DType::U32 => Ok(HostBuffer::U32(self.download_slice::<u32>(storage)?)),
            other => fuel_core_types::bail!("VulkanBackend::download: unsupported {other:?}"),
        }
    }

    fn try_clone(
        &self, storage: &Self::Storage, layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        let n = layout.shape().elem_count();
        let byte_size = (n * dtype_size(storage.dtype)) as u64;
        let (dst_buf, dst_mem) = Buffer::new_bound(
            &self.device,
            &self.physical,
            BufferCreateInfo {
                size: byte_size,
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
            },
            MemoryPropertyFlags::DEVICE_LOCAL,
        ).map_err(|e| fuel_core_types::Error::Msg(format!("clone alloc: {e:?}")))?;

        self.queue.one_shot(&self.device, self.queue_family, |cmd| {
            cmd.copy_buffer(&storage.buffer, &dst_buf, &[BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: byte_size,
            }]);
            Ok(())
        }).map_err(|e| fuel_core_types::Error::Msg(format!("clone copy: {e:?}")))?;

        Ok(VulkanStorage {
            buffer: dst_buf,
            memory: dst_mem,
            elem_count: n,
            dtype: storage.dtype,
        })
    }

    fn copy_strided_src(
        &self,
        _src: &Self::Storage,
        _dst: &mut Self::Storage,
        _dst_offset: usize,
        _src_layout: &Layout,
    ) -> fuel_core_types::Result<()> {
        // TODO: implement via compute shader for strided patterns.
        // For now, this will hit the CPU fallback path for ops that
        // need strided copies (permute, broadcast, concat, slice).
        fuel_core_types::bail!("VulkanBackend: copy_strided_src not yet implemented — use CPU fallback")
    }

    fn storage_dtype(&self, storage: &Self::Storage) -> DType {
        storage.dtype
    }

    // -- compute ops: CPU fallback for now --
    // The generic executor's cpu_fallback() handles these by
    // downloading inputs, running the reference backend, re-uploading.
    // As we add WGSL compute shaders, each op moves from fallback to
    // native Vulkan dispatch.

    fn matmul(
        &self, _a: &Self::Storage, _b: &Self::Storage,
        _bmnk: (usize, usize, usize, usize),
        _la: &Layout, _lb: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: matmul not yet implemented")
    }

    fn unary(
        &self, _op: UnaryOp,
        _a: &Self::Storage, _layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: unary not yet implemented")
    }

    fn binary(
        &self, _op: BinaryOp,
        _a: &Self::Storage, _b: &Self::Storage,
        _la: &Layout, _lb: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: binary not yet implemented")
    }

    fn affine(
        &self, _a: &Self::Storage, _layout: &Layout,
        _mul: f64, _add: f64,
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: affine not yet implemented")
    }

    fn powf(
        &self, _a: &Self::Storage, _layout: &Layout,
        _exp: f64,
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: powf not yet implemented")
    }

    fn cast(
        &self, _a: &Self::Storage, _layout: &Layout,
        _dtype: DType,
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: cast not yet implemented")
    }

    fn reduce(
        &self, _op: fuel_core_types::op::ReduceOp,
        _a: &Self::Storage, _layout: &Layout,
        _dims: &[usize],
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: reduce not yet implemented")
    }

    fn softmax_last_dim(
        &self, _a: &Self::Storage, _layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: softmax not yet implemented")
    }

    fn index_select(
        &self, _src: &Self::Storage, _ids: &Self::Storage,
        _src_l: &Layout, _ids_l: &Layout, _dim: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: index_select not yet implemented")
    }

    fn gather(
        &self, _src: &Self::Storage, _ids: &Self::Storage,
        _src_l: &Layout, _ids_l: &Layout, _dim: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: gather not yet implemented")
    }
}

fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::F32 | DType::U32 | DType::I32 => 4,
        DType::F64 | DType::I64 => 8,
        DType::F16 | DType::BF16 | DType::I16 => 2,
        DType::U8 => 1,
        _ => 4, // fallback
    }
}

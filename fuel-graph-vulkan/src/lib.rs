//! Vulkan GPU executor for fuel-graph computation graphs.
//!
//! Uses Vulkane for Vulkan device management and dispatches compute
//! ops through WGSL shaders compiled to SPIR-V via naga. Third
//! backend for fuel's generic `GraphExecutor<B>`.

pub mod pipelines;

use fuel_core_types::{DType, Layout, Shape};
use fuel_graph_executor::{BinaryOp, GraphBackend, UnaryOp};
use pipelines::Pipelines;
use vulkane::safe::*;

/// Vulkan storage: device-local buffer + metadata.
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

/// Vulkan compute backend with pre-compiled shader pipelines.
pub struct VulkanBackend {
    pub device: Device,
    pub physical: PhysicalDevice,
    pub queue: Queue,
    pub queue_family: u32,
    pub pipelines: Pipelines,
    pub device_name: String,
}

/// How to select a Vulkan physical device.
pub enum DeviceSelection {
    /// Pick by index in the enumeration order (0 = first).
    Index(usize),
    /// Prefer discrete GPU over integrated. Falls back to first
    /// available if no discrete GPU exists.
    PreferDiscrete,
    /// Match by substring in the device name (case-insensitive).
    ByName(String),
}

impl VulkanBackend {
    /// Initialize with the default device selection: prefer discrete GPU.
    pub fn new() -> fuel_core_types::Result<Self> {
        Self::with_selection(DeviceSelection::PreferDiscrete)
    }

    /// Initialize with explicit device selection.
    pub fn with_selection(selection: DeviceSelection) -> fuel_core_types::Result<Self> {
        let instance = Instance::new(InstanceCreateInfo {
            application_name: Some("fuel"),
            application_version: ApiVersion::V1_0,
            engine_name: Some("fuel-graph-vulkan"),
            engine_version: ApiVersion::V1_0,
            api_version: ApiVersion::V1_2,
            ..Default::default()
        }).map_err(vk_err)?;

        let physicals = instance.enumerate_physical_devices().map_err(vk_err)?;
        if physicals.is_empty() {
            return Err(fuel_core_types::Error::Msg("no Vulkan devices found".into()));
        }

        let physical = match selection {
            DeviceSelection::Index(idx) => {
                physicals.into_iter().nth(idx)
                    .ok_or_else(|| fuel_core_types::Error::Msg(
                        format!("Vulkan device index {idx} out of range"),
                    ))?
            }
            DeviceSelection::PreferDiscrete => {
                // Try discrete first, then any GPU, then anything.
                let mut best = None;
                for p in &physicals {
                    let props = p.properties();
                    let dt = props.device_type();
                    if dt == PhysicalDeviceType::DISCRETE_GPU {
                        best = Some(p);
                        break;
                    }
                    if best.is_none()
                        && dt != PhysicalDeviceType::CPU
                        && dt != PhysicalDeviceType::OTHER
                    {
                        best = Some(p);
                    }
                }
                match best {
                    Some(p) => p.clone(),
                    None => physicals.into_iter().next().unwrap(),
                }
            }
            DeviceSelection::ByName(ref needle) => {
                let needle_lower = needle.to_lowercase();
                physicals.into_iter()
                    .find(|p| {
                        p.properties().device_name().to_lowercase().contains(&needle_lower)
                    })
                    .ok_or_else(|| fuel_core_types::Error::Msg(
                        format!("no Vulkan device matching {needle:?}"),
                    ))?
            }
        };

        let props = physical.properties();
        let device_name = props.device_name();
        let device_type = props.device_type();
        tracing::info!(
            name = %device_name,
            r#type = ?device_type,
            "Selected Vulkan device",
        );

        let queue_family = physical
            .find_queue_family(QueueFlags::COMPUTE)
            .ok_or_else(|| fuel_core_types::Error::Msg("no compute queue".into()))?;

        let device = physical.create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo::single(queue_family)],
            ..Default::default()
        }).map_err(vk_err)?;

        let queue = device.get_queue(queue_family, 0);

        let pipelines = Pipelines::new(&device).map_err(vk_err)?;

        Ok(Self { device, physical, queue, queue_family, pipelines, device_name })
    }

    /// List all available Vulkan physical devices.
    pub fn list_devices() -> fuel_core_types::Result<Vec<(usize, String, String)>> {
        let instance = Instance::new(InstanceCreateInfo {
            application_name: Some("fuel"),
            application_version: ApiVersion::V1_0,
            engine_name: Some("fuel-graph-vulkan"),
            engine_version: ApiVersion::V1_0,
            api_version: ApiVersion::V1_2,
            ..Default::default()
        }).map_err(vk_err)?;
        let physicals = instance.enumerate_physical_devices().map_err(vk_err)?;
        Ok(physicals.iter().enumerate().map(|(i, p)| {
            let props = p.properties();
            let dt = props.device_type();
            let type_str = if dt == PhysicalDeviceType::DISCRETE_GPU { "discrete" }
                else if dt == PhysicalDeviceType::INTEGRATED_GPU { "integrated" }
                else if dt == PhysicalDeviceType::VIRTUAL_GPU { "virtual" }
                else if dt == PhysicalDeviceType::CPU { "cpu" }
                else { "other" };
            (i, props.device_name(), type_str.to_string())
        }).collect())
    }

    // -- helpers --

    fn upload_slice<T: Copy + 'static>(
        &self, data: &[T], dtype: DType,
    ) -> fuel_core_types::Result<VulkanStorage> {
        let usage = BufferUsage::STORAGE_BUFFER
            | BufferUsage::TRANSFER_SRC
            | BufferUsage::TRANSFER_DST;
        let (buffer, memory) = self.queue.upload_buffer(
            &self.device, &self.physical, self.queue_family, data, usage,
        ).map_err(vk_err)?;
        Ok(VulkanStorage { buffer, memory, elem_count: data.len(), dtype })
    }

    fn download_slice<T: Copy + Default + 'static>(
        &self, storage: &VulkanStorage,
    ) -> fuel_core_types::Result<Vec<T>> {
        let byte_size = storage.byte_size();
        let (staging_buf, mut staging_mem) = Buffer::new_bound(
            &self.device, &self.physical,
            BufferCreateInfo { size: byte_size, usage: BufferUsage::TRANSFER_DST },
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        ).map_err(vk_err)?;
        self.queue.one_shot(&self.device, self.queue_family, |cmd| {
            cmd.copy_buffer(&storage.buffer, &staging_buf, &[BufferCopy {
                src_offset: 0, dst_offset: 0, size: byte_size,
            }]);
            Ok(())
        }).map_err(vk_err)?;
        let mapped = staging_mem.map().map_err(vk_err)?;
        let bytes = mapped.as_slice();
        let n = storage.elem_count;
        let mut out = vec![T::default(); n];
        let dst = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * std::mem::size_of::<T>())
        };
        dst.copy_from_slice(&bytes[..dst.len()]);
        Ok(out)
    }

    fn alloc_device(&self, byte_size: u64, n: usize, dtype: DType) -> fuel_core_types::Result<VulkanStorage> {
        let (buffer, memory) = Buffer::new_bound(
            &self.device, &self.physical,
            BufferCreateInfo {
                size: byte_size,
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
            },
            MemoryPropertyFlags::DEVICE_LOCAL,
        ).map_err(vk_err)?;
        Ok(VulkanStorage { buffer, memory, elem_count: n, dtype })
    }

    /// Upload a typed slice as a device-local storage buffer (no
    /// VulkanStorage wrapper — used for internal dispatch metadata
    /// like shape/strides arrays).
    fn upload_slice_raw<T: Copy + 'static>(&self, data: &[T]) -> fuel_core_types::Result<(Buffer, DeviceMemory)> {
        self.queue.upload_buffer(
            &self.device, &self.physical, self.queue_family, data,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
        ).map_err(vk_err)
    }

    /// Upload a small params struct as a uniform buffer.
    fn upload_params<T: Copy + 'static>(&self, params: &T) -> fuel_core_types::Result<(Buffer, DeviceMemory)> {
        let bytes = unsafe { as_bytes(params) };
        // Uniform buffers must be at least 16 bytes on some implementations.
        let size = (bytes.len().max(16)) as u64;
        let (buf, mut mem) = Buffer::new_bound(
            &self.device, &self.physical,
            BufferCreateInfo { size, usage: BufferUsage::UNIFORM_BUFFER },
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        ).map_err(vk_err)?;
        let mut mapped = mem.map().map_err(vk_err)?;
        mapped.as_slice_mut()[..bytes.len()].copy_from_slice(bytes);
        drop(mapped);
        Ok((buf, mem))
    }

    /// Dispatch a 2-storage + 1-uniform compute shader.
    fn dispatch_2buf(
        &self,
        pipeline: &ComputePipeline,
        pipe_layout: &PipelineLayout,
        input: &VulkanStorage,
        output: &VulkanStorage,
        params_buf: &Buffer,
        params_size: u64,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) -> fuel_core_types::Result<()> {
        let desc = self.pipelines.desc_pool.allocate(&self.pipelines.layout_2s1u).map_err(vk_err)?;
        desc.write_buffer(0, DescriptorType::STORAGE_BUFFER, &input.buffer, 0, input.byte_size());
        desc.write_buffer(1, DescriptorType::STORAGE_BUFFER, &output.buffer, 0, output.byte_size());
        desc.write_buffer(2, DescriptorType::UNIFORM_BUFFER, params_buf, 0, params_size);
        self.queue.one_shot(&self.device, self.queue_family, |cmd| {
            cmd.bind_compute_pipeline(pipeline);
            cmd.bind_compute_descriptor_sets(pipe_layout, 0, &[&desc]);
            cmd.dispatch(groups_x, groups_y, groups_z);
            Ok(())
        }).map_err(vk_err)
    }

    /// Dispatch a 3-storage + 1-uniform compute shader.
    fn dispatch_3buf(
        &self,
        pipeline: &ComputePipeline,
        pipe_layout: &PipelineLayout,
        a: &VulkanStorage,
        b: &VulkanStorage,
        output: &VulkanStorage,
        params_buf: &Buffer,
        params_size: u64,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) -> fuel_core_types::Result<()> {
        let desc = self.pipelines.desc_pool.allocate(&self.pipelines.layout_3s1u).map_err(vk_err)?;
        desc.write_buffer(0, DescriptorType::STORAGE_BUFFER, &a.buffer, 0, a.byte_size());
        desc.write_buffer(1, DescriptorType::STORAGE_BUFFER, &b.buffer, 0, b.byte_size());
        desc.write_buffer(2, DescriptorType::STORAGE_BUFFER, &output.buffer, 0, output.byte_size());
        desc.write_buffer(3, DescriptorType::UNIFORM_BUFFER, params_buf, 0, params_size);
        self.queue.one_shot(&self.device, self.queue_family, |cmd| {
            cmd.bind_compute_pipeline(pipeline);
            cmd.bind_compute_descriptor_sets(pipe_layout, 0, &[&desc]);
            cmd.dispatch(groups_x, groups_y, groups_z);
            Ok(())
        }).map_err(vk_err)
    }

    fn workgroups(n: usize) -> u32 {
        ((n + 255) / 256) as u32
    }
}

impl GraphBackend for VulkanBackend {
    type Storage = VulkanStorage;

    fn alloc_zeros(&self, shape: &Shape, dtype: DType) -> fuel_core_types::Result<Self::Storage> {
        let n = shape.elem_count();
        let byte_size = (n * dtype_size(dtype)) as u64;
        let storage = self.alloc_device(byte_size, n, dtype)?;
        self.queue.one_shot(&self.device, self.queue_family, |cmd| {
            cmd.fill_buffer(&storage.buffer, 0, byte_size, 0);
            Ok(())
        }).map_err(vk_err)?;
        Ok(storage)
    }

    fn upload(&self, buf: &fuel_core_types::HostBuffer, _shape: &Shape) -> fuel_core_types::Result<Self::Storage> {
        use fuel_core_types::HostBuffer;
        match buf {
            HostBuffer::F32(v) => self.upload_slice(v, DType::F32),
            HostBuffer::F64(v) => self.upload_slice(v, DType::F64),
            HostBuffer::U32(v) => self.upload_slice(v, DType::U32),
            _ => fuel_core_types::bail!("VulkanBackend: unsupported upload dtype"),
        }
    }

    fn download(&self, storage: &Self::Storage) -> fuel_core_types::Result<fuel_core_types::HostBuffer> {
        use fuel_core_types::HostBuffer;
        match storage.dtype {
            DType::F32 => Ok(HostBuffer::F32(self.download_slice::<f32>(storage)?)),
            DType::F64 => Ok(HostBuffer::F64(self.download_slice::<f64>(storage)?)),
            DType::U32 => Ok(HostBuffer::U32(self.download_slice::<u32>(storage)?)),
            other => fuel_core_types::bail!("VulkanBackend: unsupported download {other:?}"),
        }
    }

    fn try_clone(&self, storage: &Self::Storage, layout: &Layout) -> fuel_core_types::Result<Self::Storage> {
        let n = layout.shape().elem_count();
        let byte_size = (n * dtype_size(storage.dtype)) as u64;
        let dst = self.alloc_device(byte_size, n, storage.dtype)?;
        self.queue.one_shot(&self.device, self.queue_family, |cmd| {
            cmd.copy_buffer(&storage.buffer, &dst.buffer, &[BufferCopy {
                src_offset: 0, dst_offset: 0, size: byte_size,
            }]);
            Ok(())
        }).map_err(vk_err)?;
        Ok(dst)
    }

    fn copy_strided_src(
        &self, src: &Self::Storage, dst: &mut Self::Storage,
        dst_offset: usize, src_layout: &Layout,
    ) -> fuel_core_types::Result<()> {
        let shape = src_layout.shape();
        let dims = shape.dims();
        let strides = src_layout.stride();
        let rank = dims.len();
        let out_size = shape.elem_count();

        // Pack shape + strides into a single storage buffer.
        let mut sd: Vec<u32> = Vec::with_capacity(rank * 2);
        for &d in dims { sd.push(d as u32); }
        for &s in strides.iter() { sd.push(s as u32); }
        let (sd_buf, _sd_mem) = self.upload_slice_raw(&sd)?;

        // Params uniform buffer.
        #[repr(C)] #[derive(Clone, Copy)]
        struct SParams { out_size: u32, rank: u32, src_offset: u32, dst_offset: u32 }
        let p = SParams {
            out_size: out_size as u32,
            rank: rank as u32,
            src_offset: src_layout.start_offset() as u32,
            dst_offset: dst_offset as u32,
        };
        let (pbuf, _pmem) = self.upload_params(&p)?;

        // Allocate descriptor set: bindings 0=input, 1=output, 2=shape_strides, 3=params
        let desc = self.pipelines.desc_pool.allocate(&self.pipelines.layout_3s1u).map_err(vk_err)?;
        desc.write_buffer(0, DescriptorType::STORAGE_BUFFER, &src.buffer, 0, src.byte_size());
        desc.write_buffer(1, DescriptorType::STORAGE_BUFFER, &dst.buffer, 0, dst.byte_size());
        let sd_byte_size = (sd.len() * 4) as u64;
        desc.write_buffer(2, DescriptorType::STORAGE_BUFFER, &sd_buf, 0, sd_byte_size);
        desc.write_buffer(3, DescriptorType::UNIFORM_BUFFER, &pbuf, 0, 16);

        let groups = Self::workgroups(out_size);
        self.queue.one_shot(&self.device, self.queue_family, |cmd| {
            cmd.bind_compute_pipeline(&self.pipelines.strided_copy_pipeline);
            cmd.bind_compute_descriptor_sets(&self.pipelines.strided_copy_layout, 0, &[&desc]);
            cmd.dispatch(groups, 1, 1);
            Ok(())
        }).map_err(vk_err)
    }

    fn storage_dtype(&self, storage: &Self::Storage) -> DType {
        storage.dtype
    }

    // -- native GPU compute ops -----------------------------------------------

    fn matmul(
        &self, a: &Self::Storage, b: &Self::Storage,
        bmnk: (usize, usize, usize, usize),
        _la: &Layout, _lb: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        let (batch, m, n, k) = bmnk;
        let out_n = batch * m * n;
        let out = self.alloc_device((out_n * 4) as u64, out_n, DType::F32)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct MatmulParams { m: u32, n: u32, k: u32, sa: u32, sb: u32, sc: u32 }
        let params = MatmulParams {
            m: m as u32, n: n as u32, k: k as u32,
            sa: (m * k) as u32, sb: (k * n) as u32, sc: (m * n) as u32,
        };
        let (pbuf, _pmem) = self.upload_params(&params)?;
        let gx = ((n + 63) / 64) as u32;
        let gy = ((m + 63) / 64) as u32;
        let gz = batch as u32;
        self.dispatch_3buf(
            &self.pipelines.matmul_pipeline,
            &self.pipelines.matmul_layout,
            a, b, &out, &pbuf, std::mem::size_of::<MatmulParams>() as u64, gx, gy, gz,
        )?;
        Ok(out)
    }

    fn unary(&self, op: UnaryOp, a: &Self::Storage, _layout: &Layout) -> fuel_core_types::Result<Self::Storage> {
        let out = self.alloc_device(a.byte_size(), a.elem_count, a.dtype)?;

        let op_id: u32 = match op {
            UnaryOp::Neg => 0, UnaryOp::Sqr => 1, UnaryOp::Sqrt => 2,
            UnaryOp::Exp => 3, UnaryOp::Log => 4, UnaryOp::Sin => 5,
            UnaryOp::Cos => 6, UnaryOp::Tanh => 7, UnaryOp::Sigmoid => 8,
            UnaryOp::Silu => 9, UnaryOp::Gelu => 10, UnaryOp::Relu => 11,
            UnaryOp::Step => 12,
        };
        #[repr(C)] #[derive(Clone, Copy)]
        struct UParams { n: u32, op_id: u32 }
        let p = UParams { n: a.elem_count as u32, op_id };
        let (pbuf, _pmem) = self.upload_params(&p)?;
        self.dispatch_2buf(
            &self.pipelines.unary_pipeline,
            &self.pipelines.unary_layout,
            a, &out, &pbuf, 8, Self::workgroups(a.elem_count), 1, 1,
        )?;
        Ok(out)
    }

    fn binary(
        &self, op: BinaryOp,
        a: &Self::Storage, b: &Self::Storage,
        _la: &Layout, _lb: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        let out = self.alloc_device(a.byte_size(), a.elem_count, a.dtype)?;

        let op_id: u32 = match op {
            BinaryOp::Add => 0, BinaryOp::Sub => 1, BinaryOp::Mul => 2,
            BinaryOp::Div => 3, BinaryOp::Maximum => 4, BinaryOp::Minimum => 5,
        };
        #[repr(C)] #[derive(Clone, Copy)]
        struct BParams { n: u32, op_id: u32 }
        let p = BParams { n: a.elem_count as u32, op_id };
        let (pbuf, _pmem) = self.upload_params(&p)?;
        self.dispatch_3buf(
            &self.pipelines.binary_pipeline,
            &self.pipelines.binary_layout,
            a, b, &out, &pbuf, 8, Self::workgroups(a.elem_count), 1, 1,
        )?;
        Ok(out)
    }

    fn affine(
        &self, a: &Self::Storage, _layout: &Layout,
        mul: f64, add: f64,
    ) -> fuel_core_types::Result<Self::Storage> {
        let out = self.alloc_device(a.byte_size(), a.elem_count, a.dtype)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct AffParams { n: u32, _pad: u32, mul: f32, add: f32 }
        let p = AffParams { n: a.elem_count as u32, _pad: 0, mul: mul as f32, add: add as f32 };
        let (pbuf, _pmem) = self.upload_params(&p)?;
        self.dispatch_2buf(
            &self.pipelines.affine_pipeline,
            &self.pipelines.affine_layout,
            a, &out, &pbuf, 16, Self::workgroups(a.elem_count), 1, 1,
        )?;
        Ok(out)
    }

    fn powf(
        &self, _a: &Self::Storage, _layout: &Layout, _exp: f64,
    ) -> fuel_core_types::Result<Self::Storage> {
        // powf: x^exp = exp(exp * ln(x)). Can compose from affine + unary
        // but for now fall back to CPU.
        fuel_core_types::bail!("VulkanBackend: powf not yet native")
    }

    fn cast(
        &self, _a: &Self::Storage, _layout: &Layout, _dtype: DType,
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: cast not yet native")
    }

    fn reduce(
        &self, op: fuel_core_types::op::ReduceOp,
        a: &Self::Storage, _layout: &Layout,
        dims: &[usize],
    ) -> fuel_core_types::Result<Self::Storage> {
        // Only support full reduction (all dims) for now.
        if dims.len() < 2 {
            fuel_core_types::bail!("VulkanBackend: per-dim reduce not yet native");
        }
        let out = self.alloc_device(4, 1, DType::F32)?;

        let op_id: u32 = match op {
            fuel_core_types::op::ReduceOp::Sum => 0,
            fuel_core_types::op::ReduceOp::Max => 1,
            fuel_core_types::op::ReduceOp::Min => 2,
            _ => fuel_core_types::bail!("VulkanBackend: unsupported reduce op"),
        };
        #[repr(C)] #[derive(Clone, Copy)]
        struct RParams { n: u32, op_id: u32 }
        let p = RParams { n: a.elem_count as u32, op_id };
        let (pbuf, _pmem) = self.upload_params(&p)?;
        self.dispatch_2buf(
            &self.pipelines.reduce_pipeline,
            &self.pipelines.reduce_layout,
            a, &out, &pbuf, 8, 1, 1, 1,
        )?;
        Ok(out)
    }

    fn softmax_last_dim(
        &self, a: &Self::Storage, layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        let shape = layout.shape();
        let dims = shape.dims();
        let n_cols = *dims.last().expect("softmax: empty shape");
        let n_rows = (a.elem_count / n_cols) as u32;
        let out = self.alloc_device(a.byte_size(), a.elem_count, a.dtype)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct SoftParams { n_rows: u32, n_cols: u32 }
        let p = SoftParams { n_rows, n_cols: n_cols as u32 };
        let (pbuf, _pmem) = self.upload_params(&p)?;
        self.dispatch_2buf(
            &self.pipelines.softmax_pipeline,
            &self.pipelines.softmax_layout,
            a, &out, &pbuf, 8, n_rows, 1, 1,
        )?;
        Ok(out)
    }

    fn index_select(
        &self, _src: &Self::Storage, _ids: &Self::Storage,
        _src_l: &Layout, _ids_l: &Layout, _dim: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: index_select not yet native")
    }

    fn gather(
        &self, _src: &Self::Storage, _ids: &Self::Storage,
        _src_l: &Layout, _ids_l: &Layout, _dim: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: gather not yet native")
    }
}

// -- utilities ----------------------------------------------------------------

fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::F32 | DType::U32 | DType::I32 => 4,
        DType::F64 | DType::I64 => 8,
        DType::F16 | DType::BF16 | DType::I16 => 2,
        DType::U8 => 1,
        _ => 4,
    }
}

fn vk_err(e: impl std::fmt::Debug) -> fuel_core_types::Error {
    fuel_core_types::Error::Msg(format!("Vulkan: {e:?}"))
}

/// Reinterpret a #[repr(C)] struct as a byte slice for push constants.
unsafe fn as_bytes<T: Sized>(p: &T) -> &[u8] {
    std::slice::from_raw_parts(p as *const T as *const u8, std::mem::size_of::<T>())
}

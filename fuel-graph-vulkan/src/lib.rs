//! Vulkan GPU executor for fuel-graph computation graphs.
//!
//! Uses Vulkane for Vulkan device management and dispatches compute
//! ops through WGSL shaders compiled to SPIR-V via naga. Third
//! backend for fuel's generic `GraphExecutor<B>`.

pub mod pipelines;
mod recorder;

use fuel_core_types::{DType, Layout, Shape};
use fuel_graph_executor::{BinaryOp, GraphBackend, UnaryOp};
use pipelines::Pipelines;
use recorder::{OpStats, OpStatEntry, Recorder};
use std::cell::RefCell;
use std::time::Instant;
use tracing::{debug_span, info_span};
use vulkane::safe::*;

/// The Arc-shared GPU buffer + its backing allocation. Separating this
/// from `VulkanStorage` lets us cheaply clone a storage handle (just
/// bump the Arc refcount) for pure-shape-relabel clones like reshape
/// and pure-pad broadcast — no GPU memcpy, no device allocation.
///
/// `allocation` is a sub-allocation from vulkane's VMA-style allocator.
/// Dropping the `VulkanBuffer` destroys the `Buffer` (vkDestroyBuffer)
/// and returns the `Allocation` to its pool. The underlying
/// `VkDeviceMemory` block is shared with many other buffers, so we
/// never hit `maxMemoryAllocationCount` (~4096) no matter how many
/// buffers we create in one forward.
pub struct VulkanBuffer {
    pub buffer: Buffer,
    pub allocation: Allocation,
}

/// Vulkan storage: Arc-shared device buffer + per-view metadata.
pub struct VulkanStorage {
    pub inner: std::sync::Arc<VulkanBuffer>,
    pub elem_count: usize,
    pub dtype: DType,
}

impl VulkanStorage {
    pub fn buffer(&self) -> &Buffer { &self.inner.buffer }

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
    /// Shared VMA-style sub-allocator. Every buffer we create goes
    /// through this so the number of live `VkDeviceMemory` blocks
    /// stays O(GB-of-memory / 256MB), not O(number-of-buffers).
    pub allocator: std::sync::Arc<Allocator>,
    /// Async-submission state: pool of in-flight command buffers and
    /// their transient resources. `RefCell` because `GraphBackend`
    /// methods take `&self` — we need interior mutability to push
    /// pending work. Single-threaded; no contention.
    recorder: RefCell<Recorder>,
    /// Supported cooperative-matrix tile shapes, queried at init from
    /// `VK_KHR_cooperative_matrix`. Empty if the extension is not
    /// available. Used by the matmul dispatch to decide whether to
    /// route large-M × bf16-B matmuls through a tensor-core kernel.
    coop_matrix_shapes: Vec<CooperativeMatrixProperties>,
    /// Per-op-kind host-side timing. Counts and cumulative wall time
    /// spent inside `record_dispatch` for each op category. Useful
    /// for diagnosing whether submission overhead is the bottleneck
    /// and for feeding future backend cost estimates to a scheduler.
    pub op_stats: OpStats,
}

impl VulkanBackend {
    /// Snapshot of per-op-kind timing accumulated since init or since
    /// the last `reset_op_stats()` call. Sorted by total time
    /// descending. Host-side only — does not include GPU execution
    /// time (that would require Vulkan timestamp queries).
    pub fn op_stats_snapshot(&self) -> Vec<(&'static str, OpStatEntry)> {
        self.op_stats.snapshot()
    }

    /// Zero the op-stats counters. Useful between timed phases
    /// (e.g. skip model-load stats; just measure generation).
    pub fn reset_op_stats(&self) {
        self.op_stats.reset();
    }
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

        // Probe for optional extensions. Cooperative matrix gives us
        // tensor-core-class matmul on hardware that supports it
        // (NVIDIA Volta+, AMD RDNA 3+).
        let ext_props = physical.enumerate_extension_properties().map_err(vk_err)?;
        let has_coop_matrix = ext_props.iter()
            .any(|e| e.name() == "VK_KHR_cooperative_matrix");

        let features = if has_coop_matrix {
            Some(DeviceFeatures::new().with_cooperative_matrix())
        } else {
            None
        };
        let extensions = if has_coop_matrix {
            Some(DeviceExtensions::new().khr_cooperative_matrix())
        } else {
            None
        };

        let device = physical.create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo::single(queue_family)],
            enabled_features: features.as_ref(),
            enabled_extensions: extensions.as_ref(),
            ..Default::default()
        }).map_err(vk_err)?;

        // Query supported cooperative-matrix tile shapes. If the
        // extension isn't enabled, the query returns empty.
        let coop_matrix_shapes: Vec<CooperativeMatrixProperties> = if has_coop_matrix {
            unsafe { physical.cooperative_matrix_properties() }
        } else {
            Vec::new()
        };
        if !coop_matrix_shapes.is_empty() {
            tracing::info!(
                n_shapes = coop_matrix_shapes.len(),
                "VK_KHR_cooperative_matrix supported — queried tile shapes",
            );
            for (i, s) in coop_matrix_shapes.iter().enumerate() {
                tracing::debug!(
                    shape = i,
                    m = s.m_size(), n = s.n_size(), k = s.k_size(),
                    a_type = ?s.a_type(), b_type = ?s.b_type(),
                    c_type = ?s.c_type(), result_type = ?s.result_type(),
                    "coop matrix shape",
                );
                eprintln!(
                    "  coop[{i}] M={} N={} K={} A={:?} B={:?} C={:?} R={:?} sat={}",
                    s.m_size(), s.n_size(), s.k_size(),
                    s.a_type(), s.b_type(), s.c_type(), s.result_type(),
                    s.saturating_accumulation(),
                );
            }
        } else {
            eprintln!("  [coop-matrix] not available (has_coop_matrix={has_coop_matrix})");
        }

        let queue = device.get_queue(queue_family, 0);

        let pipelines = Pipelines::new(&device, has_coop_matrix).map_err(vk_err)?;
        let recorder = RefCell::new(Recorder::new(&device, queue_family).map_err(vk_err)?);
        let allocator = std::sync::Arc::new(Allocator::new(&device, &physical).map_err(vk_err)?);

        Ok(Self {
            device,
            physical,
            queue,
            queue_family,
            pipelines,
            device_name,
            allocator,
            recorder,
            op_stats: OpStats::default(),
            coop_matrix_shapes,
        })
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
        let byte_size = (data.len() * std::mem::size_of::<T>()) as u64;
        let _span = debug_span!("vk_upload_slice", bytes = byte_size).entered();
        // Staging: host-visible + mapped. Sub-allocated from the
        // host-visible pool.
        let (staging_buf, staging_alloc) = self
            .allocator
            .create_buffer(
                BufferCreateInfo {
                    size: byte_size.max(1),
                    usage: BufferUsage::TRANSFER_SRC,
                },
                AllocationCreateInfo {
                    usage: AllocationUsage::HostVisible,
                    mapped: true,
                    ..Default::default()
                },
            )
            .map_err(vk_err)?;
        // Write the bytes into the staging buffer via its mapped pointer.
        let mapped = staging_alloc
            .mapped_ptr()
            .ok_or_else(|| fuel_core_types::Error::Msg(
                "upload_slice: staging alloc not mapped".into()))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                mapped as *mut u8,
                byte_size as usize,
            );
        }
        // Device-local target.
        let (gpu_buf, gpu_alloc) = self
            .allocator
            .create_buffer(
                BufferCreateInfo {
                    size: byte_size.max(1),
                    usage: BufferUsage::STORAGE_BUFFER
                        | BufferUsage::TRANSFER_SRC
                        | BufferUsage::TRANSFER_DST,
                },
                AllocationCreateInfo {
                    usage: AllocationUsage::DeviceLocal,
                    ..Default::default()
                },
            )
            .map_err(vk_err)?;
        // One-shot copy staging -> device. This syncs on its own
        // fence, so when it returns the GPU has fully processed
        // the copy.
        self.queue
            .one_shot(&self.device, self.queue_family, |cmd| {
                cmd.copy_buffer(
                    &staging_buf,
                    &gpu_buf,
                    &[BufferCopy { src_offset: 0, dst_offset: 0, size: byte_size.max(1) }],
                );
                Ok(())
            })
            .map_err(vk_err)?;
        // staging_buf + staging_alloc drop here, returning their
        // sub-allocation to the pool. gpu_buf + gpu_alloc live on
        // inside the returned VulkanStorage.
        drop(staging_buf);
        drop(staging_alloc);
        Ok(VulkanStorage {
            inner: std::sync::Arc::new(VulkanBuffer {
                buffer: gpu_buf,
                allocation: gpu_alloc,
            }),
            elem_count: data.len(),
            dtype,
        })
    }

    fn download_slice<T: Copy + Default + 'static>(
        &self, storage: &VulkanStorage,
    ) -> fuel_core_types::Result<Vec<T>> {
        let byte_size = storage.byte_size();
        let n = storage.elem_count;
        let pending = self.recorder.borrow().pending.len();
        let _span = info_span!("vk_download", bytes = byte_size, pending).entered();
        // First make sure every previously-submitted async op has
        // finished on the GPU. flush_pending host-waits on our
        // timeline semaphore and drops in-flight resources.
        self.flush_pending()?;
        // Staging via the allocator (host-visible + mapped).
        let (staging_buf, staging_alloc) = {
            let _s = debug_span!("vk_download_alloc_staging").entered();
            self.allocator.create_buffer(
                BufferCreateInfo { size: byte_size.max(1), usage: BufferUsage::TRANSFER_DST },
                AllocationCreateInfo {
                    usage: AllocationUsage::HostVisible,
                    mapped: true,
                    ..Default::default()
                },
            ).map_err(vk_err)?
        };
        {
            let _s = info_span!("vk_download_copy").entered();
            self.queue.one_shot(&self.device, self.queue_family, |cmd| {
                cmd.copy_buffer(storage.buffer(), &staging_buf, &[BufferCopy {
                    src_offset: 0, dst_offset: 0, size: byte_size,
                }]);
                Ok(())
            }).map_err(vk_err)?;
        }
        let _s = debug_span!("vk_download_memcpy").entered();
        let mapped = staging_alloc
            .mapped_ptr()
            .ok_or_else(|| fuel_core_types::Error::Msg(
                "download_slice: staging alloc not mapped".into()))?;
        let mut out = vec![T::default(); n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped as *const u8,
                out.as_mut_ptr() as *mut u8,
                n * std::mem::size_of::<T>(),
            );
        }
        drop(staging_buf);
        drop(staging_alloc);
        Ok(out)
    }

    fn alloc_device(&self, byte_size: u64, n: usize, dtype: DType) -> fuel_core_types::Result<VulkanStorage> {
        let (buffer, allocation) = self.allocator.create_buffer(
            BufferCreateInfo {
                size: byte_size.max(1),
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
            },
            AllocationCreateInfo {
                usage: AllocationUsage::DeviceLocal,
                ..Default::default()
            },
        ).map_err(vk_err)?;
        Ok(VulkanStorage {
            inner: std::sync::Arc::new(VulkanBuffer { buffer, allocation }),
            elem_count: n,
            dtype,
        })
    }

    /// Upload a typed slice as a host-visible storage buffer. Used
    /// for small per-dispatch metadata (shape/strides arrays, index
    /// tables). Sub-allocates from the shared allocator's host-visible
    /// pool so we don't hit `maxMemoryAllocationCount` even when
    /// issuing thousands of these per forward.
    fn upload_slice_raw<T: Copy + 'static>(&self, data: &[T]) -> fuel_core_types::Result<(Buffer, Allocation)> {
        let byte_size = (data.len() * std::mem::size_of::<T>()) as u64;
        let _span = debug_span!("vk_upload_slice_raw", bytes = byte_size).entered();
        let size = byte_size.max(16);
        let (buf, alloc) = self.allocator.create_buffer(
            BufferCreateInfo { size, usage: BufferUsage::STORAGE_BUFFER },
            AllocationCreateInfo {
                usage: AllocationUsage::HostVisible,
                mapped: true,
                ..Default::default()
            },
        ).map_err(vk_err)?;
        let mapped = alloc.mapped_ptr()
            .ok_or_else(|| fuel_core_types::Error::Msg(
                "upload_slice_raw: alloc not mapped".into()))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                mapped as *mut u8,
                byte_size as usize,
            );
        }
        Ok((buf, alloc))
    }

    /// Upload a small params struct as a uniform buffer. Sub-allocated
    /// from the shared allocator's host-visible pool.
    fn upload_params<T: Copy + 'static>(&self, params: &T) -> fuel_core_types::Result<(Buffer, Allocation)> {
        let _span = debug_span!("vk_upload_params", bytes = std::mem::size_of::<T>()).entered();
        let bytes = unsafe { as_bytes(params) };
        let size = (bytes.len().max(16)) as u64;
        let (buf, alloc) = self.allocator.create_buffer(
            BufferCreateInfo { size, usage: BufferUsage::UNIFORM_BUFFER },
            AllocationCreateInfo {
                usage: AllocationUsage::HostVisible,
                mapped: true,
                ..Default::default()
            },
        ).map_err(vk_err)?;
        let mapped = alloc.mapped_ptr()
            .ok_or_else(|| fuel_core_types::Error::Msg(
                "upload_params: alloc not mapped".into()))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped as *mut u8,
                bytes.len(),
            );
        }
        Ok((buf, alloc))
    }

    /// Record one op into a fresh command buffer, attach a leading
    /// compute→compute WRITE→READ memory barrier so prior work is
    /// visible, submit to the queue without waiting, and stash the CB
    /// + transient resources on the pending list. The CPU returns as
    /// soon as the driver accepts the submit — the GPU runs the op in
    /// the background. Matches CUDA's async stream-launch semantics.
    ///
    /// `desc` is passed by value but a reference is handed to
    /// `record_fn` so the closure can bind it; the descriptor moves
    /// into the pending list afterward to keep it alive until the GPU
    /// consumes this CB.
    /// Max in-flight submits before we auto-flush. Windows WDDM's
    /// TDR kills GPU contexts whose "current run" of work exceeds
    /// ~2 seconds without a yield point. With the async refactor +
    /// native kernels, we eliminated the implicit sync points that
    /// cpu_fallback downloads were providing — so the queue can
    /// grow until the final download, and a multi-thousand-op run
    /// easily blows past 2s. Bounding queue depth keeps each GPU
    /// run short enough that the driver stays happy. 128 is a
    /// compromise: deep enough to keep the GPU busy, shallow enough
    /// that each flush completes well under the TDR window.
    const AUTO_FLUSH_THRESHOLD: usize = 128;

    fn record_dispatch<F>(
        &self,
        op_name: &'static str,
        transient_buffers: Vec<(Buffer, Allocation)>,
        desc: Option<DescriptorSet>,
        record_fn: F,
    ) -> fuel_core_types::Result<()>
    where
        F: FnOnce(&mut CommandBufferRecording<'_>, Option<&DescriptorSet>) -> Result<()>,
    {
        let _span = debug_span!("vk_record_dispatch", op = op_name).entered();
        let t0 = Instant::now();

        // Auto-flush: bound the pending queue to keep each GPU
        // "chunk" small enough for WDDM TDR and small enough that
        // intermediate results can make progress for downstream ops.
        if self.recorder.borrow().pending.len() >= Self::AUTO_FLUSH_THRESHOLD {
            self.flush_pending()?;
        }
        let mut rec = self.recorder.borrow_mut();
        let pending_before = rec.pending.len();
        let mut cmd = {
            let _a = debug_span!("vk_alloc_cb").entered();
            rec.pool.allocate_primary().map_err(vk_err)?
        };
        {
            let _r = debug_span!("vk_record_cb").entered();
            let mut recording = cmd.begin().map_err(vk_err)?;
            record_fn(&mut recording, desc.as_ref()).map_err(vk_err)?;
            recording.end().map_err(vk_err)?;
        }
        // Cross-submit synchronization: chain this submit onto the
        // prior one via a timeline semaphore. Each submit waits for
        // the previous counter value before starting and signals
        // (counter+1) on completion. On NVIDIA, relying on an
        // in-CB `vkCmdPipelineBarrier` alone was not reliable — we
        // crashed with `ERROR_DEVICE_LOST` at any queue depth ≥ 2.
        // Timeline semaphores are the spec-canonical primitive for
        // this and drivers handle them reliably. The chain serializes
        // GPU execution (same behavior as having one big command
        // buffer) while keeping the CPU free to queue more work.
        let wait_value = rec.counter;
        let signal_value = wait_value + 1;
        rec.counter = signal_value;
        {
            let _s = debug_span!("vk_queue_submit", pending = pending_before).entered();
            let waits: Vec<WaitSemaphore<'_>> = if wait_value == 0 {
                Vec::new()
            } else {
                vec![WaitSemaphore::timeline(
                    &rec.timeline,
                    wait_value,
                    PipelineStage::ALL_COMMANDS,
                )]
            };
            let signals = [SignalSemaphore::timeline(&rec.timeline, signal_value)];
            self.queue
                .submit_with_sync(&[&cmd], &waits, &signals, None)
                .map_err(vk_err)?;
        }
        rec.pending.push(recorder::PendingSubmit {
            cmd,
            transient_buffers,
            transient_desc: desc,
        });
        drop(rec);
        self.op_stats.record(op_name, t0.elapsed());
        Ok(())
    }

    /// Called after a synchronous sync point (D2H copy's one_shot).
    /// Drops every in-flight CB + its transient resources, recycles
    /// the command pool, and also destroys any descriptor pools that
    /// were retired during the forward pass. The retired pools held
    /// descriptors that were potentially in-flight on the GPU; now
    /// that the fence has signaled, the GPU is confirmed idle and
    /// their backing `vkDestroyDescriptorPool` is safe to call.
    fn drain_recorder(&self) -> fuel_core_types::Result<()> {
        let pending = self.recorder.borrow().pending.len();
        let _span = info_span!("vk_drain_recorder", pending).entered();
        self.recorder
            .borrow_mut()
            .drain(&self.device, self.queue_family)
            .map_err(vk_err)?;
        // Pending CBs (which held the descriptor sets) are now
        // dropped. Safe to destroy any retired descriptor pools.
        self.pipelines.retire_pools_post_drain();
        Ok(())
    }

    /// Force the GPU to catch up. Host-waits on the recorder's
    /// timeline semaphore for the most recently signaled value,
    /// which guarantees every previously-submitted CB has completed
    /// on the GPU. Then drops all in-flight resources and recycles
    /// the pool. Called periodically to bound queue depth so
    /// Windows WDDM doesn't kill us with a TDR, and from
    /// `download_slice` before the D2H copy.
    fn flush_pending(&self) -> fuel_core_types::Result<()> {
        let (pending, target_value) = {
            let rec = self.recorder.borrow();
            (rec.pending.len(), rec.counter)
        };
        let _span = info_span!("vk_flush_pending", pending, target_value).entered();
        if target_value > 0 {
            self.recorder
                .borrow()
                .timeline
                .wait_value(target_value, u64::MAX)
                .map_err(vk_err)?;
        }
        // GPU is now idle (up through target_value). Drop everything pending.
        self.drain_recorder()
    }

    /// Dispatch a 2-storage + 1-uniform compute shader.
    /// `params_buf` + `params_mem` transfer ownership; they're kept
    /// alive by the recorder until the GPU consumes this CB.
    fn dispatch_2buf(
        &self,
        op_name: &'static str,
        pipeline: &ComputePipeline,
        pipe_layout: &PipelineLayout,
        input: &VulkanStorage,
        output: &VulkanStorage,
        params_buf: Buffer,
        params_alloc: Allocation,
        params_size: u64,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) -> fuel_core_types::Result<()> {
        let desc = self.pipelines.allocate_desc(&self.pipelines.layout_2s1u).map_err(vk_err)?;
        desc.write_buffer(0, DescriptorType::STORAGE_BUFFER, input.buffer(), 0, input.byte_size());
        desc.write_buffer(1, DescriptorType::STORAGE_BUFFER, output.buffer(), 0, output.byte_size());
        desc.write_buffer(2, DescriptorType::UNIFORM_BUFFER, &params_buf, 0, params_size);
        self.record_dispatch(
            op_name,
            vec![(params_buf, params_alloc)],
            Some(desc),
            |cmd, desc| {
                let desc = desc.expect("dispatch_2buf: descriptor set missing");
                cmd.bind_compute_pipeline(pipeline);
                cmd.bind_compute_descriptor_sets(pipe_layout, 0, &[desc]);
                cmd.dispatch(groups_x, groups_y, groups_z);
                Ok(())
            },
        )
    }

    /// Dispatch a 3-storage + 1-uniform compute shader.
    fn dispatch_3buf(
        &self,
        op_name: &'static str,
        pipeline: &ComputePipeline,
        pipe_layout: &PipelineLayout,
        a: &VulkanStorage,
        b: &VulkanStorage,
        output: &VulkanStorage,
        params_buf: Buffer,
        params_alloc: Allocation,
        params_size: u64,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) -> fuel_core_types::Result<()> {
        let desc = self.pipelines.allocate_desc(&self.pipelines.layout_3s1u).map_err(vk_err)?;
        desc.write_buffer(0, DescriptorType::STORAGE_BUFFER, a.buffer(), 0, a.byte_size());
        desc.write_buffer(1, DescriptorType::STORAGE_BUFFER, b.buffer(), 0, b.byte_size());
        desc.write_buffer(2, DescriptorType::STORAGE_BUFFER, output.buffer(), 0, output.byte_size());
        desc.write_buffer(3, DescriptorType::UNIFORM_BUFFER, &params_buf, 0, params_size);
        self.record_dispatch(
            op_name,
            vec![(params_buf, params_alloc)],
            Some(desc),
            |cmd, desc| {
                let desc = desc.expect("dispatch_3buf: descriptor set missing");
                cmd.bind_compute_pipeline(pipeline);
                cmd.bind_compute_descriptor_sets(pipe_layout, 0, &[desc]);
                cmd.dispatch(groups_x, groups_y, groups_z);
                Ok(())
            },
        )
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
        // No zero-fill dispatch. Every downstream op (matmul, unary,
        // binary, permute, broadcast, concat, reduce, softmax, …)
        // writes every element of its output buffer, so the fill was
        // pure overhead — ~24µs of host-side dispatch cost ×22K calls
        // per 32-token generation = ~550ms wasted. If a future op
        // genuinely needs zero-initialized storage, add an explicit
        // fill_buffer at that call site rather than taxing every alloc.
        self.alloc_device(byte_size, n, dtype)
    }

    fn upload(&self, buf: &fuel_core_types::HostBuffer, _shape: &Shape) -> fuel_core_types::Result<Self::Storage> {
        // Uploads are synchronous (queue.upload_buffer submits its own
        // CB + fence and waits) but the fence only covers the upload
        // itself — not our async submit chain. On Windows/NVIDIA we
        // empirically see DEVICE_LOST when upload CBs race with
        // concurrently-executing compute CBs from our async queue.
        // Flushing our pending chain before each upload keeps the
        // queue quiet while the upload runs, and is cheap (idempotent
        // if nothing is pending).
        self.flush_pending()?;
        use fuel_core_types::HostBuffer;
        use half::{bf16, f16};
        match buf {
            HostBuffer::F32(v) => self.upload_slice(v, DType::F32),
            HostBuffer::F64(v) => self.upload_slice(v, DType::F64),
            HostBuffer::U32(v) => self.upload_slice(v, DType::U32),
            // Half-precision storage. The upload path is generic over
            // `Copy + 'static` so the bytes land on device in their
            // native 2-byte layout — shaders that want to read them
            // natively will need the 16-bit-storage extension, or
            // they can unpack u32-packed pairs manually.
            HostBuffer::BF16(v) => {
                let _: &[bf16] = v; // type witness
                self.upload_slice(v, DType::BF16)
            }
            HostBuffer::F16(v) => {
                let _: &[f16] = v;
                self.upload_slice(v, DType::F16)
            }
            _ => fuel_core_types::bail!("VulkanBackend: unsupported upload dtype"),
        }
    }

    fn download(&self, storage: &Self::Storage) -> fuel_core_types::Result<fuel_core_types::HostBuffer> {
        // Optional allocator-stats tracing. Set FUEL_VK_ALLOC_LOG=1 to
        // print snapshots before and after the download-time flush.
        // The pre/post delta localizes whether alloc accumulation is
        // inside a single realize() (freed by flush) or across them
        // (persists past flush — KVCache retention, const pool, etc).
        let alloc_log = std::env::var("FUEL_VK_ALLOC_LOG").is_ok();
        if alloc_log {
            let s = self.allocator.statistics();
            eprintln!(
                "[vk-alloc pre ] allocs={} bytes={} blocks={} block_bytes={} free_regions={}",
                s.allocation_count, s.allocation_bytes, s.block_count,
                s.block_bytes, s.free_region_count,
            );
        }
        use fuel_core_types::HostBuffer;
        use half::{bf16, f16};
        let result = match storage.dtype {
            DType::F32 => Ok(HostBuffer::F32(self.download_slice::<f32>(storage)?)),
            DType::F64 => Ok(HostBuffer::F64(self.download_slice::<f64>(storage)?)),
            DType::U32 => Ok(HostBuffer::U32(self.download_slice::<u32>(storage)?)),
            DType::BF16 => Ok(HostBuffer::BF16(self.download_slice::<bf16>(storage)?)),
            DType::F16 => Ok(HostBuffer::F16(self.download_slice::<f16>(storage)?)),
            other => fuel_core_types::bail!("VulkanBackend: unsupported download {other:?}"),
        };
        if alloc_log {
            let s = self.allocator.statistics();
            eprintln!(
                "[vk-alloc post] allocs={} bytes={} blocks={} block_bytes={} free_regions={}",
                s.allocation_count, s.allocation_bytes, s.block_count,
                s.block_bytes, s.free_region_count,
            );
        }
        result
    }

    fn try_clone(&self, storage: &Self::Storage, layout: &Layout) -> fuel_core_types::Result<Self::Storage> {
        let n = layout.shape().elem_count();
        // Zero-copy fast path: if the target element count matches the
        // source, this clone is a pure shape relabel (reshape, pure-pad
        // broadcast). Share the Arc'd buffer instead of memcpying. On
        // an 8GB GPU with ~4GB of weights, this is the difference
        // between fitting and OOMing.
        if n == storage.elem_count {
            return Ok(VulkanStorage {
                inner: std::sync::Arc::clone(&storage.inner),
                elem_count: n,
                dtype: storage.dtype,
            });
        }
        let byte_size = (n * dtype_size(storage.dtype)) as u64;
        let dst = self.alloc_device(byte_size, n, storage.dtype)?;
        let src_arc = storage.inner.clone();
        let dst_arc = dst.inner.clone();
        self.record_dispatch("try_clone.memcpy", Vec::new(), None, move |cmd, _| {
            cmd.copy_buffer(&src_arc.buffer, &dst_arc.buffer, &[BufferCopy {
                src_offset: 0, dst_offset: 0, size: byte_size,
            }]);
            Ok(())
        })?;
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
        let (sd_buf, sd_mem) = self.upload_slice_raw(&sd)?;

        // Params uniform buffer.
        #[repr(C)] #[derive(Clone, Copy)]
        struct SParams { out_size: u32, rank: u32, src_offset: u32, dst_offset: u32 }
        let p = SParams {
            out_size: out_size as u32,
            rank: rank as u32,
            src_offset: src_layout.start_offset() as u32,
            dst_offset: dst_offset as u32,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;

        // Allocate descriptor set: bindings 0=input, 1=output, 2=shape_strides, 3=params
        let desc = self.pipelines.allocate_desc(&self.pipelines.layout_3s1u).map_err(vk_err)?;
        desc.write_buffer(0, DescriptorType::STORAGE_BUFFER, src.buffer(), 0, src.byte_size());
        desc.write_buffer(1, DescriptorType::STORAGE_BUFFER, dst.buffer(), 0, dst.byte_size());
        let sd_byte_size = (sd.len() * 4) as u64;
        desc.write_buffer(2, DescriptorType::STORAGE_BUFFER, &sd_buf, 0, sd_byte_size);
        desc.write_buffer(3, DescriptorType::UNIFORM_BUFFER, &pbuf, 0, 16);

        let groups = Self::workgroups(out_size);
        let pipeline = &self.pipelines.strided_copy_pipeline;
        let layout = &self.pipelines.strided_copy_layout;
        self.record_dispatch(
            "strided_copy",
            vec![(sd_buf, sd_mem), (pbuf, pmem)],
            Some(desc),
            |cmd, desc| {
                let desc = desc.expect("copy_strided_src: descriptor missing");
                cmd.bind_compute_pipeline(pipeline);
                cmd.bind_compute_descriptor_sets(layout, 0, &[desc]);
                cmd.dispatch(groups, 1, 1);
                Ok(())
            },
        )
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
        struct MatmulParams {
            m: u32, n: u32, k: u32,
            // A strides: per-batch-head, per-row, per-col
            sa_batch: u32, sa_row: u32, sa_col: u32,
            // B strides: per-batch-head, per-row, per-col
            sb_batch: u32, sb_row: u32, sb_col: u32,
            // C batch stride (output always contiguous: row=N, col=1)
            sc_batch: u32,
            n_rep: u32,
            _pad: u32,
        }

        // Extract per-dim strides from Layout. The last two dims are
        // (rows, cols); everything before is batched.
        let a_strides = _la.stride();
        let b_strides = _lb.stride();
        let a_rank = a_strides.len();
        let b_rank = b_strides.len();

        // Batch stride = stride of the first "batch" dim if rank >= 3.
        // For rank-2 (no batch), batch_stride = m*k / k*n — doesn't
        // matter since batch==1 and we never index past 0.
        let sa_batch = if a_rank >= 3 { a_strides[a_rank - 3] } else { m * k };
        let sa_row = a_strides[a_rank - 2];
        let sa_col = a_strides[a_rank - 1];

        let sb_batch = if b_rank >= 3 { b_strides[b_rank - 3] } else { k * n };
        let sb_row = b_strides[b_rank - 2];
        let sb_col = b_strides[b_rank - 1];

        // GQA-aware: infer n_rep from the A/B batch stride ratio.
        // For [1,32,...] × [1,4,...]: the B batch stride covers fewer
        // heads; the kernel reads B[batch/n_rep].
        let b_batch_count = if sb_batch > 0 { b.elem_count / (sb_batch as usize).max(1) } else { batch };
        let n_rep = if batch > b_batch_count && b_batch_count > 0 && batch % b_batch_count == 0 {
            batch / b_batch_count
        } else {
            1
        };

        let params = MatmulParams {
            m: m as u32, n: n as u32, k: k as u32,
            sa_batch: sa_batch as u32, sa_row: sa_row as u32, sa_col: sa_col as u32,
            sb_batch: sb_batch as u32, sb_row: sb_row as u32, sb_col: sb_col as u32,
            sc_batch: (m * n) as u32,
            n_rep: n_rep as u32, _pad: 0,
        };
        let (pbuf, pmem) = self.upload_params(&params)?;
        let gz = batch as u32;
        let params_size = std::mem::size_of::<MatmulParams>() as u64;

        // Shape- and dtype-based pipeline selection:
        //   A:f32, B:f32  — existing all-f32 paths
        //   A:f32, B:bf16 — mixed-precision path (decode w/ bf16 weights)
        //                   Only the M==1 gemv variant exists today;
        //                   reg-tile/tiled bf16 variants are a follow-up.
        //   M == 1 -> gemv (subgroup-reduced dot, one wg per column)
        //   M small -> WGSL register-tile (no shared-mem barriers)
        //   M large -> GLSL shared-memory tiled matmul
        let mixed_bf16 = a.dtype == DType::F32 && b.dtype == DType::BF16;
        if !(a.dtype == DType::F32 && b.dtype == DType::F32) && !mixed_bf16 {
            fuel_core_types::bail!(
                "VulkanBackend::matmul: unsupported dtypes A={:?} B={:?}",
                a.dtype, b.dtype
            );
        }
        if m == 1 {
            let gx = n as u32;
            let gy = 1u32;
            let (pipeline, pipe_layout, op_name) = if mixed_bf16 {
                (
                    &self.pipelines.matvec_bf16_b_pipeline,
                    &self.pipelines.matvec_bf16_b_layout,
                    "matvec_bf16_b",
                )
            } else {
                (
                    &self.pipelines.matvec_pipeline,
                    &self.pipelines.matvec_layout,
                    "matvec",
                )
            };
            self.dispatch_3buf(
                op_name, pipeline, pipe_layout,
                a, b, &out, pbuf, pmem, params_size, gx, gy, gz,
            )?;
        } else if mixed_bf16 {
            // Mixed-precision: try cooperative-matrix (tensor-core)
            // path first for large tiles; fall back to the tiled path.
            // Cooperative matrix requires tile-aligned N (coopMatStore
            // writes full 16-col blocks, no per-element bounds check).
            // M and K only need to be ≥ 16; out-of-bounds M-rows get
            // safe extra padding in the output buffer.
            if m >= 16 && n >= 16 && k >= 16
                && n % 16 == 0
                && self.pipelines.matmul_coop_pipeline.is_some()
            {
                // Pad M to next multiple of 16 so the coop kernel's
                // coopMatStore doesn't write past the output buffer.
                // The extra rows are wasted but harmless.
                let padded_m = ((m + 15) / 16) * 16;
                let padded_out_n = batch * padded_m * n;
                let padded_out = self.alloc_device(
                    (padded_out_n * 4) as u64, padded_out_n, DType::F32,
                )?;

                let gx = ((n + 63) / 64) as u32;
                let gy = ((padded_m + 15) / 16) as u32;
                self.dispatch_3buf(
                    "matmul_coop",
                    self.pipelines.matmul_coop_pipeline.as_ref().unwrap(),
                    self.pipelines.matmul_coop_layout.as_ref().unwrap(),
                    a, b, &padded_out, pbuf, pmem, params_size, gx, gy, gz,
                )?;

                // Return the padded buffer but with the original
                // logical element count. Downstream code only reads
                // m*n elements so the padded rows are invisible.
                return Ok(VulkanStorage {
                    inner: padded_out.inner,
                    elem_count: out_n,
                    dtype: DType::F32,
                });
            } else {
                // Fallback: software tiled matmul (no tensor cores).
                let gx = ((n + 63) / 64) as u32;
                let gy = ((m + 63) / 64) as u32;
                self.dispatch_3buf(
                    "matmul_tiled_bf16_b",
                    &self.pipelines.matmul_tiled_bf16_b_pipeline,
                    &self.pipelines.matmul_tiled_bf16_b_layout,
                    a, b, &out, pbuf, pmem, params_size, gx, gy, gz,
                )?;
            }
        } else if m < 32 {
            let gx = ((n + 63) / 64) as u32;
            let gy = ((m + 63) / 64) as u32;
            self.dispatch_3buf(
                "matmul",
                &self.pipelines.matmul_pipeline,
                &self.pipelines.matmul_layout,
                a, b, &out, pbuf, pmem, params_size, gx, gy, gz,
            )?;
        } else {
            let gx = ((n + 63) / 64) as u32;
            let gy = ((m + 63) / 64) as u32;
            self.dispatch_3buf(
                "matmul_tiled",
                &self.pipelines.matmul_tiled_pipeline,
                &self.pipelines.matmul_tiled_layout,
                a, b, &out, pbuf, pmem, params_size, gx, gy, gz,
            )?;
        }
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
        let (pbuf, pmem) = self.upload_params(&p)?;
        self.dispatch_2buf(
            "unary",
            &self.pipelines.unary_pipeline,
            &self.pipelines.unary_layout,
            a, &out, pbuf, pmem, 8, Self::workgroups(a.elem_count), 1, 1,
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
        let (pbuf, pmem) = self.upload_params(&p)?;
        self.dispatch_3buf(
            "binary",
            &self.pipelines.binary_pipeline,
            &self.pipelines.binary_layout,
            a, b, &out, pbuf, pmem, 8, Self::workgroups(a.elem_count), 1, 1,
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
        let (pbuf, pmem) = self.upload_params(&p)?;
        self.dispatch_2buf(
            "affine",
            &self.pipelines.affine_pipeline,
            &self.pipelines.affine_layout,
            a, &out, pbuf, pmem, 16, Self::workgroups(a.elem_count), 1, 1,
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
        a: &Self::Storage, layout: &Layout,
        dims: &[usize],
    ) -> fuel_core_types::Result<Self::Storage> {
        let op_id: u32 = match op {
            fuel_core_types::op::ReduceOp::Sum => 0,
            fuel_core_types::op::ReduceOp::Max => 1,
            fuel_core_types::op::ReduceOp::Min => 2,
            _ => fuel_core_types::bail!("VulkanBackend: unsupported reduce op"),
        };

        // Fast path 1: full reduction — every dim collapses to a scalar.
        let shape = layout.shape();
        let rank = shape.dims().len();
        if dims.len() == rank || dims.is_empty() {
            let out = self.alloc_device(4, 1, DType::F32)?;
            #[repr(C)] #[derive(Clone, Copy)]
            struct RParams { n: u32, op_id: u32 }
            let p = RParams { n: a.elem_count as u32, op_id };
            let (pbuf, pmem) = self.upload_params(&p)?;
            self.dispatch_2buf(
                "reduce",
                &self.pipelines.reduce_pipeline,
                &self.pipelines.reduce_layout,
                a, &out, pbuf, pmem, 8, 1, 1, 1,
            )?;
            return Ok(out);
        }

        // Fast path 2: single-dim reduction along the LAST dim. Covers
        // RMSNorm / LayerNorm / softmax prep — the hot path that was
        // hitting CPU fallback ~44× per Llama forward before this
        // kernel existed.
        if dims.len() == 1 && dims[0] == rank - 1 {
            let dims_slice = shape.dims();
            let n_cols = dims_slice[rank - 1];
            let n_rows: usize = dims_slice[..rank - 1].iter().product::<usize>().max(1);

            // Validate that the input storage is actually shaped the
            // way we're telling the shader to read it. A mismatch
            // here (e.g. storage.elem_count != n_rows*n_cols) would
            // cause the shader to read past the buffer → GPU hang or
            // DEVICE_LOST. Fail loudly in Rust instead.
            let expected_elems = n_rows
                .checked_mul(n_cols)
                .ok_or_else(|| fuel_core_types::Error::Msg(
                    "reduce_last_dim: n_rows * n_cols overflow".into()))?;
            if a.elem_count != expected_elems {
                fuel_core_types::bail!(
                    "reduce_last_dim: storage.elem_count ({}) != n_rows*n_cols ({}*{}={}); shape={:?}",
                    a.elem_count, n_rows, n_cols, expected_elems, dims_slice
                );
            }
            if a.dtype != DType::F32 {
                fuel_core_types::bail!(
                    "reduce_last_dim: input must be f32, got {:?}", a.dtype
                );
            }
            if n_rows == 0 || n_cols == 0 {
                fuel_core_types::bail!(
                    "reduce_last_dim: degenerate shape (n_rows={n_rows}, n_cols={n_cols})"
                );
            }

            let out_elems = n_rows;
            let out = self.alloc_device((out_elems * 4) as u64, out_elems, DType::F32)?;

            #[repr(C)] #[derive(Clone, Copy)]
            struct RLParams { n_rows: u32, n_cols: u32, op_id: u32, _pad: u32 }
            let p = RLParams {
                n_rows: n_rows as u32,
                n_cols: n_cols as u32,
                op_id,
                _pad: 0,
            };
            let (pbuf, pmem) = self.upload_params(&p)?;

            tracing::debug!(
                target: "vk_reduce_last_dim",
                n_rows, n_cols, op_id,
                input_bytes = a.byte_size(),
                output_bytes = out.byte_size(),
                "reduce_last_dim dispatch",
            );

            self.dispatch_2buf(
                "reduce_last_dim",
                &self.pipelines.reduce_last_dim_pipeline,
                &self.pipelines.reduce_last_dim_layout,
                a, &out, pbuf, pmem, 16, n_rows as u32, 1, 1,
            )?;
            return Ok(out);
        }

        // Any other dim combo: fall back to CPU. Rare; reducing along
        // middle / leading dims needs a strided kernel we haven't
        // written yet.
        fuel_core_types::bail!("VulkanBackend: reduce along non-last dim(s) {:?} not yet native", dims)
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
        let (pbuf, pmem) = self.upload_params(&p)?;
        self.dispatch_2buf(
            "softmax",
            &self.pipelines.softmax_pipeline,
            &self.pipelines.softmax_layout,
            a, &out, pbuf, pmem, 8, n_rows, 1, 1,
        )?;
        Ok(out)
    }

    fn add_assign_scaled(
        &self,
        dst: &mut Self::Storage,
        src: &Self::Storage,
        scale: f32,
    ) -> fuel_core_types::Result<()> {
        if dst.dtype != DType::F32 || src.dtype != DType::F32 {
            fuel_core_types::bail!(
                "VulkanBackend: add_assign_scaled requires f32, got dst={:?} src={:?}",
                dst.dtype, src.dtype,
            );
        }
        if dst.elem_count != src.elem_count {
            fuel_core_types::bail!(
                "VulkanBackend: add_assign_scaled shape mismatch: dst={} src={}",
                dst.elem_count, src.elem_count,
            );
        }
        let n = dst.elem_count;

        #[repr(C)] #[derive(Clone, Copy)]
        struct AasParams { n: u32, scale: f32 }
        let p = AasParams { n: n as u32, scale };
        let (pbuf, pmem) = self.upload_params(&p)?;

        // 2s1u layout: binding 0 = dst (read_write), 1 = src (read),
        // 2 = params (uniform).
        let desc = self.pipelines.allocate_desc(&self.pipelines.layout_2s1u).map_err(vk_err)?;
        desc.write_buffer(0, DescriptorType::STORAGE_BUFFER, dst.buffer(), 0, dst.byte_size());
        desc.write_buffer(1, DescriptorType::STORAGE_BUFFER, src.buffer(), 0, src.byte_size());
        desc.write_buffer(2, DescriptorType::UNIFORM_BUFFER, &pbuf, 0, 8);

        let pipeline = &self.pipelines.add_assign_scaled_pipeline;
        let layout = &self.pipelines.add_assign_scaled_layout;
        let groups = Self::workgroups(n);
        self.record_dispatch(
            "add_assign_scaled",
            vec![(pbuf, pmem)],
            Some(desc),
            |cmd, desc| {
                let desc = desc.expect("add_assign_scaled: descriptor missing");
                cmd.bind_compute_pipeline(pipeline);
                cmd.bind_compute_descriptor_sets(layout, 0, &[desc]);
                cmd.dispatch(groups, 1, 1);
                Ok(())
            },
        )
    }

    fn rms_norm_last_dim(
        &self, a: &Self::Storage, layout: &Layout, eps: f64,
    ) -> fuel_core_types::Result<Self::Storage> {
        if a.dtype != DType::F32 {
            fuel_core_types::bail!(
                "VulkanBackend: rms_norm_last_dim requires f32 input, got {:?}", a.dtype
            );
        }
        let shape = layout.shape();
        let dims = shape.dims();
        let n_cols = *dims.last().expect("rms_norm: empty shape");
        let n_rows = (a.elem_count / n_cols) as u32;
        let out = self.alloc_device(a.byte_size(), a.elem_count, a.dtype)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct RmsParams { n_rows: u32, n_cols: u32, eps: f32, _pad: u32 }
        let p = RmsParams {
            n_rows,
            n_cols: n_cols as u32,
            eps: eps as f32,
            _pad: 0,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;
        self.dispatch_2buf(
            "rms_norm_last_dim",
            &self.pipelines.rms_norm_last_dim_pipeline,
            &self.pipelines.rms_norm_last_dim_layout,
            a, &out, pbuf, pmem, 16, n_rows, 1, 1,
        )?;
        Ok(out)
    }

    fn layer_norm_last_dim_backward(
        &self,
        x: &Self::Storage,
        upstream: &Self::Storage,
        x_layout: &Layout,
        _up_layout: &Layout,
        eps: f64,
    ) -> fuel_core_types::Result<Self::Storage> {
        if x.dtype != DType::F32 || upstream.dtype != DType::F32 {
            fuel_core_types::bail!("VulkanBackend: layer_norm_last_dim_backward requires f32");
        }
        let dims = x_layout.shape().dims();
        if dims.is_empty() {
            fuel_core_types::bail!("layer_norm_last_dim_backward: rank >= 1 required");
        }
        let n_cols = *dims.last().unwrap();
        let n_rows = (x.elem_count / n_cols) as u32;
        let out = self.alloc_device(x.byte_size(), x.elem_count, x.dtype)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct LnBwdParams { n_rows: u32, n_cols: u32, eps: f32, _pad: u32 }
        let p = LnBwdParams {
            n_rows,
            n_cols: n_cols as u32,
            eps: eps as f32,
            _pad: 0,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;
        self.dispatch_3buf(
            "layer_norm_last_dim_backward",
            &self.pipelines.layer_norm_last_dim_backward_pipeline,
            &self.pipelines.layer_norm_last_dim_backward_layout,
            x, upstream, &out, pbuf, pmem,
            std::mem::size_of::<LnBwdParams>() as u64,
            n_rows, 1, 1,
        )?;
        Ok(out)
    }

    fn softmax_last_dim_backward(
        &self,
        y: &Self::Storage,
        upstream: &Self::Storage,
        y_layout: &Layout,
        _up_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        if y.dtype != DType::F32 || upstream.dtype != DType::F32 {
            fuel_core_types::bail!("VulkanBackend: softmax_last_dim_backward requires f32");
        }
        let dims = y_layout.shape().dims();
        if dims.is_empty() {
            fuel_core_types::bail!("softmax_last_dim_backward: rank >= 1 required");
        }
        let n_cols = *dims.last().unwrap();
        let n_rows = (y.elem_count / n_cols) as u32;
        let out = self.alloc_device(y.byte_size(), y.elem_count, y.dtype)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct SoftBwdParams { n_rows: u32, n_cols: u32 }
        let p = SoftBwdParams { n_rows, n_cols: n_cols as u32 };
        let (pbuf, pmem) = self.upload_params(&p)?;
        self.dispatch_3buf(
            "softmax_last_dim_backward",
            &self.pipelines.softmax_last_dim_backward_pipeline,
            &self.pipelines.softmax_last_dim_backward_layout,
            y, upstream, &out, pbuf, pmem,
            std::mem::size_of::<SoftBwdParams>() as u64,
            n_rows, 1, 1,
        )?;
        Ok(out)
    }

    fn rms_norm_last_dim_backward(
        &self,
        x: &Self::Storage,
        upstream: &Self::Storage,
        x_layout: &Layout,
        _up_layout: &Layout,
        eps: f64,
    ) -> fuel_core_types::Result<Self::Storage> {
        if x.dtype != DType::F32 || upstream.dtype != DType::F32 {
            fuel_core_types::bail!("VulkanBackend: rms_norm_last_dim_backward requires f32");
        }
        let shape = x_layout.shape();
        let dims = shape.dims();
        if dims.is_empty() {
            fuel_core_types::bail!("rms_norm_last_dim_backward: rank >= 1 required");
        }
        let n_cols = *dims.last().unwrap();
        let n_rows = (x.elem_count / n_cols) as u32;
        let out = self.alloc_device(x.byte_size(), x.elem_count, x.dtype)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct RmsBwdParams { n_rows: u32, n_cols: u32, eps: f32, _pad: u32 }
        let p = RmsBwdParams {
            n_rows,
            n_cols: n_cols as u32,
            eps: eps as f32,
            _pad: 0,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;
        self.dispatch_3buf(
            "rms_norm_last_dim_backward",
            &self.pipelines.rms_norm_last_dim_backward_pipeline,
            &self.pipelines.rms_norm_last_dim_backward_layout,
            x, upstream, &out, pbuf, pmem,
            std::mem::size_of::<RmsBwdParams>() as u64,
            n_rows, 1, 1,
        )?;
        Ok(out)
    }

    fn concat_along_dim(
        &self,
        a: &Self::Storage,
        b: &Self::Storage,
        dim: usize,
        a_shape: &Shape,
        b_shape: &Shape,
    ) -> fuel_core_types::Result<Self::Storage> {
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            fuel_core_types::bail!("VulkanBackend: concat_along_dim requires f32");
        }
        let a_dims = a_shape.dims();
        let b_dims = b_shape.dims();
        if a_dims.len() != b_dims.len() || dim >= a_dims.len() {
            fuel_core_types::bail!("concat_along_dim: rank/dim mismatch");
        }
        for (i, (&da, &db)) in a_dims.iter().zip(b_dims.iter()).enumerate() {
            if i != dim && da != db {
                fuel_core_types::bail!("concat_along_dim: non-concat dims disagree");
            }
        }
        let a_dim = a_dims[dim];
        let b_dim = b_dims[dim];
        let outer: usize = a_dims[..dim].iter().product::<usize>().max(1);
        let inner: usize = a_dims[dim + 1..].iter().product::<usize>().max(1);
        let out_elems = outer * (a_dim + b_dim) * inner;
        let out = self.alloc_device((out_elems * 4) as u64, out_elems, DType::F32)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct CParams { outer: u32, a_dim: u32, b_dim: u32, inner: u32, total: u32, _p0: u32, _p1: u32, _p2: u32 }
        let p = CParams {
            outer: outer as u32,
            a_dim: a_dim as u32,
            b_dim: b_dim as u32,
            inner: inner as u32,
            total: out_elems as u32,
            _p0: 0, _p1: 0, _p2: 0,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;

        let groups = ((out_elems as u32 + 63) / 64).max(1);
        self.dispatch_3buf(
            "concat_along_dim",
            &self.pipelines.concat_along_dim_pipeline,
            &self.pipelines.concat_along_dim_layout,
            a, b, &out, pbuf, pmem, std::mem::size_of::<CParams>() as u64, groups, 1, 1,
        )?;
        Ok(out)
    }

    fn rope(
        &self,
        x: &Self::Storage,
        cos: &Self::Storage,
        sin: &Self::Storage,
        x_layout: &Layout,
        _cos_layout: &Layout,
        _sin_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        if x.dtype != DType::F32 || cos.dtype != DType::F32 || sin.dtype != DType::F32 {
            fuel_core_types::bail!("VulkanBackend: rope requires f32 inputs");
        }
        let dims = x_layout.shape().dims();
        let rank = dims.len();
        if rank < 2 {
            fuel_core_types::bail!("VulkanBackend: rope requires rank >= 2, got {dims:?}");
        }
        let seq = dims[rank - 2] as u32;
        let head_dim = dims[rank - 1] as u32;
        if head_dim % 2 != 0 {
            fuel_core_types::bail!("VulkanBackend: rope head_dim must be even, got {head_dim}");
        }
        let outer: u32 = dims[..rank - 2].iter().product::<usize>().max(1) as u32;
        let half = head_dim / 2;
        let total = outer * seq * half;

        let out = self.alloc_device(x.byte_size(), x.elem_count, x.dtype)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct RopeParams { outer: u32, seq: u32, head_dim: u32, total: u32 }
        let p = RopeParams { outer, seq, head_dim, total };
        let (pbuf, pmem) = self.upload_params(&p)?;

        let desc = self.pipelines.allocate_desc(&self.pipelines.layout_4s1u).map_err(vk_err)?;
        desc.write_buffer(0, DescriptorType::STORAGE_BUFFER, x.buffer(), 0, x.byte_size());
        desc.write_buffer(1, DescriptorType::STORAGE_BUFFER, cos.buffer(), 0, cos.byte_size());
        desc.write_buffer(2, DescriptorType::STORAGE_BUFFER, sin.buffer(), 0, sin.byte_size());
        desc.write_buffer(3, DescriptorType::STORAGE_BUFFER, out.buffer(), 0, out.byte_size());
        desc.write_buffer(4, DescriptorType::UNIFORM_BUFFER, &pbuf, 0, std::mem::size_of::<RopeParams>() as u64);

        let pipeline = &self.pipelines.rope_pipeline;
        let pipe_layout = &self.pipelines.rope_layout;
        let groups = ((total + 63) / 64).max(1);
        self.record_dispatch(
            "rope",
            vec![(pbuf, pmem)],
            Some(desc),
            |cmd, desc| {
                let desc = desc.expect("rope: descriptor set missing");
                cmd.bind_compute_pipeline(pipeline);
                cmd.bind_compute_descriptor_sets(pipe_layout, 0, &[desc]);
                cmd.dispatch(groups, 1, 1);
                Ok(())
            },
        )?;
        Ok(out)
    }

    fn index_select(
        &self, src: &Self::Storage, ids: &Self::Storage,
        src_l: &Layout, ids_l: &Layout, dim: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        if src.dtype != DType::F32 {
            fuel_core_types::bail!(
                "VulkanBackend: index_select requires f32 source, got {:?}", src.dtype
            );
        }
        if ids.dtype != DType::U32 {
            fuel_core_types::bail!(
                "VulkanBackend: index_select requires u32 ids, got {:?}", ids.dtype
            );
        }
        let src_dims = src_l.shape().dims();
        let rank = src_dims.len();
        if dim >= rank {
            fuel_core_types::bail!(
                "VulkanBackend: index_select dim {dim} out of range for rank {rank}"
            );
        }

        let outer: usize = src_dims[..dim].iter().product::<usize>().max(1);
        let axis_in = src_dims[dim];
        let inner: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let axis_out = ids_l.shape().elem_count();
        let out_size = outer * axis_out * inner;
        let out = self.alloc_device((out_size * 4) as u64, out_size, DType::F32)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct IParams {
            out_size: u32,
            outer: u32,
            axis_out: u32,
            inner: u32,
            axis_in: u32,
            _pad0: u32, _pad1: u32, _pad2: u32,
        }
        let p = IParams {
            out_size: out_size as u32,
            outer: outer as u32,
            axis_out: axis_out as u32,
            inner: inner as u32,
            axis_in: axis_in as u32,
            _pad0: 0, _pad1: 0, _pad2: 0,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;

        // Bind src, ids, out, params. Layout is 3s1u, same as matmul.
        let desc = self.pipelines.allocate_desc(&self.pipelines.layout_3s1u).map_err(vk_err)?;
        desc.write_buffer(0, DescriptorType::STORAGE_BUFFER, src.buffer(), 0, src.byte_size());
        desc.write_buffer(1, DescriptorType::STORAGE_BUFFER, ids.buffer(), 0, ids.byte_size());
        desc.write_buffer(2, DescriptorType::STORAGE_BUFFER, out.buffer(), 0, out.byte_size());
        desc.write_buffer(3, DescriptorType::UNIFORM_BUFFER, &pbuf, 0, std::mem::size_of::<IParams>() as u64);

        let pipeline = &self.pipelines.index_select_pipeline;
        let layout = &self.pipelines.index_select_layout;
        let groups = Self::workgroups(out_size);
        self.record_dispatch(
            "index_select",
            vec![(pbuf, pmem)],
            Some(desc),
            |cmd, desc| {
                let desc = desc.expect("index_select: descriptor missing");
                cmd.bind_compute_pipeline(pipeline);
                cmd.bind_compute_descriptor_sets(layout, 0, &[desc]);
                cmd.dispatch(groups, 1, 1);
                Ok(())
            },
        )?;
        Ok(out)
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
unsafe fn as_bytes<T: Sized>(p: &T) -> &[u8] { unsafe {
    std::slice::from_raw_parts(p as *const T as *const u8, std::mem::size_of::<T>())
}}

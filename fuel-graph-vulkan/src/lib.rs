//! Vulkan GPU executor for fuel-graph computation graphs.
//!
//! Uses Vulkane for Vulkan device management and dispatches compute
//! ops through WGSL shaders compiled to SPIR-V via naga. Third
//! backend for fuel's generic `GraphExecutor<B>`.

pub mod byte_storage;
pub mod pipelines;
pub mod probe;
mod recorder;
pub mod residency;

pub use byte_storage::VulkanStorageBytes;

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
    buffer: Option<Buffer>,
    allocation: Option<Allocation>,
    byte_size: u64,
    /// If set, the buffer is returned to this pool on Drop instead
    /// of being freed. This is how the buffer recycler works: every
    /// buffer created via `alloc_device` gets a back-reference to
    /// the pool. When the Arc drops to 0 → VulkanBuffer::drop fires
    /// → buffer goes back to the pool for reuse.
    ///
    /// Keyed by byte_size → stack of buffers of that exact size. The
    /// BTreeMap enables O(log n) best-fit lookup (smallest size ≥
    /// requested) without a linear scan.
    recycle_pool: Option<std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<u64, Vec<(Buffer, Allocation)>>>>>,
}

impl VulkanBuffer {
    pub fn buffer(&self) -> &Buffer { self.buffer.as_ref().unwrap() }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        let buf = self.buffer.take();
        let alloc = self.allocation.take();
        if let (Some(b), Some(a)) = (buf, alloc) {
            if let Some(pool) = &self.recycle_pool {
                // Return to pool for reuse.
                if let Ok(mut p) = pool.lock() {
                    p.entry(self.byte_size).or_default().push((b, a));
                    return;
                }
            }
            // No pool or lock failed — normal drop.
            drop(a);
            drop(b);
        }
    }
}

/// Residency tier for a Vulkan-allocated tensor. Today every buffer
/// is [`Tier::OnDevice`]; P5 (tiered residency) will introduce
/// [`Tier::OnHost`] for tensors spilled to a mmap-backed host file
/// when VRAM is exhausted.
///
/// The field is a tag only — no eviction or fault-back logic lives
/// in the allocator yet. When those land the allocator will consult
/// this tag to decide whether a read/write needs to stage through a
/// host-visible path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Tier {
    /// Buffer is live in VRAM on the Vulkan device.
    #[default]
    OnDevice,
    /// Buffer has been evicted; backing is a mmap'd host file that
    /// the OS pages between RAM and disk. Reads require fault-back
    /// to VRAM before any compute op. Not yet emitted by the
    /// allocator — the variant exists so downstream code can pattern
    /// match on it now.
    OnHost,
}

/// Backing for a [`VulkanStorage`] — either live VRAM, or evicted
/// bytes in the host-side [`residency::ResidencyFile`] when VRAM is
/// tight. An evicted storage can only be read via `fault_back`; ops
/// that require a device buffer will panic cleanly if handed one.
pub enum StorageBacking {
    Device(std::sync::Arc<VulkanBuffer>),
    Host {
        file: std::sync::Arc<residency::ResidencyFile>,
        slot: residency::Slot,
    },
}

/// Vulkan storage: backing (device or host-evicted) + per-view metadata.
pub struct VulkanStorage {
    backing: StorageBacking,
    pub elem_count: usize,
    pub dtype: DType,
    /// Current residency. Tracks the [`StorageBacking`] variant and
    /// stays consistent with it. Set automatically by allocator /
    /// eviction paths.
    pub tier: Tier,
}

impl VulkanStorage {
    /// Device buffer. Panics if the storage has been evicted to host —
    /// callers that can handle both tiers should use [`Self::buffer_opt`].
    pub fn buffer(&self) -> &Buffer {
        match &self.backing {
            StorageBacking::Device(b) => b.buffer(),
            StorageBacking::Host { .. } => panic!(
                "VulkanStorage::buffer called on host-backed storage; \
                 fault it back to VRAM first via VulkanBackend::fault_back"
            ),
        }
    }

    /// Device buffer if on-device, None if evicted to host.
    pub fn buffer_opt(&self) -> Option<&Buffer> {
        match &self.backing {
            StorageBacking::Device(b) => Some(b.buffer()),
            StorageBacking::Host { .. } => None,
        }
    }

    /// Access the backing for code that needs to distinguish tiers
    /// (the eviction path, future LRU tracker). External callers
    /// generally should use [`Self::tier`] + [`Self::buffer_opt`].
    pub fn backing(&self) -> &StorageBacking { &self.backing }

    /// Arc clone of the device buffer, for refcount-sharing zero-copy
    /// views. Returns None for host-backed storages (zero-copy doesn't
    /// apply — they'd need a fault-back first).
    pub fn device_buffer_arc(&self) -> Option<std::sync::Arc<VulkanBuffer>> {
        match &self.backing {
            StorageBacking::Device(b) => Some(std::sync::Arc::clone(b)),
            StorageBacking::Host { .. } => None,
        }
    }

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
    /// Recycled buffer pool. Buffers returned here via VulkanBuffer::Drop
    /// are reused by alloc_device before allocating fresh from VMA.
    /// BTreeMap<byte_size, stack-of-free-buffers-of-that-size>. Enables
    /// O(log n) best-fit lookup via `range(size..).next()`.
    buffer_pool: std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<u64, Vec<(Buffer, Allocation)>>>>,
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

        let buffer_pool = std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::new()));
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
            buffer_pool,
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

    /// Phase 7.5 A4 substrate alloc. Allocates `byte_count` bytes of
    /// device-local storage and wraps them in a fresh
    /// `VulkanStorageBytes`. No initialization — caller is responsible
    /// for filling via [`Self::upload_bytes`] or via a kernel write
    /// before reading. Mirrors the alloc shape on CUDA / CPU; the
    /// per-op kernel migration uses this for output allocation.
    pub fn alloc_bytes(&self, byte_count: usize) -> fuel_core_types::Result<VulkanStorageBytes> {
        let size = (byte_count as u64).max(1);
        let _span = debug_span!("vk_alloc_bytes", bytes = byte_count).entered();
        let (gpu_buf, gpu_alloc) = self
            .allocator
            .create_buffer(
                BufferCreateInfo {
                    size,
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
        Ok(VulkanStorageBytes::from_device(
            std::sync::Arc::new(VulkanBuffer {
                buffer: Some(gpu_buf),
                allocation: Some(gpu_alloc),
                byte_size: size,
                recycle_pool: Some(self.buffer_pool.clone()),
            }),
            byte_count,
        ))
    }

    /// Phase 7.5 A4 substrate H2D. Stages a host byte slice into a
    /// fresh device-local `VulkanStorageBytes`. The staging buffer
    /// is a host-visible mapped sub-allocation; the device copy is
    /// submitted via `queue.one_shot` which fences before returning,
    /// so the result is observable to subsequent ops.
    pub fn upload_bytes(&self, src: &[u8]) -> fuel_core_types::Result<VulkanStorageBytes> {
        let byte_size = src.len() as u64;
        let _span = debug_span!("vk_upload_bytes", bytes = byte_size).entered();
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
        if !src.is_empty() {
            let mapped = staging_alloc
                .mapped_ptr()
                .ok_or_else(|| fuel_core_types::Error::Msg(
                    "upload_bytes: staging alloc not mapped".into()))?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    mapped as *mut u8,
                    src.len(),
                );
            }
        }
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
        drop(staging_buf);
        drop(staging_alloc);
        Ok(VulkanStorageBytes::from_device(
            std::sync::Arc::new(VulkanBuffer {
                buffer: Some(gpu_buf),
                allocation: Some(gpu_alloc),
                byte_size: byte_size.max(1),
                recycle_pool: Some(self.buffer_pool.clone()),
            }),
            src.len(),
        ))
    }

    /// Phase 7.5 A4 substrate D2H. Reads a `VulkanStorageBytes`'s
    /// bytes back to host as a fresh `Vec<u8>`. Flushes any pending
    /// async ops first, then runs a one-shot device→staging copy
    /// and reads through the staging buffer's mapped pointer.
    /// Returns an error if the storage is currently host-evicted
    /// (caller must fault-back first via the residency machinery).
    pub fn download_bytes(
        &self,
        storage: &VulkanStorageBytes,
    ) -> fuel_core_types::Result<Vec<u8>> {
        let byte_size = storage.len_bytes() as u64;
        let _span = info_span!("vk_download_bytes", bytes = byte_size).entered();
        let buffer = storage.buffer_opt().ok_or_else(|| {
            fuel_core_types::Error::Msg(
                "download_bytes: storage is host-evicted; \
                 fault back via residency machinery before reading".into(),
            )
        })?;
        self.flush_pending()?;
        let (staging_buf, staging_alloc) = self
            .allocator
            .create_buffer(
                BufferCreateInfo {
                    size: byte_size.max(1),
                    usage: BufferUsage::TRANSFER_DST,
                },
                AllocationCreateInfo {
                    usage: AllocationUsage::HostVisible,
                    mapped: true,
                    ..Default::default()
                },
            )
            .map_err(vk_err)?;
        self.queue
            .one_shot(&self.device, self.queue_family, |cmd| {
                cmd.copy_buffer(
                    buffer,
                    &staging_buf,
                    &[BufferCopy { src_offset: 0, dst_offset: 0, size: byte_size.max(1) }],
                );
                Ok(())
            })
            .map_err(vk_err)?;
        let mut out = vec![0_u8; storage.len_bytes()];
        if !out.is_empty() {
            let mapped = staging_alloc
                .mapped_ptr()
                .ok_or_else(|| fuel_core_types::Error::Msg(
                    "download_bytes: staging alloc not mapped".into()))?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mapped as *const u8,
                    out.as_mut_ptr(),
                    out.len(),
                );
            }
        }
        drop(staging_buf);
        drop(staging_alloc);
        Ok(out)
    }

    pub fn upload_slice<T: Copy + 'static>(
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
            backing: StorageBacking::Device(std::sync::Arc::new(VulkanBuffer {
                buffer: Some(gpu_buf),
                allocation: Some(gpu_alloc),
                byte_size: byte_size.max(1),
                recycle_pool: Some(self.buffer_pool.clone()),
            })),
            elem_count: data.len(),
            dtype,
            tier: Tier::OnDevice,
        })
    }

    fn download_slice<T: Copy + Default + 'static>(
        &self, storage: &VulkanStorage,
    ) -> fuel_core_types::Result<Vec<T>> {
        let byte_size = storage.byte_size();
        let n = storage.elem_count;
        let pending = self.recorder.borrow().batch_count;
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
        let size = byte_size.max(1);
        // Best-fit recycle via BTreeMap: smallest pooled size ≥ requested,
        // capped at 2× to avoid wasting VRAM on oversized leftovers.
        // Three eviction levers keep the pool bounded on long generations:
        //   1. `MAX_BUCKETS`: cap distinct size buckets (evict smallest)
        //   2. `MAX_PER_BUCKET`: cap duplicate buffers in a single bucket
        //      (matters for KV-cache where N layers × 2 (K+V) buffers
        //      all arrive at the same size each step)
        //   3. `MAX_POOL_BYTES`: total-bytes cap as a backstop (evict
        //      smallest sizes until under), so the pool can never hoard
        //      more VRAM than needed
        const MAX_BUCKETS: usize = 64;
        const MAX_PER_BUCKET: usize = 4;
        const MAX_POOL_BYTES: u64 = 512 * 1024 * 1024; // 512 MB cap
        let recycled = {
            let mut pool = self.buffer_pool.lock().unwrap();
            // O(log n) best-fit: first bucket in [size, size*2].
            let found_size = pool
                .range(size..=size.saturating_mul(2))
                .next()
                .map(|(&k, _)| k);
            let picked = found_size.and_then(|k| {
                let vec = pool.get_mut(&k).unwrap();
                let item = vec.pop();
                if vec.is_empty() { pool.remove(&k); }
                item
            });
            // 1. Bucket-count cap: drop smallest sizes until ≤ MAX_BUCKETS.
            while pool.len() > MAX_BUCKETS {
                let smallest = *pool.keys().next().unwrap();
                pool.remove(&smallest);
            }
            // 2. Per-bucket depth cap: for each bucket, keep at most
            //    MAX_PER_BUCKET buffers. Extras are dropped (VMA frees).
            //    Kept is the END of the Vec (most recent pushes, in case
            //    sizes drift over time).
            for (_, vec) in pool.iter_mut() {
                if vec.len() > MAX_PER_BUCKET {
                    let drop_count = vec.len() - MAX_PER_BUCKET;
                    vec.drain(0..drop_count);
                }
            }
            // 3. Total-bytes backstop: if pool > MAX_POOL_BYTES, evict
            //    smallest-size buckets first (they're typically stale).
            let mut total_bytes: u64 = pool.iter()
                .map(|(&sz, v)| sz * v.len() as u64).sum();
            while total_bytes > MAX_POOL_BYTES {
                let smallest = match pool.keys().next() {
                    Some(&k) => k,
                    None => break,
                };
                let vec = pool.remove(&smallest).unwrap();
                total_bytes = total_bytes.saturating_sub(smallest * vec.len() as u64);
            }
            picked
        };
        let (buffer, allocation) = if let Some((b, a)) = recycled {
            (b, a)
        } else {
            self.allocator.create_buffer(
                BufferCreateInfo {
                    size,
                    usage: BufferUsage::STORAGE_BUFFER
                        | BufferUsage::TRANSFER_SRC
                        | BufferUsage::TRANSFER_DST,
                },
                AllocationCreateInfo {
                    usage: AllocationUsage::DeviceLocal,
                    ..Default::default()
                },
            ).map_err(vk_err)?
        };
        Ok(VulkanStorage {
            backing: StorageBacking::Device(std::sync::Arc::new(VulkanBuffer {
                buffer: Some(buffer),
                allocation: Some(allocation),
                byte_size: size,
                recycle_pool: Some(self.buffer_pool.clone()),
            })),
            elem_count: n,
            dtype,
            tier: Tier::OnDevice,
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
    /// Record a compute dispatch into the current batch command
    /// buffer. Pipeline barrier + bind + dispatch are recorded via
    /// raw Vulkan calls (bypassing vulkane's RAII CommandBufferRecording
    /// so the CB stays in recording state across calls). The batch is
    /// submitted in one shot at flush time, eliminating the per-op
    /// vkQueueSubmit overhead that was the dominant host-side cost.
    fn record_dispatch_batched(
        &self,
        op_name: &'static str,
        pipeline: &ComputePipeline,
        pipe_layout: &PipelineLayout,
        desc: DescriptorSet,
        groups: (u32, u32, u32),
        transient_buffers: Vec<(Buffer, Allocation)>,
        read_bufs: &[u64],
        write_bufs: &[u64],
    ) -> fuel_core_types::Result<()> {
        let t0 = Instant::now();

        // Auto-flush if the batch is getting large (TDR safety).
        if self.recorder.borrow().should_flush() {
            self.flush_pending()?;
        }

        self.recorder
            .borrow_mut()
            .record_batch_dispatch(
                &self.device,
                pipeline,
                pipe_layout,
                desc,
                groups,
                transient_buffers,
                read_bufs,
                write_bufs,
            )
            .map_err(vk_err)?;

        self.op_stats.record(op_name, t0.elapsed());
        Ok(())
    }

    /// Flush the current batch: end recording, submit the single CB,
    /// wait for the GPU, drop transient resources, retire descriptor
    /// pools.
    fn flush_pending(&self) -> fuel_core_types::Result<()> {
        let batch_count = self.recorder.borrow().batch_count;
        if batch_count == 0 { return Ok(()); }
        let _span = info_span!("vk_flush_batch", batch_count).entered();
        self.recorder
            .borrow_mut()
            .flush_batch(&self.device, &self.queue, self.queue_family)
            .map_err(vk_err)?;
        self.pipelines.retire_pools_post_drain();
        Ok(())
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
        let rb = [input.buffer().raw() as u64];
        let wb = [output.buffer().raw() as u64];
        self.record_dispatch_batched(
            op_name, pipeline, pipe_layout, desc,
            (groups_x, groups_y, groups_z),
            vec![(params_buf, params_alloc)],
            &rb, &wb,
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
        let rb = [a.buffer().raw() as u64, b.buffer().raw() as u64];
        let wb = [output.buffer().raw() as u64];
        self.record_dispatch_batched(
            op_name, pipeline, pipe_layout, desc,
            (groups_x, groups_y, groups_z),
            vec![(params_buf, params_alloc)],
            &rb, &wb,
        )
    }

    fn workgroups(n: usize) -> u32 {
        ((n + 255) / 256) as u32
    }

    // -- quantized weight dequantization ---------------------------------------

    /// Dequantize a raw Q4_0 blob (18-byte blocks, 32 elements per block)
    /// directly on the GPU to an f32 storage buffer. The input is the
    /// unmodified block byte stream as stored in GGUF files; this
    /// function uploads it once to a temporary device buffer and
    /// dispatches the dequant kernel. Caller controls `n_blocks` and
    /// the resulting `n_elements = n_blocks * 32`.
    pub fn dequantize_q4_0(
        &self,
        blocks: &[u8],
        n_blocks: usize,
    ) -> fuel_core_types::Result<VulkanStorage> {
        const BYTES_PER_BLOCK: usize = 18;
        const BLCK_SIZE: usize = 32;
        if blocks.len() != n_blocks * BYTES_PER_BLOCK {
            fuel_core_types::bail!(
                "dequantize_q4_0: expected {} bytes for {n_blocks} blocks, got {}",
                n_blocks * BYTES_PER_BLOCK, blocks.len(),
            );
        }
        let n_elements = n_blocks * BLCK_SIZE;
        let out = self.alloc_device((n_elements * 4) as u64, n_elements, DType::F32)?;
        // Upload Q4_0 bytes as-is to a device storage buffer.
        let input = self.upload_slice(blocks, DType::U32)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct Q4Params { n_blocks: u32, out_elements: u32, _pad0: u32, _pad1: u32 }
        let p = Q4Params {
            n_blocks: n_blocks as u32,
            out_elements: n_elements as u32,
            _pad0: 0, _pad1: 0,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;
        let total_pairs = n_blocks * (BLCK_SIZE / 2);
        self.dispatch_2buf(
            "dequant_q4_0",
            &self.pipelines.dequant_q4_0_pipeline,
            &self.pipelines.dequant_q4_0_layout,
            &input, &out, pbuf, pmem,
            std::mem::size_of::<Q4Params>() as u64,
            Self::workgroups(total_pairs), 1, 1,
        )?;
        Ok(out)
    }

    /// Dequantize a raw Q8_0 blob (34-byte blocks, 32 elements per block)
    /// directly on the GPU to an f32 storage buffer.
    pub fn dequantize_q8_0(
        &self,
        blocks: &[u8],
        n_blocks: usize,
    ) -> fuel_core_types::Result<VulkanStorage> {
        const BYTES_PER_BLOCK: usize = 34;
        if blocks.len() != n_blocks * BYTES_PER_BLOCK {
            fuel_core_types::bail!(
                "dequantize_q8_0: expected {} bytes for {n_blocks} blocks, got {}",
                n_blocks * BYTES_PER_BLOCK, blocks.len(),
            );
        }
        let input = self.upload_slice(blocks, DType::U32)?;
        self.dequantize_q8_0_from_storage(&input, n_blocks)
    }

    /// Same as `dequantize_q8_0` but takes an already-on-device U32-typed
    /// block stream. Used by the KV-cache read path where blocks are
    /// produced by `quantize_q8_0` and never leave the GPU.
    /// Total VRAM budget across device-local heaps, in bytes. Returns
    /// `0` if `VK_EXT_memory_budget` isn't supported on this device
    /// (old drivers, unusual configurations). Use
    /// [`Self::has_memory_budget_support`] to distinguish "no budget"
    /// from "no query support."
    ///
    /// Paired with [`Self::vram_used`] for the scheduler's
    /// budget-aware residency planning:
    ///
    /// ```ignore
    /// let frac = backend.vram_used() as f64 / backend.vram_budget() as f64;
    /// if frac > 0.85 {
    ///     scheduler.evict_cold_tensors();
    /// }
    /// ```
    pub fn vram_budget(&self) -> u64 {
        self.allocator.vram_budget()
    }

    /// Total VRAM currently in use across device-local heaps. Driver
    /// estimate includes this process, other processes, and driver
    /// internals. Returns `0` if unsupported.
    pub fn vram_used(&self) -> u64 {
        self.allocator.vram_used()
    }

    /// True iff the `VK_EXT_memory_budget` extension is loaded and
    /// functional. When false, [`Self::vram_budget`] /
    /// [`Self::vram_used`] both return `0` and schedulers should
    /// fall back to conservative sizing heuristics.
    pub fn has_memory_budget_support(&self) -> bool {
        self.allocator.has_memory_budget_support()
    }

    /// Projected fit-check for an allocation of `size` bytes against
    /// a specific memory type. Predictive: fires the allocator's
    /// pressure callbacks with any thresholds the projection would
    /// cross, *before* an actual allocation is attempted.
    ///
    /// Use from the residency planner to decide whether the next
    /// scheduled op needs an explicit evict beforehand.
    pub fn would_fit(&self, size: u64, memory_type_index: u32) -> vulkane::safe::FitStatus {
        self.allocator.would_fit(size, memory_type_index)
    }

    /// Register a VRAM-pressure callback. Fires when usage crosses
    /// `threshold` on any device-local heap, or when predicted usage
    /// (via [`Self::would_fit`]) would cross. `hysteresis` is the
    /// relief gap below which [`PressureKind::Relieved`] fires —
    /// prevents rapid re-fire as usage oscillates. Typical values:
    /// `threshold=0.85, hysteresis=0.05` (fire at 85 %, relieve at 80 %).
    ///
    /// The callback runs on whatever thread freed memory or called
    /// `would_fit`. Vulkane releases its internal locks before firing,
    /// so the callback may freely call back into the allocator (e.g.,
    /// to trigger scheduler-driven eviction).
    ///
    /// Returns an id used to unregister via
    /// [`Self::unregister_vram_pressure_callback`].
    pub fn register_vram_pressure_callback<F>(
        &self,
        threshold: f64,
        hysteresis: f64,
        callback: F,
    ) -> vulkane::safe::PressureCallbackId
    where
        F: Fn(vulkane::safe::PressureEvent) + Send + Sync + 'static,
    {
        self.allocator.register_pressure_callback(threshold, hysteresis, callback)
    }

    /// Unregister a previously-registered pressure callback. Returns
    /// `true` if found and removed.
    pub fn unregister_vram_pressure_callback(&self, id: vulkane::safe::PressureCallbackId) -> bool {
        self.allocator.unregister_pressure_callback(id)
    }

    /// Probe the device-local memory-type index this backend's allocator
    /// uses for regular device allocations. Does a tiny throwaway alloc
    /// to learn which memory type `AllocationUsage::DeviceLocal` resolves
    /// to on this physical device. Callers (defrag pool, pressure-callback
    /// setup) generally invoke this once at init time and cache the index.
    pub fn device_local_memory_type_index(&self) -> fuel_core_types::Result<u32> {
        let (buf, alloc) = self.allocator.create_buffer(
            BufferCreateInfo { size: 1, usage: BufferUsage::STORAGE_BUFFER },
            AllocationCreateInfo { usage: AllocationUsage::DeviceLocal, ..Default::default() },
        ).map_err(vk_err)?;
        let idx = alloc.memory_type_index();
        drop(buf);
        drop(alloc);
        Ok(idx)
    }

    /// Create a dedicated [`FreeList`][vulkane::safe::AllocationStrategy::FreeList]
    /// custom pool on the device-local memory type, suitable for holding
    /// long-lived weight tensors that need defragmentation support.
    ///
    /// Returns a [`PoolHandle`][vulkane::safe::PoolHandle] that can later be
    /// passed to [`Self::build_defrag_plan`] / [`Self::apply_defrag_plan`]
    /// or to [`Self::destroy_weight_pool`].
    ///
    /// `block_size` is the per-block size in bytes (0 = allocator default,
    /// typically 256 MiB on ≥ 4 GiB heaps). `max_blocks` caps pool growth
    /// (0 = unlimited).
    ///
    /// ## NOT YET integrated with weight allocation
    ///
    /// Today's [`alloc_device`][Self::alloc_device] path routes through
    /// the default (per-memory-type) pool, not any custom pool. Wiring
    /// weights through this custom pool — so defrag actually moves them
    /// — is a follow-up. This method exposes the primitive so that
    /// follow-up has a stable handle to work with; calling it today
    /// allocates zero bytes until `alloc_device_weight` (TODO) hands
    /// allocations to this pool.
    pub fn create_weight_pool(
        &self,
        block_size: u64,
        max_blocks: u32,
    ) -> fuel_core_types::Result<vulkane::safe::PoolHandle> {
        let mt = self.device_local_memory_type_index()?;
        self.allocator.create_pool(vulkane::safe::PoolCreateInfo {
            memory_type_index: mt,
            strategy: vulkane::safe::AllocationStrategy::FreeList,
            block_size,
            max_block_count: max_blocks,
        }).map_err(vk_err)
    }

    /// Destroy a pool previously created with [`Self::create_weight_pool`].
    /// The caller must ensure no live allocations from this pool are in use.
    pub fn destroy_weight_pool(&self, handle: vulkane::safe::PoolHandle) {
        self.allocator.destroy_pool(handle);
    }

    /// Statistics for a previously-created pool. Returns `None` if the
    /// pool handle is unknown.
    pub fn weight_pool_statistics(
        &self,
        handle: vulkane::safe::PoolHandle,
    ) -> Option<vulkane::safe::AllocationStatistics> {
        self.allocator.pool_statistics(handle)
    }

    /// Build a defragmentation plan for the given pool. The returned
    /// plan enumerates GPU-side copies the caller must execute before
    /// passing the plan to [`Self::apply_defrag_plan`]. An empty plan
    /// (no moves) indicates the pool is already compact.
    ///
    /// See [`vulkane::safe::Allocator::build_defragmentation_plan`] for
    /// the full contract.
    pub fn build_defrag_plan(
        &self,
        pool: vulkane::safe::PoolHandle,
    ) -> vulkane::safe::DefragmentationPlan {
        self.allocator.build_defragmentation_plan(pool)
    }

    /// Apply a defragmentation plan.
    ///
    /// **Preconditions** (caller must guarantee):
    /// - For every move in `plan.moves`, the caller has issued a GPU
    ///   `vkCmdCopyBuffer` from `(src_memory, src_offset)` to
    ///   `(dst_memory, dst_offset)`, destroyed the old `VkBuffer`
    ///   bound to the source, and created / rebound a new one to the
    ///   destination.
    /// - The caller has waited for that GPU work to complete.
    /// - No other thread is racing on the affected allocations.
    ///
    /// After this call returns, every live `Allocation` in the pool
    /// reports its new `(memory, offset)` via `memory()` / `offset()`.
    ///
    /// ## Fuel-side status
    ///
    /// VulkanBackend does not yet own the machinery to issue the GPU
    /// copies and rebind buffers — that requires weight-pool allocation
    /// (TODO) plus a rebinding path through VulkanBuffer. Until those
    /// land, callers should treat this method as a pass-through to the
    /// underlying Vulkane primitive and supply the copy/rebind
    /// themselves. See `vulkane/docs/DEFRAG_FOR_ML.md`.
    pub fn apply_defrag_plan(&self, plan: vulkane::safe::DefragmentationPlan) {
        self.allocator.apply_defragmentation_plan(plan)
    }

    /// Evict a device-resident storage to a [`residency::ResidencyFile`]
    /// slot. Returns a new `VulkanStorage` with [`StorageBacking::Host`]
    /// pointing at the allocated slot. The caller should replace their
    /// reference to the old storage with the returned one — once the
    /// old storage's Arc<VulkanBuffer> refcount drops to zero, its VRAM
    /// is reclaimed by the buffer pool.
    ///
    /// Byte-level copy: downloads the raw buffer, allocates a slot,
    /// writes. Preserves `elem_count` + `dtype` on the new storage so
    /// a subsequent `fault_back` can reconstruct equivalently.
    ///
    /// This is a manual / explicit eviction. P5 step 2c integrates it
    /// with an OOM-triggered LRU policy inside `alloc_device`.
    pub fn evict(
        &self,
        storage: &VulkanStorage,
        file: &std::sync::Arc<residency::ResidencyFile>,
    ) -> fuel_core_types::Result<VulkanStorage> {
        if !matches!(storage.backing, StorageBacking::Device(_)) {
            fuel_core_types::bail!(
                "VulkanBackend::evict: storage is already Host-backed"
            );
        }
        let bytes = self.download_raw_bytes(storage)?;
        let slot = file.alloc(bytes.len() as u64).ok_or_else(|| {
            fuel_core_types::Error::Msg(format!(
                "evict: ResidencyFile has no contiguous slot for {} bytes \
                 (file capacity={}, free={})",
                bytes.len(), file.capacity(), file.bytes_free()
            ))
        })?;
        file.write(slot, &bytes);
        Ok(VulkanStorage {
            backing: StorageBacking::Host { file: std::sync::Arc::clone(file), slot },
            elem_count: storage.elem_count,
            dtype: storage.dtype,
            tier: Tier::OnHost,
        })
    }

    /// Bring a host-evicted storage back to VRAM. Allocates a fresh
    /// device buffer, copies the saved bytes from the file slot into
    /// it, returns the new on-device storage. Frees the file slot
    /// since we no longer need the host copy.
    ///
    /// Caller substitutes the returned storage for the old one; the
    /// old one's Arc<ResidencyFile> refcount drops and the slot is
    /// returned to the freelist via `file.free(slot)` inside this
    /// method.
    pub fn fault_back(
        &self,
        storage: &VulkanStorage,
    ) -> fuel_core_types::Result<VulkanStorage> {
        let (file, slot) = match &storage.backing {
            StorageBacking::Host { file, slot } => (file.clone(), *slot),
            StorageBacking::Device(_) => {
                fuel_core_types::bail!(
                    "VulkanBackend::fault_back: storage is already Device-backed"
                );
            }
        };
        let bytes = file.read(slot);
        // Reupload as the stored dtype. upload_slice handles the
        // byte-alignment via its generic over T: Copy + 'static.
        let new_storage = self.upload_slice(&bytes, DType::U8)?;
        // upload_slice returned storage has dtype=U8, elem_count=bytes.len().
        // Restore the original dtype + elem_count so downstream ops see
        // the same logical tensor they evicted.
        let fixed = VulkanStorage {
            backing: new_storage.backing,
            elem_count: storage.elem_count,
            dtype: storage.dtype,
            tier: Tier::OnDevice,
        };
        file.free(slot);
        Ok(fixed)
    }

    /// Evict storages from a caller-supplied list until at least
    /// `target_bytes` of VRAM have been freed, or the list is
    /// exhausted. The caller passes candidates in LRU order (oldest
    /// first); the backend walks them and evicts each until the target
    /// is met.
    ///
    /// Returns a parallel `Vec<Option<VulkanStorage>>` — `Some(new)`
    /// for each evicted storage (caller substitutes their ref),
    /// `None` for storages left untouched.
    ///
    /// ## Why caller-provided candidates?
    ///
    /// Full automated eviction — backend decides on OOM which storage
    /// to evict — needs interior mutability on `VulkanStorage.backing`
    /// so the backend can swap a live caller's storage from Device to
    /// Host behind their back. That refactor cascades through every
    /// `.buffer()` call site in the op methods. Deferred to step 2d.
    ///
    /// For now, the caller (typically a KV-cache manager) knows its
    /// working set and can enumerate cold entries. It invokes this
    /// method when it wants to free VRAM and substitutes the evicted
    /// refs in its own data structure.
    ///
    /// ## Reporting
    ///
    /// Bytes freed = sum of `byte_size()` over evicted candidates. This
    /// is the bytes reclaimed from the device allocator only when
    /// caller drops their old references to the evicted storages.
    pub fn evict_from_candidates(
        &self,
        candidates: &[&VulkanStorage],
        target_bytes: u64,
        file: &std::sync::Arc<residency::ResidencyFile>,
    ) -> fuel_core_types::Result<Vec<Option<VulkanStorage>>> {
        let mut freed: u64 = 0;
        let mut out: Vec<Option<VulkanStorage>> = Vec::with_capacity(candidates.len());
        for cand in candidates {
            if freed >= target_bytes {
                out.push(None);
                continue;
            }
            // Skip any candidate that's already host-backed — re-evicting
            // is a no-op (and would fail the Device-only guard in evict).
            if !matches!(cand.backing, StorageBacking::Device(_)) {
                out.push(None);
                continue;
            }
            let bytes = cand.byte_size();
            let evicted = self.evict(cand, file)?;
            freed += bytes;
            out.push(Some(evicted));
        }
        Ok(out)
    }

    /// Download the raw bytes of a device-resident storage. Used by
    /// [`Self::evict`]. Not a trait method because byte-level
    /// download is a tier-management concern, not part of the op API.
    fn download_raw_bytes(&self, storage: &VulkanStorage) -> fuel_core_types::Result<Vec<u8>> {
        match storage.backing {
            StorageBacking::Device(_) => {}
            StorageBacking::Host { .. } => {
                fuel_core_types::bail!(
                    "download_raw_bytes: storage is on host, not device"
                );
            }
        }
        // Piggyback on the typed download path; convert back to bytes.
        use fuel_core_types::HostBuffer;
        let hb = <Self as GraphBackend>::download(self, storage)?;
        Ok(match hb {
            HostBuffer::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            HostBuffer::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            HostBuffer::BF16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            HostBuffer::F16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            HostBuffer::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            HostBuffer::U8(v) => v,
            _ => fuel_core_types::bail!("download_raw_bytes: unsupported dtype"),
        })
    }

    pub fn dequantize_q8_0_from_storage(
        &self,
        input: &VulkanStorage,
        n_blocks: usize,
    ) -> fuel_core_types::Result<VulkanStorage> {
        const BLCK_SIZE: usize = 32;
        let n_elements = n_blocks * BLCK_SIZE;
        let out = self.alloc_device((n_elements * 4) as u64, n_elements, DType::F32)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct Q8Params { n_blocks: u32, out_elements: u32, _pad0: u32, _pad1: u32 }
        let p = Q8Params {
            n_blocks: n_blocks as u32,
            out_elements: n_elements as u32,
            _pad0: 0, _pad1: 0,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;
        self.dispatch_2buf(
            "dequant_q8_0",
            &self.pipelines.dequant_q8_0_pipeline,
            &self.pipelines.dequant_q8_0_layout,
            input, &out, pbuf, pmem,
            std::mem::size_of::<Q8Params>() as u64,
            Self::workgroups(n_elements), 1, 1,
        )?;
        Ok(out)
    }

    /// Fused Q4_0 × F32 gemv: computes `C = A @ W` where A is an f32
    /// vector of length K and W is a Q4_0-quantized matrix of logical
    /// shape `[N, K]` stored as `N × K/32` Q4_0 blocks (18 bytes each).
    ///
    /// This is the decode hot path for quantized inference — Q4_0 blocks
    /// stay resident in device memory at ~4× compression vs F32 (2× vs
    /// BF16). Dequant happens inline inside the shader, per element.
    ///
    /// `w_q4_0_storage` is expected to hold the raw block byte stream
    /// uploaded via `upload_slice(&blocks, DType::U32)` (the same
    /// representation `dequantize_q4_0` takes).
    pub fn qmatvec_q4_0(
        &self,
        a_f32: &VulkanStorage,
        w_q4_0_storage: &VulkanStorage,
        k: usize,
        n: usize,
    ) -> fuel_core_types::Result<VulkanStorage> {
        if a_f32.dtype != DType::F32 {
            fuel_core_types::bail!("qmatvec_q4_0: A must be F32, got {:?}", a_f32.dtype);
        }
        if k % 32 != 0 {
            fuel_core_types::bail!("qmatvec_q4_0: K must be multiple of 32, got {k}");
        }
        let out = self.alloc_device((n * 4) as u64, n, DType::F32)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct QmvParams { n: u32, k: u32, blocks_per_row: u32, _pad: u32 }
        let p = QmvParams {
            n: n as u32,
            k: k as u32,
            blocks_per_row: (k / 32) as u32,
            _pad: 0,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;

        self.dispatch_3buf(
            "qmatvec_q4_0",
            &self.pipelines.qmatvec_q4_0_pipeline,
            &self.pipelines.qmatvec_q4_0_layout,
            a_f32, w_q4_0_storage, &out, pbuf, pmem,
            std::mem::size_of::<QmvParams>() as u64,
            n as u32, 1, 1,
        )?;
        Ok(out)
    }

    /// Dispatch qmatvec for a single row of A. `a_f32` is the full
    /// activations buffer [..., M, K]; `row_a_offset_elems` is the
    /// element offset to the start of this row. `out` is the full
    /// output buffer [..., M, N]; `row_out_offset_elems` is the
    /// element offset for this row's output slice.
    fn qmatvec_q4_0_slice(
        &self,
        a_f32: &VulkanStorage,
        row_a_offset_elems: u64,
        w_q4_0_storage: &VulkanStorage,
        out: &VulkanStorage,
        row_out_offset_elems: u64,
        k: usize,
        n: usize,
    ) -> fuel_core_types::Result<()> {
        #[repr(C)] #[derive(Clone, Copy)]
        struct QmvParams { n: u32, k: u32, blocks_per_row: u32, _pad: u32 }
        let p = QmvParams {
            n: n as u32,
            k: k as u32,
            blocks_per_row: (k / 32) as u32,
            _pad: 0,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;

        let desc = self.pipelines.allocate_desc(&self.pipelines.layout_3s1u).map_err(vk_err)?;
        let a_byte_off = row_a_offset_elems * 4;
        let a_byte_len = (k * 4) as u64;
        let out_byte_off = row_out_offset_elems * 4;
        let out_byte_len = (n * 4) as u64;
        desc.write_buffer(0, DescriptorType::STORAGE_BUFFER, a_f32.buffer(), a_byte_off, a_byte_len);
        desc.write_buffer(1, DescriptorType::STORAGE_BUFFER, w_q4_0_storage.buffer(), 0, w_q4_0_storage.byte_size());
        desc.write_buffer(2, DescriptorType::STORAGE_BUFFER, out.buffer(), out_byte_off, out_byte_len);
        desc.write_buffer(3, DescriptorType::UNIFORM_BUFFER, &pbuf, 0, std::mem::size_of::<QmvParams>() as u64);

        let rb = [a_f32.buffer().raw() as u64, w_q4_0_storage.buffer().raw() as u64];
        let wb = [out.buffer().raw() as u64];
        self.record_dispatch_batched(
            "qmatvec_q4_0",
            &self.pipelines.qmatvec_q4_0_pipeline,
            &self.pipelines.qmatvec_q4_0_layout,
            desc,
            (n as u32, 1, 1),
            vec![(pbuf, pmem)],
            &rb, &wb,
        )?;
        Ok(())
    }

    /// Dequantize a raw Q4_K_M blob (144-byte super-blocks, 256
    /// elements per super-block) directly on the GPU to an f32
    /// storage buffer.
    ///
    /// Takes the block byte stream as an already-on-device U32-typed
    /// VulkanStorage (produced by `upload_slice(&blocks, DType::U32)`).
    /// Mirrors `dequantize_q8_0_from_storage` for the Q4_K_M format.
    ///
    /// Matmul integration (dispatching `Op::QMatMul { quant_type: Q4KM }`
    /// through to a fused gemv kernel) is a follow-up; today this method
    /// covers dequant-then-matmul and future KV-cache-style flows.
    pub fn dequantize_q4_km(
        &self,
        blocks: &VulkanStorage,
        n_blocks: usize,
    ) -> fuel_core_types::Result<VulkanStorage> {
        const QK_K: usize = 256;
        let n_elements = n_blocks * QK_K;
        let out = self.alloc_device((n_elements * 4) as u64, n_elements, DType::F32)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct Q4KMParams { n_blocks: u32, out_elements: u32, _p0: u32, _p1: u32 }
        let p = Q4KMParams {
            n_blocks: n_blocks as u32,
            out_elements: n_elements as u32,
            _p0: 0, _p1: 0,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;

        // 64 threads per workgroup, one workgroup per super-block.
        self.dispatch_2buf(
            "dequant_q4_km",
            &self.pipelines.dequant_q4_km_pipeline,
            &self.pipelines.dequant_q4_km_layout,
            blocks, &out, pbuf, pmem,
            std::mem::size_of::<Q4KMParams>() as u64,
            n_blocks as u32, 1, 1,
        )?;
        Ok(out)
    }

    /// Fused Q4_0 × F32 tiled matmul for M > 1 (prefill path).
    /// One workgroup per (m_tile, n_col). TM = 8 M-rows per tile.
    /// Activation `a_f32` is [M, K] contiguous F32; `w_q4_0_storage`
    /// is the Q4_0 block byte stream in [N, K/32] layout. Returns
    /// [M, N] F32 output.
    ///
    /// Decode (M=1) should go through `qmatvec_q4_0` instead — that
    /// kernel is tuned for the single-row case and avoids the
    /// register pressure of TM=8 accumulators.
    pub fn matmul_q4_0_tiled(
        &self,
        a_f32: &VulkanStorage,
        w_q4_0_storage: &VulkanStorage,
        m: usize, k: usize, n: usize,
    ) -> fuel_core_types::Result<VulkanStorage> {
        if a_f32.dtype != DType::F32 {
            fuel_core_types::bail!("matmul_q4_0_tiled: A must be F32, got {:?}", a_f32.dtype);
        }
        if k % 32 != 0 {
            fuel_core_types::bail!("matmul_q4_0_tiled: K must be multiple of 32, got {k}");
        }
        const TM: usize = 8;
        let out_elems = m * n;
        let out = self.alloc_device((out_elems * 4) as u64, out_elems, DType::F32)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct TiledParams { m: u32, n: u32, k: u32, blocks_per_row: u32 }
        let p = TiledParams {
            m: m as u32, n: n as u32, k: k as u32,
            blocks_per_row: (k / 32) as u32,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;

        // Grid: one workgroup per (n, m_tile).
        let n_tiles_m = ((m + TM - 1) / TM) as u32;
        self.dispatch_3buf(
            "matmul_q4_0_tiled",
            &self.pipelines.matmul_q4_0_tiled_pipeline,
            &self.pipelines.matmul_q4_0_tiled_layout,
            a_f32, w_q4_0_storage, &out, pbuf, pmem,
            std::mem::size_of::<TiledParams>() as u64,
            n as u32, n_tiles_m, 1,
        )?;
        Ok(out)
    }

    /// Quantize an F32 tensor to GGML Q8_0 blocks (34 bytes / 32
    /// elements). Used for KV-cache quantization: between decode
    /// steps, the cached K/V are stored as Q8_0 (1 byte/element vs F32's
    /// 4) to double-or-more the effective context at the same VRAM.
    ///
    /// Returns a U32-typed VulkanStorage holding the raw block byte
    /// stream (paired with `dequantize_q8_0` for readback).
    pub fn quantize_q8_0(
        &self,
        src_f32: &VulkanStorage,
        n_elements: usize,
    ) -> fuel_core_types::Result<VulkanStorage> {
        if src_f32.dtype != DType::F32 {
            fuel_core_types::bail!(
                "VulkanBackend::quantize_q8_0: src must be F32, got {:?}",
                src_f32.dtype
            );
        }
        const BLCK_SIZE: usize = 32;
        const BYTES_PER_BLOCK: usize = 34;
        if n_elements % BLCK_SIZE != 0 {
            fuel_core_types::bail!(
                "quantize_q8_0: n_elements {n_elements} must be multiple of {BLCK_SIZE}"
            );
        }
        let n_blocks = n_elements / BLCK_SIZE;
        let out_bytes = n_blocks * BYTES_PER_BLOCK;
        // Round up to u32 multiple (4 bytes per u32).
        let out_u32_len = (out_bytes + 3) / 4;
        let out = self.alloc_device(
            (out_u32_len * 4) as u64, out_u32_len, DType::U32
        )?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct QQParams { n_elements: u32, n_blocks: u32, _p0: u32, _p1: u32 }
        let p = QQParams {
            n_elements: n_elements as u32,
            n_blocks: n_blocks as u32,
            _p0: 0, _p1: 0,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;

        // 64 threads per workgroup, one thread per block.
        let groups = ((n_blocks + 63) / 64) as u32;
        self.dispatch_2buf(
            "quantize_q8_0",
            &self.pipelines.quantize_q8_0_pipeline,
            &self.pipelines.quantize_q8_0_layout,
            src_f32, &out, pbuf, pmem,
            std::mem::size_of::<QQParams>() as u64,
            groups, 1, 1,
        )?;
        Ok(out)
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
            // Zero-copy share only makes sense for device-backed storages.
            // A host-backed storage can't be Arc-shared into a device ref
            // without fault-back first; bail for clarity.
            let shared = storage.device_buffer_arc().ok_or_else(|| {
                fuel_core_types::Error::Msg(
                    "try_clone: host-backed storage needs fault-back first".into()
                )
            })?;
            return Ok(VulkanStorage {
                backing: StorageBacking::Device(shared),
                elem_count: n,
                dtype: storage.dtype,
                tier: storage.tier,
            });
        }
        let byte_size = (n * dtype_size(storage.dtype)) as u64;
        let dst = self.alloc_device(byte_size, n, storage.dtype)?;
        // Memcpy is a transfer op — flush the compute batch first,
        // then run the copy synchronously via one_shot.
        self.flush_pending()?;
        self.queue
            .one_shot(&self.device, self.queue_family, |cmd| {
                cmd.copy_buffer(storage.buffer(), dst.buffer(), &[BufferCopy {
                    src_offset: 0, dst_offset: 0, size: byte_size,
                }]);
                Ok(())
            })
            .map_err(vk_err)?;
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
        let rb = [src.buffer().raw() as u64];
        let wb = [dst.buffer().raw() as u64];
        self.record_dispatch_batched(
            "strided_copy",
            &self.pipelines.strided_copy_pipeline,
            &self.pipelines.strided_copy_layout,
            desc,
            (groups, 1, 1),
            vec![(sd_buf, sd_mem), (pbuf, pmem)],
            &rb, &wb,
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

        // GQA-aware: infer n_rep from the SHAPES, not strides.
        // For non-contiguous B, stride-based elem_count/stride is wrong.
        // Use the actual batch-head count from B's shape.
        let b_dims = _lb.shape().dims();
        let b_batch_count: usize = b_dims[..b_rank.saturating_sub(2)]
            .iter().product::<usize>().max(1);
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
                // Reuse padded_out's backing (it was freshly alloc_device'd,
                // so this Arc has refcount 1 — move rather than clone).
                let padded_backing = padded_out.backing;
                return Ok(VulkanStorage {
                    backing: padded_backing,
                    elem_count: out_n,
                    dtype: DType::F32,
                    tier: Tier::OnDevice,
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

    fn conv2d(
        &self,
        input:  &Self::Storage,
        weight: &Self::Storage,
        input_layout:  &Layout,
        weight_layout: &Layout,
        stride:  (usize, usize),
        padding: (usize, usize),
        groups:  usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        // Phase 1 of Vulkan conv2d: im2col + matmul, groups=1 only.
        // Matches the current CUDA backend's parity surface; depthwise
        // (groups != 1) will land on both backends together once the
        // baracuda-cudnn group-count API ships.
        if groups != 1 {
            fuel_core_types::bail!(
                "VulkanBackend::conv2d: groups != 1 not yet supported \
                 (got groups={groups}); falling back to CPU"
            );
        }
        let i_dims = input_layout.shape().dims();
        let w_dims = weight_layout.shape().dims();
        if i_dims.len() != 4 || w_dims.len() != 4 {
            fuel_core_types::bail!(
                "VulkanBackend::conv2d: expected rank-4 input + weight, got {i_dims:?} and {w_dims:?}"
            );
        }
        if input.dtype != DType::F32 || weight.dtype != DType::F32 {
            fuel_core_types::bail!(
                "VulkanBackend::conv2d: only F32 inputs supported today (got input={:?}, weight={:?})",
                input.dtype, weight.dtype
            );
        }
        if !input_layout.is_contiguous() || !weight_layout.is_contiguous() {
            fuel_core_types::bail!(
                "VulkanBackend::conv2d: strided inputs not supported; \
                 the executor's materialize_if_needed should have handled this"
            );
        }
        let s = fuel_conv::ConvShape {
            batch: i_dims[0], c_in: i_dims[1], h: i_dims[2], w: i_dims[3],
            c_out: w_dims[0], k_h: w_dims[2], k_w: w_dims[3],
            stride, padding, groups,
        };
        s.validate().map_err(|e| fuel_core_types::Error::Msg(
            format!("VulkanBackend::conv2d: shape validation: {e}")
        ))?;
        let h_out = s.h_out();
        let w_out = s.w_out();
        let m = s.c_out;                         // weight rows
        let k_dim = s.c_in_per_group() * s.k_h * s.k_w; // weight cols / patches rows
        let n = h_out * w_out;                   // patches cols / out spatial

        // Allocate the patches scratch + the output buffer.
        let patches_n = s.im2col_len();
        let patches = self.alloc_device((patches_n * 4) as u64, patches_n, DType::F32)?;
        let out_n = s.output_len();
        let out = self.alloc_device((out_n * 4) as u64, out_n, DType::F32)?;

        // -------- im2col dispatch --------
        #[repr(C)] #[derive(Clone, Copy)]
        struct Im2ColParams {
            batch: u32, c_in: u32, h: u32, w: u32,
            h_out: u32, w_out: u32,
            k_h: u32, k_w: u32,
            stride_h: u32, stride_w: u32,
            pad_h: u32, pad_w: u32,
            groups: u32, cin_per_g: u32,
            total_elements: u32, _pad: u32,
        }
        let total = patches_n as u32;
        let im2col_params = Im2ColParams {
            batch:    s.batch as u32,
            c_in:     s.c_in as u32,
            h:        s.h as u32,
            w:        s.w as u32,
            h_out:    h_out as u32,
            w_out:    w_out as u32,
            k_h:      s.k_h as u32,
            k_w:      s.k_w as u32,
            stride_h: s.stride.0 as u32,
            stride_w: s.stride.1 as u32,
            pad_h:    s.padding.0 as u32,
            pad_w:    s.padding.1 as u32,
            groups:   s.groups as u32,
            cin_per_g: s.c_in_per_group() as u32,
            total_elements: total,
            _pad: 0,
        };
        let (im2col_pbuf, im2col_pmem) = self.upload_params(&im2col_params)?;
        let im2col_wg = (total + 255) / 256;
        self.dispatch_2buf(
            "conv2d_im2col",
            &self.pipelines.conv2d_im2col_pipeline,
            &self.pipelines.conv2d_im2col_layout,
            input, &patches, im2col_pbuf, im2col_pmem,
            std::mem::size_of::<Im2ColParams>() as u64,
            im2col_wg, 1, 1,
        )?;

        // -------- matmul dispatch --------
        // groups == 1: a single batched matmul where weight broadcasts
        // across `batch`.
        //   A = weight,  shape [m, k_dim]    (no batch dim → broadcast)
        //   B = patches, shape [batch, k_dim, n]
        //   C = out,     shape [batch, m, n]
        //
        // The matmul shader computes a_off = batch * sa_batch and
        // b_off = (batch / n_rep) * sb_batch. To get A-broadcast +
        // B-per-batch we set `sa_batch = 0` (so a_off = 0 every batch)
        // and `n_rep = 1` (so b_off advances per batch). The shader's
        // n_rep mechanism is GQA-shaped — designed for multiple A
        // heads sharing one B — which is the opposite of what conv2d
        // needs, so we don't use it here.
        #[repr(C)] #[derive(Clone, Copy)]
        struct MatmulParams {
            m: u32, n: u32, k: u32,
            sa_batch: u32, sa_row: u32, sa_col: u32,
            sb_batch: u32, sb_row: u32, sb_col: u32,
            sc_batch: u32,
            n_rep: u32,
            _pad: u32,
        }
        let matmul_params = MatmulParams {
            m: m as u32, n: n as u32, k: k_dim as u32,
            sa_batch: 0,                   // A is shared across batches
            sa_row:   k_dim as u32,
            sa_col:   1,
            sb_batch: (k_dim * n) as u32,  // B walks per batch
            sb_row:   n as u32,
            sb_col:   1,
            sc_batch: (m * n) as u32,
            n_rep:    1,
            _pad: 0,
        };
        let (mm_pbuf, mm_pmem) = self.upload_params(&matmul_params)?;
        let mm_params_size = std::mem::size_of::<MatmulParams>() as u64;
        let gz = s.batch as u32;

        // Choose pipeline: M==1 → matvec; otherwise the WGSL register-tile
        // matmul. (The tiled GLSL variant pays barriers that aren't
        // worth it for typical conv2d M sizes — c_out is usually 64–512.)
        if m == 1 {
            let gx = n as u32;
            self.dispatch_3buf(
                "conv2d.matvec",
                &self.pipelines.matvec_pipeline,
                &self.pipelines.matvec_layout,
                weight, &patches, &out, mm_pbuf, mm_pmem, mm_params_size,
                gx, 1, gz,
            )?;
        } else {
            // WGSL register-tile: 16x16 workgroups, 4x4 output tile each
            // → groups_x = ceil(n/64), groups_y = ceil(m/64).
            let gx = ((n + 63) / 64) as u32;
            let gy = ((m + 63) / 64) as u32;
            self.dispatch_3buf(
                "conv2d.matmul",
                &self.pipelines.matmul_pipeline,
                &self.pipelines.matmul_layout,
                weight, &patches, &out, mm_pbuf, mm_pmem, mm_params_size,
                gx, gy, gz,
            )?;
        }

        Ok(out)
    }

    fn matmul_q4_0(
        &self,
        a: &Self::Storage,
        w_q_bytes: &Self::Storage,
        k: usize,
        n: usize,
        a_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        if a.dtype != DType::F32 {
            fuel_core_types::bail!("VulkanBackend: matmul_q4_0 A must be F32, got {:?}", a.dtype);
        }
        if !a_layout.is_contiguous() {
            // Fallback for strided A: executor handles via materialize_if_needed
            // upstream in most cases, but bail here for safety — contiguous A
            // is what our gemv kernel expects.
            fuel_core_types::bail!("VulkanBackend: matmul_q4_0 requires contiguous A");
        }
        let a_dims = a_layout.shape().dims();
        let rank = a_dims.len();
        if rank < 2 {
            fuel_core_types::bail!("VulkanBackend: matmul_q4_0 A must be rank ≥ 2");
        }
        let m = a_dims[rank - 2];
        let batch: usize = a_dims[..rank - 2].iter().product::<usize>().max(1);
        let total_rows = batch * m;

        // For M=1 (decode hot path), use the tuned qmatvec. For M>1
        // (prefill), use the tiled kernel that reuses each weight
        // load across TM=8 A rows — one dispatch vs total_rows.
        if total_rows == 1 {
            let out = self.alloc_device((n * 4) as u64, n, DType::F32)?;
            self.qmatvec_q4_0_slice(a, 0, w_q_bytes, &out, 0, k, n)?;
            Ok(out)
        } else {
            VulkanBackend::matmul_q4_0_tiled(self, a, w_q_bytes, total_rows, k, n)
        }
    }

    fn matmul_q4_km(
        &self,
        a: &Self::Storage,
        w_q_bytes: &Self::Storage,
        k: usize,
        n: usize,
        a_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        if a.dtype != DType::F32 {
            fuel_core_types::bail!("VulkanBackend: matmul_q4_km A must be F32, got {:?}", a.dtype);
        }
        if !a_layout.is_contiguous() {
            fuel_core_types::bail!("VulkanBackend: matmul_q4_km requires contiguous A");
        }
        let a_dims = a_layout.shape().dims();
        let rank = a_dims.len();
        if rank < 2 {
            fuel_core_types::bail!("VulkanBackend: matmul_q4_km A must be rank ≥ 2");
        }
        let m = a_dims[rank - 2];
        let batch: usize = a_dims[..rank - 2].iter().product::<usize>().max(1);
        let total_rows = batch * m;

        // First-pass implementation: dequantize W to F32 on-device, then
        // use the standard matmul. Keeps the weight bytes compressed on
        // disk/RAM; only the dequantized view is materialized for this
        // forward. Fused qmatvec/matmul variants for Q4_K_M are a perf
        // follow-up (matches where Q4_0 was two sessions ago).
        const QK_K: usize = 256;
        if k % QK_K != 0 {
            fuel_core_types::bail!(
                "VulkanBackend: matmul_q4_km K ({k}) must be multiple of {QK_K}"
            );
        }
        let n_blocks = n * (k / QK_K);
        let w_f32 = self.dequantize_q4_km(w_q_bytes, n_blocks)?;
        // Dequantized weight is [n*k] linear; treat as [n, k] row-major.
        // Our matmul expects [K, N] for the right operand, so we need
        // a transpose view. Build the layout explicitly.
        let w_shape = Shape::from_dims(&[n, k]);
        let w_layout_nk = Layout::contiguous(&w_shape);
        // Transpose to [K, N]: permute (0,1) → (1,0). Resulting layout
        // has shape [K, N] and strided access pattern.
        let w_layout_kn = w_layout_nk.transpose(rank - 2, rank - 1)
            .map_err(|e| fuel_core_types::Error::Msg(
                format!("matmul_q4_km: transpose layout error: {e}")))?;
        // Build A's [batch, m, k] layout and dispatch matmul.
        self.matmul(
            a, &w_f32,
            (batch, m, n, k),
            a_layout, &w_layout_kn,
        )
    }

    fn quantize_q8_0(
        &self,
        src_f32: &Self::Storage,
        n_elements: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        VulkanBackend::quantize_q8_0(self, src_f32, n_elements)
    }

    fn dequantize_q8_0(
        &self,
        blocks: &Self::Storage,
        n_blocks: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        self.dequantize_q8_0_from_storage(blocks, n_blocks)
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
        la: &Layout, lb: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        // Output shape = la.shape() (they must match for a non-broadcasting
        // binary op; for broadcast-binary the caller passes operands already
        // broadcast to the output shape, but with stride=0 on broadcast dims).
        let out_dims = la.shape().dims();
        let out_elem = la.shape().elem_count();
        if out_elem != lb.shape().elem_count() {
            fuel_core_types::bail!(
                "VulkanBackend: binary shape mismatch a={:?} b={:?}",
                la.shape(), lb.shape()
            );
        }
        let rank = out_dims.len();
        if rank > 4 {
            fuel_core_types::bail!(
                "VulkanBackend: binary supports rank ≤ 4, got {rank}"
            );
        }
        let out = self.alloc_device(
            (out_elem * dtype_size(a.dtype)) as u64, out_elem, a.dtype)?;

        let op_id: u32 = match op {
            BinaryOp::Add => 0, BinaryOp::Sub => 1, BinaryOp::Mul => 2,
            BinaryOp::Div => 3, BinaryOp::Maximum => 4, BinaryOp::Minimum => 5,
        };

        // Pad shape and strides to rank 4 (leading dims = 1, strides = 0).
        let mut shape = [1u32; 4];
        let mut a_s = [0u32; 4];
        let mut b_s = [0u32; 4];
        let pad = 4 - rank;
        for i in 0..rank {
            shape[pad + i] = out_dims[i] as u32;
            a_s[pad + i] = la.stride()[i] as u32;
            b_s[pad + i] = lb.stride()[i] as u32;
        }

        // Fast-path flag: contiguous AND matches output shape exactly
        // (i.e. no broadcast, no permute). stride=0 on any dim rules
        // out the fast path.
        let a_contig = la.is_contiguous()
            && la.shape().dims() == out_dims
            && la.stride().iter().all(|&s| s != 0);
        let b_contig = lb.is_contiguous()
            && lb.shape().dims() == out_dims
            && lb.stride().iter().all(|&s| s != 0);
        let flags = (a_contig as u32) | ((b_contig as u32) << 1);

        #[repr(C)] #[derive(Clone, Copy)]
        struct BParams {
            out_size: u32, op_id: u32, rank: u32, flags: u32,
            shape0: u32, shape1: u32, shape2: u32, shape3: u32,
            a_s0: u32, a_s1: u32, a_s2: u32, a_s3: u32,
            b_s0: u32, b_s1: u32, b_s2: u32, b_s3: u32,
        }
        let p = BParams {
            out_size: out_elem as u32, op_id, rank: rank as u32, flags,
            shape0: shape[0], shape1: shape[1], shape2: shape[2], shape3: shape[3],
            a_s0: a_s[0], a_s1: a_s[1], a_s2: a_s[2], a_s3: a_s[3],
            b_s0: b_s[0], b_s1: b_s[1], b_s2: b_s[2], b_s3: b_s[3],
        };
        let (pbuf, pmem) = self.upload_params(&p)?;
        self.dispatch_3buf(
            "binary",
            &self.pipelines.binary_pipeline,
            &self.pipelines.binary_layout,
            a, b, &out, pbuf, pmem,
            std::mem::size_of::<BParams>() as u64,
            Self::workgroups(out_elem), 1, 1,
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

        let groups = Self::workgroups(n);
        let rb = [src.buffer().raw() as u64, dst.buffer().raw() as u64];
        let wb = [dst.buffer().raw() as u64];
        self.record_dispatch_batched(
            "add_assign_scaled",
            &self.pipelines.add_assign_scaled_pipeline,
            &self.pipelines.add_assign_scaled_layout,
            desc,
            (groups, 1, 1),
            vec![(pbuf, pmem)],
            &rb, &wb,
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
        a_layout: &Layout,
        b_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            fuel_core_types::bail!("VulkanBackend: concat_along_dim requires f32");
        }
        let a_dims = a_layout.shape().dims();
        let b_dims = b_layout.shape().dims();
        if a_dims.len() != b_dims.len() || dim >= a_dims.len() {
            fuel_core_types::bail!("concat_along_dim: rank/dim mismatch");
        }
        for (i, (&da, &db)) in a_dims.iter().zip(b_dims.iter()).enumerate() {
            if i != dim && da != db {
                fuel_core_types::bail!("concat_along_dim: non-concat dims disagree");
            }
        }
        let rank = a_dims.len();
        if rank > 4 {
            fuel_core_types::bail!("VulkanBackend: concat supports rank ≤ 4, got {rank}");
        }
        let a_dim = a_dims[dim];
        let b_dim = b_dims[dim];
        // Output shape = a_dims with dim replaced by a_dim + b_dim.
        let mut out_dims_vec: Vec<usize> = a_dims.to_vec();
        out_dims_vec[dim] = a_dim + b_dim;
        let out_elems: usize = out_dims_vec.iter().product();
        let out = self.alloc_device((out_elems * 4) as u64, out_elems, DType::F32)?;

        // Pad shape + strides to rank 4 (leading dims = 1, strides = 0
        // for padded positions). `concat_dim` shifts accordingly.
        let pad = 4 - rank;
        let mut out_d = [1u32; 4];
        let mut a_s = [0u32; 4];
        let mut b_s = [0u32; 4];
        for i in 0..rank {
            out_d[pad + i] = out_dims_vec[i] as u32;
            a_s[pad + i] = a_layout.stride()[i] as u32;
            b_s[pad + i] = b_layout.stride()[i] as u32;
        }
        let concat_dim_padded = (pad + dim) as u32;

        #[repr(C)] #[derive(Clone, Copy)]
        struct CParams {
            out_d0: u32, out_d1: u32, out_d2: u32, out_d3: u32,
            concat_dim: u32, a_dim: u32, b_dim: u32, total: u32,
            a_s0: u32, a_s1: u32, a_s2: u32, a_s3: u32,
            b_s0: u32, b_s1: u32, b_s2: u32, b_s3: u32,
        }
        let p = CParams {
            out_d0: out_d[0], out_d1: out_d[1], out_d2: out_d[2], out_d3: out_d[3],
            concat_dim: concat_dim_padded,
            a_dim: a_dim as u32,
            b_dim: b_dim as u32,
            total: out_elems as u32,
            a_s0: a_s[0], a_s1: a_s[1], a_s2: a_s[2], a_s3: a_s[3],
            b_s0: b_s[0], b_s1: b_s[1], b_s2: b_s[2], b_s3: b_s[3],
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

        // Compute x stride params. Support up to 2 outer dims (rank ≤ 4).
        let x_strides = x_layout.stride();
        let contiguous = x_layout.is_contiguous();
        let (x_s0, x_s1, x_s_seq, x_s_hd, x_outer1) = if contiguous {
            // Fast path values (unused by shader when x_contiguous == 1).
            (0u32, 0u32, 0u32, 0u32, 1u32)
        } else {
            match rank {
                2 => (
                    // [seq, head_dim]
                    (x_strides[0] * dims[0]) as u32, // unused (outer=1)
                    (x_strides[0] * dims[0]) as u32, // unused
                    x_strides[0] as u32,
                    x_strides[1] as u32,
                    1u32,
                ),
                3 => (
                    x_strides[0] as u32,
                    x_strides[0] as u32, // unused (outer1=1)
                    x_strides[1] as u32,
                    x_strides[2] as u32,
                    1u32,
                ),
                4 => (
                    x_strides[0] as u32,
                    x_strides[1] as u32,
                    x_strides[2] as u32,
                    x_strides[3] as u32,
                    dims[1] as u32,
                ),
                _ => fuel_core_types::bail!(
                    "VulkanBackend: rope stride-aware path supports rank 2-4, got {rank}"
                ),
            }
        };

        let out = self.alloc_device(x.byte_size(), x.elem_count, x.dtype)?;

        #[repr(C)] #[derive(Clone, Copy)]
        struct RopeParams {
            outer: u32, seq: u32, head_dim: u32, total: u32,
            x_s0: u32, x_s1: u32, x_s_seq: u32, x_s_hd: u32,
            x_outer1: u32, x_contiguous: u32, _pad0: u32, _pad1: u32,
        }
        let p = RopeParams {
            outer, seq, head_dim, total,
            x_s0, x_s1, x_s_seq, x_s_hd,
            x_outer1, x_contiguous: contiguous as u32, _pad0: 0, _pad1: 0,
        };
        let (pbuf, pmem) = self.upload_params(&p)?;

        let desc = self.pipelines.allocate_desc(&self.pipelines.layout_4s1u).map_err(vk_err)?;
        desc.write_buffer(0, DescriptorType::STORAGE_BUFFER, x.buffer(), 0, x.byte_size());
        desc.write_buffer(1, DescriptorType::STORAGE_BUFFER, cos.buffer(), 0, cos.byte_size());
        desc.write_buffer(2, DescriptorType::STORAGE_BUFFER, sin.buffer(), 0, sin.byte_size());
        desc.write_buffer(3, DescriptorType::STORAGE_BUFFER, out.buffer(), 0, out.byte_size());
        desc.write_buffer(4, DescriptorType::UNIFORM_BUFFER, &pbuf, 0, std::mem::size_of::<RopeParams>() as u64);

        let groups = ((total + 63) / 64).max(1);
        let rb = [x.buffer().raw() as u64, cos.buffer().raw() as u64, sin.buffer().raw() as u64];
        let wb = [out.buffer().raw() as u64];
        self.record_dispatch_batched(
            "rope",
            &self.pipelines.rope_pipeline,
            &self.pipelines.rope_layout,
            desc,
            (groups, 1, 1),
            vec![(pbuf, pmem)],
            &rb, &wb,
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

        let groups = Self::workgroups(out_size);
        let rb = [src.buffer().raw() as u64, ids.buffer().raw() as u64];
        let wb = [out.buffer().raw() as u64];
        self.record_dispatch_batched(
            "index_select",
            &self.pipelines.index_select_pipeline,
            &self.pipelines.index_select_layout,
            desc,
            (groups, 1, 1),
            vec![(pbuf, pmem)],
            &rb, &wb,
        )?;
        Ok(out)
    }

    fn gather(
        &self, _src: &Self::Storage, _ids: &Self::Storage,
        _src_l: &Layout, _ids_l: &Layout, _dim: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        fuel_core_types::bail!("VulkanBackend: gather not yet native")
    }

    fn flash_attn(
        &self,
        q: &Self::Storage,
        k: &Self::Storage,
        v: &Self::Storage,
        alibi_slopes: Option<&Self::Storage>,
        q_layout: &Layout,
        k_layout: &Layout,
        v_layout: &Layout,
        _alibi_layout: Option<&Layout>,
        softmax_scale: f32,
        causal: bool,
        window_size_left: Option<usize>,
        window_size_right: Option<usize>,
        softcap: Option<f32>,
    ) -> fuel_core_types::Result<Self::Storage> {
        // F32-only, contiguous-only first cut. Strided / non-f32
        // returns Err -> executor falls back to attention_naive.
        if q.dtype != DType::F32 || k.dtype != DType::F32 || v.dtype != DType::F32 {
            fuel_core_types::bail!(
                "VulkanBackend::flash_attn: only F32 supported (got q={:?} k={:?} v={:?})",
                q.dtype, k.dtype, v.dtype,
            );
        }
        if !q_layout.is_contiguous() || !k_layout.is_contiguous() || !v_layout.is_contiguous() {
            fuel_core_types::bail!("VulkanBackend::flash_attn: strided inputs not yet supported");
        }
        let q_dims = q_layout.shape().dims();
        let k_dims = k_layout.shape().dims();
        let v_dims = v_layout.shape().dims();
        if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
            fuel_core_types::bail!(
                "VulkanBackend::flash_attn: expected rank-4 q/k/v, got {q_dims:?} {k_dims:?} {v_dims:?}"
            );
        }
        let (b, hq, sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let (_, hkv, sk, _) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
        // Shader's D_MAX is 128. Larger head_dim → fall back to CPU.
        if d > 128 {
            fuel_core_types::bail!("VulkanBackend::flash_attn: head_dim={d} exceeds D_MAX=128");
        }
        if hq % hkv != 0 {
            fuel_core_types::bail!("VulkanBackend::flash_attn: Hq={hq} must be a multiple of Hkv={hkv}");
        }

        let out_n = b * hq * sq * d;
        let out = self.alloc_device((out_n * 4) as u64, out_n, DType::F32)?;

        // Alibi binding: bind a 1-element dummy buffer when no slopes
        // (the descriptor needs *something* there even if has_alibi=0).
        let dummy_alibi;
        let alibi_storage = match alibi_slopes {
            Some(a) => a,
            None => {
                dummy_alibi = self.alloc_device(4, 1, DType::F32)?;
                &dummy_alibi
            }
        };

        #[repr(C)] #[derive(Clone, Copy)]
        struct Params {
            b: u32,
            hq: u32,
            hkv: u32,
            sq: u32,
            sk: u32,
            d: u32,
            groups: u32,
            causal: u32,
            window_left: u32,
            window_right: u32,
            has_window_left: u32,
            has_window_right: u32,
            has_alibi: u32,
            has_softcap: u32,
            softmax_scale: f32,
            softcap: f32,
        }
        let params = Params {
            b: b as u32,
            hq: hq as u32,
            hkv: hkv as u32,
            sq: sq as u32,
            sk: sk as u32,
            d: d as u32,
            groups: (hq / hkv) as u32,
            causal: if causal { 1 } else { 0 },
            window_left: window_size_left.unwrap_or(0) as u32,
            window_right: window_size_right.unwrap_or(0) as u32,
            has_window_left: if window_size_left.is_some() { 1 } else { 0 },
            has_window_right: if window_size_right.is_some() { 1 } else { 0 },
            has_alibi: if alibi_slopes.is_some() { 1 } else { 0 },
            has_softcap: if softcap.is_some() { 1 } else { 0 },
            softmax_scale,
            softcap: softcap.unwrap_or(0.0),
        };
        let (pbuf, pmem) = self.upload_params(&params)?;

        // Workgroup grid: (B, Hq, ceil(Sq / BR=16))
        let groups_x = b as u32;
        let groups_y = hq as u32;
        let groups_z = ((sq + 15) / 16) as u32;

        // 5-storage + 1-uniform descriptor: (q, k, v, alibi, o).
        let desc = self.pipelines
            .allocate_desc(&self.pipelines.layout_5s1u)
            .map_err(vk_err)?;
        desc.write_buffer(0, DescriptorType::STORAGE_BUFFER, q.buffer(), 0, q.byte_size());
        desc.write_buffer(1, DescriptorType::STORAGE_BUFFER, k.buffer(), 0, k.byte_size());
        desc.write_buffer(2, DescriptorType::STORAGE_BUFFER, v.buffer(), 0, v.byte_size());
        desc.write_buffer(3, DescriptorType::STORAGE_BUFFER, alibi_storage.buffer(), 0, alibi_storage.byte_size());
        desc.write_buffer(4, DescriptorType::STORAGE_BUFFER, out.buffer(), 0, out.byte_size());
        desc.write_buffer(5, DescriptorType::UNIFORM_BUFFER, &pbuf, 0, std::mem::size_of::<Params>() as u64);

        let rb = [
            q.buffer().raw() as u64,
            k.buffer().raw() as u64,
            v.buffer().raw() as u64,
            alibi_storage.buffer().raw() as u64,
        ];
        let wb = [out.buffer().raw() as u64];
        self.record_dispatch_batched(
            "flash_attention",
            &self.pipelines.flash_attention_pipeline,
            &self.pipelines.flash_attention_layout,
            desc,
            (groups_x, groups_y, groups_z),
            vec![(pbuf, pmem)],
            &rb, &wb,
        )?;
        Ok(out)
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

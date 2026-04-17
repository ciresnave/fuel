//! Batched command buffer recorder for VulkanBackend.
//!
//! Records ALL dispatches for a realize() pass into a SINGLE command
//! buffer with pipeline barriers between ops. Submitted once at
//! download time. Replaces the old per-op submit pattern that incurred
//! ~30µs of host overhead per dispatch × ~9K dispatches per 32-token
//! run = ~270ms of pure submit overhead.
//!
//! The batch recording uses raw Vulkan calls to keep the command
//! buffer in recording state across multiple `record_batch_dispatch`
//! calls — vulkane's RAII `CommandBufferRecording` wrapper calls
//! `vkEndCommandBuffer` on drop, which would close the recording
//! after each dispatch. The raw calls are safe because:
//! - All handles come from vulkane's safe RAII types (`.raw()`)
//! - The CB is in recording state for the entire batch
//! - Transient resources (descs, params buffers) are kept alive
//!   until the fence signals

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Duration;
use vulkane::safe::*;
use vulkane::raw::bindings::*;

/// Host-side op timing.
#[derive(Default)]
pub struct OpStats {
    inner: RefCell<HashMap<&'static str, OpStatEntry>>,
}

#[derive(Default, Clone, Copy)]
pub struct OpStatEntry {
    pub count: u64,
    pub total_ns: u128,
}

impl OpStats {
    pub fn record(&self, name: &'static str, elapsed: Duration) {
        let mut map = self.inner.borrow_mut();
        let e = map.entry(name).or_default();
        e.count += 1;
        e.total_ns += elapsed.as_nanos();
    }

    pub fn snapshot(&self) -> Vec<(&'static str, OpStatEntry)> {
        let map = self.inner.borrow();
        let mut v: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        v.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));
        v
    }

    pub fn reset(&self) {
        self.inner.borrow_mut().clear();
    }
}

pub(crate) struct Recorder {
    pub pool: CommandPool,
    /// The single CB being recorded into for the current batch.
    /// `None` when no batch is active.
    batch_cb: Option<CommandBuffer>,
    /// Transient resources (params uniform buffers, shape/strides
    /// buffers) from all dispatches in the current batch. Dropped
    /// after the fence signals.
    batch_transients: Vec<(Buffer, Allocation)>,
    /// Descriptor sets from all dispatches. Must survive until GPU
    /// consumes the CB.
    batch_descs: Vec<DescriptorSet>,
    /// Number of dispatches recorded in the current batch.
    pub(crate) batch_count: usize,
    /// VkBuffer handles written by dispatches since the last barrier.
    /// Used for dependency-aware barrier placement: a barrier is only
    /// inserted when a dispatch READS a buffer in this set.
    dirty_buffers: std::collections::HashSet<u64>,
}

/// Max dispatches per batch CB. Keeps each GPU submission well
/// under the WDDM TDR timeout (~2s). At ~0.5ms GPU time per
/// dispatch, 500 dispatches ≈ 0.25s — safe margin.
const BATCH_LIMIT: usize = 500;

impl Recorder {
    pub fn new(device: &Device, queue_family: u32) -> Result<Self> {
        Ok(Self {
            pool: CommandPool::new(device, queue_family)?,
            batch_cb: None,
            batch_transients: Vec::new(),
            batch_descs: Vec::new(),
            batch_count: 0,
            dirty_buffers: std::collections::HashSet::new(),
        })
    }

    /// Record a compute dispatch into the current batch CB.
    /// If no batch CB exists, allocates one and begins recording.
    /// Only inserts a pipeline barrier when a READ buffer overlaps
    /// with a previously-written (dirty) buffer — independent ops
    /// can overlap on the GPU without barriers.
    pub fn record_batch_dispatch(
        &mut self,
        device: &Device,
        pipeline: &ComputePipeline,
        pipe_layout: &PipelineLayout,
        desc: DescriptorSet,
        groups: (u32, u32, u32),
        transient_buffers: Vec<(Buffer, Allocation)>,
        read_bufs: &[u64],
        write_bufs: &[u64],
    ) -> Result<()> {
        if self.batch_cb.is_none() {
            let cmd = self.pool.allocate_primary()?;
            let dt = device.dispatch();
            unsafe {
                let begin = dt.vkBeginCommandBuffer
                    .ok_or(Error::MissingFunction("vkBeginCommandBuffer"))?;
                let info = VkCommandBufferBeginInfo {
                    sType: VkStructureType::STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    ..Default::default()
                };
                let r = begin(cmd.raw(), &info);
                if r != VkResult::SUCCESS { return Err(Error::Vk(r)); }
            }
            self.batch_cb = Some(cmd);
        }
        let cmd_handle = self.batch_cb.as_ref().unwrap().raw();
        let dt = device.dispatch();

        unsafe {
            // Dependency-aware barrier: only insert when this dispatch
            // reads a buffer that was written by a prior dispatch
            // without an intervening barrier. Independent ops skip
            // the barrier and can overlap on the GPU.
            let needs_barrier = read_bufs.iter().any(|b| self.dirty_buffers.contains(b));
            if needs_barrier {
                if let Some(barrier_fn) = dt.vkCmdPipelineBarrier {
                    let mem_barrier = VkMemoryBarrier {
                        sType: VkStructureType::STRUCTURE_TYPE_MEMORY_BARRIER,
                        pNext: std::ptr::null(),
                        srcAccessMask: 0x40, // VK_ACCESS_SHADER_WRITE_BIT
                        dstAccessMask: 0x20 | 0x40, // SHADER_READ | SHADER_WRITE
                    };
                    barrier_fn(
                        cmd_handle,
                        0x800, // VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                        0x800,
                        0,
                        1, &mem_barrier,
                        0, std::ptr::null(),
                        0, std::ptr::null(),
                    );
                }
                self.dirty_buffers.clear();
            }

            // Bind pipeline.
            if let Some(bind_fn) = dt.vkCmdBindPipeline {
                bind_fn(
                    cmd_handle,
                    VkPipelineBindPoint::PIPELINE_BIND_POINT_COMPUTE,
                    pipeline.raw(),
                );
            }

            // Bind descriptor set.
            let desc_handle = desc.raw();
            if let Some(bind_ds) = dt.vkCmdBindDescriptorSets {
                bind_ds(
                    cmd_handle,
                    VkPipelineBindPoint::PIPELINE_BIND_POINT_COMPUTE,
                    pipe_layout.raw(),
                    0, 1, &desc_handle,
                    0, std::ptr::null(),
                );
            }

            // Dispatch.
            if let Some(dispatch_fn) = dt.vkCmdDispatch {
                dispatch_fn(cmd_handle, groups.0, groups.1, groups.2);
            }
        }

        self.batch_transients.extend(transient_buffers);
        self.batch_descs.push(desc);
        self.batch_count += 1;
        // Mark write buffers as dirty for dependency tracking.
        for &wb in write_bufs {
            self.dirty_buffers.insert(wb);
        }
        Ok(())
    }

    /// True if the batch should be flushed (hit the per-batch limit).
    pub fn should_flush(&self) -> bool {
        self.batch_count >= BATCH_LIMIT
    }

    /// End the current batch CB recording, submit it, and wait for
    /// the GPU to finish. Drops all transient resources afterward.
    pub fn flush_batch(
        &mut self,
        device: &Device,
        queue: &Queue,
        queue_family: u32,
    ) -> Result<()> {
        let Some(cmd) = self.batch_cb.take() else {
            return Ok(());
        };
        let dt = device.dispatch();

        // End recording.
        unsafe {
            let end = dt.vkEndCommandBuffer
                .ok_or(Error::MissingFunction("vkEndCommandBuffer"))?;
            let r = end(cmd.raw());
            if r != VkResult::SUCCESS { return Err(Error::Vk(r)); }
        }

        // Submit with a fence and wait.
        let fence = Fence::new(device)?;
        queue.submit(&[&cmd], Some(&fence))?;
        fence.wait(u64::MAX)?;

        // Drop transient resources now that the GPU is done.
        self.batch_transients.clear();
        self.batch_descs.clear();
        self.batch_count = 0;
        self.dirty_buffers.clear();

        // Recycle the command pool to release CB backing memory.
        drop(cmd);
        self.pool = CommandPool::new(device, queue_family)?;
        Ok(())
    }

    /// Drain without submitting (for cleanup).
    pub fn drain(&mut self, device: &Device, queue_family: u32) -> Result<()> {
        if self.batch_cb.is_some() {
            // There's an active batch — need to end + discard it.
            // End the recording so the CB transitions out of recording
            // state before we drop it.
            let cmd = self.batch_cb.take().unwrap();
            let dt = device.dispatch();
            unsafe {
                if let Some(end) = dt.vkEndCommandBuffer {
                    end(cmd.raw());
                }
            }
            drop(cmd);
        }
        self.batch_transients.clear();
        self.batch_descs.clear();
        self.batch_count = 0;
        self.dirty_buffers.clear();
        self.pool = CommandPool::new(device, queue_family)?;
        Ok(())
    }
}

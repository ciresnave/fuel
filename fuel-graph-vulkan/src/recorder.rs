//! Asynchronous command buffer recorder + op timing for VulkanBackend.
//!
//! Replaces the synchronous-per-op `queue.one_shot` pattern with:
//!
//! 1. Allocate a small command buffer per op.
//! 2. Record the op with a leading compute→compute memory barrier so
//!    any prior compute writes are visible to this op's reads.
//! 3. Submit with no fence — `vkQueueSubmit` is asynchronous by
//!    default; the CPU returns as soon as the driver accepts the work.
//! 4. Push the CB plus any transient per-op resources (uniform
//!    buffers, shape/strides buffers, descriptor sets) onto a pending
//!    list so they outlive the in-flight GPU work.
//!
//! A single compute queue executes submissions in order, so there is
//! no need for semaphores between ops. The CPU's wait is deferred
//! until something actually needs the results — today that's always
//! `download()`, which issues its D2H copy via `queue.one_shot`. Since
//! that copy submit follows all our async submits on the same queue,
//! `one_shot`'s wait-on-fence drains every prior submit too. At that
//! point it is safe to drop the pending list and recycle the pool.
//!
//! The recorder owns one `CommandPool` at a time. When we drain, we
//! drop the pool entirely (which frees any memory it had allocated for
//! CB backing storage) and make a fresh one. This keeps the per-op
//! steady-state footprint bounded by what's in flight between syncs.

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Duration;
use vulkane::safe::*;

/// Host-side op timing. Measures the CPU cost of each dispatch
/// (descriptor alloc + record + submit) — NOT actual GPU execution
/// time, which would require Vulkan timestamp queries. Host cost is
/// what matters when the bottleneck is submission overhead; GPU cost
/// matters when the GPU is actually the critical path. We start with
/// the host-side numbers because they're free and they answer
/// "is submission overhead dominating our forward pass?" directly.
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

    /// Snapshot the current counts sorted by total time descending,
    /// suitable for printing a "where did the CPU go?" report.
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

/// Keeps transient resources alive for the lifetime of their in-flight
/// command buffer. The fields are never read — they exist so Drop
/// fires at the right moment (after the next sync point).
#[allow(dead_code)]
pub(crate) struct PendingSubmit {
    pub cmd: CommandBuffer,
    // Uniform / shape-strides buffers used by this CB's descriptors.
    // Each tuple is a Buffer + its sub-allocation from the shared
    // allocator. Dropping returns the sub-allocation to its pool.
    pub transient_buffers: Vec<(Buffer, Allocation)>,
    // The descriptor set itself — must survive until the GPU consumes it.
    pub transient_desc: Option<DescriptorSet>,
}

pub(crate) struct Recorder {
    pub pool: CommandPool,
    pub pending: Vec<PendingSubmit>,
    /// Monotonic timeline semaphore used to order GPU work across
    /// submissions. On NVIDIA, relying on a `vkCmdPipelineBarrier` at
    /// the start of each submitted CB to synchronize with writes from
    /// previously-submitted CBs is not reliable — we empirically
    /// reproduced `ERROR_DEVICE_LOST` at any queue depth ≥ 2. A
    /// timeline semaphore chain (each submit waits for the prior
    /// counter value and signals the next) is the spec-canonical way
    /// to synchronize cross-submission compute work.
    pub timeline: Semaphore,
    /// The counter value the most recent submit was configured to
    /// signal. Next submit waits for this and signals `counter + 1`.
    pub counter: u64,
}

impl Recorder {
    pub fn new(device: &Device, queue_family: u32) -> Result<Self> {
        Ok(Self {
            pool: CommandPool::new(device, queue_family)?,
            pending: Vec::new(),
            timeline: Semaphore::timeline(device, 0)?,
            counter: 0,
        })
    }

    /// Called by the backend's `download` path after the one_shot D2H
    /// copy has completed. Every prior async submit on this queue is
    /// now done on the GPU, so we can safely drop every CB + its
    /// transient resources, and replace the pool to release any
    /// accumulated backing memory. We also reset the timeline
    /// semaphore's counter tracking — the semaphore object itself
    /// stays alive; its internal counter continues monotonically.
    pub fn drain(&mut self, device: &Device, queue_family: u32) -> Result<()> {
        self.pending.clear();
        self.pool = CommandPool::new(device, queue_family)?;
        Ok(())
    }
}

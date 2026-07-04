//! Per-call workspace allocation for baracuda kernels.
//!
//! Each baracuda kernel has a `_workspace_size(...)` query that
//! reports its required scratch in bytes. The launch site allocates a
//! fresh `DeviceBuffer<u8>` of that size, passes its pointer + length,
//! and drops it when the kernel returns.
//!
//! ## Pooling (deferred)
//!
//! A per-stream scratch pool — reuse one large device buffer across
//! kernel launches — is an obvious optimization but not yet
//! implemented. Today's alloc-per-call model is correct, the typical
//! transformer launch has O(layers × heads) kernel invocations which
//! is bounded, and `cuMemAlloc` is fast enough on modern drivers that
//! this hasn't shown up as a measurable hotspot. When it does, the
//! pool lives here.
//!
//! ## Grow-only per-device workspace cache ([`WorkspaceCache`])
//!
//! The FIRST consumer of a cache is the flash_decoding wrapper
//! (`baracuda/attention.rs`): a decode session sizes its workspace at a
//! FIXED KV capacity, so the per-step workspace is byte-identical every
//! step. [`WorkspaceCache`] holds ONE grow-only device buffer per device
//! (shared across `CudaDevice` clones via an `Arc`, matching the device's
//! other cached resources); after the first step the cache converges to a
//! single allocation and every later step reuses it. It never shrinks — a
//! larger request grows the held buffer, a smaller one reuses the bigger
//! one — so a stable-capacity decode loop allocates exactly once.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use baracuda_driver::DeviceBuffer;
use fuel_ir::Result;

use crate::CudaDevice;

/// Workspace buffer for one kernel launch.
///
/// Holds the underlying `DeviceBuffer<u8>` so it stays live for the
/// duration of the launch (the kernel writes into it as scratch).
/// Drop frees the device memory; no manual cleanup needed.
pub struct Workspace {
    buf: Option<DeviceBuffer<u8>>,
    bytes: usize,
}

impl Workspace {
    /// Allocate a fresh workspace of `bytes` bytes on `device`. When
    /// `bytes == 0` returns a no-op workspace whose `as_raw` is a
    /// null pointer — matches baracuda's "no scratch needed" contract.
    pub fn alloc(device: &CudaDevice, bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Ok(Self { buf: None, bytes: 0 });
        }
        let buf = device.alloc_zeros::<u8>(bytes)?;
        Ok(Self {
            buf: Some(buf),
            bytes,
        })
    }

    /// Raw device pointer for the kernel-launch ABI. `null` when
    /// `bytes == 0`.
    pub fn as_raw(&self) -> *mut std::ffi::c_void {
        match self.buf.as_ref() {
            Some(b) => b.as_raw().0 as *mut std::ffi::c_void,
            None => std::ptr::null_mut(),
        }
    }

    /// Byte size — what the kernel sees as `workspace_bytes`.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// A per-device, grow-only workspace cache for baracuda kernel scratch.
///
/// Holds a single device buffer that is reused across launches on the same
/// device. GROW-ONLY: a request for at most the held buffer's size reuses it
/// (no allocation); a larger request reallocates to the larger size and the
/// old buffer is dropped. It never shrinks, so a fixed-capacity decode loop
/// (whose per-step workspace size is constant) allocates exactly ONCE and
/// reuses thereafter.
///
/// Shared across `CudaDevice` clones behind an `Arc` on the device, so it is
/// genuinely per-device. The internal `Mutex` is held for the duration of the
/// caller's launch closure ([`WorkspaceCache::with`]) so the scratch is never
/// concurrently reused — Fuel dispatches one stream per device, so there is no
/// contention on the common path.
pub struct WorkspaceCache {
    /// The held grow-only buffer (`None` until the first non-zero request).
    held: Mutex<Option<DeviceBuffer<u8>>>,
    /// Number of device allocations performed (the first fill + each grow).
    /// A pure reuse does NOT bump it. Test-observable proof of reuse.
    allocations: AtomicU64,
}

impl Default for WorkspaceCache {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkspaceCache {
    /// A fresh, empty cache (no buffer held, zero allocations).
    pub fn new() -> Self {
        Self {
            held: Mutex::new(None),
            allocations: AtomicU64::new(0),
        }
    }

    /// The grow-only reuse decision: reuse the held buffer iff it exists and
    /// is at least as large as the request. `None` (nothing held) ⇒ allocate;
    /// a larger request ⇒ allocate (grow); an equal-or-smaller request ⇒
    /// reuse (never shrink). Pure — the unit-testable core of the cache.
    fn should_reuse(held_bytes: Option<usize>, requested: usize) -> bool {
        matches!(held_bytes, Some(held) if held >= requested)
    }

    /// Run `f` with a workspace of at least `bytes` bytes: `(ptr, bytes)`.
    ///
    /// The held buffer is grown if needed (bumping the allocation counter),
    /// then its pointer + the *requested* byte length are handed to `f`. The
    /// lock is held across `f` so the scratch stays live and is never reused
    /// concurrently. `bytes == 0` yields a null pointer + 0 (baracuda's "no
    /// scratch" contract) and touches neither the buffer nor the counter.
    pub fn with<R>(
        &self,
        device: &CudaDevice,
        bytes: usize,
        f: impl FnOnce(*mut std::ffi::c_void, usize) -> R,
    ) -> Result<R> {
        if bytes == 0 {
            return Ok(f(std::ptr::null_mut(), 0));
        }
        let mut guard = self.held.lock().expect("workspace cache poisoned");
        let held_bytes = guard.as_ref().map(DeviceBuffer::len);
        if !Self::should_reuse(held_bytes, bytes) {
            // Grow (or first fill): allocate the requested size and replace.
            *guard = Some(device.alloc_zeros::<u8>(bytes)?);
            self.allocations.fetch_add(1, Ordering::Relaxed);
        }
        let buf = guard.as_ref().expect("buffer present after ensure");
        let ptr = buf.as_raw().0 as *mut std::ffi::c_void;
        Ok(f(ptr, bytes))
    }

    /// Number of device allocations performed so far (first fill + grows).
    /// A pure reuse does not increment it — the test observable proving the
    /// second same-capacity request did not allocate anew.
    pub fn allocation_count(&self) -> u64 {
        self.allocations.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::WorkspaceCache;

    /// BORN-RED: the grow-only reuse decision. A second request at the SAME
    /// capacity must REUSE the held buffer (no new allocation); a larger
    /// request grows; a smaller one reuses the bigger buffer (never shrink).
    ///
    /// RED with the pre-cache stub (`should_reuse` always `false` ⇒ every call
    /// allocates); GREEN once the grow-only comparison lands.
    #[test]
    fn grow_only_reuse_decision() {
        // Nothing held yet ⇒ must allocate the first time.
        assert!(
            !WorkspaceCache::should_reuse(None, 128),
            "first request (nothing held) must allocate",
        );
        // Second call at the SAME capacity ⇒ reuse, do NOT allocate anew.
        assert!(
            WorkspaceCache::should_reuse(Some(128), 128),
            "a same-capacity request must reuse the held buffer",
        );
        // A larger request ⇒ grow (allocate the bigger size).
        assert!(
            !WorkspaceCache::should_reuse(Some(128), 256),
            "a larger request must grow (allocate)",
        );
        // A smaller request ⇒ reuse the bigger buffer (grow-only, never shrink).
        assert!(
            WorkspaceCache::should_reuse(Some(256), 128),
            "a smaller request must reuse the larger held buffer",
        );
    }
}

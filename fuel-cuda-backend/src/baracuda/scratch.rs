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

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use baracuda_driver::DeviceBuffer;
use fuel_ir::Result;

use crate::CudaDevice;

/// Grow-only reuse decision shared by every per-device scratch cache in this
/// module ([`WorkspaceCache`], [`RopeTableCache`]): reuse the held buffer iff
/// it exists and is at least as large as the request. `None` (nothing held) ⇒
/// allocate; a larger request ⇒ allocate (grow); an equal-or-smaller request ⇒
/// reuse (never shrink). Pure — the unit-testable core of every cache here.
pub(crate) fn grow_only_should_reuse(held_bytes: Option<usize>, requested: usize) -> bool {
    matches!(held_bytes, Some(held) if held >= requested)
}

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

    /// The grow-only reuse decision — delegates to the module-level
    /// [`grow_only_should_reuse`] (shared with [`RopeTableCache`]). Kept as an
    /// associated fn so the existing `grow_only_reuse_decision` unit test still
    /// names it directly.
    fn should_reuse(held_bytes: Option<usize>, requested: usize) -> bool {
        grow_only_should_reuse(held_bytes, requested)
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

/// Which of the two independent RoPE-table slots a narrow targets. `cos` and
/// `sin` MUST use DIFFERENT slots: the fused `rope_apply` launch reads both
/// tables simultaneously, so a single shared buffer would make them alias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeTableSlot {
    /// The cosine table slot.
    Cos,
    /// The sine table slot.
    Sin,
}

/// Per-device, grow-only cache of the two narrowed RoPE tables (cos, sin) for
/// the fused `FusedOps::ROPE` CUDA candidate.
///
/// The fused driver narrows Fuel's FULL-WIDTH `[seq, head_dim]` cos/sin down to
/// baracuda's HALF-WIDTH `[seq, head_dim/2]` via a D2D `cuMemcpy2DAsync` before
/// each launch (see `baracuda/attention.rs`). Narrowing into a FRESH device
/// buffer every call — the original posture — allocates inside a CapturedRun
/// capture scope, which the zero-alloc-during-capture invariant forbids. This
/// cache holds ONE grow-only `Arc<DeviceBuffer<u8>>` PER SLOT (cos, sin): a
/// stable-capacity decode loop sizes both tables identically every step, so
/// after the first step each slot converges to a single allocation and every
/// later narrow reuses it (the D2D copy targets a fixed device address — zero
/// alloc during capture).
///
/// Two SEPARATE slots (not one buffer) because the launch consumes cos AND sin
/// at once: sharing one buffer would make them alias. The returned `Arc` keeps
/// a slot's buffer live for the caller (and, crucially, for the async kernel
/// that reads it) even after the caller's `CudaStorageBytes` view drops and
/// even if a later, larger narrow grows that slot and replaces the held `Arc`.
///
/// Shared across `CudaDevice` clones behind an `Arc` on the device (per-device,
/// exactly like [`WorkspaceCache`]). Fuel dispatches one stream per device, so
/// the two per-slot `Mutex`es never actually contend on the common path; each
/// is held only for the grow-or-reuse decision, not across the launch (the held
/// `Arc` — not the lock — is what keeps the buffer live across the async copy +
/// kernel).
pub struct RopeTableCache {
    /// The cos slot's held grow-only buffer (`None` until the first narrow).
    cos: Mutex<Option<Arc<DeviceBuffer<u8>>>>,
    /// The sin slot's held grow-only buffer (`None` until the first narrow).
    sin: Mutex<Option<Arc<DeviceBuffer<u8>>>>,
    /// Device allocations across BOTH slots (first fill + each grow). A pure
    /// reuse does NOT bump it — the test-observable proof of zero-alloc reuse.
    allocations: AtomicU64,
}

impl Default for RopeTableCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RopeTableCache {
    /// A fresh, empty cache (no buffers held, zero allocations).
    pub fn new() -> Self {
        Self {
            cos: Mutex::new(None),
            sin: Mutex::new(None),
            allocations: AtomicU64::new(0),
        }
    }

    fn slot(&self, slot: RopeTableSlot) -> &Mutex<Option<Arc<DeviceBuffer<u8>>>> {
        match slot {
            RopeTableSlot::Cos => &self.cos,
            RopeTableSlot::Sin => &self.sin,
        }
    }

    /// Ensure `slot` holds a buffer of at least `bytes` bytes (grow-only) and
    /// return an `Arc` clone of it for the caller to narrow into.
    ///
    /// A same-or-smaller request reuses the held buffer (no device allocation,
    /// no counter bump); a larger request — or an empty slot — allocates the
    /// requested size, replaces the held `Arc`, and bumps `allocation_count`.
    /// The lock is released before the caller does its D2D copy; the returned
    /// `Arc` (plus the cache's own retained `Arc`) is what keeps the buffer
    /// live across that copy and the subsequent async kernel, so no lock need
    /// be held for the launch. Callers must guarantee `bytes > 0` (a
    /// zero-length narrow is short-circuited upstream before reaching here).
    pub fn ensure(
        &self,
        device: &CudaDevice,
        slot: RopeTableSlot,
        bytes: usize,
    ) -> Result<Arc<DeviceBuffer<u8>>> {
        let mut guard = self.slot(slot).lock().expect("rope table cache poisoned");
        let held = guard.as_ref().map(|b| b.len());
        if !grow_only_should_reuse(held, bytes) {
            *guard = Some(Arc::new(device.alloc_zeros::<u8>(bytes)?));
            self.allocations.fetch_add(1, Ordering::Relaxed);
        }
        Ok(guard.as_ref().expect("buffer present after ensure").clone())
    }

    /// Total device allocations across BOTH slots so far (first fill + grows).
    /// A pure reuse does not increment it — the observable proving a
    /// same-capacity narrow reused its slot rather than allocating anew (the
    /// CapturedRun zero-alloc-during-capture invariant).
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

    /// The shared grow-only decision consumed by BOTH caches. Same table as
    /// `grow_only_reuse_decision` but asserting the free function directly —
    /// `RopeTableCache::ensure` allocates iff this returns `false`, so the
    /// cache's zero-alloc reuse property is exactly this predicate.
    #[test]
    fn shared_grow_only_decision_table() {
        use super::grow_only_should_reuse;
        assert!(!grow_only_should_reuse(None, 128), "empty slot must allocate");
        assert!(grow_only_should_reuse(Some(128), 128), "same size must reuse");
        assert!(!grow_only_should_reuse(Some(128), 256), "larger must grow");
        assert!(grow_only_should_reuse(Some(256), 128), "smaller must reuse (never shrink)");
    }

    /// GPU: the two-slot cache's zero-alloc reuse + non-aliasing invariants —
    /// the CapturedRun-critical properties `grow_only_reuse_decision` can only
    /// assert in the abstract. Requires a live device (it actually allocates),
    /// so `#[ignore]`'d like every other GPU test in this crate.
    ///
    /// Asserts:
    /// 1. First `ensure(Cos, N)` allocates (count 0 → 1).
    /// 2. A SECOND same-size `ensure(Cos, N)` REUSES (count stays 1) and returns
    ///    the SAME device pointer — the zero-alloc-during-capture property.
    /// 3. `ensure(Sin, N)` allocates a SEPARATE buffer (count 1 → 2) whose
    ///    pointer DIFFERS from the cos slot's — cos/sin never alias.
    /// 4. A larger `ensure(Cos, 2N)` grows (count 2 → 3); a smaller follow-up
    ///    reuses the grown buffer (count stays 3).
    #[test]
    #[ignore = "requires a live CUDA device"]
    fn rope_table_cache_reuses_and_does_not_alias() {
        use super::{RopeTableCache, RopeTableSlot};
        use crate::CudaDevice;
        let Ok(device) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        let cache = RopeTableCache::new();
        const N: usize = 2 * 2 * 4; // seq=2, head_dim/2=2, F32 → 16 bytes

        let cos1 = cache.ensure(&device, RopeTableSlot::Cos, N).expect("cos ensure 1");
        assert_eq!(cache.allocation_count(), 1, "first cos ensure allocates");

        let cos2 = cache.ensure(&device, RopeTableSlot::Cos, N).expect("cos ensure 2");
        assert_eq!(cache.allocation_count(), 1, "same-size cos ensure must REUSE (zero alloc)");
        assert_eq!(
            cos1.as_raw().0, cos2.as_raw().0,
            "reused cos slot must hand back the SAME device address",
        );

        let sin1 = cache.ensure(&device, RopeTableSlot::Sin, N).expect("sin ensure 1");
        assert_eq!(cache.allocation_count(), 2, "first sin ensure allocates a separate buffer");
        assert_ne!(
            cos1.as_raw().0, sin1.as_raw().0,
            "cos and sin slots must NOT alias (the launch reads both at once)",
        );

        let _cos_grown = cache.ensure(&device, RopeTableSlot::Cos, 2 * N).expect("cos grow");
        assert_eq!(cache.allocation_count(), 3, "a larger cos ensure must grow (allocate)");
        let _cos_smaller = cache.ensure(&device, RopeTableSlot::Cos, N).expect("cos reuse grown");
        assert_eq!(cache.allocation_count(), 3, "a smaller cos ensure reuses the grown buffer");
    }
}

//! Backend storage capability traits — the object-safe contract surface
//! every backend implements.
//!
//! After step 8 of the backend-agnostic refactor (2026-04-30), the static
//! `BackendStorage`/`BackendDevice` generic traits were deleted: every backend
//! now implements [`DynBackendStorage`](crate::dyn_backend::DynBackendStorage)
//! and [`DynBackendDevice`](crate::dyn_backend::DynBackendDevice) directly.
//!
//! What remains here is [`HostStorage`], the orthogonal capability trait
//! marking storage types whose data lives in host-addressable RAM, plus
//! the Phase 7.5 storage-unification trait additions ([`BackendStorage`],
//! [`BackendCapabilityProvider`], [`BackendRuntime`]). Their *data* types —
//! `SubstrateClass`, `TransferPath`, `BackendCapabilities`, `FitStatus` —
//! live in [`fuel_ir::backend`].
use fuel_ir::backend::{BackendCapabilities, FitStatus};
use fuel_ir::{HostBuffer, HostBufferRef, Result};

/// Capability trait for storage types whose data lives in
/// host-addressable RAM — i.e., the storage can be viewed as a
/// typed slice without a device-to-host copy.
///
/// This trait is orthogonal to `DynBackendStorage`: a storage type can
/// implement either, both, or neither.
///
/// Implementors:
///
/// - `CpuStorage` (owned `Vec<T>` via [`HostBuffer`])
/// - `MmappedHostStorage` — memory-mapped weights via `mmap2`
/// - `PinnedHostStorage` — page-locked memory for GPU DMA
/// - `SharedMemHostStorage` — inter-process shared memory
/// - `RemoteHostStorage` — network-accessible buffers for multi-host (future)
pub trait HostStorage {
    /// Borrow the underlying data as a [`HostBufferRef`] (zero-copy).
    fn as_host_buffer_ref(&self) -> Result<HostBufferRef<'_>>;

    /// Extract the underlying data as an owned [`HostBuffer`].
    ///
    /// Default impl materializes via `as_host_buffer_ref().to_owned()`,
    /// which is a full copy. Owned-buffer implementors should override to
    /// hand out the existing buffer without copying.
    fn into_host_buffer(self) -> Result<HostBuffer>
    where
        Self: Sized,
    {
        Ok(self.as_host_buffer_ref()?.to_owned())
    }
}

/// Phase 7.5 storage unification — see [docs/storage-unification.md].
///
/// Minimum contract every per-backend storage type implements. The
/// trait defines only the universally-required surface today
/// (`len_bytes`); allocation, copy-from-other-backend, and the
/// capability advertisement land in subsequent phases as the rest of
/// the design fills in.
///
/// Bounds:
///
/// - `Send + Sync` so storage handles can cross thread boundaries
///   (`Arc<RwLock<Storage>>` lives in graph slots accessed from
///   compiler + executor threads).
/// - `Debug` for diagnostic error messages and tracing.
///
/// Implementors:
///
/// - `fuel_cpu_backend::CpuStorageBytes` (Phase A3.0)
/// - `fuel_metal_backend::MetalStorageBytes` (Phase A3.1)
/// - `fuel_cuda_backend::CudaStorageBytes` (Phase A3.2)
/// - `fuel_vulkan_backend::VulkanStorageBytes` (Phase A3.3)
pub trait BackendStorage: Send + Sync + std::fmt::Debug {
    /// Total addressable byte count, regardless of dtype.
    ///
    /// The dtype tag lives on the `Storage` wrapper (in fuel-storage),
    /// not on the variant — `len_bytes` is dtype-agnostic.
    fn len_bytes(&self) -> usize;
}

/// Backends implement this to advertise their capabilities at
/// registration time. Typical impl is on the backend's device type
/// (e.g., `CpuDevice`, `CudaDevice`); each device instance reports
/// what it can do.
///
/// The advertised [`BackendCapabilities`] *data* lives in
/// [`fuel_ir::backend`].
pub trait BackendCapabilityProvider {
    /// Snapshot of the backend's capabilities. Capabilities are
    /// static at backend instantiation — no runtime mutation, no
    /// versioning. Adding a new dtype or op to a backend requires
    /// recompiling Fuel.
    fn capabilities(&self) -> BackendCapabilities;
}

/// Runtime state every backend reports. Phase 5.1/5.2 substrate of
/// the picker arc — replaces backend-specific inherent methods (e.g.
/// `VulkanBackend::vram_budget`, future `BaracudaDevice::vram_free`)
/// with a uniform contract surface. See architecture v0.3
/// (`docs/architecture/05-backend-contract.md`) §Trait surface for
/// the full tiering.
///
/// # Honesty contract
///
/// `Option<u64>` returns let backends say "I genuinely can't measure
/// this" without forcing fabrications. Selectors MUST treat `None`
/// as "no signal — fall back to static cost," NEVER as "zero
/// available memory."
///
/// # Caching
///
/// Implementations are expected to be cheap to call (sub-millisecond
/// for cache hits, well under a microsecond on a hot path).
/// Implementations that wrap non-trivial queries (parsing
/// `/proc/meminfo`, OS syscalls) SHOULD internally cache results for
/// ~100ms to amortize the cost of selector polling.
pub trait BackendRuntime {
    /// Bytes currently available for new allocations on this
    /// backend's device. `None` when the backend genuinely cannot
    /// measure (no OS query exposed, no vendor API supports it).
    ///
    /// "Available" semantics are device-relative:
    ///
    /// - CPU backends report system-wide free memory (OS query).
    ///   Note this is shared with the whole OS — other processes
    ///   can inflate / deflate the value unpredictably. The signal
    ///   is noisier than per-process VRAM tracking; selectors
    ///   should weight it accordingly.
    /// - GPU backends report device-local free memory (driver query).
    ///   Driver estimates include this process, other processes,
    ///   and driver internals.
    /// - Reference / synthetic backends MAY return `Some(u64::MAX)`
    ///   to advertise "unbounded" capacity (never reports pressure).
    fn available_bytes(&self) -> Option<u64>;

    /// Total memory on this backend's device. Static after first
    /// call; implementations cache unconditionally. `None` for
    /// backends with unbounded notional capacity (e.g. Reference
    /// returns `Some(u64::MAX)`, or `None` if the backend prefers
    /// to advertise "unknowable").
    fn total_bytes(&self) -> Option<u64>;

    /// Predictive fit-check: would an allocation of `size` bytes
    /// likely succeed given current state? Returns [`FitStatus::Tight`]
    /// when projected post-allocation usage crosses the pressure
    /// threshold (default 0.85 of total).
    ///
    /// Default implementation derives the answer from
    /// [`Self::available_bytes`] + [`Self::total_bytes`]. Backends
    /// with native predictive APIs (Vulkan `VK_EXT_memory_budget`)
    /// override for accuracy — driver-level predictive checks can
    /// detect fragmentation that a simple bytes-available subtraction
    /// would miss.
    fn would_fit(&self, size: u64) -> FitStatus {
        match (self.available_bytes(), self.total_bytes()) {
            (Some(a), _) if size > a => FitStatus::WontFit,
            (Some(a), Some(t)) if t > 0 => {
                let post_used = t.saturating_sub(a.saturating_sub(size));
                if (post_used as f64) / (t as f64) > 0.85 {
                    FitStatus::Tight
                } else {
                    FitStatus::Comfortable
                }
            }
            _ => FitStatus::Unknown,
        }
    }

    /// Tier-2 upcast seam: a runtime handle that also exposes a
    /// deferred-execution queue model returns `Some(self)` so a
    /// selector holding only a `&dyn BackendRuntime` (the type the
    /// route picker's [`BackendRuntimeLookup`] hands out) can reach
    /// the [`BackendStreams`] live-load surface without naming the
    /// concrete handle type.
    ///
    /// Defaults to `None` — the honesty contract for Tier 2: a backend
    /// with no queue concept (CPU, Reference) is **not** a
    /// `BackendStreams` and says so, exactly as `available_bytes`
    /// returns `None` for "I can't measure this." Returning `Some` is
    /// a per-handle opt-in (see `DeviceRuntimeHandle` in
    /// `fuel-core::pipelined_bridge`), never forced on the base trait.
    ///
    /// (Step E Phase C / B1: the seam the load-aware selector — C2 —
    /// reads through. B1 wires the signal; nothing consults it yet.)
    fn as_backend_streams(&self) -> Option<&dyn BackendStreams> {
        None
    }
}

/// Tier-2 conditional contract — backends with a deferred-execution
/// (queue / stream / command-buffer) model expose live in-flight load
/// here. See architecture v0.4
/// (`docs/architecture/05-backend-contract.md` §Tier 2 → `BackendStreams`).
///
/// # Why a separate trait (the honesty contract)
///
/// The base [`BackendRuntime`] stays VRAM-only: a backend that
/// dispatches synchronously (CPU, Reference) has no queue depth to
/// report, and forcing a `pending_work_count` onto every backend would
/// fabricate a "0 in flight" that reads as a real signal. `BackendStreams`
/// is the Tier-2 extension only stream-like backends (CUDA streams,
/// Vulkan queues / command buffers) implement; selectors reach it via
/// [`BackendRuntime::as_backend_streams`] and treat its absence as
/// "no load signal — fall back to static cost," never as "idle."
///
/// # Where the count comes from (Step E Phase C / B1)
///
/// The in-flight count a selector needs is the **executor's own
/// submitted-but-not-drained async-op count**, not a driver query
/// (`cuStreamQuery` is a busy/idle bool, not a depth). Fuel tracks it
/// in a process-wide per-[`DeviceLocation`] atomic counter
/// (`fuel-dispatch::dispatch::inflight_count`) incremented when the
/// executor submits an async op and decremented when the completion
/// handle retires. A `BackendStreams` impl reads that counter for its
/// own device — see `DeviceRuntimeHandle` in `fuel-core::pipelined_bridge`.
pub trait BackendStreams: BackendRuntime {
    /// Number of slots (streams / queues / command buffers) currently
    /// busy with submitted-but-not-yet-finished work, for this handle's
    /// device. `None` when the backend dispatches synchronously and has
    /// no queue concept (selectors then have no load signal — fall back
    /// to static cost, never treat `None` as "idle").
    fn pending_work_count(&self) -> Option<u32>;

    /// Maximum concurrent in-flight work this backend advertises. Used
    /// by the runtime route picker for load tiering (`count /
    /// slot_capacity`) and dispatch lookahead sizing.
    fn slot_capacity(&self) -> u32;

    /// Block until all submitted work on this backend's slots has
    /// completed. Used at realize boundaries and by training-loop
    /// barriers (gradient accumulation, optimizer step).
    fn flush(&self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::backend::{BackendCapabilities, SubstrateClass};
    use fuel_ir::dispatch::OpKind;
    use fuel_ir::{BackendId, DType, DeviceLocation};
    use std::collections::HashSet;

    /// Smoke: the BackendCapabilityProvider trait is object-safe and
    /// implementable. (The `BackendCapabilities` *data* round-trip lives
    /// with the data type in `fuel-ir`.)
    #[test]
    fn capability_provider_is_implementable() {
        struct DummyDevice;
        impl BackendCapabilityProvider for DummyDevice {
            fn capabilities(&self) -> BackendCapabilities {
                BackendCapabilities {
                    backend_id: BackendId::Cpu,
                    device_location: DeviceLocation::Cpu,
                    op_dtype_support: HashSet::new(),
                    required_alignment: 1,
                    access_granularity_bits: 8,
                    transfer_paths: Vec::new(),
                    storage_substrate: SubstrateClass::HostBytes,
                }
            }
        }
        let d = DummyDevice;
        let caps = d.capabilities();
        assert_eq!(caps.backend_id, BackendId::Cpu);
        let _ = (OpKind::MatMul, DType::F32); // vocab reachable from the contract crate
    }

    // ===== BackendRuntime default `would_fit` impl =====

    /// Test harness: an in-memory backend with configurable available
    /// / total bytes for testing the default `would_fit` derivation.
    struct MockRuntime {
        available: Option<u64>,
        total: Option<u64>,
    }
    impl BackendRuntime for MockRuntime {
        fn available_bytes(&self) -> Option<u64> { self.available }
        fn total_bytes(&self) -> Option<u64> { self.total }
    }

    /// Allocation strictly larger than available bytes → `WontFit`.
    #[test]
    fn would_fit_wont_fit_when_size_exceeds_available() {
        let r = MockRuntime { available: Some(1_000), total: Some(10_000) };
        assert_eq!(r.would_fit(1_001), FitStatus::WontFit);
    }

    /// Small allocation that leaves usage well under the threshold
    /// → `Comfortable`.
    #[test]
    fn would_fit_comfortable_when_post_alloc_under_threshold() {
        // total=100, available=80 → currently using 20.
        // size=10 → post-used=30, post-used/total=0.3 < 0.85.
        let r = MockRuntime { available: Some(80), total: Some(100) };
        assert_eq!(r.would_fit(10), FitStatus::Comfortable);
    }

    /// Allocation that fits but pushes usage above the 0.85
    /// threshold → `Tight`.
    #[test]
    fn would_fit_tight_when_post_alloc_above_threshold() {
        // total=100, available=20 → currently using 80.
        // size=10 → post-used=90, post-used/total=0.9 > 0.85.
        let r = MockRuntime { available: Some(20), total: Some(100) };
        assert_eq!(r.would_fit(10), FitStatus::Tight);
    }

    /// `None` for either field → `Unknown`. Honest "I don't know,"
    /// not a false `Comfortable` or `WontFit`.
    #[test]
    fn would_fit_unknown_on_none_signals() {
        let r1 = MockRuntime { available: None, total: Some(100) };
        let r2 = MockRuntime { available: Some(50), total: None };
        let r3 = MockRuntime { available: None, total: None };
        assert_eq!(r1.would_fit(10), FitStatus::Unknown);
        assert_eq!(r2.would_fit(10), FitStatus::Unknown);
        assert_eq!(r3.would_fit(10), FitStatus::Unknown);
    }

    /// `total == 0` produces `Unknown` rather than a divide-by-zero
    /// or a spurious `Tight`. Defensive.
    #[test]
    fn would_fit_unknown_on_zero_total() {
        let r = MockRuntime { available: Some(0), total: Some(0) };
        assert_eq!(r.would_fit(0), FitStatus::Unknown);
    }

    /// Zero-byte allocation is always Comfortable on a non-saturated
    /// backend.
    #[test]
    fn would_fit_zero_byte_alloc() {
        let r = MockRuntime { available: Some(50), total: Some(100) };
        assert_eq!(r.would_fit(0), FitStatus::Comfortable);
    }

    // ===== Tier-2 BackendStreams + the as_backend_streams seam =====

    /// A base `BackendRuntime` with no queue model is NOT a
    /// `BackendStreams`: the default `as_backend_streams` returns `None`
    /// (the honesty contract — a selector must read "no load signal,"
    /// never a fabricated "idle"). `MockRuntime` does not implement
    /// `BackendStreams`, so it keeps the default.
    #[test]
    fn base_runtime_is_not_streams_by_default() {
        let r = MockRuntime { available: Some(50), total: Some(100) };
        assert!(
            r.as_backend_streams().is_none(),
            "a backend with no queue concept must report None via as_backend_streams",
        );
    }

    /// A handle that DOES implement `BackendStreams` and opts in via
    /// `as_backend_streams` is reachable through a bare
    /// `&dyn BackendRuntime` — the exact upcast the load-aware selector
    /// (C2) performs over the route picker's `BackendRuntimeLookup`
    /// handle. `pending_work_count` is the carried live-load value.
    #[test]
    fn streams_handle_reachable_through_runtime_upcast() {
        struct StreamingHandle {
            inflight: u32,
        }
        impl BackendRuntime for StreamingHandle {
            fn available_bytes(&self) -> Option<u64> {
                None
            }
            fn total_bytes(&self) -> Option<u64> {
                None
            }
            fn as_backend_streams(&self) -> Option<&dyn BackendStreams> {
                Some(self)
            }
        }
        impl BackendStreams for StreamingHandle {
            fn pending_work_count(&self) -> Option<u32> {
                Some(self.inflight)
            }
            fn slot_capacity(&self) -> u32 {
                4
            }
            fn flush(&self) -> Result<()> {
                Ok(())
            }
        }

        let handle = StreamingHandle { inflight: 3 };
        let as_runtime: &dyn BackendRuntime = &handle;
        let streams = as_runtime
            .as_backend_streams()
            .expect("streaming handle exposes BackendStreams");
        assert_eq!(streams.pending_work_count(), Some(3));
        assert_eq!(streams.slot_capacity(), 4);
        streams.flush().expect("flush is Ok");
    }
}

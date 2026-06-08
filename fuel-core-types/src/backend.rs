//! Backend storage capability traits.
//!
//! After step 8 of the backend-agnostic refactor (2026-04-30), the static
//! `BackendStorage` and `BackendDevice` traits were deleted: every backend
//! now implements [`DynBackendStorage`](crate::dyn_backend::DynBackendStorage)
//! and [`DynBackendDevice`](crate::dyn_backend::DynBackendDevice) directly.
//!
//! What remains here is [`HostStorage`], the orthogonal capability trait
//! marking storage types whose data lives in host-addressable RAM, plus
//! the Phase 7.5 storage-unification additions ([`BackendStorage`] trait,
//! [`BackendCapabilities`] / [`BackendCapabilityProvider`] for capability
//! advertisement, and [`TransferPath`] for inter-device transfers).
use crate::{dispatch::OpKind, probe::BackendId, DType, DeviceLocation, HostBuffer, HostBufferRef, Result};
use std::collections::HashSet;

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

// =============================================================================
// Phase 7.5 A4 — capability advertisement
// =============================================================================

/// Classifies the underlying allocator / pointer namespace a backend's
/// storage lives in. Two backends share a substrate class **iff their
/// storage variants are byte-compatible on the same device** — calling
/// one backend's kernel after the other's is a vtable swap, not a data
/// copy.
///
/// SystemTopology consumes this for its `shares_storage` predicate
/// (Phase 7.6 system-topology session, 2026-05-30). When both backends
/// declare the same `SubstrateClass` *and* target the same
/// [`DeviceLocation`], they can freely interleave on the same `Storage`
/// handle without an Op::Copy.
///
/// Forward-extension: a new backend that introduces a new pointer
/// namespace adds a new variant. The enum is `#[non_exhaustive]` so
/// downstream pattern matches stay forward-compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SubstrateClass {
    /// Host-addressable bytes — the CPU trio (Cpu / Aocl / Mkl) plus
    /// `Reference` all share this. A storage allocated as
    /// `CpuStorageBytes` flows through any of them without a copy.
    HostBytes,
    /// CUDA untyped device buffer. Two CUDA backends on the same
    /// `gpu_id` share; CUDA on `gpu_id=0` vs `gpu_id=1` do not.
    CudaUntyped,
    /// Vulkan device buffer (`VkBuffer`). Two Vulkan backends on the
    /// same physical device share; Vulkan vs CUDA on the same silicon
    /// do not (different pointer namespaces, separate allocators).
    VulkanBuffer,
    /// Metal device buffer (`MTLBuffer`). Reserved for future Metal
    /// backend wiring.
    MetalBuffer,
}

/// How bytes can move from one device to another. Backends advertise
/// the paths they support as the source; Router consumes this to
/// pick the cheapest available path between two specific devices.
///
/// Bandwidth + latency estimates land in A5 alongside Router
/// integration; for A4 the variants exist as discriminators only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferPath {
    /// No transfer needed (same device instance).
    SameDevice,
    /// Direct peer-to-peer (CUDA P2P, NVLink, AMD Infinity Fabric,
    /// GPUDirect). Requires explicit enablement on both ends.
    Peer,
    /// Zero-copy via shared memory mapping (UMA, ResizableBAR,
    /// dma-buf). Common on integrated GPUs and Apple Silicon.
    SharedMemory,
    /// Bulk transfer engine (cudaMemcpy, vkCmdCopyBuffer + staging
    /// buffer). The default for cross-device GPU↔GPU on standard
    /// hardware.
    DeviceCopy,
    /// Through CPU as intermediary. Universal fallback — every
    /// backend supports this. Highest latency but always available.
    HostStaging,
}

/// What a backend can do — advertised once at registration, consumed
/// by Router during dispatch planning.
///
/// The fields are facts: which (op, dtype) pairs the backend has
/// kernels for, what alignment its allocations satisfy, what
/// transfer paths it can use, etc. Routing policy (which backend
/// gets which op given input residency) lives in Router, not here.
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Identifies the backend variant (Cpu / Cuda / Vulkan / ...).
    pub backend_id: BackendId,
    /// Identifies the specific device within the backend's family
    /// (e.g., GPU 0 vs GPU 1 for CUDA).
    pub device_location: DeviceLocation,
    /// Set of `(op, dtype)` pairs this backend has kernels for.
    /// Routing checks `contains(&(op, dtype))` before dispatching.
    pub op_dtype_support: HashSet<(OpKind, DType)>,
    /// Required alignment in bytes for storage allocations on this
    /// backend. Router pads/repacks if a source storage doesn't
    /// meet the destination's alignment.
    pub required_alignment: usize,
    /// Smallest addressable unit in bits. Most CPUs/GPUs are byte-
    /// addressable (8); some accelerators are 32-bit-only or
    /// 128-bit-vector-only. Router routes around granularity
    /// limits or refuses to route there.
    pub access_granularity_bits: u32,
    /// Outbound transfer paths this backend supports as the source.
    /// Each entry is the destination's `DeviceLocation` plus the
    /// path that connects them. Router builds a transfer matrix
    /// from these entries at registration.
    pub transfer_paths: Vec<(DeviceLocation, TransferPath)>,
    /// Allocator / pointer-namespace class. SystemTopology's
    /// `shares_storage(a, b)` predicate is true iff both backends
    /// declare the same class **and** target the same device. CPU
    /// trio all declare [`SubstrateClass::HostBytes`]; CUDA declares
    /// [`SubstrateClass::CudaUntyped`]; Vulkan declares
    /// [`SubstrateClass::VulkanBuffer`].
    pub storage_substrate: SubstrateClass,
}

/// Backends implement this to advertise their capabilities at
/// registration time. Typical impl is on the backend's device type
/// (e.g., `CpuDevice`, `CudaDevice`); each device instance reports
/// what it can do.
pub trait BackendCapabilityProvider {
    /// Snapshot of the backend's capabilities. Capabilities are
    /// static at backend instantiation — no runtime mutation, no
    /// versioning. Adding a new dtype or op to a backend requires
    /// recompiling Fuel.
    fn capabilities(&self) -> BackendCapabilities;
}

/// Predictive fit-check result for a projected allocation. Consumed
/// by [`BackendRuntime::would_fit`] and read by Picker 2's pressure-
/// aware selectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitStatus {
    /// Allocation projected to fit comfortably (post-alloc usage
    /// below the pressure threshold, default 0.85 of total).
    Comfortable,
    /// Allocation projected to fit but leaves the device tight
    /// (post-alloc usage above the pressure threshold). Selectors
    /// should prefer a less-loaded co-located backend if available.
    Tight,
    /// Allocation projected NOT to fit (size > available bytes).
    /// Selectors should pick a different backend; planner-level
    /// surfaces a typed error if no alternative.
    WontFit,
    /// Backend cannot answer — `available_bytes` returned `None`.
    /// Selectors fall back to static cost.
    Unknown,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceLocation;

    /// Smoke: BackendCapabilities can be constructed and queried.
    #[test]
    fn capability_construction_round_trip() {
        let caps = BackendCapabilities {
            backend_id: BackendId::Cpu,
            device_location: DeviceLocation::Cpu,
            op_dtype_support: [
                (OpKind::MatMul, DType::F32),
                (OpKind::MatMul, DType::F64),
                (OpKind::AddElementwise, DType::F32),
            ]
            .into_iter()
            .collect(),
            required_alignment: 64,
            access_granularity_bits: 8,
            transfer_paths: vec![(DeviceLocation::Cpu, TransferPath::SameDevice)],
            storage_substrate: SubstrateClass::HostBytes,
        };
        assert_eq!(caps.backend_id, BackendId::Cpu);
        assert_eq!(caps.required_alignment, 64);
        assert_eq!(caps.access_granularity_bits, 8);
        assert!(caps.op_dtype_support.contains(&(OpKind::MatMul, DType::F32)));
        assert!(!caps.op_dtype_support.contains(&(OpKind::MatMul, DType::BF16)));
    }

    /// Smoke: the BackendCapabilityProvider trait is object-safe and
    /// implementable.
    #[test]
    fn capability_provider_is_implementable() {
        struct DummyDevice;
        impl BackendCapabilityProvider for DummyDevice {
            fn capabilities(&self) -> BackendCapabilities {
                BackendCapabilities {
                    backend_id: BackendId::Reference,
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
        assert_eq!(caps.backend_id, BackendId::Reference);
    }

    /// Smoke: TransferPath enum is comparable + hashable.
    #[test]
    fn transfer_path_traits() {
        let a = TransferPath::SameDevice;
        let b = TransferPath::SameDevice;
        let c = TransferPath::HostStaging;
        assert_eq!(a, b);
        assert_ne!(a, c);
        // Hashable — can be a key in HashSet/HashMap.
        let mut set = HashSet::new();
        set.insert(TransferPath::Peer);
        set.insert(TransferPath::Peer);  // dup
        set.insert(TransferPath::DeviceCopy);
        assert_eq!(set.len(), 2);
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
}

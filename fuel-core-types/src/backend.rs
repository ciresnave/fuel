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
/// - `fuel_graph_vulkan::VulkanStorageBytes` (Phase A3.3)
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
}

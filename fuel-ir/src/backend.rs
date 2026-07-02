//! Backend capability **data** types.
//!
//! After step 8 of the backend-agnostic refactor (2026-04-30), the static
//! `BackendStorage`/`BackendDevice` generic traits were deleted; every backend
//! implements `DynBackendStorage`/`DynBackendDevice` directly. As of B0.3 the
//! object-safe contract *traits* (`HostStorage`, `BackendStorage`,
//! `BackendCapabilityProvider`, `BackendRuntime`) moved to the
//! `fuel-backend-contract` crate (it sits above this vocabulary crate, below the
//! backends). What remains here is the **data** they traffic in: the capability
//! advertisement ([`BackendCapabilities`] / [`SubstrateClass`] / [`TransferPath`])
//! and the predictive [`FitStatus`].
use crate::{dispatch::OpKind, probe::BackendId, DType, DeviceLocation};
use std::collections::HashSet;

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
    /// Peak sustained compute throughput as a Layer-1 cost prior, in
    /// FLOPs per nanosecond (numerically GFLOP/s). The ranker's
    /// `composite_ns` divides a kernel's FLOP count by this to estimate
    /// compute time, so a higher value models a genuinely faster
    /// backend — the placement DP prefers it for large parallel work
    /// while transfer cost still keeps small ops local. The historical
    /// neutral prior was 1.0 for every backend (1 FLOP ≈ 1 ns); real
    /// per-backend tiers replace it, and the Judge's Layer-2 latency
    /// measurements refine the effective figure per (op, dtype, size)
    /// cell. Never zero/negative — consumers guard and fall back to the
    /// CPU prior if it is.
    pub compute_throughput_flops_per_ns: f64,
    /// Peak sustained memory bandwidth as a Layer-1 cost prior, in bytes
    /// per nanosecond (numerically GB/s). `composite_ns` divides
    /// bytes-moved by this to estimate memory-transfer time. Historical
    /// neutral prior: 4.0 for every backend.
    pub mem_bandwidth_bytes_per_ns: f64,
}

/// Predictive fit-check result for a projected allocation. Consumed
/// by `BackendRuntime::would_fit` (in `fuel-backend-contract`) and read
/// by Picker 2's pressure-aware selectors.
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
            compute_throughput_flops_per_ns: 1.0,
            mem_bandwidth_bytes_per_ns: 4.0,
        };
        assert_eq!(caps.backend_id, BackendId::Cpu);
        assert_eq!(caps.required_alignment, 64);
        assert_eq!(caps.access_granularity_bits, 8);
        assert!(caps.op_dtype_support.contains(&(OpKind::MatMul, DType::F32)));
        assert!(!caps.op_dtype_support.contains(&(OpKind::MatMul, DType::BF16)));
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

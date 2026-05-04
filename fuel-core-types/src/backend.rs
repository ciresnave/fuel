//! Backend storage capability traits.
//!
//! After step 8 of the backend-agnostic refactor (2026-04-30), the static
//! `BackendStorage` and `BackendDevice` traits were deleted: every backend
//! now implements [`DynBackendStorage`](crate::dyn_backend::DynBackendStorage)
//! and [`DynBackendDevice`](crate::dyn_backend::DynBackendDevice) directly.
//!
//! What remains here is [`HostStorage`], the orthogonal capability trait
//! marking storage types whose data lives in host-addressable RAM.
use crate::{HostBuffer, HostBufferRef, Result};

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

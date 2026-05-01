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

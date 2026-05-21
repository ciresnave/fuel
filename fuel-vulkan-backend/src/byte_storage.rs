//! Byte-shaped Vulkan storage — Phase 7.5 storage-unification target.
//!
//! `VulkanStorageBytes` is the new Vulkan storage type that replaces
//! the legacy [`crate::VulkanStorage`] (typed `elem_count + dtype`
//! shape) once kernel migration completes. Both types coexist:
//!
//! - **Legacy `VulkanStorage`**: holds `StorageBacking` + `elem_count`
//!   + `dtype` + `tier`. Used by every existing Vulkan op kernel.
//!   Eviction/fault-back machinery (per Phase 7.5 P5) operates on
//!   this type.
//! - **`VulkanStorageBytes`** (this module): holds the same
//!   `StorageBacking` + `tier` plus a single `len_bytes` field.
//!   Dtype lives on the [`fuel_storage::Storage`] wrapper, not
//!   here. Implements [`fuel_core_types::backend::BackendStorage`].
//!
//! Per-op kernels migrate during Phase B/C. The eviction / fault-
//! back paths in [`crate::residency`] continue to operate on the
//! legacy `VulkanStorage` until Phase D, when the migrated kernels
//! force the residency machinery onto the new type.

use std::sync::Arc;

use fuel_core_types::backend::BackendStorage;
use vulkane::safe::Buffer;

use crate::{StorageBacking, Tier, VulkanBuffer};

/// Byte-shaped Vulkan storage. Backing matches the legacy type
/// (Device-resident or host-evicted) but the size field is bytes,
/// not elements, and there's no `dtype` field — dtype lives on the
/// `Storage` wrapper.
pub struct VulkanStorageBytes {
    /// VRAM or host-evicted backing. Same shape as legacy
    /// [`crate::VulkanStorage`]; preserves the residency machinery.
    pub backing: StorageBacking,
    /// Total byte count addressable through `backing`. Independent
    /// of dtype.
    pub len_bytes: usize,
    /// Current residency tier.
    pub tier: Tier,
    /// Handle back to the [`VulkanBackend`] that allocated this
    /// storage. `Some(_)` when constructed via the handle-aware
    /// path (`VulkanBackend::alloc_bytes_handle`,
    /// `upload_bytes_handle`); `None` for legacy GraphBackend-trait
    /// callers that constructed via `alloc_bytes` / `upload_bytes`
    /// without an `Arc<VulkanBackend>` wrapper.
    ///
    /// Threaded through so the pipelined-executor binding-table
    /// dispatch can reach the backend from any input's storage —
    /// mirroring CUDA's `CudaStorageBytes::device()` pattern. The
    /// CUDA equivalent is `Arc<CudaDevice>` (the device handle);
    /// for Vulkan the whole backend goes through since dispatch
    /// needs pipelines + recorder + allocator together.
    pub backend: Option<Arc<crate::VulkanBackend>>,
}

// Manual Debug impl: VulkanBuffer and StorageBacking don't derive
// Debug (the underlying vulkane Buffer + Allocation types don't),
// so we summarize the relevant fields by hand.
impl std::fmt::Debug for VulkanStorageBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let backing_tag = match &self.backing {
            StorageBacking::Device(_) => "Device",
            StorageBacking::Host { .. } => "Host",
        };
        f.debug_struct("VulkanStorageBytes")
            .field("backing", &backing_tag)
            .field("len_bytes", &self.len_bytes)
            .field("tier", &self.tier)
            .finish()
    }
}

impl VulkanStorageBytes {
    /// Build a `VulkanStorageBytes` from a device-resident buffer +
    /// byte count. Tier defaults to `OnDevice`. Legacy constructor
    /// — `backend` is `None`. Use [`Self::from_device_with_backend`]
    /// when the storage needs to be reachable from the pipelined-
    /// executor binding-table dispatch path.
    pub fn from_device(buffer: Arc<VulkanBuffer>, len_bytes: usize) -> Self {
        Self {
            backing: StorageBacking::Device(buffer),
            len_bytes,
            tier: Tier::OnDevice,
            backend: None,
        }
    }

    /// Build a `VulkanStorageBytes` from a device-resident buffer +
    /// byte count AND attach a back-reference to the
    /// [`VulkanBackend`] Arc. The pipelined-executor binding-table
    /// dispatch wrappers reach the backend through this field to
    /// allocate outputs + dispatch kernels.
    pub fn from_device_with_backend(
        buffer: Arc<VulkanBuffer>,
        len_bytes: usize,
        backend: Arc<crate::VulkanBackend>,
    ) -> Self {
        Self {
            backing: StorageBacking::Device(buffer),
            len_bytes,
            tier: Tier::OnDevice,
            backend: Some(backend),
        }
    }

    /// Borrow the `VulkanBackend` handle that allocated this storage,
    /// if one was attached. Returns `None` for storages constructed
    /// via the legacy `from_device` / `alloc_bytes` path — those
    /// can't drive the pipelined-executor binding-table dispatch
    /// (which needs to reach the backend from a `&Storage`).
    pub fn backend(&self) -> Option<&Arc<crate::VulkanBackend>> {
        self.backend.as_ref()
    }

    /// Borrow the underlying device buffer. Returns `None` if the
    /// storage has been evicted to host — callers that handle both
    /// tiers should use [`Self::buffer_opt`].
    pub fn buffer_opt(&self) -> Option<&Buffer> {
        match &self.backing {
            StorageBacking::Device(b) => Some(b.buffer()),
            StorageBacking::Host { .. } => None,
        }
    }

    /// Borrow the device buffer Arc, cloning the refcount. Returns
    /// `None` if the storage has been evicted to host.
    pub fn device_buffer_arc(&self) -> Option<Arc<VulkanBuffer>> {
        match &self.backing {
            StorageBacking::Device(b) => Some(Arc::clone(b)),
            StorageBacking::Host { .. } => None,
        }
    }

    /// Total byte count.
    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    /// Current residency tier (OnDevice or OnHost).
    pub fn tier(&self) -> Tier {
        self.tier
    }
}

impl BackendStorage for VulkanStorageBytes {
    fn len_bytes(&self) -> usize {
        self.len_bytes
    }
}

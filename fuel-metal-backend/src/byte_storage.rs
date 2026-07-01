//! Byte-shaped Metal storage — Phase 7.5 storage-unification target.
//!
//! `MetalStorageBytes` is the new Metal storage type that replaces
//! the legacy [`crate::MetalStorage`] (typed storage with `count` and
//! `dtype` fields). Both types coexist during the migration:
//!
//! - **Legacy `MetalStorage`** (`storage::MetalStorage`): owns a
//!   `Buffer` plus `count: usize` (element count) and `dtype: DType`.
//!   Used by every existing Metal op kernel.
//! - **`MetalStorageBytes`** (this module): wraps a `Buffer` plus
//!   `len_bytes: usize`. Dtype lives on the `Storage` wrapper
//!   (fuel-storage), not here. Implements
//!   [`fuel_backend_contract::backend::BackendStorage`].
//!
//! Per-op kernels migrate one family at a time during Phase B/C.
//! When the last kernel migrates, the legacy `MetalStorage` retires
//! and `MetalStorageBytes` can be renamed to `MetalStorage`.
//!
//! Module gated to macOS/iOS like the rest of fuel-metal-backend.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use std::sync::Arc;

use fuel_backend_contract::backend::BackendStorage;
use fuel_metal_kernels::metal::Buffer;

use crate::device::MetalDevice;

/// Byte-shaped Metal storage. Holds an opaque `Buffer` (Metal's
/// native byte container), the owning device, and a byte count.
/// The Metal API treats buffers as bytes regardless of element
/// type; dtype lives on the [`fuel_memory::Storage`] wrapper.
#[derive(Debug, Clone)]
pub struct MetalStorageBytes {
    /// Underlying Metal buffer. Cheap to clone (Arc-shared).
    buffer: Arc<Buffer>,
    /// Owning device — Metal buffers must be released on the device
    /// that allocated them.
    device: MetalDevice,
    /// Byte count addressable through `buffer`. Independent of
    /// dtype; dtype is on the Storage wrapper.
    len_bytes: usize,
}

impl MetalStorageBytes {
    /// Build a `MetalStorageBytes` from an already-allocated Metal
    /// buffer plus the device that owns it. Caller is responsible
    /// for the buffer being correctly sized (`len_bytes` matching
    /// the buffer's actual byte capacity).
    pub fn from_parts(buffer: Arc<Buffer>, device: MetalDevice, len_bytes: usize) -> Self {
        Self { buffer, device, len_bytes }
    }

    /// Borrow the underlying Metal buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Borrow the owning device.
    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    /// Total byte count.
    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }
}

impl BackendStorage for MetalStorageBytes {
    fn len_bytes(&self) -> usize {
        self.len_bytes
    }
}

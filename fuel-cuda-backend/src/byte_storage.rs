//! Byte-shaped CUDA storage — Phase 7.5 storage-unification target.
//!
//! `CudaStorageBytes` is the new CUDA storage type that replaces the
//! legacy [`crate::CudaStorage`] (typed `CudaStorageSlice` enum with
//! 14 dtype variants). Both types coexist during migration:
//!
//! - **Legacy `CudaStorage`** (`storage::CudaStorage`): wraps
//!   `CudaStorageSlice` (an enum holding `CudaSlice<T>` per dtype)
//!   plus `CudaDevice`. Used by every existing op kernel via
//!   match-on-variant. The `CudaDType` trait provides typed
//!   slice extraction.
//! - **`CudaStorageBytes`** (this module): wraps a single
//!   `DeviceBuffer<u8>` (raw bytes on device) plus `CudaDevice`
//!   plus `len_bytes`. Dtype lives on the [`fuel_storage::Storage`]
//!   wrapper, not here. Implements
//!   [`fuel_core_types::backend::BackendStorage`].
//!
//! Per-op kernels migrate one family at a time during Phase B/C.
//! When the last kernel migrates, the legacy `CudaStorage` retires
//! and `CudaStorageBytes` can be renamed to `CudaStorage`.

use std::sync::Arc;

use baracuda_driver::DeviceBuffer;
use fuel_core_types::backend::BackendStorage;

use crate::CudaDevice;

/// Byte-shaped CUDA storage. Holds a raw `DeviceBuffer<u8>` (CUDA-
/// allocated byte buffer), the owning device, and a byte count.
/// CUDA itself is dtype-erased at the buffer level
/// (`cudaMalloc` returns `void*`); the typed `CudaSlice<T>` views
/// happen at kernel boundaries via byte-pointer reinterpretation.
#[derive(Debug)]
pub struct CudaStorageBytes {
    /// CUDA-allocated bytes. Cheap to clone (`Arc`-shared).
    buffer: Arc<DeviceBuffer<u8>>,
    /// Owning device — buffers must be freed on the device that
    /// allocated them.
    device: CudaDevice,
    /// Byte count addressable through `buffer`. Independent of
    /// dtype; dtype is on the Storage wrapper.
    len_bytes: usize,
}

impl CudaStorageBytes {
    /// Build a `CudaStorageBytes` from an already-allocated CUDA
    /// byte buffer plus the device that owns it. Caller is
    /// responsible for `len_bytes` matching the buffer's actual byte
    /// capacity.
    pub fn from_parts(
        buffer: Arc<DeviceBuffer<u8>>,
        device: CudaDevice,
        len_bytes: usize,
    ) -> Self {
        Self { buffer, device, len_bytes }
    }

    /// Borrow the underlying byte buffer.
    pub fn buffer(&self) -> &DeviceBuffer<u8> {
        &self.buffer
    }

    /// Borrow the owning device.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Total byte count.
    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }
}

impl Clone for CudaStorageBytes {
    fn clone(&self) -> Self {
        // Cheap: bumps Arc refcount on the device buffer.
        Self {
            buffer: Arc::clone(&self.buffer),
            device: self.device.clone(),
            len_bytes: self.len_bytes,
        }
    }
}

impl BackendStorage for CudaStorageBytes {
    fn len_bytes(&self) -> usize {
        self.len_bytes
    }
}

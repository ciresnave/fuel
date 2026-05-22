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

use baracuda_driver::{DeviceBuffer, DeviceSlice};
use baracuda_types::DeviceRepr;
use fuel_core_types::backend::BackendStorage;
use fuel_core_types::Result;

use crate::error::{CudaError, WrapErr};
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

    /// Reinterpret the byte buffer as a typed `DeviceSlice<T>` view.
    /// The seam used by CUTLASS-backed kernels (and any future safe-API
    /// CUDA library that takes typed slices) to consume the
    /// dtype-erased byte substrate. Returns
    /// [`CudaError::InvalidDtypeBoundary`] when the byte length is not
    /// an integer multiple of `size_of::<T>()`. The returned slice
    /// borrows from `self`; `cuMemAlloc`'s 256-byte alignment satisfies
    /// any `DeviceRepr` we ship.
    pub fn view_as<T: DeviceRepr>(&self) -> Result<DeviceSlice<'_, T>> {
        let elem = std::mem::size_of::<T>();
        if elem != 0 && self.len_bytes % elem != 0 {
            return Err(CudaError::InvalidDtypeBoundary {
                byte_len: self.len_bytes,
                dtype_size: elem,
                dtype_name: std::any::type_name::<T>(),
            }
            .into());
        }
        Ok(self.buffer.view_as::<T>())
    }

    /// Phase 7.5 A4 substrate alloc. Allocates `byte_count` zero-
    /// initialized bytes on `device` via `device.alloc_zeros::<u8>`.
    pub fn alloc(device: &CudaDevice, byte_count: usize) -> Result<Self> {
        let buffer = device.alloc_zeros::<u8>(byte_count)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            device: device.clone(),
            len_bytes: byte_count,
        })
    }

    /// Phase 7.5 A4 substrate H2D. Allocates a fresh device buffer
    /// on `device` of size `src.len()` bytes, then copies the host
    /// slice into it. Used by `Op::Copy` / `Op::Move` from a CPU
    /// source and by graph-`Op::Const` upload paths post-migration.
    pub fn from_cpu_bytes(device: &CudaDevice, src: &[u8]) -> Result<Self> {
        let storage = Self::alloc(device, src.len())?;
        if !src.is_empty() {
            storage.buffer.copy_from_host(src).w()?;
            // copy_from_host on a DeviceBuffer is async on the
            // default stream; sync so the result is observable
            // before we hand the storage off. Async fence handles
            // are a Phase A5 follow-on.
            device.synchronize()?;
        }
        Ok(storage)
    }

    /// Phase 7.5 A4 substrate D2H. Reads the device buffer's bytes
    /// back to host as a fresh `Vec<u8>`. Called by the
    /// `(OpKind::Copy, [dt, dt], Cuda)` binding-table wrapper in
    /// `fuel-storage::dispatch::copy_to_cpu_cuda_wrapper`
    /// (bridge-retirement Phase 2, post-9c).
    pub fn to_cpu_bytes(&self) -> Result<Vec<u8>> {
        let mut out = vec![0_u8; self.len_bytes];
        if self.len_bytes > 0 {
            self.buffer.copy_to_host(&mut out).w()?;
            // copy_to_host is sync in baracuda's surface (the
            // legacy clone_dtoh path uses cuMemcpyDtoH, also sync),
            // so no extra synchronize call is needed here. We
            // still sync defensively in case async-paths are wired
            // upstream of us in the future.
            self.device.synchronize()?;
        }
        Ok(out)
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

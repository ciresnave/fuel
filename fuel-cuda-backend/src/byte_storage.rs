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

    /// Bridge-retirement Phase 3a follow-up: uninit alloc on `device`.
    /// Wraps the raw `CudaDevice::alloc::<u8>` (uninit `cuMemAlloc`).
    /// Callers must zero or write the bytes before reading — typically
    /// by following with an `Op::ZeroFill` graph node, whose CUDA
    /// kernel calls [`Self::zero_async`].
    ///
    /// The `unsafe` on `CudaDevice::alloc` is wrapped here; the safety
    /// contract is "bytes are uninitialized; reads before a write are
    /// UB at the typed-slice boundary". Internal to the executor's
    /// `WorkItemKind::Alloc` → `WorkItemKind::ZeroFill` (or other
    /// init op) sequence, this contract is upheld by construction.
    pub fn alloc_uninit(device: &CudaDevice, byte_count: usize) -> Result<Self> {
        // SAFETY: returned bytes are uninit; the caller (executor's
        // WorkItemKind::Alloc arm) guarantees a subsequent Op::ZeroFill
        // or full-buffer-write op runs before any reader observes the
        // bytes. The byte-level Arc wrapper has no `as_slice<T>()` arm
        // that would dereference uninit bytes between Alloc and Fill.
        let buffer = unsafe { device.alloc::<u8>(byte_count) }?;
        Ok(Self {
            buffer: Arc::new(buffer),
            device: device.clone(),
            len_bytes: byte_count,
        })
    }

    /// Multi-output Option C: extract one bundle slot as a fresh
    /// standalone `CudaStorageBytes`. The destination has its own
    /// `Arc<DeviceBuffer<u8>>` — independent of the source bundle's
    /// Arc — so the bundle can drop once every other View of it has
    /// also dropped. Used by `Op::ViewOwned` on CUDA-resident
    /// bundled producers.
    ///
    /// `byte_offset + len_bytes` must be ≤ `self.len_bytes`.
    pub fn slot_copy_to_new(
        &self,
        byte_offset: usize,
        len_bytes:   usize,
    ) -> Result<Self> {
        let end = byte_offset.checked_add(len_bytes).ok_or_else(|| {
            fuel_core_types::Error::Msg(format!(
                "CudaStorageBytes::slot_copy_to_new: byte_offset {byte_offset} \
                 + len_bytes {len_bytes} overflows",
            ))
            .bt()
        })?;
        if end > self.len_bytes {
            return Err(fuel_core_types::Error::Msg(format!(
                "CudaStorageBytes::slot_copy_to_new: slot byte range \
                 [{byte_offset}..{end}) exceeds source byte length {}",
                self.len_bytes,
            ))
            .bt());
        }
        // Allocate a raw uninit DeviceBuffer first so we can get a
        // DeviceSliceMut for the destination — the Arc<DeviceBuffer>
        // shape on CudaStorageBytes blocks `view_as_mut` because
        // mutable view requires `&mut DeviceBuffer`. We do the D2D
        // before wrapping in the Arc.
        //
        // SAFETY: the immediately-following memcpy_dtod fully
        // initializes `len_bytes` bytes. Empty-slot fast path falls
        // through to the zero-init alloc below.
        if len_bytes == 0 {
            return Self::alloc(&self.device, 0);
        }
        let mut dst_buffer: DeviceBuffer<u8> =
            unsafe { self.device.alloc::<u8>(len_bytes) }?;
        {
            let src_view = self.buffer.view_as::<u8>();
            let src_slice = src_view.slice(byte_offset..byte_offset + len_bytes);
            let mut dst_view = dst_buffer.view_as_mut::<u8>();
            let mut dst_slice = dst_view.slice_mut(0..len_bytes);
            self.device.memcpy_dtod(&src_slice, &mut dst_slice)?;
        }
        // Sync so consumers see the bytes through subsequent
        // metadata-only Arc reads. Mirrors `write_from_host` / the
        // existing `to_cpu_bytes` synchronize semantics.
        self.device.synchronize()?;
        Ok(Self {
            buffer: Arc::new(dst_buffer),
            device: self.device.clone(),
            len_bytes,
        })
    }

    /// Extract `outer_count` strided tiles into a fresh contiguous
    /// `CudaStorageBytes`. Tile `t` lives at
    /// `t * stride_bytes + offset_in_outer .. + chunk_row_bytes` in
    /// the source. Result is `outer_count` tiles packed back-to-back
    /// in the new buffer.
    ///
    /// Used by `Op::WriteSliceRotating` to gather one half of a
    /// ring-boundary split when the rotating axis is not the leading
    /// dim — for axis 0 the byte split is a prefix/suffix and
    /// [`Self::slot_copy_to_new`] is enough. Implementation does one
    /// `memcpy_dtod` per outer tile; for typical KV-cache shapes
    /// (outer_count ≤ ~4) the launch overhead is small.
    pub fn extract_strided_to_new(
        &self,
        outer_count:     usize,
        stride_bytes:    usize,
        offset_in_outer: usize,
        chunk_row_bytes: usize,
    ) -> Result<Self> {
        let dest_total = outer_count
            .checked_mul(chunk_row_bytes)
            .ok_or_else(|| {
                fuel_core_types::Error::Msg(
                    "extract_strided_to_new: outer_count * chunk_row_bytes overflows".into(),
                ).bt()
            })?;
        if dest_total == 0 {
            return Self::alloc(&self.device, 0);
        }
        // Per-tile end must fit inside the source.
        if outer_count > 0 {
            let last_tile_end = (outer_count - 1)
                .checked_mul(stride_bytes)
                .and_then(|x| x.checked_add(offset_in_outer))
                .and_then(|x| x.checked_add(chunk_row_bytes))
                .ok_or_else(|| {
                    fuel_core_types::Error::Msg(
                        "extract_strided_to_new: tile span overflow".into(),
                    ).bt()
                })?;
            if last_tile_end > self.len_bytes {
                return Err(fuel_core_types::Error::Msg(format!(
                    "extract_strided_to_new: last tile end {last_tile_end} > src bytes {}",
                    self.len_bytes,
                )).bt());
            }
        }
        let mut dst_buffer: DeviceBuffer<u8> =
            unsafe { self.device.alloc::<u8>(dest_total) }?;
        {
            let src_view = self.buffer.view_as::<u8>();
            let mut dst_view = dst_buffer.view_as_mut::<u8>();
            for t in 0..outer_count {
                let src_off = t * stride_bytes + offset_in_outer;
                let dst_off = t * chunk_row_bytes;
                let src_slice = src_view.slice(src_off..src_off + chunk_row_bytes);
                let mut dst_slice = dst_view.slice_mut(dst_off..dst_off + chunk_row_bytes);
                self.device.memcpy_dtod(&src_slice, &mut dst_slice)?;
            }
        }
        self.device.synchronize()?;
        Ok(Self {
            buffer: Arc::new(dst_buffer),
            device: self.device.clone(),
            len_bytes: dest_total,
        })
    }

    /// Bridge-retirement Phase 3b: H2D into an already-allocated
    /// CUDA buffer. Pairs with [`Self::alloc_uninit`] for the
    /// `Op::Alloc → Op::Copy { target: Cuda }` H2D pattern — the
    /// executor allocates uninit storage, then the Copy kernel
    /// writes host bytes into it.
    ///
    /// `src.len()` must equal `self.len_bytes` — the buffer is sized
    /// by the executor to the destination's exact byte count. Empty
    /// buffers are a no-op (baracuda's copy_from_host short-circuits
    /// on zero-length transfers).
    pub fn write_from_host(&self, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        if src.len() != self.len_bytes {
            return Err(fuel_core_types::Error::Msg(format!(
                "CudaStorageBytes::write_from_host: src.len() ({}) != \
                 storage.len_bytes ({})",
                src.len(), self.len_bytes,
            )).bt());
        }
        self.buffer.copy_from_host(src).w()?;
        // copy_from_host is async on the default stream; sync so the
        // result is observable before the next op picks up the
        // storage. Mirrors `from_cpu_bytes`'s sync.
        self.device.synchronize()?;
        Ok(())
    }

    /// Bridge-retirement Phase 3a follow-up: in-place device-side
    /// zero-fill via baracuda alpha.30's `DeviceBuffer::zero_async`
    /// (`cuMemsetD8Async`). The buffer's identity (CUdeviceptr)
    /// doesn't change — `Arc` clones held elsewhere see the same
    /// post-zero bytes.
    ///
    /// Used by `fuel-storage::pipelined::WorkItemKind::ZeroFill` for
    /// `Op::ZeroFill` nodes. Pairs with [`Self::alloc_uninit`] to
    /// give the architecturally clean `Op::Alloc` (uninit) →
    /// `Op::ZeroFill` (explicit fill) pipeline.
    pub fn zero_async(&self) -> Result<()> {
        if self.len_bytes == 0 {
            return Ok(());
        }
        let stream = self.device.cuda_stream();
        self.buffer.zero_async(&stream).w()?;
        Ok(())
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

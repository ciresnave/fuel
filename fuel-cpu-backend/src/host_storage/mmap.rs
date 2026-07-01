//! Memory-mapped [`HostStorage`] — zero-copy view over a file on disk.
//!
//! Primary use case: loading large weight tensors (safetensors, GGUF) without
//! reading the whole file into RAM. An `MmappedHostStorage` owns a
//! [`memmap2::Mmap`] (or `Arc<Mmap>`) plus a dtype-tagged byte range, and
//! exposes [`HostStorage::as_host_buffer_ref`] as a borrowed typed slice
//! reinterpreted from the mapped bytes.
//!
//! # Example
//!
//! ```no_run
//! use std::fs::File;
//! use fuel_cpu_backend::host_storage::MmappedHostStorage;
//! use fuel_backend_contract::backend::HostStorage; use fuel_ir::DType;
//!
//! let file = File::open("weights.bin").unwrap();
//! // Reinterpret the entire file as a flat f32 slice.
//! let storage = unsafe {
//!     MmappedHostStorage::from_file(&file, DType::F32, 0, file.metadata().unwrap().len() as usize / 4)
//! }.unwrap();
//! let view = storage.as_host_buffer_ref().unwrap();
//! assert_eq!(view.dtype(), DType::F32);
//! ```
//!
//! # Safety invariants
//!
//! The caller is responsible for ensuring that the mapped region actually
//! contains a valid bit-pattern for the advertised dtype. Reading garbage
//! into a typed slice is not UB for numeric primitives (all bit patterns
//! are valid `f32`/`u32`/etc.), so the worst case on a corrupted file is
//! wrong numbers, not memory-unsafety. Types with validity invariants
//! (`bool`, enums, references) are not supported as dtypes here — the
//! [`DType`] catalog only contains numeric primitives, which makes the
//! reinterpretation sound.
//!
//! The mmap is held via `Arc<Mmap>` so multiple [`MmappedHostStorage`]
//! values can view different slices of the same file without copying.
//! Dropping all views releases the mapping.

use fuel_backend_contract::backend::HostStorage;
use fuel_ir::{DType, Error, HostBuffer, HostBufferRef, Result};
use memmap2::Mmap;
use std::fs::File;
use std::sync::Arc;

/// Zero-copy host storage backed by a memory-mapped file region.
///
/// Construct via [`from_file`](Self::from_file) or
/// [`from_shared_mmap`](Self::from_shared_mmap). The range `[offset,
/// offset + element_count)` (in elements, not bytes) must lie inside the
/// mapped region, and the byte offset must satisfy the dtype's alignment.
#[derive(Debug, Clone)]
pub struct MmappedHostStorage {
    mmap: Arc<Mmap>,
    dtype: DType,
    /// Starting element offset into the mmap, measured in elements of `dtype`.
    elem_offset: usize,
    /// Number of elements (of `dtype`) this view exposes.
    elem_count: usize,
}

impl MmappedHostStorage {
    /// Memory-map `file` and reinterpret the range `[byte_offset, byte_offset
    /// + elem_count * sizeof(dtype))` as a typed slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The file content at that range is a valid bit-pattern for `dtype`.
    /// - The file is not mutated by another process for the lifetime of the
    ///   returned mapping (required by `memmap2::Mmap`).
    ///
    /// `byte_offset` must be aligned to `dtype.size_in_bytes()`. Returns
    /// an error if the range is out of bounds or misaligned.
    pub unsafe fn from_file(
        file: &File,
        dtype: DType,
        byte_offset: usize,
        elem_count: usize,
    ) -> Result<Self> {
        // SAFETY: caller upheld the "file won't be mutated" precondition.
        let mmap = unsafe { Mmap::map(file) }
            .map_err(|e| Error::Msg(format!("mmap failed: {e}")).bt())?;
        // SAFETY: caller upheld the bit-pattern precondition.
        unsafe { Self::from_shared_mmap(Arc::new(mmap), dtype, byte_offset, elem_count) }
    }

    /// Construct from an already-mmapped region. Useful when many
    /// `MmappedHostStorage` values share one mapping (e.g. each tensor in a
    /// safetensors file points to a different slice of the same mmap).
    ///
    /// # Safety
    ///
    /// Same preconditions as [`from_file`](Self::from_file). Additionally,
    /// the caller must not mutate `mmap`'s backing file while any
    /// `MmappedHostStorage` over it is live.
    pub unsafe fn from_shared_mmap(
        mmap: Arc<Mmap>,
        dtype: DType,
        byte_offset: usize,
        elem_count: usize,
    ) -> Result<Self> {
        let elem_size = dtype.size_in_bytes();
        if elem_size == 0 {
            return Err(Error::Msg("dtype has zero size".into()).bt());
        }
        if byte_offset % elem_size != 0 {
            return Err(Error::Msg(format!(
                "mmap byte_offset {byte_offset} not aligned to {elem_size}-byte dtype {dtype:?}",
            ))
            .bt());
        }
        let end = byte_offset
            .checked_add(
                elem_count
                    .checked_mul(elem_size)
                    .ok_or_else(|| Error::Msg("elem_count overflow".into()).bt())?,
            )
            .ok_or_else(|| Error::Msg("byte range overflow".into()).bt())?;
        if end > mmap.len() {
            return Err(Error::Msg(format!(
                "mmap range [{byte_offset}, {end}) out of bounds (len = {})",
                mmap.len()
            ))
            .bt());
        }
        Ok(Self {
            mmap,
            dtype,
            elem_offset: byte_offset / elem_size,
            elem_count,
        })
    }

    /// Element dtype of the view.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Number of elements this view exposes.
    pub fn len(&self) -> usize {
        self.elem_count
    }

    /// `true` if the view covers zero elements.
    pub fn is_empty(&self) -> bool {
        self.elem_count == 0
    }

    /// Borrow the mmap'd bytes as a typed slice of `T`. Returns `None` if
    /// `T` doesn't match the stored dtype or alignment is insufficient.
    ///
    /// # Safety
    ///
    /// Caller must ensure `T` matches `self.dtype()` — the public typed
    /// entry point [`as_host_buffer_ref`](HostStorage::as_host_buffer_ref)
    /// dispatches correctly on dtype, so prefer that.
    unsafe fn typed_slice<T>(&self) -> &[T] {
        let byte_offset = self.elem_offset * self.dtype.size_in_bytes();
        let ptr = unsafe { self.mmap.as_ptr().add(byte_offset) } as *const T;
        // SAFETY: caller asserted T matches dtype, bounds checked at
        // construction, mmap outlives the returned slice (it's owned by
        // &self).
        unsafe { std::slice::from_raw_parts(ptr, self.elem_count) }
    }
}

impl HostStorage for MmappedHostStorage {
    fn as_host_buffer_ref(&self) -> Result<HostBufferRef<'_>> {
        // SAFETY: we own the mmap via Arc; the slice lifetime ties to &self;
        // the construction-time bounds+alignment checks guarantee the typed
        // slice stays inside the mapped region and is correctly aligned for
        // the dtype.
        let r = unsafe {
            match self.dtype {
                DType::U8 => HostBufferRef::U8(self.typed_slice::<u8>()),
                DType::I8 => HostBufferRef::I8(self.typed_slice::<i8>()),
                DType::U32 => HostBufferRef::U32(self.typed_slice::<u32>()),
                DType::I16 => HostBufferRef::I16(self.typed_slice::<i16>()),
                DType::I32 => HostBufferRef::I32(self.typed_slice::<i32>()),
                DType::I64 => HostBufferRef::I64(self.typed_slice::<i64>()),
                DType::BF16 => HostBufferRef::BF16(self.typed_slice::<half::bf16>()),
                DType::F16 => HostBufferRef::F16(self.typed_slice::<half::f16>()),
                DType::F32 => HostBufferRef::F32(self.typed_slice::<f32>()),
                DType::F64 => HostBufferRef::F64(self.typed_slice::<f64>()),
                DType::F8E4M3 => HostBufferRef::F8E4M3(self.typed_slice::<float8::F8E4M3>()),
                DType::F6E2M3 => HostBufferRef::F6E2M3(self.typed_slice::<u8>()),
                DType::F6E3M2 => HostBufferRef::F6E3M2(self.typed_slice::<u8>()),
                DType::F4 => HostBufferRef::F4(self.typed_slice::<u8>()),
                DType::F8E8M0 => HostBufferRef::F8E8M0(self.typed_slice::<u8>()),
            }
        };
        Ok(r)
    }

    // Default `into_host_buffer` materializes via `.to_owned()`, which is
    // correct here — mmap-backed storage can't hand out a `HostBuffer`
    // owning the mmap'd bytes without lying about allocation ownership, so
    // a copy at the materialization boundary is required. Leave the
    // default in place.
    fn into_host_buffer(self) -> Result<HostBuffer> {
        Ok(self.as_host_buffer_ref()?.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write `bytes` to a tempfile and mmap it. Returns (file, storage).
    /// Kept inline so the test scope contains the `tempfile::NamedTempFile`
    /// and the mmap consistently.
    fn mmap_of(
        bytes: &[u8],
        dtype: DType,
        elem_count: usize,
    ) -> (tempfile::NamedTempFile, MmappedHostStorage) {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        let storage = unsafe {
            MmappedHostStorage::from_file(f.as_file(), dtype, 0, elem_count)
        }
        .unwrap();
        (f, storage)
    }

    #[test]
    fn mmap_f32_round_trip() {
        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&values);
        let (_keep_file_alive, storage) = mmap_of(bytes, DType::F32, values.len());
        let view = storage.as_host_buffer_ref().unwrap();
        assert_eq!(view.dtype(), DType::F32);
        assert_eq!(view.len(), 4);
        match view {
            HostBufferRef::F32(s) => assert_eq!(s, &values),
            _ => panic!("unexpected dtype"),
        }
    }

    #[test]
    fn mmap_rejects_misaligned_offset() {
        let bytes = vec![0u8; 16];
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&bytes).unwrap();
        f.flush().unwrap();
        let err = unsafe {
            MmappedHostStorage::from_file(f.as_file(), DType::F32, 1, 1)
        }
        .unwrap_err();
        assert!(
            format!("{err}").contains("not aligned"),
            "expected alignment error, got: {err}"
        );
    }

    #[test]
    fn mmap_rejects_out_of_bounds() {
        let bytes = vec![0u8; 8];
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&bytes).unwrap();
        f.flush().unwrap();
        // 3 f32 = 12 bytes > 8
        let err = unsafe {
            MmappedHostStorage::from_file(f.as_file(), DType::F32, 0, 3)
        }
        .unwrap_err();
        assert!(
            format!("{err}").contains("out of bounds"),
            "expected OOB error, got: {err}"
        );
    }

    #[test]
    fn mmap_zero_length_is_ok() {
        let bytes = vec![0u8; 0];
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&bytes).unwrap();
        f.flush().unwrap();
        // macOS/Windows mmap of zero-length files can fail; we test
        // explicitly for f32 with elem_count=0 over a zero-length file.
        // If the mmap call itself fails, skip the rest.
        let storage = match unsafe {
            MmappedHostStorage::from_file(f.as_file(), DType::F32, 0, 0)
        } {
            Ok(s) => s,
            Err(e) if format!("{e}").contains("mmap failed") => return,
            Err(e) => panic!("unexpected error: {e}"),
        };
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
    }

    #[test]
    fn mmap_into_host_buffer_copies() {
        let values = [10.0_f32, 20.0, 30.0];
        let bytes: &[u8] = bytemuck::cast_slice(&values);
        let (_keep, storage) = mmap_of(bytes, DType::F32, values.len());
        let owned = storage.into_host_buffer().unwrap();
        match owned {
            HostBuffer::F32(v) => assert_eq!(v, values),
            _ => panic!("unexpected dtype"),
        }
    }

    #[test]
    fn mmap_shared_mmap_multi_view() {
        // Four f32s: two views, first 2 and last 2.
        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&values);
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        let mmap = Arc::new(unsafe { Mmap::map(f.as_file()) }.unwrap());
        let head = unsafe {
            MmappedHostStorage::from_shared_mmap(mmap.clone(), DType::F32, 0, 2)
        }
        .unwrap();
        let tail = unsafe {
            MmappedHostStorage::from_shared_mmap(mmap, DType::F32, 8, 2)
        }
        .unwrap();
        let (h, t) = (
            head.as_host_buffer_ref().unwrap(),
            tail.as_host_buffer_ref().unwrap(),
        );
        match (h, t) {
            (HostBufferRef::F32(h), HostBufferRef::F32(t)) => {
                assert_eq!(h, &[1.0, 2.0]);
                assert_eq!(t, &[3.0, 4.0]);
            }
            _ => panic!("unexpected dtype"),
        }
    }
}

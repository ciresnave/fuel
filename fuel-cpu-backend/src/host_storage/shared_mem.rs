//! Inter-process shared [`HostStorage`] — a file-backed, shared
//! memory-mapped tensor view.
//!
//! Use case: two or more processes coordinating on the same tensor
//! without IPC round-trips — e.g. a producer writing KV-cache updates
//! that a sibling inference worker consumes, or a weight-preloader that
//! populates a large allocation once and hands it off to per-request
//! workers.
//!
//! Both `create` (producer) and `open` (consumer) take the same filesystem
//! path and the same dtype + element count. The typed slice the two ends
//! see refers to the same physical pages — a write through the producer's
//! slice is immediately visible to the consumer.
//!
//! No synchronization primitives are provided here. Consumers are
//! responsible for their own locking (atomics for fine-grained mutations,
//! file locks for coarse-grained handoff, etc.).
//!
//! # Safety
//!
//! The caller must ensure the file is not truncated, unlinked, or mutated
//! in an incompatible way for the lifetime of any `SharedMemHostStorage`
//! over it. On Windows the file backing is locked against truncation
//! automatically by the mapping; on Linux the backing file can be
//! `unlink`'d after mapping without invalidating the mapping, but
//! truncating it to a smaller size is UB.
//!
//! # Example
//!
//! ```no_run
//! # use fuel_cpu_backend::host_storage::shared_mem::SharedMemHostStorage;
//! # use fuel_ir::{backend::HostStorage, DType};
//! // Process A: create + fill
//! let mut prod = SharedMemHostStorage::create("/tmp/fuel_shm_kv", DType::F32, 1024).unwrap();
//! if let Some(s) = prod.as_mut_slice_f32() { s[0] = 1.5; }
//!
//! // Process B: open + read (same path + dtype + len)
//! let cons = SharedMemHostStorage::open("/tmp/fuel_shm_kv", DType::F32, 1024).unwrap();
//! let view = cons.as_host_buffer_ref().unwrap();
//! ```

use fuel_ir::backend::HostStorage;
use fuel_ir::{DType, Error, HostBuffer, HostBufferRef, Result};
use half::{bf16, f16};
use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::Path;

/// Shared, file-backed, memory-mapped host storage.
///
/// Constructed via [`create`](Self::create) (producer side — creates and
/// sizes the backing file) or [`open`](Self::open) (consumer side —
/// attaches to an existing file).
pub struct SharedMemHostStorage {
    // Order matters for Drop: mmap must be released before file on Windows.
    mmap: MmapMut,
    #[allow(dead_code)]
    file: File,
    dtype: DType,
    elem_count: usize,
}

impl std::fmt::Debug for SharedMemHostStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedMemHostStorage")
            .field("dtype", &self.dtype)
            .field("elem_count", &self.elem_count)
            .field("bytes", &(self.elem_count * self.dtype.size_in_bytes()))
            .finish()
    }
}

impl SharedMemHostStorage {
    /// Create a new shared memory region at `path`, sized for `elem_count`
    /// elements of `dtype`. If the file exists, it's truncated to the new
    /// size.
    pub fn create<P: AsRef<Path>>(path: P, dtype: DType, elem_count: usize) -> Result<Self> {
        let elem_size = dtype.size_in_bytes();
        if elem_size == 0 {
            return Err(Error::Msg("dtype has zero size".into()).bt());
        }
        let bytes = elem_count
            .checked_mul(elem_size)
            .ok_or_else(|| Error::Msg("elem_count overflow".into()).bt())?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())
            .map_err(|e| Error::Msg(format!("shared_mem create: {e}")).bt())?;
        // Size the file. mmap of a zero-length file is platform-quirky;
        // we forbid it here for consistency with PinnedHostStorage.
        if bytes == 0 {
            return Err(Error::Msg(
                "SharedMemHostStorage: zero-length regions are not supported".into(),
            )
            .bt());
        }
        file.set_len(bytes as u64)
            .map_err(|e| Error::Msg(format!("shared_mem set_len: {e}")).bt())?;
        // SAFETY: file is sized to `bytes`, alignment is handled by the OS.
        let mmap = unsafe { MmapOptions::new().len(bytes).map_mut(&file) }
            .map_err(|e| Error::Msg(format!("shared_mem map_mut: {e}")).bt())?;
        Ok(Self {
            mmap,
            file,
            dtype,
            elem_count,
        })
    }

    /// Attach to an existing shared memory region at `path`. The file must
    /// exist and be at least `elem_count * dtype.size_in_bytes()` bytes.
    pub fn open<P: AsRef<Path>>(path: P, dtype: DType, elem_count: usize) -> Result<Self> {
        let elem_size = dtype.size_in_bytes();
        if elem_size == 0 {
            return Err(Error::Msg("dtype has zero size".into()).bt());
        }
        let bytes = elem_count
            .checked_mul(elem_size)
            .ok_or_else(|| Error::Msg("elem_count overflow".into()).bt())?;
        if bytes == 0 {
            return Err(Error::Msg(
                "SharedMemHostStorage: zero-length regions are not supported".into(),
            )
            .bt());
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.as_ref())
            .map_err(|e| Error::Msg(format!("shared_mem open: {e}")).bt())?;
        let file_len = file
            .metadata()
            .map(|m| m.len())
            .map_err(|e| Error::Msg(format!("shared_mem metadata: {e}")).bt())?;
        if (file_len as usize) < bytes {
            return Err(Error::Msg(format!(
                "shared_mem file too small: {file_len} < {bytes}"
            ))
            .bt());
        }
        // SAFETY: file is large enough for the requested range.
        let mmap = unsafe { MmapOptions::new().len(bytes).map_mut(&file) }
            .map_err(|e| Error::Msg(format!("shared_mem map_mut: {e}")).bt())?;
        Ok(Self {
            mmap,
            file,
            dtype,
            elem_count,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.elem_count
    }

    pub fn is_empty(&self) -> bool {
        self.elem_count == 0
    }

    /// Mutable typed-slice accessors for producer-side writes. Return
    /// `None` if the stored dtype doesn't match.
    pub fn as_mut_slice_u8(&mut self) -> Option<&mut [u8]> {
        self.typed_slice_mut::<u8>(DType::U8)
    }
    pub fn as_mut_slice_f32(&mut self) -> Option<&mut [f32]> {
        self.typed_slice_mut::<f32>(DType::F32)
    }
    pub fn as_mut_slice_f64(&mut self) -> Option<&mut [f64]> {
        self.typed_slice_mut::<f64>(DType::F64)
    }

    fn typed_slice_mut<T>(&mut self, expected: DType) -> Option<&mut [T]> {
        if self.dtype != expected {
            return None;
        }
        let ptr = self.mmap.as_mut_ptr() as *mut T;
        // SAFETY: construction checks ensure `elem_count * size_of::<T>()`
        // fits within the mapped region; ptr is never null (bytes > 0
        // validated in ctor).
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.elem_count) })
    }

    /// Reinterpret the mmap'd bytes as a typed slice of `T`. Caller must
    /// match `T` to the stored dtype — the public entry point
    /// [`HostStorage::as_host_buffer_ref`] dispatches correctly.
    unsafe fn typed_slice<T>(&self) -> &[T] {
        let ptr = self.mmap.as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, self.elem_count) }
    }
}

impl HostStorage for SharedMemHostStorage {
    fn as_host_buffer_ref(&self) -> Result<HostBufferRef<'_>> {
        let r = unsafe {
            match self.dtype {
                DType::U8 => HostBufferRef::U8(self.typed_slice::<u8>()),
                DType::I8 => HostBufferRef::I8(self.typed_slice::<i8>()),
                DType::U32 => HostBufferRef::U32(self.typed_slice::<u32>()),
                DType::I16 => HostBufferRef::I16(self.typed_slice::<i16>()),
                DType::I32 => HostBufferRef::I32(self.typed_slice::<i32>()),
                DType::I64 => HostBufferRef::I64(self.typed_slice::<i64>()),
                DType::BF16 => HostBufferRef::BF16(self.typed_slice::<bf16>()),
                DType::F16 => HostBufferRef::F16(self.typed_slice::<f16>()),
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

    fn into_host_buffer(self) -> Result<HostBuffer> {
        Ok(self.as_host_buffer_ref()?.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(suffix: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fuel_shared_mem_test_{}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            suffix
        ));
        p
    }

    #[test]
    fn create_then_open_sees_same_bytes() {
        let path = tmp_path("cross_handle");
        let values = [1.0_f32, 2.0, 3.0, 4.0];
        {
            let mut prod =
                SharedMemHostStorage::create(&path, DType::F32, values.len()).unwrap();
            let slice = prod.as_mut_slice_f32().unwrap();
            slice.copy_from_slice(&values);
        }
        let cons = SharedMemHostStorage::open(&path, DType::F32, values.len()).unwrap();
        match cons.as_host_buffer_ref().unwrap() {
            HostBufferRef::F32(s) => assert_eq!(s, &values),
            _ => panic!("unexpected dtype"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_is_visible_through_sibling_handle() {
        // Same process, two handles — the canonical pattern for thread-
        // level sharing; IPC sharing uses the same API across processes.
        let path = tmp_path("sibling_write");
        let mut a = SharedMemHostStorage::create(&path, DType::F64, 2).unwrap();
        let b = SharedMemHostStorage::open(&path, DType::F64, 2).unwrap();
        a.as_mut_slice_f64().unwrap().copy_from_slice(&[1.5, 2.5]);
        match b.as_host_buffer_ref().unwrap() {
            HostBufferRef::F64(s) => assert_eq!(s, &[1.5, 2.5]),
            _ => panic!("unexpected dtype"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_too_small_file() {
        let path = tmp_path("too_small");
        // Create with 2 f32 (8 bytes)
        let _prod = SharedMemHostStorage::create(&path, DType::F32, 2).unwrap();
        // Ask for 100 f32 (400 bytes) via open — should fail.
        let err =
            SharedMemHostStorage::open(&path, DType::F32, 100).unwrap_err();
        assert!(
            format!("{err}").contains("too small"),
            "expected too-small error, got: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn zero_length_is_rejected() {
        let path = tmp_path("zero_length");
        let err =
            SharedMemHostStorage::create(&path, DType::F32, 0).unwrap_err();
        assert!(
            format!("{err}").contains("zero-length"),
            "expected zero-length error, got: {err}"
        );
    }

    #[test]
    fn into_host_buffer_materializes_copy() {
        let path = tmp_path("materialize");
        let mut prod = SharedMemHostStorage::create(&path, DType::I32, 3).unwrap();
        prod.typed_slice_mut::<i32>(DType::I32)
            .unwrap()
            .copy_from_slice(&[-1, 0, 1]);
        let owned = prod.into_host_buffer().unwrap();
        match owned {
            HostBuffer::I32(v) => assert_eq!(v, [-1, 0, 1]),
            _ => panic!("unexpected dtype"),
        }
        let _ = std::fs::remove_file(&path);
    }
}

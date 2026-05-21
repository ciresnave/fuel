//! Mmap-backed host storage for evicted GPU tensors.
//!
//! When VRAM is tight, cold tensors get spilled out of VRAM and land
//! here. The file is mmap'd so the OS manages RAM-vs-disk paging
//! transparently, and evicted tensors stay warm enough for fault-back
//! when compute reaches them again.
//!
//! ## Scope of this module (P5 step 2a)
//!
//! Just the primitive: a fixed-size mmap file plus a simple slot
//! allocator (first-fit free list). The VulkanBackend integration
//! (download to slot on eviction, read on fault-back, OOM trigger)
//! lives in a follow-up — that change touches VulkanStorage
//! ownership and the async command-buffer chain, so it needs its
//! own focused session.
//!
//! ## Future
//!
//! - Growth: today the file is fixed-size; growing requires
//!   remapping. memmap2 doesn't support in-place growth, so we'd
//!   unmap + resize + remap with care around existing Slot offsets.
//! - Slab allocator: first-fit on a linked-list freelist is fine for
//!   hundreds of allocations but not thousands. If eviction churns
//!   frequently we'd want size-class buckets or a buddy allocator.
//! - DirectStorage: Windows (RTX IO) / Linux (GPUDirect Storage)
//!   allow NVMe → GPU direct DMA. When that lands, this file stays
//!   as the host-side landing but compute paths bypass the RAM hop.

use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Mutex;

use memmap2::{MmapMut, MmapOptions};

/// A byte-range slot inside a [`ResidencyFile`].
///
/// Opaque offsets into the file. Handing one to a
/// `ResidencyFile::read`/`write` reads/writes that byte range;
/// handing one to `free` returns it to the freelist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    pub offset: u64,
    pub len: u64,
}

/// Internal free-list entry. Freelist is kept sorted by offset so
/// `free` can coalesce adjacent regions cheaply.
#[derive(Debug, Clone, Copy)]
struct FreeRegion {
    offset: u64,
    len: u64,
}

/// Mmap-backed host storage for evicted device tensors. Fixed-size
/// file; slot allocator with coalescing first-fit.
pub struct ResidencyFile {
    _file: File,          // kept alive so the mmap stays valid
    mmap: Mutex<MmapMut>, // writes need exclusive access
    capacity: u64,
    // Free regions sorted by offset. `free()` coalesces touching
    // neighbors to keep fragmentation bounded.
    free_regions: Mutex<Vec<FreeRegion>>,
}

impl ResidencyFile {
    /// Create a new residency file of the given capacity (bytes).
    /// Truncates the file to `capacity` and mmaps the whole thing.
    /// The file is exclusively owned by this ResidencyFile; callers
    /// should pick a unique path (e.g., include a PID / session id).
    pub fn create<P: AsRef<Path>>(path: P, capacity: u64) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true).write(true).create(true).truncate(true)
            .open(path)?;
        file.seek(SeekFrom::Start(capacity.saturating_sub(1)))?;
        file.write_all(&[0])?;
        file.seek(SeekFrom::Start(0))?;
        file.sync_all()?;

        let mmap = unsafe { MmapOptions::new().len(capacity as usize).map_mut(&file)? };
        Ok(Self {
            _file: file,
            mmap: Mutex::new(mmap),
            capacity,
            free_regions: Mutex::new(vec![FreeRegion { offset: 0, len: capacity }]),
        })
    }

    /// Total file size in bytes.
    pub fn capacity(&self) -> u64 { self.capacity }

    /// Sum of free-region bytes. Not the largest contiguous free run
    /// — fragmentation means `bytes_free()` ≥ largest allocatable.
    pub fn bytes_free(&self) -> u64 {
        self.free_regions.lock().unwrap().iter().map(|r| r.len).sum()
    }

    /// Allocate a slot of `bytes` bytes. Returns `None` if no
    /// contiguous free region is large enough (i.e., fragmentation
    /// blocks the request even if total free bytes suffice).
    pub fn alloc(&self, bytes: u64) -> Option<Slot> {
        if bytes == 0 { return Some(Slot { offset: 0, len: 0 }); }
        let mut regions = self.free_regions.lock().unwrap();
        // First-fit. For our workload (few-thousand allocations over
        // a session) this is adequate; worst-case O(n) per alloc.
        for i in 0..regions.len() {
            if regions[i].len >= bytes {
                let slot = Slot { offset: regions[i].offset, len: bytes };
                if regions[i].len == bytes {
                    regions.remove(i);
                } else {
                    regions[i].offset += bytes;
                    regions[i].len -= bytes;
                }
                return Some(slot);
            }
        }
        None
    }

    /// Return a slot to the freelist. Coalesces with adjacent free
    /// regions so fragmentation doesn't accumulate after a
    /// recycle-heavy workload.
    pub fn free(&self, slot: Slot) {
        if slot.len == 0 { return; }
        let mut regions = self.free_regions.lock().unwrap();
        // Find insertion point (sorted by offset).
        let pos = regions.partition_point(|r| r.offset < slot.offset);
        regions.insert(pos, FreeRegion { offset: slot.offset, len: slot.len });
        // Coalesce with right neighbor, then with left.
        if pos + 1 < regions.len()
            && regions[pos].offset + regions[pos].len == regions[pos + 1].offset
        {
            regions[pos].len += regions[pos + 1].len;
            regions.remove(pos + 1);
        }
        if pos > 0
            && regions[pos - 1].offset + regions[pos - 1].len == regions[pos].offset
        {
            regions[pos - 1].len += regions[pos].len;
            regions.remove(pos);
        }
    }

    /// Write `data` to the slot. Panics if `data.len() != slot.len`.
    pub fn write(&self, slot: Slot, data: &[u8]) {
        assert_eq!(data.len() as u64, slot.len,
            "ResidencyFile::write: data/slot length mismatch");
        let mut mmap = self.mmap.lock().unwrap();
        let lo = slot.offset as usize;
        let hi = lo + slot.len as usize;
        mmap[lo..hi].copy_from_slice(data);
    }

    /// Read the slot into a fresh `Vec<u8>`.
    pub fn read(&self, slot: Slot) -> Vec<u8> {
        let mmap = self.mmap.lock().unwrap();
        let lo = slot.offset as usize;
        let hi = lo + slot.len as usize;
        mmap[lo..hi].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Unique path in the OS temp dir. The PID + static counter
    /// combo avoids collisions across parallel tests + runs.
    fn temp_path(label: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("fuel_residency_test_{pid}_{n}_{label}.bin"))
    }

    struct PathGuard(PathBuf);
    impl Drop for PathGuard {
        fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
    }

    #[test]
    fn alloc_write_read_free_roundtrip() {
        let path = temp_path("roundtrip");
        let _guard = PathGuard(path.clone());
        let rf = ResidencyFile::create(&path, 1024).unwrap();
        assert_eq!(rf.capacity(), 1024);
        assert_eq!(rf.bytes_free(), 1024);

        let data = vec![0x42u8; 100];
        let slot = rf.alloc(100).unwrap();
        rf.write(slot, &data);
        assert_eq!(rf.bytes_free(), 924);

        let back = rf.read(slot);
        assert_eq!(back, data);

        rf.free(slot);
        assert_eq!(rf.bytes_free(), 1024);
    }

    #[test]
    fn alloc_multiple_slots_and_coalesce_on_free() {
        let path = temp_path("coalesce");
        let _guard = PathGuard(path.clone());
        let rf = ResidencyFile::create(&path, 512).unwrap();
        let a = rf.alloc(100).unwrap();
        let b = rf.alloc(100).unwrap();
        let c = rf.alloc(100).unwrap();
        assert_eq!(rf.bytes_free(), 212);
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, 100);
        assert_eq!(c.offset, 200);

        // Free middle first. Freelist should keep the gap.
        rf.free(b);
        assert_eq!(rf.bytes_free(), 312);
        // Freelist should have two regions: [100..200) and [300..512).
        assert_eq!(rf.free_regions.lock().unwrap().len(), 2);

        // Free the other two; all should coalesce back to one region.
        rf.free(a);
        rf.free(c);
        assert_eq!(rf.bytes_free(), 512);
        assert_eq!(rf.free_regions.lock().unwrap().len(), 1);
    }

    #[test]
    fn alloc_returns_none_when_fragmented() {
        let path = temp_path("fragmented");
        let _guard = PathGuard(path.clone());
        let rf = ResidencyFile::create(&path, 300).unwrap();
        let a = rf.alloc(100).unwrap();
        let _b = rf.alloc(100).unwrap();
        let c = rf.alloc(100).unwrap();
        rf.free(a);
        rf.free(c);
        // bytes_free == 200 but the largest contiguous region is 100.
        assert_eq!(rf.bytes_free(), 200);
        assert!(rf.alloc(150).is_none(),
            "expected 150-byte alloc to fail due to fragmentation");
        // 100 fits.
        assert!(rf.alloc(100).is_some());
    }

    #[test]
    fn zero_byte_alloc_is_trivial() {
        let path = temp_path("zero");
        let _guard = PathGuard(path.clone());
        let rf = ResidencyFile::create(&path, 128).unwrap();
        let slot = rf.alloc(0).unwrap();
        assert_eq!(slot.len, 0);
        rf.free(slot); // free of zero-len is a no-op
        assert_eq!(rf.bytes_free(), 128);
    }

    #[test]
    fn alloc_recycles_after_free() {
        let path = temp_path("recycle");
        let _guard = PathGuard(path.clone());
        let rf = ResidencyFile::create(&path, 256).unwrap();
        let s1 = rf.alloc(200).unwrap();
        rf.write(s1, &vec![1u8; 200]);
        rf.free(s1);
        // Next alloc of same size reuses the same offset.
        let s2 = rf.alloc(200).unwrap();
        assert_eq!(s2.offset, s1.offset);
    }
}

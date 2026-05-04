//! CPU-side `BackendStorage` variant.
//!
//! A1 (this commit): minimal placeholder. `Vec<u8>` payload, no
//! alignment guarantees, no allocator integration. Just enough for
//! the scaffolding to compile and tests to construct values.
//!
//! A2 fills in the real shape: `Arc<[u8]>` for cheap Arc-based
//! sharing, 64-byte allocator alignment for AVX-512-friendly SIMD,
//! `bytemuck::cast_slice` typed views.

use std::sync::Arc;

/// CPU storage holding bytes addressable by Rust pointer.
///
/// **A1 placeholder**: the bytes live in a `Vec<u8>` wrapped in an
/// `Arc`. A2 will replace this with a 64-byte-aligned allocator that
/// returns `Arc<[u8]>` directly, plus the `bytemuck`-cast typed-view
/// surface that the kernel migration needs.
#[derive(Debug, Clone)]
pub struct CpuStorage {
    bytes: Arc<Vec<u8>>,
}

impl CpuStorage {
    /// Build a CPU storage of the given byte length, zero-initialized.
    pub fn from_zero_bytes(len_bytes: usize) -> Self {
        Self { bytes: Arc::new(vec![0u8; len_bytes]) }
    }

    /// Build a CPU storage by adopting an already-built byte vector.
    /// Useful for serialization paths and migration from the legacy
    /// `HostBuffer`-shaped CPU storage.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes: Arc::new(bytes) }
    }

    /// Total byte count.
    pub fn len_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Borrow the raw bytes immutably.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: from_zero_bytes / len_bytes / bytes round-trip.
    #[test]
    fn from_zero_bytes_round_trip() {
        let s = CpuStorage::from_zero_bytes(8);
        assert_eq!(s.len_bytes(), 8);
        assert_eq!(s.bytes(), &[0u8; 8]);
    }

    /// Smoke: from_bytes adopts the input vector.
    #[test]
    fn from_bytes_round_trip() {
        let s = CpuStorage::from_bytes(vec![1u8, 2, 3, 4]);
        assert_eq!(s.len_bytes(), 4);
        assert_eq!(s.bytes(), &[1u8, 2, 3, 4]);
    }

    /// Smoke: clone is cheap (Arc-shared bytes; no copy).
    #[test]
    fn clone_shares_bytes() {
        let a = CpuStorage::from_bytes(vec![42u8; 1024]);
        let b = a.clone();
        // Both views see the same bytes.
        assert_eq!(a.bytes(), b.bytes());
        // Pointer equality on the underlying Vec storage proves no
        // copy happened (same Arc inner).
        assert_eq!(a.bytes().as_ptr(), b.bytes().as_ptr());
    }
}

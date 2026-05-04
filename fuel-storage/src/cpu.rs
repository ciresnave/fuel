//! CPU-side `BackendStorage` variant.
//!
//! Holds bytes in a 64-byte-aligned heap allocation, wrapped in an
//! `Arc` so cheap clone (refcount bump) is the default and CoW
//! mutation kicks in only when the caller actually writes through a
//! shared handle.
//!
//! The 64-byte alignment is for AVX-512-friendly SIMD on CPU.
//! `bytemuck::cast_slice` consumes the bytes as typed slices for any
//! `Pod` type. Because the backing buffer is always 64-byte aligned,
//! every standard ML dtype (u8 through f64, half/bf16) gets aligned
//! access without an additional staging copy.

use std::sync::Arc;

use bytemuck::Pod;
use fuel_core_types::{Error, Result};

use crate::aligned::AlignedBytes;

/// Required alignment for CPU storage allocations. AVX-512-friendly;
/// also sufficient for AVX2 (32) and NEON (16).
pub const CPU_ALIGN_BYTES: usize = 64;

/// CPU storage holding bytes addressable by Rust pointer.
///
/// Cloning is cheap — bumps the inner `Arc`'s refcount without
/// copying bytes. Mutating through `bytes_mut` / `as_slice_mut`
/// performs copy-on-write: if the `Arc` is uniquely held, the bytes
/// are mutated in place; if shared, the bytes clone before mutation
/// so other holders see the unchanged data.
#[derive(Debug, Clone)]
pub struct CpuStorage {
    bytes: Arc<AlignedBytes>,
}

impl CpuStorage {
    /// Allocate `len_bytes` zero-initialized bytes, 64-byte aligned.
    pub fn from_zero_bytes(len_bytes: usize) -> Self {
        Self {
            bytes: Arc::new(AlignedBytes::new_zeroed(len_bytes, CPU_ALIGN_BYTES)),
        }
    }

    /// Allocate a fresh aligned buffer and copy `src` into it.
    pub fn from_bytes(src: &[u8]) -> Self {
        let mut buf = AlignedBytes::new_zeroed(src.len(), CPU_ALIGN_BYTES);
        if !src.is_empty() {
            buf.as_slice_mut().copy_from_slice(src);
        }
        Self { bytes: Arc::new(buf) }
    }

    /// Allocate a fresh aligned buffer and copy a typed slice into
    /// it. `T` must be `Pod` (Copy + Zeroable + 'static + no
    /// padding); covers every standard ML dtype.
    pub fn from_slice<T: Pod>(data: &[T]) -> Self {
        Self::from_bytes(bytemuck::cast_slice(data))
    }

    /// Total byte count.
    pub fn len_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Alignment in bytes (always [`CPU_ALIGN_BYTES`]).
    pub fn align(&self) -> usize {
        self.bytes.align()
    }

    /// Borrow the raw bytes immutably.
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    /// Borrow the bytes as a typed slice. Returns `Err` if the byte
    /// length isn't a multiple of `T`'s size or the buffer alignment
    /// can't satisfy `T`'s alignment requirement.
    ///
    /// Production-correct: no panic. The 64-byte allocation
    /// alignment is sufficient for every `Pod` type up to and
    /// including `u64`/`f64`, so the alignment-check failure path
    /// is unreachable for standard dtypes — but we still surface it
    /// as `Result` for unusual `Pod` types (e.g., `[u8; 128]`).
    pub fn as_slice<T: Pod>(&self) -> Result<&[T]> {
        bytemuck::try_cast_slice(self.bytes()).map_err(|e| {
            Error::Msg(format!(
                "CpuStorage::as_slice<{}>: cast failed ({e:?}); \
                 len_bytes={}, T::size={}, T::align={}",
                std::any::type_name::<T>(),
                self.len_bytes(),
                std::mem::size_of::<T>(),
                std::mem::align_of::<T>(),
            ))
            .bt()
        })
    }

    /// Copy-on-write mutable byte access. If this storage is the
    /// sole holder of its `Arc`, returns a `&mut [u8]` into the
    /// existing buffer. If the `Arc` is shared, clones the bytes
    /// into a fresh `Arc` first so other holders are unaffected.
    ///
    /// This is the centralized CoW boundary: every mutating call
    /// site goes through here, so callers don't need to remember
    /// to clone manually. Cost when `Arc` is unique: one atomic
    /// load (no copy). Cost when shared: one allocation + one
    /// memcpy of the bytes.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        Arc::make_mut(&mut self.bytes).as_slice_mut()
    }

    /// Copy-on-write typed mutable view. Same CoW semantics as
    /// `bytes_mut`; returns `Err` on a `bytemuck` cast failure.
    pub fn as_slice_mut<T: Pod>(&mut self) -> Result<&mut [T]> {
        // Cast first: we want the cast error before any CoW clone
        // happens, otherwise a misuse-by-caller forces an unnecessary
        // copy.
        let len_bytes = self.len_bytes();
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        if size != 0 && len_bytes % size != 0 {
            return Err(Error::Msg(format!(
                "CpuStorage::as_slice_mut<{}>: byte length {} not a multiple of T size {}",
                std::any::type_name::<T>(), len_bytes, size,
            )).bt());
        }
        if align > CPU_ALIGN_BYTES {
            return Err(Error::Msg(format!(
                "CpuStorage::as_slice_mut<{}>: T alignment {} exceeds buffer alignment {}",
                std::any::type_name::<T>(), align, CPU_ALIGN_BYTES,
            )).bt());
        }
        // Cast is statically guaranteed to succeed past those checks
        // (Pod size/align are the only failure modes for cast_slice_mut).
        let bytes = self.bytes_mut();
        bytemuck::try_cast_slice_mut(bytes).map_err(|e| {
            Error::Msg(format!(
                "CpuStorage::as_slice_mut<{}>: cast failed unexpectedly: {e:?}",
                std::any::type_name::<T>(),
            ))
            .bt()
        })
    }

    /// Whether this storage's bytes are uniquely owned (no other
    /// `CpuStorage` shares the same `Arc`). Useful for tests and for
    /// reasoning about CoW behavior.
    pub fn is_uniquely_owned(&self) -> bool {
        Arc::strong_count(&self.bytes) == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};

    #[test]
    fn from_zero_bytes_round_trip() {
        let s = CpuStorage::from_zero_bytes(8);
        assert_eq!(s.len_bytes(), 8);
        assert_eq!(s.bytes(), &[0u8; 8]);
        assert_eq!(s.align(), CPU_ALIGN_BYTES);
    }

    #[test]
    fn from_bytes_round_trip() {
        let s = CpuStorage::from_bytes(&[1, 2, 3, 4]);
        assert_eq!(s.len_bytes(), 4);
        assert_eq!(s.bytes(), &[1, 2, 3, 4]);
    }

    #[test]
    fn from_slice_typed() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let s = CpuStorage::from_slice(&data);
        assert_eq!(s.len_bytes(), 16);
        let typed: &[f32] = s.as_slice().expect("f32 cast");
        assert_eq!(typed, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn as_slice_supports_bf16_and_f16() {
        let bf: Vec<bf16> = vec![bf16::from_f32(1.5), bf16::from_f32(-2.5)];
        let s = CpuStorage::from_slice(&bf);
        let typed: &[bf16] = s.as_slice().expect("bf16 cast");
        assert_eq!(typed.len(), 2);
        assert_eq!(typed[0].to_f32(), 1.5);

        let h: Vec<f16> = vec![f16::from_f32(0.25)];
        let s2 = CpuStorage::from_slice(&h);
        let typed2: &[f16] = s2.as_slice().expect("f16 cast");
        assert_eq!(typed2.len(), 1);
    }

    #[test]
    fn as_slice_errors_on_size_mismatch() {
        // 5 bytes can't be split evenly into f32 (4 bytes each).
        let s = CpuStorage::from_bytes(&[0u8; 5]);
        let result: Result<&[f32]> = s.as_slice();
        assert!(result.is_err(), "size mismatch should error, not panic");
    }

    #[test]
    fn buffer_is_64_byte_aligned() {
        for size in [1, 4, 16, 64, 1024, 4096] {
            let s = CpuStorage::from_zero_bytes(size);
            let addr = s.bytes().as_ptr() as usize;
            assert_eq!(addr % 64, 0, "size={size} not 64-byte aligned");
        }
    }

    #[test]
    fn clone_is_cheap_and_shares_bytes() {
        let a = CpuStorage::from_bytes(&[42u8; 1024]);
        assert!(a.is_uniquely_owned());
        let b = a.clone();
        assert!(!a.is_uniquely_owned(), "shared after clone");
        assert!(!b.is_uniquely_owned());
        // Same pointer — no copy.
        assert_eq!(a.bytes().as_ptr(), b.bytes().as_ptr());
    }

    #[test]
    fn cow_isolates_shared_writers() {
        let a = CpuStorage::from_bytes(&[1u8, 2, 3, 4]);
        let mut b = a.clone();
        // Pre-mutation: shared.
        assert_eq!(a.bytes().as_ptr(), b.bytes().as_ptr());

        // Mutate through b. Since the Arc is shared, CoW clones.
        b.bytes_mut()[0] = 99;

        // a unchanged; b sees the new byte.
        assert_eq!(a.bytes(), &[1, 2, 3, 4]);
        assert_eq!(b.bytes(), &[99, 2, 3, 4]);
        // Pointers diverged: b got a fresh allocation.
        assert_ne!(a.bytes().as_ptr(), b.bytes().as_ptr());
    }

    #[test]
    fn cow_no_clone_when_unique() {
        let mut a = CpuStorage::from_bytes(&[1u8, 2, 3, 4]);
        let original_ptr = a.bytes().as_ptr();
        a.bytes_mut()[0] = 99;
        // Same pointer: no CoW clone (was uniquely owned).
        assert_eq!(a.bytes().as_ptr(), original_ptr);
        assert_eq!(a.bytes(), &[99, 2, 3, 4]);
    }

    #[test]
    fn typed_mut_view_round_trip() {
        let mut s = CpuStorage::from_slice(&[1.0_f32, 2.0, 3.0]);
        {
            let view: &mut [f32] = s.as_slice_mut().expect("f32 mut cast");
            view[1] = 99.0;
        }
        let view: &[f32] = s.as_slice().expect("f32 ref cast");
        assert_eq!(view, &[1.0, 99.0, 3.0]);
    }

    #[test]
    fn typed_mut_view_errors_on_size_mismatch() {
        let mut s = CpuStorage::from_bytes(&[0u8; 5]);
        let result: Result<&mut [f32]> = s.as_slice_mut();
        assert!(result.is_err(), "size mismatch must error, not panic");
    }
}

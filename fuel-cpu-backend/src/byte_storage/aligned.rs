//! Aligned byte buffer used as the storage backing for `CpuStorage`.
//!
//! Standard `Vec<u8>` doesn't guarantee alignment beyond what the
//! system allocator decides. For SIMD-heavy CPU kernels we want
//! 64-byte alignment so AVX-512 (and future AVX10) loads/stores can
//! use aligned variants. This type wraps the unsafe allocation
//! plumbing in a safe interface and implements `Clone` (deep copy)
//! and `Drop` (correct dealloc) so it can be wrapped in `Arc` for
//! cheap-clone, copy-on-write behavior at the `CpuStorage` layer.

use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::ptr::NonNull;

/// Aligned byte buffer: heap allocation with caller-specified
/// alignment. The buffer carries its alignment so `Drop` can call
/// `dealloc` with the matching `Layout`.
///
/// Owns its bytes; `Clone` performs a deep copy. To share without
/// copying, wrap in `Arc<AlignedBytes>`.
pub struct AlignedBytes {
    /// Pointer to the allocation. Dangling for zero-length buffers
    /// (matches `Vec`'s convention so the dangling pointer is never
    /// dereferenced).
    ptr: NonNull<u8>,
    /// Length in bytes.
    len: usize,
    /// Alignment in bytes — preserved so `Drop` can dealloc with the
    /// matching layout. Power of two, ≥ 1.
    align: usize,
}

// SAFETY: `AlignedBytes` owns its bytes and provides only `&` /
// `&mut` access through the type system. Bytes are POD; sharing
// `&self` across threads is safe (no interior mutability), and
// transferring ownership across threads is safe (no thread-local
// state). Same as `Vec<u8>`.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

impl AlignedBytes {
    /// Allocate `len` zero-initialized bytes aligned to `align`.
    /// `align` must be a power of two; `len` may be zero.
    pub fn new_zeroed(len: usize, align: usize) -> Self {
        assert!(
            align.is_power_of_two(),
            "AlignedBytes::new_zeroed: align ({align}) must be a power of two",
        );
        if len == 0 {
            // Zero-length buffer: use a dangling pointer aligned to
            // `align`. We never dereference it; the slice methods
            // short-circuit on `len == 0`.
            let ptr = NonNull::new(align as *mut u8).unwrap_or(NonNull::dangling());
            return Self { ptr, len: 0, align };
        }
        let layout = Layout::from_size_align(len, align)
            .expect("AlignedBytes layout: size+align overflow");
        // SAFETY: layout is non-zero (len > 0 above) and properly
        // aligned. alloc_zeroed returns either a valid pointer or
        // null on failure.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = match NonNull::new(raw) {
            Some(p) => p,
            None => handle_alloc_error(layout),
        };
        Self { ptr, len, align }
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Alignment of the underlying allocation, in bytes.
    pub fn align(&self) -> usize {
        self.align
    }

    /// Borrow the buffer immutably as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` is valid for `len` bytes (allocated above
        // and not yet freed; no mutation possible through `&self`).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Borrow the buffer mutably as a byte slice.
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: `ptr` is valid for `len` bytes; `&mut self`
        // guarantees no aliasing.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        let layout = Layout::from_size_align(self.len, self.align)
            .expect("AlignedBytes Drop: layout was valid at construction");
        // SAFETY: `ptr` was allocated with the same layout (same len
        // + same align), and we're freeing exactly once (Drop runs
        // once per value).
        unsafe { dealloc(self.ptr.as_ptr(), layout) };
    }
}

impl Clone for AlignedBytes {
    fn clone(&self) -> Self {
        let mut new = Self::new_zeroed(self.len, self.align);
        if self.len > 0 {
            new.as_slice_mut().copy_from_slice(self.as_slice());
        }
        new
    }
}

impl std::fmt::Debug for AlignedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBytes")
            .field("len", &self.len)
            .field("align", &self.align)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_aligned() {
        let buf = AlignedBytes::new_zeroed(128, 64);
        assert_eq!(buf.len(), 128);
        assert_eq!(buf.align(), 64);
        let addr = buf.as_slice().as_ptr() as usize;
        assert_eq!(addr % 64, 0, "buffer base must be 64-byte aligned");
    }

    #[test]
    fn zero_length_is_safe() {
        let buf = AlignedBytes::new_zeroed(0, 64);
        assert_eq!(buf.len(), 0);
        assert!(buf.as_slice().is_empty());
        // Drop runs without panicking on zero-length.
    }

    #[test]
    fn clone_deep_copies() {
        let mut a = AlignedBytes::new_zeroed(16, 64);
        a.as_slice_mut()[0] = 42;
        let b = a.clone();
        // Mutate a; b stays.
        a.as_slice_mut()[0] = 99;
        assert_eq!(b.as_slice()[0], 42, "clone must be a deep copy");
        assert_eq!(a.as_slice()[0], 99);
    }

    #[test]
    fn read_write_round_trip() {
        let mut buf = AlignedBytes::new_zeroed(8, 16);
        for (i, byte) in buf.as_slice_mut().iter_mut().enumerate() {
            *byte = i as u8;
        }
        assert_eq!(buf.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn allocates_zero_initialized() {
        let buf = AlignedBytes::new_zeroed(64, 64);
        assert!(buf.as_slice().iter().all(|&b| b == 0));
    }

    #[test]
    #[should_panic(expected = "must be a power of two")]
    fn rejects_non_power_of_two_align() {
        let _ = AlignedBytes::new_zeroed(8, 7);
    }
}

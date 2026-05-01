//! Pinned (page-locked) host memory as a [`HostStorage`] impl.
//!
//! Backed by `baracuda_driver::PinnedBuffer<T>`, which calls
//! `cuMemHostAlloc` so the allocation is page-locked and CUDA's async
//! H2D/D2H memcpy can fast-path on it (no staging through the driver's
//! private pinned pool). Useful when you'll upload / download the same
//! tensor many times — e.g. per-step training-loop inputs, per-token
//! KV cache reload — and the cost of a one-time pinned allocation is
//! worth the per-transfer DMA speedup.
//!
//! # Example
//!
//! ```no_run
//! use fuel_cuda_backend::{CudaDevice, PinnedHostStorage};
//! use fuel_core_types::backend::HostStorage;
//!
//! let dev = CudaDevice::new(0)?;
//! let mut buf = PinnedHostStorage::zeros_f32(&dev, 4096)?;
//! // fill the pinned buffer with data, then upload …
//! if let Some(slice) = buf.as_mut_slice_f32() {
//!     for (i, v) in slice.iter_mut().enumerate() { *v = i as f32; }
//! }
//! let view = buf.as_host_buffer_ref()?;
//! # Ok::<(), fuel_core_types::Error>(())
//! ```
//!
//! # Alternative: `PinnedRegistration`
//!
//! Baracuda also exposes `PinnedRegistration`, which pins an existing
//! Rust allocation via `cuMemHostRegister`. That's the right primitive
//! when the buffer already exists and changing its allocator isn't an
//! option; we don't wrap it here because fuel's upload path always
//! manufactures the buffer itself, so the allocate-pinned path is the
//! one that matters.

use crate::{CudaDevice, CudaError, WrapErr};
use baracuda_driver::pinned::PinnedBuffer;
use fuel_core_types::backend::HostStorage;
use fuel_core_types::{DType, Error, HostBuffer, HostBufferRef, Result};
use half::{bf16, f16};

/// Dtype-tagged pinned host allocation.
///
/// Dispatches over the dtype catalog exactly like [`HostBuffer`], but each
/// variant holds a CUDA-pinned allocation instead of a Rust `Vec<T>`.
/// Implements [`HostStorage`] so it plugs into the same upload path the
/// owned / mmap-backed storages use.
pub enum PinnedHostStorage {
    U8(PinnedBuffer<u8>),
    U32(PinnedBuffer<u32>),
    I16(PinnedBuffer<i16>),
    I32(PinnedBuffer<i32>),
    I64(PinnedBuffer<i64>),
    BF16(PinnedBuffer<bf16>),
    F16(PinnedBuffer<f16>),
    F32(PinnedBuffer<f32>),
    F64(PinnedBuffer<f64>),
    F8E4M3(PinnedBuffer<float8::F8E4M3>),
    /// Raw bytes for the sub-byte / shared-exp dummy dtypes.
    F6E2M3(PinnedBuffer<u8>),
    F6E3M2(PinnedBuffer<u8>),
    F4(PinnedBuffer<u8>),
    F8E8M0(PinnedBuffer<u8>),
}

impl std::fmt::Debug for PinnedHostStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedHostStorage")
            .field("dtype", &self.dtype())
            .field("len", &self.len())
            .finish()
    }
}

macro_rules! ctor {
    ($name:ident, $variant:ident, $t:ty) => {
        /// Allocate a zeroed pinned buffer of `len` elements.
        ///
        /// Zero-length allocations are supported (baracuda 235c37e +
        /// later): `PinnedBuffer` returns a `NonNull::dangling` sentinel
        /// matching stdlib's empty-`Vec` trick, so `Deref` into `&[T]`
        /// stays sound and `Drop` skips the free.
        pub fn $name(dev: &CudaDevice, len: usize) -> Result<Self> {
            let buf = PinnedBuffer::<$t>::new(dev.context_ref(), len).w()?;
            // cuMemHostAlloc does NOT zero the allocation; most callers
            // want a zeroed buffer, so do it here. If the caller wants
            // to skip the memset they can fill the buffer themselves
            // before any reads — but constructors that return "garbage"
            // are a footgun worth paying a memset to avoid.
            let mut this = Self::$variant(buf);
            if let Some(slice) = this.as_bytes_mut() {
                for byte in slice.iter_mut() {
                    *byte = 0;
                }
            }
            Ok(this)
        }
    };
}

impl PinnedHostStorage {
    ctor!(zeros_u8, U8, u8);
    ctor!(zeros_u32, U32, u32);
    ctor!(zeros_i16, I16, i16);
    ctor!(zeros_i32, I32, i32);
    ctor!(zeros_i64, I64, i64);
    ctor!(zeros_bf16, BF16, bf16);
    ctor!(zeros_f16, F16, f16);
    ctor!(zeros_f32, F32, f32);
    ctor!(zeros_f64, F64, f64);
    ctor!(zeros_f8e4m3, F8E4M3, float8::F8E4M3);

    /// Allocate a zeroed pinned buffer for any supported dtype.
    pub fn zeros(dev: &CudaDevice, dtype: DType, len: usize) -> Result<Self> {
        match dtype {
            DType::U8 => Self::zeros_u8(dev, len),
            DType::U32 => Self::zeros_u32(dev, len),
            DType::I16 => Self::zeros_i16(dev, len),
            DType::I32 => Self::zeros_i32(dev, len),
            DType::I64 => Self::zeros_i64(dev, len),
            DType::BF16 => Self::zeros_bf16(dev, len),
            DType::F16 => Self::zeros_f16(dev, len),
            DType::F32 => Self::zeros_f32(dev, len),
            DType::F64 => Self::zeros_f64(dev, len),
            DType::F8E4M3 => Self::zeros_f8e4m3(dev, len),
            dt => {
                Err(Error::from(CudaError::UnsupportedDtype {
                    dtype: dt,
                    op: "PinnedHostStorage::zeros",
                }))
            }
        }
    }

    /// Dtype of the allocation.
    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::F8E4M3(_) => DType::F8E4M3,
            Self::F6E2M3(_) => DType::F6E2M3,
            Self::F6E3M2(_) => DType::F6E3M2,
            Self::F4(_) => DType::F4,
            Self::F8E8M0(_) => DType::F8E8M0,
        }
    }

    /// Number of elements in the allocation.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(b) => b.len(),
            Self::U32(b) => b.len(),
            Self::I16(b) => b.len(),
            Self::I32(b) => b.len(),
            Self::I64(b) => b.len(),
            Self::BF16(b) => b.len(),
            Self::F16(b) => b.len(),
            Self::F32(b) => b.len(),
            Self::F64(b) => b.len(),
            Self::F8E4M3(b) => b.len(),
            Self::F6E2M3(b) => b.len(),
            Self::F6E3M2(b) => b.len(),
            Self::F4(b) => b.len(),
            Self::F8E8M0(b) => b.len(),
        }
    }

    /// `true` if the allocation has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the full allocation as raw bytes (mutable).
    ///
    /// Exposed mostly as a constructor helper — a generic zeroing loop
    /// — since the `ptr as *mut u8` reinterpretation is dtype-agnostic.
    /// Returns `None` for zero-length allocations so a null ptr never
    /// reaches a slice constructor.
    fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        let (ptr, bytes): (*mut u8, usize) = match self {
            Self::U8(b) => (b.as_mut_ptr() as *mut u8, b.len()),
            Self::U32(b) => (b.as_mut_ptr() as *mut u8, b.len() * 4),
            Self::I16(b) => (b.as_mut_ptr() as *mut u8, b.len() * 2),
            Self::I32(b) => (b.as_mut_ptr() as *mut u8, b.len() * 4),
            Self::I64(b) => (b.as_mut_ptr() as *mut u8, b.len() * 8),
            Self::BF16(b) => (b.as_mut_ptr() as *mut u8, b.len() * 2),
            Self::F16(b) => (b.as_mut_ptr() as *mut u8, b.len() * 2),
            Self::F32(b) => (b.as_mut_ptr() as *mut u8, b.len() * 4),
            Self::F64(b) => (b.as_mut_ptr() as *mut u8, b.len() * 8),
            Self::F8E4M3(b) => (b.as_mut_ptr() as *mut u8, b.len()),
            Self::F6E2M3(b) => (b.as_mut_ptr() as *mut u8, b.len()),
            Self::F6E3M2(b) => (b.as_mut_ptr() as *mut u8, b.len()),
            Self::F4(b) => (b.as_mut_ptr() as *mut u8, b.len()),
            Self::F8E8M0(b) => (b.as_mut_ptr() as *mut u8, b.len()),
        };
        if bytes == 0 || ptr.is_null() {
            return None;
        }
        // SAFETY: ptr is live for `bytes` bytes until Drop; exclusive
        // access via &mut self.
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, bytes) })
    }

    /// Mutable typed-slice accessors. Return `None` if the dtype doesn't
    /// match. Callers fill these before uploading; the read path goes
    /// through [`as_host_buffer_ref`](HostStorage::as_host_buffer_ref).
    pub fn as_mut_slice_u8(&mut self) -> Option<&mut [u8]> {
        if let Self::U8(b) = self { Some(&mut **b) } else { None }
    }
    pub fn as_mut_slice_u32(&mut self) -> Option<&mut [u32]> {
        if let Self::U32(b) = self { Some(&mut **b) } else { None }
    }
    pub fn as_mut_slice_f32(&mut self) -> Option<&mut [f32]> {
        if let Self::F32(b) = self { Some(&mut **b) } else { None }
    }
    pub fn as_mut_slice_f16(&mut self) -> Option<&mut [f16]> {
        if let Self::F16(b) = self { Some(&mut **b) } else { None }
    }
    pub fn as_mut_slice_bf16(&mut self) -> Option<&mut [bf16]> {
        if let Self::BF16(b) = self { Some(&mut **b) } else { None }
    }
    pub fn as_mut_slice_f64(&mut self) -> Option<&mut [f64]> {
        if let Self::F64(b) = self { Some(&mut **b) } else { None }
    }
}

impl HostStorage for PinnedHostStorage {
    fn as_host_buffer_ref(&self) -> Result<HostBufferRef<'_>> {
        Ok(match self {
            Self::U8(b) => HostBufferRef::U8(&**b),
            Self::U32(b) => HostBufferRef::U32(&**b),
            Self::I16(b) => HostBufferRef::I16(&**b),
            Self::I32(b) => HostBufferRef::I32(&**b),
            Self::I64(b) => HostBufferRef::I64(&**b),
            Self::BF16(b) => HostBufferRef::BF16(&**b),
            Self::F16(b) => HostBufferRef::F16(&**b),
            Self::F32(b) => HostBufferRef::F32(&**b),
            Self::F64(b) => HostBufferRef::F64(&**b),
            Self::F8E4M3(b) => HostBufferRef::F8E4M3(&**b),
            Self::F6E2M3(b) => HostBufferRef::F6E2M3(&**b),
            Self::F6E3M2(b) => HostBufferRef::F6E3M2(&**b),
            Self::F4(b) => HostBufferRef::F4(&**b),
            Self::F8E8M0(b) => HostBufferRef::F8E8M0(&**b),
        })
    }

    /// Default `into_host_buffer` materializes via `to_owned()` — a copy
    /// out of the pinned allocation into a regular `Vec<T>`. That's the
    /// only sound path: the `Vec` `HostBuffer` variants must own their
    /// allocator, and the pinned pointer belongs to CUDA.
    fn into_host_buffer(self) -> Result<HostBuffer> {
        Ok(self.as_host_buffer_ref()?.to_owned())
    }
}


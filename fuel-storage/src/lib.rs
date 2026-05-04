//! # fuel-storage
//!
//! Unified storage abstraction for fuel. Phase 7.5 foundation work item
//! (see [docs/storage-unification.md](../../docs/storage-unification.md)).
//!
//! `Storage` is the single entry point that holds bytes, a dtype tag,
//! and a backend memory region (closed enum over CPU/CUDA/Vulkan/Metal).
//! Backends provide *kernels* that operate on these types — backends
//! do not own their own storage type.
//!
//! This crate is the foundation that the rest of the fuel stack
//! depends on for "where the bytes live and what they mean." It does
//! not depend on any backend crate; backends depend on it.
//!
//! ## Status
//!
//! A1 (this commit): substrate scaffolding. The `BackendStorage` enum
//! has skeleton variants whose internals are placeholders. Subsequent
//! phases fill them in:
//!
//! - **A2**: real CPU storage (`bytes: Arc<[u8]>` with 64-byte
//!   alignment, allocator integration).
//! - **A3**: CUDA / Vulkan / Metal variants with their backend
//!   handles.
//! - **A4**: `BackendCapabilities` advertisement.
//! - **A5**: Router collects capabilities; builds dispatch tables.
//!
//! The legacy `fuel_core_types::Storage` continues to work in parallel
//! during the migration; this crate defines the new shape and
//! consumers migrate piecewise.

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(feature = "metal")]
pub mod metal;

pub use cpu::CpuStorage;
#[cfg(feature = "cuda")]
pub use cuda::CudaStorage;
#[cfg(feature = "vulkan")]
pub use vulkan::VulkanStorage;
#[cfg(feature = "metal")]
pub use metal::MetalStorage;

use fuel_core_types::{DType, Result};

/// Closed enum over backend storage variants. Each variant holds a
/// concrete storage type defined within this crate. Feature flags
/// gate the GPU variants so a CPU-only build doesn't carry GPU
/// dependencies.
#[derive(Debug)]
pub enum BackendStorage {
    Cpu(CpuStorage),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanStorage),
    #[cfg(feature = "metal")]
    Metal(MetalStorage),
}

/// Top-level storage type: byte-erased payload + runtime dtype tag.
/// Layout (shape + strides + start_offset) lives separately on the
/// consuming `Tensor` — `Storage` owns only the bytes and which
/// device/dtype they represent.
#[derive(Debug)]
pub struct Storage {
    /// Backend variant + the bytes themselves.
    pub inner: BackendStorage,
    /// How to interpret the bytes. Storage's `len_bytes` is the byte
    /// count; the element count is `len_bytes / dtype.size_in_bytes()`.
    pub dtype: DType,
}

/// Feature-aware match over `BackendStorage` variants. Used wherever
/// the dispatch shape `match s { Cpu(...) => ..., Cuda(...) => ... }`
/// would otherwise need `#[cfg(feature = "...")]` arms inline.
///
/// ```
/// # use fuel_storage::{BackendStorage, dispatch_storage};
/// fn len_bytes(s: &BackendStorage) -> usize {
///     dispatch_storage!(s, inner => inner.len_bytes())
/// }
/// ```
#[macro_export]
macro_rules! dispatch_storage {
    ($s:expr, $name:ident => $body:expr) => {
        match $s {
            $crate::BackendStorage::Cpu($name) => $body,
            #[cfg(feature = "cuda")]
            $crate::BackendStorage::Cuda($name) => $body,
            #[cfg(feature = "vulkan")]
            $crate::BackendStorage::Vulkan($name) => $body,
            #[cfg(feature = "metal")]
            $crate::BackendStorage::Metal($name) => $body,
        }
    };
}

impl BackendStorage {
    /// Total addressable byte count, regardless of dtype.
    pub fn len_bytes(&self) -> usize {
        dispatch_storage!(self, inner => inner.len_bytes())
    }
}

impl Storage {
    /// Build a Storage from an already-allocated backend variant
    /// plus its dtype tag.
    pub fn new(inner: BackendStorage, dtype: DType) -> Self {
        Self { inner, dtype }
    }

    /// The `DType` tag attached to these bytes.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Total addressable byte count.
    pub fn len_bytes(&self) -> usize {
        self.inner.len_bytes()
    }

    /// Element count = `len_bytes / dtype.size_in_bytes()`. Panics
    /// only via the `unreachable!` arm in `DType::size_in_bytes` for
    /// dtypes with a size definition; otherwise a clean integer
    /// division.
    pub fn elem_count(&self) -> usize {
        let bps = self.dtype.size_in_bytes();
        if bps == 0 { 0 } else { self.len_bytes() / bps }
    }
}

/// Allocate freshly on the CPU backend with the given dtype + element
/// count. Bytes are zeroed.
///
/// A1 placeholder: real allocator (with 64-byte alignment) lands in
/// A2. This stub uses `vec![0u8; ...]` so the surface compiles and
/// tests can construct Storages, but performance and alignment
/// guarantees are not yet what the design promises.
pub fn alloc_cpu_zeroed(dtype: DType, elem_count: usize) -> Result<Storage> {
    let len_bytes = elem_count.saturating_mul(dtype.size_in_bytes());
    Ok(Storage::new(
        BackendStorage::Cpu(CpuStorage::from_zero_bytes(len_bytes)),
        dtype,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: building a Storage via the CPU stub and reading back
    /// dtype + len_bytes + elem_count works.
    #[test]
    fn cpu_storage_basic_shape() {
        let s = alloc_cpu_zeroed(DType::F32, 4).expect("alloc");
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.len_bytes(), 16);
        assert_eq!(s.elem_count(), 4);
    }

    /// Smoke: dispatch_storage! macro picks the right variant arm.
    #[test]
    fn dispatch_macro_routes_to_variant() {
        let bs = BackendStorage::Cpu(CpuStorage::from_zero_bytes(8));
        let n = dispatch_storage!(&bs, inner => inner.len_bytes());
        assert_eq!(n, 8);
    }

    /// Smoke: BackendStorage::len_bytes goes through dispatch_storage!
    /// and matches the underlying CpuStorage's len_bytes.
    #[test]
    fn backend_storage_len_bytes_dispatches() {
        let bs = BackendStorage::Cpu(CpuStorage::from_zero_bytes(32));
        assert_eq!(bs.len_bytes(), 32);
    }

    /// Smoke: zero-element allocations still produce a valid Storage
    /// with elem_count 0 and dtype intact.
    #[test]
    fn zero_element_allocation() {
        let s = alloc_cpu_zeroed(DType::F64, 0).expect("alloc");
        assert_eq!(s.dtype(), DType::F64);
        assert_eq!(s.len_bytes(), 0);
        assert_eq!(s.elem_count(), 0);
    }
}

//! # fuel-storage
//!
//! Unified storage abstraction for fuel. Phase 7.5 foundation work item
//! (see [docs/storage-unification.md](../../docs/storage-unification.md)).
//!
//! `Storage` is the single entry point that holds bytes, a dtype tag,
//! and a backend memory region (closed enum over CPU/CUDA/Vulkan/Metal).
//! Backends provide *kernels* that operate on these types — backend
//! storage types live in their own crates and implement the
//! [`fuel_core_types::backend::BackendStorage`] trait.
//!
//! This crate owns the closed-enum dispatch wrapper and the public
//! `Storage` API. The per-backend storage types are imported from
//! their backend crates as feature-gated dependencies.
//!
//! ## Where things live
//!
//! - [`fuel_core_types::backend::BackendStorage`] — the abstract trait
//!   (just `len_bytes()` today; alloc/copy_from land in A4).
//! - [`fuel_cpu_backend::CpuStorageBytes`] — CPU storage (Phase A3.0).
//!   Bytes-based, 64-byte aligned, `Arc`-clonable, CoW on mutation.
//! - `fuel_metal_backend::MetalStorageBytes` — Metal storage (A3.1, pending).
//! - `fuel_cuda_backend::CudaStorageBytes` — CUDA storage (A3.2, pending).
//! - `fuel_graph_vulkan::VulkanStorageBytes` — Vulkan storage (A3.3, pending).
//!
//! ## Status
//!
//! Phase A3.0 (this commit): trait moved to `fuel_core_types::backend`,
//! `CpuStorageBytes` lives in `fuel-cpu-backend`. fuel-storage holds the
//! enum + wrapper + the `dispatch_storage!` macro, plus feature-gated
//! GPU placeholder variants. A3.1/A3.2/A3.3 replace those placeholders
//! with real types from each GPU backend.

pub mod cast_fusion;
pub mod compiled;
pub mod cost;
pub mod dispatch;
pub mod fused;
pub mod kernel;
pub mod pipelined;

pub use compiled::{compile_node, execute_compiled, CompiledNode};
pub use kernel::{KernelBindingTable, KernelDTypes, KernelRef, OpParams};
pub use pipelined::PipelinedExecutor;

/// Vulkan storage variant — re-exported from fuel-graph-vulkan when
/// the vulkan feature is enabled.
#[cfg(feature = "vulkan")]
pub use fuel_graph_vulkan::VulkanStorageBytes as VulkanStorage;

/// CUDA storage variant — re-exported from fuel-cuda-backend when
/// the cuda feature is enabled.
#[cfg(feature = "cuda")]
pub use fuel_cuda_backend::CudaStorageBytes as CudaStorage;

/// Metal storage variant — re-exported from fuel-metal-backend on
/// Apple platforms when the metal feature is enabled. The metal
/// feature has no effect on non-Apple platforms (the dep is
/// target-gated).
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use fuel_metal_backend::MetalStorageBytes as MetalStorage;

use fuel_core_types::{DType, Result};
#[cfg(any(
    feature = "vulkan",
    all(feature = "metal", any(target_os = "macos", target_os = "ios")),
))]
use fuel_core_types::Error;
use fuel_cpu_backend::CpuStorageBytes;

/// Closed enum over backend storage variants. The `Cpu` variant
/// holds [`CpuStorageBytes`] from `fuel-cpu-backend`. GPU variants
/// (feature-gated) currently hold placeholder types defined in this
/// crate; A3.1/A3.2/A3.3 replace them with the real reshaped types
/// from each GPU backend crate.
#[derive(Debug)]
pub enum BackendStorage {
    Cpu(CpuStorageBytes),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanStorage),
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
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
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            $crate::BackendStorage::Metal($name) => $body,
        }
    };
}

impl BackendStorage {
    /// Total addressable byte count, regardless of dtype.
    pub fn len_bytes(&self) -> usize {
        dispatch_storage!(self, inner => inner.len_bytes())
    }

    /// Phase 7.5 A4 substrate D2H. Read this storage's bytes back
    /// to host as a fresh `Vec<u8>`. Universal across backends —
    /// every variant can produce its bytes on demand. Used as the
    /// host-staging fallback for cross-backend `Op::Copy` /
    /// `Op::Move` and as the test-side oracle for D2H paths.
    ///
    /// CPU is a memcpy from the underlying `Arc<[u8]>`; GPU
    /// variants run a synchronous D2H. For Vulkan, this is not
    /// yet wired — the legacy `VulkanBackend::download_*` path
    /// requires a backend-runtime handle that the byte-storage
    /// type doesn't carry today; that's a follow-on commit.
    pub fn read_to_cpu_bytes(&self) -> Result<Vec<u8>> {
        match self {
            BackendStorage::Cpu(s) => Ok(s.bytes().to_vec()),
            #[cfg(feature = "cuda")]
            BackendStorage::Cuda(s) => s.to_cpu_bytes(),
            #[cfg(feature = "vulkan")]
            BackendStorage::Vulkan(_) => Err(Error::Msg(
                "BackendStorage::read_to_cpu_bytes: Vulkan D2H requires a \
                 VulkanBackend handle (allocator + queue) that this \
                 enum-level method can't reach. Use \
                 `VulkanBackend::download_bytes(&storage)` directly. \
                 Cross-backend dispatch through this method will be \
                 unified at the Router/registry layer in a later phase."
                .to_string(),
            )
            .bt()),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            BackendStorage::Metal(_) => Err(Error::Msg(
                "BackendStorage::read_to_cpu_bytes: Metal A4 D2H \
                 substrate not yet wired (follow-on commit)".to_string(),
            )
            .bt()),
        }
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

    /// Element count = `len_bytes / dtype.size_in_bytes()`.
    pub fn elem_count(&self) -> usize {
        let bps = self.dtype.size_in_bytes();
        if bps == 0 { 0 } else { self.len_bytes() / bps }
    }
}

/// Allocate freshly on the CPU backend with the given dtype + element
/// count. Bytes are zero-initialized and 64-byte aligned (suitable
/// for AVX-512 SIMD).
pub fn alloc_cpu_zeroed(dtype: DType, elem_count: usize) -> Result<Storage> {
    let len_bytes = elem_count.saturating_mul(dtype.size_in_bytes());
    Ok(Storage::new(
        BackendStorage::Cpu(CpuStorageBytes::from_zero_bytes(len_bytes)),
        dtype,
    ))
}

/// Build a CPU `Storage` from a typed slice, copying the bytes. The
/// result has the dtype matching `T` and is 64-byte aligned.
pub fn from_slice_cpu<T: bytemuck::Pod + fuel_core_types::WithDType>(
    data: &[T],
) -> Storage {
    Storage::new(
        BackendStorage::Cpu(CpuStorageBytes::from_slice(data)),
        T::DTYPE,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: building a Storage via the CPU backend and reading back
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
        let bs = BackendStorage::Cpu(CpuStorageBytes::from_zero_bytes(8));
        let n = dispatch_storage!(&bs, inner => inner.len_bytes());
        assert_eq!(n, 8);
    }

    /// Smoke: BackendStorage::len_bytes goes through dispatch_storage!
    /// and matches the underlying CpuStorageBytes len_bytes.
    #[test]
    fn backend_storage_len_bytes_dispatches() {
        let bs = BackendStorage::Cpu(CpuStorageBytes::from_zero_bytes(32));
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

    /// Smoke: from_slice_cpu preserves dtype + values via Pod cast.
    #[test]
    fn from_slice_cpu_round_trip() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let s = from_slice_cpu(&data);
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.elem_count(), 4);
        assert_eq!(s.len_bytes(), 16);
    }

    /// A4: BackendStorage::read_to_cpu_bytes returns the exact byte
    /// stream on the CPU variant.
    #[test]
    fn read_to_cpu_bytes_cpu_variant() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        let s = from_slice_cpu(&data);
        let bytes = s.inner.read_to_cpu_bytes().expect("d2h");
        assert_eq!(bytes.len(), 16);
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        assert_eq!(got, &data);
    }

    /// A4: alloc symmetry — CpuStorageBytes::alloc and from_zero_bytes
    /// produce the same shape.
    #[test]
    fn cpu_storage_alloc_alias() {
        let a = fuel_cpu_backend::CpuStorageBytes::alloc(24);
        let b = fuel_cpu_backend::CpuStorageBytes::from_zero_bytes(24);
        assert_eq!(a.len_bytes(), b.len_bytes());
        assert_eq!(a.bytes(), b.bytes());
    }
}

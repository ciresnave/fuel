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
//! This crate now owns ONLY the closed-enum dispatch wrapper + the
//! public `Storage` API. Dispatch infrastructure (KernelBindingTable,
//! registration wrappers, CompiledNode, ExecutionPlan, the picker,
//! FusedKernelRegistry, cost functions, the PipelinedExecutor) was
//! extracted to [`fuel-dispatch`](../../fuel-dispatch/) 2026-05-31;
//! see [docs/session-prompts/dispatch-move-to-fuel-core.md](
//! ../../docs/session-prompts/dispatch-move-to-fuel-core.md) for the
//! move's rationale.
//!
//! ## Where things live
//!
//! - [`fuel_core_types::backend::BackendStorage`] — the abstract trait
//!   (just `len_bytes()` today; alloc/copy_from land in A4).
//! - [`fuel_cpu_backend::CpuStorageBytes`] — CPU storage (Phase A3.0).
//!   Bytes-based, 64-byte aligned, `Arc`-clonable, CoW on mutation.
//! - `fuel_metal_backend::MetalStorageBytes` — Metal storage (A3.1).
//! - `fuel_cuda_backend::CudaStorageBytes` — CUDA storage (A3.2).
//! - `fuel_vulkan_backend::VulkanStorageBytes` — Vulkan storage (A3.3).
//! - **`fuel_dispatch`** — every binding-table, registration wrapper,
//!   picker, executor, and cost-fn that used to live here.

/// Vulkan storage variant — re-exported from fuel-vulkan-backend when
/// the vulkan feature is enabled.
#[cfg(feature = "vulkan")]
pub use fuel_vulkan_backend::VulkanStorageBytes as VulkanStorage;

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
use fuel_cpu_backend::CpuStorageBytes;

/// Closed enum over backend storage variants. The `Cpu` variant
/// holds [`CpuStorageBytes`] from `fuel-cpu-backend`. GPU variants
/// (feature-gated) hold the per-backend `*StorageBytes` types from
/// each backend crate.
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

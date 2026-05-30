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
//! - `fuel_vulkan_backend::VulkanStorageBytes` — Vulkan storage (A3.3, pending).
//!
//! ## Status
//!
//! Phase A3.0 (this commit): trait moved to `fuel_core_types::backend`,
//! `CpuStorageBytes` lives in `fuel-cpu-backend`. fuel-storage holds the
//! enum + wrapper + the `dispatch_storage!` macro, plus feature-gated
//! GPU placeholder variants. A3.1/A3.2/A3.3 replace those placeholders
//! with real types from each GPU backend.

pub mod baracuda_dispatch;
pub mod cast_fusion;
pub mod compiled;
pub mod cost;
pub mod dispatch;
pub mod fused;
pub mod kernel;
pub mod pipelined;
pub mod plan;
pub mod vulkan_dispatch;

pub use compiled::{compile_node, execute_compiled, CompiledNode};
pub use kernel::{KernelBindingTable, KernelDTypes, KernelRef, OpParams};
pub use pipelined::PipelinedExecutor;
pub use plan::{compile_plan, resolve_kernel, ExecutionPlan, NodeKernelBinding, TolerancePolicy};

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

    /// Bridge-retirement Phase 3a follow-up: realize an
    /// `Op::Alloc → Op::ZeroFill` chain on CPU and verify the
    /// resulting storage is zeroed. Exercises the executor's
    /// `WorkItemKind::ZeroFill` arm on the CPU branch (which does an
    /// explicit `bytes_mut().fill(0)` even though CPU's Op::Alloc
    /// already returns zero-init storage — the explicit fill is the
    /// architecturally-honest path; future uninit-CPU-alloc would
    /// still need it).
    #[test]
    fn op_zero_fill_cpu_zeroes_alloc_output() {
        use crate::pipelined::{PipelinedExecutor, StorageCache};
        use fuel_core_types::{DeviceLocation, Shape};
        use fuel_graph::{Graph, Node, Op};
        use std::sync::{Arc, RwLock};

        let graph = Arc::new(RwLock::new(Graph::new()));
        let zero_id = {
            let mut g = graph.write().unwrap();
            let alloc_id = g.push(Node {
                op: Op::Alloc { target: DeviceLocation::Cpu },
                inputs: vec![],
                shape: Shape::from_dims(&[16]),
                dtype: DType::F32,
            });
            g.push(Node {
                op: Op::ZeroFill,
                inputs: vec![alloc_id],
                shape: Shape::from_dims(&[16]),
                dtype: DType::F32,
            })
        };

        let (storage_arc, _layout) =
            PipelinedExecutor::realize(graph, zero_id, StorageCache::new())
                .expect("Op::Alloc → Op::ZeroFill realize");

        let guard = storage_arc.read().unwrap();
        match &guard.inner {
            BackendStorage::Cpu(c) => {
                assert_eq!(c.len_bytes(), 64);  // 16 * sizeof(f32) = 64
                let typed: &[f32] = c.as_slice().expect("f32 cast");
                assert!(typed.iter().all(|&x| x == 0.0_f32),
                    "Op::ZeroFill must produce all-zero bytes; got {typed:?}");
            }
            other => panic!("expected CPU storage; got {other:?}"),
        }
    }

    /// Bridge-retirement Phase 3a (post-9c): realize an `Op::Alloc {
    /// target: Cpu }` and verify the executor produces a zero-init CPU
    /// storage. Exercises the executor's `WorkItemKind::Alloc` arm on
    /// the CPU branch (the only branch that doesn't need a device
    /// anchor in the cache).
    #[test]
    fn op_alloc_cpu_produces_zero_init_storage() {
        use crate::pipelined::{PipelinedExecutor, StorageCache};
        use fuel_core_types::{DeviceLocation, Shape};
        use fuel_graph::{Graph, Node, Op};
        use std::sync::{Arc, RwLock};

        let graph = Arc::new(RwLock::new(Graph::new()));
        let alloc_id = {
            let mut g = graph.write().unwrap();
            g.push(Node {
                op: Op::Alloc { target: DeviceLocation::Cpu },
                inputs: vec![],
                shape: Shape::from_dims(&[8]),
                dtype: DType::F32,
            })
        };

        let (storage_arc, _layout) =
            PipelinedExecutor::realize(graph, alloc_id, StorageCache::new())
                .expect("Op::Alloc { target: Cpu } realize");

        let guard = storage_arc.read().unwrap();
        match &guard.inner {
            BackendStorage::Cpu(c) => {
                assert_eq!(c.len_bytes(), 32);  // 8 * sizeof(f32) = 32
                let typed: &[f32] = c.as_slice().expect("f32 cast");
                assert!(typed.iter().all(|&x| x == 0.0_f32),
                    "Op::Alloc must produce zero-init storage; got {typed:?}");
            }
            other => panic!("Op::Alloc {{ target: Cpu }} must produce \
                BackendStorage::Cpu; got {other:?}"),
        }
    }

    /// Bridge-retirement Phase 2 (post-9c): round-trip a CPU storage
    /// through `Op::Copy { target: Cpu }` via the PipelinedExecutor.
    /// Replaces the deleted `read_to_cpu_bytes_cpu_variant` test.
    ///
    /// The CPU→CPU Copy kernel registered in
    /// [`crate::dispatch::register_cpu_kernels`] is the universal
    /// memcpy noop; this exercises the binding-table-dispatch path
    /// end-to-end on the universal fallback. Per-source-backend D2H
    /// (CUDA, Vulkan) is exercised by the live-GPU test sets.
    #[test]
    fn op_copy_to_cpu_round_trip_via_pipelined_executor() {
        use crate::pipelined::{PipelinedExecutor, StorageCache};
        use fuel_core_types::{probe::BackendId, DeviceLocation, Shape};
        use fuel_graph::{Graph, Node, NodeId, Op};
        use std::sync::{Arc, RwLock};

        let data = [1.0_f32, 2.0, 3.0, 4.0];
        let src_storage = from_slice_cpu(&data);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (src_id, copy_id): (NodeId, NodeId) = {
            let mut g = graph.write().unwrap();
            let src = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[4]),
                dtype: DType::F32,
            });
            let copy = g.push(Node {
                op: Op::Copy { target: DeviceLocation::Cpu },
                inputs: vec![src],
                shape: Shape::from_dims(&[4]),
                dtype: DType::F32,
            });
            g.set_target_backend(copy, BackendId::Cpu);
            (src, copy)
        };

        let mut cache = StorageCache::new();
        cache.insert(src_id, Arc::new(RwLock::new(src_storage)));

        let (result_arc, _result_layout) =
            PipelinedExecutor::realize(graph, copy_id, cache)
                .expect("Op::Copy { target: Cpu } realize");

        let guard = result_arc.read().unwrap();
        if let BackendStorage::Cpu(c) = &guard.inner {
            assert_eq!(c.len_bytes(), 16);
            let got: &[f32] = c.as_slice().expect("f32 cast");
            assert_eq!(got, &data);
        } else {
            panic!("Op::Copy {{ target: Cpu }} must produce BackendStorage::Cpu");
        }
    }

    /// Phase 3 of the in-place ops infrastructure
    /// (docs/session-prompts/in-place-ops-infrastructure.md): realize a
    /// graph node `Op::Fused(INPLACE_AFFINE, {mul, add})` on top of a
    /// CPU const tensor and verify the executor mutates the const's
    /// bytes via the `WorkItemKind::InplaceKernel` arm + the
    /// `inplace_affine_f32_cpu_wrapper` registered in the binding
    /// table.
    #[test]
    fn op_inplace_affine_cpu_mutates_target_storage() {
        use crate::pipelined::{PipelinedExecutor, StorageCache};
        use fuel_core_types::{probe::BackendId, Shape};
        use fuel_graph::{
            registry::{FusedOpParams, FusedOps},
            Graph, Node, NodeId, Op,
        };
        use std::sync::{Arc, RwLock};

        let data = [1.0_f32, 2.0, 3.0, 4.0];
        let src_storage = from_slice_cpu(&data);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (src_id, affine_id): (NodeId, NodeId) = {
            let mut g = graph.write().unwrap();
            let src = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[4]),
                dtype: DType::F32,
            });
            let affine = g.push(Node {
                op: Op::Fused(
                    FusedOps::INPLACE_AFFINE,
                    FusedOpParams::InplaceAffine { mul: 2.0, add: 0.5 },
                ),
                inputs: vec![src],
                shape: Shape::from_dims(&[4]),
                dtype: DType::F32,
            });
            g.set_target_backend(affine, BackendId::Cpu);
            (src, affine)
        };

        let mut cache = StorageCache::new();
        cache.insert(src_id, Arc::new(RwLock::new(src_storage)));

        let (result_arc, _layout) =
            PipelinedExecutor::realize(graph, affine_id, cache)
                .expect("Op::Fused(INPLACE_AFFINE, _) realize");

        let guard = result_arc.read().unwrap();
        match &guard.inner {
            BackendStorage::Cpu(c) => {
                let got: &[f32] = c.as_slice().expect("f32 cast");
                // 2 · [1,2,3,4] + 0.5 = [2.5, 4.5, 6.5, 8.5]
                assert_eq!(got, &[2.5_f32, 4.5, 6.5, 8.5]);
            }
            other => panic!("expected CPU storage; got {other:?}"),
        }
    }

    /// Phase 3e of the in-place ops infrastructure: realize a graph
    /// `Op::ReluInplace` node on top of a CPU const tensor and verify
    /// the executor mutates the const's bytes via
    /// `relu_inplace_f32_cpu_wrapper`. Smoke test for the unary
    /// in-place dispatch path; the other 4 activations (Silu/Gelu/Tanh/
    /// Sigmoid) go through the identical wrapper+macro layer so a
    /// single round-trip is sufficient validation at this layer.
    #[test]
    fn op_relu_inplace_cpu_mutates_target_storage() {
        use crate::pipelined::{PipelinedExecutor, StorageCache};
        use fuel_core_types::{probe::BackendId, Shape};
        use fuel_graph::{Graph, Node, NodeId, Op};
        use std::sync::{Arc, RwLock};

        let data = [-1.0_f32, 0.0, 1.0, 2.0];
        let src_storage = from_slice_cpu(&data);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (src_id, relu_id): (NodeId, NodeId) = {
            let mut g = graph.write().unwrap();
            let src = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[4]),
                dtype: DType::F32,
            });
            let relu = g.push(Node {
                op: Op::ReluInplace,
                inputs: vec![src],
                shape: Shape::from_dims(&[4]),
                dtype: DType::F32,
            });
            g.set_target_backend(relu, BackendId::Cpu);
            (src, relu)
        };

        let mut cache = StorageCache::new();
        cache.insert(src_id, Arc::new(RwLock::new(src_storage)));

        let (result_arc, _layout) =
            PipelinedExecutor::realize(graph, relu_id, cache)
                .expect("Op::ReluInplace realize");

        let guard = result_arc.read().unwrap();
        match &guard.inner {
            BackendStorage::Cpu(c) => {
                let got: &[f32] = c.as_slice().expect("f32 cast");
                // ReLU: [-1, 0, 1, 2] -> [0, 0, 1, 2]
                assert_eq!(got, &[0.0_f32, 0.0, 1.0, 2.0]);
            }
            other => panic!("expected CPU storage; got {other:?}"),
        }
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

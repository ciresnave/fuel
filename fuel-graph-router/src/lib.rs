//! # fuel-graph-router
//!
//! Multi-backend executor for the fuel lazy-graph layer. Phase 2+3 of
//! the unified scheduler work.
//!
//! The [`Router`] holds `Vec<Arc<dyn DynBackend>>` — one entry per
//! concrete backend available at runtime (CPU always; Vulkan/CUDA
//! when feature-enabled AND a device was attached). It implements
//! [`GraphBackend`] with `Storage = AnyStorage`, dispatching each op
//! to whichever backend matches the input storage's device identity.
//!
//! ## Design choice: dyn over enum
//!
//! Originally the Router was an enum over concrete backend variants.
//! The pivot to trait objects gives:
//!
//! - Zero central coupling: adding a new backend means implementing
//!   [`DynBackend`] for it, nothing in this crate changes.
//! - Multi-instance naturally: `Vec<Arc<dyn DynBackend>>` holds as
//!   many Vulkan GPUs (or whatever) as the user attaches.
//! - Plugin-style third-party backends become possible.
//!
//! Cost: one vtable call per op. Negligible at op-millisecond
//! granularity.
//!
//! ## What this doesn't do yet
//!
//! - Auto-insert moves when a node's inputs live on different
//!   devices. Callers use [`fuel_graph::Tensor::copy_to_device`] or
//!   invoke [`Router::copy_to`] directly. Phase 3.5 adds auto-insertion.
//! - Honor `graph.placement()` as a dispatch hint. For now routing
//!   goes by input device. Phase 4's scheduler closes that loop.
//! - Pick an execution order that minimizes transfers. Phase 4.

pub mod scheduler;
pub use scheduler::{
    apply_placement, BaselineRule, ConstLoweringRule, GraphMutatingSchedulerRule, Placement,
    RuleScheduler, Scheduler, SchedulerRule, SimpleScheduler,
};

pub mod residency_planner;
pub use residency_planner::{LiveRange, ResidencyPlanner, ResidencyReport};

pub mod residency_eviction;
pub use residency_eviction::ResidencyEvictionRule;

use fuel_core_types::{bail, Capability, DType, DeviceLocation, HostBuffer, Layout, Result, Shape};
use fuel_core_types::dispatch::{Criterion, DispatchTable, OpKind, Pick, SizeClass};
use fuel_core_types::probe::BackendId;
use fuel_graph_cpu::CpuBackend;
use fuel_graph_executor::{BinaryOp, GraphBackend, UnaryOp};
use fuel_reference_backend::exec::AnyRefTensor;
use std::sync::Arc;

#[cfg(feature = "aocl")]
use fuel_aocl_cpu_backend::AoclBackend;

#[cfg(feature = "onemkl")]
use fuel_mkl_cpu_backend::MklBackend;

#[cfg(feature = "vulkan")]
use fuel_vulkan_backend::{VulkanBackend, VulkanStorage};

#[cfg(feature = "cuda")]
use fuel_cuda_backend::CudaBackend;
#[cfg(feature = "cuda")]
use fuel_cuda_backend::CudaStorage;

// -- AnyStorage ------------------------------------------------------------

/// Cross-backend storage wrapper. Each variant carries the native
/// storage type for one backend.
pub enum AnyStorage {
    Cpu(AnyRefTensor),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanStorage),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
}

impl AnyStorage {
    pub fn device(&self) -> DeviceLocation {
        match self {
            AnyStorage::Cpu(_) => DeviceLocation::Cpu,
            #[cfg(feature = "vulkan")]
            AnyStorage::Vulkan(_) => DeviceLocation::Vulkan { gpu_id: 0 },
            #[cfg(feature = "cuda")]
            AnyStorage::Cuda(_) => DeviceLocation::Cuda { gpu_id: 0 },
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            AnyStorage::Cpu(t) => t.dtype(),
            #[cfg(feature = "vulkan")]
            AnyStorage::Vulkan(s) => s.dtype,
            #[cfg(feature = "cuda")]
            AnyStorage::Cuda(s) => s.dtype(),
        }
    }

    #[allow(unreachable_patterns)]
    pub fn as_cpu(&self) -> Result<&AnyRefTensor> {
        match self { AnyStorage::Cpu(s) => Ok(s), _ => bail!("AnyStorage: expected Cpu, got {:?}", self.device()) }
    }

    #[cfg(feature = "vulkan")]
    pub fn as_vulkan(&self) -> Result<&VulkanStorage> {
        match self { AnyStorage::Vulkan(s) => Ok(s), _ => bail!("AnyStorage: expected Vulkan, got {:?}", self.device()) }
    }

    #[cfg(feature = "cuda")]
    pub fn as_cuda(&self) -> Result<&CudaStorage> {
        match self { AnyStorage::Cuda(s) => Ok(s), _ => bail!("AnyStorage: expected Cuda, got {:?}", self.device()) }
    }
}

fn same_device(a: &AnyStorage, b: &AnyStorage, op: &str) -> Result<DeviceLocation> {
    let (da, db) = (a.device(), b.device());
    if da != db {
        bail!("Router::{op}: input device mismatch ({da:?} vs {db:?}); insert Op::Copy first");
    }
    Ok(da)
}

// -- DynBackend trait ------------------------------------------------------

/// Object-safe backend trait where every op consumes and produces
/// [`AnyStorage`]. Concrete backends like [`CpuBackend`] impl both
/// [`GraphBackend`] (typed, zero-cost) and [`DynBackend`] (erased,
/// one-vtable-hop). The Router holds `Vec<Arc<dyn DynBackend>>` and
/// dispatches by device identity.
///
/// Implementations should downcast AnyStorage to their concrete storage
/// type, delegate to the typed GraphBackend method, and rewrap the
/// result. Most methods have default impls that bail, so implementers
/// only need to override the ops they want to support directly;
/// anything else falls through to the executor's CPU fallback.
// NOTE: no Send+Sync bound yet. VulkanBackend today is neither Send
// nor Sync (queue access is single-threaded). When a multi-threaded
// scheduler lands we'll add per-backend sync wrappers (Mutex) rather
// than force the unsound bound onto all backends.
pub trait DynBackend {
    /// The device identity this backend represents.
    fn device(&self) -> DeviceLocation;

    /// Stable identifier for this backend implementation, matching
    /// the Phase 6b probe's [`BackendId`]. Routers with multiple
    /// backends sharing the same `device()` (e.g. CpuBackend and
    /// AoclBackend both at `DeviceLocation::Cpu`) use this to
    /// dispatch the right one based on the empirical dispatch
    /// table's pick.
    fn backend_id(&self) -> BackendId;

    /// Kernel-source tag distinguishing this backend instance when
    /// multiple instances share the same `(backend_id, device)` pair.
    /// Mirrors `Pick::kernel_source` / `BindingEntry::kernel_source`:
    /// `"portable-cpu"`, `"aocl"`, `"mkl"`, etc.
    ///
    /// Routers use this to resolve a [`Pick`] to the specific
    /// `DynBackend` whose kernel produced the winning measurement.
    /// Default `""` matches the legacy single-impl-per-slot
    /// convention and the `Pick::kernel_source` fallback for cells
    /// with no sibling — appropriate for Vulkan / CUDA where each
    /// `(BackendId, device)` has exactly one registered backend.
    fn kernel_source(&self) -> &'static str { "" }

    /// Ops this backend implements natively. The Router/scheduler
    /// consults this slice to route nodes, plan transfers, and
    /// (Phase 4) weight cost models. Default: empty slice.
    ///
    /// Every capability returned here must correspond to a working
    /// method below — stale announcements (claim `Rope`, bail at
    /// runtime) are a bug.
    fn capabilities(&self) -> &[Capability] { &[] }

    // -- memory --
    fn alloc_zeros(&self, _shape: &Shape, _dtype: DType) -> Result<AnyStorage> {
        bail!("DynBackend: alloc_zeros not implemented")
    }
    fn upload(&self, _buf: &HostBuffer, _shape: &Shape) -> Result<AnyStorage> {
        bail!("DynBackend: upload not implemented")
    }
    fn download(&self, _storage: &AnyStorage) -> Result<HostBuffer> {
        bail!("DynBackend: download not implemented")
    }
    fn try_clone(&self, _storage: &AnyStorage, _layout: &Layout) -> Result<AnyStorage> {
        bail!("DynBackend: try_clone not implemented")
    }
    fn copy_strided_src(
        &self, _src: &AnyStorage, _dst: &mut AnyStorage,
        _dst_offset: usize, _src_layout: &Layout,
    ) -> Result<()> {
        bail!("DynBackend: copy_strided_src not implemented")
    }

    // -- compute --
    fn matmul(
        &self, _a: &AnyStorage, _b: &AnyStorage,
        _bmnk: (usize, usize, usize, usize), _la: &Layout, _lb: &Layout,
    ) -> Result<AnyStorage> { bail!("DynBackend: matmul not implemented") }

    fn unary(
        &self, _op: UnaryOp, _a: &AnyStorage, _layout: &Layout,
    ) -> Result<AnyStorage> { bail!("DynBackend: unary not implemented") }

    fn binary(
        &self, _op: BinaryOp, _a: &AnyStorage, _b: &AnyStorage,
        _la: &Layout, _lb: &Layout,
    ) -> Result<AnyStorage> { bail!("DynBackend: binary not implemented") }

    fn affine(
        &self, _a: &AnyStorage, _layout: &Layout, _mul: f64, _add: f64,
    ) -> Result<AnyStorage> { bail!("DynBackend: affine not implemented") }

    fn powf(
        &self, _a: &AnyStorage, _layout: &Layout, _exp: f64,
    ) -> Result<AnyStorage> { bail!("DynBackend: powf not implemented") }

    fn cast(
        &self, _a: &AnyStorage, _layout: &Layout, _dtype: DType,
    ) -> Result<AnyStorage> { bail!("DynBackend: cast not implemented") }

    fn reduce(
        &self, _op: fuel_core_types::op::ReduceOp, _a: &AnyStorage,
        _layout: &Layout, _dims: &[usize],
    ) -> Result<AnyStorage> { bail!("DynBackend: reduce not implemented") }

    fn softmax_last_dim(
        &self, _a: &AnyStorage, _layout: &Layout,
    ) -> Result<AnyStorage> { bail!("DynBackend: softmax_last_dim not implemented") }

    fn index_select(
        &self, _src: &AnyStorage, _ids: &AnyStorage,
        _src_l: &Layout, _ids_l: &Layout, _dim: usize,
    ) -> Result<AnyStorage> { bail!("DynBackend: index_select not implemented") }

    fn gather(
        &self, _src: &AnyStorage, _ids: &AnyStorage,
        _src_l: &Layout, _ids_l: &Layout, _dim: usize,
    ) -> Result<AnyStorage> { bail!("DynBackend: gather not implemented") }

    // -- quantized matmul, one method per quant format --

    fn matmul_q4_0(
        &self, _a: &AnyStorage, _w: &AnyStorage,
        _k: usize, _n: usize, _la: &Layout,
    ) -> Result<AnyStorage> { bail!("DynBackend: matmul_q4_0 not implemented") }

    fn matmul_q4_km(
        &self, _a: &AnyStorage, _w: &AnyStorage,
        _k: usize, _n: usize, _la: &Layout,
    ) -> Result<AnyStorage> { bail!("DynBackend: matmul_q4_km not implemented") }

    fn matmul_q8_0(
        &self, _a: &AnyStorage, _w: &AnyStorage,
        _k: usize, _n: usize, _la: &Layout,
    ) -> Result<AnyStorage> { bail!("DynBackend: matmul_q8_0 not implemented") }

    // -- KV cache Q8 quantize / dequantize --

    fn quantize_q8_0(
        &self, _src_f32: &AnyStorage, _n_elements: usize,
    ) -> Result<AnyStorage> { bail!("DynBackend: quantize_q8_0 not implemented") }

    fn dequantize_q8_0(
        &self, _blocks: &AnyStorage, _n_blocks: usize,
    ) -> Result<AnyStorage> { bail!("DynBackend: dequantize_q8_0 not implemented") }

    // -- fused forward --

    fn rms_norm_last_dim(
        &self, _a: &AnyStorage, _layout: &Layout, _eps: f64,
    ) -> Result<AnyStorage> { bail!("DynBackend: rms_norm_last_dim not implemented") }

    fn concat_along_dim(
        &self, _a: &AnyStorage, _b: &AnyStorage, _dim: usize,
        _a_layout: &Layout, _b_layout: &Layout,
    ) -> Result<AnyStorage> { bail!("DynBackend: concat_along_dim not implemented") }

    fn rope(
        &self, _x: &AnyStorage, _cos: &AnyStorage, _sin: &AnyStorage,
        _x_layout: &Layout, _cos_layout: &Layout, _sin_layout: &Layout,
    ) -> Result<AnyStorage> { bail!("DynBackend: rope not implemented") }

    fn add_assign_scaled(
        &self, _dst: &mut AnyStorage, _src: &AnyStorage, _scale: f32,
    ) -> Result<()> { bail!("DynBackend: add_assign_scaled not implemented") }

    // -- fused backward (training) --

    fn rms_norm_last_dim_backward(
        &self, _x: &AnyStorage, _upstream: &AnyStorage,
        _x_layout: &Layout, _up_layout: &Layout, _eps: f64,
    ) -> Result<AnyStorage> { bail!("DynBackend: rms_norm_last_dim_backward not implemented") }

    fn layer_norm_last_dim_backward(
        &self, _x: &AnyStorage, _upstream: &AnyStorage,
        _x_layout: &Layout, _up_layout: &Layout, _eps: f64,
    ) -> Result<AnyStorage> { bail!("DynBackend: layer_norm_last_dim_backward not implemented") }

    fn softmax_last_dim_backward(
        &self, _y: &AnyStorage, _upstream: &AnyStorage,
        _y_layout: &Layout, _up_layout: &Layout,
    ) -> Result<AnyStorage> { bail!("DynBackend: softmax_last_dim_backward not implemented") }

    /// Copy storage to `target`. Source stays resident on its original
    /// device (non-destructive). Same-device is either try_clone or
    /// returns self. Cross-device on a single-backend impl errors —
    /// the Router intercepts cross-device copies and handles them via
    /// host round-trip.
    fn copy_to(
        &self, storage: &AnyStorage, layout: &Layout, target: DeviceLocation,
    ) -> Result<AnyStorage> {
        if storage.device() == target {
            return self.try_clone(storage, layout);
        }
        bail!(
            "DynBackend({:?}): cannot copy to {target:?}; single-backend has no peer",
            self.device()
        )
    }
}

// -- Macro for routing a method on one backend -----------------------------
//
// The pattern repeats for every op: downcast one or more AnyStorage
// inputs, call the typed GraphBackend method, rewrap the output.

/// Generate a `DynBackend::op` impl that routes to one concrete
/// backend. `$cast` is the AnyStorage accessor (e.g. `as_cpu`);
/// `$wrap` is the AnyStorage variant constructor (e.g. `AnyStorage::Cpu`).
macro_rules! impl_dyn_backend {
    // Full form: caller specifies a kernel_source tag (e.g.
    // "portable-cpu", "aocl", "mkl") that disambiguates siblings
    // sharing the same `(BackendId, DeviceLocation)`. Required for
    // every CPU-family backend now that AOCL/MKL register under
    // `BackendId::Cpu` and the dispatch table tracks them via
    // `Pick::kernel_source`.
    ($backend:ty, $cast:ident, $wrap:path, $device:expr, $caps:expr, $backend_id:expr, $kernel_source:expr) => {
        impl DynBackend for $backend {
            fn device(&self) -> DeviceLocation { $device }
            fn backend_id(&self) -> BackendId { $backend_id }
            fn kernel_source(&self) -> &'static str { $kernel_source }
            fn capabilities(&self) -> &[Capability] { $caps }

            fn alloc_zeros(&self, shape: &Shape, dtype: DType) -> Result<AnyStorage> {
                <$backend as GraphBackend>::alloc_zeros(self, shape, dtype).map($wrap)
            }
            fn upload(&self, buf: &HostBuffer, shape: &Shape) -> Result<AnyStorage> {
                <$backend as GraphBackend>::upload(self, buf, shape).map($wrap)
            }
            fn download(&self, storage: &AnyStorage) -> Result<HostBuffer> {
                <$backend as GraphBackend>::download(self, storage.$cast()?)
            }
            fn try_clone(&self, storage: &AnyStorage, layout: &Layout) -> Result<AnyStorage> {
                <$backend as GraphBackend>::try_clone(self, storage.$cast()?, layout).map($wrap)
            }
            fn copy_strided_src(
                &self, src: &AnyStorage, dst: &mut AnyStorage,
                dst_offset: usize, src_layout: &Layout,
            ) -> Result<()> {
                let s = src.$cast()?;
                let d = match dst {
                    $wrap(inner) => inner,
                    _ => bail!("copy_strided_src: dst device mismatch"),
                };
                <$backend as GraphBackend>::copy_strided_src(self, s, d, dst_offset, src_layout)
            }
            fn matmul(
                &self, a: &AnyStorage, b: &AnyStorage,
                bmnk: (usize, usize, usize, usize), la: &Layout, lb: &Layout,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::matmul(self, a.$cast()?, b.$cast()?, bmnk, la, lb).map($wrap)
            }
            fn unary(&self, op: UnaryOp, a: &AnyStorage, layout: &Layout) -> Result<AnyStorage> {
                <$backend as GraphBackend>::unary(self, op, a.$cast()?, layout).map($wrap)
            }
            fn binary(
                &self, op: BinaryOp, a: &AnyStorage, b: &AnyStorage,
                la: &Layout, lb: &Layout,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::binary(self, op, a.$cast()?, b.$cast()?, la, lb).map($wrap)
            }
            fn affine(
                &self, a: &AnyStorage, layout: &Layout, mul: f64, add: f64,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::affine(self, a.$cast()?, layout, mul, add).map($wrap)
            }
            fn powf(
                &self, a: &AnyStorage, layout: &Layout, exp: f64,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::powf(self, a.$cast()?, layout, exp).map($wrap)
            }
            fn cast(
                &self, a: &AnyStorage, layout: &Layout, dtype: DType,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::cast(self, a.$cast()?, layout, dtype).map($wrap)
            }
            fn reduce(
                &self, op: fuel_core_types::op::ReduceOp, a: &AnyStorage,
                layout: &Layout, dims: &[usize],
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::reduce(self, op, a.$cast()?, layout, dims).map($wrap)
            }
            fn softmax_last_dim(&self, a: &AnyStorage, layout: &Layout) -> Result<AnyStorage> {
                <$backend as GraphBackend>::softmax_last_dim(self, a.$cast()?, layout).map($wrap)
            }
            fn index_select(
                &self, src: &AnyStorage, ids: &AnyStorage,
                src_l: &Layout, ids_l: &Layout, dim: usize,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::index_select(
                    self, src.$cast()?, ids.$cast()?, src_l, ids_l, dim
                ).map($wrap)
            }
            fn gather(
                &self, src: &AnyStorage, ids: &AnyStorage,
                src_l: &Layout, ids_l: &Layout, dim: usize,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::gather(
                    self, src.$cast()?, ids.$cast()?, src_l, ids_l, dim
                ).map($wrap)
            }
            fn matmul_q4_0(
                &self, a: &AnyStorage, w: &AnyStorage,
                k: usize, n: usize, la: &Layout,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::matmul_q4_0(
                    self, a.$cast()?, w.$cast()?, k, n, la
                ).map($wrap)
            }
            fn matmul_q4_km(
                &self, a: &AnyStorage, w: &AnyStorage,
                k: usize, n: usize, la: &Layout,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::matmul_q4_km(
                    self, a.$cast()?, w.$cast()?, k, n, la
                ).map($wrap)
            }
            fn matmul_q8_0(
                &self, a: &AnyStorage, w: &AnyStorage,
                k: usize, n: usize, la: &Layout,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::matmul_q8_0(
                    self, a.$cast()?, w.$cast()?, k, n, la
                ).map($wrap)
            }
            fn quantize_q8_0(
                &self, src_f32: &AnyStorage, n_elements: usize,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::quantize_q8_0(self, src_f32.$cast()?, n_elements).map($wrap)
            }
            fn dequantize_q8_0(
                &self, blocks: &AnyStorage, n_blocks: usize,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::dequantize_q8_0(self, blocks.$cast()?, n_blocks).map($wrap)
            }
            fn rms_norm_last_dim(
                &self, a: &AnyStorage, layout: &Layout, eps: f64,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::rms_norm_last_dim(self, a.$cast()?, layout, eps).map($wrap)
            }
            fn concat_along_dim(
                &self, a: &AnyStorage, b: &AnyStorage, dim: usize,
                a_layout: &Layout, b_layout: &Layout,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::concat_along_dim(
                    self, a.$cast()?, b.$cast()?, dim, a_layout, b_layout
                ).map($wrap)
            }
            fn rope(
                &self, x: &AnyStorage, cos: &AnyStorage, sin: &AnyStorage,
                x_layout: &Layout, cos_layout: &Layout, sin_layout: &Layout,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::rope(
                    self, x.$cast()?, cos.$cast()?, sin.$cast()?,
                    x_layout, cos_layout, sin_layout,
                ).map($wrap)
            }
            fn add_assign_scaled(
                &self, dst: &mut AnyStorage, src: &AnyStorage, scale: f32,
            ) -> Result<()> {
                let s = src.$cast()?;
                let d = match dst {
                    $wrap(inner) => inner,
                    _ => bail!("add_assign_scaled: dst device mismatch"),
                };
                <$backend as GraphBackend>::add_assign_scaled(self, d, s, scale)
            }
            fn rms_norm_last_dim_backward(
                &self, x: &AnyStorage, upstream: &AnyStorage,
                x_layout: &Layout, up_layout: &Layout, eps: f64,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::rms_norm_last_dim_backward(
                    self, x.$cast()?, upstream.$cast()?, x_layout, up_layout, eps,
                ).map($wrap)
            }
            fn layer_norm_last_dim_backward(
                &self, x: &AnyStorage, upstream: &AnyStorage,
                x_layout: &Layout, up_layout: &Layout, eps: f64,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::layer_norm_last_dim_backward(
                    self, x.$cast()?, upstream.$cast()?, x_layout, up_layout, eps,
                ).map($wrap)
            }
            fn softmax_last_dim_backward(
                &self, y: &AnyStorage, upstream: &AnyStorage,
                y_layout: &Layout, up_layout: &Layout,
            ) -> Result<AnyStorage> {
                <$backend as GraphBackend>::softmax_last_dim_backward(
                    self, y.$cast()?, upstream.$cast()?, y_layout, up_layout,
                ).map($wrap)
            }
            // copy_to uses the default impl — cross-device handled by
            // Router, same-device falls back to try_clone via default.
        }
    };
    // Legacy 6-arg form: kernel_source defaults to "". Use for
    // backends with exactly one impl per `(BackendId, DeviceLocation)`
    // — Vulkan, CUDA today.
    ($backend:ty, $cast:ident, $wrap:path, $device:expr, $caps:expr, $backend_id:expr) => {
        impl_dyn_backend!($backend, $cast, $wrap, $device, $caps, $backend_id, "");
    };
}

/// The 16 ops every concrete backend implements on `GraphBackend`
/// (no default impls in the trait) plus same-device `CopyTo`.
/// Per-backend lists below are CORE + backend-specific native ops.
///
/// As P2.5 wires more ops through DynBackend, each list grows
/// independently to reflect actual native support. A capability
/// drift test catches mismatches between the list and what
/// [`DynBackend`] can actually dispatch.
const CORE_CAPABILITIES: &[Capability] = &[
    Capability::Alloc,
    Capability::Upload,
    Capability::Download,
    Capability::TryClone,
    Capability::CopyStridedSrc,
    Capability::MatMul,
    Capability::Unary,
    Capability::Binary,
    Capability::Affine,
    Capability::Powf,
    Capability::Cast,
    Capability::Reduce,
    Capability::SoftmaxLastDim,
    Capability::IndexSelect,
    Capability::Gather,
    Capability::CopyTo,
];

const CPU_CAPABILITIES: &[Capability] = CORE_CAPABILITIES;

#[cfg(feature = "vulkan")]
const VULKAN_CAPABILITIES: &[Capability] = &[
    Capability::Alloc, Capability::Upload, Capability::Download,
    Capability::TryClone, Capability::CopyStridedSrc,
    Capability::MatMul, Capability::Unary, Capability::Binary,
    Capability::Affine,
    // Powf, Cast, and Gather bail in VulkanBackend today (CPU fallback
    // via executor). Not declared as native capabilities until
    // kernels land. Catching drift on these is the point of the
    // drift guard test — it pinned each absence.
    Capability::Reduce, Capability::SoftmaxLastDim,
    Capability::IndexSelect,
    Capability::CopyTo,
    // Native Q4_0 gemv kernel.
    Capability::MatMulQ4_0,
    // Q4_K_M: dequantize kernel native; matmul uses dequant-then-matmul.
    Capability::DequantizeQ4KM,
    Capability::MatMulQ4KM,
    // KV-cache Q8 quantize/dequantize (Slang kernels).
    Capability::QuantizeQ8_0,
    Capability::DequantizeQ8_0,
    // Fused forward kernels.
    Capability::RmsNormLastDim,
    Capability::ConcatAlongDim,
    Capability::Rope,
    Capability::AddAssignScaled,
    // Fused backward kernels.
    Capability::RmsNormLastDimBackward,
    Capability::LayerNormLastDimBackward,
    Capability::SoftmaxLastDimBackward,
];

#[cfg(feature = "cuda")]
const CUDA_CAPABILITIES: &[Capability] = &[
    Capability::Alloc,
    Capability::Upload,
    Capability::Download,
    Capability::TryClone,
    Capability::CopyStridedSrc,
    Capability::CopyTo,
    Capability::MatMul,
    Capability::Unary,
    Capability::Binary,
    Capability::Affine,
    Capability::Powf,
    Capability::Cast,
    Capability::Reduce,
    Capability::SoftmaxLastDim,
    Capability::IndexSelect,
    Capability::Gather,
    // New with the kernel-parity PR.
    Capability::Rope,
    Capability::RmsNormLastDim,
    Capability::MatMulQ4_0,
    Capability::MatMulQ4KM,
];

impl_dyn_backend!(CpuBackend, as_cpu, AnyStorage::Cpu, DeviceLocation::Cpu, CPU_CAPABILITIES, BackendId::Cpu, "portable-cpu");

#[cfg(feature = "vulkan")]
impl_dyn_backend!(
    VulkanBackend, as_vulkan, AnyStorage::Vulkan,
    DeviceLocation::Vulkan { gpu_id: 0 },
    VULKAN_CAPABILITIES,
    BackendId::Vulkan
);

#[cfg(feature = "cuda")]
impl_dyn_backend!(
    CudaBackend, as_cuda, AnyStorage::Cuda,
    DeviceLocation::Cuda { gpu_id: 0 },
    CUDA_CAPABILITIES,
    BackendId::Cuda
);

// AOCL is a CPU backend — its storage type is `AnyRefTensor`, the
// same as `CpuBackend`'s. AOCL and oneMKL both register their
// kernels under `BackendId::Cpu` (with `kernel_source: "aocl"` /
// `"mkl"` tags); the Router treats them as the CPU substrate at
// the device-placement level.
#[cfg(feature = "aocl")]
impl_dyn_backend!(
    AoclBackend, as_cpu, AnyStorage::Cpu,
    DeviceLocation::Cpu,
    CPU_CAPABILITIES,
    BackendId::Cpu,
    "aocl"
);

#[cfg(feature = "onemkl")]
impl_dyn_backend!(
    MklBackend, as_cpu, AnyStorage::Cpu,
    DeviceLocation::Cpu,
    CPU_CAPABILITIES,
    BackendId::Cpu,
    "mkl"
);

// -- Router ----------------------------------------------------------------

/// Multi-backend graph dispatcher.
pub struct Router {
    backends: Vec<Arc<dyn DynBackend>>,
    default_device: DeviceLocation,
    /// Precomputed at each `add_*`: for every `Capability` any
    /// attached backend advertises, the list of device identities
    /// that can execute it. Used by the Phase 4 scheduler to answer
    /// "which devices can host this node?" in O(1). Empty today —
    /// filled as part of Phase 4 wiring.
    capability_index: std::collections::HashMap<Capability, Vec<DeviceLocation>>,
    /// Phase 6b empirical dispatch table. When set, op-dispatch sites
    /// consult it to pick between competing backends sharing the same
    /// `DeviceLocation` — e.g. the portable CpuBackend vs an
    /// AoclBackend at `DeviceLocation::Cpu`. Populated externally by
    /// `fuel_core::judge::populate_dispatch_table()` and passed in
    /// via [`Router::with_dispatch_table`]. None means no empirical
    /// data available yet — Router falls through to first-registered.
    dispatch_table: Option<Arc<DispatchTable>>,
    /// Selection criterion when consulting `dispatch_table`. Defaults
    /// to `Criterion::Fastest`.
    dispatch_criterion: Criterion,
}

impl Router {
    /// Empty router. Use `add_cpu()` / `add_vulkan(..)` / etc. to
    /// attach backends.
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
            default_device: DeviceLocation::Cpu,
            capability_index: std::collections::HashMap::new(),
            dispatch_table: None,
            dispatch_criterion: Criterion::Fastest,
        }
    }

    /// Attach an empirical dispatch table. When set, op-dispatch
    /// sites consult it to pick between competing backends sharing
    /// the same `DeviceLocation` (e.g. CpuBackend vs AoclBackend at
    /// `DeviceLocation::Cpu`). Apps typically obtain the table from
    /// `fuel_core::judge::cached()` after calling
    /// `populate_dispatch_table()` either eagerly or in a background
    /// thread.
    pub fn with_dispatch_table(mut self, table: Arc<DispatchTable>) -> Self {
        self.dispatch_table = Some(table);
        self
    }

    /// Pick the criterion used when consulting the dispatch table.
    /// Default: `Criterion::Fastest`.
    pub fn with_dispatch_criterion(mut self, c: Criterion) -> Self {
        self.dispatch_criterion = c;
        self
    }

    /// Register a backend's capabilities into the router's lookup
    /// table. Called by each `add_*` constructor.
    fn register_capabilities(&mut self, device: DeviceLocation, caps: &[Capability]) {
        for &cap in caps {
            self.capability_index.entry(cap).or_default().push(device);
        }
    }

    /// Attach the CPU backend. Subsequent nullary ops (alloc_zeros,
    /// upload) route to this unless a different default device is
    /// explicitly set.
    pub fn add_cpu(mut self) -> Self {
        let b: Arc<dyn DynBackend> = Arc::new(CpuBackend);
        self.register_capabilities(DeviceLocation::Cpu, b.capabilities());
        self.backends.push(b);
        if self.backends.len() == 1 {
            self.default_device = DeviceLocation::Cpu;
        }
        self
    }

    #[cfg(feature = "vulkan")]
    pub fn add_vulkan(mut self, backend: VulkanBackend) -> Self {
        let device = DeviceLocation::Vulkan { gpu_id: 0 };
        let b: Arc<dyn DynBackend> = Arc::new(backend);
        self.register_capabilities(device, b.capabilities());
        self.backends.push(b);
        self.default_device = device;
        self
    }

    #[cfg(feature = "cuda")]
    pub fn add_cuda(mut self, backend: CudaBackend) -> Self {
        let device = DeviceLocation::Cuda { gpu_id: 0 };
        let b: Arc<dyn DynBackend> = Arc::new(backend);
        self.register_capabilities(device, b.capabilities());
        self.backends.push(b);
        self.default_device = device;
        self
    }

    /// Attach the AOCL CPU backend (AMD AOCL-BLAS / BLIS) if it
    /// loads on the current host. Coexists with the portable
    /// `CpuBackend` at `DeviceLocation::Cpu`; when a dispatch table
    /// is attached, op-dispatch picks between them per-op.
    ///
    /// Returns the Router unchanged on hosts where AOCL isn't
    /// loadable (no `libaocl_blas` on the dynamic loader path) so
    /// builders can chain `add_cpu().add_aocl()` unconditionally.
    #[cfg(feature = "aocl")]
    pub fn add_aocl(mut self) -> Self {
        match AoclBackend::try_new() {
            Ok(backend) => {
                let b: Arc<dyn DynBackend> = Arc::new(backend);
                // AOCL announces only the core CPU capabilities; the
                // capability_index already lists DeviceLocation::Cpu
                // from the prior add_cpu(), so we skip duplicate-add.
                self.backends.push(b);
                self
            }
            Err(e) => {
                eprintln!("Router::add_aocl: AOCL not loadable on this host, skipping: {e}");
                self
            }
        }
    }

    /// Attach the oneMKL CPU backend if `mkl_rt` loads on the current
    /// host. Same shape as `add_aocl`: silent no-op on missing runtime,
    /// chainable. The dispatch table picks empirically per op.
    #[cfg(feature = "onemkl")]
    pub fn add_mkl(mut self) -> Self {
        match MklBackend::try_new() {
            Ok(backend) => {
                let b: Arc<dyn DynBackend> = Arc::new(backend);
                self.backends.push(b);
                self
            }
            Err(e) => {
                eprintln!("Router::add_mkl: oneMKL not loadable on this host, skipping: {e}");
                self
            }
        }
    }

    /// List all devices capable of running a given op. Returns an
    /// empty slice if no attached backend advertises the capability.
    /// O(1) lookup.
    pub fn devices_for(&self, cap: Capability) -> &[DeviceLocation] {
        self.capability_index.get(&cap).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Quick check: is there any attached backend that supports `cap`?
    pub fn supports(&self, cap: Capability) -> bool {
        self.capability_index.contains_key(&cap)
    }

    /// The full capability index — exposed so the Phase-4 scheduler
    /// can iterate all (cap, devices) pairs when building its cost
    /// model.
    pub fn capability_index(&self) -> &std::collections::HashMap<Capability, Vec<DeviceLocation>> {
        &self.capability_index
    }

    /// Override the default device. Only useful when multiple backends
    /// are attached and you want alloc_zeros/upload to go somewhere
    /// other than the most-recently-attached.
    pub fn with_default_device(mut self, loc: DeviceLocation) -> Self {
        self.default_device = loc;
        self
    }

    /// The router's default device — where nullary ops (alloc_zeros,
    /// upload) land and where the scheduler falls back when no other
    /// placement constraint applies.
    pub fn default_device(&self) -> DeviceLocation {
        self.default_device
    }

    /// Best-effort autodetection: always attach CPU; attach a Vulkan
    /// backend if the feature is on AND a device can be enumerated;
    /// CUDA likewise. Returns a Router with the first successfully
    /// attached non-CPU backend as the default, or CPU if none.
    pub fn autodetect() -> Self {
        let mut r = Self::new().add_cpu();
        #[cfg(feature = "vulkan")]
        {
            if let Ok(vk) = VulkanBackend::with_selection(
                fuel_vulkan_backend::DeviceSelection::PreferDiscrete
            ) {
                r = r.add_vulkan(vk);
            }
        }
        // CUDA autodetection requires an enumeration entry point on
        // fuel-cuda-backend that doesn't exist yet; skip for now.
        r
    }

    /// Find the backend for a given device.
    fn backend_for(&self, loc: DeviceLocation) -> Result<&dyn DynBackend> {
        self.backends.iter()
            .find(|b| b.device() == loc)
            .map(|b| b.as_ref())
            .ok_or_else(|| fuel_core_types::Error::Msg(
                format!("Router: no backend attached for {loc:?}")
            ))
    }

    /// Find the attached backend matching a [`Pick`] — `(BackendId,
    /// DeviceLocation, kernel_source)`. Multiple backends can share
    /// the same `(BackendId, DeviceLocation)` slot (e.g. portable
    /// `CpuBackend`, `AoclBackend`, `MklBackend` all at
    /// `(BackendId::Cpu, DeviceLocation::Cpu)`); the `kernel_source`
    /// tag picked by the dispatch table is what disambiguates which
    /// kernel actually won the contest.
    ///
    /// Resolution order:
    ///
    /// 1. Exact match on all three of `(BackendId, DeviceLocation,
    ///    kernel_source)`. This is the path the Phase 6b dispatch
    ///    table is designed to drive end-to-end.
    /// 2. First backend matching `(BackendId, DeviceLocation)`
    ///    ignoring `kernel_source`, **with a `eprintln!` warning**.
    ///    Reached when a Pick names a `kernel_source` no attached
    ///    backend declares — typically because the profile was
    ///    produced on a host that had AOCL/MKL loaded but this
    ///    Router doesn't (compiled out, or the app didn't add it).
    /// 3. `None` if no `(BackendId, DeviceLocation)` match at all.
    fn backend_for_pick(&self, pick: &Pick, loc: DeviceLocation) -> Option<&dyn DynBackend> {
        // 1. Exact (backend_id, device, kernel_source) match.
        if let Some(b) = self.backends.iter().find(|b| {
            b.backend_id() == pick.backend
                && b.device() == loc
                && b.kernel_source() == pick.kernel_source
        }) {
            return Some(b.as_ref());
        }
        // 2. (backend_id, device) match with a different kernel_source.
        //    The Pick named a sibling kernel this Router doesn't carry;
        //    fall back to whatever sibling IS attached and warn so the
        //    drift surfaces in logs.
        if let Some(b) = self.backends.iter().find(|b| {
            b.backend_id() == pick.backend && b.device() == loc
        }) {
            eprintln!(
                "Router::backend_for_pick: pick named kernel_source={:?} for \
                 ({:?}, {:?}) but no attached backend carries that tag; \
                 falling back to first-registered ({:?}). Profile table may \
                 reflect a host with siblings this Router doesn't load.",
                pick.kernel_source, pick.backend, loc, b.kernel_source(),
            );
            return Some(b.as_ref());
        }
        // 3. No `(backend_id, device)` match at all.
        None
    }

    /// Pick a backend for a single op based on the empirical
    /// dispatch table (when present). Falls through to
    /// `backend_for(target)` if the table is absent, the op isn't
    /// profiled, or the picked `(BackendId, kernel_source)` isn't
    /// attached.
    ///
    /// `n_elements` is the bucketing input — typically the output
    /// element count for matmul / unary / binary ops. The dispatch
    /// table uses log2-bucketed size classes, so off-by-a-few is
    /// fine; nearest-class lookup handles it.
    fn pick_for_op(
        &self,
        op: OpKind,
        dtype: DType,
        n_elements: usize,
        target: DeviceLocation,
    ) -> Result<&dyn DynBackend> {
        if let Some(table) = &self.dispatch_table {
            let class = SizeClass::from_elem_count(n_elements);
            if let Some(pick) = table.pick_nearest(op, dtype, class, self.dispatch_criterion) {
                if let Some(b) = self.backend_for_pick(&pick, target) {
                    return Ok(b);
                }
                // Pick named a backend that isn't attached to this
                // Router at all (compiled out, or the app didn't add
                // it). Silent fall-through — the table is advisory,
                // not authoritative.
            }
        }
        self.backend_for(target)
    }

    /// Copy a storage to `target` via host round-trip. Source stays
    /// resident on its original device.
    pub fn copy_to(
        &self,
        storage: &AnyStorage,
        layout: &Layout,
        target: DeviceLocation,
    ) -> Result<AnyStorage> {
        if storage.device() == target {
            return self.backend_for(target)?.try_clone(storage, layout);
        }
        let src_backend = self.backend_for(storage.device())?;
        let host = src_backend.download(storage)?;
        self.upload_to(&host, layout.shape(), target)
    }

    /// Upload to a specific target device (vs the router's default).
    pub fn upload_to(
        &self,
        buf: &HostBuffer,
        shape: &Shape,
        target: DeviceLocation,
    ) -> Result<AnyStorage> {
        self.backend_for(target)?.upload(buf, shape)
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new().add_cpu()
    }
}

// -- GraphBackend impl for Router -----------------------------------------
//
// Every method: pick the right backend (by default device for nullary
// ops, by input-storage device for ops with inputs) and hand off to
// that DynBackend.

impl GraphBackend for Router {
    type Storage = AnyStorage;

    fn alloc_zeros(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.backend_for(self.default_device)?.alloc_zeros(shape, dtype)
    }

    fn upload(&self, buf: &HostBuffer, shape: &Shape) -> Result<Self::Storage> {
        self.upload_to(buf, shape, self.default_device)
    }

    fn download(&self, storage: &Self::Storage) -> Result<HostBuffer> {
        self.backend_for(storage.device())?.download(storage)
    }

    fn try_clone(&self, storage: &Self::Storage, layout: &Layout) -> Result<Self::Storage> {
        self.backend_for(storage.device())?.try_clone(storage, layout)
    }

    fn copy_strided_src(
        &self,
        src: &Self::Storage,
        dst: &mut Self::Storage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> Result<()> {
        let dev = src.device();
        if dst.device() != dev {
            bail!("Router::copy_strided_src: src/dst device mismatch");
        }
        self.backend_for(dev)?.copy_strided_src(src, dst, dst_offset, src_layout)
    }

    fn storage_dtype(&self, storage: &Self::Storage) -> DType { storage.dtype() }

    fn copy_to(
        &self, storage: &Self::Storage, layout: &Layout, target: DeviceLocation,
    ) -> Result<Self::Storage> {
        Router::copy_to(self, storage, layout, target)
    }

    fn matmul(
        &self, a: &Self::Storage, b: &Self::Storage,
        bmnk: (usize, usize, usize, usize),
        la: &Layout, lb: &Layout,
    ) -> Result<Self::Storage> {
        let dev = same_device(a, b, "matmul")?;
        // Output element count = batch * m * n. Used to bucket the
        // op into a SizeClass for the empirical dispatch lookup.
        let (batch, m, n, _k) = bmnk;
        let out_elems = batch.max(1) * m * n;
        self.pick_for_op(OpKind::MatMul, a.dtype(), out_elems, dev)?
            .matmul(a, b, bmnk, la, lb)
    }

    fn unary(
        &self, op: UnaryOp,
        a: &Self::Storage, layout: &Layout,
    ) -> Result<Self::Storage> {
        self.backend_for(a.device())?.unary(op, a, layout)
    }

    fn binary(
        &self, op: BinaryOp,
        a: &Self::Storage, b: &Self::Storage,
        la: &Layout, lb: &Layout,
    ) -> Result<Self::Storage> {
        let dev = same_device(a, b, "binary")?;
        self.backend_for(dev)?.binary(op, a, b, la, lb)
    }

    fn affine(
        &self, a: &Self::Storage, layout: &Layout, mul: f64, add: f64,
    ) -> Result<Self::Storage> {
        self.backend_for(a.device())?.affine(a, layout, mul, add)
    }

    fn powf(
        &self, a: &Self::Storage, layout: &Layout, exp: f64,
    ) -> Result<Self::Storage> {
        self.backend_for(a.device())?.powf(a, layout, exp)
    }

    fn cast(
        &self, a: &Self::Storage, layout: &Layout, dtype: DType,
    ) -> Result<Self::Storage> {
        self.backend_for(a.device())?.cast(a, layout, dtype)
    }

    fn reduce(
        &self, op: fuel_core_types::op::ReduceOp,
        a: &Self::Storage, layout: &Layout, dims: &[usize],
    ) -> Result<Self::Storage> {
        self.backend_for(a.device())?.reduce(op, a, layout, dims)
    }

    fn softmax_last_dim(
        &self, a: &Self::Storage, layout: &Layout,
    ) -> Result<Self::Storage> {
        self.backend_for(a.device())?.softmax_last_dim(a, layout)
    }

    fn index_select(
        &self, src: &Self::Storage, ids: &Self::Storage,
        src_l: &Layout, ids_l: &Layout, dim: usize,
    ) -> Result<Self::Storage> {
        let dev = same_device(src, ids, "index_select")?;
        self.backend_for(dev)?.index_select(src, ids, src_l, ids_l, dim)
    }

    fn gather(
        &self, src: &Self::Storage, ids: &Self::Storage,
        src_l: &Layout, ids_l: &Layout, dim: usize,
    ) -> Result<Self::Storage> {
        let dev = same_device(src, ids, "gather")?;
        self.backend_for(dev)?.gather(src, ids, src_l, ids_l, dim)
    }

    fn matmul_q4_0(
        &self, a: &Self::Storage, w: &Self::Storage,
        k: usize, n: usize, la: &Layout,
    ) -> Result<Self::Storage> {
        let dev = same_device(a, w, "matmul_q4_0")?;
        self.backend_for(dev)?.matmul_q4_0(a, w, k, n, la)
    }

    fn matmul_q4_km(
        &self, a: &Self::Storage, w: &Self::Storage,
        k: usize, n: usize, la: &Layout,
    ) -> Result<Self::Storage> {
        let dev = same_device(a, w, "matmul_q4_km")?;
        self.backend_for(dev)?.matmul_q4_km(a, w, k, n, la)
    }

    fn matmul_q8_0(
        &self, a: &Self::Storage, w: &Self::Storage,
        k: usize, n: usize, la: &Layout,
    ) -> Result<Self::Storage> {
        let dev = same_device(a, w, "matmul_q8_0")?;
        self.backend_for(dev)?.matmul_q8_0(a, w, k, n, la)
    }

    fn quantize_q8_0(
        &self, src_f32: &Self::Storage, n_elements: usize,
    ) -> Result<Self::Storage> {
        self.backend_for(src_f32.device())?.quantize_q8_0(src_f32, n_elements)
    }

    fn dequantize_q8_0(
        &self, blocks: &Self::Storage, n_blocks: usize,
    ) -> Result<Self::Storage> {
        self.backend_for(blocks.device())?.dequantize_q8_0(blocks, n_blocks)
    }

    fn rms_norm_last_dim(
        &self, a: &Self::Storage, layout: &Layout, eps: f64,
    ) -> Result<Self::Storage> {
        self.backend_for(a.device())?.rms_norm_last_dim(a, layout, eps)
    }

    fn concat_along_dim(
        &self, a: &Self::Storage, b: &Self::Storage, dim: usize,
        a_layout: &Layout, b_layout: &Layout,
    ) -> Result<Self::Storage> {
        let dev = same_device(a, b, "concat_along_dim")?;
        self.backend_for(dev)?.concat_along_dim(a, b, dim, a_layout, b_layout)
    }

    fn rope(
        &self, x: &Self::Storage, cos: &Self::Storage, sin: &Self::Storage,
        x_layout: &Layout, cos_layout: &Layout, sin_layout: &Layout,
    ) -> Result<Self::Storage> {
        // All three operands must live on the same device.
        let dev = same_device(x, cos, "rope")?;
        let _ = same_device(x, sin, "rope")?;
        self.backend_for(dev)?.rope(x, cos, sin, x_layout, cos_layout, sin_layout)
    }

    fn add_assign_scaled(
        &self, dst: &mut Self::Storage, src: &Self::Storage, scale: f32,
    ) -> Result<()> {
        if dst.device() != src.device() {
            bail!("Router::add_assign_scaled: dst/src device mismatch");
        }
        self.backend_for(dst.device())?.add_assign_scaled(dst, src, scale)
    }

    fn rms_norm_last_dim_backward(
        &self, x: &Self::Storage, upstream: &Self::Storage,
        x_layout: &Layout, up_layout: &Layout, eps: f64,
    ) -> Result<Self::Storage> {
        let dev = same_device(x, upstream, "rms_norm_last_dim_backward")?;
        self.backend_for(dev)?.rms_norm_last_dim_backward(x, upstream, x_layout, up_layout, eps)
    }

    fn layer_norm_last_dim_backward(
        &self, x: &Self::Storage, upstream: &Self::Storage,
        x_layout: &Layout, up_layout: &Layout, eps: f64,
    ) -> Result<Self::Storage> {
        let dev = same_device(x, upstream, "layer_norm_last_dim_backward")?;
        self.backend_for(dev)?.layer_norm_last_dim_backward(x, upstream, x_layout, up_layout, eps)
    }

    fn softmax_last_dim_backward(
        &self, y: &Self::Storage, upstream: &Self::Storage,
        y_layout: &Layout, up_layout: &Layout,
    ) -> Result<Self::Storage> {
        let dev = same_device(y, upstream, "softmax_last_dim_backward")?;
        self.backend_for(dev)?.softmax_last_dim_backward(y, upstream, y_layout, up_layout)
    }

    // Remaining trait methods (qmatmul, quantize_q8_0, dequantize_q8_0,
    // rms_norm_last_dim, concat_along_dim, *_backward, rope,
    // add_assign_scaled) keep their default `bail` impls, and the
    // executor falls back to CPU reference via `cpu_fallback` for them.
    // Phase 2.5 fills these in as needed.
}

#[cfg(test)]
mod tests {
    //! `backend_for_pick` / `pick_for_op` regression tests for the
    //! kernel_source-aware routing landed 2026-06-08. Validates the
    //! Router resolves a [`Pick`] to the specific `DynBackend` whose
    //! `kernel_source()` matches the picked tag — not just the first
    //! `(BackendId, DeviceLocation)` match.
    //!
    //! Uses an in-test fake `DynBackend` so the tests don't depend on
    //! the `aocl` / `onemkl` features being enabled at build time. The
    //! fake declares no capabilities and bails on every op method — we
    //! only exercise `device()` / `backend_id()` / `kernel_source()`,
    //! which is all `backend_for_pick` reads.
    use super::*;
    use fuel_core_types::dispatch::{
        DispatchTable, OpKind, ProfileEntry, ProfileReport, SizeClass,
        PROFILE_REPORT_VERSION,
    };

    /// Test DynBackend that records nothing — used only to verify
    /// the Router picks the right one by `kernel_source`.
    struct FakeCpuBackend {
        ks: &'static str,
    }

    impl DynBackend for FakeCpuBackend {
        fn device(&self) -> DeviceLocation { DeviceLocation::Cpu }
        fn backend_id(&self) -> BackendId { BackendId::Cpu }
        fn kernel_source(&self) -> &'static str { self.ks }
        // All op methods default to `bail` — we never call them in
        // these tests.
    }

    /// Build a dispatch table with one entry naming the given
    /// kernel_source as the winner for `MatMul/F32` at SizeClass(12).
    fn dispatch_table_for(kernel_source: &str) -> DispatchTable {
        let report = ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![ProfileEntry {
                op: OpKind::MatMul,
                dtype: DType::F32,
                size_class: SizeClass(12),
                backend: BackendId::Cpu,
                device_index: 0,
                latency_ns: 100,
                iterations: 1,
                max_rel_error: 0.0,
                kernel_source: kernel_source.into(),
            }],
        };
        DispatchTable::build(&report)
    }

    fn router_with_portable_and_aocl_fakes() -> Router {
        let mut r = Router::new();
        // Register portable-cpu first so it's the "first-registered"
        // fallback. AOCL second — the pick named "aocl" must still
        // route to it.
        let portable: Arc<dyn DynBackend> = Arc::new(FakeCpuBackend { ks: "portable-cpu" });
        let aocl: Arc<dyn DynBackend> = Arc::new(FakeCpuBackend { ks: "aocl" });
        r.backends.push(portable);
        r.backends.push(aocl);
        r.default_device = DeviceLocation::Cpu;
        r
    }

    #[test]
    fn backend_for_pick_returns_aocl_when_pick_names_aocl() {
        let r = router_with_portable_and_aocl_fakes();
        let pick = Pick {
            backend: BackendId::Cpu,
            device_index: 0,
            kernel_source: "aocl",
        };
        let b = r.backend_for_pick(&pick, DeviceLocation::Cpu)
            .expect("aocl sibling attached");
        assert_eq!(b.kernel_source(), "aocl",
            "kernel_source-matched backend must win even when not first-registered");
    }

    #[test]
    fn backend_for_pick_returns_portable_when_pick_names_portable() {
        let r = router_with_portable_and_aocl_fakes();
        let pick = Pick {
            backend: BackendId::Cpu,
            device_index: 0,
            kernel_source: "portable-cpu",
        };
        let b = r.backend_for_pick(&pick, DeviceLocation::Cpu)
            .expect("portable-cpu sibling attached");
        assert_eq!(b.kernel_source(), "portable-cpu");
    }

    #[test]
    fn backend_for_pick_falls_back_to_first_when_kernel_source_absent() {
        // Pick names "mkl" — neither attached backend carries that
        // tag. Router falls back to the first matching
        // `(BackendId, DeviceLocation)` — the portable-cpu we added
        // first — and emits an eprintln warning (not asserted; just
        // observed in test output).
        let r = router_with_portable_and_aocl_fakes();
        let pick = Pick {
            backend: BackendId::Cpu,
            device_index: 0,
            kernel_source: "mkl",
        };
        let b = r.backend_for_pick(&pick, DeviceLocation::Cpu)
            .expect("fallback to first-registered for unknown kernel_source");
        assert_eq!(b.kernel_source(), "portable-cpu",
            "first-registered fallback when kernel_source has no match");
    }

    #[test]
    fn backend_for_pick_returns_none_when_no_backend_id_match() {
        // Pick names BackendId::Vulkan but only CPU siblings are
        // attached. No backend_id match → None (caller falls back to
        // `backend_for(target)` per `pick_for_op` semantics).
        let r = router_with_portable_and_aocl_fakes();
        let pick = Pick {
            backend: BackendId::Vulkan,
            device_index: 0,
            kernel_source: "slang",
        };
        assert!(r.backend_for_pick(&pick, DeviceLocation::Cpu).is_none());
    }

    #[test]
    fn pick_for_op_routes_to_kernel_source_winner() {
        // End-to-end check: with a dispatch table that names "aocl"
        // the winner at MatMul/F32/SizeClass(12), pick_for_op should
        // route through `backend_for_pick` to the aocl-tagged fake.
        let mut r = router_with_portable_and_aocl_fakes();
        let table = Arc::new(dispatch_table_for("aocl"));
        r.dispatch_table = Some(table);
        // SizeClass(12) = log2-bucket for ~4K elements.
        let n_elems = 1 << 12;
        let b = r.pick_for_op(OpKind::MatMul, DType::F32, n_elems, DeviceLocation::Cpu)
            .expect("pick_for_op resolves");
        assert_eq!(b.kernel_source(), "aocl");
    }

    #[test]
    fn cpu_backend_declares_portable_cpu_kernel_source() {
        // Sanity: the real CpuBackend reports "portable-cpu" so it
        // matches a Pick whose `kernel_source` came back as
        // `"portable-cpu"` (the interned-tag convention).
        assert_eq!(CpuBackend.kernel_source(), "portable-cpu");
    }
}

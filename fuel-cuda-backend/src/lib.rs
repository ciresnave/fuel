//! CUDA GPU executor for `fuel-graph` computation graphs + low-level
//! CUDA primitives (storage, device, kernel dispatch, cuBLAS / cuDNN
//! wrappers). The parallel Vulkan stack (`vulkane` → `fuel-vulkan-backend`)
//! uses the same shape: the external FFI crate family (`baracuda-*` here,
//! `vulkane` there) provides raw bindings; this crate layers the
//! ML-specific dtype-tagged storage, kernel dispatch, and graph
//! integration on top.
//!
//! ## Modules
//!
//! - [`device`] — `CudaDevice` wrapping `baracuda-driver`'s Context + Stream
//!   + module cache + cuBLAS handle + curand generator.
//! - [`storage`] — `CudaStorage` (dtype-tagged tensor) + all tensor-op
//!   dispatch (matmul, conv, softmax, rope, rms_norm, quantized matmul,
//!   gather/scatter, pooling, upsample).
//! - [`utils`] — `Map1` / `Map2` / `Map3` / `Map*Any` dtype-dispatch traits.
//! - [`dyn_impl`] — object-safe `BackendDevice` / `BackendStorage` impls.
//! - [`error`] — `CudaError` + `WrapErr` trait for baracuda error conversion.
//! - [`cudnn`] — optional convolution wrapper (feature: `cudnn`).
//! - [`backend`] — `CudaBackend` implementing `GraphBackend` (this crate's
//!   primary public surface — the bridge from the fuel-graph executor
//!   into the CUDA primitives above).
//!
//! ## Execution model
//!
//! All intermediates stay in GPU memory; host↔device transfer happens
//! only at `Const` upload (H2D) and `realize_*` readback (D2H).
//!
//! Model weights upload **once** (first forward pass) and persist in
//! the executor's `const_pool` for the executor's lifetime. KV-cache
//! consts and computed intermediates are owned per-realize and freed
//! at the end of each call.

// --- fuel-cuda primitives (formerly a separate crate) -----------------------
pub use fuel_core_types::{DType, Error, Layout, Result, Shape};

// `crate::cudnn` retired in Phase 5b of the fuel-cuda-kernels retirement
// (2026-05-25). Fuel's internal cuDNN wrapper for conv2d/conv1d (252 LOC)
// is no longer needed — conv dispatch goes through
// `baracuda-kernels-sys::baracuda_kernels_conv_*_run` instead. The
// `cudnn` Cargo feature is now a near-no-op (only used to gate the
// transitive `baracuda-cudnn{,-sys}` deps that don't have other users
// in Fuel).
#[cfg(feature = "flash-attn")]
pub mod flash_attn;
#[cfg(feature = "flash-attn-v3")]
pub mod flash_attn_v3;
pub mod baracuda;
/// Re-export of `baracuda_kernels_sys` so downstream crates (like
/// `fuel-core`) can call baracuda FFI symbols without pulling
/// `baracuda-kernels-sys` in as a direct dep.
pub use baracuda_kernels_sys;
pub mod byte_kernels;
pub mod byte_storage;
pub mod cutlass;
pub mod device;
pub mod dyn_impl;
pub mod error;
pub mod pinned;
pub mod probe;
pub mod quantized;
pub mod storage;
#[cfg(feature = "ug")]
pub mod ug;
pub mod utils;

pub use byte_storage::CudaStorageBytes;
pub use device::{CublasHandle, CudaDevice, CudaFunc, DeviceId, LaunchArgs, LaunchConfig};
pub use dyn_impl::{CudaBackendDevice, CudaBackendStorage};
pub use error::{CudaError, WrapErr};
pub use pinned::PinnedHostStorage;
pub use storage::{CudaStorage, CudaStorageSlice, SlicePtrOrNull, kernel_name};
pub use utils::{Map1, Map1Any, Map2, Map2Any, Map2InPlace, Map3, S};

// Re-export the kernel PTX bundle so downstream code (e.g. fuel-nn's
// launch-site helpers) can reach it via this crate.

// --- graph executor integration --------------------------------------------
mod backend;
pub use backend::CudaBackend;

use fuel_graph::{topo_order, topo_order_multi, NodeId, Op, Tensor};
use fuel_core_types::DimVec;
use fuel_reference_backend::exec::AnyRefTensor as AnyRef;
use fuel_reference_backend::RefTensor;
use std::collections::HashMap;
use tracing::{debug_span, info_span};

/// GPU tensor: storage + shape (CudaStorage doesn't track shape).
struct GpuTensor {
    storage: CudaStorage,
    shape: Shape,
}

impl GpuTensor {
    fn layout(&self) -> Layout {
        Layout::contiguous(&self.shape)
    }
}

/// A node-cache entry: either a reference to a persistent const_pool
/// entry (zero-cost on cache hit) or an owned computed tensor.
enum CacheEntry {
    /// Points into `CudaGraphExecutor::const_pool`. The pool outlives
    /// the per-realize cache, so the GPU memory stays valid.
    ConstRef(usize),
    /// An intermediate computed during this realize pass. Freed when
    /// the cache is dropped at the end of realize_*.
    Owned(GpuTensor),
}

/// CUDA graph executor with a persistent weight cache.
pub struct CudaGraphExecutor {
    pub device: CudaDevice,
    /// Weights uploaded on first encounter, keyed on host-side
    /// `Arc<[T]>` data pointer. Never cleared — lives for the
    /// executor's lifetime.
    const_pool: HashMap<usize, GpuTensor>,
    /// Pre-populated entries for the NEXT realize call. Drained at
    /// the start of each realize. Used by the KV cache path to
    /// inject GPU-resident cached K/V without a host round-trip.
    injected: HashMap<NodeId, GpuTensor>,
}

impl CudaGraphExecutor {
    pub fn new(device: CudaDevice) -> Self {
        Self {
            device,
            const_pool: HashMap::new(),
            injected: HashMap::new(),
        }
    }

    pub fn for_device(ordinal: usize) -> fuel_core_types::Result<Self> {
        Ok(Self::new(CudaDevice::new(ordinal)?))
    }

    pub fn realize_f32(&mut self, tensor: &Tensor) -> RefTensor<f32> {
        let _span = info_span!("realize_f32").entered();
        let graph = tensor.graph().read().unwrap();
        let order = topo_order(&graph, tensor.id());
        let num_nodes = order.len();
        let _walk = info_span!("topo_walk", nodes = num_nodes).entered();
        let mut cache: HashMap<NodeId, CacheEntry> = HashMap::new();
        for (id, gt) in self.injected.drain() {
            cache.insert(id, CacheEntry::Owned(gt));
        }
        for id in order {
            if cache.contains_key(&id) { continue; }
            let node = graph.node(id);
            // Phase 7.5 G2: slot-first dispatch.
            if let Some(entry) = self.try_adopt_slot_cuda(&graph, id, &node.shape) {
                cache.insert(id, entry);
                continue;
            }
            let entry = self.eval_node(
                &node.op, &node.inputs, &node.shape, node.dtype, &cache,
            );
            cache.insert(id, entry);
        }
        drop(_walk);
        let _readback = info_span!("d2h_readback").entered();
        let gt = self.take_owned(cache.remove(&tensor.id())
            .expect("realize: missing root"));
        gpu_to_ref_f32(gt)
    }

    pub fn realize_many_f32(&mut self, tensors: &[&Tensor]) -> Vec<RefTensor<f32>> {
        let _span = info_span!("realize_many_f32", roots = tensors.len()).entered();
        if tensors.is_empty() {
            return Vec::new();
        }
        let graph_rc = tensors[0].graph();
        let graph = graph_rc.read().unwrap();
        let roots: Vec<NodeId> = tensors.iter().map(|t| t.id()).collect();
        let order = topo_order_multi(&graph, &roots);
        let num_nodes = order.len();
        let _walk = info_span!("topo_walk", nodes = num_nodes).entered();
        let mut cache: HashMap<NodeId, CacheEntry> = HashMap::new();
        for (id, gt) in self.injected.drain() {
            cache.insert(id, CacheEntry::Owned(gt));
        }
        for id in order {
            if cache.contains_key(&id) { continue; }
            let node = graph.node(id);
            // Phase 7.5 G2: slot-first dispatch.
            if let Some(entry) = self.try_adopt_slot_cuda(&graph, id, &node.shape) {
                cache.insert(id, entry);
                continue;
            }
            let entry = self.eval_node(
                &node.op, &node.inputs, &node.shape, node.dtype, &cache,
            );
            cache.insert(id, entry);
        }
        drop(_walk);
        let _readback = info_span!("d2h_readback", roots = roots.len()).entered();
        roots
            .iter()
            .map(|id| {
                let gt = self.resolve(cache.get(id).expect("realize_many: missing root"));
                gpu_to_ref_f32_ref(gt)
            })
            .collect()
    }

    /// Pre-populate a graph node with an existing GPU tensor. When the
    /// topo walk encounters this node, it uses the pre-populated storage
    /// instead of uploading from host. Used by the KV cache path to
    /// feed cached K/V that already lives on GPU.
    ///
    /// The `node_id` comes from `lazy_tensor.graph_tensor().id()` after
    /// building the graph but before calling realize.
    pub fn pre_populate(&mut self, node_id: NodeId, storage: CudaStorage, shape: Shape) {
        self.injected.insert(node_id, GpuTensor { storage, shape });
    }

    /// Realize a mixed set of roots: the first `n_d2h` are downloaded
    /// to CPU as `Vec<f32>`; the rest stay on GPU as `(CudaStorage, Shape)`.
    ///
    /// Used by the KV cache path: logits are D2H'd for sampling, but
    /// fresh K/V stay on GPU for the next decode step.
    pub fn realize_split(
        &mut self,
        tensors: &[&Tensor],
        n_d2h: usize,
    ) -> (Vec<Vec<f32>>, Vec<(CudaStorage, Shape)>) {
        let _span = info_span!("realize_split", roots = tensors.len(), n_d2h).entered();
        if tensors.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let graph_rc = tensors[0].graph();
        let graph = graph_rc.read().unwrap();
        let roots: Vec<NodeId> = tensors.iter().map(|t| t.id()).collect();
        let order = topo_order_multi(&graph, &roots);
        let num_nodes = order.len();
        let _walk = info_span!("topo_walk", nodes = num_nodes).entered();
        let mut cache: HashMap<NodeId, CacheEntry> = HashMap::new();

        // Move injected entries into the cache before the walk.
        for (id, gt) in self.injected.drain() {
            cache.insert(id, CacheEntry::Owned(gt));
        }

        for id in order {
            if cache.contains_key(&id) {
                continue; // pre-populated or injected
            }
            let node = graph.node(id);
            // Phase 7.5 G2: slot-first dispatch.
            if let Some(entry) = self.try_adopt_slot_cuda(&graph, id, &node.shape) {
                cache.insert(id, entry);
                continue;
            }
            let entry = self.eval_node(
                &node.op, &node.inputs, &node.shape, node.dtype, &cache,
            );
            cache.insert(id, entry);
        }
        drop(_walk);

        // Split: first n_d2h roots go to CPU, rest stay on GPU.
        let _readback = info_span!("d2h_readback", n_d2h).entered();
        let mut cpu_results = Vec::with_capacity(n_d2h);
        let mut gpu_results = Vec::with_capacity(roots.len() - n_d2h);

        for (i, id) in roots.iter().enumerate() {
            if i < n_d2h {
                let gt = self.resolve(cache.get(id).expect("split: missing root"));
                cpu_results.push(gpu_to_ref_f32_ref(gt).into_vec());
            } else {
                // Extract as owned GPU tensor.
                match cache.remove(id) {
                    Some(CacheEntry::Owned(gt)) => {
                        gpu_results.push((gt.storage, gt.shape));
                    }
                    Some(CacheEntry::ConstRef(key)) => {
                        // Rare: root is a const. Clone to avoid holding const_pool ref.
                        let pooled = self.const_pool.get(&key).expect("dangling");
                        let s = pooled.storage.try_clone(&pooled.layout()).expect("split clone");
                        gpu_results.push((s, pooled.shape.clone()));
                    }
                    None => panic!("split: missing root"),
                }
            }
        }
        (cpu_results, gpu_results)
    }

    // --- cache resolution ---

    /// Resolve a CacheEntry to a &GpuTensor, following ConstRef indirection.
    fn resolve<'a>(&'a self, entry: &'a CacheEntry) -> &'a GpuTensor {
        match entry {
            CacheEntry::ConstRef(key) => self.const_pool.get(key)
                .expect("dangling ConstRef"),
            CacheEntry::Owned(gt) => gt,
        }
    }

    /// Get the GpuTensor for a node from the cache.
    fn get_gt<'a>(
        &'a self,
        inputs: &[NodeId],
        idx: usize,
        cache: &'a HashMap<NodeId, CacheEntry>,
    ) -> &'a GpuTensor {
        let entry = cache.get(&inputs[idx]).expect("topo order missing input");
        self.resolve(entry)
    }

    /// Extract an owned GpuTensor from a CacheEntry. For ConstRef,
    /// does a D2H + H2D round-trip (only used for the final realize
    /// readback when the root happens to be a const — rare in practice).
    fn take_owned(&self, entry: CacheEntry) -> GpuTensor {
        match entry {
            CacheEntry::Owned(gt) => gt,
            CacheEntry::ConstRef(key) => {
                let pooled = self.const_pool.get(&key).expect("dangling ConstRef");
                let cpu = pooled.storage.to_cpu_storage().expect("take_owned D2H");
                let gpu = self.device.storage_from_cpu_storage(&cpu)
                    .expect("take_owned H2D");
                GpuTensor { storage: gpu, shape: pooled.shape.clone() }
            }
        }
    }

    // --- eval_node dispatcher ---

    fn eval_node(
        &mut self,
        op: &Op,
        inputs: &[NodeId],
        shape: &Shape,
        dtype: DType,
        cache: &HashMap<NodeId, CacheEntry>,
    ) -> CacheEntry {
        let op_name = op_short_name(op);
        let _span = debug_span!("eval_node", op = op_name, elems = shape.elem_count()).entered();

        let result_storage = match op {
            // Op::Const is intercepted by slot-first dispatch in the
            // realize loops (try_adopt_slot_cuda). Reaching eval_node
            // with a Const node means a constructor failed to slot-
            // populate — a bug.
            Op::Const => unreachable!(
                "fuel-cuda-backend eval_node: Op::Const must be handled \
                 by slot-first dispatch in the realize loop, never reach \
                 eval_node",
            ),

            Op::MatMul => {
                let (a, b) = (self.get_gt(inputs, 0, cache), self.get_gt(inputs, 1, cache));
                let ad = a.shape.dims();
                let bd = b.shape.dims();
                let rank = ad.len();
                let (m, k, n) = (ad[rank - 2], ad[rank - 1], bd[rank - 1]);
                let batch: usize = ad[..rank - 2].iter().product::<usize>().max(1);
                a.storage.matmul(&b.storage, (batch, m, n, k), &a.layout(), &b.layout())
                    .expect("MatMul")
            }

            // Unary ops via native CUDA kernels.
            Op::Neg => self.unary_cuda("uneg", inputs, cache),
            Op::Sqr => self.unary_cuda("usqr", inputs, cache),
            Op::Sqrt => self.unary_cuda("usqrt", inputs, cache),
            Op::Exp => self.unary_cuda("uexp", inputs, cache),
            Op::Log => self.unary_cuda("ulog", inputs, cache),
            Op::Sin => self.unary_cuda("usin", inputs, cache),
            Op::Cos => self.unary_cuda("ucos", inputs, cache),
            Op::Tanh => self.unary_cuda("utanh", inputs, cache),
            Op::Sigmoid => self.unary_cuda("usigmoid", inputs, cache),
            Op::Silu => self.unary_cuda("usilu", inputs, cache),
            Op::Gelu => self.unary_cuda("ugelu", inputs, cache),
            Op::Relu => self.unary_cuda("urelu", inputs, cache),
            Op::Step => self.unary_cuda("ustep", inputs, cache),
            Op::Recip => self.unary_cuda("urecip", inputs, cache),
            Op::Abs => self.unary_cuda("uabs", inputs, cache),

            // Binary ops via native CUDA kernels.
            Op::Add => self.binary_cuda("badd", inputs, cache),
            Op::Sub => self.binary_cuda("bsub", inputs, cache),
            Op::Mul => self.binary_cuda("bmul", inputs, cache),
            Op::Div => self.binary_cuda("bdiv", inputs, cache),
            Op::Maximum => self.binary_cuda("bmaximum", inputs, cache),
            Op::Minimum => self.binary_cuda("bminimum", inputs, cache),

            // Scalar affine ops.
            Op::AddScalar(c) => {
                let a = self.get_gt(inputs, 0, cache);
                a.storage.affine(&a.layout(), 1.0, *c).expect("AddScalar")
            }
            Op::MulScalar(c) => {
                let a = self.get_gt(inputs, 0, cache);
                a.storage.affine(&a.layout(), *c, 0.0).expect("MulScalar")
            }
            Op::PowI(n) => {
                let a = self.get_gt(inputs, 0, cache);
                a.storage.powf(&a.layout(), *n as f64).expect("PowI")
            }

            // Cast.
            Op::Cast(target) => {
                let a = self.get_gt(inputs, 0, cache);
                a.storage.to_dtype(&a.layout(), *target).expect("Cast")
            }

            // Reshape: if the input is an Owned entry, steal its storage
            // (avoiding a GPU memcpy). If it's a ConstRef, we must clone
            // since the const_pool owns the original.
            Op::Reshape(_) => {
                let entry = cache.get(&inputs[0]).expect("reshape: missing input");
                match entry {
                    CacheEntry::ConstRef(_) => {
                        let a = self.resolve(entry);
                        let storage = a.storage.try_clone(&a.layout()).expect("Reshape");
                        return CacheEntry::Owned(GpuTensor { storage, shape: shape.clone() });
                    }
                    CacheEntry::Owned(a) => {
                        // Can't move out of the cache (it's borrowed), so
                        // still need try_clone. But this is at least bounded
                        // to computed intermediates, not large weight tensors.
                        let storage = a.storage.try_clone(&a.layout()).expect("Reshape");
                        return CacheEntry::Owned(GpuTensor { storage, shape: shape.clone() });
                    }
                }
            }

            Op::Transpose => {
                let a = self.get_gt(inputs, 0, cache);
                let rank = a.shape.dims().len();
                let mut perm: Vec<usize> = (0..rank).collect();
                perm.swap(rank - 2, rank - 1);
                return CacheEntry::Owned(self.do_permute(a, &perm, shape));
            }
            Op::Permute(axes) => {
                let a = self.get_gt(inputs, 0, cache);
                return CacheEntry::Owned(self.do_permute(a, axes, shape));
            }

            Op::BroadcastTo(target) => {
                let a = self.get_gt(inputs, 0, cache);
                return CacheEntry::Owned(self.do_broadcast(a, target));
            }

            // Reductions.
            Op::SumAll | Op::MeanAll => {
                let a = self.get_gt(inputs, 0, cache);
                let axes: Vec<usize> = (0..a.shape.dims().len()).collect();
                let mut r = a.storage.reduce_op(
                    fuel_core_types::op::ReduceOp::Sum, &a.layout(), &axes,
                ).expect("SumAll");
                if matches!(op, Op::MeanAll) {
                    let n = a.shape.elem_count() as f64;
                    r = r.affine(&Layout::contiguous(shape), 1.0 / n, 0.0)
                        .expect("MeanAll scale");
                }
                r
            }
            Op::MaxAll => {
                let a = self.get_gt(inputs, 0, cache);
                let axes: Vec<usize> = (0..a.shape.dims().len()).collect();
                a.storage.reduce_op(fuel_core_types::op::ReduceOp::Max, &a.layout(), &axes)
                    .expect("MaxAll")
            }
            Op::MinAll => {
                let a = self.get_gt(inputs, 0, cache);
                let axes: Vec<usize> = (0..a.shape.dims().len()).collect();
                a.storage.reduce_op(fuel_core_types::op::ReduceOp::Min, &a.layout(), &axes)
                    .expect("MinAll")
            }
            Op::SumDim(d) | Op::MeanDim(d) => {
                let a = self.get_gt(inputs, 0, cache);
                let mut r = a.storage.reduce_op(
                    fuel_core_types::op::ReduceOp::Sum, &a.layout(), &[*d],
                ).expect("SumDim");
                if matches!(op, Op::MeanDim(_)) {
                    let n = a.shape.dims()[*d] as f64;
                    r = r.affine(&Layout::contiguous(shape), 1.0 / n, 0.0)
                        .expect("MeanDim scale");
                }
                r
            }
            Op::MaxDim(d) => {
                let a = self.get_gt(inputs, 0, cache);
                a.storage.reduce_op(fuel_core_types::op::ReduceOp::Max, &a.layout(), &[*d])
                    .expect("MaxDim")
            }
            Op::MinDim(d) => {
                let a = self.get_gt(inputs, 0, cache);
                a.storage.reduce_op(fuel_core_types::op::ReduceOp::Min, &a.layout(), &[*d])
                    .expect("MinDim")
            }

            // Indexing.
            Op::IndexSelect { dim } => {
                let (src, ids) = (self.get_gt(inputs, 0, cache), self.get_gt(inputs, 1, cache));
                src.storage.index_select(&ids.storage, &src.layout(), &ids.layout(), *dim)
                    .expect("IndexSelect")
            }
            Op::Gather { dim } => {
                let (src, ids) = (self.get_gt(inputs, 0, cache), self.get_gt(inputs, 1, cache));
                src.storage.gather(&src.layout(), &ids.storage, &ids.layout(), *dim)
                    .expect("Gather")
            }

            // Concat / Slice.
            Op::Concat { dim } => return CacheEntry::Owned(self.eval_concat(*dim, inputs, shape, cache)),
            Op::Slice { dim, start, len } => {
                let a = self.get_gt(inputs, 0, cache);
                return CacheEntry::Owned(self.eval_slice(*dim, *start, *len, a, shape));
            }

            // Native CUDA softmax — the kernel from reduce.cu.
            // Phase 7.6 step 5 (2026-05-11): SoftmaxLastDim now flows
            // through `Op::Fused(SOFTMAX_LAST_DIM, _)` per the registry
            // split; the legacy `Op::SoftmaxLastDim` arm was retired
            // with the variant.
            Op::Fused(fid, _)
                if *fid == fuel_graph::registry::FusedOps::SOFTMAX_LAST_DIM =>
            {
                let a = self.get_gt(inputs, 0, cache);
                a.storage.softmax_last_dim(&a.layout()).expect("SoftmaxLastDim")
            }

            // Everything else: CPU fallback.
            _ => {
                return CacheEntry::Owned(self.cpu_fallback(inputs, shape, cache, |ni, ns, cc| {
                    fuel_reference_backend::exec::eval_node_with_op(op, ni, ns, dtype, cc)
                }));
            }
        };

        CacheEntry::Owned(GpuTensor { storage: result_storage, shape: shape.clone() })
    }

    // --- op helpers ---

    fn unary_cuda(
        &self,
        kernel: &'static str,
        inputs: &[NodeId],
        cache: &HashMap<NodeId, CacheEntry>,
    ) -> CudaStorage {
        let a = self.get_gt(inputs, 0, cache);
        a.storage.unary_by_name(kernel, &a.layout()).expect(kernel)
    }

    fn binary_cuda(
        &self,
        kernel: &'static str,
        inputs: &[NodeId],
        cache: &HashMap<NodeId, CacheEntry>,
    ) -> CudaStorage {
        let (a, b) = (self.get_gt(inputs, 0, cache), self.get_gt(inputs, 1, cache));
        a.storage
            .binary_by_name(&b.storage, &a.layout(), &b.layout(), kernel)
            .expect(kernel)
    }

    /// Phase 7.5 G2: slot-first dispatch for fuel-cuda-backend. If
    /// the graph's storage_map has a populated slot for `id`, adopt
    /// its bytes via host-buffer download + H2D upload to the CUDA
    /// device. Returns `None` if no slot is registered.
    ///
    /// const_pool keying on slot-Arc identity is intentionally
    /// disabled here for the same reason as fuel-graph-executor:
    /// graphs drop and re-create slots across realize calls (training
    /// loop, fresh graph per step), and pointer recycling can cause
    /// stale cache hits on a recycled address. Step 3c re-adds
    /// pooling with proper liveness tracking.
    fn try_adopt_slot_cuda(
        &mut self,
        graph: &fuel_graph::Graph,
        id: NodeId,
        shape: &Shape,
    ) -> Option<CacheEntry> {
        let slot_arc = graph.storage_for(id)?;
        let buf = {
            let slot = slot_arc.read().unwrap();
            slot.as_dyn().to_host_buffer_dyn().expect("slot D2H")
        };
        let gpu = self.device.storage_from_cpu_storage(&buf)
            .expect("slot H2D");
        Some(CacheEntry::Owned(GpuTensor { storage: gpu, shape: shape.clone() }))
    }

    fn do_permute(&self, a: &GpuTensor, axes: &[usize], out_shape: &Shape) -> GpuTensor {
        let _s = debug_span!("permute", elems = out_shape.elem_count()).entered();
        let in_dims = a.shape.dims();
        let rank = in_dims.len();
        let mut strides: fuel_core_types::StrideVec = fuel_core_types::StrideVec::from_elem(0_isize, rank);
        let mut s: isize = 1;
        for i in (0..rank).rev() {
            strides[i] = s;
            s *= in_dims[i] as isize;
        }
        let permuted_strides: fuel_core_types::StrideVec = axes.iter().map(|&ax| strides[ax]).collect();
        let permuted_dims: Vec<usize> = axes.iter().map(|&ax| in_dims[ax]).collect();
        let src_layout = Layout::new(
            Shape::from_dims(&permuted_dims),
            permuted_strides,
            0,
        );
        let mut dst = self.device.zeros_impl(out_shape, a.storage.dtype())
            .expect("permute alloc");
        a.storage.copy_strided_src(&mut dst, 0, &src_layout)
            .expect("permute copy");
        GpuTensor { storage: dst, shape: out_shape.clone() }
    }

    fn do_broadcast(&self, a: &GpuTensor, target: &Shape) -> GpuTensor {
        let src_dims = a.shape.dims();
        let dst_dims = target.dims();

        // Pure-pad shortcut: if the broadcast only adds leading 1-dims
        // and all aligned dims match, the element count is unchanged
        // and the memory layout is identical. Just relabel the shape
        // — a simple try_clone (D2D copy) instead of the expensive
        // strided-copy kernel, and the layout is still correct.
        let pad = dst_dims.len().saturating_sub(src_dims.len());
        let is_pure_pad = dst_dims[..pad].iter().all(|&d| d == 1)
            && src_dims.iter().zip(&dst_dims[pad..]).all(|(s, d)| s == d);

        if src_dims == dst_dims || is_pure_pad {
            let _s = debug_span!("broadcast_pure_pad", elems = target.elem_count()).entered();
            return GpuTensor {
                storage: a.storage.try_clone(&a.layout()).expect("broadcast pad"),
                shape: target.clone(),
            };
        }

        let _s = debug_span!("broadcast_strided", src_elems = a.shape.elem_count(), dst_elems = target.elem_count()).entered();
        let mut strides: fuel_core_types::StrideVec = fuel_core_types::StrideVec::from_elem(0_isize, dst_dims.len());
        let mut s: isize = 1;
        for i in (0..src_dims.len()).rev() {
            if src_dims[i] == dst_dims[pad + i] {
                strides[pad + i] = s;
            }
            s *= src_dims[i] as isize;
        }
        let src_layout = Layout::new(target.clone(), strides, 0);
        let mut dst = self.device.zeros_impl(target, a.storage.dtype())
            .expect("broadcast alloc");
        a.storage.copy_strided_src(&mut dst, 0, &src_layout)
            .expect("broadcast copy");
        GpuTensor { storage: dst, shape: target.clone() }
    }

    fn eval_concat(
        &self,
        dim: usize,
        inputs: &[NodeId],
        out_shape: &Shape,
        cache: &HashMap<NodeId, CacheEntry>,
    ) -> GpuTensor {
        let _s = debug_span!("concat", dim, elems = out_shape.elem_count()).entered();
        let a = self.get_gt(inputs, 0, cache);
        let b = self.get_gt(inputs, 1, cache);
        let mut dst = self.device.zeros_impl(out_shape, a.storage.dtype())
            .expect("concat alloc");

        let out_dims = out_shape.dims();
        let a_dim = a.shape.dims()[dim];
        let b_dim = b.shape.dims()[dim];
        let inner: usize = out_dims[dim + 1..].iter().product::<usize>().max(1);
        let outer: usize = out_dims[..dim].iter().product::<usize>().max(1);
        let out_row = out_dims[dim] * inner;

        if outer == 1 {
            a.storage.copy_strided_src(&mut dst, 0, &a.layout()).expect("concat a");
            b.storage.copy_strided_src(&mut dst, a_dim * inner, &b.layout()).expect("concat b");
        } else {
            let a_slice_size = a_dim * inner;
            let b_slice_size = b_dim * inner;
            for o in 0..outer {
                let a_layout = Layout::contiguous_with_offset(
                    &Shape::from_dims(&[a_slice_size]),
                    o * a_slice_size,
                );
                a.storage.copy_strided_src(&mut dst, o * out_row, &a_layout)
                    .expect("concat a slice");
                let b_layout = Layout::contiguous_with_offset(
                    &Shape::from_dims(&[b_slice_size]),
                    o * b_slice_size,
                );
                b.storage.copy_strided_src(&mut dst, o * out_row + a_slice_size, &b_layout)
                    .expect("concat b slice");
            }
        }
        GpuTensor { storage: dst, shape: out_shape.clone() }
    }

    fn eval_slice(
        &self,
        dim: usize,
        start: usize,
        _len: usize,
        a: &GpuTensor,
        out_shape: &Shape,
    ) -> GpuTensor {
        let in_dims = a.shape.dims();
        let rank = in_dims.len();
        let mut strides: fuel_core_types::StrideVec = fuel_core_types::StrideVec::from_elem(0_isize, rank);
        let mut s: isize = 1;
        for i in (0..rank).rev() {
            strides[i] = s;
            s *= in_dims[i] as isize;
        }
        let offset = (start as isize * strides[dim]) as usize;
        let src_layout = Layout::new(out_shape.clone(), strides, offset);
        let mut dst = self.device.zeros_impl(out_shape, a.storage.dtype())
            .expect("slice alloc");
        a.storage.copy_strided_src(&mut dst, 0, &src_layout)
            .expect("slice copy");
        GpuTensor { storage: dst, shape: out_shape.clone() }
    }

    fn cpu_fallback(
        &self,
        inputs: &[NodeId],
        shape: &Shape,
        cache: &HashMap<NodeId, CacheEntry>,
        f: impl FnOnce(&[NodeId], &Shape, &HashMap<NodeId, AnyRef>) -> AnyRef,
    ) -> GpuTensor {
        let _s = info_span!("cpu_fallback", elems = shape.elem_count()).entered();
        let mut cpu_cache: HashMap<NodeId, AnyRef> = HashMap::new();
        for &id in inputs {
            let gt = self.resolve(cache.get(&id).expect("cpu_fallback: missing input"));
            let cpu_buf = gt.storage.to_cpu_storage().expect("D2H fallback");
            cpu_cache.insert(id, host_buffer_to_any_ref(cpu_buf, &gt.shape));
        }
        let result = f(inputs, shape, &cpu_cache);
        let out_buf = any_ref_to_host_buffer(result);
        let gpu = self.device.storage_from_cpu_storage(&out_buf)
            .expect("H2D fallback");
        GpuTensor { storage: gpu, shape: shape.clone() }
    }
}

// --- free-function helpers ---

fn gpu_to_ref_f32(gt: GpuTensor) -> RefTensor<f32> {
    let cpu = gt.storage.to_cpu_storage().expect("D2H");
    match cpu {
        fuel_core_types::HostBuffer::F32(v) => RefTensor::from_vec(v, gt.shape),
        other => panic!("gpu_to_ref_f32: dtype {:?}", other.dtype()),
    }
}

fn gpu_to_ref_f32_ref(gt: &GpuTensor) -> RefTensor<f32> {
    let cpu = gt.storage.to_cpu_storage().expect("D2H");
    match cpu {
        fuel_core_types::HostBuffer::F32(v) => RefTensor::from_vec(v, gt.shape.clone()),
        other => panic!("gpu_to_ref_f32: dtype {:?}", other.dtype()),
    }
}

fn host_buffer_to_any_ref(buf: fuel_core_types::HostBuffer, shape: &Shape) -> AnyRef {
    match buf {
        fuel_core_types::HostBuffer::F32(v) => AnyRef::F32(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::F64(v) => AnyRef::F64(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::BF16(v) => AnyRef::BF16(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::F16(v) => AnyRef::F16(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::U32(v) => AnyRef::U32(RefTensor::from_vec(v, shape.clone())),
        _ => panic!("host_buffer_to_any_ref: unsupported dtype"),
    }
}

fn op_short_name(op: &Op) -> &'static str {
    match op {
        Op::Const => "Const",
        Op::MatMul => "MatMul",
        Op::Add => "Add", Op::Sub => "Sub", Op::Mul => "Mul", Op::Div => "Div",
        Op::Neg => "Neg", Op::Sqr => "Sqr", Op::Sqrt => "Sqrt",
        Op::Exp => "Exp", Op::Log => "Log",
        Op::Sin => "Sin", Op::Cos => "Cos", Op::Tanh => "Tanh",
        Op::Sigmoid => "Sigmoid", Op::Silu => "Silu", Op::Gelu => "Gelu",
        Op::Relu => "Relu", Op::Step => "Step",
        Op::Maximum => "Maximum", Op::Minimum => "Minimum",
        Op::AddScalar(_) => "AddScalar", Op::MulScalar(_) => "MulScalar",
        Op::PowI(_) => "PowI",
        Op::Cast(_) => "Cast",
        Op::Reshape(_) => "Reshape",
        Op::Transpose => "Transpose", Op::Permute(_) => "Permute",
        Op::BroadcastTo(_) => "BroadcastTo",
        Op::SumAll => "SumAll", Op::MeanAll => "MeanAll",
        Op::MaxAll => "MaxAll", Op::MinAll => "MinAll",
        Op::SumDim(_) => "SumDim", Op::MeanDim(_) => "MeanDim",
        Op::MaxDim(_) => "MaxDim", Op::MinDim(_) => "MinDim",
        Op::IndexSelect { .. } => "IndexSelect",
        Op::Gather { .. } => "Gather",
        Op::Concat { .. } => "Concat",
        Op::Slice { .. } => "Slice",
        // Phase 7.6 step 5: fused ops (SoftmaxLastDim et al.) hit
        // the catch-all here. Per-id names land alongside the
        // step-7/8 PrecisionGuarantee/cost-model pass.
        _ => "Other",
    }
}

fn any_ref_to_host_buffer(any: AnyRef) -> fuel_core_types::HostBuffer {
    use fuel_core_types::HostBuffer;
    match any {
        AnyRef::F32(t) => HostBuffer::F32(t.into_vec()),
        AnyRef::F64(t) => HostBuffer::F64(t.into_vec()),
        AnyRef::BF16(t) => HostBuffer::BF16(t.into_vec()),
        AnyRef::F16(t) => HostBuffer::F16(t.into_vec()),
        AnyRef::U32(t) => HostBuffer::U32(t.into_vec()),
    }
}

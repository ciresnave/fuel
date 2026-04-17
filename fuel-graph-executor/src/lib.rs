//! Backend-generic graph executor for fuel.
//!
//! This crate provides [`GraphExecutor<B>`], a generic executor that
//! walks a `fuel-graph` computation graph using any backend that
//! implements the [`GraphBackend`] trait. All the shared infrastructure
//! — topological walk, const pool with Arc-pointer dedup, cache entry
//! management, realize loop, CPU fallback bridge, tracing spans,
//! layout ops (reshape, permute, broadcast, concat, slice) — is
//! written once here and automatically benefits every backend.
//!
//! Backend crates (`fuel-graph-cpu`, `fuel-graph-cuda`, future
//! `fuel-graph-metal`) implement `GraphBackend` in ~200 lines each,
//! providing only the device-specific pieces: memory allocation,
//! matmul, unary/binary kernels, reductions, and softmax.

pub mod shaders;

use fuel_core_types::{DType, DimVec, Layout, Shape};
use fuel_graph::{topo_order, topo_order_multi, ConstData, NodeId, Op, Tensor};
use fuel_reference_backend::exec::AnyRefTensor;
use fuel_reference_backend::RefTensor;
use std::collections::HashMap;
use tracing::{debug_span, info_span};

// ---- Op sub-enums for dispatch to backend -----------------------------------

/// Unary ops dispatched to the backend's native implementation.
#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg, Sqr, Sqrt, Exp, Log, Sin, Cos, Tanh,
    Sigmoid, Silu, Gelu, Relu, Step,
}

/// Binary ops dispatched to the backend's native implementation.
#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Maximum, Minimum,
}

// ---- Tracked tensor ---------------------------------------------------------

/// A storage value paired with its shape, backed by `Arc<S>` so that
/// "pure-pad broadcast" and "reshape" can share the same underlying
/// backend storage with a new shape label — no GPU memcpy, no
/// device allocation. On CPU this mirrors the existing `Arc<[T]>`
/// pattern in `RefTensor`. On GPU this is the difference between
/// "reshape costs 256 MB of device memcpy" and "reshape is free."
pub struct TrackedTensor<S> {
    pub storage: std::sync::Arc<S>,
    pub shape: Shape,
    /// Non-contiguous layout for views (broadcast with stride 0,
    /// sliced offsets, etc.). `None` means contiguous row-major.
    /// When set, `layout()` returns this instead of computing
    /// contiguous strides — downstream ops like `copy_strided_src`
    /// then read from the correct physical locations.
    custom_layout: Option<Layout>,
}

impl<S> TrackedTensor<S> {
    pub fn new(storage: S, shape: Shape) -> Self {
        Self { storage: std::sync::Arc::new(storage), shape, custom_layout: None }
    }

    pub fn with_custom_layout(storage: S, shape: Shape, layout: Layout) -> Self {
        Self { storage: std::sync::Arc::new(storage), shape, custom_layout: Some(layout) }
    }

    pub fn layout(&self) -> Layout {
        match &self.custom_layout {
            Some(l) => l.clone(),
            None => Layout::contiguous(&self.shape),
        }
    }

    /// Cheap: just bumps the Arc and copies the shape.
    pub fn with_shape(&self, new_shape: Shape) -> Self {
        Self {
            storage: std::sync::Arc::clone(&self.storage),
            shape: new_shape,
            custom_layout: None,
        }
    }

    /// Borrow the inner storage for read-only backend calls.
    pub fn inner(&self) -> &S {
        &self.storage
    }
}

// ---- Cache entry ------------------------------------------------------------

/// Per-node cache entry during a realize pass.
pub enum CacheEntry<S> {
    /// Points into the executor's persistent `const_pool`. The pool
    /// outlives the per-realize cache — zero GPU/CPU alloc on hit.
    ConstRef(usize),
    /// An intermediate computed during this realize pass. Freed when
    /// the cache is dropped at the end of realize.
    Owned(TrackedTensor<S>),
}

// ---- GraphBackend trait -----------------------------------------------------

/// The pluggable backend interface. Implement this for each compute
/// target (CPU, CUDA, Metal, …). All methods receive borrowed storage
/// and return new owned storage.
pub trait GraphBackend {
    /// The concrete storage type — `RefTensor<_>`, `CudaStorage`, etc.
    type Storage;

    // -- memory --

    /// Allocate a zero-initialized tensor on the device.
    fn alloc_zeros(&self, shape: &Shape, dtype: DType) -> fuel_core_types::Result<Self::Storage>;

    /// Upload host data to the device. The `shape` describes the
    /// logical tensor shape — backends that store shape in their
    /// storage type (like CPU's RefTensor) should use it.
    fn upload(&self, buf: &fuel_core_types::HostBuffer, shape: &Shape) -> fuel_core_types::Result<Self::Storage>;

    /// Download device data to host.
    fn download(&self, storage: &Self::Storage) -> fuel_core_types::Result<fuel_core_types::HostBuffer>;

    /// Clone a contiguous region described by `layout`.
    fn try_clone(&self, storage: &Self::Storage, layout: &Layout) -> fuel_core_types::Result<Self::Storage>;

    /// Copy a strided region from `src` into `dst` at `dst_offset`.
    fn copy_strided_src(
        &self,
        src: &Self::Storage,
        dst: &mut Self::Storage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> fuel_core_types::Result<()>;

    /// The dtype of a storage value.
    fn storage_dtype(&self, storage: &Self::Storage) -> DType;

    // -- compute --

    fn matmul(
        &self,
        a: &Self::Storage, b: &Self::Storage,
        bmnk: (usize, usize, usize, usize),
        la: &Layout, lb: &Layout,
    ) -> fuel_core_types::Result<Self::Storage>;

    fn unary(
        &self, op: UnaryOp,
        a: &Self::Storage, layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage>;

    fn binary(
        &self, op: BinaryOp,
        a: &Self::Storage, b: &Self::Storage,
        la: &Layout, lb: &Layout,
    ) -> fuel_core_types::Result<Self::Storage>;

    fn affine(
        &self, a: &Self::Storage, layout: &Layout,
        mul: f64, add: f64,
    ) -> fuel_core_types::Result<Self::Storage>;

    fn powf(
        &self, a: &Self::Storage, layout: &Layout,
        exp: f64,
    ) -> fuel_core_types::Result<Self::Storage>;

    fn cast(
        &self, a: &Self::Storage, layout: &Layout,
        dtype: DType,
    ) -> fuel_core_types::Result<Self::Storage>;

    fn reduce(
        &self, op: fuel_core_types::op::ReduceOp,
        a: &Self::Storage, layout: &Layout,
        dims: &[usize],
    ) -> fuel_core_types::Result<Self::Storage>;

    fn softmax_last_dim(
        &self, a: &Self::Storage, layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage>;

    /// Fused root-mean-square normalization along the last dimension.
    /// `y = x / sqrt(mean(x², last) + eps)`.
    ///
    /// Default impl returns `Err` — the executor then falls back to
    /// the CPU reference implementation. Backends that can run this
    /// natively (single-dispatch fused kernel) override this method
    /// and save ~8 dispatches per call vs the decomposed form.
    fn rms_norm_last_dim(
        &self, a: &Self::Storage, layout: &Layout, eps: f64,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (a, layout, eps);
        fuel_core_types::bail!(
            "GraphBackend: rms_norm_last_dim not implemented natively for this backend"
        )
    }

    /// Concatenate `a` and `b` along `dim` in a single dispatch. Both
    /// inputs must be contiguous. Output has shape `a.shape` with
    /// `a.shape[dim] + b.shape[dim]` at `dim`.
    ///
    /// Default impl returns `Err`; the executor falls back to the
    /// `outer × 2` strided-copy loop. Backends override this when a
    /// single-dispatch kernel exists.
    fn concat_along_dim(
        &self,
        a: &Self::Storage,
        b: &Self::Storage,
        dim: usize,
        a_shape: &Shape,
        b_shape: &Shape,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (a, b, dim, a_shape, b_shape);
        fuel_core_types::bail!("GraphBackend: concat_along_dim not implemented natively")
    }

    /// Fused backward for RMSNorm-last-dim. Inputs: (x, upstream).
    /// Output: grad_x. Formula:
    ///
    /// ```text
    ///   s       = sum(upstream * x, last)
    ///   mean_sq = mean(x², last)
    ///   grad_x  = r_rms * (upstream - x * s / (n * (mean_sq + eps)))
    /// ```
    ///
    /// Default impl returns `Err`; executor falls back to CPU.
    fn rms_norm_last_dim_backward(
        &self,
        x: &Self::Storage,
        upstream: &Self::Storage,
        x_layout: &Layout,
        up_layout: &Layout,
        eps: f64,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (x, upstream, x_layout, up_layout, eps);
        fuel_core_types::bail!("GraphBackend: rms_norm_last_dim_backward not implemented natively")
    }

    /// Fused layer-norm backward. Inputs: (x, upstream). Takes eps.
    fn layer_norm_last_dim_backward(
        &self,
        x: &Self::Storage,
        upstream: &Self::Storage,
        x_layout: &Layout,
        up_layout: &Layout,
        eps: f64,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (x, upstream, x_layout, up_layout, eps);
        fuel_core_types::bail!("GraphBackend: layer_norm_last_dim_backward not implemented natively")
    }

    /// Fused softmax backward: `dx = y * (g - dot(y, g))`.
    /// Inputs: (softmax_output, upstream). Default returns Err.
    fn softmax_last_dim_backward(
        &self,
        y: &Self::Storage,
        upstream: &Self::Storage,
        y_layout: &Layout,
        up_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (y, upstream, y_layout, up_layout);
        fuel_core_types::bail!("GraphBackend: softmax_last_dim_backward not implemented natively")
    }

    /// Fused rotary position embedding. Applies the rotate_half-form
    /// rotation in a single dispatch. `x` has shape `[..., seq, head_dim]`
    /// (head_dim even). `cos` and `sin` both have shape `[seq, head_dim]`
    /// and broadcast across all leading dims of x.
    ///
    /// Default impl returns `Err`; executor falls back to CPU. Backends
    /// that implement this natively avoid the ~72 dispatches the
    /// slice+concat decomposition produces per call on GPU backends.
    fn rope(
        &self,
        x: &Self::Storage,
        cos: &Self::Storage,
        sin: &Self::Storage,
        x_layout: &Layout,
        cos_layout: &Layout,
        sin_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (x, cos, sin, x_layout, cos_layout, sin_layout);
        fuel_core_types::bail!("GraphBackend: rope not implemented natively")
    }

    /// In-place scaled accumulate: `dst += src * scale`. All three
    /// tensors share the same shape and dtype. No new allocation —
    /// `dst` is mutated in place.
    ///
    /// Used primarily by training loops to do SGD's `w ← w − lr·g`
    /// update without allocating a fresh buffer for the new `w`.
    /// Default impl returns `Err` so the training code can fall
    /// back to the alloc-every-step path.
    fn add_assign_scaled(
        &self,
        dst: &mut Self::Storage,
        src: &Self::Storage,
        scale: f32,
    ) -> fuel_core_types::Result<()> {
        let _ = (dst, src, scale);
        fuel_core_types::bail!(
            "GraphBackend: add_assign_scaled not implemented natively for this backend"
        )
    }

    fn index_select(
        &self, src: &Self::Storage, ids: &Self::Storage,
        src_l: &Layout, ids_l: &Layout, dim: usize,
    ) -> fuel_core_types::Result<Self::Storage>;

    fn gather(
        &self, src: &Self::Storage, ids: &Self::Storage,
        src_l: &Layout, ids_l: &Layout, dim: usize,
    ) -> fuel_core_types::Result<Self::Storage>;
}

// ---- GraphExecutor<B> -------------------------------------------------------

/// Backend-generic graph executor. Walks a fuel-graph in topological
/// order, dispatching each op through a `GraphBackend` implementation.
///
/// Shared infrastructure (written once, benefits all backends):
/// - Persistent const pool with Arc-pointer dedup
/// - Per-realize cache with ConstRef / Owned entries
/// - Pre-populated (injected) node support for KV cache
/// - Pure-pad broadcast shortcut
/// - Layout computation for permute, slice, concat
/// - CPU fallback bridge for unsupported ops
/// - Tracing spans at realize / topo-walk / per-op granularity
pub struct GraphExecutor<B: GraphBackend> {
    pub backend: B,
    /// Persistent weight cache, keyed on host Arc data pointer.
    /// Only caches consts with Arc::strong_count > 1 (model weights).
    const_pool: HashMap<usize, TrackedTensor<B::Storage>>,
    /// Pre-populated entries for the next realize call.
    injected: HashMap<NodeId, TrackedTensor<B::Storage>>,
    /// If true, realize runs the backend-agnostic `fuel_graph::opt`
    /// pass (CSE + algebraic simplification) on the graph before
    /// walking it. Off by default because it mutates the shared graph
    /// arena (appends canonical nodes), which existing test code may
    /// not expect; opt-in per-executor via `with_optimization(true)`.
    optimize: bool,
}

impl<B: GraphBackend> GraphExecutor<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            const_pool: HashMap::new(),
            injected: HashMap::new(),
            optimize: false,
        }
    }

    /// Enable or disable graph-level optimization (CSE, algebraic
    /// simplification) before each realize. Pre-populated / injected
    /// nodes are preserved — they're leaves from the optimizer's view
    /// and can't be eliminated.
    pub fn with_optimization(mut self, enabled: bool) -> Self {
        self.optimize = enabled;
        self
    }

    /// Pre-populate a node with an existing device-side tensor.
    pub fn pre_populate(&mut self, node_id: NodeId, storage: B::Storage, shape: Shape) {
        self.injected.insert(node_id, TrackedTensor::new(storage, shape));
    }

    /// Resolve the (possibly-rewritten) root NodeIds for a slice of
    /// tensor handles. When optimization is disabled this is a noop
    /// identity map. When enabled it runs the optimizer pass, which
    /// may redirect roots to canonicalized nodes.
    fn resolve_roots(&self, tensors: &[&Tensor]) -> Vec<NodeId> {
        let original: Vec<NodeId> = tensors.iter().map(|t| t.id()).collect();
        if !self.optimize || tensors.is_empty() {
            return original;
        }
        let graph = tensors[0].graph();
        fuel_graph::opt::optimize(graph, &original)
    }

    // -- realize entry points -------------------------------------------------

    pub fn realize_f32(&mut self, tensor: &Tensor) -> RefTensor<f32> {
        let _span = info_span!("realize_f32").entered();
        let root_id = self.resolve_roots(&[tensor])[0];
        let graph = tensor.graph().borrow();
        let order = topo_order(&graph, root_id);
        let _walk = info_span!("topo_walk", nodes = order.len()).entered();
        let mut cache = self.drain_injected();
        for id in order {
            if cache.contains_key(&id) { continue; }
            let node = graph.node(id);
            let entry = self.eval_node(&node.op, &node.inputs, &node.shape, node.dtype, &cache);
            cache.insert(id, entry);
        }
        drop(_walk);
        let _rb = info_span!("d2h_readback").entered();
        let gt = self.take_owned(cache.remove(&root_id).expect("realize: missing root"));
        let buf = self.backend.download(&gt.storage).expect("D2H");
        match buf {
            fuel_core_types::HostBuffer::F32(v) => RefTensor::from_vec(v, gt.shape),
            other => panic!("realize_f32: got {:?}", other.dtype()),
        }
    }

    pub fn realize_many_f32(&mut self, tensors: &[&Tensor]) -> Vec<RefTensor<f32>> {
        let _span = info_span!("realize_many_f32", roots = tensors.len()).entered();
        if tensors.is_empty() { return Vec::new(); }
        let roots: Vec<NodeId> = self.resolve_roots(tensors);
        let graph_rc = tensors[0].graph();
        let graph = graph_rc.borrow();
        let order = topo_order_multi(&graph, &roots);
        let _walk = info_span!("topo_walk", nodes = order.len()).entered();
        let mut cache = self.drain_injected();
        for id in order {
            if cache.contains_key(&id) { continue; }
            let node = graph.node(id);
            let entry = self.eval_node(&node.op, &node.inputs, &node.shape, node.dtype, &cache);
            cache.insert(id, entry);
        }
        drop(_walk);
        let _rb = info_span!("d2h_readback").entered();
        roots.iter().map(|id| {
            let gt = self.resolve(cache.get(id).expect("realize_many: missing"));
            let buf = self.backend.download(&gt.storage).expect("D2H");
            match buf {
                fuel_core_types::HostBuffer::F32(v) => RefTensor::from_vec(v, gt.shape.clone()),
                other => panic!("realize_many_f32: got {:?}", other.dtype()),
            }
        }).collect()
    }

    /// Split realize: first `n_d2h` roots download to CPU, rest stay on device.
    pub fn realize_split(
        &mut self,
        tensors: &[&Tensor],
        n_d2h: usize,
    ) -> (Vec<Vec<f32>>, Vec<(B::Storage, Shape)>) {
        let _span = info_span!("realize_split", roots = tensors.len(), n_d2h).entered();
        if tensors.is_empty() { return (Vec::new(), Vec::new()); }
        let roots: Vec<NodeId> = self.resolve_roots(tensors);
        let graph_rc = tensors[0].graph();
        let graph = graph_rc.borrow();
        let order = topo_order_multi(&graph, &roots);
        let _walk = info_span!("topo_walk", nodes = order.len()).entered();
        let mut cache = self.drain_injected();
        for id in order {
            if cache.contains_key(&id) { continue; }
            let node = graph.node(id);
            let entry = self.eval_node(&node.op, &node.inputs, &node.shape, node.dtype, &cache);
            cache.insert(id, entry);
        }
        drop(_walk);
        let _rb = info_span!("d2h_readback", n_d2h).entered();
        let mut cpu_out = Vec::with_capacity(n_d2h);
        let mut gpu_out = Vec::with_capacity(roots.len() - n_d2h);
        for (i, id) in roots.iter().enumerate() {
            if i < n_d2h {
                let gt = self.resolve(cache.get(id).expect("split: missing"));
                let buf = self.backend.download(&gt.storage).expect("D2H");
                match buf {
                    fuel_core_types::HostBuffer::F32(v) => cpu_out.push(v),
                    other => panic!("split: got {:?}", other.dtype()),
                }
            } else {
                match cache.remove(id) {
                    Some(CacheEntry::Owned(gt)) => {
                        let shape = gt.shape.clone();
                        let layout = gt.layout();
                        let s = std::sync::Arc::try_unwrap(gt.storage)
                            .unwrap_or_else(|arc| self.backend.try_clone(&arc, &layout).expect("split clone"));
                        gpu_out.push((s, shape));
                    }
                    Some(CacheEntry::ConstRef(key)) => {
                        let p = self.const_pool.get(&key).expect("dangling");
                        let s = self.backend.try_clone(&p.storage, &p.layout()).expect("clone");
                        gpu_out.push((s, p.shape.clone()));
                    }
                    None => panic!("split: missing root"),
                }
            }
        }
        (cpu_out, gpu_out)
    }

    // -- internal helpers -----------------------------------------------------

    fn drain_injected(&mut self) -> HashMap<NodeId, CacheEntry<B::Storage>> {
        let mut cache = HashMap::new();
        for (id, gt) in self.injected.drain() {
            cache.insert(id, CacheEntry::Owned(gt));
        }
        cache
    }

    fn resolve<'a>(&'a self, entry: &'a CacheEntry<B::Storage>) -> &'a TrackedTensor<B::Storage> {
        match entry {
            CacheEntry::ConstRef(key) => self.const_pool.get(key).expect("dangling ConstRef"),
            CacheEntry::Owned(gt) => gt,
        }
    }

    fn get_gt<'a>(
        &'a self,
        inputs: &[NodeId],
        idx: usize,
        cache: &'a HashMap<NodeId, CacheEntry<B::Storage>>,
    ) -> &'a TrackedTensor<B::Storage> {
        self.resolve(cache.get(&inputs[idx]).expect("topo: missing input"))
    }

    fn take_owned(&self, entry: CacheEntry<B::Storage>) -> TrackedTensor<B::Storage> {
        match entry {
            CacheEntry::Owned(gt) => gt,
            CacheEntry::ConstRef(key) => {
                let p = self.const_pool.get(&key).expect("dangling");
                let s = self.backend.try_clone(&p.storage, &p.layout()).expect("take clone");
                TrackedTensor::new(s, p.shape.clone())
            }
        }
    }

    // -- eval_node: the big dispatcher ----------------------------------------

    fn eval_node(
        &mut self,
        op: &Op,
        inputs: &[NodeId],
        shape: &Shape,
        dtype: DType,
        cache: &HashMap<NodeId, CacheEntry<B::Storage>>,
    ) -> CacheEntry<B::Storage> {
        let _span = debug_span!("eval_node", op = op_short_name(op), elems = shape.elem_count()).entered();

        let result_storage = match op {
            Op::Const(data) => return self.eval_const(data, shape),

            // -- matmul --
            Op::MatMul => {
                let (a, b) = (self.get_gt(inputs, 0, cache), self.get_gt(inputs, 1, cache));
                let ad = a.shape.dims();
                let bd = b.shape.dims();
                let rank = ad.len();
                let (m, k, n) = (ad[rank - 2], ad[rank - 1], bd[rank - 1]);
                let batch: usize = ad[..rank - 2].iter().product::<usize>().max(1);
                self.backend.matmul(&a.storage, &b.storage, (batch, m, n, k), &a.layout(), &b.layout())
                    .expect("MatMul")
            }

            // -- unary --
            Op::Neg => self.do_unary(UnaryOp::Neg, inputs, cache),
            Op::Sqr => self.do_unary(UnaryOp::Sqr, inputs, cache),
            Op::Sqrt => self.do_unary(UnaryOp::Sqrt, inputs, cache),
            Op::Exp => self.do_unary(UnaryOp::Exp, inputs, cache),
            Op::Log => self.do_unary(UnaryOp::Log, inputs, cache),
            Op::Sin => self.do_unary(UnaryOp::Sin, inputs, cache),
            Op::Cos => self.do_unary(UnaryOp::Cos, inputs, cache),
            Op::Tanh => self.do_unary(UnaryOp::Tanh, inputs, cache),
            Op::Sigmoid => self.do_unary(UnaryOp::Sigmoid, inputs, cache),
            Op::Silu => self.do_unary(UnaryOp::Silu, inputs, cache),
            Op::Gelu => self.do_unary(UnaryOp::Gelu, inputs, cache),
            Op::Relu => self.do_unary(UnaryOp::Relu, inputs, cache),
            Op::Step => self.do_unary(UnaryOp::Step, inputs, cache),

            // -- binary --
            Op::Add => self.do_binary(BinaryOp::Add, inputs, cache),
            Op::Sub => self.do_binary(BinaryOp::Sub, inputs, cache),
            Op::Mul => self.do_binary(BinaryOp::Mul, inputs, cache),
            Op::Div => self.do_binary(BinaryOp::Div, inputs, cache),
            Op::Maximum => self.do_binary(BinaryOp::Maximum, inputs, cache),
            Op::Minimum => self.do_binary(BinaryOp::Minimum, inputs, cache),

            // -- scalar --
            Op::AddScalar(c) => {
                let a = self.get_gt(inputs, 0, cache);
                self.backend.affine(&a.storage, &a.layout(), 1.0, *c).expect("AddScalar")
            }
            Op::MulScalar(c) => {
                let a = self.get_gt(inputs, 0, cache);
                self.backend.affine(&a.storage, &a.layout(), *c, 0.0).expect("MulScalar")
            }
            Op::PowI(n) => {
                let a = self.get_gt(inputs, 0, cache);
                match self.backend.powf(&a.storage, &a.layout(), *n as f64) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- cast (CPU fallback if backend doesn't implement) --
            Op::Cast(target) => {
                let a = self.get_gt(inputs, 0, cache);
                match self.backend.cast(&a.storage, &a.layout(), *target) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- layout ops (SHARED across all backends) --
            Op::Reshape(_) => {
                let a = self.get_gt(inputs, 0, cache);
                // Pass output shape in the layout so backends that
                // store shape relabel correctly.
                let target_layout = Layout::contiguous(shape);
                let s = self.backend.try_clone(&a.storage, &target_layout).expect("Reshape");
                return CacheEntry::Owned(TrackedTensor::new(s, shape.clone()));
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
            Op::Concat { dim } => {
                return CacheEntry::Owned(self.do_concat(*dim, inputs, shape, cache));
            }
            Op::Slice { dim, start, len: _ } => {
                let a = self.get_gt(inputs, 0, cache);
                return CacheEntry::Owned(self.do_slice(*dim, *start, a, shape));
            }

            // -- reductions --
            Op::SumAll | Op::MeanAll => {
                let a = self.get_gt(inputs, 0, cache);
                let axes: Vec<usize> = (0..a.shape.dims().len()).collect();
                let mut r = self.backend.reduce(
                    fuel_core_types::op::ReduceOp::Sum, &a.storage, &a.layout(), &axes,
                ).expect("SumAll");
                if matches!(op, Op::MeanAll) {
                    let n = a.shape.elem_count() as f64;
                    r = self.backend.affine(&r, &Layout::contiguous(shape), 1.0 / n, 0.0)
                        .expect("MeanAll scale");
                }
                r
            }
            Op::MaxAll => {
                let a = self.get_gt(inputs, 0, cache);
                let axes: Vec<usize> = (0..a.shape.dims().len()).collect();
                self.backend.reduce(fuel_core_types::op::ReduceOp::Max, &a.storage, &a.layout(), &axes)
                    .expect("MaxAll")
            }
            Op::MinAll => {
                let a = self.get_gt(inputs, 0, cache);
                let axes: Vec<usize> = (0..a.shape.dims().len()).collect();
                self.backend.reduce(fuel_core_types::op::ReduceOp::Min, &a.storage, &a.layout(), &axes)
                    .expect("MinAll")
            }
            Op::SumDim(d) | Op::MeanDim(d) => {
                let a = self.get_gt(inputs, 0, cache);
                let r = self.backend.reduce(
                    fuel_core_types::op::ReduceOp::Sum, &a.storage, &a.layout(), &[*d],
                );
                match r {
                    Ok(mut r) => {
                        if matches!(op, Op::MeanDim(_)) {
                            let n = a.shape.dims()[*d] as f64;
                            r = self.backend.affine(&r, &Layout::contiguous(shape), 1.0 / n, 0.0)
                                .expect("MeanDim scale");
                        }
                        r
                    }
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }
            Op::MaxDim(d) => {
                let a = self.get_gt(inputs, 0, cache);
                match self.backend.reduce(fuel_core_types::op::ReduceOp::Max, &a.storage, &a.layout(), &[*d]) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }
            Op::MinDim(d) => {
                let a = self.get_gt(inputs, 0, cache);
                match self.backend.reduce(fuel_core_types::op::ReduceOp::Min, &a.storage, &a.layout(), &[*d]) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- softmax --
            Op::SoftmaxLastDim => {
                let a = self.get_gt(inputs, 0, cache);
                self.backend.softmax_last_dim(&a.storage, &a.layout()).expect("SoftmaxLastDim")
            }

            // -- rms norm (fused) --
            Op::RmsNormLastDim { eps } => {
                let a = self.get_gt(inputs, 0, cache);
                match self.backend.rms_norm_last_dim(&a.storage, &a.layout(), *eps) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- layer norm backward (fused) --
            Op::LayerNormLastDimBackward { eps } => {
                let x = self.get_gt(inputs, 0, cache);
                let up = self.get_gt(inputs, 1, cache);
                match self.backend.layer_norm_last_dim_backward(
                    &x.storage, &up.storage, &x.layout(), &up.layout(), *eps,
                ) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- softmax backward (fused) --
            Op::SoftmaxLastDimBackward => {
                let y = self.get_gt(inputs, 0, cache);
                let up = self.get_gt(inputs, 1, cache);
                match self.backend.softmax_last_dim_backward(
                    &y.storage, &up.storage, &y.layout(), &up.layout(),
                ) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- rms norm backward (fused) --
            Op::RmsNormLastDimBackward { eps } => {
                let x = self.get_gt(inputs, 0, cache);
                let up = self.get_gt(inputs, 1, cache);
                match self.backend.rms_norm_last_dim_backward(
                    &x.storage, &up.storage, &x.layout(), &up.layout(), *eps,
                ) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- rope (fused) --
            Op::Rope => {
                let x = self.get_gt(inputs, 0, cache);
                let cos = self.get_gt(inputs, 1, cache);
                let sin = self.get_gt(inputs, 2, cache);
                match self.backend.rope(
                    &x.storage, &cos.storage, &sin.storage,
                    &x.layout(), &cos.layout(), &sin.layout(),
                ) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- indexing (CPU fallback if backend doesn't implement) --
            Op::IndexSelect { dim } => {
                let (src, ids) = (self.get_gt(inputs, 0, cache), self.get_gt(inputs, 1, cache));
                match self.backend.index_select(&src.storage, &ids.storage, &src.layout(), &ids.layout(), *dim) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }
            Op::Gather { dim } => {
                let (src, ids) = (self.get_gt(inputs, 0, cache), self.get_gt(inputs, 1, cache));
                match self.backend.gather(&src.storage, &ids.storage, &src.layout(), &ids.layout(), *dim) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- fallback for anything else --
            _ => {
                return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache));
            }
        };

        CacheEntry::Owned(TrackedTensor::new(result_storage, shape.clone()))
    }

    // -- shared layout ops (same for ALL backends) ----------------------------

    fn do_unary(
        &self, op: UnaryOp,
        inputs: &[NodeId],
        cache: &HashMap<NodeId, CacheEntry<B::Storage>>,
    ) -> B::Storage {
        let a = self.get_gt(inputs, 0, cache);
        self.backend.unary(op, &a.storage, &a.layout()).expect("unary")
    }

    fn do_binary(
        &self, op: BinaryOp,
        inputs: &[NodeId],
        cache: &HashMap<NodeId, CacheEntry<B::Storage>>,
    ) -> B::Storage {
        let (a, b) = (self.get_gt(inputs, 0, cache), self.get_gt(inputs, 1, cache));
        self.backend.binary(op, &a.storage, &b.storage, &a.layout(), &b.layout()).expect("binary")
    }

    fn do_permute(&self, a: &TrackedTensor<B::Storage>, axes: &[usize], out_shape: &Shape) -> TrackedTensor<B::Storage> {
        let _s = debug_span!("permute", elems = out_shape.elem_count()).entered();
        let in_dims = a.shape.dims();
        let rank = in_dims.len();
        let mut strides: DimVec = DimVec::from_elem(0, rank);
        let mut s = 1usize;
        for i in (0..rank).rev() { strides[i] = s; s *= in_dims[i]; }
        let permuted_strides: DimVec = axes.iter().map(|&ax| strides[ax]).collect();
        let permuted_dims: Vec<usize> = axes.iter().map(|&ax| in_dims[ax]).collect();
        let src_layout = Layout::new(Shape::from_dims(&permuted_dims), permuted_strides, 0);
        let mut dst = self.backend.alloc_zeros(out_shape, self.backend.storage_dtype(&a.storage)).expect("permute alloc");
        self.backend.copy_strided_src(&a.storage, &mut dst, 0, &src_layout).expect("permute copy");
        TrackedTensor::new(dst, out_shape.clone())
    }

    fn do_broadcast(&self, a: &TrackedTensor<B::Storage>, target: &Shape) -> TrackedTensor<B::Storage> {
        let src_dims = a.shape.dims();
        let dst_dims = target.dims();
        let pad = dst_dims.len().saturating_sub(src_dims.len());
        let is_pure_pad = dst_dims[..pad].iter().all(|&d| d == 1)
            && src_dims.iter().zip(&dst_dims[pad..]).all(|(s, d)| s == d);
        if src_dims == dst_dims || is_pure_pad {
            let _s = debug_span!("broadcast_pure_pad", elems = target.elem_count()).entered();
            // Pass target shape in the layout so backends that store
            // shape in their storage (CpuBackend's RefTensor) relabel
            // correctly.
            let target_layout = Layout::contiguous(target);
            let s = self.backend.try_clone(&a.storage, &target_layout).expect("broadcast pad");
            return TrackedTensor::new(s, target.clone());
        }
        let _s = debug_span!("broadcast_strided", src = a.shape.elem_count(), dst = target.elem_count()).entered();
        let mut strides: DimVec = DimVec::from_elem(0, dst_dims.len());
        let mut s = 1usize;
        for i in (0..src_dims.len()).rev() {
            if src_dims[i] == dst_dims[pad + i] { strides[pad + i] = s; }
            s *= src_dims[i];
        }
        let src_layout = Layout::new(target.clone(), strides, 0);
        let mut dst = self.backend.alloc_zeros(target, self.backend.storage_dtype(&a.storage)).expect("broadcast alloc");
        self.backend.copy_strided_src(&a.storage, &mut dst, 0, &src_layout).expect("broadcast copy");
        TrackedTensor::new(dst, target.clone())
    }

    fn do_concat(
        &self, dim: usize,
        inputs: &[NodeId], out_shape: &Shape,
        cache: &HashMap<NodeId, CacheEntry<B::Storage>>,
    ) -> TrackedTensor<B::Storage> {
        let _s = debug_span!("concat", dim, elems = out_shape.elem_count()).entered();
        let a = self.get_gt(inputs, 0, cache);
        let b = self.get_gt(inputs, 1, cache);

        // Fast path: backend provides a single-dispatch concat.
        if let Ok(s) = self.backend.concat_along_dim(&a.storage, &b.storage, dim, &a.shape, &b.shape) {
            return TrackedTensor::new(s, out_shape.clone());
        }

        let dtype = self.backend.storage_dtype(&a.storage);
        let mut dst = self.backend.alloc_zeros(out_shape, dtype).expect("concat alloc");
        let out_dims = out_shape.dims();
        let a_dim = a.shape.dims()[dim];
        let b_dim = b.shape.dims()[dim];
        let inner: usize = out_dims[dim + 1..].iter().product::<usize>().max(1);
        let outer: usize = out_dims[..dim].iter().product::<usize>().max(1);
        let out_row = out_dims[dim] * inner;
        if outer == 1 {
            self.backend.copy_strided_src(&a.storage, &mut dst, 0, &a.layout()).expect("concat a");
            self.backend.copy_strided_src(&b.storage, &mut dst, a_dim * inner, &b.layout()).expect("concat b");
        } else {
            let a_ss = a_dim * inner;
            let b_ss = b_dim * inner;
            for o in 0..outer {
                let al = Layout::contiguous_with_offset(&Shape::from_dims(&[a_ss]), o * a_ss);
                self.backend.copy_strided_src(&a.storage, &mut dst, o * out_row, &al).expect("concat a");
                let bl = Layout::contiguous_with_offset(&Shape::from_dims(&[b_ss]), o * b_ss);
                self.backend.copy_strided_src(&b.storage, &mut dst, o * out_row + a_ss, &bl).expect("concat b");
            }
        }
        TrackedTensor::new(dst, out_shape.clone())
    }

    fn do_slice(&self, dim: usize, start: usize, a: &TrackedTensor<B::Storage>, out_shape: &Shape) -> TrackedTensor<B::Storage> {
        let in_dims = a.shape.dims();
        let rank = in_dims.len();
        let mut strides: DimVec = DimVec::from_elem(0, rank);
        let mut s = 1usize;
        for i in (0..rank).rev() { strides[i] = s; s *= in_dims[i]; }
        let offset = start * strides[dim];
        let src_layout = Layout::new(out_shape.clone(), strides, offset);
        let dtype = self.backend.storage_dtype(&a.storage);
        let mut dst = self.backend.alloc_zeros(out_shape, dtype).expect("slice alloc");
        self.backend.copy_strided_src(&a.storage, &mut dst, 0, &src_layout).expect("slice copy");
        TrackedTensor::new(dst, out_shape.clone())
    }

    // -- const pool -----------------------------------------------------------

    fn eval_const(&mut self, data: &ConstData, shape: &Shape) -> CacheEntry<B::Storage> {
        let ptr = const_data_arc_ptr(data);
        let refcount = const_data_arc_strong_count(data);
        let elems = data.elem_count();
        if refcount > 1 {
            if self.const_pool.contains_key(&ptr) {
                let _s = debug_span!("const_cache_hit", elems).entered();
                return CacheEntry::ConstRef(ptr);
            }
            let _s = debug_span!("const_upload_persistent", elems).entered();
            let buf = const_data_to_host_buffer(data);
            let storage = self.backend.upload(&buf, shape).expect("const upload");
            self.const_pool.insert(ptr, TrackedTensor::new(storage, shape.clone()));
            return CacheEntry::ConstRef(ptr);
        }
        let _s = debug_span!("const_upload_ephemeral", elems).entered();
        let buf = const_data_to_host_buffer(data);
        let storage = self.backend.upload(&buf, shape).expect("const upload");
        CacheEntry::Owned(TrackedTensor::new(storage, shape.clone()))
    }

    // -- CPU fallback ---------------------------------------------------------

    fn cpu_fallback(
        &self,
        op: &Op,
        inputs: &[NodeId],
        shape: &Shape,
        dtype: DType,
        cache: &HashMap<NodeId, CacheEntry<B::Storage>>,
    ) -> TrackedTensor<B::Storage> {
        let _s = info_span!("cpu_fallback", elems = shape.elem_count()).entered();
        let mut cpu_cache: HashMap<NodeId, AnyRefTensor> = HashMap::new();
        for &id in inputs {
            let gt = self.resolve(cache.get(&id).expect("fallback: missing"));
            let buf = self.backend.download(&gt.storage).expect("D2H fallback");
            cpu_cache.insert(id, host_buffer_to_any_ref(buf, &gt.shape));
        }
        let result = fuel_reference_backend::exec::eval_node_with_op(op, inputs, shape, dtype, &cpu_cache);
        let out_buf = any_ref_to_host_buffer(result);
        let storage = self.backend.upload(&out_buf, shape).expect("H2D fallback");
        TrackedTensor::new(storage, shape.clone())
    }
}

// ---- free-function helpers --------------------------------------------------

fn op_short_name(op: &Op) -> &'static str {
    match op {
        Op::Const(_) => "Const", Op::MatMul => "MatMul",
        Op::Add => "Add", Op::Sub => "Sub", Op::Mul => "Mul", Op::Div => "Div",
        Op::Neg => "Neg", Op::Sqr => "Sqr", Op::Sqrt => "Sqrt",
        Op::Exp => "Exp", Op::Log => "Log",
        Op::Sin => "Sin", Op::Cos => "Cos", Op::Tanh => "Tanh",
        Op::Sigmoid => "Sigmoid", Op::Silu => "Silu", Op::Gelu => "Gelu",
        Op::Relu => "Relu", Op::Step => "Step",
        Op::Maximum => "Maximum", Op::Minimum => "Minimum",
        Op::AddScalar(_) => "AddScalar", Op::MulScalar(_) => "MulScalar",
        Op::PowI(_) => "PowI", Op::Cast(_) => "Cast",
        Op::Reshape(_) => "Reshape", Op::Transpose => "Transpose",
        Op::Permute(_) => "Permute", Op::BroadcastTo(_) => "BroadcastTo",
        Op::SumAll => "SumAll", Op::MeanAll => "MeanAll",
        Op::MaxAll => "MaxAll", Op::MinAll => "MinAll",
        Op::SumDim(_) => "SumDim", Op::MeanDim(_) => "MeanDim",
        Op::MaxDim(_) => "MaxDim", Op::MinDim(_) => "MinDim",
        Op::IndexSelect { .. } => "IndexSelect", Op::Gather { .. } => "Gather",
        Op::Concat { .. } => "Concat", Op::Slice { .. } => "Slice",
        Op::SoftmaxLastDim => "SoftmaxLastDim",
        Op::RmsNormLastDim { .. } => "RmsNormLastDim",
        Op::RmsNormLastDimBackward { .. } => "RmsNormLastDimBackward",
        Op::SoftmaxLastDimBackward => "SoftmaxLastDimBackward",
        Op::LayerNormLastDimBackward { .. } => "LayerNormLastDimBackward",
        Op::Rope => "Rope",
        _ => "Other",
    }
}

fn const_data_arc_ptr(data: &ConstData) -> usize {
    match data {
        ConstData::F32(v) => std::sync::Arc::as_ptr(v) as *const f32 as usize,
        ConstData::F64(v) => std::sync::Arc::as_ptr(v) as *const f64 as usize,
        ConstData::BF16(v) => std::sync::Arc::as_ptr(v) as *const () as usize,
        ConstData::F16(v) => std::sync::Arc::as_ptr(v) as *const () as usize,
        ConstData::U32(v) => std::sync::Arc::as_ptr(v) as *const u32 as usize,
    }
}

fn const_data_arc_strong_count(data: &ConstData) -> usize {
    match data {
        ConstData::F32(v) => std::sync::Arc::strong_count(v),
        ConstData::F64(v) => std::sync::Arc::strong_count(v),
        ConstData::BF16(v) => std::sync::Arc::strong_count(v),
        ConstData::F16(v) => std::sync::Arc::strong_count(v),
        ConstData::U32(v) => std::sync::Arc::strong_count(v),
    }
}

fn const_data_to_host_buffer(data: &ConstData) -> fuel_core_types::HostBuffer {
    use fuel_core_types::HostBuffer;
    match data {
        ConstData::F32(v) => HostBuffer::F32(v.to_vec()),
        ConstData::F64(v) => HostBuffer::F64(v.to_vec()),
        ConstData::BF16(v) => HostBuffer::BF16(v.to_vec()),
        ConstData::F16(v) => HostBuffer::F16(v.to_vec()),
        ConstData::U32(v) => HostBuffer::U32(v.to_vec()),
    }
}

fn host_buffer_to_any_ref(buf: fuel_core_types::HostBuffer, shape: &Shape) -> AnyRefTensor {
    match buf {
        fuel_core_types::HostBuffer::F32(v) => AnyRefTensor::F32(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::F64(v) => AnyRefTensor::F64(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::BF16(v) => AnyRefTensor::BF16(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::F16(v) => AnyRefTensor::F16(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::U32(v) => AnyRefTensor::U32(RefTensor::from_vec(v, shape.clone())),
        _ => panic!("host_buffer_to_any_ref: unsupported dtype"),
    }
}

fn any_ref_to_host_buffer(any: AnyRefTensor) -> fuel_core_types::HostBuffer {
    use fuel_core_types::HostBuffer;
    match any {
        AnyRefTensor::F32(t) => HostBuffer::F32(t.into_vec()),
        AnyRefTensor::F64(t) => HostBuffer::F64(t.into_vec()),
        AnyRefTensor::BF16(t) => HostBuffer::BF16(t.into_vec()),
        AnyRefTensor::F16(t) => HostBuffer::F16(t.into_vec()),
        AnyRefTensor::U32(t) => HostBuffer::U32(t.into_vec()),
    }
}

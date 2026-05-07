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
//! Backend crates (`fuel-graph-cpu`, `fuel-cuda-backend`, future
//! `fuel-graph-metal`) implement `GraphBackend` in ~200 lines each,
//! providing only the device-specific pieces: memory allocation,
//! matmul, unary/binary kernels, reductions, and softmax.


use fuel_core_types::{DType, DimVec, Layout, Shape};
use fuel_graph::{NodeId, Op, Tensor};
use fuel_graph::opt::{execution_plan, RuleRegistry};

/// Merge the graph's side-effect roots into the user's requested roots
/// for the purposes of executing the plan. Deduplicates; preserves the
/// user's roots in order (they come first, then side-effect roots).
fn extend_with_side_effect_roots(
    graph: &fuel_graph::Graph,
    user_roots: &[NodeId],
) -> Vec<NodeId> {
    let side = graph.side_effect_roots();
    if side.is_empty() {
        return user_roots.to_vec();
    }
    let mut out = Vec::with_capacity(user_roots.len() + side.len());
    out.extend_from_slice(user_roots);
    for &s in side {
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}
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
    /// Shared ref into the executor's persistent `const_pool` via an
    /// Arc clone. The Arc ensures the weight storage survives even
    /// if the const_pool evicts the underlying entry mid-walk (which
    /// can happen when the pool is size-bounded). Arc clone is O(1).
    ConstRef(std::sync::Arc<TrackedTensor<S>>),
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

    /// Transfer a storage to the target device. For a single-backend
    /// executor (CPU-only, Vulkan-only, CUDA-only) this is either a
    /// clone (same device) or an error (cross-device — the standalone
    /// backend has no peer to hand off to). The router backend
    /// (`fuel-graph-router`) overrides this with a real host-round-trip
    /// implementation.
    ///
    /// `layout` describes the logical tensor shape; backends that
    /// serialize through a host buffer need it to construct the
    /// HostBuffer.
    fn copy_to(
        &self,
        storage: &Self::Storage,
        layout: &Layout,
        target: fuel_core_types::DeviceLocation,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (storage, layout, target);
        fuel_core_types::bail!(
            "GraphBackend: copy_to not implemented; this backend is single-device"
        )
    }

    // -- compute --

    fn matmul(
        &self,
        a: &Self::Storage, b: &Self::Storage,
        bmnk: (usize, usize, usize, usize),
        la: &Layout, lb: &Layout,
    ) -> fuel_core_types::Result<Self::Storage>;

    /// Quantized matmul for Q4_0-weights: `C = A @ dequant_q4_0(W)`.
    /// `w_q_bytes` holds the raw Q4_0 block byte stream (18 bytes per
    /// 32-element block, stored as U32 for alignment). Logical weight
    /// shape: `[n, k]`. Activation `a`: `[..., m, k]`. Output:
    /// `[..., m, n]` F32.
    ///
    /// One trait method per quant format instead of a single
    /// `qmatmul(QuantType)` dispatcher — kernels are already
    /// specialized per format, and the executor already has
    /// `quant_type` in hand from the `Op::QMatMul` variant, so
    /// collapsing the dispatch there saves one hop.
    ///
    /// Default impl returns `Err`; executor falls back to CPU.
    fn matmul_q4_0(
        &self,
        a: &Self::Storage,
        w_q_bytes: &Self::Storage,
        k: usize,
        n: usize,
        a_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (a, w_q_bytes, k, n, a_layout);
        fuel_core_types::bail!("GraphBackend: matmul_q4_0 not implemented natively")
    }

    /// Quantized matmul for Q4_K_M weights (k-quant family with
    /// per-row scales + mins). Not yet kernel-implemented in any
    /// backend. Default bails.
    fn matmul_q4_km(
        &self,
        a: &Self::Storage,
        w_q_bytes: &Self::Storage,
        k: usize,
        n: usize,
        a_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (a, w_q_bytes, k, n, a_layout);
        fuel_core_types::bail!("GraphBackend: matmul_q4_km not implemented natively")
    }

    /// Quantized matmul for Q8_0 weights (34 bytes per 32 elems, one
    /// f16 scale + 32 signed i8 quants). Default bails — follow-up
    /// kernel work.
    fn matmul_q8_0(
        &self,
        a: &Self::Storage,
        w_q_bytes: &Self::Storage,
        k: usize,
        n: usize,
        a_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (a, w_q_bytes, k, n, a_layout);
        fuel_core_types::bail!("GraphBackend: matmul_q8_0 not implemented natively")
    }

    /// 2-D convolution. `input` has logical shape `[N, C_in, H, W]`,
    /// `weight` has shape `[C_out, C_in/groups, K_h, K_w]`. Returns
    /// just the conv result — bias add (if any) is composed by the
    /// executor as a separate broadcast-add over the c_out axis.
    ///
    /// Default impl bails so backends without native conv2d fall
    /// through to CPU fallback in the executor's `Op::Conv2D` arm.
    /// Backends that implement native conv2d (CUDA, Vulkan, AOCL, MKL)
    /// override this.
    ///
    /// Contract: symmetric stride / padding (`stride.0 == stride.1`
    /// and `padding.0 == padding.1`) — the executor screens this
    /// before calling. `groups` may be any value; backends that
    /// don't support `groups > 1` should `bail!` and the executor
    /// falls back to CPU.
    fn conv2d(
        &self,
        input:        &Self::Storage,
        weight:       &Self::Storage,
        input_layout:  &Layout,
        weight_layout: &Layout,
        stride:  (usize, usize),
        padding: (usize, usize),
        groups:  usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (input, weight, input_layout, weight_layout, stride, padding, groups);
        fuel_core_types::bail!("GraphBackend: conv2d not implemented natively")
    }

    /// Multi-head scaled-dot-product attention (FlashAttention-shaped).
    /// `q` is `[B, Hq, Sq, D]`, `k`/`v` are `[B, Hkv, Sk, D]` with
    /// `Hq` a multiple of `Hkv` (GQA). `alibi_slopes` (if present) is
    /// `[Hq]`. Output is `[B, Hq, Sq, D]`.
    ///
    /// Default impl bails so backends without a native flash-attn
    /// kernel fall through to CPU fallback. The reference backend
    /// uses `attention_naive` for the fallback path.
    #[allow(clippy::too_many_arguments)]
    fn flash_attn(
        &self,
        q: &Self::Storage,
        k: &Self::Storage,
        v: &Self::Storage,
        alibi_slopes: Option<&Self::Storage>,
        q_layout: &Layout,
        k_layout: &Layout,
        v_layout: &Layout,
        alibi_layout: Option<&Layout>,
        softmax_scale: f32,
        causal: bool,
        window_size_left: Option<usize>,
        window_size_right: Option<usize>,
        softcap: Option<f32>,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (q, k, v, alibi_slopes, q_layout, k_layout, v_layout,
                 alibi_layout, softmax_scale, causal, window_size_left,
                 window_size_right, softcap);
        fuel_core_types::bail!("GraphBackend: flash_attn not implemented natively")
    }

    /// Paged-cache scaled-dot-product attention. Inputs (in order):
    /// q, k_cache, v_cache, block_table, context_lens, optional alibi.
    /// Layouts: q `[B, Hq, Sq, D]`, k/v `[num_blocks, block_size, Hkv, D]`,
    /// block_table `[B, max_blocks]` u32, context_lens `[B]` u32.
    /// Output is `[B, Hq, Sq, D]`.
    ///
    /// Default impl bails so backends without a native paged kernel
    /// fall through to CPU fallback.
    #[allow(clippy::too_many_arguments)]
    fn paged_attn(
        &self,
        q: &Self::Storage,
        k_cache: &Self::Storage,
        v_cache: &Self::Storage,
        block_table: &Self::Storage,
        context_lens: &Self::Storage,
        alibi_slopes: Option<&Self::Storage>,
        q_layout: &Layout,
        k_cache_layout: &Layout,
        v_cache_layout: &Layout,
        block_table_layout: &Layout,
        context_lens_layout: &Layout,
        alibi_layout: Option<&Layout>,
        softmax_scale: f32,
        block_size: usize,
        softcap: Option<f32>,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (q, k_cache, v_cache, block_table, context_lens, alibi_slopes,
                 q_layout, k_cache_layout, v_cache_layout, block_table_layout,
                 context_lens_layout, alibi_layout, softmax_scale, block_size, softcap);
        fuel_core_types::bail!("GraphBackend: paged_attn not implemented natively")
    }

    /// 2-D transposed convolution. `input` is `[N, Cin, H, W]`,
    /// `weight` is `[Cin, Cout/groups, Kh, Kw]`. Returns the bare
    /// transposed-conv result; bias compose is done by callers.
    ///
    /// Default impl bails so backends without native support fall
    /// through to CPU fallback in the executor's `Op::ConvTranspose2D`
    /// arm.
    fn conv_transpose2d(
        &self,
        input:         &Self::Storage,
        weight:        &Self::Storage,
        input_layout:  &Layout,
        weight_layout: &Layout,
        stride:         (usize, usize),
        padding:        (usize, usize),
        output_padding: (usize, usize),
        dilation:       (usize, usize),
        groups:         usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (input, weight, input_layout, weight_layout,
                 stride, padding, output_padding, dilation, groups);
        fuel_core_types::bail!("GraphBackend: conv_transpose2d not implemented natively")
    }

    /// Quantize an F32 storage buffer to GGML Q8_0 blocks (34 bytes per
    /// 32 elements). Returns a U32-typed storage holding the raw block
    /// byte stream. Used for KV-cache compression between decode steps.
    ///
    /// Default impl returns Err. Backends that support on-device Q8
    /// quantization override this.
    fn quantize_q8_0(
        &self,
        src_f32: &Self::Storage,
        n_elements: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (src_f32, n_elements);
        fuel_core_types::bail!("GraphBackend: quantize_q8_0 not implemented natively")
    }

    /// Dequantize a U32-typed Q8_0 block stream to an F32 storage buffer.
    /// `n_blocks` is the number of Q8_0 blocks (each 34 bytes / 32 elems).
    fn dequantize_q8_0(
        &self,
        blocks: &Self::Storage,
        n_blocks: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (blocks, n_blocks);
        fuel_core_types::bail!("GraphBackend: dequantize_q8_0 not implemented natively")
    }

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

    /// Concatenate `a` and `b` along `dim` in a single dispatch. Inputs
    /// may be strided (lazy permute views, lazy broadcast) — per-operand
    /// layouts carry both shape and strides. Output has shape `a.shape`
    /// with `a.shape[dim] + b.shape[dim]` at `dim` and is always
    /// contiguous.
    ///
    /// Default impl returns `Err`; the executor falls back to the
    /// `outer × 2` strided-copy loop. Backends override this when a
    /// single-dispatch kernel exists.
    fn concat_along_dim(
        &self,
        a: &Self::Storage,
        b: &Self::Storage,
        dim: usize,
        a_layout: &Layout,
        b_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        let _ = (a, b, dim, a_layout, b_layout);
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
/// Internal entry in the executor's size-bounded const pool.
struct ConstPoolEntry<S> {
    /// Arc so walk-time CacheEntry::ConstRef clones survive pool
    /// eviction: the pool drops its Arc, but outstanding clones keep
    /// the storage alive until the walk completes.
    tensor: std::sync::Arc<TrackedTensor<S>>,
    bytes: usize,
    /// Monotonic access-time counter. Lower = older. `get` / `insert`
    /// bump it; eviction picks the smallest values.
    last_access: u64,
    /// Phase 7.5 G2 step 3c: weak reference to the slot Arc this
    /// entry was uploaded from. Used as a liveness witness — if the
    /// Weak fails to upgrade, the slot Arc has been dropped and any
    /// new Arc allocated at the same pointer is unrelated. The cache
    /// treats that as a miss.
    slot_witness: std::sync::Weak<std::sync::RwLock<fuel_core_types::Storage>>,
}

/// Size-bounded LRU cache for slot-populating const tensors, keyed on
/// the slot Arc pointer with a `Weak<RwLock<Storage>>` liveness
/// witness. When a new entry would push total bytes over `max_bytes`,
/// older entries are evicted — evicting just drops the device-side
/// Arc<Storage>, reclaiming VRAM. The next access re-uploads from
/// the slot Arc.
///
/// Tiering mechanism for weights specifically: the canonical weight
/// bytes live in the slot Arc (owned by the Graph); evicting pushes
/// the weight back to that natural tier. For computed tensors
/// without a host canonical copy (KV cache, activations), use the
/// ResidencyFile-based evict/fault_back flow on VulkanBackend
/// instead.
pub(crate) struct ConstPool<S> {
    entries: HashMap<usize, ConstPoolEntry<S>>,
    total_bytes: usize,
    max_bytes: Option<usize>,
    access_counter: u64,
}

impl<S> ConstPool<S> {
    fn new() -> Self {
        Self { entries: HashMap::new(), total_bytes: 0, max_bytes: None, access_counter: 0 }
    }

    fn set_max_bytes(&mut self, max: Option<usize>) {
        self.max_bytes = max;
        self.evict_to_fit(0);
    }

    pub(crate) fn len(&self) -> usize { self.entries.len() }

    pub(crate) fn total_bytes(&self) -> usize { self.total_bytes }

    /// Look up the cached upload for `slot_arc`. Returns `Some` only
    /// when a cached entry exists for the slot's pointer AND the
    /// stored Weak still upgrades to the same Arc — i.e. the slot
    /// hasn't been dropped and recycled to a different Arc at the
    /// same address.
    fn get_for_slot(
        &mut self,
        slot_arc: &std::sync::Arc<std::sync::RwLock<fuel_core_types::Storage>>,
    ) -> Option<std::sync::Arc<TrackedTensor<S>>> {
        let key = std::sync::Arc::as_ptr(slot_arc) as *const () as usize;
        self.access_counter += 1;
        let c = self.access_counter;
        // Check witness first: if the slot Arc this entry was made
        // for has dropped, the entry is stale (raw ptr may have been
        // recycled to an unrelated Arc).
        let is_alive = self.entries.get(&key).map(|e| {
            e.slot_witness.upgrade().is_some_and(|alive_arc| {
                std::sync::Arc::ptr_eq(&alive_arc, slot_arc)
            })
        }).unwrap_or(false);
        if !is_alive {
            // Purge any stale entry under this key.
            if let Some(entry) = self.entries.remove(&key) {
                self.total_bytes -= entry.bytes;
            }
            return None;
        }
        self.entries.get_mut(&key).map(|e| {
            e.last_access = c;
            std::sync::Arc::clone(&e.tensor)
        })
    }

    /// Insert a new entry keyed on `slot_arc`'s pointer with a
    /// liveness witness. Evicts LRU entries to stay under `max_bytes`.
    fn insert_for_slot(
        &mut self,
        slot_arc: &std::sync::Arc<std::sync::RwLock<fuel_core_types::Storage>>,
        tensor: TrackedTensor<S>,
        bytes: usize,
    ) -> std::sync::Arc<TrackedTensor<S>> {
        let key = std::sync::Arc::as_ptr(slot_arc) as *const () as usize;
        self.evict_to_fit(bytes);
        self.access_counter += 1;
        let last_access = self.access_counter;
        let arc_tensor = std::sync::Arc::new(tensor);
        let entry = ConstPoolEntry {
            tensor: std::sync::Arc::clone(&arc_tensor),
            bytes,
            last_access,
            slot_witness: std::sync::Arc::downgrade(slot_arc),
        };
        if let Some(existing) = self.entries.insert(key, entry) {
            self.total_bytes -= existing.bytes;
        }
        self.total_bytes += bytes;
        arc_tensor
    }

    /// Evict LRU entries until `total_bytes + incoming <= max_bytes`.
    /// No-op if `max_bytes` is None. Returns the number of entries evicted.
    fn evict_to_fit(&mut self, incoming: usize) -> usize {
        let Some(max) = self.max_bytes else { return 0 };
        let mut evicted = 0;
        while self.total_bytes + incoming > max && !self.entries.is_empty() {
            // O(n) LRU pick — n is tens to hundreds of cached weights,
            // well within acceptable for the eviction path.
            let Some((&lru_key, _)) = self.entries.iter()
                .min_by_key(|(_, e)| e.last_access)
            else { break };
            if let Some(entry) = self.entries.remove(&lru_key) {
                self.total_bytes -= entry.bytes;
                evicted += 1;
            }
        }
        evicted
    }
}

pub struct GraphExecutor<B: GraphBackend> {
    pub backend: B,
    /// Size-bounded LRU cache for weight constants. Only caches
    /// consts with Arc::strong_count > 1 (model weights — tensors
    /// held somewhere outside the graph, so re-upload on eviction
    /// is a free Arc-clone of the host data).
    const_pool: ConstPool<B::Storage>,
    /// Pre-populated entries for the next realize call.
    injected: HashMap<NodeId, TrackedTensor<B::Storage>>,
    /// If true, realize runs the backend-agnostic `fuel_graph::opt`
    /// pass (CSE + algebraic simplification) on the graph before
    /// walking it. Off by default because it mutates the shared graph
    /// arena (appends canonical nodes), which existing test code may
    /// not expect; opt-in per-executor via `with_optimization(true)`.
    ///
    /// When enabled, the rule-registry pipeline (lowering + fusion,
    /// see [`fuel_graph::opt::RuleRegistry`]) runs *before* CSE: rules
    /// rewrite the graph first, then CSE canonicalizes whatever the
    /// rules produced. The shipped `RuleRegistry::default_rules()` is
    /// inverse-pair only (SoftmaxLastDim lower + fuse), so the
    /// post-rules graph round-trips to the input for canonical
    /// SoftmaxLastDim graphs. Other registries (e.g.
    /// `RuleRegistry::lowering_only()`) leave a different post-rules
    /// state and are configurable via [`with_rule_registry`](Self::with_rule_registry).
    optimize: bool,
    /// Rule-registry consulted when `optimize` is enabled. Default is
    /// [`RuleRegistry::default_rules`]. Tests that want to force the
    /// lowered form (e.g. live CUDA equivalence checks) replace this
    /// with [`RuleRegistry::lowering_only`].
    rule_registry: RuleRegistry,
    /// Phase-1 placement: the device this executor represents. When
    /// set, `validate_placements` checks every graph node's optional
    /// placement hint matches this device. Phase 2 will turn this into
    /// a `Vec<BackendInstance>` for true multi-device dispatch.
    pub default_device: Option<fuel_core_types::DeviceLocation>,
}

impl<B: GraphBackend> GraphExecutor<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            const_pool: ConstPool::new(),
            injected: HashMap::new(),
            optimize: false,
            rule_registry: RuleRegistry::default_rules(),
            default_device: None,
        }
    }

    /// Tag this executor with the device it represents. Used by
    /// `validate_placements` to check that graph-level placement hints
    /// are consistent with what the executor can actually run.
    pub fn with_default_device(mut self, loc: fuel_core_types::DeviceLocation) -> Self {
        self.default_device = Some(loc);
        self
    }

    /// Walk every node reachable from `roots` and verify that any
    /// set placement hint matches this executor's `default_device`.
    /// Nodes with no placement hint are skipped (they inherit the
    /// default). Returns `Err` on the first mismatch.
    ///
    /// Phase 1 only surfaces the API; callers invoke it explicitly
    /// before `realize*`. Phase 2 will fold this into the dispatch
    /// path and use per-node placement to route work between backends.
    pub fn validate_placements(&self, roots: &[&Tensor]) -> fuel_core_types::Result<()> {
        let Some(my_dev) = self.default_device else {
            return Ok(());
        };
        if roots.is_empty() {
            return Ok(());
        }
        let graph = roots[0].graph().read().unwrap();
        let root_ids: Vec<NodeId> = roots.iter().map(|t| t.id()).collect();
        for id in fuel_graph::topo_order_multi(&graph, &root_ids) {
            if let Some(placement) = graph.placement(id) {
                if placement != my_dev {
                    fuel_core_types::bail!(
                        "validate_placements: node {:?} requests {:?} but executor is on {:?}",
                        id, placement, my_dev
                    );
                }
            }
        }
        Ok(())
    }

    /// Enable or disable graph-level optimization (rule-registry
    /// pipeline + CSE + algebraic simplification) before each realize.
    /// Pre-populated / injected nodes are preserved — they're leaves
    /// from the optimizer's view and can't be eliminated.
    pub fn with_optimization(mut self, enabled: bool) -> Self {
        self.optimize = enabled;
        self
    }

    /// Replace the rule-registry consulted when optimization is
    /// enabled. Default is [`RuleRegistry::default_rules`].
    ///
    /// Use [`RuleRegistry::lowering_only`] in tests that want the
    /// lowered form to reach the executor (so the composed-math
    /// path is exercised end-to-end). Use [`RuleRegistry::new`] to
    /// disable just the rules while keeping CSE active.
    pub fn with_rule_registry(mut self, registry: RuleRegistry) -> Self {
        self.rule_registry = registry;
        self
    }

    /// Set a maximum byte budget for the weight cache (`const_pool`).
    /// When adding a weight would exceed the limit, the least-recently-
    /// used cached weight is evicted to free space. Evicting just
    /// drops the device Arc — subsequent access re-uploads from the
    /// original host `Arc<[T]>`.
    ///
    /// Useful for running models larger than VRAM: configure the
    /// limit to some fraction of available VRAM, leaving headroom
    /// for activations, KV cache, and working-set buffers. Pair with
    /// [`KVCache::park`] for idle KV eviction.
    ///
    /// `None` disables the limit (today's default — unbounded
    /// accumulation, matches pre-tiering behavior).
    pub fn with_const_pool_limit(mut self, max_bytes: Option<usize>) -> Self {
        self.const_pool.set_max_bytes(max_bytes);
        self
    }

    /// Current weight-cache size in bytes. Exposed for tests and
    /// observability.
    pub fn const_pool_bytes(&self) -> usize { self.const_pool.total_bytes() }

    /// Current number of cached weight entries. Exposed for tests.
    pub fn const_pool_entries(&self) -> usize { self.const_pool.len() }

    /// Pre-populate a node with an existing device-side tensor.
    pub fn pre_populate(&mut self, node_id: NodeId, storage: B::Storage, shape: Shape) {
        self.injected.insert(node_id, TrackedTensor::new(storage, shape));
    }

    /// Per-node eval wrapper that catches panics and re-panics with
    /// graph-location context prepended. Mirrors what
    /// `fuel_reference_backend::exec::eval_node_with_graph_context`
    /// does for the reference backend, so realize-time panics from
    /// either backend tell you "Node#1734 (Conv2D, ...)" instead of
    /// just "shape mismatch".
    fn eval_node_with_graph_context(
        &mut self,
        graph: &fuel_graph::Graph,
        id: NodeId,
        op: &Op,
        inputs: &[NodeId],
        shape: &Shape,
        dtype: fuel_core_types::DType,
        cache: &std::collections::HashMap<NodeId, CacheEntry<B::Storage>>,
    ) -> CacheEntry<B::Storage> {
        use std::panic::{catch_unwind, AssertUnwindSafe, resume_unwind};
        let result = catch_unwind(AssertUnwindSafe(|| {
            self.eval_node(op, inputs, shape, dtype, cache)
        }));
        match result {
            Ok(t) => t,
            Err(payload) => {
                let original = panic_payload_to_string(&payload);
                let location = graph.describe_node(id);
                let msg = format!(
                    "fuel-graph-executor realize: panic at {location}\n  original panic: {original}"
                );
                resume_unwind(Box::new(msg))
            }
        }
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
        // Phase 7.5 PR 3: rule-registry pipeline runs first (lowering
        // → fusion to fixpoint), then CSE + algebraic simplification
        // canonicalizes whatever the rules produced. Both phases share
        // the `optimize` flag so callers opt in once for the whole
        // pre-realize transform pipeline.
        let after_rules = self.rule_registry.optimize_to_fixpoint(graph, &original);
        fuel_graph::opt::optimize(graph, &after_rules)
    }

    /// Phase 7.5 G2: slot-first dispatch. If the graph's storage_map
    /// has a populated slot for `id`, adopt it directly into the
    /// executor's CacheEntry world (host-roundtrip + upload to the
    /// backend's target device, with const_pool keyed on the slot
    /// Arc's pointer identity for reuse across realize calls).
    ///
    /// Returns `Some(entry)` when the slot is consumed; `None` when the
    /// slot is empty and the caller should fall through to eval_node.
    ///
    /// Today the host roundtrip happens unconditionally — even when
    /// the slot's Storage already lives on the backend's target device.
    /// A future fast-path can downcast and try a same-device adopt;
    /// the const_pool keyed on slot-Arc identity already amortises the
    /// roundtrip across repeated realize calls.
    fn try_adopt_slot(
        &mut self,
        graph: &fuel_graph::Graph,
        id: NodeId,
        shape: &Shape,
    ) -> Option<CacheEntry<B::Storage>> {
        let slot_arc = graph.storage_for(id)?;
        // Phase 7.5 G2 step 3c: liveness-aware const_pool lookup.
        // The pool keys on slot Arc pointer with a Weak<RwLock<Storage>>
        // witness — same pointer + still-upgrading witness = safe hit.
        // If the witness fails to upgrade (slot Arc dropped, allocator
        // may have recycled the address), the entry is purged.
        if let Some(arc) = self.const_pool.get_for_slot(&slot_arc) {
            return Some(CacheEntry::ConstRef(arc));
        }
        let (buf, dtype) = {
            let slot = slot_arc.read().unwrap();
            let buf = slot.as_dyn().to_host_buffer_dyn().expect("slot D2H");
            let dtype = slot.dtype();
            (buf, dtype)
        };
        let storage = self.backend.upload(&buf, shape).expect("slot upload");
        let bytes = shape.elem_count() * dtype.size_in_bytes();
        let arc = self.const_pool.insert_for_slot(
            &slot_arc, TrackedTensor::new(storage, shape.clone()), bytes,
        );
        Some(CacheEntry::ConstRef(arc))
    }

    // -- realize entry points -------------------------------------------------

    pub fn realize_f32(&mut self, tensor: &Tensor) -> RefTensor<f32> {
        let _span = info_span!("realize_f32").entered();
        let root_id = self.resolve_roots(&[tensor])[0];
        let graph = tensor.graph().read().unwrap();
        let effective_roots = extend_with_side_effect_roots(&graph, &[root_id]);
        let order = execution_plan(&graph, &effective_roots);
        let _walk = info_span!("topo_walk", nodes = order.len()).entered();
        let mut cache = self.drain_injected();
        for id in order {
            if cache.contains_key(&id) { continue; }
            let node = graph.node(id);
            // Phase 7.5 G2: slot-first dispatch — adopt graph-owned
            // storage directly when present, skipping eval_node.
            if let Some(entry) = self.try_adopt_slot(&graph, id, &node.shape) {
                cache.insert(id, entry);
                continue;
            }
            let entry = self.eval_node_with_graph_context(&graph, id, &node.op, &node.inputs, &node.shape, node.dtype, &cache);
            cache.insert(id, entry);
            // If this op destroyed an input (Op::Release et al.), drop
            // the input from cache — derive_ordering guaranteed every
            // non-destructive reader of it has already run.
            if let Some(d_idx) = node.op.destructive_input() {
                if let Some(&destroyed) = node.inputs.get(d_idx) {
                    cache.remove(&destroyed);
                }
            }
        }
        drop(_walk);
        let _rb = info_span!("d2h_readback").entered();
        let gt = self.take_owned(cache.remove(&root_id).expect("realize: missing root"));
        // Materialize lazy views so the buffer order matches the logical shape.
        let gt = self.materialize_if_needed(&gt);
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
        let graph = graph_rc.read().unwrap();
        let effective_roots = extend_with_side_effect_roots(&graph, &roots);
        let order = execution_plan(&graph, &effective_roots);
        let _walk = info_span!("topo_walk", nodes = order.len()).entered();
        let mut cache = self.drain_injected();
        let root_set: std::collections::HashSet<NodeId> = roots.iter().copied().collect();
        for id in order {
            if cache.contains_key(&id) { continue; }
            let node = graph.node(id);
            // Phase 7.5 G2: slot-first dispatch.
            if let Some(entry) = self.try_adopt_slot(&graph, id, &node.shape) {
                cache.insert(id, entry);
                continue;
            }
            let entry = self.eval_node_with_graph_context(&graph, id, &node.op, &node.inputs, &node.shape, node.dtype, &cache);
            cache.insert(id, entry);
            // Drop destroyed input from cache once a destructive op runs —
            // ordering guarantees no downstream reader still needs it.
            // Exception: don't drop requested output roots.
            if let Some(d_idx) = node.op.destructive_input() {
                if let Some(&destroyed) = node.inputs.get(d_idx) {
                    if !root_set.contains(&destroyed) {
                        cache.remove(&destroyed);
                    }
                }
            }
        }
        drop(_walk);
        let _rb = info_span!("d2h_readback").entered();
        roots.iter().map(|id| {
            let gt = self.materialize_if_needed(
                self.resolve(cache.get(id).expect("realize_many: missing")));
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
        let graph = graph_rc.read().unwrap();
        let effective_roots = extend_with_side_effect_roots(&graph, &roots);
        let order = execution_plan(&graph, &effective_roots);
        let _walk = info_span!("topo_walk", nodes = order.len()).entered();
        let mut cache = self.drain_injected();
        let root_set: std::collections::HashSet<NodeId> = roots.iter().copied().collect();
        for id in order {
            if cache.contains_key(&id) { continue; }
            let node = graph.node(id);
            // Phase 7.5 G2: slot-first dispatch.
            if let Some(entry) = self.try_adopt_slot(&graph, id, &node.shape) {
                cache.insert(id, entry);
                continue;
            }
            let entry = self.eval_node_with_graph_context(&graph, id, &node.op, &node.inputs, &node.shape, node.dtype, &cache);
            cache.insert(id, entry);
            if let Some(d_idx) = node.op.destructive_input() {
                if let Some(&destroyed) = node.inputs.get(d_idx) {
                    if !root_set.contains(&destroyed) {
                        cache.remove(&destroyed);
                    }
                }
            }
        }
        drop(_walk);
        let _rb = info_span!("d2h_readback", n_d2h).entered();
        let mut cpu_out = Vec::with_capacity(n_d2h);
        let mut gpu_out = Vec::with_capacity(roots.len() - n_d2h);
        for (i, id) in roots.iter().enumerate() {
            if i < n_d2h {
                // Materialize lazy views before download so the
                // buffer order matches the logical shape.
                let gt = self.materialize_if_needed(
                    self.resolve(cache.get(id).expect("split: missing")));
                let buf = self.backend.download(&gt.storage).expect("D2H");
                match buf {
                    fuel_core_types::HostBuffer::F32(v) => cpu_out.push(v),
                    other => panic!("split: got {:?}", other.dtype()),
                }
            } else {
                // Materialize lazy views so the returned storage is
                // contiguous in the logical shape order.
                let gt = match cache.remove(id) {
                    Some(entry) => {
                        let resolved = match &entry {
                            CacheEntry::ConstRef(arc) => arc.as_ref(),
                            CacheEntry::Owned(gt) => gt,
                        };
                        self.materialize_if_needed(resolved)
                    }
                    None => panic!("split: missing root"),
                };
                let shape = gt.shape.clone();
                let s = std::sync::Arc::try_unwrap(gt.storage)
                    .unwrap_or_else(|arc| {
                        let layout = Layout::contiguous(&shape);
                        self.backend.try_clone(&arc, &layout).expect("split clone")
                    });
                gpu_out.push((s, shape));
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
            CacheEntry::ConstRef(arc) => arc.as_ref(),
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

    /// Like `get_gt` but materializes non-contiguous views into fresh
    /// contiguous buffers. Used by ops that assume contiguous storage
    /// (everything except matmul).
    fn get_gt_c(
        &self,
        inputs: &[NodeId],
        idx: usize,
        cache: &HashMap<NodeId, CacheEntry<B::Storage>>,
    ) -> TrackedTensor<B::Storage> {
        self.materialize_if_needed(self.get_gt(inputs, idx, cache))
    }

    /// If the tensor has a non-contiguous custom_layout (e.g. from a
    /// metadata-only permute), materialize it into a fresh contiguous
    /// buffer. Otherwise return a cheap Arc-clone of the original.
    /// Used as a gate before ops that assume contiguous storage.
    fn materialize_if_needed(&self, gt: &TrackedTensor<B::Storage>) -> TrackedTensor<B::Storage> {
        if gt.custom_layout.is_none() {
            TrackedTensor {
                storage: std::sync::Arc::clone(&gt.storage),
                shape: gt.shape.clone(),
                custom_layout: None,
            }
        } else {
            let layout = gt.layout();
            let dtype = self.backend.storage_dtype(&gt.storage);
            let mut dst = self.backend.alloc_zeros(&gt.shape, dtype).expect("materialize");
            self.backend.copy_strided_src(&gt.storage, &mut dst, 0, &layout).expect("materialize copy");
            TrackedTensor::new(dst, gt.shape.clone())
        }
    }

    fn take_owned(&self, entry: CacheEntry<B::Storage>) -> TrackedTensor<B::Storage> {
        match entry {
            CacheEntry::Owned(gt) => gt,
            CacheEntry::ConstRef(arc) => {
                let p = arc.as_ref();
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
            // Op::Const is intercepted by slot-first dispatch
            // (try_adopt_slot) in the realize loops above. Reaching
            // eval_node with a Const node means a constructor failed
            // to slot-populate — a bug.
            Op::Const => unreachable!(
                "fuel-graph-executor eval_node: Op::Const must be \
                 handled by slot-first dispatch in the realize loop, \
                 never reach eval_node",
            ),

            // -- matmul (stride-aware — reads A/B via per-dim strides,
            //    permuted/transposed views work without materialization) --
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

            // -- quantized matmul: C = A @ dequant(W_Q) --
            // Dispatch flat-per-quant-type so the backend doesn't do
            // a second match on quant_type; we have it in hand on the
            // op variant and there's no reason to nest dispatches.
            Op::QMatMul { quant_type, k, n } => {
                let a = self.get_gt(inputs, 0, cache);
                // Weight bytes are a const U32 buffer — always contiguous.
                let w = self.get_gt_c(inputs, 1, cache);
                let result = match quant_type {
                    fuel_graph::QuantType::Q4_0 =>
                        self.backend.matmul_q4_0(&a.storage, &w.storage, *k, *n, &a.layout()),
                    fuel_graph::QuantType::Q8_0 =>
                        self.backend.matmul_q8_0(&a.storage, &w.storage, *k, *n, &a.layout()),
                    fuel_graph::QuantType::Q4_K_M =>
                        self.backend.matmul_q4_km(&a.storage, &w.storage, *k, *n, &a.layout()),
                    // Legacy executor only wires the three quant types
                    // it has trait methods for. New variants
                    // (Q4_1/Q5_0/Q5_1/Q8_1/Q2K/Q3K/Q5K/Q6K) flow only
                    // through the pipelined executor for now.
                    _ => Err(fuel_core_types::Error::Msg(format!(
                        "legacy executor: QMatMul {:?} not wired (use pipelined executor)",
                        quant_type
                    ))),
                };
                match result {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- 2-D convolution --
            //
            // Inputs: [input, weight, optional bias]. The trait method
            // takes only input + weight (the conv proper); bias-add
            // is composed as a separate broadcast-add over the c_out
            // axis using the existing binary infrastructure. Backends
            // that don't implement native conv2d return Err from the
            // default trait impl and we fall through to cpu_fallback.
            //
            // CPU fallback is used for asymmetric stride/padding (no
            // backend handles those today) and for any case the backend
            // can't take — including groups != 1 on backends that don't
            // implement grouped conv (Vulkan, im2col-fallback CUDA).
            // CUDA-cuDNN, AOCL, and MKL all handle groups > 1 natively
            // and the executor lets them try.
            Op::Conv2D { stride, padding, groups } => {
                let input  = self.get_gt_c(inputs, 0, cache);
                let weight = self.get_gt_c(inputs, 1, cache);
                let symmetric = stride.0 == stride.1 && padding.0 == padding.1;
                if !symmetric {
                    return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache));
                }
                let conv_storage = match self.backend.conv2d(
                    &input.storage, &weight.storage,
                    &input.layout(), &weight.layout(),
                    *stride, *padding, *groups,
                ) {
                    Ok(s) => s,
                    Err(_) => {
                        return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache));
                    }
                };
                if inputs.len() < 3 {
                    // No bias — return the raw conv output.
                    conv_storage
                } else {
                    // Bias has shape [c_out]; reshape it as
                    // [1, c_out, 1, 1] then broadcast to the conv
                    // output shape, producing stride [0, 1, 0, 0].
                    // The CUDA binary kernel handles strided inputs
                    // including stride-0 broadcasts.
                    let bias = self.get_gt_c(inputs, 2, cache);
                    let dims = shape.dims();
                    let c_out = dims[1];
                    let bias_layout_4d = Layout::contiguous(Shape::from_dims(&[1, c_out, 1, 1]));
                    let bias_layout = match bias_layout_4d.broadcast_as(shape.clone()) {
                        Ok(l) => l,
                        Err(_) => {
                            return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache));
                        }
                    };
                    match self.backend.binary(
                        BinaryOp::Add,
                        &conv_storage, &bias.storage,
                        &Layout::contiguous(shape),
                        &bias_layout,
                    ) {
                        Ok(s) => s,
                        Err(_) => {
                            // Bias broadcast not supported by this
                            // backend's binary path — fall back to CPU
                            // for the whole conv2d.
                            return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache));
                        }
                    }
                }
            }

            // -- transposed convolution --
            //
            // Inputs: [input, weight]. Used for upsamplers and as the
            // dX half of Conv2D's gradient rule. Asymmetric stride /
            // padding / dilation is forwarded to the backend as-is —
            // unlike Conv2D, no symmetry pre-screen, since the
            // backward path produces non-square cases naturally.
            Op::ConvTranspose2D { stride, padding, output_padding, dilation, groups } => {
                let input  = self.get_gt_c(inputs, 0, cache);
                let weight = self.get_gt_c(inputs, 1, cache);
                match self.backend.conv_transpose2d(
                    &input.storage, &weight.storage,
                    &input.layout(), &weight.layout(),
                    *stride, *padding, *output_padding, *dilation, *groups,
                ) {
                    Ok(s) => s,
                    Err(_) => {
                        return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache));
                    }
                }
            }

            // -- multi-head attention --
            //
            // Inputs: [q, k, v, optional alibi_slopes]. Backends without
            // a native flash-attn kernel return Err from the default
            // trait impl; the executor catches it and falls back to
            // attention_naive via cpu_fallback.
            Op::FlashAttn { softmax_scale, causal, window_size_left, window_size_right, softcap } => {
                let q = self.get_gt_c(inputs, 0, cache);
                let k = self.get_gt_c(inputs, 1, cache);
                let v = self.get_gt_c(inputs, 2, cache);
                let alibi = if inputs.len() >= 4 {
                    Some(self.get_gt_c(inputs, 3, cache))
                } else {
                    None
                };
                let alibi_layout = alibi.as_ref().map(|t| t.layout());
                match self.backend.flash_attn(
                    &q.storage, &k.storage, &v.storage,
                    alibi.as_ref().map(|t| t.storage.as_ref()),
                    &q.layout(), &k.layout(), &v.layout(),
                    alibi_layout.as_ref(),
                    *softmax_scale, *causal, *window_size_left, *window_size_right, *softcap,
                ) {
                    Ok(s) => s,
                    Err(_) => {
                        return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache));
                    }
                }
            }

            Op::FusedLinear => {
                // Currently dispatches as backend.matmul + backend.binary(Add).
                // Backends with a true fused kernel (cuBLAS gemm-with-bias-
                // epilogue, hand-written Slang) can opt in by adding an
                // override later. Same numerical output either way; this
                // arm exists so the IR pattern survives optimization
                // passes intact.
                let a = self.get_gt(inputs, 0, cache);
                let b = self.get_gt(inputs, 1, cache);
                let bias = self.get_gt_c(inputs, 2, cache);
                let ad = a.shape.dims();
                let bd = b.shape.dims();
                let rank = ad.len();
                let (m, k, n) = (ad[rank - 2], ad[rank - 1], bd[rank - 1]);
                let batch: usize = ad[..rank - 2].iter().product::<usize>().max(1);
                match (|| -> fuel_core_types::Result<B::Storage> {
                    let mm = self.backend.matmul(
                        &a.storage, &b.storage, (batch, m, n, k),
                        &a.layout(), &b.layout(),
                    )?;
                    let mm_layout = Layout::contiguous(shape.clone());
                    // Bias is rank-1 [N]; broadcast to mm's [..., M, N] by
                    // reshaping to [1...1, 1, N] and broadcasting to the
                    // matmul output shape.
                    let mut leading: Vec<usize> = vec![1; shape.dims().len() - 1];
                    leading.push(shape.dims()[shape.dims().len() - 1]);
                    let bias_4d = Layout::contiguous(Shape::from_dims(&leading));
                    let bias_layout = bias_4d.broadcast_as(shape.clone()).map_err(|e| {
                        fuel_core_types::Error::Msg(format!("FusedLinear bias broadcast: {e}"))
                    })?;
                    self.backend.binary(
                        BinaryOp::Add,
                        &mm, &bias.storage,
                        &mm_layout, &bias_layout,
                    )
                })() {
                    Ok(s) => s,
                    Err(_) => {
                        return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache));
                    }
                }
            }

            Op::PagedAttn { softmax_scale, block_size, softcap } => {
                let q  = self.get_gt_c(inputs, 0, cache);
                let kc = self.get_gt_c(inputs, 1, cache);
                let vc = self.get_gt_c(inputs, 2, cache);
                let bt = self.get_gt_c(inputs, 3, cache);
                let cl = self.get_gt_c(inputs, 4, cache);
                let alibi = if inputs.len() >= 6 {
                    Some(self.get_gt_c(inputs, 5, cache))
                } else {
                    None
                };
                let alibi_layout = alibi.as_ref().map(|t| t.layout());
                match self.backend.paged_attn(
                    &q.storage, &kc.storage, &vc.storage, &bt.storage, &cl.storage,
                    alibi.as_ref().map(|t| t.storage.as_ref()),
                    &q.layout(), &kc.layout(), &vc.layout(),
                    &bt.layout(), &cl.layout(),
                    alibi_layout.as_ref(),
                    *softmax_scale, *block_size, *softcap,
                ) {
                    Ok(s) => s,
                    Err(_) => {
                        return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache));
                    }
                }
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
                let a = self.get_gt_c(inputs, 0, cache);
                self.backend.affine(&a.storage, &a.layout(), 1.0, *c).expect("AddScalar")
            }
            Op::MulScalar(c) => {
                let a = self.get_gt_c(inputs, 0, cache);
                self.backend.affine(&a.storage, &a.layout(), *c, 0.0).expect("MulScalar")
            }
            Op::PowI(n) => {
                let a = self.get_gt_c(inputs, 0, cache);
                match self.backend.powf(&a.storage, &a.layout(), *n as f64) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- cast (CPU fallback if backend doesn't implement) --
            Op::Cast(target) => {
                let a = self.get_gt_c(inputs, 0, cache);
                match self.backend.cast(&a.storage, &a.layout(), *target) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- layout ops (SHARED across all backends) --
            Op::Reshape(_) => {
                let a = self.get_gt_c(inputs, 0, cache);
                // Pass output shape in the layout so backends that
                // store shape relabel correctly.
                let target_layout = Layout::contiguous(shape);
                let s = self.backend.try_clone(&a.storage, &target_layout).expect("Reshape");
                return CacheEntry::Owned(TrackedTensor::new(s, shape.clone()));
            }
            Op::Transpose => {
                let a = self.get_gt_c(inputs, 0, cache);
                let rank = a.shape.dims().len();
                let mut perm: Vec<usize> = (0..rank).collect();
                perm.swap(rank - 2, rank - 1);
                return CacheEntry::Owned(self.do_permute(&a, &perm, shape));
            }
            Op::Permute(axes) => {
                let a = self.get_gt_c(inputs, 0, cache);
                return CacheEntry::Owned(self.do_permute(&a, axes, shape));
            }
            Op::BroadcastTo(target) => {
                let a = self.get_gt_c(inputs, 0, cache);
                return CacheEntry::Owned(self.do_broadcast(&a, target));
            }
            Op::Unsqueeze { dim } => {
                // Metadata-only view: insert a size-1 axis at `dim`.
                // We use `get_gt` (not `get_gt_c`) to preserve any
                // upstream strided/transposed/broadcast layout —
                // unsqueeze on a non-contiguous input stays
                // non-contiguous via the layered Layout, so downstream
                // consumers materialize lazily only when they actually
                // need contiguous bytes.
                let a = self.get_gt(inputs, 0, cache);
                let unsqueezed = a.layout().unsqueeze(*dim).expect("unsqueeze layout derive");
                return CacheEntry::Owned(TrackedTensor {
                    storage: std::sync::Arc::clone(&a.storage),
                    shape: shape.clone(),
                    custom_layout: Some(unsqueezed),
                });
            }
            Op::Concat { dim } => {
                return CacheEntry::Owned(self.do_concat(*dim, inputs, shape, cache));
            }
            Op::Slice { dim, start, len: _ } => {
                let a = self.get_gt_c(inputs, 0, cache);
                return CacheEntry::Owned(self.do_slice(*dim, *start, &a, shape));
            }

            // -- reductions --
            Op::SumAll | Op::MeanAll => {
                let a = self.get_gt_c(inputs, 0, cache);
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
                let a = self.get_gt_c(inputs, 0, cache);
                let axes: Vec<usize> = (0..a.shape.dims().len()).collect();
                self.backend.reduce(fuel_core_types::op::ReduceOp::Max, &a.storage, &a.layout(), &axes)
                    .expect("MaxAll")
            }
            Op::MinAll => {
                let a = self.get_gt_c(inputs, 0, cache);
                let axes: Vec<usize> = (0..a.shape.dims().len()).collect();
                self.backend.reduce(fuel_core_types::op::ReduceOp::Min, &a.storage, &a.layout(), &axes)
                    .expect("MinAll")
            }
            Op::SumDim(d) | Op::MeanDim(d) => {
                let a = self.get_gt_c(inputs, 0, cache);
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
                let a = self.get_gt_c(inputs, 0, cache);
                match self.backend.reduce(fuel_core_types::op::ReduceOp::Max, &a.storage, &a.layout(), &[*d]) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // ReduceSumTo / ReduceMaxTo: derive reduce_dims from
            // (input_shape, target_shape), then dispatch through the
            // backend's reduce(). Output rank == target rank; the
            // backend produces rank-shrunk bytes matching the
            // keepdim-or-shrink byte count, and the executor's
            // TrackedTensor wraps with the graph node's actual shape.
            //
            // Edge case: when input_shape already matches padded
            // target on every axis, reduce_dims is empty and the
            // op is an identity. Skipping the backend.reduce call is
            // important because some backends conventionally treat
            // empty-dims as "full reduction"; our ReduceSumTo
            // semantics treat it as "no reduction".
            // ReduceSumTo / ReduceMaxTo: derive reduce_dims from
            // (input_shape, target_shape), dispatch through the
            // backend's reduce(), then relabel the result's internal
            // shape to match the keepdim target. The relabel matters
            // because backend.reduce produces a rank-shrunk storage
            // whose internal shape doesn't match the graph node's
            // target shape; downstream ops that introspect the
            // storage's own shape (rather than TrackedTensor.shape)
            // would see a mismatch otherwise. Mirrors Op::Reshape's
            // try_clone-with-target-layout pattern.
            Op::ReduceSumTo(target_shape) | Op::ReduceMaxTo(target_shape) => {
                let a = self.get_gt_c(inputs, 0, cache);
                let in_dims = a.shape.dims();
                let dst_dims = target_shape.dims();
                if dst_dims.len() > in_dims.len() {
                    return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache));
                }
                let pad = in_dims.len() - dst_dims.len();
                let mut padded = vec![1_usize; pad];
                padded.extend_from_slice(dst_dims);
                let mut reduce_dims: Vec<usize> = Vec::new();
                let mut shape_ok = true;
                for (axis, (&s, &t)) in in_dims.iter().zip(padded.iter()).enumerate() {
                    if t == s {
                    } else if t == 1 {
                        if s > 1 { reduce_dims.push(axis); }
                    } else {
                        shape_ok = false;
                        break;
                    }
                }
                if !shape_ok {
                    return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache));
                }
                if reduce_dims.is_empty() {
                    // Identity: input already matches target. Defer to
                    // CPU fallback for the trivial copy — keeps the
                    // empty-reduce-dims convention out of every
                    // backend's reduce impl.
                    return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache));
                }
                let reduce_op = match op {
                    Op::ReduceSumTo(_) => fuel_core_types::op::ReduceOp::Sum,
                    Op::ReduceMaxTo(_) => fuel_core_types::op::ReduceOp::Max,
                    _ => unreachable!(),
                };
                let reduced = match self.backend.reduce(
                    reduce_op, &a.storage, &a.layout(), &reduce_dims,
                ) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                };
                // Relabel the storage's inner shape to the target
                // (keepdim) shape. The byte / element count is
                // unchanged.
                let target_layout = Layout::contiguous(target_shape.clone());
                let relabeled = match self.backend.try_clone(&reduced, &target_layout) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                };
                return CacheEntry::Owned(TrackedTensor::new(relabeled, target_shape.clone()));
            }
            Op::MinDim(d) => {
                let a = self.get_gt_c(inputs, 0, cache);
                match self.backend.reduce(fuel_core_types::op::ReduceOp::Min, &a.storage, &a.layout(), &[*d]) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- softmax --
            Op::SoftmaxLastDim => {
                let a = self.get_gt_c(inputs, 0, cache);
                self.backend.softmax_last_dim(&a.storage, &a.layout()).expect("SoftmaxLastDim")
            }

            // -- rms norm (fused) --
            Op::RmsNormLastDim { eps } => {
                let a = self.get_gt_c(inputs, 0, cache);
                match self.backend.rms_norm_last_dim(&a.storage, &a.layout(), *eps) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- layer norm backward (fused) --
            Op::LayerNormLastDimBackward { eps } => {
                let x = self.get_gt_c(inputs, 0, cache);
                let up = self.get_gt_c(inputs, 1, cache);
                match self.backend.layer_norm_last_dim_backward(
                    &x.storage, &up.storage, &x.layout(), &up.layout(), *eps,
                ) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- softmax backward (fused) --
            Op::SoftmaxLastDimBackward => {
                let y = self.get_gt_c(inputs, 0, cache);
                let up = self.get_gt_c(inputs, 1, cache);
                match self.backend.softmax_last_dim_backward(
                    &y.storage, &up.storage, &y.layout(), &up.layout(),
                ) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- rms norm backward (fused) --
            Op::RmsNormLastDimBackward { eps } => {
                let x = self.get_gt_c(inputs, 0, cache);
                let up = self.get_gt_c(inputs, 1, cache);
                match self.backend.rms_norm_last_dim_backward(
                    &x.storage, &up.storage, &x.layout(), &up.layout(), *eps,
                ) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- rope (fused, stride-aware on x) --
            Op::Rope => {
                let x = self.get_gt(inputs, 0, cache);
                let cos = self.get_gt_c(inputs, 1, cache);
                let sin = self.get_gt_c(inputs, 2, cache);
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
                let (src, ids) = (self.get_gt_c(inputs, 0, cache), self.get_gt_c(inputs, 1, cache));
                match self.backend.index_select(&src.storage, &ids.storage, &src.layout(), &ids.layout(), *dim) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }
            Op::Gather { dim } => {
                let (src, ids) = (self.get_gt_c(inputs, 0, cache), self.get_gt_c(inputs, 1, cache));
                match self.backend.gather(&src.storage, &ids.storage, &src.layout(), &ids.layout(), *dim) {
                    Ok(s) => s,
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- cross-device transfer --
            Op::Copy { target } | Op::Move { target } => {
                // Copy and Move produce the same output (fresh tensor
                // on target); they differ only in whether the scheduler
                // lets the source die. The executor's post-op cache
                // eviction handles that automatically via
                // destructive_input().
                let a = self.get_gt(inputs, 0, cache);
                let layout = a.layout();
                match self.backend.copy_to(&a.storage, &layout, *target) {
                    Ok(s) => s,
                    // Single-device backends bail on cross-device; in
                    // that case fall through to the CPU fallback,
                    // which on the reference backend is a pass-through.
                    Err(_) => return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache)),
                }
            }

            // -- destructive release --
            Op::Release => {
                // Release produces a zero-element marker. The scheduler's
                // ordering pass (derive_ordering, arriving in a follow-up
                // PR) pins this op to run after every reader of its input.
                // Until that pass lands, Release emitted into the graph
                // today still runs in topo order — it just can't be
                // safely scheduled ahead of non-destructive readers by
                // the rule author. Backends allocate a zero-element
                // output via cpu_fallback.
                return CacheEntry::Owned(self.cpu_fallback(op, inputs, shape, dtype, cache));
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
        let a = self.get_gt_c(inputs, 0, cache);
        self.backend.unary(op, &a.storage, &a.layout()).expect("unary")
    }

    fn do_binary(
        &self, op: BinaryOp,
        inputs: &[NodeId],
        cache: &HashMap<NodeId, CacheEntry<B::Storage>>,
    ) -> B::Storage {
        // Stride-aware: binary shader handles non-contiguous inputs
        // (lazy permute views, broadcasts with stride=0) directly.
        let a = self.get_gt(inputs, 0, cache);
        let b = self.get_gt(inputs, 1, cache);
        self.backend.binary(op, &a.storage, &b.storage, &a.layout(), &b.layout()).expect("binary")
    }

    fn do_permute(&self, a: &TrackedTensor<B::Storage>, axes: &[usize], out_shape: &Shape) -> TrackedTensor<B::Storage> {
        let rank = a.shape.dims().len();

        // Lazy view: reorder strides without data movement. Works for
        // any permutation of any layout (contiguous or already strided).
        // Downstream ops that need contiguous data call get_gt_c which
        // auto-materializes; stride-aware ops (matmul, RoPE) use
        // get_gt and handle strides natively.
        if axes.len() == rank {
            let _s = debug_span!("permute_view", elems = out_shape.elem_count()).entered();
            let src_layout = a.layout();
            let permuted = src_layout.permute(axes).expect("permute axes valid");
            return TrackedTensor {
                storage: std::sync::Arc::clone(&a.storage),
                shape: out_shape.clone(),
                custom_layout: Some(permuted),
            };
        }

        // Fallback: axes.len() != rank (shouldn't happen for valid permutations).
        let _s = debug_span!("permute_copy", elems = out_shape.elem_count()).entered();
        let in_dims = a.shape.dims();
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
        // Lazy broadcast view: set stride=0 on broadcast dims. Binary
        // ops (the typical consumer of broadcasts, e.g. norm-gain *
        // activations) handle stride=0 natively. get_gt_c auto-
        // materializes for ops that need contiguous storage.
        let _s = debug_span!("broadcast_view", elems = target.elem_count()).entered();
        let src_layout = a.layout();
        let src_stride = src_layout.stride();
        let mut strides: DimVec = DimVec::from_elem(0, dst_dims.len());
        for i in 0..src_dims.len() {
            if src_dims[i] == dst_dims[pad + i] {
                strides[pad + i] = src_stride[i];
            }
            // else: size-1 source dim broadcast to > 1 target dim, stride stays 0
        }
        let layout = Layout::new(target.clone(), strides, src_layout.start_offset());
        TrackedTensor {
            storage: std::sync::Arc::clone(&a.storage),
            shape: target.clone(),
            custom_layout: Some(layout),
        }
    }

    fn do_concat(
        &self, dim: usize,
        inputs: &[NodeId], out_shape: &Shape,
        cache: &HashMap<NodeId, CacheEntry<B::Storage>>,
    ) -> TrackedTensor<B::Storage> {
        let _s = debug_span!("concat", dim, elems = out_shape.elem_count()).entered();
        // Stride-aware: native concat handles per-operand strides
        // (lazy permute / broadcast views) directly.
        let a = self.get_gt(inputs, 0, cache);
        let b = self.get_gt(inputs, 1, cache);

        // Fast path: backend provides a single-dispatch concat.
        if let Ok(s) = self.backend.concat_along_dim(&a.storage, &b.storage, dim, &a.layout(), &b.layout()) {
            return TrackedTensor::new(s, out_shape.clone());
        }

        // Fallback: materialize both inputs to contiguous, then use
        // copy_strided_src to build the concat output.
        let a_c = self.materialize_if_needed(a);
        let b_c = self.materialize_if_needed(b);
        let dtype = self.backend.storage_dtype(&a_c.storage);
        let mut dst = self.backend.alloc_zeros(out_shape, dtype).expect("concat alloc");
        let out_dims = out_shape.dims();
        let a_dim = a_c.shape.dims()[dim];
        let b_dim = b_c.shape.dims()[dim];
        let inner: usize = out_dims[dim + 1..].iter().product::<usize>().max(1);
        let outer: usize = out_dims[..dim].iter().product::<usize>().max(1);
        let out_row = out_dims[dim] * inner;
        if outer == 1 {
            self.backend.copy_strided_src(&a_c.storage, &mut dst, 0, &a_c.layout()).expect("concat a");
            self.backend.copy_strided_src(&b_c.storage, &mut dst, a_dim * inner, &b_c.layout()).expect("concat b");
        } else {
            let a_ss = a_dim * inner;
            let b_ss = b_dim * inner;
            for o in 0..outer {
                let al = Layout::contiguous_with_offset(&Shape::from_dims(&[a_ss]), o * a_ss);
                self.backend.copy_strided_src(&a_c.storage, &mut dst, o * out_row, &al).expect("concat a");
                let bl = Layout::contiguous_with_offset(&Shape::from_dims(&[b_ss]), o * b_ss);
                self.backend.copy_strided_src(&b_c.storage, &mut dst, o * out_row + a_ss, &bl).expect("concat b");
            }
        }
        TrackedTensor::new(dst, out_shape.clone())
    }

    fn do_slice(&self, dim: usize, start: usize, a: &TrackedTensor<B::Storage>, out_shape: &Shape) -> TrackedTensor<B::Storage> {
        // TODO(lazy-slice): a lazy stride view would be zero-copy but
        // our stride-aware shaders (binary, concat, matmul, rope) don't
        // currently add the view's `start_offset` to computed offsets,
        // which silently breaks any downstream op that reads a sliced
        // view with non-zero start (partial RoPE's `x_pass` is the
        // motivating case in Phi-2). Materialize the slice for now;
        // revisit once all stride-aware kernels take start_offset.
        let _s = debug_span!("slice_copy", dim, start, elems = out_shape.elem_count()).entered();
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

fn panic_payload_to_string(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&'static str>() { return s.to_string(); }
    if let Some(s) = p.downcast_ref::<String>()       { return s.clone();     }
    "<non-string panic payload>".to_string()
}

fn op_short_name(op: &Op) -> &'static str {
    match op {
        Op::Const => "Const", Op::MatMul => "MatMul",
        Op::Add => "Add", Op::Sub => "Sub", Op::Mul => "Mul", Op::Div => "Div",
        Op::Neg => "Neg", Op::Sqr => "Sqr", Op::Sqrt => "Sqrt",
        Op::Exp => "Exp", Op::Log => "Log",
        Op::Sin => "Sin", Op::Cos => "Cos", Op::Tanh => "Tanh",
        Op::Sigmoid => "Sigmoid", Op::Silu => "Silu", Op::Gelu => "Gelu",
        Op::Relu => "Relu", Op::Step => "Step",
        Op::Maximum => "Maximum", Op::Minimum => "Minimum",
        Op::AddScalar(_) => "AddScalar", Op::MulScalar(_) => "MulScalar",
        Op::PowI(_) => "PowI", Op::Cast(_) => "Cast",
        Op::Reshape(_) => "Reshape", Op::Unsqueeze{..} => "Unsqueeze", Op::Transpose => "Transpose",
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
        Op::QMatMul { .. } => "QMatMul",
        _ => "Other",
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

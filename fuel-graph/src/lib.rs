//! # fuel-graph
//!
//! Lazy computation graph for Phase 6 of the fuel ML framework.
//!
//! ## Role
//!
//! Phase 6 transitions fuel from eager execution to a lazy computation graph
//! with an autonomous router. This crate holds the minimal vocabulary that
//! makes that transition possible:
//!
//! - [`Op`] — a closed enum describing every operation that can appear in a
//!   graph node. New ops are added here and consumed by every backend.
//! - [`Graph`] — an arena of [`Node`]s. Nodes are immutable once added and
//!   referenced by index. The arena grows; it never shrinks.
//! - [`Tensor`] — a cheaply-cloneable handle to a `(graph, node_id)` pair.
//!   Users build computations by calling methods on `Tensor`; each method
//!   appends a node to the underlying graph and returns a new handle.
//!
//! Execution is deliberately not implemented here. A separate executor crate
//! (today [`fuel_reference_backend::exec`](../fuel_reference_backend/exec/))
//! walks a `Graph` and produces concrete outputs. Future backends (CPU-fast,
//! CUDA, Metal, etc.) will plug in via the same pattern.
//!
//! ## Scope of the MVP
//!
//! This is the first landing of the graph types. It is deliberately minimal:
//!
//! - Four float dtypes: `f32`, `f64`, `bf16`, `f16`. Integer dtypes will be
//!   added alongside the reference backend's integer op coverage, not
//!   speculatively here.
//! - A small starter `Op` catalog: `Const`, `Add`, `Mul`, `MatMul`,
//!   `Transpose`, `Relu`, `Sqr`, `Exp`. More ops are added as validation
//!   tests for them land.
//! - Unfused backward autograd: [`Tensor::backward`] walks the forward
//!   graph in reverse, applies per-op gradient rules, and emits new graph
//!   nodes for the gradient of every leaf. Fused backward and symbolic
//!   graph rewriting are deferred to Phase 6d.
//! - No fusion, no planner. Those belong to later sub-phases of Phase 6.
//! - Single-threaded. The graph is wrapped in `Rc<RefCell<_>>` because the
//!   MVP needs no cross-thread sharing. Moving to `Arc<Mutex<_>>` is a
//!   one-line change when multi-threaded building becomes relevant.

pub mod opt;

use fuel_core_types::{DeviceLocation, DType, Shape};
use half::{bf16, f16};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

/// Compute the topological order of every node reachable from `root`.
///
/// The returned order places inputs before the nodes that depend on them,
/// so iterating the list and caching each result as you go guarantees every
/// dependency is available when its dependents are computed. Uses an
/// explicit stack iterative post-order DFS — no recursion, so arbitrarily
/// deep graphs are safe.
///
/// This is the utility an executor walks to realize a tensor, and the
/// utility [`Tensor::backward`] walks in reverse to construct the backward
/// graph.
pub fn topo_order(graph: &Graph, root: NodeId) -> Vec<NodeId> {
    topo_order_multi(graph, &[root])
}

/// Compute the topological order of every node reachable from ANY of the
/// given roots. Each node appears in the output exactly once; the order
/// still satisfies the dependency-before-dependent property, so walking
/// it and caching results as you go computes every input before any node
/// that depends on it.
///
/// Used by the multi-output `realize_many` entry points in executors:
/// KV-cached inference wants the per-step logits AND the updated K/V
/// tensors for every layer as outputs of a single forward pass. Asking
/// the executor for them via `topo_order_multi` walks the graph exactly
/// once for the combined dependency set, then the executor reads each
/// requested root out of the cache.
pub fn topo_order_multi(graph: &Graph, roots: &[NodeId]) -> Vec<NodeId> {
    let mut order = Vec::new();
    let mut visited: HashSet<NodeId> = HashSet::new();
    // Each stack entry is (node, processed). `processed = false` means
    // "push children then revisit me"; `processed = true` means "all my
    // children are already in `order`, add me now."
    let mut stack: Vec<(NodeId, bool)> = roots.iter().map(|&r| (r, false)).collect();
    while let Some((id, processed)) = stack.pop() {
        if processed {
            order.push(id);
            continue;
        }
        if visited.contains(&id) {
            continue;
        }
        visited.insert(id);
        stack.push((id, true));
        for &child in &graph.node(id).inputs {
            if !visited.contains(&child) {
                stack.push((child, false));
            }
        }
    }
    order
}

/// A node ID in the arena. Stable for the lifetime of the [`Graph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub usize);

/// Quantization block format for [`Op::QMatMul`]. Matches the
/// GGML/GGUF block layouts; each variant implies a fixed bytes-per-block
/// and elements-per-block. Only variants for which a backend has a
/// fused dequant-in-kernel matmul are currently exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantType {
    /// 32-element block = 2 bytes f16 scale + 16 bytes packed u4 quants.
    Q4_0,
    /// 32-element block = 2 bytes f16 scale + 32 bytes i8 quants.
    Q8_0,
    /// 256-element super-block = 2 bytes f16 d + 2 bytes f16 dmin +
    /// 12 bytes of 6-bit-packed sub-block scales/mins + 128 bytes
    /// of 4-bit-packed quants. GGML k-quant "medium" format.
    Q4_K_M,
}

impl QuantType {
    /// Bytes per quantization block.
    pub fn bytes_per_block(self) -> usize {
        match self {
            QuantType::Q4_0 => 18,
            QuantType::Q8_0 => 34,
            QuantType::Q4_K_M => 144,
        }
    }
    /// Elements per quantization block.
    pub fn elements_per_block(self) -> usize {
        match self {
            QuantType::Q4_0 | QuantType::Q8_0 => 32,
            QuantType::Q4_K_M => 256,
        }
    }
}

/// The closed enum of operations a graph node can represent.
///
/// This is the API contract between the graph layer and every backend.
/// Adding a new op means (1) adding a variant here, (2) teaching every
/// backend how to execute it, and (3) teaching the reference backend how
/// to compute its textbook answer.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    // --- leaves ---
    /// A concrete constant tensor. The data lives on the node itself and
    /// is the only kind of node with no inputs.
    Const(ConstData),

    // --- element-wise binary ---
    /// Element-wise addition.
    Add,
    /// Element-wise subtraction (`a - b`).
    Sub,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise division (`a / b`).
    Div,

    // --- element-wise unary ---
    /// Element-wise negation (`-x`).
    Neg,
    /// Element-wise square (`x * x`).
    Sqr,
    /// Element-wise square root.
    Sqrt,
    /// Element-wise exponential (`e^x`).
    Exp,
    /// Element-wise natural logarithm.
    Log,
    /// Element-wise sine.
    Sin,
    /// Element-wise cosine.
    Cos,
    /// Element-wise hyperbolic tangent.
    Tanh,
    /// Element-wise logistic sigmoid.
    Sigmoid,
    /// SiLU activation (x · sigmoid(x)), also called Swish.
    Silu,
    /// GELU activation (tanh approximation).
    Gelu,
    /// Rectified linear unit (`max(0, x)`), element-wise.
    Relu,
    /// Heaviside step function: `1` where `x > 0`, `0` otherwise. Serves
    /// as the derivative of `Relu` and as a building block for other
    /// comparison-based gradients.
    Step,

    // --- linear algebra and shape ---
    /// Rank-2 matrix multiply.
    MatMul,
    /// Rank-2 transpose.
    Transpose,
    /// N-dimensional permutation. The parameter is the new axis order:
    /// `out.shape[i] = in.shape[axes[i]]`. For a rank-3 input `[a, b, c]`
    /// with `axes = [2, 0, 1]`, output shape is `[c, a, b]`. The axes
    /// vector must be a permutation of `0..rank`.
    Permute(Vec<usize>),

    // --- dtype, shape, and broadcasting ---
    /// Convert a tensor to a different dtype. The target dtype is carried
    /// on the op itself; the source dtype is whatever the input node's
    /// dtype is.
    Cast(DType),
    /// Broadcast a tensor to a larger shape. The input shape must be
    /// broadcast-compatible with the target (NumPy rules). Used directly
    /// by user code and as the backward rule for `SumAll`, `MeanAll`,
    /// and axis reductions.
    BroadcastTo(Shape),
    /// Reshape a tensor to a new shape with the same element count. Data
    /// is unchanged; only the shape metadata is replaced. Serves as the
    /// building block for inserting and removing size-1 dimensions (the
    /// "unsqueeze"/"squeeze" operations most frameworks expose separately).
    Reshape(Shape),
    /// Sum-reduce a tensor to a smaller shape by summing along any dims
    /// where the source was broadcast against the target. This is the
    /// backward rule for `BroadcastTo` and is symmetric with it: both ops
    /// are each other's gradient. For a source shape `[2, 3, 4]` being
    /// reduced to `[3, 4]`, the sum is along dim 0; for `[2, 3, 4]` to
    /// `[2, 1, 4]`, the sum is along dim 1 while keeping the dim.
    ReduceSumTo(Shape),

    // --- reductions to a scalar ---
    /// Sum of every element, producing a rank-0 tensor.
    SumAll,
    /// Maximum of every element, producing a rank-0 tensor.
    MaxAll,
    /// Minimum of every element, producing a rank-0 tensor.
    MinAll,
    /// Arithmetic mean of every element, producing a rank-0 tensor.
    MeanAll,

    // --- reductions along one dimension ---
    /// Sum along the given dim; the reduced dim is removed from the output.
    SumDim(usize),
    /// Max along the given dim; the reduced dim is removed from the output.
    MaxDim(usize),
    /// Min along the given dim; the reduced dim is removed from the output.
    MinDim(usize),
    /// Mean along the given dim; the reduced dim is removed from the output.
    MeanDim(usize),

    // --- compositions ---
    /// Softmax along the last dimension.
    SoftmaxLastDim,
    /// Layer normalization along the last dimension, without affine params.
    /// The epsilon is carried on the op and converted to the target dtype
    /// by the executor.
    LayerNormLastDim { eps: f64 },
    /// Root-mean-square normalization along the last dimension, no
    /// affine parameters. Formula:
    ///   y = x / sqrt(mean(x², last) + eps)
    /// This is the norm RMSNorm-family models (LLaMA, Qwen, Gemma,
    /// etc.) use on every attention and MLP input. Decomposed in
    /// fuel-graph proper into sqr → mean_dim → reshape → add_scalar
    /// → sqrt → broadcast_to → div (9 nodes), this fused op exists
    /// so backends can dispatch it as a single kernel — the
    /// difference between 9 kernel launches and 1 at 45+ sites per
    /// forward pass.
    RmsNormLastDim { eps: f64 },
    /// Fused rotary position embedding. Inputs: `(x, cos, sin)`.
    /// `x` has shape `[..., seq, head_dim]` (head_dim even). `cos` and
    /// `sin` both have shape `[seq, head_dim]` and broadcast across
    /// any leading dims. Output shape == x shape.
    ///
    /// Formula (rotate_half convention):
    ///   out[..., s, i]        = x[..., s, i]         * cos[s, i]         - x[..., s, i + h] * sin[s, i]
    ///   out[..., s, i + h]    = x[..., s, i + h]     * cos[s, i + h]     + x[..., s, i]     * sin[s, i + h]
    /// where `h = head_dim / 2`.
    ///
    /// Replaces the slice+neg+concat+broadcast_mul decomposition in
    /// `Tensor::rope_with_tables`, which was dispatching 72+ kernels
    /// per layer (~1760 per TinyLlama token). This fuses it to 1.
    Rope,

    /// Quantized matrix multiply: `C = A @ dequant(W_Q)`. The second
    /// input is a U32-typed tensor holding raw quantization-block
    /// bytes; the backend dequantizes on the fly inside its matmul
    /// kernel (avoiding a full dequant roundtrip through F32/BF16).
    ///
    /// Input shapes:
    ///   A: `[..., M, K]` F32 (activations)
    ///   W_Q: `[n_bytes / 4]` U32 — a row-major stream of Q-type blocks
    ///         for a `[N, K]` weight matrix (llama.cpp/GGUF convention).
    /// Output shape: `[..., M, N]` F32.
    ///
    /// Backward: gradient through W_Q is zero (quantized weights are
    /// frozen in the expected use case). Gradient through A is not
    /// implemented at the moment — add it if/when we need to fine-tune
    /// over Q-weights.
    QMatMul {
        /// Quantization type for the weight blocks.
        quant_type: QuantType,
        /// Weight input-feature dim (the contracted dim).
        k: usize,
        /// Weight output-feature dim.
        n: usize,
    },

    // --- backward helpers ---
    //
    // These ops encapsulate the "all at once" backward rules for
    // compositions whose gradients are awkward to express as
    // compositions of primitives. They take (forward_output_or_input,
    // upstream) and emit the gradient of the input.
    //
    /// Softmax-last-dim backward. Inputs: (forward_softmax_output, upstream).
    /// Output: the gradient of the input to the softmax. Formula:
    /// `s * (g - sum(g * s, last_dim, keepdim=true))` where `s` is the
    /// forward output and `g` is the upstream gradient.
    SoftmaxLastDimBackward,
    /// Layer-norm-last-dim backward. Inputs: (original_x, upstream).
    /// Output: the gradient of `x`. Computed in full from scratch
    /// because the forward normalized tensor alone doesn't carry enough
    /// info (we'd also need `rstd`, which the forward node doesn't
    /// expose as an output). Takes `eps` as a parameter so the backward
    /// op recomputes the same statistics the forward used.
    LayerNormLastDimBackward { eps: f64 },
    /// Fused RMSNorm-last-dim backward. Inputs: (original_x, upstream).
    /// Output: grad_x = r_rms * (upstream - x * s / (n * (mean_sq + eps))).
    /// Takes `eps` so the backward op recomputes the same
    /// normalization constant the forward used. Replaces the
    /// 12-node primitive synthesis the autograd graph used to emit.
    /// Backends that don't ship a fused kernel fall back to the
    /// primitive decomposition.
    RmsNormLastDimBackward { eps: f64 },

    // --- integer-producing reductions ---
    //
    // These reduce a float tensor along one dim and emit a U32 tensor of
    // indices. Non-differentiable — they exist for classification
    // prediction and similar discrete workloads, not as part of any
    // training graph. Backward panics cleanly if a user builds these
    // inside a graph they then differentiate.
    /// Index of the maximum along `dim`. Output: U32 with the reduced
    /// dim removed.
    ArgMaxDim(usize),
    /// Index of the minimum along `dim`. Output: U32 with the reduced
    /// dim removed.
    ArgMinDim(usize),

    // --- shape manipulation ---
    /// Concatenate two tensors along `dim`. Both must have the same rank,
    /// same dtype, and equal sizes in every dim except `dim`.
    Concat { dim: usize },
    /// Slice (narrow) a tensor along `dim`: take elements `[start, start+len)`.
    /// Output shape: same as input but with `dim` shrunk to `len`.
    Slice { dim: usize, start: usize, len: usize },

    // --- scalar-by-tensor ops ---
    //
    // These exist to avoid building a full-shape const just to add or
    // multiply by a constant. Stored as f64 on the op and converted to
    // the target dtype at realize time.
    /// Add a scalar to every element.
    AddScalar(f64),
    /// Multiply every element by a scalar.
    MulScalar(f64),
    /// Raise every element to an integer power via repeated multiplication.
    /// Works without an extra `log`/`exp` pair so it's numerically clean
    /// for small integer exponents like squaring or cubing.
    PowI(i32),
    /// Clamp every element to the inclusive range `[min, max]`.
    Clamp { min: f64, max: f64 },

    // --- element-wise max/min between two tensors ---
    /// Element-wise maximum of two tensors with matching shape.
    Maximum,
    /// Element-wise minimum of two tensors with matching shape.
    Minimum,

    // --- indexing ops ---
    //
    // These take a "data" tensor (floats) as their first input and an
    // "index" tensor (integer) as their second input. The executor
    // reads both operands from the cache, dispatches on the data
    // tensor's dtype for the output, and uses the index tensor as-is.
    /// Index-select along a single dimension with a 1-D index vector.
    /// Output shape: same as input, but with `dim` replaced by
    /// `indices.elem_count()`.
    IndexSelect { dim: usize },
    /// N-dimensional gather along `dim`. Output shape matches the index
    /// tensor's shape. `out[..., k, ...] = data[..., indices[..., k, ...], ...]`
    /// with `dim` being the position where the index value substitutes
    /// for the output coordinate.
    Gather { dim: usize },
    /// Index-add: the backward of `IndexSelect`. Takes a base tensor,
    /// a 1-D `U32` index vector, and a src tensor, and adds src to base
    /// at the indexed positions along `dim`. `out[..., indices[i], ...] +=
    /// src[..., i, ...]`. Non-indexed positions of base are copied
    /// through unchanged. Works functionally (returns a new tensor).
    IndexAdd { dim: usize },
    /// Scatter-add: the backward of `Gather`. Takes a base tensor, an
    /// N-D `U32` index tensor matching the src shape, and a src tensor,
    /// and accumulates src values into base at positions given by
    /// substituting `indices` at position `dim`. `out[p with dim ←
    /// indices[p]] += src[p]`.
    ScatterAdd { dim: usize },

    // --- 2-D convolution ---
    /// 2-D convolution with stride, symmetric padding, and grouped
    /// channels. Inputs: `(x, weight)` or `(x, weight, bias)`.
    ///   - `x`:     `[N, Cin, H, W]`
    ///   - `weight`: `[Cout, Cin/groups, Kh, Kw]`
    ///   - `bias`:  `[Cout]` (optional)
    ///
    /// Output shape: `[N, Cout, Hout, Wout]` where
    ///   Hout = (H + 2·pad.0 − Kh) / stride.0 + 1
    ///   Wout = (W + 2·pad.1 − Kw) / stride.1 + 1
    ///
    /// `groups` splits the channels into `groups` independent
    /// convolutions (`Cin/groups` input channels per group mapped to
    /// `Cout/groups` output channels). `groups=1` is the standard
    /// cross-channel conv; `groups=Cin=Cout` is the depthwise case
    /// (ConvNeXt's kernel-7 block mixer).
    ///
    /// The op exists as a primitive so backends can fuse the full
    /// im2col+gemm (or direct conv) pipeline into a single kernel
    /// launch. The lazy-graph composition path (slice + concat +
    /// matmul) was correct but spawned ~9·Cin ops per forward pass;
    /// this op is the unblock for conv-heavy anchor inference speed.
    Conv2D {
        stride:  (usize, usize),
        padding: (usize, usize),
        groups:  usize,
    },

    // --- cross-device transfer ---
    /// Copy the input tensor to a specific device. Source stays
    /// resident (non-destructive). Source device is implicit from
    /// `inputs[0]`'s current residency. When source and target
    /// devices match, the backend is free to turn this into a cheap
    /// clone; when they differ, the backend must actually transfer
    /// the data (today: through a host buffer — Phase 3 has no P2P
    /// fast path yet).
    ///
    /// Phase 3 of the unified scheduler work. The scheduler (Phase 4)
    /// inserts these nodes automatically; users can also construct
    /// them explicitly via `Tensor::copy_to_device`.
    Copy { target: DeviceLocation },

    /// Release the input tensor's device-resident storage. Produces a
    /// zero-element marker output (NodeId placeholder for graph
    /// bookkeeping; no meaningful data). After this op runs, the
    /// input's device storage is gone; any subsequent reader of the
    /// input's NodeId must get its data from a different path (a
    /// sibling Copy output, or re-execution of the producer).
    ///
    /// Destructive on `inputs[0]`. Scheduler must pin `Op::Release`
    /// to run after every other reader of its input via
    /// [`opt::derive_ordering`].
    Release,

    /// Transfer the input tensor to `target` device, destroying the
    /// source in the process. Semantically equivalent to
    /// `Op::Copy { target }` followed by `Op::Release` on the source,
    /// with the explicit guarantee that backends MAY fast-path the
    /// fused form as a single transfer (e.g., a zero-copy handoff if
    /// source and target are the same memory type).
    ///
    /// Output is a fresh tensor on `target` with the input's shape
    /// and dtype. Destructive on `inputs[0]`; the scheduler pins this
    /// op to run after every non-destructive reader of the source via
    /// [`opt::derive_ordering`].
    Move { target: DeviceLocation },
}

impl Op {
    /// Index into `inputs` that this op destroys on execution. `None`
    /// means the op is non-destructive — every input remains readable
    /// after the op completes. Destructive ops need the scheduler to
    /// pin them to run after all other readers of the destroyed input,
    /// via ordering edges derived by [`opt::derive_ordering`].
    pub fn destructive_input(&self) -> Option<usize> {
        match self {
            Op::Release | Op::Move { .. } => Some(0),
            _ => None,
        }
    }
}

/// Concrete data stored on a `Const` node. One variant per supported
/// dtype; the executor matches on the variant to extract the correctly
/// typed buffer.
///
/// The index-tensor variants (`U32` today, future `I64`/`U64`/etc.) carry
/// integer data used for gather/scatter and similar indexing operations.
/// They are not expected to participate in arithmetic ops — doing so
/// panics at the executor level.
///
/// Variants hold [`Arc<[T]>`] rather than `Vec<T>` so that:
///
/// - cloning the enum (which happens on every `graph.node(id)` fetch
///   because `Node: Clone`) is a refcount bump, not a memcpy;
/// - weight buffers loaded once at model-load time can be shared
///   across every forward pass and every layer that reuses them,
///   which for a TinyLlama-sized model turns gigabytes of per-call
///   memcpy into zero-cost Arc bumps.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstData {
    F32(Arc<[f32]>),
    F64(Arc<[f64]>),
    BF16(Arc<[bf16]>),
    F16(Arc<[f16]>),
    /// Unsigned 32-bit integer tensor, used for index tensors in
    /// gather/scatter/index_select. Matches the convention of every real
    /// backend fuel has (Candle, CUDA, Metal all use `u32` for indices).
    U32(Arc<[u32]>),
}

impl ConstData {
    /// The number of elements in this constant, for shape validation.
    pub fn elem_count(&self) -> usize {
        match self {
            ConstData::F32(v) => v.len(),
            ConstData::F64(v) => v.len(),
            ConstData::BF16(v) => v.len(),
            ConstData::F16(v) => v.len(),
            ConstData::U32(v) => v.len(),
        }
    }

    /// The dtype of this constant. Useful when constructing a `Node` from
    /// a `ConstData` without duplicating the dtype tag.
    pub fn dtype(&self) -> DType {
        match self {
            ConstData::F32(_) => DType::F32,
            ConstData::F64(_) => DType::F64,
            ConstData::BF16(_) => DType::BF16,
            ConstData::F16(_) => DType::F16,
            ConstData::U32(_) => DType::U32,
        }
    }
}

/// A single node in the graph: an operation and the IDs of its inputs.
/// The node also caches its output shape and dtype so downstream builders
/// can validate without walking back through the graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub op:     Op,
    pub inputs: Vec<NodeId>,
    pub shape:  Shape,
    pub dtype:  DType,
}

/// The graph arena. Stores every node added during a computation-building
/// session. Nodes are append-only and indexed by [`NodeId`].
///
/// Placement metadata (`DeviceLocation` per node) lives in a side-table
/// so existing `Node { ... }` construction sites don't need to be
/// modified. This is the Phase-1 shape: placements are opt-in, inert
/// hints that the executor may validate but does not yet act on.
#[derive(Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    /// Sparse map of per-node placement hints. Entries are only present
    /// when explicitly set via [`Graph::set_placement`]. Absent entries
    /// mean "inherit from the executor's default device."
    placements: HashMap<NodeId, DeviceLocation>,
    /// NodeIds the executor must schedule even when their outputs are
    /// unreachable from the user's roots. Populated by graph-mutating
    /// rules that emit side-effecting ops (e.g. `Op::Release` for
    /// residency eviction). See [`Graph::add_side_effect_root`].
    side_effect_roots: Vec<NodeId>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self { nodes: Vec::new(), placements: HashMap::new(), side_effect_roots: Vec::new() }
    }

    /// Register a NodeId as a side-effect root — a node whose output
    /// isn't reachable from any user-requested root but whose
    /// execution the executor must still schedule. Used for
    /// destructive ops (e.g. `Op::Release`) emitted by graph-mutating
    /// rules: the rule needs Release to run (freeing device memory),
    /// but Release's zero-element marker has no consumer.
    ///
    /// The executor's realize paths walk the user's roots AND these
    /// side-effect roots via a single combined `execution_plan` call.
    pub fn add_side_effect_root(&mut self, id: NodeId) {
        assert!(id.0 < self.nodes.len(), "add_side_effect_root: id out of bounds");
        if !self.side_effect_roots.contains(&id) {
            self.side_effect_roots.push(id);
        }
    }

    /// Snapshot the registered side-effect roots. Executor concatenates
    /// these with the user's requested roots before computing the plan.
    pub fn side_effect_roots(&self) -> &[NodeId] {
        &self.side_effect_roots
    }

    /// The number of nodes currently in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has any nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Borrow a node by ID. Panics if the ID is out of bounds — it should
    /// not be possible for a valid `Tensor` handle to produce an invalid
    /// ID, so any such panic indicates a bug in graph construction.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    /// Append a node and return its fresh ID. Internal helper used by the
    /// `Tensor` builders and by `opt` passes that canonicalize or rewrite
    /// the graph by appending fresh nodes.
    pub(crate) fn push(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    /// Rewrite every occurrence of `old_input` in `node`'s input list to
    /// `new_input`. In-place mutation — the *only* exception to the
    /// otherwise append-only invariant.
    ///
    /// Callers are responsible for ensuring `new_input` is semantically
    /// equivalent to `old_input` at this edge: same shape, same dtype,
    /// and produces data the consumer can read without error on the
    /// relevant device. Used by transform passes (residency eviction,
    /// fusion) that need to redirect a specific consumer's edge.
    pub(crate) fn rewrite_input(
        &mut self,
        node: NodeId,
        old_input: NodeId,
        new_input: NodeId,
    ) {
        assert!(node.0 < self.nodes.len(), "rewrite_input: node out of bounds");
        let inputs = &mut self.nodes[node.0].inputs;
        for inp in inputs.iter_mut() {
            if *inp == old_input {
                *inp = new_input;
            }
        }
    }

    /// Tag a node with a target device. The executor will validate
    /// (post-Phase-1) that the node's op can be evaluated on that
    /// device. In Phase 1 the tag is informational only.
    pub fn set_placement(&mut self, id: NodeId, loc: DeviceLocation) {
        assert!(id.0 < self.nodes.len(), "set_placement: id out of bounds");
        self.placements.insert(id, loc);
    }

    /// Read a node's placement hint, or `None` if the node inherits from
    /// the executor default.
    pub fn placement(&self, id: NodeId) -> Option<DeviceLocation> {
        self.placements.get(&id).copied()
    }
}

/// A shared, mutable graph. Builders clone this cheaply and hand it to
/// every derived tensor so that all tensors in one computation point to
/// the same arena.
pub type SharedGraph = Rc<RefCell<Graph>>;

/// A handle to a node in a shared graph. Cheap to clone. Carries the
/// graph reference so builder methods have everything they need to append
/// a new node.
#[derive(Debug, Clone)]
pub struct Tensor {
    graph: SharedGraph,
    id:    NodeId,
}

impl Tensor {
    /// Wrap an existing `(graph, node_id)` pair as a Tensor handle.
    /// Used by graph-rewriting passes (e.g., `opt::insert_copies`)
    /// that produce new root `NodeId`s and need to hand them back to
    /// call sites as Tensors.
    pub fn from_existing(graph: SharedGraph, id: NodeId) -> Self {
        Self { graph, id }
    }

    /// The node ID this handle points to.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// The shared graph this handle belongs to. Exposed so executors and
    /// tests can walk the underlying arena.
    pub fn graph(&self) -> &SharedGraph {
        &self.graph
    }

    /// The shape of this tensor, read from the underlying node.
    pub fn shape(&self) -> Shape {
        self.graph.borrow().node(self.id).shape.clone()
    }

    /// The dtype of this tensor, read from the underlying node.
    pub fn dtype(&self) -> DType {
        self.graph.borrow().node(self.id).dtype
    }

    /// The placement hint for this tensor's node, if one was set. `None`
    /// means "inherit from the executor's default device."
    pub fn placement(&self) -> Option<DeviceLocation> {
        self.graph.borrow().placement(self.id)
    }

    /// Tag this tensor's node with a target device. The executor will
    /// validate (post-Phase-1) that the node's op can be evaluated on
    /// that device. Returns the same `Tensor` handle so this composes
    /// with builder-style graph construction:
    ///
    /// ```ignore
    /// let y = a.matmul(&b).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
    /// ```
    pub fn on_device(self, loc: DeviceLocation) -> Self {
        self.graph.borrow_mut().set_placement(self.id, loc);
        self
    }

    /// Release this tensor's device-resident storage. Appends an
    /// `Op::Release` node whose output is a zero-element marker (shape
    /// `[0]`, same dtype as self). Destructive on `self` — after this
    /// op runs, `self`'s storage is gone.
    ///
    /// The ordering-analysis pass (`opt::derive_ordering`, arriving in
    /// a follow-up PR) automatically pins the Release to run after
    /// every non-destructive reader of `self`. Callers should NOT
    /// invoke this directly in a graph that has other readers of
    /// `self` until that pass is in place — today's executor has no
    /// mechanism to stop them running in the wrong order.
    pub fn release(&self) -> Self {
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op:     Op::Release,
            inputs: vec![self.id],
            shape:  Shape::from_dims(&[0]),
            dtype,
        });
        Tensor { graph: Rc::clone(&self.graph), id }
    }

    /// Move this tensor's data to `target` device — destroying the
    /// source in the process. Appends an `Op::Move` node. The output
    /// has the same shape and dtype as the input and lives on
    /// `target`; the source's device storage is freed once the op
    /// runs.
    ///
    /// Semantically equivalent to `copy_to_device(target)` followed
    /// by dropping the source. Use this when you KNOW the source
    /// won't be needed after the transfer — it lets backends skip
    /// the intermediate alloc in fast-path cases and lets the
    /// scheduler free the source's device memory immediately.
    ///
    /// If the source may still be read by other ops, use
    /// [`Self::copy_to_device`] instead — Copy leaves the source
    /// resident.
    ///
    /// The ordering-analysis pass (`opt::derive_ordering`)
    /// automatically pins this op to run after every non-destructive
    /// reader of the source. The caller does NOT need to worry about
    /// topo ordering — just emit the Move wherever the data-flow
    /// says it should go.
    pub fn move_to_device(&self, target: DeviceLocation) -> Self {
        let shape = self.shape();
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op:     Op::Move { target },
            inputs: vec![self.id],
            shape,
            dtype,
        });
        Tensor { graph: Rc::clone(&self.graph), id }
    }

    /// Copy this tensor's data to `target` device. Source stays
    /// resident. Appends an `Op::Copy` node. The output has the same
    /// shape and dtype as the input; only residency changes.
    ///
    /// Same-device copies are still a legal thing to ask for — the
    /// backend can optimize them into a cheap clone (or the scheduler
    /// can elide the Copy node entirely). Cross-device copies go
    /// through the active backend's `GraphBackend::copy_to` — today a
    /// host round-trip in the
    /// [`Router`](../fuel_graph_router/struct.Router.html); P2P comes
    /// later.
    pub fn copy_to_device(&self, target: DeviceLocation) -> Self {
        let shape = self.shape();
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op:     Op::Copy { target },
            inputs: vec![self.id],
            shape,
            dtype,
        });
        Tensor { graph: Rc::clone(&self.graph), id }
    }

    /// Build a `Const` tensor from an `f32` slice and shape on a fresh
    /// graph. The new graph is returned via the tensor's `graph` handle.
    ///
    /// `data` takes `impl Into<Arc<[f32]>>` so both `Vec<f32>` (one-time
    /// conversion at the call site) and `Arc<[f32]>` (free clone that
    /// shares buffers across forward passes) work without changing any
    /// existing callers.
    pub fn from_f32(data: impl Into<Arc<[f32]>>, shape: impl Into<Shape>) -> Self {
        Self::from_const(ConstData::F32(data.into()), shape)
    }

    /// Build a `Const` tensor from an `f64` slice and shape on a fresh graph.
    pub fn from_f64(data: impl Into<Arc<[f64]>>, shape: impl Into<Shape>) -> Self {
        Self::from_const(ConstData::F64(data.into()), shape)
    }

    /// Build a `Const` tensor from a `bf16` slice and shape on a fresh graph.
    pub fn from_bf16(data: impl Into<Arc<[bf16]>>, shape: impl Into<Shape>) -> Self {
        Self::from_const(ConstData::BF16(data.into()), shape)
    }

    /// Build a `Const` tensor from an `f16` slice and shape on a fresh graph.
    pub fn from_f16(data: impl Into<Arc<[f16]>>, shape: impl Into<Shape>) -> Self {
        Self::from_const(ConstData::F16(data.into()), shape)
    }

    /// Build a `Const` tensor from a `u32` slice and shape on a fresh
    /// graph. Primarily used to construct index tensors for gather /
    /// scatter / index_select. `u32` is the index type all real fuel
    /// backends use (Candle CPU, CUDA, Metal), so keeping the reference
    /// on the same type means oracle-equivalence tests do not need any
    /// index-type translation.
    pub fn from_u32(data: impl Into<Arc<[u32]>>, shape: impl Into<Shape>) -> Self {
        Self::from_const(ConstData::U32(data.into()), shape)
    }

    /// Build a `Const` tensor from any [`ConstData`] value on a fresh graph.
    /// The per-dtype `from_*` methods funnel through this.
    pub fn from_const(data: ConstData, shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        assert_eq!(
            data.elem_count(),
            shape.elem_count(),
            "Tensor::from_const: data length {} does not match shape element count {}",
            data.elem_count(),
            shape.elem_count(),
        );
        let dtype = data.dtype();
        let graph = Rc::new(RefCell::new(Graph::new()));
        let id = graph.borrow_mut().push(Node {
            op:     Op::Const(data),
            inputs: vec![],
            shape,
            dtype,
        });
        Self { graph, id }
    }

    /// Build a second `Const` tensor that lives on the same graph as
    /// `self`. Use this to add more inputs to an existing computation.
    ///
    /// Pass an `Arc<[f32]>` when you already have one (e.g. model
    /// weights loaded once at startup) to avoid any copy; pass a
    /// `Vec<f32>` when you're building fresh data inline.
    pub fn const_f32_like(&self, data: impl Into<Arc<[f32]>>, shape: impl Into<Shape>) -> Self {
        self.const_like(ConstData::F32(data.into()), shape)
    }

    /// Build a second `Const f64` tensor on the same graph as `self`.
    pub fn const_f64_like(&self, data: impl Into<Arc<[f64]>>, shape: impl Into<Shape>) -> Self {
        self.const_like(ConstData::F64(data.into()), shape)
    }

    /// Build a second `Const bf16` tensor on the same graph as `self`.
    pub fn const_bf16_like(&self, data: impl Into<Arc<[bf16]>>, shape: impl Into<Shape>) -> Self {
        self.const_like(ConstData::BF16(data.into()), shape)
    }

    /// Build a second `Const f16` tensor on the same graph as `self`.
    pub fn const_f16_like(&self, data: impl Into<Arc<[f16]>>, shape: impl Into<Shape>) -> Self {
        self.const_like(ConstData::F16(data.into()), shape)
    }

    /// Build a second `Const u32` (index) tensor on the same graph as `self`.
    pub fn const_u32_like(&self, data: impl Into<Arc<[u32]>>, shape: impl Into<Shape>) -> Self {
        self.const_like(ConstData::U32(data.into()), shape)
    }

    /// Build a second `Const` tensor on the same graph as `self` from any
    /// [`ConstData`] value.
    pub fn const_like(&self, data: ConstData, shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        assert_eq!(
            data.elem_count(),
            shape.elem_count(),
            "const_like: data length {} does not match shape element count {}",
            data.elem_count(),
            shape.elem_count(),
        );
        let dtype = data.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op:     Op::Const(data),
            inputs: vec![],
            shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append an `Add` node `self + other` to the shared graph and return
    /// a handle to the result. Requires matching shapes and matching
    /// graphs.
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.binary_op("add", Op::Add, other, self.shape())
    }

    /// Append a `Mul` node `self * other`.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.binary_op("mul", Op::Mul, other, self.shape())
    }

    /// Append a `MatMul` node. Both operands must have rank ≥ 2. The
    /// last two dims are the matrix dims; leading dims are batch dims.
    ///
    /// - If both operands have the same rank, their batch dims must
    ///   match exactly.
    /// - If one operand is rank-2 and the other is higher rank, the
    ///   rank-2 operand is auto-broadcast to match the higher-rank
    ///   operand's batch prefix. This is the common "linear layer
    ///   across a batch" pattern (`[batch, seq, d_model] @ [d_model, d_out]`).
    /// - Any other rank combination panics; users must reshape first.
    ///
    /// Shape examples:
    /// - `[m, k] @ [k, n]` → `[m, n]`
    /// - `[batch, m, k] @ [batch, k, n]` → `[batch, m, n]`
    /// - `[batch, seq, k] @ [k, n]` → `[batch, seq, n]` (rhs auto-broadcast)
    /// - `[k, n]` @ `[batch, n, m]` → `[batch, k, m]` (lhs auto-broadcast)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(
            Rc::ptr_eq(&self.graph, &other.graph),
            "matmul: tensors must live on the same graph",
        );
        // Mixed-precision matmul: activations stay in their native
        // dtype while weights can live as a lower-precision type on
        // device. Currently supported: (F32 × BF16) → F32 for bf16-
        // quantized LLM serving. Other heterogeneous combinations
        // are rejected.
        let dtypes_ok = self.dtype() == other.dtype()
            || (self.dtype() == DType::F32 && other.dtype() == DType::BF16);
        assert!(
            dtypes_ok,
            "matmul: unsupported dtype combination: lhs={:?}, rhs={:?}",
            self.dtype(),
            other.dtype(),
        );
        let l = self.shape();
        let r = other.shape();
        let ld = l.dims();
        let rd = r.dims();
        assert!(
            ld.len() >= 2 && rd.len() >= 2,
            "matmul: both operands must be rank ≥ 2, got {ld:?} and {rd:?}",
        );

        // Auto-broadcast the rank-2 operand against the other's batch
        // prefix. If ranks already match, this is a no-op.
        let (lhs, rhs) = if ld.len() == rd.len() {
            (self.clone(), other.clone())
        } else if ld.len() == 2 && rd.len() > 2 {
            let mut bc_shape: Vec<usize> = rd[..rd.len() - 2].to_vec();
            bc_shape.push(ld[0]);
            bc_shape.push(ld[1]);
            (self.broadcast_to(Shape::from_dims(&bc_shape)), other.clone())
        } else if rd.len() == 2 && ld.len() > 2 {
            let mut bc_shape: Vec<usize> = ld[..ld.len() - 2].to_vec();
            bc_shape.push(rd[0]);
            bc_shape.push(rd[1]);
            (self.clone(), other.broadcast_to(Shape::from_dims(&bc_shape)))
        } else {
            panic!(
                "matmul: unsupported rank combination {} vs {} — only same-rank or (rank-2 × higher-rank) is supported; reshape first",
                ld.len(),
                rd.len(),
            );
        };

        // Now both operands have the same rank. Validate batch prefix
        // and build the output node.
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        let l = lhs_shape.dims();
        let r = rhs_shape.dims();
        let rank = l.len();
        let batch_rank = rank - 2;
        for i in 0..batch_rank {
            // GQA-style matmul: A may have more batch heads than B
            // (e.g. Q[1,32,1,64] @ K^T[1,4,64,S]). The n_rep factor
            // is inferred by the backend at dispatch time; at the
            // graph level we just validate divisibility.
            let (la, ra) = (l[i], r[i]);
            let ok = la == ra || (la > ra && ra > 0 && la % ra == 0);
            assert!(
                ok,
                "matmul: batch dim mismatch at axis {i}: {la} vs {ra} (not equal and not a GQA-divisible pair)",
            );
        }
        let m = l[rank - 2];
        let k = l[rank - 1];
        let k2 = r[rank - 2];
        let n = r[rank - 1];
        assert_eq!(
            k, k2,
            "matmul: inner dim mismatch (lhs k={k}, rhs k={k2})",
        );
        let mut out_dims: Vec<usize> = l[..batch_rank].to_vec();
        out_dims.push(m);
        out_dims.push(n);
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op:     Op::MatMul,
            inputs: vec![lhs.id, rhs.id],
            shape:  Shape::from_dims(&out_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a [`Op::QMatMul`] node that multiplies `self` (F32
    /// activations, shape `[..., M, K]`) with a Q-type quantized
    /// weight matrix of logical shape `[N, K]` stored as a raw byte
    /// stream (passed in as a U32 tensor; length = n_bytes / 4).
    ///
    /// Output shape: `[..., M, N]` F32. The backend dequantizes the
    /// weight blocks on the fly inside the matmul kernel, so the
    /// quantized blocks stay resident at their compressed size.
    pub fn qmatmul(
        &self,
        weight_bytes: &Tensor,
        quant_type: QuantType,
        k: usize,
        n: usize,
    ) -> Tensor {
        assert!(
            Rc::ptr_eq(&self.graph, &weight_bytes.graph),
            "qmatmul: tensors must live on the same graph",
        );
        assert_eq!(
            self.dtype(),
            DType::F32,
            "qmatmul: activations must be F32, got {:?}",
            self.dtype(),
        );
        assert_eq!(
            weight_bytes.dtype(),
            DType::U32,
            "qmatmul: weight_bytes must be U32 (raw block bytes reinterpreted), got {:?}",
            weight_bytes.dtype(),
        );
        let a_dims = self.shape();
        let a_dims = a_dims.dims();
        assert!(
            a_dims.len() >= 2,
            "qmatmul: activations must be rank ≥ 2, got {a_dims:?}",
        );
        assert_eq!(
            a_dims[a_dims.len() - 1], k,
            "qmatmul: last dim of activations ({}) must equal k ({k})",
            a_dims[a_dims.len() - 1],
        );
        assert_eq!(
            k % quant_type.elements_per_block(),
            0,
            "qmatmul: k={k} must be a multiple of {quant_type:?}'s block size ({})",
            quant_type.elements_per_block(),
        );
        // Validate the weight byte count matches [N, K/block_size] blocks.
        let expected_bytes = n * (k / quant_type.elements_per_block()) * quant_type.bytes_per_block();
        let expected_u32_elems = expected_bytes / 4;
        assert_eq!(
            weight_bytes.shape().elem_count(), expected_u32_elems,
            "qmatmul: weight_bytes has {} u32 elements, expected {expected_u32_elems} for N={n}, K={k}, {quant_type:?}",
            weight_bytes.shape().elem_count(),
        );
        let mut out_dims: Vec<usize> = a_dims[..a_dims.len() - 1].to_vec();
        out_dims.push(n);
        let id = self.graph.borrow_mut().push(Node {
            op:     Op::QMatMul { quant_type, k, n },
            inputs: vec![self.id, weight_bytes.id],
            shape:  Shape::from_dims(&out_dims),
            dtype:  DType::F32,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a [`Op::Conv2D`] node. `self` must be `[N, Cin, H, W]`
    /// (rank 4); `weight` must be `[Cout, Cin/groups, Kh, Kw]` (rank
    /// 4) and live on the same graph. `bias` is optional — when
    /// present it must be rank 1 with length `Cout`. Returns a rank-4
    /// tensor of shape `[N, Cout, Hout, Wout]`.
    ///
    /// Panics if the input ranks don't match, the channel counts are
    /// inconsistent with `groups`, or the output spatial dims would
    /// be non-positive.
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Tensor {
        assert!(
            Rc::ptr_eq(&self.graph, &weight.graph),
            "conv2d: x and weight must live on the same graph",
        );
        if let Some(b) = bias {
            assert!(
                Rc::ptr_eq(&self.graph, &b.graph),
                "conv2d: bias must live on the same graph",
            );
        }
        assert!(groups >= 1, "conv2d: groups must be ≥ 1, got {groups}");
        let x_dims = self.shape();
        let x_dims = x_dims.dims();
        let w_dims = weight.shape();
        let w_dims = w_dims.dims();
        assert_eq!(
            x_dims.len(), 4,
            "conv2d: x must be rank 4 [N, Cin, H, W], got {x_dims:?}",
        );
        assert_eq!(
            w_dims.len(), 4,
            "conv2d: weight must be rank 4 [Cout, Cin/groups, Kh, Kw], got {w_dims:?}",
        );
        let (n, cin, h_in, w_in) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
        let (cout, cin_per_g, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
        assert_eq!(
            cin, cin_per_g * groups,
            "conv2d: x has {cin} in-channels but weight expects {} ({}·{groups})",
            cin_per_g * groups, cin_per_g,
        );
        assert_eq!(
            cout % groups, 0,
            "conv2d: Cout={cout} must be divisible by groups={groups}",
        );
        if let Some(b) = bias {
            let b_dims = b.shape();
            let b_dims = b_dims.dims();
            assert_eq!(
                b_dims, &[cout],
                "conv2d: bias shape {b_dims:?} must match [Cout={cout}]",
            );
        }
        // Hout = (H + 2·pad.0 − Kh) / stride.0 + 1
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        assert!(stride_h >= 1 && stride_w >= 1, "conv2d: stride must be ≥ 1");
        let h_padded = h_in + 2 * pad_h;
        let w_padded = w_in + 2 * pad_w;
        assert!(
            h_padded >= kh && w_padded >= kw,
            "conv2d: padded input ({h_padded}×{w_padded}) smaller than kernel ({kh}×{kw})",
        );
        let h_out = (h_padded - kh) / stride_h + 1;
        let w_out = (w_padded - kw) / stride_w + 1;
        let dtype = self.dtype();
        let mut inputs = vec![self.id, weight.id];
        if let Some(b) = bias {
            inputs.push(b.id);
        }
        let id = self.graph.borrow_mut().push(Node {
            op: Op::Conv2D { stride, padding, groups },
            inputs,
            shape: Shape::from_dims(&[n, cout, h_out, w_out]),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `Relu` node `max(0, self)`.
    pub fn relu(&self) -> Tensor {
        self.unary_op(Op::Relu)
    }

    /// Append a `Sqr` node `self * self`.
    pub fn sqr(&self) -> Tensor {
        self.unary_op(Op::Sqr)
    }

    /// Append an `Exp` node `e^self`.
    pub fn exp(&self) -> Tensor {
        self.unary_op(Op::Exp)
    }

    /// Append a `Permute` node rearranging the axes according to `axes`.
    /// `out.shape[i] = self.shape[axes[i]]`. The axes vector must be a
    /// permutation of `0..rank`. For a rank-3 input with `axes = [2, 0, 1]`,
    /// the shape transform is `[a, b, c] → [c, a, b]`.
    pub fn permute(&self, axes: &[usize]) -> Tensor {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        let rank = in_dims.len();
        assert_eq!(
            axes.len(),
            rank,
            "permute: axes length {} must equal tensor rank {}",
            axes.len(),
            rank,
        );
        // Validate that axes is a permutation of 0..rank.
        let mut seen = vec![false; rank];
        for &ax in axes {
            assert!(ax < rank, "permute: axis {ax} out of bounds for rank {rank}");
            assert!(!seen[ax], "permute: duplicate axis {ax} in axes");
            seen[ax] = true;
        }
        let out_dims: Vec<usize> = axes.iter().map(|&ax| in_dims[ax]).collect();
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op: Op::Permute(axes.to_vec()),
            inputs: vec![self.id],
            shape: Shape::from_dims(&out_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `Transpose` node that swaps the last two dimensions of a
    /// tensor of rank ≥ 2. Leading dims are unchanged. For a rank-2
    /// tensor this is the ordinary matrix transpose; for higher ranks,
    /// every batch slice is transposed independently.
    pub fn transpose(&self) -> Tensor {
        let in_dims = self.shape();
        let d = in_dims.dims();
        assert!(
            d.len() >= 2,
            "transpose: input must be rank ≥ 2, got shape {d:?}",
        );
        let rank = d.len();
        let mut out: Vec<usize> = d.to_vec();
        out.swap(rank - 2, rank - 1);
        let out_shape = Shape::from_dims(&out);
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op: Op::Transpose,
            inputs: vec![self.id],
            shape: out_shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    // --- additional element-wise binary ops ---

    /// Append a `Sub` node `self - other`. Requires matching shapes and dtypes.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.binary_op("sub", Op::Sub, other, self.shape())
    }

    /// Append a `Div` node `self / other`. Requires matching shapes and dtypes.
    pub fn div(&self, other: &Tensor) -> Tensor {
        self.binary_op("div", Op::Div, other, self.shape())
    }

    // --- additional element-wise unary ops ---

    /// Append a `Neg` node `-self`.
    pub fn neg(&self) -> Tensor {
        self.unary_op(Op::Neg)
    }

    /// Append a `Sqrt` node `sqrt(self)`.
    pub fn sqrt(&self) -> Tensor {
        self.unary_op(Op::Sqrt)
    }

    /// Append a `Log` node `ln(self)`.
    pub fn log(&self) -> Tensor {
        self.unary_op(Op::Log)
    }

    /// Append a `Sin` node.
    pub fn sin(&self) -> Tensor {
        self.unary_op(Op::Sin)
    }

    /// Append a `Cos` node.
    pub fn cos(&self) -> Tensor {
        self.unary_op(Op::Cos)
    }

    /// Append a `Tanh` node.
    pub fn tanh(&self) -> Tensor {
        self.unary_op(Op::Tanh)
    }

    /// Append a `Sigmoid` node.
    pub fn sigmoid(&self) -> Tensor {
        self.unary_op(Op::Sigmoid)
    }

    /// Append a `Silu` node (SiLU/Swish activation: `x · sigmoid(x)`).
    pub fn silu(&self) -> Tensor {
        self.unary_op(Op::Silu)
    }

    /// Append a `Gelu` node (tanh-approximation GELU).
    pub fn gelu(&self) -> Tensor {
        self.unary_op(Op::Gelu)
    }

    /// Append a `Step` node (Heaviside step: `1` where `x > 0`, else `0`).
    pub fn step(&self) -> Tensor {
        self.unary_op(Op::Step)
    }

    // --- dtype and broadcasting ---

    /// Append a `Cast` node converting this tensor's element type to
    /// `target`. Shape is preserved.
    pub fn cast(&self, target: DType) -> Tensor {
        let shape = self.shape();
        let id = self.graph.borrow_mut().push(Node {
            op: Op::Cast(target),
            inputs: vec![self.id],
            shape,
            dtype: target,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `BroadcastTo` node that expands `self` to the given target
    /// shape using NumPy broadcasting rules (right-align, pad with 1s,
    /// expand size-1 dims). The new tensor has the target shape and the
    /// same dtype as `self`.
    pub fn broadcast_to(&self, target: impl Into<Shape>) -> Tensor {
        let target = target.into();
        let src_dims = self.shape();
        // Validate broadcast compatibility up front so users hear about
        // bad target shapes at build time, not realize time.
        check_broadcast_compatible(src_dims.dims(), target.dims());
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op: Op::BroadcastTo(target.clone()),
            inputs: vec![self.id],
            shape: target,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `Reshape` node producing `self`'s data under a new shape.
    /// The new shape must have the same total element count as the
    /// current shape.
    pub fn reshape(&self, target: impl Into<Shape>) -> Tensor {
        let target = target.into();
        assert_eq!(
            self.shape().elem_count(),
            target.elem_count(),
            "reshape: element count mismatch: from {} to {}",
            self.shape().elem_count(),
            target.elem_count(),
        );
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op: Op::Reshape(target.clone()),
            inputs: vec![self.id],
            shape: target,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `ReduceSumTo` node that sum-reduces `self` to a smaller
    /// shape. The target must be reachable from `self.shape()` via
    /// reduction of dims (i.e. `self.shape()` could be produced from
    /// `target` by broadcasting).
    pub fn reduce_sum_to(&self, target: impl Into<Shape>) -> Tensor {
        let target = target.into();
        // Symmetric check: self must be broadcast-compatible FROM the
        // smaller target. That is, the target must be broadcast-ready to
        // self.shape().
        check_broadcast_compatible(target.dims(), self.shape().dims());
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op: Op::ReduceSumTo(target.clone()),
            inputs: vec![self.id],
            shape: target,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    // --- reductions to a scalar ---

    /// Append a `SumAll` node reducing every element to a scalar (rank-0).
    pub fn sum_all(&self) -> Tensor {
        self.scalar_reduction(Op::SumAll)
    }

    /// Append a `MaxAll` node producing the max element as a scalar.
    pub fn max_all(&self) -> Tensor {
        self.scalar_reduction(Op::MaxAll)
    }

    /// Append a `MinAll` node producing the min element as a scalar.
    pub fn min_all(&self) -> Tensor {
        self.scalar_reduction(Op::MinAll)
    }

    /// Append a `MeanAll` node producing the arithmetic mean as a scalar.
    pub fn mean_all(&self) -> Tensor {
        self.scalar_reduction(Op::MeanAll)
    }

    // --- reductions along one dimension ---

    /// Append a `SumDim(dim)` node. Reduces along `dim`; output rank is
    /// `input rank - 1` (the reduced dim is removed).
    pub fn sum_dim(&self, dim: usize) -> Tensor {
        self.axis_reduction("sum_dim", Op::SumDim(dim), dim)
    }

    /// Append a `MaxDim(dim)` node.
    pub fn max_dim(&self, dim: usize) -> Tensor {
        self.axis_reduction("max_dim", Op::MaxDim(dim), dim)
    }

    /// Append a `MinDim(dim)` node.
    pub fn min_dim(&self, dim: usize) -> Tensor {
        self.axis_reduction("min_dim", Op::MinDim(dim), dim)
    }

    /// Append a `MeanDim(dim)` node.
    pub fn mean_dim(&self, dim: usize) -> Tensor {
        self.axis_reduction("mean_dim", Op::MeanDim(dim), dim)
    }

    /// Append an `ArgMaxDim(dim)` node. Produces a U32 tensor with the
    /// reduced dim removed, whose values are the index of the maximum
    /// along that dim. Non-differentiable — trying to run `backward()`
    /// through an ArgMax node will panic.
    pub fn argmax_dim(&self, dim: usize) -> Tensor {
        self.index_reduction("argmax_dim", Op::ArgMaxDim(dim), dim)
    }

    /// Append an `ArgMinDim(dim)` node. Same semantics as `argmax_dim`
    /// but for the minimum.
    pub fn argmin_dim(&self, dim: usize) -> Tensor {
        self.index_reduction("argmin_dim", Op::ArgMinDim(dim), dim)
    }

    /// Internal helper for index-producing reductions. Validates the
    /// dim, builds the reduced shape, and stamps the output with
    /// `DType::U32` regardless of the input dtype.
    fn index_reduction(&self, name: &'static str, op: Op, dim: usize) -> Tensor {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        assert!(
            dim < in_dims.len(),
            "{name}: dim {dim} out of bounds for shape {in_dims:?}",
        );
        let out_dims: Vec<usize> = in_dims
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dim)
            .map(|(_, &d)| d)
            .collect();
        let id = self.graph.borrow_mut().push(Node {
            op,
            inputs: vec![self.id],
            shape: Shape::from_dims(&out_dims),
            dtype: DType::U32,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    // --- compositions ---

    /// Append a `SoftmaxLastDim` node. Shape is preserved.
    pub fn softmax_last_dim(&self) -> Tensor {
        assert!(
            !self.shape().dims().is_empty(),
            "softmax_last_dim: input must be rank >= 1",
        );
        self.unary_op(Op::SoftmaxLastDim)
    }

    /// Append a `LayerNormLastDim` node with the given epsilon. Shape is
    /// preserved.
    pub fn layer_norm_last_dim(&self, eps: f64) -> Tensor {
        let dims = self.shape();
        let d = dims.dims();
        assert!(
            !d.is_empty() && *d.last().unwrap() > 0,
            "layer_norm_last_dim: input must have a non-zero last dim, got {d:?}",
        );
        self.unary_op(Op::LayerNormLastDim { eps })
    }

    /// Apply rotary position embeddings (RoPE) to `self`. The input is
    /// expected to have shape `[..., seq, d]` where `seq` is the
    /// sequence length and `d` is the per-head feature dim (which must
    /// be even). `base` is the frequency base — 10000.0 for LLaMA.
    /// `start_pos` lets callers apply RoPE at a non-zero offset, which
    /// is what KV-cached inference does when the query for position
    /// `n + k` is processed against a cache that already holds
    /// positions `0..n+k`.
    ///
    /// Uses the half-split layout (the one LLaMA and HuggingFace use):
    /// pair feature `i` with feature `i + d/2`. Under that layout the
    /// rotation reduces to the identity `x' = x·cos + rotate_half(x)·sin`
    /// where `rotate_half(x)` concatenates `[-x[d/2:], x[:d/2]]` along
    /// the last dim.
    ///
    /// Implemented as a composition of existing graph primitives —
    /// `Slice`, `Neg`, `Concat`, `Reshape`, `BroadcastTo`, `Mul`, `Add`
    /// — plus two precomputed `Const` tensors for the cos/sin frequency
    /// tables. Every primitive already has a working backward rule, so
    /// gradients flow through RoPE without any new backward machinery.
    ///
    /// Only `f32` is supported today. Extending to other float dtypes
    /// is mechanical: build the frequency tables in the target dtype.
    pub fn rope(&self, base: f64, start_pos: usize) -> Tensor {
        let (seq, d) = {
            let dims = self.shape();
            let v = dims.dims();
            let rank = v.len();
            assert!(rank >= 2, "rope: input must have rank ≥ 2");
            (v[rank - 2], v[rank - 1])
        };
        let (cos, sin) = build_rope_tables(base, start_pos, seq, d);
        let cos_t = self.const_f32_like(cos, Shape::from_dims(&[seq, d]));
        let sin_t = self.const_f32_like(sin, Shape::from_dims(&[seq, d]));
        self.rope_with_tables(&cos_t, &sin_t)
    }

    /// Apply RoPE using caller-supplied `cos` and `sin` tables. Each
    /// table has shape `[seq, head_dim]` and matches the layout
    /// [`build_rope_tables`] produces. This is the hot-path entry
    /// point: in a transformer forward pass every attention layer
    /// applies RoPE to Q and K with the *same* `(start_pos, seq,
    /// head_dim)`, so the caller can build the tables once and share
    /// the const nodes across all layers rather than re-duplicating
    /// them inside each `.rope()` call. The classic [`rope`] entry
    /// point funnels through this after building the tables itself.
    pub fn rope_with_tables(&self, cos: &Tensor, sin: &Tensor) -> Tensor {
        assert_eq!(
            self.dtype(),
            DType::F32,
            "rope: only f32 is supported today (cast explicitly for other dtypes)",
        );
        let in_shape = self.shape();
        let dims_vec: Vec<usize> = in_shape.dims().to_vec();
        let rank = dims_vec.len();
        assert!(rank >= 2, "rope: input must have rank ≥ 2, got {dims_vec:?}");
        let seq = dims_vec[rank - 2];
        let d = dims_vec[rank - 1];
        assert!(
            d.is_multiple_of(2),
            "rope: feature dim {d} must be even",
        );
        let cos_shape = cos.shape();
        let sin_shape = sin.shape();
        assert_eq!(
            cos_shape.dims(),
            &[seq, d],
            "rope_with_tables: cos shape {:?} does not match [seq, d] = [{seq}, {d}]",
            cos_shape.dims(),
        );
        assert_eq!(
            sin_shape.dims(),
            &[seq, d],
            "rope_with_tables: sin shape {:?} does not match [seq, d] = [{seq}, {d}]",
            sin_shape.dims(),
        );
        let _ = rank;
        // Emit a single fused Op::Rope node. The decomposed version
        // (slice+neg+concat+broadcast+mul+add) produces ~72 dispatches
        // on GPU backends because concat-along-last-dim has a per-row
        // host loop. Fused path dispatches once.
        let out_shape = in_shape.clone();
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op: Op::Rope,
            inputs: vec![self.id, cos.id, sin.id],
            shape: out_shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// The pre-fused decomposition of `rope_with_tables`. Retained so
    /// backends without a native `Op::Rope` kernel can synthesize from
    /// primitives, and so correctness tests can cross-check the fused
    /// path against the primitive path.
    #[doc(hidden)]
    pub fn rope_with_tables_decomposed(&self, cos: &Tensor, sin: &Tensor) -> Tensor {
        let in_shape = self.shape();
        let dims_vec: Vec<usize> = in_shape.dims().to_vec();
        let rank = dims_vec.len();
        let seq = dims_vec[rank - 2];
        let d = dims_vec[rank - 1];
        let half = d / 2;

        let mut broadcast_shape: Vec<usize> = vec![1_usize; rank];
        broadcast_shape[rank - 2] = seq;
        broadcast_shape[rank - 1] = d;
        let cos_reshaped = cos.reshape(Shape::from_dims(&broadcast_shape));
        let sin_reshaped = sin.reshape(Shape::from_dims(&broadcast_shape));
        let cos_bcast = cos_reshaped.broadcast_to(in_shape.clone());
        let sin_bcast = sin_reshaped.broadcast_to(in_shape);

        let first_half = self.slice(rank - 1, 0, half);
        let second_half = self.slice(rank - 1, half, half);
        let neg_second = second_half.neg();
        let rotated_half = neg_second.concat(&first_half, rank - 1);

        let left = self.mul(&cos_bcast);
        let right = rotated_half.mul(&sin_bcast);
        left.add(&right)
    }

    /// Root-Mean-Square Normalization along the last dim, without
    /// affine parameters. `y = x / sqrt(mean(x², last) + eps)`.
    ///
    /// This is the normalization layer used by LLaMA-family models.
    /// Unlike LayerNorm, RmsNorm does not subtract the mean — it only
    /// divides by the root-mean-square, which is both cheaper and
    /// empirically just as effective in transformer blocks.
    ///
    /// Emits a single fused `Op::RmsNormLastDim { eps }` node.
    /// Backends that have a native implementation dispatch it as one
    /// kernel; ones that don't get a CPU fallback via the reference
    /// implementation.
    ///
    /// Use [`rms_norm_last_dim_decomposed`](Self::rms_norm_last_dim_decomposed)
    /// instead if you need to differentiate through this op —
    /// backward through the fused op is not yet implemented.
    pub fn rms_norm_last_dim(&self, eps: f64) -> Tensor {
        let x_dims_vec: Vec<usize> = self.shape().dims().to_vec();
        assert!(
            !x_dims_vec.is_empty() && *x_dims_vec.last().unwrap() > 0,
            "rms_norm_last_dim: input must have non-zero last dim, got {x_dims_vec:?}",
        );
        self.unary_op(Op::RmsNormLastDim { eps })
    }

    /// Decomposed version — emits the (sqr → mean_dim → reshape →
    /// add_scalar → sqrt → broadcast_to → div) primitive chain. Has
    /// working backward rules through its primitive subgraph. Use
    /// this when you need to differentiate through RMSNorm; use
    /// [`rms_norm_last_dim`](Self::rms_norm_last_dim) for inference.
    pub fn rms_norm_last_dim_decomposed(&self, eps: f64) -> Tensor {
        let x_shape = self.shape();
        let x_dims_vec: Vec<usize> = x_shape.dims().to_vec();
        assert!(
            !x_dims_vec.is_empty() && *x_dims_vec.last().unwrap() > 0,
            "rms_norm_last_dim_decomposed: input must have non-zero last dim, got {x_dims_vec:?}",
        );
        let last = x_dims_vec.len() - 1;
        let mean_sq = self.sqr().mean_dim(last);
        let mut keepdim_dims = x_dims_vec.clone();
        keepdim_dims[last] = 1;
        let mean_sq_kd = mean_sq.reshape(Shape::from_dims(&keepdim_dims));
        let denom = mean_sq_kd.add_scalar(eps).sqrt();
        let denom_bcast = denom.broadcast_to(x_shape);
        self.div(&denom_bcast)
    }

    // --- indexing ---

    /// Append an `IndexSelect` node that picks slices from `self` along
    /// `dim` using a 1-D `u32` index tensor. The output has the same
    /// shape as `self` except dimension `dim` is replaced by
    /// `indices.shape()[0]`.
    pub fn index_select(&self, dim: usize, indices: &Tensor) -> Tensor {
        assert!(
            Rc::ptr_eq(&self.graph, &indices.graph),
            "index_select: data and index tensors must live on the same graph",
        );
        assert_eq!(
            indices.dtype(),
            DType::U32,
            "index_select: index tensor must be U32, got {:?}",
            indices.dtype(),
        );
        let idx_dims = indices.shape();
        assert_eq!(
            idx_dims.dims().len(),
            1,
            "index_select: index tensor must be rank 1, got {:?}",
            idx_dims.dims(),
        );
        let data_dims = self.shape();
        assert!(
            dim < data_dims.dims().len(),
            "index_select: dim {dim} out of bounds for data shape {:?}",
            data_dims.dims(),
        );
        let mut out_dims: Vec<usize> = data_dims.dims().to_vec();
        out_dims[dim] = idx_dims.dims()[0];
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op: Op::IndexSelect { dim },
            inputs: vec![self.id, indices.id],
            shape: Shape::from_dims(&out_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `Gather` node that performs an N-dimensional gather along
    /// `dim`. The `indices` tensor must be `U32` with the same rank as
    /// `self`. Output shape equals `indices.shape()`.
    pub fn gather(&self, dim: usize, indices: &Tensor) -> Tensor {
        assert!(
            Rc::ptr_eq(&self.graph, &indices.graph),
            "gather: data and index tensors must live on the same graph",
        );
        assert_eq!(
            indices.dtype(),
            DType::U32,
            "gather: index tensor must be U32, got {:?}",
            indices.dtype(),
        );
        let data_rank = self.shape().dims().len();
        let idx_rank = indices.shape().dims().len();
        assert_eq!(
            data_rank, idx_rank,
            "gather: data and index must have the same rank, got {data_rank} vs {idx_rank}",
        );
        assert!(
            dim < data_rank,
            "gather: dim {dim} out of bounds for rank {data_rank}",
        );
        let dtype = self.dtype();
        let out_shape = indices.shape();
        let id = self.graph.borrow_mut().push(Node {
            op: Op::Gather { dim },
            inputs: vec![self.id, indices.id],
            shape: out_shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `Concat` node joining `self` and `other` along `dim`.
    pub fn concat(&self, other: &Tensor, dim: usize) -> Tensor {
        assert!(
            Rc::ptr_eq(&self.graph, &other.graph),
            "concat: tensors must live on the same graph",
        );
        assert_eq!(
            self.dtype(),
            other.dtype(),
            "concat: dtype mismatch {:?} vs {:?}",
            self.dtype(),
            other.dtype(),
        );
        let a = self.shape();
        let b = other.shape();
        let ad = a.dims();
        let bd = b.dims();
        assert_eq!(ad.len(), bd.len(), "concat: rank mismatch");
        assert!(dim < ad.len(), "concat: dim out of bounds");
        for i in 0..ad.len() {
            if i != dim {
                assert_eq!(
                    ad[i], bd[i],
                    "concat: non-dim shapes must match (dim {i}: {} vs {})",
                    ad[i], bd[i],
                );
            }
        }
        let mut out_dims: Vec<usize> = ad.to_vec();
        out_dims[dim] = ad[dim] + bd[dim];
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op: Op::Concat { dim },
            inputs: vec![self.id, other.id],
            shape: Shape::from_dims(&out_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `Slice` (narrow) node taking elements `[start, start+len)`
    /// along `dim`. The output has the same rank; only `dim` shrinks.
    pub fn slice(&self, dim: usize, start: usize, len: usize) -> Tensor {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        assert!(dim < in_dims.len(), "slice: dim out of bounds");
        assert!(
            start + len <= in_dims[dim],
            "slice: [start={start}, len={len}) exceeds dim size {}",
            in_dims[dim],
        );
        let mut out_dims: Vec<usize> = in_dims.to_vec();
        out_dims[dim] = len;
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op: Op::Slice { dim, start, len },
            inputs: vec![self.id],
            shape: Shape::from_dims(&out_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append an `AddScalar` node: `y[i] = x[i] + c`. Convenient for
    /// avoiding a full-shape const node when you just want to bias a
    /// tensor by a constant.
    pub fn add_scalar(&self, c: f64) -> Tensor {
        self.unary_op(Op::AddScalar(c))
    }

    /// Append a `MulScalar` node: `y[i] = x[i] * c`.
    pub fn mul_scalar(&self, c: f64) -> Tensor {
        self.unary_op(Op::MulScalar(c))
    }

    /// Append a `PowI` node raising each element to an integer power.
    pub fn powi(&self, n: i32) -> Tensor {
        self.unary_op(Op::PowI(n))
    }

    /// Append a `Clamp` node restricting each element to `[min, max]`.
    pub fn clamp(&self, min: f64, max: f64) -> Tensor {
        assert!(
            min <= max,
            "clamp: min ({min}) must be <= max ({max})",
        );
        self.unary_op(Op::Clamp { min, max })
    }

    /// Append a `Maximum` node `max(self, other)` element-wise. Matching
    /// shapes required.
    pub fn maximum(&self, other: &Tensor) -> Tensor {
        self.binary_op("maximum", Op::Maximum, other, self.shape())
    }

    /// Append a `Minimum` node `min(self, other)` element-wise.
    pub fn minimum(&self, other: &Tensor) -> Tensor {
        self.binary_op("minimum", Op::Minimum, other, self.shape())
    }

    /// Element-wise addition with automatic broadcasting. Unlike `add`,
    /// which requires matching shapes, `broadcast_add` computes the
    /// broadcast shape of the two operands, inserts explicit
    /// `BroadcastTo` nodes as needed, and then emits a regular `Add`.
    /// Useful for bias addition (`[batch, hidden] + [hidden]`) without
    /// the caller writing the broadcast out explicitly.
    pub fn broadcast_add(&self, other: &Tensor) -> Tensor {
        let (a, b) = self.auto_broadcast_pair("broadcast_add", other);
        a.add(&b)
    }

    /// Element-wise subtraction with automatic broadcasting.
    pub fn broadcast_sub(&self, other: &Tensor) -> Tensor {
        let (a, b) = self.auto_broadcast_pair("broadcast_sub", other);
        a.sub(&b)
    }

    /// Element-wise multiplication with automatic broadcasting.
    pub fn broadcast_mul(&self, other: &Tensor) -> Tensor {
        let (a, b) = self.auto_broadcast_pair("broadcast_mul", other);
        a.mul(&b)
    }

    /// Element-wise division with automatic broadcasting.
    pub fn broadcast_div(&self, other: &Tensor) -> Tensor {
        let (a, b) = self.auto_broadcast_pair("broadcast_div", other);
        a.div(&b)
    }

    /// Build both operands broadcast to their common shape. Used by the
    /// `broadcast_*` wrappers above. Inserts explicit `BroadcastTo` nodes
    /// on whichever side needs them — either or both may pass through
    /// unchanged if already at the target shape.
    fn auto_broadcast_pair(&self, op: &'static str, other: &Tensor) -> (Tensor, Tensor) {
        assert!(
            Rc::ptr_eq(&self.graph, &other.graph),
            "{op}: tensors must live on the same graph",
        );
        assert_eq!(
            self.dtype(),
            other.dtype(),
            "{op}: dtype mismatch {:?} vs {:?}",
            self.dtype(),
            other.dtype(),
        );
        let a_shape = self.shape();
        let b_shape = other.shape();
        let target_dims = compute_broadcast_shape(a_shape.dims(), b_shape.dims());
        let target = Shape::from_dims(&target_dims);
        let a = if a_shape.dims() == target.dims() {
            self.clone()
        } else {
            self.broadcast_to(target.clone())
        };
        let b = if b_shape.dims() == target.dims() {
            other.clone()
        } else {
            other.broadcast_to(target)
        };
        (a, b)
    }

    /// Append an `IndexAdd` node — the functional inverse of
    /// `IndexSelect`. Returns `base` with `src` added at positions given
    /// by `indices` along `dim`.
    pub fn index_add(&self, dim: usize, indices: &Tensor, src: &Tensor) -> Tensor {
        assert!(
            Rc::ptr_eq(&self.graph, &indices.graph) && Rc::ptr_eq(&self.graph, &src.graph),
            "index_add: all tensors must live on the same graph",
        );
        assert_eq!(indices.dtype(), DType::U32, "index_add: index must be U32");
        assert_eq!(self.dtype(), src.dtype(), "index_add: base and src dtypes must match");
        let base_dims = self.shape();
        let src_dims = src.shape();
        assert_eq!(
            base_dims.dims().len(),
            src_dims.dims().len(),
            "index_add: base and src must have the same rank",
        );
        assert_eq!(
            indices.shape().dims().len(),
            1,
            "index_add: index must be rank 1",
        );
        assert_eq!(
            src_dims.dims()[dim],
            indices.shape().dims()[0],
            "index_add: src dim {dim} ({}) must match index length ({})",
            src_dims.dims()[dim],
            indices.shape().dims()[0],
        );
        let dtype = self.dtype();
        let out_shape = base_dims;
        let id = self.graph.borrow_mut().push(Node {
            op: Op::IndexAdd { dim },
            inputs: vec![self.id, indices.id, src.id],
            shape: out_shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `ScatterAdd` node — the functional inverse of `Gather`.
    /// Returns `base` with values from `src` accumulated at positions
    /// given by `indices` (with `indices[p]` substituted at `dim`).
    pub fn scatter_add(&self, dim: usize, indices: &Tensor, src: &Tensor) -> Tensor {
        assert!(
            Rc::ptr_eq(&self.graph, &indices.graph) && Rc::ptr_eq(&self.graph, &src.graph),
            "scatter_add: all tensors must live on the same graph",
        );
        assert_eq!(indices.dtype(), DType::U32, "scatter_add: index must be U32");
        assert_eq!(self.dtype(), src.dtype(), "scatter_add: base and src dtypes must match");
        assert_eq!(
            indices.shape().dims(),
            src.shape().dims(),
            "scatter_add: index and src must have the same shape",
        );
        let dtype = self.dtype();
        let out_shape = self.shape();
        let id = self.graph.borrow_mut().push(Node {
            op: Op::ScatterAdd { dim },
            inputs: vec![self.id, indices.id, src.id],
            shape: out_shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    // --- internal helpers for the new builders ---

    fn scalar_reduction(&self, op: Op) -> Tensor {
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op,
            inputs: vec![self.id],
            shape: Shape::from_dims(&[]),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    fn axis_reduction(&self, name: &'static str, op: Op, dim: usize) -> Tensor {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        assert!(
            dim < in_dims.len(),
            "{name}: dim {dim} out of bounds for shape {in_dims:?}",
        );
        let out_dims: Vec<usize> = in_dims
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dim)
            .map(|(_, &d)| d)
            .collect();
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op,
            inputs: vec![self.id],
            shape: Shape::from_dims(&out_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    // Internal helpers. Both shape-validate and append.

    fn binary_op(&self, name: &'static str, op: Op, other: &Tensor, out_shape: Shape) -> Tensor {
        assert!(
            Rc::ptr_eq(&self.graph, &other.graph),
            "{name}: tensors must live on the same graph",
        );
        assert_eq!(
            self.dtype(),
            other.dtype(),
            "{name}: dtype mismatch: lhs={:?}, rhs={:?}",
            self.dtype(),
            other.dtype(),
        );
        assert_eq!(
            self.shape().dims(),
            other.shape().dims(),
            "{name}: shape mismatch: lhs={:?}, rhs={:?}",
            self.shape().dims(),
            other.shape().dims(),
        );
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op,
            inputs: vec![self.id, other.id],
            shape: out_shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    fn unary_op(&self, op: Op) -> Tensor {
        let shape = self.shape();
        let dtype = self.dtype();
        let id = self.graph.borrow_mut().push(Node {
            op,
            inputs: vec![self.id],
            shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Build the backward graph from `self`, treating `self` as the output
    /// whose gradient drives the rest of the computation.
    ///
    /// Walks the forward graph in reverse topological order, applies the
    /// per-op gradient rule to each node, and emits new graph nodes for
    /// the gradient with respect to every node reachable from `self`. The
    /// returned [`GradMap`] lets you look up the gradient tensor for any
    /// forward tensor from the same graph.
    ///
    /// The initial upstream gradient is a ones tensor of the same shape
    /// and dtype as `self` — the conventional `dL/dL = 1` seed for
    /// backprop. For a scalar loss this is `[1.0]`; for a non-scalar
    /// output it is "each element of the output contributes with weight 1
    /// to the total derivative."
    ///
    /// ## Supported ops in the MVP
    ///
    /// `Const`, `Add`, `Mul`, `MatMul`, `Transpose`, `Sqr`, `Exp` all have
    /// gradient rules implemented. `Relu` panics — its gradient needs a
    /// `Step`/`Where`/`Sign` op that is not yet in the catalog and will
    /// land in a follow-up once an indicator-valued op is defined.
    ///
    /// ## Ownership and the forward graph
    ///
    /// Backward extends the same graph that held the forward pass. The
    /// gradient nodes live alongside the forward nodes in one arena. After
    /// `backward()`, realizing a gradient tensor re-executes whatever part
    /// of the forward pass the gradient depends on — the executor walks
    /// the combined graph as one.
    pub fn backward(&self) -> GradMap {
        let graph_handle = self.graph.clone();

        // Compute topological order of reachable nodes, then reverse it
        // so we walk from the root (output) toward the leaves (inputs).
        let mut order = topo_order(&graph_handle.borrow(), self.id);
        order.reverse();

        // Initial upstream gradient for the root: ones tensor of matching
        // shape and dtype. For a scalar loss this is [1.0]; for vector
        // outputs it seeds each element with weight 1.
        let (root_shape, root_dtype) = {
            let g = graph_handle.borrow();
            let n = g.node(self.id);
            (n.shape.clone(), n.dtype)
        };
        let ones_id = build_ones(&graph_handle, root_shape, root_dtype);

        let mut upstream: HashMap<NodeId, NodeId> = HashMap::new();
        upstream.insert(self.id, ones_id);

        for id in order {
            let up_id = match upstream.get(&id).copied() {
                Some(v) => v,
                None => continue, // unreachable from the root, no gradient to propagate
            };
            // Snapshot the op and its input IDs so we can drop the read
            // borrow before taking a mutable borrow to append new nodes.
            let (op, inputs) = {
                let g = graph_handle.borrow();
                let node = g.node(id);
                (node.op.clone(), node.inputs.clone())
            };

            match op {
                Op::Const(_) => {
                    // Leaf. The upstream value has already been stored and
                    // is what `GradMap::get` will return for this input.
                }
                Op::Add => {
                    // d(a + b)/da = 1, d(a + b)/db = 1.
                    // Upstream flows unchanged into both inputs.
                    accumulate_grad(&mut upstream, inputs[0], up_id, &graph_handle);
                    accumulate_grad(&mut upstream, inputs[1], up_id, &graph_handle);
                }
                Op::Mul => {
                    // d(a * b)/da = b, d(a * b)/db = a.
                    // Upstream * b goes to a; upstream * a goes to b.
                    let a = inputs[0];
                    let b = inputs[1];
                    let a_shape = node_shape(&graph_handle, a);
                    let b_shape = node_shape(&graph_handle, b);
                    let dtype = node_dtype(&graph_handle, a);
                    let grad_a = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, b],
                        a_shape,
                        dtype,
                    );
                    let grad_b = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, a],
                        b_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::MatMul => {
                    // Forward: Y = A @ B,  A: [m, k], B: [k, n], Y: [m, n]
                    // Backward: dA = dY @ B^T,  dB = A^T @ dY.
                    let a = inputs[0];
                    let b = inputs[1];
                    let a_shape = node_shape(&graph_handle, a);
                    let b_shape = node_shape(&graph_handle, b);
                    let dtype = node_dtype(&graph_handle, a);
                    // B^T: shape [n, k]
                    let b_t_shape = transposed_shape(&b_shape);
                    let b_t = push_node(
                        &graph_handle,
                        Op::Transpose,
                        vec![b],
                        b_t_shape,
                        dtype,
                    );
                    // dA = upstream @ B^T, shape [m, k] = a_shape
                    let grad_a = push_node(
                        &graph_handle,
                        Op::MatMul,
                        vec![up_id, b_t],
                        a_shape.clone(),
                        dtype,
                    );
                    // A^T: shape [k, m]
                    let a_t_shape = transposed_shape(&a_shape);
                    let a_t = push_node(
                        &graph_handle,
                        Op::Transpose,
                        vec![a],
                        a_t_shape,
                        dtype,
                    );
                    // dB = A^T @ upstream, shape [k, n] = b_shape
                    let grad_b = push_node(
                        &graph_handle,
                        Op::MatMul,
                        vec![a_t, up_id],
                        b_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::Transpose => {
                    // d(x^T)/dx is a transpose on the way back.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Transpose,
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Permute(ref axes) => {
                    // Backward: apply the inverse permutation to upstream.
                    // If forward permutes with `axes` (out[i] = in[axes[i]]),
                    // the inverse `inv` satisfies `inv[axes[i]] = i`.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let mut inv = vec![0_usize; axes.len()];
                    for (i, &ax) in axes.iter().enumerate() {
                        inv[ax] = i;
                    }
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Permute(inv),
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Sqr => {
                    // d(x^2)/dx = 2x. Expressed as `x + x` to avoid
                    // needing a scalar constant broadcast — both sums
                    // have the same shape as x, the existing Add op
                    // handles them natively.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let two_x = push_node(
                        &graph_handle,
                        Op::Add,
                        vec![x, x],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, two_x],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Exp => {
                    // d(e^x)/dx = e^x, which is the forward output — i.e.
                    // this node `id`. Reuse it directly instead of
                    // recomputing.
                    let x = inputs[0];
                    let out_shape = node_shape(&graph_handle, id);
                    let dtype = node_dtype(&graph_handle, id);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, id],
                        out_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Sub => {
                    // d(a - b)/da = 1, d(a - b)/db = -1.
                    let a = inputs[0];
                    let b = inputs[1];
                    accumulate_grad(&mut upstream, a, up_id, &graph_handle);
                    let b_shape = node_shape(&graph_handle, b);
                    let dtype = node_dtype(&graph_handle, b);
                    let neg_up = push_node(
                        &graph_handle,
                        Op::Neg,
                        vec![up_id],
                        b_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, b, neg_up, &graph_handle);
                }
                Op::Div => {
                    // y = a / b
                    // dy/da = 1/b     →  grad_a = upstream / b
                    // dy/db = -a/b²   →  grad_b = -(upstream * a / (b*b))
                    let a = inputs[0];
                    let b = inputs[1];
                    let a_shape = node_shape(&graph_handle, a);
                    let b_shape = node_shape(&graph_handle, b);
                    let dtype = node_dtype(&graph_handle, a);
                    // grad_a
                    let grad_a = push_node(
                        &graph_handle,
                        Op::Div,
                        vec![up_id, b],
                        a_shape.clone(),
                        dtype,
                    );
                    accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                    // grad_b: -(upstream * a / (b*b))
                    let b_sq = push_node(
                        &graph_handle,
                        Op::Sqr,
                        vec![b],
                        b_shape.clone(),
                        dtype,
                    );
                    let up_times_a = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, a],
                        a_shape,
                        dtype,
                    );
                    let quotient = push_node(
                        &graph_handle,
                        Op::Div,
                        vec![up_times_a, b_sq],
                        b_shape.clone(),
                        dtype,
                    );
                    let grad_b = push_node(
                        &graph_handle,
                        Op::Neg,
                        vec![quotient],
                        b_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::Neg => {
                    // d(-x)/dx = -1. grad_x = -upstream.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Neg,
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Sqrt => {
                    // y = sqrt(x), dy/dx = 1/(2*sqrt(x)) = 1/(2y).
                    // grad_x = upstream / (y + y). Reuse the forward node.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let two_y = push_node(
                        &graph_handle,
                        Op::Add,
                        vec![id, id],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Div,
                        vec![up_id, two_y],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Log => {
                    // d(ln(x))/dx = 1/x. grad_x = upstream / x.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Div,
                        vec![up_id, x],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Sin => {
                    // d(sin(x))/dx = cos(x). grad_x = upstream * cos(x).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let cos_x = push_node(
                        &graph_handle,
                        Op::Cos,
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, cos_x],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Cos => {
                    // d(cos(x))/dx = -sin(x). grad_x = -(upstream * sin(x)).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let sin_x = push_node(
                        &graph_handle,
                        Op::Sin,
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let up_sin = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, sin_x],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Neg,
                        vec![up_sin],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Tanh => {
                    // y = tanh(x), dy/dx = 1 - y². grad_x = upstream * (1 - y*y).
                    // Build: ones_of_y_shape - y*y, then multiply by upstream.
                    let x = inputs[0];
                    let y_shape = node_shape(&graph_handle, id);
                    let dtype = node_dtype(&graph_handle, id);
                    let y_sq = push_node(
                        &graph_handle,
                        Op::Sqr,
                        vec![id],
                        y_shape.clone(),
                        dtype,
                    );
                    let ones = build_ones(&graph_handle, y_shape.clone(), dtype);
                    let one_minus_sq = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, y_sq],
                        y_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, one_minus_sq],
                        y_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Sigmoid => {
                    // y = sigmoid(x), dy/dx = y * (1 - y).
                    // grad_x = upstream * y * (1 - y).
                    let x = inputs[0];
                    let y_shape = node_shape(&graph_handle, id);
                    let dtype = node_dtype(&graph_handle, id);
                    let ones = build_ones(&graph_handle, y_shape.clone(), dtype);
                    let one_minus_y = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, id],
                        y_shape.clone(),
                        dtype,
                    );
                    let y_times_1my = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![id, one_minus_y],
                        y_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, y_times_1my],
                        y_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Silu => {
                    // y = x * sigmoid(x)
                    // dy/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                    //       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                    //
                    // Simpler: let s = sigmoid(x). dy/dx = s + y * (1 - s),
                    // where y = x * s is the forward output (this node).
                    // grad_x = upstream * (s + y * (1 - s)).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let s = push_node(
                        &graph_handle,
                        Op::Sigmoid,
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let ones = build_ones(&graph_handle, x_shape.clone(), dtype);
                    let one_minus_s = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, s],
                        x_shape.clone(),
                        dtype,
                    );
                    let y_times_1ms = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![id, one_minus_s],
                        x_shape.clone(),
                        dtype,
                    );
                    let inner = push_node(
                        &graph_handle,
                        Op::Add,
                        vec![s, y_times_1ms],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, inner],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Gelu => {
                    // GELU's tanh-approximation backward is non-trivial.
                    // For the MVP, compute it numerically stable via
                    // central differences — no, that's wrong for
                    // backward. The proper rule differentiates
                    // `0.5 x (1 + tanh(c * (x + 0.044715 x³)))` by chain
                    // rule. That requires several new intermediate
                    // nodes. Implemented below:
                    //
                    // Let u = c * (x + 0.044715 x³), t = tanh(u).
                    // y = 0.5 x (1 + t)
                    // du/dx = c * (1 + 3 * 0.044715 * x²)
                    //       = c + 3 c * 0.044715 * x²
                    // dt/dx = (1 - t²) * du/dx
                    // dy/dx = 0.5 * (1 + t) + 0.5 * x * dt/dx
                    //
                    // This is enough ops that it deserves a dedicated
                    // backward node. For the MVP, panic — users who need
                    // GELU backward should use Silu, which we just
                    // implemented properly.
                    panic!(
                        "backward: Gelu gradient is not yet supported. \
                         Use Silu for differentiable training; Gelu is \
                         currently inference-only."
                    );
                }
                Op::Relu => {
                    // d(relu(x))/dx = step(x). grad_x = upstream * step(x).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let step_x = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, step_x],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Step => {
                    // d(step(x))/dx is 0 almost everywhere (and a Dirac
                    // delta at 0 which has no finite representation). The
                    // standard move is to treat the derivative as 0 and
                    // stop propagation — Step's backward is silently a
                    // no-op. Callers that need a smooth surrogate should
                    // use Sigmoid or Tanh instead.
                }
                Op::Cast(_) => {
                    // Forward: y = cast(x, target_dtype).
                    // Backward: dL/dx = cast(upstream, source_dtype).
                    let x = inputs[0];
                    let x_dtype = node_dtype(&graph_handle, x);
                    let x_shape = node_shape(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Cast(x_dtype),
                        vec![up_id],
                        x_shape,
                        x_dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::SumAll => {
                    // Forward: y = sum(x), scalar.
                    // Backward: dL/dx = broadcast_to(upstream, x.shape).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::MeanAll => {
                    // Forward: y = mean(x) = sum(x) / n, scalar.
                    // Backward: dL/dx = broadcast_to(upstream / n, x.shape).
                    // We build `ones / n` as a constant of shape x.shape
                    // and multiply by broadcast_to(upstream, x.shape).
                    // Simpler: broadcast upstream, then divide by an n-const.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let n_elem = x_shape.elem_count();
                    let broadcast_up = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![up_id],
                        x_shape.clone(),
                        dtype,
                    );
                    let n_const =
                        build_filled_const(&graph_handle, x_shape.clone(), dtype, n_elem as f64);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Div,
                        vec![broadcast_up, n_const],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::BroadcastTo(_) => {
                    // Forward: y = broadcast_to(x, target).
                    // Backward: dL/dx = reduce_sum_to(upstream, x.shape) —
                    // sum away every dim that was expanded or newly added.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::ReduceSumTo(x_shape.clone()),
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Reshape(_) => {
                    // Forward: y = reshape(x, target). Data unchanged.
                    // Backward: reshape upstream back to x.shape.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Reshape(x_shape.clone()),
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::ReduceSumTo(_) => {
                    // Forward: y = reduce_sum_to(x, target).
                    // Backward: broadcast upstream back to x.shape.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::MaxAll | Op::MinAll => {
                    // For y = max(x) (or min), the gradient is an
                    // indicator at the argmax positions, distributed
                    // across ties. Expressed without a dedicated argmax
                    // op using the Step trick:
                    //
                    //   broadcasted_y = broadcast_to(y, x.shape)
                    //   diff = broadcasted_y - x   (for Max)
                    //        | x - broadcasted_y   (for Min)
                    //   indicator = ones - step(diff)
                    //   grad_x    = broadcast_to(upstream, x.shape) * indicator
                    //
                    // At argmax positions `diff == 0`, `step(0) == 0`, so
                    // the indicator is 1. Elsewhere `diff > 0` so step is
                    // 1 and the indicator is 0. Ties share the gradient.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let is_max = matches!(op, Op::MaxAll);
                    let broadcasted_y = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![id],
                        x_shape.clone(),
                        dtype,
                    );
                    let diff = if is_max {
                        push_node(
                            &graph_handle,
                            Op::Sub,
                            vec![broadcasted_y, x],
                            x_shape.clone(),
                            dtype,
                        )
                    } else {
                        push_node(
                            &graph_handle,
                            Op::Sub,
                            vec![x, broadcasted_y],
                            x_shape.clone(),
                            dtype,
                        )
                    };
                    let step_diff = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![diff],
                        x_shape.clone(),
                        dtype,
                    );
                    let ones = build_ones(&graph_handle, x_shape.clone(), dtype);
                    let indicator = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, step_diff],
                        x_shape.clone(),
                        dtype,
                    );
                    let broadcast_up = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![up_id],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![broadcast_up, indicator],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::SumDim(dim) => {
                    // Insert a size-1 at `dim` via reshape, then broadcast
                    // back to x.shape.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let keepdim_shape = reshape_to_keepdim(x_shape.dims(), dim);
                    let reshaped = push_node(
                        &graph_handle,
                        Op::Reshape(keepdim_shape.clone()),
                        vec![up_id],
                        keepdim_shape,
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![reshaped],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::MeanDim(dim) => {
                    // SumDim-style backward, then divide by the reduced dim size.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let reduced_size = x_shape.dims()[dim];
                    let keepdim_shape = reshape_to_keepdim(x_shape.dims(), dim);
                    let reshaped = push_node(
                        &graph_handle,
                        Op::Reshape(keepdim_shape.clone()),
                        vec![up_id],
                        keepdim_shape,
                        dtype,
                    );
                    let broadcast = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![reshaped],
                        x_shape.clone(),
                        dtype,
                    );
                    let n_const = build_filled_const(
                        &graph_handle,
                        x_shape.clone(),
                        dtype,
                        reduced_size as f64,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Div,
                        vec![broadcast, n_const],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::MaxDim(dim) | Op::MinDim(dim) => {
                    // Same indicator trick as MaxAll/MinAll, but apply the
                    // "reshape to keepdim then broadcast" pattern to both
                    // the forward output and upstream before multiplying.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let is_max = matches!(op, Op::MaxDim(_));
                    let keepdim_shape = reshape_to_keepdim(x_shape.dims(), dim);
                    let y_keepdim = push_node(
                        &graph_handle,
                        Op::Reshape(keepdim_shape.clone()),
                        vec![id],
                        keepdim_shape.clone(),
                        dtype,
                    );
                    let broadcasted_y = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![y_keepdim],
                        x_shape.clone(),
                        dtype,
                    );
                    let diff = if is_max {
                        push_node(
                            &graph_handle,
                            Op::Sub,
                            vec![broadcasted_y, x],
                            x_shape.clone(),
                            dtype,
                        )
                    } else {
                        push_node(
                            &graph_handle,
                            Op::Sub,
                            vec![x, broadcasted_y],
                            x_shape.clone(),
                            dtype,
                        )
                    };
                    let step_diff = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![diff],
                        x_shape.clone(),
                        dtype,
                    );
                    let ones = build_ones(&graph_handle, x_shape.clone(), dtype);
                    let indicator = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, step_diff],
                        x_shape.clone(),
                        dtype,
                    );
                    let up_keepdim = push_node(
                        &graph_handle,
                        Op::Reshape(keepdim_shape.clone()),
                        vec![up_id],
                        keepdim_shape,
                        dtype,
                    );
                    let broadcast_up = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![up_keepdim],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![broadcast_up, indicator],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::SoftmaxLastDim => {
                    // grad_x = softmax_last_dim_backward(y, upstream)
                    // where y is this forward node's output.
                    let x = inputs[0];
                    let y_shape = node_shape(&graph_handle, id);
                    let dtype = node_dtype(&graph_handle, id);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::SoftmaxLastDimBackward,
                        vec![id, up_id],
                        y_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::LayerNormLastDim { eps } => {
                    // grad_x = layer_norm_last_dim_backward(x, upstream, eps)
                    // Uses the original input x rather than the forward
                    // output because the backward formula needs both
                    // mean/variance statistics and the centered values.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::LayerNormLastDimBackward { eps },
                        vec![x, up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::RmsNormLastDim { eps } => {
                    // Emit a single Op::RmsNormLastDimBackward node so
                    // backends can dispatch a fused kernel. Backends
                    // without one fall through to the executor's
                    // cpu_fallback path, which resolves via the
                    // reference implementation (still one op in the
                    // graph — the executor does the decomposition
                    // transparently via the op dispatch).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::RmsNormLastDimBackward { eps },
                        vec![x, up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Rope => {
                    // d_x = rope(upstream, cos, -sin). The cos/sin
                    // tables are treated as constants (no gradient
                    // flows back through them).
                    let x = inputs[0];
                    let cos = inputs[1];
                    let sin = inputs[2];
                    let x_shape = node_shape(&graph_handle, x);
                    let sin_shape = node_shape(&graph_handle, sin);
                    let dtype = node_dtype(&graph_handle, x);
                    let neg_sin = push_node(
                        &graph_handle, Op::Neg, vec![sin], sin_shape, dtype);
                    let grad_x = push_node(
                        &graph_handle, Op::Rope,
                        vec![up_id, cos, neg_sin], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::ArgMaxDim(_) | Op::ArgMinDim(_) => {
                    // Argmax/argmin produce integer indices, which are
                    // non-differentiable. Panic rather than silently
                    // zero-propagating — if the user has one of these in
                    // their differentiated graph, they've almost
                    // certainly made a mistake.
                    panic!(
                        "backward: ArgMaxDim/ArgMinDim produce integer \
                         indices and cannot be differentiated through. \
                         Use MaxDim/MinDim if you want a differentiable \
                         selection."
                    );
                }
                Op::QMatMul { .. } => {
                    // Quantized matmul is only used for frozen weights
                    // in the expected serving-inference use case, so
                    // backward is not implemented. If we ever need to
                    // fine-tune through quantized weights (e.g. QLoRA
                    // on adapter params attached to a frozen Q-weight),
                    // the plan is: dequantize the weight once, run a
                    // standard matmul backward, zero the gradient on
                    // the Q-bytes input.
                    panic!(
                        "backward: QMatMul is not differentiable (quantized \
                         weights are frozen). Use a dequantize + standard \
                         matmul if you need gradients through this input."
                    );
                }
                Op::SoftmaxLastDimBackward
                | Op::LayerNormLastDimBackward { .. }
                | Op::RmsNormLastDimBackward { .. } => {
                    // Higher-order gradients through the backward helper
                    // ops are not supported in the MVP — they'd require
                    // either a full symbolic-derivative pass over the
                    // backward op or a dedicated second-order backward
                    // helper. For now, panic with a clear message.
                    panic!(
                        "backward: higher-order gradients through \
                         softmax/layer_norm backward helpers are not yet \
                         supported in the MVP."
                    );
                }
                Op::IndexSelect { dim } => {
                    // Forward: out = index_select(data, dim, indices).
                    // Backward: grad_data = index_add(zeros_like(data), dim, indices, upstream).
                    let data = inputs[0];
                    let idx = inputs[1];
                    let data_shape = node_shape(&graph_handle, data);
                    let dtype = node_dtype(&graph_handle, data);
                    let zeros = build_filled_const(
                        &graph_handle,
                        data_shape.clone(),
                        dtype,
                        0.0,
                    );
                    let grad_data = push_node(
                        &graph_handle,
                        Op::IndexAdd { dim },
                        vec![zeros, idx, up_id],
                        data_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, data, grad_data, &graph_handle);
                    // Index tensor is itself non-differentiable (integer dtype).
                }
                Op::Gather { dim } => {
                    // Forward: out = gather(data, dim, indices).
                    // Backward: grad_data = scatter_add(zeros_like(data), dim, indices, upstream).
                    let data = inputs[0];
                    let idx = inputs[1];
                    let data_shape = node_shape(&graph_handle, data);
                    let dtype = node_dtype(&graph_handle, data);
                    let zeros = build_filled_const(
                        &graph_handle,
                        data_shape.clone(),
                        dtype,
                        0.0,
                    );
                    let grad_data = push_node(
                        &graph_handle,
                        Op::ScatterAdd { dim },
                        vec![zeros, idx, up_id],
                        data_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, data, grad_data, &graph_handle);
                }
                Op::IndexAdd { dim } => {
                    // Forward: out = index_add(base, dim, indices, src).
                    // Backward:
                    //   grad_base = upstream (pass-through — base flows
                    //               through the non-indexed positions and
                    //               the accumulated positions don't zero
                    //               it out, they add to it)
                    //   grad_src  = index_select(upstream, dim, indices)
                    //               (the pieces of upstream that came from
                    //                src in the forward).
                    let base = inputs[0];
                    let idx = inputs[1];
                    let src = inputs[2];
                    let src_shape = node_shape(&graph_handle, src);
                    let dtype = node_dtype(&graph_handle, src);
                    // grad_base: upstream passes through.
                    accumulate_grad(&mut upstream, base, up_id, &graph_handle);
                    // grad_src: select from upstream at the indices.
                    let grad_src = push_node(
                        &graph_handle,
                        Op::IndexSelect { dim },
                        vec![up_id, idx],
                        src_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, src, grad_src, &graph_handle);
                }
                Op::Concat { dim } => {
                    // Forward: out = concat(a, b, dim).
                    // Backward: split upstream along `dim` at a.size(dim),
                    // routing the left slice to a and the right slice to b.
                    let a = inputs[0];
                    let b = inputs[1];
                    let a_shape = node_shape(&graph_handle, a);
                    let b_shape = node_shape(&graph_handle, b);
                    let dtype = node_dtype(&graph_handle, a);
                    let a_len = a_shape.dims()[dim];
                    let b_len = b_shape.dims()[dim];
                    let grad_a = push_node(
                        &graph_handle,
                        Op::Slice { dim, start: 0, len: a_len },
                        vec![up_id],
                        a_shape,
                        dtype,
                    );
                    let grad_b = push_node(
                        &graph_handle,
                        Op::Slice { dim, start: a_len, len: b_len },
                        vec![up_id],
                        b_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::Slice { dim, start, len: _ } => {
                    // Forward: out = slice(x, dim, start, len).
                    // Backward: build a zeros tensor of x.shape, then
                    // scatter the upstream into the slice range. We use
                    // a concat of three pieces: left zeros, upstream,
                    // right zeros (along `dim`).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let x_dims = x_shape.dims();
                    let dim_size = x_dims[dim];
                    let up_len = x_dims[dim]
                        .min(node_shape(&graph_handle, up_id).dims()[dim]);
                    let left_len = start;
                    let right_len = dim_size - start - up_len;
                    // Build left zero pad if needed.
                    let mut pieces: Vec<NodeId> = Vec::new();
                    if left_len > 0 {
                        let mut left_dims: Vec<usize> = x_dims.to_vec();
                        left_dims[dim] = left_len;
                        let left_shape = Shape::from_dims(&left_dims);
                        pieces.push(build_filled_const(
                            &graph_handle,
                            left_shape,
                            dtype,
                            0.0,
                        ));
                    }
                    pieces.push(up_id);
                    if right_len > 0 {
                        let mut right_dims: Vec<usize> = x_dims.to_vec();
                        right_dims[dim] = right_len;
                        let right_shape = Shape::from_dims(&right_dims);
                        pieces.push(build_filled_const(
                            &graph_handle,
                            right_shape,
                            dtype,
                            0.0,
                        ));
                    }
                    // Fold concat-left-to-right.
                    let mut current = pieces[0];
                    let mut current_dims: Vec<usize> = {
                        let n = graph_handle.borrow();
                        n.node(current).shape.dims().to_vec()
                    };
                    for &next in &pieces[1..] {
                        let next_dims: Vec<usize> = {
                            let n = graph_handle.borrow();
                            n.node(next).shape.dims().to_vec()
                        };
                        let mut combined = current_dims.clone();
                        combined[dim] = current_dims[dim] + next_dims[dim];
                        let combined_shape = Shape::from_dims(&combined);
                        current = push_node(
                            &graph_handle,
                            Op::Concat { dim },
                            vec![current, next],
                            combined_shape,
                            dtype,
                        );
                        current_dims = combined;
                    }
                    accumulate_grad(&mut upstream, x, current, &graph_handle);
                }
                Op::AddScalar(_) => {
                    // y = x + c, dy/dx = 1. Upstream passes through.
                    let x = inputs[0];
                    accumulate_grad(&mut upstream, x, up_id, &graph_handle);
                }
                Op::MulScalar(c) => {
                    // y = x * c, dy/dx = c. Multiply upstream by c via
                    // another MulScalar node.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::MulScalar(c),
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::PowI(n) => {
                    // y = x^n, dy/dx = n * x^(n-1).
                    // grad_x = upstream * n * x^(n-1).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let x_pow_nm1 = push_node(
                        &graph_handle,
                        Op::PowI(n - 1),
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let scaled = push_node(
                        &graph_handle,
                        Op::MulScalar(n as f64),
                        vec![x_pow_nm1],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, scaled],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Clamp { min, max } => {
                    // Subgradient: 1 where min <= x <= max, 0 elsewhere.
                    // Expressed via (step(x - min) - step(x - max))
                    // which is 1 in the interior (x > min and x <= max...
                    // actually step(0) = 0, so the right end is a bit
                    // tricky). For a reference implementation the
                    // simplest approach is a dedicated "clamp backward"
                    // indicator. We build it out of the existing ops:
                    //
                    //   lower_ok = step(x - min)   // 1 where x > min
                    //   upper_ok = step(max - x)   // 1 where x < max
                    //   indicator = lower_ok * upper_ok
                    //   grad_x = upstream * indicator
                    //
                    // This underestimates the gradient on the exact
                    // boundary by a measure-zero amount — standard.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let min_shifted = push_node(
                        &graph_handle,
                        Op::AddScalar(-min),
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let lower_ok = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![min_shifted],
                        x_shape.clone(),
                        dtype,
                    );
                    let max_minus_x = push_node(
                        &graph_handle,
                        Op::MulScalar(-1.0),
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let max_minus_x = push_node(
                        &graph_handle,
                        Op::AddScalar(max),
                        vec![max_minus_x],
                        x_shape.clone(),
                        dtype,
                    );
                    let upper_ok = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![max_minus_x],
                        x_shape.clone(),
                        dtype,
                    );
                    let indicator = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![lower_ok, upper_ok],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, indicator],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Maximum => {
                    // y_i = max(a_i, b_i). Gradient flows to whichever
                    // input was the max at that position:
                    //   mask_a = step(a - b)   (1 where a > b, else 0)
                    //   mask_b = 1 - mask_a    (approximately — at ties
                    //                          gradient goes to b due to
                    //                          step(0) = 0)
                    //   grad_a = upstream * mask_a
                    //   grad_b = upstream * mask_b
                    let a = inputs[0];
                    let b = inputs[1];
                    let shape = node_shape(&graph_handle, a);
                    let dtype = node_dtype(&graph_handle, a);
                    let diff = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![a, b],
                        shape.clone(),
                        dtype,
                    );
                    let mask_a = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![diff],
                        shape.clone(),
                        dtype,
                    );
                    let ones = build_ones(&graph_handle, shape.clone(), dtype);
                    let mask_b = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, mask_a],
                        shape.clone(),
                        dtype,
                    );
                    let grad_a = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, mask_a],
                        shape.clone(),
                        dtype,
                    );
                    let grad_b = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, mask_b],
                        shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::Minimum => {
                    // Symmetric to Maximum. mask_a = step(b - a).
                    let a = inputs[0];
                    let b = inputs[1];
                    let shape = node_shape(&graph_handle, a);
                    let dtype = node_dtype(&graph_handle, a);
                    let diff = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![b, a],
                        shape.clone(),
                        dtype,
                    );
                    let mask_a = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![diff],
                        shape.clone(),
                        dtype,
                    );
                    let ones = build_ones(&graph_handle, shape.clone(), dtype);
                    let mask_b = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, mask_a],
                        shape.clone(),
                        dtype,
                    );
                    let grad_a = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, mask_a],
                        shape.clone(),
                        dtype,
                    );
                    let grad_b = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, mask_b],
                        shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::ScatterAdd { dim } => {
                    // Forward: out = scatter_add(base, dim, indices, src).
                    // Backward:
                    //   grad_base = upstream (pass-through)
                    //   grad_src  = gather(upstream, dim, indices)
                    let base = inputs[0];
                    let idx = inputs[1];
                    let src = inputs[2];
                    let src_shape = node_shape(&graph_handle, src);
                    let dtype = node_dtype(&graph_handle, src);
                    accumulate_grad(&mut upstream, base, up_id, &graph_handle);
                    let grad_src = push_node(
                        &graph_handle,
                        Op::Gather { dim },
                        vec![up_id, idx],
                        src_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, src, grad_src, &graph_handle);
                }
                Op::Copy { .. } => {
                    // Forward: out = copy(x, target). Backward: the
                    // gradient flows back to `x`'s original device, so
                    // the reverse would be another Copy. The source
                    // device is x's residency — we don't have the typed
                    // "where was x?" here without tracking it, so we
                    // fall back to the graph's placement hint on x (if
                    // any) or trust the scheduler to re-insert the
                    // correct copy during Phase 4. For Phase 3 we
                    // assume inference-only use and pass upstream through.
                    let x = inputs[0];
                    accumulate_grad(&mut upstream, x, up_id, &graph_handle);
                }
                Op::Release => {
                    // Release has no data output and no gradient to
                    // propagate. Input is destroyed; any downstream
                    // node that needs gradient through the original
                    // tensor must get it from a sibling Copy's backward
                    // path, not this one.
                }
                Op::Move { .. } => {
                    // Move is Copy + destroy-source. For backward,
                    // treat it the same as Copy — gradient flows back
                    // through the move. The Phase 3-style assumption
                    // still applies: we pass upstream through and rely
                    // on the scheduler to re-insert the correct reverse
                    // transfer in a training pass. For inference-only
                    // use (the current case), this is a no-op.
                    let x = inputs[0];
                    accumulate_grad(&mut upstream, x, up_id, &graph_handle);
                }
                Op::Conv2D { .. } => {
                    // Backward for Conv2D is straightforward in
                    // principle (transposed-conv for x-grad, correlation
                    // for w-grad, sum-over-NHW for bias-grad) but needs
                    // its own set of ops and rule family. Not required
                    // for Phase 6a forward-path anchor validation.
                    // When an anchor or task needs conv backward,
                    // extend this arm with the standard rules.
                    panic!(
                        "Tensor::backward: Op::Conv2D does not yet have a \
                         gradient rule. Conv2D is a forward-only primitive \
                         for Phase 6a's inference-focused anchor suite.",
                    );
                }
            }
        }

        GradMap {
            graph:           graph_handle,
            forward_to_grad: upstream,
        }
    }
}

/// Map from forward-graph node IDs to the IDs of their accumulated
/// backward-graph gradient nodes, returned by [`Tensor::backward`].
///
/// All gradient nodes live on the same graph as the forward nodes. Getting
/// a gradient by calling [`GradMap::get`] returns a new [`Tensor`] handle
/// pointing at the gradient node so callers can realize it, plug it into
/// further computation, or back-propagate through it again.
pub struct GradMap {
    graph:           SharedGraph,
    forward_to_grad: HashMap<NodeId, NodeId>,
}

impl GradMap {
    /// Get the gradient tensor for a forward tensor from the same graph.
    ///
    /// Returns `None` if the forward tensor was not reachable from the
    /// root passed to `backward` (i.e. not part of the computation being
    /// differentiated).
    pub fn get(&self, forward: &Tensor) -> Option<Tensor> {
        assert!(
            Rc::ptr_eq(&self.graph, &forward.graph),
            "GradMap::get: tensor is from a different graph",
        );
        let &grad_id = self.forward_to_grad.get(&forward.id)?;
        Some(Tensor {
            graph: self.graph.clone(),
            id:    grad_id,
        })
    }

    /// Number of forward nodes that have a recorded gradient.
    pub fn len(&self) -> usize {
        self.forward_to_grad.len()
    }

    /// Whether this map has no gradients (e.g. a backward over a lone const).
    pub fn is_empty(&self) -> bool {
        self.forward_to_grad.is_empty()
    }
}

// -------- internal helpers shared by the backward pass --------------------

/// Read a node's shape without holding the borrow past the call site.
fn node_shape(graph: &SharedGraph, id: NodeId) -> Shape {
    graph.borrow().node(id).shape.clone()
}

/// Read a node's dtype without holding the borrow past the call site.
fn node_dtype(graph: &SharedGraph, id: NodeId) -> DType {
    graph.borrow().node(id).dtype
}

/// Shape with its last two dims swapped. Used by MatMul's backward rule
/// to construct transpose-node output shapes for any rank ≥ 2.
fn transposed_shape(shape: &Shape) -> Shape {
    let d = shape.dims();
    assert!(
        d.len() >= 2,
        "transposed_shape: expected rank ≥ 2, got {d:?}",
    );
    let rank = d.len();
    let mut out: Vec<usize> = d.to_vec();
    out.swap(rank - 2, rank - 1);
    Shape::from_dims(&out)
}

/// Given a full tensor shape and a dim that was reduced away, produce the
/// keepdim shape — i.e. the full shape with size 1 at `dim`. Used by
/// axis-reduction backward rules to reshape the reduced upstream back to
/// a shape that broadcasts against the original input.
fn reshape_to_keepdim(full_dims: &[usize], dim: usize) -> Shape {
    assert!(
        dim < full_dims.len(),
        "reshape_to_keepdim: dim {dim} out of bounds for {full_dims:?}",
    );
    let mut out: Vec<usize> = full_dims.to_vec();
    out[dim] = 1;
    Shape::from_dims(&out)
}

/// Validate that `src` can be broadcast to `dst` using NumPy rules:
/// right-align, pad the shorter shape with 1s on the left, and for each
/// aligned dimension the src size must equal the dst size or be 1.
fn check_broadcast_compatible(src: &[usize], dst: &[usize]) {
    assert!(
        src.len() <= dst.len(),
        "broadcast_to: source rank {} exceeds target rank {}",
        src.len(),
        dst.len(),
    );
    let pad = dst.len() - src.len();
    for (i, &s) in src.iter().enumerate() {
        let d = dst[pad + i];
        assert!(
            s == d || s == 1,
            "broadcast_to: dim {i} of source ({s}) is incompatible with dim {} of target ({d})",
            pad + i,
        );
    }
}

/// Compute the NumPy broadcast shape of two input shapes. Right-align,
/// pad the shorter with leading 1s, and take the element-wise max at
/// each position. Panics if the shapes are incompatible (neither can
/// expand to match the other at some dim).
fn compute_broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let n = a.len().max(b.len());
    let a_pad = n - a.len();
    let b_pad = n - b.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ai = if i < a_pad { 1 } else { a[i - a_pad] };
        let bi = if i < b_pad { 1 } else { b[i - b_pad] };
        let oi = if ai == bi {
            ai
        } else if ai == 1 {
            bi
        } else if bi == 1 {
            ai
        } else {
            panic!(
                "compute_broadcast_shape: incompatible shapes {a:?} vs {b:?} at axis {i}: {ai} vs {bi}",
            );
        };
        out.push(oi);
    }
    out
}

/// Push a node to the graph and return its ID. Used by backward to append
/// gradient-computing nodes without going through the public builders,
/// which would re-validate shapes the backward pass has already derived.
fn push_node(
    graph: &SharedGraph,
    op: Op,
    inputs: Vec<NodeId>,
    shape: Shape,
    dtype: DType,
) -> NodeId {
    graph.borrow_mut().push(Node {
        op,
        inputs,
        shape,
        dtype,
    })
}

/// Build a ones tensor of the given shape and dtype as a `Const` node.
/// Used to seed the initial upstream gradient at the root of the backward
/// pass (`dL/dL = 1`).
fn build_ones(graph: &SharedGraph, shape: Shape, dtype: DType) -> NodeId {
    build_filled_const(graph, shape, dtype, 1.0)
}

/// Build a constant tensor of the given shape and dtype filled with
/// `value` (expressed as `f64` and converted to the target dtype). Used
/// by backward rules that need a scalar-valued broadcast, for example
/// `MeanAll`'s `1/n` factor expressed as "divide by an n-filled const."
///
/// Build RoPE cos/sin tables for a given `(base, start_pos, seq,
/// head_dim)`. Each returned `Vec<f32>` has shape `[seq, head_dim]`
/// (row-major) in the half-split layout [`Tensor::rope_with_tables`]
/// expects: positions `[:, :half]` and `[:, half:]` hold the same
/// value for the same `(p, i)`, so a single elementwise multiply with
/// `self` does the right thing alongside `rotate_half(self) * sin`.
///
/// Exposed as a free function so callers can share the tables across
/// many RoPE applications (e.g. every attention layer in a transformer
/// forward pass).
pub fn build_rope_tables(
    base: f64,
    start_pos: usize,
    seq: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        head_dim.is_multiple_of(2),
        "build_rope_tables: head_dim {head_dim} must be even",
    );
    let half = head_dim / 2;
    let mut cos_data = vec![0.0_f32; seq * head_dim];
    let mut sin_data = vec![0.0_f32; seq * head_dim];
    for p in 0..seq {
        let pos = (start_pos + p) as f64;
        for i in 0..half {
            let freq = base.powf(-2.0 * (i as f64) / (head_dim as f64));
            let theta = pos * freq;
            let c = theta.cos() as f32;
            let s = theta.sin() as f32;
            cos_data[p * head_dim + i] = c;
            cos_data[p * head_dim + i + half] = c;
            sin_data[p * head_dim + i] = s;
            sin_data[p * head_dim + i + half] = s;
        }
    }
    (cos_data, sin_data)
}

/// Only float dtypes are supported here — integer/index tensors never
/// appear as gradients, so a `U32` call indicates a bug in a backward
/// rule rather than a missing feature.
fn build_filled_const(graph: &SharedGraph, shape: Shape, dtype: DType, value: f64) -> NodeId {
    let n = shape.elem_count();
    let data = match dtype {
        DType::F32 => ConstData::F32(Arc::from(vec![value as f32; n])),
        DType::F64 => ConstData::F64(Arc::from(vec![value; n])),
        DType::BF16 => ConstData::BF16(Arc::from(vec![bf16::from_f64(value); n])),
        DType::F16 => ConstData::F16(Arc::from(vec![f16::from_f64(value); n])),
        other => panic!(
            "backward: build_filled_const: unsupported dtype {other:?} \
             (gradients are always floats — this would indicate a bug in \
             a backward rule that tried to differentiate through an \
             integer tensor)",
        ),
    };
    push_node(graph, Op::Const(data), vec![], shape, dtype)
}

/// Accumulate a new gradient contribution into the upstream map.
///
/// If `forward_id` already has an upstream gradient, emit an `Add` node to
/// sum the two contributions. If not, record the new one as-is. This is
/// the multi-variable chain rule in its operational form: when a forward
/// node has multiple downstream consumers, the gradients from each must
/// be summed.
fn accumulate_grad(
    upstream: &mut HashMap<NodeId, NodeId>,
    forward_id: NodeId,
    new_grad_id: NodeId,
    graph: &SharedGraph,
) {
    match upstream.get(&forward_id).copied() {
        None => {
            upstream.insert(forward_id, new_grad_id);
        }
        Some(existing_id) => {
            let shape = node_shape(graph, forward_id);
            let dtype = node_dtype(graph, forward_id);
            let combined = push_node(
                graph,
                Op::Add,
                vec![existing_id, new_grad_id],
                shape,
                dtype,
            );
            upstream.insert(forward_id, combined);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destructive_input_release_is_some_zero() {
        assert_eq!(Op::Release.destructive_input(), Some(0));
    }

    #[test]
    fn conv2d_builder_emits_conv2d_node_with_right_shape() {
        // k=3 s=1 p=1 keeps H and W.
        let x = Tensor::from_f32(vec![0.0_f32; 1 * 2 * 4 * 4], Shape::from_dims(&[1, 2, 4, 4]));
        let w = x.const_f32_like(vec![0.0_f32; 3 * 2 * 3 * 3], Shape::from_dims(&[3, 2, 3, 3]));
        let b = x.const_f32_like(vec![0.0_f32; 3], Shape::from_dims(&[3]));
        let y = x.conv2d(&w, Some(&b), (1, 1), (1, 1), 1);
        assert_eq!(y.shape().dims(), &[1, 3, 4, 4]);
    }

    #[test]
    fn conv2d_builder_stride_and_no_padding() {
        // k=3 s=2 p=0 on H=W=8 gives (8-3)/2+1 = 3.
        let x = Tensor::from_f32(vec![0.0_f32; 1 * 2 * 8 * 8], Shape::from_dims(&[1, 2, 8, 8]));
        let w = x.const_f32_like(vec![0.0_f32; 4 * 2 * 3 * 3], Shape::from_dims(&[4, 2, 3, 3]));
        let y = x.conv2d(&w, None, (2, 2), (0, 0), 1);
        assert_eq!(y.shape().dims(), &[1, 4, 3, 3]);
    }

    #[test]
    fn conv2d_builder_depthwise_groups() {
        // groups=Cin=Cout=4 is the depthwise case. Weight per channel is [Cin/groups=1, kH, kW].
        let x = Tensor::from_f32(vec![0.0_f32; 1 * 4 * 4 * 4], Shape::from_dims(&[1, 4, 4, 4]));
        let w = x.const_f32_like(vec![0.0_f32; 4 * 1 * 3 * 3], Shape::from_dims(&[4, 1, 3, 3]));
        let y = x.conv2d(&w, None, (1, 1), (1, 1), 4);
        assert_eq!(y.shape().dims(), &[1, 4, 4, 4]);
    }

    #[test]
    fn destructive_input_move_is_some_zero() {
        assert_eq!(Op::Move { target: DeviceLocation::Cpu }.destructive_input(), Some(0));
    }

    #[test]
    fn destructive_input_non_destructive_ops_are_none() {
        assert_eq!(Op::Copy { target: DeviceLocation::Cpu }.destructive_input(), None);
        assert_eq!(Op::Add.destructive_input(), None);
        assert_eq!(Op::Mul.destructive_input(), None);
        assert_eq!(Op::MatMul.destructive_input(), None);
        assert_eq!(Op::Relu.destructive_input(), None);
    }

    #[test]
    fn move_to_device_emits_op_move_node() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let moved = a.move_to_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let g = moved.graph().borrow();
        match &g.node(moved.id()).op {
            Op::Move { target } => {
                assert_eq!(*target, DeviceLocation::Vulkan { gpu_id: 0 });
            }
            other => panic!("expected Op::Move, got {other:?}"),
        }
        assert_eq!(g.node(moved.id()).inputs, vec![a.id()]);
        // Output has the input's shape (residency changes, data doesn't).
        assert_eq!(g.node(moved.id()).shape.elem_count(), 2);
    }

    #[test]
    fn copy_to_device_emits_op_copy_node() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let b = a.copy_to_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let g = b.graph().borrow();
        match &g.node(b.id()).op {
            Op::Copy { target } => {
                assert_eq!(*target, DeviceLocation::Vulkan { gpu_id: 0 });
            }
            other => panic!("expected Op::Copy, got {other:?}"),
        }
        // Source stays resident: the original `a` node still in the graph
        // and is the input of the Copy node.
        assert_eq!(g.node(b.id()).inputs, vec![a.id()]);
    }

    #[test]
    fn release_builder_emits_op_release_node() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let released = a.release();
        let g = released.graph().borrow();
        assert!(matches!(g.node(released.id()).op, Op::Release));
        assert_eq!(g.node(released.id()).inputs, vec![a.id()]);
        // Output is a zero-element marker.
        assert_eq!(g.node(released.id()).shape.elem_count(), 0);
    }

    #[test]
    fn placement_is_none_by_default() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        assert_eq!(a.placement(), None);
    }

    #[test]
    fn on_device_sets_placement_hint() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        // Only tag the Add node; the const leaves remain unplaced.
        let c = a.add(&b).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
        assert_eq!(c.placement(), Some(DeviceLocation::Vulkan { gpu_id: 0 }));
        assert_eq!(a.placement(), None);
        assert_eq!(b.placement(), None);
    }

    #[test]
    fn placement_survives_graph_re_reads() {
        let a = Tensor::from_f32(vec![1.0], Shape::from_dims(&[1]));
        let tagged = a.clone().on_device(DeviceLocation::Cpu);
        // Re-read from a fresh borrow — round-trips through the side-table.
        assert_eq!(tagged.graph().borrow().placement(tagged.id()), Some(DeviceLocation::Cpu));
    }

    #[test]
    fn from_f32_creates_single_const_node() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        assert_eq!(a.graph().borrow().len(), 1);
        assert_eq!(a.shape().dims(), &[3]);
        assert_eq!(a.dtype(), DType::F32);
        let node = a.graph().borrow().node(a.id()).clone();
        assert!(matches!(node.op, Op::Const(ConstData::F32(_))));
        assert!(node.inputs.is_empty());
    }

    #[test]
    fn add_appends_a_node_and_tracks_inputs() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        let c = a.add(&b);
        assert_eq!(c.graph().borrow().len(), 3); // const, const, add
        let node = c.graph().borrow().node(c.id()).clone();
        assert!(matches!(node.op, Op::Add));
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.inputs[0], a.id());
        assert_eq!(node.inputs[1], b.id());
        assert_eq!(c.shape().dims(), &[3]);
    }

    #[test]
    fn chained_ops_all_share_one_graph() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        let c = a.add(&b).mul(&a).sqr().relu();
        assert_eq!(c.graph().borrow().len(), 6); // 2 consts + add + mul + sqr + relu
        assert_eq!(c.shape().dims(), &[3]);
    }

    #[test]
    fn matmul_validates_shapes_and_produces_correct_output_shape() {
        let a = Tensor::from_f32(vec![1.0; 6], Shape::from_dims(&[2, 3]));
        let b = a.const_f32_like(vec![1.0; 12], Shape::from_dims(&[3, 4]));
        let c = a.matmul(&b);
        assert_eq!(c.shape().dims(), &[2, 4]);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn add_panics_on_shape_mismatch() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let b = a.const_f32_like(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let _ = a.add(&b);
    }

    #[test]
    #[should_panic(expected = "inner dim mismatch")]
    fn matmul_panics_on_inner_dim_mismatch() {
        let a = Tensor::from_f32(vec![1.0; 6], Shape::from_dims(&[2, 3]));
        let b = a.const_f32_like(vec![1.0; 8], Shape::from_dims(&[4, 2]));
        let _ = a.matmul(&b);
    }

    #[test]
    #[should_panic(expected = "must live on the same graph")]
    fn cross_graph_op_is_rejected() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let b = Tensor::from_f32(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        let _ = a.add(&b);
    }

    // ----- multi-dtype graph builders -----

    #[test]
    fn from_f64_tags_node_with_f64_dtype() {
        let a = Tensor::from_f64(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        assert_eq!(a.dtype(), DType::F64);
        let node = a.graph().borrow().node(a.id()).clone();
        assert!(matches!(node.op, Op::Const(ConstData::F64(_))));
    }

    #[test]
    fn from_bf16_tags_node_with_bf16_dtype() {
        let a = Tensor::from_bf16(
            vec![bf16::from_f32(1.0), bf16::from_f32(2.0)],
            Shape::from_dims(&[2]),
        );
        assert_eq!(a.dtype(), DType::BF16);
        let node = a.graph().borrow().node(a.id()).clone();
        assert!(matches!(node.op, Op::Const(ConstData::BF16(_))));
    }

    #[test]
    fn from_f16_tags_node_with_f16_dtype() {
        let a = Tensor::from_f16(
            vec![f16::from_f32(1.0), f16::from_f32(2.0)],
            Shape::from_dims(&[2]),
        );
        assert_eq!(a.dtype(), DType::F16);
        let node = a.graph().borrow().node(a.id()).clone();
        assert!(matches!(node.op, Op::Const(ConstData::F16(_))));
    }

    #[test]
    #[should_panic(expected = "dtype mismatch")]
    fn add_panics_on_mixed_dtype() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let b = a.const_f64_like(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let _ = a.add(&b);
    }

    // ----- transpose -----

    #[test]
    fn transpose_swaps_shape_dims() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from_dims(&[2, 3]));
        let t = a.transpose();
        assert_eq!(t.shape().dims(), &[3, 2]);
        let node = t.graph().borrow().node(t.id()).clone();
        assert!(matches!(node.op, Op::Transpose));
        assert_eq!(node.inputs, vec![a.id()]);
    }

    #[test]
    #[should_panic(expected = "rank ≥ 2")]
    fn transpose_rejects_rank_1() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let _ = a.transpose();
    }

    #[test]
    fn transpose_on_rank_3_swaps_last_two_dims() {
        // [2, 3, 4] → [2, 4, 3]
        let a = Tensor::from_f32(vec![0.0_f32; 24], Shape::from_dims(&[2, 3, 4]));
        let t = a.transpose();
        assert_eq!(t.shape().dims(), &[2, 4, 3]);
    }

    // ----- additional builder validation tests -----

    #[test]
    fn matmul_rank_3_batched_shape() {
        // [2, 3, 4] @ [2, 4, 5] → [2, 3, 5]
        let a = Tensor::from_f32(vec![0.0; 24], Shape::from_dims(&[2, 3, 4]));
        let b = a.const_f32_like(vec![0.0; 40], Shape::from_dims(&[2, 4, 5]));
        let c = a.matmul(&b);
        assert_eq!(c.shape().dims(), &[2, 3, 5]);
    }

    #[test]
    #[should_panic(expected = "batch dim mismatch")]
    fn matmul_rank_3_rejects_batch_dim_mismatch() {
        let a = Tensor::from_f32(vec![0.0; 24], Shape::from_dims(&[2, 3, 4]));
        let b = a.const_f32_like(vec![0.0; 60], Shape::from_dims(&[3, 4, 5]));
        let _ = a.matmul(&b);
    }

    #[test]
    fn matmul_auto_broadcasts_rank_2_rhs_against_batched_lhs() {
        // [batch=2, seq=3, k=4] @ [k=4, n=5] → [2, 3, 5]. This is the
        // canonical "linear layer across a batch" pattern and should
        // Just Work without an explicit broadcast_to on the RHS.
        let a = Tensor::from_f32(vec![0.0; 24], Shape::from_dims(&[2, 3, 4]));
        let b = a.const_f32_like(vec![0.0; 20], Shape::from_dims(&[4, 5]));
        let c = a.matmul(&b);
        assert_eq!(c.shape().dims(), &[2, 3, 5]);
    }

    #[test]
    fn matmul_auto_broadcasts_rank_2_lhs_against_batched_rhs() {
        // [m=3, k=4] @ [batch=2, k=4, n=5] → [2, 3, 5].
        let a = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]));
        let b = a.const_f32_like(vec![0.0; 40], Shape::from_dims(&[2, 4, 5]));
        let c = a.matmul(&b);
        assert_eq!(c.shape().dims(), &[2, 3, 5]);
    }

    #[test]
    fn concat_output_shape_sums_along_dim() {
        // [2, 3] concat [2, 4] along dim 1 → [2, 7]
        let a = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]));
        let b = a.const_f32_like(vec![0.0; 8], Shape::from_dims(&[2, 4]));
        let c = a.concat(&b, 1);
        assert_eq!(c.shape().dims(), &[2, 7]);
    }

    #[test]
    #[should_panic(expected = "non-dim shapes")]
    fn concat_rejects_nondim_shape_mismatch() {
        let a = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]));
        let b = a.const_f32_like(vec![0.0; 12], Shape::from_dims(&[3, 4]));
        let _ = a.concat(&b, 1);
    }

    #[test]
    #[should_panic(expected = "rank mismatch")]
    fn concat_rejects_rank_mismatch() {
        let a = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]));
        let b = a.const_f32_like(vec![0.0; 6], Shape::from_dims(&[6]));
        let _ = a.concat(&b, 0);
    }

    #[test]
    fn slice_shrinks_only_the_slice_dim() {
        // [3, 4] slice dim 1, start 1, len 2 → [3, 2]
        let x = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]));
        let s = x.slice(1, 1, 2);
        assert_eq!(s.shape().dims(), &[3, 2]);
    }

    #[test]
    #[should_panic(expected = "exceeds dim size")]
    fn slice_rejects_out_of_bounds_range() {
        let x = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]));
        let _ = x.slice(1, 1, 3); // start=1, len=3 → would need dim>=4
    }

    #[test]
    fn broadcast_add_shape_promotes_to_common_shape() {
        // [4, 1] + [1, 3] → [4, 3]
        let a = Tensor::from_f32(vec![0.0; 4], Shape::from_dims(&[4, 1]));
        let b = a.const_f32_like(vec![0.0; 3], Shape::from_dims(&[1, 3]));
        let c = a.broadcast_add(&b);
        assert_eq!(c.shape().dims(), &[4, 3]);
    }

    #[test]
    fn broadcast_sub_pads_shorter_shape_with_leading_ones() {
        // [3] - [2, 3] → [2, 3]
        let a = Tensor::from_f32(vec![0.0; 3], Shape::from_dims(&[3]));
        let b = a.const_f32_like(vec![0.0; 6], Shape::from_dims(&[2, 3]));
        let c = a.broadcast_sub(&b);
        assert_eq!(c.shape().dims(), &[2, 3]);
    }

    #[test]
    #[should_panic(expected = "incompatible shapes")]
    fn broadcast_add_rejects_incompatible_shapes() {
        let a = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]));
        let b = a.const_f32_like(vec![0.0; 8], Shape::from_dims(&[2, 4]));
        let _ = a.broadcast_add(&b);
    }

    #[test]
    fn argmax_dim_is_u32_and_removes_reduced_dim() {
        let x = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]));
        let am = x.argmax_dim(1);
        assert_eq!(am.dtype(), DType::U32);
        assert_eq!(am.shape().dims(), &[2]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn argmax_dim_rejects_bad_dim() {
        let x = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]));
        let _ = x.argmax_dim(5);
    }

    #[test]
    fn index_add_shape_validation() {
        let base = Tensor::from_f32(vec![0.0; 10], Shape::from_dims(&[10]));
        let idx = base.const_u32_like(vec![1, 3, 5], Shape::from_dims(&[3]));
        let src = base.const_f32_like(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let out = base.index_add(0, &idx, &src);
        assert_eq!(out.shape().dims(), &[10]);
    }

    #[test]
    #[should_panic(expected = "dtypes must match")]
    fn index_add_rejects_dtype_mismatch() {
        let base = Tensor::from_f32(vec![0.0; 5], Shape::from_dims(&[5]));
        let idx = base.const_u32_like(vec![0, 2], Shape::from_dims(&[2]));
        let src = base.const_f64_like(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let _ = base.index_add(0, &idx, &src);
    }

    #[test]
    fn scatter_add_validates_index_matches_src() {
        let base = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]));
        let idx = base.const_u32_like(vec![0, 2, 1, 0], Shape::from_dims(&[2, 2]));
        let src = base.const_f32_like(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]));
        let out = base.scatter_add(1, &idx, &src);
        assert_eq!(out.shape().dims(), &[2, 3]);
    }

    #[test]
    #[should_panic(expected = "same shape")]
    fn scatter_add_rejects_index_src_shape_mismatch() {
        let base = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]));
        let idx = base.const_u32_like(vec![0, 1], Shape::from_dims(&[2]));
        let src = base.const_f32_like(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]));
        let _ = base.scatter_add(1, &idx, &src);
    }

    #[test]
    fn reduce_sum_to_validates_compatibility() {
        // [3, 4] can reduce to [4] (sum along dim 0) or [3, 1] (sum along dim 1).
        let x = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]));
        let r1 = x.reduce_sum_to(Shape::from_dims(&[4]));
        assert_eq!(r1.shape().dims(), &[4]);
        let r2 = x.reduce_sum_to(Shape::from_dims(&[3, 1]));
        assert_eq!(r2.shape().dims(), &[3, 1]);
    }

    #[test]
    #[should_panic(expected = "incompatible")]
    fn reduce_sum_to_rejects_non_broadcast_target() {
        // [3, 4] cannot reduce to [3, 2] — target must be broadcast-into-source.
        let x = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]));
        let _ = x.reduce_sum_to(Shape::from_dims(&[3, 2]));
    }

    #[test]
    fn reshape_preserves_element_count() {
        let x = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]));
        let r = x.reshape(Shape::from_dims(&[2, 6]));
        assert_eq!(r.shape().dims(), &[2, 6]);
    }

    #[test]
    #[should_panic(expected = "element count mismatch")]
    fn reshape_rejects_different_element_count() {
        let x = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]));
        let _ = x.reshape(Shape::from_dims(&[3, 3]));
    }

    #[test]
    #[should_panic(expected = "same graph")]
    fn concat_across_graphs_panics() {
        let a = Tensor::from_f32(vec![0.0; 3], Shape::from_dims(&[3]));
        let b = Tensor::from_f32(vec![0.0; 3], Shape::from_dims(&[3]));
        let _ = a.concat(&b, 0);
    }

    #[test]
    fn scalar_ops_preserve_shape_and_dtype() {
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let y = x.add_scalar(5.0).mul_scalar(2.0).powi(2).clamp(0.0, 100.0);
        assert_eq!(y.shape().dims(), &[3]);
        assert_eq!(y.dtype(), DType::F32);
    }

    #[test]
    fn maximum_requires_matching_shapes() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let b = a.const_f32_like(vec![4.0, 1.0, 5.0], Shape::from_dims(&[3]));
        let m = a.maximum(&b);
        assert_eq!(m.shape().dims(), &[3]);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn maximum_rejects_shape_mismatch() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let b = a.const_f32_like(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let _ = a.maximum(&b);
    }

    // ----- topo_order -----

    #[test]
    fn topo_order_places_inputs_before_dependents() {
        // Build: c = (a + b) * a
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let sum = a.add(&b);
        let c = sum.mul(&a);
        let order = topo_order(&c.graph().borrow(), c.id());
        // The order should contain exactly 4 nodes (a, b, sum, c) and
        // place each input strictly before its dependents.
        assert_eq!(order.len(), 4);
        let pos = |id: NodeId| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(a.id()) < pos(sum.id()));
        assert!(pos(b.id()) < pos(sum.id()));
        assert!(pos(sum.id()) < pos(c.id()));
        assert!(pos(a.id()) < pos(c.id()));
    }

    #[test]
    fn topo_order_visits_each_node_once_when_shared() {
        // (a + a) — `a` appears twice in the Add's inputs but the topo
        // pass should still visit it exactly once.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let double = a.add(&a);
        let order = topo_order(&double.graph().borrow(), double.id());
        assert_eq!(order.len(), 2);
        assert_eq!(order[0], a.id());
        assert_eq!(order[1], double.id());
    }

    #[test]
    fn topo_order_multi_unions_multiple_roots() {
        // Build a DAG where two "output" nodes share a common input.
        //
        //      a ─┬─> add1 (= a + b)  ← root_1
        //         └─> add2 (= a + c)  ← root_2
        //      b ───> add1
        //      c ───> add2
        //
        // topo_order_multi(&[add1, add2]) must contain all 5 nodes
        // (a, b, c, add1, add2) with a before add1/add2 and b before
        // add1 and c before add2.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let c = a.const_f32_like(vec![5.0, 6.0], Shape::from_dims(&[2]));
        let add1 = a.add(&b);
        let add2 = a.add(&c);
        let order = topo_order_multi(
            &add1.graph().borrow(),
            &[add1.id(), add2.id()],
        );
        assert_eq!(order.len(), 5);
        let pos = |id: NodeId| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(a.id()) < pos(add1.id()));
        assert!(pos(a.id()) < pos(add2.id()));
        assert!(pos(b.id()) < pos(add1.id()));
        assert!(pos(c.id()) < pos(add2.id()));
    }

    // ----- backward -----

    #[test]
    fn backward_of_lone_const_seeds_ones() {
        // backward(a) gives a = 1s (the root's upstream is a ones tensor,
        // which is the gradient stored for the root itself).
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let grads = a.backward();
        let g_a = grads.get(&a).expect("root gets a seed gradient");
        // The seed is a Const ones node of matching shape.
        let node = g_a.graph().borrow().node(g_a.id()).clone();
        assert!(matches!(node.op, Op::Const(ConstData::F32(_))));
        assert_eq!(g_a.shape().dims(), &[3]);
    }

    #[test]
    fn backward_of_add_passes_upstream_through() {
        // c = a + b  ⇒  dc/da = 1, dc/db = 1.
        // Upstream seed is a ones tensor. So grad_a and grad_b are both
        // the same ones node (no new math emitted for Add's backward).
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let c = a.add(&b);
        let grads = c.backward();
        let g_a = grads.get(&a).unwrap();
        let g_b = grads.get(&b).unwrap();
        assert_eq!(g_a.id(), g_b.id(), "Add backward routes upstream unchanged");
        assert_eq!(g_a.shape().dims(), &[2]);
    }

    #[test]
    fn backward_of_mul_emits_two_mul_nodes() {
        // c = a * b  ⇒  dc/da = b, dc/db = a (upstream is 1s).
        // The backward pass should emit two new Mul nodes (upstream * b,
        // upstream * a).
        let a = Tensor::from_f32(vec![2.0, 3.0], Shape::from_dims(&[2]));
        let b = a.const_f32_like(vec![5.0, 7.0], Shape::from_dims(&[2]));
        let c = a.mul(&b);
        let nodes_before = c.graph().borrow().len();
        let grads = c.backward();
        let nodes_after = grads.graph.borrow().len();
        // Backward adds: one ones const + two Mul nodes = 3 new nodes.
        assert_eq!(nodes_after - nodes_before, 3);
        let g_a = grads.get(&a).unwrap();
        let g_b = grads.get(&b).unwrap();
        // Both gradient nodes should be Mul nodes.
        let node_a = g_a.graph().borrow().node(g_a.id()).clone();
        let node_b = g_b.graph().borrow().node(g_b.id()).clone();
        assert!(matches!(node_a.op, Op::Mul));
        assert!(matches!(node_b.op, Op::Mul));
    }

    #[test]
    fn backward_accumulates_when_node_used_twice() {
        // c = a * a  ⇒  dc/da = 2a via two separate contributions.
        // After backward, the gradient for a should be an Add node
        // combining the two Mul contributions (one from each input slot
        // of the forward Mul).
        let a = Tensor::from_f32(vec![3.0, 5.0], Shape::from_dims(&[2]));
        let c = a.mul(&a);
        let grads = c.backward();
        let g_a = grads.get(&a).unwrap();
        let node = g_a.graph().borrow().node(g_a.id()).clone();
        // The final gradient for `a` is an Add combining the two
        // upstream-times-other-input contributions.
        assert!(
            matches!(node.op, Op::Add),
            "expected accumulated gradient to be an Add node, got {:?}",
            node.op,
        );
    }

    #[test]
    fn backward_of_matmul_emits_transpose_and_matmul_nodes() {
        // Forward: Y = A @ B,  A:[2,3], B:[3,4], Y:[2,4].
        // Backward: dA = dY @ B^T (shape [2,3]),  dB = A^T @ dY (shape [3,4]).
        let a = Tensor::from_f32(vec![1.0; 6], Shape::from_dims(&[2, 3]));
        let b = a.const_f32_like(vec![1.0; 12], Shape::from_dims(&[3, 4]));
        let y = a.matmul(&b);
        let grads = y.backward();
        let g_a = grads.get(&a).unwrap();
        let g_b = grads.get(&b).unwrap();
        // Gradient shapes must match the forward shapes.
        assert_eq!(g_a.shape().dims(), &[2, 3]);
        assert_eq!(g_b.shape().dims(), &[3, 4]);
        // Both should be MatMul nodes (the outermost op of each gradient).
        let node_a = g_a.graph().borrow().node(g_a.id()).clone();
        let node_b = g_b.graph().borrow().node(g_b.id()).clone();
        assert!(matches!(node_a.op, Op::MatMul));
        assert!(matches!(node_b.op, Op::MatMul));
    }

    // ----- new builder validation -----

    #[test]
    fn cast_tags_node_with_target_dtype() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let b = a.cast(DType::F64);
        assert_eq!(b.dtype(), DType::F64);
        assert_eq!(b.shape().dims(), &[2]);
        let node = b.graph().borrow().node(b.id()).clone();
        assert!(matches!(node.op, Op::Cast(DType::F64)));
    }

    #[test]
    fn broadcast_to_accepts_right_aligned_expansion() {
        // [3] broadcasts to [2, 3]: pad with leading 1, expand.
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let b = a.broadcast_to(Shape::from_dims(&[2, 3]));
        assert_eq!(b.shape().dims(), &[2, 3]);
    }

    #[test]
    fn broadcast_to_accepts_size_one_expansion() {
        // [3, 1] broadcasts to [3, 4]: size-1 dim expands.
        let a = Tensor::from_f32(vec![10.0, 20.0, 30.0], Shape::from_dims(&[3, 1]));
        let b = a.broadcast_to(Shape::from_dims(&[3, 4]));
        assert_eq!(b.shape().dims(), &[3, 4]);
    }

    #[test]
    #[should_panic(expected = "incompatible")]
    fn broadcast_to_rejects_incompatible_dim() {
        // [3] cannot broadcast to [2, 4] — the source dim 3 must match 4.
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let _ = a.broadcast_to(Shape::from_dims(&[2, 4]));
    }

    #[test]
    fn sum_all_produces_rank_zero_output() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let s = a.sum_all();
        assert_eq!(s.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn sum_dim_removes_reduced_dim_from_shape() {
        let a = Tensor::from_f32(vec![1.0; 24], Shape::from_dims(&[2, 3, 4]));
        // Reducing dim 1 should give shape [2, 4].
        let s = a.sum_dim(1);
        assert_eq!(s.shape().dims(), &[2, 4]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn sum_dim_rejects_bad_dim() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let _ = a.sum_dim(5);
    }

    #[test]
    fn softmax_and_layer_norm_preserve_shape() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]));
        assert_eq!(a.softmax_last_dim().shape().dims(), &[2, 2]);
        assert_eq!(a.layer_norm_last_dim(1e-5).shape().dims(), &[2, 2]);
    }

    #[test]
    fn neg_sub_div_sqrt_log_sin_cos_tanh_sigmoid_all_build() {
        // Smoke test: every new builder produces a node with the expected
        // shape and dtype. Numerical correctness is exercised in exec.rs.
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let b = a.const_f32_like(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        for tensor in [
            a.neg(),
            a.sub(&b),
            a.div(&b),
            a.sqrt(),
            a.log(),
            a.sin(),
            a.cos(),
            a.tanh(),
            a.sigmoid(),
            a.step(),
        ] {
            assert_eq!(tensor.shape().dims(), &[3]);
            assert_eq!(tensor.dtype(), DType::F32);
        }
    }

    // ----- index tensors and indexing ops -----

    #[test]
    fn from_u32_tags_node_with_u32_dtype() {
        let a = Tensor::from_u32(vec![1, 2, 3], Shape::from_dims(&[3]));
        assert_eq!(a.dtype(), DType::U32);
        let node = a.graph().borrow().node(a.id()).clone();
        assert!(matches!(node.op, Op::Const(ConstData::U32(_))));
    }

    #[test]
    fn index_select_produces_shape_with_dim_replaced() {
        let data = Tensor::from_f32(vec![1.0; 12], Shape::from_dims(&[3, 4]));
        let idx = data.const_u32_like(vec![0, 2, 1, 0, 2], Shape::from_dims(&[5]));
        let out = data.index_select(0, &idx);
        assert_eq!(out.shape().dims(), &[5, 4]);
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    #[should_panic(expected = "must be U32")]
    fn index_select_rejects_float_index() {
        let data = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let bad = data.const_f32_like(vec![0.0, 1.0], Shape::from_dims(&[2]));
        let _ = data.index_select(0, &bad);
    }

    #[test]
    #[should_panic(expected = "must be rank 1")]
    fn index_select_rejects_multi_dim_index() {
        let data = Tensor::from_f32(vec![1.0; 6], Shape::from_dims(&[2, 3]));
        let idx = data.const_u32_like(vec![0, 1, 0, 1], Shape::from_dims(&[2, 2]));
        let _ = data.index_select(0, &idx);
    }

    #[test]
    fn gather_output_shape_matches_index_shape() {
        let data = Tensor::from_f32(vec![1.0; 12], Shape::from_dims(&[3, 4]));
        // Index shape [2, 5] — same rank as data (rank 2).
        let idx = data.const_u32_like(vec![0; 10], Shape::from_dims(&[2, 5]));
        let out = data.gather(1, &idx);
        assert_eq!(out.shape().dims(), &[2, 5]);
    }

    #[test]
    #[should_panic(expected = "same rank")]
    fn gather_rejects_rank_mismatch() {
        let data = Tensor::from_f32(vec![1.0; 6], Shape::from_dims(&[2, 3]));
        // Rank-1 index for rank-2 data → error.
        let idx = data.const_u32_like(vec![0, 1, 0], Shape::from_dims(&[3]));
        let _ = data.gather(1, &idx);
    }

    #[test]
    fn backward_of_relu_emits_step_node() {
        // Before: this used to panic. After adding Step + Relu backward,
        // it should successfully emit a backward graph rooted in a Mul
        // whose second input is a Step node.
        let a = Tensor::from_f32(vec![-1.0, 2.0, -3.0], Shape::from_dims(&[3]));
        let y = a.relu();
        let grads = y.backward();
        let g_a = grads.get(&a).unwrap();
        let node = g_a.graph().borrow().node(g_a.id()).clone();
        assert!(matches!(node.op, Op::Mul));
        // Find a Step node somewhere in the node's inputs.
        let any_step = node.inputs.iter().any(|&id| {
            matches!(
                g_a.graph().borrow().node(id).op,
                Op::Step,
            )
        });
        assert!(any_step, "Relu backward must reference a Step node");
    }

    #[test]
    fn backward_reuses_exp_forward_output() {
        // Exp's backward rule uses the forward output directly. The
        // gradient for x should be a Mul whose inputs include the
        // forward Exp node (not a new Exp node).
        let a = Tensor::from_f32(vec![0.0, 1.0], Shape::from_dims(&[2]));
        let e = a.exp();
        let exp_forward_id = e.id();
        let grads = e.backward();
        let g_a = grads.get(&a).unwrap();
        let node = g_a.graph().borrow().node(g_a.id()).clone();
        assert!(matches!(node.op, Op::Mul));
        // One of the Mul's inputs should be the original forward Exp node.
        assert!(
            node.inputs.contains(&exp_forward_id),
            "Exp backward should reference the forward output ({exp_forward_id:?}), \
             got inputs {:?}",
            node.inputs,
        );
    }
}

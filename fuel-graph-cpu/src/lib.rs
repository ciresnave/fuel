//! # fuel-graph-cpu
//!
//! Fast CPU executor for `fuel-graph` computation graphs.
//!
//! The [`fuel-reference-backend`] crate provides a textbook-correct
//! executor whose primary purpose is to act as the correctness oracle
//! for every other backend. It is slow by design: every op is a plain
//! loop over elements with no parallelism, no SIMD, no BLAS. A Llama 3
//! 8B forward pass through the reference backend takes tens of
//! seconds per token.
//!
//! This crate provides a **fast** CPU executor that walks the exact
//! same graph structure (`fuel_graph::Graph`, `fuel_graph::Op`) but
//! dispatches the hot-path operation — matrix multiply — through the
//! [`gemm`] crate's BLAS-level kernels. For every other op it simply
//! re-uses [`fuel_reference_backend::ops`] functions, which are
//! per-element but fast enough for the ~5% of non-matmul work in a
//! transformer forward pass.
//!
//! The expected speedup on a matmul-dominated workload like a Llama
//! forward pass is 50-200x on a modern desktop CPU — enough to move
//! from "minutes per token" to "a few seconds per token."
//!
//! This is still the CPU executor. Production GPU execution through
//! CUDA or Metal is a separate future crate (`fuel-cuda-backend`,
//! `fuel-graph-metal`) with the same public API.
//!
//! # Public API
//!
//! Four realize functions, one per supported dtype:
//!
//! - [`realize_f32`] — the primary entry point for most workloads
//! - [`realize_f64`] — higher precision when needed
//! - [`realize_bf16`] — the dtype most loaded Llama weights use on
//!   disk; faster than promoting to f32 for large models
//! - [`realize_f16`] — standard half precision
//!
//! Each has the same signature as the corresponding function in
//! [`fuel_reference_backend::exec`], so swapping executors is a
//! one-line change in calling code.

use fuel_graph::{topo_order, topo_order_multi, NodeId, Op, Tensor};
use fuel_reference_backend::{ops, RefTensor};
use half::{bf16, f16};
use std::collections::HashMap;
use tracing::info_span;

mod backend;
pub use backend::CpuBackend;

mod fast_matmul;

/// Dtype-erased cached tensor, mirroring
/// [`fuel_reference_backend::exec::AnyRefTensor`] but scoped to this
/// crate to avoid a pub re-export across the module boundary.
#[derive(Debug, Clone)]
enum AnyTensor {
    F32(RefTensor<f32>),
    F64(RefTensor<f64>),
    BF16(RefTensor<bf16>),
    F16(RefTensor<f16>),
    U32(RefTensor<u32>),
}

impl AnyTensor {
    fn dtype(&self) -> fuel_core_types::DType {
        use fuel_core_types::DType;
        match self {
            AnyTensor::F32(_) => DType::F32,
            AnyTensor::F64(_) => DType::F64,
            AnyTensor::BF16(_) => DType::BF16,
            AnyTensor::F16(_) => DType::F16,
            AnyTensor::U32(_) => DType::U32,
        }
    }
}

/// Realize a lazy graph tensor to a concrete `f32` `RefTensor`.
/// This is the fast-path entry point.
pub fn realize_f32(tensor: &Tensor) -> RefTensor<f32> {
    match realize_any(tensor) {
        AnyTensor::F32(t) => t,
        other => panic!("realize_f32: root dtype is {:?}, not F32", other.dtype()),
    }
}

/// Realize a lazy graph tensor to a concrete `f64` `RefTensor`.
pub fn realize_f64(tensor: &Tensor) -> RefTensor<f64> {
    match realize_any(tensor) {
        AnyTensor::F64(t) => t,
        other => panic!("realize_f64: root dtype is {:?}, not F64", other.dtype()),
    }
}

/// Realize a lazy graph tensor to a concrete `bf16` `RefTensor`.
pub fn realize_bf16(tensor: &Tensor) -> RefTensor<bf16> {
    match realize_any(tensor) {
        AnyTensor::BF16(t) => t,
        other => panic!("realize_bf16: root dtype is {:?}, not BF16", other.dtype()),
    }
}

/// Realize a lazy graph tensor to a concrete `f16` `RefTensor`.
pub fn realize_f16(tensor: &Tensor) -> RefTensor<f16> {
    match realize_any(tensor) {
        AnyTensor::F16(t) => t,
        other => panic!("realize_f16: root dtype is {:?}, not F16", other.dtype()),
    }
}

/// Realize many tensors in a single walk of the combined graph and
/// unwrap every result as `f32`. All tensors must belong to the same
/// graph and have root dtype `F32`.
///
/// The KV-cache path uses this to compute logits plus every layer's
/// updated K/V in one topological walk, rather than n separate walks
/// that would each recompute the shared prefix.
pub fn realize_many_f32(tensors: &[&Tensor]) -> Vec<RefTensor<f32>> {
    if tensors.is_empty() {
        return Vec::new();
    }
    let graph_rc = tensors[0].graph();
    for t in &tensors[1..] {
        assert!(
            std::sync::Arc::ptr_eq(graph_rc, t.graph()),
            "realize_many_f32: all tensors must belong to the same graph",
        );
    }
    let graph = graph_rc.read().unwrap();
    let roots: Vec<NodeId> = tensors.iter().map(|t| t.id()).collect();
    let order = topo_order_multi(&graph, &roots);
    let mut cache: HashMap<NodeId, AnyTensor> = HashMap::new();

    for id in order {
        let node = graph.node(id);
        // Phase 7.5 G2: slot-first dispatch.
        if let Some(adopted) = try_adopt_slot_cpu(&graph, id, &node.shape) {
            cache.insert(id, adopted);
            continue;
        }
        let result = eval_node_with_graph_context(&graph, id, node, &cache);
        cache.insert(id, result);
    }

    roots
        .iter()
        .map(|id| match cache.get(id).cloned() {
            Some(AnyTensor::F32(t)) => t,
            Some(other) => panic!(
                "realize_many_f32: root dtype is {:?}, not F32",
                other.dtype()
            ),
            None => panic!("realize_many_f32: root missing from cache after topo walk"),
        })
        .collect()
}

/// Phase 7.5 G2: slot-first dispatch for fuel-graph-cpu. If the
/// graph's storage_map has a populated slot for `id`, adopt its bytes
/// via host-buffer download and wrap as an `AnyTensor`.
fn try_adopt_slot_cpu(
    graph: &fuel_graph::Graph,
    id: NodeId,
    shape: &fuel_core_types::Shape,
) -> Option<AnyTensor> {
    let slot_arc = graph.storage_for(id)?;
    let buf = {
        let slot = slot_arc.read().unwrap();
        slot.as_dyn().to_host_buffer_dyn().expect("slot D2H")
    };
    let any = match buf {
        fuel_core_types::HostBuffer::F32(v) => AnyTensor::F32(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::F64(v) => AnyTensor::F64(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::BF16(v) => AnyTensor::BF16(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::F16(v) => AnyTensor::F16(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::U32(v) => AnyTensor::U32(RefTensor::from_vec(v, shape.clone())),
        other => panic!(
            "fuel-graph-cpu slot adopt: unsupported host-buffer dtype {:?}",
            other.dtype(),
        ),
    };
    Some(any)
}

/// Core realize loop: walk the graph in topological order, caching
/// each node's output and dispatching `MatMul` to the fast path.
fn realize_any(tensor: &Tensor) -> AnyTensor {
    let _span = info_span!("realize_cpu").entered();
    let graph = tensor.graph().read().unwrap();
    let order = topo_order(&graph, tensor.id());
    let num_nodes = order.len();
    let _walk = info_span!("topo_walk", nodes = num_nodes).entered();
    let mut cache: HashMap<NodeId, AnyTensor> = HashMap::new();

    for id in order {
        let node = graph.node(id);
        // Phase 7.5 G2: slot-first dispatch.
        if let Some(adopted) = try_adopt_slot_cpu(&graph, id, &node.shape) {
            cache.insert(id, adopted);
            continue;
        }
        let result = eval_node_with_graph_context(&graph, id, node, &cache);
        cache.insert(id, result);
    }
    drop(_walk);

    cache
        .remove(&tensor.id())
        .expect("realize: target tensor missing from cache after topo walk")
}

/// Wrap per-node `eval_node` in `catch_unwind` so a downstream panic
/// (unsupported dtype combo, shape mismatch the builder didn't catch,
/// etc.) re-panics with a prepended graph-location identifier. See
/// the sibling helper in `fuel-reference-backend/src/exec.rs` for the
/// same pattern — both realize paths produce identically-formatted
/// augmented error messages so debug output looks the same regardless
/// of which executor was running.
fn eval_node_with_graph_context(
    graph: &fuel_graph::Graph,
    id: NodeId,
    node: &fuel_graph::Node,
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    use std::panic::{catch_unwind, AssertUnwindSafe, resume_unwind};
    let inputs = node.inputs.clone();
    let shape = node.shape.clone();
    let op = node.op.clone();
    let result = catch_unwind(AssertUnwindSafe(|| {
        eval_node(&op, &inputs, &shape, cache)
    }));
    match result {
        Ok(t) => t,
        Err(payload) => {
            let original = panic_payload_to_string(&payload);
            let location = graph.describe_node(id);
            let msg = format!(
                "fuel-graph-cpu realize: panic at {location}\n  original panic: {original}"
            );
            resume_unwind(Box::new(msg))
        }
    }
}

fn panic_payload_to_string(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&'static str>() { return s.to_string(); }
    if let Some(s) = p.downcast_ref::<String>()       { return s.clone();     }
    "<non-string panic payload>".to_string()
}

// Dispatch macros mirroring `fuel_reference_backend::exec`'s, but
// specialized to this crate's `AnyTensor` variant. Every match arm
// delegates to a `fuel_reference_backend::ops` function — except
// `MatMul`, which dispatches to the fast path defined in this crate.

macro_rules! unary {
    ($inputs:expr, $cache:expr, $func:path) => {{
        let x = $cache.get(&$inputs[0]).expect("topo order missing input");
        match x {
            AnyTensor::F32(t) => AnyTensor::F32($func(t)),
            AnyTensor::F64(t) => AnyTensor::F64($func(t)),
            AnyTensor::BF16(t) => AnyTensor::BF16($func(t)),
            AnyTensor::F16(t) => AnyTensor::F16($func(t)),
            AnyTensor::U32(_) => panic!(
                "{}: not supported on U32 tensors",
                stringify!($func),
            ),
        }
    }};
}

macro_rules! unary_with_dim {
    ($inputs:expr, $cache:expr, $func:path, $dim:expr) => {{
        let x = $cache.get(&$inputs[0]).expect("topo order missing input");
        match x {
            AnyTensor::F32(t) => AnyTensor::F32($func(t, $dim)),
            AnyTensor::F64(t) => AnyTensor::F64($func(t, $dim)),
            AnyTensor::BF16(t) => AnyTensor::BF16($func(t, $dim)),
            AnyTensor::F16(t) => AnyTensor::F16($func(t, $dim)),
            AnyTensor::U32(_) => panic!(
                "{}: not supported on U32 tensors",
                stringify!($func),
            ),
        }
    }};
}

macro_rules! binary {
    ($inputs:expr, $cache:expr, $func:path) => {{
        let a = $cache.get(&$inputs[0]).expect("topo order missing lhs");
        let b = $cache.get(&$inputs[1]).expect("topo order missing rhs");
        match (a, b) {
            (AnyTensor::F32(a), AnyTensor::F32(b)) => AnyTensor::F32($func(a, b)),
            (AnyTensor::F64(a), AnyTensor::F64(b)) => AnyTensor::F64($func(a, b)),
            (AnyTensor::BF16(a), AnyTensor::BF16(b)) => AnyTensor::BF16($func(a, b)),
            (AnyTensor::F16(a), AnyTensor::F16(b)) => AnyTensor::F16($func(a, b)),
            (a, b) => panic!(
                "{}: unsupported operand dtypes (lhs={:?}, rhs={:?})",
                stringify!($func),
                a.dtype(),
                b.dtype(),
            ),
        }
    }};
}

fn eval_node(
    op: &Op,
    inputs: &[NodeId],
    shape: &fuel_core_types::Shape,
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    match op {
        // Op::Const is intercepted by slot-first dispatch in the
        // realize loops above (try_adopt_slot_cpu). If we get here
        // it means a Const node was constructed without slot-population
        // — a bug.
        Op::Const => unreachable!(
            "fuel-graph-cpu eval_node: Op::Const must be handled by \
             slot-first dispatch in the realize loop, never reach eval_node",
        ),

        // --- the fast path ---
        //
        // MatMul dispatches to a gemm-backed implementation that is
        // 50-200x faster than the reference matmul for the matrix
        // sizes that appear in transformer forward passes. All other
        // ops go through the reference backend.
        Op::MatMul => eval_matmul(inputs, cache),

        // --- element-wise binary ---
        Op::Add => binary!(inputs, cache, ops::add),
        Op::Sub => binary!(inputs, cache, ops::sub),
        Op::Mul => binary!(inputs, cache, ops::mul),
        Op::Div => binary!(inputs, cache, ops::div),

        // --- element-wise unary ---
        Op::Neg => unary!(inputs, cache, ops::neg),
        Op::Sqr => unary!(inputs, cache, ops::sqr),
        Op::Sqrt => unary!(inputs, cache, ops::sqrt),
        Op::Exp => unary!(inputs, cache, ops::exp),
        Op::Log => unary!(inputs, cache, ops::log),
        Op::Sin => unary!(inputs, cache, ops::sin),
        Op::Cos => unary!(inputs, cache, ops::cos),
        Op::Tanh => unary!(inputs, cache, ops::tanh),
        Op::Sigmoid => unary!(inputs, cache, ops::sigmoid),
        Op::Silu => unary!(inputs, cache, ops::silu),
        Op::Gelu => unary!(inputs, cache, ops::gelu),
        Op::Relu => unary!(inputs, cache, ops::relu),
        Op::Step => unary!(inputs, cache, ops::step),
        Op::Recip => unary!(inputs, cache, ops::recip),
        Op::Abs => unary!(inputs, cache, ops::abs),
        Op::Floor => unary!(inputs, cache, ops::floor),
        Op::Ceil => unary!(inputs, cache, ops::ceil),
        Op::Round => unary!(inputs, cache, ops::round),
        Op::Sign => unary!(inputs, cache, ops::sign),
        Op::Erf => unary!(inputs, cache, ops::erf),
        Op::GeluErf => unary!(inputs, cache, ops::gelu_erf),
        Op::Pow => binary!(inputs, cache, ops::pow),
        Op::Rsqrt => unary!(inputs, cache, ops::rsqrt),
        Op::Rem => binary!(inputs, cache, ops::rem),

        // --- comparison family (output dtype = U8) ---
        // Comparison ops produce a U8 mask; the legacy AnyTensor enum
        // here only carries float + U32 variants, so realize-via-graph-cpu
        // can't represent the result. The storage-path executor
        // (`PipelinedExecutor`) is the canonical implementation; tests
        // and downstream consumers should route through it for
        // comparison-op coverage.
        Op::Equal => panic!(
            "Op::Equal: legacy fuel-graph-cpu executor doesn't support \
             U8-output ops; use the storage-path PipelinedExecutor instead",
        ),
        Op::Ne => panic!(
            "Op::Ne: legacy fuel-graph-cpu executor doesn't support \
             U8-output ops; use the storage-path PipelinedExecutor instead",
        ),
        Op::Lt => panic!(
            "Op::Lt: legacy fuel-graph-cpu executor doesn't support \
             U8-output ops; use the storage-path PipelinedExecutor instead",
        ),
        Op::Le => panic!(
            "Op::Le: legacy fuel-graph-cpu executor doesn't support \
             U8-output ops; use the storage-path PipelinedExecutor instead",
        ),
        Op::Gt => panic!(
            "Op::Gt: legacy fuel-graph-cpu executor doesn't support \
             U8-output ops; use the storage-path PipelinedExecutor instead",
        ),
        Op::Ge => panic!(
            "Op::Ge: legacy fuel-graph-cpu executor doesn't support \
             U8-output ops; use the storage-path PipelinedExecutor instead",
        ),
        Op::Where => panic!(
            "Op::Where: legacy fuel-graph-cpu executor doesn't support \
             ternary U8-cond ops; use the storage-path PipelinedExecutor instead",
        ),

        // --- linear algebra & shape ---
        Op::Transpose => unary!(inputs, cache, ops::transpose_last_two),
        Op::Permute(axes) => eval_permute(axes, inputs, cache),

        // --- 2-D convolution (defers to reference nested loops for
        // now; a gemm-backed im2col fast-path is a follow-up) ---
        Op::Conv2D { stride, padding, groups } => {
            eval_conv2d(*stride, *padding, *groups, inputs, cache)
        }
        Op::ConvTranspose2D { stride, padding, output_padding, dilation, groups } => {
            eval_conv_transpose2d(*stride, *padding, *output_padding, *dilation, *groups, inputs, cache)
        }
        Op::FlashAttn { softmax_scale, causal, window_size_left, window_size_right, softcap } => {
            eval_flash_attn(
                *softmax_scale, *causal, *window_size_left, *window_size_right, *softcap,
                inputs, cache,
            )
        }
        Op::PagedAttn { softmax_scale, block_size, softcap } => {
            eval_paged_attn(*softmax_scale, *block_size, *softcap, inputs, cache)
        }
        Op::FusedLinear => eval_fused_linear(inputs, cache),

        // --- dtype, shape, broadcasting ---
        Op::Cast(target) => eval_cast(*target, inputs, cache),
        Op::BroadcastTo(target_shape) => eval_broadcast_to(target_shape, inputs, cache),
        Op::Reshape(target_shape) => eval_reshape(target_shape, inputs, cache),
        Op::ReduceSumTo(target_shape) => eval_reduce_sum_to(target_shape, inputs, cache),
        Op::ReduceMaxTo(target_shape) => eval_reduce_max_to(target_shape, inputs, cache),
        Op::Unsqueeze { dim } => eval_unsqueeze(*dim, inputs, cache),
        Op::Squeeze { dim } => eval_squeeze(*dim, inputs, cache),

        // --- reductions ---
        Op::SumAll => unary!(inputs, cache, ops::sum_all),
        Op::MaxAll => unary!(inputs, cache, ops::max_all),
        Op::MinAll => unary!(inputs, cache, ops::min_all),
        Op::MeanAll => unary!(inputs, cache, ops::mean_all),
        Op::SumDim(d) => unary_with_dim!(inputs, cache, ops::sum_dim, *d),
        Op::MaxDim(d) => unary_with_dim!(inputs, cache, ops::max_dim, *d),
        Op::MinDim(d) => unary_with_dim!(inputs, cache, ops::min_dim, *d),
        Op::MeanDim(d) => unary_with_dim!(inputs, cache, ops::mean_dim, *d),
        Op::ArgMaxDim(d) => eval_argindex_dim(*d, inputs, cache, true),
        Op::ArgMinDim(d) => eval_argindex_dim(*d, inputs, cache, false),

        // --- compositions ---
        Op::SoftmaxLastDim => unary!(inputs, cache, ops::softmax_last_dim),
        // Phase 7.6 step 3: dispatch the registry-extended fused arm to
        // the same primitive kernel as the legacy variant.
        Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::SOFTMAX_LAST_DIM => {
            unary!(inputs, cache, ops::softmax_last_dim)
        }
        Op::LayerNormLastDim { eps } => eval_layer_norm_last_dim(*eps, inputs, cache),
        Op::RmsNormLastDim { eps } => eval_rms_norm_last_dim(*eps, inputs, cache),
        Op::Rope => eval_rope(inputs, cache),
        Op::QMatMul { quant_type, k, n } => eval_qmatmul(*quant_type, *k, *n, inputs, cache),
        Op::RmsNormLastDimBackward { eps } => eval_rms_norm_last_dim_backward(*eps, inputs, cache),
        Op::SoftmaxLastDimBackward => eval_softmax_last_dim_backward(inputs, cache),
        Op::ReduceMaxToBackward => eval_reduce_max_to_backward(inputs, cache),
        Op::LayerNormLastDimBackward { eps } => {
            eval_layer_norm_last_dim_backward(*eps, inputs, cache)
        }

        // --- indexing ---
        Op::IndexSelect { dim } => eval_index_select(*dim, inputs, cache),
        Op::Gather { dim } => eval_gather(*dim, inputs, cache),
        Op::IndexAdd { dim } => eval_index_add(*dim, inputs, cache),
        Op::ScatterAdd { dim } => eval_scatter_add(*dim, inputs, cache),

        // --- shape manipulation ---
        Op::Concat { dim } => eval_concat(*dim, inputs, cache),
        Op::Slice { dim, start, len } => eval_slice(*dim, *start, *len, inputs, cache),

        // --- scalar ops ---
        Op::AddScalar(c) => eval_add_scalar(*c, inputs, cache),
        Op::MulScalar(c) => eval_mul_scalar(*c, inputs, cache),
        Op::PowI(n) => eval_powi(*n, inputs, cache),
        Op::Clamp { min, max } => eval_clamp(*min, *max, inputs, cache),
        Op::Maximum => binary!(inputs, cache, ops::maximum),
        Op::Minimum => binary!(inputs, cache, ops::minimum),
        Op::Copy { .. } | Op::Move { .. } => {
            // CPU-only context: Copy/Move is a pass-through (the input
            // is already on CPU; the target is trivially CPU since
            // there's no other device in this executor). The
            // destructive semantics of Move kick in at the executor
            // cache layer, not here.
            cache.get(&inputs[0]).expect("topo order missing copy/move input").clone()
        }
        Op::Release => {
            // Release on CPU: return a zero-element marker. The actual
            // refcount-based dealloc happens when the caller's cache
            // drops its entry; Release signals the scheduler that the
            // input should not be considered live beyond this point.
            AnyTensor::F32(RefTensor::from_arc(
                std::sync::Arc::<[f32]>::from(Vec::<f32>::new()),
                fuel_core_types::Shape::from_dims(&[0]),
            ))
        }
        Op::Fused(fid, _params) => {
            // Phase 7.6 step 3: per-id arms handle the migrated fused
            // ops (only SoftmaxLastDim today; step 4 adds the rest).
            unreachable!(
                "fuel-graph-cpu eval_node: Op::Fused id {fid:?} has no \
                 dispatch arm wired yet. Step 4 extends this match.",
            );
        }
    }
}

// The fast path: dispatch matmul to gemm for f32/f64, fall through to
// reference for bf16/f16 (since gemm doesn't support them directly —
// callers wanting fast bf16 matmul should cast to f32 first).
fn eval_matmul(inputs: &[NodeId], cache: &HashMap<NodeId, AnyTensor>) -> AnyTensor {
    let a = cache.get(&inputs[0]).expect("matmul missing lhs");
    let b = cache.get(&inputs[1]).expect("matmul missing rhs");
    match (a, b) {
        (AnyTensor::F32(a), AnyTensor::F32(b)) => {
            AnyTensor::F32(fast_matmul::matmul_f32(a, b))
        }
        (AnyTensor::F64(a), AnyTensor::F64(b)) => {
            AnyTensor::F64(fast_matmul::matmul_f64(a, b))
        }
        // bf16/f16: fall back to the reference matmul. This is slow
        // but correct. For speed, cast to f32 first via `Op::Cast`.
        (AnyTensor::BF16(a), AnyTensor::BF16(b)) => AnyTensor::BF16(ops::matmul(a, b)),
        (AnyTensor::F16(a), AnyTensor::F16(b)) => AnyTensor::F16(ops::matmul(a, b)),
        // Mixed-precision: activations f32 × weights bf16 → f32. Upcast
        // B to f32 and run the fast f32 matmul. The result matches what
        // the Vulkan bf16-unpack kernels compute (both read B as bf16,
        // extend to f32 exactly before FMA).
        (AnyTensor::F32(a), AnyTensor::BF16(b)) => {
            let b_data: Vec<f32> = b.as_slice().iter().map(|x| x.to_f32()).collect();
            let b_f32 = fuel_reference_backend::RefTensor::from_vec(b_data, b.shape().clone());
            AnyTensor::F32(fast_matmul::matmul_f32(a, &b_f32))
        }
        (a, b) => panic!(
            "matmul: unsupported operand dtypes (lhs={:?}, rhs={:?})",
            a.dtype(),
            b.dtype(),
        ),
    }
}

fn eval_conv2d(
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let x = cache.get(&inputs[0]).expect("conv2d: missing x");
    let w = cache.get(&inputs[1]).expect("conv2d: missing weight");
    let b = inputs.get(2).and_then(|id| cache.get(id));
    match (x, w, b) {
        (AnyTensor::F32(x), AnyTensor::F32(w), Some(AnyTensor::F32(bias))) => {
            AnyTensor::F32(ops::conv2d(x, w, Some(bias), stride, padding, groups))
        }
        (AnyTensor::F32(x), AnyTensor::F32(w), None) => {
            AnyTensor::F32(ops::conv2d(x, w, None, stride, padding, groups))
        }
        (AnyTensor::F64(x), AnyTensor::F64(w), Some(AnyTensor::F64(bias))) => {
            AnyTensor::F64(ops::conv2d(x, w, Some(bias), stride, padding, groups))
        }
        (AnyTensor::F64(x), AnyTensor::F64(w), None) => {
            AnyTensor::F64(ops::conv2d(x, w, None, stride, padding, groups))
        }
        (a, b_, c_) => panic!(
            "conv2d: unsupported operand dtype combination x={:?} w={:?} bias={:?}",
            a.dtype(), b_.dtype(), c_.map(|t| t.dtype()),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_conv_transpose2d(
    stride:         (usize, usize),
    padding:        (usize, usize),
    output_padding: (usize, usize),
    dilation:       (usize, usize),
    groups:         usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let x = cache.get(&inputs[0]).expect("conv_transpose2d: missing x");
    let w = cache.get(&inputs[1]).expect("conv_transpose2d: missing weight");
    match (x, w) {
        (AnyTensor::F32(x), AnyTensor::F32(w)) => {
            AnyTensor::F32(ops::conv_transpose2d(x, w, stride, padding, output_padding, dilation, groups))
        }
        (AnyTensor::F64(x), AnyTensor::F64(w)) => {
            AnyTensor::F64(ops::conv_transpose2d(x, w, stride, padding, output_padding, dilation, groups))
        }
        (a, b) => panic!(
            "conv_transpose2d: unsupported operand dtype combination x={:?} w={:?}",
            a.dtype(), b.dtype(),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_flash_attn(
    softmax_scale:     f32,
    causal:            bool,
    window_size_left:  Option<usize>,
    window_size_right: Option<usize>,
    softcap:           Option<f32>,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    use fuel_reference_backend::attention::{attention_naive, AttentionParams};
    let q = cache.get(&inputs[0]).expect("flash_attn: missing q");
    let k = cache.get(&inputs[1]).expect("flash_attn: missing k");
    let v = cache.get(&inputs[2]).expect("flash_attn: missing v");
    let alibi = inputs.get(3).and_then(|id| cache.get(id));
    let p = AttentionParams {
        softmax_scale, causal, window_size_left, window_size_right, softcap,
    };
    match (q, k, v, alibi) {
        (AnyTensor::F32(q), AnyTensor::F32(k), AnyTensor::F32(v), Some(AnyTensor::F32(a))) => {
            AnyTensor::F32(attention_naive(q, k, v, Some(a), &p))
        }
        (AnyTensor::F32(q), AnyTensor::F32(k), AnyTensor::F32(v), None) => {
            AnyTensor::F32(attention_naive(q, k, v, None, &p))
        }
        (AnyTensor::F64(q), AnyTensor::F64(k), AnyTensor::F64(v), Some(AnyTensor::F64(a))) => {
            AnyTensor::F64(attention_naive(q, k, v, Some(a), &p))
        }
        (AnyTensor::F64(q), AnyTensor::F64(k), AnyTensor::F64(v), None) => {
            AnyTensor::F64(attention_naive(q, k, v, None, &p))
        }
        (qa, ka, va, alba) => panic!(
            "flash_attn: unsupported operand dtype combination q={:?} k={:?} v={:?} alibi={:?}",
            qa.dtype(), ka.dtype(), va.dtype(), alba.map(|t| t.dtype()),
        ),
    }
}

fn eval_fused_linear(
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let a = cache.get(&inputs[0]).expect("fused_linear: missing a");
    let b = cache.get(&inputs[1]).expect("fused_linear: missing b");
    let bias = cache.get(&inputs[2]).expect("fused_linear: missing bias");
    let mm = match (a, b) {
        (AnyTensor::F32(a), AnyTensor::F32(b)) => AnyTensor::F32(ops::matmul(a, b)),
        (AnyTensor::F64(a), AnyTensor::F64(b)) => AnyTensor::F64(ops::matmul(a, b)),
        _ => panic!("fused_linear: unsupported matmul dtype combination a={:?} b={:?}", a.dtype(), b.dtype()),
    };
    match (&mm, bias) {
        (AnyTensor::F32(mm_t), AnyTensor::F32(bt)) => {
            let bias_b = ops::broadcast_to(bt, mm_t.shape());
            AnyTensor::F32(ops::add(mm_t, &bias_b))
        }
        (AnyTensor::F64(mm_t), AnyTensor::F64(bt)) => {
            let bias_b = ops::broadcast_to(bt, mm_t.shape());
            AnyTensor::F64(ops::add(mm_t, &bias_b))
        }
        (mm_a, b_a) => panic!(
            "fused_linear: bias dtype {:?} must match matmul dtype {:?}",
            b_a.dtype(), mm_a.dtype(),
        ),
    }
}

fn eval_paged_attn(
    softmax_scale: f32,
    block_size: usize,
    softcap: Option<f32>,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    use fuel_reference_backend::attention::attention_paged_naive;
    let q  = cache.get(&inputs[0]).expect("paged_attn: missing q");
    let kc = cache.get(&inputs[1]).expect("paged_attn: missing k_cache");
    let vc = cache.get(&inputs[2]).expect("paged_attn: missing v_cache");
    let bt = cache.get(&inputs[3]).expect("paged_attn: missing block_table");
    let cl = cache.get(&inputs[4]).expect("paged_attn: missing context_lens");
    let alibi = inputs.get(5).and_then(|id| cache.get(id));
    let block_table = match bt {
        AnyTensor::U32(t) => t,
        other => panic!("paged_attn: block_table must be U32, got {:?}", other.dtype()),
    };
    let context_lens = match cl {
        AnyTensor::U32(t) => t,
        other => panic!("paged_attn: context_lens must be U32, got {:?}", other.dtype()),
    };
    match (q, kc, vc, alibi) {
        (AnyTensor::F32(q), AnyTensor::F32(kc), AnyTensor::F32(vc), Some(AnyTensor::F32(a))) => {
            AnyTensor::F32(attention_paged_naive(q, kc, vc, block_table, context_lens, Some(a), softmax_scale, block_size, softcap))
        }
        (AnyTensor::F32(q), AnyTensor::F32(kc), AnyTensor::F32(vc), None) => {
            AnyTensor::F32(attention_paged_naive(q, kc, vc, block_table, context_lens, None, softmax_scale, block_size, softcap))
        }
        (AnyTensor::F64(q), AnyTensor::F64(kc), AnyTensor::F64(vc), Some(AnyTensor::F64(a))) => {
            AnyTensor::F64(attention_paged_naive(q, kc, vc, block_table, context_lens, Some(a), softmax_scale, block_size, softcap))
        }
        (AnyTensor::F64(q), AnyTensor::F64(kc), AnyTensor::F64(vc), None) => {
            AnyTensor::F64(attention_paged_naive(q, kc, vc, block_table, context_lens, None, softmax_scale, block_size, softcap))
        }
        (qa, kca, vca, alba) => panic!(
            "paged_attn: unsupported operand dtype combination q={:?} k={:?} v={:?} alibi={:?}",
            qa.dtype(), kca.dtype(), vca.dtype(), alba.map(|t| t.dtype()),
        ),
    }
}

// All of the remaining eval_* functions are direct copies of
// `fuel_reference_backend::exec`'s implementations. They exist here
// because the reference backend's exec internals are not public — we
// can't import `eval_cast` etc. directly, so we re-implement each
// dispatcher to call the public `ops::*` functions. If the reference
// crate ever exposes its dispatchers, these can become one-line
// delegates.

fn eval_permute(
    axes: &[usize],
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let x = cache.get(&inputs[0]).expect("permute missing input");
    match x {
        AnyTensor::F32(t) => AnyTensor::F32(ops::permute(t, axes)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::permute(t, axes)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::permute(t, axes)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::permute(t, axes)),
        AnyTensor::U32(t) => AnyTensor::U32(ops::permute(t, axes)),
    }
}

fn eval_cast(
    target: fuel_core_types::DType,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    use fuel_core_types::DType;
    let src = cache.get(&inputs[0]).expect("cast missing input");
    match (src, target) {
        // Identity casts.
        (AnyTensor::F32(t), DType::F32) => AnyTensor::F32(t.clone()),
        (AnyTensor::F64(t), DType::F64) => AnyTensor::F64(t.clone()),
        (AnyTensor::BF16(t), DType::BF16) => AnyTensor::BF16(t.clone()),
        (AnyTensor::F16(t), DType::F16) => AnyTensor::F16(t.clone()),
        (AnyTensor::U32(t), DType::U32) => AnyTensor::U32(t.clone()),

        // Float-to-float.
        (AnyTensor::F32(t), DType::F64) => AnyTensor::F64(ops::cast_f32_to_f64(t)),
        (AnyTensor::F32(t), DType::BF16) => AnyTensor::BF16(ops::cast_f32_to_bf16(t)),
        (AnyTensor::F32(t), DType::F16) => AnyTensor::F16(ops::cast_f32_to_f16(t)),
        (AnyTensor::F64(t), DType::F32) => AnyTensor::F32(ops::cast_f64_to_f32(t)),
        (AnyTensor::F64(t), DType::BF16) => AnyTensor::BF16(ops::cast_f64_to_bf16(t)),
        (AnyTensor::F64(t), DType::F16) => AnyTensor::F16(ops::cast_f64_to_f16(t)),
        (AnyTensor::BF16(t), DType::F32) => AnyTensor::F32(ops::cast_bf16_to_f32(t)),
        (AnyTensor::BF16(t), DType::F64) => AnyTensor::F64(ops::cast_bf16_to_f64(t)),
        (AnyTensor::BF16(t), DType::F16) => AnyTensor::F16(ops::cast_bf16_to_f16(t)),
        (AnyTensor::F16(t), DType::F32) => AnyTensor::F32(ops::cast_f16_to_f32(t)),
        (AnyTensor::F16(t), DType::F64) => AnyTensor::F64(ops::cast_f16_to_f64(t)),
        (AnyTensor::F16(t), DType::BF16) => AnyTensor::BF16(ops::cast_f16_to_bf16(t)),

        // Int/float.
        (AnyTensor::U32(t), DType::F32) => AnyTensor::F32(ops::cast_u32_to_f32(t)),
        (AnyTensor::U32(t), DType::F64) => AnyTensor::F64(ops::cast_u32_to_f64(t)),
        (AnyTensor::F32(t), DType::U32) => AnyTensor::U32(ops::cast_f32_to_u32(t)),
        (AnyTensor::F64(t), DType::U32) => AnyTensor::U32(ops::cast_f64_to_u32(t)),

        (src, dst) => panic!(
            "cast: unsupported dtype combination {:?} -> {dst:?}",
            src.dtype(),
        ),
    }
}

fn eval_broadcast_to(
    target: &fuel_core_types::Shape,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let src = cache.get(&inputs[0]).expect("broadcast_to missing input");
    match src {
        AnyTensor::F32(t) => AnyTensor::F32(ops::broadcast_to(t, target)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::broadcast_to(t, target)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::broadcast_to(t, target)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::broadcast_to(t, target)),
        AnyTensor::U32(_) => panic!("broadcast_to: not supported on U32 tensors"),
    }
}

fn eval_reshape(
    target: &fuel_core_types::Shape,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let src = cache.get(&inputs[0]).expect("reshape missing input");
    match src {
        AnyTensor::F32(t) => AnyTensor::F32(ops::reshape(t, target)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::reshape(t, target)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::reshape(t, target)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::reshape(t, target)),
        AnyTensor::U32(t) => AnyTensor::U32(ops::reshape(t, target)),
    }
}

fn eval_reduce_sum_to(
    target: &fuel_core_types::Shape,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let src = cache.get(&inputs[0]).expect("reduce_sum_to missing input");
    match src {
        AnyTensor::F32(t) => AnyTensor::F32(ops::reduce_sum_to(t, target)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::reduce_sum_to(t, target)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::reduce_sum_to(t, target)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::reduce_sum_to(t, target)),
        AnyTensor::U32(_) => panic!("reduce_sum_to: not supported on U32 tensors"),
    }
}

fn eval_reduce_max_to(
    target: &fuel_core_types::Shape,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let src = cache.get(&inputs[0]).expect("reduce_max_to missing input");
    match src {
        AnyTensor::F32(t) => AnyTensor::F32(ops::reduce_max_to(t, target)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::reduce_max_to(t, target)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::reduce_max_to(t, target)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::reduce_max_to(t, target)),
        AnyTensor::U32(_) => panic!("reduce_max_to: not supported on U32 tensors"),
    }
}

fn eval_unsqueeze(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    // Unsqueeze is bytes-identical with reshape; only the metadata
    // shape differs.
    let src = cache.get(&inputs[0]).expect("unsqueeze missing input");
    let in_dims = match src {
        AnyTensor::F32(t) => t.shape().dims().to_vec(),
        AnyTensor::F64(t) => t.shape().dims().to_vec(),
        AnyTensor::BF16(t) => t.shape().dims().to_vec(),
        AnyTensor::F16(t) => t.shape().dims().to_vec(),
        AnyTensor::U32(t) => t.shape().dims().to_vec(),
    };
    let mut out_dims = in_dims;
    assert!(
        dim <= out_dims.len(),
        "unsqueeze: dim {dim} out of bounds for rank {}",
        out_dims.len(),
    );
    out_dims.insert(dim, 1);
    let target = fuel_core_types::Shape::from_dims(&out_dims);
    match src {
        AnyTensor::F32(t) => AnyTensor::F32(ops::reshape(t, &target)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::reshape(t, &target)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::reshape(t, &target)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::reshape(t, &target)),
        AnyTensor::U32(t) => AnyTensor::U32(ops::reshape(t, &target)),
    }
}

/// Inverse of [`eval_unsqueeze`]: drop a size-1 dimension. Bytes are
/// unchanged; only the metadata shape differs. The builder already
/// validates `dim < rank` and `shape[dim] == 1`, so the executor just
/// reshapes.
fn eval_squeeze(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let src = cache.get(&inputs[0]).expect("squeeze missing input");
    let in_dims: Vec<usize> = match src {
        AnyTensor::F32(t) => t.shape().dims().to_vec(),
        AnyTensor::F64(t) => t.shape().dims().to_vec(),
        AnyTensor::BF16(t) => t.shape().dims().to_vec(),
        AnyTensor::F16(t) => t.shape().dims().to_vec(),
        AnyTensor::U32(t) => t.shape().dims().to_vec(),
    };
    assert!(
        dim < in_dims.len(),
        "squeeze: dim {dim} out of bounds for rank {}",
        in_dims.len(),
    );
    assert_eq!(
        in_dims[dim], 1,
        "squeeze: dim {dim} has size {}, expected 1",
        in_dims[dim],
    );
    let out_dims: Vec<usize> = in_dims.iter().enumerate()
        .filter_map(|(i, &d)| if i == dim { None } else { Some(d) })
        .collect();
    let target = fuel_core_types::Shape::from_dims(&out_dims);
    match src {
        AnyTensor::F32(t) => AnyTensor::F32(ops::reshape(t, &target)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::reshape(t, &target)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::reshape(t, &target)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::reshape(t, &target)),
        AnyTensor::U32(t) => AnyTensor::U32(ops::reshape(t, &target)),
    }
}

fn eval_argindex_dim(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
    is_max: bool,
) -> AnyTensor {
    let x = cache.get(&inputs[0]).expect("argindex missing input");
    let result = match x {
        AnyTensor::F32(t) => {
            if is_max {
                ops::argmax_dim(t, dim)
            } else {
                ops::argmin_dim(t, dim)
            }
        }
        AnyTensor::F64(t) => {
            if is_max {
                ops::argmax_dim(t, dim)
            } else {
                ops::argmin_dim(t, dim)
            }
        }
        AnyTensor::BF16(t) => {
            if is_max {
                ops::argmax_dim(t, dim)
            } else {
                ops::argmin_dim(t, dim)
            }
        }
        AnyTensor::F16(t) => {
            if is_max {
                ops::argmax_dim(t, dim)
            } else {
                ops::argmin_dim(t, dim)
            }
        }
        AnyTensor::U32(_) => panic!("argmax/argmin not supported on U32 input tensors"),
    };
    AnyTensor::U32(result)
}

fn eval_layer_norm_last_dim(
    eps: f64,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let src = cache.get(&inputs[0]).expect("layer_norm missing input");
    match src {
        AnyTensor::F32(t) => AnyTensor::F32(ops::layer_norm_last_dim(t, eps)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::layer_norm_last_dim(t, eps)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::layer_norm_last_dim(t, eps)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::layer_norm_last_dim(t, eps)),
        AnyTensor::U32(_) => panic!("layer_norm_last_dim: not supported on U32 tensors"),
    }
}

fn eval_rms_norm_last_dim(
    eps: f64,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let src = cache.get(&inputs[0]).expect("rms_norm missing input");
    match src {
        AnyTensor::F32(t) => AnyTensor::F32(ops::rms_norm_last_dim(t, eps)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::rms_norm_last_dim(t, eps)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::rms_norm_last_dim(t, eps)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::rms_norm_last_dim(t, eps)),
        AnyTensor::U32(_) => panic!("rms_norm_last_dim: not supported on U32 tensors"),
    }
}

fn eval_rms_norm_last_dim_backward(
    eps: f64,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let x = cache.get(&inputs[0]).expect("rms_norm_bwd missing x");
    let g = cache.get(&inputs[1]).expect("rms_norm_bwd missing g");
    match (x, g) {
        (AnyTensor::F32(x), AnyTensor::F32(g)) => {
            AnyTensor::F32(ops::rms_norm_last_dim_backward(x, g, eps))
        }
        (AnyTensor::F64(x), AnyTensor::F64(g)) => {
            AnyTensor::F64(ops::rms_norm_last_dim_backward(x, g, eps))
        }
        (AnyTensor::BF16(x), AnyTensor::BF16(g)) => {
            AnyTensor::BF16(ops::rms_norm_last_dim_backward(x, g, eps))
        }
        (AnyTensor::F16(x), AnyTensor::F16(g)) => {
            AnyTensor::F16(ops::rms_norm_last_dim_backward(x, g, eps))
        }
        _ => panic!("rms_norm_last_dim_backward: dtype mismatch"),
    }
}

fn eval_rope(
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let x = cache.get(&inputs[0]).expect("rope missing x");
    let cos = cache.get(&inputs[1]).expect("rope missing cos");
    let sin = cache.get(&inputs[2]).expect("rope missing sin");
    match (x, cos, sin) {
        (AnyTensor::F32(x), AnyTensor::F32(c), AnyTensor::F32(s)) => {
            AnyTensor::F32(ops::rope(x, c, s))
        }
        (AnyTensor::F64(x), AnyTensor::F64(c), AnyTensor::F64(s)) => {
            AnyTensor::F64(ops::rope(x, c, s))
        }
        (AnyTensor::BF16(x), AnyTensor::BF16(c), AnyTensor::BF16(s)) => {
            AnyTensor::BF16(ops::rope(x, c, s))
        }
        (AnyTensor::F16(x), AnyTensor::F16(c), AnyTensor::F16(s)) => {
            AnyTensor::F16(ops::rope(x, c, s))
        }
        (a, b, c) => panic!(
            "rope: dtype mismatch x={:?} cos={:?} sin={:?}",
            a.dtype(), b.dtype(), c.dtype()
        ),
    }
}

fn eval_qmatmul(
    quant_type: fuel_graph::QuantType,
    k: usize,
    n: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let a = cache.get(&inputs[0]).expect("qmatmul missing activations");
    let w = cache.get(&inputs[1]).expect("qmatmul missing weight bytes");
    let a_f32 = match a {
        AnyTensor::F32(t) => t,
        _ => panic!("qmatmul: activations must be F32, got {:?}", a.dtype()),
    };
    let w_u32 = match w {
        AnyTensor::U32(t) => t,
        _ => panic!("qmatmul: weight bytes must be U32, got {:?}", w.dtype()),
    };
    let w_bytes: Vec<u8> = w_u32.as_slice().iter().flat_map(|&u| u.to_le_bytes()).collect();
    let w_deq = cpu_dequantize_blocks(&w_bytes, quant_type, n, k);
    let w_ref = RefTensor::from_vec(w_deq, fuel_core_types::Shape::from_dims(&[n, k]));
    // [N, K] → transpose → [K, N] for X @ W_t convention.
    let w_t = ops::transpose_last_two(&w_ref);
    AnyTensor::F32(ops::matmul(a_f32, &w_t))
}

/// CPU reference dequantization of Q-type blocks to F32 [n_rows, k_cols]
/// row-major. Must match the GPU `dequant_q4_0` / `dequant_q8_0` output.
fn cpu_dequantize_blocks(
    bytes: &[u8],
    quant_type: fuel_graph::QuantType,
    n_rows: usize,
    k_cols: usize,
) -> Vec<f32> {
    use half::f16;
    let bpb = quant_type.bytes_per_block();
    let epb = quant_type.elements_per_block();
    let blocks_per_row = k_cols / epb;
    let mut out = vec![0.0_f32; n_rows * k_cols];
    for row in 0..n_rows {
        for bi in 0..blocks_per_row {
            let block_off = (row * blocks_per_row + bi) * bpb;
            let out_base = row * k_cols + bi * epb;
            match quant_type {
                fuel_graph::QuantType::Q4_0 => {
                    let scale = f16::from_le_bytes([bytes[block_off], bytes[block_off + 1]]).to_f32();
                    for kk in 0..16 {
                        let packed = bytes[block_off + 2 + kk];
                        let lo = (packed & 0x0F) as i32 - 8;
                        let hi = ((packed >> 4) & 0x0F) as i32 - 8;
                        out[out_base + kk]       = lo as f32 * scale;
                        out[out_base + 16 + kk]  = hi as f32 * scale;
                    }
                }
                fuel_graph::QuantType::Q8_0 => {
                    let scale = f16::from_le_bytes([bytes[block_off], bytes[block_off + 1]]).to_f32();
                    for kk in 0..32 {
                        let q = bytes[block_off + 2 + kk] as i8 as i32;
                        out[out_base + kk] = q as f32 * scale;
                    }
                }
                fuel_graph::QuantType::Q4_K_M => {
                    cpu_dequantize_q4_km_block(
                        &bytes[block_off..block_off + 144],
                        &mut out[out_base..out_base + 256],
                    );
                }
                other => unimplemented!(
                    "fuel-graph-cpu legacy dequantize_blocks does not support {other:?} yet"
                ),
            }
        }
    }
    out
}

/// CPU reference dequant for one Q4_K_M super-block. Mirrors the
/// fuel-reference-backend implementation and the GPU kernel.
fn cpu_dequantize_q4_km_block(bytes: &[u8], out: &mut [f32]) {
    use half::f16;
    debug_assert_eq!(bytes.len(), 144);
    debug_assert_eq!(out.len(), 256);
    let d    = f16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
    let dmin = f16::from_le_bytes([bytes[2], bytes[3]]).to_f32();
    let scales: [u8; 12] = bytes[4..16].try_into().unwrap();
    let qs = &bytes[16..144];
    let get_sm = |j: usize| -> (u8, u8) {
        if j < 4 {
            (scales[j] & 63, scales[j + 4] & 63)
        } else {
            let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
            let m  = (scales[j + 4] >> 4)  | ((scales[j] >> 6) << 4);
            (sc, m)
        }
    };
    let mut is = 0;
    let mut ys_idx = 0;
    for j in (0..256).step_by(64) {
        let qsub = &qs[j / 2 .. j / 2 + 32];
        let (sc, m) = get_sm(is);
        let d1 = d * sc as f32;
        let m1 = dmin * m as f32;
        let (sc, m) = get_sm(is + 1);
        let d2 = d * sc as f32;
        let m2 = dmin * m as f32;
        for &q in qsub {
            out[ys_idx] = d1 * (q & 0xF) as f32 - m1;
            ys_idx += 1;
        }
        for &q in qsub {
            out[ys_idx] = d2 * (q >> 4) as f32 - m2;
            ys_idx += 1;
        }
        is += 2;
    }
}

fn eval_softmax_last_dim_backward(
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let y = cache.get(&inputs[0]).expect("softmax_bwd missing y");
    let g = cache.get(&inputs[1]).expect("softmax_bwd missing g");
    match (y, g) {
        (AnyTensor::F32(y), AnyTensor::F32(g)) => {
            AnyTensor::F32(ops::softmax_last_dim_backward(y, g))
        }
        (AnyTensor::F64(y), AnyTensor::F64(g)) => {
            AnyTensor::F64(ops::softmax_last_dim_backward(y, g))
        }
        (AnyTensor::BF16(y), AnyTensor::BF16(g)) => {
            AnyTensor::BF16(ops::softmax_last_dim_backward(y, g))
        }
        (AnyTensor::F16(y), AnyTensor::F16(g)) => {
            AnyTensor::F16(ops::softmax_last_dim_backward(y, g))
        }
        (a, b) => panic!("softmax_bwd dtype mismatch: {:?} vs {:?}", a.dtype(), b.dtype()),
    }
}

fn eval_reduce_max_to_backward(
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let x = cache.get(&inputs[0]).expect("reduce_max_to_bwd missing x");
    let up = cache.get(&inputs[1]).expect("reduce_max_to_bwd missing upstream");
    let target = match up {
        AnyTensor::F32(t) => t.shape().clone(),
        AnyTensor::F64(t) => t.shape().clone(),
        AnyTensor::BF16(t) => t.shape().clone(),
        AnyTensor::F16(t) => t.shape().clone(),
        AnyTensor::U32(_) => panic!(
            "reduce_max_to_backward: upstream must be float, got U32"
        ),
    };
    match (x, up) {
        (AnyTensor::F32(x), AnyTensor::F32(up)) => {
            AnyTensor::F32(ops::reduce_max_to_backward(x, up, &target))
        }
        (AnyTensor::F64(x), AnyTensor::F64(up)) => {
            AnyTensor::F64(ops::reduce_max_to_backward(x, up, &target))
        }
        (AnyTensor::BF16(x), AnyTensor::BF16(up)) => {
            AnyTensor::BF16(ops::reduce_max_to_backward(x, up, &target))
        }
        (AnyTensor::F16(x), AnyTensor::F16(up)) => {
            AnyTensor::F16(ops::reduce_max_to_backward(x, up, &target))
        }
        (a, b) => panic!("reduce_max_to_bwd dtype mismatch: {:?} vs {:?}", a.dtype(), b.dtype()),
    }
}

fn eval_layer_norm_last_dim_backward(
    eps: f64,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let x = cache.get(&inputs[0]).expect("ln_bwd missing x");
    let g = cache.get(&inputs[1]).expect("ln_bwd missing g");
    match (x, g) {
        (AnyTensor::F32(x), AnyTensor::F32(g)) => {
            AnyTensor::F32(ops::layer_norm_last_dim_backward(x, g, eps))
        }
        (AnyTensor::F64(x), AnyTensor::F64(g)) => {
            AnyTensor::F64(ops::layer_norm_last_dim_backward(x, g, eps))
        }
        (AnyTensor::BF16(x), AnyTensor::BF16(g)) => {
            AnyTensor::BF16(ops::layer_norm_last_dim_backward(x, g, eps))
        }
        (AnyTensor::F16(x), AnyTensor::F16(g)) => {
            AnyTensor::F16(ops::layer_norm_last_dim_backward(x, g, eps))
        }
        (a, b) => panic!("ln_bwd dtype mismatch: {:?} vs {:?}", a.dtype(), b.dtype()),
    }
}

fn eval_index_select(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let data = cache.get(&inputs[0]).expect("index_select missing data");
    let idx = match cache.get(&inputs[1]) {
        Some(AnyTensor::U32(t)) => t,
        _ => panic!("index_select: second input must be U32"),
    };
    match data {
        AnyTensor::F32(t) => AnyTensor::F32(ops::index_select_tensor(t, dim, idx)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::index_select_tensor(t, dim, idx)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::index_select_tensor(t, dim, idx)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::index_select_tensor(t, dim, idx)),
        AnyTensor::U32(t) => AnyTensor::U32(ops::index_select_tensor(t, dim, idx)),
    }
}

fn eval_gather(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let data = cache.get(&inputs[0]).expect("gather missing data");
    let idx = match cache.get(&inputs[1]) {
        Some(AnyTensor::U32(t)) => t,
        _ => panic!("gather: second input must be U32"),
    };
    match data {
        AnyTensor::F32(t) => AnyTensor::F32(ops::gather(t, dim, idx)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::gather(t, dim, idx)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::gather(t, dim, idx)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::gather(t, dim, idx)),
        AnyTensor::U32(t) => AnyTensor::U32(ops::gather(t, dim, idx)),
    }
}

fn eval_index_add(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let base = cache.get(&inputs[0]).expect("index_add missing base");
    let idx = match cache.get(&inputs[1]) {
        Some(AnyTensor::U32(t)) => t,
        _ => panic!("index_add: second input must be U32"),
    };
    let src = cache.get(&inputs[2]).expect("index_add missing src");
    match (base, src) {
        (AnyTensor::F32(b), AnyTensor::F32(s)) => {
            AnyTensor::F32(ops::index_add(b, dim, idx, s))
        }
        (AnyTensor::F64(b), AnyTensor::F64(s)) => {
            AnyTensor::F64(ops::index_add(b, dim, idx, s))
        }
        (AnyTensor::BF16(b), AnyTensor::BF16(s)) => {
            AnyTensor::BF16(ops::index_add(b, dim, idx, s))
        }
        (AnyTensor::F16(b), AnyTensor::F16(s)) => {
            AnyTensor::F16(ops::index_add(b, dim, idx, s))
        }
        (b, s) => panic!("index_add dtype mismatch: {:?} vs {:?}", b.dtype(), s.dtype()),
    }
}

fn eval_scatter_add(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let base = cache.get(&inputs[0]).expect("scatter_add missing base");
    let idx = match cache.get(&inputs[1]) {
        Some(AnyTensor::U32(t)) => t,
        _ => panic!("scatter_add: second input must be U32"),
    };
    let src = cache.get(&inputs[2]).expect("scatter_add missing src");
    match (base, src) {
        (AnyTensor::F32(b), AnyTensor::F32(s)) => {
            AnyTensor::F32(ops::scatter_add(b, dim, idx, s))
        }
        (AnyTensor::F64(b), AnyTensor::F64(s)) => {
            AnyTensor::F64(ops::scatter_add(b, dim, idx, s))
        }
        (AnyTensor::BF16(b), AnyTensor::BF16(s)) => {
            AnyTensor::BF16(ops::scatter_add(b, dim, idx, s))
        }
        (AnyTensor::F16(b), AnyTensor::F16(s)) => {
            AnyTensor::F16(ops::scatter_add(b, dim, idx, s))
        }
        (b, s) => panic!("scatter_add dtype mismatch: {:?} vs {:?}", b.dtype(), s.dtype()),
    }
}

fn eval_concat(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let a = cache.get(&inputs[0]).expect("concat missing lhs");
    let b = cache.get(&inputs[1]).expect("concat missing rhs");
    match (a, b) {
        (AnyTensor::F32(a), AnyTensor::F32(b)) => AnyTensor::F32(ops::concat(a, b, dim)),
        (AnyTensor::F64(a), AnyTensor::F64(b)) => AnyTensor::F64(ops::concat(a, b, dim)),
        (AnyTensor::BF16(a), AnyTensor::BF16(b)) => AnyTensor::BF16(ops::concat(a, b, dim)),
        (AnyTensor::F16(a), AnyTensor::F16(b)) => AnyTensor::F16(ops::concat(a, b, dim)),
        (AnyTensor::U32(a), AnyTensor::U32(b)) => AnyTensor::U32(ops::concat(a, b, dim)),
        (a, b) => panic!("concat dtype mismatch: {:?} vs {:?}", a.dtype(), b.dtype()),
    }
}

fn eval_slice(
    dim: usize,
    start: usize,
    len: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let x = cache.get(&inputs[0]).expect("slice missing input");
    match x {
        AnyTensor::F32(t) => AnyTensor::F32(ops::slice(t, dim, start, len)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::slice(t, dim, start, len)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::slice(t, dim, start, len)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::slice(t, dim, start, len)),
        AnyTensor::U32(t) => AnyTensor::U32(ops::slice(t, dim, start, len)),
    }
}

fn eval_add_scalar(c: f64, inputs: &[NodeId], cache: &HashMap<NodeId, AnyTensor>) -> AnyTensor {
    let x = cache.get(&inputs[0]).expect("add_scalar missing input");
    match x {
        AnyTensor::F32(t) => AnyTensor::F32(ops::add_scalar(t, c)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::add_scalar(t, c)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::add_scalar(t, c)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::add_scalar(t, c)),
        AnyTensor::U32(_) => panic!("add_scalar: not supported on U32 tensors"),
    }
}

fn eval_mul_scalar(c: f64, inputs: &[NodeId], cache: &HashMap<NodeId, AnyTensor>) -> AnyTensor {
    let x = cache.get(&inputs[0]).expect("mul_scalar missing input");
    match x {
        AnyTensor::F32(t) => AnyTensor::F32(ops::mul_scalar(t, c)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::mul_scalar(t, c)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::mul_scalar(t, c)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::mul_scalar(t, c)),
        AnyTensor::U32(_) => panic!("mul_scalar: not supported on U32 tensors"),
    }
}

fn eval_powi(n: i32, inputs: &[NodeId], cache: &HashMap<NodeId, AnyTensor>) -> AnyTensor {
    let x = cache.get(&inputs[0]).expect("powi missing input");
    match x {
        AnyTensor::F32(t) => AnyTensor::F32(ops::powi(t, n)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::powi(t, n)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::powi(t, n)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::powi(t, n)),
        AnyTensor::U32(_) => panic!("powi: not supported on U32 tensors"),
    }
}

fn eval_clamp(
    min: f64,
    max: f64,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyTensor>,
) -> AnyTensor {
    let x = cache.get(&inputs[0]).expect("clamp missing input");
    match x {
        AnyTensor::F32(t) => AnyTensor::F32(ops::clamp(t, min, max)),
        AnyTensor::F64(t) => AnyTensor::F64(ops::clamp(t, min, max)),
        AnyTensor::BF16(t) => AnyTensor::BF16(ops::clamp(t, min, max)),
        AnyTensor::F16(t) => AnyTensor::F16(ops::clamp(t, min, max)),
        AnyTensor::U32(_) => panic!("clamp: not supported on U32 tensors"),
    }
}

#[cfg(test)]
mod tests {
    /// Phase 7.5 G2: tests need a real device for slot-populating
    /// constructors. Singleton CpuBackendDevice via OnceLock.
    fn cpu_dev() -> &'static std::sync::Arc<dyn fuel_core_types::DynBackendDevice> {
        static D: std::sync::OnceLock<std::sync::Arc<dyn fuel_core_types::DynBackendDevice>>
            = std::sync::OnceLock::new();
        D.get_or_init(|| std::sync::Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
    }

    use super::*;
    use fuel_core_types::Shape;
    use fuel_graph::Tensor;

    /// Realize through both the reference backend and the fast backend
    /// and assert the results are element-wise equal. Shared helper
    /// for the equivalence tests below.
    fn assert_equivalent_f32(tensor: &Tensor) {
        let reference = fuel_reference_backend::exec::realize_f32(tensor);
        let fast = realize_f32(tensor);
        assert_eq!(reference.shape().dims(), fast.shape().dims(), "shape mismatch");
        assert_eq!(
            reference.as_slice().len(),
            fast.as_slice().len(),
            "length mismatch",
        );
        for (i, (&r, &f)) in reference
            .as_slice()
            .iter()
            .zip(fast.as_slice())
            .enumerate()
        {
            // For well-conditioned matmuls we should match exactly
            // modulo sum-order differences. Gemm uses a different
            // accumulation order than the naive triple-loop reference,
            // so we allow a small tolerance.
            let tol = 1e-4_f32;
            let diff = (r - f).abs();
            let rel = if r.abs() > 1e-6 { diff / r.abs() } else { diff };
            assert!(
                rel < tol,
                "at index {i}: reference={r}, fast={f} (rel {rel})",
            );
        }
    }

    #[test]
    fn matmul_matches_reference_small() {
        // 3×4 @ 4×5 — the smallest non-trivial matmul.
        let a = Tensor::from_f32(
            (0..12).map(|i| i as f32 * 0.1 - 0.5).collect::<Vec<_>>(),
            Shape::from_dims(&[3, 4]),
            cpu_dev(),
        );
        let b = a.const_f32_like(
            (0..20).map(|i| (i as f32 - 10.0) * 0.05).collect::<Vec<_>>(),
            Shape::from_dims(&[4, 5]),
        );
        let c = a.matmul(&b);
        assert_equivalent_f32(&c);
    }

    #[test]
    fn matmul_matches_reference_medium() {
        // 16×32 @ 32×8, mid-sized — exercises gemm's blocking.
        let a_data: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let b_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.02).cos()).collect();
        let a = Tensor::from_f32(a_data, Shape::from_dims(&[16, 32]), cpu_dev());
        let b = a.const_f32_like(b_data, Shape::from_dims(&[32, 8]));
        let c = a.matmul(&b);
        assert_equivalent_f32(&c);
    }

    #[test]
    fn matmul_matches_reference_batched() {
        // [2, 3, 4] @ [2, 4, 5] — batched rank-3 matmul.
        let a_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..40).map(|i| (i as f32 * 0.2) - 1.0).collect();
        let a = Tensor::from_f32(a_data, Shape::from_dims(&[2, 3, 4]), cpu_dev());
        let b = a.const_f32_like(b_data, Shape::from_dims(&[2, 4, 5]));
        let c = a.matmul(&b);
        assert_equivalent_f32(&c);
    }

    #[test]
    fn non_matmul_chain_matches_reference() {
        // (a + b) * a → relu → sqr — exercises the delegation paths.
        let a = Tensor::from_f32(
            vec![-1.0, 2.0, -3.0, 4.0],
            Shape::from_dims(&[4]),
            cpu_dev(),
        );
        let b = a.const_f32_like(vec![0.5, -0.5, 1.5, -1.5], Shape::from_dims(&[4]));
        let out = a.add(&b).mul(&a).relu().sqr();
        assert_equivalent_f32(&out);
    }

    #[test]
    fn full_mini_transformer_block_matches_reference() {
        // A small attention-only block. Exercises matmul, softmax,
        // transpose, permute, reshape, broadcast, and mul_scalar all
        // through the fast executor, verifying end-to-end equivalence.
        let seq = 3;
        let d_head = 4;
        let num_heads = 2;
        let d_model = num_heads * d_head;

        let x_data: Vec<f32> = (0..seq * d_model).map(|i| i as f32 * 0.02).collect();
        let x = Tensor::from_f32(x_data, Shape::from_dims(&[1, seq, d_model]), cpu_dev());
        let identity: Vec<f32> = {
            let mut v = vec![0.0_f32; d_model * d_model];
            for i in 0..d_model {
                v[i * d_model + i] = 1.0;
            }
            v
        };
        let w_q = x.const_f32_like(identity.clone(), Shape::from_dims(&[d_model, d_model]));
        let w_k = x.const_f32_like(identity.clone(), Shape::from_dims(&[d_model, d_model]));
        let w_v = x.const_f32_like(identity.clone(), Shape::from_dims(&[d_model, d_model]));

        let q = x.matmul(&w_q);
        let k = x.matmul(&w_k);
        let v = x.matmul(&w_v);
        let q_h = q
            .reshape(Shape::from_dims(&[1, seq, num_heads, d_head]))
            .permute(&[0, 2, 1, 3]);
        let k_h = k
            .reshape(Shape::from_dims(&[1, seq, num_heads, d_head]))
            .permute(&[0, 2, 1, 3]);
        let v_h = v
            .reshape(Shape::from_dims(&[1, seq, num_heads, d_head]))
            .permute(&[0, 2, 1, 3]);
        let scores = q_h.matmul(&k_h.transpose());
        let attn = scores.mul_scalar(1.0 / (d_head as f64).sqrt()).softmax_last_dim();
        let out = attn.matmul(&v_h);
        assert_equivalent_f32(&out);
    }

    #[test]
    fn recip_forward_returns_inverse() {
        // recip(2.0) == 0.5, recip(4.0) == 0.25 — IEEE-correct 1/x.
        let a = Tensor::from_f32(vec![2.0_f32, 4.0, 8.0], Shape::from_dims(&[3]), cpu_dev());
        let r = a.recip();
        let out = realize_f32(&r);
        let s = out.as_slice();
        assert!((s[0] - 0.5).abs()  < 1e-7, "recip(2.0)  = {}", s[0]);
        assert!((s[1] - 0.25).abs() < 1e-7, "recip(4.0)  = {}", s[1]);
        assert!((s[2] - 0.125).abs() < 1e-7, "recip(8.0) = {}", s[2]);
        // Cross-backend bit-for-bit (cpu_fast and reference both run 1.0/x).
        assert_equivalent_f32(&r);
    }

    #[test]
    fn abs_forward_returns_magnitude() {
        // abs(-3.0) == 3.0, abs(0.0) == 0.0, abs(3.0) == 3.0.
        let a = Tensor::from_f32(
            vec![-3.0_f32, 0.0, 3.0, -1.5],
            Shape::from_dims(&[4]),
            cpu_dev(),
        );
        let b = a.abs();
        let out = realize_f32(&b);
        let s = out.as_slice();
        assert_eq!(s, &[3.0, 0.0, 3.0, 1.5]);
        assert_equivalent_f32(&b);
    }

    #[test]
    fn recip_backward_matches_minus_one_over_x_squared() {
        // y = 1/x. dy/dx = -1/x². At x = 2.0, gradient = -0.25.
        // At x = 4.0, gradient = -1/16 = -0.0625.
        let a = Tensor::from_f32(vec![2.0_f32, 4.0], Shape::from_dims(&[2]), cpu_dev());
        let y = a.recip();
        let grads = y.backward();
        let g_a = grads.get(&a).unwrap();
        let out = realize_f32(&g_a);
        let s = out.as_slice();
        assert!((s[0] - (-0.25)).abs()   < 1e-6, "grad at x=2 = {}", s[0]);
        assert!((s[1] - (-0.0625)).abs() < 1e-6, "grad at x=4 = {}", s[1]);
    }

    #[test]
    fn abs_backward_matches_sign() {
        // y = |x|. dy/dx = sign(x), with sign(0) = 0 by convention.
        // x = -2 → -1, x = 2 → +1, x = 0 → 0.
        let a = Tensor::from_f32(
            vec![-2.0_f32, 2.0, 0.0],
            Shape::from_dims(&[3]),
            cpu_dev(),
        );
        let y = a.abs();
        let grads = y.backward();
        let g_a = grads.get(&a).unwrap();
        let out = realize_f32(&g_a);
        let s = out.as_slice();
        assert_eq!(s, &[-1.0, 1.0, 0.0],
            "Abs backward: sign(-2)=-1, sign(2)=+1, sign(0)=0; got {s:?}");
    }

    #[test]
    fn rem_forward_uses_pytorch_convention() {
        // PyTorch: result has sign of divisor (a - floor(a/b) * b).
        //   rem( 5,  3) =  2
        //   rem(-5,  3) =  1     (NOT -2 like C99 fmod)
        //   rem( 5, -3) = -1     (NOT 2 like rem_euclid)
        //   rem(-5, -3) = -2
        //   rem( 7.5, 2.5) = 0
        //   rem( 7.3, 2.0) = 1.3 (within float precision)
        let a = Tensor::from_f32(
            vec![5.0_f32, -5.0,  5.0, -5.0, 7.5, 7.3],
            Shape::from_dims(&[6]),
            cpu_dev(),
        );
        let b = a.const_f32_like(
            vec![3.0_f32,  3.0, -3.0, -3.0, 2.5, 2.0],
            Shape::from_dims(&[6]),
        );
        let r = a.rem(&b);
        let out = realize_f32(&r);
        let s = out.as_slice();
        let expected = [2.0_f32, 1.0, -1.0, -2.0, 0.0, 1.3];
        for (i, (&got, &want)) in s.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-5,
                "rem[{i}] = {got}, want {want}");
        }
        assert_equivalent_f32(&r);
    }

    #[test]
    fn rem_backward_da_is_identity_db_is_neg_floor_div() {
        // d/da = 1, d/db = -floor(a/b).
        // At (a=5, b=3):  grad_a = 1,    grad_b = -floor(5/3) = -1
        // At (a=-5, b=3): grad_a = 1,    grad_b = -floor(-5/3) = -(-2) = 2
        // At (a=7, b=4):  grad_a = 1,    grad_b = -floor(7/4) = -1
        let a = Tensor::from_f32(vec![5.0_f32, -5.0, 7.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![3.0_f32, 3.0, 4.0], Shape::from_dims(&[3]));
        let y = a.rem(&b);
        let grads = y.backward();
        let g_a = grads.get(&a).expect("gradient for a");
        let g_b = grads.get(&b).expect("gradient for b");
        let g_a_out = realize_f32(&g_a);
        let g_b_out = realize_f32(&g_b);
        assert_eq!(g_a_out.as_slice(), &[1.0_f32, 1.0, 1.0],
            "rem grad_a is identity");
        assert_eq!(g_b_out.as_slice(), &[-1.0_f32, 2.0, -1.0],
            "rem grad_b is -floor(a/b)");
    }

    #[test]
    fn rsqrt_forward_returns_one_over_sqrt() {
        // rsqrt(1)  = 1
        // rsqrt(4)  = 0.5
        // rsqrt(0.25) = 2
        // rsqrt(9)  ≈ 0.3333333
        // rsqrt(100) = 0.1
        let a = Tensor::from_f32(
            vec![1.0_f32, 4.0, 0.25, 9.0, 100.0],
            Shape::from_dims(&[5]),
            cpu_dev(),
        );
        let r = a.rsqrt();
        let out = realize_f32(&r);
        let s = out.as_slice();
        let expected = [1.0_f32, 0.5, 2.0, 0.333_333_34, 0.1];
        for (i, (&got, &want)) in s.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6,
                "rsqrt[{i}] = {got}, want {want}");
        }
        assert_equivalent_f32(&r);
    }

    #[test]
    fn rsqrt_backward_matches_minus_half_y_cubed() {
        // y = x^(-1/2). dy/dx = -0.5 * x^(-3/2) = -0.5 * y³.
        // At x=1:  y=1,    grad = -0.5 * 1   = -0.5
        // At x=4:  y=0.5,  grad = -0.5 * 0.125 = -0.0625
        // At x=0.25: y=2,  grad = -0.5 * 8   = -4.0
        let a = Tensor::from_f32(
            vec![1.0_f32, 4.0, 0.25],
            Shape::from_dims(&[3]),
            cpu_dev(),
        );
        let y = a.rsqrt();
        let grads = y.backward();
        let g_a = grads.get(&a).expect("gradient for a");
        let out = realize_f32(&g_a);
        let s = out.as_slice();
        let expected = [-0.5_f32, -0.0625, -4.0];
        for (i, (&got, &want)) in s.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6,
                "rsqrt'[{i}] = {got}, want {want}");
        }
    }

    #[test]
    fn pow_forward_matches_powf() {
        // pow(2, 3) = 8, pow(4, 0.5) = 2, pow(9, 0.5) = 3,
        // pow(2.5, 2) = 6.25, pow(1, anything) = 1.
        let a = Tensor::from_f32(
            vec![2.0_f32, 4.0, 9.0, 2.5, 1.0],
            Shape::from_dims(&[5]),
            cpu_dev(),
        );
        let b = a.const_f32_like(
            vec![3.0_f32, 0.5, 0.5, 2.0, 7.5],
            Shape::from_dims(&[5]),
        );
        let y = a.pow(&b);
        let out = realize_f32(&y);
        let s = out.as_slice();
        let expected = [8.0_f32, 2.0, 3.0, 6.25, 1.0];
        for (i, (&got, &want)) in s.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-5,
                "pow[{i}] = {got}, want {want}");
        }
        assert_equivalent_f32(&y);
    }

    #[test]
    fn pow_backward_matches_partials() {
        // y = pow(a, b).
        // dy/da = b * pow(a, b-1)
        // dy/db = pow(a, b) * ln(a) = y * ln(a)
        //
        // At a=2, b=3:  dy/da = 3 * 2^2 = 12;  dy/db = 8 * ln(2) ≈ 5.5452
        // At a=4, b=2:  dy/da = 2 * 4^1 = 8;   dy/db = 16 * ln(4) ≈ 22.181
        let a = Tensor::from_f32(vec![2.0_f32, 4.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f32_like(vec![3.0_f32, 2.0], Shape::from_dims(&[2]));
        let y = a.pow(&b);
        let grads = y.backward();
        let g_a = grads.get(&a).expect("gradient for a");
        let g_b = grads.get(&b).expect("gradient for b");
        let g_a_out = realize_f32(&g_a);
        let g_b_out = realize_f32(&g_b);
        let sa = g_a_out.as_slice();
        let sb = g_b_out.as_slice();
        let expected_a = [12.0_f32, 8.0];
        let expected_b = [5.545_177_4_f32, 22.180_71];
        for (i, (&got, &want)) in sa.iter().zip(expected_a.iter()).enumerate() {
            assert!((got - want).abs() < 1e-4,
                "pow grad_a[{i}] = {got}, want {want}");
        }
        for (i, (&got, &want)) in sb.iter().zip(expected_b.iter()).enumerate() {
            assert!((got - want).abs() < 1e-4,
                "pow grad_b[{i}] = {got}, want {want}");
        }
    }

    #[test]
    fn squeeze_round_trip_preserves_data() {
        // squeeze(x, dim) is metadata-only; bytes unchanged. Verify by
        // building x → squeeze(1) → unsqueeze(1) and confirming the
        // round-trip output matches x exactly.
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let a = Tensor::from_f32(
            data.clone(),
            Shape::from_dims(&[2, 1, 3]),
            cpu_dev(),
        );
        let squeezed = a.squeeze(1);
        assert_eq!(squeezed.shape().dims(), &[2, 3]);
        let restored = squeezed.unsqueeze(1);
        let out = realize_f32(&restored);
        assert_eq!(out.shape().dims(), &[2, 1, 3]);
        assert_eq!(out.as_slice(), data.as_slice());
        assert_equivalent_f32(&restored);
    }

    #[test]
    fn floor_forward_returns_round_down() {
        // floor(2.7) = 2, floor(-1.2) = -2, floor(0.0) = 0,
        // floor(-0.5) = -1 (round-half-to-floor by definition).
        let a = Tensor::from_f32(
            vec![2.7_f32, -1.2, 0.0, -0.5, 5.0],
            Shape::from_dims(&[5]),
            cpu_dev(),
        );
        let b = a.floor();
        let out = realize_f32(&b);
        let s = out.as_slice();
        assert_eq!(s, &[2.0, -2.0, 0.0, -1.0, 5.0]);
        assert_equivalent_f32(&b);
    }

    #[test]
    fn gelu_erf_forward_matches_known_values() {
        // gelu_erf(x) = 0.5 * x * (1 + erf(x/√2)).
        //   gelu_erf(0)  = 0
        //   gelu_erf(1)  = 0.5 * (1 + erf(1/√2)) ≈ 0.8413447461
        //   gelu_erf(-1) ≈ -0.1586552540
        //   gelu_erf(2)  = 1.0 * (1 + erf(√2))   ≈ 1.9544997361
        //   gelu_erf(0.5) ≈ 0.34573123
        let a = Tensor::from_f32(
            vec![0.0_f32, 1.0, -1.0, 2.0, 0.5],
            Shape::from_dims(&[5]),
            cpu_dev(),
        );
        let b = a.gelu_erf();
        let out = realize_f32(&b);
        let s = out.as_slice();
        let expected = [0.0_f32, 0.841_344_75, -0.158_655_25, 1.954_499_7, 0.345_731_23];
        for (i, (&got, &want)) in s.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6,
                "gelu_erf[{i}] = {got}, want {want}");
        }
        assert_equivalent_f32(&b);
    }

    #[test]
    fn gelu_erf_backward_matches_cdf_plus_x_pdf() {
        // d/dx gelu_erf(x) = Φ(x) + x * φ(x), where Φ is the standard
        // normal CDF and φ the PDF.
        //   x=0:  0.5 + 0          = 0.5
        //   x=1:  Φ(1) + φ(1)      ≈ 0.84134 + 0.24197 ≈ 1.08332
        //   x=-1: Φ(-1) - φ(-1)    ≈ 0.15866 - 0.24197 ≈ -0.08332
        let a = Tensor::from_f32(
            vec![0.0_f32, 1.0, -1.0],
            Shape::from_dims(&[3]),
            cpu_dev(),
        );
        let y = a.gelu_erf();
        let grads = y.backward();
        let g_a = grads.get(&a).expect("gradient for a");
        let out = realize_f32(&g_a);
        let s = out.as_slice();
        let expected = [0.5_f32, 1.083_315_47, -0.083_315_47];
        for (i, (&got, &want)) in s.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-5,
                "gelu_erf'[{i}] = {got}, want {want}");
        }
    }

    #[test]
    fn erf_forward_matches_known_values() {
        // Reference erf values (libm-correct):
        //   erf(0)   = 0
        //   erf(1)   ≈ 0.8427007929
        //   erf(-1)  ≈ -0.8427007929
        //   erf(2)   ≈ 0.9953222650
        //   erf(0.5) ≈ 0.5204998778
        let a = Tensor::from_f32(
            vec![0.0_f32, 1.0, -1.0, 2.0, 0.5],
            Shape::from_dims(&[5]),
            cpu_dev(),
        );
        let b = a.erf();
        let out = realize_f32(&b);
        let s = out.as_slice();
        let expected = [0.0_f32, 0.842_700_8, -0.842_700_8, 0.995_322_3, 0.520_499_88];
        for (i, (&got, &want)) in s.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6,
                "erf[{i}] = {got}, want {want}");
        }
        // Cross-backend bit-equal: both paths route through libm::erff.
        assert_equivalent_f32(&b);
    }

    #[test]
    fn erf_backward_matches_two_over_sqrt_pi_times_exp_neg_x_squared() {
        // d/dx erf(x) = (2/√π) * exp(-x²).
        // At x=0:  2/√π ≈ 1.1283791671
        // At x=1:  (2/√π) * e^-1 ≈ 0.4151074974
        // At x=-1: same as x=1 (even function in the derivative).
        let a = Tensor::from_f32(
            vec![0.0_f32, 1.0, -1.0],
            Shape::from_dims(&[3]),
            cpu_dev(),
        );
        let y = a.erf();
        let grads = y.backward();
        let g_a = grads.get(&a).expect("gradient for a");
        let out = realize_f32(&g_a);
        let s = out.as_slice();
        let expected = [1.128_379_2_f32, 0.415_107_5, 0.415_107_5];
        for (i, (&got, &want)) in s.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6,
                "erf'[{i}] = {got}, want {want}");
        }
    }

    #[test]
    fn sign_forward_returns_minus_one_zero_or_one() {
        // sign(-3.0) = -1, sign(0.0) = 0, sign(2.5) = 1, sign(-0.0) = 0,
        // sign(0.5) = 1.
        let a = Tensor::from_f32(
            vec![-3.0_f32, 0.0, 2.5, -0.0, 0.5, -1e-30],
            Shape::from_dims(&[6]),
            cpu_dev(),
        );
        let b = a.sign();
        let out = realize_f32(&b);
        let s = out.as_slice();
        // -0.0 compares equal to 0.0 with `<` and `>`, so sign(-0.0) = 0.
        assert_eq!(s, &[-1.0, 0.0, 1.0, 0.0, 1.0, -1.0]);
        assert_equivalent_f32(&b);
    }

    #[test]
    fn round_forward_uses_bankers_rounding_at_ties() {
        // Banker's rounding (round-half-to-even / IEEE 754 roundeven):
        //   0.5 → 0    (NOT 1, the C99-default)
        //   1.5 → 2
        //   2.5 → 2    (NOT 3 — round to even)
        //   3.5 → 4
        //  -0.5 → 0    (NOT -1 — even is 0)
        //  -1.5 → -2
        //  Non-tie cases match the obvious answer:
        //   0.4 → 0
        //   0.6 → 1
        //  -0.6 → -1
        let a = Tensor::from_f32(
            vec![0.5_f32, 1.5, 2.5, 3.5, -0.5, -1.5, 0.4, 0.6, -0.6],
            Shape::from_dims(&[9]),
            cpu_dev(),
        );
        let b = a.round();
        let out = realize_f32(&b);
        let s = out.as_slice();
        assert_eq!(s, &[0.0, 2.0, 2.0, 4.0, 0.0, -2.0, 0.0, 1.0, -1.0]);
        // Cross-backend bit-equal: both legacy and storage paths use
        // the same `round_ties_even` / manual roundeven impl.
        assert_equivalent_f32(&b);
    }

    #[test]
    fn ceil_forward_returns_round_up() {
        // ceil(2.3) = 3, ceil(-1.7) = -1, ceil(0.0) = 0,
        // ceil(0.5) = 1, ceil(5.0) = 5.
        let a = Tensor::from_f32(
            vec![2.3_f32, -1.7, 0.0, 0.5, 5.0],
            Shape::from_dims(&[5]),
            cpu_dev(),
        );
        let b = a.ceil();
        let out = realize_f32(&b);
        let s = out.as_slice();
        assert_eq!(s, &[3.0, -1.0, 0.0, 1.0, 5.0]);
        assert_equivalent_f32(&b);
    }

    #[test]
    fn deep_matmul_chain_doesnt_explode() {
        // Chain 20 small matmuls; verifies that the executor handles
        // deep dependency graphs without issues and produces the same
        // result as reference.
        let init = Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0], Shape::from_dims(&[2, 2]), cpu_dev());
        let mut current = init.clone();
        for i in 0..20 {
            let step_data = vec![1.0 + (i as f32) * 0.01, 0.0, 0.0, 1.0 - (i as f32) * 0.01];
            let step = init.const_f32_like(step_data, Shape::from_dims(&[2, 2]));
            current = current.matmul(&step);
        }
        assert_equivalent_f32(&current);
    }

    /// Realize-time panic augmentation: a `Log` on a U32 tensor panics
    /// inside `eval_node` (unary ops aren't implemented for integer
    /// tensors). Verify the re-raised panic message prepends graph-
    /// location context ("Node#N", op short name, and the input's
    /// shape + dtype) so the user can locate the failing site in
    /// production graphs.
    #[test]
    fn realize_panic_includes_graph_location() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        // Any tensor handle works as a "graph anchor" — we just need
        // the graph object to attach the U32 const to.
        let anchor = Tensor::from_f32(vec![0.0], Shape::from_dims(&[1]), cpu_dev());
        let idx = anchor.const_u32_like(vec![1_u32, 2, 3], Shape::from_dims(&[3]));
        let bad = idx.log();  // Op::Log, dtype=U32
        let result = catch_unwind(AssertUnwindSafe(|| realize_f32(&bad)));
        let err = result.expect_err("realize of Log(U32) should panic");
        let msg = if let Some(s) = err.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = err.downcast_ref::<&'static str>() {
            s.to_string()
        } else {
            panic!("unknown panic payload type")
        };
        assert!(
            msg.contains("fuel-graph-cpu realize: panic at Node#"),
            "expected graph-location prefix, got: {msg}"
        );
        assert!(msg.contains("Log"),
            "expected op short name 'Log' in message, got: {msg}");
        assert!(msg.contains("U32"),
            "expected input dtype 'U32' in message, got: {msg}");
    }
}

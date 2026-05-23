//! Reference executor: walks a [`fuel_graph::Graph`] and computes the
//! concrete value of any requested tensor using the textbook [`crate::ops`]
//! implementations.
//!
//! This is the bridge between the lazy graph layer and the correctness
//! oracle. Design:
//!
//! - A topological-order pass computes every reachable node exactly once
//!   per realize call. The topo sort is iterative (explicit-stack DFS) so
//!   deep graphs do not blow the Rust recursion limit.
//! - The cache holds [`AnyRefTensor`] values, a dtype-erased enum with one
//!   variant per supported float dtype. This lets a single graph contain
//!   nodes of different dtypes and lets `Cast` convert between them.
//! - Each op's dispatch is a small `match` on the cached tensor variants.
//!   Dtype mismatches at op inputs panic with a clear message.
//! - Per-dtype `realize_f32` / `realize_f64` / etc. wrappers are thin
//!   shims that call [`realize`] and unwrap the result to the requested
//!   dtype. The root tensor's dtype is not required to match — you can
//!   build a graph that ends in a `Cast` to whatever final dtype you want.
//!
//! The executor is NOT intended for production use. It is the textbook
//! answer every real backend will be validated against.

use crate::ops;
use crate::RefTensor;
use fuel_core_types::{DType, Shape};
use fuel_graph::{topo_order, topo_order_multi, NodeId, Op, Tensor};
use half::{bf16, f16};
use std::collections::HashMap;

/// A dtype-erased reference tensor. The executor caches one of these per
/// graph node as it walks the topo order.
///
/// Float variants hold arithmetic data; `U32` holds index data for
/// gather/scatter/index_select. The index variant never participates in
/// arithmetic dispatch — the indexing ops pull it out explicitly to
/// convert each element to `usize`.
#[derive(Debug, Clone)]
pub enum AnyRefTensor {
    F32(RefTensor<f32>),
    F64(RefTensor<f64>),
    BF16(RefTensor<bf16>),
    F16(RefTensor<f16>),
    U32(RefTensor<u32>),
}

impl AnyRefTensor {
    /// The dtype of the tensor this variant holds.
    pub fn dtype(&self) -> DType {
        match self {
            AnyRefTensor::F32(_) => DType::F32,
            AnyRefTensor::F64(_) => DType::F64,
            AnyRefTensor::BF16(_) => DType::BF16,
            AnyRefTensor::F16(_) => DType::F16,
            AnyRefTensor::U32(_) => DType::U32,
        }
    }

    /// Extract an `f32` tensor, or panic if the variant holds a different dtype.
    pub fn into_f32(self) -> RefTensor<f32> {
        match self {
            AnyRefTensor::F32(t) => t,
            other => panic!("AnyRefTensor::into_f32: got {:?}", other.dtype()),
        }
    }
    /// Extract an `f64` tensor, or panic if the variant holds a different dtype.
    pub fn into_f64(self) -> RefTensor<f64> {
        match self {
            AnyRefTensor::F64(t) => t,
            other => panic!("AnyRefTensor::into_f64: got {:?}", other.dtype()),
        }
    }
    /// Extract a `bf16` tensor, or panic if the variant holds a different dtype.
    pub fn into_bf16(self) -> RefTensor<bf16> {
        match self {
            AnyRefTensor::BF16(t) => t,
            other => panic!("AnyRefTensor::into_bf16: got {:?}", other.dtype()),
        }
    }
    /// Extract an `f16` tensor, or panic if the variant holds a different dtype.
    pub fn into_f16(self) -> RefTensor<f16> {
        match self {
            AnyRefTensor::F16(t) => t,
            other => panic!("AnyRefTensor::into_f16: got {:?}", other.dtype()),
        }
    }
    /// Extract a `u32` (index) tensor, or panic if the variant holds a different dtype.
    pub fn into_u32(self) -> RefTensor<u32> {
        match self {
            AnyRefTensor::U32(t) => t,
            other => panic!("AnyRefTensor::into_u32: got {:?}", other.dtype()),
        }
    }

    /// Borrow a `u32` (index) tensor, or return `None` if this variant
    /// holds a different dtype. Used by gather/index_select to pull the
    /// index operand without consuming the cache entry.
    pub fn as_u32(&self) -> Option<&RefTensor<u32>> {
        match self {
            AnyRefTensor::U32(t) => Some(t),
            _ => None,
        }
    }
}

/// Phase 7.5 G2: slot-first dispatch for the reference backend. If
/// the graph's storage_map has a populated slot for `id`, adopt its
/// bytes via host-buffer download and wrap as an `AnyRefTensor`.
/// Returns `None` when no slot exists; callers fall through to
/// `eval_node`.
fn try_adopt_slot_ref(
    graph: &fuel_graph::Graph,
    id: NodeId,
    shape: &Shape,
) -> Option<AnyRefTensor> {
    let slot_arc = graph.storage_for(id)?;
    let buf = {
        let slot = slot_arc.read().unwrap();
        slot.as_dyn().to_host_buffer_dyn().expect("slot D2H")
    };
    Some(host_buffer_to_any_ref(buf, shape))
}

fn host_buffer_to_any_ref(buf: fuel_core_types::HostBuffer, shape: &Shape) -> AnyRefTensor {
    match buf {
        fuel_core_types::HostBuffer::F32(v) => AnyRefTensor::F32(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::F64(v) => AnyRefTensor::F64(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::BF16(v) => AnyRefTensor::BF16(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::F16(v) => AnyRefTensor::F16(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::U32(v) => AnyRefTensor::U32(RefTensor::from_vec(v, shape.clone())),
        other => panic!(
            "fuel-reference-backend slot adopt: unsupported host-buffer dtype {:?}",
            other.dtype(),
        ),
    }
}

/// Compute the concrete value of `tensor` as an [`AnyRefTensor`] by
/// walking its graph. The returned variant's dtype matches the root
/// tensor's dtype.
pub fn realize(tensor: &Tensor) -> AnyRefTensor {
    let graph = tensor.graph().read().unwrap();
    let order = topo_order(&graph, tensor.id());
    let mut cache: HashMap<NodeId, AnyRefTensor> = HashMap::new();

    for id in order {
        let node = graph.node(id);
        // Phase 7.5 G2: slot-first dispatch.
        if let Some(adopted) = try_adopt_slot_ref(&graph, id, &node.shape) {
            cache.insert(id, adopted);
            continue;
        }
        let result = eval_node_with_graph_context(&graph, id, node, &cache);
        cache.insert(id, result);
    }

    cache
        .remove(&tensor.id())
        .expect("realize: target tensor missing from cache after topo walk")
}

/// Wrap per-node `eval_node_with_op` in `catch_unwind` so a downstream
/// panic (unsupported dtype combo, shape mismatch the builder didn't
/// catch, etc.) re-panics with a prepended graph-location identifier.
/// The augmented message tells you *which* node blew up and what its
/// immediate inputs looked like, so "realize panicked somewhere in
/// 4,000 ops" becomes "realize panicked at Node#1734 (Conv2D,
/// inputs=[Node#1733 Conv2D [1,64,32,32]f32, Node#12 Const [64,3,3,3]f32])".
fn eval_node_with_graph_context(
    graph: &fuel_graph::Graph,
    id: NodeId,
    node: &fuel_graph::Node,
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    use std::panic::{catch_unwind, AssertUnwindSafe, resume_unwind};
    let inputs = node.inputs.clone();
    let shape = node.shape.clone();
    let dtype = node.dtype;
    let op = node.op.clone();
    let result = catch_unwind(AssertUnwindSafe(|| {
        eval_node_with_op(&op, &inputs, &shape, dtype, cache)
    }));
    match result {
        Ok(t) => t,
        Err(payload) => {
            let original = panic_payload_to_string(&payload);
            let location = graph.describe_node(id);
            let msg = format!(
                "fuel-reference-backend realize: panic at {location}\n  original panic: {original}"
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

// ---- convenience wrappers --------------------------------------------------

/// Realize `tensor` and unwrap the result as an `f32` tensor.
pub fn realize_f32(tensor: &Tensor) -> RefTensor<f32> {
    realize(tensor).into_f32()
}

/// Realize `tensor` and unwrap the result as an `f64` tensor.
pub fn realize_f64(tensor: &Tensor) -> RefTensor<f64> {
    realize(tensor).into_f64()
}

/// Realize `tensor` and unwrap the result as a `bf16` tensor.
pub fn realize_bf16(tensor: &Tensor) -> RefTensor<bf16> {
    realize(tensor).into_bf16()
}

/// Realize `tensor` and unwrap the result as an `f16` tensor.
pub fn realize_f16(tensor: &Tensor) -> RefTensor<f16> {
    realize(tensor).into_f16()
}

/// Realize many tensors in a single walk of the combined graph. All
/// tensors must belong to the same graph. Returns one [`AnyRefTensor`]
/// per requested root, in the same order as `tensors`.
///
/// This is the primitive used by the KV-cache path: the forward graph
/// has one logits root plus 2*n_layers updated-K/V roots, and we want
/// to evaluate them in a single topo walk rather than n times.
pub fn realize_many(tensors: &[&Tensor]) -> Vec<AnyRefTensor> {
    if tensors.is_empty() {
        return Vec::new();
    }
    let graph_rc = tensors[0].graph();
    for t in &tensors[1..] {
        assert!(
            std::sync::Arc::ptr_eq(graph_rc, t.graph()),
            "realize_many: all tensors must belong to the same graph",
        );
    }
    let graph = graph_rc.read().unwrap();
    let roots: Vec<NodeId> = tensors.iter().map(|t| t.id()).collect();
    let order = topo_order_multi(&graph, &roots);
    let mut cache: HashMap<NodeId, AnyRefTensor> = HashMap::new();

    for id in order {
        let node = graph.node(id);
        // Phase 7.5 G2: slot-first dispatch.
        if let Some(adopted) = try_adopt_slot_ref(&graph, id, &node.shape) {
            cache.insert(id, adopted);
            continue;
        }
        let result = eval_node_with_graph_context(&graph, id, node, &cache);
        cache.insert(id, result);
    }

    roots
        .iter()
        .map(|id| {
            cache
                .get(id)
                .cloned()
                .expect("realize_many: root missing from cache after topo walk")
        })
        .collect()
}

/// Realize many tensors and unwrap every result as `f32`.
pub fn realize_many_f32(tensors: &[&Tensor]) -> Vec<RefTensor<f32>> {
    realize_many(tensors)
        .into_iter()
        .map(|t| t.into_f32())
        .collect()
}

// ---- per-op dispatch -------------------------------------------------------

// Dispatch each op to the correct monomorphization of the corresponding
// `ops::` function via a small set of macros. Rust can't express "a single
// function that satisfies `Fn(f32) + Fn(f64)`", so we fall back to a
// textual expansion per dtype. The macros make this one line per op in
// the main `eval_node` match.

macro_rules! unary {
    ($inputs:expr, $cache:expr, $func:path) => {{
        let x = $cache.get(&$inputs[0]).expect("topo order missing input");
        match x {
            AnyRefTensor::F32(t) => AnyRefTensor::F32($func(t)),
            AnyRefTensor::F64(t) => AnyRefTensor::F64($func(t)),
            AnyRefTensor::BF16(t) => AnyRefTensor::BF16($func(t)),
            AnyRefTensor::F16(t) => AnyRefTensor::F16($func(t)),
            AnyRefTensor::U32(_) => panic!(
                "{} is not supported on U32 (index) tensors",
                stringify!($func),
            ),
        }
    }};
}

macro_rules! unary_with_dim {
    ($inputs:expr, $cache:expr, $func:path, $dim:expr) => {{
        let x = $cache.get(&$inputs[0]).expect("topo order missing input");
        match x {
            AnyRefTensor::F32(t) => AnyRefTensor::F32($func(t, $dim)),
            AnyRefTensor::F64(t) => AnyRefTensor::F64($func(t, $dim)),
            AnyRefTensor::BF16(t) => AnyRefTensor::BF16($func(t, $dim)),
            AnyRefTensor::F16(t) => AnyRefTensor::F16($func(t, $dim)),
            AnyRefTensor::U32(_) => panic!(
                "{} is not supported on U32 (index) tensors",
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
            (AnyRefTensor::F32(a), AnyRefTensor::F32(b)) => AnyRefTensor::F32($func(a, b)),
            (AnyRefTensor::F64(a), AnyRefTensor::F64(b)) => AnyRefTensor::F64($func(a, b)),
            (AnyRefTensor::BF16(a), AnyRefTensor::BF16(b)) => AnyRefTensor::BF16($func(a, b)),
            (AnyRefTensor::F16(a), AnyRefTensor::F16(b)) => AnyRefTensor::F16($func(a, b)),
            (a, b) => panic!(
                "{}: unsupported operand dtypes (lhs={:?}, rhs={:?})",
                stringify!($func),
                a.dtype(),
                b.dtype(),
            ),
        }
    }};
}

/// Evaluate a single op given its cached inputs. Exposed publicly so
/// other executors (e.g. fuel-cuda-backend) can fall back to the reference
/// implementation for ops they don't handle natively.
pub fn eval_node_with_op(
    op: &Op,
    inputs: &[NodeId],
    shape: &Shape,
    _dtype: DType,
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    match op {
        // Op::Const is intercepted by slot-first dispatch
        // (try_adopt_slot_ref) in the realize loops above.
        Op::Const => unreachable!(
            "fuel-reference-backend eval_node: Op::Const must be handled \
             by slot-first dispatch in the realize loop, never reach \
             eval_node",
        ),

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
        Op::Flip { dim } => {
            let src = cache.get(&inputs[0]).expect("flip missing input");
            match src {
                AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::flip(t, *dim)),
                AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::flip(t, *dim)),
                AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::flip(t, *dim)),
                AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::flip(t, *dim)),
                AnyRefTensor::U32(t) => AnyRefTensor::U32(ops::flip(t, *dim)),
            }
        }
        Op::Roll { dim, shift } => {
            let src = cache.get(&inputs[0]).expect("roll missing input");
            match src {
                AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::roll(t, *dim, *shift)),
                AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::roll(t, *dim, *shift)),
                AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::roll(t, *dim, *shift)),
                AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::roll(t, *dim, *shift)),
                AnyRefTensor::U32(t) => AnyRefTensor::U32(ops::roll(t, *dim, *shift)),
            }
        }
        Op::CumSum { dim } => {
            let src = cache.get(&inputs[0]).expect("cumsum missing input");
            match src {
                AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::cumsum(t, *dim)),
                AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::cumsum(t, *dim)),
                AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::cumsum(t, *dim)),
                AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::cumsum(t, *dim)),
                AnyRefTensor::U32(_) => panic!("cumsum: not supported on U32 tensors"),
            }
        }
        Op::Triu { diagonal } => {
            let src = cache.get(&inputs[0]).expect("triu missing input");
            match src {
                AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::triu(t, *diagonal)),
                AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::triu(t, *diagonal)),
                AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::triu(t, *diagonal)),
                AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::triu(t, *diagonal)),
                AnyRefTensor::U32(t) => AnyRefTensor::U32(ops::triu(t, *diagonal)),
            }
        }
        Op::Tril { diagonal } => {
            let src = cache.get(&inputs[0]).expect("tril missing input");
            match src {
                AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::tril(t, *diagonal)),
                AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::tril(t, *diagonal)),
                AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::tril(t, *diagonal)),
                AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::tril(t, *diagonal)),
                AnyRefTensor::U32(t) => AnyRefTensor::U32(ops::tril(t, *diagonal)),
            }
        }
        Op::LogSoftmaxLastDim => {
            let src = cache.get(&inputs[0]).expect("log_softmax_last_dim missing input");
            match src {
                AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::log_softmax_last_dim(t)),
                AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::log_softmax_last_dim(t)),
                AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::log_softmax_last_dim(t)),
                AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::log_softmax_last_dim(t)),
                AnyRefTensor::U32(_) => panic!("log_softmax_last_dim: not supported on U32 tensors"),
            }
        }
        Op::LogSoftmaxLastDimBackward => {
            let y = cache.get(&inputs[0]).expect("log_softmax_last_dim_backward: missing y");
            let g = cache.get(&inputs[1]).expect("log_softmax_last_dim_backward: missing grad");
            match (y, g) {
                (AnyRefTensor::F32(y), AnyRefTensor::F32(g)) => AnyRefTensor::F32(ops::log_softmax_last_dim_backward(y, g)),
                (AnyRefTensor::F64(y), AnyRefTensor::F64(g)) => AnyRefTensor::F64(ops::log_softmax_last_dim_backward(y, g)),
                (AnyRefTensor::BF16(y), AnyRefTensor::BF16(g)) => AnyRefTensor::BF16(ops::log_softmax_last_dim_backward(y, g)),
                (AnyRefTensor::F16(y), AnyRefTensor::F16(g)) => AnyRefTensor::F16(ops::log_softmax_last_dim_backward(y, g)),
                _ => panic!("log_softmax_last_dim_backward: dtype mismatch or unsupported dtype"),
            }
        }
        Op::MaskedFill { .. } => panic!(
            "Op::MaskedFill: legacy fuel-reference-backend executor doesn't \
             support U8-mask ops; use the storage-path \
             PipelinedExecutor instead",
        ),
        Op::Pad { padding, mode, value } => {
            let src = cache.get(&inputs[0]).expect("pad missing input");
            match (src, mode) {
                (AnyRefTensor::F32(t), fuel_graph::PadMode::Constant) => AnyRefTensor::F32(ops::pad_const(t, padding, *value)),
                (AnyRefTensor::F32(t), fuel_graph::PadMode::Reflect) => AnyRefTensor::F32(ops::pad_reflect(t, padding)),
                (AnyRefTensor::F32(t), fuel_graph::PadMode::Replicate) => AnyRefTensor::F32(ops::pad_replicate(t, padding)),
                (AnyRefTensor::F64(t), fuel_graph::PadMode::Constant) => AnyRefTensor::F64(ops::pad_const(t, padding, *value)),
                (AnyRefTensor::F64(t), fuel_graph::PadMode::Reflect) => AnyRefTensor::F64(ops::pad_reflect(t, padding)),
                (AnyRefTensor::F64(t), fuel_graph::PadMode::Replicate) => AnyRefTensor::F64(ops::pad_replicate(t, padding)),
                (AnyRefTensor::BF16(t), fuel_graph::PadMode::Constant) => AnyRefTensor::BF16(ops::pad_const(t, padding, *value)),
                (AnyRefTensor::BF16(t), fuel_graph::PadMode::Reflect) => AnyRefTensor::BF16(ops::pad_reflect(t, padding)),
                (AnyRefTensor::BF16(t), fuel_graph::PadMode::Replicate) => AnyRefTensor::BF16(ops::pad_replicate(t, padding)),
                (AnyRefTensor::F16(t), fuel_graph::PadMode::Constant) => AnyRefTensor::F16(ops::pad_const(t, padding, *value)),
                (AnyRefTensor::F16(t), fuel_graph::PadMode::Reflect) => AnyRefTensor::F16(ops::pad_reflect(t, padding)),
                (AnyRefTensor::F16(t), fuel_graph::PadMode::Replicate) => AnyRefTensor::F16(ops::pad_replicate(t, padding)),
                (AnyRefTensor::U32(_), _) => panic!("pad: not supported on U32 tensors"),
            }
        }
        Op::PadBackward { in_shape, padding, mode } => {
            let mode_tag: u8 = match mode {
                fuel_graph::PadMode::Constant => 0,
                fuel_graph::PadMode::Reflect => 1,
                fuel_graph::PadMode::Replicate => 2,
            };
            let in_dims = in_shape.dims().to_vec();
            let go = cache.get(&inputs[0]).expect("pad_backward missing grad_out");
            match go {
                AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::pad_backward(t, &in_dims, padding, mode_tag)),
                AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::pad_backward(t, &in_dims, padding, mode_tag)),
                AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::pad_backward(t, &in_dims, padding, mode_tag)),
                AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::pad_backward(t, &in_dims, padding, mode_tag)),
                AnyRefTensor::U32(_) => panic!("pad_backward: not supported on U32"),
            }
        }

        // --- comparison family (output dtype = U8) ---
        // Output dtype differs from inputs (always U8); AnyRefTensor
        // doesn't carry a U8 variant, so realize-via-reference-backend
        // can't represent the result. Comparison ops are validated via
        // the storage-path executor (`PipelinedExecutor`), which
        // natively handles U8 output through the binding-table key
        // `[T, T, U8]`.
        Op::Equal => panic!(
            "Op::Equal: legacy fuel-reference-backend executor doesn't \
             support U8-output ops; use the storage-path \
             PipelinedExecutor instead",
        ),
        Op::Ne => panic!(
            "Op::Ne: legacy fuel-reference-backend executor doesn't \
             support U8-output ops; use the storage-path \
             PipelinedExecutor instead",
        ),
        Op::Lt => panic!(
            "Op::Lt: legacy fuel-reference-backend executor doesn't \
             support U8-output ops; use the storage-path \
             PipelinedExecutor instead",
        ),
        Op::Le => panic!(
            "Op::Le: legacy fuel-reference-backend executor doesn't \
             support U8-output ops; use the storage-path \
             PipelinedExecutor instead",
        ),
        Op::Gt => panic!(
            "Op::Gt: legacy fuel-reference-backend executor doesn't \
             support U8-output ops; use the storage-path \
             PipelinedExecutor instead",
        ),
        Op::Ge => panic!(
            "Op::Ge: legacy fuel-reference-backend executor doesn't \
             support U8-output ops; use the storage-path \
             PipelinedExecutor instead",
        ),
        Op::Where => panic!(
            "Op::Where: legacy fuel-reference-backend executor doesn't \
             support ternary U8-cond ops; use the storage-path \
             PipelinedExecutor instead",
        ),

        // --- linear algebra ---
        Op::MatMul => eval_matmul(inputs, cache),
        Op::Transpose => unary!(inputs, cache, ops::transpose_last_two),
        Op::Permute(axes) => eval_permute(axes, inputs, cache),

        // --- 2-D convolution (registry-routed) ---
        Op::Fused(fid, params)
            if *fid == fuel_graph::registry::FusedOps::CONV2D =>
        {
            let (stride, padding, groups) = match params {
                fuel_graph::registry::FusedOpParams::Conv2D { stride, padding, groups } => {
                    (*stride, *padding, *groups)
                }
                _ => panic!(
                    "Op::Fused(CONV2D, _) expected \
                     FusedOpParams::Conv2D, got {params:?}",
                ),
            };
            eval_conv2d(stride, padding, groups, inputs, cache)
        }
        // Phase 7.6 step 5 (final): legacy `Op::ConvTranspose2D` arm
        // dropped with the variant.
        Op::Fused(fid, params)
            if *fid == fuel_graph::registry::FusedOps::CONV_TRANSPOSE2D =>
        {
            let (stride, padding, output_padding, dilation, groups) = match params {
                fuel_graph::registry::FusedOpParams::ConvTranspose2D {
                    stride, padding, output_padding, dilation, groups,
                } => (*stride, *padding, *output_padding, *dilation, *groups),
                _ => panic!(
                    "Op::Fused(CONV_TRANSPOSE2D, _) expected \
                     FusedOpParams::ConvTranspose2D, got {params:?}",
                ),
            };
            eval_conv_transpose2d(stride, padding, output_padding, dilation, groups, inputs, cache)
        }
        // Phase 7.6 step 5 (final): legacy `Op::FlashAttn` arm
        // dropped with the variant.
        Op::Fused(fid, params)
            if *fid == fuel_graph::registry::FusedOps::FLASH_ATTN =>
        {
            let (softmax_scale, causal, window_size_left, window_size_right, softcap) = match params {
                fuel_graph::registry::FusedOpParams::FlashAttn {
                    softmax_scale, causal, window_size_left, window_size_right, softcap,
                } => (*softmax_scale, *causal, *window_size_left, *window_size_right, *softcap),
                _ => panic!(
                    "Op::Fused(FLASH_ATTN, _) expected FusedOpParams::FlashAttn, got {params:?}",
                ),
            };
            eval_flash_attn(softmax_scale, causal, window_size_left, window_size_right, softcap, inputs, cache)
        }
        // Phase 7.6 step 5 (final): legacy `Op::PagedAttn` arm
        // dropped with the variant.
        Op::Fused(fid, params)
            if *fid == fuel_graph::registry::FusedOps::PAGED_ATTN =>
        {
            let (softmax_scale, block_size, softcap) = match params {
                fuel_graph::registry::FusedOpParams::PagedAttn {
                    softmax_scale, block_size, softcap,
                } => (*softmax_scale, *block_size, *softcap),
                _ => panic!(
                    "Op::Fused(PAGED_ATTN, _) expected FusedOpParams::PagedAttn, got {params:?}",
                ),
            };
            eval_paged_attn(softmax_scale, block_size, softcap, inputs, cache)
        }
        Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::FUSED_LINEAR => {
            eval_fused_linear(inputs, cache)
        }

        // --- dtype, shape, and broadcasting ---
        Op::Cast(target) => eval_cast(*target, inputs, cache),
        Op::BroadcastTo(target_shape) => eval_broadcast_to(target_shape, inputs, cache),
        Op::Reshape(target_shape) => eval_reshape(target_shape, inputs, cache),
        Op::Unsqueeze { dim } => eval_unsqueeze(*dim, inputs, cache),
        Op::Squeeze { dim } => eval_squeeze(*dim, inputs, cache),
        Op::ReduceSumTo(target_shape) => eval_reduce_sum_to(target_shape, inputs, cache),
        Op::ReduceMaxTo(target_shape) => eval_reduce_max_to(target_shape, inputs, cache),

        // --- reductions to scalar ---
        Op::SumAll => unary!(inputs, cache, ops::sum_all),
        Op::MaxAll => unary!(inputs, cache, ops::max_all),
        Op::MinAll => unary!(inputs, cache, ops::min_all),
        Op::MeanAll => unary!(inputs, cache, ops::mean_all),

        // --- reductions along one dim ---
        Op::SumDim(d) => unary_with_dim!(inputs, cache, ops::sum_dim, *d),
        Op::MaxDim(d) => unary_with_dim!(inputs, cache, ops::max_dim, *d),
        Op::MinDim(d) => unary_with_dim!(inputs, cache, ops::min_dim, *d),
        Op::MeanDim(d) => unary_with_dim!(inputs, cache, ops::mean_dim, *d),

        // --- integer-producing reductions ---
        Op::ArgMaxDim(d) => eval_argindex_dim(*d, inputs, cache, /*is_max=*/ true),
        Op::ArgMinDim(d) => eval_argindex_dim(*d, inputs, cache, /*is_max=*/ false),

        // --- compositions (registry-routed) ---
        // Phase 7.6 step 5 (2026-05-11): all fused-op dispatch now
        // flows through `Op::Fused(fid, params)` arms; the legacy
        // `Op::SoftmaxLastDim` / `Op::LayerNormLastDim` /
        // `Op::RmsNormLastDim` / `Op::Rope` / `Op::FusedLinear` /
        // `Op::Conv2D` and the four backward-helper arms were dropped
        // together with their `Op` variants.
        Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::SOFTMAX_LAST_DIM => {
            unary!(inputs, cache, ops::softmax_last_dim)
        }
        Op::Fused(fid, params)
            if *fid == fuel_graph::registry::FusedOps::RMS_NORM_LAST_DIM =>
        {
            let eps = match params {
                fuel_graph::registry::FusedOpParams::RmsNormLastDim { eps } => *eps,
                _ => panic!(
                    "Op::Fused(RMS_NORM_LAST_DIM, _) expected \
                     FusedOpParams::RmsNormLastDim, got {params:?}",
                ),
            };
            eval_rms_norm_last_dim(eps, inputs, cache)
        }
        Op::Fused(fid, params)
            if *fid == fuel_graph::registry::FusedOps::LAYER_NORM_LAST_DIM =>
        {
            let eps = match params {
                fuel_graph::registry::FusedOpParams::LayerNormLastDim { eps } => *eps,
                _ => panic!(
                    "Op::Fused(LAYER_NORM_LAST_DIM, _) expected \
                     FusedOpParams::LayerNormLastDim, got {params:?}",
                ),
            };
            eval_layer_norm_last_dim(eps, inputs, cache)
        }
        Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::ROPE => {
            eval_rope(inputs, cache)
        }
        // Phase 7.6 step 5 (final): legacy `Op::QMatMul` arm dropped
        // with the variant.
        Op::Fused(fid, params)
            if *fid == fuel_graph::registry::FusedOps::QMATMUL =>
        {
            let (quant_type, k, n) = match params {
                fuel_graph::registry::FusedOpParams::QMatMul { quant_type, k, n } => {
                    (*quant_type, *k, *n)
                }
                _ => panic!(
                    "Op::Fused(QMATMUL, _) expected FusedOpParams::QMatMul, got {params:?}",
                ),
            };
            eval_qmatmul(quant_type, k, n, inputs, cache)
        }
        // Phase 7.6 step 4 (backward-helper batch): registry-extended
        // backward helpers route to the same reference kernels.
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::SOFTMAX_LAST_DIM_BACKWARD =>
        {
            eval_softmax_last_dim_backward(inputs, cache)
        }
        Op::Fused(fid, params)
            if *fid == fuel_graph::registry::FusedOps::LAYER_NORM_LAST_DIM_BACKWARD =>
        {
            let eps = match params {
                fuel_graph::registry::FusedOpParams::LayerNormLastDimBackward { eps } => *eps,
                _ => panic!(
                    "Op::Fused(LAYER_NORM_LAST_DIM_BACKWARD, _) expected \
                     FusedOpParams::LayerNormLastDimBackward, got {params:?}",
                ),
            };
            eval_layer_norm_last_dim_backward(eps, inputs, cache)
        }
        Op::Fused(fid, params)
            if *fid == fuel_graph::registry::FusedOps::RMS_NORM_LAST_DIM_BACKWARD =>
        {
            let eps = match params {
                fuel_graph::registry::FusedOpParams::RmsNormLastDimBackward { eps } => *eps,
                _ => panic!(
                    "Op::Fused(RMS_NORM_LAST_DIM_BACKWARD, _) expected \
                     FusedOpParams::RmsNormLastDimBackward, got {params:?}",
                ),
            };
            eval_rms_norm_last_dim_backward(eps, inputs, cache)
        }
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::REDUCE_MAX_TO_BACKWARD =>
        {
            eval_reduce_max_to_backward(inputs, cache)
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
        Op::AddScalar(c) => {
            let x = cache.get(&inputs[0]).expect("topo order missing input");
            match x {
                AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::add_scalar(t, *c)),
                AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::add_scalar(t, *c)),
                AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::add_scalar(t, *c)),
                AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::add_scalar(t, *c)),
                AnyRefTensor::U32(_) => panic!("add_scalar: not supported on U32 tensors"),
            }
        }
        Op::MulScalar(c) => {
            let x = cache.get(&inputs[0]).expect("topo order missing input");
            match x {
                AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::mul_scalar(t, *c)),
                AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::mul_scalar(t, *c)),
                AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::mul_scalar(t, *c)),
                AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::mul_scalar(t, *c)),
                AnyRefTensor::U32(_) => panic!("mul_scalar: not supported on U32 tensors"),
            }
        }
        Op::PowI(n) => {
            let x = cache.get(&inputs[0]).expect("topo order missing input");
            match x {
                AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::powi(t, *n)),
                AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::powi(t, *n)),
                AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::powi(t, *n)),
                AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::powi(t, *n)),
                AnyRefTensor::U32(_) => panic!("powi: not supported on U32 tensors"),
            }
        }
        Op::Clamp { min, max } => {
            let x = cache.get(&inputs[0]).expect("topo order missing input");
            match x {
                AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::clamp(t, *min, *max)),
                AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::clamp(t, *min, *max)),
                AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::clamp(t, *min, *max)),
                AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::clamp(t, *min, *max)),
                AnyRefTensor::U32(_) => panic!("clamp: not supported on U32 tensors"),
            }
        }
        Op::Maximum => binary!(inputs, cache, ops::maximum),
        Op::Minimum => binary!(inputs, cache, ops::minimum),
        Op::Copy { .. } | Op::Move { .. } => {
            // Reference backend has no notion of device — everything
            // lives in host memory. Copy/Move is a pass-through; the
            // target field is validated by the caller (only CPU-
            // resident transfers should reach here). Move's destructive
            // semantics are enforced by the executor cache layer.
            let x = cache.get(&inputs[0]).expect("topo order missing copy/move input");
            x.clone()
        }
        Op::Release => {
            // Reference backend: Release is a no-op — produces a
            // zero-element F32 marker. The input's memory is freed
            // when the cache entry is dropped.
            AnyRefTensor::F32(RefTensor::from_arc(
                std::sync::Arc::<[f32]>::from(Vec::<f32>::new()),
                Shape::from_dims(&[0]),
            ))
        }
        Op::Fused(fid, _params) => {
            // Phase 7.6 step 3: per-id arms handle the migrated fused
            // ops (only SoftmaxLastDim today; step 4 adds the rest).
            // Any unmigrated id reaching here is a programming bug —
            // the lowering rule should have decomposed it before
            // execute, or its dispatch arm above should have caught it.
            let _ = shape;
            unreachable!(
                "fuel-reference-backend eval_node: Op::Fused id {:?} has \
                 no dispatch arm wired yet. Step 4 extends this match.",
                fid,
            );
        }
        Op::WriteSlice { .. } => {
            // Phase 7.6 step 9c E.3.2: in-place scatter writes back
            // KV-cache mutation through the pipelined executor path,
            // not the reference-backend eager path. The reference
            // backend has no concept of pre-allocated destination
            // buffers (every node materializes a fresh tensor), so
            // a faithful implementation here would be a no-op clone
            // — not useful for the reference-backend's purpose of
            // bit-stable reference outputs.
            //
            // If a test surfaces WriteSlice through this path, the
            // test is mis-targeted: WriteSlice belongs on the
            // PipelinedExecutor path with an `InferenceContext`.
            unreachable!(
                "fuel-reference-backend eval_node: Op::WriteSlice is \
                 a pipelined-executor-only op (KV cache writes); \
                 reference-backend tests should not invoke it.",
            );
        }
        Op::Alloc { .. } => {
            // Phase 3a of bridge-retirement (post-9c): zero-init
            // device allocation through the PipelinedExecutor's
            // `WorkItemKind::Alloc` arm. The reference backend has
            // no notion of devices; if a test surfaces Op::Alloc
            // through this path, the test is mis-targeted — Op::Alloc
            // belongs on the PipelinedExecutor path (used today by
            // `KvCache::with_capacity` for KV-buffer init).
            unreachable!(
                "fuel-reference-backend eval_node: Op::Alloc is a \
                 pipelined-executor-only op (graph-level alloc); \
                 reference-backend tests should not invoke it.",
            );
        }
        Op::ZeroFill => {
            // Phase 3a follow-up: explicit in-place zero-fill paired
            // with the uninit Op::Alloc. Pipelined-executor-only op.
            unreachable!(
                "fuel-reference-backend eval_node: Op::ZeroFill is a \
                 pipelined-executor-only op (in-place zero-fill); \
                 reference-backend tests should not invoke it.",
            );
        }
    }
}

fn eval_concat(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let a = cache.get(&inputs[0]).expect("topo order missing concat lhs");
    let b = cache.get(&inputs[1]).expect("topo order missing concat rhs");
    match (a, b) {
        (AnyRefTensor::F32(a), AnyRefTensor::F32(b)) => AnyRefTensor::F32(ops::concat(a, b, dim)),
        (AnyRefTensor::F64(a), AnyRefTensor::F64(b)) => AnyRefTensor::F64(ops::concat(a, b, dim)),
        (AnyRefTensor::BF16(a), AnyRefTensor::BF16(b)) => {
            AnyRefTensor::BF16(ops::concat(a, b, dim))
        }
        (AnyRefTensor::F16(a), AnyRefTensor::F16(b)) => AnyRefTensor::F16(ops::concat(a, b, dim)),
        (AnyRefTensor::U32(a), AnyRefTensor::U32(b)) => AnyRefTensor::U32(ops::concat(a, b, dim)),
        (a, b) => panic!("concat: dtype mismatch {:?} vs {:?}", a.dtype(), b.dtype()),
    }
}

fn eval_slice(
    dim: usize,
    start: usize,
    len: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let x = cache.get(&inputs[0]).expect("topo order missing slice input");
    match x {
        AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::slice(t, dim, start, len)),
        AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::slice(t, dim, start, len)),
        AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::slice(t, dim, start, len)),
        AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::slice(t, dim, start, len)),
        AnyRefTensor::U32(t) => AnyRefTensor::U32(ops::slice(t, dim, start, len)),
    }
}

// ---- cast, broadcast, and layer_norm need their own dispatch -------------

fn eval_cast(
    target: DType,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let src = cache.get(&inputs[0]).expect("topo order missing cast input");
    match (src, target) {
        // Identity casts.
        (AnyRefTensor::F32(t), DType::F32) => AnyRefTensor::F32(t.clone()),
        (AnyRefTensor::F64(t), DType::F64) => AnyRefTensor::F64(t.clone()),
        (AnyRefTensor::BF16(t), DType::BF16) => AnyRefTensor::BF16(t.clone()),
        (AnyRefTensor::F16(t), DType::F16) => AnyRefTensor::F16(t.clone()),

        // From f32.
        (AnyRefTensor::F32(t), DType::F64) => {
            AnyRefTensor::F64(ops::cast_f32_to_f64(t))
        }
        (AnyRefTensor::F32(t), DType::BF16) => {
            AnyRefTensor::BF16(ops::cast_f32_to_bf16(t))
        }
        (AnyRefTensor::F32(t), DType::F16) => {
            AnyRefTensor::F16(ops::cast_f32_to_f16(t))
        }

        // From f64.
        (AnyRefTensor::F64(t), DType::F32) => {
            AnyRefTensor::F32(ops::cast_f64_to_f32(t))
        }
        (AnyRefTensor::F64(t), DType::BF16) => {
            AnyRefTensor::BF16(ops::cast_f64_to_bf16(t))
        }
        (AnyRefTensor::F64(t), DType::F16) => {
            AnyRefTensor::F16(ops::cast_f64_to_f16(t))
        }

        // From bf16.
        (AnyRefTensor::BF16(t), DType::F32) => {
            AnyRefTensor::F32(ops::cast_bf16_to_f32(t))
        }
        (AnyRefTensor::BF16(t), DType::F64) => {
            AnyRefTensor::F64(ops::cast_bf16_to_f64(t))
        }
        (AnyRefTensor::BF16(t), DType::F16) => {
            AnyRefTensor::F16(ops::cast_bf16_to_f16(t))
        }

        // From f16.
        (AnyRefTensor::F16(t), DType::F32) => {
            AnyRefTensor::F32(ops::cast_f16_to_f32(t))
        }
        (AnyRefTensor::F16(t), DType::F64) => {
            AnyRefTensor::F64(ops::cast_f16_to_f64(t))
        }
        (AnyRefTensor::F16(t), DType::BF16) => {
            AnyRefTensor::BF16(ops::cast_f16_to_bf16(t))
        }

        // U32 ↔ float conversions.
        (AnyRefTensor::U32(t), DType::F32) => AnyRefTensor::F32(ops::cast_u32_to_f32(t)),
        (AnyRefTensor::U32(t), DType::F64) => AnyRefTensor::F64(ops::cast_u32_to_f64(t)),
        (AnyRefTensor::U32(t), DType::U32) => AnyRefTensor::U32(t.clone()),
        (AnyRefTensor::F32(t), DType::U32) => AnyRefTensor::U32(ops::cast_f32_to_u32(t)),
        (AnyRefTensor::F64(t), DType::U32) => AnyRefTensor::U32(ops::cast_f64_to_u32(t)),

        (src, dst) => panic!(
            "cast: unsupported dtype combination {:?} -> {dst:?}",
            src.dtype(),
        ),
    }
}

fn eval_broadcast_to(
    target: &Shape,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let src = cache.get(&inputs[0]).expect("topo order missing bcast input");
    match src {
        AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::broadcast_to(t, target)),
        AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::broadcast_to(t, target)),
        AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::broadcast_to(t, target)),
        AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::broadcast_to(t, target)),
        AnyRefTensor::U32(_) => panic!("broadcast_to: not supported on U32 (index) tensors"),
    }
}

fn eval_reshape(
    target: &Shape,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    // Reshape is shape-only and works uniformly for every variant,
    // including integer index tensors, so it has a U32 arm too.
    let src = cache.get(&inputs[0]).expect("topo order missing reshape input");
    match src {
        AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::reshape(t, target)),
        AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::reshape(t, target)),
        AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::reshape(t, target)),
        AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::reshape(t, target)),
        AnyRefTensor::U32(t) => AnyRefTensor::U32(ops::reshape(t, target)),
    }
}

fn eval_unsqueeze(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    // Unsqueeze is bytes-identical with reshape; only the metadata
    // shape differs. Insert a size-1 axis at `dim`, then dispatch
    // through ops::reshape for each dtype.
    let src = cache.get(&inputs[0]).expect("topo order missing unsqueeze input");
    let in_shape = match src {
        AnyRefTensor::F32(t) => t.shape().dims().to_vec(),
        AnyRefTensor::F64(t) => t.shape().dims().to_vec(),
        AnyRefTensor::BF16(t) => t.shape().dims().to_vec(),
        AnyRefTensor::F16(t) => t.shape().dims().to_vec(),
        AnyRefTensor::U32(t) => t.shape().dims().to_vec(),
    };
    let mut out_dims = in_shape;
    assert!(
        dim <= out_dims.len(),
        "unsqueeze: dim {dim} out of bounds for rank {}",
        out_dims.len(),
    );
    out_dims.insert(dim, 1);
    let target = Shape::from_dims(&out_dims);
    match src {
        AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::reshape(t, &target)),
        AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::reshape(t, &target)),
        AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::reshape(t, &target)),
        AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::reshape(t, &target)),
        AnyRefTensor::U32(t) => AnyRefTensor::U32(ops::reshape(t, &target)),
    }
}

/// Inverse of [`eval_unsqueeze`]: drop the size-1 dimension at `dim`
/// from the metadata shape. Bytes unchanged. The builder validates the
/// preconditions (`dim < rank`, `shape[dim] == 1`); the executor just
/// dispatches.
fn eval_squeeze(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let src = cache.get(&inputs[0]).expect("topo order missing squeeze input");
    let in_dims: Vec<usize> = match src {
        AnyRefTensor::F32(t) => t.shape().dims().to_vec(),
        AnyRefTensor::F64(t) => t.shape().dims().to_vec(),
        AnyRefTensor::BF16(t) => t.shape().dims().to_vec(),
        AnyRefTensor::F16(t) => t.shape().dims().to_vec(),
        AnyRefTensor::U32(t) => t.shape().dims().to_vec(),
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
    let target = Shape::from_dims(&out_dims);
    match src {
        AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::reshape(t, &target)),
        AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::reshape(t, &target)),
        AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::reshape(t, &target)),
        AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::reshape(t, &target)),
        AnyRefTensor::U32(t) => AnyRefTensor::U32(ops::reshape(t, &target)),
    }
}

fn eval_reduce_sum_to(
    target: &Shape,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let src = cache.get(&inputs[0]).expect("topo order missing reduce_sum_to input");
    match src {
        AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::reduce_sum_to(t, target)),
        AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::reduce_sum_to(t, target)),
        AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::reduce_sum_to(t, target)),
        AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::reduce_sum_to(t, target)),
        AnyRefTensor::U32(_) => panic!("reduce_sum_to: not supported on U32 (index) tensors"),
    }
}

fn eval_reduce_max_to(
    target: &Shape,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let src = cache.get(&inputs[0]).expect("topo order missing reduce_max_to input");
    match src {
        AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::reduce_max_to(t, target)),
        AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::reduce_max_to(t, target)),
        AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::reduce_max_to(t, target)),
        AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::reduce_max_to(t, target)),
        AnyRefTensor::U32(_) => panic!("reduce_max_to: not supported on U32 (index) tensors"),
    }
}

fn eval_layer_norm_last_dim(
    eps: f64,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let src = cache.get(&inputs[0]).expect("topo order missing ln input");
    match src {
        AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::layer_norm_last_dim(t, eps)),
        AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::layer_norm_last_dim(t, eps)),
        AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::layer_norm_last_dim(t, eps)),
        AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::layer_norm_last_dim(t, eps)),
        AnyRefTensor::U32(_) => panic!("layer_norm_last_dim: cannot apply to U32 tensor"),
    }
}

fn eval_rms_norm_last_dim(
    eps: f64,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let src = cache.get(&inputs[0]).expect("topo order missing rms input");
    match src {
        AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::rms_norm_last_dim(t, eps)),
        AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::rms_norm_last_dim(t, eps)),
        AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::rms_norm_last_dim(t, eps)),
        AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::rms_norm_last_dim(t, eps)),
        AnyRefTensor::U32(_) => panic!("rms_norm_last_dim: cannot apply to U32 tensor"),
    }
}

fn eval_rms_norm_last_dim_backward(
    eps: f64,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let x = cache.get(&inputs[0]).expect("rms_norm_bwd missing x");
    let g = cache.get(&inputs[1]).expect("rms_norm_bwd missing g");
    match (x, g) {
        (AnyRefTensor::F32(x), AnyRefTensor::F32(g)) => {
            AnyRefTensor::F32(ops::rms_norm_last_dim_backward(x, g, eps))
        }
        (AnyRefTensor::F64(x), AnyRefTensor::F64(g)) => {
            AnyRefTensor::F64(ops::rms_norm_last_dim_backward(x, g, eps))
        }
        (AnyRefTensor::BF16(x), AnyRefTensor::BF16(g)) => {
            AnyRefTensor::BF16(ops::rms_norm_last_dim_backward(x, g, eps))
        }
        (AnyRefTensor::F16(x), AnyRefTensor::F16(g)) => {
            AnyRefTensor::F16(ops::rms_norm_last_dim_backward(x, g, eps))
        }
        (a, b) => panic!(
            "rms_norm_last_dim_backward: dtype mismatch {:?} vs {:?}",
            a.dtype(),
            b.dtype(),
        ),
    }
}

fn eval_matmul(
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let a = cache.get(&inputs[0]).expect("matmul missing lhs");
    let b = cache.get(&inputs[1]).expect("matmul missing rhs");
    match (a, b) {
        (AnyRefTensor::F32(a), AnyRefTensor::F32(b)) => AnyRefTensor::F32(ops::matmul(a, b)),
        (AnyRefTensor::F64(a), AnyRefTensor::F64(b)) => AnyRefTensor::F64(ops::matmul(a, b)),
        (AnyRefTensor::BF16(a), AnyRefTensor::BF16(b)) => AnyRefTensor::BF16(ops::matmul(a, b)),
        (AnyRefTensor::F16(a), AnyRefTensor::F16(b)) => AnyRefTensor::F16(ops::matmul(a, b)),
        // Mixed-precision: activations f32 × weights bf16 → f32.
        // Upcast B to f32 (bf16→f32 is exact) and run the f32 matmul.
        (AnyRefTensor::F32(a), AnyRefTensor::BF16(b)) => {
            let b_f32 = RefTensor::from_vec(
                b.as_slice().iter().map(|x| x.to_f32()).collect(),
                b.shape().clone(),
            );
            AnyRefTensor::F32(ops::matmul(a, &b_f32))
        }
        (a, b) => panic!(
            "matmul: unsupported operand dtypes (lhs={:?}, rhs={:?})",
            a.dtype(), b.dtype()
        ),
    }
}

fn eval_conv2d(
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let x = cache.get(&inputs[0]).expect("conv2d: missing x");
    let w = cache.get(&inputs[1]).expect("conv2d: missing weight");
    let b = inputs.get(2).and_then(|id| cache.get(id));
    // Require x, weight, and bias to all be the same float dtype.
    match (x, w, b) {
        (AnyRefTensor::F32(x), AnyRefTensor::F32(w), Some(AnyRefTensor::F32(bias))) => {
            AnyRefTensor::F32(ops::conv2d(x, w, Some(bias), stride, padding, groups))
        }
        (AnyRefTensor::F32(x), AnyRefTensor::F32(w), None) => {
            AnyRefTensor::F32(ops::conv2d(x, w, None, stride, padding, groups))
        }
        (AnyRefTensor::F64(x), AnyRefTensor::F64(w), Some(AnyRefTensor::F64(bias))) => {
            AnyRefTensor::F64(ops::conv2d(x, w, Some(bias), stride, padding, groups))
        }
        (AnyRefTensor::F64(x), AnyRefTensor::F64(w), None) => {
            AnyRefTensor::F64(ops::conv2d(x, w, None, stride, padding, groups))
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
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let x = cache.get(&inputs[0]).expect("conv_transpose2d: missing x");
    let w = cache.get(&inputs[1]).expect("conv_transpose2d: missing weight");
    match (x, w) {
        (AnyRefTensor::F32(x), AnyRefTensor::F32(w)) => {
            AnyRefTensor::F32(ops::conv_transpose2d(x, w, stride, padding, output_padding, dilation, groups))
        }
        (AnyRefTensor::F64(x), AnyRefTensor::F64(w)) => {
            AnyRefTensor::F64(ops::conv_transpose2d(x, w, stride, padding, output_padding, dilation, groups))
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
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    use crate::attention::{attention_naive, AttentionParams};
    let q = cache.get(&inputs[0]).expect("flash_attn: missing q");
    let k = cache.get(&inputs[1]).expect("flash_attn: missing k");
    let v = cache.get(&inputs[2]).expect("flash_attn: missing v");
    let alibi = inputs.get(3).and_then(|id| cache.get(id));
    let p = AttentionParams {
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
    };
    match (q, k, v, alibi) {
        (AnyRefTensor::F32(q), AnyRefTensor::F32(k), AnyRefTensor::F32(v), Some(AnyRefTensor::F32(a))) => {
            AnyRefTensor::F32(attention_naive(q, k, v, Some(a), &p))
        }
        (AnyRefTensor::F32(q), AnyRefTensor::F32(k), AnyRefTensor::F32(v), None) => {
            AnyRefTensor::F32(attention_naive(q, k, v, None, &p))
        }
        (AnyRefTensor::F64(q), AnyRefTensor::F64(k), AnyRefTensor::F64(v), Some(AnyRefTensor::F64(a))) => {
            AnyRefTensor::F64(attention_naive(q, k, v, Some(a), &p))
        }
        (AnyRefTensor::F64(q), AnyRefTensor::F64(k), AnyRefTensor::F64(v), None) => {
            AnyRefTensor::F64(attention_naive(q, k, v, None, &p))
        }
        (qa, ka, va, alba) => panic!(
            "flash_attn: unsupported operand dtype combination q={:?} k={:?} v={:?} alibi={:?}",
            qa.dtype(), ka.dtype(), va.dtype(), alba.map(|t| t.dtype()),
        ),
    }
}

/// FusedLinear reference: `(a @ b) + bias` where bias broadcasts
/// along the trailing matmul-output axis. Reference impl runs
/// matmul + bias-add as two passes; backends with a fused kernel
/// override the GraphBackend trait method to do it in one launch.
fn eval_fused_linear(
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let a = cache.get(&inputs[0]).expect("fused_linear: missing a");
    let b = cache.get(&inputs[1]).expect("fused_linear: missing b");
    let bias = cache.get(&inputs[2]).expect("fused_linear: missing bias");
    // Step 1: matmul.
    let mm = match (a, b) {
        (AnyRefTensor::F32(a), AnyRefTensor::F32(b)) => AnyRefTensor::F32(ops::matmul(a, b)),
        (AnyRefTensor::F64(a), AnyRefTensor::F64(b)) => AnyRefTensor::F64(ops::matmul(a, b)),
        _ => panic!("fused_linear: unsupported matmul dtype combination a={:?} b={:?}", a.dtype(), b.dtype()),
    };
    // Step 2: broadcast-add bias along the last axis. Bias must be rank-1
    // with length equal to the matmul output's last dim.
    match (&mm, bias) {
        (AnyRefTensor::F32(mm_t), AnyRefTensor::F32(bt)) => {
            let bias_b = ops::broadcast_to(bt, mm_t.shape());
            AnyRefTensor::F32(ops::add(mm_t, &bias_b))
        }
        (AnyRefTensor::F64(mm_t), AnyRefTensor::F64(bt)) => {
            let bias_b = ops::broadcast_to(bt, mm_t.shape());
            AnyRefTensor::F64(ops::add(mm_t, &bias_b))
        }
        (mm_a, b_a) => panic!(
            "fused_linear: bias dtype {:?} must match matmul dtype {:?}",
            b_a.dtype(), mm_a.dtype(),
        ),
    }
}

fn eval_paged_attn(
    softmax_scale: f32,
    block_size:    usize,
    softcap:       Option<f32>,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    use crate::attention::attention_paged_naive;
    let q  = cache.get(&inputs[0]).expect("paged_attn: missing q");
    let kc = cache.get(&inputs[1]).expect("paged_attn: missing k_cache");
    let vc = cache.get(&inputs[2]).expect("paged_attn: missing v_cache");
    let bt = cache.get(&inputs[3]).expect("paged_attn: missing block_table");
    let cl = cache.get(&inputs[4]).expect("paged_attn: missing context_lens");
    let alibi = inputs.get(5).and_then(|id| cache.get(id));
    let block_table = match bt {
        AnyRefTensor::U32(t) => t,
        other => panic!("paged_attn: block_table must be U32, got {:?}", other.dtype()),
    };
    let context_lens = match cl {
        AnyRefTensor::U32(t) => t,
        other => panic!("paged_attn: context_lens must be U32, got {:?}", other.dtype()),
    };
    match (q, kc, vc, alibi) {
        (AnyRefTensor::F32(q), AnyRefTensor::F32(kc), AnyRefTensor::F32(vc), Some(AnyRefTensor::F32(a))) => {
            AnyRefTensor::F32(attention_paged_naive(q, kc, vc, block_table, context_lens, Some(a), softmax_scale, block_size, softcap))
        }
        (AnyRefTensor::F32(q), AnyRefTensor::F32(kc), AnyRefTensor::F32(vc), None) => {
            AnyRefTensor::F32(attention_paged_naive(q, kc, vc, block_table, context_lens, None, softmax_scale, block_size, softcap))
        }
        (AnyRefTensor::F64(q), AnyRefTensor::F64(kc), AnyRefTensor::F64(vc), Some(AnyRefTensor::F64(a))) => {
            AnyRefTensor::F64(attention_paged_naive(q, kc, vc, block_table, context_lens, Some(a), softmax_scale, block_size, softcap))
        }
        (AnyRefTensor::F64(q), AnyRefTensor::F64(kc), AnyRefTensor::F64(vc), None) => {
            AnyRefTensor::F64(attention_paged_naive(q, kc, vc, block_table, context_lens, None, softmax_scale, block_size, softcap))
        }
        (qa, kca, vca, alba) => panic!(
            "paged_attn: unsupported operand dtype combination q={:?} k={:?} v={:?} alibi={:?}",
            qa.dtype(), kca.dtype(), vca.dtype(), alba.map(|t| t.dtype()),
        ),
    }
}

fn eval_rope(
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let x = cache.get(&inputs[0]).expect("topo order missing rope x input");
    let cos = cache.get(&inputs[1]).expect("topo order missing rope cos input");
    let sin = cache.get(&inputs[2]).expect("topo order missing rope sin input");
    match (x, cos, sin) {
        (AnyRefTensor::F32(x), AnyRefTensor::F32(c), AnyRefTensor::F32(s)) => {
            AnyRefTensor::F32(ops::rope(x, c, s))
        }
        (AnyRefTensor::F64(x), AnyRefTensor::F64(c), AnyRefTensor::F64(s)) => {
            AnyRefTensor::F64(ops::rope(x, c, s))
        }
        (AnyRefTensor::BF16(x), AnyRefTensor::BF16(c), AnyRefTensor::BF16(s)) => {
            AnyRefTensor::BF16(ops::rope(x, c, s))
        }
        (AnyRefTensor::F16(x), AnyRefTensor::F16(c), AnyRefTensor::F16(s)) => {
            AnyRefTensor::F16(ops::rope(x, c, s))
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
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let a = cache.get(&inputs[0]).expect("qmatmul missing activations");
    let w = cache.get(&inputs[1]).expect("qmatmul missing weight bytes");
    let a_f32 = match a {
        AnyRefTensor::F32(t) => t,
        _ => panic!("qmatmul: activations must be F32, got {:?}", a.dtype()),
    };
    let w_u32 = match w {
        AnyRefTensor::U32(t) => t,
        _ => panic!("qmatmul: weight bytes must be U32, got {:?}", w.dtype()),
    };
    // Reinterpret U32 as raw bytes for block decoding.
    let w_u32_slice = w_u32.as_slice();
    let w_bytes: Vec<u8> = w_u32_slice.iter().flat_map(|&u| u.to_le_bytes()).collect();

    // Dequantize W: [N, K] F32 row-major (same as Vulkan's dequant_q4_0 output).
    let w_deq = dequantize_blocks(&w_bytes, quant_type, n, k);
    let w_ref = crate::RefTensor::from_vec(w_deq, crate::Shape::from_dims(&[n, k]));

    // HF weight convention is [N, K] (out × in); our matmul wants [K, N].
    // Transpose W_ref to [K, N].
    let w_t = ops::transpose_last_two(&w_ref);

    // Matmul: A @ W^T → [..., M, N].
    AnyRefTensor::F32(ops::matmul(a_f32, &w_t))
}

/// Reference CPU dequantization of Q-type blocks to F32 row-major
/// weight matrix of shape `[n_rows, k_cols]`. This must bit-match the
/// GPU `dequant_q4_0` / `dequant_q8_0` kernels' output.
fn dequantize_blocks(
    bytes: &[u8],
    quant_type: fuel_graph::QuantType,
    n_rows: usize,
    k_cols: usize,
) -> Vec<f32> {
    use half::f16;
    let bpb = quant_type.bytes_per_block();
    let epb = quant_type.elements_per_block();
    assert_eq!(k_cols % epb, 0, "dequantize_blocks: k_cols must be multiple of {epb}");
    let blocks_per_row = k_cols / epb;
    let expected_bytes = n_rows * blocks_per_row * bpb;
    assert_eq!(bytes.len(), expected_bytes, "dequantize_blocks: byte count mismatch");
    let mut out = vec![0.0_f32; n_rows * k_cols];
    for row in 0..n_rows {
        for bi in 0..blocks_per_row {
            let block_off = (row * blocks_per_row + bi) * bpb;
            let out_base = row * k_cols + bi * epb;
            match quant_type {
                fuel_graph::QuantType::Q4_0 => {
                    // Single f16 scale at bytes[0..2]; 16 packed u4
                    // pairs: low nibble → element k, high → k+16.
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
                    dequantize_q4_km_block(&bytes[block_off..block_off + 144], &mut out[out_base..out_base + 256]);
                }
                other => unimplemented!(
                    "fuel-reference-backend dequantize_blocks does not support {other:?} yet"
                ),
            }
        }
    }
    out
}

/// Dequantize one 144-byte Q4_K_M super-block to 256 f32 elements.
/// Mirrors llama.cpp k_quants.c reference and must bit-match the
/// Vulkan `dequant_q4_km` kernel.
fn dequantize_q4_km_block(bytes: &[u8], out: &mut [f32]) {
    use half::f16;
    debug_assert_eq!(bytes.len(), 144);
    debug_assert_eq!(out.len(), 256);
    let d    = f16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
    let dmin = f16::from_le_bytes([bytes[2], bytes[3]]).to_f32();
    let scales: [u8; 12] = bytes[4..16].try_into().unwrap();
    let qs = &bytes[16..144];

    // llama.cpp `get_scale_min_k4` packing: 6-bit scale + 6-bit min
    // per sub-block, 8 sub-blocks total.
    let get_scale_min_k4 = |j: usize| -> (u8, u8) {
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
        let (sc, m) = get_scale_min_k4(is);
        let d1 = d * sc as f32;
        let m1 = dmin * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1);
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

fn eval_argindex_dim(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
    is_max: bool,
) -> AnyRefTensor {
    let x = cache.get(&inputs[0]).expect("topo order missing input");
    let result = match x {
        AnyRefTensor::F32(t) => {
            if is_max {
                ops::argmax_dim(t, dim)
            } else {
                ops::argmin_dim(t, dim)
            }
        }
        AnyRefTensor::F64(t) => {
            if is_max {
                ops::argmax_dim(t, dim)
            } else {
                ops::argmin_dim(t, dim)
            }
        }
        AnyRefTensor::BF16(t) => {
            if is_max {
                ops::argmax_dim(t, dim)
            } else {
                ops::argmin_dim(t, dim)
            }
        }
        AnyRefTensor::F16(t) => {
            if is_max {
                ops::argmax_dim(t, dim)
            } else {
                ops::argmin_dim(t, dim)
            }
        }
        AnyRefTensor::U32(_) => panic!("argmax/argmin not supported on U32 input tensors"),
    };
    AnyRefTensor::U32(result)
}

fn eval_softmax_last_dim_backward(
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let y = cache.get(&inputs[0]).expect("topo order missing y input");
    let g = cache.get(&inputs[1]).expect("topo order missing g input");
    match (y, g) {
        (AnyRefTensor::F32(y), AnyRefTensor::F32(g)) => {
            AnyRefTensor::F32(ops::softmax_last_dim_backward(y, g))
        }
        (AnyRefTensor::F64(y), AnyRefTensor::F64(g)) => {
            AnyRefTensor::F64(ops::softmax_last_dim_backward(y, g))
        }
        (AnyRefTensor::BF16(y), AnyRefTensor::BF16(g)) => {
            AnyRefTensor::BF16(ops::softmax_last_dim_backward(y, g))
        }
        (AnyRefTensor::F16(y), AnyRefTensor::F16(g)) => {
            AnyRefTensor::F16(ops::softmax_last_dim_backward(y, g))
        }
        (a, b) => panic!(
            "softmax_last_dim_backward: dtype mismatch {:?} vs {:?}",
            a.dtype(),
            b.dtype(),
        ),
    }
}

fn eval_reduce_max_to_backward(
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    // Inputs: x (original forward input, shape S_in) + upstream
    // (gradient w.r.t. y, shape S_target). The forward `target` is
    // upstream's shape — recover it directly from the second input.
    let x = cache.get(&inputs[0]).expect("topo order missing x input");
    let up = cache.get(&inputs[1]).expect("topo order missing upstream");
    let target = match up {
        AnyRefTensor::F32(t) => t.shape().clone(),
        AnyRefTensor::F64(t) => t.shape().clone(),
        AnyRefTensor::BF16(t) => t.shape().clone(),
        AnyRefTensor::F16(t) => t.shape().clone(),
        AnyRefTensor::U32(_) => panic!(
            "reduce_max_to_backward: upstream cannot be U32 (gradient must be float)"
        ),
    };
    match (x, up) {
        (AnyRefTensor::F32(x), AnyRefTensor::F32(up)) => {
            AnyRefTensor::F32(ops::reduce_max_to_backward(x, up, &target))
        }
        (AnyRefTensor::F64(x), AnyRefTensor::F64(up)) => {
            AnyRefTensor::F64(ops::reduce_max_to_backward(x, up, &target))
        }
        (AnyRefTensor::BF16(x), AnyRefTensor::BF16(up)) => {
            AnyRefTensor::BF16(ops::reduce_max_to_backward(x, up, &target))
        }
        (AnyRefTensor::F16(x), AnyRefTensor::F16(up)) => {
            AnyRefTensor::F16(ops::reduce_max_to_backward(x, up, &target))
        }
        (a, b) => panic!(
            "reduce_max_to_backward: dtype mismatch {:?} vs {:?}",
            a.dtype(),
            b.dtype(),
        ),
    }
}

fn eval_layer_norm_last_dim_backward(
    eps: f64,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let x = cache.get(&inputs[0]).expect("topo order missing x input");
    let g = cache.get(&inputs[1]).expect("topo order missing g input");
    match (x, g) {
        (AnyRefTensor::F32(x), AnyRefTensor::F32(g)) => {
            AnyRefTensor::F32(ops::layer_norm_last_dim_backward(x, g, eps))
        }
        (AnyRefTensor::F64(x), AnyRefTensor::F64(g)) => {
            AnyRefTensor::F64(ops::layer_norm_last_dim_backward(x, g, eps))
        }
        (AnyRefTensor::BF16(x), AnyRefTensor::BF16(g)) => {
            AnyRefTensor::BF16(ops::layer_norm_last_dim_backward(x, g, eps))
        }
        (AnyRefTensor::F16(x), AnyRefTensor::F16(g)) => {
            AnyRefTensor::F16(ops::layer_norm_last_dim_backward(x, g, eps))
        }
        (a, b) => panic!(
            "layer_norm_last_dim_backward: dtype mismatch {:?} vs {:?}",
            a.dtype(),
            b.dtype(),
        ),
    }
}

fn eval_permute(
    axes: &[usize],
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let x = cache.get(&inputs[0]).expect("topo order missing permute input");
    match x {
        AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::permute(t, axes)),
        AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::permute(t, axes)),
        AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::permute(t, axes)),
        AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::permute(t, axes)),
        AnyRefTensor::U32(t) => AnyRefTensor::U32(ops::permute(t, axes)),
    }
}

fn eval_index_select(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let data = cache.get(&inputs[0]).expect("topo order missing data input");
    let idx = cache
        .get(&inputs[1])
        .expect("topo order missing index input")
        .as_u32()
        .expect("index_select: second input must be U32");
    match data {
        AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::index_select_tensor(t, dim, idx)),
        AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::index_select_tensor(t, dim, idx)),
        AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::index_select_tensor(t, dim, idx)),
        AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::index_select_tensor(t, dim, idx)),
        AnyRefTensor::U32(t) => AnyRefTensor::U32(ops::index_select_tensor(t, dim, idx)),
    }
}

fn eval_gather(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let data = cache.get(&inputs[0]).expect("topo order missing data input");
    let idx = cache
        .get(&inputs[1])
        .expect("topo order missing index input")
        .as_u32()
        .expect("gather: second input must be U32");
    match data {
        AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::gather(t, dim, idx)),
        AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::gather(t, dim, idx)),
        AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::gather(t, dim, idx)),
        AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::gather(t, dim, idx)),
        AnyRefTensor::U32(t) => AnyRefTensor::U32(ops::gather(t, dim, idx)),
    }
}

fn eval_index_add(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let base = cache.get(&inputs[0]).expect("topo order missing base input");
    let idx = cache
        .get(&inputs[1])
        .expect("topo order missing index input")
        .as_u32()
        .expect("index_add: second input must be U32");
    let src = cache.get(&inputs[2]).expect("topo order missing src input");
    match (base, src) {
        (AnyRefTensor::F32(b), AnyRefTensor::F32(s)) => {
            AnyRefTensor::F32(ops::index_add(b, dim, idx, s))
        }
        (AnyRefTensor::F64(b), AnyRefTensor::F64(s)) => {
            AnyRefTensor::F64(ops::index_add(b, dim, idx, s))
        }
        (AnyRefTensor::BF16(b), AnyRefTensor::BF16(s)) => {
            AnyRefTensor::BF16(ops::index_add(b, dim, idx, s))
        }
        (AnyRefTensor::F16(b), AnyRefTensor::F16(s)) => {
            AnyRefTensor::F16(ops::index_add(b, dim, idx, s))
        }
        (b, s) => panic!(
            "index_add: base and src dtype mismatch: {:?} vs {:?}",
            b.dtype(),
            s.dtype(),
        ),
    }
}

fn eval_scatter_add(
    dim: usize,
    inputs: &[NodeId],
    cache: &HashMap<NodeId, AnyRefTensor>,
) -> AnyRefTensor {
    let base = cache.get(&inputs[0]).expect("topo order missing base input");
    let idx = cache
        .get(&inputs[1])
        .expect("topo order missing index input")
        .as_u32()
        .expect("scatter_add: second input must be U32");
    let src = cache.get(&inputs[2]).expect("topo order missing src input");
    match (base, src) {
        (AnyRefTensor::F32(b), AnyRefTensor::F32(s)) => {
            AnyRefTensor::F32(ops::scatter_add(b, dim, idx, s))
        }
        (AnyRefTensor::F64(b), AnyRefTensor::F64(s)) => {
            AnyRefTensor::F64(ops::scatter_add(b, dim, idx, s))
        }
        (AnyRefTensor::BF16(b), AnyRefTensor::BF16(s)) => {
            AnyRefTensor::BF16(ops::scatter_add(b, dim, idx, s))
        }
        (AnyRefTensor::F16(b), AnyRefTensor::F16(s)) => {
            AnyRefTensor::F16(ops::scatter_add(b, dim, idx, s))
        }
        (b, s) => panic!(
            "scatter_add: base and src dtype mismatch: {:?} vs {:?}",
            b.dtype(),
            s.dtype(),
        ),
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

    fn approx_vec(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "at index {i}: expected {y}, got {x} (tol {tol})",
            );
        }
    }

    // ---- forward realization ----

    #[test]
    fn realize_single_const_f32() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let result = realize_f32(&a);
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn realize_add_then_mul_chain() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        // (a + b) * a = [5*1, 7*2, 9*3] = [5, 14, 27]
        let c = a.add(&b).mul(&a);
        let result = realize_f32(&c);
        assert_eq!(result.as_slice(), &[5.0, 14.0, 27.0]);
    }

    #[test]
    fn realize_conv2d_identity_3x3() {
        // 1×1×3×3 identity input, 1×1×1×1 kernel = 1.0 → should equal input.
        let x = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Shape::from_dims(&[1, 1, 3, 3]),
            cpu_dev(),
        );
        let w = x.const_f32_like(vec![1.0], Shape::from_dims(&[1, 1, 1, 1]));
        let y = x.conv2d(&w, None, (1, 1), (0, 0), 1);
        let result = realize_f32(&y);
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn realize_conv2d_k3_s1_p1_known() {
        // 1×1×3×3 input of ones; 1×1×3×3 kernel of ones with zero bias, stride 1, pad 1.
        // Each output position sums the 3×3 window (with zero-pad edges).
        // Interior (1,1) sees a full 3×3 of 1s → 9. Corners see 2×2 = 4. Edges see 2×3 = 6.
        let x = Tensor::from_f32(
            vec![1.0; 9],
            Shape::from_dims(&[1, 1, 3, 3]),
            cpu_dev(),
        );
        let w = x.const_f32_like(vec![1.0; 9], Shape::from_dims(&[1, 1, 3, 3]));
        let b = x.const_f32_like(vec![0.0_f32], Shape::from_dims(&[1]));
        let y = x.conv2d(&w, Some(&b), (1, 1), (1, 1), 1);
        let out = realize_f32(&y);
        assert_eq!(
            out.as_slice(),
            &[4.0, 6.0, 4.0, 6.0, 9.0, 6.0, 4.0, 6.0, 4.0],
        );
    }

    #[test]
    fn realize_conv2d_depthwise_matches_per_channel() {
        // Depthwise 3×3 conv. Two channels with independent per-channel
        // kernels; verify each output channel depends only on its own
        // input channel.
        let x = Tensor::from_f32(
            vec![
                1.0, 2.0, 3.0, 4.0,   // ch 0
                10.0, 20.0, 30.0, 40.0, // ch 1
            ],
            Shape::from_dims(&[1, 2, 2, 2]),
            cpu_dev(),
        );
        // Kernel [2, 1, 1, 1] — a single scalar per channel.
        let w = x.const_f32_like(vec![2.0, 0.5], Shape::from_dims(&[2, 1, 1, 1]));
        let y = x.conv2d(&w, None, (1, 1), (0, 0), 2);
        let out = realize_f32(&y);
        assert_eq!(
            out.as_slice(),
            &[
                2.0, 4.0, 6.0, 8.0,       // ch 0 × 2.0
                5.0, 10.0, 15.0, 20.0,    // ch 1 × 0.5
            ],
        );
    }

    #[test]
    fn realize_matmul_hand_computed() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from_dims(&[2, 3]), cpu_dev());
        let b = a.const_f32_like(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::from_dims(&[3, 2]),
        );
        let c = a.matmul(&b);
        let result = realize_f32(&c);
        assert_eq!(result.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    // ---- multi-dtype realization ----

    #[test]
    fn realize_f64_matmul() {
        let a = Tensor::from_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from_dims(&[2, 3]), cpu_dev());
        let b = a.const_f64_like(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::from_dims(&[3, 2]),
        );
        let c = a.matmul(&b);
        let result = realize_f64(&c);
        assert_eq!(result.as_slice(), &[58.0_f64, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn realize_bf16_add() {
        let a = Tensor::from_bf16(
            vec![bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0)],
            Shape::from_dims(&[3]),
            cpu_dev(),
        );
        let b = a.const_bf16_like(
            vec![bf16::from_f32(4.0), bf16::from_f32(5.0), bf16::from_f32(6.0)],
            Shape::from_dims(&[3]),
        );
        let c = a.add(&b);
        let result = realize_bf16(&c);
        assert_eq!(result.as_slice()[0], bf16::from_f32(5.0));
        assert_eq!(result.as_slice()[1], bf16::from_f32(7.0));
        assert_eq!(result.as_slice()[2], bf16::from_f32(9.0));
    }

    #[test]
    fn realize_f16_mul() {
        let a = Tensor::from_f16(
            vec![f16::from_f32(2.0), f16::from_f32(3.0)],
            Shape::from_dims(&[2]),
            cpu_dev(),
        );
        let b = a.const_f16_like(
            vec![f16::from_f32(4.0), f16::from_f32(5.0)],
            Shape::from_dims(&[2]),
        );
        let c = a.mul(&b);
        let result = realize_f16(&c);
        assert_eq!(result.as_slice()[0], f16::from_f32(8.0));
        assert_eq!(result.as_slice()[1], f16::from_f32(15.0));
    }

    // ---- cast ----

    #[test]
    fn cast_f32_to_f64_preserves_values() {
        let a = Tensor::from_f32(vec![1.0, 2.5, -3.25], Shape::from_dims(&[3]), cpu_dev());
        let b = a.cast(DType::F64);
        let result = realize_f64(&b);
        assert_eq!(result.as_slice(), &[1.0_f64, 2.5, -3.25]);
    }

    #[test]
    fn cast_f64_to_f32_roundtrip() {
        // f64 → f32 → f64 loses precision, but for exactly-representable
        // values it's a no-op.
        let a = Tensor::from_f64(vec![1.0, 2.5, -3.25], Shape::from_dims(&[3]), cpu_dev());
        let b = a.cast(DType::F32).cast(DType::F64);
        let result = realize_f64(&b);
        assert_eq!(result.as_slice(), &[1.0_f64, 2.5, -3.25]);
    }

    #[test]
    fn cast_chain_through_bf16() {
        // f32 → bf16 → f32. Round trip loses precision but small integers
        // are exactly representable.
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
        let b = a.cast(DType::BF16).cast(DType::F32);
        let result = realize_f32(&b);
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn mixed_precision_compute_f64_matmul_output_cast_to_f32() {
        // Build the matmul in f64 for precision, then cast the result to
        // f32 for output. This is the canonical "high-precision
        // accumulator, low-precision storage" mixed-precision pattern.
        let a = Tensor::from_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from_dims(&[2, 3]), cpu_dev());
        let b = a.const_f64_like(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::from_dims(&[3, 2]),
        );
        let c = a.matmul(&b).cast(DType::F32);
        assert_eq!(c.dtype(), DType::F32);
        let result = realize_f32(&c);
        assert_eq!(result.as_slice(), &[58.0_f32, 64.0, 139.0, 154.0]);
    }

    // ---- new unary ops ----

    #[test]
    fn realize_neg_and_sub() {
        let a = Tensor::from_f32(vec![10.0, 20.0, 30.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let c = a.sub(&b);
        assert_eq!(realize_f32(&c).as_slice(), &[9.0, 18.0, 27.0]);
        let d = a.neg();
        assert_eq!(realize_f32(&d).as_slice(), &[-10.0, -20.0, -30.0]);
    }

    #[test]
    fn realize_div() {
        let a = Tensor::from_f32(vec![10.0, 20.0, 30.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![2.0, 4.0, 5.0], Shape::from_dims(&[3]));
        let c = a.div(&b);
        assert_eq!(realize_f32(&c).as_slice(), &[5.0, 5.0, 6.0]);
    }

    #[test]
    fn realize_sqrt_log_sin_cos_tanh_sigmoid_step() {
        let x = Tensor::from_f32(vec![0.0, 1.0, 4.0], Shape::from_dims(&[3]), cpu_dev());
        // sqrt(0, 1, 4) = (0, 1, 2)
        assert_eq!(realize_f32(&x.sqrt()).as_slice(), &[0.0, 1.0, 2.0]);
        // log(e, e², 1) — build a fresh input
        let x2 = Tensor::from_f32(
            vec![std::f32::consts::E, std::f32::consts::E.powi(2), 1.0],
            Shape::from_dims(&[3]),
            cpu_dev(),
        );
        approx_vec(realize_f32(&x2.log()).as_slice(), &[1.0, 2.0, 0.0], 1e-5);
        // sin(0, pi/2, pi) ≈ (0, 1, 0)
        let x3 = Tensor::from_f32(
            vec![0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI],
            Shape::from_dims(&[3]),
            cpu_dev(),
        );
        approx_vec(realize_f32(&x3.sin()).as_slice(), &[0.0, 1.0, 0.0], 1e-5);
        approx_vec(realize_f32(&x3.cos()).as_slice(), &[1.0, 0.0, -1.0], 1e-5);
        // tanh(0) = 0, tanh(100) ~ 1
        let x4 = Tensor::from_f32(vec![-100.0, 0.0, 100.0], Shape::from_dims(&[3]), cpu_dev());
        let t = realize_f32(&x4.tanh());
        approx_vec(t.as_slice(), &[-1.0, 0.0, 1.0], 1e-5);
        // sigmoid(0) = 0.5, sigmoid(±∞) → (0, 1)
        let s = realize_f32(&x4.sigmoid());
        approx_vec(s.as_slice(), &[0.0, 0.5, 1.0], 1e-5);
        // step: [0, 0, 1]
        let x5 = Tensor::from_f32(vec![-1.0, 0.0, 2.0], Shape::from_dims(&[3]), cpu_dev());
        assert_eq!(realize_f32(&x5.step()).as_slice(), &[0.0, 0.0, 1.0]);
    }

    // ---- reductions and broadcast ----

    #[test]
    fn realize_sum_mean_max_min_all() {
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::from_dims(&[5]), cpu_dev());
        assert_eq!(realize_f32(&x.sum_all()).as_slice(), &[15.0]);
        assert_eq!(realize_f32(&x.mean_all()).as_slice(), &[3.0]);
        assert_eq!(realize_f32(&x.max_all()).as_slice(), &[5.0]);
        assert_eq!(realize_f32(&x.min_all()).as_slice(), &[1.0]);
    }

    #[test]
    fn realize_axis_reductions() {
        // [[1, 2, 3], [4, 5, 6]]
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from_dims(&[2, 3]), cpu_dev());
        assert_eq!(realize_f32(&x.sum_dim(1)).as_slice(), &[6.0, 15.0]);
        assert_eq!(realize_f32(&x.mean_dim(1)).as_slice(), &[2.0, 5.0]);
        assert_eq!(realize_f32(&x.max_dim(0)).as_slice(), &[4.0, 5.0, 6.0]);
        assert_eq!(realize_f32(&x.min_dim(0)).as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn realize_broadcast_to_matches_ops() {
        // Broadcast a scalar to [3]
        let a = Tensor::from_f32(vec![7.0], Shape::from_dims(&[]), cpu_dev());
        let b = a.broadcast_to(Shape::from_dims(&[3]));
        assert_eq!(realize_f32(&b).as_slice(), &[7.0, 7.0, 7.0]);
    }

    #[test]
    fn realize_softmax_last_dim_sums_to_one() {
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from_dims(&[2, 3]), cpu_dev());
        let s = realize_f32(&x.softmax_last_dim());
        for row in 0..2 {
            let total: f32 = s.as_slice()[row * 3..row * 3 + 3].iter().sum();
            assert!((total - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn realize_layer_norm_last_dim_zero_mean_unit_variance() {
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
        let y = realize_f32(&x.layer_norm_last_dim(1e-12));
        let slice = y.as_slice();
        let mean: f32 = slice.iter().sum::<f32>() / 4.0;
        let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-5);
    }

    // ---- backward numerical correctness ----

    #[test]
    fn backward_of_sub_realizes_to_ones_and_neg_ones() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        let c = a.sub(&b);
        let grads = c.backward();
        assert_eq!(realize_f32(&grads.get(&a).unwrap()).as_slice(), &[1.0, 1.0, 1.0]);
        assert_eq!(realize_f32(&grads.get(&b).unwrap()).as_slice(), &[-1.0, -1.0, -1.0]);
    }

    #[test]
    fn backward_of_div() {
        // y = a / b, so dy/da = 1/b and dy/db = -a/b².
        let a_vals = vec![2.0_f32, 8.0, 15.0];
        let b_vals = vec![1.0_f32, 4.0, 5.0];
        let a = Tensor::from_f32(a_vals.clone(), Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(b_vals.clone(), Shape::from_dims(&[3]));
        let y = a.div(&b);
        let grads = y.backward();
        let g_a = realize_f32(&grads.get(&a).unwrap());
        let g_b = realize_f32(&grads.get(&b).unwrap());
        let expected_a: Vec<f32> = b_vals.iter().map(|&v| 1.0 / v).collect();
        let expected_b: Vec<f32> = a_vals
            .iter()
            .zip(b_vals.iter())
            .map(|(&av, &bv)| -av / (bv * bv))
            .collect();
        approx_vec(g_a.as_slice(), &expected_a, 1e-5);
        approx_vec(g_b.as_slice(), &expected_b, 1e-5);
    }

    #[test]
    fn backward_of_sqrt_matches_analytic() {
        // y = sqrt(x), dy/dx = 1/(2*sqrt(x))
        let x_vals = vec![1.0_f32, 4.0, 9.0, 16.0];
        let x = Tensor::from_f32(x_vals.clone(), Shape::from_dims(&[4]), cpu_dev());
        let y = x.sqrt();
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        let expected: Vec<f32> = x_vals.iter().map(|&v| 1.0 / (2.0 * v.sqrt())).collect();
        approx_vec(g_x.as_slice(), &expected, 1e-5);
    }

    #[test]
    fn backward_of_log_is_reciprocal() {
        // y = ln(x), dy/dx = 1/x.
        let x_vals = vec![1.0_f32, 2.0, 4.0, 10.0];
        let x = Tensor::from_f32(x_vals.clone(), Shape::from_dims(&[4]), cpu_dev());
        let y = x.log();
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        let expected: Vec<f32> = x_vals.iter().map(|&v| 1.0 / v).collect();
        approx_vec(g_x.as_slice(), &expected, 1e-5);
    }

    #[test]
    fn backward_of_sin_is_cos() {
        let x_vals = vec![0.0_f32, 0.5, 1.0, 1.5];
        let x = Tensor::from_f32(x_vals.clone(), Shape::from_dims(&[4]), cpu_dev());
        let y = x.sin();
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        let expected: Vec<f32> = x_vals.iter().map(|v| v.cos()).collect();
        approx_vec(g_x.as_slice(), &expected, 1e-5);
    }

    #[test]
    fn backward_of_cos_is_neg_sin() {
        let x_vals = vec![0.0_f32, 0.5, 1.0, 1.5];
        let x = Tensor::from_f32(x_vals.clone(), Shape::from_dims(&[4]), cpu_dev());
        let y = x.cos();
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        let expected: Vec<f32> = x_vals.iter().map(|v| -v.sin()).collect();
        approx_vec(g_x.as_slice(), &expected, 1e-5);
    }

    #[test]
    fn backward_of_tanh_matches_one_minus_sq() {
        let x_vals = vec![-1.0_f32, 0.0, 0.5, 1.0];
        let x = Tensor::from_f32(x_vals.clone(), Shape::from_dims(&[4]), cpu_dev());
        let y = x.tanh();
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        let expected: Vec<f32> = x_vals
            .iter()
            .map(|v| {
                let t = v.tanh();
                1.0 - t * t
            })
            .collect();
        approx_vec(g_x.as_slice(), &expected, 1e-5);
    }

    #[test]
    fn backward_of_sigmoid_matches_y_one_minus_y() {
        let x_vals = vec![-2.0_f32, -0.5, 0.0, 1.0];
        let x = Tensor::from_f32(x_vals.clone(), Shape::from_dims(&[4]), cpu_dev());
        let y = x.sigmoid();
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let expected: Vec<f32> = x_vals
            .iter()
            .map(|&v| {
                let s = sig(v);
                s * (1.0 - s)
            })
            .collect();
        approx_vec(g_x.as_slice(), &expected, 1e-5);
    }

    #[test]
    fn backward_of_relu_via_step() {
        // dL/dx for relu is step(x): 1 where x > 0, 0 elsewhere.
        let x_vals = vec![-2.0_f32, -0.5, 0.0, 1.0, 3.0];
        let x = Tensor::from_f32(x_vals.clone(), Shape::from_dims(&[5]), cpu_dev());
        let y = x.relu();
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        assert_eq!(g_x.as_slice(), &[0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn backward_of_cast_roundtrips_dtype() {
        // y = cast(x_f32, f64), dy/dx = cast(upstream, f32).
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let y = x.cast(DType::F64);
        let grads = y.backward();
        let g_x_tensor = grads.get(&x).unwrap();
        // The gradient lives in f32 space (same as x).
        assert_eq!(g_x_tensor.dtype(), DType::F32);
        // The value is 1.0 everywhere (upstream was ones, cast back to f32).
        let g_x = realize_f32(&g_x_tensor);
        assert_eq!(g_x.as_slice(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn backward_of_sum_all_broadcasts_upstream() {
        // y = sum(x) (scalar), so dy/dx = broadcast(1.0, x.shape) = ones.
        let x = Tensor::from_f32(vec![2.0, 3.0, 5.0, 7.0], Shape::from_dims(&[4]), cpu_dev());
        let y = x.sum_all();
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        assert_eq!(g_x.as_slice(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn backward_of_mean_all_is_one_over_n() {
        // y = mean(x) (scalar), so dy/dx = 1/n everywhere.
        let x = Tensor::from_f32(vec![2.0, 3.0, 5.0, 7.0], Shape::from_dims(&[4]), cpu_dev());
        let y = x.mean_all();
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        approx_vec(g_x.as_slice(), &[0.25, 0.25, 0.25, 0.25], 1e-6);
    }

    // ---- deep graph stress test ----

    // ---- restored regression tests from pre-refactor ----

    #[test]
    fn backward_of_mul_realizes_to_other_input() {
        let a_vals = vec![2.0_f32, 3.0, 5.0];
        let b_vals = vec![7.0_f32, 11.0, 13.0];
        let a = Tensor::from_f32(a_vals.clone(), Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(b_vals.clone(), Shape::from_dims(&[3]));
        let c = a.mul(&b);
        let grads = c.backward();
        let g_a = realize_f32(&grads.get(&a).unwrap());
        let g_b = realize_f32(&grads.get(&b).unwrap());
        assert_eq!(g_a.as_slice(), b_vals.as_slice());
        assert_eq!(g_b.as_slice(), a_vals.as_slice());
    }

    #[test]
    fn backward_of_sqr_realizes_to_two_x() {
        let a_vals = vec![2.0_f32, 3.0, 5.0, 7.0];
        let a = Tensor::from_f32(a_vals.clone(), Shape::from_dims(&[4]), cpu_dev());
        let y = a.sqr();
        let grads = y.backward();
        let g_a = realize_f32(&grads.get(&a).unwrap());
        let expected: Vec<f32> = a_vals.iter().map(|&v| 2.0 * v).collect();
        assert_eq!(g_a.as_slice(), expected.as_slice());
    }

    #[test]
    fn backward_of_exp_realizes_to_exp_of_input() {
        let a = Tensor::from_f32(vec![0.0, 1.0, 2.0], Shape::from_dims(&[3]), cpu_dev());
        let y = a.exp();
        let grads = y.backward();
        let g_a = realize_f32(&grads.get(&a).unwrap());
        let expected: Vec<f32> = vec![0.0_f32, 1.0, 2.0].into_iter().map(|v| v.exp()).collect();
        approx_vec(g_a.as_slice(), &expected, 1e-5);
    }

    #[test]
    fn backward_accumulates_via_multi_use() {
        // f(a) = a * a, so df/da = 2a via both inputs of the same Mul.
        let a_vals = vec![3.0_f32, 5.0, 7.0];
        let a = Tensor::from_f32(a_vals.clone(), Shape::from_dims(&[3]), cpu_dev());
        let y = a.mul(&a);
        let grads = y.backward();
        let g_a = realize_f32(&grads.get(&a).unwrap());
        let expected: Vec<f32> = a_vals.iter().map(|&v| 2.0 * v).collect();
        assert_eq!(g_a.as_slice(), expected.as_slice());
    }

    #[test]
    fn backward_of_matmul_hand_computed() {
        // Y = A @ B, A:[2,3], B:[3,2], upstream = ones.
        // Hand-derived: dA = [[15, 19, 23], [15, 19, 23]],
        //               dB = [[5, 5], [7, 7], [9, 9]].
        let a = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let b = a.const_f32_like(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::from_dims(&[3, 2]),
        );
        let y = a.matmul(&b);
        let grads = y.backward();
        let g_a = realize_f32(&grads.get(&a).unwrap());
        let g_b = realize_f32(&grads.get(&b).unwrap());
        assert_eq!(g_a.as_slice(), &[15.0, 19.0, 23.0, 15.0, 19.0, 23.0]);
        assert_eq!(g_b.as_slice(), &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
    }

    #[test]
    fn realize_transpose_matches_ops_transpose_2d() {
        let a = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let t = a.transpose();
        let result = realize_f32(&t);
        assert_eq!(result.shape().dims(), &[3, 2]);
        assert_eq!(result.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // ---- Phase 1: reshape, reduce_sum_to, axis-reduction and max/min backward ----

    #[test]
    fn realize_reshape_is_data_identity() {
        // Reshape [2, 3] → [3, 2] → [6] keeps the data unchanged in row-major order.
        let x = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let y = x.reshape(Shape::from_dims(&[3, 2]));
        assert_eq!(y.shape().dims(), &[3, 2]);
        assert_eq!(realize_f32(&y).as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let z = x.reshape(Shape::from_dims(&[6]));
        assert_eq!(realize_f32(&z).as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn realize_reduce_sum_to_scalar() {
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
        let y = x.reduce_sum_to(Shape::from_dims(&[]));
        assert_eq!(realize_f32(&y).as_slice(), &[10.0]);
    }

    #[test]
    fn realize_reduce_sum_to_along_leading_dim() {
        // [2, 3] → [3]: sum along the leading dim.
        //   [[1, 2, 3], [4, 5, 6]] → [5, 7, 9]
        let x = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let y = x.reduce_sum_to(Shape::from_dims(&[3]));
        assert_eq!(realize_f32(&y).as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn realize_reduce_sum_to_collapses_size_one() {
        // [3, 4] → [3, 1]: sum within each row.
        let x = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::from_dims(&[3, 4]),
            cpu_dev(),
        );
        let y = x.reduce_sum_to(Shape::from_dims(&[3, 1]));
        // row sums: 10, 26, 42
        assert_eq!(realize_f32(&y).as_slice(), &[10.0, 26.0, 42.0]);
    }

    #[test]
    fn realize_unsqueeze_inserts_axis_preserves_data() {
        // Input [2, 3] with values 1..6, unsqueeze at dim 1 → shape
        // [2, 1, 3]. Bytes are unchanged; data should round-trip.
        let x = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let y = x.unsqueeze(1);
        assert_eq!(y.shape().dims(), &[2, 1, 3]);
        assert_eq!(realize_f32(&y).as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn realize_unsqueeze_then_broadcast_then_add() {
        // [3] -unsqueeze(0)→ [1, 3] -broadcast→ [2, 3] -add x→ ...
        // Composed end-to-end through the reference exec.
        let bias = Tensor::from_f32(
            vec![10.0, 20.0, 30.0],
            Shape::from_dims(&[3]),
            cpu_dev(),
        );
        let x = bias.const_f32_like(
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            Shape::from_dims(&[2, 3]),
        );
        let bias_un = bias.unsqueeze(0);
        let bias_b = bias_un.broadcast_to(Shape::from_dims(&[2, 3]));
        let y = x.add(&bias_b);
        // Expected: [11, 21, 31, 12, 22, 32]
        assert_eq!(realize_f32(&y).as_slice(), &[11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    }

    /// Backward of ReduceMaxTo on a scalar reduction with a unique max:
    /// gradient flows entirely to the argmax position.
    #[test]
    fn reduce_max_to_backward_unique_max_routes_full_grad() {
        // x = [1.0, 5.0, 3.0, 2.0]; reduce-max-to [] gives 5.0 at index 1.
        // y = max(x); upstream dL/dy = 7.0; expected dL/dx = [0, 7, 0, 0].
        let x = Tensor::from_f32(
            vec![1.0_f32, 5.0, 3.0, 2.0],
            Shape::from_dims(&[4]),
            cpu_dev(),
        );
        let y = x.reduce_max_to(Shape::from_dims(&[]));
        let grads = y.backward();
        let grad_x = grads.get(&x).expect("dL/dx exists");
        // Default backward starts upstream as ones — for a scalar y,
        // dL/dx is exactly the mask of argmax positions.
        let got = realize_f32(&grad_x);
        assert_eq!(got.as_slice(), &[0.0, 1.0, 0.0, 0.0]);
    }

    /// Backward of ReduceMaxTo with tied maxes: upstream is split
    /// equally (fair-share subgradient) — the standard convention.
    #[test]
    fn reduce_max_to_backward_tied_max_splits_grad_equally() {
        // x = [5.0, 5.0, 3.0, 5.0]; max = 5.0 at indices {0, 1, 3} (3 ties).
        // dL/dy = 1 (default seed). Each tied position gets 1/3.
        let x = Tensor::from_f32(
            vec![5.0_f32, 5.0, 3.0, 5.0],
            Shape::from_dims(&[4]),
            cpu_dev(),
        );
        let y = x.reduce_max_to(Shape::from_dims(&[]));
        let grads = y.backward();
        let grad_x = grads.get(&x).expect("dL/dx exists");
        let got = realize_f32(&grad_x);
        let third = 1.0_f32 / 3.0;
        for (i, &v) in got.as_slice().iter().enumerate() {
            let expected = if i == 2 { 0.0 } else { third };
            assert!(
                (v - expected).abs() < 1e-6,
                "grad_x[{i}] = {v}, expected {expected}",
            );
        }
        // Conservation check: sum of grad_x equals upstream (1.0).
        let sum: f32 = got.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "grad sum = {sum}, want 1.0");
    }

    /// Backward of ReduceMaxTo over a multi-row reduction: gradient
    /// per-row routes independently to that row's argmax.
    #[test]
    fn reduce_max_to_backward_keepdim_routes_per_row() {
        // x [2,3]: [[1, 7, 3], [4, 2, 6]]; reduce-max-to [2,1] = [[7],[6]].
        // upstream = [[2.0], [3.0]].
        // expected dL/dx = [[0, 2, 0], [0, 0, 3]].
        let x = Tensor::from_f32(
            vec![1.0_f32, 7.0, 3.0, 4.0, 2.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let y = x.reduce_max_to(Shape::from_dims(&[2, 1]));
        // Multiply by an explicit upstream-shaped const so the seeded
        // upstream is non-uniform: dL/dy = [[2.0], [3.0]].
        let upstream = x.const_f32_like(vec![2.0, 3.0], Shape::from_dims(&[2, 1]));
        let scaled = y.mul(&upstream);
        let loss = scaled.sum_all();
        let grads = loss.backward();
        let grad_x = grads.get(&x).expect("dL/dx exists");
        let got = realize_f32(&grad_x);
        assert_eq!(got.as_slice(), &[0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn realize_reduce_max_to_along_leading_dim() {
        // [2, 3] → [3]: max along the leading dim.
        //   [[1, 5, 3], [4, 2, 6]] → [4, 5, 6]
        let x = Tensor::from_f32(
            vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let y = x.reduce_max_to(Shape::from_dims(&[3]));
        assert_eq!(realize_f32(&y).as_slice(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn realize_reduce_max_to_collapses_size_one() {
        // [3, 4] → [3, 1]: max within each row.
        let x = Tensor::from_f32(
            vec![1.0, 7.0, 3.0, 4.0, 12.0, 6.0, 5.0, 8.0, 9.0, 10.0, 11.0, 2.0],
            Shape::from_dims(&[3, 4]),
            cpu_dev(),
        );
        let y = x.reduce_max_to(Shape::from_dims(&[3, 1]));
        // row maxes: 7, 12, 11
        assert_eq!(realize_f32(&y).as_slice(), &[7.0, 12.0, 11.0]);
    }

    #[test]
    fn backward_of_broadcast_to_is_reduce_sum_to() {
        // y = broadcast_to(x, [2, 3]) where x has shape [3]. Backward
        // should sum along the leading dim, giving grad_x of shape [3] =
        // [2, 2, 2] (each column summed across 2 rows of ones).
        let x = Tensor::from_f32(vec![10.0, 20.0, 30.0], Shape::from_dims(&[3]), cpu_dev());
        let y = x.broadcast_to(Shape::from_dims(&[2, 3]));
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        assert_eq!(g_x.shape().dims(), &[3]);
        assert_eq!(g_x.as_slice(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn backward_of_reshape_routes_gradient_through_shape() {
        // y = reshape(x, [3, 2]). Backward reshapes upstream back to x.shape.
        let x = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let y = x.reshape(Shape::from_dims(&[3, 2]));
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        assert_eq!(g_x.shape().dims(), &[2, 3]);
        assert_eq!(g_x.as_slice(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn backward_of_sum_dim_distributes_upstream_along_reduced_axis() {
        // y = sum_dim(x, dim=1) on a [2, 3] input → shape [2].
        // Backward should distribute each element of upstream to 3 copies
        // along dim 1. With upstream ones (shape [2]) the result is all 1s
        // in x.shape.
        let x = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let y = x.sum_dim(1);
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        assert_eq!(g_x.shape().dims(), &[2, 3]);
        assert_eq!(g_x.as_slice(), &[1.0; 6]);
    }

    #[test]
    fn backward_of_mean_dim_is_one_over_reduced_size() {
        // mean_dim(x, dim=1) with x.shape = [2, 3]. Backward: each element
        // of gradient is 1/3 (upstream is ones, divided by 3).
        let x = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let y = x.mean_dim(1);
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        assert_eq!(g_x.shape().dims(), &[2, 3]);
        let third = 1.0_f32 / 3.0;
        for &v in g_x.as_slice() {
            assert!((v - third).abs() < 1e-6, "got {v}, expected ~1/3");
        }
    }

    #[test]
    fn backward_of_max_all_routes_gradient_to_argmax() {
        // max([1, 5, 2, 5, 3]) = 5 at positions 1 and 3 (tie).
        // The indicator-via-step gradient distributes upstream=1 equally:
        // positions 1 and 3 each get 1.0 (no 0.5/0.5 split since our rule
        // sums indicator values, not normalizes). Non-max positions get 0.
        let x = Tensor::from_f32(
            vec![1.0, 5.0, 2.0, 5.0, 3.0],
            Shape::from_dims(&[5]),
            cpu_dev(),
        );
        let y = x.max_all();
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        assert_eq!(g_x.as_slice(), &[0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn backward_of_min_all_routes_gradient_to_argmin() {
        let x = Tensor::from_f32(
            vec![3.0, 1.0, 4.0, 1.0, 5.0],
            Shape::from_dims(&[5]),
            cpu_dev(),
        );
        let y = x.min_all();
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        assert_eq!(g_x.as_slice(), &[0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn backward_of_max_dim_is_per_row_indicator() {
        // [[1, 5, 2], [4, 3, 6]], max along dim 1 → [5, 6]
        // Gradient: the position of the max in each row gets 1, others 0.
        //   Row 0: col 1 is max → [0, 1, 0]
        //   Row 1: col 2 is max → [0, 0, 1]
        let x = Tensor::from_f32(
            vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let y = x.max_dim(1);
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        assert_eq!(g_x.shape().dims(), &[2, 3]);
        assert_eq!(g_x.as_slice(), &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    // ---- Phase 3: index tensors and gather ----

    #[test]
    fn realize_u32_const_via_graph() {
        let a = Tensor::from_u32(vec![10, 20, 30], Shape::from_dims(&[3]), cpu_dev());
        let result = realize(&a).into_u32();
        assert_eq!(result.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn realize_index_select_along_first_dim() {
        // Data: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        // Indices: [2, 0, 2]
        // Expected (rows 2, 0, 2):
        //   [[7, 8, 9], [1, 2, 3], [7, 8, 9]]
        let data = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Shape::from_dims(&[3, 3]),
            cpu_dev(),
        );
        let idx = data.const_u32_like(vec![2, 0, 2], Shape::from_dims(&[3]));
        let out = data.index_select(0, &idx);
        assert_eq!(out.shape().dims(), &[3, 3]);
        let result = realize_f32(&out);
        assert_eq!(
            result.as_slice(),
            &[7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
        );
    }

    #[test]
    fn realize_index_select_along_second_dim() {
        // Pick columns 2, 0 from a 3×3 → shape [3, 2].
        let data = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Shape::from_dims(&[3, 3]),
            cpu_dev(),
        );
        let idx = data.const_u32_like(vec![2, 0], Shape::from_dims(&[2]));
        let out = data.index_select(1, &idx);
        assert_eq!(out.shape().dims(), &[3, 2]);
        let result = realize_f32(&out);
        assert_eq!(result.as_slice(), &[3.0, 1.0, 6.0, 4.0, 9.0, 7.0]);
    }

    #[test]
    fn realize_gather_with_nd_indices() {
        // data = [[1, 2, 3], [4, 5, 6]]
        // idx  = [[0, 2], [1, 0]] (along dim 1)
        // out[0,0] = data[0, 0] = 1
        // out[0,1] = data[0, 2] = 3
        // out[1,0] = data[1, 1] = 5
        // out[1,1] = data[1, 0] = 4
        let data = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let idx = data.const_u32_like(vec![0, 2, 1, 0], Shape::from_dims(&[2, 2]));
        let out = data.gather(1, &idx);
        assert_eq!(out.shape().dims(), &[2, 2]);
        let result = realize_f32(&out);
        assert_eq!(result.as_slice(), &[1.0, 3.0, 5.0, 4.0]);
    }

    #[test]
    fn embedding_lookup_via_index_select() {
        // Simulates an embedding table: 5 rows (vocab) × 3 columns (hidden).
        // Look up tokens [3, 1, 4] → rows 3, 1, 4.
        let table = Tensor::from_f32(
            vec![
                0.0, 0.0, 0.0,   // id 0
                1.0, 1.0, 1.0,   // id 1
                2.0, 2.0, 2.0,   // id 2
                3.0, 3.0, 3.0,   // id 3
                4.0, 4.0, 4.0,   // id 4
            ],
            Shape::from_dims(&[5, 3]),
            cpu_dev(),
        );
        let ids = table.const_u32_like(vec![3, 1, 4], Shape::from_dims(&[3]));
        let embeddings = table.index_select(0, &ids);
        assert_eq!(embeddings.shape().dims(), &[3, 3]);
        let result = realize_f32(&embeddings);
        assert_eq!(
            result.as_slice(),
            &[3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
        );
    }

    #[test]
    #[should_panic(expected = "must be U32")]
    fn index_select_rejects_non_u32_indices() {
        let data = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let bad_idx = data.const_f32_like(vec![0.0, 1.0], Shape::from_dims(&[2]));
        let _ = data.index_select(0, &bad_idx);
    }

    // ---- Phase 4: end-to-end 2-layer MLP ----

    #[test]
    fn two_layer_mlp_forward_and_backward_end_to_end() {
        // Build a tiny 2-layer MLP:
        //
        //   h  = relu(x @ W1 + b1)      // shape [batch, hidden]
        //   y  = h  @ W2 + b2           // shape [batch, out]
        //   L  = sum_all(sqr(y))        // scalar loss (sum of squared outputs)
        //
        // With:
        //   batch   = 2
        //   in_dim  = 3
        //   hidden  = 4
        //   out_dim = 2
        //
        // Backward computes dL/dW1, dL/db1, dL/dW2, dL/db2. We verify all
        // the parameter gradient shapes and spot-check values by
        // hand-computing a few elements. This is the most end-to-end test
        // in the reference backend — it exercises matmul, bias add with
        // broadcasting, relu with its step-based backward, sqr, and the
        // sum-all-to-scalar reduction, all composed through an automatic
        // backward graph.
        let x = Tensor::from_f32(
            // 2 rows × 3 cols
            vec![1.0, 2.0, 3.0, 0.5, -0.5, 1.5],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let w1 = x.const_f32_like(
            // 3 rows × 4 cols — simple integer weights
            vec![
                1.0, 0.0, -1.0, 0.5, //
                0.0, 1.0, 0.5, -0.5, //
                -1.0, 0.5, 0.0, 1.0, //
            ],
            Shape::from_dims(&[3, 4]),
        );
        let b1 = x.const_f32_like(
            // Broadcast [4] over [2, 4] via broadcast_to before add.
            vec![0.1, 0.2, 0.3, 0.4],
            Shape::from_dims(&[4]),
        );
        let w2 = x.const_f32_like(
            // 4 rows × 2 cols
            vec![
                1.0, -1.0, //
                0.5, 0.5, //
                -1.0, 1.0, //
                0.0, 1.0, //
            ],
            Shape::from_dims(&[4, 2]),
        );
        let b2 = x.const_f32_like(
            vec![-0.1, 0.05],
            Shape::from_dims(&[2]),
        );

        // Forward pass.
        let xw1 = x.matmul(&w1);
        let b1_bcast = b1.broadcast_to(Shape::from_dims(&[2, 4]));
        let pre1 = xw1.add(&b1_bcast);
        let h = pre1.relu();
        let hw2 = h.matmul(&w2);
        let b2_bcast = b2.broadcast_to(Shape::from_dims(&[2, 2]));
        let y = hw2.add(&b2_bcast);
        let sq = y.sqr();
        let loss = sq.sum_all();

        // Loss should be a finite positive scalar.
        let loss_val = realize_f32(&loss);
        assert_eq!(loss_val.shape().dims(), &[] as &[usize]);
        assert!(loss_val.as_slice()[0].is_finite());
        assert!(loss_val.as_slice()[0] >= 0.0);

        // Backward pass and parameter gradient shape checks.
        let grads = loss.backward();
        let g_w1 = realize_f32(&grads.get(&w1).expect("W1 must have a gradient"));
        let g_b1 = realize_f32(&grads.get(&b1).expect("b1 must have a gradient"));
        let g_w2 = realize_f32(&grads.get(&w2).expect("W2 must have a gradient"));
        let g_b2 = realize_f32(&grads.get(&b2).expect("b2 must have a gradient"));
        assert_eq!(g_w1.shape().dims(), &[3, 4]);
        assert_eq!(g_b1.shape().dims(), &[4]);
        assert_eq!(g_w2.shape().dims(), &[4, 2]);
        assert_eq!(g_b2.shape().dims(), &[2]);

        // Verify g_b2 equals the column-sum of `2*y` (since dL/dy = 2y and
        // the bias broadcasts, its gradient sums across the batch).
        // Compute it directly in Rust to compare.
        let y_val = realize_f32(&y);
        let y_data = y_val.as_slice();
        let expected_g_b2: Vec<f32> = (0..2)
            .map(|col| {
                (0..2)
                    .map(|row| 2.0 * y_data[row * 2 + col])
                    .sum::<f32>()
            })
            .collect();
        for (i, (&got, &exp)) in g_b2.as_slice().iter().zip(&expected_g_b2).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "g_b2[{i}]: got {got}, expected {exp}",
            );
        }
    }

    // ---- Phase 5: higher-order gradient smoke test ----

    #[test]
    fn higher_order_gradient_of_sqr_is_two() {
        // f(x)  = x²           (element-wise)
        // f'(x) = 2x
        // f''(x)= 2
        //
        // Build the forward pass, run backward once to get g = 2x, then
        // sum g to a scalar and run backward again. The second derivative
        // should be a tensor of all 2.0s of the same shape as x, summed
        // down. Actually: the gradient of `sum(2x)` w.r.t. x is `2`
        // everywhere (element-wise), so the second-order gradient is
        // `[2, 2, 2]`.
        //
        // This test's real value is that the nested backward does not
        // panic or loop infinitely. It also catches regressions in
        // backward composability: every op emitted by the first backward
        // pass must itself have a backward rule.
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let y = x.sqr();
        let first_grads = y.backward();
        let g_x = first_grads.get(&x).unwrap();
        // Sum the gradient to get a scalar we can differentiate again.
        // sum(2x) over x = [1,2,3] = 2 + 4 + 6 = 12.
        let g_scalar = g_x.sum_all();
        // First-order sanity check: the realized scalar is 12.
        let g_scalar_val = realize_f32(&g_scalar);
        assert_eq!(g_scalar_val.as_slice(), &[12.0]);

        // Now differentiate that scalar with respect to x.
        // d(sum(2x))/dx = [2, 2, 2].
        let second_grads = g_scalar.backward();
        let gg_x = second_grads
            .get(&x)
            .expect("x must have a second-order gradient");
        let gg_x_val = realize_f32(&gg_x);
        assert_eq!(gg_x_val.shape().dims(), &[3]);
        assert_eq!(gg_x_val.as_slice(), &[2.0, 2.0, 2.0]);
    }

    // ---- index_add / scatter_add forward + gather/index_select backward ----

    #[test]
    fn realize_index_add_accumulates_into_base() {
        // base: [10, 20, 30, 40, 50]; indices: [1, 3]; src: [100, 200]
        // Expected: [10, 120, 30, 240, 50]
        let base = Tensor::from_f32(
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            Shape::from_dims(&[5]),
            cpu_dev(),
        );
        let idx = base.const_u32_like(vec![1, 3], Shape::from_dims(&[2]));
        let src = base.const_f32_like(vec![100.0, 200.0], Shape::from_dims(&[2]));
        let out = base.index_add(0, &idx, &src);
        let result = realize_f32(&out);
        assert_eq!(result.as_slice(), &[10.0, 120.0, 30.0, 240.0, 50.0]);
    }

    #[test]
    fn realize_scatter_add_nd_accumulates() {
        // base: [[0, 0, 0], [0, 0, 0]], dim = 1
        // indices: [[0, 2], [1, 0]], src: [[1, 3], [5, 4]]
        // out[0, 0] += 1, out[0, 2] += 3
        // out[1, 1] += 5, out[1, 0] += 4
        // Expected: [[1, 0, 3], [4, 5, 0]]
        let base = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let idx = base.const_u32_like(vec![0, 2, 1, 0], Shape::from_dims(&[2, 2]));
        let src = base.const_f32_like(vec![1.0, 3.0, 5.0, 4.0], Shape::from_dims(&[2, 2]));
        let out = base.scatter_add(1, &idx, &src);
        let result = realize_f32(&out);
        assert_eq!(result.as_slice(), &[1.0, 0.0, 3.0, 4.0, 5.0, 0.0]);
    }

    #[test]
    fn backward_of_index_select_scatters_upstream_to_indices() {
        // data: [10, 20, 30, 40], indices: [2, 0, 2]
        // out = [30, 10, 30]
        // grad_data: position 2 gets 2 contributions (2 upstream 1s),
        //            position 0 gets 1 contribution,
        //            positions 1 and 3 get 0.
        // Expected grad_data: [1, 0, 2, 0]
        let data = Tensor::from_f32(
            vec![10.0, 20.0, 30.0, 40.0],
            Shape::from_dims(&[4]),
            cpu_dev(),
        );
        let idx = data.const_u32_like(vec![2, 0, 2], Shape::from_dims(&[3]));
        let out = data.index_select(0, &idx);
        let grads = out.backward();
        let g_data = realize_f32(&grads.get(&data).unwrap());
        assert_eq!(g_data.as_slice(), &[1.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn backward_of_gather_scatters_upstream_to_indices() {
        // data: [[1, 2, 3], [4, 5, 6]], dim = 1
        // indices: [[0, 2], [1, 0]]
        // out = [[1, 3], [5, 4]]
        // Backward: each upstream position contributes to the gathered
        // data position. With upstream ones:
        //   grad_data[0, 0] += 1, grad_data[0, 2] += 1
        //   grad_data[1, 1] += 1, grad_data[1, 0] += 1
        // Expected: [[1, 0, 1], [1, 1, 0]]
        let data = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let idx = data.const_u32_like(vec![0, 2, 1, 0], Shape::from_dims(&[2, 2]));
        let out = data.gather(1, &idx);
        let grads = out.backward();
        let g_data = realize_f32(&grads.get(&data).unwrap());
        assert_eq!(g_data.as_slice(), &[1.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    // ---- softmax + layer_norm backward ----

    #[test]
    fn backward_of_softmax_last_dim_hand_verified() {
        // For x = [1, 2, 3], softmax(x) ≈ [0.0900, 0.2447, 0.6652].
        // With upstream ones, the backward is:
        //   dot = sum(y * 1) = 1.0
        //   grad_x_i = y_i * (1 - 1) = 0
        // So the gradient is all zeros when upstream is uniform.
        // This reflects the fact that softmax is invariant to uniform
        // shifts in its upstream (since probabilities sum to 1).
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let y = x.softmax_last_dim();
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        for &v in g_x.as_slice() {
            assert!(v.abs() < 1e-5, "expected ~0, got {v}");
        }
    }

    #[test]
    fn backward_of_softmax_finite_difference_check() {
        // Check softmax backward against a finite-difference estimate
        // with non-uniform upstream so the gradient is non-zero.
        // Use a "pretend loss" L = sum(softmax(x) * w) where w is fixed.
        // Then dL/dx should equal softmax's backward with upstream = w.
        // We verify by perturbing each x_i and recomputing the loss.
        let x_vals = vec![0.5_f32, 1.5, -0.5, 2.0];
        let w_vals = vec![0.1_f32, 0.3, 0.2, 0.4];
        let x = Tensor::from_f32(x_vals.clone(), Shape::from_dims(&[4]), cpu_dev());
        let w = x.const_f32_like(w_vals.clone(), Shape::from_dims(&[4]));
        let loss = x.softmax_last_dim().mul(&w).sum_all();
        let grads = loss.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());

        // Finite-difference.
        let eps = 1e-3_f32;
        let base_loss = {
            let s = softmax_vec(&x_vals);
            s.iter().zip(&w_vals).map(|(a, b)| a * b).sum::<f32>()
        };
        for i in 0..x_vals.len() {
            let mut perturbed = x_vals.clone();
            perturbed[i] += eps;
            let new_loss: f32 = {
                let s = softmax_vec(&perturbed);
                s.iter().zip(&w_vals).map(|(a, b)| a * b).sum::<f32>()
            };
            let fd = (new_loss - base_loss) / eps;
            let analytic = g_x.as_slice()[i];
            assert!(
                (fd - analytic).abs() < 1e-2,
                "softmax backward disagrees at {i}: fd={fd}, analytic={analytic}",
            );
        }
    }

    fn softmax_vec(x: &[f32]) -> Vec<f32> {
        let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|v| v / sum).collect()
    }

    #[test]
    fn backward_of_layer_norm_finite_difference_check() {
        // Check LN backward against finite differences with non-uniform w.
        let x_vals = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let w_vals = vec![0.5_f32, -0.3, 0.8, 0.1, -0.7];
        let x = Tensor::from_f32(x_vals.clone(), Shape::from_dims(&[5]), cpu_dev());
        let w = x.const_f32_like(w_vals.clone(), Shape::from_dims(&[5]));
        let eps_ln = 1e-5_f64;
        let loss = x.layer_norm_last_dim(eps_ln).mul(&w).sum_all();
        let grads = loss.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());

        let eps_fd = 1e-3_f32;
        let base_loss = ln_loss(&x_vals, &w_vals, eps_ln as f32);
        for i in 0..x_vals.len() {
            let mut perturbed = x_vals.clone();
            perturbed[i] += eps_fd;
            let new_loss = ln_loss(&perturbed, &w_vals, eps_ln as f32);
            let fd = (new_loss - base_loss) / eps_fd;
            let analytic = g_x.as_slice()[i];
            assert!(
                (fd - analytic).abs() < 5e-2,
                "LN backward disagrees at {i}: fd={fd}, analytic={analytic}",
            );
        }
    }

    fn ln_loss(x: &[f32], w: &[f32], eps: f32) -> f32 {
        let n = x.len() as f32;
        let mean: f32 = x.iter().sum::<f32>() / n;
        let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let rstd = 1.0 / (var + eps).sqrt();
        x.iter()
            .zip(w)
            .map(|(&xv, &wv)| (xv - mean) * rstd * wv)
            .sum()
    }

    // ---- concat, slice, scalar ops ----

    #[test]
    fn realize_concat_along_dim_1() {
        // [[1, 2], [3, 4]] ++ [[5, 6, 7], [8, 9, 10]] along dim 1
        // = [[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]]
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), cpu_dev());
        let b = a.const_f32_like(
            vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            Shape::from_dims(&[2, 3]),
        );
        let c = a.concat(&b, 1);
        assert_eq!(c.shape().dims(), &[2, 5]);
        assert_eq!(
            realize_f32(&c).as_slice(),
            &[1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0],
        );
    }

    #[test]
    fn realize_slice_narrows_along_dim() {
        let x = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Shape::from_dims(&[3, 3]),
            cpu_dev(),
        );
        // Slice rows [1, 3) → [[4,5,6], [7,8,9]]
        let s = x.slice(0, 1, 2);
        assert_eq!(s.shape().dims(), &[2, 3]);
        assert_eq!(realize_f32(&s).as_slice(), &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        // Slice cols [0, 2) → [[1,2], [4,5], [7,8]]
        let s2 = x.slice(1, 0, 2);
        assert_eq!(s2.shape().dims(), &[3, 2]);
        assert_eq!(realize_f32(&s2).as_slice(), &[1.0, 2.0, 4.0, 5.0, 7.0, 8.0]);
    }

    #[test]
    fn backward_of_concat_splits_upstream() {
        // a: [1, 2, 3], b: [4, 5], concat dim 0 → [1, 2, 3, 4, 5]
        // Upstream ones, so grad_a = [1, 1, 1], grad_b = [1, 1]
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![4.0, 5.0], Shape::from_dims(&[2]));
        let c = a.concat(&b, 0);
        let grads = c.backward();
        assert_eq!(realize_f32(&grads.get(&a).unwrap()).as_slice(), &[1.0, 1.0, 1.0]);
        assert_eq!(realize_f32(&grads.get(&b).unwrap()).as_slice(), &[1.0, 1.0]);
    }

    #[test]
    fn backward_of_slice_zero_pads_around_upstream() {
        // x: [10, 20, 30, 40, 50], slice dim 0 start 1 len 3 → [20, 30, 40]
        // Upstream ones → grad_x = [0, 1, 1, 1, 0]
        let x = Tensor::from_f32(
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            Shape::from_dims(&[5]),
            cpu_dev(),
        );
        let s = x.slice(0, 1, 3);
        let grads = s.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        assert_eq!(g_x.as_slice(), &[0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn realize_scalar_add_and_mul() {
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let y = x.add_scalar(10.0);
        assert_eq!(realize_f32(&y).as_slice(), &[11.0, 12.0, 13.0]);
        let z = x.mul_scalar(2.5);
        assert_eq!(realize_f32(&z).as_slice(), &[2.5, 5.0, 7.5]);
    }

    #[test]
    fn backward_of_scalar_ops() {
        // y = x * 3 + 7, dy/dx = 3 everywhere.
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let y = x.mul_scalar(3.0).add_scalar(7.0);
        let grads = y.backward();
        assert_eq!(
            realize_f32(&grads.get(&x).unwrap()).as_slice(),
            &[3.0, 3.0, 3.0],
        );
    }

    // ---- broadcast-aware binary ops in the graph ----

    #[test]
    fn broadcast_add_row_vector_against_matrix() {
        // Matrix [2, 3] + row vector [3] → matrix [2, 3]
        let m = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let r = m.const_f32_like(vec![10.0, 20.0, 30.0], Shape::from_dims(&[3]));
        let out = m.broadcast_add(&r);
        assert_eq!(out.shape().dims(), &[2, 3]);
        assert_eq!(
            realize_f32(&out).as_slice(),
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
        );
    }

    #[test]
    fn broadcast_mul_col_against_row_makes_outer_product() {
        // [3, 1] * [1, 4] → [3, 4] outer product
        let col = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3, 1]), cpu_dev());
        let row = col.const_f32_like(vec![10.0, 20.0, 30.0, 40.0], Shape::from_dims(&[1, 4]));
        let out = col.broadcast_mul(&row);
        assert_eq!(out.shape().dims(), &[3, 4]);
        assert_eq!(
            realize_f32(&out).as_slice(),
            &[
                10.0, 20.0, 30.0, 40.0, //
                20.0, 40.0, 60.0, 80.0, //
                30.0, 60.0, 90.0, 120.0, //
            ],
        );
    }

    #[test]
    fn backward_through_broadcast_add_sums_along_broadcast_dims() {
        // Matrix [2, 3] + row vector [3] → matrix [2, 3]. Each element of
        // the row vector contributes to 2 rows of the output, so its
        // gradient should be the column sums of the upstream (ones here).
        // Upstream ones → grad_row = [2, 2, 2], grad_matrix = ones.
        let m = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let r = m.const_f32_like(vec![10.0, 20.0, 30.0], Shape::from_dims(&[3]));
        let out = m.broadcast_add(&r);
        let grads = out.backward();
        let g_m = realize_f32(&grads.get(&m).unwrap());
        let g_r = realize_f32(&grads.get(&r).unwrap());
        assert_eq!(g_m.shape().dims(), &[2, 3]);
        assert_eq!(g_m.as_slice(), &[1.0; 6]);
        assert_eq!(g_r.shape().dims(), &[3]);
        assert_eq!(g_r.as_slice(), &[2.0, 2.0, 2.0]);
    }

    // ---- training loop ----

    #[test]
    fn train_2_layer_mlp_reduces_loss_on_fixed_input() {
        // Fit a tiny 2-layer MLP to a fixed target using SGD. Each
        // iteration rebuilds the graph with the current parameter
        // values, runs forward+backward, extracts gradients, and updates
        // the parameter vectors in place. After N iterations the loss
        // should be meaningfully lower than when we started.
        let lr = 0.05_f32;
        let iters = 50;

        // Fixed training example: x → target
        let x_data = vec![1.0_f32, 2.0, -1.0];
        let target_data = vec![0.5_f32, -0.3];

        // Parameters (initialized with small integer values).
        let mut w1 = vec![
            0.1_f32, 0.0, -0.1, 0.2, //
            0.0, 0.1, 0.2, -0.1, //
            -0.1, 0.2, 0.0, 0.1, //
        ]; // [3, 4]
        let mut b1 = vec![0.0_f32; 4];
        let mut w2 = vec![
            0.1_f32, -0.1, //
            0.2, 0.1, //
            -0.2, 0.2, //
            0.1, 0.0, //
        ]; // [4, 2]
        let mut b2 = vec![0.0_f32; 2];

        let initial_loss = {
            let loss = build_mlp_loss(&x_data, &target_data, &w1, &b1, &w2, &b2);
            realize_f32(&loss).as_slice()[0]
        };

        for _ in 0..iters {
            let x = Tensor::from_f32(x_data.clone(), Shape::from_dims(&[1, 3]), cpu_dev());
            let target = x.const_f32_like(target_data.clone(), Shape::from_dims(&[1, 2]));
            let w1_t = x.const_f32_like(w1.clone(), Shape::from_dims(&[3, 4]));
            let b1_t = x.const_f32_like(b1.clone(), Shape::from_dims(&[4]));
            let w2_t = x.const_f32_like(w2.clone(), Shape::from_dims(&[4, 2]));
            let b2_t = x.const_f32_like(b2.clone(), Shape::from_dims(&[2]));

            let h = x.matmul(&w1_t).broadcast_add(&b1_t).relu();
            let y = h.matmul(&w2_t).broadcast_add(&b2_t);
            let diff = y.sub(&target);
            let loss = diff.sqr().sum_all();

            let grads = loss.backward();
            let g_w1 = realize_f32(&grads.get(&w1_t).unwrap());
            let g_b1 = realize_f32(&grads.get(&b1_t).unwrap());
            let g_w2 = realize_f32(&grads.get(&w2_t).unwrap());
            let g_b2 = realize_f32(&grads.get(&b2_t).unwrap());

            for (w, &g) in w1.iter_mut().zip(g_w1.as_slice()) {
                *w -= lr * g;
            }
            for (bv, &g) in b1.iter_mut().zip(g_b1.as_slice()) {
                *bv -= lr * g;
            }
            for (w, &g) in w2.iter_mut().zip(g_w2.as_slice()) {
                *w -= lr * g;
            }
            for (bv, &g) in b2.iter_mut().zip(g_b2.as_slice()) {
                *bv -= lr * g;
            }
        }

        let final_loss = {
            let loss = build_mlp_loss(&x_data, &target_data, &w1, &b1, &w2, &b2);
            realize_f32(&loss).as_slice()[0]
        };

        assert!(
            final_loss < initial_loss * 0.2,
            "loss should drop substantially: initial={initial_loss}, final={final_loss}",
        );
    }

    fn build_mlp_loss(
        x_data: &[f32],
        target_data: &[f32],
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
    ) -> Tensor {
        let x = Tensor::from_f32(x_data.to_vec(), Shape::from_dims(&[1, 3]), cpu_dev());
        let target = x.const_f32_like(target_data.to_vec(), Shape::from_dims(&[1, 2]));
        let w1_t = x.const_f32_like(w1.to_vec(), Shape::from_dims(&[3, 4]));
        let b1_t = x.const_f32_like(b1.to_vec(), Shape::from_dims(&[4]));
        let w2_t = x.const_f32_like(w2.to_vec(), Shape::from_dims(&[4, 2]));
        let b2_t = x.const_f32_like(b2.to_vec(), Shape::from_dims(&[2]));

        let h = x.matmul(&w1_t).broadcast_add(&b1_t).relu();
        let y = h.matmul(&w2_t).broadcast_add(&b2_t);
        y.sub(&target).sqr().sum_all()
    }

    // ---- powi, clamp, max/min between tensors ----

    #[test]
    fn realize_powi_cubes_and_inverses() {
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
        assert_eq!(realize_f32(&x.powi(3)).as_slice(), &[1.0, 8.0, 27.0, 64.0]);
        // Negative exponent: reciprocal of square.
        let inv = realize_f32(&x.powi(-2));
        let expected = [1.0_f32, 0.25, 1.0 / 9.0, 1.0 / 16.0];
        for (got, exp) in inv.as_slice().iter().zip(&expected) {
            assert!((got - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn backward_of_powi_is_n_times_x_to_nm1() {
        // f(x) = x^3, f'(x) = 3x².
        let x_vals = vec![1.0_f32, 2.0, 3.0];
        let x = Tensor::from_f32(x_vals.clone(), Shape::from_dims(&[3]), cpu_dev());
        let y = x.powi(3);
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        let expected: Vec<f32> = x_vals.iter().map(|v| 3.0 * v * v).collect();
        approx_vec(g_x.as_slice(), &expected, 1e-5);
    }

    #[test]
    fn realize_clamp_enforces_bounds() {
        let x = Tensor::from_f32(
            vec![-5.0, -1.0, 0.0, 1.0, 5.0],
            Shape::from_dims(&[5]),
            cpu_dev(),
        );
        let y = x.clamp(-1.0, 1.0);
        assert_eq!(realize_f32(&y).as_slice(), &[-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn backward_of_clamp_is_zero_outside_range() {
        // Only the interior positions get upstream; the clamped ones are zero.
        let x = Tensor::from_f32(
            vec![-5.0, -1.0, 0.0, 1.0, 5.0],
            Shape::from_dims(&[5]),
            cpu_dev(),
        );
        let y = x.clamp(-1.0, 1.0);
        let grads = y.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        // Our step-at-boundary convention: step(0) = 0, so positions
        // exactly at -1.0, 0.0 where x > -1 fails (x == -1 is exactly 0),
        // and at 1.0 similarly. Interior position 0.0 is the only one
        // with both step factors equal to 1.
        // Expected: [0, 0, 1, 0, 0] (only position 2, value 0.0, is in the
        //           strict interior).
        assert_eq!(g_x.as_slice(), &[0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn realize_maximum_and_minimum_elementwise() {
        let a = Tensor::from_f32(vec![1.0, 5.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
        let b = a.const_f32_like(vec![2.0, 4.0, 6.0, 1.0], Shape::from_dims(&[4]));
        let max_ab = a.maximum(&b);
        let min_ab = a.minimum(&b);
        assert_eq!(realize_f32(&max_ab).as_slice(), &[2.0, 5.0, 6.0, 4.0]);
        assert_eq!(realize_f32(&min_ab).as_slice(), &[1.0, 4.0, 3.0, 1.0]);
    }

    #[test]
    fn backward_of_maximum_routes_to_larger_input() {
        // For each position the gradient goes to whichever input is larger.
        let a = Tensor::from_f32(vec![1.0, 5.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
        let b = a.const_f32_like(vec![2.0, 4.0, 6.0, 1.0], Shape::from_dims(&[4]));
        let out = a.maximum(&b);
        let grads = out.backward();
        let g_a = realize_f32(&grads.get(&a).unwrap());
        let g_b = realize_f32(&grads.get(&b).unwrap());
        // pos 0: b wins (2 > 1) → grad_a = 0, grad_b = 1
        // pos 1: a wins (5 > 4) → grad_a = 1, grad_b = 0
        // pos 2: b wins (6 > 3) → grad_a = 0, grad_b = 1
        // pos 3: a wins (4 > 1) → grad_a = 1, grad_b = 0
        assert_eq!(g_a.as_slice(), &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(g_b.as_slice(), &[1.0, 0.0, 1.0, 0.0]);
    }

    // ---- negative log likelihood classification loss using softmax + gather ----

    #[test]
    fn train_softmax_classifier_via_nll_loss() {
        // Train a tiny single-layer classifier: logits = x @ W + b,
        // loss = -log(softmax(logits)[target_class]) per sample, summed.
        //
        // This exercises softmax backward, gather backward, log, and
        // everything composed through a real training loop.
        //
        // Setup: 2 training examples, 3 input features, 4 classes.
        let x_data = vec![
            1.0_f32, 0.0, -1.0, //
            0.0, 1.0, 1.0,     //
        ];
        let targets = vec![0_u32, 2_u32];
        let mut w = vec![
            0.01_f32, 0.02, -0.01, 0.0, //
            -0.02, 0.01, 0.01, 0.02, //
            0.02, -0.01, 0.02, -0.02, //
        ];
        let mut b = vec![0.0_f32; 4];
        let lr = 0.1_f32;

        let build_loss = |w: &[f32], b: &[f32]| -> (Tensor, Tensor, Tensor) {
            let x = Tensor::from_f32(x_data.clone(), Shape::from_dims(&[2, 3]), cpu_dev());
            let w_t = x.const_f32_like(w.to_vec(), Shape::from_dims(&[3, 4]));
            let b_t = x.const_f32_like(b.to_vec(), Shape::from_dims(&[4]));
            let logits = x.matmul(&w_t).broadcast_add(&b_t);
            let probs = logits.softmax_last_dim();
            // Build index for gather along dim 1: want probs[i, targets[i]].
            let tgt_tensor = x.const_u32_like(targets.clone(), Shape::from_dims(&[2, 1]));
            // gather along dim 1 with indices shape [2, 1] → output [2, 1]
            let picked = probs.gather(1, &tgt_tensor);
            // NLL loss: -log(picked), summed.
            let neg_logp = picked.log().mul_scalar(-1.0);
            let loss = neg_logp.sum_all();
            (loss, w_t, b_t)
        };

        let initial_loss = {
            let (loss, _, _) = build_loss(&w, &b);
            realize_f32(&loss).as_slice()[0]
        };

        for _ in 0..30 {
            let (loss, w_t, b_t) = build_loss(&w, &b);
            let grads = loss.backward();
            let g_w = realize_f32(&grads.get(&w_t).unwrap());
            let g_b = realize_f32(&grads.get(&b_t).unwrap());
            for (param, &g) in w.iter_mut().zip(g_w.as_slice()) {
                *param -= lr * g;
            }
            for (param, &g) in b.iter_mut().zip(g_b.as_slice()) {
                *param -= lr * g;
            }
        }

        let final_loss = {
            let (loss, _, _) = build_loss(&w, &b);
            realize_f32(&loss).as_slice()[0]
        };

        assert!(
            final_loss < initial_loss * 0.7,
            "NLL loss should drop: initial={initial_loss}, final={final_loss}",
        );
    }

    // ---- Batched matmul + N-D transpose ----

    #[test]
    fn realize_batched_matmul_rank_3() {
        // [batch=2, m=2, k=3] @ [batch=2, k=3, n=2] → [2, 2, 2]
        // Batch 0:                        Batch 1:
        //   A0 = [[1,2,3],                  A1 = [[0,0,1],
        //         [4,5,6]]                         [1,1,1]]
        //   B0 = [[1,0],                   B1 = [[1,2],
        //         [0,1],                          [3,4],
        //         [1,1]]                          [5,6]]
        //
        //   A0 @ B0 = [[1+3, 2+3],         A1 @ B1 = [[5, 6],
        //              [4+6, 5+6]]                     [9, 12]]
        //           = [[4, 5], [10, 11]]
        let a = Tensor::from_f32(
            vec![
                // batch 0
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                // batch 1
                0.0, 0.0, 1.0, //
                1.0, 1.0, 1.0, //
            ],
            Shape::from_dims(&[2, 2, 3]),
            cpu_dev(),
        );
        let b = a.const_f32_like(
            vec![
                // batch 0
                1.0, 0.0, //
                0.0, 1.0, //
                1.0, 1.0, //
                // batch 1
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0, //
            ],
            Shape::from_dims(&[2, 3, 2]),
        );
        let c = a.matmul(&b);
        assert_eq!(c.shape().dims(), &[2, 2, 2]);
        let result = realize_f32(&c);
        assert_eq!(
            result.as_slice(),
            &[
                4.0, 5.0, 10.0, 11.0, // batch 0: A0 @ B0
                5.0, 6.0, 9.0, 12.0,  // batch 1: A1 @ B1
            ],
        );
    }

    #[test]
    fn realize_transpose_last_two_on_rank_3() {
        // [2, 3, 4] input. Each batch slice is transposed independently
        // to produce shape [2, 4, 3].
        let mut data = Vec::new();
        for b in 0..2 {
            for i in 0..3 {
                for j in 0..4 {
                    data.push((b * 100 + i * 10 + j) as f32);
                }
            }
        }
        let x = Tensor::from_f32(data, Shape::from_dims(&[2, 3, 4]), cpu_dev());
        let t = x.transpose();
        assert_eq!(t.shape().dims(), &[2, 4, 3]);
        let result = realize_f32(&t);
        // Batch 0: transpose of [[0,1,2,3],[10,11,12,13],[20,21,22,23]]
        //        = [[0,10,20], [1,11,21], [2,12,22], [3,13,23]]
        // Batch 1: +100 to every element
        assert_eq!(
            result.as_slice(),
            &[
                // batch 0
                0.0, 10.0, 20.0, //
                1.0, 11.0, 21.0, //
                2.0, 12.0, 22.0, //
                3.0, 13.0, 23.0, //
                // batch 1
                100.0, 110.0, 120.0, //
                101.0, 111.0, 121.0, //
                102.0, 112.0, 122.0, //
                103.0, 113.0, 123.0, //
            ],
        );
    }

    #[test]
    fn backward_of_batched_matmul_produces_correct_shapes() {
        // Just verify the backward graph builds + realizes with the
        // correct gradient shapes for a batched matmul. Full numerical
        // verification is covered by the rank-2 test.
        let a = Tensor::from_f32(vec![1.0; 12], Shape::from_dims(&[2, 2, 3]), cpu_dev());
        let b = a.const_f32_like(vec![1.0; 24], Shape::from_dims(&[2, 3, 4]));
        let y = a.matmul(&b);
        assert_eq!(y.shape().dims(), &[2, 2, 4]);
        let loss = y.sum_all();
        let grads = loss.backward();
        let g_a = realize_f32(&grads.get(&a).unwrap());
        let g_b = realize_f32(&grads.get(&b).unwrap());
        assert_eq!(g_a.shape().dims(), &[2, 2, 3]);
        assert_eq!(g_b.shape().dims(), &[2, 3, 4]);
    }

    // ---- Argmax / Argmin returning U32 ----

    #[test]
    fn realize_argmax_dim_simple() {
        // [[1, 5, 2], [4, 3, 6]]
        // argmax along dim 0: [col 0: row 1 (4>1), col 1: row 0 (5>3), col 2: row 1 (6>2)]
        //                   = [1, 0, 1]
        // argmax along dim 1: [row 0: col 1 (5), row 1: col 2 (6)]
        //                   = [1, 2]
        let x = Tensor::from_f32(
            vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let am0 = x.argmax_dim(0);
        let am1 = x.argmax_dim(1);
        assert_eq!(am0.dtype(), DType::U32);
        assert_eq!(am0.shape().dims(), &[3]);
        let r0 = realize(&am0).into_u32();
        let r1 = realize(&am1).into_u32();
        assert_eq!(r0.as_slice(), &[1, 0, 1]);
        assert_eq!(r1.as_slice(), &[1, 2]);
    }

    #[test]
    fn realize_argmin_dim_simple() {
        let x = Tensor::from_f32(
            vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        // argmin along dim 1: [row 0: col 0 (1), row 1: col 1 (3)]
        //                   = [0, 1]
        let result = realize(&x.argmin_dim(1)).into_u32();
        assert_eq!(result.as_slice(), &[0, 1]);
    }

    #[test]
    #[should_panic]
    fn backward_through_argmax_panics() {
        // Argmax produces a U32 index tensor which is non-differentiable.
        // Calling `backward()` on an argmax output must panic — either
        // from the `build_ones` seed attempting to make a U32 ones
        // tensor (first line of defense), or from the ArgMaxDim backward
        // match arm if execution ever reaches it. Either panic proves
        // the invariant.
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let y = x.argmax_dim(0);
        let _ = y.backward();
    }

    // ---- Batched-training regression: 2-layer MLP on a small fitting task ----

    #[test]
    fn train_2_layer_mlp_on_batched_regression() {
        // Batched training demonstration. 8 training examples, each with
        // 2 input features and 1 scalar target. The target is a simple
        // smooth nonlinear function of the inputs — well inside the
        // capacity of a small ReLU MLP, and deterministic so the test
        // converges reliably regardless of initialization luck.
        //
        // Architecture: input [8, 2] → hidden [8, 6] via relu →
        //               output [8, 1] linear. MSE loss.
        let x_data = vec![
            0.0_f32, 0.0, //
            0.0, 1.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            0.5, 0.5, //
            -0.5, 0.5, //
            0.5, -0.5, //
            -0.5, -0.5, //
        ];
        // target_i = 0.5*x0 + 0.25*x1 + 0.1 (linear — a 2-layer net
        // with enough capacity should fit it exactly modulo numerical
        // noise).
        let y_target: Vec<f32> = x_data
            .chunks(2)
            .map(|p| 0.5 * p[0] + 0.25 * p[1] + 0.1)
            .collect();

        // Bigger init so every ReLU unit has some chance of firing.
        let mut w1 = vec![
            0.8_f32, -0.6, 0.5, -0.4, 0.7, -0.5, //
            -0.3, 0.9, -0.7, 0.6, -0.4, 0.8, //
        ]; // [2, 6]
        let mut b1 = vec![0.1_f32; 6];
        let mut w2 = vec![0.3_f32, -0.2, 0.4, -0.3, 0.2, -0.1]; // [6, 1]
        let mut b2 = vec![0.0_f32; 1];

        let lr = 0.05_f32;
        let iters = 500;

        let build_loss =
            |w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]| -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
                let x = Tensor::from_f32(x_data.clone(), Shape::from_dims(&[8, 2]), cpu_dev());
                let tgt = x.const_f32_like(y_target.clone(), Shape::from_dims(&[8, 1]));
                let w1_t = x.const_f32_like(w1.to_vec(), Shape::from_dims(&[2, 6]));
                let b1_t = x.const_f32_like(b1.to_vec(), Shape::from_dims(&[6]));
                let w2_t = x.const_f32_like(w2.to_vec(), Shape::from_dims(&[6, 1]));
                let b2_t = x.const_f32_like(b2.to_vec(), Shape::from_dims(&[1]));

                let h = x.matmul(&w1_t).broadcast_add(&b1_t).relu();
                let y = h.matmul(&w2_t).broadcast_add(&b2_t);
                let diff = y.sub(&tgt);
                let loss = diff.sqr().sum_all();
                (loss, w1_t, b1_t, w2_t, b2_t)
            };

        let initial_loss = {
            let (loss, _, _, _, _) = build_loss(&w1, &b1, &w2, &b2);
            realize_f32(&loss).as_slice()[0]
        };

        for _ in 0..iters {
            let (loss, w1_t, b1_t, w2_t, b2_t) = build_loss(&w1, &b1, &w2, &b2);
            let grads = loss.backward();
            let g_w1 = realize_f32(&grads.get(&w1_t).unwrap());
            let g_b1 = realize_f32(&grads.get(&b1_t).unwrap());
            let g_w2 = realize_f32(&grads.get(&w2_t).unwrap());
            let g_b2 = realize_f32(&grads.get(&b2_t).unwrap());
            for (p, &g) in w1.iter_mut().zip(g_w1.as_slice()) {
                *p -= lr * g;
            }
            for (p, &g) in b1.iter_mut().zip(g_b1.as_slice()) {
                *p -= lr * g;
            }
            for (p, &g) in w2.iter_mut().zip(g_w2.as_slice()) {
                *p -= lr * g;
            }
            for (p, &g) in b2.iter_mut().zip(g_b2.as_slice()) {
                *p -= lr * g;
            }
        }

        let final_loss = {
            let (loss, _, _, _, _) = build_loss(&w1, &b1, &w2, &b2);
            realize_f32(&loss).as_slice()[0]
        };

        assert!(
            final_loss < initial_loss * 0.1,
            "batched regression should converge: initial={initial_loss}, final={final_loss}",
        );
    }

    // ---- Single-head attention block, forward + backward ----

    #[test]
    fn single_head_attention_forward_backward() {
        // Build a single-head attention block:
        //
        //   scores = Q @ K^T / sqrt(d_k)
        //   attn   = softmax_last_dim(scores)
        //   out    = attn @ V
        //
        // Q, K, V all have shape [seq_len, d_k] (batch=1, head=1).
        // Every op in that chain exists in the catalog; the whole
        // graph builds + backwards end-to-end. This test is the first
        // that exercises a fragment resembling a real transformer.
        let seq_len = 4;
        let d_k = 3;

        // Small deterministic inputs so we can spot-check the output.
        let q_data: Vec<f32> = (0..seq_len * d_k).map(|i| (i as f32) * 0.1).collect();
        let k_data: Vec<f32> = (0..seq_len * d_k)
            .map(|i| ((i * 7) % 5) as f32 * 0.1)
            .collect();
        let v_data: Vec<f32> = (0..seq_len * d_k)
            .map(|i| ((i * 3) % 7) as f32 * 0.1)
            .collect();

        let q = Tensor::from_f32(q_data, Shape::from_dims(&[seq_len, d_k]), cpu_dev());
        let k = q.const_f32_like(k_data, Shape::from_dims(&[seq_len, d_k]));
        let v = q.const_f32_like(v_data, Shape::from_dims(&[seq_len, d_k]));

        // K^T
        let k_t = k.transpose();
        assert_eq!(k_t.shape().dims(), &[d_k, seq_len]);
        // Scores: Q @ K^T
        let scores = q.matmul(&k_t);
        assert_eq!(scores.shape().dims(), &[seq_len, seq_len]);
        // Scale by 1/sqrt(d_k)
        let scale = 1.0_f64 / (d_k as f64).sqrt();
        let scaled = scores.mul_scalar(scale);
        // Softmax along the last dim (attention weights)
        let attn = scaled.softmax_last_dim();
        // Every row should sum to ~1 — we'll verify during realization.
        // Output: attn @ V
        let out = attn.matmul(&v);
        assert_eq!(out.shape().dims(), &[seq_len, d_k]);

        // Realize forward.
        let out_val = realize_f32(&out);
        assert_eq!(out_val.shape().dims(), &[seq_len, d_k]);
        for &v in out_val.as_slice() {
            assert!(v.is_finite(), "attention output should be finite: got {v}");
        }
        // Verify attention weights sum to 1 per row.
        let attn_val = realize_f32(&attn);
        for row in 0..seq_len {
            let row_sum: f32 = attn_val.as_slice()[row * seq_len..(row + 1) * seq_len]
                .iter()
                .sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "attention row {row} sum = {row_sum}, expected 1",
            );
        }

        // Backward pass: treat sum_all(out) as the loss.
        let loss = out.sum_all();
        let grads = loss.backward();
        let g_q = realize_f32(&grads.get(&q).unwrap());
        let g_k = realize_f32(&grads.get(&k).unwrap());
        let g_v = realize_f32(&grads.get(&v).unwrap());
        assert_eq!(g_q.shape().dims(), &[seq_len, d_k]);
        assert_eq!(g_k.shape().dims(), &[seq_len, d_k]);
        assert_eq!(g_v.shape().dims(), &[seq_len, d_k]);
        // All gradients should be finite — if softmax or LN backward
        // produces NaN/Inf, this catches it.
        for (name, g) in [("g_q", &g_q), ("g_k", &g_k), ("g_v", &g_v)] {
            for &val in g.as_slice() {
                assert!(
                    val.is_finite(),
                    "{name} gradient should be finite: got {val}",
                );
            }
        }
    }

    // ---- int↔float casts ----

    #[test]
    fn cast_u32_to_f32_and_back() {
        let x = Tensor::from_u32(vec![0, 1, 42, 1000], Shape::from_dims(&[4]), cpu_dev());
        let as_f32 = x.cast(DType::F32);
        assert_eq!(as_f32.dtype(), DType::F32);
        assert_eq!(
            realize_f32(&as_f32).as_slice(),
            &[0.0, 1.0, 42.0, 1000.0],
        );
        // Round-trip back to u32.
        let back = as_f32.cast(DType::U32);
        assert_eq!(back.dtype(), DType::U32);
        let result = realize(&back).into_u32();
        assert_eq!(result.as_slice(), &[0, 1, 42, 1000]);
    }

    #[test]
    fn classifier_accuracy_via_argmax_and_cast() {
        // End-to-end "how do I compute accuracy" test: run argmax_dim on
        // classifier logits to get predicted class indices, compare to
        // ground-truth labels via argmax of the target distribution (or
        // direct label tensor), cast to float, mean. This exercises the
        // full argmax → cast → reduction chain.
        //
        // Logits for 3 examples × 4 classes:
        let logits = Tensor::from_f32(
            vec![
                0.1, 0.2, 0.5, 0.2, // example 0: argmax = 2
                0.8, 0.1, 0.05, 0.05, // example 1: argmax = 0
                0.1, 0.4, 0.3, 0.2, // example 2: argmax = 1
            ],
            Shape::from_dims(&[3, 4]),
            cpu_dev(),
        );
        let targets = logits.const_u32_like(vec![2, 0, 3], Shape::from_dims(&[3]));
        let predictions = logits.argmax_dim(1);
        // Compare predictions (u32) with targets (u32) by casting both to f32
        // and using equality-via-sub/abs-test. We don't have an Eq op, but
        // (pred == target) iff (pred_f - target_f == 0).
        let pred_f = predictions.cast(DType::F32);
        let target_f = targets.cast(DType::F32);
        let diff = pred_f.sub(&target_f);
        // `1 - step(|diff|) → 1 at equal, 0 at unequal` using our step
        // convention (step(0)=0). abs via sqr + sqrt or just compare
        // against zero. Simplest: compute (diff == 0) as (1 - step(|diff|))
        // — but we need abs. Use sqr: diff² is 0 iff equal, >0 otherwise.
        let diff_sq = diff.sqr();
        // (1 - step(diff_sq)) = 1 at equal positions (since step(0) = 0),
        //                      0 everywhere else.
        let step_flags = diff_sq.step();
        let correct = step_flags.mul_scalar(-1.0).add_scalar(1.0);
        let accuracy = correct.mean_all();
        let acc_val = realize_f32(&accuracy).as_slice()[0];
        // pred = [2, 0, 1], target = [2, 0, 3]. Two correct out of three
        // → accuracy ≈ 0.667.
        assert!(
            (acc_val - 2.0 / 3.0).abs() < 1e-5,
            "accuracy should be 2/3, got {acc_val}",
        );
    }

    // ---- Multi-head attention via batched matmul ----

    #[test]
    fn multi_head_attention_forward_backward() {
        // Multi-head self-attention, expressed via rank-3 tensors with
        // heads as the leading batch dim. Single sample (batch=1 elided).
        //
        // Config: seq_len = 3, num_heads = 2, d_head = 2, d_model = 4.
        //
        // Q, K, V start as [seq, d_model], get reshaped to
        // [seq, heads, d_head], transposed to [heads, seq, d_head],
        // then operated on with batched matmul across the head dim.
        let seq_len = 3;
        let num_heads = 2;
        let d_head = 2;
        let d_model = num_heads * d_head; // = 4

        let q_flat: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32) * 0.1 - 0.4)
            .collect();
        let k_flat: Vec<f32> = (0..seq_len * d_model)
            .map(|i| ((i * 3 + 1) as f32) * 0.07 - 0.3)
            .collect();
        let v_flat: Vec<f32> = (0..seq_len * d_model)
            .map(|i| ((i * 5 + 2) as f32) * 0.05 - 0.2)
            .collect();

        let q = Tensor::from_f32(q_flat, Shape::from_dims(&[seq_len, d_model]), cpu_dev());
        let k = q.const_f32_like(k_flat, Shape::from_dims(&[seq_len, d_model]));
        let v = q.const_f32_like(v_flat, Shape::from_dims(&[seq_len, d_model]));

        // Reshape to [seq, heads, d_head] then transpose to [heads, seq, d_head].
        //
        // Note: this reshape assumes the d_model dim is laid out as
        // head-major (i.e. [h0d0, h0d1, h1d0, h1d1]). The transpose
        // swaps the LAST TWO dims only, which gives us [seq, d_head,
        // heads] — that's wrong. To get [heads, seq, d_head] we'd need
        // a full permutation (not supported yet), OR reshape first to
        // [seq, heads, d_head], then transpose to [heads, seq, d_head]
        // via a rank-3 transpose of the first two dims — also not
        // supported. The MVP transpose only swaps the last two dims.
        //
        // Workaround: build Q/K/V directly in [heads, seq, d_head] form
        // by precomputing the reshape. Since we're building f32 consts
        // from scratch, we just rearrange the input vectors.
        let q_mh = rearrange_to_heads(&q, seq_len, num_heads, d_head);
        let k_mh = rearrange_to_heads(&k, seq_len, num_heads, d_head);
        let v_mh = rearrange_to_heads(&v, seq_len, num_heads, d_head);
        assert_eq!(q_mh.shape().dims(), &[num_heads, seq_len, d_head]);

        // scores = Q @ K^T / sqrt(d_head). Batched matmul over heads.
        let k_mh_t = k_mh.transpose(); // [heads, d_head, seq]
        assert_eq!(k_mh_t.shape().dims(), &[num_heads, d_head, seq_len]);
        let scores = q_mh.matmul(&k_mh_t); // [heads, seq, seq]
        assert_eq!(scores.shape().dims(), &[num_heads, seq_len, seq_len]);
        let scale = 1.0_f64 / (d_head as f64).sqrt();
        let attn = scores.mul_scalar(scale).softmax_last_dim();
        // Output: attn @ V → [heads, seq, d_head]
        let out_mh = attn.matmul(&v_mh);
        assert_eq!(out_mh.shape().dims(), &[num_heads, seq_len, d_head]);

        // Forward realization — verify every attention row sums to ~1.
        let attn_val = realize_f32(&attn);
        for h in 0..num_heads {
            for s in 0..seq_len {
                let start = h * seq_len * seq_len + s * seq_len;
                let row_sum: f32 = attn_val.as_slice()[start..start + seq_len].iter().sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-5,
                    "attn[{h}][{s}] row sum = {row_sum}",
                );
            }
        }
        let out_val = realize_f32(&out_mh);
        assert!(out_val.as_slice().iter().all(|v| v.is_finite()));

        // Backward.
        let loss = out_mh.sum_all();
        let grads = loss.backward();
        // Verify gradient shapes for the rearranged (multi-head form) inputs.
        // We didn't store gradients for the original q/k/v because the rearrange
        // was done in plain Rust — q_mh / k_mh / v_mh are the graph-level leaves.
        let g_q = realize_f32(&grads.get(&q_mh).unwrap());
        let g_k = realize_f32(&grads.get(&k_mh).unwrap());
        let g_v = realize_f32(&grads.get(&v_mh).unwrap());
        assert_eq!(g_q.shape().dims(), &[num_heads, seq_len, d_head]);
        assert_eq!(g_k.shape().dims(), &[num_heads, seq_len, d_head]);
        assert_eq!(g_v.shape().dims(), &[num_heads, seq_len, d_head]);
        for g in [&g_q, &g_k, &g_v] {
            for &v in g.as_slice() {
                assert!(v.is_finite());
            }
        }
    }

    /// Rearrange a `[seq, d_model]` tensor into `[heads, seq, d_head]` by
    /// shuffling in plain Rust before constructing the graph tensor. A
    /// full graph-level implementation would need a rank-3 transpose of
    /// non-last dims (not in the MVP transpose).
    fn rearrange_to_heads(
        t: &Tensor,
        seq_len: usize,
        num_heads: usize,
        d_head: usize,
    ) -> Tensor {
        let source = realize_f32(t).into_vec();
        let d_model = num_heads * d_head;
        let mut out = vec![0.0_f32; num_heads * seq_len * d_head];
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..d_head {
                    let src_idx = s * d_model + h * d_head + d;
                    let dst_idx = h * seq_len * d_head + s * d_head + d;
                    out[dst_idx] = source[src_idx];
                }
            }
        }
        t.const_f32_like(out, Shape::from_dims(&[num_heads, seq_len, d_head]))
    }

    // ---- Higher-order chain: d²(x³)/dx² = 6x ----

    #[test]
    fn higher_order_chain_x_cubed() {
        // f(x) = sum(x³)
        // f'(x) = 3x²  (per element)
        // f''(x) = 6x  (per element, via another sum→backward pass)
        //
        // This composes differently from `higher_order_gradient_of_sqr`:
        // it uses powi(3) which internally builds a chain of Mul nodes
        // (via the powi backward rule), stressing the backward pass on
        // a deeper forward graph.
        let x_vals = vec![1.0_f32, 2.0, 3.0];
        let x = Tensor::from_f32(x_vals.clone(), Shape::from_dims(&[3]), cpu_dev());
        let y = x.powi(3).sum_all();

        // First backward: df/dx should be [3, 12, 27].
        let grads = y.backward();
        let g_x = grads.get(&x).unwrap();
        let g_x_val = realize_f32(&g_x);
        let expected_first: Vec<f32> = x_vals.iter().map(|&v| 3.0 * v * v).collect();
        approx_vec(g_x_val.as_slice(), &expected_first, 1e-5);

        // Second backward: differentiate sum(g_x) with respect to x.
        // sum(3x²) → d/dx = 6x → [6, 12, 18].
        let g_scalar = g_x.sum_all();
        let second = g_scalar.backward();
        let gg_x = realize_f32(&second.get(&x).unwrap());
        let expected_second: Vec<f32> = x_vals.iter().map(|&v| 6.0 * v).collect();
        approx_vec(gg_x.as_slice(), &expected_second, 1e-4);
    }

    // ---- Permute ----

    #[test]
    fn realize_permute_rank_3_reorder() {
        // [2, 3, 4] with axes [2, 0, 1] → [4, 2, 3]
        let x = Tensor::from_f32(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            Shape::from_dims(&[2, 3, 4]),
            cpu_dev(),
        );
        let y = x.permute(&[2, 0, 1]);
        assert_eq!(y.shape().dims(), &[4, 2, 3]);
        // Hand-check: y[k, i, j] should equal x[i, j, k].
        // For i=0, j=0: x[0,0,k] for k in 0..4 = [0,1,2,3]
        //   so y[0,0,0]=0, y[1,0,0]=1, y[2,0,0]=2, y[3,0,0]=3
        // For i=0, j=1: x[0,1,k] for k in 0..4 = [4,5,6,7]
        //   so y[0,0,1]=4, y[1,0,1]=5, y[2,0,1]=6, y[3,0,1]=7
        let result = realize_f32(&y);
        assert_eq!(result.as_slice()[0], 0.0); // y[0,0,0] = x[0,0,0] = 0
        assert_eq!(result.as_slice()[1], 4.0); // y[0,0,1] = x[0,1,0] = 4
        assert_eq!(result.as_slice()[2], 8.0); // y[0,0,2] = x[0,2,0] = 8
        // Row y[0, 1, :] = x[1, :, 0] = [12, 16, 20]
        assert_eq!(result.as_slice()[3], 12.0);
        assert_eq!(result.as_slice()[4], 16.0);
        assert_eq!(result.as_slice()[5], 20.0);
    }

    #[test]
    fn permute_is_self_inverse_under_double_apply() {
        // Applying a permutation and then its inverse recovers x.
        let x = Tensor::from_f32(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            Shape::from_dims(&[2, 3, 4]),
            cpu_dev(),
        );
        let axes = [2, 0, 1];
        let inv = [1, 2, 0]; // inverse of [2, 0, 1]
        let y = x.permute(&axes).permute(&inv);
        assert_eq!(y.shape().dims(), &[2, 3, 4]);
        let result = realize_f32(&y);
        let expected: Vec<f32> = (0..24).map(|i| i as f32).collect();
        assert_eq!(result.as_slice(), expected.as_slice());
    }

    #[test]
    fn backward_of_permute_uses_inverse_permutation() {
        // y = permute(x, [2, 0, 1]), y has shape [4, 2, 3].
        // With upstream ones of shape [4, 2, 3], grad_x should be ones
        // of shape [2, 3, 4].
        let x = Tensor::from_f32(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            Shape::from_dims(&[2, 3, 4]),
            cpu_dev(),
        );
        let y = x.permute(&[2, 0, 1]);
        let loss = y.sum_all();
        let grads = loss.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        assert_eq!(g_x.shape().dims(), &[2, 3, 4]);
        for &v in g_x.as_slice() {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    // ---- RmsNorm ----

    #[test]
    fn realize_rms_norm_last_dim_normalizes_rows() {
        // For each row, RmsNorm divides by sqrt(mean(x²) + eps). For
        // row [3, 4] the RMS is sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355.
        // Result ≈ [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314].
        let x = Tensor::from_f32(
            vec![3.0, 4.0, 5.0, 12.0],
            Shape::from_dims(&[2, 2]),
            cpu_dev(),
        );
        let y = x.rms_norm_last_dim(1e-6);
        let result = realize_f32(&y);
        assert_eq!(result.shape().dims(), &[2, 2]);
        // Row 0 RMS: sqrt((9 + 16)/2) = sqrt(12.5) ≈ 3.5355339
        let rms0 = (12.5_f32).sqrt();
        assert!((result.as_slice()[0] - 3.0 / rms0).abs() < 1e-5);
        assert!((result.as_slice()[1] - 4.0 / rms0).abs() < 1e-5);
        // Row 1 RMS: sqrt((25 + 144)/2) = sqrt(84.5)
        let rms1 = (84.5_f32).sqrt();
        assert!((result.as_slice()[2] - 5.0 / rms1).abs() < 1e-5);
        assert!((result.as_slice()[3] - 12.0 / rms1).abs() < 1e-5);
    }

    #[test]
    fn backward_of_rms_norm_finite_difference_check() {
        // Sanity check: the composition of primitives that makes up
        // rms_norm_last_dim has a working backward. Compare against a
        // finite-difference estimate.
        let x_vals = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w_vals = vec![0.5_f32, -0.3, 0.8, 0.1];
        let x = Tensor::from_f32(x_vals.clone(), Shape::from_dims(&[4]), cpu_dev());
        let w = x.const_f32_like(w_vals.clone(), Shape::from_dims(&[4]));
        let eps = 1e-5_f64;
        let loss = x.rms_norm_last_dim(eps).mul(&w).sum_all();
        let grads = loss.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());

        let fd_eps = 1e-3_f32;
        let base = rms_norm_loss(&x_vals, &w_vals, eps as f32);
        for i in 0..x_vals.len() {
            let mut perturbed = x_vals.clone();
            perturbed[i] += fd_eps;
            let new = rms_norm_loss(&perturbed, &w_vals, eps as f32);
            let fd = (new - base) / fd_eps;
            let analytic = g_x.as_slice()[i];
            assert!(
                (fd - analytic).abs() < 5e-2,
                "RmsNorm backward disagrees at {i}: fd={fd}, analytic={analytic}",
            );
        }
    }

    fn rms_norm_loss(x: &[f32], w: &[f32], eps: f32) -> f32 {
        let n = x.len() as f32;
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
        let rms = (mean_sq + eps).sqrt();
        x.iter().zip(w).map(|(&xv, &wv)| (xv / rms) * wv).sum()
    }

    // ---- Llama-style decoder block integration test ----
    //
    // A complete transformer decoder block with the LLaMA architecture:
    //
    //   h1 = x + Attention(RmsNorm(x))
    //   h2 = h1 + FFN(RmsNorm(h1))
    //
    // where:
    //   Attention(x) = out_proj( reshape( split_heads(x W_qkv) ) )
    //                             (standard multi-head)
    //   FFN(x)       = down_proj( silu(x W_gate) * (x W_up) )
    //                             (SwiGLU)
    //
    // This is every architectural primitive needed for a single LLaMA
    // transformer layer except rotary position embeddings (RoPE), which
    // can be added as learned positional add-ons or a dedicated op
    // later. Every op in this graph has a forward AND a backward, so
    // this test also runs the backward pass and verifies parameter
    // gradients are finite and have the right shapes.

    #[test]
    fn llama_style_decoder_block_forward_backward() {
        // Dimensions: 1 batch, seq=4, d_model=6, num_heads=2, d_head=3
        // (so 2*3 = 6 = d_model). FFN inner = 12 (2x d_model).
        let batch = 1;
        let seq = 4;
        let num_heads = 2;
        let d_head = 3;
        let d_model = num_heads * d_head; // 6
        let ffn_inner = 12;

        // Fake "token embeddings" — the input to the decoder block.
        let x_data: Vec<f32> = (0..batch * seq * d_model)
            .map(|i| (i as f32) * 0.05 - 0.6)
            .collect();
        let x = Tensor::from_f32(x_data, Shape::from_dims(&[batch, seq, d_model]), cpu_dev());

        // Attention weights: W_q, W_k, W_v, W_o (four [d_model, d_model] projections).
        let rand = |seed: u32| {
            // Deterministic "random" weights via a linear congruential sequence.
            // Not great but keeps the test reproducible without a dep.
            let mut s = seed;
            move |n: usize| {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    s = s.wrapping_mul(1103515245).wrapping_add(12345);
                    let f = ((s >> 16) as u16 as f32 / 65535.0) - 0.5; // in [-0.5, 0.5)
                    v.push(f * 0.3);
                }
                v
            }
        };
        let mut rng = rand(42);
        let w_q = x.const_f32_like(rng(d_model * d_model), Shape::from_dims(&[d_model, d_model]));
        let w_k = x.const_f32_like(rng(d_model * d_model), Shape::from_dims(&[d_model, d_model]));
        let w_v = x.const_f32_like(rng(d_model * d_model), Shape::from_dims(&[d_model, d_model]));
        let w_o = x.const_f32_like(rng(d_model * d_model), Shape::from_dims(&[d_model, d_model]));
        // FFN weights (SwiGLU).
        let w_gate = x.const_f32_like(
            rng(d_model * ffn_inner),
            Shape::from_dims(&[d_model, ffn_inner]),
        );
        let w_up = x.const_f32_like(
            rng(d_model * ffn_inner),
            Shape::from_dims(&[d_model, ffn_inner]),
        );
        let w_down = x.const_f32_like(
            rng(ffn_inner * d_model),
            Shape::from_dims(&[ffn_inner, d_model]),
        );
        let eps = 1e-5_f64;

        // --- Attention sub-block ---
        let x_norm = x.rms_norm_last_dim(eps);
        // Project to Q, K, V using auto-broadcasting matmul.
        // [batch, seq, d_model] @ [d_model, d_model] → [batch, seq, d_model]
        let q = x_norm.matmul(&w_q);
        let k = x_norm.matmul(&w_k);
        let v = x_norm.matmul(&w_v);
        assert_eq!(q.shape().dims(), &[batch, seq, d_model]);

        // Split heads: [batch, seq, d_model] → [batch, seq, num_heads, d_head]
        // via reshape, then permute to [batch, num_heads, seq, d_head].
        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, num_heads, d_head]))
            .permute(&[0, 2, 1, 3]);
        let k_h = k
            .reshape(Shape::from_dims(&[batch, seq, num_heads, d_head]))
            .permute(&[0, 2, 1, 3]);
        let v_h = v
            .reshape(Shape::from_dims(&[batch, seq, num_heads, d_head]))
            .permute(&[0, 2, 1, 3]);
        assert_eq!(q_h.shape().dims(), &[batch, num_heads, seq, d_head]);

        // Attention scores: [batch, num_heads, seq, d_head] @ [batch, num_heads, d_head, seq]
        // Need to transpose last two dims of k_h for the score matmul.
        let k_h_t = k_h.transpose(); // [batch, num_heads, d_head, seq]
        let scores = q_h.matmul(&k_h_t); // [batch, num_heads, seq, seq]
        let scale = 1.0_f64 / (d_head as f64).sqrt();
        let scaled = scores.mul_scalar(scale);
        let attn = scaled.softmax_last_dim();
        // Attention-weighted values: [batch, num_heads, seq, seq] @ [batch, num_heads, seq, d_head]
        let attn_v = attn.matmul(&v_h); // [batch, num_heads, seq, d_head]

        // Merge heads back: permute [batch, num_heads, seq, d_head] →
        // [batch, seq, num_heads, d_head] → reshape [batch, seq, d_model]
        let merged = attn_v
            .permute(&[0, 2, 1, 3])
            .reshape(Shape::from_dims(&[batch, seq, d_model]));
        // Output projection.
        let attn_out = merged.matmul(&w_o);
        // Residual.
        let h1 = x.add(&attn_out);
        assert_eq!(h1.shape().dims(), &[batch, seq, d_model]);

        // --- FFN sub-block (SwiGLU) ---
        let h1_norm = h1.rms_norm_last_dim(eps);
        let gate = h1_norm.matmul(&w_gate); // [batch, seq, ffn_inner]
        let up = h1_norm.matmul(&w_up);
        // SwiGLU: silu(gate) * up
        let swiglu = gate.silu().mul(&up);
        // Down projection.
        let ffn_out = swiglu.matmul(&w_down);
        // Residual.
        let h2 = h1.add(&ffn_out);
        assert_eq!(h2.shape().dims(), &[batch, seq, d_model]);

        // Forward realization.
        let h2_val = realize_f32(&h2);
        for &v in h2_val.as_slice() {
            assert!(v.is_finite(), "decoder block output should be finite, got {v}");
        }

        // Backward pass on sum(h2) as a scalar loss. Every learnable
        // parameter should receive a finite gradient with the correct
        // shape.
        let loss = h2.sum_all();
        let grads = loss.backward();

        for (name, param, expected_shape) in [
            ("W_q", &w_q, vec![d_model, d_model]),
            ("W_k", &w_k, vec![d_model, d_model]),
            ("W_v", &w_v, vec![d_model, d_model]),
            ("W_o", &w_o, vec![d_model, d_model]),
            ("W_gate", &w_gate, vec![d_model, ffn_inner]),
            ("W_up", &w_up, vec![d_model, ffn_inner]),
            ("W_down", &w_down, vec![ffn_inner, d_model]),
        ] {
            let g = grads
                .get(param)
                .unwrap_or_else(|| panic!("parameter {name} must have a gradient"));
            let g_val = realize_f32(&g);
            assert_eq!(
                g_val.shape().dims(),
                expected_shape.as_slice(),
                "{name} gradient shape mismatch",
            );
            // Also check finiteness.
            for &v in g_val.as_slice() {
                assert!(
                    v.is_finite(),
                    "{name} gradient contains non-finite value {v}",
                );
            }
        }
    }

    // ---- RoPE tests ----

    #[test]
    fn rope_identity_at_position_zero() {
        // At position 0, every angle θ = 0 * freq = 0, so cos = 1 and
        // sin = 0. That means the first row of rope(x, _, 0) equals
        // the first row of x exactly. (Later rows pick up non-zero
        // angles and are not identity.)
        let x = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::from_dims(&[2, 4]), // seq=2, d=4
            cpu_dev(),
        );
        let y = x.rope(10000.0, 0);
        assert_eq!(y.shape().dims(), &[2, 4]);
        let result = realize_f32(&y);
        // Row 0 at position 0 should be unchanged.
        approx_vec(&result.as_slice()[..4], &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn rope_preserves_norm_per_position() {
        // RoPE is a rotation, which preserves L2 norm. For any random
        // input x, ||rope(x)|| = ||x|| per row.
        let x = Tensor::from_f32(
            vec![
                0.3, -0.7, 1.2, 0.5, //
                -0.1, 0.8, -0.4, 0.6, //
                0.9, -0.2, 0.1, -0.5, //
            ],
            Shape::from_dims(&[3, 4]),
            cpu_dev(),
        );
        let y = x.rope(10000.0, 0);
        let x_val = realize_f32(&x);
        let y_val = realize_f32(&y);
        let seq = 3;
        let d = 4;
        for p in 0..seq {
            let x_norm_sq: f32 = x_val.as_slice()[p * d..(p + 1) * d]
                .iter()
                .map(|v| v * v)
                .sum();
            let y_norm_sq: f32 = y_val.as_slice()[p * d..(p + 1) * d]
                .iter()
                .map(|v| v * v)
                .sum();
            assert!(
                (x_norm_sq - y_norm_sq).abs() < 1e-5,
                "rope should preserve norm at pos {p}: {x_norm_sq} vs {y_norm_sq}",
            );
        }
    }

    #[test]
    fn backward_of_rope_is_finite_and_correct_shape() {
        let x = Tensor::from_f32(
            vec![
                0.3, -0.7, 1.2, 0.5, //
                -0.1, 0.8, -0.4, 0.6, //
            ],
            Shape::from_dims(&[2, 4]),
            cpu_dev(),
        );
        let y = x.rope(10000.0, 0);
        let loss = y.sum_all();
        let grads = loss.backward();
        let g_x = realize_f32(&grads.get(&x).unwrap());
        assert_eq!(g_x.shape().dims(), &[2, 4]);
        for &v in g_x.as_slice() {
            assert!(v.is_finite(), "rope gradient should be finite, got {v}");
        }
    }

    // ---- Stacked decoder blocks (3-layer LLaMA-style transformer) ----

    /// Parameters of one decoder block, laid out as plain `Vec<f32>`
    /// rather than graph tensors. Tests build the graph tensors at the
    /// start of each forward pass from these vectors.
    struct BlockParams {
        w_q:    Vec<f32>,
        w_k:    Vec<f32>,
        w_v:    Vec<f32>,
        w_o:    Vec<f32>,
        w_gate: Vec<f32>,
        w_up:   Vec<f32>,
        w_down: Vec<f32>,
    }

    /// Fixed-config decoder block — writes the full LLaMA-style forward
    /// pass (RmsNorm → QKV → RoPE → attention → out proj → residual →
    /// RmsNorm → SwiGLU FFN → residual) and returns the new hidden
    /// state plus handles to every weight tensor it created (so tests
    /// can inspect parameter gradients afterward).
    struct BlockHandles {
        h_out:  Tensor,
        w_q:    Tensor,
        w_k:    Tensor,
        w_v:    Tensor,
        w_o:    Tensor,
        w_gate: Tensor,
        w_up:   Tensor,
        w_down: Tensor,
    }

    fn apply_decoder_block(
        x: &Tensor,
        params: &BlockParams,
        num_heads: usize,
        d_head: usize,
        ffn_inner: usize,
        eps: f64,
    ) -> BlockHandles {
        let dims = x.shape().dims().to_vec();
        let batch = dims[0];
        let seq = dims[1];
        let d_model = dims[2];
        assert_eq!(num_heads * d_head, d_model);

        // Create weight tensors on the same graph as x.
        let w_q = x.const_f32_like(params.w_q.clone(), Shape::from_dims(&[d_model, d_model]));
        let w_k = x.const_f32_like(params.w_k.clone(), Shape::from_dims(&[d_model, d_model]));
        let w_v = x.const_f32_like(params.w_v.clone(), Shape::from_dims(&[d_model, d_model]));
        let w_o = x.const_f32_like(params.w_o.clone(), Shape::from_dims(&[d_model, d_model]));
        let w_gate = x.const_f32_like(
            params.w_gate.clone(),
            Shape::from_dims(&[d_model, ffn_inner]),
        );
        let w_up = x.const_f32_like(
            params.w_up.clone(),
            Shape::from_dims(&[d_model, ffn_inner]),
        );
        let w_down = x.const_f32_like(
            params.w_down.clone(),
            Shape::from_dims(&[ffn_inner, d_model]),
        );

        // --- Self-attention sub-block ---
        let x_norm = x.rms_norm_last_dim(eps);
        let q = x_norm.matmul(&w_q); // [batch, seq, d_model]
        let k = x_norm.matmul(&w_k);
        let v = x_norm.matmul(&w_v);

        // Split heads: [batch, seq, d_model] →
        // [batch, seq, num_heads, d_head] → [batch, num_heads, seq, d_head].
        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, num_heads, d_head]))
            .permute(&[0, 2, 1, 3]);
        let k_h = k
            .reshape(Shape::from_dims(&[batch, seq, num_heads, d_head]))
            .permute(&[0, 2, 1, 3]);
        let v_h = v
            .reshape(Shape::from_dims(&[batch, seq, num_heads, d_head]))
            .permute(&[0, 2, 1, 3]);

        // Apply RoPE to Q and K (not V). start_pos = 0 for first-time
        // training; a real inference session would thread the cached
        // position through here.
        let q_r = q_h.rope(10000.0, 0);
        let k_r = k_h.rope(10000.0, 0);

        // Scaled dot-product attention.
        let k_t = k_r.transpose(); // [batch, num_heads, d_head, seq]
        let scores = q_r.matmul(&k_t); // [batch, num_heads, seq, seq]
        let scale = 1.0_f64 / (d_head as f64).sqrt();
        let attn = scores.mul_scalar(scale).softmax_last_dim();
        let attn_v = attn.matmul(&v_h); // [batch, num_heads, seq, d_head]

        // Merge heads: [batch, num_heads, seq, d_head] →
        // [batch, seq, num_heads, d_head] → reshape [batch, seq, d_model].
        let merged = attn_v
            .permute(&[0, 2, 1, 3])
            .reshape(Shape::from_dims(&[batch, seq, d_model]));
        let attn_out = merged.matmul(&w_o);
        let h1 = x.add(&attn_out);

        // --- FFN sub-block (SwiGLU) ---
        let h1_norm = h1.rms_norm_last_dim(eps);
        let gate = h1_norm.matmul(&w_gate);
        let up = h1_norm.matmul(&w_up);
        let swiglu = gate.silu().mul(&up);
        let ffn_out = swiglu.matmul(&w_down);
        let h2 = h1.add(&ffn_out);

        BlockHandles {
            h_out: h2,
            w_q,
            w_k,
            w_v,
            w_o,
            w_gate,
            w_up,
            w_down,
        }
    }

    /// Generate deterministic "random" weights for N decoder blocks
    /// with the given dimensions.
    fn make_stacked_block_params(
        num_layers: usize,
        d_model: usize,
        ffn_inner: usize,
    ) -> Vec<BlockParams> {
        let mut s = 1234_u32;
        let mut next = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.3
        };
        let mut vec_of = |n: usize| -> Vec<f32> { (0..n).map(|_| next()).collect() };
        (0..num_layers)
            .map(|_| BlockParams {
                w_q:    vec_of(d_model * d_model),
                w_k:    vec_of(d_model * d_model),
                w_v:    vec_of(d_model * d_model),
                w_o:    vec_of(d_model * d_model),
                w_gate: vec_of(d_model * ffn_inner),
                w_up:   vec_of(d_model * ffn_inner),
                w_down: vec_of(ffn_inner * d_model),
            })
            .collect()
    }

    #[test]
    fn stacked_3_layer_llama_style_forward_backward() {
        // A 3-layer LLaMA-style transformer. This is a complete multi-
        // layer model (minus embedding/output projection) built by
        // stacking the decoder block helper three times. Every layer
        // gets its own set of weights; the output of each layer feeds
        // into the next. At the end we take sum_all as a scalar "loss"
        // and backprop through every layer.
        //
        // Architectural coverage for each layer:
        //   - RmsNorm → Q/K/V projections → head reshape/permute
        //   - RoPE on Q and K
        //   - Scaled dot-product attention (softmax)
        //   - Output projection and residual connection
        //   - Second RmsNorm → SwiGLU FFN → residual
        //
        // Every differentiable parameter in all 3 layers must receive
        // a finite gradient with the correct shape.
        let batch = 1;
        let seq = 4;
        let num_heads = 2;
        let d_head = 4; // even — required by RoPE
        let d_model = num_heads * d_head; // 8
        let ffn_inner = 16;
        let num_layers = 3;

        let params = make_stacked_block_params(num_layers, d_model, ffn_inner);

        let x_data: Vec<f32> = (0..batch * seq * d_model)
            .map(|i| (i as f32) * 0.05 - 0.6)
            .collect();
        let x = Tensor::from_f32(x_data, Shape::from_dims(&[batch, seq, d_model]), cpu_dev());

        // Chain the blocks.
        let mut current = x.clone();
        let mut all_handles: Vec<BlockHandles> = Vec::new();
        for p in &params {
            let handles =
                apply_decoder_block(&current, p, num_heads, d_head, ffn_inner, 1e-5);
            current = handles.h_out.clone();
            all_handles.push(handles);
        }

        assert_eq!(current.shape().dims(), &[batch, seq, d_model]);

        // Forward: verify the final output is finite.
        let y_val = realize_f32(&current);
        for &v in y_val.as_slice() {
            assert!(v.is_finite(), "stacked forward output non-finite: {v}");
        }

        // Backward on sum_all as scalar loss. Every parameter in every
        // layer must receive a finite gradient.
        let loss = current.sum_all();
        let grads = loss.backward();

        for (layer_idx, handles) in all_handles.iter().enumerate() {
            for (name, param, expected_shape) in [
                ("W_q", &handles.w_q, vec![d_model, d_model]),
                ("W_k", &handles.w_k, vec![d_model, d_model]),
                ("W_v", &handles.w_v, vec![d_model, d_model]),
                ("W_o", &handles.w_o, vec![d_model, d_model]),
                ("W_gate", &handles.w_gate, vec![d_model, ffn_inner]),
                ("W_up", &handles.w_up, vec![d_model, ffn_inner]),
                ("W_down", &handles.w_down, vec![ffn_inner, d_model]),
            ] {
                let g = grads.get(param).unwrap_or_else(|| {
                    panic!("layer {layer_idx} {name} has no gradient")
                });
                let g_val = realize_f32(&g);
                assert_eq!(
                    g_val.shape().dims(),
                    expected_shape.as_slice(),
                    "layer {layer_idx} {name} gradient shape wrong",
                );
                for &v in g_val.as_slice() {
                    assert!(
                        v.is_finite(),
                        "layer {layer_idx} {name} gradient non-finite: {v}",
                    );
                }
            }
        }
    }

    #[test]
    fn realize_deep_chain_of_10k_adds_without_stack_overflow() {
        // Build a chain: a + a + a + ... (10 000 adds). This would have
        // blown the Rust recursion limit under the old memoized-recursive
        // executor. With the iterative topo walk it should complete fine.
        //
        // Expected value per element: 1.0 * 10 001 (the original plus
        // 10 000 additions of itself).
        let a = Tensor::from_f32(vec![1.0, 1.0, 1.0], Shape::from_dims(&[3]), cpu_dev());
        let mut current = a.clone();
        for _ in 0..10_000 {
            current = current.add(&a);
        }
        let result = realize_f32(&current);
        let expected = 10_001.0;
        for &v in result.as_slice() {
            assert!((v - expected).abs() < 1.0, "got {v}, expected {expected}");
        }
    }
}

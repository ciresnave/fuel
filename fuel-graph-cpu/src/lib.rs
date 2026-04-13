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
//! CUDA or Metal is a separate future crate (`fuel-graph-cuda`,
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

use fuel_graph::{topo_order, topo_order_multi, ConstData, NodeId, Op, Tensor};
use fuel_reference_backend::{ops, RefTensor};
use half::{bf16, f16};
use std::collections::HashMap;
use tracing::{debug_span, info_span};

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
            std::rc::Rc::ptr_eq(graph_rc, t.graph()),
            "realize_many_f32: all tensors must belong to the same graph",
        );
    }
    let graph = graph_rc.borrow();
    let roots: Vec<NodeId> = tensors.iter().map(|t| t.id()).collect();
    let order = topo_order_multi(&graph, &roots);
    let mut cache: HashMap<NodeId, AnyTensor> = HashMap::new();

    for id in order {
        let node = graph.node(id);
        let result = eval_node(&node.op, &node.inputs, &node.shape, &cache);
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

/// Core realize loop: walk the graph in topological order, caching
/// each node's output and dispatching `MatMul` to the fast path.
fn realize_any(tensor: &Tensor) -> AnyTensor {
    let _span = info_span!("realize_cpu").entered();
    let graph = tensor.graph().borrow();
    let order = topo_order(&graph, tensor.id());
    let num_nodes = order.len();
    let _walk = info_span!("topo_walk", nodes = num_nodes).entered();
    let mut cache: HashMap<NodeId, AnyTensor> = HashMap::new();

    for id in order {
        let node = graph.node(id);
        let result = eval_node(&node.op, &node.inputs, &node.shape, &cache);
        cache.insert(id, result);
    }
    drop(_walk);

    cache
        .remove(&tensor.id())
        .expect("realize: target tensor missing from cache after topo walk")
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
        Op::Const(data) => eval_const(data, shape),

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

        // --- linear algebra & shape ---
        Op::Transpose => unary!(inputs, cache, ops::transpose_last_two),
        Op::Permute(axes) => eval_permute(axes, inputs, cache),

        // --- dtype, shape, broadcasting ---
        Op::Cast(target) => eval_cast(*target, inputs, cache),
        Op::BroadcastTo(target_shape) => eval_broadcast_to(target_shape, inputs, cache),
        Op::Reshape(target_shape) => eval_reshape(target_shape, inputs, cache),
        Op::ReduceSumTo(target_shape) => eval_reduce_sum_to(target_shape, inputs, cache),

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
        Op::LayerNormLastDim { eps } => eval_layer_norm_last_dim(*eps, inputs, cache),
        Op::SoftmaxLastDimBackward => eval_softmax_last_dim_backward(inputs, cache),
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
    }
}

fn eval_const(data: &ConstData, shape: &fuel_core_types::Shape) -> AnyTensor {
    // `v.clone()` is a cheap `Arc::clone`; `from_arc` wraps without
    // copying. Weight-sized `Const` nodes stay shared all the way
    // through the executor cache.
    match data {
        ConstData::F32(v) => AnyTensor::F32(RefTensor::from_arc(v.clone(), shape.clone())),
        ConstData::F64(v) => AnyTensor::F64(RefTensor::from_arc(v.clone(), shape.clone())),
        ConstData::BF16(v) => AnyTensor::BF16(RefTensor::from_arc(v.clone(), shape.clone())),
        ConstData::F16(v) => AnyTensor::F16(RefTensor::from_arc(v.clone(), shape.clone())),
        ConstData::U32(v) => AnyTensor::U32(RefTensor::from_arc(v.clone(), shape.clone())),
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
        (a, b) => panic!(
            "matmul: unsupported operand dtypes (lhs={:?}, rhs={:?})",
            a.dtype(),
            b.dtype(),
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
        let a = Tensor::from_f32(a_data, Shape::from_dims(&[16, 32]));
        let b = a.const_f32_like(b_data, Shape::from_dims(&[32, 8]));
        let c = a.matmul(&b);
        assert_equivalent_f32(&c);
    }

    #[test]
    fn matmul_matches_reference_batched() {
        // [2, 3, 4] @ [2, 4, 5] — batched rank-3 matmul.
        let a_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..40).map(|i| (i as f32 * 0.2) - 1.0).collect();
        let a = Tensor::from_f32(a_data, Shape::from_dims(&[2, 3, 4]));
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
        let x = Tensor::from_f32(x_data, Shape::from_dims(&[1, seq, d_model]));
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
    fn deep_matmul_chain_doesnt_explode() {
        // Chain 20 small matmuls; verifies that the executor handles
        // deep dependency graphs without issues and produces the same
        // result as reference.
        let init = Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0], Shape::from_dims(&[2, 2]));
        let mut current = init.clone();
        for i in 0..20 {
            let step_data = vec![1.0 + (i as f32) * 0.01, 0.0, 0.0, 1.0 - (i as f32) * 0.01];
            let step = init.const_f32_like(step_data, Shape::from_dims(&[2, 2]));
            current = current.matmul(&step);
        }
        assert_equivalent_f32(&current);
    }
}

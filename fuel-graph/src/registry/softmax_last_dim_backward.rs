//! SoftmaxLastDimBackward — fused backward helper for the
//! `SoftmaxLastDim` forward. Phase 7.6 step 4 (continued — first
//! backward helper migrated; activates the registry's
//! `BackwardKind::Fused(id)` connection for the first time).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry`.
//!
//! The backward formula is `s * (g - sum(g * s, last_dim,
//! keepdim=true))` where `s` is the forward output and `g` is the
//! upstream gradient. Two inputs `[y, upstream]`; parameterless.
//!
//! ## Architectural note — registry purpose for backward helpers
//!
//! Backward helper entries serve a different role from forward
//! entries: there is no user-decomposed form to recognize (the
//! matcher is always stubbed), and the registry's `decompose`
//! function isn't a "synthesize from primitives" fallback because
//! the closed-form backward expression doesn't simplify to a
//! small primitive subgraph that's worth materializing. Instead
//! the registry entry exists to:
//!
//! - declare the op's identity (FusedOpId, FusedOpParams variant,
//!   shape/dtype rules);
//! - serve as the target of `BackwardKind::Fused(id)` from the
//!   matching forward entry (the architectural connection this
//!   commit activates);
//! - host per-backend `BackendImpl` registrations in the
//!   `FusedKernelRegistry` (step 6 extension; pending — the
//!   binding-table already covers CPU dispatch via the legacy
//!   `OpKind` route).
//!
//! Higher-order gradients (backward through this helper itself)
//! panic per `Tensor::backward`'s MVP behavior. The registry entry's
//! `backward` field reflects this: [`BackwardKind::NotDifferentiable`].

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_core_types::{DType, Shape};

/// Metadata-side registry entry for SoftmaxLastDimBackward.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::SOFTMAX_LAST_DIM_BACKWARD,
        name:       "SoftmaxLastDimBackward",
        family:     FusedOpFamily::Backward,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Shape rule: output equals input 0 (the forward softmax output).
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 2,
        "SoftmaxLastDimBackward takes 2 inputs (y, upstream)",
    );
    input_shapes[0].clone()
}

/// Dtype rule: output dtype equals input 0 (the forward softmax output).
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "SoftmaxLastDimBackward takes 2 inputs",
    );
    input_dtypes[0]
}

/// Decompose to the closed-form softmax backward
/// `grad_x = s · (g − sum(g·s, last_dim, keepdim=true))`, where `s` (input 0)
/// is the forward output and `g` (input 1) is the upstream gradient. Every
/// primitive exists (`Mul`, `ReduceSumTo` to a keepdim shape, `BroadcastTo`,
/// `Sub`) — the same `ReduceSumTo([…,1]) + BroadcastTo` idiom the forward
/// `softmax_last_dim::decompose` uses — so per G2 this is a real 5-node
/// decomposition, not a basis-gap self-return.
pub fn decompose(graph: &mut Graph, id: NodeId, _params: &FusedOpParams) -> NodeId {
    let (s_id, g_id, x_shape, dtype) = {
        let n = graph.node(id);
        (n.inputs[0], n.inputs[1], n.shape.clone(), n.dtype)
    };
    // keepdim shape: last dim → 1.
    let mut kd = x_shape.dims().to_vec();
    let last = kd.len() - 1;
    kd[last] = 1;
    let keepdim = Shape::from_dims(&kd);

    let gs = graph.push(Node {
        op: Op::Mul,
        inputs: vec![g_id, s_id],
        shape: x_shape.clone(),
        dtype,
    });
    let summed = graph.push(Node {
        op: Op::ReduceSumTo(keepdim.clone()),
        inputs: vec![gs],
        shape: keepdim,
        dtype,
    });
    let summed_b = graph.push(Node {
        op: Op::BroadcastTo(x_shape.clone()),
        inputs: vec![summed],
        shape: x_shape.clone(),
        dtype,
    });
    let sub = graph.push(Node {
        op: Op::Sub,
        inputs: vec![g_id, summed_b],
        shape: x_shape.clone(),
        dtype,
    });
    graph.push(Node {
        op: Op::Mul,
        inputs: vec![s_id, sub],
        shape: x_shape,
        dtype,
    })
}

/// Matcher stub — backward-helper nodes originate from autograd
/// emitting `Op::Fused(SOFTMAX_LAST_DIM_BACKWARD, _)`, not from
/// user-decomposed forms.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

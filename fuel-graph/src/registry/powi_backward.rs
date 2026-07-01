//! PowIBackward — fused backward helper for the `Op::PowI(exp)`
//! primitive.
//!
//! Closed-form: `grad_x = exp · x^(exp-1) · upstream`. Two inputs
//! `[x, upstream]` + `exp: i32`. Replaces the autograd primitive
//! decomposition `PowI(n-1) → MulScalar(n as f64) → Mul` (3 nodes,
//! 3 dispatches, 2 intermediate buffers) with a single launch of
//! baracuda alpha.31's `unary_powi_backward_*` kernel.
//!
//! Like `ReduceMaxToBackward`, this is the backward of a primitive
//! (`Op::PowI`), not of a fused forward — autograd reaches it directly
//! from `Op::PowI`'s arm in `Tensor::backward` rather than through a
//! `BackwardKind::Fused` edge.
//!
//! See `softmax_last_dim_backward.rs` for the shared architectural
//! rationale around backward-helper entries.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, Shape};

/// Metadata-side registry entry for PowIBackward.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::POWI_BACKWARD,
        name:       "PowIBackward",
        family:     FusedOpFamily::Backward,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 2,
        "PowIBackward takes 2 inputs (x, upstream)",
    );
    // grad_x preserves x's shape.
    input_shapes[0].clone()
}

fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "PowIBackward takes 2 inputs",
    );
    input_dtypes[0]
}

/// Decompose to the closed-form gradient `grad_x = exp · x^(exp-1) · upstream`,
/// the same recipe the pre-PowIBackward autograd path emitted inline. Inputs
/// are `[x, upstream]`; `exp` rides on the params. Every primitive exists
/// (`Op::PowI`, `Op::MulScalar`, `Op::Mul`), so per G2 this is a real
/// decomposition, not a basis-gap self-return. Edge cases fall out: `exp==1` →
/// `PowI(0)·1·upstream = upstream`; `exp==0` → `MulScalar(0) = 0`.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (x_id, up_id, shape, dtype) = {
        let n = graph.node(id);
        (n.inputs[0], n.inputs[1], n.shape.clone(), n.dtype)
    };
    let exp = match params {
        FusedOpParams::PowIBackward { exp } => *exp,
        // G2: total + never-panic — non-PowIBackward params are an impossible
        // registry-dispatch invariant violation; return self.
        _ => return id,
    };
    let pow = graph.push(Node {
        op: Op::PowI(exp - 1),
        inputs: vec![x_id],
        shape: shape.clone(),
        dtype,
    });
    let scaled = graph.push(Node {
        op: Op::MulScalar(exp as f64),
        inputs: vec![pow],
        shape: shape.clone(),
        dtype,
    });
    graph.push(Node {
        op: Op::Mul,
        inputs: vec![scaled, up_id],
        shape,
        dtype,
    })
}

pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

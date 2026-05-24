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
use crate::{Graph, NodeId};
use fuel_core_types::{DType, Shape};

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

pub fn decompose(_graph: &mut Graph, _id: NodeId, _params: &FusedOpParams) -> NodeId {
    panic!(
        "powi_backward::decompose: backward helpers have no primitive \
         decomposition exposed at the registry layer. (The pre-PowIBackward \
         autograd path emitted PowI(n-1) → MulScalar → Mul directly; that \
         decomposition lives in Tensor::backward and is the fallback when \
         the fused kernel isn't registered for the target backend.) See \
         module docs.",
    );
}

pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

//! ReduceMaxToBackward — fused backward helper for the `Op::ReduceMaxTo`
//! primitive. Phase 7.6 step 4 (continued).
//!
//! Routes the upstream gradient to position(s) where x equals its
//! per-window max; tied maxes share equally (fair-share subgradient).
//! Two inputs `[x, upstream]`, parameterless.
//!
//! Unique among the four backward-helper entries: this one is the
//! backward of a **primitive** (`Op::ReduceMaxTo`), not of a fused
//! forward. There is no `BackwardKind::Fused(REDUCE_MAX_TO_BACKWARD)`
//! edge from any forward entry — autograd reaches this helper
//! directly from `Op::ReduceMaxTo`'s arm in `Tensor::backward`. The
//! registry entry still exists so the executor's dispatch stays
//! consistent (every fused op routes through `Op::Fused(id, _)`) and
//! so step 5 can drop the legacy `Op::ReduceMaxToBackward` variant.
//!
//! See `softmax_last_dim_backward.rs` for the shared architectural
//! rationale around backward-helper entries.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_core_types::{DType, Shape};

/// Metadata-side registry entry for ReduceMaxToBackward.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::REDUCE_MAX_TO_BACKWARD,
        name:       "ReduceMaxToBackward",
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
        "ReduceMaxToBackward takes 2 inputs (x, upstream)",
    );
    // grad_x has the original x's shape (input 0); upstream's shape
    // is the forward target_shape.
    input_shapes[0].clone()
}

fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "ReduceMaxToBackward takes 2 inputs",
    );
    input_dtypes[0]
}

pub fn decompose(_graph: &mut Graph, _id: NodeId, _params: &FusedOpParams) -> NodeId {
    panic!(
        "reduce_max_to_backward::decompose: backward helpers have \
         no primitive decomposition exposed at the registry layer \
         (and ReduceMaxTo's backward would need an equality \
         primitive that doesn't exist today regardless). See module \
         docs.",
    );
}

pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

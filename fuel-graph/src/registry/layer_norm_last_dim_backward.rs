//! LayerNormLastDimBackward — fused backward helper for the
//! `LayerNormLastDim` forward. Phase 7.6 step 4 (continued).
//!
//! Recomputes mean/variance from the original x (rather than caching
//! intermediates from forward) and produces grad_x. Two inputs
//! `[x, upstream]` + `eps`.
//!
//! See `softmax_last_dim_backward.rs` for the architectural rationale
//! shared by all four backward-helper entries: the registry's role is
//! identity + `BackwardKind::Fused` target + future `BackendImpl`
//! host, not lowering-to-primitives.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_core_types::{DType, Shape};

/// Metadata-side registry entry for LayerNormLastDimBackward.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::LAYER_NORM_LAST_DIM_BACKWARD,
        name:       "LayerNormLastDimBackward",
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
        "LayerNormLastDimBackward takes 2 inputs (x, upstream)",
    );
    input_shapes[0].clone()
}

fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "LayerNormLastDimBackward takes 2 inputs",
    );
    input_dtypes[0]
}

pub fn decompose(_graph: &mut Graph, _id: NodeId, _params: &FusedOpParams) -> NodeId {
    panic!(
        "layer_norm_last_dim_backward::decompose: backward helpers \
         have no primitive decomposition exposed at the registry \
         layer. See module docs.",
    );
}

pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

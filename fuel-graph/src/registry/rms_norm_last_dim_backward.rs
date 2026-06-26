//! RmsNormLastDimBackward — fused backward helper for the
//! `RmsNormLastDim` forward. Phase 7.6 step 4 (continued).
//!
//! Closed-form: `grad_x = r_rms · (upstream - x · s / (n · (mean_sq + eps)))`
//! where `r_rms = 1/sqrt(mean_sq + eps)`, `s = sum(upstream · x, last)`,
//! `n = last_dim_size`. Two inputs `[x, upstream]` + `eps`.
//!
//! See `softmax_last_dim_backward.rs` for the shared architectural
//! rationale.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};

/// Metadata-side registry entry for RmsNormLastDimBackward.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::RMS_NORM_LAST_DIM_BACKWARD,
        name:       "RmsNormLastDimBackward",
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
        "RmsNormLastDimBackward takes 2 inputs (x, upstream)",
    );
    input_shapes[0].clone()
}

fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "RmsNormLastDimBackward takes 2 inputs",
    );
    input_dtypes[0]
}

pub fn decompose(_graph: &mut Graph, _id: NodeId, _params: &FusedOpParams) -> NodeId {
    panic!(
        "rms_norm_last_dim_backward::decompose: backward helpers \
         have no primitive decomposition exposed at the registry \
         layer. See module docs.",
    );
}

pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

//! LayerNormLastDim — `(x - mean) / sqrt(variance + eps)` along the
//! last dim, no affine params. Phase 7.6 step 4 (continued — third
//! op migrated after SoftmaxLastDim, FusedLinear, RmsNormLastDim).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (decompose function,
//!   pattern matcher, shape/dtype rules).
//! - [`canonical_pattern`] — placeholder returning `None`. Step 4's
//!   bar is "the registry entry exists, builder emits Op::Fused,
//!   dispatch works." A canonical matcher for the 11-node
//!   LayerNorm subgraph is future work; until it lands, fusion only
//!   fires through the builder, never through pattern recognition.
//! - [`decompose`] — emits the 11-node primitive subgraph.
//!
//! The decomposition mirrors the standard layer-norm formula:
//!
//! ```text
//!   mean        = MeanDim(last)(x)                # rank-reduced
//!   mean_kd     = Reshape(keepdim)(mean)
//!   mean_bcast  = BroadcastTo(x_shape)(mean_kd)
//!   centered    = Sub(x, mean_bcast)
//!   centered_sq = Sqr(centered)
//!   var         = MeanDim(last)(centered_sq)      # rank-reduced
//!   var_kd      = Reshape(keepdim)(var)
//!   var_eps     = AddScalar(eps)(var_kd)
//!   denom       = Sqrt(var_eps)
//!   denom_bcast = BroadcastTo(x_shape)(denom)
//!   out         = Div(centered, denom_bcast)
//! ```

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, Shape};

/// Metadata-side registry entry for LayerNormLastDim.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::LAYER_NORM_LAST_DIM,
        name:       "LayerNormLastDim",
        family:     FusedOpFamily::Norm,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        // Phase 7.6 step 4 (backward-helper batch): the
        // architecturally-correct BackwardKind::Fused edge is now
        // live. `Tensor::backward`'s Op::Fused arm reads this and
        // emits Op::Fused(LAYER_NORM_LAST_DIM_BACKWARD, _) instead
        // of the legacy variant.
        backward:   BackwardKind::Fused(FusedOps::LAYER_NORM_LAST_DIM_BACKWARD),
        shape_rule: shape_passthrough,
        dtype_rule: dtype_passthrough,
        output_views: None,
    }
}

/// Shape rule: LayerNormLastDim preserves its single input's shape.
fn shape_passthrough(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(input_shapes.len(), 1, "LayerNormLastDim takes one input");
    input_shapes[0].clone()
}

/// Dtype rule: LayerNormLastDim preserves its single input's dtype.
fn dtype_passthrough(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(input_dtypes.len(), 1, "LayerNormLastDim takes one input");
    input_dtypes[0]
}

/// Lower a fused LayerNormLastDim node to its canonical 11-node
/// primitive subgraph and return the new root id.
///
/// The fused node `id` may be either `Op::LayerNormLastDim { eps }`
/// (legacy emission) or `Op::Fused(FusedOps::LAYER_NORM_LAST_DIM,
/// FusedOpParams::LayerNormLastDim { eps })` (the new builder path).
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (x_id, x_shape, dtype) = {
        let n = graph.node(id);
        (n.inputs[0], n.shape.clone(), n.dtype)
    };
    let eps = match params {
        FusedOpParams::LayerNormLastDim { eps } => *eps,
        _ => panic!(
            "layer_norm_last_dim::decompose called with non-LayerNorm \
             params {params:?}; node op = {:?}",
            graph.node(id).op
        ),
    };
    let dims = x_shape.dims().to_vec();
    let rank = dims.len();
    debug_assert!(rank >= 1, "LayerNormLastDim requires rank >= 1");
    let last = rank - 1;

    let mut keepdim_dims = dims.clone();
    keepdim_dims[last] = 1;
    let keepdim_shape = Shape::from_dims(&keepdim_dims);
    let mut reduced_dims = dims.clone();
    reduced_dims.remove(last);
    let reduced_shape = Shape::from_dims(&reduced_dims);

    let mean_id = graph.push(Node {
        op:     Op::MeanDim(last),
        inputs: vec![x_id],
        shape:  reduced_shape.clone(),
        dtype,
    });
    let mean_kd_id = graph.push(Node {
        op:     Op::Reshape(keepdim_shape.clone()),
        inputs: vec![mean_id],
        shape:  keepdim_shape.clone(),
        dtype,
    });
    let mean_bcast_id = graph.push(Node {
        op:     Op::BroadcastTo(x_shape.clone()),
        inputs: vec![mean_kd_id],
        shape:  x_shape.clone(),
        dtype,
    });
    let centered_id = graph.push(Node {
        op:     Op::Sub,
        inputs: vec![x_id, mean_bcast_id],
        shape:  x_shape.clone(),
        dtype,
    });
    let centered_sq_id = graph.push(Node {
        op:     Op::Sqr,
        inputs: vec![centered_id],
        shape:  x_shape.clone(),
        dtype,
    });
    let var_id = graph.push(Node {
        op:     Op::MeanDim(last),
        inputs: vec![centered_sq_id],
        shape:  reduced_shape,
        dtype,
    });
    let var_kd_id = graph.push(Node {
        op:     Op::Reshape(keepdim_shape.clone()),
        inputs: vec![var_id],
        shape:  keepdim_shape.clone(),
        dtype,
    });
    let var_eps_id = graph.push(Node {
        op:     Op::AddScalar(eps),
        inputs: vec![var_kd_id],
        shape:  keepdim_shape.clone(),
        dtype,
    });
    let denom_id = graph.push(Node {
        op:     Op::Sqrt,
        inputs: vec![var_eps_id],
        shape:  keepdim_shape,
        dtype,
    });
    let denom_bcast_id = graph.push(Node {
        op:     Op::BroadcastTo(x_shape.clone()),
        inputs: vec![denom_id],
        shape:  x_shape.clone(),
        dtype,
    });
    let out_id = graph.push(Node {
        op:     Op::Div,
        inputs: vec![centered_id, denom_bcast_id],
        shape:  x_shape,
        dtype,
    });
    out_id
}

/// Placeholder matcher: returns `None` for every input. The
/// 11-node LayerNorm subgraph is structurally larger than the
/// SoftmaxLastDim / RmsNormLastDim matchers, and the matcher is
/// not on the critical path for step 4 (lowering still works
/// through the builder; the matcher only matters when fusion
/// fires on hand-built decomposed forms).
///
/// A canonical matcher recognizing the 11-node pattern + checking
/// single-consumer guards on every intermediate is a follow-up
/// extension. Until then, this rule effectively reads as a
/// one-way migration: builder-emitted `Op::Fused` becomes the
/// canonical form; user-decomposed forms stay decomposed.
pub fn canonical_pattern(_graph: &Graph, _div_id: NodeId) -> Option<PatternMatch> {
    None
}

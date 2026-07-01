//! RmsNormLastDim — `x / sqrt(mean(x²) + eps)` along the last dim.
//! Phase 7.6 step 4 (continued — second op migrated after SoftmaxLastDim
//! + FusedLinear).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (decompose function,
//!   pattern matcher, shape/dtype rules).
//! - [`canonical_pattern`] — recognizes the 7-node decomposed
//!   subgraph and returns the bound `x` input plus the eps parameter.
//! - [`decompose`] — emits the canonical 7-node primitive subgraph
//!   from a fused-op node carrying `FusedOpParams::RmsNormLastDim`.
//!
//! The decomposition mirrors [`crate::Tensor::rms_norm_last_dim_decomposed`]
//! exactly — `Sqr → MeanDim → Reshape → AddScalar → Sqrt → BroadcastTo
//! → Div` — so the round-trip lower-then-fuse is shape-identical to the
//! existing decomposed path. The matcher refuses to fire when any
//! intermediate has consumers outside the canonical pattern (same
//! conservatism as the SoftmaxLastDim matcher).

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, Shape};
use std::collections::HashMap;

/// Metadata-side registry entry for RmsNormLastDim.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::RMS_NORM_LAST_DIM,
        name:       "RmsNormLastDim",
        family:     FusedOpFamily::Norm,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        // Phase 7.6 step 4 (backward-helper batch): the
        // architecturally-correct BackwardKind::Fused edge is now
        // live. `Tensor::backward`'s Op::Fused arm reads this and
        // emits Op::Fused(RMS_NORM_LAST_DIM_BACKWARD, _) instead of
        // the legacy variant.
        backward:   BackwardKind::Fused(FusedOps::RMS_NORM_LAST_DIM_BACKWARD),
        shape_rule: shape_passthrough,
        dtype_rule: dtype_passthrough,
        output_views: None,
    }
}

/// Shape rule: RmsNormLastDim preserves its single input's shape.
fn shape_passthrough(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(input_shapes.len(), 1, "RmsNormLastDim takes one input");
    input_shapes[0].clone()
}

/// Dtype rule: RmsNormLastDim preserves its single input's dtype.
fn dtype_passthrough(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(input_dtypes.len(), 1, "RmsNormLastDim takes one input");
    input_dtypes[0]
}

/// Lower a fused RmsNormLastDim node to its canonical 7-node primitive
/// subgraph and return the new root id. Mirrors
/// [`crate::Tensor::rms_norm_last_dim_decomposed`] node-for-node:
///
/// ```text
///   sq          = Sqr(x)
///   mean        = MeanDim(last)(sq)               # rank-reduced
///   mean_kd     = Reshape(keepdim)(mean)
///   denom_sq    = AddScalar(eps)(mean_kd)
///   denom       = Sqrt(denom_sq)
///   denom_bcast = BroadcastTo(x_shape)(denom)
///   out         = Div(x, denom_bcast)
/// ```
///
/// The fused node `id` may be either `Op::RmsNormLastDim { eps }`
/// (legacy emission) or `Op::Fused(FusedOps::RMS_NORM_LAST_DIM,
/// FusedOpParams::RmsNormLastDim { eps })` (the new builder path).
/// The decomposition is identical for both; only the eps extraction
/// differs.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (x_id, x_shape, dtype) = {
        let n = graph.node(id);
        (n.inputs[0], n.shape.clone(), n.dtype)
    };
    let eps = match params {
        FusedOpParams::RmsNormLastDim { eps } => *eps,
        // G2: decompose is total + never-panic. A non-RmsNorm params payload
        // is an impossible registry-dispatch invariant violation; return self
        // rather than crash.
        _ => return id,
    };
    let dims = x_shape.dims().to_vec();
    let rank = dims.len();
    debug_assert!(rank >= 1, "RmsNormLastDim requires rank >= 1");
    let last = rank - 1;

    let mut keepdim_dims = dims.clone();
    keepdim_dims[last] = 1;
    let keepdim_shape = Shape::from_dims(&keepdim_dims);
    // `mean_dim` drops the reduced axis from its output shape — same
    // contract as Op::MeanDim's node-shape.
    let mut reduced_dims = dims.clone();
    reduced_dims.remove(last);
    let reduced_shape = Shape::from_dims(&reduced_dims);

    let sq_id = graph.push(Node {
        op:     Op::Sqr,
        inputs: vec![x_id],
        shape:  x_shape.clone(),
        dtype,
    });
    let mean_id = graph.push(Node {
        op:     Op::MeanDim(last),
        inputs: vec![sq_id],
        shape:  reduced_shape,
        dtype,
    });
    let mean_kd_id = graph.push(Node {
        op:     Op::Reshape(keepdim_shape.clone()),
        inputs: vec![mean_id],
        shape:  keepdim_shape.clone(),
        dtype,
    });
    let denom_sq_id = graph.push(Node {
        op:     Op::AddScalar(eps),
        inputs: vec![mean_kd_id],
        shape:  keepdim_shape.clone(),
        dtype,
    });
    let denom_id = graph.push(Node {
        op:     Op::Sqrt,
        inputs: vec![denom_sq_id],
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
        inputs: vec![x_id, denom_bcast_id],
        shape:  x_shape,
        dtype,
    });
    out_id
}

/// Match the canonical 7-node RmsNormLastDim subgraph rooted at the
/// final `Div` node. Returns a [`PatternMatch`] binding `x` to input
/// 0 and stamping `FusedOpParams::RmsNormLastDim { eps }` extracted
/// from the AddScalar's scalar.
///
/// Conservative: every intermediate must be consumed only within the
/// canonical pattern so fusing doesn't discard a value the user reads
/// from one of the intermediates.
pub fn canonical_pattern(graph: &Graph, div_id: NodeId) -> Option<PatternMatch> {
    let div = graph.node(div_id);
    if !matches!(div.op, Op::Div) { return None; }
    if div.inputs.len() != 2 { return None; }
    let x_id = div.inputs[0];
    let denom_bcast_id = div.inputs[1];

    let denom_bcast = graph.node(denom_bcast_id);
    if !matches!(denom_bcast.op, Op::BroadcastTo(_)) { return None; }
    if denom_bcast.inputs.len() != 1 { return None; }
    let denom_id = denom_bcast.inputs[0];

    let denom = graph.node(denom_id);
    if !matches!(denom.op, Op::Sqrt) { return None; }
    if denom.inputs.len() != 1 { return None; }
    let denom_sq_id = denom.inputs[0];

    let denom_sq = graph.node(denom_sq_id);
    let eps = match denom_sq.op {
        Op::AddScalar(e) => e,
        _ => return None,
    };
    if denom_sq.inputs.len() != 1 { return None; }
    let mean_kd_id = denom_sq.inputs[0];

    let mean_kd = graph.node(mean_kd_id);
    if !matches!(mean_kd.op, Op::Reshape(_)) { return None; }
    if mean_kd.inputs.len() != 1 { return None; }
    let mean_id = mean_kd.inputs[0];

    let mean = graph.node(mean_id);
    let last_axis_via_mean = match mean.op {
        Op::MeanDim(d) => d,
        _ => return None,
    };
    if mean.inputs.len() != 1 { return None; }
    let sq_id = mean.inputs[0];

    let sq = graph.node(sq_id);
    if !matches!(sq.op, Op::Sqr) { return None; }
    if sq.inputs.len() != 1 { return None; }
    // Sqr's input must be the same x that Div consumes — otherwise it
    // isn't the rms-norm pattern (it's an unrelated `x / sqrt(... + eps)`
    // expression).
    if sq.inputs[0] != x_id { return None; }

    // Shape sanity checks: the MeanDim must be along the last axis,
    // and Reshape's target must be the x shape with last-dim=1.
    let x_shape = &graph.node(x_id).shape;
    if x_shape.rank() == 0 { return None; }
    let last = x_shape.rank() - 1;
    if last_axis_via_mean != last { return None; }
    let kd_shape = &graph.node(mean_kd_id).shape;
    if kd_shape.rank() != x_shape.rank() { return None; }
    for axis in 0..x_shape.rank() {
        let expected = if axis == last { 1 } else { x_shape.dims()[axis] };
        if kd_shape.dims()[axis] != expected { return None; }
    }

    // Conservativeness: every intermediate consumed only within this
    // pattern.
    let intermediates_with_expected_count = [
        (sq_id, 1),
        (mean_id, 1),
        (mean_kd_id, 1),
        (denom_sq_id, 1),
        (denom_id, 1),
        (denom_bcast_id, 1),
    ];
    let consumer_counts = count_consumers(graph);
    for (nid, expected) in intermediates_with_expected_count {
        if consumer_counts.get(&nid).copied().unwrap_or(0) != expected {
            return None;
        }
    }

    Some(PatternMatch {
        bindings: vec![(0, x_id)],
        params:   FusedOpParams::RmsNormLastDim { eps },
    })
}

/// Build a consumer-count index across the entire graph. Mirrors the
/// helper in `softmax_last_dim`; replicated here so the matcher is
/// self-contained.
fn count_consumers(graph: &Graph) -> HashMap<NodeId, usize> {
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    let n = graph.len();
    for nid in 0..n {
        let node = graph.node(NodeId(nid));
        for &input in &node.inputs {
            *counts.entry(input).or_insert(0) += 1;
        }
    }
    counts
}

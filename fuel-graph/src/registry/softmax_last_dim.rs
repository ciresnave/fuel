//! SoftmaxLastDim — first fused op migrated through the FusedOpRegistry.
//! Phase 7.6 step 3.
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (decompose function,
//!   pattern matcher, shape/dtype rules).
//! - [`canonical_pattern`] — the pattern callable that recognizes the
//!   7-node primitive subgraph and returns the bound `x` input.
//! - [`decompose`] — emits the canonical 7-node primitive subgraph for a
//!   given fused-op node.
//!
//! The auto-generated `LoweringRule` and `FusionRule` in `crate::opt`
//! consume this entry; PR 3's hand-written `SoftmaxLastDimLowerRule`
//! and `SoftmaxLastDimFuseRule` are deleted.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_core_types::{DType, Shape};
use std::collections::HashMap;

/// Metadata-side registry entry for SoftmaxLastDim.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::SOFTMAX_LAST_DIM,
        name:       "SoftmaxLastDim",
        family:     FusedOpFamily::Forward,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        // Step 3 wires backward through the legacy
        // Op::SoftmaxLastDimBackward variant; the backward fused-op
        // isn't migrated to a registry entry until step 4. The
        // `Decompose` flavor is unused here because Tensor::backward
        // dispatches Op::Fused(SOFTMAX_LAST_DIM, _) directly through
        // a hard-coded arm in step 3 (see fuel-graph/src/lib.rs).
        backward:   BackwardKind::NotDifferentiable,
        shape_rule: shape_passthrough,
        dtype_rule: dtype_passthrough,
    }
}

/// Shape rule: SoftmaxLastDim preserves its single input's shape.
fn shape_passthrough(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(input_shapes.len(), 1, "SoftmaxLastDim takes one input");
    input_shapes[0].clone()
}

/// Dtype rule: SoftmaxLastDim preserves its single input's dtype.
fn dtype_passthrough(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(input_dtypes.len(), 1, "SoftmaxLastDim takes one input");
    input_dtypes[0]
}

/// Lower a fused SoftmaxLastDim node to the canonical 7-node primitive
/// subgraph and return the new root id. Mirrors PR 3's
/// `SoftmaxLastDimLowerRule::rewrite`; the only difference is that the
/// fused node identified by `id` may be either `Op::SoftmaxLastDim`
/// (legacy emission site, e.g. the pipelined direct-construct test) or
/// `Op::Fused(FusedOps::SOFTMAX_LAST_DIM, _)` (the new builder path).
/// The decomposition is identical for both — it reads the input id +
/// shape + dtype off the node and emits primitives.
///
/// Lowered form (7 nodes, symmetric across max/sum sides):
///
/// ```text
///   m   = ReduceMaxTo([..., 1])(x)   # max-keepdim in one node
///   mb  = BroadcastTo([..., last])(m)
///   s   = Sub(x, mb)                 # numerically-stable shift
///   e   = Exp(s)
///   d   = ReduceSumTo([..., 1])(e)   # sum-keepdim in one node
///   db  = BroadcastTo([..., last])(d)
///   out = Div(e, db)
/// ```
pub fn decompose(graph: &mut Graph, id: NodeId, _params: &FusedOpParams) -> NodeId {
    let (x_id, x_shape, dtype) = {
        let n = graph.node(id);
        (n.inputs[0], n.shape.clone(), n.dtype)
    };
    let dims = x_shape.dims().to_vec();
    let rank = dims.len();
    let last = rank - 1;

    let mut keepdim_dims = dims.clone();
    keepdim_dims[last] = 1;
    let keepdim_shape = Shape::from_dims(&keepdim_dims);

    let m_id = graph.push(Node {
        op:     Op::ReduceMaxTo(keepdim_shape.clone()),
        inputs: vec![x_id],
        shape:  keepdim_shape.clone(),
        dtype,
    });
    let mb_id = graph.push(Node {
        op:     Op::BroadcastTo(x_shape.clone()),
        inputs: vec![m_id],
        shape:  x_shape.clone(),
        dtype,
    });
    let s_id = graph.push(Node {
        op:     Op::Sub,
        inputs: vec![x_id, mb_id],
        shape:  x_shape.clone(),
        dtype,
    });
    let e_id = graph.push(Node {
        op:     Op::Exp,
        inputs: vec![s_id],
        shape:  x_shape.clone(),
        dtype,
    });
    let d_id = graph.push(Node {
        op:     Op::ReduceSumTo(keepdim_shape.clone()),
        inputs: vec![e_id],
        shape:  keepdim_shape,
        dtype,
    });
    let db_id = graph.push(Node {
        op:     Op::BroadcastTo(x_shape.clone()),
        inputs: vec![d_id],
        shape:  x_shape.clone(),
        dtype,
    });
    let out_id = graph.push(Node {
        op:     Op::Div,
        inputs: vec![e_id, db_id],
        shape:  x_shape,
        dtype,
    });
    out_id
}

/// Match the canonical 7-node SoftmaxLastDim subgraph rooted at a
/// `Div` node. Returns a [`PatternMatch`] with `bindings = [(0, x_id)]`
/// when the pattern fires. Conservative: every intermediate must be
/// consumed only within the canonical pattern so fusing doesn't
/// discard a value the user reads.
///
/// Direct port of PR 3's `canonical_softmax_pattern`; this is the
/// matcher referenced from [`SubgraphPattern::Callable`] in the
/// registry entry.
pub fn canonical_pattern(graph: &Graph, div_id: NodeId) -> Option<PatternMatch> {
    let div = graph.node(div_id);
    if !matches!(div.op, Op::Div) { return None; }
    if div.inputs.len() != 2 { return None; }
    let e_id = div.inputs[0];
    let db_id = div.inputs[1];

    let db = graph.node(db_id);
    if !matches!(db.op, Op::BroadcastTo(_)) { return None; }
    if db.inputs.len() != 1 { return None; }
    let d_id = db.inputs[0];

    let d = graph.node(d_id);
    if !matches!(d.op, Op::ReduceSumTo(_)) { return None; }
    if d.inputs.len() != 1 || d.inputs[0] != e_id { return None; }

    let e = graph.node(e_id);
    if !matches!(e.op, Op::Exp) { return None; }
    if e.inputs.len() != 1 { return None; }
    let s_id = e.inputs[0];

    let s = graph.node(s_id);
    if !matches!(s.op, Op::Sub) { return None; }
    if s.inputs.len() != 2 { return None; }
    let x_id = s.inputs[0];
    let mb_id = s.inputs[1];

    let mb = graph.node(mb_id);
    if !matches!(mb.op, Op::BroadcastTo(_)) { return None; }
    if mb.inputs.len() != 1 { return None; }
    let m_id = mb.inputs[0];

    let m = graph.node(m_id);
    if !matches!(m.op, Op::ReduceMaxTo(_)) { return None; }
    if m.inputs.len() != 1 || m.inputs[0] != x_id { return None; }

    let x_shape = &graph.node(x_id).shape;
    if x_shape.rank() == 0 { return None; }
    let m_shape = &graph.node(m_id).shape;
    if m_shape.rank() != x_shape.rank() { return None; }
    let last = x_shape.rank() - 1;
    for axis in 0..x_shape.rank() {
        let expected = if axis == last { 1 } else { x_shape.dims()[axis] };
        if m_shape.dims()[axis] != expected { return None; }
    }

    // Conservativeness: every intermediate (m, mb, d, db, s) must be
    // consumed ONLY within this pattern. e is consumed twice (-> d
    // via ReduceSumTo, -> div as numerator) which is part of the
    // canonical pattern.
    let intermediates_with_expected_count = [
        (m_id, 1),
        (mb_id, 1),
        (d_id, 1),
        (db_id, 1),
        (s_id, 1),
        (e_id, 2),
    ];
    let consumer_counts = count_consumers(graph);
    for (nid, expected) in intermediates_with_expected_count {
        if consumer_counts.get(&nid).copied().unwrap_or(0) != expected {
            return None;
        }
    }

    Some(PatternMatch {
        bindings: vec![(0, x_id)],
        params:   FusedOpParams::SoftmaxLastDim,
    })
}

/// Build a consumer-count index across the entire graph. Mirrors
/// `opt::count_consumers`; replicated here so the matcher is
/// self-contained and the registry module doesn't pull `opt` into
/// its dependency surface.
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

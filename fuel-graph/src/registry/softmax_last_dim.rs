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
    PatternMatch, SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId, Op};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::shape_expr::ShapeExpr;
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Metadata-side registry entry for SoftmaxLastDim.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::SOFTMAX_LAST_DIM,
        name:       "SoftmaxLastDim",
        family:     FusedOpFamily::Forward,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        // Phase 7.6 step 4 (backward-helper batch): SOFTMAX_LAST_DIM_BACKWARD
        // is now a registry entry, so the architecturally-correct
        // BackwardKind::Fused(id) edge is live. `Tensor::backward`'s
        // `Op::Fused(SOFTMAX_LAST_DIM, _)` arm reads this entry and
        // emits `Op::Fused(SOFTMAX_LAST_DIM_BACKWARD, _)` for the
        // gradient node. The architectural connection latent since
        // step 3 is now exercised.
        backward:   BackwardKind::Fused(FusedOps::SOFTMAX_LAST_DIM_BACKWARD),
        shape_rule: shape_passthrough,
        dtype_rule: dtype_passthrough,
        output_views: None,
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

/// The op's primitive recipe as **portable data** (Increment C slice 1, T5 —
/// the pilot migration). Shape-polymorphic across ranks/extents: keepdim is
/// spelled with the RATIFIED shrink-via-swap (`MaxDim`/`SumDim` at
/// `axis_last` + `Unsqueeze` append — D3), the broadcast targets are
/// `SameAs {{ operand: 0 }}` over the Bind space (D2), and the shared
/// `e = Exp(..)` interior is a repeated subtree the emitter identity-shares
/// into ONE node. Nothing in the datum bakes a shape.
///
/// Emitted form (9 nodes; `e` shared by the denominator reduce and the Div):
///
/// ```text
///   m   = MaxDim(last)(x)            # shrink…
///   mk  = Unsqueeze(append)(m)       # …then restore keepdim ([..., 1])
///   mb  = BroadcastTo(shape_of(x))(mk)
///   s   = Sub(x, mb)                 # numerically-stable shift
///   e   = Exp(s)
///   d   = SumDim(last)(e)
///   dk  = Unsqueeze(append)(d)
///   db  = BroadcastTo(shape_of(x))(dk)
///   out = Div(e, db)
/// ```
fn recipe() -> &'static PatternNode {
    static RECIPE: OnceLock<PatternNode> = OnceLock::new();
    RECIPE.get_or_init(|| {
        let axis_last = || OpAttrs { axis_last: true, ..OpAttrs::default() };
        let same_as_x = || OpAttrs {
            target_shape_rel: Some(ShapeExpr::SameAs { operand: 0 }),
            ..OpAttrs::default()
        };
        let op = |op, attrs, operands| PatternNode::Op { op, attrs, operands };
        let x = || PatternNode::Bind { index: 0 };
        let e = op(OpTag::Exp, OpAttrs::default(), vec![
            op(OpTag::Sub, OpAttrs::default(), vec![
                x(),
                op(OpTag::BroadcastTo, same_as_x(), vec![
                    op(OpTag::Unsqueeze, axis_last(), vec![
                        op(OpTag::MaxDim, axis_last(), vec![x()]),
                    ]),
                ]),
            ]),
        ]);
        op(OpTag::Div, OpAttrs::default(), vec![
            e.clone(),
            op(OpTag::BroadcastTo, same_as_x(), vec![
                op(OpTag::Unsqueeze, axis_last(), vec![
                    op(OpTag::SumDim, axis_last(), vec![e]),
                ]),
            ]),
        ])
    })
}

/// Per-entry scalar projection: SoftmaxLastDim is parameterless, so the
/// right payload projects to ZERO open-slot scalars and any other payload is
/// a typed decline (`None` ⇒ the bridge returns the node unchanged — G2).
fn scalars(params: &FusedOpParams) -> Option<Vec<f64>> {
    match params {
        FusedOpParams::SoftmaxLastDim => Some(Vec::new()),
        _ => None,
    }
}

/// Lower a fused SoftmaxLastDim node to its primitive subgraph and return
/// the new root id — since T5 a re-emit of [`recipe`]'s data through the
/// [`decompose_via_recipe`] bridge (the fused node's input is the bind, the
/// resolving emit derives every interior shape/dtype). Any failure — wrong
/// params payload, a resolution decline at these shapes — returns `id`
/// (fixpoint, surfaced gap, never a panic): exactly the G2 posture the
/// imperative body had.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    decompose_via_recipe(graph, id, recipe(), scalars(params))
}

/// Match a canonical SoftmaxLastDim subgraph rooted at a `Div` node, in
/// EITHER spelling. Returns a [`PatternMatch`] with `bindings = [(0, x_id)]`
/// when a pattern fires. Conservative: every intermediate must be consumed
/// only within the matched pattern so fusing doesn't discard a value the
/// user reads.
///
/// Two spellings, both kept (T5 / risk-3 posture — never delete the old
/// match):
/// * the LEGACY user-spelled 7-node form (`ReduceMaxTo`/`ReduceSumTo`
///   keepdim) — what user graphs and pre-T5 lowerings contain;
/// * the RECIPE 9-node form (`MaxDim`/`SumDim` + `Unsqueeze` append, shared
///   `e`) — what [`recipe`]'s emission contains, so lower→fuse still
///   round-trips.
///
/// This is the matcher referenced from [`SubgraphPattern::Callable`] in the
/// registry entry.
pub fn canonical_pattern(graph: &Graph, div_id: NodeId) -> Option<PatternMatch> {
    legacy_spelled_pattern(graph, div_id).or_else(|| recipe_spelled_pattern(graph, div_id))
}

/// The legacy 7-node spelling (direct port of PR 3's
/// `canonical_softmax_pattern`).
fn legacy_spelled_pattern(graph: &Graph, div_id: NodeId) -> Option<PatternMatch> {
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

/// The recipe 9-node spelling (T5): `Div(e, Bcast(Unsq(SumDim(e))))` with
/// `e = Exp(Sub(x, Bcast(Unsq(MaxDim(x)))))` — both reduces target the last
/// axis, both `Unsqueeze`s restore keepdim (append at `rank − 1` of the
/// reduced tensor), both broadcasts restore x's full shape, and `e` is the
/// SHARED interior (consumed exactly twice: the denominator reduce + the
/// Div numerator).
fn recipe_spelled_pattern(graph: &Graph, div_id: NodeId) -> Option<PatternMatch> {
    let div = graph.node(div_id);
    if !matches!(div.op, Op::Div) { return None; }
    if div.inputs.len() != 2 { return None; }
    let e_id = div.inputs[0];
    let db_id = div.inputs[1];

    let db = graph.node(db_id);
    if !matches!(db.op, Op::BroadcastTo(_)) { return None; }
    if db.inputs.len() != 1 { return None; }
    let u2_id = db.inputs[0];

    let u2 = graph.node(u2_id);
    let Op::Unsqueeze { dim: u2_dim } = u2.op else { return None; };
    if u2.inputs.len() != 1 { return None; }
    let d_id = u2.inputs[0];

    let d = graph.node(d_id);
    let Op::SumDim(sum_axis) = d.op else { return None; };
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
    let u1_id = mb.inputs[0];

    let u1 = graph.node(u1_id);
    let Op::Unsqueeze { dim: u1_dim } = u1.op else { return None; };
    if u1.inputs.len() != 1 { return None; }
    let m_id = u1.inputs[0];

    let m = graph.node(m_id);
    let Op::MaxDim(max_axis) = m.op else { return None; };
    if m.inputs.len() != 1 || m.inputs[0] != x_id { return None; }

    // Axis discipline: both reduces AND both keepdim-restores target x's
    // LAST axis (the reduce drops rank by one, so the restoring Unsqueeze
    // appends at `last` of the reduced tensor — which IS `last` of x).
    let x_shape = &graph.node(x_id).shape;
    if x_shape.rank() == 0 { return None; }
    let last = x_shape.rank() - 1;
    if max_axis != last || sum_axis != last || u1_dim != last || u2_dim != last {
        return None;
    }
    // Both broadcasts restore x's full shape.
    let full = x_shape.dims();
    if graph.node(mb_id).shape.dims() != full { return None; }
    if graph.node(db_id).shape.dims() != full { return None; }

    // Conservativeness: every intermediate consumed only within this
    // pattern; e is consumed twice (→ d via SumDim, → div as numerator).
    let intermediates_with_expected_count = [
        (m_id, 1),
        (u1_id, 1),
        (mb_id, 1),
        (s_id, 1),
        (e_id, 2),
        (d_id, 1),
        (u2_id, 1),
        (db_id, 1),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Node;
    use fuel_ir::{DType, Shape};

    /// T5 re-fuse closure: the recipe emission itself is matched by
    /// `canonical_pattern` (the new-spelling arm), so lower → fuse still
    /// round-trips after the pilot migration.
    #[test]
    fn canonical_pattern_matches_the_recipe_spelling() {
        let mut g = Graph::new();
        let sh = Shape::from_dims(&[2, 4]);
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
        let fused = g.push(Node {
            op: Op::Fused(FusedOps::SOFTMAX_LAST_DIM, FusedOpParams::SoftmaxLastDim),
            inputs: vec![x],
            shape: sh,
            dtype: DType::F32,
        });
        let root = decompose(&mut g, fused, &FusedOpParams::SoftmaxLastDim);
        assert_ne!(root, fused, "recipe decompose fires");
        let m = canonical_pattern(&g, root).expect("the recipe emission must re-fuse");
        assert_eq!(m.bindings, vec![(0, x)], "bound external input = x");
        assert!(matches!(m.params, FusedOpParams::SoftmaxLastDim));
    }
}

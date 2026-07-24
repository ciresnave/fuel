//! RmsNormLastDim — `x / sqrt(mean(x²) + eps)` along the last dim.
//! Increment C slice 1, T7 — migrated (with LayerNormLastDim) to a portable
//! `PatternNode` DATA recipe, and the first op to carry an OPEN scalar slot
//! (the `eps` term).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (decompose function,
//!   pattern matcher, shape/dtype rules).
//! - [`recipe`] — the op's primitive subgraph as portable, shape-/rank-
//!   polymorphic data (7 nodes; the keepdim restore is the D3 shrink-via-swap
//!   `Unsqueeze` append, and the `eps` `AddScalar` is an OPEN slot filled by
//!   [`scalars`] from the params projection).
//! - [`scalars`] — the per-entry projection `RmsNormLastDim { eps } →
//!   vec![eps]` filling the recipe's one open slot.
//! - [`decompose`] — re-emits [`recipe`] through the
//!   [`crate::registry::decompose_via_recipe`] bridge.
//! - [`canonical_pattern`] — recognizes the decomposed subgraph in EITHER
//!   spelling (legacy `Reshape`-keepdim OR the T7 recipe `Unsqueeze`-append
//!   keepdim) and returns the bound `x` input plus the eps parameter. The
//!   recipe arm mirrors `softmax_last_dim`'s dual-spelling matcher, so the
//!   framework's OWN lowered subgraph re-fuses (not just user-spelled forms).
//!
//! The numeric relation is [`crate::Tensor::rms_norm_last_dim_decomposed`]'s
//! `Sqr → MeanDim → keepdim → AddScalar → Sqrt → BroadcastTo → Div`; the
//! keepdim node is now `Unsqueeze(append)` in place of `Reshape(keepdim)` (a
//! metadata-only D3 swap, numerically bit-exact). The matcher refuses to fire
//! when any intermediate has consumers outside the canonical pattern (same
//! conservatism as the SoftmaxLastDim matcher).

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

/// RmsNormLastDim's primitive recipe as **portable data** (Increment C slice
/// 1, T7 — the sixth op migrated, and the first with an OPEN scalar slot).
/// Shape-/rank-polymorphic: the mean reduces `MeanDim(axis_last)` and the
/// keepdim is restored by `Unsqueeze(axis_last = append)` (the D3 shrink-via-
/// swap replacing the baked `Reshape(keepdim)`); the final broadcast targets
/// `SameAs { operand: 0 }` (x's shape over the Bind space, D2). The `eps`
/// `AddScalar` carries EMPTY `scalars` — it is an OPEN slot, filled at emit
/// time from [`scalars`]'s projection (pattern pre-order, one slot). Nothing
/// in the datum bakes a shape, a rank, or the eps value. Bind: `0 = x`.
///
/// ```text
///   sq          = Sqr(x)
///   mean        = MeanDim(axis_last)(sq)             # rank-reduced
///   mean_kd     = Unsqueeze(axis_last = append)(mean)  # keepdim restore
///   denom_sq    = AddScalar[open slot](mean_kd)      # + eps
///   denom       = Sqrt(denom_sq)
///   denom_bcast = BroadcastTo(SameAs 0)(denom)
///   out         = Div(x, denom_bcast)
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
        op(OpTag::Div, OpAttrs::default(), vec![
            x(),
            op(OpTag::BroadcastTo, same_as_x(), vec![
                op(OpTag::Sqrt, OpAttrs::default(), vec![
                    // AddScalar with EMPTY scalars = the eps OPEN slot.
                    op(OpTag::AddScalar, OpAttrs::default(), vec![
                        op(OpTag::Unsqueeze, axis_last(), vec![
                            op(OpTag::MeanDim, axis_last(), vec![
                                op(OpTag::Sqr, OpAttrs::default(), vec![x()]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ])
    })
}

/// Per-entry scalar projection: RmsNormLastDim's one open slot is the `eps`
/// term, so the right payload projects to `vec![eps]` (one scalar) and any
/// other payload is a typed decline (`None` ⇒ the bridge returns the node
/// unchanged — G2).
fn scalars(params: &FusedOpParams) -> Option<Vec<f64>> {
    match params {
        FusedOpParams::RmsNormLastDim { eps } => Some(vec![*eps]),
        _ => None,
    }
}

/// Lower a fused RmsNormLastDim node to its primitive subgraph and return the
/// new root id — since T7 a re-emit of [`recipe`]'s data through the
/// [`decompose_via_recipe`] bridge (the fused node's single input is the bind
/// `[x]`; [`scalars`] fills the eps open slot; the resolving emit derives every
/// interior shape/dtype). Any failure — wrong params payload, a resolution
/// decline at these shapes (symbolic extent, …) — returns `id` (fixpoint,
/// surfaced gap, never a panic): exactly the G2 posture the imperative body
/// had.
///
/// The fused node `id` may be either `Op::RmsNormLastDim { eps }` (legacy
/// emission) or `Op::Fused(FusedOps::RMS_NORM_LAST_DIM,
/// FusedOpParams::RmsNormLastDim { eps })` (the builder path); the
/// decomposition is identical for both.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    decompose_via_recipe(graph, id, recipe(), scalars(params))
}

/// Match the canonical 7-node RmsNormLastDim subgraph rooted at the final
/// `Div` node, in EITHER spelling. Returns a [`PatternMatch`] binding `x` to
/// input 0 and stamping `FusedOpParams::RmsNormLastDim { eps }` extracted from
/// the AddScalar's scalar.
///
/// Two spellings, both kept (mirrors `softmax_last_dim`'s dual-spelling
/// matcher — never delete the old match):
/// * the LEGACY form (`Reshape` keepdim) — what user graphs and pre-T7
///   lowerings contain;
/// * the RECIPE form (`Unsqueeze` append keepdim, the D3 shrink-via-swap) —
///   what [`recipe`]'s emission contains, so lower → fuse round-trips on the
///   framework's OWN lowered subgraph.
///
/// The two spellings differ ONLY in the keepdim node (`Reshape` vs
/// `Unsqueeze`); every other node and every guard is identical, so the arms
/// share [`match_common`] and differ only in the keepdim predicate.
pub fn canonical_pattern(graph: &Graph, div_id: NodeId) -> Option<PatternMatch> {
    legacy_spelled_pattern(graph, div_id).or_else(|| recipe_spelled_pattern(graph, div_id))
}

/// The legacy keepdim spelling: `mean_kd = Reshape(mean)`.
fn legacy_spelled_pattern(graph: &Graph, div_id: NodeId) -> Option<PatternMatch> {
    match_common(graph, div_id, |op| matches!(op, Op::Reshape(_)))
}

/// The T7 recipe keepdim spelling: `mean_kd = Unsqueeze { dim }` (append at
/// the reduced tensor's last axis — the D3 shrink-via-swap).
fn recipe_spelled_pattern(graph: &Graph, div_id: NodeId) -> Option<PatternMatch> {
    match_common(graph, div_id, |op| matches!(op, Op::Unsqueeze { .. }))
}

/// Shared body of both spellings. `keepdim_ok` decides whether the keepdim
/// node (`mean_kd`) is the right kind for the spelling being matched; the
/// downstream shape guard (`mean_kd` = x's shape with last-dim = 1) is the
/// same for both, so `Reshape` and `Unsqueeze(append)` are validated
/// identically once the op-kind gate passes.
///
/// Conservative: every intermediate must be consumed only within the
/// canonical pattern so fusing doesn't discard a value the user reads from one
/// of the intermediates.
fn match_common(
    graph: &Graph,
    div_id: NodeId,
    keepdim_ok: impl Fn(&Op) -> bool,
) -> Option<PatternMatch> {
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
    if !keepdim_ok(&mean_kd.op) { return None; }
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
    // and the keepdim node's target must be the x shape with last-dim=1
    // (identical for the Reshape and Unsqueeze-append spellings).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Node;
    use fuel_ir::{DType, Shape};

    /// Slice-2 re-fuse closure (mirrors softmax's
    /// `canonical_pattern_matches_the_recipe_spelling`): the recipe emission
    /// itself must be matched by `canonical_pattern`, so lower → fuse still
    /// round-trips after the T7 keepdim swap.
    #[test]
    fn canonical_pattern_matches_the_recipe_spelling() {
        let mut g = Graph::new();
        let sh = Shape::from_dims(&[2, 4]);
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
        let params = FusedOpParams::RmsNormLastDim { eps: 1e-5 };
        let fused = g.push(Node {
            op: Op::Fused(FusedOps::RMS_NORM_LAST_DIM, params.clone()),
            inputs: vec![x],
            shape: sh,
            dtype: DType::F32,
        });
        let root = decompose(&mut g, fused, &params);
        assert_ne!(root, fused, "recipe decompose fires");
        let m = canonical_pattern(&g, root).expect("the recipe emission must re-fuse");
        assert_eq!(m.bindings, vec![(0, x)], "bound external input = x");
        match m.params {
            FusedOpParams::RmsNormLastDim { eps } => {
                assert_eq!(eps, 1e-5, "eps recovered from the AddScalar");
            }
            other => panic!("wrong params variant: {other:?}"),
        }
    }

    /// The legacy user-spelled `Reshape`-keepdim form keeps matching (risk-3
    /// posture: never delete the old match).
    #[test]
    fn canonical_pattern_still_matches_the_legacy_spelling() {
        let mut g = Graph::new();
        let sh = Shape::from_dims(&[2, 4]);
        let kd = Shape::from_dims(&[2, 1]);
        let f32_node = |op, inputs, shape| Node { op, inputs, shape, dtype: DType::F32 };
        let x = g.push(f32_node(Op::Const, vec![], sh.clone()));
        let sq = g.push(f32_node(Op::Sqr, vec![x], sh.clone()));
        let mean = g.push(f32_node(Op::MeanDim(1), vec![sq], kd.clone()));
        let mean_kd = g.push(f32_node(Op::Reshape(kd.clone()), vec![mean], kd.clone()));
        let denom_sq = g.push(f32_node(Op::AddScalar(1e-5), vec![mean_kd], kd.clone()));
        let denom = g.push(f32_node(Op::Sqrt, vec![denom_sq], kd));
        let denom_b = g.push(f32_node(Op::BroadcastTo(sh.clone()), vec![denom], sh.clone()));
        let div = g.push(f32_node(Op::Div, vec![x, denom_b], sh));
        let m = canonical_pattern(&g, div).expect("legacy spelling must still match");
        assert_eq!(m.bindings, vec![(0, x)]);
        assert!(matches!(m.params, FusedOpParams::RmsNormLastDim { eps } if eps == 1e-5));
    }
}

//! LayerNormLastDim — `(x - mean) / sqrt(variance + eps)` along the
//! last dim, no affine params. Increment C slice 1, T7 — migrated (with
//! RmsNormLastDim) to a portable `PatternNode` DATA recipe with an OPEN
//! `eps` scalar slot.
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (decompose function,
//!   pattern matcher, shape/dtype rules).
//! - [`recipe`] — the op's primitive subgraph as portable, shape-/rank-
//!   polymorphic data (11 nodes; the two keepdim restores are D3 shrink-via-
//!   swap `Unsqueeze` appends, the `centered` subterm is identity-shared, and
//!   the `eps` `AddScalar` is an OPEN slot filled from the params projection).
//! - [`scalars`] — the per-entry projection `LayerNormLastDim { eps } →
//!   vec![eps]`.
//! - [`decompose`] — re-emits [`recipe`] through the
//!   [`crate::registry::decompose_via_recipe`] bridge.
//! - [`canonical_pattern`] — recognizes the 11-node recipe subgraph (the T7
//!   `Unsqueeze`-append spelling, with the shared `centered` subterm) and
//!   returns the bound `x` input plus the eps parameter, so the framework's
//!   OWN lowered LayerNorm re-fuses. Mirrors the softmax / rms recipe matchers.
//!
//! The decomposition mirrors the standard layer-norm formula; the two keepdim
//! restores are now `Unsqueeze(append)` in place of `Reshape(keepdim)` (a
//! metadata-only D3 swap, numerically bit-exact):
//!
//! ```text
//!   mean        = MeanDim(axis_last)(x)                # rank-reduced
//!   mean_kd     = Unsqueeze(axis_last = append)(mean)
//!   mean_bcast  = BroadcastTo(SameAs 0)(mean_kd)
//!   centered    = Sub(x, mean_bcast)                   # SHARED subterm
//!   centered_sq = Sqr(centered)
//!   var         = MeanDim(axis_last)(centered_sq)      # rank-reduced
//!   var_kd      = Unsqueeze(axis_last = append)(var)
//!   var_eps     = AddScalar[open slot](var_kd)         # + eps
//!   denom       = Sqrt(var_eps)
//!   denom_bcast = BroadcastTo(SameAs 0)(denom)
//!   out         = Div(centered, denom_bcast)
//! ```

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

/// LayerNormLastDim's primitive recipe as **portable data** (Increment C
/// slice 1, T7). Shape-/rank-polymorphic: both keepdim restores are the D3
/// shrink-via-swap `Unsqueeze(axis_last = append)` (replacing the baked
/// `Reshape(keepdim)`), the two broadcasts target `SameAs { operand: 0 }`
/// (x's shape over the Bind space, D2), and the `eps` `AddScalar` carries
/// EMPTY `scalars` — an OPEN slot filled at emit time from [`scalars`]'s
/// projection. The `centered = Sub(x, mean_bcast)` subterm is a repeated
/// slot-free subtree the emitter identity-shares into ONE node (consumed by
/// both `Sqr` and the final `Div`), so the emitted graph is the 11-node DAG,
/// not a duplicated-compute tree. Nothing bakes a shape, a rank, or eps.
/// Bind: `0 = x`.
///
/// ```text
///   mean        = MeanDim(axis_last)(x)                # rank-reduced
///   mean_kd     = Unsqueeze(axis_last = append)(mean)
///   mean_bcast  = BroadcastTo(SameAs 0)(mean_kd)
///   centered    = Sub(x, mean_bcast)                   # SHARED
///   var         = MeanDim(axis_last)(Sqr(centered))    # rank-reduced
///   var_kd      = Unsqueeze(axis_last = append)(var)
///   var_eps     = AddScalar[open slot](var_kd)         # + eps
///   denom_bcast = BroadcastTo(SameAs 0)(Sqrt(var_eps))
///   out         = Div(centered, denom_bcast)
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
        // centered = Sub(x, BroadcastTo(SameAs 0)(Unsqueeze(MeanDim(x)))) —
        // the SHARED subterm (emitter identity-shares it into one node).
        let centered = op(OpTag::Sub, OpAttrs::default(), vec![
            x(),
            op(OpTag::BroadcastTo, same_as_x(), vec![
                op(OpTag::Unsqueeze, axis_last(), vec![
                    op(OpTag::MeanDim, axis_last(), vec![x()]),
                ]),
            ]),
        ]);
        // denom_bcast = BroadcastTo(SameAs 0)(Sqrt(AddScalar[eps](
        //   Unsqueeze(MeanDim(Sqr(centered)))))).
        let denom_bcast = op(OpTag::BroadcastTo, same_as_x(), vec![
            op(OpTag::Sqrt, OpAttrs::default(), vec![
                // AddScalar with EMPTY scalars = the eps OPEN slot.
                op(OpTag::AddScalar, OpAttrs::default(), vec![
                    op(OpTag::Unsqueeze, axis_last(), vec![
                        op(OpTag::MeanDim, axis_last(), vec![
                            op(OpTag::Sqr, OpAttrs::default(), vec![centered.clone()]),
                        ]),
                    ]),
                ]),
            ]),
        ]);
        op(OpTag::Div, OpAttrs::default(), vec![centered, denom_bcast])
    })
}

/// Per-entry scalar projection: LayerNormLastDim's one open slot is the `eps`
/// term, so the right payload projects to `vec![eps]` and any other payload is
/// a typed decline (`None` ⇒ the bridge returns the node unchanged — G2).
fn scalars(params: &FusedOpParams) -> Option<Vec<f64>> {
    match params {
        FusedOpParams::LayerNormLastDim { eps } => Some(vec![*eps]),
        _ => None,
    }
}

/// Lower a fused LayerNormLastDim node to its primitive subgraph and return the
/// new root id — since T7 a re-emit of [`recipe`]'s data through the
/// [`decompose_via_recipe`] bridge (the fused node's single input is the bind
/// `[x]`; [`scalars`] fills the eps open slot; the resolving emit derives every
/// interior shape/dtype and identity-shares `centered`). Any failure — wrong
/// params payload, a resolution decline at these shapes — returns `id`
/// (fixpoint, surfaced gap, never a panic): exactly the G2 posture the
/// imperative body had.
///
/// The fused node `id` may be either `Op::LayerNormLastDim { eps }` (legacy
/// emission) or `Op::Fused(FusedOps::LAYER_NORM_LAST_DIM,
/// FusedOpParams::LayerNormLastDim { eps })` (the builder path).
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    decompose_via_recipe(graph, id, recipe(), scalars(params))
}

/// Match the canonical 11-node LayerNormLastDim subgraph rooted at the final
/// `Div` node, in the T7 recipe spelling (`Unsqueeze`-append keepdim, the D3
/// shrink-via-swap). Returns a [`PatternMatch`] binding `x` to input 0 and
/// stamping `FusedOpParams::LayerNormLastDim { eps }` extracted from the
/// AddScalar's scalar.
///
/// Slice-2: this replaces the earlier placeholder (which returned `None` for
/// every input, so the framework's OWN lowered LayerNorm never re-fused). It
/// mirrors the `softmax_last_dim` / `rms_norm_last_dim` recipe matchers; the
/// only structural addition is the SHARED `centered = Sub(x, mean_bcast)`
/// subterm, consumed twice (by `Sqr` and by the final `Div`), so its
/// single-consumer guard is a count of **2** and the `Sqr`'s input must be the
/// SAME `centered` node the `Div` numerator reads (the identity check that
/// distinguishes layer-norm from an unrelated `(a - b) / sqrt(c + eps)`).
///
/// The emitted subgraph (x is the external bind):
///
/// ```text
///   mean        = MeanDim(last)(x)
///   mean_kd     = Unsqueeze(append)(mean)
///   mean_bcast  = BroadcastTo(x)(mean_kd)
///   centered    = Sub(x, mean_bcast)            # SHARED (consumers = 2)
///   csq         = Sqr(centered)
///   var         = MeanDim(last)(csq)
///   var_kd      = Unsqueeze(append)(var)
///   var_eps     = AddScalar[eps](var_kd)
///   denom       = Sqrt(var_eps)
///   denom_bcast = BroadcastTo(x)(denom)
///   out         = Div(centered, denom_bcast)
/// ```
///
/// Conservative: every intermediate must be consumed only within the canonical
/// pattern (`centered` exactly twice, all others exactly once) so fusing
/// doesn't discard a value the user reads. There is no legacy `Reshape`
/// spelling to preserve — the LayerNorm decompose has emitted `Unsqueeze`
/// since T7 — so this is a single-spelling matcher (unlike the dual-spelling
/// softmax/rms matchers).
pub fn canonical_pattern(graph: &Graph, div_id: NodeId) -> Option<PatternMatch> {
    let div = graph.node(div_id);
    if !matches!(div.op, Op::Div) { return None; }
    if div.inputs.len() != 2 { return None; }
    let centered_id = div.inputs[0];
    let denom_bcast_id = div.inputs[1];

    // --- centered = Sub(x, mean_bcast) branch (the SHARED subterm) ---
    let centered = graph.node(centered_id);
    if !matches!(centered.op, Op::Sub) { return None; }
    if centered.inputs.len() != 2 { return None; }
    let x_id = centered.inputs[0];
    let mean_bcast_id = centered.inputs[1];

    let mean_bcast = graph.node(mean_bcast_id);
    if !matches!(mean_bcast.op, Op::BroadcastTo(_)) { return None; }
    if mean_bcast.inputs.len() != 1 { return None; }
    let mean_kd_id = mean_bcast.inputs[0];

    let mean_kd = graph.node(mean_kd_id);
    if !matches!(mean_kd.op, Op::Unsqueeze { .. }) { return None; }
    if mean_kd.inputs.len() != 1 { return None; }
    let mean_id = mean_kd.inputs[0];

    let mean = graph.node(mean_id);
    let mean_axis = match mean.op {
        Op::MeanDim(d) => d,
        _ => return None,
    };
    if mean.inputs.len() != 1 { return None; }
    // The mean is over the SAME x that `centered` subtracts from.
    if mean.inputs[0] != x_id { return None; }

    // --- denom_bcast = BroadcastTo(Sqrt(AddScalar[eps](Unsqueeze(MeanDim(Sqr(centered)))))) ---
    let denom_bcast = graph.node(denom_bcast_id);
    if !matches!(denom_bcast.op, Op::BroadcastTo(_)) { return None; }
    if denom_bcast.inputs.len() != 1 { return None; }
    let denom_id = denom_bcast.inputs[0];

    let denom = graph.node(denom_id);
    if !matches!(denom.op, Op::Sqrt) { return None; }
    if denom.inputs.len() != 1 { return None; }
    let var_eps_id = denom.inputs[0];

    let var_eps = graph.node(var_eps_id);
    let eps = match var_eps.op {
        Op::AddScalar(e) => e,
        _ => return None,
    };
    if var_eps.inputs.len() != 1 { return None; }
    let var_kd_id = var_eps.inputs[0];

    let var_kd = graph.node(var_kd_id);
    if !matches!(var_kd.op, Op::Unsqueeze { .. }) { return None; }
    if var_kd.inputs.len() != 1 { return None; }
    let var_id = var_kd.inputs[0];

    let var = graph.node(var_id);
    let var_axis = match var.op {
        Op::MeanDim(d) => d,
        _ => return None,
    };
    if var.inputs.len() != 1 { return None; }
    let csq_id = var.inputs[0];

    let csq = graph.node(csq_id);
    if !matches!(csq.op, Op::Sqr) { return None; }
    if csq.inputs.len() != 1 { return None; }
    // The SHARED-subterm identity: `Sqr` must square the SAME `centered` node
    // the `Div` numerator reads — otherwise it isn't layer-norm.
    if csq.inputs[0] != centered_id { return None; }

    // Shape / axis discipline: both reduces target x's LAST axis, and both
    // keepdim restores yield x's shape with last-dim = 1. Both broadcasts
    // restore x's full shape.
    let x_shape = &graph.node(x_id).shape;
    if x_shape.rank() == 0 { return None; }
    let last = x_shape.rank() - 1;
    if mean_axis != last || var_axis != last { return None; }
    let full = x_shape.dims().to_vec();
    for kd_id in [mean_kd_id, var_kd_id] {
        let kd_shape = &graph.node(kd_id).shape;
        if kd_shape.rank() != x_shape.rank() { return None; }
        for axis in 0..x_shape.rank() {
            let expected = if axis == last { 1 } else { full[axis] };
            if kd_shape.dims()[axis] != expected { return None; }
        }
    }
    if graph.node(mean_bcast_id).shape.dims() != full.as_slice() { return None; }
    if graph.node(denom_bcast_id).shape.dims() != full.as_slice() { return None; }

    // Conservativeness: every intermediate consumed only within this pattern —
    // `centered` exactly twice (→ Sqr and → Div), all others exactly once.
    let intermediates_with_expected_count = [
        (mean_id, 1),
        (mean_kd_id, 1),
        (mean_bcast_id, 1),
        (centered_id, 2),
        (csq_id, 1),
        (var_id, 1),
        (var_kd_id, 1),
        (var_eps_id, 1),
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
        params:   FusedOpParams::LayerNormLastDim { eps },
    })
}

/// Build a consumer-count index across the entire graph. Mirrors the helper in
/// `softmax_last_dim` / `rms_norm_last_dim`; replicated here so the matcher is
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
    use crate::{Node, Op};
    use fuel_ir::{DType, Shape};

    /// Slice-2 re-fuse closure (mirrors softmax / rms): the recipe emission
    /// must be matched by `canonical_pattern`, so lower → fuse round-trips on
    /// the framework's OWN lowered LayerNorm subgraph.
    #[test]
    fn canonical_pattern_matches_the_recipe_spelling() {
        let mut g = Graph::new();
        let sh = Shape::from_dims(&[2, 4]);
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
        let params = FusedOpParams::LayerNormLastDim { eps: 1e-5 };
        let fused = g.push(Node {
            op: Op::Fused(FusedOps::LAYER_NORM_LAST_DIM, params.clone()),
            inputs: vec![x],
            shape: sh,
            dtype: DType::F32,
        });
        let root = decompose(&mut g, fused, &params);
        assert_ne!(root, fused, "recipe decompose fires");
        let m = canonical_pattern(&g, root).expect("the recipe emission must re-fuse");
        assert_eq!(m.bindings, vec![(0, x)], "bound external input = x");
        match m.params {
            FusedOpParams::LayerNormLastDim { eps } => {
                assert_eq!(eps, 1e-5, "eps recovered from the AddScalar");
            }
            other => panic!("wrong params variant: {other:?}"),
        }
    }
}

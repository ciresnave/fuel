//! LayerNormLastDimBackward — fused backward helper for the
//! `LayerNormLastDim` forward. Phase 7.6 step 4; migrated to a portable
//! `PatternNode` DATA recipe in Increment C slice 2 (S2-1).
//!
//! Recomputes mean/variance from the original x (rather than caching
//! intermediates from forward) and produces
//! `grad_x = istd · (g − mean(g) − xhat·mean(g·xhat))`, where `mean(·)` is over
//! the last dim, `xhat = (x − mean(x))·istd`, and `istd = rsqrt(var + eps)`.
//! Two inputs `[x, upstream]` + `eps`.
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry`.
//! - [`recipe`] — the op's closed-form backward as portable, shape-/rank-
//!   polymorphic `PatternNode` data (Increment C slice 2). The four keepdim
//!   restores are the D3 shrink-via-swap (`MeanDim(axis_last)` + `Unsqueeze(
//!   axis_last = append)`, replacing the baked `Reshape(keepdim)`) and each
//!   broadcast targets `SameAs { operand: 0 }` (x's full shape, D2). One `eps`
//!   `AddScalar` OPEN slot, filled from the params projection.
//! - [`scalars`] — the per-entry projection `LayerNormLastDimBackward { eps } →
//!   vec![eps; open-slot-count]` (one eps per open slot — see the RISK-A note
//!   on [`recipe`]).
//! - [`decompose`] — re-emits [`recipe`] through the
//!   [`crate::registry::decompose_via_recipe`] bridge.
//!
//! See `softmax_last_dim_backward.rs` for the architectural rationale shared by
//! all four backward-helper entries: the registry's role is identity +
//! `BackwardKind::Fused` target + future `BackendImpl` host, and the matcher is
//! always stubbed (backward-helper nodes originate from autograd emitting
//! `Op::Fused(LAYER_NORM_LAST_DIM_BACKWARD, _)`, not from user-decomposed
//! forms). The `decompose` IS a "synthesize from primitives" recipe — every op
//! is a functional primitive, so per G2 it decomposes totally, never a
//! basis-gap self-return.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::shape_expr::ShapeExpr;
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};
use std::sync::OnceLock;

/// Metadata-side registry entry for LayerNormLastDimBackward.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: None,
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

/// Shape rule: output equals input 0 (the forward layer-norm input `x`).
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 2,
        "LayerNormLastDimBackward takes 2 inputs (x, upstream)",
    );
    input_shapes[0].clone()
}

/// Dtype rule: output dtype equals input 0.
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "LayerNormLastDimBackward takes 2 inputs",
    );
    input_dtypes[0]
}

/// LayerNormLastDimBackward's primitive recipe as **portable data** (Increment
/// C slice 2, S2-1). Shape-/rank-polymorphic: every last-axis reduce is
/// `MeanDim(axis_last)` and the keepdim is restored by `Unsqueeze(axis_last =
/// append)` (the D3 shrink-via-swap replacing the baked `Reshape(keepdim)`);
/// each broadcast targets `SameAs { operand: 0 }` — x's full shape over the
/// Bind space (D2). Binds: `0 = x`, `1 = g` (the upstream gradient) — the order
/// the autograd `BackwardKind::Fused` edge emits (`inputs = [x, upstream]`).
///
/// The `eps` `AddScalar` carries EMPTY `scalars` — an OPEN slot filled at emit
/// time from [`scalars`]'s projection. D4 never fires: the `Unsqueeze` rebuilds
/// rank BEFORE every broadcast, so no leading-1 pad `Reshape` is materialized
/// (equal rank throughout).
///
/// Emitted form (`mean_b(v) = BroadcastTo(SameAs 0)(Unsqueeze(MeanDim(v)))`):
///
/// ```text
///   xc       = Sub(x, mean_b(x))                 # centered
///   var_eps  = AddScalar[open slot](mean_b(Sqr(xc)))
///   istd     = Rsqrt(var_eps)
///   xhat     = Mul(xc, istd)
///   g_xhat   = Mul(g, xhat)
///   t1       = Sub(g, mean_b(g))
///   t2       = Mul(xhat, mean_b(g_xhat))
///   grad_x   = Mul(istd, Sub(t1, t2))            # = istd·(g − mean(g) − xhat·mean(g·xhat))
/// ```
///
/// **RISK-A (accepted redundancy, documented — no emit extension in this
/// slice).** `xhat` (and its `istd`) carry the eps OPEN slot, so the emitter's
/// slot-free identity-share does NOT dedup them: `istd` occurs at three
/// semantic use sites (`grad_x`, and each of `xhat`'s two consumers) and `xhat`
/// at two, so the datum tree holds THREE `AddScalar` open slots and the emitted
/// base map re-computes `xhat`/`istd`/`var_eps` per occurrence (heavier than the
/// let-shared imperative DAG — verified 27 vs 22 primitive nodes). This is
/// numerically identical (deterministic recompute) and the downstream
/// full-optimizer CSE re-collapses the duplicates back to the shared DAG
/// (both verified in `runtime_fused`'s
/// `layer_norm_backward_recipe_open_slot_xhat_is_recomputed_then_cse_recollapses`).
/// An open-slot subtree-sharing emit extension is deliberately out of scope for
/// this slice (a flagged future extension).
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
        let g = || PatternNode::Bind { index: 1 };
        // mean_b(src) = BroadcastTo(SameAs 0)(Unsqueeze(append)(MeanDim(last)(src)))
        let mean_b = |src: PatternNode| -> PatternNode {
            op(OpTag::BroadcastTo, same_as_x(), vec![
                op(OpTag::Unsqueeze, axis_last(), vec![
                    op(OpTag::MeanDim, axis_last(), vec![src]),
                ]),
            ])
        };
        // xc = Sub(x, mean_b(x))  — the centered term (slot-free, shared).
        let xc = op(OpTag::Sub, OpAttrs::default(), vec![x(), mean_b(x())]);
        // var_eps = AddScalar[open](mean_b(Sqr(xc)));  istd = Rsqrt(var_eps).
        // Both carry the eps slot, so each `.clone()` re-emits (RISK-A).
        let istd = || {
            op(OpTag::Rsqrt, OpAttrs::default(), vec![
                op(OpTag::AddScalar, OpAttrs::default(), vec![
                    mean_b(op(OpTag::Sqr, OpAttrs::default(), vec![xc.clone()])),
                ]),
            ])
        };
        // xhat = Mul(xc, istd)  — two consumers (g_xhat, t2).
        let xhat = || op(OpTag::Mul, OpAttrs::default(), vec![xc.clone(), istd()]);
        // g_xhat = Mul(g, xhat).
        let g_xhat = op(OpTag::Mul, OpAttrs::default(), vec![g(), xhat()]);
        // t1 = Sub(g, mean_b(g));  t2 = Mul(xhat, mean_b(g_xhat)).
        let t1 = op(OpTag::Sub, OpAttrs::default(), vec![g(), mean_b(g())]);
        let t2 = op(OpTag::Mul, OpAttrs::default(), vec![xhat(), mean_b(g_xhat)]);
        let inner = op(OpTag::Sub, OpAttrs::default(), vec![t1, t2]);
        // grad_x = Mul(istd, inner).
        op(OpTag::Mul, OpAttrs::default(), vec![istd(), inner])
    })
}

/// Per-entry scalar projection: LayerNormLastDimBackward's open slots are all
/// the `eps` term (one per `AddScalar` occurrence in the datum — see the RISK-A
/// note on [`recipe`]), so the right payload projects to `vec![eps; slots]` and
/// any other payload is a typed decline (`None` ⇒ the bridge returns the node
/// unchanged — G2). Deriving the count from [`recipe`]'s open-slot count keeps
/// the projection in lockstep with the datum's structure (all slots take the
/// same eps value, so pre-order alignment is trivially satisfied).
fn scalars(params: &FusedOpParams) -> Option<Vec<f64>> {
    match params {
        FusedOpParams::LayerNormLastDimBackward { eps } => {
            Some(vec![*eps; crate::runtime_fused::count_scalar_slots(recipe())])
        }
        _ => None,
    }
}

/// Decompose to the affine-free LayerNorm backward
/// `grad_x = istd · (g − mean(g) − xhat·mean(g·xhat))` — since S2-1 a re-emit of
/// [`recipe`]'s data through the [`decompose_via_recipe`] bridge (the fused
/// node's two inputs `[x, g]` are the binds; [`scalars`] fills the eps open
/// slots; the resolving emit derives every interior shape/dtype). Any failure —
/// wrong params payload, a resolution decline at these shapes (symbolic extent,
/// …) — returns `id` (fixpoint, surfaced gap, never a panic): exactly the G2
/// posture the imperative body had.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    decompose_via_recipe(graph, id, recipe(), scalars(params))
}

/// Matcher stub — backward-helper nodes originate from autograd emitting
/// `Op::Fused(LAYER_NORM_LAST_DIM_BACKWARD, _)`, not from user-decomposed forms.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

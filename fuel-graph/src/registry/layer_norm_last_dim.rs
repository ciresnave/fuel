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
//! - [`canonical_pattern`] — placeholder returning `None` (unchanged; a
//!   canonical matcher for the LayerNorm subgraph stays follow-up work).
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
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::shape_expr::ShapeExpr;
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};
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

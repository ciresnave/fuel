//! RmsNormLastDimBackward — fused backward helper for the
//! `RmsNormLastDim` forward. Phase 7.6 step 4; migrated to a portable
//! `PatternNode` DATA recipe in Increment C carriers (A1 — the first recipe
//! with a SHAPE-DERIVED scalar slot).
//!
//! Closed-form: `grad_x = r_rms · (upstream - x · s / (n · (mean_sq + eps)))`
//! where `r_rms = rsqrt(mean_sq + eps)`, `mean_sq = mean(x², last)`,
//! `s = sum(upstream · x, last)`, `n = last_dim_size`. Two inputs
//! `[x, upstream]` + `eps`.
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry`.
//! - [`recipe`] — the op's closed-form backward as portable, shape-/rank-
//!   polymorphic `PatternNode` data (Increment C carriers, A1). Sibling of the
//!   layer-norm-backward recipe: every last-axis reduce is `MeanDim`/`SumDim(
//!   axis_last)` with the keepdim restored by `Unsqueeze(axis_last = append)`
//!   (the D3 shrink-via-swap replacing the baked `Reshape(keepdim)`), and each
//!   broadcast targets `SameAs { operand: 0 }` (x's full shape, D2). The `eps`
//!   `AddScalar` is an OPEN slot; the reduced-count divisor `n = dims[last]` is
//!   a `MulScalar(scalar_rel = Extent(0, LAST))` — filled from x's shape at emit
//!   time, NOT from the params projection (the A1 carrier).
//! - [`scalars`] — the per-entry projection `RmsNormLastDimBackward { eps } →
//!   vec![eps; open-slot-count]` (one eps per open slot — see the RISK-A note
//!   on [`recipe`]).
//! - [`decompose`] — re-emits [`recipe`] through the
//!   [`crate::registry::decompose_via_recipe`] bridge.
//!
//! See `softmax_last_dim_backward.rs` for the architectural rationale shared by
//! all four backward-helper entries: the registry's role is identity +
//! `BackwardKind::Fused` target + future `BackendImpl` host, and the matcher is
//! always stubbed (backward-helper nodes originate from autograd emitting
//! `Op::Fused(RMS_NORM_LAST_DIM_BACKWARD, _)`, not from user-decomposed forms).
//! The `decompose` IS a "synthesize from primitives" recipe — every op is a
//! functional primitive, so per G2 it decomposes totally, never a basis-gap
//! self-return.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps, PatternMatch,
    SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::shape_expr::{Dim, LAST, ShapeExpr};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};
use std::sync::OnceLock;

/// Metadata-side registry entry for RmsNormLastDimBackward.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: None,
        id: FusedOps::RMS_NORM_LAST_DIM_BACKWARD,
        name: "RmsNormLastDimBackward",
        family: FusedOpFamily::Backward,
        pattern: SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward: BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Shape rule: output equals input 0 (the forward rms-norm input `x`).
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(),
        2,
        "RmsNormLastDimBackward takes 2 inputs (x, upstream)",
    );
    input_shapes[0].clone()
}

/// Dtype rule: output dtype equals input 0.
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(),
        2,
        "RmsNormLastDimBackward takes 2 inputs",
    );
    input_dtypes[0]
}

/// RmsNormLastDimBackward's primitive recipe as **portable data** (Increment C
/// carriers, A1). Shape-/rank-polymorphic: every last-axis reduce is
/// `MeanDim`/`SumDim(axis_last)` and the keepdim is restored by `Unsqueeze(
/// axis_last = append)` (the D3 shrink-via-swap replacing the baked
/// `Reshape(keepdim)`); each broadcast targets `SameAs { operand: 0 }` — x's
/// full shape over the Bind space (D2). Binds: `0 = x`, `1 = up` (the upstream
/// gradient) — the order the autograd `BackwardKind::Fused` edge emits
/// (`inputs = [x, upstream]`).
///
/// Two kinds of non-baked scalar ride this datum:
/// * the `eps` `AddScalar` carries EMPTY `scalars` — an OPEN slot filled at emit
///   time from [`scalars`]'s params projection.
/// * the reduced-count divisor `n = dims[last]` is a `MulScalar(scalar_rel =
///   Extent(0, LAST))` — filled at emit time from x's SHAPE (the A1 carrier),
///   NOT from the params cursor, so it is never counted as an open slot.
///
/// D4 never fires: the `Unsqueeze` rebuilds rank BEFORE every broadcast, so no
/// leading-1 pad `Reshape` is materialized (equal rank throughout).
///
/// Emitted form (`reduce_kd(tag, v) = Unsqueeze(append)(tag(v))`,
/// `bcast(v) = BroadcastTo(SameAs 0)(v)`):
///
/// ```text
///   denom_kd = AddScalar[open slot](reduce_kd(MeanDim, Sqr(x)))  # mean(x²)+eps
///   rrms_b   = bcast(Rsqrt(denom_kd))
///   s_b      = bcast(reduce_kd(SumDim, Mul(up, x)))              # sum(up·x)
///   ndenom_b = bcast(MulScalar[n = Extent(0,LAST)](denom_kd))    # n·denom
///   term     = Div(Mul(x, s_b), ndenom_b)                        # x·s / (n·denom)
///   grad_x   = Mul(rrms_b, Sub(up, term))                        # r_rms·(g − term)
/// ```
///
/// **RISK-A (accepted redundancy, documented — no emit extension in this
/// slice).** `denom_kd` carries the eps OPEN slot, so the emitter's slot-free
/// identity-share does NOT dedup it: it occurs at TWO semantic use sites
/// (`rrms_b`'s `Rsqrt` and `ndenom_b`'s `MulScalar`), so the datum tree holds
/// TWO `AddScalar` open slots and the emitted base map re-computes
/// `mean_sq + eps` per occurrence (heavier than the let-shared imperative DAG,
/// which shares the single `denom_kd`). This is numerically identical
/// (deterministic recompute) and the downstream full-optimizer CSE re-collapses
/// the duplicates back to the shared DAG. An open-slot subtree-sharing emit
/// extension is deliberately out of scope for this slice (a flagged future
/// extension). Mirrors the layer-norm-backward recipe's RISK-A posture.
fn recipe() -> &'static PatternNode {
    static RECIPE: OnceLock<PatternNode> = OnceLock::new();
    RECIPE.get_or_init(|| {
        let axis_last = || OpAttrs {
            axis_last: true,
            ..OpAttrs::default()
        };
        let same_as_x = || OpAttrs {
            target_shape_rel: Some(ShapeExpr::SameAs { operand: 0 }),
            ..OpAttrs::default()
        };
        // n = extent of x's last axis (the reduced_count), filled from x's shape
        // at emit time — the A1 shape-derived scalar carrier.
        let n_scalar = || OpAttrs {
            scalar_rel: Some(Dim::Extent {
                operand: 0,
                axis: LAST,
            }),
            ..OpAttrs::default()
        };
        let op = |op, attrs, operands| PatternNode::Op {
            op,
            attrs,
            operands,
        };
        let x = || PatternNode::Bind { index: 0 };
        let up = || PatternNode::Bind { index: 1 };
        // reduce_kd(tag, src) = Unsqueeze(append)(tag(axis_last)(src)) — the
        // rank-reduced last-axis reduce with the keepdim restored (D3 swap).
        let reduce_kd = |tag, src: PatternNode| -> PatternNode {
            op(
                OpTag::Unsqueeze,
                axis_last(),
                vec![op(tag, axis_last(), vec![src])],
            )
        };
        // bcast(src) = BroadcastTo(SameAs 0)(src) — back to x's full shape (D2).
        let bcast =
            |src: PatternNode| -> PatternNode { op(OpTag::BroadcastTo, same_as_x(), vec![src]) };
        // denom_kd = AddScalar[open eps](reduce_kd(MeanDim, Sqr(x))); carries the
        // eps OPEN slot, so each occurrence re-emits (RISK-A — two use sites).
        let denom_kd = || {
            op(
                OpTag::AddScalar,
                OpAttrs::default(),
                vec![reduce_kd(
                    OpTag::MeanDim,
                    op(OpTag::Sqr, OpAttrs::default(), vec![x()]),
                )],
            )
        };
        // r_rms broadcast = bcast(Rsqrt(denom_kd)).
        let rrms_b = bcast(op(OpTag::Rsqrt, OpAttrs::default(), vec![denom_kd()]));
        // s broadcast = bcast(reduce_kd(SumDim, Mul(up, x))).
        let s_b = bcast(reduce_kd(
            OpTag::SumDim,
            op(OpTag::Mul, OpAttrs::default(), vec![up(), x()]),
        ));
        // n·denom broadcast = bcast(MulScalar[n = shape-derived](denom_kd)).
        let ndenom_b = bcast(op(OpTag::MulScalar, n_scalar(), vec![denom_kd()]));
        // term = Div(Mul(x, s_b), ndenom_b).
        let xs = op(OpTag::Mul, OpAttrs::default(), vec![x(), s_b]);
        let term = op(OpTag::Div, OpAttrs::default(), vec![xs, ndenom_b]);
        // grad_x = Mul(rrms_b, Sub(up, term)).
        let inner = op(OpTag::Sub, OpAttrs::default(), vec![up(), term]);
        op(OpTag::Mul, OpAttrs::default(), vec![rrms_b, inner])
    })
}

/// Per-entry scalar projection: RmsNormLastDimBackward's open slots are all the
/// `eps` term (one per `AddScalar` occurrence in the datum — see the RISK-A note
/// on [`recipe`]; the reduced-count `MulScalar` is a shape-derived `scalar_rel`,
/// NOT an open slot, so it is excluded from the count). The right payload
/// projects to `vec![eps; slots]` and any other payload is a typed decline
/// (`None` ⇒ the bridge returns the node unchanged — G2). Deriving the count
/// from [`recipe`]'s open-slot count keeps the projection in lockstep with the
/// datum's structure (all slots take the same eps value, so pre-order alignment
/// is trivially satisfied).
fn scalars(params: &FusedOpParams) -> Option<Vec<f64>> {
    match params {
        FusedOpParams::RmsNormLastDimBackward { eps } => Some(vec![
                *eps;
                crate::runtime_fused::count_scalar_slots(recipe())
            ]),
        _ => None,
    }
}

/// Decompose to the closed-form RMSNorm backward
/// `grad_x = r_rms · (g − x·s / (n·(mean_sq + eps)))` — since A1 a re-emit of
/// [`recipe`]'s data through the [`decompose_via_recipe`] bridge (the fused
/// node's two inputs `[x, up]` are the binds; [`scalars`] fills the eps open
/// slots; the `MulScalar(n)` divisor resolves from x's shape via the
/// `scalar_rel` carrier; the resolving emit derives every interior shape/dtype).
/// Any failure — wrong params payload, a resolution decline at these shapes
/// (symbolic extent, …) — returns `id` (fixpoint, surfaced gap, never a panic):
/// exactly the G2 posture the imperative body had.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    decompose_via_recipe(graph, id, recipe(), scalars(params))
}

/// Matcher stub — backward-helper nodes originate from autograd emitting
/// `Op::Fused(RMS_NORM_LAST_DIM_BACKWARD, _)`, not from user-decomposed forms.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

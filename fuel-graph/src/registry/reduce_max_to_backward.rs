//! ReduceMaxToBackward — fused backward helper for the `Op::ReduceMaxTo`
//! primitive. Phase 7.6 step 4; migrated to a portable `PatternNode` DATA
//! recipe in Increment C carriers (A2 — the first recipe to carry a
//! `MaskedFill`, via the A2 fill-Scalar re-emit carrier).
//!
//! Routes the upstream gradient to position(s) where x equals its
//! per-window max; tied maxes share equally (fair-share subgradient).
//! Two inputs `[x, upstream]`, parameterless.
//!
//! Unique among the four backward-helper entries: this one is the
//! backward of a **primitive** (`Op::ReduceMaxTo`), not of a fused
//! forward. There is no `BackwardKind::Fused(REDUCE_MAX_TO_BACKWARD)`
//! edge from any forward entry — autograd reaches this helper
//! directly from `Op::ReduceMaxTo`'s arm in `Tensor::backward`. The
//! registry entry still exists so the executor's dispatch stays
//! consistent (every fused op routes through `Op::Fused(id, _)`) and
//! so step 5 can drop the legacy `Op::ReduceMaxToBackward` variant.
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry`.
//! - [`recipe`] — the fair-share max subgradient as portable, shape-polymorphic
//!   `PatternNode` data (Increment C carriers, A2).
//! - [`decompose`] — re-emits [`recipe`] through the
//!   [`crate::registry::decompose_via_recipe`] bridge.
//!
//! See `softmax_last_dim_backward.rs` for the shared architectural
//! rationale around backward-helper entries (identity + future `BackendImpl`
//! host; the matcher is always stubbed — autograd emits the fused node).

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::shape_expr::ShapeExpr;
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};
use std::sync::OnceLock;

/// Metadata-side registry entry for ReduceMaxToBackward.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::REDUCE_MAX_TO_BACKWARD,
        name:       "ReduceMaxToBackward",
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
        "ReduceMaxToBackward takes 2 inputs (x, upstream)",
    );
    // grad_x has the original x's shape (input 0); upstream's shape
    // is the forward target_shape.
    input_shapes[0].clone()
}

fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "ReduceMaxToBackward takes 2 inputs",
    );
    input_dtypes[0]
}

/// ReduceMaxToBackward's primitive recipe as **portable data** (Increment C
/// carriers, A2). Shape-polymorphic: the reduce/count targets are `SameAs {
/// operand: 1 }` (upstream's shape = the forward reduce target) and the two
/// broadcasts target `SameAs { operand: 0 }` (x's full shape) — all D2, nothing
/// baked. Binds: `0 = x`, `1 = up` (the upstream gradient) — the order autograd
/// emits (`inputs = [x, up_id]`).
///
/// Routes `upstream` to the position(s) where `x` equals its per-window max;
/// tied maxes share equally. The U8 `mask = (x == max)` becomes a float mask via
/// `MaskedFill(value=1, zeros, mask)` — the A2 carrier (this avoids an integer→
/// float `Cast` the CPU cast matrix doesn't carry). The `MaskedFill` fill Scalar
/// is authored dtype-polymorphically (baked value `1.0`, no `cast_dtype`), so
/// emit re-resolves it to x's dtype — bit-identical to the imperative
/// `Scalar::one(dtype)`.
///
/// `mask_f` is a SLOT-FREE subtree (every scalar baked), so emit's identity-share
/// dedups its TWO use sites (`ReduceSumTo` and the final `Mul`) to ONE emitted
/// node — the same single-compute DAG the imperative builder had. This is a
/// DIRECT structural mirror: no D3 keepdim swap and no D4 pad fire (the reduces
/// are `ReduceMaxTo`/`ReduceSumTo` which carry the kept shape, and every
/// broadcast is equal-rank), so the recipe base map matches the legacy body
/// exactly.
///
/// Emitted form (`bcast_x(v) = BroadcastTo(SameAs 0)(v)`,
/// `to_up(tag, v) = tag(SameAs 1)(v)`):
///
/// ```text
///   y       = to_up(ReduceMaxTo, x)          # per-window max (upstream shape)
///   mask_u8 = Equal(x, bcast_x(y))           # U8: 1 where x == max
///   mask_f  = MaskedFill[1.0](MulScalar[0](x), mask_u8)   # float mask (x-dtype)
///   ties    = to_up(ReduceSumTo, mask_f)     # count per window
///   share   = Div(up, ties)                  # fair share for ties
///   grad_x  = Mul(mask_f, bcast_x(share))
/// ```
fn recipe() -> &'static PatternNode {
    static RECIPE: OnceLock<PatternNode> = OnceLock::new();
    RECIPE.get_or_init(|| {
        let same_as_x = || OpAttrs {
            target_shape_rel: Some(ShapeExpr::SameAs { operand: 0 }),
            ..OpAttrs::default()
        };
        let same_as_up = || OpAttrs {
            target_shape_rel: Some(ShapeExpr::SameAs { operand: 1 }),
            ..OpAttrs::default()
        };
        let op = |op, attrs, operands| PatternNode::Op { op, attrs, operands };
        let x = || PatternNode::Bind { index: 0 };
        let up = || PatternNode::Bind { index: 1 };
        let bcast_x = |v: PatternNode| op(OpTag::BroadcastTo, same_as_x(), vec![v]);
        // mask_f = MaskedFill(1.0 into zeros, where x == broadcast(reduce_max(x))).
        // Slot-free ⇒ its two uses (ReduceSumTo + final Mul) identity-share to
        // ONE node. Authored twice so both occurrences are structurally equal.
        let mask_f = || {
            let y = op(OpTag::ReduceMaxTo, same_as_up(), vec![x()]);
            let mask_u8 = op(OpTag::Equal, OpAttrs::default(), vec![x(), bcast_x(y)]);
            let zeros = op(
                OpTag::MulScalar,
                OpAttrs { scalars: vec![0.0], ..OpAttrs::default() },
                vec![x()],
            );
            op(
                OpTag::MaskedFill,
                OpAttrs { scalars: vec![1.0], ..OpAttrs::default() },
                vec![zeros, mask_u8],
            )
        };
        // ties = ReduceSumTo(mask_f); share = up / ties; grad_x = mask_f · bcast(share).
        let ties = op(OpTag::ReduceSumTo, same_as_up(), vec![mask_f()]);
        let share = op(OpTag::Div, OpAttrs::default(), vec![up(), ties]);
        op(OpTag::Mul, OpAttrs::default(), vec![mask_f(), bcast_x(share)])
    })
}

/// Per-entry scalar projection: ReduceMaxToBackward is parameterless (the fill
/// `1.0` and the `MulScalar(0.0)` are BAKED pattern constants, not open slots),
/// so the right payload projects to ZERO open-slot scalars and any other payload
/// is a typed decline (`None` ⇒ the bridge returns the node unchanged — G2).
fn scalars(params: &FusedOpParams) -> Option<Vec<f64>> {
    match params {
        FusedOpParams::ReduceMaxToBackward => Some(Vec::new()),
        _ => None,
    }
}

/// Decompose to the fair-share max subgradient — since A2 a re-emit of [`recipe`]'s
/// data through the [`decompose_via_recipe`] bridge (the fused node's two inputs
/// `[x, up]` are the binds; the resolving emit derives every interior shape/dtype
/// and re-resolves the `MaskedFill` fill Scalar to x's dtype). Any failure —
/// wrong params payload, a resolution decline at these shapes (symbolic extent,
/// …) — returns `id` (fixpoint, surfaced gap, never a panic): exactly the G2
/// posture the imperative body had.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    decompose_via_recipe(graph, id, recipe(), scalars(params))
}

pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

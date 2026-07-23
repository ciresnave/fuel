//! SoftmaxLastDimBackward — fused backward helper for the
//! `SoftmaxLastDim` forward. Phase 7.6 step 4 (continued — first
//! backward helper migrated; activates the registry's
//! `BackwardKind::Fused(id)` connection for the first time).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry`.
//! - [`recipe`] — the op's closed-form backward as portable, shape-/rank-
//!   polymorphic `PatternNode` data (Increment C slice 1, T8).
//! - [`decompose`] — re-emits [`recipe`] through the
//!   [`crate::registry::decompose_via_recipe`] bridge.
//!
//! The backward formula is `s * (g - sum(g * s, last_dim,
//! keepdim=true))` where `s` is the forward output and `g` is the
//! upstream gradient. Two inputs `[y, upstream]`; parameterless.
//!
//! ## Architectural note — registry purpose for backward helpers
//!
//! Backward helper entries serve a different role from forward
//! entries: there is no user-decomposed form to recognize (the
//! matcher is always stubbed). The `decompose` IS a "synthesize from
//! primitives" recipe, though: the closed-form backward expression
//! is a real 6-node primitive subgraph (the same `SumDim` + keepdim
//! + `BroadcastTo` idiom the forward `softmax_last_dim` recipe uses),
//! so per G2 it decomposes totally — never a basis-gap self-return.
//! T8 (Increment C slice 1) migrated that subgraph from an imperative
//! builder to the portable [`recipe`] datum, and in doing so exercised
//! the registry's `BackwardKind::Fused(id)` edge end-to-end on a data
//! recipe for the first time. Beyond the recipe, the registry entry
//! also exists to:
//!
//! - declare the op's identity (FusedOpId, FusedOpParams variant,
//!   shape/dtype rules);
//! - serve as the target of `BackwardKind::Fused(id)` from the
//!   matching forward entry (the architectural connection this
//!   commit activates);
//! - host per-backend `BackendImpl` registrations in the
//!   `FusedKernelRegistry` (step 6 extension; pending — the
//!   binding-table already covers CPU dispatch via the legacy
//!   `OpKind` route).
//!
//! Higher-order gradients (backward through this helper itself)
//! panic per `Tensor::backward`'s MVP behavior. The registry entry's
//! `backward` field reflects this: [`BackwardKind::NotDifferentiable`].

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::shape_expr::ShapeExpr;
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};
use std::sync::OnceLock;

/// Metadata-side registry entry for SoftmaxLastDimBackward.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::SOFTMAX_LAST_DIM_BACKWARD,
        name:       "SoftmaxLastDimBackward",
        family:     FusedOpFamily::Backward,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Shape rule: output equals input 0 (the forward softmax output).
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 2,
        "SoftmaxLastDimBackward takes 2 inputs (y, upstream)",
    );
    input_shapes[0].clone()
}

/// Dtype rule: output dtype equals input 0 (the forward softmax output).
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "SoftmaxLastDimBackward takes 2 inputs",
    );
    input_dtypes[0]
}

/// SoftmaxLastDimBackward's primitive recipe as **portable data** (Increment C
/// slice 1, T8 — the first *backward* helper migrated, and the op that
/// activates the registry's `BackwardKind::Fused(SOFTMAX_LAST_DIM_BACKWARD)`
/// edge end-to-end on a data recipe). Shape-/rank-polymorphic: the last-axis
/// sum reduces `SumDim(axis_last)` and the keepdim is restored by
/// `Unsqueeze(axis_last = append)` (the D3 shrink-via-swap replacing the baked
/// `ReduceSumTo(keepdim)`); the broadcast targets `SameAs { operand: 0 }` —
/// `s`'s full shape over the Bind space (D2). Bind: `0 = s` (the forward
/// softmax output), `1 = g` (the upstream gradient) — the order the autograd
/// edge emits (`lib.rs` softmax-backward arm: `vec![id, up_id]`). Nothing in
/// the datum bakes a shape or a rank; there are no open scalar slots
/// (parameterless).
///
/// Emitted form (6 op nodes; the `Unsqueeze` rebuilds rank BEFORE the
/// broadcast, so no D4 leading-1 pad is materialized):
///
/// ```text
///   gs      = Mul(g, s)                    # g · s
///   summed  = SumDim(axis_last)(gs)        # shrink…
///   summed_kd = Unsqueeze(axis_last)(summed)  # …restore keepdim ([..., 1])
///   summed_b  = BroadcastTo(SameAs 0)(summed_kd)
///   sub     = Sub(g, summed_b)             # g − sum(g·s)
///   out     = Mul(s, sub)                  # s · (…)
/// ```
fn recipe() -> &'static PatternNode {
    static RECIPE: OnceLock<PatternNode> = OnceLock::new();
    RECIPE.get_or_init(|| {
        let axis_last = || OpAttrs { axis_last: true, ..OpAttrs::default() };
        let same_as_s = || OpAttrs {
            target_shape_rel: Some(ShapeExpr::SameAs { operand: 0 }),
            ..OpAttrs::default()
        };
        let op = |op, attrs, operands| PatternNode::Op { op, attrs, operands };
        let s = || PatternNode::Bind { index: 0 };
        let g = || PatternNode::Bind { index: 1 };
        op(OpTag::Mul, OpAttrs::default(), vec![
            s(),
            op(OpTag::Sub, OpAttrs::default(), vec![
                g(),
                op(OpTag::BroadcastTo, same_as_s(), vec![
                    op(OpTag::Unsqueeze, axis_last(), vec![
                        op(OpTag::SumDim, axis_last(), vec![
                            op(OpTag::Mul, OpAttrs::default(), vec![g(), s()]),
                        ]),
                    ]),
                ]),
            ]),
        ])
    })
}

/// Per-entry scalar projection: SoftmaxLastDimBackward is parameterless, so the
/// right payload projects to ZERO open-slot scalars and any other payload is a
/// typed decline (`None` ⇒ the bridge returns the node unchanged — G2).
fn scalars(params: &FusedOpParams) -> Option<Vec<f64>> {
    match params {
        FusedOpParams::SoftmaxLastDimBackward => Some(Vec::new()),
        _ => None,
    }
}

/// Decompose to the closed-form softmax backward
/// `grad_x = s · (g − sum(g·s, last_dim, keepdim=true))`, where `s` (input 0)
/// is the forward output and `g` (input 1) is the upstream gradient — since T8
/// a re-emit of [`recipe`]'s data through the [`decompose_via_recipe`] bridge
/// (the fused node's two inputs `[s, g]` are the binds; the resolving emit
/// derives every interior shape/dtype). Any failure — wrong params payload, a
/// resolution decline at these shapes (symbolic extent, …) — returns `id`
/// (fixpoint, surfaced gap, never a panic): exactly the G2 posture the
/// imperative body had.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    decompose_via_recipe(graph, id, recipe(), scalars(params))
}

/// Matcher stub — backward-helper nodes originate from autograd
/// emitting `Op::Fused(SOFTMAX_LAST_DIM_BACKWARD, _)`, not from
/// user-decomposed forms.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

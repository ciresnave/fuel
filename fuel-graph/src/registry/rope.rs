//! Rope — rotary position embedding with caller-supplied cos/sin
//! tables. Increment C slice 1, T6 — the fifth op migrated to a portable
//! `PatternNode` DATA recipe (after SoftmaxLastDim), and the first to carry
//! `DimExpr` slice offsets (the rotate-half split).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (decompose function,
//!   pattern matcher, shape/dtype rules).
//! - [`recipe`] — the op's primitive subgraph as portable, shape-/rank-
//!   polymorphic data (9 nodes; the two leading-1 rank-pad `Reshape`s are
//!   materialized by the emit resolver on a rank-raise, D4, not baked here).
//! - [`decompose`] — re-emits [`recipe`] through the
//!   [`crate::registry::decompose_via_recipe`] bridge (mirrors
//!   [`crate::Tensor::rope_with_tables_decomposed`]).
//! - [`canonical_pattern`] — placeholder returning `None`. The Rope
//!   decomposition is structurally large (slice/concat + per-axis broadcast
//!   prep); until a canonical matcher lands, fusion fires only through the
//!   builder, never through pattern recognition.
//!
//! The decomposition is provided so backends without a native Rope
//! kernel can synthesize from primitives (today every backend has
//! one, but the lowering rule is wired regardless for completeness
//! and so cross-checks against the primitive path remain available).

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::shape_expr::{Dim, LAST, ShapeExpr};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};
use std::sync::OnceLock;

/// Metadata-side registry entry for Rope.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::ROPE,
        name:       "Rope",
        family:     FusedOpFamily::Forward,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        // Rope's backward is another Rope (with negated sin). It is
        // expressed in `Tensor::backward`'s Op::Fused arm directly
        // rather than through `BackwardKind::Fused(id)` because the
        // backward IS the same fused op — the registry's `Fused(id)`
        // variant is intended for backward helpers that have a
        // distinct id (SoftmaxLastDimBackward etc.).
        backward:   BackwardKind::NotDifferentiable,
        shape_rule: shape_passthrough,
        dtype_rule: dtype_passthrough,
        output_views: None,
    }
}

/// Shape rule: Rope preserves the x input's shape (input 0).
fn shape_passthrough(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(input_shapes.len(), 3, "Rope takes three inputs (x, cos, sin)");
    input_shapes[0].clone()
}

/// Dtype rule: Rope preserves the x input's dtype (input 0).
fn dtype_passthrough(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(input_dtypes.len(), 3, "Rope takes three inputs (x, cos, sin)");
    input_dtypes[0]
}

/// Rope's primitive recipe as **portable data** (Increment C slice 1, T6 —
/// the fifth op migrated, and the first with `DimExpr` slice offsets). Shape-
/// AND rank-polymorphic: the two cos/sin broadcasts target `SameAs { operand:
/// 0 }` (x's shape over the Bind space, D2); the rotate-half split is two
/// last-axis `Slice`s whose start/len are `DimExpr`s over x's last extent
/// `E = Extent { operand: 0, axis: LAST }` — the reference-doc worked example:
///
/// ```text
///   first_half  = Slice(axis_last, start = Const(0),  len = E / 2)(x)
///   second_half = Slice(axis_last, start = E / 2,     len = E − E / 2)(x)
/// ```
///
/// (the `Sub` remainder form is exact for the even `E` Rope requires, and is
/// the general last-axis-remainder spelling). Nothing in the datum bakes a
/// shape or a rank.
///
/// The datum is **9 nodes**; its EMISSION is byte-identical to the legacy
/// 11-node imperative form wherever the broadcast RANK-RAISES (the real
/// attention consumer — cos/sin `[seq, d]`, x `[.., seq, d]` rank ≥ 3): the
/// emit resolver materializes the two leading-1-padded prep `Reshape`s (D4).
/// At EQUAL rank (x itself `[seq, d]`) it emits the leaner 9-node form,
/// eliding legacy's no-op `Reshape([seq,d] → [seq,d])` — numerically
/// identical. Binds: `0 = x`, `1 = cos`, `2 = sin`.
///
/// ```text
///   cos_bcast   = BroadcastTo(SameAs 0)(cos)        # +Reshape pad on rank-raise
///   sin_bcast   = BroadcastTo(SameAs 0)(sin)        # +Reshape pad on rank-raise
///   first_half  = Slice(axis_last, 0,     E/2)(x)
///   second_half = Slice(axis_last, E/2,   E−E/2)(x)
///   rotated     = Concat(axis_last)(Neg(second_half), first_half)
///   left        = Mul(x, cos_bcast)
///   right       = Mul(rotated, sin_bcast)
///   out         = Add(left, right)
/// ```
fn recipe() -> &'static PatternNode {
    static RECIPE: OnceLock<PatternNode> = OnceLock::new();
    RECIPE.get_or_init(|| {
        let op = |op, attrs, operands| PatternNode::Op { op, attrs, operands };
        let x = || PatternNode::Bind { index: 0 };
        let cos = || PatternNode::Bind { index: 1 };
        let sin = || PatternNode::Bind { index: 2 };
        let same_as_x = || OpAttrs {
            target_shape_rel: Some(ShapeExpr::SameAs { operand: 0 }),
            ..OpAttrs::default()
        };
        // E = x's last extent; half = E / 2 (floor).
        let e = || Dim::Extent { operand: 0, axis: LAST };
        let half = || Dim::Div(Box::new(e()), Box::new(Dim::Const(2)));
        let axis_last = || OpAttrs { axis_last: true, ..OpAttrs::default() };
        // first_half = Slice(axis_last, start=0, len=E/2)(x).
        let first_half = op(
            OpTag::Slice,
            OpAttrs {
                slice_start_rel: Some(Dim::Const(0)),
                slice_len_rel: Some(half()),
                ..axis_last()
            },
            vec![x()],
        );
        // second_half = Slice(axis_last, start=E/2, len=E−E/2)(x).
        let second_half = op(
            OpTag::Slice,
            OpAttrs {
                slice_start_rel: Some(half()),
                slice_len_rel: Some(Dim::Sub(Box::new(e()), Box::new(half()))),
                ..axis_last()
            },
            vec![x()],
        );
        // rotated = Concat(axis_last)(Neg(second_half), first_half).
        let rotated = op(OpTag::Concat, axis_last(), vec![
            op(OpTag::Neg, OpAttrs::default(), vec![second_half]),
            first_half,
        ]);
        // out = Add(Mul(x, cos_bcast), Mul(rotated, sin_bcast)).
        op(OpTag::Add, OpAttrs::default(), vec![
            op(OpTag::Mul, OpAttrs::default(), vec![
                x(),
                op(OpTag::BroadcastTo, same_as_x(), vec![cos()]),
            ]),
            op(OpTag::Mul, OpAttrs::default(), vec![
                rotated,
                op(OpTag::BroadcastTo, same_as_x(), vec![sin()]),
            ]),
        ])
    })
}

/// Per-entry scalar projection: Rope is parameterless (its cos/sin tables are
/// INPUTS, not baked scalars), so the right payload projects to ZERO open-slot
/// scalars and any other payload is a typed decline (`None` ⇒ the bridge
/// returns the node unchanged — G2).
fn scalars(params: &FusedOpParams) -> Option<Vec<f64>> {
    match params {
        FusedOpParams::Rope => Some(Vec::new()),
        _ => None,
    }
}

/// Lower a fused Rope node to its primitive subgraph and return the new root
/// id — since T6 a re-emit of [`recipe`]'s data through the
/// [`decompose_via_recipe`] bridge (the fused node's three inputs are the
/// binds `[x, cos, sin]`; the resolving emit derives every interior
/// shape/dtype and materializes the D4 rank-pad `Reshape`s). Any failure —
/// wrong params payload, a resolution decline at these shapes (symbolic
/// extent, …) — returns `id` (fixpoint, surfaced gap, never a panic): exactly
/// the G2 posture the imperative body had.
///
/// The fused node `id` may be either `Op::Rope` (legacy emission) or
/// `Op::Fused(FusedOps::ROPE, FusedOpParams::Rope)` (the builder path); the
/// decomposition is identical for both. Mirrors
/// [`crate::Tensor::rope_with_tables_decomposed`].
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    decompose_via_recipe(graph, id, recipe(), scalars(params))
}

/// Placeholder matcher: returns `None` for every input. The Rope
/// decomposition is structurally large (slice/concat/reshape +
/// per-axis broadcast prep), and the migration bar is "the registry entry
/// exists, builder emits Op::Fused, dispatch works." A canonical
/// matcher recognizing the full pattern + single-consumer
/// guards is follow-up work. Until then this reads as a one-way
/// migration: builder→fused works, hand-built decomposed forms (e.g.
/// `rope_with_tables_decomposed`) stay decomposed.
pub fn canonical_pattern(_graph: &Graph, _add_id: NodeId) -> Option<PatternMatch> {
    None
}

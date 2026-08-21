//! InplaceAffine — `x = mul·x + add`, mutating input 0.
//!
//! Phase 1 of the in-place ops infrastructure
//! (`docs/session-prompts/in-place-ops-infrastructure.md`).
//! Single input. The output node aliases input 0 by contract;
//! `Op::destructive_input` marks index 0 destructive so that
//! `opt::derive_ordering` pins this node to run after every
//! non-destructive reader of the input.
//!
//! Backend dispatch (CPU + CUDA `affine_inplace_*`) lands in Phase 3.
//! Autograd integration via the mutation-safety pass lands in Phase 4.
//! Until then, the metadata-side entry exists so CSE, telemetry, and
//! the registry's shape/dtype dispatch work for `Op::Fused(INPLACE_AFFINE, _)`
//! nodes constructed in tests or by future model code.
//!
//! `decompose` carries the functional **value** recipe `mul·x + add`
//! (`MulScalar → AddScalar`) — the base-map cover + correctness-floor fallback
//! (Increment C). The destructive/aliasing facet rides `destructive_input()` +
//! the FKC in-place contract on the `Op`, which the decompose does not touch;
//! see [`decompose`]'s doc for why this is NOT a basis gap.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps, PatternMatch,
    SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};
use std::sync::OnceLock;

pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: Some(0),
        id: FusedOps::INPLACE_AFFINE,
        name: "InplaceAffine",
        family: FusedOpFamily::Forward,
        pattern: SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward: BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(),
        1,
        "InplaceAffine takes 1 input (the mutated tensor)",
    );
    input_shapes[0].clone()
}

fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(input_dtypes.len(), 1, "InplaceAffine takes 1 input",);
    input_dtypes[0]
}

/// InplaceAffine's primitive **value** recipe: the functional affine
/// `x = mul·x + add` as `AddScalar(add)(MulScalar(mul)(x))`. One bind (`0 = x`,
/// the mutated tensor) and TWO open scalar slots — both filled from the params
/// projection ([`scalars`]) at emit time. Slot fill is pattern PRE-order (the
/// current node consumes the cursor before descending into its operands), so the
/// OUTER `AddScalar` takes `scalars[0]` and the INNER `MulScalar` takes
/// `scalars[1]` — hence [`scalars`] projects `vec![add, mul]`, NOT `[mul, add]`.
///
/// ```text
///   mul_x = MulScalar[open slot 1 = mul](x)
///   out   = AddScalar[open slot 0 = add](mul_x)   #  = mul·x + add
/// ```
fn recipe() -> &'static PatternNode {
    static RECIPE: OnceLock<PatternNode> = OnceLock::new();
    RECIPE.get_or_init(|| {
        let op = |op, attrs, operands| PatternNode::Op {
            op,
            attrs: Box::new(attrs),
            operands,
        };
        let x = || PatternNode::Bind { index: 0 };
        // mul_x = MulScalar[open](x) — empty scalars = an open slot (filled 2nd).
        let mul_x = op(OpTag::MulScalar, OpAttrs::default(), vec![x()]);
        // out = AddScalar[open](mul_x) — the outer node, filled FIRST (pre-order).
        op(OpTag::AddScalar, OpAttrs::default(), vec![mul_x])
    })
}

/// Per-entry scalar projection. The recipe's two open slots are filled in
/// pattern pre-order: the outer `AddScalar` first (`scalars[0]`), the inner
/// `MulScalar` second (`scalars[1]`). So the right payload projects to
/// `vec![add, mul]`; any other payload is a typed decline (`None` ⇒ the bridge
/// returns the node unchanged — G2 fixpoint).
fn scalars(params: &FusedOpParams) -> Option<Vec<f64>> {
    match params {
        FusedOpParams::InplaceAffine { mul, add } => Some(vec![*add, *mul]),
        _ => None,
    }
}

/// Decompose to the functional affine value `mul·x + add` — a re-emit of
/// [`recipe`] through the [`decompose_via_recipe`] bridge (the fused node's one
/// input `[x]` is the bind; [`scalars`] fills the mul/add open slots).
///
/// **The corrected finding (supersedes the pre-migration "basis gap" note).**
/// This is NOT a basis gap. The over-conservative earlier reasoning conflated
/// two independent layers:
/// * `Op::destructive_input() -> Some(0)` — a method on the `Op` (a per-fused-id
///   match arm in `fuel-graph/src/lib.rs`) that drives `opt::derive_ordering` on
///   the EXECUTION graph, where the fused `Op::Fused(INPLACE_AFFINE)` node lives.
///   It is UNAFFECTED by what `decompose` returns.
/// * `decompose` feeds only the BASE-MAP COVER (value / `base_map_hash` /
///   verify-oracle / correctness-floor fallback), NOT execution — the fused op
///   executes via its own in-place kernel (`affine_inplace_*`, Phase 3).
///
/// So the functional `MulScalar → AddScalar` decompose is value-correct, does
/// NOT "drop" the destructive contract (which rides `destructive_input()` + the
/// FKC in-place / aliasing contract on the `Op`, both untouched here), and is
/// self-consistent — the functional form does not mutate, so it needs no
/// ordering pin. It makes `decompose` total (the recipe principle) and gives
/// `InplaceAffine` an executable functional fallback until the in-place kernel
/// lands. In-place-ness is a KISS-Contract §4.6/§5.4 + `destructive_input` facet,
/// NOT an op-basis or decompose concern — no new primitive, no standard change.
///
/// Totality (G2): a wrong params payload declines via [`scalars`] `= None`
/// BEFORE any emission, and any later bridge failure returns `id` (fixpoint,
/// surfaced gap, never a panic).
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    decompose_via_recipe(graph, id, recipe(), scalars(params))
}

pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Node, Op};

    /// Build a fused `INPLACE_AFFINE` node over `x [dims]` with the given
    /// `(mul, add)` params (single input — the mutated tensor).
    fn fused_inplace_affine(g: &mut Graph, dims: &[usize], mul: f64, add: f64) -> NodeId {
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(dims),
            dtype: DType::F32,
        });
        g.push(Node {
            op: Op::Fused(
                FusedOps::INPLACE_AFFINE,
                FusedOpParams::InplaceAffine { mul, add },
            ),
            inputs: vec![x],
            shape: Shape::from_dims(dims),
            dtype: DType::F32,
        })
    }

    /// (a) BORN-RED then green: `decompose` now lowers `Op::Fused(INPLACE_AFFINE,
    /// {mul, add})` to the functional `AddScalar(add)(MulScalar(mul)(x))`
    /// primitive subgraph (value form `x = mul·x + add`). Structural — asserts
    /// the emitted node ops AND that each open scalar slot received the right
    /// value (the outer `AddScalar` fills first in pattern pre-order → `add`;
    /// the inner `MulScalar` fills second → `mul`).
    #[test]
    fn inplace_affine_decompose_lowers_to_mul_then_add_scalar() {
        let mut g = Graph::new();
        let (mul, add) = (2.0f64, 1.0f64);
        let fused = fused_inplace_affine(&mut g, &[4], mul, add);
        let x = g.node(fused).inputs[0];
        let out_shape = g.node(fused).shape.clone();
        let params = FusedOpParams::InplaceAffine { mul, add };

        let root = decompose(&mut g, fused, &params);

        assert_ne!(
            root, fused,
            "InplaceAffine now lowers to a functional subgraph"
        );
        // root = AddScalar(add)(...)
        assert!(
            matches!(g.node(root).op, Op::AddScalar(v) if v == add),
            "root is AddScalar(add); got {:?}",
            g.node(root).op,
        );
        // operand = MulScalar(mul)(x)
        let ms = g.node(root).inputs[0];
        assert!(
            matches!(g.node(ms).op, Op::MulScalar(v) if v == mul),
            "operand is MulScalar(mul); got {:?}",
            g.node(ms).op,
        );
        assert_eq!(
            g.node(ms).inputs,
            vec![x],
            "MulScalar reads the fused node's input 0"
        );
        assert_eq!(
            g.node(root).shape,
            out_shape,
            "value form preserves the input shape"
        );
        assert_eq!(g.node(root).dtype, DType::F32);
    }

    /// (c) DESTRUCTIVE FACET SURVIVES (load-bearing — the whole point of the
    /// migration). The value migrates, but `destructive_input()` is a method on
    /// the `Op` (a per-fused-id match arm in `fuel-graph/src/lib.rs`), untouched
    /// by what `decompose` returns — so the in-place / aliasing contract the
    /// optimizer reasons about still holds after the migration.
    #[test]
    fn inplace_affine_destructive_input_survives_migration() {
        let op = Op::Fused(
            FusedOps::INPLACE_AFFINE,
            FusedOpParams::InplaceAffine { mul: 2.0, add: 1.0 },
        );
        assert_eq!(
            op.destructive_input(),
            Some(0),
            "INPLACE_AFFINE still marks input 0 destructive after the decompose migration",
        );
    }

    /// (d) Totality (G2): a wrong params payload declines to a fixpoint — never a
    /// crash — before any emission (the `scalars` projection returns `None`, so
    /// the bridge returns the node unchanged without pushing any recipe nodes).
    #[test]
    fn inplace_affine_wrong_params_is_a_fixpoint_not_a_crash() {
        let mut g = Graph::new();
        let fused = fused_inplace_affine(&mut g, &[4], 2.0, 1.0);
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::Rope);
        assert_eq!(out, fused, "wrong params => typed decline => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }
}

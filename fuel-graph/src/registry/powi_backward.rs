// SPDX-License-Identifier: MIT OR Apache-2.0
//! PowIBackward — fused backward helper for the `Op::PowI(exp)`
//! primitive. Phase 7.6 step 4; migrated to a portable `PatternNode`
//! DATA recipe in Increment C carriers (A3 — the first recipe to carry a
//! `PowI`, via the A3 i32-exponent re-emit carrier, and the first whose
//! STRUCTURE is param-derived).
//!
//! Closed-form: `grad_x = exp · x^(exp-1) · upstream`. Two inputs
//! `[x, upstream]` + `exp: i32`. Replaces the autograd primitive
//! decomposition `PowI(n-1) → MulScalar(n as f64) → Mul` (3 nodes,
//! 3 dispatches, 2 intermediate buffers) with a single launch of
//! baracuda alpha.31's `unary_powi_backward_*` kernel.
//!
//! Like `ReduceMaxToBackward`, this is the backward of a primitive
//! (`Op::PowI`), not of a fused forward — autograd reaches it directly
//! from `Op::PowI`'s arm in `Tensor::backward` rather than through a
//! `BackwardKind::Fused` edge.
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry`.
//! - [`recipe`] — the op's closed-form backward as portable `PatternNode` data
//!   (Increment C carriers, A3). Unlike the shape-polymorphic slice-1/-2
//!   recipes, this datum is a pure function of `exp` (see its doc for why).
//! - [`decompose`] — re-emits [`recipe`] through the
//!   [`crate::registry::decompose_via_recipe`] bridge.
//!
//! See `softmax_last_dim_backward.rs` for the shared architectural
//! rationale around backward-helper entries (identity + future `BackendImpl`
//! host; the matcher is always stubbed — autograd emits the fused node).

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps, PatternMatch,
    SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};

/// Metadata-side registry entry for PowIBackward.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: None,
        id: FusedOps::POWI_BACKWARD,
        name: "PowIBackward",
        family: FusedOpFamily::Backward,
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
        2,
        "PowIBackward takes 2 inputs (x, upstream)",
    );
    // grad_x preserves x's shape.
    input_shapes[0].clone()
}

fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(input_dtypes.len(), 2, "PowIBackward takes 2 inputs",);
    input_dtypes[0]
}

/// PowIBackward's primitive recipe as portable `PatternNode` data (Increment C
/// carriers, A3) — the first recipe to carry a `PowI` (via the A3 i32-exponent
/// re-emit carrier). The closed-form gradient `grad_x = exp · x^(exp-1) ·
/// upstream` is a DIRECT three-node mirror of the pre-migration imperative body
/// (`PowI(exp-1) → MulScalar(exp) → Mul`), so no D3 keepdim swap and no D4 pad
/// fire and the recipe base map is bit-identical to the legacy body (a
/// structure-preserving migration — it inherits the slice-1 numeric proof).
/// Binds: `0 = x`, `1 = upstream` — the order autograd's `Op::PowI` arm emits
/// (`inputs = [id, up_id]`).
///
/// Unlike the shape-polymorphic slice-1/-2 recipes (a process-lifetime
/// `OnceLock` datum + open scalar slots filled from the params projection),
/// PowIBackward's recipe is a pure function of `exp`: the exponent is not a
/// shape and not an f64 slot — it is the structural `i32` of `Op::PowI`, which
/// only the A3 carrier can transport and which no open-slot mechanism can fill.
/// So both param-derived constants are CONSTANT-FOLDED into the datum at build
/// time (the minimal "param-derived const in the recipe" the Increment C plan
/// authorizes for C-4-class values, rather than restructuring to dodge the
/// missing thread):
/// * the `PowI` carries `scalars = [(exp-1) as f64]` — the A3 carrier
///   reconstructs `Op::PowI(exp-1)` (exact for every i32 through f64);
/// * the `MulScalar` carries `scalars = [exp as f64]` — a BAKED constant.
///
/// Both are baked (non-empty `scalars`), so the datum has ZERO open scalar slots
/// and [`decompose`] fills the projection with the empty vec. Edge cases fall
/// out exactly as the imperative body: `exp==1` → `PowI(0)·1·upstream =
/// upstream`; `exp==0` → `MulScalar(0) = 0`.
///
/// Emitted form:
///
/// ```text
///   pow    = PowI(exp-1)(x)
///   scaled = MulScalar(exp)(pow)
///   grad_x = Mul(scaled, upstream)
/// ```
fn recipe(exp: i32) -> PatternNode {
    let op = |op, attrs, operands| PatternNode::Op {
        op,
        attrs,
        operands,
    };
    let x = || PatternNode::Bind { index: 0 };
    let up = || PatternNode::Bind { index: 1 };
    // pow = PowI(exp-1)(x) — the exponent rides scalars[0] (the A3 carrier).
    let pow = op(
        OpTag::PowI,
        OpAttrs {
            scalars: vec![(exp - 1) as f64],
            ..OpAttrs::default()
        },
        vec![x()],
    );
    // scaled = MulScalar(exp)(pow) — a baked constant, not an open slot.
    let scaled = op(
        OpTag::MulScalar,
        OpAttrs {
            scalars: vec![exp as f64],
            ..OpAttrs::default()
        },
        vec![pow],
    );
    // grad_x = Mul(scaled, upstream).
    op(OpTag::Mul, OpAttrs::default(), vec![scaled, up()])
}

/// Decompose to the closed-form gradient `grad_x = exp · x^(exp-1) · upstream`,
/// where `x` (input 0) is the forward `PowI` input and `upstream` (input 1) is
/// the upstream gradient — since A3 a re-emit of [`recipe`]'s param-derived data
/// through the [`decompose_via_recipe`] bridge (the fused node's two inputs
/// `[x, upstream]` are the binds; every scalar is baked, so the projection is
/// the empty vec; the resolving emit derives every interior shape/dtype). A
/// wrong params payload declines HERE (before building the recipe) and any later
/// failure (a resolution decline at these shapes, …) returns `id` (fixpoint,
/// surfaced gap, never a panic): exactly the G2 posture the imperative body had.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let exp = match params {
        FusedOpParams::PowIBackward { exp } => *exp,
        // G2: total + never-panic — non-PowIBackward params are an impossible
        // registry-dispatch invariant violation; return self.
        _ => return id,
    };
    // All scalars are BAKED param-derived constants (exp, exp-1), so the datum
    // has zero open slots ⇒ the projection is the empty vec.
    decompose_via_recipe(graph, id, &recipe(exp), Some(Vec::new()))
}

/// Matcher stub — backward-helper nodes originate from autograd emitting
/// `Op::Fused(POWI_BACKWARD, _)`, not from user-decomposed forms.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

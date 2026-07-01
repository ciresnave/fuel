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

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};

pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::INPLACE_AFFINE,
        name:       "InplaceAffine",
        family:     FusedOpFamily::Forward,
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
        input_shapes.len(), 1,
        "InplaceAffine takes 1 input (the mutated tensor)",
    );
    input_shapes[0].clone()
}

fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 1,
        "InplaceAffine takes 1 input",
    );
    input_dtypes[0]
}

/// InplaceAffine is a genuine **basis gap** of a different kind (G2,
/// 2026-06-20). Its *value* `mul·x + add` is trivially expressible
/// (`MulScalar(mul) → AddScalar(add)`), but this op is *defined by* its
/// in-place / destructive-aliasing semantics: `destructive_input() -> Some(0)`,
/// the output aliases input 0's storage, and `opt::derive_ordering` pins it
/// after every non-destructive reader of that input. Fuel has no primitive that
/// expresses an in-place affine update (the in-place family is unary, with no
/// parameterized affine member), so decomposing to the *functional*
/// `MulScalar → AddScalar` subgraph would silently drop the destructive
/// contract the optimizer reasons about — not a faithful recipe. Per G2
/// `decompose` is total + never-panic: with no semantics-preserving recipe it
/// returns **self**, decomposing once an in-place affine primitive
/// (`Op::AffineInplace { mul, add }`) lands. Callers wanting the functional
/// form should use `Tensor::affine` directly.
pub fn decompose(_graph: &mut Graph, id: NodeId, _params: &FusedOpParams) -> NodeId {
    id
}

pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

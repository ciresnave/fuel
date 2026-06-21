//! ReduceMaxToBackward — fused backward helper for the `Op::ReduceMaxTo`
//! primitive. Phase 7.6 step 4 (continued).
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
//! See `softmax_last_dim_backward.rs` for the shared architectural
//! rationale around backward-helper entries.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_core_types::{DType, Scalar, Shape};

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

/// Decompose to the fair-share max subgradient: route `upstream` to the
/// position(s) where `x` equals its per-window max; tied maxes share equally.
/// Inputs `[x, upstream]`; `upstream`'s shape is the forward reduce target.
///
/// Recipe: `y = ReduceMaxTo(x)` → `BroadcastTo` → `mask = (x == y)` (`Op::Equal`
/// → U8), then build a float mask via `MaskedFill(value=1, zeros, mask)` (this
/// avoids an integer→float `Cast`, which the CPU cast matrix doesn't carry —
/// see `Tensor::cast`); `ties = ReduceSumTo(mask)`, `share = upstream / ties`,
/// `grad_x = mask · broadcast(share)`. The module's old "needs an equality
/// primitive that doesn't exist" claim was wrong — `Op::Equal` exists — so per
/// G2 this is a real decomposition, not a basis-gap self-return.
pub fn decompose(graph: &mut Graph, id: NodeId, _params: &FusedOpParams) -> NodeId {
    let (x_id, up_id, x_shape, dtype) = {
        let n = graph.node(id);
        (n.inputs[0], n.inputs[1], n.shape.clone(), n.dtype)
    };
    let target = graph.node(up_id).shape.clone();

    // y = per-window max, broadcast back to x's shape.
    let y = graph.push(Node {
        op: Op::ReduceMaxTo(target.clone()),
        inputs: vec![x_id],
        shape: target.clone(),
        dtype,
    });
    let y_b = graph.push(Node {
        op: Op::BroadcastTo(x_shape.clone()),
        inputs: vec![y],
        shape: x_shape.clone(),
        dtype,
    });
    // U8 mask = (x == max), then a float mask = MaskedFill(1.0 into zeros).
    let mask_u8 = graph.push(Node {
        op: Op::Equal,
        inputs: vec![x_id, y_b],
        shape: x_shape.clone(),
        dtype: DType::U8,
    });
    let zeros = graph.push(Node {
        op: Op::MulScalar(0.0),
        inputs: vec![x_id],
        shape: x_shape.clone(),
        dtype,
    });
    let mask_f = graph.push(Node {
        op: Op::MaskedFill {
            value: Scalar::one(dtype),
        },
        inputs: vec![zeros, mask_u8],
        shape: x_shape.clone(),
        dtype,
    });
    // ties = count per window; share = upstream / ties (fair share for ties).
    let ties = graph.push(Node {
        op: Op::ReduceSumTo(target.clone()),
        inputs: vec![mask_f],
        shape: target.clone(),
        dtype,
    });
    let share = graph.push(Node {
        op: Op::Div,
        inputs: vec![up_id, ties],
        shape: target,
        dtype,
    });
    let share_b = graph.push(Node {
        op: Op::BroadcastTo(x_shape.clone()),
        inputs: vec![share],
        shape: x_shape.clone(),
        dtype,
    });
    graph.push(Node {
        op: Op::Mul,
        inputs: vec![mask_f, share_b],
        shape: x_shape,
        dtype,
    })
}

pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

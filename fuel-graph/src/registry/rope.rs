//! Rope — rotary position embedding with caller-supplied cos/sin
//! tables. Phase 7.6 step 4 (continued — fifth op migrated after
//! SoftmaxLastDim, FusedLinear, RmsNormLastDim, LayerNormLastDim).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (decompose function,
//!   pattern matcher, shape/dtype rules).
//! - [`canonical_pattern`] — placeholder returning `None`. Same
//!   rationale as LayerNormLastDim: the 12-node Rope decomposition is
//!   structurally large, and step 4's bar is "registry entry exists,
//!   builder emits Op::Fused, dispatch works." Until a canonical
//!   matcher lands, fusion only fires through the builder, never
//!   through pattern recognition.
//! - [`decompose`] — emits the canonical 12-node primitive subgraph
//!   that mirrors [`crate::Tensor::rope_with_tables_decomposed`].
//!
//! The decomposition is provided so backends without a native Rope
//! kernel can synthesize from primitives (today every backend has
//! one, but the lowering rule is wired regardless for completeness
//! and so cross-checks against the primitive path remain available).

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_core_types::{DType, Shape};

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

/// Lower a fused Rope node to its canonical 12-node primitive subgraph
/// and return the new root id. Mirrors
/// [`crate::Tensor::rope_with_tables_decomposed`] node-for-node:
///
/// ```text
///   cos_reshaped = Reshape([1, ..., seq, d])(cos)
///   sin_reshaped = Reshape([1, ..., seq, d])(sin)
///   cos_bcast    = BroadcastTo(x_shape)(cos_reshaped)
///   sin_bcast    = BroadcastTo(x_shape)(sin_reshaped)
///   first_half   = Slice(dim=-1, start=0,    len=half)(x)
///   second_half  = Slice(dim=-1, start=half, len=half)(x)
///   neg_second   = Neg(second_half)
///   rotated_half = Concat(dim=-1)(neg_second, first_half)
///   left         = Mul(x, cos_bcast)
///   right        = Mul(rotated_half, sin_bcast)
///   out          = Add(left, right)
/// ```
///
/// The fused node `id` may be either `Op::Rope` (legacy emission) or
/// `Op::Fused(FusedOps::ROPE, FusedOpParams::Rope)` (the new builder
/// path). The decomposition is identical for both.
pub fn decompose(graph: &mut Graph, id: NodeId, _params: &FusedOpParams) -> NodeId {
    let (x_id, cos_id, sin_id, x_shape, dtype) = {
        let n = graph.node(id);
        debug_assert_eq!(n.inputs.len(), 3, "Rope expects 3 inputs (x, cos, sin)");
        (n.inputs[0], n.inputs[1], n.inputs[2], n.shape.clone(), n.dtype)
    };
    let dims = x_shape.dims().to_vec();
    let rank = dims.len();
    debug_assert!(rank >= 2, "Rope requires rank >= 2");
    let seq = dims[rank - 2];
    let d = dims[rank - 1];
    debug_assert!(d.is_multiple_of(2), "Rope: feature dim {d} must be even");
    let half = d / 2;
    let last = rank - 1;

    // Build the broadcast-prep shape: 1s everywhere except seq/d.
    let mut broadcast_shape_dims: Vec<usize> = vec![1usize; rank];
    broadcast_shape_dims[rank - 2] = seq;
    broadcast_shape_dims[rank - 1] = d;
    let broadcast_shape = Shape::from_dims(&broadcast_shape_dims);

    let cos_reshaped_id = graph.push(Node {
        op:     Op::Reshape(broadcast_shape.clone()),
        inputs: vec![cos_id],
        shape:  broadcast_shape.clone(),
        dtype,
    });
    let sin_reshaped_id = graph.push(Node {
        op:     Op::Reshape(broadcast_shape.clone()),
        inputs: vec![sin_id],
        shape:  broadcast_shape,
        dtype,
    });
    let cos_bcast_id = graph.push(Node {
        op:     Op::BroadcastTo(x_shape.clone()),
        inputs: vec![cos_reshaped_id],
        shape:  x_shape.clone(),
        dtype,
    });
    let sin_bcast_id = graph.push(Node {
        op:     Op::BroadcastTo(x_shape.clone()),
        inputs: vec![sin_reshaped_id],
        shape:  x_shape.clone(),
        dtype,
    });

    // Slice produces a half-width shape along the last dim.
    let mut half_dims = dims.clone();
    half_dims[last] = half;
    let half_shape = Shape::from_dims(&half_dims);

    let first_half_id = graph.push(Node {
        op:     Op::Slice { dim: last, start: 0, len: half },
        inputs: vec![x_id],
        shape:  half_shape.clone(),
        dtype,
    });
    let second_half_id = graph.push(Node {
        op:     Op::Slice { dim: last, start: half, len: half },
        inputs: vec![x_id],
        shape:  half_shape.clone(),
        dtype,
    });
    let neg_second_id = graph.push(Node {
        op:     Op::Neg,
        inputs: vec![second_half_id],
        shape:  half_shape,
        dtype,
    });
    let rotated_half_id = graph.push(Node {
        op:     Op::Concat { dim: last },
        inputs: vec![neg_second_id, first_half_id],
        shape:  x_shape.clone(),
        dtype,
    });

    let left_id = graph.push(Node {
        op:     Op::Mul,
        inputs: vec![x_id, cos_bcast_id],
        shape:  x_shape.clone(),
        dtype,
    });
    let right_id = graph.push(Node {
        op:     Op::Mul,
        inputs: vec![rotated_half_id, sin_bcast_id],
        shape:  x_shape.clone(),
        dtype,
    });
    let out_id = graph.push(Node {
        op:     Op::Add,
        inputs: vec![left_id, right_id],
        shape:  x_shape,
        dtype,
    });
    out_id
}

/// Placeholder matcher: returns `None` for every input. The 12-node
/// Rope decomposition is structurally large (slice/concat/reshape +
/// per-axis broadcast prep), and step 4's bar is "the registry entry
/// exists, builder emits Op::Fused, dispatch works." A canonical
/// matcher recognizing the full 12-node pattern + single-consumer
/// guards is follow-up work. Until then this reads as a one-way
/// migration: builder→fused works, hand-built decomposed forms (e.g.
/// `rope_with_tables_decomposed`) stay decomposed.
pub fn canonical_pattern(_graph: &Graph, _add_id: NodeId) -> Option<PatternMatch> {
    None
}

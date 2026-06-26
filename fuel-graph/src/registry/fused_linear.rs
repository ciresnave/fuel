//! FusedLinear — first multi-input fused op migrated through the
//! FusedOpRegistry. Phase 7.6 step 4 (continuation).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (decompose, pattern
//!   matcher, shape/dtype rules).
//! - [`canonical_pattern`] — recognizes `Add(MatMul(a, b),
//!   BroadcastTo(rank-1 bias))` and produces a [`PatternMatch`] with
//!   `bindings = [(0, a), (1, b), (2, bias)]`.
//! - [`decompose`] — emits the primitive subgraph
//!   `Add(MatMul(a, b), BroadcastTo(bias))`.
//!
//! Replaces the hand-written rewrite in [`crate::opt::fuse_linear`]:
//! that function now also emits `Op::Fused(FUSED_LINEAR, _)` so direct
//! callers (the `fuse_linear` oracle test) and the auto-generated
//! `FusionRule` produce the same shape.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, Shape};
use std::collections::HashMap;

/// Metadata-side registry entry for FusedLinear.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::FUSED_LINEAR,
        name:       "FusedLinear",
        family:     FusedOpFamily::Forward,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        // The backward fused-op isn't part of the registry yet (a later
        // step migrates each fused-backward helper). Tensor::backward
        // dispatches `Op::Fused(FUSED_LINEAR, _)` through a per-id arm
        // that emits the same three-grad decomposition as the legacy
        // `Op::FusedLinear` arm.
        backward:   BackwardKind::NotDifferentiable,
        shape_rule: matmul_output_shape,
        dtype_rule: dtype_passthrough,
        output_views: None,
    }
}

/// Shape rule: FusedLinear output shape is the matmul output —
/// `[..., M, N]` where `a` is `[..., M, K]` and `b` is `[..., K, N]`.
/// Bias is rank-1 `[N]` and broadcasts implicitly.
fn matmul_output_shape(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(input_shapes.len(), 3, "FusedLinear takes 3 inputs (a, b, bias)");
    let a = &input_shapes[0];
    let b = &input_shapes[1];
    let a_rank = a.rank();
    let b_rank = b.rank();
    debug_assert!(a_rank >= 2 && b_rank >= 2);
    debug_assert_eq!(a_rank, b_rank, "FusedLinear: a/b ranks must match");
    let mut dims: Vec<usize> = a.dims()[..a_rank - 2].to_vec();
    dims.push(a.dims()[a_rank - 2]);          // M
    dims.push(b.dims()[b_rank - 1]);          // N
    Shape::from_dims(&dims)
}

/// Dtype rule: FusedLinear output dtype matches `a` (all three inputs
/// must agree at construction time).
fn dtype_passthrough(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(input_dtypes.len(), 3, "FusedLinear takes 3 inputs");
    input_dtypes[0]
}

/// Lower a fused FusedLinear node to its primitive subgraph and return
/// the new root id. The primitive form is the standard matmul +
/// broadcast + add chain — backends without a true fused kernel
/// execute this directly; backends with one (CUTLASS bias-epilogue,
/// cuBLAS gemm-with-bias) register a `BackendImpl` in the kernel-side
/// registry and skip the lowering entirely.
///
/// ```text
///   mm        = MatMul(a, b)               # [..., M, N]
///   bias_bcst = BroadcastTo([..., M, N])(bias)
///   out       = Add(mm, bias_bcst)
/// ```
pub fn decompose(graph: &mut Graph, id: NodeId, _params: &FusedOpParams) -> NodeId {
    let (a_id, b_id, bias_id, out_shape, dtype) = {
        let n = graph.node(id);
        debug_assert_eq!(n.inputs.len(), 3, "FusedLinear node must have 3 inputs");
        (n.inputs[0], n.inputs[1], n.inputs[2], n.shape.clone(), n.dtype)
    };
    let mm_id = graph.push(Node {
        op:     Op::MatMul,
        inputs: vec![a_id, b_id],
        shape:  out_shape.clone(),
        dtype,
    });
    let bias_bcst_id = graph.push(Node {
        op:     Op::BroadcastTo(out_shape.clone()),
        inputs: vec![bias_id],
        shape:  out_shape.clone(),
        dtype,
    });
    graph.push(Node {
        op:     Op::Add,
        inputs: vec![mm_id, bias_bcst_id],
        shape:  out_shape,
        dtype,
    })
}

/// Match the canonical FusedLinear pattern rooted at an `Add` node:
/// `Add(MatMul(a, b), bias_broadcast)` where `bias_broadcast` is
/// either `BroadcastTo(rank-1 bias)` or a rank-1 bias directly, and
/// the rank-1 bias's length equals the matmul output's last dim.
///
/// Returns `bindings = [(0, a), (1, b), (2, bias_src)]` and
/// `params = FusedOpParams::FusedLinear`. Conservative: only matches
/// when the inner MatMul has exactly one consumer (this Add).
/// Otherwise fusing would force a duplicated matmul computation.
///
/// Direct port of the hand-written walker in [`crate::opt::fuse_linear`];
/// that walker continues to exist as a stand-alone API (the oracle
/// test calls it directly) but now emits `Op::Fused(FUSED_LINEAR, _)`
/// the same way the auto-generated FusionRule does.
pub fn canonical_pattern(graph: &Graph, add_id: NodeId) -> Option<PatternMatch> {
    let add = graph.node(add_id);
    if !matches!(add.op, Op::Add) { return None; }
    if add.inputs.len() != 2 { return None; }
    let mm_id = add.inputs[0];
    let rhs_id = add.inputs[1];

    let mm = graph.node(mm_id);
    if !matches!(mm.op, Op::MatMul) { return None; }
    if mm.inputs.len() != 2 { return None; }
    let a_id = mm.inputs[0];
    let b_id = mm.inputs[1];

    let mm_dims = mm.shape.dims();
    if mm_dims.is_empty() { return None; }
    let last_dim = mm_dims[mm_dims.len() - 1];

    let rhs = graph.node(rhs_id);
    let bias_src_id =
        if matches!(rhs.op, Op::BroadcastTo(_)) && rhs.inputs.len() == 1 {
            rhs.inputs[0]
        } else {
            // Defensive: a rank-1 bias directly on Add's RHS only
            // type-checks if Add allows implicit broadcasting — the
            // legacy walker accepted it, so we mirror that.
            rhs_id
        };

    let bias_dims = graph.node(bias_src_id).shape.dims();
    if bias_dims.len() != 1 || bias_dims[0] != last_dim { return None; }

    // Conservativeness: the inner MatMul must have THIS Add as its sole
    // consumer. Otherwise the fusion duplicates the matmul; we'd rather
    // leave the graph alone.
    let consumer_counts = count_consumers(graph);
    if consumer_counts.get(&mm_id).copied().unwrap_or(0) != 1 {
        return None;
    }

    Some(PatternMatch {
        bindings: vec![(0, a_id), (1, b_id), (2, bias_src_id)],
        params:   FusedOpParams::FusedLinear,
    })
}

/// Build a consumer-count index across the entire graph. Mirrors
/// `softmax_last_dim::count_consumers`; replicated rather than shared
/// so the registry module's matchers stay self-contained.
fn count_consumers(graph: &Graph) -> HashMap<NodeId, usize> {
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    let n = graph.len();
    for nid in 0..n {
        let node = graph.node(NodeId(nid));
        for &input in &node.inputs {
            *counts.entry(input).or_insert(0) += 1;
        }
    }
    counts
}

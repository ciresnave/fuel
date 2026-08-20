// SPDX-License-Identifier: MIT OR Apache-2.0
//! FusedLinear — first multi-input fused op migrated through the
//! FusedOpRegistry. Phase 7.6 step 4; migrated to a portable `PatternNode`
//! DATA recipe in Increment C slice 2 (S2-2) — the FIRST recipe to drive
//! `WithDim` through live emit.
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (decompose, pattern
//!   matcher, shape/dtype rules).
//! - [`recipe`] — the op's primitive subgraph as portable data:
//!   `Add(MatMul(a, b), BroadcastTo(WithDim{op:0, axis:LAST, dim:Extent{op:1,
//!   LAST}})(bias))`. The bias broadcast target is a's shape with its LAST axis
//!   (K) replaced by b's LAST extent (N) = the matmul output `[..batch, M, N]`.
//! - [`decompose`] — re-emits [`recipe`] through the
//!   [`crate::registry::decompose_via_recipe`] bridge.
//! - [`canonical_pattern`] — recognizes `Add(MatMul(a, b), bias_broadcast)` in
//!   BOTH spellings: the legacy direct-rank-1-bias form AND the recipe form,
//!   which inserts a leading-1 pad `Reshape` on the bias (the D4 rank-raise
//!   materialized by emit for the WithDim-derived broadcast target). Produces a
//!   [`PatternMatch`] with `bindings = [(0, a), (1, b), (2, bias)]`.
//!
//! Replaces the hand-written rewrite in [`crate::opt::fuse_linear`]: that
//! function now also emits `Op::Fused(FUSED_LINEAR, _)` so direct callers (the
//! `fuse_linear` oracle test) and the auto-generated `FusionRule` produce the
//! same shape.
//!
//! Note: the WithDim broadcast target is Fuel-INTERNAL shape resolution (the
//! recipe interior), NOT §6.19 wire emission — so this migration is NOT gated
//! on KISS #86; exercising `WithDim` through live emit here IS the end-to-end
//! proof of the evaluator + routing that already existed unexercised.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps, PatternMatch,
    SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId, Op};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::shape_expr::{Dim, LAST, ShapeExpr};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Metadata-side registry entry for FusedLinear.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: None,
        id: FusedOps::FUSED_LINEAR,
        name: "FusedLinear",
        family: FusedOpFamily::Forward,
        pattern: SubgraphPattern::Callable(canonical_pattern),
        decompose,
        // The backward fused-op isn't part of the registry yet (a later
        // step migrates each fused-backward helper). NodeHandle::backward
        // dispatches `Op::Fused(FUSED_LINEAR, _)` through a per-id arm
        // that emits the same three-grad decomposition as the legacy
        // `Op::FusedLinear` arm.
        backward: BackwardKind::NotDifferentiable,
        shape_rule: matmul_output_shape,
        dtype_rule: dtype_passthrough,
        output_views: None,
    }
}

/// Shape rule: FusedLinear output shape is the matmul output —
/// `[..., M, N]` where `a` is `[..., M, K]` and `b` is `[..., K, N]`.
/// Bias is rank-1 `[N]` and broadcasts implicitly.
fn matmul_output_shape(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(),
        3,
        "FusedLinear takes 3 inputs (a, b, bias)"
    );
    let a = &input_shapes[0];
    let b = &input_shapes[1];
    let a_rank = a.rank();
    let b_rank = b.rank();
    debug_assert!(a_rank >= 2 && b_rank >= 2);
    debug_assert_eq!(a_rank, b_rank, "FusedLinear: a/b ranks must match");
    let mut dims: Vec<usize> = a.dims()[..a_rank - 2].to_vec();
    dims.push(a.dims()[a_rank - 2]); // M
    dims.push(b.dims()[b_rank - 1]); // N
    Shape::from_dims(&dims)
}

/// Dtype rule: FusedLinear output dtype matches `a` (all three inputs
/// must agree at construction time).
fn dtype_passthrough(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(input_dtypes.len(), 3, "FusedLinear takes 3 inputs");
    input_dtypes[0]
}

/// FusedLinear's primitive recipe as **portable data** (Increment C slice 2,
/// S2-2). The FIRST recipe to drive `WithDim` through live emit: the bias
/// broadcast target is `WithDim { operand: 0, axis: LAST, dim: Extent {
/// operand: 1, axis: LAST } }` — a's shape (bind 0) with its LAST axis (K)
/// replaced by b's LAST extent (N), i.e. the matmul output `[..batch, M, N]`.
/// This is Fuel-INTERNAL shape resolution (recipe interior), NOT §6.19 wire
/// emission. Shape-/rank-polymorphic: nothing bakes a shape or a rank; the
/// `MatMul` carries empty roles (the rank-polymorphic implicit-accept cell).
/// Parameterless (no open scalar slots). Binds: `0 = a`, `1 = b`, `2 = bias`.
///
/// ```text
///   mm        = MatMul(a, b)                                   # [..batch, M, N]
///   bias_bcst = BroadcastTo(WithDim{op0, LAST, Extent{op1, LAST}})(bias)
///   out       = Add(mm, bias_bcst)
/// ```
///
/// The rank-1 bias `[N]` rank-raises to the matmul output, so emit materializes
/// a leading-1 pad `Reshape([N] → [1,..,1,N])` (the D4 path) between the bias
/// and the BroadcastTo — [`canonical_pattern`] sees through that pad so the
/// lower→fuse round-trip stays green.
fn recipe() -> &'static PatternNode {
    static RECIPE: OnceLock<PatternNode> = OnceLock::new();
    RECIPE.get_or_init(|| {
        let op = |op, attrs, operands| PatternNode::Op {
            op,
            attrs,
            operands,
        };
        let a = || PatternNode::Bind { index: 0 };
        let b = || PatternNode::Bind { index: 1 };
        let bias = || PatternNode::Bind { index: 2 };
        // bias broadcast target = the matmul output shape = a's shape with its
        // LAST axis (K) replaced by b's LAST extent (N).
        let bias_target = OpAttrs {
            target_shape_rel: Some(ShapeExpr::WithDim {
                operand: 0,
                axis: LAST,
                dim: Box::new(Dim::Extent {
                    operand: 1,
                    axis: LAST,
                }),
            }),
            ..OpAttrs::default()
        };
        op(
            OpTag::Add,
            OpAttrs::default(),
            vec![
                op(OpTag::MatMul, OpAttrs::default(), vec![a(), b()]),
                op(OpTag::BroadcastTo, bias_target, vec![bias()]),
            ],
        )
    })
}

/// Per-entry scalar projection: FusedLinear is parameterless (no per-instance
/// params — the bias is an INPUT, not a baked scalar), so the right payload
/// projects to ZERO open-slot scalars and any other payload is a typed decline
/// (`None` ⇒ the bridge returns the node unchanged — G2).
fn scalars(params: &FusedOpParams) -> Option<Vec<f64>> {
    match params {
        FusedOpParams::FusedLinear => Some(Vec::new()),
        _ => None,
    }
}

/// Lower a fused FusedLinear node to its primitive subgraph and return the new
/// root id — since S2-2 a re-emit of [`recipe`]'s data through the
/// [`decompose_via_recipe`] bridge (the fused node's three inputs are the binds
/// `[a, b, bias]`; the resolving emit derives the WithDim bias-broadcast target
/// and materializes the D4 rank-pad `Reshape`). Backends without a true fused
/// kernel execute this primitive form; backends with one (CUTLASS bias-
/// epilogue, cuBLAS gemm-with-bias) register a `BackendImpl` and skip lowering.
///
/// Any failure — wrong params payload, a resolution decline at these shapes —
/// returns `id` (fixpoint, surfaced gap, never a panic): exactly the G2 posture
/// the imperative body had.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    decompose_via_recipe(graph, id, recipe(), scalars(params))
}

/// Match the canonical FusedLinear pattern rooted at an `Add` node:
/// `Add(MatMul(a, b), bias_broadcast)` where `bias_broadcast` is
/// `BroadcastTo(bias_src)` (or a rank-1 bias directly), and the rank-1
/// `bias_src`'s length equals the matmul output's last dim.
///
/// Two spellings, both kept (mirrors the norm recipe matchers — never delete
/// the old match):
/// * the LEGACY form — the BroadcastTo operand IS the rank-1 bias directly
///   (what `crate::opt::fuse_linear` and pre-S2-2 lowerings emit);
/// * the RECIPE form — the BroadcastTo operand is a leading-1 pad
///   `Reshape([N] → [1,..,1,N])` of the rank-1 bias (the D4 rank-raise emit
///   materializes for the WithDim broadcast target), which this matcher peels
///   before checking the rank-1 length.
///
/// Returns `bindings = [(0, a), (1, b), (2, bias_src)]` and
/// `params = FusedOpParams::FusedLinear`. Conservative: the inner MatMul must
/// have exactly one consumer (this Add), and a peeled pad `Reshape` must be
/// consumed only by the BroadcastTo — otherwise fusing would force a duplicated
/// matmul or drop a value a consumer reads.
///
/// Direct port of the hand-written walker in [`crate::opt::fuse_linear`]; that
/// walker continues to exist as a stand-alone API (the oracle test calls it
/// directly) but now emits `Op::Fused(FUSED_LINEAR, _)` the same way the
/// auto-generated FusionRule does.
pub fn canonical_pattern(graph: &Graph, add_id: NodeId) -> Option<PatternMatch> {
    let add = graph.node(add_id);
    if !matches!(add.op, Op::Add) {
        return None;
    }
    if add.inputs.len() != 2 {
        return None;
    }
    let mm_id = add.inputs[0];
    let rhs_id = add.inputs[1];

    let mm = graph.node(mm_id);
    if !matches!(mm.op, Op::MatMul) {
        return None;
    }
    if mm.inputs.len() != 2 {
        return None;
    }
    let a_id = mm.inputs[0];
    let b_id = mm.inputs[1];

    let mm_dims = mm.shape.dims();
    if mm_dims.is_empty() {
        return None;
    }
    let last_dim = mm_dims[mm_dims.len() - 1];

    let rhs = graph.node(rhs_id);
    let bcst_operand = if matches!(rhs.op, Op::BroadcastTo(_)) && rhs.inputs.len() == 1 {
        rhs.inputs[0]
    } else {
        // Defensive: a rank-1 bias directly on Add's RHS only type-checks if
        // Add allows implicit broadcasting — the legacy walker accepted it,
        // so we mirror that.
        rhs_id
    };

    // Peel an OPTIONAL leading-1 pad Reshape (the recipe/D4 spelling): the
    // WithDim-derived broadcast target rank-raises the rank-1 bias, so emit
    // inserts a `Reshape([N] → [1,..,1,N])` between the bias and the
    // BroadcastTo. The legacy spelling has no such node.
    let mut pad_reshape_id: Option<NodeId> = None;
    let bias_src_id = {
        let n = graph.node(bcst_operand);
        if matches!(n.op, Op::Reshape(_)) && n.inputs.len() == 1 {
            let inner = n.inputs[0];
            let inner_dims = graph.node(inner).shape.dims();
            let pad_dims = n.shape.dims();
            let is_leading_1_pad = inner_dims.len() == 1
                && !pad_dims.is_empty()
                && pad_dims[pad_dims.len() - 1] == inner_dims[0]
                && pad_dims[..pad_dims.len() - 1].iter().all(|&d| d == 1);
            if is_leading_1_pad {
                pad_reshape_id = Some(bcst_operand);
                inner
            } else {
                bcst_operand
            }
        } else {
            bcst_operand
        }
    };

    let bias_dims = graph.node(bias_src_id).shape.dims();
    if bias_dims.len() != 1 || bias_dims[0] != last_dim {
        return None;
    }

    // Conservativeness: the inner MatMul must have THIS Add as its sole
    // consumer (else fusing duplicates the matmul); and a peeled pad Reshape
    // must be consumed only by the BroadcastTo (else fusing drops a value).
    let consumer_counts = count_consumers(graph);
    if consumer_counts.get(&mm_id).copied().unwrap_or(0) != 1 {
        return None;
    }
    if let Some(pad) = pad_reshape_id {
        if consumer_counts.get(&pad).copied().unwrap_or(0) != 1 {
            return None;
        }
    }

    Some(PatternMatch {
        bindings: vec![(0, a_id), (1, b_id), (2, bias_src_id)],
        params: FusedOpParams::FusedLinear,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Node;

    /// Build a fused FusedLinear node over `a [a_dims]`, `b [b_dims]`, and a
    /// rank-1 `bias [N]` (N = b's last dim). Returns `(a, b, bias, fused)`.
    fn fused_node(
        g: &mut Graph,
        a_dims: &[usize],
        b_dims: &[usize],
    ) -> (NodeId, NodeId, NodeId, NodeId) {
        let ar = a_dims.len();
        let n = b_dims[b_dims.len() - 1];
        let mut out_dims = a_dims[..ar - 2].to_vec();
        out_dims.push(a_dims[ar - 2]);
        out_dims.push(n);
        let a = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(a_dims),
            dtype: DType::F32,
        });
        let b = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(b_dims),
            dtype: DType::F32,
        });
        let bias = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[n]),
            dtype: DType::F32,
        });
        let fused = g.push(Node {
            op: Op::Fused(FusedOps::FUSED_LINEAR, FusedOpParams::FusedLinear),
            inputs: vec![a, b, bias],
            shape: Shape::from_dims(&out_dims),
            dtype: DType::F32,
        });
        (a, b, bias, fused)
    }

    /// Slice-2 re-fuse closure (RISK-C): the recipe emission — which inserts a
    /// leading-1 pad `Reshape` on the bias (the D4 rank-raise driven by the
    /// WithDim broadcast target) — must still be matched by `canonical_pattern`,
    /// so lower → fuse round-trips on the framework's OWN lowered FusedLinear at
    /// rank-2, rank-3 batched, and a GQA-ish batched shape.
    #[test]
    fn canonical_pattern_matches_the_recipe_spelling() {
        for (a_dims, b_dims) in [
            (vec![2usize, 3], vec![3usize, 4]),  // rank-2
            (vec![5, 2, 3], vec![5, 3, 4]),      // rank-3 batched
            (vec![8, 16, 64], vec![8, 64, 128]), // GQA-ish batched
        ] {
            let mut g = Graph::new();
            let (a, b, bias, fused) = fused_node(&mut g, &a_dims, &b_dims);
            let root = decompose(&mut g, fused, &FusedOpParams::FusedLinear);
            assert_ne!(root, fused, "recipe decompose fires at a={a_dims:?}");
            // The recipe emission rank-raises the rank-1 bias → a D4 pad Reshape.
            let reachable = crate::topo_order_multi(&g, &[root]);
            assert!(
                reachable
                    .iter()
                    .any(|&n| matches!(g.node(n).op, Op::Reshape(_))),
                "the WithDim rank-raise materializes a D4 pad Reshape at a={a_dims:?}",
            );
            let m = canonical_pattern(&g, root)
                .expect("the recipe emission must re-fuse (see-through the D4 pad)");
            assert_eq!(
                m.bindings,
                vec![(0, a), (1, b), (2, bias)],
                "binds recovered = [a, b, bias] at a={a_dims:?}",
            );
            assert!(matches!(m.params, FusedOpParams::FusedLinear));
        }
    }

    /// The legacy direct-rank-1-bias spelling keeps matching (never delete the
    /// old match) — hand-built `Add(MatMul, BroadcastTo(rank-1 bias))` with no
    /// pad Reshape.
    #[test]
    fn canonical_pattern_still_matches_the_legacy_spelling() {
        let mut g = Graph::new();
        let f32_node = |op, inputs, shape| Node {
            op,
            inputs,
            shape,
            dtype: DType::F32,
        };
        let a = g.push(f32_node(Op::Const, vec![], Shape::from_dims(&[2, 3])));
        let b = g.push(f32_node(Op::Const, vec![], Shape::from_dims(&[3, 4])));
        let bias = g.push(f32_node(Op::Const, vec![], Shape::from_dims(&[4])));
        let mm = g.push(f32_node(Op::MatMul, vec![a, b], Shape::from_dims(&[2, 4])));
        let bcst = g.push(f32_node(
            Op::BroadcastTo(Shape::from_dims(&[2, 4])),
            vec![bias],
            Shape::from_dims(&[2, 4]),
        ));
        let add = g.push(f32_node(Op::Add, vec![mm, bcst], Shape::from_dims(&[2, 4])));
        let m = canonical_pattern(&g, add).expect("legacy spelling must still match");
        assert_eq!(m.bindings, vec![(0, a), (1, b), (2, bias)]);
        assert!(matches!(m.params, FusedOpParams::FusedLinear));
    }
}

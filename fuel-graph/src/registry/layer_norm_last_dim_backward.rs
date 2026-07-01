//! LayerNormLastDimBackward — fused backward helper for the
//! `LayerNormLastDim` forward. Phase 7.6 step 4 (continued).
//!
//! Recomputes mean/variance from the original x (rather than caching
//! intermediates from forward) and produces grad_x. Two inputs
//! `[x, upstream]` + `eps`.
//!
//! See `softmax_last_dim_backward.rs` for the architectural rationale
//! shared by all four backward-helper entries: the registry's role is
//! identity + `BackwardKind::Fused` target + future `BackendImpl`
//! host, not lowering-to-primitives.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, Shape};

/// Metadata-side registry entry for LayerNormLastDimBackward.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::LAYER_NORM_LAST_DIM_BACKWARD,
        name:       "LayerNormLastDimBackward",
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
        "LayerNormLastDimBackward takes 2 inputs (x, upstream)",
    );
    input_shapes[0].clone()
}

fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "LayerNormLastDimBackward takes 2 inputs",
    );
    input_dtypes[0]
}

/// Decompose to the affine-free LayerNorm backward
/// `grad_x = istd · (g − mean(g) − xhat·mean(g·xhat))`, where
/// `mean(·)` is over the last dim, `xhat = (x − mean(x))·istd`, and
/// `istd = rsqrt(var + eps)`. Recomputes mean/var from `x` using the same
/// `MeanDim → Reshape(keepdim) → BroadcastTo` idiom as the forward
/// `layer_norm_last_dim::decompose`; every primitive exists, so per G2 this is
/// a real decomposition, not a basis-gap self-return.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (x_id, g_id, x_shape, dtype) = {
        let n = graph.node(id);
        (n.inputs[0], n.inputs[1], n.shape.clone(), n.dtype)
    };
    let eps = match params {
        FusedOpParams::LayerNormLastDimBackward { eps } => *eps,
        // G2: total + never-panic — impossible params; return self.
        _ => return id,
    };
    let dims = x_shape.dims().to_vec();
    let last = dims.len() - 1;
    let mut kd = dims.clone();
    kd[last] = 1;
    let keepdim = Shape::from_dims(&kd);
    let mut rd = dims.clone();
    rd.remove(last);
    let reduced = Shape::from_dims(&rd);

    // reduce-mean over the last dim, keepdim, broadcast back to x_shape.
    let mean_b = |graph: &mut Graph, src: NodeId| -> NodeId {
        let m = graph.push(Node {
            op: Op::MeanDim(last),
            inputs: vec![src],
            shape: reduced.clone(),
            dtype,
        });
        let m_kd = graph.push(Node {
            op: Op::Reshape(keepdim.clone()),
            inputs: vec![m],
            shape: keepdim.clone(),
            dtype,
        });
        graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![m_kd],
            shape: x_shape.clone(),
            dtype,
        })
    };

    // xhat = (x − mean(x)) · istd ; istd = rsqrt(var + eps).
    let mean_x = mean_b(graph, x_id);
    let xc = graph.push(Node {
        op: Op::Sub,
        inputs: vec![x_id, mean_x],
        shape: x_shape.clone(),
        dtype,
    });
    let xc_sq = graph.push(Node {
        op: Op::Sqr,
        inputs: vec![xc],
        shape: x_shape.clone(),
        dtype,
    });
    let var = mean_b(graph, xc_sq);
    let var_eps = graph.push(Node {
        op: Op::AddScalar(eps),
        inputs: vec![var],
        shape: x_shape.clone(),
        dtype,
    });
    let istd = graph.push(Node {
        op: Op::Rsqrt,
        inputs: vec![var_eps],
        shape: x_shape.clone(),
        dtype,
    });
    let xhat = graph.push(Node {
        op: Op::Mul,
        inputs: vec![xc, istd],
        shape: x_shape.clone(),
        dtype,
    });

    // grad_x = istd · (g − mean(g) − xhat·mean(g·xhat)).
    let mean_g = mean_b(graph, g_id);
    let g_xhat = graph.push(Node {
        op: Op::Mul,
        inputs: vec![g_id, xhat],
        shape: x_shape.clone(),
        dtype,
    });
    let mean_gxh = mean_b(graph, g_xhat);
    let t1 = graph.push(Node {
        op: Op::Sub,
        inputs: vec![g_id, mean_g],
        shape: x_shape.clone(),
        dtype,
    });
    let t2 = graph.push(Node {
        op: Op::Mul,
        inputs: vec![xhat, mean_gxh],
        shape: x_shape.clone(),
        dtype,
    });
    let inner = graph.push(Node {
        op: Op::Sub,
        inputs: vec![t1, t2],
        shape: x_shape.clone(),
        dtype,
    });
    graph.push(Node {
        op: Op::Mul,
        inputs: vec![istd, inner],
        shape: x_shape,
        dtype,
    })
}

pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

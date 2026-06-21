//! RmsNormLastDimBackward — fused backward helper for the
//! `RmsNormLastDim` forward. Phase 7.6 step 4 (continued).
//!
//! Closed-form: `grad_x = r_rms · (upstream - x · s / (n · (mean_sq + eps)))`
//! where `r_rms = 1/sqrt(mean_sq + eps)`, `s = sum(upstream · x, last)`,
//! `n = last_dim_size`. Two inputs `[x, upstream]` + `eps`.
//!
//! See `softmax_last_dim_backward.rs` for the shared architectural
//! rationale.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_core_types::{DType, Shape};

/// Metadata-side registry entry for RmsNormLastDimBackward.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::RMS_NORM_LAST_DIM_BACKWARD,
        name:       "RmsNormLastDimBackward",
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
        "RmsNormLastDimBackward takes 2 inputs (x, upstream)",
    );
    input_shapes[0].clone()
}

fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "RmsNormLastDimBackward takes 2 inputs",
    );
    input_dtypes[0]
}

/// Decompose to the closed-form RMSNorm backward
/// `grad_x = r_rms · (g − x·s / (n·(mean_sq + eps)))`, where
/// `r_rms = rsqrt(mean_sq + eps)`, `mean_sq = mean(x², last)`,
/// `s = sum(g·x, last)`, `n = last-dim size`, `g = upstream`. Mirrors the
/// forward `rms_norm_last_dim::decompose` `MeanDim → Reshape(keepdim)` idiom;
/// every primitive exists (`Sqr`, `MeanDim`, `SumDim`, `Reshape`, `AddScalar`,
/// `MulScalar`, `Rsqrt`, `BroadcastTo`, `Mul`, `Sub`, `Div`), so per G2 this is
/// a real decomposition, not a basis-gap self-return.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (x_id, up_id, x_shape, dtype) = {
        let n = graph.node(id);
        (n.inputs[0], n.inputs[1], n.shape.clone(), n.dtype)
    };
    let eps = match params {
        FusedOpParams::RmsNormLastDimBackward { eps } => *eps,
        // G2: total + never-panic — impossible params; return self.
        _ => return id,
    };
    let dims = x_shape.dims().to_vec();
    let last = dims.len() - 1;
    let n = dims[last] as f64;
    let mut kd = dims.clone();
    kd[last] = 1;
    let keepdim = Shape::from_dims(&kd);
    let mut rd = dims.clone();
    rd.remove(last);
    let reduced = Shape::from_dims(&rd);

    // denom = mean(x²) + eps  (keepdim).
    let sq = graph.push(Node {
        op: Op::Sqr,
        inputs: vec![x_id],
        shape: x_shape.clone(),
        dtype,
    });
    let mean = graph.push(Node {
        op: Op::MeanDim(last),
        inputs: vec![sq],
        shape: reduced.clone(),
        dtype,
    });
    let mean_kd = graph.push(Node {
        op: Op::Reshape(keepdim.clone()),
        inputs: vec![mean],
        shape: keepdim.clone(),
        dtype,
    });
    let denom_kd = graph.push(Node {
        op: Op::AddScalar(eps),
        inputs: vec![mean_kd],
        shape: keepdim.clone(),
        dtype,
    });
    // r_rms = rsqrt(denom), broadcast.
    let rrms_kd = graph.push(Node {
        op: Op::Rsqrt,
        inputs: vec![denom_kd],
        shape: keepdim.clone(),
        dtype,
    });
    let rrms_b = graph.push(Node {
        op: Op::BroadcastTo(x_shape.clone()),
        inputs: vec![rrms_kd],
        shape: x_shape.clone(),
        dtype,
    });
    // s = sum(g·x, last)  (keepdim).
    let gx = graph.push(Node {
        op: Op::Mul,
        inputs: vec![up_id, x_id],
        shape: x_shape.clone(),
        dtype,
    });
    let s = graph.push(Node {
        op: Op::SumDim(last),
        inputs: vec![gx],
        shape: reduced,
        dtype,
    });
    let s_kd = graph.push(Node {
        op: Op::Reshape(keepdim.clone()),
        inputs: vec![s],
        shape: keepdim.clone(),
        dtype,
    });
    let s_b = graph.push(Node {
        op: Op::BroadcastTo(x_shape.clone()),
        inputs: vec![s_kd],
        shape: x_shape.clone(),
        dtype,
    });
    // term = x·s / (n·denom).
    let ndenom_kd = graph.push(Node {
        op: Op::MulScalar(n),
        inputs: vec![denom_kd],
        shape: keepdim.clone(),
        dtype,
    });
    let ndenom_b = graph.push(Node {
        op: Op::BroadcastTo(x_shape.clone()),
        inputs: vec![ndenom_kd],
        shape: x_shape.clone(),
        dtype,
    });
    let xs = graph.push(Node {
        op: Op::Mul,
        inputs: vec![x_id, s_b],
        shape: x_shape.clone(),
        dtype,
    });
    let term = graph.push(Node {
        op: Op::Div,
        inputs: vec![xs, ndenom_b],
        shape: x_shape.clone(),
        dtype,
    });
    // grad_x = r_rms · (g − term).
    let inner = graph.push(Node {
        op: Op::Sub,
        inputs: vec![up_id, term],
        shape: x_shape.clone(),
        dtype,
    });
    graph.push(Node {
        op: Op::Mul,
        inputs: vec![rrms_b, inner],
        shape: x_shape,
        dtype,
    })
}

pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

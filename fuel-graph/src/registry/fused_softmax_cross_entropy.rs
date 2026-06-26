//! FusedSoftmaxCrossEntropy — fused softmax + negative log-likelihood
//! with integer (class-index) targets. The standard PyTorch /
//! Liger-Kernel training-time loss. Phase 7.6 step 4 (continued —
//! 11th op migrated; the first non-baracuda-derived fused op added
//! through the registry).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   `decompose` to a stable primitive subgraph, stubbed pattern).
//!
//! Inputs: `[logits, targets]`.
//!   - `logits`:  `[..., V]` F32 — unnormalized class scores.
//!   - `targets`: `[...]` I64 — class indices. I64 matches PyTorch's
//!     `CrossEntropyLoss(target: int64)` and baracuda's `per_row`
//!     kernel signature, picked over Fuel's U32 indexing convention
//!     because CE callers typically port from external code (see
//!     `feedback_match_external_convention_for_well_known_ops`).
//!
//! Output: F32 regardless of input dtype — loss values accumulate in
//! F32 for stability, matching PyTorch and the baracuda kernel.
//! Output shape depends on the [`crate::registry::Reduction`] mode:
//!   - `Mean` / `Sum`: scalar `[]`.
//!   - `None`:         same as `targets.shape` (`[...]`).
//!
//! ## Why this exists (memory win)
//!
//! `fuel-core::train::cross_entropy_with_logits` materializes ~7
//! `[..., V]`-shaped intermediates (max-broadcast, shifted, exp,
//! log-sum-broadcast, log-softmax, per-elem product). For Llama-7B
//! with `V=32000, batch=8, seq=2048` this is ~12 GiB of transient
//! allocation. The fused CPU forward kernel only needs a `[N]`
//! per-row accumulator plus the inputs — ~6000× less memory.
//!
//! ## Backward
//!
//! [`crate::registry::BackwardKind::Decompose`]: autograd lowers the
//! fused node via [`decompose`] and runs the normal primitive
//! backward. The lowered form re-introduces the `[..., V]`
//! intermediates, so training-time peak memory matches the primitive
//! `cross_entropy_with_logits` chain — only the forward saves memory.
//! A Liger-style in-place fused backward kernel (single launch that
//! writes `softmax(x) - one_hot(target)` straight back over `x`) is a
//! later session and requires the in-place mutation infrastructure
//! (Phase 4 of in-place ops).
//!
//! ## `ignore_index` in `decompose`
//!
//! The decomposition does NOT mask `ignore_index` rows — every row
//! contributes to the lowered subgraph's loss. The CPU forward kernel
//! does mask correctly. Practically this only matters if (a) the user
//! calls backward through the fused op AND (b) some real target value
//! equals the `ignore_index` sentinel — the conventional `-100` never
//! collides with a non-negative class index, so the lowered backward
//! is correct for the typical case. Datasets that actually use
//! ignore-masking during training should keep using
//! [`fuel_core::train::loss::cross_entropy_with_logits`] (the
//! primitive composition path) until the in-place fused backward
//! lands.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, Reduction, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, Shape};

/// Metadata-side registry entry for FusedSoftmaxCrossEntropy.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY,
        name:       "FusedSoftmaxCrossEntropy",
        family:     FusedOpFamily::Forward,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::Decompose,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Output shape rule. Mean/Sum collapse to scalar; None preserves the
/// per-row shape (`targets.shape`, which is `logits.shape` minus the
/// trailing vocab dim).
fn shape_rule(input_shapes: &[Shape], params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 2,
        "FusedSoftmaxCrossEntropy takes 2 inputs (logits, targets)",
    );
    let reduction = match params {
        FusedOpParams::FusedSoftmaxCrossEntropy { reduction, .. } => *reduction,
        other => panic!(
            "fused_softmax_cross_entropy::shape_rule got non-FSCE params: {other:?}"
        ),
    };
    match reduction {
        Reduction::Mean | Reduction::Sum => Shape::from_dims(&[]),
        Reduction::None => input_shapes[1].clone(),
    }
}

/// Output dtype rule — always F32 (loss accumulator dtype),
/// independent of logits/targets dtype.
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "FusedSoftmaxCrossEntropy takes 2 inputs",
    );
    DType::F32
}

/// Lower a fused FusedSoftmaxCrossEntropy node to its primitive
/// subgraph and return the new root id. The lowered form is:
///
/// ```text
///   // log-softmax along last dim (numerically stable)
///   m            = ReduceMaxTo([..., 1])(logits)
///   mb           = BroadcastTo([..., V])(m)
///   shifted      = Sub(logits, mb)
///   exp_shift    = Exp(shifted)
///   sum_exp      = ReduceSumTo([..., 1])(exp_shift)
///   log_sum_exp  = Log(sum_exp)
///   log_sum_bcst = BroadcastTo([..., V])(log_sum_exp)
///   log_softmax  = Sub(shifted, log_sum_bcst)        # [..., V] F32
///
///   // Gather per-row NLL at the target indices.
///   targets_u32       = Cast(U32)(targets)            # [...]
///   targets_keepdim   = Unsqueeze{dim: last}(targets_u32)  # [..., 1]
///   gathered          = Gather{dim: last}(log_softmax, targets_keepdim)
///                                                    # [..., 1]
///   per_row_log_lik   = Squeeze{dim: last}(gathered)  # [...]
///   per_row_nll       = MulScalar(-1.0)(per_row_log_lik)  # [...]
///
///   // Reduce per the mode.
///   match reduction:
///     None: out = per_row_nll                         # [...]
///     Sum:  out = ReduceSumTo([])(per_row_nll)        # []
///     Mean: out = MulScalar(1/N)(ReduceSumTo([])(per_row_nll))  # []
/// ```
///
/// `ignore_index` is not honored in the lowered form (see module
/// docs). Targets must be I64; the cast to U32 happens here because
/// `Op::Gather` indexes via U32 today.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (reduction, _ignore_index) = match params {
        FusedOpParams::FusedSoftmaxCrossEntropy { reduction, ignore_index } => {
            (*reduction, *ignore_index)
        }
        other => panic!(
            "fused_softmax_cross_entropy::decompose got non-FSCE params: {other:?}"
        ),
    };
    let (logits_id, targets_id, logits_shape, targets_shape, _logits_dtype, targets_dtype) = {
        let n = graph.node(id);
        debug_assert_eq!(n.inputs.len(), 2, "FSCE node must have 2 inputs");
        let logits = graph.node(n.inputs[0]);
        let targets = graph.node(n.inputs[1]);
        (
            n.inputs[0],
            n.inputs[1],
            logits.shape.clone(),
            targets.shape.clone(),
            logits.dtype,
            targets.dtype,
        )
    };
    let logits_dims = logits_shape.dims().to_vec();
    let rank = logits_dims.len();
    debug_assert!(
        rank >= 1,
        "FusedSoftmaxCrossEntropy: logits must have rank ≥ 1 (got {logits_dims:?})"
    );
    let last = rank - 1;
    let vocab = logits_dims[last];

    // logits dtype is whatever the user supplied; the lowered subgraph
    // computes log-softmax in that dtype to mirror cross_entropy_with_logits.
    let work_dtype = graph.node(logits_id).dtype;
    let mut keepdim_dims = logits_dims.clone();
    keepdim_dims[last] = 1;
    let keepdim_shape = Shape::from_dims(&keepdim_dims);

    // log-softmax along last dim, numerically stable.
    let m_id = graph.push(Node {
        op:     Op::ReduceMaxTo(keepdim_shape.clone()),
        inputs: vec![logits_id],
        shape:  keepdim_shape.clone(),
        dtype:  work_dtype,
    });
    let mb_id = graph.push(Node {
        op:     Op::BroadcastTo(logits_shape.clone()),
        inputs: vec![m_id],
        shape:  logits_shape.clone(),
        dtype:  work_dtype,
    });
    let shifted_id = graph.push(Node {
        op:     Op::Sub,
        inputs: vec![logits_id, mb_id],
        shape:  logits_shape.clone(),
        dtype:  work_dtype,
    });
    let exp_shift_id = graph.push(Node {
        op:     Op::Exp,
        inputs: vec![shifted_id],
        shape:  logits_shape.clone(),
        dtype:  work_dtype,
    });
    let sum_exp_id = graph.push(Node {
        op:     Op::ReduceSumTo(keepdim_shape.clone()),
        inputs: vec![exp_shift_id],
        shape:  keepdim_shape.clone(),
        dtype:  work_dtype,
    });
    let log_sum_id = graph.push(Node {
        op:     Op::Log,
        inputs: vec![sum_exp_id],
        shape:  keepdim_shape.clone(),
        dtype:  work_dtype,
    });
    let log_sum_bcst_id = graph.push(Node {
        op:     Op::BroadcastTo(logits_shape.clone()),
        inputs: vec![log_sum_id],
        shape:  logits_shape.clone(),
        dtype:  work_dtype,
    });
    let log_softmax_id = graph.push(Node {
        op:     Op::Sub,
        inputs: vec![shifted_id, log_sum_bcst_id],
        shape:  logits_shape.clone(),
        dtype:  work_dtype,
    });

    // Gather log-softmax at the target indices.
    //
    // Op::Gather expects U32 indices; targets are I64 (PyTorch
    // convention). Insert a Cast first. If the upstream already
    // produced U32 (some test setups do), the Cast is a no-op the
    // executor short-circuits.
    let targets_u32_id = if targets_dtype == DType::U32 {
        targets_id
    } else {
        graph.push(Node {
            op:     Op::Cast(DType::U32),
            inputs: vec![targets_id],
            shape:  targets_shape.clone(),
            dtype:  DType::U32,
        })
    };
    let targets_keepdim_id = graph.push(Node {
        op:     Op::Unsqueeze { dim: last },
        inputs: vec![targets_u32_id],
        shape:  keepdim_shape.clone(),
        dtype:  DType::U32,
    });
    let gathered_id = graph.push(Node {
        op:     Op::Gather { dim: last },
        inputs: vec![log_softmax_id, targets_keepdim_id],
        shape:  keepdim_shape.clone(),
        dtype:  work_dtype,
    });
    let per_row_log_lik_id = graph.push(Node {
        op:     Op::Squeeze { dim: last },
        inputs: vec![gathered_id],
        shape:  targets_shape.clone(),
        dtype:  work_dtype,
    });
    let per_row_nll_id = graph.push(Node {
        op:     Op::MulScalar(-1.0),
        inputs: vec![per_row_log_lik_id],
        shape:  targets_shape.clone(),
        dtype:  work_dtype,
    });

    // Reduce per the requested mode.
    //
    // FSCE's declared output dtype is F32 (see `dtype_rule`); when
    // the working dtype is not F32, the reduction path ends with a
    // Cast so the lowered subgraph's output dtype matches the fused
    // node's declared dtype. Otherwise consumer dtype-checks will
    // trip on the substitution.
    let needs_cast = work_dtype != DType::F32;
    match reduction {
        Reduction::None => {
            if needs_cast {
                graph.push(Node {
                    op:     Op::Cast(DType::F32),
                    inputs: vec![per_row_nll_id],
                    shape:  targets_shape,
                    dtype:  DType::F32,
                })
            } else {
                per_row_nll_id
            }
        }
        Reduction::Sum => {
            let sum_id = graph.push(Node {
                op:     Op::ReduceSumTo(Shape::from_dims(&[])),
                inputs: vec![per_row_nll_id],
                shape:  Shape::from_dims(&[]),
                dtype:  work_dtype,
            });
            if needs_cast {
                graph.push(Node {
                    op:     Op::Cast(DType::F32),
                    inputs: vec![sum_id],
                    shape:  Shape::from_dims(&[]),
                    dtype:  DType::F32,
                })
            } else {
                sum_id
            }
        }
        Reduction::Mean => {
            // Mean over the count of rows (vocab is the inner dim
            // we just reduced across via gather; the outer count is
            // logits.elem_count() / vocab).
            let n_rows: usize = logits_dims[..last].iter().product::<usize>().max(1);
            let _ = vocab; // satisfy linter; vocab is intrinsic to logits_shape
            let sum_id = graph.push(Node {
                op:     Op::ReduceSumTo(Shape::from_dims(&[])),
                inputs: vec![per_row_nll_id],
                shape:  Shape::from_dims(&[]),
                dtype:  work_dtype,
            });
            let mean_id = graph.push(Node {
                op:     Op::MulScalar(1.0 / n_rows as f64),
                inputs: vec![sum_id],
                shape:  Shape::from_dims(&[]),
                dtype:  work_dtype,
            });
            if needs_cast {
                graph.push(Node {
                    op:     Op::Cast(DType::F32),
                    inputs: vec![mean_id],
                    shape:  Shape::from_dims(&[]),
                    dtype:  DType::F32,
                })
            } else {
                mean_id
            }
        }
    }
}

/// Pattern matcher stub. FusedSoftmaxCrossEntropy doesn't autoregister
/// itself by recognizing primitive subgraphs — users opt in via the
/// explicit `Tensor::fused_softmax_cross_entropy` builder. The
/// primitive `cross_entropy_with_logits` chain stays in place for
/// callers that don't want the fused kernel.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

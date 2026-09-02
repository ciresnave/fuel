// SPDX-License-Identifier: MIT OR Apache-2.0
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
//! `fuel::train::loss::cross_entropy_with_logits` (the
//! primitive composition path) until the in-place fused backward
//! lands.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps, PatternMatch, Reduction,
    SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};

/// Metadata-side registry entry for FusedSoftmaxCrossEntropy.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: None,
        id: FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY,
        name: "FusedSoftmaxCrossEntropy",
        family: FusedOpFamily::Forward,
        pattern: SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward: BackwardKind::Decompose,
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
        input_shapes.len(),
        2,
        "FusedSoftmaxCrossEntropy takes 2 inputs (logits, targets)",
    );
    let reduction = match params {
        FusedOpParams::FusedSoftmaxCrossEntropy { reduction, .. } => *reduction,
        other => panic!("fused_softmax_cross_entropy::shape_rule got non-FSCE params: {other:?}"),
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
        input_dtypes.len(),
        2,
        "FusedSoftmaxCrossEntropy takes 2 inputs",
    );
    DType::F32
}

/// FusedSoftmaxCrossEntropy's log-softmax + gather-NLL primitive recipe as
/// portable [`PatternNode`] DATA (Increment C, C-T1) — the structure-preserving
/// migration of the pre-C-T1 imperative `decompose` body onto the re-emit
/// machinery. This is the cheap proof of the **config-branch** mechanism: it is
/// a `recipe(shapes, dtypes, reduction) -> PatternNode` builder that uses
/// ORDINARY Rust control flow to select structure per call — no new bridge
/// machinery (`decompose_via_recipe` is param-agnostic; the caller builds the
/// per-call recipe then hands it to the bridge). Three config branches:
///   * the `reduction` tail (`None` / `Sum` / `Mean`);
///   * the `targets` dtype (`U32` feeds `Gather` directly; anything else takes a
///     `Cast(U32)` — `Op::Gather` indexes via U32 today);
///   * the output-dtype tail `Cast(F32)` when `work_dtype != F32` (FSCE's
///     declared output dtype is F32; the log-softmax computes in the logits'
///     dtype to mirror `cross_entropy_with_logits`).
///
/// Bind space: `0 = logits [..,V]`, `1 = targets [..]` — the fused node's input
/// order. Every shape is a concrete integer at decompose time, so the
/// shape-target ops (`ReduceMaxTo`/`ReduceSumTo`/`BroadcastTo`) carry BAKED
/// absolute `target_shape` attrs and the `n_rows` Mean divisor is a baked
/// `MulScalar` constant — no open scalar slots, no rel carriers needed. The
/// interior elementwise/gather nodes carry NO shape attr (`primitive_shape`
/// derives them). `shifted` feeds both `exp_shift` and `log_softmax`; emit's
/// within-call identity-share collapses the two occurrences to one node,
/// reconstructing the imperative DAG's sharing.
///
/// The `Sum`/`Mean` tails reduce to a rank-0 (`[]`, scalar) target via
/// `ReduceSumTo` with the [`OpAttrs::rank0_target`] marker (C-T1's rank-0
/// shape-target representation) — an intentional empty-dims shape, distinct from
/// an unset/wildcard `target_shape`.
///
/// The lowered form (per the pre-migration imperative body):
///
/// ```text
///   m            = ReduceMaxTo([..., 1])(logits)
///   mb           = BroadcastTo([..., V])(m)
///   shifted      = Sub(logits, mb)
///   exp_shift    = Exp(shifted)
///   sum_exp      = ReduceSumTo([..., 1])(exp_shift)
///   log_sum_exp  = Log(sum_exp)
///   log_sum_bcst = BroadcastTo([..., V])(log_sum_exp)
///   log_softmax  = Sub(shifted, log_sum_bcst)
///   targets_u32       = [Cast(U32)](targets)          # Cast only if not U32
///   targets_keepdim   = Unsqueeze{dim: last}(targets_u32)
///   gathered          = Gather{dim: last}(log_softmax, targets_keepdim)
///   per_row_log_lik   = Squeeze{dim: last}(gathered)
///   per_row_nll       = MulScalar(-1.0)(per_row_log_lik)
///   match reduction:
///     None: out = [Cast(F32)](per_row_nll)
///     Sum:  out = [Cast(F32)](ReduceSumTo([])(per_row_nll))
///     Mean: out = [Cast(F32)](MulScalar(1/N)(ReduceSumTo([])(per_row_nll)))
/// ```
fn recipe(
    logits_shape: &Shape,
    _targets_shape: &Shape,
    work_dtype: DType,
    targets_dtype: DType,
    reduction: Reduction,
) -> PatternNode {
    use OpTag as T;
    let logits_dims = logits_shape.dims();
    let rank = logits_dims.len();
    let last = rank - 1; // caller guarantees rank >= 1
    let mut keepdim_dims = logits_dims.to_vec();
    keepdim_dims[last] = 1;

    let as_ts = |dims: &[usize]| -> Vec<i64> { dims.iter().map(|&d| d as i64).collect() };
    let logits_ts = as_ts(logits_dims);
    let keepdim_ts = as_ts(&keepdim_dims);

    let op = |op, attrs, operands| PatternNode::Op {
        op,
        attrs,
        operands,
    };
    let bind = |i: u8| PatternNode::Bind { index: i };
    let shape_attr = |ts: Vec<i64>| OpAttrs {
        target_shape: ts,
        ..OpAttrs::default()
    };
    let cast_attr = |dt: DType| OpAttrs {
        cast_dtype: Some(dt.as_str().to_string()),
        ..OpAttrs::default()
    };
    let scalar_attr = |v: f64| OpAttrs {
        scalars: vec![v],
        ..OpAttrs::default()
    };

    // ---- log-softmax along the last dim, numerically stable ---------------
    let m = op(
        T::ReduceMaxTo,
        shape_attr(keepdim_ts.clone()),
        vec![bind(0)],
    );
    let mb = op(T::BroadcastTo, shape_attr(logits_ts.clone()), vec![m]);
    let shifted = op(T::Sub, OpAttrs::default(), vec![bind(0), mb]);
    let exp_shift = op(T::Exp, OpAttrs::default(), vec![shifted.clone()]);
    let sum_exp = op(T::ReduceSumTo, shape_attr(keepdim_ts), vec![exp_shift]);
    let log_sum = op(T::Log, OpAttrs::default(), vec![sum_exp]);
    let log_sum_bcst = op(T::BroadcastTo, shape_attr(logits_ts), vec![log_sum]);
    // `shifted` re-used (identity-share dedups) — mirrors the shared imperative node.
    let log_softmax = op(T::Sub, OpAttrs::default(), vec![shifted, log_sum_bcst]);

    // ---- gather per-row NLL at the target indices -------------------------
    // Config branch: U32 targets index Gather directly; anything else casts.
    let targets_u32 = if targets_dtype == DType::U32 {
        bind(1)
    } else {
        op(T::Cast, cast_attr(DType::U32), vec![bind(1)])
    };
    let targets_keepdim = op(
        T::Unsqueeze,
        OpAttrs {
            dims: vec![last as u8],
            ..OpAttrs::default()
        },
        vec![targets_u32],
    );
    let gathered = op(
        T::Gather,
        OpAttrs {
            axis: Some(last as i64),
            ..OpAttrs::default()
        },
        vec![log_softmax, targets_keepdim],
    );
    let per_row_log_lik = op(
        T::Squeeze,
        OpAttrs {
            dims: vec![last as u8],
            ..OpAttrs::default()
        },
        vec![gathered],
    );
    let per_row_nll = op(T::MulScalar, scalar_attr(-1.0), vec![per_row_log_lik]);

    // ---- reduce per the mode; Cast(F32) tail when work_dtype != F32 -------
    let needs_cast = work_dtype != DType::F32;
    let cast_f32 = |src| op(T::Cast, cast_attr(DType::F32), vec![src]);
    // Rank-0 (`[]`, scalar) reduce target via the C-T1 marker (empty
    // `target_shape` + `rank0_target`), NOT an unset/wildcard target.
    let rank0 = || OpAttrs {
        rank0_target: true,
        ..OpAttrs::default()
    };

    match reduction {
        Reduction::None => {
            if needs_cast {
                cast_f32(per_row_nll)
            } else {
                per_row_nll
            }
        }
        Reduction::Sum => {
            let sum = op(T::ReduceSumTo, rank0(), vec![per_row_nll]);
            if needs_cast { cast_f32(sum) } else { sum }
        }
        Reduction::Mean => {
            // Mean over the outer row count (the gathered inner V-dim is already
            // reduced): logits.elem_count() / vocab = product of the leading dims.
            let n_rows: usize = logits_dims[..last].iter().product::<usize>().max(1);
            let sum = op(T::ReduceSumTo, rank0(), vec![per_row_nll]);
            let mean = op(T::MulScalar, scalar_attr(1.0 / n_rows as f64), vec![sum]);
            if needs_cast { cast_f32(mean) } else { mean }
        }
    }
}

/// Lower a fused FusedSoftmaxCrossEntropy node to its primitive subgraph and
/// return the new root id. Since Increment C C-T1 a re-emit of [`recipe`]'s
/// portable data through the [`decompose_via_recipe`] bridge (structure-
/// preserving: the emitted base map is node-for-node identical to the pre-C-T1
/// imperative body — see the parity test in `tests`). The per-call recipe bakes
/// the concrete shapes and selects the reduction / cast structure via ordinary
/// Rust control flow (the config-branch mechanism).
///
/// Per G2 this is total + never-panic: a wrong-params payload or a malformed
/// node (wrong input arity, rank-0 logits) returns `id` (the driver's fixpoint
/// signal), and any bridge decline (validation, bind-arity, emit) returns `id`
/// too. `ignore_index` is not honored in the lowered form (see module docs);
/// targets must be I64 (or U32) — the U32 cast for `Op::Gather` happens in the
/// recipe.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let reduction = match params {
        FusedOpParams::FusedSoftmaxCrossEntropy { reduction, .. } => *reduction,
        // G2: decompose is total + never-panic. Non-FSCE params here are an
        // impossible registry-dispatch invariant violation; degrade to identity
        // (return self) rather than crash.
        _ => return id,
    };
    let (logits_shape, targets_shape, work_dtype, targets_dtype) = {
        let n = graph.node(id);
        // Malformed node → fixpoint self-return (never panic).
        if n.inputs.len() != 2 {
            return id;
        }
        let logits = graph.node(n.inputs[0]);
        let targets = graph.node(n.inputs[1]);
        (
            logits.shape.clone(),
            targets.shape.clone(),
            logits.dtype,
            targets.dtype,
        )
    };
    // FSCE requires rank ≥ 1 logits (the last dim is the vocab axis). Malformed
    // → fixpoint self-return (the `rank - 1` in `recipe` would otherwise wrap).
    if logits_shape.rank() < 1 {
        return id;
    }

    let recipe_node = recipe(
        &logits_shape,
        &targets_shape,
        work_dtype,
        targets_dtype,
        reduction,
    );
    // No open scalar slots (the -1.0 / 1/N / softplus-free constants are baked).
    decompose_via_recipe(graph, id, &recipe_node, Some(Vec::new()))
}

/// Pattern matcher stub. FusedSoftmaxCrossEntropy doesn't autoregister
/// itself by recognizing primitive subgraphs — users opt in via the
/// explicit `NodeHandle::fused_softmax_cross_entropy` builder. The
/// primitive `cross_entropy_with_logits` chain stays in place for
/// callers that don't want the fused kernel.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::FusedOps;
    use crate::{Node, Op};

    /// FROZEN copy of the pre-C-T1 imperative `fused_softmax_cross_entropy::
    /// decompose` body, verbatim (the explicit-`Shape` `graph.push` spelling),
    /// before C-T1 replaced the live body with the data recipe. The C-T1
    /// structure-preservation oracle: the migrated recipe re-emit must produce a
    /// graph structurally identical to this.
    fn frozen_legacy_decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
        let (reduction, _ignore_index) = match params {
            FusedOpParams::FusedSoftmaxCrossEntropy {
                reduction,
                ignore_index,
            } => (*reduction, *ignore_index),
            _ => return id,
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
        let last = rank - 1;
        let vocab = logits_dims[last];

        let work_dtype = graph.node(logits_id).dtype;
        let mut keepdim_dims = logits_dims.clone();
        keepdim_dims[last] = 1;
        let keepdim_shape = Shape::from_dims(&keepdim_dims);

        let m_id = graph.push(Node {
            op: Op::ReduceMaxTo(keepdim_shape.clone()),
            inputs: vec![logits_id],
            shape: keepdim_shape.clone(),
            dtype: work_dtype,
        });
        let mb_id = graph.push(Node {
            op: Op::BroadcastTo(logits_shape.clone()),
            inputs: vec![m_id],
            shape: logits_shape.clone(),
            dtype: work_dtype,
        });
        let shifted_id = graph.push(Node {
            op: Op::Sub,
            inputs: vec![logits_id, mb_id],
            shape: logits_shape.clone(),
            dtype: work_dtype,
        });
        let exp_shift_id = graph.push(Node {
            op: Op::Exp,
            inputs: vec![shifted_id],
            shape: logits_shape.clone(),
            dtype: work_dtype,
        });
        let sum_exp_id = graph.push(Node {
            op: Op::ReduceSumTo(keepdim_shape.clone()),
            inputs: vec![exp_shift_id],
            shape: keepdim_shape.clone(),
            dtype: work_dtype,
        });
        let log_sum_id = graph.push(Node {
            op: Op::Log,
            inputs: vec![sum_exp_id],
            shape: keepdim_shape.clone(),
            dtype: work_dtype,
        });
        let log_sum_bcst_id = graph.push(Node {
            op: Op::BroadcastTo(logits_shape.clone()),
            inputs: vec![log_sum_id],
            shape: logits_shape.clone(),
            dtype: work_dtype,
        });
        let log_softmax_id = graph.push(Node {
            op: Op::Sub,
            inputs: vec![shifted_id, log_sum_bcst_id],
            shape: logits_shape.clone(),
            dtype: work_dtype,
        });

        let targets_u32_id = if targets_dtype == DType::U32 {
            targets_id
        } else {
            graph.push(Node {
                op: Op::Cast(DType::U32),
                inputs: vec![targets_id],
                shape: targets_shape.clone(),
                dtype: DType::U32,
            })
        };
        let targets_keepdim_id = graph.push(Node {
            op: Op::Unsqueeze { dim: last },
            inputs: vec![targets_u32_id],
            shape: keepdim_shape.clone(),
            dtype: DType::U32,
        });
        let gathered_id = graph.push(Node {
            op: Op::Gather { dim: last },
            inputs: vec![log_softmax_id, targets_keepdim_id],
            shape: keepdim_shape.clone(),
            dtype: work_dtype,
        });
        let per_row_log_lik_id = graph.push(Node {
            op: Op::Squeeze { dim: last },
            inputs: vec![gathered_id],
            shape: targets_shape.clone(),
            dtype: work_dtype,
        });
        let per_row_nll_id = graph.push(Node {
            op: Op::MulScalar(-1.0),
            inputs: vec![per_row_log_lik_id],
            shape: targets_shape.clone(),
            dtype: work_dtype,
        });

        let needs_cast = work_dtype != DType::F32;
        match reduction {
            Reduction::None => {
                if needs_cast {
                    graph.push(Node {
                        op: Op::Cast(DType::F32),
                        inputs: vec![per_row_nll_id],
                        shape: targets_shape,
                        dtype: DType::F32,
                    })
                } else {
                    per_row_nll_id
                }
            }
            Reduction::Sum => {
                let sum_id = graph.push(Node {
                    op: Op::ReduceSumTo(Shape::from_dims(&[])),
                    inputs: vec![per_row_nll_id],
                    shape: Shape::from_dims(&[]),
                    dtype: work_dtype,
                });
                if needs_cast {
                    graph.push(Node {
                        op: Op::Cast(DType::F32),
                        inputs: vec![sum_id],
                        shape: Shape::from_dims(&[]),
                        dtype: DType::F32,
                    })
                } else {
                    sum_id
                }
            }
            Reduction::Mean => {
                let n_rows: usize = logits_dims[..last].iter().product::<usize>().max(1);
                let _ = vocab;
                let sum_id = graph.push(Node {
                    op: Op::ReduceSumTo(Shape::from_dims(&[])),
                    inputs: vec![per_row_nll_id],
                    shape: Shape::from_dims(&[]),
                    dtype: work_dtype,
                });
                let mean_id = graph.push(Node {
                    op: Op::MulScalar(1.0 / n_rows as f64),
                    inputs: vec![sum_id],
                    shape: Shape::from_dims(&[]),
                    dtype: work_dtype,
                });
                if needs_cast {
                    graph.push(Node {
                        op: Op::Cast(DType::F32),
                        inputs: vec![mean_id],
                        shape: Shape::from_dims(&[]),
                        dtype: DType::F32,
                    })
                } else {
                    mean_id
                }
            }
        }
    }

    /// Recursively assert two subgraphs are node-for-node identical (op, shape,
    /// dtype, arity, recursively-equal inputs). A shared leaf (same NodeId — a
    /// bound external input) matches by identity. Shape-sensitive at EVERY node,
    /// so it catches the rank-0 `ReduceSumTo([])` regression: without the C-T1
    /// marker the recipe declines to a fixpoint (no root), and with a WRONG
    /// rank-0 shape it trips here.
    fn assert_structural_eq(g: &Graph, a: NodeId, b: NodeId) {
        if a == b {
            return;
        }
        let na = g.node(a);
        let nb = g.node(b);
        assert_eq!(na.op, nb.op, "op mismatch: {:?} vs {:?}", na.op, nb.op);
        assert_eq!(
            na.shape, nb.shape,
            "shape mismatch at {:?}: {:?} vs {:?}",
            na.op, na.shape, nb.shape
        );
        assert_eq!(na.dtype, nb.dtype, "dtype mismatch at {:?}", na.op);
        assert_eq!(
            na.inputs.len(),
            nb.inputs.len(),
            "arity mismatch at {:?}",
            na.op
        );
        for (&ia, &ib) in na.inputs.iter().zip(nb.inputs.iter()) {
            assert_structural_eq(g, ia, ib);
        }
    }

    /// Build a fused FSCE node over `logits [.., V]` (dtype `work`) and
    /// `targets [..]` (dtype `tgt`). Returns the fused NodeId.
    fn fused_node(
        g: &mut Graph,
        leading: &[usize],
        vocab: usize,
        work: DType,
        tgt: DType,
        reduction: Reduction,
    ) -> NodeId {
        let mut logits_dims = leading.to_vec();
        logits_dims.push(vocab);
        let logits = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&logits_dims),
            dtype: work,
        });
        let targets = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(leading),
            dtype: tgt,
        });
        let out_shape = match reduction {
            Reduction::Mean | Reduction::Sum => Shape::from_dims(&[]),
            Reduction::None => Shape::from_dims(leading),
        };
        g.push(Node {
            op: Op::Fused(
                FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY,
                FusedOpParams::FusedSoftmaxCrossEntropy {
                    reduction,
                    ignore_index: -100,
                },
            ),
            inputs: vec![logits, targets],
            shape: out_shape,
            dtype: DType::F32,
        })
    }

    /// C-T1: the recipe decompose fires for every reduction × targets-dtype
    /// config, and its emitted base map is node-for-node identical to the FROZEN
    /// pre-C-T1 imperative body. Born-red for the `Sum`/`Mean` variants WITHOUT
    /// the rank-0 `target_shape` marker: `ReduceSumTo([])` is unrepresentable
    /// (`shape_from_attr` reads an empty `target_shape` as unset), so
    /// `decompose_via_recipe` declines to a fixpoint and `assert_ne` trips.
    #[test]
    fn fsce_recipe_decompose_matches_frozen_legacy_for_every_config() {
        for reduction in [Reduction::None, Reduction::Sum, Reduction::Mean] {
            for tgt in [DType::I64, DType::U32] {
                for leading in [vec![4usize], vec![2usize, 3]] {
                    let params = FusedOpParams::FusedSoftmaxCrossEntropy {
                        reduction,
                        ignore_index: -100,
                    };
                    let mut g = Graph::new();
                    let fused = fused_node(&mut g, &leading, 5, DType::F32, tgt, reduction);
                    let out_sh = g.node(fused).shape.clone();

                    let new_root = decompose(&mut g, fused, &params);
                    assert_ne!(
                        new_root, fused,
                        "recipe decompose fires (reduction={reduction:?}, tgt={tgt:?}, leading={leading:?})"
                    );
                    assert_eq!(
                        g.node(new_root).shape,
                        out_sh,
                        "output shape matches shape_rule (reduction={reduction:?})"
                    );
                    assert_eq!(
                        g.node(new_root).dtype,
                        DType::F32,
                        "FSCE output dtype is F32"
                    );

                    let legacy_root = frozen_legacy_decompose(&mut g, fused, &params);
                    assert_structural_eq(&g, new_root, legacy_root);
                }
            }
        }
    }

    /// The rank-0 reduce tail exercised through a non-F32 work dtype so the
    /// `Cast(F32)` output tail (the third config branch) also re-emits and
    /// matches the frozen legacy.
    #[test]
    fn fsce_recipe_non_f32_work_dtype_cast_tail_matches_frozen_legacy() {
        for reduction in [Reduction::None, Reduction::Sum, Reduction::Mean] {
            let params = FusedOpParams::FusedSoftmaxCrossEntropy {
                reduction,
                ignore_index: -100,
            };
            let mut g = Graph::new();
            let fused = fused_node(&mut g, &[2, 3], 5, DType::F16, DType::I64, reduction);
            let new_root = decompose(&mut g, fused, &params);
            assert_ne!(
                new_root, fused,
                "recipe decompose fires (F16 work, reduction={reduction:?})"
            );
            assert_eq!(g.node(new_root).dtype, DType::F32, "Cast(F32) output tail");
            let legacy_root = frozen_legacy_decompose(&mut g, fused, &params);
            assert_structural_eq(&g, new_root, legacy_root);
        }
    }

    /// Totality (G2): a wrong params payload declines to a fixpoint, never a
    /// crash, before any emission.
    #[test]
    fn fsce_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
        let mut g = Graph::new();
        let fused = fused_node(&mut g, &[2, 3], 5, DType::F32, DType::I64, Reduction::Mean);
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::Rope);
        assert_eq!(out, fused, "wrong params => typed decline => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }
}

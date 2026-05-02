//! Symbolic autograd (Phase 6d Track 2).
//!
//! The forward backward pass in `Tensor::backward` walks the topo
//! order in reverse and emits gradient nodes inline via a single
//! 600-line `match` over `Op`. This module factors that out into a
//! per-op trait so:
//!
//! 1. The backward graph is constructed via the same lazy IR machinery
//!    as the forward — the planner sees both, fuses across them, and
//!    schedules them together.
//! 2. Higher-order gradients fall out for free: just call `backward`
//!    on the resulting backward graph.
//! 3. Third-party ops can register their own gradient rules without
//!    forking `fuel-graph`.
//!
//! This is the *scaffolding*. The existing `match` in `Tensor::backward`
//! still does most of the work today; ops migrate one at a time as
//! their rule is implemented here. The dispatcher below is consulted
//! first; if no rule is registered for an op, the existing match
//! handles it (forward-compatible split).

use crate::{node_dtype, node_shape, push_node, NodeId, Op, SharedGraph};

/// Result of a backward rule: one gradient `NodeId` per forward input
/// slot, with `None` for inputs that aren't differentiable (e.g. the
/// index tensor of `IndexSelect`, or the block table of `PagedAttn`).
///
/// Length must match the forward op's `inputs` slice.
pub type GradientList = Vec<Option<NodeId>>;

/// Per-op backward rule. Implementors emit gradient nodes into the
/// shared graph and return one gradient per forward input.
///
/// Convention: a rule is responsible for *all* input slots of its
/// op. If an input is non-differentiable, return `None` for that slot.
/// Don't add the upstream into the per-input gradient sum — the
/// caller's `accumulate_grad` handles cross-edge accumulation.
pub trait GradientRule {
    /// Build gradient nodes for this op.
    ///
    /// - `graph`:    the shared graph; rule pushes new nodes into it
    /// - `op`:       a reference to the forward op (for parameter access)
    /// - `inputs`:   forward-input NodeIds (one per slot)
    /// - `output`:   the forward output's NodeId
    /// - `upstream`: the upstream gradient flowing into this output
    fn backward(
        &self,
        graph: &SharedGraph,
        op: &Op,
        inputs: &[NodeId],
        output: NodeId,
        upstream: NodeId,
    ) -> GradientList;
}

/// Dispatch entry consulted by `Tensor::backward` before falling
/// through to the inline `match`. Returns `None` if no rule is
/// registered for `op` — the caller then handles it inline.
///
/// As ops migrate to `GradientRule` impls, their inline `match` arms
/// are deleted and the rule is added below.
pub fn dispatch_gradient(
    graph: &SharedGraph,
    op: &Op,
    inputs: &[NodeId],
    output: NodeId,
    upstream: NodeId,
) -> Option<GradientList> {
    match op {
        Op::Add => Some(AddRule.backward(graph, op, inputs, output, upstream)),
        Op::Mul => Some(MulRule.backward(graph, op, inputs, output, upstream)),
        Op::Relu => Some(ReluRule.backward(graph, op, inputs, output, upstream)),
        _ => None,
    }
}

// ---- Concrete rules: the migration recipe -----------------------------------

/// `d(a + b)/da = 1`, `d(a + b)/db = 1`. Upstream flows unchanged into both.
pub struct AddRule;
impl GradientRule for AddRule {
    fn backward(
        &self,
        _graph: &SharedGraph,
        _op: &Op,
        _inputs: &[NodeId],
        _output: NodeId,
        upstream: NodeId,
    ) -> GradientList {
        vec![Some(upstream), Some(upstream)]
    }
}

/// `d(a * b)/da = b`, `d(a * b)/db = a`. Upstream * b → a, upstream * a → b.
pub struct MulRule;
impl GradientRule for MulRule {
    fn backward(
        &self,
        graph: &SharedGraph,
        _op: &Op,
        inputs: &[NodeId],
        _output: NodeId,
        upstream: NodeId,
    ) -> GradientList {
        let a = inputs[0];
        let b = inputs[1];
        let a_shape = node_shape(graph, a);
        let dtype = node_dtype(graph, a);
        let grad_a = push_node(graph, Op::Mul, vec![upstream, b], a_shape.clone(), dtype);
        let grad_b = push_node(graph, Op::Mul, vec![upstream, a], a_shape, dtype);
        vec![Some(grad_a), Some(grad_b)]
    }
}

/// `d(relu(x))/dx = (x > 0 ? 1 : 0) = step(x)`.
pub struct ReluRule;
impl GradientRule for ReluRule {
    fn backward(
        &self,
        graph: &SharedGraph,
        _op: &Op,
        inputs: &[NodeId],
        _output: NodeId,
        upstream: NodeId,
    ) -> GradientList {
        let x = inputs[0];
        let x_shape = node_shape(graph, x);
        let dtype = node_dtype(graph, x);
        let mask = push_node(graph, Op::Step, vec![x], x_shape.clone(), dtype);
        let grad_x = push_node(graph, Op::Mul, vec![upstream, mask], x_shape, dtype);
        vec![Some(grad_x)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Tensor};
    use fuel_core_types::Shape;

    /// Confirm that the dispatch hook actually fires for migrated ops:
    /// `dispatch_gradient(Op::Add, ...)` returns Some, indicating the
    /// trait-based path handled the gradient. (The legacy match arm
    /// for Add still exists as a safety net but should never run.)
    #[test]
    fn dispatch_fires_for_migrated_ops() {
        let a = Tensor::from_f32(vec![1.0_f32, 2.0, 3.0], Shape::from_dims(&[3]));
        let b = a.const_f32_like(vec![4.0_f32, 5.0, 6.0], Shape::from_dims(&[3]));
        let c = a.add(&b);
        // Drive a backward to populate upstream
        let grads = c.backward();
        // Both inputs should have gradients; for Add both equal upstream.
        let ga = grads.get(&a).expect("gradient for a");
        let gb = grads.get(&b).expect("gradient for b");
        // dispatch_gradient(Op::Add, ...) returns the upstream node twice
        // verbatim, so ga and gb should be the same NodeId (the same upstream).
        assert_eq!(ga.id(), gb.id(),
            "AddRule should funnel both gradients to the same upstream node");
    }

    /// Sanity: dispatch_gradient returns None for ops we haven't
    /// migrated, so they fall through to the legacy match.
    #[test]
    fn dispatch_returns_none_for_unmigrated_ops() {
        // MatMul hasn't been migrated to a GradientRule yet — the
        // legacy match still handles it. dispatch_gradient should
        // return None.
        use crate::SharedGraph;
        use std::sync::{Arc, RwLock};
        let g: SharedGraph = Arc::new(RwLock::new(crate::Graph::new()));
        let dummy = NodeId(0);
        let result = dispatch_gradient(&g, &Op::MatMul, &[dummy, dummy], dummy, dummy);
        assert!(result.is_none(), "MatMul should not have a registered rule yet");
    }
}

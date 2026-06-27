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
        // --- comparison family: non-differentiable, terminate cleanly ---
        Op::Equal | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => Some(NoGradientBinaryRule.backward(graph, op, inputs, output, upstream)),
        // --- ternary select: differentiable through `a` and `b` ---
        Op::Where => Some(WhereRule.backward(graph, op, inputs, output, upstream)),
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

/// Backward rule for non-differentiable two-input ops (the comparison
/// family: `Equal`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`). Returns `None` for
/// both input slots so the autograd traversal terminates without
/// extending the gradient graph through the op. Comparison-on-loss-
/// path is a user error; the missing gradient downstream surfaces it
/// rather than panicking.
pub struct NoGradientBinaryRule;
impl GradientRule for NoGradientBinaryRule {
    fn backward(
        &self,
        _graph: &SharedGraph,
        _op: &Op,
        _inputs: &[NodeId],
        _output: NodeId,
        _upstream: NodeId,
    ) -> GradientList {
        vec![None, None]
    }
}

/// Backward rule for [`Op::Where`] (ternary select).
///
/// Forward: `out[i] = if cond[i] != 0 { a[i] } else { b[i] }`.
/// Inputs: `(cond, a, b)` where `cond` is `U8`, `a` and `b` share
/// dtype `T`.
///
/// Gradients:
/// - `cond`: `None` (`U8` mask is non-differentiable).
/// - `a`: `upstream * cast(cond, T)` — gradient flows only at the
///   slots where `a` was picked.
/// - `b`: `upstream * cast(1 - cond, T)` — gradient flows only at
///   the slots where `b` was picked.
///
/// The `1 - cond` mask is built via `Op::AddScalar(-1.0)` followed by
/// `Op::Neg`: `m_b = -(m_a - 1) = 1 - m_a`. This avoids needing a
/// `Const(1)` factory inside the rule (which would require synthesizing
/// a slot-populated leaf via the device handle).
pub struct WhereRule;
impl GradientRule for WhereRule {
    fn backward(
        &self,
        graph: &SharedGraph,
        _op: &Op,
        inputs: &[NodeId],
        _output: NodeId,
        upstream: NodeId,
    ) -> GradientList {
        let cond = inputs[0];
        let a = inputs[1];
        let b = inputs[2];
        let shape = node_shape(graph, a);
        let dtype = node_dtype(graph, a);
        // m_a = cast(cond, dtype) — the "pick a" mask in float space.
        // Op::Cast is keyed on the target dtype; the source dtype
        // (U8 here) flows from the input node's dtype.
        let m_a = push_node(graph, Op::Cast(dtype), vec![cond], shape.clone(), dtype);
        // m_b = -(m_a + (-1)) = 1 - m_a — the complementary mask.
        // AddScalar(-1.0) produces (m_a - 1); Neg flips the sign.
        let m_a_minus_one =
            push_node(graph, Op::AddScalar(-1.0), vec![m_a], shape.clone(), dtype);
        let m_b = push_node(graph, Op::Neg, vec![m_a_minus_one], shape.clone(), dtype);
        // grad_a = upstream * m_a; grad_b = upstream * m_b.
        let grad_a = push_node(graph, Op::Mul, vec![upstream, m_a], shape.clone(), dtype);
        let grad_b = push_node(graph, Op::Mul, vec![upstream, m_b], shape, dtype);
        vec![None, Some(grad_a), Some(grad_b)]
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
    use fuel_ir::Shape;
    use std::sync::Arc;

    /// Phase 7.5 G2: tests need a real device for slot-populating
    /// constructors. Singleton CpuBackendDevice via OnceLock.
    fn cpu_dev() -> &'static Arc<dyn fuel_backend_contract::DynBackendDevice> {
        static D: std::sync::OnceLock<Arc<dyn fuel_backend_contract::DynBackendDevice>>
            = std::sync::OnceLock::new();
        D.get_or_init(|| Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
    }

    /// Confirm that the dispatch hook actually fires for migrated ops:
    /// `dispatch_gradient(Op::Add, ...)` returns Some, indicating the
    /// trait-based path handled the gradient. (The legacy match arm
    /// for Add still exists as a safety net but should never run.)
    #[test]
    fn dispatch_fires_for_migrated_ops() {
        let a = Tensor::from_f32(vec![1.0_f32, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
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

    /// `WhereRule` returns 3 entries: `None` for cond (non-differentiable
    /// U8), `Some(grad_a)` reaching `a` only at picked positions,
    /// `Some(grad_b)` reaching `b` at non-picked positions. The
    /// per-input grads are built from `Op::Cast`, `Op::AddScalar`,
    /// `Op::Neg`, `Op::Mul` — all primitives.
    #[test]
    fn dispatch_where_returns_none_for_cond_some_for_a_and_b() {
        use crate::SharedGraph;
        use std::sync::{Arc, RwLock};
        use fuel_ir::{DType, Shape};
        // Build a real graph so push_node calls inside the rule have
        // somewhere to land. Three placeholder Const nodes for the
        // three inputs.
        let g: SharedGraph = Arc::new(RwLock::new(crate::Graph::new()));
        let (cond, a, b, output, upstream) = {
            let mut gw = g.write().unwrap();
            let cond = gw.push(crate::Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::U8,
            });
            let a = gw.push(crate::Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let b = gw.push(crate::Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let output = gw.push(crate::Node {
                op: Op::Where, inputs: vec![cond, a, b],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let upstream = gw.push(crate::Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            (cond, a, b, output, upstream)
        };
        let result = dispatch_gradient(&g, &Op::Where, &[cond, a, b], output, upstream)
            .expect("Op::Where must have a registered GradientRule");
        assert_eq!(result.len(), 3, "Where has 3 inputs");
        assert!(result[0].is_none(), "cond gradient must be None (U8 non-differentiable)");
        assert!(result[1].is_some(), "a gradient must be Some (differentiable through pick)");
        assert!(result[2].is_some(), "b gradient must be Some (differentiable through fallback)");
        // Sanity: the two grads should reference different nodes.
        assert_ne!(result[1].unwrap(), result[2].unwrap(),
            "grad_a and grad_b should be distinct backward nodes");
    }

    /// Comparison family terminates the autograd traversal cleanly:
    /// the dispatcher returns Some(vec![None, None]) so the caller
    /// extends nothing into the gradient graph for either input.
    /// Without this rule, the legacy fallthrough would emit a U8 gradient
    /// (impossible) or panic.
    #[test]
    fn dispatch_returns_none_gradients_for_eq() {
        use crate::SharedGraph;
        use std::sync::{Arc, RwLock};
        let g: SharedGraph = Arc::new(RwLock::new(crate::Graph::new()));
        let dummy = NodeId(0);
        let result = dispatch_gradient(&g, &Op::Equal, &[dummy, dummy], dummy, dummy)
            .expect("Op::Equal must have a registered GradientRule");
        assert_eq!(result.len(), 2, "Op::Equal has 2 inputs");
        assert!(result[0].is_none(), "lhs gradient must be None (non-differentiable)");
        assert!(result[1].is_none(), "rhs gradient must be None (non-differentiable)");
    }
}
